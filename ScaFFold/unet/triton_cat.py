# Copyright (c) 2014-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LBANN/ScaFFold.
#
# SPDX-License-Identifier: (Apache-2.0)

"""Channels-last-native Triton channel concatenation (NDHWC in, NDHWC out).

Why this exists
===============
Every ``Up`` block of the UNet joins the decoder's upsampled activation to the
encoder's skip activation with ``torch.cat([skip, up], dim=1)`` and feeds the
result to a convolution.  Profiled at scale 7 (channels-last, bf16 autocast,
DCTensor) that concatenation and the copies autocast wraps around it cost
**~10.4 ms of a 92 ms step, 11.6%** -- the largest remaining item in ScaFFold's
own code once the GroupNorm kernel landed.  It is not one problem but two, and
neither is inherent:

**1. The dtypes are wrong on the wire.**  Under ``torch.autocast`` the two
inputs do not have the same dtype.  GroupNorm carries autocast's ``fp32`` cast
policy, so the skip tensor -- which is a ``DoubleConv`` output, i.e. a
GroupNorm output -- arrives as **fp32**, while the upsampled tensor comes
straight out of a ``ConvTranspose3d`` and is **bf16**.  ``aten::cat`` carries
the ``promote`` policy, so autocast *widens the bf16 input to fp32*, cats at
fp32, and then the following convolution -- ``lower_precision_fp`` policy --
casts the whole double-width result straight back down to bf16.  At the largest
decoder block ([1,64,128^3] skip + [1,64,128^3] up) that is

    up->fp32   read 268 MB  write 537 MB
    cat        read 1074 MB write 1074 MB
    conv cast  read 1074 MB write 537 MB   = 4.56 GB of traffic

to deliver 537 MB of bf16 to the convolution.  Emitting bf16 *directly from the
concatenation* is **bitwise identical** to what the convolution receives today
-- the fp32 intermediate holds exact copies of an fp32 tensor and of a widened
bf16 tensor, so rounding it to bf16 recovers exactly ``(bf16(skip), up)`` --
and costs one pass: read 537 + 268, write 537 = **1.34 GB, 3.4x less**.

**2. The kernel iterates the wrong order.**  This is the same defect the
GroupNorm kernel exists to fix.  A ``channels_last_3d`` tensor ``(N, C, D, H,
W)`` is *physically* a dense ``(N*D*H*W, C)`` array, so a channel
concatenation is, in memory, ``out[m, :Ca] = a[m, :]`` and ``out[m, Ca:] =
b[m, :]`` -- a pure streaming join of two dense arrays into one, 2 reads and 1
write, perfectly coalescable.  ATen reaches it through the *logical* NCDHW
order and lands on TensorIterator's generic offset-calculator path, where the
output tile is never contiguous.

The kernels below own a ``(BLOCK_M, C)`` tile of the *physical* array: they
issue one fully contiguous store per tile and gather the two halves of it from
the two sources, whose valid lanes are themselves contiguous runs.  The dtype
conversion rides along in registers, so the retyping in point 1 is free.

Public API
==========
``cat_channels(a, b, out_dtype=None)``
    ``torch.cat([a, b], dim=1)`` with an optional output dtype override.
    ``out_dtype=None`` reproduces ``torch.cat``'s own promotion exactly.
    Anything :func:`is_supported` declines is served by ``torch.cat`` itself.

``is_supported(a, b, out_dtype=None)``
    Cheap, side-effect-free predicate: ``True`` exactly when the Triton kernel
    will run.

``skip_concat(skip, upsampled)``
    The UNet decoder's call: chooses the output dtype (see
    :func:`consumer_dtype`), unwraps DistConv's ``DCTensor`` to its local shard
    and rewraps the result.  This is the only function ``unet_parts`` calls.

Contract
========
For every input :func:`is_supported` accepts, ``cat_channels(a, b, dt)`` equals
``torch.cat([a, b], dim=1).to(dt)`` **bitwise**, with:

* **memory format** -- the output is ``channels_last_3d``-contiguous, which is
  also what ``torch.cat`` returns for channels-last inputs.
* **dtype** -- exactly ``out_dtype``, or ``torch.promote_types(a.dtype,
  b.dtype)`` when that is ``None``.  Narrowing is a *single* rounding of each
  source value: every supported dtype widens exactly into fp32, which is the
  kernel's compute type, so rounding fp32->bf16 once in the kernel is the same
  bits as ATen's widen-then-narrow.
* **autograd** -- first order, ``d_a`` and ``d_b`` in the *inputs'* dtypes,
  which is what autograd requires and what the current chain produces after
  autocast's cast nodes run their own backward.  Second-order raises, exactly
  as :mod:`ScaFFold.unet.triton_group_norm` does and for the same reason: the
  backward is itself a custom op with no autograd formula.
* **determinism** -- trivially bitwise reproducible.  There is no reduction,
  no atomic and no autotuning; every output element is a copy of exactly one
  input element and the tile shape is a pure function of the channel count.
* **device** -- the kernels run on the inputs' device whatever device is
  current; see ``_device_guard``, which exists because a Triton launch follows
  the *current* device and not its arguments'.
* **rejections** -- :func:`is_supported` declines anything the kernel cannot
  serve physically (non-channels-last, non-5-D, unsupported dtype, CPU,
  mismatched spatial extent, empty), and :func:`cat_channels` then routes it to
  ``torch.cat``, so the function is total.

Measured cost
=============
See ``review/skip-path/RESULTS.md`` for the interleaved A/B, the op-level
before/after and the gradient check.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import sys
from typing import Optional, Tuple

import torch

__all__ = [
    "cat_channels",
    "is_supported",
    "skip_concat",
    "consumer_dtype",
    "CatKernelError",
]


class CatKernelError(RuntimeError):
    """A failure of the Triton kernels themselves, with the original as ``__cause__``.

    Mirrors :class:`ScaFFold.unet.triton_group_norm.TritonKernelError`: it is
    raised only from a *closed* region that allocates and launches and runs no
    autograd-observable op, so a caller may retry the call on ``torch.cat``
    without worrying that half a graph was already recorded.
    ``torch.OutOfMemoryError`` is passed through untagged -- it is a resource
    condition, not a defect, and the fallback would allocate the same bytes.
    """


#: Dtypes the kernels read and write directly.  All three widen exactly into
#: fp32, which is what makes the single-rounding claim in the docstring hold.
SUPPORTED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

_CL_FORMAT = torch.channels_last_3d
_INT32_MAX = 2**31 - 1

#: Elements per tile and the cap on voxels per program.  Both are pure
#: functions of the channel count, so a run cannot change the tiling underneath
#: a comparison.  Chosen by a sweep of BLOCK_M in {1..64} x num_warps in
#: {1,2,4,8} at the four scale-7 decoder shapes
#: (``review/skip-path/logs/cat_bench_tune.log``,
#: ``split_bench_tune.log``): with ``BLOCK_M = clamp(4096 // next_pow2(C), 1,
#: 32)`` and four warps the kernels are within 0.8% of the per-shape optimum at
#: the two shapes that dominate and within 4% at the two launch-bound ones,
#: which is not worth a frozen table.
_TILE_ELEMS = 4096
_MAX_BLOCK_M = 32
_NUM_WARPS = 4


# --------------------------------------------------------------------------- #
# Triton kernels
# --------------------------------------------------------------------------- #
triton = None
tl = None
_cat_kernel = None
_split_kernel = None

_TRITON_AVAILABLE: Optional[bool] = None


def triton_available() -> bool:
    """Whether ``triton`` can be imported, cached, without importing it."""
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            _TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
        except (ImportError, ValueError):
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


def _build_kernels():
    """Import Triton and install the JIT kernels into this module's globals.

    Defined inside a function purely so ``import triton`` is deferred to the
    first GPU call, and written into ``globals()`` because Triton resolves
    names through ``fn.__globals__``.
    """
    global triton, tl
    import triton as _triton
    import triton.language as _tl

    triton = _triton
    tl = _tl

    @_triton.jit
    def _cat_kernel(
        A,
        B,
        OUT,
        M,
        CA: tl.constexpr,
        CB: tl.constexpr,
        C: tl.constexpr,
        CAP: tl.constexpr,
        CBP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        INT64: tl.constexpr,
    ):
        """One program per ``BLOCK_M`` voxels: join two dense rows into one.

        Each source keeps its **own** tile width (``next_pow2`` of its channel
        count) and is copied by its own load/store pair, rather than both being
        gathered into one ``(BLOCK_M, C)`` tile and stored once.  The single
        fully contiguous store is the more obvious design and was measured
        first; it is **20% slower** end to end over the four decoder shapes
        (0.718 vs 0.571 ms, ``review/skip-path/logs/cat_bench_tune.log``).  The
        reason is that the one-store form has to mask *both* loads down to
        complementary halves of a double-width lane space, which halves the
        useful work per instruction and defeats vectorization, and it buys only
        a contiguous store -- whereas a store of ``CA`` contiguous elements at a
        stride of ``C`` already covers whole cache lines whenever
        ``CA * itemsize`` is a multiple of the line, which it is for every
        channel count this network uses.  The strided store is not the problem;
        the mask was.
        """
        pid = tl.program_id(0)
        m0 = pid * BLOCK_M
        if INT64:
            wide = m0.to(tl.int64)
            base_a = wide * CA
            base_b = wide * CB
            base_o = wide * C
        else:
            base_a = m0 * CA
            base_b = m0 * CB
            base_o = m0 * C

        rows = tl.arange(0, BLOCK_M)
        rmask = rows < M - m0
        ca = tl.arange(0, CAP)
        cb = tl.arange(0, CBP)
        # Clamp the padding lanes' column index: their loads and stores are
        # masked and never touch memory, but keeping the arithmetic inside the
        # allocation avoids forming a pointer the compiler may treat as poison.
        cam = ca < CA
        cbm = cb < CB
        ca = tl.where(cam, ca, 0)
        cb = tl.where(cbm, cb, 0)
        am = rmask[:, None] & cam[None, :]
        bm = rmask[:, None] & cbm[None, :]

        av = tl.load(A + base_a + rows[:, None] * CA + ca[None, :], mask=am, other=0.0)
        tl.store(
            OUT + base_o + rows[:, None] * C + ca[None, :],
            av.to(OUT.dtype.element_ty),
            mask=am,
        )
        bv = tl.load(B + base_b + rows[:, None] * CB + cb[None, :], mask=bm, other=0.0)
        tl.store(
            OUT + base_o + rows[:, None] * C + CA + cb[None, :],
            bv.to(OUT.dtype.element_ty),
            mask=bm,
        )

    @_triton.jit
    def _split_kernel(
        G,
        DA,
        DB,
        SN,
        SD,
        SH,
        D,
        H,
        W,
        CA: tl.constexpr,
        CB: tl.constexpr,
        C: tl.constexpr,
        CAP: tl.constexpr,
        CBP: tl.constexpr,
        BLOCK_W: tl.constexpr,
        WANT_A: tl.constexpr,
        WANT_B: tl.constexpr,
        INT64: tl.constexpr,
    ):
        """The transpose of :func:`_cat_kernel`: two strided loads, two stores.

        This is the pass ATen is genuinely bad at.  ``cat``'s backward hands the
        consumer a *narrowed view*, and every consumer then forces it
        contiguous, so the work happens as a generic strided ``copy_`` that
        reaches only **51-63%** of this device's streaming roofline at the four
        decoder shapes; this kernel reaches **90-103%**
        (``review/skip-path/logs/split_bench_tune.log``).

        **The incoming gradient is addressed by its strides, not assumed dense**,
        and that is not a nicety.  Under DistConv -- which production uses even
        at ``dc_num_shards=1`` -- the convolution that consumes the
        concatenation is reached through a halo exchange that materialises a
        *padded* tensor, so the cotangent that comes back here is a narrowed
        view of a ``(D+2, H+2, W+2)`` one: channels-last within each voxel and
        contiguous along W, but with a gap at every H and D boundary.  An
        earlier version simply called ``.contiguous(memory_format=channels_last_3d)``
        on it, which costs a **whole extra full-resolution pass** that ATen's
        view-based backward never pays -- 0.46 ms per step at the largest
        decoder shape, enough on its own to turn this kernel from a win into a
        loss against plain ``torch.cat`` at the right dtype (measured: the
        isolated four-block sum went from 53.40 ms with the relayout to 52.10
        without).

        The requirement is therefore only that channels are innermost
        (``stride(1) == 1``) and that a voxel's neighbours along W are one
        channel-run apart (``stride(4) == C``); everything above W is addressed
        through ``SN``/``SD``/``SH``.  That admits a dense channels-last tensor
        and any narrowing of one on D, H or W, which is every case this op
        sees.  The driver falls back to a relayout for anything else.
        """
        pid = tl.program_id(0)
        h = tl.program_id(1)
        nd = tl.program_id(2)
        n = nd // D
        d = nd % D

        w0 = pid * BLOCK_W
        ws = w0 + tl.arange(0, BLOCK_W)
        wmask = ws < W

        if INT64:
            gbase = n.to(tl.int64) * SN + d.to(tl.int64) * SD + h.to(tl.int64) * SH
            row0 = ((n.to(tl.int64) * D + d) * H + h) * W + w0
        else:
            gbase = n * SN + d * SD + h * SH
            row0 = ((n * D + d) * H + h) * W + w0
        gbase = gbase + w0 * C

        wl = tl.arange(0, BLOCK_W)
        ca = tl.arange(0, CAP)
        cb = tl.arange(0, CBP)
        cam = ca < CA
        cbm = cb < CB
        ca = tl.where(cam, ca, 0)
        cb = tl.where(cbm, cb, 0)
        am = wmask[:, None] & cam[None, :]
        bm = wmask[:, None] & cbm[None, :]

        if WANT_A:
            ga = tl.load(G + gbase + wl[:, None] * C + ca[None, :], mask=am, other=0.0)
            tl.store(
                DA + row0 * CA + wl[:, None] * CA + ca[None, :],
                ga.to(DA.dtype.element_ty),
                mask=am,
            )
        if WANT_B:
            gb = tl.load(
                G + gbase + wl[:, None] * C + CA + cb[None, :], mask=bm, other=0.0
            )
            tl.store(
                DB + row0 * CB + wl[:, None] * CB + cb[None, :],
                gb.to(DB.dtype.element_ty),
                mask=bm,
            )

    globals().update(_cat_kernel=_cat_kernel, _split_kernel=_split_kernel)


def _ensure_kernels():
    if _cat_kernel is None:
        _build_kernels()


# --------------------------------------------------------------------------- #
# python drivers
# --------------------------------------------------------------------------- #
_NO_GUARD = contextlib.nullcontext()


def _device_guard(device: torch.device):
    """Make ``device`` current for the kernel launches inside the ``with``.

    A Triton launch goes to whatever device is *current*, not to the device its
    arguments live on; without this a tensor on ``cuda:1`` while ``cuda:0`` is
    current makes the kernel dereference another device's pointers and the
    process dies with ``Memory access fault by GPU node-N``.  ATen ops carry a
    ``DeviceGuard`` and handle the same call, so this is required for the
    drop-in contract.  The ``current_device()`` test keeps the common
    (already-current) path free; see the same helper in
    :mod:`ScaFFold.unet.triton_group_norm` for the measurement.
    """
    if device.index == torch.cuda.current_device():
        return _NO_GUARD
    return torch.cuda.device(device)


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@functools.lru_cache(maxsize=64)
def _block_m(channels_pow2: int) -> int:
    """Voxels per program.  A pure function of the padded channel count.

    Bitwise determinism does not actually depend on this -- a copy has no
    reduction order to perturb -- but keeping it a pure function of the shape
    means a run cannot change tiling underneath a comparison, which is the
    property the GroupNorm kernel's frozen table exists to give and is worth
    having for free here too.
    """
    return max(1, min(_MAX_BLOCK_M, _TILE_ELEMS // channels_pow2))


@functools.lru_cache(maxsize=64)
def _block_w(channels_pow2: int) -> int:
    """Voxels per program for the backward, which tiles along W within a line.

    Same rule and same sweep as :func:`_block_m`; the backward cannot tile over
    a flat voxel index because its source is only guaranteed contiguous *within*
    a ``(n, d, h)`` line -- see ``_split_kernel``.
    """
    return max(1, min(_MAX_BLOCK_M, _TILE_ELEMS // channels_pow2))


def _rows_of(t: torch.Tensor) -> int:
    """``N * D * H * W`` -- the number of voxels in the physical (rows, C) view."""
    rows = t.shape[0]
    for d in t.shape[2:]:
        rows *= d
    return rows


def _tag_kernel_failures(fn):
    """Re-raise anything ``fn`` raises as :class:`CatKernelError`.

    Applied to the two functions that do nothing but allocate and launch, so
    the tagged region cannot swallow framework control flow (there is no pack
    hook, no recompute stop and no functorch layer inside it).
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except torch.OutOfMemoryError:
            raise
        except CatKernelError:
            raise
        except Exception as e:  # noqa: BLE001 -- re-raised, see docstring
            raise CatKernelError(
                f"{fn.__name__} failed ({type(e).__name__}: {e})"
            ) from e

    return wrapper


@_tag_kernel_failures
def _forward(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype):
    _ensure_kernels()
    ca, cb = a.shape[1], b.shape[1]
    c = ca + cb
    rows = _rows_of(a)
    cp = _next_pow2(c)
    block_m = _block_m(cp)

    with _device_guard(a.device):
        out = torch.empty(
            (a.shape[0], c, *a.shape[2:]),
            dtype=out_dtype,
            device=a.device,
            memory_format=_CL_FORMAT,
        )
        _cat_kernel[(_cdiv(rows, block_m),)](
            a,
            b,
            out,
            rows,
            CA=ca,
            CB=cb,
            C=c,
            CAP=_next_pow2(ca),
            CBP=_next_pow2(cb),
            BLOCK_M=block_m,
            INT64=rows * c > _INT32_MAX,
            num_warps=_NUM_WARPS,
        )
    return out


@_tag_kernel_failures
def _backward(grad, ca, cb, a_dtype, b_dtype, want_a, want_b):
    _ensure_kernels()
    c = ca + cb
    n, _, d, h, w = grad.shape
    sn, _, sd, sh, _ = grad.stride()
    block_w = _block_w(_next_pow2(c))

    with _device_guard(grad.device):
        da = torch.empty(
            (n, ca, d, h, w) if want_a else (0,),
            dtype=a_dtype,
            device=grad.device,
            **({"memory_format": _CL_FORMAT} if want_a else {}),
        )
        db = torch.empty(
            (n, cb, d, h, w) if want_b else (0,),
            dtype=b_dtype,
            device=grad.device,
            **({"memory_format": _CL_FORMAT} if want_b else {}),
        )
        if want_a or want_b:
            _split_kernel[(_cdiv(w, block_w), h, n * d)](
                grad,
                da,
                db,
                sn,
                sd,
                sh,
                d,
                h,
                w,
                CA=ca,
                CB=cb,
                C=c,
                CAP=_next_pow2(ca),
                CBP=_next_pow2(cb),
                BLOCK_W=block_w,
                WANT_A=want_a,
                WANT_B=want_b,
                # The widest index the kernel forms is the *source* base, which
                # spans the (possibly padded) parent tensor, so it is bounded by
                # the largest stride and not by this tensor's own element count.
                INT64=max(sn * n, n * d * h * w * c) > _INT32_MAX,
                num_warps=_NUM_WARPS,
            )
    return da, db


# --------------------------------------------------------------------------- #
# torch.library registration
# --------------------------------------------------------------------------- #
def _line_addressable(grad: torch.Tensor) -> bool:
    """Whether ``_split_kernel`` can read ``grad`` in place.

    It needs channels innermost and voxels one channel-run apart along W; the
    D and H axes are addressed through their own strides, so any narrowing of a
    channels-last tensor qualifies -- which is what DistConv's halo-padded
    cotangent is.  A dense channels-last tensor is the special case where the
    strides happen to be exact multiples.
    """
    if grad.dim() != 5:
        return False
    stride = grad.stride()
    return stride[1] == 1 and stride[4] == grad.shape[1]


def _validate(a, b, out_dtype):
    if a.dim() != 5 or b.dim() != 5:
        raise ValueError(
            f"expected two 5-D NCDHW tensors, got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    if a.shape[0] != b.shape[0] or tuple(a.shape[2:]) != tuple(b.shape[2:]):
        raise ValueError(
            f"shapes must agree except on dim 1, got {tuple(a.shape)} and "
            f"{tuple(b.shape)}"
        )
    if a.shape[1] < 1 or b.shape[1] < 1 or a.numel() == 0 or b.numel() == 0:
        raise ValueError("both inputs must have at least one channel and be non-empty")
    for name, t in (("a", a), ("b", b)):
        if t.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype {t.dtype} for {name}")
        if not t.is_contiguous(memory_format=_CL_FORMAT):
            raise ValueError(
                f"{name} must be channels_last_3d-contiguous; use cat_channels() "
                "which falls back to torch.cat for other layouts"
            )
    if out_dtype is not None and out_dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported out_dtype {out_dtype}")


@torch.library.custom_op(
    "scaffold_cat::cat_channels", mutates_args=(), device_types="cuda"
)
def _cat_op(
    a: torch.Tensor, b: torch.Tensor, out_dtype: Optional[torch.dtype]
) -> torch.Tensor:
    """``torch.cat([a, b], dim=1)`` for channels-last-3d inputs."""
    _validate(a, b, out_dtype)
    return _forward(a, b, out_dtype or torch.promote_types(a.dtype, b.dtype))


@_cat_op.register_fake
def _(a, b, out_dtype):
    return torch.empty(
        (a.shape[0], a.shape[1] + b.shape[1], *a.shape[2:]),
        dtype=out_dtype or torch.promote_types(a.dtype, b.dtype),
        device=a.device,
        memory_format=_CL_FORMAT,
    )


@torch.library.custom_op(
    "scaffold_cat::split_channels", mutates_args=(), device_types="cuda"
)
def _split_op(
    grad: torch.Tensor,
    ca: int,
    cb: int,
    a_dtype: Optional[torch.dtype],
    b_dtype: Optional[torch.dtype],
    want_a: bool,
    want_b: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``grad`` back into its ``ca``- and ``cb``-channel halves.

    Returns a zero-element placeholder for a half whose ``want_*`` is False.

    ``grad`` is relaid out only when it is not *line addressable* -- see
    :func:`_line_addressable`, and see ``_split_kernel`` for why paying that
    copy unconditionally is what made an earlier version of this op lose to
    plain ``torch.cat``.
    """
    if not _line_addressable(grad):
        grad = grad.contiguous(memory_format=_CL_FORMAT)
    return _backward(
        grad, ca, cb, a_dtype or grad.dtype, b_dtype or grad.dtype, want_a, want_b
    )


@_split_op.register_fake
def _(grad, ca, cb, a_dtype, b_dtype, want_a, want_b):
    spatial = tuple(grad.shape[2:])
    da = (
        torch.empty(
            (grad.shape[0], ca, *spatial),
            dtype=a_dtype or grad.dtype,
            device=grad.device,
            memory_format=_CL_FORMAT,
        )
        if want_a
        else grad.new_empty(0, dtype=a_dtype or grad.dtype)
    )
    db = (
        torch.empty(
            (grad.shape[0], cb, *spatial),
            dtype=b_dtype or grad.dtype,
            device=grad.device,
            memory_format=_CL_FORMAT,
        )
        if want_b
        else grad.new_empty(0, dtype=b_dtype or grad.dtype)
    )
    return da, db


def _setup_context(ctx, inputs, output):
    a, b, _out_dtype = inputs
    # Only metadata is saved: the backward of a concatenation does not read
    # either input, so saving them would pin two full-resolution activations
    # for the whole backward pass for nothing.
    ctx.ca = a.shape[1]
    ctx.cb = b.shape[1]
    ctx.a_dtype = a.dtype
    ctx.b_dtype = b.dtype
    ctx.needs = (ctx.needs_input_grad[0], ctx.needs_input_grad[1])


def _autograd_backward(ctx, grad_out):
    need_a, need_b = ctx.needs
    if not (need_a or need_b):
        return None, None, None
    da, db = torch.ops.scaffold_cat.split_channels(
        grad_out, ctx.ca, ctx.cb, ctx.a_dtype, ctx.b_dtype, need_a, need_b
    )
    return (da if need_a else None), (db if need_b else None), None


torch.library.register_autograd(
    "scaffold_cat::cat_channels", _autograd_backward, setup_context=_setup_context
)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def is_supported(a, b, out_dtype: Optional[torch.dtype] = None) -> bool:
    """Whether the Triton kernel can serve ``cat_channels(a, b, out_dtype)``.

    Cheap (attribute reads and two stride checks) and side-effect free: it does
    not import Triton, allocate or launch.  ``False`` means "use
    ``torch.cat``".

    Note that this accepts any ``torch.Tensor`` *instance*, so it must be
    called on the tensor the kernel will actually touch.  ``skip_concat``
    unwraps DistConv's ``DCTensor`` before asking, for exactly the reason
    :class:`ScaFFold.unet.group_norm.FastGroupNorm` documents: a wrapper
    subclass's mirrored metadata is not the shard's.
    """
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return False
    if a.device.type != "cuda" or a.device != b.device:
        return False
    if not triton_available():
        return False
    if a.dim() != 5 or b.dim() != 5:
        return False
    if a.dtype not in SUPPORTED_DTYPES or b.dtype not in SUPPORTED_DTYPES:
        return False
    if out_dtype is not None and out_dtype not in SUPPORTED_DTYPES:
        return False
    if a.shape[0] != b.shape[0] or tuple(a.shape[2:]) != tuple(b.shape[2:]):
        return False
    if a.shape[1] < 1 or b.shape[1] < 1:
        return False
    if a.numel() == 0 or b.numel() == 0:
        return False
    if not a.is_contiguous(memory_format=_CL_FORMAT):
        return False
    if not b.is_contiguous(memory_format=_CL_FORMAT):
        return False
    return True


def cat_channels(a, b, out_dtype: Optional[torch.dtype] = None):
    """``torch.cat([a, b], dim=1)``, optionally emitting ``out_dtype`` directly.

    Total: anything :func:`is_supported` declines is served by ``torch.cat``
    (followed by a cast when ``out_dtype`` asks for one), so this is a drop-in
    on CPU, on non-channels-last input and without Triton.

    The fallback casts the *inputs* rather than the concatenated result when
    ``out_dtype`` is narrower.  That is bitwise the same answer -- every
    supported dtype widens exactly into the promoted type, so narrowing before
    or after the copy rounds the same values once -- and it does not
    materialize a double-width intermediate.
    """
    if not is_supported(a, b, out_dtype):
        if out_dtype is None:
            return torch.cat([a, b], dim=1)
        return torch.cat([a.to(out_dtype), b.to(out_dtype)], dim=1)
    return torch.ops.scaffold_cat.cat_channels(a, b, out_dtype)


def consumer_dtype(*tensors) -> torch.dtype:
    """The dtype the convolution that consumes the concatenation will see.

    Under an enabled autocast region the answer is autocast's ``dtype``,
    because ``aten::convolution`` carries the ``lower_precision_fp`` cast
    policy and casts whatever it is handed.  Producing that dtype from the
    concatenation is therefore *bitwise identical* to producing ATen's promoted
    dtype and letting the convolution narrow it -- the promoted tensor holds
    exact widenings of both sources -- while writing (and reading back) half
    the bytes, and it removes autocast's own widening of the narrower input on
    the way in.

    Outside autocast the answer is ``torch.cat``'s ordinary promotion, so the
    result is unchanged for eval, ``inference_mode`` and pure-fp32 runs.
    """
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    device_type = tensors[0].device.type
    try:
        if torch.is_autocast_enabled(device_type):
            autocast_dtype = torch.get_autocast_dtype(device_type)
        else:
            return dtype
    except (RuntimeError, TypeError):  # a device type autocast does not know
        return dtype
    # Only ever narrow: if autocast's dtype is not one the kernel can hold, or
    # is wider than the inputs, keep the promotion ATen would have done.
    if autocast_dtype not in SUPPORTED_DTYPES:
        return dtype
    if torch.promote_types(autocast_dtype, dtype) is autocast_dtype:
        return dtype
    return autocast_dtype


def _dctensor_ops(input):
    """The ``distconv.distconv`` module when ``input`` is a DCTensor, else None.

    Resolved through ``sys.modules`` rather than an import, so this module
    stays importable (and the CPU suite runnable) without DistConv installed.
    """
    distconv = sys.modules.get("distconv.distconv")
    if distconv is not None and isinstance(input, distconv.DCTensor):
        return distconv
    return None


def skip_concat(skip, upsampled):
    """Join a decoder skip activation to an upsampled one, as ``Up.forward`` needs.

    Equivalent to ``torch.cat([skip, upsampled], dim=1)`` as the following
    convolution observes it -- see :func:`consumer_dtype` for why the dtype may
    legitimately differ from ``torch.cat``'s own.

    DistConv's ``DCTensor`` is unwrapped to its local shard in front of the
    kernel and rewrapped after, rather than being left to
    ``DCTensor.__torch_dispatch__``.  Both work -- these are real dispatcher
    ops -- but the explicit unwrap is what ``FastGroupNorm`` already does, and
    it keeps the subclass policy in one place: ``is_supported`` accepts any
    ``torch.Tensor`` instance, so relying on dispatch would silently extend the
    fast path to every unknown wrapper subclass.  The unwrap goes through
    DistConv's ``_ToTensor``/``from_shard`` autograd pair and not a bare
    ``._tensor`` read, which would sever the graph back to the producing
    convolution.  DistConv has no concatenation-specific handling, so the
    semantics are identical at every shard count.
    """
    distconv = _dctensor_ops(skip)
    if distconv is None or _dctensor_ops(upsampled) is None:
        return cat_channels(skip, upsampled, consumer_dtype(skip, upsampled))
    if skip._parallel_strategy != upsampled._parallel_strategy:
        raise ValueError(
            "skip and upsampled tensors have different parallel strategies"
        )
    local_skip = distconv._ToTensor.apply(skip)
    local_up = distconv._ToTensor.apply(upsampled)
    out = cat_channels(local_skip, local_up, consumer_dtype(local_skip, local_up))
    return distconv.DCTensor.from_shard(out, skip._parallel_strategy)
