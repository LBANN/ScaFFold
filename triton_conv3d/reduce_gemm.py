# SPDX-License-Identifier: (Apache-2.0)
"""Backward-weight: a split-K reduction GEMM over the whole output volume.

    dW[co, ci, kd, kh, kw] = sum_{n,d,h,w} dY[n,d,h,w,co] * X[n, d*s+kd*dil-p, ..., ci]

As a GEMM that is ``M = Cout``, ``N = taps * Cin``, ``K = N*OD*OH*OW`` -- a tiny
output reduced over an enormous K.  That is the transpose of the situation the
forward and backward-data face, and it is why this is the one direction that
needs a kernel of its own.

Why the forward kernel cannot serve this
========================================

It *can*, algebraically.  Swapping the batch and channel axes of both
activations turns backward-weight into a forward convolution::

    dW^T (Cin, Cout, KD, KH, KW) = conv3d(X^T (Cin, N, ID, IH, IW),
                                          weight = dY^T (Cout, N, OD, OH, OW))

with ``N`` as the channel count and the *output volume* as the kernel extent.
``test_bwd_weight.py::test_the_forward_kernel_can_express_backward_weight`` runs
exactly that and checks it bitwise.

It is also unusable.  ScaFFold runs ``N = 1``, so the reused kernel's ``Cin`` is
1: ``BLOCK_K`` would have to be 16 to reach the matrix core and 15 of every 16
lanes would be padding.  Worse, the forward's reduction loop runs
``KD*KH*KW * ceil(Cin/BLOCK_K)`` iterations, and ``KD*KH*KW`` here is the whole
output volume -- millions of trip counts of a six-compare boundary predicate,
with no split-K anywhere.  The reuse is correct and hopelessly slow.

Shape of the kernel
===================

The contraction is a "TN" GEMM: both operands have the reduction axis (the
output voxel) *slowest* and the GEMM's M / N axes contiguous, because NDHWC puts
the channel last.  So the A tile is loaded ``(BLOCK_K, BLOCK_M)`` and
transposed, which costs one ``tl.trans`` and keeps both global loads
contiguous -- the alternative, addressing A as ``(BLOCK_M, BLOCK_K)``, strides by
``Cout`` down the fast axis and devectorizes every load.

Split-K is mandatory, not optional.  ``Cout`` alone is one or two ``BLOCK_M``
rows, so an unsplit grid is ``taps * ceil(Cin/BLOCK_N)`` programs -- a few dozen,
on a device with 228 CUs.  :func:`split_count` derives the split count from the
shape.

Several taps per tile.  The obvious tiling gives each program one tap, so that
the tap's spatial shift stays a scalar addend on the row offset (which is what
the forward does, and what makes NDHWC pay).  It is by far the slower of the
two, and the reason is arithmetic intensity: per reduction element a tile loads
``BLOCK_M + BLOCK_N`` values and does ``2*BLOCK_M*BLOCK_N`` flops, so its
intensity is ``BLOCK_M*BLOCK_N/(BLOCK_M+BLOCK_N)`` flops per byte, and with one
tap ``Cout`` caps ``BLOCK_M`` while ``Cin`` caps ``BLOCK_N``.  Every operand is
then re-read once per tap, at HBM speed rather than cache speed.

Widening ``BLOCK_N`` across ``TAP_BLOCK`` taps fixes both halves of that at once.
The upstream gradient is read ``taps/TAP_BLOCK`` times instead of ``taps``, and
the ``TAP_BLOCK`` shifted reads of the input land in the same instruction stream
on overlapping cache lines instead of in unrelated programs on different XCDs.
The cost is that the tap shift is no longer a scalar: it becomes a per-column
addend, hoisted out of the reduction loop, and -- only when the convolution is
padded -- a two-dimensional boundary predicate instead of a one-dimensional one.

Padding does not veto the wide tile, and padded is the case production takes:
``ScaFFold/unet/conv3d.py`` performs the halo exchange itself and only on axes
that are *genuinely split*, so every ``k > 1`` convolution reaches this kernel
with a non-zero padding on at least one axis.  The two-dimensional predicate is
real and it costs something; it costs far less than the arithmetic intensity the
wide tile buys, at every affected site, and
``triton_conv3d/bench/conv_bench.py`` regenerates that comparison.

Determinism
===========

This direction is where ScaFFold's reproducibility is decided.  MIOpen serves it
with ``kernel_batched_gemm_xdlops_bwd_weight``, a split-K GEMM using float
atomics, which is why a run on the MIOpen rung is not bitwise reproducible
unless ``more_determinism`` -- which disables far more than convolution -- is on.

Two paths, one kernel, one ``ATOMIC`` constexpr:

* deterministic (the default): each split writes its own slice of an fp32
  workspace ``[splits, Cout, taps*Cin]``, and :func:`_reduce_partials_kernel`
  sums the splits in index order.  No float atomics, a grid and a split count
  that are pure functions of the shape, and accumulation order fixed by the
  compiled code.  The claim is the one ``ScaFFold/unet/triton_group_norm.py``
  already makes: bitwise identical run to run and process to process, for the
  same input, dtype, shape, device and tuning config -- not bitwise against
  MIOpen, and not across configs.
* atomic: ``tl.atomic_add`` into a zeroed fp32 accumulator, i.e. what CK does.
  It exists solely to price determinism and is never selected on its own; a
  caller has to ask for ``deterministic=False``.

:func:`split_count` deliberately does not consult free memory, occupancy or a
runtime autotuner.  Anything that lets the split count vary between two runs of
the same shape breaks the claim above, and it would break it *intermittently*,
which is worse than breaking it outright.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import torch
import triton
import triton.language as tl

from .gather_gemm import (
    _LDS_BYTES,
    _MFMA_KDIM,
    ConvConfig,
    _pow2_at_most,
    _triple,
    tune_key,
)

__all__ = [
    "BwdWeightConfig",
    "conv3d_backward_weight",
    "is_supported_bwd_weight",
    "bwd_weight_config",
    "default_bwd_weight_config",
    "candidate_bwd_weight_configs",
    "split_count",
    "workspace_elements",
    "grad_weight_empty",
    "register_tuned_bwd_weight",
]


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@triton.jit
def _conv3d_bwd_weight_kernel(
    X,
    GY,
    OUT,
    # Sizes.  ``K_TOTAL`` is ``BATCH * OUT_D * OUT_H * OUT_W``, the reduction
    # length; ``K_CHUNK`` is how much of it one split owns.
    IN_D,
    IN_H,
    IN_W,
    OUT_D,
    OUT_H,
    OUT_W,
    CIN,
    COUT,
    K_TOTAL,
    K_CHUNK,
    GRID,
    # Element strides.  The channel stride of X and GY is 1 by construction --
    # that is what NDHWC means -- so it is neither passed nor multiplied by.
    stride_xn,
    stride_xd,
    stride_xh,
    stride_xw,
    stride_gn,
    stride_gd,
    stride_gh,
    stride_gw,
    # Destination: ``[split][Cout][tap][Cin]`` with the last three contiguous,
    # so one output channel's whole gradient is ``stride_wo`` long.
    stride_ws,
    stride_wo,
    NUM_M: tl.constexpr,
    NUM_CI: tl.constexpr,
    NUM_TG: tl.constexpr,
    TAPS: tl.constexpr,
    TAP_BLOCK: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PD: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DD: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    PADDED: tl.constexpr,
    ROW_ALIGNED: tl.constexpr,
    ATOMIC: tl.constexpr,
    NUM_XCD: tl.constexpr,
    INDEX_DTYPE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    # -- which tile of dW, and which slice of the reduction ------------------
    #
    # Split slowest, tiles fastest.  Every program in one split reads the *same*
    # range of output voxels, so the taps and Cin blocks of a chunk are
    # co-resident and their overlapping reads of X hit cache rather than HBM.
    # The opposite order (splits fastest) spreads concurrent programs over the
    # whole volume and has no reuse at all.
    pid = tl.program_id(0)
    if NUM_XCD > 1:
        # MI300A dispatches workgroups round-robin over its six XCDs, each with
        # its own 4 MiB L2, so the tiles of one split -- which read the *same*
        # chunk of both activations -- land on six different caches and share
        # nothing but the MALL.  Remapping the id so that consecutive logical
        # tiles are consecutive *within* an XCD puts them back together; it is
        # the same device fact that makes ``GROUP_M`` a multiple of 6 in the
        # gather kernel.
        per = GRID // NUM_XCD
        rem = GRID % NUM_XCD
        xcd = pid % NUM_XCD
        seq = pid // NUM_XCD
        pid = (
            tl.where(xcd < rem, xcd * (per + 1), rem * (per + 1) + (xcd - rem) * per)
            + seq
        )
    num_tiles = NUM_M * NUM_CI * NUM_TG
    split = pid // num_tiles
    tile = pid % num_tiles
    pid_m = tile % NUM_M
    rest = tile // NUM_M
    pid_ci = rest % NUM_CI
    tap0 = (rest // NUM_CI) * TAP_BLOCK

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output channels

    # -- the N axis: TAP_BLOCK taps x BLOCK_NC input channels ---------------
    #
    # All of this is hoisted out of the reduction: the column decomposition
    # depends on the tile, not on the voxel.  ``BLOCK_NC``, ``KH`` and ``KW`` are
    # constexpr, so the divisions fold away.
    col = tl.arange(0, BLOCK_N)
    t_local = col // BLOCK_NC
    offs_n = pid_ci * BLOCK_NC + (col % BLOCK_NC)  # input channels
    # ``tl.arange`` needs a power of two, so ``TAP_BLOCK`` is one and cannot
    # divide 27.  Rather than mask the load for the ragged last group -- a
    # predicate on every B tile of every group -- the tap is *clamped*: those
    # columns read a real, in-bounds tap, compute a value nobody wants, and are
    # dropped by the store mask.
    tap_ok = (tap0 + t_local) < TAPS
    tap = tl.minimum(tap0 + t_local, TAPS - 1)
    kd = tap // (KH * KW)
    khw = tap % (KH * KW)
    kh = khw // KW
    kw = khw % KW
    # The column part of the X address: the tap's spatial shift plus the channel.
    x_col = (
        (kd * DD).to(INDEX_DTYPE) * stride_xd
        + (kh * DH).to(INDEX_DTYPE) * stride_xh
        + (kw * DW).to(INDEX_DTYPE) * stride_xw
        + offs_n.to(INDEX_DTYPE)
    )
    col_ok = (offs_n < CIN) & tap_ok

    k_begin = split * K_CHUNK
    k_end = min(k_begin + K_CHUNK, K_TOTAL)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(k_begin, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # -- unravel the fused ndhw voxel index ---------------------------
        #
        # Unlike the forward, this cannot be hoisted out of the reduction: here
        # the *reduction* axis is the volume.  ROW_ALIGNED is what keeps it
        # cheap.  When BLOCK_K divides OUT_W and the chunk is row-aligned a
        # K-tile lies inside a single row of the output, so the whole unravel is
        # scalar (four SALU divisions) and the only vector term is ``ow``.  Every
        # real ScaFFold volume has a power-of-two output extent, so this is the
        # path that runs in production; the general branch below exists for the
        # 8^3 bottleneck (where BLOCK_K > OUT_W) and for the test shapes.
        if ROW_ALIGNED:
            row = k0 // OUT_W
            ow = (k0 - row * OUT_W) + tl.arange(0, BLOCK_K)
            oh = row % OUT_H
            tmp = row // OUT_H
            od = tmp % OUT_D
            idn = tmp // OUT_D
        else:
            ow = offs_k % OUT_W
            tmp = offs_k // OUT_W
            oh = tmp % OUT_H
            tmp = tmp // OUT_H
            od = tmp % OUT_D
            idn = tmp // OUT_D

        # -- A: the upstream gradient, (BLOCK_K, BLOCK_M) -----------------
        #
        # Contiguous along Cout, which is the GEMM's M.  Cast per term rather
        # than after the sum: at scale 8 a single term overflows int32 and the
        # sum would already be wrong before any widening.
        g_row = (
            idn.to(INDEX_DTYPE) * stride_gn
            + od.to(INDEX_DTYPE) * stride_gd
            + oh.to(INDEX_DTYPE) * stride_gh
            + ow.to(INDEX_DTYPE) * stride_gw
        )
        a_ptrs = GY + g_row[:, None] + offs_m[None, :]
        if EVEN_K and EVEN_M:
            a = tl.load(a_ptrs)
        elif EVEN_K:
            a = tl.load(a_ptrs, mask=(offs_m < COUT)[None, :], other=0.0)
        elif EVEN_M:
            a = tl.load(a_ptrs, mask=(offs_k < k_end)[:, None], other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=(offs_k < k_end)[:, None] & (offs_m < COUT)[None, :],
                other=0.0,
            )

        # -- B: the input, (BLOCK_K, BLOCK_N) -----------------------------
        #
        # The row part is the voxel, the column part is (tap shift, channel).
        src_d = od * SD - PD
        src_h = oh * SH - PH
        src_w = ow * SW - PW
        x_row = (
            idn.to(INDEX_DTYPE) * stride_xn
            + src_d.to(INDEX_DTYPE) * stride_xd
            + src_h.to(INDEX_DTYPE) * stride_xh
            + src_w.to(INDEX_DTYPE) * stride_xw
        )
        b_ptrs = X + x_row[:, None] + x_col[None, :]
        if PADDED:
            # Two-dimensional, because the tap now varies down the columns.
            # Unpadded, every tap of an in-range output voxel is in range and
            # all of this compiles out -- but the production case is padded, so
            # this is the body that runs.  See the module docstring.
            in_d = src_d[:, None] + (kd * DD)[None, :]
            in_h = src_h[:, None] + (kh * DH)[None, :]
            in_w = src_w[:, None] + (kw * DW)[None, :]
            mask_b = (
                (in_d >= 0)
                & (in_d < IN_D)
                & (in_h >= 0)
                & (in_h < IN_H)
                & (in_w >= 0)
                & (in_w < IN_W)
            )
            if not EVEN_N:
                mask_b = mask_b & (offs_n < CIN)[None, :]
            if not EVEN_K:
                mask_b = mask_b & (offs_k < k_end)[:, None]
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        elif EVEN_K and EVEN_N:
            b = tl.load(b_ptrs)
        elif EVEN_K:
            b = tl.load(b_ptrs, mask=(offs_n < CIN)[None, :], other=0.0)
        elif EVEN_N:
            b = tl.load(b_ptrs, mask=(offs_k < k_end)[:, None], other=0.0)
        else:
            b = tl.load(
                b_ptrs,
                mask=(offs_k < k_end)[:, None] & (offs_n < CIN)[None, :],
                other=0.0,
            )

        # ``tl.trans`` rather than a strided A load: see the module docstring.
        # ``input_precision`` only bites for fp32 operands, where the backend
        # default splits the dot into reduced-precision pieces; bf16 already
        # accumulates in fp32.  ``more_determinism`` runs in fp32 and has to
        # actually be fp32, so it is asked for explicitly.
        acc = tl.dot(tl.trans(a), b, acc, input_precision=INPUT_PRECISION)

    # -- epilogue ---------------------------------------------------------
    #
    # One expression serves three destinations.  With one split ``stride_ws`` is
    # 0 and ``OUT`` is the real gradient in its own dtype, so the workspace and
    # the reduction pass disappear entirely; with several it is an fp32 slice;
    # with ATOMIC it is a single fp32 accumulator every split adds into.
    out_ptrs = (
        OUT
        + split.to(INDEX_DTYPE) * stride_ws
        + offs_m.to(INDEX_DTYPE)[:, None] * stride_wo
        + (tap * CIN + offs_n)[None, :]
    )
    if EVEN_M:
        mask_o = tl.broadcast_to(col_ok[None, :], (BLOCK_M, BLOCK_N))
    else:
        mask_o = (offs_m < COUT)[:, None] & col_ok[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask_o, sem="relaxed")
    else:
        tl.store(out_ptrs, acc.to(OUT.dtype.element_ty), mask=mask_o)


@triton.jit
def _reduce_partials_kernel(
    PARTIAL,
    OUT,
    N_ELEM,
    SPLITS,
    BLOCK: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Sum the split-K partials in index order and cast to the output dtype.

    The determinism of the whole direction rests on this loop.  ``SPLITS`` is a
    runtime argument rather than a constexpr (one compile serves every shape),
    but the loop is sequential, its bound is a pure function of the problem, and
    ``tl.sum`` over a fixed tile shape is a fixed order -- so two runs of the
    same problem add the same numbers in the same order.  A tree reduction is
    equally reproducible; a ``tl.atomic_add`` is not, which is the whole point.

    ``BLOCK_S`` splits are read at a time rather than one because the gradient
    can be *smaller* than one program's tile (the ``k=1`` head is 384 elements):
    a one-split-at-a-time loop is then a single workgroup serializing one
    dependent memory latency per split, and that is the only shape where the
    deterministic path costs anything against the atomic one.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_ELEM
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = offs.to(tl.int64)
    stride = N_ELEM.to(tl.int64)
    offs_s = tl.arange(0, BLOCK_S)
    for s0 in range(0, SPLITS, BLOCK_S):
        s = s0 + offs_s
        tile = tl.load(
            PARTIAL + s.to(tl.int64)[:, None] * stride + base[None, :],
            mask=(s < SPLITS)[:, None] & mask[None, :],
            other=0.0,
        )
        acc += tl.sum(tile, axis=0)
    tl.store(OUT + offs, acc.to(OUT.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BwdWeightConfig(ConvConfig):
    """A launch configuration with the knobs only this direction has.

    A subclass rather than more fields on :class:`ConvConfig`, because the
    gather directions would never set them and a config printed in a forward
    sweep should not grow suffixes it cannot use.  Everything else -- the gfx942
    legality rules, the LDS model, ``launch_kwargs`` -- is inherited unchanged,
    and ``BLOCK_N`` keeps its meaning as the *full* tile width so that both of
    those stay correct.
    """

    #: Split-K partial count.  ``0`` means "derive it from the shape", which is
    #: what the shipped path does; a non-zero value pins it so that a sweep can
    #: see the shape of the curve rather than one point on it.
    SPLIT_K: int = 0
    #: How many taps one tile spans.  ``BLOCK_N = TAP_BLOCK * BLOCK_NC``.
    TAP_BLOCK: int = 1
    #: XCD count to swizzle the program id for; 0 or 1 disables it.  Six is
    #: MI300A's.  The swizzle buys little on its own, but every row of this
    #: direction's tuning table was measured with it on, so turning it off would
    #: make those rows describe a kernel nobody ran.
    NUM_XCD: int = 6

    @property
    def BLOCK_NC(self) -> int:
        """Input channels per tap in the tile."""
        return self.BLOCK_N // self.TAP_BLOCK

    def __str__(self) -> str:
        return (
            super().__str__()
            + (f"/tb{self.TAP_BLOCK}" if self.TAP_BLOCK != 1 else "")
            + (f"/x{self.NUM_XCD}" if self.NUM_XCD != 6 else "")
            + (f"/sk{self.SPLIT_K}" if self.SPLIT_K else "")
        )

    def validate(self, dtype: torch.dtype) -> str | None:
        why = super().validate(dtype)
        if why is not None:
            return why
        if self.SPLIT_K < 0:
            return "SPLIT_K must be non-negative (0 means derive from shape)"
        if self.TAP_BLOCK < 1:
            return "TAP_BLOCK must be at least 1"
        if self.BLOCK_N % self.TAP_BLOCK:
            return f"BLOCK_N must be a multiple of TAP_BLOCK={self.TAP_BLOCK}"
        if self.NUM_XCD < 0:
            return "NUM_XCD must be non-negative"
        return None


#: MI300A's compute units.  It appears here rather than being read from the
#: device on purpose: the split count has to be a pure function of the *problem*
#: for the determinism claim to hold, so a device with a different CU count gets
#: a differently-tuned kernel rather than a differently-ordered reduction.
_CU_COUNT = 228

#: Waves of programs the split count aims for.  A sweep chose four, but the
#: *whole* wave matters more than the value: every program in this kernel does
#: the same amount of work, so a grid of four and a half waves runs five and
#: idles through half of the last one.  Hence the snap in :func:`split_count`,
#: which is worth more than the target itself.
_SPLIT_TARGET_WAVES = 4

#: A split must be worth its epilogue.  Every split writes a full
#: ``BLOCK_M x BLOCK_N`` fp32 tile and the reduction reads every one of them
#: back, so at a short reduction the partials become the dominant traffic: the
#: rule is that they stay under this fraction of the main loop's.  Without it a
#: short reduction into a large gradient asks for splits it cannot pay for.
_MAX_EPILOGUE_FRACTION = 10

#: And a split must be at least a few K-tiles long, or the loop's own prologue
#: is most of it.
_MIN_K_TILES_PER_SPLIT = 4

#: Ceiling on the fp32 partial workspace.  The product to watch is
#: ``splits * Cout * taps * Cin * 4``, which at the widest channel pairs is
#: already large for a single split.  Bounded here so that no shape in the
#: corpus can ask for an allocation that fails mid-run.
_WORKSPACE_BYTES = 256 * 1024 * 1024


def _fit_bwd_weight_to_lds(cfg: BwdWeightConfig, dtype: torch.dtype) -> BwdWeightConfig:
    """Shrink ``BLOCK_K`` until the operand tiles fit in LDS.

    Only ``BLOCK_K`` moves.  ``BLOCK_M`` and ``BLOCK_N`` are already bounded by
    ``Cout`` and ``TAP_BLOCK * Cin`` -- shrinking either throws away the
    arithmetic intensity this kernel is short of -- whereas ``BLOCK_K`` is the
    reduction depth and costs only reuse.  That is the same ordering
    :func:`~triton_conv3d.gather_gemm._fit_to_lds` uses and the same reason.
    """
    kdim = _MFMA_KDIM.get(dtype, {}).get(cfg.matrix_instr_nonkdim)
    if kdim is None:
        return cfg
    while cfg.lds_bytes(dtype) > _LDS_BYTES and cfg.BLOCK_K // 2 >= kdim:
        cfg = dataclasses.replace(
            cfg,
            BLOCK_K=cfg.BLOCK_K // 2,
            kpack=1 if cfg.BLOCK_K // 2 <= 16 else cfg.kpack,
        )
    warps = max(1, min(cfg.num_warps, cfg.BLOCK_M * cfg.BLOCK_N // 256))
    return dataclasses.replace(cfg, num_warps=1 << (warps.bit_length() - 1))


def default_bwd_weight_config(
    cout: int,
    cin: int,
    kernel: Sequence[int],
    k_total: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    padded: bool = False,
) -> BwdWeightConfig:
    """A config that is legal for any shape and close to tuned for most.

    The tile is bounded by the *problem*, not chosen freely: ``BLOCK_M`` cannot
    usefully exceed ``Cout``, and one tap's N extent is exactly ``Cin``.  That is
    the opposite of the forward, where M is the volume and a tall tile is always
    available -- and it is why ``TAP_BLOCK`` exists.

    ``TAP_BLOCK`` brings the tile *width* up to a target of 256 columns, or 512
    when ``Cout`` is 64 or less and the tile has no height to trade against.
    Where ``Cin`` is 256 or more the channels reach 256 columns unaided and
    adding taps on top costs; where ``Cout`` is small the taps are the only way
    to get there at all.

    ``padded`` is accepted and *not consulted*: the wide tile wins on padded
    problems too (see the module docstring).  It stays in the signature because
    it describes the problem rather than the policy and every caller already
    computes it, but nothing here branches on it.
    """
    k = _triple(kernel, "kernel")
    taps = math.prod(k)
    # 256, not 128: at ``Cout >= 256`` a 128-row tile leaves half the available
    # M on the floor, and the tile's arithmetic intensity
    # ``BLOCK_M*BLOCK_N/(BLOCK_M+BLOCK_N)`` rises with the shorter side.
    # ``_pow2_at_most`` keeps ``BLOCK_M <= Cout``, which binds at ``Cout = 128``;
    # widening N further instead of raising M there is a dead end.  See
    # ``_TUNED_BWD_W``.
    block_m = _pow2_at_most(cout, 256)
    block_nc = _pow2_at_most(cin, 256)
    # ``TAP_BLOCK`` is a power of two because ``BLOCK_N`` has to be one
    # (``tl.arange`` refuses anything else), so it never divides 27 exactly; the
    # ragged last group is handled by clamping in the kernel.
    target_width = 512 if block_m <= 64 else 256
    tap_block = 1
    while tap_block * 2 <= taps and block_nc * tap_block * 2 <= target_width:
        tap_block *= 2
    block_n = block_nc * tap_block
    nonkdim = 16
    kdim = _MFMA_KDIM[dtype][nonkdim]
    # The deepest K-tile that still leaves the operands inside LDS.  Deeper is
    # better for the loop's own overhead and does nothing for the intensity, so
    # it is the axis that gives way -- and it is the only one that can, since
    # BLOCK_M and BLOCK_N are pinned to the problem above.
    itemsize = torch.empty((), dtype=dtype).element_size()
    # ...but only to 32 once the tile is 256 rows tall.  A 256-row, 256-column
    # tile with a 64-deep K falls off a cliff at the long-reduction sites, at an
    # identical program count, so it is not parallelism.  The LDS budget at
    # ``256 + 256`` columns lands on exactly 64 in bf16 -- straight into the
    # cliff -- so the cap has to be explicit here.  The tuned table is free to
    # ship 64 where a measurement says so.
    block_k = _pow2_at_most(
        _LDS_BYTES // (itemsize * (block_m + block_n)), 32 if block_m >= 256 else 64
    )
    block_k = max(kdim, block_k - block_k % kdim)
    # A K axis shorter than one tile is not an error, only waste; shrink so the
    # tiny synthetic shapes do not run a mostly-masked reduction.
    while block_k > kdim and block_k > k_total:
        block_k //= 2
    return _fit_bwd_weight_to_lds(
        BwdWeightConfig(
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=6,
            num_warps=min(8, max(1, block_m * block_n // 256)),
            num_stages=2,
            matrix_instr_nonkdim=nonkdim,
            kpack=1 if block_k <= 16 else 2,
            TAP_BLOCK=tap_block,
        ),
        dtype,
    )


def _row_aligned(block_k: int, out_w: int) -> bool:
    """Whether a K-tile is guaranteed to lie inside one row of the output."""
    return out_w % block_k == 0


def split_count(
    cfg: BwdWeightConfig,
    cout: int,
    cin: int,
    taps: int,
    k_total: int,
    out_w: int,
) -> tuple[int, int]:
    """``(splits, chunk)``: how the reduction axis is divided, and by how much.

    A pure function of the shape and the config, which is what makes the
    deterministic path reproducible process to process:

    * up: enough programs to fill the device :data:`_SPLIT_TARGET_WAVES` times
      over, which is the whole reason split-K is here;
    * down: the epilogue must stay a minority of the traffic
      (:data:`_MAX_EPILOGUE_FRACTION`), a split must be a few K-tiles long, and
      the workspace ceiling must hold.  The first two bound the split count the
      *target* asks for; only the workspace ceiling survives the wave snap
      below, and the comment there says why;
    * and then the result is snapped to a whole number of waves, which is the
      step that actually matters.  Every program here does the same work, so a
      grid of four and a half waves runs five and idles through half of the
      last; the split-count curve is mostly that sawtooth, and reading it as
      "the cache prefers short chunks" leads to far more splits than the shape
      wants.

    ``chunk`` is rounded up to a whole number of output *rows* when the tile is
    row-aligned, because the kernel's cheap scalar unravel needs every K-tile to
    stay inside one row; otherwise to a whole number of K-tiles.
    """
    tiles = (
        -(-cout // cfg.BLOCK_M) * -(-cin // cfg.BLOCK_NC) * -(-taps // cfg.TAP_BLOCK)
    )
    k_tiles = -(-k_total // cfg.BLOCK_K)
    per_split = cout * taps * cin * 4
    ceiling = max(
        1, min(k_tiles // _MIN_K_TILES_PER_SPLIT, _WORKSPACE_BYTES // per_split)
    )

    if cfg.SPLIT_K:
        want = min(cfg.SPLIT_K, ceiling)
    else:
        want = -(-_SPLIT_TARGET_WAVES * _CU_COUNT // tiles)
        # The partials are written once and read once, in fp32; the loop reads
        # ``BLOCK_M + BLOCK_N`` operand elements per reduction element per tile.
        # Where the gradient is large and the volume small this is the bound
        # that bites, and without it such a site pays for splits it cannot use.
        loop_elems = tiles * k_total * (cfg.BLOCK_M + cfg.BLOCK_N)
        epilogue_elems = cout * taps * cin * 2 * 2
        epilogue_bound = max(
            1, loop_elems // (_MAX_EPILOGUE_FRACTION * max(1, epilogue_elems))
        )
        want = max(1, min(want, epilogue_bound, ceiling))
        # The snap goes last and outranks the epilogue bound; that ordering is
        # deliberate.  ``round`` here can only move ``want`` up: where
        # ``tiles * want`` is under half a wave it gives 0, ``max(1, ...)``
        # forces one whole wave, and ``want`` becomes ``_CU_COUNT // tiles``,
        # which can be several times the epilogue-bounded value.  Re-applying
        # the bound after the snap was tried and loses: half an idle device
        # costs more than a doubled epilogue whenever the grid is that small,
        # and a grid that small is exactly the condition under which ``round``
        # rounds to zero.  The workspace ceiling is different in kind -- an
        # allocation that fails is not a slow kernel -- and is re-applied.
        waves = max(1, round(tiles * want / _CU_COUNT))
        want = max(1, min(waves * _CU_COUNT // tiles, ceiling))
    want = max(1, want)

    align = out_w if _row_aligned(cfg.BLOCK_K, out_w) else cfg.BLOCK_K
    chunk = -(-(-(-k_total // want)) // align) * align
    return -(-k_total // chunk), chunk


#: Seed tiles, ``(BLOCK_M, BLOCK_NC, TAP_BLOCK, BLOCK_K, num_warps)``.  Not the
#: forward's grid: there M is the volume and the useful tiles are tall, here M is
#: ``Cout``, one tap of N is ``Cin``, and the volume lives in ``BLOCK_K`` and in
#: the split count.  So the axes worth sweeping are ``TAP_BLOCK`` and
#: ``BLOCK_K``, neither of which the forward's seed grid varies at all.
_SEED_TILES: tuple[tuple[int, int, int, int, int], ...] = (
    (64, 64, 1, 64, 4),
    (64, 64, 1, 128, 4),
    (64, 64, 1, 256, 4),
    (64, 64, 2, 64, 4),
    (64, 64, 2, 128, 8),
    (64, 64, 4, 32, 4),
    (64, 64, 4, 64, 8),
    (64, 64, 8, 16, 4),
    (64, 64, 8, 32, 8),
    (64, 128, 1, 64, 4),
    (64, 128, 1, 128, 4),
    (64, 128, 2, 32, 4),
    (64, 128, 2, 64, 8),
    (64, 128, 4, 32, 8),
    (128, 64, 1, 64, 4),
    (128, 64, 1, 128, 8),
    (128, 64, 2, 32, 4),
    (128, 64, 2, 64, 8),
    (128, 64, 4, 32, 8),
    (128, 128, 1, 64, 8),
    (128, 128, 1, 128, 8),
    (128, 128, 2, 32, 8),
    (128, 128, 2, 64, 8),
    (128, 256, 1, 64, 8),
    (256, 128, 1, 64, 8),
    (256, 64, 1, 64, 4),
    # Tall *and* wide, for the ``Cout >= 256`` sites where the M axis has the
    # room.  Both ``BLOCK_K`` are here because the choice between them is
    # shape-dependent rather than a preference: 64 wins at the short-reduction
    # sites and collapses at the long ones.
    (256, 256, 1, 32, 8),
    (256, 256, 1, 64, 8),
    (32, 64, 1, 128, 4),
    (32, 64, 4, 64, 4),
    (16, 64, 1, 128, 4),
    (16, 64, 4, 64, 4),
)

#: Split counts worth trying.  0 means "let :func:`split_count` decide", which is
#: what the shipped path does.
_SEED_SPLITS: tuple[int, ...] = (0, 1, 4, 16, 64, 256)


def candidate_bwd_weight_configs(
    cout: int,
    cin: int,
    kernel: Sequence[int],
    k_total: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    splits: Sequence[int] = _SEED_SPLITS,
    padded: bool = False,
) -> list[BwdWeightConfig]:
    """Configs worth timing for one shape, already pruned to legal ones.

    Pruned rather than shrunk: shrinking an oversized tile folds two seed
    entries onto one config and silently double-counts it in a best-of sweep.
    """
    taps = math.prod(_triple(kernel, "kernel"))
    m2 = max(16, triton.next_power_of_2(cout))
    c2 = max(16, triton.next_power_of_2(cin))
    k2 = max(16, triton.next_power_of_2(k_total))
    out: list[BwdWeightConfig] = []
    seen: set[BwdWeightConfig] = set()
    for bm, bnc, tb, bk, seed_warps in _SEED_TILES:
        # A BLOCK_M past Cout is pure padding (M is Cout, not a volume), a
        # BLOCK_NC past Cin is padding for the same reason, and a BLOCK_K past
        # the whole reduction is padding too.
        if bm > m2 or bnc > c2 or bk > k2 or tb > taps:
            continue
        for warps in {4, 8, seed_warps}:
            for sk in splits:
                cfg = BwdWeightConfig(
                    BLOCK_M=bm,
                    BLOCK_N=bnc * tb,
                    BLOCK_K=bk,
                    GROUP_M=6,
                    num_warps=warps,
                    num_stages=2,
                    matrix_instr_nonkdim=16,
                    kpack=1 if bk <= 16 else 2,
                    SPLIT_K=sk,
                    TAP_BLOCK=tb,
                )
                if (
                    cfg.validate(dtype) is not None
                    or cfg.lds_bytes(dtype) > _LDS_BYTES
                    or cfg in seen
                ):
                    continue
                seen.add(cfg)
                out.append(cfg)
    if not out:
        out.append(
            default_bwd_weight_config(cout, cin, kernel, k_total, dtype, padded=padded)
        )
    return out


def _tuned(
    bm: int, bnc: int, tb: int, bk: int, warps: int, sk: int = 0, nk: int = 32
) -> BwdWeightConfig:
    """One tuned row.

    ``nk`` defaults to 32 here and to 16 everywhere else in the package;
    :data:`_TUNED_BWD_W` says why that asymmetry is deliberate.
    """
    return BwdWeightConfig(
        BLOCK_M=bm,
        BLOCK_N=bnc * tb,
        BLOCK_K=bk,
        GROUP_M=6,
        num_warps=warps,
        num_stages=2,
        matrix_instr_nonkdim=nk,
        kpack=1 if bk <= 16 else 2,
        SPLIT_K=sk,
        TAP_BLOCK=tb,
    )


#: Backward-weight winners, keyed by ``(dtype, Cin, Cout, kernel)`` -- the
#: convolution a reader would name, as in the other two tables, even though this
#: direction's GEMM has ``Cout`` on M and ``taps * Cin`` on N.
#:
#: A sweep over the corpus chose the rows and
#: ``triton_conv3d/bench/conv_bench.py`` reproduces it.  Only channel pairs that
#: were actually timed appear; a miss falls to
#: :func:`default_bwd_weight_config` plus :func:`split_count` rather than to an
#: extrapolation.
#:
#: ``SPLIT_K`` is left at 0 -- "derive from the shape" -- in every entry, and
#: that is a finding rather than an omission.  The same channel pair occurs at
#: volumes orders of magnitude apart, the split count is the one knob that
#: genuinely has to follow the volume, and pinning a sweep's winner would carry
#: one volume's answer to every other.
#:
#: ``matrix_instr_nonkdim`` is 32 in this table and 16 in every other table in
#: the package.  This direction's ``tl.dot(tl.trans(a), b)`` lowers on gfx942 to
#: an *element-wise* transpose of A through LDS -- 128 two-byte ``ds_read_u16``
#: per loop body against 16 ``ds_read_b128`` for the untransposed operand, at a
#: structural 50% bank-conflict rate -- which leaves the LDS pipe busy while the
#: matrix core idles.  The 32x32x8 fragment halves the MFMA instruction count
#: for identical FLOPs and so stops the MFMA stream competing with that
#: transpose for issue slots.  It does *not* remove the transpose: the ``nk32``
#: ISA still emits 128 ``ds_read_u16``.
#:
#: The rule is per direction, and within this direction it is per channel pair.
#: The forward has no transposed operand and wants 16; backward-data reuses the
#: forward kernel and wants 16; and here the sign reverses at ``Cout = 1024``,
#: so those two pairs keep 16 as well.  :func:`default_bwd_weight_config` --
#: which serves fp32, the transposed operator and every untuned pair -- stays at
#: 16 too: a heuristic is the path with no measurement behind it, and a knob
#: whose sign is shape-dependent is exactly the extrapolation this table refuses
#: to make elsewhere.
#:
#: Results are *not* bitwise identical to the ``nonkdim=16`` kernel -- a
#: different MFMA fragment sums the same products in a different order.  They
#: are still bitwise reproducible, which is what the determinism claim says: the
#: split-K partition, the reduction's tiling and its ``tl.sum`` order are all
#: unchanged.
_TUNED_BWD_W: dict[tuple, BwdWeightConfig] = {
    # The segmentation head.  ``BLOCK_M`` is ``Cout = 6`` rounded up to 16, so
    # ``nonkdim=32`` is illegal here and the axis above does not apply.  What
    # does apply is ``num_warps``, which ``candidate_bwd_weight_configs`` never
    # takes below 4; one warp wins at every head volume.  This site can still
    # lose to MIOpen, and the row is here so that the adapter's block-list is
    # decided against the best this kernel can do.
    tune_key(torch.bfloat16, 64, 6, (1, 1, 1)): _tuned(16, 64, 1, 64, 1, nk=16),
    # The one transposed row.  ``conv_transpose3d_backward_weight`` calls this
    # module with the operator's widths *swapped* -- it passes the transposed
    # weight's own ``(Cin, Cout, k, k, k)`` shape, whose first axis is this
    # reduction's M -- so the key below reads ``(64, 128)`` and the module a
    # reader would name is ``ConvTranspose3d(128, 64, 2, stride=2)``.
    #
    # It is the only one of the four transposed channel pairs where
    # ``nonkdim=32`` wins; the other three have no consistent sign and stay on
    # ``default_bwd_weight_config``, i.e. at 16.  The split is mechanical, not
    # luck: this pair is the only transposed site whose tile is
    # ``BLOCK_M = 128``, and the other three reach ``BLOCK_M = 256``, which
    # already amortises the transpose -- the same boundary the ``k=3`` rows
    # show.  Everything except ``nonkdim`` here restates what the heuristic
    # already picks, so the row cannot drift away from it silently.
    tune_key(torch.bfloat16, 64, 128, (2, 2, 2)): _tuned(128, 64, 4, 64, 8),
    **{
        tune_key(torch.bfloat16, cin, cout, (3, 3, 3)): cfg
        for (cin, cout), cfg in {
            # The UNet stem, where the N axis is the forward's disease one axis
            # over: ``BLOCK_NC = _pow2_at_most(Cin, 256) = 16`` against
            # ``Cin = 3``, times ``TAP_BLOCK = 16`` covering 27 taps in two
            # groups, issues 512 columns where only 81 are live.
            # ``BLOCK_NC = 4`` (which is what ``bnc=4, tb=16`` spells, keeping
            # ``BLOCK_N = 64``) issues 128 columns for the same 81.  That is the
            # structural optimum for this axis and no kernel change can beat it:
            # a dense N would need ``BLOCK_N = 96``, which ``tl.arange`` cannot
            # express, and a dense ``BLOCK_N = 128`` is the same ratio.
            #
            # ``nk=16`` rather than this table's 32, and one warp rather than
            # the 4 ``candidate_bwd_weight_configs`` floors at: both were
            # measured for this row rather than inherited.
            (3, 64): _tuned(64, 4, 16, 64, 1, nk=16),
            # ``Cout = 64`` is where TAP_BLOCK earns its existence: the tile
            # can only reach 512 columns through the taps, and reaching them is
            # worth a large factor over the one-tap form.
            (64, 64): _tuned(64, 64, 8, 16, 4),
            # ``(64, 128)`` is the one pair of the three whose ``Cout`` has room
            # for a 128-row tile, and it wants one.  This table is shared by the
            # padded and the unpadded form of the problem, and the row wins in
            # both.  ``nk=32`` is worth a further margin over the ``nk16`` twin
            # of the same tile, which is this table's usual sign;
            # ``BLOCK_K=32`` loses and is not taken.
            (64, 128): _tuned(128, 64, 4, 64, 8),
            (128, 64): _tuned(64, 64, 8, 16, 4),
            (128, 128): _tuned(128, 128, 2, 64, 8),
            (128, 256): _tuned(128, 128, 2, 64, 8),
            # From ``Cin = 256`` up the channels alone reach a 256-column tile
            # and the taps are not needed for it; ``TAP_BLOCK > 1`` then
            # *costs*, because a wider tile past 256 buys less than the register
            # pressure takes.
            (256, 128): _tuned(128, 256, 1, 64, 8),
            (512, 256): _tuned(128, 256, 1, 64, 8),
            # ``BLOCK_M = 256`` wherever ``Cout`` has the room.  Every entry
            # below was raced against the 128-row tile at every volume the
            # corpus has for that channel pair -- per pair, not per site,
            # because that is what this table is keyed on -- and the taller tile
            # wins at all of them.  The two smallest sites in the corpus would
            # rather have a 16-deep K-tile, and are not tuned for: their pair's
            # other volumes are several times larger and want 64.
            #
            # ``BLOCK_K`` is 32 at ``(256, 256)`` and 64 elsewhere, and that is
            # a measurement rather than an oversight: at ``(256, 256)`` the
            # 64-deep tile is behind at all three volumes, while at the sites
            # below it is ahead.  See ``default_bwd_weight_config`` for the
            # cliff that makes the heuristic refuse 64 at this width.
            (256, 256): _tuned(256, 256, 1, 32, 8),
            (256, 512): _tuned(256, 256, 1, 64, 8),
            (512, 512): _tuned(256, 256, 1, 64, 8),
            (1024, 512): _tuned(256, 256, 1, 64, 8),
            # ``nk=16``: the two ``Cout = 1024`` pairs, the only rows here that
            # keep the forward's value.  ``nonkdim=32`` loses at their larger
            # volumes; see the note above the table.
            (512, 1024): _tuned(256, 256, 1, 64, 8, nk=16),
            (1024, 1024): _tuned(256, 256, 1, 64, 8, nk=16),
        }.items()
    },
}
# ``(128, 256)`` and ``(512, 256)`` keep their 128-row tile, and that is a
# result rather than an omission -- both were raced with ``BLOCK_M = 256`` at
# every ``BLOCK_K`` and volume the corpus has for them:
#
# * ``(128, 256)``: a genuine tie.  ``Cin = 128`` forces ``TAP_BLOCK = 2`` to
#   reach 256 columns, and the tap traffic eats the intensity the taller tile
#   buys.
# * ``(512, 256)``: a win at the smallest volume and a loss at the two larger
#   ones, so the pair is a net loss, and this table is keyed per channel pair
#   and cannot split one by volume.  A 256-row, 256-column tile with a 64-deep
#   K there is the cliff named in :func:`default_bwd_weight_config`.


def register_tuned_bwd_weight(dtype, cin, cout, kernel, config) -> None:
    _TUNED_BWD_W[tune_key(dtype, cin, cout, tuple(kernel))] = config


def bwd_weight_config(
    cout: int,
    cin: int,
    kernel: Sequence[int],
    k_total: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    padded: bool = False,
) -> BwdWeightConfig:
    """The config :func:`conv3d_backward_weight` would pick for this problem.

    ``padded`` does not select a different row.  Every ScaFFold convolution with
    ``k > 1`` reaches here with ``padded=True``, and the wide-tile rows win
    there too (see the module docstring), so the argument is threaded through
    for :func:`default_bwd_weight_config` alone.

    Determinism is unaffected in the sense the package claims it: a different
    config is a different reduction order, so results are *not* bitwise equal
    across configs, but they remain bitwise reproducible run to run and process
    to process, because :func:`split_count` is still a pure function of the
    shape and the config.  See the module docstring.

    Two configs can therefore differ even where :func:`split_count` returns the
    same splits: ``matrix_instr_nonkdim`` changes the MFMA fragment, and a
    different fragment sums the same products in a different order within one
    ``BLOCK_K``.  The split count alone is not enough to assert that two configs
    agree.  Where they do differ the difference stays inside
    :func:`~triton_conv3d.reference.error_bound`.
    """
    k = _triple(kernel, "kernel")
    tuned = _TUNED_BWD_W.get(tune_key(dtype, cin, cout, tuple(k)))
    if tuned is not None:
        return _fit_bwd_weight_to_lds(tuned, dtype)
    return default_bwd_weight_config(cout, cin, k, k_total, dtype, padded=padded)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def workspace_elements(splits: int, cout: int, cin: int, kernel: Sequence[int]) -> int:
    """fp32 elements :func:`conv3d_backward_weight` needs for ``splits`` splits."""
    return splits * cout * cin * math.prod(_triple(kernel, "kernel"))


def grad_weight_empty(
    cout: int, cin: int, kernel: Sequence[int], *, dtype, device
) -> torch.Tensor:
    """An empty gradient in the layout the kernel writes: ``[Cout][tap][Cin]``.

    That is exactly ``channels_last_3d`` for a ``(Cout, Cin, kd, kh, kw)``
    tensor -- its memory order is ``Cout, kd, kh, kw, Cin`` -- so the natural
    output of this GEMM is already the memory format ScaFFold runs in.  The
    forward and backward-data both need a weight transform on the way *in*; this
    direction gets the layout for free on the way out.

    ``memory_format=`` on the allocation rather than ``.contiguous(...)`` after
    it, and that is not cosmetic: ``torch.empty(shape).contiguous(memory_format=
    channels_last_3d)`` allocates the tensor in NCDHW and then runs a permuting
    device copy to reach the layout it was going to be asked for anyway.  The
    contents are undefined either way, so the copy transports nothing and costs
    far more than the one-shot allocation.
    """
    k = _triple(kernel, "kernel")
    return torch.empty(
        (cout, cin, *k),
        dtype=dtype,
        device=device,
        memory_format=torch.channels_last_3d,
    )


def _validate_out(
    gw: torch.Tensor,
    cout: int,
    cin: int,
    k: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> str | None:
    """``None`` if ``gw`` can be written as this problem's gradient, else why not.

    Shaped like :meth:`ConvConfig.validate` because it is the same kind of
    guard: nothing downstream looks at ``gw`` again.  The reduction pass is
    launched with ``n_elem = Cout * taps * Cin`` derived from ``weight_shape``
    and stores that many elements into ``gw`` whatever ``gw`` actually is, and
    the one-split fast path hands the tile kernel ``gw``'s data pointer
    directly.

    All four clauses have teeth:

    * shape, which the stride comparison below *cannot* see.  None of the five
      strides depends on ``Cout``, so a gradient allocated for a small ``Cout``
      is stride-identical to one allocated for a large ``Cout`` with the same
      ``Cin`` and kernel.  Without this clause the large problem writes past the
      end of the small allocation, into whatever the caching allocator handed
      out next: no fault and no exception, just some other live tensor wrong
      later.
    * strides: the reduction pass treats both the workspace and the destination
      as flat ``[Cout][tap][Cin]`` arrays, so an ``out=`` in the default
      contiguous layout would be filled with a correctly shaped, *transposed*
      answer.
    * device: the kernel launches on the current device and dereferences
      whatever pointer it is given.  ScaFFold runs four GPUs per node, and with
      peer access enabled a foreign pointer does not fault -- it scribbles on
      another rank.
    * dtype: the epilogue casts to ``OUT.dtype.element_ty``, so a mismatched
      ``out=`` returns a gradient in a dtype the caller's optimizer is not
      expecting rather than raising, and at an integer dtype the cast truncates.
    """
    want_shape = (cout, cin, *k)
    if tuple(gw.shape) != want_shape:
        return (
            f"shape must be {want_shape} (Cout x Cin x kernel); got "
            f"{tuple(gw.shape)} -- the strides alone cannot tell these apart, "
            "because none of them depends on Cout"
        )
    if gw.device != device:
        return f"device must be {device}; got {gw.device}"
    if gw.dtype != dtype:
        return f"dtype must match the operands' {dtype}; got {gw.dtype}"
    taps = k[0] * k[1] * k[2]
    want = (taps * cin, 1, k[1] * k[2] * cin, k[2] * cin, cin)
    got = tuple(gw.stride())
    # An extent of 1 makes its stride unobservable, so only compare the ones
    # that can be told apart -- ``k=1`` is a real corpus shape.
    if not all(g == w for g, w, n in zip(got, want, want_shape) if n > 1):
        return (
            "must have channels_last_3d strides -- the reduction pass treats it "
            f"as [Cout][tap][Cin]; want {want}, got {got}"
        )
    return None


def is_supported_bwd_weight(
    input: torch.Tensor,
    weight_shape: Sequence[int],
    grad_output: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> bool:
    """Whether :func:`conv3d_backward_weight` will serve this call.

    Same asymmetry as the other two predicates: the caller's fallback is MIOpen,
    which is correct everywhere, so a false negative costs a little speed and a
    false positive returns a wrong gradient.

    Unlike backward-data this direction has no stride restriction.  The
    reduction axis is the *output* voxel and the input coordinate
    ``o*s + t*dil - p`` is a function of it, so a stride is three extra
    multiplies rather than a scatter into a sub-lattice.  It is supported and
    tested, though ScaFFold's corpus never uses one on a non-transposed
    convolution.
    """
    if groups != 1:
        return False
    if input.dim() != 5 or grad_output.dim() != 5 or len(tuple(weight_shape)) != 5:
        return False
    if input.dtype != grad_output.dtype or input.dtype not in _MFMA_KDIM:
        return False
    # Same device, not merely both on *a* device.  Triton launches on the current
    # device and dereferences the other pointer anyway; ScaFFold runs four GPUs
    # per node, where peer access turns that into another rank's data rather than
    # a fault -- a plausible wrong gradient instead of a crash.  The same clause
    # is in ``gather_gemm.is_supported``: the caller picks between the gates by
    # direction, so a hole in one of them is a hole in the ladder.
    if (
        not input.is_cuda
        or not grad_output.is_cuda
        or grad_output.device != input.device
    ):
        return False
    try:
        s = _triple(stride, "stride")
        p = _triple(padding, "padding")
        d = _triple(dilation, "dilation")
    except ValueError:
        return False
    cout, cin, *k = (int(v) for v in weight_shape)
    if any(v < 1 for v in s + d + tuple(k)) or any(v < 0 for v in p):
        return False
    if cout < 1 or cin < 1:
        return False
    n, in_c, *in_sp = (int(v) for v in input.shape)
    if in_c != cin or int(grad_output.shape[1]) != cout:
        return False
    if int(grad_output.shape[0]) != n:
        return False
    for i in range(3):
        eff = d[i] * (k[i] - 1) + 1
        if in_sp[i] + 2 * p[i] < eff:
            return False
        if int(grad_output.shape[2 + i]) != (in_sp[i] + 2 * p[i] - eff) // s[i] + 1:
            return False
    return True


def conv3d_backward_weight(
    input: torch.Tensor,
    weight_shape: Sequence[int],
    grad_output: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    *,
    deterministic: bool = True,
    config: BwdWeightConfig | None = None,
    workspace: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gradient of a 3-D convolution with respect to its weight.

    ``input`` and ``grad_output`` are ``channels_last_3d``; the returned
    gradient is ``channels_last_3d`` too, which for a weight is the layout this
    GEMM produces natively (see :func:`grad_weight_empty`).  The argument order
    mirrors ``torch.nn.grad.conv3d_weight``.

    ``deterministic`` defaults to True and deliberately does *not* follow
    ``torch.are_deterministic_algorithms_enabled()``: the atomic path exists to
    price determinism, not to be selected, and tying it to torch's flag would
    hand every ScaFFold run the nonreproducible path silently.  A caller who
    wants it has to say so.

    ``workspace`` hoists the fp32 partial buffer out of the call, the way
    ``weight_rsck`` hoists the weight transform in the other two directions.
    :func:`split_count` and :func:`workspace_elements` say how big it must be.
    """
    if not is_supported_bwd_weight(
        input, weight_shape, grad_output, stride, padding, dilation, groups
    ):
        raise NotImplementedError(
            f"unsupported: input={tuple(input.shape)}/{input.dtype} "
            f"weight_shape={tuple(weight_shape)} "
            f"grad_output={tuple(grad_output.shape)}/{grad_output.dtype} "
            f"stride={stride} padding={padding} dilation={dilation} "
            f"groups={groups}"
        )
    sd, sh, sw = _triple(stride, "stride")
    pd, ph, pw = _triple(padding, "padding")
    dd, dh, dw = _triple(dilation, "dilation")
    cout, cin, *k = (int(v) for v in weight_shape)
    kd, kh, kw = k
    taps = kd * kh * kw
    padded = pd > 0 or ph > 0 or pw > 0

    # NDHWC is not a preference, it is the layout the addressing assumes.
    x = input.contiguous(memory_format=torch.channels_last_3d)
    gy = grad_output.contiguous(memory_format=torch.channels_last_3d)
    n, _, in_d, in_h, in_w = (int(v) for v in x.shape)
    out_d, out_h, out_w = (int(v) for v in gy.shape[2:])
    k_total = n * out_d * out_h * out_w

    if out is None:
        gw = grad_weight_empty(cout, cin, k, dtype=x.dtype, device=x.device)
    else:
        gw = out
        why = _validate_out(gw, cout, cin, (kd, kh, kw), x.dtype, x.device)
        if why is not None:
            raise ValueError(f"out= is not usable for this problem: {why}")

    if config is None:
        config = bwd_weight_config(cout, cin, k, k_total, x.dtype, padded=padded)
    why = config.validate(x.dtype)
    if why is not None:
        raise ValueError(f"illegal config {config}: {why}")

    splits, chunk = split_count(config, cout, cin, taps, k_total, out_w)
    num_m = triton.cdiv(cout, config.BLOCK_M)
    num_ci = triton.cdiv(cin, config.BLOCK_NC)
    num_tg = triton.cdiv(taps, config.TAP_BLOCK)
    grid = (num_m * num_ci * num_tg * splits,)

    n_elem = cout * taps * cin
    atomic = not deterministic
    if atomic or splits > 1:
        need = n_elem * (1 if atomic else splits)
        if workspace is None:
            ws = torch.empty(need, dtype=torch.float32, device=x.device)
        else:
            if workspace.numel() < need or workspace.dtype is not torch.float32:
                # Say the size, not just that this one is wrong: a caller who
                # hoists the workspace sizes it once, out of the step, and the
                # arithmetic is repeated in the message so that a traceback
                # alone is enough to fix the call.  ``workspace_elements`` with
                # ``split_count``'s own ``splits`` is the supported way to get
                # the number.
                raise ValueError(
                    f"workspace must be at least {need} float32 elements "
                    f"({need * 4 / 2**20:.1f} MiB) -- {splits} splits x "
                    f"{cout} Cout x {taps} taps x {cin} Cin; got "
                    f"{workspace.numel()} of {workspace.dtype} "
                    f"({workspace.numel() * workspace.element_size() / 2**20:.1f}"
                    " MiB)"
                )
            ws = workspace
        if atomic:
            # CK's own shape: an fp32 accumulator every split adds into, zeroed
            # first.  The zeroing and the cast below belong to the atomic path's
            # cost and are timed as such -- excluding them would price
            # determinism against a variant that does not exist.
            ws[:n_elem].zero_()
        dest, stride_ws = ws, (0 if atomic else n_elem)
    else:
        # One split: the kernel writes the answer straight out in its own dtype
        # and neither the workspace nor the reduction pass exists.  Worth the
        # branch -- at the widest channel pairs a round trip through fp32
        # partials moves far more traffic than the kernel itself.
        dest, stride_ws = gw, 0

    big = max(x.numel(), gy.numel()) > 2**31 - 1
    index_dtype = tl.int64 if big else tl.int32

    _conv3d_bwd_weight_kernel[grid](
        x,
        gy,
        dest,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        cin,
        cout,
        k_total,
        chunk,
        grid[0],
        x.stride(0),
        x.stride(2),
        x.stride(3),
        x.stride(4),
        gy.stride(0),
        gy.stride(2),
        gy.stride(3),
        gy.stride(4),
        stride_ws,
        taps * cin,
        NUM_M=num_m,
        NUM_CI=num_ci,
        NUM_TG=num_tg,
        TAPS=taps,
        TAP_BLOCK=config.TAP_BLOCK,
        BLOCK_NC=config.BLOCK_NC,
        KD=kd,
        KH=kh,
        KW=kw,
        SD=sd,
        SH=sh,
        SW=sw,
        PD=pd,
        PH=ph,
        PW=pw,
        DD=dd,
        DH=dh,
        DW=dw,
        BLOCK_M=config.BLOCK_M,
        BLOCK_N=config.BLOCK_N,
        BLOCK_K=config.BLOCK_K,
        EVEN_M=(cout % config.BLOCK_M == 0),
        EVEN_N=(cin % config.BLOCK_NC == 0 and taps % config.TAP_BLOCK == 0),
        EVEN_K=(k_total % config.BLOCK_K == 0 and chunk % config.BLOCK_K == 0),
        PADDED=padded,
        ROW_ALIGNED=_row_aligned(config.BLOCK_K, out_w),
        ATOMIC=atomic,
        NUM_XCD=config.NUM_XCD,
        INDEX_DTYPE=index_dtype,
        INPUT_PRECISION="ieee",
        **config.launch_kwargs(),
    )
    if atomic or splits > 1:
        # A narrower tile when the whole gradient is small, so that the grid is
        # not one program: at ``Cout=6, k=1`` the gradient is 384 elements.
        block = min(1024, max(64, triton.next_power_of_2(n_elem)))
        _reduce_partials_kernel[(triton.cdiv(n_elem, block),)](
            ws,
            gw,
            n_elem,
            1 if atomic else splits,
            BLOCK=block,
            BLOCK_S=8,
            num_warps=4,
        )
    return gw


# ---------------------------------------------------------------------------
# ISA verification
# ---------------------------------------------------------------------------


def verify_isa_bwd_weight(
    problem_shape: Sequence[int] | None = None,
    config: BwdWeightConfig | None = None,
    padding: int = 1,
    kernel: int = 3,
    deterministic: bool = True,
) -> None:  # pragma: no cover
    """Compile and launch one configuration so its ISA can be inspected.

    Run under ``AMDGCN_ENABLE_DUMP=1`` with a cold ``TRITON_CACHE_DIR``; a
    cache hit skips the compile and the empty grep that follows is
    indistinguishable from a kernel with no MFMA in it.  Grep ``v_mfma``, not
    ``v_mfma.*_1k``: the emitted mnemonic has no ``_1k`` suffix.

    ``padding`` defaults to 1, matching :func:`~triton_conv3d.gather_gemm.
    verify_isa`: ``PADDED`` is a ``constexpr``, so it selects a different kernel
    body, and every production ScaFFold convolution with ``k > 1`` compiles the
    padded one.  Inspecting the unpadded body would describe a kernel no
    ScaFFold site launches.
    """
    n, cin, cout, d, h, w = problem_shape or (1, 64, 64, 32, 64, 64)
    k = (kernel, kernel, kernel)
    out = tuple(v + 2 * padding - (kernel - 1) for v in (d, h, w))
    x = torch.randn((n, cin, d, h, w), device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d
    )
    gy = torch.randn((n, cout, *out), device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d
    )
    k_total = n * out[0] * out[1] * out[2]
    cfg = config or bwd_weight_config(
        cout, cin, k, k_total, torch.bfloat16, padded=padding > 0
    )
    splits, chunk = split_count(cfg, cout, cin, kernel**3, k_total, out[2])
    gw = conv3d_backward_weight(
        x, (cout, cin, *k), gy, padding=padding, config=cfg, deterministic=deterministic
    )
    torch.cuda.synchronize()
    print(
        f"ISA-DUMP-CONFIG [bwd-weight] {cfg} cin={cin} cout={cout} "
        f"spatial={(d, h, w)} k={kernel} pad={padding} splits={splits} "
        f"chunk={chunk} det={deterministic} "
        f"row_aligned={_row_aligned(cfg.BLOCK_K, out[2])} "
        f"x_storage={x.untyped_storage().size()} "
        f"gw_storage={gw.untyped_storage().size()}"
    )
