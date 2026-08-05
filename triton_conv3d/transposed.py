# SPDX-License-Identifier: (Apache-2.0)
"""Transposed 3-D convolution at ``kernel == stride``, on NDHWC tensors.

ScaFFold's decoder upsamples with four ``nn.ConvTranspose3d(k=2, s=2, p=0)``
sites.  That case is *much* simpler than a general transposed convolution, and
the whole design here follows from one observation:

    at ``kernel == stride`` and no padding the scatter windows **tile** the
    output rather than overlapping, so every output voxel receives exactly one
    contribution.

Concretely, with ``k = s`` the map ``(d, kd) -> D = d*k + kd`` is a bijection
onto ``[0, k*ID)`` -- it is just base-``k`` positional notation -- so

    y[n, oc, d*KD+kd, h*KH+kh, w*KW+kw] = sum_ic x[n, ic, d, h, w] * w[ic, oc, kd, kh, kw]

with **no sum over taps at all**.  There is no accumulation across windows and
no overlap-add: the operator is a pointwise GEMM from ``Cin`` to ``Cout * taps``
channels, followed by an interleave of those ``taps`` groups into the ``taps``
sub-lattices of the output volume -- a 3-D pixel shuffle.

Three directions, one new kernel
================================

Only the forward needs a kernel.  Both backward directions are the *ordinary*
strided convolution this operator is the transpose of, which this package
already serves:

    let  C(u, w) = conv3d(u, weight=w, stride=k, padding=0)
    with u an NDHWC tensor of Cout channels and w read as (Cin, Cout, kd, kh, kw)
    -- i.e. PyTorch's ConvTranspose3d weight *unpermuted*, whose dim 0 is the
    convolution's output-channel axis and whose dim 1 is its input-channel axis.

    C(u, w)[n, ic, d, h, w'] = sum_{oc, t} u[n, oc, d*k+kd, ...] * w[ic, oc, t]

Comparing that with the display above:

* **backward-data is exactly** ``C(grad_output, w)``.  A strided forward
  convolution, so :func:`~triton_conv3d.gather_gemm.conv3d_forward` serves it
  with no new code and no permute of the parameter -- the transposed operator's
  weight already *is* the shape a ``Cout -> Cin`` convolution wants.
* **backward-weight is exactly** ``C``'s backward-weight, with ``grad_output``
  in the "input" slot and ``x`` in the "grad_output" slot.  Backward-weight has
  no stride restriction (its reduction is indexed by the output voxel), so
  :func:`~triton_conv3d.reduce_gemm.conv3d_backward_weight` serves it unchanged,
  and it produces the gradient in ``channels_last_3d`` -- which for a
  ``(Cin, Cout, k, k, k)`` parameter is the layout ScaFFold's optimizer wants.
* **the forward is** ``C``'s backward-*data*, which
  :mod:`~triton_conv3d.bwd_data` refuses: its kernel-free formulation (the
  forward contraction on a flipped weight) holds only at unit stride, and at
  ``stride > 1`` the gather becomes a scatter into a sub-lattice.  That scatter
  is what :func:`_convT3d_fwd_kernel` below is.

So this module adds one ``@triton.jit`` function and two thin re-expressions.
The package's structural claim becomes: **four operators, three kernels.**

The FLOP count has no per-tap factor
====================================
``2 * in_vol * Cin * Cout * taps`` looks like the general transposed formula and
is not: the ``taps`` here is the *output/input volume ratio*, not a per-tap
gather.  Each output voxel takes ``Cin`` MACs per output channel and there are
``taps * in_vol`` of them.  Applying both factors at once overstates the count
by ``taps`` -- 8x at ``k=2`` -- which this project did once;
``shapes.ConvProblem.flops`` has it right and
``test_shapes.py::test_transposed_flops_have_no_phantom_tap_factor`` pins it.
Check any new arithmetic against :meth:`ConvProblem.gemm_shape`, which reports
``(in_vol, Cout*taps, Cin)`` for the forward -- one ``K = Cin``, no taps in it.

Why the taps go in N and not in M
=================================
The GEMM is ``M = N*ID*IH*IW`` input voxels, ``N = Cout * taps``, ``K = Cin``,
and the only real design question is which axis carries the taps.

Putting them in **M** -- tiling the *output* volume, so each row is one output
voxel and N is plain ``Cout`` -- gives a perfectly coalesced store and a
correctly-once-written output, and then dies: the weight column a row needs
depends on that row's tap, and ``kw`` alternates between adjacent rows along W.
The B operand would have to vary down the M axis, which is not a GEMM.

Putting them in **N** keeps B constant per tile, and the tile's tap group is a
function of the program id alone.  The store is then a scatter -- but a
*structured* one: within one tap the columns are consecutive output channels at
one voxel, i.e. contiguous, and the row-to-row step is ``k`` voxels.  With
``TAP_BLOCK`` covering the ``kw`` pair and ``BLOCK_NC == Cout`` the tile is one
dense run of memory outright.

``TAP_BLOCK`` is why the tap axis has to be in the tile rather than in the grid.
Every tap of a given input voxel reads the *same* A row, so ``taps`` separate
programs would read the input ``taps`` times -- 8x at ``k=2``, against an output
that is only 4x the input at ``128 -> 64``, i.e. more traffic than the answer.
One program spanning ``TAP_BLOCK`` taps loads A once for all of them, in exactly
the way :mod:`~triton_conv3d.reduce_gemm` widens its N across taps and for
exactly the same reason.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import torch
import triton
import triton.language as tl

from .gather_gemm import (
    ConvConfig,
    _LDS_BYTES,
    _MFMA_KDIM,
    _check_out,
    _index_dtype,
    _pow2_at_most,
    _triple,
    conv3d_forward,
    is_supported as _is_supported_fwd,
)
from .reduce_gemm import conv3d_backward_weight, is_supported_bwd_weight

__all__ = [
    "TransposedConfig",
    "conv_transpose3d_forward",
    "conv_transpose3d_backward_data",
    "conv_transpose3d_backward_weight",
    "default_transposed_config",
    "is_supported_transposed",
    "is_supported_transposed_all",
    "is_supported_transposed_bwd_data",
    "is_supported_transposed_bwd_weight",
    "candidate_transposed_configs",
    "grad_transposed_weight_empty",
    "register_tuned_transposed",
    "to_tkn",
    "transposed_config",
    "verify_isa_transposed",
]


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@triton.jit
def _convT3d_fwd_kernel(
    X, W, Y, BIAS,
    # Sizes.  ``M_TOTAL`` is ``BATCH * IN_D * IN_H * IN_W`` -- the *input*
    # volume, because that is what a scatter is indexed by.
    BATCH, IN_D, IN_H, IN_W,
    CIN, COUT, M_TOTAL,
    # Element strides.  The channel stride of X and Y is 1 by construction --
    # that is what NDHWC means -- so it is neither passed nor multiplied by.
    stride_xn, stride_xd, stride_xh, stride_xw,
    # The weight over the effective GEMM's axes: the fused tap index, the
    # reduction axis K (Cin), and the output axis N (Cout).  Which of the two
    # channel strides is 1 is a constexpr (``W_ORDER``), as in the forward.
    stride_wt, stride_wk, stride_wn,
    stride_yn, stride_yd, stride_yh, stride_yw,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_NC: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_K_COUNT: tl.constexpr,
    TAP_BLOCK: tl.constexpr, GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr, EVEN_K: tl.constexpr, EVEN_N: tl.constexpr,
    INDEX_DTYPE: tl.constexpr, INPUT_PRECISION: tl.constexpr,
    W_ORDER: tl.constexpr,
):
    # -- which tile this program owns --------------------------------------
    #
    # The tap group is the *fastest*-varying part of the id, which is a cache
    # decision rather than a cosmetic one: the programs that share an A tile are
    # the ones differing only in tap group, and consecutive ids are dispatched
    # together, so the second read of a row lands while the first is still in
    # L2/MALL.  With the tap group slowest, every tap group would sweep the
    # whole volume before the next one started and each sweep would come from
    # HBM.  Within a tap group the ordinary grouped-M swizzle applies.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M_TOTAL, BLOCK_M)
    grid_nc = tl.cdiv(COUT, BLOCK_NC)
    # Every operand is constexpr, so this is a compile-time constant and the
    # ``%`` / ``//`` below fold into shifts at ``TAP_BLOCK`` a power of two.
    grid_t = (KD * KH * KW) // TAP_BLOCK
    pid_t = pid % grid_t
    pid_mn = pid // grid_t
    width = GROUP_M * grid_nc
    group_id = pid_mn // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_nc = (pid_mn % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_valid = offs_m < M_TOTAL

    # -- unravel the fused ndhw index over the INPUT volume ----------------
    idx_w = offs_m % IN_W
    tmp = offs_m // IN_W
    idx_h = tmp % IN_H
    tmp = tmp // IN_H
    idx_d = tmp % IN_D
    idx_n = tmp // IN_D

    # A's row.  Cast per term rather than after the sum: ``idx_n * stride_xn``
    # alone overflows int32 on a batched scale-8 volume and the sum would then
    # already be wrong before the widening happened.
    x_row = (
        idx_n.to(INDEX_DTYPE) * stride_xn
        + idx_d.to(INDEX_DTYPE) * stride_xd
        + idx_h.to(INDEX_DTYPE) * stride_xh
        + idx_w.to(INDEX_DTYPE) * stride_xw
    )
    # The destination row: the *corner* of this input voxel's output window.
    # The tap's offset within the window is a per-column addend below, so the
    # scatter costs one vector add in the epilogue and nothing in the loop.
    y_row = (
        idx_n.to(INDEX_DTYPE) * stride_yn
        + (idx_d * KD).to(INDEX_DTYPE) * stride_yd
        + (idx_h * KH).to(INDEX_DTYPE) * stride_yh
        + (idx_w * KW).to(INDEX_DTYPE) * stride_yw
    )

    # -- the N axis: TAP_BLOCK taps x BLOCK_NC output channels --------------
    #
    # All hoisted out of the reduction: the column decomposition depends on the
    # tile and not on the reduction index.  ``BLOCK_NC``, ``KH`` and ``KW`` are
    # constexpr, so the divisions fold away.  No tap needs clamping here (unlike
    # ``reduce_gemm``, whose 27 taps cannot be divided by a power of two):
    # ``TAP_BLOCK`` is required to divide ``taps`` exactly, so every column
    # addresses a real tap.
    col = tl.arange(0, BLOCK_N)
    tap = pid_t * TAP_BLOCK + col // BLOCK_NC
    offs_n = pid_nc * BLOCK_NC + (col % BLOCK_NC)
    kd = tap // (KH * KW)
    khw = tap % (KH * KW)
    kh = khw // KW
    kw = khw % KW
    col_ok = offs_n < COUT

    # Where this column lands in the output: the tap's corner offset inside the
    # window, plus the channel.  ``stride_y*`` are element strides of a
    # channels-last tensor, so this is the same expression the forward's
    # ``x_row`` uses, read in the other direction.
    y_col = (
        kd.to(INDEX_DTYPE) * stride_yd
        + kh.to(INDEX_DTYPE) * stride_yh
        + kw.to(INDEX_DTYPE) * stride_yw
        + offs_n.to(INDEX_DTYPE)
    )
    # B's column.  ``W_ORDER == 0`` means Cout is unit-stride, which is what a
    # ``channels_last_3d`` ConvTranspose3d parameter is: its memory order is
    # ``[Cin][kd][kh][kw][Cout]``, i.e. this GEMM's ``[K][tap][N]`` with N dense.
    # That is the *good* case for this direction and it needs no transform,
    # unlike the ordinary forward, where the same parameter layout puts the
    # reduction axis in the contiguous slot.
    if W_ORDER == 0:
        w_col = tap.to(INDEX_DTYPE) * stride_wt + offs_n.to(INDEX_DTYPE)
    else:
        w_col = (
            tap.to(INDEX_DTYPE) * stride_wt
            + offs_n.to(INDEX_DTYPE) * stride_wn
        )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -- the reduction: over Cin alone.  There is no tap loop --------------
    for k0 in range(BLOCK_K_COUNT):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)

        # A: no gather and no boundary predicate.  Every input voxel of an
        # in-range row contributes to every one of its taps, so the six compares
        # the ordinary forward runs per tap do not exist here -- which is the
        # whole reason ``kernel == stride`` is worth a kernel of its own.
        x_ptrs = X + x_row[:, None] + offs_k[None, :]
        if EVEN_K:
            a = tl.load(x_ptrs, mask=m_valid[:, None], other=0.0)
        else:
            a = tl.load(
                x_ptrs, mask=m_valid[:, None] & (offs_k < CIN)[None, :], other=0.0
            )

        w_ptrs = W + (offs_k.to(INDEX_DTYPE) * stride_wk)[:, None] + w_col[None, :]
        if EVEN_K and EVEN_N:
            b = tl.load(w_ptrs)
        elif EVEN_K:
            b = tl.load(w_ptrs, mask=col_ok[None, :], other=0.0)
        elif EVEN_N:
            b = tl.load(w_ptrs, mask=(offs_k < CIN)[:, None], other=0.0)
        else:
            b = tl.load(
                w_ptrs, mask=(offs_k < CIN)[:, None] & col_ok[None, :], other=0.0
            )

        # ``input_precision`` only bites for fp32 operands, where the backend's
        # default splits the dot into reduced-precision pieces.  bf16 already
        # accumulates in fp32 and is unaffected; fp32 is the ``more_determinism``
        # path and has to actually be fp32, so it is asked for explicitly.
        acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)

    if HAS_BIAS:
        # Indexed by the output *channel*, not by the column: the same bias
        # value serves every tap, which is what makes ``ConvTranspose3d``'s bias
        # a per-channel constant over the upsampled volume.
        bias = tl.load(BIAS + offs_n, mask=col_ok, other=0.0)
        acc += bias[None, :].to(tl.float32)

    y_ptrs = Y + y_row[:, None] + y_col[None, :]
    mask_y = tl.broadcast_to(m_valid[:, None], (BLOCK_M, BLOCK_N))
    if not EVEN_N:
        mask_y = mask_y & col_ok[None, :]
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=mask_y)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TransposedConfig(ConvConfig):
    """A launch configuration with the one knob only this direction has.

    A subclass rather than another field on :class:`ConvConfig`, for the reason
    :class:`~triton_conv3d.reduce_gemm.BwdWeightConfig` is one: the gather
    directions have no tap axis in their tile and a config printed in a forward
    sweep should not grow a suffix it cannot use.  ``BLOCK_N`` keeps its meaning
    as the *full* tile width, so the inherited gfx942 legality rules and the
    measured LDS model stay correct unchanged.
    """

    #: How many taps one tile spans.  ``BLOCK_N = TAP_BLOCK * BLOCK_NC``.
    TAP_BLOCK: int = 1

    @property
    def BLOCK_NC(self) -> int:
        """Output channels per tap in the tile."""
        return self.BLOCK_N // self.TAP_BLOCK

    def __str__(self) -> str:
        return super().__str__() + (
            f"/tb{self.TAP_BLOCK}" if self.TAP_BLOCK != 1 else ""
        )

    def validate(self, dtype: torch.dtype) -> str | None:
        why = super().validate(dtype)
        if why is not None:
            return why
        if self.TAP_BLOCK < 1:
            return "TAP_BLOCK must be at least 1"
        if self.BLOCK_N % self.TAP_BLOCK:
            return f"BLOCK_N must be a multiple of TAP_BLOCK={self.TAP_BLOCK}"
        if self.BLOCK_NC < 1:
            return "BLOCK_N // TAP_BLOCK must be at least 1"
        return None


#: Below this many programs the grid cannot fill MI300A's 228 CUs.  The same
#: value the gather kernel uses, restated rather than imported so that a change
#: there is a deliberate change here too -- the two kernels have different
#: occupancies and there is no measurement saying they should track.
_MIN_PROGRAMS = 114


def _fit_transposed(
    cfg: TransposedConfig, m: int, cout: int, taps: int, dtype: torch.dtype
) -> TransposedConfig:
    """Shrink a tile until it fits LDS *and* the grid fills the device.

    Two shrinks, in this order, for the same reasons the gather kernel's
    :func:`~triton_conv3d.gather_gemm._fit_to_lds` and ``_fit_to_grid`` give:
    ``BLOCK_K`` first, because it changes neither the grid nor the parallelism;
    then ``BLOCK_M``.  ``BLOCK_N`` is shrunk last and only down to
    ``TAP_BLOCK * nonkdim``, because halving it below that would drop
    ``BLOCK_NC`` under the MFMA granularity and the tile would be mostly
    padding.

    The grid clause differs from the gather kernel's in one term and it matters:
    this grid has ``taps // TAP_BLOCK`` tap groups in it, so a problem whose M
    and N alone look too small for 228 CUs may already fill them.  Leaving that
    factor out would halve ``BLOCK_M`` at every decoder site and lose the reuse
    for nothing.
    """
    nk = cfg.matrix_instr_nonkdim
    kdim = _MFMA_KDIM.get(dtype, {}).get(nk)
    if kdim is None:
        return cfg
    while cfg.lds_bytes(dtype) > _LDS_BYTES:
        half_k, half_m, half_n = cfg.BLOCK_K // 2, cfg.BLOCK_M // 2, cfg.BLOCK_N // 2
        if half_k >= kdim and half_k % kdim == 0:
            cfg = dataclasses.replace(cfg, BLOCK_K=half_k,
                                      kpack=1 if half_k <= 16 else cfg.kpack)
        elif half_m >= nk and half_m % nk == 0:
            cfg = dataclasses.replace(cfg, BLOCK_M=half_m)
        elif half_n >= cfg.TAP_BLOCK * nk and half_n % nk == 0:
            cfg = dataclasses.replace(cfg, BLOCK_N=half_n)
        else:
            break  # nothing left to shrink; let the launch say so
    while (
        cfg.BLOCK_M > max(16, nk)
        and (cfg.BLOCK_M // 2) % nk == 0
        and (-(-m // cfg.BLOCK_M) * -(-cout // cfg.BLOCK_NC)
             * (taps // cfg.TAP_BLOCK)) < _MIN_PROGRAMS
    ):
        cfg = dataclasses.replace(cfg, BLOCK_M=cfg.BLOCK_M // 2)
    warps = max(1, min(cfg.num_warps, cfg.BLOCK_M * cfg.BLOCK_N // 256))
    return dataclasses.replace(cfg, num_warps=1 << (warps.bit_length() - 1))


def _largest_pow2_divisor(n: int, cap: int) -> int:
    """The largest power of two that divides ``n`` and is at most ``cap``."""
    d = 1
    while d * 2 <= cap and n % (d * 2) == 0:
        d *= 2
    return d


def default_transposed_config(
    m: int, cin: int, cout: int, taps: int, dtype: torch.dtype = torch.bfloat16
) -> TransposedConfig:
    """A config that is legal for any shape this module accepts.

    ``TAP_BLOCK`` is the only choice here that is not the gather kernel's, and
    it is chosen to cut the A traffic rather than to fill a tile: every tap of
    an input voxel reads the same row, so a program spanning ``TAP_BLOCK`` taps
    reads the input ``taps / TAP_BLOCK`` times instead of ``taps`` times.  It is
    capped so the tile stays 256 columns wide -- past that the accumulator alone
    is 128 registers per lane at four warps and occupancy collapses.
    """
    block_nc = _pow2_at_most(cout, 128)
    tap_block = _largest_pow2_divisor(taps, max(1, 256 // block_nc))
    block_k = 128 if cin >= 512 else _pow2_at_most(cin, 64)
    block_m = _pow2_at_most(m, 128)
    nonkdim = 16
    kdim = _MFMA_KDIM[dtype][nonkdim]
    block_k = max(kdim, block_k - block_k % kdim)
    block_n = tap_block * block_nc
    return _fit_transposed(
        TransposedConfig(
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=6,
            num_warps=8 if block_n >= 256 or block_k >= 128 else 4,
            num_stages=2, matrix_instr_nonkdim=nonkdim,
            kpack=1 if block_k <= 16 else 2, TAP_BLOCK=tap_block,
        ),
        m, cout, taps, dtype,
    )


def _tuned(bm: int, bnc: int, tb: int, bk: int, warps: int,
           group_m: int = 6) -> TransposedConfig:
    return TransposedConfig(
        BLOCK_M=bm, BLOCK_N=tb * bnc, BLOCK_K=bk, GROUP_M=group_m,
        num_warps=warps, num_stages=2, matrix_instr_nonkdim=16,
        kpack=1 if bk <= 16 else 2, TAP_BLOCK=tb,
    )


def transposed_tune_key(dtype: torch.dtype, cin: int, cout: int,
                        kernel: tuple[int, ...]) -> tuple:
    return (str(dtype), cin, cout, tuple(kernel))


#: Measured winners for the transposed forward, keyed by ``(dtype, Cin, Cout,
#: kernel)``.  Source: ``work/triton-conv/m5_fwd.json`` -- the four
#: ``ConvTranspose3d`` channel pairs the model contains, swept over the tile and
#: ``TAP_BLOCK`` grid of :func:`candidate_transposed_configs` and then raced
#: against MIOpen.  A miss falls back to :func:`default_transposed_config`.
#:
#: Keyed on the channel widths and not the volume, as the gather kernel's table
#: is -- and with the same caveat, which this project has now paid for twice: a
#: *speedup ratio* does not transfer across volume even when the winning tile
#: does.  So only the *tile* is claimed to transfer, and only where it was
#: measured winning at every volume the pair occurs at.  Each pair below occurs
#: at three volumes (one per profiled configuration) and the entry named won all
#: three; the speedups they produce differ by up to 2.3x between those volumes,
#: are recorded per volume in ``work/triton-conv/m5_shipped_*.json``, and are
#: never averaged.
#:
#: **Two of the four pairs are deliberately absent.**  ``512 -> 256`` and
#: ``1024 -> 512`` were swept just as thoroughly and
#: :func:`default_transposed_config` picked the winner or a tie at every volume
#: (within 0.4-8%, and the sweep's nominal best flipped tile between volumes),
#: so an entry would restate the heuristic while claiming to have improved on
#: it.  An absent row here means "measured, and the heuristic was right", which
#: is a different statement from "never measured" -- the gather kernel's table
#: had to learn that distinction the hard way.
_TUNED_T: dict[tuple, TransposedConfig] = {
    transposed_tune_key(torch.bfloat16, cin, cout, (2, 2, 2)): cfg
    for (cin, cout), cfg in {
        # Both winners are ``BLOCK_NC = 64`` with ``TAP_BLOCK = 4``, i.e. a
        # 256-column tile spanning half the taps, against the heuristic's
        # ``BLOCK_NC = Cout, TAP_BLOCK = 2``.  Same column count, twice the tap
        # reuse: the input is read twice instead of four times, which is what
        # this operator is short of at these channel widths.  Worth 1.12-1.23x
        # over the heuristic and it is the whole gap between them.
        (128, 64): _tuned(256, 64, 4, 64, 8),
        (256, 128): _tuned(128, 64, 4, 64, 8),
    }.items()
}


def register_tuned_transposed(dtype, cin, cout, kernel, config) -> None:
    _TUNED_T[transposed_tune_key(dtype, cin, cout, kernel)] = config


def transposed_config(
    m: int, cin: int, cout: int, kernel: Sequence[int],
    dtype: torch.dtype = torch.bfloat16,
) -> TransposedConfig:
    """The config :func:`conv_transpose3d_forward` would pick for this problem."""
    k = _triple(kernel, "kernel")
    taps = k[0] * k[1] * k[2]
    tuned = _TUNED_T.get(transposed_tune_key(dtype, cin, cout, tuple(k)))
    if tuned is not None:
        return _fit_transposed(tuned, m, cout, taps, dtype)
    return default_transposed_config(m, cin, cout, taps, dtype)


#: Seed tiles for a sweep, ``(BLOCK_M, BLOCK_NC, BLOCK_K, num_warps)``.  Narrower
#: than the gather kernel's grid because this GEMM's K is ``Cin`` alone -- there
#: is no tap factor in it -- so a ``BLOCK_K`` above ``Cin`` is pure padding, and
#: because ``TAP_BLOCK`` multiplies the column count on top of ``BLOCK_NC``.
_SEED_TILES: tuple[tuple[int, int, int, int], ...] = (
    (64, 64, 32, 4),
    (64, 64, 64, 4),
    (128, 64, 64, 4),
    (256, 64, 64, 8),
    (64, 128, 64, 4),
    (128, 128, 64, 8),
    (64, 64, 128, 4),
    (128, 64, 128, 8),
    (64, 128, 128, 8),
    (32, 64, 64, 4),
    (32, 128, 64, 4),
)


def candidate_transposed_configs(
    m: int, cin: int, cout: int, taps: int,
    dtype: torch.dtype = torch.bfloat16,
    *, tap_blocks: Sequence[int] = (1, 2, 4, 8), nonkdims: Sequence[int] = (16, 32),
) -> list[TransposedConfig]:
    """Configs worth timing for one transposed problem, pruned to legal ones."""
    n2 = max(16, triton.next_power_of_2(cout))
    k2 = max(16, triton.next_power_of_2(cin))
    m2 = max(16, triton.next_power_of_2(m))
    out: list[TransposedConfig] = []
    seen: set[TransposedConfig] = set()
    for bm, bnc, bk, seed_warps in _SEED_TILES:
        if bm > 2 * m2 or bnc > 2 * n2 or bk > k2:
            continue
        for tb in tap_blocks:
            if taps % tb:
                continue
            for warps in {4, 8, seed_warps}:
                for nonkdim in nonkdims:
                    cfg = TransposedConfig(
                        BLOCK_M=bm, BLOCK_N=tb * bnc, BLOCK_K=bk, GROUP_M=6,
                        num_warps=warps, num_stages=2,
                        matrix_instr_nonkdim=nonkdim,
                        kpack=1 if bk <= 16 else 2, TAP_BLOCK=tb,
                    )
                    if (cfg.validate(dtype) is not None
                            or cfg.lds_bytes(dtype) > _LDS_BYTES
                            or cfg in seen):
                        continue
                    seen.add(cfg)
                    out.append(cfg)
    if not out:
        out.append(default_transposed_config(m, cin, cout, taps, dtype))
    return out


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


_W_N_CONTIG = 0
_W_GENERAL = 1


def to_tkn(w: torch.Tensor) -> torch.Tensor:
    """A ``(Cin, Cout, kd, kh, kw)`` transposed weight as ``(kd, kh, kw, Cin, Cout)``.

    The B tile wants ``[tap][K=Cin][N=Cout]`` with N dense, which is what this
    produces.  **It is off the shipped path**: a ``channels_last_3d`` parameter
    -- which is what ``worker.py`` makes every 5-D parameter -- already has
    ``Cout`` unit-stride and the three kernel axes fused, so
    :func:`_transposed_weight_plan` addresses it in place and this copy never
    runs.  Note that this is the opposite of the ordinary forward's situation,
    where the same layout puts the *reduction* axis in the dense slot and the
    tile has to be gathered.

    Kept for the layouts the plan refuses -- chiefly PyTorch's default, where
    neither channel axis is unit-stride and every element of the B tile is its
    own cache line.
    """
    return w.permute(2, 3, 4, 0, 1).contiguous()


def _transposed_weight_plan(w: torch.Tensor) -> tuple[int, int, int, int] | None:
    """``(W_ORDER, stride_wt, stride_wk, stride_wn)`` for ``w``, or ``None``.

    ``w`` is the weight as PyTorch stores it for ``ConvTranspose3d``:
    ``(Cin, Cout, kd, kh, kw)``, i.e. dim 0 is this GEMM's reduction axis and
    dim 1 is its N.  That is the transpose of the ordinary convolution's
    convention, which is why this cannot simply call
    :func:`~triton_conv3d.gather_gemm._weight_plan`; everything else about it is
    the same computation, including why it is a stride test rather than a
    ``is_contiguous(memory_format=...)`` one.

    ``None`` means materialize :func:`to_tkn` instead, for one of two reasons:
    the three kernel axes are not one fused axis of constant stride, which is
    what the kernel's single ``tap * stride_wt`` assumes; or neither channel
    axis is unit-stride, which is correctness-neutral and a large performance
    cliff, since every element of the B tile is then its own cache line.

    Extents of 1 carry no observable stride, so they constrain nothing and are
    skipped -- the same reason the gather kernel's plan skips them.
    """
    cin, cout, kd, kh, kw = (int(v) for v in w.shape)
    s = tuple(int(v) for v in w.stride())
    if kw > 1:
        st = s[4]
    elif kh > 1:
        st = s[3]
    elif kd > 1:
        st = s[2]
    else:
        st = 0  # one tap: ``tap`` is always 0, so any stride is the right one
    if ((kw > 1 and s[4] != st)
            or (kh > 1 and s[3] != st * kw)
            or (kd > 1 and s[2] != st * kw * kh)):
        return None
    if cout == 1 or s[1] == 1:
        return (_W_N_CONTIG, st, s[0], 1)
    if cin == 1 or s[0] == 1:
        return (_W_GENERAL, st, s[0], s[1])
    return None


def _transposed_out_spatial(
    in_spatial: Sequence[int], kernel: tuple[int, int, int]
) -> tuple[int, int, int]:
    """``k * extent`` per axis -- the only output shape this module produces.

    Spelled from ``kernel`` rather than from PyTorch's general
    ``(i-1)*s - 2p + d*(k-1) + 1 + output_padding`` because the gate has already
    pinned ``s == k``, ``p == 0``, ``d == 1`` and ``output_padding == 0``, at
    which point that formula collapses to exactly this.  Writing the general one
    here would suggest the module served the general case.
    """
    return tuple(int(i) * k for i, k in zip(in_spatial, kernel))  # type: ignore[return-value]


def _transposed_shape_ok(
    x: torch.Tensor, w: torch.Tensor, stride, padding, output_padding, dilation,
    groups: int,
) -> tuple[int, int, int] | None:
    """The kernel triple if this is a ``kernel == stride`` upsample, else ``None``.

    Total, like the gates that call it: an argument it cannot interpret is a
    ``None`` and never an exception, because these are the predicates of a
    Triton -> MIOpen rung ladder and a caller asking a question must not be
    taken down by the answer.
    """
    if groups != 1:
        return None
    if x.dim() != 5 or w.dim() != 5:
        return None
    try:
        s = _triple(stride, "stride")
        p = _triple(padding, "padding")
        op = _triple(output_padding, "output_padding")
        d = _triple(dilation, "dilation")
    except (ValueError, TypeError):
        return None
    k = tuple(int(v) for v in w.shape[2:])
    # The whole of this module's mathematics: windows that tile rather than
    # overlap.  ``k != s`` overlaps (or leaves gaps), a padding crops the
    # result, an ``output_padding`` extends it asymmetrically and a dilation
    # interleaves the window with holes -- each one breaks the bijection
    # ``(d, kd) -> d*k + kd`` that makes every output voxel a single
    # contribution, and none of them occurs in ScaFFold.
    if s != k or p != (0, 0, 0) or op != (0, 0, 0) or d != (1, 1, 1):
        return None
    if any(v < 1 for v in k):
        return None
    # Degenerate extents, refused for the same reason ``is_supported`` refuses
    # them: each clears every other test here and then disagrees with torch.  A
    # zero-length spatial axis produces an empty output where the M-unravel has
    # no rows to index; ``Cin = 0`` returns ``Cout`` channels of zeros where
    # torch returns a tensor with no channels at all.
    if any(int(v) < 1 for v in x.shape[2:]):
        return None
    if int(w.shape[0]) < 1 or int(w.shape[1]) < 1:
        return None
    return k


def is_supported_transposed(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups: int = 1,
) -> bool:
    """Whether :func:`conv_transpose3d_forward` will serve this call.

    Deliberately conservative and **total**, for the reasons
    :func:`~triton_conv3d.gather_gemm.is_supported` gives: the caller's fallback
    is MIOpen, which is correct everywhere, and an argument this cannot
    interpret has to be a ``False`` rather than an exception.

    ``w`` is PyTorch's ``ConvTranspose3d`` weight, ``(Cin, Cout, kd, kh, kw)``
    -- the channel axes the other way round from ``nn.Conv3d``'s.  That is not a
    detail: passing a ``Conv3d`` weight here would be accepted whenever the two
    channel counts happen to match and would compute a transposed answer.

    **This gates the forward alone.**  Unlike the ordinary convolution, all three
    of this operator's directions accept exactly the same problems -- both
    backward directions are the *same* ``k == s`` convolution seen from the
    other side -- so :func:`is_supported_transposed_all` should agree with this
    on every input.  It exists anyway, and asks all three for real, because
    "should agree" is an argument and the ladder needs a fact: the three gates
    of the ordinary convolution were also expected to agree until ``stride > 1``
    showed that they do not.
    """
    k = _transposed_shape_ok(x, w, stride, padding, output_padding, dilation,
                             groups)
    if k is None:
        return False
    if x.dtype != w.dtype or x.dtype not in _MFMA_KDIM:
        return False
    # Same device, not merely both on *a* device.  Triton launches on the current
    # device and dereferences the other pointer anyway; ScaFFold runs four GPUs
    # per node, where peer access turns that into another rank's data rather
    # than a fault.
    if not x.is_cuda or not w.is_cuda or w.device != x.device:
        return False
    if int(x.shape[1]) != int(w.shape[0]):
        return False
    if bias is not None:
        # The kernel masks the bias load against ``Cout`` -- which says nothing
        # about how long the bias actually is -- and indexes it with an element
        # stride of 1.  A short bias reads past the end and a stride-2 view of
        # the right length silently applies every other value.
        # ``torch.conv_transpose3d`` rejects both; so does this.  ``Cout`` is
        # ``w.shape[1]`` here, not ``w.shape[0]``.
        if (bias.dim() != 1 or int(bias.shape[0]) != int(w.shape[1])
                or bias.dtype != x.dtype or not bias.is_cuda
                or bias.device != x.device or bias.stride(0) != 1):
            return False
    return True


def is_supported_transposed_bwd_data(
    grad_output: torch.Tensor,
    w: torch.Tensor,
    input_shape: Sequence[int],
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups: int = 1,
) -> bool:
    """Whether :func:`conv_transpose3d_backward_data` will serve this call.

    Asks the *ordinary* forward's gate about the strided convolution this
    direction actually is -- ``conv3d(grad_output, w, stride=k)`` -- rather than
    re-deriving a predicate, so the two can never drift apart.  The extra checks
    on top are the ones that gate cannot see: that the problem is a
    ``kernel == stride`` upsample at all, and that ``input_shape`` is the shape
    this ``grad_output`` came from.
    """
    k = _transposed_shape_ok(grad_output, w, stride, padding, output_padding,
                             dilation, groups)
    if k is None:
        return False
    try:
        shape = tuple(int(v) for v in input_shape)
    except (TypeError, ValueError):
        return False
    if len(shape) != 5:
        return False
    n, cin, *in_sp = shape
    if n < 1 or any(v < 1 for v in in_sp):
        return False
    if int(w.shape[0]) != cin or int(grad_output.shape[1]) != int(w.shape[1]):
        return False
    if int(grad_output.shape[0]) != n:
        return False
    if tuple(int(v) for v in grad_output.shape[2:]) != _transposed_out_spatial(
        in_sp, k
    ):
        return False
    # The effective convolution, asked of the gate that will actually serve it.
    return bool(_is_supported_fwd(grad_output, w, None, k, 0, 1, 1))


def is_supported_transposed_bwd_weight(
    x: torch.Tensor,
    weight_shape: Sequence[int],
    grad_output: torch.Tensor,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups: int = 1,
) -> bool:
    """Whether :func:`conv_transpose3d_backward_weight` will serve this call.

    As with backward-data, this asks the incumbent gate about the problem that
    will really run -- ``conv3d_backward_weight`` on the strided convolution,
    with ``grad_output`` in the input slot and ``x`` in the gradient slot -- and
    adds only what that gate cannot see.
    """
    try:
        ws = tuple(int(v) for v in weight_shape)
    except (TypeError, ValueError):
        return False
    if len(ws) != 5 or groups != 1:
        return False
    if x.dim() != 5 or grad_output.dim() != 5:
        return False
    try:
        s = _triple(stride, "stride")
        p = _triple(padding, "padding")
        op = _triple(output_padding, "output_padding")
        d = _triple(dilation, "dilation")
    except (ValueError, TypeError):
        return False
    k = ws[2:]
    if s != k or p != (0, 0, 0) or op != (0, 0, 0) or d != (1, 1, 1):
        return False
    if any(v < 1 for v in k) or ws[0] < 1 or ws[1] < 1:
        return False
    if int(x.shape[1]) != ws[0] or int(grad_output.shape[1]) != ws[1]:
        return False
    if any(int(v) < 1 for v in x.shape[2:]):
        return False
    return bool(
        is_supported_bwd_weight(grad_output, (ws[0], ws[1], *k), x, k, 0, 1, 1)
    )


def is_supported_transposed_all(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups: int = 1,
) -> bool:
    """Whether **every** direction of this transposed convolution will be served.

    The gate for a caller that is going to differentiate, and the counterpart of
    :func:`~triton_conv3d.gather_gemm.is_supported_all`.  A forward this package
    serves and a backward it cannot is discovered inside ``backward()``, where
    the caller's fallback kernel is no longer reachable -- so a training caller
    must ask this one.

    The gradient is passed as a **metadata-only stand-in**: all three predicates
    read rank, shape, dtype, device and ``is_cuda`` and never a stride, a value
    or a contiguity, so a one-element allocation expanded to the output shape
    answers exactly as the real gradient would.  ``expand`` gives every dim a
    stride of 0, so if a predicate ever grows a stride test it will see those
    zeros and answer ``False`` -- a fallback to the caller's other kernel, which
    is the safe direction.
    """
    if not is_supported_transposed(x, w, bias, stride, padding, output_padding,
                                   dilation, groups):
        return False
    k = tuple(int(v) for v in w.shape[2:])
    grad_shape = (int(x.shape[0]), int(w.shape[1])) + _transposed_out_spatial(
        tuple(int(v) for v in x.shape[2:]), k
    )
    grad = x.new_empty((1, 1, 1, 1, 1)).expand(grad_shape)
    if not is_supported_transposed_bwd_data(
        grad, w, tuple(x.shape), stride, padding, output_padding, dilation, groups
    ):
        return False
    return bool(
        is_supported_transposed_bwd_weight(
            x, tuple(w.shape), grad, stride, padding, output_padding, dilation,
            groups,
        )
    )


def conv_transpose3d_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups: int = 1,
    *,
    config: TransposedConfig | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transposed 3-D convolution at ``kernel == stride``.  NDHWC in and out.

    ``w`` is PyTorch's ``ConvTranspose3d`` weight, ``(Cin, Cout, kd, kh, kw)``,
    and is read where it lies: a ``channels_last_3d`` parameter has ``Cout``
    unit-stride, which is this GEMM's N, so no transform runs.  A weight in
    PyTorch's *default* layout is copied and has to be; see
    :func:`_transposed_weight_plan`.

    ``out=`` is checked rather than trusted, for the reason
    :func:`~triton_conv3d.gather_gemm._check_out` gives: the store addressing is
    derived from *this* call's shapes, so a mismatched buffer is an
    out-of-bounds device write with no error and an NCDHW one is a full-rate
    kernel returning a scrambled answer.
    """
    if not is_supported_transposed(x, w, bias, stride, padding, output_padding,
                                   dilation, groups):
        raise NotImplementedError(
            f"unsupported: x={tuple(x.shape)}/{x.dtype} w={tuple(w.shape)} "
            f"stride={stride} padding={padding} output_padding={output_padding} "
            f"dilation={dilation} groups={groups}"
        )
    kd, kh, kw = (int(v) for v in w.shape[2:])
    taps = kd * kh * kw

    # NDHWC is not a preference here, it is the layout the addressing assumes.
    x = x.contiguous(memory_format=torch.channels_last_3d)
    n, cin, in_d, in_h, in_w = (int(v) for v in x.shape)
    cout = int(w.shape[1])
    out_d, out_h, out_w = _transposed_out_spatial((in_d, in_h, in_w),
                                                  (kd, kh, kw))

    y_shape = (n, cout, out_d, out_h, out_w)
    if out is None:
        # One allocation, already in the layout the kernel stores into.  Spelling
        # it ``torch.empty(shape).contiguous(memory_format=...)`` allocates NCDHW
        # and then copies the whole thing -- 235x, measured on the gather
        # kernel's identically-shaped defect.
        y = torch.empty(y_shape, device=x.device, dtype=x.dtype,
                        memory_format=torch.channels_last_3d)
    else:
        y = out
        _check_out(y, y_shape, x)

    plan = _transposed_weight_plan(w)
    if plan is None:
        # The only path that copies the weight; see the docstring of
        # :func:`to_tkn`.  Contiguous ``(kd, kh, kw, Cin, Cout)``, so the three
        # strides are exactly these.
        wt = to_tkn(w)
        plan = (_W_N_CONTIG, cin * cout, cout, 1)
    else:
        wt = w

    m_total = n * in_d * in_h * in_w
    if config is None:
        config = transposed_config(m_total, cin, cout, (kd, kh, kw), x.dtype)
    why = config.validate(x.dtype)
    if why is not None:
        raise ValueError(f"illegal config {config}: {why}")
    if taps % config.TAP_BLOCK:
        raise ValueError(
            f"illegal config {config}: TAP_BLOCK must divide the tap count "
            f"{taps}; a ragged last group would address a tap that is not there"
        )

    index_dtype = _index_dtype(x, y, wt)
    grid = (
        triton.cdiv(m_total, config.BLOCK_M)
        * triton.cdiv(cout, config.BLOCK_NC)
        * (taps // config.TAP_BLOCK),
    )
    _convT3d_fwd_kernel[grid](
        x, wt, y, bias,
        n, in_d, in_h, in_w,
        cin, cout, m_total,
        x.stride(0), x.stride(2), x.stride(3), x.stride(4),
        plan[1], plan[2], plan[3],
        y.stride(0), y.stride(2), y.stride(3), y.stride(4),
        KD=kd, KH=kh, KW=kw,
        BLOCK_M=config.BLOCK_M, BLOCK_N=config.BLOCK_N,
        BLOCK_NC=config.BLOCK_NC, BLOCK_K=config.BLOCK_K,
        BLOCK_K_COUNT=triton.cdiv(cin, config.BLOCK_K),
        TAP_BLOCK=config.TAP_BLOCK, GROUP_M=config.GROUP_M,
        HAS_BIAS=bias is not None,
        EVEN_K=(cin % config.BLOCK_K == 0),
        EVEN_N=(cout % config.BLOCK_NC == 0),
        INDEX_DTYPE=index_dtype,
        INPUT_PRECISION="ieee",
        W_ORDER=plan[0],
        **config.launch_kwargs(),
    )
    return y


def conv_transpose3d_backward_data(
    grad_output: torch.Tensor,
    w: torch.Tensor,
    input_shape: Sequence[int],
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups: int = 1,
    *,
    config: ConvConfig | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gradient of a ``k == s`` transposed convolution with respect to its input.

    **This is an ordinary strided forward convolution**, and the module
    docstring derives it: ``grad_input = conv3d(grad_output, w, stride=k)``,
    with ``w`` passed *unpermuted*.  PyTorch stores a ``ConvTranspose3d`` weight
    as ``(Cin, Cout, k, k, k)``, and that already is the ``(out_channels,
    in_channels, k, k, k)`` an ordinary ``Cout -> Cin`` convolution wants -- the
    transpose is in the storage convention, so it costs nothing here.

    No bias term: the bias is added to the forward's output, so its gradient is
    a reduction of ``grad_output`` and not part of this direction at all.
    """
    if not is_supported_transposed_bwd_data(
        grad_output, w, input_shape, stride, padding, output_padding, dilation,
        groups
    ):
        raise NotImplementedError(
            f"unsupported: grad_output={tuple(grad_output.shape)}/"
            f"{grad_output.dtype} w={tuple(w.shape)} "
            f"input_shape={tuple(input_shape)} stride={stride} "
            f"padding={padding} output_padding={output_padding} "
            f"dilation={dilation} groups={groups}"
        )
    k = tuple(int(v) for v in w.shape[2:])
    return conv3d_forward(grad_output, w, None, k, 0, 1, 1, config=config,
                          out=out)


def conv_transpose3d_backward_weight(
    x: torch.Tensor,
    weight_shape: Sequence[int],
    grad_output: torch.Tensor,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups: int = 1,
    *,
    config=None,
    workspace: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """Gradient of a ``k == s`` transposed convolution with respect to its weight.

    The *same* reduction ``conv3d_backward_weight`` already performs, with the
    two activations in the slots the strided convolution of the module docstring
    puts them in: ``grad_output`` is that convolution's input and ``x`` is its
    output gradient.  Reading the call and expecting ``x`` first is the one way
    to misuse this function, which is why the argument order still matches
    ``conv3d_backward_weight``'s -- the swap happens inside, once, here.

    The returned gradient is ``(Cin, Cout, k, k, k)`` in ``channels_last_3d``,
    which is both the parameter's own shape and the layout ``worker.py`` puts it
    in, so the optimizer's elementwise update is contiguous.
    """
    if not is_supported_transposed_bwd_weight(
        x, weight_shape, grad_output, stride, padding, output_padding, dilation,
        groups
    ):
        raise NotImplementedError(
            f"unsupported: x={tuple(x.shape)}/{x.dtype} "
            f"weight_shape={tuple(weight_shape)} "
            f"grad_output={tuple(grad_output.shape)} stride={stride} "
            f"padding={padding} output_padding={output_padding} "
            f"dilation={dilation} groups={groups}"
        )
    ws = tuple(int(v) for v in weight_shape)
    k = ws[2:]
    return conv3d_backward_weight(
        grad_output, ws, x, k, 0, 1, 1,
        config=config, workspace=workspace, out=out,
        deterministic=deterministic,
    )


def grad_transposed_weight_empty(
    cin: int, cout: int, kernel: Sequence[int], *, dtype, device
) -> torch.Tensor:
    """An empty transposed-weight gradient in the layout the kernel writes.

    ``(Cin, Cout, kd, kh, kw)`` in ``channels_last_3d``, i.e. memory order
    ``Cin, kd, kh, kw, Cout``.  The ``Cin``/``Cout`` order is the only thing
    that differs from
    :func:`~triton_conv3d.reduce_gemm.grad_weight_empty`, and it differs because
    a ``ConvTranspose3d`` parameter is stored the other way round; passing the
    ordinary one here allocates a correctly-strided buffer of the wrong shape,
    which ``conv3d_backward_weight``'s ``out=`` check catches.
    """
    k = _triple(kernel, "kernel")
    return torch.empty((cin, cout, *k), dtype=dtype, device=device,
                       memory_format=torch.channels_last_3d)


# ---------------------------------------------------------------------------
# ISA verification
# ---------------------------------------------------------------------------


def verify_isa_transposed(
    problem_shape: Sequence[int] | None = None,
    config: "TransposedConfig | None" = None,
    kernel: int = 2,
    weight_layout: str = "channels_last",
) -> None:  # pragma: no cover
    """Compile and launch one configuration so its ISA can be inspected.

    Run under ``AMDGCN_ENABLE_DUMP=1`` with a **cold** ``TRITON_CACHE_DIR``: a
    cache hit skips the compile and therefore the dump, and an empty grep then
    looks exactly like a kernel with no MFMA in it.  The emitted mnemonic is
    ``v_mfma_f32_16x16x16_bf16`` with **no** ``_1k`` suffix, despite Triton's
    internal table entry being named ``_1k``.

    ``weight_layout`` selects which of the two B loads is compiled, for the same
    reason the gather kernel's does: ``W_ORDER`` is a constexpr and the two
    orders emit different instructions for the operand that feeds the matrix
    core.
    """
    n, cin, cout, d, h, wd = problem_shape or (1, 128, 64, 64, 64, 64)
    k = (kernel, kernel, kernel)
    w = torch.randn((cin, cout, *k), device="cuda", dtype=torch.bfloat16)
    if weight_layout == "channels_last":
        w = w.contiguous(memory_format=torch.channels_last_3d)
    elif weight_layout != "tkn":
        raise ValueError(f"unknown weight_layout {weight_layout!r}")
    x = torch.randn((n, cin, d, h, wd), device="cuda",
                    dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d)
    cfg = config or transposed_config(n * d * h * wd, cin, cout, k,
                                      torch.bfloat16)
    y = conv_transpose3d_forward(x, w, None, k, config=cfg)
    torch.cuda.synchronize()
    print(
        f"ISA-DUMP-CONFIG [convT/{weight_layout}] {cfg} cin={cin} cout={cout} "
        f"spatial={(d, h, wd)} k={kernel} "
        f"x_storage={x.untyped_storage().size()} "
        f"y_storage={y.untyped_storage().size()}"
    )
