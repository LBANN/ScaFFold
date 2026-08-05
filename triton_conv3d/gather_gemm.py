# SPDX-License-Identifier: (Apache-2.0)
"""Forward 3-D convolution as a fused implicit GEMM over NDHWC tensors.

The convolution is evaluated as a single GEMM whose ``A`` operand is gathered
rather than materialized::

    M = N * OD * OH * OW      output voxels, a flat linear index
    N = Cout                  output channels
    K = kd * kh * kw * Cin    taps x input channels

Nothing is written to memory between the gather and the matrix core: for each
tap the kernel re-reads the input at a *constant* voxel shift, which is what
makes NDHWC the right layout.  ``Cin`` is the fastest-varying axis of the input
and is also the GEMM's reduction axis, so a K-tile is a contiguous vector load,
and moving from one tap to the next is a scalar addend on the row offset rather
than per-element index arithmetic.

Provenance
==========

The tiling is PyTorch Inductor's ``conv3d_template``
(``torch/_inductor/kernel/conv.py``), not a fresh derivation.  Three things are
taken from it unchanged because they are already right:

* the M-unravel of a fused ``ndhw`` linear index by successive ``%`` / ``//``;
* the fused ``dijk`` reduction loop with **channel blocks innermost and taps
  outermost**, which keeps the contiguous ``C`` axis fast-varying so the loads
  vectorize (Inductor's own comment records that the nested-loop form is
  slightly slower);
* halo handling as pure predication -- no shared-memory staging, no im2col.

What is *not* taken from it is the address arithmetic.  The template is written
against NCDHW; every offset here is re-derived for NDHWC.  What survives that
re-derivation, deliberately, is the pointer *shape*: every global access is
``splat(scalar_base) + offset_tensor``.  That is condition 1 of the AMD backend's
``canUseBufferOps``, and a tensor-of-pointers formulation loses buffer-op
lowering outright -- the documented reason naive Triton convolutions are slow.

The weight is read where it lies
================================

There is no weight transform on the shipped path, in any direction, and the
reason is worth stating because the obvious design has one.  The B tile wants
``(BLOCK_K, BLOCK_N) = (Cin, Cout)`` per tap, PyTorch stores neither channel
axis in that position, and the natural fix -- materialize
``(kd, kh, kw, Cin, Cout)`` once and reuse it -- costs **0.786 ms/step** for the
forward and 0.531 for backward-data across one configuration's 19 Conv3d sites.
Not per call: per *optimizer step*, because the optimizer dirties every
parameter every step, so no cache removes it.

So the kernel addresses the weight through its strides instead, and which axis
is unit-stride selects the load (``W_ORDER``):

===========================  ==================  ====================  ==========
weight layout                forward             backward-data         copies
===========================  ==================  ====================  ==========
``channels_last_3d``         gathered columns    contiguous rows       **no**
RSCK-strided or ``rsck=``    contiguous rows     gathered columns      **no**
PyTorch default              --                  --                    yes
===========================  ==================  ====================  ==========

A ScaFFold model is entirely in the first row: ``worker.py`` moves it to
``channels_last_3d`` at construction, which makes every conv weight
``[Cout][kd][kh][kw][Cin]``.  The two rows that copy nothing are within 0.8% of
each other over the eight hottest config-A sites, so the choice between them is
not a performance question; the third is 4.8-9.0x slower if addressed in place,
because neither tile axis is dense, and is therefore copied.

Two designs that look better and measure worse, both raced per site: loading the
tile coalesced and transposing it in registers (1.05-1.55x, the transpose is the
cost); and holding the
parameter in RSCK order, which is 0.159 ms/step better in the kernels and
**2.6 ms/step worse in the optimizer**, since the gradient this package produces
is channels-last and the elementwise update would then be strided.

Configuration constraints are hard
==================================

On gfx942 an illegal MFMA configuration does not fail.  It emits **zero** MFMA
instructions, falls back to vector FMA, and returns correct results at a
fraction of the speed.  So :class:`ConvConfig` refuses rather than deprioritises:
``BLOCK_M``/``BLOCK_N`` must be multiples of ``matrix_instr_nonkdim`` and
``BLOCK_K`` a multiple of the intrinsic's ``kDim`` (16 at nonkdim 16, 8 at 32,
for bf16).  :func:`verify_isa` exists because the only way to know the
constraints were met is to read the emitted ISA.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import torch
import triton
import triton.language as tl

__all__ = [
    "ConvConfig",
    "conv3d_forward",
    "default_config",
    "candidate_configs",
    "is_supported",
    "is_supported_all",
    "select_config",
    "to_rsck",
    "tune_key",
]


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@triton.jit
def _conv3d_fwd_kernel(
    X,
    W,
    Y,
    BIAS,
    # Sizes.  ``M_TOTAL`` is ``BATCH * OUT_D * OUT_H * OUT_W``.
    BATCH,
    IN_D,
    IN_H,
    IN_W,
    OUT_D,
    OUT_H,
    OUT_W,
    CIN,
    COUT,
    M_TOTAL,
    # Element strides.  The channel stride of X and Y is 1 by construction --
    # that is what NDHWC means -- so it is not passed and not multiplied by.
    stride_xn,
    stride_xd,
    stride_xh,
    stride_xw,
    # The weight, described by three strides over the *effective* GEMM's axes:
    # the fused tap index, the reduction axis K (Cin), and the output axis N
    # (Cout).  Which of the two channel strides is 1 is a constexpr (``W_ORDER``)
    # rather than a runtime fact, because it decides how the tile is loaded.
    stride_wt,
    stride_wk,
    stride_wn,
    stride_yn,
    stride_yd,
    stride_yh,
    stride_yw,
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
    BLOCK_K_COUNT: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    PADDED: tl.constexpr,
    INDEX_DTYPE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    W_ORDER: tl.constexpr,
    W_FLIP: tl.constexpr,
):
    # -- which output tile this program owns ------------------------------
    #
    # A flat program id with grouped-M ordering rather than a 2-D grid: the
    # group width is the L2 swizzle, and on MI300A it wants to be a multiple of
    # the 6 XCDs rather than MI300X's 8.  Programs in a group share their B
    # tiles, which for a convolution is the whole weight -- small and hot.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M_TOTAL, BLOCK_M)
    grid_n = tl.cdiv(COUT, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # -- unravel the fused ndhw index -------------------------------------
    #
    # Done once, outside the reduction.  The divisions are expensive and the
    # whole point of hoisting them is that the tap shift below is then a scalar.
    idx_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    idx_h = tmp % OUT_H
    tmp = tmp // OUT_H
    idx_d = tmp % OUT_D
    idx_n = tmp // OUT_D

    # Input coordinate of tap (0,0,0); tap (d,i,j) is this plus a scalar.
    src_d = idx_d * SD - PD
    src_h = idx_h * SH - PH
    src_w = idx_w * SW - PW

    # The row offset of the A operand.  Cast per term rather than after the sum:
    # ``idx_n * stride_xn`` alone overflows int32 for a batched scale-8 volume,
    # and the sum would then be wrong before the widening ever happened.
    x_row = (
        idx_n.to(INDEX_DTYPE) * stride_xn
        + src_d.to(INDEX_DTYPE) * stride_xd
        + src_h.to(INDEX_DTYPE) * stride_xh
        + src_w.to(INDEX_DTYPE) * stride_xw
    )
    m_valid = offs_m < M_TOTAL

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -- reduction: taps outermost, channel blocks innermost ---------------
    for dijk in range(KD * KH * KW * BLOCK_K_COUNT):
        k = (dijk % BLOCK_K_COUNT) * BLOCK_K
        dij = dijk // BLOCK_K_COUNT
        j = dij % KW
        di = dij // KW
        i = di % KH
        d = di // KH

        offs_k = k + tl.arange(0, BLOCK_K)

        # A: one voxel shift.  The addend is a scalar, so this is a uniform
        # bump of the row offset rather than a recomputed gather.
        tap_off = (d * DD) * stride_xd + (i * DH) * stride_xh + (j * DW) * stride_xw
        x_ptrs = X + (x_row + tap_off.to(INDEX_DTYPE))[:, None] + offs_k[None, :]

        if PADDED:
            in_d = src_d + d * DD
            in_h = src_h + i * DH
            in_w = src_w + j * DW
            row_ok = (
                m_valid
                & (in_d >= 0)
                & (in_d < IN_D)
                & (in_h >= 0)
                & (in_h < IN_H)
                & (in_w >= 0)
                & (in_w < IN_W)
            )
        else:
            # Unpadded: every tap of an in-range output voxel is in range, so
            # the six compares above are dead.  Worth compiling out -- they run
            # 27 times per K sweep.  This is the *rarer* arm at a ScaFFold site:
            # the adapter halos only the split axis, so a k>1 production
            # convolution compiles the PADDED branch above at every
            # configuration.  The transposed upsamplers and the k=1 head land
            # here.
            row_ok = m_valid
        mask_x = tl.broadcast_to(row_ok[:, None], (BLOCK_M, BLOCK_K))
        if not EVEN_K:
            mask_x = mask_x & (offs_k < CIN)[None, :]
        a = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # B: the weight tile, in whatever layout the weight arrived in.
        #
        #   W_ORDER == 0  Cout is contiguous, so the tile is a run of BLOCK_N
        #                 elements per row -- the RSCK buffer :func:`to_rsck`
        #                 materializes, and also a channels-last *parameter* seen
        #                 from backward-data, whose N is Cin.
        #   W_ORDER == 1  Cout is *not* contiguous, so each column of the tile is
        #                 addressed on its own.  A channels-last parameter seen
        #                 from the *forward* is this: Cin is unit-stride, and Cin
        #                 is K.
        #
        # An uncoalesced B tile sounds like it should be much worse than a
        # vectorized one and is not -- 0.96-1.05x per site, winning 6 of the 8
        # hottest forward cells outright (``1024->512 @ 18^3``: 0.384 vs 0.408 ms
        # before the transform it avoids is charged at all).  What matters is not
        # that the *lanes* are contiguous but that the tile's K-run is: at
        # ``BLOCK_K`` consecutive unit-stride elements each column costs two
        # cache lines, the weight is small and stays hot, and B is not where the
        # bandwidth goes.  Take that away -- a weight where neither channel axis
        # is unit-stride -- and the same instruction sequence is 4.8-9.0x
        # slower, which is why :func:`_weight_plan` refuses it rather than
        # compiling it.
        #
        # Two alternatives were implemented and measured worse, both on the eight
        # hot config-A sites: loading the tile coalesced along K and
        # transposing it in registers into the ``(BLOCK_K, BLOCK_N)`` the dot
        # wants costs 1.05-1.55x, and holding the
        # parameter in RSCK order costs the *backward* direction the same, since
        # the axis RSCK makes contiguous is backward-data's reduction axis.
        #
        # ``W_FLIP`` reverses the fused tap index.  Flipping all three kernel
        # axes is the complement of a mixed-radix index, i.e. exactly
        # ``taps - 1 - dij``, so backward-data's tap flip is this scalar rather
        # than a materialized copy of the weight.
        #
        # Offsets are widened by the same ``INDEX_DTYPE`` as A.  The widest
        # weight in this project is 28.3 M elements, 80x below int32, but
        # ``taps * Cin * Cout`` is bounded by nothing a caller cannot exceed: at
        # 2.30e9 elements the truncated offset goes *negative* and faults the
        # GPU.  Cast per term rather than after the sum -- ``dij * stride_wt`` is
        # the term that overflows on its own.  On the int32 path the casts are
        # frontend no-ops, so the operand keeps the buffer-load eligibility it
        # has today; it only loses it at sizes where the *storage* is already
        # over the buffer-op limit and has lost it anyway (see
        # :data:`~triton_conv3d.shapes.BUFFER_OP_MAX_BYTES`).
        dij_w = (KD * KH * KW - 1 - dij) if W_FLIP else dij
        w_row = dij_w.to(INDEX_DTYPE) * stride_wt + offs_k.to(INDEX_DTYPE) * stride_wk
        if W_ORDER == 0:
            w_ptrs = W + w_row[:, None] + offs_n[None, :]
        else:
            w_ptrs = W + w_row[:, None] + offs_n[None, :].to(INDEX_DTYPE) * stride_wn
        if EVEN_K and EVEN_N:
            b = tl.load(w_ptrs)
        elif EVEN_K:
            b = tl.load(w_ptrs, mask=(offs_n < COUT)[None, :], other=0.0)
        elif EVEN_N:
            b = tl.load(w_ptrs, mask=(offs_k < CIN)[:, None], other=0.0)
        else:
            b = tl.load(
                w_ptrs,
                mask=(offs_k < CIN)[:, None] & (offs_n < COUT)[None, :],
                other=0.0,
            )

        # ``input_precision`` only bites for fp32 operands, where the backend's
        # default splits the dot into reduced-precision pieces.  bf16 already
        # accumulates in fp32 and is unaffected; fp32 is the ``more_determinism``
        # path and has to actually be fp32, so it is asked for explicitly.
        acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_n, mask=offs_n < COUT, other=0.0)
        acc += bias[None, :].to(tl.float32)

    y_row = (
        idx_n.to(INDEX_DTYPE) * stride_yn
        + idx_d.to(INDEX_DTYPE) * stride_yd
        + idx_h.to(INDEX_DTYPE) * stride_yh
        + idx_w.to(INDEX_DTYPE) * stride_yw
    )
    y_ptrs = Y + y_row[:, None] + offs_n[None, :]
    mask_y = tl.broadcast_to(m_valid[:, None], (BLOCK_M, BLOCK_N))
    if not EVEN_N:
        mask_y = mask_y & (offs_n < COUT)[None, :]
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=mask_y)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


#: MFMA reduction depth per ``matrix_instr_nonkdim`` on gfx942, by operand
#: dtype.  ``BLOCK_K`` must be a multiple of this or ``chooseMfmaInstruction``
#: rejects the shape ("would introduce data duplication") and the dot silently
#: lowers to FMA.  Source: Triton v3.7.0 ``MfmaGroup.cpp`` plus the
#: ``inputKSize % kDim`` check in ``AccelerateAMDMatmul.cpp``.
_MFMA_KDIM = {
    torch.bfloat16: {16: 16, 32: 8},
    torch.float16: {16: 16, 32: 8},
    torch.float32: {16: 4, 32: 2},
}


@dataclasses.dataclass(frozen=True)
class ConvConfig:
    """One launch configuration, with the gfx942 constraints enforced.

    ``validate`` is not advisory.  Every constraint here has a *silent* failure
    mode: an illegal ``matrix_instr_nonkdim``, or a ``BLOCK_K`` that is not a
    multiple of the intrinsic's reduction depth, produces a kernel that runs and
    returns the right answer with no MFMA instruction in it at all.  A config
    generator that merely ranked such a config last would still be feeding
    meaningless entries into a best-of sweep.
    """

    BLOCK_M: int = 128
    BLOCK_N: int = 64
    BLOCK_K: int = 64
    GROUP_M: int = 6
    num_warps: int = 4
    num_stages: int = 2
    matrix_instr_nonkdim: int = 16
    kpack: int = 2
    waves_per_eu: int = 0

    def __str__(self) -> str:
        return (
            f"{self.BLOCK_M}x{self.BLOCK_N}x{self.BLOCK_K}"
            f"/g{self.GROUP_M}/w{self.num_warps}/s{self.num_stages}"
            f"/nk{self.matrix_instr_nonkdim}/kp{self.kpack}"
            + (f"/we{self.waves_per_eu}" if self.waves_per_eu else "")
        )

    def validate(self, dtype: torch.dtype) -> str | None:
        """``None`` if the config can reach the matrix core, else the reason."""
        kdims = _MFMA_KDIM.get(dtype)
        if kdims is None:
            return f"unsupported operand dtype {dtype}"
        if self.matrix_instr_nonkdim not in kdims:
            return (
                f"matrix_instr_nonkdim={self.matrix_instr_nonkdim} is not one of "
                f"{sorted(kdims)}; anything else falls back to FMA"
            )
        nk = self.matrix_instr_nonkdim
        if self.BLOCK_M % nk or self.BLOCK_N % nk:
            return f"BLOCK_M/BLOCK_N must be multiples of nonkdim={nk}"
        if self.BLOCK_K % kdims[nk]:
            return f"BLOCK_K must be a multiple of {kdims[nk]} at nonkdim={nk}"
        if self.num_warps < 1 or self.num_warps & (self.num_warps - 1):
            return "num_warps must be a positive power of two"
        if self.num_warps > self.BLOCK_M * self.BLOCK_N // 256:
            return "more warps than 16x16 tiles in the output block"
        if self.num_stages < 2:
            # Block-pingpong is on by default for gfx942 and needs > 1.
            return "num_stages must be at least 2 on gfx942"
        if self.GROUP_M < 1:
            # The swizzle divides by ``GROUP_M * grid_n`` and takes ``pid %
            # group_size``; at 0 that is a division by zero inside the kernel,
            # which on gfx942 is not a trap but a garbage ``pid_m`` and a memory
            # access fault.  A negative value reaches the kernel just as far.
            # Every legal value is fine, including ones that do not divide
            # ``grid_m`` and ones far larger than it.
            return "GROUP_M must be at least 1"
        return None

    def lds_bytes(self, dtype: torch.dtype) -> int:
        """Shared memory the two operand tiles need, in bytes.

        Measured rather than assumed: across 22 tile/dtype combinations the
        compiler's own ``metadata.shared`` equalled
        ``(BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * itemsize`` exactly, with no
        double-buffering factor -- Triton's gfx942 pipeliner keeps one LDS
        buffer at ``num_stages=2``.  So this is the real number and not a bound.
        """
        elem = torch.empty((), dtype=dtype).element_size()
        return (self.BLOCK_M * self.BLOCK_K + self.BLOCK_K * self.BLOCK_N) * elem

    def launch_kwargs(self) -> dict:
        return {
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "matrix_instr_nonkdim": self.matrix_instr_nonkdim,
            "kpack": self.kpack,
            "waves_per_eu": self.waves_per_eu,
        }


def _pow2_at_most(x: int, cap: int) -> int:
    return max(16, min(cap, 1 << max(0, (max(1, x)).bit_length() - 1)))


#: gfx942's shared memory per workgroup.  Exceeding it raises ``OutOfResources``
#: at launch -- loudly, unlike the MFMA constraints, which is why M1 left it to
#: fail rather than guarding it statically.  That is the right call for a *sweep*
#: candidate and the wrong one for the config the entry point picks on its own,
#: which is what :func:`_fit_to_lds` is for.
_LDS_BYTES = 64 * 1024


def _fit_to_lds(cfg: ConvConfig, dtype: torch.dtype) -> ConvConfig:
    """Shrink a tile until its operands fit in LDS, then legalize the warps.

    This exists because of an fp32 hole that M2's tests fell into: the block
    sizes in :func:`default_config` were chosen against bf16, and fp32 operands
    are twice the bytes, so ``Cin >= 512`` in fp32 asked for 128 KiB and the
    *shipped* configuration raised.  ``more_determinism`` runs the model in
    fp32, so that was reachable from a real ScaFFold configuration.

    ``BLOCK_K`` is halved first: it is the reduction depth, so shortening it
    costs some reuse but changes neither the grid nor the parallelism, whereas
    halving ``BLOCK_M`` doubles the program count.  The loud failure remains as
    the backstop for an explicitly supplied ``config=``.
    """
    nk = cfg.matrix_instr_nonkdim
    kdim = _MFMA_KDIM.get(dtype, {}).get(nk)
    if kdim is None:
        return cfg
    while cfg.lds_bytes(dtype) > _LDS_BYTES:
        half_k, half_m, half_n = cfg.BLOCK_K // 2, cfg.BLOCK_M // 2, cfg.BLOCK_N // 2
        if half_k >= kdim and half_k % kdim == 0:
            cfg = dataclasses.replace(
                cfg, BLOCK_K=half_k, kpack=1 if half_k <= 16 else cfg.kpack
            )
        elif half_m >= nk and half_m % nk == 0:
            cfg = dataclasses.replace(cfg, BLOCK_M=half_m)
        elif half_n >= nk and half_n % nk == 0:
            cfg = dataclasses.replace(cfg, BLOCK_N=half_n)
        else:
            break  # nothing left to shrink; let the launch say so
    warps = max(1, min(cfg.num_warps, cfg.BLOCK_M * cfg.BLOCK_N // 256))
    return dataclasses.replace(cfg, num_warps=1 << (warps.bit_length() - 1))


#: Below this many programs the grid cannot fill MI300A's 228 CUs, and a
#: narrower ``BLOCK_M`` buys more parallelism than it loses in reuse.  Half a
#: wave rather than a whole one: measured, the ``1024 -> 1024`` bottleneck at
#: ``M = 512`` wants 128 programs and is *slower* when pushed to 256.
_MIN_PROGRAMS = 114


def default_config(
    m: int, cin: int, cout: int, dtype: torch.dtype = torch.bfloat16
) -> ConvConfig:
    """A config that is legal for any shape and close to tuned for most.

    Measured, not guessed: over 15 corpus shapes and ~1000 timed configurations
    the surface is remarkably flat and almost entirely determined by the channel
    widths.  ``BLOCK_M=128`` won 14 of 15, ``BLOCK_N`` tracks ``Cout`` up to 128,
    and ``BLOCK_K`` is 64 below ``Cin=512`` and 128 above.

    Two things this does *not* do, both because the measurement said not to:

    * It does not use ``matrix_instr_nonkdim=32``.  The M0 ceiling probe found 32
      winning 9 of 10 ``N=64`` cells on a plain GEMM and worth 1.21x there, which
      is why the M1 brief said to sweep it.  Swept here on the real convolution,
      **16 won all 15 cells**, by 0.6-13% (geometric mean 5.8%).  The GEMM result
      does not transfer: the convolution's inner loop carries a per-tap boundary
      predicate and a 27x longer reduction, so it is not the same instruction
      mix the ceiling probe measured.
    * It does not scale ``BLOCK_M`` with ``M``.  A tall tile only pays while the
      grid still fills the device, which at these shapes it always does; the one
      place it does not is handled by :data:`_MIN_PROGRAMS`.
    """
    block_n = _pow2_at_most(cout, 128)
    block_k = 128 if cin >= 512 else _pow2_at_most(cin, 64)
    block_m = _pow2_at_most(m, 128)
    nonkdim = 16
    kdim = _MFMA_KDIM[dtype][nonkdim]
    block_k = max(kdim, block_k - block_k % kdim)
    return _fit_to_lds(
        _fit_to_grid(
            ConvConfig(
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_M=6,
                num_warps=8 if block_k >= 128 or block_n >= 256 else 4,
                num_stages=2,
                matrix_instr_nonkdim=nonkdim,
                kpack=1 if block_k <= 16 else 2,
            ),
            m,
            cout,
        ),
        dtype,
    )


def _fit_to_grid(cfg: ConvConfig, m: int, cout: int) -> ConvConfig:
    """Shrink ``BLOCK_M`` until the grid can fill the device, then legalize warps.

    Applied to tuned entries as well as to the heuristic, because a tuned entry
    is keyed on the channel widths and so can be reused at an ``M`` far smaller
    than the one it was measured at -- which is exactly where a 128-row tile
    stops being a good idea.
    """
    nk = cfg.matrix_instr_nonkdim
    while (
        cfg.BLOCK_M > max(16, nk)
        and (cfg.BLOCK_M // 2) % nk == 0
        and -(-m // cfg.BLOCK_M) * -(-cout // cfg.BLOCK_N) < _MIN_PROGRAMS
    ):
        cfg = dataclasses.replace(cfg, BLOCK_M=cfg.BLOCK_M // 2)
    warps = max(1, min(cfg.num_warps, cfg.BLOCK_M * cfg.BLOCK_N // 256))
    warps = 1 << (warps.bit_length() - 1)
    return dataclasses.replace(cfg, num_warps=warps)


#: PyTorch Inductor's ROCm convolution seed grid, ``(BLOCK_M, BLOCK_N, BLOCK_K,
#: num_warps)``.  Preferred over a blind sweep because these values are already
#: tuned on ROCm; its per-config ``num_stages`` is dropped because
#: ``ROCmConfigHeuristic._filter_configs`` overwrites it with 2 on HIP anyway.
_SEED_TILES: tuple[tuple[int, int, int, int], ...] = (
    (64, 256, 16, 4),
    (256, 64, 16, 4),
    (128, 128, 32, 8),
    (64, 64, 32, 4),
    (64, 256, 32, 8),
    (256, 64, 32, 8),
    (128, 128, 64, 8),
    (64, 128, 64, 4),
    (128, 64, 64, 4),
    (256, 128, 64, 8),
    (128, 256, 64, 8),
    (128, 128, 128, 8),
    (64, 128, 128, 4),
    (256, 128, 128, 8),
    (128, 256, 128, 8),
)

#: Extra tiles for the skinny-N regime.  ``Cout=64`` is the model's most common
#: output width and the seed grid has little there; M0 found every winner in
#: this band was tall in M with ``BLOCK_N=64``.
_SKINNY_N_TILES: tuple[tuple[int, int, int, int], ...] = (
    (256, 64, 64, 8),
    (512, 64, 64, 8),
    (256, 64, 128, 8),
    (512, 64, 32, 8),
    (128, 64, 128, 4),
    (64, 64, 64, 4),
    (64, 64, 128, 4),
    (1024, 64, 32, 8),
)

#: Tiles for the *narrow*-N regime, ``Cout <= 16``.  Only the segmentation head
#: (``64 -> 6``) reaches it in ScaFFold, and until 2026-08-03 the head had no
#: tile evidence at all: ``candidate_configs`` prunes on ``bn > 2 * n2``, which
#: at ``Cout = 6`` gives ``n2 = 16`` and removes every entry of both grids
#: above, so the generator fell through to ``default_config`` and the "sweep"
#: recorded for that site timed the shipped config twice with two ``GROUP_M``.
#:
#: The ``num_warps`` column is the point of this grid, not the tile.  A
#: ``BLOCK_N`` of 16 is one MFMA fragment wide, so there is no N work to hand a
#: second wave; four warps each take a quarter of ``BLOCK_M`` and replicate the
#: whole per-K-tile address computation for a fragment that is 10/16 padding.
#: Measured at all three head volumes, one warp is 1.02-1.22x over the shipped
#: four.
#:
#: Gated to ``Cout <= 16`` in :func:`candidate_configs` rather than added to the
#: grids above, because a 16-column tile at ``Cout = 512`` is 32x padding and
#: would only lengthen every other site's sweep -- and because ``num_warps=1``
#: is a **catastrophe** outside this regime: raced on the shipped tile it is
#: 0.166x on the ``128 -> 128 @ 66^3`` forward and 0.245x on its backward-data.
_NARROW_N_TILES: tuple[tuple[int, int, int, int], ...] = (
    (128, 16, 64, 1),
    (64, 16, 64, 1),
    (128, 16, 64, 2),
    (256, 16, 64, 1),
    (128, 16, 32, 1),
    (64, 16, 32, 1),
    (128, 32, 64, 1),
)


def candidate_configs(
    m: int,
    cin: int,
    cout: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    group_ms: Sequence[int] = (6,),
    nonkdims: Sequence[int] = (16, 32),
) -> list[ConvConfig]:
    """Configs worth timing for one shape, already pruned to legal ones.

    ``matrix_instr_nonkdim`` is *swept* over {16, 32} rather than fixed at 16.
    Fixing it at 16 is what Inductor does and what AMD's guidance says, and at
    ``Cout=64`` the M0 probe measured that advice costing 11-18%.

    ``GROUP_M`` defaults to 6 alone -- MI300A's XCD count, and the value that won
    the M0 square-GEMM probe -- because sweeping it doubles a list whose cost is
    almost entirely JIT compilation.  The caller refines it on the finalists
    instead, which is where a few percent of L2 locality is actually decidable.
    """
    m2 = max(16, triton.next_power_of_2(m))
    n2 = max(16, triton.next_power_of_2(cout))
    k2 = max(16, triton.next_power_of_2(cin))
    out: list[ConvConfig] = []
    seen: set[ConvConfig] = set()
    tiles = _SEED_TILES + _SKINNY_N_TILES
    if n2 <= 16:
        tiles += _NARROW_N_TILES
    for bm, bn, bk, seed_warps in tiles:
        # Skip tiles that would mostly compute padding.  BLOCK_K is capped at
        # the channel count rather than twice it because the reduction is
        # per-tap: a BLOCK_K above Cin wastes a whole tap's worth of MFMA.
        if bm > 2 * m2 or bn > 2 * n2 or bk > k2:
            continue
        for warps in {4, 8, seed_warps}:
            for nonkdim in nonkdims:
                for group_m in group_ms:
                    cfg = ConvConfig(
                        BLOCK_M=bm,
                        BLOCK_N=bn,
                        BLOCK_K=bk,
                        GROUP_M=group_m,
                        num_warps=warps,
                        num_stages=2,
                        matrix_instr_nonkdim=nonkdim,
                        kpack=1 if bk <= 16 else 2,
                    )
                    # LDS overflow is pruned rather than shrunk: shrinking would
                    # fold two seed tiles onto one entry and silently
                    # double-count it in the sweep.  Only configs that could not
                    # have run at all are removed, so no measured winner is lost.
                    if (
                        cfg.validate(dtype) is not None
                        or cfg.lds_bytes(dtype) > _LDS_BYTES
                        or cfg in seen
                    ):
                        continue
                    seen.add(cfg)
                    out.append(cfg)
    if not out:
        out.append(default_config(m, cin, cout, dtype))
    return out


def tune_key(dtype: torch.dtype, cin: int, cout: int, kernel: tuple[int, ...]) -> tuple:
    return (str(dtype), cin, cout, tuple(kernel))


def _tuned(
    bm: int, bn: int, bk: int, warps: int, group_m: int = 6, nk: int = 16
) -> ConvConfig:
    """One measured row.

    ``nk`` defaults to 16 because that is what the forward measured -- 16 won
    all 15 cells of M1's 1090-config sweep, and :func:`default_config` says why.
    It is a parameter at all for exactly one row, the ``3 -> 64`` stem, where 32
    is not chosen for its own sake: ``_MFMA_KDIM[bf16][32] = 8`` is what makes
    ``BLOCK_K = 8`` legal, and ``BLOCK_K = 8`` is the whole effect.  Raced as a
    control, ``nonkdim=32`` on the *shipped* ``128x64x16`` tile measures 0.9682
    ms against 0.9554 -- i.e. nothing.  Spelled the same way
    :func:`~triton_conv3d.reduce_gemm._tuned` spells it, and for the same
    reason: a per-row knob whose default carries the rule.
    """
    return ConvConfig(
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=group_m,
        num_warps=warps,
        num_stages=2,
        matrix_instr_nonkdim=nk,
        kpack=1 if bk <= 16 else 2,
    )


#: Measured winners, keyed by ``(dtype, Cin, Cout, kernel)``; a miss falls back to
#: :func:`default_config`, and every hit still goes through :func:`_fit_to_grid`.
#:
#: Keyed on the channel widths and *not* on the spatial extent because that is
#: what the measurement showed: where a channel pair occurs at more than one
#: volume in the corpus -- ``128 -> 64`` at three, ``512 -> 512`` at two --
#: the same tile won at each.  ``GROUP_M`` is the exception; it flips between 6
#: and 8 across volumes but is worth under 1% either way at the shapes where it
#: flips, so 6 (MI300A's XCD count) is used throughout.
#:
#: Deliberately a table and not ``@triton.autotune``: ScaFFold's figure of merit
#: is total wall time, so a recompile inside a training step is a direct loss.
#: Drawn from a forward sweep of 15 problems and ~1050 timed configs.
#: Only channel pairs that were actually timed appear here.  The unmeasured
#: pairs -- ``64 -> 128``, ``128 -> 256``, ``256 -> 512``, ``512 -> 1024``, all
#: encoder-side -- fall to the heuristic on purpose: an extrapolated entry in a
#: table called "measured winners" is worse than no entry, because it cannot be
#: told apart from one.  Priced since: served by the heuristic those eight cells
#: span 0.80x to 1.32x and are worth **+0.02 ms/step at config A and +0.12 at
#: C** -- a wash, so tuning them is not the action.  Three of them are *losses*
#: (``256 -> 512 @ 18^3`` 0.84x, ``512 -> 1024 @ 10^3`` 0.80x, ``512 -> 1024 @
#: 6x18x18`` 0.90x) and belong on the adapter's block-list instead.  (An earlier
#: version of this comment priced the four pairs at "12.7 ms/step"; that figure
#: is config B's *all-directions* total at those pairs, not the forward time an
#: absent row here governs.)
_TUNED: dict[tuple, ConvConfig] = {
    # The segmentation head, and the one entry here that is *not* from the main
    # forward sweep but from a follow-up race at this site alone.
    # ``Cout = 6`` prunes every seed tile, so that sweep never timed anything
    # but ``default_config`` at this site and the tile below is that same
    # ``128x16x64`` with **one warp instead of four** -- see
    # :data:`_NARROW_N_TILES` for why one, and why only here.  Raced against the
    # shipped four at all three head volumes in one interleaved block:
    # 1.024x @ 128^3, 1.137x @ 64x256^2, 1.217x @ 128x256^2, i.e. 1.19-1.21x of
    # MIOpen where the shipped config was 1.16x, 1.04x and **0.99x**.  The
    # runner-up ``64x16x64/w1`` wins the two smaller volumes by 2-3% and loses
    # the largest by 8%, so it is not shipped; the choice between them is worth
    # under 0.01 ms/step either way.
    tune_key(torch.bfloat16, 64, 6, (1, 1, 1)): _tuned(128, 16, 64, 1),
    **{
        tune_key(torch.bfloat16, cin, cout, (3, 3, 3)): cfg
        for (cin, cout), cfg in {
            # The UNet stem, and the row that refutes the standing verdict that
            # "``conv 3->64`` is genuinely hopeless, leave it on MIOpen".  It was
            # never hopeless and it was never a matrix-core feeding problem: the
            # reduction axis of this kernel's ``tl.dot`` is ``Cin`` **alone**
            # (``BLOCK_K_COUNT = cdiv(Cin, BLOCK_K)``, taps outermost), and
            # ``BLOCK_K`` is floored both by the MFMA intrinsic's reduction depth
            # and by ``_pow2_at_most``'s own floor of 16 -- so at ``Cin = 3``
            # every dot had **3 live columns of 16** and 81% of the matrix-core
            # work multiplied padding this kernel put there itself.
            # ``SQ_INSTS_MFMA`` measures 14,155,776 per call at ``130^3``,
            # exactly 5.333x the useful FLOPs, against MIOpen's 3,145,728
            # (2.370x -- CK contracts over the merged ``(Z,Y,X,C)`` axis, dense
            # 81 padded once to 96).  On *issued* MFMA the old config already ran
            # at 23.7% of the measured ``tl.dot`` ceiling against CK's 16.2%; it
            # just issued 2.25x more of it.
            #
            # ``BLOCK_K = 8`` is the fix and ``nonkdim=32`` is only how it is
            # spelled -- see :func:`_tuned`.  Raced against the heuristic's
            # ``128x64x16/nk16/w4`` and MIOpen, one interleaved block per volume,
            # kernel-only, at every volume the corpus has for this pair:
            #   130^3      0.5236 ms vs MIOpen 0.6236 -- 1.193x [1.190,1.196]
            #   130x258^2  2.0522 ms vs MIOpen 2.4907 -- 1.214x [1.211,1.217]
            #   66x258^2   1.0364 ms vs MIOpen 1.2468 -- 1.203x [1.201,1.206]
            # i.e. 1.83-1.86x over the config it replaces, which was 0.651-0.653x
            # of MIOpen.  Flat across the pair's 4.0x span of volume, which is the
            # property this table has twice been burned by not checking.
            #
            # Bitwise **identical** to the config it replaces on random operands
            # (0 of 134 M elements differ), because at ``Cin = 3`` only 3 products
            # per tap are non-zero whatever ``BLOCK_K`` is and the 27 taps are
            # still visited in order.  So no determinism baseline moves.
            #
            # ``kpack = 1`` is not a rounding detail here: ``kp2`` on the same
            # tile is 0.6085 ms, 1.17x worse.  Taller than 512 turns over
            # (``1024x64x8/w16`` 0.6042); ``GROUP_M`` is inert at this site
            # (g1/g6/g12 within 1%).
            (3, 64): _tuned(512, 64, 8, 8, nk=32),
            (64, 64): _tuned(128, 64, 64, 4),
            (128, 64): _tuned(128, 64, 64, 4),
            (128, 128): _tuned(128, 128, 64, 4),
            (256, 128): _tuned(128, 128, 64, 4),
            (256, 256): _tuned(128, 128, 64, 4),
            # The one place the heuristic's "Cin >= 512 wants BLOCK_K=128" rule
            # is wrong: here BLOCK_K=64 is 7% faster.  Cout=256 rather than 512
            # is what distinguishes it, on one data point, so the rule stands.
            (512, 256): _tuned(128, 128, 64, 4),
            (512, 512): _tuned(128, 128, 128, 8),
            (1024, 512): _tuned(128, 128, 128, 8),
            (1024, 1024): _tuned(64, 64, 128, 8),
            # The two 2048-channel bottleneck pairs.  They are scale-8 sites
            # that appeared in no corpus until a shape census of running steps
            # found them -- the corpus's scale-8 model was a *four*-layer
            # network and the harness runs a five-layer one -- so until now
            # they fell to the heuristic, and against fresh MIOpen the forward
            # **lost**, 0.952x and 0.977x.
            #
            # ``128x64x128`` wins at **every volume both pairs occur at**,
            # which is the property this table has twice been burned by not
            # checking, and it is one row rather than three because
            # ``_fit_to_grid`` walks ``BLOCK_M`` down 128 -> 64 -> 32 as ``M``
            # falls 512 -> 256 -> 128.  Raced against the heuristic it
            # replaces, quiet node, **four independently allocated operand sets
            # per cell**, both arms sharing each build's operands so the pair is
            # immune to the placement effect below; kernel-only, median over
            # builds with the worst build in brackets:
            #   (1024,2048)  8^3     0.3196 vs 0.3224 ms -- 1.010x [1.008]
            #   (1024,2048)  6x8^2   0.2208 vs 0.2769    -- 1.253x [1.251]
            #   (1024,2048)  4x8^2   0.1760 vs 0.2304    -- 1.308x [1.306]
            #   (2048,2048)  8^3     0.6663 vs 0.7321    -- 1.099x [1.097]
            #   (2048,2048)  6x8^2   0.4819 vs 0.6354    -- 1.318x [1.315]
            #   (2048,2048)  4x8^2   0.3844 vs 0.5066    -- 1.317x [1.303]
            #
            # The builds are not ceremony, and one caveat has to travel with
            # this row.  ``(2048, 2048)`` is the one site in this project whose
            # time depends on **where its weight lands**: at 216 MiB against a
            # 256 MiB MALL the heuristic is bimodal, two tight states up to
            # 14.7% apart and fixed for the life of the allocation.  Measured
            # solo -- one config, one operand set, six rebuilds, which is what
            # a caller actually sees -- this row is **stable**: 0.5% / 0.7% /
            # 1.4% spread at the three volumes against the heuristic's 15% / 5%
            # / 20%.  But at ``8^3``
            # its 0.6681 ms sits *between* the heuristic's two states (0.6472
            # and 0.7453), so it beats the unlucky allocation by 1.12x and
            # **loses to the lucky one by 0.97x**.  It is shipped because the
            # expected value and both sharded volumes are clear wins and the
            # variance goes away, not because it dominates.
            #
            # ``(2048, 1024)`` is deliberately **absent**, and that is a result
            # rather than an omission: ``128x256x64`` is **1.433x** at ``16^3``
            # and **0.804x** and **0.504x** at the two sharded volumes of the
            # same pair (two builds each, every interval tight).  The
            # discriminator is ``BLOCK_K`` against ``M`` -- 64 wins at ``M =
            # 4096`` and 128 wins at 2048 and 1024 -- this table is keyed per
            # channel pair and cannot say that, and an
            # entry would trade config B's 0.68 ms saving for config C and D's
            # 0.23 and 0.16 ms losses.  ``_fit_to_grid`` already walks
            # ``BLOCK_M`` with ``M``; making it walk ``BLOCK_K`` too is the
            # change this measurement argues for, and it is not made here
            # because one channel pair is not enough evidence to move a rule
            # every pair goes through.
            (1024, 2048): _tuned(128, 64, 128, 8),
            (2048, 2048): _tuned(128, 64, 128, 8),
        }.items()
    },
}


def register_tuned(dtype, cin, cout, kernel, config: ConvConfig) -> None:
    _TUNED[tune_key(dtype, cin, cout, kernel)] = config


def select_config(
    m: int,
    cin: int,
    cout: int,
    kernel: Sequence[int],
    dtype: torch.dtype,
    *,
    table: dict | None = None,
    key: tuple | None = None,
) -> ConvConfig:
    """The config the kernel will run: tuned entry if there is one, else heuristic.

    ``m``/``cin``/``cout`` always describe the GEMM that will actually be issued
    -- ``(M, N, K) = (m, cout, cin * prod(kernel))`` -- because that is what
    :func:`_fit_to_grid` has to reason about.  ``table`` and ``key`` are separate
    so that :mod:`triton_conv3d.bwd_data`, whose effective GEMM has the channel
    widths *swapped*, can keep a table keyed on the problem a reader recognises
    while the tile is still fitted to the grid it will really launch on.
    """
    table = _TUNED if table is None else table
    if key is None:
        key = tune_key(dtype, cin, cout, tuple(kernel))
    tuned = table.get(key)
    if tuned is not None:
        return _fit_to_lds(_fit_to_grid(tuned, m, cout), dtype)
    return default_config(m, cin, cout, dtype)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def _triple(v, name: str) -> tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    t = tuple(int(x) for x in v)
    if len(t) != 3:
        raise ValueError(f"{name} must be an int or a length-3 sequence, got {v!r}")
    return t  # type: ignore[return-value]


def to_rsck(w: torch.Tensor) -> torch.Tensor:
    """PyTorch's ``(Cout, Cin, kd, kh, kw)`` weight as ``(kd, kh, kw, Cin, Cout)``.

    A B tile whose row is a contiguous run wants Cout fastest-varying, which is
    what this produces.  **It is no longer on any shipped path.**  The kernel
    reads the parameter wherever it lies, and this copy ran once per layer per
    *optimizer step* -- 0.786 ms/step over the 19 Conv3d sites of one
    configuration, 8.4x the single-copy floor, because 19 small strided
    ``permute().contiguous()`` launches are latency-bound rather than
    bandwidth-bound, and no caching could remove it because the optimizer
    dirties every parameter every step.  Measured against reading a
    channels-last parameter in place, materializing this buffer is *slower* on 6
    of the 8 hottest forward sites before the copy is charged at all.

    It is kept, and still supported through ``weight_rsck=``, for the weights
    :func:`_weight_plan` refuses -- chiefly PyTorch's *default* layout, in which
    neither channel axis is unit-stride and the gathered load is 4.8-9.0x slower
    than copying.  A ScaFFold parameter is never in it, because the model is
    moved to ``channels_last_3d`` at construction.
    """
    return w.permute(2, 3, 4, 1, 0).contiguous()


#: How the kernel's B operand is laid out -- the values of the kernel's
#: ``W_ORDER`` constexpr.  ``_W_GENERAL`` costs nothing measurable against
#: ``_W_N_CONTIG`` (0.96-1.05x per site), which is why there is no third value:
#: a "coalesce along K and ``tl.trans``" order was implemented and measured at
#: 1.05-1.55x, i.e. a real loss, and deleted.
_W_N_CONTIG = 0
_W_GENERAL = 1


def _weight_plan(w: torch.Tensor) -> tuple[int, int, int, int] | None:
    """``(W_ORDER, stride_wt, stride_wk, stride_wn)`` for ``w``, or ``None``.

    ``w`` is the weight *as this GEMM sees it*: ``(Cout, Cin, kd, kh, kw)``,
    where for backward-data the two channel widths are the real convolution's
    swapped and ``w`` is a permuted view.  Strides, not memory format, are what
    the kernel needs, so this is a stride computation and not a
    ``is_contiguous(memory_format=...)`` test -- the backward-data view is
    neither contiguous nor channels-last and is still perfectly addressable.

    ``None`` means materialize :func:`to_rsck` instead, for one of two reasons:
    the three kernel axes are not one fused axis of constant stride, which is
    what the kernel's single ``dij * stride_wt`` assumes (a weight sliced along a
    kernel axis; nothing in this project produces one), or *neither* channel axis
    is unit-stride, which is a correctness-neutral but 4.8-9.0x performance
    cliff -- see the comment below.

    Extents of 1 carry no observable stride, so they constrain nothing and are
    skipped -- ``k=1x1x1`` is a real corpus shape (the segmentation head), and
    demanding ``stride(4) == 1`` of it would reject the weights of every model
    that has one.
    """
    cout, cin, kd, kh, kw = (int(v) for v in w.shape)
    s = tuple(int(v) for v in w.stride())
    if kw > 1:
        st = s[4]
    elif kh > 1:
        st = s[3]
    elif kd > 1:
        st = s[2]
    else:
        st = 0  # one tap: ``dij`` is always 0, so any stride is the right one
    if (
        (kw > 1 and s[4] != st)
        or (kh > 1 and s[3] != st * kw)
        or (kd > 1 and s[2] != st * kw * kh)
    ):
        return None
    if cout == 1 or s[0] == 1:
        return (_W_N_CONTIG, st, s[1], 1)
    if cin == 1 or s[1] == 1:
        return (_W_GENERAL, st, s[1], s[0])
    # Neither channel axis is unit-stride -- PyTorch's *default* weight layout,
    # where the only dense axis is the 27-element tap axis, which is not a tile
    # axis.  Every element of the B tile is then its own cache line: measured
    # **4.8-9.0x** slower than materializing RSCK across the eight hottest
    # forward sites, the copy charged to every call (9.31 ms against 1.15 at
    # ``256->128 @ 66^3``), and 2.2-6.2x on backward-data.  So this one really
    # does have to be copied, and it is the only layout left that does.
    return None


def is_supported(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> bool:
    """Whether :func:`conv3d_forward` will serve this call.

    Deliberately conservative: the caller's fallback is MIOpen, which is correct
    everywhere, so a false negative costs a little speed and a false positive
    costs a wrong answer.

    It is also **total**.  This is the gate of a Triton -> MIOpen rung ladder, so
    an argument it cannot interpret has to be a ``False`` and not an exception:
    ``padding=None`` and ``padding=1.5`` are ``TypeError`` out of :func:`_triple`
    and would otherwise take down a caller that was only asking a question.

    **This gates the forward and nothing else, and the three gates do not
    agree.**  A ``stride > 1`` call is served here and by
    :func:`~triton_conv3d.reduce_gemm.is_supported_bwd_weight`, and *refused* by
    :func:`~triton_conv3d.bwd_data.is_supported_bwd_data`, whose kernel-free
    formulation (backward-data as the forward contraction on a flipped weight)
    only holds at unit stride.  A caller that will differentiate the result must
    therefore ask :func:`is_supported_all` instead: a ``True`` from this function
    alone builds a graph node whose backward this package cannot answer, and by
    then the caller's fallback is gone.  A forward-only caller (inference) should
    keep asking this one -- the stride support is real, and the combined gate
    would take it away.
    """
    if groups != 1:
        return False
    if x.dim() != 5 or w.dim() != 5:
        return False
    if x.dtype != w.dtype or x.dtype not in _MFMA_KDIM:
        return False
    # Same device, not merely both on *a* device.  Triton launches on the current
    # device and dereferences the other pointer anyway; ScaFFold runs four GPUs
    # per node, where peer access turns that into another rank's data rather than
    # a fault.
    if not x.is_cuda or not w.is_cuda or w.device != x.device:
        return False
    if bias is not None:
        # The kernel masks the bias load against ``Cout``, which says nothing
        # about how long the bias actually is, and indexes it with an element
        # stride of 1.  So a short bias reads past the end -- whatever is in
        # memory there becomes the bias, ``nan`` if you are lucky -- and a
        # stride-2 view of the right length silently applies every other value.
        # ``torch.conv3d`` rejects both; so does this.
        if (
            bias.dim() != 1
            or int(bias.shape[0]) != int(w.shape[0])
            or bias.dtype != x.dtype
            or not bias.is_cuda
            or bias.device != x.device
            or bias.stride(0) != 1
        ):
            return False
    if x.shape[1] != w.shape[1]:
        return False
    try:
        s = _triple(stride, "stride")
        p = _triple(padding, "padding")
        d = _triple(dilation, "dilation")
    except (ValueError, TypeError):
        return False
    k = tuple(w.shape[2:])
    if any(v < 1 for v in s + d) or any(v < 0 for v in p):
        return False
    # Degenerate extents.  Each of these clears the output-voxel test below and
    # then disagrees with torch, which is the asymmetry this predicate exists to
    # prevent: a zero-length spatial axis with padding returns a volume of pure
    # padding where torch raises; a zero-size kernel returns an output *larger*
    # than the input, because ``(in + 2p - d(k-1) - 1)//s + 1`` gains one at
    # ``k = 0``; and ``Cin = 0`` returns ``Cout`` channels of zeros where torch
    # returns a tensor with no channels at all -- a different shape.  ``N = 0``
    # is not here: it agrees with torch (an empty grid, an empty result).
    if any(v < 1 for v in x.shape[2:]) or any(v < 1 for v in k):
        return False
    if w.shape[0] < 1 or w.shape[1] < 1:
        return False
    # Every output voxel must exist: a kernel wider than the padded input has
    # an empty output, which the M-unravel cannot express.
    for i in range(3):
        eff = d[i] * (k[i] - 1) + 1
        if x.shape[2 + i] + 2 * p[i] < eff:
            return False
    return True


def is_supported_all(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> bool:
    """Whether **every** direction of this convolution will be served.

    The gate for a caller that is going to differentiate: :func:`is_supported`
    and ``bwd_data.is_supported_bwd_data`` and
    ``reduce_gemm.is_supported_bwd_weight``, asked about the one call the caller
    has in hand and about the gradient it does not have yet.

    It exists because the three direction gates genuinely disagree and the
    disagreement is a trap.  ``stride > 1`` is supported by the forward (its
    output-voxel unravel simply steps by ``s``) and by backward-weight (the
    reduction is indexed by the *output* voxel, so a stride is three extra
    multiplies), and is not supported by backward-data, which has no kernel of
    its own: at unit stride it *is* the forward contraction on a flipped,
    channel-transposed weight, and a stride turns that into a scatter into a
    sub-lattice.  So a training caller that asks only the forward gate gets a
    ``True``, builds a graph node, and discovers at ``backward()`` -- when its
    own fallback is no longer reachable, because the node is already in the
    graph -- that the gradient cannot be computed.

    The direction gates are deliberately left as they are.  Narrowing the
    forward's to the intersection would agree with backward-data by taking a
    capability away from inference, which asks only the forward and for which
    strided convolution works today; and there is no single "the backward"
    answer to agree with anyway, since backward-weight accepts the stride the
    forward does.  The asymmetry is a fact about the three kernels; what was
    wrong was that a caller had to know it.  Now it can ask.

    Total for the same reason :func:`is_supported` is: an argument that cannot
    be interpreted is a ``False``, never an exception.  The forward's gate runs
    first and validates the triples, so the arithmetic below is reached only
    with arguments it has already accepted.

    The gradient is passed as a **metadata-only stand-in**: all three predicates
    read rank, shape, dtype, device and ``is_cuda`` and never a stride, a value
    or a contiguity, so a one-element allocation expanded to the output shape
    answers exactly as the real gradient would.  ``expand`` gives every dim a
    stride of 0, so if a predicate ever grows a stride test it will see those
    zeros and answer ``False`` -- a fallback to the caller's other kernel, which
    is the safe direction.
    """
    if not is_supported(x, w, bias, stride, padding, dilation, groups):
        return False
    s = _triple(stride, "stride")
    p = _triple(padding, "padding")
    d = _triple(dilation, "dilation")
    k = tuple(int(v) for v in w.shape[2:])
    grad_shape = (int(x.shape[0]), int(w.shape[0])) + tuple(
        (int(x.shape[2 + i]) + 2 * p[i] - d[i] * (k[i] - 1) - 1) // s[i] + 1
        for i in range(3)
    )
    grad = x.new_empty((1, 1, 1, 1, 1)).expand(grad_shape)

    # Imported here rather than at module scope: both backward modules import
    # this one, so a top-level import would be a cycle.  By the time this runs
    # they are ordinary already-initialized modules.
    from .bwd_data import is_supported_bwd_data
    from .reduce_gemm import is_supported_bwd_weight

    if not is_supported_bwd_data(
        grad, w, tuple(x.shape), stride, padding, dilation, groups
    ):
        return False
    return bool(
        is_supported_bwd_weight(
            x, tuple(w.shape), grad, stride, padding, dilation, groups
        )
    )


def _check_out(y: torch.Tensor, shape: tuple[int, ...], like: torch.Tensor) -> None:
    """Reject an ``out=`` the kernel would write outside of, or write wrongly.

    Nothing downstream catches either failure.  The grid is sized from the
    *problem* and not from ``out``, and the store addresses come from
    ``out.stride(0/2/3/4)`` with a channel stride of 1 assumed -- so an
    undersized buffer is an out-of-bounds device write (1920 elements into a
    128-element allocation, observed, with no error), and an NCDHW buffer is a
    full-rate kernel that returns a scrambled answer.

    The shape is compared explicitly rather than inferred from the strides.
    ``reduce_gemm._layout_ok`` checks strides alone and cannot see ``Cout`` --
    none of the five channels-last strides depends on it -- so a buffer built
    for a different output width has byte-identical strides and passes.
    """
    if tuple(y.shape) != tuple(shape):
        raise ValueError(f"out= must have shape {tuple(shape)}, got {tuple(y.shape)}")
    if y.dtype != like.dtype:
        raise ValueError(f"out= must have dtype {like.dtype}, got {y.dtype}")
    if y.device != like.device:
        raise ValueError(f"out= must be on {like.device}, got {y.device}")
    if not y.is_contiguous(memory_format=torch.channels_last_3d):
        raise ValueError(
            "out= must have channels_last_3d strides -- the store addressing "
            f"assumes a channel stride of 1; got {tuple(y.stride())}"
        )


def _check_weight_rsck(
    wr: torch.Tensor, shape: tuple[int, ...], like: torch.Tensor
) -> None:
    """Reject a hoisted weight that is not the one this call needs.

    ``weight_rsck`` supplies every weight *value* the kernel reads -- ``w`` is
    then consulted only for its shape -- so a wrong one is a smooth, correctly
    shaped, entirely wrong result.  That is a live hazard rather than a
    "you asked for it": the transform is meant to be cached across calls, and a
    cache keyed on the parameter's version is exactly the thing that can go
    stale without changing shape.

    Checked against this tensor's own shape, never against ``w``'s strides:
    :mod:`~triton_conv3d.bwd_data` deliberately passes a permuted *view* as
    ``w`` and supplies the values through here.
    """
    if tuple(wr.shape) != tuple(shape):
        raise ValueError(
            f"weight_rsck= must have shape {tuple(shape)} (kd, kh, kw, Cin, "
            f"Cout), got {tuple(wr.shape)}"
        )
    if wr.dtype != like.dtype:
        raise ValueError(f"weight_rsck= must have dtype {like.dtype}, got {wr.dtype}")
    if wr.device != like.device:
        raise ValueError(f"weight_rsck= must be on {like.device}, got {wr.device}")
    if not wr.is_contiguous():
        raise ValueError(
            "weight_rsck= must be contiguous -- the B tile is loaded as a "
            f"contiguous vector along Cout; got strides {tuple(wr.stride())}"
        )


def _index_dtype(*operands: torch.Tensor):
    """``tl.int64`` offsets, and only for the shapes that need them.

    They are not free -- the AMD backend's buffer-load path requires an i32
    offset tensor -- but triton 3.7.1 narrows i64 offsets it can prove safe, so
    the cost is paid only where the storage really is over 2 GiB.  Storage size,
    not offset dtype, is the lever, and it is not one the kernel controls.

    *Every* operand the kernel indexes has to be passed here, the weight
    included -- it is the one this decision used to omit, on an assumption
    ("weights are never that large") that nothing enforced.  ``numel`` is the
    right quantity for each: the largest element offset a contiguous operand
    sees is ``numel - 1``, and offsets computed for masked-off lanes can exceed
    it but are never dereferenced.
    """
    return tl.int64 if max(t.numel() for t in operands) > 2**31 - 1 else tl.int32


def conv3d_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    *,
    config: ConvConfig | None = None,
    weight_rsck: torch.Tensor | None = None,
    weight_flip: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward 3-D convolution.  Input and output are ``channels_last_3d``.

    **The weight is read where it lies**, decided from its strides.  A
    ``channels_last_3d`` parameter -- which is what a ScaFFold model's weights
    already are, since ``worker.py`` moves the whole model to that format --
    costs *no* weight transform at all.  That matters because the transform was
    per optimizer step rather than per call: the optimizer dirties every
    parameter every step, so no amount of caching removed it.  A weight in
    PyTorch's *default* layout is still copied, and has to be; see
    :func:`_weight_plan`.

    ``weight_rsck`` remains for a caller who has an RSCK buffer already, and is
    a wash against reading the parameter (0.96-1.10x per site, both directions).
    It is checked rather than trusted, since it supplies every weight value the
    kernel reads and ``w`` is then consulted only for its shape.

    ``weight_flip`` consumes the taps in reverse.  It exists for
    :mod:`~triton_conv3d.bwd_data`, whose gather is the forward's with the taps
    flipped: doing it with a constexpr index rather than a ``torch.flip`` copy is
    what lets backward-data share the forward's weight buffer instead of
    materializing a second one.

    ``out=`` is checked rather than trusted for the same reason as
    ``weight_rsck``: the kernel writes it with addressing derived from *this*
    call's shapes, so a mismatched one is an out-of-bounds write.  See
    :func:`_check_out` and :func:`_check_weight_rsck`.
    """
    if not is_supported(x, w, bias, stride, padding, dilation, groups):
        raise NotImplementedError(
            f"unsupported: x={tuple(x.shape)}/{x.dtype} w={tuple(w.shape)} "
            f"stride={stride} padding={padding} dilation={dilation} groups={groups}"
        )
    sd, sh, sw = _triple(stride, "stride")
    pd, ph, pw = _triple(padding, "padding")
    dd, dh, dw = _triple(dilation, "dilation")
    kd, kh, kw = (int(v) for v in w.shape[2:])

    # NDHWC is not a preference here, it is the layout the addressing assumes.
    x = x.contiguous(memory_format=torch.channels_last_3d)
    n, cin, in_d, in_h, in_w = (int(v) for v in x.shape)
    cout = int(w.shape[0])
    out_d = (in_d + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    out_h = (in_h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (in_w + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    y_shape = (n, cout, out_d, out_h, out_w)
    if out is None:
        # One allocation, already in the layout the kernel stores into.  Spelling
        # it ``torch.empty(shape).contiguous(memory_format=...)`` allocates NCDHW
        # and then copies the whole thing: 2.82 ms against 0.012 ms on a 256 MiB
        # output, 235x, on a path a training step takes about 19 times.
        y = torch.empty(
            y_shape,
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d,
        )
    else:
        y = out
        _check_out(y, y_shape, x)
    if weight_rsck is not None:
        wr = weight_rsck
        _check_weight_rsck(wr, (kd, kh, kw, cin, cout), x)
        # RSCK is contiguous by the check above, so the strides are exactly these.
        plan = (_W_N_CONTIG, wr.stride(2), wr.stride(3), 1)
    else:
        plan = _weight_plan(w)
        if plan is None:
            # The only path left that copies the weight; see :func:`_weight_plan`.
            wr = to_rsck(w)
            plan = (_W_N_CONTIG, wr.stride(2), wr.stride(3), 1)
        else:
            wr = w

    m_total = n * out_d * out_h * out_w
    if config is None:
        config = select_config(m_total, cin, cout, (kd, kh, kw), x.dtype)
    why = config.validate(x.dtype)
    if why is not None:
        raise ValueError(f"illegal config {config}: {why}")

    index_dtype = _index_dtype(x, y, wr)

    block_k_count = triton.cdiv(cin, config.BLOCK_K)
    grid = (triton.cdiv(m_total, config.BLOCK_M) * triton.cdiv(cout, config.BLOCK_N),)
    _conv3d_fwd_kernel[grid](
        x,
        wr,
        y,
        bias,
        n,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        cin,
        cout,
        m_total,
        x.stride(0),
        x.stride(2),
        x.stride(3),
        x.stride(4),
        plan[1],
        plan[2],
        plan[3],
        y.stride(0),
        y.stride(2),
        y.stride(3),
        y.stride(4),
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
        BLOCK_K_COUNT=block_k_count,
        GROUP_M=config.GROUP_M,
        HAS_BIAS=bias is not None,
        EVEN_K=(cin % config.BLOCK_K == 0),
        EVEN_N=(cout % config.BLOCK_N == 0),
        PADDED=(pd > 0 or ph > 0 or pw > 0),
        INDEX_DTYPE=index_dtype,
        INPUT_PRECISION="ieee",
        W_ORDER=plan[0],
        W_FLIP=bool(weight_flip),
        **config.launch_kwargs(),
    )
    return y


# ---------------------------------------------------------------------------
# ISA verification
# ---------------------------------------------------------------------------


def verify_isa(
    problem_shape: Sequence[int] | None = None,
    direction: str = "fwd",
    config: "ConvConfig | None" = None,
    padding: int = 1,
    kernel: int = 3,
    weight_layout: str = "channels_last",
) -> None:  # pragma: no cover
    """Compile and launch one configuration so its ISA can be inspected.

    Run under ``AMDGCN_ENABLE_DUMP=1`` with a **cold** ``TRITON_CACHE_DIR``: a
    cache hit skips the compile and therefore the dump, and an empty grep then
    looks exactly like a kernel with no MFMA in it.  The other trap is the
    mnemonic -- the emitted instruction is ``v_mfma_f32_16x16x16_bf16`` with no
    ``_1k`` suffix even though Triton's internal table entry is named ``_1k``, so
    grepping for ``_1k`` reports zero on a healthy kernel.

    ``direction="bwd-data"`` runs the same kernel through
    :func:`~triton_conv3d.bwd_data.conv3d_backward_data`.  It is the *same*
    ``@triton.jit`` function, so a reader could reasonably ask why it needs
    checking again: because the constexprs differ.  Backward-data's ``PADDED``
    is true where the halo'd forward's is false, its ``EVEN_K``/``EVEN_N`` are
    computed from the swapped channel widths, and its tile comes from a
    different table -- and every one of those changes the code that is emitted.

    ``weight_layout`` selects which of the three B loads is compiled, and it has
    to be gated separately for the same reason: ``W_ORDER`` is a constexpr, and
    ``channels_last`` (the shipped path, a transposing load) and ``rsck`` (a
    hoisted buffer, a straight load) emit different instructions for the operand
    that feeds the matrix core.
    """
    n, cin, cout, d, h, wd = problem_shape or (1, 64, 64, 32, 64, 64)
    k = (kernel, kernel, kernel)
    w = torch.randn((cout, cin, *k), device="cuda", dtype=torch.bfloat16)
    if weight_layout == "channels_last":
        w = w.contiguous(memory_format=torch.channels_last_3d)
    elif weight_layout not in ("rsck", "contiguous"):
        raise ValueError(f"unknown weight_layout {weight_layout!r}")
    rsck = to_rsck(w) if weight_layout == "rsck" else None
    if direction == "fwd":
        x = torch.randn(
            (n, cin, d, h, wd), device="cuda", dtype=torch.bfloat16
        ).contiguous(memory_format=torch.channels_last_3d)
        cfg = config or default_config(n * d * h * wd, cin, cout, torch.bfloat16)
        y = conv3d_forward(x, w, padding=padding, config=cfg, weight_rsck=rsck)
        big = x
    elif direction == "bwd-data":
        # Local import: bwd_data imports this module, so a top-level import here
        # would be a cycle.  It is a wrapper over this file's kernel, not a peer.
        from .bwd_data import bwd_data_config, conv3d_backward_data

        out = tuple(v + 2 * padding - (kernel - 1) for v in (d, h, wd))
        gy = torch.randn(
            (n, cout, *out), device="cuda", dtype=torch.bfloat16
        ).contiguous(memory_format=torch.channels_last_3d)
        cfg = config or bwd_data_config(
            gy.shape, cin, k, torch.bfloat16, padding=padding
        )
        y = conv3d_backward_data(
            gy,
            w,
            (n, cin, d, h, wd),
            padding=padding,
            config=cfg,
            weight_rsck=rsck,
        )
        big = gy
    else:
        raise ValueError(f"unknown direction {direction!r}")
    torch.cuda.synchronize()
    print(
        f"ISA-DUMP-CONFIG [{direction}/{weight_layout}] {cfg} cin={cin} cout={cout} "
        f"spatial={(d, h, wd)} k={kernel} pad={padding} "
        f"x_storage={big.untyped_storage().size()} "
        f"y_storage={y.untyped_storage().size()}"
    )
