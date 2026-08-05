# SPDX-License-Identifier: (Apache-2.0)
"""M0 gate: what can ``tl.dot`` actually reach on this device, at our shapes?

Every convolution direction reduces to a GEMM.  Before writing a gather-GEMM
convolution it is worth knowing the ceiling of the thing it is built on, because
no amount of clever addressing recovers throughput the matrix core never had.

Three measurements, in increasing specificity:

``peak``
    A large square GEMM.  Calibrates Triton against ``torch.matmul``
    (hipBLASLt) and against the 600 TFLOP/s bf16 constant, so every later
    percentage has a known reference.

``compute``
    The conv-implied ``(M, N, K)``, but with ``A``'s M-stride set to zero so
    every row tile reads the same cached rows.  The FLOP count is unchanged and
    DRAM traffic is negligible, which isolates matrix-core throughput in the
    *shape regime* our convolutions live in -- skinny ``N`` (64-512), long ``K``
    (81-27,648), enormous ``M``.  This is the real ceiling for a fused kernel:
    a convolution reads its input once, so it is compute-bound wherever its
    arithmetic intensity exceeds the 182 FLOP/byte crossover, which is almost
    everywhere in the corpus.

``dram``
    The same shape with real strides.  Always slower, and *not* a ceiling for
    the convolution -- materializing im2col multiplies ``A``'s bytes by the tap
    count, which is exactly the traffic a fused kernel avoids.  Measured anyway,
    because the gap between ``compute`` and ``dram`` is how much a fused kernel
    stands to gain over the explicit-GEMM approach.

Usage::

    python -m triton_conv3d.bench.gemm_probe --mode peak
    python -m triton_conv3d.bench.gemm_probe --top 8 --out probe.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import pathlib
import sys
import time

import torch
import triton
import triton.language as tl

from ..shapes import (
    DIRECTIONS,
    HBM_BYTES_PER_S,
    PEAK_FLOPS,
    ConvProblem,
    hot_corpus,
)
from .harness import flush_caches, format_table, interleaved


# ---------------------------------------------------------------------------
# A plain, honest GEMM
# ---------------------------------------------------------------------------


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    INT32_OFF: tl.constexpr,
):
    """Textbook tiled GEMM with grouped-M ordering and optional split-K.

    Deliberately unremarkable: the point of the probe is to measure what a
    competent-but-ordinary Triton GEMM achieves, so that a later convolution
    kernel's number can be read as "this much of the available throughput"
    rather than against an unknown.
    """
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # Group consecutive programs along M so that a group shares B tiles in L2.
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # ``INT32_OFF`` keeps every offset tensor ``i32``, which is condition 2 of
    # the buffer-load fast path (briefing 1.3): ``canUseBufferOps`` bails out
    # with ``if (ofstBit != 32) return false;`` before it ever looks at the
    # range.  The int64 form below is the safe default -- ``M`` reaches 8.4M
    # here and ``M * stride`` overflows i32 for the real strides -- so the flag
    # exists to *measure* what the promotion costs, not to be switched on
    # blindly.
    if INT32_OFF:
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    else:
        a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :].to(tl.int64) * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_step = BLOCK_K * SPLIT_K
    for k in range(pid_k * BLOCK_K, K, k_step):
        k_mask = offs_k[None, :] + k - pid_k * BLOCK_K < K - k + pid_k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] + k - pid_k * BLOCK_K < K - k + pid_k * BLOCK_K)
            & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
        a_ptrs += k_step * stride_ak
        b_ptrs += k_step * stride_bk

    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)


@dataclasses.dataclass(frozen=True)
class GemmConfig:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    GROUP_M: int = 8
    SPLIT_K: int = 1
    num_warps: int = 8
    num_stages: int = 2
    #: AMD backend kernargs.  ``None`` means "do not pass it at all", which is
    #: not the same as passing 0: the M0 sweep never passed them, so keeping a
    #: distinct sentinel lets the old numbers be reproduced exactly.
    matrix_instr_nonkdim: int | None = None
    kpack: int | None = None
    waves_per_eu: int | None = None
    int32_offsets: bool = False

    def __str__(self) -> str:
        s = (
            f"{self.BLOCK_M}x{self.BLOCK_N}x{self.BLOCK_K}"
            f"/g{self.GROUP_M}/sk{self.SPLIT_K}/w{self.num_warps}/s{self.num_stages}"
        )
        if self.matrix_instr_nonkdim is not None:
            s += f"/nk{self.matrix_instr_nonkdim}"
        if self.kpack is not None:
            s += f"/kp{self.kpack}"
        if self.waves_per_eu:
            s += f"/we{self.waves_per_eu}"
        if self.int32_offsets:
            s += "/i32"
        return s

    def amd_kwargs(self) -> dict:
        kw = {}
        if self.matrix_instr_nonkdim is not None:
            kw["matrix_instr_nonkdim"] = self.matrix_instr_nonkdim
        if self.kpack is not None:
            kw["kpack"] = self.kpack
        if self.waves_per_eu is not None:
            kw["waves_per_eu"] = self.waves_per_eu
        return kw


#: Curated tile shapes rather than a full product sweep.  A product over
#: (BM, BN, BK, warps, stages) is ~250 configs per shape, and every one costs a
#: JIT compile; at 57 problems x 3 directions that is hours of compilation to
#: answer a yes/no question.  These are the shapes that matter on CDNA3: square
#: tiles for balanced GEMMs, tall-skinny tiles for the huge-M/small-N regime the
#: convolutions actually live in, and a couple of small tiles for the 8^3 sites.
_TILES: tuple[tuple[int, int, int], ...] = (
    (256, 128, 64),
    (256, 64, 64),
    (128, 256, 64),
    (128, 128, 64),
    (128, 128, 32),
    (128, 64, 64),
    (128, 64, 128),
    (64, 128, 64),
    (64, 64, 64),
    (64, 64, 128),
    (32, 64, 128),
    (64, 32, 128),
)

#: PyTorch Inductor's ROCm convolution seed grid, ``(BLOCK_M, BLOCK_N, BLOCK_K,
#: num_warps)``.  Verbatim from ``torch/_inductor/heuristics/template/triton.py``
#: (``BaseConfigHeuristic.conv_configs``, which ``ROCmConfigHeuristic``
#: inherits); its per-config ``num_stages`` is dropped because
#: ``ROCmConfigHeuristic._filter_configs`` force-overwrites it with
#: ``get_backend_num_stages()`` == 2 on HIP.  Preferred over a blind sweep
#: because these values are already tuned on ROCm.
_ROCM_CONV_TILES: tuple[tuple[int, int, int, int], ...] = (
    (64, 256, 16, 4),
    (256, 64, 16, 4),
    (1024, 16, 16, 8),
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

#: Extra tiles for the skinny-N regime, where the seed grid runs out of shapes.
#: At ``N=64`` a ``BLOCK_N=64`` tile is only four 16x16 MFMA tiles wide, so the
#: N axis cannot absorb warps; these trade N width for M depth (and, at
#: ``BLOCK_N=32``, test whether going *narrower* and taller helps at all).
_SKINNY_N_TILES: tuple[tuple[int, int, int, int], ...] = (
    (256, 64, 64, 8),
    (512, 64, 64, 8),
    (256, 64, 128, 8),
    (512, 64, 32, 8),
    (128, 64, 128, 4),
    (64, 64, 64, 4),
    (64, 64, 128, 4),
    (256, 32, 128, 8),
    (512, 32, 64, 8),
    (128, 32, 128, 4),
    (1024, 64, 32, 8),
)

#: MFMA k-dimension per ``matrix_instr_nonkdim`` for bf16 on gfx942
#: (``mfmaVersion == 3``).  ``BLOCK_K`` must be a multiple of this or
#: ``chooseMfmaInstruction`` fails outright ("would introduce data duplication")
#: and the dot silently lowers to FMA.  Source: Triton v3.7.0
#: ``MfmaGroup.cpp`` (``TRITON_MFMA_v(3, 16, 16, bf16T, bf16T,
#: mfma_f32_16x16x16bf16_1k, 16, 4)`` and the 32x32x8 entry) plus the
#: ``inputKSize % kDim`` check in ``AccelerateAMDMatmul.cpp``.
_MFMA_KDIM_BF16 = {16: 16, 32: 8}


def _default_kpack(block_k: int) -> int:
    """Inductor's arch-aware kpack default, ``get_default_kpack`` in utils.py.

    ``kWidth = kBase * kPack``; ``kpack=2`` means ``ds_read_b128``, the widest
    LDS load.  On gfx942 Inductor keeps it at 1 for ``BLOCK_K <= 16``, where the
    wider read has nothing to read.
    """
    return 1 if block_k <= 16 else 2


def _amd_config(bm: int, bn: int, bk: int, warps: int, *, group_m: int,
                split_k: int, nonkdim: int, kpack: int | None = None,
                waves_per_eu: int = 0, int32: bool = False) -> GemmConfig | None:
    """One AMD-knob config, or ``None`` if a hard constraint rejects it.

    The constraints are not preferences.  ``BLOCK_M``/``BLOCK_N`` not divisible
    by ``nonkdim`` is what Inductor's ``_finalize_mm_configs`` prunes; a
    ``BLOCK_K`` that is not a multiple of the intrinsic's ``kDim`` is what the
    Triton pass rejects.  Both failure modes are *silent* -- the kernel still
    runs, just on the FMA path -- so a config that violates them would quietly
    contribute a meaningless number to a best-of sweep.
    """
    if nonkdim and (bm % nonkdim or bn % nonkdim):
        return None
    if bk % _MFMA_KDIM_BF16.get(nonkdim, 16):
        return None
    # Each warp owns a 16x16 tile; more warps than tiles leaves warps idle.
    warps = min(warps, bm * bn // 256)
    if warps < 1:
        return None
    return GemmConfig(
        bm, bn, bk, group_m, split_k, warps, num_stages=2,
        matrix_instr_nonkdim=nonkdim,
        kpack=_default_kpack(bk) if kpack is None else kpack,
        waves_per_eu=waves_per_eu,
        int32_offsets=int32,
    )


def _split_ks(m: int, n: int, k: int, bm: int, bn: int) -> tuple[int, ...]:
    """Split-K only where the M/N grid cannot fill the 228 CUs on its own."""
    tiles = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
    return (1, 4, 16) if tiles < 228 and k >= 2048 else (1,)


def _candidate_configs(m: int, n: int, k: int, knobs: str = "legacy") -> list[GemmConfig]:
    """Configs worth trying for one shape.

    Not ``@triton.autotune``: that recompiles inside whatever is running at the
    time, which is the wrong behaviour both here (it would pollute the timing)
    and in production (ScaFFold's figure of merit is total wall time).

    ``knobs``:

    ``legacy``
        The M0 sweep: curated tiles, ``GROUP_M=8``, no AMD kernargs at all.
        Kept verbatim so the earlier numbers stay reproducible.

    ``amd``
        Inductor's ROCm conv seed grid under the gfx942 constraints, with
        ``matrix_instr_nonkdim=16`` and the arch-aware ``kpack``.  ``GROUP_M``
        sweeps 6 (MI300A has 6 XCDs; AMD's L2-swizzle rule is "multiple of the
        XCD count") against the MI300X-derived 8 that the M0 sweep used, so the
        report can say which actually won rather than assuming.

    ``amd-wide``
        ``amd`` plus the skinny-N tiles, ``matrix_instr_nonkdim`` in
        ``{16, 32}`` and ``GROUP_M`` in ``{6, 8, 12}``.  For the ``N=64``
        interrogation; several times the configs, so not the default.
    """
    m2 = max(32, triton.next_power_of_2(m))
    n2 = max(16, triton.next_power_of_2(n))
    k2 = max(32, triton.next_power_of_2(k))

    def oversized(bm: int, bn: int, bk: int) -> bool:
        # Skip tiles that mostly compute padding.
        return bm > 2 * m2 or bn > 2 * n2 or bk > 2 * k2

    out: list[GemmConfig] = []
    if knobs == "legacy":
        for bm, bn, bk in _TILES:
            if oversized(bm, bn, bk):
                continue
            for warps in (4, 8):
                for sk in _split_ks(m, n, k, bm, bn):
                    out.append(GemmConfig(bm, bn, bk, 8, sk, warps, num_stages=2))
        return out

    if knobs == "amd":
        tiles, nonkdims, group_ms = _ROCM_CONV_TILES, (16,), (6, 8)
    elif knobs == "amd-wide":
        tiles = _ROCM_CONV_TILES + _SKINNY_N_TILES
        nonkdims, group_ms = (16, 32), (6, 8, 12)
    else:
        raise ValueError(f"unknown knob set {knobs!r}")

    seen: set[GemmConfig] = set()
    for bm, bn, bk, seed_warps in tiles:
        if oversized(bm, bn, bk):
            continue
        # Sweep both warp counts, then clamp; Inductor ships one value per tile
        # but we are measuring a ceiling, not reproducing its choice.
        for warps in {4, 8, seed_warps}:
            for nonkdim in nonkdims:
                for group_m in group_ms:
                    for sk in _split_ks(m, n, k, bm, bn):
                        cfg = _amd_config(bm, bn, bk, warps, group_m=group_m,
                                          split_k=sk, nonkdim=nonkdim)
                        if cfg is not None and cfg not in seen:
                            seen.add(cfg)
                            out.append(cfg)
    return out


def _randn(shape: tuple[int, ...], device, dtype: torch.dtype) -> torch.Tensor:
    """Random operand allocated directly in ``dtype``.

    Going through fp32 and casting doubles peak memory, which for the largest
    corpus shapes means a 29 GiB operand briefly needs 87 GiB and the allocator
    spends minutes thrashing before it gets there.
    """
    return torch.randn(shape, device=device, dtype=dtype)


def _launch(a, b, c, cfg: GemmConfig, *, m: int, n: int, k: int,
            stride_am: int, stride_bn: int):
    grid = (triton.cdiv(m, cfg.BLOCK_M) * triton.cdiv(n, cfg.BLOCK_N), cfg.SPLIT_K)
    _gemm_kernel[grid](
        a, b, c,
        m, n, k,
        stride_am, 1,
        b.stride(0), stride_bn,
        c.stride(0), c.stride(1),
        BLOCK_M=cfg.BLOCK_M, BLOCK_N=cfg.BLOCK_N, BLOCK_K=cfg.BLOCK_K,
        GROUP_M=cfg.GROUP_M, SPLIT_K=cfg.SPLIT_K,
        INT32_OFF=cfg.int32_offsets,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
        **cfg.amd_kwargs(),
    )


#: Refuse to allocate an operand bigger than this in ``compute`` mode; above it
#: the operand is replaced by a stride-0 broadcast of the reduction axis.  1 GiB
#: comfortably exceeds MI300A's 256 MiB last level, so anything under it streams
#: from cache and anything over it would be measuring DRAM instead of the
#: matrix core.
_RESIDENT_BUDGET = 1 << 30
#: Refuse a ``dram``-mode shape whose materialized operands would not fit.
_DRAM_BUDGET = 24 << 30


def plan_operands(m: int, n: int, k: int, mode: str, elem: int) -> dict | None:
    """Decide how to allocate A and B, or ``None`` if the shape cannot be run.

    In ``compute`` mode an operand's non-reduction axis gets stride 0 whenever
    materializing it would spill out of cache: the same rows (or columns) are
    re-read by every tile.  The FLOP count is untouched, so the measured rate is
    matrix-core throughput at this ``(M, N, K)`` with DRAM taken out of the
    picture -- which is the ceiling a *fused* convolution kernel is entitled to
    aim at, since it reads its input once rather than ``tap_count`` times.
    """
    a_bytes, b_bytes, c_bytes = m * k * elem, k * n * elem, m * n * 4
    if mode == "dram":
        if a_bytes + b_bytes + c_bytes > _DRAM_BUDGET:
            return None
        return {"a_rows": m, "stride_am": None, "b_cols": n, "stride_bn": None}
    plan = {"a_rows": m, "stride_am": None, "b_cols": n, "stride_bn": None}
    if a_bytes > _RESIDENT_BUDGET:
        plan["a_rows"], plan["stride_am"] = min(m, 256), 0
    if b_bytes > _RESIDENT_BUDGET:
        # Broadcast one column across N.  Only reachable for backward-weight,
        # where K is the whole volume and B is the im2col'd activation -- the
        # 58 GiB tensor a fused kernel never builds.
        plan["b_cols"], plan["stride_bn"] = 1, 0
    if c_bytes > _DRAM_BUDGET:
        return None
    return plan


def best_triton_gemm(
    m: int, n: int, k: int, *, dtype=torch.bfloat16, mode: str = "compute",
    device="cuda", max_configs: int = 0, verbose: bool = False,
    knobs: str = "legacy", sink: list | None = None,
) -> tuple[float, GemmConfig | None, int]:
    """Sweep configs, return ``(best ms/call, config, n_configs_that_ran)``.

    ``float('inf')`` with a ``None`` config means the shape could not be run at
    all in this mode -- which for ``dram`` is itself the finding.

    ``sink``, if given, collects ``(config string, ms)`` for every config that
    ran.  Only the winner is reported normally, but "which knob won, and by how
    much over second place" is the question the N=64 investigation asks, and it
    is not answerable from a single best time.
    """
    plan = plan_operands(m, n, k, mode, torch.finfo(dtype).bits // 8)
    if plan is None:
        return float("inf"), None, 0

    a, b, c, stride_am, stride_bn = _alloc(plan, m, n, k, device, dtype)
    configs = _candidate_configs(m, n, k, knobs)
    if max_configs:
        configs = configs[:max_configs]
    try:
        return _sweep(a, b, c, configs, m=m, n=n, k=k, stride_am=stride_am,
                      stride_bn=stride_bn, verbose=verbose, sink=sink)
    finally:
        del a, b, c
        torch.cuda.empty_cache()


def _alloc(plan: dict, m: int, n: int, k: int, device, dtype):
    a = _randn((plan["a_rows"], k), device, dtype)
    b = _randn((k, plan["b_cols"]), device, dtype)
    c = torch.empty((m, n), device=device, dtype=torch.float32)
    return (a, b, c,
            a.stride(0) if plan["stride_am"] is None else 0,
            b.stride(1) if plan["stride_bn"] is None else 0)


def _sweep(a, b, c, configs: list[GemmConfig], *, m: int, n: int, k: int,
           stride_am: int, stride_bn: int, verbose: bool = False,
           sink: list | None = None) -> tuple[float, GemmConfig | None, int]:
    """Time each config on already-allocated operands; return the winner."""
    best_ms, best_cfg, ran = float("inf"), None, 0
    for cfg in configs:
        launch = _launcher(a, b, c, cfg, m=m, n=n, k=k,
                           stride_am=stride_am, stride_bn=stride_bn)
        try:
            if cfg.SPLIT_K > 1:
                c.zero_()
            launch()
            torch.cuda.synchronize()
        except Exception as exc:  # OOM, LDS overflow, unsupported tiling
            if verbose:
                print(f"    skip {cfg}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        ran += 1
        meas = interleaved({"t": launch}, warmup=2, iters=3, rounds=3)["t"]
        if sink is not None:
            sink.append((str(cfg), meas.median))
        if meas.median < best_ms:
            best_ms, best_cfg = meas.median, cfg
    return best_ms, best_cfg, ran


def _launcher(a, b, c, cfg: GemmConfig, **kw):
    return lambda: _launch(a, b, c, cfg, **kw)


def torch_gemm_ms(m: int, n: int, k: int, *, dtype=torch.bfloat16, device="cuda") -> float:
    """hipBLASLt's time for the same GEMM -- the library reference."""
    a = _randn((m, k), device, dtype)
    b = _randn((k, n), device, dtype)
    try:
        meas = interleaved({"t": lambda: torch.matmul(a, b)}, warmup=5, iters=5, rounds=5)
        return meas["t"].median
    finally:
        del a, b
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_peak(sizes=(2048, 4096, 8192), dtype=torch.bfloat16,
             knobs: str = "legacy") -> list[dict]:
    peak = PEAK_FLOPS["bf16"] if dtype is torch.bfloat16 else PEAK_FLOPS["fp32"]
    rows = []
    for s in sizes:
        flops = 2 * s * s * s
        tri_ms, cfg, ran = best_triton_gemm(s, s, s, dtype=dtype, mode="dram",
                                            knobs=knobs)
        tor_ms = torch_gemm_ms(s, s, s, dtype=dtype)
        rows.append({
            "size": s,
            "triton_ms": tri_ms, "triton_tflops": flops / (tri_ms * 1e-3) / 1e12,
            "triton_pct_peak": 100 * flops / (tri_ms * 1e-3) / peak,
            "triton_config": str(cfg), "configs_ran": ran,
            "torch_ms": tor_ms, "torch_tflops": flops / (tor_ms * 1e-3) / 1e12,
            "torch_pct_peak": 100 * flops / (tor_ms * 1e-3) / peak,
            "triton_vs_torch": tor_ms / tri_ms,
        })
        print(format_table(
            [[
                r["size"],
                f"{r['triton_tflops']:.1f}", f"{r['triton_pct_peak']:.1f}%",
                f"{r['torch_tflops']:.1f}", f"{r['torch_pct_peak']:.1f}%",
                f"{r['triton_vs_torch']:.2f}x", r["triton_config"],
            ] for r in rows[-1:]],
            ["MNK", "triton TF/s", "%peak", "torch TF/s", "%peak", "tri/torch", "config"],
            aligns="rrrrrrl",
        ) if len(rows) == 1 else "  " + "  ".join([
            str(rows[-1]["size"]),
            f"{rows[-1]['triton_tflops']:.1f}", f"{rows[-1]['triton_pct_peak']:.1f}%",
            f"{rows[-1]['torch_tflops']:.1f}", f"{rows[-1]['torch_pct_peak']:.1f}%",
            f"{rows[-1]['triton_vs_torch']:.2f}x", rows[-1]["triton_config"],
        ]))
        sys.stdout.flush()
    return rows


def run_peak_compare(sizes=(2048, 4096, 8192), knob_sets=("legacy", "amd"),
                     dtype=torch.bfloat16, device="cuda") -> list[dict]:
    """Peak calibration where the knob sets are compared *against each other*.

    :func:`run_peak` sweeps one knob set and reports its winner, which is fine
    for a single number but useless for a before/after: two sweeps run minutes
    apart are two different machines.  Measured here, three passes over the same
    ``legacy`` grid spanned 386-428 TF/s at 8192 -- an 11% band with no code
    change at all, which is larger than most knob effects we are looking for.

    So: sweep each knob set to find its own champion, then put the champions
    (and hipBLASLt) into a single :func:`interleaved` call.  Drift then hits
    every variant equally and lands in the reported spread instead of in the
    conclusion.  The sweep-time numbers are kept alongside as ``*_sweep_tflops``
    precisely so the size of that effect stays visible.
    """
    peak = PEAK_FLOPS["bf16"] if dtype is torch.bfloat16 else PEAK_FLOPS["fp32"]
    rows = []
    for s in sizes:
        flops = 2 * s * s * s
        plan = plan_operands(s, s, s, "dram", torch.finfo(dtype).bits // 8)
        assert plan is not None
        a, b, c, stride_am, stride_bn = _alloc(plan, s, s, s, device, dtype)
        try:
            variants: dict = {}
            owner: dict = {}
            row: dict = {"size": s}
            for ks in knob_sets:
                configs = _candidate_configs(s, s, s, ks)
                sink: list = []
                ms, cfg, ran = _sweep(a, b, c, configs, m=s, n=s, k=s,
                                      stride_am=stride_am, stride_bn=stride_bn,
                                      sink=sink)
                row[f"{ks}_config"] = str(cfg)
                row[f"{ks}_configs_ran"] = ran
                row[f"{ks}_sweep_tflops"] = flops / (ms * 1e-3) / 1e12
                by_str = {str(x): x for x in configs}
                for i, (name, _) in enumerate(
                        sorted(sink, key=lambda kv: kv[1])[:_FINALISTS]):
                    owner[f"{ks}#{i}"] = (ks, by_str[name])
                    variants[f"{ks}#{i}"] = _launcher(
                        a, b, c, by_str[name], m=s, n=s, k=s,
                        stride_am=stride_am, stride_bn=stride_bn)
            owner["torch"] = ("torch", None)
            variants["torch"] = lambda: torch.matmul(a, b)
            meas = interleaved(variants, warmup=5, iters=5,
                               rounds=2 * len(variants))
            for name, m_ in meas.items():
                ks, cfg = owner[name]
                if m_.median >= row.get(f"{ks}_ms", float("inf")):
                    continue
                row[f"{ks}_ms"] = m_.median
                row[f"{ks}_tflops"] = flops / (m_.median * 1e-3) / 1e12
                row[f"{ks}_pct_peak"] = 100 * flops / (m_.median * 1e-3) / peak
                row[f"{ks}_spread"] = m_.spread
                # Launch-gap inflation, from ``harness._time_block``.  A value
                # far above 1 means the GPU idled between launches and the
                # number is not kernel time; recorded so a reader can see
                # whether the device was actually ours for the duration.
                row[f"{ks}_stall"] = m_.stall_ratio
                if cfg is not None:
                    row[f"{ks}_config"] = str(cfg)
        finally:
            del a, b, c
            torch.cuda.empty_cache()
        rows.append(row)
        names = list(knob_sets) + ["torch"]
        print("  " + "  ".join(
            [f"MNK={s:<5d}"]
            + [f"{name}={row[f'{name}_tflops']:6.1f} TF/s"
               f" ({row[f'{name}_pct_peak']:5.1f}%, spread {row[f'{name}_spread']:.1%},"
               f" stall {row[f'{name}_stall']:.2f}x)"
               for name in names if f"{name}_tflops" in row]
        ))
        for ks in knob_sets:
            print(f"      {ks:8s} winner {row[f'{ks}_config']} "
                  f"[{row[f'{ks}_configs_ran']} configs, "
                  f"{row[f'{ks}_sweep_tflops']:.1f} TF/s during the sweep]")
        sys.stdout.flush()
    return rows


#: Named knob-set *comparisons*.  A bare knob-set name sweeps that set alone;
#: these run several sets over the same operands and finish with a single
#: interleaved head-to-head between their champions, which is the only honest
#: way to state a before/after: two sweeps minutes apart need not have shared
#: the device with the same neighbours.
_KNOB_SETS: dict[str, tuple[str, ...]] = {
    "compare": ("legacy", "amd"),
    "compare-wide": ("legacy", "amd", "amd-wide"),
}


#: How many of each knob set's fastest configs go into the run-off.
#: One would be enough if the sweep were noise-free.  It is not: a sweep reports
#: ``min`` over ~50 noisy samples, so its winner is partly the config that got
#: the luckiest sample, and re-timing that one config reproduces the luck only
#: sometimes.  Racing the top few and taking each set's best restores the
#: like-for-like comparison -- both sides get a best-of, in the same interleaved
#: measurement.
_FINALISTS = 3


def _measure_cell(m: int, n: int, k: int, *, mode: str, dtype, knob_sets,
                  keep_all: bool, max_configs: int, device="cuda") -> dict | None:
    """Sweep each knob set on one shape, then race the finalists.

    Returns ``None`` when the shape does not fit this mode's budget.  Operands
    are allocated once and shared by every knob set, so the comparison is not
    confounded by a different allocation or a differently warmed cache.
    """
    plan = plan_operands(m, n, k, mode, torch.finfo(dtype).bits // 8)
    if plan is None:
        return None
    a, b, c, stride_am, stride_bn = _alloc(plan, m, n, k, device, dtype)
    try:
        out: dict = {"winners": {}, "ran": {}, "sweep": {}, "sinks": {},
                     "headtohead": {}, "spread": {}, "stall": {}}
        finalists: dict[str, list[GemmConfig]] = {}
        for ks in knob_sets:
            configs = _candidate_configs(m, n, k, ks)
            if max_configs:
                configs = configs[:max_configs]
            sink: list = []
            ms, cfg, ran = _sweep(a, b, c, configs, m=m, n=n, k=k,
                                  stride_am=stride_am, stride_bn=stride_bn,
                                  sink=sink)
            if cfg is None:
                continue
            out["winners"][ks], out["ran"][ks], out["sweep"][ks] = cfg, ran, ms
            ranked = sorted(sink, key=lambda kv: kv[1])
            if keep_all:
                out["sinks"][ks] = ranked
            by_str = {str(x): x for x in configs}
            finalists[ks] = [by_str[name] for name, _ in ranked[:_FINALISTS]]

        if len(out["winners"]) > 1:
            variants, owner = {}, {}
            for ks, cfgs in finalists.items():
                for i, cfg in enumerate(cfgs):
                    name = f"{ks}#{i}"
                    owner[name] = ks
                    variants[name] = _launcher(a, b, c, cfg, m=m, n=n, k=k,
                                               stride_am=stride_am,
                                               stride_bn=stride_bn)
                    if cfg.SPLIT_K > 1:
                        c.zero_()
            # Rounds a multiple of the variant count, so each variant occupies
            # each slot the same number of times -- with 2 variants over 5
            # rounds one of them gets the post-warmup slot three times and the
            # other twice, which is worth several percent here.
            meas = interleaved(variants, warmup=3, iters=5,
                               rounds=2 * len(variants))
            out["h2h_best"] = {}
            for name, m_ in meas.items():
                ks = owner[name]
                out["h2h_best"][ks] = min(out["h2h_best"].get(ks, float("inf")),
                                          m_.best)
                if m_.median < out["headtohead"].get(ks, float("inf")):
                    out["headtohead"][ks] = m_.median
                    out["spread"][ks] = m_.spread
                    # See ``Measurement.stall_ratio``: how much of this cell's
                    # elapsed time was the GPU waiting for the host.  Carried
                    # into the JSON so "was the node quiet" is answerable from
                    # the artifact instead of from a contemporaneous rocm-smi.
                    out["stall"][ks] = m_.stall_ratio
                    out["winners"][ks] = finalists[ks][int(name.split("#")[1])]
        else:
            out["headtohead"] = dict(out["sweep"])
            out["h2h_best"] = dict(out["sweep"])
            out["spread"] = {ks: 0.0 for ks in out["winners"]}
            out["stall"] = {ks: float("nan") for ks in out["winners"]}
        return out
    finally:
        del a, b, c
        torch.cuda.empty_cache()


def run_shapes(problems: list[ConvProblem], modes=("compute", "dram"),
               dtype=torch.bfloat16, max_configs: int = 0,
               knobs: str = "legacy", only_n: int | None = None,
               keep_all: bool = False,
               prior: dict[tuple[str, str], dict] | None = None,
               flush=None) -> list[dict]:
    """Sweep every (problem, direction) cell.

    ``prior`` supplies cells a previous run already measured, keyed by
    ``(label, direction)``; they are carried through untouched.  ``flush``, if
    given, is called with the row list after every cell.  Both exist because a
    full ``compare-wide`` pass over the corpus is ~90 minutes and the JSON used
    to be written only at the end -- a run interrupted at cell 53 of 60 left
    nothing behind but a log, twice.
    """
    peak = PEAK_FLOPS["bf16"]
    rows = []
    for p in problems:
        for direction in DIRECTIONS:
            m, n, k = p.gemm_shape(direction)
            if only_n is not None and n != only_n:
                continue
            if prior and (p.label, direction) in prior:
                rows.append(prior[(p.label, direction)])
                print(f"  {p.label:34s} {direction:11s} [resumed]")
                sys.stdout.flush()
                if flush:
                    flush(rows)
                continue
            flops = 2 * m * n * k
            row = {
                "problem": p.label, "direction": direction,
                "M": m, "N": n, "K": k,
                "conv_flops": p.flops(direction),
                "conv_ai": p.arithmetic_intensity(direction),
                "conv_roofline_tflops": p.roofline_flops(direction) / 1e12,
            }
            best = p.measured_for(direction, config="A") or p.measured_for(direction)
            if best:
                row["miopen_ms"] = best[0]["ms_per_call"]
                row["miopen_pct_roofline"] = best[0]["pct_roofline"]
                row["miopen_solver"] = best[0]["solvers"][0]
            knob_sets = _KNOB_SETS.get(knobs, (knobs,))
            for mode in modes:
                try:
                    cell = _measure_cell(m, n, k, mode=mode, dtype=dtype,
                                         knob_sets=knob_sets, keep_all=keep_all,
                                         max_configs=max_configs)
                except torch.OutOfMemoryError:
                    cell = None
                if cell is None or not cell["winners"]:
                    # Shape not runnable in this mode.  For ``dram`` that is the
                    # finding: materialized im2col does not fit in 128 GiB.
                    row[f"{mode}_ms"] = None
                    row[f"{mode}_skipped"] = "operands exceed budget"
                    continue

                def record(prefix: str, ms: float) -> None:
                    rate = flops / (ms * 1e-3)
                    row[f"{prefix}_ms"] = ms
                    row[f"{prefix}_tflops"] = rate / 1e12
                    row[f"{prefix}_pct_peak"] = 100 * rate / peak
                    # What the convolution would take at this FLOP rate.
                    row[f"{prefix}_implied_conv_ms"] = p.flops(direction) / rate * 1e3
                    row[f"{prefix}_implied_pct_roofline"] = (
                        100 * rate / p.roofline_flops(direction)
                    )

                for ks in cell["winners"]:
                    if keep_all:
                        row[f"{mode}_{ks}_all_configs"] = cell["sinks"][ks]
                    if len(knob_sets) > 1:
                        record(f"{mode}_{ks}", cell["headtohead"][ks])
                        row[f"{mode}_{ks}_config"] = str(cell["winners"][ks])
                        row[f"{mode}_{ks}_configs_ran"] = cell["ran"][ks]
                        # Kept because it is the number the M0 run reported, and
                        # a large gap between the two is itself a finding about
                        # how long this device holds its clocks.
                        row[f"{mode}_{ks}_sweep_tflops"] = (
                            flops / (cell["sweep"][ks] * 1e-3) / 1e12
                        )
                        # Min over rounds.  This node is shared: a neighbouring
                        # job on another die drags whole rounds down by 10x and
                        # shows up as a spread in the hundreds of percent.  The
                        # median is then a measure of the neighbour, and the
                        # minimum is the best available estimate of what the
                        # kernel can actually do.
                        row[f"{mode}_{ks}_best_tflops"] = (
                            flops / (cell["h2h_best"][ks] * 1e-3) / 1e12
                        )
                        row[f"{mode}_{ks}_spread"] = cell["spread"][ks]
                        row[f"{mode}_{ks}_stall"] = cell["stall"].get(ks)
                # The headline ``{mode}_*`` keys carry the *last* knob set, which
                # is the tuned one in a comparison run and the only one in a
                # single-set run.  Keeps the JSON shape compatible with
                # ``probe.json`` so before/after can be diffed key for key.
                head = knob_sets[-1]
                record(mode, (cell["headtohead"] if len(knob_sets) > 1
                              else cell["sweep"])[head])
                row[f"{mode}_config"] = str(cell["winners"][head])
                row[f"{mode}_configs_ran"] = cell["ran"][head]
                if len(knob_sets) > 1:
                    row[f"{mode}_headtohead"] = list(knob_sets)
                    row[f"{mode}_spread"] = cell["spread"][head]
                    row[f"{mode}_stall"] = cell["stall"].get(head)
            rows.append(row)
            print(
                f"  {p.label:34s} {direction:11s} "
                f"M={m:<9d} N={n:<5d} K={k:<6d}  "
                + "  ".join(
                    f"{mode}={row.get(f'{mode}_tflops', float('nan')):6.1f} TF/s"
                    f" ({row.get(f'{mode}_pct_peak', float('nan')):5.1f}% peak)"
                    for mode in modes
                )
                + (f"  miopen={row['miopen_ms']:.3f} ms"
                   f" ({row['miopen_pct_roofline']:.0f}%)" if "miopen_ms" in row else "")
            )
            if len(knob_sets) > 1:
                for mode in modes:
                    if row.get(f"{mode}_ms") is None:
                        continue
                    print("      " + "  |  ".join(
                        f"{ks}: {row[f'{mode}_{ks}_pct_peak']:5.1f}% "
                        f"(sweep {row[f'{mode}_{ks}_sweep_tflops']:.0f} TF/s, "
                        f"stall {row[f'{mode}_{ks}_stall']:.2f}x) "
                        f"{row[f'{mode}_{ks}_config']}"
                        for ks in knob_sets if f"{mode}_{ks}_pct_peak" in row
                    ))
            sys.stdout.flush()
            if flush:
                flush(rows)
    return rows


def run_isa(spec: str, *, m: int, n: int, k: int, device="cuda",
            dtype=torch.bfloat16) -> None:
    """Compile and launch exactly one config, so its ISA can be inspected.

    Run under ``AMDGCN_ENABLE_DUMP=1`` with a cold ``TRITON_CACHE_DIR`` -- a
    cache hit skips the compile and therefore the dump, which is an easy way to
    conclude "no MFMA" from an empty grep.  Neither ``matrix_instr_nonkdim`` nor
    ``kpack`` is validated at the Python level: an illegal value falls back to
    FMA with only an MLIR remark, so this check is not optional before trusting
    a number.
    """
    parts = spec.split(",")
    if len(parts) != 8:
        raise SystemExit(
            "--isa expects BM,BN,BK,GROUP_M,warps,nonkdim,kpack,int32 "
            f"(got {spec!r})"
        )
    bm, bn, bk, gm, warps, nonkdim, kpack, int32 = (int(x) for x in parts)
    cfg = GemmConfig(bm, bn, bk, gm, 1, warps, num_stages=2,
                     matrix_instr_nonkdim=nonkdim, kpack=kpack, waves_per_eu=0,
                     int32_offsets=bool(int32))
    a = _randn((m, k), device, dtype)
    b = _randn((k, n), device, dtype)
    c = torch.empty((m, n), device=device, dtype=torch.float32)
    _launch(a, b, c, cfg, m=m, n=n, k=k,
            stride_am=a.stride(0), stride_bn=b.stride(1))
    torch.cuda.synchronize()
    print(f"ISA-DUMP-CONFIG {cfg} M={m} N={n} K={k} "
          f"a_storage={a.untyped_storage().size()} "
          f"b_storage={b.untyped_storage().size()} "
          f"c_storage={c.untyped_storage().size()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("peak", "shapes", "both", "isa"),
                    default="both")
    ap.add_argument("--knobs",
                    choices=("legacy", "amd", "amd-wide",
                             "compare", "compare-wide"),
                    default="legacy",
                    help="config generator.  'legacy' reproduces the M0 sweep "
                         "(no AMD kernargs).  'amd' is Inductor's ROCm conv "
                         "seed grid under the gfx942 MFMA constraints.  "
                         "'amd-wide' adds skinny-N tiles and sweeps "
                         "matrix_instr_nonkdim.  'compare[-wide]' runs several "
                         "sets and races their champions in one interleaved "
                         "measurement, which is the only before/after immune "
                         "to a change of tenancy on the device.")
    ap.add_argument("--only-n", type=int, default=None,
                    help="restrict to cells whose GEMM N equals this")
    ap.add_argument("--keep-all-configs", action="store_true",
                    help="record every config's time, not just the winner")
    ap.add_argument("--isa", default="128,128,64,6,8,16,2,0",
                    help="BM,BN,BK,GROUP_M,warps,nonkdim,kpack,int32 for --mode isa")
    ap.add_argument("--isa-mnk", default="4096,4096,4096",
                    help="M,N,K for --mode isa")
    ap.add_argument("--modes", default="compute",
                    help="comma-separated subset of compute,dram.  'dram' is "
                         "informative but slow -- it materializes im2col, which "
                         "for the corpus's larger shapes means a 15-30 GiB "
                         "operand per config -- and it is not the ceiling a "
                         "fused kernel is measured against.")
    ap.add_argument("--top", type=int, default=6, help="hottest N corpus problems")
    ap.add_argument("--max-configs", type=int, default=0,
                    help="cap the config sweep (0 = no cap)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="reuse cells (and the peak block) already present in "
                         "--out instead of re-measuring them")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    if args.mode == "isa":
        m, n, k = (int(x) for x in args.isa_mnk.split(","))
        run_isa(args.isa, m=m, n=n, k=k)
        return
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name}, {props.multi_processor_count} CUs, "
          f"torch {torch.__version__}, triton {triton.__version__}")
    print(f"roofline constants: {PEAK_FLOPS['bf16']/1e12:.0f} TFLOP/s bf16, "
          f"{HBM_BYTES_PER_S/1e12:.1f} TB/s HBM "
          f"(crossover at {PEAK_FLOPS['bf16']/HBM_BYTES_PER_S:.0f} FLOP/byte)\n")

    result: dict = {"device": props.name, "torch": torch.__version__,
                    "triton": triton.__version__, "knobs": args.knobs}
    prior: dict[tuple[str, str], dict] = {}
    out_path = pathlib.Path(args.out) if args.out else None
    if args.resume and out_path and out_path.exists():
        old = json.loads(out_path.read_text())
        prior = {(r["problem"], r["direction"]): r for r in old.get("shapes", [])}
        if old.get("peak"):
            result["peak"] = old["peak"]
        print(f"resuming: {len(prior)} cells already measured"
              + (", peak block reused" if "peak" in result else ""))

    def write(rows=None) -> None:
        if not out_path:
            return
        payload = dict(result)
        if rows is not None:
            payload["shapes"] = rows
        out_path.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"knob set: {args.knobs}"
          + (f", restricted to N={args.only_n}" if args.only_n else ""))
    t0 = time.time()
    if args.mode in ("peak", "both") and "peak" not in result:
        print("== peak: square bf16 GEMM, Triton vs hipBLASLt ==")
        if args.knobs in _KNOB_SETS:
            result["peak"] = run_peak_compare(knob_sets=_KNOB_SETS[args.knobs])
        else:
            result["peak"] = run_peak(knobs=args.knobs)
        write()
        print()
    if args.mode in ("shapes", "both"):
        print(f"== shapes: conv-implied GEMMs, top {args.top} corpus problems ==")
        result["shapes"] = run_shapes(list(hot_corpus(args.top)),
                                      modes=tuple(args.modes.split(",")),
                                      max_configs=args.max_configs,
                                      knobs=args.knobs, only_n=args.only_n,
                                      keep_all=args.keep_all_configs,
                                      prior=prior, flush=write)
    result["elapsed_s"] = time.time() - t0
    print(f"\nelapsed {result['elapsed_s']:.0f} s")

    if out_path:
        write(result.get("shapes"))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
