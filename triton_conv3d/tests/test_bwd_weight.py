# SPDX-License-Identifier: (Apache-2.0)
"""Tests for backward-weight, the one direction with a kernel of its own.

Three things are tested here that the other two directions do not have:

* **the split-K decomposition**, which is not an optimization but the only way
  this GEMM fills the device -- ``M = Cout`` is one or two tile rows.  Its
  correctness lives in :func:`split_count` agreeing with what the kernel and the
  reduction pass assume about each other, so the tests pin the arithmetic (every
  voxel in exactly one split) as well as the answer;
* **determinism**, tested the way the package's determinism claim is worded:
  bitwise identical run to run *and process to process*, for the
  same input, dtype, shape, device and tuning config.  There are three tests --
  in-process repetition, three separate interpreters, and a negative control on
  the atomic path -- because a determinism test that cannot fail is the most
  comfortable kind to write and the least useful;
* **the reuse that was checked and rejected.**
  :func:`test_the_forward_kernel_can_express_backward_weight` runs the algebraic
  identity that would have made this file unnecessary, and passes; the reason it
  is not used is a trip count, which the same test asserts.

The bitwise standard has the same teeth as elsewhere in this suite and the same
two guards against being vacuous: a shifted operand must fail the comparison, and
the specific bug this module can uniquely have -- writing a tap to the wrong slot
of the ``[Cout][tap][Cin]`` output -- is constructed and required to fail.
"""

from __future__ import annotations

import math
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch
import triton

from triton_conv3d import reference
from triton_conv3d.gather_gemm import conv3d_forward, default_config
from triton_conv3d.reduce_gemm import (
    _CU_COUNT,
    _MAX_EPILOGUE_FRACTION,
    _SPLIT_TARGET_WAVES,
    _WORKSPACE_BYTES,
    BwdWeightConfig,
    _row_aligned,
    bwd_weight_config,
    candidate_bwd_weight_configs,
    conv3d_backward_weight,
    default_bwd_weight_config,
    grad_weight_empty,
    is_supported_bwd_weight,
    split_count,
    workspace_elements,
)
from triton_conv3d.shapes import ConvProblem, edge_cases, scaffold_corpus

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

#: The synthetic corpus, minus the transposed upsample (a later milestone).
EDGE = [p for p in edge_cases() if not p.transposed]


def _corpus_channel_pairs() -> list[ConvProblem]:
    """Every distinct ``(Cin, Cout, kernel)`` in the corpus, at a testable volume.

    Same construction as ``test_bwd_data.py``, and it is needed for the same
    reason with a different arithmetic: backward-weight reduces over the whole
    *output volume*, so a sum of ``{-1,0,1}`` products at a real ScaFFold shape
    runs to about ``sqrt(2.1e6) = 1450`` while bf16 holds integers only to 256.
    Restating each pair at ``6x7x8`` keeps what the corpus is for -- the channel
    widths, and with them ``EVEN_M``/``EVEN_N``, the tile selection and the
    512-byte row strides -- and brings the reduction down to 336 terms, which
    bf16 does hold.

    **All three paddings** are generated, because ScaFFold issues all three
    (``shapes.py``'s module docstring): ``p=(1,1,1)`` is what the adapter hands
    the kernel unsharded and the module's own statement everywhere,
    ``p=(0,1,1)`` is what it hands the kernel at two or four shards, and
    ``p=(0,0,0)`` is what upstream DistConv hands MIOpen.  They do not differ in
    the *predicate* the kernel compiles the way backward-data's do -- this
    direction reads X at ``o + t - p``, and ``PADDED`` is on for any non-zero
    padding -- but they differ in the output extent and therefore in the
    reduction length, the split count and whether ``BLOCK_K`` divides a row, and
    the anisotropic one is the only case where the ``d`` half of the boundary
    predicate is dead while the ``h``/``w`` halves are live.
    """
    seen: set[tuple] = set()
    out: list[ConvProblem] = []
    for p in scaffold_corpus():
        if p.transposed or (p.cin, p.cout, p.kernel) in seen:
            continue
        seen.add((p.cin, p.cout, p.kernel))
        shard = tuple(0 if i == 0 else v for i, v in enumerate(p.padding))
        forms = [(p.padding, ""), ((0, 0, 0), "-halo")]
        if shard != p.padding and shard != (0, 0, 0):
            forms.insert(1, (shard, "-shard"))
        for pad, tag in forms:
            out.append(ConvProblem(
                f"{p.cin}to{p.cout}{tag}", p.cin, p.cout, (6, 7, 8), p.kernel,
                padding=pad, sites=("corpus-pair",),
            ))
    return out


#: See :func:`_corpus_channel_pairs`.
CORPUS_PAIRS = _corpus_channel_pairs()

#: Real ScaFFold shapes, at their real volumes, small enough to reference in
#: fp64.  Used only for the fp32 test below.
CORPUS_SMALL = [
    p for p in scaffold_corpus()
    if not p.transposed
    and math.prod(p.halo_variant.spatial) * max(p.cin, p.cout) <= 1 << 22
]
CORPUS_SMALL += [p.halo_variant for p in CORPUS_SMALL]


def _ids(problems):
    return [p.name or p.label for p in problems]


def _run(problem: ConvProblem, ops: dict, **kwargs) -> torch.Tensor:
    return conv3d_backward_weight(
        ops["input"], problem.weight_shape, ops["grad_output"],
        problem.stride, problem.padding, **kwargs,
    )


# ---------------------------------------------------------------------------
# The split decomposition, before any GPU is involved
# ---------------------------------------------------------------------------


def _cfg(**kw) -> BwdWeightConfig:
    base = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4)
    base.update(kw)
    return BwdWeightConfig(**base)


def test_the_splits_partition_the_reduction_exactly_once():
    """Every output voxel lands in exactly one split, for every shape.

    This is the invariant the whole direction rests on: the kernel clamps its
    last tile with ``k_end = min(k_begin + chunk, K)`` and the reduction adds
    every split unconditionally, so a chunk arithmetic that overlapped would
    double-count silently and one that fell short would drop voxels at the end
    of the volume -- both of which produce a plausible gradient.
    """
    for out_w in (8, 16, 128, 256, 7, 13):
        for k_total in (out_w, out_w * 3, out_w * 4096, out_w * 4097):
            for bk in (16, 32, 64, 128):
                for sk in (0, 1, 3, 64, 1000):
                    cfg = _cfg(BLOCK_K=bk, SPLIT_K=sk)
                    splits, chunk = split_count(cfg, 64, 64, 27, k_total, out_w)
                    assert splits >= 1 and chunk >= 1
                    assert (splits - 1) * chunk < k_total <= splits * chunk
                    # The kernel's cheap scalar unravel is only valid when a
                    # K-tile cannot straddle a row, which needs the chunk to be
                    # row-aligned as well as the tile.
                    if out_w % bk == 0:
                        assert chunk % out_w == 0
                    else:
                        assert chunk % bk == 0


def test_the_grid_lands_on_whole_waves():
    """The snap that the measured split-count curve turned out to be about.

    Every program in this kernel does the same amount of work, so a grid of 4.5
    waves runs five and idles through half the last one.  At
    ``64 -> 64 @ 130x258x258`` that sawtooth is an 18% effect -- 596 splits is
    16% more parallelism than 512 and 9% slower -- and it is easy to misread as
    a statement about cache footprint.  So the property is pinned here rather
    than left to the constant that happens to produce it.
    """
    for cout, cin, taps, k_total, out_w in (
        (64, 64, 27, 8_388_608, 256), (64, 128, 27, 2_097_152, 128),
        (128, 128, 27, 1_048_576, 128), (256, 256, 27, 131_072, 64),
        (512, 512, 27, 16_384, 32), (6, 64, 1, 2_097_152, 256),
    ):
        cfg = bwd_weight_config(cout, cin, (3, 3, 3) if taps > 1 else (1, 1, 1),
                                k_total, torch.bfloat16)
        splits, _ = split_count(cfg, cout, cin, taps, k_total, out_w)
        tiles = (-(-cout // cfg.BLOCK_M) * -(-cin // cfg.BLOCK_NC)
                 * -(-taps // cfg.TAP_BLOCK))
        progs = tiles * splits
        if progs <= _CU_COUNT:
            continue                       # one wave or less: nothing to snap
        waste = (-(-progs // _CU_COUNT) * _CU_COUNT - progs) / progs
        assert waste < 0.10, (cout, cin, k_total, splits, tiles, progs, waste)
    assert _SPLIT_TARGET_WAVES >= 1


def test_the_split_count_is_a_pure_function_of_the_shape():
    """Determinism starts here: no clock, no free memory, no autotuner.

    Stated as a test because it is an easy thing to give away later -- a split
    count that adapted to the device's current occupancy would be a perfectly
    reasonable optimization and would silently end the reproducibility claim.
    """
    cfg = _cfg()
    first = split_count(cfg, 128, 256, 27, 2_097_152, 128)
    for _ in range(4):
        assert split_count(cfg, 128, 256, 27, 2_097_152, 128) == first
    # And it responds to the shape, so the property above is not vacuous.
    assert split_count(cfg, 128, 256, 27, 4096, 16)[0] < first[0]


#: The largest fp32 partial workspace any corpus problem asks for, in MiB, on
#: the path :func:`conv3d_backward_weight` actually takes.  Quoted to anyone
#: sizing a hoisted ``workspace=`` once and out of the step, so it is pinned by
#: a test rather than recorded in a document: the previously published figure
#: (111 MiB) was the maximum over the ten shapes of the determinism table
#: measured on the *heuristic* config, and understated the real corpus maximum
#: by 1.46x.  An integration that had sized from it would have taken a
#: ``ValueError`` mid-run.
def _every_form(problems):
    """Each non-transposed problem in all three of the forms ScaFFold issues.

    Bounds like the workspace ceiling have to hold on the shape the *kernel*
    is handed, and there are three of those: the module's own padded statement,
    the adapter's (padded on every unsplit axis) and upstream DistConv's (halo'd
    and unpadded).  Iterating only the last of them -- which this file did until
    2026-08-04 -- bounds the one form production never issues.  Deduplicated on
    the qualified label, since the three coincide wherever nothing is split.
    """
    seen: set[str] = set()
    for p in problems:
        if p.transposed:
            continue        # a later milestone; ``weight_shape`` is swapped too
        for q in (p, p.production_variant, p.halo_variant):
            if q.qualified_label in seen:
                continue
            seen.add(q.qualified_label)
            yield q


_WORST_WORKSPACE_MIB = 216


def test_the_partial_workspace_is_bounded_across_the_whole_corpus():
    """``splits * Cout * taps * Cin * 4`` bytes, on every problem ScaFFold runs.

    The number to watch is the *product*: one split at ``1024 -> 1024`` is
    113 MiB, and a split count picked for a shallow site would ask for
    gigabytes of it.  The two bounds pull against each other -- the sites that
    want many splits are the ones with a small ``Cout * taps * Cin`` -- but that
    is an observation about this corpus, not a theorem, so it is checked.

    Two things this test used to get wrong, both of which made it unable to
    fail:

    * it built its config with :func:`default_bwd_weight_config` while the
      entry point uses :func:`bwd_weight_config`, which prefers the tuned table
      and therefore a different ``BLOCK_M``/``BLOCK_NC``/``TAP_BLOCK``, a
      different tile count and a different split count.  The two disagree by up
      to 1.5x in practice, so the bound it certified was not the shipped one;
    * ``mib <= _WORKSPACE_BYTES`` is *trivially* true -- ``split_count``'s own
      ``ceiling`` is ``_WORKSPACE_BYTES // per_split``, so no config it returns
      can violate it.  The assertion that carries the weight is the pinned
      maximum, which is a number people size allocations from.
    """
    worst, worst_label = 0.0, ""
    for hp in _every_form(list(scaffold_corpus()) + EDGE):
        k_total = hp.n * math.prod(hp.out_spatial)
        # Both, and the worst of the two: a caller who passes no ``config=``
        # gets the resolver's answer, and one who builds a config from
        # :func:`default_bwd_weight_config` gets the heuristic's, which at an
        # untuned pair is what production launches.  Since 2026-08-05 the two
        # agree on padded problems that they used to disagree on -- the
        # ``TAP_BLOCK`` decline is gone -- so the worst of the pair is a smaller
        # set than it was, and it is still the number to size an allocation
        # from.
        for cfg in (
            bwd_weight_config(hp.cout, hp.cin, hp.kernel, k_total,
                              torch.bfloat16, padded=any(hp.padding)),
            default_bwd_weight_config(hp.cout, hp.cin, hp.kernel, k_total,
                                      torch.bfloat16, padded=any(hp.padding)),
        ):
            splits, _ = split_count(cfg, hp.cout, hp.cin, hp.tap_count, k_total,
                                    hp.out_spatial[2])
            mib = workspace_elements(splits, hp.cout, hp.cin,
                                     hp.kernel) * 4 / 2**20
            assert mib <= _WORKSPACE_BYTES / 2**20, f"{hp.label}: {mib:.0f} MiB"
            if mib > worst:
                worst, worst_label = mib, f"{hp.label} {cfg}"
    # A ceiling nothing approaches would not be a useful test either.
    assert worst > 1.0
    assert round(worst) == _WORST_WORKSPACE_MIB, (
        f"the corpus workspace maximum moved to {worst:.1f} MiB at "
        f"{worst_label}, from the {_WORST_WORKSPACE_MIB} MiB pinned here.  "
        "That number is what an integration sizes a hoisted workspace= from, "
        "so update it deliberately -- do not widen this assertion"
    )


def test_the_wave_snap_outranks_the_epilogue_bound_and_only_below_one_wave():
    """The bound the docstring states is the bound the code applies.

    :func:`split_count` clamps its target against three ceilings and *then*
    snaps to a whole number of waves, and the snap can push the result back
    above the epilogue bound.  That reads like an oversight and is not: the
    alternative was implemented and raced, and it loses badly.  Re-applying the
    epilogue bound after the snap takes ``128 -> 256 @ 34^3`` from 16 splits to
    7 -- a 98-program grid on 228 CUs -- and the site from 0.2536 to 0.4563 ms
    (**1.80x**); ``256 -> 512 @ 10x34x34`` goes 0.2731 -> 0.6553 ms.  Half an
    idle device costs more than a doubled epilogue, and the shapes where the
    snap overrides the bound are *exactly* the shapes with a sub-wave grid,
    because that is the condition under which ``round`` rounds to zero.

    So what is pinned here is the ordering itself, in both directions:

    * the snap may exceed the epilogue bound only by *rounding the grid to the
      nearest whole wave* -- at most half a wave of extra programs, or one
      whole wave where the bounded grid does not fill even that.  Anything
      beyond that would mean the bound had stopped constraining anything;
    * the **workspace** ceiling is different in kind (a failed allocation at
      step 400 is not a slow kernel) and is re-applied after the snap, so it is
      never exceeded.

    Both halves have failed at some point in this function's history, in
    opposite directions.
    """
    def bounds(cfg, cout, cin, taps, k_total):
        tiles = (-(-cout // cfg.BLOCK_M) * -(-cin // cfg.BLOCK_NC)
                 * -(-taps // cfg.TAP_BLOCK))
        loop_elems = tiles * k_total * (cfg.BLOCK_M + cfg.BLOCK_N)
        epi = max(1, loop_elems
                  // (_MAX_EPILOGUE_FRACTION * max(1, cout * taps * cin * 4)))
        return tiles, epi

    checked = overridden = 0
    for hp in _every_form(list(scaffold_corpus()) + EDGE):
        k_total = hp.n * math.prod(hp.out_spatial)
        for cfg in (
            bwd_weight_config(hp.cout, hp.cin, hp.kernel, k_total,
                              torch.bfloat16, padded=any(hp.padding)),
            default_bwd_weight_config(hp.cout, hp.cin, hp.kernel, k_total,
                                      torch.bfloat16, padded=any(hp.padding)),
        ):
            splits, _ = split_count(cfg, hp.cout, hp.cin, hp.tap_count, k_total,
                                    hp.out_spatial[2])
            tiles, epi = bounds(cfg, hp.cout, hp.cin, hp.tap_count, k_total)
            checked += 1
            if splits <= epi:
                continue
            overridden += 1
            # Over the epilogue bound is allowed, but only by the rounding the
            # snap does: to the *nearest* whole wave, so at most half a wave of
            # extra programs -- or one whole wave where the bounded grid does
            # not fill even one.
            assert tiles * splits <= max(_CU_COUNT,
                                         tiles * epi + _CU_COUNT // 2), (
                f"{hp.label} {cfg}: {splits} splits against an epilogue bound "
                f"of {epi} is {tiles * splits} programs, more than a wave past "
                f"the bounded grid's {tiles * epi}"
            )
            # And the workspace ceiling still holds, which is the one bound the
            # snap is *not* allowed to escape.
            mib = workspace_elements(splits, hp.cout, hp.cin,
                                     hp.kernel) * 4 / 2**20
            assert mib <= _WORKSPACE_BYTES / 2**20, f"{hp.label}: {mib:.0f} MiB"
    assert checked > 40

    assert overridden, (
        "no shape in the corpus reaches the sub-wave regime any more, so this "
        "test no longer covers the ordering it exists to pin"
    )


@pytest.mark.parametrize("problem", EDGE + CORPUS_PAIRS,
                         ids=_ids(EDGE + CORPUS_PAIRS))
def test_selected_config_is_legal_for_every_shape(problem: ConvProblem):
    """The config picked for backward-weight must reach the matrix core.

    Not implied by the other two directions' versions of this test: here
    ``BLOCK_M`` is bounded by ``Cout`` rather than by a volume, so a shape whose
    forward tile is legal can select a tile here that is not -- and an illegal
    MFMA configuration on gfx942 runs, returns the right answer, and emits no
    matrix instruction at all.
    """
    dtype = reference.torch_dtype(problem)
    hp = problem.halo_variant
    k_total = hp.n * math.prod(hp.out_spatial)
    cfg = bwd_weight_config(hp.cout, hp.cin, hp.kernel, k_total, dtype,
                            padded=any(hp.padding))
    assert cfg.validate(dtype) is None, f"{hp.label}: {cfg} -> {cfg.validate(dtype)}"
    assert cfg.lds_bytes(dtype) <= 64 * 1024, f"{hp.label}: {cfg}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32],
                         ids=["bf16", "fp16", "fp32"])
def test_default_config_fits_in_lds_in_every_dtype(dtype):
    """fp32 operands are twice the bytes, and ``more_determinism`` runs in fp32.

    M2 found exactly this hole in the *forward*'s shipped heuristic, where
    ``128x128x128`` is 64 KiB in bf16 and 128 KiB in fp32 and the shipped
    configuration raised ``OutOfResources``.  This direction's tiles are wider
    still -- ``TAP_BLOCK`` multiplies ``BLOCK_N`` -- so the same trap is closer,
    not further away.
    """
    for cout in (6, 64, 128, 256, 512, 1024):
        for cin in (3, 64, 128, 256, 512, 1024):
            for k in ((1, 1, 1), (3, 3, 3)):
                cfg = default_bwd_weight_config(cout, cin, k, 1 << 20, dtype)
                assert cfg.validate(dtype) is None, cfg
                assert cfg.lds_bytes(dtype) <= 64 * 1024, (cout, cin, k, cfg)


@pytest.mark.parametrize("problem", CORPUS_PAIRS, ids=_ids(CORPUS_PAIRS))
def test_every_backward_weight_candidate_config_is_legal(problem: ConvProblem):
    """The sweep that produced the tuned table must not contain an FMA kernel.

    Same reasoning as the other two directions': an illegal config runs and
    returns the right answer slowly, so a best-of sweep that merely ranked it
    last would still be reporting a meaningless winner.
    """
    dtype = reference.torch_dtype(problem)
    k_total = problem.n * math.prod(problem.out_spatial)
    cfgs = candidate_bwd_weight_configs(problem.cout, problem.cin,
                                        problem.kernel, k_total, dtype)
    assert cfgs
    for cfg in cfgs:
        assert cfg.validate(dtype) is None, f"{cfg}: {cfg.validate(dtype)}"
        assert cfg.lds_bytes(dtype) <= 64 * 1024, cfg
        assert cfg.BLOCK_N == cfg.BLOCK_NC * cfg.TAP_BLOCK


def test_config_validate_refuses_the_two_knobs_this_direction_adds():
    bf16 = torch.bfloat16
    assert BwdWeightConfig().validate(bf16) is None
    assert BwdWeightConfig(SPLIT_K=-1).validate(bf16)
    assert BwdWeightConfig(TAP_BLOCK=0).validate(bf16)
    # BLOCK_N is the *full* tile width, so it has to divide into whole taps --
    # otherwise BLOCK_NC is a truncated integer and the column decode silently
    # addresses the wrong channels.
    assert BwdWeightConfig(BLOCK_N=64, TAP_BLOCK=3).validate(bf16)
    assert BwdWeightConfig(BLOCK_N=192, TAP_BLOCK=3).validate(bf16) is None
    # And the inherited gfx942 rules still apply.
    assert BwdWeightConfig(BLOCK_K=8).validate(bf16)


# ---------------------------------------------------------------------------
# Support predicate
# ---------------------------------------------------------------------------


@requires_gpu
def test_is_supported_declines_what_the_kernel_cannot_express():
    """Note what is *not* refused: ``stride > 1``.

    Backward-data has to refuse a stride because its substitution turns into a
    scatter into a sub-lattice.  This direction does not: its reduction axis is
    the output voxel and the input coordinate ``o*s + t*dil - p`` is a function
    of it, so a stride is three extra multiplies.  The asymmetry is real and is
    pinned here so that a later reader does not "fix" it by symmetry.
    """
    x = torch.empty((1, 8, 6, 6, 6), device="cuda", dtype=torch.bfloat16)
    gy = torch.empty((1, 8, 6, 6, 6), device="cuda", dtype=torch.bfloat16)
    ws = (8, 8, 3, 3, 3)
    assert is_supported_bwd_weight(x, ws, gy, padding=1)

    strided = torch.empty((1, 8, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    assert is_supported_bwd_weight(x, ws, strided, stride=2, padding=1)

    assert not is_supported_bwd_weight(x, ws, gy, padding=1, groups=2)
    assert not is_supported_bwd_weight(x, ws, gy.float(), padding=1)

    # Both operands on *the same* device, not merely both on a device.  Triton
    # launches on the current device and dereferences the other pointer anyway,
    # and ScaFFold runs four GPUs per node: with peer access enabled a foreign
    # pointer does not fault, it reads another rank's activations and returns a
    # plausible wrong gradient.  ``gather_gemm.is_supported`` refuses the same
    # thing; the two gates sit behind one rung ladder and a hole in either is a
    # hole in the ladder.
    assert not is_supported_bwd_weight(x, ws, gy.cpu(), padding=1)
    assert not is_supported_bwd_weight(x.cpu(), ws, gy, padding=1)
    if torch.cuda.device_count() >= 2:
        # The clause above ``is_cuda`` cannot reach: two *CUDA* devices.  Only
        # runnable on a multi-GPU node -- this suite is normally run with one
        # device pinned -- so the CPU cases above stay unconditional rather than
        # letting the whole check disappear behind the guard.
        assert not is_supported_bwd_weight(x, ws, gy.to("cuda:1"), padding=1)
    # ...and the same-device pair is still accepted, so none of this is a
    # predicate that has simply started refusing everything.
    assert is_supported_bwd_weight(x, ws, gy, padding=1)
    assert not is_supported_bwd_weight(x, (8, 4, 3, 3, 3), gy, padding=1)
    assert not is_supported_bwd_weight(x, (4, 8, 3, 3, 3), gy, padding=1)
    # grad_output's extent has to be the one this problem produces, or the
    # reduction would run over a volume the input does not have.
    assert not is_supported_bwd_weight(x, ws, gy, padding=0)
    assert not is_supported_bwd_weight(
        x, ws, torch.empty((1, 8, 4, 6, 6), device="cuda", dtype=torch.bfloat16),
        padding=1,
    )


@requires_gpu
def test_unsupported_calls_raise_rather_than_return_garbage():
    x = torch.randn((1, 8, 6, 6, 6), device="cuda", dtype=torch.bfloat16)
    gy = torch.randn((1, 8, 6, 6, 6), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):
        conv3d_backward_weight(x, (8, 8, 3, 3, 3), gy, padding=1, groups=2)
    with pytest.raises(NotImplementedError):
        conv3d_backward_weight(x, (8, 8, 3, 3, 3), gy, padding=0)
    # An out= in the wrong layout is refused rather than filled transposed.
    with pytest.raises(ValueError):
        conv3d_backward_weight(
            x, (8, 8, 3, 3, 3), gy, padding=1,
            out=torch.empty((8, 8, 3, 3, 3), device="cuda", dtype=torch.bfloat16),
        )


# ---------------------------------------------------------------------------
# Correctness: the bitwise standard
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("problem", EDGE, ids=_ids(EDGE))
def test_exact_operands_match_bitwise(problem: ConvProblem):
    """Bitwise against ``torch.autograd.grad`` in fp64, on the nasty shapes.

    The synthetic corpus earns its place here differently than in the other two
    directions: ``Cout=6`` and ``Cout=7`` land on the GEMM's *M*, which is the
    axis this kernel has least of, and ``spatial_thin`` (2x31x3) gives an output
    volume of 12 -- a reduction shorter than one ``BLOCK_K``.
    """
    ops = reference.make_inputs(problem, seed=3, exact=True)
    expected = reference.reference(problem, ops, "bwd-weight")
    dtype = reference.torch_dtype(problem)
    if not reference.is_exactly_representable(expected, dtype):
        pytest.skip("realized magnitudes exceed the mantissa in this dtype")
    actual = _run(problem, ops)
    report = reference.compare(actual, expected.to(dtype))
    assert report.bitwise, f"{problem.label}: {report}"


@requires_gpu
@pytest.mark.parametrize("problem", CORPUS_PAIRS, ids=_ids(CORPUS_PAIRS))
def test_corpus_channel_pairs_match_bitwise(problem: ConvProblem):
    """Every channel pair ScaFFold runs, in both paddings, bitwise in bf16."""
    ops = reference.make_inputs(problem, seed=5, exact=True)
    expected = reference.reference(problem, ops, "bwd-weight")
    dtype = reference.torch_dtype(problem)
    if not reference.is_exactly_representable(expected, dtype):
        pytest.skip("realized magnitudes exceed the mantissa in this dtype")
    actual = _run(problem, ops)
    assert reference.compare(actual, expected.to(dtype)).bitwise


@requires_gpu
def test_the_bitwise_corpus_is_not_entirely_skipped():
    """A regression guard on this file, not on the kernel.

    ``is_exactly_representable`` declining is correct behaviour, but if it
    declines for every parametrized case the suite reports a wall of passes and
    tests nothing.  That is what the first version of ``test_bwd_data.py`` did.
    """
    exact = sum(
        reference.is_exactly_representable(
            reference.reference(
                p, reference.make_inputs(p, seed=5, exact=True), "bwd-weight"
            ),
            reference.torch_dtype(p),
        )
        for p in CORPUS_PAIRS
    )
    assert exact >= len(CORPUS_PAIRS) // 2, (
        f"only {exact}/{len(CORPUS_PAIRS)} corpus pairs are bf16-exact; the "
        "bitwise corpus test is close to vacuous"
    )


@requires_gpu
@pytest.mark.parametrize(
    "problem", CORPUS_PAIRS + CORPUS_SMALL, ids=_ids(CORPUS_PAIRS + CORPUS_SMALL)
)
def test_deep_corpus_shapes_match_bitwise_in_fp32(problem: ConvProblem):
    """The shapes bf16 cannot express exactly, at their real widths and volumes.

    A reduction over a real ScaFFold output volume runs to about ``sqrt(K)`` in
    ``{-1,0,1}`` arithmetic -- 1450 at the 128^3 sites -- which bf16's 8-bit
    mantissa provably cannot hold, as a property of the arithmetic and not of
    the test.  fp32 has 24 bits, which covers it, and the addressing under test
    is dtype-independent: what changes is the MFMA intrinsic and therefore the
    legal ``BLOCK_K``, so this is also the only bitwise coverage the fp32 tile
    selection gets at real widths.

    This test does not skip.  If the fp32 reference is ever not exact either,
    that is a fact worth failing on rather than stepping around.
    """
    ops = reference.make_inputs(problem, seed=7, exact=True, dtype=torch.float32)
    expected = reference.reference(problem, ops, "bwd-weight")
    assert reference.is_exactly_representable(expected, torch.float32)
    actual = _run(problem, ops)
    assert actual.dtype is torch.float32
    assert reference.compare(actual, expected.to(torch.float32)).bitwise


#: Shapes that compile the ``PADDED and ROW_ALIGNED`` pair of constexprs.  Every
#: other padded shape in this file has ``out_w < BLOCK_K``, so ``_row_aligned``
#: is False at all of them and this combination had never been compiled by the
#: suite at all.  See the test below for why that is worth fixing.
_PADDED_ROW_ALIGNED = [
    # ``IN_D = IN_H = 1`` under ``padding=1``: ``src_d`` is -1 at every voxel and
    # the three taps land at -1, 0 and +1, so both the low and the high ``d``/
    # ``h`` boundaries fire on every K-tile rather than only at the volume's
    # edge.  16 reduction terms, so bf16 holds the result exactly.
    ConvProblem("pad-rowaligned-thin", 16, 16, (1, 1, 16)),
    # The logical (non-halo'd) form of a real corpus site: ``256->128 k3 @
    # 64x128x128, padding=1`` has ``out_w = 128`` against ``BLOCK_K = 64``.  Same
    # shape of predicate at a width the corpus actually produces; fp32 because
    # a 2048-term reduction is past bf16's mantissa.
    ConvProblem("pad-rowaligned-corpus", 32, 32, (4, 4, 128), dtype="fp32"),
]


@requires_gpu
@pytest.mark.parametrize("problem", _PADDED_ROW_ALIGNED, ids=_ids(_PADDED_ROW_ALIGNED))
def test_the_padded_row_aligned_corner_is_compiled_and_correct(problem):
    """The one ``constexpr`` pair nothing else in this suite reaches.

    ``PADDED`` and ``ROW_ALIGNED`` are independent, and they interact.  In the
    ``ROW_ALIGNED`` branch ``row``, ``od``, ``oh`` and ``idn`` collapse to
    **rank-0 scalars** -- the whole point of that branch is that the unravel
    becomes four SALU divisions -- so the padded branch's boundary predicate
    ``src_d[:, None] + (kd*DD)[None, :]`` is a different expression there than
    in the general branch: broadcast from a scalar rather than from a
    ``BLOCK_K`` vector, and collapsed to one row of the mask instead of
    ``BLOCK_K`` of them.  It is the right predicate, because within a
    row-aligned K-tile ``od`` and ``oh`` really are constant -- but "it is
    correct" and "it is tested" are different claims, and a bug planted in the
    ``d`` or ``h`` half of it passed the entire suite.

    **This is a production branch, not a hypothetical one.**  It used to be
    documented as reachable only by "a caller who bypasses DistConv", on the
    premise that every ScaFFold convolution is issued halo'd and unpadded.  That
    premise is false: the shipped adapter halos only the split axis, so
    ``256->128 k3 @ 64x128x128`` arrives padded with ``out_w = 128`` against
    ``BLOCK_K = 64`` -- exactly this branch -- every step.
    """
    hp = problem
    k_total = hp.n * math.prod(hp.out_spatial)
    dtype = reference.torch_dtype(problem)
    cfg = bwd_weight_config(hp.cout, hp.cin, hp.kernel, k_total, dtype,
                            padded=any(hp.padding))
    # The two constexprs, asserted rather than hoped for: this test's whole
    # value is that it compiles a branch, so it has to fail loudly if a config
    # change ever stops it reaching that branch.
    assert any(hp.padding), "PADDED would be False"
    assert _row_aligned(cfg.BLOCK_K, hp.out_spatial[2]), (
        f"ROW_ALIGNED is False: BLOCK_K={cfg.BLOCK_K} does not divide "
        f"out_w={hp.out_spatial[2]}"
    )

    ops = reference.make_inputs(problem, seed=97, exact=True, dtype=dtype)
    expected = reference.reference(problem, ops, "bwd-weight")
    assert reference.is_exactly_representable(expected, dtype)
    assert reference.compare(_run(problem, ops), expected.to(dtype)).bitwise


#: The channel pairs whose tuned backward-weight row widens ``TAP_BLOCK``.
#: Until 2026-08-05 these were exactly the rows *declined* on a padded problem,
#: i.e. at every production ScaFFold site; the decline is gone and this set is
#: now the rows that must survive the padding.  Resolved from the table rather
#: than listed, so a retune moves this set instead of stranding it.
def _tap_widened_pairs() -> list[tuple[ConvProblem, BwdWeightConfig]]:
    from triton_conv3d.reduce_gemm import (
        _TUNED_BWD_W, _fit_bwd_weight_to_lds, tune_key,
    )

    out, seen = [], set()
    for p in scaffold_corpus():
        if p.transposed or (p.cin, p.cout, p.kernel) in seen:
            continue
        seen.add((p.cin, p.cout, p.kernel))
        row = _TUNED_BWD_W.get(
            tune_key(torch.bfloat16, p.cin, p.cout, tuple(p.kernel)))
        if row is not None and row.TAP_BLOCK > 1:
            out.append((p, _fit_bwd_weight_to_lds(row, torch.bfloat16)))
    return out


_TAP_WIDENED = _tap_widened_pairs()


def test_a_tuned_tap_block_row_survives_the_padding():
    """The replacement for ``..._is_declined_when_padded``, and why it flipped.

    Until 2026-08-05 ``bwd_weight_config`` refused a tuned row with
    ``TAP_BLOCK > 1`` whenever the convolution was padded, and
    ``default_bwd_weight_config`` refused to widen ``TAP_BLOCK`` there at all.
    Both clauses were written believing they could not fire -- "no real ScaFFold
    convolution is padded, DistConv halos them all" -- and that was false:
    ScaFFold's own adapter halos only the split axis, so every ``k > 1`` site
    arrives padded and **eight sites over six channel pairs** took the decline
    at every configuration.

    The old test asserted the decline and asked whoever relaxed it to replace
    the assertion with a measurement.  That is what happened.  Raced on the
    padded production form of all 18 affected cells, the tuned row against the
    config the decline produced, one interleaved block per cell with 95%
    intervals: the tuned row wins **18 of 18**, geometric mean **1.946x**, range
    1.137x-5.336x, worst cell 7.9505 ms declined against 1.4910 ms with the
    row.  The heuristic's half was raced separately on the six pairs that reach
    it and widening wins **6 of 6**, 1.263x-2.084x.

    So this test now pins the opposite property, and it is the one that matters
    for production: the tuned row must be what a *padded* problem resolves,
    because a padded problem is the only kind ScaFFold issues.
    """
    assert _TAP_WIDENED, (
        "no tuned backward-weight row widens TAP_BLOCK any more; this test and "
        "the behaviour it pins are both about a table that has changed"
    )
    for p, row in _TAP_WIDENED:
        k_total = p.n * math.prod(p.out_spatial)
        padded = bwd_weight_config(p.cout, p.cin, p.kernel, k_total,
                                   torch.bfloat16, padded=True)
        unpadded = bwd_weight_config(p.cout, p.cin, p.kernel, k_total,
                                     torch.bfloat16, padded=False)
        assert unpadded == row, (
            f"{p.cin}->{p.cout}: the tuned row is not selected even unpadded"
        )
        assert padded == row, (
            f"{p.cin}->{p.cout}: a padded problem resolved {padded} instead of "
            f"the tuned row {row}. Production issues nothing but padded "
            "convolutions, so this is the whole of what the table buys -- read "
            "this test's docstring before accepting it"
        )
        assert padded.TAP_BLOCK > 1


def test_the_heuristic_widens_tap_block_under_padding_too():
    """The other half of the same predicate, pinned separately.

    :func:`default_bwd_weight_config` used to pin ``TAP_BLOCK`` to 1 on a padded
    problem.  It no longer does, and the two are now the same config: padding
    changes the boundary predicate inside the kernel and nothing about the tile
    the host picks.  Kept apart from the test above because this one governs
    every channel pair the tuned table does *not* list, which is where a new
    ScaFFold site lands.
    """
    for p, _row in _TAP_WIDENED:
        k_total = p.n * math.prod(p.out_spatial)
        wide = default_bwd_weight_config(p.cout, p.cin, p.kernel, k_total,
                                         torch.bfloat16, padded=False)
        padded = default_bwd_weight_config(p.cout, p.cin, p.kernel, k_total,
                                           torch.bfloat16, padded=True)
        assert padded == wide, (
            f"{p.cin}->{p.cout}: the heuristic still answers differently under "
            f"padding ({padded} vs {wide})"
        )
        assert wide.TAP_BLOCK > 1, (
            f"{p.cin}->{p.cout}: the heuristic did not widen TAP_BLOCK at all; "
            "this test is about a rule that has changed"
        )


@requires_gpu
@pytest.mark.parametrize(
    "problem,cfg",
    [(ConvProblem(f"{p.cin}to{p.cout}-padded", p.cin, p.cout, (6, 7, 8),
                  p.kernel, padding=p.padding, sites=("tap-widened",)), c)
     for p, c in _TAP_WIDENED],
    ids=[f"{p.cin}to{p.cout}" for p, _ in _TAP_WIDENED],
)
def test_a_padded_tap_block_row_is_still_bitwise_correct(problem, cfg):
    """The gradient a widened row produces on a padded problem, bitwise.

    Written while the row was still *declined* on a padded problem, to establish
    that what the decline protected was a performance argument and not a
    correctness one.  Since 2026-08-05 the decline is gone and this is no longer
    a hypothetical: it is the gradient every ``k = 3`` ScaFFold site computes,
    so a failure here is a wrong weight gradient in production rather than a
    reason not to relax a clause.
    """
    assert cfg.TAP_BLOCK > 1 and any(problem.padding)
    ops = reference.make_inputs(problem, seed=1234, exact=True)
    expected = reference.reference(problem, ops, "bwd-weight")
    dtype = reference.torch_dtype(problem)
    assert reference.is_exactly_representable(expected, dtype)
    actual = _run(problem, ops, config=cfg)
    assert reference.compare(actual, expected.to(dtype)).bitwise, (
        f"{problem.label} with {cfg} (TAP_BLOCK>1 on a padded convolution) is "
        "not bitwise correct"
    )


#: The triple ``PADDED and ROW_ALIGNED and TAP_BLOCK > 1``, and -- in the last
#: entry -- the *quintuple* production actually launches.  ``_PADDED_ROW_
#: ALIGNED`` above reaches the first two but not the third, so until 2026-08-05
#: the combination was compiled nowhere; it is now compiled at every ``k = 3``
#: site of every configuration.  ``out_w`` is chosen equal to ``BLOCK_K`` so a
#: K-tile is exactly one output row.
#:
#: ``block_nc`` is carried per case because the stem needs it.  ``3 -> 64``
#: resolves ``64x64x64/tb16``, i.e. ``BLOCK_NC = 4`` against ``Cin = 3``, so it
#: adds two raggednesses to the triple -- a partial channel group *and* a
#: partial tap group (27 taps in blocks of 16) -- inside the ``ROW_ALIGNED``
#: branch where ``src_d``/``src_h`` collapse to scalars.  Nothing else in the
#: suite compiles that: the three cases above hold ``Cin = BLOCK_NC = 32``, and
#: the ragged-``Cin`` tests elsewhere are not row-aligned.  It is also the
#: largest single win in the round (5.3x), which is a poor thing to have
#: untested.
_PADDED_ROW_ALIGNED_TAPS = [
    (ConvProblem("triple-tb8", 32, 32, (2, 3, 16)), 8, 16, 32),
    (ConvProblem("triple-tb2", 32, 32, (2, 2, 64), dtype="fp32"), 2, 64, 32),
    (ConvProblem("triple-tb16", 32, 64, (2, 2, 32)), 16, 32, 32),
    # The sharded production padding, which is anisotropic: the ``d`` half of
    # the boundary predicate is dead and the ``h``/``w`` halves are live, inside
    # the branch where ``src_d`` is a rank-0 scalar.
    (ConvProblem("triple-shardpad", 32, 32, (4, 4, 32), padding=(0, 1, 1)),
     4, 32, 32),
    # The stem, in both of its production paddings.  ``BLOCK_M`` is 32 here
    # rather than the shipped 64 only because this test fixes it; every other
    # constexpr is the one the resolver returns.
    (ConvProblem("quintuple-stem", 3, 64, (2, 2, 64)), 16, 64, 4),
    (ConvProblem("quintuple-stem-shardpad", 3, 64, (4, 4, 64),
                 padding=(0, 1, 1)), 16, 64, 4),
]


@requires_gpu
@pytest.mark.parametrize(
    "problem,tap_block,block_k,block_nc", _PADDED_ROW_ALIGNED_TAPS,
    ids=[p.name for p, _, _, _ in _PADDED_ROW_ALIGNED_TAPS],
)
def test_the_padded_row_aligned_tap_widened_corner_is_correct(
    problem, tap_block, block_k, block_nc
):
    """Three independent ``constexpr`` at once, which nothing else compiles.

    ``PADDED`` selects a two-dimensional boundary predicate; ``ROW_ALIGNED``
    collapses ``od``/``oh``/``idn`` to rank-0 scalars; ``TAP_BLOCK > 1`` makes
    the tap vary down the *columns*.  Together the predicate is a scalar
    broadcast against a per-column tap shift, and since 2026-08-05 it is the
    shape production launches at every ``k = 3`` site with a widened row -- see
    :func:`test_a_tuned_tap_block_row_survives_the_padding`.  It was written
    while the combination was still unreachable, which is why it forces the
    constexpr triple by hand rather than going through the resolver.

    The last two cases add the stem's two raggednesses on top, which is the
    combination the shipped ``3 -> 64`` row launches and which nothing else
    reaches; the assertions below say which case is which so a failure names the
    axis rather than the tile.
    """
    cfg = BwdWeightConfig(BLOCK_M=32, BLOCK_N=block_nc * tap_block,
                          BLOCK_K=block_k, TAP_BLOCK=tap_block, num_warps=4,
                          matrix_instr_nonkdim=16, kpack=1)
    assert cfg.BLOCK_NC == block_nc
    assert any(problem.padding), "PADDED would be False"
    assert _row_aligned(cfg.BLOCK_K, problem.out_spatial[2]), (
        f"ROW_ALIGNED is False: BLOCK_K={cfg.BLOCK_K} does not divide "
        f"out_w={problem.out_spatial[2]}"
    )
    # 27 taps never divide by a power of two, so *every* case here has a ragged
    # last tap group; the stem cases add a ragged channel group on top, and that
    # is the axis the four original cases do not reach.
    assert math.prod(problem.kernel) % cfg.TAP_BLOCK != 0
    assert (problem.cin % cfg.BLOCK_NC != 0) == (problem.cin == 3), (
        "the stem cases are the ragged-Cin ones; the others must not be"
    )
    dtype = reference.torch_dtype(problem)
    ops = reference.make_inputs(problem, seed=31, exact=True, dtype=dtype)
    expected = reference.reference(problem, ops, "bwd-weight")
    assert reference.is_exactly_representable(expected, dtype)
    assert reference.compare(_run(problem, ops, config=cfg),
                             expected.to(dtype)).bitwise


@requires_gpu
def test_bitwise_standard_rejects_a_shifted_input():
    """Prove the comparison discriminates: a one-voxel shift must fail it."""
    problem = ConvProblem("shift", 16, 16, (6, 6, 6))
    ops = reference.make_inputs(problem, seed=11, exact=True)
    actual = _run(problem, ops)
    correct = reference.reference(problem, ops, "bwd-weight").to(torch.bfloat16)
    assert reference.compare(actual, correct).bitwise

    shifted = torch.roll(ops["input"], shifts=1, dims=-1)
    wrong = reference.reference(
        problem, {**ops, "input": shifted}, "bwd-weight"
    ).to(torch.bfloat16)
    assert not reference.compare(actual, wrong).bitwise, (
        "a one-voxel shift of the input produced a bitwise-identical gradient; "
        "the comparison is not discriminating"
    )


@requires_gpu
def test_a_permuted_tap_axis_is_detected():
    """The bug this module can uniquely have, pinned.

    The kernel's N axis is ``(tap, Cin)`` and its output offset is
    ``co*taps*Cin + tap*Cin + ci``.  Getting the tap ordering wrong -- reversing
    it, or transposing (kd,kh,kw) -- produces a correctly shaped, correctly
    scaled, entirely plausible weight gradient, and would pass every tolerance
    test one could write.  At ``k=3`` with a symmetric volume nothing else in
    this file would catch it, so the wrong answer is constructed and required to
    differ.
    """
    problem = ConvProblem("taps", 16, 16, (6, 7, 8))
    ops = reference.make_inputs(problem, seed=13, exact=True)
    actual = _run(problem, ops)
    correct = reference.reference(problem, ops, "bwd-weight").to(torch.bfloat16)
    assert reference.compare(actual, correct).bitwise

    for wrong in (correct.flip(2, 3, 4), correct.transpose(2, 4).contiguous()):
        assert wrong.shape == actual.shape
        assert not reference.compare(actual, wrong).bitwise, (
            "a permuted tap axis gave a bitwise-identical gradient; the "
            "[Cout][tap][Cin] output ordering is untested by this suite"
        )


@requires_gpu
@pytest.mark.parametrize("problem", EDGE[:8], ids=_ids(EDGE[:8]))
def test_every_config_gives_the_same_answer(problem: ConvProblem):
    """The tuning surface, not one point on it.

    This matters more here than in the other two directions because the
    candidate list varies ``TAP_BLOCK`` and ``SPLIT_K``, and both change the
    *decomposition* rather than only the tiling: a wrong tap-column decode shows
    up only at ``TAP_BLOCK > 1``, and an off-by-one in the chunk arithmetic only
    at split counts the shipped heuristic happens not to pick.
    """
    ops = reference.make_inputs(problem, seed=2, exact=True)
    expected = reference.reference(problem, ops, "bwd-weight")
    dtype = reference.torch_dtype(problem)
    if not reference.is_exactly_representable(expected, dtype):
        pytest.skip("realized magnitudes exceed the mantissa in this dtype")
    expected = expected.to(dtype)
    k_total = problem.n * math.prod(problem.out_spatial)
    cfgs = candidate_bwd_weight_configs(
        problem.cout, problem.cin, problem.kernel, k_total, dtype,
        splits=(0, 1, 3, 64),
    )
    ran = 0
    for cfg in cfgs:
        try:
            actual = _run(problem, ops, config=cfg)
        except triton.runtime.errors.OutOfResources:
            continue  # a loud failure; the sweep skips these too
        ran += 1
        assert reference.compare(actual, expected).bitwise, f"{problem.label} {cfg}"
    assert ran, "no candidate configuration was runnable"


@requires_gpu
def test_the_atomic_path_agrees_with_the_deterministic_one():
    """Same answer, different summation order -- so *not* bitwise, but close.

    The atomic path exists only to price determinism, and the price is only
    meaningful if the two compute the same thing.  The bar is the fp64 reference
    rather than each other, because "equal to the wrong answer" is exactly what
    a shared bug would look like.
    """
    problem = ConvProblem("atomic", 64, 64, (10, 12, 16), padding=(0, 0, 0))
    ops = reference.make_inputs(problem, seed=19, exact=True)
    expected = reference.reference(problem, ops, "bwd-weight")
    assert reference.is_exactly_representable(expected, torch.bfloat16)
    expected = expected.to(torch.bfloat16)
    assert reference.compare(_run(problem, ops), expected).bitwise
    assert reference.compare(_run(problem, ops, deterministic=False), expected).bitwise


@requires_gpu
def test_the_forward_kernel_can_express_backward_weight():
    """The reuse M2 got for free, checked here and then rejected on a trip count.

    Swapping the batch and channel axes of both activations turns
    backward-weight into a forward convolution whose kernel extent is the
    *output volume*.  It is a real identity and the forward kernel really
    computes it, which is what this half of the test shows.

    The other half is why ``reduce_gemm.py`` exists anyway.  At config B's
    ``dec3`` site that convolution has 8.4 million taps and a channel count of
    ``N = 1``, so the forward's reduction loop -- ``taps * ceil(Cin/BLOCK_K)``
    iterations, each carrying a six-compare boundary predicate -- runs 8.4
    million times with 15 of every 16 ``BLOCK_K`` lanes masked off, and there is
    no split-K anywhere in it.  Both numbers are asserted rather than described,
    because "too slow" is the kind of claim that rots.
    """
    problem = ConvProblem("reuse", 4, 5, (4, 5, 6), padding=(0, 0, 0))
    ops = reference.make_inputs(problem, seed=29, exact=True)
    expected = reference.reference(problem, ops, "bwd-weight")
    assert reference.is_exactly_representable(expected, torch.bfloat16)

    # (Cin, N, ID, IH, IW) convolved with (Cout, N, OD, OH, OW) -> (Cin, Cout, k)
    as_conv = conv3d_forward(
        ops["input"].transpose(0, 1).contiguous(memory_format=torch.channels_last_3d),
        ops["grad_output"].transpose(0, 1).contiguous(
            memory_format=torch.channels_last_3d),
        padding=0,
    )
    assert tuple(as_conv.shape) == (problem.cin, problem.cout, *problem.kernel)
    assert reference.compare(
        as_conv.transpose(0, 1), expected.to(torch.bfloat16)
    ).bitwise

    # And the shape of that same reuse at a real site.
    big = ConvProblem("dec3", 128, 64, (130, 258, 258), padding=(0, 0, 0))
    reused_taps = math.prod(big.out_spatial)
    cfg = default_config(big.cin * reused_taps, 1, big.cout, torch.bfloat16)
    assert reused_taps == 8_388_608
    assert cfg.BLOCK_K >= 16 and big.n == 1, (
        "the reused kernel's reduction is Cin=N=1 deep but BLOCK_K cannot go "
        "below the MFMA's kDim"
    )
    assert reused_taps * triton.cdiv(big.n, cfg.BLOCK_K) > 8e6


# ---------------------------------------------------------------------------
# Correctness: the tolerance standards
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("problem", EDGE + CORPUS_PAIRS,
                         ids=_ids(EDGE + CORPUS_PAIRS))
def test_no_worse_than_miopen(problem: ConvProblem):
    """The honest bar at realistic magnitudes, against MIOpen on the same data."""
    ops = reference.make_inputs(problem, seed=17)
    expected = reference.reference(problem, ops, "bwd-weight")
    incumbent_err = reference.compare(
        reference.incumbent(problem, ops, "bwd-weight"), expected
    )
    actual = _run(problem, ops)
    reference.assert_close(actual, expected, problem, "bwd-weight",
                           incumbent_error=incumbent_err)


@requires_gpu
def test_split_k_is_more_accurate_than_miopen_at_a_long_reduction():
    """A claim worth making in the other direction, for once.

    Splitting a 32k-term fp32 reduction into fixed chunks and summing the
    partials is not just reproducible, it is *more accurate* than one long
    accumulation -- the error of a sum of ``K`` terms grows like ``sqrt(K)`` and
    a two-level sum trades that for ``sqrt(K/S) + sqrt(S)``.  Measured here so
    that "deterministic" is not read as "at some cost in accuracy": Triton's
    error lands at the bf16 rounding limit of the output, and MIOpen's is
    several times larger.
    """
    problem = ConvProblem("acc", 64, 64, (34, 34, 34), padding=(0, 0, 0))
    ops = reference.make_inputs(problem, seed=31)
    expected = reference.reference(problem, ops, "bwd-weight")
    mine = reference.compare(_run(problem, ops), expected)
    theirs = reference.compare(
        reference.incumbent(problem, ops, "bwd-weight"), expected
    )
    assert mine.max_abs < theirs.max_abs, f"triton {mine} vs miopen {theirs}"


@requires_gpu
def test_fp32_accumulates_in_fp32():
    """``more_determinism`` runs the model in fp32, and the backward too.

    A tf32-style split dot would pass any bf16-sized tolerance, so the bound is
    fp32-sized and held against fp64.
    """
    problem = ConvProblem("fp32", 48, 32, (7, 9, 5), dtype="fp32")
    ops = reference.make_inputs(problem, seed=23)
    expected = reference.reference(problem, ops, "bwd-weight")
    actual = _run(problem, ops)
    assert actual.dtype is torch.float32
    report = reference.compare(actual, expected)
    peak = expected.abs().max().item()
    assert report.max_abs < 1e-4 * peak, f"looks like a reduced-precision dot: {report}"


@requires_gpu
@pytest.mark.parametrize("stride,padding", [(2, 1), (2, 0), (3, 2)])
def test_a_strided_convolution_is_served_correctly(stride, padding):
    """Backward-data refuses a stride; this direction does not, so it is tested.

    ScaFFold's corpus has no strided non-transposed convolution, so nothing else
    in this suite would exercise the ``o*s`` term at all, and an unexercised
    multiply that is *also* not refused by ``is_supported`` is the combination
    that returns a wrong gradient silently.
    """
    problem = ConvProblem("strided", 16, 24, (9, 11, 13),
                          stride=(stride,) * 3, padding=(padding,) * 3)
    ops = reference.make_inputs(problem, seed=37, exact=True)
    expected = reference.reference(problem, ops, "bwd-weight")
    assert reference.is_exactly_representable(expected, torch.bfloat16)
    assert reference.compare(_run(problem, ops),
                             expected.to(torch.bfloat16)).bitwise


# ---------------------------------------------------------------------------
# Determinism -- the property this milestone exists for
# ---------------------------------------------------------------------------


#: A shape whose split count is well above 1, so that the deterministic path is
#: actually exercising the workspace and the reduction pass rather than the
#: single-split shortcut that trivially cannot disagree with itself.
_DET = ConvProblem("determinism", 64, 64, (18, 34, 34), padding=(0, 0, 0))


@requires_gpu
def test_repeated_calls_are_bitwise_reproducible_in_process():
    problem = _DET
    ops = reference.make_inputs(problem, seed=71)
    cfg = bwd_weight_config(problem.cout, problem.cin, problem.kernel,
                            math.prod(problem.out_spatial), torch.bfloat16)
    assert split_count(cfg, problem.cout, problem.cin, problem.tap_count,
                       math.prod(problem.out_spatial),
                       problem.out_spatial[2])[0] > 1, "not exercising split-K"
    first = _run(problem, ops)
    for _ in range(4):
        assert torch.equal(first, _run(problem, ops))


#: The ``k=1`` segmentation head, at a volume that splits ~800 ways, in **fp32**.
#: The dtype is the entire point -- see the negative-control test below.
_DET_K1 = ConvProblem("determinism-k1", 64, 6, (64, 64, 64), (1, 1, 1),
                      padding=(0, 0, 0), dtype="fp32")


@requires_gpu
@pytest.mark.parametrize("problem", [_DET, _DET_K1], ids=["k3-bf16", "k1-fp32"])
def test_the_atomic_path_is_not_bitwise_reproducible(problem: ConvProblem):
    """The negative control, and the reason the default is not the atomic one.

    Without this the reproducibility test above could pass on a kernel that was
    reproducible for some unrelated reason -- a grid too small to race, say --
    and the claim would be about the shape rather than about the mechanism.
    Float addition is not associative and ``tl.atomic_add`` fixes no order, so
    at 100-odd racing splits a repeat that agrees bitwise every time would mean
    the atomic path is not doing what it says.

    **The second cell is the interesting one, and it is the reason this test is
    parametrized at all.**  The ``k=1`` head at ``64 -> 6 @ 128^3`` was recorded
    elsewhere in this project as a shape where the atomic control "reproduced by
    scheduling accident".  That is not the mechanism.  The atomic accumulator is
    fp32 and the *result* is bf16, so a reordering that perturbs the sum at the
    fp32 ulp is simply invisible after the cast -- measured, the perturbation
    there is about 300x below one bf16 ulp of the output.  The splits are racing
    the whole time; the race is under the resolution of the dtype it is being
    observed in.  Run the identical shape in fp32 and the control fires every
    single time (15/15, ~400 ulps of the fp32 output).

    That distinction matters because it says what the control *can* certify: it
    is informative wherever the reordering is resolvable in the output dtype,
    and it certifies nothing on a short-``Cout`` bf16 shape -- a change that
    made the deterministic path non-deterministic at the ``k=1`` head would be
    invisible in a bf16 cell.  So the ``k=1`` head is covered here in the dtype
    where the control has teeth.

    If this ever goes flaky it is worth reading as a result rather than as a
    flake: it would mean the splits stopped racing.
    """
    dtype = reference.torch_dtype(problem)
    k_total = problem.n * math.prod(problem.out_spatial)
    cfg = bwd_weight_config(problem.cout, problem.cin, problem.kernel, k_total,
                            dtype)
    splits = split_count(cfg, problem.cout, problem.cin, problem.tap_count,
                         k_total, problem.out_spatial[2])[0]
    assert splits > 8, f"{splits} splits: too few writers to contend"

    ops = reference.make_inputs(problem, seed=71, dtype=dtype)
    first = _run(problem, ops, deterministic=False)
    differed = any(
        not torch.equal(first, _run(problem, ops, deterministic=False))
        for _ in range(15)
    )
    assert differed, (
        f"16 runs of the atomic path agreed bitwise at {splits} splits; either "
        "the splits are not racing, or the reordering is below one ulp of "
        f"{dtype} and this cell certifies nothing"
    )


_CHILD = textwrap.dedent(
    """
    import hashlib, sys, torch
    sys.path.insert(0, {repo!r})
    from triton_conv3d import reference
    from triton_conv3d.reduce_gemm import conv3d_backward_weight
    from triton_conv3d.shapes import ConvProblem
    p = ConvProblem("determinism", 64, 64, (18, 34, 34), padding=(0, 0, 0))
    ops = reference.make_inputs(p, seed=71)
    gw = conv3d_backward_weight(ops["input"], p.weight_shape,
                                ops["grad_output"], p.stride, p.padding)
    # bf16 has no numpy dtype; widening to fp32 is exact, so the digest still
    # answers the bitwise question.
    print(hashlib.sha256(gw.float().cpu().numpy().tobytes()).hexdigest())
    """
)


@requires_gpu
def test_three_separate_processes_agree_bitwise():
    """Process to process, which is the half of the claim a loop cannot test.

    An in-process repeat shares the allocator state, the JIT cache and the
    module-level tuning table, so it would still pass if any of those were what
    fixed the reduction order.  Separate interpreters share none of it, which is
    what makes this the test that the *shape* determines the split count.
    """
    repo = str(pathlib.Path(__file__).resolve().parents[2])
    digests = []
    for _ in range(3):
        proc = subprocess.run([sys.executable, "-c", _CHILD.format(repo=repo)],
                              capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, proc.stderr[-2000:]
        digests.append(proc.stdout.strip().splitlines()[-1])
    assert len(set(digests)) == 1, digests


# ---------------------------------------------------------------------------
# Entry-point behaviour
# ---------------------------------------------------------------------------


@requires_gpu
def test_the_output_is_a_channels_last_weight_of_the_right_shape():
    """The GEMM writes ``[Cout][tap][Cin]``, which *is* channels_last_3d.

    Worth asserting rather than assuming: it is the reason this direction needs
    no layout transform at all, and a future change to the epilogue that
    produced a contiguous weight instead would still pass every value test in
    this file while costing the integration a permute per parameter per step.
    """
    problem = ConvProblem("shape", 16, 40, (3, 11, 5))
    ops = reference.make_inputs(problem, seed=61)
    gw = _run(problem, ops)
    ref = torch.nn.grad.conv3d_weight(
        ops["input"], problem.weight_shape, ops["grad_output"],
        stride=problem.stride, padding=problem.padding,
    )
    assert gw.shape == ref.shape
    assert gw.is_contiguous(memory_format=torch.channels_last_3d)
    assert gw.stride(1) == 1


@requires_gpu
def test_an_out_the_kernel_would_overrun_is_refused():
    """The ``Cout`` extent is invisible to a stride check, and it is the extent.

    ``[Cout][kd][kh][kw][Cin]`` strides are
    ``(taps*Cin, 1, kh*kw*Cin, kw*Cin, Cin)`` -- **not one of them mentions
    Cout**.  So a gradient allocated for ``Cout=8`` is stride-identical to one
    allocated for ``Cout=64`` with the same ``Cin`` and kernel, and the
    reduction pass takes its element count from ``weight_shape`` rather than
    from ``gw``: passing the small one used to be accepted and wrote 55 296
    elements into a 6 912-element allocation.  No fault and no exception -- the
    write lands in whatever the caching allocator has next, and some other live
    tensor is wrong later.

    The other three clauses are here for the same reason they are in the
    function: a foreign device is a pointer this kernel will happily
    dereference (ScaFFold runs four ranks per node), and a mismatched dtype
    silently changes the dtype of the gradient the caller gets back.
    """
    k = (3, 3, 3)
    x = torch.randn((1, 32, 6, 6, 6), device="cuda", dtype=torch.bfloat16
                    ).contiguous(memory_format=torch.channels_last_3d)
    gy = torch.randn((1, 64, 6, 6, 6), device="cuda", dtype=torch.bfloat16
                     ).contiguous(memory_format=torch.channels_last_3d)
    ws = (64, 32, *k)

    right = grad_weight_empty(64, 32, k, dtype=torch.bfloat16, device="cuda")
    small = grad_weight_empty(8, 32, k, dtype=torch.bfloat16, device="cuda")
    # The trap, stated: the guard that used to be here could not tell these two
    # apart, because the only thing that differs is an extent.
    assert small.stride() == right.stride()
    assert small.numel() * 8 == right.numel()

    for bad, what in (
        (small, "shape"),
        (grad_weight_empty(64, 32, k, dtype=torch.float32, device="cuda"),
         "dtype"),
        (grad_weight_empty(64, 32, k, dtype=torch.bfloat16, device="cpu"),
         "device"),
        (torch.empty(ws, device="cuda", dtype=torch.bfloat16), "strides"),
    ):
        with pytest.raises(ValueError, match="out="):
            conv3d_backward_weight(x, ws, gy, padding=1, out=bad)
    # ...and the buffer that *is* right is still accepted, so the guard is not
    # simply refusing everything.
    assert conv3d_backward_weight(x, ws, gy, padding=1,
                                  out=right).data_ptr() == right.data_ptr()


@requires_gpu
def test_the_gradient_buffer_is_allocated_in_the_layout_it_is_used_in():
    """One allocation, no copy.

    ``torch.empty(shape).contiguous(memory_format=channels_last_3d)`` allocates
    NCDHW and then runs a permuting device copy to reach the layout it was
    always going to be asked for.  The contents are undefined either way, so
    the copy transports nothing -- it is pure waste on a buffer this direction
    allocates once per parameter per step, and the identical defect in the
    forward measured 235x the cost of the one-shot allocation.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    seen: list[str] = []

    class _Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(str(func))
            return func(*args, **(kwargs or {}))

    with _Record():
        gw = grad_weight_empty(64, 32, (3, 3, 3), dtype=torch.bfloat16,
                               device="cuda")
    assert gw.shape == (64, 32, 3, 3, 3)
    assert gw.is_contiguous(memory_format=torch.channels_last_3d)
    assert not [op for op in seen if "copy" in op or "clone" in op], seen
    assert len(seen) == 1, f"expected one allocation and nothing else: {seen}"


@requires_gpu
def test_hoisted_workspace_and_out_are_equivalent():
    """Both are optimizations, so both must change nothing."""
    problem = ConvProblem("hoist", 32, 48, (10, 12, 16), padding=(0, 0, 0))
    ops = reference.make_inputs(problem, seed=53, exact=True)
    inline = _run(problem, ops)

    k_total = math.prod(problem.out_spatial)
    cfg = bwd_weight_config(problem.cout, problem.cin, problem.kernel, k_total,
                            torch.bfloat16)
    splits, _ = split_count(cfg, problem.cout, problem.cin, problem.tap_count,
                            k_total, problem.out_spatial[2])
    ws = torch.empty(
        workspace_elements(splits, problem.cout, problem.cin, problem.kernel),
        dtype=torch.float32, device="cuda",
    )
    gw = grad_weight_empty(problem.cout, problem.cin, problem.kernel,
                           dtype=torch.bfloat16, device="cuda")
    hoisted = _run(problem, ops, workspace=ws, out=gw)
    assert hoisted.data_ptr() == gw.data_ptr()
    assert torch.equal(inline, hoisted)

    # An undersized workspace has to say *how big* it needed to be.  A hoisted
    # workspace is sized once, out of the step, from a number someone read
    # somewhere -- and the number that was published for this corpus was
    # understated by 1.46x, so the first thing that caller sees is this
    # exception at step 1 with no way to compute the right size from it.
    need = workspace_elements(splits, problem.cout, problem.cin, problem.kernel)
    with pytest.raises(ValueError, match=rf"at least {need} float32 elements"):
        _run(problem, ops, workspace=ws[:8])
    with pytest.raises(ValueError, match=rf"at least {need} float32 elements"):
        _run(problem, ops, workspace=ws.double())


@requires_gpu
def test_ncdhw_operands_are_converted_rather_than_misread():
    """The addressing assumes ``stride_c == 1`` on both activations.

    An NCDHW operand read with NDHWC strides gives a full-rate kernel and a
    completely wrong gradient.  ScaFFold's own backward can hand us either
    layout depending on what produced the tensor, so this is not hypothetical.
    """
    problem = ConvProblem("layout", 24, 16, (5, 6, 7))
    ops = reference.make_inputs(problem, seed=41, exact=True)
    ndhwc = _run(problem, ops)
    nc = {k: (v.contiguous() if torch.is_tensor(v) else v) for k, v in ops.items()}
    assert nc["input"].stride(1) != 1
    assert torch.equal(ndhwc, _run(problem, nc))
