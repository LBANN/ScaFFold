# SPDX-License-Identifier: (Apache-2.0)
"""Tests for backward-data, which is the forward kernel on a transformed weight.

Because no new kernel is introduced, these tests are not re-testing the gather
-- ``test_gather_gemm.py`` does that.  What they test is the *transform*, and
the transform is the part with a uniquely nasty failure mode: it flips the tap
axes and swaps the two channel axes, and getting either half wrong produces a
gradient that is the right shape, the right magnitude, smooth, and wrong.  A
tolerance test cannot see that.  Two of the tests here exist purely to prove the
bitwise standard is not vacuous:

* :func:`test_bitwise_standard_rejects_a_shifted_gather` -- shift the upstream
  gradient by one voxel and the comparison must fail;
* :func:`test_an_unflipped_weight_is_detected` -- omit the tap flip and the
  comparison must fail.  This is the specific bug the whole module could have,
  and without this test a passing suite would not rule it out.

The other thing these tests cover that the forward's do not is that
backward-data's effective convolution is **always padded** for ``k > 1``, even
when the forward was not: DistConv issues an unpadded ``130^3`` convolution and
its backward-data has ``p' = 2``.  So the halo'd corpus is parametrized here in
its own right rather than only in its logical, padded form.
"""

from __future__ import annotations

import math

import pytest
import torch
import triton

from triton_conv3d import reference
from triton_conv3d.bwd_data import (
    bwd_data_config,
    bwd_data_padding,
    conv3d_backward_data,
    is_supported_bwd_data,
)
from triton_conv3d.gather_gemm import candidate_configs, default_config, to_rsck
from triton_conv3d.shapes import ConvProblem, edge_cases, scaffold_corpus

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

#: The synthetic corpus, minus the transposed upsample (a later milestone).
EDGE = [p for p in edge_cases() if not p.transposed]


def _corpus_channel_pairs() -> list[ConvProblem]:
    """Every distinct ``(Cin, Cout, kernel)`` in the corpus, at a testable volume.

    The forward suite selects corpus problems by *volume* -- small enough for an
    fp64 reference -- and that works there.  It does not work here, and the way
    it fails is worth recording because the first version of this file shipped
    with it: backward-data reduces over ``Cout * taps``, so the surviving
    problems are exactly the deep, wide ones, and a sum of ``27648`` random
    signs runs to about 500 while bf16 holds integers only to 256.  Every single
    corpus case then hit ``is_exactly_representable`` and skipped, and the file
    reported "89 passed" with zero real-shape coverage.

    Restating each channel pair at ``6x7x8`` instead keeps what the corpus is
    *for* -- the channel widths, and with them ``EVEN_K``/``EVEN_N``, the tile
    selection and the 512-byte row strides -- while making the reference cheap.

    **All three paddings** are generated, because ScaFFold issues all three and
    they are three different problems (``shapes.py``'s module docstring):

    * ``p = (1,1,1)`` -- what the adapter hands the kernel at one GPU, and the
      module's own statement everywhere;
    * ``p = (0,1,1)`` -- what it hands the kernel at two or four GPUs, where D
      is halo'd and H and W are not.  Anisotropic, which no other case in this
      file is: the backward's ``p'`` is then ``(2,1,1)``, so one axis reads a
      two-voxel boundary shell and the other two read one;
    * ``p = (0,0,0)`` -- what upstream DistConv hands MIOpen, and the form every
      published baseline was measured in.

    None of the three subsumes another, and the middle one is the one that used
    to be missing.
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
#: fp64.  Used only for the fp32 test below: their bf16 references are never
#: exactly representable, which is the whole point of the note above.
CORPUS_SMALL = [
    p for p in scaffold_corpus()
    if not p.transposed
    and math.prod(p.halo_variant.spatial) * max(p.cin, p.cout) <= 1 << 22
]
CORPUS_SMALL += [p.halo_variant for p in CORPUS_SMALL]


def _ids(problems):
    return [p.name or p.label for p in problems]


def _run(problem: ConvProblem, ops: dict, **kwargs) -> torch.Tensor:
    return conv3d_backward_data(
        ops["grad_output"], ops["weight"], problem.input_shape,
        problem.stride, problem.padding, **kwargs,
    )


# ---------------------------------------------------------------------------
# The algebra, before any GPU is involved
# ---------------------------------------------------------------------------


def test_the_padding_identity_is_the_one_the_derivation_claims():
    """``p' = dil*(k-1) - p``, and the output extent then lands on the input's.

    Stated as a test rather than left in a docstring because every other file in
    this module depends on it and it is one sign error away from producing a
    gradient of the wrong *shape* -- which at least fails loudly -- or, at
    ``k=3, p=1``, the right shape and the wrong answer, which does not.
    """
    assert bwd_data_padding(1, 1, 3) == (1, 1, 1)
    assert bwd_data_padding(0, 1, 3) == (2, 2, 2)      # the halo'd form
    assert bwd_data_padding(0, 1, 1) == (0, 0, 0)      # k=1: no gather at all
    assert bwd_data_padding((0, 1, 1), 1, (1, 3, 3)) == (0, 1, 1)
    assert bwd_data_padding(1, 2, 3) == (3, 3, 3)      # dilation widens the reach

    # And the extent identity: OD + 2p' - dil*(k-1) == ID, for every combination.
    for k in (1, 2, 3, 5):
        for dil in (1, 2, 3):
            for p in range(0, dil * (k - 1) + 1):
                for in_d in (1, 4, 17):
                    out_d = in_d + 2 * p - dil * (k - 1)
                    if out_d < 1:
                        continue
                    pp = bwd_data_padding(p, dil, k)[0]
                    assert out_d + 2 * pp - dil * (k - 1) == in_d, (k, dil, p, in_d)


def test_flipping_every_tap_axis_is_complementing_the_fused_index():
    """The identity the kernel's ``taps - 1 - dij`` rests on.

    The transform this replaced -- ``permute(2,3,4,0,1).flip((0,1,2))`` --
    materialized a whole second copy of every weight, once per optimizer step,
    to express a *reindexing*.  The kernel now flips by walking the fused tap
    index backwards, which is only the same thing because the fused index is a
    mixed-radix number and complementing every digit complements the number.
    That is exactly the sort of claim that is obvious, load-bearing and one
    off-by-one away from a silently wrong gradient, so it is checked over
    anisotropic kernels rather than argued.

    ``k=(1,3,1)``-shaped cases are in the list on purpose: an axis of extent 1
    contributes ``0`` to both sides, which is where a formula that got the radix
    order wrong would still look right.
    """
    for kd, kh, kw in [(3, 3, 3), (1, 1, 1), (2, 3, 4), (1, 3, 1), (5, 1, 2)]:
        taps = kd * kh * kw
        for d in range(kd):
            for i in range(kh):
                for j in range(kw):
                    flipped = (((kd - 1 - d) * kh + (kh - 1 - i)) * kw
                               + (kw - 1 - j))
                    fused = (d * kh + i) * kw + j
                    assert flipped == taps - 1 - fused, (kd, kh, kw, d, i, j)


# ---------------------------------------------------------------------------
# Configuration legality -- the failure mode is silent, so it is checked apart
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem", EDGE + CORPUS_PAIRS,
                         ids=_ids(EDGE + CORPUS_PAIRS))
def test_selected_config_is_legal_for_every_shape(problem: ConvProblem):
    """The config picked for backward-data must still reach the matrix core.

    Not implied by the forward's version of this test: the effective GEMM has
    ``Cin`` and ``Cout`` swapped, so a shape whose forward tile is legal can
    have a backward tile that is not -- ``Cout=6`` becomes ``BLOCK_K`` rather
    than ``BLOCK_N``, and ``BLOCK_K`` is the one with the hard MFMA constraint.
    """
    dtype = reference.torch_dtype(problem)
    cfg = bwd_data_config(problem.output_shape, problem.cin, problem.kernel, dtype,
                          padding=problem.padding, dilation=(1, 1, 1))
    assert cfg.validate(dtype) is None, f"{problem.label}: {cfg} -> {cfg.validate(dtype)}"


@pytest.mark.parametrize("problem", CORPUS_PAIRS, ids=_ids(CORPUS_PAIRS))
def test_every_backward_candidate_config_is_legal(problem: ConvProblem):
    """The sweep that produced ``_TUNED_BWD`` must not contain an FMA kernel.

    Same reasoning as the forward's: an illegal config runs and returns the
    right answer slowly, so the sweep's reported winner could be one.  The
    argument order is what differs -- the candidate list is generated for the
    *effective* widths.
    """
    dtype = reference.torch_dtype(problem)
    m = problem.n * math.prod(problem.spatial)
    cfgs = candidate_configs(m, problem.cout, problem.cin, dtype)
    assert cfgs
    for cfg in cfgs:
        assert cfg.validate(dtype) is None, f"{cfg}: {cfg.validate(dtype)}"


# ---------------------------------------------------------------------------
# Support predicate
# ---------------------------------------------------------------------------


@requires_gpu
def test_is_supported_declines_what_the_algebra_cannot_express():
    """Two of these refusals are backward-only and both are load-bearing.

    ``stride > 1`` makes the backward a scatter into a sub-lattice, and
    ``padding > dil*(k-1)`` makes ``p'`` negative -- a crop.  Neither is a
    forward gather, and neither raises anything by itself: the kernel would run
    and write a plausible, wrong gradient.
    """
    gy = torch.empty((1, 8, 4, 4, 4), device="cuda", dtype=torch.bfloat16)
    w = torch.empty((8, 8, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    shape = (1, 8, 4, 4, 4)
    assert is_supported_bwd_data(gy, w, shape, padding=1)

    assert not is_supported_bwd_data(gy, w, shape, stride=2, padding=1)
    assert not is_supported_bwd_data(gy, w, shape, padding=3)   # p > dil*(k-1)
    assert not is_supported_bwd_data(gy, w, shape, padding=1, groups=2)
    assert not is_supported_bwd_data(gy, w.float(), shape, padding=1)
    # Cin of the weight must match the gradient being asked for ...
    assert not is_supported_bwd_data(gy, w, (1, 4, 4, 4, 4), padding=1)
    # ... Cout of the weight must match grad_output ...
    assert not is_supported_bwd_data(
        gy, torch.empty((4, 8, 3, 3, 3), device="cuda", dtype=torch.bfloat16),
        shape, padding=1,
    )
    # ... and grad_output's spatial extent must be the one this problem produces.
    assert not is_supported_bwd_data(gy, w, (1, 8, 6, 4, 4), padding=1)
    assert is_supported_bwd_data(gy, w, (1, 8, 6, 6, 6), padding=0)


@requires_gpu
def test_is_supported_declines_an_empty_batch():
    """The predicate bounded every spatial extent below and not ``N``.

    Degenerate rather than dangerous -- the grid comes out empty -- but this
    gate's own "every output voxel must exist" reasoning excludes a batch with no
    samples in it, and a ``True`` here is the gate asserting something it never
    looked at.  The cost of declining is one call's worth of MIOpen on a problem
    that has nothing to compute.
    """
    gy = torch.empty((0, 8, 4, 4, 4), device="cuda", dtype=torch.bfloat16)
    w = torch.empty((8, 8, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    assert not is_supported_bwd_data(gy, w, (0, 8, 4, 4, 4), padding=1)
    with pytest.raises(NotImplementedError):
        conv3d_backward_data(gy, w, (0, 8, 4, 4, 4), padding=1)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_is_supported_declines_operands_on_different_devices():
    """Both operands on *a* GPU is not both on the *same* GPU.

    Triton launches on the current device and dereferences the foreign pointer
    regardless.  ScaFFold runs four ranks to a node, and with peer access enabled
    that reads another rank's weights rather than faulting -- a wrong gradient
    with no symptom at all.  Skipped, not absent, on a single-GPU box.
    """
    gy = torch.empty((1, 8, 4, 4, 4), device="cuda:0", dtype=torch.bfloat16)
    w = torch.empty((8, 8, 3, 3, 3), device="cuda:0", dtype=torch.bfloat16)
    assert is_supported_bwd_data(gy, w, (1, 8, 4, 4, 4), padding=1)
    assert not is_supported_bwd_data(gy, w.to("cuda:1"), (1, 8, 4, 4, 4), padding=1)


# ---------------------------------------------------------------------------
# Correctness: the bitwise standard
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("problem", EDGE, ids=_ids(EDGE))
def test_exact_operands_match_bitwise(problem: ConvProblem):
    """Bitwise against ``torch.autograd.grad`` in fp64, on the nasty shapes.

    The same synthetic corpus as the forward, and it earns its place twice over
    here: ``Cin=3`` and ``Cout=6`` land on the GEMM's *N* rather than its K, and
    ``smaller_than_kernel`` is where the flipped gather masks every tap
    somewhere.
    """
    ops = reference.make_inputs(problem, seed=3, exact=True)
    expected = reference.reference(problem, ops, "bwd-data")
    dtype = reference.torch_dtype(problem)
    if not reference.is_exactly_representable(expected, dtype):
        pytest.skip("realized magnitudes exceed the mantissa in this dtype")
    actual = _run(problem, ops)
    report = reference.compare(actual, expected.to(dtype))
    assert report.bitwise, f"{problem.label}: {report}"


@requires_gpu
@pytest.mark.parametrize("problem", CORPUS_PAIRS, ids=_ids(CORPUS_PAIRS))
def test_corpus_channel_pairs_match_bitwise(problem: ConvProblem):
    """Every channel pair ScaFFold runs, in all three paddings, bitwise in bf16.

    The three arms are the three forms of the same site, and each has something
    the others do not.  ``p=0`` (DistConv's) has a backward padding of 2, so it
    reads a boundary shell two voxels thick, which no forward convolution in
    this project ever does.  ``p=(0,1,1)`` (the adapter's, sharded) is
    anisotropic: ``p'`` is ``(2,1,1)`` and the two shell thicknesses coexist in
    one kernel.  ``p=1`` (the adapter's, unsharded) is the ordinary one.
    Running only one of them would leave a shape ScaFFold actually issues
    untested; the middle one was the one missing until 2026-08-04.

    The five deepest pairs skip here and are picked up by the fp32 test below;
    see :func:`test_the_bitwise_corpus_is_not_entirely_skipped` for why that is
    checked rather than assumed.
    """
    ops = reference.make_inputs(problem, seed=5, exact=True)
    expected = reference.reference(problem, ops, "bwd-data")
    dtype = reference.torch_dtype(problem)
    if not reference.is_exactly_representable(expected, dtype):
        pytest.skip("realized magnitudes exceed the mantissa in this dtype")
    actual = _run(problem, ops)
    assert reference.compare(actual, expected.to(dtype)).bitwise


@requires_gpu
def test_the_bitwise_corpus_is_not_entirely_skipped():
    """A regression guard on this file, not on the kernel.

    ``is_exactly_representable`` declining is the correct behaviour, but if it
    declines for *every* parametrized case the suite reports a wall of passes
    and tests nothing.  That is exactly what the first version of this file did.
    So pin a floor: most of the corpus's channel pairs must actually reach the
    bitwise comparison in bf16.
    """
    exact = 0
    for problem in CORPUS_PAIRS:
        ops = reference.make_inputs(problem, seed=5, exact=True)
        expected = reference.reference(problem, ops, "bwd-data")
        exact += reference.is_exactly_representable(
            expected, reference.torch_dtype(problem)
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

    ``Cout >= 512`` makes backward-data's reduction 13824 or 27648 terms long,
    and no choice of ``{-1,0,1}`` operands keeps that inside bf16's 8-bit
    mantissa -- it is a property of the arithmetic, not of the test.  fp32 has
    24 bits, which covers it with room to spare, and the addressing under test
    is dtype-independent: what changes is the MFMA intrinsic and therefore the
    legal ``BLOCK_K``, so this is also the only bitwise coverage the fp32 tile
    selection gets at real widths.

    This test does not skip.  If the fp32 reference is ever not exact either,
    that is a fact worth failing on rather than stepping around.
    """
    ops = reference.make_inputs(problem, seed=7, exact=True, dtype=torch.float32)
    expected = reference.reference(problem, ops, "bwd-data")
    assert reference.is_exactly_representable(expected, torch.float32)
    actual = _run(problem, ops)
    assert actual.dtype is torch.float32
    assert reference.compare(actual, expected.to(torch.float32)).bitwise


@requires_gpu
def test_bitwise_standard_rejects_a_shifted_gather():
    """Prove the comparison discriminates: a one-voxel shift must fail it.

    Same argument as the forward's version -- ``{-1,0,1}`` operands could in
    principle make everything agree, and this project has shipped two vacuous
    exact tests before.
    """
    problem = ConvProblem("shift", 16, 16, (6, 6, 6))
    ops = reference.make_inputs(problem, seed=11, exact=True)
    actual = _run(problem, ops)
    correct = reference.reference(problem, ops, "bwd-data").to(torch.bfloat16)
    assert reference.compare(actual, correct).bitwise

    shifted = torch.roll(ops["grad_output"], shifts=1, dims=-1)
    wrong = reference.reference(
        problem, {**ops, "grad_output": shifted}, "bwd-data"
    ).to(torch.bfloat16)
    assert not reference.compare(actual, wrong).bitwise, (
        "a one-voxel shift of the upstream gradient produced a bitwise-identical "
        "result; the comparison is not discriminating"
    )


@requires_gpu
def test_an_unflipped_weight_is_detected():
    """The one bug this module can uniquely have, pinned.

    The gather reads tap ``t`` of this direction from tap ``taps-1-t`` of the
    weight.  Omitting that -- the single most plausible mistake in the whole
    file, and now a constexpr in the kernel rather than a ``torch.flip``, which
    makes it easier to get wrong and no easier to see -- still produces a
    correctly shaped, correctly scaled, smooth gradient, and would pass every
    tolerance test written.  So construct exactly that wrong answer and require
    a mismatch.

    ``padding=1`` with ``k=3`` is deliberate: it is the case where flipped and
    unflipped agree on the *shape*, so nothing else catches it.
    """
    problem = ConvProblem("flip", 16, 16, (6, 7, 8))
    ops = reference.make_inputs(problem, seed=13, exact=True)
    actual = _run(problem, ops)
    correct = reference.reference(problem, ops, "bwd-data").to(torch.bfloat16)
    assert reference.compare(actual, correct).bitwise

    unflipped = reference.reference(
        problem, {**ops, "weight": ops["weight"].flip(2, 3, 4)}, "bwd-data"
    ).to(torch.bfloat16)
    assert unflipped.shape == actual.shape
    assert not reference.compare(actual, unflipped).bitwise, (
        "a weight with the taps un-flipped gave a bitwise-identical gradient; "
        "the kernel's W_FLIP is untested by this suite"
    )


@requires_gpu
@pytest.mark.parametrize("problem", EDGE, ids=_ids(EDGE))
def test_every_config_gives_the_same_answer(problem: ConvProblem):
    """The tuning surface, not one point on it.

    ``_TUNED_BWD`` is free to pick any of these, and the backward's boundary
    shell is two voxels thick rather than one -- so a mask that is right when
    ``BLOCK_M`` divides the row length and wrong when it does not has more room
    to hide here than in the forward.  That argument applies with most force to
    the cases this stopped short of when it swept only ``EDGE[:8]``: ``batched``
    (the only ``n > 1`` shape), ``kernel_aniso``, ``smaller_than_kernel``,
    ``unpadded`` -- whose backward is padded where its forward is not -- and both
    non-bf16 dtypes, which move the MFMA reduction depth and so the set of legal
    ``BLOCK_K`` values.
    """
    ops = reference.make_inputs(problem, seed=2, exact=True)
    expected = reference.reference(problem, ops, "bwd-data")
    dtype = reference.torch_dtype(problem)
    if not reference.is_exactly_representable(expected, dtype):
        pytest.skip("realized magnitudes exceed the mantissa in this dtype")
    expected = expected.to(dtype)
    m = problem.n * math.prod(problem.spatial)
    # Effective widths: the reduction is over Cout and the GEMM's N is Cin.
    cfgs = candidate_configs(m, problem.cout, problem.cin, dtype, group_ms=(6, 8))
    cfgs = list(dict.fromkeys(
        cfgs + [default_config(m, problem.cout, problem.cin, dtype)]
    ))
    ran = 0
    for cfg in cfgs:
        try:
            actual = _run(problem, ops, config=cfg)
        except triton.runtime.errors.OutOfResources:
            # Operands that do not fit in 64 KiB of LDS.  A loud failure, so no
            # static guard is wanted -- the sweep skips it and so does this.
            continue
        ran += 1
        assert reference.compare(actual, expected).bitwise, f"{problem.label} {cfg}"
    assert ran, "no candidate configuration was runnable"


# ---------------------------------------------------------------------------
# Correctness: the tolerance standards
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("problem", EDGE + CORPUS_PAIRS,
                         ids=_ids(EDGE + CORPUS_PAIRS))
def test_no_worse_than_miopen(problem: ConvProblem):
    """The honest bar at realistic magnitudes, against MIOpen on the same data.

    Worth stating separately from the forward's because backward-data's
    reduction is over ``Cout * taps`` rather than ``Cin * taps``, so on the
    asymmetric decoder convolutions the two directions accumulate over
    different lengths and inherit different error.
    """
    ops = reference.make_inputs(problem, seed=17)
    expected = reference.reference(problem, ops, "bwd-data")
    incumbent_err = reference.compare(
        reference.incumbent(problem, ops, "bwd-data"), expected
    )
    actual = _run(problem, ops)
    reference.assert_close(actual, expected, problem, "bwd-data",
                           incumbent_error=incumbent_err)


@requires_gpu
def test_fp32_accumulates_in_fp32():
    """``more_determinism`` runs the model in fp32, and the backward too.

    A tf32-style split dot would pass any bf16-sized tolerance, so the bound is
    fp32-sized and held against fp64.
    """
    problem = ConvProblem("fp32", 48, 32, (7, 9, 5), dtype="fp32")
    ops = reference.make_inputs(problem, seed=23)
    expected = reference.reference(problem, ops, "bwd-data")
    actual = _run(problem, ops)
    assert actual.dtype is torch.float32
    report = reference.compare(actual, expected)
    peak = expected.abs().max().item()
    assert report.max_abs < 1e-4 * peak, f"looks like a reduced-precision dot: {report}"


# ---------------------------------------------------------------------------
# Entry-point behaviour
# ---------------------------------------------------------------------------


@requires_gpu
def test_out_buffer_is_written_in_place_and_is_validated():
    """``out=`` is forwarded straight to the forward entry point, unexamined.

    Handing a preallocated gradient buffer to the backward is exactly what a
    DistConv integration does, and both halves of the hole were reachable from
    here: an undersized buffer wrote 10752 elements into a 256-element
    allocation with no error, and an NCDHW buffer returned a gradient with
    ``max_abs = 99.0``.  ``reduce_gemm`` validated the same parameter and this
    direction did not.

    The check lives in the forward, and that is exact rather than approximate:
    the effective forward's output shape *is* ``input_shape``.  This test is what
    says so.
    """
    problem = ConvProblem("out", 16, 24, (4, 5, 6))
    ops = reference.make_inputs(problem, seed=67, exact=True)
    expected = _run(problem, ops)
    assert tuple(expected.shape) == problem.input_shape

    buf = torch.empty_like(expected)
    got = _run(problem, ops, out=buf)
    assert got.data_ptr() == buf.data_ptr(), "out= was allocated over, not written"
    assert torch.equal(got, expected)

    bf16 = torch.bfloat16
    with pytest.raises(ValueError):
        _run(problem, ops, out=torch.empty((1, 16, 2, 2, 2), device="cuda", dtype=bf16))
    with pytest.raises(ValueError):   # right shape, NCDHW
        _run(problem, ops,
             out=torch.empty(problem.input_shape, device="cuda", dtype=bf16))
    with pytest.raises(ValueError):
        _run(problem, ops, out=torch.empty_like(expected, dtype=torch.float32))


@requires_gpu
def test_hoisted_weight_buffer_is_validated():
    """``weight_rsck`` supplies every weight value; ``weight`` supplies a shape.

    So a buffer belonging to another parameter -- a stale cache entry is the
    realistic way to get one -- is a smooth, correctly shaped, entirely wrong
    gradient.  This direction has its own trap on top of the forward's: the
    buffer it takes is the **forward's** ``(kd, kh, kw, Cin, Cout)``, so the
    transposed spelling, which is what a reader who knows backward-data reduces
    over ``Cout`` would reach for, has to be rejected rather than quietly
    transposing the answer.
    """
    problem = ConvProblem("wr", 16, 24, (4, 5, 6))
    ops = reference.make_inputs(problem, seed=67, exact=True)
    good = to_rsck(ops["weight"])
    assert torch.equal(_run(problem, ops, weight_rsck=good), _run(problem, ops))

    other = torch.randn((24, 16, 1, 1, 1), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _run(problem, ops, weight_rsck=to_rsck(other))
    with pytest.raises(ValueError):
        _run(problem, ops, weight_rsck=good.float())
    # (kd, kh, kw, Cout, Cin) -- the layout the deleted ``to_bwd_rsck`` produced.
    with pytest.raises(ValueError):
        _run(problem, ops, weight_rsck=good.transpose(3, 4).contiguous())


@requires_gpu
def test_every_weight_layout_gives_the_same_gradient():
    """The parameter is read where it lies, so its strides pick the B load.

    Three layouts reach three different ``W_ORDER``/copy decisions and must not
    reach three different answers.  Bitwise, not close: they are the same
    multiply-accumulate in the same order, and anything less would mean the
    layout had leaked into the arithmetic.

    The RSCK-strided case is the one worth having a test for.  It is a weight
    with PyTorch's shape and this kernel's storage order, which is what an
    integration that wanted the forward's B tile contiguous would allocate; here
    it is the layout in which ``weight_rsck`` and ``weight`` are the *same
    tensor*, so it is also the case that would hide a mix-up between them.
    """
    problem = ConvProblem("layouts", 32, 48, (4, 5, 6))
    ops = reference.make_inputs(problem, seed=53, exact=True)
    w = ops["weight"]
    cout, cin, kd, kh, kw = w.shape
    layouts = {
        "channels_last": w.contiguous(memory_format=torch.channels_last_3d),
        "contiguous": w.contiguous(),
        "rsck_strided": (w.permute(2, 3, 4, 1, 0).contiguous()
                         .permute(4, 3, 0, 1, 2)),
    }
    ref = _run(problem, ops)
    for name, wl in layouts.items():
        assert torch.equal(wl, w), name          # same values, different strides
        got = _run(problem, {**ops, "weight": wl})
        assert torch.equal(ref, got), name
    # And the hoisted buffer, which is a fourth spelling of the same values.
    assert torch.equal(ref, _run(problem, ops, weight_rsck=to_rsck(w)))


@requires_gpu
def test_output_is_channels_last_and_matches_torch_grad_shape():
    problem = ConvProblem("shape", 16, 40, (3, 11, 5))
    ops = reference.make_inputs(problem, seed=61)
    gx = _run(problem, ops)
    ref = torch.nn.grad.conv3d_input(
        problem.input_shape, ops["weight"], ops["grad_output"],
        stride=problem.stride, padding=problem.padding,
    )
    assert gx.shape == ref.shape
    assert gx.is_contiguous(memory_format=torch.channels_last_3d)


@requires_gpu
def test_ncdhw_grad_output_is_converted_rather_than_misread():
    """The addressing assumes ``stride_c == 1`` on the upstream gradient.

    An NCDHW ``grad_output`` read with NDHWC strides gives a full-rate kernel
    and a completely wrong gradient.  ScaFFold's own backward can hand us either
    layout depending on what produced the gradient, so this is not hypothetical.
    """
    problem = ConvProblem("layout", 24, 16, (5, 6, 7))
    ops = reference.make_inputs(problem, seed=41, exact=True)
    ndhwc = _run(problem, ops)
    nc = {k: (v.contiguous() if torch.is_tensor(v) else v) for k, v in ops.items()}
    assert nc["grad_output"].stride(1) != 1
    ncdhw = _run(problem, nc)
    assert torch.equal(ndhwc, ncdhw)


@requires_gpu
def test_repeated_calls_are_bitwise_reproducible():
    """MIOpen's backward-data is not; this is the direction where that is fixed.

    ScaFFold's default configuration is nonreproducible today, and backward-data
    is one of the contributors.  Stating the property as a test is what stops a
    later split-K variant from quietly giving it up.
    """
    problem = ConvProblem("determinism", 64, 64, (8, 12, 10))
    ops = reference.make_inputs(problem, seed=71)
    first = _run(problem, ops)
    for _ in range(4):
        assert torch.equal(first, _run(problem, ops))


@requires_gpu
def test_unsupported_calls_raise_rather_than_return_garbage():
    gy = torch.randn((1, 8, 4, 4, 4), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((8, 8, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):
        conv3d_backward_data(gy, w, (1, 8, 4, 4, 4), stride=2, padding=1)
    with pytest.raises(NotImplementedError):
        conv3d_backward_data(gy, w, (1, 8, 4, 4, 4), padding=3)
    with pytest.raises(NotImplementedError):
        conv3d_backward_data(gy, w, (1, 8, 5, 4, 4), padding=1)
