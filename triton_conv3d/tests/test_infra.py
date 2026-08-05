# SPDX-License-Identifier: (Apache-2.0)
"""Tests for the measurement infrastructure itself.

No kernels exist yet.  What exists is a shape model, a cost model, a reference
and a timing harness, and every performance claim we make later is only as good
as those.  So they get tested first, and mostly by cross-checking them against
PyTorch rather than against my own arithmetic: :func:`test_output_shape_matches_torch`
and :func:`test_flops_match_gemm_decomposition` between them caught a real error
in the transposed-convolution FLOP count, where the tap factor was applied twice.

The GPU tests are skipped without a device; the shape and cost model tests are
pure Python and always run.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch
import torch.nn.functional as F

from triton_conv3d import reference
from triton_conv3d.shapes import (
    _CORPUS_PATH,
    BUFFER_OP_MAX_BYTES,
    DIRECTIONS,
    INT32_MAX,
    ConvProblem,
    edge_cases,
    scaffold_corpus,
)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

#: ``include_large=True`` because the shape and cost model tests below are meta
#: tensors and integer arithmetic -- a 4 GiB activation costs nothing here, and
#: until this call existed the two int32-boundary cases were never instantiated
#: by anything at all.  The GPU tests parametrize over ``SMALL`` instead.
ALL = list(scaffold_corpus()) + list(edge_cases(include_large=True))
SMALL = [p for p in edge_cases() if math.prod(p.spatial) * p.cin <= 1 << 16]


def _ids(problems):
    return [p.name or p.label for p in problems]


# ---------------------------------------------------------------------------
# Shape model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem", ALL, ids=_ids(ALL))
def test_output_shape_matches_torch(problem: ConvProblem):
    """The derived output shape is what PyTorch actually produces.

    Cross-checking the whole corpus against the operator it models is what makes
    the extracted shapes trustworthy; an off-by-one in the padding arithmetic
    would otherwise propagate silently into every FLOP count and every roofline.
    Run on meta tensors so a 2 GiB scale-8 activation costs nothing.
    """
    x = torch.empty(problem.input_shape, device="meta")
    w = torch.empty(problem.weight_shape, device="meta")
    op = F.conv_transpose3d if problem.transposed else F.conv3d
    y = op(x, w, None, stride=problem.stride, padding=problem.padding)
    assert tuple(y.shape) == problem.output_shape


@pytest.mark.parametrize("problem", ALL, ids=_ids(ALL))
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_flops_match_gemm_decomposition(problem: ConvProblem, direction):
    """``flops()`` and ``gemm_shape()`` must describe the same contraction.

    They are derived independently -- one from the convolution's definition, one
    from the implicit-GEMM decomposition the kernels will use -- so agreement is
    a real check rather than a tautology.
    """
    m, n, k = problem.gemm_shape(direction)
    assert 2 * m * n * k == problem.flops(direction)


@pytest.mark.parametrize("problem", ALL, ids=_ids(ALL))
def test_bytes_counts_each_tensor_once(problem: ConvProblem):
    """Compulsory traffic is the three tensors, nothing more and nothing less."""
    x = math.prod(problem.input_shape) * problem.elem_bytes
    y = math.prod(problem.output_shape) * problem.elem_bytes
    w = math.prod(problem.weight_shape) * problem.elem_bytes
    for direction in DIRECTIONS:
        assert problem.bytes(direction) == x + y + w


def test_transposed_flops_have_no_phantom_tap_factor():
    """A ``k == s`` transposed convolution does one MAC per output voxel.

    With kernel equal to stride the scatter windows tile the output rather than
    overlapping, so each output voxel receives exactly one contribution.  Scaling
    the input volume by the tap count *and* keeping the tap factor would inflate
    the count eightfold, which is exactly the bug this pins.
    """
    p = ConvProblem(
        "t", 64, 32, (8, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), transposed=True
    )
    assert p.out_spatial == (16, 16, 16)
    macs = math.prod(p.out_spatial) * p.cin * p.cout
    assert p.flops("fwd") == 2 * macs


def test_the_int32_edge_cases_bracket_the_element_boundary():
    """The pair has to sit either side of 2**31 elements, or it pins nothing.

    It did not: ``int32_below`` was ``64 -> 64 @ 512^3``, 8.59e9 elements --
    four times *above* the boundary it is named for, so both cases were above
    it and the transition was unbracketed.  Asserted on the element count
    rather than on the predicate so that this fails if either the shape or the
    predicate moves.
    """
    cases = {p.name: p for p in edge_cases(include_large=True)}
    below, above = cases["int32_below"], cases["int32_above"]
    assert below.max_elements < 2**31 <= above.max_elements
    assert not below.index_exceeds_int32
    assert above.index_exceeds_int32
    # Same channel pair and kernel: the only thing that differs is the volume.
    assert (below.cin, below.cout, below.kernel) == (
        above.cin,
        above.cout,
        above.kernel,
    )
    # And the boundary is where the largest index -- not the count -- crosses.
    assert below.max_elements - 1 <= INT32_MAX < above.max_elements - 1

    small = ConvProblem("small", 32, 32, (8, 8, 8))
    assert not small.index_exceeds_int32
    #: ``bench/baseline.py`` records the predicate under its old name.
    assert small.needs_int64 is small.index_exceeds_int32


def test_the_2gib_cliff_is_a_byte_problem_and_not_an_index_problem():
    """The two int32 predicates are about different quantities, in both senses.

    ``conv 128->64 @ 130x258x258`` is the shape behind the project's two
    largest numbers (769x and 2789x): DistConv's halo pushes its activation
    3.2% past 2 GiB, MIOpen falls off its solver database, and Triton does not.
    That shape holds 1.11e9 elements -- about half of int32's range -- so an
    element-counting predicate says nothing about it, and ``needs_int64``
    used to be read as though it did.  What it exceeds is the *byte* limit on
    the whole storage, which is what decides buffer-op eligibility.
    """
    cliff = next(
        p.halo_variant
        for p in scaffold_corpus()
        if p.halo_variant.label == "conv 128->64 k3x3x3 @ 130x258x258"
    )
    assert not cliff.index_exceeds_int32
    assert cliff.max_elements / 2**31 < 0.55  # half int32's range
    assert not cliff.buffer_ops_eligible
    assert cliff.max_activation_bytes / BUFFER_OP_MAX_BYTES == pytest.approx(
        1.032,
        abs=0.002,  # 3.2% past 2 GiB
    )

    # It is the only corpus shape on either side of that line, in either shape
    # mode -- and *no* corpus shape needs a 64-bit element index.  A test or a
    # dispatch rule parametrized on the index predicate selects nothing.
    over = [
        p.halo_variant.label
        for p in scaffold_corpus()
        if not p.halo_variant.buffer_ops_eligible
    ]
    assert over == ["conv 128->64 k3x3x3 @ 130x258x258"]
    assert not any(
        p.index_exceeds_int32 or p.halo_variant.index_exceeds_int32
        for p in scaffold_corpus()
    )


def test_corpus_covers_the_three_scaffold_configurations():
    corpus = scaffold_corpus()
    assert len(corpus) > 40
    kinds = {(p.kernel, p.stride, p.padding, p.transposed) for p in corpus}
    assert kinds == {
        ((3, 3, 3), (1, 1, 1), (1, 1, 1), False),
        ((2, 2, 2), (2, 2, 2), (0, 0, 0), True),
        ((1, 1, 1), (1, 1, 1), (0, 0, 0), False),
    }
    # Ordered by measured cost, so truncation keeps what matters.
    costs = [sum(m["ms_per_step"] for m in p.measured) for p in corpus]
    assert costs == sorted(costs, reverse=True)


def test_halo_variant_is_the_shape_distconv_actually_issues():
    """The halo'd form is derived, not guessed, and it matches the shape dump.

    Upstream DistConv concatenates a ``k // 2`` halo and zeroes the padding on
    every axis it manages -- including unsplit ones, where the slab is provably
    zeros -- so a convolution routed through it reaches MIOpen two voxels larger
    per axis and unpadded.  ``halo_variant`` reconstructs that from ``halo``
    alone; this pins the reconstruction against ``halo_in_shape``, which the
    shape dump recorded independently.  If they ever disagree, every profiled
    number in ``measured`` is attached to the wrong problem.

    This is the *incumbent's* form.  What ScaFFold's own Triton rung issues is
    :meth:`ConvProblem.production_variant`, pinned separately below against a
    census of real calls.
    """
    raw = json.loads(_CORPUS_PATH.read_text())["problems"]
    corpus = scaffold_corpus()
    assert len(raw) == len(corpus)
    for entry, problem in zip(raw, corpus):
        halo = problem.halo_variant
        assert list(halo.input_shape) == entry["halo_in_shape"]
        # The halo changes the input, never the output -- that is what makes it
        # a halo and not a padding change.
        assert halo.output_shape == problem.output_shape
        # Padding is dropped exactly on the axes that gained a halo, and left
        # alone elsewhere -- that swap is the whole transformation.
        assert halo.padding == tuple(
            0 if h else p for h, p in zip(problem.halo, problem.padding)
        )
        assert halo.flops("fwd") == problem.flops("fwd")


def test_halo_variant_is_a_distinct_miopen_problem():
    """The two forms must not collide in any table keyed by label.

    MIOpen keys its find database on the full descriptor including padding, so
    ``64-128-128-128-...-1x1x1`` and ``64-130-130-130-...-0x0x0`` tune
    separately and can land on different kernels.  A baseline that labelled
    them the same would let a cell measured on one be compared against a
    profile of the other -- silently, and in whichever direction flatters the
    kernel under test.
    """
    hot = [p for p in scaffold_corpus() if any(p.halo)]
    assert hot, "corpus has no halo'd problems; the dump lost halo_dhw"
    for p in hot:
        assert p.halo_variant.label != p.label
    # And a problem with no halo is its own variant, so "both" shape modes do
    # not measure the transposed upsamples and 1x1x1 convs twice.
    for p in scaffold_corpus():
        if not any(p.halo):
            assert p.halo_variant is p


def test_the_production_variant_is_what_a_real_step_issues():
    """The one that was wrong, pinned against a measurement.

    Every published ``conv`` cell in this project measures
    :meth:`ConvProblem.halo_variant`, on the premise that "DistConv halos them
    all".  ScaFFold does not route its convolutions through DistConv: the
    adapter in ``ScaFFold/unet/conv3d.py`` exchanges a halo only on axes with
    more than one shard, so H and W keep ``padding = 1`` at every configuration
    and all three axes do at one GPU.  A projection built on the halo'd cells
    over-credited a tuning commit by 3x before anyone checked.

    Checked here against ``census_corpus()`` -- a recording of the shapes and
    paddings real ``FastConv3d`` calls handed the kernels at all four
    configurations -- rather than against the same arithmetic twice.  The
    segmentation head is excluded on ``cout``: its output channel count is
    ``n_categories + 1``, a dataset knob, and the corpus and the census were
    taken with different values of it (6 and 3).
    """
    from triton_conv3d.shapes import census_corpus, production_corpus

    def key(p):
        return (
            p.transposed,
            p.cin,
            p.cout,
            tuple(p.kernel),
            tuple(p.spatial),
            tuple(p.padding),
            tuple(p.stride),
            p.n,
        )

    census = {key(p) for p in census_corpus()}
    assert len(census) > 60, "the census is missing; nothing is being checked"
    missing = [
        p for p in production_corpus() if key(p) not in census and p.kernel != (1, 1, 1)
    ]
    assert not missing, (
        "the corpus's production form does not match what a real step issued: "
        + ", ".join(p.qualified_label for p in missing)
    )
    # And the census is in the form it claims: every k>1 convolution padded.
    unpadded = [
        p for p in census_corpus() if p.kernel == (3, 3, 3) and not any(p.padding)
    ]
    assert not unpadded, (
        "a k=3 production convolution arrived unpadded, which would mean the "
        "adapter's halo plan changed: " + ", ".join(p.qualified_label for p in unpadded)
    )


def test_the_three_forms_are_told_apart_by_the_qualified_label():
    """A cell must never be quotable as a form it is not.

    ``label`` carries the extent but not the padding, and the two sharded forms
    of one site differ in *both* -- while the unsharded production form differs
    from the DistConv one in the padding alone at some extents.  Any table that
    mixes forms therefore has to key on ``qualified_label``.
    """
    sharded = [p for p in scaffold_corpus() if any(p.shard_halo)]
    assert sharded, "corpus has no sharded problems; shard_halo_dhw was lost"
    for p in sharded:
        forms = {
            p.qualified_label,
            p.production_variant.qualified_label,
            p.halo_variant.qualified_label,
        }
        assert len(forms) == 3, f"{p.label}: forms collide -> {forms}"
        # The adapter halos D and leaves H and W padded; DistConv does neither.
        assert p.production_variant.padding == (0, *p.padding[1:])
        assert p.halo_variant.padding == (0, 0, 0)
        assert p.production_variant.spatial == (p.spatial[0] + 2, *p.spatial[1:])
    # Unsharded, the adapter form *is* the logical one and says so by identity.
    for p in scaffold_corpus():
        if not any(p.shard_halo):
            assert p.production_variant is p


def test_the_production_corpus_is_padded_where_the_halo_corpus_is_not():
    """The headline of the whole distinction, as a number.

    If this ever reads "0 padded" again, either the adapter has started haloing
    every axis or ``shard_halo`` has been confused with ``halo`` -- and the
    consequence is that every backward-weight kernel silently stops compiling
    the ``PADDED`` body it compiles at every ``k = 3`` site today, so every
    number in this project's adapter-form tables would describe a kernel
    production no longer launches.  (Until 2026-08-05 the consequence was
    larger still: ``bwd_weight_config`` declined a tuned ``TAP_BLOCK > 1`` row
    on a padded problem, so this count decided which *tile* eight sites ran.)
    """
    from triton_conv3d.shapes import halo_corpus, production_corpus

    padded_prod = [p for p in production_corpus() if any(p.padding)]
    padded_halo = [p for p in halo_corpus() if any(p.padding)]
    assert len(padded_prod) == 42, len(padded_prod)
    assert padded_halo == []
    # Every one of them is a k=3 convolution; the k=1 head and the k=2
    # upsamplers are genuinely unpadded in every form.
    assert {p.kernel for p in padded_prod} == {(3, 3, 3)}


def test_stored_efficiency_agrees_with_the_cost_model():
    """The corpus's ``pct_roofline`` must be what ``efficiency(ms_per_call)`` says.

    They are computed by different code -- one by ``make_corpus.py`` out of the
    profile's own FLOP and byte counts, one here out of the shape -- so agreement
    is a real cross-check, and it caught a real error.  ``make_corpus.py`` used
    to divide the profile's *per-step* FLOP count by the *per-call* time, which
    multiplies the efficiency by the number of call sites.  Every affected
    problem is a symmetric ``C -> C`` convolution occurring at two sites, so the
    artifact read as "MIOpen is excellent on symmetric convolutions and poor on
    asymmetric ones" and produced the three forward points that appeared to
    exceed 100% of roofline.  With it fixed, MIOpen's forward spans 21-68%
    everywhere and the three impossible points are gone.
    """
    for problem in scaffold_corpus():
        for m in problem.measured:
            got = 100 * problem.efficiency(m["ms_per_call"], m["direction"])
            assert got == pytest.approx(m["pct_roofline"], abs=0.06, rel=0.01), (
                f"{problem.label} [{m['direction']}, config {m['config']}]: "
                f"stored {m['pct_roofline']}%, cost model {got:.3f}%"
            )
    # And no forward cell exceeds the roof, which is what the artifact implied.
    fwd = [
        m["pct_roofline"]
        for p in scaffold_corpus()
        for m in p.measured
        if m["direction"] == "fwd"
    ]
    assert fwd and max(fwd) < 100


def test_roofline_switches_at_the_crossover():
    """Below ~182 FLOP/byte the memory roof binds; above it, compute does."""
    memory_bound = ConvProblem("thin", 3, 8, (16, 16, 16))
    compute_bound = ConvProblem("fat", 512, 512, (16, 16, 16))
    assert memory_bound.arithmetic_intensity() < 182
    assert memory_bound.roofline_flops() < 600e12
    assert compute_bound.arithmetic_intensity() > 182
    assert compute_bound.roofline_flops() == 600e12


# ---------------------------------------------------------------------------
# Reference and tolerance policy
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("problem", SMALL, ids=_ids(SMALL))
def test_reference_agrees_with_miopen_within_tolerance(problem: ConvProblem):
    """MIOpen itself must pass the bar we intend to hold our kernel to.

    If the incumbent failed this, the tolerance would be wrong rather than
    MIOpen -- so this is a test of the policy, not of MIOpen.
    """
    ops = reference.make_inputs(problem, seed=7)
    for direction in DIRECTIONS:
        expected = reference.reference(problem, ops, direction)
        actual = reference.incumbent(problem, ops, direction)
        # MIOpen's backward-weight is the one direction that is not a single
        # rounding: it reduces with atomics, so two identical calls differ
        # bitwise and the result carries several roundings rather than the one
        # ``error_bound`` charges by default.  Measured at ``conv 32->32 k3x3x3
        # @ 8x8x8``, its error wanders over 0.61-1.05 ulps of the peak from call
        # to call while the forward sits at a fixed 0.284 -- and a one-ulp bound
        # therefore does not merely fail it, it fails it *intermittently*, which
        # is the worse outcome.  ``convT 64->32 k2x2x2 @ 8x8x8`` is the other
        # cell that reaches past one ulp.  The nondeterminism and the size of
        # the excess are both pinned by
        # :func:`test_the_incumbents_extra_roundings_are_the_atomic_ones`, so
        # this is a measured allowance rather than a tolerance nudged until the
        # test passed.  Only this direction and only the incumbent get it: our
        # own backward-weight reduces its split-K partials in fp32 and stores
        # once, so it is held to ``roundings=1`` like everything else.
        reference.assert_close(
            actual,
            expected,
            problem,
            direction,
            roundings=2 if direction == "bwd-weight" else 1,
        )


@requires_gpu
@pytest.mark.parametrize("problem", SMALL, ids=_ids(SMALL))
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_exact_inputs_give_a_bitwise_reference(problem: ConvProblem, direction):
    """With ``{-1,0,1}`` operands the contraction is exact, so equality holds.

    This is the standard that catches indexing and masking bugs: a kernel that
    reads a neighbouring voxel still produces a plausible number, and only an
    exact comparison rejects it.  MIOpen passing it is what establishes that the
    standard is attainable rather than aspirational.
    """
    ops = reference.make_inputs(problem, seed=3, exact=True)
    expected = reference.reference(problem, ops, direction)
    if not reference.is_exactly_representable(expected, reference.torch_dtype(problem)):
        pytest.skip("realized magnitudes exceed the mantissa in this dtype")
    actual = reference.incumbent(problem, ops, direction)
    report = reference.compare(actual, expected)
    assert report.bitwise, f"{problem.label} [{direction}]: {report}"


def test_error_bound_grows_with_reduction_length_and_shrinks_with_precision():
    expected = torch.randn(4096, dtype=torch.float64)
    short = ConvProblem("short", 8, 8, (8, 8, 8))
    long_ = ConvProblem("long", 1024, 1024, (8, 8, 8))
    assert reference.error_bound(long_, expected) > reference.error_bound(
        short, expected
    )
    fp32 = ConvProblem("fp32", 64, 64, (8, 8, 8), dtype="fp32")
    bf16 = ConvProblem("bf16", 64, 64, (8, 8, 8), dtype="bf16")
    assert reference.error_bound(fp32, expected) < reference.error_bound(bf16, expected)


def test_error_bound_tracks_peak_not_just_rms():
    """A tensor with a big outlier gets a proportionally bigger absolute bound.

    This is the property whose absence made the bound too tight: the final
    rounding to bf16 costs an ulp of the *largest* element, so a spiky tensor
    legitimately admits more absolute error than a flat one of the same RMS.
    """
    problem = ConvProblem("p", 64, 64, (8, 8, 8))
    flat = torch.ones(4096, dtype=torch.float64)
    spiky = flat.clone()
    spiky[0] = 100.0
    assert reference.error_bound(problem, spiky) > 10 * reference.error_bound(
        problem, flat
    )


def test_the_store_term_is_charged_as_one_rounding_not_four():
    """The safety factor belongs on the walk, not on the deterministic store.

    ``error_bound`` used to be ``8 * (accum + store)``, four ulps of the peak
    for a rounding that is bounded by half an ulp of the *element* outright.
    There is no random walk in a single store to take a factor against, and the
    consequence was not academic: the static arm then won
    :func:`reference.assert_close`'s ``max()`` in 46 of 48 measured cells, so
    ``test_no_worse_than_miopen`` in all three kernel files was not holding the
    kernel to the standard its name and docstring claim.

    Pinned arithmetically rather than by measurement so that it fails on the
    formula rather than on a GPU: at ``K`` short enough that the accumulation
    term is negligible, the bound must be one ulp of the peak per rounding.
    """
    problem = ConvProblem("p", 8, 8, (4, 4, 4))  # K = 216
    peak = torch.zeros(4096, dtype=torch.float64)
    peak[0] = 64.0
    ulp = 2.0 * reference.unit_roundoff(torch.bfloat16) * 64.0
    assert reference.error_bound(problem, peak) == pytest.approx(ulp, rel=1e-3)
    assert reference.error_bound(problem, peak, roundings=2) == pytest.approx(
        2 * ulp, rel=1e-3
    )


@requires_gpu
def test_the_incumbents_extra_roundings_are_the_atomic_ones():
    """Why the incumbent gets ``roundings=2`` in exactly one direction.

    The store term models one deterministic rounding into the working dtype.
    That is what the forward and backward-data do -- both are bitwise
    reproducible here, and their error lands under one ulp of the peak.
    MIOpen's backward-weight is not: it reduces with atomics, two identical
    calls differ, and the extra roundings can carry it past one ulp.  Without
    this the allowance in
    :func:`test_reference_agrees_with_miopen_within_tolerance` looks like a
    tolerance that was widened until the test passed.

    ``conv 32->32 k3x3x3 @ 8x8x8`` because it is the cell that measures the
    excess most clearly; the nondeterminism is a property of the direction, not
    of the shape.  What is asserted is the *call-to-call spread*, not the error
    on any one call, and that distinction is the finding: the error itself
    wanders (0.61 to 1.05 ulps of the peak over eight calls) precisely because
    the reduction order does, so an assertion on a single call would be as
    intermittent as the bound it is defending.  A single rounding has a spread
    of exactly zero, which is what the other two directions measure.
    """
    problem = ConvProblem("atomic", 32, 32, (8, 8, 8))
    ops = reference.make_inputs(problem, seed=7)
    ulp = 2.0 * reference.unit_roundoff(reference.torch_dtype(problem))

    def probe(direction, repeats=6):
        expected = reference.reference(problem, ops, direction)
        scale = ulp * expected.abs().max().item()
        runs = [reference.incumbent(problem, ops, direction) for _ in range(repeats)]
        errs = [reference.compare(r, expected).max_abs / scale for r in runs]
        spread = max((a - b).abs().max().item() for a in runs for b in runs) / scale
        return max(errs), spread

    deterministic = 0.0
    for direction in ("fwd", "bwd-data"):
        err, spread = probe(direction)
        assert spread == 0.0, f"{direction}: MIOpen disagreed with itself by {spread}"
        assert err < 1.0, f"{direction}: {err:.3f} ulps of the peak"
        deterministic = max(deterministic, err)

    err, spread = probe("bwd-weight")
    assert spread > 0.25, (
        f"MIOpen's backward-weight agreed with itself to {spread:.3f} ulps of "
        "the peak; if it has stopped reducing with atomics then the roundings=2 "
        "allowance it is given has lost its reason and should be dropped"
    )
    assert err > deterministic, (
        f"backward-weight ({err:.3f} ulps) is no worse than the directions that "
        f"round once ({deterministic:.3f}); the allowance is unmotivated"
    )
    # And the allowance is an envelope, not a blank cheque: two roundings must
    # still be enough.  If this trips, the right response is to find out how
    # many partials MIOpen is accumulating, not to raise the number.
    assert err < 2.0, f"backward-weight needs more than two roundings: {err:.3f}"


@requires_gpu
def test_the_incumbent_clause_binds_more_often_than_the_static_bound():
    """The anti-vacuity guard on ``assert_close``'s ``max()``.

    A ``max()`` is only worth writing if both arms can win.  Under the old
    four-ulp store term the static arm won essentially always and the "no worse
    than MIOpen by more than ``margin``" standard was dead code -- documented,
    named in three test functions, and never applied.  So pin the property that
    made it live: over these cells the incumbent arm must be the operative one
    more often than not.

    A floor rather than a per-cell assertion because which arm wins is a real
    measurement and does move: it is the incumbent in 12 of these 13 cells, and
    the one that goes the other way is a shape where MIOpen happens to be
    unusually accurate -- exactly the case the ``max()`` exists to stop from
    tightening the test beyond what the numerics justify.  Not parametrized,
    because a per-case fixture cannot state a floor over the set and this file's
    whole reason for existing is that a test which reports a pass without
    testing anything is worse than no test.
    """
    binds = []
    for problem in SMALL:
        ops = reference.make_inputs(problem, seed=7)
        expected = reference.reference(problem, ops, "fwd")
        err = reference.compare(reference.incumbent(problem, ops, "fwd"), expected)
        binds.append(
            4.0 * err.max_abs > reference.error_bound(problem, expected, "fwd")
        )
    assert sum(binds) > len(binds) // 2, (
        f"the incumbent clause bound only {sum(binds)}/{len(binds)} cells; the "
        "static bound has drifted back to swallowing it"
    )


@requires_gpu
def test_assert_close_rejects_a_wrong_answer():
    """The policy has to fail when it should; a tolerance nobody can trip is not one.

    A one-voxel shift is the realistic failure mode for a gather kernel, and it
    is the one a loose elementwise tolerance would wave through.
    """
    problem = ConvProblem("shift", 16, 16, (8, 8, 8))
    ops = reference.make_inputs(problem, seed=11)
    expected = reference.reference(problem, ops, "fwd")
    shifted = reference.incumbent(problem, ops, "fwd").roll(1, dims=-1)
    with pytest.raises(AssertionError):
        reference.assert_close(shifted, expected, problem, "fwd")


@requires_gpu
def test_channels_last_is_preserved_by_make_inputs():
    problem = ConvProblem("cl", 32, 32, (8, 8, 8))
    ops = reference.make_inputs(problem)
    assert ops["input"].is_contiguous(memory_format=torch.channels_last_3d)
    assert ops["grad_output"].is_contiguous(memory_format=torch.channels_last_3d)


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


@requires_gpu
def test_interleaved_rotates_variants_and_reports_spread():
    """Every variant occupies every slot, so no one of them owns the fast one."""
    from triton_conv3d.bench.harness import interleaved

    a = torch.randn(512, 512, device="cuda")
    seen: dict[str, list[int]] = {"x": [], "y": [], "z": []}
    order: list[str] = []

    def make(name):
        def fn():
            order.append(name)
            return a @ a

        return fn

    result = interleaved({k: make(k) for k in seen}, warmup=1, iters=1, rounds=3)
    assert set(result) == set(seen)
    assert all(len(m.rounds) == 3 for m in result.values())
    # Rotation: the first variant of each round differs from round to round.
    starts = {order[i] for i in range(0, len(order), 1) if i % 3 == 0}
    assert len(starts) > 1, "rounds did not rotate"
    # The old assertion here was ``m.spread >= 0``, which is true by
    # construction of ``(max - min) / median`` and could not fail.  What is
    # worth pinning is that pinning ``warmup``/``iters``/``rounds`` still runs
    # exactly the calls it says: 1 warmup and 3 rounds of 1 iteration each, per
    # variant, with no calibration probe smuggled in.
    assert len(order) == 3 * (1 + 3 * 1)
    assert all(
        m.iters == 1 and m.group == 1 and m.stop == "fixed" for m in result.values()
    )


def test_the_round_order_is_position_and_adjacency_balanced():
    """Rotating by one position per round de-biases slots but not neighbours.

    Under the old rule -- ``names[r % n:] + names[:r % n]`` -- variant B ran
    immediately after variant A in *every* round, so whatever A left in the
    caches was a constant charged to B and averaged out of nothing.  Measured on
    the adversarial case (a 1 GiB cache-polluting arm plus two arms doing
    byte-identical work, 40 replications): cyclic rotation reported the two
    identical arms **2.8% apart**, this rule 0.2% apart, a random order 0.7%.
    2.8% is larger than several of the per-cell differences this project
    publishes, so the design property is worth asserting rather than trusting.

    Pure Python and exhaustive, so it fails on the *rule* rather than on a
    measurement: over ``2 * n`` rounds every variant must occupy every position
    equally often **and** every ordered adjacent pair must occur equally often.
    Reverting :func:`_order` to the cyclic rotation fails the second clause at
    every ``n >= 3`` (it makes the count of ``(A, B)`` equal to the number of
    rounds and the count of ``(B, A)`` zero).
    """
    from triton_conv3d.bench.harness import _order

    for n in range(1, 7):
        names = [chr(ord("A") + i) for i in range(n)]
        rounds = 2 * n
        positions = {x: [0] * n for x in names}
        adjacency: dict[tuple[str, str], int] = {}
        for r in range(rounds):
            got = _order(names, r)
            assert sorted(got) == sorted(names), f"{n}: {got} is not a permutation"
            for slot, x in enumerate(got):
                positions[x][slot] += 1
            for pair in zip(got, got[1:]):
                adjacency[pair] = adjacency.get(pair, 0) + 1
        for x in names:
            assert len(set(positions[x])) == 1, (
                f"n={n}: {x} occupied positions unevenly: {positions[x]}"
            )
        if n >= 2:
            assert len(set(adjacency.values())) == 1, (
                f"n={n}: adjacency is not balanced: {adjacency}"
            )
            assert len(adjacency) == n * (n - 1), (
                f"n={n}: only {len(adjacency)} of {n * (n - 1)} ordered pairs occur"
            )


def test_spread_is_a_range_statistic_and_the_interval_is_not():
    """Why ``spread`` cannot support a claim about how much the machine moved.

    ``(max - min) / median`` is a *range*, and the expected range of ``n``
    samples grows like ``d2(n)`` even on a perfectly stationary device.
    Measured on this node with one kernel held constant for 14 minutes (47,686
    blocks) the median of this statistic runs 0.23% at 2 rounds, 0.63% at 6,
    0.98% at 20 and 2.70% at 100 -- all of it arithmetic, none of it the
    machine.  Since ``rounds`` is now chosen per cell, two cells' spreads are
    not comparable to each other at all, and the replacement has to be an
    interval.

    Pinned on a fixed draw so it tests the formulae, not the GPU.
    """
    import random

    from triton_conv3d.bench.harness import Measurement

    rng = random.Random(20260803)

    def draw(n):
        return tuple(1.0 + 0.01 * rng.gauss(0, 1) for _ in range(n))

    short = Measurement("short", draw(4))
    long_ = Measurement("long", draw(1000))
    # Same underlying dispersion, by construction.
    assert abs(long_.cov - 0.01) < 0.002
    # The range grows with n ...
    assert long_.spread > 2.5 * short.spread
    # ... while the interval, which is the thing to quote, shrinks.
    assert long_.rel_half_width < 0.2 * short.rel_half_width
    assert short.rel_half_width > 0.005


@requires_gpu
def test_a_paired_ratio_of_two_identical_arms_covers_one():
    """The anti-vacuity guard on the interval: it must be right *and* narrow.

    Two arms that are the same callable have a true ratio of exactly 1, so an
    interval that misses 1 is too narrow and one that spans a factor of two is
    useless.  Both failures are live: an interval computed on the *mean* of
    per-iteration times rather than on the round medians is too narrow, and one
    taken over two rounds is too wide.

    This also puts a number on what a published ratio has to beat.  At a
    0.08 ms kernel two identical arms measured the old way -- 6 rounds of 10 --
    came out **0.941x to 1.058x** over 40 replications (sd 2.5%), so a "1.02x"
    at that size was never a measurement.  Nothing here asserts that; it is why
    the interval exists.
    """
    from triton_conv3d.bench.harness import interleaved, ratio

    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    fn = lambda: a @ a  # noqa: E731
    meas = interleaved({"x": fn, "y": fn}, target_rel=0.02, budget_s=15.0)
    r = ratio(meas["y"], meas["x"])
    assert r.lo <= 1.0 <= r.hi, f"interval missed the truth: {r}"
    assert not r.significant, f"identical arms declared different: {r}"
    assert r.rel_half_width < 0.06, f"interval uselessly wide: {r}"
    assert abs(r.point - 1.0) < 0.05, f"identical arms differ by {r.point:.4f}"


@requires_gpu
def test_the_block_is_sized_from_the_measured_duration():
    """``iters`` is chosen online, and it has to move with the kernel.

    The corpus spans five orders of magnitude -- 0.06 ms at the transposed sites
    against 45,241 ms for one call at the 2 GiB cliff -- and a fixed
    ``iters=10, rounds=6`` is 60 calls either way: microseconds for one cell and
    45 minutes for the other.

    ``torch.cuda._sleep`` rather than a real kernel: it consumes a stated number
    of device cycles with no memory traffic and no tuning database, so the test
    asserts the *sizing rule* and cannot fail because MIOpen picked a different
    solver today.
    """
    from triton_conv3d.bench.harness import time_callable

    fast = time_callable(lambda: torch.cuda._sleep(200_000), budget_s=5.0)
    slow = time_callable(lambda: torch.cuda._sleep(60_000_000), budget_s=5.0)
    assert slow.median > 20 * fast.median, "the two probes are not far apart"
    assert slow.iters <= 2, f"a {slow.median:.1f} ms call got iters={slow.iters}"
    assert fast.iters >= 10 * slow.iters, (
        f"iters did not track duration: {fast.iters} at {fast.median:.4f} ms "
        f"vs {slow.iters} at {slow.median:.2f} ms"
    )
    # And the block lands near its target rather than anywhere at all.
    assert 0.1 <= fast.iters * fast.median / 15.0 <= 10.0


@requires_gpu
def test_a_slow_kernel_stops_on_the_budget_and_says_so():
    """The ceiling, and the flag that makes a loose measurement visible.

    With an unreachable precision target the only way out is the wall clock, so
    this pins both that the budget is honoured and that ``stop`` reports it.
    Without the budget check the same call runs to ``max_rounds`` -- 64 rounds
    of a ~0.5 s kernel, half a minute -- which is what the assertion on elapsed
    time detects.
    """
    import time

    from triton_conv3d.bench.harness import time_callable

    t0 = time.perf_counter()
    m = time_callable(
        lambda: torch.cuda._sleep(1_000_000_000), budget_s=1.0, target_rel=1e-9
    )
    elapsed = time.perf_counter() - t0
    assert m.stop == "budget", f"stopped for the wrong reason: {m.stop}"
    assert not m.converged
    assert len(m.rounds) <= 8, f"{len(m.rounds)} rounds against a 1 s budget"
    assert elapsed < 15.0, f"budget not honoured: {elapsed:.1f} s"
    assert m.rel_half_width > 0, "a budget-stopped cell must still report a width"


@requires_gpu
def test_the_instrument_tax_is_measured_and_grouped_away():
    """The sub-0.15 ms regime, with its own negative control.

    An ``hipEventRecord`` costs ~9.5 us of host time, and at a 0.017 ms kernel a
    block with an event between every iteration reports **1.5x** what the same
    kernel's wall-clock throughput does.  That is not noise, it is not the node,
    and it does not cancel in a ratio because it is per-arm: measured on
    ``convT 1024->512 @ 8^3``, the Triton forward pays 10.1 us and the MIOpen
    weight-gradient control 11.9 us on times of 0.057 and 0.067 ms.

    The grouped block is checked against an event-free wall-clock measurement of
    the same callable, and against the *ungrouped* harness in the same run.  The
    second is the control: if grouping ever stops working, the two agree and
    this fails, rather than both drifting together unnoticed.
    """
    import time

    from triton_conv3d.bench.harness import time_callable

    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    fn = lambda: a @ a  # noqa: E731
    for _ in range(50):
        fn()
    torch.cuda.synchronize()

    def wall(n=4000):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / n

    # The *minimum* of several wall-clock runs, not the median.  This reference
    # has no events in it at all, which is the point, but it is therefore
    # host-throughput-bound: it can only be inflated by contention, never
    # deflated.  Taking the median made this test fail once inside the full
    # suite -- reference 0.0214 ms against 0.0167 ms in isolation, while the
    # harness's own number moved by 9% -- which is the harness's stall rejection
    # working and the reference's absence of it showing.
    reference = min(wall() for _ in range(5))
    grouped = time_callable(fn, budget_s=10.0)
    # ``tax_budget`` above 1.0 can never be exceeded, which disables grouping
    # and reproduces the historical instrument exactly.
    ungrouped = time_callable(fn, budget_s=10.0, tax_budget=10.0)

    assert grouped.group > 1, "a 0.02 ms kernel was left at one event per call"
    assert ungrouped.group == 1
    assert 0.80 <= grouped.median / reference <= 1.25, (
        f"grouped block disagrees with the event-free reference: "
        f"{grouped.median:.5f} vs {reference:.5f} ms"
    )
    assert ungrouped.median > 1.2 * grouped.median, (
        f"the instrument tax has vanished on its own ({ungrouped.median:.5f} "
        f"vs {grouped.median:.5f} ms); if that is real this test's premise is "
        "gone and the grouping can be removed, but check the ruler first"
    )


@requires_gpu
def test_flush_caches_reuses_one_buffer_and_reaches_only_the_first_sample():
    """Two defects in one small function, both of which had teeth.

    ``torch.device("cuda")`` carries no index and a tensor made on it does, so
    the guard ``_flush_buffer.device != torch.device(device)`` was *always*
    true: every flush allocated a fresh 512 MiB tensor while the old one was
    still live, on the critical path of every timed round.

    And a flush before a block reaches only the block's **first** call, while
    the block reports the median over ``iters`` of them -- so at ``iters=10``
    the one cold sample is precisely the one the median throws away.  Measured
    ``median_moved_by_flush`` is 0.99-1.01 at every real workload while the
    first iteration moves 1.02-1.54x.  The adaptive path therefore measures
    ``iters=1`` when ``flush`` is on, and :attr:`Measurement.cold` records the
    first sample either way.
    """
    from triton_conv3d.bench import harness as H

    H.flush_caches()
    first = H._flush_buffer.data_ptr()
    for _ in range(5):
        H.flush_caches()
    assert H._flush_buffer.data_ptr() == first, (
        "flush_caches reallocated its buffer; the device comparison is wrong again"
    )
    assert H._flush_buffer.numel() == H._FLUSH_BYTES

    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    cold = H.time_callable(lambda: a @ a, flush=True, rounds=4)
    assert cold.iters == 1, (
        f"flush=True measured {cold.iters} calls per block, so {cold.iters - 1} "
        "of them are hot and the median reports a hot number"
    )
    assert cold.cold == cold.median  # with one sample per block they coincide


@requires_gpu
def test_pinning_iters_and_rounds_reproduces_the_fixed_protocol():
    """Backward compatibility, asserted on the call count rather than assumed.

    Every existing driver and every stored result JSON was produced by pinned
    ``warmup``/``iters``/``rounds``.  Those callers must keep issuing exactly
    the calls they always did -- no calibration probe, no warmup of its own, no
    grouping -- or a re-capture is not comparable with what is on disk.
    """
    from triton_conv3d.bench.harness import time_callable

    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return a @ a

    m = time_callable(fn, warmup=3, iters=10, rounds=6)
    assert calls["n"] == 3 + 6 * 10, f"issued {calls['n']} calls, expected 63"
    assert len(m.rounds) == 6 and m.iters == 10 and m.group == 1
    assert m.stop == "fixed" and m.converged


# ---------------------------------------------------------------------------
# The baseline is the control for every later claim, so it gets its own guards
# ---------------------------------------------------------------------------


def test_importing_the_baseline_turns_on_the_miopen_find():
    """A baseline taken with ``cudnn.benchmark`` off is not a baseline.

    On ROCm that flag decides whether PyTorch asks MIOpen to *search* for a
    tuning config or to answer from its AI heuristic.  The heuristic's answer
    for the corpus' hottest problems is 5-12x slower than the searched one --
    same solver, same device op, just 16x16 MFMA tiles with 2-element global
    loads instead of 32x32 with 8-element loads.  The first version of this
    harness left the flag at its default and overstated MIOpen by up to 12x,
    which would have become a fabricated speedup for every kernel measured
    against it.  ScaFFold itself sets it (``worker.py:171``) and so does the
    profiler the reference numbers come from (``prof_bench.py:125``).
    """
    from triton_conv3d.bench import baseline

    assert baseline.REQUIRE_CUDNN_BENCHMARK is True
    assert torch.backends.cudnn.benchmark is True, (
        "importing the baseline module must leave the process in the "
        "configuration its recorded numbers were taken in"
    )


def test_measure_one_refuses_to_report_a_heuristic_time():
    """The guard has to be at the measurement, not only at import.

    Anything may flip ``cudnn.benchmark`` between import and use -- a
    determinism experiment, another test, a notebook cell.  Refusing loudly is
    the only outcome that cannot end up in a JSON file that looks like a
    control.
    """
    from triton_conv3d.bench.baseline import measure_one

    problem = ConvProblem("guard", 8, 8, (4, 4, 4))
    previous = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = False
        with pytest.raises(RuntimeError, match="cudnn.benchmark is off"):
            measure_one(problem, "fwd")
    finally:
        torch.backends.cudnn.benchmark = previous


#: Corpus cells the harness is anchored to.  Chosen because their isolated and
#: profiled shapes genuinely match: the halo'd input is 281 MiB, well under the
#: 2 GiB threshold above which MIOpen abandons its tuned solvers for the naive
#: non-packed ones and the isolated and profiled numbers legitimately diverge.
#: The profiled time is read from the corpus rather than copied here so there
#: is one source of truth for it.
ANCHOR_CELLS = ((6, "fwd"), (6, "bwd-data"))


@requires_gpu
@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("index, direction", ANCHOR_CELLS)
def test_baseline_reproduces_the_profiled_scaffold_conv(index, direction):
    """An end-to-end anchor: the harness lands on the profiled number.

    The two tests above check the settings; this one checks the thing the
    settings are for, and would still fail if the harness went wrong in a way
    nobody anticipated -- a memory-format regression, a dtype regression, a
    future PyTorch that stops honouring ``benchmark`` on ROCm.

    The band is asymmetric and generous.  Isolated *should* come out a little
    faster than profiled -- no contention for bandwidth, no other kernel in
    flight, no allocator pressure -- but never much faster, and a run that is
    slower than the profile has lost the find.

    Judged on the *best* round, not the median: this is a shared node and a
    neighbouring job can inflate all five rounds at once (observed once while
    writing this, at 2.1x).  That is a fact about the node, not about the
    harness, and a test that fails on it teaches people to ignore it.  The
    failure this test is for -- a lost find, the wrong shape, an inert memory
    format -- is 5-12x and survives taking the minimum easily.
    """
    from triton_conv3d.bench.baseline import measure_one

    logical = scaffold_corpus()[index]
    problem = logical.halo_variant
    profiled_ms = logical.measured_for(direction)[-1]["ms_per_call"]
    record = measure_one(problem, direction)
    assert "error" not in record, record.get("error")
    ratio = record["best_ms"] / profiled_ms
    assert 0.5 <= ratio <= 1.4, (
        f"{problem.label} {direction}: {record['best_ms']:.3f} ms isolated "
        f"(best of {record['rounds']}) vs {profiled_ms:.3f} ms profiled "
        f"({ratio:.2f}x). Off by this much means MIOpen is not solving the "
        f"problem ScaFFold solves -- check cudnn.benchmark, "
        f"PYTORCH_MIOPEN_SUGGEST_NHWC and the halo shape."
    )


@requires_gpu
def test_sporadic_host_stall_is_rejected_from_the_median_and_flagged():
    """An occasional slow launch must not be charged to the kernel.

    This is the harness bug that produced last session's 250-2363% spreads,
    which were then misdiagnosed twice -- first as host jitter, then as a rogue
    tenant on the GPU -- before turning out to be a duplicate driver process of
    our own.  The old ``_time_block`` bracketed a whole block of iterations with
    two events, so any launch gap inside it was silently added to kernel time.

    One stalled launch in ten is the realistic shape of the problem: contention
    is intermittent, so a mean absorbs it and a median rejects it.  The stall
    ratio exists so that rejecting it is not the same as hiding it.
    """
    import time

    from triton_conv3d.bench.harness import interleaved

    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)

    # The control needs a quiet host and this test cannot assume one: the node
    # is shared, and a neighbouring job stalls our launches exactly as well as
    # the duplicate driver of our own did.  When that happens the diagnostic is
    # firing *correctly* and it is the premise -- "this run is clean" -- that is
    # false.  Observed at 12.87 against a threshold of 2.0 while a sibling job
    # was running, passing three times in a row on the same tree once the node
    # went idle.  So take the quietest of several attempts, and if none of them
    # is quiet, say the host was loaded rather than assert something this run
    # cannot decide -- failing here would re-enact the original misdiagnosis in
    # test form, blaming the measurement for observing real contention.
    clean = min(
        (
            interleaved({"g": lambda: a @ a}, warmup=3, iters=20, rounds=3)["g"]
            for _ in range(5)
        ),
        key=lambda m: m.stall_ratio,
    )
    if clean.stall_ratio >= 2.0:
        pytest.skip(
            f"host too loaded for a quiet control (best stall ratio "
            f"{clean.stall_ratio:.2f} over 5 attempts); the diagnostic is "
            "reporting real contention, so this test cannot separate a false "
            "positive from a true one"
        )

    calls = {"n": 0}

    def sporadic():
        calls["n"] += 1
        if calls["n"] % 10 == 0:
            time.sleep(0.02)
        return a @ a

    stalled = interleaved({"g": sporadic}, warmup=3, iters=20, rounds=3)["g"]

    # The reported time is still the kernel's, not the kernel plus the gap.
    assert stalled.median < 2.0 * clean.median, (
        f"stall leaked into the median: {stalled.median:.4f} vs {clean.median:.4f}"
    )
    # And the gap is visible rather than absorbed.
    assert stalled.stall_ratio > 3.0, f"stall not flagged: {stalled.stall_ratio:.2f}"
    # The converse -- that a quiet run is *not* flagged -- is established by the
    # skip above rather than here, because on a loaded node it is not true and
    # should not be asserted.
    assert clean.stall_ratio < 2.0, (
        f"clean run falsely flagged: {clean.stall_ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# What is inside the timed region
# ---------------------------------------------------------------------------
#
# The published per-shape number is *kernel* time: the Python-side dispatch,
# the tuned-table lookup and the launcher in front of the kernel are outside it.
# That is a decision about what to measure, and it has exactly one way to go
# wrong -- taking the launcher out of one arm and not the other, which at these
# sizes is worth up to 1.4x in the direction that flatters us.  These tests are
# the guard on that, and each of them was verified by mutation -- breaking the
# thing it tests and confirming it fails.


def test_the_graph_chunk_is_one_ruler_for_every_arm():
    """``chunk`` is a function of the shortest arm's duration, and nothing else.

    A CUDA graph replay costs 3.9-12.8 us of device time whatever is inside it
    -- measured by fitting ``per_call(chunk) = kernel + cost / chunk`` to graphs
    of 1, 2, 4, 8, 16 and 32 calls on four real arms.  At ``chunk = 1`` that is
    45% of a 0.028 ms kernel and only 19% of a 0.068 ms one, so a per-arm chunk
    would be a per-arm instrument: exactly the failure ``_common_group`` already
    documents, one level up, where two byte-identical arms picked different
    event groups and read 4% apart.

    Hence: the rule reads only ``min(durations)``, so two arms of the same call
    can never be given different rulers.
    """
    from triton_conv3d.bench.harness import _REPLAY_COST_MS, common_chunk

    # Only the minimum matters: a slow second arm cannot loosen the ruler.
    assert common_chunk([0.03, 0.03]) == common_chunk([0.03, 3.0])
    assert common_chunk([0.03, 3.0]) == common_chunk([3.0, 0.03])
    # Monotone: a shorter kernel needs a wider graph.
    chunks = [common_chunk([d]) for d in (0.01, 0.03, 0.1, 0.3, 1.0, 10.0)]
    assert chunks == sorted(chunks, reverse=True), chunks
    # And the residual really is inside the budget it claims.
    for d in (0.01, 0.02, 0.05, 0.1, 0.5):
        c = common_chunk([d])
        residual = _REPLAY_COST_MS / c / d
        assert residual <= 0.011 or c == 128, (
            f"at {d} ms the chunk {c} leaves {residual:.1%} of replay cost in"
        )
    # A kernel long enough not to care is left alone.
    assert common_chunk([5.0]) == 1


def test_no_graph_where_the_launcher_is_already_negligible():
    """Above 40 ms per call the exclusion is not worth the capture.

    The largest host launch cost measured on this node is 0.08 ms -- the
    autograd engine's, on the MIOpen backward control.  At 40 ms per call that
    is 0.2% of either arm, a fifth of the harness's own 2% target, so both arms
    stay eager and the exclusion is negligible *for both* rather than applied to
    one.  Below it the same 0.08 ms reaches 190% of the kernel and decides the
    answer.
    """
    from triton_conv3d.bench.harness import graph_is_worthwhile

    assert graph_is_worthwhile([0.03])
    assert graph_is_worthwhile([0.03, 5000.0]), "the shortest arm decides"
    assert not graph_is_worthwhile([100.0])
    assert not graph_is_worthwhile([45241.0]), "the 2 GiB cliff cell"


@requires_gpu
def test_an_empty_graph_is_a_capture_failure():
    """PyTorch only *warns* when a capture caught nothing.

    "The CUDA Graph is empty.  This usually means that the graph was attempted
    to be captured on wrong device or stream."  It is a ``UserWarning``, and a
    caller that ignored it would publish the cost of ``cudaGraphLaunch`` -- a
    few microseconds -- as a kernel time.  That is the fastest wrong answer
    available and it looks like a spectacular win, so the warning is promoted to
    a refusal.

    It is not hypothetical: the first version of this work built the MIOpen
    backward control's forward graph on the default stream, captured on another,
    and got an empty graph plus this warning on two of six cells.
    """
    from triton_conv3d.bench.harness import CaptureError, capture

    with pytest.raises(CaptureError, match="empty"):
        capture(lambda: None, 1)


@requires_gpu
def test_a_captured_ratio_of_two_identical_arms_covers_one():
    """The null experiment for the launcher-exclusion boundary.

    Two arms doing byte-identical work have a true ratio of exactly 1.000, so
    anything else is the instrument.  Measured over 12 replications on
    ``convT 1024->512 @ 8^3`` through the shipped decision path: under
    ``exclude`` the median is 0.9996, the range 0.9982-1.0021, and **12 of 12**
    intervals cover 1.000.

    The sibling test for the *event* instrument is
    :func:`test_a_paired_ratio_of_two_identical_arms_covers_one`; this one is
    for the graph.
    """
    from triton_conv3d.bench.conv_bench import _timed_region
    from triton_conv3d.bench.harness import interleaved, ratio

    # 512, not a "nicer" 256 or 384: on this torch/ROCm build a bf16
    # ``a @ a`` is **~600 ms** at 128, 192, 256, 320, 384, 448, 640 and
    # 768, and 0.019 ms at 512 and 1024.  That is the ``torch.mm`` bf16
    # pathology this project already owes upstream, measured here from
    # a second direction; a test that picked one of the slow sizes would
    # be timing a 600 ms kernel and would correctly be told it does not
    # need a graph.
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    fn = lambda: a @ a  # noqa: E731
    region = _timed_region({"x": fn, "y": fn}, "exclude")
    assert region.kind == "kernel", region.note
    out = interleaved(region.fns, budget_s=10.0)
    r = ratio(out["y"], out["x"])
    # A tolerance, not the interval, and deliberately.  At 0.019 ms with a
    # 64-call graph one race converges to a *within-race* half-width of ~0.04%,
    # which is narrower than the between-race scatter of the same pair (sd
    # 0.13%, range 0.9982-1.0021 over 12 replications) -- so an interval that
    # misses 1 by 0.2% here is the same residual the sequential protocol has
    # (0.32%), not a biased instrument.  What a biased instrument looks like
    # is 4% (per-arm event groups) or 45% (a one-call graph), and 1% catches
    # both.  The coverage claim is the 12-replication
    # experiment, where 12 of 12 intervals contained 1.
    assert abs(r.point - 1.0) < 0.01, (
        f"two byte-identical arms read {r} under the kernel-time definition"
    )
    assert r.rel_half_width < 0.05, f"interval uselessly wide: {r}"


@requires_gpu
def test_an_inflated_launcher_does_not_move_the_reported_kernel_time():
    """The whole point of the exclusion, stated as a property.

    Three arms run the **same kernel** with deliberately different launchers.
    Under ``exclude`` they must be indistinguishable; under ``include`` they
    must not be, or the experiment proves nothing and the exclusion is
    measuring something that was not there.

    That negative control is deliberate.  Measured on the real thing
    (``launcher_symmetry.py --only inflate``, ``convT 1024->512 @ 8^3``): the
    entry point's own per-call table lookup reads 1.0003x of the hoisted config
    under ``exclude`` and **1.404x** under ``include``, and 500 us of Python in
    front of the launch reads 0.9993x and **13.12x**.
    """
    import time

    from triton_conv3d.bench.conv_bench import _timed_region
    from triton_conv3d.bench.harness import interleaved, ratio

    # 512, not a "nicer" 256 or 384: on this torch/ROCm build a bf16
    # ``a @ a`` is **~600 ms** at 128, 192, 256, 320, 384, 448, 640 and
    # 768, and 0.019 ms at 512 and 1024.  That is the ``torch.mm`` bf16
    # pathology this project already owes upstream, measured here from
    # a second direction; a test that picked one of the slow sizes would
    # be timing a 600 ms kernel and would correctly be told it does not
    # need a graph.
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)

    def plain():
        return a @ a

    def slow_launcher():
        end = time.perf_counter() + 300e-6
        while time.perf_counter() < end:
            pass
        return a @ a

    variants = {"plain": plain, "slow": slow_launcher}
    kern = _timed_region(dict(variants), "exclude")
    assert kern.kind == "kernel", kern.note
    k = interleaved(kern.fns, budget_s=10.0)
    rk = ratio(k["slow"], k["plain"])
    # 1%, for the reason given in
    # ``test_a_captured_ratio_of_two_identical_arms_covers_one``.  300 us in
    # front of a 0.019 ms kernel is a 16x effect if it is inside the timed
    # region, so 1% is not a generous threshold here.
    assert abs(rk.point - 1.0) < 0.01, (
        f"300 us of host work moved the kernel time: {rk}"
    )

    eager = _timed_region(dict(variants), "include")
    assert eager.kind == "call"
    e = interleaved(eager.fns, budget_s=10.0)
    re_ = ratio(e["slow"], e["plain"])
    assert re_.point > 2.0, (
        f"the negative control did not fire: the same host work read {re_} "
        "under the launcher-inclusive definition, so this test would pass "
        "against a version that excludes nothing"
    )


@requires_gpu
def test_the_replay_cost_is_amortized_by_the_chunk():
    """With its own negative control, like the event-tax test.

    One graph replay costs up to 12.8 us of device time whatever is in it, so a
    one-call graph is 45% instrument at a 0.028 ms kernel.  The chunk divides
    that away.  The control is the *same* kernel measured at ``chunk = 1`` in
    the same run: if the replay ever becomes free, the two agree and this fails
    rather than both drifting together unnoticed.
    """
    from triton_conv3d.bench.harness import capture, common_chunk, interleaved

    # 512, not a "nicer" 256 or 384: on this torch/ROCm build a bf16
    # ``a @ a`` is **~600 ms** at 128, 192, 256, 320, 384, 448, 640 and
    # 768, and 0.019 ms at 512 and 1024.  That is the ``torch.mm`` bf16
    # pathology this project already owes upstream, measured here from
    # a second direction; a test that picked one of the slow sizes would
    # be timing a 600 ms kernel and would correctly be told it does not
    # need a graph.
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    fn = lambda: a @ a  # noqa: E731
    for _ in range(50):
        fn()
    torch.cuda.synchronize()

    one = capture(fn, 1)
    chunk = common_chunk([0.02])
    assert chunk >= 8, chunk
    many = capture(fn, chunk)
    out = interleaved({"one": one, "many": many}, budget_s=10.0)
    per_one = out["one"].median
    per_many = out["many"].median / chunk
    assert per_many < per_one, (
        f"a {chunk}-call graph is not cheaper per call ({per_many:.5f}) than a "
        f"one-call graph ({per_one:.5f}); the replay cost has vanished and this "
        "test's premise with it -- check the ruler before deleting the chunk"
    )
    assert per_one - per_many < 0.05, "implausible replay cost; something else moved"


@requires_gpu
def test_a_capture_failure_takes_the_whole_cell_back_to_eager():
    """Never a mixed measurement.

    If one arm cannot be captured, the other must not be either: comparing a
    launcher-exclusive number against a launcher-inclusive one is worth 1.4x at
    the transposed sites and 3.0x on the backward controls.  So the fallback is
    a property of the *cell*, and ``_Region`` is one object for all of its arms.
    """
    from triton_conv3d.bench.conv_bench import _timed_region
    from triton_conv3d.bench.harness import Captured

    # 512, not a "nicer" 256 or 384: on this torch/ROCm build a bf16
    # ``a @ a`` is **~600 ms** at 128, 192, 256, 320, 384, 448, 640 and
    # 768, and 0.019 ms at 512 and 1024.  That is the ``torch.mm`` bf16
    # pathology this project already owes upstream, measured here from
    # a second direction; a test that picked one of the slow sizes would
    # be timing a 600 ms kernel and would correctly be told it does not
    # need a graph.
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)

    def fine():
        return a @ a

    def uncapturable():
        # A device-to-host read inside a capture is illegal, and this is how a
        # real arm fails: any control that peeks at a value does it.
        return (a @ a).sum().item()

    ok = _timed_region({"x": fine, "y": fine}, "exclude")
    assert ok.kind == "kernel"
    assert all(isinstance(f, Captured) for f in ok.fns.values())

    mixed = _timed_region({"x": fine, "y": uncapturable}, "exclude")
    assert mixed.kind == "call", "a cell with an uncapturable arm was captured"
    assert not any(isinstance(f, Captured) for f in mixed.fns.values()), (
        "one arm kept its graph while the other fell back -- that is the "
        "asymmetry the whole exclusion exists to avoid"
    )
    assert "could not be captured" in mixed.note


# ---------------------------------------------------------------------------
# The operator x direction table
# ---------------------------------------------------------------------------


def test_the_operator_direction_table_is_complete_and_has_six_distinct_cells():
    """Two operators, three directions, six builders, no sharing.

    The driver these replaced argued that a transposed convolution could not be
    a fourth value of ``--direction`` because it is a different *operator*.  It
    was right, and the answer is a second axis rather than a fourth case: what
    is per-operator (the shape form, the ordering) lives on ``_Op``, what is
    per-cell (operands, control, candidates, shipped config, reference) lives in
    one function per cell, and nothing is shared by accident.
    """
    from triton_conv3d.bench.conv_bench import _OPERATORS, OPERATORS

    assert set(_OPERATORS) == set(OPERATORS) == {"conv", "convT"}
    builders = []
    for op in _OPERATORS.values():
        assert set(op.build) == set(DIRECTIONS), op.name
        builders += list(op.build.values())
    assert len(set(builders)) == 6, "two cells share a builder"


def test_no_builder_asks_a_problem_which_operator_it_is():
    """The design property, asserted rather than trusted.

    The objection to folding the transposed driver in was that ``_build`` would
    "branch on ``problem.transposed`` in every arm to run the same code".  It
    does not: the operator is resolved **once**, in ``operator_of``, and the
    builder it selects never asks again.  If a future edit puts the question
    back inside a builder, this fails.
    """
    import inspect

    from triton_conv3d.bench.conv_bench import _OPERATORS, operator_of

    assert ".transposed" in inspect.getsource(operator_of)
    for op in _OPERATORS.values():
        for direction, builder in op.build.items():
            src = inspect.getsource(builder)
            assert ".transposed" not in src, (
                f"{op.name}/{direction} branches on the operator inside the "
                "builder; that is the switch this factoring exists to remove"
            )


def test_a_backward_control_is_never_a_fabricated_operand():
    """``torch.nn.grad.conv3d_*`` must appear nowhere in this driver.

    It has no real tensor for the operand being differentiated, so it fabricates
    ``grad_output.new_empty(1).expand(input_size)`` -- zero-strided, and
    therefore not channels-last.  ``convolution_backward`` picks its solver from
    that operand's layout, so at the ``k=1x1x1`` head MIOpen declined its own
    NDHWC path and ran **3.2x** slower than the same call inside a real
    backward (0.9649 vs 0.2972 ms).  A published 4.51x for that head came from
    the fabricated control; against the real one the cell is 1.39x.

    Both drivers that existed before this one used it somewhere, which is why
    the rule is a grep and not a convention.
    """
    import inspect

    from triton_conv3d.bench import conv_bench

    for op in conv_bench._OPERATORS.values():
        for direction, builder in op.build.items():
            if direction == "fwd":
                continue
            bsrc = inspect.getsource(builder)
            assert "torch.nn.grad" not in bsrc, (
                f"{op.name}/{direction} uses a fabricated operand for its "
                "MIOpen control"
            )
            assert "torch.autograd.grad" in bsrc, (
                f"{op.name}/{direction} has no real autograd control"
            )


def test_the_transposed_problems_are_never_haloed_and_the_others_follow_the_form():
    """The one shape decision that is per-operator, and it is silent when wrong.

    Upstream DistConv concatenates a ``k // 2`` halo onto every axis it manages
    and zeroes that axis's padding, so an ordinary convolution reaches MIOpen at
    ``130^3`` unpadded rather than ``128^3`` padded -- two problems MIOpen tunes
    independently.  At ``k = 2`` the halo is ``2 // 2 = 1``... which is why the
    *corpus* is the authority and not the arithmetic: every transposed problem
    in it records ``halo = (0, 0, 0)``, because ScaFFold's transposed sites are
    not sharded convolutions at all.  Applying ``halo_variant`` to them anyway
    would silently grow the input by two voxels per axis and measure a different
    problem -- **under any of the three ``--form`` names**, which is what this
    test pins now that there is more than one.
    """
    from triton_conv3d.bench.conv_bench import _FORMS, _OPERATORS

    conv, convt = _OPERATORS["conv"], _OPERATORS["convT"]
    assert set(_FORMS) == {"distconv", "adapter", "logical"}
    # The *function*, not just its effect on today's corpus: every transposed
    # problem happens to record a zero halo, so a form that called
    # ``halo_variant`` would be indistinguishable from the right one until the
    # day one of them did not.  Pin the rule instead.
    haloed = ConvProblem(
        "would_halo",
        32,
        16,
        (4, 4, 4),
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        transposed=True,
        halo=(1, 1, 1),
        shard_halo=(1, 1, 1),
    )
    for name in _FORMS:
        assert convt.form(haloed, name) is haloed, (
            f"--form {name} gave a transposed problem a halo; at k=2 there is "
            "none, and adding one measures a convolution the model never runs"
        )
    plain = dataclasses.replace(haloed, transposed=False)
    assert conv.form(plain, "distconv").spatial == (6, 6, 6)
    assert conv.form(plain, "adapter").spatial == (6, 6, 6)
    assert conv.form(plain, "logical").spatial == (4, 4, 4)

    seen = {"conv": 0, "convT": 0}
    for p in scaffold_corpus():
        if p.transposed:
            assert convt.selects(p) and not conv.selects(p)
            for name in _FORMS:
                assert convt.form(p, name) is p, f"{p.label} was haloed"
            assert p.halo == (0, 0, 0)
            seen["convT"] += 1
        else:
            assert conv.selects(p) and not convt.selects(p)
            assert conv.form(p, "distconv") == p.halo_variant
            assert conv.form(p, "adapter") == p.production_variant
            assert conv.form(p, "logical") is p
            seen["conv"] += 1
    assert seen["convT"] == 12 and seen["conv"] > 0, seen


@requires_gpu
def test_the_shipped_config_is_the_one_the_entry_point_resolves():
    """``--shipped`` must measure the shipped *kernel*, not a lookalike.

    The launcher-exclusive definition means the config cannot be resolved inside
    the timed region, so the driver resolves it outside and passes it in.  That
    is only honest if the two agree, and nothing except this test makes them:
    the six cells reach four different resolvers across three modules, with the
    channel widths swapped on three of them.

    Checked by spying on the resolver each entry point actually calls, rather
    than by re-deriving the answer here -- which would be the same arithmetic
    twice and would agree with itself while both were wrong.
    """
    from triton_conv3d import bwd_data, gather_gemm, reduce_gemm, transposed
    from triton_conv3d.bench.conv_bench import _OPERATORS, _build

    problems = {
        "conv": ConvProblem("t", 32, 16, (8, 8, 8)),
        "convT": ConvProblem(
            "tt", 32, 16, (4, 4, 4), (2, 2, 2), (2, 2, 2), (0, 0, 0), transposed=True
        ),
    }
    spied = []

    def spy(mod, name):
        real = getattr(mod, name)

        def wrapper(*a, **kw):
            cfg = real(*a, **kw)
            spied.append(cfg)
            return cfg

        return real, wrapper

    patched = [
        (gather_gemm, "select_config"),
        (bwd_data, "select_config"),
        (reduce_gemm, "bwd_weight_config"),
        (transposed, "transposed_config"),
    ]
    originals = {}
    for mod, name in patched:
        real, wrapper = spy(mod, name)
        originals[(mod, name)] = real
        setattr(mod, name, wrapper)
    try:
        for opname, op in _OPERATORS.items():
            for direction in DIRECTIONS:
                case = _build(problems[opname], direction, operator=opname)
                declared = case.shipped_config()
                spied.clear()
                case.triton(None)()
                torch.cuda.synchronize()
                assert spied, f"{opname}/{direction}: no resolver was called"
                assert declared in spied, (
                    f"{opname}/{direction}: --shipped would time {declared}, "
                    f"but the entry point resolves {spied}"
                )
                del case
                torch.cuda.empty_cache()
    finally:
        for (mod, name), real in originals.items():
            setattr(mod, name, real)


@requires_gpu
def test_the_published_time_is_per_call_and_never_exceeds_the_eager_call():
    """``chunk`` calls sit behind one replay; the row must report one call.

    The division happens in the driver rather than in the harness, because every
    *relative* quantity the harness computes -- the half-widths, the convergence
    test, the paired ratio -- is scale-invariant and only the absolute times need
    it.  That is easy to forget, and forgetting it multiplies every published
    time by up to 128 while leaving every interval and every speedup looking
    perfectly healthy.

    The invariant that catches it: kernel time is the eager call *minus* its
    launcher, so it can never exceed the eager call.
    """
    from triton_conv3d.bench.conv_bench import measure_problem

    p = ConvProblem(
        "tt", 64, 32, (4, 4, 4), (2, 2, 2), (2, 2, 2), (0, 0, 0), transposed=True
    )
    row = measure_problem(p, direction="fwd", shipped=True, budget_s=5.0)
    assert "error" not in row, row.get("error")
    assert row["timed_region"] == "kernel", row["timed_region_note"]
    assert row["graph_chunk"] > 1
    for arm in ("triton", "miopen"):
        kernel, eager = row[f"{arm}_ms"], row[f"{arm}_eager_ms"]
        assert 0.0 < kernel <= 1.05 * eager, (
            f"{arm}: reported kernel time {kernel:.5f} ms exceeds the eager "
            f"call it is part of ({eager:.5f} ms) -- the chunk divisor is "
            "missing or wrong"
        )
        assert row[f"{arm}_launcher_ms"] > 0


@requires_gpu
def test_a_control_free_row_omits_the_control_rather_than_zeroing_it():
    """``--control none`` must leave MIOpen *absent*, not present and zero.

    Two failures this pins, and they are opposite ones.

    A row that carried ``miopen_ms = 0.0`` and ``speedup = 0.0`` would be read
    by every consumer of these captures -- the report generator, the aggregate
    scripts, a human scanning a table -- as a measured 0.000x result rather than
    as "no control ran here".  Absence has to be representable.

    And a case built with ``control=False`` must not construct the control
    either.  For a backward direction the control is a real ``F.conv3d`` forward
    graph, and *running* it once is where MIOpen's find is paid -- 92-174 s per
    cell on this corpus.  Dropping the arm from the timing while still building
    it would save the timing and none of the cost, which is the whole point of
    the flag.
    """
    from triton_conv3d.bench.conv_bench import _build, measure_problem

    p = ConvProblem(
        "tt", 64, 32, (4, 4, 4), (2, 2, 2), (2, 2, 2), (0, 0, 0), transposed=True
    )

    for direction in DIRECTIONS:
        case = _build(p, direction, control=False)
        assert case.miopen is None, f"{direction}: a control was built anyway"
        assert case.reference is None, f"{direction}: a reference was built"
        del case
        torch.cuda.empty_cache()

    row = measure_problem(
        p, direction="bwd-weight", shipped=True, budget_s=5.0, control="none"
    )
    assert "error" not in row, row.get("error")
    assert row["control"] == "none"
    # The Triton half is unchanged: same region, same interval, same stop rule.
    assert row["timed_region"] == "kernel", row["timed_region_note"]
    assert row["triton_ms"] > 0.0
    # Present and finite, not strictly positive: on a kernel this small every
    # round can read the same value to the last bit, and ``stdev`` of identical
    # samples is exactly 0.  A zero half-width there is the honest answer, not a
    # missing one -- what would be wrong is the key being absent or ``inf``.
    assert math.isfinite(row["triton_rel_ci"]) and row["triton_rel_ci"] >= 0.0
    assert row["measure_stop"] in ("converged", "budget", "max_rounds")
    # The MIOpen half is gone, not zeroed.
    for key in (
        "miopen_ms",
        "miopen_rel_ci",
        "miopen_eager_ms",
        "speedup",
        "speedup_lo",
        "speedup_hi",
        "speedup_significant",
    ):
        assert key not in row, (
            f"{key} is present in a --control none row; an absent measurement "
            "must stay absent, because a zero here reads as a result"
        )
