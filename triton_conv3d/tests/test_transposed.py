# SPDX-License-Identifier: (Apache-2.0)
"""Tests for the ``kernel == stride`` transposed convolution.

Three directions, one new kernel, so these tests split unevenly on purpose.

**The forward** is a new ``@triton.jit`` function with a *scatter* store, which
is the one addressing pattern nothing else in this package has.  Its failure
mode is a permutation: write tap ``(kd,kh,kw)`` into the wrong sub-lattice and
the result is the right shape, the right magnitude, smooth, and wrong -- a
tolerance test cannot see it, and neither can a test that only checks sums.  So
the bar here is bitwise, and two tests exist purely to prove that bar is not
vacuous (:func:`test_a_transposed_tap_permutation_is_detected` and
:func:`test_bitwise_standard_rejects_a_shifted_scatter`), because this project
has shipped a vacuous exact test before.

**Both backward directions** are re-expressions: backward-data is
``conv3d_forward`` at ``stride = k`` and backward-weight is
``conv3d_backward_weight`` with the two activations swapped.  There is no new
arithmetic in either, so what is tested is the *re-expression* -- above all the
swap, which is the single most plausible mistake in the file and which produces
a correctly shaped gradient when it is wrong (:func:`test_backward_weight_
operand_swap_is_not_reversible`).

**The FLOP count** is checked in its own right.  ``k == s`` makes the per-tap
factor illusory (the windows tile rather than overlap) and this project once
counted it anyway, 8x too high.  ``shapes.py`` has it right; here it is checked
against the elementary MAC count of the reference implementation rather than
against another formula.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from triton_conv3d import reference
from triton_conv3d.gather_gemm import is_supported
from triton_conv3d.shapes import ConvProblem, scaffold_corpus
from triton_conv3d.transposed import (
    TransposedConfig,
    candidate_transposed_configs,
    conv_transpose3d_backward_data,
    conv_transpose3d_backward_weight,
    conv_transpose3d_forward,
    default_transposed_config,
    grad_transposed_weight_empty,
    is_supported_transposed,
    is_supported_transposed_all,
    is_supported_transposed_bwd_data,
    is_supported_transposed_bwd_weight,
    to_tkn,
    transposed_config,
)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _problem(
    name, cin, cout, spatial, k=(2, 2, 2), *, bias=False, dtype="bf16", n=1
) -> ConvProblem:
    return ConvProblem(
        name,
        cin,
        cout,
        spatial,
        k,
        k,
        (0, 0, 0),
        n=n,
        transposed=True,
        bias=bias,
        dtype=dtype,
        sites=("synthetic",),
    )


#: Synthetic problems chosen to break the scatter rather than to be fast.  Each
#: one targets a specific way a tiling store goes wrong.
EDGE: list[ConvProblem] = [
    # The four channel pairs the model contains, at a volume an fp64 reference
    # can afford.  The pairs matter because they decide BLOCK_NC and TAP_BLOCK.
    _problem("128to64", 128, 64, (4, 4, 4), bias=True),
    _problem("256to128", 256, 128, (4, 4, 4), bias=True),
    _problem("512to256", 512, 256, (2, 2, 2), bias=True),
    _problem("1024to512", 1024, 512, (2, 2, 2), bias=True),
    # Channel counts that divide no plausible tile: Cout below the MFMA
    # granularity is the interesting one, since BLOCK_NC cannot go under 16 and
    # the column mask is then the only thing keeping the store in bounds.
    _problem("cout_tiny", 64, 6, (4, 5, 6), bias=True),
    _problem("cout_odd", 32, 7, (3, 4, 5)),
    _problem("cin_prime", 17, 24, (3, 5, 7), bias=True),
    _problem("cin_one", 1, 32, (4, 4, 4)),
    _problem("cout_one", 32, 1, (4, 4, 4), bias=True),
    # Spatial extents that do not divide BLOCK_M, so the M-unravel's rows wrap
    # the W axis inside a tile and the scatter's rows are no longer a dense run.
    _problem("spatial_prime", 32, 32, (13, 11, 7)),
    _problem("spatial_one", 32, 32, (1, 5, 8)),
    _problem("spatial_thin", 32, 32, (2, 31, 3)),
    # Batch > 1: ScaFFold never does this, but the M decomposition must.
    _problem("batched", 32, 32, (3, 4, 5), n=3, bias=True),
    # Kernels other than 2.  ``k=3`` gives 27 taps, which no power of two
    # divides, so TAP_BLOCK must fall back to 1; the anisotropic ones check that
    # the fused tap index is unpacked in the right radix order.
    _problem("k3", 32, 32, (3, 4, 5), (3, 3, 3), bias=True),
    _problem("k_aniso", 32, 32, (3, 4, 5), (1, 2, 4), bias=True),
    _problem("k_aniso2", 32, 32, (4, 3, 2), (4, 2, 1)),
    _problem("k1", 32, 32, (4, 5, 6), (1, 1, 1), bias=True),
    # fp32 (more_determinism) and fp16, which change the LDS budget and the
    # MFMA reduction depth.
    _problem("fp32", 64, 64, (4, 4, 4), dtype="fp32", bias=True),
    _problem("fp16", 64, 64, (4, 4, 4), dtype="fp16", bias=True),
]

#: The corpus's real transposed problems, restated at a volume an fp64 reference
#: can afford.  What survives the restatement is what matters: the channel
#: widths, and with them ``EVEN_N``, ``TAP_BLOCK`` and the tile selection.  The
#: extents are deliberately not powers of two so ``BLOCK_M`` does not divide
#: ``M`` and the store's rows wrap.
CORPUS_PAIRS: list[ConvProblem] = [
    _problem(
        f"{p.cin}to{p.cout}-corpus",
        p.cin,
        p.cout,
        (3, 4, 5),
        tuple(p.kernel),
        bias=p.bias,
    )
    for p in {
        (q.cin, q.cout, tuple(q.kernel), q.bias): q
        for q in scaffold_corpus()
        if q.transposed
    }.values()
]


def _ids(problems):
    return [p.name or p.label for p in problems]


def _ops(problem: ConvProblem, seed: int = 0, direction="fwd") -> dict:
    """Operands drawn so the *realized* sums stay inside the mantissa.

    ``exact_density`` rather than a dense ``{-1,0,1}`` draw, for the reason it
    documents: at ``Cin = 1024`` the forward reduces over 1024 terms and a sum
    of that many random signs runs past bf16's integer limit of 256, so a dense
    draw would skip every wide problem -- which is where the coverage is needed.
    The shape is untouched, so the channel widths and the tile selection under
    test stay exactly what ScaFFold runs.
    """
    dtype = reference.torch_dtype(problem)
    return reference.make_inputs(
        problem,
        seed=seed,
        exact=True,
        density=reference.exact_density(problem, direction, dtype=dtype),
    )


def _reference(problem: ConvProblem, ops: dict, direction: str) -> torch.Tensor:
    return reference.reference(problem, ops, direction)


def _fwd(problem: ConvProblem, ops: dict, **kw) -> torch.Tensor:
    return conv_transpose3d_forward(
        ops["input"], ops["weight"], ops["bias"], problem.stride, **kw
    )


def _bwd_data(problem: ConvProblem, ops: dict, **kw) -> torch.Tensor:
    return conv_transpose3d_backward_data(
        ops["grad_output"],
        ops["weight"],
        problem.input_shape,
        problem.stride,
        **kw,
    )


def _bwd_weight(problem: ConvProblem, ops: dict, **kw) -> torch.Tensor:
    return conv_transpose3d_backward_weight(
        ops["input"],
        problem.weight_shape,
        ops["grad_output"],
        problem.stride,
        **kw,
    )


# ---------------------------------------------------------------------------
# The algebra, before any GPU is involved
# ---------------------------------------------------------------------------


def test_the_windows_tile_the_output_exactly_once():
    """The identity the whole module rests on, checked by counting.

    At ``k == s`` every output voxel must be written by exactly one ``(input
    voxel, tap)`` pair.  A ``k != s`` case is included as the negative control:
    there the count is not 1 everywhere, which is precisely why this module
    refuses it rather than generalising.
    """
    for k in ((2, 2, 2), (3, 3, 3), (1, 2, 4), (4, 2, 1)):
        extents = (3, 4, 5)
        hits = torch.zeros(tuple(e * kk for e, kk in zip(extents, k)))
        for d in range(extents[0]):
            for h in range(extents[1]):
                for w in range(extents[2]):
                    for kd in range(k[0]):
                        for kh in range(k[1]):
                            for kw in range(k[2]):
                                hits[d * k[0] + kd, h * k[1] + kh, w * k[2] + kw] += 1
        assert torch.equal(hits, torch.ones_like(hits)), k

    # ``k=3, s=2`` overlaps: the windows cover some voxels twice.  If this ever
    # stops being true the gate could be widened; it is here so that widening it
    # by accident is impossible.
    k, s, extents = 3, 2, (4, 1, 1)
    hits = torch.zeros((extents[0] - 1) * s + k)
    for d in range(extents[0]):
        for kd in range(k):
            hits[d * s + kd] += 1
    assert hits.max() > 1


def test_transposed_flops_have_no_phantom_tap_factor():
    """``flops()`` against the elementary MAC count, not against another formula.

    The trap: the general transposed FLOP count carries a per-tap factor, and at
    ``k == s`` it does not apply, because the windows tile rather than overlap.
    Applying it anyway overstates the count by ``taps`` -- 8x at ``k=2`` -- which
    this project did once, and a wrong FLOP count is invisible: it produces a
    plausible roofline percentage and a wrong conclusion about where the
    opportunity is.

    So the count is derived here from first principles: one MAC per (output
    voxel, output channel, input channel), times two.
    """
    for cin, cout, spatial, k in [
        (128, 64, (4, 5, 6), (2, 2, 2)),
        (32, 32, (3, 3, 3), (3, 3, 3)),
        (16, 8, (2, 3, 4), (1, 2, 4)),
    ]:
        p = _problem("f", cin, cout, spatial, k)
        out_vol = math.prod(p.out_spatial)
        macs = out_vol * cout * cin
        assert p.flops("fwd") == 2 * macs, p.label
        # Every direction performs the same contraction, so all three agree.
        assert p.flops("bwd-data") == 2 * macs
        assert p.flops("bwd-weight") == 2 * macs
        # And the GEMM decomposition has to describe the same contraction:
        # M*N*K must equal the MAC count, with K = Cin and no taps in it.
        m, n, kk = p.gemm_shape("fwd")
        assert m * n * kk == macs, (p.label, (m, n, kk))
        assert kk == cin, "the forward's K carries a tap factor it should not"


def test_the_gemm_decomposition_matches_the_kernels_grid():
    """``gemm_shape`` and the launch have to agree about what N is.

    ``gemm_shape`` reports ``N = Cout * taps`` and the kernel tiles that as
    ``(taps // TAP_BLOCK)`` groups of ``TAP_BLOCK * BLOCK_NC`` columns.  If the
    two ever disagree the cost model is describing a different kernel from the
    one that runs, which is the class of error that produced this project's
    largest published mistake.
    """
    for p in CORPUS_PAIRS + EDGE:
        m, n, k = p.gemm_shape("fwd")
        taps = p.tap_count
        assert n == p.cout * taps
        assert m == p.n * math.prod(p.spatial)
        cfg = transposed_config(m, p.cin, p.cout, p.kernel, reference.torch_dtype(p))
        assert taps % cfg.TAP_BLOCK == 0, (p.label, cfg)
        columns = (
            (taps // cfg.TAP_BLOCK)
            * cfg.TAP_BLOCK
            * (-(-p.cout // cfg.BLOCK_NC) * cfg.BLOCK_NC)
        )
        assert columns >= n, (p.label, cfg)


# ---------------------------------------------------------------------------
# Configuration legality -- the failure mode is silent, so it is checked apart
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem", EDGE + CORPUS_PAIRS, ids=_ids(EDGE + CORPUS_PAIRS))
def test_selected_config_is_legal_for_every_shape(problem: ConvProblem):
    """An illegal MFMA config does not fail, it silently emits vector FMA.

    So legality is asserted rather than discovered from a timing.  ``TAP_BLOCK``
    adds two constraints the other kernels do not have -- it must divide the tap
    count, and it multiplies ``BLOCK_NC`` into ``BLOCK_N`` -- and both are
    checked here at every shape the module can be handed, including ``k=3``
    (27 taps, which no power of two divides) and ``Cout=1``.
    """
    dtype = reference.torch_dtype(problem)
    m = problem.n * math.prod(problem.spatial)
    cfg = transposed_config(m, problem.cin, problem.cout, problem.kernel, dtype)
    assert cfg.validate(dtype) is None, (problem.label, cfg)
    assert cfg.lds_bytes(dtype) <= 64 * 1024, (problem.label, cfg)
    assert problem.tap_count % cfg.TAP_BLOCK == 0, (problem.label, cfg)
    assert cfg.BLOCK_N == cfg.TAP_BLOCK * cfg.BLOCK_NC


@pytest.mark.parametrize("problem", CORPUS_PAIRS, ids=_ids(CORPUS_PAIRS))
def test_every_candidate_config_is_legal(problem: ConvProblem):
    """The sweep must never time a config that cannot reach the matrix core.

    A ranked-last illegal config still pollutes a best-of: it runs, it is
    correct, and it is slow for a reason that has nothing to do with the tile.
    """
    dtype = reference.torch_dtype(problem)
    m = problem.n * math.prod(problem.spatial)
    cands = candidate_transposed_configs(
        m, problem.cin, problem.cout, problem.tap_count, dtype
    )
    assert cands
    for cfg in cands:
        assert cfg.validate(dtype) is None, cfg
        assert cfg.lds_bytes(dtype) <= 64 * 1024, cfg
        assert problem.tap_count % cfg.TAP_BLOCK == 0, cfg


def test_the_fp32_config_fits_lds():
    """fp32 operands are twice the bytes, and that hole has bitten before.

    ``more_determinism`` runs the model in fp32, and the gather kernel shipped a
    ``default_config`` that asked for 128 KiB there -- reachable from a real
    ScaFFold configuration.  This kernel's tile is *wider* than that one's
    (``TAP_BLOCK`` multiplies the column count), so the same hole is closer.
    """
    for cin, cout, taps in [
        (1024, 512, 8),
        (512, 256, 8),
        (256, 128, 8),
        (128, 64, 8),
        (64, 64, 27),
        (2048, 1024, 8),
    ]:
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            cfg = default_transposed_config(1 << 16, cin, cout, taps, dtype)
            assert cfg.validate(dtype) is None, (cin, cout, dtype, cfg)
            assert cfg.lds_bytes(dtype) <= 64 * 1024, (cin, cout, dtype, cfg)


def test_transposed_config_is_a_pure_function_of_its_arguments():
    """No device state, no clock, no allocator: two calls must agree.

    The same property ``split_count`` needs and for a weaker but related reason
    -- a tuning choice that varied between two runs of the same shape would make
    the kernel's own reproducibility claim untestable.
    """
    args = (1 << 20, 128, 64, (2, 2, 2), torch.bfloat16)
    assert transposed_config(*args) == transposed_config(*args)


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


def test_is_supported_declines_what_the_tiling_argument_does_not_cover():
    """Everything outside ``kernel == stride, p = 0, output_padding = 0, dil = 1``.

    Each of these breaks the bijection ``(d, kd) -> d*k + kd`` in a different
    way, and the failure is not a crash: ``k != s`` would write some output
    voxels twice and others never, which is a smooth, plausible, wrong answer.
    """
    x = torch.zeros(1, 8, 4, 4, 4)
    w = torch.zeros(8, 4, 2, 2, 2)
    # The shape checks run on CPU tensors, so a True is impossible here; what is
    # asserted is that each of these is refused *before* the device test, which
    # is why the positive control below is on the GPU.
    assert not is_supported_transposed(x, w, None, 2, 0, 0, 1, 1)  # not cuda
    for stride, padding, output_padding, dilation, groups in [
        (1, 0, 0, 1, 1),  # k != s: the windows overlap
        (3, 0, 0, 1, 1),  # k != s: the windows leave gaps
        (2, 1, 0, 1, 1),  # padding crops the tiled result
        (2, 0, 1, 1, 1),  # output_padding extends it asymmetrically
        (2, 0, 0, 2, 1),  # dilation interleaves the window with holes
        (2, 0, 0, 1, 2),  # groups
        ((2, 2, 1), 0, 0, 1, 1),  # anisotropic mismatch on one axis only
    ]:
        assert not is_supported_transposed(
            x.cuda() if torch.cuda.is_available() else x,
            w.cuda() if torch.cuda.is_available() else w,
            None,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        ), (stride, padding, output_padding, dilation, groups)


def test_the_gates_are_total():
    """An argument the gate cannot interpret is a ``False``, never an exception.

    This is the gate of a Triton -> MIOpen rung ladder.  A caller that is only
    asking a question must not be taken down by the answer, and ``_triple``
    raises ``TypeError`` on ``None`` and ``ValueError`` on a bad length.
    """
    x = torch.zeros(1, 8, 4, 4, 4)
    w = torch.zeros(8, 4, 2, 2, 2)
    for bad in (None, 1.5, "2", (2, 2), (2, 2, 2, 2), object()):
        assert is_supported_transposed(x, w, None, bad, 0, 0, 1, 1) is False
        assert is_supported_transposed(x, w, None, 2, bad, 0, 1, 1) is False
        assert is_supported_transposed(x, w, None, 2, 0, bad, 1, 1) is False
        assert is_supported_transposed(x, w, None, 2, 0, 0, bad, 1) is False
        assert is_supported_transposed_all(x, w, None, bad, 0, 0, 1, 1) is False
        assert (
            is_supported_transposed_bwd_data(x, w, (1, 8, 4, 4, 4), bad, 0, 0, 1, 1)
            is False
        )
        assert (
            is_supported_transposed_bwd_weight(x, (8, 4, 2, 2, 2), x, bad, 0, 0, 1, 1)
            is False
        )
    # A malformed ``input_shape`` / ``weight_shape`` is the same kind of
    # question and gets the same kind of answer.
    for bad in (None, (1, 8, 4, 4), "abcde", 5):
        assert is_supported_transposed_bwd_data(x, w, bad, 2, 0, 0, 1, 1) is False
        assert is_supported_transposed_bwd_weight(x, bad, x, 2, 0, 0, 1, 1) is False


def test_is_supported_declines_degenerate_extents():
    """Zero-length axes and empty channel counts, which torch handles otherwise.

    Each clears the tiling argument and then disagrees with torch: a zero-length
    spatial axis gives an output the M-unravel has no rows to index, and
    ``Cin = 0`` returns ``Cout`` channels of zeros where torch returns a tensor
    with no channels at all -- a different *shape*, not a different value.
    """
    good_x = torch.zeros(1, 8, 4, 4, 4)
    good_w = torch.zeros(8, 4, 2, 2, 2)
    assert not is_supported_transposed(
        torch.zeros(1, 8, 0, 4, 4), good_w, None, 2, 0, 0, 1, 1
    )
    assert not is_supported_transposed(
        good_x, torch.zeros(0, 4, 2, 2, 2), None, 2, 0, 0, 1, 1
    )
    assert not is_supported_transposed(
        good_x, torch.zeros(8, 0, 2, 2, 2), None, 2, 0, 0, 1, 1
    )
    assert not is_supported_transposed(
        good_x, torch.zeros(8, 4, 0, 2, 2), None, (0, 2, 2), 0, 0, 1, 1
    )


@requires_gpu
def test_is_supported_reads_the_transposed_weight_convention():
    """``(Cin, Cout, k, k, k)``, not ``(Cout, Cin, k, k, k)``.

    The two operators store their weights with the channel axes the other way
    round.  A gate that read ``nn.Conv3d``'s convention would accept a weight
    whose channel counts happen to match and compute a transposed answer -- the
    right shape, the wrong numbers, silently.
    """
    x = torch.zeros(1, 8, 4, 4, 4, device="cuda", dtype=torch.bfloat16)
    assert is_supported_transposed(
        x,
        torch.zeros(8, 4, 2, 2, 2, device="cuda", dtype=torch.bfloat16),
        None,
        2,
        0,
        0,
        1,
        1,
    )
    # Same tensor read the other way round: Cin=4 does not match x's 8 channels.
    assert not is_supported_transposed(
        x,
        torch.zeros(4, 8, 2, 2, 2, device="cuda", dtype=torch.bfloat16),
        None,
        2,
        0,
        0,
        1,
        1,
    )
    # A bias is ``Cout`` = w.shape[1] long, not w.shape[0].
    w = torch.zeros(8, 4, 2, 2, 2, device="cuda", dtype=torch.bfloat16)
    assert is_supported_transposed(
        x, w, torch.zeros(4, device="cuda", dtype=torch.bfloat16), None or 2, 0, 0, 1, 1
    )
    assert not is_supported_transposed(
        x, w, torch.zeros(8, device="cuda", dtype=torch.bfloat16), 2, 0, 0, 1, 1
    )
    # A stride-2 view of the right length applies every other value; the kernel
    # indexes the bias with an element stride of 1 and cannot see this.
    long_bias = torch.zeros(8, device="cuda", dtype=torch.bfloat16)
    assert not is_supported_transposed(x, w, long_bias[::2], 2, 0, 0, 1, 1)


@requires_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_is_supported_declines_operands_on_different_devices():
    """Triton launches on the current device and dereferences the other pointer.

    ScaFFold runs four GPUs per node with peer access, so a foreign pointer does
    not fault -- it reads another rank's memory, which is a plausible wrong
    answer rather than a crash.
    """
    x = torch.zeros(1, 8, 4, 4, 4, device="cuda:0", dtype=torch.bfloat16)
    w = torch.zeros(8, 4, 2, 2, 2, device="cuda:1", dtype=torch.bfloat16)
    assert not is_supported_transposed(x, w, None, 2, 0, 0, 1, 1)
    assert not is_supported_transposed_all(x, w, None, 2, 0, 0, 1, 1)


@requires_gpu
@pytest.mark.parametrize("problem", EDGE + CORPUS_PAIRS, ids=_ids(EDGE + CORPUS_PAIRS))
def test_all_three_gates_accept_every_problem_this_module_serves(problem):
    """``is_supported_transposed_all`` must not be narrower than the forward.

    Unlike the ordinary convolution -- whose three gates genuinely disagree
    about ``stride > 1``, which is a trap the package documents -- all three
    directions of this operator accept the same problems, because both backward
    directions are the same ``k == s`` convolution seen from the other side.
    That is an argument, and the adapter needs a fact: a site only leaves the
    block-list if the *combined* gate says yes, so it is asked for real here at
    every shape the module claims.
    """
    dtype = reference.torch_dtype(problem)
    x = torch.zeros(problem.input_shape, device="cuda", dtype=dtype)
    w = torch.zeros(problem.weight_shape, device="cuda", dtype=dtype)
    b = torch.zeros(problem.cout, device="cuda", dtype=dtype) if problem.bias else None
    args = (problem.stride, 0, 0, 1, 1)
    assert is_supported_transposed(x, w, b, *args), problem.label
    assert is_supported_transposed_all(x, w, b, *args), problem.label


def test_the_ordinary_forward_gate_would_not_have_served_these():
    """Why this module exists at all, stated as a test.

    The ordinary ``is_supported`` takes no ``transposed`` parameter, so a caller
    holding a ``ConvTranspose3d`` has no way to ask it the right question: it
    answers about the *non*-transposed convolution with the same tensors, whose
    output shape is 8x smaller.  Asking it and believing the answer is precisely
    the bug the adapter's ``module.transposed`` check exists to prevent.
    """
    x = torch.zeros(1, 128, 4, 4, 4)
    w = torch.zeros(128, 64, 2, 2, 2)
    # It answers -- about a 128 -> 64 strided convolution, not about the
    # upsample -- and the answer says nothing about this operator.
    assert is_supported(x, w, None, 2, 0, 1, 1) in (True, False)
    p = _problem("t", 128, 64, (4, 4, 4))
    assert p.out_spatial == (8, 8, 8)
    non_transposed = (4 + 2 * 0 - 2) // 2 + 1
    assert non_transposed == 2 != 8


# ---------------------------------------------------------------------------
# Correctness: bitwise, because a permuted scatter is invisible to a tolerance
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("problem", EDGE, ids=_ids(EDGE))
def test_forward_matches_bitwise(problem: ConvProblem):
    ops = _ops(problem, seed=3, direction="fwd")
    expected = _reference(problem, ops, "fwd")
    dtype = reference.torch_dtype(problem)
    assert reference.is_exactly_representable(expected, dtype), (
        f"{problem.label}: the draw is not exact, so this case tests nothing"
    )
    actual = _fwd(problem, ops)
    assert reference.compare(actual, expected.to(dtype)).bitwise, problem.label


@requires_gpu
@pytest.mark.parametrize("problem", EDGE, ids=_ids(EDGE))
def test_backward_data_matches_bitwise(problem: ConvProblem):
    ops = _ops(problem, seed=5, direction="bwd-data")
    expected = _reference(problem, ops, "bwd-data")
    dtype = reference.torch_dtype(problem)
    assert reference.is_exactly_representable(expected, dtype), problem.label
    actual = _bwd_data(problem, ops)
    assert reference.compare(actual, expected.to(dtype)).bitwise, problem.label


@requires_gpu
@pytest.mark.parametrize("problem", EDGE, ids=_ids(EDGE))
def test_backward_weight_matches_bitwise(problem: ConvProblem):
    ops = _ops(problem, seed=7, direction="bwd-weight")
    expected = _reference(problem, ops, "bwd-weight")
    dtype = reference.torch_dtype(problem)
    assert reference.is_exactly_representable(expected, dtype), problem.label
    actual = _bwd_weight(problem, ops)
    assert reference.compare(actual, expected.to(dtype)).bitwise, problem.label


@requires_gpu
@pytest.mark.parametrize("problem", CORPUS_PAIRS, ids=_ids(CORPUS_PAIRS))
@pytest.mark.parametrize("direction", ["fwd", "bwd-data", "bwd-weight"])
def test_corpus_channel_pairs_match_bitwise(problem: ConvProblem, direction: str):
    """Every ``ConvTranspose3d`` channel pair ScaFFold runs, bitwise in bf16.

    ``exact_density`` is what makes this reachable at ``Cin = 1024``: it thins
    the activations so the *realized* sums stay inside bf16's mantissa while the
    shape -- and so the tile, ``TAP_BLOCK`` and the 512-byte row strides -- is
    exactly what the model runs.  Asserted rather than skipped, so this cannot
    quietly become a wall of passes that tests nothing, which is how a sibling
    file lost its whole real-shape coverage once.
    """
    ops = _ops(problem, seed=11, direction=direction)
    expected = _reference(problem, ops, direction)
    dtype = reference.torch_dtype(problem)
    assert reference.is_exactly_representable(expected, dtype), problem.label
    actual = {"fwd": _fwd, "bwd-data": _bwd_data, "bwd-weight": _bwd_weight}[direction](
        problem, ops
    )
    assert reference.compare(actual, expected.to(dtype)).bitwise, problem.label


@requires_gpu
def test_a_transposed_tap_permutation_is_detected():
    """The bug this kernel can uniquely have, pinned.

    Each output voxel takes its value from one tap, and which tap is decided by
    ``(D % kd, H % kh, W % kw)``.  Unpack the fused tap index in the wrong radix
    order -- swap ``kd`` and ``kw``, the single most plausible mistake in the
    epilogue -- and every value written is a value that *belongs* somewhere in
    the output, just not there.  The norms are identical, the histogram is
    identical, and every tolerance test ever written passes.

    So construct exactly that wrong answer, from the same operands, and require
    a bitwise mismatch.  An anisotropic volume is used so that the permutation
    cannot coincide with a symmetry of the data.
    """
    problem = _problem("perm", 32, 32, (3, 4, 5))
    ops = _ops(problem, seed=13)
    actual = _fwd(problem, ops)
    correct = _reference(problem, ops, "fwd").to(torch.bfloat16)
    assert reference.compare(actual, correct).bitwise

    # The same convolution with the weight's three kernel axes permuted, which
    # is exactly what unpacking the fused index in the wrong order computes.
    permuted = _reference(
        problem,
        {
            **ops,
            "weight": ops["weight"]
            .permute(0, 1, 4, 3, 2)
            .contiguous(memory_format=torch.channels_last_3d),
        },
        "fwd",
    ).to(torch.bfloat16)
    assert permuted.shape == actual.shape
    assert not reference.compare(actual, permuted).bitwise, (
        "a kd/kw-swapped tap unpacking gave a bitwise-identical answer; the "
        "scatter's radix order is untested by this suite"
    )
    # And the sums agree, which is the point: nothing weaker than bitwise sees it.
    assert torch.allclose(actual.double().sum(), permuted.double().sum())


@requires_gpu
def test_bitwise_standard_rejects_a_shifted_scatter():
    """Prove the comparison discriminates at all.

    ``{-1,0,1}`` operands could in principle make two different answers agree,
    and this project has shipped two vacuous exact tests before.  A one-voxel
    roll of the input is a different convolution and must be rejected.
    """
    problem = _problem("shift", 16, 16, (3, 4, 5))
    ops = _ops(problem, seed=17)
    actual = _fwd(problem, ops)
    correct = _reference(problem, ops, "fwd").to(torch.bfloat16)
    assert reference.compare(actual, correct).bitwise

    rolled = torch.roll(ops["input"], shifts=1, dims=-1)
    wrong = _reference(problem, {**ops, "input": rolled}, "fwd").to(torch.bfloat16)
    assert not reference.compare(actual, wrong).bitwise


@requires_gpu
def test_backward_weight_operand_swap_is_not_reversible():
    """The one mistake the backward-weight re-expression can make.

    ``conv_transpose3d_backward_weight`` hands ``grad_output`` to
    ``conv3d_backward_weight``'s *input* slot and ``x`` to its *grad_output*
    slot.  The swap is checked twice, because it has two regimes and only one of
    them is dangerous:

    * at ``k > 1`` the swap is **not shape-legal** -- the strided convolution's
      input is the ``k``-times-larger volume, so ``is_supported_bwd_weight``
      refuses it.  That is worth pinning as a fact rather than assumed: it is
      the reason the swap cannot silently produce a wrong gradient at any real
      ScaFFold site.
    * at ``k == 1`` the two activations have the *same* shape, the gate cannot
      tell them apart, and the swap returns a correctly shaped, transposed
      gradient.  That is the case where only the operand order stands between a
      right and a wrong answer, so it is constructed and required to differ.
    """
    from triton_conv3d.reduce_gemm import conv3d_backward_weight

    problem = _problem("swap", 32, 32, (3, 4, 5))
    ops = _ops(problem, seed=19, direction="bwd-weight")
    actual = _bwd_weight(problem, ops)
    expected = _reference(problem, ops, "bwd-weight").to(torch.bfloat16)
    assert reference.compare(actual, expected).bitwise
    with pytest.raises(NotImplementedError):
        conv3d_backward_weight(
            ops["input"],
            problem.weight_shape,
            ops["grad_output"],
            problem.stride,
            0,
            1,
            1,
        )

    # ``k=1``: same shapes, so nothing but the argument order decides.
    flat = _problem("swap1", 32, 32, (3, 4, 5), (1, 1, 1))
    ops1 = _ops(flat, seed=19, direction="bwd-weight")
    got = _bwd_weight(flat, ops1)
    want = _reference(flat, ops1, "bwd-weight").to(torch.bfloat16)
    assert reference.compare(got, want).bitwise
    swapped = conv3d_backward_weight(
        ops1["input"], flat.weight_shape, ops1["grad_output"], flat.stride, 0, 1, 1
    )
    assert swapped.shape == got.shape
    assert not reference.compare(got, swapped).bitwise, (
        "swapping the two activations gave the same gradient; the operand "
        "order of the backward-weight re-expression is untested"
    )


@requires_gpu
def test_backward_data_is_the_strided_convolution_it_claims_to_be():
    """The re-expression, stated as an identity and checked bitwise.

    ``grad_input = conv3d(grad_output, w, stride=k)`` with ``w`` *unpermuted*.
    If a permute were needed the two would differ, and the difference would be a
    transposed gradient of the right shape whenever ``Cin == Cout``.
    """
    problem = _problem("bd", 64, 32, (3, 4, 5))
    ops = _ops(problem, seed=23, direction="bwd-data")
    from triton_conv3d.gather_gemm import conv3d_forward

    direct = conv3d_forward(
        ops["grad_output"], ops["weight"], None, problem.stride, 0, 1, 1
    )
    assert torch.equal(direct, _bwd_data(problem, ops))


# ---------------------------------------------------------------------------
# The tuning surface, the layouts, and the entry-point contract
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("problem", EDGE[:8], ids=_ids(EDGE[:8]))
def test_every_config_gives_the_same_answer(problem: ConvProblem):
    """The whole tuning surface, not the one point the table happens to pick.

    ``TAP_BLOCK`` is the axis that matters here: it changes how many taps share
    an accumulator and therefore the column decomposition of the store, so a
    mask that is right at ``TAP_BLOCK=1`` and wrong at 8 would be invisible to
    a test that only ran the shipped config.  Capped at four configs per shape
    to keep the JIT cost bounded; they are chosen to span ``TAP_BLOCK``.
    """
    dtype = reference.torch_dtype(problem)
    ops = _ops(problem, seed=29)
    expected = _reference(problem, ops, "fwd")
    assert reference.is_exactly_representable(expected, dtype)
    m = problem.n * math.prod(problem.spatial)
    cands = candidate_transposed_configs(
        m, problem.cin, problem.cout, problem.tap_count, dtype
    )
    by_tb: dict[int, TransposedConfig] = {}
    for cfg in cands:
        by_tb.setdefault(cfg.TAP_BLOCK, cfg)
    chosen = list(by_tb.values())[:4]
    assert chosen, problem.label
    assert len({c.TAP_BLOCK for c in chosen}) == len(chosen)
    for cfg in chosen:
        actual = _fwd(problem, ops, config=cfg)
        assert reference.compare(actual, expected.to(dtype)).bitwise, (
            f"{problem.label} with {cfg}"
        )


@requires_gpu
def test_every_weight_layout_gives_the_same_answer():
    """Channels-last, the materialized ``(t, K, N)`` buffer, and PyTorch's default.

    Three layouts, one answer.  The middle one is the copy
    :func:`~triton_conv3d.transposed.to_tkn` makes for a weight the plan
    refuses, and the last one is what the plan refuses -- a weight where neither
    channel axis is unit-stride.  Getting the stride plan wrong is a *silent*
    wrong answer, because the kernel will happily read whatever the strides say.
    """
    problem = _problem("layout", 64, 32, (3, 4, 5), bias=True)
    ops = _ops(problem, seed=31)
    expected = _reference(problem, ops, "fwd").to(torch.bfloat16)
    cl = ops["weight"]
    assert cl.is_contiguous(memory_format=torch.channels_last_3d)
    plain = cl.contiguous()
    assert not plain.is_contiguous(memory_format=torch.channels_last_3d)
    for w in (cl, plain, to_tkn(cl).permute(3, 4, 0, 1, 2)):
        got = conv_transpose3d_forward(ops["input"], w, ops["bias"], problem.stride)
        assert reference.compare(got, expected).bitwise, tuple(w.stride())


@requires_gpu
def test_out_buffer_is_written_in_place_and_is_validated():
    """``out=`` is checked rather than trusted, and nothing downstream catches it.

    The grid is sized from the *problem*, not from ``out``, and the store
    addressing assumes a channel stride of 1 -- so an undersized buffer is an
    out-of-bounds device write with no error and an NCDHW one is a full-rate
    kernel returning a scrambled answer.
    """
    problem = _problem("outbuf", 32, 16, (3, 4, 5), bias=True)
    ops = _ops(problem, seed=37)
    expected = _reference(problem, ops, "fwd").to(torch.bfloat16)
    y = torch.empty(
        problem.output_shape,
        device="cuda",
        dtype=torch.bfloat16,
        memory_format=torch.channels_last_3d,
    )
    got = _fwd(problem, ops, out=y)
    assert got.data_ptr() == y.data_ptr()
    assert reference.compare(y, expected).bitwise

    small = torch.empty(
        (1, 16, 2, 2, 2),
        device="cuda",
        dtype=torch.bfloat16,
        memory_format=torch.channels_last_3d,
    )
    with pytest.raises(ValueError, match="shape"):
        _fwd(problem, ops, out=small)
    ncdhw = torch.empty(problem.output_shape, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="channels_last_3d"):
        _fwd(problem, ops, out=ncdhw)
    wrong_dtype = torch.empty(
        problem.output_shape,
        device="cuda",
        dtype=torch.float32,
        memory_format=torch.channels_last_3d,
    )
    with pytest.raises(ValueError, match="dtype"):
        _fwd(problem, ops, out=wrong_dtype)


@requires_gpu
def test_an_illegal_tap_block_is_refused_rather_than_run():
    """A ``TAP_BLOCK`` that does not divide the tap count.

    The kernel's ``pid % (taps // TAP_BLOCK)`` would then address a tap group
    that runs off the end of the weight -- a wrong answer, not a fault, because
    the offsets stay inside the allocation for small kernels.  Refused at the
    entry point, loudly, since it can only arrive through an explicit
    ``config=``.
    """
    problem = _problem("tb", 32, 32, (3, 4, 5), (3, 3, 3))
    ops = _ops(problem, seed=41)
    bad = TransposedConfig(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, TAP_BLOCK=2)
    with pytest.raises(ValueError, match="TAP_BLOCK"):
        _fwd(problem, ops, config=bad)
    # And an outright illegal MFMA config is refused by the inherited rules.
    with pytest.raises(ValueError, match="nonkdim"):
        _fwd(
            problem,
            ops,
            config=TransposedConfig(
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, matrix_instr_nonkdim=64
            ),
        )


@requires_gpu
def test_ncdhw_input_is_converted_rather_than_misread():
    """A plain-contiguous input is relayouted, not read as if it were NDHWC.

    The addressing assumes a channel stride of 1.  Reading an NCDHW tensor with
    it would produce a full-rate kernel and a scrambled answer, which is why the
    entry point calls ``contiguous(memory_format=...)`` rather than asserting.
    """
    problem = _problem("ncdhw", 32, 16, (3, 4, 5), bias=True)
    ops = _ops(problem, seed=43)
    expected = _reference(problem, ops, "fwd").to(torch.bfloat16)
    plain = ops["input"].contiguous()
    assert not plain.is_contiguous(memory_format=torch.channels_last_3d)
    got = conv_transpose3d_forward(plain, ops["weight"], ops["bias"], problem.stride)
    assert got.is_contiguous(memory_format=torch.channels_last_3d)
    assert reference.compare(got, expected).bitwise


@requires_gpu
def test_output_matches_torchs_shape_and_layout():
    """Shape and memory format against ``F.conv_transpose3d``, at every kernel."""
    for k in ((2, 2, 2), (3, 3, 3), (1, 2, 4)):
        problem = _problem("shape", 32, 16, (3, 4, 5), k, bias=True)
        ops = _ops(problem, seed=47)
        got = _fwd(problem, ops)
        want = F.conv_transpose3d(ops["input"], ops["weight"], ops["bias"], stride=k)
        assert got.shape == want.shape, k
        assert got.is_contiguous(memory_format=torch.channels_last_3d)
        assert tuple(got.shape[2:]) == problem.out_spatial


@requires_gpu
def test_no_worse_than_miopen():
    """Error against fp64, held to the incumbent's own error where possible.

    ``assert_close``'s policy, unchanged and not reinvented: it once failed on
    MIOpen's *transposed* backward-weight, and the resolution was that the
    tolerance was wrong -- it charged the final store like an accumulation.
    ``roundings`` is 2 for MIOpen's backward-weight because that direction
    reduces with atomics and disagrees with itself bitwise between two calls.
    """
    for problem in [
        _problem("mi", 64, 32, (4, 5, 6), bias=True),
        _problem("mi3", 32, 32, (3, 4, 5), (3, 3, 3)),
    ]:
        for direction in ("fwd", "bwd-data", "bwd-weight"):
            ops = reference.make_inputs(problem, seed=53)
            expected = _reference(problem, ops, direction)
            incumbent = reference.compare(
                reference.incumbent(problem, ops, direction), expected
            )
            actual = {"fwd": _fwd, "bwd-data": _bwd_data, "bwd-weight": _bwd_weight}[
                direction
            ](problem, ops)
            reference.assert_close(
                actual, expected, problem, direction, incumbent_error=incumbent
            )


@requires_gpu
def test_repeated_calls_are_bitwise_reproducible():
    """The whole operator, run twice, must agree bitwise in all three directions.

    Backward-weight is the one at risk: it is ``conv3d_backward_weight``, whose
    deterministic split-K path is the default and whose atomic path is not
    reproducible.  Nothing here asks for the atomic path, and this test is what
    says so.
    """
    problem = _problem("repro", 64, 32, (4, 5, 6), bias=True)
    ops = reference.make_inputs(problem, seed=59)
    for run in (_fwd, _bwd_data, _bwd_weight):
        first = run(problem, ops)
        for _ in range(3):
            assert torch.equal(first, run(problem, ops))


@requires_gpu
def test_grad_weight_buffer_has_the_transposed_shape():
    """``(Cin, Cout, k, k, k)``, channels-last -- the parameter's own layout.

    The ordinary ``grad_weight_empty`` allocates ``(Cout, Cin, ...)``.  Passing
    that here is a correctly-strided buffer of the wrong shape, which the
    reduction's ``out=`` check catches only because it compares the shape
    explicitly -- none of the five channels-last strides depends on the first
    dimension.
    """
    from triton_conv3d.reduce_gemm import grad_weight_empty

    gw = grad_transposed_weight_empty(
        128, 64, (2, 2, 2), dtype=torch.bfloat16, device="cuda"
    )
    assert tuple(gw.shape) == (128, 64, 2, 2, 2)
    assert gw.is_contiguous(memory_format=torch.channels_last_3d)

    problem = _problem("gw", 32, 16, (3, 4, 5))
    ops = _ops(problem, seed=61, direction="bwd-weight")
    out = grad_transposed_weight_empty(
        32, 16, (2, 2, 2), dtype=torch.bfloat16, device="cuda"
    )
    got = _bwd_weight(problem, ops, out=out)
    assert got.data_ptr() == out.data_ptr()
    expected = _reference(problem, ops, "bwd-weight").to(torch.bfloat16)
    assert reference.compare(got, expected).bitwise

    wrong = grad_weight_empty(32, 16, (2, 2, 2), dtype=torch.bfloat16, device="cuda")
    assert tuple(wrong.shape) == (32, 16, 2, 2, 2)
    other = _problem("gw2", 16, 32, (3, 4, 5))
    ops2 = _ops(other, seed=61, direction="bwd-weight")
    with pytest.raises(ValueError, match="shape"):
        _bwd_weight(other, ops2, out=wrong)


@requires_gpu
def test_unsupported_calls_raise_rather_than_return_garbage():
    """Each entry point re-asks its own gate and refuses, never guesses."""
    problem = _problem("raise", 32, 16, (3, 4, 5))
    ops = _ops(problem, seed=67)
    with pytest.raises(NotImplementedError):
        conv_transpose3d_forward(ops["input"], ops["weight"], None, 3)
    with pytest.raises(NotImplementedError):
        conv_transpose3d_backward_data(
            ops["grad_output"], ops["weight"], problem.input_shape, 3
        )
    with pytest.raises(NotImplementedError):
        conv_transpose3d_backward_weight(
            ops["input"], problem.weight_shape, ops["grad_output"], 3
        )
    # And a padding, which is the one a caller is most likely to pass by habit.
    with pytest.raises(NotImplementedError):
        conv_transpose3d_forward(ops["input"], ops["weight"], None, problem.stride, 1)


@requires_gpu
def test_fp32_accumulates_in_fp32():
    """``more_determinism`` runs the model in fp32 and it has to really be fp32.

    The backend's default ``input_precision`` splits an fp32 dot into
    reduced-precision pieces, which is a ~10-bit mantissa and passes every
    tolerance this package has.  Only a bitwise test over a long reduction sees
    it, so the reduction here is long enough to matter.
    """
    problem = _problem("fp32acc", 512, 64, (2, 3, 4), dtype="fp32")
    ops = _ops(problem, seed=71)
    expected = _reference(problem, ops, "fwd")
    assert reference.is_exactly_representable(expected, torch.float32)
    actual = _fwd(problem, ops)
    assert actual.dtype is torch.float32
    assert reference.compare(actual, expected.to(torch.float32)).bitwise


@requires_gpu
def test_bias_is_per_channel_and_not_per_column():
    """One bias value per output channel, shared by all ``taps`` sub-lattices.

    In the kernel the bias is indexed by ``offs_n`` and not by the column, which
    is the difference between a bias and a per-tap offset.  Indexing it by the
    column would read ``TAP_BLOCK * Cout`` values from a ``Cout``-long tensor --
    past the end for every tap but the first, and wrong even where it is in
    bounds.  Checked by making the bias the only nonzero operand, so the answer
    *is* the bias broadcast over the upsampled volume.
    """
    problem = _problem("bias", 64, 48, (3, 4, 5), bias=True)
    ops = _ops(problem, seed=73)
    ops = {
        **ops,
        "weight": torch.zeros_like(ops["weight"]),
        "bias": torch.arange(1, 49, device="cuda", dtype=torch.bfloat16),
    }
    got = _fwd(problem, ops)
    want = ops["bias"].view(1, 48, 1, 1, 1).expand(got.shape)
    assert torch.equal(got, want.to(got.dtype))
