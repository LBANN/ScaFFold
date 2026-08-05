# SPDX-License-Identifier: (Apache-2.0)
"""Tests for the forward gather-GEMM convolution.

The organising idea is that a convolution kernel fails by *reading the wrong
voxel*, and a wrong voxel holds a plausible number.  A tolerance-based test
waves that through: swap a tap, drop a boundary compare, or transpose two
spatial strides and the result is still smooth, still the right magnitude, and
still passes ``allclose``.  So the primary standard here is **bitwise**, made
attainable by drawing operands from ``{-1, 0, 1}`` -- every product is exact and
every partial sum is a small integer, so the reference and the kernel must agree
exactly or the kernel is wrong.  :func:`test_bitwise_standard_rejects_a_shifted_gather`
exists to prove that standard has teeth rather than being vacuously satisfied.

The tolerance-based tests are still here, but as the *second* line: they are the
ones that hold at real magnitudes and real reduction lengths, where exactness is
not available.

Everything that needs a GPU is skipped without one; the configuration-legality
tests are pure Python and always run, which matters because an illegal MFMA
config on gfx942 does not raise -- it silently emits no matrix instructions and
returns correct results at a fraction of the speed.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_conv3d import reference
from triton_conv3d.gather_gemm import (
    ConvConfig,
    candidate_configs,
    conv3d_forward,
    default_config,
    is_supported,
    is_supported_all,
    to_rsck,
)
from triton_conv3d.shapes import ConvProblem, edge_cases, scaffold_corpus

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

#: The synthetic corpus, minus the transposed upsample (a later milestone).
EDGE = [p for p in edge_cases() if not p.transposed]

#: Real ScaFFold shapes small enough to test against an fp64 reference.  The hot
#: ones are 2 GiB activations; correctness does not need them, and every bug
#: these tests are looking for reproduces at 16^3.
#:
#: Selecting by ``volume * channels`` has a consequence worth stating, because
#: it is not obvious and it is what made the bitwise test on this list skip 11
#: of 11 for two milestones: the problems that survive the filter are the
#: *widest* ones -- 256->512 up to 1024->1024, K = 6912 to 27648 -- because
#: those are the ones ScaFFold runs at a small spatial extent.  So this list is
#: precisely the regime where the forward's reduction is longest, which is the
#: regime the exactness question is hardest in.  See
#: :func:`test_corpus_shapes_match_bitwise`.
CORPUS_SMALL = [
    p
    for p in scaffold_corpus()
    if not p.transposed and math.prod(p.spatial) * max(p.cin, p.cout) <= 1 << 22
]


def _ids(problems):
    return [p.name or p.label for p in problems]


def _run(problem: ConvProblem, ops: dict, **kwargs) -> torch.Tensor:
    return conv3d_forward(
        ops["input"],
        ops["weight"],
        ops["bias"],
        problem.stride,
        problem.padding,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Configuration legality -- no GPU needed, and the failure mode is silent
# ---------------------------------------------------------------------------


def test_validate_rejects_the_silent_mfma_failures():
    """Each of these produces a working kernel with zero MFMA instructions.

    Verified by negative control in M0: ``BLOCK_K=8`` at ``nonkdim=16`` and an
    illegal ``nonkdim=64`` both compile, run, and return correct results with the
    dot lowered to vector FMA.  Nothing raises and nothing warns, so a config
    generator that merely ranked them last would still feed meaningless entries
    into a best-of sweep -- which is why :meth:`ConvConfig.validate` refuses.
    """
    bf16 = torch.bfloat16
    assert ConvConfig(BLOCK_K=8, matrix_instr_nonkdim=16).validate(bf16)
    assert ConvConfig(matrix_instr_nonkdim=64).validate(bf16)
    assert ConvConfig(BLOCK_M=24, matrix_instr_nonkdim=16).validate(bf16)
    assert ConvConfig(BLOCK_M=64, BLOCK_N=16, num_warps=8).validate(bf16)
    assert ConvConfig(num_warps=3).validate(bf16)
    assert ConvConfig(num_stages=1).validate(bf16)
    assert ConvConfig().validate(torch.float64)
    # And the default is legal, or none of the above means anything.
    assert ConvConfig().validate(bf16) is None


def test_validate_rejects_a_group_m_that_faults_the_gpu():
    """``GROUP_M`` was the one config field ``validate`` did not look at.

    Its failure mode is not the silent FMA fallback the others have, which is
    why it is stated apart from them: the swizzle computes ``width = GROUP_M *
    grid_n`` and then ``pid // width``, so ``GROUP_M = 0`` divides by zero inside
    the kernel -- on gfx942 that is a garbage ``pid_m`` and a memory access
    fault, not a trap -- and ``GROUP_M = -3`` reaches the kernel just as far.

    The other half of the pin is that every *legal* value is accepted, including
    values that do not divide ``grid_m`` and values far larger than it; the
    swizzle is a bijection for all of them and rejecting them would cost tuning
    range for nothing.
    """
    bf16 = torch.bfloat16
    assert ConvConfig(GROUP_M=0).validate(bf16)
    assert ConvConfig(GROUP_M=-3).validate(bf16)
    for group_m in (1, 5, 6, 7, 8, 4096):
        assert ConvConfig(GROUP_M=group_m).validate(bf16) is None, group_m


def test_the_index_width_decision_covers_every_operand_including_the_weight():
    """The predicate behind ``INDEX_DTYPE``, checked without allocating 4 GiB.

    It used to be ``max(x.numel(), y.numel())``, and the weight's absence from it
    was an assumption ("weights are never large enough") that nothing enforced:
    at ``taps * Cin * Cout = 2.30e9`` the int32 offset wraps negative and the GPU
    faults, with ``is_supported`` returning ``True``.  Meta tensors carry a
    ``numel`` and no storage, so this end of the fix costs nothing to state.
    """
    from triton_conv3d.gather_gemm import _index_dtype

    small = torch.empty((1 << 20,), device="meta")
    huge = torch.empty((1 << 31,), device="meta")  # numel > 2**31 - 1 by one
    assert _index_dtype(small, small, small) == tl.int32
    assert _index_dtype(huge, small, small) == tl.int64
    assert _index_dtype(small, huge, small) == tl.int64
    assert _index_dtype(small, small, huge) == tl.int64


def test_block_k_constraint_follows_the_intrinsic_not_the_tile():
    """``BLOCK_K`` is constrained by the MFMA's reduction depth, which moves.

    bf16 on gfx942 has one intrinsic per shape: ``16x16x16`` at nonkdim 16 and
    ``32x32x8`` at 32.  So a ``BLOCK_K`` of 8 is legal at nonkdim 32 and illegal
    at 16 -- the constraint is not a property of the block size alone, and the
    briefing's claim that ``BLOCK_K=16`` rows get pruned at nonkdim 16 is simply
    wrong arithmetic (16 % 16 == 0).
    """
    ok32 = ConvConfig(
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=8, matrix_instr_nonkdim=32, num_warps=4, kpack=1
    )
    assert ok32.validate(torch.bfloat16) is None
    assert ConvConfig(
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=8, matrix_instr_nonkdim=16, num_warps=4
    ).validate(torch.bfloat16)
    assert (
        ConvConfig(BLOCK_K=16, matrix_instr_nonkdim=16, kpack=1).validate(
            torch.bfloat16
        )
        is None
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=["bf16", "fp16", "fp32"],
)
def test_default_config_fits_in_lds_in_every_dtype(dtype):
    """The block sizes were chosen against bf16; fp32 operands are twice the bytes.

    This is a regression test for a hole M2 fell into rather than a hypothetical:
    ``default_config`` returned ``128x128x128`` for ``Cin >= 512``, which is 64
    KiB in bf16 and 128 KiB in fp32, and the *shipped* configuration therefore
    raised ``OutOfResources`` on any wide fp32 convolution.  ``more_determinism``
    runs the model in fp32, so a real ScaFFold configuration reached it.

    An explicitly supplied ``config=`` is still allowed to overflow and still
    fails loudly; what must never overflow is the one the entry point picks by
    itself.
    """
    from triton_conv3d.gather_gemm import _LDS_BYTES

    for cin in (3, 6, 64, 128, 256, 512, 1024):
        for cout in (6, 64, 128, 256, 512, 1024):
            for m in (512, 4096, 2 << 20):
                cfg = default_config(m, cin, cout, dtype)
                assert cfg.validate(dtype) is None, f"{cin}->{cout} m={m}: {cfg}"
                assert cfg.lds_bytes(dtype) <= _LDS_BYTES, (
                    f"{cin}->{cout} m={m}: {cfg} needs {cfg.lds_bytes(dtype)} B of LDS"
                )


@pytest.mark.parametrize("problem", EDGE + CORPUS_SMALL, ids=_ids(EDGE + CORPUS_SMALL))
def test_default_config_is_legal_for_every_shape(problem: ConvProblem):
    """The heuristic must never hand back a config that loses the matrix core.

    It is the config used when no tuned entry exists, which is most of the time,
    and it derives block sizes from the shape -- so the tiny synthetic problems
    are exactly where it can round itself into an illegal combination.
    """
    dtype = reference.torch_dtype(problem)
    m = problem.n * math.prod(problem.out_spatial)
    cfg = default_config(m, problem.cin, problem.cout, dtype)
    assert cfg.validate(dtype) is None, (
        f"{problem.label}: {cfg} -> {cfg.validate(dtype)}"
    )


@pytest.mark.parametrize("problem", CORPUS_SMALL[:6], ids=_ids(CORPUS_SMALL[:6]))
def test_every_candidate_config_is_legal(problem: ConvProblem):
    """The sweep must not contain a config that cannot reach the matrix core.

    Otherwise the sweep's *reported* winner could be an FMA kernel that happened
    to beat the others, and the whole tuning surface would be measuring the
    wrong thing.
    """
    dtype = reference.torch_dtype(problem)
    m = problem.n * math.prod(problem.out_spatial)
    cfgs = candidate_configs(m, problem.cin, problem.cout, dtype)
    assert cfgs
    for cfg in cfgs:
        assert cfg.validate(dtype) is None, f"{cfg}: {cfg.validate(dtype)}"


# ---------------------------------------------------------------------------
# Support predicate
# ---------------------------------------------------------------------------


def test_is_supported_declines_what_the_kernel_cannot_do():
    """A false positive returns a wrong answer; a false negative costs speed.

    The caller's fallback is MIOpen, which is correct everywhere, so the
    predicate is deliberately asymmetric and this test pins that asymmetry.
    """
    x = torch.empty((1, 8, 4, 4, 4), device="meta", dtype=torch.bfloat16)
    w = torch.empty((8, 8, 3, 3, 3), device="meta", dtype=torch.bfloat16)
    # Meta tensors are not on a device, so the real predicate rejects them; the
    # checks below are about everything *except* device placement.
    assert not is_supported(x, w, padding=1)

    if not torch.cuda.is_available():
        pytest.skip("the remaining branches need a real device")
    x = torch.empty((1, 8, 4, 4, 4), device="cuda", dtype=torch.bfloat16)
    w = torch.empty((8, 8, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    assert is_supported(x, w, padding=1)
    assert not is_supported(x, w, padding=1, groups=2)
    assert not is_supported(x, w.float(), padding=1)
    assert not is_supported(
        x, torch.empty((8, 4, 3, 3, 3), device="cuda", dtype=torch.bfloat16), padding=1
    )
    # A kernel wider than the padded input has no output voxels at all, which
    # the M-unravel cannot express.
    tiny = torch.empty((1, 8, 1, 4, 4), device="cuda", dtype=torch.bfloat16)
    assert not is_supported(tiny, w, padding=0)
    assert is_supported(tiny, w, padding=1)


@requires_gpu
def test_is_supported_never_raises_on_an_argument_it_cannot_parse():
    """A gate that throws is not a gate.

    This predicate is the first rung of a Triton -> MIOpen ladder, so a caller
    asking "will you serve this?" about a ``padding`` it holds in a variable must
    get an answer.  ``_triple`` raises ``TypeError`` for anything neither ``int``
    nor iterable, and only ``ValueError`` was caught -- so ``padding=None`` and
    ``padding=1.5`` came back out of the *predicate* as exceptions while
    ``padding=(1, 1)`` and ``padding='same'`` correctly returned ``False``.
    """
    x = torch.empty((1, 8, 4, 4, 4), device="cuda", dtype=torch.bfloat16)
    w = torch.empty((8, 8, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    assert is_supported(x, w, padding=1)
    for bad in (None, 1.5, "same", (1, 1), [1, 2, 3, 4], object()):
        assert not is_supported(x, w, padding=bad), bad
        assert not is_supported(x, w, stride=bad), bad
        assert not is_supported(x, w, dilation=bad), bad


@requires_gpu
def test_is_supported_declines_a_bias_torch_itself_rejects():
    """The kernel masks the bias load against ``Cout`` and assumes stride 1.

    Neither of those is a property of the bias, so both failures are silent: a
    short bias reads whatever is in memory past its end (``nan`` if you are
    lucky, a plausible finite number if you are not), and a stride-2 view of the
    right length applies ``[0,1,2,...]`` where the caller passed ``[0,2,4,...]``.
    ``torch.conv3d`` refuses the first outright; this predicate now refuses both.
    """
    bf16 = torch.bfloat16
    x = torch.empty((1, 8, 4, 5, 6), device="cuda", dtype=bf16)
    w = torch.empty((32, 8, 3, 3, 3), device="cuda", dtype=bf16)
    bias = torch.empty(32, device="cuda", dtype=bf16)
    assert is_supported(x, w, bias, padding=1)

    assert not is_supported(x, w, bias[:4], padding=1)  # too short
    assert not is_supported(
        x, w, torch.empty(64, device="cuda", dtype=bf16)[::2], padding=1
    )
    assert not is_supported(x, w, bias.float(), padding=1)
    assert not is_supported(x, w, bias.cpu(), padding=1)
    assert not is_supported(x, w, bias.view(1, 32), padding=1)
    # And the entry point declines rather than running on it.
    with pytest.raises(NotImplementedError):
        conv3d_forward(x, w, bias[:4], padding=1)


@requires_gpu
def test_the_forward_gate_alone_is_a_trap_for_a_caller_that_differentiates():
    """``stride=2``: served forward, served backward-weight, refused backward-data.

    The disagreement is real and each side of it is deliberate -- the forward's
    M-unravel simply steps by ``s``, backward-weight is indexed by the *output*
    voxel so a stride is three extra multiplies, and backward-data has no kernel
    of its own and is the forward contraction on a flipped weight, which is only
    the right contraction at unit stride.  What was wrong was that nothing said
    so: a training caller who asked :func:`is_supported`, got ``True`` and built
    a graph node found out at ``backward()``, where its own MIOpen fallback is no
    longer reachable because the node is already in the graph.

    So this pins both halves: the trap still exists at the direction gates (they
    describe their own kernels and must keep doing so), and
    :func:`is_supported_all` is the one question that closes it.
    """
    from triton_conv3d.bwd_data import conv3d_backward_data, is_supported_bwd_data
    from triton_conv3d.reduce_gemm import is_supported_bwd_weight

    bf16 = torch.bfloat16
    x = torch.empty((1, 8, 8, 8, 8), device="cuda", dtype=bf16)
    w = torch.empty((16, 8, 3, 3, 3), device="cuda", dtype=bf16)
    gy = torch.empty((1, 16, 4, 4, 4), device="cuda", dtype=bf16)
    args = dict(stride=2, padding=1)

    assert is_supported(x, w, **args)
    assert is_supported_bwd_weight(x, w.shape, gy, **args)
    assert not is_supported_bwd_data(gy, w, x.shape, **args)
    assert not is_supported_all(x, w, **args)

    # The trap itself, run: the forward serves the call and the gradient this
    # very forward produces cannot be turned back into an input gradient.
    y = conv3d_forward(
        x.contiguous(memory_format=torch.channels_last_3d),
        w.contiguous(memory_format=torch.channels_last_3d),
        **args,
    )
    assert tuple(y.shape) == (1, 16, 4, 4, 4)
    with pytest.raises(NotImplementedError):
        conv3d_backward_data(
            y.contiguous(memory_format=torch.channels_last_3d),
            w.contiguous(memory_format=torch.channels_last_3d),
            x.shape,
            **args,
        )

    # And the same problem at unit stride, where all three do agree, is not
    # collateral damage: the combined gate must still say yes.
    assert is_supported_all(x, w, padding=1)


@requires_gpu
def test_is_supported_all_is_exactly_the_three_gates_conjoined():
    """The combined gate against the conjunction it stands for, term by term.

    Two things could rot here and neither would fail loudly.  The gradient is
    passed to the two backward predicates as a metadata-only stand-in -- a
    one-element allocation expanded to the output shape -- which is sound only
    while those predicates read metadata and nothing else, so it is compared
    against the answer a *real* gradient gets.  And the output shape is computed
    here rather than by the caller, so a wrong one would be a gate answering
    about a different problem than the one it was asked about.
    """
    from triton_conv3d.bwd_data import is_supported_bwd_data
    from triton_conv3d.reduce_gemm import is_supported_bwd_weight

    bf16 = torch.bfloat16
    cases = [
        # (x shape, w shape, kwargs)
        ((1, 8, 8, 8, 8), (16, 8, 3, 3, 3), dict(padding=1)),  # all yes
        ((1, 8, 8, 8, 8), (16, 8, 3, 3, 3), dict(stride=2, padding=1)),  # bwd-data no
        ((1, 8, 8, 8, 8), (16, 8, 3, 3, 3), dict(padding=1, groups=2)),  # fwd no
        ((1, 8, 8, 8, 8), (16, 8, 3, 3, 3), dict(padding=0)),  # all yes
        ((1, 8, 8, 8, 8), (16, 8, 1, 1, 1), dict(padding=0)),  # k=1
        ((1, 8, 8, 8, 8), (16, 8, 3, 3, 3), dict(padding=2)),  # p > dil*(k-1)
        ((1, 8, 1, 8, 8), (16, 8, 3, 3, 3), dict(padding=1)),  # thin D
        ((1, 8, 8, 8, 8), (16, 8, 3, 3, 3), dict(dilation=2, padding=2)),
    ]
    for x_shape, w_shape, kwargs in cases:
        x = torch.empty(x_shape, device="cuda", dtype=bf16)
        w = torch.empty(w_shape, device="cuda", dtype=bf16)
        s = kwargs.get("stride", 1)
        p = kwargs.get("padding", 0)
        d = kwargs.get("dilation", 1)
        out = tuple(
            (x_shape[2 + i] + 2 * p - d * (w_shape[2 + i] - 1) - 1) // s + 1
            for i in range(3)
        )
        gy = torch.empty((x_shape[0], w_shape[0]) + out, device="cuda", dtype=bf16)
        expected = (
            is_supported(x, w, **kwargs)
            and is_supported_bwd_data(gy, w, x.shape, **kwargs)
            and is_supported_bwd_weight(x, w.shape, gy, **kwargs)
        )
        assert is_supported_all(x, w, **kwargs) is expected, (x_shape, w_shape, kwargs)


@requires_gpu
def test_is_supported_all_never_raises_on_an_argument_it_cannot_parse():
    """Total, for the same reason :func:`is_supported` is: it is a gate.

    The forward's predicate runs first and refuses everything unparsable, so the
    output-shape arithmetic below it is never reached with an argument that would
    make it throw -- but the caller's contract is "you get an answer", and that
    has to be checked and not argued.
    """
    x = torch.empty((1, 8, 4, 4, 4), device="cuda", dtype=torch.bfloat16)
    w = torch.empty((8, 8, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    assert is_supported_all(x, w, padding=1)
    for bad in (None, 1.5, "same", (1, 1), [1, 2, 3, 4], object()):
        assert not is_supported_all(x, w, padding=bad), bad
        assert not is_supported_all(x, w, stride=bad), bad
        assert not is_supported_all(x, w, dilation=bad), bad


@requires_gpu
def test_is_supported_declines_degenerate_extents():
    """Three shapes where the kernel returned something ``torch.conv3d`` does not.

    Each clears the "every output voxel must exist" test and then diverges, which
    is the asymmetry the predicate exists to prevent -- the MIOpen fallback would
    have raised on all three and the Triton path silently did not.

    ``N = 0`` is deliberately *not* in the rejection list: it agrees with torch,
    both in the shape it returns and in doing no work to return it, so declining
    it would be a false negative with nothing behind it.
    """
    bf16 = torch.bfloat16
    x = torch.empty((1, 8, 4, 5, 6), device="cuda", dtype=bf16)
    w = torch.empty((16, 8, 3, 3, 3), device="cuda", dtype=bf16)

    # A zero-length spatial axis: returned a volume of pure padding, where torch
    # raises "Only zero batch or zero channel inputs are supported".
    assert not is_supported(
        torch.empty((1, 8, 0, 5, 6), device="cuda", dtype=bf16), w, padding=2
    )
    # A zero-size kernel: ``(in + 2p - d(k-1) - 1)//s + 1`` gains one at k=0, so
    # the returned output was *larger* than the input.
    assert not is_supported(
        x, torch.empty((16, 8, 0, 0, 0), device="cuda", dtype=bf16), padding=0
    )
    # Cin = 0: returned Cout channels of zeros where torch returns a tensor with
    # no channels at all -- a different shape, not a different value.
    assert not is_supported(
        torch.empty((1, 0, 4, 5, 6), device="cuda", dtype=bf16),
        torch.empty((16, 0, 3, 3, 3), device="cuda", dtype=bf16),
        padding=1,
    )

    empty_batch = torch.empty((0, 8, 4, 5, 6), device="cuda", dtype=bf16).contiguous(
        memory_format=torch.channels_last_3d
    )
    assert is_supported(empty_batch, w, padding=1)
    assert tuple(conv3d_forward(empty_batch, w, padding=1).shape) == (0, 16, 4, 5, 6)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_is_supported_declines_operands_on_different_devices():
    """Both operands on *a* GPU is not the same as both on the *same* GPU.

    Triton launches on the current device and dereferences the foreign pointer
    regardless.  ScaFFold runs four ranks per node, and with peer access enabled
    that reads another rank's activations instead of faulting -- a wrong answer
    with no symptom.  Skipped, not absent, on a single-GPU box.
    """
    bf16 = torch.bfloat16
    x = torch.empty((1, 8, 4, 4, 4), device="cuda:0", dtype=bf16)
    w = torch.empty((8, 8, 3, 3, 3), device="cuda:0", dtype=bf16)
    bias = torch.empty(8, device="cuda:0", dtype=bf16)
    assert is_supported(x, w, bias, padding=1)
    assert not is_supported(x, w.to("cuda:1"), padding=1)
    assert not is_supported(x, w, bias.to("cuda:1"), padding=1)


# ---------------------------------------------------------------------------
# Correctness: the bitwise standard
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("problem", EDGE, ids=_ids(EDGE))
def test_exact_operands_match_bitwise(problem: ConvProblem):
    """The strictest standard, on the problems chosen to break addressing.

    The corpus covers channel counts that are not multiples of any plausible
    ``BLOCK_K``, prime and unit spatial extents, anisotropic volumes and kernels,
    ``N > 1``, and a volume smaller than the kernel in every axis -- where every
    tap is masked somewhere and the boundary predicate is the whole computation.
    """
    ops = reference.make_inputs(problem, seed=3, exact=True)
    expected = reference.reference(problem, ops, "fwd")
    dtype = reference.torch_dtype(problem)
    if not reference.is_exactly_representable(expected, dtype):
        pytest.skip("realized magnitudes exceed the mantissa in this dtype")
    actual = _run(problem, ops)
    report = reference.compare(actual, expected.to(dtype))
    assert report.bitwise, f"{problem.label}: {report}"


@requires_gpu
def test_bitwise_standard_rejects_a_shifted_gather():
    """Prove the bitwise test has teeth: a one-voxel shift must fail it.

    Without this, ``test_exact_operands_match_bitwise`` could be passing because
    ``{-1,0,1}`` operands happen to make everything agree -- a vacuous test, and
    this project has already shipped two of those in the GroupNorm suite.  So
    compare the kernel's output against a reference computed for a *deliberately
    wrong* padding and require a mismatch.
    """
    problem = ConvProblem("shift", 16, 16, (6, 6, 6))
    ops = reference.make_inputs(problem, seed=11, exact=True)
    actual = _run(problem, ops)
    correct = reference.reference(problem, ops, "fwd").to(torch.bfloat16)
    assert reference.compare(actual, correct).bitwise

    # Same problem, but the reference gathers from one voxel further along W.
    shifted = torch.roll(ops["input"], shifts=1, dims=-1)
    wrong = reference.reference(problem, {**ops, "input": shifted}, "fwd").to(
        torch.bfloat16
    )
    assert not reference.compare(actual, wrong).bitwise, (
        "a one-voxel shift of the input produced a bitwise-identical result; "
        "the comparison is not discriminating"
    )


@requires_gpu
@pytest.mark.parametrize("problem", CORPUS_SMALL, ids=_ids(CORPUS_SMALL))
def test_corpus_shapes_match_bitwise(problem: ConvProblem):
    """Same standard, on the shapes ScaFFold actually runs.

    The synthetic cases break addressing; these check that nothing about the
    real channel widths -- 64 through 1024, all multiples of 256 bytes in bf16
    and so all candidates for the stride hazard -- changes the answer.

    This test used to skip 11 of 11 cases, every run, for two milestones, and
    the mechanism is worth stating because the obvious fix does not work here.
    A dense ``{-1,0,1}`` draw is exactly representable only while the realized
    sums stay inside the mantissa, and bf16 holds integers only to 256; the
    forward reduces over ``Cin * taps``, which is 27 648 terms at ``Cin = 1024``
    and gives sums running to several hundred.  ``test_bwd_data.py`` and
    ``test_bwd_weight.py`` hit the same wall and got out of it by restating each
    channel pair at ``6x7x8`` -- their reductions run over ``Cout * taps`` and
    over the *output volume*, so a smaller volume shortens them.  The forward's
    does not depend on the volume at all, so no shape substitution can help it:
    the only lever is the operands.

    So the activations are thinned to :func:`reference.exact_density` -- about
    1-4% here, which leaves ~300 live terms per output element and a realized
    maximum of 58-71 against the limit of 256 -- while the shape, the channel
    widths and the weight stay exactly as they are.  It is the gather that is
    under test, not the arithmetic.  Two things then have to be asserted rather
    than assumed, because a thinned draw is exactly how one would accidentally
    build a test that compares zeros against zeros: that the answer is mostly
    nonzero, and that the comparison still rejects a one-voxel shift.

    And it no longer skips.  If the draw is ever not exact, that is a fact worth
    failing on rather than stepping around -- the skip is what hid the hole.
    """
    density = reference.exact_density(problem, "fwd")
    ops = reference.make_inputs(problem, seed=5, exact=True, density=density)
    expected = reference.reference(problem, ops, "fwd")
    dtype = reference.torch_dtype(problem)
    assert reference.is_exactly_representable(expected, dtype), (
        f"{problem.label}: the thinned draw at density {density:.4g} still "
        f"realizes |max| = {expected.abs().max().item():g}, which "
        f"{problem.dtype} cannot hold exactly"
    )
    nonzero = (expected != 0).to(torch.float64).mean().item()
    assert nonzero > 0.5, (
        f"{problem.label}: only {nonzero:.1%} of the reference is nonzero; the "
        "thinning has gone far enough to make the comparison vacuous"
    )

    actual = _run(problem, ops)
    assert reference.compare(actual, expected.to(dtype)).bitwise

    # The negative control, per case rather than once: run the kernel over an
    # input shifted by one voxel and require the *same* comparison to reject it.
    # A tolerance would wave that through, and so would a draw thinned until
    # everything it touches is zero -- this is what says the passing assertion
    # above is coverage rather than a coincidence, at this width.
    shifted = _run(problem, {**ops, "input": torch.roll(ops["input"], 1, dims=-1)})
    assert not reference.compare(shifted, expected.to(dtype)).bitwise, (
        f"{problem.label}: a one-voxel shift of the input produced a "
        "bitwise-identical result; the comparison is not discriminating"
    )


def test_the_bitwise_corpus_is_not_entirely_skipped():
    """A regression guard on this file, not on the kernel.

    ``test_bwd_data.py`` and ``test_bwd_weight.py`` both carry a guard of this
    name because this suite has already once reported "89 passed" while
    skipping 100% of its real-shape cases.  The forward had no such guard, which
    is how it went two milestones with 11 of 11 skipping and nothing saying so.

    The shape of the guard differs from theirs, because the fix here differs:
    :func:`test_corpus_shapes_match_bitwise` has no skip branch left, so what
    needs pinning is not "enough cases are representable" but the two ways the
    test could still stop meaning anything -- the parametrization collapsing to
    nothing or to only narrow shapes, and the thinning going so far that every
    output element is a sum of nothing.  Both are pure arithmetic, so this runs
    without a GPU, which is the other half of the point: a guard that skips with
    the thing it guards is not a guard.
    """
    assert len(CORPUS_SMALL) >= 8, CORPUS_SMALL
    # The widths are the reason this list exists.  A filter that quietly stopped
    # selecting the deep encoder problems would leave the forward tested bitwise
    # only at the synthetic sizes again.
    assert max(p.cin for p in CORPUS_SMALL) >= 1024
    assert max(p.cout for p in CORPUS_SMALL) >= 1024
    for problem in CORPUS_SMALL:
        k = problem.gemm_shape("fwd")[2]
        density = reference.exact_density(problem, "fwd")
        assert 0.0 < density <= 1.0, f"{problem.label}: density {density}"
        # Live terms per output element: the reduction that actually happens.
        # At 64 a wrong gather still has dozens of independent chances to show
        # up in every element; below it the draw would be approaching a test of
        # whether zero equals zero.
        assert density * k >= 64.0, (
            f"{problem.label}: only {density * k:.1f} of {k} terms contribute; "
            "the thinned draw is close to vacuous"
        )


@requires_gpu
@pytest.mark.parametrize("problem", EDGE, ids=_ids(EDGE))
def test_every_config_gives_the_same_answer(problem: ConvProblem):
    """Tiling must not be observable in the result.

    A boundary bug usually only shows up at one tile shape: a mask that is right
    when ``BLOCK_M`` divides ``OUT_W`` and wrong when it does not, or a ``BLOCK_K``
    remainder that is only exercised when ``Cin`` is not a multiple of the tile.
    Sweeping the whole candidate list against a bitwise reference tests the
    *tuning surface* rather than one point on it, which matters because the
    tuned table is free to pick any of them.

    Over all of ``EDGE`` rather than its first eight.  The eight are the channel
    and spatial oddities, and stopping there left ``batched`` (the only ``n > 1``
    case), ``kernel_aniso``, ``smaller_than_kernel``, ``unpadded``, ``pointwise``
    and both non-bf16 dtypes checked at the *default* config alone -- so a tiling
    bug that needed ``BLOCK_M=256`` with ``n > 1``, or fp32 at ``nonkdim=32``,
    had nowhere to show up.  fp32 and fp16 matter here in their own right: the
    dtype moves the MFMA intrinsic's reduction depth and therefore which
    ``BLOCK_K`` values are even legal.
    """
    ops = reference.make_inputs(problem, seed=2, exact=True)
    expected = reference.reference(problem, ops, "fwd")
    dtype = reference.torch_dtype(problem)
    if not reference.is_exactly_representable(expected, dtype):
        pytest.skip("realized magnitudes exceed the mantissa in this dtype")
    expected = expected.to(dtype)
    m = problem.n * math.prod(problem.out_spatial)
    # Plus the shipped default, which for a shape too small for any seed tile
    # (``Cout=6``) is the only candidate there is.
    cfgs = candidate_configs(m, problem.cin, problem.cout, dtype, group_ms=(6, 8))
    cfgs = list(
        dict.fromkeys(cfgs + [default_config(m, problem.cin, problem.cout, dtype)])
    )
    ran = 0
    for cfg in cfgs:
        try:
            actual = _run(problem, ops, config=cfg)
        except triton.runtime.errors.OutOfResources:
            # A tile whose operands do not fit in 64 KiB of LDS.  Unlike the
            # MFMA constraints this failure is *loud*: Triton refuses at compile
            # time and says so, so it needs no static guard -- the sweep skips
            # it and so does this test.
            continue
        ran += 1
        assert reference.compare(actual, expected).bitwise, f"{problem.label} {cfg}"
    assert ran, "no candidate configuration was runnable"


# ---------------------------------------------------------------------------
# Correctness: the tolerance standards
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("problem", EDGE + CORPUS_SMALL, ids=_ids(EDGE + CORPUS_SMALL))
def test_no_worse_than_miopen(problem: ConvProblem):
    """The honest bar for a replacement: not better than MIOpen, but not worse.

    Held against an fp64 reference with MIOpen measured on the same operands, so
    the bar adapts to shape and reduction length instead of being a constant
    somebody picked.  Random operands rather than ``{-1,0,1}`` because this is
    the standard that has to hold at realistic magnitudes, where the reduction
    genuinely does lose bits.
    """
    ops = reference.make_inputs(problem, seed=17)
    expected = reference.reference(problem, ops, "fwd")
    incumbent_err = reference.compare(
        reference.incumbent(problem, ops, "fwd"), expected
    )
    actual = _run(problem, ops)
    reference.assert_close(
        actual, expected, problem, "fwd", incumbent_error=incumbent_err
    )


@requires_gpu
def test_fp32_accumulates_in_fp32():
    """fp32 in, fp32 out, and no silent demotion to a reduced-precision dot.

    ``more_determinism`` runs the model in fp32, and on this backend it is not
    obvious whether ``tl.dot`` on fp32 operands uses the exact ``f32`` MFMA or a
    tf32-style split.  A tf32 dot would still pass a bf16-sized tolerance, so the
    check is against fp64 with an fp32-sized bound.
    """
    problem = ConvProblem("fp32", 48, 32, (7, 9, 5), dtype="fp32")
    ops = reference.make_inputs(problem, seed=23)
    expected = reference.reference(problem, ops, "fwd")
    actual = _run(problem, ops)
    assert actual.dtype is torch.float32
    report = reference.compare(actual, expected)
    # tf32 keeps 10 explicit mantissa bits; fp32 keeps 23.  A bound between the
    # two separates them, which a dtype-generic tolerance would not.
    peak = expected.abs().max().item()
    assert report.max_abs < 1e-4 * peak, f"looks like a reduced-precision dot: {report}"


# ---------------------------------------------------------------------------
# Entry-point behaviour
# ---------------------------------------------------------------------------


@requires_gpu
def test_bias_is_added_once_and_broadcast_over_channels():
    problem = ConvProblem(
        "bias", 32, 24, (5, 6, 7), (1, 1, 1), padding=(0, 0, 0), bias=True
    )
    ops = reference.make_inputs(problem, seed=31, exact=True)
    with_bias = _run(problem, ops)
    without = conv3d_forward(
        ops["input"], ops["weight"], None, problem.stride, problem.padding
    )
    delta = with_bias.float() - without.float()
    # The difference must be exactly the bias, in every voxel.
    expected = ops["bias"].float().view(1, -1, 1, 1, 1).expand_as(delta)
    assert torch.equal(delta, expected)


@requires_gpu
def test_ncdhw_input_is_converted_rather_than_misread():
    """A contiguous NCDHW input must give the same answer, not a transposed one.

    The addressing assumes ``stride_xc == 1``.  Silently reading an NCDHW tensor
    with NDHWC strides produces a full-rate kernel and a completely wrong result,
    so the entry point converts; this pins that it converts rather than assumes.
    """
    problem = ConvProblem("layout", 24, 16, (5, 6, 7))
    ops = reference.make_inputs(problem, seed=41, exact=True)
    ndhwc = _run(problem, ops)
    nc = {k: (v.contiguous() if torch.is_tensor(v) else v) for k, v in ops.items()}
    assert nc["input"].stride(1) != 1
    ncdhw = _run(problem, nc)
    assert torch.equal(ndhwc, ncdhw)


@requires_gpu
def test_out_buffer_is_written_in_place_and_is_validated():
    """``out=`` had no check of any kind, and nothing downstream can catch one.

    The grid is sized from the problem rather than from the buffer, so an
    undersized ``out=`` is an out-of-bounds *device write* -- 1920 elements into
    a 128-element allocation, observed, with no error and no fault, surviving
    only because the allocator slab happened to be bigger.  An NCDHW buffer is
    the other half: the store addressing writes NDHWC strides into it and returns
    a scrambled answer at full speed.

    Handing a preallocated gradient buffer to the backward is precisely what the
    ``nn.Module`` adapter will do, so this is the parameter that most needs the
    check and had none.
    """
    problem = ConvProblem("out", 16, 24, (4, 5, 6))
    ops = reference.make_inputs(problem, seed=67, exact=True)
    expected = _run(problem, ops)

    buf = torch.empty_like(expected)
    got = _run(problem, ops, out=buf)
    assert got.data_ptr() == buf.data_ptr(), "out= was allocated over, not written"
    assert torch.equal(got, expected)

    shape = tuple(expected.shape)
    bf16 = torch.bfloat16
    # Undersized, right layout: the write ran off the end.
    with pytest.raises(ValueError):
        _run(problem, ops, out=torch.empty((1, 24, 2, 2, 2), device="cuda", dtype=bf16))
    # Right shape, NCDHW: read with NDHWC strides.
    with pytest.raises(ValueError):
        _run(problem, ops, out=torch.empty(shape, device="cuda", dtype=bf16))
    with pytest.raises(ValueError):
        _run(problem, ops, out=torch.empty_like(expected, dtype=torch.float32))
    with pytest.raises(ValueError):
        _run(problem, ops, out=torch.empty(shape, dtype=bf16))  # on the CPU


@requires_gpu
def test_the_output_is_allocated_directly_in_channels_last():
    """One allocation in the final layout, not an NCDHW one plus a full copy.

    ``torch.empty(shape).contiguous(memory_format=channels_last_3d)`` is a
    correct way to spell an expensive thing: it allocates NCDHW and then copies
    the whole tensor, which measured **2.82 ms against 0.012 ms** on a 256 MiB
    output -- 235x, on a path a training step takes about 19 times.  The copy is
    invisible in the result, so what pins it is the peak allocation: the wrong
    form needs two output-sized buffers live at once, the right form needs one.
    """
    x = torch.randn(
        (1, 64, 64, 64, 64), device="cuda", dtype=torch.bfloat16
    ).contiguous(memory_format=torch.channels_last_3d)
    w = torch.randn((64, 64, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    wr = to_rsck(w)
    conv3d_forward(x, w, padding=1, weight_rsck=wr)  # warm the JIT out of the way

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y = conv3d_forward(x, w, padding=1, weight_rsck=wr)
    peak = torch.cuda.max_memory_allocated() - base
    want = y.numel() * y.element_size()
    assert peak < 1.5 * want, (
        f"the call peaked at {peak} B for a {want} B output; that is the "
        "allocate-then-copy form, not the one-shot one"
    )


def test_the_layout_conversion_is_a_no_op_only_where_stride_c_is_moot():
    """The kernel's unstated ``stride_c == 1`` rests on a PyTorch detail.

    ``contiguous(memory_format=channels_last_3d)`` is a *no-op* on an
    NCDHW-contiguous tensor whenever enough dims are size 1 that the two formats
    cannot be told apart -- PyTorch skips size-1 dims in its format predicate.
    The entry point converts unconditionally, so in those shapes it converts
    nothing and the kernel reads NCDHW strides as if they were NDHWC.

    That is safe, but for a reason outside this code: every such shape either has
    ``stride(1) == 1`` outright (all three spatial extents are 1) or has
    ``Cin == 1``, which makes the channel stride unobservable because the only
    channel index the kernel ever dereferences is 0.  87 of the 243 shapes over
    ``{1,2,3}^5`` are ambiguous and all 87 land in one of those two cases -- a
    property of PyTorch's predicate rather than of ours, so it is pinned rather
    than assumed.
    """
    ambiguous = 0
    for shape in itertools.product((1, 2, 3), repeat=5):
        t = torch.empty(shape)
        if not (
            t.is_contiguous() and t.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            continue
        ambiguous += 1
        assert t.stride(1) == 1 or shape[1] == 1, shape
    assert ambiguous, "no shape was ambiguous; the enumeration is vacuous"


@requires_gpu
def test_an_ambiguous_layout_still_gives_the_right_answer():
    """One of the shapes above, end to end: ``Cin = 1``, where nothing converts.

    The conversion is a no-op, the strides the kernel is handed are NCDHW's, and
    the result still has to be the reference's -- which it is only because the
    one stride that differs is the one a single-channel input never uses.
    """
    x = torch.randint(-1, 2, (1, 1, 4, 5, 6), device="cuda", dtype=torch.int8).to(
        torch.bfloat16
    )
    w = torch.randint(-1, 2, (8, 1, 3, 3, 3), device="cuda", dtype=torch.int8).to(
        torch.bfloat16
    )
    assert x.is_contiguous()
    assert x.contiguous(memory_format=torch.channels_last_3d).data_ptr() == x.data_ptr()
    assert torch.equal(conv3d_forward(x, w, padding=1), F.conv3d(x, w, padding=1))


@requires_gpu
def test_hoisted_weight_transform_is_validated():
    """``weight_rsck`` supplies every weight *value* the kernel reads.

    ``w`` is then consulted only for its shape, so a hoisted transform of the
    wrong parameter runs and returns a smooth, correctly shaped, entirely wrong
    result -- measured ``max_abs = 60.0``.  That is a live hazard rather than a
    "you asked for it": the transform exists to be cached across calls, and a
    cache keyed on the parameter's version is exactly the thing that goes stale.
    """
    problem = ConvProblem("wr", 16, 24, (4, 5, 6))
    ops = reference.make_inputs(problem, seed=67, exact=True)
    good = to_rsck(ops["weight"])
    assert torch.equal(_run(problem, ops, weight_rsck=good), _run(problem, ops))

    other = torch.randn((24, 16, 1, 1, 1), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _run(problem, ops, weight_rsck=to_rsck(other))  # a different kernel
    with pytest.raises(ValueError):
        _run(problem, ops, weight_rsck=good.float())
    # Right shape, wrong layout: the B tile load assumes Cout is contiguous.
    with pytest.raises(ValueError):
        _run(
            problem, ops, weight_rsck=good.transpose(3, 4).contiguous().transpose(3, 4)
        )


@requires_gpu
def test_every_weight_layout_gives_the_same_answer():
    """The weight is read where it lies, so its strides pick the B load.

    Three layouts take three different decisions -- ``channels_last_3d`` is
    addressed in place with a gathered tile, PyTorch's default is copied because
    a gathered tile is 5-8x slower when *neither* channel axis is unit-stride,
    and an RSCK-strided weight is addressed in place with a contiguous one --
    and they must not produce three answers.  Bitwise, not close: it is the same
    multiply-accumulate in the same order, and anything less would mean the
    layout had leaked into the arithmetic.

    The RSCK-strided case is the one a test is really needed for.  It has
    PyTorch's shape over this kernel's storage order, which is what an
    integration would allocate to make the B tile contiguous, and it is the case
    where ``to_rsck`` is a no-op and ``weight`` and ``weight_rsck`` are the same
    tensor -- so it is also where a mix-up between them would hide.
    """
    problem = ConvProblem("layouts", 32, 48, (4, 5, 6))
    ops = reference.make_inputs(problem, seed=53, exact=True)
    w = ops["weight"]
    layouts = {
        "channels_last": w.contiguous(memory_format=torch.channels_last_3d),
        "contiguous": w.contiguous(),
        "rsck_strided": (w.permute(2, 3, 4, 1, 0).contiguous().permute(4, 3, 0, 1, 2)),
    }
    ref = _run(problem, ops)
    for name, wl in layouts.items():
        assert torch.equal(wl, w), name  # same values, different strides
        assert torch.equal(ref, _run(problem, {**ops, "weight": wl})), name
    assert torch.equal(ref, _run(problem, ops, weight_rsck=to_rsck(w)))
    # ``to_rsck`` of an already-RSCK-strided weight must not copy: that is what
    # makes the layout free for a caller who chooses it, and ``.contiguous()``
    # returning ``self`` is the whole mechanism.
    assert (
        to_rsck(layouts["rsck_strided"]).data_ptr()
        == layouts["rsck_strided"].data_ptr()
    )


@requires_gpu
def test_hoisted_weight_transform_is_equivalent():
    """``weight_rsck`` is an optimization, so it must change nothing observable.

    A caller that hoists the transform out of a training step must get the
    identical result to one that lets the entry point decide.
    """
    problem = ConvProblem("hoist", 32, 48, (4, 5, 6))
    ops = reference.make_inputs(problem, seed=53, exact=True)
    inline = _run(problem, ops)
    hoisted = _run(problem, ops, weight_rsck=to_rsck(ops["weight"]))
    assert torch.equal(inline, hoisted)


@requires_gpu
def test_output_is_channels_last_and_matches_torch_shape():
    problem = ConvProblem("shape", 16, 40, (3, 11, 5))
    ops = reference.make_inputs(problem, seed=61)
    y = _run(problem, ops)
    ref = F.conv3d(
        ops["input"],
        ops["weight"],
        ops["bias"],
        stride=problem.stride,
        padding=problem.padding,
    )
    assert y.shape == ref.shape
    assert y.is_contiguous(memory_format=torch.channels_last_3d)


@requires_gpu
def test_unsupported_calls_raise_rather_than_return_garbage():
    x = torch.randn((1, 8, 4, 4, 4), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((8, 4, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):
        conv3d_forward(x, w, padding=1, groups=2)
    good_w = torch.randn((8, 8, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        conv3d_forward(
            x, good_w, padding=1, config=ConvConfig(BLOCK_K=8, matrix_instr_nonkdim=16)
        )


@requires_gpu
def test_repeated_calls_are_bitwise_reproducible():
    """No float atomics, fixed grid, fixed accumulation order.

    ScaFFold's default configuration is *not* bitwise reproducible today because
    MIOpen's backward-weight uses atomics.  The forward has no reason to inherit
    that, and stating the property as a test is what stops a later split-K
    variant from quietly giving it up.
    """
    problem = ConvProblem("determinism", 64, 64, (8, 12, 10))
    ops = reference.make_inputs(problem, seed=71)
    first = _run(problem, ops)
    for _ in range(4):
        assert torch.equal(first, _run(problem, ops))


@requires_gpu
@pytest.mark.slow
def test_indices_beyond_int32_are_addressed_correctly():
    """A 2.2 GiB activation: the offsets must widen, and the far end must be read.

    Unsharded scale 8 is ``1 x 128 x 258^3 = 2.20e9`` elements, which is where
    MIOpen itself asserts (upstream bug, reproducer filed) and where the
    buffer-load fast path is lost because ``is_within_2gb`` reads the whole
    storage.  Losing buffer loads is a performance question; getting the *index*
    wrong is a correctness one, and only a tensor this size asks it.

    The check is placed at the far end deliberately: a truncated 32-bit offset
    aliases back to the start of the tensor, so a spot check near the end catches
    it while a check of the mean would not.
    """
    free, _ = torch.cuda.mem_get_info()
    if free < 12 << 30:
        pytest.skip("needs ~12 GiB free")
    cin, cout, sp = 128, 16, (258, 258, 258)
    x = torch.zeros((1, cin, *sp), device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d
    )
    w = torch.zeros((cout, cin, 1, 1, 1), device="cuda", dtype=torch.bfloat16)
    # One channel of one weight, so the output is a copy of one input channel.
    w[0, 0, 0, 0, 0] = 1.0
    x[0, 0, -1, -1, -1] = 3.0
    x[0, 0, 0, 0, 0] = 5.0
    y = conv3d_forward(x, w, padding=0)
    assert y[0, 0, -1, -1, -1].item() == 3.0
    assert y[0, 0, 0, 0, 0].item() == 5.0
    assert y.sum().item() == 8.0


@requires_gpu
@pytest.mark.slow
def test_indices_beyond_int32_are_addressed_correctly_with_taps_and_padding():
    """The same widening where ``tap_off`` and the ``PADDED`` predicate are live.

    The test above uses a ``1x1x1`` weight and no padding, so the widened row
    offset is never bumped by a tap and the six boundary compares are compiled
    out entirely -- a change that widened only the pointwise path would pass it.
    This is the same ``1 x 128 x 258^3`` activation (2.20e9 elements, unsharded
    scale 8) at ``k=3, padding=1``.

    One tap, the last one, so that ``tap_off`` is at its maximum and the far
    corner of the output reads the far corner of the input: a truncated 32-bit
    offset aliases back towards the start, which a spot check at the end catches
    and a check of the mean does not.
    """
    free, _ = torch.cuda.mem_get_info()
    if free < 12 << 30:
        pytest.skip("needs ~12 GiB free")
    cin, cout, sp = 128, 16, (258, 258, 258)
    x = torch.empty(
        (1, cin, *sp),
        device="cuda",
        dtype=torch.bfloat16,
        memory_format=torch.channels_last_3d,
    ).zero_()
    w = torch.zeros((cout, cin, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    # Tap (2,2,2) of channel 0 alone.  At padding 1 that is y[o] = x[o + 1].
    w[0, 0, 2, 2, 2] = 1.0
    x[0, 0, -1, -1, -1] = 3.0
    x[0, 0, 1, 1, 1] = 5.0
    y = conv3d_forward(x, w, padding=1)
    assert y[0, 0, -2, -2, -2].item() == 3.0
    assert y[0, 0, 0, 0, 0].item() == 5.0
    assert y.sum().item() == 8.0


@requires_gpu
@pytest.mark.slow
def test_a_weight_beyond_int32_is_addressed_correctly():
    """``taps * Cin * Cout`` over ``2**31``, which used to fault the GPU.

    The weight offset was int32 unconditionally, on a stated assumption
    ("weights are never large enough") that ``is_supported`` did not enforce: the
    reviewer's matched pair at ``k=13`` differing only in ``Cout`` ran clean at
    1.15e9 weight elements and took a memory access fault at 2.30e9.  Not
    reachable from this model -- its widest weight is 28.3 M elements, 80x below
    -- but the package is meant to be lifted into DistConv and released, and a
    kernel whose reason for existing is MIOpen's int32 overflow should not have
    one of its own.

    Shaped for the *offset* and not for the arithmetic: 2 taps, ``M = 1``, and
    the widths chosen so the GEMM stays trivial while the row offset does not.
    What has to overflow is ``dij * stride_wt + offs_k * stride_wc``, because
    ``offs_n`` is a *second* ``addptr`` and is sign-extended on its own -- a first
    attempt at this test put the excess there and passed against the unfixed
    kernel.  So the quantity to push past ``2**31`` is ``taps*Cin*Cout - Cout``,
    which here is 65,537 elements over.

    ``w`` is an expanded view: ``weight_rsck`` supplies the values and ``w`` is
    read only for its shape, which keeps this to one 4.29 GiB allocation rather
    than two.
    """
    free, _ = torch.cuda.mem_get_info()
    if free < 8 << 30:
        pytest.skip("needs ~8 GiB free")
    bf16, cin, cout = torch.bfloat16, 16385, 65536
    wr = torch.zeros((2, 1, 1, cin, cout), device="cuda", dtype=bf16)
    assert (2 * cin - 1) * cout > 2**31 - 1  # the largest row offset
    w = torch.zeros((), device="cuda", dtype=bf16).expand(cout, cin, 2, 1, 1)
    x = torch.zeros((1, cin, 2, 1, 1), device="cuda", dtype=bf16).contiguous(
        memory_format=torch.channels_last_3d
    )

    wr[1, 0, 0, cin - 1, cout - 1] = 1.0  # the last element of the weight
    x[0, cin - 1, 1, 0, 0] = 3.0
    wr[0, 0, 0, 0, 0] = 1.0  # and the first
    x[0, 0, 0, 0, 0] = 5.0

    y = conv3d_forward(x, w, padding=0, weight_rsck=wr)
    assert tuple(y.shape) == (1, cout, 1, 1, 1)
    assert y[0, cout - 1, 0, 0, 0].item() == 3.0
    assert y[0, 0, 0, 0, 0].item() == 5.0
    assert y.sum().item() == 8.0
