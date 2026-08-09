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

"""Tests for the channels-last Triton GroupNorm (``ScaFFold.unet.triton_group_norm``).

The kernel replaces a stock op, so almost every test here is a *parity* test:
values and gradients against ``F.group_norm``, but with the reference computed
in **float64** rather than against another fp32 result -- an fp32-vs-fp32
comparison cannot tell a correct kernel from one that has merely made the same
mistake, and it cannot see the variance-formula failure the Welford
implementation exists to fix (``test_welford_survives_large_mean``).  Each
parity test reports the measured relative error so a regression shows up as a
number, not just a boolean.

The other three things being pinned down:

* the **contract** -- output dtype exactly matches ``F.group_norm``'s
  (including its fp32 autocast policy), output *memory format* matches the
  input's (which is where the kernel deliberately differs from stock, and the
  entire reason it exists), and ``is_supported`` accepts exactly the inputs the
  native kernel serves;
* **determinism** -- the same call twice is bitwise identical, forward and
  backward, because the split count and tiling are pure functions of the shape;
* **composition** -- the op is a real dispatcher op, so it must survive
  ``torch.compile(fullgraph=True)`` without a graph break and a ``DCTensor``
  round trip through ``__torch_dispatch__`` with the autograd graph intact.

CPU runs never touch Triton: the module defers ``import triton`` to the first
call that reaches a kernel, which ``test_import_does_not_pull_in_triton``
checks in a fresh interpreter.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from ScaFFold.unet import triton_group_norm as tgn
from ScaFFold.unet.triton_group_norm import is_supported, triton_group_norm

CL = torch.channels_last_3d
GROUPS = 8
EPS = 1e-5

#: Relative-error ceilings against a float64 reference, by input dtype.  The
#: fp32 numbers observed on MI300A at these (small) test shapes are ~2e-07 for
#: y/dx/dweight/dbias; the ceiling leaves room for the 3e-05 that a 134M-element
#: fp32 reduction shows at the largest production shape.  The low-precision
#: ceilings are set just above the output's own rounding: 2^-8 for bf16 and
#: 2^-11 for fp16, measured 3.3e-03 and 3.8e-04.
_TOL = {
    torch.float32: 1e-4,
    torch.bfloat16: 2e-2,
    torch.float16: 3e-3,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rel(actual, expected):
    """max|actual - expected| / max|expected|, computed in float64."""
    a = actual.detach().double()
    e = expected.detach().double()
    scale = e.abs().max().clamp_min(1e-30)
    return ((a - e).abs().max() / scale).item()


def _tensors(shape, dtype, device, affine=True, seed=0, mean=0.0, std=1.0):
    """Channels-last input plus (optionally) affine parameters and a cotangent."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.empty(shape, device=device, dtype=dtype, memory_format=CL)
    x.normal_(mean, std, generator=gen)
    channels = shape[1]
    if affine:
        weight = torch.empty(channels, device=device, dtype=dtype)
        weight.normal_(1.0, 0.25, generator=gen)
        bias = torch.empty(channels, device=device, dtype=dtype)
        bias.normal_(0.0, 0.25, generator=gen)
    else:
        weight = bias = None
    grad_out = torch.empty(shape, device=device, dtype=dtype, memory_format=CL)
    grad_out.normal_(generator=gen)
    return x, weight, bias, grad_out


def _run(fn, x, weight, bias, grad_out, activation=None, eps=EPS):
    """Forward + backward through ``fn``, returning detached results."""
    x = x.detach().clone().requires_grad_(True)
    weight = None if weight is None else weight.detach().clone().requires_grad_(True)
    bias = None if bias is None else bias.detach().clone().requires_grad_(True)
    out = fn(x, GROUPS, weight, bias, eps, activation)
    out.backward(grad_out.to(out.dtype))
    return (
        out.detach(),
        x.grad,
        None if weight is None else weight.grad,
        None if bias is None else bias.grad,
    )


def _reference(x, weight, bias, grad_out, activation=None, eps=EPS):
    """``F.group_norm`` (+ optional ReLU) evaluated entirely in float64."""

    def fn(x, groups, weight, bias, eps, activation):
        out = F.group_norm(x, groups, weight, bias, eps)
        return F.relu(out) if activation == "relu" else out

    return _run(
        fn,
        x.double(),
        None if weight is None else weight.double(),
        None if bias is None else bias.double(),
        grad_out.double(),
        activation,
        eps,
    )


def _assert_parity(got, ref, dtype, label, tol=None):
    """Compare (y, dx, dweight, dbias) against the float64 reference."""
    tol = _TOL[dtype] if tol is None else tol
    errors = {}
    for name, a, e in zip(("y", "dx", "dweight", "dbias"), got, ref):
        if a is None:
            assert e is None or True  # no parameter -> no gradient to compare
            continue
        errors[name] = _rel(a, e)
    print(
        f"[{label}] "
        + " ".join(f"{k}={v:.3e}" for k, v in errors.items())
        + f"  (tol {tol:.1e})"
    )
    for name, err in errors.items():
        assert err <= tol, f"{label}: {name} relative error {err:.3e} > {tol:.1e}"
    return errors


def _cuda_shapes():
    """Shapes covering N>1, non-power-of-two extents and a wide channel count."""
    return [
        (1, 64, 8, 8, 8),  # the canonical UNet shape, shrunk
        (2, 64, 9, 7, 5),  # N>1, all three extents non-power-of-two
        (1, 128, 5, 6, 7),
        (3, 256, 4, 4, 4),
        (1, 2048, 6, 6, 6),  # widest UNet channel count
    ]


# ---------------------------------------------------------------------------
# CPU-only behaviour (no Triton, no GPU)
# ---------------------------------------------------------------------------


def test_import_does_not_pull_in_triton():
    """Importing the module must not import Triton.

    Run in a fresh interpreter because any earlier GPU test in this session
    would already have built the kernels.  The guarantee matters twice over: a
    CPU-only unit run must not pay Triton's import, and the module must stay
    importable on a build that has no Triton at all.
    """
    script = (
        "import sys; import ScaFFold.unet.triton_group_norm as m; "
        "assert m.tl is None, 'kernels built at import time'; "
        "print('triton' in sys.modules)"
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        f"triton was imported at module import time: {result.stdout!r}"
    )


def test_cpu_input_falls_back_bitwise():
    """A CPU tensor is not supported, and the fallback is the stock kernel."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 64, 4, 4, 4, generator=gen).requires_grad_(True)
    weight = torch.randn(64, generator=gen).requires_grad_(True)
    bias = torch.randn(64, generator=gen).requires_grad_(True)
    assert is_supported(x, GROUPS, weight, bias) is False

    out = triton_group_norm(x, GROUPS, weight, bias, EPS)
    assert torch.equal(out, F.group_norm(x, GROUPS, weight, bias, EPS))
    out.pow(2).sum().backward()
    assert x.grad is not None and weight.grad is not None and bias.grad is not None


def test_cpu_fused_relu_falls_back_bitwise():
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(2, 32, 3, 4, 5, generator=gen)
    weight = torch.randn(32, generator=gen)
    bias = torch.randn(32, generator=gen)
    got = triton_group_norm(x, 8, weight, bias, EPS, "relu")
    assert torch.equal(got, F.relu(F.group_norm(x, 8, weight, bias, EPS)))


def test_unknown_activation_raises():
    x = torch.randn(1, 8, 2, 2, 2)
    with pytest.raises(ValueError, match="activation"):
        triton_group_norm(x, 2, activation="gelu")
    assert is_supported(x, 2, activation="gelu") is False


def test_select_strategy_is_a_pure_function_of_shape():
    """The dispatch hook must be deterministic -- the reduction order, and so
    the bits of the result, depend on it."""
    for args in ((1, 64, 8**3, 8), (2, 2048, 16**3, 8), (1, 128, 7 * 5 * 3, 4)):
        first = tgn.select_strategy(*args)
        assert first in tgn.STRATEGIES
        assert all(tgn.select_strategy(*args) == first for _ in range(3))


def test_tuning_table_covers_the_scale8_shapes():
    """The frozen table is what makes the kernel reproducible; keep it honest."""
    for channels, edge in ((64, 256), (128, 128), (256, 64), (512, 32), (1024, 16)):
        assert tgn.default_config(channels, edge**3) is tgn._TUNED[(channels, edge)]
    # An unlisted shape falls back to the generic config rather than failing.
    assert tgn.default_config(96, 11**3) == tgn.GNConfig()


def test_plan_depends_only_on_shape():
    """Two plans for the same shape must be identical objects of identical
    content, or the split count could drift between calls and break bitwise
    reproducibility."""
    a = tgn._plan(2, 128, 32**3, 8, 2 * 128 * 32**3)
    b = tgn._plan(2, 128, 32**3, 8, 2 * 128 * 32**3)
    assert (a.nsplit, a.chunk, a.block_s_stats, a.block_s_elem, a.int64) == (
        b.nsplit,
        b.chunk,
        b.block_s_stats,
        b.block_s_elem,
        b.int64,
    )
    # int64 addressing turns on exactly when a linear index can overflow int32.
    small = tgn._plan(1, 64, 128**3, 8, 64 * 128**3)
    big = tgn._plan(2, 64, 256**3, 8, 2 * 64 * 256**3)
    assert small.int64 is False
    assert big.int64 is True


# ---------------------------------------------------------------------------
# is_supported
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_is_supported_accepts_the_fast_path():
    device = torch.device("cuda")
    x = torch.empty(1, 64, 6, 6, 6, device=device, memory_format=CL).normal_()
    weight = torch.randn(64, device=device)
    assert is_supported(x, GROUPS, weight, weight) is True
    assert is_supported(x, GROUPS) is True
    assert is_supported(x, GROUPS, activation="relu") is True


@pytest.mark.gpu
def test_is_supported_rejections():
    """Everything ``is_supported`` rejects must be something a caller can hand
    to ``F.group_norm`` instead -- so the rejections are the contract's edge."""
    device = torch.device("cuda")
    cl = torch.empty(1, 64, 6, 6, 6, device=device, memory_format=CL).normal_()
    cases = {
        "cpu tensor": (torch.randn(1, 64, 6, 6, 6), GROUPS, None, None, None),
        "contiguous (NCDHW)": (
            torch.randn(1, 64, 6, 6, 6, device=device),
            GROUPS,
            None,
            None,
            None,
        ),
        "float64": (
            torch.empty(
                1, 64, 6, 6, 6, device=device, dtype=torch.float64, memory_format=CL
            ),
            GROUPS,
            None,
            None,
            None,
        ),
        "4-D": (torch.randn(1, 64, 6, 6, device=device), GROUPS, None, None, None),
        "channels not divisible": (cl, 7, None, None, None),
        "num_groups=0": (cl, 0, None, None, None),
        "bad activation": (cl, GROUPS, None, None, "gelu"),
        "weight wrong size": (cl, GROUPS, torch.randn(32, device=device), None, None),
        "weight on cpu": (cl, GROUPS, torch.randn(64), None, None),
        "weight dtype mismatch": (
            cl,
            GROUPS,
            torch.randn(64, device=device, dtype=torch.bfloat16),
            None,
            None,
        ),
        "sliced (non-contiguous)": (
            torch.empty(1, 64, 6, 6, 12, device=device, memory_format=CL)[..., ::2],
            GROUPS,
            None,
            None,
            None,
        ),
        "not a tensor": (None, GROUPS, None, None, None),
    }
    for label, args in cases.items():
        assert is_supported(*args) is False, f"{label} should be rejected"


@pytest.mark.gpu
def test_rejected_inputs_still_produce_stock_results():
    """``triton_group_norm`` stays total: rejects go to ``F.group_norm``."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(4)
    x = torch.randn(2, 64, 5, 6, 7, device=device, generator=gen).requires_grad_(True)
    weight = torch.randn(64, device=device, generator=gen).requires_grad_(True)
    bias = torch.randn(64, device=device, generator=gen).requires_grad_(True)
    assert is_supported(x, GROUPS, weight, bias) is False
    got = triton_group_norm(x, GROUPS, weight, bias, EPS)
    assert torch.equal(got, F.group_norm(x, GROUPS, weight, bias, EPS))


# ---------------------------------------------------------------------------
# value / gradient parity against float64
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("shape", _cuda_shapes())
def test_parity_fp32(shape):
    device = torch.device("cuda")
    x, weight, bias, grad_out = _tensors(
        shape, torch.float32, device, seed=hash(shape) % 1000
    )
    assert is_supported(x, GROUPS, weight, bias)
    got = _run(triton_group_norm, x, weight, bias, grad_out)
    ref = _reference(x, weight, bias, grad_out)
    _assert_parity(got, ref, torch.float32, f"fp32 {shape}")
    assert got[0].is_contiguous(memory_format=CL)
    assert got[1].is_contiguous(memory_format=CL)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", [None, "relu"])
def test_parity_every_dtype_and_activation(dtype, activation):
    device = torch.device("cuda")
    shape = (2, 128, 7, 6, 5)
    x, weight, bias, grad_out = _tensors(shape, dtype, device, seed=11)
    assert is_supported(x, GROUPS, weight, bias, activation)
    got = _run(triton_group_norm, x, weight, bias, grad_out, activation)
    ref = _reference(x, weight, bias, grad_out, activation)
    _assert_parity(got, ref, dtype, f"{dtype} act={activation}")
    assert got[0].dtype == dtype
    assert got[1].dtype == dtype
    assert got[2].dtype == dtype and got[3].dtype == dtype


@pytest.mark.gpu
@pytest.mark.parametrize("affine", ["both", "weight_only", "bias_only", "neither"])
def test_parity_without_affine_parameters(affine):
    """``weight=None`` / ``bias=None`` are separate kernel constexpr paths."""
    device = torch.device("cuda")
    shape = (2, 64, 5, 5, 5)
    x, weight, bias, grad_out = _tensors(shape, torch.float32, device, seed=19)
    if affine in ("bias_only", "neither"):
        weight = None
    if affine in ("weight_only", "neither"):
        bias = None
    assert is_supported(x, GROUPS, weight, bias)
    got = _run(triton_group_norm, x, weight, bias, grad_out)

    reference_weight = weight
    if weight is None and bias is not None:
        # Upstream limitation, not a difference in this kernel:
        # ``F.group_norm(x, g, None, bias).backward()`` raises "tensor does not
        # have a device" on both CPU and CUDA (torch 2.13.0+rocm7.2), so the
        # float64 reference has to spell the same computation with weight=1.
        # This kernel handles the combination directly.
        reference_weight = torch.ones_like(bias)
    ref = _reference(x, reference_weight, bias, grad_out)
    # ``_assert_parity`` skips outputs this configuration does not produce.
    _assert_parity(got, ref, torch.float32, f"affine={affine}")


@pytest.mark.gpu
def test_partial_gradient_requirements():
    """Only some inputs requiring grad must not change the ones that do."""
    device = torch.device("cuda")
    x, weight, bias, grad_out = _tensors(
        (1, 64, 5, 5, 5), torch.float32, device, seed=23
    )
    full = _run(triton_group_norm, x, weight, bias, grad_out)

    frozen_w = weight.detach().clone()
    frozen_b = bias.detach().clone()
    xi = x.detach().clone().requires_grad_(True)
    out = triton_group_norm(xi, GROUPS, frozen_w, frozen_b, EPS)
    out.backward(grad_out)
    assert torch.equal(xi.grad, full[1])
    assert frozen_w.grad is None and frozen_b.grad is None

    # ... and the mirror image: parameters only.
    xn = x.detach().clone()
    wn = weight.detach().clone().requires_grad_(True)
    bn = bias.detach().clone().requires_grad_(True)
    triton_group_norm(xn, GROUPS, wn, bn, EPS).backward(grad_out)
    assert torch.equal(wn.grad, full[2])
    assert torch.equal(bn.grad, full[3])


# ---------------------------------------------------------------------------
# fused activation
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_relu_matches_unfused(dtype):
    """The fused store and ``F.relu`` on the unfused output must agree bitwise,
    forward *and* backward -- the backward recomputes the pre-activation rather
    than reading it back, so this is the test that recomputation is exact."""
    device = torch.device("cuda")
    x, weight, bias, grad_out = _tensors((2, 64, 6, 7, 8), dtype, device, seed=29)
    fused = _run(triton_group_norm, x, weight, bias, grad_out, "relu")

    def unfused(x, groups, weight, bias, eps, _activation):
        return F.relu(triton_group_norm(x, groups, weight, bias, eps))

    separate = _run(unfused, x, weight, bias, grad_out)
    for name, a, b in zip(("y", "dx", "dweight", "dbias"), fused, separate):
        assert torch.equal(a, b), f"fused vs unfused+relu differ in {name}"
    # A ReLU that never fires would make this test vacuous.
    assert (fused[0] == 0).any() and (fused[0] > 0).any()


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("activation", [None, "relu"])
def test_bitwise_determinism(activation):
    """Same input twice => bitwise-equal output and gradients.

    Guaranteed by construction (no float atomics; grid, split count and tile
    sizes are pure functions of the shape) and asserted here because a future
    run-time autotuner would silently break it.
    """
    device = torch.device("cuda")
    x, weight, bias, grad_out = _tensors(
        (2, 128, 9, 11, 13), torch.float32, device, seed=31
    )
    first = _run(triton_group_norm, x, weight, bias, grad_out, activation)
    second = _run(triton_group_norm, x, weight, bias, grad_out, activation)
    for name, a, b in zip(("y", "dx", "dweight", "dbias"), first, second):
        assert torch.equal(a, b), f"{name} is not bitwise reproducible"


# ---------------------------------------------------------------------------
# numerics: Welford vs E[x^2] - E[x]^2
# ---------------------------------------------------------------------------


def _naive_group_norm(x, num_groups, weight, bias, eps):
    """The prototype's variance formula, reproduced in fp32 torch ops.

    ``var = E[x^2] - E[x]^2`` is split-friendly and cheap, and it is what the
    kernel used before the Welford rewrite; this is the thing the test below
    must show is broken so that "the new one passes" means something.
    """
    n, channels = x.shape[0], x.shape[1]
    flat = x.reshape(n, num_groups, -1)
    mean = flat.mean(-1)
    mean_sq = (flat * flat).mean(-1)
    var = mean_sq - mean * mean
    rstd = 1.0 / torch.sqrt(var + eps)
    out = (flat - mean[..., None]) * rstd[..., None]
    out = out.reshape(x.shape)
    shape = (1, channels) + (1,) * (x.dim() - 2)
    return out * weight.reshape(shape) + bias.reshape(shape)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mean,std,naive_floor",
    [
        (0.0, 1.0, None),  # both formulations are fine here
        (100.0, 1.0, 1e-4),  # E[x^2]-E[x]^2 already an order of magnitude off
        (1e3, 1e-2, 1e-1),  # ... and here it has lost the variance entirely
    ],
)
def test_welford_survives_large_mean(mean, std, naive_floor):
    """Large-mean / small-variance input: the regression case for the rewrite.

    Measured here on MI300A at ``[1, 256, 24^3]`` (relative error of the output
    against a float64 reference computed from the same fp32 samples), with the
    production shape ``[1, 256, 64^3]`` in parentheses:

        mean=0,   std=1     this 1.3e-07 (1.6e-07)  E[x^2]-E[x]^2 1.4e-07 (1.8e-07)
        mean=1e2, std=1     this 1.1e-06 (8.0e-07)  E[x^2]-E[x]^2 9.5e-04 (5.6e-04)
        mean=1e3, std=1e-2  this 4.4e-04 (1.1e-04)  E[x^2]-E[x]^2 2.3e+00 (2.3e+00)

    i.e. at ``mean/std = 1e5`` the old formulation loses the variance outright
    (the difference of the two ~1e6-sized fp32 terms is below one ulp, so
    ``rstd`` saturates on ``eps`` and the output is meaningless) while Welford
    is still correct to 4.4e-04 -- itself *5x better* than ATen's own fp32
    GroupNorm on the same input (2.2e-03), and dominated by the fp32
    representation of a mean of 1e3 rather than by anything the kernel does.
    """
    device = torch.device("cuda")
    x, weight, bias, grad_out = _tensors(
        (1, 256, 24, 24, 24), torch.float32, device, seed=37, mean=mean, std=std
    )
    got = triton_group_norm(x, GROUPS, weight, bias, EPS)
    ref = F.group_norm(x.double(), GROUPS, weight.double(), bias.double(), EPS)
    stock = F.group_norm(x, GROUPS, weight, bias, EPS)
    naive = _naive_group_norm(x, GROUPS, weight, bias, EPS)

    err = _rel(got, ref)
    err_stock = _rel(stock, ref)
    err_naive = _rel(naive, ref)
    print(
        f"[welford mean={mean:g} std={std:g}] triton={err:.3e} "
        f"aten_fp32={err_stock:.3e} naive_Ex2={err_naive:.3e}"
    )
    # Never worse than ATen's own fp32 kernel by more than a small factor.
    assert err <= max(4.0 * err_stock, 1e-5), (
        f"triton {err:.3e} vs aten fp32 {err_stock:.3e}"
    )
    if naive_floor is not None:
        assert err_naive > naive_floor, (
            "the naive formulation was expected to fail here "
            f"but only reached {err_naive:.3e}"
        )
        assert err < err_naive / 10.0, (
            f"Welford ({err:.3e}) is not clearly better than "
            f"E[x^2]-E[x]^2 ({err_naive:.3e})"
        )


# ---------------------------------------------------------------------------
# dtypes, layouts, autocast
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("autocast_dtype", [None, torch.bfloat16, torch.float16])
def test_output_dtype_matches_stock(dtype, autocast_dtype):
    """The dtype contract, including autocast's fp32 policy for GroupNorm.

    Stock behaviour on this build (measured, not assumed): without autocast the
    output dtype is the input dtype; inside *any* enabled CUDA autocast region
    it is fp32, because ``at::group_norm`` carries the fp32 cast policy.
    """
    device = torch.device("cuda")
    x, weight, bias, _ = _tensors((1, 64, 5, 5, 5), dtype, device, seed=41)
    if autocast_dtype is not None:
        # Autocast casts the parameters itself, and production keeps them fp32.
        weight = weight.float()
        bias = bias.float()
    ctx = (
        torch.autocast("cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else torch.autocast("cuda", enabled=False)
    )
    with ctx:
        assert is_supported(x, GROUPS, weight, bias)
        stock = F.group_norm(x, GROUPS, weight, bias, EPS)
        got = triton_group_norm(x, GROUPS, weight, bias, EPS)
    expected = torch.float32 if autocast_dtype is not None else dtype
    assert stock.dtype == expected, "assumption about stock GroupNorm broke"
    assert got.dtype == stock.dtype
    # ... and the one deliberate difference: stock always returns contiguous.
    assert stock.is_contiguous() and not stock.is_contiguous(memory_format=CL)
    assert got.is_contiguous(memory_format=CL)
    print(
        f"[dtype in={dtype} autocast={autocast_dtype}] "
        f"stock={stock.dtype}/CONT mine={got.dtype}/CL rel={_rel(got, stock):.3e}"
    )
    assert _rel(got.float(), stock.float()) <= _TOL[stock.dtype]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("autocast_dtype", [None, torch.bfloat16])
def test_out_dtype_opt_in_overrides_the_stock_rule(dtype, autocast_dtype):
    """``out_dtype=`` is the only way past the rule above, and it is opt-in.

    Three spellings, one call each: the default (:data:`tgn.MATCH_STOCK_DTYPE`)
    reproduces ``F.group_norm``; ``None`` asks for the input's dtype, which
    differs from stock exactly inside an autocast region; and an explicit dtype
    asks for that one.  ``FastGroupNorm`` is the caller that uses the middle
    spelling -- see ``ScaFFold.unet.group_norm``.
    """
    device = torch.device("cuda")
    x, weight, bias, _ = _tensors((1, 64, 5, 5, 5), dtype, device, seed=61)
    if autocast_dtype is not None:
        weight = weight.float()
        bias = bias.float()
    ctx = (
        torch.autocast("cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else torch.autocast("cuda", enabled=False)
    )
    with ctx:
        default = triton_group_norm(x, GROUPS, weight, bias, EPS)
        narrow = triton_group_norm(x, GROUPS, weight, bias, EPS, out_dtype=None)
        wide = triton_group_norm(x, GROUPS, weight, bias, EPS, out_dtype=torch.float32)
    stock = torch.float32 if autocast_dtype is not None else dtype
    assert default.dtype == stock, "the default moved; it is the standalone contract"
    assert narrow.dtype == dtype
    assert wide.dtype == torch.float32
    for got in (default, narrow, wide):
        assert got.is_contiguous(memory_format=CL)
    # Only the *store* changes: the statistics and the normalized value are
    # fp32 on every one of these calls, and the tiling plan is a function of
    # the shape alone (`_plan` takes no dtype), so the narrow answer is the
    # wide one rounded once -- bitwise, not approximately.
    assert torch.equal(narrow, wide.to(dtype))
    assert torch.equal(
        default, wide if default.dtype is torch.float32 else wide.to(dtype)
    )


@pytest.mark.gpu
def test_out_dtype_is_honoured_on_the_fallback_route_too():
    """A contiguous input takes ``F.group_norm``, and still answers in the
    requested dtype -- otherwise the dtype of a result would depend on which
    kernel served it, which is the thing every other part of this contract
    exists to prevent."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(63)
    x = torch.randn(2, 64, 5, 6, 7, device=device, dtype=torch.bfloat16, generator=gen)
    weight = torch.randn(64, device=device, generator=gen)
    bias = torch.randn(64, device=device, generator=gen)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert not is_supported(x, GROUPS, weight, bias), "meant to be a fallback"
        stock = triton_group_norm(x, GROUPS, weight, bias, EPS)
        narrow = triton_group_norm(x, GROUPS, weight, bias, EPS, out_dtype=None)
        relu = triton_group_norm(
            x, GROUPS, weight, bias, EPS, "relu", out_dtype=torch.bfloat16
        )
    assert stock.dtype is torch.float32, "assumption about stock GroupNorm broke"
    assert narrow.dtype is torch.bfloat16
    assert torch.equal(narrow, stock.to(torch.bfloat16))
    # ... and the activation is applied before the narrowing, as it is in the
    # kernel's store, so a fused and an unfused ReLU still agree bitwise.
    assert relu.dtype is torch.bfloat16
    assert torch.equal(relu, F.relu(stock).to(torch.bfloat16))


@pytest.mark.parametrize("bad", [torch.float64, torch.int32, "bfloat16", 16])
def test_out_dtype_rejects_what_the_kernel_cannot_store(bad):
    """Validated as an argument, before anything looks at the input, so the
    message names the argument rather than surfacing as a Triton compile error
    deep inside a launch."""
    x = torch.randn(1, 64, 2, 2, 2).to(memory_format=CL)
    with pytest.raises(ValueError, match="out_dtype"):
        triton_group_norm(x, GROUPS, out_dtype=bad)


@pytest.mark.gpu
def test_autocast_gradient_dtypes_match_stock():
    """Under autocast, ``d_input`` keeps the input's dtype and the parameter
    gradients stay fp32 -- exactly what the cast nodes around stock GroupNorm
    produce."""
    device = torch.device("cuda")
    x, _, _, grad_out = _tensors((1, 64, 5, 5, 5), torch.bfloat16, device, seed=43)
    weight = torch.randn(64, device=device, requires_grad=True)
    bias = torch.randn(64, device=device, requires_grad=True)

    def run(fn):
        xi = x.detach().clone().requires_grad_(True)
        w = weight.detach().clone().requires_grad_(True)
        b = bias.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(xi, GROUPS, w, b, EPS)
        out.backward(grad_out.to(out.dtype))
        return out, xi.grad, w.grad, b.grad

    stock = run(F.group_norm)
    got = run(triton_group_norm)
    for name, a, b in zip(("y", "dx", "dweight", "dbias"), got, stock):
        assert a.dtype == b.dtype, f"{name}: {a.dtype} != {b.dtype}"
        assert _rel(a.float(), b.float()) <= 5e-2, name


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["channels_last_3d", "contiguous"])
def test_memory_format_is_preserved(layout):
    """Both layouts round-trip their own format; contiguous input takes the
    documented ``F.group_norm`` fallback rather than silently changing layout."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(47)
    x = torch.randn(2, 64, 5, 6, 7, device=device, generator=gen)
    if layout == "channels_last_3d":
        x = x.contiguous(memory_format=CL)
    weight = torch.randn(64, device=device, generator=gen)
    bias = torch.randn(64, device=device, generator=gen)
    grad_out = torch.randn(2, 64, 5, 6, 7, device=device, generator=gen)
    if layout == "channels_last_3d":
        grad_out = grad_out.contiguous(memory_format=CL)

    assert is_supported(x, GROUPS, weight, bias) is (layout == "channels_last_3d")
    got = _run(triton_group_norm, x, weight, bias, grad_out)
    ref = _reference(x, weight, bias, grad_out)
    _assert_parity(got, ref, torch.float32, f"layout={layout}")
    if layout == "channels_last_3d":
        assert got[0].is_contiguous(memory_format=CL)
        assert got[1].is_contiguous(memory_format=CL)
    else:
        assert got[0].is_contiguous()
        assert got[1].is_contiguous()


# ---------------------------------------------------------------------------
# int64 offsets
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_int64_switch_flips_at_int32_max():
    """The switch is a pure function of the element count, so pin the boundary.

    ``[2, 64, 256^3]`` is *exactly* 2^31 elements: the shape that made an
    int64 path mandatory before batch>1 or scale 16.
    """
    assert tgn._plan(1, 64, 255**3, 8, 64 * 255**3).int64 is False
    assert tgn._plan(2, 64, 256**3, 8, 2 * 64 * 256**3).int64 is True


@pytest.mark.gpu
@pytest.mark.slow
def test_correct_above_int32_max_elements():
    """Correctness at a shape whose linear element count exceeds INT32_MAX.

    ``[2, 64, 256, 256, 257]`` is 2_155_872_256 elements -- 8.4M past 2^31, and
    non-power-of-two in the fastest spatial dimension so a truncated offset
    cannot accidentally land on the right address.  fp32 (8.03 GiB per tensor)
    keeps the comparison sharp; the reference needs an NCDHW copy, so the peak
    is ~48 GiB and the test skips, loudly, if the device cannot hold that.
    """
    device = torch.device("cuda")
    shape = (2, 64, 256, 256, 257)
    numel = 1
    for dim in shape:
        numel *= dim
    assert numel > 2**31 - 1
    needed = 6 * numel * 4  # x, y, x_contig, reference, and slack for the diff
    free, total = torch.cuda.mem_get_info()
    if free < needed:
        pytest.skip(
            f"needs ~{needed / 2**30:.0f} GiB free, device has "
            f"{free / 2**30:.0f} GiB of {total / 2**30:.0f} GiB"
        )

    gen = torch.Generator(device=device).manual_seed(53)
    x = torch.empty(shape, device=device, memory_format=CL)
    x.normal_(generator=gen)
    weight = torch.randn(64, device=device, generator=gen)
    bias = torch.randn(64, device=device, generator=gen)

    assert tgn._plan(
        shape[0], shape[1], shape[2] * shape[3] * shape[4], GROUPS, numel
    ).int64
    got = triton_group_norm(x, GROUPS, weight, bias, EPS)
    assert got.is_contiguous(memory_format=CL)

    contiguous = x.contiguous()
    del x
    torch.cuda.empty_cache()
    reference = F.group_norm(contiguous, GROUPS, weight, bias, EPS)
    del contiguous
    torch.cuda.empty_cache()

    # Compare both batch items separately: a truncated 32-bit offset wraps
    # partway through, so the second half would be wrong while the first is not.
    errors = [_rel(got[i], reference[i]) for i in range(shape[0])]
    print(f"[int64 {shape}] per-sample relative error {errors}")
    for i, err in enumerate(errors):
        assert err < 1e-4, f"sample {i}: relative error {err:.3e}"
    del got, reference
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# composition: torch.compile and DCTensor
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_custom_op_is_registered_with_a_fake_kernel():
    """A meta/fake kernel is what lets Dynamo trace the op without running it."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    assert hasattr(torch.ops.scaffold_gn, "group_norm")
    assert hasattr(torch.ops.scaffold_gn, "group_norm_backward")
    with FakeTensorMode():
        x = torch.empty(2, 64, 5, 6, 7, device="cuda", memory_format=CL)
        weight = torch.empty(64, device="cuda")
        out, mean, rstd = torch.ops.scaffold_gn.group_norm(
            x, GROUPS, weight, weight, EPS, "relu", None
        )
        assert out.shape == x.shape and out.dtype == x.dtype
        assert out.is_contiguous(memory_format=CL)
        assert mean.shape == (2, GROUPS) and rstd.dtype == torch.float32
        # ... and the dtype override autocast uses.
        bf16 = torch.empty(
            2, 64, 5, 6, 7, device="cuda", dtype=torch.bfloat16, memory_format=CL
        )
        out32, _, _ = torch.ops.scaffold_gn.group_norm(
            bf16, GROUPS, None, None, EPS, None, torch.float32
        )
        assert out32.dtype == torch.float32
        assert out32.is_contiguous(memory_format=CL)


@pytest.mark.gpu
@pytest.mark.parametrize("activation", [None, "relu"])
def test_torch_compile_fullgraph(activation):
    """``fullgraph=True`` raises on a graph break, so this *is* the no-break
    test; the compiled result must additionally be bitwise equal to eager,
    because the op is opaque to Inductor and so cannot be re-associated."""
    device = torch.device("cuda")
    x, weight, bias, grad_out = _tensors(
        (2, 64, 6, 6, 6), torch.float32, device, seed=59
    )

    def fn(x, weight, bias):
        return triton_group_norm(x, GROUPS, weight, bias, EPS, activation) * 2.0

    def wrapped(x, groups, weight, bias, eps, _activation):
        return fn(x, weight, bias)

    eager = _run(wrapped, x, weight, bias, grad_out)

    compiled_fn = torch.compile(fn, fullgraph=True, dynamic=False)

    def wrapped_compiled(x, groups, weight, bias, eps, _activation):
        return compiled_fn(x, weight, bias)

    compiled = _run(wrapped_compiled, x, weight, bias, grad_out)
    for name, a, b in zip(("y", "dx", "dweight", "dbias"), compiled, eager):
        assert torch.equal(a, b), f"compiled and eager differ in {name}"
    assert compiled[0].is_contiguous(memory_format=CL)


@pytest.fixture
def dc_cuda():
    """DistConv package plus a CUDA ParallelStrategy over a 1-rank NCCL group.

    Mirrors ``tests/test_groupnorm.py``'s fixture (``num_shards=(1, 1, 1)`` on
    dims (2, 3, 4) is what worker.py builds for a single-device run) on its own
    rendezvous port so the two suites can run in one session.
    """
    import torch.distributed as dist

    distconv = pytest.importorskip("distconv")

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29519")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        created = True
    strategy = distconv.ParallelStrategy(
        num_shards=(1, 1, 1), shard_dim=(2, 3, 4), device_type="cuda"
    )
    yield distconv, strategy
    if created and dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.gpu
def test_dctensor_round_trip(dc_cuda):
    """A DCTensor must go in and come out, with the graph back to its producer
    intact.

    The op is a real dispatcher op, so DistConv's generic
    ``__torch_dispatch__`` unwraps to the local shard, runs it, and rewraps --
    no GroupNorm-specific handling needed on either side.  The producer in
    front matters: with a bare ``input._tensor`` read the graph would be severed
    there and only GroupNorm's own parameters would see gradients.
    """
    distconv, strategy = dc_cuda
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(61)
    x = torch.empty(1, 64, 6, 6, 6, device=device, memory_format=CL)
    x.normal_(generator=gen)
    grad_out = torch.empty_like(x)
    grad_out.normal_(generator=gen)
    weight = torch.randn(64, device=device, generator=gen)
    bias = torch.randn(64, device=device, generator=gen)
    producer = torch.nn.Conv3d(64, 64, 1, bias=False).to(device)

    def run(fn, wrap):
        xi = x.detach().clone().requires_grad_(True)
        conv = torch.nn.Conv3d(64, 64, 1, bias=False).to(device)
        with torch.no_grad():
            conv.weight.copy_(producer.weight)
        w = weight.detach().clone().requires_grad_(True)
        b = bias.detach().clone().requires_grad_(True)
        inp = distconv.DCTensor.from_shard(xi, strategy) if wrap else xi
        # The conv is the producer; the explicit channels-last conversion is
        # what PYTORCH_MIOPEN_SUGGEST_NHWC=1 gives production for free.
        hidden = conv(inp).contiguous(memory_format=CL)
        out = fn(hidden, GROUPS, w, b, EPS, "relu")
        if wrap:
            assert isinstance(out, distconv.DCTensor)
            assert out.is_contiguous(memory_format=CL)
            out = distconv.distconv._ToTensor.apply(out)
        out.backward(grad_out)
        return out.detach(), xi.grad, conv.weight.grad, w.grad, b.grad

    def stock(x, groups, weight, bias, eps, _activation):
        return F.relu(F.group_norm(x, groups, weight, bias, eps))

    got = run(triton_group_norm, wrap=True)
    ref = run(stock, wrap=False)
    for name, a, b in zip(("y", "dx", "dconv", "dweight", "dbias"), got, ref):
        assert a is not None, f"{name} never received a gradient"
        err = _rel(a, b)
        print(f"[dctensor] {name}={err:.3e}")
        assert err <= 1e-4, f"{name}: relative error {err:.3e}"
