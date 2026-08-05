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

"""Adversarial edge-case tests for the channels-last Triton GroupNorm.

Companion to ``tests/test_triton_group_norm.py``, written independently during
an audit of the kernel.  It covers the ground the author's suite does not, and
pins the divergences that audit found.

The two structural gaps this file closes:

* **The masked channel axis is never exercised upstream.**  Every GPU test in
  ``test_triton_group_norm.py`` uses ``num_groups=8`` with a channel count of
  64/128/256/2048, so ``G`` and ``C/G`` are *always* powers of two and
  ``_Plan.masked_c`` is always ``False``.  The entire ``MASKED_C=True`` code
  path -- the ``cmask``/``wbm`` predicates in all four kernels, and the
  ``inner`` offsets that deliberately run past the end of a voxel -- ships
  untested.  :func:`test_masked_channel_axis_parity` and friends run it.

* **Uninitialised split-K scratch is never checked.**  ``_forward`` and
  ``_backward`` allocate their partial buffers with ``torch.empty``, so a slot
  that is read before it is written would surface as *plausible* numbers, not
  as a crash.  :func:`test_scratch_slots_are_all_written` poisons every
  ``torch.empty`` with NaN for the duration of the call, which turns that class
  of bug into a hard failure.

The audit's six findings -- no device guard, the backward fake kernel's stride
promise, silently-differentiable ``mean``/``rstd``, accepting a shape
``F.group_norm`` rejects, a non-zero ``d_input`` for single-element groups, and
undocumented double backward -- were tested here as ``xfail(strict=True)``
first and fixed afterwards; the tests remain, without the markers, as the
regression pins.  The last section adds the coverage a mutation sweep of the
kernels found thinnest: the ``INT64=True`` branch (which a default run never
compiled), the split-K Welford merge on *unequal* split counts, ``eps``
placement, and the tile-mean correction term.
"""

import contextlib
import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F

from ScaFFold.unet import triton_group_norm as tgn
from ScaFFold.unet.triton_group_norm import is_supported, triton_group_norm

CL = torch.channels_last_3d
EPS = 1e-5

#: Relative-error ceiling against the float64 reference below.  fp32 parity at
#: these (small) shapes measures ~1e-07; the ceiling leaves room for the
#: reduction noise a production-sized split-K reduction shows.
FP32_TOL = 1e-4

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# independent float64 reference (deliberately *not* F.group_norm, and
# deliberately not the helper the author's suite uses)
# ---------------------------------------------------------------------------


def _ref64(x, groups, weight, bias, eps, activation=None):
    """GroupNorm written from scratch in float64, in the (N, G, ...) view."""
    xd = x.double()
    n, channels = xd.shape[0], xd.shape[1]
    flat = xd.reshape(n, groups, -1)
    mu = flat.mean(-1, keepdim=True)
    var = ((flat - mu) ** 2).mean(-1, keepdim=True)
    y = ((flat - mu) / torch.sqrt(var + eps)).reshape(xd.shape)
    shape = (1, channels) + (1,) * (xd.dim() - 2)
    if weight is not None:
        y = y * weight.double().reshape(shape)
    if bias is not None:
        y = y + bias.double().reshape(shape)
    return torch.relu(y) if activation == "relu" else y


def _rel(actual, expected):
    a = actual.detach().double()
    e = expected.detach().double()
    return ((a - e).abs().max() / e.abs().max().clamp_min(1e-300)).item()


def _make(shape, groups, dtype=torch.float32, affine=True, seed=0, mean=0.0, std=1.0):
    device = torch.device("cuda")
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


def _parity(shape, groups, activation=None, affine=True, seed=0, eps=EPS, label=""):
    """Forward + backward against the float64 reference.  Returns the errors."""
    x, weight, bias, grad_out = _make(shape, groups, affine=affine, seed=seed)
    assert is_supported(x, groups, weight, bias, activation), (
        f"is_supported rejected {shape} groups={groups}"
    )

    xi = x.detach().clone().requires_grad_(True)
    wi = None if weight is None else weight.detach().clone().requires_grad_(True)
    bi = None if bias is None else bias.detach().clone().requires_grad_(True)
    triton_group_norm(xi, groups, wi, bi, eps, activation).backward(grad_out)

    xd = x.detach().clone().double().requires_grad_(True)
    wd = (
        None
        if weight is None
        else weight.detach().clone().double().requires_grad_(True)
    )
    bd = None if bias is None else bias.detach().clone().double().requires_grad_(True)
    _ref64(xd, groups, wd, bd, eps, activation).backward(grad_out.double())

    errors = {"dx": _rel(xi.grad, xd.grad)}
    if wi is not None:
        errors["dw"] = _rel(wi.grad, wd.grad)
    if bi is not None:
        errors["db"] = _rel(bi.grad, bd.grad)
    xj = x.detach().clone()
    errors["y"] = _rel(
        triton_group_norm(xj, groups, weight, bias, eps, activation),
        _ref64(x, groups, weight, bias, eps, activation),
    )
    print(f"[{label or shape}] " + " ".join(f"{k}={v:.2e}" for k, v in errors.items()))
    for name, err in errors.items():
        assert err <= FP32_TOL, f"{label or shape}: {name} rel err {err:.3e}"
    return errors


# ---------------------------------------------------------------------------
# 1. the masked channel axis (MASKED_C=True) -- never reached upstream
# ---------------------------------------------------------------------------

#: ``(shape, num_groups)`` pairs for which ``_Plan.masked_c`` is True, i.e.
#: ``num_groups`` and/or ``num_channels // num_groups`` is not a power of two,
#: so ``GP``/``CGP`` over-cover the channel axis and every load, store and
#: reduction in all four kernels has to be predicated.
_MASKED_CASES = [
    ((1, 6, 4, 4, 4), 3),  # G=3 -> GP=4, CG=2
    ((1, 15, 5, 5, 5), 3),  # G=3, CG=5 -> both padded
    ((2, 12, 7, 5, 3), 3),  # N>1 with a padded group axis
    ((1, 24, 9, 9, 9), 6),  # G=6 -> GP=8, CG=4
    ((1, 20, 4, 4, 4), 5),  # G=5 -> GP=8, CG=4
    ((1, 20, 4, 4, 4), 4),  # G=4, CG=5 -> only the inner axis padded
    ((3, 20, 3, 5, 7), 5),
    ((1, 63, 5, 5, 5), 7),  # G=7, CG=9
    ((2, 63, 5, 5, 5), 7),
    ((1, 96, 5, 5, 5), 6),  # G=6, CG=16
    ((1, 10, 3, 3, 3), 10),  # G == C, neither a power of two
]


@pytest.mark.gpu
@pytest.mark.parametrize("shape,groups", _MASKED_CASES)
@pytest.mark.parametrize("activation", [None, "relu"])
def test_masked_channel_axis_parity(shape, groups, activation):
    """``MASKED_C=True``: the padded (G, C/G) tile must be fully predicated.

    ``inner = g * CG + j`` deliberately runs past the end of a voxel for the
    padding lanes, so a wrong ``cmask``/``wbm`` predicate reads (or writes) the
    *next* voxel's channels, and a wrong ``other=`` poisons the Welford sums.
    Neither shows up anywhere in the author's suite, which only ever runs
    ``num_groups=8`` over 64/128/256/2048 channels.
    """
    plan = tgn._plan(shape[0], shape[1], shape[2] * shape[3] * shape[4], groups, 0)
    assert plan.masked_c, "case is supposed to exercise the padded channel axis"
    _parity(shape, groups, activation, seed=abs(hash((shape, groups))) % 997)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,groups",
    [
        ((1, 64, 4, 4, 4), 64),  # instance norm, C/G == 1
        ((1, 7, 3, 3, 3), 7),  # instance norm, prime channel count
        ((1, 64, 4, 4, 4), 1),  # layer norm, G == 1
        ((1, 7, 3, 3, 3), 1),  # layer norm, prime channel count
        ((1, 1, 4, 4, 4), 1),  # single channel
        ((2, 1, 4, 4, 4), 1),
        ((1, 3, 5, 5, 5), 3),
    ],
)
def test_extreme_group_counts(shape, groups):
    """``num_groups == num_channels`` (instance norm) and ``== 1`` (layer norm).

    Both collapse one axis of the ``(BLOCK_S, GP, CGP)`` tile to length 1 and
    are accepted by ``is_supported``; neither appears upstream.
    """
    _parity(shape, groups, seed=abs(hash((shape, groups))) % 997)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 1, 1, 1),  # S == 1: a single voxel, far below one tile
        (2, 64, 1, 1, 1),
        (4, 64, 2, 1, 1),
        (1, 64, 1, 1, 127),  # just under a 128-voxel stats tile
        (1, 64, 1, 1, 128),  # exactly one tile
        (1, 64, 1, 1, 129),  # just over
        (1, 64, 1, 1, 257),
        (1, 64, 13, 17, 19),  # three primes
        (5, 64, 3, 3, 3),  # N not a power of two
        (1, 2048, 1, 1, 2),  # widest channel count, two voxels
    ],
)
def test_ragged_spatial_tails(shape):
    """Spatial extents that are prime, or sit just either side of a tile edge.

    The ragged tail is where ``offs_s < S - s0`` in ``_normalize_kernel`` /
    ``_dx_kernel`` and ``nvalid = min(BLOCK_S, s_end - s0)`` in the two partial
    kernels have to agree; ``cnt_t = nvalid * CG`` also has to be the *valid*
    lane count or the Welford mean is scaled wrong.
    """
    _parity(shape, 8, seed=abs(hash(shape)) % 997)


# ---------------------------------------------------------------------------
# 2. split-K scratch
# ---------------------------------------------------------------------------


def _empty_split_count(n, channels, spatial, groups):
    plan = tgn._plan(n, channels, spatial, groups, n * channels * spatial)
    return sum(1 for sp in range(plan.nsplit) if sp * plan.chunk >= spatial), plan


#: Shapes whose ``ceil(S / nsplit)`` chunking leaves at least one split with
#: ``s_begin >= S``, i.e. a program that writes an all-zero ``(cnt, mean, M2)``
#: partial that the finalize tree then has to absorb.  Found by search over the
#: plan; ``_welford_combine``'s ``cnt == 0`` guard is what makes them harmless.
_EMPTY_SPLIT_CASES = [
    ((1, 64, 1, 1, 32775), 8),
    ((2, 64, 1, 1, 32775), 8),
    ((1, 128, 1, 1, 8198), 8),
    ((1, 256, 1, 1, 2049), 8),
    ((1, 2048, 1, 1, 33), 8),
]


@pytest.mark.gpu
@pytest.mark.parametrize("shape,groups", _EMPTY_SPLIT_CASES)
def test_empty_split_slots(shape, groups):
    """A split whose whole chunk lies past ``S`` still has to combine cleanly.

    ``chunk = ceil(S / nsplit)`` can leave trailing splits entirely empty; that
    program's loop never runs, so it stores ``(0, 0, 0)``.  Chan's combine is
    only exact for those because of its ``cnt == 0`` guard, and no upstream
    shape produces one.
    """
    empties, plan = _empty_split_count(
        shape[0], shape[1], shape[2] * shape[3] * shape[4], groups
    )
    assert empties > 0, (
        f"expected an empty split for {shape}; plan has nsplit={plan.nsplit} "
        f"chunk={plan.chunk}"
    )
    _parity(shape, groups, seed=abs(hash(shape)) % 997)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,groups",
    [
        ((1, 64, 8, 8, 8), 8),
        ((2, 64, 8, 8, 8), 8),
        ((3, 15, 5, 5, 5), 3),
        ((1, 64, 1, 1, 32775), 8),  # has an empty split
        ((2, 2048, 1, 1, 1), 8),
    ],
)
def test_scratch_slots_are_all_written(shape, groups):
    """Every split-K partial slot must be written before it is read.

    ``_forward``/``_backward`` allocate ``pcnt/pmean/pm2`` and
    ``ps1/ps2/pdw/pdb`` with ``torch.empty``.  A slot that is read but never
    written would inherit whatever the caching allocator last left there --
    usually finite, plausible numbers, which no parity test can be relied on to
    catch.  Poisoning every ``torch.empty``/``empty_like`` with NaN for the
    duration of the call turns that into a hard failure, and also proves the
    output buffer itself is fully covered by the store masks.
    """
    x, weight, bias, grad_out = _make(shape, groups, seed=5)
    real_empty, real_empty_like = torch.empty, torch.empty_like

    def poisoned_empty(*args, **kwargs):
        t = real_empty(*args, **kwargs)
        return t.fill_(float("nan")) if t.is_floating_point() else t

    def poisoned_empty_like(*args, **kwargs):
        t = real_empty_like(*args, **kwargs)
        return t.fill_(float("nan")) if t.is_floating_point() else t

    xi = x.detach().clone().requires_grad_(True)
    wi = weight.detach().clone().requires_grad_(True)
    bi = bias.detach().clone().requires_grad_(True)
    torch.empty, torch.empty_like = poisoned_empty, poisoned_empty_like
    try:
        out = triton_group_norm(xi, groups, wi, bi, EPS)
        out.backward(grad_out)
    finally:
        torch.empty, torch.empty_like = real_empty, real_empty_like

    for name, t in (("y", out), ("dx", xi.grad), ("dw", wi.grad), ("db", bi.grad)):
        assert torch.isfinite(t).all(), (
            f"{name} contains NaN with poisoned scratch: a split-K slot (or an "
            f"output element) is read/returned without ever being written"
        )
    ref = _ref64(x, groups, weight, bias, EPS)
    assert _rel(out, ref) <= FP32_TOL


# ---------------------------------------------------------------------------
# 3. numerics
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("value", [0.0, 3.0, 1e3])
@pytest.mark.parametrize("eps", [1e-5, 1e-12])
def test_all_equal_input_has_exactly_zero_variance(value, eps):
    """Variance exactly 0 => ``rstd = 1/sqrt(eps)`` and ``xhat`` exactly 0.

    This is the sharpest possible statement of the Welford claim: with
    ``weight=1, bias=0`` the output must be *identically* zero, with no
    tolerance at all.  ATen's fp32 GroupNorm does not manage it (it forms the
    variance by cancellation and leaves ~1e-05 of noise at ``value=3`` and
    ~3e-03 at ``value=1e3``), which is asserted here so the comparison stays
    honest if ATen ever changes.
    """
    device = torch.device("cuda")
    shape = (2, 64, 8, 8, 8)
    x = torch.full(shape, value, device=device).contiguous(memory_format=CL)
    weight = torch.ones(64, device=device)
    bias = torch.zeros(64, device=device)

    got = triton_group_norm(x, 8, weight, bias, eps)
    assert torch.equal(got, torch.zeros_like(got)), (
        f"all-equal input must normalise to exactly 0, got max "
        f"{got.abs().max().item():.3e}"
    )
    # ... and rstd really is 1/sqrt(eps), which only the output scale can show.
    _out, _mean, rstd = torch.ops.scaffold_gn.group_norm(
        x, 8, None, None, eps, None, None
    )
    assert torch.allclose(rstd, torch.full_like(rstd, 1.0 / eps**0.5), rtol=1e-6), (
        f"rstd={rstd.flatten()[0].item():.6e} != 1/sqrt(eps)={1.0 / eps**0.5:.6e}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mean,std,naive_floor",
    [(1e4, 1e-2, 1e-1), (1e6, 1.0, 1e1)],
)
def test_welford_at_extreme_mean_to_std_ratio(mean, std, naive_floor):
    """Past the ratios the author's suite tests (mu/sigma up to 1e5).

    At ``mu/sigma = 1e6`` the ``E[x^2]-E[x]^2`` formulation is off by ~2e0 to
    ~3e2 relative while this kernel holds ~5e-04, and ATen's own fp32 kernel is
    50-70x worse than this one.  Measured on MI300A at ``[1, 256, 24^3]``.
    """
    x, weight, bias, _ = _make((1, 256, 24, 24, 24), 8, seed=37, mean=mean, std=std)
    got = triton_group_norm(x, 8, weight, bias, EPS)
    ref = _ref64(x, 8, weight, bias, EPS)
    stock = F.group_norm(x, 8, weight, bias, EPS)

    flat = x.reshape(1, 8, -1)
    mu = flat.mean(-1)
    var = (flat * flat).mean(-1) - mu * mu
    naive = (flat - mu[..., None]) / torch.sqrt(var + EPS)[..., None]
    naive = naive.reshape(x.shape) * weight.reshape(1, 256, 1, 1, 1) + bias.reshape(
        1, 256, 1, 1, 1
    )

    err, err_stock, err_naive = _rel(got, ref), _rel(stock, ref), _rel(naive, ref)
    print(
        f"[mu={mean:g} sd={std:g}] triton={err:.3e} aten={err_stock:.3e} "
        f"naive={err_naive:.3e}"
    )
    assert err_naive > naive_floor, "the naive formulation was supposed to fail here"
    assert err < err_naive / 100.0
    assert err <= err_stock, (
        f"triton {err:.3e} is worse than ATen fp32 {err_stock:.3e} at mu/sigma="
        f"{mean / std:g}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("eps", [1e-5, 1e-8, 1e-12])
def test_tiny_eps_with_tiny_variance(eps):
    """``eps`` far below the default with data whose std is ~1e-4.

    ``rstd = 1/sqrt(var + eps)`` reaches ~1e4 here, so any error in the
    variance is amplified by that factor before it reaches the output.
    """
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(7)
    x = torch.empty(1, 64, 16, 16, 16, device=device, memory_format=CL)
    x.normal_(0.0, 1e-4, generator=gen)
    weight = torch.ones(64, device=device)
    bias = torch.zeros(64, device=device)
    got = triton_group_norm(x, 8, weight, bias, eps)
    assert _rel(got, _ref64(x, 8, weight, bias, eps)) <= FP32_TOL


# ---------------------------------------------------------------------------
# 4. autograd plumbing
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    "kind", ["contiguous", "sliced", "expanded", "transposed", "channels_last"]
)
def test_grad_out_layout_variants(kind):
    """A cotangent that is not channels-last-contiguous.

    ``_group_norm_backward_op`` relayouts it; the kernels index it with the
    *input's* channels-last stride pattern, so a missed relayout silently
    permutes the gradient rather than raising.  The author's suite only ever
    feeds a channels-last-contiguous cotangent to the fast path.
    """
    shape = (2, 64, 5, 6, 7)
    x, weight, bias, _ = _make(shape, 8, seed=71)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(77)
    base = torch.empty(shape, device=device)
    base.normal_(generator=gen)
    if kind == "contiguous":
        grad_out = base.contiguous()
    elif kind == "channels_last":
        grad_out = base.contiguous(memory_format=CL)
    elif kind == "sliced":
        wide = torch.empty(
            (shape[0], shape[1], shape[2], shape[3], shape[4] * 2), device=device
        )
        wide.normal_(generator=gen)
        grad_out = wide.contiguous(memory_format=CL)[..., ::2]
    elif kind == "expanded":
        col = torch.empty((shape[0], shape[1], shape[2], shape[3], 1), device=device)
        col.normal_(generator=gen)
        grad_out = col.expand(shape)
    else:  # transposed
        swapped = torch.empty(
            (shape[0], shape[1], shape[2], shape[4], shape[3]), device=device
        )
        swapped.normal_(generator=gen)
        grad_out = swapped.contiguous(memory_format=CL).transpose(3, 4)

    xi = x.detach().clone().requires_grad_(True)
    wi = weight.detach().clone().requires_grad_(True)
    bi = bias.detach().clone().requires_grad_(True)
    triton_group_norm(xi, 8, wi, bi, EPS).backward(grad_out)

    xd = x.detach().clone().double().requires_grad_(True)
    wd = weight.detach().clone().double().requires_grad_(True)
    bd = bias.detach().clone().double().requires_grad_(True)
    _ref64(xd, 8, wd, bd, EPS).backward(grad_out.double())

    assert _rel(xi.grad, xd.grad) <= FP32_TOL
    assert _rel(wi.grad, wd.grad) <= FP32_TOL
    assert _rel(bi.grad, bd.grad) <= FP32_TOL
    # d_input keeps the *input's* format regardless of the cotangent's.
    assert xi.grad.is_contiguous(memory_format=CL)


@pytest.mark.gpu
def test_affine_parameters_may_be_non_contiguous_views():
    """``weight``/``bias`` sliced out of a bigger parameter tensor.

    ``is_supported`` only checks rank, numel, device and dtype, so a strided or
    offset 1-D parameter reaches the op, which is why it calls ``.contiguous()``
    on both.  Nothing upstream tests that.
    """
    x, _weight, _bias, _ = _make((1, 64, 4, 4, 4), 8, seed=83)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(83)
    strided = torch.empty(128, device=device)
    strided.normal_(1.0, 0.25, generator=gen)
    weight = strided[::2]  # stride 2
    pack = torch.empty(4, 64, device=device)
    pack.normal_(0.0, 0.25, generator=gen)
    bias = pack[2]  # storage offset
    assert not weight.is_contiguous()
    assert is_supported(x, 8, weight, bias)
    got = triton_group_norm(x, 8, weight, bias, EPS)
    assert _rel(got, _ref64(x, 8, weight, bias, EPS)) <= FP32_TOL


@pytest.mark.gpu
@pytest.mark.parametrize("lo,hi", [(0, 2), (1, 3), (3, 4)])
def test_channels_last_views_with_a_storage_offset(lo, hi):
    """A batch slice of a bigger channels-last tensor stays channels-last
    contiguous but has a non-zero storage offset -- the kernels must address
    from ``data_ptr()``, not from the storage base."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(89)
    big = torch.empty((4, 64, 5, 6, 7), device=device, memory_format=CL)
    big.normal_(generator=gen)
    weight = torch.empty(64, device=device)
    weight.normal_(1.0, 0.25, generator=gen)
    bias = torch.empty(64, device=device)
    bias.normal_(0.0, 0.25, generator=gen)
    view = big[lo:hi]
    assert view.is_contiguous(memory_format=CL) and is_supported(view, 8, weight, bias)
    got = triton_group_norm(view, 8, weight, bias, EPS)
    assert _rel(got, _ref64(view, 8, weight, bias, EPS)) <= FP32_TOL


@pytest.mark.gpu
def test_double_backward_raises_instead_of_returning_garbage():
    """Higher-order gradients are *not* supported, and must say so.

    ``scaffold_gn::group_norm_backward`` has no autograd formula of its own, so
    a second ``torch.autograd.grad`` through the kernel raises.  Stock
    ``F.group_norm`` supports double backward, so this is a real (if narrow)
    behavioural difference from the op it replaces -- anything that needs a
    gradient penalty or a Hessian-vector product cannot use this kernel.  The
    test pins "raises loudly", which is the safe half of the story.
    """
    x, weight, bias, grad_out = _make((1, 64, 4, 4, 4), 8, seed=97)
    xi = x.detach().clone().requires_grad_(True)
    wi = weight.detach().clone().requires_grad_(True)
    bi = bias.detach().clone().requires_grad_(True)
    y = triton_group_norm(xi, 8, wi, bi, EPS)
    (gx,) = torch.autograd.grad(y, xi, grad_out, create_graph=True)
    with pytest.raises(RuntimeError, match="no autograd formula was registered"):
        torch.autograd.grad(gx.sum(), xi)

    # ... and stock really does support it, so this is a divergence not a law.
    xr = x.detach().clone().requires_grad_(True)
    yr = F.group_norm(xr, 8, weight, bias, EPS)
    (gxr,) = torch.autograd.grad(yr, xr, grad_out, create_graph=True)
    (ggr,) = torch.autograd.grad(gxr.sum(), xr)
    assert torch.isfinite(ggr).all()


# ---------------------------------------------------------------------------
# 5. fake / meta kernel
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("out_dtype", [None, torch.float32])
@pytest.mark.parametrize("has_w,has_b", [(True, True), (False, False), (True, False)])
def test_fake_forward_matches_real_in_every_branch(dtype, out_dtype, has_w, has_b):
    """The fake kernel must promise the real shape, dtype, stride *and* device.

    A meta mismatch is invisible in eager and silently corrupts
    ``torch.compile``; the author's suite spot-checks two combinations, this
    walks the whole cross product of dtype x out_dtype override x affine.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    device = torch.device("cuda")
    shape = (2, 64, 5, 6, 7)
    x = torch.empty(shape, device=device, dtype=dtype, memory_format=CL).normal_()
    weight = torch.randn(64, device=device, dtype=dtype) if has_w else None
    bias = torch.randn(64, device=device, dtype=dtype) if has_b else None

    real = torch.ops.scaffold_gn.group_norm(x, 8, weight, bias, EPS, "relu", out_dtype)
    with FakeTensorMode() as mode:
        args = [None if t is None else mode.from_tensor(t) for t in (x, weight, bias)]
        fake = torch.ops.scaffold_gn.group_norm(
            args[0], 8, args[1], args[2], EPS, "relu", out_dtype
        )
    for i, (r, f) in enumerate(zip(real, fake)):
        assert r.shape == f.shape, f"out[{i}] shape"
        assert r.dtype == f.dtype, f"out[{i}] dtype {r.dtype} != {f.dtype}"
        assert r.stride() == f.stride(), f"out[{i}] stride {r.stride()} != {f.stride()}"
        assert r.device.type == f.device.type, f"out[{i}] device"


@pytest.mark.gpu
@pytest.mark.parametrize(
    "layout", ["contiguous", "channels_last", "sliced", "degenerate"]
)
@pytest.mark.parametrize("has_w,has_b", [(True, True), (False, False), (True, False)])
def test_fake_backward_matches_real_in_every_branch(layout, has_w, has_b):
    """The backward's fake kernel must promise what the real op returns.

    The real op relayouts a non-channels-last ``input`` and *always* returns a
    channels-last ``d_input``; ``torch.empty_like(input)`` would instead
    preserve the input's own format, so for a plain contiguous NCDHW input the
    two disagree ((13440, 1, 2688, 448, 64) against (13440, 210, 42, 7, 1)).  A
    meta mismatch is invisible in eager and silently corrupts
    ``torch.compile``, so every branch of the promise -- both layouts, a
    non-contiguous view, the shape where the two formats coincide, and each
    affine combination -- is checked here rather than only the CL case.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    device = torch.device("cuda")
    if layout == "degenerate":
        shape = (2, 64, 1, 1, 1)  # contiguous *is* channels_last_3d here
        x = torch.randn(shape, device=device)
    else:
        shape = (2, 64, 5, 6, 7)
        if layout == "contiguous":
            x = torch.randn(shape, device=device)
        elif layout == "channels_last":
            x = torch.randn(shape, device=device).contiguous(memory_format=CL)
        else:  # sliced: neither contiguous nor channels-last contiguous
            x = torch.randn((2, 64, 5, 6, 14), device=device)[..., ::2]
    grad_out = torch.randn(shape, device=device).contiguous(memory_format=CL)
    weight = torch.randn(64, device=device) if has_w else None
    bias = torch.randn(64, device=device) if has_b else None
    mean = torch.zeros(2, 8, device=device)
    rstd = torch.ones(2, 8, device=device)

    real = torch.ops.scaffold_gn.group_norm_backward(
        grad_out, x, weight, bias, mean, rstd, 8, None
    )
    with FakeTensorMode() as mode:
        a = [
            None if t is None else mode.from_tensor(t)
            for t in (grad_out, x, weight, bias, mean, rstd)
        ]
        fake = torch.ops.scaffold_gn.group_norm_backward(
            a[0], a[1], a[2], a[3], a[4], a[5], 8, None
        )
    names = ("d_input", "d_weight", "d_bias")
    for name, r, f in zip(names, real, fake):
        assert r.shape == f.shape, f"{name} shape {r.shape} != {f.shape}"
        assert r.dtype == f.dtype, f"{name} dtype {r.dtype} != {f.dtype}"
        assert r.stride() == f.stride(), f"{name} stride {r.stride()} != {f.stride()}"
        assert r.device.type == f.device.type, f"{name} device"
    assert real[0].is_contiguous(memory_format=CL)


@pytest.mark.gpu
def test_mean_and_rstd_are_not_silently_differentiable():
    """``mean``/``rstd`` are backward state, so they must refuse, not lie.

    They are documented as "not differentiable".  Before they were marked as
    such, they came back with ``requires_grad=True`` and differentiating
    through them *succeeded*: autograd materialised an all-zero cotangent for
    the unused ``out``, ran the entire backward (a full-size zeros allocation
    plus four kernels) and returned zeros -- a plausible wrong answer where the
    true value is ~6e-04.  ``ctx.mark_non_differentiable`` turns that into an
    error, which is the only safe outcome short of a real formula.
    """
    device = torch.device("cuda")
    shape = (2, 64, 5, 6, 7)
    x = torch.empty(shape, device=device, memory_format=CL).normal_()
    xi = x.clone().requires_grad_(True)
    out, mean, rstd = torch.ops.scaffold_gn.group_norm(
        xi, 8, None, None, EPS, None, None
    )
    assert out.requires_grad, "the forward output must still be differentiable"
    assert not mean.requires_grad, "mean must be marked non-differentiable"
    assert not rstd.requires_grad, "rstd must be marked non-differentiable"
    for name, t in (("mean", mean), ("rstd", rstd)):
        with pytest.raises(RuntimeError, match="does not require grad"):
            torch.autograd.grad(t.sum(), xi)
        assert xi.grad is None, f"differentiating {name} left a gradient behind"
    # The value that used to come back silently wrong is genuinely non-zero,
    # so "returns zeros" was never defensible as an answer.
    xd = x.clone().double().requires_grad_(True)
    (want,) = torch.autograd.grad(xd.reshape(2, 8, -1).mean(-1).sum(), xd)
    assert want.abs().max() > 0


# ---------------------------------------------------------------------------
# 6. contract / drop-in divergences
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,groups",
    [((1, 64, 1, 1, 1), 8), ((2, 64, 1, 1, 1), 8), ((1, 1, 4, 5, 6), 1)],
)
def test_is_supported_accepts_layout_ambiguous_contiguous_input(shape, groups):
    """``is_supported`` is *not* simply "False for contiguous input".

    For shapes whose spatial or channel extents are all 1 the contiguous and
    channels-last-3d stride patterns coincide, so a plain ``torch.randn``
    tensor is accepted by the fast path.  That is benign -- the two layouts are
    the same bytes -- but it means callers cannot use ``is_supported`` as a
    layout *classifier*.  Pinned here so the behaviour is deliberate.
    """
    device = torch.device("cuda")
    x = torch.randn(shape, device=device)  # never asked for channels_last
    assert x.is_contiguous()
    assert x.is_contiguous(memory_format=CL)
    assert is_supported(x, groups) is True
    got = triton_group_norm(x, groups, None, None, EPS)
    assert _rel(got, _ref64(x, groups, None, None, EPS)) <= FP32_TOL


@pytest.mark.gpu
@pytest.mark.parametrize("shape,groups", [((1, 8, 1, 1, 1), 8), ((1, 1, 1, 1, 1), 1)])
def test_one_value_per_channel_matches_stock_rejection(shape, groups):
    """``N*(C/G)*D*H*W == 1`` is a shape ``F.group_norm`` refuses to run.

    The kernel *can* compute it (every group has zero variance, so the answer
    is ``bias``), and it used to: ``is_supported`` returned True and
    ``triton_group_norm`` returned a value where the op it is a drop-in for
    raises ``ValueError``.  A caller branching on ``is_supported`` would then
    get a different answer from the reference path, which is worse than being
    slower, so all three of ``is_supported``, the public wrapper and the raw op
    now reject it the same way stock does.
    """
    device = torch.device("cuda")
    x = torch.empty(shape, device=device, memory_format=CL).normal_()
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        F.group_norm(x, groups, None, None, EPS)
    assert is_supported(x, groups) is False, (
        "is_supported accepts a shape F.group_norm rejects"
    )
    # The public wrapper reaches the same rejection through its fallback...
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        triton_group_norm(x, groups, None, None, EPS)
    # ... and the op itself refuses too, for anyone calling it directly.
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        torch.ops.scaffold_gn.group_norm(x, groups, None, None, EPS, None, None)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,groups",
    [((2, 8, 1, 1, 1), 8), ((1, 8, 2, 1, 1), 8), ((1, 16, 1, 1, 1), 8)],
)
def test_neighbours_of_the_one_value_per_channel_shape_are_still_served(shape, groups):
    """The rejection must be exactly stock's, not a shape family around it.

    ``_verify_batch_size`` rejects ``N*(C/G)*spatial == 1`` and nothing else, so
    bumping *any one* of N, C/G or the spatial extent to 2 has to come back to
    the fast path -- including ``(2, 8, 1, 1, 1)``, which still has a single
    element per group.
    """
    device = torch.device("cuda")
    x = torch.empty(shape, device=device, memory_format=CL).normal_()
    F.group_norm(x, groups, None, None, EPS)  # stock accepts it
    assert is_supported(x, groups) is True
    got = triton_group_norm(x, groups, None, None, EPS)
    assert _rel(got, _ref64(x, groups, None, None, EPS)) <= FP32_TOL


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,groups,activation",
    [
        ((2, 8, 1, 1, 1), 8, None),
        ((2, 8, 1, 1, 1), 8, "relu"),
        ((1, 8, 2, 1, 1), 8, None),  # 2 elements per group: NOT the degenerate case
        ((3, 16, 1, 1, 1), 16, None),
    ],
)
def test_single_element_group_gradient_is_exactly_zero(shape, groups, activation):
    """One element per group => y is constant in x => dx must be identically 0.

    ``mean == x`` and ``var == 0`` identically, so ``xhat`` is the constant 0
    and nothing downstream depends on ``x``.  ``_dx_kernel`` used to answer
    2.2e-05 instead: the compiler contracts ``dy*w - c1`` to
    ``fma(dy, w, -c1)`` while ``c1`` was accumulated from the *rounded*
    product, so what survives is the product's rounding error (7.0e-08, well
    under one ulp of ``dyw``), amplified by ``rstd = 1/sqrt(eps) = 316``.
    ``_backward`` now recognises the degenerate case and returns the exact
    zero; ATen, on the shapes where it will run at all, leaves ~3e-05 there.

    The ``(1, 8, 2, 1, 1)`` case is the control: two elements per group, so the
    gradient is *not* identically zero and the kernel must not zero it.
    """
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(3)
    channels = shape[1]
    x = torch.empty(shape, device=device, memory_format=CL)
    x.normal_(generator=gen)
    weight = torch.empty(channels, device=device)
    weight.normal_(1.0, 0.25, generator=gen)
    bias = torch.empty(channels, device=device)
    bias.normal_(0.0, 0.25, generator=gen)
    grad_out = torch.empty(shape, device=device, memory_format=CL)
    grad_out.normal_(generator=gen)

    assert is_supported(x, groups, weight, bias, activation)
    xi = x.clone().requires_grad_(True)
    wi = weight.clone().requires_grad_(True)
    bi = bias.clone().requires_grad_(True)
    triton_group_norm(xi, groups, wi, bi, EPS, activation).backward(grad_out)

    xd = x.clone().double().requires_grad_(True)
    wd = weight.clone().double().requires_grad_(True)
    bd = bias.clone().double().requires_grad_(True)
    _ref64(xd, groups, wd, bd, EPS, activation).backward(grad_out.double())

    if channels // groups * shape[2] * shape[3] * shape[4] == 1:
        assert torch.equal(xi.grad, torch.zeros_like(xi.grad)), (
            f"dx should be exactly 0, got {xi.grad.abs().max().item():.3e}"
        )
        assert xd.grad.abs().max() == 0, "the float64 reference disagrees"
        # d_weight is exactly 0 too (xhat is exactly 0); d_bias is not.
        assert torch.equal(wi.grad, torch.zeros_like(wi.grad))
        assert _rel(bi.grad, bd.grad) <= FP32_TOL
    else:
        assert xd.grad.abs().max() > 0, "control case is supposed to be non-trivial"
        assert xi.grad.abs().max() > 0, "the kernel zeroed a non-degenerate gradient"
        # A *two*-element group is merely ill-conditioned, not degenerate:
        # xhat is +-1/sqrt(1+eps/var) and dx is a difference of near-equal
        # terms, so every fp32 implementation loses digits here -- 4.9e-04
        # relative for this kernel and 1.4e-04 for ATen on this input.  The
        # bound is therefore loose against float64, and tight against ATen,
        # which suffers the same cancellation.
        assert _rel(xi.grad, xd.grad) <= 1e-3
        xa = x.clone().requires_grad_(True)
        F.group_norm(xa, groups, weight, bias, EPS).backward(grad_out)
        assert _rel(xi.grad, xa.grad) <= 1e-3


# ---------------------------------------------------------------------------
# 7. composition
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("activation", [None, "relu"])
def test_torch_compile_with_dynamic_shapes(activation):
    """``dynamic=True`` as well as the author's ``dynamic=False``.

    With dynamic shapes the fake kernel is invoked on *symbolic* sizes, so a
    shape/stride promise that only happens to hold for a concrete size shows up
    here and nowhere else.  ``fullgraph=True`` is the no-graph-break assertion.
    """
    x, weight, bias, grad_out = _make((2, 64, 6, 6, 6), 8, seed=59)

    def fn(x, weight, bias):
        return triton_group_norm(x, 8, weight, bias, EPS, activation) * 2.0

    def run(f):
        xi = x.detach().clone().requires_grad_(True)
        wi = weight.detach().clone().requires_grad_(True)
        bi = bias.detach().clone().requires_grad_(True)
        out = f(xi, wi, bi)
        out.backward(grad_out)
        return out.detach(), xi.grad, wi.grad, bi.grad

    torch._dynamo.reset()
    eager = run(fn)
    compiled = run(torch.compile(fn, fullgraph=True, dynamic=True))
    for name, a, b in zip(("y", "dx", "dweight", "dbias"), compiled, eager):
        assert torch.equal(a, b), f"dynamic-shape compile differs from eager in {name}"
    assert compiled[0].is_contiguous(memory_format=CL)


# ---------------------------------------------------------------------------
# 8. determinism, across processes
# ---------------------------------------------------------------------------

_DETERMINISM_SCRIPT = textwrap.dedent(
    """
    import hashlib, sys, torch
    from ScaFFold.unet.triton_group_norm import triton_group_norm
    CL = torch.channels_last_3d

    def h(t):
        b = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(b).hexdigest()

    if sys.argv[1] == "warm":
        # Different JIT order, different lru_cache occupancy, different
        # allocator state and different free memory before the real work.
        junk = []
        for shape, g in (((3, 128, 7, 7, 7), 8), ((1, 15, 5, 5, 5), 3)):
            a = torch.empty(shape, device="cuda", memory_format=CL).normal_()
            triton_group_norm(a, g, None, None, 1e-5, "relu")
            junk.append(torch.empty(1 << 25, device="cuda"))
        del junk
        torch.cuda.empty_cache()

    for shape, groups, act, dtype in (
        ((2, 128, 9, 11, 13), 8, None, torch.float32),
        ((2, 64, 6, 7, 8), 8, "relu", torch.bfloat16),
        ((3, 15, 5, 5, 5), 3, None, torch.float32),
    ):
        gen = torch.Generator(device="cuda").manual_seed(31)
        x = torch.empty(shape, device="cuda", dtype=dtype, memory_format=CL)
        x.normal_(generator=gen)
        w = torch.empty(shape[1], device="cuda", dtype=dtype)
        w.normal_(1.0, 0.25, generator=gen)
        b = torch.empty(shape[1], device="cuda", dtype=dtype)
        b.normal_(0.0, 0.25, generator=gen)
        go = torch.empty(shape, device="cuda", dtype=dtype, memory_format=CL)
        go.normal_(generator=gen)
        xi = x.clone().requires_grad_(True)
        wi = w.clone().requires_grad_(True)
        bi = b.clone().requires_grad_(True)
        y = triton_group_norm(xi, groups, wi, bi, 1e-5, act)
        y.backward(go)
        print(shape, groups, act, dtype,
              h(y), h(xi.grad), h(wi.grad), h(bi.grad), flush=True)
    """
)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_bitwise_determinism_across_processes():
    """Process-to-process bitwise reproducibility, which is half the claim.

    ``test_bitwise_determinism`` upstream only calls the kernel twice in *one*
    process, where the plan is already memoised and the JIT cache already warm.
    This runs three fresh interpreters -- one of which first JITs other shapes,
    churns the caching allocator and changes how much memory is free -- and
    compares SHA-256 of the raw output bytes.  Anything that made the split
    count, tile size or launch geometry depend on device state rather than on
    the shape would show up only here.
    """
    outputs = []
    for mode in ("plain", "plain", "warm"):
        result = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SCRIPT, mode],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=600,
        )
        assert result.returncode == 0, result.stderr[-3000:]
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1], "two identical processes disagree"
    assert outputs[0] == outputs[2], (
        "a process that JITted other shapes first disagrees:\n"
        f"{outputs[0]}\n--- vs ---\n{outputs[2]}"
    )
    assert outputs[0].count("\n") >= 3


# ---------------------------------------------------------------------------
# 9. multi-device
# ---------------------------------------------------------------------------

_DEVICE_GUARD_SCRIPT = textwrap.dedent(
    """
    import sys, torch
    import torch.nn.functional as F
    from ScaFFold.unet.triton_group_norm import triton_group_norm
    CL = torch.channels_last_3d
    torch.cuda.set_device(0)                      # current device = 0
    other = "cuda:1"
    g = torch.Generator(device=other).manual_seed(1)
    x = torch.empty((1, 64, 4, 4, 4), device=other, memory_format=CL)
    x.normal_(generator=g)
    w = torch.empty(64, device=other); w.normal_(1.0, 0.25, generator=g)
    b = torch.empty(64, device=other); b.normal_(0.0, 0.25, generator=g)
    go = torch.empty((1, 64, 4, 4, 4), device=other, memory_format=CL)
    go.normal_(generator=g)

    def rel(a, e):
        return ((a.double() - e.double()).abs().max()
                / e.double().abs().max().clamp_min(1e-30)).item()

    # ATen carries a DeviceGuard, so this is the behaviour to match.
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    br = b.clone().requires_grad_(True)
    F.group_norm(xr, 8, wr, br, 1e-5).backward(go)

    xi = x.clone().requires_grad_(True)
    wi = w.clone().requires_grad_(True)
    bi = b.clone().requires_grad_(True)
    y = triton_group_norm(xi, 8, wi, bi, 1e-5)    # tensors on 1, current is 0
    y.backward(go)                                # ... and so is the backward
    torch.cuda.synchronize()
    assert torch.cuda.current_device() == 0, "the guard leaked the device"
    for name, got, want in (("y", y, F.group_norm(xr.detach(), 8, w, b, 1e-5)),
                            ("dx", xi.grad, xr.grad),
                            ("dw", wi.grad, wr.grad),
                            ("db", bi.grad, br.grad)):
        assert got.device == torch.device(other), f"{name} on {got.device}"
        e = rel(got, want)
        assert e < 1e-4, f"{name}: rel err {e}"
    print("OK")
    """
)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_kernel_runs_on_the_inputs_device_not_the_current_one():
    """Tensors on cuda:1 while cuda:0 is current, forward *and* backward.

    A Triton launch goes to whatever device is *current*, so without a device
    guard the kernel dereferences another device's pointers and the process
    dies with ``Memory access fault by GPU node-N``.  ``F.group_norm`` carries
    ATen's ``DeviceGuard`` and handles the identical call, so this is a
    divergence from the op being replaced, not a PyTorch limitation.

    Run in a subprocess because the failure mode is an unrecoverable GPU memory
    fault, which would take the whole pytest session with it.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 visible CUDA devices")
    result = subprocess.run(
        [sys.executable, "-c", _DEVICE_GUARD_SCRIPT],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=480,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        f"returncode={result.returncode}\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-2000:]}"
    )


@pytest.mark.gpu
def test_device_guard_helper_is_a_no_op_on_the_current_device():
    """The guard must be free on the hot path and real off it.

    ``_device_guard`` skips ``torch.cuda.device`` when the tensor already lives
    on the current device (1.55 us against 0.51 us of host time per call, which
    is 0.5% of the two smallest scale-8 shapes' 0.65 ms fwd+bwd because they
    are host-dispatch bound).  This pins both halves of that shortcut so a
    future edit cannot quietly turn it into "no guard at all"; the multi-device
    behaviour itself is covered by the subprocess test above.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    guard = tgn._device_guard(device)
    assert guard is tgn._NO_GUARD, "should not build a guard for the current device"
    # Constructing a guard for another index does not touch that device.
    other = torch.device("cuda", device.index + 1)
    assert tgn._device_guard(other) is not tgn._NO_GUARD, (
        "a foreign device must get a real guard"
    )


# ---------------------------------------------------------------------------
# 10. addressing at the int32 boundary
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_int32_addressing_at_its_documented_maximum():
    """``numel = INT32_MAX - 127`` with N=2, i.e. ``plan.int64 is False``.

    The author's suite tests the shape *above* the switch
    (``test_correct_above_int32_max_elements``) but never the largest shape the
    **int32** path itself has to serve, which is where a missing term in the
    ``numel + channels > INT32_MAX`` guard would bite.  ``65 * 63 * 4097`` is
    ``2^24 - 1`` voxels, so nothing about the extents is a power of two.

    Verified without materialising an NCDHW reference: the statistics are
    checked against a chunked float64 reduction over the physical (N, S, C)
    view, and the output against an elementwise recomputation done per batch
    item (a truncated offset wraps partway through, so sample 1 would break
    while sample 0 did not).
    """
    device = torch.device("cuda")
    shape = (2, 64, 65, 63, 4097)
    n, channels = shape[0], shape[1]
    spatial = shape[2] * shape[3] * shape[4]
    numel = n * channels * spatial
    assert numel == 2**31 - 128, numel

    plan = tgn._plan(n, channels, spatial, 8, numel)
    assert plan.int64 is False, "this shape is supposed to use the int32 path"

    free, total = torch.cuda.mem_get_info()
    needed = 4 * numel * 4
    if free < needed:
        pytest.skip(
            f"needs ~{needed / 2**30:.0f} GiB free, device has "
            f"{free / 2**30:.0f} GiB of {total / 2**30:.0f} GiB"
        )

    gen = torch.Generator(device=device).manual_seed(53)
    x = torch.empty(shape, device=device, memory_format=CL)
    x.normal_(generator=gen)
    weight = torch.randn(channels, device=device, generator=gen)
    bias = torch.randn(channels, device=device, generator=gen)
    out, mean, rstd = torch.ops.scaffold_gn.group_norm(
        x, 8, weight, bias, EPS, None, None
    )

    group_channels = channels // 8
    flat = x.permute(0, 2, 3, 4, 1).reshape(n, spatial, channels)  # no copy
    chunk = 1 << 20
    for i in range(n):
        acc = torch.zeros(8, dtype=torch.float64, device=device)
        for s in range(0, spatial, chunk):
            acc += (
                flat[i, s : s + chunk]
                .double()
                .reshape(-1, 8, group_channels)
                .sum(dim=(0, 2))
            )
        mu = acc / (spatial * group_channels)
        acc2 = torch.zeros(8, dtype=torch.float64, device=device)
        for s in range(0, spatial, chunk):
            d = (
                flat[i, s : s + chunk].double().reshape(-1, 8, group_channels)
                - mu[None, :, None]
            )
            acc2 += (d * d).sum(dim=(0, 2))
        var = acc2 / (spatial * group_channels)
        assert _rel(mean[i], mu) <= 1e-5, f"sample {i} mean"
        assert _rel(rstd[i], 1.0 / torch.sqrt(var + EPS)) <= 1e-5, f"sample {i} rstd"
    del flat

    mv = (
        mean.reshape(n, 8, 1).expand(n, 8, group_channels).reshape(n, channels, 1, 1, 1)
    )
    rv = (
        rstd.reshape(n, 8, 1).expand(n, 8, group_channels).reshape(n, channels, 1, 1, 1)
    )
    for i in range(n):
        recomputed = (x[i : i + 1] - mv[i : i + 1]) * rv[i : i + 1] * weight.reshape(
            1, channels, 1, 1, 1
        ) + bias.reshape(1, channels, 1, 1, 1)
        # fp32 subtraction of two near-equal fp32 values is exact.
        err = (out[i : i + 1] - recomputed).abs().max().item()
        scale = recomputed.abs().max().item()
        print(f"[int32-max sample {i}] elementwise rel err {err / scale:.3e}")
        assert err / scale < 1e-5, f"sample {i}"
        del recomputed
    del x, out, mean, rstd, mv, rv
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 11. coverage the mutation sweep found thin
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _force_int64_addressing():
    """Make every plan take the int64 tile-base path, whatever the shape.

    ``_Plan`` sets ``int64 = numel + channels > _INT32_MAX``, so dropping the
    threshold turns the wide path on for a shape that fits in a few MiB.  The
    plan cache is keyed on the shape, not on the threshold, so it has to be
    cleared on the way in *and* on the way out.
    """
    real = tgn._INT32_MAX
    tgn._plan.cache_clear()
    tgn._INT32_MAX = -1
    try:
        yield
    finally:
        tgn._INT32_MAX = real
        tgn._plan.cache_clear()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,groups",
    [
        ((2, 64, 5, 6, 7), 8),  # ragged tail, several splits
        ((1, 2048, 6, 6, 6), 8),  # widest channel count
        ((3, 15, 5, 5, 5), 3),  # masked channel axis as well
        ((1, 64, 1, 1, 32775), 8),  # has an empty split
    ],
)
@pytest.mark.parametrize("activation", [None, "relu"])
def test_int64_addressing_path_is_behaviourally_correct(shape, groups, activation):
    """Run the ``INT64=True`` branch of all seven kernels on a small shape.

    ``INT64`` is a ``tl.constexpr``, so the wide and narrow paths are *different
    compiled kernels*; only shapes above 2^31 elements reach the wide one
    naturally, and the one test that does is ``@pytest.mark.slow`` and needs
    8 GiB.  In a default ``-m "not slow"`` run the int64 branch therefore has no
    behavioural coverage at all -- forcing ``self.int64`` gives it some for the
    price of a few MiB.

    The two paths differ only in the *type* of the scalar tile base, so the
    results must be **bitwise** identical, which is a far sharper assertion than
    a tolerance and would catch a widened offset that lost or duplicated a tile.
    """
    x, weight, bias, grad_out = _make(shape, groups, seed=abs(hash(shape)) % 997)

    def run():
        xi = x.detach().clone().requires_grad_(True)
        wi = weight.detach().clone().requires_grad_(True)
        bi = bias.detach().clone().requires_grad_(True)
        out = triton_group_norm(xi, groups, wi, bi, EPS, activation)
        out.backward(grad_out)
        return out.detach(), xi.grad, wi.grad, bi.grad

    plan32 = tgn._plan(
        shape[0], shape[1], shape[2] * shape[3] * shape[4], groups, x.numel()
    )
    assert plan32.int64 is False, "shape is supposed to fit the int32 path"
    narrow = run()

    with _force_int64_addressing():
        plan64 = tgn._plan(
            shape[0], shape[1], shape[2] * shape[3] * shape[4], groups, x.numel()
        )
        assert plan64.int64 is True, "the int64 path was not forced on"
        wide = run()

    for name, a, b in zip(("y", "dx", "dweight", "dbias"), wide, narrow):
        assert torch.equal(a, b), f"int64 path differs from int32 in {name}"
    # ... and both are actually right, not identically wrong.
    ref = _ref64(x, groups, weight, bias, EPS, activation)
    assert _rel(wide[0], ref) <= FP32_TOL


#: ``(shape, groups)`` whose split-K partials have *unequal* counts, because
#: ``chunk = ceil(S / nsplit)`` does not divide ``S``.  Chan's combine weights
#: the delta by ``cnt_b / (cnt_a + cnt_b)``; with equal counts every level of
#: the reduction tree has ``cnt_a == cnt_b``, so weighting by the wrong one is
#: invisible.  Only a ragged (or empty) trailing split exposes it -- which is
#: why the mutation sweep killed that bug with exactly two parametrizations of
#: one test upstream.
_UNEVEN_SPLIT_CASES = [
    ((2, 64, 9, 7, 5), 8),
    ((1, 2048, 6, 6, 6), 8),
    ((1, 64, 1, 1, 32775), 8),
    ((1, 256, 1, 1, 2049), 8),
    ((2, 128, 11, 13, 17), 8),
    ((2, 15, 9, 9, 9), 3),  # masked channel axis as well
    ((1, 20, 17, 17, 17), 5),  # 16 splits, trailing split 15 voxels short
]


@pytest.mark.gpu
@pytest.mark.parametrize("shape,groups", _UNEVEN_SPLIT_CASES)
@pytest.mark.parametrize("eps", [1e-5, 0.5])
def test_group_statistics_match_float64_with_uneven_splits(shape, groups, eps):
    """Assert ``mean``/``rstd`` themselves, not just the output they feed.

    Two things hide inside the output's 1e-4 tolerance and show up here:

    * **the Welford merge.**  The shapes above all have at least one split with
      a different element count from its neighbours, which is the only
      configuration in which mis-weighting Chan's delta changes the answer.
    * **where ``eps`` goes.**  Every parity test in both files uses
      ``eps=1e-5`` against a variance of ~1, where ``1/sqrt(var+eps)`` and
      ``1/(sqrt(var)+eps)`` agree to ~1e-5 -- inside that tolerance.  At
      ``eps=0.5`` they are 0.816 and 0.667, a 22% difference that no tolerance
      can absorb.
    """
    spatial = shape[2] * shape[3] * shape[4]
    plan = tgn._plan(shape[0], shape[1], spatial, groups, shape[0] * shape[1] * spatial)
    counts = {
        max(0, min(sp * plan.chunk + plan.chunk, spatial) - sp * plan.chunk)
        for sp in range(plan.nsplit)
    }
    assert plan.nsplit > 1 and len(counts) > 1, (
        f"{shape} was supposed to give unequal split counts; nsplit="
        f"{plan.nsplit} chunk={plan.chunk} counts={sorted(counts)}"
    )

    x, _weight, _bias, _ = _make(shape, groups, seed=abs(hash(shape)) % 997)
    _out, mean, rstd = torch.ops.scaffold_gn.group_norm(
        x, groups, None, None, eps, None, None
    )
    flat = x.double().reshape(shape[0], groups, -1)
    mean64 = flat.mean(-1)
    var64 = ((flat - mean64[..., None]) ** 2).mean(-1)
    rstd64 = 1.0 / torch.sqrt(var64 + eps)
    assert _rel(mean, mean64) <= 1e-5, "group mean"
    assert _rel(rstd, rstd64) <= 1e-5, "group rstd (eps placement / Welford merge)"


@pytest.mark.gpu
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_welford_correction_recovers_rstd_in_a_single_tile_reduction(groups, seed):
    """The third reduction pass (``corr``) is load-bearing, and here is where.

    ``mean0 = sum(x)/n`` loses digits in proportion to the tile's element count
    times ``mu/sigma``; ``corr = sum(x-mean0)/n`` recovers them, and ``M2`` is
    then formed around the corrected mean.  The effect is largest when one tile
    carries a whole group's reduction, which is this shape: ``block_s_stats``
    covers all 128 voxels and ``nsplit == 1``, so 8192/``G`` elements per group
    go through a single ``mean0``.

    At ``mu/sigma = 1e6`` the correction is worth **216x** (G=1), **580x**
    (G=2) and **1472x** (G=4) on the relative error of ``rstd`` -- measured by
    running a copy of this module with the term deleted.  Corrected lands at
    ~1e-07 for every seed and group count; without it, at 2.1e-05 to 1.6e-04.
    The 1e-06 ceiling below sits an order of magnitude above the first and an
    order of magnitude below the second.

    The *output* is not a witness for this: ``y`` moves by at most ~1.4x with
    or without the term, because it is dominated by the fp32 representation of
    the mean.  That is why this asserts ``rstd`` directly.
    """
    device = torch.device("cuda")
    shape = (2, 64, 8, 4, 4)
    spatial = shape[2] * shape[3] * shape[4]
    plan = tgn._plan(shape[0], shape[1], spatial, groups, shape[0] * shape[1] * spatial)
    assert plan.nsplit == 1 and plan.block_s_stats >= spatial, (
        f"case is supposed to be a single-tile reduction; nsplit={plan.nsplit} "
        f"block_s_stats={plan.block_s_stats} spatial={spatial}"
    )

    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.empty(shape, device=device, memory_format=CL)
    x.normal_(1e4, 1e-2, generator=gen)  # mu/sigma = 1e6
    _out, mean, rstd = torch.ops.scaffold_gn.group_norm(
        x, groups, None, None, EPS, None, None
    )
    flat = x.double().reshape(shape[0], groups, -1)
    mean64 = flat.mean(-1)
    var64 = ((flat - mean64[..., None]) ** 2).mean(-1)
    err = _rel(rstd, 1.0 / torch.sqrt(var64 + EPS))
    print(f"[corr G={groups} seed={seed}] rstd rel err {err:.3e}")
    assert err <= 1e-6, (
        f"rstd rel err {err:.3e} at mu/sigma=1e6 with G={groups}: the tile mean "
        f"correction is not doing its job"
    )


# ---------------------------------------------------------------------------
# 12. the fused finalize and the capped elementwise grid
# ---------------------------------------------------------------------------
#
# ``_stats_finalize``/``_bwd_finalize``/``_dwdb_reduce`` are no longer their own
# launches: each is recomputed inside the elementwise kernel that consumes it.
# Two consequences need pinning.
#
# * The elementwise grid is capped at ``GNConfig.elem_progs`` and each program
#   *strides* over its share of the tiles, so that the fused finalize costs
#   ``nprog_elem * nsplit`` and not ``nblk_elem * nsplit`` reads.  No scale-8
#   shape and no shape in either suite reaches that path with the shipped
#   table -- ``nprog_elem == nblk_elem`` at every small shape -- so it has to be
#   reached deliberately.
# * The cap is a *performance* knob.  If it could change a single bit of the
#   output it would break the module's reproducibility contract, since it is
#   the one plan field that does not follow from the shape alone.


@contextlib.contextmanager
def _forced_config(channels, spatial, **overrides):
    """Temporarily install a tiling config for one ``(channels, spatial)`` key.

    ``default_config`` keys the frozen table by ``(num_channels, cube-root
    spatial extent)``, so the spatial extent has to be a perfect cube here.
    Restores the previous entry (or its absence) and clears the plan cache on
    the way out, so no other test can see it.
    """
    edge = round(spatial ** (1.0 / 3.0))
    assert edge**3 == spatial, "forced configs need a cube spatial extent"
    key = (channels, edge)
    cfg = tgn.GNConfig(*tgn.default_config(channels, spatial).key())
    for name, value in overrides.items():
        assert hasattr(cfg, name), name
        setattr(cfg, name, value)
    sentinel = object()
    saved = tgn._TUNED.get(key, sentinel)
    tgn._TUNED[key] = cfg
    tgn._plan.cache_clear()
    try:
        yield cfg
    finally:
        if saved is sentinel:
            del tgn._TUNED[key]
        else:
            tgn._TUNED[key] = saved
        tgn._plan.cache_clear()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,groups",
    [
        ((1, 64, 16, 16, 16), 8),
        ((2, 64, 16, 16, 16), 8),  # N > 1: the stride is per (blk, n) program
        ((1, 20, 8, 8, 8), 5),  # capped grid *and* a padded channel axis
    ],
)
@pytest.mark.parametrize("elem_progs", [1, 3, 8])
@pytest.mark.parametrize("activation", [None, "relu"])
def test_capped_elementwise_grid_strides_over_its_tiles(
    shape, groups, elem_progs, activation
):
    """Fewer elementwise programs than tiles: each must cover several tiles.

    A grid-stride loop that got its start, stride or trip count wrong leaves
    part of the output (and of ``d_input``) unwritten -- which, since both are
    ``torch.empty``, surfaces as plausible stale numbers rather than as a
    crash.  ``elem_progs=1`` is the extreme: one program per sample walks every
    tile, so it also pins that the fused statistics are hoisted out of the loop
    correctly rather than being recomputed per iteration from stale state.
    """
    spatial = shape[2] * shape[3] * shape[4]
    with _forced_config(shape[1], spatial, elem_tile=1024, elem_progs=elem_progs):
        plan = tgn._plan(shape[0], shape[1], spatial, groups, 0)
        assert plan.nprog_elem == min(plan.nblk_elem, elem_progs)
        assert plan.nprog_elem < plan.nblk_elem, (
            f"the cap has to actually bite: nprog={plan.nprog_elem} "
            f"nblk={plan.nblk_elem}"
        )
        _parity(
            shape,
            groups,
            activation,
            seed=abs(hash((shape, elem_progs))) % 997,
            label=f"{shape} elem_progs={elem_progs}",
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,groups", [((2, 64, 16, 16, 16), 8), ((1, 20, 8, 8, 8), 5)]
)
def test_elementwise_grid_cap_is_bitwise_neutral(shape, groups):
    """``elem_progs`` may not change a single bit of any output.

    It is the only field of ``_Plan`` that is a free parameter rather than a
    consequence of the shape, and the module promises bitwise reproducibility.
    That promise holds only because the elementwise kernels carry nothing
    across loop iterations: the fused finalize is computed once per program
    from the *same* partials with the *same* tile shape, and the tile bodies
    are pure elementwise.  If tuning this knob ever moved a result, the frozen
    table would have become part of the numerical contract.
    """
    spatial = shape[2] * shape[3] * shape[4]
    x, weight, bias, grad_out = _make(shape, groups, seed=11)
    results = []
    for elem_progs in (0, 1, 5, 64, 4096):
        with _forced_config(shape[1], spatial, elem_tile=1024, elem_progs=elem_progs):
            xi = x.detach().clone().requires_grad_(True)
            wi = weight.detach().clone().requires_grad_(True)
            bi = bias.detach().clone().requires_grad_(True)
            y = triton_group_norm(xi, groups, wi, bi, EPS)
            y.backward(grad_out)
            results.append(
                (y.detach().clone(), xi.grad.clone(), wi.grad.clone(), bi.grad.clone())
            )
    for elem_progs, got in zip((1, 5, 64, 4096), results[1:]):
        for name, a, b in zip(("y", "dx", "dweight", "dbias"), got, results[0]):
            assert torch.equal(a, b), (
                f"elem_progs={elem_progs} changed {name} bitwise; the grid cap "
                f"is supposed to be a pure performance knob"
            )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,groups,elem_progs",
    [
        ((1, 64, 16, 16, 16), 8, 0),
        ((1, 64, 16, 16, 16), 8, 3),
        ((2, 96, 8, 8, 8), 6, 2),  # padded channel axis, N > 1, capped grid
    ],
)
def test_fused_finalize_publishes_the_statistics(shape, groups, elem_progs):
    """``mean``/``rstd`` are published by program 0 of the *normalize* kernel.

    There is no separate finalize launch any more: every elementwise program
    re-derives the statistics from the split-K Welford partials, and program 0
    is the one that stores them for the backward pass.  A wrong publishing
    program, a wrong partials index, or a group-mask slip in that fused
    reduction would hand the backward garbage while leaving the forward -- which
    uses its own locally computed copy -- perfectly correct.  So check the
    published tensors directly against float64.
    """
    spatial = shape[2] * shape[3] * shape[4]
    with _forced_config(shape[1], spatial, elem_tile=1024, elem_progs=elem_progs):
        x, weight, bias, _grad = _make(shape, groups, seed=3)
        _y, mean, rstd = torch.ops.scaffold_gn.group_norm(
            x, groups, weight, bias, EPS, None, None
        )
        flat = x.double().reshape(shape[0], groups, -1)
        mean64 = flat.mean(-1)
        var64 = ((flat - mean64[..., None]) ** 2).mean(-1)
        assert _rel(mean, mean64) <= FP32_TOL
        assert _rel(rstd, 1.0 / torch.sqrt(var64 + EPS)) <= FP32_TOL


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,groups", [((2, 2048, 1, 1, 1), 8), ((1, 1024, 2, 1, 1), 8)]
)
def test_dweight_blocks_are_covered_when_there_are_more_of_them_than_tiles(
    shape, groups
):
    """The dweight/dbias reduction rides in ``_dx_kernel``'s first NDW programs.

    Those blocks are per-*channel*, the elementwise tiles are per-*voxel*, and
    nothing makes the first outnumber the second: at ``(2, 2048, 1, 1, 1)``
    there is one elementwise tile and eight dweight blocks.  The grid is
    ``max(nprog_elem, dwdb_progs)`` for exactly that reason, and a grid of
    ``nprog_elem`` alone would silently leave 7/8 of ``d_weight`` unwritten.
    """
    spatial = shape[2] * shape[3] * shape[4]
    plan = tgn._plan(shape[0], shape[1], spatial, groups, 0)
    assert plan.dwdb_progs > plan.nprog_elem, (
        f"case is supposed to have more dweight blocks ({plan.dwdb_progs}) than "
        f"elementwise programs ({plan.nprog_elem})"
    )
    assert plan.grid_dx == plan.dwdb_progs
    _parity(shape, groups, seed=13)


# ---------------------------------------------------------------------------
# the kernel-failure boundary
# ---------------------------------------------------------------------------


def test_kernel_failures_are_tagged_and_carry_their_cause():
    """Everything the launch region raises comes out as ``TritonKernelError``.

    The tag is what lets a caller with a fallback (``FastGroupNorm``'s ladder)
    catch *exactly* "the kernel is broken" instead of catching ``Exception`` and
    then trying to enumerate every framework mechanism -- saved-tensor pack
    hooks, ``torch.utils.checkpoint``'s recompute control flow, functorch --
    that legitimately raises through a forward.  The region it wraps is closed
    (allocations and launches, no autograd-observable op), so a blanket catch
    inside it is sound where one at the call site is not.

    The tag must survive the *type* of the original error, whatever it was: a
    mismatched Triton release raises ``TypeError``/``AttributeError`` from a
    changed signature, an unwritable JIT cache ``OSError``, a bad launch
    ``RuntimeError``.
    """
    for original in (
        RuntimeError("launch failed"),
        TypeError("triton API changed"),
        AttributeError("no such attribute"),
        OSError("unwritable cache dir"),
        ImportError("no module named triton"),
    ):

        @tgn._tag_kernel_failures
        def _boom():
            raise original

        with pytest.raises(tgn.TritonKernelError) as caught:
            _boom()
        assert caught.value.__cause__ is original
        assert type(original).__name__ in str(caught.value)


def test_out_of_memory_is_not_tagged_as_a_kernel_failure():
    """An OOM is a resource condition, and every fallback allocates as much.

    Tagging it would make the ladder retry on a rung that is about to OOM in
    the same place, and would latch a rung off for the rest of the process on a
    transient, per-rank event.  It has to come out unchanged.
    """

    @tgn._tag_kernel_failures
    def _oom():
        raise torch.OutOfMemoryError("simulated OOM")

    with pytest.raises(torch.OutOfMemoryError):
        _oom()
    assert not issubclass(torch.OutOfMemoryError, tgn.TritonKernelError)


def test_contract_violations_are_not_tagged():
    """``_validate``'s ``ValueError``s are caller errors, and stay loud.

    ``is_supported`` accepts exactly what ``_validate`` accepts, so a caller
    that branches on the predicate can never see one; if the two ever disagree,
    the failure must not be laundered into "the kernel is broken" and silently
    fall back.
    """
    with pytest.raises(ValueError, match="activation must be one of"):
        tgn._validate(torch.zeros(1, 8, 2, 2, 2), 8, None, None, "gelu")
    with pytest.raises(ValueError, match="expected a 5-D"):
        tgn._validate(torch.zeros(1, 8, 2, 2), 8, None, None, None)


@pytest.mark.gpu
def test_a_real_launch_failure_is_tagged(monkeypatch):
    """End to end: break the launch and the public op raises the tagged type."""
    x = torch.randn(1, 64, 4, 4, 4, device="cuda").to(memory_format=CL)

    def _broken(*args, **kwargs):
        raise RuntimeError("simulated HIP launch failure")

    tgn._ensure_kernels()
    monkeypatch.setattr(tgn, "_stats_partial_kernel", _Unlaunchable(_broken))
    with pytest.raises(tgn.TritonKernelError):
        torch.ops.scaffold_gn.group_norm(x, 8, None, None, EPS, None, None)


class _Unlaunchable:
    """A stand-in for a ``triton.jit`` kernel whose launch raises."""

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, grid):
        return self._fn


# ---------------------------------------------------------------------------
# the fused activation on non-finite values
# ---------------------------------------------------------------------------

#: NaN, +Inf, -Inf, -0.0 and four ordinary values.  ``tl.maximum(y, 0)`` returns
#: the non-NaN operand and ``tl.where(y > 0, y, 0)`` fails ``NaN > 0``, so both
#: of the obvious spellings map NaN to 0.0 where ``F.relu`` propagates it.
_SPECIALS = [float("nan"), float("inf"), float("-inf"), -0.0, 0.0, -1.0, 1.0, 2.0]


def _zero_weight_case(activation, seed=3):
    """A case whose pre-activation is exactly ``bias``, elementwise.

    Poisoning the *input* can only produce NaN pre-activations -- one non-finite
    value makes the whole group's statistics NaN -- so the values that actually
    distinguish the spellings of ReLU have to be placed directly.  A zero
    ``weight`` does that: ``xhat * 0 + bias == bias``.
    """
    x = torch.randn(
        1,
        64,
        4,
        4,
        4,
        device="cuda",
        generator=torch.Generator("cuda").manual_seed(seed),
    ).to(memory_format=CL)
    weight = torch.zeros(64, device="cuda")
    bias = torch.tensor(_SPECIALS * 8, device="cuda")
    reference = F.group_norm(x, 8, weight, bias, EPS)
    if activation == "relu":
        reference = F.relu(reference)
    return x, weight, bias, reference


@pytest.mark.gpu
def test_fused_relu_matches_f_relu_on_nan_inf_and_negative_zero():
    """The fused store must be ``F.relu``, bit for bit, on every special value.

    NaN in, NaN out -- and that matters beyond numerics: ScaFFold aborts a run
    whose loss goes non-finite, so an activation that turns a diverging NaN into
    a finite 0.0 makes the forward look healthy while the backward is still NaN,
    and the run checkpoints a broken model.  ``-Inf`` and both signed zeros must
    come out as ``+0.0``, never ``-0.0``.
    """
    x, weight, bias, reference = _zero_weight_case("relu")
    out, _mean, _rstd = torch.ops.scaffold_gn.group_norm(
        x, 8, weight, bias, EPS, "relu", None
    )
    assert torch.equal(out.cpu().view(torch.int32), reference.cpu().view(torch.int32))
    # ... and the control: without the fusion the same values pass through.
    plain, _m, _r = torch.ops.scaffold_gn.group_norm(
        x, 8, weight, bias, EPS, None, None
    )
    assert plain[0, 0, 0, 0, 0].isnan() and plain[0, 1, 0, 0, 0].isinf()


@pytest.mark.gpu
def test_fused_relu_backward_gates_like_threshold_backward():
    """ReLU's backward is ``result <= 0 ? 0 : grad``, so a NaN passes.

    The kernel recomputes the pre-activation and must gate with the same
    complement: ``pre > 0 ? dy : 0`` reads identically on every finite value and
    silently zeroes the NaN lane, which is the backward half of the same defect.
    """
    x, weight, bias, _reference = _zero_weight_case("relu")
    grad_out = torch.ones_like(x)

    weight = weight.requires_grad_(True)
    bias = bias.requires_grad_(True)
    xg = x.clone().requires_grad_(True)
    reference = F.relu(F.group_norm(xg, 8, weight, bias, EPS))
    reference.backward(grad_out)
    ref_dbias = bias.grad.detach().clone()
    ref_dx = xg.grad.detach().clone()

    weight.grad = bias.grad = xg.grad = None
    triton_group_norm(xg, 8, weight, bias, EPS, "relu").backward(grad_out)

    # d_bias counts exactly the elements whose gradient the gate let through.
    assert ref_dbias[0].item() == 64, "the reference gated the NaN lane off"
    assert torch.equal(bias.grad.cpu(), ref_dbias.cpu())
    assert torch.equal(xg.grad.cpu(), ref_dx.cpu())
