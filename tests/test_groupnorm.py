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

"""Tests for the compiled GroupNorm fast path (``ScaFFold.unet.group_norm``).

The optimization must be invisible everywhere except in the profile: the same
state dict as a stock ``nn.GroupNorm`` model (checkpoints stay interchangeable
in both directions), the same numbers within reduction-order noise, and an
eager fallback for every input the compiled kernel cannot or should not take
(CPU, tensor subclasses such as DistConv's ``DCTensor``, a broken compiler).
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from ScaFFold.unet import group_norm as gn_mod
from ScaFFold.unet.group_norm import FastGroupNorm
from ScaFFold.unet.unet_model import UNet

_N = 16
_N_CHANNELS = 3
_N_CLASSES = 2
_GROUPS = 8


@pytest.fixture(autouse=True)
def _restore_compile_state():
    """Keep per-test overrides of the module-level compile state contained."""
    previous = gn_mod.set_compile_enabled(None)
    failed = gn_mod._compile_failed
    yield
    gn_mod._compile_override = previous
    gn_mod._compile_failed = failed


def _make_unet(seed: int, group_norm_cls=None):
    """Build the worker.py-shaped UNet, optionally with a different norm class."""
    torch.manual_seed(seed)
    if group_norm_cls is None:
        return UNet(
            n_channels=_N_CHANNELS,
            n_classes=_N_CLASSES,
            trilinear=False,
            layers=2,
            group_norm_groups=_GROUPS,
        )
    import ScaFFold.unet.unet_parts as parts

    original = parts.FastGroupNorm
    parts.FastGroupNorm = group_norm_cls
    try:
        return UNet(
            n_channels=_N_CHANNELS,
            n_classes=_N_CLASSES,
            trilinear=False,
            layers=2,
            group_norm_groups=_GROUPS,
        )
    finally:
        parts.FastGroupNorm = original


def _make_input(seed: int = 0, channels: int = _N_CHANNELS, size: int = _N):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, channels, size, size, size, generator=generator)


# ---------------------------------------------------------------------------
# state dict compatibility
# ---------------------------------------------------------------------------


def test_state_dict_matches_plain_groupnorm_model():
    """Names, shapes and dtypes must be unchanged from the nn.GroupNorm build.

    A checkpoint written before this optimization has to keep loading, so the
    parameter inventory of the model may not shift by even one key.
    """
    new_model = _make_unet(seed=0)
    old_model = _make_unet(seed=0, group_norm_cls=nn.GroupNorm)

    new_sd = new_model.state_dict()
    old_sd = old_model.state_dict()
    assert list(new_sd.keys()) == list(old_sd.keys())
    for key in old_sd:
        assert new_sd[key].shape == old_sd[key].shape, key
        assert new_sd[key].dtype == old_sd[key].dtype, key
    # The optimization must not have introduced buffers either.
    assert [name for name, _ in new_model.named_buffers()] == [
        name for name, _ in old_model.named_buffers()
    ]


def test_checkpoint_round_trip_both_directions(tmp_path):
    """An old checkpoint loads into the new model and vice versa, strict=True.

    Both directions matter: runs resumed onto the new code must accept old
    checkpoints, and checkpoints written by the new code must stay readable by
    anything still building plain ``nn.GroupNorm`` (e.g. an older analysis
    script).  After each load the two models must agree bit for bit.
    """
    new_model = _make_unet(seed=0)
    old_model = _make_unet(seed=1, group_norm_cls=nn.GroupNorm)
    x = _make_input(seed=3)

    old_path = tmp_path / "old.pth"
    torch.save({"model_state_dict": old_model.state_dict()}, old_path)
    loaded = torch.load(old_path, weights_only=True)
    missing = new_model.load_state_dict(loaded["model_state_dict"], strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys

    new_model.eval()
    old_model.eval()
    with torch.no_grad():
        assert torch.equal(new_model(x), old_model(x))

    # ... and the reverse: new checkpoint into the plain-GroupNorm model.
    fresh_new = _make_unet(seed=2)
    new_path = tmp_path / "new.pth"
    torch.save({"model_state_dict": fresh_new.state_dict()}, new_path)
    reloaded = torch.load(new_path, weights_only=True)
    result = old_model.load_state_dict(reloaded["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys

    fresh_new.eval()
    with torch.no_grad():
        assert torch.equal(old_model(x), fresh_new(x))


def test_unet_uses_fast_group_norm():
    """Every norm layer in the model is the fast one -- no half-converted build."""
    model = _make_unet(seed=0)
    norms = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
    assert norms, "UNet should contain GroupNorm layers"
    assert all(isinstance(m, FastGroupNorm) for m in norms)


# ---------------------------------------------------------------------------
# CPU behavior: identical numerics, and no compilation at all
# ---------------------------------------------------------------------------


def test_cpu_output_bit_identical_to_eager():
    """On CPU the fast module is literally the stock kernel, so bits must match."""
    fast = FastGroupNorm(_GROUPS, 64)
    plain = nn.GroupNorm(_GROUPS, 64)
    with torch.no_grad():
        plain.weight.copy_(fast.weight)
        plain.bias.copy_(fast.bias)
    x = _make_input(seed=5, channels=64, size=8)
    assert torch.equal(fast(x), plain(x))


def test_cpu_never_invokes_torch_compile(monkeypatch):
    """The CPU unit suite must not pay Inductor's compile latency.

    Guards the ``input.is_cuda`` check: if it ever regresses, a CPU-only test
    run would start building C++ kernels for every GroupNorm shape.
    """
    calls = []

    def _boom(*a, **kw):
        calls.append(a)
        raise AssertionError("torch.compile must not be called for CPU tensors")

    monkeypatch.setattr(torch, "compile", _boom)
    monkeypatch.setattr(gn_mod, "_compiled_group_norm", None)
    gn_mod.set_compile_enabled(True)  # even when explicitly forced on

    model = _make_unet(seed=0)
    with torch.no_grad():
        model(_make_input(seed=6))
    assert not calls


def test_tensor_subclass_input_stays_eager():
    """DistConv wraps activations in a ``__torch_dispatch__`` tensor subclass.

    Dynamo cannot trace those wrappers, so the predicate must reject anything
    that is not exactly ``torch.Tensor`` before a compile is attempted.
    """

    class _Wrapper(torch.Tensor):
        pass

    plain = torch.randn(1, 8, 4, 4, 4)
    assert gn_mod._use_compiled(plain) is False  # CPU
    assert gn_mod._use_compiled(plain.as_subclass(_Wrapper)) is False


def test_compile_failure_falls_back_to_eager(monkeypatch, caplog):
    """A broken compiler degrades to the stock kernel instead of killing the run.

    Simulated by making the compiled callable raise; the module must return the
    eager result, warn once, and stop trying for the rest of the process.
    """

    def _raises(*args, **kwargs):
        raise RuntimeError("simulated Inductor failure")

    monkeypatch.setattr(gn_mod, "_use_compiled", lambda _input: True)
    monkeypatch.setattr(gn_mod, "_get_compiled_group_norm", lambda: _raises)
    gn_mod._compile_failed = False

    fast = FastGroupNorm(_GROUPS, 64)
    x = _make_input(seed=7, channels=64, size=8)
    with caplog.at_level(logging.WARNING, logger=gn_mod.__name__):
        out = fast(x)
    assert torch.equal(
        out, nn.functional.group_norm(x, _GROUPS, fast.weight, fast.bias)
    )
    assert any("falling back to the eager kernel" in r.message for r in caplog.records)
    assert gn_mod._compile_failed is True
    # Latched off: the predicate now refuses even a would-be eligible tensor.
    monkeypatch.undo()
    assert gn_mod._use_compiled(torch.randn(1, 8, 4, 4, 4)) is False


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0", False),
        ("false", False),
        ("OFF", False),
        ("no", False),
        ("1", True),
        ("true", True),
        ("On", True),
        ("yes", True),
        ("maybe", None),
    ],
)
def test_env_var_controls_the_fast_path(monkeypatch, value, expected):
    """``SCAFFOLD_GROUPNORM_COMPILE`` is the documented run-time opt-out."""
    monkeypatch.setenv(gn_mod.COMPILE_ENV_VAR, value)
    gn_mod.set_compile_enabled(None)
    assert gn_mod._compile_override is expected


def test_env_var_unset_means_auto(monkeypatch):
    monkeypatch.delenv(gn_mod.COMPILE_ENV_VAR, raising=False)
    gn_mod.set_compile_enabled(None)
    assert gn_mod._compile_override is None


def test_recompile_limit_is_raised_never_lowered():
    """Dynamo's stock cap of 8 is below what one UNet needs.

    A scale-7 UNet presents 5 distinct GroupNorm shapes, and evaluation runs the
    same 5 again under ``no_grad`` -- 10 cache entries (measured).  Past the cap
    Dynamo gives up and every GroupNorm silently reverts to the slow kernel, so
    the module raises the limit; it must never lower one a caller chose.
    """
    import torch._dynamo

    config = torch._dynamo.config
    name = (
        "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    )
    original = getattr(config, name)
    try:
        setattr(config, name, 8)
        gn_mod._raise_recompile_limit()
        assert getattr(config, name) >= 10
        setattr(config, name, 4096)
        gn_mod._raise_recompile_limit()
        assert getattr(config, name) == 4096
    finally:
        setattr(config, name, original)


# ---------------------------------------------------------------------------
# GPU behavior: numerics, single compile, checkpointing
# ---------------------------------------------------------------------------


def _assert_close(actual, expected, tol, what):
    diff = (actual.float() - expected.float()).abs().max().item()
    scale = expected.float().abs().max().item()
    assert diff <= tol * max(scale, 1.0), f"{what}: max|diff|={diff:.3e}"
    return diff


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 64, 32, 32, 32), (2, 128, 16, 16, 16)])
@pytest.mark.parametrize("autocast", [False, True])
def test_gpu_compiled_matches_eager(shape, autocast):
    """Compiled forward and gradients match eager, fp32 and under bf16 autocast.

    ``(1, 64, 32^3)`` is the hot production shape (``[1, 64, 128^3]``) at
    reduced size -- same channel count and group count, same reduction
    structure, small enough for a unit test.  Tolerances are loose enough for
    reduction-order differences and tight enough to catch a real numerics bug;
    observed maxima on MI300A are ~1e-6 relative.
    """
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(11)
    x = torch.randn(*shape, device=device, generator=generator)
    grad_out = torch.randn(*shape, device=device, generator=generator)

    fast = FastGroupNorm(_GROUPS, shape[1]).to(device)
    with torch.no_grad():
        fast.weight.normal_(1.0, 0.1, generator=generator)
        fast.bias.normal_(0.0, 0.1, generator=generator)

    def run(compiled):
        gn_mod.set_compile_enabled(compiled)
        inp = x.clone().requires_grad_(True)
        fast.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = fast(inp)
        out.backward(grad_out.to(out.dtype))
        return (
            out.detach(),
            inp.grad.detach(),
            fast.weight.grad.detach().clone(),
            fast.bias.grad.detach().clone(),
        )

    eager = run(False)
    compiled = run(True)
    assert gn_mod._compiled_group_norm is not None, "compiled path was not taken"
    assert not gn_mod._compile_failed

    _assert_close(compiled[0], eager[0], 1e-5, "output")
    _assert_close(compiled[1], eager[1], 1e-4, "d_input")
    _assert_close(compiled[2], eager[2], 1e-4, "d_weight")
    _assert_close(compiled[3], eager[3], 1e-4, "d_bias")
    # Autocast policy must be preserved: GroupNorm is an fp32 op, so the
    # compiled path may not quietly hand back bf16 activations.
    assert compiled[0].dtype == eager[0].dtype


@pytest.mark.gpu
def test_gpu_steady_state_does_not_recompile():
    """Fixed shapes must compile once and then never again.

    A recompile inside a timed epoch would show up as a multi-second outlier in
    ``epoch_duration`` (and therefore the FOM), so the guard set has to be
    stable across steps.
    """
    from torch._dynamo.utils import counters

    gn_mod.set_compile_enabled(True)
    device = torch.device("cuda")
    fast = FastGroupNorm(_GROUPS, 64).to(device)
    x = torch.randn(1, 64, 16, 16, 16, device=device, requires_grad=True)

    def step():
        fast.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fast(x)
        out.sum().backward()

    step()  # first call: compiles
    before = counters["stats"]["unique_graphs"]
    for _ in range(5):
        step()
    assert counters["stats"]["unique_graphs"] == before, "recompiled in steady state"


@pytest.mark.gpu
def test_gpu_activation_checkpointing_matches_eager():
    """The compiled kernel must survive recompute under use_checkpointing().

    Non-reentrant checkpointing replays the block's forward inside the backward
    pass; a compiled region has to produce the same activations both times or
    the gradients silently change.

    Compared as relative L2 error per gradient tensor, because whole-network
    agreement is not bitwise even without this change: with cudnn.benchmark on
    and bf16 autocast, two eager runs of this model differ by ~4e-3 relative
    (measured), and checkpointing on vs. off differs by the same amount.
    Measured here: compiled vs. eager 6.4e-3, i.e. the same order as that noise
    floor -- while a genuinely wrong kernel would be O(1).
    """
    device = torch.device("cuda")
    x = _make_input(seed=9).to(device).requires_grad_(True)
    tolerance = 5e-2

    def grads(compiled, checkpointing):
        gn_mod.set_compile_enabled(compiled)
        model = _make_unet(seed=0).to(device)
        if checkpointing:
            model.use_checkpointing()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out.float().pow(2).mean().backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    def assert_agrees(actual, expected, label):
        for name in expected:
            reference = expected[name].float()
            error = (actual[name].float() - reference).norm().item()
            relative = error / max(reference.norm().item(), 1e-12)
            assert relative < tolerance, f"{label} {name}: rel L2 {relative:.3e}"

    eager = grads(False, True)
    compiled = grads(True, True)
    compiled_nockpt = grads(True, False)
    assert gn_mod._compiled_group_norm is not None, "compiled path was not taken"
    assert not gn_mod._compile_failed
    assert_agrees(compiled, eager, "checkpointed grad")
    assert_agrees(compiled_nockpt, compiled, "grad")
