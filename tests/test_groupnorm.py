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

"""Tests for the GroupNorm fast paths (``ScaFFold.unet.group_norm``).

The optimization must be invisible everywhere except in the profile: the same
state dict as a stock ``nn.GroupNorm`` model (checkpoints stay interchangeable
in both directions), the same numbers within reduction-order noise, and a
fallback for every input the fast kernels cannot or should not take (CPU,
unknown tensor subclasses, a broken Triton or Inductor install).  The ladder is
Triton -> compiled -> eager, and a failure at any rung latches that rung off and
drops to the next, never to the bottom.

DistConv's ``DCTensor`` is not in the rejection list: ``forward`` unwraps it to
its local shard around both fast kernels, so the wrapped production path is
served too.  That unwrap is *not* the bare attribute read DistConv's own
dispatch does -- dispatch runs below autograd, where a bare read is safe, while
``forward`` runs above it and must go through DistConv's
``_ToTensor``/``_FromTensor`` pair to keep the graph connected.

The ReLU that used to follow every GroupNorm now lives inside it
(``activation="relu"``), fused into the Triton store and applied explicitly on
the other two paths.  ``DoubleConv`` keeps an ``nn.Identity`` in the vacated
``nn.Sequential`` slot, so the state dict does not move by one key -- which is
what the checkpoint tests here pin.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ScaFFold.unet import group_norm as gn_mod
from ScaFFold.unet.group_norm import FastGroupNorm
from ScaFFold.unet.unet_model import UNet
from tests.helpers import mpi_runner

_N = 16
_N_CHANNELS = 3
_N_CLASSES = 2
_GROUPS = 8


@pytest.fixture(autouse=True)
def _restore_compile_state():
    """Keep per-test overrides of the module-level routing state contained."""
    previous_compile = gn_mod.set_compile_enabled(None)
    previous_triton = gn_mod.set_triton_enabled(None)
    compile_failed = gn_mod._compile_failed
    triton_failed = gn_mod._triton_failed
    yield
    gn_mod._compile_override = previous_compile
    gn_mod._triton_override = previous_triton
    gn_mod._compile_failed = compile_failed
    gn_mod._triton_failed = triton_failed


def _make_unet(seed: int):
    """Build the worker.py-shaped UNet."""
    torch.manual_seed(seed)
    return UNet(
        n_channels=_N_CHANNELS,
        n_classes=_N_CLASSES,
        trilinear=False,
        layers=2,
        group_norm_groups=_GROUPS,
    )


def _make_plain_unet(seed: int):
    """The pre-fusion build: stock ``nn.GroupNorm`` followed by ``nn.ReLU``.

    Built by *converting* a normal UNet rather than by patching the norm class
    at construction time, because the fusion moved the ReLU into the norm: a
    class swap alone would leave ``DoubleConv``'s ``nn.Identity`` placeholders
    in place and produce a model with no activations at all, which would make
    every numeric comparison below vacuous.  Converting reproduces exactly the
    module graph this branch replaced -- ``nn.GroupNorm`` where the fast norm
    sits, an in-place ``nn.ReLU`` where the placeholder sits -- and consumes no
    RNG (``nn.GroupNorm`` initializes to ones/zeros), so the parameters are
    bit-identical to what ``_make_unet(seed)`` draws.
    """
    model = _make_unet(seed)
    for parent in [m for m in model.modules() if isinstance(m, nn.Sequential)]:
        for index, child in enumerate(list(parent)):
            if isinstance(child, FastGroupNorm):
                plain = nn.GroupNorm(
                    child.num_groups,
                    child.num_channels,
                    eps=child.eps,
                    affine=child.affine,
                )
                if child.affine:
                    with torch.no_grad():
                        plain.weight.copy_(child.weight)
                        plain.bias.copy_(child.bias)
                parent[index] = plain
            elif isinstance(child, nn.Identity):
                parent[index] = nn.ReLU(inplace=True)
    return model


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
    old_model = _make_plain_unet(seed=0)

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
    old_model = _make_plain_unet(seed=1)
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


def test_state_dict_bytes_identical_to_plain_groupnorm_model():
    """Not just the same keys: the serialized checkpoint must be byte identical.

    ``test_state_dict_matches_plain_groupnorm_model`` compares names, shapes and
    dtypes; this compares the actual bytes ``torch.save`` writes, which is the
    thing that has to stay interchangeable.  It is the direct guard on the
    ``nn.ReLU`` -> ``nn.Identity`` swap: ``nn.Sequential`` names its children by
    position, so *removing* the activation slot rather than holding it open
    would renumber ``3.weight`` and ``4.weight``/``4.bias`` and silently
    invalidate every checkpoint on disk.
    """
    import io

    new_model = _make_unet(seed=0)
    old_model = _make_plain_unet(seed=0)

    def blob(model):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()

    assert blob(new_model) == blob(old_model)


def test_double_conv_keeps_the_activation_slots():
    """The fused build keeps six positional slots, with nothing in the spares.

    Pins both halves of the fusion design: the ReLU is *in* the norm
    (``activation == "relu"`` at positions 1 and 4) and its old slots (2 and 5)
    are parameterless placeholders rather than deletions.
    """
    from ScaFFold.unet.unet_parts import DoubleConv

    block = DoubleConv(3, 16, _GROUPS)
    children = list(block.double_conv)
    assert len(children) == 6
    for norm_index, spare_index in ((1, 2), (4, 5)):
        norm = children[norm_index]
        assert isinstance(norm, FastGroupNorm)
        assert norm.activation == "relu"
        spare = children[spare_index]
        assert isinstance(spare, nn.Identity)
        assert list(spare.parameters()) == []
        assert list(spare.buffers()) == []
    # And the positional key numbering is exactly the pre-fusion one.
    assert list(block.state_dict().keys()) == [
        "double_conv.0.weight",
        "double_conv.1.weight",
        "double_conv.1.bias",
        "double_conv.3.weight",
        "double_conv.4.weight",
        "double_conv.4.bias",
    ]


def test_double_conv_output_matches_the_explicit_relu_build():
    """Folding the ReLU into the norm may not change a single bit of the output.

    The fused module applies the ReLU itself on every path, so on CPU (eager)
    the block must reproduce ``conv -> GroupNorm -> ReLU`` exactly, gradients
    included.
    """
    from ScaFFold.unet.unet_parts import DoubleConv

    torch.manual_seed(4)
    fused = DoubleConv(3, 16, _GROUPS)
    reference = DoubleConv(3, 16, _GROUPS)
    reference.load_state_dict(fused.state_dict())
    for index in (1, 4):
        reference.double_conv[index].activation = None
    for index in (2, 5):
        reference.double_conv[index] = nn.ReLU(inplace=True)

    x_fused = _make_input(seed=8, channels=3, size=8).requires_grad_(True)
    x_reference = x_fused.detach().clone().requires_grad_(True)

    out_fused = fused(x_fused)
    out_reference = reference(x_reference)
    assert torch.equal(out_fused, out_reference)
    # A block whose activation silently vanished would still pass an
    # output-equality test against another activation-free block, so assert the
    # ReLU is really there.
    assert (out_fused < 0).sum() == 0
    assert out_fused.max() > 0

    out_fused.pow(2).sum().backward()
    out_reference.pow(2).sum().backward()
    assert torch.equal(x_fused.grad, x_reference.grad)
    for (name, a), (_, b) in zip(
        fused.named_parameters(), reference.named_parameters()
    ):
        assert torch.equal(a.grad, b.grad), name


def test_activation_argument_is_validated():
    """An unknown activation must fail at construction, not at the first step."""
    with pytest.raises(ValueError, match="activation"):
        FastGroupNorm(_GROUPS, 16, activation="gelu")


def test_supported_activations_match_the_kernels():
    """The module's activation list may not drift from the kernel's.

    ``group_norm`` spells the tuple out rather than importing it (importing the
    kernel module has to stay off the CPU path), so nothing but this test stops
    the two copies from diverging into a runtime ``ValueError`` from inside the
    custom op.
    """
    from ScaFFold.unet import triton_group_norm as triton_mod

    assert set(gn_mod.SUPPORTED_ACTIVATIONS) <= set(triton_mod.SUPPORTED_ACTIVATIONS)


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
    """Unknown tensor subclasses must stay on the eager path.

    Dynamo cannot trace ``__torch_dispatch__`` wrappers, so the predicate must
    reject anything that is not exactly ``torch.Tensor`` before a compile is
    attempted.  DistConv's ``DCTensor`` is handled separately -- ``forward``
    unwraps it before consulting the predicate -- but any other wrapper has
    unknown semantics and keeps the stock kernel.
    """

    class _Wrapper(torch.Tensor):
        pass

    plain = torch.randn(1, 8, 4, 4, 4)
    assert gn_mod._use_compiled(plain) is False  # CPU
    assert gn_mod._use_compiled(plain.as_subclass(_Wrapper)) is False


def test_compile_failure_falls_back_to_eager(monkeypatch, caplog):
    """A broken compiler degrades to the stock kernel instead of killing the run.

    Simulated by making the compiled callable raise the real thing Dynamo and
    Inductor raise (``BackendCompilerFailed``/``Unsupported`` share the
    ``TorchDynamoException`` root the ladder allowlists); the module must return
    the eager result, warn once, and stop trying for the rest of the process.
    """
    import torch._dynamo.exc

    def _raises(*args, **kwargs):
        raise torch._dynamo.exc.Unsupported("simulated Inductor failure")

    monkeypatch.setattr(gn_mod, "_use_compiled", lambda _input, **kw: True)
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


def test_triton_failure_falls_back_to_the_compiled_kernel(monkeypatch, caplog):
    """A broken Triton install drops to *compiled*, not all the way to eager.

    The distinction is worth 10x on the shapes that dominate the step, so the
    ladder must have three rungs and not two.  Simulated by forcing the Triton
    predicate on and making its kernel raise; the compiled stand-in must then be
    the one that answers, exactly once, with the Triton path latched off.
    """
    from ScaFFold.unet.triton_group_norm import TritonKernelError

    compiled_calls = []

    def _raises(*args, **kwargs):
        raise TritonKernelError("simulated Triton failure")

    def _recording(input, num_groups, weight, bias, eps):
        compiled_calls.append(type(input))
        return nn.functional.group_norm(input, num_groups, weight, bias, eps)

    monkeypatch.setattr(gn_mod, "_use_triton", lambda *a, **kw: True)
    monkeypatch.setattr(gn_mod, "_get_triton_module", _raises)
    monkeypatch.setattr(gn_mod, "_use_compiled", lambda _input, **kw: True)
    monkeypatch.setattr(gn_mod, "_get_compiled_group_norm", lambda: _recording)
    gn_mod._triton_failed = False
    gn_mod._compile_failed = False

    fast = FastGroupNorm(_GROUPS, 64)
    x = _make_input(seed=44, channels=64, size=8)
    with caplog.at_level(logging.WARNING, logger=gn_mod.__name__):
        out = fast(x)

    assert compiled_calls == [torch.Tensor], "compiled kernel was not the fallback"
    assert torch.equal(
        out, nn.functional.group_norm(x, _GROUPS, fast.weight, fast.bias)
    )
    assert any(
        "falling back to the compiled kernel" in r.message for r in caplog.records
    )
    assert gn_mod._triton_failed is True
    assert gn_mod._compile_failed is False
    # Latched off: the predicate now refuses even a would-be eligible tensor.
    monkeypatch.undo()
    assert gn_mod._use_triton(torch.randn(1, 8, 4, 4, 4), 8, None, None, None) is False


@pytest.mark.parametrize("rung", ["triton", "compiled"])
def test_checkpoint_recompute_stop_is_re_raised(monkeypatch, rung):
    """``_StopRecomputationError`` is control flow, not a kernel failure.

    ``torch.utils.checkpoint``'s non-reentrant recompute stops itself by raising
    it from a saved-tensor *pack hook*, i.e. from inside whichever op is saving
    a tensor at that moment -- which, now that the ReLU is fused and nothing
    follows GroupNorm in a ``DoubleConv``, is this module.  A blanket
    ``except Exception`` would swallow it, latch the fast kernel off and drop the
    whole model to eager mid-run.  (Observed exactly that on
    ``test_gpu_activation_checkpointing_matches_eager`` before the re-raise.)
    """
    import torch.utils.checkpoint as checkpoint_mod

    stop = checkpoint_mod._StopRecomputationError

    def _raises(*args, **kwargs):
        raise stop()

    if rung == "triton":
        monkeypatch.setattr(gn_mod, "_use_triton", lambda *a, **kw: True)
        monkeypatch.setattr(gn_mod, "_get_triton_module", _raises)
    else:
        monkeypatch.setattr(gn_mod, "_use_compiled", lambda _input, **kw: True)
        monkeypatch.setattr(gn_mod, "_get_compiled_group_norm", lambda: _raises)
    gn_mod._triton_failed = False
    gn_mod._compile_failed = False

    fast = FastGroupNorm(_GROUPS, 16)
    with pytest.raises(stop):
        fast(_make_input(seed=45, channels=16, size=4))
    assert gn_mod._triton_failed is False
    assert gn_mod._compile_failed is False


def test_cpu_activation_checkpointing_keeps_the_fast_path(monkeypatch):
    """End-to-end version of the above, on the real model.

    The fast path is forced on for CPU tensors (with the stock kernel standing
    in for the compiled one, so only the *routing* is under test) and the model
    is run with activation checkpointing.  Gradients must match the
    non-checkpointed run and the fast path must still be live afterwards -- a
    swallowed recompute-stop shows up as a latched-off kernel here.
    """
    monkeypatch.setattr(
        gn_mod, "_use_compiled", lambda t, **kw: type(t) is torch.Tensor
    )
    monkeypatch.setattr(
        gn_mod, "_get_compiled_group_norm", lambda: nn.functional.group_norm
    )
    gn_mod._compile_failed = False

    x = _make_input(seed=46).requires_grad_(True)

    def grads(checkpointing):
        model = _make_unet(seed=0)
        if checkpointing:
            model.use_checkpointing()
        model.zero_grad(set_to_none=True)
        model(x).pow(2).sum().backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    direct = grads(False)
    checkpointed = grads(True)
    assert gn_mod._compile_failed is False, "recompute-stop was swallowed"
    for name in direct:
        assert torch.allclose(direct[name], checkpointed[name]), name


def test_triton_rejects_unknown_tensor_subclasses():
    """``is_supported`` only asks ``isinstance``, so the type check lives here.

    A ``__torch_dispatch__`` wrapper other than DCTensor has unknown semantics
    and must keep the stock kernel, exactly as it does for the compiled path --
    but ``triton_group_norm.is_supported`` would happily accept one, so
    ``_use_triton`` has to reject it itself rather than delegating.
    """

    class _Wrapper(torch.Tensor):
        pass

    plain = torch.randn(1, 8, 4, 4, 4)
    assert gn_mod._use_triton(plain, 8, None, None, None) is False  # CPU
    assert gn_mod._use_triton(plain.as_subclass(_Wrapper), 8, None, None, None) is False


def test_triton_env_opt_out_skips_the_kernel_module_entirely(monkeypatch):
    """``SCAFFOLD_GROUPNORM_TRITON=0`` is checked before anything is imported."""

    def _boom():
        raise AssertionError("the kernel module must not be imported when opted out")

    monkeypatch.setattr(gn_mod, "_get_triton_module", _boom)
    gn_mod.set_triton_enabled(False)
    assert gn_mod._use_triton(torch.randn(1, 8, 4, 4, 4), 8, None, None, None) is False


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
def test_triton_env_var_controls_the_fast_path(monkeypatch, value, expected):
    """``SCAFFOLD_GROUPNORM_TRITON`` is parsed exactly like its compile twin."""
    monkeypatch.setenv(gn_mod.TRITON_ENV_VAR, value)
    gn_mod.set_triton_enabled(None)
    assert gn_mod._triton_override is expected


def test_triton_env_var_unset_means_auto(monkeypatch):
    """Unset means "on wherever is_supported accepts" -- the production default."""
    monkeypatch.delenv(gn_mod.TRITON_ENV_VAR, raising=False)
    gn_mod.set_triton_enabled(None)
    assert gn_mod._triton_override is None


def test_triton_env_var_garbage_warns(monkeypatch, caplog):
    """An unparsable value is ignored *loudly*, like the compile variable."""
    monkeypatch.setenv(gn_mod.TRITON_ENV_VAR, "sometimes")
    with caplog.at_level(logging.WARNING, logger=gn_mod.__name__):
        gn_mod.set_triton_enabled(None)
    assert any(gn_mod.TRITON_ENV_VAR in r.message for r in caplog.records)
    assert gn_mod._triton_override is None


def test_set_triton_enabled_returns_the_previous_setting(monkeypatch):
    """The save/restore contract tests rely on, matching set_compile_enabled."""
    monkeypatch.delenv(gn_mod.TRITON_ENV_VAR, raising=False)
    gn_mod.set_triton_enabled(None)
    assert gn_mod.set_triton_enabled(False) is None
    assert gn_mod.set_triton_enabled(True) is False
    assert gn_mod.set_triton_enabled(None) is True
    assert gn_mod._triton_override is None


def test_cpu_never_imports_triton(fresh_python):
    """A CPU-only process must not import triton, nor the kernel module.

    Two separate costs, both of which the CPU unit suite would otherwise pay on
    every run: ``import triton`` (seconds, and it is not installed everywhere),
    and importing ``ScaFFold.unet.triton_group_norm``, which registers two
    dispatcher ops and an autograd formula at import time.  ``_use_triton``
    rejects non-CUDA tensors *before* it touches the module, which is what this
    pins -- run in a fresh interpreter because the test session itself has long
    since imported the kernel module for the kernel's own tests.
    """
    out = fresh_python(
        "import sys\n"
        "import torch\n"
        "from ScaFFold.unet.unet_model import UNet\n"
        "m = UNet(n_channels=3, n_classes=2, trilinear=False, layers=1, "
        "group_norm_groups=8)\n"
        "with torch.no_grad():\n"
        "    m(torch.randn(1, 3, 16, 16, 16))\n"
        "print('triton', 'triton' in sys.modules)\n"
        "print('kernel', 'ScaFFold.unet.triton_group_norm' in sys.modules)\n"
    )
    assert "triton False" in out, out
    assert "kernel False" in out, out


def test_cpu_activation_is_applied_on_the_eager_path():
    """``activation="relu"`` is a promise of the module, not of the kernel."""
    fast = FastGroupNorm(_GROUPS, 16, activation="relu")
    plain = nn.GroupNorm(_GROUPS, 16)
    with torch.no_grad():
        plain.weight.copy_(fast.weight)
        plain.bias.copy_(fast.bias)
    x = _make_input(seed=41, channels=16, size=8)
    assert torch.equal(fast(x), torch.relu(plain(x)))


def test_cpu_activation_none_leaves_the_output_alone():
    """The default stays a bare GroupNorm -- no accidental global activation."""
    fast = FastGroupNorm(_GROUPS, 16)
    x = _make_input(seed=42, channels=16, size=8)
    assert torch.equal(
        fast(x), nn.functional.group_norm(x, _GROUPS, fast.weight, fast.bias)
    )


def test_activation_does_not_allocate_a_second_output():
    """The absorbed ReLU keeps ``nn.ReLU(inplace=True)``'s memory behaviour.

    The old ``nn.Sequential`` spelling mutated the GroupNorm output in place;
    an out-of-place ``F.relu`` here would add a full activation-sized allocation
    at all 22 sites.  Checked by handing the module a stand-in kernel whose
    output we still hold: the ReLU must have rewritten *that* tensor.
    """
    fast = FastGroupNorm(_GROUPS, 16, activation="relu")
    x = _make_input(seed=43, channels=16, size=4)
    produced = nn.functional.group_norm(x, _GROUPS, fast.weight, fast.bias, fast.eps)
    assert (produced < 0).any(), "test input must have negatives to clamp"
    out = fast._activate(produced)
    assert out.data_ptr() == produced.data_ptr()
    assert (produced < 0).sum() == 0


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


def test_the_compiled_region_carries_its_own_recompile_limit(monkeypatch):
    """The global limit is thread-local, so the region must carry one too.

    ``torch._dynamo.config`` keeps user overrides in a ``ContextVar``
    ("User overrides are thread-local", ``torch/utils/_config_module.py``), so
    what :func:`_raise_recompile_limit` writes is invisible from every *other*
    thread -- and one of those threads matters: ``torch.utils.checkpoint``'s
    non-reentrant recompute runs inside the backward pass, i.e. on the autograd
    engine's device worker thread.  On a ``DCTensor`` that recompute has to
    compile (it reaches this module with ``__torch_function__`` subclass
    handling disabled, which is part of Dynamo's ``GLOBAL_STATE`` guard, so it
    misses every entry the forward built), and there the limit read the stock 8
    however large the global had been set -- ``FailOnRecompileLimitHit``, run
    over.  ``torch.compile``'s ``recompile_limit=`` is applied by Dynamo around
    the compile itself, on whichever thread that compile happens on, which is
    the only spelling that reaches the worker; this pins that we ask for it.
    """
    seen = {}

    def _fake_compile(fn, **kwargs):
        seen.update(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)
    assert gn_mod._compile_group_norm() is gn_mod._group_norm
    assert seen.get("recompile_limit") == gn_mod._MIN_RECOMPILE_LIMIT
    assert seen.get("fullgraph") is True and seen.get("dynamic") is False


def test_compiling_still_works_without_a_per_region_limit(monkeypatch):
    """A torch too old for ``recompile_limit=`` must still get a callable.

    The keyword is the fix for the worker thread, not a requirement for
    compiling at all; dropping the whole rung on a ``TypeError`` would be a far
    bigger regression than the case it addresses.
    """
    calls = []

    def _fake_compile(fn, **kwargs):
        calls.append(kwargs)
        if "recompile_limit" in kwargs:
            raise TypeError("compile() got an unexpected keyword 'recompile_limit'")
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)
    assert gn_mod._compile_group_norm() is gn_mod._group_norm
    assert len(calls) == 2 and "recompile_limit" not in calls[1]


def test_a_recompile_limit_hit_is_a_kernel_failure_not_a_crash(monkeypatch, caplog):
    """``FailOnRecompileLimitHit`` has to land in the ladder, not in the run.

    It is what ``fullgraph=True`` raises when a frame needs more cache entries
    than the recompile limit allows, and -- unlike every other Dynamo failure --
    it derives straight from ``Exception`` rather than from
    ``TorchDynamoException``, so an allowlist that names only the latter lets it
    escape and kill the step (observed at ``5943389``).  It is raised while
    compiling, before the callable has run or saved anything, so the eager
    retry underneath it is safe.
    """
    import torch._dynamo.exc

    limit_hit = torch._dynamo.exc.FailOnRecompileLimitHit
    assert not issubclass(limit_hit, torch._dynamo.exc.TorchDynamoException), (
        "naming it separately is only needed while it sits outside that root"
    )

    def _kernel(*args, **kwargs):
        raise limit_hit("simulated recompile limit hit")

    monkeypatch.setattr(
        gn_mod, "_use_compiled", lambda t, **kw: type(t) is torch.Tensor
    )
    monkeypatch.setattr(gn_mod, "_get_compiled_group_norm", lambda: _kernel)
    gn_mod._compile_failed = False

    fast = _seeded_norm()
    x = _make_input(seed=51, channels=16, size=4)
    expected = nn.functional.group_norm(x, _GROUPS, fast.weight, fast.bias, fast.eps)

    with caplog.at_level(logging.WARNING, logger=gn_mod.__name__):
        out = fast(x)

    assert torch.allclose(out, expected), "the eager fallback did not run"
    assert gn_mod._compile_failed is True, "the failure did not latch the rung off"
    assert fast._compiled_ok is False
    assert any("falling back" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# DCTensor routing: unwrap -> compiled kernel -> rewrap
# ---------------------------------------------------------------------------


@pytest.fixture
def dc_cpu(gloo_group_1rank):
    """DistConv package plus a CPU ParallelStrategy over the 1-rank group.

    ``num_shards=(1, 1, 1)`` on dims (2, 3, 4) is exactly what worker.py builds
    for a single-device run; a process group must exist even then.
    """
    import distconv

    ps = distconv.ParallelStrategy(
        num_shards=(1, 1, 1), shard_dim=(2, 3, 4), device_type="cpu"
    )
    return distconv, ps


def _seeded_norm(channels=16):
    """A FastGroupNorm with non-default affine params (the defaults are 1/0,
    which would let a kernel that drops weight/bias slip through)."""
    fast = FastGroupNorm(_GROUPS, channels)
    generator = torch.Generator().manual_seed(97)
    with torch.no_grad():
        fast.weight.normal_(1.0, 0.1, generator=generator)
        fast.bias.normal_(0.0, 0.1, generator=generator)
    return fast


def test_dctensor_routes_through_compiled_kernel(monkeypatch, dc_cpu):
    """A DCTensor input reaches the compiled callable as its plain local shard.

    Verified with a recording stand-in for the compiled callable: it must see
    exactly ``torch.Tensor`` (Dynamo cannot trace the wrapper), and the caller
    must get a DCTensor back with the same values the stock kernel produces.
    """
    distconv, ps = dc_cpu
    seen = []

    def _recording(input, num_groups, weight, bias, eps):
        seen.append(type(input))
        return nn.functional.group_norm(input, num_groups, weight, bias, eps)

    monkeypatch.setattr(
        gn_mod, "_use_compiled", lambda t, **kw: type(t) is torch.Tensor
    )
    monkeypatch.setattr(gn_mod, "_get_compiled_group_norm", lambda: _recording)

    fast = _seeded_norm()
    x = _make_input(seed=21, channels=16, size=4)
    out = fast(distconv.DCTensor.from_shard(x, ps))

    assert seen == [torch.Tensor]
    assert isinstance(out, distconv.DCTensor)
    reference = nn.functional.group_norm(x, _GROUPS, fast.weight, fast.bias, fast.eps)
    assert torch.equal(out._tensor, reference)


def test_dctensor_gradients_reach_the_layer_upstream(monkeypatch, dc_cpu):
    """Gradients must flow past GroupNorm into the layer that produced its input.

    The unwrap has to go through DistConv's autograd pair rather than a bare
    ``input._tensor`` read.  The distinction is invisible when the DCTensor
    wraps a leaf -- the leaf *is* ``_tensor``, so even a bare read reaches it --
    which is why this test puts a producer in front, as production does
    (``conv -> GroupNorm`` in every block).  With a bare read the graph is
    severed there: the input and the producer's weight get no gradient at all
    while GroupNorm's own weight/bias still look healthy.
    """
    distconv, ps = dc_cpu
    monkeypatch.setattr(
        gn_mod, "_use_compiled", lambda t, **kw: type(t) is torch.Tensor
    )
    monkeypatch.setattr(
        gn_mod, "_get_compiled_group_norm", lambda: nn.functional.group_norm
    )

    fast = _seeded_norm()
    reference = nn.GroupNorm(_GROUPS, 16)
    producer = nn.Conv3d(16, 16, 1, bias=False)
    reference_producer = nn.Conv3d(16, 16, 1, bias=False)
    with torch.no_grad():
        reference.weight.copy_(fast.weight)
        reference.bias.copy_(fast.bias)
        reference_producer.weight.copy_(producer.weight)

    x_fast = _make_input(seed=22, channels=16, size=4).requires_grad_(True)
    x_ref = x_fast.detach().clone().requires_grad_(True)

    # A 1x1x1 conv on a DCTensor takes DistConv's convolution path, so the
    # DCTensor handed to GroupNorm is a genuine intermediate, not a leaf.
    out = fast(producer(distconv.DCTensor.from_shard(x_fast, ps)))
    assert isinstance(out, distconv.DCTensor)
    # Unwrap the way a downstream consumer would (autograd-aware) and drive a
    # scalar backward through it.
    distconv.distconv._ToTensor.apply(out).pow(2).sum().backward()
    reference(reference_producer(x_ref)).pow(2).sum().backward()

    assert x_fast.grad is not None, "gradient never reached the input"
    assert producer.weight.grad is not None, "gradient never reached the producer"
    assert torch.equal(x_fast.grad, x_ref.grad)
    assert torch.equal(producer.weight.grad, reference_producer.weight.grad)
    assert torch.equal(fast.weight.grad, reference.weight.grad)
    assert torch.equal(fast.bias.grad, reference.bias.grad)


def test_dctensor_on_cpu_never_invokes_torch_compile(monkeypatch, dc_cpu):
    """The unwrap route obeys the same CPU guard as plain tensors.

    On CPU the local shard fails the ``is_cuda`` check, so a DCTensor must fall
    through to the stock eager dispatch -- and still come back wrapped.  The
    stand-in records instead of only raising: ``forward`` catches ``Exception``
    to fall back, so a raise alone would be swallowed by the very code path
    under test and the assertion would never fire.
    """
    distconv, ps = dc_cpu
    calls = []

    def _boom(*a, **kw):
        calls.append(a)
        raise AssertionError("torch.compile must not be called for CPU tensors")

    monkeypatch.setattr(torch, "compile", _boom)
    monkeypatch.setattr(gn_mod, "_compiled_group_norm", None)
    gn_mod.set_compile_enabled(True)  # even when explicitly forced on

    fast = _seeded_norm()
    x = _make_input(seed=23, channels=16, size=4)
    out = fast(distconv.DCTensor.from_shard(x, ps))

    assert not calls
    assert isinstance(out, distconv.DCTensor)
    reference = nn.functional.group_norm(x, _GROUPS, fast.weight, fast.bias, fast.eps)
    assert torch.equal(out._tensor, reference)


def test_dctensor_compile_failure_falls_back_to_eager(monkeypatch, caplog, dc_cpu):
    """A broken compiler degrades the wrapped path to eager, like the plain one."""
    distconv, ps = dc_cpu
    import torch._dynamo.exc

    def _raises(*args, **kwargs):
        raise torch._dynamo.exc.Unsupported("simulated Inductor failure")

    monkeypatch.setattr(
        gn_mod, "_use_compiled", lambda t, **kw: type(t) is torch.Tensor
    )
    monkeypatch.setattr(gn_mod, "_get_compiled_group_norm", lambda: _raises)
    gn_mod._compile_failed = False

    fast = _seeded_norm()
    x = _make_input(seed=24, channels=16, size=4)
    with caplog.at_level(logging.WARNING, logger=gn_mod.__name__):
        out = fast(distconv.DCTensor.from_shard(x, ps))

    assert isinstance(out, distconv.DCTensor)
    reference = nn.functional.group_norm(x, _GROUPS, fast.weight, fast.bias, fast.eps)
    assert torch.equal(out._tensor, reference)
    assert gn_mod._compile_failed is True
    assert any("falling back to the eager kernel" in r.message for r in caplog.records)


def test_dctensor_two_shards_matches_eager_and_normalizes_per_shard():
    """Shard count > 1: same values as the eager route, still per-shard stats.

    Every other DCTensor test here runs ``num_shards=(1, 1, 1)``, where the
    local shard is the whole tensor and the sharding is a no-op -- so none of
    them can catch a fast path that quietly reduced over the wrong set of
    elements.  This one splits a spatial dim over two ranks and asserts both
    halves of the claim the fast path rests on: bit-identical to the eager
    wrapped route, and normalizing the local shard rather than the global
    volume (which is DistConv's existing semantics, not something this change
    introduces).
    """
    script = (
        Path(__file__).resolve().parent
        / "helpers"
        / "rank_scripts"
        / "groupnorm_shards_2rank.py"
    )
    rc, out, err = mpi_runner.torchrun_gloo(str(script), n=2, timeout=180)
    assert rc == 0, f"2-rank job failed rc={rc}\nstdout:\n{out}\nstderr:\n{err[-3000:]}"

    results = {
        rank: match
        for rank, *match in re.findall(
            # Bounded alternatives, not \S+: the ranks' lines can arrive
            # concatenated, so a greedy final field would swallow the next
            # line's "RESULT".
            r"RESULT rank=(\d+) shape=(\S+) identical=(True|False) "
            r"per_shard=(True|False) global=(True|False)",
            out,
        )
    }
    assert set(results) == {"0", "1"}, f"missing ranks\nstdout:\n{out}"
    for rank, (shape, identical, per_shard, matches_global) in results.items():
        assert identical == "True", f"rank {rank}: compiled route differs from eager"
        assert per_shard == "True", f"rank {rank}: not per-shard statistics"
        assert matches_global == "False", f"rank {rank}: matched global statistics"
        # Each rank holds half of the sharded dim.
        assert shape == "1x16x4x8x8", f"rank {rank}: unexpected shard shape {shape}"


@pytest.fixture
def dc_cuda():
    """DistConv package plus a CUDA ParallelStrategy over a 1-rank NCCL group."""
    import distconv
    import torch.distributed as dist

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        created = True
    ps = distconv.ParallelStrategy(
        num_shards=(1, 1, 1), shard_dim=(2, 3, 4), device_type="cuda"
    )
    yield distconv, ps
    if created and dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last_3d"])
def test_gpu_dctensor_matches_eager_dctensor(dc_cuda, autocast, layout):
    """The compiled unwrap path matches today's eager wrapped path on GPU.

    This is the production configuration: worker.py wraps every activation in
    a DCTensor (even at dc_num_shards=[1,1,1]), which used to force the eager
    kernel.  Values and gradients must agree within reduction-order noise, the
    output must still be a DCTensor, and the compile must actually engage.

    Both layouts are covered because production requests ``channels_last_3d``
    (worker.py) and, with ``PYTORCH_MIOPEN_SUGGEST_NHWC=1`` set as it is there,
    the convolutions really do hand GroupNorm channels-last activations.  Only
    parity is asserted, not the output layout: both routes compared here return
    contiguous regardless of the input layout (the Triton kernel, which does
    not, is pinned off below and covered by its own tests).
    """
    distconv, ps = dc_cuda
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(31)
    x = torch.randn(1, 64, 32, 32, 32, device=device, generator=generator)
    if layout == "channels_last_3d":
        x = x.to(memory_format=torch.channels_last_3d)
    grad_out = torch.randn(*x.shape, device=device, generator=generator)

    fast = FastGroupNorm(_GROUPS, 64).to(device)
    with torch.no_grad():
        fast.weight.normal_(1.0, 0.1, generator=generator)
        fast.bias.normal_(0.0, 0.1, generator=generator)

    def plain(t):
        return t._tensor if isinstance(t, distconv.DCTensor) else t

    def run(compiled):
        # This test is about the compiled rung of the ladder, which the Triton
        # one would otherwise pre-empt on the channels-last parametrization.
        gn_mod.set_triton_enabled(False)
        gn_mod.set_compile_enabled(compiled)
        inp = x.clone().requires_grad_(True)
        fast.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = fast(distconv.DCTensor.from_shard(inp, ps))
        assert isinstance(out, distconv.DCTensor)
        local = distconv.distconv._ToTensor.apply(out)
        local.backward(grad_out.to(local.dtype))
        return (
            local.detach(),
            inp.grad.detach().clone(),
            plain(fast.weight.grad).detach().clone(),
            plain(fast.bias.grad).detach().clone(),
        )

    eager = run(False)
    compiled = run(True)
    assert gn_mod._compiled_group_norm is not None, "compiled path was not taken"
    assert not gn_mod._compile_failed

    _assert_close(compiled[0], eager[0], 1e-5, "output")
    _assert_close(compiled[1], eager[1], 1e-4, "d_input")
    _assert_close(compiled[2], eager[2], 1e-4, "d_weight")
    _assert_close(compiled[3], eager[3], 1e-4, "d_bias")
    assert compiled[0].dtype == eager[0].dtype


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
        # Compiled-rung test: the inputs here are contiguous, which the Triton
        # kernel declines anyway, but pin it off so the routing cannot drift.
        gn_mod.set_triton_enabled(False)
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

    gn_mod.set_triton_enabled(False)  # this is the compiled rung's guard set
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


# ---------------------------------------------------------------------------
# Whole-network gradient comparisons
# ---------------------------------------------------------------------------
#
# Two tests below run the whole UNet twice, changing only which rung serves
# GroupNorm, and ask whether the gradients agree.  *How* that is asked matters
# more than it looks, and both tests used to ask it in a way that could only
# pass by luck.
#
# **Per-parameter relative L2 under bf16 autocast is not a bounded quantity for
# this model.**  Against an fp64 reference, every arm -- eager, compiled and
# Triton alike -- is 12.1-12.5% off on the bottleneck's parameters, whose
# gradients are 170x smaller than the largest in the network.  The difference
# between two arms is therefore the difference of two ~12% errors, and it is
# small only when they happen to cancel.  Whether they cancel is settled
# *outside the source tree*: two of this model's 33 convolution problems have
# MIOpen algorithms whose benchmark times tie to within 11%, so which one wins
# is frozen into the machine-local find database, and rewriting only those two
# recorded times moves "compiled vs. eager" from 5.3e-3 to 5.4e-2 with nothing
# else changed.  Three of the four algorithm combinations land at 5.3-6.7e-3
# and the fourth at 5.4e-2 -- which is exactly the history of this file, an
# assertion that was intermittent and then, once a cache went warm, failed
# deterministically at a bit-identical value.
#
# It is not a gate either: injecting a 1e-4 relative error into the compiled
# rung moves the per-parameter figure only from 5.4e-2 to 1.1e-1, because both
# are already saturated by the bf16 floor.
#
# So rung equivalence is asserted where it is measurable -- **without
# autocast**, where the same comparison reads 1.3e-6 and that same injected
# 1e-4 error reads 1.9e-4, a 154x separation -- while the bf16 autocast run,
# which is the production combination and the one the checkpoint machinery has
# to survive, is asserted on the *aggregate* gradient.  That is stable
# (5.0e-3 against a 3.2e-3 run-to-run floor, across every algorithm combination
# measured) and still catches a wrong eps (6.9e-2) or a wrong group count
# (5.2e-1).

#: Cross-rung agreement without autocast.  Measured 1.3e-6 (compiled vs. eager)
#: and 2.2e-6 (Triton vs. compiled); 1e-4 leaves ~50x headroom and still fails
#: on a 1e-4 relative kernel error, which reads 1.9e-4 here.
_FP32_RUNG_TOLERANCE = 1e-4

#: Aggregate agreement under bf16 autocast, and per-parameter agreement between
#: two runs *on the same rung* (where the bf16 error is common-mode and the
#: figure sits on the model's own run-to-run floor).
_BF16_TOLERANCE = 5e-2


def _relative_l2(actual, expected):
    """Relative L2 over the concatenated gradient of every parameter."""
    names = sorted(expected)
    a = torch.cat([actual[name].float().flatten() for name in names])
    b = torch.cat([expected[name].float().flatten() for name in names])
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _worst_relative_l2(actual, expected):
    """The largest per-parameter relative L2, with the parameter it belongs to."""
    worst = (0.0, "")
    for name, reference in expected.items():
        reference = reference.float()
        error = (actual[name].float() - reference).norm().item()
        worst = max(worst, (error / max(reference.norm().item(), 1e-12), name))
    return worst


@pytest.mark.gpu
def test_gpu_activation_checkpointing_matches_eager():
    """The compiled kernel must survive recompute under use_checkpointing().

    Non-reentrant checkpointing replays the block's forward inside the backward
    pass; a compiled region has to produce the same activations both times or
    the gradients silently change.

    Three assertions, each on a quantity it can actually bound -- see the
    "Whole-network gradient comparisons" note above for why that distinction is
    the whole point here:

    * checkpointed vs. non-checkpointed **on the same rung**, per parameter.
      This is the one the test is named for, and it is well posed because the
      bf16 error is common-mode: it reads 3.9e-3, the model's own run-to-run
      floor, under every convolution algorithm measured.
    * compiled vs. eager under bf16 autocast, on the *aggregate* gradient
      (5.0e-3 measured, against the same 3.2e-3 floor).
    * compiled vs. eager **without autocast**, per parameter -- the sharp one,
      1.3e-6 measured against a 1e-4 tolerance.
    """
    device = torch.device("cuda")
    x = _make_input(seed=9).to(device).requires_grad_(True)

    def grads(compiled, checkpointing, autocast=True):
        gn_mod.set_triton_enabled(False)
        gn_mod.set_compile_enabled(compiled)
        model = _make_unet(seed=0).to(device)
        if checkpointing:
            model.use_checkpointing()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = model(x)
        out.float().pow(2).mean().backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    eager = grads(False, True)
    compiled = grads(True, True)
    compiled_nockpt = grads(True, False)
    assert gn_mod._compiled_group_norm is not None, "compiled path was not taken"
    assert not gn_mod._compile_failed

    relative, name = _worst_relative_l2(compiled_nockpt, compiled)
    assert relative < _BF16_TOLERANCE, f"grad {name}: rel L2 {relative:.3e}"

    relative = _relative_l2(compiled, eager)
    assert relative < _BF16_TOLERANCE, f"checkpointed grad: rel L2 {relative:.3e}"

    relative, name = _worst_relative_l2(
        grads(True, True, autocast=False), grads(False, True, autocast=False)
    )
    assert relative < _FP32_RUNG_TOLERANCE, f"fp32 grad {name}: rel L2 {relative:.3e}"


@pytest.mark.gpu
def test_gpu_the_recompile_limit_holds_on_a_worker_thread():
    """Past Dynamo's stock 8 entries, compiling from a thread that never set it.

    ``torch._dynamo.config``'s user overrides live in a ``ContextVar``, so the
    limit :func:`_raise_recompile_limit` writes on the main thread is not the
    limit another thread reads -- and the compiles that matter happen on
    another thread, because ``torch.utils.checkpoint``'s recompute runs inside
    the backward pass, on the autograd engine's device worker.  This drives the
    same shape of traffic directly: entries live on ``_group_norm``'s code
    object and are shared between threads, so a worker that pushes the count
    past 8 is exactly the situation the recompute creates.  Before the
    per-region ``recompile_limit=``, the ninth compile raised
    ``FailOnRecompileLimitHit`` here.

    ``torch._dynamo.reset()`` first because those entries also accumulate
    across the whole test session, which would otherwise decide the outcome.
    """
    import threading

    torch._dynamo.reset()
    gn_mod.set_triton_enabled(False)  # this is the compiled rung's limit
    gn_mod.set_compile_enabled(True)
    gn_mod._compile_failed = False

    device = torch.device("cuda")
    # Nine distinct channel counts: nine cache entries, one more than the stock
    # limit allows, and the one that overflows must land on the worker thread.
    norms = [FastGroupNorm(_GROUPS, 8 * n).to(device) for n in range(1, 10)]
    failures = []

    def run(subset):
        try:
            for norm in subset:
                norm(torch.randn(1, norm.num_channels, 2, 2, 2, device=device))
        except BaseException as error:  # noqa: BLE001 - re-raised below
            failures.append(error)

    run(norms[:2])
    worker = threading.Thread(target=run, args=(norms[2:],))
    worker.start()
    worker.join()

    if failures:
        raise AssertionError(f"compiling off the main thread failed: {failures[0]}")
    assert gn_mod._compile_failed is False, "the rung latched itself off"
    assert all(norm._compiled_ok for norm in norms), (
        "some module never had a call served by the compiled rung"
    )


@pytest.mark.gpu
def test_gpu_checkpointed_dctensor_recompute_keeps_the_compiled_rung(dc_cuda):
    """The three-way combination that used to die: ckpt + compiled rung + DCTensor.

    ``activation_checkpointing: true`` with ``SCAFFOLD_GROUPNORM_TRITON=0`` on
    DistConv activations is a supported configuration and it crashed: the
    recompute reaches this module with ``__torch_function__`` subclass handling
    *disabled* (DistConv's backward runs below it), which is part of Dynamo's
    ``GLOBAL_STATE`` guard, so it misses every cache entry the forward built and
    compiles a second set beside them -- twice the shapes, past 8 -- on the
    autograd worker thread, where the module's raised limit was invisible.  Each
    pair of the three is fine on its own; all three together raised
    ``FailOnRecompileLimitHit`` (at ``5943389``) or, once the ladder caught it
    and dropped a *proven* module to eager mid-recompute, ``CheckpointError``.

    Five norms is the smallest count that reproduces it: 5 forward entries plus
    5 recompute entries is 10, and the ninth compile is the one that overflows.
    The convolutions are what make the block's backward run below torch-function
    (a bare unwrap does not), and the loss is taken on the ``DCTensor`` for the
    same reason the trainer's is.
    """
    import threading

    import torch.utils.checkpoint

    distconv, ps = dc_cuda
    device = torch.device("cuda")
    channels = (8, 16, 24, 32, 40)

    torch._dynamo.reset()
    gn_mod.set_triton_enabled(False)
    gn_mod.set_compile_enabled(True)
    gn_mod._compile_failed = False

    torch.manual_seed(5)
    norms = [FastGroupNorm(_GROUPS, c).to(device) for c in channels]
    convs = [
        nn.Conv3d(previous, c, 1, bias=False).to(device)
        for previous, c in zip((1,) + channels[:-1], channels)
    ]
    tail = nn.Conv3d(channels[-1], 1, 1, bias=False).to(device)

    # Where each GroupNorm call happens, as Dynamo's GLOBAL_STATE guard sees it.
    states = set()

    def block(t):
        for conv, norm in zip(convs, norms):
            states.add(
                (
                    threading.current_thread() is threading.main_thread(),
                    torch._C._is_torch_function_enabled(),
                )
            )
            t = norm(conv(t))
        return tail(t)

    x = torch.randn(1, 1, 4, 4, 4, device=device)

    def step(checkpointing):
        for parameter in [x] + [
            p for m in convs + norms + [tail] for p in m.parameters()
        ]:
            parameter.grad = None
        x.requires_grad_(True)
        wrapped = distconv.DCTensor.from_shard(x, ps)
        if checkpointing:
            out = torch.utils.checkpoint.checkpoint(block, wrapped, use_reentrant=False)
        else:
            out = block(wrapped)
        out.float().square().mean().backward()
        return [norm.weight.grad.detach().clone() for norm in norms]

    checkpointed = step(True)
    step(True)  # a second step must not compile anything new either
    direct = step(False)

    assert (False, False) in states, (
        "the recompute did not run below torch-function off the main thread; "
        "this configuration no longer reproduces the guard split it targets"
    )
    assert gn_mod._compile_failed is False, "the compiled rung latched itself off"
    assert all(norm._compiled_ok for norm in norms), "a norm never ran compiled"
    for index, (recomputed, plain) in enumerate(zip(checkpointed, direct)):
        assert torch.isfinite(recomputed).all(), index
        _assert_close(recomputed, plain, 1e-4, f"norm {index} weight grad")


# ---------------------------------------------------------------------------
# GPU behavior: the Triton rung
# ---------------------------------------------------------------------------


def _channels_last(t):
    return t.is_contiguous(memory_format=torch.channels_last_3d)


@pytest.mark.gpu
@pytest.mark.parametrize("activation", [None, "relu"])
@pytest.mark.parametrize("autocast", [False, True])
def test_gpu_triton_matches_eager(activation, autocast):
    """The Triton kernel is the default for channels-last input and matches eager.

    ``(1, 64, 32^3)`` channels-last is the production shape family at unit-test
    size.  Three claims at once: the routing really picks Triton when nothing is
    forced; the values and gradients match the eager reference within
    reduction-order noise; and the fused activation equals an explicit ReLU on
    the eager result.

    The rung is established by spying on the entry point rather than by
    inspecting the output's layout: every rung now returns the *input's* memory
    format, deliberately (a fallback that returned contiguous re-broke the
    channels-last chain for every convolution after it), so layout no longer
    distinguishes them.  The layout is asserted separately, of both.
    """
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(11)
    x = torch.randn(1, 64, 32, 32, 32, device=device, generator=generator).to(
        memory_format=torch.channels_last_3d
    )
    grad_out = torch.randn(*x.shape, device=device, generator=generator)

    fast = FastGroupNorm(_GROUPS, 64, activation=activation).to(device)
    with torch.no_grad():
        fast.weight.normal_(1.0, 0.1, generator=generator)
        fast.bias.normal_(0.0, 0.1, generator=generator)

    calls = []
    original_triton_forward = FastGroupNorm._triton_forward

    def run(triton):
        gn_mod.set_triton_enabled(triton)
        gn_mod.set_compile_enabled(False)  # eager reference, not Inductor
        inp = x.clone().requires_grad_(True)
        fast.zero_grad(set_to_none=True)
        before = len(calls)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = fast(inp)
        out.backward(grad_out.to(out.dtype))
        return (
            out.detach(),
            inp.grad.detach().clone(),
            fast.weight.grad.detach().clone(),
            fast.bias.grad.detach().clone(),
            len(calls) - before,
        )

    def spy(self, local):
        calls.append(tuple(local.shape))
        return original_triton_forward(self, local)

    FastGroupNorm._triton_forward = spy
    try:
        eager = run(False)
        triton = run(None)  # None = the production default, no override at all
    finally:
        FastGroupNorm._triton_forward = original_triton_forward
    assert not gn_mod._triton_failed

    assert triton[4] == 1, "Triton path was not taken"
    # ... and the control: the reference really did *not* take it, so the
    # comparison below is between two kernels and not one kernel with itself.
    assert eager[4] == 0
    # Both preserve the input's channels-last layout; that is the contract now,
    # not a rung signature.
    assert _channels_last(triton[0]) and _channels_last(eager[0])
    _assert_close(triton[0], eager[0], 1e-5, "output")
    _assert_close(triton[1], eager[1], 1e-4, "d_input")
    _assert_close(triton[2], eager[2], 1e-4, "d_weight")
    _assert_close(triton[3], eager[3], 1e-4, "d_bias")
    # Autocast's fp32 policy for GroupNorm must survive the swap.
    assert triton[0].dtype == eager[0].dtype
    if activation == "relu":
        assert (triton[0] < 0).sum() == 0
        assert triton[0].max() > 0


@pytest.mark.gpu
@pytest.mark.parametrize("activation", [None, "relu"])
def test_gpu_triton_dctensor_matches_eager_and_stays_wrapped(dc_cuda, activation):
    """The production configuration: DCTensor in, DCTensor out, NDHWC preserved.

    worker.py wraps every activation in a DCTensor even at
    ``dc_num_shards=[1,1,1]``, so this -- not the plain-tensor case -- is the
    path the benchmark actually runs.  A producing convolution sits in front so
    that the DCTensor handed to GroupNorm is a genuine intermediate: the unwrap
    has to be the autograd-aware one or the gradient never reaches the conv.
    """
    distconv, ps = dc_cuda
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(53)
    x = torch.randn(1, 64, 16, 16, 16, device=device, generator=generator).to(
        memory_format=torch.channels_last_3d
    )

    fast = FastGroupNorm(_GROUPS, 64, activation=activation).to(device)
    producer = nn.Conv3d(64, 64, 1, bias=False).to(
        device, memory_format=torch.channels_last_3d
    )
    with torch.no_grad():
        fast.weight.normal_(1.0, 0.1, generator=generator)
        fast.bias.normal_(0.0, 0.1, generator=generator)

    def plain(t):
        return t._tensor if isinstance(t, distconv.DCTensor) else t

    def run(triton):
        gn_mod.set_triton_enabled(triton)
        gn_mod.set_compile_enabled(False)
        inp = x.clone().requires_grad_(True)
        fast.zero_grad(set_to_none=True)
        producer.zero_grad(set_to_none=True)
        out = fast(producer(distconv.DCTensor.from_shard(inp, ps)))
        assert isinstance(out, distconv.DCTensor), "DCTensor did not survive"
        local = distconv.distconv._ToTensor.apply(out)
        local.float().pow(2).sum().backward()
        assert inp.grad is not None, "gradient never reached the input"
        assert producer.weight.grad is not None, "gradient never reached the producer"
        return (
            local.detach().clone(),
            inp.grad.detach().clone(),
            plain(producer.weight.grad).detach().clone(),
            plain(fast.weight.grad).detach().clone(),
            plain(fast.bias.grad).detach().clone(),
        )

    eager = run(False)
    triton = run(None)
    assert not gn_mod._triton_failed
    assert _channels_last(triton[0]), "Triton path was not taken (output not NDHWC)"

    _assert_close(triton[0], eager[0], 1e-5, "output")
    for index, what in (
        (1, "d_input"),
        (2, "d_producer"),
        (3, "d_weight"),
        (4, "d_bias"),
    ):
        _assert_close(triton[index], eager[index], 1e-4, what)


@pytest.mark.gpu
def test_gpu_unet_keeps_the_channels_last_chain(monkeypatch):
    """The whole point: GroupNorm stops breaking the layout chain in the model.

    Before this kernel, every one of the model's GroupNorms consumed
    ``channels_last_3d`` and emitted contiguous, forcing the next convolution to
    convert back -- 22 breaks per scale-8 forward.  A hook census asserts that
    every ``FastGroupNorm`` invocation now takes NDHWC in *and* hands NDHWC out.

    Needs ``PYTORCH_MIOPEN_SUGGEST_NHWC=1`` in the environment for the
    convolutions to emit channels-last at all; without it there is nothing to
    preserve and the test skips rather than passing vacuously.
    """
    device = torch.device("cuda")
    model = _make_unet(seed=0).to(device, memory_format=torch.channels_last_3d)
    x = _make_input(seed=9).to(device).contiguous(memory_format=torch.channels_last_3d)

    census = []

    def hook(module, inputs, output):
        census.append((_channels_last(inputs[0]), _channels_last(output)))

    for module in model.modules():
        if isinstance(module, FastGroupNorm):
            module.register_forward_hook(hook)

    gn_mod.set_triton_enabled(None)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        model(x)

    assert census, "no GroupNorm ran"
    if not any(seen_in for seen_in, _ in census):
        pytest.skip(
            "convolutions did not emit channels_last_3d; set "
            "PYTORCH_MIOPEN_SUGGEST_NHWC=1 (the production setting)"
        )
    breaks = [i for i, (seen_in, seen_out) in enumerate(census) if seen_in != seen_out]
    assert not breaks, f"GroupNorm broke the layout chain at sites {breaks}"
    assert all(seen_out for _, seen_out in census)


@pytest.mark.gpu
def test_gpu_unet_triton_matches_the_compiled_build(monkeypatch):
    """Whole-model gradients with the Triton kernel vs. without it.

    Asked twice, on the two quantities the "Whole-network gradient comparisons"
    note above establishes are bounded: the *aggregate* gradient under bf16
    autocast (1.9e-3 measured, against a 3.2e-3 run-to-run floor) and the
    per-parameter gradient **without** autocast, which is the sharp one --
    2.2e-6 measured, and the same 1e-4 tolerance the compiled rung is held to.
    Per-parameter under autocast, which this test used to assert, reads 1.5e-2
    against its 5e-2 tolerance here for reasons that have nothing to do with
    either kernel; see the note.

    Skips (rather than passing vacuously) when the convolutions are not emitting
    channels-last, since the Triton kernel would then never engage.
    """
    device = torch.device("cuda")
    x = _make_input(seed=9).to(device).contiguous(memory_format=torch.channels_last_3d)

    engaged = []
    original = FastGroupNorm._triton_forward

    def spy(self, local):
        engaged.append(tuple(local.shape))
        return original(self, local)

    monkeypatch.setattr(FastGroupNorm, "_triton_forward", spy)

    def grads(triton, autocast=True):
        gn_mod.set_triton_enabled(triton)
        gn_mod.set_compile_enabled(True)
        model = _make_unet(seed=0).to(device, memory_format=torch.channels_last_3d)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = model(x)
        out.float().pow(2).mean().backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    without = grads(False)
    assert not engaged
    with_triton = grads(None)
    if not engaged:
        pytest.skip(
            "Triton kernel never engaged; set PYTORCH_MIOPEN_SUGGEST_NHWC=1 "
            "(the production setting) so the convolutions emit channels-last"
        )
    assert not gn_mod._triton_failed

    relative = _relative_l2(with_triton, without)
    assert relative < _BF16_TOLERANCE, f"grad: rel L2 {relative:.3e}"

    engaged.clear()
    without_fp32 = grads(False, autocast=False)
    assert not engaged
    with_triton_fp32 = grads(None, autocast=False)
    assert engaged, "the Triton rung declined the same input without autocast"
    assert not gn_mod._triton_failed

    relative, name = _worst_relative_l2(with_triton_fp32, without_fp32)
    assert relative < _FP32_RUNG_TOLERANCE, f"fp32 grad {name}: rel L2 {relative:.3e}"
