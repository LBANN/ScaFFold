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

"""Edge cases of the GroupNorm *wiring* (``FastGroupNorm``'s three-rung ladder).

Written as an adversarial review of the wiring: ``tests/test_groupnorm.py``
covers the happy paths and the routing predicates, this file covers the places
where the ladder, the latches and the absorbed ReLU interact with the rest of
torch.

The review left ten of these as ``xfail(strict=True)``, one per defect, each
asserting the behaviour the module *should* have.  All ten are fixed and the
markers are gone; the tests stay, now as regression guards.  The properties
they pin, in the order the defects were found:

* the fused activation is bit-identical to ``F.relu`` on NaN, the infinities
  and ``-0.0``, forward and backward, on all three rungs -- and a NaN produced
  under it still reaches the trainer's non-finite-loss abort;
* a latch may not change the rung a checkpointed block is *recomputed* on;
* the ladder catches "the kernel is broken" and nothing else -- not the
  checkpoint machinery's control flow, not a user's saved-tensor hook, not an
  exception from after the rung already saved something;
* ``torch.func`` is a routing question, not a kernel failure;
* a latch can be cleared;
* ``activation`` is validated where it is used, and a module pickled before it
  existed still runs.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ScaFFold.unet import group_norm as gn_mod
from ScaFFold.unet.group_norm import FastGroupNorm
from ScaFFold.unet.unet_model import UNet

_GROUPS = 8


@pytest.fixture(autouse=True)
def _restore_routing_state():
    """Keep per-test overrides of the module-level routing state contained.

    Same contract as ``tests/test_groupnorm.py``'s fixture: several tests here
    deliberately trip a latch, which is a process global.
    """
    previous_compile = gn_mod.set_compile_enabled(None)
    previous_triton = gn_mod.set_triton_enabled(None)
    compile_failed = gn_mod._compile_failed
    triton_failed = gn_mod._triton_failed
    yield
    gn_mod._compile_override = previous_compile
    gn_mod._triton_override = previous_triton
    gn_mod._compile_failed = compile_failed
    gn_mod._triton_failed = triton_failed


def _cl(t):
    return t.is_contiguous(memory_format=torch.channels_last_3d)


def _cuda_norm(channels=64, activation="relu", size=16, seed=11):
    """A seeded ``FastGroupNorm`` plus a channels-last CUDA input for it."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, channels, size, size, size, device=device, generator=generator)
    x = x.to(memory_format=torch.channels_last_3d)
    module = FastGroupNorm(_GROUPS, channels, activation=activation).to(device)
    with torch.no_grad():
        module.weight.normal_(1.0, 0.1, generator=generator)
        module.bias.normal_(0.0, 0.1, generator=generator)
    return module, x


def _small_unet(device="cpu", channels_last=False):
    torch.manual_seed(0)
    model = UNet(
        n_channels=3, n_classes=2, trilinear=False, layers=1, group_norm_groups=_GROUPS
    )
    if channels_last:
        return model.to(device, memory_format=torch.channels_last_3d)
    return model.to(device)


def _triton_spy(monkeypatch):
    """Record every call that actually reached the Triton rung."""
    calls = []
    original = FastGroupNorm._triton_forward

    def spy(self, local):
        calls.append(tuple(local.shape))
        return original(self, local)

    monkeypatch.setattr(FastGroupNorm, "_triton_forward", spy)
    return calls


# ---------------------------------------------------------------------------
# activation semantics: every rung must apply the same function
# ---------------------------------------------------------------------------


def test_activate_handles_every_supported_activation():
    """``_activate`` must implement every activation the module advertises.

    ``SUPPORTED_ACTIVATIONS`` is what the *constructor* accepts and what
    ``is_supported`` is asked about, but the compiled and eager rungs apply it
    through ``_activate``, which tests one literal string.  Adding a second
    activation to both tuples (the only thing
    ``test_supported_activations_match_the_kernels`` checks) would fuse it into
    the Triton store and silently drop it everywhere else -- i.e. the network's
    function would depend on the memory format of its input.  This is the guard
    on that: for every non-``None`` activation, ``_activate`` has to *change*
    an input that the identity would leave alone.
    """
    x = torch.linspace(-2.0, 2.0, 64).reshape(1, 8, 2, 2, 2)
    for activation in gn_mod.SUPPORTED_ACTIVATIONS:
        module = FastGroupNorm(_GROUPS, 8, activation=activation)
        out = module._activate(x.clone())
        if activation is None:
            assert torch.equal(out, x)
        else:
            assert not torch.equal(out, x), (
                f"_activate is a no-op for activation={activation!r}: the "
                "compiled and eager rungs would silently skip it while the "
                "Triton rung fused it in"
            )


def test_affine_false_still_applies_the_activation():
    """``affine=False`` leaves ``weight``/``bias`` ``None`` on every rung."""
    module = FastGroupNorm(_GROUPS, 16, affine=False, activation="relu")
    x = torch.randn(1, 16, 4, 4, 4, generator=torch.Generator().manual_seed(5))
    assert torch.equal(
        module(x), F.relu(F.group_norm(x, _GROUPS, None, None, module.eps))
    )
    assert list(module.state_dict().keys()) == []


def test_activation_is_not_part_of_the_state():
    """``activation`` may not become a parameter, a buffer or a state-dict key."""
    with_relu = FastGroupNorm(_GROUPS, 16, activation="relu")
    without = FastGroupNorm(_GROUPS, 16)
    assert list(with_relu.state_dict().keys()) == list(without.state_dict().keys())
    assert list(with_relu.buffers()) == []
    # ... and a checkpoint written by one loads into the other, strict.
    result = without.load_state_dict(with_relu.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys


def test_double_conv_bytes_match_a_hand_built_pre_fusion_block():
    """Byte-for-byte state-dict identity against an independently built block.

    ``test_state_dict_bytes_identical_to_plain_groupnorm_model`` compares against
    a model produced by *converting* the fused one, which shares its
    construction order by definition.  This builds the pre-fusion
    ``nn.Sequential`` from scratch -- ``Conv3d, GroupNorm, ReLU, Conv3d,
    GroupNorm, ReLU`` -- and compares the serialized bytes, which is the
    independent version of the same claim.
    """
    from ScaFFold.unet.unet_parts import DoubleConv

    torch.manual_seed(17)
    fused = DoubleConv(3, 16, _GROUPS)

    torch.manual_seed(17)
    reference = nn.Sequential(
        nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(_GROUPS, 16),
        nn.ReLU(inplace=True),
        nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(_GROUPS, 16),
        nn.ReLU(inplace=True),
    )

    def blob(state_dict):
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()

    assert list(fused.double_conv.state_dict().keys()) == list(
        reference.state_dict().keys()
    )
    assert blob(fused.double_conv.state_dict()) == blob(reference.state_dict())


def test_module_pickled_before_the_fusion_still_runs():
    """A whole-module pickle predates ``self.activation``; forward must cope.

    ``nn.Module.__setstate__`` replaces ``__dict__`` wholesale, so an instance
    restored from a ``torch.save(model)`` written before the fusion has no
    ``activation`` at all -- and none of the routing state added since either.
    Every attribute this module reads outside ``__init__`` therefore needs a
    class-level default.
    """
    module = FastGroupNorm(_GROUPS, 16, activation="relu")
    state = module.__dict__.copy()
    for added_since in ("activation", "_triton_ok", "_compiled_ok"):
        state.pop(added_since, None)  # exactly what a pre-fusion pickle carries

    restored = FastGroupNorm.__new__(FastGroupNorm)
    nn.Module.__setstate__(restored, state)

    x = torch.randn(1, 16, 4, 4, 4, generator=torch.Generator().manual_seed(6))
    out = restored(x)
    # A pre-fusion pickle had no activation, so it must behave as one.
    assert torch.equal(
        out, F.group_norm(x, _GROUPS, restored.weight, restored.bias, restored.eps)
    )


def test_unsupported_activation_assigned_after_construction_is_caught():
    """``activation`` is validated where it is *used*, not only at construction.

    It is a plain attribute, so it can be assigned afterwards; ``is_supported``
    would then decline the Triton rung while ``_activate`` silently applied
    nothing, i.e. the module would quietly become a bare GroupNorm.  The same
    hole is the forward-looking risk: adding a third entry to both
    ``SUPPORTED_ACTIVATIONS`` tuples without implementing it in ``_activate``
    must not produce a network whose activation depends on its input's memory
    format.  Failing loudly on the rung that cannot apply it closes both.
    """
    module = FastGroupNorm(_GROUPS, 16, activation="relu")
    module.activation = "gelu"
    x = torch.randn(1, 16, 4, 4, 4, generator=torch.Generator().manual_seed(7))
    with pytest.raises(ValueError, match="activation must be one of"):
        module(x)


# ---------------------------------------------------------------------------
# the ladder must not swallow torch's own control flow
# ---------------------------------------------------------------------------


def test_base_exceptions_are_not_caught():
    """``KeyboardInterrupt``/``SystemExit`` must escape the ladder untouched."""
    for exception in (KeyboardInterrupt, SystemExit):

        def _raises(*args, **kwargs):
            raise exception()

        previous = gn_mod._get_triton_module
        gn_mod._get_triton_module = _raises
        original_use = gn_mod._use_triton
        gn_mod._use_triton = lambda *a, **kw: True
        gn_mod._triton_failed = False
        try:
            module = FastGroupNorm(_GROUPS, 16)
            with pytest.raises(exception):
                module(torch.randn(1, 16, 4, 4, 4))
            assert gn_mod._triton_failed is False
        finally:
            gn_mod._get_triton_module = previous
            gn_mod._use_triton = original_use


@pytest.mark.parametrize("rung", ["triton", "compiled"])
def test_checkpoint_error_is_re_raised(monkeypatch, rung):
    """``CheckpointError`` is the checkpoint machinery talking, not a kernel.

    It is raised by the recompute pack hook -- i.e. from inside whichever op is
    saving a tensor -- exactly like ``_StopRecomputationError``, and it is a
    ``RuntimeError`` subclass, so any handler wide enough to catch "a broken
    kernel" by type catches it too.  Swallowing it latches the rung off, retries
    on the next one and leaves the checkpoint frame in a state the machinery
    never expected.  The allowlist has to be narrow enough that this propagates
    untouched and nothing latches.
    """
    import torch.utils.checkpoint as checkpoint_mod

    def _raises(*args, **kwargs):
        raise checkpoint_mod.CheckpointError("simulated recompute mismatch")

    if rung == "triton":
        monkeypatch.setattr(gn_mod, "_use_triton", lambda *a, **kw: True)
        monkeypatch.setattr(gn_mod, "_get_triton_module", _raises)
    else:
        monkeypatch.setattr(gn_mod, "_use_compiled", lambda _input, **kw: True)
        monkeypatch.setattr(gn_mod, "_get_compiled_group_norm", lambda: _raises)
    gn_mod._triton_failed = False
    gn_mod._compile_failed = False

    module = FastGroupNorm(_GROUPS, 16)
    with pytest.raises(checkpoint_mod.CheckpointError):
        module(torch.randn(1, 16, 4, 4, 4))
    assert gn_mod._triton_failed is False
    assert gn_mod._compile_failed is False


def test_a_rung_failure_does_not_re_fire_saved_tensor_hooks(monkeypatch):
    """The retry has to be idempotent with respect to saved-tensor hooks.

    Under non-reentrant activation checkpointing the recompute counts pack-hook
    firings and requires the count *and* the metadata to match the forward's, so
    a rung that packed some tensors and then failed -- with the fallback packing
    its own set on top -- corrupts the frame.  A user's offloading hook has the
    same problem in a less dramatic way (an offload failure retried by
    offloading a second, larger set).

    The property is structural rather than defensive: the ladder catches only
    failures that are raised *before* their rung saves anything (the Triton op
    saves in ``_setup_context``, after the launch region its ``TritonKernelError``
    comes from; a Dynamo/Inductor error is a compile-time error, before
    execution).  Both halves are asserted here -- a caught failure packs exactly
    what a clean fallback packs, and an exception from *after* the packing is
    not the ladder's to swallow.
    """
    import torch._dynamo.exc

    class _Boom(Exception):
        pass

    def _fails_before_packing(input, num_groups, weight, bias, eps):
        raise torch._dynamo.exc.Unsupported("failed while compiling")

    def _fails_after_packing(input, num_groups, weight, bias, eps):
        F.group_norm(input, num_groups, weight, bias, eps)
        raise _Boom("failed after packing")

    monkeypatch.setattr(gn_mod, "_use_compiled", lambda _input, **kw: True)
    module = FastGroupNorm(_GROUPS, 16)
    x = torch.randn(1, 16, 4, 4, 4).requires_grad_(True)

    def packs_during(run):
        packed = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (packed.append(1), t)[1], lambda t: t
        ):
            run()
        return len(packed)

    baseline = packs_during(lambda: module._eager_forward(x))

    monkeypatch.setattr(
        gn_mod, "_get_compiled_group_norm", lambda: _fails_before_packing
    )
    gn_mod._compile_failed = False
    retried = packs_during(lambda: module(x))
    assert retried == baseline, (
        f"the failed rung packed {retried - baseline} extra tensors before the "
        "fallback ran"
    )

    # ... and a failure that *did* have observable effects is not retried at all.
    monkeypatch.setattr(
        gn_mod, "_get_compiled_group_norm", lambda: _fails_after_packing
    )
    gn_mod._compile_failed = False
    with pytest.raises(_Boom):
        module(x)


# ---------------------------------------------------------------------------
# latches
# ---------------------------------------------------------------------------


def test_forcing_a_rung_on_clears_its_failure_latch():
    """``set_*_enabled(True)`` is the documented way to retry after a failure.

    Without this a one-off failure (a transient OOM, a cache-directory hiccup)
    costs the rung for the rest of the process with no recovery at all, and the
    function's own docstring -- "forcing it on is overridden only by the
    correctness checks" -- is false.  ``None`` deliberately does *not* clear it:
    that restores a preference, it does not assert that the kernel works again.
    """
    for setter, latch in (
        (gn_mod.set_triton_enabled, "_triton_failed"),
        (gn_mod.set_compile_enabled, "_compile_failed"),
    ):
        setattr(gn_mod, latch, True)
        setter(True)
        assert getattr(gn_mod, latch) is False

        setattr(gn_mod, latch, True)
        setter(None)
        assert getattr(gn_mod, latch) is True
        setter(False)
        assert getattr(gn_mod, latch) is True
        setattr(gn_mod, latch, False)


@pytest.mark.parametrize("rung", ["triton", "compiled"])
def test_out_of_memory_is_not_recorded_as_a_kernel_failure(monkeypatch, rung):
    """A transient OOM must propagate, and must not latch a rung off forever.

    ``torch.OutOfMemoryError`` is a resource condition, not a defect: every
    fallback allocates an output of the same size, so retrying one is a second,
    differently-shaped OOM at a call site the caller never asked about.
    Latching on it is worse still -- a per-rank, nondeterministic event that
    permanently changes which kernel that rank runs, and therefore (measured)
    the all-reduced gradients of the whole job.
    """
    from ScaFFold.unet.triton_group_norm import TritonKernelError

    def _oom(*args, **kwargs):
        raise torch.OutOfMemoryError("simulated OOM")

    if rung == "triton":
        monkeypatch.setattr(gn_mod, "_use_triton", lambda *a, **kw: True)
        monkeypatch.setattr(gn_mod, "_get_triton_module", _oom)
    else:
        monkeypatch.setattr(gn_mod, "_use_compiled", lambda _input, **kw: True)
        monkeypatch.setattr(gn_mod, "_get_compiled_group_norm", lambda: _oom)
    gn_mod._triton_failed = False
    gn_mod._compile_failed = False

    module = FastGroupNorm(_GROUPS, 16)
    with pytest.raises(torch.OutOfMemoryError):
        module(torch.randn(1, 16, 4, 4, 4))
    assert gn_mod._triton_failed is False
    assert gn_mod._compile_failed is False
    # It is a RuntimeError, so a handler that caught the kernel's own error by
    # base class would have swallowed it; the allowlist is the tagged type.
    assert issubclass(torch.OutOfMemoryError, RuntimeError)
    assert not issubclass(torch.OutOfMemoryError, TritonKernelError)


def test_a_global_latch_does_not_demote_a_module_that_already_used_the_rung(
    monkeypatch, caplog
):
    """The unit of the latch is the module, not the process.

    A rung that has already served a module keeps serving it; only modules that
    have never used it are steered away.  That is what makes a checkpointed
    block's forward and its recompute agree (they save different tensors on
    different rungs, so a mid-graph change is fatal), and it is why a broken
    install still costs exactly one attempt per module rather than one per call.

    The corollary tested here too: because a proven module keeps retrying, the
    warning has to be emitted on the latch's edge rather than per call, or a
    persistently broken kernel floods the log for the rest of the run.
    """
    import logging

    import torch._dynamo.exc

    calls = []

    def _kernel(input, num_groups, weight, bias, eps):
        calls.append(1)
        if len(calls) > 1:
            raise torch._dynamo.exc.Unsupported("simulated Inductor failure")
        return F.group_norm(input, num_groups, weight, bias, eps)

    monkeypatch.setattr(gn_mod, "_use_compiled", gn_mod._use_compiled)
    monkeypatch.setattr(gn_mod, "_get_compiled_group_norm", lambda: _kernel)
    monkeypatch.setattr(
        gn_mod,
        "_use_compiled",
        lambda t, proven=False, **kw: (not gn_mod._compile_failed) or proven,
    )
    gn_mod._compile_failed = False

    proven = FastGroupNorm(_GROUPS, 16)
    fresh = FastGroupNorm(_GROUPS, 16)
    x = torch.randn(1, 16, 4, 4, 4)

    proven(x)  # succeeds: this module is now proven on the compiled rung
    assert proven._compiled_ok is True

    with caplog.at_level(logging.WARNING, logger=gn_mod.__name__):
        proven(x)  # fails, latches, falls back to eager
        assert gn_mod._compile_failed is True
        warned = sum("falling back" in r.message for r in caplog.records)
        proven(x)  # ... and still *tries* the rung, because it is proven
        proven(x)
        assert sum("falling back" in r.message for r in caplog.records) == warned, (
            "a persistently failing rung warned once per call"
        )
    assert len(calls) == 4, "the proven module stopped trying its rung"

    fresh(x)  # never used it, so the global latch keeps it away entirely
    assert len(calls) == 4
    assert fresh._compiled_ok is False


@pytest.mark.parametrize("rung", ["triton", "compiled"])
def test_a_proven_rung_does_not_degrade_while_a_backward_replays_it(monkeypatch, rung):
    """A fallback *during a recompute* corrupts rather than degrades.

    The latch already refuses to demote a module that has used a rung, but the
    fallback itself sidestepped that: the failing call still got answered from
    the next rung down, and if that call is ``torch.utils.checkpoint``'s
    recompute of a forward that ran on the failing rung, the recomputed forward
    saves a different set of tensors than the original did and torch rejects
    the whole step (``CheckpointError``; on one measured shape a GPU memory
    fault instead).  Neither is a degradation, so this one case re-raises.

    It is narrow on purpose -- ``_replaying_a_forward()`` is false in an
    ordinary forward, where ``test_a_global_latch_does_not_demote_a_module_...``
    still requires the fallback -- and the second half here pins the other side
    of the narrowness: a module that has *not* used the rung degrades even
    inside the backward, because the forward it is replaying went down the
    ladder too and the two agree.
    """
    import torch._dynamo.exc
    import torch.utils.checkpoint as checkpoint_mod

    from ScaFFold.unet.triton_group_norm import TritonKernelError

    failure = TritonKernelError if rung == "triton" else torch._dynamo.exc.Unsupported
    latch = "_triton_failed" if rung == "triton" else "_compile_failed"
    proven_flag = "_triton_ok" if rung == "triton" else "_compiled_ok"

    def make_kernel(fail_always):
        def _kernel(input, num_groups, weight, bias, eps, *activation):
            if fail_always or gn_mod._replaying_a_forward():
                raise failure("simulated kernel failure")
            return F.group_norm(input, num_groups, weight, bias, eps)

        return _kernel

    def run(fail_always):
        gn_mod._triton_failed = False
        gn_mod._compile_failed = False
        kernel = make_kernel(fail_always)
        if rung == "triton":
            monkeypatch.setattr(gn_mod, "_use_triton", lambda *a, **kw: True)
            module_stub = type("_Stub", (), {"triton_group_norm": staticmethod(kernel)})
            monkeypatch.setattr(gn_mod, "_get_triton_module", lambda: module_stub)
        else:
            monkeypatch.setattr(gn_mod, "_use_compiled", lambda t, **kw: True)
            monkeypatch.setattr(gn_mod, "_get_compiled_group_norm", lambda: kernel)
        module = FastGroupNorm(_GROUPS, 16)
        x = torch.randn(1, 16, 4, 4, 4, requires_grad=True)
        out = checkpoint_mod.checkpoint(module, x, use_reentrant=False)
        out.pow(2).sum().backward()
        return module, x

    # Proven in the forward, failing in the recompute: answering from another
    # rung would be the metadata mismatch, so the failure has to come back out.
    with pytest.raises(failure):
        run(fail_always=False)

    # Never served by the rung: the forward already went down the ladder, so
    # the recompute doing the same agrees with it and the step survives.
    module, x = run(fail_always=True)
    assert getattr(module, proven_flag) is False
    assert getattr(gn_mod, latch) is True
    assert torch.isfinite(x.grad).all()


def test_a_latch_flip_mid_forward_is_not_a_numerics_error(monkeypatch):
    """Two rungs inside one forward still compose (values, not bits, agree)."""
    monkeypatch.setattr(
        gn_mod, "_use_compiled", lambda t, **kw: type(t) is torch.Tensor
    )
    monkeypatch.setattr(
        gn_mod, "_get_compiled_group_norm", lambda: nn.functional.group_norm
    )
    gn_mod._compile_failed = False

    module = FastGroupNorm(_GROUPS, 16, activation="relu")
    x = torch.randn(1, 16, 4, 4, 4).requires_grad_(True)
    first = module(x)
    gn_mod._compile_failed = True  # latch flips between the two calls
    second = module(first)
    second.pow(2).sum().backward()
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------


def test_predicates_reject_a_parameter_input():
    """``nn.Parameter`` is a subclass, so both fast rungs decline it.

    Not a bug -- the model never feeds a Parameter to a norm -- but it is the
    documented consequence of the ``type(input) is torch.Tensor`` policy, and a
    regression that loosened it to ``isinstance`` would route real
    ``__torch_dispatch__`` wrappers into the kernel.
    """
    parameter = nn.Parameter(torch.randn(1, 8, 4, 4, 4))
    assert gn_mod._use_triton(parameter, _GROUPS, None, None, None) is False
    assert gn_mod._use_compiled(parameter) is False


def test_use_triton_is_side_effect_free_for_rejected_inputs(monkeypatch):
    """The predicate may look, but it may not allocate, launch or mutate."""
    module = FastGroupNorm(_GROUPS, 16)
    x = torch.randn(1, 16, 4, 4, 4)
    before = x.clone()
    assert gn_mod._use_triton(x, _GROUPS, module.weight, module.bias, None) is False
    assert torch.equal(x, before)


# ---------------------------------------------------------------------------
# GPU: the Triton rung in situ
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_gpu_double_backward_fails_loudly():
    """The kernel is first-order only; a second derivative must *raise*.

    ``triton_group_norm``'s backward is itself a custom op with no autograd
    formula, so a gradient penalty or an HVP through the wired model has to die
    with a clear message rather than silently return a wrong number -- and the
    failure must not be mistaken for a broken kernel and latch the rung off.
    """
    module, x = _cuda_norm(activation=None)
    x = x.clone().requires_grad_(True)
    out = module(x)
    assert _cl(out), "Triton rung was not taken; the test would be vacuous"

    (first,) = torch.autograd.grad(out.pow(2).sum(), x, create_graph=True)
    with pytest.raises(RuntimeError, match="no autograd formula"):
        torch.autograd.grad(first.pow(2).sum(), x)
    assert gn_mod._triton_failed is False, "a caller error latched the kernel off"


@pytest.mark.gpu
def test_gpu_model_double_backward_fails_loudly():
    """Same, through the wired model, which is where a user would hit it.

    Either rung may raise first and the message differs, so this matches both:
    ``triton_group_norm``'s backward is a custom op with **no autograd formula**,
    and the Triton convolution's is marked **@once_differentiable**.  Which one
    the traversal reaches first is a routing detail -- before the block-list was
    emptied on 2026-08-04 the ``Cin == 3`` stem was on MIOpen, which *is* twice
    differentiable, so the walk got all the way to GroupNorm.  Now the stem's own
    backward stops it, one op earlier.

    What must not change is that it raises *at all*.  Neither decorator was in
    place on the convolution until that routing change, and without it the
    failure was ``first`` silently arriving with no ``grad_fn`` -- a second
    backward then contributing zero rather than erroring.
    """
    model = _small_unet("cuda", channels_last=True)
    x = (
        torch.randn(1, 3, 16, 16, 16, device="cuda")
        .contiguous(memory_format=torch.channels_last_3d)
        .requires_grad_(True)
    )
    gn_mod.set_triton_enabled(None)
    out = model(x)
    (first,) = torch.autograd.grad(out.pow(2).sum(), x, create_graph=True)
    assert first.grad_fn is not None, (
        "the double-backward graph was severed instead of raising; a second "
        "backward would contribute zero silently"
    )
    with pytest.raises(RuntimeError, match="no autograd formula|once_differentiable"):
        first.pow(2).sum().backward()


@pytest.mark.gpu
@pytest.mark.parametrize("fullgraph", [True, False])
def test_gpu_triton_rung_inside_a_compiled_region(monkeypatch, fullgraph):
    """The Triton rung is *allowed* inside ``torch.compile``; prove it works.

    ``_use_compiled`` bails out when ``torch.compiler.is_compiling()`` so the
    functional GroupNorm inlines, but ``_use_triton`` has no such guard: an
    enclosing compiled region traces straight into the custom op.  Nothing in
    ScaFFold compiles ``FastGroupNorm.forward`` today, so this was untested in
    situ.  Values, gradients *and* the channels-last output must survive
    Dynamo/AOTAutograd unchanged.
    """
    import torch._dynamo

    module, x = _cuda_norm(activation="relu")
    reference_input = x.clone().requires_grad_(True)
    reference = module(reference_input)
    reference.pow(2).sum().backward()
    reference_grad = reference_input.grad.detach().clone()
    module.zero_grad(set_to_none=True)
    assert _cl(reference), "Triton rung was not taken; the test would be vacuous"

    torch._dynamo.reset()
    calls = _triton_spy(monkeypatch)
    compiled = torch.compile(lambda t: module(t), fullgraph=fullgraph, dynamic=False)

    compiled_input = x.clone().requires_grad_(True)
    out = compiled(compiled_input)
    out.pow(2).sum().backward()

    assert calls, "the Triton rung was not traced inside the compiled region"
    assert _cl(out), "the compiled region lost the channels-last output"
    assert torch.equal(out, reference)
    assert torch.equal(compiled_input.grad, reference_grad)


@pytest.mark.gpu
@pytest.mark.parametrize("proven", [False, True])
def test_gpu_the_fallback_path_traces_under_fullgraph(monkeypatch, proven):
    """A rung failure *while Dynamo is tracing* must still fall back, not die.

    The handler used to call ``logger.warning``, which Dynamo cannot trace
    ("Unsupported: logging.Logger method not supported for non-export cases"),
    so a caller compiling this forward with ``fullgraph=True`` got a hard error
    instead of the fallback -- the one caller for whom the fallback matters
    most, since the thing it is reacting to is usually a compile-time failure.
    Nothing in ScaFFold compiles ``FastGroupNorm.forward`` today; this pins the
    claim that it can.

    Both halves of the handler's guard have to trace, which is why ``proven``
    is parametrized: with ``_triton_ok`` false Dynamo folds the ``and`` away
    without ever looking at ``_replaying_a_forward()``, so only the ``True``
    arm reaches it -- and a probe Dynamo cannot trace there would be the same
    defect as the logging call, reintroduced.
    """
    import torch._dynamo

    from ScaFFold.unet.triton_group_norm import TritonKernelError

    module, x = _cuda_norm(activation="relu")
    reference = module(x).detach().clone()
    assert _cl(reference), "Triton rung was not taken; the test would be vacuous"

    real = gn_mod._get_triton_module()

    class _BrokenKernelModule:
        def __getattr__(self, name):
            if name == "triton_group_norm":

                def _raises(*args, **kwargs):
                    raise TritonKernelError("simulated Triton failure")

                return _raises
            return getattr(real, name)

    monkeypatch.setattr(gn_mod, "_get_triton_module", _BrokenKernelModule)
    gn_mod._triton_failed = False
    module._triton_ok = proven
    torch._dynamo.reset()

    compiled = torch.compile(lambda t: module(t), fullgraph=True, dynamic=False)
    out = compiled(x)

    assert gn_mod._triton_failed is True, "the latch was not recorded"
    assert _cl(out), "the fallback rung dropped the channels-last chain"
    assert (out - reference).abs().max().item() < 1e-5


@pytest.mark.gpu
def test_gpu_inference_mode_takes_the_triton_rung(monkeypatch):
    """``evaluate()`` runs the whole model under ``torch.inference_mode``."""
    module, x = _cuda_norm(activation="relu")
    module.eval()
    reference = F.relu(F.group_norm(x, _GROUPS, module.weight, module.bias, module.eps))
    calls = _triton_spy(monkeypatch)
    with torch.inference_mode():
        out = module(x)
    assert calls, "inference_mode fell off the Triton rung"
    assert _cl(out)
    assert out.is_inference()
    assert (out.float() - reference.float()).abs().max().item() < 1e-5


@pytest.mark.gpu
def test_gpu_evaluation_shaped_forward_matches_training_shaped_one(monkeypatch):
    """``eval()`` + ``inference_mode`` + autocast is the evaluate() combination."""
    model = _small_unet("cuda", channels_last=True)
    x = torch.randn(1, 3, 16, 16, 16, device="cuda").contiguous(
        memory_format=torch.channels_last_3d
    )
    gn_mod.set_triton_enabled(None)
    model.eval()
    calls = _triton_spy(monkeypatch)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
    if not calls:
        pytest.skip(
            "convolutions did not emit channels_last_3d; set "
            "PYTORCH_MIOPEN_SUGGEST_NHWC=1 (the production setting)"
        )
    assert torch.isfinite(out.float()).all()
    assert gn_mod._triton_failed is False


@pytest.mark.gpu
def test_gpu_vmap_over_the_module_still_works():
    """``torch.func`` is a routing question, not a kernel failure.

    ``is_supported``'s ``is_contiguous(memory_format=...)`` raises outright
    under a ``vmap`` layer ("NYI"), so a predicate that reaches it -- or a
    relayout helper that does -- turns a plain ``nn.GroupNorm`` drop-in into a
    hard error for any caller using ``torch.func``.  Both fast rungs must
    decline while a transform is active and let the stock kernel answer.
    """
    module, x = _cuda_norm(activation=None)
    batched = torch.stack([x[0], x[0]])
    out = torch.func.vmap(lambda t: module(t.unsqueeze(0)).squeeze(0))(batched)
    assert out.shape == batched.shape
    assert torch.allclose(out[0], module(x[None, 0]).squeeze(0), atol=1e-5)


@pytest.mark.gpu
def test_gpu_a_predicate_that_cannot_answer_falls_back_without_latching(
    monkeypatch, caplog
):
    """``is_supported`` raising is a routing miss, and "no" is always a valid answer.

    The predicate runs *outside* the ladder's try, so anything it raises escapes
    ``forward()`` -- which is how a ``torch.func`` transform used to turn a
    drop-in ``nn.GroupNorm`` into a hard error.  The functorch check upstream
    covers the one caller known to trip it; this covers the shape of the
    problem, because ``is_supported`` inspects an *arbitrary* tensor and the set
    of wrappers that can make an attribute read raise is not closed.  A broad
    catch is right here and nowhere else in this module: the predicate has done
    no work anyone can observe and a correct answer ("use the stock kernel") is
    always available -- so it must fall back, and must not latch, because
    nothing about the kernel has been learned.
    """
    import logging

    class _Unanswerable:
        def __getattr__(self, name):
            if name == "is_supported":

                def _raises(*args, **kwargs):
                    raise RuntimeError("NYI: querying is_contiguous inside of vmap")

                return _raises
            return getattr(gn_mod._get_triton_module(), name)

    module, x = _cuda_norm(activation="relu")
    reference = module(x).detach().clone()
    monkeypatch.setattr(gn_mod, "_get_triton_module", _Unanswerable)
    monkeypatch.setattr(gn_mod, "_predicate_warned", False)
    gn_mod._triton_failed = False
    module._triton_ok = False

    with caplog.at_level(logging.WARNING, logger=gn_mod.__name__):
        out = module(x)
    assert (out - reference).abs().max().item() < 1e-5
    assert gn_mod._triton_failed is False, "a routing miss latched the kernel off"
    assert gn_mod._compile_failed is False
    assert any("routing check failed" in r.message for r in caplog.records)


@pytest.mark.gpu
def test_gpu_torch_func_grad_does_not_latch_the_rungs_off():
    """A ``torch.func`` call anywhere must not demote the whole process.

    ``torch.func.grad`` used to reach the kernel, fail, and latch *both* fast
    rungs off permanently -- i.e. one transform anywhere in a process silently
    dropped every GroupNorm in the model to the stock kernel for the rest of
    the run.  Recording a routing miss as a kernel failure is the general shape
    of the bug; this pins the specific instance.
    """
    module, x = _cuda_norm(activation=None)
    gn_mod._triton_failed = False
    gn_mod._compile_failed = False

    grad = torch.func.grad(lambda t: module(t).pow(2).sum())(x)

    assert torch.isfinite(grad).all()
    assert gn_mod._triton_failed is False
    assert gn_mod._compile_failed is False
    # ... and the module is still on the Triton rung afterwards.
    assert _cl(module(x))


@pytest.mark.gpu
def test_gpu_a_triton_failure_during_a_checkpointed_step_degrades_not_dies():
    """A latch may not change the rung a checkpointed block is recomputed on.

    Non-reentrant checkpointing compares the metadata of every tensor the
    recomputed forward saves against the forward's, and the three rungs save
    *different* tensors -- Triton ``(input, weight, bias, mean, rstd)``, the
    other two ``(input, weight, mean, rstd, relu_output)``.  So a rung change
    between a block's forward and its recompute kills the step with a
    ``CheckpointError``, which is the exact opposite of the ladder's contract
    ("a broken Triton install must degrade a multi-node run, not kill it") and
    is reachable whenever the ``activation_checkpointing`` option is on
    (``worker.py:230``).  Matching the output memory format is *not* enough on
    its own -- measured; the saved sets still differ -- so the fix is that a
    global latch does not demote a module that has already used the rung.

    The same hazard predates the Triton rung: flipping ``_compile_failed``
    between forward and recompute dies too, which is why both latches are
    checked here.
    """
    for latch in ("_triton_failed", "_compile_failed"):
        model = _small_unet("cuda", channels_last=True)
        model.use_checkpointing()
        x = (
            torch.randn(1, 3, 16, 16, 16, device="cuda")
            .contiguous(memory_format=torch.channels_last_3d)
            .requires_grad_(True)
        )
        gn_mod.set_triton_enabled(None)
        gn_mod._triton_failed = False
        gn_mod._compile_failed = False

        out = model(x)
        # A one-off failure at any *later* GroupNorm site latches the rung off
        # while the blocks already run are waiting to be recomputed.
        setattr(gn_mod, latch, True)
        out.pow(2).sum().backward()

        assert torch.isfinite(x.grad).all(), latch


@pytest.mark.gpu
@pytest.mark.parametrize("channels_last", [True, False])
def test_gpu_every_rung_returns_the_inputs_memory_format(channels_last):
    """All three rungs must agree on the output layout, not just the values.

    ``F.group_norm`` -- eager or Inductor-compiled -- returns a *contiguous*
    tensor whatever it was given, so a single fallback used to re-break the
    channels-last chain for every convolution after it, which is the exact
    thing this module exists to prevent.  It also made the rungs distinguishable
    to anything that inspects metadata (``torch.utils.checkpoint``, a compiled
    caller's guards), which is a correctness problem rather than a speed one.
    """
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(1, 64, 8, 8, 8, device=device, generator=generator)
    x = x.to(memory_format=torch.channels_last_3d) if channels_last else x.contiguous()
    module = FastGroupNorm(_GROUPS, 64, activation="relu").to(device)

    outputs = _run_on_every_rung(module, x)
    for label, out in outputs.items():
        assert _cl(out) is channels_last, (
            f"the {label} rung returned "
            f"{'channels_last_3d' if _cl(out) else 'contiguous'} for a "
            f"{'channels_last_3d' if channels_last else 'contiguous'} input"
        )
    reference = outputs["eager"].detach().float()
    for label, out in outputs.items():
        assert (out.detach().float() - reference).abs().max().item() < 1e-5, label


@pytest.mark.gpu
def test_gpu_triton_rejects_a_cuda_tensor_subclass():
    """The subclass check has to be tested on a tensor that would otherwise pass.

    ``test_triton_rejects_unknown_tensor_subclasses`` hands ``_use_triton`` a
    *CPU* subclass, which the ``is_cuda`` check rejects one line later -- so it
    cannot tell whether the ``type(input) is torch.Tensor`` test exists at all
    (a mutation deleting that line survives the whole suite).  The check is
    load-bearing: ``is_supported`` only asks ``isinstance``, so without it every
    unknown ``__torch_dispatch__`` wrapper would be routed into the kernel.
    """

    class _Wrapper(torch.Tensor):
        pass

    x = torch.randn(1, 64, 8, 8, 8, device="cuda").to(
        memory_format=torch.channels_last_3d
    )
    wrapped = x.as_subclass(_Wrapper)
    # The control: everything *except* the subclass test accepts this tensor.
    from ScaFFold.unet import triton_group_norm as triton_mod

    assert triton_mod.is_supported(wrapped, _GROUPS, None, None, None) is True
    assert gn_mod._use_triton(wrapped, _GROUPS, None, None, None) is False
    assert gn_mod._use_compiled(wrapped) is False


#: NaN, +Inf, -Inf, -0.0 and four ordinary values -- everything the fused
#: activation has to agree with ``F.relu`` on.  ``tl.maximum(y, 0)`` and
#: ``tl.where(y > 0, y, 0)`` both map NaN to 0.0 (the first returns the non-NaN
#: operand, the second because ``NaN > 0`` is False); ``F.relu`` propagates it.
_SPECIAL_VALUES = [float("nan"), float("inf"), float("-inf"), -0.0, 0.0, -1.0, 1.0, 2.0]


def _run_on_every_rung(module, x):
    """``{rung: output}`` for the same module and input on all three rungs."""
    results = {}
    for label, triton, compiled in (
        ("triton", True, False),
        ("compiled", False, True),
        ("eager", False, False),
    ):
        gn_mod.set_triton_enabled(triton)
        gn_mod.set_compile_enabled(compiled)
        results[label] = module(x)
    return results


def _bits(t):
    return t.detach().float().cpu().contiguous().view(torch.int32)


@pytest.mark.gpu
@pytest.mark.parametrize("activation", ["relu", None])
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_gpu_all_rungs_agree_on_nan_and_inf(poison, activation):
    """One non-finite input value must poison the same elements on every rung.

    The Triton store used ``tl.maximum(y, 0.0)``, which returns the *non*-NaN
    operand, so a diverging activation came back from the Triton rung as a
    finite 0.0 while ``F.relu`` on the other two kept it NaN.  That is worse
    than a numerics discrepancy: the forward looks finite while the backward is
    still NaN, so the run sails past ScaFFold's non-finite-loss abort and
    checkpoints a broken model -- and the model's output becomes a function of
    its input's memory format.  ``activation=None`` is the control: all three
    agreed there even before the fix, which is what localizes the divergence to
    the fused activation.
    """
    device = torch.device("cuda")
    x = torch.randn(1, 64, 4, 4, 4, device=device)
    x.view(-1)[0] = poison
    x = x.to(memory_format=torch.channels_last_3d)
    module = FastGroupNorm(_GROUPS, 64, activation=activation).to(device)

    results = {
        label: out.detach().float().cpu().contiguous().isnan()
        for label, out in _run_on_every_rung(module, x).items()
    }
    assert int(results["eager"].sum()) > 0, "the poison did not reach the output"
    assert torch.equal(results["compiled"], results["eager"])
    assert torch.equal(results["triton"], results["eager"]), (
        f"the fused activation turned {int(results['eager'].sum())} NaNs into "
        f"{int(results['triton'].sum())}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("activation", ["relu", None])
def test_gpu_fused_activation_is_bit_identical_to_relu(activation):
    """Every special value of the *pre-activation*, on every rung, bit for bit.

    Poisoning the input can only produce NaN pre-activations (one NaN or Inf
    makes the whole group's statistics NaN), so the four values that actually
    distinguish the spellings of ReLU are reached the other way round: a zero
    ``weight`` makes the pre-activation exactly ``bias``, elementwise, so the
    bias vector chooses what the activation sees.  Expected, per
    ``F.relu``: NaN stays NaN, ``+Inf`` stays ``+Inf``, ``-Inf`` and both zeros
    become ``+0.0`` (never ``-0.0``).

    With ``activation=None`` the zeros are normalized (``+ 0.0`` maps ``-0.0``
    to ``+0.0`` and leaves NaN, the infinities and every normal value alone)
    before the same bitwise comparison: a ``-0.0`` bias survives to the output
    there, and whether ``xhat * 0 + (-0.0)`` keeps the sign depends on whether
    the kernel contracted the multiply-add into an FMA -- true of the Triton
    *and* the Inductor rung, false of eager, and nothing to do with the
    activation.  ``torch.equal`` is no use for either case: it reports NaN as
    unequal to itself.
    """
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(3)
    x = torch.randn(1, 64, 4, 4, 4, device=device, generator=generator).to(
        memory_format=torch.channels_last_3d
    )
    module = FastGroupNorm(_GROUPS, 64, activation=activation).to(device)
    with torch.no_grad():
        module.weight.zero_()
        module.bias.copy_(torch.tensor(_SPECIAL_VALUES * 8, device=device))

    reference = F.group_norm(x, _GROUPS, module.weight, module.bias, module.eps)
    if activation == "relu":
        reference = F.relu(reference)

    for label, out in _run_on_every_rung(module, x).items():
        if activation == "relu":
            assert torch.equal(_bits(out), _bits(reference)), (
                f"{label} rung differs from F.relu(F.group_norm(...)) in the "
                "bit pattern of at least one special value"
            )
        else:
            assert torch.equal(_bits(out + 0.0), _bits(reference + 0.0)), label


@pytest.mark.gpu
def test_gpu_fused_relu_backward_gates_like_threshold_backward():
    """ReLU's backward passes the gradient where the output is NaN, too.

    ``threshold_backward(grad, result, 0)`` zeroes where ``result <= 0``, and
    ``NaN <= 0`` is False -- so a NaN pre-activation passes its gradient.  The
    kernel recomputes the pre-activation and must gate with the same
    complement; ``pre > 0 ? dy : 0`` would silently zero it.
    """
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(4)
    base = torch.randn(1, 64, 4, 4, 4, device=device, generator=generator).to(
        memory_format=torch.channels_last_3d
    )
    module = FastGroupNorm(_GROUPS, 64, activation="relu").to(device)
    with torch.no_grad():
        module.weight.zero_()
        module.bias.copy_(torch.tensor(_SPECIAL_VALUES * 8, device=device))

    grads = {}
    for label, triton in (("triton", True), ("eager", False)):
        gn_mod.set_triton_enabled(triton)
        gn_mod.set_compile_enabled(False)
        x = base.clone().requires_grad_(True)
        module.zero_grad(set_to_none=True)
        module(x).sum().backward()
        grads[label] = (x.grad.detach().clone(), module.bias.grad.detach().clone())

    # d_bias is exactly the gate: one per element that passed.
    assert torch.equal(_bits(grads["triton"][1]), _bits(grads["eager"][1]))
    assert grads["eager"][1][0].item() > 0, "the NaN lane's gradient was gated off"
    assert torch.equal(
        _bits(grads["triton"][0].float()), _bits(grads["eager"][0].float())
    )


@pytest.mark.gpu
def test_gpu_fused_relu_nan_still_trips_the_trainers_non_finite_guard(
    monkeypatch, tiny_trainer
):
    """End to end: a NaN produced under the fused path reaches the abort.

    ScaFFold aborts a run whose reduced epoch losses are non-finite, precisely
    so a diverged run stops instead of overwriting ``checkpoint_last.pth`` with
    NaN weights.  A fused activation that ate the NaN would hand that guard a
    finite loss and let the run continue on a model whose *gradients* are still
    NaN.  The loss below is the real one: a real UNet, on the GPU, with the
    Triton rung verified to have served every GroupNorm in it.
    """
    from ScaFFold.utils import trainer as trainer_mod

    model = _small_unet("cuda", channels_last=True)
    poisoned = torch.randn(1, 3, 16, 16, 16, device="cuda")
    poisoned.view(-1)[0] = float("nan")
    x = poisoned.contiguous(memory_format=torch.channels_last_3d).requires_grad_(True)
    gn_mod.set_triton_enabled(None)
    calls = _triton_spy(monkeypatch)
    loss = model(x).float().mean()
    if not calls:
        pytest.skip(
            "convolutions did not emit channels_last_3d; set "
            "PYTORCH_MIOPEN_SUGGEST_NHWC=1 (the production setting)"
        )
    assert not torch.isfinite(loss).item(), (
        "the fused activation swallowed the NaN: the forward is finite while "
        "the backward is not, which is exactly what hides divergence"
    )

    trainer = tiny_trainer(config_overrides={"checkpoint_interval": 1, "epochs": 3})
    monkeypatch.setattr(
        trainer,
        "_run_training_batch",
        lambda batch, **kw: (1, loss.detach().cpu(), torch.tensor(0.0)),
    )
    # A *finite* validation loss, so the abort can only come from the model's.
    monkeypatch.setattr(
        trainer_mod, "evaluate", lambda *a, **kw: (7.4e-10, 0.5, 0.5, 2, 2)
    )
    trainer.cleanup_or_resume()
    with pytest.raises(ValueError, match="[Nn]on-finite"):
        trainer.train()
    assert not trainer.checkpoint_manager.last_ckpt_path.exists()


@pytest.mark.gpu
def test_gpu_ddp_wrapped_model_takes_the_triton_rung(monkeypatch):
    """DDP's module-tree walk must not be confused by the nn.Identity slots."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    created = False
    if not dist.is_initialized():
        import os

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29623")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        created = True
    try:
        model = _small_unet("cuda", channels_last=True)
        wrapped = DistributedDataParallel(model, device_ids=[0])
        x = torch.randn(1, 3, 16, 16, 16, device="cuda").contiguous(
            memory_format=torch.channels_last_3d
        )
        gn_mod.set_triton_enabled(None)
        calls = _triton_spy(monkeypatch)
        wrapped(x).pow(2).sum().backward()
        if not calls:
            pytest.skip(
                "convolutions did not emit channels_last_3d; set "
                "PYTORCH_MIOPEN_SUGGEST_NHWC=1 (the production setting)"
            )
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()
