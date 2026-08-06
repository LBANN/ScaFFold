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

"""The hardware guard: ``_rungs._platform_declines`` and both ladders' wiring.

**Every interesting branch here is one this node cannot take.**  The guard's job
is to keep the Triton rungs off hardware they were not tuned on, and the machine
running these tests is the hardware they *were* tuned on -- so a suite that only
exercised the accept path would ship the entire decline path unexecuted, which
is the opposite of what the change is for.  ``_rungs._device_fingerprint`` exists
as one small seam for that reason: substituting a tuple for it poses the MI300X,
the partitioned MI300A, the MI250X and the NVIDIA questions to the *real*
predicate, the real cache and the real message, rather than to a re-implementation
of them.

Two properties are easy to get wrong and are pinned separately.  The cache is
process-global, so :func:`_clean_platform_state` clears it around every test --
without that the first question asked would fix the answer for the rest of the
session and half these tests would silently be re-asking it.  And the guard must
be evaluated *once*, which is asserted as a call count on the seam rather than as
a duration, because a timing assertion on a driver query measures the driver.

Every property here was verified by mutation: breaking it in the guard and
checking that the test named alongside it notices.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from ScaFFold.unet import _rungs
from ScaFFold.unet import conv3d as conv_mod
from ScaFFold.unet import group_norm as gn_mod
from ScaFFold.unet._rungs import format_kernel_selection, kernel_selection
from ScaFFold.unet.conv3d import FastConv3d, FastConvTranspose3d
from ScaFFold.unet.group_norm import FastGroupNorm

_CHANNELS_LAST = torch.channels_last_3d

#: What this node reports, and the only fingerprint the guard accepts.
_MI300A = ("gfx942", 228, "AMD Instinct MI300A")

#: The parts an arch-only test would wrongly accept.  MI300X and MI325X are
#: ``gfx942`` too -- discrete GPUs with 304 CUs rather than an APU with 228 --
#: and a compute-partitioned MI300A reports the same arch with one XCD's worth
#: of CUs, which makes both the 228 and ``gather_gemm``'s ``GROUP_M = 6``
#: (MI300A's XCD count) fiction while the arch string never moves.
_UNTUNED = {
    "mi300x": ("gfx942", 304, "AMD Instinct MI300X"),
    "mi325x": ("gfx942", 304, "AMD Instinct MI325X"),
    "mi300a-cpx": ("gfx942", 38, "AMD Instinct MI300A"),
    "mi250x": ("gfx90a", 104, "AMD Instinct MI250X"),
    "next-gen": ("gfx950", 256, "AMD Instinct MI355X"),
    # A CUDA build of torch has no ``gcnArchName`` at all, so the fingerprint
    # reports an empty arch.  Declining is right for a reason beyond tuning:
    # the kernels' launch rules are MFMA rules and mean nothing without MFMA.
    "nvidia": ("", 132, "NVIDIA H100"),
}


@pytest.fixture(autouse=True)
def _clean_platform_state():
    """Clear the verdict cache and both ladders' overrides around every test.

    The cache is a process-global memo, on purpose -- the whole point is that
    the question is asked once.  That makes it test state: a faked MI300X left
    behind would turn every later GPU test in the session into a fallback test,
    and a real verdict left behind would make a decline test pass by answering
    the wrong question.  Cleared on both sides for that reason.
    """
    _rungs._reset_platform_cache()
    saved = (
        conv_mod._triton_override,
        conv_mod._triton_failed,
        gn_mod._triton_override,
        gn_mod._triton_failed,
    )
    yield
    (
        conv_mod._triton_override,
        conv_mod._triton_failed,
        gn_mod._triton_override,
        gn_mod._triton_failed,
    ) = saved
    _rungs._reset_platform_cache()


def _fake_device(monkeypatch, fingerprint, *, count=None):
    """Make every device look like ``fingerprint``; optionally count the asks.

    ``count`` is a list the seam appends each asked-about index to, which is how
    the "evaluated once" tests assert on a call count rather than on a duration.

    The cache is dropped here as well, and that is not tidying: swapping the
    hardware out from under a memo whose whole purpose is never to ask twice
    would otherwise leave the *previous* answer in place -- which on this node
    is "yes, MI300A", so every decline test would quietly become another accept
    test.  That is exactly the failure this file exists to avoid.
    """

    def fingerprint_of(index):
        if count is not None:
            count.append(index)
        return fingerprint

    monkeypatch.setattr(_rungs, "_device_fingerprint", fingerprint_of)
    _rungs._reset_platform_cache()


def _cuda(index=0):
    return torch.device("cuda", index)


# ---------------------------------------------------------------------------
# the predicate
# ---------------------------------------------------------------------------


def test_the_tuned_fingerprint_is_the_only_one_accepted(monkeypatch):
    """Arch *and* CU count, which is what makes this MI300A and not gfx942."""
    _fake_device(monkeypatch, _MI300A)
    ok, described = _rungs._platform_verdict(_cuda())
    assert ok is True
    assert "MI300A" in described


@pytest.mark.parametrize("name", sorted(_UNTUNED))
def test_every_other_device_is_declined(monkeypatch, name):
    """Including the three an arch-only predicate would have accepted."""
    _fake_device(monkeypatch, _UNTUNED[name])
    ok, described = _rungs._platform_verdict(_cuda())
    assert ok is False, f"{name} is not the device anything here was tuned on"
    assert _UNTUNED[name][2] in described
    assert _rungs._platform_declines(_cuda(), None) is True


def test_the_arch_feature_suffixes_are_not_part_of_the_comparison(monkeypatch):
    """``gcnArchName`` carries build features; exact equality would be a trap.

    This node reports ``gfx942:sramecc+:xnack-``, and those suffixes describe
    how the *build* was configured rather than which silicon is present -- so a
    string comparison against ``"gfx942"`` would decline the very device
    everything was tuned on, and one against the full string would decline the
    same chip under a different HIP build.  Driven through a stub property
    object so the CPU suite exercises it too; the GPU test below pins that the
    real device really does carry a suffix, i.e. that this is not hypothetical.
    """

    class _Props:
        gcnArchName = "gfx942:sramecc+:xnack-"
        multi_processor_count = 228
        name = "AMD Instinct MI300A"

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: _Props())
    assert _rungs._device_fingerprint(0) == _MI300A
    assert _rungs._platform_declines(_cuda(), None) is False


def test_a_device_that_cannot_be_answered_for_is_not_the_tuned_one(monkeypatch):
    """ "I could not find out" is not "yes"; the description says which."""

    def explode(index):
        raise RuntimeError("HIP error: no device")

    monkeypatch.setattr(_rungs, "_device_fingerprint", explode)
    ok, described = _rungs._platform_verdict(_cuda())
    assert ok is False
    assert "RuntimeError" in described and "no device" in described


def test_the_verdict_is_computed_once_per_device(monkeypatch):
    """A call count, not a timing: the query is a driver call on a hot path.

    Also the per-device half of the decision: device 1 gets its own answer
    rather than inheriting device 0's, so a node exposing two different parts
    cannot have one of them silently answered for by the other.
    """
    asked = []
    _fake_device(monkeypatch, _MI300A, count=asked)
    for _ in range(20):
        _rungs._platform_declines(_cuda(0), None)
    assert asked == [0]
    for _ in range(20):
        _rungs._platform_declines(_cuda(1), None)
    assert asked == [0, 1]


def test_a_mixed_node_answers_per_device(monkeypatch):
    """One MI300A and one MI300X: each device gets the routing it deserves."""
    fingerprints = {0: _MI300A, 1: _UNTUNED["mi300x"]}
    monkeypatch.setattr(_rungs, "_device_fingerprint", lambda i: fingerprints[i])
    assert _rungs._platform_declines(_cuda(0), None) is False
    assert _rungs._platform_declines(_cuda(1), None) is True


# ---------------------------------------------------------------------------
# what it says
# ---------------------------------------------------------------------------


def test_declining_is_quiet_but_says_so_exactly_once(monkeypatch, caplog):
    """One message per device: not per call, and not silence.

    Silence is the state this change ends -- a user on an MI300X should be able
    to discover why the fast path is off.  Per call would be a log line every
    few milliseconds, since a scale-7 step routes some forty of these.
    """
    _fake_device(monkeypatch, _UNTUNED["mi300x"])
    with caplog.at_level(logging.WARNING, logger=_rungs.__name__):
        for _ in range(25):
            assert _rungs._platform_declines(_cuda(), None) is True
    records = [r for r in caplog.records if "tuned for" in r.message]
    assert len(records) == 1, [r.message for r in records]
    message = records[0].message
    # It has to name what it found and what it wanted, or it is not actionable.
    assert "MI300X" in message and "304 CUs" in message
    assert "gfx942" in message and "228" in message
    assert "SCAFFOLD_CONV_TRITON=1" in message
    assert "SCAFFOLD_GROUPNORM_TRITON=1" in message


def test_the_tuned_platform_says_nothing_at_all(monkeypatch, caplog):
    """No message on the machine everything was measured on."""
    _fake_device(monkeypatch, _MI300A)
    with caplog.at_level(logging.WARNING, logger=_rungs.__name__):
        for _ in range(5):
            assert _rungs._platform_declines(_cuda(), None) is False
    assert caplog.records == []


def test_a_mixed_node_names_each_declining_device(monkeypatch, caplog):
    """The bound is one message per device, which is what makes it discoverable."""
    fingerprints = {0: _UNTUNED["mi300x"], 1: _UNTUNED["mi250x"]}
    monkeypatch.setattr(_rungs, "_device_fingerprint", lambda i: fingerprints[i])
    with caplog.at_level(logging.WARNING, logger=_rungs.__name__):
        for index in (0, 1, 0, 1, 0):
            _rungs._platform_declines(_cuda(index), None)
    messages = [r.message for r in caplog.records]
    assert len(messages) == 2
    assert any("cuda:0" in m and "MI300X" in m for m in messages)
    assert any("cuda:1" in m and "MI250X" in m for m in messages)


# ---------------------------------------------------------------------------
# the override
# ---------------------------------------------------------------------------


def test_an_explicit_opt_in_takes_the_rung_anyway_and_is_loud(monkeypatch, caplog):
    """The guard is a preference, so an explicit "yes" wins -- audibly.

    Loud because every figure either ladder is read against was measured on the
    other machine, so a timing produced under this override is not comparable
    with any of them and the log is the only place that survives the run.
    """
    _fake_device(monkeypatch, _UNTUNED["mi300x"])
    with caplog.at_level(logging.WARNING, logger=_rungs.__name__):
        for _ in range(25):
            assert _rungs._platform_declines(_cuda(), True) is False
    records = [r for r in caplog.records if "explicitly enabled" in r.message]
    assert len(records) == 1, [r.message for r in records]
    assert "MI300X" in records[0].message


def test_the_default_is_not_an_opt_in(monkeypatch):
    """``None`` means "on wherever it is safe", and this device is not that.

    The tri-state is the whole override: an unset ``SCAFFOLD_CONV_TRITON`` and
    an explicit ``1`` differ *here* and nowhere else.
    """
    _fake_device(monkeypatch, _UNTUNED["mi300x"])
    assert _rungs._platform_declines(_cuda(), None) is True
    assert _rungs._platform_declines(_cuda(), True) is False


@pytest.mark.parametrize(
    "module, setter",
    [
        (conv_mod, "set_conv_triton_enabled"),
        (gn_mod, "set_triton_enabled"),
    ],
)
def test_each_ladders_setter_is_the_override_for_that_ladder(
    monkeypatch, module, setter
):
    """``set_*_triton_enabled(True)`` is the in-process spelling of the opt-in.

    Per ladder, deliberately: the two kernels' tuning tables are separate bodies
    of work, so a developer who has satisfied themselves about one has said
    nothing about the other.
    """
    _fake_device(monkeypatch, _UNTUNED["mi300x"])
    getattr(module, setter)(None)
    assert _rungs._platform_declines(_cuda(), module._triton_override) is True
    getattr(module, setter)(True)
    assert _rungs._platform_declines(_cuda(), module._triton_override) is False
    getattr(module, setter)(False)
    # ``False`` never reaches the guard -- both callers decline on it first --
    # but it must not be mistaken for the opt-in if it ever does.
    assert _rungs._platform_declines(_cuda(), module._triton_override) is True


@pytest.mark.parametrize("module", [conv_mod, gn_mod])
def test_the_env_var_opt_in_reaches_the_guard(monkeypatch, module):
    """``SCAFFOLD_*_TRITON=1`` and the setter are the same statement."""
    monkeypatch.setenv(module.TRITON_ENV_VAR, "1")
    monkeypatch.setattr(
        module, "_triton_override", _rungs._env_override(module.TRITON_ENV_VAR)
    )
    _fake_device(monkeypatch, _UNTUNED["mi300x"])
    assert _rungs._platform_declines(_cuda(), module._triton_override) is False


# ---------------------------------------------------------------------------
# the CPU path
# ---------------------------------------------------------------------------


def test_a_cpu_convolution_never_asks_the_hardware(monkeypatch):
    """Ordering: ``is_cuda`` is tested before the device is ever queried.

    Not a nicety.  ``torch.cuda.get_device_properties`` initializes torch's CUDA
    state, so asking it on the routing path of a CPU tensor would drag a GPU
    context into the whole CPU unit suite -- and into any CPU-only run of
    ScaFFold, which must keep working.

    Asserted as "the seam was never reached" rather than as a raising stub: the
    verdict is computed inside a broad ``except``, which would swallow the stub's
    own exception and let the test pass while the query happened.
    """
    asked = []
    _fake_device(monkeypatch, _MI300A, count=asked)
    conv = FastConv3d(16, 16, kernel_size=3, padding=1, bias=False)
    x = torch.randn(1, 16, 8, 8, 8)
    assert conv_mod._use_triton(conv, x, None, None) is False
    assert asked == []
    torch.testing.assert_close(conv(x), nn.Conv3d.forward(conv, x))


def test_a_cpu_group_norm_never_asks_the_hardware(monkeypatch):
    """The same ordering in the other ladder, asserted the same way."""
    asked = []
    _fake_device(monkeypatch, _MI300A, count=asked)
    module = FastGroupNorm(8, 16)
    x = torch.randn(1, 16, 8, 8, 8)
    assert (
        gn_mod._use_triton(x, module.num_groups, module.weight, module.bias, None)
        is False
    )
    assert asked == []
    torch.testing.assert_close(module(x), nn.GroupNorm.forward(module, x))


# ---------------------------------------------------------------------------
# the wiring: this node, told it is another one
# ---------------------------------------------------------------------------


def _gpu_conv(cin=16, cout=16, **kwargs):
    kwargs.setdefault("kernel_size", 3)
    kwargs.setdefault("padding", 1)
    kwargs.setdefault("bias", False)
    torch.manual_seed(11)
    conv = FastConv3d(cin, cout, **kwargs)
    return conv.cuda().to(memory_format=_CHANNELS_LAST).to(torch.bfloat16)


def _gpu_input(shape, dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
    return x.to(dtype).contiguous(memory_format=_CHANNELS_LAST)


@pytest.mark.gpu
def test_this_node_is_the_tuned_platform(caplog):
    """The accept path, against the real driver rather than a stub.

    Also the check that the constants have not drifted away from the machine
    every measurement in this project was taken on -- and that the suffix strip
    is load-bearing here rather than defensive, since the raw string this device
    reports really does carry ``:sramecc+:xnack-``.
    """
    props = torch.cuda.get_device_properties(0)
    assert ":" in props.gcnArchName, props.gcnArchName
    arch, cus, _name = _rungs._device_fingerprint(0)
    assert (arch, cus) == (_rungs.TUNED_ARCH, _rungs.TUNED_CU_COUNT)
    with caplog.at_level(logging.WARNING, logger=_rungs.__name__):
        assert _rungs._platform_declines(_cuda(), None) is False
    assert caplog.records == []


@pytest.mark.gpu
def test_an_untuned_device_sends_the_convolution_to_miopen(monkeypatch):
    """Declining is the ordinary fallback, not an error: same answer, MIOpen."""
    conv = _gpu_conv()
    x = _gpu_input((1, 16, 16, 16, 16))
    assert conv_mod._use_triton(conv, x, None, None) is True

    _fake_device(monkeypatch, _UNTUNED["mi300x"])
    assert conv_mod._use_triton(conv, x, None, None) is False
    torch.testing.assert_close(conv(x), nn.Conv3d.forward(conv, x))

    conv_mod.set_conv_triton_enabled(True)
    assert conv_mod._use_triton(conv, x, None, None) is True


@pytest.mark.gpu
def test_an_untuned_device_sends_the_upsampler_to_miopen(monkeypatch):
    """The transposed ladder shares ``_routing_declines``, so it shares this."""
    torch.manual_seed(3)
    module = FastConvTranspose3d(16, 8, kernel_size=2, stride=2)
    module = module.cuda().to(memory_format=_CHANNELS_LAST).to(torch.bfloat16)
    x = _gpu_input((1, 16, 8, 8, 8))
    assert conv_mod._use_triton_transposed(module, x, None, None) is True

    _fake_device(monkeypatch, _UNTUNED["mi300x"])
    assert conv_mod._use_triton_transposed(module, x, None, None) is False
    torch.testing.assert_close(module(x), nn.ConvTranspose3d.forward(module, x))

    conv_mod.set_conv_triton_enabled(True)
    assert conv_mod._use_triton_transposed(module, x, None, None) is True


@pytest.mark.gpu
def test_an_untuned_device_sends_group_norm_to_the_stock_kernel(monkeypatch):
    """The second ladder, guarded by the same one predicate."""
    module = FastGroupNorm(8, 16).cuda()
    x = _gpu_input((1, 16, 8, 8, 8), dtype=torch.float32)
    args = (x, module.num_groups, module.weight, module.bias, None)
    assert gn_mod._use_triton(*args) is True

    _fake_device(monkeypatch, _UNTUNED["mi300x"])
    assert gn_mod._use_triton(*args) is False
    torch.testing.assert_close(
        module(x), nn.GroupNorm.forward(module, x), rtol=1e-5, atol=1e-5
    )

    gn_mod.set_triton_enabled(True)
    assert gn_mod._use_triton(*args) is True


@pytest.mark.gpu
def test_both_ladders_share_one_verdict_and_ask_for_it_once(monkeypatch):
    """One source of truth, one query, whichever ladder gets there first.

    Two copies of this decision would drift -- the tables they protect were
    tuned in two separate sessions on the same device -- so the cheapest
    available proof that there is only one is that the second ladder's routing
    call does not produce a second driver query.
    """
    asked = []
    _fake_device(monkeypatch, _MI300A, count=asked)
    conv = _gpu_conv()
    x = _gpu_input((1, 16, 16, 16, 16))
    gn = FastGroupNorm(8, 16).cuda()
    for _ in range(3):
        conv_mod._use_triton(conv, x, None, None)
        gn_mod._use_triton(
            x.float().contiguous(memory_format=_CHANNELS_LAST),
            gn.num_groups,
            gn.weight,
            gn.bias,
            None,
        )
    assert asked == [0]


@pytest.mark.gpu
def test_a_whole_unet_step_asks_the_hardware_once(monkeypatch):
    """The property that matters in production: once, not once per operation.

    A scale-7 step routes 19 convolutions, 4 upsamplers and 18 GroupNorms
    through these predicates; the guard has to be a dictionary lookup after the
    first of them.
    """
    from ScaFFold.unet.unet_model import UNet

    asked = []
    _fake_device(monkeypatch, _MI300A, count=asked)
    torch.manual_seed(0)
    model = UNet(
        n_channels=3, n_classes=2, trilinear=False, layers=2, group_norm_groups=8
    )
    model = model.cuda().to(memory_format=_CHANNELS_LAST)
    x = _gpu_input((1, 3, 16, 16, 16), dtype=torch.float32)
    with torch.no_grad():
        model(x)
    assert asked == [0]


# ---------------------------------------------------------------------------
# The startup kernel-selection line
# ---------------------------------------------------------------------------


class _Ladder(torch.nn.Module):
    """Stands in for a rung-bearing module: the reporter is duck-typed."""

    _triton_ok = False
    _rung_label = "Ladder"


class _OtherLadder(torch.nn.Module):
    _triton_ok = False
    _rung_label = "Other"


class _Unlabelled(torch.nn.Module):
    _triton_ok = False


def test_kernel_selection_counts_each_ladder_separately():
    model = torch.nn.Sequential(_Ladder(), _Ladder(), _OtherLadder(), torch.nn.ReLU())
    model[0]._triton_ok = True
    assert kernel_selection(model) == [("Ladder", 1, 2), ("Other", 0, 1)]


def test_kernel_selection_ignores_modules_without_a_rung():
    """A plain module must not appear -- the line is about ladders only."""
    model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Identity())
    assert kernel_selection(model) == []


def test_a_ladder_without_a_label_still_reports_under_its_class_name():
    """Adding a ladder must not require remembering to declare a label."""
    assert kernel_selection(torch.nn.Sequential(_Unlabelled())) == [
        ("_Unlabelled", 0, 1)
    ]


def test_a_split_ladder_names_both_kernels():
    """The mixed case is the informative one and must not read as uniform."""
    line = format_kernel_selection([("Convolution", 17, 19)])[0]
    assert "Triton 17/19" in line and "Native 2/19" in line


@pytest.mark.parametrize(
    "selection,expected",
    [([("C", 3, 3)], "Triton"), ([("C", 0, 3)], "Native")],
)
def test_an_unsplit_ladder_names_one_kernel(selection, expected):
    line = format_kernel_selection(selection)[0]
    assert expected in line
    assert ("Native" if expected == "Triton" else "Triton") not in line


@pytest.mark.gpu
def test_the_real_model_reports_triton_on_every_site_after_a_forward():
    """The shipped configuration is all-Triton, and the line must say so.

    Also pins the placement rule: the same model reports ``Native`` everywhere
    *before* a forward, because ``_triton_ok`` is a latch. That is why
    ``_log_kernel_selection`` is called after warmup and after the first batch
    rather than at construction.
    """
    from ScaFFold.unet.unet_model import UNet

    model = UNet(n_channels=3, n_classes=6, trilinear=False, layers=3)
    model = model.cuda().to(memory_format=_CHANNELS_LAST)
    assert all(triton == 0 for _, triton, _ in kernel_selection(model))

    x = _gpu_input((1, 3, 32, 32, 32), dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        model(x)

    selection = kernel_selection(model)
    labels = {label for label, _, _ in selection}
    assert labels == {"Convolution", "Convolution (transposed)", "GroupNorm"}
    assert all(triton == total for _, triton, total in selection), selection
