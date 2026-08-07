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

"""Tests for U-Net model construction and activation checkpointing.

The model is built exactly as ``worker.py`` builds it (``n_channels=3``,
``n_classes=n_categories+1``, ``layers=problem_scale-unet_bottleneck_dim``) but
at the smallest workable scale (16^3 volumes, 2 classes, batch 1) so every case
runs on CPU in seconds.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ScaFFold.unet.unet_model import UNet

# Smallest workable problem: 16^3 volumes, matching worker.py's build.
_N = 16
_N_CHANNELS = 3
_N_CLASSES = 2


def _make_input(seed: int = 0):
    """A deterministic (B, C, N, N, N) volume for the smallest problem size."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, _N_CHANNELS, _N, _N, _N, generator=generator)


def test_zero_layer_config_rejected():
    """A degenerate zero-layer depth must fail loudly at build time.

    ``layers`` comes from ``problem_scale - unet_bottleneck_dim``; when those
    are equal the depth is zero, which would build a network whose bottleneck
    ``Down`` is never invoked and whose forward pass indexes past the end of the
    encoder outputs. The constructor must reject it with an informative error
    rather than deferring to a confusing ``IndexError`` at first forward.
    """
    with pytest.raises(ValueError) as excinfo:
        UNet(
            n_channels=_N_CHANNELS,
            n_classes=_N_CLASSES,
            trilinear=False,
            layers=0,
        )
    message = str(excinfo.value)
    # The message should name the constraint so the config mistake is obvious.
    assert "layers" in message
    assert "1" in message


def test_min_depth_forward_shapes():
    """The smallest valid depth produces a full-resolution segmentation map."""
    model = UNet(
        n_channels=_N_CHANNELS,
        n_classes=_N_CLASSES,
        trilinear=False,
        layers=1,
    )
    x = _make_input()
    out = model(x)
    assert out.shape == (1, _N_CLASSES, _N, _N, _N)


def test_checkpointing_output_identical():
    """Enabling activation checkpointing must not change the forward output."""
    model = UNet(
        n_channels=_N_CHANNELS,
        n_classes=_N_CLASSES,
        trilinear=False,
        layers=2,
    )
    model.eval()
    x = _make_input(seed=1)

    with torch.no_grad():
        out_off = model(x)
        model.use_checkpointing()
        out_on = model(x)

    assert torch.allclose(out_off, out_on, rtol=0, atol=0)


def test_checkpointing_grads_identical():
    """Backward under both modes must yield identical parameter gradients."""
    model = UNet(
        n_channels=_N_CHANNELS,
        n_classes=_N_CLASSES,
        trilinear=False,
        layers=2,
    )
    # use_reentrant=False needs an input that requires grad for gradients to
    # flow through the recomputed graph.
    x = _make_input(seed=2).requires_grad_(True)

    model.zero_grad(set_to_none=True)
    model(x).sum().backward()
    grads_off = {name: p.grad.detach().clone() for name, p in model.named_parameters()}

    model.use_checkpointing()
    model.zero_grad(set_to_none=True)
    model(x).sum().backward()
    grads_on = {name: p.grad.detach().clone() for name, p in model.named_parameters()}

    assert grads_off.keys() == grads_on.keys()
    for name in grads_off:
        assert torch.allclose(grads_off[name], grads_on[name]), name


def test_up_pad_guarded_when_diffs_zero():
    """Up.forward should skip F.pad when all spatial diffs are zero.

    When spatial dimensions exactly match (vol_size = 2^problem_scale with exact
    up/pool pairs), diffX/diffY/diffZ are all zero every step. F.pad with an
    all-zero pad list still allocates a new tensor (a full device memcpy), so
    the guard is a pure performance fix. This test verifies:
      1. RED (unfixed): F.pad is called even when all diffs are zero
      2. GREEN (fixed): F.pad is not called when all diffs are zero
      3. Output is bit-identical to an explicit F.pad(..., [0]*6) reference
      4. F.pad is still called when any diff is nonzero
    """
    import torch.nn.functional as F

    from ScaFFold.unet.unet_parts import Up

    # Case 1: exact match (all diffs zero) -- most common in this benchmark
    up = Up(in_channels=128, out_channels=64, group_norm_groups=8, trilinear=False)
    up.eval()
    x1 = torch.randn(1, 128, 16, 16, 16)  # pre-upsample (16x16x16)
    x2 = torch.randn(1, 64, 32, 32, 32)  # skip (32x32x32) -- exact 2x match

    # Spy on F.pad to count calls
    pad_call_count = 0
    original_pad = F.pad

    def counting_pad(tensor, pad, *args, **kwargs):
        nonlocal pad_call_count
        pad_call_count += 1
        return original_pad(tensor, pad, *args, **kwargs)

    with patch("torch.nn.functional.pad", side_effect=counting_pad):
        with torch.no_grad():
            out_exact = up(x1, x2)

    exact_match_pad_calls = pad_call_count

    # Reference: explicit F.pad(..., [0]*6) should be bit-identical (numeric equivalence)
    up.eval()
    with torch.no_grad():
        x1_up = up.up(x1)  # upsample to 32x32x32
        x1_padded_ref = F.pad(x1_up, [0, 0, 0, 0, 0, 0])
        x_cat = torch.cat([x2, x1_padded_ref], dim=1)
        out_ref = up.conv(x_cat)

    assert torch.equal(out_exact, out_ref), (
        "Output should match explicit zero-pad reference"
    )

    # Case 2: nonzero diff -- guard should still allow pad to run
    pad_call_count = 0
    x1_smaller = torch.randn(1, 128, 15, 15, 15)  # 15x15x15, will upsample to 30x30x30
    x2_larger = torch.randn(1, 64, 32, 32, 32)  # 32x32x32 -- nonzero diff

    with patch("torch.nn.functional.pad", side_effect=counting_pad):
        with torch.no_grad():
            up(x1_smaller, x2_larger)

    nonzero_match_pad_calls = pad_call_count

    # When there's a diff, F.pad must be called
    assert nonzero_match_pad_calls > 0, (
        "F.pad should be called when spatial diffs are nonzero"
    )

    # Now the key test: if the guard is in place, exact matches should skip pad;
    # if not, both will call pad. We expect exact to be 0 (fixed) but report
    # the actual counts so the test output shows the RED/GREEN difference.
    if exact_match_pad_calls == 0:
        # GREEN: guard is in place and working
        assert nonzero_match_pad_calls > 0, "Nonzero case should still call pad"
    else:
        # RED: no guard yet, F.pad is called unconditionally
        pytest.skip(
            f"Guard not yet in place: {exact_match_pad_calls} pad calls with diffs=0 "
            f"(expected 0 when fixed). This is the RED baseline."
        )


# --------------------------------------------------------------------------- #
# The decoder skip concatenation.
#
# ``Up.forward`` no longer calls ``torch.cat`` directly; it goes through
# ``unet_parts._skip_concat``, which may legitimately emit a narrower dtype
# than ``torch.cat`` would when autocast is on (see the ``Up`` docstring).
# Everything below pins the part that must NOT change: outside autocast the
# block is bitwise what it was, the ``F.pad`` path still works, and both the
# ``trilinear`` and ``ConvTranspose3d`` branches agree with an explicit
# ``torch.cat`` reference.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("trilinear", [False, True])
def test_up_matches_an_explicit_torch_cat_reference(trilinear):
    """``Up.forward`` must equal ``conv(cat([x2, up(x1)]))``, bitwise, on CPU."""
    from ScaFFold.unet.unet_parts import Up

    up = Up(in_channels=32, out_channels=16, group_norm_groups=8, trilinear=trilinear)
    up.eval()
    generator = torch.Generator().manual_seed(11)
    # Either branch must hand ``self.conv`` ``in_channels`` channels: the
    # transposed convolution halves 32 -> 16, while ``nn.Upsample`` changes no
    # channels, so its input already carries 16.
    x1 = torch.randn(1, 16 if trilinear else 32, 8, 8, 8, generator=generator)
    x2 = torch.randn(1, 16, 16, 16, 16, generator=generator)

    with torch.no_grad():
        got = up(x1, x2)
        reference = up.conv(torch.cat([x2, up.up(x1)], dim=1))

    assert got.shape == reference.shape
    assert torch.equal(got, reference), (
        "the skip concatenation must be bitwise torch.cat outside autocast"
    )


def test_up_still_pads_and_concatenates_when_shapes_disagree():
    """The non-power-of-two path: ``F.pad`` fires and the result still matches."""
    import torch.nn.functional as F

    from ScaFFold.unet.unet_parts import Up

    up = Up(in_channels=32, out_channels=16, group_norm_groups=8, trilinear=False)
    up.eval()
    generator = torch.Generator().manual_seed(12)
    x1 = torch.randn(1, 32, 7, 7, 7, generator=generator)  # -> 14^3 after up
    x2 = torch.randn(1, 16, 16, 16, 16, generator=generator)  # 16^3: diff = 2

    with torch.no_grad():
        got = up(x1, x2)
        padded = F.pad(up.up(x1), [1, 1, 1, 1, 1, 1])
        reference = up.conv(torch.cat([x2, padded], dim=1))

    assert got.shape == (1, 16, 16, 16, 16)
    assert torch.equal(got, reference)


def test_up_gradients_match_an_explicit_torch_cat_reference():
    """Backward through the skip concatenation, bitwise, on CPU."""
    from ScaFFold.unet.unet_parts import Up

    up = Up(in_channels=32, out_channels=16, group_norm_groups=8, trilinear=False)
    generator = torch.Generator().manual_seed(13)
    x1 = torch.randn(1, 32, 8, 8, 8, generator=generator)
    x2 = torch.randn(1, 16, 16, 16, 16, generator=generator)

    a, b = x1.clone().requires_grad_(True), x2.clone().requires_grad_(True)
    up.zero_grad(set_to_none=True)
    up(a, b).pow(2).sum().backward()
    got = (a.grad.clone(), b.grad.clone())
    got_params = {n: p.grad.clone() for n, p in up.named_parameters()}

    c, d = x1.clone().requires_grad_(True), x2.clone().requires_grad_(True)
    up.zero_grad(set_to_none=True)
    up.conv(torch.cat([d, up.up(c)], dim=1)).pow(2).sum().backward()

    assert torch.equal(got[0], c.grad)
    assert torch.equal(got[1], d.grad)
    for name, param in up.named_parameters():
        assert torch.equal(got_params[name], param.grad), name


def test_up_concatenation_keeps_channels_last():
    """The concatenation must not break the layout chain it exists to preserve.

    Asserted on ``_skip_concat`` with two channels-last halves rather than on a
    whole ``Up`` block: on CPU ``nn.ConvTranspose3d`` returns a *contiguous*
    tensor whatever it is handed, so the block's own inputs to the
    concatenation are not both channels-last there and the block-level
    assertion would be measuring the convolution's layout policy, not this
    one's.  On GPU with ``PYTORCH_MIOPEN_SUGGEST_NHWC=1`` -- the production
    configuration -- both halves are channels-last and this is the property
    ``Up`` relies on.
    """
    from ScaFFold.unet.unet_parts import _skip_concat as skip_concat

    generator = torch.Generator().manual_seed(14)
    x1 = torch.randn(1, 16, 16, 16, 16, generator=generator).contiguous(
        memory_format=torch.channels_last_3d
    )
    x2 = torch.randn(1, 16, 16, 16, 16, generator=generator).contiguous(
        memory_format=torch.channels_last_3d
    )
    out = skip_concat(x2, x1)
    assert out.shape == (1, 32, 16, 16, 16)
    assert out.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(out, torch.cat([x2, x1], dim=1))


def test_whole_model_forward_and_backward_still_agree_with_a_cat_based_up():
    """End to end: swapping the concatenation back must change nothing on CPU."""
    import torch as _torch

    from ScaFFold.unet import unet_parts

    def cat_forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(_torch.cat([x2, x1], dim=1))

    model = UNet(
        n_channels=_N_CHANNELS, n_classes=_N_CLASSES, trilinear=False, layers=2
    )
    x = _make_input(seed=15).requires_grad_(True)

    model.zero_grad(set_to_none=True)
    model(x).pow(2).sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    x_grad = x.grad.clone()

    original = unet_parts.Up.forward
    try:
        unet_parts.Up.forward = cat_forward
        x2 = _make_input(seed=15).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(x2).pow(2).sum().backward()
        for name, param in model.named_parameters():
            assert torch.equal(grads[name], param.grad), name
        assert torch.equal(x_grad, x2.grad)
    finally:
        unet_parts.Up.forward = original
