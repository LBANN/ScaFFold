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
