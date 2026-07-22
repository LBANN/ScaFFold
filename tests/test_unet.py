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
