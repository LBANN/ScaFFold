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

"""Parts of the U-Net model"""

import torch.nn as nn
import torch.nn.functional as F

from ScaFFold.utils.perf_measure import annotate

from .group_norm import FastGroupNorm
from .triton_cat import skip_concat

_doubleconv_annotate = annotate(fmt="DoubleConv.{}")
_down_annotate = annotate(fmt="Down.{}")
_up_annotate = annotate(fmt="Up.{}")
_outconv_annotate = annotate(fmt="OutConv.{}")


def _group_norm(num_groups, num_channels, activation=None):
    if num_channels % num_groups != 0:
        raise ValueError(
            f"group_norm_groups={num_groups} must evenly divide num_channels={num_channels}"
        )
    # FastGroupNorm is nn.GroupNorm plus a Triton/compiled GPU kernel; it holds
    # the same parameters under the same names, and `activation` is a plain
    # attribute rather than a submodule, so checkpoints are unaffected.
    return FastGroupNorm(num_groups, num_channels, activation=activation)


class DoubleConv(nn.Module):
    """(convolution => GroupNorm => ReLU) * 2

    The ReLU lives *inside* the GroupNorm (``activation="relu"``), because the
    Triton GroupNorm kernel folds it into its forward store for free and thereby
    removes an entire streaming pass -- 38% of the forward at the shapes that
    dominate the step.  ``FastGroupNorm`` applies the ReLU on every path,
    including eager, so the network's function is unchanged; only the number of
    memory passes differs.

    The ``nn.ReLU`` slots are held open by ``nn.Identity`` rather than removed:
    ``nn.Sequential`` names its children by position, so deleting them would
    renumber the two convolutions and the second GroupNorm and invalidate every
    existing checkpoint.  Neither ``nn.ReLU`` nor ``nn.Identity`` has parameters
    or buffers, so with the placeholders in place the state dict is byte
    identical to the pre-fusion model's (pinned by
    ``tests/test_groupnorm.py::test_state_dict_matches_plain_groupnorm_model``).
    """

    def __init__(self, in_channels, out_channels, group_norm_groups, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(group_norm_groups, mid_channels, activation="relu"),
            nn.Identity(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(group_norm_groups, out_channels, activation="relu"),
            nn.Identity(),
        )

    @_doubleconv_annotate
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, group_norm_groups):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, group_norm_groups),
        )

    @_down_annotate
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv

    The skip concatenation goes through :func:`ScaFFold.unet.triton_cat.skip_concat`
    rather than ``torch.cat``.  That does two things, both of which leave the
    tensor the following convolution reads *bitwise* unchanged (verified in
    ``tests/test_triton_cat.py``):

    * It emits the dtype the convolution will use instead of ``torch.cat``'s
      promoted one.  Under ``torch.autocast`` the two halves do not have the
      same dtype -- the skip comes from a GroupNorm, an fp32-policy op, and the
      upsampled half comes from a ``ConvTranspose3d`` and is bf16 -- so ``cat``
      widens the bf16 half to fp32, concatenates at fp32, and the convolution
      then narrows the whole double-width result straight back down.  Three
      full-resolution passes to deliver one.
    * It runs a channels-last-native kernel, which matters most for the
      *backward*: ``cat``'s backward is a narrowed view that every consumer
      then forces contiguous, and that strided copy reaches only 51-63% of this
      device's streaming roofline against the kernel's 90-103%.

    Both rest on ``self.conv`` beginning with a convolution, which the
    constructor below guarantees on either branch.  ``skip_concat`` is total:
    anything its ``is_supported`` declines -- CPU, non-channels-last, no Triton
    -- falls back to ``torch.cat``.
    """

    def __init__(self, in_channels, out_channels, group_norm_groups, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                group_norm_groups,
                in_channels // 2,
            )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, group_norm_groups)

    @_up_annotate
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        if diffX or diffY or diffZ:
            x1 = F.pad(
                x1,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2,
                    diffZ // 2,
                    diffZ - diffZ // 2,
                ],
            )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # torch.cat([x2, x1], dim=1) with the dtype and the layout the
        # convolution below actually wants; see the class docstring.
        x = skip_concat(x2, x1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    @_outconv_annotate
    def forward(self, x):
        return self.conv(x)
