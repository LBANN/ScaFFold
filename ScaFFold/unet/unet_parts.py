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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ScaFFold.utils.perf_measure import annotate

_doubleconv_annotate = annotate(fmt="DoubleConv.{}")
_down_annotate = annotate(fmt="Down.{}")
_up_annotate = annotate(fmt="Up.{}")
_outconv_annotate = annotate(fmt="OutConv.{}")


def _group_norm(num_groups, num_channels):
    if num_channels % num_groups != 0:
        raise ValueError(
            f"group_norm_groups={num_groups} must evenly divide num_channels={num_channels}"
        )
    return nn.GroupNorm(num_groups, num_channels)


class DoubleConv(nn.Module):
    """(convolution => GroupNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, group_norm_groups, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(group_norm_groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(group_norm_groups, out_channels),
            nn.ReLU(inplace=True),
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
    """Upscaling then double conv"""

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
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    @_outconv_annotate
    def forward(self, x):
        return self.conv(x)
