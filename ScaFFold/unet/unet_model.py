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

"""Full assembly of the parts to form the complete network"""

import torch
import torch.nn as nn

from ScaFFold.utils.perf_measure import annotate

from .unet_parts import DoubleConv, Down, OutConv, Up

_unet_annotate = annotate(fmt="UNet.{}")


class UNet(nn.Module):
    """
    Unrolled 4 layers model

    self.inc = (DoubleConv(n_channels, layer_channels))
    self.down1 = (Down(64, 128))
    self.down2 = (Down(128, 256))
    self.down3 = (Down(256, 512))
    self.down4 = (Down(512, 1024 // factor))
    self.up1 = (Up(1024, 512 // factor, trilinear))
    self.up2 = (Up(512, 256 // factor, trilinear))
    self.up3 = (Up(256, 128 // factor, trilinear))
    self.up4 = (Up(128, 64, trilinear))
    self.outc = (OutConv(64, n_classes))
    """

    @_unet_annotate
    def __init__(
        self,
        n_channels,
        n_classes,
        trilinear=False,
        layers=4,
        group_norm_groups=8,
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear
        self.layers = layers
        self.group_norm_groups = group_norm_groups
        factor = 2 if trilinear else 1

        self.down_list = nn.ModuleList([])
        layer_channels = 64
        self.down_list.append(
            DoubleConv(n_channels, layer_channels, self.group_norm_groups)
        )

        for i in range(self.layers - 1):
            self.down_list.append(
                Down(layer_channels, layer_channels * 2, self.group_norm_groups)
            )
            layer_channels *= 2

        self.down_list.append(
            Down(
                layer_channels,
                (layer_channels * 2) // factor,
                self.group_norm_groups,
            )
        )
        layer_channels *= 2

        self.up_list = nn.ModuleList([])
        for i in range(self.layers - 1):
            self.up_list.append(
                Up(
                    layer_channels,
                    (layer_channels // 2) // factor,
                    self.group_norm_groups,
                    trilinear,
                )
            )
            layer_channels //= 2

        self.up_list.append(
            Up(
                layer_channels,
                layer_channels // 2,
                self.group_norm_groups,
                trilinear,
            )
        )
        layer_channels //= 2

        self.up_list.append(OutConv(layer_channels, n_classes))

    """
    Unrolled 4 layer model

    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    logits = self.outc(x)
    return logits
    """

    @_unet_annotate
    def forward(self, x):
        results = []
        results.append(self.down_list[0](x))
        for i in range(1, self.layers + 1):
            results.append(self.down_list[i](results[i - 1]))

        results.reverse()

        # Takes last layer output to use for concatenation
        output = results[0]
        # Iterates up the list of up layers, passing it the previous ouput and concatenating with result at corresponding backward index
        for i in range(len(self.up_list) - 1):
            output = self.up_list[i](output, results[i + 1])

        return self.up_list[-1](output)

    """
    Unrolled 4 layer model

    self.inc = torch.utils.checkpoint(self.inc)
    self.down1 = torch.utils.checkpoint(self.down1)
    self.down2 = torch.utils.checkpoint(self.down2)
    self.down3 = torch.utils.checkpoint(self.down3)
    self.down4 = torch.utils.checkpoint(self.down4)
    self.up1 = torch.utils.checkpoint(self.up1)
    self.up2 = torch.utils.checkpoint(self.up2)
    self.up3 = torch.utils.checkpoint(self.up3)
    self.up4 = torch.utils.checkpoint(self.up4)
    self.outc = torch.utils.checkpoint(self.outc)
    """

    def use_checkpointing(self):
        for i in range(len(self.down_list)):
            self.down_list[i] = torch.utils.checkpoint(self.down_list[i])

        for i in range(len(self.up_list)):
            self.up_list[i] = torch.utils.checkpoint(self.up_list[i])
