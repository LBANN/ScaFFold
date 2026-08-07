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

from .conv3d import FastConv3d, FastConvTranspose3d
from .group_norm import FastGroupNorm

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


def _conv3d(in_channels, out_channels, **kwargs):
    """The model's non-transposed convolutions, in one place.

    ``FastConv3d`` is ``nn.Conv3d`` plus a Triton GPU kernel; it holds the same
    parameters under the same names and adds no buffers, so checkpoints are
    unaffected in either direction.  It falls back to MIOpen for anything the
    kernel does not serve, and it serves the sharded configurations too: it
    performs the halo exchange itself, above autograd, rather than leaving it to
    the one DistConv does below.  Note what that means for the *shape* the
    kernel sees -- only the split axis is halo'd, so ``padding=1`` survives on
    the other two and these convolutions are padded at every configuration; see
    :mod:`ScaFFold.unet.conv3d`.

    The four ``nn.ConvTranspose3d`` in ``Up`` do not come through here: they are
    a different operator, with the weight's channel axes the other way round and
    a different set of kernels behind them, so they have a factory of their own
    (:func:`_conv_transpose3d`) rather than a flag on this one.
    """
    return FastConv3d(in_channels, out_channels, **kwargs)


def _conv_transpose3d(in_channels, out_channels, **kwargs):
    """The model's transposed convolutions -- the decoder's upsamplers.

    ``FastConvTranspose3d`` is ``nn.ConvTranspose3d`` plus a Triton GPU kernel;
    it holds the same parameters (``weight`` *and* ``bias``, which these sites
    have and the ordinary convolutions mostly do not) under the same names and
    adds no buffers, so checkpoints are unaffected in either direction.  It falls
    back to MIOpen for anything the kernel does not serve, which is everything
    except the ``kernel == stride``, no-padding upsample built below.
    """
    return FastConvTranspose3d(in_channels, out_channels, **kwargs)


def _consumer_dtype(*tensors):
    """The dtype the convolution consuming a concatenation will actually see.

    Inside an enabled autocast region the answer is autocast's dtype, because
    ``aten::convolution`` carries the ``lower_precision_fp`` cast policy and
    casts whatever it is handed.  Producing that dtype from the concatenation
    is *bitwise identical* to producing ATen's promoted dtype and letting the
    convolution narrow it -- the promoted tensor holds exact widenings of both
    sources, so narrowing before or after the copy rounds the same values once
    -- while writing and reading back half the bytes.

    Outside autocast the answer is ``torch.cat``'s ordinary promotion, so eval,
    ``inference_mode`` and pure-fp32 runs are unchanged.
    """
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    device_type = tensors[0].device.type
    try:
        if not torch.is_autocast_enabled(device_type):
            return dtype
        autocast_dtype = torch.get_autocast_dtype(device_type)
    except (RuntimeError, TypeError):  # a device type autocast does not know
        return dtype
    # Only ever narrow: if autocast's dtype is the wider of the two, keep the
    # promotion ATen would have done.
    if torch.promote_types(autocast_dtype, dtype) is autocast_dtype:
        return dtype
    return autocast_dtype


def _skip_concat(skip, upsampled):
    """``torch.cat([skip, upsampled], dim=1)`` at the consumer's dtype.

    Under ``torch.autocast`` the two halves do not share a dtype: the skip
    comes from a GroupNorm, an fp32-policy op, while the upsampled half comes
    from a ``ConvTranspose3d`` and is bf16.  ``torch.cat`` carries the
    ``promote`` policy, so it widens the bf16 half to fp32, concatenates at
    fp32, and the following convolution narrows the whole double-width result
    straight back down -- three full-resolution passes to deliver one.  Casting
    the inputs first collapses that to one, and the convolution reads the same
    bits either way (see :func:`_consumer_dtype`).
    """
    dtype = _consumer_dtype(skip, upsampled)
    return torch.cat([skip.to(dtype), upsampled.to(dtype)], dim=1)


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
            _conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(group_norm_groups, mid_channels, activation="relu"),
            nn.Identity(),
            _conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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

    The skip concatenation goes through :func:`_skip_concat` rather than
    ``torch.cat`` directly, so that it emits the dtype the following
    convolution will use instead of ``torch.cat``'s promoted one.  The tensor
    that convolution reads is bitwise unchanged either way; it is written and
    read back at half the width.  This rests on ``self.conv`` beginning with a
    convolution, which the constructor below guarantees on either branch.

    Measured at scale 7: 1.09 ms of a 92.8 ms step, and 0.50 GiB of peak
    memory.  A channels-last-native Triton concatenation kernel was built and
    measured too -- ``cat``'s *backward* is a narrowed view that consumers force
    contiguous, at 51-63% of this device's streaming roofline against the
    kernel's 90-103% -- but it was worth a further 0.08 ms of the step, which
    did not justify a second hand-written kernel in a benchmark other people
    have to trust.
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
            self.up = _conv_transpose3d(
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
        x = _skip_concat(x2, x1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = _conv3d(in_channels, out_channels, kernel_size=1)

    @_outconv_annotate
    def forward(self, x):
        return self.conv(x)
