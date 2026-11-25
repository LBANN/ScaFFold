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

"""Native local convolutions with halo communication supplied by DistConv."""

import torch
import torch.nn.functional as F
from distconv import distconv as distconv_ops

from .conv3d import _autocast_dtype, _cast_operand


class _DistConvHalo(torch.autograd.Function):
    """Pair DistConv's plain-tensor forward and backward halo helpers.

    DistConv normally invokes these helpers through DCTensor convolution
    dispatch. A local-tensor model needs this explicit autograd boundary;
    DistConv's exchange custom operator supplies the opaque communication.
    """

    @staticmethod
    def forward(ctx, tensor, plan):
        ctx.plan = plan
        for dim_index, _dim, halo in plan.exchanges:
            tensor = distconv_ops.forward_halo_exchange(
                tensor, halo, plan.strategy, dim_index
            )
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # DistConv accumulates into the supplied gradient and returns a view.
        # Own that storage: grad_output can be shared or have expanded strides.
        grad = grad_output.clone(memory_format=torch.preserve_format)
        for dim_index, _dim, halo in ctx.plan.exchanges:
            grad = distconv_ops.backward_halo_exchange(
                grad, halo, ctx.plan.strategy, dim_index
            )
        return grad, None


def bind_native_convolution(module, plan):
    """Install a native forward for a convolution with a validated halo plan.

    The plan determines which axes exchange and their adjusted convolution
    padding. Parameters stay on the original module, preserving optimizer and
    checkpoint identity. No ScaFFold Triton kernel or halo transport is used.
    """
    if module.padding_mode != "zeros":
        raise ValueError("Native compiled convolution requires padding_mode='zeros'")

    def forward(local):
        # Match convolution autocast before communicating, so backward halo
        # accumulation also uses the convolution's dtype (not FP32 under AMP).
        local = _cast_operand(local, _autocast_dtype(local))
        if plan.exchanges:
            local = _DistConvHalo.apply(local, plan)
        return F.conv3d(
            local,
            module.weight,
            module.bias,
            module.stride,
            plan.padding,
            module.dilation,
            module.groups,
        )

    module.forward = forward
