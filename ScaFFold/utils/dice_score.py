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

import torch
import torch.distributed as dist
from torch import Tensor

from ScaFFold.utils.perf_measure import annotate


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    assert input.size() == target.size(), (
        f"Got predictions (input) of {input.size()} and target of {target.size()}"
    )
    assert input.dim() == 4 or not reduce_batch_first

    sum_dim = (
        (-1, -2, -3) if input.dim() == 3 or not reduce_batch_first else (-1, -2, -3, -4)
    )

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum_raw = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum_raw == 0, inter, sets_sum_raw)

    dice = (inter + epsilon) / (sets_sum + epsilon)

    return dice.mean()


def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all classes
    return dice_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


@annotate()
def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class SpatialAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, spatial_mesh):
        output = input.clone()
        for mesh_dim in range(spatial_mesh.ndim):
            pg = spatial_mesh.get_group(mesh_dim)
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=pg)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


@annotate()
def compute_sharded_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    spatial_mesh,
    epsilon: float = 1e-6,
):
    """
    Computes the globally sharded Dice score.
    Returns the raw score tensor of shape [Batch, Channels].
    """
    assert preds.size() == targets.size(), (
        f"Shape mismatch: {preds.size()} vs {targets.size()}"
    )
    assert preds.dim() == 5, f"Expected 5D tensor, got {preds.dim()}D"

    sum_dim = (-1, -2, -3)  # D, H, W

    local_inter = 2.0 * (preds * targets).sum(dim=sum_dim)
    local_sets_sum_raw = preds.sum(dim=sum_dim) + targets.sum(dim=sum_dim)

    packed = torch.stack([local_inter, local_sets_sum_raw])

    # Global reduce across spatial mesh
    packed_global = SpatialAllReduce.apply(packed, spatial_mesh)

    global_inter = packed_global[0]
    global_sets_sum_raw = packed_global[1]

    global_sets_sum = torch.where(
        global_sets_sum_raw == 0, global_inter, global_sets_sum_raw
    )

    # Calculate score
    dice_score = (global_inter + epsilon) / (global_sets_sum + epsilon)

    return dice_score
