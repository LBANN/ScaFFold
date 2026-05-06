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

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ScaFFold.utils.dice_score import SpatialAllReduce


def _sample_ce_weight_indices(n_train, sample_fraction):
    """Pick a small, deterministic subset of masks to estimate CE weights."""
    if n_train <= 0:
        return []

    if sample_fraction is None:
        sample_fraction = 0.1

    sample_count = min(
        max(math.ceil(n_train * float(sample_fraction)), 1),
        n_train,
    )
    if sample_count == n_train:
        return list(range(n_train))

    return torch.linspace(0, n_train - 1, steps=sample_count).long().tolist()


def _compute_ce_class_weights(
    train_set,
    n_train,
    n_categories,
    device,
    sample_fraction=0.1,
    dist_enabled=False,
    world_rank=0,
    log=None,
):
    """
    Estimate background vs foreground CE weights from a few training masks.

    Background keeps its own inverse-frequency weight, and every non-zero
    fractal class shares the foreground weight derived from the aggregate
    non-empty voxel count.
    """

    num_classes = n_categories + 1
    class_weights = torch.ones(num_classes, device=device, dtype=torch.float32)

    if n_train == 0:
        if log is not None:
            log.warning(
                "Training set is empty while computing CE class weights. Falling back to uniform weights."
            )
        return class_weights

    sample_indices = _sample_ce_weight_indices(n_train, sample_fraction)
    sampled_class_counts = torch.zeros(num_classes, dtype=torch.long)

    for sample_idx in sample_indices:
        mask = train_set[sample_idx]["mask"]
        sampled_class_counts += torch.bincount(mask.reshape(-1), minlength=num_classes)

    # The dataset may already return only this rank's local spatial shard,
    # so combine per-rank counts before deriving the global CE weights.
    sampled_class_counts = sampled_class_counts.to(device=device)
    if dist_enabled:
        dist.all_reduce(sampled_class_counts, op=dist.ReduceOp.SUM)

    background_voxels = int(sampled_class_counts[0].item())
    foreground_voxels = int(sampled_class_counts[1:].sum().item())

    if background_voxels > 0 and foreground_voxels > 0:
        total_voxels = background_voxels + foreground_voxels
        class_weights[0] = total_voxels / background_voxels
        class_weights[1:] = total_voxels / foreground_voxels
    elif log is not None:
        log.warning(
            "Sampled masks did not contain both background and foreground voxels. Falling back to uniform CE weights."
        )

    if log is not None and (not dist_enabled or world_rank == 0):
        log.info(
            f"CE weights estimated from {len(sample_indices)} training masks "
            f"(sample_fraction={sample_fraction}, indices={sample_indices}): "
            f"background_voxels={background_voxels} "
            f"foreground_voxels={foreground_voxels} "
            f"weights={class_weights.detach().cpu().tolist()}"
        )

    return class_weights


def compute_sharded_cross_entropy_loss(
    local_preds,
    local_labels,
    spatial_mesh,
    _num_shards,
    device_type,
    class_weights=None,
):
    """
    Compute the CE loss for a spatially sharded volume.

    Each rank only sees a local spatial shard, so we cannot use the local
    `reduction="mean"` result directly. Instead we:
    1. compute the local CE numerator with `reduction="sum"`,
    2. build the correct global denominator,
    3. all-reduce across the spatial mesh, and
    4. divide to recover the same value we would get from a non-sharded tensor.

    When `class_weights` is provided, PyTorch's CE "mean" divides by the sum of
    the target weights, not the raw voxel count, so we reproduce that behavior
    explicitly here.
    """

    autocast_device = device_type if device_type != "mps" else "cpu"
    with torch.autocast(autocast_device, enabled=False):
        # Accumulate CE in full precision. Using reduction="sum" gives us the
        # numerator of the final global mean; if class weights are present,
        # PyTorch applies the target-class weight to each voxel here.
        local_ce_sum = F.cross_entropy(
            local_preds.float(),
            local_labels,
            weight=class_weights,
            reduction="sum",
        )

        if class_weights is None:
            # Sum the actual local voxel counts across spatial shards. We use
            # an all-reduced count instead of numel()*num_shards because shard
            # sizes can differ at chunk boundaries.
            local_voxel_count = local_ce_sum.new_tensor(float(local_labels.numel()))
            global_normalizer = SpatialAllReduce.apply(local_voxel_count, spatial_mesh)
        else:
            # Weighted CE divides by sum(weight[target_i]) over all voxels.
            # Build that denominator from the local label histogram, then
            # all-reduce it across the spatial mesh.
            local_class_counts = torch.bincount(
                local_labels.reshape(-1), minlength=class_weights.numel()
            ).to(dtype=local_ce_sum.dtype)
            local_weight_sum = torch.dot(
                local_class_counts, class_weights.to(dtype=local_ce_sum.dtype)
            )
            global_normalizer = SpatialAllReduce.apply(local_weight_sum, spatial_mesh)

    # Sum the local CE numerators from each spatial shard to get the global CE
    # numerator, then divide by the matching global denominator.
    global_ce_sum = SpatialAllReduce.apply(local_ce_sum, spatial_mesh)
    # Clamp to avoid a divide-by-zero in degenerate cases.
    return global_ce_sum / global_normalizer.clamp_min(
        torch.finfo(global_ce_sum.dtype).eps
    )
