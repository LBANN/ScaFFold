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
import torch.nn.functional as F
from distconv import DCTensor
from tqdm import tqdm

from ScaFFold.utils.dice_score import (
    SpatialAllReduce,
    compute_sharded_dice,
)
from ScaFFold.utils.perf_measure import annotate


@annotate()
@torch.inference_mode()
def evaluate(
    net, dataloader, device, amp, primary, criterion, n_categories, parallel_strategy
):

    def foreground_dice_mean(dice_scores):
        if dice_scores.size(1) > 1:
            return dice_scores[:, 1:].mean().item()
        return dice_scores.mean().item()

    net.eval()
    num_val_batches = len(dataloader)
    total_dice_score = 0.0
    processed_batches = 0

    spatial_mesh = parallel_strategy.device_mesh[parallel_strategy.distconv_dim_names]

    if primary:
        print(
            f"[eval] ps.shard_dim={parallel_strategy.shard_dim} num_shards={parallel_strategy.num_shards}"
        )

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        val_loss_epoch = 0.0
        for batch in tqdm(
            dataloader,
            total=num_val_batches,
            desc="Validation round",
            unit="batch",
            leave=False,
            disable=not primary,
        ):
            image, mask_true = batch["image"], batch["mask"]

            image = image.to(
                device=device,
                dtype=torch.float32,
                memory_format=torch.channels_last_3d,
            )
            mask_true = mask_true.to(device=device, dtype=torch.long).contiguous()

            # Dummy channel dimension [B, 1, D, H, W]
            mask_true = mask_true.unsqueeze(1)

            # Inputs are already loaded as local shards by the dataset.
            dcx = DCTensor.from_shard(image, parallel_strategy)
            mask_true_dc = DCTensor.from_shard(mask_true, parallel_strategy)

            # Forward pass on sharded data
            dcy = net(dcx)

            # Extract underlying local tensors (STAY SHARDED)
            local_preds = dcy
            local_labels_5d = mask_true_dc
            local_labels = local_labels_5d.squeeze(1)

            # Skip empty batches
            if local_preds.size(0) == 0 or local_labels.size(0) == 0:
                continue

            # --- 1. Sharded CE Loss ---
            with torch.autocast(
                device.type if device.type != "mps" else "cpu", enabled=False
            ):
                local_ce_sum = F.cross_entropy(
                    local_preds.float(), local_labels, reduction="sum"
                )
            global_ce_sum = SpatialAllReduce.apply(local_ce_sum, spatial_mesh)

            # Divide by the actual global voxel count to handle uneven shards.
            local_voxel_count = torch.tensor(
                float(local_labels.numel()),
                device=local_labels.device,
                dtype=torch.float32,
            )
            global_total_voxels = SpatialAllReduce.apply(
                local_voxel_count, spatial_mesh
            )
            CE_loss = global_ce_sum / global_total_voxels

            # --- 2. Format Predictions & Labels (Strictly Multiclass) ---
            mask_pred_probs = F.softmax(local_preds, dim=1).float()
            mask_true_onehot = (
                F.one_hot(local_labels, n_categories + 1).permute(0, 4, 1, 2, 3).float()
            )

            # Dice loss uses probabilities
            dice_score_probs = compute_sharded_dice(
                mask_pred_probs, mask_true_onehot, spatial_mesh
            )
            # Eval metric (excluding background class 0)
            # dice_score_probs shape is [Batch, Channels].
            batch_dice_score = foreground_dice_mean(dice_score_probs)

            # --- Combine and Accumulate ---
            loss = CE_loss + (1.0 - batch_dice_score)
            val_loss_epoch += loss.item()
            total_dice_score += batch_dice_score.item()
            processed_batches += 1

    net.train()

    val_loss_avg = val_loss_epoch / max(processed_batches, 1)
    if primary:
        print(
            f"evaluate.py: dice_score={total_dice_score}, val_loss_epoch={val_loss_epoch}, val_loss_avg={val_loss_avg}, num_val_batches={processed_batches}"
        )
    return total_dice_score, val_loss_epoch, val_loss_avg, processed_batches
