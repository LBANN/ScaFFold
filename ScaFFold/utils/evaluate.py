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
from distconv import DCTensor
from tqdm import tqdm

from ScaFFold.utils.data_types import AMP_DTYPE
from ScaFFold.utils.dice_score import compute_sharded_dice, labels_to_onehot
from ScaFFold.utils.losses import compute_sharded_cross_entropy_loss
from ScaFFold.utils.perf_measure import annotate


@annotate()
@torch.inference_mode()
def evaluate(
    net,
    dataloader,
    device,
    amp,
    primary,
    criterion,
    n_categories,
    parallel_strategy,
    max_batches=None,
    log=None,
):
    """Run validation over ``dataloader`` and return aggregate metrics.

    Returns a 5-tuple ``(total_dice_score, val_loss_epoch, val_loss_avg,
    processed_batches, processed_samples)``:

    * ``total_dice_score`` -- sum over samples of the per-sample mean foreground
      *hard* (argmax) Dice. The caller recovers the mean val Dice as
      ``total_dice_score / processed_samples`` (summing both across data-parallel
      replicas first when distributed).
    * ``val_loss_epoch`` -- sample-weighted total validation loss: the sum over
      batches of each batch's mean loss times its sample count.
    * ``val_loss_avg`` -- ``val_loss_epoch / processed_samples``, i.e. the true
      per-sample mean loss, independent of how samples were grouped into
      batches.
    * ``processed_batches`` -- number of batches actually evaluated.
    * ``processed_samples`` -- number of samples actually evaluated.
    """
    # A single output channel (n_categories == 0) has no meaningful multiclass
    # validation metric: softmax over one channel and one-hot over one class are
    # both identically 1, so the Dice and loss would be fixed regardless of the
    # model. Reject it here rather than report a fake perfect score.
    if n_categories < 1:
        raise ValueError(
            f"evaluate requires n_categories >= 1 (got {n_categories}); a single "
            "output channel has no meaningful multiclass validation metric."
        )
    n_classes = n_categories + 1

    def foreground_dice_stats(dice_scores):
        # Channel 0 is background; average Dice over the foreground classes
        # only. n_categories >= 1 guarantees at least one foreground channel.
        # The sum stays on-device (no .item() sync per batch); the sample count
        # is read from tensor shape metadata, which does not synchronize.
        per_sample_scores = dice_scores[:, 1:].mean(dim=1)
        return per_sample_scores.sum(), per_sample_scores.numel()

    net.eval()
    autocast_device_type = device.type if device.type != "mps" else "cpu"
    autocast_kwargs = {"device_type": autocast_device_type, "enabled": amp}
    if amp:
        autocast_kwargs["dtype"] = AMP_DTYPE
    num_val_batches = len(dataloader)
    if max_batches is not None:
        num_val_batches = min(num_val_batches, max_batches)
    # Dice and loss accumulate as device tensors so the per-batch reads do not
    # synchronize the host with the accelerator; a single .item() after the loop
    # drains them. processed_samples/batches are plain Python counts.
    total_dice_score = None
    val_loss_epoch = None
    processed_batches = 0
    processed_samples = 0

    if parallel_strategy is not None:
        spatial_mesh = parallel_strategy.device_mesh[
            parallel_strategy.distconv_dim_names
        ]
        if primary and log is not None:
            log.debug(
                "[eval] ps.shard_dim=%s num_shards=%s",
                parallel_strategy.shard_dim,
                parallel_strategy.num_shards,
            )
    else:
        # No parallel strategy: no spatial sharding, no mesh to reduce over.
        spatial_mesh = None

    with torch.autocast(**autocast_kwargs):
        class_weights = getattr(criterion, "weight", None)
        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                total=num_val_batches,
                desc="Validation round",
                unit="batch",
                leave=False,
                disable=not primary,
            )
        ):
            if batch_idx >= num_val_batches:
                break
            image, mask_true = batch["image"], batch["mask"]

            # non_blocking overlaps the pinned-memory H2D copy with prior kernel
            # work; the loaders are built with pin_memory=True precisely for
            # this. The forward pass and the end-of-loop .item() provide the
            # necessary ordering before any value is read on the host.
            image = image.to(
                device=device,
                dtype=torch.float32,
                memory_format=torch.channels_last_3d,
                non_blocking=True,
            )
            mask_true = mask_true.to(
                device=device, dtype=torch.long, non_blocking=True
            ).contiguous()

            # Dummy channel dimension [B, 1, D, H, W]
            mask_true = mask_true.unsqueeze(1)

            # Inputs are already loaded as local shards by the dataset.
            # Without a parallel strategy there is nothing to shard.
            if parallel_strategy is not None:
                dcx = DCTensor.from_shard(image, parallel_strategy)
                mask_true_dc = DCTensor.from_shard(mask_true, parallel_strategy)
            else:
                dcx = image
                mask_true_dc = mask_true

            # Forward pass on sharded data
            dcy = net(dcx)

            # Extract underlying local tensors (STAY SHARDED)
            local_preds = dcy
            local_labels_5d = mask_true_dc
            local_labels = local_labels_5d.squeeze(1)

            # Skip empty batches
            if local_preds.size(0) == 0 or local_labels.size(0) == 0:
                continue

            # Calculate CE and Dice loss in single precision for numerical stability.
            with torch.autocast(device_type=autocast_device_type, enabled=False):
                # Share one upcast + log-softmax between the CE term (via NLL)
                # and the soft-dice term (via exp), instead of upcasting the
                # logits twice and recomputing softmax on top of CE's internal
                # log-softmax.
                log_probs = torch.log_softmax(local_preds.float(), dim=1)
                CE_loss = compute_sharded_cross_entropy_loss(
                    local_preds,
                    local_labels,
                    spatial_mesh,
                    parallel_strategy.num_shards
                    if parallel_strategy is not None
                    else (1,),
                    autocast_device_type,
                    class_weights,
                    log_probs=log_probs,
                )

                mask_true_onehot = labels_to_onehot(local_labels, n_classes)

                # The reported score is the hard (argmax) segmentation Dice: it
                # must reflect the discrete prediction, not the model's
                # confidence. argmax is per-voxel, so hardening is shard-local
                # and the sharded reduction is unaffected. One-hot hard
                # predictions also make the empty/empty guard in
                # compute_sharded_dice reachable (a correctly-predicted absent
                # class scores 1, not ~0).
                mask_pred_hard = labels_to_onehot(local_preds.argmax(dim=1), n_classes)
                dice_score_hard = compute_sharded_dice(
                    mask_pred_hard, mask_true_onehot, spatial_mesh
                )
                batch_dice_sum, batch_sample_count = foreground_dice_stats(
                    dice_score_hard
                )

                # The loss term mirrors the training objective, which uses the
                # soft (softmax-probability) Dice.
                mask_pred_probs = log_probs.exp()
                dice_score_probs = compute_sharded_dice(
                    mask_pred_probs, mask_true_onehot, spatial_mesh
                )
                soft_dice_sum, soft_sample_count = foreground_dice_stats(
                    dice_score_probs
                )
                soft_dice_score = soft_dice_sum / max(soft_sample_count, 1)

                # Sum global CE Loss and Dice loss.
                loss = CE_loss + (1.0 - soft_dice_score)
                # Weight each batch's mean loss by its sample count so a ragged
                # final batch is not overweighted; divide by the total sample
                # count below, mirroring the sample-weighted dice accumulation.
                # Accumulate on-device: no host sync until the single .item()
                # after the loop.
                batch_loss_weighted = loss * batch_sample_count
                val_loss_epoch = (
                    batch_loss_weighted
                    if val_loss_epoch is None
                    else val_loss_epoch + batch_loss_weighted
                )
                total_dice_score = (
                    batch_dice_sum
                    if total_dice_score is None
                    else total_dice_score + batch_dice_sum
                )
            processed_batches += 1
            processed_samples += batch_sample_count

    net.train()

    # Drain the device-side accumulators exactly once here (a single host sync
    # for the whole validation pass rather than one per batch).
    total_dice_score = float(total_dice_score) if total_dice_score is not None else 0.0
    val_loss_epoch = float(val_loss_epoch) if val_loss_epoch is not None else 0.0

    # val_loss_epoch is the sample-weighted total loss (sum of each batch's mean
    # loss times its sample count), so dividing by the total sample count gives
    # the true per-sample mean, independent of how samples were batched.
    val_loss_avg = val_loss_epoch / max(processed_samples, 1)
    if primary and log is not None:
        log.debug(
            "evaluate.py: dice_score=%s, val_loss_epoch=%s, val_loss_avg=%s, "
            "num_val_batches=%s, num_val_samples=%s",
            total_dice_score,
            val_loss_epoch,
            val_loss_avg,
            processed_batches,
            processed_samples,
        )
    return (
        total_dice_score,
        val_loss_epoch,
        val_loss_avg,
        processed_batches,
        processed_samples,
    )
