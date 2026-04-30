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
import os
import shutil
import time
from pathlib import Path

# Third party
import torch
import torch.nn as nn
import torch.nn.functional as F
from distconv import DCTensor
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ScaFFold.utils.checkpointing import CheckpointManager
from ScaFFold.utils.data_loading import FractalDataset, SpatialShardSpec
from ScaFFold.utils.data_types import AMP_DTYPE, VOLUME_DTYPE
from ScaFFold.utils.dice_score import (
    SpatialAllReduce,
    compute_sharded_dice,
)
from ScaFFold.utils.distributed import get_local_rank, get_world_rank, get_world_size

# Local
from ScaFFold.utils.evaluate import evaluate
from ScaFFold.utils.perf_measure import adiak_value, begin_code_region, end_code_region
from ScaFFold.utils.utils import gather_and_print_mem


class BaseTrainer:
    """
    A class that encapsulates some basic functionality for training our model.
    """

    def __init__(self, model, config, device, log):
        self.model = model
        self.config = config
        self.device = device
        self.log = log
        self.amp_device_type = self.device.type if self.device.type != "mps" else "cpu"
        self.amp_dtype = AMP_DTYPE
        self.use_grad_scaler = False
        self.world_size = get_world_size(required=self.config.dist)
        self.world_rank = get_world_rank(required=self.config.dist)
        self.local_rank = get_local_rank(required=self.config.dist)

        # Initialize placeholders for attributes that will be set up later
        self.train_set = None
        self.val_set = None
        self.n_train = None
        self.n_val = None
        self.train_sampler = None
        self.val_sampler = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.grad_scaler = None
        self.criterion = None
        self.global_step = 0
        self.start_epoch = -1
        self.ps = getattr(self.config, "_parallel_strategy", None)
        self.spatial_mesh = None  # Spatial mesh for use w/ DistConv
        self.data_num_replicas = self.world_size
        self.data_replica_rank = self.world_rank
        if self.ps is not None:
            self.spatial_mesh = self.ps.device_mesh[self.ps.distconv_dim_names]
            self.data_num_replicas = self.ps.ddp_ranks
            self.data_replica_rank = self.ps.ddp_ind

        self.checkpoint_path_absolute = str(
            self.config.run_dir + "/" + self.config.checkpoint_dir
        )
        # We will instantiate the manager in the child class (PyTorchTrainer)
        # after components (optimizer, scaler) are created.
        self.checkpoint_manager = None

        # Create dataloaders
        self.create_dataloaders()

        # Set up optimizer, scheduler, and loss function
        self.setup_training_components()

        # Get initial mem state
        gather_and_print_mem(self.log, "after_trainer_setup")

    def create_dataset(self):
        """Create train and validation datasets."""
        dataset_dir = Path(self.config.dataset_dir)
        train_vol_dir = dataset_dir / "volumes/training"
        train_mask_dir = dataset_dir / "masks/training"
        val_vol_dir = dataset_dir / "volumes/validation"
        val_mask_dir = dataset_dir / "masks/validation"
        train_unique_masks_path = dataset_dir / "train_unique_mask_vals"
        val_unique_masks_path = dataset_dir / "val_unique_mask_vals"
        spatial_shard_spec = None
        if self.ps is not None:
            spatial_shard_spec = SpatialShardSpec(
                shard_dims=tuple(self.ps.shard_dim),
                num_shards=tuple(self.ps.num_shards),
                shard_indices=tuple(self.ps.shard_ind),
            )

        self.train_set = FractalDataset(
            train_vol_dir,
            train_mask_dir,
            data_dir=train_unique_masks_path,
            spatial_shard_spec=spatial_shard_spec,
        )
        self.val_set = FractalDataset(
            val_vol_dir,
            val_mask_dir,
            data_dir=val_unique_masks_path,
            spatial_shard_spec=spatial_shard_spec,
        )
        self.n_train = len(self.train_set)
        self.n_val = len(self.val_set)
        self.log.debug(
            f"Datasets created with n_train={self.n_train}, n_val={self.n_val}"
        )

    def create_sampler(self):
        """Create DistributedSamplers for train and validation datasets."""
        if self.config.dist:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_set,
                num_replicas=self.data_num_replicas,
                rank=self.data_replica_rank,
            )
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_set,
                num_replicas=self.data_num_replicas,
                rank=self.data_replica_rank,
                shuffle=False,
            )
        else:
            self.train_sampler = torch.utils.data.RandomSampler(self.train_set)
            self.val_sampler = torch.utils.data.SequentialSampler(self.val_set)

    def create_dataloaders(self):
        """Create dataloaders for training and validation."""
        self.create_dataset()
        self.create_sampler()

        num_workers = self.config.dataloader_num_workers
        loader_args = dict(
            batch_size=self.config.batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        if num_workers > 0:
            loader_args["persistent_workers"] = True
            loader_args["prefetch_factor"] = 2
        self.log.debug(
            f"dataloader num_workers={loader_args['num_workers']}, prefetch_factor={loader_args.get('prefetch_factor')}, persistent_workers={loader_args.get('persistent_workers', False)}, os.cpu_count()={os.cpu_count()}, self.world_size={self.world_size} "
        )
        self.train_loader = DataLoader(
            self.train_set, sampler=self.train_sampler, **loader_args
        )
        self.val_loader = DataLoader(
            self.val_set, sampler=self.val_sampler, drop_last=True, **loader_args
        )

    def setup_training_components(self):
        """Set up the optimizer, scheduler, gradient scaler, and loss function."""
        # Set up optimizer
        if self.config.optimizer == "ADAM":
            self.log.info("Using ADAM optimizer.")
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate
            )
        elif self.config.optimizer == "SGD":
            self.log.info("Using SGD optimizer.")
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.config.learning_rate
            )
        else:
            self.log.info("Using RMSprop optimizer.")
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=self.config.learning_rate, foreach=True
            )

        # Set up learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "max", patience=25
        )

        # Set up gradient scaler for AMP (Automatic Mixed Precision)
        # bfloat does not need grad scaler
        self.use_grad_scaler = (
            self.config.torch_amp and self.amp_dtype != torch.bfloat16
        )
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)

        # Set up loss function
        self.criterion = (
            nn.CrossEntropyLoss()
            if self.config.n_categories + 1 > 1
            else nn.BCEWithLogitsLoss()
        )

        self.log.info(
            f"Optimizer: {self.optimizer}, Scheduler: {self.scheduler}, AMP dtype: {self.amp_dtype}, Gradient Scaler Enabled: {self.use_grad_scaler}"
        )

    def _autocast_kwargs(self, enabled=None):
        if enabled is None:
            enabled = self.config.torch_amp

        kwargs = {"device_type": self.amp_device_type, "enabled": enabled}
        if enabled:
            kwargs["dtype"] = self.amp_dtype
        return kwargs

    @staticmethod
    def _foreground_dice_mean(dice_scores):
        """Match optimization to the reported validation metric by excluding background."""
        if dice_scores.size(1) > 1:
            return dice_scores[:, 1:].mean()
        return dice_scores.mean()


class PyTorchTrainer(BaseTrainer):
    """
    A class for training our model with PyTorch.
    """

    def __init__(self, model, config, device, log):
        super().__init__(model, config, device, log)

        self.outfile_path = str(self.config.run_dir) + "/train_stats.csv"

        self.checkpoint_manager = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            grad_scaler=self.grad_scaler,
            base_dir=self.checkpoint_path_absolute,
            log=self.log,
            world_rank=self.world_rank,
            dist_enabled=self.config.dist,
            # Check config for async setting, default to False
            async_save=getattr(self.config, "async_save", False),
        )

    def cleanup_or_resume(self):
        """
        Clean up existing train stats and checkpoints,
        or resume training from the latest checkpoint.
        """

        self.checkpoint_manager.cleanup(self.config.train_from_scratch)

        # If we cleaned up (train_from_scratch=True), this deletes the files.
        # If we didn't, we can try to load.
        if self.config.train_from_scratch:
            # Clear stats file on rank 0
            if self.world_rank == 0:
                if os.path.exists(self.outfile_path):
                    os.remove(self.outfile_path)
                # Clear predictions (logic from original code)
                pred_path = os.path.join(str(self.config.run_dir), "predictions")
                if os.path.exists(pred_path):
                    try:
                        shutil.rmtree(pred_path)
                    except Exception:
                        pass

            self.start_epoch = 1
        else:
            # Load checkpoint via manager
            self.start_epoch = self.checkpoint_manager.load_from_checkpoint()

            # Restore extra metadata if needed (e.g. mask values)
            if "train_mask_values" in self.checkpoint_manager.restored_extras:
                self.train_set.mask_values = self.checkpoint_manager.restored_extras[
                    "train_mask_values"
                ]

            # If we loaded a checkpoint (start_epoch > 1), we must ensure the CSV
            # matches the state of that checkpoint.
            if (
                self.world_rank == 0
                and self.start_epoch > 1
                and os.path.exists(self.outfile_path)
            ):
                self._truncate_stats_file(self.start_epoch)

        # Set up the output file headers
        headers = [
            "epoch",
            "epoch_loss",
            "overall_loss",
            "val_loss_epoch",
            "val_loss_avg",
            "train_dice",
            "val_dice",
            "epoch_duration",
        ]
        if self.world_rank == 0 and self.start_epoch == 1:
            with open(self.outfile_path, "a", newline="") as outfile:
                outfile.write(",".join(headers) + "\n")

    def _truncate_stats_file(self, start_epoch):
        """
        Scans the stats file and truncates it at the first occurrence of
        an epoch >= start_epoch. This is O(1) memory and safe for large logs.
        """
        self.log.info(
            f"Truncating {self.outfile_path} to remove epochs >= {start_epoch}"
        )

        try:
            # Open in read+update mode ('r+') to allow seeking and truncating
            with open(self.outfile_path, "r+") as f:
                header = f.readline()
                if not header:
                    return

                # Identify the index of the 'epoch' column
                headers = header.strip().split(",")
                try:
                    epoch_idx = headers.index("epoch")
                except ValueError:
                    epoch_idx = 0

                while True:
                    # Save the current file position (start of the line)
                    current_pos = f.tell()
                    line = f.readline()

                    # End of file reached
                    if not line:
                        break

                    parts = line.strip().split(",")
                    try:
                        row_epoch = int(float(parts[epoch_idx]))

                        # If we find a row that is "from the future" (or the current restarting epoch)
                        if row_epoch >= start_epoch:
                            # Move pointer back to the start of this line
                            f.seek(current_pos)
                            # Cut the file off right here
                            f.truncate()
                            self.log.info(
                                f"Truncated stats file at byte {current_pos} (found epoch {row_epoch})"
                            )
                            break
                    except (ValueError, IndexError):
                        # Skip malformed lines, or decide to stop.
                        # Usually safe to continue scanning.
                        pass

        except Exception as e:
            self.log.warning(f"Failed to truncate stats file: {e}")

    def _get_memsize(self, tensor, tensor_label: str, verbosity: int = 0):
        """Log size of tensor in memory"""

        if verbosity < 2:
            return
        tensor_memory_bytes = tensor[0].element_size() * tensor[0].nelement()
        tensor_memory_gb = tensor_memory_bytes / (1024**3)
        self.log.info(f"{tensor_label} size on GPU: {tensor_memory_gb:.2f} GB")

    def warmup(self):
        """Run warmup iterations before the main training loop."""
        warmup_batches = self.config.warmup_batches
        if warmup_batches <= 0:
            return

        if self.config.dist:
            self.train_loader.sampler.set_epoch(0)

        # Match the main training path as closely as possible.
        self.model.train()
        self.optimizer.zero_grad(set_to_none=False)
        start_warmup = time.time()
        max_batches = min(warmup_batches, len(self.train_loader))
        self.log.info(f"Running {max_batches} warmup batch(es) per rank")

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx >= max_batches:
                break

            images, true_masks = batch["image"], batch["mask"]

            images = images.to(
                device=self.device,
                dtype=torch.float32,
                memory_format=torch.channels_last_3d,
                non_blocking=True,
            )
            true_masks = true_masks.to(
                device=self.device, dtype=torch.long, non_blocking=True
            ).contiguous()

            # Add a dummy channel dimension to get 5D [B, 1, D, H, W]
            true_masks = true_masks.unsqueeze(1)

            # Inputs are already loaded as local shards by the dataset.
            images_dc = DCTensor.from_shard(images, self.ps)
            true_masks_dc = DCTensor.from_shard(true_masks, self.ps)
            self._get_memsize(images_dc, "Sharded image", self.config.verbose)

            with torch.autocast(**self._autocast_kwargs()):
                # Forward on DCTensor
                self.log.debug("  warmup: running forward pass")
                masks_pred_dc = self.model(images_dc)
                self.log.debug("  warmup: forward pass complete")

                # Extract the underlying PyTorch local tensors
                local_preds = masks_pred_dc
                local_labels_5d = true_masks_dc

                # Remove the dummy channel dimension so CE Loss is happy [B, D, H, W]
                local_labels = local_labels_5d.squeeze(1)
                if self.world_rank == 0:
                    self.log.debug(f"  warmup: Local Preds Shape: {local_preds.shape}")
                    # Should be something like [1, 6, 128, 128, 64] if sharding Width by 2
                    self.log.debug(
                        f"  warmup: Local Labels Shape: {local_labels.shape}"
                    )
                    # Should be something like [1, 128, 128, 64]

                # --- SHARDED LOSS CALCULATION ---
                current_mem = torch.cuda.memory_allocated() / (1024**3)
                self.log.debug(
                    f"  warmup: Calculating sharded loss. Mem: {current_mem:.2f} GB."
                )

                # Calculate CE and Dice loss in single precision for numerical stability.
                with torch.autocast(**self._autocast_kwargs(enabled=False)):
                    # Compute global CE loss from sharded CE loss
                    local_ce_sum = F.cross_entropy(
                        local_preds.float(), local_labels, reduction="sum"
                    )
                    global_ce_sum = SpatialAllReduce.apply(
                        local_ce_sum, self.spatial_mesh
                    )
                    local_voxel_count = torch.tensor(
                        float(local_labels.numel()),
                        device=local_labels.device,
                        dtype=VOLUME_DTYPE,
                    )
                    global_total_voxels = SpatialAllReduce.apply(
                        local_voxel_count, self.spatial_mesh
                    )
                    loss_ce = global_ce_sum / global_total_voxels

                    # Compute global dice loss from sharded dice loss
                    local_preds_softmax = F.softmax(local_preds.float(), dim=1)
                    local_labels_one_hot = (
                        F.one_hot(
                            local_labels, num_classes=self.config.n_categories + 1
                        )
                        .permute(0, 4, 1, 2, 3)
                        .float()
                    )
                    dice_scores = compute_sharded_dice(
                        local_preds_softmax, local_labels_one_hot, self.spatial_mesh
                    )
                    batch_dice_score = self._foreground_dice_mean(dice_scores)

                    # Sum global CE Loss and Dice loss
                    loss = loss_ce + (1.0 - batch_dice_score)

            self.log.debug(
                "  warmup: loss calculation complete. Proceeding to backward pass"
            )

            # Backward pass
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.log.debug("  warmup: backward pass complete. Stepping optimizer")

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # Free memory aggressively
            del images_dc, true_masks_dc, masks_pred_dc
            del (
                local_preds,
                local_labels,
                local_preds_softmax,
                local_labels_one_hot,
            )
            del loss_ce, loss

            if self.world_rank == 0:
                peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
                peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                self.log.debug(
                    f"[MEM-PEAK] Peak alloc: {peak_alloc:.2f} GiB | Peak reserved: {peak_reserved:.2f} GiB",
                )
            batch_t_end = time.time()
            self.log.debug(
                f"  warmup: batch {batch_idx} completed in {batch_t_end - start_warmup} seconds"
            )

        # Nuke any accumulated grads so the first real step starts clean
        for p in self.model.parameters():
            p.grad = None
        self.optimizer.zero_grad(set_to_none=True)

        if self.config.dist:
            self.val_loader.sampler.set_epoch(0)

        evaluate(
            self.model,
            self.val_loader,
            self.device,
            self.config.torch_amp,
            self.world_rank == 0,
            self.criterion,
            self.config.n_categories,
            self.config._parallel_strategy,
        )
        self.model.train()

        if self.config.dist:
            torch.distributed.barrier()
        self.log.info(f"Done warmup. Took {int(time.time() - start_warmup)}s")

    def train(self):
        """
        Execute model training
        """

        epoch = 1
        dice_score_train = 0
        with open(self.outfile_path, "a", newline="") as outfile:
            start = time.time()
            while dice_score_train < self.config.target_dice:
                if self.config.epochs != -1 and epoch > self.config.epochs:
                    print(
                        f"Maxmimum epochs reached '{self.config.epochs}'. Concluding training early (may have not converged)."
                    )
                    break

                # Timer and tracking variables
                epoch_start_time = time.time()
                train_dice_total = 0
                epoch_loss = 0  # Accumulator for per-batch losses

                # Set necessary modes/states
                if self.config.dist:
                    self.train_loader.sampler.set_epoch(epoch)
                    self.val_loader.sampler.set_epoch(epoch)
                self.model.train()
                self.optimizer.zero_grad(set_to_none=False)

                estr = (
                    f"{epoch}"
                    if self.config.epochs == -1
                    else f"{epoch}/{self.config.epochs}"
                )
                with tqdm(
                    total=len(self.train_sampler),
                    desc=f"({os.path.basename(self.config.run_dir)}) \
                            Epoch {estr}",
                    unit="img",
                    disable=True if self.world_rank != 0 else False,
                ) as pbar:
                    begin_code_region("batch_loop")
                    for batch in self.train_loader:
                        # Load initial samples and labels
                        images, true_masks = batch["image"], batch["mask"]

                        begin_code_region("image_to_device")
                        images = images.to(
                            device=self.device,
                            dtype=torch.float32,
                            memory_format=torch.channels_last_3d,  # NDHWC (channels last) vs NCDHW (channels first)
                            non_blocking=True,
                        )
                        true_masks = true_masks.to(
                            device=self.device, dtype=torch.long, non_blocking=True
                        ).contiguous()  # masks no channels NDHW, but ensure continuity.
                        end_code_region("image_to_device")
                        gather_and_print_mem(self.log, "after_batch_to_device")

                        # Add a dummy channel dimension to get 5D [B, 1, D, H, W]
                        true_masks = true_masks.unsqueeze(1)

                        # Inputs are already loaded as local shards by the dataset.
                        images_dc = DCTensor.from_shard(images, self.ps)
                        true_masks_dc = DCTensor.from_shard(true_masks, self.ps)
                        del images, true_masks
                        self._get_memsize(
                            images_dc, "Sharded image", self.config.verbose
                        )

                        with torch.autocast(**self._autocast_kwargs()):
                            # Predict on this batch
                            torch.cuda.reset_peak_memory_stats()
                            gather_and_print_mem(self.log, "pre_forward")
                            begin_code_region("predict")
                            masks_pred_dc = self.model(images_dc)
                            end_code_region("predict")
                            gather_and_print_mem(self.log, "post_forward")

                            # Extract the underlying PyTorch local tensors
                            local_preds = masks_pred_dc
                            local_labels_5d = true_masks_dc

                            # Remove the dummy channel dimension so CE Loss is happy [B, D, H, W]
                            local_labels = local_labels_5d.squeeze(1)
                            if self.world_rank == 0:
                                self.log.debug(
                                    f"Local Preds Shape: {local_preds.shape}"
                                )
                                # Should be something like [1, 6, 128, 128, 64] if sharding Width by 2
                                self.log.debug(
                                    f"Local Labels Shape: {local_labels.shape}"
                                )
                                # Should be something like [1, 128, 128, 64]

                            begin_code_region("calculate_loss")
                            # --- SHARDED LOSS CALCULATION ---
                            current_mem = torch.cuda.memory_allocated() / (1024**3)
                            self.log.debug(
                                f"Calculating sharded loss. Mem: {current_mem:.2f} GB."
                            )

                            # Calculate CE and Dice loss in single precision for numerical stability.
                            with torch.autocast(**self._autocast_kwargs(enabled=False)):
                                # Compute global CE loss from sharded CE loss
                                local_ce_sum = F.cross_entropy(
                                    local_preds.float(),
                                    local_labels,
                                    reduction="sum",
                                )
                                global_ce_sum = SpatialAllReduce.apply(
                                    local_ce_sum, self.spatial_mesh
                                )
                                local_voxel_count = torch.tensor(
                                    float(local_labels.numel()),
                                    device=local_labels.device,
                                    dtype=VOLUME_DTYPE,
                                )
                                global_total_voxels = SpatialAllReduce.apply(
                                    local_voxel_count, self.spatial_mesh
                                )
                                loss_ce = global_ce_sum / global_total_voxels

                                # Compute global dice loss from sharded dice loss
                                local_preds_softmax = F.softmax(
                                    local_preds.float(), dim=1
                                )
                                local_labels_one_hot = (
                                    F.one_hot(
                                        local_labels,
                                        num_classes=self.config.n_categories + 1,
                                    )
                                    .permute(0, 4, 1, 2, 3)
                                    .float()
                                )
                                dice_scores = compute_sharded_dice(
                                    local_preds_softmax,
                                    local_labels_one_hot,
                                    self.spatial_mesh,
                                )
                                batch_dice_score = self._foreground_dice_mean(
                                    dice_scores
                                )

                                # Sum global CE Loss and Dice loss
                                loss = loss_ce + (1.0 - batch_dice_score)
                                train_dice_total += batch_dice_score

                            end_code_region("calculate_loss")

                        gather_and_print_mem(self.log, "pre_backward")
                        begin_code_region("backward")
                        self.grad_scaler.scale(loss).backward()
                        end_code_region("backward")
                        gather_and_print_mem(self.log, "post_backward")

                        begin_code_region("step_and_update")
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        self.grad_scaler.step(self.optimizer)
                        gather_and_print_mem(self.log, "after_optim_step")
                        self.grad_scaler.update()
                        self.optimizer.zero_grad(set_to_none=False)
                        end_code_region("step_and_update")

                        # Update the loss
                        begin_code_region("update_loss")
                        pbar.update(images_dc.shape[0])
                        self.global_step += 1
                        # Stay on GPU
                        epoch_loss += loss.detach()
                        end_code_region("update_loss")
                    end_code_region("batch_loop")

                # Calculate overall loss as average of per-batch loss
                overall_loss = epoch_loss.item() / len(self.train_loader)

                #
                # Evaluate model on validation set, update LR if necessary
                #
                dice_sum, val_loss_epoch, val_loss_avg, numbatch = evaluate(
                    self.model,
                    self.val_loader,
                    self.device,
                    self.config.torch_amp,
                    self.world_rank == 0,
                    self.criterion,
                    self.config.n_categories,
                    self.config._parallel_strategy,
                )
                dice_info = torch.tensor([dice_sum, numbatch])
                if self.config.dist:
                    dice_info = dice_info.to(device=self.device)
                    torch.distributed.all_reduce(
                        dice_info, op=torch.distributed.ReduceOp.SUM
                    )
                val_score = dice_info[0].item() / max(dice_info[1].item(), 1)
                if not self.config.disable_scheduler:
                    # The following is true when trying to overfit,
                    # in which case we only care about train loss
                    if self.n_train == 1 or "overfit" in self.outfile_path:
                        self.log.debug(
                            "WARNING: scheduler step by overall_loss, \
                                    not val_score (n_train==1 or overfit in outfile_path)"
                        )
                        self.scheduler.step(overall_loss)
                    else:  # Otherwise, we're really trying to optimize for validation dice score
                        self.scheduler.step(val_score)
                else:
                    self.log.debug("scheduler disabled, no LR update this step")

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                #
                # Write out data for this epoch to train stats csv
                #
                train_dice = float(train_dice_total.item() / len(self.train_loader))
                self.log.info(
                    f" epoch {epoch} \
                            | train_dice_loss {train_dice:.6f} (type {type(train_dice)}) \
                            | val_dice_score {val_score:.6f} \
                            | lr {self.config.learning_rate:.8f}"
                )
                self.log.debug(f" writing to csv at {self.outfile_path}")
                if self.world_rank == 0:
                    outfile.write(
                        ",".join(
                            [
                                str(epoch),
                                str(epoch_loss.item()),
                                str(overall_loss),
                                str(val_loss_epoch),
                                str(val_loss_avg),
                                str(train_dice),
                                str(val_score),
                                str(epoch_duration),
                            ]
                        )
                        + "\n"
                    )
                    outfile.flush()
                    print(
                        f"Epoch {epoch} completed in {epoch_duration} seconds. Total train time so far: {time.time() - start}"
                    )

                #
                # Checkpointing
                #
                begin_code_region("checkpoint")

                # A checkpoint interval of -1 disables checkpointing entirely.
                if (
                    self.config.checkpoint_interval > 0
                    and epoch % self.config.checkpoint_interval == 0
                ):
                    extras = {"train_mask_values": self.train_set.mask_values}
                    self.checkpoint_manager.save_checkpoint(epoch, val_loss_avg, extras)

                end_code_region("checkpoint")

                dice_score_train = val_score
                epoch += 1

                # This check must exist otherwise the condition dice_score_train < self.config.target_dice will evaluate to False and incorrectly exit the training
                if math.isnan(dice_score_train):
                    raise ValueError(
                        "Invalid value (NaN) encountered in dice score computation"
                    )

        adiak_value("final_epochs", epoch)
