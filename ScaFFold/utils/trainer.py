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

# Standard library
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

# Third party
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from distconv import DCTensor
from torch import optim
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ScaFFold.utils.checkpointing import CheckpointManager
from ScaFFold.utils.data_loading import FractalDataset
from ScaFFold.utils.dice_score import dice_loss
from ScaFFold.utils.distributed import get_local_rank, get_world_rank, get_world_size

# Local
from ScaFFold.utils.evaluate import evaluate
from ScaFFold.utils.perf_measure import begin_code_region, end_code_region
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

        self.train_set = FractalDataset(
            train_vol_dir, train_mask_dir, data_dir=train_unique_masks_path
        )
        self.val_set = FractalDataset(
            val_vol_dir, val_mask_dir, data_dir=val_unique_masks_path
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
                self.train_set
            )
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_set, shuffle=False
            )
        else:
            self.train_sampler = torch.utils.data.RandomSampler(self.train_set)
            self.val_sampler = torch.utils.data.SequentialSampler(self.val_set)

    def create_dataloaders(self):
        """Create dataloaders for training and validation."""
        self.create_dataset()
        self.create_sampler()

        loader_args = dict(
            batch_size=self.config.batch_size, num_workers=1, pin_memory=True
        )
        self.log.debug(
            f"dataloader num_workers={loader_args['num_workers']}, os.cpu_count()={os.cpu_count()}, self.world_size={self.world_size} "
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
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.config.torch_amp)

        # Set up loss function
        self.criterion = (
            nn.CrossEntropyLoss()
            if self.config.n_categories + 1 > 1
            else nn.BCEWithLogitsLoss()
        )

        self.log.info(
            f"Optimizer: {self.optimizer}, Scheduler: {self.scheduler}, Gradient Scaler Enabled: {self.config.torch_amp}"
        )


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

    def train(self):
        """
        Execute model training
        """

        self.cleanup_or_resume()

        warmup_epochs = self.config.warmup_epochs
        if warmup_epochs > 0:
            begin_code_region("warmup")
            # Keep BN/Dropout from changing behavior/statistics
            self.model.eval()
            start_warmup = time.time()
            self.log.info(f"Running {warmup_epochs} warmup epoch(s)")

            ps = getattr(self.config, "_parallel_strategy", None)

            for _ in range(warmup_epochs):
                for batch in self.train_loader:
                    images, true_masks = batch["image"], batch["mask"]

                    images = images.to(
                        device=self.device,
                        dtype=torch.float32,
                        memory_format=torch.channels_last_3d,
                        non_blocking=False,
                    )
                    images_dc = DCTensor.distribute(images, ps)

                    true_masks = true_masks.to(
                        device=self.device, dtype=torch.long, non_blocking=True
                    )

                    with torch.autocast(
                        self.device.type if self.device.type != "mps" else "cpu",
                        enabled=self.config.torch_amp,
                    ):
                        # Forward on DCTensor
                        masks_pred_dc = self.model(images_dc)

                        # Convert predictions for loss
                        if isinstance(ps.num_shards, tuple) and len(ps.num_shards) == 1:
                            n_shards = ps.num_shards[0]
                        else:
                            n_shards = ps.num_shards
                        if images.size(0) < n_shards:
                            # For small batches (e.g., N=1 with dc_num_shards=2), replicate outputs
                            masks_pred = masks_pred_dc.to_replicate()
                            labels_for_loss = true_masks
                        else:
                            # Otherwise, shard labels across batch dim to match to_ddp layout
                            masks_pred = masks_pred_dc.to_ddp()
                            dt_labels = distribute_tensor(
                                true_masks,
                                device_mesh=ps.device_mesh[
                                    f"dc{self.config.shard_dim + 2}"
                                ],
                                placements=[Shard(0)],
                            )
                            labels_for_loss = dt_labels.to_local()

                        CE_loss = self.criterion(masks_pred, labels_for_loss)

                        # Calculate the train dice loss
                        masks_pred_softmax = F.softmax(masks_pred, dim=1).float()
                        true_masks_onehot = (
                            F.one_hot(labels_for_loss, self.config.n_categories + 1)
                            .permute(0, 4, 1, 2, 3)
                            .float()
                        )
                        train_dice_curr = dice_loss(
                            masks_pred_softmax,
                            true_masks_onehot,
                            multiclass=True,
                        )
                        loss = CE_loss + train_dice_curr

                        # Fine as long as we don't step/update
                        self.grad_scaler.scale(loss).backward()

            # Nuke any accumulated grads so the first real step starts clean
            for p in self.model.parameters():
                p.grad = None
            torch.distributed.barrier()
            end_code_region("warmup")
            self.log.info(f"Done warmup. Took {int(time.time() - start_warmup)}s")

        with open(self.outfile_path, "a", newline="") as outfile:
            start = time.time()
            for epoch in range(self.start_epoch, self.config.epochs + 1):
                # DistConv ParallelStrategy
                ps = getattr(self.config, "_parallel_strategy", None)
                if ps is None:
                    raise RuntimeError(
                        "ParallelStrategy not found in config. Set config._parallel_strategy when wrapping model with DistConvDDP."
                    )

                # Timer and tracking variables
                epoch_start_time = time.time()
                train_dice_curr = 0
                train_dice_total = 0
                CE_loss = 0
                epoch_loss = 0  # Accumulator for per-batch losses

                # Set necessary modes/states
                if self.config.dist:
                    self.train_loader.sampler.set_epoch(epoch)
                    self.val_loader.sampler.set_epoch(epoch)
                self.model.train()

                with tqdm(
                    total=self.n_train // self.world_size,
                    desc=f"({os.path.basename(self.config.run_dir)}) \
                            Epoch {epoch}/{self.config.epochs}",
                    unit="img",
                    disable=True if self.world_rank != 0 else False,
                ) as pbar:
                    batch_step = 0

                    begin_code_region("batch_loop")
                    for batch in self.train_loader:
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

                        # Replicate batch across dc mesh, shard batch across ddp mesh.
                        # This ensures all dc ranks in the same ddp group see the same samples,
                        # and ddp ranks see disjoint samples.
                        images_dp = distribute_tensor(
                            images, ps.device_mesh, placements=[Shard(0), Replicate()]
                        ).to_local()
                        true_masks_dp = distribute_tensor(
                            true_masks,
                            ps.device_mesh,
                            placements=[Shard(0), Replicate()],
                        ).to_local()

                        with torch.autocast(
                            self.device.type if self.device.type != "mps" else "cpu",
                            enabled=self.config.torch_amp,
                        ):
                            # Predict on this batch
                            torch.cuda.reset_peak_memory_stats()
                            gather_and_print_mem(self.log, "pre_forward")
                            begin_code_region("predict")

                            # Spatially shard the chosen dimension across dc mesh
                            dcx = DCTensor.distribute(images_dp, ps)
                            dcy = self.model(dcx)
                            # Convert back to batch-sharded layout across the dc mesh
                            masks_pred = dcy.to_ddp()

                            end_code_region("predict")
                            gather_and_print_mem(self.log, "post_forward")

                            # Reshard labels across dc mesh to match masks_pred's batch partition
                            # Start from dc-replicated labels, then shard batch across dc
                            true_masks_ddp = (
                                DTensor.from_local(
                                    true_masks_dp,
                                    device_mesh=ps.device_mesh[
                                        f"dc{self.config.shard_dim + 2}"
                                    ],
                                    placements=[Replicate()],
                                )
                                .redistribute(
                                    device_mesh=ps.device_mesh[
                                        f"dc{self.config.shard_dim + 2}"
                                    ],
                                    placements=[Shard(0)],
                                )
                                .to_local()
                            )

                            begin_code_region("calculate_loss")
                            # Calculate the loss
                            if self.config.n_categories + 1 == 1:
                                loss = self.criterion(
                                    masks_pred.squeeze(1), true_masks_ddp.float()
                                )
                                loss += dice_loss(
                                    F.sigmoid(masks_pred.squeeze(1)),
                                    true_masks_ddp.float(),
                                    multiclass=False,
                                )
                            else:
                                # Calculate the CrossEntropy loss
                                CE_loss = self.criterion(masks_pred, true_masks_ddp)

                                # Calculate the train dice loss
                                masks_pred_softmax = F.softmax(
                                    masks_pred, dim=1
                                ).float()
                                true_masks_onehot = (
                                    F.one_hot(
                                        true_masks_ddp, self.config.n_categories + 1
                                    )
                                    .permute(0, 4, 1, 2, 3)
                                    .float()
                                )
                                train_dice_curr = dice_loss(
                                    masks_pred_softmax,
                                    true_masks_onehot,
                                    multiclass=True,
                                )

                                # Our loss function is CE loss + dice loss
                                loss = CE_loss + train_dice_curr

                                # Track the train dice loss separately for debugging
                                train_dice_total += train_dice_curr
                            end_code_region("calculate_loss")

                        gather_and_print_mem(self.log, "pre_backward")
                        begin_code_region("backward")
                        self.grad_scaler.scale(loss).backward()
                        end_code_region("backward")
                        gather_and_print_mem(self.log, "post_backward")

                        begin_code_region("step_and_update")
                        if batch_step + 1 == len(self.train_loader):
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
                        pbar.update(images_dp.shape[0])
                        self.global_step += 1
                        batch_step += 1
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
                    True if self.world_rank == 0 else False,
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
                train_dice = float(train_dice_total / len(self.train_loader))
                self.log.info(
                    f" epoch {epoch} \
                            | train_dice_loss {train_dice:.6f} (type {type(train_dice)}) \
                            | val_dice_score {val_score:.6f}"
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

                if epoch % self.config.checkpoint_interval == 0:
                    extras = {"train_mask_values": self.train_set.mask_values}
                    self.checkpoint_manager.save_checkpoint(epoch, val_loss_avg, extras)

                end_code_region("checkpoint")

                if val_score >= 0.95:
                    self.log.info(
                        f"val_score of {val_score} is > threshold of 0.95. Benchmark run complete. Wrapping up..."
                    )
                    return 0
