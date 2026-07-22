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
import math
import os
import shutil
import statistics
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
from ScaFFold.utils.data_types import AMP_DTYPE, VOLUME_TORCH_DTYPE
from ScaFFold.utils.dice_score import compute_sharded_dice
from ScaFFold.utils.distributed import get_local_rank, get_world_rank, get_world_size

# Local
from ScaFFold.utils.evaluate import evaluate
from ScaFFold.utils.losses import (
    _compute_ce_class_weights,
    compute_sharded_cross_entropy_loss,
)
from ScaFFold.utils.perf_measure import adiak_value, begin_code_region, end_code_region
from ScaFFold.utils.utils import gather_and_print_mem


class _UnpaddedDistributedSampler(torch.utils.data.Sampler):
    """Shard a dataset into contiguous, unpadded, per-rank index ranges.

    Unlike ``DistributedSampler`` (which pads by repeating leading samples so
    every rank iterates ``ceil(n / num_replicas)`` items), this hands each
    sample to exactly one rank. Ranks therefore receive uneven counts and a
    rank may legitimately receive zero samples, but no sample is ever visited
    twice. That property is required for an unbiased validation metric that is
    aggregated by SUM across replicas: duplicated samples would otherwise be
    counted multiple times in both the score numerator and its sample count.

    Sample order is preserved (matching ``shuffle=False`` semantics). The split
    is near-even and contiguous: the first ``n % num_replicas`` ranks each take
    one extra sample.
    """

    def __init__(self, dataset, num_replicas, rank):
        self.num_replicas = num_replicas
        self.rank = rank
        n = len(dataset)
        base, remainder = divmod(n, num_replicas)
        start = rank * base + min(rank, remainder)
        count = base + (1 if rank < remainder else 0)
        self.indices = list(range(start, start + count))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        # Present for API parity with DistributedSampler (the trainer calls
        # set_epoch every epoch). Order is fixed (shuffle=False), so this is a
        # no-op; validation must be reproducible across epochs.
        return None


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
        self.world_size = get_world_size(required=True)
        self.world_rank = get_world_rank(required=True)
        self.local_rank = get_local_rank(required=True)

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
        self.ce_class_weights = None
        self.global_step = 0
        self.total_optimizer_steps = 0
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
        self.log.info(
            f"Datasets created with n_train={self.n_train}, n_val={self.n_val}"
        )

    def create_sampler(self):
        """Create DistributedSamplers for train and validation datasets."""
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_set,
            num_replicas=self.data_num_replicas,
            rank=self.data_replica_rank,
        )
        # Validation is sharded WITHOUT padding: the metric aggregation
        # sums each replica's dice and sample count across the data-parallel
        # group, so a padded (duplicated) sample would be double counted and
        # bias val_score. Contiguous uneven shards give every sample to
        # exactly one replica; the SUM all_reduce tolerates the uneven (and
        # possibly zero) per-rank counts.
        self.val_sampler = _UnpaddedDistributedSampler(
            self.val_set,
            num_replicas=self.data_num_replicas,
            rank=self.data_replica_rank,
        )

    def create_dataloaders(self):
        """Create dataloaders for training and validation."""
        self.create_dataset()
        self.create_sampler()

        num_workers = self.config.dataloader_num_workers
        loader_args = dict(
            batch_size=self.config.local_batch_size,
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
            self.val_set, sampler=self.val_sampler, drop_last=False, **loader_args
        )
        if len(self.val_loader) == 0:
            # With unpadded validation sharding a rank can legitimately hold
            # zero samples (fewer validation samples than data-parallel
            # replicas). The metric reduction sums dice and sample counts across
            # the data-parallel group, so an empty rank simply contributes zero
            # and the global mean stays correct. Only the non-distributed /
            # single-replica case -- where an empty loader means there is no
            # validation data at all -- is a real error.
            if self.config.dist and self.data_num_replicas > 1:
                self.log.warning(
                    "Validation DataLoader has zero batches on this rank "
                    f"(n_val={self.n_val}, data_num_replicas={self.data_num_replicas}); "
                    "this rank contributes nothing to the reduced validation metric."
                )
            else:
                raise ValueError(
                    "Validation DataLoader has zero batches. "
                    f"n_val={self.n_val}, local_batch_size={self.config.local_batch_size}, "
                    f"data_num_replicas={self.data_num_replicas}. "
                    "Reduce local_batch_size or adjust validation sharding."
                )

    def setup_training_components(self):
        """Set up the optimizer, scheduler, gradient scaler, and loss function."""
        # Set up optimizer
        if self.config.optimizer == "ADAM":
            self.log.info("Using ADAM optimizer.")
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.starting_learning_rate
            )
        elif self.config.optimizer == "SGD":
            self.log.info("Using SGD optimizer.")
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.config.starting_learning_rate
            )
        else:
            self.log.info("Using RMSprop optimizer.")
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config.starting_learning_rate,
                foreach=True,
            )

        # Set up learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.T_0,
            T_mult=self.config.T_mult,
            eta_min=self.config.min_learning_rate,
        )

        # Set up gradient scaler for AMP (Automatic Mixed Precision)
        # bfloat does not need grad scaler
        self.use_grad_scaler = (
            self.config.torch_amp and self.amp_dtype != torch.bfloat16
        )
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)

        # Set up loss function
        ce_class_weights = _compute_ce_class_weights(
            train_set=self.train_set,
            n_train=self.n_train,
            n_categories=self.config.n_categories,
            device=self.device,
            sample_fraction=self.config.ce_weight_sample_fraction,
            dist_enabled=True,
            world_rank=self.world_rank,
            log=self.log,
        )
        self.criterion = nn.CrossEntropyLoss(weight=ce_class_weights).to(self.device)
        self.ce_class_weights = self.criterion.weight

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
        return dice_scores[:, 1:].mean()

    def _current_learning_rate(self):
        if self.optimizer is None or not self.optimizer.param_groups:
            return self.config.starting_learning_rate
        return self.optimizer.param_groups[0]["lr"]

    def _data_parallel_group(self):
        """Process group over which to reduce epoch-level metrics.

        Metrics such as loss and dice are already reduced across the *spatial*
        mesh inside the sharded loss/dice kernels, so every rank within one
        data-parallel replica holds the same global-over-space value. Reducing
        those across all world ranks would therefore multiply each replica's
        contribution by its number of spatial shards. To get one contribution
        per replica we reduce only over the data-parallel ("ddp") mesh
        dimension when a parallel strategy is present. Without a parallel
        strategy there is no spatial sharding (one rank per replica), so the
        default world group is correct.
        """
        if self.ps is not None:
            return self.ps.device_mesh["ddp"].get_group()
        return None  # default: the world group

    def _all_reduce_data_parallel(self, tensor):
        """SUM-reduce ``tensor`` across the data-parallel replicas in place."""
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=self._data_parallel_group(),
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
            dist_enabled=True,
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
            # Load checkpoint via manager. An explicit restart must find a
            # checkpoint; a plain non-scratch launch may simply start fresh.
            self.start_epoch = self.checkpoint_manager.load_from_checkpoint(
                require_checkpoint=getattr(self.config, "restart", False)
            )

            # Restore extra metadata if needed (e.g. mask values)
            restored = self.checkpoint_manager.restored_extras
            if "train_mask_values" in restored:
                self.train_set.mask_values = restored["train_mask_values"]

            # Continue the optimizer-step count from where the checkpoint left
            # off; otherwise a resumed run restarts it at 0 and undercounts all
            # pre-resume work in the reported step total.
            if "global_step" in restored:
                self.global_step = restored["global_step"]

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
            "optimizer_steps",
            "total_optimizer_steps",
        ]
        if self.world_rank == 0:
            header_line = ",".join(headers) + "\n"
            if self.start_epoch == 1:
                # Fresh start (train_from_scratch, or a non-scratch launch that
                # found no checkpoint to resume): truncate any stale stats file
                # and write a single header. Appending here would leave a
                # second header mid-file, which the CSV reader parses as a row
                # of NaNs and corrupts the benchmark score.
                with open(self.outfile_path, "w", newline="") as outfile:
                    outfile.write(header_line)
            elif not os.path.exists(self.outfile_path):
                # Real resume (a checkpoint set start_epoch > 1) but the stats
                # file is gone: recreate it with a header so the epoch rows
                # appended below are not read as column names.
                with open(self.outfile_path, "w", newline="") as outfile:
                    outfile.write(header_line)

    def _truncate_stats_file(self, start_epoch, path=None):
        """
        Scans the stats file and truncates it at the first occurrence of
        an epoch >= start_epoch. This is O(1) memory and safe for large logs.
        """
        if path is None:
            path = self.outfile_path
        self.log.info(f"Truncating {path} to remove epochs >= {start_epoch}")

        try:
            # Open in read+update mode ('r+') to allow seeking and truncating
            with open(path, "r+") as f:
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
            self.log.warning(f"Failed to truncate stats file {path}: {e}")

    def _get_memsize(self, tensor, tensor_label: str, verbosity: int = 0):
        """Log size of tensor in memory"""

        if verbosity < 2:
            return
        tensor_memory_bytes = tensor[0].element_size() * tensor[0].nelement()
        tensor_memory_gb = tensor_memory_bytes / (1024**3)
        self.log.info(f"{tensor_label} size on GPU: {tensor_memory_gb:.2f} GB")

    def _optimizer_step_applied(self, scale_before_update):
        """Return whether the last GradScaler step actually updated parameters.

        When the scaler is enabled it skips optimizer.step() on inf/nan grads
        and backs the loss scale off in update(); a decreased scale therefore
        means the step was skipped. When the scaler is disabled the scale is a
        constant, so every step is applied.
        """
        if not self.use_grad_scaler:
            return True
        return self.grad_scaler.get_scale() >= scale_before_update

    def _run_training_batch(
        self,
        batch,
        *,
        log_prefix="",
        gather_mem_stats=False,
        log_peak_mem=False,
    ):
        """Run one training batch and return batch size, detached loss, and dice."""
        images, true_masks = batch["image"], batch["mask"]

        begin_code_region("image_to_device")
        images = images.to(
            device=self.device,
            dtype=VOLUME_TORCH_DTYPE,
            memory_format=torch.channels_last_3d,
            non_blocking=True,
        )
        true_masks = true_masks.to(
            device=self.device, dtype=torch.long, non_blocking=True
        ).contiguous()
        end_code_region("image_to_device")
        if gather_mem_stats:
            gather_and_print_mem(self.log, "after_batch_to_device")

        # Add a dummy channel dimension to get 5D [B, 1, D, H, W]
        true_masks = true_masks.unsqueeze(1)

        # Inputs are already loaded as local shards by the dataset. Without a
        # parallel strategy there is nothing to shard; use the tensors as-is.
        if self.ps is not None:
            images_dc = DCTensor.from_shard(images, self.ps)
            true_masks_dc = DCTensor.from_shard(true_masks, self.ps)
        else:
            images_dc = images
            true_masks_dc = true_masks
        del images, true_masks
        self._get_memsize(images_dc, "Sharded image", self.config.verbose)

        with torch.autocast(**self._autocast_kwargs()):
            if gather_mem_stats:
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                gather_and_print_mem(self.log, "pre_forward")
            begin_code_region("predict")
            self.log.debug(f"  {log_prefix}running forward pass")
            masks_pred_dc = self.model(images_dc)
            end_code_region("predict")
            if gather_mem_stats:
                gather_and_print_mem(self.log, "post_forward")
            self.log.debug(f"  {log_prefix}forward pass complete")

            # Extract the underlying PyTorch local tensors
            local_preds = masks_pred_dc
            local_labels_5d = true_masks_dc

            # Remove the dummy channel dimension so CE Loss is happy [B, D, H, W]
            local_labels = local_labels_5d.squeeze(1)
            if self.world_rank == 0:
                self.log.debug(f"  {log_prefix}Local Preds Shape: {local_preds.shape}")
                self.log.debug(
                    f"  {log_prefix}Local Labels Shape: {local_labels.shape}"
                )

            begin_code_region("calculate_loss")
            if self.device.type == "cuda":
                current_mem = torch.cuda.memory_allocated() / (1024**3)
                self.log.debug(
                    f"  {log_prefix}Calculating sharded loss. Mem: {current_mem:.2f} GB."
                )

            # Calculate CE and Dice loss in single precision for numerical stability.
            with torch.autocast(**self._autocast_kwargs(enabled=False)):
                loss_ce = compute_sharded_cross_entropy_loss(
                    local_preds,
                    local_labels,
                    self.spatial_mesh,
                    self.config.dc_num_shards,
                    self.amp_device_type,
                    self.ce_class_weights,
                )

                local_preds_softmax = F.softmax(local_preds.float(), dim=1)
                local_labels_one_hot = (
                    F.one_hot(local_labels, num_classes=self.config.n_categories + 1)
                    .permute(0, 4, 1, 2, 3)
                    .float()
                )
                dice_scores = compute_sharded_dice(
                    local_preds_softmax,
                    local_labels_one_hot,
                    self.spatial_mesh,
                )
                batch_dice_score = self._foreground_dice_mean(dice_scores)

                # Sum global CE Loss and Dice loss
                loss = loss_ce + (1.0 - batch_dice_score)
            end_code_region("calculate_loss")

        self.log.debug(
            f"  {log_prefix}loss calculation complete. Proceeding to backward pass"
        )
        if gather_mem_stats:
            gather_and_print_mem(self.log, "pre_backward")
        begin_code_region("backward")
        self.grad_scaler.scale(loss).backward()
        end_code_region("backward")
        if gather_mem_stats:
            gather_and_print_mem(self.log, "post_backward")

        begin_code_region("step_and_update")
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.log.debug(f"  {log_prefix}backward pass complete. Stepping optimizer")
        # Record the scale before update() so we can detect a skipped step: on
        # inf/nan gradients GradScaler.step() silently skips optimizer.step()
        # and update() backs the scale off, leaving parameters unchanged.
        scale_before_update = self.grad_scaler.get_scale()
        self.grad_scaler.step(self.optimizer)
        if gather_mem_stats:
            gather_and_print_mem(self.log, "after_optim_step")
        self.grad_scaler.update()
        self._last_step_applied = self._optimizer_step_applied(scale_before_update)
        self.optimizer.zero_grad(set_to_none=False)
        end_code_region("step_and_update")

        batch_size = images_dc.shape[0]
        detached_loss = loss.detach()

        # Free memory aggressively
        del images_dc, true_masks_dc, masks_pred_dc
        del local_preds, local_labels, local_preds_softmax, local_labels_one_hot
        del loss_ce, loss

        if log_peak_mem and self.world_rank == 0 and self.device.type == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            self.log.debug(
                f"[MEM-PEAK] Peak alloc: {peak_alloc:.2f} GiB | Peak reserved: {peak_reserved:.2f} GiB",
            )

        return batch_size, detached_loss, batch_dice_score

    def _sync_gather_minibatch_timer(self, minibatch_events):
        minibatch_events[-1][1].synchronize()
        local_minibatch_times = torch.tensor(
            [
                start_event.elapsed_time(end_event) / 1000.0
                for start_event, end_event in minibatch_events
            ],
            device=self.device,
        )
        gathered_minibatch_times = [
            torch.empty_like(local_minibatch_times) for _ in range(self.world_size)
        ]
        torch.distributed.all_gather(gathered_minibatch_times, local_minibatch_times)
        minibatch_times = torch.stack(gathered_minibatch_times)
        minibatch_times = torch.max(minibatch_times, dim=0).values
        minibatch_time_s = statistics.median(minibatch_times.cpu().tolist())
        return minibatch_time_s

    def warmup(self):
        """Run warmup iterations before the main training loop."""
        warmup_batches = self.config.warmup_batches
        if warmup_batches <= 0:
            return

        self.train_loader.sampler.set_epoch(0)

        start_warmup = time.time()
        max_batches = min(warmup_batches, len(self.train_loader))
        max_val_batches = min(warmup_batches, len(self.val_loader))
        self.log.info(
            f"Running {max_batches} training warmup batch(es) and {max_val_batches} validation warmup batch(es) per rank"
        )
        snapshot = self.checkpoint_manager.snapshot_training_state()

        # Match the main training path as closely as possible, but roll back all
        # mutable state so warmup does not affect convergence.
        self.model.train()
        self.optimizer.zero_grad(set_to_none=False)

        try:
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx >= max_batches:
                    break

                self._run_training_batch(
                    batch,
                    log_prefix="warmup: ",
                    log_peak_mem=True,
                )
                batch_t_end = time.time()
                self.log.debug(
                    f"  warmup: batch {batch_idx} completed in {batch_t_end - start_warmup} seconds"
                )

            self.val_loader.sampler.set_epoch(0)

            if max_val_batches > 0:
                self.log.debug("  warmup: running validation warmup pass")
                evaluate(
                    self.model,
                    self.val_loader,
                    self.device,
                    self.config.torch_amp,
                    False,
                    self.criterion,
                    self.config.n_categories,
                    self.config._parallel_strategy,
                    max_batches=max_val_batches,
                    log=self.log,
                )
        finally:
            self.checkpoint_manager.restore_training_state(snapshot)

        torch.distributed.barrier()
        self.log.info(f"Done warmup. Took {int(time.time() - start_warmup)}s")

    def train(self):
        """
        Execute model training
        """

        epoch = self.start_epoch
        dice_score_train = 0
        epoch_minibatch_times_s = []
        # Track the last epoch checkpointed inside the loop so the final-save
        # decision below is identical on every rank. The in-loop checkpoint
        # condition depends only on the epoch number and the interval, so this
        # stays consistent across ranks -- unlike the checkpoint manager's
        # last_saved_epoch, which it records on rank 0 only. Reading that on the
        # other ranks would make them disagree about whether to call
        # save_checkpoint on exit, deadlocking its internal collective.
        last_checkpoint_epoch = None
        with open(self.outfile_path, "a", newline="") as outfile:
            start = time.time()
            while dice_score_train < self.config.target_dice:
                if self.config.epochs != -1 and epoch > self.config.epochs:
                    self.log.warning(
                        "Maximum epochs reached '%s'. Concluding training early "
                        "(may have not converged).",
                        self.config.epochs,
                    )
                    break

                # Timer and tracking variables. The loss/dice accumulators hold
                # sample-weighted sums (per-batch mean times the batch's sample
                # count) so a ragged final batch is not overweighted; dividing by
                # the total sample count below yields the true per-sample mean.
                epoch_start_time = time.time()
                train_dice_total = 0
                epoch_loss = 0  # Sample-weighted sum of per-batch losses
                epoch_optimizer_steps = 0
                train_sample_count = 0
                minibatch_time_s = None
                minibatch_events = []

                # Set necessary modes/states
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
                    for batch_idx, batch in enumerate(self.train_loader):
                        # We don't want to time partial batches, i.e. last batch (time will be lower than expected).
                        # CUDA events need a CUDA device; skip timing on CPU.
                        time_minibatch = self.device.type == "cuda" and (
                            batch_idx
                            < len(self.train_sampler) // self.config.local_batch_size
                        )
                        if time_minibatch:
                            minibatch_start_event = torch.cuda.Event(enable_timing=True)
                            minibatch_end_event = torch.cuda.Event(enable_timing=True)
                            minibatch_start_event.record()
                            minibatch_events.append(
                                (minibatch_start_event, minibatch_end_event)
                            )
                        begin_code_region("minibatch_time")
                        begin_code_region("run_training_batch")
                        batch_size, batch_loss, batch_dice_score = (
                            self._run_training_batch(
                                batch,
                                gather_mem_stats=True,
                            )
                        )
                        # Weight the (spatial-mesh-reduced) per-batch dice by
                        # the batch's sample count so a ragged final batch does
                        # not skew the epoch mean.
                        train_dice_total += batch_dice_score * batch_size
                        end_code_region("run_training_batch")

                        # Update the loss
                        begin_code_region("update_loss")
                        pbar.update(batch_size)
                        # Only count steps the optimizer actually applied; a
                        # GradScaler that skipped this step on inf/nan grads
                        # left the parameters unchanged.
                        if getattr(self, "_last_step_applied", True):
                            self.global_step += 1
                            epoch_optimizer_steps += 1
                        # Stay on GPU; accumulate the sample-weighted loss sum.
                        epoch_loss += batch_loss * batch_size
                        train_sample_count += batch_size
                        end_code_region("update_loss")
                        end_code_region("minibatch_time")

                        if time_minibatch:
                            minibatch_end_event.record()
                    end_code_region("batch_loop")

                self.total_optimizer_steps += epoch_optimizer_steps

                # Reduce the sample-weighted train loss/dice sums and the sample
                # count across the data-parallel replicas so the logged metrics
                # reflect the whole epoch's data, not just this replica's shard.
                # Each replica's per-batch values are already reduced over the
                # spatial mesh, so reducing over the "ddp" dimension only (see
                # _all_reduce_data_parallel) counts every replica exactly once.
                train_loss_sum = (
                    float(epoch_loss.item())
                    if torch.is_tensor(epoch_loss)
                    else float(epoch_loss)
                )
                train_dice_sum = (
                    float(train_dice_total.item())
                    if torch.is_tensor(train_dice_total)
                    else float(train_dice_total)
                )
                train_info = torch.tensor(
                    [train_loss_sum, train_dice_sum, float(train_sample_count)],
                    dtype=VOLUME_TORCH_DTYPE,
                )
                train_info = train_info.to(device=self.device)
                self._all_reduce_data_parallel(train_info)
                global_train_samples = max(train_info[2].item(), 1)
                # epoch_loss column keeps the (reduced) sample-weighted loss sum;
                # overall_loss is the global per-sample mean.
                epoch_loss_total = train_info[0].item()
                overall_loss = epoch_loss_total / global_train_samples
                train_dice = float(train_info[1].item() / global_train_samples)

                #
                # Evaluate model on validation set, update LR if necessary
                #
                (
                    dice_sum,
                    val_loss_epoch,
                    val_loss_avg,
                    numbatch,
                    numsamples,
                ) = evaluate(
                    self.model,
                    self.val_loader,
                    self.device,
                    self.config.torch_amp,
                    self.world_rank == 0,
                    self.criterion,
                    self.config.n_categories,
                    self.config._parallel_strategy,
                    log=self.log,
                )
                # Reduce the validation dice sum, sample-weighted loss sum, and
                # sample count together across the data-parallel replicas. Like
                # the train metrics these are already spatial-mesh-reduced, so
                # the "ddp"-only reduction counts each replica once. val_loss is
                # bundled here (rather than inside evaluate) so best-checkpoint
                # selection and the CSV use the global sample-weighted mean, not
                # rank 0's replica-local value.
                val_info = torch.tensor(
                    [dice_sum, val_loss_epoch, numsamples],
                    dtype=VOLUME_TORCH_DTYPE,
                )
                val_info = val_info.to(device=self.device)
                self._all_reduce_data_parallel(val_info)
                global_val_samples = max(val_info[2].item(), 1)
                val_score = val_info[0].item() / global_val_samples
                # Reduced sample-weighted total and per-sample mean val loss.
                val_loss_epoch = val_info[1].item()
                val_loss_avg = val_loss_epoch / global_val_samples
                if not self.config.disable_scheduler:
                    self.scheduler.step()
                else:
                    self.log.debug("scheduler disabled, no LR update this step")

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time

                # Sync for batch time happens once after epoch is already done (low overhead)
                if len(minibatch_events) > 0:
                    minibatch_time_s = self._sync_gather_minibatch_timer(
                        minibatch_events
                    )
                    epoch_minibatch_times_s.append(minibatch_time_s)

                #
                # Write out data for this epoch to train stats csv
                #
                self.log.info(
                    f" epoch {epoch} | train_loss={overall_loss:.6f} | val_loss={val_loss_avg:.6f} | train_dice_score {train_dice:.6f} | val_dice_score {val_score:.6f} | lr {self._current_learning_rate():.8f} | optimizer_steps {epoch_optimizer_steps} | total_optimizer_steps {self.total_optimizer_steps}"
                )
                self.log.debug(f" writing to csv at {self.outfile_path}")
                if self.world_rank == 0:
                    outfile.write(
                        ",".join(
                            [
                                str(epoch),
                                str(epoch_loss_total),
                                str(overall_loss),
                                str(val_loss_epoch),
                                str(val_loss_avg),
                                str(train_dice),
                                str(val_score),
                                str(epoch_duration),
                                str(epoch_optimizer_steps),
                                str(self.total_optimizer_steps),
                            ]
                        )
                        + "\n"
                    )
                    outfile.flush()
                    # minibatch_time_s stays None when every batch was a
                    # partial (untimed) batch; skip the fragment rather than
                    # crash formatting None.
                    minibatch_msg = (
                        f" Median of minibatch times: {minibatch_time_s:.6f} seconds."
                        if minibatch_time_s is not None
                        else ""
                    )
                    self.log.info(
                        "Epoch %s completed in %.6f seconds. Total train time so "
                        "far: %.6f seconds.%s Optimizer steps this epoch: %s. "
                        "Total optimizer steps: %s.",
                        epoch,
                        epoch_duration,
                        time.time() - start,
                        minibatch_msg,
                        epoch_optimizer_steps,
                        self.total_optimizer_steps,
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
                    extras = {
                        "train_mask_values": self.train_set.mask_values,
                        "global_step": self.global_step,
                    }
                    self.checkpoint_manager.save_checkpoint(epoch, val_loss_avg, extras)
                    last_checkpoint_epoch = epoch

                end_code_region("checkpoint")

                dice_score_train = val_score
                epoch += 1

                # This check must exist otherwise the condition dice_score_train < self.config.target_dice will evaluate to False and incorrectly exit the training
                if math.isnan(dice_score_train):
                    raise ValueError(
                        "Invalid value (NaN) encountered in dice score computation"
                    )

        completed_epochs = epoch - 1

        # Save a final checkpoint when the run exits (convergence or max epochs)
        # at an epoch that was not a checkpoint interval, so the converged
        # weights that produced the reported metrics are not lost. Skipped when
        # checkpointing is disabled, when no epoch completed, or when the last
        # completed epoch was already checkpointed inside the loop.
        if (
            self.config.checkpoint_interval > 0
            and completed_epochs >= 1
            and last_checkpoint_epoch != completed_epochs
        ):
            extras = {
                "train_mask_values": self.train_set.mask_values,
                "global_step": self.global_step,
            }
            self.checkpoint_manager.save_checkpoint(
                completed_epochs, val_loss_avg, extras
            )

        if epoch_minibatch_times_s:
            minibatch_time_s = statistics.median(epoch_minibatch_times_s)
            adiak_value("minibatch_time_s", minibatch_time_s)
            if self.world_rank == 0:
                self.log.info(
                    f"Median of epoch minibatch time medians: {minibatch_time_s:.6f} seconds."
                )
        adiak_value("final_epochs", completed_epochs)
        adiak_value("total_optimizer_steps", self.total_optimizer_steps)
