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
import socket
import time
from argparse import Namespace

import numpy as np
import psutil
import torch
import torch.distributed as dist
from distconv import DistConvDDP, ParallelStrategy
from torch.distributed.tensor import Replicate, Shard

from ScaFFold.datagen.get_dataset import get_dataset
from ScaFFold.unet import UNet
from ScaFFold.utils.distributed import (
    get_device,
    get_local_rank,
    get_local_size,
    get_world_rank,
    get_world_size,
    initialize_dist,
)
from ScaFFold.utils.perf_measure import (
    adiak_value,
    annotate,
    begin_code_region,
    end_code_region,
    get_torch_context,
)
from ScaFFold.utils.trainer import PyTorchTrainer
from ScaFFold.utils.utils import set_seeds, setup_mpi_logger
from ScaFFold.viz import standard_viz


def check_resource_utilization(log, rank, world_size):
    """Check that we are properly utilizing resources"""

    # CPU
    log.debug(f"rank {rank}, world_size {world_size}")
    log.debug(f"Number of Physical Cores: {psutil.cpu_count(logical=False)}")
    log.debug(f"Number of Logical Cores: {psutil.cpu_count(logical=True)}")
    # GPU
    if torch.cuda.is_available() and rank == 0:
        # Get the number of GPUs available
        num_gpus = torch.cuda.device_count()
        log.debug(f"Number of GPUs available: {num_gpus}")

    if torch.cuda.is_available():
        this_gpu = torch.cuda.current_device()
        log.debug(f"rank {rank}: Details of GPU {this_gpu}:")
        log.debug(f"  Name: {torch.cuda.get_device_name(this_gpu)}")
        log.debug(f"  Capability: {torch.cuda.get_device_capability(this_gpu)}")
        log.debug(
            f"  Memory Allocated: {torch.cuda.memory_allocated(this_gpu) / 1024**2:.2f} MB"
        )
        log.debug(
            f"  Memory Cached: {torch.cuda.memory_reserved(this_gpu) / 1024**2:.2f} MB"
        )
        log.debug(
            f"  Total Memory: {torch.cuda.get_device_properties(this_gpu).total_memory / 1024**2:.2f} MB"
        )
        log.debug(f"  Device Properties: {torch.cuda.get_device_properties(this_gpu)}")


@annotate()
def main(kwargs_dict: dict = {}):
    #
    # Setup
    #

    config = Namespace(**kwargs_dict)

    # Set logging options
    log = setup_mpi_logger(__file__, config.verbose)

    # Set random seeds for reproducibility
    set_seeds(config.seed)
    log.debug(f"random seeds set to {config.seed}")

    # Get MPI information
    rank = get_world_rank(required=True)
    world_size = get_world_size(required=True)

    # Optionally enable additional determinism settings
    if config.more_determinism:
        log.info(
            "more_determinism TRUE -- enabling additional determinism settings to improve training reproducibility"
        )
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Default
        torch.backends.cudnn.benchmark = True

    # Initialize DDP. ScaFFold always runs the benchmark as a distributed job;
    # a one-rank launch is the supported singleton case.
    begin_code_region("init_ddp")
    if not dist.is_initialized():
        log.info("Initializing distributed process group...")
        initialize_dist(log, rendezvous="env")
    else:
        log.info("Distributed process group already initialized by launcher.")
    end_code_region("init_ddp")

    # More useful info
    log.debug(f"Host={socket.gethostname()} PID={os.getpid()}")
    log.debug(f"PyTorch {torch.__version__}, CUDA/ROCm {torch.version.cuda}")
    log.debug(
        f"Backend={dist.get_backend()}, world_size={world_size}, rank={rank}, local_rank={get_local_rank()}"
    )
    log.info(f"rank={rank}, world_size={world_size}")

    # Generate or retrieve dataset
    begin_code_region("get_dataset")
    dataset_dir = get_dataset(
        config, require_commit=config.dataset_reuse_enforce_commit_id
    )
    config.dataset_dir = dataset_dir
    end_code_region("get_dataset")

    # Initialize model
    begin_code_region("init_model")
    log.info(
        f"DistConv num_shards={config.dc_num_shards}, shard_dim={config.dc_shard_dims}"
    )
    device = get_device()
    log.info(f"Using device: {device}")
    model = UNet(
        n_channels=3,
        n_classes=config.n_categories + 1,
        trilinear=False,
        layers=config.unet_layers,
        group_norm_groups=config.group_norm_groups,
    )
    # DDP + DistConv setup
    # Ensure world_size is divisible by total distconv shards
    total_distconv_shards = math.prod(config.dc_num_shards)
    if world_size % total_distconv_shards != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by total number of "
            f"distconv shards={total_distconv_shards}"
        )

    ps = ParallelStrategy(
        num_shards=config.dc_num_shards,
        shard_dim=config.dc_shard_dims,
        device_type=device.type,
    )

    model = model.to(device, memory_format=torch.channels_last_3d)
    ddp_device_ids = [device.index] if device.type == "cuda" else None
    ddp_output_device = device.index if device.type == "cuda" else None
    # Wrap with DistConvDDP that corrects gradient scaling for dc submesh
    model = DistConvDDP(
        model,
        parallel_strategy=ps,
        device_ids=ddp_device_ids,
        output_device=ddp_output_device,
    )
    # Store ps for use in the training loop
    config._parallel_strategy = ps
    end_code_region("init_model")

    check_resource_utilization(log, rank, world_size)

    #
    # Initialize trainer
    #
    if config.framework == "torch":
        # Optionally enable additional determinism settings
        if config.more_determinism:
            log.info(
                "Enabling additional determinism settings to improve training reproducibility"
            )
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        trainer = PyTorchTrainer(model, config, device, log)
        trainer.ps = ps
        trainer.spatial_mesh = ps.device_mesh[ps.distconv_dim_names]
        num_spatial_dims = len(ps.shard_dim)
        trainer.ddp_placements = [Shard(0)] + [Replicate()] * num_spatial_dims
        total_shards = math.prod(config.dc_num_shards)
        global_batch_size = config.batch_size * (world_size // total_shards)
        config.global_batch_size = global_batch_size
        ddp_ranks = world_size // total_shards
        adiak_value("global_batch_size", global_batch_size)
        adiak_value("ddp_ranks", ddp_ranks)
        adiak_value("total_shards", total_shards)
        adiak_value("num_spatial_dims", num_spatial_dims)
        if rank == 0:
            log.info(
                f"Effective global batch size = {global_batch_size} "
                f"(batch_size={config.batch_size} * "
                f"(world_size={world_size} / prod(dc_num_shards)={total_shards}))"
            )
            log.info(
                f"DDP ranks = {ddp_ranks} "
                f"world_size={world_size} // prod(dc_num_shards)={total_shards}"
            )
        too_small_splits = []
        if global_batch_size > trainer.n_train:
            too_small_splits.append(f"training n_train={trainer.n_train}")
        if global_batch_size > trainer.n_val:
            too_small_splits.append(f"validation n_val={trainer.n_val}")
        if too_small_splits:
            raise ValueError(
                "Effective global batch size exceeds available samples: "
                f"global_batch_size={global_batch_size}, "
                f"{', '.join(too_small_splits)}, "
                f"batch_size={config.batch_size}, world_size={world_size}, "
                f"dc_num_shards={config.dc_num_shards}"
            )

    else:
        raise RuntimeError(
            "Invalid framework specified. Currently [torch] is the supported framework."
        )

    #
    # Run the training
    #
    ranks_per_node = get_local_size()
    prof_ctx, TORCH_PERF_LOCAL = get_torch_context(ranks_per_node, rank)
    with prof_ctx as prof:
        begin_code_region("cleanup_or_resume")
        trainer.cleanup_or_resume()
        end_code_region("cleanup_or_resume")
        begin_code_region("warmup")
        trainer.warmup()
        end_code_region("warmup")
        begin_code_region("train")
        trainer.train()
        end_code_region("train")
    if TORCH_PERF_LOCAL:
        hostname = socket.gethostname()
        tracename = f"torch-{hostname}-r{rank}-N{world_size // ranks_per_node}-n{world_size}-ps{config.problem_scale}-e{config.epochs}-nipf{config.n_instances_used_per_fractal}-{int(time.time())}.json"
        prof.export_chrome_trace(tracename)
        log.info("Wrote PyTorch trace '%s'", tracename)

    #
    # Calculate benchmark score
    #
    if rank == 0:
        outfile_path = trainer.outfile_path
        train_data = np.genfromtxt(outfile_path, dtype=float, delimiter=",", names=True)
        total_train_time = train_data["epoch_duration"].sum()
        if "total_optimizer_steps" in train_data.dtype.names:
            optimizer_steps = np.atleast_1d(train_data["total_optimizer_steps"])
            total_optimizer_steps = int(optimizer_steps[-1])
        elif "optimizer_steps" in train_data.dtype.names:
            total_optimizer_steps = int(
                np.atleast_1d(train_data["optimizer_steps"]).sum()
            )
        else:
            total_optimizer_steps = int(getattr(trainer, "total_optimizer_steps", 0))
        adiak_value("total_optimizer_steps", total_optimizer_steps)
        epochs = np.atleast_1d(train_data["epoch"])
        total_epochs = int(epochs[-1])
        if config.epochs == -1:
            fom = 1.0 / total_train_time
            adiak_value("FOM", fom)
            log.info(
                f"FOM = {fom} (1 / total_train_time={total_train_time:.6f} seconds). "
                f"This FOM is specific to problem_scale={config.problem_scale}, "
                f"target_dice={config.target_dice}, seed={config.seed}."
            )
            extra_msg = f"Trained to >= {config.target_dice} validation dice score in {total_train_time:.2f} seconds, {total_epochs} epochs, {total_optimizer_steps} optimizer steps."
        else:
            extra_msg = f"Completed in {total_train_time:.2f} seconds, {total_epochs} epochs, {total_optimizer_steps} optimizer steps."

        log.info(
            f"Benchmark run at scale {config.problem_scale} complete. \n{extra_msg}"
        )

        # Generate plots
        log.info("Generating figures on rank 0...")
        begin_code_region("generate_figures")
        standard_viz.main(config)
        end_code_region("generate_figures")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return 0
