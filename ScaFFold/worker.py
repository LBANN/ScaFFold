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

import os
import socket
import sys
import time
from argparse import Namespace

import numpy as np
import psutil
import torch
import torch.distributed as dist
import yaml
from distconv import DCTensor, DistConvDDP, ParallelStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from ScaFFold.datagen.get_dataset import get_dataset
from ScaFFold.unet import UNet
from ScaFFold.utils.distributed import (
    get_device,
    get_job_id,
    get_local_rank,
    get_local_size,
    get_world_rank,
    get_world_size,
    initialize_dist,
)
from ScaFFold.utils.perf_measure import (
    annotate,
    begin_code_region,
    end_code_region,
    get_torch_context,
)
from ScaFFold.utils.trainer import PyTorchTrainer
from ScaFFold.utils.utils import set_seeds, setup_mpi_logger
from ScaFFold.viz import standard_viz

if hasattr(os, "sched_getaffinity"):
    _orig_affinity = os.sched_getaffinity(0)
else:
    _orig_affinity = None


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


def override_config(config) -> None:
    """Override base run config if additional configs are provided."""
    if "--config" in sys.argv:
        config_idx = 1  # Start at 1 to skip the base run config
        while True:
            try:
                config_idx = sys.argv.index("--config", config_idx) + 1
            except ValueError:
                break
            config_file = sys.argv[config_idx]
            if not os.path.isfile(config_file):
                raise ValueError(f"Additional config file {config_file} does not exist")
            with open(config_file) as f:
                override_config = yaml.full_load(f)
            for k, v in override_config.items():
                if not hasattr(config, k):
                    raise ValueError(f"Unknown configuration option {k}={v}")
                setattr(config, k, v)


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
    rank = get_world_rank(required=config.dist)
    world_size = get_world_size(required=config.dist)

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

    # Initialize DDP
    begin_code_region("init_ddp")
    if config.dist:
        if not dist.is_initialized():
            log.info("Initializing distributed process group...")
            initialize_dist(rendezvous="env")
        else:
            log.info("Distributed process group already initialized by launcher.")
    end_code_region("init_ddp")

    # More useful info
    log.debug(f"Host={socket.gethostname()} PID={os.getpid()}")
    log.debug(f"PyTorch {torch.__version__}, CUDA/ROCm {torch.version.cuda}")
    log.debug(
        f"Backend={dist.get_backend()}, world_size={world_size}, rank={rank}, local_rank={get_local_rank()}"
    )
    log.info(f"rank={rank}, world_size={world_size} test")

    # Generate or retrieve dataset
    begin_code_region("get_dataset")
    dataset_dir = get_dataset(
        config, require_commit=config.dataset_reuse_enforce_commit_id
    )
    config.dataset_dir = dataset_dir
    end_code_region("get_dataset")

    # Initialize model
    begin_code_region("init_model")
    config.dc_num_shards = getattr(config, "dc_num_shards", config.num_shards)
    config.dc_shard_dim = getattr(config, "dc_shard_dim", config.shard_dim)
    log.info(
        f"DistConv num_shards={config.dc_num_shards}, shard_dim={config.dc_shard_dim}"
    )
    device = get_device()
    log.info(f"Using device: {device}")
    model = UNet(
        n_channels=3,
        n_classes=config.n_categories + 1,
        trilinear=False,
        layers=config.unet_layers,
    )
    if config.dist:
        # DDP + DistConv setup
        # Ensure world_size is divisible by dc_num_shards
        assert dist.get_world_size() % config.dc_num_shards == 0, (
            f"world_size={dist.get_world_size()} must be divisible by dc_num_shards={config.dc_num_shards}"
        )
        # Select which full-tensor dim to shard: 2 + dc_shard_dim
        shard_dim = 2 + int(config.dc_shard_dim)
        ps = ParallelStrategy(
            num_shards=int(config.dc_num_shards),
            shard_dim=shard_dim,
            device_type=device.type,
        )
        model = model.to(device).to(memory_format=torch.contiguous_format)
        # Wrap with DistConvDDP that corrects gradient scaling for dc submesh
        model = DistConvDDP(
            model,
            parallel_strategy=ps,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
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
            print(
                "Enabling additional determinism settings to improve training reproducibility"
            )
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        trainer = PyTorchTrainer(model, config, device, log)
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
        begin_code_region("train")
        trainer.train()
        end_code_region("train")
    if TORCH_PERF_LOCAL:
        hostname = socket.gethostname()
        tracename = f"torch-{hostname}-r{rank}-N{world_size // ranks_per_node}-n{world_size}-ps{config.problem_scale}-e{config.epochs}-nipf{config.n_instances_used_per_fractal}-{int(time.time())}.json"
        prof.export_chrome_trace(tracename)
        print(f"Wrote PyTorch trace '{tracename}'")

    #
    # Calculate benchmark score
    #
    outfile_path = trainer.outfile_path
    train_data = np.genfromtxt(outfile_path, dtype=float, delimiter=",", names=True)
    total_train_time = train_data["epoch_duration"].sum()
    epochs = np.atleast_1d(train_data["epoch"])
    total_epochs = int(epochs[-1])
    log.info(
        f"Benchmark run at scale {config.problem_scale} complete. \n\
        Trained to >= 0.95 validation dice score in {total_train_time:.2f} seconds, {total_epochs} epochs."
    )

    #
    # Generate plots
    #
    if rank == 0:
        log.info(f"Generating figures on rank 0...")
        begin_code_region("generate_figures")
        standard_viz.main(config)
        end_code_region("generate_figures")

    dist.barrier()
    dist.destroy_process_group()

    return 0
