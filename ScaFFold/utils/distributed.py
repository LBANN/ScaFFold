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
import os.path
import socket
import time
from typing import Literal, Optional

import torch
import torch.distributed


def get_num_gpus() -> int:
    """Return the number of GPUs on this node."""
    return torch.cuda.device_count()


def get_local_rank(required: bool = False) -> int:
    """Return the local MPI rank."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "MV2_COMM_WORLD_LOCAL_RANK" in os.environ:
        return int(os.environ["MV2_COMM_WORLD_LOCAL_RANK"])
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    if "FLUX_TASK_LOCAL_ID" in os.environ:
        return int(os.environ["FLUX_TASK_LOCAL_ID"])
    if required:
        raise RuntimeError("Could not get local rank")
    return 0


def get_local_size(required: bool = False) -> int:
    """Return the number of local MPI ranks."""
    if "MV2_COMM_WORLD_LOCAL_SIZE" in os.environ:
        return int(os.environ["MV2_COMM_WORLD_LOCAL_SIZE"])
    if "OMPI_COMM_WORLD_LOCAL_SIZE" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
    if "SLURM_NNODES" in os.environ and "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) // int(os.environ["SLURM_NNODES"])
    # Flux does not have an env variable for this, so we assume an
    # even distribution.
    if "FLUX_JOB_SIZE" in os.environ and "FLUX_JOB_NNODES" in os.environ:
        return int(os.environ["FLUX_JOB_SIZE"]) // int(os.environ["FLUX_JOB_NNODES"])
    if required:
        raise RuntimeError("Could not get local size")
    return 1


def get_world_rank(required: bool = False) -> int:
    """Return the global MPI rank.."""
    if "MV2_COMM_WORLD_RANK" in os.environ:
        return int(os.environ["MV2_COMM_WORLD_RANK"])
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    if "FLUX_TASK_RANK" in os.environ:
        return int(os.environ["FLUX_TASK_RANK"])
    if required:
        raise RuntimeError("Could not get world rank")
    return 0


def get_world_size(required: bool = False) -> int:
    """Return the number of MPI ranks."""
    if "MV2_COMM_WORLD_SIZE" in os.environ:
        return int(os.environ["MV2_COMM_WORLD_SIZE"])
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    if "FLUX_JOB_SIZE" in os.environ:
        return int(os.environ["FLUX_JOB_SIZE"])
    if required:
        raise RuntimeError("Could not get world size")
    return 1


def force_cuda_visible_devices(force: bool = False) -> None:
    """Set CUDA_VISIBLE_DEVICES.

    This seems to help avoid PyTorch or something else from touching
    other GPUs.

    """
    print("force_cuda_visible_devices is deprecated. Skipping...")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.init()

        num_devices = torch.cuda.device_count()
        local_rank = get_local_rank()

        if num_devices == 1:
            # Case A: Flux/Slurm Masking is active.
            # We have 1 GPU visible. It is ALWAYS at index 0.
            target_device_index = 0
        else:
            # Case B: No masking
            # We see all GPUs, so we pick the one matching our rank.
            target_device_index = local_rank

        # Verify we aren't asking for an impossible device
        if target_device_index >= num_devices:
            raise RuntimeError(
                f"Rank {local_rank} requesting device index {target_device_index}, "
                f"but only {num_devices} devices are visible/available."
            )

        device = torch.device(f"cuda:{target_device_index}")
        torch.cuda.set_device(device)
        return device
    else:
        return torch.device("cpu")


def get_job_id() -> Optional[str]:
    """Return a generated job ID if possible."""
    if "SLURM_JOBID" in os.environ:
        return os.environ["SLURM_JOBID"]
    if "LSB_JOBID" in os.environ:
        return os.environ["LSB_JOBID"]
    if "FLUX_JOB_ID" in os.environ:
        return os.environ["FLUX_JOB_ID"]
    return None


def initialize_dist(
    init_file: Optional[str] = None, rendezvous: Literal["env", "tcp", "file"] = "env"
) -> None:
    """Initialize the PyTorch distributed backend and set up NCCL."""

    if rendezvous == "env":
        init_method = "env://"
    elif rendezvous == "tcp":
        if init_file is None:
            raise ValueError("init_file must be provided for tcp rendezvous")

        init_file = os.path.abspath(init_file)
        init_method = None
        if get_world_rank() == 0:
            # Check whether the init file exists already, as this can break things.
            if os.path.exists(init_file):
                raise RuntimeError(
                    f"Init file {init_file} exists at startup. This can break things"
                )
            # Get an IP and port to use.
            ip = socket.gethostbyname(socket.gethostname())
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", 0))  # Get a free port provided by the host.
            port = s.getsockname()[1]
            init_method = f"tcp://{ip}:{port}"
            with open(init_file, "w") as f:
                f.write(init_method)
        else:
            while not os.path.exists(init_file):
                time.sleep(1)
            with open(init_file, "r") as f:
                init_method = f.read()
    elif rendezvous == "file":
        if init_file is None:
            raise ValueError("init_file must be provided for file rendezvous")
        init_file = os.path.abspath(init_file)
        init_method = f"file://{init_file}"
    else:
        raise ValueError(f'Unrecognized scheme "{rendezvous}"')

    print(
        f"distributed.py: rank {get_world_rank()} / {get_world_size()} calling init_process_group()"
    )

    # Initialize
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=get_world_rank(),
        world_size=get_world_size(),
    )

    torch.distributed.barrier()

    # Only clean up file if we actually used a file-based method
    if (
        rendezvous in ["tcp", "file"]
        and init_file
        and get_world_rank() == 0
        and os.path.exists(init_file)
    ):
        os.unlink(init_file)
