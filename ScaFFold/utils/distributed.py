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

import logging
import os
import os.path
import socket
import time
from typing import Literal, Optional

import torch
import torch.distributed

logger = logging.getLogger(__name__)


def _env_int(name: str) -> Optional[int]:
    """Return the launcher variable ``name`` as an int, or None if unusable.

    Launcher variables are not always what they claim to be. A site wrapper
    that exports ``WORLD_SIZE=`` (empty, e.g. from an unset shell variable) or
    a placeholder like ``auto`` is common enough, and a bare ``int()`` turned
    it into a ``ValueError`` raised from the first of these helpers anything
    called -- killing every ``scaffold`` invocation, ``--help`` included, with
    a traceback that named neither the variable nor a remedy.

    An unusable value is treated as absent so the next source in the priority
    order is consulted (ultimately the MPI communicator, or the documented
    default). A non-empty value that is not an integer is warned about, because
    unlike an empty one it looks deliberate and the fallback may not be what
    its author intended. ``create_restart_script._sniff_launch_shape`` skips
    empty values for the same reason.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning(
            "Ignoring launcher variable %s=%r: not an integer. Falling back to "
            "the next source for the job shape.",
            name,
            value,
        )
        return None


def _first_env_int(names) -> Optional[int]:
    """Return the first usable integer among ``names``, in priority order."""
    for name in names:
        value = _env_int(name)
        if value is not None:
            return value
    return None


def get_num_gpus() -> int:
    """Return the number of GPUs on this node."""
    return torch.cuda.device_count()


def _mpi_comm_world():
    """Return MPI.COMM_WORLD, importing mpi4py lazily.

    Returns None if mpi4py is unavailable, so env-only callers keep working.
    """
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD
    except ImportError:
        return None


_LOCAL_RANK_VARS = (
    "LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "PMI_LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    "SLURM_LOCALID",
    "FLUX_TASK_LOCAL_ID",
)

_LOCAL_SIZE_VARS = (
    "LOCAL_WORLD_SIZE",
    "MV2_COMM_WORLD_LOCAL_SIZE",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "PMI_LOCAL_SIZE",
    "PALS_LOCAL_SIZE",
)

_WORLD_RANK_VARS = (
    "RANK",
    "MV2_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PALS_RANKID",
    "SLURM_PROCID",
    "FLUX_TASK_RANK",
)

_WORLD_SIZE_VARS = (
    "WORLD_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PALS_NRANKS",
    "SLURM_NTASKS",
    "FLUX_JOB_SIZE",
)


def get_local_rank(required: bool = False) -> int:
    """Return the local MPI rank."""
    value = _first_env_int(_LOCAL_RANK_VARS)
    if value is not None:
        return value
    if required:
        raise RuntimeError("Could not get local rank")
    return 0


def get_local_size(required: bool = False) -> int:
    """Return the number of local MPI ranks.

    Recognizes the same launchers as ``get_local_rank``: a variable honored
    there but not here silently yields 1, which makes per-node logic (e.g. the
    profiler's one-rank-per-node gate) treat every rank as node-local.
    """
    value = _first_env_int(_LOCAL_SIZE_VARS)
    if value is not None:
        return value
    # Slurm and Flux report only totals; assume an even distribution. A zero
    # node count is as unusable as a non-numeric one, so it falls through
    # rather than raising ZeroDivisionError.
    ntasks, nnodes = _env_int("SLURM_NTASKS"), _env_int("SLURM_NNODES")
    if ntasks is not None and nnodes:
        return ntasks // nnodes
    job_size, job_nodes = _env_int("FLUX_JOB_SIZE"), _env_int("FLUX_JOB_NNODES")
    if job_size is not None and job_nodes:
        return job_size // job_nodes
    if required:
        raise RuntimeError("Could not get local size")
    return 1


def get_world_rank(required: bool = False) -> int:
    """Return the global MPI rank."""
    value = _first_env_int(_WORLD_RANK_VARS)
    if value is not None:
        return value
    comm = _mpi_comm_world()
    if comm is not None:
        return comm.Get_rank()
    if required:
        raise RuntimeError("Could not get world rank")
    return 0


def get_world_size(required: bool = False) -> int:
    """Return the number of MPI ranks."""
    value = _first_env_int(_WORLD_SIZE_VARS)
    if value is not None:
        return value
    comm = _mpi_comm_world()
    if comm is not None:
        return comm.Get_size()
    if required:
        raise RuntimeError("Could not get world size")
    return 1


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
    log,
    init_file: Optional[str] = None,
    rendezvous: Literal["env", "tcp", "file"] = "env",
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

    log.debug(
        "rank %s / %s calling init_process_group()",
        get_world_rank(),
        get_world_size(),
    )

    # Bind this process to its compute device BEFORE the first collective.
    # NCCL binds communicators to whatever device is current when it first
    # runs; without an explicit set_device it guesses (rank % num_gpus),
    # which breaks non-block rank placements and can hang on duplicate GPUs.
    device = get_device()

    # Initialize. NCCL requires GPUs; fall back to gloo on CPU-only hosts.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_kwargs = {}
    if device.type == "cuda":
        init_kwargs["device_id"] = device
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=get_world_rank(),
        world_size=get_world_size(),
        **init_kwargs,
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
