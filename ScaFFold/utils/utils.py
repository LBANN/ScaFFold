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
import random
import sys

import torch
import torch.distributed as dist

from ScaFFold.utils.distributed import get_world_rank

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s(%(filename)s): %(message)s"
)


def plot_img_and_mask(img, mask):
    import matplotlib.pyplot as plt

    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title("Input image")
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f"Mask (class {i + 1})")
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def set_seeds(seed_value=42):
    """Set seeds for reproducibility."""

    import numpy as np

    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed_all(seed_value)  # PyTorch for GPUs


def customlog(msg: str, ranks=(0,), allranks=False, level=0, verbose=0):
    rank = get_world_rank()
    if (rank in ranks or allranks) and level <= verbose:
        logging.info(f"(rank {rank}): {msg}")


class MPIRankFilter(logging.Filter):
    def __init__(self, ranks: set[int] or None = None):
        super().__init__()
        self.allowed_ranks = ranks

    def filter(self, record: logging.LogRecord) -> bool:
        rank = get_world_rank()
        record.mpi_rank = rank
        # If no allowed ranks specified, only rank 0 logs
        if self.allowed_ranks is None:
            return rank == 0
        # Otherwise only allow logs from the specified ranks
        return rank in self.allowed_ranks


def setup_mpi_logger(
    name: str, verbosity: int, ranks: set[int] or None = None
) -> logging.Logger:
    # Map verbosity to logging levels
    if verbosity == 0:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Set formatting, including MPI rank
    fmt = "[%(asctime)s][%(filename)s:%(lineno)d][rank=%(mpi_rank)d][%(levelname)s] %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")

    for handler in logger.handlers:
        if getattr(handler, "_scaffold_mpi_handler", False):
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
            handler.filters = [
                f for f in handler.filters if not isinstance(f, MPIRankFilter)
            ]
            handler.addFilter(MPIRankFilter(ranks))
            break
    else:
        # Create a StreamHandler (to stderr) and attach MPI filter
        handler = logging.StreamHandler(stream=sys.stderr)
        handler._scaffold_mpi_handler = True
        handler.setLevel(log_level)
        handler.addFilter(MPIRankFilter(ranks))
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent duplicate logging from Basic Logger
    logger.propagate = False
    return logger


def mem_stats():
    dev = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info()  # device-level (driver) view
    stats = torch.cuda.memory_stats(dev)  # allocator internals
    return {
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "device": dev,
        "allocated": torch.cuda.memory_allocated(
            dev
        ),  # bytes currently used by tensors
        "reserved": torch.cuda.memory_reserved(
            dev
        ),  # bytes reserved by the caching allocator
        "max_allocated": stats["allocated_bytes.all.peak"],  # peak since last reset
        "max_reserved": stats["reserved_bytes.all.peak"],
        "free_bytes": free,
        "total_bytes": total,  # whole-device view (includes non-PyTorch)
    }


def gather_and_print_mem(log, tag=""):
    if log.getEffectiveLevel() > 10:  # 10 -> DEBUG
        return
    stats = mem_stats()
    if dist.is_initialized():
        world = dist.get_world_size()
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, stats)
        if dist.get_rank() == 0:
            log.debug(f"=== {tag} ===")
            for s in sorted(gathered, key=lambda d: d["rank"]):
                gb = 1024**3
                log.debug(
                    f"rank{str(s['rank']).rjust(2)} | dev {s['device']} | "
                    f"alloc {s['allocated'] / gb:5.2f} GB | res {s['reserved'] / gb:5.2f} GB "
                    f"| max_alloc {s['max_allocated'] / gb:5.2f} GB | "
                    f"| max_res {s['max_reserved'] / gb:5.2f} GB | "
                    f"| free {s['free_bytes'] / gb:5.2f} GB"
                )
    else:
        log.debug(stats)
