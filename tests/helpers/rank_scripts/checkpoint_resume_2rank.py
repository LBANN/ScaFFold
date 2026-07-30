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

"""Two-rank rank script exercising checkpoint resume I/O (gloo, CPU).

Launched under ``torchrun`` by ``tests/test_checkpointing.py`` to prove that on
resume only rank 0 reads the checkpoint file from disk -- the other ranks
receive the deserialized state over the process group instead of each opening
the (potentially multi-GB) file, which would be an N-way concurrent read of one
file from the shared filesystem at job start.

Each rank instruments ``torch.load`` with a per-rank counter (restricted to the
checkpoint files) that is installed *after* construction, so only the loads
performed by ``load_from_checkpoint`` are counted. Rank 0 also writes the
checkpoint(s) up front (with a non-distributed manager, so the peers can wait on
a plain barrier) and, in the corruption mode, truncates ``last`` to force the
fallback to ``best``.

Behaviour is selected by ``CKPT_MODE``:

* ``read_guard``      -- one good checkpoint at epoch 3; both ranks resume at
  epoch 4. Only rank 0 should load from disk.
* ``corrupt_fallback`` -- ``best`` at epoch 1, ``last`` at epoch 2 then
  truncated; both ranks resume at the best's epoch 2. Rank 0 renames the corrupt
  ``last`` aside and consults ``best``; non-zero ranks still never read.

Markers (rank-tagged, one per line, flushed):
  ``RANK <r> LOADS <n>``            -- torch.load calls on checkpoint files
  ``RANK <r> START_EPOCH <n>``      -- returned start_epoch
  ``RANK <r> CORRUPT_EXISTS <0|1>`` -- rank 0: a ``*.corrupt`` file was produced
  ``RANK <r> BEST_CONSULTED <0|1>`` -- rank 0: ``best`` was among the loads
  ``RANK <r> DONE``                 -- clean completion on every rank
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# Import lazily-safe: the launcher puts the repo root on PYTHONPATH, but be
# explicit so a direct invocation still resolves the package.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ScaFFold.utils.checkpointing import CheckpointManager  # noqa: E402


def _make_manager(base_dir: str, rank: int, *, dist_enabled: bool):
    """A CPU CheckpointManager over a tiny, deterministically-seeded model."""
    torch.manual_seed(0)
    model = torch.nn.Linear(64, 64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return CheckpointManager(
        model=model,
        optimizer=optimizer,
        base_dir=base_dir,
        world_rank=rank,
        dist_enabled=dist_enabled,
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    base_dir = os.environ["CKPT_DIR"]
    mode = os.environ.get("CKPT_MODE", "read_guard")

    dist.init_process_group(backend="gloo")

    last_path = Path(base_dir) / "checkpoint_last.pth"
    best_path = Path(base_dir) / "checkpoint_best.pth"

    # Rank 0 writes the checkpoint(s) with a non-distributed manager so the
    # peers can simply wait on the barrier below rather than participating in
    # the save's (no-op) collectives.
    if rank == 0:
        saver = _make_manager(base_dir, 0, dist_enabled=False)
        if mode == "corrupt_fallback":
            # epoch 1 is the (better) 'best'; epoch 2 is only 'last'.
            saver.save_checkpoint(epoch=1, val_loss_avg=0.1)
            saver.save_checkpoint(epoch=2, val_loss_avg=0.9)
            # Truncate 'last' to simulate a mid-write kill.
            size = last_path.stat().st_size
            with open(last_path, "r+b") as handle:
                handle.truncate(size // 2)
        else:
            saver.save_checkpoint(epoch=3, val_loss_avg=0.5)

    dist.barrier()

    # Every rank builds the resume manager. On rank 0 this reads 'best' once to
    # seed best_val_loss; that happens before the spy is installed so it is not
    # attributed to the resume path under test.
    mgr = _make_manager(base_dir, rank, dist_enabled=True)

    # Count only the checkpoint-file loads performed by load_from_checkpoint.
    # NB: broadcast_object_list itself calls torch.load on an in-memory buffer
    # to deserialize the broadcast payload, so ignore any non-path argument --
    # only a real checkpoint-file path counts as a disk read.
    ckpt_names = {last_path.name, best_path.name}
    loads: list[str] = []
    real_load = torch.load

    def spy_load(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path).name in ckpt_names:
            loads.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    torch.load = spy_load
    try:
        dist.barrier()
        start_epoch = mgr.load_from_checkpoint()
    finally:
        torch.load = real_load

    print(f"RANK {rank} LOADS {len(loads)}", flush=True)
    print(f"RANK {rank} START_EPOCH {start_epoch}", flush=True)
    if rank == 0:
        corrupt_exists = int((Path(base_dir) / "checkpoint_last.pth.corrupt").exists())
        best_consulted = int(any(name == best_path.name for name in loads))
        print(f"RANK {rank} CORRUPT_EXISTS {corrupt_exists}", flush=True)
        print(f"RANK {rank} BEST_CONSULTED {best_consulted}", flush=True)
    print(f"RANK {rank} DONE", flush=True)
    sys.stdout.flush()

    # Make sure every rank has emitted its markers before any rank tears the
    # group down, so the parent never misses a rank's output.
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
