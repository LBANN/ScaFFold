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

"""Two-rank rank script exercising the dataset id-consistency guard.

Launched under ``torchrun`` (gloo) by ``tests/test_data_loading.py``. Each rank
monkeypatches ``data_loading.listdir`` so that the ranks observe *different*
file sets for the same directory -- the residual divergence a parallel
filesystem can produce even after sorting. With the guard in place, dataset
construction must raise on every rank; without it, both ranks build mismatched
datasets and exit cleanly (the silent-corruption bug).

Markers on stdout:
  ``RANK <r> IDS_HASH <hex>``   -- the rank's own id-list digest (always)
  ``RANK <r> CONSTRUCTED ...``  -- construction returned (no guard fired)
  ``RANK <r> GUARD_RAISED ...`` -- construction raised on the mismatch

Exit code is 0 when construction succeeded and 3 when the guard raised.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from os.path import isfile, join, splitext
from pathlib import Path

import numpy as np
import torch.distributed as dist
import yaml

import ScaFFold.utils.data_loading as dl
from ScaFFold.utils.data_loading import FractalDataset
from ScaFFold.utils.data_types import MASK_DTYPE, VOLUME_DTYPE


def _build_v2_dataset(root: Path, n_volumes: int, n: int) -> None:
    """Write a minimal v2 dataset (channels-first volumes + dense masks)."""
    vol_dir = root / "volumes" / "training"
    mask_dir = root / "masks" / "training"
    vol_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n_volumes):
        img = np.full((3, n, n, n), float(k), dtype=VOLUME_DTYPE)
        msk = np.zeros((n, n, n), dtype=MASK_DTYPE)
        np.save(vol_dir / f"vol_{k:02d}.npy", img)
        np.save(mask_dir / f"vol_{k:02d}_mask.npy", msk)
    for name in ("train_unique_mask_vals", "val_unique_mask_vals"):
        with open(root / name, "wb") as handle:
            pickle.dump({"mask_values": [0]}, handle)
    with open(root / "meta.yaml", "w") as handle:
        yaml.safe_dump({"dataset_format_version": 2}, handle)


def main() -> None:
    dataset_dir = Path(os.environ["DATASET_DIR"])
    n_volumes = int(os.environ.get("N_VOLUMES", "6"))

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()

    vol_dir = dataset_dir / "volumes" / "training"
    mask_dir = dataset_dir / "masks" / "training"

    # Rank 0 materializes the shared dataset; peers wait for it on the barrier.
    if rank == 0:
        _build_v2_dataset(dataset_dir, n_volumes, 4)
    dist.barrier()

    real_listdir = dl.listdir

    def fake_listdir(path):
        entries = list(real_listdir(path))
        if Path(path) == vol_dir:
            entries = sorted(entries)
            # Divergent readdir view: non-zero ranks drop the last entry, so the
            # id *set* differs across ranks even after sorting.
            if rank != 0:
                entries = entries[:-1]
        return entries

    dl.listdir = fake_listdir

    ids = sorted(
        splitext(f)[0]
        for f in fake_listdir(vol_dir)
        if isfile(join(vol_dir, f)) and not f.startswith(".")
    )
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    print(f"RANK {rank} IDS_HASH {digest}", flush=True)
    print(f"RANK {rank} IDS {ids}", flush=True)
    sys.stdout.flush()
    # Make sure every rank has emitted (and flushed) its digest line before any
    # rank can raise and exit -- otherwise the launcher may tear a peer down
    # mid-flush and the parent would miss one rank's hash.
    dist.barrier()

    try:
        dataset = FractalDataset(
            vol_dir,
            mask_dir,
            data_dir=dataset_dir / "train_unique_mask_vals",
        )
    except RuntimeError as exc:
        print(f"RANK {rank} GUARD_RAISED {type(exc).__name__}", flush=True)
        # Every rank raises symmetrically (the guard collective completes before
        # the comparison), so no peer is left waiting -- just exit non-zero.
        sys.exit(3)

    print(f"RANK {rank} CONSTRUCTED len={len(dataset)}", flush=True)
    # Unguarded path: both ranks built (mismatched) datasets. Tear down cleanly
    # so the silent-divergence case exits 0.
    try:
        dist.barrier()
    finally:
        dist.destroy_process_group()
    sys.exit(0)


if __name__ == "__main__":
    main()
