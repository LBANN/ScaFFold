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

"""Two-rank rank script exercising ``get_dataset``'s consensus paths.

Launched under a real MPI launcher by ``tests/datagen/test_mpi_consensus.py``.
The behavior is selected by the ``FAULT_MODE`` environment variable:

* ``none``    -- clean run; both ranks must RETURN the same finalized path.
* ``all``     -- ``volumegen.main`` raises on every rank; both must RAISE.
* ``rank1``   -- ``volumegen.main`` raises on rank 1 only; both must RAISE.
* ``missing`` -- one instance file needed by a rank is deleted, so volumegen
  raises ``FileNotFoundError``; both ranks must RAISE (no ``SystemExit``, no
  peer hang).
* ``reuse``   -- a reusable dataset exists but non-root ranks' directory scan is
  monkeypatched to see nothing; both ranks must RETURN the *same* reused path.

Markers on stdout (scanned by the parent test):
  ``RANK <r> RETURNED <path>``  -- get_dataset returned a path
  ``RANK <r> RAISED <ExcType>`` -- get_dataset raised
  ``RANK <r> SYSEXIT``          -- a SystemExit escaped get_dataset (a regression)

Exit code: 0 when this rank returned, 3 when it raised as designed.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from mpi4py import MPI

import ScaFFold.datagen.get_dataset as gd
from ScaFFold.datagen import layout, volumegen

VT = 0.15
PN = 64
NCAT = 2
NINST = 2  # -> 4 volumes total, 2 per rank at 2 ranks


def _config(dataset_dir: Path, fract_base: Path) -> Namespace:
    return Namespace(
        dataset_dir=str(dataset_dir),
        fract_base_dir=str(fract_base),
        n_categories=NCAT,
        n_instances_used_per_fractal=NINST,
        problem_scale=4,
        seed=1234,
        variance_threshold=VT,
        n_fracts_per_vol=1,
        val_split=0,
        vol_size=16,
        point_num=PN,
        scale=1,
    )


def _instance_path(fract_base: Path, cat: int, inst: int) -> Path:
    # The instance library is keyed by seed; derive the path from the same
    # config the run under test uses.
    inst_root = Path(layout.instance_dir(_config(Path("unused"), fract_base)))
    return inst_root / f"{cat:06d}" / f"{cat:06d}_{inst:04d}.npy"


def _seed_instances(fract_base: Path) -> None:
    rng = np.random.default_rng(0)
    for cat in range(NCAT):
        d = _instance_path(fract_base, cat, 0).parent
        d.mkdir(parents=True, exist_ok=True)
        for inst in range(145):
            np.save(_instance_path(fract_base, cat, inst), rng.random((PN, 3)))


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mode = os.environ.get("FAULT_MODE", "none")
    workdir = Path(os.environ["WORKDIR"])
    fract_base = workdir / "fractals"
    dataset_dir = workdir / "datasets"

    # Rank 0 lays down shared inputs; peers wait on the barrier.
    if rank == 0:
        _seed_instances(fract_base)
        dataset_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    config = _config(dataset_dir, fract_base)

    # Fault injection: wrap volumegen.main so it raises where requested.
    real_main = volumegen.main

    def faulty_main(cfg):
        if mode == "all":
            raise RuntimeError(f"injected failure on rank {rank}")
        if mode == "rank1" and rank == 1:
            raise RuntimeError(f"injected failure on rank {rank}")
        return real_main(cfg)

    if mode in ("all", "rank1"):
        volumegen.main = faulty_main

    if mode == "missing":
        # Delete an instance file every rank's partition may reference so the
        # owning rank hits the missing-file path.
        if rank == 0:
            _instance_path(fract_base, 0, 0).unlink(missing_ok=True)
        comm.Barrier()

    if mode == "reuse":
        # Rank 0 plants a finalized reusable dataset, then non-root ranks' scan
        # is blinded so only the broadcast decision can keep them in agreement.
        _plant_reuse(config, dataset_dir, rank, comm)

    try:
        result = gd.get_dataset(config)
        print(f"RANK {rank} RETURNED {result}", flush=True)
        sys.exit(0)
    except SystemExit:
        # A SystemExit escaping get_dataset would hang peers -- report it.
        print(f"RANK {rank} SYSEXIT", flush=True)
        raise
    except BaseException as exc:  # noqa: BLE001 - report any failure type
        print(f"RANK {rank} RAISED {type(exc).__name__}", flush=True)
        sys.exit(3)


def _plant_reuse(config, dataset_dir, rank, comm) -> None:
    """Create a reusable dataset and blind non-root ranks' directory scan."""
    import yaml

    config_dict = vars(config).copy()
    config_dict["dataset_format_version"] = gd.DATASET_FORMAT_VERSION
    volume_config = gd._get_required_keys_dict(config_dict, gd.INCLUDE_KEYS)
    config_id = gd._hash_volume_config(volume_config)
    base = dataset_dir / config_id

    if rank == 0:
        chosen = base / "20260101-000000__deadbee"
        chosen.mkdir(parents=True, exist_ok=True)
        meta = {
            "config_id": config_id,
            "dataset_format_version": gd.DATASET_FORMAT_VERSION,
        }
        (chosen / gd.META_FILENAME).write_text(yaml.safe_dump(meta))
    comm.Barrier()

    if rank != 0:
        # Simulate a stale cache: this rank's own scan of base sees nothing.
        real_iterdir = Path.iterdir

        def blind_iterdir(self):
            if self == base:
                return iter(())
            return real_iterdir(self)

        Path.iterdir = blind_iterdir


if __name__ == "__main__":
    main()
