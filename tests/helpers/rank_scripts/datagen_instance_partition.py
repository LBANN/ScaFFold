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

"""Two-rank rank script checking ``instance.main``'s work partition.

Launched under a real MPI launcher by ``tests/datagen/test_mpi_consensus.py``.
Each rank's per-category ``glob`` is monkeypatched to return a *different*
phantom set of already-existing instances (what stale metadata caches on a
parallel filesystem would yield). With the fix, only rank 0's scan feeds the
work list, which is then broadcast, so both ranks slice the identical list.

After ``instance.main`` finishes, rank 0 inspects the actual instance tree and
reports coverage: every desired ``(category, instance)`` must exist exactly once
on disk -- no pair written by two ranks (duplicate/corrupt writes) and none
orphaned (never generated).

Markers on stdout (scanned by the parent test):
  ``RANK <r> DONE``                 -- instance.main returned
  ``COVERAGE ok=<0|1> missing=<n> extra=<n>`` -- rank 0's disk audit

Exit code: 0 on full clean coverage, 4 otherwise.
"""

from __future__ import annotations

import glob as glob_mod
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from mpi4py import MPI

import ScaFFold.datagen.instance as inst
from ScaFFold.datagen import layout

VT = 0.15
PN = 64
NCAT = 2  # -> 2 * 145 = 290 desired instances


def _config(fract_base: Path) -> Namespace:
    return Namespace(
        fract_base_dir=str(fract_base),
        n_categories=NCAT,
        seed=1234,
        variance_threshold=VT,
        point_num=PN,
        datagen_from_scratch=False,
    )


def _seed_ifs_params(fract_base: Path) -> None:
    param_dir = Path(layout.category_param_dir(_config(fract_base)))
    param_dir.mkdir(parents=True, exist_ok=True)
    params = np.zeros((2, 13), dtype=np.float64)
    params[:, 0] = params[:, 4] = params[:, 8] = 0.5
    params[1, 9] = params[1, 10] = params[1, 11] = 0.5
    params[0, 12] = 0.5
    for category in range(NCAT):
        np.savetxt(param_dir / f"{category:06d}.csv", params, delimiter=",")


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    workdir = Path(os.environ["WORKDIR"])
    fract_base = workdir / "fractals"

    if rank == 0:
        _seed_ifs_params(fract_base)
    comm.Barrier()

    inst_root = Path(layout.instance_dir(_config(fract_base)))

    # Give each rank a divergent view of pre-existing instances. Only rank 0's
    # view should matter after the fix (its list is broadcast); rank 1's phantom
    # entries must not perturb the partition.
    real_glob = glob_mod.glob

    def fake_glob(pattern, *args, **kwargs):
        result = list(real_glob(pattern, *args, **kwargs))
        # Inject a phantom "existing" instance file into rank 1's file glob so
        # its independently-computed list would differ from rank 0's.
        if rank == 1 and pattern.endswith("_[0-9][0-9][0-9][0-9].npy"):
            result.append(str(inst_root / "000000" / "000000_0000.npy"))
        return result

    inst.glob.glob = fake_glob

    inst.main(_config(fract_base))
    print(f"RANK {rank} DONE", flush=True)
    comm.Barrier()

    if rank == 0:
        missing = 0
        for category in range(NCAT):
            for instance in range(145):
                path = (
                    inst_root / f"{category:06d}" / f"{category:06d}_{instance:04d}.npy"
                )
                if not path.exists():
                    missing += 1
        # Any stray temp files would indicate an interrupted/duplicate write.
        extra = len(list(inst_root.glob(f"*/{inst.TEMP_PREFIX}*")))
        ok = 1 if (missing == 0 and extra == 0) else 0
        print(f"COVERAGE ok={ok} missing={missing} extra={extra}", flush=True)
        sys.exit(0 if ok else 4)
    sys.exit(0)


if __name__ == "__main__":
    main()
