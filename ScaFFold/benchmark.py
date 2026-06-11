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

import shutil
from argparse import Namespace
from pathlib import Path, PosixPath

from mpi4py import MPI

from ScaFFold import worker
from ScaFFold.utils.distributed import get_world_rank
from ScaFFold.utils.perf_measure import adiak_init, adiak_value
from ScaFFold.utils.utils import setup_mpi_logger


def main(kwargs_dict: dict = {}):
    args = Namespace(**kwargs_dict)
    log = setup_mpi_logger(__file__, args.verbose)

    # Get MPI information
    comm = MPI.COMM_WORLD
    rank = get_world_rank(required=True)
    log.debug("args found: %s", args)

    kdict = None
    # Now set up and start benchmark run(s)
    if args.restart:
        kdict = {k: v for k, v in vars(args).items() if k not in ["command"]}
    elif rank == 0:
        # Get run dir
        benchmark_run_dir = args.benchmark_run_dir

        # Save copy of benchmark config yml to run dir
        bench_config_path = Path(args.config)
        shutil.copy(bench_config_path, benchmark_run_dir)

        run_dir_with_iter = Path(f"{benchmark_run_dir}/run")
        kdict = {k: v for k, v in vars(args).items() if k not in ["command"]}
        kdict["run_dir"] = str(benchmark_run_dir)
        kdict["run_iter"] = run_dir_with_iter

    comm.Barrier()
    kdict = comm.bcast(kdict, root=0)

    # Add all config params as metadata
    adiak_init(comm)
    for key, value in kdict.items():
        if isinstance(value, dict):
            log.debug("Adiak: skipping key with dict value '%s'", key)
            continue
        if isinstance(value, PosixPath):
            value = str(value)
        adiak_value(key, value)

    worker.main(kwargs_dict=kdict)
