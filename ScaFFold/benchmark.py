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

    run_dict = None
    # Now set up and start the benchmark run. Each invocation runs exactly one
    # benchmark run, in the run directory the CLI resolved.
    benchmark_run_dir = args.benchmark_run_dir
    if args.restart:
        # Resume the run in that directory. The worker path reads
        # config.run_dir/run_iter directly (a missing run_dir would crash
        # BaseTrainer), so fill both in here just as the fresh path does below.
        run_dict = {k: v for k, v in vars(args).items() if k not in ["command"]}
        run_dict["run_dir"] = str(benchmark_run_dir)
        run_dict["run_iter"] = Path(f"{benchmark_run_dir}/run")
    elif rank == 0:
        # Save a copy of the benchmark config yml in the run dir, under a fixed
        # name of its own. Copying it in under its original name overwrites the
        # merged config.yaml the CLI just wrote whenever the base config file
        # happens to be called config.yaml -- and restart.sh points -c at
        # $RUN_DIR/config.yaml, so the restart would reload the raw base config
        # with none of the run's overrides or metadata.
        bench_config_path = Path(args.config)
        shutil.copy(bench_config_path, Path(benchmark_run_dir) / "base_config.yaml")

        run_dict = {k: v for k, v in vars(args).items() if k not in ["command"]}
        run_dict["run_dir"] = str(benchmark_run_dir)
        run_dict["run_iter"] = Path(f"{benchmark_run_dir}/run")

    comm.Barrier()
    run_dict = comm.bcast(run_dict, root=0)

    adiak_init(comm)
    # Add all config params as metadata
    for key, value in run_dict.items():
        if isinstance(value, dict):
            log.debug("Adiak: skipping key with dict value '%s'", key)
            continue
        if isinstance(value, PosixPath):
            value = str(value)
        adiak_value(key, value)

    worker.main(kwargs_dict=run_dict)
