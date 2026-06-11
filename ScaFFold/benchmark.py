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
import shutil
from argparse import Namespace
from pathlib import Path, PosixPath

import yaml
from mpi4py import MPI

from ScaFFold import worker
from ScaFFold.utils.distributed import get_world_rank
from ScaFFold.utils.perf_measure import adiak_init, adiak_value


def create_run_directory(base_dir, combination_index, num_runs):
    """
    Create new directory for current run, named using unique combination_index
    """
    run_dir = base_dir / f"param_set_{combination_index}"
    for i in range(num_runs):
        run_dir_with_iter = Path(f"{run_dir}/run{i}")
        run_dir_with_iter.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_config(run_dir, iter, keys, combination):
    """
    Write run config to a yaml file, and create optional override yaml
    """
    run_config = {key: value for key, value in zip(keys, combination)}
    run_config["run_dir"] = str(
        run_dir.resolve()
    )  # Add abs path to run dir as entry in dict
    run_config["run_iter"] = iter  # Add run_iter identifier as entry in dict
    run_config_path = run_dir / "run_config.yaml"
    with open(run_config_path, "w") as file:
        yaml.dump(run_config, file)
    return run_config_path


def main(kwargs_dict: dict = {}):
    args = Namespace(**kwargs_dict)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Get MPI information
    comm = MPI.COMM_WORLD
    rank = get_world_rank(required=True)
    if rank == 0:
        print(f"args found: {args}")

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
            print(f"Adiak: skipping key with dict value '{key}'")
            continue
        if isinstance(value, PosixPath):
            value = str(value)
        adiak_value(key, value)

    worker.main(kwargs_dict=kdict)
