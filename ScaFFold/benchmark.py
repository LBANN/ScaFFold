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

import itertools
import shutil
from argparse import Namespace
from pathlib import Path, PosixPath

import yaml
from mpi4py import MPI

from ScaFFold import worker
from ScaFFold.utils.config_utils import Config
from ScaFFold.utils.distributed import get_world_rank
from ScaFFold.utils.perf_measure import adiak_init, adiak_value
from ScaFFold.utils.utils import setup_mpi_logger


def expand_sweep_combinations(config_dict):
    """
    Expand list-valued sweep parameters into per-run scalar configs.

    Scalar-typed config fields holding lists are treated as sweep dimensions;
    the cross product of their values yields one config dict per combination.
    An all-scalar config yields exactly one combination. Legitimately
    list-typed fields (e.g. dc_num_shards) are never treated as sweeps.
    """
    sweep_keys = sorted(
        k for k in Config._SCALAR_KEYS if isinstance(config_dict.get(k), list)
    )
    if not sweep_keys:
        return [dict(config_dict)]

    combinations = []
    value_lists = [config_dict[k] for k in sweep_keys]
    for values in itertools.product(*value_lists):
        combo = dict(config_dict)
        combo.update(dict(zip(sweep_keys, values)))
        combinations.append(combo)
    return combinations


def main(kwargs_dict: dict = {}):
    args = Namespace(**kwargs_dict)
    log = setup_mpi_logger(__file__, args.verbose)

    # Get MPI information
    comm = MPI.COMM_WORLD
    rank = get_world_rank(required=True)
    log.debug("args found: %s", args)

    run_dicts = None
    # Now set up and start benchmark run(s)
    if args.restart:
        # Resume the run in the directory the CLI resolved. The worker path
        # reads config.run_dir/run_iter directly (a missing run_dir would crash
        # BaseTrainer), so fill both from the resolved benchmark_run_dir here
        # just as the fresh, single-combination path does below.
        benchmark_run_dir = args.benchmark_run_dir
        kdict = {k: v for k, v in vars(args).items() if k not in ["command"]}
        kdict["run_dir"] = str(benchmark_run_dir)
        kdict["run_iter"] = Path(f"{benchmark_run_dir}/run")
        run_dicts = [kdict]
    elif rank == 0:
        # Get run dir
        benchmark_run_dir = args.benchmark_run_dir

        # Save copy of benchmark config yml to run dir
        bench_config_path = Path(args.config)
        shutil.copy(bench_config_path, benchmark_run_dir)

        base_dict = {k: v for k, v in vars(args).items() if k not in ["command"]}
        combinations = expand_sweep_combinations(base_dict)

        # One run (and run directory) per parameter combination. The single
        # all-scalar combination keeps the original flat layout.
        run_dicts = []
        if len(combinations) == 1:
            kdict = combinations[0]
            kdict["run_dir"] = str(benchmark_run_dir)
            kdict["run_iter"] = Path(f"{benchmark_run_dir}/run")
            run_dicts.append(kdict)
        else:
            for i, kdict in enumerate(combinations):
                run_dir = Path(benchmark_run_dir) / f"param_set_{i}"
                run_dir.mkdir(parents=True, exist_ok=True)
                kdict["run_dir"] = str(run_dir)
                kdict["run_iter"] = run_dir / "run"
                with open(run_dir / "run_config.yaml", "w") as file:
                    yaml.dump(
                        {k: str(v) if isinstance(v, PosixPath) else v
                         for k, v in kdict.items()},
                        file,
                    )
                run_dicts.append(kdict)

    comm.Barrier()
    run_dicts = comm.bcast(run_dicts, root=0)

    adiak_init(comm)
    for kdict in run_dicts:
        # Add all config params as metadata
        for key, value in kdict.items():
            if isinstance(value, dict):
                log.debug("Adiak: skipping key with dict value '%s'", key)
                continue
            if isinstance(value, PosixPath):
                value = str(value)
            adiak_value(key, value)

        worker.main(kwargs_dict=kdict)
