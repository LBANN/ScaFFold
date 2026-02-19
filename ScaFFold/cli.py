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

import argparse
import socket
import sys
from datetime import datetime
from pathlib import Path

import yaml
from mpi4py import MPI

from ScaFFold import benchmark, generate_fractals
from ScaFFold.utils import config_utils
from ScaFFold.utils.collect_scheduler_info import collect_scheduler_metadata
from ScaFFold.utils.create_restart_script import create_restart_script
from ScaFFold.utils.utils import customlog


def main():
    """
    Command line interface for ScaFFold.
    Serves as a unified entry point for users to run fractal
    generation and benchmarking.
    """

    # Create top-level parser
    parser = argparse.ArgumentParser(
        prog="scaffold",
        description="Scaffold CLI: A command-line tool for the ScaFFold AI Benchmark.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="verbosity level: v=DEBUG."
    )

    # Create subparsers for different subcommands (generate_fractals, benchmark, etc.).
    subparsers = parser.add_subparsers(
        description="Valid subcommands for running ScaFFold",
        help="Additional help available for each subcommand.",
        dest="command",
        required=True,
    )

    generate_fractals_parser = subparsers.add_parser(
        "generate_fractals",
        help="Generate fractal classes and instances.",
        description="Must be ran before 'benchmark'",
    )
    generate_fractals_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to config file for fractal generation",
        required=True,
    )
    generate_fractals_parser.add_argument(
        "--datagen-batch-size",
        type=int,
        default=10000,
        help="Batch size for per-rank category generation",
    )

    # Config overrides
    generate_fractals_parser.add_argument(
        "--problem-scale",
        type=int,
        help="Determines dataset resolution and number of UNet layers.",
    )
    generate_fractals_parser.add_argument(
        "--n-categories",
        type=int,
        help="Number of fractal categories present in the dataset.",
    )
    generate_fractals_parser.add_argument(
        "--fract-base-dir",
        type=str,
        help="Base directory for fractal IFS and instances.",
    )

    # --------------
    # Subcommand: benchmark
    # --------------
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run the benchmark.",
        description=(
            "The default run method for ScaFFold."
            "Users may specify lists of run parameters in the config file."
            "This subcommand runs one instance of the benchmark for each parameter combination."
            "Requires path to config file."
        ),
    )
    # Specify config file
    benchmark_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to config file for running benchmark",
        required=True,
    )

    # Arguments from benchmark_default.yml
    benchmark_parser.add_argument(
        "--base-run-dir", type=str, help="Subfolder of $(pwd) in which to run jobs."
    )
    benchmark_parser.add_argument(
        "--n-categories",
        type=int,
        help="Number of fractal categories present in the dataset.",
    )
    benchmark_parser.add_argument(
        "--n-instances-used-per-fractal",
        type=int,
        help="Number of unique instances to pull from each fractal class.",
    )
    benchmark_parser.add_argument(
        "--problem-scale",
        type=int,
        help="Determines dataset resolution and number of UNet layers.",
    )
    benchmark_parser.add_argument(
        "--unet-bottleneck-dim",
        type=int,
        help="Power of 2 of the UNet bottleneck layer dimension.",
    )
    benchmark_parser.add_argument("--seed", type=int, help="Random seed.")
    benchmark_parser.add_argument(
        "--batch-size", type=int, nargs="+", help="Batch sizes for each volume size."
    )
    benchmark_parser.add_argument(
        "--optimizer",
        type=str,
        choices=["ADAM", "RMSProp"],
        help="Optimizer for training.",
    )
    benchmark_parser.add_argument(
        "--restart",
        action="store_true",
        help="Indicates this run is a restart/resume of a previous run.",
    )
    benchmark_parser.add_argument(
        "--run-dir",
        type=str,
        help="Resume execution in this specific directory. Overrides --base-run-dir.",
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Parse the command-line arguments.
    args = parser.parse_args()
    combined_config = None

    if rank == 0:
        print(f"args = {args}")

        bench_config = config_utils.load_config(Path(args.config), "sweep")
        bench_config_dict = (
            vars(bench_config) if not isinstance(bench_config, dict) else bench_config
        )
        cli_args = vars(args)

        # Combine configs: CLI args override config file values
        combined_config = bench_config_dict.copy()
        for key, value in cli_args.items():
            if key not in combined_config:
                combined_config[key] = value
            elif value is not None and key != "command":
                print(f"Overriding '{key}={combined_config[key]}' with '{key}={value}'")
                combined_config[key] = value

        # Recalculate unet_layers to capture any CLI overrides
        combined_config["unet_layers"] = (
            combined_config["problem_scale"]
            - combined_config["unet_bottleneck_dim"]
            + 1
        )

        # Resolve paths to absolute, matching Config() behavior
        if "base_run_dir" in combined_config and combined_config["base_run_dir"]:
            combined_config["base_run_dir"] = str(
                Path(combined_config["base_run_dir"]).resolve()
            )

        if "dataset_dir" in combined_config and combined_config["dataset_dir"]:
            combined_config["dataset_dir"] = str(
                Path(combined_config["dataset_dir"]).resolve()
            )

        # Calculate these variables after override
        combined_config["vol_size"] = pow(2, combined_config["problem_scale"])
        combined_config["point_num"] = int(combined_config["vol_size"] ** 3 / 256)

        # Handle Restart / Resume logic
        if hasattr(args, "restart") and args.restart == True:
            print("Restart flag detected: Forcing train_from_scratch = False")
            combined_config["train_from_scratch"] = False
            combined_config["restart"] = True

        # If user manually supplied --run-dir (via restart script), use it.
        if hasattr(args, "run_dir") and args.run_dir is not None:
            print(f"Resuming in existing directory: {args.run_dir}")
            benchmark_run_dir = Path(args.run_dir)
            # Ensure we don't accidentally wipe checkpoints even if --restart wasn't explicitly passed
            combined_config["train_from_scratch"] = False
        else:
            base_run_dir = Path(combined_config["base_run_dir"])
            timestamp = datetime.now().strftime(
                f"{combined_config.get('job_name')}_%Y%m%d-%H%M%S"
            )
            benchmark_run_dir = base_run_dir / timestamp
            customlog(
                f"benchmark_run_dir created at path {Path.resolve(benchmark_run_dir)}"
            )

            combined_config["benchmark_run_dir"] = str(benchmark_run_dir)
        benchmark_run_dir.mkdir(parents=True, exist_ok=True)

        # Add scheduler metadata and machine name to config.yaml
        combined_config["scheduler_metadata"] = collect_scheduler_metadata()
        combined_config["machine_name"] = socket.gethostname()

        # Dump configs (Overwrite is okay/desired on restart to capture new job IDs)
        overrides = {
            k: v for k, v in cli_args.items() if v is not None and k != "command"
        }
        with open(benchmark_run_dir / "overrides.yaml", "w") as file:
            yaml.dump(overrides, file)
        with open(benchmark_run_dir / "config.yaml", "w") as file:
            yaml.dump(combined_config, file)

        # 4. Generate/Update the restart script in the directory
        create_restart_script(benchmark_run_dir)

    comm.Barrier()
    combined_config = comm.bcast(combined_config, root=0)
    if rank == 0:
        print(f"combined_config = {combined_config}")

    if args.command == "benchmark":
        benchmark.main(kwargs_dict=combined_config)
    elif args.command == "generate_fractals":
        generate_fractals.main(kwargs_dict=combined_config)
    else:
        raise ValueError(
            f"Missing or invalid subcommand: {args.command}. Please consult ScaFFold documentation."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
