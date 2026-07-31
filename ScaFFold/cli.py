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

from ScaFFold.utils import config_utils
from ScaFFold.utils.collect_scheduler_info import collect_scheduler_metadata
from ScaFFold.utils.create_restart_script import create_restart_script
from ScaFFold.utils.distributed import get_world_size
from ScaFFold.utils.utils import setup_mpi_logger


def check_launcher_world_size(mpi_world_size):
    """Verify that the MPI world spans the whole job.

    ScaFFold uses two different sources of truth for the job shape: the CLI and
    the benchmark driver make their job-wide decisions on ``MPI.COMM_WORLD``
    rank 0, while the training path takes its rank and world size from the
    launcher's environment (``get_world_rank`` / ``get_world_size``). Those
    agree only when the job was started by an MPI-aware launcher.

    Under a plain ``torchrun`` (or a bare ``python`` invocation of several
    processes) mpi4py initializes as an independent singleton in every process,
    so every process believes it is MPI rank 0: each one runs the rank-0 block,
    atomically claims its *own* timestamped run directory, and the job then
    diverges -- non-zero launcher ranks crash on a broadcast that never
    happened while rank 0 blocks in the first collective until it times out.

    Fail loudly here, before any run directory is created, instead of leaving
    that mess behind. ``get_world_size`` falls back to the MPI communicator
    when the environment reports nothing, so an unlauncher-ed single process
    trivially agrees with itself.
    """
    env_world_size = get_world_size()
    if env_world_size != mpi_world_size:
        raise RuntimeError(
            f"Launcher/MPI world size mismatch: MPI.COMM_WORLD reports "
            f"{mpi_world_size} rank(s) but the launcher environment reports "
            f"{env_world_size}. ScaFFold decides run directories and restart "
            "state on MPI rank 0 and broadcasts them, so an MPI world that "
            "does not span the job is unrecoverable: every process acts as "
            "rank 0, each claims a separate run directory, and the job hangs "
            "in the first collective. This is what a plain 'torchrun' (or "
            "launching the processes directly) produces, because mpi4py then "
            "initializes as a singleton in every process. Launch ScaFFold "
            "with an MPI-aware launcher (torchrun-hpc, flux run, srun, "
            "mpirun) so that MPI spans all ranks, or run a single process "
            "with no launcher environment set."
        )


def _make_fresh_run_dir(base_run_dir, timestamp):
    """Create a fresh timestamped run directory without clobbering an existing one.

    The bare timestamped name is attempted first. If it already exists -- two
    jobs launched under the same base directory within the same wall-clock
    second name their directories identically -- a numeric suffix is appended
    and incremented until an unused name is found. ``mkdir(exist_ok=False)`` is
    atomic, so each name is claimed by exactly one creator and neither job
    overwrites the other's config or stats.

    Returns the created directory as a ``Path``.
    """
    candidate = base_run_dir / timestamp
    suffix = 0
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = base_run_dir / f"{timestamp}-{suffix}"


def missing_checkpoint_error(combined_config):
    """Return the "nothing to resume from" message, or None if a restart can run.

    Reports the first problem found rather than raising, so the caller can make
    this a rank-0 decision and broadcast the verdict instead of letting every
    rank stat the shared filesystem and possibly disagree.
    """
    checkpoint_dir = Path(combined_config["run_dir"]) / combined_config.get(
        "checkpoint_dir", "checkpoints"
    )
    expected_checkpoints = (
        checkpoint_dir / "checkpoint_last.pth",
        checkpoint_dir / "checkpoint_best.pth",
    )
    if any(path.exists() for path in expected_checkpoints):
        return None
    expected = " or ".join(str(path) for path in expected_checkpoints)
    return f"Restart requested but no checkpoint was found. Expected {expected}."


def resolve_run_dir(args_dict, combined_config):
    """Decide the benchmark run directory and whether this launch resumes a run.

    The semantics are fixed and unambiguous:

    * ``--run-dir DIR`` (with or without ``--restart``): resume in that exact
      directory. ``train_from_scratch`` is forced off and ``restart`` on so the
      downstream benchmark driver takes its restart path.
    * ``--restart`` without ``--run-dir``: rejected with a clear error. The run
      directory to resume must be named explicitly; the most recent directory
      is never guessed.
    * neither flag: create a fresh timestamped directory under ``base_run_dir``,
      retrying with a numeric suffix on a same-second name collision.

    ``combined_config['benchmark_run_dir']`` is set in every path so the driver
    can always read it. Returns ``(benchmark_run_dir: Path, restarting: bool)``.
    """
    restart_flag = bool(args_dict.get("restart"))
    run_dir_arg = args_dict.get("run_dir")

    if run_dir_arg is not None:
        benchmark_run_dir = Path(run_dir_arg)
        # An explicit run dir is expected to already exist; tolerate a missing
        # one but never treat a pre-existing dir as an error here.
        benchmark_run_dir.mkdir(parents=True, exist_ok=True)
        restarting = True
    elif restart_flag:
        raise ValueError(
            "--restart requires --run-dir: pass the directory of the run to "
            "resume (e.g. '--restart --run-dir <path>'). The most recent run "
            "directory is not resolved automatically."
        )
    else:
        base_run_dir = Path(combined_config["base_run_dir"])
        timestamp = datetime.now().strftime(
            f"{combined_config.get('job_name')}_%Y%m%d-%H%M%S"
        )
        benchmark_run_dir = _make_fresh_run_dir(base_run_dir, timestamp)
        log = setup_mpi_logger(__file__, args_dict.get("verbose", 0))
        log.info(
            "benchmark_run_dir created at path %s", Path.resolve(benchmark_run_dir)
        )
        restarting = False

    combined_config["benchmark_run_dir"] = str(benchmark_run_dir)
    if restarting:
        combined_config["train_from_scratch"] = False
        combined_config["restart"] = True
    return benchmark_run_dir, restarting


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
        description="Must be run before 'benchmark'",
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
            "The default run method for ScaFFold. "
            "Each invocation runs exactly one instance of the benchmark, using "
            "the single-valued run parameters given in the config file. "
            "Requires path to config file."
        ),
    )
    # Specify config file(s): the first is the complete base config, any
    # additional -c/--config files are partial overrides applied in order.
    benchmark_parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="append",
        default=None,
        help=(
            "Path to config file for running benchmark. May be given more "
            "than once: the first file is the base config and later files "
            "are partial overrides."
        ),
        required=True,
    )

    # Arguments from benchmark_default.yml
    benchmark_parser.add_argument(
        "--base-run-dir", type=str, help="Subfolder of $(pwd) in which to run jobs."
    )
    benchmark_parser.add_argument(
        "--fract-base-dir",
        type=str,
        help="Base directory for fractal IFS and instances.",
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
        "--local-batch-size",
        type=int,
        help="Batch size for each vol size per DDP rank.",
    )
    benchmark_parser.add_argument(
        "--warmup-batches",
        type=int,
        help="Number of warmup batches to run per rank before training.",
    )
    benchmark_parser.add_argument(
        "--group-norm-groups",
        type=int,
        help="Number of groups used by GroupNorm in the UNet blocks.",
    )
    benchmark_parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        help="Number of DataLoader worker processes per rank.",
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
    benchmark_parser.add_argument(
        "--dc-num-shards",
        type=int,
        nargs=3,
        help="DistConv param: number of shards to divide the tensor into. It's best to choose the fewest ranks needed to fit one sample in GPU memory, since that keeps communication at a minimum",
    )
    benchmark_parser.add_argument(
        "--dc-shard-dims",
        type=int,
        nargs=3,
        help="DistConv param: tensor dimensions to shard.",
    )
    benchmark_parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs.",
    )
    benchmark_parser.add_argument(
        "--starting-learning-rate",
        type=float,
        help="Initial learning rate for training.",
    )
    benchmark_parser.add_argument(
        "--min-learning-rate",
        type=float,
        help="Minimum learning rate for CosineAnnealingWarmRestarts.",
    )
    benchmark_parser.add_argument(
        "--T-0",
        dest="T_0",
        type=int,
        help="Epochs in the first cosine restart cycle.",
    )
    benchmark_parser.add_argument(
        "--T-mult",
        dest="T_mult",
        type=int,
        help="Restart cycle growth factor.",
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Every rank runs this identically, before any run directory is created,
    # so a mis-launched job aborts uniformly instead of leaving per-rank run
    # dirs behind and hanging.
    check_launcher_world_size(comm.Get_size())
    # Parse the command-line arguments.
    args = parser.parse_args()
    log = setup_mpi_logger(__file__, args.verbose)
    combined_config = None

    # Reject an incoherent resume request on every rank, before the rank-0
    # config work and its collective barrier, so the job fails fast and
    # uniformly instead of leaving the non-zero ranks blocked in a barrier
    # while rank 0 aborts.
    if (
        args.command == "benchmark"
        and getattr(args, "restart", False)
        and getattr(args, "run_dir", None) is None
    ):
        parser.error(
            "--restart requires --run-dir: pass the directory of the run to "
            "resume (e.g. '--restart --run-dir <path>')."
        )

    if rank == 0:
        log.debug("args = %s", args)

        # --config may be a single path (generate_fractals) or a list of
        # paths (benchmark, action="append"): base config plus overrides.
        config_paths = args.config if isinstance(args.config, list) else [args.config]
        merged_dict = config_utils.load_config_files(config_paths)
        # Validate the merged result and derive dependent settings. Every run
        # parameter must be single-valued; a list is rejected here by name.
        bench_config = config_utils.Config(merged_dict)
        bench_config_dict = vars(bench_config)
        cli_args = vars(args)
        # Downstream consumers expect a single config path (e.g. to copy it
        # into the run dir); keep the base config there.
        cli_args["config"] = config_paths[0]

        # Combine configs: CLI args override config file values
        combined_config = bench_config_dict.copy()
        for key, value in cli_args.items():
            if key not in combined_config:
                combined_config[key] = value
            elif value is not None and key != "command":
                log.info(
                    "Overriding '%s=%s' with '%s=%s'",
                    key,
                    combined_config[key],
                    key,
                    value,
                )
                combined_config[key] = value

        # Recalculate unet_layers to capture any CLI overrides
        combined_config["unet_layers"] = (
            combined_config["problem_scale"] - combined_config["unet_bottleneck_dim"]
        )
        config_utils.require_positive_int(
            "n_categories", combined_config["n_categories"]
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

        if "fract_base_dir" in combined_config and combined_config["fract_base_dir"]:
            combined_config["fract_base_dir"] = str(
                Path(combined_config["fract_base_dir"]).resolve()
            )

        # Calculate these variables after override
        combined_config["vol_size"] = pow(2, combined_config["problem_scale"])
        combined_config["point_num"] = int(combined_config["vol_size"] ** 3 / 256)

        # Resolve the run directory and whether this launch resumes a run.
        # This sets combined_config["benchmark_run_dir"] on every path and,
        # when resuming, forces train_from_scratch off / restart on.
        benchmark_run_dir, restarting = resolve_run_dir(vars(args), combined_config)
        if restarting:
            log.info("Resuming in existing directory: %s", benchmark_run_dir)

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

        # 4. Generate/Update the restart script in the directory. The
        # communicator size is ground truth for the job scale; environment
        # sniffing is only the fallback for callers that lack it.
        create_restart_script(benchmark_run_dir, world_size=comm.Get_size())

    comm.Barrier()
    combined_config = comm.bcast(combined_config, root=0)

    # Restart pre-check. Like every other decision here it is made once, on
    # rank 0, and broadcast: the check reads the filesystem, and ranks can see
    # different views of a shared filesystem (stale NFS/Lustre attribute
    # caches). A rank that decided for itself would either abort alone --
    # stranding its peers in benchmark.py's timeout-less barrier -- or keep
    # running after rank 0 had already aborted.
    restart_precheck_error = None
    if combined_config.get("restart", False):
        if not combined_config.get("run_dir"):
            raise ValueError("--restart requires --run-dir")
        if rank == 0:
            restart_precheck_error = missing_checkpoint_error(combined_config)
    restart_precheck_error = comm.bcast(restart_precheck_error, root=0)
    if restart_precheck_error is not None:
        raise FileNotFoundError(restart_precheck_error)

    if rank == 0:
        log.debug("combined_config = %s", combined_config)

    if args.command == "benchmark":
        from ScaFFold import benchmark

        benchmark.main(kwargs_dict=combined_config)
    elif args.command == "generate_fractals":
        from ScaFFold import generate_fractals

        generate_fractals.main(kwargs_dict=combined_config)
    else:
        raise ValueError(
            f"Missing or invalid subcommand: {args.command}. Please consult ScaFFold documentation."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
