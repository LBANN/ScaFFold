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
import builtins
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


def explicit_cli_keys(args, parsers):
    """Return the names of the options actually given on the command line.

    argparse does not record which options were supplied, so a value counts as
    explicit when it differs from the default of the first parser in
    ``parsers`` that defines one (subcommand parser first, then the top-level
    parser). Only these may outrank a config-file setting; everything else in
    the namespace is an argparse default, which is the weakest source.

    The one ambiguity is a flag passed with exactly its default value: it looks
    absent, so a config-file entry wins over it. Both spellings then agree on
    the default, which is the only value the flag could have contributed.
    """
    explicit = set()
    for name, value in vars(args).items():
        default = None
        for parser in parsers:
            default = parser.get_default(name)
            if default is not None:
                break
        if value != default:
            explicit.add(name)
    return explicit


def missing_checkpoint_error(combined_config):
    """Return the "nothing to resume from" message, or None if a restart can run.

    Reports the first problem found rather than raising, so the caller can make
    this a rank-0 decision and broadcast the verdict instead of letting every
    rank stat the shared filesystem and possibly disagree.

    The directory checked is the *resolved* benchmark run dir -- the one this
    launch will actually train in -- and not the raw ``run_dir`` key. The two
    can differ: a config.yaml dumped by a restarted run carries that run's
    ``run_dir``, so reusing it as a base config had this check stat another
    run's checkpoints, pass, and let the job die hours later with its dataset
    already generated. ``resolve_run_dir`` keeps the two in agreement; this
    prefers the resolved value so they cannot drift apart again.
    """
    run_dir = combined_config.get("benchmark_run_dir") or combined_config.get("run_dir")
    if not run_dir:
        return (
            "Restart requested but no run directory was resolved. Pass "
            "'--run-dir <path>' (or set run_dir in the config file)."
        )
    checkpoint_dir = Path(run_dir) / combined_config.get(
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

    * a run directory (``--run-dir DIR``, or ``run_dir`` in the config file),
      with or without a restart flag: resume in that exact directory.
      ``train_from_scratch`` is forced off and ``restart`` on so the downstream
      benchmark driver takes its restart path.
    * a restart requested with no run directory anywhere: rejected with a clear
      error, before any directory is created. The directory to resume must be
      named explicitly; the most recent one is never guessed.
    * neither: create a fresh timestamped directory under ``base_run_dir``,
      retrying with a numeric suffix on a same-second name collision.

    Both keys are read from the *merged* config, not from the command line
    alone. A config file is a first-class source for them -- the generated
    ``restart.sh`` replays a dumped ``config.yaml``, which carries both -- and
    an absent ``--restart`` cannot outrank a file that sets it, because an
    unset ``store_true`` flag is indistinguishable from its default. Reading
    only the command line here let the two disagree: a config-file restart
    created a *fresh* run directory and only then failed for want of a run dir,
    and a stale ``run_dir`` inherited from a reused ``config.yaml`` was left in
    the config for the restart pre-check to stat while training happened
    somewhere else entirely.

    The resolved answer is written back to ``benchmark_run_dir``, ``restart``
    and ``run_dir``, so every later reader -- the pre-check, the dumped
    ``config.yaml``, the benchmark driver -- sees exactly what was decided here.
    Returns ``(benchmark_run_dir: Path, restarting: bool)``.
    """
    restart_flag = bool(combined_config.get("restart") or args_dict.get("restart"))
    run_dir_arg = combined_config.get("run_dir") or args_dict.get("run_dir")

    if run_dir_arg is not None:
        benchmark_run_dir = Path(run_dir_arg)
        # An explicit run dir is expected to already exist; tolerate a missing
        # one but never treat a pre-existing dir as an error here.
        benchmark_run_dir.mkdir(parents=True, exist_ok=True)
        restarting = True
    elif restart_flag:
        raise ValueError(
            "A restart was requested (--restart, or restart: true in the "
            "config file) but no run directory was given: pass the directory "
            "of the run to resume (e.g. '--restart --run-dir <path>'). The "
            "most recent run directory is not resolved automatically."
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

    # Write the resolution back, so nothing downstream can re-derive a
    # different answer from the raw inputs. ``run_dir`` is cleared for a fresh
    # run: left set, a value inherited from a reused config.yaml names a
    # directory this run has nothing to do with.
    combined_config["benchmark_run_dir"] = str(benchmark_run_dir)
    combined_config["restart"] = restarting
    combined_config["run_dir"] = str(benchmark_run_dir) if restarting else None
    if restarting:
        combined_config["train_from_scratch"] = False
    return benchmark_run_dir, restarting


def rebuild_error(type_name, message):
    """Rebuild rank 0's configuration error on a peer that never saw it.

    Only the type name and message cross the wire: an arbitrary exception
    object may not survive a pickle round trip, and a failure to unpickle on
    the receiving side would turn the error being reported back into the hang
    it was reported to avoid. Builtin exception types are rebuilt as
    themselves, so a caller's ``except ValueError`` still catches what rank 0
    raised; anything else degrades to ``RuntimeError`` naming the original
    type.
    """
    cls = getattr(builtins, type_name, None)
    if isinstance(cls, type) and issubclass(cls, Exception):
        return cls(message)
    return RuntimeError(f"{type_name}: {message}")


def build_run_config(args, parsers, log, world_size):
    """Build the job-wide config, and lay down the benchmark's run directory.

    Rank-0-only work: it reads and merges the config files, applies the
    command-line overrides, validates the result, and -- for the benchmark
    subcommand -- resolves the run directory, dumps the configs into it and
    writes its restart script. Returns the merged config the caller broadcasts.

    Raises whatever the validation or the filesystem raises; the caller runs
    this inside a guard and broadcasts the outcome, since every peer is already
    waiting in the barrier that follows.
    """
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

    # Combine configs, in increasing order of precedence:
    # argparse default < config file < explicit command-line flag.
    combined_config = bench_config_dict.copy()
    # Config only keeps the keys it consumes; the auxiliary keys it accepts
    # (verbose, datagen_batch_size, ...) never become attributes, so put
    # the file's values back first. Without this they are absent below and
    # the argparse default overwrites what the user wrote in the config.
    for key, value in merged_dict.items():
        combined_config.setdefault(key, value)

    explicit_cli = explicit_cli_keys(args, parsers)
    for key, value in cli_args.items():
        if key == "command":
            continue
        if key not in combined_config:
            combined_config[key] = value
        elif key in explicit_cli and value is not None:
            log.info(
                "Overriding '%s=%s' with '%s=%s'",
                key,
                combined_config[key],
                key,
                value,
            )
            combined_config[key] = value
    # The subcommand is always owned by the command line.
    combined_config["command"] = cli_args["command"]

    # Recalculate unet_layers to capture any CLI overrides. The overridden
    # pair has to be re-validated: Config only saw the config-file values.
    config_utils.validate_unet_dims(
        combined_config["problem_scale"], combined_config["unet_bottleneck_dim"]
    )
    combined_config["unet_layers"] = (
        combined_config["problem_scale"] - combined_config["unet_bottleneck_dim"]
    )
    config_utils.require_positive_int("n_categories", combined_config["n_categories"])

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

    # The run directory, its config dumps and its restart script belong to
    # the benchmark subcommand alone. Fractal generation writes nothing
    # there, and the restart script it used to get replayed
    # `generate_fractals --restart --run-dir ...` -- flags that subparser
    # rejects, so the script could only ever exit 2.
    if args.command == "benchmark":
        # Resolve the run directory and whether this launch resumes a run.
        # This sets combined_config["benchmark_run_dir"] on every path, and
        # writes the resolved restart/run_dir back into the config.
        benchmark_run_dir, restarting = resolve_run_dir(cli_args, combined_config)
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
        create_restart_script(benchmark_run_dir, world_size=world_size)

    return combined_config


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
    # Parse the command-line arguments.
    args = parser.parse_args()
    # Every rank runs this identically, before any run directory is created, so
    # a mis-launched job aborts uniformly instead of leaving per-rank run dirs
    # behind and hanging. It runs *after* parsing so that the arguments argparse
    # handles by itself -- ``--help``, a usage error -- still behave: asking
    # what the flags are is not a job launch, and answering it with a launcher
    # mismatch (which is exactly the environment someone debugging one is
    # sitting in) helps nobody.
    check_launcher_world_size(comm.Get_size())
    subcommand_parsers = {
        "benchmark": benchmark_parser,
        "generate_fractals": generate_fractals_parser,
    }
    active_parser = subcommand_parsers[args.command]
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

    # Rank 0 builds the config for the whole job while every other rank waits
    # in the barrier below. Everything in there can fail on user input (an
    # unknown config key, an out-of-range bottleneck, a restart with no run
    # dir) or on the filesystem, and a rank-0-only raise leaves the peers
    # blocked in that barrier -- a hang instead of the error message the user
    # needs. The outcome is therefore broadcast, exactly like the restart
    # pre-check below, and every rank raises the same error together.
    config_error = None
    rank0_error = None
    if rank == 0:
        try:
            combined_config = build_run_config(
                args, (active_parser, parser), log, comm.Get_size()
            )
        except Exception as e:
            combined_config = None
            rank0_error = e
            config_error = (type(e).__name__, str(e))

    comm.Barrier()
    config_error = comm.bcast(config_error, root=0)
    if config_error is not None:
        # Rank 0 re-raises the original (keeping its traceback); the peers
        # rebuild it from what crossed the wire.
        raise rank0_error if rank0_error is not None else rebuild_error(*config_error)
    combined_config = comm.bcast(combined_config, root=0)

    # Restart pre-check. Like every other decision here it is made once, on
    # rank 0, and broadcast: the check reads the filesystem, and ranks can see
    # different views of a shared filesystem (stale NFS/Lustre attribute
    # caches). A rank that decided for itself would either abort alone --
    # stranding its peers in benchmark.py's timeout-less barrier -- or keep
    # running after rank 0 had already aborted.
    # Only the benchmark subcommand has run directories or checkpoints, so it
    # is the only one this applies to: fractal generation reading a benchmark's
    # config.yaml must not be judged on that run's restart state.
    restart_precheck_error = None
    if args.command == "benchmark" and combined_config.get("restart", False):
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
