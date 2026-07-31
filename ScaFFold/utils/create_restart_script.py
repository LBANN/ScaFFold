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

# restart_script.py
from __future__ import annotations

import os
import shlex
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import List, Union

# Profiling toggles that must be reproduced on restart -- but only when they
# were active in the generating run. Names mirror ScaFFold.utils.perf_measure.
_PROFILING_ENV_VARS = ("PROFILE_TORCH", "CALI_CONFIG")

# Launcher variables carrying the total rank count, in the same priority order
# as ScaFFold.utils.distributed.get_world_size. The rank side and the restart
# generator must recognize the same set, or a job launched under a launcher
# only one of them knows about (e.g. Cray PALS) gets a restart script for the
# wrong number of ranks.
_WORLD_SIZE_ENV_VARS = (
    "WORLD_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PALS_NRANKS",
    "SLURM_NTASKS",
    "FLUX_JOB_SIZE",
)


def _rewrite_config_and_add_restart(cli_args: List[str]) -> List[str]:
    """
    Rewrite args for restart:
    1. Point --config to $RUN_DIR/config.yaml
    2. Remove --base-run-dir, --job-name (prevent new dir creation)
    3. Add --run-dir pointing to $RUN_DIR
    4. Ensure --restart is present
    """
    new_args = []
    skip_next = False

    # Args to strip because they trigger new directory creation or shouldn't change
    args_to_remove = {"--base-run-dir", "--job-name"}

    for i, tok in enumerate(cli_args):
        if skip_next:
            skip_next = False
            continue

        # Handle --config substitution
        if tok in ("-c", "--config"):
            new_args.append(tok)
            new_args.append("__CFG__")  # Placeholder for $RUN_DIR/config.yaml
            skip_next = True
            continue
        elif tok.startswith("--config="):
            new_args.append("--config=__CFG__")
            continue

        # Handle removal of directory creation args
        if tok in args_to_remove:
            skip_next = True  # Skip the value following the flag
            continue
        # Handle --arg=value format removal
        if any(tok.startswith(f"{x}=") for x in args_to_remove):
            continue

        new_args.append(tok)

    # Add the explicit resume flags
    if "--restart" not in new_args:
        new_args.append("--restart")

    # Point to the current directory (placeholder will be replaced by Bash variable)
    new_args.append("--run-dir")
    new_args.append("__RUN_DIR__")

    return new_args


def _substitute_placeholder(tok: str, var_subs: dict[str, str]) -> str:
    """Return ``tok`` with a placeholder replaced by its Bash expansion.

    A placeholder may be a whole token (``--config __CFG__``) or the value half
    of a combined token (``--config=__CFG__``); argparse accepts both spellings
    on the command line, so the rewriter can emit either. Anything else is
    shell-quoted verbatim.
    """
    if tok in var_subs:
        return var_subs[tok]  # e.g., "$RUN_DIR/config.yaml"
    flag, sep, value = tok.partition("=")
    if sep and value in var_subs:
        # --config=__CFG__ -> --config="$RUN_DIR/config.yaml"
        return shlex.quote(flag + sep) + var_subs[value]
    return shlex.quote(tok)


def _bash_array(var_name: str, argv: List[str], var_subs: dict[str, str]) -> str:
    """Render a Bash array declaration VAR=( ... ), safely quoted, with simple placeholder substitution."""
    parts = [_substitute_placeholder(tok, var_subs) for tok in argv]
    return f"{var_name}=( " + " ".join(parts) + " )"


def _detect_venv() -> str | None:
    """Return the path to the active virtualenv, or None if not in one.

    Prefers VIRTUAL_ENV (set by ``activate``); falls back to sys.prefix only
    when it differs from the base interpreter prefix (i.e. a venv is active).
    """
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return venv
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    if sys.prefix != base_prefix:
        return sys.prefix
    return None


def _get_env_setup(env: Mapping[str, str] | None = None) -> str:
    """Return a bash block that reproduces only the generating run's environment.

    Site-specific module loads and library paths are intentionally NOT emitted:
    the user's shell environment (modules, LD_PRELOAD, etc.) is responsible for
    those, and hardcoding one site's values makes the script abort or misbehave
    elsewhere. We only:

      * re-activate the virtualenv that was active when this script was
        generated, if one can be detected (non-fatal if the path is gone); and
      * re-export the profiling toggles (PROFILE_TORCH / CALI_CONFIG) that were
        set in the generating process, so a restarted run reproduces the
        original run's profiling behavior instead of silently flipping it on.
    """
    if env is None:
        env = os.environ

    lines = ["", "# --- Begin Environment Setup ---"]

    venv_path = _detect_venv()
    if venv_path is not None:
        activate = shlex.quote(f"{venv_path}/bin/activate")
        lines += [
            "# Re-activate the virtualenv that was active when this script was",
            "# generated. Non-fatal if it no longer exists on this machine.",
            f"if [ -f {activate} ]; then",
            f"    source {activate} || true",
            "else",
            f'    echo "WARNING: virtualenv activate script not found at {venv_path}/bin/activate" >&2',
            "fi",
        ]
    else:
        lines += [
            "# No virtualenv was active when this script was generated; the",
            "# caller's shell environment is responsible for the Python setup",
            "# (module loads, activation, LD_PRELOAD, etc.).",
        ]

    # Re-export profiling toggles only if the generating run had them set, so
    # the restart reproduces (rather than perturbs) the original run.
    for name in _PROFILING_ENV_VARS:
        if name in env:
            lines.append(f"export {name}={shlex.quote(env[name])}")

    lines.append("# --- End Environment Setup ---")
    lines.append("")
    return "\n".join(lines)


def _render_torchrun_hpc_restart(
    py_array_decl: str,
    captured_nodes: Union[str, int],
    captured_tasks_per_node: Union[str, int],
    env_setup: str,
) -> str:
    """
    Renders a unified restart script using torchrun-hpc.
    NOTE: captured_tasks_per_node maps to -n in torchrun-hpc.
    """
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail

# Directory containing this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
RUN_DIR="$SCRIPT_DIR"

{env_setup}

# --- Torchrun-HPC Configuration ---
# Use values captured when this script was generated.
# NODES = Total number of nodes (-N)
# TASKS_PER_NODE = Tasks per node (-n)
NODES="{captured_nodes}"
TASKS_PER_NODE="{captured_tasks_per_node}"
GPUS_PER_PROC="1" # Defaulting to 1, adjust if needed

# Additional torchrun-hpc arguments (e.g. --launcher-args for specific scheduler flags)
LAUNCHER_ADDITIONAL_ARGS=''

# Use a proper Bash array for arguments to handle paths with spaces safely
LAUNCHER_ARGS=(
    -l "$RUN_DIR"
    -N "$NODES" 
    -n "$TASKS_PER_NODE" 
    --gpus-per-proc "$GPUS_PER_PROC"
    $LAUNCHER_ADDITIONAL_ARGS
)

# Exact Python command to rerun the CLI
{py_array_decl}

echo "Restarting in $RUN_DIR via torchrun-hpc:"
echo "  torchrun-hpc ${{LAUNCHER_ARGS[*]}} ..."
printf '  python cmd: '; printf '%q ' "${{PY[@]}}"; echo

cd "$RUN_DIR"
# Invoking torchrun-hpc to handle scheduler interaction (Flux/Slurm)
exec torchrun-hpc "${{LAUNCHER_ARGS[@]}}" "${{PY[@]}}"
"""


def _render_local_restart(py_array_decl: str, env_setup: str) -> str:
    """Fallback for local restarts without torchrun-hpc."""
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
RUN_DIR="$SCRIPT_DIR"

{env_setup}

# Exact Python command to rerun the CLI
{py_array_decl}

echo "Restarting locally in $RUN_DIR:"
printf '  python cmd: '; printf '%q ' "${{PY[@]}}"; echo

cd "$RUN_DIR"
exec "${{PY[@]}}"
"""


def _sniff_launch_shape(env: Mapping[str, str]) -> tuple[int | None, int, int]:
    """Infer (nodes, total_tasks, world_size) from the launch environment.

    Reads, in priority order:
      1. Flux (FLUX_JOB_SIZE is total tasks, FLUX_JOB_NNODES is node count),
      2. Slurm (SLURM_NTASKS / SLURM_NPROCS total tasks, SLURM_*NODES nodes),
      3. generic launcher hints for the total rank count, in the same order
         and covering the same variables as
         ``ScaFFold.utils.distributed.get_world_size``: keeping the two in
         sync is what stops a restart script from relaunching the job at the
         wrong scale.

    ``nodes`` is None when the environment does not report a node count.
    ``world_size`` is the best available total-rank estimate (>= 1).
    """
    nodes: int | None = None
    total_tasks: int | None = None

    if env.get("FLUX_JOB_ID"):
        nodes = int(env.get("FLUX_JOB_NNODES", 1))
        total_tasks = int(env.get("FLUX_JOB_SIZE", 1))
    elif env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID"):
        nodes = int(env.get("SLURM_JOB_NUM_NODES") or env.get("SLURM_NNODES") or 1)
        total_tasks = int(env.get("SLURM_NTASKS") or env.get("SLURM_NPROCS") or 1)
    else:
        # No scheduler: fall back to generic launcher hints for the rank count.
        for key in _WORLD_SIZE_ENV_VARS:
            val = env.get(key)
            if val:
                total_tasks = int(val)
                break

    world_size = total_tasks if total_tasks else 1
    return nodes, (total_tasks or 1), world_size


def create_restart_script(run_dir: str | Path, world_size: int | None = None) -> Path:
    """Create ``run_dir/restart.sh`` for resuming this run.

    The launch shape (single-process vs. multi-rank) is derived from ground
    truth available at generation time. Callers that already know the true rank
    count (e.g. from ``MPI.COMM_WORLD.Get_size()``) should pass ``world_size``;
    it takes precedence over environment sniffing. When it is not given, the
    count is inferred from the scheduler / launcher environment (Flux, Slurm,
    torchrun, Open MPI, PMI). The multi-rank torchrun-hpc template is emitted
    whenever the resulting world size is greater than one; the local
    single-process template is used only for a world size of one.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Filter args to remove base-dir and add run-dir
    cli_args = _rewrite_config_and_add_restart(sys.argv[1:])

    # Detect Environment
    env = os.environ
    env_setup = _get_env_setup(env)

    # Detect current job scale from the launch environment.
    nodes, total_tasks, env_world_size = _sniff_launch_shape(env)

    # An explicit world_size (from the caller's communicator) is ground truth
    # and overrides whatever the environment reported.
    if world_size is not None:
        effective_world_size = int(world_size)
        # Keep total_tasks consistent so tasks-per-node math below is correct.
        total_tasks = effective_world_size
    else:
        effective_world_size = env_world_size

    use_torchrun = effective_world_size > 1

    if use_torchrun:
        py_cmd = [sys.argv[0]] + cli_args
    else:
        py_cmd = [sys.executable] + [sys.argv[0]] + cli_args

    # Create Bash array with placeholders
    py_array_decl = _bash_array(
        "PY",
        py_cmd,
        var_subs={"__CFG__": '"$RUN_DIR/config.yaml"', "__RUN_DIR__": '"$RUN_DIR"'},
    )

    if use_torchrun:
        # Calculate tasks per node for torchrun (-n arg).
        if nodes is None:
            nodes = 1
        tasks_per_node = max(1, total_tasks // nodes)

        script = _render_torchrun_hpc_restart(
            py_array_decl, nodes, tasks_per_node, env_setup
        )
    else:
        script = _render_local_restart(py_array_decl, env_setup)

    out_path = run_dir / "restart.sh"
    out_path.write_text(script, encoding="utf-8")
    out_path.chmod(out_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out_path
