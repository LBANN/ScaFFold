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
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch

# We now primarily rely on torchrun-hpc, but might still detect scheduler for sizing
Sched = Literal["flux", "slurm", "local"]


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


def _bash_array(var_name: str, argv: List[str], var_subs: dict[str, str]) -> str:
    """Render a Bash array declaration VAR=( ... ), safely quoted, with simple placeholder substitution."""
    parts = []
    for tok in argv:
        if tok in var_subs:
            parts.append(var_subs[tok])  # e.g., "$RUN_DIR/config.yaml"
        else:
            parts.append(shlex.quote(tok))
    return f"{var_name}=( " + " ".join(parts) + " )"


def _get_env_setup() -> str:
    """Return the bash block that sets up the environment based on your stable configuration."""
    # Dynamically determine the current virtualenv path to reuse the active one
    venv_path = sys.prefix

    return f"""
# --- Begin Environment Setup ---
# Load Modules
if command -v module &> /dev/null; then
    module load rocm/6.4.2 rccl/fast-env-slows-mpi
fi

# Activate Virtual Environment
# (Using the one active when this script was generated)
if [ -f "{venv_path}/bin/activate" ]; then
    source "{venv_path}/bin/activate"
else
    echo "WARNING: Could not find venv activate script at {venv_path}/bin/activate"
fi

# Environment variables
export SPINDLE_FLUXOPT=off
export LD_PRELOAD=/opt/rocm-6.4.2/llvm/lib/libomp.so

export PROFILE_TORCH=ON
# --- End Environment Setup ---
"""


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


def create_restart_script(run_dir: str | Path) -> Path:
    """
    Create run_dir/restart.sh using torchrun-hpc.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Filter args to remove base-dir and add run-dir
    cli_args = _rewrite_config_and_add_restart(sys.argv[1:])

    # Detect Environment
    env = os.environ
    env_setup = _get_env_setup()

    # Detect current job scale
    nodes = None
    total_tasks = None

    # Try Flux (FLUX_JOB_SIZE is TOTAL tasks)
    if env.get("FLUX_JOB_ID"):
        nodes = int(env.get("FLUX_JOB_NNODES", 1))
        total_tasks = int(env.get("FLUX_JOB_SIZE", 1))
    # Try Slurm (SLURM_NTASKS is TOTAL tasks)
    elif env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID"):
        nodes = int(env.get("SLURM_JOB_NUM_NODES") or env.get("SLURM_NNODES") or 1)
        total_tasks = int(env.get("SLURM_NTASKS") or env.get("SLURM_NPROCS") or 1)

    # Determine if we should use torchrun-hpc
    is_hpc_env = (
        (env.get("FLUX_JOB_ID") is not None)
        or (env.get("SLURM_JOB_ID") is not None)
        or (env.get("SLURM_JOBID") is not None)
    )

    use_torchrun = is_hpc_env or torch.distributed.is_initialized()

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
        # Calculate tasks per node for torchrun (-n arg)
        tasks_per_node = 1
        if nodes and total_tasks:
            tasks_per_node = total_tasks // nodes
        if nodes is None:
            nodes = 1

        script = _render_torchrun_hpc_restart(
            py_array_decl, nodes, tasks_per_node, env_setup
        )
    else:
        script = _render_local_restart(py_array_decl, env_setup)

    out_path = run_dir / "restart.sh"
    out_path.write_text(script, encoding="utf-8")
    out_path.chmod(out_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out_path
