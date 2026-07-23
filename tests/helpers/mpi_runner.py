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

"""Subprocess launchers that turn distributed hangs into evidence.

Two launchers are provided:

* :func:`mpi_run` -- launch a rank script under a real MPI launcher
  (``mpirun``/``mpiexec``/``srun``/``flux``).  If no launcher is present (as in
  the sandboxed review environment) it raises ``pytest.skip`` so ``mpi``-marked
  tests are skipped rather than failing spuriously.

* :func:`torchrun_gloo` -- launch a rank script under ``torchrun`` with the
  gloo backend.  This works anywhere PyTorch is installed (no MPI launcher and
  no NCCL/RCCL needed) and is the workhorse for multi-process tests in this
  environment.

Both launchers wrap the user script so that every rank installs
``faulthandler.dump_traceback_later``.  When a collective deadlocks, each rank
prints its own stack to stderr *before* the harness kills it -- the deadlock
becomes a diagnosable traceback instead of an opaque CI hang.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

# Repo root == three levels up from this file (tests/helpers/mpi_runner.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Interpreter running the tests -- reused for the child ranks so they share the
# same virtualenv (torch, distconv, mpi4py, ...).
PYTHON = sys.executable

# torchrun living next to the active interpreter.
TORCHRUN = str(Path(PYTHON).with_name("torchrun"))

# Launchers we know how to drive, most-preferred first.
_MPI_LAUNCHERS = ("mpirun", "mpiexec", "srun", "flux")


def _faulthandler_preamble(timeout: int) -> str:
    """Source that dumps every rank's stack ``timeout-10`` seconds in.

    Installed at the top of each rank script so a stuck collective yields
    per-rank tracebacks on stderr instead of a silent hang.
    """
    dump_after = max(timeout - 10, 5)
    return textwrap.dedent(
        f"""\
        import faulthandler as _fh
        import sys as _sys
        _fh.enable()
        # Emit per-rank stacks shortly before the harness timeout fires so a
        # deadlocked collective is visible on stderr.
        _fh.dump_traceback_later({dump_after}, repeat=True, file=_sys.stderr)
        """
    )


def _wrap_script(script: str, timeout: int) -> str:
    """Return source that installs the faulthandler preamble then runs ``script``."""
    script_path = Path(script).resolve()
    return _faulthandler_preamble(timeout) + textwrap.dedent(
        f"""\
            import runpy as _runpy
            _runpy.run_path({str(script_path)!r}, run_name="__main__")
            """
    )


def _child_env(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Base environment for child ranks: repo on PYTHONPATH, gloo-friendly."""
    child_env = os.environ.copy()
    existing = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )
    # gloo cannot always resolve the hostname on HPC login nodes; loopback is
    # correct for single-node multi-process runs.
    child_env.setdefault("GLOO_SOCKET_IFNAME", "lo")
    if env:
        child_env.update(env)
    return child_env


def _free_port() -> int:
    """Bind an ephemeral port, release it, and return the number.

    Small TOCTOU window, but adequate for launching a local rendezvous.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def detect_mpi_launcher() -> Optional[str]:
    """Return the first available MPI launcher on PATH, or ``None``."""
    for name in _MPI_LAUNCHERS:
        if shutil.which(name):
            return name
    return None


def _launcher_argv(launcher: str, n: int) -> List[str]:
    """Build the launcher-specific argv prefix for ``n`` ranks."""
    if launcher in ("mpirun", "mpiexec"):
        argv = [launcher, "--oversubscribe", "-np", str(n)]
        # OpenMPI refuses to run as root without this flag; harmless elsewhere
        # is not guaranteed, so only add it when actually running as root.
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            argv.append("--allow-run-as-root")
        return argv
    if launcher == "srun":
        return [launcher, "-n", str(n)]
    if launcher == "flux":
        return [launcher, "run", "-n", str(n)]
    raise ValueError(f"Unsupported launcher: {launcher}")


def mpi_run(
    script: str,
    n: int = 2,
    timeout: int = 60,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    """Run ``script`` under a real MPI launcher with ``n`` ranks.

    Detects an available launcher (``mpirun``/``mpiexec``/``srun``/``flux``).
    When none is available -- as in this sandbox -- raises ``pytest.skip`` with
    the message ``"no MPI launcher available"`` so ``mpi``-marked tests are
    skipped cleanly instead of failing.

    Each rank installs ``faulthandler.dump_traceback_later`` so a deadlock
    produces per-rank stack traces on stderr before the ``timeout`` kill.

    Returns ``(returncode, stdout, stderr)``. On timeout, returncode is a
    negative/nonzero value and stderr contains the dumped tracebacks.
    """
    launcher = detect_mpi_launcher()
    if launcher is None:
        pytest.skip("no MPI launcher available")

    wrapped = _wrap_script(script, timeout)
    argv = _launcher_argv(launcher, n) + [PYTHON, "-c", wrapped]

    try:
        proc = subprocess.run(
            argv,
            env=_child_env(env),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return -1, stdout, stderr

    return proc.returncode, proc.stdout, proc.stderr


def torchrun_gloo(
    script: str,
    n: int = 2,
    timeout: int = 120,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    """Run ``script`` under ``torchrun`` with ``n`` ranks and the gloo backend.

    This works in the sandbox (no MPI launcher, no NCCL/RCCL required). The
    script is expected to call ``torch.distributed.init_process_group`` with no
    explicit backend (or ``backend="gloo"``); ``torchrun`` supplies ``RANK``,
    ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR`` and ``MASTER_PORT`` via the
    environment.

    A free ``master_port`` is chosen by binding an ephemeral socket. Each rank
    installs ``faulthandler.dump_traceback_later`` so deadlocks surface as
    per-rank stacks on stderr.

    Returns ``(returncode, stdout, stderr)``; on timeout returncode is ``-1``.
    """
    if not os.path.exists(TORCHRUN):
        pytest.skip(f"torchrun not found at {TORCHRUN}")

    wrapped = _wrap_script(script, timeout)
    port = _free_port()
    child_env = _child_env(env)
    # Force gloo everywhere the process group is created without an explicit
    # backend -- the sandbox has no working NCCL/RCCL.
    child_env.setdefault("PL_TORCH_DISTRIBUTED_BACKEND", "gloo")

    # torchrun takes a script path (not ``-c``), so materialize the wrapped
    # source to a temp file for the duration of the run.
    tmp_dir = child_env.get("CLAUDE_CODE_TMPDIR") or tempfile.gettempdir()
    fd, wrapper_path = tempfile.mkstemp(suffix="_torchrun_wrapper.py", dir=tmp_dir)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(wrapped)

        argv = [
            TORCHRUN,
            "--nnodes=1",
            f"--nproc_per_node={n}",
            f"--master_port={port}",
            "--master_addr=127.0.0.1",
            wrapper_path,
        ]

        try:
            proc = subprocess.run(
                argv,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(REPO_ROOT),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return -1, stdout, stderr
    finally:
        try:
            os.unlink(wrapper_path)
        except OSError:
            pass

    return proc.returncode, proc.stdout, proc.stderr
