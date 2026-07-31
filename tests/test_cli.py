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

"""Tests for the ``scaffold`` CLI entry point (``ScaFFold.cli.main``).

``cli.main`` makes every job-wide decision on MPI rank 0 and broadcasts it, so
these tests drive it with a *fake* communicator: the real ``MPI.COMM_WORLD`` in
this environment is always a one-rank singleton, which cannot express the
multi-rank shapes the CLI must get right (rank-0-decides-and-broadcasts, and
the mismatch between the MPI world and the launcher's environment).

``_FakeComm`` records what rank 0 broadcasts and, for a non-zero rank, replays
a scripted sequence of values as if rank 0 had sent them. That makes it
possible to assert that a non-zero rank *uses the broadcast decision* instead
of consulting its own view of the filesystem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

import ScaFFold.cli as cli

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "ScaFFold" / "configs" / "benchmark_default.yml"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class _FakeComm:
    """A stand-in for ``MPI.COMM_WORLD`` with a settable rank and size.

    ``bcast`` returns the caller's object on rank 0 (recording it); on a
    non-zero rank it pops the next scripted value from ``bcast_returns``,
    falling back to the caller's object when the script is exhausted.
    """

    def __init__(self, rank: int = 0, size: int = 1, bcast_returns=None):
        self._rank = rank
        self._size = size
        self._scripted = list(bcast_returns or [])
        self.broadcast = []
        self.barriers = 0

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def Barrier(self) -> None:
        self.barriers += 1

    def bcast(self, obj, root=0):
        self.broadcast.append(obj)
        if self._rank == root:
            return obj
        if self._scripted:
            return self._scripted.pop(0)
        return obj


class _FakeMPI:
    """Minimal ``mpi4py.MPI`` stand-in exposing only ``COMM_WORLD``."""

    def __init__(self, comm):
        self.COMM_WORLD = comm


def write_config(tmp_path, updates=None, name="bench.yml"):
    """Write a complete benchmark config into ``tmp_path`` and return its path."""
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config["base_run_dir"] = str(tmp_path / "runs")
    config["dataset_dir"] = str(tmp_path / "datasets")
    config["fract_base_dir"] = str(tmp_path / "fractals")
    if updates:
        config.update(updates)
    path = tmp_path / name
    path.write_text(yaml.dump(config))
    return path


def run_cli(monkeypatch, argv, *, comm=None):
    """Run ``cli.main`` with a fake communicator and stubbed subcommand drivers.

    Returns ``(comm, calls)`` where ``calls`` maps the subcommand name to the
    list of config dicts its driver was invoked with.
    """
    import ScaFFold.benchmark as benchmark_mod
    import ScaFFold.generate_fractals as generate_fractals_mod

    comm = comm if comm is not None else _FakeComm()
    calls = {"benchmark": [], "generate_fractals": []}

    monkeypatch.setattr(sys, "argv", list(argv))
    monkeypatch.setattr(cli, "MPI", _FakeMPI(comm))
    monkeypatch.setattr(
        benchmark_mod,
        "main",
        lambda kwargs_dict={}: calls["benchmark"].append(dict(kwargs_dict)),
    )
    monkeypatch.setattr(
        generate_fractals_mod,
        "main",
        lambda kwargs_dict={}: calls["generate_fractals"].append(dict(kwargs_dict)),
    )
    cli.main()
    return comm, calls


# ---------------------------------------------------------------------------
# R13: the MPI world must span the whole job
# ---------------------------------------------------------------------------


def test_mpi_singleton_under_multirank_launcher_aborts(monkeypatch, tmp_path):
    """A 1-rank MPI world inside a 2-rank launcher job aborts with an explanation.

    This is the plain-``torchrun`` shape: every process is an mpi4py singleton
    while the launcher says WORLD_SIZE=2. Left unchecked, every process runs
    cli.py's rank-0 block, claims its own run directory, and the job then hangs
    in the first real collective.
    """
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    cfg = write_config(tmp_path)

    with pytest.raises(RuntimeError) as excinfo:
        run_cli(
            monkeypatch,
            ["scaffold", "benchmark", "-c", str(cfg)],
            comm=_FakeComm(rank=0, size=1),
        )

    message = str(excinfo.value)
    assert "1" in message and "2" in message
    assert "torchrun" in message.lower()
    # The abort happens before any run directory is claimed.
    assert not (tmp_path / "runs").exists()


def test_matching_world_sizes_are_accepted(monkeypatch, tmp_path):
    """An MPI world that matches the launcher environment runs normally."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    cfg = write_config(tmp_path)

    _, calls = run_cli(
        monkeypatch,
        ["scaffold", "benchmark", "-c", str(cfg)],
        comm=_FakeComm(rank=0, size=4),
    )

    assert len(calls["benchmark"]) == 1


def test_no_launcher_env_is_not_a_mismatch(monkeypatch, tmp_path):
    """With no launcher variables set, the MPI world alone defines the size."""
    for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "SLURM_NTASKS", "FLUX_JOB_SIZE"):
        monkeypatch.delenv(var, raising=False)
    cfg = write_config(tmp_path)

    _, calls = run_cli(
        monkeypatch,
        ["scaffold", "benchmark", "-c", str(cfg)],
        comm=_FakeComm(rank=0, size=1),
    )

    assert len(calls["benchmark"]) == 1
