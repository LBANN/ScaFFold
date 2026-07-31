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


def run_cli(monkeypatch, argv, *, comm=None, sync_env=True):
    """Run ``cli.main`` with a fake communicator and stubbed subcommand drivers.

    ``sync_env`` makes the launcher environment agree with the fake
    communicator, which is what a correctly launched job looks like; tests of
    the mismatch check itself pass ``sync_env=False`` and set the environment
    themselves.

    Returns ``(comm, calls)`` where ``calls`` maps the subcommand name to the
    list of config dicts its driver was invoked with.
    """
    import ScaFFold.benchmark as benchmark_mod
    import ScaFFold.generate_fractals as generate_fractals_mod

    comm = comm if comm is not None else _FakeComm()
    calls = {"benchmark": [], "generate_fractals": []}

    if sync_env:
        monkeypatch.setenv("WORLD_SIZE", str(comm.Get_size()))
        monkeypatch.setenv("RANK", str(comm.Get_rank()))
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
            sync_env=False,
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
        sync_env=False,
    )

    assert len(calls["benchmark"]) == 1


# ---------------------------------------------------------------------------
# R17: the restart script is generated at the true job scale
# ---------------------------------------------------------------------------


def test_restart_script_gets_the_mpi_world_size(monkeypatch, tmp_path):
    """The CLI passes its communicator size to the restart-script generator.

    Without it the generator falls back to sniffing the environment, which
    misses launcher variables the rank side honors (e.g. PALS_NRANKS) and
    silently emits a single-process restart script for a multi-rank job.
    """
    recorded = {}

    def _recorder(run_dir, world_size=None):
        recorded["run_dir"] = run_dir
        recorded["world_size"] = world_size
        return Path(run_dir) / "restart.sh"

    monkeypatch.setattr(cli, "create_restart_script", _recorder)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    cfg = write_config(tmp_path)

    run_cli(
        monkeypatch,
        ["scaffold", "benchmark", "-c", str(cfg)],
        comm=_FakeComm(rank=0, size=4),
    )

    assert recorded["world_size"] == 4


# ---------------------------------------------------------------------------
# R18: the restart pre-check is a rank-0 decision, broadcast to everyone
# ---------------------------------------------------------------------------


def _restart_argv(cfg, run_dir):
    return [
        "scaffold",
        "benchmark",
        "-c",
        str(cfg),
        "--restart",
        "--run-dir",
        str(run_dir),
    ]


def _make_checkpoint(run_dir):
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "checkpoint_last.pth").write_bytes(b"")
    return ckpt_dir


def test_restart_without_checkpoint_is_rejected_on_rank0(monkeypatch, tmp_path):
    """Rank 0 still rejects a restart with no checkpoint, naming the paths."""
    cfg = write_config(tmp_path)
    run_dir = tmp_path / "prior"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        run_cli(monkeypatch, _restart_argv(cfg, run_dir), comm=_FakeComm(rank=0))

    assert "checkpoint_last.pth" in str(excinfo.value)


def test_restart_precheck_follows_the_broadcast_decision(monkeypatch, tmp_path):
    """A non-zero rank trusts rank 0's verdict instead of stat-ing the FS itself.

    Simulates a stale attribute cache: rank 0 saw the checkpoint and broadcast
    "go", while this rank's view of the shared filesystem shows nothing. A rank
    that re-decides locally raises alone and strands its peers in the next
    barrier.
    """
    cfg = write_config(tmp_path)
    run_dir = tmp_path / "prior"
    run_dir.mkdir()  # deliberately empty: this rank sees no checkpoint
    rank0_config = {
        "restart": True,
        "run_dir": str(run_dir),
        "checkpoint_dir": "checkpoints",
        "verbose": 0,
    }

    comm = _FakeComm(rank=1, size=2, bcast_returns=[rank0_config, None])
    _, calls = run_cli(monkeypatch, _restart_argv(cfg, run_dir), comm=comm)

    assert len(calls["benchmark"]) == 1


def test_restart_precheck_failure_raises_on_every_rank(monkeypatch, tmp_path):
    """Rank 0's rejection is broadcast, so non-zero ranks raise too.

    Here the local filesystem view *does* show a checkpoint; the rank must
    still fail, because rank 0 -- the only rank whose verdict counts -- did not
    find one. Otherwise the job splits: rank 0 aborts and the rest run on.
    """
    cfg = write_config(tmp_path)
    run_dir = tmp_path / "prior"
    run_dir.mkdir()
    _make_checkpoint(run_dir)
    rank0_config = {
        "restart": True,
        "run_dir": str(run_dir),
        "checkpoint_dir": "checkpoints",
        "verbose": 0,
    }
    rank0_error = "Restart requested but no checkpoint was found. Expected /nope."

    comm = _FakeComm(rank=1, size=2, bcast_returns=[rank0_config, rank0_error])
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        run_cli(monkeypatch, _restart_argv(cfg, run_dir), comm=comm)
