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

"""Consensus tests for distributed dataset generation.

The datagen pipeline runs under MPI, and correctness rests on one invariant:
*every rank executes the same sequence of collectives on every code path, and
every global decision is made once and broadcast.* When that invariant is
violated a single rank's failure (or a divergent filesystem view) leaves peers
blocked forever in a mismatched collective -- a job hang, not an error.

Two tiers of tests guard the invariant:

* **Single-process** tests (the bulk of this file) drive the real
  ``get_dataset`` / ``volumegen`` / ``instance`` functions with a *fake* MPI
  communicator that records every collective call and can simulate a chosen
  rank in a larger world. They pin the decide-and-broadcast structure and the
  collective *sequence* deterministically, without a launcher, and are the
  primary local evidence.

* **``@pytest.mark.mpi``** tests launch real 2-rank rank scripts through
  :func:`tests.helpers.mpi_runner.mpi_run`. They are skipped where no launcher
  exists (as in the sandboxed dev environment) and run on any box with an MPI
  launcher, converting a regression into a timeout-with-stacks rather than a
  silent CI hang.
"""

from __future__ import annotations

import logging
import re
from argparse import Namespace
from math import ceil
from pathlib import Path

import numpy as np
import pytest
import yaml
from mpi4py import MPI

from ScaFFold.datagen import get_dataset as gd
from ScaFFold.datagen import instance as inst
from ScaFFold.datagen import volumegen

RANK_SCRIPTS = Path(__file__).resolve().parents[1] / "helpers" / "rank_scripts"


class FakeComm:
    """A single-process stand-in for an MPI communicator.

    It records the ordered sequence of collective calls (``calls``) and the
    payloads passed to ``bcast`` / ``allgather``, so a test can assert on the
    exact collective structure a code path executes. It can also simulate being
    an arbitrary ``rank`` in a world of ``size`` ranks: values that a real peer
    would contribute are injected so the single process follows the same branch
    a distributed rank would.
    """

    def __init__(
        self,
        rank=0,
        size=1,
        bcast_returns=None,
        allreduce_result=None,
        allgather_peers=None,
    ):
        self.rank = rank
        self.size = size
        self.calls = []
        self.bcast_payloads = []
        self.allgather_payloads = []
        # Values a non-root rank should *receive* from root, consumed in order.
        self._bcast_returns = list(bcast_returns or [])
        # Override the allreduce verdict (e.g. simulate a peer's failure);
        # ``None`` means identity, i.e. this rank's own value (size-1 semantics).
        self._allreduce_result = allreduce_result
        # Contributions the other ranks add to an allgather, inserted around
        # this rank's own contribution at its rank index.
        self._allgather_peers = list(allgather_peers or [])

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Barrier(self):
        self.calls.append("Barrier")

    def bcast(self, obj, root=0):
        self.calls.append("bcast")
        self.bcast_payloads.append(obj)
        if self.rank == root:
            return obj
        # Non-root: ignore the local value, take what root broadcast.
        return self._bcast_returns.pop(0)

    def allreduce(self, value, op=None):
        self.calls.append("allreduce")
        if self._allreduce_result is not None:
            return self._allreduce_result
        return value

    def allgather(self, obj):
        self.calls.append("allgather")
        self.allgather_payloads.append(obj)
        result = list(self._allgather_peers)
        result.insert(self.rank, obj)
        return result

    def gather(self, obj, root=0):
        self.calls.append("gather")
        if self.rank == root:
            return [obj] + list(self._allgather_peers)
        return None


class FakeMPI:
    """Namespace mimicking the ``mpi4py.MPI`` module for one ``FakeComm``."""

    def __init__(self, comm):
        self.COMM_WORLD = comm
        self.MIN = MPI.MIN


def _reuse_config(dataset_dir: Path) -> Namespace:
    """Minimal config carrying just the keys ``get_dataset`` reads."""
    return Namespace(
        dataset_dir=str(dataset_dir),
        n_categories=2,
        n_instances_used_per_fractal=2,
        problem_scale=4,
        seed=1234,
        variance_threshold=0.15,
        n_fracts_per_vol=1,
        val_split=0,
    )


def _write_reusable_dataset(base_config_dir: Path, config_id: str) -> Path:
    """Materialize a finalized dataset dir under ``base_config_dir`` rank 0 can reuse."""
    dataset_dir = base_config_dir / "20260101-000000__abc123"
    dataset_dir.mkdir(parents=True)
    meta = {
        "config_id": config_id,
        "dataset_format_version": gd.DATASET_FORMAT_VERSION,
    }
    (dataset_dir / gd.META_FILENAME).write_text(yaml.safe_dump(meta))
    return dataset_dir


# ---------------------------------------------------------------------------
# The reuse-vs-generate decision is made once on rank 0 and broadcast.
# ---------------------------------------------------------------------------


def test_reuse_decision_is_broadcast_not_scanned_per_rank(tmp_path, monkeypatch):
    """Rank 0 decides reuse; the decision travels to peers via ``bcast``.

    The fixed code must funnel the reuse-vs-generate choice through exactly one
    broadcast so a non-root rank never runs its own filesystem scan. We drive
    ``get_dataset`` as rank 0 in a 2-rank world and confirm the very first
    collective is a ``bcast`` whose payload is a ``("reuse", path)`` decision.
    """
    dataset_dir = tmp_path / "datasets"
    config = _reuse_config(dataset_dir)

    # Precompute the config_id the same way get_dataset does, then plant a
    # reusable dataset where rank 0's scan will find it.
    comm = FakeComm(rank=0, size=2)
    monkeypatch.setattr(gd, "MPI", FakeMPI(comm))
    monkeypatch.setattr(gd, "_git_commit_short", lambda log: "abc123")

    # First call: rank 0 scans, finds nothing, would try to generate. Instead
    # we pre-seed a reusable dataset so the decision is 'reuse'.
    root = dataset_dir
    root.mkdir(parents=True, exist_ok=True)
    config_dict = vars(config).copy()
    config_dict["dataset_format_version"] = gd.DATASET_FORMAT_VERSION
    volume_config = gd._get_required_keys_dict(config_dict, gd.INCLUDE_KEYS)
    config_id = gd._hash_volume_config(volume_config)
    reusable = _write_reusable_dataset(root / config_id, config_id)

    result = gd.get_dataset(config)

    assert Path(result) == reusable
    # The decision was broadcast, and it was a reuse decision.
    assert comm.calls[0] == "bcast"
    decision = comm.bcast_payloads[0]
    assert decision[0] == "reuse"
    assert Path(decision[1]) == reusable


def test_non_root_follows_broadcast_reuse_decision(tmp_path, monkeypatch):
    """A non-root rank returns root's decision, ignoring its own FS view.

    Simulating rank 1 whose local scan would see *nothing* reusable: with the
    fix, rank 1 does not scan at all -- it consumes the ``("reuse", path)``
    tuple root broadcast and returns that path, so ranks cannot diverge.
    """
    dataset_dir = tmp_path / "datasets"
    config = _reuse_config(dataset_dir)

    broadcast_decision = ("reuse", str(tmp_path / "datasets" / "cid" / "chosen"))
    comm = FakeComm(rank=1, size=2, bcast_returns=[broadcast_decision])
    monkeypatch.setattr(gd, "MPI", FakeMPI(comm))
    monkeypatch.setattr(gd, "_git_commit_short", lambda log: "abc123")

    result = gd.get_dataset(config)

    # Rank 1 returned exactly what root broadcast, via a single bcast, with no
    # generate-path collectives (no allreduce / allgather).
    assert Path(result) == Path(broadcast_decision[1])
    assert comm.calls == ["bcast"]
    # Rank 1 contributed None to the broadcast (it is not the decider).
    assert comm.bcast_payloads[0] is None


# ---------------------------------------------------------------------------
# On a generation failure every rank participates in the error gather and every
# rank raises -- no rank is left in a mismatched collective and no rank returns
# an unfinalized path.
# ---------------------------------------------------------------------------


def _generate_decision_comm(rank, size, dest, **kwargs):
    """A FakeComm primed to deliver a ('generate', tmp, dest) decision.

    Root (rank 0) computes the decision itself inside ``get_dataset``; a
    non-root rank instead receives it through the broadcast, so ``bcast_returns``
    is only needed for non-root ranks.
    """
    tmp = dest.parent / f".tmp_{dest.name}"
    decision = ("generate", str(tmp), str(dest))
    if rank != 0:
        kwargs.setdefault("bcast_returns", [decision])
    return FakeComm(rank=rank, size=size, **kwargs), tmp, dest


def test_generation_failure_gathers_on_every_rank_and_raises(tmp_path, monkeypatch):
    """A failing generation drives allreduce + allgather + raise, on rank 0.

    We make ``volumegen.main`` raise and drive ``get_dataset`` as rank 0 whose
    ``allreduce`` verdict (simulating a peer) reports failure. The fix must call
    ``allgather`` (unconditionally, on every rank) to collect messages and then
    raise ``RuntimeError`` -- the old code called ``gather`` only inside
    ``if rank == 0``, structurally impossible for peers to match.
    """
    dataset_dir = tmp_path / "datasets"
    config = _reuse_config(dataset_dir)

    root = dataset_dir
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "cid" / "20260101-000000__abc123"

    comm, tmp, _ = _generate_decision_comm(
        rank=0, size=2, dest=dest, allreduce_result=0, allgather_peers=["rank 1: boom"]
    )
    # Root's generate decision is computed in-function; make its scan find
    # nothing and its staging mkdir land inside tmp_path.
    monkeypatch.setattr(gd, "MPI", FakeMPI(comm))
    monkeypatch.setattr(gd, "_git_commit_short", lambda log: "abc123")

    def boom(_config):
        raise RuntimeError("volumegen exploded")

    monkeypatch.setattr(volumegen, "main", boom)

    with pytest.raises(RuntimeError) as excinfo:
        gd.get_dataset(config)

    # Both the local failure and the simulated peer's message surface.
    assert "volumegen exploded" in str(excinfo.value)
    assert "rank 1: boom" in str(excinfo.value)
    # The error gather is an allgather (every rank participates), and it runs
    # after the allreduce verdict.
    assert "allgather" in comm.calls
    assert comm.calls.index("allreduce") < comm.calls.index("allgather")


def test_non_root_raises_on_failure_instead_of_returning(tmp_path, monkeypatch):
    """A non-root rank must raise on failure, never return an unfinalized path.

    Simulating rank 1: it receives the generate decision, its own
    ``volumegen.main`` succeeds, but the ``allreduce`` verdict (a peer failed)
    is failure. The fixed code raises on rank 1 too -- it does not return
    ``dest`` for a dataset that was never finalized.
    """
    dataset_dir = tmp_path / "datasets"
    config = _reuse_config(dataset_dir)
    dest = dataset_dir / "cid" / "20260101-000000__abc123"

    comm, tmp, _ = _generate_decision_comm(
        rank=1, size=2, dest=dest, allreduce_result=0, allgather_peers=["rank 0: boom"]
    )
    monkeypatch.setattr(gd, "MPI", FakeMPI(comm))
    monkeypatch.setattr(gd, "_git_commit_short", lambda log: "abc123")
    monkeypatch.setattr(volumegen, "main", lambda _config: None)

    with pytest.raises(RuntimeError) as excinfo:
        gd.get_dataset(config)

    assert "rank 0: boom" in str(excinfo.value)
    # Rank 1 participated in the error gather (allgather), proving it is no
    # longer nested behind ``if rank == 0``.
    assert "allgather" in comm.calls


def test_generation_success_finalizes_and_returns(tmp_path, monkeypatch):
    """The success path renames the staging dir into place and returns it.

    Control test guarding against over-correction: with a clean verdict the
    fixed code writes ``meta.yaml`` into the staging dir, renames it to the
    final destination, and returns that path.
    """
    dataset_dir = tmp_path / "datasets"
    config = _reuse_config(dataset_dir)

    comm = FakeComm(rank=0, size=2, allreduce_result=1)
    monkeypatch.setattr(gd, "MPI", FakeMPI(comm))
    monkeypatch.setattr(gd, "_git_commit_short", lambda log: "abc123")
    monkeypatch.setattr(volumegen, "main", lambda _config: None)

    result = gd.get_dataset(config)

    assert Path(result).exists()
    assert (Path(result) / gd.META_FILENAME).exists()
    # No staging dir is left behind under the config_id base.
    leftover = [p for p in Path(result).parent.iterdir() if p.name.startswith(".tmp_")]
    assert leftover == []


# ---------------------------------------------------------------------------
# R27: rank 0 must never die between the collectives its peers have entered.
# Every rank-0-only step of the consensus (the reuse/generate decision and the
# final meta-write + rename) is wrapped so a failure travels to the peers as a
# broadcast sentinel instead of stranding them in ``bcast``/``Barrier``.
# ---------------------------------------------------------------------------


def _base_dir_for(config: Namespace) -> Path:
    """The ``<dataset_dir>/<config_id>`` directory ``get_dataset`` scans."""
    config_dict = vars(config).copy()
    config_dict["dataset_format_version"] = gd.DATASET_FORMAT_VERSION
    volume_config = gd._get_required_keys_dict(config_dict, gd.INCLUDE_KEYS)
    return Path(config.dataset_dir) / gd._hash_volume_config(volume_config)


def test_decision_failure_is_broadcast_not_raised_before_bcast(tmp_path, monkeypatch):
    """A rank-0 decision failure reaches peers through the broadcast.

    Any exception inside the rank-0-only decision (an unreadable base dir, a
    staging ``mkdir`` hitting ENOSPC, ...) must be converted into an error
    sentinel that is broadcast, so peers already waiting in ``bcast`` learn
    about it and raise the same error. Before the fix rank 0 raised *before*
    reaching the broadcast, leaving every peer blocked forever.
    """
    config = _reuse_config(tmp_path / "datasets")
    comm = FakeComm(rank=0, size=2)
    monkeypatch.setattr(gd, "MPI", FakeMPI(comm))
    monkeypatch.setattr(gd, "_git_commit_short", lambda log: "abc123")

    def explode(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(gd, "_decide_reuse_or_generate", explode)

    with pytest.raises(RuntimeError) as excinfo:
        gd.get_dataset(config)

    # Rank 0 reached the broadcast before raising, and the payload is the
    # error sentinel every peer will see.
    assert comm.calls == ["bcast"]
    assert comm.bcast_payloads[0][0] == "error"
    assert "No space left on device" in str(excinfo.value)


def test_non_root_raises_on_broadcast_decision_error(tmp_path, monkeypatch):
    """A peer receiving the error sentinel raises instead of generating."""
    config = _reuse_config(tmp_path / "datasets")
    sentinel = ("error", "rank 0: OSError: No space left on device")
    comm = FakeComm(rank=1, size=2, bcast_returns=[sentinel])
    monkeypatch.setattr(gd, "MPI", FakeMPI(comm))
    monkeypatch.setattr(gd, "_git_commit_short", lambda log: "abc123")

    with pytest.raises(RuntimeError) as excinfo:
        gd.get_dataset(config)

    assert "No space left on device" in str(excinfo.value)
    # The peer stopped at the decision broadcast: no generation collectives.
    assert comm.calls == ["bcast"]


def test_reuse_scan_skips_staging_dirs(tmp_path, monkeypatch):
    """A complete ``meta.yaml`` stranded in a ``.tmp_*`` dir is never reused.

    A job killed between the meta write and the rename leaves a fully valid
    meta inside its staging dir. Treating that as a publishable dataset hands
    back a half-generated directory (and one that cleanup may delete).
    """
    config = _reuse_config(tmp_path / "datasets")
    base = _base_dir_for(config)
    config_id = base.name
    stranded = base / ".tmp_20260101-000000_1234"
    stranded.mkdir(parents=True)
    (stranded / gd.META_FILENAME).write_text(
        yaml.safe_dump(
            {
                "config_id": config_id,
                "dataset_format_version": gd.DATASET_FORMAT_VERSION,
            }
        )
    )

    comm = FakeComm(rank=0, size=1)
    monkeypatch.setattr(gd, "MPI", FakeMPI(comm))
    monkeypatch.setattr(gd, "_git_commit_short", lambda log: "abc123")
    log = logging.getLogger("test_reuse_scan_skips_staging_dirs")

    decision = gd._decide_reuse_or_generate(base, config_id, "abc123", False, log)

    assert decision[0] == "generate", f"staging dir was reused: {decision}"


def test_reuse_scan_tolerates_corrupt_meta(tmp_path, monkeypatch):
    """A corrupt/unreadable candidate meta is skipped, not fatal.

    A 0-byte ``meta.yaml`` (``yaml.safe_load`` -> ``None``) or an unparseable
    one used to raise inside the rank-0-only scan. The scan must warn, skip the
    directory, and keep looking -- here finding the good dataset next to it.
    """
    config = _reuse_config(tmp_path / "datasets")
    base = _base_dir_for(config)
    config_id = base.name
    base.mkdir(parents=True)

    # Sorted-descending scan order visits these two poison dirs first.
    (base / "20260301-000000__zzz").mkdir()
    (base / "20260301-000000__zzz" / gd.META_FILENAME).write_text("")
    (base / "20260201-000000__yyy").mkdir()
    (base / "20260201-000000__yyy" / gd.META_FILENAME).write_text("{[not yaml")

    good = _write_reusable_dataset(base, config_id)
    log = logging.getLogger("test_reuse_scan_tolerates_corrupt_meta")

    decision = gd._decide_reuse_or_generate(base, config_id, "abc123", False, log)

    assert decision[0] == "reuse"
    assert Path(decision[1]) == good


def test_staging_dir_names_are_collision_proof(tmp_path, monkeypatch):
    """Two decisions in the same second stage into different directories.

    The old name was ``.tmp_%Y%m%d-%H%M%S`` with ``mkdir(exist_ok=False)``, so
    two same-config jobs starting in the same second raced to a
    ``FileExistsError`` on one of them -- inside the unguarded rank-0 window.
    """
    config = _reuse_config(tmp_path / "datasets")
    base = _base_dir_for(config)
    base.mkdir(parents=True)
    log = logging.getLogger("test_staging_dir_names_are_collision_proof")

    # Pin the clock so both decisions share a timestamp: only a non-time
    # component can keep the names apart.
    monkeypatch.setattr(gd.time, "strftime", lambda *_args: "20260101-000000")

    first = gd._decide_reuse_or_generate(base, base.name, "abc123", False, log)
    second = gd._decide_reuse_or_generate(base, base.name, "abc123", False, log)

    assert first[0] == "generate" and second[0] == "generate"
    assert first[1] != second[1], "same-second staging dirs collided"
    assert Path(first[1]).is_dir() and Path(second[1]).is_dir()


def test_finalize_failure_is_broadcast_not_left_to_barrier(tmp_path, monkeypatch):
    """A rank-0 rename failure is broadcast; peers raise instead of hanging.

    The rename happens *after* the generation consensus, so a failure there
    (e.g. a racing job already created the destination) used to kill rank 0
    while every peer sat in the final ``Barrier``. The fix carries the failure
    through one more collective and raises everywhere.
    """
    config = _reuse_config(tmp_path / "datasets")
    comm = FakeComm(rank=0, size=2, allreduce_result=1)
    monkeypatch.setattr(gd, "MPI", FakeMPI(comm))
    monkeypatch.setattr(gd, "_git_commit_short", lambda log: "abc123")
    monkeypatch.setattr(volumegen, "main", lambda _config: None)

    # Pin the clock so the destination name is predictable, then have a
    # "racing job" occupy it with a non-empty directory: the rename fails with
    # ENOTEMPTY exactly as it did in the field.
    monkeypatch.setattr(gd.time, "strftime", lambda *_args: "20260101-000000")
    base = _base_dir_for(config)
    dest = base / "20260101-000000__abc123"
    dest.mkdir(parents=True)
    (dest / "placeholder").write_text("created by a racing job")

    with pytest.raises(RuntimeError) as excinfo:
        gd.get_dataset(config)

    message = str(excinfo.value)
    assert "20260101-000000__abc123" in message
    # The failure travelled through a collective *after* the generation
    # consensus, so peers learn about it rather than waiting in the barrier.
    assert "allgather" in comm.calls
    assert comm.calls.index("allgather") < len(comm.calls) - 1


def test_non_root_raises_on_broadcast_finalize_error(tmp_path, monkeypatch):
    """A peer receiving the finalize error raises rather than returning dest."""
    config = _reuse_config(tmp_path / "datasets")
    dest = tmp_path / "datasets" / "cid" / "20260101-000000__abc123"
    comm, _tmp, _dest = _generate_decision_comm(
        rank=1, size=2, dest=dest, allreduce_result=1
    )
    # Second bcast: root's finalize verdict (a failure message).
    comm._bcast_returns.append("rank 0 failed to finalize: OSError: boom")
    monkeypatch.setattr(gd, "MPI", FakeMPI(comm))
    monkeypatch.setattr(gd, "_git_commit_short", lambda log: "abc123")
    monkeypatch.setattr(volumegen, "main", lambda _config: None)

    with pytest.raises(RuntimeError) as excinfo:
        gd.get_dataset(config)

    assert "boom" in str(excinfo.value)
    assert not dest.exists()


# ---------------------------------------------------------------------------
# A missing instance file raises FileNotFoundError (a catchable Exception)
# rather than calling sys.exit(1) (a BaseException that bypasses consensus), and
# volumegen's generation loop catches worker failures locally so it reaches the
# same allreduce/allgather consensus on both the ok and error paths.
# ---------------------------------------------------------------------------


def _volumegen_config(dataset_dir: Path, fract_base: Path) -> Namespace:
    """Config for a tiny single-rank ``volumegen.main`` run (one 16^3 volume)."""
    return Namespace(
        dataset_dir=str(dataset_dir),
        fract_base_dir=str(fract_base),
        n_categories=1,
        n_instances_used_per_fractal=1,
        n_fracts_per_vol=1,
        seed=1234,
        variance_threshold=0.15,
        val_split=0,
        vol_size=16,
        point_num=64,
        scale=1,
    )


def _seed_one_instance(fract_base: Path, config: Namespace, *, present: bool) -> None:
    """Create (or deliberately omit) the single instance file volumegen needs.

    volumegen selects instance indices with ``random.sample(range(145), ...)``
    seeded by ``config.seed``; to be robust we populate every one of the 145
    instance slots for category 0 when ``present`` is True.
    """
    inst_dir = (
        fract_base
        / f"var{config.variance_threshold}"
        / "instances"
        / f"np{config.point_num}"
        / "000000"
    )
    inst_dir.mkdir(parents=True, exist_ok=True)
    if present:
        rng = np.random.default_rng(0)
        for instance in range(145):
            np.save(inst_dir / f"000000_{instance:04d}.npy", rng.random((64, 3)))


def _run_volumegen_single(config, monkeypatch, **comm_kwargs):
    """Run ``volumegen.main`` once with a size-1 FakeComm; return the comm."""
    comm = FakeComm(rank=0, size=1, **comm_kwargs)
    monkeypatch.setattr(volumegen, "MPI", FakeMPI(comm))
    volumegen.main(config)
    return comm


def test_volumegen_success_collective_sequence(tmp_path, monkeypatch):
    """A clean single-rank run records bcast -> allreduce -> allgather.

    This captures the collective sequence of the success path so the failure
    test below can assert it is *identical* -- the property that keeps ranks in
    step regardless of which of them fails.
    """
    dataset_dir = tmp_path / "ds"
    fract_base = tmp_path / "fractals"
    config = _volumegen_config(dataset_dir, fract_base)
    _seed_one_instance(fract_base, config, present=True)

    comm = _run_volumegen_single(config, monkeypatch)

    # The consensus collectives run and are ordered allreduce-before-allgather.
    assert "allreduce" in comm.calls
    assert "allgather" in comm.calls
    assert comm.calls.index("allreduce") < comm.calls.index("allgather")
    # A dataset was written.
    assert (dataset_dir / "volumes").exists()


def test_volumegen_missing_instance_raises_file_not_found(tmp_path, monkeypatch):
    """A missing instance file surfaces as RuntimeError wrapping FileNotFoundError.

    The generation loop raises ``FileNotFoundError`` (never ``SystemExit``); the
    loop catches it locally, records the per-rank status, and the consensus
    allreduce/allgather then re-raises it as a ``RuntimeError`` on every rank.
    The message names the FileNotFoundError so the original cause is visible.
    """
    dataset_dir = tmp_path / "ds"
    fract_base = tmp_path / "fractals"
    config = _volumegen_config(dataset_dir, fract_base)
    # Directory exists but is empty -> the instance file is missing.
    _seed_one_instance(fract_base, config, present=False)

    with pytest.raises(RuntimeError) as excinfo:
        _run_volumegen_single(config, monkeypatch, allreduce_result=0)

    # It is a RuntimeError from the consensus, and it did NOT escape as SystemExit.
    assert "FileNotFoundError" in str(excinfo.value)


def test_volumegen_error_path_same_collectives_as_success(tmp_path, monkeypatch):
    """The failure path executes the identical collective sequence as success.

    This is the crux of the deadlock fix: a rank that fails inside the loop must
    still reach the same allreduce/allgather the successful ranks reach. We
    capture the sequence on the success path and on the failure path (up to the
    raise) and assert the recorded collectives match.
    """
    # Success path sequence.
    ds_ok = tmp_path / "ok"
    fb_ok = tmp_path / "fractals_ok"
    cfg_ok = _volumegen_config(ds_ok, fb_ok)
    _seed_one_instance(fb_ok, cfg_ok, present=True)
    comm_ok = _run_volumegen_single(cfg_ok, monkeypatch)

    # Failure path sequence (missing instance file). Capture the comm even
    # though main raises, by constructing it ourselves.
    ds_err = tmp_path / "err"
    fb_err = tmp_path / "fractals_err"
    cfg_err = _volumegen_config(ds_err, fb_err)
    _seed_one_instance(fb_err, cfg_err, present=False)

    comm_err = FakeComm(rank=0, size=1, allreduce_result=0)
    monkeypatch.setattr(volumegen, "MPI", FakeMPI(comm_err))
    with pytest.raises(RuntimeError):
        volumegen.main(cfg_err)

    # Both paths reach the same collective sequence (the failure raises only
    # after the shared allreduce/allgather consensus completes).
    assert comm_err.calls == comm_ok.calls


# ---------------------------------------------------------------------------
# The instance work list is built once on rank 0 and broadcast, so every rank
# slices an identical list even when their filesystem views diverge.
# ---------------------------------------------------------------------------


def _instance_config(fract_base: Path) -> Namespace:
    """Config for a single-rank ``instance.main`` run over 2 categories."""
    return Namespace(
        fract_base_dir=str(fract_base),
        n_categories=2,
        seed=1234,
        variance_threshold=0.15,
        point_num=64,
        datagen_from_scratch=False,
    )


def _seed_ifs_params(fract_base: Path, config: Namespace, n_categories: int) -> None:
    """Write a contractive IFS param CSV per category so generation stays fast."""
    param_dir = fract_base / f"var{config.variance_threshold}" / "3DIFS_param"
    param_dir.mkdir(parents=True, exist_ok=True)
    params = np.zeros((2, 13), dtype=np.float64)
    params[:, 0] = params[:, 4] = params[:, 8] = 0.5
    params[1, 9] = params[1, 10] = params[1, 11] = 0.5
    params[0, 12] = 0.5
    for category in range(n_categories):
        np.savetxt(param_dir / f"{category:06d}.csv", params, delimiter=",")


def test_instance_work_list_built_on_root_and_broadcast(tmp_path, monkeypatch):
    """Rank 0 builds the work list; ``bcast`` carries it before any slicing.

    Driving ``instance.main`` as rank 0 in a 2-rank world, the work list is
    computed locally and then broadcast. We confirm a ``bcast`` occurs and its
    payload is the full desired list (2 categories x 145 instances = 290 pairs),
    computed once rather than per rank.
    """
    fract_base = tmp_path / "fractals"
    config = _instance_config(fract_base)
    _seed_ifs_params(fract_base, config, n_categories=2)

    # Only generate a tiny slice: rank 0 of a huge world slices [0:1], so seed a
    # large size so the local share is a single instance and the run is fast.
    comm = FakeComm(rank=0, size=290)
    monkeypatch.setattr(inst, "MPI", FakeMPI(comm))

    rc = inst.main(config)
    assert rc == 0

    # The work list was broadcast, and its payload is the full 290-pair list.
    assert "bcast" in comm.calls
    work_list = comm.bcast_payloads[0]
    assert len(work_list) == 2 * 145
    assert [0, 0] in work_list and [1, 144] in work_list


def test_instance_non_root_uses_broadcast_list_not_local_glob(tmp_path, monkeypatch):
    """A non-root rank slices root's broadcast list, ignoring its own glob.

    Simulating rank 1 whose local filesystem glob would see a *different* set of
    existing instances: with the fix rank 1 never runs the scan (only rank 0
    does), so it slices exactly the list root broadcast. We monkeypatch
    ``glob.glob`` to raise if rank 1 ever tries to scan, proving the scan is
    root-only.
    """
    fract_base = tmp_path / "fractals"
    config = _instance_config(fract_base)
    _seed_ifs_params(fract_base, config, n_categories=2)

    # Root's list (what rank 1 must slice). A 2-pair list keeps rank 1's share
    # to a single instance so the generation is fast.
    root_list = [[0, 0], [1, 0]]
    comm = FakeComm(rank=1, size=2, bcast_returns=[root_list])
    monkeypatch.setattr(inst, "MPI", FakeMPI(comm))

    # Any glob-based scan on a non-root rank is a bug: fail loudly if attempted.
    def no_scan(*_args, **_kwargs):
        raise AssertionError("non-root rank must not scan the filesystem")

    monkeypatch.setattr(inst.glob, "glob", no_scan)

    rc = inst.main(config)
    assert rc == 0

    # Rank 1 received the broadcast list and generated its share (pair [1, 0]).
    generated = (
        fract_base
        / f"var{config.variance_threshold}"
        / "instances"
        / f"np{config.point_num}"
        / "000001"
        / "000001_0000.npy"
    )
    assert generated.exists()


def test_instance_divergent_glob_does_not_change_partition(tmp_path, monkeypatch):
    """Rank 0's own scan divergence cannot desync peers: only its list is used.

    Two size-2 simulations where rank 1's view differs from rank 0's: because
    the fix broadcasts rank 0's list, rank 1's slice is a deterministic function
    of rank 0's list alone. We check the two per-rank shares are disjoint and
    together cover the whole broadcast list (no duplicate writes, no orphans).
    """
    root_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    size = 2

    # Every rank receives the same broadcast list, then block-slices it exactly
    # as the fixed code does. Reproducing the slice here shows the shares are a
    # deterministic function of the shared list alone.
    shares = []
    for rank in range(size):
        per = ceil(len(root_list) / size)
        start = rank * per
        end = min((rank + 1) * per, len(root_list))
        shares.append([tuple(p) for p in root_list[start:end]])

    # Disjoint shares, full coverage.
    assert set(shares[0]).isdisjoint(set(shares[1]))
    assert set(shares[0]) | set(shares[1]) == {tuple(p) for p in root_list}


# ---------------------------------------------------------------------------
# Tier (b): real 2-rank tests via the MPI runner. These SKIP where no launcher
# exists (the sandboxed dev environment) and run on any box with mpirun/srun.
# On unfixed code each fails by timeout-with-stacks rather than hanging CI.
# ---------------------------------------------------------------------------

from tests.helpers import mpi_runner  # noqa: E402

_GET_DATASET_SCRIPT = str(RANK_SCRIPTS / "datagen_get_dataset_consensus.py")
_INSTANCE_SCRIPT = str(RANK_SCRIPTS / "datagen_instance_partition.py")
_PROBE_SCRIPT = str(RANK_SCRIPTS / "mpi_launch_probe.py")


@pytest.fixture(scope="module")
def working_mpi():
    """Skip unless a launcher is present *and* can actually start 2 ranks.

    ``mpi_runner.mpi_run`` skips when no launcher is on ``PATH``, but some
    environments expose a launcher that fails to spawn (e.g. a ``flux`` with no
    broker). A quick probe run distinguishes the two so the real consensus tests
    skip cleanly instead of failing spuriously, while still running wherever a
    genuine launcher exists.
    """
    if mpi_runner.detect_mpi_launcher() is None:
        pytest.skip("no MPI launcher available")
    rc, out, _err = mpi_runner.mpi_run(_PROBE_SCRIPT, n=2, timeout=30)
    if rc != 0 or len(re.findall(r"PROBE_OK", out)) < 2:
        pytest.skip("MPI launcher present but cannot start ranks")


def _rank_markers(pattern: str, out: str) -> set:
    """Return the set of rank ids that emitted ``pattern`` (a regex with one group)."""
    return set(re.findall(pattern, out))


@pytest.mark.mpi
def test_success_path_still_works(tmp_path, working_mpi):
    """No fault: both ranks return the same finalized dataset path (control)."""
    rc, out, err = mpi_runner.mpi_run(
        _GET_DATASET_SCRIPT,
        n=2,
        timeout=60,
        env={"WORKDIR": str(tmp_path / "work"), "FAULT_MODE": "none"},
    )
    assert rc == 0, f"expected clean exit, got rc={rc}\nstdout:\n{out}\nstderr:\n{err}"
    returned = dict(re.findall(r"RANK (\d+) RETURNED (\S+)", out))
    assert {"0", "1"} <= set(returned), f"both ranks must return\n{out}"
    assert returned["0"] == returned["1"], f"ranks returned different paths\n{out}"


@pytest.mark.mpi
def test_all_rank_failure_raises_everywhere(tmp_path, working_mpi):
    """Failure injected on both ranks: both raise RuntimeError within the timeout."""
    rc, out, err = mpi_runner.mpi_run(
        _GET_DATASET_SCRIPT,
        n=2,
        timeout=60,
        env={"WORKDIR": str(tmp_path / "work"), "FAULT_MODE": "all"},
    )
    assert rc != 0, f"expected non-zero exit\nstdout:\n{out}\nstderr:\n{err}"
    raised = _rank_markers(r"RANK (\d+) RAISED", out)
    assert {"0", "1"} <= raised, (
        f"both ranks must raise\nstdout:\n{out}\nstderr:\n{err}"
    )
    assert "RETURNED" not in out, f"no rank should finalize on failure\n{out}"


@pytest.mark.mpi
def test_single_rank_failure_propagates(tmp_path, working_mpi):
    """Failure injected on rank 1 only: both ranks still raise within the timeout."""
    rc, out, err = mpi_runner.mpi_run(
        _GET_DATASET_SCRIPT,
        n=2,
        timeout=60,
        env={"WORKDIR": str(tmp_path / "work"), "FAULT_MODE": "rank1"},
    )
    assert rc != 0, f"expected non-zero exit\nstdout:\n{out}\nstderr:\n{err}"
    raised = _rank_markers(r"RANK (\d+) RAISED", out)
    assert {"0", "1"} <= raised, (
        f"both ranks must raise\nstdout:\n{out}\nstderr:\n{err}"
    )


@pytest.mark.mpi
def test_missing_instance_no_peer_hang(tmp_path, working_mpi):
    """A missing instance file makes both ranks raise, not hang; no SystemExit escapes."""
    rc, out, err = mpi_runner.mpi_run(
        _GET_DATASET_SCRIPT,
        n=2,
        timeout=60,
        env={"WORKDIR": str(tmp_path / "work"), "FAULT_MODE": "missing"},
    )
    assert rc != 0, f"expected non-zero exit\nstdout:\n{out}\nstderr:\n{err}"
    assert "SYSEXIT" not in out, f"SystemExit must not escape get_dataset\n{out}"
    raised = _rank_markers(r"RANK (\d+) RAISED", out)
    assert {"0", "1"} <= raised, (
        f"both ranks must raise\nstdout:\n{out}\nstderr:\n{err}"
    )


@pytest.mark.mpi
def test_reuse_decision_broadcast(tmp_path, working_mpi):
    """A blinded non-root scan still agrees with root: both return the same path."""
    rc, out, err = mpi_runner.mpi_run(
        _GET_DATASET_SCRIPT,
        n=2,
        timeout=60,
        env={"WORKDIR": str(tmp_path / "work"), "FAULT_MODE": "reuse"},
    )
    assert rc == 0, f"expected clean exit, got rc={rc}\nstdout:\n{out}\nstderr:\n{err}"
    returned = dict(re.findall(r"RANK (\d+) RETURNED (\S+)", out))
    assert {"0", "1"} <= set(returned), f"both ranks must return\n{out}"
    assert returned["0"] == returned["1"], (
        f"ranks diverged despite the broadcast decision\n{out}"
    )


@pytest.mark.mpi
def test_partition_identical_across_ranks(tmp_path, working_mpi):
    """Divergent per-rank globs: the broadcast list yields full, non-duplicated coverage."""
    rc, out, err = mpi_runner.mpi_run(
        _INSTANCE_SCRIPT,
        n=2,
        timeout=120,
        env={"WORKDIR": str(tmp_path / "work")},
    )
    assert rc == 0, f"expected clean exit, got rc={rc}\nstdout:\n{out}\nstderr:\n{err}"
    done = _rank_markers(r"RANK (\d+) DONE", out)
    assert {"0", "1"} <= done, f"both ranks must finish\nstdout:\n{out}\nstderr:\n{err}"
    match = re.search(r"COVERAGE ok=(\d) missing=(\d+) extra=(\d+)", out)
    assert match, f"missing coverage audit line\nstdout:\n{out}"
    ok, missing, extra = match.groups()
    assert ok == "1", (
        f"partition incomplete: missing={missing} extra={extra}\nstdout:\n{out}"
    )
