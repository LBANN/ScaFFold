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

"""Tests for category-search round sizing and its work-scan consensus."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from ScaFFold.datagen import category_search as cs
from ScaFFold.datagen import layout
from ScaFFold.datagen.category_search import compute_round_attempts


def test_full_batch_when_no_acceptance_data():
    # First round (no observed acceptance yet) runs the full per-rank batch so
    # an unknown regime is not starved.
    assert compute_round_attempts(1000, 512, 10000, 0.0) == 10000


def test_round_shrinks_when_few_remain():
    # With a healthy acceptance rate and only a handful of categories left, the
    # round must be far smaller than the fixed batch (the whole point: no
    # generating thousands of categories only to discard them).
    attempts = compute_round_attempts(1, 512, 10000, 0.16)
    assert attempts < 10000
    assert attempts >= 1


def test_round_never_exceeds_batch_cap():
    # Even a huge remaining count with a tiny acceptance rate is capped at the
    # configured per-rank batch size.
    assert compute_round_attempts(10**9, 1, 10000, 0.001) == 10000


def test_zero_remaining_runs_nothing():
    assert compute_round_attempts(0, 8, 10000, 0.1) == 0


def test_round_scales_with_remaining_and_rate():
    # remaining / (size * accept_rate), ceil-ed with a small margin, split over
    # ranks. 1000 remaining over 4 ranks at 5% acceptance needs ~5000 per rank.
    attempts = compute_round_attempts(1000, 4, 10000, 0.05)
    assert 5000 <= attempts <= 6000


def test_at_least_one_attempt_per_rank_when_work_remains():
    # A positive remaining count always yields at least one attempt per rank so
    # the loop makes progress and can keep learning the acceptance rate.
    assert compute_round_attempts(1, 1024, 10000, 0.99) >= 1


# ---------------------------------------------------------------------------
# R30: the initial work scan is made once on rank 0 and broadcast.
#
# ``categories_remaining`` gates a while loop that contains collectives, so it
# must be identical on every rank. Deriving it from a per-rank filesystem scan
# lets divergent views (a stale metadata cache, a racing job, a partially
# visible directory) put one rank inside the loop issuing ``bcast`` while
# another is past it issuing ``reduce`` -- mismatched collectives on
# COMM_WORLD, i.e. a hang. This mirrors the fix already applied in
# ``instance.py``: rank 0 scans, everyone else consumes the broadcast.
# ---------------------------------------------------------------------------


class CategorySearchComm:
    """Single-process stand-in for COMM_WORLD recording the collective order."""

    def __init__(self, rank=0, size=1, bcast_returns=None):
        self.rank = rank
        self.size = size
        self.calls = []
        self.bcast_payloads = []
        self._bcast_returns = list(bcast_returns or [])

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
        return self._bcast_returns.pop(0)

    def gather(self, obj, root=0):
        self.calls.append("gather")
        return [obj] if self.rank == root else None

    def reduce(self, value, op=None, root=0):
        self.calls.append("reduce")
        return value if self.rank == root else None


class FakeMPI:
    """Namespace mimicking ``mpi4py.MPI`` for one ``CategorySearchComm``."""

    def __init__(self, comm):
        import mpi4py.MPI as real_mpi

        self.COMM_WORLD = comm
        self.SUM = real_mpi.SUM


def _cs_config(fract_base: Path) -> Namespace:
    return Namespace(
        fract_base_dir=str(fract_base),
        n_categories=1,
        seed=42,
        variance_threshold=0.15,
        point_num=60,
        normalize=1,
        datagen_from_scratch=False,
        datagen_batch_size=4,
        verbose=0,
    )


def _seed_one_category(config: Namespace) -> None:
    """Write the single category CSV this config asks for."""
    param_dir = Path(layout.category_param_dir(config))
    param_dir.mkdir(parents=True, exist_ok=True)
    params = np.zeros((2, 13), dtype=np.float64)
    params[:, 0] = params[:, 4] = params[:, 8] = 0.5
    params[1, 9] = params[1, 10] = params[1, 11] = 0.5
    params[0, 12] = 0.5
    np.savetxt(param_dir / "000000.csv", params, delimiter=",")


def test_work_scan_is_root_only_and_broadcast(tmp_path, monkeypatch):
    """A non-root rank never scans; it consumes root's broadcast index list."""
    config = _cs_config(tmp_path / "fractals")
    comm = CategorySearchComm(rank=1, size=2, bcast_returns=[("ok", [0])])
    monkeypatch.setattr(cs, "MPI", FakeMPI(comm))

    def no_scan(*_args, **_kwargs):
        raise AssertionError("non-root rank must not scan the filesystem")

    monkeypatch.setattr(cs, "parse_category_indices", no_scan)

    cs.main(config)

    # Root said category 0 already exists, so this rank has nothing to do and
    # went straight to the post-loop reductions.
    assert comm.calls == ["Barrier", "bcast", "reduce", "reduce", "reduce", "reduce"]
    # It contributed nothing to the scan broadcast (it is not the scanner).
    assert comm.bcast_payloads[0] is None


def test_divergent_fs_views_take_the_same_collective_path(tmp_path, monkeypatch):
    """Ranks disagreeing about the directory still issue identical collectives.

    Rank 0 sees the finished category; rank 1's own view is empty. Before the
    fix rank 1 entered the work loop (``bcast``) while rank 0 was already past
    it (``reduce``). With the scan broadcast, rank 1's view is irrelevant.
    """
    # Rank 0: the category is on disk, so its scan finds it.
    root_config = _cs_config(tmp_path / "root_view")
    _seed_one_category(root_config)
    root_comm = CategorySearchComm(rank=0, size=2)
    monkeypatch.setattr(cs, "MPI", FakeMPI(root_comm))
    cs.main(root_config)

    # Rank 1: an empty directory (a divergent view), but root broadcast [0].
    peer_config = _cs_config(tmp_path / "peer_view")
    peer_comm = CategorySearchComm(rank=1, size=2, bcast_returns=[("ok", [0])])
    monkeypatch.setattr(cs, "MPI", FakeMPI(peer_comm))
    cs.main(peer_config)

    assert root_comm.bcast_payloads[0] == ("ok", [0])
    assert peer_comm.calls == root_comm.calls, (
        "ranks with divergent filesystem views issued different collectives: "
        f"rank 0 {root_comm.calls} vs rank 1 {peer_comm.calls}"
    )


# ---------------------------------------------------------------------------
# VB-4: temp files stranded by killed writes are swept, as in instance.py.
#
# Both atomic writers name their temp file after the writing pid, so a job
# killed mid-write leaves one behind per killed process, forever: nothing in
# this module ever looked at them again.
# ---------------------------------------------------------------------------


def test_stale_temp_files_are_swept(tmp_path, monkeypatch):
    """Category and attempt-counter temps from dead pids are removed."""
    config = _cs_config(tmp_path / "fractals")
    param_dir = Path(layout.category_param_dir(config))
    param_dir.mkdir(parents=True, exist_ok=True)
    _seed_one_category(config)  # n_categories=1, so the search has no work

    stale_csv = param_dir / ".000001.csv.tmp999999"
    stale_csv.write_text("0.5,0.5\n")
    stale_counter = param_dir / ".rng_attempt_rank3.tmp999999"
    stale_counter.write_text("17")

    comm = CategorySearchComm(rank=0, size=1)
    monkeypatch.setattr(cs, "MPI", FakeMPI(comm))

    cs.main(config)

    assert not stale_csv.exists(), "a stranded category temp file was kept"
    assert not stale_counter.exists(), "a stranded attempt-counter temp was kept"
    # The real artifact is untouched: only the temp names are swept.
    assert (param_dir / "000000.csv").exists()


# ---------------------------------------------------------------------------
# VB-6: a library in the old, seed-agnostic layout is explained, not ignored.
# ---------------------------------------------------------------------------


def test_old_layout_library_is_reported(tmp_path, monkeypatch, caplog):
    """A pre-relayout library produces one warning naming both directories."""
    config = _cs_config(tmp_path / "fractals")
    legacy = Path(layout.legacy_category_param_dir(config))
    legacy.mkdir(parents=True)
    (legacy / "000000.csv").write_text("")
    _seed_one_category(config)  # nothing to generate under the new layout

    comm = CategorySearchComm(rank=0, size=1)
    monkeypatch.setattr(cs, "MPI", FakeMPI(comm))

    with caplog.at_level("WARNING"):
        cs.main(config)

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert str(legacy) in messages
    assert layout.category_param_dir(config) in messages


def test_no_warning_without_an_old_layout(tmp_path, monkeypatch, caplog):
    """The warning does not fire for a fresh library (control)."""
    config = _cs_config(tmp_path / "fractals")
    _seed_one_category(config)

    comm = CategorySearchComm(rank=0, size=1)
    monkeypatch.setattr(cs, "MPI", FakeMPI(comm))

    with caplog.at_level("WARNING"):
        cs.main(config)

    assert "seed-agnostic" not in " ".join(
        record.getMessage() for record in caplog.records
    )


# ---------------------------------------------------------------------------
# VB-2: the whole rank-0 scan window is fenced, not just the index parse.
#
# Rank 0 also reads the parameters of every category already on disk, right
# after the scan broadcast. A CSV that will not parse therefore killed rank 0
# while its peers had already consumed the broadcast and moved on -- the same
# stranding the scan broadcast was introduced to prevent.
# ---------------------------------------------------------------------------


def test_unparseable_existing_category_is_broadcast_not_raised_on_root(
    tmp_path, monkeypatch
):
    """A ragged category CSV becomes a broadcast error, not a rank-0-only death."""
    config = _cs_config(tmp_path / "fractals")
    param_dir = Path(layout.category_param_dir(config))
    param_dir.mkdir(parents=True, exist_ok=True)
    # Six-digit name, so the scan accepts it; ragged rows, so loadtxt raises.
    (param_dir / "000000.csv").write_text("0.5,0.5,0.5\n0.5,0.5\n")

    comm = CategorySearchComm(rank=0, size=2)
    monkeypatch.setattr(cs, "MPI", FakeMPI(comm))

    with pytest.raises(RuntimeError) as excinfo:
        cs.main(config)

    # Rank 0 stopped at the scan broadcast, and what it broadcast is the error
    # sentinel its peers need in order to abort with it.
    assert comm.calls == ["Barrier", "bcast"]
    assert comm.bcast_payloads[-1][0] == "error"
    assert "000000.csv" in str(excinfo.value) or "3DIFS_param" in str(excinfo.value)


def test_peer_raises_on_broadcast_scan_error(tmp_path, monkeypatch):
    """A peer receiving the scan sentinel raises instead of entering the loop."""
    config = _cs_config(tmp_path / "fractals")
    sentinel = ("error", "rank 0 failed to scan existing categories: ValueError: boom")
    comm = CategorySearchComm(rank=1, size=2, bcast_returns=[sentinel])
    monkeypatch.setattr(cs, "MPI", FakeMPI(comm))

    with pytest.raises(RuntimeError, match="boom"):
        cs.main(config)

    # It never reached the work loop's collectives.
    assert comm.calls == ["Barrier", "bcast"]


# ---------------------------------------------------------------------------
# R31: category CSVs appear complete or not at all.
#
# A category file truncated by a killed job is still counted as "done" by the
# resume scan, so nothing ever regenerates it: instance generation then dies
# parsing it, and the category search's own resume dies re-loading it. The
# pipeline cannot self-heal -- the file has to be deleted by hand.
# ---------------------------------------------------------------------------


def _params() -> np.ndarray:
    params = np.zeros((2, 13), dtype=np.float64)
    params[:, 0] = params[:, 4] = params[:, 8] = 0.5
    params[0, 12] = 0.5
    return params


def test_category_csv_write_is_atomic(tmp_path, monkeypatch):
    """A killed mid-write leaves no category file under a name resume accepts."""
    param_dir = tmp_path / "3DIFS_param"
    param_dir.mkdir()

    # One complete category, saved normally.
    indices, saved = [], []
    first = _params()
    assert cs.save_valid_category(str(param_dir), first, indices, saved) == 0
    assert np.loadtxt(param_dir / "000000.csv", delimiter=",").shape == (2, 13)

    # The next save is interrupted after some bytes have been written.
    observed = {}

    def partial_then_raise(fname, arr, *args, **kwargs):
        observed["listing"] = sorted(p.name for p in param_dir.iterdir())
        handle = fname if hasattr(fname, "write") else open(fname, "w")
        handle.write("0.5,0.5,0.5\n")
        handle.flush()
        if handle is not fname:
            handle.close()
        raise OSError("simulated SIGKILL mid-write")

    monkeypatch.setattr(cs.np, "savetxt", partial_then_raise)

    second = _params()
    second[0, 0] = 0.25
    with pytest.raises(OSError):
        cs.save_valid_category(str(param_dir), second, indices, saved)

    # No truncated category is visible: the resume scan still sees exactly the
    # one complete category, and every file it names parses.
    assert cs.parse_category_indices(str(param_dir)) == [0]
    assert not (param_dir / "000001.csv").exists()
    for idx in cs.parse_category_indices(str(param_dir)):
        assert np.loadtxt(param_dir / f"{idx:06d}.csv", delimiter=",").shape == (2, 13)

    # Mid-write, the partial data lived under a name neither the resume scan
    # nor the instance loader (which takes every ``*.csv``) would pick up.
    partial_names = [n for n in observed["listing"] if n != "000000.csv"]
    assert all(not name.endswith(".csv") for name in partial_names), partial_names

    # And nothing was left behind afterwards.
    assert sorted(p.name for p in param_dir.iterdir()) == ["000000.csv"]
