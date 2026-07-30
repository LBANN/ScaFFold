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

"""Resume and run-directory lifecycle tests.

Covers the CLI run-directory resolution matrix (fresh vs. resume vs. the
incoherent ``--restart`` without ``--run-dir`` case), fresh-second collision
avoidance, and the trainer's stats-file and step-counter handling across a
resume: no duplicate CSV header when nothing was resumed, a header rewritten
when the stats file is missing, appends without a second header on a real
resume, and the optimizer-step counter continuing from the saved value.

The CLI resolution tests call ``resolve_run_dir`` directly (a pure function of
its argument dict and the merged config), so they need neither MPI nor a real
benchmark launch. The trainer tests use the CPU, single-process fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import ScaFFold.cli as cli

# ---------------------------------------------------------------------------
# resolve_run_dir matrix (F02)
# ---------------------------------------------------------------------------


def test_resolve_run_dir_matrix(tmp_path):
    """Every supported flag combination resolves coherently.

    * ``--run-dir`` alone and ``--restart --run-dir`` both resume in the given
      directory with ``restarting=True``.
    * neither flag creates a fresh directory under ``base_run_dir`` with
      ``restarting=False``.
    * ``--restart`` without ``--run-dir`` is rejected with a clear ValueError.
    In every non-error path ``benchmark_run_dir`` is written into the config.
    """
    base = tmp_path / "runs"
    base.mkdir()
    explicit = tmp_path / "prior_run"
    explicit.mkdir()

    def cfg():
        return {"base_run_dir": str(base), "job_name": "benchmark"}

    # --run-dir alone -> resume in that dir.
    c = cfg()
    run_dir, restarting = cli.resolve_run_dir(
        {"restart": False, "run_dir": str(explicit)}, c
    )
    assert run_dir == explicit
    assert restarting is True
    assert c["benchmark_run_dir"] == str(explicit)
    assert c["train_from_scratch"] is False
    assert c["restart"] is True

    # --restart + --run-dir -> same dir, same semantics.
    c = cfg()
    run_dir, restarting = cli.resolve_run_dir(
        {"restart": True, "run_dir": str(explicit)}, c
    )
    assert run_dir == explicit
    assert restarting is True
    assert c["benchmark_run_dir"] == str(explicit)

    # neither -> a fresh dir under base_run_dir.
    c = cfg()
    run_dir, restarting = cli.resolve_run_dir({"restart": False, "run_dir": None}, c)
    assert restarting is False
    assert run_dir.parent == base
    assert run_dir.exists()
    assert c["benchmark_run_dir"] == str(run_dir)

    # --restart alone -> clear error, no dir created.
    c = cfg()
    with pytest.raises(ValueError, match="--run-dir"):
        cli.resolve_run_dir({"restart": True, "run_dir": None}, c)


def test_same_second_run_dirs_distinct(tmp_path, monkeypatch):
    """Two fresh resolutions in the same wall-clock second get distinct dirs.

    The timestamped name has 1-second resolution, so two launches in the same
    second would collide. With datetime frozen, the second resolution must fall
    back to a suffixed name rather than silently share the first dir.
    """
    base = tmp_path / "runs"
    base.mkdir()

    class _FrozenDateTime:
        @staticmethod
        def now():
            import datetime as _dt

            return _dt.datetime(2026, 7, 22, 1, 2, 3)

    monkeypatch.setattr(cli, "datetime", _FrozenDateTime)

    def cfg():
        return {"base_run_dir": str(base), "job_name": "benchmark"}

    d1, r1 = cli.resolve_run_dir({"restart": False, "run_dir": None}, cfg())
    d2, r2 = cli.resolve_run_dir({"restart": False, "run_dir": None}, cfg())

    assert r1 is False and r2 is False
    assert d1 != d2
    assert d1.exists() and d2.exists()
    # Both empty at creation (neither clobbered the other).
    assert not any(d1.iterdir())
    assert not any(d2.iterdir())


# ---------------------------------------------------------------------------
# stats-file header handling on resume (F01) + step counters (F14)
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from ScaFFold.utils.checkpointing import CheckpointManager  # noqa: E402
from ScaFFold.utils.trainer import PyTorchTrainer  # noqa: E402

_HEADER = (
    "epoch,epoch_loss,overall_loss,val_loss_epoch,"
    "val_loss_avg,train_dice,val_dice,epoch_duration,"
    "optimizer_steps,total_optimizer_steps"
)


def _stub_trainer(run_dir, *, train_from_scratch, log):
    """A PyTorchTrainer carrying only what cleanup_or_resume/train touch.

    Built via ``object.__new__`` (as the checkpointing tests do) so no dataset,
    model, or process group is needed. A real CheckpointManager over a tiny
    linear model backs the resume path so load/save round-trip faithfully.
    """
    t = object.__new__(PyTorchTrainer)
    t.config = SimpleNamespace(
        train_from_scratch=train_from_scratch,
        run_dir=str(run_dir),
        checkpoint_interval=-1,
    )
    t.world_rank = 0
    t.global_step = 0
    t.total_optimizer_steps = 0
    t.log = log
    t.train_set = SimpleNamespace(mask_values=None)
    t.outfile_path = str(run_dir) + "/train_stats.csv"
    t.checkpoint_manager = CheckpointManager(
        model=torch.nn.Linear(2, 2),
        base_dir=str(Path(run_dir) / "checkpoints"),
        log=log,
        world_rank=0,
        dist_enabled=False,
        async_save=False,
    )
    return t


def _write_rows(path, epochs, dur=10.0):
    """Append epoch rows in the trainer's rank-0 CSV format."""
    with open(path, "a", newline="") as f:
        for e in epochs:
            f.write(
                ",".join(
                    [str(e), "0.5", "0.5", "0.4", "0.4", "0.8", "0.8", str(dur)]
                    + ["4", str(4 * int(e))]
                )
                + "\n"
            )


def _header_count(path):
    return sum(1 for ln in Path(path).read_text().splitlines() if ln == _HEADER)


def test_no_double_header_without_checkpoint(tmp_path, caplog):
    """Resuming a dir that has a CSV but no checkpoint keeps ONE header.

    A killed run with checkpointing disabled leaves epoch rows but no
    checkpoint; a relaunch with ``train_from_scratch`` off (but no explicit
    ``--restart``) finds nothing to resume and starts fresh (start_epoch stays
    1). The stale CSV must be truncated to a single header so the CSV reader
    does not parse a second embedded header as NaNs.
    """
    log = __import__("logging").getLogger("resume.f01")
    run = tmp_path / "run"
    run.mkdir()
    csv = run / "train_stats.csv"

    # A prior run wrote a header and three epochs but never checkpointed.
    csv.write_text(_HEADER + "\n")
    _write_rows(csv, [1, 2, 3])

    t = _stub_trainer(run, train_from_scratch=False, log=log)
    t.cleanup_or_resume()

    assert t.start_epoch == 1  # nothing to resume
    assert _header_count(csv) == 1

    # A fresh run's epochs append cleanly and parse to finite values.
    _write_rows(csv, [1, 2, 3, 4])
    data = np.genfromtxt(str(csv), dtype=float, delimiter=",", names=True)
    durations = np.atleast_1d(data["epoch_duration"])
    assert np.all(np.isfinite(durations))
    assert durations.sum() == pytest.approx(40.0)


def test_resume_appends_without_header(tmp_path):
    """A real resume keeps one header and truncates future rows.

    With a checkpoint at epoch K, cleanup_or_resume restores start_epoch=K+1,
    keeps the single existing header, and truncates any rows >= K+1 so the
    resumed run overwrites them rather than duplicating.
    """
    log = __import__("logging").getLogger("resume.append")
    run = tmp_path / "run"
    run.mkdir()
    csv = run / "train_stats.csv"
    csv.write_text(_HEADER + "\n")
    _write_rows(csv, [1, 2, 3])

    # Save a real checkpoint at epoch 2 via the trainer's own manager.
    t = _stub_trainer(run, train_from_scratch=False, log=log)
    t.checkpoint_manager.save_checkpoint(
        epoch=2, val_loss_avg=0.5, extras={"train_mask_values": None}
    )

    t.cleanup_or_resume()
    assert t.start_epoch == 3  # resume from the epoch after the checkpoint
    assert _header_count(csv) == 1
    # Row for epoch 3 (>= start_epoch) was truncated; epochs 1,2 remain.
    epochs = [ln.split(",")[0] for ln in csv.read_text().splitlines()[1:]]
    assert epochs == ["1", "2"]


def test_header_rewritten_if_csv_missing(tmp_path):
    """Resuming with a checkpoint but a deleted CSV rewrites the header first.

    Otherwise the first appended epoch row is read as the column names and the
    score computation crashes.
    """
    log = __import__("logging").getLogger("resume.missing")
    run = tmp_path / "run"
    run.mkdir()
    csv = run / "train_stats.csv"

    t = _stub_trainer(run, train_from_scratch=False, log=log)
    t.checkpoint_manager.save_checkpoint(
        epoch=2, val_loss_avg=0.5, extras={"train_mask_values": None}
    )

    # The CSV never existed (or was removed) even though a checkpoint is present.
    assert not csv.exists()
    t.cleanup_or_resume()
    assert t.start_epoch == 3
    assert csv.exists()
    assert _header_count(csv) == 1

    _write_rows(csv, [3, 4])
    data = np.genfromtxt(str(csv), dtype=float, delimiter=",", names=True)
    assert np.all(np.isfinite(np.atleast_1d(data["epoch_duration"])))


def test_explicit_restart_requires_checkpoint(tmp_path):
    """An explicit --restart with no checkpoint on disk fails loudly.

    Without a checkpoint the run would silently retrain from scratch while the
    user believes they are resuming; the resume path must raise instead. A
    non-restart launch in the same state (previous test) starts fresh quietly.
    """
    log = __import__("logging").getLogger("resume.restart.missing")
    run = tmp_path / "run"
    run.mkdir()

    t = _stub_trainer(run, train_from_scratch=False, log=log)
    t.config.restart = True
    with pytest.raises(FileNotFoundError, match="no checkpoint was found"):
        t.cleanup_or_resume()


def test_explicit_restart_with_checkpoint_resumes(tmp_path):
    """An explicit --restart with a checkpoint present resumes normally."""
    log = __import__("logging").getLogger("resume.restart.present")
    run = tmp_path / "run"
    run.mkdir()

    t1 = _stub_trainer(run, train_from_scratch=False, log=log)
    t1.checkpoint_manager.save_checkpoint(
        epoch=2, val_loss_avg=0.5, extras={"train_mask_values": None}
    )

    t2 = _stub_trainer(run, train_from_scratch=False, log=log)
    t2.config.restart = True
    t2.cleanup_or_resume()
    assert t2.start_epoch == 3


def test_step_counters_roundtrip(tmp_path):
    """Step counters are saved in the checkpoint extras and restored on resume.

    A fresh trainer starts the counters at 0; after resuming a checkpoint that
    recorded non-zero step counts, it must continue from those values rather
    than restart the accounting.
    """
    log = __import__("logging").getLogger("resume.steps")
    run = tmp_path / "run"
    run.mkdir()

    # First run: advance the counters and checkpoint at epoch 2 with the exact
    # extras train() now passes.
    t1 = _stub_trainer(run, train_from_scratch=False, log=log)
    t1.global_step = 37
    t1.total_optimizer_steps = 37
    t1.checkpoint_manager.save_checkpoint(
        epoch=2,
        val_loss_avg=0.5,
        extras={
            "train_mask_values": None,
            "global_step": t1.global_step,
            "total_optimizer_steps": t1.total_optimizer_steps,
        },
    )

    # Fresh trainer/manager (counters re-initialised to 0) resuming the run.
    t2 = _stub_trainer(run, train_from_scratch=False, log=log)
    assert t2.global_step == 0
    t2.cleanup_or_resume()
    assert t2.start_epoch == 3
    assert t2.global_step == 37
    assert t2.total_optimizer_steps == 37


def test_total_steps_resume_predates_dedicated_key(tmp_path):
    """A checkpoint recording only global_step still resumes the step total.

    global_step and total_optimizer_steps advance in lockstep, so an older
    checkpoint without the dedicated total key falls back to global_step
    instead of resetting the total to 0.
    """
    log = __import__("logging").getLogger("resume.steps.legacy")
    run = tmp_path / "run"
    run.mkdir()

    t1 = _stub_trainer(run, train_from_scratch=False, log=log)
    t1.checkpoint_manager.save_checkpoint(
        epoch=2,
        val_loss_avg=0.5,
        extras={"train_mask_values": None, "global_step": 21},
    )

    t2 = _stub_trainer(run, train_from_scratch=False, log=log)
    t2.cleanup_or_resume()
    assert t2.global_step == 21
    assert t2.total_optimizer_steps == 21
