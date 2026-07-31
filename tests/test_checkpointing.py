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

"""Checkpointing correctness tests.

Covers the CheckpointManager save/load path (atomic writes, corruption
fallback, exception propagation, race-free best selection, CPU snapshot
isolation) and the trainer control flow around it (a final checkpoint on
exit, and step counters that ignore GradScaler-skipped steps).

All CheckpointManager tests use the single-process CPU path
(``dist_enabled=False``, ``world_rank=0``) with a tiny ``nn.Linear`` model so
they need neither a GPU nor a process group.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import ScaFFold.utils.trainer as trainer_mod
from ScaFFold.utils.checkpointing import CheckpointManager
from ScaFFold.utils.trainer import PyTorchTrainer
from tests.helpers import mpi_runner

# Two-rank rank script that resumes from a checkpoint and reports, per rank, how
# many times it read the checkpoint file from disk.
RESUME_RANK_SCRIPT = (
    Path(__file__).resolve().parent
    / "helpers"
    / "rank_scripts"
    / "checkpoint_resume_2rank.py"
)

_requires_gloo = pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    reason="requires torch.distributed with the gloo backend",
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_manager(base_dir, *, async_save=False):
    """A CPU, single-process CheckpointManager over a tiny linear model."""
    model = torch.nn.Linear(8, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    mgr = CheckpointManager(
        model=model,
        optimizer=optimizer,
        base_dir=str(base_dir),
        world_rank=0,
        dist_enabled=False,
        async_save=async_save,
    )
    return mgr, model


class _FakeCudaEvent:
    """CPU stand-in for torch.cuda.Event so the trainer's timing path runs."""

    def __init__(self, enable_timing=False):
        self._t = None

    def record(self):
        self._t = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return (other._t - self._t) * 1000.0


# ---------------------------------------------------------------------------
# F10 -- atomic writes + fallback + no swallowing
# ---------------------------------------------------------------------------


def test_save_atomic_on_crash(tmp_path, monkeypatch):
    """A crash mid-write leaves the previous checkpoint intact and readable.

    The writer serializes to a temp file and only renames it onto the final
    name on success, so a torch.save that writes partial bytes then raises
    must NOT clobber the good ``checkpoint_last.pth`` and must NOT leave a
    truncated file at the final path.
    """
    mgr, _ = _make_manager(tmp_path)

    # A valid previous checkpoint at epoch 1.
    mgr.save_checkpoint(epoch=1, val_loss_avg=0.5, extras={"train_mask_values": [0, 1]})
    assert mgr.last_ckpt_path.exists()

    # Simulate a walltime SIGKILL mid-write: emit a few bytes, then fail.
    real_save = torch.save

    def partial_then_raise(obj, f, *args, **kwargs):
        # The manager writes to an open temp file handle; emit a few bytes to
        # simulate a partial write, then fail.
        f.write(b"\x80\x04partial-checkpoint-bytes")
        raise RuntimeError("simulated disk failure mid-write")

    monkeypatch.setattr(torch, "save", partial_then_raise)

    # Stop-swallowing: the failing save must surface to the caller.
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        mgr.save_checkpoint(epoch=2, val_loss_avg=0.9)

    monkeypatch.setattr(torch, "save", real_save)

    # The previous good checkpoint is untouched and still loads.
    reloaded = torch.load(mgr.last_ckpt_path, map_location="cpu", weights_only=False)
    assert reloaded["epoch"] == 1

    # No half-written temp file was left behind in the directory.
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == []


def test_load_falls_back_to_best(tmp_path, caplog):
    """A corrupt ``last`` falls back to ``best`` and quarantines the bad file."""
    mgr, _ = _make_manager(tmp_path)

    # epoch 1 becomes 'best'; epoch 2 (worse) is only 'last'.
    mgr.save_checkpoint(epoch=1, val_loss_avg=0.1, extras={"train_mask_values": [7, 8]})
    mgr.save_checkpoint(epoch=2, val_loss_avg=0.9)
    assert mgr.best_ckpt_path.exists() and mgr.last_ckpt_path.exists()

    # Corrupt 'last' (mid-write kill analogue).
    size = mgr.last_ckpt_path.stat().st_size
    with open(mgr.last_ckpt_path, "r+b") as handle:
        handle.truncate(size // 2)

    # A restarted job: fresh manager loading from the same directory.
    mgr2, _ = _make_manager(tmp_path)
    loaded_paths = []
    real_load = torch.load

    def spy_load(path, *args, **kwargs):
        loaded_paths.append(str(path))
        return real_load(path, *args, **kwargs)

    torch.load = spy_load
    try:
        start_epoch = mgr2.load_from_checkpoint()
    finally:
        torch.load = real_load

    # Fell back to the valid best (epoch 1) -> resume at epoch 2.
    assert start_epoch == 2
    assert mgr2.restored_extras.get("train_mask_values") == [7, 8]

    # 'best' was consulted, and the corrupt 'last' was renamed aside.
    assert any(name.endswith("checkpoint_best.pth") for name in loaded_paths)
    assert not mgr2.last_ckpt_path.exists()
    assert (tmp_path / "checkpoint_last.pth.corrupt").exists()


def test_save_errors_not_swallowed(tmp_path, monkeypatch):
    """A writer exception propagates instead of being silently swallowed."""
    mgr, _ = _make_manager(tmp_path)

    def always_raise(obj, f, *args, **kwargs):
        raise RuntimeError("writer boom")

    monkeypatch.setattr(torch, "save", always_raise)

    with pytest.raises(RuntimeError, match="writer boom"):
        mgr.save_checkpoint(epoch=1, val_loss_avg=0.5)


# ---------------------------------------------------------------------------
# F41 -- race-free best decision (cached best loss, no per-save probe)
# ---------------------------------------------------------------------------


def test_async_best_decision_not_racy(tmp_path, monkeypatch):
    """is_best is decided from the cached best loss, never by re-reading best.

    With a slow background writer, re-reading ``checkpoint_best.pth`` to decide
    is_best would race the writer. The decision must come from an in-memory
    cache; we prove the disk probe is gone by asserting torch.load is never
    called on the best checkpoint during the saves.
    """
    mgr, _ = _make_manager(tmp_path, async_save=True)

    # Commit a first 'best' (loss 0.5) so the best checkpoint exists on disk;
    # this is the state the racy per-save probe would re-read.
    mgr.save_checkpoint(epoch=1, val_loss_avg=0.5)
    mgr.wait_for_save()
    assert mgr.best_ckpt_path.exists()

    # Slow the background writer so any probe would read a half-written file.
    real_write = CheckpointManager._write_to_disk

    def slow_write(state_dict, last_path, best_path, is_best, log=None):
        time.sleep(0.3)
        return real_write(state_dict, last_path, best_path, is_best, log)

    monkeypatch.setattr(CheckpointManager, "_write_to_disk", staticmethod(slow_write))

    best_loads = []
    real_load = torch.load

    def spy_load(path, *args, **kwargs):
        if str(path).endswith("checkpoint_best.pth"):
            best_loads.append(str(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)

    # Two more saves with improving loss; each is the new best.
    is_best_2 = mgr.save_checkpoint(epoch=2, val_loss_avg=0.3)
    time.sleep(0.05)  # background writer for epoch 2 is still in flight
    is_best_3 = mgr.save_checkpoint(epoch=3, val_loss_avg=0.2)
    mgr.wait_for_save()

    assert is_best_2 is True
    assert is_best_3 is True

    # The decision never re-read the best checkpoint from disk (probe is gone).
    assert best_loads == []

    monkeypatch.setattr(torch, "load", real_load)
    final_best = torch.load(mgr.best_ckpt_path, map_location="cpu", weights_only=False)
    assert final_best["epoch"] == 3
    assert final_best["val_loss_avg"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# R02 -- a from-scratch cleanup drops the deleted run's best, not just its files
# ---------------------------------------------------------------------------


def test_cleanup_from_scratch_resets_best(tmp_path):
    """``cleanup(train_from_scratch=True)`` resets the cached best-loss state.

    The manager seeds ``best_val_loss`` from ``checkpoint_best.pth`` at
    construction so a resumed run does not call its first epoch "best". When
    the same directory is then wiped for a fresh run, that cached score
    outlives the file it came from: every ``is_best`` decision of the new run
    is gated by a deleted run's score, so no ``checkpoint_best.pth`` is written
    until the retrain beats it -- leaving the run with no best-checkpoint
    fallback at all.
    """
    mgr, _ = _make_manager(tmp_path)
    mgr.save_checkpoint(epoch=1, val_loss_avg=0.01)
    assert mgr.best_ckpt_path.exists()

    # A driver reusing the run directory: the new manager seeds from disk.
    mgr2, _ = _make_manager(tmp_path)
    assert mgr2.best_val_loss == pytest.approx(0.01)
    mgr2.save_checkpoint(epoch=2, val_loss_avg=0.9)
    assert mgr2.last_saved_epoch == 2

    mgr2.cleanup(train_from_scratch=True)

    assert not mgr2.last_ckpt_path.exists()
    assert not mgr2.best_ckpt_path.exists()
    assert mgr2.best_val_loss == math.inf
    assert mgr2.last_saved_epoch is None

    # The fresh run's first epoch is its best, and a best checkpoint exists.
    assert mgr2.save_checkpoint(epoch=1, val_loss_avg=0.5) is True
    assert mgr2.best_ckpt_path.exists()


# ---------------------------------------------------------------------------
# F71 -- CPU tensors are cloned into the snapshot
# ---------------------------------------------------------------------------


def test_cpu_tensors_cloned(tmp_path):
    """_transfer_dict_to_cpu snapshots CPU tensors instead of aliasing them."""
    mgr, _ = _make_manager(tmp_path)

    state = {"step": torch.tensor(1.0), "nested": [torch.ones(3)]}
    snapshot = mgr._transfer_dict_to_cpu(state)

    # Mutate the originals in place, as a resumed training loop would.
    state["step"].add_(5)
    state["nested"][0].add_(9)

    assert snapshot["step"].item() == 1.0
    assert torch.equal(snapshot["nested"][0], torch.ones(3))


# ---------------------------------------------------------------------------
# F50 -- a final checkpoint is written when the run exits between intervals
# ---------------------------------------------------------------------------


def test_final_checkpoint_on_convergence(tiny_trainer, monkeypatch):
    """Converging at epoch 2 with interval 5 still checkpoints epoch 2 on exit."""
    trainer = tiny_trainer(
        config_overrides={
            "checkpoint_interval": 5,
            "epochs": -1,
            "target_dice": 0.9,
        }
    )

    # Drive epochs without the DistConv forward path or CUDA timing.
    monkeypatch.setattr(
        trainer,
        "_run_training_batch",
        lambda batch, **kw: (1, torch.tensor(0.3), torch.tensor(0.5)),
    )
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)

    # Reported val dice reaches the target at epoch 2.
    calls = {"n": 0}

    def fake_evaluate(*args, **kwargs):
        calls["n"] += 1
        dice_sum = 0.95 if calls["n"] >= 2 else 0.1
        return (dice_sum, 0.4, 0.4, 1, 1)

    monkeypatch.setattr(trainer_mod, "evaluate", fake_evaluate)

    trainer.cleanup_or_resume()
    trainer.train()

    assert calls["n"] == 2  # converged at epoch 2
    last_path = trainer.checkpoint_manager.last_ckpt_path
    assert last_path.exists()
    saved = torch.load(last_path, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 2


# ---------------------------------------------------------------------------
# F49 -- GradScaler-skipped steps do not advance the optimizer-step counter
# ---------------------------------------------------------------------------


def _run_and_flag(*, enabled, poison):
    """Run one scaler step (optionally poisoned with an inf grad) and return
    whether the trainer's skip-detection seam reports the step as applied."""
    scaler = torch.amp.GradScaler("cpu", enabled=enabled)
    stub = object.__new__(PyTorchTrainer)
    stub.grad_scaler = scaler
    stub.use_grad_scaler = enabled

    p = torch.nn.Parameter(torch.ones(4))
    opt = torch.optim.SGD([p], lr=0.1)

    scaler.scale((p * 2).sum()).backward()
    if poison:
        p.grad[0] = float("inf")
    scaler.unscale_(opt)

    scale_before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    return stub, stub._optimizer_step_applied(scale_before)


def test_skipped_step_not_counted():
    """An inf gradient (GradScaler backoff) is detected as a skipped step and
    does not increment the optimizer-step counter."""
    # Applied step: finite grads, scale not backed off.
    _, applied = _run_and_flag(enabled=True, poison=False)
    assert applied is True

    # Disabled scaler: scale is constant at 1.0, always an applied step.
    _, applied_disabled = _run_and_flag(enabled=False, poison=False)
    assert applied_disabled is True

    # Skipped step: inf grad backs the scale off -> not applied.
    stub, skipped = _run_and_flag(enabled=True, poison=True)
    assert skipped is False

    # The train loop gates the counter on this flag.
    stub._last_step_applied = skipped
    global_step = 5
    if getattr(stub, "_last_step_applied", True):
        global_step += 1
    assert global_step == 5  # unchanged: the skipped step was not counted


# ---------------------------------------------------------------------------
# F72 -- on resume only rank 0 reads the checkpoint file; peers get the
# deserialized state over the process group (no N-way filesystem read storm)
# ---------------------------------------------------------------------------


def test_resume_nonzero_rank_never_reads_disk(tmp_path, monkeypatch):
    """A simulated non-zero rank restores state without ever touching disk.

    Single-process stand-in for a 2-rank resume: a ``world_rank=1`` manager runs
    the real ``load_from_checkpoint`` with ``dist`` collectives stubbed so rank 0
    "broadcasts" the decision. The broadcast stub adapts to whatever rank 1
    hands it, so the test is meaningful against both the fixed code (rank 1 sends
    a ``None`` placeholder and receives the deserialized checkpoint) and the
    pre-fix code (rank 1 sends a candidate list and receives a path list, then
    loads it itself). The invariant: rank 1 must never call ``torch.load`` on a
    checkpoint file.
    """
    # Rank 0's on-disk checkpoint, and the object it would broadcast.
    mgr0, _ = _make_manager(tmp_path)
    mgr0.save_checkpoint(epoch=5, val_loss_avg=0.5, extras={"train_mask_values": [3]})
    good_ckpt = torch.load(mgr0.last_ckpt_path, map_location="cpu", weights_only=False)

    # A peer rank restoring from the same directory.
    mgr1, _ = _make_manager(tmp_path)
    mgr1.world_rank = 1
    mgr1.dist_enabled = True

    def fake_broadcast(objs, src=0):
        # Supply whatever rank 0 would have sent, matched to the payload shape
        # the code under test broadcasts.
        payload = objs[0]
        if payload is None:
            # Fixed code: rank 0 sends the (status, checkpoint) decision.
            objs[0] = ("ok", good_ckpt)
        elif isinstance(payload, list):
            # Pre-fix code: rank 0 sends the candidate PATH list.
            objs[0] = [mgr0.last_ckpt_path]

    monkeypatch.setattr(dist, "broadcast_object_list", fake_broadcast)
    monkeypatch.setattr(dist, "barrier", lambda *a, **k: None)

    ckpt_loads = []
    real_load = torch.load

    def spy_load(path, *args, **kwargs):
        ckpt_loads.append(str(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)

    start_epoch = mgr1.load_from_checkpoint()

    # The peer restored the broadcast state (resume at epoch 6) ...
    assert start_epoch == 6
    assert mgr1.restored_extras.get("train_mask_values") == [3]
    # ... without ever reading the checkpoint file itself.
    assert ckpt_loads == []


@_requires_gloo
def test_resume_only_rank0_reads_disk(tmp_path):
    """Under 2 gloo ranks, only rank 0 reads the checkpoint file on resume.

    A single good checkpoint (epoch 3) is written, then both ranks resume. Rank
    0 loads once and broadcasts; the peer must receive the state over the
    process group and perform zero checkpoint-file reads. Both ranks resume at
    epoch 4.
    """
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rc, out, err = mpi_runner.torchrun_gloo(
        str(RESUME_RANK_SCRIPT),
        n=2,
        timeout=90,
        env={"CKPT_DIR": str(ckpt_dir), "CKPT_MODE": "read_guard"},
    )

    done = set(re.findall(r"RANK (\d+) DONE", out))
    assert rc == 0 and {"0", "1"} <= done, (
        f"expected clean 2-rank completion, rc={rc}\n"
        f"stdout:\n{out}\nstderr:\n{err[-3000:]}"
    )

    loads = {r: int(n) for r, n in re.findall(r"RANK (\d+) LOADS (\d+)", out)}
    epochs = {r: int(n) for r, n in re.findall(r"RANK (\d+) START_EPOCH (\d+)", out)}
    assert loads.get("0", 0) >= 1, f"rank 0 should read the checkpoint\nstdout:\n{out}"
    assert loads.get("1", -1) == 0, (
        f"non-zero rank must not read the checkpoint file from disk; got "
        f"{loads.get('1')} read(s)\nstdout:\n{out}"
    )
    assert epochs.get("0") == 4 and epochs.get("1") == 4, (
        f"both ranks must resume at epoch 4\nstdout:\n{out}"
    )


@_requires_gloo
def test_resume_corruption_fallback_multirank(tmp_path):
    """Corruption fallback survives the rank-0-load + broadcast change.

    ``last`` (epoch 2) is truncated and ``best`` (epoch 1) is intact. Rank 0
    must fall through the corrupt ``last`` -- renaming it ``*.corrupt`` -- and
    load ``best``; the peer still performs zero reads. Both ranks resume at the
    best's epoch (2).
    """
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rc, out, err = mpi_runner.torchrun_gloo(
        str(RESUME_RANK_SCRIPT),
        n=2,
        timeout=90,
        env={"CKPT_DIR": str(ckpt_dir), "CKPT_MODE": "corrupt_fallback"},
    )

    done = set(re.findall(r"RANK (\d+) DONE", out))
    assert rc == 0 and {"0", "1"} <= done, (
        f"expected clean 2-rank completion, rc={rc}\n"
        f"stdout:\n{out}\nstderr:\n{err[-3000:]}"
    )

    loads = {r: int(n) for r, n in re.findall(r"RANK (\d+) LOADS (\d+)", out)}
    epochs = {r: int(n) for r, n in re.findall(r"RANK (\d+) START_EPOCH (\d+)", out)}
    corrupt = dict(re.findall(r"RANK (\d+) CORRUPT_EXISTS (\d+)", out))
    best = dict(re.findall(r"RANK (\d+) BEST_CONSULTED (\d+)", out))

    # Rank 0 read (last attempt + best) and quarantined the corrupt last.
    assert loads.get("0", 0) >= 1, f"rank 0 should read on fallback\nstdout:\n{out}"
    assert best.get("0") == "1", f"rank 0 should consult best\nstdout:\n{out}"
    assert corrupt.get("0") == "1", (
        f"the corrupt last should be renamed aside\nstdout:\n{out}"
    )
    # The peer still never reads the file.
    assert loads.get("1", -1) == 0, (
        f"non-zero rank must not read on fallback either\nstdout:\n{out}"
    )
    # Both ranks agree on the best's resume epoch (epoch 1 -> start 2).
    assert epochs.get("0") == 2 and epochs.get("1") == 2, (
        f"both ranks must resume from best at epoch 2\nstdout:\n{out}"
    )
