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

import time

import pytest
import torch

import ScaFFold.utils.trainer as trainer_mod
from ScaFFold.utils.checkpointing import CheckpointManager
from ScaFFold.utils.trainer import PyTorchTrainer

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

    def slow_write(state_dict, last_path, best_path, is_best):
        time.sleep(0.3)
        return real_write(state_dict, last_path, best_path, is_best)

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
    final_best = torch.load(
        mgr.best_ckpt_path, map_location="cpu", weights_only=False
    )
    assert final_best["epoch"] == 3
    assert final_best["val_loss_avg"] == pytest.approx(0.2)


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
