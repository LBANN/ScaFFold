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

import math
import os
import random
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist


class CheckpointSaveError(RuntimeError):
    """A rank-0 checkpoint operation failed (a write, a cleanup, a load).

    Raised identically on every rank. Only rank 0 touches the run directory,
    but its outcome is broadcast, so the peers report the real disk error
    instead of the unmatched-collective symptom (an opaque gloo transport
    error, or an NCCL watchdog timeout minutes later) that a rank-0-only raise
    produces.
    """


class CheckpointManager:
    """
    Checkpoint Manager for DDP/Single-Process.
    Supports Synchronous (Blocking) and Asynchronous (Non-blocking) saving.

    Args:
        async_save (bool): If True, Rank 0 offloads disk I/O to a background thread.
                           Requires sufficient CPU RAM to hold a full model copy.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_scaler: Optional[torch.amp.GradScaler] = None,
        base_dir: str,
        log: Optional[Any] = None,
        world_rank: int = 0,
        dist_enabled: bool = False,
        async_save: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.base_dir = Path(base_dir)
        self.log = log
        self.world_rank = world_rank
        self.dist_enabled = dist_enabled
        self.async_save = async_save

        # Paths
        self.last_ckpt_path = self.base_dir / "checkpoint_last.pth"
        self.best_ckpt_path = self.base_dir / "checkpoint_best.pth"

        self.restored_extras: Dict[str, Any] = {}

        # Single source of truth for the best validation loss seen so far.
        # Seeded once from disk (if a best checkpoint already exists, e.g. on
        # resume) so a restarted run does not treat its first epoch as best
        # unless it truly improves on the previously saved best. Reading it
        # here avoids a per-save disk probe that would otherwise race the
        # background writer in async mode.
        # Epoch of the most recently written checkpoint (None until one is
        # saved or loaded); lets callers avoid a redundant final save when the
        # last completed epoch was already checkpointed.
        self.last_saved_epoch: Optional[int] = None

        self.best_val_loss = math.inf
        if self.world_rank == 0 and self.best_ckpt_path.exists():
            try:
                prev = torch.load(
                    self.best_ckpt_path, map_location="cpu", weights_only=False
                )
                self.best_val_loss = prev.get("val_loss_avg", math.inf)
            except Exception as e:
                self._log(
                    f"Could not read best val loss from {self.best_ckpt_path}: {e}"
                )

        # Async handling
        self.executor = None
        self.future = None
        # The exception behind the most recently reported save failure, kept
        # only so rank 0 can chain it (and its traceback) onto the
        # CheckpointSaveError every rank raises.
        self._save_error_exc: Optional[BaseException] = None
        if self.async_save and self.world_rank == 0:
            # We only need 1 worker for serializing writes
            self.executor = ThreadPoolExecutor(max_workers=1)

        # Ensure base directory exists (Rank 0 only)
        if self.world_rank == 0:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._sweep_orphaned_tmp_files()
            except Exception as e:
                # Construction is rank-0-only work outside any collective, so a
                # raise here aborts rank 0 alone and leaves the peers waiting in
                # the manager's first collective (``cleanup``). The sweep only
                # reclaims space, so a directory that cannot be listed (a stale
                # NFS/Lustre handle, a permissions oddity) degrades to a warning
                # rather than taking the job down asymmetrically.
                self._log(
                    f"Could not sweep orphaned checkpoint temp files in "
                    f"{self.base_dir}: {type(e).__name__}: {e}"
                )

    def cleanup(self, train_from_scratch: bool) -> None:
        """Clear existing checkpoints if training from scratch.

        Rank-symmetric, like every other collective point here: any pending
        async write is drained, the rank-0 deletion is fenced, and whichever
        failed is broadcast, so a failure raises on all ranks together (see
        ``save_checkpoint``). The broadcast payload is ``(phase, description)``
        so the peers -- which saw neither the write nor the deletion -- report
        the same operation rank 0 did.
        """
        # Ensure any pending async save is finished before deleting.
        error = self._drain_pending_save()
        failure = None if error is None else ("save", error)

        if train_from_scratch:
            # Drop the cached state that described the run being deleted. Both
            # fields are seeded from disk (or a previous save), so keeping them
            # would let a deleted run's best gate this run's is_best decisions
            # -- the fresh run would then never write a best checkpoint until
            # it beat a score no file backs any more, leaving it with no
            # best-checkpoint fallback.
            self.best_val_loss = math.inf
            self.last_saved_epoch = None

            if self.world_rank == 0:
                # Rank 0 alone touches the filesystem here, while every peer is
                # already committed to the broadcast below. Individual unlinks
                # are tolerated inside, but the glob/stat around them can still
                # raise on a shared filesystem (ESTALE, EACCES), and raising in
                # this window strands the peers in an unmatched collective.
                # Report the failure through the broadcast, like a failed write.
                try:
                    self._remove_checkpoint_files()
                except Exception as e:
                    if failure is None:
                        self._save_error_exc = e
                        failure = ("cleanup", f"{type(e).__name__}: {e}")
                    else:
                        # A drained write already failed; that outcome is the
                        # one being reported, so this is only logged.
                        self._log(f"Clearing existing checkpoints also failed: {e}")

        failure = self._broadcast_obj(failure)
        self._barrier()
        if failure is not None:
            phase, description = failure
            self._raise_save_error(description, phase=phase)

    def _remove_checkpoint_files(self) -> None:
        """Delete this run's checkpoint files and debris (rank 0 only).

        Besides the two canonical files, a run directory can hold
        ``checkpoint_*.pth.tmp.<pid>`` (an interrupted write whose Python-level
        cleanup never ran) and ``checkpoint_*.pth.corrupt`` (a checkpoint
        quarantined on resume). Both are full-checkpoint-sized and nothing else
        removes them, so a "from scratch" cleanup that left them would claim to
        have cleared the checkpoints while keeping their bytes on disk.
        """
        debris = sorted(
            set(self.base_dir.glob("checkpoint_*.pth.tmp.*"))
            | set(self.base_dir.glob("checkpoint_*.pth.corrupt"))
        )
        for p in (self.last_ckpt_path, self.best_ckpt_path, *debris):
            if p.exists():
                try:
                    p.unlink()
                    self._log(f"Removed existing checkpoint: {p}")
                except Exception as e:
                    self._log(f"Failed to remove {p}: {e}")

    def _sweep_orphaned_tmp_files(self) -> None:
        """Delete temp files stranded by checkpoint writes that were killed.

        ``_atomic_save`` unlinks its ``<name>.tmp.<pid>`` file when the write
        raises, but a SIGKILL (walltime, node failure) skips that Python-level
        cleanup and strands a full-checkpoint-sized file. These accumulate one
        per killed pid: the kill/restart cycle that produces them always takes
        the *resume* path, so ``cleanup(train_from_scratch=True)`` never gets a
        chance to clear them.

        Sweeping at construction is safe because run directories are per-run
        and not shared between concurrently running jobs, so any temp
        file here belongs to a dead process -- except one from this pid, which
        another manager in this process could still be writing.

        ``*.corrupt`` files are deliberately left alone here:
        ``_quarantine_corrupt`` renames onto a fixed name, so at most two can
        ever exist (they cannot accumulate) and they are the only evidence of
        what a resume discarded. The from-scratch cleanup removes them.
        """
        own_suffix = f".tmp.{os.getpid()}"
        for path in sorted(self.base_dir.glob("checkpoint_*.pth.tmp.*")):
            if path.name.endswith(own_suffix):
                continue
            try:
                path.unlink()
                self._log(f"Removed orphaned checkpoint temp file: {path}")
            except OSError as e:
                self._log(f"Failed to remove {path}: {e}")

    def wait_for_save(self):
        """Block until the background save (if any) is complete.

        The writer's exception is re-raised rather than logged and dropped: a
        save that could not complete has to surface, otherwise the manager
        state (and the process exit code) claims a checkpoint that is not on
        disk. The future is consumed exactly once, so the failure is reported
        at exactly one point.

        Callers that are inside a collective region must not let this
        propagate directly -- use ``_drain_pending_save`` instead, per the
        collective invariant documented on ``save_checkpoint``.
        """
        if self.future is None:
            return
        # Clear the handle *before* blocking on it so a failed write is
        # reported once and does not re-raise at some later, arbitrary point.
        future, self.future = self.future, None
        if not future.done():
            self._log("Waiting for background checkpoint save to complete...")
        future.result()  # Blocks and re-raises whatever the writer raised

    def _drain_pending_save(self) -> Optional[str]:
        """Consume the in-flight async save, reporting failure without raising.

        Returns a description of the writer's failure, or ``None``. Only rank 0
        ever has a pending write, so raising here directly would leave the
        peers blocked in the next collective; callers broadcast this result and
        raise on every rank together.
        """
        try:
            self.wait_for_save()
        except Exception as e:
            self._save_error_exc = e
            return f"{type(e).__name__}: {e}"
        return None

    def _raise_save_error(self, description: str, phase: str = "save") -> None:
        """Raise a broadcast rank-0 failure on this rank.

        ``phase`` names the operation that failed (``save``, ``cleanup``,
        ``load``) so the message describes what actually went wrong. Rank 0
        chains the original exception so its traceback survives; the peers never
        saw it and raise the same message on its own.
        """
        cause, self._save_error_exc = self._save_error_exc, None
        raise CheckpointSaveError(
            f"Checkpoint {phase} failed on rank 0: {description}"
        ) from cause

    def finalize_saves(self) -> None:
        """Consume the outcome of the run's last save before the run ends.

        Nothing touches the manager after the training loop, so an
        asynchronous write that failed there would never be observed: the
        process would exit successfully having written no checkpoint (or left
        a stale one), the benchmark would report success, and a later
        ``--restart`` would resume from the wrong epoch or fail its pre-check.
        Rank-symmetric, like ``save_checkpoint``.
        """
        error = self._drain_pending_save()
        error = self._broadcast_obj(error)
        self._barrier()
        if error is not None:
            self._raise_save_error(error)

    def snapshot_training_state(self) -> Dict[str, Any]:
        """Capture mutable in-memory training state without writing a checkpoint.

        The copy is taken on the *host*. Cloning device-to-device instead would
        hold roughly 3x parameter bytes of accelerator memory (the weights plus
        Adam's two moment buffers) for as long as the snapshot lives -- which is
        the whole of warmup, precisely where the run establishes its peak
        memory. ``restore_training_state`` puts the state back on the model's
        own device, so the detour is invisible to callers.
        """
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        return {
            "model_state_dict": self._transfer_dict_to_cpu(model_ref.state_dict()),
            "optimizer_state_dict": self._transfer_dict_to_cpu(
                self.optimizer.state_dict()
            )
            if self.optimizer
            else None,
            "scheduler_state_dict": self._transfer_dict_to_cpu(
                self.scheduler.state_dict()
            )
            if self.scheduler
            else None,
            "grad_scaler_state_dict": self._transfer_dict_to_cpu(
                self.grad_scaler.state_dict()
            )
            if self.grad_scaler
            else None,
            "model_training": model_ref.training,
            **self._get_rng_snapshot(),
        }

    def restore_training_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore an in-memory training snapshot.

        The snapshot is host-resident (see ``snapshot_training_state``); the
        ``load_state_dict`` calls below copy into the live parameters and move
        optimizer state onto each parameter's device, so the restored state
        ends up exactly where it started.
        """
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        model_ref.load_state_dict(snapshot["model_state_dict"])

        if self.optimizer and snapshot.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])

        if self.scheduler and snapshot.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(snapshot["scheduler_state_dict"])

        if self.grad_scaler and snapshot.get("grad_scaler_state_dict") is not None:
            self.grad_scaler.load_state_dict(snapshot["grad_scaler_state_dict"])

        self._restore_rng(snapshot)
        model_ref.train(snapshot.get("model_training", True))

        if self.optimizer:
            self.optimizer.zero_grad(set_to_none=True)

    def load_from_checkpoint(self, require_checkpoint: bool = False) -> int:
        """Load the latest checkpoint. Returns start_epoch (default 1).

        With ``require_checkpoint`` (an explicit ``--restart``), a missing
        checkpoint raises instead of silently starting over.
        """
        # Safety: don't load while writing. A failure from that write is
        # folded into the decision broadcast below rather than raised here, so
        # rank 0 never abandons its peers inside the broadcast.
        error = self._drain_pending_save()

        # 1. Rank 0 is the sole reader: it selects the newest readable
        # checkpoint and deserializes it once, then broadcasts the loaded
        # object to the other ranks. Because DDP replicates identical state
        # across ranks, a single read + broadcast avoids an N-way concurrent
        # read of one (multi-GB) file from the shared filesystem on restart --
        # a restart I/O storm that serializes on the parallel FS. Peer ranks
        # therefore never open the checkpoint files at all.
        result = None
        if self.world_rank == 0:
            if error is not None:
                result = ("save_failed", error)
            else:
                # The selection itself stats, deserializes and renames files
                # while the peers are already blocked in the broadcast below, so
                # anything it raises (a stale handle on ``exists``, a rename
                # denied, an unpickling MemoryError) has to become a decision
                # rather than a rank-0-only death.
                try:
                    result = self._select_and_load()
                except Exception as e:
                    self._save_error_exc = e
                    result = ("load_failed", f"{type(e).__name__}: {e}")
        status, payload = self._broadcast_obj(result)

        # 2. Every rank acts on the same decision rank 0 reached.
        if status == "save_failed":
            self._raise_save_error(payload)
        if status == "load_failed":
            self._raise_save_error(payload, phase="load")
        if status == "empty":
            if require_checkpoint:
                # An explicit restart must resume real state; silently
                # retraining from scratch would waste the allocation.
                raise FileNotFoundError(
                    "Restart requested but no checkpoint was found. "
                    f"Expected {self.last_ckpt_path} or {self.best_ckpt_path}."
                )
            # No checkpoint on disk anywhere (e.g. the first run with
            # train_from_scratch disabled): start a fresh run.
            return 1
        if status == "unreadable":
            # Every candidate failed to deserialize on rank 0; fail identically
            # on all ranks instead of leaving peers to hang or silently proceed.
            raise RuntimeError(
                "No loadable checkpoint found; all candidates were unreadable: "
                f"{payload}"
            )
        checkpoint = payload

        # 3. Restore weights
        model_to_load = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        if "model_state_dict" in checkpoint:
            model_to_load.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            if checkpoint["scheduler_state_dict"] is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.grad_scaler and "grad_scaler_state_dict" in checkpoint:
            if checkpoint["grad_scaler_state_dict"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

        self._restore_rng(checkpoint)
        start_epoch = checkpoint.get("epoch", 0) + 1

        # Refresh the best-loss cache and last-saved epoch from the loaded
        # checkpoint so a resumed run keeps a consistent view of both.
        loaded_epoch = checkpoint.get("epoch")
        if loaded_epoch is not None:
            self.last_saved_epoch = loaded_epoch
        if "val_loss_avg" in checkpoint:
            self.best_val_loss = min(
                self.best_val_loss, checkpoint.get("val_loss_avg", math.inf)
            )

        # Restore extras
        self.restored_extras = {
            k: v
            for k, v in checkpoint.items()
            if k
            not in {
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "grad_scaler_state_dict",
                "epoch",
                "rng_state_pytorch",
                "rng_state_pytorch_cuda",
                "rng_state_numpy",
                "rng_state_python",
            }
        }

        self._barrier()
        return start_epoch

    def _select_and_load(self):
        """Pick and deserialize the newest readable checkpoint (rank 0 only).

        Prefers the most recent 'last', then falls back to 'best'. A candidate
        that fails to deserialize (e.g. a 'last' truncated by a mid-write kill)
        is renamed aside with a ``.corrupt`` suffix so it is not retried on the
        next restart, then the next candidate is tried.

        Returns a ``(status, payload)`` decision that is safe to broadcast:

        * ``("empty", None)``           -- no checkpoint files exist;
        * ``("ok", checkpoint_dict)``   -- a candidate deserialized cleanly;
        * ``("unreadable", [paths])``   -- every candidate was corrupt.

        The filesystem calls around those decisions (``exists``, the quarantine
        rename) can still fail outright; the caller runs this inside a guard
        that turns such a failure into a broadcast ``("load_failed", ...)``
        decision, because raising here would strand the peers.
        """
        candidates = []
        if self.last_ckpt_path.exists():
            candidates.append(self.last_ckpt_path)
        if self.best_ckpt_path.exists():
            candidates.append(self.best_ckpt_path)

        if not candidates:
            return ("empty", None)

        for path in candidates:
            self._log(f"Loading checkpoint from {path}")
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                return ("ok", checkpoint)
            except Exception as e:
                self._log(f"Failed to load checkpoint {path}: {e}")
                self._quarantine_corrupt(path)
                self._log("Falling back to the next available checkpoint.")

        return ("unreadable", [str(p) for p in candidates])

    def save_checkpoint(
        self, epoch: int, val_loss_avg: float, extras: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save checkpoint.
        If async_save is True, this returns immediately after CPU transfer.

        Collective invariant: every rank posts exactly one ``_broadcast_obj``
        followed by exactly one ``_barrier`` on every path through this method,
        failures included. Rank 0 is the sole writer, but it never raises
        before those collectives: what gets broadcast is the write's *outcome*
        -- ``is_best`` on success, or an error description on failure,
        including a *previous* asynchronous write whose failure is surfaced
        here, at the next collective point. Every rank then raises the same
        ``CheckpointSaveError`` together. Raising on rank 0 before the
        broadcast would strand the peers in an unmatched collective, where a
        plain disk error resurfaces as a gloo transport error or an NCCL
        watchdog timeout that hides the real cause.
        """
        # Non-zero ranks contribute nothing; their placeholder is overwritten
        # by rank 0's outcome in the broadcast below (and is a harmless no-op
        # in the degenerate non-distributed case).
        outcome = ("ok", False)
        if self.world_rank == 0:
            outcome = self._rank0_save(epoch, val_loss_avg, extras)

        status, payload = self._broadcast_obj(outcome)

        # Barrier: ensure Rank 0 has finished the "Snapshot" phase before anyone continues.
        # Even in async mode, we must wait for the CPU transfer to finish.
        self._barrier()

        if status == "error":
            self._raise_save_error(payload)
        return payload

    def _rank0_save(self, epoch, val_loss_avg, extras):
        """Perform rank 0's write and REPORT its outcome; never raises.

        Returns ``("ok", is_best)`` or ``("error", description)``. The caller
        broadcasts that outcome so every rank fails together -- see the
        collective invariant on ``save_checkpoint``.
        """
        try:
            # 1. Wait for previous async save to prevent OOM or race. If that
            # write failed, this is where it surfaces.
            if self.async_save:
                self.wait_for_save()

            # Decide is_best from the cached best loss (single source of truth),
            # not by re-reading checkpoint_best.pth from disk. The cache is
            # seeded once at construction and updated below, so the decision
            # never races the background writer that may still be replacing the
            # best checkpoint in async mode.
            is_best = val_loss_avg < self.best_val_loss

            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )

            # Construct dictionary
            state_dict = {
                "epoch": epoch,
                "val_loss_avg": val_loss_avg,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                if self.optimizer
                else None,
                "grad_scaler_state_dict": self.grad_scaler.state_dict()
                if self.grad_scaler
                else None,
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
                **self._get_rng_snapshot(),
            }
            if extras:
                state_dict.update(extras)

            # 2. Save Trigger
            if self.async_save:
                # We must clone tensors to CPU now, because training will resume
                # and modify the GPU tensors while the thread is writing.
                cpu_state_dict = self._transfer_dict_to_cpu(state_dict)

                # Submit to background thread
                self.future = self.executor.submit(
                    self._write_to_disk,
                    cpu_state_dict,
                    self.last_ckpt_path,
                    self.best_ckpt_path,
                    is_best,
                    self.log,
                )
                self._log("Async checkpoint offloaded to background thread.")
            else:
                # Synchronous Save
                self._write_to_disk(
                    state_dict,
                    self.last_ckpt_path,
                    self.best_ckpt_path,
                    is_best,
                    self.log,
                )

            # Only now claim the save: the bytes are on disk (sync) or handed
            # to the writer (async). Recording the epoch lets callers tell
            # whether the last completed epoch has already been checkpointed.
            # An async write that fails later aborts the run at the next
            # collective point, so this optimistic state is never observed by
            # a run that keeps going.
            if is_best:
                self.best_val_loss = val_loss_avg
            self.last_saved_epoch = epoch
            return ("ok", is_best)
        except Exception as e:
            self._save_error_exc = e
            return ("error", f"{type(e).__name__}: {e}")

    @staticmethod
    def _atomic_save(state_dict, path):
        """Serialize to a temp file in the same directory, fsync, then replace.

        Readers therefore never see a half-written checkpoint, and a crash
        mid-write cannot corrupt the previous good file at ``path`` (the
        replace is atomic; only the temp file is ever partial).
        """
        path = Path(path)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            with open(tmp_path, "wb") as f:
                torch.save(state_dict, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            # Clean up the partial temp file so it cannot be mistaken for a
            # real checkpoint, then re-raise so the failure is not silent.
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_copy(src, dst):
        """Copy an already-committed checkpoint onto another name atomically.

        Same discipline as ``_atomic_save`` -- copy into a temp file in the
        same directory, fsync it, then ``os.replace`` -- so ``dst`` is never
        observed half-written and a crash mid-copy cannot damage the previous
        good file there.
        """
        src = Path(src)
        dst = Path(dst)
        tmp_path = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
        try:
            with open(src, "rb") as fsrc, open(tmp_path, "wb") as fdst:
                # A large buffer keeps the syscall count low on parallel
                # filesystems (the default 64 KiB means ~1k read/write pairs
                # per 64 MiB checkpoint).
                shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
                fdst.flush()
                os.fsync(fdst.fileno())
            os.replace(tmp_path, dst)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise

    @classmethod
    def _write_to_disk(cls, state_dict, last_path, best_path, is_best, log):
        """Worker function to perform actual disk I/O.

        Writes are atomic (temp file + os.replace). Failures are logged with a
        full traceback and re-raised rather than swallowed, so a save that
        cannot complete surfaces to the trainer instead of silently producing
        no checkpoint.
        """
        try:
            # Save 'last' atomically.
            cls._atomic_save(state_dict, last_path)
            # 'best' is byte-identical to the 'last' just committed, so copy
            # that file instead of pickling the same state a second time. The
            # saving is the serialization CPU (pickle + zip of the full state
            # dict); the filesystem traffic is roughly a wash -- the copy
            # writes the same bytes and adds a read (measured ~1.02x faster on
            # Lustre). Trade-off: 'best' is now a byte copy of 'last', so a
            # silently corrupted 'last' write would propagate into 'best'
            # rather than being an independent serialization.
            #
            # There is no concurrent writer to race: checkpoint writes are
            # serialized through a single writer -- the caller's thread in sync
            # mode, or the one-worker ThreadPoolExecutor in async mode, whose
            # previous write ``_rank0_save`` drains before submitting the next
            # -- and only rank 0 ever writes. So this very thread performed the
            # ``os.replace`` onto ``last_path`` a moment ago and nothing else
            # can be replacing it now.
            if is_best:
                cls._atomic_copy(last_path, best_path)
        except Exception:
            if log is not None:
                log.error("Saving checkpoint failed:\n%s", traceback.format_exc())
            else:
                traceback.print_exc()
            raise

    def _transfer_dict_to_cpu(self, obj):
        """Recursively move tensors to CPU, cloning so the result is a true
        snapshot rather than an alias of live state.

        ``Tensor.cpu()`` is a no-op for tensors already on CPU (it returns the
        same object), so CPU-resident state must be cloned explicitly;
        otherwise the async writer -- or the warmup snapshot -- would keep an
        alias of tensors the training loop mutates in place.
        """
        if torch.is_tensor(obj):
            t = obj.detach()
            return t.clone() if t.device.type == "cpu" else t.cpu()
        elif isinstance(obj, dict):
            return {k: self._transfer_dict_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._transfer_dict_to_cpu(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._transfer_dict_to_cpu(v) for v in obj)
        else:
            return obj

    def _quarantine_corrupt(self, path):
        """Rename an unreadable checkpoint aside so it is not retried on the
        next restart (a persistent corrupt 'last' would otherwise break every
        subsequent resume)."""
        path = Path(path)
        corrupt_path = path.with_name(path.name + ".corrupt")
        try:
            os.replace(path, corrupt_path)
            self._log(f"Renamed corrupt checkpoint {path} -> {corrupt_path}")
        except OSError as e:
            self._log(f"Could not quarantine corrupt checkpoint {path}: {e}")

    def _barrier(self):
        if self.dist_enabled:
            dist.barrier()

    def _broadcast_obj(self, obj):
        if self.dist_enabled:
            objs = [obj]
            dist.broadcast_object_list(objs, src=0)
            return objs[0]
        return obj

    def _log(self, msg):
        if self.log:
            self.log.info(msg)
        elif self.world_rank == 0:
            print(msg)

    def _get_rng_snapshot(self) -> Dict[str, Any]:
        snap = {"rng_state_pytorch": torch.get_rng_state()}
        if torch.cuda.is_available():
            snap["rng_state_pytorch_cuda"] = torch.cuda.get_rng_state_all()
        try:
            snap["rng_state_numpy"] = np.random.get_state()
        except ImportError:
            pass
        try:
            snap["rng_state_python"] = random.getstate()
        except Exception:
            pass
        return snap

    def _restore_rng(self, snap: Dict[str, Any]):
        try:
            if "rng_state_pytorch" in snap:
                torch.set_rng_state(snap["rng_state_pytorch"])
            if "rng_state_pytorch_cuda" in snap and torch.cuda.is_available():
                cuda_state = snap["rng_state_pytorch_cuda"]
                if isinstance(cuda_state, list):
                    torch.cuda.set_rng_state_all(cuda_state)
                else:
                    torch.cuda.set_rng_state(cuda_state)
            if "rng_state_numpy" in snap:
                np.random.set_state(snap["rng_state_numpy"])
            if "rng_state_python" in snap:
                random.setstate(snap["rng_state_python"])
        except Exception as e:
            self._log(f"RNG Restore warning: {e}")
