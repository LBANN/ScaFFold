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
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist


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
        if self.async_save and self.world_rank == 0:
            # We only need 1 worker for serializing writes
            self.executor = ThreadPoolExecutor(max_workers=1)

        # Ensure base directory exists (Rank 0 only)
        if self.world_rank == 0:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self, train_from_scratch: bool) -> None:
        """Clear existing checkpoints if training from scratch."""
        # Ensure any pending async saves are finished before deleting
        self.wait_for_save()

        if not train_from_scratch:
            self._barrier()
            return

        if self.world_rank == 0:
            for p in (self.last_ckpt_path, self.best_ckpt_path):
                if p.exists():
                    try:
                        p.unlink()
                        self._log(f"Removed existing checkpoint: {p}")
                    except Exception as e:
                        self._log(f"Failed to remove {p}: {e}")
        self._barrier()

    def wait_for_save(self):
        """Blocks until the background save (if any) is complete."""
        if self.future is not None:
            # check if running
            if not self.future.done():
                self._log("Waiting for background checkpoint save to complete...")
            try:
                self.future.result()  # Blocks and raises exceptions if any occurred
            except Exception as e:
                self._log(f"Background save failed with error: {e}")
            self.future = None

    def snapshot_training_state(self) -> Dict[str, Any]:
        """Capture mutable in-memory training state without writing a checkpoint."""
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        return {
            "model_state_dict": self._clone_state_dict(model_ref.state_dict()),
            "optimizer_state_dict": self._clone_state_dict(self.optimizer.state_dict())
            if self.optimizer
            else None,
            "scheduler_state_dict": self._clone_state_dict(self.scheduler.state_dict())
            if self.scheduler
            else None,
            "grad_scaler_state_dict": self._clone_state_dict(
                self.grad_scaler.state_dict()
            )
            if self.grad_scaler
            else None,
            "model_training": model_ref.training,
            **self._get_rng_snapshot(),
        }

    def restore_training_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore an in-memory training snapshot."""
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
        self.wait_for_save()  # Safety: don't load while writing

        # 1. Rank 0 is the sole reader: it selects the newest readable
        # checkpoint and deserializes it once, then broadcasts the loaded
        # object to the other ranks. Because DDP replicates identical state
        # across ranks, a single read + broadcast avoids an N-way concurrent
        # read of one (multi-GB) file from the shared filesystem on restart --
        # a restart I/O storm that serializes on the parallel FS. Peer ranks
        # therefore never open the checkpoint files at all.
        result = self._select_and_load() if self.world_rank == 0 else None
        status, payload = self._broadcast_obj(result)

        # 2. Every rank acts on the same decision rank 0 reached.
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
        """
        is_best = False
        if self.world_rank == 0:
            # Decide is_best from the cached best loss (single source of truth),
            # not by re-reading checkpoint_best.pth from disk. The cache is
            # seeded once at construction and updated below, so the decision
            # never races the background writer that may still be replacing the
            # best checkpoint in async mode.
            if val_loss_avg < self.best_val_loss:
                is_best = True
                self.best_val_loss = val_loss_avg

        if self.world_rank == 0:
            # 1. Wait for previous async save to prevent OOM or race
            if self.async_save:
                self.wait_for_save()

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

            # Record the epoch being written so callers can tell whether the
            # last completed epoch has already been checkpointed.
            self.last_saved_epoch = epoch

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

        # Broadcast result (for logging elsewhere)
        is_best = self._broadcast_obj(is_best)

        # Barrier: ensure Rank 0 has finished the "Snapshot" phase before anyone continues.
        # Even in async mode, we must wait for the CPU transfer to finish.
        self._barrier()
        return is_best

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
            # Save 'best' atomically (re-serialize rather than copy a file that
            # a concurrent writer might still be replacing).
            if is_best:
                cls._atomic_save(state_dict, best_path)
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
        otherwise the async writer would serialize tensors the training loop
        keeps mutating in place.
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

    def _clone_state_dict(self, obj):
        """Recursively clone tensors so in-memory snapshots are isolated."""
        if torch.is_tensor(obj):
            return obj.detach().clone()
        elif isinstance(obj, dict):
            return {k: self._clone_state_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clone_state_dict(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._clone_state_dict(v) for v in obj)
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
