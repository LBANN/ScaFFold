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

import copy
import math
import random
import shutil
import time
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

    def load_from_checkpoint(self) -> int:
        """Load the latest checkpoint. Returns start_epoch (default 1)."""
        self.wait_for_save()  # Safety: don't load while writing

        # 1. Decision phase
        candidate = None
        if self.world_rank == 0:
            if self.last_ckpt_path.exists():
                candidate = self.last_ckpt_path
            elif self.best_ckpt_path.exists():
                candidate = self.best_ckpt_path

        candidate = self._broadcast_obj(candidate)
        if not candidate:
            return 1

        self._log(f"Loading checkpoint from {candidate}")

        # 2. Load to CPU
        try:
            checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception as e:
            self._log(f"Failed to load checkpoint {candidate}: {e}")
            raise e

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

    def save_checkpoint(
        self, epoch: int, val_loss_avg: float, extras: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save checkpoint.
        If async_save is True, this returns immediately after CPU transfer.
        """
        is_best = False
        if self.world_rank == 0:
            current_best_loss = math.inf
            if self.best_ckpt_path.exists():
                try:
                    prev = torch.load(
                        self.best_ckpt_path, map_location="cpu", weights_only=False
                    )
                    current_best_loss = prev.get("val_loss_avg", math.inf)
                except Exception:
                    pass

            if val_loss_avg < current_best_loss:
                is_best = True

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
                )
                self._log(f"Async checkpoint offloaded to background thread.")
            else:
                # Synchronous Save
                self._write_to_disk(
                    state_dict, self.last_ckpt_path, self.best_ckpt_path, is_best
                )

        # Broadcast result (for logging elsewhere)
        is_best = self._broadcast_obj(is_best)

        # Barrier: ensure Rank 0 has finished the "Snapshot" phase before anyone continues.
        # Even in async mode, we must wait for the CPU transfer to finish.
        self._barrier()
        return is_best

    @staticmethod
    def _write_to_disk(state_dict, last_path, best_path, is_best):
        """Worker function to perform actual disk I/O."""
        # Save 'last'
        try:
            torch.save(state_dict, last_path)
        except Exception as e:
            print("Saving checkpoint failed. Continuing...")
            print(e)
        # Save 'best' (copy logic)
        if is_best:
            # Copy is often faster than re-serializing
            if last_path.exists():
                shutil.copyfile(last_path, best_path)
            else:
                try:
                    torch.save(state_dict, best_path)
                except Exception as e:
                    print("Saving checkpoint failed. Continuing...")
                    print(e)

    def _transfer_dict_to_cpu(self, obj):
        """Recursively move tensors to CPU."""
        if torch.is_tensor(obj):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: self._transfer_dict_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._transfer_dict_to_cpu(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._transfer_dict_to_cpu(v) for v in obj)
        else:
            return obj

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
            snap["rng_state_pytorch_cuda"] = torch.cuda.get_rng_state()
        try:
            snap["rng_state_numpy"] = np.random.get_state()
        except ImportError:
            pass
        try:
            snap["rng_state_python"] = random.getstate()
        except:
            pass
        return snap

    def _restore_rng(self, snap: Dict[str, Any]):
        try:
            if "rng_state_pytorch" in snap:
                torch.set_rng_state(snap["rng_state_pytorch"])
            if "rng_state_pytorch_cuda" in snap and torch.cuda.is_available():
                torch.cuda.set_rng_state(snap["rng_state_pytorch_cuda"])
            if "rng_state_numpy" in snap:
                np.random.set_state(snap["rng_state_numpy"])
            if "rng_state_python" in snap:
                random.setstate(snap["rng_state_python"])
        except Exception as e:
            self._log(f"RNG Restore warning: {e}")
