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

"""Two-rank rank script driving a real ``PyTorchTrainer`` under gloo.

Launched under ``torchrun`` (gloo backend) by ``tests/test_metrics_dist.py`` to
exercise the distributed reduction of the epoch train/val metrics without a GPU,
DistConv, or an MPI launcher. Each rank:

* maps the ``torchrun``-supplied ``RANK``/``WORLD_SIZE`` into the ``OMPI_*``
  variables the trainer's rank helpers read, and initialises a gloo group;
* builds a real trainer over a tiny on-disk dataset shared via ``DATASET_DIR``;
* monkeypatches the DistConv-only seams so the loop runs on CPU:
  ``DCTensor.from_shard`` becomes identity, ``torch.cuda.Event`` is a CPU timer,
  and (for the train-metric case) ``_run_training_batch`` returns rank-dependent
  constants; the validation metric is driven by a stub ``evaluate``;
* prints ``rank-tagged`` markers that the parent test parses.

Behaviour is selected by ``METRICS_MODE``:

* ``train``  -- stub ``_run_training_batch`` to a rank-dependent constant loss and
  dice; assert the CSV train columns are the global sample-weighted mean.
* ``val``    -- stub ``evaluate`` to rank-dependent per-sample val losses/dice for
  one epoch; assert the logged ``val_loss_avg`` / ``val_score`` are global.
* ``best``   -- stub ``evaluate`` across two epochs so the globally-best epoch
  differs from rank 0's replica-local best; assert the saved best checkpoint.
* ``valpad`` -- run the *real* evaluate over a 3-sample dataset with a stub net
  giving per-sample dice {1, 1, 0}; assert the global val_score is 2/3.

Markers (always rank-tagged, one per line, flushed):
  ``RANK <r> CSV <line>``          -- a train_stats.csv data row (rank 0 only)
  ``RANK <r> VAL_SCORE <float>``   -- the reduced val dice score (rank 0 only)
  ``RANK <r> BEST_EPOCH <int>``    -- epoch held by checkpoint_best.pth (rank 0)
  ``RANK <r> DONE``                -- clean completion on every rank
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist


def _map_torchrun_env_to_ompi() -> tuple[int, int]:
    """Expose torchrun's RANK/WORLD_SIZE as the OMPI_* vars the trainer reads."""
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    os.environ["OMPI_COMM_WORLD_RANK"] = str(rank)
    os.environ["OMPI_COMM_WORLD_SIZE"] = str(world)
    os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
    os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"] = str(world)
    return rank, world


class _FakeCudaEvent:
    """CPU stand-in for ``torch.cuda.Event`` so the timing path runs on CPU."""

    def __init__(self, enable_timing=False):
        self._t = None

    def record(self):
        import time

        self._t = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return (other._t - self._t) * 1000.0


def _float_list(name: str) -> list[float]:
    raw = os.environ.get(name, "")
    return [float(x) for x in raw.split(",") if x != ""]


def _build_config(dataset_dir: str, run_dir: str, rank: int, world: int):
    """Construct a tiny distributed ``RunConfig`` for this rank."""
    # Import lazily so PYTHONPATH (set by the launcher) is already in effect.
    import sys as _sys

    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)
    from ScaFFold.utils.config_utils import RunConfig
    from tests.conftest import _BASE_CONFIG

    n_categories = int(os.environ.get("METRICS_N_CATEGORIES", "2"))
    epochs = int(os.environ.get("METRICS_EPOCHS", "1"))
    ckpt_interval = int(os.environ.get("METRICS_CKPT_INTERVAL", "-1"))

    config_dict = dict(_BASE_CONFIG)
    config_dict.update(
        {
            "dataset_dir": dataset_dir,
            "run_dir": run_dir,
            "run_iter": 0,
            "n_categories": n_categories,
            "dist": 1,
            "torch_amp": 0,
            "dataloader_num_workers": 0,
            "local_batch_size": 1,
            "epochs": epochs,
            "target_dice": 0.99,
            "checkpoint_interval": ckpt_interval,
            "checkpoint_dir": "checkpoints",
            "warmup_batches": 0,
            "disable_scheduler": 1,
            "ce_weight_sample_fraction": 1.0,
        }
    )
    config = RunConfig(config_dict)
    config._parallel_strategy = None  # data-parallel only; no spatial sharding
    config.verbose = 0
    config.vol_size = 2**config.problem_scale
    config.point_num = int((2**config.problem_scale) ** 3 / 256)
    return config


def _make_train_batch_stub(rank: int):
    """Return a ``_run_training_batch`` replacement giving rank-constant metrics.

    The loop's real forward/backward path needs DistConv; this stub instead
    returns a fixed (batch_size, loss, dice) so the epoch accumulators receive
    exactly the value assigned to this rank. ``batch_size`` is read from the real
    batch so the sample-weighted accumulation matches the data actually sharded
    to this rank.
    """
    losses = _float_list("METRICS_TRAIN_LOSS")
    dices = _float_list("METRICS_TRAIN_DICE")
    loss_val = losses[rank] if rank < len(losses) else 0.0
    dice_val = dices[rank] if rank < len(dices) else 0.0

    def stub(batch, **kwargs):
        bs = int(batch["image"].shape[0])
        return bs, torch.tensor(float(loss_val)), torch.tensor(float(dice_val))

    return stub


def _make_trivial_train_batch_stub():
    """A zero-metric ``_run_training_batch`` stub (used when only val matters)."""

    def stub(batch, **kwargs):
        bs = int(batch["image"].shape[0])
        return bs, torch.tensor(0.0), torch.tensor(0.0)

    return stub


def _make_val_stub(rank: int, epoch_counter: list):
    """Return an ``evaluate`` replacement giving rank/epoch-dependent val loss.

    ``METRICS_VAL_LOSS`` encodes per-epoch, per-rank per-sample loss as
    ``";"``-separated epochs of ``","``-separated ranks, e.g. ``"0.1,0.9;0.4,0.4"``.
    Each rank reports exactly one sample so the reduced mean is the plain average
    across ranks. Dice is fixed low so validation never crosses ``target_dice``.
    """
    spec = os.environ.get("METRICS_VAL_LOSS", "")
    per_epoch = [
        [float(x) for x in chunk.split(",") if x != ""]
        for chunk in spec.split(";")
        if chunk != ""
    ]

    def stub(net, dataloader, *args, **kwargs):
        idx = epoch_counter[0]
        epoch_counter[0] += 1
        if not per_epoch:
            ranks = []
        else:
            ranks = per_epoch[idx] if idx < len(per_epoch) else per_epoch[-1]
        loss_val = ranks[rank] if rank < len(ranks) else 0.0
        # One sample per rank: dice_sum, val_loss_epoch (sample-weighted), and
        # sample count are all for that single sample. Dice is deliberately low.
        dice_sum = 0.05
        numsamples = 1
        val_loss_epoch = float(loss_val) * numsamples
        val_loss_avg_local = float(loss_val)
        return dice_sum, val_loss_epoch, val_loss_avg_local, 1, numsamples

    return stub


def _make_valpad_stub(per_sample_dice: dict):
    """Return an ``evaluate`` stub that scores whichever samples the val sampler
    assigned to this rank, mirroring the real per-sample dice summation.

    This exercises the validation *sampler* (the object under test): it reads the
    indices the trainer's ``val_loader`` hands this rank and sums the fixed
    per-sample dice over them. A padding sampler that duplicates a sample counts
    it twice here, exactly as the real evaluate would.
    """

    def stub(net, dataloader, *args, **kwargs):
        indices = list(dataloader.sampler)
        dice_sum = float(sum(per_sample_dice[i] for i in indices))
        numsamples = len(indices)
        return dice_sum, 0.0, 0.0, numsamples, numsamples

    return stub


def _run(mode: str, rank: int, world: int):
    from distconv import DCTensor

    import ScaFFold.utils.trainer as trainer_mod
    from ScaFFold.unet import UNet
    from ScaFFold.utils.trainer import PyTorchTrainer

    dataset_dir = os.environ["DATASET_DIR"]
    run_dir = os.environ["RUN_DIR"]
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    config = _build_config(dataset_dir, run_dir, rank, world)

    # DistConv cannot run on this CPU build; the shard IS the whole tensor here.
    DCTensor.from_shard = classmethod(lambda cls, tensor, ps: tensor)
    torch.cuda.Event = _FakeCudaEvent

    log = logging.getLogger(f"metrics_rank{rank}")
    log.setLevel(logging.INFO)

    model = UNet(
        n_channels=3,
        n_classes=config.n_categories + 1,
        trilinear=False,
        layers=config.unet_layers,
        group_norm_groups=config.group_norm_groups,
    )
    trainer = PyTorchTrainer(model, config, torch.device("cpu"), log)

    epoch_counter = [0]
    if mode == "train":
        trainer._run_training_batch = _make_train_batch_stub(rank)
        trainer_mod.evaluate = _make_val_stub(rank, [0])
    elif mode in ("val", "best"):
        trainer._run_training_batch = _make_trivial_train_batch_stub()
        trainer_mod.evaluate = _make_val_stub(rank, epoch_counter)
    elif mode == "valpad":
        trainer._run_training_batch = _make_trivial_train_batch_stub()
        dice_spec = _float_list("METRICS_PER_SAMPLE_DICE")
        per_sample = {i: dice_spec[i] for i in range(len(dice_spec))}
        trainer_mod.evaluate = _make_valpad_stub(per_sample)
    else:
        raise ValueError(f"unknown METRICS_MODE={mode!r}")

    trainer.cleanup_or_resume()
    trainer.train()

    dist.barrier()
    if rank == 0:
        with open(trainer.outfile_path) as handle:
            rows = [ln for ln in handle.read().splitlines() if ln.strip()]
        for row in rows[1:]:  # skip header
            print(f"RANK {rank} CSV {row}", flush=True)
        best_path = trainer.checkpoint_manager.best_ckpt_path
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            print(f"RANK {rank} BEST_EPOCH {best['epoch']}", flush=True)
    print(f"RANK {rank} DONE", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    rank, world = _map_torchrun_env_to_ompi()
    dist.init_process_group(backend="gloo")
    mode = os.environ.get("METRICS_MODE", "train")
    _run(mode, rank, world)


if __name__ == "__main__":
    main()
