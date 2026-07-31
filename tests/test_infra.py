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

"""Smoke tests proving the shared fixtures work against the current code.

These are Batch 0's own tests: they exercise the fixtures end-to-end so later
batches can rely on them. They must pass against the *current* (partially
buggy) code, so they deliberately steer around known product bugs (documented
inline).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from ScaFFold.utils.data_loading import FractalDataset
from ScaFFold.utils.data_types import MASK_DTYPE, VOLUME_DTYPE
from ScaFFold.utils.utils import gather_and_print_mem, mem_stats
from tests.helpers import mpi_runner

# ---------------------------------------------------------------------------
# tiny_config
# ---------------------------------------------------------------------------


def test_tiny_config_builds(tiny_config):
    """The factory yields a fully-populated RunConfig at the smallest scale."""
    config = tiny_config()

    # RunConfig-specific attributes.
    assert config.run_iter == 0
    assert os.path.isdir(config.run_dir)

    # Shrunk problem size: 16^3 volumes, exactly one U-Net layer.
    assert config.problem_scale == 4
    assert config.unet_bottleneck_dim == 3
    assert config.unet_layers == 1
    assert config.n_categories == 2
    assert config.local_batch_size == 1

    # Extra attributes the trainer/worker read that Config.__init__ omits.
    assert config._parallel_strategy is None
    assert config.verbose == 0
    assert config.vol_size == 16


def test_tiny_config_overrides(tiny_config):
    """Overrides flow through to the constructed config."""
    config = tiny_config(n_categories=3, local_batch_size=2, target_dice=0.0)
    assert config.n_categories == 3
    assert config.local_batch_size == 2
    assert config.target_dice == 0.0


# ---------------------------------------------------------------------------
# datasets: v2 and v1 load through FractalDataset
# ---------------------------------------------------------------------------


def test_tiny_dataset_v2_loads(tiny_dataset):
    """A v2 dataset loads one item with the expected shapes/dtypes."""
    root = tiny_dataset(n_categories=2, n_train=4, n_val=2, n=16)

    assert (root / "meta.yaml").exists()
    train = FractalDataset(
        root / "volumes" / "training",
        root / "masks" / "training",
        data_dir=root / "train_unique_mask_vals",
    )
    assert len(train) == 4
    assert train.dataset_format_version == 2

    item = train[0]
    # Channels-first volume, float; single-channel narrow-int mask carrier
    # (widened to long on the compute device by the trainer/evaluator).
    assert tuple(item["image"].shape) == (3, 16, 16, 16)
    assert item["image"].dtype == torch.float32
    assert tuple(item["mask"].shape) == (16, 16, 16)
    assert item["mask"].dtype == torch.int16
    assert int(item["mask"].max()) <= 2

    # The raw on-disk arrays match the format the loader/trainer expect.
    raw_vol = np.load(root / "volumes" / "training" / "0.npy")
    raw_mask = np.load(root / "masks" / "training" / "0_mask.npy")
    assert raw_vol.shape == (3, 16, 16, 16)
    assert raw_vol.dtype == VOLUME_DTYPE
    assert raw_mask.shape == (16, 16, 16)
    assert raw_mask.dtype == MASK_DTYPE


def test_tiny_v1_dataset_loads_and_remaps(tiny_v1_dataset):
    """A v1 dataset loads via the legacy path: channels-last + value remap."""
    root = tiny_v1_dataset(n_categories=2, n_train=4, n_val=2, n=16)

    # v1 has no meta.yaml -> loader falls back to the legacy format version.
    assert not (root / "meta.yaml").exists()

    train = FractalDataset(
        root / "volumes" / "training",
        root / "masks" / "training",
        data_dir=root / "train_unique_mask_vals",
    )
    assert train.dataset_format_version == 1

    # On disk: channels-last volume, raw (non-contiguous) mask values.
    raw_vol = np.load(root / "volumes" / "training" / "0.npy")
    raw_mask = np.load(root / "masks" / "training" / "0_mask.npy")
    assert raw_vol.shape == (16, 16, 16, 3)
    assert set(np.unique(raw_mask).tolist()).issubset({0, 10, 20})

    item = train[0]
    # Legacy loader transposes to channels-first and remaps values to 0..n.
    assert tuple(item["image"].shape) == (3, 16, 16, 16)
    assert item["image"].dtype == torch.float32
    assert tuple(item["mask"].shape) == (16, 16, 16)
    assert set(item["mask"].reshape(-1).tolist()).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# tiny_trainer: construct + cleanup_or_resume + train() control flow
# ---------------------------------------------------------------------------


def test_tiny_trainer_constructs(tiny_trainer):
    """A real PyTorchTrainer builds on CPU with a 1-layer U-Net and ps=None."""
    trainer = tiny_trainer()
    assert trainer.n_train == 4
    assert trainer.n_val == 2
    assert trainer.ps is None
    # CE weights were estimated at construction (n_categories+1 = 3 classes).
    assert trainer.ce_class_weights is not None
    assert trainer.ce_class_weights.numel() == 3


def test_tiny_trainer_cleanup_and_train(tiny_trainer):
    """cleanup_or_resume + train() run cleanly and write the stats header.

    ``target_dice=0.0`` makes ``train()``'s ``while dice < target_dice`` guard
    false on entry, so training terminates before the first batch. This
    exercises the full setup / teardown control flow (and CSV bootstrap)
    WITHOUT hitting the DistConv forward path, which cannot run with ``ps=None``
    (``DCTensor.from_shard(x, None)`` dereferences ``None.shard_dim`` and
    segfaults -- a known DistConv limitation).
    """
    trainer = tiny_trainer(config_overrides={"target_dice": 0.0, "epochs": 1})

    trainer.cleanup_or_resume()
    assert trainer.start_epoch == 1

    # Stats file bootstrapped with just the header on a from-scratch run.
    assert os.path.exists(trainer.outfile_path)
    with open(trainer.outfile_path) as handle:
        header = handle.readline().strip().split(",")
    assert header[0] == "epoch"
    assert "val_dice" in header

    # Runs to completion (0 epochs) without raising / segfaulting.
    trainer.train()

    # No epoch rows were appended (guard was false on entry).
    with open(trainer.outfile_path) as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    assert len(lines) == 1  # header only


# ---------------------------------------------------------------------------
# fresh_python
# ---------------------------------------------------------------------------


def test_fresh_python_runs_snippet(fresh_python):
    """fresh_python runs code in a subprocess with the repo importable."""
    out = fresh_python(
        "import ScaFFold.utils.data_types as dt; print(dt.VOLUME_DTYPE_NAME)"
    )
    assert out.strip() == "float32"


# ---------------------------------------------------------------------------
# mpi_runner helpers
# ---------------------------------------------------------------------------


def test_mpi_run_skips_without_launcher():
    """In an environment without an MPI launcher, mpi_run raises pytest.skip.

    We assert the *behavior* the plan requires: when no launcher is on PATH,
    ``mpi_run`` skips with a clear message rather than failing. When a launcher
    IS present, we don't force a skip -- we just confirm detection is truthy.
    """
    launcher = mpi_runner.detect_mpi_launcher()
    if launcher is None:
        import pytest

        with pytest.raises(pytest.skip.Exception, match="no MPI launcher available"):
            mpi_runner.mpi_run("/nonexistent/script.py", n=2, timeout=5)
    else:
        assert isinstance(launcher, str)


def test_torchrun_gloo_two_ranks(tmp_path):
    """torchrun_gloo launches a 2-rank gloo job and captures per-rank output."""
    script = tmp_path / "rank_allreduce.py"
    script.write_text(
        "import torch\n"
        "torch.distributed.init_process_group(backend='gloo')\n"
        "rank = torch.distributed.get_rank()\n"
        "ws = torch.distributed.get_world_size()\n"
        "t = torch.tensor([float(rank)])\n"
        "torch.distributed.all_reduce(t)\n"
        "print(f'RANK {rank}/{ws} sum={t.item()}', flush=True)\n"
        "torch.distributed.barrier()\n"
        "torch.distributed.destroy_process_group()\n"
    )
    rc, out, err = mpi_runner.torchrun_gloo(str(script), n=2, timeout=90)
    assert rc == 0, f"torchrun failed rc={rc}\nstdout:\n{out}\nstderr:\n{err}"
    # Both ranks reported; all_reduce of ranks {0,1} sums to 1.
    assert "RANK 0/2 sum=1.0" in out
    assert "RANK 1/2 sum=1.0" in out


# ---------------------------------------------------------------------------
# R21: memory diagnostics on a CPU-only run
# ---------------------------------------------------------------------------


def _debug_logger(name):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    return log


def test_mem_stats_without_cuda(caplog):
    """``mem_stats`` reports "no GPU" instead of raising on a CPU-only host."""
    if torch.cuda.is_available():
        import pytest

        pytest.skip("test covers the CPU-only path")

    stats = mem_stats()

    assert stats["cuda_available"] is False
    assert "rank" in stats


def test_gather_and_print_mem_without_cuda(caplog):
    """A DEBUG-level CPU run logs a fallback instead of crashing.

    ``BaseTrainer.__init__`` calls this unconditionally, so a CPU/gloo run with
    ``-v`` used to die in trainer construction with "No CUDA GPUs are
    available".
    """
    if torch.cuda.is_available():
        import pytest

        pytest.skip("test covers the CPU-only path")

    log = _debug_logger("test_gather_and_print_mem_without_cuda")
    with caplog.at_level(logging.DEBUG, logger=log.name):
        gather_and_print_mem(log, "after_trainer_setup")

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "after_trainer_setup" in messages
    assert "cuda" in messages.lower() or "gpu" in messages.lower()


def test_trainer_constructs_with_debug_logging(tiny_trainer):
    """The real call site survives: a trainer builds with a DEBUG logger."""
    trainer = tiny_trainer(log_level=logging.DEBUG)

    assert trainer.log.getEffectiveLevel() == logging.DEBUG
