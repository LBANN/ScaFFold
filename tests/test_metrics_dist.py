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

"""Distributed epoch-metric reduction tests (2 ranks, gloo, CPU).

These drive a real ``PyTorchTrainer`` across two data-parallel replicas under
``torchrun`` with the gloo backend -- no GPU, no DistConv forward path, and no
MPI launcher. The DistConv-only seams are monkeypatched inside the rank script
(``tests/helpers/rank_scripts/metrics_train_2rank.py``): ``DCTensor.from_shard``
becomes identity, ``torch.cuda.Event`` a CPU timer, and the per-batch training
step / validation ``evaluate`` are stubbed to rank-dependent constants so the
exact values each replica contributes are known.

Each test asserts on rank-tagged markers the rank script prints to stdout. The
reductions under test:

* validation is sharded without padding, so the aggregated val score counts each
  sample once (a padding sampler double-counts duplicated samples);
* the CSV train loss/dice are the global sample-weighted means over all
  replicas, not rank 0's replica-local batch means;
* the logged / checkpoint-selecting validation loss is the global mean.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from tests.helpers import mpi_runner

RANK_SCRIPT = (
    Path(__file__).resolve().parent
    / "helpers"
    / "rank_scripts"
    / "metrics_train_2rank.py"
)

pytestmark = pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    reason="requires torch.distributed with the gloo backend",
)


def _csv_rows(stdout: str):
    """Return the CSV data rows (as column lists) rank 0 emitted."""
    rows = []
    for line in stdout.splitlines():
        m = re.search(r"RANK 0 CSV (.+)$", line)
        if m:
            rows.append(m.group(1).split(","))
    return rows


def _assert_all_done(rc, out, err, n=2):
    """Both ranks reached the DONE marker and the job exited cleanly."""
    done = set(re.findall(r"RANK (\d+) DONE", out))
    assert rc == 0 and {str(r) for r in range(n)} <= done, (
        f"expected clean 2-rank completion, rc={rc}\n"
        f"stdout:\n{out}\nstderr:\n{err[-3000:]}"
    )


# ---------------------------------------------------------------------------
# validation sharding must not pad (duplicated samples would bias the score)
# ---------------------------------------------------------------------------


def test_val_no_padding_bias(tmp_path, tiny_dataset):
    """3 val samples over 2 ranks with per-sample dice {1, 1, 0} -> 2/3.

    A padding validation sampler would hand a duplicate of sample 0 to the rank
    that runs short, so the summed dice/count would be 3/4 instead of the true
    2/3. The unpadded sampler gives every sample to exactly one rank.
    """
    dataset_dir = tiny_dataset(
        root=tmp_path / "ds", n_categories=2, n_train=4, n_val=3, n=16
    )
    rc, out, err = mpi_runner.torchrun_gloo(
        str(RANK_SCRIPT),
        n=2,
        timeout=120,
        env={
            "DATASET_DIR": str(dataset_dir),
            "RUN_DIR": str(tmp_path / "run"),
            "METRICS_MODE": "valpad",
            "METRICS_EPOCHS": "1",
            "METRICS_PER_SAMPLE_DICE": "1.0,1.0,0.0",
        },
    )
    _assert_all_done(rc, out, err)

    rows = _csv_rows(out)
    assert rows, f"no CSV row emitted\nstdout:\n{out}\nstderr:\n{err[-2000:]}"
    # val_dice is column index 6 (epoch, epoch_loss, overall_loss,
    # val_loss_epoch, val_loss_avg, train_dice, val_dice, epoch_duration).
    val_score = float(rows[0][6])
    assert val_score == pytest.approx(2.0 / 3.0, abs=1e-6), (
        f"expected unpadded val_score 2/3, got {val_score}\nstdout:\n{out}"
    )


# ---------------------------------------------------------------------------
# train loss/dice must be the global sample-weighted mean over replicas
# ---------------------------------------------------------------------------


def test_train_metrics_globally_reduced(tmp_path, tiny_dataset):
    """rank 0 -> constant 1.0, rank 1 -> constant 3.0 => CSV mean 2.0.

    Each replica's per-batch loss/dice is fixed, so the global sample-weighted
    mean is (1.0 + 3.0) / 2 = 2.0. Rank 0's replica-local value alone is 1.0.
    """
    dataset_dir = tiny_dataset(
        root=tmp_path / "ds", n_categories=2, n_train=4, n_val=2, n=16
    )
    rc, out, err = mpi_runner.torchrun_gloo(
        str(RANK_SCRIPT),
        n=2,
        timeout=120,
        env={
            "DATASET_DIR": str(dataset_dir),
            "RUN_DIR": str(tmp_path / "run"),
            "METRICS_MODE": "train",
            "METRICS_EPOCHS": "1",
            "METRICS_TRAIN_LOSS": "1.0,3.0",
            "METRICS_TRAIN_DICE": "1.0,3.0",
        },
    )
    _assert_all_done(rc, out, err)

    rows = _csv_rows(out)
    assert rows, f"no CSV row emitted\nstdout:\n{out}\nstderr:\n{err[-2000:]}"
    overall_loss = float(rows[0][2])
    train_dice = float(rows[0][5])
    assert overall_loss == pytest.approx(2.0, abs=1e-6), (
        f"expected globally reduced train loss 2.0, got {overall_loss}\nstdout:\n{out}"
    )
    assert train_dice == pytest.approx(2.0, abs=1e-6), (
        f"expected globally reduced train dice 2.0, got {train_dice}\nstdout:\n{out}"
    )


# ---------------------------------------------------------------------------
# logged validation loss must be the global sample-weighted mean
# ---------------------------------------------------------------------------


def test_val_loss_globally_reduced(tmp_path, tiny_dataset):
    """rank losses (0.1, 0.9) -> logged val_loss_avg is the global mean 0.5.

    Each replica reports one validation sample, so the global sample-weighted
    mean is (0.1 + 0.9) / 2 = 0.5. Rank 0's replica-local value alone is 0.1.
    """
    dataset_dir = tiny_dataset(
        root=tmp_path / "ds", n_categories=2, n_train=4, n_val=2, n=16
    )
    rc, out, err = mpi_runner.torchrun_gloo(
        str(RANK_SCRIPT),
        n=2,
        timeout=120,
        env={
            "DATASET_DIR": str(dataset_dir),
            "RUN_DIR": str(tmp_path / "run"),
            "METRICS_MODE": "val",
            "METRICS_EPOCHS": "1",
            "METRICS_VAL_LOSS": "0.1,0.9",
        },
    )
    _assert_all_done(rc, out, err)

    rows = _csv_rows(out)
    assert rows, f"no CSV row emitted\nstdout:\n{out}\nstderr:\n{err[-2000:]}"
    val_loss_avg = float(rows[0][4])
    assert val_loss_avg == pytest.approx(0.5, abs=1e-6), (
        f"expected globally reduced val_loss_avg 0.5, got {val_loss_avg}\n"
        f"stdout:\n{out}"
    )


# ---------------------------------------------------------------------------
# best-checkpoint selection must use the global validation loss
# ---------------------------------------------------------------------------


def test_best_checkpoint_uses_global_val(tmp_path, tiny_dataset):
    """epoch A rank losses (0.1, 0.9); epoch B (0.4, 0.4) -> best = epoch B.

    Global means: epoch 1 -> 0.5, epoch 2 -> 0.4, so epoch 2 is globally best.
    Rank 0's replica-local view (0.1 vs 0.4) would wrongly keep epoch 1.
    """
    dataset_dir = tiny_dataset(
        root=tmp_path / "ds", n_categories=2, n_train=4, n_val=2, n=16
    )
    rc, out, err = mpi_runner.torchrun_gloo(
        str(RANK_SCRIPT),
        n=2,
        timeout=120,
        env={
            "DATASET_DIR": str(dataset_dir),
            "RUN_DIR": str(tmp_path / "run"),
            "METRICS_MODE": "best",
            "METRICS_EPOCHS": "2",
            "METRICS_CKPT_INTERVAL": "1",
            "METRICS_VAL_LOSS": "0.1,0.9;0.4,0.4",
        },
    )
    _assert_all_done(rc, out, err)

    m = re.search(r"RANK 0 BEST_EPOCH (\d+)", out)
    assert m, f"no BEST_EPOCH marker emitted\nstdout:\n{out}\nstderr:\n{err[-2000:]}"
    best_epoch = int(m.group(1))
    assert best_epoch == 2, (
        f"expected globally-best epoch 2, got {best_epoch}\nstdout:\n{out}"
    )
