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

"""Single-process validation-metric tests for ``ScaFFold.utils.evaluate``.

``evaluate`` computes its metrics through ``compute_sharded_dice`` /
``compute_sharded_cross_entropy_loss``, both of which reduce across a spatial
device mesh with ``dist.all_reduce``. To exercise that real code path on CPU
without a launcher we initialise a one-rank gloo process group for the module.

At ``world_size == 1`` a rank's local shard *is* the full global tensor, so we
replace ``DCTensor.from_shard`` with an identity pass-through: this feeds plain
CPU tensors through ``evaluate``'s exact arithmetic (the ``DCTensor`` wrapper
itself is unusable on this CPU-only build) while the ``all_reduce`` collectives
still run for real over the one-rank group.

The stub "networks" return logits built directly from label volumes, so the
argmax prediction and its confidence margin are both controlled exactly.
"""

from __future__ import annotations

import socket

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# one-rank gloo process group + identity DCTensor pass-through
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def eval_env():
    """One-rank gloo group + real ``ParallelStrategy``; identity ``from_shard``.

    Yields a ``(parallel_strategy, evaluate)`` pair. Tears the group down and
    restores ``DCTensor.from_shard`` afterwards.
    """
    from distconv import DCTensor, ParallelStrategy

    created_group = False
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{_free_port()}?rank=0&world_size=1",
        )
        created_group = True

    original_from_shard = DCTensor.from_shard
    # world_size == 1: the local shard is the whole tensor, so pass it through.
    DCTensor.from_shard = classmethod(lambda cls, tensor, ps: tensor)

    from ScaFFold.utils.evaluate import evaluate

    parallel_strategy = ParallelStrategy(
        num_shards=(1,), shard_dim=(2,), device_type="cpu"
    )

    try:
        yield parallel_strategy, evaluate
    finally:
        DCTensor.from_shard = original_from_shard
        if created_group and dist.is_initialized():
            dist.destroy_process_group()


DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# helpers: build stub batches and an independent Dice oracle
# ---------------------------------------------------------------------------


class _ScaledLogits(nn.Module):
    """Return the (one-hot) input scaled by a fixed margin.

    The input volume is a one-hot encoding of the desired argmax labels, so the
    prediction is fixed and its confidence is exactly ``margin``.
    """

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, x):
        return x * self.margin


def _one_hot_logits(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """[B, D, H, W] labels -> [B, C, D, H, W] float one-hot volume."""
    return F.one_hot(labels, n_classes).permute(0, 4, 1, 2, 3).float()


def _make_batch(pred_labels, true_labels, n_classes):
    """Pack per-sample (pred, true) label volumes into one evaluate batch dict.

    ``image`` is the one-hot encoding of the *predicted* labels (the stub net
    turns it into logits); ``mask`` holds the ground-truth labels.
    """
    images = torch.stack(
        [_one_hot_logits(p.unsqueeze(0), n_classes)[0] for p in pred_labels]
    )
    masks = torch.stack(list(true_labels))
    return {"image": images, "mask": masks}


def _foreground_hard_dice(pred_labels, true_labels, n_classes, eps=1e-6):
    """Independent oracle: mean foreground hard Dice for one sample.

    Mirrors ``compute_sharded_dice`` + ``foreground_dice_stats`` at
    ``world_size == 1``, including the empty/empty -> 1 guard, but computed
    directly from label volumes rather than through the code under test.
    """
    dices = []
    for c in range(1, n_classes):
        pred_c = (pred_labels == c).float()
        true_c = (true_labels == c).float()
        inter = 2.0 * (pred_c * true_c).sum()
        sets_sum = pred_c.sum() + true_c.sum()
        if sets_sum.item() == 0:
            sets_sum = inter
        dices.append(((inter + eps) / (sets_sum + eps)).item())
    return sum(dices) / len(dices)


def _val_score(evaluate_out):
    """Reproduce trainer.py's dice aggregation: total_dice_score / samples."""
    total_dice_score, _, _, _, processed_samples = evaluate_out
    return total_dice_score / max(processed_samples, 1)


def _run(evaluate, ps, net, batches, n_categories):
    return evaluate(
        net,
        batches,
        DEVICE,
        False,  # amp
        False,  # primary (silence prints)
        nn.CrossEntropyLoss(),
        n_categories,
        ps,
    )


# ---------------------------------------------------------------------------
# Reported Dice must be the hard (argmax) segmentation Dice
# ---------------------------------------------------------------------------


def test_perfect_model_dice_is_one(eval_env):
    """A perfect, confident model scores exactly 1.0 even when a validation
    sample lacks a foreground class."""
    ps, evaluate = eval_env
    n_categories, n_classes = 2, 3

    # Four samples with every class present, plus one sample missing class 2.
    # With soft-probability Dice the missing-class sample is scored ~0 for that
    # class, capping a perfect model well below 1.0; hard Dice's empty/empty
    # guard restores the correct 1.0.
    torch.manual_seed(0)
    samples = [torch.randint(0, n_classes, (4, 4, 4)) for _ in range(4)]
    samples.append(torch.randint(0, 2, (4, 4, 4)))  # class 2 absent
    assert not (samples[-1] == 2).any()

    batches = [_make_batch([lbl], [lbl], n_classes) for lbl in samples]
    net = _ScaledLogits(margin=10.0)  # perfect argmax, high confidence

    out = _run(evaluate, ps, net, batches, n_categories)
    assert _val_score(out) == pytest.approx(1.0, abs=1e-6)


def test_dice_monotone_in_confidence(eval_env):
    """Identical argmax predictions score identically regardless of logit
    margin (the reported Dice must not depend on confidence)."""
    ps, evaluate = eval_env
    n_categories, n_classes = 2, 3

    true_labels = torch.zeros(4, 4, 4, dtype=torch.long)
    true_labels[0] = 1
    true_labels[1] = 2
    pred_labels = true_labels.clone()
    pred_labels[0, 0, 0] = 0  # a few deliberately-wrong voxels
    pred_labels[1, 0, 0] = 0

    batch = _make_batch([pred_labels], [true_labels], n_classes)
    expected = _foreground_hard_dice(pred_labels, true_labels, n_classes)

    score_low = _val_score(
        _run(evaluate, ps, _ScaledLogits(2.0), [batch], n_categories)
    )
    score_high = _val_score(
        _run(evaluate, ps, _ScaledLogits(20.0), [batch], n_categories)
    )

    assert score_low == pytest.approx(score_high, abs=1e-6)
    assert score_low == pytest.approx(expected, abs=1e-5)


def test_imperfect_model_dice_below_one(eval_env):
    """A model with ~10% wrong foreground voxels scores the analytic hard
    Dice for that error rate."""
    ps, evaluate = eval_env
    n_categories, n_classes = 1, 2

    # 64 voxels, 32 foreground; flip 3 foreground voxels to background (~9.4%).
    flat_true = torch.zeros(64, dtype=torch.long)
    flat_true[:32] = 1
    true_labels = flat_true.view(4, 4, 4)
    flat_pred = flat_true.clone()
    flat_pred[:3] = 0
    pred_labels = flat_pred.view(4, 4, 4)

    batch = _make_batch([pred_labels], [true_labels], n_classes)
    expected = _foreground_hard_dice(pred_labels, true_labels, n_classes)
    assert expected < 1.0  # sanity: this is an imperfect model

    out = _run(evaluate, ps, _ScaledLogits(2.0), [batch], n_categories)
    assert _val_score(out) == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# The degenerate single-class path must be rejected, not faked
# ---------------------------------------------------------------------------


def test_n_categories_zero_rejected(eval_env):
    """``n_categories=0`` (a single output channel) has no meaningful
    multiclass validation metric and must raise, not report Dice=1/loss=0."""
    ps, evaluate = eval_env

    with pytest.raises(ValueError, match="n_categories"):
        _run(evaluate, ps, _ScaledLogits(1.0), [], 0)


# ---------------------------------------------------------------------------
# val_loss_avg must be sample-weighted, not equal-weighted per batch
# ---------------------------------------------------------------------------


def test_val_loss_sample_weighted(eval_env):
    """With a ragged final batch, val_loss_avg equals the sample-weighted mean
    of the per-sample losses, independent of batch grouping."""
    ps, evaluate = eval_env
    n_categories, n_classes = 1, 2

    # Three samples with clearly different per-sample losses: correct+confident,
    # correct+unsure, and wrong. Each sample's confidence margin is baked into
    # its one-hot logits so a single identity net reproduces every grouping.
    def spec(fg_correct: bool, margin: float):
        true_labels = torch.zeros(64, dtype=torch.long)
        true_labels[:32] = 1
        true_labels = true_labels.view(4, 4, 4)
        pred_labels = true_labels if fg_correct else 1 - true_labels
        return pred_labels, true_labels, margin

    specs = [spec(True, 6.0), spec(True, 3.0), spec(False, 2.0)]

    def scaled_batch(indices):
        images = torch.stack(
            [
                _one_hot_logits(specs[i][0].unsqueeze(0), n_classes)[0] * specs[i][2]
                for i in indices
            ]
        )
        masks = torch.stack([specs[i][1] for i in indices])
        return {"image": images, "mask": masks}

    identity = _ScaledLogits(1.0)

    # A single-sample batch's val_loss_avg is exactly that sample's loss.
    per_sample_losses = [
        _run(evaluate, ps, identity, [scaled_batch([i])], n_categories)[2]
        for i in range(len(specs))
    ]
    sample_weighted_mean = sum(per_sample_losses) / len(per_sample_losses)

    ragged = [scaled_batch([0, 1]), scaled_batch([2])]  # batches of 2 + 1
    equal = [scaled_batch([0]), scaled_batch([1]), scaled_batch([2])]
    single = [scaled_batch([0, 1, 2])]

    avg_ragged = _run(evaluate, ps, identity, ragged, n_categories)[2]
    avg_equal = _run(evaluate, ps, identity, equal, n_categories)[2]
    avg_single = _run(evaluate, ps, identity, single, n_categories)[2]

    # Batching must not change the average, and it must be the sample-weighted
    # mean (the lone final sample is not overweighted).
    assert avg_ragged == pytest.approx(sample_weighted_mean, abs=1e-5)
    assert avg_ragged == pytest.approx(avg_equal, abs=1e-5)
    assert avg_ragged == pytest.approx(avg_single, abs=1e-5)
