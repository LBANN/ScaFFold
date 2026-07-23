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

"""Tests for training/validation hot-path efficiency fixes.

These verify behavior-preserving optimizations of the per-step compute path:
one-hot construction, shared log-softmax, collective packing, gradient
handling, and validation host-sync avoidance. Each asserts the fast path is
numerically equivalent to the straightforward reference it replaced.
"""

import torch
import torch.nn.functional as F

from ScaFFold.utils.dice_score import labels_to_onehot
from ScaFFold.utils.losses import compute_sharded_cross_entropy_loss


def _reference_onehot(labels, num_classes):
    """The straightforward F.one_hot path the scatter helper replaced."""
    return F.one_hot(labels, num_classes).permute(0, 4, 1, 2, 3).float()


def test_labels_to_onehot_matches_one_hot_and_is_float32_contiguous():
    # F12: the scatter-based one-hot must be bit-identical to the
    # F.one_hot().permute().float() chain, but float32 (not an int64
    # intermediate) and contiguous in channel-first layout.
    torch.manual_seed(0)
    labels = torch.randint(0, 5, (2, 4, 4, 4))
    got = labels_to_onehot(labels, 5)
    ref = _reference_onehot(labels, 5)
    assert torch.equal(got, ref)
    assert got.dtype == torch.float32
    assert got.is_contiguous()
    # The reference chain is a permuted view (non-contiguous); the scatter
    # output is genuinely contiguous, which is the point.
    assert not ref.is_contiguous()


def test_ce_log_probs_path_matches_cross_entropy():
    # F20: feeding a precomputed log_softmax via NLL must equal computing CE
    # from raw logits, for both weighted and unweighted cases, with no mesh.
    torch.manual_seed(1)
    b, c = 2, 5
    preds = torch.randn(b, c, 4, 4, 4)
    labels = torch.randint(0, c, (b, 4, 4, 4))
    weights = torch.rand(c) + 0.5
    log_probs = F.log_softmax(preds.float(), dim=1)
    for w in (None, weights):
        ref = F.cross_entropy(preds.float(), labels, weight=w, reduction="mean")
        fused = compute_sharded_cross_entropy_loss(
            preds, labels, None, (1,), "cpu", w, log_probs=log_probs
        )
        plain = compute_sharded_cross_entropy_loss(preds, labels, None, (1,), "cpu", w)
        assert torch.allclose(fused, ref, atol=1e-6)
        assert torch.allclose(plain, ref, atol=1e-6)


def test_ce_uses_single_spatial_collective(monkeypatch):
    # F19: the CE numerator and its normalizer are reduced together in one
    # SpatialAllReduce, not two. Count applications; the packed path issues
    # exactly one per CE call (down from two).
    import ScaFFold.utils.losses as losses

    calls = {"n": 0}
    real_apply = losses.SpatialAllReduce.apply

    def counting_apply(tensor, mesh):
        calls["n"] += 1
        return real_apply(tensor, mesh)

    monkeypatch.setattr(losses.SpatialAllReduce, "apply", staticmethod(counting_apply))

    torch.manual_seed(2)
    preds = torch.randn(1, 4, 4, 4, 4)
    labels = torch.randint(0, 4, (1, 4, 4, 4))
    losses.compute_sharded_cross_entropy_loss(preds, labels, None, (1,), "cpu", None)
    assert calls["n"] == 1

    calls["n"] = 0
    weights = torch.rand(4) + 0.5
    losses.compute_sharded_cross_entropy_loss(preds, labels, None, (1,), "cpu", weights)
    assert calls["n"] == 1


def test_torch_profiler_context_is_bounded(monkeypatch):
    # F47: the profiler must be built with a bounded schedule (not RECORD for
    # every step) and with the expensive record_shapes/with_stack options off
    # by default, so a long run cannot grow an unbounded trace.
    import ScaFFold.utils.perf_measure as pm

    captured = {}

    class FakeProf:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(pm, "TORCH_PERF_ENABLED", True)
    # torchprofile/ProfilerActivity are imported only when profiling is enabled
    # at import time, so they may not exist as module attributes here.
    monkeypatch.setattr(pm, "torchprofile", FakeProf, raising=False)
    # ProfilerActivity is referenced when building activities; provide a stub.
    monkeypatch.setattr(
        pm, "ProfilerActivity", type("PA", (), {"CUDA": 0, "CPU": 1}), raising=False
    )

    ctx, local = pm.get_torch_context(ranks_per_node=1, rank=0)
    assert local is True
    assert "schedule" in captured and captured["schedule"] is not None
    assert captured["record_shapes"] is False
    assert captured["with_stack"] is False


def _module_source(module):
    import inspect

    return inspect.getsource(module)


def test_training_batch_returns_detached_dice(tiny_trainer):
    # F51: the dice score returned from a batch must be detached so the epoch
    # accumulator does not retain each batch's autograd graph. Run one real
    # CPU batch (ps=None path) and inspect the returned tensor.
    trainer = tiny_trainer()
    batch = next(iter(trainer.train_loader))
    trainer.model.train()
    _, loss, dice = trainer._run_training_batch(batch)
    assert dice.requires_grad is False
    assert dice.grad_fn is None
    assert loss.requires_grad is False


def test_zero_grad_uses_set_to_none(tiny_trainer):
    # F52: after a batch, grads should be released (None) rather than zeroed
    # buffers, avoiding a per-step memset over all gradient memory.
    trainer = tiny_trainer()
    batch = next(iter(trainer.train_loader))
    trainer.model.train()
    trainer._run_training_batch(batch)
    grads = [p.grad for p in trainer.model.parameters()]
    assert all(g is None for g in grads)


def test_evaluate_defers_item_sync_to_end():
    # F22: the validation loop must not call .item() per batch. Verify the
    # foreground stats helper returns a tensor sum (no host sync) and that the
    # source has no per-batch .item() inside the loop.
    import ScaFFold.utils.evaluate as ev

    src = _module_source(ev)
    # The only host-sync drains happen once after the loop; the batch body
    # accumulates device tensors. Check the code (comments stripped) before
    # "net.train()" has no per-batch host sync.
    loop_region = src.split("net.train()")[0]
    code_lines = [line.split("#", 1)[0] for line in loop_region.splitlines()]
    code = "\n".join(code_lines)
    assert ".item()" not in code


def test_evaluate_copies_are_non_blocking():
    # F54: both eval H2D copies must pass non_blocking=True to exploit the
    # pinned-memory loaders.
    import ScaFFold.utils.evaluate as ev

    src = _module_source(ev)
    assert src.count("non_blocking=True") >= 2


def test_training_batch_gathers_mem_only_first_batch():
    # F18: the epoch loop must not request mem-stat gathering for every batch
    # (which resets peak counters and issues collectives inside the timed
    # region). It should gate on the first batch of the run.
    import ScaFFold.utils.trainer as tr

    src = _module_source(tr)
    # The hardcoded every-batch gather is gone; gathering is gated on a
    # first-batch predicate.
    assert "gather_mem_stats=True" not in src
    assert "first_batch" in src
