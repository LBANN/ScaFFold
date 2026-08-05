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
    # The scatter-based one-hot must be bit-identical to the
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
    # Feeding a precomputed log_softmax via NLL must equal computing CE
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
    # The CE numerator and its normalizer are reduced together in one
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
    # The profiler must be built with a bounded schedule (not RECORD for
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
    # The dice score returned from a batch must be detached so the epoch
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
    # After a batch, grads should be released (None) rather than zeroed
    # buffers, avoiding a per-step memset over all gradient memory.
    trainer = tiny_trainer()
    batch = next(iter(trainer.train_loader))
    trainer.model.train()
    trainer._run_training_batch(batch)
    grads = [p.grad for p in trainer.model.parameters()]
    assert all(g is None for g in grads)


def test_evaluate_defers_item_sync_to_end():
    # The validation loop must not call .item() per batch. Verify the
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
    # Both eval H2D copies must pass non_blocking=True to exploit the
    # pinned-memory loaders.
    import ScaFFold.utils.evaluate as ev

    src = _module_source(ev)
    assert src.count("non_blocking=True") >= 2


def test_training_batch_gathers_mem_only_first_batch():
    # The epoch loop must not request mem-stat gathering for every batch
    # (which resets peak counters and issues collectives inside the timed
    # region). It should gate on the first batch of the run.
    import ScaFFold.utils.trainer as tr

    src = _module_source(tr)
    # The hardcoded every-batch gather is gone; gathering is gated on a
    # first-batch predicate.
    assert "gather_mem_stats=True" not in src
    assert "first_batch" in src


# ---------------------------------------------------------------------------
# Warmup covers the ragged final batch
#
# Warmup only ever runs the *leading* batches of the train loader, which are
# all local_batch_size wide, and neither loader drops its last batch. When a
# rank's sample count is not a multiple of the batch size, the narrower final
# batch is therefore a set of convolution shapes nothing has warmed, and with
# cudnn.benchmark on the algorithm search for it runs inside the first *timed*
# epoch (measured: 95 s) -- straight into epoch_duration, the FOM denominator.
# ---------------------------------------------------------------------------


def _stub_warmup_steps(trainer, monkeypatch, *, mutate=False):
    """Record the batch sizes warmup runs; optionally mutate model state.

    The real training step is hardwired through DistConv and cannot run on the
    CPU ``ps=None`` fixture, so the step itself is stubbed; what is under test
    is which batches warmup feeds it.
    """
    import ScaFFold.utils.trainer as tr

    sizes = []

    def fake_training_batch(batch, **kwargs):
        sizes.append(int(batch["image"].shape[0]))
        if mutate:
            with torch.no_grad():
                for param in trainer.model.parameters():
                    param.add_(1.0)
        return int(batch["image"].shape[0]), torch.tensor(0.0), torch.tensor(0.0)

    monkeypatch.setattr(trainer, "_run_training_batch", fake_training_batch)
    monkeypatch.setattr(tr, "evaluate", lambda *args, **kwargs: (0.0, 0.0))
    return sizes


def test_warmup_covers_the_ragged_train_batch(tiny_trainer, monkeypatch):
    # 5 local training samples at local_batch_size 2: every epoch ends with a
    # 1-sample batch that the leading warmup batches never present.
    trainer = tiny_trainer(
        n_train=5,
        n_val=4,
        config_overrides={"local_batch_size": 2, "warmup_batches": 2},
    )
    sizes = _stub_warmup_steps(trainer, monkeypatch)

    trainer.warmup()

    assert sizes == [2, 2, 1]


def test_warmup_adds_no_extra_batch_when_shards_divide_evenly(
    tiny_trainer, monkeypatch
):
    # local_batch_size 1 can never produce a ragged batch: no extra work.
    trainer = tiny_trainer(
        n_train=4,
        n_val=2,
        config_overrides={"local_batch_size": 1, "warmup_batches": 2},
    )
    sizes = _stub_warmup_steps(trainer, monkeypatch)

    trainer.warmup()

    assert sizes == [1, 1]


def test_warmup_covers_a_ragged_validation_batch(tiny_trainer, monkeypatch):
    # Training divides evenly (4 / 2) but validation does not (3 / 2), so the
    # 1-sample shape still has to be warmed.
    trainer = tiny_trainer(
        n_train=4,
        n_val=3,
        config_overrides={"local_batch_size": 2, "warmup_batches": 2},
    )
    sizes = _stub_warmup_steps(trainer, monkeypatch)

    trainer.warmup()

    assert sizes == [2, 2, 1]


def test_warmup_ragged_sizes_are_agreed_across_ranks(tiny_trainer, monkeypatch):
    # Validation is sharded unpadded, so peers can end their epoch with a
    # different partial size. Every rank must run the same extra steps or the
    # collectives inside them diverge; the peer's remainder (2) is gathered and
    # warmed here even though this rank's own is 1.
    import torch.distributed as dist

    trainer = tiny_trainer(
        n_train=6,
        n_val=4,
        config_overrides={"local_batch_size": 3, "warmup_batches": 2},
    )
    sizes = _stub_warmup_steps(trainer, monkeypatch)
    trainer.world_size = 2

    def fake_all_gather(tensor_list, tensor, *args, **kwargs):
        tensor_list[0].copy_(tensor)
        tensor_list[1].fill_(2)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    trainer.warmup()

    assert sizes == [3, 3, 1, 2]


def test_warmup_ragged_all_gather_is_posted_even_without_a_batch(
    tiny_trainer, monkeypatch
):
    # The ragged-size agreement is a collective: a rank whose warmup fetched
    # no batch at all (empty train loader) must still post the all_gather, or
    # its peers block in it. The no-batch early return has to come after the
    # gather, even though such a rank then runs no extra step itself.
    import torch.distributed as dist

    trainer = tiny_trainer(
        n_train=4,
        n_val=3,
        config_overrides={"local_batch_size": 2, "warmup_batches": 2},
    )
    sizes = _stub_warmup_steps(trainer, monkeypatch)
    gathers = []

    def fake_all_gather(tensor_list, tensor, *args, **kwargs):
        gathers.append(int(tensor.item()))
        for out in tensor_list:
            out.copy_(tensor)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    trainer._warmup_ragged_batches(None)

    assert gathers, "rank skipped the ragged-size all_gather when batch was None"
    assert sizes == []


def test_warmup_rolls_back_state_including_the_ragged_batch(tiny_trainer, monkeypatch):
    # The extra ragged iteration stays inside warmup's snapshot/restore
    # envelope, so nothing it touches survives into training.
    trainer = tiny_trainer(
        n_train=5,
        n_val=4,
        config_overrides={"local_batch_size": 2, "warmup_batches": 2},
    )
    sizes = _stub_warmup_steps(trainer, monkeypatch, mutate=True)
    before = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}

    trainer.warmup()

    assert sizes == [2, 2, 1]
    after = trainer.model.state_dict()
    for name, tensor in before.items():
        assert torch.equal(after[name], tensor), name
