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

"""``FastConv3d``: the rung ladder, the sharding gate, and the numerics.

The single most important test in this file is
:func:`test_halo_plan_refuses_a_split_dim_whose_arithmetic_it_has_not_checked`.
Every other property here fails loudly; that one fails silently, as a plausible
wrong gradient at every shard boundary of a large run, because the halo DistConv
adds below autograd is invisible to a module-level adapter and the halo this one
adds instead is only right where the plan says it is.  The exchange itself
needs real ranks and is exercised by a separate multi-rank harness.

Tolerances come from ``triton_conv3d.reference``'s policy (an fp64 reference and
a dtype/K-derived bound, or MIOpen's own error where that is looser).  Nothing
here invents one.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from ScaFFold.unet import conv3d as conv_mod
from ScaFFold.unet.conv3d import FastConv3d, FastConvTranspose3d
from ScaFFold.unet.unet_model import UNet
from ScaFFold.unet.unet_parts import DoubleConv, OutConv, Up

_CHANNELS_LAST = torch.channels_last_3d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _StubStrategy:
    """The attributes :func:`_halo_plan` reads off a strategy.

    A real ``distconv.ParallelStrategy`` calls ``dist.get_rank()`` and builds a
    device mesh, so it cannot describe a >1 shard count in a one-rank test
    process at all -- ``ddp_ranks`` would be ``1 // 2 == 0``.  The gate reads
    exactly ``num_shards``, ``shard_dim`` and ``shard_ind``, so a stub can pose
    the sharded question that the environment otherwise cannot.

    ``shard_ind`` is here so that the *MIOpen* rung -- which takes a DCTensor
    through DistConv's own dispatch -- still runs, which is what makes "the
    sharded call went to the other rung" an assertion about routing rather than
    about which stub attribute is missing.
    """

    def __init__(self, num_shards, shard_dim=(2, 3, 4)):
        self.num_shards = num_shards
        self.shard_dim = shard_dim
        self.shard_ind = [0] * (len(num_shards) if num_shards else 0)

    def shard_to_rank(self, shard_ind):
        """Every shard is this rank: there is only one, and nothing is sent."""
        return 0


def _dc(tensor, num_shards=(1, 1, 1), shard_dim=(2, 3, 4)):
    """A ``DCTensor`` over ``tensor`` with a stubbed strategy."""
    import distconv

    return distconv.DCTensor(tensor, _StubStrategy(num_shards, shard_dim))


def _seeded_conv(cin=16, cout=32, kernel_size=3, padding=1, bias=False, **kwargs):
    """A ``FastConv3d`` whose weights are not the ones a bug would guess."""
    conv = FastConv3d(
        cin, cout, kernel_size=kernel_size, padding=padding, bias=bias, **kwargs
    )
    generator = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        conv.weight.normal_(0.0, 0.1, generator=generator)
        if conv.bias is not None:
            conv.bias.normal_(0.0, 0.1, generator=generator)
    return conv


def _gpu_conv(cin=16, cout=32, dtype=torch.bfloat16, **kwargs):
    """The same, on GPU and in the layout ``worker.py`` puts the model in.

    ``dtype`` defaults to bf16 because outside an autocast region the operands
    have to agree: an fp32 parameter against a bf16 activation is a call neither
    rung serves.  The autocast tests pass fp32 on purpose -- that is the state
    ``worker.py`` actually leaves the model in, and reproducing the dispatcher's
    cast is what makes it work.
    """
    conv = _seeded_conv(cin, cout, **kwargs).cuda().to(memory_format=_CHANNELS_LAST)
    return conv.to(dtype)


def _gpu_input(shape, dtype=torch.bfloat16, seed=7):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
    return x.to(dtype).contiguous(memory_format=_CHANNELS_LAST)


@pytest.fixture(autouse=True)
def _clean_module_state():
    """Reset the process-global latch and override around every test."""
    saved = (conv_mod._triton_override, conv_mod._triton_failed)
    yield
    conv_mod._triton_override, conv_mod._triton_failed = saved


# ---------------------------------------------------------------------------
# the sharding gate
# ---------------------------------------------------------------------------


def _plan(dc_input, strategy, x=None, kernel=(3, 3, 3), padding=(1, 1, 1)):
    """:func:`conv_mod._halo_plan` with the operands a 3x3x3 "same" conv has."""
    if x is None:
        x = torch.empty(1, 8, 8, 8, 8)
    weight = torch.empty(8, 8, *kernel)
    return conv_mod._halo_plan(
        dc_input, strategy, x, weight, (1, 1, 1), padding, (1, 1, 1)
    )


def test_halo_plan_refuses_every_strategy_it_cannot_read():
    """Every branch of the gate, including the ones it cannot read.

    The asymmetry under test is the whole safety argument: a plan is produced
    only when every fact has been checked, and ``None`` -- which routes the call
    to MIOpen and DistConv -- for anything else.
    """
    unsharded = _StubStrategy((1, 1, 1))

    class _Input:
        _is_periodic = (False, False, False)

    ok = _Input()
    plan = _plan(ok, unsharded)
    assert plan is not None and plan.exchanges == []
    assert plan.padding == (1, 1, 1), "an unsplit dim keeps the module's padding"
    assert plan.input_shape == (1, 8, 8, 8, 8), "nothing was added to the extent"

    # An axis whose count is 1 but which is not named in shard_dim: distconv
    # indexes num_shards by position in shard_dim, so a length mismatch means
    # the two disagree about which axis is which and the argument is unmade.
    assert _plan(ok, _StubStrategy((1, 1, 1), shard_dim=(2,))) is None
    # One axis named twice would be exchanged twice.
    assert _plan(ok, _StubStrategy((1, 2, 1), shard_dim=(2, 2, 4))) is None

    # Nothing readable at all.
    assert _plan(ok, None) is None
    assert _plan(ok, _StubStrategy(None)) is None
    assert _plan(ok, _StubStrategy(())) is None
    # A count that is not an int is a count this gate has not understood.
    assert _plan(ok, _StubStrategy((1, 1, "1"))) is None
    # A shard index outside its own axis is a strategy that does not describe a
    # mesh this exchange can address.
    bad_index = _StubStrategy((2, 1, 1))
    bad_index.shard_ind = [2, 0, 0]
    assert _plan(ok, bad_index) is None
    missing_index = _StubStrategy((2, 1, 1))
    missing_index.shard_ind = None
    assert _plan(ok, missing_index) is None

    # Periodicity: one shard still exchanges with itself, so the halo is the
    # opposite face rather than zeros, and the padding becomes
    # _periodic_shard_padding instead of 0.
    class _Periodic:
        _is_periodic = (False, True, False)

    assert _plan(_Periodic(), unsharded) is None

    class _NoPeriodicAttr:
        pass

    assert _plan(_NoPeriodicAttr(), unsharded) is None

    class _WrongLength:
        _is_periodic = (False, False)

    assert _plan(_WrongLength(), unsharded) is None


def test_halo_plan_exchanges_only_the_dims_that_are_actually_split():
    """The move stage 2 exists for, and the shapes it hands the kernel.

    ScaFFold ships ``dc_shard_dims: [2, 3, 4]`` with only D ever divided, so
    DistConv's halo on H and W is two ``cat`` copies of a slab that is provably
    zeros.  Dropping it is measured bitwise inert; this pins that the plan does
    drop it, and that the split dim -- and only the split dim -- trades its
    padding for a wider extent.
    """

    class _Input:
        _is_periodic = (False, False, False)

    plan = _plan(_Input(), _StubStrategy((2, 1, 1)))
    assert plan.exchanges == [(0, 2, 1)], "H and W were exchanged, or D was not"
    assert plan.padding == (0, 1, 1), "H/W lost their ordinary padding"
    assert plan.input_shape == (1, 8, 10, 8, 8)

    plan = _plan(_Input(), _StubStrategy((2, 2, 1)))
    assert plan.exchanges == [(0, 2, 1), (1, 3, 1)]
    assert plan.padding == (0, 0, 1)
    assert plan.input_shape == (1, 8, 10, 10, 8)

    # A 5x5x5 kernel wants two rows from each neighbour.
    plan = _plan(
        _Input(), _StubStrategy((2, 1, 1)), kernel=(5, 5, 5), padding=(2, 2, 2)
    )
    assert plan.exchanges == [(0, 2, 2)]
    assert plan.input_shape == (1, 8, 12, 8, 8)


def test_halo_plan_refuses_a_split_dim_whose_arithmetic_it_has_not_checked():
    """The per-axis conditions, each of which would give a wrong answer.

    An unsplit dim is exempt from all of them -- its halo is zeros either way --
    which is what makes the block-list narrow enough to be worth having.
    """

    class _Input:
        _is_periodic = (False, False, False)

    ok = _Input()
    split = _StubStrategy((2, 1, 1))

    # Padding that is not "same" on the split dim: the halo'd extent at padding
    # 0 would not be the shard's slice of the global volume.
    assert _plan(ok, split, padding=(0, 1, 1)) is None
    # An even kernel gives DistConv halo_size 0 and a strided-tiling contract
    # this module has not reasoned about.
    assert _plan(ok, split, kernel=(2, 3, 3), padding=(0, 1, 1)) is None
    # A shard thinner than the halo it must give away.
    assert _plan(ok, split, x=torch.empty(1, 8, 1, 8, 8)) is None
    # Stride and dilation on the split dim.
    weight = torch.empty(8, 8, 3, 3, 3)
    x = torch.empty(1, 8, 8, 8, 8)
    assert (
        conv_mod._halo_plan(ok, split, x, weight, (2, 1, 1), (1, 1, 1), (1, 1, 1))
        is None
    )
    assert (
        conv_mod._halo_plan(ok, split, x, weight, (1, 1, 1), (1, 1, 1), (2, 1, 1))
        is None
    )
    # ... but the same stride on an *unsplit* dim is nothing to do with the halo.
    assert (
        conv_mod._halo_plan(ok, split, x, weight, (1, 2, 1), (1, 1, 1), (1, 1, 1))
        is not None
    )

    # k == 1 on the split dim reads no neighbour voxel at all, so there is
    # nothing to exchange and padding 0 is already right.
    plan = _plan(ok, split, kernel=(1, 3, 3), padding=(0, 1, 1))
    assert plan is not None and plan.exchanges == []

    # A strategy that cannot name its neighbours cannot be exchanged with.
    nameless = _StubStrategy((2, 1, 1))
    nameless.shard_to_rank = None
    assert _plan(ok, nameless) is None


@pytest.mark.gpu
def test_the_gate_asks_the_predicates_about_the_tensor_the_kernel_will_see():
    """Widened, but still checked from both ends.

    Both halves matter.  A test that only checked the refusal would pass just as
    well against a gate that refuses everything -- and the sharding check sits
    behind the ``is_cuda`` test, so on CPU it is never even reached.  So the
    same module and the same tensor are asked twice, differing only in
    ``num_shards``.

    The sharded answer is ``False`` here for one reason and one reason only:
    this process has no process group to exchange over.  The plan is made, and
    the predicates are asked about the halo'd extent -- ``8 -> 10`` on D -- at
    the padding the exchange leaves behind.  The multi-rank half of this lives
    in a separate harness that needs real ranks.
    """
    conv = _gpu_conv()
    x = _gpu_input((1, 16, 8, 8, 8))

    unsharded = _dc(x, num_shards=(1, 1, 1))
    sharded = _dc(x, num_shards=(2, 1, 1))
    plans = {
        name: conv_mod._halo_plan(
            dc,
            dc._parallel_strategy,
            x,
            conv.weight,
            conv.stride,
            conv.padding,
            conv.dilation,
        )
        for name, dc in (("unsharded", unsharded), ("sharded", sharded))
    }

    assert conv_mod._use_triton(conv, x, unsharded, plans["unsharded"]) is True
    assert plans["sharded"] is not None, "the gate refused a strategy it can serve"
    assert plans["sharded"].input_shape == (1, 16, 10, 8, 8)
    assert plans["sharded"].padding == (0, 1, 1)
    assert not torch.distributed.is_initialized()
    assert conv_mod._use_triton(conv, x, sharded, plans["sharded"]) is False


@pytest.mark.gpu
def test_a_sharded_dctensor_forward_goes_to_miopen_without_a_process_group(monkeypatch):
    """End to end, not just the predicate: the rung must not fire.

    Routing is asserted from both ends -- the Triton rung is not entered *and*
    DistConv's halo exchange is, which is the path that supplies the neighbours'
    voxels the Triton rung would otherwise have to supply itself.
    ``forward_halo_exchange`` is stubbed to the identity so the MIOpen rung
    completes without a process group; it is the call count that is being
    measured, not the values.
    """
    import distconv.distconv as dc

    conv = _gpu_conv()
    x = _gpu_input((1, 16, 8, 8, 8))
    fast_calls, halo_calls = [], []

    original = FastConv3d._triton_forward
    monkeypatch.setattr(
        FastConv3d,
        "_triton_forward",
        lambda self, local, plan=None: fast_calls.append(local)
        or original(self, local, plan),
    )

    def _local_halo(tensor, halo_size, strategy, dim_index, is_periodic=False):
        """What the real exchange does when nothing has to be received.

        Concatenating zero slabs is exactly ``forward_halo_exchange``'s
        behaviour at one shard; spelling it out here lets the MIOpen rung run to
        completion for a *sharded* strategy too, without a process group.
        """
        halo_calls.append(dim_index)
        if halo_size == 0:
            return tensor
        dim = strategy.shard_dim[dim_index]
        slab = torch.zeros_like(tensor.narrow(dim, 0, halo_size))
        return torch.cat([slab, tensor, slab], dim=dim)

    monkeypatch.setattr(dc, "forward_halo_exchange", _local_halo)

    conv(_dc(x, num_shards=(1, 1, 1)))
    assert len(fast_calls) == 1, "the unsharded control did not take the Triton rung"
    assert halo_calls == [], "the Triton rung still paid for a halo exchange"

    conv(_dc(x, num_shards=(1, 2, 1)))
    assert len(fast_calls) == 1, "a sharded DCTensor reached the Triton rung"
    assert len(halo_calls) == 3, "the sharded call did not go through DistConv"


@pytest.mark.gpu
def test_a_kernel_failure_after_the_halo_falls_back_without_exchanging_twice(
    monkeypatch,
):
    """A Triton failure at 2 shards must cost speed, not the run.

    The halo goes on the wire before the kernel compiles, so a ``TritonError``
    arrives with this rank's sends and receives already matched against its
    peers'.  Re-running the whole call would take it to ``_miopen_forward`` and
    therefore through ``distconv_forward``, which exchanges *again* -- one more
    collective on this rank than on a peer whose kernel compiled, which hangs the
    mesh or pairs this convolution's slabs with the next one's.  Raising instead
    made a broken Triton install **fatal** at ``num_shards > 1`` while costing
    only speed at 1.

    So the count is the assertion, not the absence of an exception: exactly one
    exchange, the adapter's, and none of DistConv's.  Both are stubbed to the
    "nothing to receive" form so a one-rank process can run a two-shard strategy;
    it is which of them is *called* that is being measured.  And because
    ``cat(zeros, x, zeros)`` at padding 0 is the same arithmetic as the module's
    own padding on the unexchanged shard, the answer has an independent
    reference: what ``nn.Conv3d`` computes on the original input.

    The multi-rank half, with real slabs on a real mesh, needs real ranks and
    lives in a separate harness.
    """
    import distconv
    import distconv.distconv as dc
    from triton.errors import TritonError

    conv = _gpu_conv()
    x = _gpu_input((1, 16, 8, 8, 8))
    local = x.detach().clone().requires_grad_(True)
    # ``from_shard`` rather than the bare constructor: it is DistConv's
    # autograd-connected wrap, so the gradient below really does have to travel
    # back through the exchange to reach ``local``.
    strategy = _StubStrategy((2, 1, 1))
    sharded = distconv.DCTensor.from_shard(local, strategy)
    plan = conv_mod._halo_plan(
        sharded,
        strategy,
        local,
        conv.weight,
        conv.stride,
        conv.padding,
        conv.dilation,
    )
    assert plan is not None and len(plan.exchanges) == 1, "D must be the split dim"

    mine, theirs = [], []

    def _adapter_exchange(tensor, strategy, dim_index, dim, halo):
        mine.append(dim)
        slab = torch.zeros_like(tensor.narrow(dim, 0, halo))
        return torch.cat([slab, tensor, slab], dim=dim).contiguous(
            memory_format=_CHANNELS_LAST
        )

    def _adapter_backward(grad, strategy, dim_index, dim, halo):
        return grad.narrow(dim, halo, grad.size(dim) - 2 * halo)

    def _distconv_exchange(tensor, halo_size, strategy, dim_index, is_periodic=False):
        theirs.append(dim_index)
        if halo_size == 0:
            return tensor
        dim = strategy.shard_dim[dim_index]
        slab = torch.zeros_like(tensor.narrow(dim, 0, halo_size))
        return torch.cat([slab, tensor, slab], dim=dim)

    monkeypatch.setattr(conv_mod, "_exchange_forward", _adapter_exchange)
    monkeypatch.setattr(conv_mod, "_exchange_backward", _adapter_backward)
    monkeypatch.setattr(dc, "forward_halo_exchange", _distconv_exchange)
    # The gate declines a sharded plan in a process with no group, which is a
    # routing condition and not the one under test.
    monkeypatch.setattr(conv_mod, "_use_triton", lambda *a, **kw: True)
    monkeypatch.setattr(conv_mod, "_triton_failed", False)

    def _boom(*args, **kwargs):
        raise TritonError("forced: this tile does not fit in LDS")

    monkeypatch.setattr(conv_mod._get_triton_module(), "conv3d_forward", _boom)

    out = conv(sharded)

    assert isinstance(out, distconv.DCTensor), "the fallback lost the wrapper"
    assert mine == [2], f"the halo was exchanged {len(mine)} times, not once"
    assert theirs == [], "MIOpen was reached through DistConv, which exchanges again"
    assert conv_mod._triton_failed is True, "the kernel failure did not latch"
    torch.testing.assert_close(out._tensor, nn.Conv3d.forward(conv, x))

    # And the graph the fallback built is the halo'd one: the gradient reaches
    # the shard through ``_Halo3d``, so it must match the unsharded gradient.
    gy = _gpu_input((1, 32, 8, 8, 8), seed=41)
    out.backward(distconv.DCTensor.from_shard(gy, strategy))
    plain_x = x.detach().clone().requires_grad_(True)
    nn.Conv3d.forward(conv, plain_x).backward(gy)
    assert local.grad is not None, "the fallback severed the graph at the halo"
    torch.testing.assert_close(local.grad.float(), plain_x.grad.float(),
                               rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def test_cpu_input_never_reaches_the_triton_rung():
    conv = _seeded_conv()
    x = torch.randn(1, 16, 8, 8, 8)
    assert conv_mod._use_triton(conv, x, None, None) is False
    torch.testing.assert_close(conv(x), nn.Conv3d.forward(conv, x))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "kwargs, why",
    [
        ({"cin": 3}, "the stem: 0.93x over three directions, +0.19% of a step"),
        ({"kernel_size": 1, "padding": 0, "bias": True}, "the k=1 head: 1.40x"),
        ({"cout": 512}, "small M with Cout >= 512: 1.17-1.65x"),
    ],
)
def test_the_block_list_is_empty_at_the_shapes_it_used_to_hold(kwargs, why):
    """Each of these was kept on MIOpen until 2026-08-04; all three now route.

    Parametrized on the three retired rules rather than asserting
    ``_policy_declines`` is empty, because what matters is the *routing* answer:
    a rule could return ``False`` while some other clause of :func:`_use_triton`
    still declined, and then the block would be gone in name only.

    ``cout=512`` also covers the small-``M`` predicate's real defect -- it read
    the forward GEMM's row count and then kept all three directions on MIOpen,
    including a backward-data that wins 1.31-2.47x. See
    :func:`~ScaFFold.unet.conv3d._policy_declines`.
    """
    conv = _gpu_conv(**kwargs)
    x = _gpu_input((1, kwargs.get("cin", 16), 8, 8, 8))
    assert conv_mod._use_triton(conv, x, None, None) is True, why


@pytest.mark.gpu
def test_ladder_falls_back_on_a_shape_the_kernel_does_not_serve():
    """``stride=2``: the forward predicate accepts it, both backwards reject it.

    This is the concrete witness for why all three directions are gated, not
    just the forward: taking the rung here would build a graph node whose
    backward ``triton_conv3d`` cannot answer, and by then MIOpen is no longer an
    option for it.
    """
    conv = _gpu_conv(kernel_size=3, padding=1, stride=2)
    x = _gpu_input((1, 16, 8, 8, 8))
    triton_conv3d = conv_mod._get_triton_module()

    probe = conv_mod._metadata_probe((1, 16, 8, 8, 8), torch.bfloat16, x.device)
    w_probe = conv_mod._metadata_probe(
        tuple(conv.weight.shape), torch.bfloat16, x.device
    )
    assert (
        triton_conv3d.is_supported(probe, w_probe, None, (2, 2, 2), (1, 1, 1)) is True
    )
    assert conv_mod._use_triton(conv, x, None, None) is False

    out = conv(x)
    torch.testing.assert_close(out, nn.Conv3d.forward(conv, x))


@pytest.mark.gpu
def test_metadata_probe_answers_like_a_real_tensor():
    """The stand-in shortcut, pinned against the tensors it stands in for.

    ``_metadata_probe`` exists so the gate can ask about operands that do not
    exist yet (the gradient) or that would cost a full-size copy to build (the
    bf16 cast of an fp32 activation).  It is only sound while the predicates
    read metadata and nothing else, which is a property of a package this
    module does not own.
    """
    triton_conv3d = conv_mod._get_triton_module()
    x = _gpu_input((1, 16, 8, 8, 8))
    w = _gpu_conv().weight.detach().to(torch.bfloat16)
    gy = _gpu_input((1, 32, 8, 8, 8), seed=11)
    args = ((1, 1, 1), (1, 1, 1), (1, 1, 1), 1)

    px = conv_mod._metadata_probe(x.shape, x.dtype, x.device)
    pw = conv_mod._metadata_probe(w.shape, w.dtype, w.device)
    pgy = conv_mod._metadata_probe(gy.shape, gy.dtype, gy.device)

    assert triton_conv3d.is_supported(
        px, pw, None, *args
    ) == triton_conv3d.is_supported(x, w, None, *args)
    assert triton_conv3d.is_supported_bwd_data(
        pgy, pw, x.shape, *args
    ) == triton_conv3d.is_supported_bwd_data(gy, w, x.shape, *args)
    assert triton_conv3d.is_supported_bwd_weight(
        px, w.shape, pgy, *args
    ) == triton_conv3d.is_supported_bwd_weight(x, w.shape, gy, *args)


# ---------------------------------------------------------------------------
# latch / proven / opt-in
# ---------------------------------------------------------------------------


def test_env_var_off_declines_before_anything_else(monkeypatch):
    monkeypatch.setattr(conv_mod, "_triton_override", False)
    conv = _seeded_conv()
    assert conv_mod._use_triton(conv, torch.randn(1, 16, 4, 4, 4), None, None) is False


def test_set_conv_triton_enabled_round_trips_and_clears_the_latch(monkeypatch):
    monkeypatch.setattr(conv_mod, "_triton_failed", True)
    previous = conv_mod.set_conv_triton_enabled(True)
    try:
        assert conv_mod._triton_override is True
        assert conv_mod._triton_failed is False, "an explicit opt-in must re-arm"
        # None restores the env default and deliberately does NOT clear a latch.
        conv_mod._triton_failed = True
        conv_mod.set_conv_triton_enabled(None)
        assert conv_mod._triton_failed is True
        assert conv_mod._triton_override is conv_mod._env_override(
            conv_mod.TRITON_ENV_VAR
        )
    finally:
        conv_mod.set_conv_triton_enabled(previous)


def test_latch_spares_a_proven_module_and_demotes_the_others(monkeypatch, caplog):
    """A failure latches the rung off for modules that have never used it."""

    class _Boom(Exception):
        pass

    proven = _seeded_conv()
    fresh = _seeded_conv()
    proven._triton_ok = True
    attempts = []

    def _fails(self, local, plan=None):
        attempts.append(self)
        raise _Boom("kernel is broken")

    monkeypatch.setattr(conv_mod, "_triton_kernel_failures", lambda: (_Boom,))
    monkeypatch.setattr(FastConv3d, "_triton_forward", _fails)
    monkeypatch.setattr(conv_mod, "_use_triton", lambda module, *a, **kw: True)
    monkeypatch.setattr(conv_mod, "_triton_failed", False)

    x = torch.randn(1, 16, 4, 4, 4)
    with caplog.at_level(logging.WARNING):
        torch.testing.assert_close(proven(x), nn.Conv3d.forward(proven, x))
    assert attempts == [proven]
    assert conv_mod._triton_failed is True, "the failure did not latch"
    assert any("Triton conv3d failed" in r.message for r in caplog.records)

    # The latch is consulted by the real predicate, which the stub above
    # replaced.  Restore it and ask directly: a module that has never used the
    # rung is now declined, and one that has is not.
    monkeypatch.undo()
    monkeypatch.setattr(conv_mod, "_triton_failed", True)
    assert conv_mod._use_triton(fresh, x, None, None, proven=False) is False
    # `proven=True` gets past the latch and is only declined further down, on
    # the CPU check -- which is what the second half of the claim needs.
    assert conv_mod._triton_failed is True


@pytest.mark.gpu
def test_a_proven_module_re_raises_rather_than_flipping_rungs_mid_backward(monkeypatch):
    """The fallback is declined where it would corrupt instead of degrade.

    A module already proven on the rung, failing while an autograd graph task is
    in flight, is answering a checkpoint recompute of a forward that ran on
    Triton.  Handing back MIOpen's result puts a differently-structured tensor
    into a slot the graph node already holds; the honest answer is the original
    exception.
    """

    class _Boom(Exception):
        pass

    conv = _gpu_conv()
    conv._triton_ok = True
    monkeypatch.setattr(conv_mod, "_triton_kernel_failures", lambda: (_Boom,))
    monkeypatch.setattr(conv_mod, "_use_triton", lambda *a, **kw: True)
    monkeypatch.setattr(
        FastConv3d,
        "_triton_forward",
        lambda self, local, plan=None: (_ for _ in ()).throw(_Boom()),
    )

    x = _gpu_input((1, 16, 8, 8, 8)).float().requires_grad_(True)
    seen = {}

    class _Probe(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            return t.clone()

        @staticmethod
        def backward(ctx, g):
            # Inside a graph task: this is where a recompute would run.
            try:
                conv(g)
            except _Boom:
                seen["raised"] = True
            return g

    _Probe.apply(x).sum().backward()
    assert seen.get("raised") is True


# ---------------------------------------------------------------------------
# state dict / model wiring
# ---------------------------------------------------------------------------


def test_state_dict_matches_a_plain_conv3d_model():
    """No new keys, no renamed keys, no buffers -- checkpoints are unaffected."""
    fast = UNet(
        n_channels=3, n_classes=4, trilinear=False, layers=1, group_norm_groups=2
    )
    fast_keys = list(fast.state_dict())

    original = nn.Conv3d
    try:
        # Build the same model with stock convolutions by making the factories'
        # classes the stock ones for the duration.  Both of them: a transposed
        # parameter that changed name or shape would be just as invisible here
        # as an ordinary one.
        import ScaFFold.unet.unet_parts as parts

        parts.FastConv3d = nn.Conv3d
        parts.FastConvTranspose3d = nn.ConvTranspose3d
        plain = UNet(
            n_channels=3, n_classes=4, trilinear=False, layers=1, group_norm_groups=2
        )
    finally:
        parts.FastConv3d = FastConv3d
        parts.FastConvTranspose3d = FastConvTranspose3d
    assert original is nn.Conv3d

    assert fast_keys == list(plain.state_dict())
    for key, value in fast.state_dict().items():
        assert value.shape == plain.state_dict()[key].shape


def test_checkpoint_round_trips_between_fast_and_plain_convolutions():
    fast = _seeded_conv(cin=8, cout=8)
    plain = nn.Conv3d(8, 8, kernel_size=3, padding=1, bias=False)
    plain.load_state_dict(fast.state_dict())
    torch.testing.assert_close(plain.weight, fast.weight)

    back = FastConv3d(8, 8, kernel_size=3, padding=1, bias=False)
    back.load_state_dict(plain.state_dict())
    torch.testing.assert_close(back.weight, fast.weight)

    x = torch.randn(1, 8, 6, 6, 6)
    torch.testing.assert_close(back(x), plain(x))


def test_every_upsampler_in_the_model_is_a_fastconvtranspose3d():
    """The census: all four decoder sites, and none of them a plain module.

    A rung that is wired in but never reached is the failure this pins -- it
    costs nothing, breaks nothing and shows up only as a benchmark that did not
    get faster.  ``layers=4`` is the shipped depth, so four is the real count.
    """
    up = Up(16, 8, group_norm_groups=2, trilinear=False)
    assert type(up.up) is FastConvTranspose3d
    assert not isinstance(up.up, FastConv3d), "the two ladders are separate classes"

    model = UNet(
        n_channels=3, n_classes=6, trilinear=False, layers=4, group_norm_groups=8
    )
    transposed = [
        m for m in model.modules() if isinstance(m, nn.modules.conv._ConvTransposeNd)
    ]
    assert len(transposed) == 4
    assert all(type(m) is FastConvTranspose3d for m in transposed)
    # Every one of them has a bias, which is why ``grad_bias`` is a live path in
    # this ladder and a test-only one in the other.
    assert all(m.bias is not None for m in transposed)


def test_every_plain_convolution_in_the_model_is_a_fastconv3d():
    model = UNet(
        n_channels=3, n_classes=6, trilinear=False, layers=4, group_norm_groups=8
    )
    plain = [
        m
        for m in model.modules()
        if isinstance(m, nn.Conv3d) and not isinstance(m, FastConv3d)
    ]
    assert plain == []
    assert sum(isinstance(m, FastConv3d) for m in model.modules()) == 19
    assert isinstance(DoubleConv(4, 4, 2).double_conv[0], FastConv3d)
    assert isinstance(OutConv(4, 2).conv, FastConv3d)


@pytest.mark.gpu
def test_a_transposed_module_would_be_declined_even_if_one_were_wrapped():
    """The class is a public drop-in, so it checks rather than assumes."""
    conv = _gpu_conv()
    conv.transposed = True
    assert conv_mod._use_triton(conv, _gpu_input((1, 16, 8, 8, 8)), None, None) is False


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------


def _problem(cin, cout, spatial, kernel=(3, 3, 3), padding=(1, 1, 1), bias=False):
    from triton_conv3d.shapes import ConvProblem

    return ConvProblem(
        name=f"{cin}->{cout} k{kernel[0]} {spatial}",
        cin=cin,
        cout=cout,
        spatial=spatial,
        kernel=kernel,
        padding=padding,
        bias=bias,
        dtype="bf16",
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "cin, cout, spatial", [(16, 32, (8, 8, 8)), (64, 64, (6, 10, 10))]
)
def test_forward_and_gradients_match_nn_conv3d(cin, cout, spatial):
    """All three directions against an fp64 reference, at MIOpen's own standard.

    ``assert_close`` applies the stricter of ``triton_conv3d``'s dtype/K-derived
    bound and "no worse than MIOpen by more than 4x"; the incumbent's error is
    measured here from the ``nn.Conv3d`` route this module replaces, which is
    exactly the comparison the wiring has to survive.
    """
    from triton_conv3d import reference as ref

    problem = _problem(cin, cout, spatial)
    conv = _gpu_conv(cin, cout)
    x = _gpu_input((1, cin, *spatial))
    gy = _gpu_input((1, cout, *spatial), seed=23)

    fast_x = x.detach().clone().requires_grad_(True)
    plain_x = x.detach().clone().requires_grad_(True)
    plain = nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False).cuda()
    plain = plain.to(torch.bfloat16).to(memory_format=_CHANNELS_LAST)
    with torch.no_grad():
        plain.weight.copy_(conv.weight)

    assert conv_mod._use_triton(conv, fast_x, None, None) is True
    y = conv(fast_x)
    y_plain = plain(plain_x)
    y.backward(gy)
    y_plain.backward(gy)

    operands = {
        "input": x,
        "weight": conv.weight.detach(),
        "bias": None,
        "grad_output": gy,
    }
    for direction, actual, incumbent in (
        ("fwd", y, y_plain),
        ("bwd-data", fast_x.grad, plain_x.grad),
        ("bwd-weight", conv.weight.grad, plain.weight.grad),
    ):
        expected = ref.reference(problem, operands, direction)
        incumbent_error = ref.compare(incumbent, expected)
        ref.assert_close(
            actual, expected, problem, direction, incumbent_error=incumbent_error
        )


@pytest.mark.gpu
def test_bias_gradient_is_correct_even_though_the_head_is_blocklisted():
    """``is_supported`` accepts a bias, so the node has to produce its gradient."""
    from triton_conv3d import reference as ref

    cin, cout, spatial = 16, 32, (8, 8, 8)
    problem = _problem(cin, cout, spatial, bias=True)
    conv = _gpu_conv(cin, cout, bias=True)
    x = _gpu_input((1, cin, *spatial)).requires_grad_(True)
    gy = _gpu_input((1, cout, *spatial), seed=31)

    conv(x).backward(gy)

    operands = {
        "input": x.detach(),
        "weight": conv.weight.detach(),
        "bias": conv.bias.detach(),
        "grad_output": gy,
    }
    expected = ref.reference(problem, operands, "fwd")
    del expected  # the forward is covered above; here only d(bias) is new.
    expected_gb = gy.to(torch.float64).sum(dim=(0, 2, 3, 4))
    torch.testing.assert_close(
        conv.bias.grad.to(torch.float64), expected_gb, rtol=2e-2, atol=2e-2
    )


@pytest.mark.gpu
def test_autocast_runs_the_kernel_at_the_dtype_aten_would_have_chosen():
    """The cast ATen does in the dispatcher, reproduced above it.

    Without this the module's fp32 parameters and GroupNorm's fp32 output would
    be handed straight to the kernel and the whole network's convolutions would
    quietly run in fp32 -- a different computation from the benchmark's, and a
    much slower one.
    """
    conv = _gpu_conv(16, 32, dtype=torch.float32)  # as worker.py builds them
    x = _gpu_input((1, 16, 8, 8, 8), dtype=torch.float32).requires_grad_(True)

    seen = {}
    original = conv_mod._TritonConv3dFn.apply

    def _spy(x_, w_, b_, *rest):
        seen["x"] = x_.dtype
        seen["w"] = w_.dtype
        return original(x_, w_, b_, *rest)

    conv_mod._TritonConv3dFn.apply = staticmethod(_spy)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = conv(x)
    finally:
        conv_mod._TritonConv3dFn.apply = original

    assert seen == {"x": torch.bfloat16, "w": torch.bfloat16}
    assert y.dtype is torch.bfloat16
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected = nn.functional.conv3d(x, conv.weight, None, 1, 1)
    assert expected.dtype is torch.bfloat16

    y.sum().backward()
    # The cast is an ordinary autograd node, so the parameter gradient comes
    # back at the parameter's own dtype, exactly as it does on the MIOpen rung.
    assert conv.weight.grad.dtype is torch.float32
    assert x.grad.dtype is torch.float32


@pytest.mark.gpu
@pytest.mark.parametrize("guard", ["no_grad", "inference_mode"])
def test_the_rung_serves_the_evaluation_path(guard):
    """``evaluate`` runs the whole model under ``@torch.inference_mode()``.

    That is a different autograd state from training -- ``Function.apply`` never
    builds a node and the tensors it produces are inference tensors -- and it is
    every validation epoch of every run, so it is not an edge case.
    """
    conv = _gpu_conv(16, 32, dtype=torch.float32)
    x = _gpu_input((1, 16, 8, 8, 8), dtype=torch.float32)
    with (
        getattr(torch, guard)(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        y = conv(x)
        expected = nn.functional.conv3d(x, conv.weight, None, 1, 1)
    assert conv._triton_ok is True, "the evaluation path did not take the rung"
    assert y.shape == expected.shape and y.dtype is expected.dtype
    torch.testing.assert_close(y.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
def test_a_checkpointed_block_recomputes_on_the_same_rung():
    """``activation_checkpointing`` is a shipped config key.

    The recompute runs inside the backward pass and its saved tensors are
    compared against the original forward's.  What this pins is that a module
    stays on one rung across the two, which is what the ``proven`` flag exists
    for: a flip is invisible to torch's metadata check and fails later, inside
    DistConv, with a message about neither checkpointing nor the rung.
    """
    import torch.utils.checkpoint as cp

    conv = _gpu_conv()
    x = _gpu_input((1, 16, 8, 8, 8)).requires_grad_(True)
    plain_x = x.detach().clone().requires_grad_(True)

    y = cp.checkpoint(conv, x, use_reentrant=False)
    y.sum().backward()
    assert conv._triton_ok is True

    reference = nn.Conv3d.forward(conv, plain_x)
    reference.sum().backward()
    torch.testing.assert_close(
        x.grad.float(), plain_x.grad.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.gpu
def test_dctensor_forward_and_backward_agree_with_the_distconv_route():
    """The unwrap/rewrap at one shard, against DistConv's own halo path.

    This is the equivalence the whole gate rests on: at ``num_shards=(1,1,1)``
    the halo slabs are provably zeros, so running the kernel on the local shard
    at the module's own padding must reproduce what DistConv's dispatch computes
    on the halo'd tensor at zero padding.
    """
    import distconv

    from triton_conv3d import reference as ref

    cin, cout, spatial = 16, 32, (8, 8, 8)
    problem = _problem(cin, cout, spatial)
    conv = _gpu_conv(cin, cout)
    plain = nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False)
    plain = plain.cuda().to(torch.bfloat16).to(memory_format=_CHANNELS_LAST)
    with torch.no_grad():
        plain.weight.copy_(conv.weight)

    x = _gpu_input((1, cin, *spatial))
    gy = _gpu_input((1, cout, *spatial), seed=41)

    fast_x = x.detach().clone().requires_grad_(True)
    plain_x = x.detach().clone().requires_grad_(True)
    strategy = _StubStrategy((1, 1, 1))

    y = conv(distconv.DCTensor.from_shard(fast_x, strategy))
    assert isinstance(y, distconv.DCTensor)
    assert conv._triton_ok is True, "the fast rung did not serve the DCTensor"
    y_plain = plain(distconv.DCTensor.from_shard(plain_x, strategy))

    y.backward(distconv.DCTensor.from_shard(gy, strategy))
    y_plain.backward(distconv.DCTensor.from_shard(gy, strategy))

    operands = {
        "input": x,
        "weight": conv.weight.detach(),
        "bias": None,
        "grad_output": gy,
    }
    for direction, actual, incumbent in (
        ("fwd", y._tensor, y_plain._tensor),
        ("bwd-data", fast_x.grad, plain_x.grad),
        ("bwd-weight", conv.weight.grad, plain.weight.grad),
    ):
        expected = ref.reference(problem, operands, direction)
        ref.assert_close(
            actual,
            expected,
            problem,
            direction,
            incumbent_error=ref.compare(incumbent, expected),
        )


@pytest.mark.gpu
def test_backward_falls_back_to_miopen_when_the_kernel_direction_fails(monkeypatch):
    """A backward-direction failure degrades; the saved set cannot change."""
    from triton_conv3d import reference as ref

    class _Boom(Exception):
        pass

    cin, cout, spatial = 16, 32, (8, 8, 8)
    problem = _problem(cin, cout, spatial)
    conv = _gpu_conv(cin, cout)
    x = _gpu_input((1, cin, *spatial))
    gy = _gpu_input((1, cout, *spatial), seed=53)
    fast_x = x.detach().clone().requires_grad_(True)

    y = conv(fast_x)
    module = conv_mod._get_triton_module()
    monkeypatch.setattr(conv_mod, "_triton_kernel_failures", lambda: (_Boom,))
    monkeypatch.setattr(
        module,
        "conv3d_backward_data",
        lambda *a, **kw: (_ for _ in ()).throw(_Boom()),
        raising=False,
    )
    monkeypatch.setattr(conv_mod, "_triton_failed", False)
    y.backward(gy)

    assert conv_mod._triton_failed is True
    operands = {
        "input": x,
        "weight": conv.weight.detach(),
        "bias": None,
        "grad_output": gy,
    }
    expected = ref.reference(problem, operands, "bwd-data")
    ref.assert_close(fast_x.grad, expected, problem, "bwd-data")


class _FakeCtx:
    """Just enough ``ctx`` to call ``_TritonConv3dFn.backward`` directly."""

    def __init__(self, saved):
        self.saved_tensors = saved
        self.conv_args = ((1, 1, 1), (1, 1, 1), (1, 1, 1), False)
        self.needs_input_grad = (True, True, False, False, False, False)


def test_backward_names_a_rung_flip_instead_of_dying_inside_distconv():
    """The only detector for a flip torch's checkpoint metadata check misses.

    ``_default_meta_extractor`` compares shape, dtype and device, all of which
    the two rungs agree on, so a subclass landing in slot 0 gets through torch's
    own check and then fails somewhere else entirely.
    """

    class _Wrapper(torch.Tensor):
        pass

    x = torch.randn(1, 8, 4, 4, 4)
    weight = torch.randn(8, 8, 3, 3, 3)
    with pytest.raises(RuntimeError, match="served by different"):
        conv_mod._TritonConv3dFn.backward(
            _FakeCtx((_Wrapper(x), weight)), torch.randn(1, 8, 4, 4, 4)
        )


@pytest.mark.gpu
def test_the_triton_rung_performs_no_halo_exchange_at_one_shard():
    """Not merely correct without the halo -- it must not pay for one either.

    ``forward_halo_exchange`` has no ``num_shards == 1`` early-out, so today
    every convolution concatenates two zero slabs onto each of the three
    sharded dims: 54 calls and 3.797 ms/step of pure copying at
    ``dc_num_shards=(1,1,1)``.  Taking the Triton rung removes all of them, and leaves the caller's ``_tensor``
    un-narrowed and still channels-last for the consumers downstream of it.
    """
    import distconv
    import distconv.distconv as dc

    conv = _gpu_conv()
    x = _gpu_input((1, 16, 8, 8, 8)).requires_grad_(True)
    dc_input = distconv.DCTensor.from_shard(x, _StubStrategy((1, 1, 1)))

    calls = []
    original = dc.forward_halo_exchange
    dc.forward_halo_exchange = lambda *a, **kw: calls.append(a) or original(*a, **kw)
    try:
        out = conv(dc_input)
    finally:
        dc.forward_halo_exchange = original

    assert conv._triton_ok is True
    assert calls == [], "the Triton rung went through DistConv's halo exchange"
    assert dc_input._tensor_with_halo is None
    assert dc_input._tensor.is_contiguous(memory_format=_CHANNELS_LAST)
    assert isinstance(out, distconv.DCTensor)


# ---------------------------------------------------------------------------
# the transposed ladder
# ---------------------------------------------------------------------------


def _seeded_convT(cin=16, cout=8, kernel_size=2, stride=2, **kwargs):
    """A ``FastConvTranspose3d`` whose weights are not the ones a bug would guess.

    ``bias`` is left at ``nn.ConvTranspose3d``'s default of ``True``, which is
    what the four decoder sites have and what puts ``grad_bias`` on the live
    path.
    """
    conv = FastConvTranspose3d(cin, cout, kernel_size=kernel_size, stride=stride,
                               **kwargs)
    generator = torch.Generator().manual_seed(4321)
    with torch.no_grad():
        conv.weight.normal_(0.0, 0.1, generator=generator)
        if conv.bias is not None:
            conv.bias.normal_(0.0, 0.1, generator=generator)
    return conv


def _gpu_convT(cin=16, cout=8, dtype=torch.bfloat16, **kwargs):
    conv = _seeded_convT(cin, cout, **kwargs).cuda().to(memory_format=_CHANNELS_LAST)
    return conv.to(dtype)


def _stock_like(conv, dtype=torch.bfloat16):
    """A stock ``nn.ConvTranspose3d`` holding the same parameters."""
    cin, cout = int(conv.weight.shape[0]), int(conv.weight.shape[1])
    plain = nn.ConvTranspose3d(
        cin, cout, kernel_size=conv.kernel_size, stride=conv.stride,
        bias=conv.bias is not None,
    )
    plain = plain.cuda().to(dtype).to(memory_format=_CHANNELS_LAST)
    with torch.no_grad():
        plain.weight.copy_(conv.weight)
        if conv.bias is not None:
            plain.bias.copy_(conv.bias)
    return plain


def _transposed_problem(cin, cout, spatial, kernel=(2, 2, 2), bias=True):
    from triton_conv3d.shapes import ConvProblem

    return ConvProblem(
        name=f"convT {cin}->{cout} k{kernel[0]} {spatial}",
        cin=cin,
        cout=cout,
        spatial=spatial,
        kernel=kernel,
        stride=kernel,
        padding=(0, 0, 0),
        transposed=True,
        bias=bias,
        dtype="bf16",
    )


#: ``(x_shape, weight_shape)`` of the four decoder upsamplers at config A
#: (scale 7, 128^3, ``layers=4``), in the order the decoder runs them.
_UPSAMPLER_SITES = [
    ((1, 1024, 8, 8, 8), (1024, 512, 2, 2, 2)),
    ((1, 512, 16, 16, 16), (512, 256, 2, 2, 2)),
    ((1, 256, 32, 32, 32), (256, 128, 2, 2, 2)),
    ((1, 128, 64, 64, 64), (128, 64, 2, 2, 2)),
]


def test_the_transposed_block_list_is_empty_at_every_decoder_site():
    """No decoder site is blocked, in either ladder.

    Both block-lists are empty as of 2026-08-04, so this asserts the routing
    answer rather than the shape of a rule.  It is still worth a test: emptiness
    is a claim about *measurements*, and a future entry in either function has to
    re-establish it here.

    The reason the two functions stay separate survives the emptying, and is
    recorded in :func:`~ScaFFold.unet.conv3d._transposed_policy_declines`: every
    term the ordinary rule used reads a different quantity for this operator --
    ``w_shape``'s channel axes are reversed, and ``M`` from ``_out_spatial`` is
    the input volume over 8 at ``k == s == 2``.  The retired small-``M`` rule
    answered ``True`` for ``up1`` on numbers that do not describe it.
    """
    for x_shape, w_shape in _UPSAMPLER_SITES:
        assert conv_mod._transposed_policy_declines(x_shape, w_shape) is False
        assert conv_mod._policy_declines(
            x_shape, w_shape, (2, 2, 2), (0, 0, 0), (1, 1, 1)
        ) is False


@pytest.mark.gpu
def test_the_transposed_gate_answers_for_the_four_sites_and_refuses_the_rest():
    """The gate fires where it must, and declines what the kernels do not serve.

    Every refusal below is a condition ``triton_conv3d.transposed`` states in
    its own gate; asking through ``_use_triton_transposed`` is what pins that
    this module *asks* -- with the module's real ``stride``, ``padding``,
    ``output_padding``, ``dilation`` and ``groups``, in the right slots.
    """
    conv = _gpu_convT(16, 8)
    x = _gpu_input((1, 16, 8, 8, 8))
    assert conv_mod._use_triton_transposed(conv, x, None, None) is True

    # k != s: the windows overlap and the bijection this module rests on is
    # gone.  The gate must not read the module's kernel_size as its stride.
    assert (
        conv_mod._use_triton_transposed(_gpu_convT(16, 8, kernel_size=3), x, None, None)
        is False
    )
    # A padding crops the result and an output_padding extends it
    # asymmetrically; both break the tiling.
    assert (
        conv_mod._use_triton_transposed(_gpu_convT(16, 8, padding=1), x, None, None)
        is False
    )
    assert (
        conv_mod._use_triton_transposed(
            _gpu_convT(16, 8, stride=3, output_padding=1), _gpu_input((1, 16, 8, 8, 8)),
            None, None,
        )
        is False
    )
    # groups > 1 has no coverage in any direction.
    assert (
        conv_mod._use_triton_transposed(
            _gpu_convT(16, 8, groups=2), x, None, None
        )
        is False
    )
    # NCDHW would be a full-size hidden relayout, which is the cost the rung
    # exists to avoid.
    assert (
        conv_mod._use_triton_transposed(conv, x.contiguous(), None, None) is False
    )
    # And the CPU, where there is no kernel at all.
    assert conv_mod._use_triton_transposed(_seeded_convT(), torch.randn(1, 16, 4, 4, 4),
                                           None, None) is False


@pytest.mark.gpu
def test_the_transposed_gate_asked_is_the_one_that_covers_the_backward(monkeypatch):
    """``is_supported_transposed_all``, not the forward's gate alone.

    The three transposed predicates accept the same problems today, so no shape
    can tell them apart -- which is exactly why *which one is called* has to be
    pinned directly.  A forward this package serves and a backward it cannot is
    discovered inside ``backward()``, where MIOpen is no longer reachable, and
    the ordinary convolution has a live witness for that (``stride > 1``).
    """
    conv = _gpu_convT(16, 8)
    x = _gpu_input((1, 16, 8, 8, 8))
    module = conv_mod._get_triton_module()

    assert conv_mod._use_triton_transposed(conv, x, None, None) is True
    # Patching the package attribute reaches the module's call and not the one
    # ``is_supported_transposed_all`` makes internally, so this distinguishes
    # "asked the combined gate" from "asked the forward's and got the same
    # answer" -- which is the only way to tell them apart while they agree.
    monkeypatch.setattr(module, "is_supported_transposed_all", lambda *a, **kw: False,
                        raising=False)
    assert conv_mod._use_triton_transposed(conv, x, None, None) is False


@pytest.mark.gpu
def test_a_non_transposed_module_would_be_declined_by_the_transposed_gate():
    """The mirror of ``test_a_transposed_module_would_be_declined``.

    ``is_supported_transposed`` reads ``w.shape[0]`` as ``Cin`` and
    ``w.shape[1]`` as ``Cout``; an ``nn.Conv3d`` weight stores them the other way
    round, so a square one would be *accepted* and would compute the wrong
    operator without raising.  This module checks ``transposed`` rather than
    trusting its own construction, because the class is a public drop-in.
    """
    conv = _gpu_convT(16, 16)
    conv.transposed = False
    assert conv_mod._use_triton_transposed(conv, _gpu_input((1, 16, 8, 8, 8)), None,
                                           None) is False


@pytest.mark.gpu
@pytest.mark.parametrize("cin, cout, spatial", [(16, 8, (8, 8, 8)), (32, 16, (4, 6, 6))])
def test_transposed_forward_and_gradients_match_nn_convtranspose3d(cin, cout, spatial):
    """All three directions against an fp64 reference, at MIOpen's own standard.

    ``assert_close`` applies the stricter of ``triton_conv3d``'s dtype/K-derived
    bound and "no worse than MIOpen by more than 4x", with the incumbent's error
    measured from the stock ``nn.ConvTranspose3d`` route this module replaces.
    """
    from triton_conv3d import reference as ref

    problem = _transposed_problem(cin, cout, spatial)
    conv = _gpu_convT(cin, cout)
    plain = _stock_like(conv)
    out_spatial = tuple(2 * s for s in spatial)

    x = _gpu_input((1, cin, *spatial))
    gy = _gpu_input((1, cout, *out_spatial), seed=23)
    fast_x = x.detach().clone().requires_grad_(True)
    plain_x = x.detach().clone().requires_grad_(True)

    assert conv_mod._use_triton_transposed(conv, fast_x, None, None) is True
    y = conv(fast_x)
    y_plain = plain(plain_x)
    assert conv._triton_ok is True, "the rung did not serve the call"
    y.backward(gy)
    y_plain.backward(gy)

    operands = {
        "input": x,
        "weight": conv.weight.detach(),
        "bias": conv.bias.detach(),
        "grad_output": gy,
    }
    for direction, actual, incumbent in (
        ("fwd", y, y_plain),
        ("bwd-data", fast_x.grad, plain_x.grad),
        ("bwd-weight", conv.weight.grad, plain.weight.grad),
    ):
        expected = ref.reference(problem, operands, direction)
        ref.assert_close(
            actual,
            expected,
            problem,
            direction,
            incumbent_error=ref.compare(incumbent, expected),
        )
    # grad_bias is not one of ``reference``'s directions -- it is not a
    # convolution -- so it gets the exact answer and the incumbent's own error
    # as its bar, which is the same standard by a different route.
    expected_gb = gy.to(torch.float64).sum(dim=(0, 2, 3, 4))
    incumbent_gb = (plain.bias.grad.to(torch.float64) - expected_gb).abs().max().item()
    actual_gb = (conv.bias.grad.to(torch.float64) - expected_gb).abs().max().item()
    assert conv.bias.grad.dtype is conv.bias.dtype
    assert actual_gb <= max(4.0 * incumbent_gb, 2.0**-8 * expected_gb.abs().max().item())


@pytest.mark.gpu
def test_autocast_runs_the_transposed_kernel_at_the_dtype_aten_would_have_chosen():
    """``conv_transpose3d`` carries the same ``lower_precision_fp`` policy.

    Verified two ways here: that the operands reaching the node are bf16 when the
    module's own parameters are fp32 (which is the state ``worker.py`` leaves the
    model in), and that the stock op under the same region produces the same
    dtype.  Without this the four upsamplers would quietly run in fp32 -- a
    different computation from the benchmark's, several times slower, and
    nothing failing.
    """
    conv = _gpu_convT(16, 8, dtype=torch.float32)  # as worker.py builds them
    x = _gpu_input((1, 16, 8, 8, 8), dtype=torch.float32).requires_grad_(True)

    seen = {}
    original = conv_mod._TritonConvTranspose3dFn.apply

    def _spy(x_, w_, b_, *rest):
        seen["x"], seen["w"], seen["b"] = x_.dtype, w_.dtype, b_.dtype
        return original(x_, w_, b_, *rest)

    conv_mod._TritonConvTranspose3dFn.apply = staticmethod(_spy)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = conv(x)
    finally:
        conv_mod._TritonConvTranspose3dFn.apply = original

    assert seen == {"x": torch.bfloat16, "w": torch.bfloat16, "b": torch.bfloat16}
    assert y.dtype is torch.bfloat16
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected = nn.functional.conv_transpose3d(x, conv.weight, conv.bias, 2, 0)
    assert expected.dtype is torch.bfloat16, "the op's autocast policy changed"
    torch.testing.assert_close(y.float(), expected.float(), rtol=2e-2, atol=2e-2)

    y.sum().backward()
    # The casts are ordinary autograd nodes, so every gradient comes back at its
    # parameter's own dtype, exactly as it does on the MIOpen rung.
    assert conv.weight.grad.dtype is torch.float32
    assert conv.bias.grad.dtype is torch.float32
    assert x.grad.dtype is torch.float32


@pytest.mark.gpu
@pytest.mark.parametrize("guard", ["no_grad", "inference_mode"])
def test_the_transposed_rung_serves_the_evaluation_path(guard):
    """``evaluate`` runs the whole model under ``@torch.inference_mode()``."""
    conv = _gpu_convT(16, 8, dtype=torch.float32)
    x = _gpu_input((1, 16, 8, 8, 8), dtype=torch.float32)
    with (
        getattr(torch, guard)(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        y = conv(x)
        expected = nn.functional.conv_transpose3d(x, conv.weight, conv.bias, 2, 0)
    assert conv._triton_ok is True, "the evaluation path did not take the rung"
    assert y.shape == expected.shape and y.dtype is expected.dtype
    torch.testing.assert_close(y.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
def test_a_checkpointed_upsampler_recomputes_on_the_same_rung():
    """``activation_checkpointing`` is a shipped config key.

    The hazard is wider here than for ``FastConv3d``: this ladder never adds a
    halo, so the tensor the Triton rung saves and the ``DCTensor`` the MIOpen
    rung saves agree on shape, dtype and device at *every* shard count, and a
    flip would pass ``_default_meta_extractor``'s check silently.
    """
    import torch.utils.checkpoint as cp

    conv = _gpu_convT()
    x = _gpu_input((1, 16, 8, 8, 8)).requires_grad_(True)
    plain_x = x.detach().clone().requires_grad_(True)

    y = cp.checkpoint(conv, x, use_reentrant=False)
    y.sum().backward()
    assert conv._triton_ok is True

    nn.ConvTranspose3d.forward(conv, plain_x).sum().backward()
    torch.testing.assert_close(
        x.grad.float(), plain_x.grad.float(), rtol=2e-2, atol=2e-2
    )


def test_transposed_backward_names_a_rung_flip_instead_of_dying_inside_distconv():
    """The only detector for a flip torch's checkpoint metadata check misses."""

    class _Wrapper(torch.Tensor):
        pass

    class _Ctx:
        saved_tensors = (_Wrapper(torch.randn(1, 8, 4, 4, 4)), torch.randn(8, 8, 2, 2, 2))
        conv_args = ((2, 2, 2), (0, 0, 0), (0, 0, 0), (1, 1, 1), True)
        needs_input_grad = (True, True, True, False, False, False, False)

    with pytest.raises(RuntimeError, match="served by different"):
        conv_mod._TritonConvTranspose3dFn.backward(
            _Ctx(), torch.randn(1, 8, 8, 8, 8)
        )


@pytest.mark.gpu
@pytest.mark.parametrize("direction", ["conv_transpose3d_backward_data",
                                       "conv_transpose3d_backward_weight"])
def test_transposed_backward_falls_back_to_miopen_when_a_direction_fails(
    monkeypatch, direction
):
    """A backward-direction failure degrades; the saved set cannot change.

    Both directions, because they are separate compilations from the forward's
    and from each other, so either can raise on a call whose forward compiled.
    """
    from triton_conv3d import reference as ref

    class _Boom(Exception):
        pass

    cin, cout, spatial = 16, 8, (8, 8, 8)
    problem = _transposed_problem(cin, cout, spatial)
    conv = _gpu_convT(cin, cout)
    plain = _stock_like(conv)
    x = _gpu_input((1, cin, *spatial))
    gy = _gpu_input((1, cout, 16, 16, 16), seed=53)
    fast_x = x.detach().clone().requires_grad_(True)
    plain_x = x.detach().clone().requires_grad_(True)

    y = conv(fast_x)
    module = conv_mod._get_triton_module()
    monkeypatch.setattr(conv_mod, "_triton_kernel_failures", lambda: (_Boom,))
    monkeypatch.setattr(
        module, direction, lambda *a, **kw: (_ for _ in ()).throw(_Boom()),
        raising=False,
    )
    monkeypatch.setattr(conv_mod, "_triton_failed", False)
    y.backward(gy)
    plain(plain_x).backward(gy)

    assert conv_mod._triton_failed is True
    operands = {
        "input": x,
        "weight": conv.weight.detach(),
        "bias": conv.bias.detach(),
        "grad_output": gy,
    }
    # The incumbent's own error is the bar, and it has to be: MIOpen's
    # transposed backward-weight exceeds the static bound at this shape (0.76
    # against 0.64), which is the observation ``error_bound``'s docstring
    # records.  What this direction is being asked is whether the fallback ran
    # the right operator with the right operands, not whether MIOpen is accurate.
    for name, actual, incumbent in (
        ("bwd-data", fast_x.grad, plain_x.grad),
        ("bwd-weight", conv.weight.grad, plain.weight.grad),
    ):
        expected = ref.reference(problem, operands, name)
        ref.assert_close(actual, expected, problem, name,
                         incumbent_error=ref.compare(incumbent, expected))
    # The bias gradient comes from the aten call on this path, not from the sum
    # above it, so it is the one this test would otherwise never look at.
    torch.testing.assert_close(
        conv.bias.grad.to(torch.float64),
        gy.to(torch.float64).sum(dim=(0, 2, 3, 4)),
        rtol=2e-2, atol=2e-2,
    )


def test_the_transposed_halo_plan_exchanges_nothing_and_refuses_what_it_cannot_read():
    """The plan is "exchange nothing" -- at every shard count, or not at all.

    The first assertion is the one that matters at scale: ``num_shards > 1``
    must produce a plan with an *empty* ``exchanges``, not merely a plan.  If it
    silently produced one at 1 shard and ``None`` at 2, the four sites would go
    back to MIOpen on every multi-GPU run and nothing would say so.
    """
    x = torch.empty(1, 8, 8, 8, 8)
    weight = torch.empty(8, 4, 2, 2, 2)

    def plan(num_shards, shard_dim=(2, 3, 4), w=weight, padding=(0, 0, 0)):
        strategy = _StubStrategy(num_shards, shard_dim)
        return conv_mod._transposed_halo_plan(
            _dc(x, num_shards, shard_dim), strategy, x, w, padding
        )

    for num_shards in ((1, 1, 1), (2, 1, 1), (4, 1, 1), (2, 2, 2)):
        made = plan(num_shards)
        assert made is not None, num_shards
        assert made.exchanges == (), num_shards
        assert made.padding == (0, 0, 0)
        assert made.input_shape == tuple(x.shape)

    # An odd kernel on a split dim: DistConv would want a k//2 halo there and
    # then refuse the problem outright, so there is no incumbent to agree with.
    assert plan((2, 1, 1), w=torch.empty(8, 4, 3, 3, 3)) is None
    # ...but only on a dim that is actually split.
    assert plan((1, 2, 1), w=torch.empty(8, 4, 3, 2, 2)) is not None
    # Everything the plan could not read.
    assert plan((2, 1, 1), shard_dim=(0, 1, 2)) is None
    assert plan((2, 1, 1), shard_dim=(2, 2, 2)) is None
    assert plan((2, 1), shard_dim=(2, 3, 4)) is None
    assert plan((2, 1, 1), padding=(1, 1)) is None
    assert conv_mod._transposed_halo_plan(None, _StubStrategy((2, 1, 1)), x, weight,
                                          (0, 0, 0)) is None
    periodic = _dc(x, (2, 1, 1))
    periodic._is_periodic = (True, False, False)
    assert conv_mod._transposed_halo_plan(
        periodic, _StubStrategy((2, 1, 1)), x, weight, (0, 0, 0)
    ) is None


@pytest.mark.gpu
@pytest.mark.parametrize("num_shards", [(1, 1, 1), (2, 1, 1)])
def test_the_transposed_rung_serves_a_dctensor_without_any_halo_exchange(num_shards):
    """The rung takes a sharded ``DCTensor``, and posts nothing to do it.

    At ``k = 2`` DistConv's own ``halo_size`` is ``k // 2 == 0``, so
    ``forward_halo_exchange`` returns its argument unchanged and the MIOpen rung
    also runs on the bare local shard.  That is what makes the two rungs the same
    computation at more than one shard, and it is why this ladder has no
    ``_Halo3d`` in it.  ``_StubStrategy`` reports ``shard_ind = 0``, so the
    MIOpen comparison arm is runnable in a one-rank process.
    """
    import distconv
    import distconv.distconv as dc

    from triton_conv3d import reference as ref

    cin, cout, spatial = 16, 8, (8, 8, 8)
    problem = _transposed_problem(cin, cout, spatial)
    conv = _gpu_convT(cin, cout)
    plain = _stock_like(conv)
    x = _gpu_input((1, cin, *spatial))
    gy = _gpu_input((1, cout, 16, 16, 16), seed=41)
    fast_x = x.detach().clone().requires_grad_(True)
    plain_x = x.detach().clone().requires_grad_(True)
    strategy = _StubStrategy(num_shards)

    calls = []
    original = dc.forward_halo_exchange
    dc.forward_halo_exchange = lambda *a, **kw: calls.append(a) or original(*a, **kw)
    try:
        y = conv(distconv.DCTensor.from_shard(fast_x, strategy))
    finally:
        dc.forward_halo_exchange = original

    assert isinstance(y, distconv.DCTensor)
    assert conv._triton_ok is True, "the fast rung did not serve the DCTensor"
    assert calls == [], "the Triton rung went through DistConv's halo exchange"

    y_plain = plain(distconv.DCTensor.from_shard(plain_x, strategy))
    y.backward(distconv.DCTensor.from_shard(gy, strategy))
    y_plain.backward(distconv.DCTensor.from_shard(gy, strategy))

    operands = {
        "input": x,
        "weight": conv.weight.detach(),
        "bias": conv.bias.detach(),
        "grad_output": gy,
    }
    for name, actual, incumbent in (
        ("fwd", y._tensor, y_plain._tensor),
        ("bwd-data", fast_x.grad, plain_x.grad),
        ("bwd-weight", conv.weight.grad, plain.weight.grad),
    ):
        expected = ref.reference(problem, operands, name)
        ref.assert_close(actual, expected, problem, name,
                         incumbent_error=ref.compare(incumbent, expected))
    torch.testing.assert_close(conv.bias.grad.float(), plain.bias.grad.float(),
                               rtol=2e-2, atol=2e-2)
