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

"""Tests for the channels-last Triton channel concatenation (``triton_cat``).

The op replaces ``torch.cat([a, b], dim=1)``, whose result is a *copy* rather
than an arithmetic expression, so unlike the GroupNorm kernel these are not
tolerance tests: every parity assertion here is **bitwise**, forward and
backward, including the mixed-dtype and narrowing cases.  A tolerance would
hide exactly the bugs this kernel can have -- an off-by-one in the channel
split, a lost tail row, a double rounding.

The other things being pinned down:

* the **dtype rule**.  ``consumer_dtype`` is what licenses emitting bf16 from a
  concatenation whose ``torch.cat`` result would be fp32; the tests check both
  that it is what the following convolution receives and that it never *widens*
  anything.
* the **fallback**.  ``is_supported`` must decline everything the kernel cannot
  serve physically, and ``cat_channels`` must then still answer -- so the CPU
  suite exercises the whole public API with no GPU and no Triton.
* **composition**: a ``DCTensor`` round trip with the autograd graph intact,
  ``torch.utils.checkpoint`` recompute, ``inference_mode``, and the
  ``trilinear`` branch of ``Up``.

CPU runs never touch Triton: the module defers ``import triton`` to the first
call that reaches a kernel, which ``test_import_does_not_pull_in_triton``
checks in a fresh interpreter.
"""

from __future__ import annotations

import itertools
import subprocess
import sys

import pytest
import torch

from ScaFFold.unet import triton_cat
from ScaFFold.unet.triton_cat import (
    cat_channels,
    consumer_dtype,
    is_supported,
    skip_concat,
)

CL = torch.channels_last_3d
DTYPES = (torch.float32, torch.bfloat16, torch.float16)

gpu = pytest.mark.gpu
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def cl_tensor(*shape, dtype=torch.float32, device="cpu", seed=0):
    """A deterministic channels-last-3d tensor."""
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(*shape, generator=generator)
        .to(device=device, dtype=dtype)
        .contiguous(memory_format=CL)
    )


# --------------------------------------------------------------------------- #
# The public API is total: it must work with no GPU and no Triton at all.
# --------------------------------------------------------------------------- #
def test_cpu_falls_back_and_matches_torch_cat_bitwise():
    """On CPU ``is_supported`` declines and ``cat_channels`` still answers."""
    a = cl_tensor(2, 5, 3, 4, 5, seed=1)
    b = cl_tensor(2, 3, 3, 4, 5, seed=2)
    assert not is_supported(a, b)
    got = cat_channels(a, b)
    assert torch.equal(got, torch.cat([a, b], dim=1))
    assert got.is_contiguous(memory_format=CL)


def test_cpu_fallback_honours_out_dtype_bitwise():
    """A narrowing ``out_dtype`` must round each value exactly once.

    ``torch.cat([a, b]).to(bf16)`` widens then narrows; the fallback narrows
    the inputs first.  Those agree bitwise because the promoted dtype is an
    exact widening of both, and that is the property the kernel relies on too.
    """
    a = cl_tensor(1, 4, 2, 3, 4, seed=3)
    b = cl_tensor(1, 4, 2, 3, 4, dtype=torch.bfloat16, seed=4)
    for out_dtype in DTYPES:
        got = cat_channels(a, b, out_dtype)
        assert got.dtype is out_dtype
        assert torch.equal(got, torch.cat([a, b], dim=1).to(out_dtype))


def test_cpu_fallback_backward_matches_torch_cat_bitwise():
    a = cl_tensor(1, 6, 2, 2, 2, seed=5).requires_grad_(True)
    b = cl_tensor(1, 2, 2, 2, 2, dtype=torch.bfloat16, seed=6).requires_grad_(True)
    grad = cl_tensor(1, 8, 2, 2, 2, seed=7)

    torch.cat([a, b], dim=1).backward(grad)
    ref = (a.grad.clone(), b.grad.clone())
    a.grad = b.grad = None

    cat_channels(a, b).backward(grad)
    assert torch.equal(a.grad, ref[0])
    assert torch.equal(b.grad, ref[1])
    assert a.grad.dtype is torch.float32
    assert b.grad.dtype is torch.bfloat16


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: (torch.randn(1, 2, 2, 2, 2), None), id="not-a-tensor"),
        pytest.param(lambda: (torch.randn(2, 2), torch.randn(2, 2)), id="rank-2"),
        pytest.param(
            lambda: (torch.randn(1, 2, 2, 2, 2).double(), torch.randn(1, 2, 2, 2, 2)),
            id="float64",
        ),
        pytest.param(
            lambda: (torch.randn(1, 2, 2, 2, 2), torch.randn(1, 2, 2, 2, 3)),
            id="spatial-mismatch",
        ),
        pytest.param(
            lambda: (torch.randn(0, 2, 2, 2, 2), torch.randn(0, 2, 2, 2, 2)),
            id="empty",
        ),
    ],
)
def test_is_supported_declines_what_the_kernel_cannot_serve(make):
    a, b = make()
    assert not is_supported(a, b)


def test_import_does_not_pull_in_triton():
    """Importing the module must not import Triton (it is deferred to first use)."""
    code = (
        "import sys; import ScaFFold.unet.triton_cat as m; "
        "assert 'triton' not in sys.modules, sorted(k for k in sys.modules "
        "if k.startswith('triton')); print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().endswith("ok")


# --------------------------------------------------------------------------- #
# The dtype rule.
# --------------------------------------------------------------------------- #
def test_consumer_dtype_outside_autocast_is_torch_cats_promotion():
    a = cl_tensor(1, 2, 2, 2, 2)
    b = cl_tensor(1, 2, 2, 2, 2, dtype=torch.bfloat16)
    assert consumer_dtype(a, b) is torch.promote_types(torch.float32, torch.bfloat16)
    assert consumer_dtype(a, a) is torch.float32
    assert consumer_dtype(b, b) is torch.bfloat16


@requires_cuda
@gpu
def test_consumer_dtype_under_autocast_is_the_autocast_dtype():
    """The mixed-dtype case the decoder actually presents.

    fp32 skip (a GroupNorm output, fp32 cast policy) + bf16 upsampled (a
    ConvTranspose3d output).  ``torch.cat`` would answer fp32; the convolution
    that consumes it narrows to bf16, so bf16 is the honest answer.
    """
    a = cl_tensor(1, 4, 2, 2, 2, device="cuda")
    b = cl_tensor(1, 4, 2, 2, 2, dtype=torch.bfloat16, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert consumer_dtype(a, b) is torch.bfloat16
        # never widens: two bf16 inputs stay bf16, and an fp16 autocast over
        # bf16 inputs must not silently change their width either
        assert consumer_dtype(b, b) is torch.bfloat16
    assert consumer_dtype(a, b) is torch.float32


@requires_cuda
@gpu
def test_skip_concat_is_bitwise_what_the_convolution_receives():
    """The load-bearing claim: the dtype shortcut changes no bits downstream."""
    a = cl_tensor(1, 8, 3, 4, 5, device="cuda")
    b = cl_tensor(1, 8, 3, 4, 5, dtype=torch.bfloat16, device="cuda", seed=11)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = skip_concat(a, b)
        # what the convolution would have been handed by the old chain:
        # torch.cat promotes to fp32, then autocast narrows for the conv.
        reference = torch.cat([a, b], dim=1).to(torch.bfloat16)
    assert got.dtype is torch.bfloat16
    assert torch.equal(got, reference)


# --------------------------------------------------------------------------- #
# GPU parity: bitwise, forward and backward.
# --------------------------------------------------------------------------- #
_SHAPES = [
    (1, 8, 8, 4, 4, 4),  # power-of-two channels, the UNet's case
    (1, 3, 5, 2, 3, 4),  # odd channel counts, C = 8
    (1, 5, 6, 2, 2, 2),  # C = 11, not a power of two
    (2, 4, 4, 2, 2, 2),  # batch > 1
    (1, 1, 1, 1, 1, 1),  # degenerate
]


@requires_cuda
@gpu
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(str(v) for v in s))
@pytest.mark.parametrize("dtypes", list(itertools.product(DTYPES, DTYPES)))
def test_forward_is_bitwise_torch_cat(shape, dtypes):
    n, ca, cb, d, h, w = shape
    da, db = dtypes
    a = cl_tensor(n, ca, d, h, w, dtype=da, device="cuda", seed=21)
    b = cl_tensor(n, cb, d, h, w, dtype=db, device="cuda", seed=22)
    assert is_supported(a, b)
    got = cat_channels(a, b)
    ref = torch.cat([a, b], dim=1)
    assert got.dtype is ref.dtype
    assert got.shape == ref.shape
    assert got.is_contiguous(memory_format=CL)
    assert torch.equal(got, ref)


@requires_cuda
@gpu
@pytest.mark.parametrize("out_dtype", DTYPES)
def test_forward_out_dtype_is_bitwise_cat_then_cast(out_dtype):
    a = cl_tensor(1, 6, 3, 4, 5, device="cuda", seed=23)
    b = cl_tensor(1, 10, 3, 4, 5, dtype=torch.bfloat16, device="cuda", seed=24)
    got = cat_channels(a, b, out_dtype)
    assert got.dtype is out_dtype
    assert torch.equal(got, torch.cat([a, b], dim=1).to(out_dtype))


@requires_cuda
@gpu
@pytest.mark.parametrize("dtypes", list(itertools.product(DTYPES, DTYPES)))
def test_backward_is_bitwise_torch_cat(dtypes):
    da, db = dtypes
    a = cl_tensor(1, 6, 2, 3, 4, dtype=da, device="cuda", seed=25).requires_grad_(True)
    b = cl_tensor(1, 10, 2, 3, 4, dtype=db, device="cuda", seed=26).requires_grad_(True)
    grad = cl_tensor(
        1, 16, 2, 3, 4, dtype=torch.promote_types(da, db), device="cuda", seed=27
    )

    torch.cat([a, b], dim=1).backward(grad)
    ref = (a.grad.clone(), b.grad.clone())
    a.grad = b.grad = None

    cat_channels(a, b).backward(grad)
    assert a.grad.dtype is da and b.grad.dtype is db
    assert a.grad.is_contiguous(memory_format=CL)
    assert b.grad.is_contiguous(memory_format=CL)
    assert torch.equal(a.grad, ref[0])
    assert torch.equal(b.grad, ref[1])


@requires_cuda
@gpu
def test_backward_with_only_one_side_requiring_grad():
    """The ``WANT_A``/``WANT_B`` kernel flags must not corrupt the other half."""
    a = cl_tensor(1, 6, 2, 2, 2, device="cuda", seed=28).requires_grad_(True)
    b = cl_tensor(1, 2, 2, 2, 2, device="cuda", seed=29)
    grad = cl_tensor(1, 8, 2, 2, 2, device="cuda", seed=30)
    cat_channels(a, b).backward(grad)
    assert b.grad is None
    assert torch.equal(a.grad, grad[:, :6])

    a2 = cl_tensor(1, 6, 2, 2, 2, device="cuda", seed=28)
    b2 = cl_tensor(1, 2, 2, 2, 2, device="cuda", seed=29).requires_grad_(True)
    cat_channels(a2, b2).backward(grad)
    assert a2.grad is None
    assert torch.equal(b2.grad, grad[:, 6:])


@requires_cuda
@gpu
def test_backward_accepts_a_non_channels_last_gradient():
    """Nothing guarantees the consumer hands back a channels-last cotangent."""
    a = cl_tensor(1, 6, 2, 2, 2, device="cuda", seed=31).requires_grad_(True)
    b = cl_tensor(1, 2, 2, 2, 2, device="cuda", seed=32).requires_grad_(True)
    grad = torch.randn(1, 8, 2, 2, 2, device="cuda")  # plain contiguous
    cat_channels(a, b).backward(grad)
    assert torch.equal(a.grad, grad[:, :6])
    assert torch.equal(b.grad, grad[:, 6:])


@requires_cuda
@gpu
@pytest.mark.parametrize("pad", [1, 2])
def test_backward_reads_a_halo_padded_gradient_in_place(pad):
    """The production cotangent: a narrowed view of a halo-padded tensor.

    DistConv reaches the convolution that consumes the concatenation through a
    halo exchange that materialises a ``(D+2, H+2, W+2)`` tensor, so what comes
    back here is a *narrow* of one -- channels-last per voxel and contiguous
    along W, but with a gap at every H and D boundary.  Relaying that out costs
    a whole extra full-resolution pass, which is exactly what made an earlier
    version of this op lose to ``torch.cat``; the kernel must read it in place.
    """
    a = cl_tensor(1, 6, 3, 4, 5, device="cuda", seed=51).requires_grad_(True)
    b = cl_tensor(1, 10, 3, 4, 5, dtype=torch.bfloat16, device="cuda", seed=52)
    b.requires_grad_(True)
    parent = cl_tensor(
        1, 16, 3 + 2 * pad, 4 + 2 * pad, 5 + 2 * pad, device="cuda", seed=53
    )
    grad = parent[:, :, pad:-pad, pad:-pad, pad:-pad]
    assert grad.shape == (1, 16, 3, 4, 5)
    assert not grad.is_contiguous(memory_format=CL)
    assert triton_cat._line_addressable(grad), "must take the in-place path"

    torch.cat([a, b], dim=1).backward(grad)
    ref = (a.grad.clone(), b.grad.clone())
    a.grad = b.grad = None

    cat_channels(a, b).backward(grad)
    assert torch.equal(a.grad, ref[0])
    assert torch.equal(b.grad, ref[1])
    assert a.grad.is_contiguous(memory_format=CL)
    assert b.grad.is_contiguous(memory_format=CL)


def test_line_addressable_accepts_narrows_and_declines_permutations():
    """The predicate that decides whether the backward can skip its relayout."""
    dense = cl_tensor(1, 8, 4, 5, 6)
    assert triton_cat._line_addressable(dense)
    parent = cl_tensor(1, 8, 6, 7, 8)
    assert triton_cat._line_addressable(parent[:, :, 1:-1, 1:-1, 1:-1])
    assert triton_cat._line_addressable(parent[:, :, :, :, 1:-1])
    # a plain contiguous (NCDHW) tensor has channels *outermost*
    assert not triton_cat._line_addressable(torch.randn(1, 8, 4, 5, 6))
    # a spatial permutation breaks the "W neighbours are one run apart" rule
    assert not triton_cat._line_addressable(dense.transpose(3, 4))
    # narrowing the channel axis breaks stride(4) == C
    assert not triton_cat._line_addressable(dense[:, :4])


@requires_cuda
@gpu
def test_non_channels_last_input_falls_back_to_torch_cat():
    a = torch.randn(1, 6, 2, 2, 2, device="cuda")  # contiguous, not CL
    b = cl_tensor(1, 2, 2, 2, 2, device="cuda", seed=33)
    assert not is_supported(a, b)
    assert torch.equal(cat_channels(a, b), torch.cat([a, b], dim=1))


@requires_cuda
@gpu
def test_repeated_calls_are_bitwise_identical():
    """A copy has no reduction order, and the tiling is a pure function of C."""
    a = cl_tensor(1, 6, 3, 4, 5, device="cuda", seed=34).requires_grad_(True)
    b = cl_tensor(1, 10, 3, 4, 5, device="cuda", seed=35).requires_grad_(True)
    grad = cl_tensor(1, 16, 3, 4, 5, device="cuda", seed=36)
    first = cat_channels(a, b).clone()
    cat_channels(a, b).backward(grad)
    ga, gb = a.grad.clone(), b.grad.clone()
    a.grad = b.grad = None
    second = cat_channels(a, b).clone()
    cat_channels(a, b).backward(grad)
    assert torch.equal(first, second)
    assert torch.equal(a.grad, ga) and torch.equal(b.grad, gb)


@requires_cuda
@gpu
def test_runs_on_a_non_current_device():
    """A Triton launch follows the *current* device; the guard must override it."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    a = cl_tensor(1, 6, 2, 2, 2, device="cuda:1", seed=37)
    b = cl_tensor(1, 2, 2, 2, 2, device="cuda:1", seed=38)
    with torch.cuda.device(0):
        got = cat_channels(a, b)
    assert got.device == a.device
    assert torch.equal(got, torch.cat([a, b], dim=1))


@requires_cuda
@gpu
def test_second_order_raises_rather_than_returning_garbage():
    """First order only, exactly like the Triton GroupNorm; it must fail loudly.

    The loss has to be *nonlinear* for this to bite.  A concatenation is a
    copy, so its own second derivative is identically zero: differentiating
    ``sum(cat(a, b))`` twice gives a cotangent that does not depend on ``a`` at
    all, and autograd correctly reports "does not require grad" without ever
    reaching this op.  With a square in the way the cotangent *does* depend on
    ``a``, the split has to be differentiated, and there is no formula for it.
    """
    a = cl_tensor(1, 4, 2, 2, 2, device="cuda", seed=39).requires_grad_(True)
    b = cl_tensor(1, 4, 2, 2, 2, device="cuda", seed=40).requires_grad_(True)
    out = cat_channels(a, b)
    (grad_a,) = torch.autograd.grad((out * out).sum(), a, create_graph=True)
    with pytest.raises(RuntimeError, match="no autograd formula was registered"):
        torch.autograd.grad(grad_a.sum(), a)


@requires_cuda
@gpu
def test_dctensor_round_trip_keeps_the_graph():
    """Production wraps activations in a DCTensor even at ``dc_num_shards=1``."""
    distconv = pytest.importorskip("distconv")
    if not torch.distributed.is_initialized():
        pytest.skip("needs an initialized process group")
    ps = distconv.ParallelStrategy(
        num_shards=(1, 1, 1), shard_dim=(2, 3, 4), device_type="cuda"
    )
    a = cl_tensor(1, 4, 2, 2, 2, device="cuda", seed=41).requires_grad_(True)
    b = cl_tensor(1, 4, 2, 2, 2, device="cuda", seed=42).requires_grad_(True)
    out = skip_concat(
        distconv.DCTensor.from_shard(a, ps), distconv.DCTensor.from_shard(b, ps)
    )
    assert isinstance(out, distconv.DCTensor)
    distconv.distconv._ToTensor.apply(out).pow(2).sum().backward()
    assert a.grad is not None and b.grad is not None


@requires_cuda
@gpu
def test_survives_activation_checkpoint_recompute():
    """The block's forward is replayed inside backward under checkpointing."""
    a = cl_tensor(1, 4, 2, 2, 2, device="cuda", seed=43).requires_grad_(True)
    b = cl_tensor(1, 4, 2, 2, 2, device="cuda", seed=44).requires_grad_(True)

    def block(x, y):
        return cat_channels(x, y) * 2.0

    ref = block(a, b)
    ref.pow(2).sum().backward()
    ga, gb = a.grad.clone(), b.grad.clone()
    a.grad = b.grad = None

    out = torch.utils.checkpoint.checkpoint(block, a, b, use_reentrant=False)
    out.pow(2).sum().backward()
    assert torch.equal(a.grad, ga)
    assert torch.equal(b.grad, gb)


@requires_cuda
@gpu
def test_inference_mode_and_no_grad():
    a = cl_tensor(1, 4, 2, 2, 2, device="cuda", seed=45)
    b = cl_tensor(1, 4, 2, 2, 2, device="cuda", seed=46)
    ref = torch.cat([a, b], dim=1)
    with torch.no_grad():
        assert torch.equal(cat_channels(a, b), ref)
    with torch.inference_mode():
        assert torch.equal(cat_channels(a, b), ref)


@requires_cuda
@gpu
def test_kernel_failure_is_tagged_so_a_caller_can_fall_back():
    """``CatKernelError`` must be what escapes when the launch itself breaks."""
    a = cl_tensor(1, 4, 2, 2, 2, device="cuda", seed=47)
    b = cl_tensor(1, 4, 2, 2, 2, device="cuda", seed=48)
    original = triton_cat._forward.__wrapped__

    def boom(*args, **kwargs):
        raise ValueError("simulated launch failure")

    triton_cat._forward.__wrapped__ = boom
    try:
        wrapped = triton_cat._tag_kernel_failures(boom)
        with pytest.raises(triton_cat.CatKernelError):
            wrapped(a, b, torch.float32)
    finally:
        triton_cat._forward.__wrapped__ = original
