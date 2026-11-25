"""Check the native halo adapter's autograd ownership and autocast boundary."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ScaFFold.unet import native_compile


def circular_pad(tensor, dim, halo):
    padding = [0] * (2 * (tensor.ndim - 2))
    offset = 2 * (tensor.ndim - 1 - dim)
    padding[offset : offset + 2] = [halo, halo]
    return F.pad(tensor, padding, mode="circular")


@pytest.fixture
def local_halo_helpers(monkeypatch):
    """Model neighboring shards with local circular faces, without a process group.

    The backward deliberately mutates its argument, matching DistConv's API.
    Circular padding's native autograd supplies an independent gradient oracle.
    """
    calls = []

    def forward(tensor, halo, strategy, dim_index):
        calls.append(("forward", dim_index, tensor.dtype))
        return circular_pad(tensor, strategy.shard_dim[dim_index], halo)

    def backward(grad, halo, strategy, dim_index):
        calls.append(("backward", dim_index, grad.dtype))
        dim = strategy.shard_dim[dim_index]
        inner = grad.narrow(dim, halo, grad.size(dim) - 2 * halo)
        inner.narrow(dim, 0, halo).add_(grad.narrow(dim, grad.size(dim) - halo, halo))
        inner.narrow(dim, inner.size(dim) - halo, halo).add_(grad.narrow(dim, 0, halo))
        return inner

    monkeypatch.setattr(native_compile.distconv_ops, "forward_halo_exchange", forward)
    monkeypatch.setattr(native_compile.distconv_ops, "backward_halo_exchange", backward)
    return calls


@pytest.mark.parametrize("dims", [(2,), (2, 4)])
@pytest.mark.parametrize("gradient_layout", ["dense", "channels_last", "expanded"])
def test_halo_backward_preserves_caller_gradient(
    local_halo_helpers, dims, gradient_layout
):
    strategy = SimpleNamespace(shard_dim=dims)
    plan = SimpleNamespace(
        strategy=strategy, exchanges=[(index, dim, 1) for index, dim in enumerate(dims)]
    )
    source = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
    actual_input = source.clone().requires_grad_()
    reference_input = source.clone().requires_grad_()
    actual = native_compile._DistConvHalo.apply(actual_input, plan)
    reference = reference_input
    for dim in dims:
        reference = circular_pad(reference, dim, 1)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    if gradient_layout == "expanded":
        upstream = torch.tensor(2.0).expand_as(actual)
    else:
        upstream = torch.arange(actual.numel(), dtype=actual.dtype).reshape(
            actual.shape
        )
        if gradient_layout == "channels_last":
            upstream = upstream.contiguous(memory_format=torch.channels_last_3d)
    saved_upstream = upstream.clone()
    saved_version = upstream._version
    actual_grad = torch.autograd.grad(actual, actual_input, upstream)[0]
    expected_grad = torch.autograd.grad(reference, reference_input, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(upstream, saved_upstream, rtol=0, atol=0)
    assert upstream._version == saved_version
    assert [
        index for name, index, _ in local_halo_helpers if name == "backward"
    ] == list(range(len(dims)))


@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("with_halo", [False, True])
def test_native_binding_preserves_parameters_and_casts_before_halo(
    local_halo_helpers, amp, with_halo
):
    torch.manual_seed(42)
    module = torch.nn.Conv3d(3, 4, 3, padding=1)
    original_parameters = tuple(module.parameters())
    original_keys = tuple(module.state_dict())
    plan = SimpleNamespace(
        strategy=SimpleNamespace(shard_dim=(2,)),
        exchanges=[(0, 2, 1)] if with_halo else [],
        padding=(0, 1, 1) if with_halo else (1, 1, 1),
    )
    native_compile.bind_native_convolution(module, plan)
    value = torch.randn(2, 3, 4, 5, 6).contiguous(memory_format=torch.channels_last_3d)
    actual_input = value.clone().requires_grad_()
    expected_input = value.clone().requires_grad_()
    dtype = torch.bfloat16 if amp else torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        actual = module(actual_input)
        expected_local = expected_input.to(dtype)
        if with_halo:
            expected_local = circular_pad(expected_local, 2, 1)
        expected = F.conv3d(
            expected_local, module.weight, module.bias, padding=plan.padding
        )
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.float().sum(), actual_input)[0]
    expected_grad = torch.autograd.grad(expected.float().sum(), expected_input)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert tuple(module.state_dict()) == original_keys
    assert all(
        current is saved
        for current, saved in zip(module.parameters(), original_parameters)
    )
    if with_halo:
        assert local_halo_helpers == [("forward", 0, dtype), ("backward", 0, dtype)]
    else:
        assert local_halo_helpers == []


def test_native_binding_rejects_unplanned_padding_mode():
    module = torch.nn.Conv3d(3, 4, 3, padding=1, padding_mode="reflect")
    with pytest.raises(ValueError, match="padding_mode='zeros'"):
        native_compile.bind_native_convolution(module, SimpleNamespace())
