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

"""Keep compiled CPU normalization identical to native forward and backward."""

import copy

import pytest
import torch
from torch._dynamo.backends.registry import lookup_backend

from ScaFFold.unet import group_norm


@pytest.fixture
def native_cpu_normalization(monkeypatch):
    monkeypatch.setattr(group_norm, "_triton_override", False)
    monkeypatch.setattr(group_norm, "_compile_override", False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch._dynamo.reset()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(571)
            yield
    finally:
        torch._dynamo.reset()
        torch.set_num_threads(previous_threads)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "transposed"])
@pytest.mark.parametrize("activation,eps", [(None, 1e-3), ("relu", 1e-5)])
def test_compiled_cpu_group_norm_matches_native(
    native_cpu_normalization, dtype, affine, layout, activation, eps
):
    reference = group_norm.FastGroupNorm(
        4, 8, eps=eps, affine=affine, activation=activation
    )
    if affine:
        with torch.no_grad():
            reference.weight.uniform_(0.8, 1.2)
            reference.bias.uniform_(-0.2, 0.2)
    candidate = copy.deepcopy(reference)
    captured = []

    def backend(graph, inputs):
        captured.append(graph)
        return lookup_backend("inductor")(graph, inputs)

    candidate = torch.compile(candidate, backend=backend, fullgraph=True)
    value = torch.randn(2, 8, 3, 4, 5).to(dtype)
    if layout == "channels_last":
        value = value.contiguous(memory_format=torch.channels_last_3d)
    elif layout == "transposed":
        value = value.transpose(2, 3)
    reference_input = value.detach().clone().requires_grad_()
    candidate_input = value.detach().clone().requires_grad_()
    expected = reference(reference_input)
    actual = candidate(candidate_input)
    assert actual.dtype == expected.dtype == dtype
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cotangent = torch.randn_like(expected)
    expected.backward(cotangent)
    actual.backward(cotangent)
    torch.testing.assert_close(
        candidate_input.grad, reference_input.grad, rtol=0, atol=0
    )
    for candidate_parameter, reference_parameter in zip(
        candidate.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(
            candidate_parameter.grad, reference_parameter.grad, rtol=0, atol=0
        )
    assert any(
        node.target == torch.ops.scaffold_group_norm.cpu_forward.default
        for graph in captured
        for node in graph.graph.nodes
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("affine", [False, True])
def test_native_cpu_group_norm_operator_contract(
    native_cpu_normalization, dtype, affine
):
    value = (
        torch.randn(2, 8, 3, 4, 5)
        .to(dtype)
        .contiguous(memory_format=torch.channels_last_3d)
        .requires_grad_()
    )
    weight = torch.randn(8, requires_grad=True) if affine else None
    bias = torch.randn(8, requires_grad=True) if affine else None
    output, mean, rstd = group_norm._cpu_group_norm(value, weight, bias, 4, 1e-3)
    assert output.dtype == dtype
    assert mean.dtype == rstd.dtype == (torch.float32 if affine else dtype)
    assert not mean.requires_grad and not rstd.requires_grad
    result = torch.library.opcheck(
        group_norm._cpu_group_norm, (value, weight, bias, 4, 1e-3)
    )
    assert all(status == "SUCCESS" for status in result.values())


@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize(
    "shape,error_type,message",
    [
        ((8,), RuntimeError, "Expected at least 2 dimensions"),
        ((1, 8, 1, 1, 1), ValueError, "Expected more than 1 value per channel"),
    ],
)
def test_compiled_cpu_group_norm_preserves_input_validation(
    native_cpu_normalization, fullgraph, shape, error_type, message
):
    model = group_norm.FastGroupNorm(8, 8)
    value = torch.randn(shape)
    with pytest.raises(error_type, match=message):
        model(value)
    compiled = torch.compile(model, backend="eager", fullgraph=fullgraph)
    if fullgraph:
        # Full-graph Dynamo reports a propagated user-code exception as
        # Unsupported; it must still reject the eager-invalid input.
        with pytest.raises(torch._dynamo.exc.Unsupported, match="Observed exception"):
            compiled(value)
    else:
        with pytest.raises(error_type, match=message):
            compiled(value)
