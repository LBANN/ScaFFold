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

"""Exercise real U-Net compilation with native PyTorch operators."""

import copy
import json

import pytest
import torch

from ScaFFold.unet import UNet, conv3d, group_norm
from tests.helpers.compile_parity import assert_bf16_close, tensor_metrics
from tests.helpers.compile_runtime import compile_timeout


def _compare_step(
    reference,
    candidate,
    reference_input,
    candidate_input,
    reference_output,
    candidate_output,
    precision,
):
    """Complete backward and collect every comparison before reporting failures."""
    errors = []
    metrics = {}
    bf16 = precision == "bf16"
    expected_dtype = torch.bfloat16 if bf16 else torch.float32

    def compare(actual, expected, name):
        if actual is None or expected is None:
            errors.append(f"{name}: missing tensor or gradient")
            return None
        try:
            result = tensor_metrics(actual, expected)
        except AssertionError as error:
            errors.append(f"{name}: {error}")
            return None
        if not (result["actual_finite"] and result["reference_finite"]):
            errors.append(f"{name}: nonfinite values: {result}")
            return result
        try:
            if bf16:
                assert_bf16_close(actual, expected)
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        except AssertionError as error:
            errors.append(f"{name}: {error}")
        return result

    for name, tensor in (
        ("eager output", reference_output),
        ("compiled output", candidate_output),
    ):
        if tensor.dtype != expected_dtype:
            errors.append(f"{name}: expected {expected_dtype}, got {tensor.dtype}")
    assert reference_input.dtype == candidate_input.dtype == torch.float32
    assert not torch.is_autocast_enabled(reference_input.device.type)
    metrics["output"] = compare(candidate_output, reference_output, "output")
    # Production computes losses in FP32, outside the model's autocast scope.
    reference_loss = reference_output.float().square().mean()
    candidate_loss = candidate_output.float().square().mean()
    assert reference_loss.dtype == candidate_loss.dtype == torch.float32
    metrics["loss"] = compare(candidate_loss, reference_loss, "loss")
    reference_loss.backward()
    candidate_loss.backward()
    metrics["input_gradient"] = compare(
        candidate_input.grad, reference_input.grad, "input gradient"
    )

    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    assert candidate_parameters.keys() == reference_parameters.keys()
    metrics["parameter_gradients"] = {}
    candidate_gradients, reference_gradients = [], []
    for name, parameter in candidate_parameters.items():
        reference_parameter = reference_parameters[name]
        assert parameter.dtype == reference_parameter.dtype == torch.float32, name
        result = compare(parameter.grad, reference_parameter.grad, name)
        metrics["parameter_gradients"][name] = result
        if result is not None:
            candidate_gradients.append(parameter.grad.detach().reshape(-1))
            reference_gradients.append(reference_parameter.grad.detach().reshape(-1))
    if candidate_gradients:
        metrics["aggregate_parameter_gradient"] = compare(
            torch.cat(candidate_gradients),
            torch.cat(reference_gradients),
            "aggregate parameter gradient",
        )
    return metrics, errors


@pytest.fixture
def native_compile(monkeypatch, compile_device):
    """Isolate routing and compiler state on the requested test device."""
    monkeypatch.setattr(conv3d, "_triton_override", False)
    monkeypatch.setattr(group_norm, "_triton_override", False)
    monkeypatch.setattr(group_norm, "_compile_override", False)

    def unexpected_kernel(*args, **kwargs):
        pytest.fail("Native U-Net compilation entered a custom or regional kernel")

    # Fail if execution enters a custom or regional kernel route.
    monkeypatch.setattr(conv3d.FastConv3d, "_triton_forward", unexpected_kernel)
    monkeypatch.setattr(
        conv3d.FastConvTranspose3d, "_triton_forward", unexpected_kernel
    )
    monkeypatch.setattr(group_norm.FastGroupNorm, "_triton_forward", unexpected_kernel)
    monkeypatch.setattr(
        group_norm.FastGroupNorm, "_compiled_forward", unexpected_kernel
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch._dynamo.reset()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1234)
            yield
    finally:
        torch._dynamo.reset()
        torch.set_num_threads(previous_threads)


@pytest.mark.slow
@pytest.mark.timeout(compile_timeout(300))
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_native_unet_fullgraph_forward_and_backward(
    native_compile, compile_device, backend, precision
):
    """Compile the complete topology and compare every gradient on new inputs."""
    from ScaFFold.worker import _compile_backend

    reference = UNet(n_channels=3, n_classes=2, layers=1, group_norm_groups=8).to(
        device=compile_device, dtype=torch.float32, memory_format=torch.channels_last_3d
    )
    candidate = copy.deepcopy(reference)
    checkpoint_keys = tuple(candidate.state_dict())
    candidate.forward = torch.compile(
        candidate.forward,
        backend=_compile_backend(backend, device_type=compile_device.type),
        fullgraph=True,
        dynamic=False,
    )

    errors = []
    for iteration in range(2):
        reference.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        reference_input = (
            torch.randn(1, 3, 4, 4, 4, dtype=torch.float32, device=compile_device)
            .contiguous(memory_format=torch.channels_last_3d)
            .requires_grad_(True)
        )
        candidate_input = reference_input.detach().clone().requires_grad_(True)

        with torch.autocast(
            compile_device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
        ):
            reference_output = reference(reference_input)
            candidate_output = candidate(candidate_input)
        metrics, step_errors = _compare_step(
            reference,
            candidate,
            reference_input,
            candidate_input,
            reference_output,
            candidate_output,
            precision,
        )
        errors.extend(f"input {iteration}: {error}" for error in step_errors)
        print(
            json.dumps(
                {
                    "backend": backend,
                    "device": str(compile_device),
                    "precision": precision,
                    "input": iteration,
                    "metrics": metrics,
                }
            ),
            flush=True,
        )

    assert tuple(candidate.state_dict()) == checkpoint_keys
    assert not errors, "\n".join(errors)


@pytest.mark.slow
@pytest.mark.timeout(compile_timeout(300))
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_native_compiled_adapter_preserves_dctensor_and_ddp_gradients(
    native_compile, compile_device, compile_group_1rank, backend, precision
):
    """Exercise the production compile boundary inside real singleton DDP."""
    import torch.distributed as dist
    from distconv import DCTensor, ParallelStrategy
    from distconv import distconv as distconv_ops

    from ScaFFold.worker import compile_unsharded_forward, wrap_model_ddp

    assert dist.get_world_size() == 1
    strategy = ParallelStrategy(
        num_shards=(1, 1, 1), shard_dim=(2, 3, 4), device_type=compile_device.type
    )
    groups = list(dict.fromkeys(strategy.device_mesh.get_all_groups()))
    try:
        reference_model = UNet(
            n_channels=3, n_classes=2, layers=1, group_norm_groups=8
        ).to(
            device=compile_device,
            dtype=torch.float32,
            memory_format=torch.channels_last_3d,
        )
        candidate_model = copy.deepcopy(reference_model)
        checkpoint_keys = tuple(candidate_model.state_dict())
        parameter_ids = tuple(id(p) for p in candidate_model.parameters())
        compile_unsharded_forward(candidate_model, fullgraph=True, backend=backend)
        reference = wrap_model_ddp(reference_model, compile_device, strategy)
        candidate = wrap_model_ddp(candidate_model, compile_device, strategy)

        assert candidate.module is candidate_model
        assert tuple(id(p) for p in candidate.parameters()) == parameter_ids
        assert tuple(candidate.state_dict()) == tuple(reference.state_dict())

        errors = []
        for iteration in range(2):
            reference.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            reference_input = (
                torch.randn(1, 3, 4, 4, 4, dtype=torch.float32, device=compile_device)
                .contiguous(memory_format=torch.channels_last_3d)
                .requires_grad_(True)
            )
            candidate_input = reference_input.detach().clone().requires_grad_(True)
            reference_dc = DCTensor.from_shard(reference_input, strategy)
            candidate_dc = DCTensor.from_shard(candidate_input, strategy)
            with torch.autocast(
                compile_device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
            ):
                reference_output_dc = reference(reference_dc)
                candidate_output_dc = candidate(candidate_dc)

            assert isinstance(reference_output_dc, DCTensor)
            assert isinstance(candidate_output_dc, DCTensor)
            assert reference_output_dc._parallel_strategy is strategy
            assert candidate_output_dc._parallel_strategy is strategy
            reference_output = distconv_ops._ToTensor.apply(reference_output_dc)
            candidate_output = distconv_ops._ToTensor.apply(candidate_output_dc)
            metrics, step_errors = _compare_step(
                reference,
                candidate,
                reference_input,
                candidate_input,
                reference_output,
                candidate_output,
                precision,
            )
            errors.extend(f"input {iteration}: {error}" for error in step_errors)
            print(
                json.dumps(
                    {
                        "backend": backend,
                        "device": str(compile_device),
                        "precision": precision,
                        "input": iteration,
                        "metrics": metrics,
                    }
                ),
                flush=True,
            )

        assert tuple(candidate.module.state_dict()) == checkpoint_keys
        assert tuple(id(p) for p in candidate.parameters()) == parameter_ids
        assert not errors, "\n".join(errors)
    finally:
        # DeviceMesh can reuse WORLD. The enclosing fixture owns that group.
        for group in groups:
            if group is not dist.group.WORLD:
                dist.destroy_process_group(group)
