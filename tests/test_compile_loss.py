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

"""Compare compiled U-Net training objectives across train/evaluation calls."""

import copy
import json

import pytest
import torch
import torch.distributed as dist
from distconv import DCTensor, ParallelStrategy

from ScaFFold.unet import UNet, conv3d, group_norm
from ScaFFold.utils.evaluate import evaluate
from ScaFFold.utils.trainer import compute_training_loss_and_dice
from ScaFFold.worker import compile_unsharded_forward, wrap_model_ddp
from tests.helpers.compile_parity import assert_bf16_close, tensor_metrics
from tests.helpers.compile_runtime import compile_timeout


@pytest.fixture
def native_loss_compile(monkeypatch, compile_device):
    """Exercise native operators with independent compiler and random state."""
    monkeypatch.setattr(conv3d, "_triton_override", False)
    monkeypatch.setattr(group_norm, "_triton_override", False)
    monkeypatch.setattr(group_norm, "_compile_override", False)

    def unexpected_kernel(*args, **kwargs):
        pytest.fail("Native compiled loss entered a custom or regional kernel")

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


def _local(tensor):
    """Inspect values without changing the loss's original autograd route."""
    return tensor._tensor if isinstance(tensor, DCTensor) else tensor


def _exercise_compiled_loss_train_eval_train(
    backend, precision, checkpointing, *, fullgraph, device
):
    """Keep real DDP, weighted CE/Dice, and the validation logits API intact."""
    assert dist.get_world_size() == 1
    strategy = ParallelStrategy(
        num_shards=(1, 1, 1), shard_dim=(2, 3, 4), device_type=device.type
    )
    groups = list(dict.fromkeys(strategy.device_mesh.get_all_groups()))
    spatial_mesh = strategy.device_mesh[strategy.distconv_dim_names]
    bf16 = precision == "bf16"
    errors = []
    metrics = {}

    def compare(actual, expected, name):
        assert actual is not None and expected is not None, (
            f"{name}: missing tensor or gradient"
        )
        actual, expected = _local(actual), _local(expected)
        assert actual.dtype == expected.dtype, (
            f"{name}: dtype mismatch: {actual.dtype} != {expected.dtype}"
        )
        assert actual.device == expected.device, (
            f"{name}: device mismatch: {actual.device} != {expected.device}"
        )
        metrics[name] = tensor_metrics(actual, expected)
        assert metrics[name]["actual_finite"], f"{name}: nonfinite actual values"
        assert metrics[name]["reference_finite"], f"{name}: nonfinite reference values"
        try:
            if bf16:
                assert_bf16_close(actual, expected)
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        except AssertionError as error:
            errors.append((name, str(error)))

    try:
        reference_model = UNet(
            n_channels=3, n_classes=2, layers=1, group_norm_groups=8
        ).to(device=device, dtype=torch.float32, memory_format=torch.channels_last_3d)
        if checkpointing:
            reference_model.use_checkpointing()
        candidate_model = copy.deepcopy(reference_model)
        parameter_ids = tuple(id(p) for p in candidate_model.parameters())
        checkpoint_keys = tuple(candidate_model.state_dict())
        compile_unsharded_forward(
            candidate_model,
            fullgraph=fullgraph,
            backend=backend,
            compile_loss=True,
            parallel_strategy=strategy,
            amp_enabled=bf16,
        )
        reference = wrap_model_ddp(reference_model, device, strategy)
        candidate = wrap_model_ddp(candidate_model, device, strategy)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
        candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=1e-3)
        class_weights = torch.tensor([0.35, 1.65], dtype=torch.float32, device=device)

        for iteration in range(5):
            reference.train()
            candidate.train()
            reference.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            reference_input = (
                torch.randn(1, 3, 4, 4, 4, device=device)
                .contiguous(memory_format=torch.channels_last_3d)
                .requires_grad_(True)
            )
            candidate_input = reference_input.detach().clone().requires_grad_(True)
            labels = torch.randint(
                0, 2, (1, 1, 4, 4, 4), dtype=torch.int64, device=device
            )
            reference_dc = DCTensor.from_shard(reference_input, strategy)
            candidate_dc = DCTensor.from_shard(candidate_input, strategy)
            labels_dc = DCTensor.from_shard(labels, strategy)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                reference_logits = reference(reference_dc)
                reference_loss, reference_dice = compute_training_loss_and_dice(
                    reference_logits,
                    labels_dc.squeeze(1),
                    class_weights,
                    spatial_mesh,
                    strategy.num_shards,
                    device.type,
                )
                candidate_loss, candidate_dice = candidate(
                    candidate_dc, labels_dc, class_weights
                )
            prefix = f"train_{iteration}"
            assert candidate_loss.dtype == reference_loss.dtype == torch.float32
            assert candidate_dice.dtype == reference_dice.dtype == torch.float32
            assert not torch.is_autocast_enabled(device.type)
            compare(candidate_loss, reference_loss, f"{prefix}/loss")
            compare(candidate_dice, reference_dice, f"{prefix}/dice")
            reference_loss.backward()
            candidate_loss.backward()
            compare(candidate_input.grad, reference_input.grad, f"{prefix}/input_grad")
            reference_parameters = dict(reference.named_parameters())
            candidate_parameters = dict(candidate.named_parameters())
            assert candidate_parameters.keys() == reference_parameters.keys()
            for name, parameter in candidate_parameters.items():
                ref_parameter = reference_parameters[name]
                assert parameter.dtype == ref_parameter.dtype == torch.float32
                compare(parameter.grad, ref_parameter.grad, f"{prefix}/grad/{name}")
            reference_optimizer.step()
            candidate_optimizer.step()
            for name, parameter in candidate_parameters.items():
                compare(
                    parameter, reference_parameters[name], f"{prefix}/parameter/{name}"
                )

            if iteration == 0:
                # The production evaluator uses inference_mode, calls without
                # labels, computes metrics, and restores the model to train().
                batches = [
                    {
                        "image": torch.randn_like(reference_input, device="cpu"),
                        "mask": torch.randint(0, 2, (1, 4, 4, 4)),
                    }
                    for _ in range(2)
                ]
                validation_results = []
                validation_logits = []
                for network in (reference, candidate):
                    previous = {
                        name: (
                            parameter.detach().clone(),
                            parameter.grad.detach().clone(),
                        )
                        for name, parameter in network.named_parameters()
                    }
                    outputs = []

                    def observe_logits(module, args, output):
                        assert not module.training
                        assert torch.is_inference_mode_enabled()
                        assert isinstance(output, DCTensor)
                        assert output._parallel_strategy is strategy
                        assert output.shape == (1, 2, 4, 4, 4)
                        assert output.dtype == (
                            torch.bfloat16 if bf16 else torch.float32
                        )
                        assert not output.requires_grad
                        outputs.append(_local(output).clone())

                    handle = network.register_forward_hook(observe_logits)
                    try:
                        result = evaluate(
                            network,
                            batches,
                            device,
                            bf16,
                            False,
                            torch.nn.CrossEntropyLoss(weight=class_weights),
                            1,
                            strategy,
                        )
                    finally:
                        handle.remove()
                    assert network.training and network.module.training
                    assert result[3:] == (2, 2)
                    assert len(outputs) == 2
                    validation_results.append(result)
                    validation_logits.append(outputs)
                    for name, parameter in network.named_parameters():
                        old_parameter, old_gradient = previous[name]
                        torch.testing.assert_close(
                            parameter, old_parameter, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            parameter.grad, old_gradient, rtol=0, atol=0
                        )
                for index, (actual, expected) in enumerate(
                    zip(validation_logits[1], validation_logits[0])
                ):
                    compare(actual, expected, f"validation/logits_{index}")
                for index, name in enumerate(
                    ("hard_dice_sum", "loss_sum", "loss_mean")
                ):
                    compare(
                        torch.tensor(validation_results[1][index], dtype=torch.float64),
                        torch.tensor(validation_results[0][index], dtype=torch.float64),
                        f"validation/{name}",
                    )

        assert candidate.training and candidate.module.training
        assert tuple(id(p) for p in candidate.parameters()) == parameter_ids
        assert tuple(candidate.module.state_dict()) == checkpoint_keys
        assert tuple(candidate.state_dict()) == tuple(reference.state_dict())
        print(
            json.dumps(
                {
                    "backend": backend,
                    "device": str(device),
                    "precision": precision,
                    "checkpointing": checkpointing,
                    "fullgraph": fullgraph,
                    "training_steps": 5,
                    "metrics": metrics,
                }
            ),
            flush=True,
        )
        if errors:
            message = "\n".join(f"{name}: {error}" for name, error in errors)
            raise AssertionError(message)
    finally:
        for group in groups:
            if group is not dist.group.WORLD:
                dist.destroy_process_group(group)


@pytest.mark.slow
@pytest.mark.timeout(compile_timeout(600))
@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compiled_loss_train_eval_train(
    native_loss_compile,
    compile_device,
    compile_group_1rank,
    backend,
    precision,
    checkpointing,
    fullgraph,
):
    """Compare five independent updates, validation, and resumed training."""
    _exercise_compiled_loss_train_eval_train(
        backend, precision, checkpointing, fullgraph=fullgraph, device=compile_device
    )
