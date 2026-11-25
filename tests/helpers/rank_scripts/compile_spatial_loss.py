"""Exercise compiled spatial CE/Dice through real DDP and validation."""

import copy
import json
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from distconv import DCTensor, ParallelStrategy

from ScaFFold.unet import UNet, conv3d, group_norm
from ScaFFold.utils.evaluate import evaluate
from ScaFFold.utils.trainer import compute_training_loss_and_dice
from ScaFFold.worker import _compile_backend, compile_spatial_forward, wrap_model_ddp
from tests.helpers.compile_parity import assert_bf16_close, tensor_metrics
from tests.helpers.compile_runtime import init_compile_process_group
from tests.helpers.rank_scripts.compile_spatial_2rank import reject_custom_kernel


def run(device):
    rank = dist.get_rank()
    bf16 = os.environ.get("SCAFFOLD_TEST_BF16", "0") == "1"
    checkpointing = os.environ.get("SCAFFOLD_TEST_CHECKPOINTING", "0") == "1"
    input_grad = os.environ.get("SCAFFOLD_TEST_INPUT_GRAD", "1") == "1"
    backend_name = os.environ.get("SCAFFOLD_TEST_COMPILE_BACKEND", "inductor")
    shards = tuple(map(int, os.environ.get("SCAFFOLD_TEST_SHARDS", "2,1,1").split(",")))
    strategy = ParallelStrategy(shards, (2, 3, 4), device_type=device.type)
    mesh = strategy.device_mesh[strategy.distconv_dim_names]
    backend = _compile_backend(
        backend_name, device_type=device.type, preserve_eager_precision=bf16
    )
    graphs = []

    def capture(graph, inputs):
        graphs.append(
            [
                str(node.target)
                for module in graph.modules()
                if isinstance(module, torch.fx.GraphModule)
                for node in module.graph.nodes
                if node.op == "call_function"
            ]
        )
        return backend(graph, inputs)

    torch.manual_seed(1234)
    reference_model = UNet(3, 2, layers=1).to(
        device=device, memory_format=torch.channels_last_3d
    )
    candidate_model = copy.deepcopy(reference_model)
    if checkpointing:
        candidate_model.use_checkpointing()
    compile_spatial_forward(
        candidate_model,
        strategy,
        backend=capture,
        compile_loss=True,
        amp_enabled=bf16,
    )
    reference = wrap_model_ddp(reference_model, device, strategy)
    candidate = wrap_model_ddp(candidate_model, device, strategy)
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=1e-3) for model in (reference, candidate)
    ]
    weights = torch.tensor([0.35, 1.65], device=device)
    errors, metrics = [], {}

    def compare(actual, expected, name):
        actual = actual._tensor if isinstance(actual, DCTensor) else actual
        expected = expected._tensor if isinstance(expected, DCTensor) else expected
        if actual is None or expected is None:
            errors.append(f"{name}: missing tensor/gradient")
            return
        metrics[name] = tensor_metrics(actual, expected)
        try:
            assert metrics[name]["actual_finite"] and metrics[name]["reference_finite"]
            if bf16:
                assert_bf16_close(actual, expected)
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        except AssertionError as error:
            errors.append(f"{name}: {error}")

    def synchronize_errors():
        all_errors = [None] * dist.get_world_size()
        dist.all_gather_object(all_errors, errors)
        assert not any(all_errors), json.dumps(
            {"errors": all_errors, "metrics": metrics}
        )

    def validate(iteration):
        generator = torch.Generator().manual_seed(900 + rank + 10 * iteration)
        batches = [
            {
                "image": torch.randn(1, 3, 4, 4, 4, generator=generator),
                "mask": torch.randint(0, 2, (1, 4, 4, 4), generator=generator),
            }
        ]
        results, logits = [], []
        for model in (reference, candidate):
            before = [
                (p.detach().clone(), None if p.grad is None else p.grad.clone())
                for p in model.parameters()
            ]
            outputs = []

            def observe(module, args, output):
                assert not module.training and torch.is_inference_mode_enabled()
                assert output.dtype == (torch.bfloat16 if bf16 else torch.float32)
                assert output._tensor.device == device
                outputs.append(output._tensor.clone())

            handle = model.register_forward_hook(observe)
            try:
                results.append(
                    evaluate(
                        model,
                        batches,
                        device,
                        bf16,
                        False,
                        torch.nn.CrossEntropyLoss(weight=weights),
                        1,
                        strategy,
                    )
                )
            finally:
                handle.remove()
            logits.append(outputs[0])
            assert (
                model.training
                and torch.is_grad_enabled()
                and not torch.is_inference_mode_enabled()
            )
            for parameter, (value, gradient) in zip(
                model.parameters(), before, strict=True
            ):
                torch.testing.assert_close(parameter, value, rtol=0, atol=0)
                if gradient is None:
                    assert parameter.grad is None
                else:
                    torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        compare(logits[1], logits[0], f"eval {iteration}/logits")
        for index, name in enumerate(("dice", "loss_sum", "loss_mean")):
            compare(
                torch.tensor(results[1][index]),
                torch.tensor(results[0][index]),
                f"eval {iteration}/{name}",
            )
        synchronize_errors()

    validate(0)
    first_graph_count = None
    for iteration in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(500 + 10 * iteration + rank)
        ref_input = (
            torch.randn(1, 3, 4, 4, 4, generator=generator)
            .to(device=device, memory_format=torch.channels_last_3d)
            .requires_grad_(input_grad)
        )
        cand_input = ref_input.detach().clone().requires_grad_(input_grad)
        labels = torch.randint(0, 2, (1, 1, 4, 4, 4), generator=generator).to(device)
        dc_labels = DCTensor.from_shard(labels, strategy)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            ref_logits = reference(DCTensor.from_shard(ref_input, strategy))
            ref_loss, ref_dice = compute_training_loss_and_dice(
                ref_logits, dc_labels.squeeze(1), weights, mesh, shards, device.type
            )
            cand_loss, cand_dice = candidate(
                DCTensor.from_shard(cand_input, strategy), dc_labels, weights
            )
        assert cand_loss.dtype == cand_dice.dtype == torch.float32
        assert cand_loss.device == cand_dice.device == device
        compare(cand_loss, ref_loss, f"train {iteration}/loss")
        compare(cand_dice, ref_dice, f"train {iteration}/dice")
        ref_loss.backward()
        cand_loss.backward()
        if input_grad:
            compare(cand_input.grad, ref_input.grad, f"train {iteration}/input_grad")
        else:
            assert cand_input.grad is None and ref_input.grad is None
        for (name, parameter), (ref_name, ref_parameter) in zip(
            candidate.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == ref_name
            compare(
                parameter.grad, ref_parameter.grad, f"train {iteration}/grad/{name}"
            )
        for optimizer in optimizers:
            optimizer.step()
        for (name, parameter), (_, ref_parameter) in zip(
            candidate.named_parameters(), reference.named_parameters(), strict=True
        ):
            compare(parameter, ref_parameter, f"train {iteration}/parameter/{name}")
        assert torch._C._distributed_c10d._get_work_registry_size() == 0
        synchronize_errors()
        if first_graph_count is None:
            first_graph_count = len(graphs)
        else:
            assert len(graphs) == first_graph_count, (
                "A repeated training/evaluation shape recompiled"
            )
        if iteration == 1:
            validate(1)
    assert graphs and any(
        "all_reduce" in target for graph in graphs for target in graph
    )
    print(
        json.dumps(
            {
                "rank": rank,
                "device": str(device),
                "distributed_backend": dist.get_backend(),
                "backend": backend_name,
                "bf16": bf16,
                "checkpointing": checkpointing,
                "graphs": len(graphs),
                "metrics": metrics,
            }
        ),
        flush=True,
    )
    print(f"RANK {rank} DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    conv3d.set_conv_triton_enabled(False)
    group_norm.set_triton_enabled(False)
    group_norm.set_compile_enabled(False)
    device = init_compile_process_group()
    try:
        with (
            patch.object(conv3d, "_get_triton_module", reject_custom_kernel),
            patch.object(conv3d.FastConv3d, "_triton_forward", reject_custom_kernel),
            patch.object(conv3d._Halo3d, "apply", reject_custom_kernel),
            patch.object(
                group_norm.FastGroupNorm, "_triton_forward", reject_custom_kernel
            ),
            patch.object(
                group_norm.FastGroupNorm, "_compiled_forward", reject_custom_kernel
            ),
        ):
            run(device)
    finally:
        dist.destroy_process_group()
