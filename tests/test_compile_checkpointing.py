"""Real compiled activation checkpointing through the unsharded DCTensor API."""

import copy
import functools
import json

import pytest
import torch
import torch.distributed as dist
import torch.utils.checkpoint as checkpoint
from distconv import DCTensor, ParallelStrategy
from distconv import distconv as distconv_ops

from ScaFFold.unet import UNet
from ScaFFold.utils.evaluate import evaluate
from ScaFFold.worker import _compile_backend, compile_unsharded_forward, wrap_model_ddp
from tests.helpers.compile_parity import assert_bf16_close, tensor_metrics
from tests.helpers.compile_runtime import compile_timeout
from tests.test_compile_native import _compare_step
from tests.test_compile_native import native_compile as native_compile


@pytest.mark.slow
@pytest.mark.timeout(compile_timeout(300))
@pytest.mark.parametrize("fullgraph", [True, False], ids=["fullgraph", "blocks"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compiled_checkpointing_preserves_all_gradients(
    native_compile,
    compile_device,
    compile_group_1rank,
    monkeypatch,
    fullgraph,
    precision,
    backend,
):
    """Compare checkpointed compilation with an eager, noncheckpointed model."""
    assert dist.get_world_size() == 1
    strategy = ParallelStrategy(
        num_shards=(1, 1, 1), shard_dim=(2, 3, 4), device_type=compile_device.type
    )
    groups = list(dict.fromkeys(strategy.device_mesh.get_all_groups()))
    captured = []
    compile_backend = _compile_backend(backend, device_type=compile_device.type)
    checkpoint_events = {"frames": 0, "recomputes": 0}

    def capture_graph(graph, inputs):
        captured.append(
            [
                str(node.target)
                for module in graph.modules()
                if isinstance(module, torch.fx.GraphModule)
                for node in module.graph.nodes
                if node.op == "call_function"
            ]
        )
        return compile_backend(graph, inputs)

    if not fullgraph:
        # In block mode the real non-reentrant checkpoint frames surround the
        # compiled blocks eagerly. Observe their real recompute calls rather
        # than expecting a checkpoint tag inside those block graphs.
        original_frame_init = checkpoint._CheckpointFrame.__init__

        def observe_frame(frame, recompute_fn, *args, **kwargs):
            checkpoint_events["frames"] += 1

            @functools.wraps(recompute_fn)
            def observe_recompute(*args, **kwargs):
                checkpoint_events["recomputes"] += 1
                return recompute_fn(*args, **kwargs)

            return original_frame_init(frame, observe_recompute, *args, **kwargs)

        monkeypatch.setattr(checkpoint._CheckpointFrame, "__init__", observe_frame)

    try:
        reference_model = UNet(
            n_channels=3, n_classes=2, layers=1, group_norm_groups=8
        ).to(
            device=compile_device,
            dtype=torch.float32,
            memory_format=torch.channels_last_3d,
        )
        candidate_model = copy.deepcopy(reference_model)
        candidate_model.use_checkpointing()
        assert candidate_model.checkpointing and not reference_model.checkpointing
        checkpoint_keys = tuple(candidate_model.state_dict())
        parameter_ids = tuple(id(p) for p in candidate_model.parameters())
        block_count = len(candidate_model.down_list) + len(candidate_model.up_list)
        compile_unsharded_forward(
            candidate_model, fullgraph=fullgraph, backend=capture_graph
        )
        reference = wrap_model_ddp(reference_model, compile_device, strategy)
        candidate = wrap_model_ddp(candidate_model, compile_device, strategy)
        errors = []
        first_capture_count = None

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
            assert isinstance(candidate_output_dc, DCTensor)
            assert candidate_output_dc._parallel_strategy is strategy
            metrics, step_errors = _compare_step(
                reference,
                candidate,
                reference_input,
                candidate_input,
                distconv_ops._ToTensor.apply(reference_output_dc),
                distconv_ops._ToTensor.apply(candidate_output_dc),
                precision,
            )
            errors.extend(f"input {iteration}: {error}" for error in step_errors)
            print(
                json.dumps(
                    {
                        "backend": backend,
                        "device": str(compile_device),
                        "precision": precision,
                        "fullgraph": fullgraph,
                        "input": iteration,
                        "metrics": metrics,
                    }
                ),
                flush=True,
            )
            if iteration == 0:
                first_capture_count = len(captured)

        checkpoint_tags = sum(
            "tag_activation_checkpoint" in target
            for targets in captured
            for target in targets
        )
        assert captured, "No compiler backend graph was captured"
        assert len(captured) == first_capture_count, "The second input recompiled"
        if fullgraph:
            assert checkpoint_tags == block_count, captured
        else:
            assert checkpoint_events["frames"] == 2 * block_count, checkpoint_events
            assert checkpoint_events["recomputes"] == 2 * block_count, checkpoint_events
        assert candidate_model.checkpointing
        assert tuple(candidate_model.state_dict()) == checkpoint_keys
        assert tuple(id(p) for p in candidate.parameters()) == parameter_ids
        assert all(
            not getattr(m, "_triton_ok", False) for m in candidate_model.modules()
        )
        assert all(
            not getattr(m, "_compiled_ok", False) for m in candidate_model.modules()
        )
        print(
            json.dumps(
                {
                    "backend": backend,
                    "device": str(compile_device),
                    "precision": precision,
                    "fullgraph": fullgraph,
                    "backend_graphs": len(captured),
                    "checkpoint_tags": checkpoint_tags,
                    "checkpoint_events": checkpoint_events,
                }
            ),
            flush=True,
        )
        assert not errors, "\n".join(errors)
    finally:
        for group in groups:
            if group is not dist.group.WORLD:
                dist.destroy_process_group(group)


@pytest.mark.slow
@pytest.mark.timeout(compile_timeout(300))
@pytest.mark.parametrize("checkpointing", [False, True], ids=["direct", "checkpoint"])
@pytest.mark.parametrize("fullgraph", [True, False], ids=["fullgraph", "blocks"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_unsharded_evaluate_train_evaluate_train(
    native_compile,
    compile_device,
    compile_group_1rank,
    checkpointing,
    fullgraph,
    precision,
    backend,
):
    """Cold validation and subsequent validation must preserve training graphs."""
    assert dist.get_world_size() == 1
    strategy = ParallelStrategy((1, 1, 1), (2, 3, 4), device_type=compile_device.type)
    groups = list(dict.fromkeys(strategy.device_mesh.get_all_groups()))
    compile_backend = _compile_backend(backend, device_type=compile_device.type)
    captured_phases = []
    phase = "eval"
    errors = []
    diagnostics = {}
    bf16 = precision == "bf16"
    expected_dtype = torch.bfloat16 if bf16 else torch.float32

    def capture_graph(graph, inputs):
        captured_phases.append(phase)
        return compile_backend(graph, inputs)

    def compare(actual, expected, name):
        metrics = tensor_metrics(actual, expected)
        diagnostics[name] = metrics
        assert metrics["actual_finite"] and metrics["reference_finite"], name
        try:
            if bf16:
                assert_bf16_close(actual, expected)
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        except AssertionError as error:
            errors.append(f"{name}: {error}")

    try:
        reference_model = UNet(
            n_channels=3, n_classes=2, layers=1, group_norm_groups=8
        ).to(
            device=compile_device,
            dtype=torch.float32,
            memory_format=torch.channels_last_3d,
        )
        candidate_model = copy.deepcopy(reference_model)
        if checkpointing:
            candidate_model.use_checkpointing()
        checkpoint_keys = tuple(candidate_model.state_dict())
        parameter_ids = tuple(id(p) for p in candidate_model.parameters())
        compile_unsharded_forward(
            candidate_model, fullgraph=fullgraph, backend=capture_graph
        )
        reference = wrap_model_ddp(reference_model, compile_device, strategy)
        candidate = wrap_model_ddp(candidate_model, compile_device, strategy)
        evaluation_capture_count = None
        training_capture_count = None

        def validate(round_index):
            nonlocal phase, evaluation_capture_count
            phase = "eval"
            generator = torch.Generator().manual_seed(700 + 10 * round_index)
            batches = [
                {
                    "image": torch.randn(1, 3, 4, 4, 4, generator=generator),
                    "mask": torch.randint(0, 2, (1, 4, 4, 4), generator=generator),
                }
                for _ in range(2)
            ]
            outputs = []
            results = []
            for network in (reference, candidate):
                snapshot = [
                    (
                        p.detach().clone(),
                        None if p.grad is None else p.grad.detach().clone(),
                    )
                    for p in network.parameters()
                ]
                logits = []

                def observe_logits(module, args, output):
                    assert not module.training
                    assert torch.is_inference_mode_enabled()
                    assert isinstance(output, DCTensor)
                    assert output._parallel_strategy is strategy
                    assert output.shape == (1, 2, 4, 4, 4)
                    assert output.dtype == expected_dtype
                    assert not output.requires_grad
                    logits.append(output._tensor.clone())

                handle = network.register_forward_hook(observe_logits)
                try:
                    result = evaluate(
                        network,
                        batches,
                        compile_device,
                        bf16,
                        False,
                        torch.nn.CrossEntropyLoss(
                            weight=torch.tensor([0.7, 1.3], device=compile_device)
                        ),
                        n_categories=1,
                        parallel_strategy=strategy,
                    )
                finally:
                    handle.remove()
                assert result[3:] == (2, 2)
                assert len(logits) == 2
                assert all(module.training for module in network.modules())
                assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
                assert not torch.is_autocast_enabled(compile_device.type)
                for parameter, (value, gradient) in zip(
                    network.parameters(), snapshot, strict=True
                ):
                    assert not parameter.is_inference()
                    torch.testing.assert_close(parameter, value, rtol=0, atol=0)
                    if gradient is None:
                        assert parameter.grad is None
                    else:
                        torch.testing.assert_close(
                            parameter.grad, gradient, rtol=0, atol=0
                        )
                outputs.append(logits)
                results.append(result)
            for index, (actual, expected) in enumerate(
                zip(outputs[1], outputs[0], strict=True)
            ):
                compare(actual, expected, f"validation {round_index} logits {index}")
            for index, name in enumerate(("hard dice", "loss sum", "loss mean")):
                compare(
                    torch.tensor(results[1][index], dtype=torch.float64),
                    torch.tensor(results[0][index], dtype=torch.float64),
                    f"validation {round_index} {name}",
                )
            captures = captured_phases.count("eval")
            assert captures > 0, "Validation did not reach the compiler backend"
            if evaluation_capture_count is None:
                evaluation_capture_count = captures
            else:
                assert captures == evaluation_capture_count, "Validation recompiled"
            phase = "train"

        # Compile inference first: a successful prior training call must not be
        # needed to establish a route or to keep parameters usable by autograd.
        validate(0)
        for iteration in range(2):
            reference.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            reference_input = (
                torch.randn(1, 3, 4, 4, 4, device=compile_device)
                .contiguous(memory_format=torch.channels_last_3d)
                .requires_grad_(True)
            )
            candidate_input = reference_input.detach().clone().requires_grad_(True)
            with torch.autocast(
                compile_device.type, dtype=torch.bfloat16, enabled=bf16
            ):
                reference_output = reference(
                    DCTensor.from_shard(reference_input, strategy)
                )
                candidate_output = candidate(
                    DCTensor.from_shard(candidate_input, strategy)
                )
            assert isinstance(candidate_output, DCTensor)
            assert candidate_output._parallel_strategy is strategy
            metrics, step_errors = _compare_step(
                reference,
                candidate,
                reference_input,
                candidate_input,
                distconv_ops._ToTensor.apply(reference_output),
                distconv_ops._ToTensor.apply(candidate_output),
                precision,
            )
            diagnostics[f"training {iteration}"] = metrics
            errors.extend(f"training {iteration}: {error}" for error in step_errors)
            captures = captured_phases.count("train")
            assert captures > 0, "Training did not reach the compiler backend"
            if training_capture_count is None:
                training_capture_count = captures
                validate(1)
            else:
                assert captures == training_capture_count, "Training recompiled"

        assert candidate_model.checkpointing == checkpointing
        assert tuple(candidate_model.state_dict()) == checkpoint_keys
        assert tuple(id(p) for p in candidate.parameters()) == parameter_ids
        assert all(
            not getattr(m, "_triton_ok", False) for m in candidate_model.modules()
        )
        assert all(
            not getattr(m, "_compiled_ok", False) for m in candidate_model.modules()
        )
        print(
            json.dumps(
                {
                    "backend": backend,
                    "device": str(compile_device),
                    "precision": precision,
                    "fullgraph": fullgraph,
                    "checkpointing": checkpointing,
                    "evaluation_graphs": evaluation_capture_count,
                    "training_graphs": training_capture_count,
                    "metrics": diagnostics,
                }
            ),
            flush=True,
        )
        assert not errors, "\n".join(errors)
    finally:
        for group in groups:
            if group is not dist.group.WORLD:
                dist.destroy_process_group(group)
