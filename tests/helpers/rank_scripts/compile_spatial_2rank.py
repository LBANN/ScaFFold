"""Compare native spatial U-Net compilation with eager DistConv on CPU or GPU ranks.

The default scenario keeps the original two-rank, one-level regression. JSON
environment values select additional shard layouts and local (batch,D,H,W)
shapes; repeated shapes must reuse their compiled graphs. The production
adapter is always used, including real forward and backward halo exchanges.
"""

import copy
import json
import os
from math import prod
from unittest.mock import patch

import torch
import torch.distributed as dist
from distconv import DCTensor, ParallelStrategy
from distconv import distconv as distconv_ops

from ScaFFold.unet import UNet, conv3d, group_norm
from ScaFFold.utils.evaluate import evaluate
from ScaFFold.worker import _compile_backend, compile_spatial_forward, wrap_model_ddp
from tests.helpers.compile_parity import assert_bf16_close, tensor_metrics
from tests.helpers.compile_runtime import init_compile_process_group


class CustomKernelReached(BaseException):
    """Escape the routing layer's ordinary exception-based fallbacks."""


def reject_custom_kernel(*args, **kwargs):
    raise CustomKernelReached("Native compilation entered a custom Triton kernel")


def run(strategy, device):
    rank = dist.get_rank()
    backend_name = os.environ["SCAFFOLD_TEST_COMPILE_BACKEND"]
    bf16 = os.environ.get("SCAFFOLD_TEST_BF16", "0") == "1"
    transitions = os.environ.get("SCAFFOLD_TEST_TRANSITIONS", "0") == "1"
    eval_first = os.environ.get("SCAFFOLD_TEST_EVAL_FIRST", "0") == "1"
    align_parameters = os.environ.get("SCAFFOLD_TEST_ALIGN_PARAMETERS", "1") == "1"
    layers = int(os.environ.get("SCAFFOLD_TEST_LAYERS", "1"))
    local_shapes = [
        tuple(shape)
        for shape in json.loads(
            os.environ.get("SCAFFOLD_TEST_LOCAL_SHAPES", "[[1,4,4,4],[1,4,4,4]]")
        )
    ]
    if not local_shapes or any(
        len(shape) != 4 or any(size <= 0 for size in shape) for shape in local_shapes
    ):
        raise ValueError("Local shapes must be nonempty positive (batch,D,H,W) tuples")
    backend = _compile_backend(
        backend_name,
        device_type=device.type,
        preserve_eager_precision=True,
        preserve_native_group_norm=device.type == "cuda",
    )
    captured = []
    captured_phases = []
    captured_shapes = []
    phase = "train"
    phase_shape = local_shapes[0]

    def capture_graph(graph, inputs):
        targets = [
            (name, str(node.target))
            for name, module in graph.named_modules()
            if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes
            if node.op == "call_function"
        ]
        captured.append(targets)
        captured_phases.append(phase)
        captured_shapes.append(phase_shape)
        return backend(graph, inputs)

    torch.manual_seed(1234)
    reference_model = UNet(n_channels=3, n_classes=2, layers=layers).to(
        device=device, memory_format=torch.channels_last_3d
    )
    candidate_model = copy.deepcopy(reference_model)
    checkpoint_keys = tuple(candidate_model.state_dict())
    parameter_ids = tuple(id(p) for p in candidate_model.parameters())
    compile_spatial_forward(candidate_model, strategy, backend=capture_graph)
    reference = wrap_model_ddp(reference_model, device, strategy)
    candidate = wrap_model_ddp(candidate_model, device, strategy)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
    candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=1e-3)
    errors = []
    max_errors = {"output": 0.0, "input_gradient": 0.0, "parameter_gradient": 0.0}
    diagnostics = {}
    seen_graph_inputs = set()

    def check_graph_reuse(captures_before):
        key = (phase, phase_shape)
        captures_added = len(captured) - captures_before
        if key in seen_graph_inputs:
            if captures_added:
                errors.append(
                    f"{phase} shape {phase_shape} recompiled into "
                    f"{captures_added} graphs"
                )
        elif not captures_added:
            errors.append(f"{phase} shape {phase_shape} did not capture a graph")
        seen_graph_inputs.add(key)

    def fail_together(current_errors=None):
        gathered_errors = [None] * dist.get_world_size()
        dist.all_gather_object(
            gathered_errors, errors if current_errors is None else current_errors
        )
        if any(gathered_errors):
            raise AssertionError(
                json.dumps(
                    {"errors": gathered_errors, "local_metrics": diagnostics}, indent=2
                )
            )

    def compare(actual, expected, kind, name):
        if not isinstance(actual, torch.Tensor) or not isinstance(
            expected, torch.Tensor
        ):
            errors.append(f"{name}: missing tensor")
            return
        if actual.shape != expected.shape:
            errors.append(f"{name}: shapes differ: {actual.shape} != {expected.shape}")
            return
        metrics = tensor_metrics(actual, expected)
        diagnostics[name] = metrics
        max_errors[kind] = max(max_errors[kind], metrics["max_absolute"])
        try:
            assert metrics["actual_finite"] and metrics["reference_finite"]
            if bf16:
                assert_bf16_close(actual, expected)
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        except AssertionError as error:
            errors.append(f"{name}: {error}")

    def validate(round_index, local_shape):
        nonlocal phase, phase_shape
        phase = "eval"
        phase_shape = local_shape
        batch_size, *spatial_shape = local_shape
        generator = torch.Generator().manual_seed(700 + 10 * round_index + rank)
        batches = [
            {
                "image": torch.randn(
                    batch_size, 3, *spatial_shape, generator=generator
                ),
                "mask": torch.randint(0, 2, local_shape, generator=generator),
            }
            for _ in range(2)
        ]
        snapshots = [
            [
                (p.detach().clone(), None if p.grad is None else p.grad.clone())
                for p in model.parameters()
            ]
            for model in (reference, candidate)
        ]
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([0.7, 1.3], device=device)
        )
        results = []
        logits = [[], []]
        for model, recorded in zip((reference, candidate), logits, strict=True):

            def record_logits(module, args, output):
                recorded.append(output._tensor.detach().clone())

            handle = model.register_forward_hook(record_logits)
            captures_before = len(captured)
            try:
                results.append(
                    evaluate(
                        model,
                        batches,
                        device,
                        bf16,
                        False,
                        criterion,
                        n_categories=1,
                        parallel_strategy=strategy,
                    )
                )
            finally:
                handle.remove()
            if model is candidate:
                check_graph_reuse(captures_before)
        for batch_index, (actual, expected) in enumerate(
            zip(logits[1], logits[0], strict=True)
        ):
            compare(
                actual,
                expected,
                "output",
                f"validation logits {round_index}/{batch_index}",
            )
        compare(
            torch.tensor(results[1][:3]),
            torch.tensor(results[0][:3]),
            "output",
            f"validation metrics {round_index}",
        )
        if any(result[3:] != (2, 2 * batch_size) for result in results):
            errors.append(f"Validation processed wrong batch/sample counts: {results}")
        for model, snapshot in zip((reference, candidate), snapshots, strict=True):
            if not model.training:
                errors.append("Validation did not restore training mode")
            for parameter, (value, gradient) in zip(
                model.parameters(), snapshot, strict=True
            ):
                if parameter.is_inference() or not torch.equal(parameter, value):
                    errors.append(
                        "Validation changed a parameter or made it an inference tensor"
                    )
                if gradient is None:
                    if parameter.grad is not None:
                        errors.append("Validation created a gradient")
                elif parameter.grad is None or not torch.equal(
                    parameter.grad, gradient
                ):
                    errors.append("Validation changed an existing gradient")
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            errors.append("Validation leaked inference/gradient state")
        if torch._C._distributed_c10d._get_work_registry_size():
            errors.append("Validation left pending collective registrations")
        phase = "train"
        fail_together()

    if transitions and eval_first:
        validate(0, local_shapes[0])

    for iteration, local_shape in enumerate(local_shapes):
        phase_shape = local_shape
        batch_size, *spatial_shape = local_shape
        reference.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(200 + 10 * iteration + rank)
        reference_input = (
            torch.randn(batch_size, 3, *spatial_shape, generator=generator)
            .to(device=device, memory_format=torch.channels_last_3d)
            .requires_grad_(True)
        )
        candidate_input = reference_input.detach().clone().requires_grad_(True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            reference_output_dc = reference(
                DCTensor.from_shard(reference_input, strategy)
            )
            captures_before = len(captured)
            candidate_output_dc = candidate(
                DCTensor.from_shard(candidate_input, strategy)
            )
        check_graph_reuse(captures_before)
        boundary_errors = []
        if not isinstance(candidate_output_dc, DCTensor):
            boundary_errors.append("Compiled forward did not return a DCTensor")
        elif candidate_output_dc._parallel_strategy is not strategy:
            boundary_errors.append("Output lost its parallel strategy")
        elif candidate_output_dc._tensor.device != device:
            boundary_errors.append("Compiled forward used the wrong device")
        expected_dtype = torch.bfloat16 if bf16 else torch.float32
        if (
            reference_output_dc.dtype != expected_dtype
            or getattr(candidate_output_dc, "dtype", None) != expected_dtype
        ):
            boundary_errors.append(
                f"Expected {expected_dtype} outputs, got {reference_output_dc.dtype}/{getattr(candidate_output_dc, 'dtype', None)}"
            )
        if reference_output_dc.shape != getattr(candidate_output_dc, "shape", None):
            boundary_errors.append("Compiled forward changed the output shape")
        fail_together(boundary_errors)
        reference_output = distconv_ops._ToTensor.apply(reference_output_dc)
        candidate_output = distconv_ops._ToTensor.apply(candidate_output_dc)
        # GroupNorm is shard-local in both routes. The eager distributed U-Net,
        # rather than a whole-volume GroupNorm, defines the reference semantics.
        compare(candidate_output, reference_output, "output", f"output {iteration}")
        # DistConvDDP sums spatial contributions, so scale the local objectives
        # to the global mean before backward.
        spatial_ranks = strategy.total_num_shards
        (reference_output.float().square().mean() / spatial_ranks).backward()
        (candidate_output.float().square().mean() / spatial_ranks).backward()
        compare(
            candidate_input.grad,
            reference_input.grad,
            "input_gradient",
            f"input {iteration}",
        )
        reference_parameters = dict(reference_model.named_parameters())
        for name, parameter in candidate_model.named_parameters():
            compare(
                parameter.grad,
                reference_parameters[name].grad,
                "parameter_gradient",
                f"{name} {iteration}",
            )
        if all(p.grad is not None for p in candidate_model.parameters()) and all(
            p.grad is not None for p in reference_model.parameters()
        ):
            compare(
                torch.cat([p.grad.flatten() for p in candidate_model.parameters()]),
                torch.cat([p.grad.flatten() for p in reference_model.parameters()]),
                "parameter_gradient",
                f"all parameter gradients {iteration}",
            )
        pending = torch._C._distributed_c10d._get_work_registry_size()
        if pending:
            errors.append(
                f"Iteration {iteration} left {pending} pending collective registrations"
            )
        if transitions:
            reference_optimizer.step()
            candidate_optimizer.step()
            for name, parameter in candidate_model.named_parameters():
                compare(
                    parameter,
                    reference_parameters[name],
                    "parameter_gradient",
                    f"updated parameter {name} {iteration}",
                )
            if align_parameters:
                # Compare each real SGD update above, then isolate mode and
                # derivative parity at identical updated weights. Independent
                # optimizer trajectories are covered by the spatial loss tests;
                # set SCAFFOLD_TEST_ALIGN_PARAMETERS=0 for a model-only control.
                with torch.no_grad():
                    for name, parameter in candidate_model.named_parameters():
                        parameter.copy_(reference_parameters[name])
            validate(iteration + 1, local_shape)

    if tuple(candidate_model.state_dict()) != checkpoint_keys:
        errors.append("Compilation changed checkpoint keys")
    if tuple(id(p) for p in candidate_model.parameters()) != parameter_ids:
        errors.append("Compilation replaced parameters")
    # Dynamo captures the whole forward, but DDP may partition its backend
    # graphs at reducer buckets. Each shape/mode gets its own partitions; a
    # repeated shape must reuse them rather than compiling again.
    training_capture_count = captured_phases.count("train")
    evaluation_capture_count = captured_phases.count("eval")
    halo_target = "distconv.exchange_halos"
    # The initial block has two padded convolutions and each U-Net level adds
    # two in its encoder and two in its decoder. Each split axis communicates.
    expected_halos = (4 * layers + 2) * sum(n > 1 for n in strategy.num_shards)
    halo_counts = {}
    for graph_phase, shape in sorted(seen_graph_inputs):
        names = [
            name
            for targets, captured_phase, captured_shape in zip(
                captured, captured_phases, captured_shapes, strict=True
            )
            if (captured_phase, captured_shape) == (graph_phase, shape)
            for name, target in targets
            if halo_target in target
        ]
        if graph_phase == "train":
            counts = {
                "forward": sum("fwd_body" in name for name in names),
                "backward": sum("bwd_body" in name for name in names),
            }
        else:
            counts = {"inference": len(names)}
        halo_counts[f"{graph_phase}:{shape}"] = counts
        if any(count != expected_halos for count in counts.values()):
            errors.append(
                f"Expected {expected_halos} halo nodes per direction for "
                f"{graph_phase} shape {shape}, found {counts}: {names}"
            )
    fail_together()
    print(
        json.dumps(
            {
                "rank": rank,
                "device": str(device),
                "distributed_backend": dist.get_backend(),
                "backend": backend_name,
                "bf16": bf16,
                "transitions": transitions,
                "eval_first": eval_first,
                "aligned_parameters": align_parameters,
                "layers": layers,
                "local_shapes": local_shapes,
                "num_shards": strategy.num_shards,
                "world_size": dist.get_world_size(),
                "data_replicas": strategy.ddp_ranks,
                "graphs": len(captured),
                "training_graphs": training_capture_count,
                "evaluation_graphs": evaluation_capture_count,
                "expected_halos_per_graph_input": expected_halos,
                "halo_nodes": halo_counts,
                "forward_halos": sum(c.get("forward", 0) for c in halo_counts.values()),
                "backward_halos": sum(
                    c.get("backward", 0) for c in halo_counts.values()
                ),
                "inference_halos": sum(
                    c.get("inference", 0) for c in halo_counts.values()
                ),
                "pending_collectives": pending,
                "max_absolute_errors": max_errors,
                "metrics": diagnostics,
            }
        ),
        flush=True,
    )
    print(f"RANK {rank} DONE", flush=True)


def main():
    torch.set_num_threads(1)
    conv3d.set_conv_triton_enabled(False)
    group_norm.set_triton_enabled(False)
    group_norm.set_compile_enabled(False)
    device = init_compile_process_group()
    try:
        num_shards = tuple(
            json.loads(os.environ.get("SCAFFOLD_TEST_NUM_SHARDS", "[2,1,1]"))
        )
        shard_dims = tuple(
            json.loads(os.environ.get("SCAFFOLD_TEST_SHARD_DIMS", "[2,3,4]"))
        )
        if dist.get_world_size() % prod(num_shards):
            raise ValueError("World size must be divisible by the spatial shard count")
        strategy = ParallelStrategy(num_shards, shard_dims, device_type=device.type)
        # Native compilation must delegate halos to DistConv, without entering
        # ScaFFold's Triton convolution or its separate halo implementation.
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
            run(strategy, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
