"""Compare compiled spatial checkpoints with eager noncheckpointed DistConv."""

import copy
import json
import os
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.distributed as dist
from distconv import DCTensor, ParallelStrategy
from distconv import distconv as distconv_ops

from ScaFFold.unet import UNet, conv3d, group_norm
from ScaFFold.worker import _compile_backend, compile_spatial_forward, wrap_model_ddp
from tests.helpers.compile_parity import assert_bf16_close, tensor_metrics
from tests.helpers.compile_runtime import init_compile_process_group


class CustomKernelReached(BaseException):
    """Escape ordinary exception-based kernel fallbacks."""


def reject_custom_kernel(*args, **kwargs):
    raise CustomKernelReached("Native checkpointing entered a custom kernel")


def run(strategy, device):
    rank = dist.get_rank()
    backend_name = os.environ["SCAFFOLD_TEST_COMPILE_BACKEND"]
    bf16 = os.environ["SCAFFOLD_TEST_BF16"] == "1"
    backend = _compile_backend(
        backend_name, device_type=device.type, preserve_eager_precision=True
    )
    captured = []
    metrics = {}
    errors = []

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
        return backend(graph, inputs)

    def compare(actual, expected, name):
        if actual is None or expected is None:
            errors.append(f"{name}: missing tensor or gradient")
            return
        metrics[name] = tensor_metrics(actual, expected)
        if not (metrics[name]["actual_finite"] and metrics[name]["reference_finite"]):
            errors.append(f"{name}: nonfinite values")
            return
        try:
            if bf16:
                assert_bf16_close(actual, expected)
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        except AssertionError as error:
            errors.append(f"{name}: {error}")

    torch.manual_seed(1234)
    reference_model = UNet(n_channels=3, n_classes=2, layers=1).to(
        device=device, memory_format=torch.channels_last_3d
    )
    candidate_model = copy.deepcopy(reference_model)
    candidate_model.use_checkpointing()
    checkpoint_keys = tuple(candidate_model.state_dict())
    parameter_ids = tuple(id(p) for p in candidate_model.parameters())
    compile_spatial_forward(candidate_model, strategy, backend=capture_graph)
    reference = wrap_model_ddp(reference_model, device, strategy)
    candidate = wrap_model_ddp(candidate_model, device, strategy)
    first_capture_count = None
    runtime_counts = {}

    for iteration in range(2):
        reference.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(200 + 10 * iteration + rank)
        reference_input = (
            torch.randn(1, 3, 4, 4, 4, generator=generator)
            .to(device=device, memory_format=torch.channels_last_3d)
            .requires_grad_(True)
        )
        candidate_input = reference_input.detach().clone().requires_grad_(True)

        def profile_if_warm():
            # Compilation and eager route preparation are excluded: only the
            # second input measures real execution of the reused graphs.
            return (
                torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
                if iteration
                else nullcontext()
            )

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            reference_output_dc = reference(
                DCTensor.from_shard(reference_input, strategy)
            )
            with profile_if_warm() as forward_profile:
                candidate_output_dc = candidate(
                    DCTensor.from_shard(candidate_input, strategy)
                )
        assert isinstance(candidate_output_dc, DCTensor)
        assert candidate_output_dc._parallel_strategy is strategy
        assert candidate_output_dc._tensor.device == device
        assert (
            reference_output_dc.dtype
            == candidate_output_dc.dtype
            == (torch.bfloat16 if bf16 else torch.float32)
        )
        reference_output = distconv_ops._ToTensor.apply(reference_output_dc)
        candidate_output = distconv_ops._ToTensor.apply(candidate_output_dc)
        compare(candidate_output, reference_output, f"output {iteration}")
        # Spatial DDP sums parameter gradients; local means therefore need the
        # world-size divisor to represent the global objective.
        (reference_output.float().square().mean() / dist.get_world_size()).backward()
        with profile_if_warm() as backward_profile:
            (
                candidate_output.float().square().mean() / dist.get_world_size()
            ).backward()
        compare(
            candidate_input.grad, reference_input.grad, f"input gradient {iteration}"
        )
        reference_parameters = dict(reference_model.named_parameters())
        for name, parameter in candidate_model.named_parameters():
            compare(
                parameter.grad, reference_parameters[name].grad, f"{name} {iteration}"
            )
        if all(p.grad is not None for p in candidate_model.parameters()) and all(
            p.grad is not None for p in reference_model.parameters()
        ):
            compare(
                torch.cat([p.grad.flatten() for p in candidate_model.parameters()]),
                torch.cat([p.grad.flatten() for p in reference_model.parameters()]),
                f"all parameter gradients {iteration}",
            )
        if iteration == 0:
            first_capture_count = len(captured)
        else:
            for phase, profile in (
                ("forward", forward_profile),
                ("backward", backward_profile),
            ):
                runtime_counts[phase] = {
                    event.key: event.count
                    for event in profile.key_averages()
                    if "exchange_halos" in event.key
                    or "halo_wires" in event.key
                    or "batch_p2p_ops" in event.key
                    or event.key == "aten::convolution"
                }
        if torch._C._distributed_c10d._get_work_registry_size():
            errors.append(
                f"Iteration {iteration} left pending collective registrations"
            )

    checkpoint_tags = sum(
        "tag_activation_checkpoint" in target
        for targets in captured
        for target in targets
    )
    if checkpoint_tags != len(candidate_model.down_list) + len(candidate_model.up_list):
        errors.append(
            f"Expected four captured checkpoint blocks, got {checkpoint_tags}"
        )
    if not captured or len(captured) != first_capture_count:
        errors.append("The second input did not reuse the compiled checkpoint graphs")
    halo_nodes = sum(
        "distconv.exchange_halos" in target
        for targets in captured
        for target in targets
    )
    if not halo_nodes:
        errors.append("The compiled graphs did not capture DistConv halo exchanges")
    if any(
        "scaffold_conv3d.halo_wires" in target or "batch_p2p_ops" in target
        for targets in captured
        for target in targets
    ):
        errors.append("Native checkpointing captured a ScaFFold or raw P2P exchange")
    halo_event = "distconv::exchange_halos"
    # DistConv's opaque exchange is replayed normally during checkpoint
    # recomputation, then called again to return halo gradient contributions.
    for phase, expected in (("forward", 6), ("backward", 12)):
        if runtime_counts[phase].get(halo_event) != expected:
            errors.append(f"Wrong runtime {phase} halo count: {runtime_counts[phase]}")
    if runtime_counts["backward"].get("aten::convolution", 0) < 6:
        errors.append("Backward did not recompute the checkpointed convolutions")
    if not candidate_model.checkpointing or reference_model.checkpointing:
        errors.append("The candidate/reference checkpointing flags changed")
    if tuple(candidate_model.state_dict()) != checkpoint_keys:
        errors.append("Compilation changed checkpoint keys")
    if tuple(id(p) for p in candidate.parameters()) != parameter_ids:
        errors.append("Compilation replaced parameters")
    if any(getattr(m, "_triton_ok", False) for m in candidate_model.modules()):
        errors.append("Native compilation latched a custom convolution kernel")
    gathered_errors = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_errors, errors)
    report = {
        "rank": rank,
        "device": str(device),
        "distributed_backend": dist.get_backend(),
        "backend": backend_name,
        "halo_operator": halo_event,
        "bf16": bf16,
        "backend_graphs": len(captured),
        "checkpoint_tags": checkpoint_tags,
        "halo_nodes": halo_nodes,
        "runtime_counts": runtime_counts,
        "metrics": metrics,
        "errors": gathered_errors,
    }
    # Reports exceed a pipe's atomic-write size. Serialize rank output so the
    # numerical evidence remains machine-readable in the launcher log.
    for writer in range(dist.get_world_size()):
        if writer == rank:
            print(json.dumps(report), flush=True)
        dist.barrier()
    assert not any(gathered_errors), json.dumps(gathered_errors, indent=2)
    print(f"RANK {rank} DONE", flush=True)


def main():
    torch.set_num_threads(1)
    conv3d.set_conv_triton_enabled(False)
    group_norm.set_triton_enabled(False)
    group_norm.set_compile_enabled(False)
    device = init_compile_process_group()
    try:
        assert dist.get_world_size() == 2
        strategy = ParallelStrategy((2, 1, 1), (2, 3, 4), device_type=device.type)
        with (
            patch.object(conv3d, "_get_triton_module", reject_custom_kernel),
            patch.object(conv3d._Halo3d, "apply", reject_custom_kernel),
            patch.object(conv3d.FastConv3d, "_triton_forward", reject_custom_kernel),
            patch.object(
                conv3d.FastConvTranspose3d, "_triton_forward", reject_custom_kernel
            ),
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
