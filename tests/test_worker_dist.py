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

"""Worker and distributed-initialization tests.

Covers the one-rank ("singleton") worker path, DDP device selection under
per-rank GPU masking, launcher environment detection, and the ordering of
device binding relative to process-group initialization.
"""

import copy
import logging
from types import SimpleNamespace
from unittest.mock import ANY

import pytest
import torch
from distconv import DCTensor, ParallelStrategy
from distconv import distconv as distconv_ops

import ScaFFold.utils.distributed as distributed_mod
import ScaFFold.worker as worker_mod
from ScaFFold.unet import conv3d, group_norm
from ScaFFold.utils.evaluate import evaluate
from ScaFFold.utils.trainer import compute_training_loss_and_dice

_TEST_LOG = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def restore_execution_routing(monkeypatch):
    """Worker setup changes process-wide preferences; isolate each test."""
    for module, names in (
        (conv3d, ("_triton_override", "_triton_failed")),
        (
            group_norm,
            (
                "_triton_override",
                "_triton_failed",
                "_compile_override",
                "_compile_failed",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))


_KERNEL_ENV_VARS = (
    "SCAFFOLD_CONV_TRITON",
    "SCAFFOLD_GROUPNORM_TRITON",
    "SCAFFOLD_GROUPNORM_COMPILE",
)


def _kernel_preferences():
    return (
        conv3d._triton_override,
        group_norm._triton_override,
        group_norm._compile_override,
    )


@pytest.mark.parametrize("mode", ["native-eager", "native-compiled"])
@pytest.mark.parametrize("env_value", [None, "0", "1"])
def test_native_execution_modes_disable_all_custom_routes(monkeypatch, mode, env_value):
    for name in _KERNEL_ENV_VARS:
        if env_value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, env_value)
    conv3d.set_conv_triton_enabled(True)
    group_norm.set_triton_enabled(True)
    group_norm.set_compile_enabled(True)

    worker_mod.configure_execution_mode(mode)

    assert _kernel_preferences() == (False, False, False)


@pytest.mark.parametrize("mode", ["triton-eager", "triton-compiled", "custom"])
@pytest.mark.parametrize(
    "env_values, expected",
    [
        ((None, None, None), (None, None, None)),
        (("0", "0", "0"), (False, False, False)),
        (("1", "1", "1"), (True, True, True)),
        (("1", "0", "1"), (True, False, True)),
    ],
)
def test_execution_mode_restores_environment_after_native_run(
    monkeypatch, mode, env_values, expected
):
    for name, value in zip(_KERNEL_ENV_VARS, env_values, strict=True):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    worker_mod.configure_execution_mode("native-eager")
    conv3d._triton_failed = True
    group_norm._triton_failed = True
    group_norm._compile_failed = True

    worker_mod.configure_execution_mode(mode)

    assert _kernel_preferences() == expected
    # Only an explicit environment opt-in retries a failed kernel. Automatic
    # selection must preserve the existing fallback latch and hardware guard.
    assert (
        conv3d._triton_failed,
        group_norm._triton_failed,
        group_norm._compile_failed,
    ) == tuple(value is not True for value in expected)


def test_triton_compiled_execution_mode_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=worker_mod.__name__):
        worker_mod.configure_execution_mode("triton-compiled")
    assert "triton-compiled is experimental" in caplog.text


@pytest.mark.parametrize("mode", ["native", "", None])
def test_invalid_execution_mode_leaves_preferences_unchanged(mode):
    conv3d.set_conv_triton_enabled(True)
    group_norm.set_triton_enabled(False)
    group_norm.set_compile_enabled(True)
    before = _kernel_preferences()
    with pytest.raises(ValueError, match="execution_mode must be one of"):
        worker_mod.configure_execution_mode(mode)
    assert _kernel_preferences() == before


# ---------------------------------------------------------------------------
# Launcher environment detection
# ---------------------------------------------------------------------------

_ENV_CASES = [
    # (env, expected world_rank, world_size, local_rank)
    ({"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "1"}, 3, 8, 1),
    ({"PMI_RANK": "2", "PMI_SIZE": "4", "PMI_LOCAL_RANK": "0"}, 2, 4, 0),
    ({"PALS_RANKID": "5", "PALS_NRANKS": "6", "PALS_LOCAL_RANKID": "1"}, 5, 6, 1),
    (
        {
            "OMPI_COMM_WORLD_RANK": "1",
            "OMPI_COMM_WORLD_SIZE": "2",
            "OMPI_COMM_WORLD_LOCAL_RANK": "1",
        },
        1,
        2,
        1,
    ),
    ({"SLURM_PROCID": "7", "SLURM_NTASKS": "16", "SLURM_LOCALID": "3"}, 7, 16, 3),
]

_LAUNCHER_VARS = [
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "PMI_RANK",
    "PMI_SIZE",
    "PMI_LOCAL_RANK",
    "PALS_RANKID",
    "PALS_NRANKS",
    "PALS_LOCAL_RANKID",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "FLUX_TASK_RANK",
    "FLUX_JOB_SIZE",
    "FLUX_TASK_LOCAL_ID",
]


def test_env_detection_matrix(monkeypatch):
    """Each launcher's env vars yield consistent world/local rank and size."""
    for env, want_rank, want_size, want_local in _ENV_CASES:
        for var in _LAUNCHER_VARS:
            monkeypatch.delenv(var, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        assert distributed_mod.get_world_rank() == want_rank, env
        assert distributed_mod.get_world_size() == want_size, env
        assert distributed_mod.get_local_rank() == want_local, env


# ---------------------------------------------------------------------------
# Device binding order and dynamic backend in initialize_dist
# ---------------------------------------------------------------------------


def _record_initialize_dist(monkeypatch):
    """Run initialize_dist with recorders; return the ordered call log."""
    calls = []
    monkeypatch.setattr(
        distributed_mod,
        "get_device",
        lambda: calls.append("get_device") or torch.device("cpu"),
    )
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kw: calls.append(("init_process_group", kw.get("backend"))),
    )
    monkeypatch.setattr(
        torch.distributed, "barrier", lambda *a, **kw: calls.append("barrier")
    )
    distributed_mod.initialize_dist(_TEST_LOG, rendezvous="env")
    return calls


def test_device_bound_before_first_collective(monkeypatch):
    """The compute device is selected/bound before any collective runs."""
    calls = _record_initialize_dist(monkeypatch)
    names = [c if isinstance(c, str) else c[0] for c in calls]
    assert names.index("get_device") < names.index("init_process_group")
    assert names.index("init_process_group") < names.index("barrier")


def test_backend_dynamic(monkeypatch):
    """Without CUDA the process group uses gloo instead of hardcoded nccl."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = _record_initialize_dist(monkeypatch)
    backends = [c[1] for c in calls if isinstance(c, tuple)]
    assert backends == ["gloo"]


# ---------------------------------------------------------------------------
# DDP wrap device selection (masked-GPU correctness)
# ---------------------------------------------------------------------------


def test_ddp_wrap_uses_get_device(monkeypatch):
    """The DDP wrapper pins device_ids to the selected device, not the local rank."""
    recorded = {}

    def fake_ddp(model, parallel_strategy=None, **kwargs):
        recorded.update(kwargs)
        return model

    monkeypatch.setattr(worker_mod, "DistConvDDP", fake_ddp)
    # Emulate per-rank masking: local rank 1 but the (single) visible device
    # is index 0 — get_device() returns cuda:0 there, and the wrapper must
    # follow it rather than addressing device index 1.
    device = torch.device("cuda:0")
    model = torch.nn.Linear(2, 2)
    worker_mod.wrap_model_ddp(model, device, ps=None)
    assert recorded["device_ids"] == [0]
    assert recorded["output_device"] == 0


def test_ddp_wrap_cpu_uses_none(monkeypatch):
    """On CPU (gloo) DDP requires device_ids/output_device of None."""
    recorded = {}

    def fake_ddp(model, parallel_strategy=None, **kwargs):
        recorded.update(kwargs)
        return model

    monkeypatch.setattr(worker_mod, "DistConvDDP", fake_ddp)
    worker_mod.wrap_model_ddp(torch.nn.Linear(2, 2), torch.device("cpu"), ps=None)
    assert recorded["device_ids"] is None
    assert recorded["output_device"] is None


# ---------------------------------------------------------------------------
# One-rank (singleton) worker path
# ---------------------------------------------------------------------------


def _run_singleton_worker(
    monkeypatch,
    tiny_config,
    tiny_dataset,
    *,
    port,
    config_overrides=None,
    on_train=None,
):
    """Run ``worker.main`` as a one-rank gloo job on CPU; return (rc, trainer).

    Training itself is stubbed (one synthetic epoch row so post-processing has
    data), so what this exercises is the worker's own setup path.
    """
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(port))
    # Force the CPU path so initialize_dist selects gloo: this test must not
    # depend on a working GPU/NCCL stack.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cfg = tiny_config(**(config_overrides or {}))
    kwargs = dict(vars(cfg))
    kwargs.update(
        {
            "verbose": 0,
            "vol_size": cfg.vol_size,
            "point_num": cfg.point_num,
        }
    )
    kwargs.pop("_parallel_strategy", None)

    dataset_path = tiny_dataset()
    monkeypatch.setattr(
        worker_mod, "get_dataset", lambda config, **kw: str(dataset_path)
    )
    monkeypatch.setattr(worker_mod, "get_device", lambda: torch.device("cpu"))

    seen = {}

    def fake_train(self, profiler=None):
        seen["trainer"] = self
        if on_train is not None:
            on_train(self)
        # Write one epoch row so post-processing has data to score.
        with open(self.outfile_path, "a", newline="") as outfile:
            outfile.write("1,1.0,1.0,1.0,1.0,0.5,0.96,10.0,4,4\n")

    monkeypatch.setattr(worker_mod.PyTorchTrainer, "train", fake_train)
    result = worker_mod.main(kwargs_dict=kwargs)
    return result, seen.get("trainer")


def test_worker_singleton_smoke(monkeypatch, tiny_config, tiny_dataset):
    """worker.main completes end to end as a one-rank gloo job on CPU.

    ScaFFold always runs distributed; the supported singleton case is a
    one-rank launch. The worker initializes the (gloo) process group itself,
    builds a real unsharded ParallelStrategy, and tears the group down before
    rank-0 post-processing.
    """
    result, trainer = _run_singleton_worker(
        monkeypatch, tiny_config, tiny_dataset, port=29513
    )

    assert result == 0
    assert trainer is not None
    # The model was moved to the selected device.
    param_device = next(trainer.model.parameters()).device
    assert param_device == torch.device("cpu")
    # A real (unsharded) parallel strategy is always constructed.
    assert trainer.ps is not None
    assert trainer.spatial_mesh is not None
    # world_size 1, one shard: the global batch equals the per-rank batch.
    assert trainer.config.global_batch_size == trainer.config.local_batch_size
    # The worker destroyed the process group before rank-0 post-processing.
    assert not torch.distributed.is_initialized()


@pytest.mark.parametrize(
    "mode",
    ["native-eager", "triton-eager", "native-compiled", "triton-compiled", "custom"],
)
def test_worker_sets_execution_mode_before_constructing_model(
    monkeypatch, tiny_config, tiny_dataset, mode
):
    for name in _KERNEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    worker_mod.configure_execution_mode("native-eager")
    preferences_at_construction = []
    compiled_forwards = []
    original_unet = worker_mod.UNet

    def record_unet(**kwargs):
        preferences_at_construction.append(_kernel_preferences())
        return original_unet(**kwargs)

    def fake_compile(forward, **kwargs):
        compiled_forwards.append(forward)
        return forward

    monkeypatch.setattr(worker_mod, "UNet", record_unet)
    monkeypatch.setattr(torch, "compile", fake_compile)
    result, trainer = _run_singleton_worker(
        monkeypatch,
        tiny_config,
        tiny_dataset,
        port=29554,
        config_overrides={"execution_mode": mode},
    )

    expected = (False, False, False) if mode.startswith("native-") else (None,) * 3
    assert result == 0
    assert trainer.config.execution_mode == mode
    assert preferences_at_construction == [expected]
    assert _kernel_preferences() == expected
    assert bool(compiled_forwards) is mode.endswith("-compiled")


# ---------------------------------------------------------------------------
# The activation-checkpointing config flag reaches the model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag, expected", [(0, False), (1, True)])
def test_activation_checkpointing_flag_reaches_the_model(
    monkeypatch, tiny_config, tiny_dataset, flag, expected
):
    """``activation_checkpointing: 1`` turns the U-Net's flag on.

    ``use_checkpointing`` had no caller at all, and it has to be invoked
    *before* the DDP wrap: afterwards the model is only reachable through
    ``.module``, which is exactly why this asserts on the wrapped model's
    inner module.
    """
    _result, trainer = _run_singleton_worker(
        monkeypatch,
        tiny_config,
        tiny_dataset,
        port=29520 + flag,
        config_overrides={"activation_checkpointing": flag},
    )

    model = getattr(trainer.model, "module", trainer.model)
    assert model.checkpointing is expected


def test_local_compile_wraps_forward_without_changing_checkpoint_keys(
    monkeypatch, tiny_config, tiny_dataset
):
    calls = []

    def fake_compile(forward, **kwargs):
        calls.append((forward, kwargs))
        return forward

    monkeypatch.setattr(torch, "compile", fake_compile)
    _result, trainer = _run_singleton_worker(
        monkeypatch,
        tiny_config,
        tiny_dataset,
        port=29522,
        config_overrides={
            "execution_mode": "custom",
            "torch_compile": 1,
            "torch_compile_unsharded": 1,
            "torch_compile_fullgraph": 1,
        },
    )

    model = trainer.model.module
    assert len(calls) == 1
    original_forward, compile_kwargs = calls[0]
    assert original_forward.__self__ is model
    assert original_forward.__func__ is worker_mod.UNet.forward
    assert compile_kwargs == {
        "fullgraph": True,
        "dynamic": False,
        "recompile_limit": 64,
        "backend": ANY,
    }
    assert all(not name.startswith("_orig_mod.") for name in model.state_dict())


@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "real_compile",
    [False, pytest.param(True, marks=pytest.mark.slow)],
    ids=["routing", "inductor"],
)
def test_worker_compiled_loss_executes_a_training_batch(
    monkeypatch, tiny_config, tiny_dataset, fullgraph, amp, real_compile
):
    if not real_compile:
        monkeypatch.setattr(torch, "compile", lambda forward, **kwargs: forward)
    checked = {}

    def run_batch(trainer):
        trainer.model.train()
        batch = next(iter(trainer.train_loader))
        parameters = list(trainer.model.parameters())
        before = [parameter.detach().clone() for parameter in parameters]
        batch_size, loss, dice = trainer._run_training_batch(batch)
        assert batch_size == trainer.config.local_batch_size
        assert torch.isfinite(loss)
        assert torch.isfinite(dice)
        assert not loss.requires_grad and not dice.requires_grad
        assert all(parameter.grad is None for parameter in parameters)
        assert any(
            not torch.equal(parameter, previous)
            for parameter, previous in zip(parameters, before, strict=True)
        )
        checked["ran"] = True

    result, trainer = _run_singleton_worker(
        monkeypatch,
        tiny_config,
        tiny_dataset,
        port=29523,
        config_overrides={
            "execution_mode": "custom",
            "torch_compile": 1,
            "torch_compile_fullgraph": int(fullgraph),
            "torch_compile_unsharded": 1,
            "torch_compile_loss": 1,
            "torch_amp": int(amp),
        },
        on_train=run_batch,
    )
    assert result == 0
    assert trainer is not None and checked["ran"]
    assert all(not key.startswith("_orig_mod.") for key in trainer.model.state_dict())


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_native_compiled_preset_matches_eager_training_and_evaluation(
    monkeypatch, tiny_config, tiny_dataset, tmp_path
):
    """Exercise the complete native preset with real Inductor and an eager loss."""
    for name in _KERNEL_ENV_VARS:
        monkeypatch.setenv(name, "1")
    snapshots = {}
    captured = []
    original_backend = worker_mod._compile_backend

    def observe_backend(backend=None, *, device_type="cpu"):
        compile_backend = original_backend(backend, device_type=device_type)

        def compile_graph(graph, inputs):
            captured.append(graph)
            return compile_backend(graph, inputs)

        return compile_graph

    monkeypatch.setattr(worker_mod, "_compile_backend", observe_backend)

    def run_batch(trainer):
        mode = trainer.config.execution_mode
        assert not trainer.config.torch_compile_loss
        assert _kernel_preferences() == (False, False, False)
        trainer.model.train()
        batch = next(iter(trainer.train_loader))
        parameters = dict(trainer.model.named_parameters())
        before = {name: p.detach().clone() for name, p in parameters.items()}
        batch_size, loss, dice = trainer._run_training_batch(batch)
        assert batch_size == trainer.config.local_batch_size
        assert trainer._last_step_applied
        assert all(p.grad is None for p in parameters.values())
        after = {name: p.detach().clone() for name, p in parameters.items()}
        assert any(not torch.equal(after[name], before[name]) for name in before)
        validation = evaluate(
            trainer.model,
            trainer.val_loader,
            trainer.device,
            False,
            False,
            trainer.criterion,
            trainer.config.n_categories,
            trainer.ps,
            max_batches=1,
        )
        assert trainer.model.training
        for name, parameter in parameters.items():
            assert torch.isfinite(parameter).all(), name
            torch.testing.assert_close(parameter, after[name], rtol=0, atol=0)
        metrics = torch.tensor([loss.item(), dice.item(), *validation[:3]])
        assert torch.isfinite(metrics).all()
        snapshots[mode] = {
            "image": batch["image"].clone(),
            "mask": batch["mask"].clone(),
            "before": before,
            "after": after,
            "metrics": metrics,
            "validation_counts": validation[3:],
        }

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch._dynamo.reset()
    try:
        with torch.random.fork_rng(devices=[]):
            for mode in ("native-eager", "native-compiled"):
                result, trainer = _run_singleton_worker(
                    monkeypatch,
                    tiny_config,
                    tiny_dataset,
                    port=29555,
                    config_overrides={
                        "execution_mode": mode,
                        "optimizer": "SGD",
                        "torch_amp": 0,
                        "run_dir": str(tmp_path / mode),
                    },
                    on_train=run_batch,
                )
                assert result == 0 and trainer is not None
                if mode == "native-eager":
                    assert not captured
                else:
                    assert captured, "The native preset did not compile a graph"

        reference = snapshots["native-eager"]
        actual = snapshots["native-compiled"]
        assert actual["validation_counts"] == reference["validation_counts"] == (1, 1)
        for key in ("image", "mask", "before"):
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        for key in ("after", "metrics"):
            torch.testing.assert_close(
                actual[key], reference[key], rtol=1e-4, atol=1e-5
            )
    finally:
        torch._dynamo.reset()
        torch.set_num_threads(previous_threads)


def test_unsharded_compile_boundary_preserves_outputs_and_gradients(monkeypatch):
    calls = []

    def fake_compile(forward, **kwargs):
        calls.append((forward, kwargs))
        return forward

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = worker_mod.UNet(n_channels=3, n_classes=2, layers=1)
    compiled_blocks = (*model.down_list, *model.up_list[:-1])
    original_forwards = [block.forward for block in compiled_blocks]
    original_head_forward = model.up_list[-1].forward
    state_dict_keys = list(model.state_dict())
    reference_input = torch.randn(1, 3, 4, 4, 4, requires_grad=True)
    reference_output = model(reference_input)
    reference_output.square().sum().backward()
    reference_input_grad = reference_input.grad.clone()
    reference_parameter_grads = [
        parameter.grad.clone() for parameter in model.parameters()
    ]

    model.zero_grad(set_to_none=True)
    worker_mod.compile_unsharded_forward(model)
    local_input = reference_input.detach().clone().requires_grad_(True)
    strategy = SimpleNamespace(shard_dim=(2, 3, 4))
    wrapped_input = DCTensor.from_shard(local_input, strategy)
    wrapped_output = model(wrapped_input)
    assert isinstance(wrapped_output, DCTensor)
    local_output = distconv_ops._ToTensor.apply(wrapped_output)
    local_output.square().sum().backward()

    torch.testing.assert_close(local_output, reference_output)
    torch.testing.assert_close(local_input.grad, reference_input_grad)
    for parameter, reference_grad in zip(
        model.parameters(), reference_parameter_grads, strict=True
    ):
        torch.testing.assert_close(parameter.grad, reference_grad)
    assert calls == [
        (
            forward,
            {
                "fullgraph": True,
                "dynamic": False,
                "recompile_limit": 64,
                "backend": ANY,
            },
        )
        for forward in original_forwards
    ]
    assert model.up_list[-1].forward == original_head_forward
    assert list(model.state_dict()) == state_dict_keys


def test_unsharded_fullgraph_compiles_one_model_forward(monkeypatch):
    calls = []

    def fake_compile(forward, **kwargs):
        calls.append((forward, kwargs))
        return forward

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = worker_mod.UNet(n_channels=3, n_classes=2, layers=1)
    original_forward = model.forward
    original_head_forward = model.up_list[-1].forward
    worker_mod.compile_unsharded_forward(model, fullgraph=True)

    assert calls == [
        (
            original_forward,
            {
                "fullgraph": True,
                "dynamic": False,
                "recompile_limit": 64,
                "backend": ANY,
            },
        )
    ]
    assert model.up_list[-1].forward == original_head_forward


def test_native_spatial_compile_rejects_a_selected_triton_convolution(
    gloo_group_1rank,
):
    worker_mod.configure_execution_mode("native-compiled")
    model = worker_mod.UNet(n_channels=3, n_classes=2, layers=1)
    strategy = ParallelStrategy((1, 1, 1), (2, 3, 4), device_type="cpu")
    convolutions = [m for m in model.modules() if isinstance(m, conv3d.FastConv3d)]
    original_forwards = [m.forward for m in convolutions]
    # Represent an already-selected Triton module without requiring a GPU.
    # The CPU metadata warmup leaves its proven-route flag intact.
    convolutions[0]._triton_ok = True
    worker_mod.compile_spatial_forward(model, strategy, backend="aot_eager")
    images = DCTensor.from_shard(torch.randn(1, 3, 4, 4, 4), strategy)

    with pytest.raises(ValueError, match="cannot include ScaFFold Triton"):
        model(images)

    # The rejected route must not partially replace any convolution forward.
    assert [m.forward for m in convolutions] == original_forwards


def test_unsharded_fullgraph_loss_and_validation_preserve_gradients(monkeypatch):
    calls = []

    def fake_compile(forward, **kwargs):
        calls.append((forward, kwargs))
        return forward

    monkeypatch.setattr(torch, "compile", fake_compile)
    reference = worker_mod.UNet(n_channels=3, n_classes=2, layers=1)
    model = copy.deepcopy(reference)
    state_dict_keys = list(model.state_dict())
    strategy = SimpleNamespace(
        shard_dim=(2, 3, 4),
        num_shards=(1, 1, 1),
        distconv_dim_names=("spatial",),
        device_mesh={("spatial",): None},
    )
    images = torch.randn(1, 3, 8, 8, 8)
    labels = torch.randint(0, 2, (1, 8, 8, 8))
    weights = torch.tensor([1.0, 3.0])

    reference_input = images.clone().requires_grad_(True)
    reference_logits = reference(reference_input)
    reference_loss, reference_dice = compute_training_loss_and_dice(
        reference_logits, labels, weights, None, strategy.num_shards, "cpu"
    )
    reference_loss.backward()

    worker_mod.compile_unsharded_forward(
        model,
        fullgraph=True,
        compile_loss=True,
        parallel_strategy=strategy,
    )
    compiled_input = images.clone().requires_grad_(True)
    dc_images = DCTensor.from_shard(compiled_input, strategy)
    dc_labels = DCTensor.from_shard(labels.unsqueeze(1), strategy)
    loss, dice = model(dc_images, dc_labels, weights)
    loss.backward()

    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(dice, reference_dice)
    torch.testing.assert_close(compiled_input.grad, reference_input.grad)
    for parameter, reference_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)

    with torch.no_grad():
        validation_output = model(dc_images)
    assert isinstance(validation_output, DCTensor)
    torch.testing.assert_close(
        distconv_ops._ToTensor.apply(validation_output), reference_logits
    )
    assert len(calls) == 2
    assert all(kwargs["fullgraph"] for _, kwargs in calls)
    assert list(model.state_dict()) == state_dict_keys


# ---------------------------------------------------------------------------
# Local size detection
# ---------------------------------------------------------------------------

_LOCAL_SIZE_CASES = [
    # torchrun exports LOCAL_WORLD_SIZE alongside LOCAL_RANK.
    ({"LOCAL_WORLD_SIZE": "4"}, 4),
    ({"MV2_COMM_WORLD_LOCAL_SIZE": "4"}, 4),
    ({"OMPI_COMM_WORLD_LOCAL_SIZE": "4"}, 4),
    ({"PMI_LOCAL_SIZE": "4"}, 4),
    ({"PALS_LOCAL_SIZE": "4"}, 4),
    ({"SLURM_NTASKS": "8", "SLURM_NNODES": "2"}, 4),
    ({"FLUX_JOB_SIZE": "8", "FLUX_JOB_NNODES": "2"}, 4),
]

_LOCAL_SIZE_VARS = [
    "LOCAL_WORLD_SIZE",
    "MV2_COMM_WORLD_LOCAL_SIZE",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "PMI_LOCAL_SIZE",
    "PALS_LOCAL_SIZE",
    "SLURM_NTASKS",
    "SLURM_NNODES",
    "FLUX_JOB_SIZE",
    "FLUX_JOB_NNODES",
]


def test_local_size_detection_matrix(monkeypatch):
    """Every launcher that reports a local rank has its local size honored too.

    An unrecognized variable silently yields 1, which makes the per-node
    profiler gate ``rank % ranks_per_node == 0`` select *every* rank and
    mislabels the trace's node count.
    """
    for env, want_local_size in _LOCAL_SIZE_CASES:
        for var in _LOCAL_SIZE_VARS:
            monkeypatch.delenv(var, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        assert distributed_mod.get_local_size() == want_local_size, env


def test_local_size_defaults_to_one(monkeypatch):
    """With nothing to go on, one rank per node is still the assumption."""
    for var in _LOCAL_SIZE_VARS:
        monkeypatch.delenv(var, raising=False)
    assert distributed_mod.get_local_size() == 1


# ---------------------------------------------------------------------------
# VC-3: an unusable launcher variable is ignored, not fatal
#
# These helpers run at the top of every entry point, so a bare int() on a
# variable a site wrapper exported empty ("WORLD_SIZE=") or as a placeholder
# ("auto") killed the invocation -- ``scaffold --help`` included -- with a
# ValueError naming neither the variable nor a remedy.
# ---------------------------------------------------------------------------


def _clear_launcher_env(monkeypatch):
    for var in set(_LAUNCHER_VARS) | set(_LOCAL_SIZE_VARS):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("value", ["", "   ", "auto"])
def test_unusable_launcher_values_fall_through(monkeypatch, value):
    """Empty and non-numeric values are treated as absent, never raise."""
    _clear_launcher_env(monkeypatch)
    for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.setenv(var, value)

    # Falls through to the next source -- here the (singleton) communicator and
    # the documented defaults.
    assert distributed_mod.get_world_size() == 1
    assert distributed_mod.get_world_rank() == 0
    assert distributed_mod.get_local_rank() == 0
    assert distributed_mod.get_local_size() == 1


def test_unusable_value_defers_to_the_next_launcher_variable(monkeypatch, caplog):
    """A garbage value does not mask a usable variable further down the order."""
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "auto")
    monkeypatch.setenv("PALS_NRANKS", "8")

    with caplog.at_level(logging.WARNING, logger=distributed_mod.logger.name):
        assert distributed_mod.get_world_size() == 8

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "WORLD_SIZE" in messages, "the ignored value was not reported"


def test_zero_node_count_does_not_divide_by_zero(monkeypatch):
    """A nonsense node count falls through instead of raising."""
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_NNODES", "0")

    assert distributed_mod.get_local_size() == 1
