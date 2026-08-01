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

import logging

import pytest
import torch

import ScaFFold.utils.distributed as distributed_mod
import ScaFFold.worker as worker_mod

_TEST_LOG = logging.getLogger(__name__)

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


def test_worker_singleton_smoke(monkeypatch, tiny_config, tiny_dataset):
    """worker.main completes end to end as a one-rank gloo job on CPU.

    ScaFFold always runs distributed; the supported singleton case is a
    one-rank launch. The worker initializes the (gloo) process group itself,
    builds a real unsharded ParallelStrategy, and tears the group down before
    rank-0 post-processing.
    """
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29513")
    # Force the CPU path so initialize_dist selects gloo: this test must not
    # depend on a working GPU/NCCL stack.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cfg = tiny_config()
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
        # Write one epoch row so post-processing has data to score.
        with open(self.outfile_path, "a", newline="") as outfile:
            outfile.write("1,1.0,1.0,1.0,1.0,0.5,0.96,10.0,4,4\n")

    monkeypatch.setattr(worker_mod.PyTorchTrainer, "train", fake_train)
    result = worker_mod.main(kwargs_dict=kwargs)
    trainer = seen.get("trainer")

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


# ---------------------------------------------------------------------------
# Local size detection (R23)
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
