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

Covers the non-distributed (dist=0) worker path, DDP device selection under
per-rank GPU masking, launcher environment detection, and the ordering of
device binding relative to process-group initialization.
"""

import torch

import ScaFFold.utils.distributed as distributed_mod
import ScaFFold.worker as worker_mod

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
    distributed_mod.initialize_dist(rendezvous="env")
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
    assert recorded["device_ids"] == [device]
    assert recorded["output_device"] == device


# ---------------------------------------------------------------------------
# Non-distributed (dist=0) worker path
# ---------------------------------------------------------------------------


def _run_worker_nondist(monkeypatch, tiny_config, tiny_dataset, drop_dc_keys):
    """Run worker.main with dist=0 on CPU; return (config kwargs, trainer)."""
    cfg = tiny_config()
    kwargs = dict(vars(cfg))
    kwargs.update(
        {
            "dist": 0,
            "verbose": 0,
            "_parallel_strategy": None,
            "vol_size": cfg.vol_size,
            "point_num": cfg.point_num,
        }
    )
    if drop_dc_keys:
        kwargs.pop("dc_num_shards", None)
        kwargs.pop("dc_shard_dims", None)
        kwargs.pop("dc_total_shards", None)

    dataset_path = tiny_dataset()
    monkeypatch.setattr(
        worker_mod, "get_dataset", lambda config, **kw: str(dataset_path)
    )
    monkeypatch.setattr(worker_mod, "get_device", lambda: torch.device("cpu"))

    seen = {}

    def fake_train(self):
        seen["trainer"] = self
        # Write one epoch row so post-processing has data to score.
        with open(self.outfile_path, "a", newline="") as outfile:
            outfile.write("1,1.0,1.0,1.0,1.0,0.5,0.96,10.0\n")

    monkeypatch.setattr(worker_mod.PyTorchTrainer, "train", fake_train)
    result = worker_mod.main(kwargs_dict=kwargs)
    return result, seen.get("trainer")


def test_worker_nondist_smoke(monkeypatch, tiny_config, tiny_dataset):
    """worker.main completes end to end with dist=0 on CPU."""
    result, trainer = _run_worker_nondist(
        monkeypatch, tiny_config, tiny_dataset, drop_dc_keys=False
    )
    assert result == 0
    assert trainer is not None
    # The model was moved to the selected device outside the DDP block.
    param_device = next(trainer.model.parameters()).device
    assert param_device == torch.device("cpu")
    # ps=None semantics: single shard, world-size-scaled global batch.
    assert trainer.ps is None
    assert trainer.spatial_mesh is None
    assert trainer.config.global_batch_size == trainer.config.local_batch_size


def test_dc_num_shards_default(monkeypatch, tiny_config, tiny_dataset):
    """A config without shard settings falls back to unsharded defaults."""
    result, trainer = _run_worker_nondist(
        monkeypatch, tiny_config, tiny_dataset, drop_dc_keys=True
    )
    assert result == 0
    assert trainer.config.dc_num_shards == [1, 1, 1]
    assert trainer.config.dc_shard_dims == [2, 3, 4]
