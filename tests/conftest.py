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

"""Shared fixtures for the ScaFFold test suite.

Everything here is designed to run at the smallest workable scale (16^3
volumes, 1 U-Net layer, CPU, no parallel strategy) so per-batch tests can
build real objects without a GPU, an MPI launcher, or the datagen pipeline.

Key facts baked into these fixtures (verified empirically against the code):

* ``Config`` / ``RunConfig`` read a fixed set of keys with no defaults; the
  ``tiny_config`` factory supplies every one of them plus the extra attributes
  the trainer/worker read via ``config.X`` that ``Config.__init__`` never sets
  (``verbose``, ``_parallel_strategy``, ``vol_size``, ``point_num``).

* v2 datasets store channels-first ``(3, N, N, N)`` float32 volumes and uint16
  ``(N, N, N)`` masks holding already-remapped class ids, plus a ``meta.yaml``
  with ``dataset_format_version: 2`` and pickled ``*_unique_mask_vals``.

* v1 datasets have no ``meta.yaml``; volumes are channels-*last*
  ``(N, N, N, 3)`` and masks hold raw values that the loader remaps through the
  pickled ``mask_values`` list.

* The trainer's forward path (``_run_training_batch``) is hardwired through
  DistConv and cannot run with ``ps=None`` (``DCTensor.from_shard(x, None)``
  dereferences ``None.shard_dim`` and segfaults). ``tiny_trainer`` therefore
  builds a CPU trainer with ``ps=None`` for *construction / control-flow*
  coverage; tests that need a training step must supply a real
  ``ParallelStrategy`` under a gloo process group.
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import yaml

# Repo root: tests/ lives directly under it.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# singleton launcher environment
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def singleton_launcher_env(monkeypatch):
    """Provide a one-rank launcher environment for every test.

    ScaFFold always runs as a distributed job; the supported singleton case is
    a one-rank launch, where the launcher exports rank/size variables. The
    trainer and worker read them with ``required=True``, so single-process
    tests need them present. Tests that exercise launcher detection delete
    them explicitly.
    """
    monkeypatch.setenv("RANK", os.environ.get("RANK", "0"))
    monkeypatch.setenv("WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))
    monkeypatch.setenv("LOCAL_RANK", os.environ.get("LOCAL_RANK", "0"))


# ---------------------------------------------------------------------------
# tiny_config
# ---------------------------------------------------------------------------

# Baseline config values, adapted from ScaFFold/configs/benchmark_default.yml
# but shrunk to the smallest workable scale. Every key that Config.__init__
# and RunConfig.__init__ read without a default is present here.
_BASE_CONFIG = {
    # paths (overridden per-fixture to point into tmp_path)
    "base_run_dir": "benchmark_runs",
    "dataset_dir": "datasets",
    "fract_base_dir": "fractals",
    "job_name": "benchmark",
    # problem size -- 16^3 volumes, 1 U-Net layer (problem_scale - bottleneck)
    "n_categories": 2,
    "n_instances_used_per_fractal": 6,
    "problem_scale": 4,
    "unet_bottleneck_dim": 3,
    "seed": 42,
    "local_batch_size": 1,
    "dataloader_num_workers": 0,
    "optimizer": "ADAM",
    "dc_num_shards": [1, 1, 1],
    "dc_shard_dims": [2, 3, 4],
    "checkpoint_interval": -1,
    # internal / dev
    "variance_threshold": 0.15,
    "n_fracts_per_vol": 3,
    "val_split": 30,
    "epochs": 1,
    "starting_learning_rate": 0.001,
    "min_learning_rate": 0.0001,
    "T_0": 100,
    "T_mult": 2,
    "disable_scheduler": 1,
    "more_determinism": 0,
    "datagen_from_scratch": 0,
    "train_from_scratch": 1,
    "torch_amp": 0,
    "framework": "torch",
    "checkpoint_dir": "checkpoints",
    "loss_freq": 1,
    "normalize": 1,
    "group_norm_groups": 8,
    "warmup_batches": 0,
    "ce_weight_sample_fraction": 1.0,
    "dataset_reuse_enforce_commit_id": 0,
    "target_dice": 0.95,
}


@pytest.fixture
def tiny_config(tmp_path):
    """Factory building a fully-populated ``RunConfig`` at the smallest scale.

    Usage::

        cfg = tiny_config()                       # defaults
        cfg = tiny_config(n_categories=3)         # override any key
        cfg = tiny_config(run_dir=str(some_dir))  # point at your own run dir

    ``run_dir`` / ``run_iter`` default into ``tmp_path`` (the dir is created).
    Beyond the constructor keys, the returned object also carries the attributes
    the trainer/worker read via ``config.X`` but ``Config.__init__`` never sets:
    ``_parallel_strategy=None``, ``verbose=0``, and the datagen/viz-only
    ``vol_size`` / ``point_num``.
    """
    from ScaFFold.utils.config_utils import RunConfig

    def make(**overrides) -> "RunConfig":
        config_dict = dict(_BASE_CONFIG)

        # Default paths into tmp_path unless the caller overrides them.
        default_run_dir = tmp_path / "run"
        config_dict["base_run_dir"] = str(tmp_path / "benchmark_runs")
        config_dict["dataset_dir"] = str(tmp_path / "datasets")
        config_dict["fract_base_dir"] = str(tmp_path / "fractals")
        config_dict["run_dir"] = str(default_run_dir)
        config_dict["run_iter"] = 0

        config_dict.update(overrides)

        # Ensure the run directory exists so the trainer can write into it.
        run_dir = Path(config_dict["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)

        config = RunConfig(config_dict)

        # Attributes referenced via config.X that Config.__init__ does not set.
        # trainer.BaseTrainer reads _parallel_strategy via getattr(..., None);
        # trainer._run_training_batch reads config.verbose directly.
        config._parallel_strategy = overrides.get("_parallel_strategy", None)
        config.verbose = overrides.get("verbose", 0)
        # datagen/viz derive these from problem_scale; harmless for training.
        config.vol_size = overrides.get("vol_size", 2**config.problem_scale)
        config.point_num = overrides.get(
            "point_num", int((2**config.problem_scale) ** 3 / 256)
        )
        return config

    return make


# ---------------------------------------------------------------------------
# dataset builders (v2 + v1) -- no datagen dependency
# ---------------------------------------------------------------------------


def _write_split(
    volumes_dir: Path,
    masks_dir: Path,
    count: int,
    n: int,
    n_categories: int,
    rng: np.random.Generator,
    *,
    legacy: bool,
    legacy_values,
) -> None:
    """Write ``count`` volume/mask pairs into one split directory.

    v2 (``legacy=False``): channels-first float32 volumes ``(3, N, N, N)`` and
    uint16 masks ``(N, N, N)`` holding remapped ids ``0..n_categories``.

    v1 (``legacy=True``): channels-last float32 volumes ``(N, N, N, 3)`` and
    uint16 masks holding *raw* values drawn from ``legacy_values`` (which the
    loader remaps through the pickled ``mask_values`` list).
    """
    volumes_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        if legacy:
            volume = rng.random((n, n, n, 3), dtype=np.float32)
            mask = rng.choice(
                np.asarray(legacy_values, dtype=np.uint16), size=(n, n, n)
            )
            mask = mask.astype(np.uint16)
        else:
            volume = rng.random((3, n, n, n), dtype=np.float32)
            mask = rng.integers(0, n_categories + 1, size=(n, n, n), dtype=np.uint16)
        np.save(volumes_dir / f"{i}.npy", volume)
        np.save(masks_dir / f"{i}_mask.npy", mask)


def _build_dataset(
    root: Path,
    *,
    n_categories: int,
    n_train: int,
    n_val: int,
    n: int,
    seed: int,
    legacy: bool,
) -> Path:
    """Build a full v1 or v2 dataset directory tree under ``root``.

    Returns the dataset root path (``root`` itself).
    """
    rng = np.random.default_rng(seed)

    if legacy:
        # v1 masks store raw values remapped via mask_values; index i -> class i.
        legacy_values = [0] + [10 * (c + 1) for c in range(n_categories)]
        mask_values = list(legacy_values)
    else:
        legacy_values = None
        mask_values = list(range(n_categories + 1))

    _write_split(
        root / "volumes" / "training",
        root / "masks" / "training",
        n_train,
        n,
        n_categories,
        rng,
        legacy=legacy,
        legacy_values=legacy_values,
    )
    _write_split(
        root / "volumes" / "validation",
        root / "masks" / "validation",
        n_val,
        n,
        n_categories,
        rng,
        legacy=legacy,
        legacy_values=legacy_values,
    )

    for name in ("train_unique_mask_vals", "val_unique_mask_vals"):
        with open(root / name, "wb") as handle:
            pickle.dump({"mask_values": mask_values}, handle)

    if not legacy:
        with open(root / "meta.yaml", "w") as handle:
            yaml.safe_dump({"dataset_format_version": 2}, handle)

    return root


@pytest.fixture
def tiny_dataset(tmp_path):
    """Factory building a v2-format dataset directory (no datagen).

    Usage::

        ds = tiny_dataset()                       # 4 train / 2 val, 16^3, 2 cats
        ds = tiny_dataset(n_categories=3, n=8)    # override

    Returns the dataset root ``Path`` (contains ``volumes/``, ``masks/``,
    ``meta.yaml``, and the two ``*_unique_mask_vals`` pickles).
    """

    def make(
        root: Optional[Path] = None,
        *,
        n_categories: int = 2,
        n_train: int = 4,
        n_val: int = 2,
        n: int = 16,
        seed: int = 1234,
    ) -> Path:
        if root is None:
            root = tmp_path / "dataset_v2"
        root = Path(root)
        return _build_dataset(
            root,
            n_categories=n_categories,
            n_train=n_train,
            n_val=n_val,
            n=n,
            seed=seed,
            legacy=False,
        )

    return make


@pytest.fixture
def tiny_v1_dataset(tmp_path):
    """Factory building a v1-format (legacy) dataset directory.

    Same shape as ``tiny_dataset`` but: no ``meta.yaml`` (so the loader falls
    back to the legacy path), channels-last volumes ``(N, N, N, 3)``, and masks
    holding raw values that get remapped through the per-split ``mask_values``
    pickle. Needed by the F42 dataset-loading tests.
    """

    def make(
        root: Optional[Path] = None,
        *,
        n_categories: int = 2,
        n_train: int = 4,
        n_val: int = 2,
        n: int = 16,
        seed: int = 5678,
    ) -> Path:
        if root is None:
            root = tmp_path / "dataset_v1"
        root = Path(root)
        return _build_dataset(
            root,
            n_categories=n_categories,
            n_train=n_train,
            n_val=n_val,
            n=n,
            seed=seed,
            legacy=True,
        )

    return make


# ---------------------------------------------------------------------------
# run_dir skeleton
# ---------------------------------------------------------------------------


@pytest.fixture
def run_dir(tmp_path):
    """Factory building a skeleton run directory for resume/checkpoint tests.

    Usage::

        rd = run_dir()                                     # empty run dir
        rd = run_dir(config={"seed": 42})                  # writes config.yaml
        rd = run_dir(train_stats="epoch,val_dice\\n1,0.5\\n")  # writes csv
        rd = run_dir(with_checkpoints=True)                # makes checkpoints/

    Returns the run directory ``Path``.
    """

    def make(
        name: str = "run",
        *,
        config: Optional[dict] = None,
        train_stats: Optional[str] = None,
        with_checkpoints: bool = False,
    ) -> Path:
        rd = tmp_path / name
        rd.mkdir(parents=True, exist_ok=True)

        if config is not None:
            with open(rd / "config.yaml", "w") as handle:
                yaml.safe_dump(config, handle)

        if train_stats is not None:
            (rd / "train_stats.csv").write_text(train_stats)

        if with_checkpoints:
            (rd / "checkpoints").mkdir(parents=True, exist_ok=True)

        return rd

    return make


# ---------------------------------------------------------------------------
# one-rank gloo process group
# ---------------------------------------------------------------------------


@pytest.fixture
def gloo_group_1rank():
    """Initialize (if needed) a one-rank gloo process group for the test.

    The trainer path is unconditionally distributed (CE-weight estimation,
    metric reductions, and checkpoint broadcasts all issue collectives), so
    even single-process trainer tests need an initialized default group. If a
    group already exists it is reused and left alone; otherwise one is created
    and destroyed when the test ends.
    """
    import torch.distributed as dist

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created = True
    yield
    if created and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# tiny_trainer
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_trainer(tiny_config, tiny_dataset, tmp_path, gloo_group_1rank):
    """Factory building a real ``PyTorchTrainer`` on CPU with ``ps=None``.

    Usage::

        trainer = tiny_trainer()                    # defaults
        trainer = tiny_trainer(config_overrides={"target_dice": 0.0})

    The trainer is constructed against a fresh ``tiny_dataset`` (v2) and a real
    1-layer U-Net, with ``ps=None`` (a real ``ParallelStrategy`` needs a
    process group, which single-process CPU tests avoid).

    KNOWN CONSTRAINT: the forward path (``_run_training_batch``) is hardwired
    through DistConv and cannot run with ``ps=None``
    (``DCTensor.from_shard(x, None)`` dereferences ``None.shard_dim`` and
    segfaults). Construction, ``cleanup_or_resume``, and ``train()`` with
    ``target_dice <= 0`` (which exits before the first batch) are all safe;
    exercising an actual training step requires a gloo process group and a real
    ``ParallelStrategy``.

    ``ce_weight_sample_fraction`` is forced to ``1.0`` so the CE-weight
    estimation at construction samples the (tiny) dataset quickly and
    deterministically.
    """
    import torch

    from ScaFFold.unet import UNet
    from ScaFFold.utils.trainer import PyTorchTrainer

    def make(
        *,
        n_categories: int = 2,
        n_train: int = 4,
        n_val: int = 2,
        n: int = 16,
        config_overrides: Optional[dict] = None,
    ) -> "PyTorchTrainer":
        dataset_root = tiny_dataset(
            n_categories=n_categories, n_train=n_train, n_val=n_val, n=n
        )

        overrides = {
            "dataset_dir": str(dataset_root),
            "n_categories": n_categories,
            "ce_weight_sample_fraction": 1.0,
            "torch_amp": 0,
            "dataloader_num_workers": 0,
        }
        if config_overrides:
            overrides.update(config_overrides)

        config = tiny_config(**overrides)

        model = UNet(
            n_channels=3,
            n_classes=config.n_categories + 1,
            trilinear=False,
            layers=config.unet_layers,
            group_norm_groups=config.group_norm_groups,
        )
        device = torch.device("cpu")

        log = logging.getLogger(f"tiny_trainer.{id(config)}")
        # INFO (20) > DEBUG (10) => gather_and_print_mem short-circuits and
        # never touches CUDA / torch.distributed.
        log.setLevel(logging.INFO)

        return PyTorchTrainer(model, config, device, log)

    return make


# ---------------------------------------------------------------------------
# fresh_python
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_python():
    """Run a Python snippet in a fresh subprocess and return its stdout.

    Uses the same interpreter running the tests, with ``PYTHONPATH`` pointed at
    the repo root. Required by numba-RNG determinism tests: numba's RNG state
    is process-global and cannot be reset from within a live interpreter, so
    each determinism check must run in its own process.

    Usage::

        out = fresh_python("import numpy; print(numpy.__version__)")
        out = fresh_python(snippet, env={"SCAFFOLD_SEED": "7"}, timeout=30)

    Raises ``AssertionError`` (with captured stderr) on non-zero exit.
    """

    def run(snippet: str, *, env: Optional[dict] = None, timeout: int = 60) -> str:
        child_env = os.environ.copy()
        existing = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
        )
        if env:
            child_env.update(env)

        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, (
            f"fresh_python subprocess failed (rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
        return proc.stdout

    return run


# ---------------------------------------------------------------------------
# marker auto-skip: gpu
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item):
    """Auto-skip ``@pytest.mark.gpu`` tests when no CUDA device is present."""
    if "gpu" in item.keywords:
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("no CUDA device available")
        except ImportError:
            pytest.skip("torch not available")
