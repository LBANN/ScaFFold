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

import math
import os
from pathlib import Path

import yaml


def require_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def require_flag(name: str, value) -> bool:
    """Validate an on/off config toggle written as 0/1 (or a YAML boolean).

    ``bool(value)`` would quietly accept ``2``, ``-1`` or ``"no"`` (all true),
    so a mistyped toggle would enable the feature it was meant to disable.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be 0, 1, or a boolean; got {value!r}")


def validate_unet_dims(problem_scale, unet_bottleneck_dim) -> int:
    """Check that ``problem_scale``/``unet_bottleneck_dim`` describe a real U-Net.

    The U-Net has ``unet_layers = problem_scale - unet_bottleneck_dim``
    down/up levels over a ``2**problem_scale`` volume, so the bottleneck
    exponent must satisfy ``0 <= unet_bottleneck_dim <= problem_scale - 1``:
    a larger value asks for a bottleneck no smaller than the input (zero or
    negative layers) and a negative one asks for more pooling levels than the
    volume has. Both are only discovered later as an opaque
    ``max_pool3d`` size error -- in production, after the whole dataset has
    been generated -- so reject them here, at config time, naming the two keys
    that have to change.

    Returns the validated bottleneck dimension.
    """
    if isinstance(unet_bottleneck_dim, bool) or not isinstance(
        unet_bottleneck_dim, int
    ):
        raise ValueError(
            f"unet_bottleneck_dim must be an integer; got {unet_bottleneck_dim!r}"
        )
    unet_layers = problem_scale - unet_bottleneck_dim
    if unet_bottleneck_dim < 0 or unet_layers < 1:
        raise ValueError(
            f"unet_bottleneck_dim={unet_bottleneck_dim} is out of range for "
            f"problem_scale={problem_scale}: it must satisfy "
            f"0 <= unet_bottleneck_dim <= problem_scale - 1 "
            f"(i.e. <= {problem_scale - 1}) so that the U-Net has at least one "
            f"layer, but unet_layers = problem_scale - unet_bottleneck_dim = "
            f"{unet_layers}. Raise problem_scale or lower unet_bottleneck_dim."
        )
    return unet_bottleneck_dim


class Config:
    """
    A class for storing configuration settings for a specific run.

    Unknown keys are rejected by default so typos and unsupported options
    fail at load time instead of being silently ignored.
    """

    # Keys read by __init__ (required and optional).
    KNOWN_KEYS = frozenset(
        {
            "base_run_dir",
            "dataset_dir",
            "fract_base_dir",
            "job_name",
            "n_categories",
            "problem_scale",
            "unet_bottleneck_dim",
            "n_fracts_per_vol",
            "n_instances_used_per_fractal",
            "local_batch_size",
            "dataloader_num_workers",
            "epochs",
            "optimizer",
            "disable_scheduler",
            "more_determinism",
            "datagen_from_scratch",
            "train_from_scratch",
            "val_split",
            "seed",
            "dist",
            "framework",
            "starting_learning_rate",
            "min_learning_rate",
            "T_0",
            "T_mult",
            "variance_threshold",
            "torch_amp",
            "loss_freq",
            "checkpoint_dir",
            "normalize",
            "group_norm_groups",
            "warmup_batches",
            "activation_checkpointing",
            "ce_weight_sample_fraction",
            "dataset_reuse_enforce_commit_id",
            "target_dice",
            "checkpoint_interval",
            "dc_num_shards",
            "dc_shard_dims",
            "async_save",
            # Derived attributes Config itself emits; accepting them keeps a
            # saved run config.yaml reloadable.
            "scale",
            "dc_total_shards",
        }
    )

    # Keys injected by the CLI/benchmark layers around the core options.
    # Accepted (and preserved on round-trip through a saved config.yaml)
    # but not consumed by Config itself.
    AUX_KEYS = frozenset(
        {
            "command",
            "config",
            "verbose",
            "restart",
            "run_dir",
            "run_iter",
            "benchmark_run_dir",
            "unet_layers",
            "vol_size",
            "point_num",
            "scheduler_metadata",
            "machine_name",
            "datagen_batch_size",
        }
    )

    # Fields that hold scalars. Parameter sweeps are not supported, so a list
    # here is a user error and is rejected by name (see _validate_keys).
    _SCALAR_KEYS = frozenset(
        {
            "n_categories",
            "problem_scale",
            "unet_bottleneck_dim",
            "n_fracts_per_vol",
            "n_instances_used_per_fractal",
            "local_batch_size",
            "dataloader_num_workers",
            "epochs",
            "optimizer",
            "val_split",
            "seed",
            "starting_learning_rate",
            "min_learning_rate",
            "T_0",
            "T_mult",
            "variance_threshold",
            "loss_freq",
            "group_norm_groups",
            "warmup_batches",
            "activation_checkpointing",
            "ce_weight_sample_fraction",
            "target_dice",
            "checkpoint_interval",
        }
    )

    @classmethod
    def _validate_keys(cls, config_dict, strict):
        allowed = cls.KNOWN_KEYS | cls.AUX_KEYS
        unknown = sorted(set(config_dict) - allowed)
        if unknown:
            message = f"Unknown config key(s): {', '.join(unknown)}"
            if strict:
                raise ValueError(message)
            print(f"WARNING: {message}")
        lists = sorted(
            k for k in cls._SCALAR_KEYS if isinstance(config_dict.get(k), list)
        )
        if lists:
            details = "; ".join(
                f"{k}: parameter sweeps are no longer supported; "
                f"got list {config_dict[k]}"
                for k in lists
            )
            raise ValueError(
                f"{details}. Each of these keys must hold a single value; "
                "launch one benchmark run per parameter setting."
            )

    def __init__(self, config_dict, strict=True):
        self._validate_keys(config_dict, strict)
        self.base_run_dir = str(Path(config_dict["base_run_dir"]).resolve())
        self.dataset_dir = str(
            Path(config_dict.get("dataset_dir", "datasets/")).resolve()
        )
        self.fract_base_dir = str(
            Path(config_dict.get("fract_base_dir", "fractals/")).resolve()
        )
        self.job_name = config_dict.get("job_name", "benchmark")
        self.n_categories = require_positive_int(
            "n_categories", config_dict["n_categories"]
        )
        self.problem_scale = config_dict["problem_scale"]
        try:
            assert isinstance(self.problem_scale, int), (
                "problem_scale must be a positive integer"
            )
        except AssertionError:
            print(
                "WARNING: problem_scale found to be non-integer. Truncating to nearest int."
            )
            self.problem_scale = math.floor(self.problem_scale)
        self.unet_bottleneck_dim = validate_unet_dims(
            self.problem_scale, config_dict["unet_bottleneck_dim"]
        )
        self.unet_layers = self.problem_scale - self.unet_bottleneck_dim
        self.n_fracts_per_vol = config_dict["n_fracts_per_vol"]
        self.n_instances_used_per_fractal = config_dict["n_instances_used_per_fractal"]
        self.scale = 1
        self.local_batch_size = config_dict["local_batch_size"]
        self.dataloader_num_workers = config_dict["dataloader_num_workers"]
        self.epochs = config_dict["epochs"]
        self.optimizer = config_dict["optimizer"]
        self.disable_scheduler = bool(config_dict["disable_scheduler"])
        self.more_determinism = bool(config_dict["more_determinism"])
        self.datagen_from_scratch = bool(config_dict["datagen_from_scratch"])
        self.train_from_scratch = bool(config_dict["train_from_scratch"])
        self.restart = bool(config_dict.get("restart", False))
        self.val_split = config_dict["val_split"]
        self.seed = config_dict["seed"]
        if "dist" in config_dict and not bool(config_dict["dist"]):
            raise ValueError(
                "The 'dist: 0' mode is no longer supported. ScaFFold benchmark "
                "training always runs with distributed execution; use a one-rank "
                "torchrun-hpc job for singleton runs."
            )
        self.framework = config_dict["framework"]
        self.starting_learning_rate = config_dict["starting_learning_rate"]
        self.min_learning_rate = config_dict["min_learning_rate"]
        self.T_0 = config_dict["T_0"]
        self.T_mult = config_dict["T_mult"]
        self.variance_threshold = config_dict["variance_threshold"]
        self.torch_amp = bool(config_dict["torch_amp"])
        self.loss_freq = config_dict["loss_freq"]
        self.checkpoint_dir = config_dict["checkpoint_dir"]
        self.normalize = config_dict["normalize"]
        self.group_norm_groups = config_dict.get("group_norm_groups", 8)
        self.warmup_batches = config_dict.get("warmup_batches")
        self.activation_checkpointing = require_flag(
            "activation_checkpointing",
            config_dict.get("activation_checkpointing", 0),
        )
        self.ce_weight_sample_fraction = config_dict.get(
            "ce_weight_sample_fraction", 0.1
        )
        self.dataset_reuse_enforce_commit_id = config_dict[
            "dataset_reuse_enforce_commit_id"
        ]
        self.target_dice = config_dict["target_dice"]
        self.checkpoint_interval = config_dict["checkpoint_interval"]
        self.async_save = bool(config_dict.get("async_save", False))
        self.dc_num_shards = config_dict["dc_num_shards"]
        self.dc_shard_dims = config_dict["dc_shard_dims"]
        self.dc_total_shards = math.prod(self.dc_num_shards)
        unsupported_dataset_keys = [
            key
            for key in ("dataset_num_shards", "dataset_shard_dims")
            if key in config_dict
        ]
        if unsupported_dataset_keys:
            raise ValueError(
                "Configuration Mismatch: dataset_num_shards/dataset_shard_dims "
                "are not supported. Use dc_num_shards/dc_shard_dims for the "
                "v3 physical dataset layout."
            )
        # Safety Check: Length mismatch
        if len(self.dc_num_shards) != len(self.dc_shard_dims):
            raise ValueError(
                f"Configuration Mismatch: num_shards {self.dc_num_shards} "
                f"must have same length as shard_dim {self.dc_shard_dims}"
            )


class RunConfig(Config):
    def __init__(self, config_dict, strict=True):
        super().__init__(config_dict, strict=strict)

        self.run_dir = config_dict["run_dir"]
        self.run_iter = config_dict["run_iter"]


def load_config_files(file_paths):
    """
    Load a base config plus optional partial overrides.

    The first file must be a complete config; each subsequent file is a
    partial override whose keys replace the base values. Every override key
    must be a recognized config key.

    Returns:
        dict: the merged configuration dictionary.
    """
    if not file_paths:
        raise ValueError("At least one config file is required")

    merged = None
    allowed = Config.KNOWN_KEYS | Config.AUX_KEYS
    for i, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file '{file_path}' not found")
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file) or {}
        if i == 0:
            merged = dict(config_dict)
            continue
        unknown = sorted(set(config_dict) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown config key(s) in override file '{file_path}': "
                f"{', '.join(unknown)}"
            )
        for key, value in config_dict.items():
            merged[key] = value
    return merged


def load_config(file_path: str, config_type: str):
    """
    Load a config from a yaml file.

    ``config_type`` is either ``"benchmark"`` (a benchmark config, before a run
    directory has been resolved) or ``"run"`` (a config for one run, which also
    carries ``run_dir``/``run_iter``). Both require single-valued parameters.

    Returns:
        Config: A Config instance with settings loaded from the yaml file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file '{file_path}' not found")

    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)

    if config_type == "benchmark":
        return Config(config_dict)
    elif config_type == "run":
        return RunConfig(config_dict)
    else:
        raise ValueError(
            f"Invalid config type specified: {config_type}. "
            "Must be either 'benchmark' or 'run'"
        )
