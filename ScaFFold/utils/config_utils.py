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

import ScaFFold.paths


class Config:
    """
    A class for storing configuration settings for a specific run.
    """

    def __init__(self, config_dict):
        self.library_root = str(ScaFFold.paths.scaffold_root).rstrip("/") + "/ScaFFold/"
        self.base_run_dir = str(Path(config_dict["base_run_dir"]).resolve())
        self.dataset_dir = str(
            Path(config_dict.get("dataset_dir", "datasets/")).resolve()
        )
        self.job_name = config_dict.get("job_name", "benchmark")
        self.n_categories = config_dict["n_categories"]
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
        self.unet_bottleneck_dim = config_dict["unet_bottleneck_dim"]
        self.unet_layers = self.problem_scale - self.unet_bottleneck_dim
        self.n_fracts_per_vol = config_dict["n_fracts_per_vol"]
        self.n_instances_used_per_fractal = config_dict["n_instances_used_per_fractal"]
        self.scale = 1
        self.batch_size = config_dict["batch_size"]
        self.dataloader_num_workers = config_dict["dataloader_num_workers"]
        self.epochs = config_dict["epochs"]
        self.optimizer = config_dict["optimizer"]
        self.disable_scheduler = bool(config_dict["disable_scheduler"])
        self.more_determinism = bool(config_dict["more_determinism"])
        self.datagen_from_scratch = bool(config_dict["datagen_from_scratch"])
        self.train_from_scratch = bool(config_dict["train_from_scratch"])
        self.val_split = config_dict["val_split"]
        self.seed = config_dict["seed"]
        self.dist = bool(config_dict["dist"])
        self.framework = config_dict["framework"]
        self.learning_rate = config_dict["learning_rate"]
        self.variance_threshold = config_dict["variance_threshold"]
        self.torch_amp = bool(config_dict["torch_amp"])
        self.loss_freq = config_dict["loss_freq"]
        self.checkpoint_dir = config_dict["checkpoint_dir"]
        self.normalize = config_dict["normalize"]
        self.warmup_batches = config_dict.get("warmup_batches")
        self.dataset_reuse_enforce_commit_id = config_dict[
            "dataset_reuse_enforce_commit_id"
        ]
        self.target_dice = config_dict["target_dice"]
        self.checkpoint_interval = config_dict["checkpoint_interval"]
        self.dc_num_shards = config_dict["dc_num_shards"]
        self.dc_shard_dims = config_dict["dc_shard_dims"]
        self.dc_total_shards = math.prod(self.dc_num_shards)
        # Safety Check: Length mismatch
        if len(self.dc_num_shards) != len(self.dc_shard_dims):
            raise ValueError(
                f"Configuration Mismatch: num_shards {self.dc_num_shards} "
                f"must have same length as shard_dim {self.dc_shard_dims}"
            )


class RunConfig(Config):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.run_dir = config_dict["run_dir"]
        self.run_iter = config_dict["run_iter"]


def load_config(file_path: str, config_type: str):
    """
    Load run config from yaml file

    Returns:
        Config: A Config instance with settings loaded from the yaml file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file '{file_path}' not found")

    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)

    if config_type == "sweep":
        return Config(config_dict)
    elif config_type == "run":
        return RunConfig(config_dict)
    else:
        raise ValueError(
            f"Invalid config type specified: {type}. Must be either 'sweep' or 'run'"
        )
