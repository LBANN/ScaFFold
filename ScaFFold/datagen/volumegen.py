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

import logging
import math
import os
import pickle
import random
import sys
import time
from math import ceil
from typing import Dict

import numpy as np
from mpi4py import MPI

from ScaFFold.utils.config_utils import Config
from ScaFFold.utils.data_types import DEFAULT_NP_DTYPE, MASK_DTYPE, VOLUME_DTYPE


def load_np_ptcloud(path: str) -> np.ndarray:
    """
    Read a .npy file and return an (N,3) array of dtype float64.
    """
    pts = np.load(path)
    return pts.astype(DEFAULT_NP_DTYPE, copy=False)


def points_to_voxelgrid(
    points: np.ndarray, grid_size: int, eps: float = 1e-6
) -> np.ndarray:
    """
    Convert an (N,3) float64 point cloud directly into a boolean voxel grid
    of shape (grid_size, grid_size, grid_size).
    """
    # 1) Axis‐aligned bounding box in float64
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    # 2) Voxel size per dimension (float64)
    voxel_size = (maxs - mins + eps) / grid_size

    # 3) Map points into [0,grid_size) indices
    scaled = (points - mins) / voxel_size
    idx = np.floor(scaled).astype(int)

    # 4) Clip to valid range
    idx = np.clip(idx, 0, grid_size - 1)

    # 5) Scatter into a boolean grid
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    return grid


def main(config: Dict):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dataset_dir = str(config.dataset_dir)

    vol_path = os.path.join(dataset_dir, "volumes")
    mask_path = os.path.join(dataset_dir, "masks")
    volumes_contents_path = os.path.join(dataset_dir, "volumes_contents.csv")

    n_fracts_per_vol = config.n_fracts_per_vol

    random.seed(config.seed)  # Python
    np.random.seed(config.seed)  # NumPy

    # Set up directories and select instances from each category
    volumes_contents = None

    if rank == 0:
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        for subdir in ["training", "validation"]:
            os.makedirs(os.path.join(vol_path, subdir), exist_ok=True)
            os.makedirs(os.path.join(mask_path, subdir), exist_ok=True)

        # Force n_instances_used_per_fractal to be multiple of n_fracts_per_vol
        if config.n_instances_used_per_fractal % n_fracts_per_vol != 0:
            print(
                f"volumegen.py: WARNING: n_instances_used_per_fractal ({config.n_instances_used_per_fractal}) \n"
                f"NOT multiple of n_fracts_per_vol={n_fracts_per_vol}. Rounding down."
            )
            config.n_instances_used_per_fractal = (
                config.n_instances_used_per_fractal
                // n_fracts_per_vol
                * n_fracts_per_vol
            )

        # Randomly select n_instances_used_per_fractal instances from each fractal class.
        instances_list = []
        for category in range(config.n_categories):
            instances_remaining = config.n_instances_used_per_fractal
            random_instances = []
            while instances_remaining > 0:
                random_instances.extend(
                    random.sample(range(145), min(145, instances_remaining))
                )
                instances_remaining -= min(145, instances_remaining)

            category_instance_pairs = [
                [category, instance] for instance in random_instances
            ]
            instances_list.extend(category_instance_pairs)

        instances_list = np.array(instances_list, dtype=int)
        np.random.shuffle(instances_list)

        volumes_contents = instances_list.reshape(-1, 2 * n_fracts_per_vol)

        indices = np.arange(volumes_contents.shape[0]).reshape(-1, 1)
        volumes_contents = np.hstack([indices, volumes_contents])

        with open(volumes_contents_path, "wb") as f:
            np.savetxt(f, volumes_contents.astype(int), fmt="%i", delimiter=",")
        print(
            f"volumegen.py({rank}): finished writing volumes_contents (shape = {volumes_contents.shape})"
        )

    # Broadcast to all ranks
    volumes_contents = comm.bcast(volumes_contents, root=0)

    # Determine train/val split globally so all ranks know where to save
    num_volumes = len(volumes_contents)
    random.seed(config.seed)  # Reset seed to ensure all ranks get same split
    val_indices = set(
        random.sample(range(num_volumes), int(num_volumes * config.val_split / 100))
    )

    # Work distribution
    num_volumes = len(volumes_contents)
    stride = ceil(num_volumes / size)
    start_idx = rank * stride
    end_idx = min(((rank + 1) * stride), num_volumes)

    if start_idx >= end_idx:
        logging.info(f"Rank {rank} given no volumes to generate")

    else:
        volumes_contents_subset = volumes_contents[start_idx:end_idx]
        print(
            f"rank {rank} responsible for volumes {volumes_contents_subset[0][0]} through {volumes_contents_subset[-1][0]}"
        )

        np.random.seed(config.seed)
        fractal_colors = np.random.rand(max(config.n_categories, n_fracts_per_vol), 3)

        grid_size = math.floor(config.vol_size * config.scale)
        library_root = str(config.library_root)

        # Generation loop
        start_time = time.time()
        for i, curr_vol in enumerate(volumes_contents_subset):
            if i % 10 == 0:
                logging.info(f"Rank {rank} processing local volume {i}...")

            volume = np.full(
                (config.vol_size, config.vol_size, config.vol_size, 3),
                0,
                dtype=VOLUME_DTYPE,
            )
            mask = np.full(
                (config.vol_size, config.vol_size, config.vol_size), 0, dtype=MASK_DTYPE
            )

            global_vol_idx = curr_vol[0]
            vol_seed = config.seed + int(global_vol_idx)
            random.seed(vol_seed)
            np.random.seed(vol_seed)

            for curr_fract in range(n_fracts_per_vol):
                curr_category = curr_vol[1 + 2 * curr_fract]
                curr_instance = curr_vol[1 + 2 * curr_fract + 1]
                fractal_color = fractal_colors[curr_category]

                instances_dir = (
                    f"var{config.variance_threshold}/instances/np{config.point_num}"
                )

                point_cloud_path = os.path.join(
                    library_root,
                    "fractals",
                    instances_dir,
                    f"{curr_category:06d}",
                    f"{curr_category:06d}_{curr_instance:04d}.npy",
                )

                if not os.path.exists(point_cloud_path):
                    print(
                        f"File {point_cloud_path} does not exist. Ensure you have run 'scaffold generate_fractals ...'"
                    )
                    sys.exit(1)

                points = load_np_ptcloud(point_cloud_path)
                mask3d = points_to_voxelgrid(points, grid_size)

                assert mask3d.shape == volume.shape[:3], (
                    f"mask3d {mask3d.shape} != volume spatial dims {volume.shape[:3]}"
                )

                volume[mask3d] = fractal_color
                mask[mask3d] = curr_category + 1

            # Determine destination folder
            subdir = "validation" if global_vol_idx in val_indices else "training"
            # Tensors must logically be channels-first, later we will change striding/storage to channels-last on GPU (metadata will always stay channels-first).
            volume_channels_first = volume.transpose((3, 0, 1, 2))
            volume_to_save = np.ascontiguousarray(
                volume_channels_first, dtype=VOLUME_DTYPE
            )
            mask_to_save = np.ascontiguousarray(mask, dtype=MASK_DTYPE)

            vol_file = os.path.join(vol_path, subdir, f"{global_vol_idx}.npy")
            with open(vol_file, "wb") as f:
                np.save(f, volume_to_save)

            mask_file = os.path.join(mask_path, subdir, f"{global_vol_idx}_mask.npy")
            with open(mask_file, "wb") as f:
                np.save(f, mask_to_save)

        end_time = time.time()
        total_time = end_time - start_time
        if rank == 0:
            print(
                f"Rank 0 generated {len(volumes_contents_subset)} volumes in {total_time:.2f} seconds | {len(volumes_contents_subset) / total_time:.2f} volumes per second"
            )

    # Barrier to ensure all ranks are finished writing
    comm.Barrier()

    if rank == 0:
        print(f"volumegen.py({rank}): All ranks done. Proceeding to split.")

    # Do the train/val split and generate lists of unique train/val masks
    if rank == 0:
        print("volumegen.py: volume gen COMPLETE. Now generating unique mask lists")

        # Directories are already created at start of script

        # Reconstruct lists for unique mask value scanning
        val_files = sorted(list(val_indices))
        train_files = sorted(list(set(range(num_volumes)) - val_indices))

        print(
            f"volumegen.py({rank}): len(val_files)={len(val_files)}, len(train_files)={len(train_files)}."
        )

        # Save lists of unique train and val mask values
        print(
            f"volumegen.py({rank}): calculating unique mask values from configuration (no file read)"
        )

        # volumes_contents layout is [vol_idx, cat1, inst1, cat2, inst2, ...]
        # We want the categories, which are at indices 1, 3, 5, etc.
        cat_cols = slice(1, None, 2)

        # Process unique train mask values
        # Extract rows corresponding to train files
        train_rows = volumes_contents[train_files]
        # Extract only the category columns and flatten to a 1D array
        train_cats = train_rows[:, cat_cols].flatten()
        # Create set of unique labels: (category + 1) and background (0)
        unique_train = set(train_cats + 1)
        unique_train.add(0)

        unique_train = sorted(list(unique_train))
        unique_train_file = f"{dataset_dir}/train_unique_mask_vals"
        with open(unique_train_file, "wb") as out:
            pickle.dump({"mask_values": unique_train}, out)

        # Process unique val mask values (same logic)
        val_rows = volumes_contents[val_files]
        val_cats = val_rows[:, cat_cols].flatten()
        unique_val = set(val_cats + 1)
        unique_val.add(0)

        unique_val = sorted(list(unique_val))
        unique_val_file = f"{dataset_dir}/val_unique_mask_vals"
        with open(unique_val_file, "wb") as out:
            pickle.dump({"mask_values": unique_val}, out)


if __name__ == "__main__":
    import yaml

    with open("run_config.yaml") as f:
        config_dict = yaml.full_load(f)
    config = Config(config_dict)
    main(config)
