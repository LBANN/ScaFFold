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

# -*- coding: utf-8 -*-
"""
@original author: ryosuke yamada
"""

import glob
import os
import shutil
import time
from math import ceil
from pathlib import Path

import numpy as np
from mpi4py import MPI

from ScaFFold.datagen.generate_fractal_points import generate_fractal_points
from ScaFFold.utils.config_utils import Config

DEFAULT_NP_DTYPE = np.float64

comm = None
rank = None
size = None


def generate_single_instance(pointcloud_point_num: int, params: np.array) -> np.array:
    """
    Generate a single fractal instance.

    Parameters
    ----------
    config.point_num : int
        An int for the number of points to generate in the fractal point cloud.
    params : np.array
        A numpy array containing IFS parameters for this category.
    """

    # Generate points in the fractal
    points, runaway_check_pass = generate_fractal_points(params, pointcloud_point_num)

    # Return result
    return points


def main(config: Config):
    """
    Generate fractal instances.

    Parameters
    ----------
    config : Config
        A Config object containing run parameters.
    """

    num_categories = config.n_categories

    global comm, rank, size
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # FIXME anything else to ensure determinism?
    np.random.seed(config.seed + rank)

    if rank == 0:
        print(f"MPI size = {size}")

    # Setup directories
    repo_src_path = config.library_root
    fracts_sub_dir = f"var{config.variance_threshold}"
    fracts_read_dir = f"{repo_src_path}/fractals/{fracts_sub_dir}/3DIFS_param"
    instance_write_dir = (
        f"{repo_src_path}fractals/{fracts_sub_dir}/instances/np{config.point_num}"
    )
    if rank == 0:
        print(
            f"Generating instances for num_points={config.point_num}, writing to {instance_write_dir}"
        )
        if os.path.exists(instance_write_dir) and config.datagen_from_scratch:
            print("Removing existing instances dir")
            shutil.rmtree(instance_write_dir)
        os.makedirs(instance_write_dir, exist_ok=True)

    # Wait until dir setup completes
    comm.Barrier()

    # Parse existing instances to get list of what we have
    existing_instance_dirs = glob.glob(
        f"{instance_write_dir}/[0-9][0-9][0-9][0-9][0-9][0-9]/"
    )
    fracts_with_existing_instances = [
        int(path_str.split("/")[-2]) for path_str in existing_instance_dirs
    ]
    all_existing_instances = []

    # Construct list of existing instances in [category, instance] pairs
    for category in fracts_with_existing_instances:
        existing_instances = [
            int(path_str.split("_")[-1].split(".")[0])
            for path_str in glob.glob(
                f"{instance_write_dir}/{category:06d}/[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].ply"
            )
        ]
        category_instance_pairs = [
            [category, instance] for instance in existing_instances
        ]
        all_existing_instances.extend(category_instance_pairs)

    # Create list of [category, instance] pairs we want to end up with
    instances_desired = []
    for category in range(num_categories):
        instances_desired_for_this_category = list(range(145))
        category_instance_pairs = [
            [category, instance] for instance in instances_desired_for_this_category
        ]
        instances_desired.extend(category_instance_pairs)

    # Diff of the above is what we need to generate
    instances_to_generate = [
        pair for pair in instances_desired if pair not in all_existing_instances
    ]

    # Distribute work among ranks
    instances_per_rank = ceil(len(instances_to_generate) / size)
    start = rank * instances_per_rank
    end = min(((rank + 1) * instances_per_rank), len(instances_to_generate))
    instances_to_generate_for_this_rank = instances_to_generate[start:end]

    # Load the fractal category IFS parameters
    IFS_param_csv_names = os.listdir(fracts_read_dir)
    IFS_param_csv_names.sort()

    # Load the weights, which will be applied during fractal generation
    # to produce more variation in the dataset
    weights_location = os.path.join(
        os.path.dirname(__file__), "../package_data/weights_ins145.csv"
    )
    weights_all = np.genfromtxt(weights_location, dtype=DEFAULT_NP_DTYPE, delimiter=",")

    start_time = time.time()

    for i, category_instance_pair in enumerate(instances_to_generate_for_this_rank):
        category, instance = category_instance_pair
        category_IFS_params = IFS_param_csv_names[category]
        params = np.genfromtxt(
            f"{fracts_read_dir}/{category_IFS_params}",
            dtype=DEFAULT_NP_DTYPE,
            delimiter=",",
        )
        weights = weights_all[instance]

        # Apply weights
        params[:, :12] *= weights

        # Generate points
        points = generate_single_instance(config.point_num, params)

        # Force point_data to be contiguous
        points_contiguous = np.ascontiguousarray(points, dtype=DEFAULT_NP_DTYPE)

        # Construct the output path
        out_dir = Path(instance_write_dir) / f"{category:06d}"
        filename = f"{category:06d}_{instance:04d}.npy"

        # Ensure parent directory exists
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save array to out_dir
        np.save(out_dir / filename, points_contiguous)

    end_time = time.time()
    total_time = end_time - start_time
    if rank == 0:
        print(
            f"Generated {len(instances_to_generate)} instances in {total_time:.2f} seconds"
        )

    return 0
