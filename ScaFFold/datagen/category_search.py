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

import numpy as np
from mpi4py import MPI

from ScaFFold.datagen.generate_fractal_points import generate_fractal_points
from ScaFFold.utils.config_utils import Config

DEFAULT_NP_DTYPE = np.float64

comm = None
rank = None
size = None


def generate_single_category(config: Config) -> tuple[bool, np.array, bool, bool, bool]:
    """
    Generate a single fractal category.

    Parameters
    ----------
    config : Config
        A Config object containing run parameters.

    Returns
    -------
    valid : bool
        A bool for whether a valid category was found on this attempt.
    params : np.array
        A numpy array containing IFS parameters for this category attempt, if attempt was valid.
    (not nan_check_pass) : bool
        A bool for whether this attempt passed the NaN check.
    (not variance_check_pass) : bool
        A bool for whether this attempt passed the variance check.
    (not runaway_check_pass) : bool
        A bool for whether this attempt passed the runaway values check.
    """

    # Bool for whether this category is valid after checks
    valid = False

    # Generate random params
    params = np.random.uniform(-1.0, 1.0, (2, 13)).astype(DEFAULT_NP_DTYPE)

    # Calculate normalized probabilities, then store in last params of each transformation
    rotation_matrices = params[:, 0:9].reshape(-1, 3, 3)
    probabilities_raw = np.absolute(np.linalg.det(rotation_matrices))
    probabilties_normalized = probabilities_raw / np.sum(probabilities_raw)
    params[:, -1] = probabilties_normalized

    # Generate points in the fractal
    points, runaway_check_pass = generate_fractal_points(
        params,
        (
            config.point_num
            if isinstance(config.point_num, int)
            else int(config.point_num)
        ),
    )

    # Sum number of NaNs
    nan_count = np.isnan(points).sum()
    nan_check_pass = nan_count == 0
    variance_check_pass = False

    if nan_check_pass:
        # Normalize + center
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        means = points.mean(axis=0)
        scales = (2 * config.normalize) / (maxs - mins)
        points = (points - means) * scales

        # Calc dimension-wise variance and compare to threshold
        points_variance = np.var(points, axis=1)
        variance_check_pass = np.all(points_variance > config.variance_threshold)
        if variance_check_pass and nan_check_pass and runaway_check_pass:
            valid = True

    # Return result
    return (
        valid,
        params,
        not nan_check_pass,
        not variance_check_pass,
        not runaway_check_pass,
    )


def generate_categories_batch(
    config: Config, datagen_batch_size: int = 1
) -> tuple[bool, np.array, int, int, int]:
    """
    Run a batch of fractal category generation attempts.

    Parameters
    ----------
    config : Config
        A Config object containing run parameters.
    datagen_batch_size : int
        An int for the number of attempts to run before MPI sync between ranks.

    Returns
    -------
    one_or_more_valid : bool
        A bool for whether at least one valid category was found in this batch of attempts.
    params : np.array
        A numpy array containing IFS parameters for this category attempt, if attempt was valid.
    failed_nan_check_count : int
        The number of attempts in this batch which failed the nan check.
    failed_var_check_count : int
        The number of attempts in this batch which failed the var check.
    runaway_failure_count : int
        The number of attempts in this batch which failed the runaway values check.
    """
    one_or_more_valid = False
    params_list = []
    failed_nan_check_count = 0
    failed_var_check_count = 0
    runaway_failure_count = 0

    for _ in range(datagen_batch_size):
        (
            attempt_valid,
            params,
            attempt_failed_nan_check,
            attempt_failed_var_check,
            runaway_failure,
        ) = generate_single_category(config)
        if attempt_valid:
            one_or_more_valid = True
            params_list.append(params)
        failed_nan_check_count += attempt_failed_nan_check
        failed_var_check_count += attempt_failed_var_check
        runaway_failure_count += runaway_failure

    return (
        one_or_more_valid,
        params_list,
        failed_nan_check_count,
        failed_var_check_count,
        runaway_failure_count,
    )


def main(config: Config) -> None:
    """
    Generate fractal categories.

    Fractal category generation works as follows:
    1. Randomly generate a set of IFS parameters representing a fractal category.
    2. Use the IFS parameters to generate a cloud of points.
    3. Perform a series of checks on the point cloud. If all checks pass, accept
        this IFS as a valid fractal category and write the parameters to a file.

    Parameters
    ----------
    config : Config
        A Config object containing run parameters.
    """

    global comm, rank, size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    datagen_batch_size = 10000

    # FIXME anything else to ensure determinism?
    np.random.seed(config.seed + rank)

    if rank == 0:
        print(f"MPI size = {size}")

    # Setup directories
    fracts_sub_dir = f"var{config.variance_threshold}"
    fracts_write_dir = os.path.join(
        config.fract_base_dir, fracts_sub_dir, "3DIFS_param"
    )
    if rank == 0:
        print(f"Writing fractals to {fracts_write_dir}")
        if os.path.exists(fracts_write_dir) and config.datagen_from_scratch:
            print("Removing existing fractals dir")
            shutil.rmtree(fracts_write_dir)
        os.makedirs(fracts_write_dir, exist_ok=True)

    # Wait until dir setup completes
    comm.Barrier()

    # Calculate number of remaining fractal categories to generate
    existing_categories = len(glob.glob(f"{fracts_write_dir}/*.csv"))
    categories_remaining = config.n_categories - existing_categories
    if rank == 0:
        print(
            f"category_search found {existing_categories} existing fractal categories | {config.n_categories} needed | {max(0, categories_remaining)} remaining"
        )

    rank_start_time = time.time()

    attempts = 0
    nan_fail_count = 0
    var_fail_count = 0
    runaway_fail_count = 0
    while categories_remaining > 0:
        attempts += size

        # Each rank attempts to generate datagen_batch_size categories
        (
            valid,
            params_list,
            attempts_failed_nan_check,
            attempts_failed_var_check,
            attempts_runaway_failures,
        ) = generate_categories_batch(config, datagen_batch_size)
        nan_fail_count += attempts_failed_nan_check
        var_fail_count += attempts_failed_var_check
        runaway_fail_count += attempts_runaway_failures

        # Gather results on rank 0
        data = params_list if valid else []
        gathered_params = comm.gather(data, root=0)

        # Process IFS params one at a time, writing each to a CSV
        if rank == 0:
            params_valid = [item for sublist in gathered_params for item in sublist]
            if attempts % 10000 * size / datagen_batch_size == 0:
                print(
                    f"cat_remaining = {categories_remaining} | total attempts = {attempts} | stats for rank 0: nan_fail_count = {nan_fail_count}, var_fail_count = {var_fail_count}, runaway_fail_count = {runaway_fail_count}"
                )
            if len(params_valid) > 0:
                print(f"Processing {len(params_valid)} param sets from this attempt")
            for p in params_valid:
                # Ensure we don't save more categories than needed
                if categories_remaining > 0:
                    # Save IFS params as new category
                    class_str = "%06d" % (config.n_categories - categories_remaining)
                    np.savetxt(
                        "{}/{}.csv".format(fracts_write_dir, class_str),
                        p,
                        delimiter=",",
                    )

                    # Update categories_remaining
                    categories_remaining -= 1
                else:
                    print(
                        "Generated all fractal categories needed. Ignoring additional found valid categories..."
                    )
                    break

        # Broadcast updated categories_remaining to all ranks
        categories_remaining = comm.bcast(categories_remaining, root=0)

        # Sync all ranks before proceeding
        comm.Barrier()

    rank_end_time = time.time()
    rank_total_time = rank_end_time - rank_start_time
    global_sum_time = comm.reduce(rank_total_time, op=MPI.SUM, root=0)
    global_nan_fail_count = comm.reduce(nan_fail_count, op=MPI.SUM, root=0)
    global_var_fail_count = comm.reduce(var_fail_count, op=MPI.SUM, root=0)
    global_runaway_fail_count = comm.reduce(runaway_fail_count, op=MPI.SUM, root=0)

    if rank == 0 and attempts > 0:
        print(
            f"Generated {config.n_categories - existing_categories} new categories in {attempts * datagen_batch_size} total attempts | {attempts * datagen_batch_size / (config.n_categories - existing_categories)} Attempts per category | Total categories is now {config.n_categories}"
        )
        print(
            f"Failures experienced: {global_nan_fail_count} nan attempts, {100 * global_nan_fail_count / (attempts * datagen_batch_size):.4f}% of all attempts, {global_var_fail_count} var fail attempts, {100 * global_var_fail_count / (attempts * datagen_batch_size):.4f}% of all attempts, {global_runaway_fail_count} runaway attempts, {100 * global_runaway_fail_count / (attempts * datagen_batch_size):.4f}% of all attempts"
        )
        print(
            f"Rank 0 wall time = {rank_total_time:.2f} | Total CPU time = {global_sum_time:.2f} | Avg wall time per rank {global_sum_time / size:.2f} | {attempts * datagen_batch_size / rank_total_time:.2f} total attempts per wall second | {attempts * datagen_batch_size / rank_total_time / size:.2f} attempts per wall second per rank"
        )

    return 0
