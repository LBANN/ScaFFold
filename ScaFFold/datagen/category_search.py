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
from ScaFFold.utils.utils import setup_mpi_logger

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
    (not value_check_pass) : bool
        A bool for whether this attempt passed the NaN/non-finite check.
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

    # Sum number of NaNs and reject infinities before normalization.
    nan_count = np.isnan(points).sum()
    value_check_pass = nan_count == 0 and np.isfinite(points).all()
    variance_check_pass = False

    if value_check_pass:
        # Normalize + center
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        means = points.mean(axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            ranges = maxs - mins
        value_check_pass = np.all(np.isfinite(ranges)) and np.all(ranges > 0)
        if value_check_pass:
            scales = (2 * config.normalize) / ranges
            with np.errstate(over="ignore", invalid="ignore"):
                points = (points - means) * scales

            value_check_pass = np.isfinite(points).all()
            if value_check_pass:
                # Calc dimension-wise variance and compare to threshold
                points_variance = np.var(points, axis=0)
                variance_check_pass = np.all(
                    points_variance > config.variance_threshold
                )
        if variance_check_pass and value_check_pass and runaway_check_pass:
            valid = True

    # Return result
    return (
        valid,
        params,
        bool(not value_check_pass),
        bool(value_check_pass and not variance_check_pass),
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
        The number of attempts in this batch which failed the NaN/non-finite check.
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
    log = setup_mpi_logger(__file__, getattr(config, "verbose", 0))

    datagen_batch_size = int(getattr(config, "datagen_batch_size", 10000))
    if datagen_batch_size <= 0:
        raise ValueError("datagen_batch_size must be positive")

    # FIXME anything else to ensure determinism?
    np.random.seed(config.seed + rank)

    log.info("MPI size = %s", size)

    # Setup directories
    fracts_sub_dir = f"var{config.variance_threshold}"
    fracts_write_dir = os.path.join(
        config.fract_base_dir, fracts_sub_dir, "3DIFS_param"
    )
    if rank == 0:
        log.info("Writing fractals to %s", fracts_write_dir)
        if os.path.exists(fracts_write_dir) and config.datagen_from_scratch:
            log.info("Removing existing fractals directory")
            shutil.rmtree(fracts_write_dir)
        os.makedirs(fracts_write_dir, exist_ok=True)

    # Wait until dir setup completes
    comm.Barrier()

    # Calculate number of remaining fractal categories to generate
    existing_categories = len(glob.glob(f"{fracts_write_dir}/*.csv"))
    categories_remaining = config.n_categories - existing_categories
    if rank == 0:
        log.info(
            "category_search found %s existing fractal categories | %s needed | "
            "%s remaining",
            existing_categories,
            config.n_categories,
            max(0, categories_remaining),
        )

    rank_start_time = time.time()

    attempts = 0
    nan_fail_count = 0
    var_fail_count = 0
    runaway_fail_count = 0
    while categories_remaining > 0:
        attempts += datagen_batch_size * size

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
            log.info(
                "cat_remaining = %s | total attempts = %s | stats for rank 0: "
                "invalid_value_fail_count = %s, var_fail_count = %s, "
                "runaway_fail_count = %s",
                categories_remaining,
                attempts,
                nan_fail_count,
                var_fail_count,
                runaway_fail_count,
            )
            if len(params_valid) > 0:
                log.info(
                    "Processing %s valid param sets from this batch",
                    len(params_valid),
                )
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
                    log.info(
                        "Generated all fractal categories needed. Ignoring additional "
                        "valid categories."
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
        categories_generated = config.n_categories - existing_categories
        log.info(
            "Generated %s new categories in %s total attempts | %.2f attempts per "
            "category | total categories is now %s",
            categories_generated,
            attempts,
            attempts / categories_generated,
            config.n_categories,
        )
        log.info(
            "Failures experienced: %s invalid-value attempts (%.4f%%), %s variance-fail "
            "attempts (%.4f%%), %s runaway attempts (%.4f%%)",
            global_nan_fail_count,
            100 * global_nan_fail_count / attempts,
            global_var_fail_count,
            100 * global_var_fail_count / attempts,
            global_runaway_fail_count,
            100 * global_runaway_fail_count / attempts,
        )
        log.info(
            "Rank 0 wall time = %.2f | total CPU time = %.2f | avg wall time per "
            "rank = %.2f | %.2f total attempts per wall second | %.2f attempts "
            "per wall second per rank",
            rank_total_time,
            global_sum_time,
            global_sum_time / size,
            attempts / rank_total_time,
            attempts / rank_total_time / size,
        )

    return 0
