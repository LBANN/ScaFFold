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
import logging
import os
import shutil
import time
from math import ceil
from pathlib import Path

import numpy as np
from mpi4py import MPI

from ScaFFold.datagen.generate_fractal_points import generate_fractal_points
from ScaFFold.datagen.rng import derive_seed, seed_numba
from ScaFFold.utils.config_utils import Config
from ScaFFold.utils.utils import setup_mpi_logger

DEFAULT_NP_DTYPE = np.float64

logger = logging.getLogger(__name__)

# Temp files written during an atomic save carry this prefix so the six-digit
# resume glob (``NNNNNN_NNNN.npy``) can never mistake a half-written file for a
# finished instance.
TEMP_PREFIX = ".tmp-"

# Factors that pull a weight vector's deviation from 1.0 progressively toward
# the unweighted parameters a category was validated against. If the requested
# weights turn a contractive map expansive, these recover a finite point cloud
# while sacrificing as little of the intended variation as possible.
_WEIGHT_ATTENUATION = (0.5, 0.25, 0.1)

comm = None
rank = None
size = None


def generate_single_instance(
    pointcloud_point_num: int, params: np.array
) -> tuple[np.array, bool]:
    """
    Generate a single fractal instance.

    Parameters
    ----------
    pointcloud_point_num : int
        An int for the number of points to generate in the fractal point cloud.
    params : np.array
        A numpy array containing IFS parameters for this category.

    Returns
    -------
    points : np.array
        The generated fractal point cloud.
    valid : bool
        True when the point cloud stayed bounded (no runaway divergence) and
        every coordinate is finite; False when the caller should reject it.
    """

    # Generate points in the fractal
    points, runaway_check_pass = generate_fractal_points(params, pointcloud_point_num)

    # A runaway (diverging) map or any non-finite coordinate makes this instance
    # unusable downstream: voxelization would cast NaN/inf to garbage indices.
    valid = bool(runaway_check_pass) and bool(np.isfinite(points).all())

    return points, valid


def generate_instance_points(
    point_num: int,
    base_params: np.array,
    weights: np.array,
    category: int,
    instance: int,
    seed: int,
) -> np.array:
    """
    Generate a validated, weighted fractal point cloud for one instance.

    The requested ``weights`` scale the IFS coefficients to add variation, but a
    weight can turn a category that was validated as contractive at weight 1.0
    into a diverging map, producing non-finite points. When that happens the
    weight deviation from 1.0 is attenuated and the attempt retried; if every
    attenuated attempt still fails, generation falls back to the unweighted
    parameters (weight 1.0), which were validated at category-search time, so
    every instance slot fills deterministically with a finite point cloud.

    Parameters
    ----------
    point_num : int
        Number of points to generate.
    base_params : np.array
        The unweighted IFS parameters for this category.
    weights : np.array
        The per-instance weight vector applied to columns 0-11.
    category, instance : int
        Identify the work item, used for the per-item seed and log messages.
    seed : int
        The configured base seed.

    Returns
    -------
    points : np.array
        A finite fractal point cloud.
    """

    # Seed numba's internal RNG per item so the point cloud is reproducible for a
    # given (seed, category, instance) regardless of retries below (each attempt
    # re-seeds identically, so the fallback result is itself deterministic).
    def _attempt(scaled_weights: np.array) -> tuple[np.array, bool]:
        params = base_params.copy()
        params[:, :12] *= scaled_weights
        seed_numba(derive_seed(seed, category, instance))
        return generate_single_instance(point_num, params)

    points, valid = _attempt(weights)
    if valid:
        return points

    # Pull the weights toward 1.0 (no variation) in stages before giving up.
    for factor in _WEIGHT_ATTENUATION:
        attenuated = 1.0 + (weights - 1.0) * factor
        points, valid = _attempt(attenuated)
        if valid:
            logger.warning(
                "Attenuated weights (factor %.2f) for non-finite instance "
                "(category %d, instance %d)",
                factor,
                category,
                instance,
            )
            return points

    # Final fallback: the unweighted parameters, validated at search time.
    points, valid = _attempt(np.ones_like(weights))
    if not valid:
        # The unweighted category itself should have been validated; surface the
        # inconsistency rather than saving corrupt data.
        raise ValueError(
            f"Unweighted parameters for category {category} produced a "
            f"non-finite point cloud (instance {instance})"
        )
    logger.warning(
        "Fell back to unweighted parameters for non-finite instance "
        "(category %d, instance %d)",
        category,
        instance,
    )
    return points


def save_instance_atomic(out_path: Path, points: np.array) -> None:
    """
    Write a point cloud to ``out_path`` atomically.

    The array is written to a temp file (a name the resume glob cannot match),
    flushed and fsynced, then ``os.replace``d onto the final name, so a job
    killed mid-write never leaves a truncated file under a name resume accepts.
    """
    tmp_path = out_path.parent / f"{TEMP_PREFIX}{out_path.name}"
    try:
        with open(tmp_path, "wb") as handle:
            np.save(handle, points)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
    except BaseException:
        # A failed or interrupted write must not leave a temp file behind, and
        # the final name must keep whatever complete file was already there.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _is_valid_npy(path: str) -> bool:
    """Return True if ``path`` is a readable .npy file.

    A truncated file left by a killed non-atomic write loads far enough to read
    the header but raises when the array data is memory-mapped; treating that as
    invalid lets the resume logic regenerate it instead of accepting it forever.
    """
    try:
        np.load(path, mmap_mode="r")
        return True
    except (ValueError, EOFError, OSError):
        return False


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
    log = setup_mpi_logger(__file__, getattr(config, "verbose", 0))

    # Each instance is seeded individually inside the generation loop from
    # (config.seed, category, instance), so its content is reproducible and
    # independent of MPI world size, rank assignment, and resume state. A single
    # per-rank seed here would make an instance depend on how the work list was
    # partitioned, which shifts with world size and with pre-existing files.

    log.info("MPI size = %s", size)

    # Setup directories
    fracts_sub_dir = f"var{config.variance_threshold}"
    fracts_read_dir = os.path.join(config.fract_base_dir, fracts_sub_dir, "3DIFS_param")
    instance_write_dir = os.path.join(
        config.fract_base_dir, fracts_sub_dir, "instances", f"np{config.point_num}"
    )
    if rank == 0:
        log.info(
            "Generating instances for num_points=%s, writing to %s",
            config.point_num,
            instance_write_dir,
        )
        if os.path.exists(instance_write_dir) and config.datagen_from_scratch:
            log.info("Removing existing instances directory")
            shutil.rmtree(instance_write_dir)
        os.makedirs(instance_write_dir, exist_ok=True)

    # Wait until dir setup completes
    comm.Barrier()

    # Remove stale temp files from any previous interrupted run. These carry a
    # dotted prefix the six-digit resume glob below cannot match, but cleaning
    # them keeps the directory tidy and reclaims space before regeneration.
    if rank == 0:
        for stale in glob.glob(f"{instance_write_dir}/*/{TEMP_PREFIX}*"):
            try:
                os.remove(stale)
            except OSError:
                pass
    comm.Barrier()

    # Build the global work list on rank 0 alone and broadcast it, so every
    # rank slices an identical list. Scanning the shared filesystem
    # independently per rank lets divergent views (stale metadata caches after
    # the rmtree above, a partially-completed prior run, or a concurrent job)
    # produce lists that differ across ranks; block-slicing divergent lists then
    # assigns some (category, instance) pairs to two ranks (duplicate writes to
    # one file) and others to none (instances that are never generated).
    instances_to_generate = None
    if rank == 0:
        existing_instance_dirs = glob.glob(
            f"{instance_write_dir}/[0-9][0-9][0-9][0-9][0-9][0-9]/"
        )
        fracts_with_existing_instances = [
            int(path_str.split("/")[-2]) for path_str in existing_instance_dirs
        ]
        all_existing_instances = set()

        # Construct the set of existing instances as (category, instance) pairs.
        # Each candidate is opened via a memory-mapped load: a file truncated by
        # a killed job reads its header but fails here, so it is treated as
        # missing (and removed) rather than accepted as complete and left to
        # crash volumegen.
        for category in fracts_with_existing_instances:
            for path_str in glob.glob(
                f"{instance_write_dir}/{category:06d}/[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].npy"
            ):
                if _is_valid_npy(path_str):
                    instance = int(path_str.split("_")[-1].split(".")[0])
                    all_existing_instances.add((category, instance))
                else:
                    logger.warning(
                        "Discarding unreadable instance file %s; it will be regenerated",
                        path_str,
                    )
                    try:
                        os.remove(path_str)
                    except OSError:
                        pass

        # Diff the desired pairs against what already exists; membership is an
        # O(1) set lookup so the diff stays linear in the number of desired
        # pairs.
        instances_to_generate = [
            [category, instance]
            for category in range(num_categories)
            for instance in range(145)
            if (category, instance) not in all_existing_instances
        ]

    # Every rank slices the same broadcast list.
    instances_to_generate = comm.bcast(instances_to_generate, root=0)

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

        # Generate a validated, weighted point cloud. Weighting can turn a
        # category validated at weight 1.0 into a diverging map; this attenuates
        # the weights or falls back to weight 1.0 so the saved cloud is finite.
        points = generate_instance_points(
            config.point_num,
            params,
            weights,
            category,
            instance,
            config.seed,
        )

        # Force point_data to be contiguous
        points_contiguous = np.ascontiguousarray(points, dtype=DEFAULT_NP_DTYPE)

        # Construct the output path
        out_dir = Path(instance_write_dir) / f"{category:06d}"
        filename = f"{category:06d}_{instance:04d}.npy"

        # Ensure parent directory exists
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save array atomically so a killed job never leaves a truncated file
        # under a name the resume glob would accept as complete.
        save_instance_atomic(out_dir / filename, points_contiguous)

    end_time = time.time()
    total_time = end_time - start_time
    log.info(
        "Generated %s instances in %.2f seconds",
        len(instances_to_generate),
        total_time,
    )

    return 0
