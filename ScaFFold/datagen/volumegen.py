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

import os
import pickle
import random
import time
from math import ceil
from typing import Dict

import numpy as np
from mpi4py import MPI

from ScaFFold.datagen import layout
from ScaFFold.utils.config_utils import Config
from ScaFFold.utils.data_types import MASK_DTYPE, VOLUME_DTYPE
from ScaFFold.utils.utils import setup_mpi_logger


def load_np_ptcloud(path: str) -> np.ndarray:
    """
    Read a .npy point cloud and return an (N, 3) float32 array.

    Coordinates are normalized to roughly [-1, 1] and only ever binned into
    integer voxel indices, so float32 carries ample precision while halving the
    read bandwidth this incurs on every rank. A legacy float64 file is downcast
    here so voxelization is dtype-consistent with freshly generated instances.
    """
    pts = np.load(path)
    return pts.astype(np.float32, copy=False)


def points_to_voxel_indices(
    points: np.ndarray, grid_size: int, eps: float = 1e-6
) -> np.ndarray:
    """
    Map an (N, 3) point cloud to the integer voxel indices it occupies.

    Returns a (K, 3) int array of the unique voxel coordinates on a
    ``grid_size**3`` grid. Scattering these straight into a volume/mask
    (``arr[idx[:, 0], idx[:, 1], idx[:, 2]] = value``) paints exactly the
    occupied voxels in O(K), avoiding a dense ``grid_size**3`` allocation and a
    full-volume boolean-mask traversal per cloud.

    Normalization is isotropic: all three axes are divided by the single
    largest extent and the result is centered in the grid, so a fractal's
    aspect ratio and relative size are preserved instead of every cloud being
    stretched to fill the cube on every axis.

    Non-finite input (NaN/inf coordinates, e.g. from a diverging IFS) is
    rejected: casting such coordinates to integer indices would otherwise yield
    platform-dependent garbage that ``np.clip`` silently forces in-bounds and
    scatters into the mask as a legitimate label.
    """
    if not np.isfinite(points).all():
        raise ValueError(
            "points_to_voxel_indices received non-finite coordinates "
            "(NaN or inf); refusing to rasterize corrupt point cloud"
        )

    # 1) Axis-aligned bounding box.
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    # 2) A single isotropic voxel size from the largest extent, so aspect ratio
    #    and relative scale survive rasterization.
    max_extent = float((maxs - mins).max())
    voxel_size = (max_extent + eps) / grid_size

    # 3) Map points into index space using the shared scale.
    scaled = (points - mins) / voxel_size

    # 4) Center the occupied region: the largest axis fills the grid while the
    #    shorter axes are offset so their span sits in the middle. The free
    #    space to split between the two margins is (grid_size - span) voxels,
    #    measured in the same voxel units as ``scaled``; subtracting an extra 1
    #    (as if the offset were an index rather than a length) shifted every
    #    cloud half a voxel toward the origin, floored the first half-voxel of
    #    each filled axis to -1, and let the clip below fold those points onto
    #    plane 0.
    span = scaled.max(axis=0)
    offset = (grid_size - span) / 2.0
    idx = np.floor(scaled + offset).astype(int)

    # 5) Clip to valid range (guards float rounding at the boundaries).
    idx = np.clip(idx, 0, grid_size - 1)

    # 6) Collapse coincident points to their shared voxel: the occupied set is
    #    what gets painted, so one index (and one write) per voxel suffices.
    return np.unique(idx, axis=0)


def points_to_voxelgrid(
    points: np.ndarray, grid_size: int, eps: float = 1e-6
) -> np.ndarray:
    """
    Convert an (N, 3) point cloud into a boolean voxel grid of shape
    (grid_size, grid_size, grid_size).

    Thin wrapper over :func:`points_to_voxel_indices` for callers that want a
    dense occupancy grid; the generation loop scatters the indices directly.
    """
    idx = points_to_voxel_indices(points, grid_size, eps)
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def resolve_grid_size(config) -> int:
    """
    Return the voxel-grid edge length, which must equal ``config.vol_size``.

    The volume and mask are allocated at ``vol_size``; the grid must match or
    the per-volume shape check fails. ``config.scale`` is not a working knob
    (it is fixed at 1 upstream), so any other value is a configuration error
    rather than a silently mismatched grid, and is rejected here with a clear
    message instead of tripping an opaque shape assertion deep in the loop.
    """
    scale = getattr(config, "scale", 1)
    if scale != 1:
        raise ValueError(
            f"config.scale must be 1 (got {scale}); scaled sub-volume "
            "rasterization is not supported"
        )
    return int(config.vol_size)


def main(config: Dict):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    log = setup_mpi_logger(__file__, getattr(config, "verbose", 0))

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
            log.warning(
                "n_instances_used_per_fractal (%s) is not a multiple of "
                "n_fracts_per_vol=%s. Rounding down.",
                config.n_instances_used_per_fractal,
                n_fracts_per_vol,
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
        log.info(
            "Finished writing volumes_contents with shape %s", volumes_contents.shape
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

    # Per-rank generation status. A rank that fails records the message here and
    # keeps executing the collective sequence below so every rank stays in step;
    # the failure is then propagated to all ranks via an allreduce.
    ok = True
    err = ""
    interrupt = None

    try:
        if start_idx >= end_idx:
            log.debug("Rank %s given no volumes to generate", rank)

        else:
            volumes_contents_subset = volumes_contents[start_idx:end_idx]
            log.debug(
                "Rank %s responsible for volumes %s through %s",
                rank,
                volumes_contents_subset[0][0],
                volumes_contents_subset[-1][0],
            )

            np.random.seed(config.seed)
            fractal_colors = np.random.rand(config.n_categories, 3)

            grid_size = resolve_grid_size(config)
            # The instance library is keyed by seed (see ScaFFold.datagen
            # .layout), so a volume can only ever be built from point clouds
            # this run's seed produced. Resolved once, outside the loop.
            instances_dir = layout.instance_dir(config)

            # Generation loop
            start_time = time.time()
            for i, curr_vol in enumerate(volumes_contents_subset):
                if i % 10 == 0:
                    log.debug("Rank %s processing local volume %s", rank, i)

                volume = np.full(
                    (config.vol_size, config.vol_size, config.vol_size, 3),
                    0,
                    dtype=VOLUME_DTYPE,
                )
                mask = np.full(
                    (config.vol_size, config.vol_size, config.vol_size),
                    0,
                    dtype=MASK_DTYPE,
                )

                global_vol_idx = curr_vol[0]
                vol_seed = config.seed + int(global_vol_idx)
                random.seed(vol_seed)
                np.random.seed(vol_seed)

                for curr_fract in range(n_fracts_per_vol):
                    curr_category = curr_vol[1 + 2 * curr_fract]
                    curr_instance = curr_vol[1 + 2 * curr_fract + 1]
                    fractal_color = fractal_colors[curr_category]

                    point_cloud_path = os.path.join(
                        instances_dir,
                        f"{curr_category:06d}",
                        f"{curr_category:06d}_{curr_instance:04d}.npy",
                    )

                    if not os.path.exists(point_cloud_path):
                        raise FileNotFoundError(
                            f"File {point_cloud_path} does not exist. "
                            "Ensure you have run 'scaffold generate_fractals ...'"
                        )

                    points = load_np_ptcloud(point_cloud_path)
                    voxel_idx = points_to_voxel_indices(points, grid_size)

                    assert voxel_idx.shape[1] == volume.ndim - 1, (
                        f"voxel index width {voxel_idx.shape[1]} != volume spatial "
                        f"dims {volume.ndim - 1}"
                    )

                    # Scatter only the occupied voxels: O(points) writes instead
                    # of two full-volume boolean-mask traversals per fractal.
                    rows, cols, depths = (
                        voxel_idx[:, 0],
                        voxel_idx[:, 1],
                        voxel_idx[:, 2],
                    )
                    volume[rows, cols, depths] = fractal_color
                    mask[rows, cols, depths] = curr_category + 1

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

                mask_file = os.path.join(
                    mask_path, subdir, f"{global_vol_idx}_mask.npy"
                )
                with open(mask_file, "wb") as f:
                    np.save(f, mask_to_save)

            end_time = time.time()
            total_time = end_time - start_time
            if rank == 0:
                log.info(
                    "Rank 0 generated %s volumes in %.2f seconds | %.2f volumes per second",
                    len(volumes_contents_subset),
                    total_time,
                    len(volumes_contents_subset) / total_time,
                )
    except BaseException as e:
        # Capture the failure locally instead of letting it unwind past the
        # collective below, which would desynchronize the ranks. BaseException,
        # not (Exception, SystemExit): a KeyboardInterrupt delivered to one rank
        # would otherwise skip the consensus and hang the others.
        ok = False
        err = f"rank {rank}: {type(e).__name__}: {e}"
        interrupt = e if isinstance(e, KeyboardInterrupt) else None

    # Consensus on the generation status. This replaces a bare Barrier: every
    # rank always executes exactly this collective (regardless of success or
    # failure), so a failing rank can never leave a peer hung in a mismatched
    # collective. If any rank failed, all ranks raise here.
    all_ok = comm.allreduce(1 if ok else 0, op=MPI.MIN) == 1
    errs = comm.allgather(err)
    if not all_ok:
        # The interrupted rank re-raises the operator's abort verbatim; the
        # others report the gathered failure.
        if interrupt is not None:
            raise interrupt
        msgs = "; ".join(e for e in errs if e)
        raise RuntimeError(f"volume generation failed: {msgs or 'unknown error'}")

    if rank == 0:
        log.info("All ranks done. Proceeding to split.")

    # Do the train/val split and generate lists of unique train/val masks
    if rank == 0:
        log.info("Volume generation complete. Generating unique mask lists.")

        # Directories are already created at start of script

        # Reconstruct lists for unique mask value scanning
        val_files = sorted(list(val_indices))
        train_files = sorted(list(set(range(num_volumes)) - val_indices))

        log.info(
            "len(val_files)=%s, len(train_files)=%s",
            len(val_files),
            len(train_files),
        )

        # Save lists of unique train and val mask values
        log.info(
            "Calculating unique mask values from configuration without reading mask files."
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
