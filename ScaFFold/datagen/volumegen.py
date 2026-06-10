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
from typing import Callable, Dict

import numpy as np
from mpi4py import MPI

from ScaFFold.datagen import layout
from ScaFFold.utils.config_utils import Config
from ScaFFold.utils.data_types import MASK_DTYPE, VOLUME_DTYPE
from ScaFFold.utils.spatial_sharding import (
    normalize_sharding,
    shard_file_suffix,
    shard_id_to_indices,
    spatial_slices,
    total_shards,
)
from ScaFFold.utils.utils import setup_mpi_logger

DEFAULT_DC_NUM_SHARDS = (1, 1, 1)
DEFAULT_DC_SHARD_DIMS = (2, 3, 4)

# Liveness marker for the directory being generated into. Volume writing is the
# long phase of a generation and it happens deep inside the tree
# (``volumes/<split>/N.npy``), so the top of the staging directory can look
# untouched for many hours while the job is perfectly healthy. Every writing
# rank therefore refreshes this file periodically, and
# ``get_dataset._staging_dir_is_live`` reads it instead of trying to infer
# liveness from mtimes it cannot cheaply see. The name is owned here, next to
# the writer; ``get_dataset`` (which already imports this module) reads it from
# here so the two sides cannot drift.
STAGING_HEARTBEAT_NAME = ".heartbeat"
# Refresh interval. Small enough to be negligible against the staleness
# threshold (a day), large enough that it is one utime per rank per few minutes
# no matter how fast volumes are written.
STAGING_HEARTBEAT_INTERVAL_SECONDS = 5 * 60


class StagingHeartbeat:
    """Periodically touch a staging directory's heartbeat file.

    ``beat()`` is called from the volume loop and is a no-op until the interval
    has elapsed, so it costs one comparison per volume. Every rank writing into
    the directory beats the same file: the marker means "somebody is still
    working here", and a last-writer-wins utime is exactly the semantics wanted.
    Failures are swallowed -- a heartbeat that cannot be written must never take
    down a generation that is otherwise fine (the cleanup's second signal, the
    bounded mtime probe, still applies).
    """

    def __init__(
        self, staging_dir, interval: float = STAGING_HEARTBEAT_INTERVAL_SECONDS
    ) -> None:
        self.path = os.path.join(str(staging_dir), STAGING_HEARTBEAT_NAME)
        self.interval = interval
        self._last_beat = float("-inf")

    def beat(self, now: float | None = None) -> bool:
        """Touch the marker if the interval has elapsed; return whether it did."""
        now = time.time() if now is None else now
        if now - self._last_beat < self.interval:
            return False
        self._last_beat = now
        try:
            with open(self.path, "a"):
                pass
            os.utime(self.path, (now, now))
        except OSError:
            return False
        return True


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


def _sharding_values(config):
    num_shards = getattr(config, "dc_num_shards", DEFAULT_DC_NUM_SHARDS)
    shard_dims = getattr(config, "dc_shard_dims", DEFAULT_DC_SHARD_DIMS)
    return num_shards, shard_dims


def _physical_sharding(config):
    """Return normalized physical sharding from the generation config."""

    num_shards, shard_dims = _sharding_values(config)
    return normalize_sharding(num_shards, shard_dims)


def _validate_generation_config(config):
    """Validate sharded generation settings and return normalized layout data."""

    num_shards, shard_dims = _physical_sharding(config)
    n_total_shards = total_shards(num_shards)
    grid_size = resolve_grid_size(config)

    return num_shards, shard_dims, n_total_shards, grid_size


def _voxelized_fractals_for_volume(
    config,
    curr_vol: np.ndarray,
    fractal_colors: np.ndarray,
    instances_dir: str,
    grid_size: int,
    point_cloud_loader: Callable[[str], np.ndarray] = load_np_ptcloud,
):
    """Load and voxelize all fractals needed for one logical volume."""

    n_fracts_per_vol = config.n_fracts_per_vol
    voxelized_fractals = []

    for curr_fract in range(n_fracts_per_vol):
        curr_category = int(curr_vol[1 + 2 * curr_fract])
        curr_instance = int(curr_vol[1 + 2 * curr_fract + 1])
        fractal_color = fractal_colors[curr_category]

        point_cloud_path = os.path.join(
            instances_dir,
            f"{curr_category:06d}",
            f"{curr_category:06d}_{curr_instance:04d}.npy",
        )
        if point_cloud_loader is load_np_ptcloud and not os.path.exists(
            point_cloud_path
        ):
            raise FileNotFoundError(
                f"File {point_cloud_path} does not exist. "
                "Ensure you have run 'scaffold generate_fractals ...'"
            )

        points = point_cloud_loader(point_cloud_path)
        idx = points_to_voxel_indices(points, grid_size)
        voxelized_fractals.append((curr_category, fractal_color, idx))

    return voxelized_fractals


def _render_volume_shard(config, voxelized_fractals, shard_id: int):
    """Render one physical shard from precomputed global voxel indices."""
    num_shards, shard_dims = _physical_sharding(config)
    shard_indices = shard_id_to_indices(shard_id, num_shards)
    slices = spatial_slices(
        (config.vol_size, config.vol_size, config.vol_size),
        shard_dims,
        num_shards,
        shard_indices,
    )
    local_shape = tuple(s.stop - s.start for s in slices)

    volume = np.full((3, *local_shape), 0, dtype=VOLUME_DTYPE)
    mask = np.full(local_shape, 0, dtype=MASK_DTYPE)

    for curr_category, fractal_color, idx in voxelized_fractals:
        keep = np.ones(idx.shape[0], dtype=bool)
        for axis, axis_slice in enumerate(slices):
            keep &= idx[:, axis] >= axis_slice.start
            keep &= idx[:, axis] < axis_slice.stop

        if not np.any(keep):
            continue

        local_idx = idx[keep]
        local_idx[:, 0] -= slices[0].start
        local_idx[:, 1] -= slices[1].start
        local_idx[:, 2] -= slices[2].start
        d = local_idx[:, 0]
        h = local_idx[:, 1]
        w = local_idx[:, 2]

        volume[:, d, h, w] = fractal_color[:, None]
        mask[d, h, w] = curr_category + 1

    return volume, mask


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
    _, _, n_total_shards, grid_size = _validate_generation_config(config)

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

    # Each writer defensively creates the output dirs after root's broadcast.
    for subdir in ["training", "validation"]:
        os.makedirs(os.path.join(vol_path, subdir), exist_ok=True)
        os.makedirs(os.path.join(mask_path, subdir), exist_ok=True)

    # Rank 0 creates shared metadata above; wait before local writer setup.
    comm.Barrier()

    for subdir in ["training", "validation"]:
        os.makedirs(os.path.join(vol_path, subdir), exist_ok=True)
        os.makedirs(os.path.join(mask_path, subdir), exist_ok=True)

    # Wait until every rank has ensured the writer directories exist.
    comm.Barrier()

    # Determine train/val split globally so all ranks know where to save
    num_volumes = len(volumes_contents)
    random.seed(config.seed)  # Reset seed to ensure all ranks get same split
    val_indices = set(
        random.sample(range(num_volumes), int(num_volumes * config.val_split / 100))
    )

    # Work distribution: each task renders one physical shard of one logical volume.
    total_tasks = num_volumes * n_total_shards
    stride = ceil(total_tasks / size)
    start_idx = rank * stride
    end_idx = min(((rank + 1) * stride), total_tasks)

    # Per-rank generation status. A rank that fails records the message here and
    # keeps executing the collective sequence below so every rank stays in step;
    # the failure is then propagated to all ranks via an allreduce.
    ok = True
    err = ""
    interrupt = None

    try:
        if start_idx >= end_idx:
            log.debug("Rank %s given no physical shard tasks to generate", rank)

        else:
            log.debug(
                "Rank %s responsible for physical shard tasks %s through %s",
                rank,
                start_idx,
                end_idx - 1,
            )

            np.random.seed(config.seed)
            fractal_colors = np.random.rand(config.n_categories, 3)

            # The instance library is keyed by seed (see ScaFFold.datagen
            # .layout), so a volume can only ever be built from point clouds
            # this run's seed produced. Resolved once, outside the loop.
            instances_dir = layout.instance_dir(config)

            # Generation loop. Every rank reports that this staging directory
            # is still being written to, so a concurrent job's orphan cleanup
            # can tell a live multi-hour generation from one killed a day ago
            # (the volumes themselves land two levels down, where a cheap
            # top-level mtime probe cannot see them).
            heartbeat = StagingHeartbeat(dataset_dir)
            heartbeat.beat()
            start_time = time.time()
            n_generated_shards = 0
            cached_volume_idx = None
            cached_global_vol_idx = None
            cached_voxelized_fractals = None

            for i, task_idx in enumerate(range(start_idx, end_idx)):
                heartbeat.beat()
                if i % 10 == 0:
                    log.debug(
                        "Rank %s processing local physical shard task %s", rank, i
                    )

                volume_idx = task_idx // n_total_shards
                shard_id = task_idx % n_total_shards

                if cached_volume_idx != volume_idx:
                    curr_vol = volumes_contents[volume_idx]
                    global_vol_idx = int(curr_vol[0])
                    vol_seed = config.seed + global_vol_idx
                    random.seed(vol_seed)
                    np.random.seed(vol_seed)

                    cached_voxelized_fractals = _voxelized_fractals_for_volume(
                        config,
                        curr_vol,
                        fractal_colors,
                        instances_dir,
                        grid_size,
                    )
                    cached_volume_idx = volume_idx
                    cached_global_vol_idx = global_vol_idx

                volume_to_save, mask_to_save = _render_volume_shard(
                    config,
                    cached_voxelized_fractals,
                    shard_id,
                )

                # Determine destination folder
                subdir = (
                    "validation" if cached_global_vol_idx in val_indices else "training"
                )
                shard_suffix = shard_file_suffix(shard_id)

                vol_file = os.path.join(
                    vol_path, subdir, f"{cached_global_vol_idx}{shard_suffix}.npy"
                )
                with open(vol_file, "wb") as f:
                    np.save(f, volume_to_save)

                mask_file = os.path.join(
                    mask_path,
                    subdir,
                    f"{cached_global_vol_idx}{shard_suffix}_mask.npy",
                )
                with open(mask_file, "wb") as f:
                    np.save(f, mask_to_save)
                n_generated_shards += 1

            end_time = time.time()
            total_time = end_time - start_time
            if rank == 0:
                shard_rate = n_generated_shards / total_time
                log.info(
                    "Rank 0 generated %s volume shards from %s physical shard "
                    "tasks in %.2f seconds | %.2f shards per second",
                    n_generated_shards,
                    end_idx - start_idx,
                    total_time,
                    shard_rate,
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
