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

from math import prod
from typing import Iterable, Tuple


def normalize_sharding(num_shards: Iterable[int], shard_dims: Iterable[int]):
    """Validate and normalize spatial sharding config."""

    num_shards = tuple(int(x) for x in num_shards)
    shard_dims = tuple(int(x) for x in shard_dims)

    if len(num_shards) != len(shard_dims):
        raise ValueError(
            f"num_shards {num_shards} must have same length as shard_dims {shard_dims}"
        )
    if len(set(shard_dims)) != len(shard_dims):
        raise ValueError(f"Shard dimensions must be unique: {shard_dims}")

    for num_shards_i, shard_dim_i in zip(num_shards, shard_dims):
        if num_shards_i < 1:
            raise ValueError(f"Invalid num_shards value {num_shards_i}")
        if shard_dim_i not in (2, 3, 4):
            raise ValueError(
                f"Invalid shard_dim {shard_dim_i}: only 3D spatial dimensions 2, 3, and 4 are supported"
            )

    return num_shards, shard_dims


def total_shards(num_shards: Iterable[int]) -> int:
    return prod(tuple(int(x) for x in num_shards))


def shard_id_to_indices(shard_id: int, num_shards: Iterable[int]) -> Tuple[int, ...]:
    """Convert row-major linear shard id to multi-dimensional shard indices."""

    num_shards = tuple(int(x) for x in num_shards)
    total = total_shards(num_shards)
    if shard_id < 0 or shard_id >= total:
        raise ValueError(f"shard_id {shard_id} out of range for num_shards={num_shards}")

    indices = []
    linear_idx = int(shard_id)
    stride = total
    for num_shards_i in num_shards:
        stride //= num_shards_i
        indices.append(linear_idx // stride)
        linear_idx %= stride
    return tuple(indices)


def shard_indices_to_id(
    shard_indices: Iterable[int], num_shards: Iterable[int]
) -> int:
    """Convert multi-dimensional shard indices to row-major linear shard id."""

    shard_indices = tuple(int(x) for x in shard_indices)
    num_shards = tuple(int(x) for x in num_shards)
    if len(shard_indices) != len(num_shards):
        raise ValueError(
            f"shard_indices {shard_indices} must match num_shards {num_shards}"
        )

    shard_id = 0
    stride = 1
    for shard_index_i, num_shards_i in zip(reversed(shard_indices), reversed(num_shards)):
        if shard_index_i < 0 or shard_index_i >= num_shards_i:
            raise ValueError(
                f"Invalid shard index {shard_index_i} for num_shards={num_shards}"
            )
        shard_id += shard_index_i * stride
        stride *= num_shards_i
    return shard_id


def chunk_slice(size: int, num_shards: int, shard_index: int) -> slice:
    """Match torch.chunk-style uneven shard boundaries."""

    chunk_size = (size + num_shards - 1) // num_shards
    start = shard_index * chunk_size
    if start >= size:
        raise ValueError(
            f"Empty local shard: dim size {size}, num_shards {num_shards}, shard_index {shard_index}"
        )
    stop = min(size, start + chunk_size)
    return slice(start, stop)


def spatial_slices(
    spatial_shape: Iterable[int],
    shard_dims: Iterable[int],
    num_shards: Iterable[int],
    shard_indices: Iterable[int],
) -> Tuple[slice, slice, slice]:
    """Return local D/H/W slices for DistConv spatial dims 2/3/4."""

    spatial_shape = tuple(int(x) for x in spatial_shape)
    if len(spatial_shape) != 3:
        raise ValueError(f"Expected 3D spatial shape, got {spatial_shape}")

    slices = [slice(0, size) for size in spatial_shape]
    for shard_dim, num_shards_i, shard_index_i in zip(
        shard_dims, num_shards, shard_indices
    ):
        spatial_axis = int(shard_dim) - 2
        slices[spatial_axis] = chunk_slice(
            spatial_shape[spatial_axis], int(num_shards_i), int(shard_index_i)
        )

    return tuple(slices)


def shard_file_suffix(shard_id: int) -> str:
    return f"_shard{int(shard_id):06d}"
