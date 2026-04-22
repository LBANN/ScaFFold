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

import pickle
from dataclasses import dataclass
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from ScaFFold.utils.data_types import MASK_DTYPE, VOLUME_DTYPE
from ScaFFold.utils.utils import customlog

DATASET_FORMAT_VERSION = 2
LEGACY_DATASET_FORMAT_VERSION = 1
META_FILENAME = "meta.yaml"


@dataclass(frozen=True)
class SpatialShardSpec:
    """Describe the local spatial shard owned by the current rank."""

    shard_dims: Tuple[int, ...]
    num_shards: Tuple[int, ...]
    shard_indices: Tuple[int, ...]

    def __post_init__(self):
        if not (
            len(self.shard_dims) == len(self.num_shards) == len(self.shard_indices)
        ):
            raise ValueError(
                "shard_dims, num_shards, and shard_indices must have matching lengths"
            )
        if len(set(self.shard_dims)) != len(self.shard_dims):
            raise ValueError(f"Shard dimensions must be unique: {self.shard_dims}")
        for shard_dim, num_shards, shard_index in zip(
            self.shard_dims, self.num_shards, self.shard_indices
        ):
            if shard_dim < 2:
                raise ValueError(
                    f"Invalid shard_dim {shard_dim}: only spatial dimensions are supported"
                )
            if num_shards < 1:
                raise ValueError(
                    f"Invalid num_shards {num_shards} for shard_dim {shard_dim}"
                )
            if shard_index < 0 or shard_index >= num_shards:
                raise ValueError(
                    f"Invalid shard_index {shard_index} for shard_dim {shard_dim} with {num_shards} shards"
                )

    @staticmethod
    def _chunk_slice(size: int, num_shards: int, shard_index: int) -> slice:
        """Match torch.chunk-style uneven shard boundaries."""

        chunk_size = (size + num_shards - 1) // num_shards
        start = shard_index * chunk_size
        if start >= size:
            raise ValueError(
                f"Empty local shard: dim size {size}, num_shards {num_shards}, shard_index {shard_index}"
            )
        stop = min(size, start + chunk_size)
        return slice(start, stop)

    def slice_array(
        self, array: np.ndarray, axis_map: Dict[int, int], array_label: str
    ) -> np.ndarray:
        if not self.shard_dims:
            return array

        slices = [slice(None)] * array.ndim
        for shard_dim, num_shards, shard_index in zip(
            self.shard_dims, self.num_shards, self.shard_indices
        ):
            if shard_dim not in axis_map:
                raise ValueError(
                    f"No axis mapping defined for {array_label} shard_dim {shard_dim}"
                )
            axis = axis_map[shard_dim]
            if axis >= array.ndim:
                raise ValueError(
                    f"Axis {axis} out of range for {array_label} with shape {array.shape}"
                )
            slices[axis] = self._chunk_slice(array.shape[axis], num_shards, shard_index)

        return array[tuple(slices)]


class BasicDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        mask_dir: str,
        mask_suffix: str = "",
        data_dir: str = "",
        spatial_shard_spec: Optional[SpatialShardSpec] = None,
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix
        self.spatial_shard_spec = spatial_shard_spec
        self.dataset_root = self.images_dir.parents[1]
        self.dataset_format_version = self._load_dataset_format_version()

        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )

        customlog(
            f"Creating dataset with {len(self.ids)} examples. Loading from {data_dir}"
        )
        with open(data_dir, "rb") as data_file:
            data = pickle.load(data_file)
        self.mask_values = data["mask_values"]
        customlog(f"Unique mask values: {self.mask_values}")
        customlog(f"Dataset format version: {self.dataset_format_version}")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _load_numpy_array(path, mmap_mode=None):
        return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)

    def _load_dataset_format_version(self):
        meta_path = self.dataset_root / META_FILENAME
        if not meta_path.exists():
            return LEGACY_DATASET_FORMAT_VERSION

        try:
            with open(meta_path, "r") as meta_file:
                meta = yaml.safe_load(meta_file) or {}
        except Exception as exc:
            customlog(
                f"Failed to read dataset metadata from {meta_path}: {exc}. Falling back to legacy loader."
            )
            return LEGACY_DATASET_FORMAT_VERSION

        return int(meta.get("dataset_format_version", LEGACY_DATASET_FORMAT_VERSION))

    @staticmethod
    def _prepare_legacy_image(img):
        return np.ascontiguousarray(img.transpose((3, 0, 1, 2)), dtype=VOLUME_DTYPE)

    @staticmethod
    def _prepare_legacy_mask(mask_values, mask):
        remapped = np.zeros(
            (mask.shape[0], mask.shape[1], mask.shape[2]), dtype=MASK_DTYPE
        )
        for i, value in enumerate(mask_values):
            if mask.ndim == 3:
                remapped[mask == value] = i
            else:
                remapped[(mask == value).all(-1)] = i

        return remapped

    @staticmethod
    def _prepare_optimized_image(img):
        return np.array(img, dtype=VOLUME_DTYPE, copy=True, order="C")

    @staticmethod
    def _prepare_optimized_mask(mask):
        return np.array(mask, dtype=MASK_DTYPE, copy=True, order="C")

    def _slice_image_array(self, img):
        if self.spatial_shard_spec is None:
            return img

        if self.dataset_format_version >= DATASET_FORMAT_VERSION:
            axis_map = {2: 1, 3: 2, 4: 3}
        else:
            axis_map = {2: 0, 3: 1, 4: 2}
        return self.spatial_shard_spec.slice_array(img, axis_map, "image")

    def _slice_mask_array(self, mask):
        if self.spatial_shard_spec is None:
            return mask

        axis_map = {2: 0, 3: 1, 4: 2}
        return self.spatial_shard_spec.slice_array(mask, axis_map, "mask")

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        assert len(img_file) == 1, (
            f"Either no image or multiple images found for the ID {name}: {img_file}"
        )
        assert len(mask_file) == 1, (
            f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        )
        mmap_mode = "r" if self.spatial_shard_spec is not None else None
        # Memmap lets each rank slice out just its local shard without eagerly
        # reading the full sample into process memory first.
        mask = self._load_numpy_array(mask_file[0], mmap_mode=mmap_mode)
        img = self._load_numpy_array(img_file[0], mmap_mode=mmap_mode)
        mask = self._slice_mask_array(mask)
        img = self._slice_image_array(img)

        if self.dataset_format_version >= DATASET_FORMAT_VERSION:
            img = self._prepare_optimized_image(img)
            mask = self._prepare_optimized_mask(mask)
        else:
            img = self._prepare_legacy_image(img)
            mask = self._prepare_legacy_mask(self.mask_values, mask)

        return {
            "image": torch.from_numpy(img).contiguous().float(),
            "mask": torch.from_numpy(mask).contiguous().long(),
        }


class FractalDataset(BasicDataset):
    def __init__(
        self,
        images_dir,
        mask_dir,
        data_dir,
        spatial_shard_spec: Optional[SpatialShardSpec] = None,
    ):
        super().__init__(
            images_dir,
            mask_dir,
            mask_suffix="_mask",
            data_dir=data_dir,
            spatial_shard_spec=spatial_shard_spec,
        )
