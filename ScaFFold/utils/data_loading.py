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

import hashlib
import pickle
from dataclasses import dataclass
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
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

        # os.listdir order is filesystem/client dependent and explicitly
        # arbitrary. Sorting makes the index -> file mapping deterministic and
        # byte-identical across processes, which spatial sharding relies on:
        # ranks that share a data-replica index must resolve the same dataset
        # index to the same volume, or shard assembly stitches together halves
        # of different samples with no error.
        image_files = [
            file
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]
        self.ids = sorted(splitext(file)[0] for file in image_files)
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )

        # Resolve every id to its full image/mask path once, up front. Sample
        # fetches then index these maps in O(1) instead of scanning (and
        # fnmatching) the whole directory on every call, which on a shared
        # filesystem turns each fetch into a burst of metadata traffic.
        self._image_paths = self._index_paths_by_stem(
            self.images_dir, image_files, "image"
        )
        mask_files = [
            entry.name
            for entry in self.mask_dir.iterdir()
            if entry.is_file() and not entry.name.startswith(".")
        ]
        self._mask_paths = self._index_paths_by_stem(self.mask_dir, mask_files, "mask")

        # Belt-and-braces: when a process group is live, verify every rank built
        # the identical id list. Any residual divergence (e.g. inconsistent
        # readdir views across parallel-filesystem clients) becomes a hard error
        # here instead of silently corrupted training samples.
        self._verify_ids_consistent_across_ranks(images_dir)

        customlog(
            f"Creating dataset with {len(self.ids)} examples. Loading from {data_dir}"
        )
        self.mask_values = self._load_mask_values(data_dir)
        customlog(f"Unique mask values: {self.mask_values}")
        customlog(f"Dataset format version: {self.dataset_format_version}")

        # Masks are handed off in a signed 16-bit carrier (widened to long on
        # the compute device), so every class id must fit that range. Legacy
        # masks are remapped to 0..len(mask_values)-1; optimized masks store
        # dense ids that stay within the same bound.
        max_class_id = len(self.mask_values) - 1
        if max_class_id > np.iinfo(np.int16).max:
            raise ValueError(
                f"{len(self.mask_values)} classes exceed the int16 mask carrier "
                f"limit ({np.iinfo(np.int16).max})"
            )

    def _load_mask_values(self, data_dir):
        """Return the label-remap table for this split.

        v2 datasets store dense class ids and never remap, so the per-split
        pickle is loaded verbatim for bookkeeping. Legacy (v1) datasets remap
        raw voxel values by their position in this list; a per-split table would
        assign the same raw value different class ids across splits whenever a
        category is missing from one split. To keep train and validation labels
        consistent, v1 uses one global table: the sorted union of every
        ``*unique_mask_vals`` pickle found in the dataset root.
        """
        with open(data_dir, "rb") as data_file:
            mask_values = pickle.load(data_file)["mask_values"]

        if self.dataset_format_version >= DATASET_FORMAT_VERSION:
            return mask_values

        union = set()
        for pickle_path in sorted(self.dataset_root.glob("*unique_mask_vals")):
            try:
                with open(pickle_path, "rb") as handle:
                    values = pickle.load(handle)["mask_values"]
            except (OSError, KeyError, pickle.UnpicklingError) as exc:
                customlog(f"Ignoring unreadable mask-values file {pickle_path}: {exc}")
                continue
            values_arr = np.asarray(values)
            if values_arr.ndim <= 1:
                union.update(values_arr.reshape(-1).tolist())
            else:
                # Composite labels (e.g. RGB rows) must stay whole: the legacy
                # remap compares each entry against the mask's channel axis, so
                # flattening rows into scalars would corrupt the table.
                union.update(
                    tuple(row)
                    for row in values_arr.reshape(values_arr.shape[0], -1).tolist()
                )

        if not union:
            # No sibling pickles discovered; fall back to this split's own list.
            return mask_values

        return sorted(union)

    def _verify_ids_consistent_across_ranks(self, images_dir):
        """Raise if ranks disagree on ``self.ids`` while a process group is up.

        A no-op when torch.distributed is unavailable or uninitialized (CPU unit
        tests), so it costs exactly one small collective only in genuinely
        distributed runs.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return

        digest = hashlib.sha256("\n".join(self.ids).encode("utf-8")).digest()
        local = torch.frombuffer(bytearray(digest), dtype=torch.uint8)
        # The digest must live on a device the process group can move: NCCL
        # only handles GPU tensors, gloo's all_gather only CPU ones. Key off
        # the backend rather than CUDA availability so a gloo group on a
        # GPU-equipped node still takes the CPU path.
        if dist.get_backend() == dist.Backend.NCCL:
            local = local.cuda()
        gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)

        for rank, other in enumerate(gathered):
            if not torch.equal(other, local):
                raise RuntimeError(
                    "Dataset id lists diverge across ranks for "
                    f"{images_dir}: rank {dist.get_rank()} disagrees with rank "
                    f"{rank}. Every rank must observe the same files (check for "
                    "inconsistent filesystem views across nodes)."
                )

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _index_paths_by_stem(directory, filenames, label):
        """Map each file's extension-less name to its full path.

        Raises if two files in ``directory`` share a stem, which would make the
        stem an ambiguous key and silently pick one of them at fetch time.
        """
        paths = {}
        for filename in filenames:
            stem = splitext(filename)[0]
            full_path = directory / filename
            existing = paths.get(stem)
            if existing is not None:
                raise RuntimeError(
                    f"Ambiguous {label} id '{stem}' in {directory}: matches both "
                    f"{existing.name} and {filename}. Every id must map to exactly "
                    "one file."
                )
            paths[stem] = full_path
        return paths

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
    def _prepare_optimized_image(img, materialize):
        # ``materialize`` is set when the array is a memory-mapped slice that
        # must be copied into RAM to detach it from the backing file. When the
        # array is already a fresh, correctly-typed, contiguous buffer (the
        # non-mmap load), asarray is a no-op and avoids duplicating the whole
        # volume a second time in the DataLoader worker.
        if materialize:
            return np.array(img, dtype=VOLUME_DTYPE, copy=True, order="C")
        return np.asarray(img, dtype=VOLUME_DTYPE, order="C")

    @staticmethod
    def _prepare_optimized_mask(mask, materialize):
        if materialize:
            return np.array(mask, dtype=MASK_DTYPE, copy=True, order="C")
        return np.asarray(mask, dtype=MASK_DTYPE, order="C")

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

    def _image_path(self, name):
        try:
            return self._image_paths[name]
        except KeyError:
            raise KeyError(
                f"No image file found for the ID {name} in {self.images_dir}"
            )

    def _mask_path(self, name):
        key = name + self.mask_suffix
        try:
            return self._mask_paths[key]
        except KeyError:
            raise KeyError(f"No mask file found for the ID {key} in {self.mask_dir}")

    def _load_prepared_mask(self, name):
        """Load, shard-slice, and label-prepare one mask as ``MASK_DTYPE``."""
        materialize = self.spatial_shard_spec is not None
        # Memmap lets each rank slice out just its local shard without eagerly
        # reading the full sample into process memory first; the prepare step
        # then copies that slice out of the backing file.
        mmap_mode = "r" if materialize else None
        mask = self._load_numpy_array(self._mask_path(name), mmap_mode=mmap_mode)
        mask = self._slice_mask_array(mask)
        if self.dataset_format_version >= DATASET_FORMAT_VERSION:
            return self._prepare_optimized_mask(mask, materialize)
        return self._prepare_legacy_mask(self.mask_values, mask)

    @staticmethod
    def _to_mask_carrier(mask):
        # Ship the mask in a narrow signed 16-bit carrier rather than widening
        # to int64 here: the consumer re-casts to long on the compute device,
        # so a wider dtype only inflates pinned-host memory and the host-to-
        # device copy. (Signed, because unsigned 16-bit has no CPU bincount.)
        return torch.from_numpy(mask.astype(np.int16, copy=False)).contiguous()

    def load_mask_only(self, idx):
        """Return only the mask for ``idx`` as the narrow int16 carrier tensor.

        Identical to the ``"mask"`` entry of :meth:`__getitem__` but without
        touching the image volume, so callers that only need mask statistics
        (e.g. class-frequency counts via ``torch.bincount(mask.reshape(-1)
        .long(), ...)``) skip reading and preparing the discarded image.
        """
        return self._to_mask_carrier(self._load_prepared_mask(self.ids[idx]))

    def __getitem__(self, idx):
        name = self.ids[idx]
        materialize = self.spatial_shard_spec is not None
        mmap_mode = "r" if materialize else None
        img = self._load_numpy_array(self._image_path(name), mmap_mode=mmap_mode)
        img = self._slice_image_array(img)

        if self.dataset_format_version >= DATASET_FORMAT_VERSION:
            img = self._prepare_optimized_image(img, materialize)
        else:
            img = self._prepare_legacy_image(img)

        mask = self._load_prepared_mask(name)

        return {
            "image": torch.from_numpy(img).contiguous().float(),
            "mask": self._to_mask_carrier(mask),
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
