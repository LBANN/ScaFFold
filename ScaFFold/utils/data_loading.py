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
import re
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
from ScaFFold.utils.spatial_sharding import (
    chunk_slice,
    normalize_sharding,
    shard_file_suffix,
    shard_indices_to_id,
    total_shards,
)
from ScaFFold.utils.utils import customlog

DATASET_FORMAT_VERSION = 2
PHYSICAL_SHARDED_DATASET_FORMAT_VERSION = 5
MAX_SUPPORTED_DATASET_FORMAT_VERSION = PHYSICAL_SHARDED_DATASET_FORMAT_VERSION
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
            slices[axis] = chunk_slice(array.shape[axis], num_shards, shard_index)

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
        self.dataset_meta = self._load_dataset_metadata()
        self.dataset_format_version = int(
            self.dataset_meta.get(
                "dataset_format_version", LEGACY_DATASET_FORMAT_VERSION
            )
        )
        if self.dataset_format_version > MAX_SUPPORTED_DATASET_FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported dataset format version {self.dataset_format_version}; "
                f"expected <= {MAX_SUPPORTED_DATASET_FORMAT_VERSION}"
            )
        self.physical_shards = (
            self.dataset_format_version >= PHYSICAL_SHARDED_DATASET_FORMAT_VERSION
        )
        self.physical_num_shards, self.physical_shard_dims = (
            self._load_physical_sharding()
        )
        self.physical_total_shards = (
            total_shards(self.physical_num_shards) if self.physical_shards else 1
        )
        self.shard_id = self._select_physical_shard_id()
        self.shard_suffix = (
            shard_file_suffix(self.shard_id) if self.physical_shards else ""
        )

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
        self.ids, self._image_paths = self._index_paths_by_id(
            self.images_dir, image_files, "image"
        )
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )

        # Resolve every id to its full image/mask path once, up front. Sample
        # fetches then index these maps in O(1) instead of scanning (and
        # fnmatching) the whole directory on every call, which on a shared
        # filesystem turns each fetch into a burst of metadata traffic.
        mask_files = [
            entry.name
            for entry in self.mask_dir.iterdir()
            if entry.is_file() and not entry.name.startswith(".")
        ]
        _, self._mask_paths = self._index_paths_by_id(self.mask_dir, mask_files, "mask")

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
        if self.physical_shards:
            customlog(
                f"Loading physical shard files with suffix {self.shard_suffix}; "
                f"dc_num_shards={self.physical_num_shards}, "
                f"dc_shard_dims={self.physical_shard_dims}"
            )

        # Masks are handed off in a signed 16-bit carrier (widened to long on
        # the compute device), so the largest class id the carrier will hold
        # must fit that range.
        max_class_id = self._max_class_id()
        if max_class_id > np.iinfo(np.int16).max:
            raise ValueError(
                f"Mask class id {max_class_id} (from {len(self.mask_values)} "
                f"classes) exceeds the int16 mask carrier limit "
                f"({np.iinfo(np.int16).max}); it would wrap negative"
            )

    def _max_class_id(self):
        """Return the largest class id ``_to_mask_carrier`` will have to carry.

        The bound differs by format, and using the wrong one is unsafe in one
        direction and needlessly strict in the other. Modern masks ship *raw*
        ``category + 1`` ids, and the per-split table lists only the categories
        present in that split -- so a sparse split can declare two classes while
        holding an id in the tens of thousands, which the class *count* check
        happily waved through. Legacy masks, by contrast, are remapped to
        ``0..len(mask_values)-1``, so the count is exactly right there and their
        (arbitrarily large) raw values are irrelevant.
        """
        if self.dataset_format_version < DATASET_FORMAT_VERSION:
            return len(self.mask_values) - 1

        ids = np.asarray(self.mask_values)
        if ids.size == 0:
            return 0
        return int(ids.max())

    def _load_mask_values(self, data_dir):
        """Return the label-remap table for this split.

        Modern datasets store dense class ids and never remap, so the per-split
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

    def _id_from_filename(self, filename, label):
        if not self.physical_shards:
            stem = splitext(filename)[0]
            if label == "mask":
                if not stem.endswith(self.mask_suffix):
                    return None
                return stem
            return stem

        suffix = re.escape(self.shard_suffix)
        if label == "mask":
            pattern = re.compile(
                rf"^(?P<id>.+){suffix}{re.escape(self.mask_suffix)}\.npy$"
            )
        else:
            pattern = re.compile(rf"^(?P<id>.+){suffix}\.npy$")
        match = pattern.match(filename)
        if match is None:
            return None
        return match.group("id")

    def _index_paths_by_id(self, directory, filenames, label):
        """Map each logical sample id to its full path.

        Raises if two files in ``directory`` share an id, which would make the
        id an ambiguous key and silently pick one of them at fetch time.
        """
        paths = {}
        for filename in filenames:
            sample_id = self._id_from_filename(filename, label)
            if sample_id is None:
                continue
            full_path = directory / filename
            existing = paths.get(sample_id)
            if existing is not None:
                raise RuntimeError(
                    f"Ambiguous {label} id '{sample_id}' in {directory}: matches "
                    f"both {existing.name} and {filename}. Every id must map to "
                    "exactly one file."
                )
            paths[sample_id] = full_path
        return sorted(paths), paths

    @staticmethod
    def _load_numpy_array(path, mmap_mode=None):
        return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)

    def _load_dataset_metadata(self):
        """Determine which on-disk layout this dataset uses.

        Only a *missing* ``meta.yaml`` means legacy v1: those datasets predate
        the metadata file. A metadata file that exists but cannot be read or
        does not carry a usable version is a damaged modern dataset, and
        falling back to the legacy loader there silently transposes
        channels-first volumes and remaps already-dense labels -- corrupt
        training data with no error. Such a dataset is rejected instead, with a
        message naming the file so it can be repaired or regenerated.
        """
        meta_path = self.dataset_root / META_FILENAME
        if not meta_path.exists():
            return {"dataset_format_version": LEGACY_DATASET_FORMAT_VERSION}

        try:
            with open(meta_path, "r") as meta_file:
                meta = yaml.safe_load(meta_file)
        except Exception as exc:
            raise ValueError(
                f"Dataset metadata {meta_path} exists but could not be read "
                f"({type(exc).__name__}: {exc}). A dataset carrying a "
                f"{META_FILENAME} is not a legacy dataset; refusing to guess its "
                "layout. Repair the file or regenerate the dataset."
            ) from exc

        version = meta.get("dataset_format_version") if isinstance(meta, dict) else None
        try:
            int(version)
        except (TypeError, ValueError):
            raise ValueError(
                f"Dataset metadata {meta_path} is missing a usable "
                f"'dataset_format_version' (got {version!r}). A dataset carrying "
                f"a {META_FILENAME} is not a legacy dataset; refusing to guess "
                "its layout. Repair the file or regenerate the dataset."
            ) from None

        return meta

    def _load_physical_sharding(self):
        """Load and normalize the physical shard layout from metadata."""

        if not self.physical_shards:
            return (), ()

        config_subset = self.dataset_meta.get("config_subset") or {}
        num_shards = config_subset.get("dc_num_shards")
        shard_dims = config_subset.get("dc_shard_dims")
        if num_shards is None or shard_dims is None:
            raise RuntimeError(
                "Physical dataset is missing shard metadata. Expected "
                "config_subset.dc_num_shards/config_subset.dc_shard_dims in meta.yaml."
            )

        return normalize_sharding(num_shards, shard_dims)

    @staticmethod
    def _layout_by_dim(num_shards, shard_dims):
        """Map each sharded dimension to its shard count."""

        return {int(dim): int(num) for num, dim in zip(num_shards, shard_dims)}

    def _physical_layout_matches_spatial_spec(self):
        """Return whether dataset shards match the requested spatial layout."""

        if self.spatial_shard_spec is None:
            return False
        return self._layout_by_dim(
            self.physical_num_shards, self.physical_shard_dims
        ) == self._layout_by_dim(
            self.spatial_shard_spec.num_shards,
            self.spatial_shard_spec.shard_dims,
        )

    def _physical_shard_id_for_spatial_spec(self):
        """Return the physical shard id selected by the spatial shard spec."""

        spec_indices_by_dim = {
            int(dim): int(index)
            for dim, index in zip(
                self.spatial_shard_spec.shard_dims,
                self.spatial_shard_spec.shard_indices,
            )
        }
        shard_indices = tuple(
            spec_indices_by_dim[int(dim)] for dim in self.physical_shard_dims
        )
        return shard_indices_to_id(shard_indices, self.physical_num_shards)

    def _select_physical_shard_id(self):
        """Select the physical shard file this dataset instance should read."""

        if not self.physical_shards:
            return 0
        if self.spatial_shard_spec is None:
            if self.physical_total_shards == 1:
                return 0
            raise RuntimeError(
                "Physical dataset has multiple shard files, but no SpatialShardSpec "
                "was provided. Use a DistConv layout matching the dataset."
            )
        if not self._physical_layout_matches_spatial_spec():
            raise RuntimeError(
                "Physical dataset shard layout does not match the requested "
                "DistConv layout. Physical dataset layout and DistConv layout "
                "must match. "
                f"dataset dc_num_shards={self.physical_num_shards}, "
                f"dataset dc_shard_dims={self.physical_shard_dims}, "
                f"dc_num_shards={self.spatial_shard_spec.num_shards}, "
                f"dc_shard_dims={self.spatial_shard_spec.shard_dims}"
            )

        return self._physical_shard_id_for_spatial_spec()

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
        if self.spatial_shard_spec is None or self.physical_shards:
            return img

        if self.dataset_format_version >= DATASET_FORMAT_VERSION:
            axis_map = {2: 1, 3: 2, 4: 3}
        else:
            axis_map = {2: 0, 3: 1, 4: 2}
        return self.spatial_shard_spec.slice_array(img, axis_map, "image")

    def _slice_mask_array(self, mask):
        if self.spatial_shard_spec is None or self.physical_shards:
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
        key = name if self.physical_shards else name + self.mask_suffix
        try:
            return self._mask_paths[key]
        except KeyError:
            raise KeyError(f"No mask file found for the ID {key} in {self.mask_dir}")

    def _materialize_shard_slice(self):
        return self.spatial_shard_spec is not None and not self.physical_shards

    def _load_prepared_mask(self, name):
        """Load, shard-slice, and label-prepare one mask as ``MASK_DTYPE``."""
        materialize = self._materialize_shard_slice()
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
        materialize = self._materialize_shard_slice()
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
