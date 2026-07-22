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

"""Tests for dataset loading (``ScaFFold/utils/data_loading.py``).

Two properties are pinned here:

* The index -> file mapping is derived from a *sorted* directory listing, so it
  is byte-identical regardless of the arbitrary order ``os.listdir`` returns
  (and, when a process group is live, a cross-rank guard turns any residual
  divergence into a hard error instead of silently stitched-together samples).

* Legacy (v1) datasets remap raw mask values through a single *global* table
  (the union of every per-split ``*unique_mask_vals`` pickle), so a category
  missing from one split can no longer shift class indices between train and
  validation. v2 datasets, which store dense class ids, are untouched.
"""

from __future__ import annotations

import pickle
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import ScaFFold.utils.data_loading as dl
from ScaFFold.utils.data_loading import BasicDataset, FractalDataset
from ScaFFold.utils.data_types import MASK_DTYPE, VOLUME_DTYPE
from tests.helpers import mpi_runner

RANK_SCRIPT = (
    Path(__file__).resolve().parent
    / "helpers"
    / "rank_scripts"
    / "data_loading_ids_hash_guard.py"
)


# ---------------------------------------------------------------------------
# local dataset builders (independent of conftest so category coverage and
# on-disk voxel values can be controlled precisely)
# ---------------------------------------------------------------------------


def _build_v2_constant_dataset(root: Path, n_volumes: int, n: int = 4) -> Path:
    """v2 dataset where volume/mask ``k`` is filled with the constant ``k``.

    The sorted id order is ``vol_00 .. vol_{n_volumes-1}``, so ``dataset[i]``
    read back as a voxel value reveals which file the index resolved to.
    """
    vol_dir = root / "volumes" / "training"
    mask_dir = root / "masks" / "training"
    vol_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n_volumes):
        img = np.full((3, n, n, n), float(k), dtype=VOLUME_DTYPE)
        msk = np.full((n, n, n), k, dtype=MASK_DTYPE)
        np.save(vol_dir / f"vol_{k:02d}.npy", img)
        np.save(mask_dir / f"vol_{k:02d}_mask.npy", msk)
    for name in ("train_unique_mask_vals", "val_unique_mask_vals"):
        with open(root / name, "wb") as handle:
            pickle.dump({"mask_values": list(range(n_volumes))}, handle)
    import yaml

    with open(root / "meta.yaml", "w") as handle:
        yaml.safe_dump({"dataset_format_version": 2}, handle)
    return root


def _build_v1_split_dataset(
    root: Path,
    raw_mask: np.ndarray,
    volume: np.ndarray,
    train_vals,
    val_vals,
) -> Path:
    """v1 (legacy, no ``meta.yaml``) dataset with per-split mask-value pickles.

    The *same* ``raw_mask`` / ``volume`` pair is written into both splits so any
    difference in the remapped labels is attributable purely to the per-split
    mask-value lists (``train_vals`` vs ``val_vals``).
    """
    dirs = {}
    for split in ("training", "validation"):
        for kind in ("volumes", "masks"):
            d = root / kind / split
            d.mkdir(parents=True, exist_ok=True)
            dirs[(kind, split)] = d
    for split in ("training", "validation"):
        np.save(dirs[("volumes", split)] / "vol_000.npy", volume)
        np.save(dirs[("masks", split)] / "vol_000_mask.npy", raw_mask)
    with open(root / "train_unique_mask_vals", "wb") as handle:
        pickle.dump({"mask_values": list(train_vals)}, handle)
    with open(root / "val_unique_mask_vals", "wb") as handle:
        pickle.dump({"mask_values": list(val_vals)}, handle)
    return root


# ---------------------------------------------------------------------------
# F11: index -> file mapping must not depend on os.listdir order
# ---------------------------------------------------------------------------


def test_ids_order_independent_of_listdir(tmp_path, monkeypatch):
    """``dataset[i]`` resolves to the same file for any ``listdir`` order."""
    root = _build_v2_constant_dataset(tmp_path / "ds", n_volumes=6)
    vol_dir = root / "volumes" / "training"
    mask_dir = root / "masks" / "training"
    data_dir = root / "train_unique_mask_vals"

    real_listdir = dl.listdir
    sorted_files = sorted(real_listdir(vol_dir))

    def make_with_order(order):
        def fake_listdir(_path):
            return list(order)

        monkeypatch.setattr(dl, "listdir", fake_listdir)
        try:
            return FractalDataset(vol_dir, mask_dir, data_dir=data_dir)
        finally:
            monkeypatch.setattr(dl, "listdir", real_listdir)

    # Sorted baseline: dataset[i] must carry constant voxel value i.
    baseline = make_with_order(sorted_files)
    baseline_vals = [int(baseline[i]["image"][0, 0, 0, 0].item()) for i in range(6)]
    assert baseline_vals == list(range(6))

    # Two arbitrary readdir orders of the identical file set.
    order_a = list(sorted_files)
    random.Random(1).shuffle(order_a)
    order_b = list(reversed(sorted_files))
    assert order_a != sorted_files and order_b != sorted_files

    for order in (order_a, order_b):
        ds = make_with_order(order)
        vals = [int(ds[i]["image"][0, 0, 0, 0].item()) for i in range(6)]
        assert vals == baseline_vals, (
            f"index->file mapping followed listdir order {order} "
            f"(got {vals}, expected sorted baseline {baseline_vals})"
        )
        assert ds.ids == baseline.ids


@pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    reason="requires torch.distributed with the gloo backend",
)
def test_ids_hash_guard_raises_on_divergence(tmp_path):
    """Two ranks with divergent ``listdir`` views must fail dataset creation.

    The rank script gives non-zero ranks a strictly smaller id set for the same
    directory. Both ranks print their own id digest (which differ, proving the
    divergence is real) and then attempt construction. The consistency guard
    must fire on every rank, so the job exits non-zero with a ``GUARD_RAISED``
    marker per rank -- rather than both ranks silently building mismatched
    datasets and exiting 0.
    """
    dataset_dir = tmp_path / "guard_ds"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rc, out, err = mpi_runner.torchrun_gloo(
        str(RANK_SCRIPT),
        n=2,
        timeout=120,
        env={"DATASET_DIR": str(dataset_dir), "N_VOLUMES": "6"},
    )

    # The two ranks genuinely observed different id sets (the precondition the
    # guard defends against); this holds whether or not the guard is present.
    # Concurrent ranks can interleave stdout without a newline, so scan the
    # whole stream for markers rather than parsing line by line.
    hashes = dict(re.findall(r"RANK (\d+) IDS_HASH ([0-9a-f]+)", out))
    assert {"0", "1"} <= set(hashes), (
        f"expected an id digest from both ranks\nstdout:\n{out}\nstderr:\n{err}"
    )
    assert hashes["0"] != hashes["1"], "ranks should observe divergent id sets"

    # Guard behavior: both ranks raise and the job exits non-zero.
    assert rc != 0, (
        f"expected non-zero exit from the id-consistency guard, got rc={rc}\n"
        f"stdout:\n{out}\nstderr:\n{err}"
    )
    raised = set(re.findall(r"RANK (\d+) GUARD_RAISED", out))
    assert {"0", "1"} <= raised, (
        f"both ranks must raise the id-consistency guard\nstdout:\n{out}"
    )
    assert "CONSTRUCTED" not in out, (
        f"a rank silently constructed a divergent dataset\nstdout:\n{out}"
    )


# ---------------------------------------------------------------------------
# F42: legacy label remapping must use one global (union) table
# ---------------------------------------------------------------------------


def test_v1_label_mapping_global(tmp_path):
    """A raw label present in both splits maps to the same class id in each.

    Category 4 (raw value 5) is present in training but absent from validation,
    so the per-split lists disagree above the gap. With a single global mapping
    (the union of both pickles) an identical raw mask must remap identically in
    the train and val datasets.
    """
    raw_mask = np.zeros((4, 4, 4), dtype=np.uint16)
    for v in range(7):  # raw labels 0..6 present in the file
        raw_mask[v // 4, v % 4, :] = v
    volume = np.random.rand(4, 4, 4, 1).astype(np.float32)

    train_vals = [0, 1, 2, 3, 4, 5, 6]
    val_vals = [0, 1, 2, 3, 4, 6]  # raw value 5 (category 4) missing from val

    root = _build_v1_split_dataset(
        tmp_path / "v1", raw_mask, volume, train_vals, val_vals
    )

    train_set = FractalDataset(
        root / "volumes" / "training",
        root / "masks" / "training",
        data_dir=root / "train_unique_mask_vals",
    )
    val_set = FractalDataset(
        root / "volumes" / "validation",
        root / "masks" / "validation",
        data_dir=root / "val_unique_mask_vals",
    )
    assert train_set.dataset_format_version == 1
    assert val_set.dataset_format_version == 1

    train_lbl = train_set[0]["mask"].numpy()
    val_lbl = val_set[0]["mask"].numpy()

    # Every raw value present in both files must map to the same class index.
    for v in range(7):
        sel = raw_mask == v
        if v == 5:
            # raw value 5 is absent from the val volumes in a real dataset;
            # here it is only meaningful that the *shared* labels agree.
            continue
        t = np.unique(train_lbl[sel])
        w = np.unique(val_lbl[sel])
        assert t.tolist() == w.tolist(), (
            f"raw value {v} remapped to train {t.tolist()} but val {w.tolist()}"
        )

    # The whole mask (for shared labels) is identical across splits.
    shared = raw_mask != 5
    assert np.array_equal(train_lbl[shared], val_lbl[shared])

    # Concretely: raw 6 keeps class index 6 in both (union table is 0..6).
    assert np.unique(train_lbl[raw_mask == 6]).tolist() == [6]
    assert np.unique(val_lbl[raw_mask == 6]).tolist() == [6]


def test_v2_datasets_unaffected(tiny_dataset):
    """v2 datasets load byte-identically to a direct np.load + prepare."""
    root = tiny_dataset(n_categories=3, n_train=3, n_val=2, n=16)
    vol_dir = root / "volumes" / "training"
    mask_dir = root / "masks" / "training"

    ds = FractalDataset(vol_dir, mask_dir, data_dir=root / "train_unique_mask_vals")
    assert ds.dataset_format_version == 2

    for idx, name in enumerate(sorted(ds.ids)):
        item = ds[ds.ids.index(name)]

        raw_vol = np.load(vol_dir / f"{name}.npy", allow_pickle=False)
        raw_mask = np.load(mask_dir / f"{name}_mask.npy", allow_pickle=False)
        expected_img = BasicDataset._prepare_optimized_image(raw_vol)
        expected_mask = BasicDataset._prepare_optimized_mask(raw_mask)

        assert np.array_equal(item["image"].numpy(), expected_img)
        assert np.array_equal(item["mask"].numpy(), expected_mask)
        assert item["image"].dtype == torch.float32
        assert item["mask"].dtype == torch.int64

    # mask_values is loaded verbatim for bookkeeping (dense ids, no remap).
    assert ds.mask_values == list(range(4))
