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

"""Tests for the mask-value scan (mask_detection)."""

from pathlib import Path

import numpy as np
import pytest

from ScaFFold.datagen import mask_detection


def _make_masks(directory, n, ext=".npy"):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = np.array([[i % 3, (i + 1) % 3]], dtype=np.uint16)
        np.save(directory / f"{i:06d}{ext}", arr)


def test_index_masks_by_stem_maps_each_id(tmp_path):
    d = tmp_path / "masks"
    _make_masks(d, 5)
    mapping = mask_detection._index_masks_by_stem(d)
    assert set(mapping) == {f"{i:06d}" for i in range(5)}
    assert all(isinstance(p, Path) and p.exists() for p in mapping.values())


def test_index_masks_rejects_duplicate_stem(tmp_path):
    d = tmp_path / "masks"
    _make_masks(d, 2)
    # A sibling sharing the stem must be rejected, naming the id.
    np.save(d / "000000.npz", np.array([9], dtype=np.uint16))
    # np.save on a ".npz" name actually writes ".npz.npy"; force a real clash.
    (d / "000000.dat").write_bytes(b"x")
    with pytest.raises(ValueError, match="000000"):
        mask_detection._index_masks_by_stem(d)


def test_unique_mask_values_reads_given_path(tmp_path):
    d = tmp_path / "masks"
    _make_masks(d, 1)
    path = d / "000000.npy"
    vals = mask_detection.unique_mask_values(path)
    assert set(vals.tolist()) == {0, 1}


def test_scan_does_not_glob_per_id(tmp_path, monkeypatch):
    # The scan must resolve paths from a single directory listing, not a
    # per-id glob. Count Path.glob calls: zero after the fix.
    d = tmp_path / "masks"
    _make_masks(d, 6)

    calls = {"n": 0}
    import pathlib

    real_glob = pathlib.Path.glob

    def counting_glob(self, pattern):
        calls["n"] += 1
        return real_glob(self, pattern)

    monkeypatch.setattr(pathlib.Path, "glob", counting_glob)

    mapping = mask_detection._index_masks_by_stem(d)
    paths = [mapping[s] for s in sorted(mapping)]
    _ = [mask_detection.unique_mask_values(p) for p in paths]
    assert calls["n"] == 0
