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

"""Data-quality and atomic-artifact tests for the datagen pipeline.

Covers four independent hazards in instance/volume generation:

* weighted point clouds that diverge to NaN/inf are rejected and regenerated
  (attenuated weights, else unweighted fallback) instead of being saved;
* ``points_to_voxelgrid`` refuses non-finite input rather than clipping
  garbage indices into the mask;
* instance files are written atomically and the resume scan ignores temp files
  and regenerates truncated ones;
* the mask scanner requires exactly one file per id;
* voxelization is isotropic and centered, and the dead ``scale`` knob is
  rejected with a clear message.

Everything runs single-process at tiny scale (small point counts, 16^3 grids)
so the real numba kernel compiles once and the suite stays fast.
"""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from ScaFFold.datagen import instance as inst
from ScaFFold.datagen import mask_detection as md
from ScaFFold.datagen.volumegen import points_to_voxelgrid, resolve_grid_size


# A contractive 2-map IFS (spectral radius < 1) whose orbit stays bounded at
# weight 1.0; columns 0-8 are the 3x3 matrix, 9-11 the translation, 12 the
# probability of selecting transformation 0.
def _contractive_params() -> np.ndarray:
    params = np.zeros((2, 13), dtype=np.float64)
    params[:, 0] = params[:, 4] = params[:, 8] = 0.5
    params[1, 9] = params[1, 10] = params[1, 11] = 0.5
    params[0, 12] = 0.5
    return params


def _make_config(fract_base: Path, *, point_num: int) -> Namespace:
    """Minimal config carrying just what ``instance.main`` reads."""
    return Namespace(
        fract_base_dir=str(fract_base),
        n_categories=1,
        seed=1234,
        variance_threshold=0.15,
        point_num=point_num,
        datagen_from_scratch=False,
    )


def _seed_category(fract_base: Path, *, point_num: int, keep: range) -> Path:
    """Create category 0's param CSV and pre-seed every instance in ``keep``.

    Pre-seeding all but instance 0 means a ``main`` run only has to generate the
    single missing instance, keeping the test fast. Returns the instance dir.
    """
    vt = 0.15
    param_dir = fract_base / f"var{vt}" / "3DIFS_param"
    param_dir.mkdir(parents=True)
    np.savetxt(param_dir / "000000.csv", _contractive_params(), delimiter=",")

    inst_dir = fract_base / f"var{vt}" / "instances" / f"np{point_num}" / "000000"
    inst_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in keep:
        np.save(inst_dir / f"000000_{i:04d}.npy", rng.random((10, 3)))
    return inst_dir


# ---------------------------------------------------------------------------
# F09: non-finite weighted instances are retried / fall back, never saved
# ---------------------------------------------------------------------------


def test_nonfinite_weighted_instance_retried():
    """An overflowing weight row yields a finite cloud via retry/fallback."""
    base = _contractive_params()

    # A weight that scales a matrix column astronomically turns the contractive
    # map expansive: the raw weighted attempt overflows to inf/NaN.
    weights = np.ones(12, dtype=np.float64)
    weights[0] = 1e300

    raw = base.copy()
    raw[:, :12] *= weights
    raw_points, raw_valid = inst.generate_single_instance(200, raw)
    assert not raw_valid  # RED intent: the naive weighted attempt is non-finite
    assert not np.isfinite(raw_points).all()

    # The validated helper recovers a finite cloud (attenuation or fallback).
    points = inst.generate_instance_points(200, base, weights, 0, 2, 42)
    assert points.shape == (200, 3)
    assert np.isfinite(points).all()


def test_nonfinite_instance_saved_finite(tmp_path, monkeypatch):
    """End-to-end: a run saves only finite instances even under bad weights."""
    fract_base = tmp_path / "fractals"
    point_num = 60
    inst_dir = _seed_category(fract_base, point_num=point_num, keep=range(1, 145))

    # Force instance 0's weight row to overflow, exercising the fallback path.
    orig_genfromtxt = np.genfromtxt

    def _patched_genfromtxt(fname, *args, **kwargs):
        arr = orig_genfromtxt(fname, *args, **kwargs)
        if "weights_ins145" in str(fname):
            arr = arr.copy()
            arr[0, 0] = 1e300
        return arr

    monkeypatch.setattr(inst.np, "genfromtxt", _patched_genfromtxt)
    inst.main(_make_config(fract_base, point_num=point_num))

    saved = inst_dir / "000000_0000.npy"
    assert saved.exists()
    assert np.isfinite(np.load(saved)).all()


# ---------------------------------------------------------------------------
# F09 defense: points_to_voxelgrid rejects non-finite input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_voxelgrid_rejects_nonfinite(bad):
    """A NaN/inf coordinate raises instead of scattering garbage voxels."""
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 1, (500, 3))
    points[0, 0] = bad
    with pytest.raises(ValueError):
        points_to_voxelgrid(points, 16)


# ---------------------------------------------------------------------------
# F32: instance save is atomic and resume handles temp / truncated files
# ---------------------------------------------------------------------------


def test_instance_save_atomic(tmp_path, monkeypatch):
    """A crash mid-save leaves no file at the final name; a prior file survives."""
    out_dir = tmp_path / "000000"
    out_dir.mkdir()
    final = out_dir / "000000_0000.npy"

    # A pre-existing complete instance that must survive the failed rewrite.
    good = np.arange(30, dtype=np.float64).reshape(10, 3)
    inst.save_instance_atomic(final, good)
    good_bytes = final.read_bytes()

    # np.save writes some bytes to the temp file, then the write is interrupted.
    def _partial_then_raise(file, arr, *args, **kwargs):
        file.write(b"\x93NUMPY partial garbage")
        raise OSError("simulated SIGKILL mid-write")

    monkeypatch.setattr(inst.np, "save", _partial_then_raise)

    with pytest.raises(OSError):
        inst.save_instance_atomic(final, np.zeros((10, 3)))

    # The final name still holds the original complete file, byte-for-byte.
    assert final.read_bytes() == good_bytes
    # No temp file was left behind under a name resume could stumble over.
    leftovers = list(out_dir.glob(f"{inst.TEMP_PREFIX}*"))
    assert leftovers == []


def test_resume_ignores_temp_files(tmp_path):
    """A leftover temp-named file is cleaned and the item is regenerated."""
    fract_base = tmp_path / "fractals"
    point_num = 60
    inst_dir = _seed_category(fract_base, point_num=point_num, keep=range(1, 145))

    # Simulate a killed job: a temp file for instance 0 with a dotted prefix the
    # six-digit resume glob cannot match. It must NOT be accepted as complete.
    temp_file = inst_dir / f"{inst.TEMP_PREFIX}000000_0000.npy"
    temp_file.write_bytes(b"\x93NUMPY truncated")
    final = inst_dir / "000000_0000.npy"
    assert not final.exists()

    inst.main(_make_config(fract_base, point_num=point_num))

    # The real instance now exists and is finite; the temp file is gone.
    assert final.exists()
    assert np.isfinite(np.load(final)).all()
    assert not temp_file.exists()


def test_resume_rejects_truncated(tmp_path):
    """A truncated .npy at the final name is detected and regenerated."""
    fract_base = tmp_path / "fractals"
    point_num = 60
    inst_dir = _seed_category(fract_base, point_num=point_num, keep=range(1, 145))

    # Generate everything once so instance 0 exists and loads cleanly.
    inst.main(_make_config(fract_base, point_num=point_num))
    victim = inst_dir / "000000_0000.npy"
    good_size = victim.stat().st_size
    assert inst._is_valid_npy(str(victim))

    # Truncate it the way a walltime SIGKILL mid-write would: valid header,
    # partial data. The old name-only glob would accept this forever.
    with open(victim, "r+b") as handle:
        handle.truncate(80)
    assert not inst._is_valid_npy(str(victim))

    # Resuming regenerates the truncated file back to a full, loadable array.
    inst.main(_make_config(fract_base, point_num=point_num))
    assert victim.stat().st_size == good_size
    assert np.isfinite(np.load(victim)).all()


# ---------------------------------------------------------------------------
# F62: mask scanner requires exactly one file per id
# ---------------------------------------------------------------------------


def _write_mask(path: Path, values) -> None:
    with open(path, "wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.uint16))


def test_mask_glob_exact_match(tmp_path):
    """No match names the id; multiple matches list the offending files."""
    masks = tmp_path / "masks"
    masks.mkdir()
    _write_mask(masks / "0_mask.npy", [[0, 1], [2, 0]])

    # An extensionless id ('README' -> glob 'README.*') matches nothing: the old
    # blind [0] raised an opaque IndexError; now it names the id.
    with pytest.raises(AssertionError) as no_match:
        md.unique_mask_values("README", masks)
    assert "README" in str(no_match.value)

    # Two files sharing a stem must not be picked arbitrarily.
    _write_mask(masks / "5_mask.npy", [0, 3, 3, 0])
    _write_mask(masks / "5_mask.npy.bak", [0, 9, 9, 9])
    with pytest.raises(AssertionError) as multi:
        md.unique_mask_values("5_mask", masks)
    message = str(multi.value)
    assert "5_mask.npy" in message and "5_mask.npy.bak" in message


# ---------------------------------------------------------------------------
# F64: isotropic + centered normalization; the scale knob is rejected
# ---------------------------------------------------------------------------


def test_normalization_isotropic():
    """A 2:1:1 point cloud keeps its aspect ratio and is centered in the grid."""
    rng = np.random.default_rng(0)
    grid_size = 16
    # Physical extents 2 : 1 : 1 (well-sampled so the occupied box fills them).
    points = rng.uniform(0, 1, (8000, 3)) * np.array([2.0, 1.0, 1.0])

    grid = points_to_voxelgrid(points, grid_size)
    occ = np.argwhere(grid)
    lo = occ.min(axis=0)
    hi = occ.max(axis=0)
    span = hi - lo + 1

    # The longest axis fills the grid; the others are ~half, within a voxel.
    assert span[0] == grid_size
    assert abs(int(span[1]) - grid_size // 2) <= 1
    assert abs(int(span[2]) - grid_size // 2) <= 1

    # The shorter axes are centered: leading and trailing margins are equal +-1.
    for axis in (1, 2):
        leading = int(lo[axis])
        trailing = int(grid_size - 1 - hi[axis])
        assert abs(leading - trailing) <= 1


def test_scale_config_rejected():
    """resolve_grid_size returns vol_size for scale==1 and rejects other values."""
    ok = Namespace(vol_size=16, scale=1)
    assert resolve_grid_size(ok) == 16

    # A non-1 scale is a dead knob; it must fail clearly here rather than trip an
    # opaque shape assertion deep inside the generation loop.
    bad = Namespace(vol_size=16, scale=0.5)
    with pytest.raises(ValueError) as info:
        resolve_grid_size(bad)
    assert "scale" in str(info.value)
