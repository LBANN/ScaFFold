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
from ScaFFold.datagen import layout
from ScaFFold.datagen import mask_detection as md
from ScaFFold.datagen.volumegen import (
    load_np_ptcloud,
    points_to_voxel_indices,
    points_to_voxelgrid,
    resolve_grid_size,
)


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
    The library lives under the seed-keyed layout, so the paths are derived from
    the same config the run under test uses.
    """
    config = _make_config(fract_base, point_num=point_num)
    param_dir = Path(layout.category_param_dir(config))
    param_dir.mkdir(parents=True)
    np.savetxt(param_dir / "000000.csv", _contractive_params(), delimiter=",")

    inst_dir = Path(layout.instance_dir(config)) / "000000"
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
# R36: a category's IFS parameters are parsed once, not once per instance
# ---------------------------------------------------------------------------


def test_category_params_parsed_once_per_category(tmp_path, monkeypatch):
    """``main`` parses each category CSV once per rank, not once per work item.

    The parse used to sit inside the per-item loop, so a full generation read
    and re-parsed the same small CSV 145 times per category off the shared
    filesystem. Work items for a category are contiguous in the block
    partition, so a one-entry cache collapses that to one parse per category.
    """
    fract_base = tmp_path / "fractals"
    point_num = 60
    n_categories = 2
    missing_per_category = 3

    config = _make_config(fract_base, point_num=point_num)
    config.n_categories = n_categories

    param_dir = Path(layout.category_param_dir(config))
    param_dir.mkdir(parents=True)
    instance_root = Path(layout.instance_dir(config))
    rng = np.random.default_rng(0)
    for category in range(n_categories):
        np.savetxt(
            param_dir / f"{category:06d}.csv", _contractive_params(), delimiter=","
        )
        # Pre-seed all but a few instances so the run stays fast; the ones left
        # missing are what the loop (and the parse) actually iterates over.
        inst_dir = instance_root / f"{category:06d}"
        inst_dir.mkdir(parents=True)
        for i in range(missing_per_category, 145):
            np.save(inst_dir / f"{category:06d}_{i:04d}.npy", rng.random((10, 3)))

    parses = []
    real_genfromtxt = np.genfromtxt

    def counting_genfromtxt(fname, *args, **kwargs):
        parses.append(Path(str(fname)).name)
        return real_genfromtxt(fname, *args, **kwargs)

    monkeypatch.setattr(inst.np, "genfromtxt", counting_genfromtxt)
    inst.main(config)

    category_parses = [name for name in parses if name[0].isdigit()]
    # Every missing instance was generated ...
    for category in range(n_categories):
        for i in range(missing_per_category):
            assert (
                instance_root / f"{category:06d}" / f"{category:06d}_{i:04d}.npy"
            ).exists()
    # ... from n_categories parses, not one per (category, instance) item.
    assert sorted(category_parses) == ["000000.csv", "000001.csv"]


# ---------------------------------------------------------------------------
# F62: mask scanner requires exactly one file per id
# ---------------------------------------------------------------------------


def _write_mask(path: Path, values) -> None:
    with open(path, "wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.uint16))


def test_mask_stem_index_rejects_ambiguous(tmp_path):
    """Two files sharing a stem are rejected by name rather than picked blind."""
    masks = tmp_path / "masks"
    masks.mkdir()
    _write_mask(masks / "0_mask.npy", [[0, 1], [2, 0]])

    # A single file per stem resolves cleanly.
    mapping = md._index_masks_by_stem(masks)
    assert set(mapping) == {"0_mask"}

    # A sibling sharing the stem must not be picked arbitrarily; the error names
    # both offending files so the ambiguity is actionable.
    _write_mask(masks / "5_mask.npy", [0, 3, 3, 0])
    (masks / "5_mask.dat").write_bytes(b"stale")
    with pytest.raises(ValueError) as multi:
        md._index_masks_by_stem(masks)
    message = str(multi.value)
    assert "5_mask" in message


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


# ---------------------------------------------------------------------------
# F34: rasterization scatters point indices instead of traversing a dense grid
# ---------------------------------------------------------------------------


def _reference_indices(points: np.ndarray, grid_size: int, eps=1e-6, *, clip=True):
    """The index computation of ``points_to_voxel_indices``, spelled out.

    Two tests need to see inside the function: the scatter-vs-dense equivalence
    check (which needs the per-point indices the dense grid was built from) and
    the centering check (which needs the indices *before* ``np.clip`` hides
    out-of-range bins). Every test using this asserts the replica reproduces the
    real function's output, so it cannot silently drift from it.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    voxel_size = (float((maxs - mins).max()) + eps) / grid_size
    scaled = (points - mins) / voxel_size
    offset = (grid_size - scaled.max(axis=0)) / 2.0
    idx = np.floor(scaled + offset).astype(int)
    return np.clip(idx, 0, grid_size - 1) if clip else idx


def test_voxel_indices_match_dense_grid():
    """The index API reproduces the dense boolean grid exactly."""
    rng = np.random.default_rng(0)
    grid_size = 16
    points = rng.uniform(-1, 1, (4000, 3))

    idx = points_to_voxel_indices(points, grid_size)

    # Reference dense grid built the old way, from the reference index math.
    ref_idx = _reference_indices(points, grid_size)
    reference = np.zeros((grid_size,) * 3, dtype=bool)
    reference[ref_idx[:, 0], ref_idx[:, 1], ref_idx[:, 2]] = True

    # Grid scattered from the returned indices is identical to the reference,
    # and the wrapper produces the same grid.
    scattered = np.zeros((grid_size,) * 3, dtype=bool)
    scattered[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    assert np.array_equal(scattered, reference)
    assert np.array_equal(points_to_voxelgrid(points, grid_size), reference)

    # Indices are unique (one write per occupied voxel) and in bounds.
    assert np.array_equal(idx, np.unique(idx, axis=0))
    assert idx.min() >= 0 and idx.max() <= grid_size - 1


def test_scatter_paint_matches_boolean_mask():
    """Scatter-painting volume/mask equals the boolean-mask-painted reference."""
    rng = np.random.default_rng(1)
    grid_size = 16
    n_fracts = 3
    clouds = [rng.uniform(-1, 1, (3000, 3)) for _ in range(n_fracts)]
    colors = rng.random((n_fracts, 3)).astype(np.float32)

    # Reference: dense boolean grid + full-volume boolean-mask assignment (the
    # pre-refactor generation-loop painting).
    vol_ref = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
    mask_ref = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint16)
    # Under test: scatter the returned indices directly.
    vol_scatter = np.zeros_like(vol_ref)
    mask_scatter = np.zeros_like(mask_ref)

    for cat, (points, color) in enumerate(zip(clouds, colors)):
        grid = points_to_voxelgrid(points, grid_size)
        vol_ref[grid] = color
        mask_ref[grid] = cat + 1

        idx = points_to_voxel_indices(points, grid_size)
        vol_scatter[idx[:, 0], idx[:, 1], idx[:, 2]] = color
        mask_scatter[idx[:, 0], idx[:, 1], idx[:, 2]] = cat + 1

    assert np.array_equal(vol_scatter, vol_ref)
    assert np.array_equal(mask_scatter, mask_ref)


# ---------------------------------------------------------------------------
# F66: instance point clouds are stored and loaded as float32
# ---------------------------------------------------------------------------


def test_instance_saved_float32(tmp_path):
    """A generated instance file is float32 on disk, and loads as float32."""
    fract_base = tmp_path / "fractals"
    point_num = 60
    inst_dir = _seed_category(fract_base, point_num=point_num, keep=range(1, 145))

    inst.main(_make_config(fract_base, point_num=point_num))

    saved = inst_dir / "000000_0000.npy"
    assert saved.exists()
    assert np.load(saved).dtype == np.float32

    # The load path used by volumegen also yields float32 (no float64 upcast).
    assert load_np_ptcloud(str(saved)).dtype == np.float32


def test_load_downcasts_legacy_float64(tmp_path):
    """A legacy float64 file is downcast to float32 on load."""
    legacy = tmp_path / "legacy.npy"
    np.save(legacy, np.random.default_rng(0).random((10, 3)).astype(np.float64))
    assert np.load(legacy).dtype == np.float64
    assert load_np_ptcloud(str(legacy)).dtype == np.float32


def test_dataset_version_bumped_past_float64_era():
    """The dataset-reuse version marker is > 2 so float64 datasets aren't reused.

    A float64-generated dataset was stamped version 2; storing float32 shifts a
    few boundary voxels, so the reuse marker must advance past 2 to keep the two
    from being silently interchanged.
    """
    from ScaFFold.datagen import get_dataset as gd

    assert gd.DATASET_FORMAT_VERSION > 2


# ---------------------------------------------------------------------------
# R33: voxel centering is a whole voxel, not half of one
# ---------------------------------------------------------------------------


def _assert_replica_tracks_real(grid_size: int = 16) -> None:
    """Pin ``_reference_indices`` to the real function on a sparse cloud.

    A sparse cloud is essential here: a dense one occupies every voxel under
    any offset, so the comparison would pass vacuously.
    """
    sparse = np.random.default_rng(7).random((300, 3)).astype(np.float32)
    assert np.array_equal(
        np.unique(_reference_indices(sparse, grid_size), axis=0),
        points_to_voxel_indices(sparse, grid_size),
    ), "the replicated index arithmetic no longer matches points_to_voxel_indices"


def test_voxelization_never_bins_outside_the_grid():
    """No point lands outside ``[0, grid_size)`` before the safety clip.

    The centering offset positions a span of ``span`` voxels inside a grid of
    ``grid_size`` voxels, so the free space to split between the two margins is
    ``grid_size - span``. Using ``grid_size - 1 - span`` shifted every cloud
    half a voxel toward the origin: points in the first half-voxel of each
    filled axis floored to -1, and ``np.clip`` quietly folded them into bin 0.
    """
    grid_size = 16
    _assert_replica_tracks_real(grid_size)
    rng = np.random.default_rng(1234)
    points = rng.random((200_000, 3)).astype(np.float32)

    pre_clip = _reference_indices(points, grid_size, clip=False)
    assert pre_clip.min() >= 0, (
        f"{int((pre_clip < 0).any(axis=1).sum())} of {len(points)} points floored "
        "below bin 0 and were clipped back in"
    )
    assert pre_clip.max() <= grid_size - 1


def test_voxelization_density_is_uniform_at_the_boundaries():
    """A uniform cloud fills the boundary planes like the interior ones.

    The half-voxel shift piled the clipped points onto plane 0 (1.5x the
    interior density) and starved the far plane (0.5x), a systematic
    misregistration in every generated volume and mask.
    """
    grid_size = 16
    _assert_replica_tracks_real(grid_size)
    rng = np.random.default_rng(1234)
    points = rng.random((200_000, 3)).astype(np.float32)

    idx = _reference_indices(points, grid_size)
    counts = np.bincount(idx[:, 0], minlength=grid_size)
    interior = counts[2:-2].mean()
    assert 0.9 <= counts[0] / interior <= 1.1, (
        f"boundary plane 0 holds {counts[0] / interior:.2f}x the interior density"
    )
    assert 0.9 <= counts[-1] / interior <= 1.1, (
        f"boundary plane {grid_size - 1} holds {counts[-1] / interior:.2f}x the "
        "interior density"
    )


def test_dataset_version_bumped_past_half_voxel_era():
    """The reuse marker advanced past 3: pre-fix datasets are misregistered.

    Correcting the offset changes the voxel contents of every generated volume
    and mask, so a dataset built before the fix must not be handed to a run
    after it. Bumping the version both stops the reuse scan from matching those
    directories and changes the config_id they hash to.
    """
    from ScaFFold.datagen import get_dataset as gd

    assert gd.DATASET_FORMAT_VERSION > 3
