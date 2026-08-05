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

"""The fractal library is keyed by the seed that produced it.

Categories and instances are *derived from* ``config.seed``: the IFS parameters
come from a seed-keyed candidate stream and every instance point cloud is
seeded from ``(seed, category, instance)``. Resume, however, is a pure
file-existence check, so a library laid out only by variance threshold and
point count let a run under one seed silently adopt another seed's data --
and then publish a dataset whose metadata claimed the new seed. Two datasets
with identical provenance and different content.

The fix puts the seed in the directory path, so the question "does this file
exist" is asked in a seed-specific place and can only ever be answered with
data that seed produced:

    <fract_base_dir>/var<variance_threshold>/seed<seed>/3DIFS_param/
    <fract_base_dir>/var<variance_threshold>/seed<seed>/instances/np<point_num>/

Everything here runs single-process at tiny scale (one category, 60-point
clouds) so the real generators run in well under a second.
"""

from __future__ import annotations

import hashlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from ScaFFold.datagen import category_search as cs
from ScaFFold.datagen import instance as inst
from ScaFFold.datagen import layout, volumegen

VT = 0.15
POINT_NUM = 60


def _contractive_params() -> np.ndarray:
    """A 2-map IFS whose orbit stays bounded, so generation is fast and finite."""
    params = np.zeros((2, 13), dtype=np.float64)
    params[:, 0] = params[:, 4] = params[:, 8] = 0.5
    params[1, 9] = params[1, 10] = params[1, 11] = 0.5
    params[0, 12] = 0.5
    return params


def _param_dir(fract_base: Path, seed: int) -> Path:
    """The category directory the new layout mandates, spelled out literally."""
    return fract_base / f"var{VT}" / f"seed{seed}" / "3DIFS_param"


def _instance_dir(fract_base: Path, seed: int) -> Path:
    """The instance directory the new layout mandates, spelled out literally."""
    return fract_base / f"var{VT}" / f"seed{seed}" / "instances" / f"np{POINT_NUM}"


def _seed_params(fract_base: Path, seed: int, n_categories: int = 1) -> Path:
    param_dir = _param_dir(fract_base, seed)
    param_dir.mkdir(parents=True, exist_ok=True)
    for category in range(n_categories):
        np.savetxt(
            param_dir / f"{category:06d}.csv", _contractive_params(), delimiter=","
        )
    return param_dir


def _inst_config(fract_base: Path, seed: int) -> Namespace:
    return Namespace(
        fract_base_dir=str(fract_base),
        n_categories=1,
        seed=seed,
        variance_threshold=VT,
        point_num=POINT_NUM,
        datagen_from_scratch=False,
        verbose=0,
    )


def _cs_config(fract_base: Path, seed: int) -> Namespace:
    return Namespace(
        fract_base_dir=str(fract_base),
        n_categories=1,
        seed=seed,
        variance_threshold=VT,
        point_num=POINT_NUM,
        normalize=1,
        datagen_from_scratch=False,
        datagen_batch_size=4,
        verbose=0,
    )


def _volumegen_config(dataset_dir: Path, fract_base: Path, seed: int) -> Namespace:
    return Namespace(
        dataset_dir=str(dataset_dir),
        fract_base_dir=str(fract_base),
        n_categories=1,
        n_instances_used_per_fractal=1,
        n_fracts_per_vol=1,
        seed=seed,
        variance_threshold=VT,
        val_split=0,
        vol_size=8,
        point_num=POINT_NUM,
        scale=1,
        verbose=0,
    )


def _library_digest(instance_dir: Path) -> tuple[int, str]:
    """(file count, content digest) for one instance directory."""
    digest = hashlib.sha256()
    files = sorted(instance_dir.rglob("*.npy"))
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return len(files), digest.hexdigest()


# ---------------------------------------------------------------------------
# The layout helper is the single definition of the seed-keyed paths.
# ---------------------------------------------------------------------------


def test_layout_helpers_key_every_path_by_seed(tmp_path):
    """Both library directories carry the seed, and differ across seeds."""
    fract_base = tmp_path / "fractals"

    for seed in (7, 999):
        config = _inst_config(fract_base, seed)
        assert Path(layout.category_param_dir(config)) == _param_dir(fract_base, seed)
        assert Path(layout.instance_dir(config)) == _instance_dir(fract_base, seed)

    assert layout.category_param_dir(_inst_config(fract_base, 7)) != (
        layout.category_param_dir(_inst_config(fract_base, 999))
    )
    assert layout.instance_dir(_inst_config(fract_base, 7)) != (
        layout.instance_dir(_inst_config(fract_base, 999))
    )


# ---------------------------------------------------------------------------
# Producers write under the seed; consumers read from under the seed.
# ---------------------------------------------------------------------------


def test_category_search_writes_under_the_seed_dir(tmp_path):
    """A generated category CSV lands in this seed's parameter directory."""
    fract_base = tmp_path / "fractals"

    cs.main(_cs_config(fract_base, seed=42))

    assert (_param_dir(fract_base, 42) / "000000.csv").exists()
    # Nothing was written to a seed-agnostic location.
    assert not (fract_base / f"var{VT}" / "3DIFS_param").exists()


def test_instances_are_written_under_the_seed_dir(tmp_path):
    """Instance point clouds land under this seed's instance directory."""
    fract_base = tmp_path / "fractals"
    _seed_params(fract_base, seed=7)

    inst.main(_inst_config(fract_base, seed=7))

    count, _digest = _library_digest(_instance_dir(fract_base, 7))
    assert count == 145
    assert not (fract_base / f"var{VT}" / "instances").exists()


def test_same_seed_resume_generates_nothing_new(tmp_path):
    """Re-running under the same seed reuses the library byte-for-byte.

    The resume path must stay cheap: the whole point of the library is that a
    second run under the same configuration regenerates nothing.
    """
    fract_base = tmp_path / "fractals"
    _seed_params(fract_base, seed=7)
    inst.main(_inst_config(fract_base, seed=7))

    instance_dir = _instance_dir(fract_base, 7)
    count_before, digest_before = _library_digest(instance_dir)
    mtimes_before = {p: p.stat().st_mtime_ns for p in sorted(instance_dir.rglob("*"))}

    inst.main(_inst_config(fract_base, seed=7))

    count_after, digest_after = _library_digest(instance_dir)
    assert (count_after, digest_after) == (count_before, digest_before)
    # No file was rewritten (0 new instances generated).
    assert {p: p.stat().st_mtime_ns for p in sorted(instance_dir.rglob("*"))} == (
        mtimes_before
    )


def test_different_seed_cannot_reuse_another_seeds_instances(tmp_path):
    """A second seed generates its own library instead of adopting the first.

    Before the fix this run reported "Generated 0 instances" and left the
    first seed's bytes in place, so the dataset built on top of it carried the
    wrong seed's data under the new seed's provenance.
    """
    fract_base = tmp_path / "fractals"
    _seed_params(fract_base, seed=7)
    _seed_params(fract_base, seed=999)

    inst.main(_inst_config(fract_base, seed=7))
    count_7, digest_7 = _library_digest(_instance_dir(fract_base, 7))

    inst.main(_inst_config(fract_base, seed=999))
    count_999, digest_999 = _library_digest(_instance_dir(fract_base, 999))

    # Both libraries are complete and independent...
    assert count_7 == count_999 == 145
    assert digest_999 != digest_7, "seed 999 reused seed 7's instances"
    # ...and the first seed's data was left untouched.
    assert _library_digest(_instance_dir(fract_base, 7)) == (count_7, digest_7)


def test_volumegen_reads_instances_for_its_own_seed(tmp_path):
    """volumegen resolves point clouds under the seed it was configured with."""
    fract_base = tmp_path / "fractals"
    _seed_params(fract_base, seed=7)
    inst.main(_inst_config(fract_base, seed=7))

    # Seed 7: the instances are where volumegen looks, so generation succeeds.
    volumegen.main(_volumegen_config(tmp_path / "ds7", fract_base, seed=7))
    assert list((tmp_path / "ds7" / "volumes").rglob("*.npy"))

    # Seed 999: a different library entirely. Nothing has been generated for
    # it, so volumegen must report the missing file rather than quietly
    # rasterizing seed 7's clouds.
    with pytest.raises(RuntimeError) as excinfo:
        volumegen.main(_volumegen_config(tmp_path / "ds999", fract_base, seed=999))
    assert "seed999" in str(excinfo.value)
