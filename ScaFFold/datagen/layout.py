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

"""On-disk layout of the fractal library.

Every artifact in the library is a deterministic function of the seed: category
IFS parameters come from a ``(seed, rank, attempt)`` candidate stream, and each
instance point cloud is generated from ``(seed, category, instance)``. Resume,
by contrast, is a pure file-existence test -- an instance is "already done" if
its file is on disk.

Those two facts only compose safely if the path itself carries the seed.
Without it, a run under a new seed found the previous seed's files, generated
nothing, and produced a dataset whose metadata advertised the new seed while
its contents came from the old one. Keying the directory by seed makes the
existence question seed-specific, so data from one seed can never be mistaken
for another's:

    <fract_base_dir>/var<variance_threshold>/seed<seed>/3DIFS_param/
    <fract_base_dir>/var<variance_threshold>/seed<seed>/instances/np<point_num>/

Libraries written under the older, seed-agnostic layout are simply not found
and are regenerated in the new location.

These helpers are the single definition of that layout; every producer and
consumer (``category_search``, ``instance``, ``volumegen``) goes through them
so the two sides cannot drift apart.
"""

from __future__ import annotations

import os


def library_root(config) -> str:
    """Return the root of the fractal library for this config's seed."""
    return os.path.join(
        str(config.fract_base_dir),
        f"var{config.variance_threshold}",
        f"seed{int(config.seed)}",
    )


def category_param_dir(config) -> str:
    """Return the directory holding this seed's category IFS parameter CSVs."""
    return os.path.join(library_root(config), "3DIFS_param")


def instance_dir(config) -> str:
    """Return the directory holding this seed's instance point clouds.

    Instances are additionally keyed by point count, which is a property of the
    cloud rather than of the category, so several point counts can coexist for
    one seed.
    """
    return os.path.join(library_root(config), "instances", f"np{config.point_num}")
