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

"""Unit tests for the numba-jitted fractal point kernel.

The jitted kernel ``generate_fractal_points`` is imported once at module load
so it compiles a single time and is reused across tests.
"""

import numpy as np

from ScaFFold.datagen.generate_fractal_points import generate_fractal_points

# A contractive map (spectral radius < 1) whose orbit stays bounded, and an
# expansive map (spectral radius >> 1) whose orbit diverges within a few steps.
# Layout: columns 0-8 are the 3x3 matrix, 9-11 the translation, 12 the
# probability of selecting transformation 0.


def _contractive_ifs_params():
    params = np.zeros((2, 13), dtype=np.float64)
    params[:, 0] = params[:, 4] = params[:, 8] = 0.5
    params[:, 9:12] = 0.1
    params[0, 12] = 0.5
    return params


def _expansive_ifs_params():
    params = np.zeros((2, 13), dtype=np.float64)
    params[:, 0] = params[:, 4] = params[:, 8] = 10.0
    params[:, 9:12] = 1.0
    params[0, 12] = 0.5
    return params


# ---------------------------------------------------------------------------
# generate_fractal_points: numpoints <= 0 is safe
# ---------------------------------------------------------------------------


def test_numpoints_zero_safe(fresh_python):
    """A request for zero points returns correctly-shaped empty arrays."""
    params = _contractive_ifs_params()
    points, ok = generate_fractal_points(params, 0)
    assert points.shape == (0, 3)
    assert points.dtype == params.dtype
    assert bool(ok) is True

    # numba reads NUMBA_BOUNDSCHECK at import time, so prove the zero-point path
    # performs no out-of-bounds access by compiling with bounds checking on in a
    # fresh process: it must return cleanly rather than raise IndexError.
    snippet = (
        "import numpy as np\n"
        "from ScaFFold.datagen.generate_fractal_points import generate_fractal_points\n"
        "params = np.zeros((2, 13), dtype=np.float64)\n"
        "params[0, 12] = 1.0\n"
        "points, ok = generate_fractal_points(params, 0)\n"
        "print('shape', points.shape[0], points.shape[1])\n"
    )
    out = fresh_python(snippet, env={"NUMBA_BOUNDSCHECK": "1"}, timeout=120)
    assert "shape 0 3" in out


# ---------------------------------------------------------------------------
# generate_fractal_points: runaway flag reflects divergence
# ---------------------------------------------------------------------------


def test_runaway_flag_detects_divergence():
    """The runaway flag is False for a diverging map and True for a bounded one."""
    _, ok_contractive = generate_fractal_points(_contractive_ifs_params(), 200)
    assert bool(ok_contractive) is True

    _, ok_expansive = generate_fractal_points(_expansive_ifs_params(), 200)
    assert bool(ok_expansive) is False
