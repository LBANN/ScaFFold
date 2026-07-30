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

import numba

# -*- coding: utf-8 -*-
import numpy as np

DEFAULT_NP_DTYPE = np.float64


@numba.njit
def generate_fractal_points(params: np.array, numpoints: int):
    """
    Generate fractal points from an IFS, using jit compilation for significant speedup.

    Assumes `params` is a 2x13 array where:
      - Columns 0-8: 3x3 matrix coefficients.
      - Columns 9-11: translation components.
      - Column 12: normalized probability for transformation 0 (transformation 1's probability is 1 - p).

    Note that, due to finite floating point precision, the fractal point generation
    process sometimes exhibits runaway values. Rather than raising (numba jit compilation
    does not play well with exception handling), the generation loop tracks whether any
    coordinate exceeds a large finite bound and reports the result via `runaway_check_pass`
    so callers can reject the attempt.

    Parameters
    ----------
    params : np.array
        A numpy array containing the IFS parameters for this fractal category attempt.
    numpoints : int
        The number of fractal points to generate.

    Returns
    -------
    points : np.array
        A numpy array of fractal points for this attempt.
    runaway_check_pass : bool
        A bool: False if runaway values were found when generating fractal points; True otherwise.
    """

    # Guard against a non-positive request: allocating and then writing the
    # origin would index past the end of the zero-length buffers.
    if numpoints <= 0:
        empty = np.empty((0, 3), dtype=params.dtype)
        return empty, True

    # Get probability for transformation 0
    p0 = params[0, 12]

    # Set up arrays for fractal data
    x_arr = np.empty(numpoints, dtype=params.dtype)
    y_arr = np.empty(numpoints, dtype=params.dtype)
    z_arr = np.empty(numpoints, dtype=params.dtype)

    # Set the first point to the origin
    x_arr[0] = 0.0
    y_arr[0] = 0.0
    z_arr[0] = 0.0

    # Coordinates whose magnitude exceeds this bound are treated as runaway.
    runaway_bound = 1e10

    # Iteratively calculate fractal points
    runaway_check_pass = True
    for n in range(1, numpoints):
        r = np.random.rand()
        if r < p0:
            t = 0
        else:
            t = 1

        # Apply affine transformation for the selected transformation t
        x_prev = x_arr[n - 1]
        y_prev = y_arr[n - 1]
        z_prev = z_arr[n - 1]

        x_arr[n] = (
            x_prev * params[t, 0]
            + y_prev * params[t, 1]
            + z_prev * params[t, 2]
            + params[t, 9]
        )
        y_arr[n] = (
            x_prev * params[t, 3]
            + y_prev * params[t, 4]
            + z_prev * params[t, 5]
            + params[t, 10]
        )
        z_arr[n] = (
            x_prev * params[t, 6]
            + y_prev * params[t, 7]
            + z_prev * params[t, 8]
            + params[t, 11]
        )

        # Flag runaway growth. A NaN comparison is always False, so any
        # non-finite coordinate produced downstream is caught by the caller's
        # value checks; this bound catches divergence before it overflows.
        if (
            abs(x_arr[n]) > runaway_bound
            or abs(y_arr[n]) > runaway_bound
            or abs(z_arr[n]) > runaway_bound
        ):
            runaway_check_pass = False

    # Group XYZ coords into single array
    points = np.empty((numpoints, 3), dtype=params.dtype)
    for i in range(numpoints):
        points[i, 0] = x_arr[i]
        points[i, 1] = y_arr[i]
        points[i, 2] = z_arr[i]

    return points, runaway_check_pass
