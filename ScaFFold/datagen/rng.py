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

"""Deterministic seeding helpers shared by the datagen pipeline.

``generate_fractal_points`` is ``@numba.njit`` and draws from numba's own
process-internal RNG. That stream is *separate* from NumPy's Python-level RNG
and can only be seeded by calling ``np.random.seed`` from inside jitted code,
so a plain ``np.random.seed(...)`` in interpreted Python has no effect on it.
``seed_numba`` provides that in-jit seeding.

``derive_seed`` maps a tuple of integer keys (e.g. the configured base seed
plus stable work-item coordinates) to a 32-bit seed with a fixed, unsalted
hash. Because it depends only on the keys it is handed, seeds are reproducible
across processes and independent of MPI world size, rank assignment, and resume
state -- unlike a single per-rank seed, whose effect on any given work item
depends on how the work happens to be partitioned.
"""

import hashlib

import numba
import numpy as np

# numba's np.random.seed consumes a 32-bit seed.
SEED_MASK = 0xFFFFFFFF


@numba.njit(cache=True)
def seed_numba(seed):
    """Seed numba's process-internal RNG from within jitted code.

    Must be called from compiled code: a Python-level ``np.random.seed`` does
    not reach the RNG used by other ``@numba.njit`` functions in this process.
    """
    np.random.seed(seed)


def derive_seed(*keys: int) -> int:
    """Derive a stable 32-bit seed from a sequence of integer keys.

    The mapping is a fixed BLAKE2b digest of the little-endian key bytes, so it
    is identical across processes and Python invocations (unlike the builtin
    ``hash``, which is salted per process). Negative keys are folded into the
    unsigned 64-bit range before hashing.
    """
    h = hashlib.blake2b(digest_size=8)
    for key in keys:
        h.update((int(key) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))
    return int.from_bytes(h.digest()[:4], "little") & SEED_MASK
