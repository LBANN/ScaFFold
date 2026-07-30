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

"""Determinism and resume tests for the datagen candidate/instance pipeline.

``generate_fractal_points`` is ``@numba.njit`` and draws from numba's
process-global RNG, which cannot be reset from a live interpreter. Every
determinism check that compares two seeded draws therefore runs in its own
fresh subprocess (via the ``fresh_python`` fixture); in-process reruns would
share the already-advanced numba stream and could not prove reproducibility.

The category-search loop is exercised through its factored helpers
(``propose_next_params`` / ``generate_categories_batch`` / ``save_valid_category``)
so the accept/reject stream can be driven with a tiny validation predicate
without standing up MPI.
"""

import hashlib
import os

import numpy as np

from ScaFFold.datagen import category_search as cs

# ---------------------------------------------------------------------------
# F06: seed_numba controls the njit RNG stream
# ---------------------------------------------------------------------------


def test_seed_numba_controls_njit_stream(fresh_python):
    """seed_numba makes an njit random draw reproducible across processes."""

    def draw_with_seed(seed):
        snippet = (
            "import numba, numpy as np\n"
            "from ScaFFold.datagen.rng import seed_numba\n"
            "@numba.njit\n"
            "def draw():\n"
            "    return np.random.rand()\n"
            f"seed_numba({seed})\n"
            "print('%.17g' % draw())\n"
        )
        return fresh_python(snippet).strip()

    # Two fresh processes, same seed -> identical draw.
    out_a = draw_with_seed(123)
    out_b = draw_with_seed(123)
    assert out_a == out_b

    # Different seed -> different draw.
    out_c = draw_with_seed(456)
    assert out_c != out_a


# ---------------------------------------------------------------------------
# F06: whole-instance generation is deterministic across fresh processes
# ---------------------------------------------------------------------------


# Snippet: generate a single (category, instance) at tiny scale from a synthetic
# category CSV, seeding numba per item exactly as instance.py does, and print
# the sha256 of the saved point cloud. Parameterized by config seed via argv.
_INSTANCE_SNIPPET = """
import hashlib, sys
import numpy as np
from ScaFFold.datagen.generate_fractal_points import generate_fractal_points
from ScaFFold.datagen.rng import derive_seed, seed_numba

seed = int(sys.argv[1])
category, instance = 0, 3

# A synthetic contractive 2-map IFS category (2x13 layout).
params = np.zeros((2, 13), dtype=np.float64)
params[:, 0] = params[:, 4] = params[:, 8] = 0.5
params[1, 9] = params[1, 10] = params[1, 11] = 0.5
params[0, 12] = 0.5

# Mirror instance.py: derive a per-item seed, seed the njit RNG, generate.
seed_numba(derive_seed(seed, category, instance))
points, _ = generate_fractal_points(params, 400)
points = np.ascontiguousarray(points, dtype=np.float64)
print(hashlib.sha256(points.tobytes()).hexdigest())
"""


def test_instance_generation_deterministic(fresh_python):
    """Same config seed -> byte-identical instance across fresh processes."""
    h1 = fresh_python(_INSTANCE_SNIPPET.replace("int(sys.argv[1])", "42")).strip()
    h2 = fresh_python(_INSTANCE_SNIPPET.replace("int(sys.argv[1])", "42")).strip()
    assert h1 == h2

    # A different config seed changes the instance content.
    h3 = fresh_python(_INSTANCE_SNIPPET.replace("int(sys.argv[1])", "99")).strip()
    assert h3 != h1


# ---------------------------------------------------------------------------
# F06: per-item seed derivation is independent of rank/world-size layout
# ---------------------------------------------------------------------------


def test_instance_seed_independent_of_rank_layout():
    """The per-item seed depends only on (seed, category, instance)."""
    from ScaFFold.datagen.rng import derive_seed

    # Same work item, regardless of which rank/world size produced it.
    assert derive_seed(42, 5, 7) == derive_seed(42, 5, 7)

    # Distinct work items get distinct seeds.
    assert derive_seed(42, 5, 7) != derive_seed(42, 5, 8)
    assert derive_seed(42, 5, 7) != derive_seed(42, 6, 7)
    assert derive_seed(42, 5, 7) != derive_seed(43, 5, 7)

    # Seeds are valid 32-bit values.
    for keys in [(0, 0, 0), (42, 5, 7), (2**31, 144, 1)]:
        s = derive_seed(*keys)
        assert 0 <= s <= 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Shared helpers for the loop-driven tests below
# ---------------------------------------------------------------------------


def _run_search_loop(write_dir, base_seed, rank, attempt_start, n_wanted, accept):
    """Drive the factored propose/accept/save loop without MPI.

    ``accept`` is a predicate ``(params) -> bool`` standing in for the real
    variance/finiteness checks so tests control which candidates are kept. The
    saved parameter rows and the final attempt counter are returned so a caller
    can persist state and resume.
    """
    existing_indices = cs.parse_category_indices(write_dir)
    existing_params = [
        np.loadtxt(os.path.join(write_dir, "%06d.csv" % i), delimiter=",")
        for i in existing_indices
    ]
    saved = []
    attempt = attempt_start
    remaining = n_wanted
    # Bounded to keep a broken predicate from looping forever.
    while remaining > 0 and attempt < attempt_start + 100000:
        params = cs.propose_next_params(base_seed, rank, attempt)
        attempt += 1
        if not accept(params):
            continue
        allocated = cs.save_valid_category(
            write_dir, params, existing_indices, existing_params
        )
        if allocated is not None:
            saved.append(params)
            remaining -= 1
    return saved, attempt


# ---------------------------------------------------------------------------
# F07: resume continues the candidate stream instead of replaying it
# ---------------------------------------------------------------------------


def test_resume_continues_param_stream(tmp_path):
    """A resumed run's saved rows never duplicate the first run's rows."""
    write_dir = tmp_path / "3DIFS_param"
    write_dir.mkdir()
    write_dir = str(write_dir)

    def accept_all(_params):
        return True

    K = 4
    first, attempt_after = _run_search_loop(
        write_dir,
        base_seed=1234,
        rank=0,
        attempt_start=0,
        n_wanted=K,
        accept=accept_all,
    )
    # Persist the attempt counter the way main() does, then resume from it.
    cs.write_attempt_counter(write_dir, 0, attempt_after)
    resumed_start = cs.read_attempt_counter(write_dir, 0)
    assert resumed_start == attempt_after

    second, _ = _run_search_loop(
        write_dir,
        base_seed=1234,
        rank=0,
        attempt_start=resumed_start,
        n_wanted=K,
        accept=accept_all,
    )

    # No resumed parameter row equals any first-run row.
    for row_b in second:
        for row_a in first:
            assert not np.array_equal(row_a, row_b)

    # And 2*K distinct category files exist on disk.
    assert len(cs.parse_category_indices(write_dir)) == 2 * K


# ---------------------------------------------------------------------------
# F07: duplicate candidates are rejected by the dedup guard
# ---------------------------------------------------------------------------


def test_duplicate_candidate_skipped(tmp_path):
    """A candidate identical to an existing CSV is not written as new."""
    write_dir = tmp_path / "3DIFS_param"
    write_dir.mkdir()
    write_dir = str(write_dir)

    params = cs.propose_next_params(1234, 0, 0)
    existing_indices = []
    existing_params = []

    first_idx = cs.save_valid_category(
        write_dir, params, existing_indices, existing_params
    )
    assert first_idx == 0
    assert len(cs.parse_category_indices(write_dir)) == 1

    # Saving the identical params again is skipped (returns None, no new file).
    dup_idx = cs.save_valid_category(
        write_dir, params.copy(), existing_indices, existing_params
    )
    assert dup_idx is None
    assert len(cs.parse_category_indices(write_dir)) == 1


# ---------------------------------------------------------------------------
# F63: index allocation fills gaps and never overwrites an existing file
# ---------------------------------------------------------------------------


def test_index_allocation_with_gap(tmp_path):
    """Existing {0,1,3} -> next index is 2; file 3 is untouched."""
    write_dir = tmp_path / "3DIFS_param"
    write_dir.mkdir()
    write_dir = str(write_dir)

    # Pre-create categories 0, 1, 3 with distinct sentinel params.
    for i in (0, 1, 3):
        np.savetxt(
            os.path.join(write_dir, "%06d.csv" % i),
            np.full((2, 13), 1000.0 + i),
            delimiter=",",
        )
    file3 = os.path.join(write_dir, "000003.csv")
    with open(file3, "rb") as handle:
        file3_hash_before = hashlib.sha256(handle.read()).hexdigest()

    assert cs.next_free_index(cs.parse_category_indices(write_dir)) == 2

    # Save a genuinely new candidate; it must land in the hole at index 2.
    existing_indices = cs.parse_category_indices(write_dir)
    existing_params = [
        np.loadtxt(os.path.join(write_dir, "%06d.csv" % i), delimiter=",")
        for i in existing_indices
    ]
    new_params = cs.propose_next_params(7, 0, 0)
    allocated = cs.save_valid_category(
        write_dir, new_params, existing_indices, existing_params
    )
    assert allocated == 2
    assert os.path.exists(os.path.join(write_dir, "000002.csv"))

    # The pre-existing file at index 3 is byte-identical after the run.
    with open(file3, "rb") as handle:
        file3_hash_after = hashlib.sha256(handle.read()).hexdigest()
    assert file3_hash_after == file3_hash_before
