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

"""Tests for category-search round sizing."""

from ScaFFold.datagen.category_search import compute_round_attempts


def test_full_batch_when_no_acceptance_data():
    # First round (no observed acceptance yet) runs the full per-rank batch so
    # an unknown regime is not starved.
    assert compute_round_attempts(1000, 512, 10000, 0.0) == 10000


def test_round_shrinks_when_few_remain():
    # With a healthy acceptance rate and only a handful of categories left, the
    # round must be far smaller than the fixed batch (the whole point: no
    # generating thousands of categories only to discard them).
    attempts = compute_round_attempts(1, 512, 10000, 0.16)
    assert attempts < 10000
    assert attempts >= 1


def test_round_never_exceeds_batch_cap():
    # Even a huge remaining count with a tiny acceptance rate is capped at the
    # configured per-rank batch size.
    assert compute_round_attempts(10**9, 1, 10000, 0.001) == 10000


def test_zero_remaining_runs_nothing():
    assert compute_round_attempts(0, 8, 10000, 0.1) == 0


def test_round_scales_with_remaining_and_rate():
    # remaining / (size * accept_rate), ceil-ed with a small margin, split over
    # ranks. 1000 remaining over 4 ranks at 5% acceptance needs ~5000 per rank.
    attempts = compute_round_attempts(1000, 4, 10000, 0.05)
    assert 5000 <= attempts <= 6000


def test_at_least_one_attempt_per_rank_when_work_remains():
    # A positive remaining count always yields at least one attempt per rank so
    # the loop makes progress and can keep learning the acceptance rate.
    assert compute_round_attempts(1, 1024, 10000, 0.99) >= 1
