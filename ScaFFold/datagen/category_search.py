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

# -*- coding: utf-8 -*-
"""
@original author: ryosuke yamada
"""

import glob
import math
import os
import shutil
import time

import numpy as np
from mpi4py import MPI

from ScaFFold.datagen import layout
from ScaFFold.datagen.generate_fractal_points import generate_fractal_points
from ScaFFold.datagen.rng import SEED_MASK, derive_seed, seed_numba
from ScaFFold.utils.config_utils import Config
from ScaFFold.utils.utils import setup_mpi_logger

DEFAULT_NP_DTYPE = np.float64

comm = None
rank = None
size = None


def compute_round_attempts(categories_remaining, size, datagen_batch_size, accept_rate):
    """Per-rank attempt count for the next round.

    A fixed ``datagen_batch_size`` per rank overshoots badly once few categories
    remain: the final round can generate thousands of valid categories per rank
    and discard all but the handful still needed. Size the round from what is
    left and the acceptance rate observed so far -- roughly
    ``remaining / (size * accept_rate)`` attempts spread over the ranks -- while
    never exceeding ``datagen_batch_size`` and always running at least one
    attempt per rank so progress (and acceptance-rate learning) continues.

    ``accept_rate`` <= 0 (no data yet, or a round that accepted nothing) falls
    back to the full batch so an unknown/hard regime is not starved.
    """
    if categories_remaining <= 0:
        return 0
    per_rank_cap = max(1, int(datagen_batch_size))
    if accept_rate <= 0.0:
        return per_rank_cap
    # Global attempts expected to yield the remaining categories, split across
    # ranks; ceil at both steps so a round is never planned short. A small
    # safety margin (10%) reduces the chance of needing an extra round.
    needed_attempts = math.ceil(1.1 * categories_remaining / accept_rate)
    per_rank = math.ceil(needed_attempts / max(size, 1))
    return max(1, min(per_rank_cap, per_rank))


def propose_next_params(base_seed: int, rank: int, attempt_index: int) -> np.array:
    """
    Propose IFS parameters for one category attempt.

    Candidate parameters are drawn from an independent ``Generator`` keyed by
    ``(base_seed, rank, attempt_index)``. Because the key includes a per-rank
    attempt counter that persists across runs, a resumed or extended run keeps
    advancing the candidate stream instead of replaying the same candidates a
    fresh run produced, while remaining reproducible for a given key.

    Parameters
    ----------
    base_seed : int
        The configured base seed.
    rank : int
        The MPI rank proposing this candidate.
    attempt_index : int
        A per-rank counter that increases with every attempt and persists
        across runs.

    Returns
    -------
    params : np.array
        A 2x13 array of IFS parameters with normalized selection probabilities
        stored in the final column of each transformation.
    """
    seed_seq = np.random.SeedSequence(
        (
            int(base_seed) & SEED_MASK,
            int(rank) & SEED_MASK,
            int(attempt_index) & SEED_MASK,
        )
    )
    generator = np.random.default_rng(seed_seq)
    params = generator.uniform(-1.0, 1.0, (2, 13)).astype(DEFAULT_NP_DTYPE)

    # Calculate normalized probabilities, then store in last params of each transformation
    rotation_matrices = params[:, 0:9].reshape(-1, 3, 3)
    probabilities_raw = np.absolute(np.linalg.det(rotation_matrices))
    probabilties_normalized = probabilities_raw / np.sum(probabilities_raw)
    params[:, -1] = probabilties_normalized

    return params


def generate_single_category(
    config: Config, base_seed: int, rank: int, attempt_index: int
) -> tuple[bool, np.array, bool, bool, bool]:
    """
    Generate a single fractal category.

    Parameters
    ----------
    config : Config
        A Config object containing run parameters.
    base_seed : int
        The configured base seed.
    rank : int
        The MPI rank running this attempt.
    attempt_index : int
        A per-rank, resume-persistent attempt counter identifying this attempt.

    Returns
    -------
    valid : bool
        A bool for whether a valid category was found on this attempt.
    params : np.array
        A numpy array containing IFS parameters for this category attempt, if attempt was valid.
    (not value_check_pass) : bool
        A bool for whether this attempt passed the NaN/non-finite check.
    (not variance_check_pass) : bool
        A bool for whether this attempt passed the variance check.
    (not runaway_check_pass) : bool
        A bool for whether this attempt passed the runaway values check.
    """

    # Bool for whether this category is valid after checks
    valid = False

    # Propose candidate params from the resume-persistent attempt stream
    params = propose_next_params(base_seed, rank, attempt_index)

    # Seed numba's internal RNG per attempt so the per-point sample used for the
    # acceptance decision is reproducible for this (base_seed, rank, attempt)
    # key -- the same candidate always yields the same accept/reject verdict.
    seed_numba(derive_seed(base_seed, rank, attempt_index))

    # Generate points in the fractal
    points, runaway_check_pass = generate_fractal_points(
        params,
        (
            config.point_num
            if isinstance(config.point_num, int)
            else int(config.point_num)
        ),
    )

    # Sum number of NaNs and reject infinities before normalization.
    nan_count = np.isnan(points).sum()
    value_check_pass = nan_count == 0 and np.isfinite(points).all()
    variance_check_pass = False

    if value_check_pass:
        # Normalize + center
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        means = points.mean(axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            ranges = maxs - mins
        value_check_pass = np.all(np.isfinite(ranges)) and np.all(ranges > 0)
        if value_check_pass:
            scales = (2 * config.normalize) / ranges
            with np.errstate(over="ignore", invalid="ignore"):
                points = (points - means) * scales

            value_check_pass = np.isfinite(points).all()
            if value_check_pass:
                # Calc dimension-wise variance and compare to threshold
                points_variance = np.var(points, axis=0)
                variance_check_pass = np.all(
                    points_variance > config.variance_threshold
                )
        if variance_check_pass and value_check_pass and runaway_check_pass:
            valid = True

    # Return result
    return (
        valid,
        params,
        bool(not value_check_pass),
        bool(value_check_pass and not variance_check_pass),
        not runaway_check_pass,
    )


def generate_categories_batch(
    config: Config,
    base_seed: int,
    rank: int,
    attempt_start: int,
    datagen_batch_size: int = 1,
) -> tuple[bool, np.array, int, int, int]:
    """
    Run a batch of fractal category generation attempts.

    Parameters
    ----------
    config : Config
        A Config object containing run parameters.
    base_seed : int
        The configured base seed.
    rank : int
        The MPI rank running this batch.
    attempt_start : int
        The per-rank attempt counter for the first attempt in this batch; each
        attempt uses ``attempt_start + i`` so the candidate stream advances
        monotonically and never replays across batches or runs.
    datagen_batch_size : int
        An int for the number of attempts to run before MPI sync between ranks.

    Returns
    -------
    one_or_more_valid : bool
        A bool for whether at least one valid category was found in this batch of attempts.
    params : np.array
        A numpy array containing IFS parameters for this category attempt, if attempt was valid.
    failed_nan_check_count : int
        The number of attempts in this batch which failed the NaN/non-finite check.
    failed_var_check_count : int
        The number of attempts in this batch which failed the var check.
    runaway_failure_count : int
        The number of attempts in this batch which failed the runaway values check.
    """
    one_or_more_valid = False
    params_list = []
    failed_nan_check_count = 0
    failed_var_check_count = 0
    runaway_failure_count = 0

    for i in range(datagen_batch_size):
        (
            attempt_valid,
            params,
            attempt_failed_nan_check,
            attempt_failed_var_check,
            runaway_failure,
        ) = generate_single_category(config, base_seed, rank, attempt_start + i)
        if attempt_valid:
            one_or_more_valid = True
            params_list.append(params)
        failed_nan_check_count += attempt_failed_nan_check
        failed_var_check_count += attempt_failed_var_check
        runaway_failure_count += runaway_failure

    return (
        one_or_more_valid,
        params_list,
        failed_nan_check_count,
        failed_var_check_count,
        runaway_failure_count,
    )


def parse_category_indices(fracts_write_dir: str) -> list[int]:
    """
    Return the sorted list of category indices already present on disk.

    Only files named ``NNNNNN.csv`` (six digits) are counted, so unrelated
    files and partial temporaries never perturb the numbering.
    """
    indices = []
    for path in glob.glob(f"{fracts_write_dir}/*.csv"):
        stem = os.path.splitext(os.path.basename(path))[0]
        if len(stem) == 6 and stem.isdigit():
            indices.append(int(stem))
    return sorted(set(indices))


def next_free_index(existing_indices) -> int:
    """
    Return the lowest non-negative index not already used.

    Holes in the numbering are filled first (so ``{0, 1, 3}`` yields ``2``);
    only once the range ``0..max`` is contiguous does this return ``max + 1``.
    Deriving the index this way -- rather than from a bare file count -- keeps a
    resumed run from colliding with an existing file when the numbering has gaps.
    """
    existing = {int(i) for i in existing_indices}
    idx = 0
    while idx in existing:
        idx += 1
    return idx


def is_duplicate_params(params: np.array, existing_params: list) -> bool:
    """Return True if ``params`` matches any already-accepted category exactly."""
    for existing in existing_params:
        if existing.shape == params.shape and np.array_equal(existing, params):
            return True
    return False


def save_valid_category(
    fracts_write_dir: str,
    params: np.array,
    existing_indices: list,
    existing_params: list,
) -> int | None:
    """
    Save one accepted candidate under the next free ``NNNNNN.csv`` index.

    Duplicates of an already-accepted category are skipped (nothing written).
    An existing file is never overwritten; a collision raises ``FileExistsError``
    rather than clobbering another category's parameters. ``existing_indices``
    and ``existing_params`` are updated in place to reflect a successful write.

    Returns
    -------
    int or None
        The allocated index, or ``None`` if the candidate duplicated an
        existing category and was skipped.
    """
    if is_duplicate_params(params, existing_params):
        return None

    idx = next_free_index(existing_indices)
    target = os.path.join(fracts_write_dir, "%06d.csv" % idx)
    if os.path.exists(target):
        raise FileExistsError(f"Refusing to overwrite existing category file: {target}")
    _savetxt_atomic(target, params)
    existing_indices.append(idx)
    existing_params.append(params)
    return idx


def _savetxt_atomic(target: str, params: np.array) -> None:
    """Write one category's parameters to ``target`` atomically.

    A category CSV truncated by a killed job is poison: the six-digit name is
    all the resume scan looks at, so the category counts as done forever, while
    every consumer (instance generation, and the search's own resume) dies
    parsing it. The file is therefore written to a temp name in the same
    directory -- one that neither the resume glob (``NNNNNN.csv``) nor the
    instance loader's ``*.csv`` filter can match -- flushed, fsynced, and only
    then ``os.replace``d onto the final name.
    """
    directory, name = os.path.split(target)
    tmp_path = os.path.join(directory, f".{name}.tmp{os.getpid()}")
    try:
        with open(tmp_path, "w") as handle:
            np.savetxt(handle, params, delimiter=",")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        # A failed write must leave nothing behind: no temp file, and no
        # partial file under the name resume would accept.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _attempt_state_path(fracts_write_dir: str, rank: int) -> str:
    return os.path.join(fracts_write_dir, f".rng_attempt_rank{rank}")


def read_attempt_counter(fracts_write_dir: str, rank: int) -> int:
    """Read this rank's persisted attempt counter (0 if none/unreadable)."""
    try:
        with open(_attempt_state_path(fracts_write_dir, rank)) as handle:
            return int(handle.read().strip())
    except (FileNotFoundError, ValueError):
        return 0


def write_attempt_counter(fracts_write_dir: str, rank: int, attempt_index: int) -> None:
    """
    Persist this rank's attempt counter atomically (temp file + ``os.replace``).

    The counter records how many candidates this rank has proposed so a resumed
    run continues the candidate stream from where it left off instead of
    replaying it. Writing after each synced batch keeps it crash-consistent: a
    crash mid-batch at worst re-proposes that batch's candidates, and the
    duplicate guard on save prevents any that were already accepted from being
    written twice.
    """
    path = _attempt_state_path(fracts_write_dir, rank)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as handle:
        handle.write(str(int(attempt_index)))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _sweep_stale_temp_files(fracts_write_dir: str, log) -> None:
    """Remove temp files stranded by killed writes in the category directory.

    Both atomic writers here (``_savetxt_atomic`` and ``write_attempt_counter``)
    unlink their temp file when the write raises, but a SIGKILL -- walltime, an
    OOM, a node failure -- skips that Python-level cleanup and strands it. The
    names carry the writer's pid, so they accumulate one per killed process and
    nothing else ever removes them; ``instance.py`` sweeps its equivalents for
    exactly this reason.

    Called on rank 0 before any rank has written anything this run, and
    best-effort: this is housekeeping, and it runs just before a Barrier the
    peers are heading into, so it must not raise.
    """
    patterns = (
        # .NNNNNN.csv.tmp<pid> -- a partially written category CSV.
        f"{fracts_write_dir}/.*.csv.tmp*",
        # .rng_attempt_rank<r>.tmp<pid> -- a partially written attempt counter.
        f"{fracts_write_dir}/.rng_attempt_rank*.tmp*",
    )
    for pattern in patterns:
        for stale in glob.glob(pattern):
            try:
                os.remove(stale)
                log.info("Removed stale category-search temp file %s", stale)
            except OSError as exc:
                log.warning("Could not remove stale temp file %s: %s", stale, exc)


def main(config: Config) -> None:
    """
    Generate fractal categories.

    Fractal category generation works as follows:
    1. Randomly generate a set of IFS parameters representing a fractal category.
    2. Use the IFS parameters to generate a cloud of points.
    3. Perform a series of checks on the point cloud. If all checks pass, accept
        this IFS as a valid fractal category and write the parameters to a file.

    Parameters
    ----------
    config : Config
        A Config object containing run parameters.
    """

    global comm, rank, size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    log = setup_mpi_logger(__file__, getattr(config, "verbose", 0))

    datagen_batch_size = int(getattr(config, "datagen_batch_size", 10000))
    if datagen_batch_size <= 0:
        raise ValueError("datagen_batch_size must be positive")

    base_seed = int(config.seed)

    log.info("MPI size = %s", size)

    # Setup directories. The library is keyed by seed (see
    # ScaFFold.datagen.layout): categories are drawn from a seed-derived
    # candidate stream, so a run under a different seed must never resume onto
    # another seed's parameter files.
    fracts_write_dir = layout.category_param_dir(config)
    if rank == 0:
        log.info("Writing fractals to %s", fracts_write_dir)
        # A library in the pre-seed layout is invisible to everything below, so
        # say why it is being ignored rather than appearing to regenerate work
        # that is plainly still on disk.
        layout.warn_if_legacy_library(config, log)
        if os.path.exists(fracts_write_dir) and config.datagen_from_scratch:
            log.info("Removing existing fractals directory")
            shutil.rmtree(fracts_write_dir)
        os.makedirs(fracts_write_dir, exist_ok=True)
        _sweep_stale_temp_files(fracts_write_dir, log)

    # Wait until dir setup completes
    comm.Barrier()

    # Each rank resumes its own candidate stream from a persisted counter, so an
    # extended/resumed run keeps proposing new candidates instead of replaying
    # the ones a fresh run produced.
    attempt_index = read_attempt_counter(fracts_write_dir, rank)

    # Parse existing category files on rank 0 alone and broadcast the result.
    # Free indices are derived from these parsed names -- filling holes, never
    # overwriting -- and, critically, the count derived below gates a loop that
    # contains collectives. Scanning the shared filesystem independently per
    # rank lets divergent views (stale metadata caches, a concurrent job, a
    # partially visible directory) put one rank inside the loop while another is
    # past it, so the two post mismatched collectives on COMM_WORLD and the job
    # hangs. One scan, one broadcast, one shared verdict.
    #
    # Rank 0 also loads the parameters of the categories already on disk (only
    # it writes, so only it needs them for the duplicate guard). That read is
    # part of the same rank-0-only window: a category CSV that will not parse --
    # ragged, or hand-edited -- would otherwise kill rank 0 *after* the peers
    # had already taken the broadcast and moved on to the next collective. Scan
    # and load are therefore one guarded decision, reported through one
    # broadcast, exactly as ``get_dataset`` reports its selection.
    existing_params = []
    interrupt = None
    if rank == 0:
        try:
            existing_indices = parse_category_indices(fracts_write_dir)
            for idx in existing_indices:
                existing_params.append(
                    np.loadtxt(
                        os.path.join(fracts_write_dir, "%06d.csv" % idx),
                        delimiter=",",
                    )
                )
            scan = ("ok", existing_indices)
        except BaseException as e:
            existing_params = []
            scan = (
                "error",
                f"rank 0 failed to scan existing categories in "
                f"{fracts_write_dir}: {type(e).__name__}: {e}",
            )
            interrupt = e if isinstance(e, KeyboardInterrupt) else None
    else:
        scan = None
    status, payload = comm.bcast(scan, root=0)
    if status == "error":
        # Rank 0 keeps an operator's interrupt; every rank aborts either way.
        if interrupt is not None:
            raise interrupt
        raise RuntimeError(f"category search failed: {payload}")
    existing_indices = payload

    # Calculate number of remaining fractal categories to generate
    existing_categories = len(existing_indices)
    categories_remaining = config.n_categories - existing_categories
    if rank == 0:
        log.info(
            "category_search found %s existing fractal categories | %s needed | "
            "%s remaining",
            existing_categories,
            config.n_categories,
            max(0, categories_remaining),
        )

    rank_start_time = time.time()

    attempts = 0
    accepted_total = 0
    nan_fail_count = 0
    var_fail_count = 0
    runaway_fail_count = 0
    while categories_remaining > 0:
        # Size this round from what is left and the acceptance rate seen so far,
        # rather than always running the full batch. Rank 0 decides and
        # broadcasts so every rank runs the same number of attempts and stays in
        # lockstep for the gather/barrier below.
        if rank == 0:
            accept_rate = accepted_total / attempts if attempts > 0 else 0.0
            round_attempts = compute_round_attempts(
                categories_remaining, size, datagen_batch_size, accept_rate
            )
        else:
            round_attempts = None
        round_attempts = comm.bcast(round_attempts, root=0)

        attempts += round_attempts * size

        # Each rank runs round_attempts attempts, advancing its persistent
        # attempt counter by exactly that many so the candidate stream stays
        # monotonic and reproducible across resumes.
        (
            valid,
            params_list,
            attempts_failed_nan_check,
            attempts_failed_var_check,
            attempts_runaway_failures,
        ) = generate_categories_batch(
            config, base_seed, rank, attempt_index, round_attempts
        )
        attempt_index += round_attempts
        nan_fail_count += attempts_failed_nan_check
        var_fail_count += attempts_failed_var_check
        runaway_fail_count += attempts_runaway_failures

        # Gather results on rank 0
        data = params_list if valid else []
        gathered_params = comm.gather(data, root=0)

        # Process IFS params one at a time, writing each to a CSV
        if rank == 0:
            params_valid = [item for sublist in gathered_params for item in sublist]
            # Track total accepted candidates (valid ones found, before dedup)
            # so the next round can size itself from the observed acceptance
            # rate = accepted / attempts.
            accepted_total += len(params_valid)
            log.info(
                "cat_remaining = %s | total attempts = %s | stats for rank 0: "
                "invalid_value_fail_count = %s, var_fail_count = %s, "
                "runaway_fail_count = %s",
                categories_remaining,
                attempts,
                nan_fail_count,
                var_fail_count,
                runaway_fail_count,
            )
            if len(params_valid) > 0:
                log.info(
                    "Processing %s valid param sets from this batch",
                    len(params_valid),
                )
            for p in params_valid:
                # Ensure we don't save more categories than needed
                if categories_remaining > 0:
                    # Save into the next free index, skipping duplicates and
                    # never overwriting an existing category file.
                    allocated = save_valid_category(
                        fracts_write_dir, p, existing_indices, existing_params
                    )

                    # Only count a newly written (non-duplicate) category.
                    if allocated is not None:
                        categories_remaining -= 1
                else:
                    log.info(
                        "Generated all fractal categories needed. Ignoring additional "
                        "valid categories."
                    )
                    break

        # Persist each rank's attempt counter so resume continues the stream.
        write_attempt_counter(fracts_write_dir, rank, attempt_index)

        # Broadcast updated categories_remaining to all ranks
        categories_remaining = comm.bcast(categories_remaining, root=0)

        # Sync all ranks before proceeding
        comm.Barrier()

    rank_end_time = time.time()
    rank_total_time = rank_end_time - rank_start_time
    global_sum_time = comm.reduce(rank_total_time, op=MPI.SUM, root=0)
    global_nan_fail_count = comm.reduce(nan_fail_count, op=MPI.SUM, root=0)
    global_var_fail_count = comm.reduce(var_fail_count, op=MPI.SUM, root=0)
    global_runaway_fail_count = comm.reduce(runaway_fail_count, op=MPI.SUM, root=0)

    if rank == 0 and attempts > 0:
        categories_generated = config.n_categories - existing_categories
        log.info(
            "Generated %s new categories in %s total attempts | %.2f attempts per "
            "category | total categories is now %s",
            categories_generated,
            attempts,
            attempts / categories_generated,
            config.n_categories,
        )
        log.info(
            "Failures experienced: %s invalid-value attempts (%.4f%%), %s variance-fail "
            "attempts (%.4f%%), %s runaway attempts (%.4f%%)",
            global_nan_fail_count,
            100 * global_nan_fail_count / attempts,
            global_var_fail_count,
            100 * global_var_fail_count / attempts,
            global_runaway_fail_count,
            100 * global_runaway_fail_count / attempts,
        )
        log.info(
            "Rank 0 wall time = %.2f | total CPU time = %.2f | avg wall time per "
            "rank = %.2f | %.2f total attempts per wall second | %.2f attempts "
            "per wall second per rank",
            rank_total_time,
            global_sum_time,
            global_sum_time / size,
            attempts / rank_total_time,
            attempts / rank_total_time / size,
        )

    return 0
