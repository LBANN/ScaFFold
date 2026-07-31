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

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import yaml
from mpi4py import MPI

from ScaFFold.datagen import volumegen
from ScaFFold.utils.utils import setup_mpi_logger

META_FILENAME = "meta.yaml"
# Datasets are generated into a staging directory carrying this prefix and only
# renamed into their final ``<timestamp>__<commit>`` name once complete, so a
# reader never observes a half-written dataset. The prefix is also what the
# reuse scan skips and what the orphan cleanup collects.
TMP_PREFIX = ".tmp_"
# Bumped from 2 to 3 when instance point clouds moved from float64 to float32:
# the storage layout is unchanged, but float32 voxel binning shifts a handful of
# boundary voxels, so a float64-era dataset must not be reused as if it were
# float32. This version stamps new datasets, gates reuse below, and feeds the
# config_id hash, so an older dataset is neither matched nor scanned. The loader
# in data_loading.py keeps its own (lower) minimum-layout version and still reads
# v3 through the modern dense path.
DATASET_FORMAT_VERSION = 3
INCLUDE_KEYS = [
    "dataset_format_version",
    "n_categories",
    "n_instances_used_per_fractal",
    "problem_scale",
    "seed",
    "variance_threshold",
    "n_fracts_per_vol",
    "val_split",
]


def canonicalize(input):
    """
    Sort dict keys, recursing on lists/dicts, for stable hashing.
    """
    if isinstance(input, dict):
        return {key: canonicalize(input[key]) for key in sorted(input)}
    elif isinstance(input, list):
        return [canonicalize(item) for item in input]
    else:
        return input


def _get_required_keys_dict(
    config: Dict[str, Any], include_keys: list[str]
) -> Dict[str, Any]:
    """
    Build a dict containing only the required keys.
    Raises KeyError if any required key is missing.
    """
    missing = [key for key in include_keys if key not in config]
    if missing:
        raise KeyError(
            f"Missing expected top-level keys in run YAML: {missing}. "
            f"Required INCLUDE_KEYS={include_keys}"
        )
    required = {key: config[key] for key in include_keys}
    return canonicalize(required)


def _hash_volume_config(volume_config: Dict[str, Any]) -> str:
    s = json.dumps(volume_config, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(s).hexdigest()[:12]


def _git_commit_short(log) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,  # Don't show console output to user
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        log.warning(
            "Tried to get git commit id in non-git repo. "
            "No commit id will be enforced for dataset reuse."
        )
        return "no-commit-id"
    except Exception:
        log.warning(
            "Exception when trying to get git commit for dataset. "
            "No commit id will be enforced for dataset reuse."
        )
        return "no-commit-id"


def _write_meta_atomic(meta_path: Path, meta: Dict[str, Any]) -> None:
    """Write ``meta`` to ``meta_path`` atomically.

    ``meta.yaml`` is what the loader reads to decide how every sample in the
    dataset is interpreted, so a partially written one is worse than none at
    all: a truncated file parses as empty and silently reclassifies a modern
    dataset as legacy v1. The document is therefore written to a temp file in
    the same directory, flushed and fsynced, and only then ``os.replace``d onto
    the final name -- an atomic rename within one filesystem.
    """
    tmp_path = meta_path.parent / f".{meta_path.name}.tmp{os.getpid()}"
    try:
        with open(tmp_path, "w") as handle:
            handle.write(yaml.safe_dump(meta, sort_keys=True, default_flow_style=False))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, meta_path)
    except BaseException:
        # A failed write must not leave a temp file behind, and the final name
        # must keep whatever complete document was already there.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _decide_reuse_or_generate(
    base: Path,
    config_id: str,
    commit: str,
    require_commit: bool,
    log,
) -> tuple:
    """Scan ``base`` for a reusable dataset, otherwise stage a fresh one.

    Runs on rank 0 only. The returned tuple is broadcast to every rank so all
    ranks take the same branch: ``("reuse", dataset_dir)`` when an existing
    directory matches, or ``("generate", tmp_dir, dest_dir)`` carrying the
    staging and final paths for a new generation. Making this decision in one
    place and broadcasting it prevents ranks from diverging when their views of
    the shared filesystem differ.

    The scan is deliberately forgiving: a candidate whose metadata is missing,
    unreadable, or malformed is warned about and skipped rather than allowed to
    raise. This function runs inside a window where every peer is already
    waiting in the decision broadcast, so a crash here is a job-wide hang; a
    poison directory (exactly what a killed job leaves behind) must never be
    able to cause one.
    """
    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True
    )
    for dataset_path in candidates:
        # Staging dirs are not datasets: a job killed between the meta write and
        # the rename leaves a complete meta.yaml inside one, and reusing it hands
        # back a partially generated (and cleanup-eligible) directory.
        if dataset_path.name.startswith(TMP_PREFIX):
            continue
        meta_path = dataset_path / META_FILENAME
        if not meta_path.exists():
            continue
        try:
            meta = yaml.safe_load(meta_path.read_text())
        except Exception as exc:
            log.warning(
                "Skipping dataset candidate %s: unreadable %s (%s: %s)",
                dataset_path,
                META_FILENAME,
                type(exc).__name__,
                exc,
            )
            continue
        if not isinstance(meta, dict):
            log.warning(
                "Skipping dataset candidate %s: %s is empty or malformed",
                dataset_path,
                META_FILENAME,
            )
            continue
        if meta.get("config_id") != config_id:
            continue
        if meta.get("dataset_format_version", 1) != DATASET_FORMAT_VERSION:
            continue
        if require_commit and meta.get("code_commit") != commit:
            continue
        # All checks passed: this dataset can be reused.
        log.info("Reusing existing dataset at %s", dataset_path)
        return ("reuse", str(dataset_path))

    log.info("No valid existing dataset found at %s. Generating new dataset.", base)
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = base / f"{ts}__{commit}"
    # The staging name must be unique per job: a bare 1-second-granularity
    # timestamp let two same-config jobs starting in the same second collide on
    # ``mkdir(exist_ok=False)``, killing one of them mid-consensus. Adding the
    # pid and a random suffix makes the name unique even across nodes.
    tmp = base / f"{TMP_PREFIX}{ts}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=False)
    return ("generate", str(tmp), str(dest))


def get_dataset(
    config: Namespace,
    require_commit: bool = False,  # default: ignore commit mismatches for reuse
) -> Path:
    """
    Get dataset matching requested config, either by:
        1. Finding an existing dataset with matching config
            (optionally enforcing matching code commits), or
        2. Generating a new dataset from the input config.
    Allows for reusing existing datasets where appropriate.

    Returns: Path to the selected (or newly created) dataset directory.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    log = setup_mpi_logger(__file__, getattr(config, "verbose", 0))

    root = Path(config.dataset_dir)
    root.mkdir(exist_ok=True)

    # Get dict of required keys and compute config_id
    config_dict = vars(config).copy()
    config_dict["dataset_format_version"] = DATASET_FORMAT_VERSION
    volume_config = _get_required_keys_dict(
        config=config_dict, include_keys=INCLUDE_KEYS
    )
    config_id = _hash_volume_config(volume_config)
    commit = _git_commit_short(log)

    base = root / config_id
    base.mkdir(parents=True, exist_ok=True)

    # Decide once on rank 0 whether an existing dataset can be reused or a new
    # one must be generated, then broadcast the decision so every rank takes the
    # same branch. Scanning the shared filesystem independently per rank lets
    # divergent views (stale metadata caches, a racing job's rename) strand some
    # ranks in the generation collectives while others return early.
    # Everything rank 0 does here happens while the peers are already blocked in
    # the broadcast below, so a rank-0 exception would strand the whole job.
    # Any failure is therefore turned into an error sentinel that travels
    # through the same broadcast and makes every rank raise the same error.
    if rank == 0:
        try:
            decision = _decide_reuse_or_generate(
                base, config_id, commit, require_commit, log
            )
        except (Exception, SystemExit) as e:
            decision = (
                "error",
                f"rank 0 failed to select a dataset under {base}: "
                f"{type(e).__name__}: {e}",
            )
    else:
        decision = None
    decision = comm.bcast(decision, root=0)

    if decision[0] == "error":
        raise RuntimeError(f"dataset selection failed: {decision[1]}")

    if decision[0] == "reuse":
        return Path(decision[1])

    _, tmp_str, dest_str = decision
    tmp = Path(tmp_str)
    dest = Path(dest_str)

    config.dataset_dir = tmp
    ok = True
    err = ""

    # A worker failure must not skip any collective below: catch everything
    # (including SystemExit, which is a BaseException and would otherwise bypass
    # the consensus) so every rank always reaches the allreduce and gather.
    try:
        volumegen.main(config)
    except (Exception, SystemExit) as e:
        ok = False
        err = f"volumegen attempt failed: rank {rank}: {type(e).__name__}: {e}"

    # Reach a global verdict, then have every rank participate in the error
    # gather so no rank is left in a mismatched collective on the failure path.
    all_ok = comm.allreduce(1 if ok else 0, op=MPI.MIN) == 1
    errs = comm.allgather(err)

    if not all_ok:
        if rank == 0:
            shutil.rmtree(tmp, ignore_errors=True)
        # Every rank raises with the collected messages, so a non-root rank
        # never returns an unfinalized dataset path.
        msgs = "; ".join(e for e in errs if e)
        raise RuntimeError(f"dataset generation failed: {msgs or 'unknown error'}")

    # rank 0 writes metadata into the staging dir, then renames it into place so
    # readers never observe a half-written dataset. This is another rank-0-only
    # window inside a collective sequence: the rename can fail (a racing job
    # already published this name, quota, ...), so the outcome is broadcast
    # rather than allowed to kill rank 0 while the peers wait for it.
    finalize_err = ""
    if rank == 0:
        try:
            meta = {
                "config_id": config_id,
                "dataset_format_version": DATASET_FORMAT_VERSION,
                "config_subset": volume_config,
                "include_keys": INCLUDE_KEYS,
                "code_commit": commit,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _write_meta_atomic(tmp / META_FILENAME, meta)
            tmp.rename(dest)
        except (Exception, SystemExit) as e:
            finalize_err = (
                f"rank 0 failed to finalize dataset at {dest}: {type(e).__name__}: {e}"
            )

    # This broadcast doubles as the synchronization the old Barrier provided: no
    # rank returns before rank 0 has published the rename (or reported that it
    # could not), so nobody observes the staging path or a missing dataset.
    finalize_err = comm.bcast(finalize_err, root=0)
    if finalize_err:
        raise RuntimeError(f"dataset generation failed: {finalize_err}")

    return dest
