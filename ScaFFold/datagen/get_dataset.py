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
import shutil
import subprocess
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import yaml
from mpi4py import MPI

from ScaFFold.datagen import volumegen
from ScaFFold.utils.utils import setup_mpi_logger

META_FILENAME = "meta.yaml"
DATASET_FORMAT_VERSION = 2
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
    """
    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True
    )
    for dataset_path in candidates:
        meta_path = dataset_path / META_FILENAME
        if not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text())
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
    tmp = base / f".tmp_{ts}"
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
    if rank == 0:
        decision = _decide_reuse_or_generate(
            base, config_id, commit, require_commit, log
        )
    else:
        decision = None
    decision = comm.bcast(decision, root=0)

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
    # readers never observe a half-written dataset.
    if rank == 0:
        meta = {
            "config_id": config_id,
            "dataset_format_version": DATASET_FORMAT_VERSION,
            "config_subset": volume_config,
            "include_keys": INCLUDE_KEYS,
            "code_commit": commit,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (tmp / META_FILENAME).write_text(
            yaml.safe_dump(meta, sort_keys=True, default_flow_style=False)
        )
        tmp.rename(dest)

    # ensure the rename is visible everywhere before returning
    comm.Barrier()
    return dest
