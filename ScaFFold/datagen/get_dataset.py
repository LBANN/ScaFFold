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


def _git_commit_short() -> str:
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
        print(
            f"Tried to get git commit id in non-git repo. No commit id will be enforced for dataset reuse."
        )
        return "no-commit-id"
    except Exception:
        print(
            f"Exception when trying to get git commit for dataset. No commit id will be enforced for dataset reuse."
        )
        return "no-commit-id"


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

    root = Path(config.dataset_dir)
    root.mkdir(exist_ok=True)

    # Get dict of required keys and compute config_id
    config_dict = vars(config).copy()
    config_dict["dataset_format_version"] = DATASET_FORMAT_VERSION
    volume_config = _get_required_keys_dict(
        config=config_dict, include_keys=INCLUDE_KEYS
    )
    config_id = _hash_volume_config(volume_config)
    commit = _git_commit_short()

    base = root / config_id
    base.mkdir(parents=True, exist_ok=True)

    # Try to reuse latest candidate dataset
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
        # If we pass the above checks, this dataset can be reused
        print(
            "Valid existing dataset found. Reusing this dataset..."
        )  # FIXME replace with updated logging
        return dataset_path

    # Otherwise, generate a new dataset
    print(f"No valid existing dataset found at {base}. Generating new dataset...")
    if rank == 0:
        ts = time.strftime("%Y%m%d-%H%M%S")
        dest = base / f"{ts}__{commit}"
        tmp = base / f".tmp_{ts}"
        tmp.mkdir(parents=True, exist_ok=False)
    else:
        tmp = dest = None

    # broadcast the staging + final paths
    tmp, dest = comm.bcast((tmp, dest) if rank == 0 else None, root=0)

    config.dataset_dir = tmp
    ok = True
    err = ""

    try:
        volumegen.main(config)
    except Exception as e:
        ok = False
        err = f"volumegen attempt failed: rank {rank}: {type(e).__name__}: {e}"

    # Check that all ranks succeeded in volumegen, then sync
    all_ok = comm.allreduce(1 if ok else 0, op=MPI.MIN) == 1
    comm.Barrier()

    # rank 0 has file write + move
    if rank == 0:
        if not all_ok:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass
            # collect & raise a representative error
            errs = comm.gather(err, root=0)
            msgs = "; ".join(e for e in errs if e)
            raise RuntimeError(f"dataset generation failed: {msgs or 'unknown error'}")

        # Write to tmp, then move, so readers never see half-written dataset
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
