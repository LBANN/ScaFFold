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

import argparse
import logging
import pickle
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path

import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="../data/masks/training",
        help="Where to check masks",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="../data/preprocess_data",
        help="Where to put the output",
    )
    args = parser.parse_args()
    return args


def unique_mask_values(mask_file):
    # The caller resolves the id -> path mapping once from a single directory
    # listing, so the worker just reads the file it is handed rather than
    # re-globbing the whole directory per id (which is O(N) metadata traffic
    # per id, O(N^2) overall on a shared filesystem).
    with open(mask_file, "rb") as f:
        mask = np.load(f)

    return np.unique(mask)


def _index_masks_by_stem(directory):
    """Map each file stem to its single mask path from one directory listing.

    Requires exactly one file per stem: a stem with two files (a stale sibling
    sharing the stem, e.g. ``000000.npy`` and ``000000.npz``) is ambiguous and
    would let the scan pick an arbitrary one, missing labels present in the file
    the training loader actually reads. Raise a clear error naming the stem
    instead.
    """
    stem_to_path = {}
    for name in listdir(directory):
        if name.startswith(".") or not isfile(join(directory, name)):
            continue
        stem = splitext(name)[0]
        if stem in stem_to_path:
            raise ValueError(
                f"Multiple files share the id {stem!r} in {directory}: "
                f"{stem_to_path[stem].name} and {name}"
            )
        stem_to_path[stem] = Path(directory) / name
    return stem_to_path


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    images_dir = Path(args.input)
    stem_to_path = _index_masks_by_stem(images_dir)
    if not stem_to_path:
        raise RuntimeError(
            f"No input file found in {images_dir}, make sure you put your images there"
        )

    mask_paths = [stem_to_path[stem] for stem in sorted(stem_to_path)]
    logging.info(f"Scanning {len(mask_paths)} masks")
    with Pool() as p:
        unique = list(
            tqdm(
                p.imap(unique_mask_values, mask_paths),
                total=len(mask_paths),
            )
        )

    mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
    logging.info(f"Unique mask values: {mask_values}")

    # Saves the values in a pickle
    data = {"mask_values": mask_values}
    outfile = open(args.output, "wb")
    pickle.dump(data, outfile)
    outfile.close()
