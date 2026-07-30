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

import numpy as np
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="./replacethis.npy",
        help="Choose which model to visualize",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    with open(args.input, "rb") as f:
        data = np.load(f)
    f.close()
    ax = plt.figure().add_subplot(projection="3d")

    # Handle both 3D and 4D arrays
    if data.ndim == 4:
        # Channels-first (C, D, H, W) -> transpose to (D, H, W, C)
        if data.shape[0] == 3:
            data = data.transpose((1, 2, 3, 0))
            # Compute occupancy: a voxel is occupied if any channel is nonzero
            occupancy = data.any(axis=-1)
            ax.voxels(occupancy, facecolors=data)
        else:
            raise ValueError(f"4D array must have shape (3, D, H, W), got {data.shape}")
    elif data.ndim == 3:
        # 3D mask or similar
        ax.voxels(data, edgecolor="k")
    else:
        raise ValueError(
            f"Expected 3D or 4D array, got {data.ndim}D with shape {data.shape}"
        )

    plt.show()
