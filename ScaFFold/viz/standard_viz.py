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

import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from ScaFFold.utils.config_utils import RunConfig

logger = logging.getLogger(__name__)


def main(config: RunConfig):
    figures_path = Path(config.run_dir) / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    try:
        epochs = []
        train_loss = []
        val_dice = []
        val_loss = []

        csv_path = Path(config.run_dir) / "train_stats.csv"
        # Read training statistics from CSV
        with open(csv_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["overall_loss"]))
                val_dice.append(float(row["val_dice"]))
                # Try to read val_loss_avg if present
                if "val_loss_avg" in row:
                    val_loss.append(float(row["val_loss_avg"]))

        plot_title = f"v={config.vol_size}, c={config.n_categories}, u={config.unet_layers}"
        line_thickness = 2
        fontsize = 20
        tick_fontsize = 14
        legend_fontsize = 16
        legend_loc = (0, -0.17)

        # Plot training loss
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss", linewidth=line_thickness)
        plt.xlabel("Epoch", fontsize=fontsize)
        plt.ylabel("Train loss", fontsize=fontsize)
        plt.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        plt.yscale("log")
        plt.title(plot_title, fontsize=12)
        plt.legend(loc="upper left", bbox_to_anchor=legend_loc, fontsize=legend_fontsize)
        plt.grid(True, axis="y")
        plt.savefig(figures_path / "train_loss.png", dpi=300, bbox_inches="tight")

        # Plot validation dice
        plt.figure()
        plt.plot(epochs, val_dice, label="Val Dice Score", linewidth=line_thickness)
        plt.xlabel("Epoch", fontsize=fontsize)
        plt.ylabel("Val dice score", fontsize=fontsize)
        plt.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        plt.title(plot_title, fontsize=12)
        plt.legend(loc="upper left", bbox_to_anchor=legend_loc, fontsize=legend_fontsize)
        plt.grid(True, axis="y")
        plt.savefig(figures_path / "val_dice.png", dpi=300, bbox_inches="tight")

        # Plot validation loss if available
        if val_loss:
            plt.figure()
            plt.plot(epochs, val_loss, label="Val Loss", linewidth=line_thickness)
            plt.xlabel("Epoch", fontsize=fontsize)
            plt.ylabel("Val loss", fontsize=fontsize)
            plt.tick_params(axis="both", which="major", labelsize=tick_fontsize)
            plt.title(plot_title, fontsize=12)
            plt.legend(loc="upper left", bbox_to_anchor=legend_loc, fontsize=legend_fontsize)
            plt.grid(True, axis="y")
            plt.savefig(figures_path / "val_loss.png", dpi=300, bbox_inches="tight")
    except Exception as e:
        logger.error(f"Failed to generate figures: {e}")
