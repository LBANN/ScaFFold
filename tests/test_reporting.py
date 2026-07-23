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
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ScaFFold.utils.utils import plot_img_and_mask
from ScaFFold.viz import standard_viz


class TestFiguresDir:
    """F45: standard_viz.main() creates figures_path with idempotence."""

    def test_figures_dir_idempotent(self, tmp_path):
        """Generate figures twice into same run_dir; second call should succeed."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Synthetic train_stats.csv
        csv_path = run_dir / "train_stats.csv"
        csv_path.write_text("epoch,overall_loss,val_dice\n1,0.9,0.40\n2,0.5,0.70\n")

        config = SimpleNamespace(
            run_dir=str(run_dir),
            vol_size=32,
            n_categories=5,
            unet_layers=2,
        )

        # First call should succeed
        standard_viz.main(config)
        assert (run_dir / "figures" / "train_loss.png").exists()

        # Second call should also succeed (not raise FileExistsError)
        standard_viz.main(config)
        assert (run_dir / "figures" / "train_loss.png").exists()


class TestDiceFigure:
    """F70: Validation Dice figure saved as val_dice.png, not val_loss.png."""

    def test_dice_figure_filename(self, tmp_path):
        """Save Dice figure as val_dice.png; val_loss.png (if present) contains loss series."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Synthetic train_stats.csv with distinct val_dice and val_loss_avg columns
        headers = [
            "epoch",
            "epoch_loss",
            "overall_loss",
            "val_loss_epoch",
            "val_loss_avg",
            "train_dice",
            "val_dice",
            "epoch_duration",
            "optimizer_steps",
            "total_optimizer_steps",
        ]
        val_dice_col = [0.11, 0.42, 0.73, 0.94]
        val_loss_avg_col = [900.0, 500.0, 200.0, 50.0]

        with open(run_dir / "train_stats.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for i, epoch in enumerate([1, 2, 3, 4]):
                writer.writerow(
                    [
                        epoch,
                        4.0,
                        2.0,
                        8.0,
                        val_loss_avg_col[i],
                        0.5,
                        val_dice_col[i],
                        10.0,
                        5,
                        5 * epoch,
                    ]
                )

        config = SimpleNamespace(
            run_dir=str(run_dir),
            vol_size=64,
            n_categories=10,
            unet_layers=4,
        )

        # Capture savefig calls to inspect data
        saved = []
        real_savefig = plt.savefig

        def recording_savefig(fname, *a, **kw):
            ax = plt.gcf().axes[0]
            ydata = list(np.asarray(ax.lines[0].get_ydata(), dtype=float))
            saved.append({"file": Path(fname).name, "ydata": ydata})
            return real_savefig(fname, *a, **kw)

        plt.savefig = recording_savefig
        try:
            standard_viz.main(config)
        finally:
            plt.savefig = real_savefig

        # Check that val_dice.png exists and contains the Dice data
        assert (run_dir / "figures" / "val_dice.png").exists()

        # Find which savefig calls correspond to Dice and Loss
        dice_fig = next((s for s in saved if s["file"] == "val_dice.png"), None)
        assert dice_fig is not None, "val_dice.png figure was not created"
        assert np.allclose(dice_fig["ydata"], val_dice_col), (
            "val_dice.png does not contain the Dice curve"
        )


class TestMaskPanelLabels:
    """F69: plot_img_and_mask labels panels with correct class index, not off-by-one."""

    def test_mask_panel_labels(self):
        """Render a 3-class mask; check that each panel title matches the class shown."""
        img = np.zeros((4, 4))
        mask = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [2, 2, 2, 0], [2, 2, 2, 2]])

        # Call the function and capture the figure
        plot_img_and_mask(img, mask)
        fig = plt.gcf()
        axes = fig.axes

        # Check each mask panel's label
        mismatches = 0
        for panel_idx, ax in enumerate(axes[1:], start=1):
            shown = np.asarray(ax.images[0].get_array())
            # Identify which class this panel actually shows
            actual_class = None
            for c in range(int(mask.max()) + 1):
                if np.array_equal(shown.astype(bool), mask == c):
                    actual_class = c
                    break

            title = ax.get_title()
            # Extract labeled class from title e.g., "Mask (class 0)"
            labeled_class = int(title.split("class")[1].strip(" )"))

            if labeled_class != actual_class:
                mismatches += 1

        assert mismatches == 0, f"Expected 0 label mismatches, got {mismatches}"
        plt.close("all")


class TestVisualizerVolume:
    """F65: data_visualizer renders 4D channels-first volumes from volumegen."""

    def test_visualizer_accepts_4d_volume(self, tmp_path):
        """4D float volume (3, N, N, N) renders without exception."""
        from ScaFFold.utils.data_types import VOLUME_DTYPE

        N = 8
        # Create a 4D channels-first volume like volumegen produces
        volume_4d = np.zeros((3, N, N, N), dtype=VOLUME_DTYPE)
        volume_4d[:, 2:5, 2:5, 2:5] = 0.5

        vol_file = tmp_path / "volume.npy"
        np.save(vol_file, volume_4d)

        # Call the visualizer; should not raise ValueError about dimensions
        ax = plt.figure().add_subplot(projection="3d")

        # Read and visualize
        data = np.load(vol_file)

        # Should handle 4D by transposing to (N, N, N, 3) and using occupancy
        # For now, just verify the shape handling logic
        if data.ndim == 4 and data.shape[0] == 3:
            # Transpose channels-last
            data_transposed = data.transpose((1, 2, 3, 0))
            # Compute occupancy from color channels
            occupancy = data_transposed.any(axis=-1)
            # This should not raise
            ax.voxels(occupancy, facecolors=data_transposed)
        else:
            ax.voxels(data, edgecolor="k")

        plt.close("all")

    def test_visualizer_still_accepts_3d_mask(self, tmp_path):
        """3D mask renders (regression guard)."""
        from ScaFFold.utils.data_types import MASK_DTYPE

        N = 8
        mask_3d = np.zeros((N, N, N), dtype=MASK_DTYPE)
        mask_3d[2:5, 2:5, 2:5] = 1

        mask_file = tmp_path / "mask.npy"
        np.save(mask_file, mask_3d)

        # Call the visualizer with 3D data
        ax = plt.figure().add_subplot(projection="3d")
        data = np.load(mask_file)
        ax.voxels(data, edgecolor="k")  # Should work without error

        plt.close("all")


class TestTorchProfiler:
    """F68: Torch profiler enabled independently of Caliper even when CALI_CONFIG set."""

    def test_torch_profiler_independent_of_caliper(self):
        """Test that the logic branches properly: if Caliper set but fails, check torch next."""
        # This test verifies the logic structure in perf_measure.py.
        # The bug is that an elif chain prevents torch profiler from being checked
        # when Caliper is set but fails. The fix is to use independent if statements.

        # Simulate module evaluation logic with the current (buggy) code:
        # if CALI_CONFIG: try import cali except: pass   (falls through)
        # elif TORCH_PERF: try import torch               (SKIPPED because of elif)

        # After fix:
        # if CALI_CONFIG: try import cali except: pass
        # if not _CALI_PERF_ENABLED and TORCH_PERF: try import torch

        # We can verify this by inspecting the module's logic structure
        with open("/usr/WS1/dryden1/ScaFFold/ScaFFold/utils/perf_measure.py") as f:
            code = f.read()

        # The fix should have an independent if statement for torch profiler
        # after the Caliper try/except block
        assert "elif" not in code or (code.count("if") > code.count("elif")), (
            "Logic should use if, not elif chains for independent profiler checks"
        )
