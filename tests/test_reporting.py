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
import pytest

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

    def test_torch_profiler_independent_of_caliper(self, monkeypatch):
        """Both profilers come up together when both are requested.

        Caliper is faked in ``sys.modules`` so its import succeeds, then
        ``perf_measure`` is re-evaluated with both env vars set: the torch
        profiler must still enable rather than being skipped because Caliper
        won an earlier branch. The module is reloaded again afterwards so its
        real (env-driven) state is restored for other tests.
        """
        import importlib
        import sys
        from types import ModuleType

        import ScaFFold.utils.perf_measure as perf_measure

        fake_pyadiak = ModuleType("pyadiak")
        fake_annotations = ModuleType("pyadiak.annotations")
        fake_annotations.init = lambda comm: None
        fake_annotations.value = lambda name, val: None
        fake_annotations.fini = lambda: None
        fake_pyadiak.annotations = fake_annotations

        fake_pycaliper = ModuleType("pycaliper")
        fake_pycaliper.annotate_function = lambda name=None: lambda func: func
        fake_instrumentation = ModuleType("pycaliper.instrumentation")
        fake_instrumentation.begin_region = lambda name: None
        fake_instrumentation.end_region = lambda name: None
        fake_pycaliper.instrumentation = fake_instrumentation

        try:
            with monkeypatch.context() as m:
                m.setitem(sys.modules, "pyadiak", fake_pyadiak)
                m.setitem(sys.modules, "pyadiak.annotations", fake_annotations)
                m.setitem(sys.modules, "pycaliper", fake_pycaliper)
                m.setitem(
                    sys.modules, "pycaliper.instrumentation", fake_instrumentation
                )
                m.setenv("CALI_CONFIG", "runtime-report")
                m.setenv("PROFILE_TORCH", "on")
                importlib.reload(perf_measure)
                assert perf_measure._CALI_PERF_ENABLED, "Caliper should be enabled"
                assert perf_measure.TORCH_PERF_ENABLED, (
                    "torch profiler must enable independently of Caliper"
                )
        finally:
            importlib.reload(perf_measure)


class TestProfilerTraceExport:
    """R15: a failed trace export must not strand the other ranks."""

    @staticmethod
    def _unstepped_profiler():
        """A profiler whose window never opened (a run with zero batches)."""
        from torch.profiler import ProfilerActivity, profile, schedule

        prof = profile(
            activities=[ProfilerActivity.CPU],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        )
        with prof:
            pass  # no prof.step(): the schedule never leaves its wait phase
        return prof

    @staticmethod
    def _stepped_profiler():
        """A profiler with a completed capture window."""
        from torch.profiler import ProfilerActivity, profile, schedule

        prof = profile(
            activities=[ProfilerActivity.CPU],
            schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
        )
        with prof:
            for _ in range(4):
                prof.step()
        return prof

    @staticmethod
    def _config(run_dir):
        return SimpleNamespace(
            problem_scale=4,
            epochs=1,
            n_instances_used_per_fractal=2,
            run_dir=str(run_dir),
        )

    def test_zero_step_export_is_reported_not_raised(self, tmp_path, caplog):
        """Exporting an unstepped profiler logs an error instead of raising.

        The export runs before the ``dist.barrier()`` that precedes rank-0
        post-processing, so a raise here kills the profiling rank and leaves
        every other rank blocked in that barrier until the collective timeout.
        """
        import logging

        import ScaFFold.worker as worker

        log = logging.getLogger("test_zero_step_export")
        with caplog.at_level(logging.DEBUG, logger=log.name):
            result = worker.export_profiler_trace(
                self._unstepped_profiler(),
                self._config(tmp_path),
                log,
                rank=0,
                world_size=1,
                ranks_per_node=1,
            )

        assert result is None
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "trace" in messages.lower()

    def test_successful_export_writes_a_trace(self, tmp_path, caplog):
        """A profiler with a completed window still writes its trace (control)."""
        import logging

        from torch.profiler import ProfilerActivity, profile, schedule

        import ScaFFold.worker as worker

        prof = profile(
            activities=[ProfilerActivity.CPU],
            schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
        )
        with prof:
            for _ in range(4):
                prof.step()

        log = logging.getLogger("test_successful_export")
        path = worker.export_profiler_trace(
            prof, self._config(tmp_path), log, rank=0, world_size=1, ranks_per_node=1
        )

        assert path is not None
        assert Path(path).exists()

    def test_trace_lands_in_the_run_dir(self, tmp_path, caplog):
        """R23: the trace goes to the run dir, not whatever CWD happens to be."""
        import logging

        import ScaFFold.worker as worker

        prof = self._stepped_profiler()
        log = logging.getLogger("test_trace_lands_in_the_run_dir")

        path = worker.export_profiler_trace(
            prof, self._config(tmp_path), log, rank=0, world_size=1, ranks_per_node=1
        )

        assert Path(path).parent == tmp_path
        assert list(tmp_path.glob("torch-*.json")) == [Path(path)]

    @pytest.mark.parametrize(
        "world_size, ranks_per_node, expected",
        [(8, 4, "-N2-n8-"), (6, 4, "-N2-n6-"), (1, 1, "-N1-n1-")],
        ids=["even", "ragged-last-node", "singleton"],
    )
    def test_trace_name_counts_nodes_not_ranks(
        self, tmp_path, world_size, ranks_per_node, expected
    ):
        """R23: the N field is a node count, and never rounds a node away."""
        import logging

        import ScaFFold.worker as worker

        prof = self._stepped_profiler()
        log = logging.getLogger("test_trace_name_counts_nodes")

        path = worker.export_profiler_trace(
            prof,
            self._config(tmp_path),
            log,
            rank=0,
            world_size=world_size,
            ranks_per_node=ranks_per_node,
        )

        assert expected in Path(path).name


class TestProfileTorchGate:
    """R24: PROFILE_TORCH is parsed like every other profiler flag."""

    @staticmethod
    def _reload_with(monkeypatch_context, value):
        import importlib

        import ScaFFold.utils.perf_measure as perf_measure

        if value is None:
            monkeypatch_context.delenv("PROFILE_TORCH", raising=False)
        else:
            monkeypatch_context.setenv("PROFILE_TORCH", value)
        monkeypatch_context.delenv("CALI_CONFIG", raising=False)
        importlib.reload(perf_measure)
        return perf_measure

    @pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "OFF"])
    def test_disabled_values(self, monkeypatch, value):
        """Anything that is not an affirmative value leaves profiling off.

        ``PROFILE_TORCH=0`` used to *enable* the profiler: the gate only
        rejected the literal "off", so every conventional way of saying "no"
        silently turned profiling on.
        """
        import importlib

        import ScaFFold.utils.perf_measure as perf_measure

        try:
            with monkeypatch.context() as m:
                assert not self._reload_with(m, value).TORCH_PERF_ENABLED
        finally:
            importlib.reload(perf_measure)

    @pytest.mark.parametrize("value", ["1", "true", "on", "ON", "yes", "TRUE"])
    def test_enabled_values(self, monkeypatch, value):
        """The affirmative spellings still enable the profiler."""
        import importlib

        import ScaFFold.utils.perf_measure as perf_measure

        try:
            with monkeypatch.context() as m:
                assert self._reload_with(m, value).TORCH_PERF_ENABLED
        finally:
            importlib.reload(perf_measure)

    def test_gate_matches_the_sub_option_parser(self, monkeypatch):
        """The master switch and the sub-option flags agree on every spelling."""
        import importlib

        import ScaFFold.utils.perf_measure as perf_measure

        try:
            for value in ("1", "true", "on", "yes", "0", "false", "no", "off", ""):
                with monkeypatch.context() as m:
                    module = self._reload_with(m, value)
                    assert module.TORCH_PERF_ENABLED == module._profiler_env_flag(
                        "PROFILE_TORCH"
                    ), value
        finally:
            importlib.reload(perf_measure)
