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

"""Tests for config loading/validation, override merging, and the benchmark driver."""

from pathlib import Path

import pytest
import yaml

from ScaFFold.utils import config_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "ScaFFold" / "configs"

# A minimal complete config dict (mirrors benchmark_default.yml).
BASE = {
    "base_run_dir": "benchmark_runs",
    "dataset_dir": "datasets",
    "fract_base_dir": "fractals",
    "n_categories": 5,
    "n_instances_used_per_fractal": 145,
    "problem_scale": 7,
    "unet_bottleneck_dim": 3,
    "seed": 42,
    "local_batch_size": 1,
    "dataloader_num_workers": 1,
    "optimizer": "ADAM",
    "dc_num_shards": [1, 1, 1],
    "dc_shard_dims": [2, 3, 4],
    "checkpoint_interval": -1,
    "variance_threshold": 0.15,
    "n_fracts_per_vol": 3,
    "val_split": 30,
    "epochs": -1,
    "starting_learning_rate": 0.001,
    "min_learning_rate": 0.0001,
    "T_0": 100,
    "T_mult": 2,
    "disable_scheduler": 0,
    "more_determinism": 0,
    "datagen_from_scratch": 0,
    "train_from_scratch": 1,
    "dist": 1,
    "torch_amp": 1,
    "framework": "torch",
    "checkpoint_dir": "checkpoints",
    "loss_freq": 1,
    "normalize": 1,
    "group_norm_groups": 8,
    "warmup_batches": 64,
    "ce_weight_sample_fraction": 0.1,
    "dataset_reuse_enforce_commit_id": 0,
    "target_dice": 0.95,
}


def test_unknown_key_rejected():
    """A typo'd/unrecognized key is rejected by name instead of silently dropped."""
    bad = dict(BASE)
    bad["asynch_save"] = True
    with pytest.raises(ValueError, match="asynch_save"):
        config_utils.Config(bad)


def test_unknown_key_warns_when_not_strict(capsys):
    """strict=False downgrades unknown keys to a warning."""
    bad = dict(BASE)
    bad["asynch_save"] = True
    config_utils.Config(bad, strict=False)
    assert "asynch_save" in capsys.readouterr().out


@pytest.mark.parametrize(
    "config_path", sorted(CONFIG_DIR.glob("*.yml")), ids=lambda p: p.name
)
def test_documented_keys_accepted(config_path):
    """Every shipped config file loads cleanly under strict validation."""
    config_utils.load_config(str(config_path), "benchmark")


def test_invalid_type_message_names_type(tmp_path):
    """The invalid-config-type error names the offending type argument."""
    path = tmp_path / "c.yml"
    path.write_text(yaml.dump(BASE))
    with pytest.raises(ValueError, match="bogus"):
        config_utils.load_config(str(path), "bogus")


def test_async_save_is_real_option():
    """async_save is an accepted, defaulted option (consumed by the trainer)."""
    cfg = config_utils.Config(dict(BASE))
    assert cfg.async_save is False
    cfg2 = config_utils.Config({**BASE, "async_save": True})
    assert cfg2.async_save is True


# ---------------------------------------------------------------------------
# --config override merging
# ---------------------------------------------------------------------------


def _write_yaml(path, data):
    path.write_text(yaml.dump(data))
    return str(path)


def test_second_config_merges(tmp_path):
    """A second --config file is a partial override of the base, not a replacement."""
    base = _write_yaml(tmp_path / "base.yml", BASE)
    override = _write_yaml(
        tmp_path / "override.yml", {"local_batch_size": 4, "epochs": 3}
    )
    merged = config_utils.load_config_files([base, override])
    assert merged["local_batch_size"] == 4
    assert merged["epochs"] == 3
    # Untouched base values persist.
    assert merged["n_categories"] == BASE["n_categories"]
    assert merged["optimizer"] == BASE["optimizer"]


def test_partial_override_no_keyerror(tmp_path):
    """A two-key override file merges cleanly (no KeyError on missing keys)."""
    base = _write_yaml(tmp_path / "base.yml", BASE)
    override = _write_yaml(tmp_path / "o.yml", {"seed": 7, "target_dice": 0.5})
    merged = config_utils.load_config_files([base, override])
    config_utils.Config(merged)  # must construct


def test_override_typo_rejected(tmp_path):
    """An unknown key in an override file is rejected by name."""
    base = _write_yaml(tmp_path / "base.yml", BASE)
    override = _write_yaml(tmp_path / "o.yml", {"bacth_size": 4})
    with pytest.raises(ValueError, match="bacth_size"):
        config_utils.load_config_files([base, override])


def test_config_yaml_roundtrip(tmp_path):
    """The combined config the CLI writes to a run dir reloads cleanly."""
    combined = dict(BASE)
    # Keys the CLI layer injects before saving config.yaml.
    combined.update(
        {
            "command": "benchmark",
            "config": "some.yml",
            "verbose": 0,
            "restart": False,
            "run_dir": None,
            "benchmark_run_dir": str(tmp_path),
            "unet_layers": 4,
            "vol_size": 128,
            "point_num": 8192,
            "scheduler_metadata": {},
            "machine_name": "host",
            "datagen_batch_size": 10000,
        }
    )
    path = _write_yaml(tmp_path / "config.yaml", combined)
    config_utils.load_config(path, "benchmark")  # must not raise


# ---------------------------------------------------------------------------
# Single-run benchmark driver; list-valued (sweep) params are rejected
# ---------------------------------------------------------------------------


def _run_benchmark(monkeypatch, tmp_path, config_updates):
    """Drive benchmark.main with a recording worker; return recorded configs."""
    import ScaFFold.benchmark as benchmark_mod
    import ScaFFold.worker as worker_mod

    calls = []
    monkeypatch.setattr(
        worker_mod, "main", lambda kwargs_dict={}: calls.append(dict(kwargs_dict))
    )
    monkeypatch.setattr(benchmark_mod.worker, "main", worker_mod.main)

    cfg_file = _write_yaml(tmp_path / "bench.yml", BASE)
    kwargs = dict(BASE)
    kwargs.update(
        {
            "command": "benchmark",
            "restart": False,
            "verbose": 0,
            "config": cfg_file,
            "benchmark_run_dir": str(tmp_path / "run"),
        }
    )
    (tmp_path / "run").mkdir()
    kwargs.update(config_updates)
    benchmark_mod.main(kwargs_dict=kwargs)
    return calls


def test_benchmark_runs_worker_once(monkeypatch, tmp_path):
    """One benchmark invocation runs the worker exactly once, in the run dir."""
    calls = _run_benchmark(monkeypatch, tmp_path, {})
    assert len(calls) == 1
    assert calls[0]["local_batch_size"] == BASE["local_batch_size"]
    # The run uses the benchmark run dir directly: no per-combination subdirs.
    run_root = tmp_path / "run"
    assert calls[0]["run_dir"] == str(run_root)
    assert list(run_root.glob("param_set_*")) == []


def test_no_sweep_expansion_helper():
    """The sweep-expansion entry point is gone from the benchmark driver."""
    import ScaFFold.benchmark as benchmark_mod

    assert not hasattr(benchmark_mod, "expand_sweep_combinations")


@pytest.mark.parametrize(
    "key, value",
    [
        ("problem_scale", [6, 7]),
        ("unet_bottleneck_dim", [3, 4]),
        ("n_categories", [5, 10]),
        ("local_batch_size", [1, 2]),
    ],
)
def test_list_valued_scalar_key_rejected(key, value):
    """A list value for a scalar key is rejected by name, not by a stray TypeError."""
    bad = dict(BASE)
    bad[key] = value
    with pytest.raises(ValueError) as excinfo:
        config_utils.Config(bad)
    message = str(excinfo.value)
    assert key in message
    assert "parameter sweeps are no longer supported" in message.lower()
    assert str(value) in message


def test_list_valued_key_rejected_by_load_config(tmp_path):
    """The benchmark config load path rejects sweeps with the same clear error."""
    path = _write_yaml(tmp_path / "sweepy.yml", {**BASE, "problem_scale": [6, 7]})
    with pytest.raises(ValueError) as excinfo:
        config_utils.load_config(path, "benchmark")
    message = str(excinfo.value)
    assert "problem_scale" in message
    assert "parameter sweeps are no longer supported" in message.lower()


def test_all_list_valued_keys_named():
    """Every offending key is named in one error, not just the first one hit."""
    bad = {**BASE, "problem_scale": [6, 7], "local_batch_size": [1, 2]}
    with pytest.raises(ValueError) as excinfo:
        config_utils.Config(bad)
    message = str(excinfo.value)
    assert "problem_scale" in message
    assert "local_batch_size" in message


def test_list_valued_key_rejected_in_run_config(tmp_path):
    """A list reaching the worker's RunConfig is rejected before any setup."""
    bad = {
        **BASE,
        "local_batch_size": [1, 2],
        "run_dir": str(tmp_path),
        "run_iter": str(tmp_path / "run"),
    }
    with pytest.raises(ValueError, match="local_batch_size"):
        config_utils.RunConfig(bad)
