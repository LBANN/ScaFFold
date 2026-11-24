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
    "dataset_seed": 523,
    "training_seed": 42,
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


def test_activation_checkpointing_is_a_real_option():
    """Activation checkpointing is reachable from a config, defaulting off.

    The U-Net has always had ``use_checkpointing``, but with no config key and
    no caller it could not be turned on: any attempt was rejected as an unknown
    key.
    """
    cfg = config_utils.Config(dict(BASE))
    assert cfg.activation_checkpointing is False
    assert (
        config_utils.Config({**BASE, "activation_checkpointing": 1})
    ).activation_checkpointing is True
    assert (
        config_utils.Config({**BASE, "activation_checkpointing": True})
    ).activation_checkpointing is True
    assert (
        config_utils.Config({**BASE, "activation_checkpointing": 0})
    ).activation_checkpointing is False


@pytest.mark.parametrize("value", [2, -1, "yes", 0.0])
def test_activation_checkpointing_rejects_non_flag_values(value):
    """Anything that is not a 0/1 (or bool) toggle is rejected by name."""
    with pytest.raises(ValueError, match="activation_checkpointing"):
        config_utils.Config({**BASE, "activation_checkpointing": value})


def test_activation_checkpointing_documented_in_the_default_config():
    """The shipped config is the parameter reference, so the key lives there."""
    text = (CONFIG_DIR / "benchmark_default.yml").read_text()
    assert "activation_checkpointing:" in text


def test_execution_mode_defaults_to_triton_eager():
    for data in (
        BASE,
        yaml.safe_load((CONFIG_DIR / "benchmark_default.yml").read_text()),
    ):
        cfg = config_utils.Config(data)
        assert cfg.execution_mode == "triton-eager"
        assert not any(getattr(cfg, name) for name in config_utils.COMPILE_FLAGS)


@pytest.mark.parametrize("mode", config_utils.EXECUTION_MODES)
@pytest.mark.parametrize("shards", [[1, 1, 1], [2, 1, 1], [1, 1, 2], [2, 2, 1]])
def test_execution_mode_selects_compilation_route(mode, shards):
    cfg = config_utils.Config({**BASE, "execution_mode": mode, "dc_num_shards": shards})
    compiled = mode in ("native-compiled", "triton-compiled")
    spatial = compiled and any(n > 1 for n in shards)
    assert cfg.execution_mode == mode
    assert cfg.torch_compile is compiled
    assert cfg.torch_compile_fullgraph is compiled
    assert cfg.torch_compile_unsharded is (compiled and not spatial)
    assert cfg.torch_compile_spatial is spatial
    assert cfg.torch_compile_loss is False


@pytest.mark.parametrize(
    "mode", [None, True, 0, "native", "NATIVE-EAGER", ["native-eager"], {}]
)
def test_execution_mode_rejects_invalid_values(mode):
    with pytest.raises(ValueError, match="execution_mode"):
        config_utils.Config({**BASE, "execution_mode": mode})


@pytest.mark.parametrize(
    "mode", ["native-eager", "triton-eager", "native-compiled", "triton-compiled"]
)
def test_presets_own_advanced_compilation_flags(mode):
    # Conflicting custom flags from an override or saved run cannot change a
    # named preset's meaning; only their types are validated in preset modes.
    cfg = config_utils.Config(
        {**BASE, "execution_mode": mode, **dict.fromkeys(config_utils.COMPILE_FLAGS, 1)}
    )
    expected = config_utils.Config({**BASE, "execution_mode": mode})
    assert vars(cfg) == vars(expected)


@pytest.mark.parametrize("name", config_utils.COMPILE_FLAGS)
@pytest.mark.parametrize("mode", ["triton-eager", "custom"])
def test_inactive_compile_flag_types_are_still_validated(name, mode):
    with pytest.raises(ValueError, match=name):
        config_utils.Config({**BASE, "execution_mode": mode, name: "yes"})


@pytest.mark.parametrize("mode", config_utils.EXECUTION_MODES)
def test_execution_mode_saved_config_roundtrip(tmp_path, mode):
    cfg = config_utils.Config(
        {**BASE, "execution_mode": mode, "dc_num_shards": [2, 1, 1]}
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(vars(cfg)))
    reloaded = config_utils.load_config(str(path), "benchmark")
    assert vars(reloaded) == vars(cfg)


def test_saved_preset_recomputes_flags_after_mode_and_shard_changes(tmp_path):
    cfg = config_utils.Config({**BASE, "execution_mode": "native-compiled"})
    saved = tmp_path / "config.yaml"
    saved.write_text(yaml.safe_dump(vars(cfg)))
    override = tmp_path / "override.yml"
    override.write_text(yaml.safe_dump({"dc_num_shards": [1, 2, 1]}))
    spatial = config_utils.Config(config_utils.load_config_files([saved, override]))
    assert spatial.torch_compile_spatial
    assert not spatial.torch_compile_unsharded
    eager = config_utils.Config({**vars(spatial), "execution_mode": "triton-eager"})
    assert not any(getattr(eager, name) for name in config_utils.COMPILE_FLAGS)


CUSTOM_BASE = {**BASE, "execution_mode": "custom"}


def test_torch_compile_config_option():
    base = yaml.safe_load((CONFIG_DIR / "benchmark_default.yml").read_text())
    base["execution_mode"] = "custom"
    assert config_utils.Config({**base, "torch_compile": 0}).torch_compile is False
    assert (
        config_utils.Config(
            {**base, "torch_compile": 1, "torch_compile_unsharded": 1}
        ).torch_compile
        is True
    )
    with pytest.raises(ValueError, match="torch_compile"):
        config_utils.Config({**base, "torch_compile": "yes"})
    assert "torch_compile:" in (CONFIG_DIR / "benchmark_default.yml").read_text()


def test_torch_compile_unsharded_requires_compile_and_one_spatial_shard():
    assert config_utils.Config(dict(CUSTOM_BASE)).torch_compile_unsharded is False
    enabled = {**CUSTOM_BASE, "torch_compile": 1, "torch_compile_unsharded": 1}
    assert config_utils.Config(enabled).torch_compile_unsharded is True
    with pytest.raises(ValueError, match="torch_compile_unsharded"):
        config_utils.Config({**CUSTOM_BASE, "torch_compile_unsharded": 1})
    with pytest.raises(ValueError, match="dc_num_shards"):
        config_utils.Config({**enabled, "dc_num_shards": [2, 1, 1]})
    with pytest.raises(ValueError, match="torch_compile_unsharded"):
        config_utils.Config({**CUSTOM_BASE, "torch_compile_unsharded": "yes"})


def test_torch_compile_spatial_requires_fullgraph_and_spatial_shards():
    enabled = {
        **CUSTOM_BASE,
        "torch_compile": 1,
        "torch_compile_fullgraph": 1,
        "torch_compile_spatial": 1,
        "dc_num_shards": [2, 1, 1],
    }
    assert config_utils.Config(enabled).torch_compile_spatial is True
    for missing in ("torch_compile", "torch_compile_fullgraph"):
        with pytest.raises(ValueError, match="torch_compile"):
            config_utils.Config({**enabled, missing: 0})
    with pytest.raises(ValueError, match="spatial shard count"):
        config_utils.Config({**enabled, "dc_num_shards": [1, 1, 1]})
    with pytest.raises(ValueError, match="exclusive"):
        config_utils.Config({**enabled, "torch_compile_unsharded": 1})
    checkpointed = config_utils.Config(
        {**enabled, "activation_checkpointing": 1, "torch_compile_loss": 1}
    )
    assert checkpointed.activation_checkpointing and checkpointed.torch_compile_loss
    with pytest.raises(ValueError, match="torch_compile_spatial"):
        config_utils.Config({**enabled, "torch_compile_spatial": "yes"})


def test_torch_compile_fullgraph_requires_compile():
    assert config_utils.Config(dict(CUSTOM_BASE)).torch_compile_fullgraph is False
    enabled = {
        **CUSTOM_BASE,
        "torch_compile": 1,
        "torch_compile_fullgraph": 1,
        "torch_compile_unsharded": 1,
    }
    assert config_utils.Config(enabled).torch_compile_fullgraph is True
    with pytest.raises(ValueError, match="torch_compile_fullgraph"):
        config_utils.Config({**CUSTOM_BASE, "torch_compile_fullgraph": 1})
    with pytest.raises(ValueError, match="torch_compile_fullgraph"):
        config_utils.Config({**CUSTOM_BASE, "torch_compile_fullgraph": "yes"})


@pytest.mark.parametrize("fullgraph", [False, True])
def test_torch_compile_requires_a_local_route(fullgraph):
    with pytest.raises(ValueError, match="requires a local compilation route"):
        config_utils.Config(
            {
                **CUSTOM_BASE,
                "torch_compile": 1,
                "torch_compile_fullgraph": fullgraph,
            }
        )


def test_torch_compile_loss_requires_local_route():
    assert config_utils.Config(dict(CUSTOM_BASE)).torch_compile_loss is False
    enabled = {
        **CUSTOM_BASE,
        "torch_compile": 1,
        "torch_compile_unsharded": 1,
        "torch_compile_fullgraph": 1,
        "torch_compile_loss": 1,
    }
    assert config_utils.Config(enabled).torch_compile_loss is True
    assert config_utils.Config(
        {**enabled, "torch_compile_fullgraph": 0}
    ).torch_compile_loss
    with pytest.raises(ValueError, match="torch_compile_loss"):
        config_utils.Config({**enabled, "torch_compile_unsharded": 0})
    spatial = {
        **enabled,
        "torch_compile_unsharded": 0,
        "torch_compile_spatial": 1,
        "dc_num_shards": [2, 1, 1],
    }
    assert config_utils.Config(spatial).torch_compile_loss
    with pytest.raises(ValueError, match="torch_compile"):
        config_utils.Config({**spatial, "torch_compile_fullgraph": 0})
    with pytest.raises(ValueError, match="torch_compile_loss"):
        config_utils.Config({**enabled, "torch_compile_loss": "yes"})
    assert "torch_compile_loss:" in (CONFIG_DIR / "benchmark_default.yml").read_text()


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
    override = _write_yaml(tmp_path / "o.yml", {"training_seed": 7, "target_dice": 0.5})
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


# ---------------------------------------------------------------------------
# The base config is preserved under a name of its own
# ---------------------------------------------------------------------------


def _run_benchmark_with_base(monkeypatch, tmp_path, base_name):
    """Run the driver with a base config named ``base_name``; return the run dir.

    The run dir starts out holding the merged ``config.yaml`` the CLI writes,
    so the test can tell whether the copy of the base config overwrote it.
    """
    import ScaFFold.benchmark as benchmark_mod
    import ScaFFold.worker as worker_mod

    monkeypatch.setattr(worker_mod, "main", lambda kwargs_dict={}: None)
    monkeypatch.setattr(benchmark_mod.worker, "main", worker_mod.main)

    base = _write_yaml(tmp_path / base_name, BASE)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    merged = {**BASE, "local_batch_size": 2, "machine_name": "some-host"}
    _write_yaml(run_dir / "config.yaml", merged)

    kwargs = {
        **merged,
        "command": "benchmark",
        "restart": False,
        "verbose": 0,
        "config": base,
        "benchmark_run_dir": str(run_dir),
    }
    benchmark_mod.main(kwargs_dict=kwargs)
    return run_dir


@pytest.mark.parametrize("base_name", ["config.yaml", "base.yml"])
def test_base_config_copy_never_clobbers_merged_config(
    monkeypatch, tmp_path, base_name
):
    """The merged config.yaml survives even when the base file has that name.

    Copying the base config into the run dir under its own name silently
    overwrote the merged dump -- losing the CLI overrides, machine name and
    scheduler metadata -- and, since restart.sh points -c at
    ``$RUN_DIR/config.yaml``, a restart then reloaded the raw base config.
    """
    run_dir = _run_benchmark_with_base(monkeypatch, tmp_path, base_name)

    merged = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert merged["local_batch_size"] == 2
    assert merged["machine_name"] == "some-host"

    preserved = yaml.safe_load((run_dir / "base_config.yaml").read_text())
    assert preserved["local_batch_size"] == BASE["local_batch_size"]
    assert "machine_name" not in preserved


# ---------------------------------------------------------------------------
# unet_bottleneck_dim range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "problem_scale, bottleneck",
    [(5, -1), (5, 5), (5, 6)],
    ids=["negative", "zero-layers", "negative-layers"],
)
def test_bottleneck_out_of_range_rejected(problem_scale, bottleneck):
    """An out-of-range bottleneck fails at config time, naming both keys.

    Left unvalidated it produced a U-Net with more pooling levels than the
    volume has, and the run died hours later inside max_pool3d with "Given
    input size: (2048x1x1x1)" -- naming no config key at all.
    """
    bad = {**BASE, "problem_scale": problem_scale, "unet_bottleneck_dim": bottleneck}

    with pytest.raises(ValueError) as excinfo:
        config_utils.Config(bad)

    message = str(excinfo.value)
    assert "unet_bottleneck_dim" in message
    assert "problem_scale" in message
    assert str(bottleneck) in message
    assert str(problem_scale) in message


@pytest.mark.parametrize("bottleneck", [0, 3, 4])
def test_bottleneck_in_range_accepted(bottleneck):
    """The full valid range (at least one U-Net layer) is accepted."""
    cfg = config_utils.Config(
        {**BASE, "problem_scale": 5, "unet_bottleneck_dim": bottleneck}
    )
    assert cfg.unet_layers == 5 - bottleneck
    assert cfg.unet_layers >= 1
