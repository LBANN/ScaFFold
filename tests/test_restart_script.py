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

"""Content tests for the generated ``restart.sh``.

Each test generates a real ``restart.sh`` into ``tmp_path`` and asserts on the
emitted text. No GPU, MPI launcher, or scheduler is required.

The generator inspects ``sys.argv`` and the process environment to decide what
to emit, so every test pins both: ``sys.argv`` is set explicitly and the launch
/ profiling environment variables are scrubbed (then re-set as needed) so the
generating-process sniffing sees exactly the intended state.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from ScaFFold.utils import create_restart_script as crs

# Launcher / scheduler variables the generator may sniff for the launch shape.
_LAUNCHER_ENV_VARS = (
    "FLUX_JOB_ID",
    "FLUX_JOB_NNODES",
    "FLUX_JOB_SIZE",
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_NTASKS",
    "SLURM_NPROCS",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NNODES",
    "WORLD_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PALS_NRANKS",
)

# Profiling variables re-exported only when set in the generating run.
_PROFILING_ENV_VARS = ("PROFILE_TORCH", "CALI_CONFIG")

# Site-specific tokens the old generator baked in unconditionally; a portable
# restart script must not contain any of them.
_FOREIGN_ENV_TOKENS = (
    "cce/21.0.0",
    "cray-mpich",
    "rccl/fast-env-slows-mpi",
    "LD_PRELOAD",
    "SPINDLE_FLUXOPT",
    "rocm/7.1.1",
)

_DEFAULT_ARGV = [
    "/usr/bin/scaffold",
    "benchmark",
    "-c",
    "/some/where/config.yml",
    "--base-run-dir",
    "/gpfs/runs",
    "--epochs",
    "10",
]


def _isolate_env(monkeypatch):
    """Scrub every variable the generator inspects so a test starts clean."""
    for var in _LAUNCHER_ENV_VARS + _PROFILING_ENV_VARS + ("VIRTUAL_ENV",):
        monkeypatch.delenv(var, raising=False)


def _generate(monkeypatch, run_dir, *, argv=None, world_size=None):
    """Generate ``restart.sh`` under ``run_dir`` and return its text.

    ``world_size=None`` exercises the default path (shape derived from the
    environment); an integer exercises the explicit-parameter path callers use.
    """
    monkeypatch.setattr(sys, "argv", list(argv if argv is not None else _DEFAULT_ARGV))
    if world_size is None:
        out_path = crs.create_restart_script(run_dir)
    else:
        out_path = crs.create_restart_script(run_dir, world_size=world_size)
    return out_path.read_text()


def test_no_foreign_env_injected(monkeypatch, tmp_path):
    """A clean generating env yields no site modules and no PROFILE_TORCH export."""
    _isolate_env(monkeypatch)

    script = _generate(monkeypatch, tmp_path / "run")

    for token in _FOREIGN_ENV_TOKENS:
        assert token not in script, f"unexpected foreign env token {token!r} in script"
    assert "PROFILE_TORCH" not in script, (
        "PROFILE_TORCH must not be exported when unset in the generating run"
    )
    assert "CALI_CONFIG" not in script


def test_profiler_env_preserved_when_set(monkeypatch, tmp_path):
    """PROFILE_TORCH set in the generating run is re-exported in the script."""
    _isolate_env(monkeypatch)
    monkeypatch.setenv("PROFILE_TORCH", "1")

    script = _generate(monkeypatch, tmp_path / "run")

    assert "export PROFILE_TORCH" in script


def test_template_multirank(monkeypatch, tmp_path):
    """OMPI env with size>1 (no SLURM/FLUX) selects the torchrun-hpc template."""
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    script = _generate(monkeypatch, tmp_path / "run")

    assert "torchrun-hpc" in script
    assert 'exec "${PY[@]}"' not in script


def test_template_singlerank(monkeypatch, tmp_path):
    """No launcher env (single rank) selects the local template."""
    _isolate_env(monkeypatch)

    script = _generate(monkeypatch, tmp_path / "run")

    assert "torchrun-hpc" not in script
    assert 'exec "${PY[@]}"' in script


def test_explicit_world_size_overrides_env(monkeypatch, tmp_path):
    """An explicit world_size>1 selects torchrun-hpc even with no launcher env.

    This is the interface the CLI uses: it holds ``MPI.COMM_WORLD`` and passes
    the true rank count, which must win over environment sniffing.
    """
    _isolate_env(monkeypatch)

    script = _generate(monkeypatch, tmp_path / "run", world_size=4)

    assert "torchrun-hpc" in script
    assert 'exec "${PY[@]}"' not in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_generated_script_is_valid_bash(monkeypatch, tmp_path):
    """Every generated variant passes ``bash -n`` (syntax check)."""
    variants = []

    _isolate_env(monkeypatch)
    variants.append(_generate(monkeypatch, tmp_path / "single"))

    _isolate_env(monkeypatch)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    variants.append(_generate(monkeypatch, tmp_path / "multi"))

    _isolate_env(monkeypatch)
    monkeypatch.setenv("PROFILE_TORCH", "1")
    monkeypatch.setenv("CALI_CONFIG", "runtime-report")
    variants.append(_generate(monkeypatch, tmp_path / "profiled"))

    for i, script in enumerate(variants):
        script_path = tmp_path / f"variant_{i}.sh"
        script_path.write_text(script)
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"bash -n failed for variant {i}:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Combined ``--flag=value`` tokens
# ---------------------------------------------------------------------------


def test_config_equals_form_is_substituted(monkeypatch, tmp_path):
    """``--config=PATH`` is repointed at the run dir, not left as a placeholder.

    The rewriter emits the placeholder as part of a combined token, so a
    substitution that only matches whole tokens leaves ``--config=__CFG__`` in
    the script and the restart dies with "Config file '__CFG__' not found".
    """
    _isolate_env(monkeypatch)
    argv = [
        "/usr/bin/scaffold",
        "benchmark",
        "--config=/some/where/config.yml",
        "--epochs",
        "10",
    ]

    script = _generate(monkeypatch, tmp_path / "run", argv=argv)

    assert "__CFG__" not in script
    assert '--config="$RUN_DIR/config.yaml"' in script
    assert "/some/where/config.yml" not in script


@pytest.mark.parametrize(
    "config_argv",
    [
        ["-c", "/some/where/config.yml"],
        ["--config", "/some/where/config.yml"],
        ["--config=/some/where/config.yml"],
    ],
    ids=["short", "long-space", "long-equals"],
)
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_every_config_spelling_expands_to_the_run_dir_config(
    monkeypatch, tmp_path, config_argv
):
    """Bash expands every ``--config`` spelling to ``$RUN_DIR/config.yaml``.

    The generated PY array is sourced and expanded by a real shell so the test
    asserts on the arguments the restarted CLI actually receives.
    """
    _isolate_env(monkeypatch)
    argv = ["/usr/bin/scaffold", "benchmark"] + config_argv

    script = _generate(monkeypatch, tmp_path / "run", argv=argv)

    py_decl = next(line for line in script.splitlines() if line.startswith("PY=("))
    probe = tmp_path / f"probe_{config_argv[0][-1]}.sh"
    probe.write_text(f'RUN_DIR=/run/dir\n{py_decl}\nprintf "%s\\n" "${{PY[@]}}"\n')
    result = subprocess.run(
        ["bash", str(probe)], capture_output=True, text=True, check=True
    )

    tokens = result.stdout.split("\n")
    assert "/run/dir/config.yaml" in tokens or (
        "--config=/run/dir/config.yaml" in tokens
    )
    assert not any("__CFG__" in tok for tok in tokens)


def test_run_dir_placeholder_is_substituted(monkeypatch, tmp_path):
    """The appended ``--run-dir`` placeholder still resolves (control)."""
    _isolate_env(monkeypatch)

    script = _generate(monkeypatch, tmp_path / "run")

    assert "__RUN_DIR__" not in script
    assert '--run-dir "$RUN_DIR"' in script


# ---------------------------------------------------------------------------
# Launch-shape sniffing must match the rank side
# ---------------------------------------------------------------------------

# Every variable ``ScaFFold.utils.distributed.get_world_size`` honors. The
# restart generator must derive the same world size from each of them, or a
# restart script silently relaunches the job at the wrong scale.
_WORLD_SIZE_ENV_VARS = (
    "WORLD_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PALS_NRANKS",
    "SLURM_NTASKS",
    "FLUX_JOB_SIZE",
)


@pytest.mark.parametrize("var", _WORLD_SIZE_ENV_VARS)
def test_sniffed_world_size_matches_rank_side(monkeypatch, var):
    """The generator and ``get_world_size`` agree on every launcher variable."""
    from ScaFFold.utils.distributed import get_world_size

    _isolate_env(monkeypatch)
    for other in _WORLD_SIZE_ENV_VARS:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(var, "8")

    _, _, sniffed = crs._sniff_launch_shape(os.environ)

    assert sniffed == 8, f"{var} not recognized by the restart generator"
    assert sniffed == get_world_size()


def test_pals_job_gets_a_multirank_restart_script(monkeypatch, tmp_path):
    """A Cray PALS launch (PALS_NRANKS) emits the multi-rank template."""
    _isolate_env(monkeypatch)
    monkeypatch.setenv("PALS_NRANKS", "8")

    script = _generate(monkeypatch, tmp_path / "run")

    assert "torchrun-hpc" in script
    assert 'exec "${PY[@]}"' not in script


# ---------------------------------------------------------------------------
# VC-5: the node shape comes from the local rank count when no scheduler
# reports one. Flux and Slurm state their node count; PALS and plain torchrun
# do not, and assuming one node relaunched an 8-rank/2-node job as
# NODES=1 TASKS_PER_NODE=8 -- an oversubscribed node, or a rejected job.
# ---------------------------------------------------------------------------


def test_pals_multinode_shape_uses_the_local_rank_count(monkeypatch, tmp_path):
    """8 PALS ranks, 4 per node -> NODES=2, TASKS_PER_NODE=4."""
    _isolate_env(monkeypatch)
    monkeypatch.delenv("PALS_LOCAL_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("PALS_NRANKS", "8")
    monkeypatch.setenv("PALS_LOCAL_SIZE", "4")

    script = _generate(monkeypatch, tmp_path / "run")

    assert 'NODES="2"' in script
    assert 'TASKS_PER_NODE="4"' in script


def test_single_node_torchrun_shape_is_unchanged(monkeypatch, tmp_path):
    """8 torchrun ranks all on one node stay NODES=1, TASKS_PER_NODE=8."""
    _isolate_env(monkeypatch)
    monkeypatch.delenv("PALS_LOCAL_SIZE", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")

    script = _generate(monkeypatch, tmp_path / "run")

    assert 'NODES="1"' in script
    assert 'TASKS_PER_NODE="8"' in script


def test_unknown_local_size_keeps_the_single_node_assumption(monkeypatch, tmp_path):
    """With nothing reporting a per-node count, the old assumption stands."""
    _isolate_env(monkeypatch)
    for var in ("PALS_LOCAL_SIZE", "LOCAL_WORLD_SIZE", "PMI_LOCAL_SIZE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PALS_NRANKS", "8")

    script = _generate(monkeypatch, tmp_path / "run")

    assert 'NODES="1"' in script
    assert 'TASKS_PER_NODE="8"' in script
