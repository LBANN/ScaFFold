"""Real compilation of native U-Net operators and halos on CPU or GPU meshes."""

import json
from pathlib import Path

import pytest

from tests.helpers.compile_runtime import (
    compile_timeout,
    require_compile_device,
    run_compile_ranks,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.timeout(compile_timeout(300) + 60),
]


def _run_spatial_case(
    backend,
    bf16,
    tmp_path,
    *,
    transitions=False,
    eval_first=False,
    num_ranks=2,
    num_shards=(2, 1, 1),
    layers=1,
    local_shapes=((1, 4, 4, 4), (1, 4, 4, 4)),
    timeout=300,
):
    require_compile_device(ranks=num_ranks, bf16=bf16)
    script = (
        Path(__file__).parent / "helpers" / "rank_scripts" / "compile_spatial_2rank.py"
    )
    rc, out, err = run_compile_ranks(
        str(script),
        n=num_ranks,
        timeout=timeout,
        env={
            "SCAFFOLD_TEST_COMPILE_BACKEND": backend,
            "SCAFFOLD_TEST_BF16": str(int(bf16)),
            "SCAFFOLD_TEST_TRANSITIONS": str(int(transitions)),
            "SCAFFOLD_TEST_EVAL_FIRST": str(int(eval_first)),
            "SCAFFOLD_TEST_NUM_SHARDS": json.dumps(num_shards),
            "SCAFFOLD_TEST_SHARD_DIMS": "[2,3,4]",
            "SCAFFOLD_TEST_LAYERS": str(layers),
            "SCAFFOLD_TEST_LOCAL_SHAPES": json.dumps(local_shapes),
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    (tmp_path / "ranks.stdout").write_text(out)
    (tmp_path / "ranks.stderr").write_text(err)
    assert rc == 0, f"{num_ranks}-rank compilation failed ({rc})\n{out}\n{err}"
    assert all(f"RANK {rank} DONE" in out for rank in range(num_ranks)), out


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_native_spatial_compile_matches_eager(backend, bf16, tmp_path):
    _run_spatial_case(backend, bf16, tmp_path)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("eval_first", [False, True], ids=["train_first", "eval_first"])
def test_native_spatial_train_eval_transitions(backend, bf16, eval_first, tmp_path):
    _run_spatial_case(backend, bf16, tmp_path, transitions=True, eval_first=eval_first)


@pytest.mark.timeout(compile_timeout(540) + 60)
@pytest.mark.parametrize(
    "backend,bf16",
    [
        pytest.param("aot_eager", False, id="aot_eager_fp32"),
        pytest.param("aot_eager", True, id="aot_eager_bf16"),
        pytest.param("inductor", False, id="inductor_fp32"),
        pytest.param("inductor", True, id="inductor_bf16"),
    ],
)
def test_deeper_spatial_unet_changes_shape_and_reuses_prior_graphs(
    backend, bf16, tmp_path
):
    """A second U-Net level and new batch/spatial shapes must retain every halo.

    Both noncubic shapes return after their first trace. This checks that the
    production adapter's initially sampled halo plans remain valid and that
    fixed-shape compilation caches both signatures instead of recompiling.
    """
    _run_spatial_case(
        backend,
        bf16,
        tmp_path,
        layers=2,
        local_shapes=(
            (1, 8, 8, 12),
            (2, 8, 12, 8),
            (2, 8, 12, 8),
            (1, 8, 8, 12),
        ),
        timeout=540,
    )


@pytest.mark.timeout(compile_timeout(540) + 60)
@pytest.mark.parametrize(
    "num_ranks,num_shards,bf16,transitions,local_shape",
    [
        pytest.param(
            4, (2, 2, 1), False, False, (1, 4, 4, 6), id="four_ranks_two_axes"
        ),
        pytest.param(4, (4, 1, 1), True, False, (1, 4, 6, 4), id="four_depth_shards"),
        pytest.param(4, (2, 1, 1), False, True, (1, 4, 6, 4), id="two_data_replicas"),
        pytest.param(2, (1, 1, 2), True, False, (1, 6, 4, 4), id="two_width_shards"),
    ],
)
def test_native_spatial_compile_mesh_layouts(
    num_ranks, num_shards, bf16, transitions, local_shape, tmp_path
):
    """Cover another shard axis, interior ranks, corners, and data replicas."""
    _run_spatial_case(
        "inductor",
        bf16,
        tmp_path,
        num_ranks=num_ranks,
        num_shards=num_shards,
        local_shapes=(local_shape, local_shape),
        transitions=transitions,
        eval_first=transitions,
        timeout=540,
    )
