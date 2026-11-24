"""Spatial compiled loss must preserve distributed training and evaluation."""

from pathlib import Path

import pytest

from tests.helpers.compile_runtime import (
    compile_timeout,
    require_compile_device,
    run_compile_ranks,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.timeout(compile_timeout(360) + 60),
]


def _run_loss_case(
    backend,
    bf16,
    checkpointing,
    tmp_path,
    *,
    ranks=2,
    shards=(2, 1, 1),
    input_grad=True,
):
    require_compile_device(ranks=ranks, bf16=bf16)
    script = (
        Path(__file__).parent / "helpers" / "rank_scripts" / "compile_spatial_loss.py"
    )
    rc, out, err = run_compile_ranks(
        str(script),
        n=ranks,
        timeout=360,
        env={
            "SCAFFOLD_TEST_COMPILE_BACKEND": backend,
            "SCAFFOLD_TEST_BF16": str(int(bf16)),
            "SCAFFOLD_TEST_CHECKPOINTING": str(int(checkpointing)),
            "SCAFFOLD_TEST_SHARDS": ",".join(map(str, shards)),
            "SCAFFOLD_TEST_INPUT_GRAD": str(int(input_grad)),
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    (tmp_path / "ranks.stdout").write_text(out)
    (tmp_path / "ranks.stderr").write_text(err)
    assert rc == 0, f"Distributed loss compilation failed ({rc})\n{out}\n{err}"
    assert all(f"RANK {rank} DONE" in out for rank in range(ranks))


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("checkpointing", [False, True], ids=["direct", "checkpoint"])
def test_spatial_compiled_loss(backend, bf16, checkpointing, tmp_path):
    _run_loss_case(backend, bf16, checkpointing, tmp_path)


@pytest.mark.parametrize(
    "backend,bf16,checkpointing,shards",
    [
        pytest.param("inductor", True, True, (2, 2, 1), id="two_axes_bf16_checkpoint"),
        pytest.param("inductor", True, True, (2, 1, 1), id="replicas_bf16_checkpoint"),
        pytest.param("aot_eager", False, False, (2, 2, 1), id="two_axes_fp32"),
    ],
)
def test_four_rank_compiled_loss(backend, bf16, checkpointing, shards, tmp_path):
    _run_loss_case(backend, bf16, checkpointing, tmp_path, ranks=4, shards=shards)


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_checkpointed_loss_with_production_images(bf16, tmp_path):
    """Training images do not require gradients, including the first halo."""
    _run_loss_case("inductor", bf16, True, tmp_path, input_grad=False)
