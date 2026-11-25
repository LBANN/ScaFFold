"""Exercise compiled spatial activation checkpointing on two CPU or GPU ranks."""

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


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_native_spatial_checkpointing(backend, bf16, tmp_path):
    require_compile_device(ranks=2, bf16=bf16)
    script = (
        Path(__file__).parent
        / "helpers"
        / "rank_scripts"
        / "compile_spatial_checkpointing_2rank.py"
    )
    rc, out, err = run_compile_ranks(
        str(script),
        n=2,
        timeout=300,
        env={
            "SCAFFOLD_TEST_COMPILE_BACKEND": backend,
            "SCAFFOLD_TEST_BF16": str(int(bf16)),
            "TORCHDYNAMO_ENABLE_P2P_COMPILATION": "0",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    (tmp_path / "ranks.stdout").write_text(out)
    (tmp_path / "ranks.stderr").write_text(err)
    assert rc == 0, f"Spatial checkpointing failed ({rc})\n{out}\n{err}"
    assert "RANK 0 DONE" in out and "RANK 1 DONE" in out, out
