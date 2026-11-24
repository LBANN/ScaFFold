"""Validate GPU-test prerequisites and launching without requiring a GPU."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.helpers import compile_runtime, mpi_runner


@pytest.fixture
def mocked_gpus(monkeypatch):
    """Represent two visible GPUs while recording device selection."""
    monkeypatch.setenv("SCAFFOLD_TEST_DEVICE", "cuda")
    monkeypatch.setenv("SCAFFOLD_TEST_GPU_TIMEOUT", "900")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    state = {"selected": 0, "events": [], "bf16_checks": []}

    def set_device(device):
        state["selected"] = torch.device(device).index
        state["events"].append(("set_device", state["selected"]))

    @contextmanager
    def device_context(index):
        previous = state["selected"]
        state["selected"] = index
        try:
            yield
        finally:
            state["selected"] = previous

    def native_bf16(*, including_emulation):
        assert including_emulation is False
        state["bf16_checks"].append(state["selected"])
        return True

    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", native_bf16)
    monkeypatch.setattr(
        torch.backends.cuda, "matmul", SimpleNamespace(fp32_precision="tf32")
    )
    monkeypatch.setattr(
        torch.backends,
        "cudnn",
        SimpleNamespace(
            conv=SimpleNamespace(fp32_precision="tf32"),
            benchmark=True,
            deterministic=False,
        ),
    )
    return state


@pytest.mark.parametrize("requested", [None, "auto"], ids=["default", "explicit_auto"])
@pytest.mark.parametrize("gpu_available", [False, True], ids=["cpu_host", "gpu_host"])
def test_auto_device_uses_available_hardware(monkeypatch, requested, gpu_available):
    if requested is None:
        monkeypatch.delenv("SCAFFOLD_TEST_DEVICE", raising=False)
    else:
        monkeypatch.setenv("SCAFFOLD_TEST_DEVICE", requested)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu_available)

    assert compile_runtime.compile_device_type() == ("cuda" if gpu_available else "cpu")


def test_invalid_device_lists_supported_choices(monkeypatch):
    monkeypatch.setenv("SCAFFOLD_TEST_DEVICE", "mps")
    with pytest.raises(ValueError, match="auto.*cpu.*cuda"):
        compile_runtime.compile_device_type()


def test_requested_gpu_cannot_silently_use_cpu(mocked_gpus, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        mpi_runner,
        "torchrun_distributed",
        lambda *args, **kwargs: pytest.fail("Launched ranks without a GPU"),
    )
    with pytest.raises(RuntimeError, match="requested.*cannot access a GPU"):
        compile_runtime.run_compile_ranks("unused.py", n=2, timeout=30, env={})


@pytest.mark.parametrize("requested", ["cuda", "auto"])
def test_requested_gpu_requires_nccl(mocked_gpus, monkeypatch, requested):
    monkeypatch.setenv("SCAFFOLD_TEST_DEVICE", requested)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires NCCL"):
        compile_runtime.require_compile_device()


@pytest.mark.parametrize("available,required", [(1, 2), (2, 4)])
@pytest.mark.parametrize("requested", ["cuda", "auto"])
def test_gpu_rank_count_cannot_oversubscribe(
    mocked_gpus, monkeypatch, available, required, requested
):
    monkeypatch.setenv("SCAFFOLD_TEST_DEVICE", requested)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: available)
    with pytest.raises(RuntimeError, match=f"requires {required} visible GPUs"):
        compile_runtime.require_compile_device(ranks=required)


@pytest.mark.parametrize("requested", ["cuda", "auto"])
def test_bf16_checks_every_gpu_without_accepting_emulation(
    mocked_gpus, monkeypatch, requested
):
    monkeypatch.setenv("SCAFFOLD_TEST_DEVICE", requested)
    compile_runtime.require_compile_device(ranks=2, bf16=True)
    assert mocked_gpus["bf16_checks"] == [0, 1]
    assert mocked_gpus["selected"] == 0

    def unsupported_second_gpu(*, including_emulation):
        assert including_emulation is False
        return mocked_gpus["selected"] == 0

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", unsupported_second_gpu)
    with pytest.raises(RuntimeError, match="GPU 1 does not support native BF16"):
        compile_runtime.require_compile_device(ranks=2, bf16=True)
    assert mocked_gpus["selected"] == 0


def test_local_rank_is_bound_before_nccl_initialization(mocked_gpus, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("SCAFFOLD_TEST_BF16", "1")

    def initialize(backend, **kwargs):
        assert mocked_gpus["events"] == [("set_device", 1)]
        assert backend == "nccl"
        assert kwargs["device_id"] == torch.device("cuda", 1)
        assert kwargs["timeout"].total_seconds() == 900
        mocked_gpus["events"].append(("initialized", backend))

    monkeypatch.setattr(torch.distributed, "init_process_group", initialize)
    monkeypatch.setattr(compile_runtime, "report_compile_runtime", lambda device: None)
    assert compile_runtime.init_compile_process_group() == torch.device("cuda", 1)
    assert mocked_gpus["bf16_checks"] == [1]
    assert torch.backends.cuda.matmul.fp32_precision == "ieee"
    assert torch.backends.cudnn.conv.fp32_precision == "ieee"
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_invalid_local_rank_is_rejected_before_device_selection(
    mocked_gpus, monkeypatch
):
    monkeypatch.setenv("LOCAL_RANK", "2")
    with pytest.raises(RuntimeError, match="LOCAL_RANK=2"):
        compile_runtime.configure_compile_device()
    assert mocked_gpus["events"] == []


def test_cpu_runtime_does_not_select_an_accelerator(monkeypatch):
    monkeypatch.setenv("SCAFFOLD_TEST_DEVICE", "cpu")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_gloo_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda *args: pytest.fail("Selected GPU for CPU run")
    )
    initialized = []
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda backend, **kwargs: initialized.append((backend, kwargs)),
    )
    monkeypatch.setattr(compile_runtime, "report_compile_runtime", lambda device: None)
    assert compile_runtime.init_compile_process_group() == torch.device("cpu")
    assert initialized[0][0] == "gloo"
    assert "device_id" not in initialized[0][1]


@pytest.mark.parametrize(
    "requested,gpu_available",
    [
        pytest.param("cuda", True, id="explicit_gpu"),
        pytest.param(None, True, id="default_gpu"),
        pytest.param("auto", True, id="auto_gpu"),
        pytest.param("auto", False, id="auto_cpu"),
    ],
)
def test_rank_launcher_preserves_protocol_and_resolves_device_and_deadline(
    mocked_gpus, monkeypatch, requested, gpu_available
):
    if requested is None:
        monkeypatch.delenv("SCAFFOLD_TEST_DEVICE", raising=False)
    else:
        monkeypatch.setenv("SCAFFOLD_TEST_DEVICE", requested)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu_available)
    monkeypatch.setattr(torch.distributed, "is_gloo_available", lambda: True)
    call = {}

    def launch(script, **kwargs):
        call.update(script=script, **kwargs)
        return 0, "ranks completed", ""

    monkeypatch.setattr(mpi_runner, "torchrun_distributed", launch)
    result = compile_runtime.run_compile_ranks(
        "rank_script.py",
        n=2,
        timeout=300,
        env={"SCAFFOLD_TEST_BF16": "1", "SCAFFOLD_TEST_COMPILE_BACKEND": "inductor"},
    )
    assert result == (0, "ranks completed", "")
    assert call["backend"] == ("nccl" if gpu_available else "gloo")
    assert call["n"] == 2
    assert call["timeout"] == (900 if gpu_available else 300)
    assert call["env"]["SCAFFOLD_TEST_COMPILE_BACKEND"] == "inductor"
    assert call["env"]["SCAFFOLD_TEST_DEVICE"] == ("cuda" if gpu_available else "cpu")
    assert call["env"]["SCAFFOLD_CONV_TRITON"] == "0"
    assert call["env"]["SCAFFOLD_GROUPNORM_TRITON"] == "0"
    assert call["env"]["SCAFFOLD_GROUPNORM_COMPILE"] == "0"
    assert call["env"]["MPI4PY_RC_INITIALIZE"] == "0"


@pytest.mark.parametrize("value", ["0", "-1"])
def test_gpu_timeout_cannot_remove_the_hang_guard(mocked_gpus, monkeypatch, value):
    monkeypatch.setenv("SCAFFOLD_TEST_GPU_TIMEOUT", value)
    with pytest.raises(ValueError, match="must be positive"):
        compile_runtime.compile_timeout(300)


@pytest.mark.skipif(mpi_runner.os.name != "posix", reason="POSIX process-group cleanup")
def test_distributed_timeout_kills_rank_group_and_removes_wrapper(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(mpi_runner, "_free_port", lambda: 29641)
    monkeypatch.setenv("CLAUDE_CODE_TMPDIR", str(tmp_path))
    killed = []
    monkeypatch.setattr(
        mpi_runner.os, "killpg", lambda pid, sig: killed.append((pid, sig))
    )
    launched = {}

    class TimedOutProcess:
        pid = 123456
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise mpi_runner.subprocess.TimeoutExpired("torchrun", timeout)
            return "partial stdout", "rank traceback"

    def popen(argv, **kwargs):
        wrapper = Path(argv[-1])
        assert wrapper.parent == tmp_path
        assert wrapper.exists()
        assert "dump_traceback_later" in wrapper.read_text()
        assert str(tmp_path / "rank.py") in wrapper.read_text()
        launched.update(argv=argv, **kwargs)
        return TimedOutProcess()

    monkeypatch.setattr(mpi_runner.subprocess, "Popen", popen)
    rc, stdout, stderr = mpi_runner.torchrun_distributed(
        str(tmp_path / "rank.py"), n=2, timeout=20, backend="nccl"
    )
    assert rc == -1
    assert stdout == "partial stdout"
    assert "rank traceback" in stderr and "timed out after 20s" in stderr
    assert killed == [(TimedOutProcess.pid, mpi_runner.signal.SIGKILL)]
    assert launched["start_new_session"] is True
    assert launched["argv"][:3] == [mpi_runner.PYTHON, "-m", "torch.distributed.run"]
    assert launched["env"]["PL_TORCH_DISTRIBUTED_BACKEND"] == "nccl"
    assert list(tmp_path.iterdir()) == []
