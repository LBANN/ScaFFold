"""Run the native compilation checks on CPU/Gloo or one GPU per NCCL rank.

``SCAFFOLD_TEST_DEVICE=auto`` (the default) selects a GPU when PyTorch can
access one, otherwise CPU. Set ``cpu`` to force CPU tests or ``cuda`` to
require a GPU, for either NVIDIA CUDA or AMD ROCm. Once GPU execution is
selected, missing prerequisites fail the test rather than switching to CPU.
"""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from tests.helpers import mpi_runner


def compile_device_type():
    """Resolve automatic selection while preserving explicit device overrides."""
    value = os.environ.get("SCAFFOLD_TEST_DEVICE", "auto")
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in ("cpu", "cuda"):
        raise ValueError(
            "SCAFFOLD_TEST_DEVICE must be 'auto', 'cpu', or 'cuda' (also for ROCm)"
        )
    return value


def compile_timeout(seconds):
    """Allow cold GPU compilation time without removing the hang guard."""
    if compile_device_type() == "cpu":
        return seconds
    gpu_timeout = int(os.environ.get("SCAFFOLD_TEST_GPU_TIMEOUT", "1800"))
    if gpu_timeout <= 0:
        raise ValueError("SCAFFOLD_TEST_GPU_TIMEOUT must be positive seconds")
    return max(seconds, gpu_timeout)


def require_compile_device(*, ranks=1, bf16=False):
    """Check all GPUs a local torchrun job will use, before launching ranks."""
    device_type = compile_device_type()
    if ranks < 1:
        raise ValueError("At least one rank is required")
    if not dist.is_available():
        raise RuntimeError("Compilation validation requires torch.distributed")
    if device_type == "cpu":
        if not dist.is_gloo_available():
            raise RuntimeError("CPU compilation validation requires Gloo")
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SCAFFOLD_TEST_DEVICE=cuda was requested, but PyTorch cannot access a GPU"
        )
    count = torch.cuda.device_count()
    if count < ranks:
        raise RuntimeError(
            f"This case requires {ranks} visible GPUs, but PyTorch sees {count}. "
            "Allocate enough GPUs or select the one-/two-GPU cases."
        )
    if not dist.is_nccl_available():
        raise RuntimeError("GPU compilation validation requires NCCL (RCCL on ROCm)")
    if bf16:
        for index in range(ranks):
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise RuntimeError(f"GPU {index} does not support native BF16")


def configure_compile_device():
    """Bind this rank before NCCL and use full FP32 for numerical comparisons."""
    require_compile_device()
    if compile_device_type() == "cpu":
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} does not identify a visible GPU")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # FP32 tests must not compare TF32 convolutions against full-precision
    # compiled operations. These are validation settings, not production policy.
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return device


def report_compile_runtime(device):
    """Include the actual stack and GPU selection in saved validation logs."""
    import distconv

    print(
        json.dumps(
            {
                "compile_validation": {
                    "torch": torch.__version__,
                    "distconv": distconv.__file__,
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None,
                    "cuda": torch.version.cuda,
                    "hip": torch.version.hip,
                    "rank": int(os.environ.get("RANK", "0")),
                    "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                    "distributed_backend": "nccl" if device.type == "cuda" else "gloo",
                    "custom_triton": False,
                    "fp32_precision": "ieee" if device.type == "cuda" else "default",
                }
            }
        ),
        flush=True,
    )


def init_compile_process_group():
    """Initialize a torchrun child with its device selected before communication."""
    device = configure_compile_device()
    require_compile_device(
        ranks=int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
    )
    if (
        device.type == "cuda"
        and os.environ.get("SCAFFOLD_TEST_BF16", "0") == "1"
        and not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        raise RuntimeError(f"{device} does not support native BF16")
    kwargs = {"device_id": device} if device.type == "cuda" else {}
    dist.init_process_group(
        "nccl" if device.type == "cuda" else "gloo",
        timeout=timedelta(seconds=compile_timeout(180)),
        **kwargs,
    )
    report_compile_runtime(device)
    return device


def run_compile_ranks(script, *, n, timeout, env):
    """Launch one local rank per GPU, preserving the existing CPU test route."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "Run one pytest process; spatial tests launch their own ranks"
        )
    require_compile_device(ranks=n, bf16=env.get("SCAFFOLD_TEST_BF16") == "1")
    device_type = compile_device_type()
    child_env = dict(env)
    child_env.update(
        SCAFFOLD_TEST_DEVICE=device_type,
        SCAFFOLD_CONV_TRITON="0",
        SCAFFOLD_GROUPNORM_TRITON="0",
        SCAFFOLD_GROUPNORM_COMPILE="0",
        # Compilation imports worker/datagen but these rank scripts never use
        # MPI. Do not let torchrun children join their parent's scheduler PMI job.
        MPI4PY_RC_INITIALIZE="0",
    )
    return mpi_runner.torchrun_distributed(
        script,
        n=n,
        timeout=compile_timeout(timeout),
        env=child_env,
        backend="nccl" if device_type == "cuda" else "gloo",
    )
