"""Fixed numerical criteria shared by CPU and GPU compilation validation."""

import torch

BF16_RELATIVE_L2_LIMIT = 0.03


def tensor_metrics(actual, expected):
    """Measure error in FP64, retaining reference norms and finiteness."""
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    actual = actual.detach().to(torch.float64)
    expected = expected.detach().to(torch.float64)
    difference = actual - expected
    difference_l2 = torch.linalg.vector_norm(difference).item()
    reference_l2 = torch.linalg.vector_norm(expected).item()
    relative_l2 = (
        difference_l2 / reference_l2
        if reference_l2
        else (0.0 if difference_l2 == 0.0 else float("inf"))
    )
    return {
        "max_absolute": difference.abs().max().item(),
        "relative_l2": relative_l2,
        "reference_l2": reference_l2,
        "actual_finite": bool(torch.isfinite(actual).all()),
        "reference_finite": bool(torch.isfinite(expected).all()),
    }


def assert_bf16_close(actual, expected):
    """Require finite values and at most 3% relative L2; zero means exact zero.

    The threshold is fixed before measurements. This criterion applies to the
    FP32 gradients of BF16-autocast computations as well as their BF16 outputs.
    """
    if actual.dtype != expected.dtype:
        raise AssertionError(f"Dtype mismatch: {actual.dtype} != {expected.dtype}")
    metrics = tensor_metrics(actual, expected)
    if not (metrics["actual_finite"] and metrics["reference_finite"]):
        raise AssertionError(f"Nonfinite values: {metrics}")
    if metrics["reference_l2"] == 0.0:
        if metrics["max_absolute"] != 0.0:
            raise AssertionError(f"A zero reference requires exact zero: {metrics}")
    elif metrics["relative_l2"] > BF16_RELATIVE_L2_LIMIT:
        raise AssertionError(f"Relative L2 exceeds {BF16_RELATIVE_L2_LIMIT}: {metrics}")
    return metrics
