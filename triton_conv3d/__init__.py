# SPDX-License-Identifier: (Apache-2.0)
"""Triton 3-D convolution kernels for NDHWC (``channels_last_3d``) tensors.

This package is deliberately **self-contained**: it imports nothing from
ScaFFold or DistConv, so it can be vendored into either (or released on its own)
unchanged.  ScaFFold plugs in through a thin adapter that lives on the ScaFFold
side.

What exists today:

- :mod:`triton_conv3d.gather_gemm` -- the forward implicit-GEMM convolution,
  ``k>=1`` with ``stride=1``, bf16 / fp16 / fp32.
- :mod:`triton_conv3d.bwd_data` -- the gradient with respect to the input.  It
  contains no kernel of its own: at ``stride=1`` backward-data *is* the forward
  contraction on a flipped, channel-transposed weight.
- :mod:`triton_conv3d.reduce_gemm` -- the gradient with respect to the weight,
  which is the one direction that needs a kernel of its own: a tiny output
  reduced over the whole volume, so split-K is mandatory rather than optional.
  It is also where reproducibility is decided, and its deterministic path is the
  default.
- :mod:`triton_conv3d.transposed` -- ``ConvTranspose3d`` at ``kernel == stride``
  and no padding, all three directions.  Only its *forward* is a kernel: with
  the windows tiling rather than overlapping, both backward directions are the
  ordinary strided convolution seen from the other side, and the two modules
  above serve them unchanged.
- :mod:`triton_conv3d.shapes` -- the convolution problems that actually occur,
  extracted from real ScaFFold runs, plus synthetic edge cases.
- :mod:`triton_conv3d.reference` -- reference implementations and the tolerance
  policy used to decide whether a kernel is correct.
- :mod:`triton_conv3d.bench` -- interleaved A/B timing, MIOpen baseline capture,
  the ``tl.dot`` ceiling probe, and the forward benchmark.

The public entry point takes and returns ``channels_last_3d`` tensors; the
autograd registration and the ScaFFold adapter live in a later milestone, so a
caller today drives :func:`conv3d_forward` directly and checks a gate first.

**The gates say nothing about the GPU, deliberately.**  They are *capability*
predicates -- "will this call be computed correctly here" -- and the answer to
that does not depend on which AMD part is running: the kernels compute the right
convolution wherever Triton lowers them.  What *is* device-specific is every
number that decides how they launch (the tile tables, ``matrix_instr_nonkdim``,
and the ``GROUP_M`` default of 6, which is MI300A's XCD count), all of which was
raced on one MI300A -- and a launch configuration that is merely wrong for the
hardware raises nothing: see :mod:`triton_conv3d.gather_gemm`'s "Configuration
constraints are hard".  Deciding whether *this* machine is one whose numbers are
trustworthy is therefore the embedder's routing question, not this package's
capability question, and putting a device allowlist inside the gates would lock
out a consumer who has retuned for their own part.  ScaFFold makes that decision
in ``ScaFFold/unet/_rungs.py`` (``_platform_declines``); a consumer that has not
retuned should do the same thing there.

**Which gate depends on what the caller will do with the answer.**  The three
directions do not accept the same problems -- ``stride > 1`` is served by the
forward and by backward-weight and refused by backward-data -- so a caller that
will differentiate the result must ask :func:`is_supported_all`, which is all
three at once.  :func:`is_supported` alone gates the forward alone, which is
what an inference caller wants and a training caller must not settle for: a
forward this package serves and a backward it cannot is discovered inside
``backward()``, where the caller's fallback kernel is no longer reachable.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.0.0.dev0"

__all__ = ["ConvConfig", "conv3d_forward", "conv3d_backward_data",
           "conv3d_backward_weight", "is_supported", "is_supported_all",
           "is_supported_bwd_data", "is_supported_bwd_weight",
           "conv_transpose3d_forward", "conv_transpose3d_backward_data",
           "conv_transpose3d_backward_weight", "is_supported_transposed",
           "is_supported_transposed_all", "is_supported_transposed_bwd_data",
           "is_supported_transposed_bwd_weight", "__version__"]

#: The public names live in :mod:`triton_conv3d.gather_gemm`, which imports
#: torch and triton.  They are re-exported lazily so that
#: ``import triton_conv3d.shapes`` stays free of both: the shape and cost model
#: is pure Python by design, and it drives test parametrization at collection
#: time on machines that have no GPU and may have no triton.
_LAZY = {"ConvConfig": "gather_gemm", "conv3d_forward": "gather_gemm",
         "is_supported": "gather_gemm", "is_supported_all": "gather_gemm",
         "conv3d_backward_data": "bwd_data",
         "is_supported_bwd_data": "bwd_data",
         "conv3d_backward_weight": "reduce_gemm",
         "is_supported_bwd_weight": "reduce_gemm",
         "conv_transpose3d_forward": "transposed",
         "conv_transpose3d_backward_data": "transposed",
         "conv_transpose3d_backward_weight": "transposed",
         "is_supported_transposed": "transposed",
         "is_supported_transposed_all": "transposed",
         "is_supported_transposed_bwd_data": "transposed",
         "is_supported_transposed_bwd_weight": "transposed"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(
            importlib.import_module(f".{_LAZY[name]}", __name__), name
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
