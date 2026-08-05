# SPDX-License-Identifier: (Apache-2.0)
"""Backward-data: the forward kernel, run on a transformed weight.

At ``stride == 1`` the gradient with respect to the input is the *same*
contraction as the forward pass.  Starting from the forward,

    y[n, oc, o] = sum_{ic, t} x[n, ic, o*1 - p + t*dil] * w[oc, ic, t]

each ``x`` voxel contributes to every ``y`` voxel whose window covers it, so

    gx[n, ic, i] = sum_{oc, t} gy[n, oc, i + p - t*dil] * w[oc, ic, t]

which is a gather with a *negative* tap stride.  Substituting ``t' = k-1-t``
flips it back::

    i + p - t*dil = i - (dil*(k-1) - p) + t'*dil

and that is exactly the forward gather with padding ``dil*(k-1) - p``.  So

    grad_input = conv3d_forward(grad_output, flip_taps(swap_channels(w)),
                                padding = dil*(k-1) - padding)

with the output spatial extent working out to the input's on its own:
``OD + 2p' - dil*(k-1) == ID`` identically, for every ``p`` and ``dil``.

Consequences, all of which the code below leans on:

* **There is no ``@triton.jit`` in this module, deliberately.**
  ``grep -c '^@triton\.jit' triton_conv3d/*.py`` reporting 1 for
  ``gather_gemm.py`` and 0 for everything else is the claim this file is making
  -- anchored at column 0 so that this very sentence does not satisfy the grep,
  which the first version of it did.  It is also why the tests here can reuse
  the forward's correctness standards unchanged.
* **There is no weight transform either, and there used to be.**  ``to_bwd_rsck``
  materialized ``(kd, kh, kw, Cout, Cin)`` with the taps flipped, once per layer
  per optimizer step -- 0.531 ms/step over one configuration's 19 Conv3d sites,
  which no caching removes because the optimizer dirties every parameter every
  step.  Both halves of it are now free, and neither cost what it looked like it
  would.  The flip is a constexpr index (``taps - 1 - dij``: flipping all three
  kernel axes is the complement of a mixed-radix index).  The transpose is *not
  performed at all* -- the kernel addresses the weight through its strides, and
  a ``permute`` supplies those, so "transposed" is a matter of which stride is
  which.  A ``channels_last_3d`` parameter is ``[Cout][tap][Cin]``, and this
  direction's N is ``Cin``, so the parameter is read here with the *same*
  contiguous-N tile the forward gets from a materialized RSCK buffer.  Measured
  against the ``to_bwd_rsck`` path it replaced, on the eight hottest sites of
  config A: 0.98-1.02x, i.e. free.
* The tuning does **not** transfer from the forward.  The effective GEMM has
  the channel widths swapped: forward ``128 -> 64`` is ``N=64, K=128*27``,
  and its backward-data is ``N=128, K=64*27``.  Same kernel, different corner
  of the tuning surface, so :data:`_TUNED_BWD` is its own table.
* ``PADDED`` is always true for ``k > 1``, whatever the forward's padding was.
  The equivalent forward has ``p' = d*(k-1) - p``, which is 2 for an unpadded
  ``k = 3`` and 1 for a "same"-padded one -- both non-zero, so the six-compare
  boundary predicate is unavoidable here even in the one case the forward
  compiles it away.  That is a property of the mathematics, not of the reuse,
  and it is why this direction is the one place the halo'd/padded distinction
  does *not* change which kernel body is compiled.

Restrictions beyond the forward's
=================================

``stride > 1`` is refused.  The substitution above needs ``o = i + p - t*dil``
to have a solution for every ``i``, which at ``stride s`` it has only when
``s`` divides ``i + p - t*dil``; the backward is then a *dilated scatter* into
a strided sub-lattice, which the forward kernel's addressing cannot express.
``padding > dil*(k-1)`` is refused for the mirror-image reason: ``p'`` would be
negative, i.e. the backward would have to *crop*, which is again not something
the forward gather does.  Neither occurs in ScaFFold's corpus.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .gather_gemm import (
    ConvConfig,
    _MFMA_KDIM,
    _check_weight_rsck,
    _triple,
    conv3d_forward,
    select_config,
    tune_key,
)

__all__ = [
    "conv3d_backward_data",
    "is_supported_bwd_data",
    "bwd_data_padding",
]


def bwd_data_padding(padding, dilation, kernel) -> tuple[int, int, int]:
    """The forward padding that reproduces backward-data: ``dil*(k-1) - p``."""
    p = _triple(padding, "padding")
    d = _triple(dilation, "dilation")
    k = _triple(kernel, "kernel")
    return tuple(d[i] * (k[i] - 1) - p[i] for i in range(3))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def _tuned(bm: int, bn: int, bk: int, warps: int, group_m: int = 6) -> ConvConfig:
    return ConvConfig(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group_m,
                      num_warps=warps, num_stages=2, matrix_instr_nonkdim=16,
                      kpack=1 if bk <= 16 else 2)


#: Measured backward-data winners, keyed by the **forward** problem's
#: ``(dtype, Cin, Cout, kernel)`` -- i.e. by the convolution a reader would name
#: -- even though the GEMM that runs has those two widths swapped.  Keying it
#: the other way round would make ``512 -> 256`` in this table and ``512 -> 256``
#: in the forward's mean different things, which is a trap not worth setting.
#:
#: Drawn from a backward-data sweep over 21 problems and 1468 timed
#: configurations.  Only channel pairs that were actually timed appear; a miss
#: falls to the heuristic on the effective widths, which is a real gap and not
#: an extrapolation dressed up as a measurement.
#:
#: Three things this table records that the forward's does *not*:
#:
#: * ``BLOCK_N`` runs to **256**.  The GEMM's N is ``Cin``, so the decoder
#:   convolutions whose forward is skinny (``Cout=64``) are the widest ones
#:   here, and the tile follows.
#: * ``BLOCK_K`` of **32** wins twice.  The reduction is ``Cout * 27``, which for
#:   ``Cout=64`` is only 1728, and a deep K-tile then wastes the tail.
#: ``GROUP_M`` and ``matrix_instr_nonkdim`` are pinned at 6 and 16, as in the
#: forward, and both were re-checked here rather than inherited.  ``nonkdim=32``
#: won exactly one pair (``64 -> 64``) and forcing 16 there costs **0.3%**.
#: ``GROUP_M=8`` won 8 of 21 problems with a geometric mean of 1.006x, and that
#: count is biased in its favour -- the sweep only tries 8 on each problem's
#: finalists -- so 6 (MI300A's XCD count) is used throughout.  Selecting each
#: entry from the ``g6``/``nk16`` arm alone, which is the arm every configuration
#: was timed in, costs a geometric mean of **1.9%** against the per-problem best.
_TUNED_BWD: dict[tuple, ConvConfig] = {
    **{
        tune_key(torch.bfloat16, cin, cout, (3, 3, 3)): cfg
        for (cin, cout), cfg in {
            (64, 64): _tuned(256, 64, 32, 4),
            (64, 128): _tuned(128, 64, 64, 4),
            (128, 64): _tuned(128, 128, 32, 4),
            (128, 128): _tuned(128, 128, 64, 4),
            (128, 256): _tuned(128, 128, 64, 4),
            (256, 128): _tuned(128, 256, 64, 8),
            (256, 256): _tuned(128, 256, 64, 8),
            (256, 512): _tuned(128, 256, 64, 8),
            (512, 256): _tuned(128, 256, 64, 8),
            (512, 512): _tuned(128, 128, 128, 8),
            (512, 1024): _tuned(64, 64, 128, 4),
            (1024, 512): _tuned(128, 256, 64, 8),
            (1024, 1024): _tuned(128, 128, 128, 8),
        }.items()
    },
    # The segmentation head.  ``k=1`` means the backward has no gather at all
    # (``p' = 0``) and a reduction of just ``Cout = 6``, so it is a different
    # regime from every entry above and gets its own key.
    #
    # ``num_warps = 2``, not the 4 the original sweep shipped, and that is the
    # only axis of this entry that moved: that sweep drew ``num_warps`` from
    # ``{4, 8, seed}``, so 1 and 2 were never timed at any site in this
    # project.  Raced at all three head volumes, 2 is 1.099x, 1.057x and 1.101x
    # over 4, and 8 is 0.89-0.90x.  The reduction here is
    # ``Cout * taps = 6``, one MFMA fragment deep, so a second pair of waves has
    # nothing to reduce and only replicates the addressing.  1 warp is within
    # noise of 2 at this site and is 0.245x at ``128 -> 128 @ 66^3``, so the
    # narrow regime is where this stops, not a direction-wide rule.
    tune_key(torch.bfloat16, 64, 6, (1, 1, 1)): _tuned(256, 64, 16, 2),
}


def register_tuned_bwd_data(dtype, cin, cout, kernel, config: ConvConfig) -> None:
    _TUNED_BWD[tune_key(dtype, cin, cout, kernel)] = config


def bwd_data_config(
    grad_output_shape: Sequence[int], cin: int, kernel: Sequence[int],
    dtype: torch.dtype = torch.bfloat16, *, padding=0, dilation=1,
) -> ConvConfig:
    """The config :func:`conv3d_backward_data` would pick for this problem.

    Exposed because the benchmark and the ISA gate both need to know what the
    shipped path chooses without having to reconstruct the effective GEMM.
    """
    n, cout, *out_sp = (int(v) for v in grad_output_shape)
    k = _triple(kernel, "kernel")
    d = _triple(dilation, "dilation")
    p = _triple(padding, "padding")
    in_sp = [o + 2 * (d[i] * (k[i] - 1) - p[i]) - d[i] * (k[i] - 1)
             for i, o in enumerate(out_sp)]
    m = n * math.prod(in_sp)
    return select_config(
        m, cout, cin, k, dtype,
        table=_TUNED_BWD, key=tune_key(dtype, cin, cout, k),
    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def is_supported_bwd_data(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    input_shape: Sequence[int],
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> bool:
    """Whether :func:`conv3d_backward_data` will serve this call.

    Same asymmetry as :func:`~triton_conv3d.gather_gemm.is_supported`: the
    caller's fallback is MIOpen, which is correct everywhere, so a false
    negative costs a little speed and a false positive returns a wrong gradient.

    The two checks that are *not* in the forward's predicate are the two the
    module docstring derives: ``stride == 1``, and ``padding <= dil*(k-1)``.
    They make this the narrowest of the three gates -- the forward and
    backward-weight both serve a stride this one refuses -- so a caller that
    will differentiate must ask
    :func:`~triton_conv3d.gather_gemm.is_supported_all` rather than assume the
    forward's ``True`` covers this direction.
    """
    if groups != 1:
        return False
    if grad_output.dim() != 5 or weight.dim() != 5 or len(tuple(input_shape)) != 5:
        return False
    if grad_output.dtype != weight.dtype or grad_output.dtype not in _MFMA_KDIM:
        return False
    # Same device, not merely both on *a* device: Triton launches on the current
    # one and dereferences the foreign pointer regardless, and on a node with
    # peer access enabled -- which is how ScaFFold runs its four GPUs -- that
    # reads another rank's memory instead of faulting.  A wrong gradient, not a
    # crash.
    if (not grad_output.is_cuda or not weight.is_cuda
            or weight.device != grad_output.device):
        return False
    try:
        s = _triple(stride, "stride")
        p = _triple(padding, "padding")
        d = _triple(dilation, "dilation")
    except ValueError:
        return False
    k = tuple(int(v) for v in weight.shape[2:])
    if any(v < 1 for v in d) or any(v < 0 for v in p):
        return False
    if s != (1, 1, 1):
        return False
    if any(p[i] > d[i] * (k[i] - 1) for i in range(3)):
        return False
    n, cin, *in_sp = (int(v) for v in input_shape)
    if int(weight.shape[0]) != int(grad_output.shape[1]):
        return False
    if int(weight.shape[1]) != cin:
        return False
    if int(grad_output.shape[0]) != n:
        return False
    # The gradient's own shape has to be the one this ``grad_output`` came from,
    # or the caller has mixed up two problems and the kernel would happily write
    # a differently-shaped answer into a buffer sized for the other one.
    for i in range(3):
        if int(grad_output.shape[2 + i]) != in_sp[i] + 2 * p[i] - d[i] * (k[i] - 1):
            return False
    # ``n`` alongside the spatial extents, for the same reason: this predicate's
    # own "every output voxel must exist" argument excludes an empty batch, and
    # a gate that answers ``True`` for a problem with no voxels in it is stating
    # something it has not checked.  Costs a fallback to MIOpen on a call that
    # has nothing to compute.
    if n < 1 or any(v < 1 for v in in_sp):
        return False
    return True


def conv3d_backward_data(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    input_shape: Sequence[int],
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    *,
    config: ConvConfig | None = None,
    weight_rsck: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gradient of a 3-D convolution with respect to its input.

    ``grad_output`` and the returned gradient are ``channels_last_3d``.
    ``input_shape`` is PyTorch's ``(N, Cin, D, H, W)``; it is redundant at
    ``stride == 1`` (the derivation recovers it exactly) and is required anyway,
    both to match ``torch.nn.grad.conv3d_input``'s signature and because a
    mismatch is the cheapest available check that the caller has not paired a
    ``grad_output`` with the wrong problem.

    ``weight_rsck`` is the **forward's** RSCK buffer, ``(kd, kh, kw, Cin,
    Cout)`` -- the same tensor
    :func:`~triton_conv3d.gather_gemm.conv3d_forward` takes, not a second one
    transformed for this direction.  It is optional and, on a
    ``channels_last_3d`` parameter, pointless: pass the parameter as ``weight``
    and the kernel reads it in place.
    """
    if not is_supported_bwd_data(grad_output, weight, input_shape, stride,
                                 padding, dilation, groups):
        raise NotImplementedError(
            f"unsupported: grad_output={tuple(grad_output.shape)}/"
            f"{grad_output.dtype} w={tuple(weight.shape)} "
            f"input_shape={tuple(input_shape)} stride={stride} "
            f"padding={padding} dilation={dilation} groups={groups}"
        )
    k = tuple(int(v) for v in weight.shape[2:])
    pad = bwd_data_padding(padding, dilation, k)

    n, cin, *in_sp = (int(v) for v in input_shape)
    cout = int(grad_output.shape[1])
    if config is None:
        config = select_config(
            n * math.prod(in_sp), cout, cin, k, grad_output.dtype,
            table=_TUNED_BWD,
            key=tune_key(grad_output.dtype, cin, cout, k),
        )

    # A *view* in both branches, never a copy.  The effective convolution's
    # channel widths are the real one's swapped, and ``permute`` is exactly that
    # relabelling: the forward's :func:`~triton_conv3d.gather_gemm._weight_plan`
    # reads the resulting strides and picks the load orientation off them, so
    # both of these are addressed in place.
    #
    # * the parameter itself becomes ``(Cin, Cout, kd, kh, kw)``.  Channels-last
    #   makes its ``Cin`` contiguous, which is this GEMM's N -- the same
    #   ``W_ORDER == 0`` load the forward has always used, at a different stride.
    # * the forward's RSCK buffer becomes the same shape with ``Cout``
    #   contiguous, i.e. this GEMM's K, so its N is strided and it takes the
    #   general load.  Tap ``t`` of this direction is tap ``flip(t)`` of the
    #   weight and its matrix is the transpose; both are addressing, and neither
    #   is a copy or a register shuffle.
    #
    # ``out=`` is still validated by the forward rather than here, and that is
    # exact rather than approximate: the effective forward's output shape
    # ``(n, Cin_eff, out_d, out_h, out_w)`` *is* ``input_shape``.
    if weight_rsck is None:
        w_view = weight.permute(1, 0, 2, 3, 4)
    else:
        # Checked here rather than by the forward: what the forward would check
        # it against is the *effective* problem's RSCK shape, and this is the
        # real problem's, with the two channel widths the other way round.
        _check_weight_rsck(weight_rsck, (*k, cin, cout), grad_output)
        w_view = weight_rsck.permute(3, 4, 0, 1, 2)
    return conv3d_forward(
        grad_output, w_view, None, 1, pad, dilation, 1,
        config=config, weight_flip=True, out=out,
    )
