# SPDX-License-Identifier: (Apache-2.0)
"""Triton against MIOpen, per operator, per direction, on the shapes ScaFFold runs.

**Two operators, three directions, one driver.**  ``--operator`` selects the
convolution or the ``k == s`` transposed convolution; ``--direction`` selects
which of its gradients (or none).  ``--operator all --direction all`` measures
every cell in the project under one methodology, in one process, from one
command.

The transposed benchmarks used to live in a separate driver
(``work/triton-conv/bin/m5_convT_bench.py``, deleted at this commit), on the
argument that ``--direction`` selects between three *directions of one operator*
while the transposed convolution is a *different* operator, and that folding it
in would make ``_build`` branch on ``problem.transposed`` in every arm to run the
same code.  The argument was right about the failure and wrong about the
factoring.  Operator and direction are **two axes of one table**, not two values
on one switch: :data:`_OPERATORS` maps ``(operator, direction)`` to a builder,
and each of the six builders is a separate function that names its own operands,
its own control, its own candidate configs and its own reference.  ``_build`` is
a lookup, so nothing branches on ``problem.transposed`` anywhere, and the four
things that *are* per-operator rather than per-direction -- the shape form, the
problem ordering, the config type, whether a direction is sweepable -- sit on
:class:`_Op` where a reader can see all four at once.

What that bought, immediately: ``m5_convT_bench``'s transposed backward
directions dropped the ``config=`` argument on the floor, because the only config
object it had was the transposed *forward*'s, and passing that to a direction
served by ``conv3d_forward`` silently benchmarks a tile nothing would select.
With a builder per cell each one names the config type its own entry point
resolves, and both transposed backward directions became sweepable for free.

Five things this driver is careful about, each because getting it wrong has
already produced a wrong answer once in this project:

**The shape.**  One ScaFFold convolution reaches a kernel in three different
shapes, and they are three different tuning problems -- MIOpen keys its find
database on the padding, ``bwd_data_config`` derives ``M`` from it, and the
kernel compiles a different ``PADDED`` body either way.  (``bwd_weight_config``
also used to change its answer on it; that clause went on 2026-08-05, and the
forms are still three different measurements without it.)  ``--form`` chooses
which one a run measures, and every row records it:

* ``distconv`` (the default, and every capture on disk): the halo'd, unpadded
  form upstream DistConv hands the backend, ``130^3`` at ``padding = 0``.  It is
  what the profiled MIOpen baseline in ``ConvProblem.measured`` is a timing
  *of*, so it is the right form for a like-for-like MIOpen comparison.
* ``adapter``: what ``ScaFFold/unet/conv3d.py`` hands the Triton kernels, which
  is **the form production actually runs** -- a halo on the genuinely split axis
  only, so ``128^3`` at ``padding = (1,1,1)`` unsharded and
  ``130x256x256`` at ``(0,1,1)`` at two shards.  Padded at every configuration.
* ``logical``: the module's own statement, unhalo'd and padded.  Identical to
  ``adapter`` wherever nothing is split.

Defaulting to ``distconv`` keeps every stored capture comparable; it does *not*
mean it is the form to quote a Triton speedup in.  A ``conv`` cell applies the
chosen form.  **No ``convT`` cell ever differs**, and that is not an oversight:
DistConv's halo is ``k // 2`` only for an odd kernel and 0 at ``k = 2``, and the
adapter exchanges nothing there either, so a transposed site is issued in
exactly the shape the corpus records under all three names.  The choice is a
field on :class:`_Op` (``form``) rather than a line inside a builder.

**The comparison.**  Never sequential.  Both implementations go into one
:func:`interleaved` call so that a neighbour arriving on the device hits both
arms at once and lands in the reported interval instead of in the conclusion.
``cudnn.benchmark`` is on, because with it off MIOpen answers from a heuristic
rather than searching and reports 5-12x worse for the *same* solver -- which
would fabricate a speedup.

**And the comparison is what a capture costs.**  ``--control none`` drops the
MIOpen arm and measures the Triton kernels alone.  It is not a corner-cutting
option, it is where essentially all of the wall clock is: ``cudnn.benchmark =
True`` puts MIOpen on the Find path, whose disk record cannot be replayed in a
fresh process (``review/MIOPEN_CACHE.md`` §2), so **every** cell pays a find --
measured on this node at 92-174 s per cell against 0.3-1.2 s for the Triton
compile, the graph capture, the calibration and the timed rounds put together
(``review/HARNESS_SPEED.md`` §1).  A Triton-only row therefore carries no
``miopen_*`` and no ``speedup`` key at all -- an absent measurement stays
absent -- and ``--check``, whose reference *is* MIOpen's answer, is refused with
it.

**The control.**  The MIOpen side of a backward direction is a real forward
graph plus :func:`torch.autograd.grad`, in **all four** backward cells.  Never
``torch.nn.grad.conv3d_input`` / ``conv3d_weight``: those have no real operand to
pass for the tensor being differentiated, so they fabricate a zero-strided
placeholder, and ``convolution_backward`` picks its solver from that operand's
layout.  At the ``k=1x1x1`` head that made MIOpen decline its own NDHWC path and
run **3.2x** slower than the same call inside a real backward, which is where a
published 4.51x came from against a true 1.39x.

**The timed region.**  See :func:`_timed_region`.  The published per-shape number
is **kernel time**: Python-side dispatch, shape re-validation, tuned-table lookup
and the launcher itself are outside it, for *both* arms, because both arms are
replayed from a CUDA graph.  ``--launcher include`` gives the other number.

**The precision.**  Every speedup is a paired per-round ratio with a 95%
interval, and every cell says how many rounds it took and whether it stopped
because it converged or because it ran out of budget.  ``--iters``/``--rounds``
default to 0, i.e. decided online; pass integers to pin them.

Usage::

    # everything, one command
    python -m triton_conv3d.bench.conv_bench --operator all --direction all \\
        --top 0 --shipped --out all.json

    # a Triton-only baseline over the form production runs: no control, so no
    # find, so minutes rather than hours
    python -m triton_conv3d.bench.conv_bench --operator all --direction all \\
        --top 0 --shipped --control none --form adapter --out triton.json

    python -m triton_conv3d.bench.conv_bench --top 8 --out m1.json
    python -m triton_conv3d.bench.conv_bench --direction bwd-data --problems 1,3,5
    python -m triton_conv3d.bench.conv_bench --operator convT --direction bwd-weight \\
        --top 0 --shipped
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import sys
import time
from typing import Callable, Literal, Mapping

import torch
import torch.nn.functional as F

from ..bwd_data import bwd_data_config, conv3d_backward_data
from ..gather_gemm import (
    ConvConfig,
    candidate_configs,
    conv3d_forward,
    select_config,
)
from ..reduce_gemm import (
    candidate_bwd_weight_configs,
    bwd_weight_config,
    conv3d_backward_weight,
    grad_weight_empty,
    split_count,
    workspace_elements,
)
from ..shapes import (
    DIRECTIONS,
    ConvProblem,
    Direction,
    census_corpus,
    scaffold_corpus,
)
from ..transposed import (
    candidate_transposed_configs,
    conv_transpose3d_backward_data,
    conv_transpose3d_backward_weight,
    conv_transpose3d_forward,
    grad_transposed_weight_empty,
    transposed_config,
)
from .harness import (
    CaptureError,
    capture,
    capture_stream,
    common_chunk,
    format_table,
    graph_is_worthwhile,
    interleaved,
    on_capture_stream,
    per_call_ms,
    ratio,
)

#: See the module docstring.  Set at import so that merely importing this module
#: puts the process in the configuration the numbers were taken in.
torch.backends.cudnn.benchmark = True

_TORCH_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}

Operator = Literal["conv", "convT"]
OPERATORS: tuple[Operator, ...] = ("conv", "convT")

#: How many of the sweep's fastest configs go into the run-off against MIOpen.
#: One would do if the sweep were noise-free; it is not, so its winner is partly
#: whichever config drew the luckiest sample.  Racing the top few restores a
#: like-for-like best-of on both sides.
_FINALISTS = 3

#: Split counts the backward-weight refinement pass pins.  Wide, because the
#: pass is free: the split count is a runtime argument, so none of these
#: triggers a recompile.
_SPLIT_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


# ---------------------------------------------------------------------------
# One cell of the operator x direction table
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Case:
    """Everything that differs between one operator-direction and another.

    The tensors are held here rather than in locals so that the caller's
    ``finally`` can drop them all at once; ``triton`` is a factory rather than a
    launcher because the sweep needs one launcher per config.

    Six of these exist -- two operators by three directions -- and no field is
    ever computed by asking the *problem* which operator it is.  That is the
    whole point of the factoring: the branch happens once, in :data:`_OPERATORS`,
    and never again.
    """

    triton: Callable[[ConvConfig | None], Callable[[], object]]
    #: The MIOpen control, or ``None`` when the case was built with
    #: ``control=False``.  Optional because building it is not free and is not
    #: always wanted: for a *backward* direction the control is a real forward
    #: graph, and running it once costs MIOpen's find -- measured at 92-174 s
    #: per cell on this node's corpus, against 0.3-1.2 s for everything else the
    #: cell does (``work/triton-conv/review/HARNESS_SPEED.md`` §1).  A
    #: Triton-only capture that still built the control would pay all of it.
    miopen: Callable[[], object] | None
    #: ``Callable[[], object]`` -- the hoistable weight prep, or ``None`` where
    #: the direction has none.  **All six cells are now ``None``**: the
    #: consuming directions read the channels-last parameter in place, and the
    #: weight-gradient directions *produce* the weight, in the layout the GEMM
    #: writes natively.  The field stays because the reporting path is the
    #: record of what a transform would have to be charged if one came back.
    transform: object
    #: ``Callable[[], list[ConvConfig]]`` -- the configs worth timing.
    candidates: Callable[[], list[ConvConfig]]
    #: ``Callable[[list[ConvConfig]], list[ConvConfig]]`` -- a second, cheap
    #: pass over the finalists on an axis the first pass held fixed.
    refine: Callable[[list[ConvConfig]], list[ConvConfig]]
    #: The config **this cell's entry point would resolve on its own**, computed
    #: once so that ``--shipped`` measures the shipped kernel without also
    #: measuring the shipped table lookup.  The two are not the same number: the
    #: lookup is 0.0164 ms, 39% of the transposed forward kernel
    #: (``HARNESS_RIGOR.md`` H13).  ``test_the_shipped_config_is_what_the_entry
    #: _point_resolves`` pins these six against the entry points.
    shipped_config: Callable[[], ConvConfig | None]
    #: ``Callable[[], tuple[Tensor, Tensor]]`` -- ``(ours, MIOpen's)`` on this
    #: cell's shape, for ``--check``.  ``None`` without a control, because
    #: MIOpen's answer *is* the reference.
    reference: Callable[[], tuple[torch.Tensor, torch.Tensor]] | None
    #: The operand whose storage decides buffer-op eligibility: the gathered one.
    primary: torch.Tensor
    keep: tuple


def _randn(shape, device, dtype):
    t = torch.randn(shape, device=device, dtype=torch.float32).to(dtype)
    return t.contiguous(memory_format=torch.channels_last_3d)


def _bias(problem: ConvProblem, device, dtype):
    if not problem.bias:
        return None
    return torch.randn(problem.cout, device=device, dtype=torch.float32).to(dtype)


def _gather_refine(top):
    # GROUP_M is an L2 swizzle width worth a few percent; sweeping it across the
    # whole grid would double a cost that is almost entirely JIT.
    return [dataclasses.replace(c, GROUP_M=8) for c in top if c.GROUP_M != 8]


def _bwd_weight_refine(top):
    # The split count is a *runtime* argument -- it changes the grid and the
    # chunk length, not a constexpr -- so this second pass costs no JIT at all,
    # which is why it can afford to be exhaustive where the tile pass cannot.
    return [dataclasses.replace(c, SPLIT_K=sk) for c in top for sk in _SPLIT_SWEEP]


def _autograd_control(build_forward: Callable[[], tuple]):
    """Build a real forward graph **on the capture stream** and keep it alive.

    Two things at once, and both are load-bearing.

    The graph has to be real, because ``torch.nn.grad.conv3d_*`` fabricates a
    zero-strided placeholder for the operand it is differentiating and
    ``convolution_backward`` chooses its solver from that operand's layout --
    measured, 3.2x slow at the ``k=1`` head, and the source of a retracted 4.51x.

    It has to be built on :func:`~triton_conv3d.bench.harness.capture_stream`,
    because otherwise the autograd node records the default stream and CUDA
    graph capture refuses it outright: *"During CUDA graph capture, autograd node
    ``ConvolutionBackward0`` has a stale reference to the default stream"*.  That
    refusal is what would push a backward cell back onto the eager path -- i.e.
    onto a launcher-inclusive number for both arms -- so it is worth the two
    lines it costs.
    """
    s = capture_stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        made = build_forward()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return made


# -- conv: the ordinary convolution -----------------------------------------


def _conv_fwd(problem: ConvProblem, device: str,
              control: bool = True) -> _Case:
    dtype = _TORCH_DTYPE[problem.dtype]
    k = tuple(problem.kernel)
    x = _randn(problem.input_shape, device, dtype)
    w = _randn(problem.weight_shape, device, dtype)
    b = _bias(problem, device, dtype)
    y = torch.empty(problem.output_shape, device=device, dtype=dtype,
                    memory_format=torch.channels_last_3d)
    m = problem.n * math.prod(problem.out_spatial)

    def triton(cfg):
        def run():
            # ``w`` itself, not a transform of it: ``_randn`` returns it
            # channels-last, which is what a ScaFFold parameter is, and the
            # kernel reads that layout in place.  Passing ``weight_rsck=`` here
            # would time a path the integration no longer takes.
            conv3d_forward(x, w, b, problem.stride, problem.padding,
                           config=cfg, out=y)
        return run

    def miopen():
        with torch.no_grad():
            F.conv3d(x, w, b, stride=problem.stride, padding=problem.padding)

    def reference():
        got = conv3d_forward(x, w, b, problem.stride, problem.padding)
        with torch.no_grad():
            ref = F.conv3d(x, w, b, stride=problem.stride, padding=problem.padding)
        return got, ref

    return _Case(
        triton=triton, miopen=miopen if control else None, transform=None,
        candidates=lambda: candidate_configs(m, problem.cin, problem.cout, dtype),
        refine=_gather_refine,
        shipped_config=lambda: select_config(m, problem.cin, problem.cout, k, dtype),
        reference=reference if control else None, primary=x, keep=(x, w, b, y),
    )


def _conv_bwd_data(problem: ConvProblem, device: str,
                   control: bool = True) -> _Case:
    dtype = _TORCH_DTYPE[problem.dtype]
    k = tuple(problem.kernel)
    w = _randn(problem.weight_shape, device, dtype)
    b = _bias(problem, device, dtype)
    gy = _randn(problem.output_shape, device, dtype)
    gx = torch.empty(problem.input_shape, device=device, dtype=dtype,
                     memory_format=torch.channels_last_3d)

    def triton(cfg):
        def run():
            # As in the forward: the channels-last parameter is the operand, and
            # the tap flip and the transpose are both constexprs in the kernel.
            conv3d_backward_data(gy, w, problem.input_shape, problem.stride,
                                 problem.padding, config=cfg, out=gx)
        return run

    def build():
        xg = _randn(problem.input_shape, device, dtype).requires_grad_(True)
        with torch.enable_grad():
            yg = F.conv3d(xg, w, b, stride=problem.stride, padding=problem.padding)
        return xg, yg

    # ``w`` does not require grad, so ``convolution_backward``'s output mask is
    # ``(True, False, False)`` and no weight gradient is computed.
    #
    # Not built at all without a control: this line is a real ``F.conv3d``, and
    # on a shape MIOpen has not found yet it is where the find is paid.
    xg, yg = _autograd_control(build) if control else (None, None)

    def miopen():
        torch.autograd.grad(yg, (xg,), gy, retain_graph=True)

    def reference():
        got = conv3d_backward_data(gy, w, problem.input_shape, problem.stride,
                                   problem.padding)
        ref = torch.autograd.grad(yg, (xg,), gy, retain_graph=True)[0]
        return got, ref

    return _Case(
        triton=triton, miopen=miopen if control else None, transform=None,
        # Swapped: backward-data reduces over Cout and its GEMM's N is Cin.
        candidates=lambda: candidate_configs(
            problem.n * math.prod(problem.spatial), problem.cout, problem.cin,
            dtype),
        refine=_gather_refine,
        shipped_config=lambda: bwd_data_config(
            problem.output_shape, problem.cin, k, dtype,
            padding=problem.padding),
        reference=reference if control else None, primary=gy,
        # ``xg``/``yg`` are kept because the graph (and so the control) dies
        # with them.
        keep=(gy, w, b, gx, xg, yg),
    )


def _conv_bwd_weight(problem: ConvProblem, device: str,
                     control: bool = True) -> _Case:
    dtype = _TORCH_DTYPE[problem.dtype]
    k = tuple(problem.kernel)
    x = _randn(problem.input_shape, device, dtype)
    w = _randn(problem.weight_shape, device, dtype)
    b = _bias(problem, device, dtype)
    gy = _randn(problem.output_shape, device, dtype)
    gw = grad_weight_empty(problem.cout, problem.cin, k, dtype=dtype, device=device)
    k_total = problem.n * math.prod(problem.out_spatial)
    padded = any(problem.padding)

    def candidates():
        return candidate_bwd_weight_configs(
            problem.cout, problem.cin, k, k_total, dtype,
            splits=(0,), padded=padded,
        )

    def splits_for(cfg):
        return split_count(cfg, problem.cout, problem.cin, problem.tap_count,
                           k_total, problem.out_spatial[2])[0]

    ws = _sweep_workspace(candidates(), splits_for, problem.cout, problem.cin,
                          k, device)

    def triton(cfg):
        def run():
            conv3d_backward_weight(x, problem.weight_shape, gy, problem.stride,
                                   problem.padding, config=cfg, workspace=ws,
                                   out=gw)
        return run

    def build():
        wg = w.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            yg = F.conv3d(x, wg, b, stride=problem.stride, padding=problem.padding)
        return wg, yg

    wg, yg = _autograd_control(build) if control else (None, None)

    def miopen():
        torch.autograd.grad(yg, (wg,), gy, retain_graph=True)

    def reference():
        got = conv3d_backward_weight(x, problem.weight_shape, gy,
                                     problem.stride, problem.padding)
        ref = torch.autograd.grad(yg, (wg,), gy, retain_graph=True)[0]
        return got, ref

    return _Case(
        triton=triton, miopen=miopen if control else None, transform=None,
        candidates=candidates, refine=_bwd_weight_refine,
        shipped_config=lambda: bwd_weight_config(
            problem.cout, problem.cin, k, k_total, dtype, padded=padded),
        reference=reference if control else None, primary=x,
        keep=(x, w, b, gy, gw, ws, wg, yg),
    )


# -- convT: the k == s transposed convolution --------------------------------
#
# All three of these are served by *four* entry points across three modules, and
# the channel widths swap on the way in.  Each builder names the swap once, in
# the call, rather than a shared helper naming it three times differently.


def _convt_fwd(problem: ConvProblem, device: str,
               control: bool = True) -> _Case:
    dtype = _TORCH_DTYPE[problem.dtype]
    k = tuple(problem.kernel)
    x = _randn(problem.input_shape, device, dtype)
    w = _randn(problem.weight_shape, device, dtype)
    b = _bias(problem, device, dtype)
    y = torch.empty(problem.output_shape, device=device, dtype=dtype,
                    memory_format=torch.channels_last_3d)
    m = problem.n * math.prod(problem.spatial)

    def triton(cfg):
        def run():
            conv_transpose3d_forward(x, w, b, k, config=cfg, out=y)
        return run

    def miopen():
        with torch.no_grad():
            F.conv_transpose3d(x, w, b, stride=k)

    def reference():
        got = conv_transpose3d_forward(x, w, b, k)
        with torch.no_grad():
            ref = F.conv_transpose3d(x, w, b, stride=k)
        return got, ref

    return _Case(
        triton=triton, miopen=miopen if control else None, transform=None,
        candidates=lambda: candidate_transposed_configs(
            m, problem.cin, problem.cout, problem.tap_count, dtype),
        refine=_gather_refine,
        shipped_config=lambda: transposed_config(m, problem.cin, problem.cout,
                                                 k, dtype),
        reference=reference if control else None, primary=x, keep=(x, w, b, y),
    )


def _convt_bwd_data(problem: ConvProblem, device: str,
                    control: bool = True) -> _Case:
    """``grad_input = conv3d(grad_output, w, stride=k)`` -- an ordinary forward.

    So the config that runs is :func:`~triton_conv3d.gather_gemm.select_config`'s
    for the *strided* convolution, whose ``(cin, cout)`` are this operator's
    ``(cout, cin)``.  ``m5_convT_bench`` had no way to say that -- the only
    config object it held was a ``TransposedConfig`` -- so it dropped ``config=``
    entirely and could not sweep this direction at all.
    """
    dtype = _TORCH_DTYPE[problem.dtype]
    k = tuple(problem.kernel)
    x = _randn(problem.input_shape, device, dtype)
    w = _randn(problem.weight_shape, device, dtype)
    b = _bias(problem, device, dtype)
    gy = _randn(problem.output_shape, device, dtype)
    gx = torch.empty(problem.input_shape, device=device, dtype=dtype,
                     memory_format=torch.channels_last_3d)
    # The strided convolution's M is this operator's *input* volume, and its
    # (cin, cout) are (Cout, Cin) of the transposed operator.
    m = problem.n * math.prod(problem.spatial)

    def triton(cfg):
        def run():
            conv_transpose3d_backward_data(gy, w, problem.input_shape, k,
                                           config=cfg, out=gx)
        return run

    def build():
        xg = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            yg = F.conv_transpose3d(xg, w, b, stride=k)
        return xg, yg

    xg, yg = _autograd_control(build) if control else (None, None)

    def miopen():
        torch.autograd.grad(yg, (xg,), gy, retain_graph=True)

    def reference():
        got = conv_transpose3d_backward_data(gy, w, problem.input_shape, k)
        ref = torch.autograd.grad(yg, (xg,), gy, retain_graph=True)[0]
        return got, ref

    return _Case(
        triton=triton, miopen=miopen if control else None, transform=None,
        candidates=lambda: candidate_configs(m, problem.cout, problem.cin, dtype),
        refine=_gather_refine,
        shipped_config=lambda: select_config(m, problem.cout, problem.cin, k, dtype),
        reference=reference if control else None, primary=gy,
        keep=(x, w, b, gy, gx, xg, yg),
    )


def _convt_bwd_weight(problem: ConvProblem, device: str,
                      control: bool = True) -> _Case:
    """The same reduction ``conv3d_backward_weight`` performs, operands swapped.

    ``grad_output`` is the strided convolution's input and ``x`` is its output
    gradient, so the reduction's ``(cout, cin)`` are this operator's
    ``(cin, cout)`` and its ``k_total`` is this operator's *input* volume.  The
    workspace is sized from the swapped widths, not from the ones a reader would
    name.
    """
    dtype = _TORCH_DTYPE[problem.dtype]
    k = tuple(problem.kernel)
    x = _randn(problem.input_shape, device, dtype)
    w = _randn(problem.weight_shape, device, dtype)
    b = _bias(problem, device, dtype)
    gy = _randn(problem.output_shape, device, dtype)
    gw = grad_transposed_weight_empty(problem.cin, problem.cout, k,
                                      dtype=dtype, device=device)
    k_total = problem.n * math.prod(problem.spatial)

    def candidates():
        return candidate_bwd_weight_configs(
            problem.cin, problem.cout, k, k_total, dtype,
            splits=(0,), padded=False,
        )

    def splits_for(cfg):
        return split_count(cfg, problem.cin, problem.cout, problem.tap_count,
                           k_total, problem.spatial[2])[0]

    ws = _sweep_workspace(candidates(), splits_for, problem.cin, problem.cout,
                          k, device)

    def triton(cfg):
        def run():
            conv_transpose3d_backward_weight(x, problem.weight_shape, gy, k,
                                             config=cfg, workspace=ws, out=gw)
        return run

    def build():
        wg = w.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            yg = F.conv_transpose3d(x, wg, b, stride=k)
        return wg, yg

    wg, yg = _autograd_control(build) if control else (None, None)

    def miopen():
        torch.autograd.grad(yg, (wg,), gy, retain_graph=True)

    def reference():
        got = conv_transpose3d_backward_weight(x, problem.weight_shape, gy, k)
        ref = torch.autograd.grad(yg, (wg,), gy, retain_graph=True)[0]
        return got, ref

    return _Case(
        triton=triton, miopen=miopen if control else None, transform=None,
        candidates=candidates, refine=_bwd_weight_refine,
        shipped_config=lambda: bwd_weight_config(
            problem.cin, problem.cout, k, k_total, dtype, padded=False),
        reference=reference if control else None, primary=x,
        keep=(x, w, b, gy, gw, ws, wg, yg),
    )


def _sweep_workspace(cands, splits_for, cout, cin, k, device) -> torch.Tensor:
    """One workspace, sized for the largest split count the sweep can ask for.

    So that allocation is outside every timed region -- MIOpen's own time
    excludes its workspace allocation too.
    """
    max_splits = max(
        [splits_for(c) for c in cands]
        + [splits_for(dataclasses.replace(c, SPLIT_K=sk))
           for c in cands for sk in _SPLIT_SWEEP]
    )
    return torch.empty(workspace_elements(max_splits, cout, cin, k),
                       dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# The two operators
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Op:
    """What is per-*operator* rather than per-direction, in one place.

    Four things, and each of them is a way to measure the wrong problem:

    ``form``
        Which of the module docstring's three shapes a cell measures, as a
        function of the requested form name.  ``conv`` honours the name;
        ``convT`` must **not** be haloed under any of them, because DistConv's
        halo is ``k // 2`` for an odd kernel and 0 at ``k = 2`` and the adapter
        exchanges nothing there either.  Haloing a transposed problem would
        grow its input by two voxels per axis and measure a convolution the
        model never runs.
    ``order``
        The order the cells are measured in, kept per operator so a re-capture
        is comparable with what is already on disk.
    ``sweepable``
        Which directions have a candidate list worth racing.
    ``build``
        The six-cell table itself.
    """

    name: str
    selects: Callable[[ConvProblem], bool]
    form: Callable[[ConvProblem, str], ConvProblem]
    form_note: Callable[[str], str]
    order: Callable[[ConvProblem], object]
    build: Mapping[Direction, Callable[[ConvProblem, str], _Case]]


#: What each ``--form`` name means, in one place, so the word a user typed and
#: the shape a kernel is handed cannot drift apart.
_FORMS: dict[str, Callable[[ConvProblem], ConvProblem]] = {
    "distconv": lambda p: p.halo_variant,
    "adapter": lambda p: p.production_variant,
    "logical": lambda p: p,
}

_FORM_NOTES = {
    "distconv": "DistConv's halo'd, unpadded form -- what the MIOpen baseline "
                "was profiled in",
    "adapter": "the form ScaFFold's Triton rung is handed -- what production "
               "runs, padded at every configuration",
    "logical": "the module's own statement, unhalo'd and padded",
}

_OPERATORS: dict[str, _Op] = {
    "conv": _Op(
        name="conv",
        selects=lambda p: not p.transposed,
        form=lambda p, form: _FORMS[form](p),
        form_note=lambda form: _FORM_NOTES[form],
        # Corpus order is measured-cost order, and it is what every stored
        # capture's ``--problems`` indices refer to.
        order=lambda p: 0,
        build={"fwd": _conv_fwd, "bwd-data": _conv_bwd_data,
               "bwd-weight": _conv_bwd_weight},
    ),
    "convT": _Op(
        name="convT",
        selects=lambda p: p.transposed,
        form=lambda p, form: p,
        form_note=lambda form: "as recorded (no halo in any form at k=2)",
        # Cheapest first, as ``m5_convT_bench`` ran them, so a re-capture lines
        # up row for row with ``m5_shipped_*.json``.
        order=lambda p: math.prod(p.spatial) * p.cin,
        build={"fwd": _convt_fwd, "bwd-data": _convt_bwd_data,
               "bwd-weight": _convt_bwd_weight},
    ),
}


def operator_of(problem: ConvProblem) -> Operator:
    """Which operator a problem is.  The **only** place this question is asked."""
    return "convT" if problem.transposed else "conv"


def _build(problem: ConvProblem, direction: Direction, device: str = "cuda",
           operator: Operator | None = None, control: bool = True) -> _Case:
    """Operands and launchers for one problem in one direction.

    A lookup into :data:`_OPERATORS`, not a switch: the operator is resolved
    once, here, and the builder it names never asks again.

    The Triton launchers exclude allocation, deliberately: MIOpen's time
    excludes its own workspace allocation, so excluding ours keeps the
    comparison like-for-like.  They no longer exclude a weight transform, for the
    stronger reason that there is not one -- the weight operand is the
    channels-last parameter itself in every direction.

    ``control=False`` builds the Triton operands and **nothing else**: no MIOpen
    launcher and, for a backward direction, no autograd graph -- which is the
    expensive half.  Building the control does not merely cost the arm's timing;
    running it once costs MIOpen's *find*, which under ``cudnn.benchmark = True``
    cannot be replayed from disk (``review/MIOPEN_CACHE.md`` §2) and which
    measures **92-174 s per cell** on this corpus against 0.3-1.2 s for
    everything else in the cell.  A Triton-only capture is therefore two orders
    of magnitude cheaper than a comparison, and that is entirely MIOpen's find.
    """
    op = _OPERATORS[operator or operator_of(problem)]
    try:
        builder = op.build[direction]
    except KeyError:
        raise ValueError(f"unsupported direction {direction!r}") from None
    return builder(problem, device, control)


# ---------------------------------------------------------------------------
# What is inside the timed region
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Region:
    """The decision about what every arm of one cell is timed with.

    One object for the whole cell, never one per arm.  ``HARNESS_RIGOR.md`` H10
    is the reason: a per-arm instrument biases a ratio *even when both arms are
    individually right*, and the version of that mistake available here --
    hoisting Triton's config lookup out while leaving PyTorch's dispatch inside
    the MIOpen arm -- flatters us by up to 1.4x on exactly the sub-0.15 ms cells
    where it is hardest to see.
    """

    #: ``"kernel"`` (both arms replayed from a graph) or ``"call"`` (both arms
    #: called from Python).
    kind: str
    chunk: int
    fns: Mapping[str, Callable[[], object]]
    eager_ms: Mapping[str, float]
    note: str

    @property
    def excludes_launcher(self) -> bool:
        return self.kind == "kernel"


def _timed_region(variants: Mapping[str, Callable[[], object]],
                  launcher: str = "exclude") -> _Region:
    """Decide, for one cell, what the published number will contain.

    **Excluded** from a ``kind="kernel"`` measurement, on **every** arm: Python
    call overhead, PyTorch's dispatcher, autocast and shape re-validation, the
    tuned-table lookup, MIOpen's descriptor construction and find-database
    probe, the autograd engine's node walk, and the launcher itself.
    **Included**: the kernels, in the order and with the operands the eager call
    issues them, back to back on one stream.

    That boundary is the same on both sides only because the *whole comparison*
    moves together.  Measured on ``convT 1024->512 @ 8^3``, per call
    (``work/triton-conv/bin/launcher_symmetry.py --only census-small``):

    ============================  ========  =======  ============  ===============
    arm                           eager     kernel   host launch   speedup in->out
    ============================  ========  =======  ============  ===============
    ``convT`` fwd Triton          0.0421    0.0282   13.9 us       1.659 -> 2.420
    ``convT`` fwd MIOpen          0.0699    0.0683   **1.6 us**
    ``convT`` bwd-data Triton     0.0539    0.0350   18.9 us       1.414 -> 1.130
    ``convT`` bwd-data MIOpen     0.0761    0.0395   **36.6 us**
    ``convT`` bwd-weight Triton   0.0712    0.0542   17.0 us       1.176 -> 0.931
    ``convT`` bwd-weight MIOpen   0.0839    0.0504   33.4 us
    ============================  ========  =======  ============  ===============

    The host costs differ by **23x** between arms of the *same* cell, so an eager
    number is not a launcher-neutral number that a graph then "improves": it is a
    number with a per-arm instrument in it.  Note the last column runs both ways
    -- excluding the launcher is not a favour to Triton.  It doubles the forward
    (the Triton arm was paying 13.9 us against a 28 us kernel while MIOpen paid
    1.6 us) and it turns the weight gradient from a 1.176x win into a **0.931x
    loss** (the MIOpen arm was paying an autograd-engine walk worth twice the
    Triton entry point's).

    Three ways this refuses to produce a mixed measurement:

    * if any arm cannot be captured, **no** arm is -- the cell falls back to
      eager whole and says so in ``note``;
    * ``chunk`` comes from the shortest arm's duration alone
      (:func:`~triton_conv3d.bench.harness.common_chunk`), never from a per-arm
      estimate of the replay cost;
    * above 40 ms per call nothing is captured, because there the largest host
      cost measured on this node (0.08 ms) is under 0.2% of either arm and the
      eager number is already launcher-exclusive to within a fifth of the
      target precision.
    """
    names = list(variants)
    if launcher == "include":
        return _Region("call", 1, variants, {},
                       "Python call: kernel + dispatch + config lookup + launcher")
    if launcher != "exclude":
        raise ValueError(f"unknown launcher policy {launcher!r}")

    eager = {n: per_call_ms(variants[n]) for n in names}
    durations = [eager[n] for n in names]
    if not graph_is_worthwhile(durations):
        return _Region("call", 1, variants, eager,
                       f"Python call: every arm is above {min(durations):.1f} ms, "
                       "where the measured host cost (<=0.08 ms) is under 0.2%")
    chunk = common_chunk(durations)
    captured: dict[str, Callable[[], object]] = {}
    try:
        for n in names:
            captured[n] = capture(variants[n], chunk)
    except CaptureError as exc:
        captured.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return _Region("call", 1, variants, eager,
                       f"Python call: {n!r} could not be captured ({exc}), so no "
                       "arm was -- a mixed measurement is worth up to 1.4x")
    return _Region("kernel", chunk, captured, eager,
                   f"CUDA graph replay, {chunk} calls per graph: kernels only, "
                   "no dispatch, no config lookup, no launcher, on every arm")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _sweep(case: _Case, configs: list[ConvConfig],
           verbose: bool = False) -> list[tuple[ConvConfig, float]]:
    """Time every config once, cheaply.  Failures (LDS overflow, OOM) are skipped.

    Eager, deliberately: this pass only has to *rank*, it runs hundreds of
    configs, and a capture per config would cost more than the ranking is worth.
    The launcher cost it carries is the same for every candidate, which is the
    property a ranking needs; the finalists are then re-measured in the race,
    where the launcher is excluded.
    """
    ranked: list[tuple[ConvConfig, float]] = []
    for cfg in configs:
        run = case.triton(cfg)
        try:
            run()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - any compile/launch failure
            if verbose:
                print(f"      skip {cfg}: {type(exc).__name__}: {str(exc)[:70]}")
            continue
        # Adaptive ``iters``, fixed 3 rounds, and no tax probe.  Its old
        # ``iters=3`` was not neutral between the configs it was ranking: the
        # first call after a synchronize pays a queue restart worth 3% at 1.4 ms
        # and 42% at 0.07 ms, so a fixed small ``iters`` charges that restart to
        # whichever config is fastest -- exactly the config the sweep is trying
        # to find.  Sizing the block by time makes the restart the same fraction
        # for every candidate.
        meas = interleaved({"t": run}, warmup=None, iters=None, rounds=3,
                           warmup_s=0.02, warmup_min=2, block_ms=5.0,
                           measure_tax=False)["t"]
        ranked.append((cfg, meas.median))
    ranked.sort(key=lambda kv: kv[1])
    return ranked


def measure_problem(problem: ConvProblem, *, direction: Direction = "fwd",
                    operator: Operator | None = None,
                    max_configs: int = 0, iters: int = 0, rounds: int = 0,
                    shipped: bool = False, verbose: bool = False,
                    budget_s: float = 20.0, target_rel: float = 0.02,
                    launcher: str = "exclude", control: str = "miopen") -> dict:
    """Sweep, then race the finalists against MIOpen in one interleaved block.

    ``control="none"`` drops the MIOpen arm and measures the Triton kernels
    alone.  The row then carries no ``miopen_*`` and no ``speedup*`` key -- an
    absent number is absent, not zero -- and says so in ``control``.  Everything
    else is unchanged: the same CUDA-graph region, the same adaptive stopping,
    the same 95% interval, the same ``stop`` reason.  What it buys is the whole
    of MIOpen's find (§1 of ``review/HARNESS_SPEED.md``: 98.3% of a three-cell
    problem's wall clock on this node); what it costs is the comparison, so use
    it when the baseline is the deliverable and the ratio is not.

    ``shipped`` skips the sweep and times the config **this cell's entry point
    would resolve on its own** -- the tuned table plus the heuristic fallback --
    resolved once, outside the timed region.  That is the kernel a caller
    actually gets; it is not the same number as the *call* a caller actually
    makes, which also pays 0.0164 ms of table lookup per call at the transposed
    sites, and `--launcher include` is how to see that.  Confirming the shipped
    config's time agrees with the sweep's is what makes the sweep's numbers a
    claim about the shipped kernel rather than about a config nobody will use.

    ``iters`` and ``rounds`` default to 0, meaning *decide online*: the race
    grows until the paired speedup's 95% interval is inside ``target_rel`` or
    ``budget_s`` of wall clock is gone, and the row records which.  Pinning both
    to integers restores the old fixed 10x6 exactly, for a capture that has to
    be byte-comparable with an earlier one.

    A fixed 10x6 is 60 calls whatever the kernel costs.  Over this corpus that
    is microseconds at the transposed sites and **45 minutes** at the 2 GiB
    cliff, where one backward-weight call is 45.2 s -- and no amount of
    averaging is going to change a 2789x ratio.
    """
    op = operator or operator_of(problem)
    row: dict = {
        "problem": problem.label,
        "operator": op,
        "direction": direction,
        "cin": problem.cin, "cout": problem.cout,
        "spatial": list(problem.spatial),
        "kernel": list(problem.kernel),
        "padding": list(problem.padding),
        "dtype": problem.dtype,
        "gemm": list(problem.gemm_shape(direction)),
        "flops": problem.flops(direction),
        "roofline_tflops": problem.roofline_flops(direction) / 1e12,
    }
    if control not in ("miopen", "none"):
        raise ValueError(f"unknown control {control!r}")
    row["control"] = control
    case = None
    region = None
    try:
        case = _build(problem, direction, operator=op,
                      control=(control == "miopen"))
        # ``UntypedStorage.size()`` is already in *bytes*, which is what the
        # specializer's ``is_within_2gb`` compares -- multiplying by the
        # element size again would report every operand as ineligible.
        row["x_storage_bytes"] = case.primary.untyped_storage().size()
        row["buffer_ops_eligible"] = bool(
            case.primary.untyped_storage().size() <= 2**31 - 1
        )

        if shipped:
            ranked: list[tuple[ConvConfig | None, float]] = [
                (case.shipped_config(), 0.0)
            ]
        else:
            configs = case.candidates()
            if max_configs:
                configs = configs[:max_configs]
            ranked = _sweep(case, configs, verbose=verbose)
            if not ranked:
                row["error"] = "no config ran"
                return row
            refine = case.refine([cfg for cfg, _ in ranked[: 2 * _FINALISTS]])
            ranked += _sweep(case, refine, verbose=verbose)
            ranked.sort(key=lambda kv: kv[1])
            row["configs_ran"] = len(ranked)
            row["sweep"] = [[str(c), ms] for c, ms in ranked]

        variants: dict[str, Callable[[], object]] = {}
        if case.miopen is not None:
            variants["miopen"] = case.miopen
        owner: dict[str, ConvConfig | None] = {}
        for i, (cfg, _) in enumerate(ranked[:_FINALISTS]):
            name = f"triton#{i}"
            owner[name] = cfg
            variants[name] = case.triton(cfg)
        # The transform a real integration would hoist out of the call.  Timed
        # here so its cost is a stated number rather than an assumption.
        if case.transform is not None:
            variants["rsck_transform"] = case.transform

        pinned = bool(iters and rounds)
        # One stream for the whole cell, both policies.  The MIOpen control for
        # a backward direction is an autograd graph built on this stream, and
        # the engine synchronizes when it is asked to run somewhere else: 35 us
        # per call, on that arm only.  See :class:`on_capture_stream`.
        with on_capture_stream():
            region = _timed_region(variants, launcher)
            meas = interleaved(region.fns,
                               warmup=3 if pinned else None,
                               iters=iters or None, rounds=rounds or None,
                               budget_s=budget_s, target_rel=target_rel)
        # ``region.chunk`` calls sit behind one replay, so every *absolute* time
        # is that many times too large.  Every *relative* one -- the half-widths,
        # the paired ratio, the convergence test the harness already ran -- is
        # scale-invariant and needs no correction, which is why the division
        # happens here and not inside the harness.
        c = region.chunk

        best_name = min(
            (nm for nm in meas if nm.startswith("triton")),
            key=lambda nm: meas[nm].median,
        )
        best = meas[best_name]
        row.update(
            timed_region=region.kind,
            timed_region_note=region.note,
            graph_chunk=c,
            triton_ms=best.median / c,
            triton_best_ms=best.best / c,
            triton_spread=best.spread,
            triton_stall=best.stall_ratio,
            triton_rel_ci=best.rel_half_width,
            triton_cov=best.cov,
            triton_half_width_ms=best.half_width / c,
            triton_tax_frac=best.tax_frac,
            triton_eager_ms=region.eager_ms.get(best_name, 0.0),
            triton_config=str(owner[best_name]),
            triton_pct_roofline=100 * problem.efficiency(best.median / c, direction),
            triton_tflops=problem.flops(direction) / (best.median / c * 1e-3) / 1e12,
            measure_rounds=len(best.rounds),
            measure_iters={nm: meas[nm].iters for nm in meas},
            measure_group={nm: meas[nm].group for nm in meas},
            measure_stop=best.stop,
            measure_balanced=best.balanced,
            measure_seconds=best.seconds,
            rsck_ms=(meas["rsck_transform"].median / c
                     if "rsck_transform" in meas else 0.0),
        )
        # Only when there *is* a control.  An absent MIOpen number is left
        # absent rather than written as zero: every consumer of these rows
        # reads ``speedup`` straight out, and a zero would read as a 0.000x
        # result instead of as "not measured here".
        if "miopen" in meas:
            mio = meas["miopen"]
            # Paired per round, not median-over-median: the two arms of a round
            # ran seconds apart under the same device state, so a common-mode
            # excursion divides out of each pair before anything is reduced.
            # This is also the only quantity here that comes with an interval,
            # and the interval is the point -- ``speedup`` alone has been quoted
            # four times in this project's history against a number that could
            # not support it.
            sp = ratio(mio, best)
            row.update(
                miopen_ms=mio.median / c,
                miopen_best_ms=mio.best / c,
                miopen_spread=mio.spread,
                miopen_stall=mio.stall_ratio,
                miopen_rel_ci=mio.rel_half_width,
                miopen_tax_frac=mio.tax_frac,
                miopen_eager_ms=region.eager_ms.get("miopen", 0.0),
                miopen_pct_roofline=100 * problem.efficiency(mio.median / c,
                                                             direction),
                miopen_tflops=problem.flops(direction) / (mio.median / c * 1e-3) / 1e12,
                speedup=sp.point,
                speedup_lo=sp.lo,
                speedup_hi=sp.hi,
                speedup_rel_ci=sp.rel_half_width,
                speedup_significant=sp.significant,
            )
        # What the launcher is worth, per arm.  A probe estimate (one bracketed
        # block, no interval) minus the measured kernel; reported because the
        # *difference* between the two arms' launchers is the bias that
        # excluding them removes, and a reader should be able to see it.
        #
        # Not a measurement, and it shows: where the launcher is already
        # negligible -- above about 0.3 ms, where an event-free bracket of a
        # host-paced loop already reaches kernel throughput -- this comes out at
        # a few microseconds of either sign.  The measured version, with
        # intervals and both policies run as full races, is
        # ``work/triton-conv/bin/launcher_symmetry.py --only census``.
        for arm in ("triton", "miopen"):
            if f"{arm}_ms" not in row:
                continue
            e = row[f"{arm}_eager_ms"]
            row[f"{arm}_launcher_ms"] = (e - row[f"{arm}_ms"]) if e else 0.0
        row["finalists"] = {
            str(owner[nm]): meas[nm].median / c for nm in owner
        }
        return row
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    finally:
        del case, region
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _correctness(problem: ConvProblem, direction: Direction,
                 operator: Operator | None = None) -> dict:
    """Error against MIOpen's own answer, on the shape just measured.

    Not a substitute for the test suite -- ``tests/`` holds the bitwise-exact
    standard -- but a benchmark that reports a time without checking the result
    is how a fast wrong kernel gets believed.  The reference is
    :attr:`_Case.reference`, so it is the same six-cell table the timing uses and
    cannot drift from it.
    """
    case = None
    try:
        case = _build(problem, direction, operator=operator, control=True)
        got, ref = case.reference()
        d = (got.float() - ref.float()).abs()
        scale = ref.float().pow(2).mean().sqrt().item() or 1.0
        return {"max_abs_vs_miopen": d.max().item(),
                "rms_rel_vs_miopen": (d.pow(2).mean().sqrt().item() / scale)}
    except Exception as exc:  # noqa: BLE001
        return {"correctness_error": f"{type(exc).__name__}: {exc}"}
    finally:
        del case
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _pick(corpus, args) -> list[tuple[int, ConvProblem]]:
    """``(corpus index, problem)`` pairs, in corpus order.

    Indices are into the **corpus**, for both operators, because that is the
    only stable name a problem has: ``--problems 1,3,5`` means what it has
    always meant, and a transposed problem is now nameable the same way
    (``m5_convT_bench``'s cheapest-first index 0 is corpus index 56).  Every
    printed row and every stored row carries its index.
    """
    if args.problems:
        return [(int(i), corpus[int(i)]) for i in args.problems.split(",")]
    keep = list(enumerate(corpus))
    return keep[: args.top] if args.top else keep


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--operator", default="all",
                    choices=["conv", "convT", "all"],
                    help="which operator; 'all' measures both (default). "
                         "Supersedes the old --skip-transposed, which was "
                         "store_true with default True and so could never be "
                         "turned off")
    ap.add_argument("--direction", default="fwd",
                    choices=["fwd", "bwd-data", "bwd-weight", "all"])
    ap.add_argument("--top", type=int, default=8,
                    help="hottest N corpus problems (0 = all)")
    ap.add_argument("--problems", default=None,
                    help="comma-separated corpus indices, overriding --top")
    ap.add_argument("--max-configs", type=int, default=0)
    ap.add_argument("--iters", type=int, default=0,
                    help="calls per timed block; 0 (default) sizes it online "
                         "from the measured per-call time")
    ap.add_argument("--rounds", type=int, default=0,
                    help="rounds of the race; 0 (default) grows until the "
                         "speedup's 95%% interval is inside --precision or "
                         "--budget seconds are spent")
    ap.add_argument("--budget", type=float, default=20.0,
                    help="wall-clock seconds per cell's race (default 20)")
    ap.add_argument("--precision", type=float, default=0.02,
                    help="target relative 95%% half-width on the reported "
                         "speedup and on each arm's median (default 0.02)")
    ap.add_argument("--launcher", default="exclude",
                    choices=["exclude", "include"],
                    help="'exclude' (default): both arms are replayed from a "
                         "CUDA graph, so the number is kernel time -- no "
                         "dispatch, no config lookup, no launcher.  'include': "
                         "both arms are called from Python, so the number is "
                         "what a caller pays today")
    ap.add_argument("--shipped", action="store_true",
                    help="skip the sweep and time the config the entry point "
                         "resolves on its own -- the kernel a caller gets")
    ap.add_argument("--control", default="miopen", choices=["miopen", "none"],
                    help="'miopen' (default): race the Triton kernel against a "
                         "real MIOpen control and report a paired speedup with "
                         "its interval. 'none': measure the Triton kernel "
                         "alone, emitting no miopen_* and no speedup key. The "
                         "control is what makes a capture expensive -- MIOpen's "
                         "find cannot be replayed from disk under "
                         "cudnn.benchmark=True and costs 92-174 s per cell on "
                         "this corpus, which is 98% of a cell's wall clock "
                         "(work/triton-conv/review/HARNESS_SPEED.md)")
    ap.add_argument("--corpus", default="scaffold",
                    choices=["scaffold", "census"],
                    help="which problem list --top/--problems index into. "
                         "'scaffold' (default) is the 57 profiled problems, "
                         "cost-ordered, and is the key every stored capture "
                         "refers to -- its indices must not move. 'census' is "
                         "the 88 problems an instrumented step actually issued "
                         "at all four configurations, which is the only list "
                         "containing configuration B and the 2048-channel "
                         "sites; it is already in the adapter form, carries no "
                         "MIOpen timings and is not cost-ordered, so --form is "
                         "ignored for it and --top means 'the first N', not "
                         "'the hottest N'")
    ap.add_argument("--form", default="distconv",
                    choices=["distconv", "adapter", "logical"],
                    help="which of the three shapes of a ScaFFold convolution "
                         "to measure. 'distconv' (default) is the halo'd, "
                         "unpadded form upstream DistConv issues and the form "
                         "every capture on disk was taken in; 'adapter' is what "
                         "ScaFFold's own Triton rung is handed, which is what "
                         "production runs and is padded everywhere; 'logical' "
                         "is the module's own statement. See the module "
                         "docstring -- these are three different tuning "
                         "problems, not three views of one")
    ap.add_argument("--check", action="store_true",
                    help="also compare each result against MIOpen's")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    if os.environ.get("PYTORCH_MIOPEN_SUGGEST_NHWC") != "1":
        raise SystemExit(
            "PYTORCH_MIOPEN_SUGGEST_NHWC=1 is not set: channels_last_3d is inert "
            "on ROCm without it, so MIOpen would be handed NCDHW"
        )
    if args.check and args.control == "none":
        # MIOpen's answer *is* --check's reference, and computing it costs the
        # same find the run just declined to pay.  Refused rather than silently
        # made expensive, because a --control none run that quietly took as long
        # as a comparison would be the worst of both.  The bitwise standard is
        # triton_conv3d/tests/, which does not need a timing run to hold.
        raise SystemExit(
            "--check compares against MIOpen's own answer, so it needs "
            "--control miopen; with --control none it would reintroduce the "
            "find the run exists to avoid.  Correctness is pinned by "
            "triton_conv3d/tests/ instead."
        )

    corpus = list(census_corpus() if args.corpus == "census"
                  else scaffold_corpus())
    picks = _pick(corpus, args)
    operators = list(OPERATORS) if args.operator == "all" else [args.operator]
    directions = list(DIRECTIONS) if args.direction == "all" else [args.direction]

    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name}, {props.multi_processor_count} CUs, "
          f"torch {torch.__version__}, cudnn.benchmark={torch.backends.cudnn.benchmark}")
    print(f"control: {args.control}"
          + ("" if args.control == "miopen" else
             "  (Triton alone; no speedup is reported and none should be "
             "inferred)"))
    print(f"timed region: {args.launcher} launcher "
          f"({'CUDA graph replay -- kernels only, both arms' if args.launcher == 'exclude' else 'Python call -- kernel + dispatch + lookup + launcher, both arms'})")

    rows: list[dict] = []
    out_path = pathlib.Path(args.out) if args.out else None
    t0 = time.time()
    for opname in operators:
        op = _OPERATORS[opname]
        mine = sorted([(i, p) for i, p in picks if op.selects(p)],
                      key=lambda ip: op.order(ip[1]))
        if not mine:
            continue
        for direction in directions:
            form_note = ("recorded from a real step; already the adapter form"
                         if args.corpus == "census" else op.form_note(args.form))
            print(f"\n== {opname} {direction} -- {len(mine)} problems, "
                  f"--corpus {args.corpus} --form "
                  f"{'adapter' if args.corpus == 'census' else args.form}: "
                  f"{form_note}\n")
            for idx, p in mine:
                # The census records the shape *as the kernel was handed it*,
                # so it is already in the adapter form and carries no halo to
                # re-derive one from.  Re-applying a form transform would be a
                # no-op today and a silent lie the day the census gains a halo
                # field, so it is skipped by name rather than by luck.
                hp = p if args.corpus == "census" else op.form(p, args.form)
                print(f"  [{idx}] {hp.qualified_label}  "
                      f"(GEMM {hp.gemm_shape(direction)})")
                sys.stdout.flush()
                row = measure_problem(
                    hp, direction=direction, operator=opname,
                    max_configs=args.max_configs, iters=args.iters,
                    rounds=args.rounds, shipped=args.shipped,
                    verbose=args.verbose, budget_s=args.budget,
                    target_rel=args.precision, launcher=args.launcher,
                    control=args.control)
                row["corpus_index"] = idx
                row["logical_problem"] = p.label
                # The form is recorded per row, not only in the header: a row
                # lifted out of one capture and quoted beside another is
                # precisely how a halo'd number became "what production runs".
                row["shape_form"] = ("adapter" if args.corpus == "census"
                                     else args.form)
                row["corpus"] = args.corpus
                row["qualified_problem"] = hp.qualified_label
                row["padding"] = list(hp.padding)
                row["sites"] = list(p.sites)
                if args.check and "error" not in row:
                    row.update(_correctness(hp, direction, operator=opname))
                rows.append(row)
                _print_row(row)
                sys.stdout.flush()
                if out_path:
                    out_path.write_text(json.dumps(
                        {"device": props.name, "torch": torch.__version__,
                         "operator": args.operator, "direction": args.direction,
                         "shape_form": ("adapter" if args.corpus == "census"
                                        else args.form),
                         "corpus": args.corpus,
                         "launcher": args.launcher,
                         "control": args.control,
                         "cudnn_benchmark": True, "rows": rows}, indent=1) + "\n")

    ok = [r for r in rows if "error" not in r]
    # Two tables, not one with empty cells: without a control there are no
    # MIOpen columns to leave blank, and a blank column in a results table is
    # read as a missing value rather than as an absent measurement.
    if args.control == "miopen":
        print("\n" + format_table(
            [[
                f"{r['operator']} {r['direction']}",
                r["problem"],
                f"{r['gemm'][0]}x{r['gemm'][1]}x{r['gemm'][2]}",
                f"{r['triton_ms']:.4f}", f"{r['triton_pct_roofline']:.0f}%",
                f"{r['miopen_ms']:.4f}", f"{r['miopen_pct_roofline']:.0f}%",
                f"{r['speedup']:.3f}x",
                f"+-{r['speedup_rel_ci']:.1%}"
                + ("" if r["speedup_significant"] else "?"),
                f"{r['measure_rounds']}/{r['measure_stop'][:4]}",
                f"{r['timed_region'][:4]}x{r['graph_chunk']}",
                r["triton_config"],
            ] for r in ok],
            ["cell", "problem", "M x N x K", "triton ms", "%roof", "miopen ms",
             "%roof", "speedup", "95% CI", "rounds", "timed", "best config"],
            aligns="lllrrrrrrrrl",
        ))
    else:
        print("\n" + format_table(
            [[
                f"{r['operator']} {r['direction']}",
                r["qualified_problem"],
                f"{r['gemm'][0]}x{r['gemm'][1]}x{r['gemm'][2]}",
                f"{r['triton_ms']:.4f}",
                f"+-{r['triton_rel_ci']:.1%}",
                f"{r['triton_cov']:.2%}",
                f"{r['triton_pct_roofline']:.0f}%",
                f"{r['triton_tflops']:.1f}",
                f"{r['measure_rounds']}/{r['measure_stop'][:4]}",
                f"{r['timed_region'][:4]}x{r['graph_chunk']}",
                r["triton_config"],
            ] for r in ok],
            ["cell", "problem", "M x N x K", "triton ms", "95% CI", "CoV",
             "%roof", "TFLOP/s", "rounds", "timed", "config"],
            aligns="lllrrrrrrrl",
        ))
    print(f"\nelapsed {time.time() - t0:.0f} s")
    if out_path:
        print(f"wrote {out_path}")


def _print_row(row: dict) -> None:
    if "error" in row:
        print(f"      ERROR {row['error'][:110]}")
        return
    def launcher(arm):
        v = row.get(f"{arm}_launcher_ms", 0.0)
        return f", launcher +{v:.4f}" if v else ""
    out = (
        f"      triton {row['triton_ms']:8.4f} +-{row['triton_rel_ci']:.1%} ms "
        f"({row['triton_pct_roofline']:5.1f}% roof, stall {row['triton_stall']:.2f}x, "
        f"instrument {row['triton_tax_frac']:+.1%}{launcher('triton')})  "
        f"{row['triton_config']}\n"
    )
    if "miopen_ms" in row:
        out += (
            f"      miopen {row['miopen_ms']:8.4f} +-{row['miopen_rel_ci']:.1%} ms "
            f"({row['miopen_pct_roofline']:5.1f}% roof, "
            f"stall {row['miopen_stall']:.2f}x, "
            f"instrument {row['miopen_tax_frac']:+.1%}{launcher('miopen')})\n"
            f"      speedup {row['speedup']:.3f}x "
            f"[{row['speedup_lo']:.3f}, {row['speedup_hi']:.3f}] "
            f"{row['measure_rounds']}r/{row['measure_stop']}"
            f"{'' if row['speedup_significant'] else ' NOT SIGNIFICANT'}"
        )
    else:
        # No control ran.  Say so where the speedup would have been, rather
        # than leaving a blank a reader could take for a missing win.
        out += (f"      no MIOpen control (--control none)  "
                f"{row['measure_rounds']}r/{row['measure_stop']}")
    out += f" in {row['measure_seconds']:.1f}s   {row['timed_region_note']}"
    if row.get("rsck_ms"):
        out += (f"\n      weight transform {row['rsck_ms']:.4f} ms "
                f"({100*row['rsck_ms']/row['triton_ms']:.1f}% of kernel)")
    if "max_abs_vs_miopen" in row:
        out += f"\n      max_abs vs MIOpen {row['max_abs_vs_miopen']:.3e}"
    print(out)


if __name__ == "__main__":
    main()
