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

"""Channels-last-native Triton GroupNorm (NDHWC in, NDHWC out).

Why this exists
===============
With ``PYTORCH_MIOPEN_SUGGEST_NHWC=1`` -- which every production ScaFFold run
sets -- every convolution in the UNet emits ``channels_last_3d`` activations,
but *every* stock GroupNorm variant (eager or Inductor-compiled) consumes them
through the *logical* NCDHW iteration order, which over a channels-last tensor
is a strided gather, and then emits a **contiguous** tensor.  Measured on one
MI300A at scale 8 that costs 6.4x on GroupNorm itself (443 ms/step of GN
fwd+bwd against 69 ms/step here) *and* breaks the channels-last chain 22 times
per forward, forcing the following convolution to convert back.

A channels-last-3d contiguous ``(N, C, D, H, W)`` tensor is *physically* a dense
``(N, S, C)`` array with ``S = D*H*W``; group ``g`` owns a contiguous run of
``C/G`` channels *inside every voxel*.  The kernels below therefore let **one
program handle all groups at once** for a chunk of voxels: they read a dense
``(BLOCK_S, C)`` run, reshape the inner axis to ``(G, C/G)``, and get perfectly
coalesced loads and stores with the group axis costing nothing.  Measured at
95-98% of this device's streaming roofline at the two largest UNet shapes.

Measured on one MI300A (228 CUs), fp32, ``num_groups=8``, median of 20, the six
scale-8 UNet GroupNorm shapes, fwd / fwd+bwd in ms::

    shape             this      compiled-CL   compiled-CONT   eager-CL fwd
    [1,64,256^3]    4.23/11.39   19.51/77.01    4.77/11.85      151.2
    [1,128,128^3]   1.15/ 3.00    7.23/24.64    1.22/ 3.07       36.5
    [1,256,64^3]    0.35/ 0.97    2.27/ 7.42    0.34/ 0.84        9.1
    [1,512,32^3]    0.14/ 0.57    0.19/ 1.03    0.11/ 0.33        1.7
    [1,1024,16^3]   0.11/ 0.57    0.10/ 0.39    0.07/ 0.33        0.4
    [1,2048,8^3]    0.11/ 0.57    0.08/ 0.40    0.07/ 0.33        0.1

Over the 22 scale-8 call sites that is **442.8 -> 67.1 ms/step** of GroupNorm
fwd+bwd against today's production path (compiled GroupNorm on channels-last
input), i.e. **376 ms/step recovered**, and a dead heat with compiled GroupNorm
on *contiguous* input (66.4 ms/step) while additionally not breaking the
layout chain.  The three smallest shapes lose on host dispatch, not on GPU
work -- see :func:`select_strategy`.

The ``this`` column and the rollup were re-measured after the launch folding
below (2+2 kernels instead of 3+4, retuned jointly); the same measurement of
the unfused chain in the same process gives 4.26/11.59, 1.19/3.21, 0.35/1.01,
0.14/0.63, 0.13/0.63, 0.13/0.62 and a 69.5 ms/step rollup, i.e. **-3.4% over
the 22 sites** and -1.7% to -9.5% per shape.  The other three columns are from
the earlier sweep and are unchanged.

Public API
==========
``triton_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5,
activation=None)``
    Drop-in for ``F.group_norm`` (plus an optionally fused ReLU) with
    first-order autograd support.  Accepts *anything* ``F.group_norm`` accepts;
    inputs the Triton kernel cannot serve fall back to ``F.group_norm``
    internally (see "Layouts" below).

``is_supported(input, num_groups, weight=None, bias=None, activation=None)``
    Cheap, side-effect-free predicate: ``True`` exactly when the native Triton
    kernel will run.  Callers that already have a good fallback (e.g. a
    ``torch.compile``d GroupNorm) should test this and route rejects
    themselves; ``triton_group_norm``'s own fallback is plain eager
    ``F.group_norm``.

Contract
========
For every input ``is_supported`` accepts, the result matches ``F.group_norm``
to within fp32 reduction-order noise, with:

* **dtype** -- output dtype is exactly ``F.group_norm``'s.  Verified
  empirically on this build (torch 2.13.0+rocm7.2): without autocast the output
  dtype is the input dtype (fp32/bf16/fp16); under ``torch.autocast("cuda",
  ...)`` GroupNorm is an fp32-policy op, so the output is **fp32** for any
  input dtype.  This module reproduces that rule (see ``_autocast_out_dtype``)
  without materializing the fp32 copy of the input that autocast's cast would
  create: the kernels read the input at its native width and accumulate in
  fp32, which is bit-for-bit the same computation as upcasting first, but reads
  half the bytes.  Gradients follow the same rule: ``d_input`` has the input's
  dtype, ``d_weight``/``d_bias`` have the parameter's dtype.  Forward time at
  ``[1,64,256^3]`` / ``[1,128,128^3]`` / ``[1,256,64^3]`` in ms: fp32
  4.28/1.20/0.35, bf16 2.42/0.68/0.22, fp16 2.43/0.66/0.22, and bf16-in with
  the fp32-out autocast contract 3.08/0.84/0.28 -- so honouring autocast's
  fp32 output still buys 1.4x over fp32 end to end, because only the read side
  narrows.
* **statistics** -- always accumulated in fp32, never in the input dtype.
* **memory format** -- the output has the *input's* memory format.  This is the
  one deliberate difference from stock GroupNorm, which returns a contiguous
  tensor for every input layout (measured, see the table in
  ``work/gn-dctensor/triton/RESULTS.md``); preserving channels-last is the
  entire point of the kernel.
* **autograd** -- ``d_input``, ``d_weight``, ``d_bias``; ``weight=None`` and/or
  ``bias=None`` supported.  **First order only**: the backward is itself a
  custom op with no autograd formula of its own, so a second
  ``torch.autograd.grad`` through this op raises ``RuntimeError: Trying to
  backward through scaffold_gn.group_norm_backward.default but no autograd
  formula was registered``.  Stock ``F.group_norm`` *does* support double
  backward, so a gradient penalty or a Hessian-vector product must route
  around this kernel (``is_supported`` says nothing about second derivatives;
  it is documented there too).  It fails loudly rather than returning garbage.
* **device** -- the kernels run on the *input's* device, whatever device is
  current, matching ATen's ``DeviceGuard`` behaviour; see ``_device_guard``.
* **determinism** -- bitwise reproducible run to run and process to process.
  There are no float atomics anywhere, and the grid, split count and tile sizes
  are pure functions of the shape (the tuning table is frozen in this file for
  exactly that reason -- a *runtime* autotuner would break reproducibility by
  changing the reduction order between runs).
* **rejections** -- every shape/dtype/parameter combination ``F.group_norm``
  raises on is one ``is_supported`` returns ``False`` for, including the
  degenerate "1 value per channel" shape (``N*(C/G)*D*H*W == 1``), so a caller
  that branches on ``is_supported`` never gets an answer where the op this
  replaces would have raised.
* **eps** -- one deliberate divergence, at a value no run uses: for a
  *subnormal* fp32 ``eps`` (``< 1.18e-38``) on a zero-variance group the GPU
  flushes ``var + eps`` to zero, so ``rstd`` is ``inf`` and ``y`` is ``NaN``
  where ATen stays finite.  The boundary is exactly the normal/subnormal one
  (``eps=1.2e-38`` gives ``rstd=9.1e18``, ``eps=1e-38`` gives ``inf``); at
  ``eps == 0`` both implementations produce non-finite output identically.
  Left as is rather than clamped because clamping would perturb every
  ordinary call to defend a value nine orders of magnitude below the smallest
  plausible one.

Reduction strategy
==================
Group statistics span ``S * C/G`` elements (134M at the largest UNet shape), so
one pass cannot produce them.  Split-K partial reductions land at a fixed
scratch index and are combined by a fixed-order tree::

    fwd:  stats_partial -> normalize    (2 kernels)
    bwd:  bwd_partial   -> dx           (2 kernels)

Traffic (``B = numel * itemsize``): 3B forward, 5B backward.

Each pass is **two** launches, not the three and four an unfused split-K chain
needs: the two finalize passes and the dweight/dbias row reduction are folded
into the elementwise kernel that consumes them.  ``_normalize_kernel``
re-derives ``mean``/``rstd`` from the split-K partials itself (and program 0
stores them for the backward); ``_dx_kernel`` re-derives ``c1``/``c2`` the same
way and its first ``ceil(C/BLOCK_C)`` programs also do the dweight/dbias
reduction.  Folding plus the retuning below is worth 8.0-9.5% of fwd+bwd at the
four smallest shapes, which are host-dispatch bound, 4.9-6.8% at the two middle
ones and 1.7% at the largest.

The catch, and the reason the tuning table was re-derived rather than inherited:
**the fusion and the tiling are one problem, not two.**  A fused finalize is
recomputed by every elementwise *program*, so its cost is
``nprog_elem * nsplit`` triples of redundant (L2-resident) traffic.  Keeping the
unfused table's ``nsplit_target=2048`` at ``[1,64,256^3]``, whose flat
elementwise grid is 131072 programs, asks for 25.7 GB of redundant reads
against a 4.3 GB tensor and costs **+34% of fwd+bwd** (+70% of the forward).
Two things fix it, both of them in ``GNConfig``: the elementwise grid is capped
at ``elem_progs`` programs which then stride over the tiles (so the redundancy
is bounded by the *grid*, not by the tile count), and ``nsplit_target`` is
retuned per shape against that cap.  With both, the same shape is 1-2% *faster*
than the unfused chain.  Do not change one without re-running the other; the
coordinate-descent tuner is ``work/gn-dctensor/kernel-opt/tune.py``.

Why not one launch per pass
---------------------------
A device-scope software barrier (int32 atomics with volatile loads, no float
atomics, so still bitwise deterministic) collapses each pass to a single
launch and was measured at 2.3-2.6x on the four smallest shapes.  It is
deliberately **not** used.  A grid barrier requires every workgroup to be
co-resident, which caps the grid at the CU count (228 here); the kernel then
tops out at 0.5-0.9 TB/s against split-K's 2.7, so it loses catastrophically
the moment the shape is bandwidth-bound rather than dispatch-bound --
**18.0 ms against 3.0 ms at [1,128,128^3]**, and it does not compile at all at
``[1,64,256^3]``.  Serving both regimes therefore means shipping two kernel
families plus a crossover rule, for a whole-model gain of ~3% (65.7 -> 63.6
ms/step at scale 8); and under CUDA-graph capture, where launch count is free,
the ten small-shape sites are already only 1.05 ms of a 64.1 ms/step total, so
the gain is zero.  Hand-rolled inter-workgroup synchronisation is not a good
trade for 3% in a benchmark whose value depends on being trustworthy and
reproducible.  The measurements are in
``work/gn-dctensor/triton-small/RESULTS.md``.

Numerics: Welford, not ``E[x^2]-E[x]^2``
========================================
The prototype accumulated ``sum(x)`` and ``sum(x*x)`` and formed
``var = E[x^2] - E[x]^2``.  That is split-friendly and cheap but cancels
catastrophically once ``mean >> std``, because it subtracts two nearly equal
large numbers to recover a small one.

Here each tile instead produces ``(count, mean, M2)`` via a *corrected*
two-pass over registers -- ``mean0 = sum(x)/n``, then ``corr = sum(x-mean0)/n``
to recover the digits the first sum lost, then ``M2 = sum((x-mean0-corr)^2)``
-- and tiles and splits are merged with Chan's parallel combine.  Every step is
register-only (the tile is read from HBM exactly once either way) and
atomic-free, so neither the traffic model nor determinism changes.

Measured at ``[1,256,64^3]``, ``num_groups=8``, affine, relative error of the
*output* against a float64 reference computed from the same fp32 samples:

    x ~ N(mu, sigma)     this kernel   ATen fp32   E[x^2]-E[x]^2
    mu=0,    sigma=1       1.6e-07      4.1e-07       1.8e-07
    mu=10,   sigma=1       3.0e-07      6.9e-07       9.9e-06
    mu=100,  sigma=1       8.0e-07      4.2e-06       5.6e-04
    mu=1e3,  sigma=1e-2    1.1e-04      2.5e-03       2.3e+00

At ``mu/sigma = 1e5`` the old formulation has lost the variance outright (the
difference of the two ~1e6-sized fp32 terms is below one ulp, so ``rstd``
saturates on ``eps`` and the output is meaningless), while this kernel is still
good to 1.1e-04 -- and is 4-23x *more* accurate than ATen's own fp32 GroupNorm
at every non-trivial mean.  The residual 1.1e-04 is the fp32 representation
floor rather than an algorithm defect: a mean of 1e3 held in fp32 is quantized
to ~6e-5, which is 6e-3 of a standard deviation here, and both kernels sit on
that floor.

Cost of the rewrite, isolated by timing the kernels alone against the
prototype's: **+0.8%** on the forward at ``[1,64,256^3]`` (the shape that
dominates the step), +3-8% at the middle shapes, and **0%** on the backward,
which does not compute a variance.  A cheaper shifted-mean variant (two tile
reductions instead of three, shift taken from a peeled first tile) would
recover most of that; it was not worth the extra failure mode for ~4 ms/step.

What the third pass (``corr``) is worth, separately
---------------------------------------------------
The accuracy above is mostly the *two-pass* structure; the ``corr`` term is a
third reduction on top of it and deserves its own accounting.  Deleting it
outright (keeping ``mean_t = mean0``, ``M2 = sum((x-mean0)^2)``) and comparing
both against float64 on the same fp32 samples, relative error of ``rstd``,
10 seeds each:

    regime                                     with corr    without    ratio
    one tile per group reduction (nsplit=1):
      [2,64,8,4,4] G=4, mu/sigma=1e6            1.1e-07     1.6e-04    1472x
      [2,64,8,4,4] G=2, mu/sigma=1e6            8.4e-08     4.9e-05     580x
      [2,64,8,4,4] G=1, mu/sigma=1e6            9.8e-08     2.1e-05     216x
      [2,64,8,4,4] G=4, mu/sigma=1e5            9.9e-08     8.4e-06      85x
    many tiles and splits (the production configs):
      [1,512,32^3]  G=8, mu/sigma=1e5           1.2e-05     4.4e-05     3.8x
      [1,1024,16^3] G=8, mu/sigma=1e5           8.6e-06     2.5e-05     2.9x
      [1,256,24^3]  G=8, mu/sigma=1e5           3.0e-05     3.7e-05     1.3x
      [1,256,24^3]  G=8, mu/sigma=1e7           5.8e-04     2.3e-06    0.004x

So it is decisively load-bearing exactly where the tile mean is formed from
many large values -- up to 1472x on ``rstd`` -- and worth a steady 1.3-4x in
the multi-split configs the tuning table actually picks, at ``mu/sigma = 1e5``.
Past ``mu/sigma ~ 1e6`` with many splits it can go the *other* way (last row):
there the true spread between tile means is smaller than one ulp of the means
themselves, so Chan's between-tile term is computed from quantization noise
either way and the uncorrected version's inflated ``M2`` partly cancels it.
That regime is past fp32's floor for this computation (a mean of 1e6 held in
fp32 quantizes to 0.06, i.e. 6% of a standard deviation at sigma=1) and no
production input is near it.

The *output* error is nearly unmoved by any of this -- at most ~1.4x in either
direction -- which is why the term looks free to delete if you only measure
``y``: the output is dominated by the fp32 representation of the mean, which
``corr`` cannot improve (``mean0 + corr`` rounds straight back to ``mean0``
once the mean is large).  It is ``rstd`` that carries the benefit.

Price, measured the same way (median of 20 forwards, correction removed
outright rather than zeroed): **+2.3%** of the forward at ``[1,64,256^3]``,
+3.0% at ``[1,128,128^3]``, +4.7% at ``[1,256,64^3]``, and nothing measurable
(-1.2% to +0.5%, i.e. noise) at the three launch-bound shapes.  At the shape
that dominates the step that is +0.10 ms of a 11.6 ms fwd+bwd, i.e. +0.9%.
Kept: a 1.3-1472x accuracy factor on the statistic the whole rewrite exists to
protect is worth ~1% of GroupNorm time.  The load-bearing case is pinned by
``test_welford_correction_recovers_rstd_in_a_single_tile_reduction`` in
``tests/test_triton_group_norm_edge.py``, so deleting the term now fails the
suite instead of passing it silently.

Layouts
=======
* ``channels_last_3d`` 5-D input -> **native Triton kernel**, channels-last
  output.  This is the fast path and the only one ``is_supported`` accepts.
* Plain contiguous NCDHW (and every other layout/rank) -> ``triton_group_norm``
  falls back to ``F.group_norm``, which returns a contiguous tensor, so the
  input's memory format is still preserved.  ``is_supported`` returns ``False``
  so that callers keep their own (probably compiled) fallback rather than
  silently dropping to the eager kernel.

  This is a deliberate scope decision, not an oversight.  A native NCDHW kernel
  would need a *different* tiling -- with C outermost the fast axis is spatial,
  so a program must own one group and stream S, rather than owning all groups
  and streaming voxels -- i.e. a second family of four kernels.  The payoff is
  small: on contiguous input Inductor's compiled GroupNorm already reaches
  89-92% of this device's measured streaming roofline (RESULTS.md 4) -- and the
  table above confirms it, 66.4 ms/step against this kernel's 67.1 -- so a
  native NCDHW kernel could win ~10% there, against the 6.4x it wins on
  channels-last input.  If a mixed-layout model ever makes that 10% matter, the
  place to add it is the strategy hook below.

Addressing
==========
``[2, 64, 256^3]`` is *exactly* 2^31 elements, so int32 linear offsets block
batch>1 at the largest UNet shape and every shape above it.  The kernels widen
only the **scalar tile base** to int64 (``INT64`` is a ``tl.constexpr``, so
shapes that fit still emit pure 32-bit code); the vector offsets inside a tile
span at most ``BLOCK_S*C + C`` elements and stay int32 either way.  That is why
the wide path is free: forcing int64 on every scale-8 shape moves fwd+bwd by
-0.8% to +0.8% and forward by -3.7% to +4% (a 0.02 ms swing on the two smallest,
launch-bound shapes) -- noise in both directions.  The switch is kept anyway
because it costs one constexpr and documents where the boundary is; correctness
above 2^31 elements is covered by a test at ``[2, 64, 256, 256, 257]``
(2_155_872_256 elements, 2.5e-05 relative error on *both* batch items, i.e. the
same reduction noise a 134M-element fp32 reduction has anywhere).

Fused activation
================
``activation="relu"`` folds the ReLU into the forward store.  In a store-bound
kernel that is free (one compare and one select) and it removes an entire 2B
streaming pass.  Measured against ``F.relu(triton_group_norm(x))``: 39% off the forward
and 35% off fwd+bwd at ``[1,64,256^3]`` (6.83 -> 4.20 ms and 17.82 -> 11.61
ms), 38%/35% at ``[1,128,128^3]``, 34%/30% at ``[1,256,64^3]``, tapering to
21%/9% at ``[1,512,32^3]`` and below, where the call is host bound and there is
less streaming pass to remove.

The backward gates the incoming gradient on the sign of the **pre-activation**
value, which it *recomputes* from the saved ``(x, mean, rstd, weight, bias)``
using the identical expression the forward used.  Recomputation costs two FLOPs
on values already in registers and is bit-exact -- same inputs, same operation
order, same fp32 rounding -- so the sign always agrees with the forward's.  The
alternative, testing ``y > 0`` on the saved output, would need the output kept
alive *in addition to* ``x`` (which the GroupNorm backward needs regardless),
and in bf16/fp16 it would also mis-gate any element whose positive
pre-activation rounded to zero on the store.

Both the store and the gate are spelled as the *complement* of the usual test
(``tl.where(y <= 0, 0, y)``, ``tl.where(pre <= 0, 0, dy)``) rather than as
``tl.maximum(y, 0)`` / ``tl.where(pre > 0, dy, 0)``.  The two are identical on
every finite value but not on NaN: ``tl.maximum`` returns the *non*-NaN operand
and ``NaN > 0`` is False, so both of the usual spellings silently map a NaN to
0.0, while ``F.relu`` propagates it and ``threshold_backward(grad, result, 0)``
-- ReLU's real backward -- passes its gradient (``NaN <= 0`` is False too).
Matching ``F.relu`` here is not pedantry: a diverging run whose forward comes
back finite because the fused activation ate the NaN passes straight through
ScaFFold's non-finite-loss abort and checkpoints a broken model.  ``+-Inf`` and
``-0.0`` are bit-identical under either spelling (``-0.0`` flushes to ``+0.0``,
as ``F.relu`` does).  Cost: nil, measured -- see ``FastGroupNorm``'s tests and
``work/gn-dctensor/wiring-fixes``.

Composition
===========
Registered as real dispatcher ops (``scaffold_gn::group_norm`` /
``scaffold_gn::group_norm_backward``) via ``torch.library.custom_op``, with a
fake/meta kernel and ``register_autograd``.  Consequences:

* ``torch.compile(..., fullgraph=True)`` traces through without a graph break.
* Tensor subclasses that dispatch via ``__torch_dispatch__`` -- notably
  DistConv's ``DCTensor`` -- intercept the op, unwrap to the local shard, run
  it, and rewrap, so a DCTensor goes in and a DCTensor comes out with the graph
  intact.  As with the rest of DistConv today, statistics are per-shard.

Where the host time goes, and what is left
==========================================
Composing has a price, and at the launch-bound shapes it is now the *dominant*
cost.  Peeling the layers at ``[1,2048,8^3]``, steady-state wall clock per
fwd+bwd (median of 200, min of 7 rounds; GPU work is 0.030 ms)::

    kernels + this file's Python (_forward/_backward called directly)  0.145 ms
    + torch.library dispatcher (both custom ops)                      +0.054 ms
    + autograd (register_autograd node, save_for_backward, ctx)       +0.338 ms
    = triton_group_norm(x).backward(dy)                                0.537 ms

    for scale: an *empty* python torch.autograd.Function, fwd+bwd      0.065 ms

So **63% of the call is the autograd layer** and 10% is the dispatcher -- both
of them the cost of being a real dispatcher op that ``torch.compile`` and
``DCTensor`` can see, which is the whole point of registering it that way.  Of
the 0.145 ms this file is responsible for, 0.030 ms is GPU and the remaining
0.115 ms is four launches plus the allocations, plan lookup and argument
binding around them: launching every kernel twice measures the marginal cost of
a whole invocation site at **35 us**, so the four of them are ~0.14 ms of host
work that the GPU work does not cover.

Two things were considered for that 0.14 ms and rejected:

* **Bypassing ``JITFunction.run`` for a cached ``CompiledKernel`` handle**
  (8.68 us -> 3.94 us per launch on this node) would recover ~19 us, i.e. 3.3%
  of the call.  It buys that by asserting that Triton's specialization key --
  including 16-byte pointer alignment -- is a pure function of the shape.  It is
  not: this module accepts channels-last *views with a storage offset* and
  non-contiguous affine parameters, both of which the test suite exercises, and
  a stale specialization there is a wrong answer rather than a crash.  3.3% on
  the host-bound shapes only is not worth a silent-miscompute failure mode.
* **Caching the scratch buffers** across calls saves ~1.7 us per allocation,
  ~20 us here, and makes the buffers shared mutable state across call sites --
  correct on one stream, wrong on two, and this module has no way to know.

What that leaves: the kernels are at 95-98% of the streaming roofline at the
two largest shapes and the fused chain is 1.7-6.8% faster than the unfused one
there, so there is no meaningful GPU headroom left.  At the launch-bound
shapes the remaining 0.4 ms is torch's own plumbing, and the two ways to remove
it are both outside this file: CUDA-graph capture of the training step (which
takes the ten small scale-8 sites to ~1.05 ms/step of GPU time in total), or a
C++ autograd node.

Triton is imported lazily, on the first call that actually reaches the kernel,
so importing this module (or running the CPU test suite) costs nothing.
"""

import contextlib
import functools
import importlib.util
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "triton_group_norm",
    "is_supported",
    "select_strategy",
    "GNConfig",
    "default_config",
    "SUPPORTED_ACTIVATIONS",
    "TritonKernelError",
]


class TritonKernelError(RuntimeError):
    """A failure of the Triton kernels themselves, with the original as ``__cause__``.

    Raised in place of whatever ``_forward``/``_backward`` raised -- a missing or
    mismatched ``triton``, an unwritable JIT cache, a compile error, a launch
    failure, an API change between Triton releases.  It exists so that a caller
    with a fallback (``ScaFFold.unet.group_norm``'s ladder) can catch *exactly*
    "the kernel is broken" and nothing else, instead of catching ``Exception``
    and trying to enumerate every framework mechanism that legitimately raises
    through a forward -- saved-tensor pack hooks, ``torch.utils.checkpoint``'s
    recompute control flow, functorch, a user's offloading hook.

    Two things are deliberately *not* tagged and therefore propagate unchanged:

    * ``torch.OutOfMemoryError``, which is a resource condition rather than a
      defect (every fallback allocates an output of the same size, so retrying
      one is a second, differently-shaped OOM at a call site the caller did not
      ask about), and
    * the ``ValueError``s ``_validate`` raises, which are contract violations by
      the caller.  ``is_supported`` accepts exactly what ``_validate`` accepts,
      so a caller that branches on it can never see one; if one escapes, that
      is a bug in this module and must be loud.

    The tagged region contains no autograd-observable work -- allocations and
    kernel launches only, with ``save_for_backward`` happening in
    ``_setup_context`` strictly *after* ``_forward`` returns -- so an exception
    that carries this type is guaranteed to have been raised before the op saved
    anything.  That is what makes retrying the call on another kernel safe.
    """


#: The activations that may be fused into the forward store.
SUPPORTED_ACTIVATIONS = (None, "relu")

#: Input dtypes the kernels read directly (statistics are always fp32).
SUPPORTED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

#: Largest linear element index representable in int32.
_INT32_MAX = 2**31 - 1


# --------------------------------------------------------------------------- #
# tiling configuration
# --------------------------------------------------------------------------- #
class GNConfig:
    """Tiling knobs.  A pure function of the shape => bitwise determinism.

    ``stats_tile``/``elem_tile`` are *element* budgets per program (the spatial
    block is ``tile // channels_per_voxel``, rounded down to a power of two);
    ``nsplit_target`` is the total number of split-K partials wanted across the
    batch, so the per-sample split count is ``nsplit_target // N``;
    ``elem_progs`` caps the elementwise grid, each program then striding over
    ``ceil(nblk_elem / elem_progs)`` tiles (0 = one program per tile).

    These are **not** independent knobs, and in particular they stopped being
    independent when the finalize passes were folded into the elementwise
    kernels: each elementwise *program* now re-reads all ``nsplit`` split-K
    partials, so the redundant traffic is ``min(nblk_elem, elem_progs) *
    nsplit`` triples.  Raising ``nsplit_target`` for stats-kernel occupancy and
    lowering ``elem_progs`` for redundancy pull against each other and were
    tuned together; see the module docstring.
    """

    __slots__ = (
        "stats_tile",
        "stats_warps",
        "nsplit_target",
        "elem_tile",
        "elem_warps",
        "elem_progs",
    )

    def __init__(
        self,
        stats_tile=8192,
        stats_warps=4,
        nsplit_target=2048,
        elem_tile=8192,
        elem_warps=4,
        elem_progs=2048,
    ):
        self.stats_tile = stats_tile
        self.stats_warps = stats_warps
        self.nsplit_target = nsplit_target
        self.elem_tile = elem_tile
        self.elem_warps = elem_warps
        self.elem_progs = elem_progs

    def key(self):
        return (
            self.stats_tile,
            self.stats_warps,
            self.nsplit_target,
            self.elem_tile,
            self.elem_warps,
            self.elem_progs,
        )

    def __eq__(self, other):
        return isinstance(other, GNConfig) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return (
            "GNConfig(stats_tile=%d, stats_warps=%d, nsplit_target=%d, "
            "elem_tile=%d, elem_warps=%d, elem_progs=%d)" % self.key()
        )


#: Frozen tuning table, produced by coordinate descent on fwd+bwd time on one
#: MI300A (228 CUs) at fp32 with ``num_groups=8``, keyed by the
#: ``(num_channels, cube-root spatial extent)`` of the scale-8 ScaFFold UNet
#: GroupNorm sites plus the ``[1,4096,4^3]`` tail.  Frozen -- never autotuned at
#: run time -- because the split count fixes the reduction order and therefore
#: the bits of the result.
#:
#: Re-derived for the fused (2+2 launch) kernels: every candidate was timed
#: **interleaved against the incumbent** in one process, so that anything which
#: changes on the device partway through a sweep -- a neighbour process above
#: all -- lands on both arms at once instead of on whichever ran later.  The
#: table is keyed on ``(C, edge)`` and not on ``N``: ``nsplit_target`` is a target for
#: the split count *summed over the batch* (the per-sample count is
#: ``nsplit_target // N``), so the same entry serves ``N > 1`` with the same
#: total number of stats programs.  Verified at ``[2,1024,16^3]``,
#: ``[4,2048,8^3]`` and ``[2,256,64^3]``.
_TUNED = {
    (64, 256): GNConfig(16384, 4, 2048, 8192, 4, 2048),
    (128, 128): GNConfig(16384, 4, 2048, 16384, 4, 912),
    (256, 64): GNConfig(16384, 4, 512, 16384, 8, 912),
    (512, 32): GNConfig(32768, 8, 4096, 16384, 4, 0),
    (1024, 16): GNConfig(65536, 4, 8192, 16384, 4, 0),
    (2048, 8): GNConfig(16384, 8, 1024, 8192, 8, 228),
    (4096, 4): GNConfig(16384, 4, 32, 4096, 8, 0),
}

_DEFAULT_CONFIG = GNConfig()


def default_config(num_channels: int, spatial: int) -> GNConfig:
    """Tiling for ``num_channels`` channels and ``spatial = D*H*W`` voxels."""
    edge = round(spatial ** (1.0 / 3.0))
    if edge**3 != spatial:
        edge = None
    return _TUNED.get((num_channels, edge), _DEFAULT_CONFIG)


# --------------------------------------------------------------------------- #
# small-shape dispatch hook
# --------------------------------------------------------------------------- #
#: Every strategy name ``select_strategy`` may return.  Only ``"split_k"`` is
#: implemented; anything else raises rather than silently doing the wrong
#: thing.
STRATEGIES = ("split_k",)

#: Spatial extent (``D*H*W``) below which the split-K chain is host-dispatch
#: bound rather than bandwidth bound.  Measured on MI300A: at ``[1,2048,8^3]``
#: the kernels do 0.030 ms of GPU work behind ~0.58 ms of
#: Python/autograd/launch cost, and 0.086 ms of that is the *empty*
#: ``torch.autograd.Function`` wrapper -- i.e. 68% of the remaining call is
#: torch's plumbing, not this file's.  Purely informational --
#: ``select_strategy`` does not use it.
SMALL_SPATIAL_THRESHOLD = 4096


def select_strategy(n: int, num_channels: int, spatial: int, num_groups: int) -> str:
    """### SMALL-SHAPE DISPATCH HOOK ### -- the single point where a different
    kernel strategy is chosen for a shape.

    Returns a name from :data:`STRATEGIES`.  Today it always returns
    ``"split_k"``: two forward and two backward kernels with split-K partial
    reductions and the finalize passes fused into their consumers.  That is
    bandwidth-optimal for the large shapes and, after the fusion, within
    ~0.09 ms of the floor a Python ``autograd.Function`` can reach at the small
    ones -- so there is much less left here than there looks.  Below roughly
    ``SMALL_SPATIAL_THRESHOLD`` voxels the call is host bound, but the host cost
    is now dominated by autograd and the dispatcher rather than by launches:
    see "Why not one launch per pass" in the module docstring for the one
    strategy that *would* cut it further and why it is not here.

    If a second strategy ever lands, add its name to :data:`STRATEGIES`, return
    it from here on a rule that is a **pure function of the shape**
    (determinism depends on it), and branch on it in ``_dispatch`` -- which is
    the only caller, sits in front of the memoized tiling plan, and is itself
    called by both ``_forward`` and ``_backward``.  Nothing else in this file
    needs to change.
    """
    return "split_k"


# --------------------------------------------------------------------------- #
# planning helpers
# --------------------------------------------------------------------------- #
def _prev_pow2(x: int) -> int:
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class _Plan:
    """Everything the launcher needs, derived only from the shape + config."""

    __slots__ = (
        "n",
        "channels",
        "spatial",
        "groups",
        "group_channels",
        "groups_p2",
        "group_channels_p2",
        "masked_c",
        "int64",
        "block_s_stats",
        "nsplit",
        "chunk",
        "block_s_elem",
        "nblk_elem",
        "nprog_elem",
        "elements_per_group",
        "dwdb_rows",
        "dwdb_block_c",
        "dwdb_block_r",
        "dwdb_progs",
        "grid_dx",
        "zero_dx",
        "cfg",
    )

    def __init__(self, n, channels, spatial, groups, cfg, numel):
        self.n = n
        self.channels = channels
        self.spatial = spatial
        self.groups = groups
        self.group_channels = channels // groups
        self.groups_p2 = _next_pow2(groups)
        self.group_channels_p2 = _next_pow2(self.group_channels)
        # Only power-of-two group/channel counts tile the (G, C/G) axes exactly;
        # anything else is rounded up and masked, which is correct but reads a
        # few lanes it throws away.
        self.masked_c = (
            self.groups_p2 != groups or self.group_channels_p2 != self.group_channels
        )
        # int64 addressing is needed once a linear element index can exceed
        # INT32_MAX.  [2,64,256^3] is exactly 2^31 elements, so this is not
        # hypothetical at scale.  Only the *scalar* tile base is widened (see
        # the kernels), which measurement showed to be free.
        self.int64 = numel + channels > _INT32_MAX
        self.cfg = cfg

        voxel = self.groups_p2 * self.group_channels_p2
        self.block_s_stats = max(1, _prev_pow2(cfg.stats_tile // max(1, voxel)))
        ntiles = max(1, spatial // self.block_s_stats)
        self.nsplit = _prev_pow2(
            max(1, min(ntiles, max(1, cfg.nsplit_target // max(1, n))))
        )
        self.chunk = _cdiv(spatial, self.nsplit)
        self.block_s_elem = max(1, _prev_pow2(cfg.elem_tile // max(1, voxel)))
        self.nblk_elem = _cdiv(spatial, self.block_s_elem)
        # Grid cap for the two elementwise kernels; each program then strides
        # over its share of the tiles.  Bounds the cost of the fused finalize,
        # which every *program* pays once.
        self.nprog_elem = (
            self.nblk_elem
            if cfg.elem_progs <= 0
            else min(self.nblk_elem, cfg.elem_progs)
        )
        self.elements_per_group = float(spatial * self.group_channels)
        # Everything the fused dweight/dbias reduction in _dx_kernel needs.
        # Precomputed rather than derived per call: the launch-bound shapes pay
        # every Python statement in _backward, and _next_pow2 is a loop.
        self.dwdb_rows = n * self.nsplit
        self.dwdb_block_c = min(256, max(64, _next_pow2(channels)))
        self.dwdb_block_r = 32 if self.dwdb_rows >= 32 else 1
        self.dwdb_progs = _cdiv(channels, self.dwdb_block_c)
        # Programs past nprog_elem run no elementwise loop iterations; they
        # exist only when there are more dweight/dbias blocks than tiles.
        self.grid_dx = max(self.nprog_elem, self.dwdb_progs)
        self.zero_dx = self.group_channels * spatial == 1


@functools.lru_cache(maxsize=256)
def _plan(n, channels, spatial, groups, numel) -> _Plan:
    """Memoized: a UNet presents a handful of shapes, and the three smallest
    GroupNorm sites are host-dispatch bound, so rebuilding the plan (two
    power-of-two loops and a dict lookup) on every call is measurable there.
    Memoization cannot affect results -- the plan is a pure function of its
    arguments, which is also what makes the kernels bitwise reproducible."""
    return _Plan(n, channels, spatial, groups, default_config(channels, spatial), numel)


def _dispatch(n, channels, spatial, groups, numel) -> _Plan:
    """Consult the strategy hook, then build (or reuse) the tiling plan."""
    strategy = select_strategy(n, channels, spatial, groups)
    if strategy != "split_k":
        raise NotImplementedError(
            f"kernel strategy {strategy!r} selected by select_strategy() is not "
            f"implemented; known strategies are {STRATEGIES}"
        )
    return _plan(n, channels, spatial, groups, numel)


# --------------------------------------------------------------------------- #
# Triton kernels (built lazily -- importing this module must not import triton)
# --------------------------------------------------------------------------- #
triton = None
tl = None
_welford_combine = None
_stats_partial_kernel = None
_normalize_kernel = None
_bwd_partial_kernel = None
_dx_kernel = None


_TRITON_AVAILABLE = None


def triton_available() -> bool:
    """Whether Triton is importable, without importing it.

    ``find_spec`` on a top-level name only touches the finders, so this stays
    side-effect free and is safe to call from :func:`is_supported`.  The answer
    is memoized in a plain global rather than an ``lru_cache`` because Dynamo
    warns (loudly, once per process) when it traces through a cache wrapper.
    """
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            _TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
        except (ImportError, ValueError):
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


def _build_kernels():
    """Import Triton and install the JIT kernels into this module's globals.

    The kernels are defined inside a function purely so that ``import triton``
    is deferred to the first GPU call; they are written into ``globals()`` so
    Triton's name resolution (which reads ``fn.__globals__``) sees them.
    """
    global triton, tl
    import triton as _triton
    import triton.language as _tl

    triton = _triton
    tl = _tl

    # ---------------------------------------------------------------- stats --
    @_triton.jit
    def _welford_combine(cnt_a, mean_a, m2_a, cnt_b, mean_b, m2_b):
        """Chan's parallel merge of two (count, mean, M2) triples.

        Exact for empty partials on either side (``cnt == 0`` leaves the other
        triple untouched), which matters because the last split of a shape
        whose spatial extent is not a multiple of the chunk size can be empty.
        """
        cnt = cnt_a + cnt_b
        denom = tl.where(cnt == 0.0, 1.0, cnt)
        delta = mean_b - mean_a
        mean = mean_a + delta * (cnt_b / denom)
        m2 = m2_a + m2_b + delta * delta * (cnt_a * cnt_b / denom)
        return cnt, mean, m2

    @_triton.jit
    def _stats_partial_kernel(
        X,
        PCNT,
        PMEAN,
        PM2,
        S,
        CHUNK,
        C: tl.constexpr,
        G: tl.constexpr,
        CG: tl.constexpr,
        GP: tl.constexpr,
        CGP: tl.constexpr,
        NSPLIT: tl.constexpr,
        BLOCK_S: tl.constexpr,
        MASKED_C: tl.constexpr,
        INT64: tl.constexpr,
    ):
        """One program per ``(split, n)``: Welford partials for every group.

        Reads a dense ``(BLOCK_S, C)`` run of memory per step -- perfectly
        coalesced -- and produces the statistics of all G groups at once, the
        group axis being the inner channel axis reshaped to ``(G, C/G)``.
        """
        sp = tl.program_id(0)
        n = tl.program_id(1)

        offs_g = tl.arange(0, GP)
        offs_j = tl.arange(0, CGP)
        offs_s = tl.arange(0, BLOCK_S)
        inner = offs_g[None, :, None] * CG + offs_j[None, None, :]
        cmask = (offs_g[None, :, None] < G) & (offs_j[None, None, :] < CG)
        off = offs_s[:, None, None] * C + inner

        s_begin = sp * CHUNK
        s_end = tl.minimum(s_begin + CHUNK, S)

        cnt = tl.zeros((GP,), dtype=tl.float32)
        mean = tl.zeros((GP,), dtype=tl.float32)
        m2 = tl.zeros((GP,), dtype=tl.float32)

        for s0 in range(s_begin, s_end, BLOCK_S):
            # Only the scalar tile base is ever widened to int64; the vector
            # offsets stay int32 because they span at most BLOCK_S*C+C
            # elements.  That keeps the wide arithmetic off the hot path.
            if INT64:
                base = (n.to(tl.int64) * S + s0) * C
            else:
                base = (n * S + s0) * C
            nvalid = tl.minimum(BLOCK_S, s_end - s0)
            m = tl.broadcast_to((offs_s < nvalid)[:, None, None], (BLOCK_S, GP, CGP))
            if MASKED_C:
                m = m & cmask
            # `other` is not load-bearing for any *bounded* value: `cnt_t`
            # counts only the valid lanes, so `corr` below evaluates to
            # `sum_valid(x)/cnt_t - mean0` and `mean_t = mean0 + corr` is the
            # true mean whatever the masked lanes contributed, while `d`/`dd`
            # are re-masked before they reach `m2_t`.  (Bounded: `other=1e30`
            # would swamp `mean0` and the correction with it, and `other=inf`
            # or `nan` would poison it outright.)  0.0 is kept because it is
            # the value that survives all three of those, not because the
            # cancellation is something to rely on.
            x = tl.load(X + base + off, mask=m, other=0.0).to(tl.float32)

            # Corrected two-pass within the tile: the first mean loses digits
            # to the magnitude of the data, `corr` puts them back, and the
            # centred squares are then accurate to fp32 roundoff.  Everything
            # here is register traffic; the tile is read from HBM exactly once.
            # `corr` is worth 1.3x to 1472x on `rstd` once the data's mean
            # dominates its spread -- see "What the third pass is worth" in the
            # module docstring for the measurements, for the one regime where
            # it goes the other way, and for why the *output* error barely
            # moves even where `rstd` improves by three orders of magnitude.
            cnt_t = (nvalid * CG).to(tl.float32)
            mean0 = tl.sum(tl.sum(x, 2), 0) / cnt_t
            d = tl.where(m, x - mean0[None, :, None], 0.0)
            corr = tl.sum(tl.sum(d, 2), 0) / cnt_t
            dd = tl.where(m, d - corr[None, :, None], 0.0)
            m2_t = tl.sum(tl.sum(dd * dd, 2), 0)
            mean_t = mean0 + corr

            new_cnt = cnt + cnt_t
            delta = mean_t - mean
            mean = mean + delta * (cnt_t / new_cnt)
            m2 = m2 + m2_t + delta * delta * (cnt * cnt_t / new_cnt)
            cnt = new_cnt

        o = (n * NSPLIT + sp) * G + offs_g
        gm = offs_g < G
        tl.store(PCNT + o, cnt, mask=gm)
        tl.store(PMEAN + o, mean, mask=gm)
        tl.store(PM2 + o, m2, mask=gm)

    # ------------------------------------------------------------ normalize --
    @_triton.jit
    def _normalize_kernel(
        X,
        Y,
        PCNT,
        PMEAN,
        PM2,
        MEAN,
        RSTD,
        W,
        B,
        S,
        M,
        eps,
        C: tl.constexpr,
        G: tl.constexpr,
        CG: tl.constexpr,
        GP: tl.constexpr,
        CGP: tl.constexpr,
        NSPLIT: tl.constexpr,
        BLOCK_S: tl.constexpr,
        NBLK: tl.constexpr,
        NPROG: tl.constexpr,
        RELU: tl.constexpr,
        HAS_W: tl.constexpr,
        HAS_B: tl.constexpr,
        MASKED_C: tl.constexpr,
        INT64: tl.constexpr,
    ):
        """Finalize the split-K statistics, then normalize NBLK/NPROG tiles.

        The finalize is *recomputed by every program* rather than round-tripped
        through its own kernel launch: merging NSPLIT Welford triples is a few
        KB of L2-resident traffic and a tree reduction over a ``(NSPLIT, GP)``
        tile, which is cheaper than the ~9 us launch it replaces.  What it is
        *not* cheap enough for is being paid once per tile at the largest
        shapes, where the flat grid is 10^5 programs: the grid is therefore
        capped at ``NPROG`` and each program strides over its share of the
        ``NBLK`` tiles, so the redundant read costs ``NPROG * NSPLIT`` and not
        ``NBLK * NSPLIT``.  See :class:`GNConfig` -- ``nsplit_target``,
        ``elem_tile`` and ``elem_progs`` are one joint tuning problem, not
        three independent knobs.

        Every program reads the same partials with the same tile shape, so they
        all get bit-identical ``mean``/``rstd``; program 0 stores them for the
        backward.  The loop carries nothing across iterations, so the striding
        cannot affect the result.
        """
        pid = tl.program_id(0)
        n = tl.program_id(1)

        offs_g = tl.arange(0, GP)
        offs_j = tl.arange(0, CGP)
        offs_s = tl.arange(0, BLOCK_S)
        inner = offs_g[None, :, None] * CG + offs_j[None, None, :]
        off = offs_s[:, None, None] * C + inner
        wb = offs_g[:, None] * CG + offs_j[None, :]
        wbm = (offs_g[:, None] < G) & (offs_j[None, :] < CG)

        gm = offs_g < G
        offs_p = tl.arange(0, NSPLIT)
        pidx = (n * NSPLIT + offs_p[:, None]) * G + offs_g[None, :]
        pm = tl.broadcast_to(gm[None, :], (NSPLIT, GP))
        # Padded group lanes load cnt == 0, which _welford_combine treats as the
        # identity, so they merge to (0, 0, 0) and are masked off on the store.
        # Reduced over axis 0 -- the *slowest* axis -- deliberately: reducing
        # the fastest axis of a 2-D tile makes Triton stage the whole tile
        # through LDS, which for a (G, NSPLIT) tile is 64 KB per array.
        cnt_p = tl.load(PCNT + pidx, mask=pm, other=0.0)
        _cnt, mu, m2 = tl.reduce(
            (
                cnt_p,
                tl.load(PMEAN + pidx, mask=pm, other=0.0),
                tl.load(PM2 + pidx, mask=pm, other=0.0),
            ),
            0,
            _welford_combine,
        )
        # `_cnt` equals M by construction; M is passed in so the divisor is the
        # exact element count rather than a float accumulated from partials.
        rs = 1.0 / tl.sqrt(m2 / M + eps)
        if pid == 0:
            tl.store(MEAN + n * G + offs_g, mu, mask=gm)
            tl.store(RSTD + n * G + offs_g, rs, mask=gm)
        mean = mu[None, :, None]
        rstd = rs[None, :, None]
        if HAS_W:
            w = tl.load(W + wb, mask=wbm, other=0.0).to(tl.float32)[None, :, :]
        else:
            w = tl.full((1, GP, CGP), 1.0, tl.float32)
        if HAS_B:
            b = tl.load(B + wb, mask=wbm, other=0.0).to(tl.float32)[None, :, :]
        else:
            b = tl.zeros((1, GP, CGP), dtype=tl.float32)

        for blk in tl.range(pid, NBLK, NPROG):
            s0 = blk * BLOCK_S
            m = tl.broadcast_to((offs_s < S - s0)[:, None, None], (BLOCK_S, GP, CGP))
            if MASKED_C:
                m = m & ((offs_g[None, :, None] < G) & (offs_j[None, None, :] < CG))
            if INT64:
                base = (n.to(tl.int64) * S + s0) * C
            else:
                base = (n * S + s0) * C
            x = tl.load(X + base + off, mask=m, other=0.0).to(tl.float32)
            xhat = (x - mean) * rstd
            y = xhat * w + b
            if RELU:
                # `tl.maximum(y, 0.0)` and `tl.where(y > 0, y, 0.0)` both map NaN
                # to 0.0 (the first returns the non-NaN operand, the second
                # because `NaN > 0` is False), while `F.relu` propagates it.
                # Testing the *complement* keeps NaN on the pass-through side:
                # `NaN <= 0` is also False, so NaN falls to `y`.  Bit-identical
                # to `F.relu` on NaN, +-Inf and -0.0 (which both flush to +0.0),
                # for one comparison and one select -- see the module docstring.
                y = tl.where(y <= 0.0, 0.0, y)
            tl.store(Y + base + off, y.to(Y.dtype.element_ty), mask=m)

    # ------------------------------------------------------------- backward --
    @_triton.jit
    def _bwd_partial_kernel(
        X,
        DY,
        MEAN,
        RSTD,
        W,
        B,
        PS1,
        PS2,
        PDW,
        PDB,
        S,
        CHUNK,
        C: tl.constexpr,
        G: tl.constexpr,
        CG: tl.constexpr,
        GP: tl.constexpr,
        CGP: tl.constexpr,
        NSPLIT: tl.constexpr,
        BLOCK_S: tl.constexpr,
        RELU: tl.constexpr,
        HAS_W: tl.constexpr,
        HAS_B: tl.constexpr,
        MASKED_C: tl.constexpr,
        INT64: tl.constexpr,
    ):
        """Partials for the two per-``(n, g)`` reductions used by dx, and for
        the per-channel dweight / dbias reductions."""
        sp = tl.program_id(0)
        n = tl.program_id(1)

        offs_g = tl.arange(0, GP)
        offs_j = tl.arange(0, CGP)
        offs_s = tl.arange(0, BLOCK_S)
        inner = offs_g[None, :, None] * CG + offs_j[None, None, :]
        cmask = (offs_g[None, :, None] < G) & (offs_j[None, None, :] < CG)
        off = offs_s[:, None, None] * C + inner
        wb = offs_g[:, None] * CG + offs_j[None, :]
        wbm = (offs_g[:, None] < G) & (offs_j[None, :] < CG)

        s_begin = sp * CHUNK
        s_end = tl.minimum(s_begin + CHUNK, S)

        gm = offs_g < G
        mean = tl.load(MEAN + n * G + offs_g, mask=gm, other=0.0)[None, :, None]
        rstd = tl.load(RSTD + n * G + offs_g, mask=gm, other=0.0)[None, :, None]
        if HAS_W:
            w = tl.load(W + wb, mask=wbm, other=0.0).to(tl.float32)[None, :, :]
        else:
            w = tl.full((1, GP, CGP), 1.0, tl.float32)
        if HAS_B:
            b = tl.load(B + wb, mask=wbm, other=0.0).to(tl.float32)[None, :, :]
        else:
            b = tl.zeros((1, GP, CGP), dtype=tl.float32)

        acc1 = tl.zeros((GP,), dtype=tl.float32)
        acc2 = tl.zeros((GP,), dtype=tl.float32)
        accdw = tl.zeros((GP, CGP), dtype=tl.float32)
        accdb = tl.zeros((GP, CGP), dtype=tl.float32)

        for s0 in range(s_begin, s_end, BLOCK_S):
            if INT64:
                base = (n.to(tl.int64) * S + s0) * C
            else:
                base = (n * S + s0) * C
            nvalid = tl.minimum(BLOCK_S, s_end - s0)
            m = tl.broadcast_to((offs_s < nvalid)[:, None, None], (BLOCK_S, GP, CGP))
            if MASKED_C:
                m = m & cmask
            x = tl.load(X + base + off, mask=m, other=0.0).to(tl.float32)
            dy = tl.load(DY + base + off, mask=m, other=0.0).to(tl.float32)
            xhat = (x - mean) * rstd
            if RELU:
                # Identical expression (and therefore identical rounding) to
                # the forward's pre-activation, so the sign test agrees with
                # the forward bit for bit.  Masked lanes carry dy == 0, so
                # gating cannot resurrect them.  Spelled as the *complement*
                # (`pre <= 0` zeroes) rather than `pre > 0` passes, so that a
                # NaN pre-activation passes the gradient through: that is what
                # `threshold_backward(grad, result, 0)` -- ReLU's real backward
                # -- does, since `NaN <= 0` is False.  See the forward store.
                dy = tl.where(xhat * w + b <= 0.0, 0.0, dy)
            dyw = dy * w
            acc1 += tl.sum(tl.sum(dyw, 2), 0)
            acc2 += tl.sum(tl.sum(dyw * xhat, 2), 0)
            accdw += tl.sum(dy * xhat, 0)
            accdb += tl.sum(dy, 0)

        o = (n * NSPLIT + sp) * G + offs_g
        tl.store(PS1 + o, acc1, mask=gm)
        tl.store(PS2 + o, acc2, mask=gm)
        row = (n * NSPLIT + sp) * C + wb
        tl.store(PDW + row, accdw, mask=wbm)
        tl.store(PDB + row, accdb, mask=wbm)

    @_triton.jit
    def _dx_kernel(
        X,
        DY,
        DX,
        MEAN,
        RSTD,
        W,
        B,
        PS1,
        PS2,
        PDW,
        PDB,
        DW,
        DB,
        ROWS,
        S,
        M,
        C: tl.constexpr,
        G: tl.constexpr,
        CG: tl.constexpr,
        GP: tl.constexpr,
        CGP: tl.constexpr,
        NSPLIT: tl.constexpr,
        BLOCK_S: tl.constexpr,
        NBLK: tl.constexpr,
        NPROG: tl.constexpr,
        NDW: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_R: tl.constexpr,
        RELU: tl.constexpr,
        HAS_W: tl.constexpr,
        HAS_B: tl.constexpr,
        MASKED_C: tl.constexpr,
        INT64: tl.constexpr,
        ZERO_DX: tl.constexpr,
    ):
        """The whole backward tail: dweight/dbias, the c1/c2 finalize, and dx.

        Two reductions that used to be their own launches ride along here.  The
        per-channel dweight/dbias row reduction is done by the first ``NDW``
        programs of ``n == 0`` (a single pass over an ``(n*nsplit, C)`` scratch,
        i.e. a few hundred KB); the per-``(n, g)`` c1/c2 finalize is recomputed
        redundantly by every program, once, before the tile loop -- exactly as
        in ``_normalize_kernel``, and capped the same way.  The grid is
        ``(max(NPROG, NDW), n)``; programs past ``NPROG`` exist only to cover
        the dweight/dbias rows and run no loop iterations.
        """
        pid = tl.program_id(0)
        n = tl.program_id(1)

        # ---- dweight / dbias: rows of the split-K scratch, once per channel --
        if n == 0:
            if pid < NDW:
                offs_c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
                mc = offs_c < C
                accw = tl.zeros((BLOCK_C,), dtype=tl.float32)
                accb = tl.zeros((BLOCK_C,), dtype=tl.float32)
                for r0 in range(0, ROWS, BLOCK_R):
                    offs_r = r0 + tl.arange(0, BLOCK_R)
                    rm = (offs_r[:, None] < ROWS) & mc[None, :]
                    roff = offs_r[:, None] * C + offs_c[None, :]
                    accw += tl.sum(tl.load(PDW + roff, mask=rm, other=0.0), 0)
                    accb += tl.sum(tl.load(PDB + roff, mask=rm, other=0.0), 0)
                tl.store(DW + offs_c, accw, mask=mc)
                tl.store(DB + offs_c, accb, mask=mc)

        offs_g = tl.arange(0, GP)
        offs_j = tl.arange(0, CGP)
        offs_s = tl.arange(0, BLOCK_S)
        inner = offs_g[None, :, None] * CG + offs_j[None, None, :]
        off = offs_s[:, None, None] * C + inner
        wb = offs_g[:, None] * CG + offs_j[None, :]
        wbm = (offs_g[:, None] < G) & (offs_j[None, :] < CG)

        if ZERO_DX:
            # One element per group: mean == x and var == 0 identically, so
            # xhat is the constant 0 and y does not depend on x at all -- the
            # exact d_input is zero everywhere.  The expression below would
            # instead return rstd * (dy*w - c1), and since the compiler
            # contracts that to fma(dy, w, -c1) while c1 was accumulated from
            # the *rounded* product, what survives is the product's rounding
            # error amplified by rstd = 1/sqrt(eps) ~ 316 (2.2e-05 at
            # eps=1e-5).  Answering with the exact zero costs one constexpr.
            zero = tl.zeros((BLOCK_S, GP, CGP), dtype=tl.float32)
            for blk in tl.range(pid, NBLK, NPROG):
                s0 = blk * BLOCK_S
                m = tl.broadcast_to(
                    (offs_s < S - s0)[:, None, None], (BLOCK_S, GP, CGP)
                )
                if MASKED_C:
                    m = m & ((offs_g[None, :, None] < G) & (offs_j[None, None, :] < CG))
                if INT64:
                    base = (n.to(tl.int64) * S + s0) * C
                else:
                    base = (n * S + s0) * C
                tl.store(DX + base + off, zero.to(DX.dtype.element_ty), mask=m)
        else:
            gm = offs_g < G
            offs_p = tl.arange(0, NSPLIT)
            pidx = (n * NSPLIT + offs_p[:, None]) * G + offs_g[None, :]
            pm = tl.broadcast_to(gm[None, :], (NSPLIT, GP))
            c1 = (tl.sum(tl.load(PS1 + pidx, mask=pm, other=0.0), 0) / M)[None, :, None]
            c2 = (tl.sum(tl.load(PS2 + pidx, mask=pm, other=0.0), 0) / M)[None, :, None]

            mean = tl.load(MEAN + n * G + offs_g, mask=gm, other=0.0)[None, :, None]
            rstd = tl.load(RSTD + n * G + offs_g, mask=gm, other=0.0)[None, :, None]
            if HAS_W:
                w = tl.load(W + wb, mask=wbm, other=0.0).to(tl.float32)[None, :, :]
            else:
                w = tl.full((1, GP, CGP), 1.0, tl.float32)
            if HAS_B:
                b = tl.load(B + wb, mask=wbm, other=0.0).to(tl.float32)[None, :, :]
            else:
                b = tl.zeros((1, GP, CGP), dtype=tl.float32)

            for blk in tl.range(pid, NBLK, NPROG):
                s0 = blk * BLOCK_S
                m = tl.broadcast_to(
                    (offs_s < S - s0)[:, None, None], (BLOCK_S, GP, CGP)
                )
                if MASKED_C:
                    m = m & ((offs_g[None, :, None] < G) & (offs_j[None, None, :] < CG))
                if INT64:
                    base = (n.to(tl.int64) * S + s0) * C
                else:
                    base = (n * S + s0) * C
                x = tl.load(X + base + off, mask=m, other=0.0).to(tl.float32)
                dy = tl.load(DY + base + off, mask=m, other=0.0).to(tl.float32)
                xhat = (x - mean) * rstd
                if RELU:
                    # Same complement spelling as _bwd_partial_kernel: a NaN
                    # pre-activation must pass the gradient, exactly as
                    # `threshold_backward(grad, result, 0)` does.
                    dy = tl.where(xhat * w + b <= 0.0, 0.0, dy)
                dyw = dy * w
                dx = rstd * (dyw - c1 - xhat * c2)
                tl.store(DX + base + off, dx.to(DX.dtype.element_ty), mask=m)

    globals().update(
        _welford_combine=_welford_combine,
        _stats_partial_kernel=_stats_partial_kernel,
        _normalize_kernel=_normalize_kernel,
        _bwd_partial_kernel=_bwd_partial_kernel,
        _dx_kernel=_dx_kernel,
    )


def _ensure_kernels():
    if _stats_partial_kernel is None:
        _build_kernels()


# --------------------------------------------------------------------------- #
# python drivers
# --------------------------------------------------------------------------- #
_CL_FORMAT = torch.channels_last_3d

#: Reused so the common (already-current device) path allocates nothing.
_NO_GUARD = contextlib.nullcontext()


def _device_guard(device: torch.device):
    """Make ``device`` current for the kernel launches inside the ``with``.

    A Triton launch goes to whatever device is *current*, not to the device the
    argument tensors live on, so without this a tensor on ``cuda:1`` while
    ``cuda:0`` is current makes the kernel dereference another device's pointers
    and the process dies with ``Memory access fault by GPU node-N``.  ATen ops
    (including ``F.group_norm``) carry a ``DeviceGuard`` and handle the same
    call, so this is required for the drop-in contract, not a nicety.

    The ``current_device()`` test is not about correctness but about *cost*.
    Measured on this node (median of 200k calls, torch 2.13.0+rocm7.2):
    ``with torch.cuda.device(t.device)`` is **1.55 us** of host time per call,
    ``with torch.cuda._DeviceGuard(t.device.index)`` **0.61 us**, and this
    helper **0.51 us** when the tensor is already on the current device --
    which it is on every ScaFFold call, since ScaFFold pins one device per
    rank.  Two of those (forward + backward) against the 0.65 ms fwd+bwd of the
    two smallest scale-8 shapes, which are host-dispatch bound, is 0.16%
    instead of 0.48%.
    """
    if device.index == torch.cuda.current_device():
        return _NO_GUARD
    return torch.cuda.device(device)


def _shape_of(input: torch.Tensor):
    n, channels = input.shape[0], input.shape[1]
    spatial = 1
    for d in input.shape[2:]:
        spatial *= d
    return n, channels, spatial


def _tag_kernel_failures(fn):
    """Re-raise anything ``fn`` raises as :class:`TritonKernelError`.

    Applied to the two functions that do nothing but import Triton, allocate
    scratch and launch kernels.  The region is *closed*: it runs no
    autograd-observable op, so a blanket ``except Exception`` here cannot
    swallow framework control flow the way one at the call site would -- there
    is no pack hook, no recompute stop and no functorch layer inside it.  That
    closure is what lets the caller's fallback ladder use a one-element
    allowlist instead of an ever-growing denylist.

    ``torch.OutOfMemoryError`` is passed through untagged; see
    :class:`TritonKernelError`.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except torch.OutOfMemoryError:
            raise
        except TritonKernelError:
            raise
        except Exception as e:
            raise TritonKernelError(
                f"{fn.__name__} failed ({type(e).__name__}: {e})"
            ) from e

    return wrapper


@_tag_kernel_failures
def _forward(input, num_groups, weight, bias, eps, activation, out_dtype):
    _ensure_kernels()
    n, channels, spatial = _shape_of(input)
    plan = _dispatch(n, channels, spatial, num_groups, input.numel())
    groups = num_groups
    device = input.device

    with _device_guard(device):
        pcnt = torch.empty(n * plan.nsplit * groups, device=device, dtype=torch.float32)
        pmean = torch.empty_like(pcnt)
        pm2 = torch.empty_like(pcnt)
        mean = torch.empty((n, groups), device=device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        out = torch.empty_like(input, dtype=out_dtype, memory_format=_CL_FORMAT)

        _stats_partial_kernel[(plan.nsplit, n)](
            input,
            pcnt,
            pmean,
            pm2,
            spatial,
            plan.chunk,
            C=channels,
            G=groups,
            CG=plan.group_channels,
            GP=plan.groups_p2,
            CGP=plan.group_channels_p2,
            NSPLIT=plan.nsplit,
            BLOCK_S=plan.block_s_stats,
            MASKED_C=plan.masked_c,
            INT64=plan.int64,
            num_warps=plan.cfg.stats_warps,
        )
        _normalize_kernel[(plan.nprog_elem, n)](
            input,
            out,
            pcnt,
            pmean,
            pm2,
            mean,
            rstd,
            weight,
            bias,
            spatial,
            plan.elements_per_group,
            eps,
            C=channels,
            G=groups,
            CG=plan.group_channels,
            GP=plan.groups_p2,
            CGP=plan.group_channels_p2,
            NSPLIT=plan.nsplit,
            BLOCK_S=plan.block_s_elem,
            NBLK=plan.nblk_elem,
            NPROG=plan.nprog_elem,
            RELU=activation == "relu",
            HAS_W=weight is not None,
            HAS_B=bias is not None,
            MASKED_C=plan.masked_c,
            INT64=plan.int64,
            num_warps=plan.cfg.elem_warps,
        )
    return out, mean, rstd


@_tag_kernel_failures
def _backward(grad_out, input, weight, bias, mean, rstd, num_groups, activation):
    _ensure_kernels()
    n, channels, spatial = _shape_of(input)
    plan = _dispatch(n, channels, spatial, num_groups, input.numel())
    groups = num_groups
    device = input.device

    with _device_guard(device):
        ps1 = torch.empty(n * plan.nsplit * groups, device=device, dtype=torch.float32)
        ps2 = torch.empty_like(ps1)
        pdw = torch.empty(
            n * plan.nsplit * channels, device=device, dtype=torch.float32
        )
        pdb = torch.empty_like(pdw)

        _bwd_partial_kernel[(plan.nsplit, n)](
            input,
            grad_out,
            mean,
            rstd,
            weight,
            bias,
            ps1,
            ps2,
            pdw,
            pdb,
            spatial,
            plan.chunk,
            C=channels,
            G=groups,
            CG=plan.group_channels,
            GP=plan.groups_p2,
            CGP=plan.group_channels_p2,
            NSPLIT=plan.nsplit,
            BLOCK_S=plan.block_s_stats,
            RELU=activation == "relu",
            HAS_W=weight is not None,
            HAS_B=bias is not None,
            MASKED_C=plan.masked_c,
            INT64=plan.int64,
            num_warps=plan.cfg.stats_warps,
        )

        d_weight = torch.empty(channels, device=device, dtype=torch.float32)
        d_bias = torch.empty_like(d_weight)
        d_input = torch.empty_like(input, memory_format=_CL_FORMAT)
        _dx_kernel[(plan.grid_dx, n)](
            input,
            grad_out,
            d_input,
            mean,
            rstd,
            weight,
            bias,
            ps1,
            ps2,
            pdw,
            pdb,
            d_weight,
            d_bias,
            plan.dwdb_rows,
            spatial,
            plan.elements_per_group,
            C=channels,
            G=groups,
            CG=plan.group_channels,
            GP=plan.groups_p2,
            CGP=plan.group_channels_p2,
            NSPLIT=plan.nsplit,
            BLOCK_S=plan.block_s_elem,
            NBLK=plan.nblk_elem,
            NPROG=plan.nprog_elem,
            NDW=plan.dwdb_progs,
            BLOCK_C=plan.dwdb_block_c,
            BLOCK_R=plan.dwdb_block_r,
            RELU=activation == "relu",
            HAS_W=weight is not None,
            HAS_B=bias is not None,
            MASKED_C=plan.masked_c,
            INT64=plan.int64,
            ZERO_DX=plan.zero_dx,
            num_warps=plan.cfg.elem_warps,
        )
    return d_input, d_weight, d_bias


# --------------------------------------------------------------------------- #
# torch.library registration
# --------------------------------------------------------------------------- #
def _one_value_per_channel(input, num_groups: int) -> bool:
    """Whether ``F.group_norm`` would reject this shape as degenerate.

    ``F.group_norm`` runs ``_verify_batch_size([N*C//G, G, *spatial])``, which
    raises ``ValueError("Expected more than 1 value per channel when
    training")`` exactly when ``N * (C/G) * D*H*W == 1``.  All three factors are
    positive, so that holds iff ``N == 1``, ``C == num_groups`` and the spatial
    extent is 1 -- i.e. iff ``numel == C == num_groups``, which is the cheap
    form used here (``numel`` is wanted by the caller anyway).

    Rejected rather than served: the kernel *can* compute it (it returns
    ``bias``, since every group has zero variance), but a caller that branches
    on :func:`is_supported` would then get a result where the op this replaces
    raises, which is a worse failure than being slower.
    """
    channels = input.shape[1]
    return channels == num_groups and input.numel() == channels


def _validate(input, num_groups, weight, bias, activation):
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {SUPPORTED_ACTIVATIONS}, got {activation!r}"
        )
    if input.dim() != 5:
        raise ValueError(f"expected a 5-D NCDHW tensor, got {tuple(input.shape)}")
    if num_groups <= 0 or input.shape[1] % num_groups != 0:
        raise ValueError(
            f"num_channels={input.shape[1]} is not divisible by num_groups={num_groups}"
        )
    if _one_value_per_channel(input, num_groups):
        # Same rejection, and the same exception type, as F.group_norm's
        # _verify_batch_size; see _one_value_per_channel.
        raise ValueError(
            f"Expected more than 1 value per channel when training, got input "
            f"size {tuple(input.shape)} with num_groups={num_groups}"
        )
    if input.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported input dtype {input.dtype}")
    if not input.is_contiguous(memory_format=_CL_FORMAT):
        # Required, not converted: the fake kernel promises the *input's*
        # memory format for the output, so silently converting here would make
        # the traced and eager results disagree on strides.  The public
        # ``triton_group_norm`` routes non-channels-last input to
        # ``F.group_norm`` before it ever reaches this op.
        raise ValueError(
            "input must be channels_last_3d-contiguous; use triton_group_norm() "
            "which falls back to F.group_norm for other layouts"
        )
    for name, t in (("weight", weight), ("bias", bias)):
        if t is not None and t.numel() != input.shape[1]:
            raise ValueError(
                f"{name} has {t.numel()} elements, expected {input.shape[1]}"
            )


@torch.library.custom_op(
    "scaffold_gn::group_norm", mutates_args=(), device_types="cuda"
)
def _group_norm_op(
    input: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    activation: Optional[str],
    out_dtype: Optional[torch.dtype],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Channels-last GroupNorm forward: returns ``(output, mean, rstd)``.

    ``mean``/``rstd`` are ``(N, num_groups)`` fp32 tensors kept for the
    backward; they are marked non-differentiable in ``_setup_context``
    (nothing produces a gradient for them), so they come back with
    ``requires_grad=False`` and differentiating through them raises rather than
    returning zeros.  Callers should treat them as opaque.
    """
    _validate(input, num_groups, weight, bias, activation)
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    out, mean, rstd = _forward(
        input, num_groups, weight, bias, eps, activation, out_dtype or input.dtype
    )
    return out, mean, rstd


@_group_norm_op.register_fake
def _(input, num_groups, weight, bias, eps, activation, out_dtype):
    # empty_like preserves the input's memory format, which is the contract.
    out = torch.empty_like(input, dtype=out_dtype or input.dtype)
    mean = input.new_empty((input.shape[0], num_groups), dtype=torch.float32)
    rstd = input.new_empty((input.shape[0], num_groups), dtype=torch.float32)
    return out, mean, rstd


@torch.library.custom_op(
    "scaffold_gn::group_norm_backward", mutates_args=(), device_types="cuda"
)
def _group_norm_backward_op(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    num_groups: int,
    activation: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(d_input, d_weight, d_bias)``.

    ``d_weight``/``d_bias`` are zero-element tensors when the corresponding
    parameter is ``None``.  ``d_input`` always has the input's dtype and
    channels-last memory format.
    """
    if not grad_out.is_contiguous(memory_format=_CL_FORMAT):
        grad_out = grad_out.contiguous(memory_format=_CL_FORMAT)
    if not input.is_contiguous(memory_format=_CL_FORMAT):
        input = input.contiguous(memory_format=_CL_FORMAT)
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    d_input, d_weight, d_bias = _backward(
        grad_out, input, weight, bias, mean, rstd, num_groups, activation
    )
    # No `d_input.to(input.dtype)`: `_backward` allocates it with
    # `empty_like(input)` and `_dx_kernel` stores through `DX.dtype.element_ty`,
    # so it already *is* the input's dtype.
    if weight is None:
        d_weight = d_weight.new_empty(0)
    else:
        d_weight = d_weight.to(weight.dtype)
    if bias is None:
        d_bias = d_bias.new_empty(0)
    else:
        d_bias = d_bias.to(bias.dtype)
    return d_input, d_weight, d_bias


@_group_norm_backward_op.register_fake
def _(grad_out, input, weight, bias, mean, rstd, num_groups, activation):
    channels = input.shape[1]
    # channels_last_3d, *not* the input's own format: the real op relayouts a
    # non-channels-last `input` and always returns a channels-last `d_input`,
    # so promising `empty_like(input)` here would hand torch.compile the wrong
    # strides for any contiguous NCDHW input -- silently, since eager never
    # consults the fake kernel.
    d_input = torch.empty_like(input, memory_format=_CL_FORMAT)
    d_weight = input.new_empty(
        channels if weight is not None else 0,
        dtype=weight.dtype if weight is not None else torch.float32,
    )
    d_bias = input.new_empty(
        channels if bias is not None else 0,
        dtype=bias.dtype if bias is not None else torch.float32,
    )
    return d_input, d_weight, d_bias


def _setup_context(ctx, inputs, output):
    input, num_groups, weight, bias, eps, activation, out_dtype = inputs
    _out, mean, rstd = output
    # Outputs 1 and 2 are backward state, not results: nothing produces a
    # gradient for them.  Without this they come back requiring grad, and
    # differentiating through them *succeeds* -- autograd materializes an
    # all-zero cotangent for the unused `out` and runs the whole backward to
    # return zeros, which is a plausible wrong answer rather than an error.
    ctx.mark_non_differentiable(mean, rstd)
    ctx.save_for_backward(input, weight, bias, mean, rstd)
    ctx.num_groups = num_groups
    ctx.activation = activation
    ctx.needs = (
        ctx.needs_input_grad[0],
        ctx.needs_input_grad[2],
        ctx.needs_input_grad[3],
    )


def _autograd_backward(ctx, grad_out, grad_mean, grad_rstd):
    input, weight, bias, mean, rstd = ctx.saved_tensors
    need_x, need_w, need_b = ctx.needs
    if not (need_x or need_w or need_b):
        return None, None, None, None, None, None, None
    d_input, d_weight, d_bias = torch.ops.scaffold_gn.group_norm_backward(
        grad_out, input, weight, bias, mean, rstd, ctx.num_groups, ctx.activation
    )
    return (
        d_input if need_x else None,
        None,  # num_groups
        d_weight if need_w else None,
        d_bias if need_b else None,
        None,  # eps
        None,  # activation
        None,  # out_dtype
    )


torch.library.register_autograd(
    "scaffold_gn::group_norm", _autograd_backward, setup_context=_setup_context
)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def _autocast_active(input: torch.Tensor) -> bool:
    """Whether autocast is enabled for this tensor's device type."""
    try:
        return bool(torch.is_autocast_enabled(input.device.type))
    except (RuntimeError, TypeError):  # device type autocast does not know
        return False


def _autocast_out_dtype(input: torch.Tensor) -> Optional[torch.dtype]:
    """``F.group_norm``'s output dtype for this input, or None for "unchanged".

    ``at::group_norm`` carries autocast's ``fp32`` cast policy, so under an
    enabled autocast region it upcasts its input and returns fp32 whatever came
    in.  Verified empirically on torch 2.13.0+rocm7.2 for fp32/bf16/fp16 input
    and both bf16 and fp16 autocast dtypes.
    """
    if input.dtype is not torch.float32 and _autocast_active(input):
        return torch.float32
    return None


def is_supported(
    input,
    num_groups: int,
    weight=None,
    bias=None,
    activation: Optional[str] = None,
) -> bool:
    """Whether the native channels-last Triton kernel can serve this call.

    Cheap (a handful of attribute reads and one stride check) and side-effect
    free -- in particular it does not import Triton, allocate, or launch.
    ``False`` means "use ``F.group_norm``": the fast path needs a 5-D CUDA
    tensor that is ``channels_last_3d``-contiguous, an fp32/bf16/fp16 dtype, a
    channel count divisible by ``num_groups``, and affine parameters whose
    dtype ``F.group_norm`` would itself accept for this input (equal to the
    input's, or fp32 under autocast, which is what autocast would produce).
    Shapes ``F.group_norm`` itself rejects are rejected here too, so that
    branching on this predicate can never turn a stock ``ValueError`` into an
    answer (see :func:`_one_value_per_channel`).

    ``True`` promises the *first* derivative only: the backward is itself a
    custom op with no autograd formula, so a second ``torch.autograd.grad``
    raises where stock ``F.group_norm`` would succeed.  Callers that need a
    gradient penalty or a Hessian-vector product must not take this path.

    Note that this is a capability predicate, not a layout classifier: for
    shapes whose spatial *and* channel extents make the contiguous and
    channels-last-3d stride patterns coincide (e.g. ``(N, C, 1, 1, 1)``), a
    plain contiguous tensor is accepted, correctly -- it is the same bytes.
    """
    if activation not in SUPPORTED_ACTIVATIONS:
        return False
    if not isinstance(input, torch.Tensor):
        return False
    if input.device.type != "cuda" or not triton_available():
        return False
    if input.dim() != 5 or input.dtype not in SUPPORTED_DTYPES:
        return False
    if not isinstance(num_groups, int) or num_groups <= 0:
        return False
    channels = input.shape[1]
    if channels % num_groups != 0 or input.numel() == 0:
        return False
    if _one_value_per_channel(input, num_groups):
        return False
    if not input.is_contiguous(memory_format=_CL_FORMAT):
        return False
    autocast = None
    for t in (weight, bias):
        if t is None:
            continue
        if not isinstance(t, torch.Tensor):
            return False
        if t.dim() != 1 or t.numel() != channels:
            return False
        if t.device != input.device:
            return False
        if t.dtype is not input.dtype:
            if t.dtype is not torch.float32:
                return False
            if autocast is None:
                autocast = _autocast_active(input)
            if not autocast:
                # F.group_norm would raise "expected scalar type ..." here;
                # reject so the caller reproduces that behaviour exactly.
                return False
    return True


def triton_group_norm(
    input,
    num_groups: int,
    weight=None,
    bias=None,
    eps: float = 1e-5,
    activation: Optional[str] = None,
):
    """GroupNorm with an optionally fused activation, channels-last native.

    A drop-in replacement for ``F.group_norm(input, num_groups, weight, bias,
    eps)`` (followed by ``F.relu`` when ``activation="relu"``).  Inputs that
    :func:`is_supported` rejects are served by ``F.group_norm`` itself, which
    keeps this function total but means such calls get the *eager* kernel --
    callers with a faster fallback should branch on :func:`is_supported`
    themselves.

    The output has the input's memory format and ``F.group_norm``'s dtype; see
    the module docstring for the full contract.
    """
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {SUPPORTED_ACTIVATIONS}, got {activation!r}"
        )
    if not is_supported(input, num_groups, weight, bias, activation):
        out = F.group_norm(input, num_groups, weight, bias, eps)
        return F.relu(out) if activation == "relu" else out
    out, _mean, _rstd = torch.ops.scaffold_gn.group_norm(
        input,
        num_groups,
        weight,
        bias,
        float(eps),
        activation,
        _autocast_out_dtype(input),
    )
    return out
