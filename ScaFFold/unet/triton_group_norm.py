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
    [1,64,256^3]    4.28/11.61   19.51/77.01    4.77/11.85      151.2
    [1,128,128^3]   1.19/ 3.20    7.23/24.64    1.22/ 3.07       36.5
    [1,256,64^3]    0.35/ 0.97    2.27/ 7.42    0.34/ 0.84        9.1
    [1,512,32^3]    0.14/ 0.59    0.19/ 1.03    0.11/ 0.33        1.7
    [1,1024,16^3]   0.13/ 0.59    0.10/ 0.39    0.07/ 0.33        0.4
    [1,2048,8^3]    0.13/ 0.58    0.08/ 0.40    0.07/ 0.33        0.1

Over the 22 scale-8 call sites that is **442.8 -> 69.0 ms/step** of GroupNorm
fwd+bwd against today's production path (compiled GroupNorm on channels-last
input), i.e. **374 ms/step recovered**, and a dead heat with compiled GroupNorm
on *contiguous* input (66.4 ms/step) while additionally not breaking the
layout chain.  The three smallest shapes lose on host dispatch, not on GPU
work -- see :func:`select_strategy`.

Public API
==========
``triton_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5,
activation=None)``
    Drop-in for ``F.group_norm`` (plus an optionally fused ReLU) with full
    autograd support.  Accepts *anything* ``F.group_norm`` accepts; inputs the
    Triton kernel cannot serve fall back to ``F.group_norm`` internally (see
    "Layouts" below).

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
  ``review/gn-dctensor/triton/RESULTS.md``); preserving channels-last is the
  entire point of the kernel.
* **autograd** -- ``d_input``, ``d_weight``, ``d_bias``; ``weight=None`` and/or
  ``bias=None`` supported.
* **determinism** -- bitwise reproducible run to run and process to process.
  There are no float atomics anywhere, and the grid, split count and tile sizes
  are pure functions of the shape (the tuning table is frozen in this file for
  exactly that reason -- a *runtime* autotuner would break reproducibility by
  changing the reduction order between runs).

Reduction strategy
==================
Group statistics span ``S * C/G`` elements (134M at the largest UNet shape), so
one pass cannot produce them.  Split-K partial reductions land at a fixed
scratch index and are combined by a fixed-order tree::

    fwd:  stats_partial -> stats_finalize -> normalize            (3 kernels)
    bwd:  bwd_partial   -> bwd_finalize   -> dwdb_reduce -> dx    (4 kernels)

Traffic (``B = numel * itemsize``): 3B forward, 5B backward.

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
  table above confirms it, 66.4 ms/step against this kernel's 69.0 -- so a
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
kernel that is free (one ``tl.maximum``) and it removes an entire 2B streaming
pass.  Measured against ``F.relu(triton_group_norm(x))``: 38% off the forward
and 35% off fwd+bwd at ``[1,64,256^3]`` (6.94 -> 4.29 ms and 18.27 -> 11.80
ms), 37%/33% at ``[1,128,128^3]``, tapering to ~11% at the launch-bound
shapes.

The backward gates the incoming gradient on the sign of the **pre-activation**
value, which it *recomputes* from the saved ``(x, mean, rstd, weight, bias)``
using the identical expression the forward used.  Recomputation costs two FLOPs
on values already in registers and is bit-exact -- same inputs, same operation
order, same fp32 rounding -- so the sign always agrees with the forward's.  The
alternative, testing ``y > 0`` on the saved output, would need the output kept
alive *in addition to* ``x`` (which the GroupNorm backward needs regardless),
and in bf16/fp16 it would also mis-gate any element whose positive
pre-activation rounded to zero on the store.

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

Going through the dispatcher costs ~35 us of host time per forward call
(measured against calling ``_forward``/``_backward`` directly), which is
invisible at the three largest shapes and is roughly two kernel launches at the
three smallest.  That is the price of composing, and it is the same order as
the launch overhead those shapes already pay; see :func:`select_strategy`.

Triton is imported lazily, on the first call that actually reaches the kernel,
so importing this module (or running the CPU test suite) costs nothing.
"""

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
]

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
    batch, so the per-sample split count is ``nsplit_target // N``.
    """

    __slots__ = (
        "stats_tile",
        "stats_warps",
        "nsplit_target",
        "elem_tile",
        "elem_warps",
    )

    def __init__(
        self,
        stats_tile=8192,
        stats_warps=4,
        nsplit_target=2048,
        elem_tile=8192,
        elem_warps=4,
    ):
        self.stats_tile = stats_tile
        self.stats_warps = stats_warps
        self.nsplit_target = nsplit_target
        self.elem_tile = elem_tile
        self.elem_warps = elem_warps

    def key(self):
        return (
            self.stats_tile,
            self.stats_warps,
            self.nsplit_target,
            self.elem_tile,
            self.elem_warps,
        )

    def __eq__(self, other):
        return isinstance(other, GNConfig) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return (
            "GNConfig(stats_tile=%d, stats_warps=%d, nsplit_target=%d, "
            "elem_tile=%d, elem_warps=%d)" % self.key()
        )


#: Frozen tuning table, produced by coordinate descent on fwd+bwd time on one
#: MI300A (228 CUs) at fp32 with ``num_groups=8``, keyed by the
#: ``(num_channels, cube-root spatial extent)`` of the scale-8 ScaFFold UNet
#: GroupNorm sites.  Frozen -- never autotuned at run time -- because the split
#: count fixes the reduction order and therefore the bits of the result.
_TUNED = {
    (64, 256): GNConfig(8192, 4, 2048, 8192, 4),
    (128, 128): GNConfig(16384, 4, 512, 16384, 4),
    (256, 64): GNConfig(4096, 4, 512, 8192, 4),
    (512, 32): GNConfig(32768, 4, 8192, 8192, 4),
    (1024, 16): GNConfig(16384, 8, 512, 8192, 4),
    (2048, 8): GNConfig(32768, 4, 512, 8192, 4),
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

#: Spatial extent (``D*H*W``) below which the split-K chain is expected to be
#: host-dispatch bound rather than bandwidth bound.  Measured on MI300A: at
#: ``[1,2048,8^3]`` the seven kernels do 0.031 ms of GPU work behind 0.600 ms
#: of Python/autograd/launch cost, a 19x overhead tax (RESULTS.md 3).  Purely
#: informational today -- ``select_strategy`` does not use it yet.
SMALL_SPATIAL_THRESHOLD = 4096


def select_strategy(n: int, num_channels: int, spatial: int, num_groups: int) -> str:
    """### SMALL-SHAPE DISPATCH HOOK ### -- the single point where a different
    kernel strategy is chosen for a shape.

    Returns a name from :data:`STRATEGIES`.  Today it always returns
    ``"split_k"``: three forward and four backward kernels with split-K partial
    reductions, which is bandwidth-optimal for the large shapes but pays seven
    kernel launches (~17 us each on this node) plus autograd overhead
    regardless of size -- so below roughly ``SMALL_SPATIAL_THRESHOLD`` voxels
    the whole call is host bound and a *single-program-per-(n, group)* kernel
    that never leaves registers would win.

    That regime is under active investigation; when a second strategy lands,
    add its name to :data:`STRATEGIES`, return it from here on a rule that is a
    **pure function of the shape** (determinism depends on it), and branch on
    it in ``_dispatch`` -- which is the only caller, sits in front of the
    memoized tiling plan, and is itself called by both ``_forward`` and
    ``_backward``.  Nothing else in this file needs to change.
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

    @property
    def elements_per_group(self) -> float:
        return float(self.spatial * self.group_channels)


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
_stats_finalize_kernel = None
_normalize_kernel = None
_bwd_partial_kernel = None
_bwd_finalize_kernel = None
_dwdb_reduce_kernel = None
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
            x = tl.load(X + base + off, mask=m, other=0.0).to(tl.float32)

            # Corrected two-pass within the tile: the first mean loses digits
            # to the magnitude of the data, `corr` puts them back, and the
            # centred squares are then accurate to fp32 roundoff.  Everything
            # here is register traffic; the tile is read from HBM exactly once.
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

    @_triton.jit
    def _stats_finalize_kernel(
        PCNT,
        PMEAN,
        PM2,
        MEAN,
        RSTD,
        M,
        eps,
        G: tl.constexpr,
        NSPLIT: tl.constexpr,
    ):
        """Merge the NSPLIT partials of one ``(n, g)`` into mean and 1/std."""
        pid = tl.program_id(0)  # n * G + g
        n = pid // G
        g = pid % G
        offs = tl.arange(0, NSPLIT)
        idx = (n * NSPLIT + offs) * G + g
        cnt, mean, m2 = tl.reduce(
            (tl.load(PCNT + idx), tl.load(PMEAN + idx), tl.load(PM2 + idx)),
            0,
            _welford_combine,
        )
        # `cnt` equals M by construction; M is passed in so the divisor is the
        # exact element count rather than a float accumulated from partials.
        var = m2 / M
        tl.store(MEAN + pid, mean)
        tl.store(RSTD + pid, 1.0 / tl.sqrt(var + eps))

    # ------------------------------------------------------------ normalize --
    @_triton.jit
    def _normalize_kernel(
        X,
        Y,
        MEAN,
        RSTD,
        W,
        B,
        S,
        C: tl.constexpr,
        G: tl.constexpr,
        CG: tl.constexpr,
        GP: tl.constexpr,
        CGP: tl.constexpr,
        BLOCK_S: tl.constexpr,
        RELU: tl.constexpr,
        HAS_W: tl.constexpr,
        HAS_B: tl.constexpr,
        MASKED_C: tl.constexpr,
        INT64: tl.constexpr,
    ):
        blk = tl.program_id(0)
        n = tl.program_id(1)

        offs_g = tl.arange(0, GP)
        offs_j = tl.arange(0, CGP)
        offs_s = tl.arange(0, BLOCK_S)
        inner = offs_g[None, :, None] * CG + offs_j[None, None, :]
        off = offs_s[:, None, None] * C + inner
        wb = offs_g[:, None] * CG + offs_j[None, :]
        wbm = (offs_g[:, None] < G) & (offs_j[None, :] < CG)

        s0 = blk * BLOCK_S
        m = tl.broadcast_to((offs_s < S - s0)[:, None, None], (BLOCK_S, GP, CGP))
        if MASKED_C:
            m = m & ((offs_g[None, :, None] < G) & (offs_j[None, None, :] < CG))
        if INT64:
            base = (n.to(tl.int64) * S + s0) * C
        else:
            base = (n * S + s0) * C

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

        x = tl.load(X + base + off, mask=m, other=0.0).to(tl.float32)
        xhat = (x - mean) * rstd
        y = xhat * w + b
        if RELU:
            y = tl.maximum(y, 0.0)
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
                # gating cannot resurrect them.
                dy = tl.where(xhat * w + b > 0.0, dy, 0.0)
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
    def _bwd_finalize_kernel(
        PS1,
        PS2,
        C1,
        C2,
        M,
        G: tl.constexpr,
        NSPLIT: tl.constexpr,
    ):
        pid = tl.program_id(0)  # n * G + g
        n = pid // G
        g = pid % G
        offs = tl.arange(0, NSPLIT)
        idx = (n * NSPLIT + offs) * G + g
        tl.store(C1 + pid, tl.sum(tl.load(PS1 + idx)) / M)
        tl.store(C2 + pid, tl.sum(tl.load(PS2 + idx)) / M)

    @_triton.jit
    def _dwdb_reduce_kernel(
        PDW,
        PDB,
        DW,
        DB,
        ROWS,
        C,
        BLOCK_C: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
        mc = offs_c < C
        accw = tl.zeros((BLOCK_C,), dtype=tl.float32)
        accb = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for r0 in range(0, ROWS, BLOCK_R):
            offs_r = r0 + tl.arange(0, BLOCK_R)
            m = (offs_r[:, None] < ROWS) & mc[None, :]
            off = offs_r[:, None] * C + offs_c[None, :]
            accw += tl.sum(tl.load(PDW + off, mask=m, other=0.0), 0)
            accb += tl.sum(tl.load(PDB + off, mask=m, other=0.0), 0)
        tl.store(DW + offs_c, accw, mask=mc)
        tl.store(DB + offs_c, accb, mask=mc)

    @_triton.jit
    def _dx_kernel(
        X,
        DY,
        DX,
        MEAN,
        RSTD,
        W,
        B,
        C1,
        C2,
        S,
        C: tl.constexpr,
        G: tl.constexpr,
        CG: tl.constexpr,
        GP: tl.constexpr,
        CGP: tl.constexpr,
        BLOCK_S: tl.constexpr,
        RELU: tl.constexpr,
        HAS_W: tl.constexpr,
        HAS_B: tl.constexpr,
        MASKED_C: tl.constexpr,
        INT64: tl.constexpr,
    ):
        blk = tl.program_id(0)
        n = tl.program_id(1)

        offs_g = tl.arange(0, GP)
        offs_j = tl.arange(0, CGP)
        offs_s = tl.arange(0, BLOCK_S)
        inner = offs_g[None, :, None] * CG + offs_j[None, None, :]
        off = offs_s[:, None, None] * C + inner
        wb = offs_g[:, None] * CG + offs_j[None, :]
        wbm = (offs_g[:, None] < G) & (offs_j[None, :] < CG)

        s0 = blk * BLOCK_S
        m = tl.broadcast_to((offs_s < S - s0)[:, None, None], (BLOCK_S, GP, CGP))
        if MASKED_C:
            m = m & ((offs_g[None, :, None] < G) & (offs_j[None, None, :] < CG))
        if INT64:
            base = (n.to(tl.int64) * S + s0) * C
        else:
            base = (n * S + s0) * C

        gm = offs_g < G
        mean = tl.load(MEAN + n * G + offs_g, mask=gm, other=0.0)[None, :, None]
        rstd = tl.load(RSTD + n * G + offs_g, mask=gm, other=0.0)[None, :, None]
        c1 = tl.load(C1 + n * G + offs_g, mask=gm, other=0.0)[None, :, None]
        c2 = tl.load(C2 + n * G + offs_g, mask=gm, other=0.0)[None, :, None]
        if HAS_W:
            w = tl.load(W + wb, mask=wbm, other=0.0).to(tl.float32)[None, :, :]
        else:
            w = tl.full((1, GP, CGP), 1.0, tl.float32)
        if HAS_B:
            b = tl.load(B + wb, mask=wbm, other=0.0).to(tl.float32)[None, :, :]
        else:
            b = tl.zeros((1, GP, CGP), dtype=tl.float32)

        x = tl.load(X + base + off, mask=m, other=0.0).to(tl.float32)
        dy = tl.load(DY + base + off, mask=m, other=0.0).to(tl.float32)
        xhat = (x - mean) * rstd
        if RELU:
            dy = tl.where(xhat * w + b > 0.0, dy, 0.0)
        dyw = dy * w
        dx = rstd * (dyw - c1 - xhat * c2)
        tl.store(DX + base + off, dx.to(DX.dtype.element_ty), mask=m)

    globals().update(
        _welford_combine=_welford_combine,
        _stats_partial_kernel=_stats_partial_kernel,
        _stats_finalize_kernel=_stats_finalize_kernel,
        _normalize_kernel=_normalize_kernel,
        _bwd_partial_kernel=_bwd_partial_kernel,
        _bwd_finalize_kernel=_bwd_finalize_kernel,
        _dwdb_reduce_kernel=_dwdb_reduce_kernel,
        _dx_kernel=_dx_kernel,
    )


def _ensure_kernels():
    if _stats_partial_kernel is None:
        _build_kernels()


# --------------------------------------------------------------------------- #
# python drivers
# --------------------------------------------------------------------------- #
_CL_FORMAT = torch.channels_last_3d


def _shape_of(input: torch.Tensor):
    n, channels = input.shape[0], input.shape[1]
    spatial = 1
    for d in input.shape[2:]:
        spatial *= d
    return n, channels, spatial


def _forward(input, num_groups, weight, bias, eps, activation, out_dtype):
    _ensure_kernels()
    n, channels, spatial = _shape_of(input)
    plan = _dispatch(n, channels, spatial, num_groups, input.numel())
    groups = num_groups
    device = input.device

    pcnt = torch.empty(n * plan.nsplit * groups, device=device, dtype=torch.float32)
    pmean = torch.empty_like(pcnt)
    pm2 = torch.empty_like(pcnt)
    mean = torch.empty((n, groups), device=device, dtype=torch.float32)
    rstd = torch.empty_like(mean)

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
    _stats_finalize_kernel[(n * groups,)](
        pcnt,
        pmean,
        pm2,
        mean,
        rstd,
        plan.elements_per_group,
        eps,
        G=groups,
        NSPLIT=plan.nsplit,
        num_warps=4,
    )

    out = torch.empty_like(input, dtype=out_dtype, memory_format=_CL_FORMAT)
    _normalize_kernel[(plan.nblk_elem, n)](
        input,
        out,
        mean,
        rstd,
        weight,
        bias,
        spatial,
        C=channels,
        G=groups,
        CG=plan.group_channels,
        GP=plan.groups_p2,
        CGP=plan.group_channels_p2,
        BLOCK_S=plan.block_s_elem,
        RELU=activation == "relu",
        HAS_W=weight is not None,
        HAS_B=bias is not None,
        MASKED_C=plan.masked_c,
        INT64=plan.int64,
        num_warps=plan.cfg.elem_warps,
    )
    return out, mean, rstd


def _backward(grad_out, input, weight, bias, mean, rstd, num_groups, activation):
    _ensure_kernels()
    n, channels, spatial = _shape_of(input)
    plan = _dispatch(n, channels, spatial, num_groups, input.numel())
    groups = num_groups
    device = input.device

    ps1 = torch.empty(n * plan.nsplit * groups, device=device, dtype=torch.float32)
    ps2 = torch.empty_like(ps1)
    pdw = torch.empty(n * plan.nsplit * channels, device=device, dtype=torch.float32)
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

    c1 = torch.empty(n * groups, device=device, dtype=torch.float32)
    c2 = torch.empty_like(c1)
    _bwd_finalize_kernel[(n * groups,)](
        ps1,
        ps2,
        c1,
        c2,
        plan.elements_per_group,
        G=groups,
        NSPLIT=plan.nsplit,
        num_warps=4,
    )

    d_weight = torch.empty(channels, device=device, dtype=torch.float32)
    d_bias = torch.empty_like(d_weight)
    rows = n * plan.nsplit
    block_c = min(256, max(64, _next_pow2(channels)))
    block_r = 32 if rows >= 32 else 1
    _dwdb_reduce_kernel[(_cdiv(channels, block_c),)](
        pdw,
        pdb,
        d_weight,
        d_bias,
        rows,
        channels,
        BLOCK_C=block_c,
        BLOCK_R=block_r,
        num_warps=4,
    )

    d_input = torch.empty_like(input, memory_format=_CL_FORMAT)
    _dx_kernel[(plan.nblk_elem, n)](
        input,
        grad_out,
        d_input,
        mean,
        rstd,
        weight,
        bias,
        c1,
        c2,
        spatial,
        C=channels,
        G=groups,
        CG=plan.group_channels,
        GP=plan.groups_p2,
        CGP=plan.group_channels_p2,
        BLOCK_S=plan.block_s_elem,
        RELU=activation == "relu",
        HAS_W=weight is not None,
        HAS_B=bias is not None,
        MASKED_C=plan.masked_c,
        INT64=plan.int64,
        num_warps=plan.cfg.elem_warps,
    )
    return d_input, d_weight, d_bias


# --------------------------------------------------------------------------- #
# torch.library registration
# --------------------------------------------------------------------------- #
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
    backward; they are *not* differentiable (nothing produces a gradient for
    them) and callers should treat them as opaque.
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
    d_input = d_input.to(input.dtype)
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
    d_input = torch.empty_like(input)
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
