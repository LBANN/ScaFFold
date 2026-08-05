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

"""3-D convolution with a Triton fast path and MIOpen behind it.

Two kernels, tried in order:

1. **Native channels-last Triton** (:mod:`triton_conv3d`, a self-contained
   package that imports nothing from ScaFFold), whenever its ``is_supported_all``
   gate -- all three direction predicates, which do not accept the same problems
   -- accepts the call.  It is an implicit-GEMM forward,
   a re-strided view of that forward for backward-data, and a deterministic
   split-K reduction for backward-weight; it reads a ``channels_last_3d`` weight
   where it lies, which is the layout ``worker.py`` already puts every
   convolution parameter in.  Measured at **1.26x of the MIOpen step** at
   ScaFFold's scale-7 single-GPU configuration, with the backward-weight
   direction reproducible at no cost -- see ``work/triton-conv/STATUS.md``.
2. **MIOpen**, via ``nn.Conv3d.forward``, which is what every rejection falls
   back to and what defines the semantics the Triton rung must match.

``FastConv3d`` is a drop-in ``nn.Conv3d``: same parameters, same names, same
shapes, no buffers and no ``state_dict`` keys of its own, so checkpoints are
interchangeable in both directions with any other ``nn.Conv3d``-based build.

Two ladders
===========
:class:`FastConvTranspose3d` is the same construction for the decoder's four
``nn.ConvTranspose3d(k=2, s=2)`` upsamplers, which ``triton_conv3d.transposed``
serves in all three directions.  It shares with :class:`FastConv3d` everything
that is genuinely one mechanism -- the failure allowlist, the latch, the
autocast reproduction, the ``DCTensor`` unwrap, the routing conditions that are
not about which operator this is (:func:`_routing_declines`) -- and shares
nothing else, because the rest reads different numbers off different tensors:

* the two parameters store their channel axes in opposite orders, so a
  block-list or a gate written for one **accepts** the other whenever the two
  channel counts happen to match, and computes a different operator quietly;
* the transposed forward's GEMM has one row per *input* voxel where the
  ordinary one has one per output voxel -- 8x apart at ``k = s = 2``;
* and the halo, which is the whole difficulty below, does not exist for
  ``kernel == stride``: every output voxel reads exactly one input voxel.  So
  that ladder has no :class:`_Halo3d` in it, and :func:`_transposed_halo_plan`
  is where "there is nothing to exchange" is *checked* -- against DistConv's own
  ``halo = k // 2``, which is 0 for an even kernel -- rather than assumed.

Sharding is the whole difficulty
================================
GroupNorm's fast path unwraps DistConv's ``DCTensor`` to its local shard, runs,
and rewraps.  That is semantically inert *for GroupNorm*, whose statistics are
per-shard at every shard count.  **It is not inert for convolution.**  A
convolution needs its neighbours' boundary voxels, and DistConv supplies them
*below* autograd, inside its interception of ``aten.convolution.default``
(``distconv.py``, ``distconv_forward``): it concatenates a halo slab of width
``k // 2`` onto both faces of every sharded dimension and then zeroes that
dimension's padding.  A module-level adapter that unwraps and calls a kernel
directly never reaches that code, so it would silently drop the halo and return
a wrong answer at every shard boundary -- 55% relative error, measured, so any
defect here is loud once it is looked for.

So this module performs the exchange itself, *above* autograd, as its own
``autograd.Function`` (:class:`_Halo3d`) stacked over the convolution's
(:class:`_TritonConv3dFn`).  The forward concatenates the neighbours' boundary
slabs onto the local shard and hands the kernel a padding with those dims
zeroed; the backward sends the outer boundary gradient slabs back and
accumulates what arrives into the inner region, which is
``distconv.backward_halo_exchange`` exactly.  ``grad_weight`` is computed
against the *halo'd* input, as ``distconv_backward`` does, so each rank counts
every filter tap that straddles its boundary once and ``DistConvDDP``'s
``world_size / ddp_ranks`` rescaling turns DDP's average back into the sum the
spatial shards need.

**Only the dims that are actually split are exchanged.**  ScaFFold ships
``dc_shard_dims: [2, 3, 4]`` with ``dc_num_shards`` of ``[1,1,1]``, ``[2,1,1]``
or ``[4,1,1]``, so D is the only axis ever divided and DistConv's halo on H and
W is two ``cat`` copies of a slab that is provably zeros.  ``cat(zeros, x,
zeros)`` at ``padding = 0`` is the same arithmetic as ``padding = k // 2`` on
``x`` -- and *measured bitwise identical* through these kernels in fp32 and
bf16 at 2 and 4 shards, which is what makes dropping it a decision with no
tolerance argument in it.  At ``dc_num_shards = (1, 1, 1)`` every dim is unsplit,
:class:`_Halo3d` is not applied at all, and the rung is stage 1's exactly:
``forward_halo_exchange`` allocates its receive buffers with ``zeros_like`` and
posts nothing when ``shard_ind`` is 0 on every axis, so the 3.797 ms/step of
zero-slab ``cat`` copies ``work/upstream-repros/DISTCONV_ACTIONS.md`` prices at one shard just
disappear.

Every fact that argument rests on is checked *positively* before the rung is
taken -- see :func:`_halo_plan`, which returns ``None`` for anything it could not
read.  "I could not find evidence of a problem" is not the same statement as "I
have checked that there is none", and only the second one is safe: the failure
mode of getting this wrong is a plausible-looking wrong gradient at scale, not a
crash.

The exchange sends **plain-contiguous** buffers, which
``forward_halo_exchange`` does not: it sends
``.contiguous(memory_format=channels_last_3d)``, and a ``channels_last_3d``
tensor is not plain-contiguous for ``C > 1``.  That, not the narrow, is why
DistConv's spatial sharding does not run under gloo.  Packing NDHWC into a
plain-contiguous slab costs one copy of 1.6% of the activation and makes the
sharded path runnable on a machine with no RCCL -- see :func:`_packed_slab`.

Autocast
========
ScaFFold trains inside ``torch.autocast(device_type="cuda",
dtype=torch.bfloat16)`` (``trainer.py``'s ``_autocast_kwargs``, on by default via
``torch_amp: 1``).  Autocast's cast for ``aten::convolution`` -- and for
``aten::conv_transpose3d``, which carries the same ``lower_precision_fp`` policy
(measured: an fp32 activation and an fp32 parameter produce a bf16 output, and
an fp64 pair is left alone) -- happens *in the
dispatcher*, so an adapter that calls a kernel directly bypasses it: the
convolution's operands arrive as the fp32 tensors GroupNorm produced and the
fp32 parameters the model holds, and running them as-is would silently execute
the whole network's convolutions in fp32 -- a different computation from the one
the benchmark defines, several times slower, and nothing fails.
:func:`_autocast_dtype` and :func:`_cast_operand` reproduce ATen's rule
(``cached_cast``: cast a floating tensor to the autocast dtype *only* if it is
exactly fp32), outside the autograd node so the cast's backward returns the fp32
parameter gradient exactly as autocast's own does.
``unet_parts._consumer_dtype`` is the same reasoning applied to ``torch.cat``'s
``promote`` policy.

The cast is applied *before* the halo exchange, which is both what DistConv does
-- its ``__torch_dispatch__`` runs below the autocast key, so the tensor it
concatenates is already bf16 -- and what halves the bytes on the wire.

Hardware
========
The rung is taken only on the GPU the kernels were tuned on -- ``gfx942`` with
228 CUs, i.e. an MI300A -- and declines quietly to MIOpen anywhere else.  That
guard is not about capability: ``triton_conv3d``'s kernels compute the right
convolution wherever Triton lowers them, and its ``is_supported*`` predicates
are right to say nothing about the device.  It is about the fact that *every*
number deciding how they launch was raced on one machine, and that a launch
configuration which is merely wrong for the hardware raises nothing at all --
``gather_gemm``'s docstring records the concrete mechanism, an illegal MFMA
configuration that emits zero MFMA instructions and returns correct results at a
fraction of the speed.  A ladder whose fallback allowlist is
``triton.errors.TritonError`` has nothing to catch there.  So the check is a
routing decision and lives with the others, in
:func:`~ScaFFold.unet._rungs._platform_declines`; the decision it encodes, and
why the explicit opt-in overrides it while none of the correctness conditions
can be overridden at all, is argued there.

Latches
=======
The Triton rung is an optimization, never a correctness requirement: a broken
Triton install must degrade a multi-node run, not kill it.  So a *kernel*
failure is caught, logged once and retried on MIOpen.

The retry has two shapes, and which one applies is decided by whether this
rank has already put a halo slab on the wire.  With nothing sent -- the
unsharded case, and every call in a one-shard run -- the whole call is re-run
from the top on ``_miopen_forward``, which is also the only rung that can be
handed a ``DCTensor``.  Once a dim really is split, that route is closed: it
goes through ``distconv_forward``, which would exchange a *second* time, giving
this rank one more collective than a peer whose kernel compiled and hanging the
mesh (or cross-pairing two convolutions' slabs) instead of merely slowing down.
So the exchange is performed once, *above* the retried region, and only the
kernel call is retried: MIOpen is handed the already-exchanged tensor at
``plan.padding`` and swaps nothing but the kernel.  Otherwise a Triton compile
failure would be **fatal** at ``num_shards > 1`` while costing only speed at 1,
which is exactly backwards for a ladder whose reason to exist is degrading
instead of dying.

"A kernel failure" is an allowlist, not the absence of one: exactly
``triton.errors.TritonError`` (see :func:`_triton_kernel_failures`).
``triton_conv3d`` has no exception type of its own -- unlike
``triton_group_norm``, it does not tag its launch region -- so this allowlist is
drawn at Triton's boundary instead of at the package's.  It is nonetheless
closed and small: that one root covers ``OutOfResources``, ``CompilationError``,
``CompileTimeAssertionFailure``, ``UnsupportedLanguageConstruct``, ``PTXASError``
and ``AutotunerError``, every one of which is raised while compiling or sizing a
launch, i.e. before any device work, which is what makes the retry safe.
Everything else propagates, and each exclusion is load-bearing:

* ``ValueError`` -- every one the package raises is a caller-contract violation
  (``_check_out``, ``_check_weight_rsck``, ``_triple``, ``ConvConfig.validate``).
  Catching them would turn the one bug class that produces silently wrong
  numbers into a quiet performance regression.
* ``NotImplementedError`` -- raised by all three entry points when their own
  ``is_supported*`` says no.  Unreachable because this module branches on those
  predicates first; if it fires, the predicate and the entry point disagree and
  that has to be seen.
* ``torch.OutOfMemoryError`` and ``torch.AcceleratorError`` -- both
  ``RuntimeError`` subclasses.  The first is a resource condition, and MIOpen
  needs *more* memory than the Triton path at the shapes where it bites; the
  second means the HIP context is already poisoned and no fallback can succeed.
* bare ``RuntimeError`` -- excluded precisely because the three above are all
  subclasses of it.

Not covered by any allowlist, and worth stating: a kernel that stores out of
bounds takes the process down with a HSA "Memory access fault" and SIGABRT
without raising anything Python can see (measured on this MI300A).  Robustness
against that lives in ``triton_conv3d``'s own argument checks, not here.

A failure latches the rung off **for modules that have never had a call served
by it**; a module that has already run on it keeps it.  That is not a
performance nicety.  ``torch.utils.checkpoint``'s non-reentrant recompute
substitutes recomputed tensors positionally into the original graph node, and
under DistConv the two rungs save tensors that are *metadata-identical* --
same shape, dtype and device -- differing only in whether slot 0 holds the
``DCTensor`` wrapper (MIOpen, which saves below DistConv's dispatch) or its
inner tensor (Triton, which saves above it).  ``_default_meta_extractor``
compares shape, dtype and device and nothing else, so a rung flip across a
recompute passes torch's own check and fails later inside DistConv with an
``AttributeError`` about ``_parallel_strategy`` or ``_is_periodic`` (measured,
both directions).  Once anything is actually sharded the Triton rung saves the
*halo'd* input, whose extent differs, so a flip there is caught by torch's own
metadata check as a plain ``CheckpointError`` (measured) -- but the unsharded
configuration is the shipped one and it is the silent case, so the latch is what
the correctness argument rests on either way.  Pinning each module instance's
choice for the life of the
process is what makes forward and recompute agree; :func:`_replaying_a_forward`
bounds the *fallback* by the same argument, and
:meth:`_TritonConv3dFn.backward` carries a cheap type check that turns the
``AttributeError`` into an actionable message if it ever does happen.

A latch is process-local, so under DDP one rank can end up on a different kernel
from its peers.  The two rungs agree to fp32 rounding, not bitwise, so a rank
that latches shifts its gradients and therefore the all-reduced ones.  That is
the price of degrading instead of dying, and it is why the latch is as narrow as
it is and why ``torch.OutOfMemoryError`` latches nothing.

Determinism
===========
``conv3d_backward_weight`` reduces its split-K partials in fp32 and stores once;
its ``deterministic=True`` default is both reproducible and *faster* than the
atomic path, so nothing here plumbs ``more_determinism`` into the kernel choice.
MIOpen's backward-weight, by contrast, reduces with atomics and disagrees with
itself bitwise between two identical calls.
"""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from ._rungs import (
    _dctensor_ops,
    _env_override,
    _functorch_active,
    _platform_declines,
    _replaying_a_forward,
    _run_local,
    _warn_rung_failure,
)

logger = logging.getLogger(__name__)

#: Opt-out (``0``/``false``/``off``/``no``) or explicit opt-in (``1``/``true``/
#: ``on``/``yes``) for the Triton convolution path.  Unset means "on wherever it
#: is safe", matching ``SCAFFOLD_GROUPNORM_TRITON``: the whole point of the
#: kernel is that production takes it, and every configuration it must not serve
#: is refused by a correctness check rather than by this default.
#:
#: The explicit opt-in is *not* just that default written out: it additionally
#: overrides the hardware guard, which is the only routing condition here that
#: is a preference rather than a correctness condition.  See
#: :func:`set_conv_triton_enabled`.
TRITON_ENV_VAR = "SCAFFOLD_CONV_TRITON"

# The triton_conv3d package, imported on the first eligible forward.  Its
# __init__ is lazy in turn -- the entry points live in submodules that import
# torch and triton -- so a CPU-only run pays neither.
_triton_module = None

# The ladder's allowlist, resolved on first use of the rung it guards.
_TRITON_KERNEL_FAILURES = None

# Set once if the Triton kernel raises; MIOpen is used from then on, except by
# modules the rung has already served.
_triton_failed = False

# None = decide per tensor; True/False = forced by SCAFFOLD_CONV_TRITON or by
# set_conv_triton_enabled().
_triton_override = _env_override(TRITON_ENV_VAR)

# Set once if a predicate raised while deciding; see _use_triton.
_predicate_warned = False

#: One-element tensors, one per ``(dtype, device)``, from which
#: :func:`_metadata_probe` expands.  Bounded by the number of dtype/device pairs
#: a process actually convolves in, which for ScaFFold is one.
_PROBE_BASES = {}


def set_conv_triton_enabled(enabled):
    """Force the Triton convolution path on (``True``) or off (``False``).

    The counterpart of ``group_norm.set_triton_enabled``: ``None`` restores the
    default (``SCAFFOLD_CONV_TRITON`` if set, otherwise "wherever it is safe"),
    and the previous setting is returned so tests can restore it.

    Forcing it on clears any failure latch, and overrides the **hardware
    guard**, and overrides nothing else -- not the device, subclass, sharding or
    ``is_supported`` checks, which are correctness conditions rather than
    preferences.  The hardware guard is on the other side of that line and the
    difference is the failure mode: an unsupported shape or an unknown tensor
    subclass would make this rung compute the *wrong thing*, while an untuned
    GPU makes it compute the right thing at a speed nobody has measured (see
    :func:`~ScaFFold.unet._rungs._platform_declines`).  The second is a judgement
    a developer is entitled to make about their own machine, so ``True`` here --
    and ``SCAFFOLD_CONV_TRITON=1``, which is the same statement spelled in the
    environment -- takes the kernels on hardware the default declines, and says
    so in the log.  Note that this makes an explicit ``1`` mean something
    stronger than leaving the variable unset, which is the one place the
    tri-state is not merely "the default, written out".

    ``None`` deliberately does *not* clear the latch: it restores a preference,
    it does not assert that the kernel works again.
    """
    global _triton_override, _triton_failed
    previous = _triton_override
    _triton_override = (
        _env_override(TRITON_ENV_VAR) if enabled is None else bool(enabled)
    )
    if _triton_override is True:
        _triton_failed = False
    return previous


def _get_triton_module():
    """Import (once) the :mod:`triton_conv3d` package.

    Deferred rather than imported at the top of this file for the same reason
    ``group_norm`` defers its kernel module: a run that never reaches the GPU
    (the whole CPU unit suite) must not pay for ``triton``.  The package's own
    ``__init__`` re-exports its entry points lazily, so even this import does not
    pull in torch's Triton stack until a predicate is actually asked.
    """
    global _triton_module
    if _triton_module is None:
        import triton_conv3d

        _triton_module = triton_conv3d
    return _triton_module


def _triton_kernel_failures():
    """The ladder's allowlist: exactly ``triton.errors.TritonError``.

    That single root is the parent of ``OutOfResources``,
    ``CompilationError``, ``CompileTimeAssertionFailure``,
    ``UnsupportedLanguageConstruct``, ``PTXASError``, ``AutotunerError`` and
    ``InterpreterError``; ``triton.runtime.errors.TritonError`` and
    ``triton.compiler.errors.TritonError`` are the same object.  Every one of
    them is raised while compiling a kernel or sizing its launch -- a missing or
    mismatched ``triton``, an unwritable JIT cache, a compile error, a tile that
    does not fit in LDS -- so nothing has executed and retrying the same call on
    MIOpen is safe.

    The module docstring lists what is deliberately left out and why.  Resolved
    on demand and cached, so a CPU-only run never imports ``triton``.  An empty
    tuple (no ``triton`` at all) means "catch nothing": the ladder then
    re-raises, which is right, because with no Triton there is nothing that
    could have failed inside one -- and the predicate would have declined the
    rung long before.
    """
    global _TRITON_KERNEL_FAILURES
    if _TRITON_KERNEL_FAILURES is None:
        try:
            from triton.errors import TritonError

            _TRITON_KERNEL_FAILURES = (TritonError,)
        except ImportError:  # pragma: no cover - triton ships it
            _TRITON_KERNEL_FAILURES = ()
    return _TRITON_KERNEL_FAILURES


def _latch_rung_failure(error, what="Triton conv3d"):
    """Latch the Triton rung off, logging only on the ``False -> True`` edge.

    Both handlers that catch an allowlisted kernel failure in a *forward* need
    exactly this, so it is written once: the one in :meth:`FastConv3d.forward`,
    which re-runs the whole call on MIOpen, and the one in
    :meth:`FastConv3d._triton_forward`, which cannot (its halo is already on the
    wire) and instead hands the exchanged tensor to MIOpen itself.  A module that
    has already used the rung keeps trying it -- that is what pins a checkpointed
    block to one rung -- so without the edge test a persistently broken kernel
    would warn once per call for the rest of the run.  Clearing the latch re-arms
    the message.

    ``what`` names the ladder in the message and nothing else: **one latch
    serves both**.  Every allowlisted failure is a property of the *install* --
    a missing or mismatched ``triton``, an unwritable JIT cache, a compile error
    -- and those break both ladders at once; the exception is ``OutOfResources``,
    which is per-launch, and there the over-latch costs little because the
    encoder's ordinary convolutions all run (and become ``proven``) before the
    decoder reaches its first transposed one.
    """
    global _triton_failed
    first = not _triton_failed
    _triton_failed = True
    if first:
        _warn_rung_failure(what, error, "MIOpen kernel", TRITON_ENV_VAR)


def _warn_once_about_the_predicate(error):
    """Log the first ``is_supported*`` failure; a repeat would log per call."""
    global _predicate_warned
    if _predicate_warned:
        return
    _predicate_warned = True
    logger.warning(
        f"Triton conv3d routing check failed ({type(error).__name__}: {error}); "
        "using MIOpen for calls like this one. This is a routing miss, not a "
        "kernel failure, so nothing is latched off."
    )


# ---------------------------------------------------------------------------
# Autocast
# ---------------------------------------------------------------------------


def _autocast_dtype(tensor):
    """The dtype autocast would cast this call's operands to, or ``None``.

    ``aten::convolution`` carries the ``lower_precision_fp`` cast policy, so
    inside an enabled autocast region for this device the answer is autocast's
    dtype.  The cast is applied by the dispatcher, *below* this module, which is
    exactly why calling a kernel directly has to reproduce it -- see the module
    docstring.

    ``None`` means "no cast", which is what a run with ``torch_amp: 0``, an
    evaluation outside the autocast region, or a device autocast does not know
    about all get.
    """
    device_type = tensor.device.type
    try:
        if not torch.is_autocast_enabled(device_type):
            return None
        return torch.get_autocast_dtype(device_type)
    except (RuntimeError, TypeError):  # a device type autocast does not know
        return None


def _cast_operand(tensor, dtype):
    """Apply :func:`_autocast_dtype`'s answer the way ATen's ``cached_cast`` does.

    Only an exactly-fp32 tensor is cast: ATen's ``is_eligible`` requires
    ``scalar_type() == kFloat``, so an operand already in the lower precision is
    left alone and an fp64 one is *not* narrowed.  Getting that wrong in either
    direction changes which computation the benchmark performs.

    Done outside :class:`_TritonConv3dFn` so the cast is an ordinary autograd
    node: the weight gradient then arrives back at the fp32 parameter through
    it, exactly as it does on the MIOpen rung.
    """
    if tensor is None or dtype is None or tensor.dtype is not torch.float32:
        return tensor
    return tensor.to(dtype)


def _cast_dtype(tensor, dtype):
    """The dtype :func:`_cast_operand` would leave ``tensor`` in."""
    if tensor is None:
        return None
    if dtype is None or tensor.dtype is not torch.float32:
        return tensor.dtype
    return dtype


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def _metadata_probe(shape, dtype, device):
    """A 5-D tensor with this shape, dtype and device, over one element.

    The ``is_supported*`` predicates read metadata only -- rank, shape, dtype,
    device, ``is_cuda`` -- and never a stride, a value or a contiguity.  That
    lets the gate ask about the bf16 operands autocast will produce without
    materializing them, which would cost a full-size copy and be wasted whenever
    the answer is no.  ``is_supported_all`` uses the same shortcut one level
    down, for the *gradient* this forward will later be handed and which does not
    exist yet -- a forward served by Triton whose backward cannot be is a trap,
    not a fallback, so that question has to be asked here and now.

    ``expand`` gives every dimension a stride of 0, so the result is safe to
    read and useless to compute with; nothing computes with it.  If a predicate
    ever grows a stride or contiguity test it will see those zeros and answer
    ``False``, routing the call to MIOpen -- the conservative direction.
    ``tests/test_conv3d.py::test_metadata_probe_answers_like_a_real_tensor``
    pins the agreement so the shortcut cannot rot silently.
    """
    key = (dtype, device)
    base = _PROBE_BASES.get(key)
    if base is None:
        base = torch.empty((1, 1, 1, 1, 1), dtype=dtype, device=device)
        _PROBE_BASES[key] = base
    return base.expand(tuple(int(v) for v in shape))


def _out_spatial(in_spatial, kernel, stride, padding, dilation):
    """PyTorch's output extents for a non-transposed convolution.

    No caller since the block-list was emptied on 2026-08-04 -- its only user was
    :func:`_policy_declines`'s small-``M`` rule.  Kept because a future entry
    there is the reason that function still takes ``stride``/``padding``/
    ``dilation``, and because two docstrings warn about what this returns for a
    *transposed* operator (the input volume over 8 at ``k == s == 2``), a warning
    that needs the thing it warns about to exist.
    """
    return tuple(
        (i + 2 * p - d * (k - 1) - 1) // s + 1
        for i, k, s, p, d in zip(in_spatial, kernel, stride, padding, dilation)
    )


class _HaloPlan:
    """Which dims this call exchanges, and what the kernel is left holding.

    Built once per forward by :func:`_halo_plan` and read by everything
    downstream -- the ``is_supported*`` probes, the policy block-list,
    :class:`_Halo3d` and :class:`_TritonConv3dFn` -- so that exactly one function
    decides which dims are split and nothing re-derives it.  An empty
    ``exchanges`` is the unsharded case: no node is applied and the kernel sees
    the module's own padding, which is stage 1 unchanged.
    """

    __slots__ = ("strategy", "exchanges", "padding", "input_shape")

    def __init__(self, strategy, exchanges, padding, input_shape):
        #: The ``ParallelStrategy``; :class:`_Halo3d` reads ``shard_ind``,
        #: ``num_shards`` and ``shard_to_rank`` off it.
        self.strategy = strategy
        #: ``(dim_index, dim, halo)`` per dim actually split, in ``shard_dim``
        #: order -- the order ``distconv_forward`` exchanges in and the order
        #: ``distconv_backward`` folds back down.
        self.exchanges = exchanges
        #: The module's padding with each exchanged dim's entry zeroed, exactly
        #: as ``distconv_forward`` mutates the caller's list.
        self.padding = padding
        #: The shape the kernel sees: the local shard, plus ``2 * halo`` on each
        #: exchanged dim.
        self.input_shape = input_shape


def _halo_plan(dc_input, strategy, x, weight, stride, padding, dilation):
    """The halo this ``DCTensor`` needs, or ``None`` if it must go to MIOpen.

    Returns a plan only when every fact the exchange rests on has been *read and
    checked*, and ``None`` for anything it could not read.  The asymmetry is
    deliberate: a false negative costs a convolution its fast kernel, a false
    positive returns a wrong gradient at every shard boundary of a large run,
    which is the worst outcome available here.

    The strategy-wide facts, each matching a line of ``distconv.py``:

    * ``shard_dim`` has the same length as ``num_shards``, because
      ``distconv_forward`` indexes ``num_shards`` and ``_is_periodic`` by
      position in ``shard_dim`` and a mismatch means the two disagree about which
      axis is which.  ``shard_ind`` likewise, since the exchange indexes it the
      same way.  And no axis may be named twice: two entries for one dim would
      exchange it twice and count the neighbour's slab twice.
    * no axis is periodic.  Periodicity is the one case where even a single
      shard exchanges: ``shard_ind == 0 and is_periodic`` posts a send and a
      receive to itself, so the halo is the tensor's own opposite face rather
      than zeros, and the padding becomes ``_periodic_shard_padding`` instead of
      0.  ScaFFold never sets it (it calls ``F.pad`` in the default constant
      mode), which is a reason to check rather than a reason to assume.
    * the shapes and the padding are the ordinary 5-D triples this module can
      reason about at all.

    Then, per axis.  An axis with ``num_shards == 1`` is **skipped**: its
    ``shard_ind`` is 0, neither ``shard_ind > 0`` nor
    ``shard_ind < num_shards - 1`` holds, no ``P2POp`` is posted and the
    ``zeros_like`` receive buffers stay zero -- so DistConv's ``cat`` there is
    ``cat(zeros, x, zeros)``, which at ``padding = 0`` is the same arithmetic as
    the module's own ``padding = k // 2``, and measured bitwise so.  It keeps its
    padding and costs nothing.

    An axis with ``num_shards > 1`` is exchanged, and only after checking what
    ``check_is_distconv_supported`` checks plus what this spelling of the
    exchange additionally needs:

    * ``2 <= dim < 5``: a spatial dim of a 5-D tensor.  ``ParallelStrategy``
      already rejects 0 and 1, but ``padding[dim - 2]`` is indexed here.
    * the kernel extent on that dim is **odd**.  An even kernel gives
      ``halo_size == 0`` in DistConv, which is only correct for the strided
      tiling ``check_is_distconv_supported`` then insists on; that is not a case
      this module has reasoned about, so it declines rather than guesses.
    * the padding on that dim is exactly ``k // 2`` ("same"), and the stride and
      dilation are 1.  Those three are what make "the halo'd extent at padding 0"
      equal to "the global volume's slice for this shard": with ``k = 2h + 1``
      the halo'd input is ``D_loc + 2h`` long and produces exactly ``D_loc``
      outputs at zero padding, aligned with the shard's global offset.
    * the shard is at least ``2 * halo`` thick, so the backward's two
      accumulation regions do not overlap.
    * ``shard_to_rank`` is callable, since the exchange has to name its
      neighbours.

    What is *not* checked here is whether a process group exists: that is a
    routing question rather than a property of the strategy, and it belongs with
    the rest of them in :func:`_use_triton`.
    """
    num_shards = getattr(strategy, "num_shards", None)
    shard_dim = getattr(strategy, "shard_dim", None)
    shard_ind = getattr(strategy, "shard_ind", None)
    if not isinstance(num_shards, (tuple, list)) or not isinstance(
        shard_dim, (tuple, list)
    ):
        return None
    if not num_shards or len(shard_dim) != len(num_shards):
        return None
    if not isinstance(shard_ind, (tuple, list)) or len(shard_ind) != len(num_shards):
        return None
    if len(set(shard_dim)) != len(shard_dim):
        return None
    for count in num_shards:
        if not isinstance(count, int) or count < 1:
            return None
    periodic = getattr(dc_input, "_is_periodic", None)
    if not isinstance(periodic, (tuple, list)) or len(periodic) != len(shard_dim):
        return None
    if any(periodic):
        return None
    if x.dim() != 5 or weight.dim() != 5:
        return None
    for triple in (stride, padding, dilation):
        if not isinstance(triple, (tuple, list)) or len(triple) != 3:
            return None

    exchanges = []
    plan_padding = list(int(p) for p in padding)
    plan_shape = list(int(s) for s in x.shape)
    for i, dim in enumerate(shard_dim):
        if num_shards[i] == 1:
            continue
        if not isinstance(dim, int) or not 2 <= dim < 5:
            return None
        index = shard_ind[i]
        if not isinstance(index, int) or not 0 <= index < num_shards[i]:
            return None
        kernel = int(weight.shape[dim])
        if kernel % 2 == 0:
            return None
        halo = kernel // 2
        if plan_padding[dim - 2] != halo:
            return None
        if int(stride[dim - 2]) != 1 or int(dilation[dim - 2]) != 1:
            return None
        if halo == 0:  # k == 1: no neighbour voxel is ever read
            continue
        if int(x.shape[dim]) < 2 * halo:
            return None
        exchanges.append((i, dim, halo))
        plan_padding[dim - 2] = 0
        plan_shape[dim] += 2 * halo

    if exchanges and not callable(getattr(strategy, "shard_to_rank", None)):
        return None
    return _HaloPlan(strategy, exchanges, tuple(plan_padding), tuple(plan_shape))


def _policy_declines(x_shape, w_shape, stride, padding, dilation):
    """Shapes the Triton rung is slower on: **none, today.**

    Empty since 2026-08-04, and the arguments are kept so a future entry has
    somewhere to go.  It held three rules, and re-measuring every one of them
    against the shipped kernels retired all three
    (``work/triton-conv/review/CONV_REMEASURE.md``; per-site sums over all
    three directions, kernel time with 95% intervals):

    * ``Cin == 3``, the stem, blocked at a quoted **0.53x**.  That figure was
      backward-weight alone.  Summed over the three directions the site is
      **0.93x**, because backward-data runs 1.58-1.70x the other way, and 0.93x
      on this site is **+0.13% to +0.19% of a step**.
    * ``k == 1``, the ``64 -> 6`` head, blocked because "the forward is 1.22x but
      backward-weight runs 0.73-1.14x".  Backward-data is **1.94-2.00x** and was
      never in that accounting: the site is **1.40x**.
    * ``M <= 4096 and Cout >= 512``, the small-``M`` class, blocked on two
      measured losses and extended by predicate to nine problems.  All nine now
      favour the rung, **1.17-1.65x**, worth **2.28% of a config-A step**.

    The rule that mattered was the last one, and its error was structural rather
    than stale: ``M`` is the *forward* GEMM's row count, but a decline keeps the
    whole site on MIOpen including its gradients, and backward-data is the
    forward contraction on a *permuted weight* -- a different GEMM, which wins
    1.31-2.47x on every problem the rule blocked.  A rule derived from one
    direction was deciding three.  **Anything added here inherits that trap**:
    either measure all three directions, or block per direction.

    Emptying it also removes the convolutions as a source of run-to-run
    variation, since MIOpen's backward-weight is not bitwise reproducible and
    the deterministic split-K path is the rung's default.

    **This function is for the non-transposed operator only.**  Every term a rule
    might use reads a different quantity for the other one: ``w_shape``'s channel
    axes are the other way round, and ``M`` is the *input* volume rather than
    ``_out_spatial``'s -- 8x larger at ``k == s == 2``.  See
    :func:`_transposed_policy_declines`, which is a separate function for
    exactly that reason.
    """
    return False


# ---------------------------------------------------------------------------
# The halo exchange
# ---------------------------------------------------------------------------


def _packed_slab(shape, dtype, device, zero=False):
    """A plain-contiguous NDHWC allocation, returned as ``(base, NCDHW view)``.

    ``base`` is what goes on the wire.  It matters that it is *plain* contiguous:
    ``forward_halo_exchange`` sends
    ``inner_halo_plus.contiguous(memory_format=channels_last_3d)``, which for
    ``C > 1`` is not plain-contiguous, and ``ProcessGroupGloo``'s send/recv
    rejects that -- which is the whole of "DistConv spatial sharding does not
    work under gloo".  The cause is the memory format, not the narrow, so
    allocating the wire buffer ourselves fixes it and lets the sharded suite run
    on a machine with no RCCL.

    ``view`` is the same storage addressed as ``(N, C, ...)``, and a permute of a
    contiguous ``(N, D, H, W, C)`` gives exactly ``channels_last_3d``'s strides
    -- so copying a shard's boundary slab into it is a straight copy of 1.6% of
    the activation and not a transpose.
    """
    n, c = int(shape[0]), int(shape[1])
    spatial = tuple(int(v) for v in shape[2:])
    allocate = torch.zeros if zero else torch.empty
    base = allocate((n, *spatial, c), dtype=dtype, device=device)
    return base, base.permute(0, 4, 1, 2, 3)


def _neighbour_ranks(strategy, dim_index):
    """The ranks holding the shards either side of this one along ``dim_index``.

    ``shard_to_rank`` on a copy of ``shard_ind``, which is what
    ``forward_halo_exchange`` does; only ever asked for a neighbour that exists,
    so its wrap-around branches (which exist for periodicity) are not reached.
    """
    minus = list(strategy.shard_ind)
    minus[dim_index] -= 1
    plus = list(strategy.shard_ind)
    plus[dim_index] += 1
    return strategy.shard_to_rank(minus), strategy.shard_to_rank(plus)


def _exchange_forward(x, strategy, dim_index, dim, halo):
    """``distconv.forward_halo_exchange`` for one dim, on plain-contiguous wires.

    Same sends, same receives, same posting order, and the same result: the
    local shard with each neighbour's ``halo``-thick boundary slab concatenated
    onto the matching face, and zeros where there is no neighbour.  The
    differences are both deliberate: the wire buffers are plain-contiguous (see
    :func:`_packed_slab`), and the output is allocated in the layout the kernel
    wants rather than left to ``torch.cat``'s memory-format inference.
    """
    shard_ind = strategy.shard_ind[dim_index]
    num_shards = strategy.num_shards[dim_index]
    minus_rank, plus_rank = _neighbour_ranks(strategy, dim_index)

    slab_shape = list(x.shape)
    slab_shape[dim] = halo
    recv_minus, recv_minus_view = _packed_slab(
        slab_shape, x.dtype, x.device, zero=True
    )
    recv_plus, recv_plus_view = _packed_slab(slab_shape, x.dtype, x.device, zero=True)

    ops = []
    if shard_ind > 0:
        send_minus, view = _packed_slab(slab_shape, x.dtype, x.device)
        view.copy_(x.narrow(dim, 0, halo))
        ops += [
            dist.P2POp(dist.irecv, recv_minus, minus_rank),
            dist.P2POp(dist.isend, send_minus, minus_rank),
        ]
    if shard_ind < num_shards - 1:
        send_plus, view = _packed_slab(slab_shape, x.dtype, x.device)
        view.copy_(x.narrow(dim, x.size(dim) - halo, halo))
        ops += [
            dist.P2POp(dist.isend, send_plus, plus_rank),
            dist.P2POp(dist.irecv, recv_plus, plus_rank),
        ]
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()

    halo_shape = list(x.shape)
    halo_shape[dim] = x.size(dim) + 2 * halo
    _, out = _packed_slab(halo_shape, x.dtype, x.device)
    out.narrow(dim, halo, x.size(dim)).copy_(x)
    out.narrow(dim, 0, halo).copy_(recv_minus_view)
    out.narrow(dim, out.size(dim) - halo, halo).copy_(recv_plus_view)
    return out


def _exchange_backward(grad, strategy, dim_index, dim, halo):
    """``distconv.backward_halo_exchange`` for one dim, on plain-contiguous wires.

    The transpose of :func:`_exchange_forward`: the gradient of a value this rank
    borrowed belongs to the rank it was borrowed from, so each outer boundary
    slab of ``grad`` goes back to the neighbour that supplied it and is
    *accumulated* into that neighbour's inner region.  ``grad`` is mutated in
    place and a narrowed view of it is returned, which is what
    ``distconv_backward`` already hands back today -- safe because the only
    producer of this gradient is :meth:`_TritonConv3dFn.backward`, whose output
    has no other consumer.
    """
    shard_ind = strategy.shard_ind[dim_index]
    num_shards = strategy.num_shards[dim_index]
    minus_rank, plus_rank = _neighbour_ranks(strategy, dim_index)

    slab_shape = list(grad.shape)
    slab_shape[dim] = halo
    recv_minus, recv_minus_view = _packed_slab(
        slab_shape, grad.dtype, grad.device, zero=True
    )
    recv_plus, recv_plus_view = _packed_slab(
        slab_shape, grad.dtype, grad.device, zero=True
    )

    ops = []
    if shard_ind > 0:
        send_minus, view = _packed_slab(slab_shape, grad.dtype, grad.device)
        view.copy_(grad.narrow(dim, 0, halo))
        ops += [
            dist.P2POp(dist.irecv, recv_minus, minus_rank),
            dist.P2POp(dist.isend, send_minus, minus_rank),
        ]
    if shard_ind < num_shards - 1:
        send_plus, view = _packed_slab(slab_shape, grad.dtype, grad.device)
        view.copy_(grad.narrow(dim, grad.size(dim) - halo, halo))
        ops += [
            dist.P2POp(dist.isend, send_plus, plus_rank),
            dist.P2POp(dist.irecv, recv_plus, plus_rank),
        ]
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()

    inner = grad.narrow(dim, halo, grad.size(dim) - 2 * halo)
    inner.narrow(dim, 0, halo).add_(recv_minus_view)
    inner.narrow(dim, inner.size(dim) - halo, halo).add_(recv_plus_view)
    return inner


class _Halo3d(torch.autograd.Function):
    """The halo exchange, as an autograd node of its own above the kernel's.

    Two ``Function``s rather than one, deliberately.  The convolution node is
    then shard-agnostic and *identical* whether or not anything is sharded -- it
    is handed an input and a padding and knows nothing about either -- so the
    unsharded path stays exactly what it was, this node is separately testable
    against ``distconv.forward_halo_exchange``/``backward_halo_exchange``, and
    the saved-tensor set stays uniform, which is what the per-rung latch's
    argument needs.

    **Nothing is saved.**  ``ctx`` carries the :class:`_HaloPlan`, which is
    Python objects: the strategy, and which dims were exchanged at what width.
    Applied only when there is something to exchange, so at
    ``dc_num_shards = (1, 1, 1)`` it is not in the graph at all.
    """

    @staticmethod
    def forward(ctx, x, plan):
        ctx.plan = plan
        for dim_index, dim, halo in plan.exchanges:
            x = _exchange_forward(x, plan.strategy, dim_index, dim, halo)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output
        # Same order as the forward, which is also the order distconv_backward
        # uses -- it iterates shard_dim forwards in both directions.
        for dim_index, dim, halo in ctx.plan.exchanges:
            grad = _exchange_backward(grad, ctx.plan.strategy, dim_index, dim, halo)
        return grad, None


# ---------------------------------------------------------------------------
# Eligibility, continued
# ---------------------------------------------------------------------------


def _routing_declines(x, dc_input, plan, proven):
    """The conditions *both* ladders decline on, in one place.

    ``True`` means "this call must not take a Triton rung", for a reason that is
    not about which convolution it is: the override, the latch, a functorch
    layer, an unknown tensor subclass, the device, a sharding this module could
    not prove it can serve, and the layout.  Everything operator-specific --
    the module's own attributes, the performance block-list and which
    ``is_supported*`` to ask -- stays with the caller, because
    :class:`FastConv3d` and :class:`FastConvTranspose3d` differ on every one of
    them: they read the weight's channel axes in opposite orders, their measured
    losses are different shapes of a different kernel, and their gates take
    different arguments.

    Written once rather than copied because each clause below is a correction
    with a measured failure behind it (the module docstring says which), and a
    copy would drift away from them.  The one thing the split cost is ordering:
    the layout test used to run *after* ``FastConv3d``'s module-level checks and
    now runs before them.  Both are pure predicates returning ``False``, so only
    the cost of reaching the answer moved, and it moved towards the cheaper end.
    """
    if _triton_override is False:
        return True
    if _triton_failed and not proven:
        return True
    if _functorch_active():
        return True
    # An unknown __torch_dispatch__ wrapper has unknown semantics and keeps
    # MIOpen.  DistConv's DCTensor never reaches this check -- forward() has
    # already unwrapped to the local shard, and its sharding is checked through
    # ``plan`` -- so anything rejected here is a subclass nobody has reasoned
    # about.  Note the identity test, not isinstance: is_supported only asks
    # isinstance and would accept any of them.
    if type(x) is not torch.Tensor:
        return True
    if not x.is_cuda:
        return True
    # ...and not merely *a* CUDA device: the one every launch configuration in
    # both packages was raced on.  Unlike every other clause here, the thing
    # this one prevents is a right answer at an unknown speed -- see
    # :func:`~ScaFFold.unet._rungs._platform_declines`, which is also where the
    # "explicit opt-in wins" decision is argued.  Cached per device, so this is
    # a dictionary lookup after the first call.
    if _platform_declines(x.device, _triton_override):
        return True
    if dc_input is not None and plan is None:
        return True
    # An exchange needs a process group.  A real ``ParallelStrategy`` cannot be
    # constructed without one, so this only fires for a hand-built strategy or a
    # torch built without distributed -- but it is the call that would otherwise
    # fail inside ``dist`` rather than routing to MIOpen.  A transposed plan
    # never has an exchange in it, so this clause is the non-transposed ladder's
    # alone; it is here because it is a property of the *plan*, not of the
    # operator.
    if plan is not None and plan.exchanges:
        if not (dist.is_available() and dist.is_initialized()):
            return True
    # A relayout is a correctness no-op -- every entry point calls
    # ``.contiguous(memory_format=channels_last_3d)`` itself -- but it is a
    # full-size hidden copy, and the whole point of the rung is that ScaFFold's
    # activations are already in that layout.  Declining is cheaper than paying
    # it silently.  Note that DistConv's narrowed ``_tensor`` stops being
    # channels-last at ``local_batch_size > 1``, which is a supported config key,
    # so this is a live branch and not a formality.
    if x.dim() != 5 or not x.is_contiguous(memory_format=torch.channels_last_3d):
        return True
    return False


def _use_triton(module, x, dc_input, plan, proven=False):
    """Whether this particular call should take the Triton rung.

    ``x`` is the tensor the halo would be added to -- a ``DCTensor``'s local
    shard, or the input itself -- read as a plain attribute, so the tests examine
    real strides and dtypes rather than a wrapper's mirrored metadata.
    ``dc_input`` is the wrapper or ``None``, and ``plan`` is what
    :func:`_halo_plan` made of its strategy: ``None`` for a plain tensor, and
    ``None`` *also* for a ``DCTensor`` whose sharding this module could not
    prove it can serve -- which is why the two are passed separately.
    ``proven`` is the caller's "this module has already had a call served by this
    rung", which keeps a proven module on it even after a global latch; see the
    module docstring's "Latches".

    **The predicates are asked about the tensor the kernel will actually see**,
    which once a dim is split is the halo'd one at the reduced padding, not the
    local shard at the module's own padding.  The two differ in extent and in
    output shape, so asking about the wrong one would gate on a call that never
    happens.

    Ordered so the cheap local tests come first and the package import last, and
    so that nothing is allocated or cast before the answer is known.
    """
    if _routing_declines(x, dc_input, plan, proven):
        return False
    # Module-level conditions the kernels have no argument for at all.  A
    # transposed convolution is served by :class:`FastConvTranspose3d` and by a
    # different set of entry points (``is_supported`` does not even take a
    # ``transposed`` parameter, and passing a ``(Cin, Cout, k, k, k)`` weight to
    # the gates below would be *accepted* whenever the two channel counts match
    # and would compute a different operator), and a non-zeros padding mode is an
    # F.pad that happens in nn.Conv3d._conv_forward and that this rung would
    # skip.  Neither can occur for a FastConv3d built by ``unet_parts``; both are
    # checked because the class is a public drop-in.
    if module.transposed or module.output_padding != (0, 0, 0):
        return False
    if module.padding_mode != "zeros" or not isinstance(module.padding, tuple):
        return False
    weight = module.weight
    # What the kernel is handed: the halo'd extent and the padding left over
    # after each exchanged dim's was zeroed.  Identical to ``x.shape`` and
    # ``module.padding`` whenever nothing is split -- which is every call at
    # ``dc_num_shards = (1,1,1)`` and no call at (2,1,1) or (4,1,1), where D is
    # exchanged and H and W keep ``padding = 1``.  So the kernel sees
    # ``(1,1,1)`` unsharded and ``(0,1,1)`` sharded, and is padded either way.
    kernel_shape = plan.input_shape if plan is not None else tuple(x.shape)
    kernel_padding = plan.padding if plan is not None else module.padding
    if _policy_declines(
        kernel_shape, weight.shape, module.stride, kernel_padding, module.dilation
    ):
        return False

    # The predicates are cheap and side-effect free: attribute reads and integer
    # arithmetic, no allocation, no launch.  The broad catch is right *here* and
    # nowhere else in this module: a predicate that cannot answer has a correct
    # answer available ("no"), it has done no work anyone can observe, and the
    # failure is a routing miss rather than a broken kernel -- so it must not
    # latch the rung off.
    try:
        conv = _get_triton_module()
        dtype = _autocast_dtype(x)
        x_probe = _metadata_probe(kernel_shape, _cast_dtype(x, dtype), x.device)
        w_probe = _metadata_probe(
            weight.shape, _cast_dtype(weight, dtype), weight.device
        )
        bias = module.bias
        # The bias is Cout elements; casting it for real is cheaper than
        # explaining a probe that also has to have stride 1.
        bias_probe = _cast_operand(bias, dtype) if bias is not None else None
        args = (module.stride, kernel_padding, module.dilation, module.groups)
        # ``is_supported_all``, not ``is_supported``: every direction the
        # backward will need, asked once.  The forward's own gate does not imply
        # the other two and the package says so -- ``stride > 1`` is served by
        # the forward and by ``is_supported_bwd_weight`` and refused by
        # ``is_supported_bwd_data``, whose kernel-free formulation only holds at
        # unit stride.  Taking the rung on the forward's answer alone would build
        # a graph node whose backward ``triton_conv3d`` cannot answer, and by
        # then MIOpen is no longer an option for it.
        return bool(conv.is_supported_all(x_probe, w_probe, bias_probe, *args))
    except Exception as e:
        _warn_once_about_the_predicate(e)
        return False


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


def _aten_backward(
    grad_output,
    x,
    weight,
    stride,
    padding,
    dilation,
    mask,
    has_bias,
    transposed=False,
    output_padding=(0, 0, 0),
):
    """MIOpen's ``convolution_backward``, for the callers that need it.

    Used as either backward rung's fallback and, in tests, as the reference the
    Triton gradients are compared against.  ``bias_sizes`` is required even when
    the mask says no bias gradient is wanted, so it is always supplied -- and it
    is the *output* channel count, which is ``weight.shape[0]`` for an ordinary
    convolution and ``weight.shape[1]`` for a transposed one, because PyTorch
    stores the two parameters with their channel axes the other way round.

    ``transposed`` and ``output_padding`` are the aten op's own arguments passed
    through rather than a mode flag: this is one call to one operator, and both
    ladders want it with the arguments their module holds.
    """
    cout = int(weight.shape[1 if transposed else 0])
    return torch.ops.aten.convolution_backward(
        grad_output,
        x,
        weight,
        [cout] if has_bias else None,
        list(stride),
        list(padding),
        list(dilation),
        bool(transposed),
        list(output_padding),
        1,  # groups
        list(mask),
    )


class _TritonConv3dFn(torch.autograd.Function):
    """The Triton rung's autograd node.

    A plain ``torch.autograd.Function`` rather than
    ``torch.library.custom_op`` + ``register_autograd``, which is the shape
    ``triton_group_norm`` uses.  That module measured the custom-op route at
    +0.054 ms of dispatcher and +0.338 ms of autograd node against 0.065 ms for
    an empty ``Function``, and paid it so its op would compose with
    ``torch.compile`` and with ``DCTensor``'s ``__torch_dispatch__``.  Neither
    reason applies here: nothing in ScaFFold compiles the convolutions, and
    ``DCTensor`` dispatch is precisely what this rung is bypassing -- a real
    dispatcher op would be intercepted by DistConv's generic unwrap, which is
    the wrong behaviour, because that path has no halo.

    Operands arrive already cast (see :func:`_cast_operand`), so this node is
    dtype-transparent and its gradients flow back to the fp32 parameters through
    the cast's own backward.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation):
        conv = _get_triton_module()
        y = conv.conv3d_forward(x, weight, bias, stride, padding, dilation, 1)
        # Saved *after* the launch, deliberately.  The ladder retries a failed
        # call on MIOpen, and that is only safe while the failing region has
        # done nothing autograd can observe; ``triton_group_norm`` gets the same
        # property from saving in ``_setup_context``, which runs after its
        # forward returns.  Nothing between here and the return can raise.
        ctx.save_for_backward(x, weight)
        ctx.conv_args = (stride, padding, dilation, bias is not None)
        return y

    # The gradients below are computed by hand from the saved operands, not by
    # composing differentiable ops, so nothing here can be differentiated again.
    # Without this decorator that is *silent*: ``create_graph=True`` returns a
    # gradient with no ``grad_fn``, and a second backward through it contributes
    # zero instead of raising -- which is a wrong number rather than an error, in
    # a project where those are the defects that keep nearly shipping.  With it,
    # the second differentiation says so.  MIOpen's rung *is* twice
    # differentiable, so this is a real difference between the rungs and it is
    # why the block-list emptying (2026-08-04) is what surfaced it: the stem is
    # the site whose input gradient a double backward actually reaches.
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        global _triton_failed
        x, weight = ctx.saved_tensors
        if type(x) is not torch.Tensor:
            # The one detector available for a rung flip across a checkpoint
            # recompute.  ``_default_meta_extractor`` compares shape, dtype and
            # device, which are identical between the rungs, so torch's own
            # check passes and a DCTensor lands in this slot instead of the
            # plain tensor this forward saved.  Without this the next line dies
            # inside DistConv with an AttributeError about ``_parallel_strategy``
            # that names neither checkpointing nor the rung.  One type check per
            # backward.
            raise RuntimeError(
                "FastConv3d: the tensor saved for backward is a "
                f"{type(x).__name__}, not a torch.Tensor. The Triton rung and "
                "the MIOpen rung save different things, so this module's "
                "forward and its checkpoint recompute were served by different "
                "rungs. Set SCAFFOLD_CONV_TRITON=0 to pin the whole run to "
                "MIOpen."
            )
        stride, padding, dilation, has_bias = ctx.conv_args
        needs_x, needs_w, needs_b = ctx.needs_input_grad[:3]
        # One relayout for both directions: conv3d_backward_data and
        # conv3d_backward_weight each call ``.contiguous(channels_last_3d)`` on
        # it, and doing it here makes the second one free.
        grad_output = grad_output.contiguous(memory_format=torch.channels_last_3d)

        conv = _get_triton_module()
        try:
            grad_x = (
                conv.conv3d_backward_data(
                    grad_output, weight, x.shape, stride, padding, dilation, 1
                )
                if needs_x
                else None
            )
            grad_w = (
                conv.conv3d_backward_weight(
                    x, weight.shape, grad_output, stride, padding, dilation, 1
                )
                if needs_w
                else None
            )
        except _triton_kernel_failures() as e:
            # A backward-direction failure *can* be answered by MIOpen, and the
            # argument that forbids it in the forward does not reach here.  That
            # argument is about the saved set: a proven module answering a
            # checkpoint recompute from the other rung saves different tensors
            # than the graph node holds.  This code is not a recompute -- it is
            # the node itself, running once, consuming exactly the tensors it
            # was given.  Nothing downstream can tell which kernel produced the
            # gradients, so degrading is a fallback here in the full sense, and
            # it matters: the backward-weight kernel is a separate compilation
            # from the forward's, so it can raise OutOfResources on a call whose
            # forward compiled cleanly.  The rung is still latched, so no module
            # that has not used it will try it again.
            first = not _triton_failed
            _triton_failed = True
            if first:
                _warn_rung_failure(
                    "Triton conv3d backward", e, "MIOpen kernel", TRITON_ENV_VAR
                )
            grad_x, grad_w, grad_b = _aten_backward(
                grad_output,
                x,
                weight,
                stride,
                padding,
                dilation,
                (needs_x, needs_w, has_bias and needs_b),
                has_bias,
            )
            return grad_x, grad_w, grad_b, None, None, None

        # d(bias) is the sum of grad_output over every axis but the channel one,
        # whatever the forward kernel was.  Only the segmentation head has a
        # bias and it is on the policy block-list, so this is unreachable in
        # ScaFFold today -- but ``is_supported`` accepts a bias, and the class is
        # a public drop-in, so it has to be right.
        grad_b = grad_output.sum(dim=(0, 2, 3, 4)) if (has_bias and needs_b) else None
        return grad_x, grad_w, grad_b, None, None, None


class FastConv3d(nn.Conv3d):
    """``nn.Conv3d`` with a Triton GPU kernel and MIOpen behind it.

    Identical state: ``weight`` of shape ``(Cout, Cin, kd, kh, kw)`` and the
    optional ``bias`` of shape ``(Cout,)``, both from ``nn.Conv3d.__init__``,
    no buffers and no extra attributes that are parameters or ``state_dict``
    keys.  ``__init__`` is not overridden at all, so the constructor signature is
    ``nn.Conv3d``'s by construction and state dicts are interchangeable in both
    directions with a plain ``nn.Conv3d`` model.

    **The weight is used exactly as it lies.**  There is no transform, no cache
    and no stride contract: ``triton_conv3d`` addresses the parameter through its
    strides, and ``worker.py``'s ``model.to(device,
    memory_format=channels_last_3d)`` already puts every 5-D parameter in the
    layout the kernel wants.  An earlier design held the parameter in RSCK order
    behind ``state_dict`` hooks; it was measured 0.159 ms/step better in the
    kernels and 2.615 ms/step *worse* in the optimizer, and deleted.

    DistConv's ``DCTensor`` takes the fast kernel by being unwrapped to its local
    shard in front of it, with the halo this module exchanges itself where a dim
    is actually split -- which, unlike GroupNorm, is a real computation and not a
    formality.  See the module docstring, and :func:`_halo_plan` for the check
    that decides it.
    """

    #: Per-module "a call has been answered by this ladder" -- by the Triton
    #: kernel, or by the MIOpen fallback :meth:`_triton_forward` runs on an
    #: already-exchanged tensor, which saves the same set.  A global latch does
    #: not demote a module that has one, which is what keeps a checkpointed
    #: block's forward and its recompute on the same rung.  A plain class
    #: attribute, so it is not a parameter, a buffer or a state-dict key, and it
    #: costs nothing until the first success -- the instance attribute is only
    #: written on the False -> True edge, because nn.Module.__setattr__ is not
    #: free.
    _triton_ok = False

    def _triton_forward(self, local, plan=None):
        """Run the Triton rung -- and its fallback -- on an unwrapped tensor.

        ``plan`` is :func:`_halo_plan`'s answer, or ``None`` for a tensor that
        is not sharded at all.  The cast comes first, so the exchange carries the
        bf16 tensor autocast's dispatcher would have produced rather than the
        fp32 one the model holds -- which is both what DistConv does and half the
        bytes on the wire.

        **The exchange is above the retry; only the kernel call is inside it.**
        That split is what makes a Triton failure survivable once a dim is
        actually split.  ``_Halo3d`` runs before the kernel compiles, so by the
        time a ``TritonError`` arrives this rank has already posted the sends and
        receives its peers are matched against.  Re-running the call from the top
        would take it to :meth:`_miopen_forward` and therefore through
        ``distconv_forward``, which exchanges *again*: one more collective on
        this rank than on a peer whose kernel compiled, which desynchronises the
        mesh and hangs, or pairs this convolution's slabs with the next one's.
        So the fallback here consumes the tensor that has already been exchanged
        -- ``x``, at ``plan.padding``, which is precisely the pair
        :func:`_halo_plan` proved equal to the module's own padding on the
        unexchanged shard -- and swaps nothing but the kernel.  Without it a
        broken Triton install is *fatal* at ``num_shards > 1`` where it is merely
        slow at 1, and the whole point of the ladder is the opposite.

        At one shard nothing has been sent and the exception is re-raised
        unchanged, so :meth:`forward`'s handler re-runs the whole call on the
        rung that defines the semantics -- which is also the only rung that can
        be handed a ``DCTensor``.
        """
        dtype = _autocast_dtype(local)
        x = _cast_operand(local, dtype)
        weight = _cast_operand(self.weight, dtype)
        bias = _cast_operand(self.bias, dtype)
        padding = self.padding
        exchanged = plan is not None and bool(plan.exchanges)
        if exchanged:
            x = _Halo3d.apply(x, plan)
            padding = plan.padding
        try:
            return _TritonConv3dFn.apply(
                x, weight, bias, self.stride, padding, self.dilation
            )
        except _triton_kernel_failures() as e:
            if not exchanged:
                raise
            _latch_rung_failure(e)
            # The one call this fallback must not answer either, for the same
            # reason :meth:`forward`'s does not: a module already proven on the
            # rung, failing while a backward is in flight, is a checkpoint
            # recompute of a forward that ran on Triton, and the honest answer is
            # the original exception rather than a differently-produced tensor
            # substituted into a graph node that already holds one.
            if self._triton_ok and _replaying_a_forward():
                raise
            # ``nn.Conv3d.forward``'s own body for ``padding_mode="zeros"``,
            # which :func:`_use_triton` has already checked, on the halo'd
            # operands.  A plain tensor, so DistConv's ``__torch_dispatch__``
            # does not see it and no second exchange happens; ``grad_x`` still
            # flows back through :class:`_Halo3d` and ``grad_weight`` is still
            # computed against the halo'd input, which is what
            # ``distconv_backward`` does.
            return F.conv3d(
                x, weight, bias, self.stride, padding, self.dilation, self.groups
            )

    def _miopen_forward(self, input):
        """The semantics-defining rung.

        ``nn.Conv3d.forward`` unchanged, which is also the only rung that takes
        a ``DCTensor`` as it stands: DistConv's ``__torch_dispatch__`` intercepts
        the ``aten::convolution`` underneath and performs the halo exchange.
        """
        return super().forward(input)

    def forward(self, input):
        distconv = _dctensor_ops(input)
        # The eligibility checks look at the local shard for a DCTensor (the
        # peek is a plain attribute read, no autograd involvement) and at the
        # tensor itself otherwise.
        local_view = input._tensor if distconv is not None else input
        # None for a plain tensor, and None also for a DCTensor whose strategy
        # this module cannot prove it can serve -- _use_triton tells the two
        # apart from ``dc_input``.
        plan = (
            _halo_plan(
                input,
                input._parallel_strategy,
                local_view,
                self.weight,
                self.stride,
                self.padding,
                self.dilation,
            )
            if distconv is not None
            else None
        )

        if _use_triton(
            self,
            local_view,
            input if distconv is not None else None,
            plan,
            proven=self._triton_ok,
        ):
            triton_failures = _triton_kernel_failures()
            try:
                out = _run_local(
                    input, distconv, lambda local: self._triton_forward(local, plan)
                )
            except triton_failures as e:
                # A broken or mismatched Triton install, an unwritable JIT cache
                # or a tile that does not fit must cost speed, not a multi-node
                # run.  Every allowlisted exception is raised while compiling or
                # sizing a launch -- before the node saved anything -- so
                # retrying this same call on MIOpen is safe.
                _latch_rung_failure(e)
                # The one call this rung must not answer from MIOpen: a module
                # already proven on it, failing while a backward is in flight, is
                # a checkpoint recompute of a forward that *did* run on Triton.
                # The rungs save metadata-identical but structurally different
                # tensors under DistConv, so handing back MIOpen's result
                # substitutes a DCTensor into a slot holding a plain one and
                # fails later, inside DistConv, with a message about neither.
                # Degrading is for modules with nothing to contradict; here the
                # honest answer is the original exception.
                if self._triton_ok and _replaying_a_forward():
                    raise
                # The second call it must not answer from MIOpen: one whose halo
                # has already been exchanged.  ``_miopen_forward`` goes through
                # ``distconv_forward``, which exchanges *again* -- one more
                # collective on this rank than on a peer whose kernel compiled,
                # which desynchronises the whole mesh and hangs, or silently
                # pairs this convolution's slabs with the next one's.  Such a
                # call is normally answered inside ``_triton_forward``, on the
                # tensor that was already exchanged, and never reaches here; this
                # is the backstop for a failure raised *outside* that try block
                # (the cast, the exchange itself, ``DCTensor.from_shard``), none
                # of which runs Triton today.  It is unconditional because the
                # invariant is: once this rank has sent a slab, it does not enter
                # a code path that sends another.
                if plan is not None and plan.exchanges:
                    raise
            else:
                # Set for a call ``_triton_forward`` answered from MIOpen on the
                # exchanged tensor too, which is not a slip: the flag pins the
                # module to *this ladder*, and what a checkpoint recompute has to
                # agree about is the saved set, not which kernel produced the
                # values.  Both of this ladder's answers save the halo'd input,
                # unwrapped, above autograd; ``_miopen_forward``'s saves what
                # DistConv's dispatch saves.  Those are the two that must not be
                # mixed.
                if not self._triton_ok:
                    self._triton_ok = True
                return out

        return self._miopen_forward(input)


# ---------------------------------------------------------------------------
# The transposed ladder
# ---------------------------------------------------------------------------


def _transposed_policy_declines(x_shape, w_shape):
    """Shapes the transposed Triton rung is *slower* on: **none, today**.

    The peer of :func:`_policy_declines`, and a separate function rather than a
    flag on that one, because two of its three entries would read the wrong
    number here and the third is a measurement of a different kernel:

    * ``w_shape`` is a ``ConvTranspose3d`` parameter, ``(Cin, Cout, kd, kh,
      kw)``.  Its channel axes are the other way round, so
      ``_policy_declines``'s ``cout, cin = w_shape[0], w_shape[1]`` reads each
      one as the other -- at ``up1`` it would test ``Cin == 3`` against 512 and
      ``Cout >= 512`` against 1024.
    * ``M``, the forward GEMM's row count, is ``N * prod(in_spatial)``: this
      operator's windows *tile* the output, so its GEMM has one row per **input**
      voxel and the taps are in N.  ``_out_spatial`` computes the non-transposed
      extents, which at ``k == s == 2`` are half the input's per axis -- an ``M``
      8x too small.
    * and the small-``M`` cliff those numbers describe is a property of
      ``gather_gemm``'s tuning, not of this kernel, whose N axis carries
      ``Cout * taps`` and is 8x wider for it.

    So the four sites were measured directly rather than inherited
    (``work/triton-conv/m5_shipped_{fwd,bwd-data,bwd-weight}.json``, config A,
    Triton over MIOpen per direction):

    ===========================  =====  ========  ==========
    site                           fwd  bwd-data  bwd-weight
    ===========================  =====  ========  ==========
    up1 ``1024 -> 512`` @ 8^3    1.26x     1.43x       0.94x
    up2 ``512 -> 256``  @ 16^3   1.66x     1.54x       1.12x
    up3 ``256 -> 128``  @ 32^3   2.44x     1.36x       1.22x
    up4 ``128 -> 64``   @ 64^3   3.31x     1.00x       1.33x
    ===========================  =====  ========  ==========

    Every site wins overall, so nothing is blocked.  Worth stating explicitly:
    the small-``M`` rule would have blocked ``up1`` even with ``M`` read
    correctly (``M = 512``, ``Cout = 512``), and the measurement says it should
    not.  The one sub-unity cell is ``up1``'s backward-weight at 0.94x, against
    +1.26x and +1.43x on the same site's other two directions.

    Checked again at the level that decides it, because those are *kernel* times
    and this ladder's per-call cost is not zero: on the real model at config A,
    interleaved, 12 steps per arm
    (``work/triton-conv/bin/convT_config_a.py``), the whole ladder is worth
    **0.51 ms/step** and a variant that keeps only ``up3``/``up4`` on the rung is
    worth 0.65 -- a 0.14 ms difference inside a 1.5 ms spread, i.e. not a
    measurement, so there is nothing here to write an entry from.  What *is*
    measurable is that the gap between 0.51 and the 0.9 ms the kernel times
    project is per-call Python and launch overhead (~0.10 ms per forward and
    ~0.15 ms per backward, measured pipelined), which is a property of the
    adapter rather than of any shape -- so it belongs in a profile, not in a
    block-list.

    Both arguments are unused today and are taken anyway: they are what an entry
    would be written in terms of, and taking them keeps this call site the same
    shape as the other ladder's.  ``tests/test_conv3d.py`` asserts that the
    answer is ``False`` at all four sites, so an entry added here cannot silently
    turn the rung off.
    """
    return False


def _transposed_halo_plan(dc_input, strategy, x, weight, padding):
    """The halo a ``k == s`` transposed convolution needs: **none, ever**.

    Returns a :class:`_HaloPlan` with an empty ``exchanges``, or ``None`` if any
    fact it rests on could not be read -- the same asymmetry :func:`_halo_plan`
    documents, and for the same reason.

    Where :func:`_halo_plan` has to *reproduce* an exchange, this one has to
    prove there is not one, and the proof has two halves:

    * **The operator needs no neighbour voxel.**  At ``kernel == stride`` and no
      padding the map ``(d, kd) -> d*k + kd`` is a bijection, so output voxel
      ``d*k + kd`` reads input voxel ``d`` and nothing else.  A shard that holds
      a contiguous block of input voxels therefore holds everything its own
      output block needs, at any shard count.  The gate
      (``is_supported_transposed``) pins ``k == s``, ``padding == 0``,
      ``output_padding == 0`` and ``dilation == 1``, which is exactly that case.
    * **DistConv agrees, so the two rungs compute the same thing.**  Its
      ``distconv_forward`` sets ``halo_size = kernel_size // 2 if odd else 0``,
      and ``forward_halo_exchange``/``backward_halo_exchange`` both return their
      argument unchanged at ``halo_size == 0``.  With ``k = 2`` on every axis the
      MIOpen rung therefore also runs on the bare local shard -- no ``cat``, no
      ``P2POp``, no padding rewrite -- at *every* shard count, which is what
      makes the fallback and the fast rung the same computation rather than
      merely both defensible.

    Hence the parity test below.  An **odd** kernel on a split dim is declined,
    even though the bijection above holds for it too: DistConv would want a
    ``k // 2`` halo there and then refuse the problem outright in
    ``check_is_distconv_supported`` ("when kernel size is odd, padding must be
    equivalent to same", and this operator's padding is 0), so there would be no
    incumbent to agree with, no measurement, and a silent change from "the run
    raises" to "the run answers".  ScaFFold's four sites are all ``k = 2``.

    An axis with ``num_shards == 1`` is skipped before that test, as in
    :func:`_halo_plan`: it is not split, so nothing about it can matter.
    """
    num_shards = getattr(strategy, "num_shards", None)
    shard_dim = getattr(strategy, "shard_dim", None)
    shard_ind = getattr(strategy, "shard_ind", None)
    if not isinstance(num_shards, (tuple, list)) or not isinstance(
        shard_dim, (tuple, list)
    ):
        return None
    if not num_shards or len(shard_dim) != len(num_shards):
        return None
    if not isinstance(shard_ind, (tuple, list)) or len(shard_ind) != len(num_shards):
        return None
    if len(set(shard_dim)) != len(shard_dim):
        return None
    for count in num_shards:
        if not isinstance(count, int) or count < 1:
            return None
    # Periodicity is the one case where even a single shard exchanges, and it
    # rewrites the padding as well.  ScaFFold never sets it; this is a check
    # rather than an assumption for the same reason it is one in _halo_plan.
    periodic = getattr(dc_input, "_is_periodic", None)
    if not isinstance(periodic, (tuple, list)) or len(periodic) != len(shard_dim):
        return None
    if any(periodic):
        return None
    if x.dim() != 5 or weight.dim() != 5:
        return None
    if not isinstance(padding, (tuple, list)) or len(padding) != 3:
        return None

    for i, dim in enumerate(shard_dim):
        if num_shards[i] == 1:
            continue
        if not isinstance(dim, int) or not 2 <= dim < 5:
            return None
        index = shard_ind[i]
        if not isinstance(index, int) or not 0 <= index < num_shards[i]:
            return None
        if int(weight.shape[dim]) % 2 != 0:
            return None

    # Empty exchanges, and the module's own padding: the kernel is handed the
    # local shard exactly as it stands.  Carrying a plan at all -- rather than a
    # bare boolean -- is what lets :func:`_routing_declines` read "a DCTensor
    # this module could not prove it can serve" the same way for both ladders.
    return _HaloPlan(strategy, (), tuple(int(p) for p in padding), tuple(x.shape))


def _use_triton_transposed(module, x, dc_input, plan, proven=False):
    """Whether this transposed call should take the Triton rung.

    :func:`_use_triton`'s peer, with the same arguments and the same contract.
    What differs after :func:`_routing_declines` has answered the shared half:
    the module-level conditions are the mirror image (``transposed`` must be
    *true* here), the block-list is :func:`_transposed_policy_declines`, and the
    gate is ``is_supported_transposed_all`` -- all three directions at once, for
    the reason its own docstring gives.  Nothing about a halo appears, because a
    transposed plan never has an exchange in it.
    """
    if _routing_declines(x, dc_input, plan, proven):
        return False
    # The mirror of the check that sends a transposed module here in the first
    # place.  ``FastConvTranspose3d`` cannot be built any other way, but the
    # class is a public drop-in and ``transposed`` is a plain attribute: were it
    # false, ``is_supported_transposed`` would read a ``(Cout, Cin, k, k, k)``
    # weight as ``(Cin, Cout, ...)`` and accept it whenever the two channel
    # counts match, computing a different operator without a word.
    if not module.transposed:
        return False
    if module.padding_mode != "zeros":
        return False
    for triple in (module.padding, module.output_padding, module.stride):
        if not isinstance(triple, tuple) or len(triple) != 3:
            return False
    weight = module.weight
    if _transposed_policy_declines(tuple(x.shape), weight.shape):
        return False

    # As in _use_triton: the predicates are attribute reads and integer
    # arithmetic, and one that cannot answer has a correct answer available
    # ("no") and has done no observable work -- a routing miss, not a kernel
    # failure, so it must not latch the rung off.
    try:
        conv = _get_triton_module()
        dtype = _autocast_dtype(x)
        x_probe = _metadata_probe(x.shape, _cast_dtype(x, dtype), x.device)
        w_probe = _metadata_probe(
            weight.shape, _cast_dtype(weight, dtype), weight.device
        )
        bias = module.bias
        # Cast for real rather than probed: ``is_supported_transposed`` reads
        # the bias's *stride*, which an expanded probe reports as 0, so a probe
        # would answer "no" for every biased call -- which is all four of them.
        bias_probe = _cast_operand(bias, dtype) if bias is not None else None
        return bool(
            conv.is_supported_transposed_all(
                x_probe,
                w_probe,
                bias_probe,
                module.stride,
                module.padding,
                module.output_padding,
                module.dilation,
                module.groups,
            )
        )
    except Exception as e:
        _warn_once_about_the_predicate(e)
        return False


class _TritonConvTranspose3dFn(torch.autograd.Function):
    """The transposed rung's autograd node.

    :class:`_TritonConv3dFn`'s peer; the reasons for a plain
    ``autograd.Function`` and for saving *after* the launch are that class's and
    unchanged.  Two things are genuinely different:

    * **``grad_bias`` is on the path.**  All four transposed sites have one
      (``nn.ConvTranspose3d`` defaults to ``bias=True`` and ``unet_parts`` does
      not turn it off), where the only biased ordinary convolution is the
      segmentation head, which the block-list keeps on MIOpen.  So the reduction
      below is reached on every step of every run, not only by a test.
    * **The rung-flip hazard is wider.**  ``DCTensor`` mirrors its local shard's
      size, strides, dtype and device exactly, and this ladder never adds a halo
      -- so the tensor this node saves and the one ``nn.ConvTranspose3d.forward``
      saves under DistConv are metadata-identical at *every* shard count, not
      only at one.  ``FastConv3d`` gets a loud ``CheckpointError`` once a dim is
      split, because its Triton rung saves the halo'd input and the extent
      differs; here there is nothing to differ, so the per-module latch is the
      whole of the defence and the type check below is the only detector.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, dilation):
        conv = _get_triton_module()
        y = conv.conv_transpose3d_forward(
            x, weight, bias, stride, padding, output_padding, dilation, 1
        )
        # After the launch: the ladder retries a failed call on MIOpen, which is
        # only safe while the failing region has done nothing autograd can
        # observe.  Nothing between here and the return can raise.
        ctx.save_for_backward(x, weight)
        ctx.conv_args = (stride, padding, output_padding, dilation, bias is not None)
        return y

    # See :meth:`_TritonConv3dFn.backward`: hand-computed gradients, so a second
    # differentiation must raise rather than silently contribute zero.
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        global _triton_failed
        x, weight = ctx.saved_tensors
        if type(x) is not torch.Tensor:
            # See the class docstring: the two rungs save tensors that agree on
            # everything ``_default_meta_extractor`` compares, so a rung flip
            # across a checkpoint recompute passes torch's own check and dies
            # inside DistConv with an AttributeError about ``_parallel_strategy``
            # that names neither checkpointing nor the rung.  One type check per
            # backward buys an actionable message instead.
            raise RuntimeError(
                "FastConvTranspose3d: the tensor saved for backward is a "
                f"{type(x).__name__}, not a torch.Tensor. The Triton rung and "
                "the MIOpen rung save different things, so this module's "
                "forward and its checkpoint recompute were served by different "
                "rungs. Set SCAFFOLD_CONV_TRITON=0 to pin the whole run to "
                "MIOpen."
            )
        stride, padding, output_padding, dilation, has_bias = ctx.conv_args
        needs_x, needs_w, needs_b = ctx.needs_input_grad[:3]
        # One relayout for both directions, as in _TritonConv3dFn: each entry
        # point would call ``.contiguous(channels_last_3d)`` itself.
        grad_output = grad_output.contiguous(memory_format=torch.channels_last_3d)

        conv = _get_triton_module()
        try:
            grad_x = (
                conv.conv_transpose3d_backward_data(
                    grad_output,
                    weight,
                    x.shape,
                    stride,
                    padding,
                    output_padding,
                    dilation,
                    1,
                )
                if needs_x
                else None
            )
            grad_w = (
                conv.conv_transpose3d_backward_weight(
                    x,
                    weight.shape,
                    grad_output,
                    stride,
                    padding,
                    output_padding,
                    dilation,
                    1,
                )
                if needs_w
                else None
            )
        except _triton_kernel_failures() as e:
            # Degrading is a genuine fallback here, for the reason
            # _TritonConv3dFn.backward gives: this is the node itself running
            # once, not a recompute, and nothing downstream can tell which
            # kernel produced the gradients.  Both backward directions are
            # separate compilations from the forward's, so either can raise
            # OutOfResources on a call whose forward compiled cleanly.
            first = not _triton_failed
            _triton_failed = True
            if first:
                _warn_rung_failure(
                    "Triton conv_transpose3d backward",
                    e,
                    "MIOpen kernel",
                    TRITON_ENV_VAR,
                )
            grad_x, grad_w, grad_b = _aten_backward(
                grad_output,
                x,
                weight,
                stride,
                padding,
                dilation,
                (needs_x, needs_w, has_bias and needs_b),
                has_bias,
                transposed=True,
                output_padding=output_padding,
            )
            return grad_x, grad_w, grad_b, None, None, None, None

        # d(bias) is the sum of grad_output over every axis but the channel one,
        # whatever the forward kernel was -- the same expression ATen's
        # convolution backward uses, and reached on every step here.
        grad_b = grad_output.sum(dim=(0, 2, 3, 4)) if (has_bias and needs_b) else None
        return grad_x, grad_w, grad_b, None, None, None, None


class FastConvTranspose3d(nn.ConvTranspose3d):
    """``nn.ConvTranspose3d`` with a Triton GPU kernel and MIOpen behind it.

    :class:`FastConv3d`'s peer, for the four ``k = 2, s = 2`` upsamplers in the
    decoder.  Identical state: ``weight`` of shape ``(Cin, Cout, kd, kh, kw)``
    and ``bias`` of shape ``(Cout,)``, both from ``nn.ConvTranspose3d.__init__``,
    which is not overridden -- so the constructor signature is the stock one by
    construction and state dicts are interchangeable in both directions with a
    plain ``nn.ConvTranspose3d`` model.

    **Sharding is not the difficulty it is for the ordinary convolution.**  At
    ``kernel == stride`` every output voxel reads exactly one input voxel, so
    there is no halo to exchange and no ``_Halo3d`` in this ladder; DistConv
    reaches the same conclusion by a different route (``halo = k // 2`` is 0 for
    an even kernel) and also runs on the bare local shard, at every shard count.
    :func:`_transposed_halo_plan` is where that is checked rather than assumed.

    What does *not* get weaker here: the per-module latch, which is if anything
    more load-bearing than it is for ``FastConv3d`` -- see
    :class:`_TritonConvTranspose3dFn` -- and the autocast reproduction, since
    ``conv_transpose3d`` carries the same ``lower_precision_fp`` cast policy as
    ``convolution`` (verified: an fp32 pair under ``autocast(bf16)`` produces a
    bf16 output, an fp64 pair is not narrowed).
    """

    #: Per-module "a call has been answered by the Triton rung"; see
    #: :attr:`FastConv3d._triton_ok`, which this mirrors exactly.
    _triton_ok = False

    def _triton_forward(self, local):
        """Run the Triton rung on an unwrapped tensor.

        No halo, so no exchange, so no retry-below-the-exchange split: a
        ``TritonError`` here has put nothing on the wire and :meth:`forward` can
        re-run the whole call on MIOpen.  The cast comes first for the same
        reason it does in ``FastConv3d._triton_forward`` -- it reproduces what
        the dispatcher would have done below this module.
        """
        dtype = _autocast_dtype(local)
        return _TritonConvTranspose3dFn.apply(
            _cast_operand(local, dtype),
            _cast_operand(self.weight, dtype),
            _cast_operand(self.bias, dtype),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )

    def _miopen_forward(self, input, output_size=None):
        """The semantics-defining rung: ``nn.ConvTranspose3d.forward``.

        Also the only rung that takes a ``DCTensor`` as it stands, since
        DistConv's ``__torch_dispatch__`` intercepts the ``aten::convolution``
        underneath it.
        """
        return super().forward(input, output_size)

    def forward(self, input, output_size=None):
        # ``output_size`` re-derives ``output_padding`` inside
        # ``_output_padding``, so the call the kernel would be gated on is not
        # the call that would run.  It is not used anywhere in ScaFFold (``Up``
        # calls ``self.up(x1)``), so the stock rung answers it rather than this
        # ladder growing a second way to compute a padding.
        if output_size is not None:
            return self._miopen_forward(input, output_size)

        distconv = _dctensor_ops(input)
        local_view = input._tensor if distconv is not None else input
        plan = (
            _transposed_halo_plan(
                input,
                input._parallel_strategy,
                local_view,
                self.weight,
                self.padding,
            )
            if distconv is not None
            else None
        )

        if _use_triton_transposed(
            self,
            local_view,
            input if distconv is not None else None,
            plan,
            proven=self._triton_ok,
        ):
            try:
                out = _run_local(input, distconv, self._triton_forward)
            except _triton_kernel_failures() as e:
                # Every allowlisted exception is raised while compiling or
                # sizing a launch -- before the node saved anything -- and this
                # ladder never puts a halo slab on the wire, so re-running the
                # whole call on MIOpen is safe at every shard count.  That is
                # the clause ``FastConv3d`` needs a second fallback for and this
                # one does not.
                _latch_rung_failure(e, "Triton conv_transpose3d")
                # The one call the fallback must not answer: a module already
                # proven on the rung, failing while a backward is in flight, is
                # a checkpoint recompute of a forward that ran on Triton.  The
                # two rungs save metadata-identical but structurally different
                # tensors under DistConv, so MIOpen's answer would substitute a
                # DCTensor into a slot holding a plain one and fail later,
                # inside DistConv, with a message about neither.
                if self._triton_ok and _replaying_a_forward():
                    raise
            else:
                if not self._triton_ok:
                    self._triton_ok = True
                return out

        return self._miopen_forward(input)
