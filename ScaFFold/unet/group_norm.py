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

"""GroupNorm with a Triton fast path and a ``torch.compile``d one behind it.

Three kernels, tried in order, all of them producing the same numbers:

1. **Native channels-last Triton** (:mod:`ScaFFold.unet.triton_group_norm`),
   whenever that module's ``is_supported`` accepts the input.  Production runs
   set ``PYTORCH_MIOPEN_SUGGEST_NHWC=1``, under which every convolution emits
   ``channels_last_3d`` -- and every *stock* GroupNorm (eager or Inductor) reads
   that layout through the logical NCDHW order, a strided gather, and returns a
   contiguous tensor, breaking the layout chain at all 22 call sites of the
   forward.  The Triton kernel is NDHWC in and NDHWC out and is 6.5x faster than
   the compiled kernel on that input; with the ReLU fused it takes another 38%
   off the forward.  See that module's docstring for the measurements.
2. **``torch.compile``d ``F.group_norm``**, for inputs the Triton kernel does
   not serve (contiguous NCDHW, non-5-D, unsupported dtypes) and as the landing
   place if the Triton path ever raises.  ATen's own kernel launches one
   workgroup per ``(batch, group)`` row -- 8 of them at this benchmark's
   defaults -- so on a 228-CU MI300A it runs at a small fraction of achievable
   bandwidth: measured 87 ms of a 187 ms step (47%) at scale 7, against 7% of a
   184.7 ms step once Inductor tiles the reduction across the device.
3. **Stock eager ``F.group_norm``**, which is what every rejection falls back
   to and what defines the semantics the other two must match.

``FastGroupNorm`` is a drop-in ``nn.GroupNorm``: same parameters, same names,
same shapes, same numerics -- only the kernel differs, so checkpoints are
interchangeable in both directions with any other GroupNorm-based build.  The
one addition is the optional fused ``activation`` (see below), which adds no
state either.

All three return the input's memory format, so the rungs are interchangeable in
everything a caller can observe (see :func:`_match_memory_format`).

Routing rejections, in the order they are tested:

* an explicit opt-out via ``SCAFFOLD_GROUPNORM_TRITON=0`` /
  ``SCAFFOLD_GROUPNORM_COMPILE=0``,
* a rung that has failed in this process, for every module that has not already
  had a call served by it (see "Latches" below),
* an active ``torch.func`` transform -- a ``vmap``/``grad``/``jvp`` layer is a
  routing miss, not a kernel failure, and the stock kernel handles it,
* non-CUDA tensors -- the CPU test suite pays neither compile latency nor the
  Triton import,
* tensor subclasses, whose ``__torch_dispatch__`` wrappers have unknown
  semantics -- except DistConv's ``DCTensor``, which is unwrapped to its local
  shard around both fast kernels (see ``FastGroupNorm``),
* for the Triton kernel, anything its ``is_supported`` rejects (a layout, dtype,
  degenerate shape or affine-parameter dtype it does not serve); for the
  compiled one, an already-compiled enclosing region (the functional call
  inlines instead).

Latches
=======
Both fast rungs are optimizations, never correctness requirements: a broken
Triton install must degrade a multi-node run, not kill it.  So a *kernel*
failure is caught, logged once and retried on the next rung down.

"A kernel failure" is an allowlist, not the absence of one.  The Triton rung is
caught on ``triton_group_norm.TritonKernelError``, which that module raises for
anything its launch region produces; the compiled rung on
``torch._dynamo.exc.TorchDynamoException``, the root of every Dynamo and
Inductor compile failure.  Everything else propagates -- saved-tensor pack
hooks, ``torch.utils.checkpoint``'s recompute control flow, a user's offloading
hook, ``torch.OutOfMemoryError``, an error from a shape the kernel mishandles
badly enough to corrupt the graph.  The previous shape of this code caught
``Exception`` and re-raised a denylist of framework mechanisms, which was wrong
twice (``_StopRecomputationError``, then ``CheckpointError``): the set of things
torch may raise through a forward is open, the set of ways a kernel can be
broken is closed at its own boundary.  Both allowlisted exceptions are also
raised strictly *before* their rung saves anything for backward (the Triton op
saves in ``_setup_context``, after its launch region; a Dynamo/Inductor failure
is a compile-time failure, before any execution), so the retry cannot double-fire
saved-tensor hooks.

A failure latches the rung off **for modules that have never had a call served
by it**.  A module that has already run on a rung keeps it.  That is not a
performance nicety: ``torch.utils.checkpoint``'s non-reentrant recompute
compares the metadata of every tensor the recomputed forward saves against the
originals, and the three rungs intrinsically save *different tensors* -- Triton
saves ``(input, weight, bias, mean, rstd)``, the other two
``(input, weight, mean, rstd, relu_output)``.  A latch that flipped between a
block's forward and its recompute would therefore kill the step with
``CheckpointError: Recomputed values ... have different metadata``, which is the
exact opposite of the contract above (measured; matching the output memory
format is *not* sufficient on its own).  Keeping a proven rung pins each
module's choice for the life of the process, so forward and recompute always
agree.

Note that a latch is process-local: under DDP one rank can end up running a
different kernel from its peers.  All three kernels agree to fp32 rounding, not
bitwise, so a rank that latches shifts that rank's gradients and therefore the
all-reduced ones -- a real (measured) change to the job's trajectory, and a
2.1x straggler besides.  That is the price of degrading instead of dying, but it
is why the latch is as narrow as it is, and why ``torch.OutOfMemoryError`` --
transient by nature, and no cheaper on any other rung -- does not latch anything
at all.  :func:`set_triton_enabled` / :func:`set_compile_enabled` with ``True``
clear the latch, which is the supported way to retry after a transient failure.

Determinism: all three kernels are bitwise reproducible.  Two separate processes
running the scale-7 UNet under ``more_determinism``
(``use_deterministic_algorithms(True, warn_only=True)``, ``cudnn.benchmark=False``,
fixed seeds) hash identically with the Triton path, the compiled path and the
eager one alike, so no determinism gate is needed.  The Triton kernel's grid,
split count and tile sizes are pure functions of the shape and it uses no float
atomics, which is what buys that.

Fused activation
================
Every GroupNorm in the UNet is immediately followed by a ReLU, and the Triton
kernel can fold that into its forward store for free (it is store-bound) while
removing a whole streaming pass -- 38% of the forward at the shapes that
dominate.  ``FastGroupNorm(..., activation="relu")`` therefore *always* applies
the ReLU: fused inside the Triton kernel where that path is taken, and as an
explicit in-place ``F.relu`` on the compiled and eager paths.  ``DoubleConv``
consequently holds an ``nn.Identity`` where its ``nn.ReLU`` used to be, so the
positional keys of its ``nn.Sequential`` -- and therefore every checkpoint --
are unchanged (neither module has parameters or buffers).  Correctness holds on
every path; the fusion is purely an optimization inside the module.
"""

import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#: Opt-out (``0``/``false``/``off``/``no``) or explicit opt-in (``1``/``true``/
#: ``on``/``yes``) for the compiled GroupNorm path.  Unset means "on wherever it
#: is safe", which is what every production run wants.
COMPILE_ENV_VAR = "SCAFFOLD_GROUPNORM_COMPILE"

#: The same, for the native channels-last Triton kernel, which is tried first.
#: Same spellings, same "unset means on wherever it is safe" default -- the
#: whole point of the kernel is that production takes it.
TRITON_ENV_VAR = "SCAFFOLD_GROUPNORM_TRITON"

#: Activations this module can apply after normalizing.  Must stay a subset of
#: ``triton_group_norm.SUPPORTED_ACTIVATIONS`` (pinned by a test); spelled out
#: here rather than imported so that constructing a module -- or running the
#: whole CPU suite -- never imports the kernel module.
SUPPORTED_ACTIVATIONS = (None, "relu")

#: Dynamo caches one entry per distinct guard set on the traced function.  A
#: UNet presents one entry per distinct activation shape (5 at scale 7) times
#: grad-enabled/no-grad (training vs. evaluation), i.e. 10 -- above the stock
#: limit of 8, which would silently drop the whole model back to eager mid-run.
#: The traced function is a single ``F.group_norm`` call, so the extra entries
#: cost only their one-time compilation.
_MIN_RECOMPILE_LIMIT = 64

# Lazily built on the first eligible forward: importing ScaFFold must not drag
# in Dynamo, and a run that never reaches the GPU must not pay for it.
_compiled_group_norm = None

# Set once if torch.compile raises; the eager path is then used everywhere.
_compile_failed = False

# None = decide per tensor; True/False = forced by SCAFFOLD_GROUPNORM_COMPILE or
# by set_compile_enabled().
_compile_override = None

# The triton_group_norm module, imported on the first CUDA forward.  Importing
# it registers two dispatcher ops, and a CPU-only run must pay neither that nor
# the `triton` import the module itself defers to its first launch.
_triton_module = None

# The ladder's two allowlists, resolved on first use of the rung they guard --
# importing either provider (the kernel module, torch._dynamo) is exactly what
# the lazy _get_* helpers exist to avoid paying for on a CPU-only run.
_TRITON_KERNEL_FAILURES = None
_COMPILED_KERNEL_FAILURES = None

# Set once if the Triton kernel raises; the compiled path is used from then on.
_triton_failed = False

# None = decide per tensor; True/False = forced by SCAFFOLD_GROUPNORM_TRITON or
# by set_triton_enabled().
_triton_override = None


def _env_override(name):
    """Read boolean env var ``name``; ``None`` when unset or unparsable."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in ("1", "true", "on", "yes"):
        return True
    if value in ("0", "false", "off", "no"):
        return False
    logger.warning(
        f"Ignoring unrecognized {name}={raw!r}; "
        "expected one of 1/0/true/false/on/off/yes/no"
    )
    return None


_compile_override = _env_override(COMPILE_ENV_VAR)
_triton_override = _env_override(TRITON_ENV_VAR)


def set_compile_enabled(enabled):
    """Force the compiled path on (``True``) or off (``False``).

    ``None`` restores the default, which is the environment variable if set and
    otherwise "compile wherever it is safe".  Forcing it on does not override
    the device and tensor-subclass checks -- those are correctness conditions,
    not preferences -- but it *does* clear a failure latch: an explicit "use
    this rung" is the supported way to retry after a transient failure, and
    leaving the latch set would make this function silently do nothing.
    Returns the previous setting so callers (tests) can restore it.
    """
    global _compile_override, _compile_failed
    previous = _compile_override
    _compile_override = (
        _env_override(COMPILE_ENV_VAR) if enabled is None else bool(enabled)
    )
    if _compile_override is True:
        _compile_failed = False
    return previous


def set_triton_enabled(enabled):
    """Force the Triton path on (``True``) or off (``False``).

    The exact counterpart of :func:`set_compile_enabled`: ``None`` restores the
    default (``SCAFFOLD_GROUPNORM_TRITON`` if set, otherwise "wherever
    ``is_supported`` accepts"), forcing it on clears any failure latch but not
    the device, subclass or ``is_supported`` checks -- those are correctness
    conditions -- and the previous setting is returned so tests can restore it.

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


def _group_norm(input, num_groups, weight, bias, eps):
    """The function Dynamo traces: plain functional GroupNorm, nothing else."""
    return F.group_norm(input, num_groups, weight, bias, eps)


def _raise_recompile_limit():
    """Lift Dynamo's per-function recompile cap to cover every UNet GN shape.

    Only ever raises it, so a caller that deliberately set a larger limit keeps
    theirs -- but note the converse: a limit deliberately set *smaller* than
    ours is clobbered up to ``_MIN_RECOMPILE_LIMIT``. ``cache_size_limit`` is
    the older spelling of ``recompile_limit``; set whichever exists.
    """
    config = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):
        current = getattr(config, name, None)
        if isinstance(current, int) and current < _MIN_RECOMPILE_LIMIT:
            setattr(config, name, _MIN_RECOMPILE_LIMIT)


def _get_compiled_group_norm():
    """Build (once) the compiled functional GroupNorm shared by every module.

    One compiled callable for the whole model, not one per module: the shapes,
    not the instances, are what Dynamo specializes on, and sharing keeps the
    18 GroupNorms of a scale-7 UNet down to 5 compilations.  ``dynamic=False``
    keeps the specialized kernels (this benchmark runs fixed shapes);
    ``fullgraph=True`` turns anything Dynamo cannot handle into an exception we
    catch, rather than a silent graph break that reintroduces the slow kernel.
    """
    global _compiled_group_norm
    if _compiled_group_norm is None:
        _raise_recompile_limit()
        _compiled_group_norm = torch.compile(_group_norm, dynamic=False, fullgraph=True)
    return _compiled_group_norm


def _get_triton_module():
    """Import (once) :mod:`ScaFFold.unet.triton_group_norm`.

    Deferred rather than imported at the top of this file: that module registers
    two dispatcher ops and builds an autograd formula at import time, and a run
    that never reaches the GPU (the whole CPU unit suite) must not pay for it.
    Only ever called after the input has been shown to be a CUDA tensor, which
    is also what keeps ``import triton`` -- which that module defers again, to
    its first kernel launch -- out of a CPU-only process entirely.
    """
    global _triton_module
    if _triton_module is None:
        from . import triton_group_norm

        _triton_module = triton_group_norm
    return _triton_module


def _triton_kernel_failures():
    """The ladder's allowlist for the Triton rung: exactly ``TritonKernelError``.

    The kernel module raises it for every failure of its own launch region --
    a missing or mismatched ``triton``, an unwritable JIT cache, a compile
    error, a bad launch -- and for nothing else, so this catches "the kernel is
    broken" without also catching the framework mechanisms that legitimately
    raise through a forward.  See that class's docstring for what is
    deliberately left untagged (``OutOfMemoryError``, contract violations).

    Resolved separately from :func:`_get_triton_module` so the except clause is
    still available when the thing that failed *is* the module lookup.  An
    empty tuple (no kernel module at all) means "catch nothing": the ladder
    then re-raises, which is right, because with no kernel module there is
    nothing that could have failed inside one.
    """
    global _TRITON_KERNEL_FAILURES
    if _TRITON_KERNEL_FAILURES is None:
        try:
            from .triton_group_norm import TritonKernelError

            _TRITON_KERNEL_FAILURES = (TritonKernelError,)
        except ImportError:  # pragma: no cover - the module is in-tree
            _TRITON_KERNEL_FAILURES = ()
    return _TRITON_KERNEL_FAILURES


def _compiled_kernel_failures():
    """The compiled rung's allowlist: every Dynamo and Inductor compile failure.

    ``torch._dynamo.exc.TorchDynamoException`` is the root of ``Unsupported``
    (``fullgraph=True`` met something untraceable), ``BackendCompilerFailed``
    and its ``InductorError`` subclass (the backend, and therefore also an
    unwritable Inductor cache or a broken C++/Triton toolchain), and
    ``InternalTorchDynamoError``.  All of them are raised while *compiling*,
    i.e. before the compiled callable has executed or saved anything, which is
    what makes the fallback safe to retry.

    Resolved on demand and cached: importing ``torch._dynamo`` is precisely the
    cost :func:`_get_compiled_group_norm` defers.  An empty tuple (a torch
    without the module) means "catch nothing", which fails loudly rather than
    silently swallowing.
    """
    global _COMPILED_KERNEL_FAILURES
    if _COMPILED_KERNEL_FAILURES is None:
        try:
            import torch._dynamo.exc

            _COMPILED_KERNEL_FAILURES = (torch._dynamo.exc.TorchDynamoException,)
        except ImportError:  # pragma: no cover - torch always ships it
            _COMPILED_KERNEL_FAILURES = ()
    return _COMPILED_KERNEL_FAILURES


#: ``True`` while a ``torch.func`` transform (``vmap``/``grad``/``jvp``) is on
#: the stack.  Both fast rungs decline then: a functorch layer is a routing
#: question, not a kernel defect, and the stock kernel handles every transform.
#: Not merely a performance choice -- ``is_supported``'s
#: ``is_contiguous(memory_format=...)`` raises outright under ``vmap``
#: ("NYI: querying is_contiguous inside of vmap"), and the Triton op has no
#: batching rule -- so without this the module is not the drop-in
#: ``nn.GroupNorm`` it claims to be for any caller using ``torch.func``.
_functorch_active = getattr(torch._C, "_are_functorch_transforms_active", lambda: False)

# Set once if a predicate raised while deciding; see _use_triton.
_predicate_warned = False


def _use_triton(input, num_groups, weight, bias, activation, proven=False):
    """Whether this particular input should take the native Triton kernel.

    ``proven`` is the caller's "this module has already had a call served by
    this rung", which keeps a proven module on it even after a *global* latch;
    see the module docstring's "Latches".

    Ordered so that the cheap local tests come first and the module import last:
    a CPU tensor is rejected before ``_get_triton_module`` is ever called.
    """
    if _triton_override is False:
        return False
    if _triton_failed and not proven:
        return False
    if _functorch_active():
        return False
    # Same policy as _use_compiled: an unknown __torch_dispatch__ wrapper has
    # unknown semantics and keeps the stock kernel.  is_supported() would accept
    # one (it only asks isinstance), so this check is load-bearing here, not a
    # copy for symmetry.  DistConv's DCTensor never reaches it -- forward()
    # unwraps to the local shard first.
    if type(input) is not torch.Tensor:
        return False
    if not input.is_cuda:
        return False
    # is_supported() is cheap and side-effect free: a handful of attribute reads
    # and one stride check, no allocation, no launch, no triton import.  The
    # broad catch is right *here* and nowhere else in this module: a predicate
    # that cannot answer has a correct answer available ("no"), it has done no
    # work anyone can observe, and the failure is a routing miss rather than a
    # broken kernel -- so it must not latch the rung off, which is what letting
    # it fall into the ladder's handler used to do.
    try:
        return _get_triton_module().is_supported(
            input, num_groups, weight, bias, activation
        )
    except Exception as e:
        _warn_once_about_the_predicate(e)
        return False


def _warn_rung_failure(what, error, fallback, env_var):
    """Log a rung failure, without graph-breaking a compiled caller.

    Dynamo cannot trace ``logging.Logger`` methods ("Unsupported: logging.Logger
    method not supported for non-export cases"), so a bare ``logger.warning``
    in the ladder's handler turns a *fallback* into a hard Dynamo error for
    anyone who wraps this ``forward`` in ``torch.compile(fullgraph=True)`` --
    the one caller for whom the fallback matters most, since the failure it is
    reacting to is usually a compile failure.  ``is_compiling()`` is a Dynamo
    intrinsic that folds to ``True`` at trace time, so the call below becomes
    dead code inside a traced region and the fallback traces cleanly.  The
    latch itself is a global assignment, which Dynamo does replay, so the
    fallback is still recorded -- only this message is dropped, and only for a
    caller that is compiling this module's forward (nothing in ScaFFold does).
    """
    if torch.compiler.is_compiling():
        return
    logger.warning(
        f"{what} failed ({type(error).__name__}: {error}); falling back to the "
        f"{fallback} for modules that have not already used it. "
        f"Set {env_var}=0 to skip this attempt entirely."
    )


def _warn_once_about_the_predicate(error):
    """Log the first ``is_supported`` failure; a repeat would log per call."""
    global _predicate_warned
    if _predicate_warned:
        return
    _predicate_warned = True
    logger.warning(
        f"Triton GroupNorm routing check failed ({type(error).__name__}: "
        f"{error}); using the stock kernel for inputs like this one. This is a "
        "routing miss, not a kernel failure, so nothing is latched off."
    )


def _use_compiled(input, proven=False):
    """Whether this particular input should take the compiled path.

    ``proven`` has the same meaning as in :func:`_use_triton`.
    """
    if _compile_override is False:
        return False
    if _compile_failed and not proven:
        return False
    # Dynamo cannot trace a functorch layer either, and under fullgraph=True
    # that is an exception rather than a graph break.
    if _functorch_active():
        return False
    # Tensor subclasses route their ops through __torch_dispatch__, which
    # Dynamo cannot trace.  DistConv's DCTensor never reaches this check --
    # forward() peeks at its local shard instead -- so anything rejected here
    # is an unknown wrapper, and eager keeps its semantics exactly as they
    # are today.
    if type(input) is not torch.Tensor:
        return False
    # CPU GroupNorm is not the bottleneck and compiling it would put a
    # multi-second C++ build in front of every unit test.
    if not input.is_cuda:
        return False
    # Already inside a compiled region: let the functional call be inlined.
    if torch.compiler.is_compiling():
        return False
    return True


def _dctensor_ops(input):
    """The ``distconv.distconv`` module when ``input`` is a DCTensor, else None.

    Resolved through ``sys.modules`` instead of an import: a DCTensor can only
    exist if DistConv is already imported, and this module must stay importable
    (and the CPU suite runnable) without DistConv installed.
    """
    distconv = sys.modules.get("distconv.distconv")
    if distconv is not None and isinstance(input, distconv.DCTensor):
        return distconv
    return None


def _match_memory_format(out, reference):
    """Give ``out`` ``reference``'s memory format, copying only if it differs.

    ``F.group_norm`` -- eager or Inductor-compiled -- reads a
    ``channels_last_3d`` input through the logical NCDHW order and returns a
    *contiguous* tensor, which is the layout break this whole module exists to
    avoid: with ``PYTORCH_MIOPEN_SUGGEST_NHWC=1`` every convolution both sides
    of it wants channels-last, so one fallback re-breaks the chain for the rest
    of the network.  One relayout of the GroupNorm output is far cheaper than
    the transposes the following convolutions would otherwise insert, and it
    makes the three rungs agree on everything a caller can observe rather than
    only on the values.

    Free on the Triton rung (already channels-last) and on any contiguous input
    (nothing to do); one copy on a fallback from a channels-last input, which is
    the only case that reaches the copy at all.
    """
    if reference.dim() != 5:
        # is_contiguous(memory_format=channels_last_3d) is only defined for 5-D.
        return out
    if _functorch_active():
        # "NYI: querying is_contiguous inside of vmap for memory_format other
        # than torch.contiguous_format" -- and a functorch transform has no
        # layout chain to preserve anyway, since both fast rungs decline it.
        return out
    if not reference.is_contiguous(memory_format=torch.channels_last_3d):
        return out
    if out.is_contiguous(memory_format=torch.channels_last_3d):
        return out
    return out.contiguous(memory_format=torch.channels_last_3d)


def _run_local(input, distconv, kernel):
    """Run ``kernel`` on a plain tensor, DCTensor in -> DCTensor out.

    ``distconv`` is ``None`` for a plain tensor, where this is just
    ``kernel(input)``.  For a ``DCTensor`` the unwrap goes through DistConv's
    ``_ToTensor``/``_FromTensor`` autograd pair (``DCTensor.from_shard`` is the
    public spelling of the latter; there is no public unwrap yet -- upstream
    ask) rather than a bare ``input._tensor`` read: DistConv's own dispatch may
    read ``_tensor`` directly because it runs *below* autograd, while this runs
    above it and a bare read would sever the graph back to the producing
    convolution.
    """
    if distconv is None:
        return kernel(input)
    local = distconv._ToTensor.apply(input)
    return distconv.DCTensor.from_shard(kernel(local), input._parallel_strategy)


class FastGroupNorm(nn.GroupNorm):
    """``nn.GroupNorm`` with a Triton GPU kernel and an optional fused ReLU.

    Identical state: ``weight``/``bias`` of shape ``(num_channels,)``, no
    buffers, so state dicts are interchangeable with plain ``nn.GroupNorm``
    in both directions.  ``activation`` is a plain Python attribute, not a
    submodule or a buffer, so setting it does not add a key either.

    ``activation="relu"`` makes this module's forward *always* apply a ReLU --
    fused into the Triton kernel's store where that path is taken, and as an
    explicit in-place ``F.relu`` on the compiled and eager paths.  The
    correctness of the model therefore does not depend on which kernel runs;
    only the number of memory passes does.

    DistConv's ``DCTensor`` gets the fast kernels too, by unwrapping to the
    local shard in front of them rather than by letting the op dispatch through
    the wrapper.  Both would work -- the Triton kernel is a real dispatcher op,
    so ``DCTensor.__torch_dispatch__`` would intercept it, unwrap, run and
    rewrap on its own -- but the explicit unwrap is what this module already
    does for the compiled kernel, and it is better here for three reasons.
    (1) It keeps the subclass policy in one place: ``is_supported`` accepts any
    ``torch.Tensor`` *instance*, so relying on dispatch would silently extend
    the fast path to every unknown wrapper subclass, which today keeps the
    stock kernel.  (2) The eligibility predicates then examine the tensor the
    kernel will actually touch -- its dtype, device, strides and shape -- rather
    than a wrapper's mirrored metadata.  (3) The Triton and compiled paths share
    one unwrap and one fallback ladder instead of needing two shapes of code,
    and a Triton failure can be retried on the compiled kernel without a second
    round trip through the wrapper.  Semantics are unchanged either way:
    DistConv's generic ``__torch_dispatch__`` has no GroupNorm-specific
    handling, so statistics are per-shard and no communication happens at any
    shard count, exactly as before.
    """

    #: Class-level defaults, so that an instance restored from a *module*
    #: pickle written before these attributes existed (``torch.save(model)``
    #: rather than a state dict) still runs.  ``nn.Module.__setstate__``
    #: replaces ``__dict__`` wholesale, so anything only ever set in
    #: ``__init__`` is simply missing on such an instance.
    activation = None

    #: Per-module "a call has been served by this rung".  A global latch does
    #: not demote a module that has one, which is what keeps a checkpointed
    #: block's forward and its recompute on the same rung; see the module
    #: docstring's "Latches".  Plain attributes, so they are not parameters,
    #: buffers or state-dict keys.
    _triton_ok = False
    _compiled_ok = False

    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
        activation=None,
    ):
        if activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {SUPPORTED_ACTIVATIONS}, got {activation!r}"
            )
        super().__init__(
            num_groups, num_channels, eps=eps, affine=affine, device=device, dtype=dtype
        )
        self.activation = activation

    def extra_repr(self):
        base = super().extra_repr()
        if self.activation is None:
            return base
        return f"{base}, activation={self.activation}"

    def _activate(self, out):
        """Apply the activation on the two paths that cannot fuse it.

        In place, which is what the ``nn.ReLU(inplace=True)`` this module
        absorbed did: ``out`` is a freshly allocated GroupNorm output with no
        other consumer, and GroupNorm's backward reads its *input*, never its
        output, so overwriting it is safe for autograd as well as for memory.

        Validated *here* rather than only in ``__init__``: ``activation`` is a
        plain attribute, so it can be assigned after construction, and the
        Triton rung would then fuse an activation this method silently skipped
        -- i.e. the network's function would depend on its input's memory
        format.  This is also the guard that makes adding a third activation to
        ``SUPPORTED_ACTIVATIONS`` a loud failure until it is implemented here.
        """
        activation = self.activation
        if activation is None:
            return out
        if activation == "relu":
            return F.relu(out, inplace=True)
        raise ValueError(
            f"activation must be one of {SUPPORTED_ACTIVATIONS}, got "
            f"{activation!r}; this rung cannot apply it"
        )

    def _triton_forward(self, local):
        """The native channels-last kernel, with the activation fused in."""
        return _get_triton_module().triton_group_norm(
            local, self.num_groups, self.weight, self.bias, self.eps, self.activation
        )

    def _compiled_forward(self, local):
        return self._activate(
            _match_memory_format(
                _get_compiled_group_norm()(
                    local, self.num_groups, self.weight, self.bias, self.eps
                ),
                local,
            )
        )

    def _eager_forward(self, input):
        # super().forward() is the stock kernel; deferring to it keeps the eager
        # path identical to nn.GroupNorm's (plus the ReLU and the relayout) by
        # construction.
        return self._activate(_match_memory_format(super().forward(input), input))

    def forward(self, input):
        global _compile_failed, _triton_failed

        distconv = _dctensor_ops(input)
        # The eligibility checks look at the local shard for a DCTensor (the
        # peek is a plain attribute read, no autograd involvement) and at the
        # tensor itself otherwise.
        local_view = input._tensor if distconv is not None else input

        if _use_triton(
            local_view,
            self.num_groups,
            self.weight,
            self.bias,
            self.activation,
            proven=self._triton_ok,
        ):
            triton_failures = _triton_kernel_failures()
            try:
                out = _run_local(input, distconv, self._triton_forward)
            except triton_failures as e:
                # A broken or mismatched Triton install, an unwritable JIT cache
                # or a shape the kernel mishandles must cost speed, not a
                # multi-node run.  GroupNorm is pure and the kernel raises this
                # only from its launch region -- before it has saved anything --
                # so retrying the same call on the compiled kernel below is
                # safe, and the compiled kernel, not eager, is the right landing
                # place: it is still ~10x the stock one.
                #
                # Logged on the latch's False->True edge only.  A module that
                # has already used the rung keeps trying it (that is what pins
                # a checkpointed block to one rung), so a persistently broken
                # kernel would otherwise warn once per call for the rest of the
                # run; clearing the latch re-arms the message.
                first = not _triton_failed
                _triton_failed = True
                if first:
                    _warn_rung_failure(
                        "Triton GroupNorm", e, "compiled kernel", TRITON_ENV_VAR
                    )
            else:
                # Only written once: nn.Module.__setattr__ is not free, and
                # after the first success this reads a class attribute.
                if not self._triton_ok:
                    self._triton_ok = True
                return out

        if not _use_compiled(local_view, proven=self._compiled_ok):
            return self._eager_forward(input)
        compile_failures = _compiled_kernel_failures()
        try:
            out = _run_local(input, distconv, self._compiled_forward)
        except compile_failures as e:
            # Compilation is an optimization, never a correctness requirement:
            # a broken Inductor install, an unwritable cache directory or an
            # untraceable input must degrade to the stock kernel, not kill a
            # multi-node run.  Every exception caught here is a *compile*-time
            # one, so nothing ran and retrying eagerly is safe.  Same
            # once-per-latch-edge logging as the Triton rung above.
            first = not _compile_failed
            _compile_failed = True
            if first:
                _warn_rung_failure(
                    "torch.compile of GroupNorm", e, "eager kernel", COMPILE_ENV_VAR
                )
            return self._eager_forward(input)
        else:
            if not self._compiled_ok:
                self._compiled_ok = True
            return out
