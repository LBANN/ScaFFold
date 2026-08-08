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

Three kernels, tried in order.  They agree to fp32 rounding and all return the
input's memory format and the input's dtype (see :func:`_match_memory_format`
and :func:`_match_input_dtype`), so a caller does not have to know which rung
served it:

1. Native channels-last Triton (:mod:`ScaFFold.unet.triton_group_norm`),
   whenever that module's ``is_supported`` accepts the input.  Production runs
   set ``PYTORCH_MIOPEN_SUGGEST_NHWC=1``, under which every convolution emits
   ``channels_last_3d``, and every stock GroupNorm (eager or Inductor) reads
   that layout through the logical NCDHW order as a strided gather and returns a
   contiguous tensor, breaking the layout chain for the convolutions that
   follow.  The Triton kernel is NDHWC in and NDHWC out, and can fold the ReLU
   into its store (see "Fused activation").
2. ``torch.compile``d ``F.group_norm``, for inputs the Triton kernel does not
   serve (contiguous NCDHW, non-5-D, unsupported dtypes) and as the landing
   place if the Triton path ever raises.  ATen's own kernel launches one
   workgroup per ``(batch, group)`` row -- 8 of them at this benchmark's
   defaults -- so on a 228-CU MI300A it leaves most of the device idle, where
   Inductor tiles the reduction across it.
3. Stock eager ``F.group_norm``, which every rejection falls back to and which
   defines the semantics the other two must match.

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
* for the Triton kernel, a GPU other than the one its launch tables were tuned
  on (``gfx942`` with 228 CUs -- an MI300A).  Alone on this list that one is a
  preference: the kernel is correct anywhere Triton lowers it and only its speed
  is unknown elsewhere, which is why an explicit opt-in overrides it and nothing
  else here can be overridden at all.  See ``_rungs._platform_declines``,
* for the Triton kernel, anything its ``is_supported`` rejects (a layout, dtype,
  degenerate shape or affine-parameter dtype it does not serve); for the
  compiled one, an already-compiled enclosing region (the functional call
  inlines instead).

Latches
=======
Both fast rungs are optimizations, never correctness requirements: a broken
Triton install must degrade a multi-node run, not kill it.  So a *kernel*
failure is caught, logged once and retried on the next rung down.

"A kernel failure" is an allowlist: ``triton_group_norm.TritonKernelError`` for
the Triton rung, ``torch._dynamo.exc.TorchDynamoException`` plus
``FailOnRecompileLimitHit`` for the compiled one (see
:func:`_compiled_kernel_failures`).  Everything else propagates -- saved-tensor
pack hooks, ``torch.utils.checkpoint``'s recompute control flow, a user's
offloading hook, ``torch.OutOfMemoryError``.  A denylist cannot work here: the
set of things torch may raise through a forward is open, while the set of ways a
kernel can be broken is closed at its own boundary.  Both allowlisted exceptions
are raised strictly before their rung saves anything for backward (the Triton op
saves in ``_setup_context``, after its launch region; a Dynamo/Inductor failure
happens at compile time, before any execution), so the retry cannot double-fire
saved-tensor hooks.

A failure latches the rung off only for modules that have never had a call
served by it; a module that has already run on a rung keeps it.  That is a
correctness requirement, not a performance nicety:
``torch.utils.checkpoint``'s non-reentrant recompute compares the metadata of
every tensor the recomputed forward saves against the originals, and the rungs
save *different tensors* -- Triton ``(input, weight, bias, mean, rstd)``, the
other two ``(input, weight, mean, rstd, relu_output)``.  A latch that flipped
between a block's forward and its recompute would kill the step with
``CheckpointError: Recomputed values ... have different metadata``.  Pinning
each module's choice for the life of the process keeps forward and recompute in
agreement.

The same reasoning bounds the fallback, which the latch alone does not: a proven
module still has to answer the call its rung just failed, and answering it
eagerly is exactly the flip forbidden above if that call is a checkpoint
recompute.  So the fallback is declined in the one case where it would corrupt
rather than degrade: a module proven on the rung, failing while an autograd
graph task is in flight (:func:`_replaying_a_forward`), re-raises.  Every other
failure -- in particular every first one, which is what a broken Triton install,
an unwritable Inductor cache or a missing compiler produce -- still degrades,
which is where the "must not kill a multi-node run" contract lives.

A latch is process-local, so under DDP one rank can end up on a different kernel
from its peers.  The kernels agree to fp32 rounding, not bitwise, so a rank that
latches shifts its own gradients and therefore the all-reduced ones, changing
the job's trajectory, and it becomes a straggler besides.  That is the
price of degrading instead of dying, and it is why the latch is as narrow as it
is, and why ``torch.OutOfMemoryError`` -- transient, and no cheaper on any other
rung -- latches nothing at all.  :func:`set_triton_enabled` /
:func:`set_compile_enabled` with ``True`` clear the latch, which is the
supported way to retry after a transient failure.

Determinism: all three kernels are bitwise reproducible, so no determinism gate
is needed.  The Triton kernel's grid, split count and tile sizes are pure
functions of the shape and it uses no float atomics, which is what buys that.

Fused activation
================
Every GroupNorm in the UNet is immediately followed by a ReLU, and the Triton
kernel folds it into its forward store -- free, since that store is what bounds
the kernel -- removing a whole streaming pass.
``FastGroupNorm(..., activation="relu")`` therefore always applies the ReLU:
fused inside the Triton kernel where that path is taken, and as an explicit
in-place ``F.relu`` on the compiled and eager paths.  ``DoubleConv``
consequently holds an ``nn.Identity`` in the activation slot of its
``nn.Sequential``, which keeps the positional keys -- and therefore every
checkpoint -- unchanged; neither module has parameters or buffers.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared with ScaFFold.unet.conv3d so the corrections these helpers carry cannot
# drift between the two ladders.  Imported by name so they stay module
# attributes here, which is what the tests monkeypatch.
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
#: ``on``/``yes``) for the compiled GroupNorm path.  Unset means "on wherever it
#: is safe", which is what every production run wants.
COMPILE_ENV_VAR = "SCAFFOLD_GROUPNORM_COMPILE"

#: The same spellings and the same default, for the native channels-last Triton
#: kernel, which is tried first.  The explicit opt-in is more than that default
#: written out: it additionally overrides the hardware guard, the one routing
#: condition here that is a preference.  See :func:`set_triton_enabled`.
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
#: ``activation_checkpointing`` on a ``DCTensor`` doubles that again: the
#: recompute reaches this module with ``__torch_function__`` subclass handling
#: disabled (DistConv's backward runs below it), which is part of Dynamo's
#: ``GLOBAL_STATE`` guard, so the recomputed forward misses every entry the
#: original forward built and compiles a second set beside it.  The traced
#: function is a single ``F.group_norm`` call, so the extra entries cost only
#: their one-time compilation.
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

# The ladder's two allowlists, resolved on first use of the rung they guard:
# importing either provider (the kernel module, torch._dynamo) is exactly what
# the lazy _get_* helpers exist to avoid on a CPU-only run.
_TRITON_KERNEL_FAILURES = None
_COMPILED_KERNEL_FAILURES = None

# Set once if the Triton kernel raises; the compiled path is used from then on.
_triton_failed = False

# None = decide per tensor; True/False = forced by SCAFFOLD_GROUPNORM_TRITON or
# by set_triton_enabled().
_triton_override = None


_compile_override = _env_override(COMPILE_ENV_VAR)
_triton_override = _env_override(TRITON_ENV_VAR)


def set_compile_enabled(enabled):
    """Force the compiled path on (``True``) or off (``False``).

    ``None`` restores the default, which is the environment variable if set and
    otherwise "compile wherever it is safe".  Forcing it on does not override
    the device and tensor-subclass checks -- those are correctness conditions,
    not preferences -- but it does clear a failure latch: an explicit "use this
    rung" is the supported way to retry after a transient failure.  Returns the
    previous setting so callers (tests) can restore it.
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
    ``is_supported`` accepts"), and the previous setting is returned so tests
    can restore it.

    Forcing it on clears any failure latch and overrides the hardware guard, and
    nothing else -- not the device, subclass or ``is_supported`` checks, which
    are correctness conditions.  The guard is on the other side of that line
    because an untuned GPU gives the right numbers at an unknown speed, where an
    unsupported input would give the wrong ones; the opt-in (here or as
    ``SCAFFOLD_GROUPNORM_TRITON=1``) is a developer asserting a judgement about
    their own hardware, and the log says so.  See
    :func:`~ScaFFold.unet._rungs._platform_declines`.

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
    """Lift Dynamo's *global* recompile cap to cover every UNet GN shape.

    Only ever raises it, so a caller that deliberately set a larger limit keeps
    theirs -- but note the converse: a limit deliberately set *smaller* than
    ours is clobbered up to ``_MIN_RECOMPILE_LIMIT``. ``cache_size_limit`` is
    the older spelling of ``recompile_limit``; set whichever exists.

    Not sufficient on its own: ``torch._dynamo.config`` keeps user overrides in
    a thread-local ``ContextVar`` (``torch/utils/_config_module.py``), so this
    assignment is invisible to every other thread, which goes on reading the
    stock default of 8.  That matters because ``torch.utils.checkpoint``'s
    non-reentrant recompute runs on the autograd engine's device worker thread,
    and on a ``DCTensor`` it has to compile (see ``_MIN_RECOMPILE_LIMIT``).
    :func:`_compile_group_norm` therefore also asks ``torch.compile`` for a
    per-region limit, which Dynamo applies on whichever thread is compiling.
    """
    config = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):
        current = getattr(config, name, None)
        if isinstance(current, int) and current < _MIN_RECOMPILE_LIMIT:
            setattr(config, name, _MIN_RECOMPILE_LIMIT)


def _compile_group_norm():
    """``torch.compile`` :func:`_group_norm` with a thread-proof recompile cap.

    ``recompile_limit=`` is the per-region spelling of the cap: Dynamo applies
    it around the compile itself, on whatever thread that compile happens on,
    which is the only spelling that survives the autograd worker thread (see
    :func:`_raise_recompile_limit`).  Older torches have no such keyword; there
    the global assignment is all there is, and the checkpoint-recompute case is
    out of reach.
    """
    try:
        return torch.compile(
            _group_norm,
            dynamic=False,
            fullgraph=True,
            recompile_limit=_MIN_RECOMPILE_LIMIT,
        )
    except TypeError:
        # A torch too old for the keyword: still compile, since the rung is
        # worth more than the one configuration the keyword rescues.
        return torch.compile(_group_norm, dynamic=False, fullgraph=True)


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
        _compiled_group_norm = _compile_group_norm()
    return _compiled_group_norm


def _get_triton_module():
    """Import (once) :mod:`ScaFFold.unet.triton_group_norm`.

    Deferred rather than imported at the top of this file: that module registers
    two dispatcher ops and builds an autograd formula at import time, and a run
    that never reaches the GPU (the whole CPU unit suite) must not pay for it.
    Only ever called once the input is known to be a CUDA tensor, which also
    keeps ``import triton`` -- deferred again to the first kernel launch -- out
    of a CPU-only process.
    """
    global _triton_module
    if _triton_module is None:
        from . import triton_group_norm

        _triton_module = triton_group_norm
    return _triton_module


def _triton_kernel_failures():
    """The ladder's allowlist for the Triton rung: exactly ``TritonKernelError``.

    The kernel module raises it for every failure of its own launch region and
    for nothing else, so this catches "the kernel is broken" without also
    catching the framework mechanisms that legitimately raise through a forward.
    See that class's docstring for what is deliberately left untagged
    (``OutOfMemoryError``, contract violations).

    Resolved separately from :func:`_get_triton_module` so the except clause is
    available even when the thing that failed *is* the module lookup.  An empty
    tuple (no kernel module at all) means "catch nothing", which is right: with
    no kernel module, nothing could have failed inside one.
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
    and its ``InductorError`` subclass (an unwritable Inductor cache, a broken
    C++/Triton toolchain), and ``InternalTorchDynamoError``.

    ``FailOnRecompileLimitHit`` -- raised when a frame needs more cache entries
    than the recompile limit allows, a hard error under ``fullgraph=True``
    rather than a drop to eager -- derives straight from ``Exception``
    (``torch/_dynamo/exc.py``) and not from that root, so catching only
    ``TorchDynamoException`` would let it kill the run.  It is added only when
    it really is outside the root, so a torch that reparents it does not produce
    a duplicate entry.

    All of these are raised while *compiling*, before the compiled callable has
    executed or saved anything, which is what makes the fallback safe to retry.

    Resolved on demand and cached: importing ``torch._dynamo`` is precisely the
    cost :func:`_get_compiled_group_norm` defers.  An empty tuple (a torch
    without the module) means "catch nothing", which fails loudly rather than
    swallowing silently.
    """
    global _COMPILED_KERNEL_FAILURES
    if _COMPILED_KERNEL_FAILURES is None:
        try:
            import torch._dynamo.exc as dynamo_exc
        except ImportError:  # pragma: no cover - torch always ships it
            _COMPILED_KERNEL_FAILURES = ()
        else:
            failures = [dynamo_exc.TorchDynamoException]
            limit_hit = getattr(dynamo_exc, "FailOnRecompileLimitHit", None)
            if isinstance(limit_hit, type) and not issubclass(
                limit_hit, dynamo_exc.TorchDynamoException
            ):
                failures.append(limit_hit)
            _COMPILED_KERNEL_FAILURES = tuple(failures)
    return _COMPILED_KERNEL_FAILURES


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
    # ...and the GPU the kernel's launch tables were tuned on: ``_TUNED``'s
    # largest entry names a grid of exactly 228 CUs, so elsewhere its answers
    # describe a different machine, and nothing downstream would notice, because
    # a mistuned launch is a *correct* answer at an unmeasured speed.  Shared
    # with the convolution ladder -- see ``_rungs._platform_declines``,
    # including why an explicit opt-in overrides this check and not the ones
    # around it.  Cached per device.
    if _platform_declines(input.device, _triton_override):
        return False
    # is_supported() is cheap and side-effect free: attribute reads and one
    # stride check, no allocation, no launch, no triton import.  The broad catch
    # is right *here* and nowhere else in this module: a predicate that cannot
    # answer has a correct answer available ("no") and has done no observable
    # work, and its failure is a routing miss rather than a broken kernel, so it
    # must not latch the rung off.
    try:
        return _get_triton_module().is_supported(
            input, num_groups, weight, bias, activation
        )
    except Exception as e:
        _warn_once_about_the_predicate(e)
        return False


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
    # Tensor subclasses route their ops through __torch_dispatch__, which Dynamo
    # cannot trace.  DistConv's DCTensor never reaches this check -- forward()
    # unwraps to its local shard first -- so anything rejected here is an
    # unknown wrapper, whose semantics eager preserves.
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


def _match_memory_format(out, reference):
    """Give ``out`` ``reference``'s memory format, copying only if it differs.

    ``F.group_norm`` -- eager or Inductor-compiled -- reads a
    ``channels_last_3d`` input through the logical NCDHW order and returns a
    *contiguous* tensor.  With ``PYTORCH_MIOPEN_SUGGEST_NHWC=1`` the
    convolutions on both sides want channels-last, so one fallback would
    re-break the layout chain for the rest of the network; relaying out the
    GroupNorm output once is cheaper than the transposes those convolutions
    would insert, and it is what makes the three rungs agree on everything a
    caller can observe rather than only on the values.

    Only a fallback from a channels-last input reaches the copy: the Triton rung
    is already channels-last, and a contiguous input has nothing to do.
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


def _match_input_dtype(out, reference):
    """Give ``out`` ``reference``'s dtype, casting only if it differs.

    The compiled and eager rungs call ``F.group_norm``, which carries
    autocast's **fp32** cast policy and therefore returns fp32 for a bf16 input
    inside an autocast region.  ``FastGroupNorm`` emits the input's dtype
    instead (see its docstring for why, and for the warning that goes with it),
    and the Triton rung does it in the store; these two have to do it here.

    Free outside autocast and free for an fp32 input, where ``F.group_norm``
    already returns the input's dtype -- the identity check, not the cast, is
    what runs.  Inside autocast on a narrower input it is one cast, which is
    exactly the cast the *consumer* was going to do a moment later, so nothing
    is spent that was not being spent already; what it buys is that a rung
    change cannot change the dtype of an activation.

    ``.to(dtype)`` preserves the memory format, so this composes with
    :func:`_match_memory_format` in either order; it is applied last because
    the fused-activation rung applies its activation before the store, and the
    other two must round after the ReLU for the same reason.
    """
    if out.dtype is reference.dtype:
        return out
    return out.to(reference.dtype)


class FastGroupNorm(nn.GroupNorm):
    """``nn.GroupNorm`` with a Triton GPU kernel and an optional fused ReLU.

    A drop-in replacement with identical state: ``weight``/``bias`` of shape
    ``(num_channels,)``, no buffers, so state dicts are interchangeable with
    plain ``nn.GroupNorm`` in both directions.  ``activation`` is a plain Python
    attribute, not a submodule or a buffer, so it adds no key either.  The one
    deliberate departure from ``nn.GroupNorm`` is the output *dtype* under
    autocast; see "Output dtype" below before consuming a GroupNorm output as
    fp32.

    ``activation="relu"`` makes this module's forward always apply a ReLU --
    fused into the Triton kernel's store where that path is taken, and as an
    explicit in-place ``F.relu`` on the compiled and eager paths.  Which kernel
    runs therefore changes the number of memory passes, never the function the
    model computes.

    Output dtype -- read this before consuming a GroupNorm output
    =============================================================
    **This module returns its input's dtype, which is not what
    ``nn.GroupNorm``/``F.group_norm`` returns under autocast.**  ``at::group_norm``
    carries autocast's ``fp32`` cast policy, so stock GroupNorm returns fp32
    for *any* input dtype inside an enabled autocast region; this module
    returns bf16 for a bf16 input, fp16 for an fp16 one, and fp32 for an fp32
    one.  Concretely, in the shipped configuration (``torch_amp: 1``, bf16) the
    activations a ``FastGroupNorm`` hands out are **bf16**; under
    ``torch_amp: 0``, or in ``eval``/``inference_mode`` outside an autocast
    region, they are **fp32** and nothing here differs from stock; under an
    fp16 autocast they are fp16.  "The input's dtype" is the rule; "bf16" is
    only what that rule evaluates to in production.

    The statistics are *not* narrowed: mean and variance are accumulated in
    fp32 on every rung, exactly as before, and the normalized value is computed
    in fp32.  Only the store changes, so the output is the fp32 answer rounded
    once -- not a narrower computation.  What this is buying is the round trip
    that rounding used to cost twice over: every consumer of a GroupNorm in
    this network immediately narrowed the fp32 result back to bf16 (see below),
    and writing fp32 to memory to read it back and write bf16 is a full-volume
    read plus two full-volume writes that do no arithmetic.  Priced end to end
    before this landed, paired and alternating arms: **-36.39 +/- 1.55 ms of a
    449.56 ms step at scale 8 on one MI300A** (peak memory 42.78 -> 38.49 GiB)
    and -3.93 +/- 1.19 ms of 66.49 at scale 7, of which -22.13 ms at scale 8 is
    the cast traffic itself and the rest is the GroupNorm's own halved store
    and a cheaper max-pool.

    **Why it is safe here, and what would invalidate that.**  Every consumer of
    a GroupNorm output in this UNet rounds it to autocast's dtype before doing
    any arithmetic: the following convolution (``aten::convolution`` carries the
    ``lower_precision_fp`` policy), the skip concatenation (``_skip_concat``
    casts to :func:`~ScaFFold.unet.unet_parts._consumer_dtype` for exactly this
    reason), and ``max_pool3d``, which is a selection and commutes with
    rounding.  So no consumer in *this* model ever sees the fp32 bits it used
    to be handed.  A future consumer that does want them -- an fp32 residual
    add, a loss term, a normalization whose statistics are taken over the
    GroupNorm output, anything reading it outside an autocast region -- will
    get bf16 **silently**, with no error and no warning, only ~3 decimal digits
    where it expected ~7.  Such a consumer must upcast explicitly, or take its
    input from somewhere else.  This is the one place ``FastGroupNorm`` is not
    a drop-in replacement, and it is deliberate.

    The departure is *not* free numerically, in one respect worth naming: it
    does not change the forward's value (which is the same fp32 number, rounded
    once), but it changes the backward's reduction order, so a run is not
    bitwise comparable with one taken before it.  Measured at 4.4e-06 relative
    on the loss by step 23 at scale 8 -- the same class and size as the other
    fp32-noise changes on this branch.  Each configuration remains bitwise
    reproducible with itself.

    All three rungs do this, not just the Triton one.  The Triton kernel is
    told to store the input's dtype (:func:`_triton_forward` passes
    ``out_dtype``); the compiled and eager rungs cast afterwards
    (:func:`_match_input_dtype`).  Divergence would have been cheaper -- the
    fallback rungs pay one cast -- and it was rejected: the rungs are chosen
    per module and per *process*, a latch can demote an unproven module
    mid-run, and a proven module still falls back when its kernel raises
    outside a backward replay.  A dtype that depended on that choice would be
    an activation width that changes mid-step, differs between DDP ranks with
    different latch histories, and -- the sharp one -- breaks
    ``torch.utils.checkpoint``, whose non-reentrant recompute compares the
    *dtype* of every saved tensor against the original and raises
    ``CheckpointError`` on a mismatch.  The rungs already go to some length to
    be indistinguishable in everything a caller can observe (see
    ``_match_memory_format`` and the module docstring's "Latches"); dtype is
    now on that list, and the cast that keeps it there is one the consumer
    would have paid anyway.

    DistConv's ``DCTensor`` gets the fast kernels by being unwrapped to its
    local shard in front of them, rather than by letting the op dispatch through
    the wrapper.  Dispatch would work -- the Triton kernel is a real dispatcher
    op, so ``DCTensor.__torch_dispatch__`` would unwrap, run and rewrap on its
    own -- but the explicit unwrap keeps the subclass policy in one place
    (``is_supported`` accepts any ``torch.Tensor`` *instance*, so dispatch would
    silently extend the fast path to every unknown wrapper), lets the
    eligibility predicates examine the tensor the kernel will actually touch
    rather than a wrapper's mirrored metadata, and lets both fast rungs share
    one unwrap and one fallback ladder.  Semantics are unchanged either way:
    DistConv's generic ``__torch_dispatch__`` has no GroupNorm-specific
    handling, so statistics stay per-shard and nothing communicates at any shard
    count.
    """

    #: Class-level defaults, so that an instance restored from a *module* pickle
    #: written before these attributes existed (``torch.save(model)`` rather
    #: than a state dict) still runs: ``nn.Module.__setstate__`` replaces
    #: ``__dict__`` wholesale, so anything only ever set in ``__init__`` is
    #: missing on such an instance.
    activation = None

    #: Per-module "a call has been served by this rung".  A global latch does
    #: not demote a module that has one, which is what keeps a checkpointed
    #: block's forward and its recompute on the same rung; see the module
    #: docstring's "Latches".  Plain attributes, so they are not parameters,
    #: buffers or state-dict keys.
    _triton_ok = False
    _compiled_ok = False

    #: How this ladder is named in the startup kernel-selection line.  The line
    #: reports Triton against everything else, so ``_compiled_ok`` does not
    #: appear there: from the outside the compiled and eager rungs are both
    #: "what PyTorch does".
    _rung_label = "GroupNorm"

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

        In place: ``out`` is a freshly allocated GroupNorm output with no other
        consumer, and GroupNorm's backward reads its *input*, never its output,
        so overwriting it is safe for autograd as well as for memory.

        Validated here rather than only in ``__init__``: ``activation`` is a
        plain attribute and can be assigned after construction, and the Triton
        rung would then fuse an activation this method silently skipped -- the
        network's function would depend on its input's memory format.  It is
        also what makes adding a third entry to ``SUPPORTED_ACTIVATIONS`` a loud
        failure until it is implemented here.
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
        """The native channels-last kernel, with the activation fused in.

        ``out_dtype=local.dtype`` is this module's departure from
        ``F.group_norm``'s autocast contract, spelled at the one call site that
        wants it rather than in the kernel module's default -- the standalone
        ``triton_group_norm`` is documented as reproducing ``F.group_norm``'s
        dtype and still does.  The kernel stores that dtype directly, so unlike
        the two rungs below there is no fp32 intermediate to cast; the
        statistics are fp32 either way.  See the class docstring's "Output
        dtype".
        """
        return _get_triton_module().triton_group_norm(
            local,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps,
            self.activation,
            out_dtype=local.dtype,
        )

    def _compiled_forward(self, local):
        return _match_input_dtype(
            self._activate(
                _match_memory_format(
                    _get_compiled_group_norm()(
                        local, self.num_groups, self.weight, self.bias, self.eps
                    ),
                    local,
                )
            ),
            local,
        )

    def _eager_forward(self, input):
        # super().forward() is the stock kernel; deferring to it keeps the eager
        # path identical to nn.GroupNorm's (plus the ReLU, the relayout and the
        # dtype narrowing) by construction.
        return _match_input_dtype(
            self._activate(_match_memory_format(super().forward(input), input)), input
        )

    def forward(self, input):
        global _compile_failed, _triton_failed

        distconv = _dctensor_ops(input)
        # The eligibility checks look at the local shard for a DCTensor (a plain
        # attribute read, no autograd involvement) and at the tensor itself
        # otherwise.
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
                # A broken Triton install, an unwritable JIT cache or a shape
                # the kernel mishandles must cost speed, not a multi-node run.
                # GroupNorm is pure and the kernel raises this only from its
                # launch region, before it has saved anything, so retrying the
                # same call on the compiled kernel below is safe -- and the
                # compiled kernel, not eager, is the right landing place.
                #
                # Logged on the latch's False->True edge only: a module that has
                # already used the rung keeps trying it (that is what pins a
                # checkpointed block to one rung), so a persistently broken
                # kernel would otherwise warn once per call for the rest of the
                # run.  Clearing the latch re-arms the message.
                first = not _triton_failed
                _triton_failed = True
                if first:
                    _warn_rung_failure(
                        "Triton GroupNorm", e, "compiled kernel", TRITON_ENV_VAR
                    )
                # ... with one exception, shared with the compiled rung below
                # and explained there: a module already proven on this rung must
                # not be answered from a different one while a backward is
                # replaying its forward.
                if self._triton_ok and _replaying_a_forward():
                    raise
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
            # The one call this rung must not answer eagerly: a module already
            # proven on it, failing while a backward is in flight, is a
            # checkpoint recompute of a forward that *did* run compiled.  The
            # rungs save different tensors, so the eager result would make the
            # recomputed saved set disagree with the original and torch would
            # reject the step with a `CheckpointError`.  Degrading is for
            # modules with nothing to contradict; here the honest answer is the
            # original exception, which names the rung and the shape that could
            # not be served.
            if self._compiled_ok and _replaying_a_forward():
                raise
            return self._eager_forward(input)
        else:
            if not self._compiled_ok:
                self._compiled_ok = True
            return out
