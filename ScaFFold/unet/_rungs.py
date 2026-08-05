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

"""Primitives shared by the modules that run a fast kernel with a fallback.

:mod:`ScaFFold.unet.group_norm` and :mod:`ScaFFold.unet.conv3d` both present the
same shape: a hand-written kernel tried first, the stock one behind it, a
per-rung failure latch and a per-module "this rung has served me" flag.  The
pieces collected here are the ones that are *not* about either kernel and that
each carry a correction paid for once already:

* :func:`_env_override` -- the opt-in/opt-out spelling, so both modules accept
  the same words and warn about the same typos.
* :func:`_platform_declines` -- the hardware guard, which is the one routing
  condition whose failure mode is a *correct* answer at the wrong speed.
* :func:`_replaying_a_forward` -- the "a backward is in flight" probe, with its
  ``is_compiling()`` guard.
* :data:`_functorch_active` -- the ``torch.func`` probe, which is a routing
  question and not a kernel defect.
* :func:`_dctensor_ops` -- DistConv resolved through ``sys.modules`` rather than
  imported, so ScaFFold stays importable without it.
* :func:`_run_local` -- the ``DCTensor`` unwrap/rewrap, through DistConv's
  autograd pair rather than a bare ``_tensor`` read.
* :func:`_warn_rung_failure` -- the fallback message, with its ``is_compiling()``
  guard.

They live here rather than being copied because a copy drifts: every one of the
guards above was added in response to a specific measured failure, and the next
module to grow a ladder should inherit them rather than rediscover them.

What is deliberately *not* here: the allowlist of exceptions a rung may fail
with.  That is a property of the kernel behind the rung, closed at that kernel's
own boundary, and it has to be written next to the rung that uses it.
"""

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# The hardware guard
# ---------------------------------------------------------------------------

#: The GPU architecture every tuning table in both ladders was raced on, with
#: its feature suffixes stripped.  ``gcnArchName`` reports
#: ``"gfx942:sramecc+:xnack-"`` here, and the suffixes are a property of how the
#: *build* was configured rather than of the silicon, so an exact string
#: comparison would decline the same chip under a different HIP build.  Compared
#: against the part before the first colon for that reason.
TUNED_ARCH = "gfx942"

#: ...and the compute-unit count, which is what makes this MI300A rather than
#: ``gfx942``.  ``gfx942`` is three parts: the MI300A APU (228 CUs), the MI300X
#: (304) and the MI325X (304).  The tables here were raced on the first one and
#: several of them are *written in terms of* its geometry --
#: ``gather_gemm.candidate_configs`` defaults ``GROUP_M`` to 6 because that is
#: MI300A's XCD count, ``triton_group_norm._TUNED``'s largest entry pins a grid
#: of 228, and ``conv3d._policy_declines``'s small-``M`` rule is the shape of the
#: cliff where a GEMM stops filling 228 CUs.  None of those is wrong on an
#: MI300X in the sense of returning bad numbers; all of them are simply about a
#: different machine.
#:
#: The CU count also declines a *partitioned* MI300A, which is the case an
#: arch-only test would silently accept and which is genuinely mistuned: CPX
#: mode presents one logical device per XCD, so both the 228 and the 6 above
#: become fiction while ``gcnArchName`` stays exactly the same.
TUNED_CU_COUNT = 228

#: Both ladders' opt-in switches, named together in the one message a user on
#: another device gets.  The verdict is a fact about the *node*, not about
#: either kernel, so a user who has just discovered that the fast path is off
#: should not have to find the second switch separately.
_OVERRIDE_SWITCHES = "SCAFFOLD_CONV_TRITON=1 / SCAFFOLD_GROUPNORM_TRITON=1"

#: ``device index -> (is the tuned platform, human description)``.  Resolved on
#: the first eligible call per device and never again; see
#: :func:`_platform_verdict`.
_PLATFORM_VERDICTS = {}

#: Device indices that have already produced their one message, per kind.
_PLATFORM_DECLINE_WARNED = set()
_PLATFORM_OVERRIDE_WARNED = set()


def _device_fingerprint(index):
    """``(arch, cu_count, name)`` for CUDA/HIP device ``index``.

    Split out from :func:`_platform_verdict` for exactly one reason: it is the
    **seam the tests replace**.  Every interesting branch of this guard is the
    one this node cannot take -- an MI300X, a partitioned MI300A, an NVIDIA
    device, a driver that will not answer -- so the predicate has to be
    injectable or those branches are shipped unexecuted.  Keeping the whole
    query (and nothing else) behind one function means a test substitutes a
    tuple and exercises the real decision, the real caching and the real
    message, rather than a parallel copy of them.

    ``gcnArchName`` exists only on a ROCm build; a CUDA build of torch has no
    such attribute, so an NVIDIA device is described as an empty arch and
    declines through the same clause an untuned AMD one does.  That is the right
    answer for a reason beyond tuning: the kernels' launch constraints are MFMA
    constraints (``gather_gemm._MFMA_KDIM``, ``matrix_instr_nonkdim``) and mean
    nothing on a device with no MFMA.
    """
    props = torch.cuda.get_device_properties(index)
    arch = getattr(props, "gcnArchName", "") or ""
    return (
        arch.split(":")[0],
        int(getattr(props, "multi_processor_count", 0)),
        str(getattr(props, "name", "")),
    )


def _device_index(device):
    """The integer this device is cached and named by.

    A tensor's ``device`` always carries an index, but a bare
    ``torch.device("cuda")`` does not, and that one means "whichever is
    current" -- which is what the kernel would launch on.
    """
    return device.index if device.index is not None else torch.cuda.current_device()


def _platform_verdict(device):
    """Whether ``device`` is the machine the tables were tuned on, cached.

    Returns ``(ok, description)`` and computes it **once per device index** for
    the life of the process.  Cached because it is asked on the routing path of
    every convolution and every GroupNorm -- 40-odd calls per step -- and
    ``get_device_properties`` is a driver query, not an attribute read; and
    computed lazily rather than at import because a CPU-only run (the whole CPU
    unit suite) must not initialize the GPU at all.  The callers only reach here
    after ``is_cuda``, so by then torch's CUDA state is already up.

    **Per device index, not per process.**  ScaFFold pins one rank per GPU, so
    in production this dictionary holds exactly one entry and the distinction is
    invisible.  It is keyed anyway because the question the guard is asking is
    "will the kernel that is about to launch be running on the machine its
    launch configuration was chosen for", and that is a property of the device
    the tensor is on -- so on a node that exposes two different GPUs a
    process-wide answer taken from device 0 would be wrong on one of them in
    whichever direction hurts more.  It costs a dictionary lookup on an int.

    A driver that will not answer is *not* the tuned platform: the guard's whole
    job is to be sure, and "I could not find out" is not "yes".
    """
    index = _device_index(device)
    verdict = _PLATFORM_VERDICTS.get(index)
    if verdict is None:
        try:
            arch, cus, name = _device_fingerprint(index)
        except Exception as e:  # a driver query has no correct answer to invent
            arch, cus, name = "", 0, f"<unavailable: {type(e).__name__}: {e}>"
        ok = arch == TUNED_ARCH and cus == TUNED_CU_COUNT
        verdict = (ok, f"{name} (arch {arch or 'unknown'}, {cus} CUs)")
        _PLATFORM_VERDICTS[index] = verdict
    return verdict


def _reset_platform_cache():
    """Forget every cached verdict and every message already emitted.

    For tests only, and it exists because the cache above is process-global:
    without it the first test to ask a question would fix the answer for every
    later one, so a suite that checks the decline path and then the accept path
    would pass while testing the first one twice.
    """
    _PLATFORM_VERDICTS.clear()
    _PLATFORM_DECLINE_WARNED.clear()
    _PLATFORM_OVERRIDE_WARNED.clear()


def _platform_declines(device, override):
    """``True`` when ``device`` is not the hardware these kernels were tuned for.

    What this is protecting against is **silent mistuning, not a crash**.  Every
    number that decides how these kernels launch was raced on one MI300A: the
    convolution tile tables and ``matrix_instr_nonkdim``/``kpack``/
    ``waves_per_eu`` choices in ``triton_conv3d``, the ``GROUP_M = 6`` that is
    MI300A's XCD count, and ``triton_group_norm._TUNED``, whose largest entry
    names a grid of 228.  Run somewhere else those are not *wrong answers*, they
    are answers to a question about a different machine -- and there is no
    mechanism downstream that would notice:

    * ``triton_conv3d.gather_gemm``'s docstring records that on gfx942 an
      illegal MFMA configuration does not fail.  It emits **zero** MFMA
      instructions, drops to vector FMA, and returns correct results at a
      fraction of the speed.  On an architecture whose legality rules differ
      from the ones ``ConvConfig.validate`` encodes, that is precisely the
      failure available: a right answer, slowly, raising nothing.
    * the ladders' fallback allowlist is ``triton.errors.TritonError`` and
      deliberately nothing else, so there is no exception for it to catch even
      in principle.
    * every ``is_supported*`` predicate reads shape, dtype, layout and stride.
      None of them reads the GPU, and none of them should: they are *capability*
      predicates, and the kernels really are capable of computing this
      convolution on other hardware.

    So the guard is a **preference, not a correctness condition**, and the code
    says so by letting an explicit opt-in through.  ``override`` is the caller's
    tri-state ``_triton_override``: ``None`` (the default, and every production
    run) means "on wherever it is safe", where this device is part of what
    "safe" means; ``True`` is a human who has typed ``SCAFFOLD_CONV_TRITON=1``
    or called ``set_conv_triton_enabled(True)`` and is therefore asserting a
    judgement about their own hardware, which is exactly the development case
    this override exists for.  ``False`` never reaches here -- the callers
    decline on it first -- so the two states this function distinguishes are
    "nobody said" and "somebody said yes".

    Both branches are loud, once per device: a decline that said nothing would
    leave a user on an MI300X with a 1.26x regression and no thread to pull, and
    an override that said nothing would leave every subsequent measurement on
    that machine uncomparable with the tables it will be read against.
    """
    ok, described = _platform_verdict(device)
    if ok:
        return False
    index = _device_index(device)
    if override is True:
        _warn_platform_override(index, described)
        return False
    _warn_platform_decline(index, described)
    return True


def _warn_platform_decline(index, described):
    """The one message a user on untuned hardware gets.

    Once per device index -- not once per call, which at 40-odd routed
    operations a step would be a log line every few milliseconds, and not
    silence, which is the state this whole change exists to end.  The message
    names the device it found, the device it wanted, and both switches, because
    a user who has just noticed the fast path is off is asking one question and
    should not have to find the second ladder's control separately.

    ``is_compiling()`` for the reason :func:`_warn_rung_failure` documents:
    Dynamo cannot trace a ``logging.Logger`` method, so a bare call here would
    turn a *routing decision* into a hard error for a caller who wrapped the
    forward in ``torch.compile(fullgraph=True)``.
    """
    if index in _PLATFORM_DECLINE_WARNED:
        return
    _PLATFORM_DECLINE_WARNED.add(index)
    if torch.compiler.is_compiling():
        return
    logger.warning(
        f"ScaFFold's Triton kernels are tuned for {TUNED_ARCH} with "
        f"{TUNED_CU_COUNT} CUs (AMD Instinct MI300A); cuda:{index} is "
        f"{described}. Using the fallback kernels there. The Triton kernels are "
        "correct on other hardware -- every launch configuration in them was "
        "chosen on that device, so what is unknown is their speed, and a "
        f"mistuned launch reports nothing. Set {_OVERRIDE_SWITCHES} to use them "
        "anyway."
    )


def _warn_platform_override(index, described):
    """The message the override owes, once per device index.

    Deliberately not quiet.  Every performance figure either ladder is read
    against -- the 1.26x step, the block-list in ``conv3d._policy_declines``,
    ``triton_group_norm``'s roofline percentages -- was measured on the device
    this run is *not* on, so a number produced under this override is not
    comparable with any of them, and a log line is the only place that fact can
    be recovered from afterwards.
    """
    if index in _PLATFORM_OVERRIDE_WARNED:
        return
    _PLATFORM_OVERRIDE_WARNED.add(index)
    if torch.compiler.is_compiling():
        return
    logger.warning(
        f"Taking the Triton kernels on cuda:{index}, which is {described}, "
        f"because they were explicitly enabled. They are tuned for "
        f"{TUNED_ARCH} with {TUNED_CU_COUNT} CUs and nothing here has been "
        "measured on this device: expect correct numbers and unknown speed, and "
        "do not compare timings from this run against the tuned ones."
    )


def _replaying_a_forward():
    """``True`` while this thread is executing inside an autograd graph task.

    ``torch._C._current_graph_task_id()`` is ``-1`` outside a backward pass and
    the running task's id inside one; it is the same signal
    ``torch.utils.checkpoint`` keys its own recompute bookkeeping on
    (``torch/utils/checkpoint.py``'s ``unpack_hook``).

    A module *forward* that runs while a backward is in flight is not a new
    call: it is a checkpoint recompute (or a double backward) replaying a
    forward that has already happened and whose saved tensors are already held.
    That is the one place where quietly answering on a different rung than the
    original forward used is not a fallback but a corruption -- the rungs do not
    save interchangeable tensors, so the recompute's saved set no longer matches
    the graph node that will consume it.  For GroupNorm that shows up as
    ``CheckpointError: Recomputed values ... have different metadata`` (measured,
    and on one shape a GPU memory fault instead); for convolution under DistConv
    it is worse, because the two rungs save tensors with *identical* shape,
    dtype and device and differ only in whether slot 0 is the ``DCTensor``
    wrapper or its inner tensor -- which is precisely what
    ``_default_meta_extractor`` does not compare, so the substitution succeeds
    and fails later as an ``AttributeError`` from inside DistConv (measured, both
    directions).  See the callers.

    ``is_compiling()`` first, for the same reason :func:`_warn_rung_failure`
    checks it: the probe below is a ``torch._C`` builtin returning an ``int``,
    which Dynamo cannot trace ("Unsupported torch.* op returned non-Tensor"),
    so a caller who wraps a ``forward`` in ``torch.compile(fullgraph=True)``
    would get a hard error where the fallback belongs.  Dynamo folds it to
    ``True`` at trace time, leaving ``False`` here as a constant -- which is
    also the right answer: tracing is not replaying, and the recompute this
    guards against runs with Dynamo disabled anyway
    (``torch.utils.checkpoint``'s ``_run_fn_with_dynamo_disabled``).
    """
    if torch.compiler.is_compiling():
        return False
    task_id = getattr(torch._C, "_current_graph_task_id", None)
    if task_id is None:  # pragma: no cover - every supported torch has it
        return False
    return task_id() != -1


#: ``True`` while a ``torch.func`` transform (``vmap``/``grad``/``jvp``) is on
#: the stack.  A fast rung declines then: a functorch layer is a routing
#: question, not a kernel defect, and the stock kernel handles every transform.
#: Not merely a performance choice -- an ``is_supported``'s
#: ``is_contiguous(memory_format=...)`` raises outright under ``vmap``
#: ("NYI: querying is_contiguous inside of vmap"), and neither hand-written op
#: has a batching rule -- so without this the modules are not the drop-in
#: replacements they claim to be for any caller using ``torch.func``.
_functorch_active = getattr(torch._C, "_are_functorch_transforms_active", lambda: False)


def _dctensor_ops(input):
    """The ``distconv.distconv`` module when ``input`` is a DCTensor, else None.

    Resolved through ``sys.modules`` instead of an import: a DCTensor can only
    exist if DistConv is already imported, and ScaFFold's model must stay
    importable (and the CPU suite runnable) without DistConv installed.
    """
    distconv = sys.modules.get("distconv.distconv")
    if distconv is not None and isinstance(input, distconv.DCTensor):
        return distconv
    return None


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

    Note what this does *not* do: it does not consult the parallel strategy.
    Whether running the kernel on the local shard alone is the same computation
    DistConv's dispatch would have performed is the caller's question, and the
    answer differs by operator -- see :class:`~ScaFFold.unet.conv3d.FastConv3d`,
    where it is only true when nothing is actually sharded.
    """
    if distconv is None:
        return kernel(input)
    local = distconv._ToTensor.apply(input)
    return distconv.DCTensor.from_shard(kernel(local), input._parallel_strategy)


def _warn_rung_failure(what, error, fallback, env_var):
    """Log a rung failure, without graph-breaking a compiled caller.

    Dynamo cannot trace ``logging.Logger`` methods ("Unsupported: logging.Logger
    method not supported for non-export cases"), so a bare ``logger.warning``
    in a ladder's handler turns a *fallback* into a hard Dynamo error for
    anyone who wraps that ``forward`` in ``torch.compile(fullgraph=True)`` --
    the one caller for whom the fallback matters most, since the failure it is
    reacting to is usually a compile failure.  ``is_compiling()`` is a Dynamo
    intrinsic that folds to ``True`` at trace time, so the call below becomes
    dead code inside a traced region and the fallback traces cleanly.  The
    latch itself is a global assignment, which Dynamo does replay, so the
    fallback is still recorded -- only this message is dropped, and only for a
    caller that is compiling the module's forward (nothing in ScaFFold does).
    """
    if torch.compiler.is_compiling():
        return
    logger.warning(
        f"{what} failed ({type(error).__name__}: {error}); falling back to the "
        f"{fallback} for modules that have not already used it. "
        f"Set {env_var}=0 to skip this attempt entirely."
    )
