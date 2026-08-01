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

"""GroupNorm with a ``torch.compile``d fast path on GPU.

ATen's GroupNorm computes its per-group statistics with a kernel that launches
one workgroup per ``(batch, group)`` row.  At this benchmark's defaults
(``local_batch_size=1``, ``group_norm_groups=8``) that is 8 workgroups, so on a
228-CU MI300A the normalization runs at a small fraction of achievable
bandwidth and dominates the step: measured 87 ms of a 187 ms step (47%) at
scale 7.  Compiling the same functional GroupNorm hands the reduction to
Inductor, which tiles it across the whole device; the same measurement then
gives a 184.7 ms step at 100.7 ms, with GroupNorm down to ~7% of it.

``FastGroupNorm`` is a drop-in ``nn.GroupNorm``: same parameters, same names,
same shapes, same numerics -- only the kernel differs, so checkpoints are
interchangeable in both directions with any other GroupNorm-based build.  The
compiled path is used only when it is safe and worthwhile, and every rejection
falls back to stock eager ``F.group_norm``:

* non-CUDA tensors (the CPU test suite never pays compile latency),
* tensor subclasses, whose ``__torch_dispatch__`` wrappers Dynamo cannot trace
  -- except DistConv's ``DCTensor``, which is unwrapped to its local shard
  around the compiled kernel instead (see ``FastGroupNorm.forward``),
* an already-compiled enclosing region (the functional call inlines instead),
* an explicit opt-out via ``SCAFFOLD_GROUPNORM_COMPILE=0``,
* any failure inside ``torch.compile`` -- logged once, then eager forever after.

Determinism: the compiled kernels are bitwise reproducible.  Two separate
processes running three fwd+bwd+Adam steps of the scale-7 UNet under
``more_determinism`` (``use_deterministic_algorithms(True, warn_only=True)``,
``cudnn.benchmark=False``, fixed seeds) hash identically with the compiled path,
exactly as they do with the eager one, so no determinism gate is needed.
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


def _env_override():
    """Read ``SCAFFOLD_GROUPNORM_COMPILE``; ``None`` when unset or unparsable."""
    raw = os.environ.get(COMPILE_ENV_VAR)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in ("1", "true", "on", "yes"):
        return True
    if value in ("0", "false", "off", "no"):
        return False
    logger.warning(
        f"Ignoring unrecognized {COMPILE_ENV_VAR}={raw!r}; "
        "expected one of 1/0/true/false/on/off/yes/no"
    )
    return None


_compile_override = _env_override()


def set_compile_enabled(enabled):
    """Force the compiled path on (``True``) or off (``False``).

    ``None`` restores the default, which is the environment variable if set and
    otherwise "compile wherever it is safe".  Forcing it on does not override
    the device and tensor-subclass checks -- those are correctness conditions,
    not preferences.  Returns the previous setting so callers (tests) can
    restore it.
    """
    global _compile_override
    previous = _compile_override
    _compile_override = _env_override() if enabled is None else bool(enabled)
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


def _use_compiled(input):
    """Whether this particular input should take the compiled path."""
    if _compile_failed or _compile_override is False:
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


class FastGroupNorm(nn.GroupNorm):
    """``nn.GroupNorm`` that runs its GPU forward through ``torch.compile``.

    Identical state: ``weight``/``bias`` of shape ``(num_channels,)``, no
    buffers, so state dicts are interchangeable with plain ``nn.GroupNorm``
    in both directions.

    DistConv's ``DCTensor`` gets the compiled kernel too: its generic
    ``__torch_dispatch__`` has no GroupNorm-specific handling -- it unwraps to
    the local shard, runs the stock aten kernels, and rewraps the outputs, so
    statistics are per-shard and no communication happens at any shard count.
    ``forward`` moves that same unwrap up in front of the compiled kernel,
    preserving those semantics exactly (DCTensor in -> DCTensor out) while
    keeping the fast kernel Dynamo's inability to trace the wrapper would
    otherwise forfeit.  It cannot copy dispatch's *mechanism*, though: dispatch
    runs below autograd, where reading ``_tensor`` directly is safe, whereas
    this runs above it, so the unwrap has to go through DistConv's
    ``_ToTensor``/``_FromTensor`` autograd pair or the graph back to the
    producing convolution is severed.
    """

    def forward(self, input):
        distconv = _dctensor_ops(input)
        # The eligibility checks look at the local shard for a DCTensor (the
        # peek is a plain attribute read, no autograd involvement) and at the
        # tensor itself otherwise.
        local_view = input._tensor if distconv is not None else input
        if not _use_compiled(local_view):
            # super().forward() is the stock kernel; deferring to it keeps the
            # eager path identical to nn.GroupNorm's by construction.
            return super().forward(input)
        global _compile_failed
        try:
            if distconv is not None:
                # _ToTensor is the autograd-aware unwrap DistConv itself uses;
                # DCTensor.from_shard is the public spelling of _FromTensor.
                # (There is no public unwrap yet -- upstream ask.)
                local = distconv._ToTensor.apply(input)
                out = _get_compiled_group_norm()(
                    local, self.num_groups, self.weight, self.bias, self.eps
                )
                return distconv.DCTensor.from_shard(out, input._parallel_strategy)
            return _get_compiled_group_norm()(
                input, self.num_groups, self.weight, self.bias, self.eps
            )
        except Exception as e:
            # Compilation is an optimization, never a correctness requirement:
            # a broken Inductor/Triton install, an unwritable cache directory or
            # an untraceable input must degrade to the stock kernel, not kill a
            # multi-node run. GroupNorm is pure, so retrying eagerly is safe.
            _compile_failed = True
            logger.warning(
                f"torch.compile of GroupNorm failed ({type(e).__name__}: {e}); "
                "falling back to the eager kernel for the rest of this run. "
                f"Set {COMPILE_ENV_VAR}=0 to skip this attempt entirely."
            )
            return super().forward(input)
