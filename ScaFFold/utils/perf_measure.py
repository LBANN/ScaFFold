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

import logging
import os
from contextlib import nullcontext

CALI_PERF_ENV_VAR = "CALI_CONFIG"
TORCH_PERF_ENV_VAR = "PROFILE_TORCH"

# This module is imported before (and independently of) the run's MPI logger,
# so it keeps its own. Everything it has to say is about the user's profiling
# request not being honored as written, which belongs on a diagnostic channel
# that a caller can filter or capture -- not on stdout, where it lands in the
# middle of whatever the run is printing.
logger = logging.getLogger(__name__)


def _profiler_env_flag(name):
    """Return True only for an affirmative value of the environment variable.

    Every profiler toggle -- the master switch and its sub-options alike --
    goes through this, so "0"/"false"/"no"/"off"/"" all mean off and there is
    no spelling that means the opposite of what it says.
    """
    return os.environ.get(name, "").lower() in ("1", "true", "on", "yes")


_CALI_PERF_ENABLED = False
TORCH_PERF_ENABLED = False
if CALI_PERF_ENV_VAR in os.environ:
    try:
        from pyadiak.annotations import fini, init, value
        from pycaliper import annotate_function
        from pycaliper.instrumentation import begin_region, end_region

        _CALI_PERF_ENABLED = True
    except Exception as e:
        logger.warning(
            "User requested Caliper annotations, but could not import Caliper: %s: %s",
            type(e).__name__,
            e,
        )

# The torch profiler is gated purely on its own environment variable: Caliper
# and the torch profiler may both be enabled at once.
if _profiler_env_flag(TORCH_PERF_ENV_VAR):
    try:
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as torchprofile

        TORCH_PERF_ENABLED = True
    except Exception:
        logger.warning(
            "User requested PyTorch profiling, but could not import the "
            "PyTorch profiler"
        )


def annotate(name=None, fmt=None):
    def inner_decorator(func):
        if not _CALI_PERF_ENABLED:
            return func
        else:
            real_name = name
            if name is None or name == "":
                real_name = func.__name__
            if fmt is not None and fmt != "":
                real_name = fmt.format(real_name)
            return annotate_function(name=real_name)(func)

    return inner_decorator


def begin_code_region(name):
    if _CALI_PERF_ENABLED:
        begin_region(name)
        return


def end_code_region(name):
    if _CALI_PERF_ENABLED:
        end_region(name)
        return


def adiak_init(comm):
    if _CALI_PERF_ENABLED:
        init(comm)
        return


def adiak_value(name, val):
    if _CALI_PERF_ENABLED:
        value(name, val)
        return


def adiak_fini():
    if _CALI_PERF_ENABLED:
        fini()
        return


def _profiler_env_int(name, default):
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def get_torch_context(ranks_per_node, rank):
    if TORCH_PERF_ENABLED:
        TORCH_PERF_LOCAL = TORCH_PERF_ENABLED and (rank % ranks_per_node == 0)
        if not TORCH_PERF_LOCAL:
            return nullcontext(), False

        from torch.profiler import schedule as torch_profiler_schedule

        # Record only a bounded window of steps instead of the whole multi-epoch
        # run: the profiler accumulates every event in host memory until export,
        # so an unscheduled long run grows without bound and emits an unusable
        # trace. The window (skip `wait`, prime `warmup`, capture `active`, once)
        # is tunable via the environment. Callers must drive it with
        # ``prof.step()`` once per training step for the schedule to advance.
        # The context is entered around checkpoint cleanup and the warmup
        # batches, but prof.step() only advances once per *training* batch, so
        # everything before the first training batch lands in step 0. The
        # window must therefore skip at least one step: with wait=0 that whole
        # prologue -- warmup_batches forward+backward passes per rank -- is
        # buffered in host memory as a single unbounded step, which is the very
        # thing the bounded window exists to prevent.
        wait = _profiler_env_int("PROFILE_TORCH_WAIT", 1)
        if wait < 1:
            logger.warning(
                "PROFILE_TORCH_WAIT must be at least 1: the profiler window "
                "opens before the warmup batches, whose work would otherwise "
                "accumulate in host memory as one unbounded step. Using "
                "PROFILE_TORCH_WAIT=1."
            )
            wait = 1
        warmup = _profiler_env_int("PROFILE_TORCH_WARMUP", 1)
        active = _profiler_env_int("PROFILE_TORCH_ACTIVE", 3) or 1

        # record_shapes and with_stack are the two most expensive options (they
        # skew the very timings being measured), so they are opt-in rather than
        # always on.
        prof_ctx = torchprofile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=torch_profiler_schedule(
                wait=wait, warmup=warmup, active=active, repeat=1
            ),
            record_shapes=_profiler_env_flag("PROFILE_TORCH_RECORD_SHAPES"),
            with_stack=_profiler_env_flag("PROFILE_TORCH_WITH_STACK"),
        )
        return prof_ctx, TORCH_PERF_LOCAL
    return nullcontext(), False
