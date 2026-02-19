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

import os
from contextlib import nullcontext

CALI_PERF_ENV_VAR = "CALI_CONFIG"
TORCH_PERF_ENV_VAR = "PROFILE_TORCH"

_CALI_PERF_ENABLED = False
TORCH_PERF_ENABLED = False
if CALI_PERF_ENV_VAR in os.environ:
    try:
        from pyadiak.annotations import fini, init, value
        from pycaliper import annotate_function
        from pycaliper.instrumentation import begin_region, end_region

        _CALI_PERF_ENABLED = True
    except Exception as e:
        print("User requested Caliper annotations, but could not import Caliper")
        print(f"Exception: {e}")
elif (
    TORCH_PERF_ENV_VAR in os.environ
    and os.environ.get(TORCH_PERF_ENV_VAR).lower() != "off"
):
    try:
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as torchprofile

        TORCH_PERF_ENABLED = True
    except Exception:
        print(
            "User requested PyTorch profiling, but could not import the PyTorch profiler"
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


def get_torch_context(ranks_per_node, rank):
    if TORCH_PERF_ENABLED:
        TORCH_PERF_LOCAL = TORCH_PERF_ENABLED and (rank % ranks_per_node == 0)
        prof_ctx = (
            torchprofile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=True,
            )
            if TORCH_PERF_LOCAL
            else nullcontext()
        )
        return prof_ctx, TORCH_PERF_LOCAL
    return nullcontext(), False
