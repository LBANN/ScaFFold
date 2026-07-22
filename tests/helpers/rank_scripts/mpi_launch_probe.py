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

"""Trivial 2-rank probe used to confirm the detected MPI launcher works.

Some environments have a launcher on ``PATH`` (e.g. ``flux``) that cannot
actually start ranks -- it fails to connect rather than being absent. The
consensus tests run a quick probe through this script; only if every rank emits
``PROBE_OK`` are the real MPI tests attempted, otherwise they skip cleanly.
"""

from mpi4py import MPI

comm = MPI.COMM_WORLD
# A collective forces genuine multi-rank wire-up before printing.
comm.Barrier()
print(f"PROBE_OK {comm.Get_rank()}", flush=True)
