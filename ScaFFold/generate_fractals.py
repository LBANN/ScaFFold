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

from argparse import Namespace

from mpi4py import MPI

from ScaFFold.datagen import category_search, instance
from ScaFFold.utils.utils import setup_mpi_logger


def main(kwargs_dict: dict = {}):
    args = Namespace(**kwargs_dict)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    log = setup_mpi_logger(__file__, getattr(args, "verbose", 0))

    log.info("Fractal generation world size = %s", size)

    comm.Barrier()

    category_search.main(args)

    comm.Barrier()

    instance.main(args)

    comm.Barrier()

    log.info("Fractal and instance generation has finished.")

    MPI.Finalize()

    return 0
