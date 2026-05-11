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

import numpy as np
import torch

DEFAULT_NP_DTYPE = np.float64
# Masks are values 0 <= x <= n_categories
MASK_DTYPE = np.uint16
# Volumes/img are 0 <= x <= 1
VOLUME_DTYPE_NAME = "float32"
VOLUME_NP_DTYPE = getattr(np, VOLUME_DTYPE_NAME)
VOLUME_TORCH_DTYPE = getattr(torch, VOLUME_DTYPE_NAME)
VOLUME_DTYPE = VOLUME_NP_DTYPE

# Shared AMP dtype selection for torch.autocast.
AMP_DTYPE = torch.bfloat16
