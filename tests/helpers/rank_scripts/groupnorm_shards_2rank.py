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

"""Two-rank check that FastGroupNorm's DCTensor route is shard-count agnostic.

Run under ``torchrun --nproc_per_node=2`` with the gloo backend (see
``tests/test_groupnorm.py``).  Every other DCTensor test uses
``num_shards=(1, 1, 1)``, where sharding is a no-op and the local shard is the
whole tensor; this one shards a spatial dim across two ranks so the claim the
fast path actually rests on -- GroupNorm statistics are per-shard, and the
route does not add communication or change which elements are reduced together
-- is exercised where it can fail.

Each rank prints one ``RESULT ...`` line plus ``DONE``; the parent asserts.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("SCAFFOLD_ROOT", "/usr/WS1/dryden1/ScaFFold"))

from ScaFFold.unet import group_norm as gn_mod  # noqa: E402
from ScaFFold.unet.group_norm import FastGroupNorm  # noqa: E402

GROUPS = 8
CHANNELS = 16
SIZE = 8  # dim 2 is split into two shards of 4


def run():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()

    import distconv

    ps = distconv.ParallelStrategy(num_shards=(2,), shard_dim=(2,), device_type="cpu")

    # The same global volume on both ranks; each takes its own slab.
    generator = torch.Generator().manual_seed(41)
    volume = torch.randn(1, CHANNELS, SIZE, SIZE, SIZE, generator=generator)
    half = SIZE // 2
    local = volume.narrow(2, rank * half, half).contiguous()

    norm = FastGroupNorm(GROUPS, CHANNELS)
    param_generator = torch.Generator().manual_seed(97)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.1, generator=param_generator)
        norm.bias.normal_(0.0, 0.1, generator=param_generator)

    def forward(compiled):
        """Run the wrapped GroupNorm with the compiled route on or off.

        The compiled route is forced on CPU by standing in the stock functional
        kernel for the compiled callable: what is under test here is the
        unwrap/rewrap plumbing at shard counts > 1, not Inductor.
        """
        original_use, original_get = (
            gn_mod._use_compiled,
            gn_mod._get_compiled_group_norm,
        )
        if compiled:
            gn_mod._use_compiled = lambda t, **kw: type(t) is torch.Tensor
            gn_mod._get_compiled_group_norm = lambda: F.group_norm
        else:
            gn_mod._use_compiled = lambda t, **kw: False
        try:
            norm.zero_grad(set_to_none=True)
            x = local.clone().requires_grad_(True)
            out = norm(distconv.DCTensor.from_shard(x, ps))
            assert isinstance(out, distconv.DCTensor), type(out)
            distconv.distconv._ToTensor.apply(out).pow(2).sum().backward()
            weight_grad = norm.weight.grad
            if isinstance(weight_grad, distconv.DCTensor):
                weight_grad = weight_grad._tensor
            return out._tensor.detach().clone(), x.grad.clone(), weight_grad.clone()
        finally:
            gn_mod._use_compiled = original_use
            gn_mod._get_compiled_group_norm = original_get

    eager = forward(compiled=False)
    compiled = forward(compiled=True)
    identical = all(torch.equal(a, b) for a, b in zip(eager, compiled))

    # Per-shard statistics: this rank's output normalizes its own slab only.
    with torch.no_grad():
        per_shard = F.group_norm(local, GROUPS, norm.weight, norm.bias, norm.eps)
        global_slice = F.group_norm(
            volume, GROUPS, norm.weight, norm.bias, norm.eps
        ).narrow(2, rank * half, half)

    # No spaces in any field: torchrun interleaves the ranks' stdout and a
    # line can arrive without its trailing newline, so the parent's regex has
    # to be able to tell two RESULT lines apart when they run together.
    shape = "x".join(str(dim) for dim in compiled[0].shape)
    print(
        f"RESULT rank={rank} shape={shape} "
        f"identical={identical} "
        f"per_shard={torch.equal(compiled[0], per_shard)} "
        f"global={torch.allclose(compiled[0], global_slice, atol=1e-6)}",
        flush=True,
    )
    print("DONE", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
