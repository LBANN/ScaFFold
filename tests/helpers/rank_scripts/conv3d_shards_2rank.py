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

"""Two-rank check that FastConv3d's halo exchange is agreed, never rank-local.

Run under ``torchrun --nproc_per_node=2`` with gloo (see ``tests/test_conv3d.py``):
two CUDA shards of one volume split on D over a real ``ParallelStrategy``,
through ``_Halo3d`` and the kernel, against ``F.conv3d`` on the whole volume;
then a local decline, a latch, and a shard too thin for a plan, each of which
must leave both ranks on the same side of the exchange.  Every scenario builds
a fresh module from one seed, since the agreement is per instance.  Each rank
prints one ``RESULT`` line per scenario plus ``DONE``; the parent asserts.
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ScaFFold.unet import conv3d as conv_mod  # noqa: E402
from ScaFFold.unet.conv3d import FastConv3d  # noqa: E402

CHANNELS = 16
SIZE = 8
_CHANNELS_LAST = torch.channels_last_3d


def _module(device):
    """A fresh ``FastConv3d(16, 16, 3, padding=1)``, bf16 channels-last."""
    conv = FastConv3d(CHANNELS, CHANNELS, 3, padding=1, bias=False)
    generator = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        conv.weight.normal_(0.0, 0.1, generator=generator)
    return conv.to(device, torch.bfloat16).to(memory_format=_CHANNELS_LAST)


def _volume(depth, device):
    """The same global volume on every rank, from one seed."""
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, CHANNELS, depth, SIZE, SIZE, generator=generator)
    return x.to(device, torch.bfloat16).contiguous(memory_format=_CHANNELS_LAST)


def _slice(tensor, rank, splits):
    """This rank's slab of a volume split on D into ``splits``."""
    start = sum(splits[:rank])
    return tensor.narrow(2, start, splits[rank]).contiguous(
        memory_format=_CHANNELS_LAST
    )


def run():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    import distconv

    ps = distconv.ParallelStrategy(
        num_shards=(2, 1, 1), shard_dim=(2, 3, 4), device_type="cuda"
    )

    # Spies on the collective, the kernel node and the MIOpen rung.  The
    # MIOpen stub never runs DistConv's own exchange: gloo rejects its
    # channels-last send buffers, and which rung was chosen is the assertion.
    votes, launched, served = [], [], []
    all_reduce = dist.all_reduce
    dist.all_reduce = lambda *a, **kw: votes.append(a) or all_reduce(*a, **kw)
    apply = conv_mod._TritonConv3dFn.apply
    conv_mod._TritonConv3dFn.apply = staticmethod(
        lambda *a: launched.append(a) or apply(*a)
    )
    FastConv3d._miopen_forward = lambda self, input: served.append(input) or input

    def scenario(name, depth, splits):
        """One forward per call on a fresh module; the parent reads the line."""
        conv = _module(device)
        volume = _volume(depth, device)
        with torch.no_grad():
            reference = _slice(F.conv3d(volume, conv.weight, None, 1, 1), rank, splits)
        del votes[:], launched[:], served[:]

        def forward():
            local = _slice(volume, rank, splits)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = conv(distconv.DCTensor.from_shard(local, ps))
            return out._tensor

        outs = [forward() for _ in range(2 if name == "uniform" else 1)]
        match = (
            "skipped"
            if served
            else all(
                torch.allclose(o.float(), reference.float(), rtol=2e-2, atol=2e-2)
                for o in outs
            )
        )
        # No spaces in any field: torchrun interleaves the ranks' stdout, so
        # the parent's regex has to tell two RESULT lines apart when they run
        # together.
        print(
            f"RESULT scenario={name} rank={rank} agreed={conv._exchange_agreed} "
            f"local={conv._local_verdict} triton_ok={conv._triton_ok} "
            f"kernel={bool(launched)} match={match} miopen={len(served)} "
            f"votes={len(votes)}",
            flush=True,
        )

    # Both ranks eligible: the exchange and the kernel, twice, one vote.
    scenario("uniform", SIZE, (4, 4))

    # Rank 1's hardware guard says no.  Rank 0 is outvoted and both take
    # DistConv, rather than rank 0 posting receives rank 1 never answers.
    platform_declines = conv_mod._platform_declines
    if rank == 1:
        conv_mod._platform_declines = lambda device, override: True
    scenario("peer_declines", SIZE, (4, 4))
    conv_mod._platform_declines = platform_declines

    # Rank 0 latched with an unproven module: it still exchanges here, on the
    # exchanged tensor runs F.conv3d instead of the kernel, and is then proven.
    if rank == 0:
        conv_mod._triton_failed = True
    scenario("rank0_latched", SIZE, (4, 4))
    conv_mod._triton_failed = False

    # A ragged split: rank 1's shard is one voxel deep, thinner than the halo,
    # so its plan is None and it votes no; rank 0's plan is fine.
    scenario("thin_shard", 3, (2, 1))

    print("DONE", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
