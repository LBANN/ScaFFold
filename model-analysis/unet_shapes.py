#!/usr/bin/env python3
"""Symbolic shape calculator for the ScaFFold 3-D U-Net.

Computes, without importing torch or any ScaFFold code, the exact shapes of
every operation the ScaFFold ``UNet`` performs in a forward pass, for a given
problem scale and an optional DistConv sharding spec.

The model logic mirrors:
  * ScaFFold/unet/unet_model.py + unet_parts.py  (architecture)
  * ScaFFold/worker.py                           (instantiation: n_channels=3,
                                                  n_classes=n_categories+1,
                                                  trilinear=False,
                                                  layers=problem_scale-unet_bottleneck_dim)
  * distconv/distconv.py @ 084b20a               (spatial sharding semantics)

Key derived quantities (from cli.py / config_utils.py):
  vol_size    = 2 ** problem_scale
  unet_layers = problem_scale - unet_bottleneck_dim

DistConv semantics reproduced here (see distconv.py):
  * The global input volume is sharded along tensor dims ``dc_shard_dims``
    (subset of the spatial dims 2/3/4 of an NCDHW tensor) into
    ``dc_num_shards`` equal pieces per dim. Batch and channel are never
    sharded; the batch dim is the per-DDP-replica local batch.
  * Odd-kernel convolutions (3x3x3, 1x1x1) require "same" padding on sharded
    dims. A halo of size k//2 is exchanged with neighbors and concatenated on
    both sides of each sharded dim -- zero slabs at the outer boundaries, so
    every rank's conv input grows by 2*(k//2) on every *listed* shard dim
    (even dims with a single shard get the zero slabs) and the conv then runs
    with padding 0 on those dims.
  * Even-kernel convolutions (ConvTranspose3d k=2 s=2) require padding 0 and
    stride divisible by kernel; no halo is exchanged and the op runs locally.
  * Everything else (MaxPool3d, GroupNorm, ReLU, cat) runs purely locally on
    the shard. Note GroupNorm therefore computes *per-shard* statistics.

Usage:
  unet_shapes.py PROBLEM_SCALE [options]
  unet_shapes.py 8
  unet_shapes.py 8 --dc-num-shards 2 2 2
  unet_shapes.py 8 --dc-num-shards 4 1 1 --dc-shard-dims 2 3 4 --json
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Sharding spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardSpec:
    """Mirror of distconv.ParallelStrategy's (num_shards, shard_dim) pair."""

    num_shards: tuple  # e.g. (2, 2, 2)
    shard_dims: tuple  # tensor dims of NCDHW, e.g. (2, 3, 4)

    def __post_init__(self):
        if len(self.num_shards) != len(self.shard_dims):
            raise ValueError(
                f"dc_num_shards length ({len(self.num_shards)}) must match "
                f"dc_shard_dims length ({len(self.shard_dims)})"
            )
        for d in self.shard_dims:
            if d < 2 or d > 4:
                raise ValueError(
                    f"Invalid shard dim {d}: DistConv only shards spatial dims "
                    "(2, 3, 4 of an NCDHW tensor)."
                )
        if len(set(self.shard_dims)) != len(self.shard_dims):
            raise ValueError(f"Duplicate entries in dc_shard_dims {self.shard_dims}")
        for s in self.num_shards:
            if s < 1:
                raise ValueError(f"num_shards entries must be >= 1, got {s}")

    @property
    def active(self):
        """True when a DistConv parallel strategy is in effect at all."""
        return len(self.shard_dims) > 0

    @property
    def total_shards(self):
        return math.prod(self.num_shards) if self.num_shards else 1

    def shards_for(self, dim):
        """Number of shards along tensor dim ``dim`` (1 if not sharded)."""
        for s, d in zip(self.num_shards, self.shard_dims):
            if d == dim:
                return s
        return 1


NO_SHARDING = ShardSpec((), ())


# ---------------------------------------------------------------------------
# Op records
# ---------------------------------------------------------------------------


@dataclass
class Op:
    idx: int
    block: str
    op: str
    detail: str = ""
    weight_shape: tuple = None
    bias_shape: tuple = None
    in_shapes: list = field(default_factory=list)  # global shapes
    out_shape: tuple = None  # global shape
    local_in_shapes: list = None
    halo: tuple = None  # halo size per spatial dim (D, H, W), convs only
    local_op_in_shape: tuple = None  # local input incl. halo slabs
    local_out_shape: tuple = None
    params: int = 0
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {
            "idx": self.idx,
            "block": self.block,
            "op": self.op,
            "detail": self.detail,
            "weight_shape": self.weight_shape,
            "bias_shape": self.bias_shape,
            "global_in_shapes": self.in_shapes,
            "global_out_shape": self.out_shape,
            "local_in_shapes": self.local_in_shapes,
            "halo_dhw": self.halo,
            "local_op_in_shape": self.local_op_in_shape,
            "local_out_shape": self.local_out_shape,
            "params": self.params,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Shape tracer
# ---------------------------------------------------------------------------


class Tracer:
    """Walks the UNet block structure and records one Op per operation."""

    def __init__(self, batch, shard: ShardSpec, group_norm_groups):
        self.batch = batch
        self.shard = shard
        self.gn_groups = group_norm_groups
        self.ops = []
        self.block = "?"

    # -- helpers ----------------------------------------------------------

    def _local(self, shape):
        """Per-rank local shape of a (sharded) global activation."""
        if not self.shard.active:
            return None
        local = list(shape)
        for s, d in zip(self.shard.num_shards, self.shard.shard_dims):
            if shape[d] % s != 0:
                raise ValueError(
                    f"[{self.block}] global size {shape[d]} on tensor dim {d} is "
                    f"not divisible by num_shards={s}; DistConv cannot "
                    "distribute this tensor evenly."
                )
            local[d] = shape[d] // s
        return tuple(local)

    def _add(self, op: Op):
        op.idx = len(self.ops) + 1
        op.block = self.block
        self.ops.append(op)

    # -- ops --------------------------------------------------------------

    def conv3d(self, shape, c_out, k, stride=1, bias=False):
        """nn.Conv3d(c_in, c_out, kernel_size=k, padding=k//2, bias=bias)."""
        n, c_in, *sp = shape
        pad = k // 2
        out = (n, c_out, *sp)  # stride is always 1 in this model
        op = Op(
            0,
            "",
            "Conv3d",
            f"k={k}³ s={stride} p={pad}" + ("" if bias else " (no bias)"),
            weight_shape=(c_out, c_in, k, k, k),
            bias_shape=(c_out,) if bias else None,
            in_shapes=[shape],
            out_shape=out,
            params=c_out * c_in * k**3 + (c_out if bias else 0),
        )
        if self.shard.active:
            local_in = self._local(shape)
            local_out = self._local(out)
            halo = [0, 0, 0]
            op_in = list(local_in)
            for i, d in enumerate(self.shard.shard_dims):
                # distconv.check_is_distconv_supported: odd kernel needs
                # "same" padding (holds: pad = k//2); halo = k//2. Halo slabs
                # (zeros at outer boundaries) are concatenated on both sides
                # of every listed shard dim, then padding is set to 0 there.
                h = k // 2
                halo[d - 2] = h
                op_in[d] += 2 * h
                if h > 0 and local_in[d] < h:
                    raise ValueError(
                        f"[{self.block}] local size {local_in[d]} on tensor "
                        f"dim {d} is smaller than the halo ({h}); DistConv "
                        "cannot form the halo slab. Reduce shards on this dim."
                    )
            op.local_in_shapes = [local_in]
            op.halo = tuple(halo)
            op.local_op_in_shape = tuple(op_in)
            op.local_out_shape = local_out
            if any(halo):
                # Exchanges run sequentially per sharded dim (distconv.py
                # feeds each dim's output into the next), so the slab for a
                # later dim includes the halos already gained on earlier dims.
                slabs = []
                grown = list(local_in)
                for i, d in enumerate(self.shard.shard_dims):
                    slab = list(grown)
                    slab[d] = halo[d - 2]
                    slabs.append("×".join(str(x) for x in slab))
                    grown[d] += 2 * halo[d - 2]
                op.notes.append(
                    "halo exchange per sharded dim; slab per neighbor: "
                    + "; ".join(slabs)
                )
                op.notes.append("padding on sharded dims replaced by halo slabs")
        self._add(op)
        return out

    def group_norm(self, shape):
        n, c, *sp = shape
        if c % self.gn_groups != 0:
            raise ValueError(
                f"group_norm_groups={self.gn_groups} must evenly divide "
                f"num_channels={c}"
            )
        op = Op(
            0,
            "",
            "GroupNorm",
            f"groups={self.gn_groups}",
            weight_shape=(c,),
            bias_shape=(c,),
            in_shapes=[shape],
            out_shape=shape,
            params=2 * c,
        )
        if self.shard.active:
            op.local_in_shapes = [self._local(shape)]
            op.local_out_shape = self._local(shape)
            op.notes.append("runs locally: statistics are per-shard under DistConv")
        self._add(op)
        return shape

    def relu(self, shape):
        op = Op(0, "", "ReLU", "inplace", in_shapes=[shape], out_shape=shape)
        if self.shard.active:
            op.local_in_shapes = [self._local(shape)]
            op.local_out_shape = self._local(shape)
        self._add(op)
        return shape

    def maxpool(self, shape):
        n, c, *sp = shape
        if self.shard.active:
            local = self._local(shape)
            for d in (2, 3, 4):
                if local[d] % 2 != 0:
                    raise ValueError(
                        f"[{self.block}] local size {local[d]} on tensor dim {d} "
                        "is not divisible by the MaxPool3d(2) stride; the local "
                        "floor-divide would drop voxels at shard boundaries. "
                        "Reduce shards on this dim or use a shallower network."
                    )
        for x in sp:
            if x < 2:
                raise ValueError(
                    f"[{self.block}] spatial size {x} is too small for "
                    "MaxPool3d(2); network is too deep for this volume."
                )
        out = (n, c, *(x // 2 for x in sp))
        op = Op(
            0,
            "",
            "MaxPool3d",
            "k=2³ s=2",
            in_shapes=[shape],
            out_shape=out,
        )
        if self.shard.active:
            op.local_in_shapes = [self._local(shape)]
            op.local_out_shape = self._local(out)
            op.notes.append("runs locally on the shard (no communication)")
        self._add(op)
        return out

    def conv_transpose3d(self, shape, c_out):
        """nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2)."""
        n, c_in, *sp = shape
        out = (n, c_out, *(2 * x for x in sp))
        op = Op(
            0,
            "",
            "ConvTranspose3d",
            "k=2³ s=2",
            weight_shape=(c_in, c_out, 2, 2, 2),
            bias_shape=(c_out,),
            in_shapes=[shape],
            out_shape=out,
            params=c_in * c_out * 8 + c_out,
        )
        if self.shard.active:
            local_in = self._local(shape)
            for d in self.shard.shard_dims:
                # distconv.check_is_distconv_supported: even kernel -> padding
                # must be 0 (holds), stride % kernel == 0 (holds), input size
                # divisible by stride.
                if local_in[d] % 2 != 0:
                    raise ValueError(
                        f"[{self.block}] local size {local_in[d]} on tensor "
                        f"dim {d} is not divisible by the ConvTranspose3d "
                        "stride (2); DistConv rejects this configuration."
                    )
            op.local_in_shapes = [local_in]
            op.halo = (0, 0, 0)
            op.local_op_in_shape = local_in
            op.local_out_shape = self._local(out)
            op.notes.append("even kernel: no halo; runs locally on the shard")
        self._add(op)
        return out

    def upsample(self, shape):
        """nn.Upsample(scale_factor=2, mode='trilinear') -- trilinear path."""
        n, c, *sp = shape
        out = (n, c, *(2 * x for x in sp))
        op = Op(
            0,
            "",
            "Upsample",
            "trilinear x2 align_corners",
            in_shapes=[shape],
            out_shape=out,
        )
        if self.shard.active:
            op.local_in_shapes = [self._local(shape)]
            op.local_out_shape = self._local(out)
            op.notes.append(
                "WARNING: interpolates locally with no halo; values at shard "
                "boundaries differ from the unsharded model"
            )
        self._add(op)
        return out

    def concat(self, skip_shape, up_shape):
        n, c1, *sp1 = skip_shape
        n2, c2, *sp2 = up_shape
        assert sp1 == sp2, (
            f"[{self.block}] skip {skip_shape} and upsampled {up_shape} spatial "
            "mismatch -- with power-of-two volumes Up's F.pad never fires"
        )
        out = (n, c1 + c2, *sp1)
        op = Op(
            0,
            "",
            "cat",
            "dim=1 (skip, upsampled)",
            in_shapes=[skip_shape, up_shape],
            out_shape=out,
        )
        if self.shard.active:
            op.local_in_shapes = [self._local(skip_shape), self._local(up_shape)]
            op.local_out_shape = self._local(out)
        self._add(op)
        return out

    # -- blocks -----------------------------------------------------------

    def double_conv(self, shape, c_out, c_mid=None):
        c_mid = c_mid or c_out
        shape = self.conv3d(shape, c_mid, k=3)
        shape = self.group_norm(shape)
        shape = self.relu(shape)
        shape = self.conv3d(shape, c_out, k=3)
        shape = self.group_norm(shape)
        shape = self.relu(shape)
        return shape


# ---------------------------------------------------------------------------
# Model trace (mirrors UNet.__init__ / UNet.forward)
# ---------------------------------------------------------------------------


def trace_unet(
    problem_scale,
    bottleneck_dim=3,
    n_categories=5,
    n_channels=3,
    batch=1,
    group_norm_groups=8,
    trilinear=False,
    shard=NO_SHARDING,
):
    layers = problem_scale - bottleneck_dim
    if layers < 1:
        raise ValueError(
            f"UNet requires layers >= 1, got layers={layers} "
            f"(problem_scale={problem_scale} - unet_bottleneck_dim={bottleneck_dim})."
        )
    vol = 2**problem_scale
    n_classes = n_categories + 1
    factor = 2 if trilinear else 1

    t = Tracer(batch, shard, group_norm_groups)
    x = (batch, n_channels, vol, vol, vol)
    skips = []  # encoder outputs, index i = level i

    # --- encoder (down_list) ---
    t.block = "enc0 (inc)"
    ch = 64
    x = t.double_conv(x, ch)
    skips.append(x)

    for i in range(1, layers):
        t.block = f"enc{i} (down{i})"
        x = t.maxpool(x)
        x = t.double_conv(x, ch * 2)
        ch *= 2
        skips.append(x)

    t.block = f"bottleneck (down{layers})"
    x = t.maxpool(x)
    x = t.double_conv(x, (ch * 2) // factor)
    ch *= 2  # mirrors unet_model.py: tracked channels ignore the factor

    # --- decoder (up_list) ---
    for i in range(layers):
        t.block = f"dec{i} (up{i + 1})"
        last = i == layers - 1
        c_in = ch  # concatenated channel count this Up was built for
        c_out = (c_in // 2) if last else (c_in // 2) // factor
        skip = skips[layers - 1 - i]
        if trilinear:
            x = t.upsample(x)
            x = t.concat(skip, x)
            x = t.double_conv(x, c_out, c_mid=c_in // 2)
        else:
            x = t.conv_transpose3d(x, c_in // 2)
            x = t.concat(skip, x)
            x = t.double_conv(x, c_out)
        ch //= 2

    t.block = "outc"
    x = t.conv3d(x, n_classes, k=1, bias=True)

    config = {
        "problem_scale": problem_scale,
        "unet_bottleneck_dim": bottleneck_dim,
        "unet_layers": layers,
        "vol_size": vol,
        "n_categories": n_categories,
        "n_classes": n_classes,
        "n_channels": n_channels,
        "local_batch_size": batch,
        "group_norm_groups": group_norm_groups,
        "trilinear": trilinear,
        "dc_num_shards": list(shard.num_shards),
        "dc_shard_dims": list(shard.shard_dims),
        "total_shards": shard.total_shards,
        "global_input_shape": [batch, n_channels, vol, vol, vol],
        "global_output_shape": list(x),
        "total_params": sum(op.params for op in t.ops),
    }
    return config, t.ops


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def fmt_shape(shape):
    if shape is None:
        return "-"
    return "×".join(str(x) for x in shape)


def fmt_shapes(shapes):
    if not shapes:
        return "-"
    return ", ".join(fmt_shape(s) for s in shapes)


def print_report(config, ops, out=sys.stdout):
    w = out.write
    sharded = config["total_shards"] > 1 or bool(config["dc_shard_dims"])
    w("ScaFFold UNet forward-pass shapes\n")
    w("=" * 78 + "\n")
    w(
        f"problem_scale={config['problem_scale']}  "
        f"vol_size={config['vol_size']}³  "
        f"unet_bottleneck_dim={config['unet_bottleneck_dim']}  "
        f"layers={config['unet_layers']}\n"
    )
    w(
        f"n_channels={config['n_channels']}  n_classes={config['n_classes']} "
        f"(n_categories={config['n_categories']}+1)  "
        f"group_norm_groups={config['group_norm_groups']}  "
        f"trilinear={config['trilinear']}  batch={config['local_batch_size']}\n"
    )
    if sharded:
        w(
            f"DistConv: dc_num_shards={config['dc_num_shards']} on tensor dims "
            f"{config['dc_shard_dims']} (NCDHW)  "
            f"-> {config['total_shards']} shard rank(s) per sample\n"
        )
        w("Shapes: global tensor, then per-rank local shard.\n")
    else:
        w("Sharding: none (single device; plain torch ops)\n")
    w(
        f"input {fmt_shape(config['global_input_shape'])} -> "
        f"output {fmt_shape(config['global_output_shape'])}   "
        f"params {config['total_params']:,}\n"
    )

    block = None
    for op in ops:
        if op.block != block:
            block = op.block
            w("\n--- " + block + " " + "-" * max(0, 70 - len(block)) + "\n")
        wcol = fmt_shape(op.weight_shape)
        if op.bias_shape:
            wcol += f" +b{fmt_shape(op.bias_shape)}"
        w(
            f"[{op.idx:3d}] {op.op:<16s} {op.detail:<22s} "
            f"W={wcol:<22s} {fmt_shapes(op.in_shapes)} -> {fmt_shape(op.out_shape)}\n"
        )
        if sharded and op.local_out_shape is not None:
            local = f"      local: {fmt_shapes(op.local_in_shapes)}"
            if op.halo is not None and any(op.halo):
                local += (
                    f"  +halo ({fmt_shape(op.halo)}) -> "
                    f"conv-in {fmt_shape(op.local_op_in_shape)}"
                )
            local += f" -> {fmt_shape(op.local_out_shape)}"
            w(local + "\n")
        for note in op.notes:
            w(f"      note: {note}\n")

    # summary
    w("\n" + "=" * 78 + "\n")
    per_block = {}
    for op in ops:
        per_block[op.block] = per_block.get(op.block, 0) + op.params
    w("Parameters by block:\n")
    for b, p in per_block.items():
        w(f"  {b:<24s} {p:>14,}\n")
    w(f"  {'TOTAL':<24s} {config['total_params']:>14,}\n")
    if sharded:
        halo_ops = [op for op in ops if op.halo is not None and any(op.halo)]
        w(
            f"\nHalo exchanges per forward pass: {len(halo_ops)} convolutions "
            f"× {len(config['dc_shard_dims'])} sharded dim(s)\n"
        )


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "problem_scale",
        type=int,
        help="Problem scale; the global volume is (2**problem_scale)^3.",
    )
    p.add_argument(
        "--unet-bottleneck-dim",
        type=int,
        default=3,
        help="Power of 2 of the UNet bottleneck dimension; layers = scale - this.",
    )
    p.add_argument("--n-categories", type=int, default=5, help="n_classes = this + 1.")
    p.add_argument("--n-channels", type=int, default=3, help="Input channels.")
    p.add_argument(
        "--local-batch-size", type=int, default=1, help="Per-DDP-replica batch size."
    )
    p.add_argument("--group-norm-groups", type=int, default=8, help="GroupNorm groups.")
    p.add_argument(
        "--trilinear",
        action="store_true",
        help="Use trilinear Upsample instead of ConvTranspose3d "
        "(ScaFFold's worker always uses False).",
    )
    p.add_argument(
        "--dc-num-shards",
        type=int,
        nargs="+",
        default=None,
        metavar="S",
        help="DistConv shards per sharded dim, e.g. 2 2 2. Default: no sharding.",
    )
    p.add_argument(
        "--dc-shard-dims",
        type=int,
        nargs="+",
        default=None,
        metavar="D",
        help="Tensor dims (NCDHW; spatial dims are 2 3 4) to shard. "
        "Default 2 3 4 when --dc-num-shards is given.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = p.parse_args(argv)

    try:
        if args.dc_num_shards is None and args.dc_shard_dims is None:
            shard = NO_SHARDING
        else:
            num_shards = args.dc_num_shards or [1, 1, 1]
            shard_dims = args.dc_shard_dims or [2, 3, 4]
            shard = ShardSpec(tuple(num_shards), tuple(shard_dims))
        config, ops = trace_unet(
            args.problem_scale,
            bottleneck_dim=args.unet_bottleneck_dim,
            n_categories=args.n_categories,
            n_channels=args.n_channels,
            batch=args.local_batch_size,
            group_norm_groups=args.group_norm_groups,
            trilinear=args.trilinear,
            shard=shard,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        json.dump(
            {"config": config, "ops": [op.to_dict() for op in ops]},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        print_report(config, ops)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # Piped to head/grep etc.; exit quietly.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)
