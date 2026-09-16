#!/usr/bin/env python3
# SPDX-License-Identifier: (Apache-2.0)
"""Regenerate ``scaffold_corpus.json``: the distinct convolutions ScaFFold runs.

The shapes come from ``model-analysis/unet_shapes.py`` (pure stdlib) traced at
the three profiled configurations; the ``measured`` column joins
``profile_points.json``, the MIOpen profile record of the same configurations,
which is what orders the corpus by cost.

    python triton_conv3d/tests/data/make_corpus.py [--check]

``--check`` reports whether the committed file would change, without writing.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "model-analysis"))

from unet_shapes import ShardSpec, trace_unet  # noqa: E402

OUT = HERE / "scaffold_corpus.json"
POINTS = HERE / "profile_points.json"

#: The profiled configurations: tag, scale, bottleneck dim, shards on
#: (D, H, W), description.  All three are four-layer models: the scale-8
#: profiles were taken at ``unet_bottleneck_dim = 4``, not the shipped 3.
CONFIGS = [
    ("A", 7, 3, (1, 1, 1), "scale 7, 1 GPU, shards (1,1,1)"),
    ("B", 8, 4, (2, 1, 1), "scale 8, 2 GPUs, shards (2,1,1)"),
    ("C", 8, 4, (4, 1, 1), "scale 8, 4 GPUs, shards (4,1,1)"),
]
CONV_OPS = {"Conv3d", "ConvTranspose3d"}


def _parse_detail(detail: str) -> dict:
    """``k=3^3 s=1 p=1 (no bias)`` -> the kernel, stride, padding and bias."""
    k = re.search(r"k=(\d+)", detail)
    s = re.search(r"s=(\d+)", detail)
    p = re.search(r"p=(\d+)", detail)
    return {
        "k": int(k.group(1)) if k else None,
        "stride": int(s.group(1)) if s else 1,
        "padding": int(p.group(1)) if p else 0,
        "bias": "no bias" not in detail,
    }


def _shard_halo(num_shards, halo_dhw) -> list:
    """The halo on the dims actually split: what the ScaFFold adapter exchanges."""
    return [h if n > 1 else 0 for n, h in zip(num_shards, halo_dhw)]


def collect_shapes() -> dict:
    problems: dict[tuple, dict] = {}
    for tag, scale, bottleneck, num_shards, desc in CONFIGS:
        shard = ShardSpec(num_shards=list(num_shards), shard_dims=[2, 3, 4])
        _, ops = trace_unet(scale, bottleneck_dim=bottleneck, shard=shard)
        for op in (o.to_dict() for o in ops):
            if op["op"] not in CONV_OPS:
                continue
            meta = _parse_detail(op["detail"])
            local_in = op["local_in_shapes"][0]
            shard_halo = _shard_halo(num_shards, op["halo_dhw"])
            # Everything a kernel specializes on; two sites with one key are
            # one tuning target.
            key = (
                op["op"],
                tuple(op["weight_shape"]),
                tuple(local_in),
                tuple(op["local_out_shape"]),
                meta["stride"],
                meta["padding"],
                meta["bias"],
            )
            entry = problems.setdefault(
                key,
                {
                    "op": op["op"],
                    "weight_shape": op["weight_shape"],
                    "in_shape": local_in,
                    "out_shape": op["local_out_shape"],
                    "halo_in_shape": op["local_op_in_shape"],
                    "halo_dhw": op["halo_dhw"],
                    "shard_halo_dhw": shard_halo,
                    **meta,
                    "sites": [],
                    "configs": [],
                },
            )
            if entry["shard_halo_dhw"] != shard_halo:
                raise SystemExit(
                    f"{op['op']} {op['weight_shape']} occurs at two configurations "
                    "with different adapter halos; one problem cannot have two forms"
                )
            entry["sites"].append(f"{tag}:{op['block']}#{op['idx']}")
            if desc not in entry["configs"]:
                entry["configs"].append(desc)
    return problems


def join_measurements(problems: dict) -> None:
    """Attach the profiled ms and roofline efficiency per direction."""
    record = json.loads(POINTS.read_text())["configs"]
    for tag, _, _, _, _ in CONFIGS:
        roofs = record[tag]["roofs"]
        hbm = roofs["hbm_TB_s"] * 1e12
        index = collections.defaultdict(list)
        for p in record[tag]["points"]:
            peak = (
                roofs["bf16_TFLOP_s"] if p["act_bytes"] == 2 else roofs["fp32_TFLOP_s"]
            ) * 1e12
            roof = min(peak, p["flops"] / p["bytes"] * hbm)
            # ``flops`` and ``ms`` are both per step, so this is per-call
            # efficiency even for a problem at two sites.
            achieved = p["flops"] / (p["ms"] * 1e-3)
            index[(tuple(p["shape_w"]), tuple(p["shape_out"]))].append(
                {
                    "config": tag,
                    "direction": p["direction"],
                    "ms_per_step": round(p["ms"], 4),
                    "ms_per_call": round(p["ms_per_call"], 4),
                    "calls": p["calls"],
                    "pct_roofline": round(100 * achieved / roof, 1),
                    "solvers": p["solvers"],
                }
            )
        for entry in problems.values():
            hits = index.get((tuple(entry["weight_shape"]), tuple(entry["out_shape"])))
            entry.setdefault("measured", []).extend(hits or [])


def build() -> str:
    problems = collect_shapes()
    join_measurements(problems)
    entries = sorted(
        problems.values(),
        key=lambda e: -sum(m["ms_per_step"] for m in e["measured"]),
    )
    doc = {
        "source": "model-analysis/unet_shapes.py",
        "configs": [{"tag": t, "desc": d} for t, _, _, _, d in CONFIGS],
        "n_problems": len(entries),
        "problems": entries,
    }
    return json.dumps(doc, indent=1) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = build()
    n = text.count('"op":')
    if args.check:
        same = OUT.exists() and OUT.read_text() == text
        print(f"{n} problems; {'unchanged' if same else 'WOULD CHANGE'}")
        return 0 if same else 1
    OUT.write_text(text)
    print(f"{n} problems -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
