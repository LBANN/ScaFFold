#!/usr/bin/env python3
# SPDX-License-Identifier: (Apache-2.0)
"""Regenerate ``scaffold_census.json`` from the captures under ``census/``.

Each capture is one rank's record of an instrumented training run
(``conv_census.py``) at one of the four benchmark configurations.  It records
every call three ways; the census keeps one row per distinct problem *in the
form the kernel was handed* -- the autograd node's view: post-halo input
extent and post-halo padding -- with bias, stride, dilation and groups taken
from the module record of the same site.

    python triton_conv3d/tests/data/make_census.py [--check]

``--check`` reports whether the committed file would change, without writing.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "scaffold_census.json"

#: The benchmark configurations and the capture for each.  Sharded
#: configurations are identical across ranks, so rank 0 stands for the mesh.
CONFIGS = [
    ("A", "scale 7, 1 GPU, shards (1,1,1)", "census/cens_A.json"),
    ("B", "scale 8, 1 GPU, shards (1,1,1)", "census/cens_B.json"),
    ("C", "scale 8, 2 GPUs, shards (2,1,1)", "census/cens_C.json"),
    ("D", "scale 8, 4 GPUs, shards (4,1,1)", "census/cens_D.json"),
]

#: Above this many bytes in the larger activation a problem is ``large``: a
#: test or a bench sweep has to opt into allocating it.
_LARGE_BYTES = 1 << 28
_ELEM_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
_DTYPES = {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}


def _out_spatial(spatial, kernel, stride, padding, transposed):
    if transposed:
        return [
            (i - 1) * s - 2 * p + k
            for i, k, s, p in zip(spatial, kernel, stride, padding)
        ]
    return [
        (i + 2 * p - k) // s + 1 for i, k, s, p in zip(spatial, kernel, stride, padding)
    ]


def collect() -> list[dict]:
    problems: dict[tuple, dict] = {}
    for tag, desc, fname in CONFIGS:
        cap = json.loads((HERE / fname).read_text())
        mods = {v["site"]: v for v in cap["modules"].values()}
        for rec in cap["autograd_fn"].values():
            mod = mods[rec["site"]]
            in_shape = list(rec["x_shape"])
            w = list(rec["weight_shape"])
            padding = list(rec["kernel_padding"])
            stride = list(mod["stride"])
            transposed = rec["op"] == "ConvTranspose3d"
            dtype = _DTYPES[rec["dtype"]]
            key = (
                rec["op"],
                tuple(w),
                tuple(in_shape),
                tuple(padding),
                tuple(stride),
                bool(mod["bias"]),
                dtype,
            )
            entry = problems.get(key)
            if entry is None:
                cin, cout = (w[0], w[1]) if transposed else (w[1], w[0])
                out_sp = _out_spatial(in_shape[2:], w[2:5], stride, padding, transposed)
                act = max(math.prod(in_shape), in_shape[0] * cout * math.prod(out_sp))
                entry = problems[key] = {
                    "op": rec["op"],
                    "weight_shape": w,
                    "in_shape": in_shape,
                    "out_shape": [in_shape[0], cout, *out_sp],
                    "kernel": list(w[2:5]),
                    "stride": stride,
                    "padding": padding,
                    "dilation": list(mod["dilation"]),
                    "groups": int(mod["groups"]),
                    "bias": bool(mod["bias"]),
                    "dtype": dtype,
                    "memory_format": rec["memfmt"],
                    "dctensor": bool(mod["dctensor"]),
                    "large": act * _ELEM_BYTES[dtype] > _LARGE_BYTES,
                    "sites": [],
                    "configs": [],
                    "name": "",
                }
            entry["sites"].append(f"{tag}:{rec['site']}")
            if desc not in entry["configs"]:
                entry["configs"].append(desc)
    for entry in problems.values():
        entry["name"] = entry["sites"][0].replace(":", "-", 1)
    # Biggest activation first, so truncating keeps the problems that dominate
    # a step; not a cost ordering, nothing here is timed.
    return sorted(problems.values(), key=lambda e: -math.prod(e["in_shape"]))


def build() -> str:
    rows = collect()
    doc = {
        "source": "conv_census.py: an instrumented training run per configuration, "
        "three steps each at n_categories 2, wrapped entry points",
        "form": "adapter",
        "note": "The shape and padding the kernel was handed, read off a real call. "
        "Every k>1 convolution here is padded: the ScaFFold adapter exchanges a "
        "halo only on genuinely split axes.",
        "configs": [{"tag": t, "desc": d, "capture": f} for t, d, f in CONFIGS],
        "n_problems": len(rows),
        "problems": rows,
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
