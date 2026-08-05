# SPDX-License-Identifier: (Apache-2.0)
"""Capture MIOpen's time for every problem in the corpus, once.

Every performance claim we make later is a comparison, and a comparison needs a
control taken on the same machine in the same session.  Re-measuring MIOpen
alongside each experiment is both slow and fragile -- MIOpen's tuning database
warms up over a run, so the "same" baseline drifts depending on what ran before
it.  Capturing it once, deliberately warmed, and storing it makes every later
number reproducible and makes database drift visible instead of silent.

Reproducing what ScaFFold actually runs
=======================================

Getting either of the following wrong makes MIOpen solve a *different problem*,
or solve this one *differently*, and then quietly reports a number that is not
the control.  The first version of this file got both wrong and overstated
MIOpen's cost by up to 12x, which would have turned into a fabricated speedup
for every kernel measured against it.  So they are enforced here, recorded in
the output header, and regression-tested in ``tests/test_infra.py``.

1.  ``torch.backends.cudnn.benchmark = True``.  ScaFFold sets this at startup
    unless ``more_determinism`` is on (``ScaFFold/worker.py:171``), and so does
    the profiling harness the reference numbers come from.  On ROCm the flag
    decides whether PyTorch asks MIOpen for an exhaustive *find* or lets it
    answer from its AI heuristic.  The two answers are not close.  For
    ``conv 64->64 k3 @ 128^3``, forward, measured with
    ``MIOPEN_ENABLE_LOGGING=1``::

        benchmark=False  findMode DYNAMIC_HYBRID(5), no search
            DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256,64,64,32,
                Default,16,16,2,2,2,1,2,1,1,1>              12.235 ms
        benchmark=True   findMode NORMAL(1), GenericSearch over 23 configs
            DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256,128,64,32,
                Default,32,32,2,1,8,8,8,1,1,1>               1.746 ms

    Same solver (``ConvHipImplicitGemm3DGroupFwdXdlops``), same device op --
    only the tuning config differs.  The heuristic picks 16x16 MFMA tiles with
    2-element global loads; the search picks 32x32 tiles with 8-element loads.
    That is the whole 7x.  It is not a naive fallback, which is why it does not
    look like one in a profile.

2.  The shape -- and there are **three** of it.  Upstream DistConv concatenates
    a ``k // 2`` halo onto every axis it manages and then sets that axis's
    padding to zero, so under the MIOpen rung ScaFFold's
    ``conv 64->64 k3 @ 128^3`` reaches MIOpen as a *130^3 unpadded* problem.
    MIOpen's find-db key includes the padding, so these are separate problems
    with separate tuning::

        padded 128^3 pad 1, benchmark=True     bwd-data  3.426 ms
        halo'd 130^3 pad 0, benchmark=True     bwd-data  3.324 ms   <- DistConv
        profiled ScaFFold (config A)           bwd-data  3.483 ms

    ``--shape halo`` (the default) measures the form **DistConv** issues, which
    is the form the profiled numbers this module cross-checks against were
    measured in -- so it stays the default and the join stays valid.  It is
    *not* the shape ScaFFold's own Triton rung runs: that adapter exchanges a
    halo only on genuinely split axes, so the convolution it issues is padded on
    H and W at every configuration and on all three axes at one GPU.
    ``--shape production`` measures **that** form, which is the one to baseline
    MIOpen in if the comparison is against a Triton kernel; ``--shape unhaloed``
    measures the logical statement; ``--shape all`` measures each distinct one.
    Every record says which it was, so a baseline cell can never be silently
    compared against the wrong profile cell.

Two more must be in the *environment* before the process starts, because MIOpen
reads them when it builds its handle and this module cannot set them for you:

* ``PYTORCH_MIOPEN_SUGGEST_NHWC=1`` -- ScaFFold's production setting.  Without
  it ``channels_last_3d`` is inert on ROCm and MIOpen is handed NCDHW, which is
  a different problem again.  Refused rather than warned about, below.
* ``MIOPEN_USER_DB_PATH`` / ``MIOPEN_CUSTOM_CACHE_DIR`` pointing somewhere
  persistent, so a find survives the process and ``--resume`` does not search
  from scratch.  Warned about; both are recorded in the output header so a
  baseline taken against a cold database is identifiable after the fact.

Cross-check
===========

``--cross-check`` joins each measured cell against the ``measured`` entries the
corpus carries from the profiled runs and reports the ratio, so a harness that
has drifted out of agreement says so rather than being believed.

Results stream to the output file as they are produced, and ``--resume`` skips
what is already there.  That matters because two things in this corpus do not
merely run slowly: the scale-8 backward-weight at ``128->64 @ 128x256x256``
takes 45 s per call, and unsharded scale 8 trips an assertion inside MIOpen that
can take the process down with it.  Losing 40 minutes of measurements to the
last problem in the list is avoidable, so it is avoided.

Usage::

    python -m triton_conv3d.bench.baseline --out baseline.json
    python -m triton_conv3d.bench.baseline --out baseline.json --resume \
        --include-edge --shape both --cross-check
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback

import torch

from ..shapes import DIRECTIONS, ConvProblem, Direction, edge_cases, scaffold_corpus
from .harness import format_table, interleaved

_MEMORY_FORMAT = torch.channels_last_3d

#: ScaFFold's default (``worker.py:171``), and the profiling harness's
#: (``prof_bench.py:125``).  See the module docstring: with this off MIOpen
#: answers from its heuristic instead of searching, and the baseline is wrong
#: by up to 12x.  Module-level so that importing this module is enough to put a
#: process in the configuration the recorded numbers were taken in.
REQUIRE_CUDNN_BENCHMARK = True
torch.backends.cudnn.benchmark = REQUIRE_CUDNN_BENCHMARK


def _key(problem: ConvProblem, direction: Direction) -> str:
    return f"{problem.label}|{problem.dtype}|{direction}"


def _build(problem: ConvProblem, device: str, dtype: torch.dtype):
    x = torch.randn(problem.input_shape, device=device, dtype=torch.float32).to(dtype)
    w = torch.randn(problem.weight_shape, device=device, dtype=torch.float32).to(dtype)
    x = x.contiguous(memory_format=_MEMORY_FORMAT).requires_grad_(True)
    w = w.contiguous(memory_format=_MEMORY_FORMAT).requires_grad_(True)
    b = (
        torch.randn(problem.cout, device=device, dtype=torch.float32).to(dtype)
        if problem.bias
        else None
    )
    return x, w, b


def _callable(problem: ConvProblem, direction: Direction, device: str, dtype):
    """A zero-argument closure that runs exactly the one direction, plus its shapes.

    The backward directions are isolated with ``torch.autograd.grad`` on a
    pre-computed forward output so that the forward is not folded into the
    measurement, and ``retain_graph`` keeps the same graph across iterations.
    """
    import torch.nn.functional as F

    x, w, b = _build(problem, device, dtype)
    op = F.conv_transpose3d if problem.transposed else F.conv3d
    fwd = lambda: op(x, w, b, stride=problem.stride, padding=problem.padding)

    if direction == "fwd":
        with torch.no_grad():
            return fwd, (x, w)

    y = fwd()
    gy = torch.randn_like(y)
    inputs = (x,) if direction == "bwd-data" else (w,)
    fn = lambda: torch.autograd.grad(y, inputs, gy, retain_graph=True)
    return fn, (x, w, y, gy)


def measure_one(
    problem: ConvProblem,
    direction: Direction,
    *,
    device: str = "cuda",
    budget_s: float = 10.0,
    target_rel: float = 0.02,
    max_call_ms: float = 60_000.0,
    shape_mode: str = "halo",
) -> dict:
    """One (problem, direction) cell.  Never raises; failures are data too."""
    if not torch.backends.cudnn.benchmark:
        raise RuntimeError(
            "cudnn.benchmark is off; MIOpen will answer from its heuristic "
            "instead of searching and the result is not a baseline"
        )
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32,
             "fp16": torch.float16}[problem.dtype]
    record: dict = {
        "problem": problem.label,
        "direction": direction,
        "dtype": problem.dtype,
        # Which of the three forms of this convolution was measured, and enough
        # of the descriptor to tell them apart without consulting the corpus.
        "shape_mode": shape_mode,
        "qualified_problem": problem.qualified_label,
        "padding": list(problem.padding),
        "input_shape": list(problem.input_shape),
        "weight_shape": list(problem.weight_shape),
        "output_shape": list(problem.output_shape),
        "flops": problem.flops(direction),
        "bytes": problem.bytes(direction),
        "arithmetic_intensity": problem.arithmetic_intensity(direction),
        "roofline_tflops": problem.roofline_flops(direction) / 1e12,
        "needs_int64": problem.needs_int64,
    }
    tensors = None
    try:
        torch.cuda.empty_cache()
        fn, tensors = _callable(problem, direction, device, dtype)

        # One untimed call decides whether this cell is measurable at all: with
        # cudnn.benchmark on, MIOpen's *find* runs on the first invocation --
        # it launches every candidate config -- and would otherwise be the whole
        # measurement.  Everything after that is sized by the harness, which
        # re-derives the same per-call time and additionally picks the round
        # count from the precision it has reached.
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        rough_ms = (time.perf_counter() - t0) * 1e3

        if rough_ms > max_call_ms:
            record.update(ms=rough_ms, best_ms=rough_ms, iters=1, rounds=1,
                          spread=0.0, rel_ci=float("inf"), stop="single-call",
                          note="single call; exceeds max_call_ms")
        else:
            meas = interleaved({"miopen": fn}, warmup=None, iters=None,
                               rounds=None, budget_s=budget_s,
                               target_rel=target_rel)["miopen"]
            # Both statistics, because they answer different questions.  The
            # median is the control -- it is what a step actually costs on a
            # shared node.  The best round is the diagnostic: this node has
            # other tenants, and a neighbour can inflate every round at once,
            # so "did MIOpen find a good kernel" has to be asked of the best
            # round or it gets a flaky answer.
            #
            # ``spread`` is kept because every stored baseline has it, but read
            # ``rel_ci`` instead: ``spread`` is a *range* and its expectation
            # grows with ``rounds``, which is now chosen per cell, so two cells'
            # spreads are no longer comparable to each other at all.
            record.update(ms=meas.median, best_ms=meas.best, iters=meas.iters,
                          rounds=len(meas.rounds), spread=meas.spread,
                          rel_ci=meas.rel_half_width, stop=meas.stop,
                          group=meas.group, tax_frac=meas.tax_frac,
                          measure_seconds=meas.seconds)
        record["tflops"] = record["flops"] / (record["ms"] * 1e-3) / 1e12
        record["pct_roofline"] = 100 * record["tflops"] / record["roofline_tflops"]
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-1500:]
    finally:
        del tensors
        torch.cuda.empty_cache()
    return record


# ---------------------------------------------------------------------------
# Cross-check against the profiled runs
# ---------------------------------------------------------------------------


def cross_check(records: list[dict], problems: list[ConvProblem]) -> list[dict]:
    """Join measured cells onto the profiled ScaFFold numbers they control for.

    Only the halo'd cells are joined, and the reason is narrower than it used to
    be stated: the profile these numbers control for was taken with DistConv on
    the path, so the calls it timed *were* the halo'd form.  The other two forms
    have no profiled counterpart to be compared against -- not because ScaFFold
    never runs them (it runs the production form at every site, every step) but
    because nobody has profiled a step in them.  Joining a production-form cell
    onto a DistConv-form profile figure is the bug this whole module is a
    response to.  The profiled figure used is the *cheapest* of the per-config
    measurements, because a profiled call can be slowed by contention with the
    rest of the step but cannot be sped up by it.
    """
    by_key = {}
    for p in problems:
        halo = p.halo_variant
        for d in DIRECTIONS:
            hits = p.measured_for(d)
            if hits:
                by_key[(halo.label, d)] = hits[-1]
    rows = []
    for r in records:
        if r.get("shape_mode") != "halo" or "ms" not in r:
            continue
        hit = by_key.get((r["problem"], r["direction"]))
        if hit is None:
            continue
        rows.append({
            "problem": r["problem"],
            "direction": r["direction"],
            "isolated_ms": r["ms"],
            "profiled_ms": hit["ms_per_call"],
            "ratio": r["ms"] / hit["ms_per_call"],
            "config": hit["config"],
            "profiled_solvers": hit.get("solvers", []),
        })
    return rows


def format_cross_check(rows: list[dict], tol: float = 0.25) -> str:
    table = format_table(
        [
            [
                r["problem"], r["direction"], f"{r['isolated_ms']:.4f}",
                f"{r['profiled_ms']:.4f}", f"{r['ratio']:.2f}x",
                "ok" if abs(r["ratio"] - 1) <= tol else "MISMATCH",
            ]
            for r in rows
        ],
        ["problem", "direction", "isolated ms", "profiled ms", "ratio", ""],
        aligns="llrrrl",
    )
    bad = [r for r in rows if abs(r["ratio"] - 1) > tol]
    return table + f"\n\n{len(rows) - len(bad)}/{len(rows)} cells within {tol:.0%}"


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="baseline.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already present in --out")
    ap.add_argument("--top", type=int, default=0,
                    help="only the N hottest corpus problems (0 = all)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N corpus problems (they are ordered by "
                         "cost, and the most expensive one takes 45 s per call)")
    ap.add_argument("--include-edge", action="store_true",
                    help="also measure the synthetic edge cases")
    ap.add_argument("--shape",
                    choices=("halo", "production", "unhaloed", "both", "all"),
                    default="halo",
                    help="halo: the form upstream DistConv issues (default, and "
                         "the form the profiled numbers measured, so the only "
                         "one --cross-check can join); production: the form "
                         "ScaFFold's own Triton adapter issues, padded on every "
                         "unsplit axis -- the right MIOpen baseline for a "
                         "Triton comparison; unhaloed: the logical statement; "
                         "both: halo+unhaloed, as before; all: every distinct "
                         "form")
    ap.add_argument("--max-call-ms", type=float, default=60_000.0,
                    help="above this, record a single call rather than a sweep")
    ap.add_argument("--budget", type=float, default=10.0,
                    help="wall-clock seconds per cell (default 10)")
    ap.add_argument("--precision", type=float, default=0.02,
                    help="target relative 95%% half-width per cell")
    ap.add_argument("--skip-slow", action="store_true",
                    help="skip cells a previous run recorded as slower than "
                         "--max-call-ms; useful for a quick re-capture")
    ap.add_argument("--cross-check", action="store_true",
                    help="join the halo'd cells onto the profiled numbers")
    ap.add_argument("--tolerance", type=float, default=0.25,
                    help="cross-check band, as a fraction of the profiled time")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    if os.environ.get("PYTORCH_MIOPEN_SUGGEST_NHWC") != "1":
        raise SystemExit(
            "PYTORCH_MIOPEN_SUGGEST_NHWC=1 is not set: channels_last_3d is inert "
            "on ROCm without it and MIOpen would see NCDHW, which is not what "
            "ScaFFold runs"
        )
    if not os.environ.get("MIOPEN_USER_DB_PATH"):
        print("WARNING: MIOPEN_USER_DB_PATH is unset -- every cell re-runs the "
              "find from scratch and nothing is reusable afterwards",
              file=sys.stderr)

    problems = list(scaffold_corpus())[args.skip:]
    if args.top:
        problems = problems[: args.top]
    corpus_problems = list(problems)
    if args.include_edge:
        problems += list(edge_cases())

    modes = {"both": ("halo", "unhaloed"),
             "all": ("halo", "production", "unhaloed")}.get(
        args.shape, (args.shape,))
    _forms = {"halo": lambda p: p.halo_variant,
              "production": lambda p: p.production_variant,
              "unhaloed": lambda p: p}
    # A problem with nothing to halo is its own variant in all three forms, so
    # the multi-mode runs would otherwise measure the synthetic edge cases and
    # the k=1/transposed convs two or three times for nothing.  Deduplicated on
    # the *qualified* label, which carries the padding: two forms can share a
    # ``label`` and be different problems.
    variants: list[tuple[ConvProblem, str]] = []
    for p in problems:
        seen: dict[str, str] = {}
        for mode in modes:
            q = _forms[mode](p)
            if q.qualified_label in seen:
                continue
            seen[q.qualified_label] = mode
            variants.append((q, mode))

    out_path = pathlib.Path(args.out)
    done: dict[str, dict] = {}
    records: list[dict] = []
    if args.resume and out_path.exists():
        prior = json.loads(out_path.read_text())
        records = list(prior.get("records", []))
        done = {r["problem"] + "|" + r["direction"]: r for r in records}
        print(f"resuming: {len(done)} cells already measured")

    props = torch.cuda.get_device_properties(0)
    header = {
        "device": props.name,
        "torch": torch.__version__,
        "miopen_suggest_nhwc": os.environ.get("PYTORCH_MIOPEN_SUGGEST_NHWC"),
        "miopen_user_db_path": os.environ.get("MIOPEN_USER_DB_PATH"),
        "miopen_custom_cache_dir": os.environ.get("MIOPEN_CUSTOM_CACHE_DIR"),
        # The two settings that decide whether this file is a control at all.
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "shape_mode": args.shape,
        "memory_format": "channels_last_3d",
    }

    def flush() -> None:
        payload = {**header, "records": records}
        if args.cross_check:
            payload["cross_check"] = cross_check(records, corpus_problems)
        out_path.write_text(json.dumps(payload, indent=1) + "\n")

    for problem, mode in variants:
        for direction in DIRECTIONS:
            key = problem.label + "|" + direction
            if key in done:
                continue
            if args.skip_slow and done.get(key, {}).get("note"):
                continue
            rec = measure_one(problem, direction, max_call_ms=args.max_call_ms,
                              shape_mode=mode, budget_s=args.budget,
                              target_rel=args.precision)
            records.append(rec)
            flush()
            if "error" in rec:
                print(f"  {problem.label:36s} {mode:8s} {direction:11s} "
                      f"ERROR {rec['error'][:80]}")
            else:
                print(
                    f"  {problem.label:36s} {mode:8s} {direction:11s} "
                    f"{rec['ms']:10.4f} ms +-{rec.get('rel_ci', 0):5.1%} "
                    f"({rec.get('rounds', 0)}r/{rec.get('stop', '?')})  "
                    f"{rec['tflops']:7.1f} TF/s "
                    f"{rec['pct_roofline']:6.1f}% roofline"
                    + (f"  [{rec['note']}]" if rec.get("note") else "")
                )
            sys.stdout.flush()

    flush()
    ok = [r for r in records if "error" not in r]
    print(f"\n{len(ok)}/{len(records)} cells measured -> {out_path}")
    if len(ok) < len(records):
        print("failures:")
        print(format_table(
            [[r["problem"], r["direction"], r["error"][:70]]
             for r in records if "error" in r],
            ["problem", "direction", "error"],
        ))
    if args.cross_check:
        rows = cross_check(records, corpus_problems)
        print("\ncross-check against the profiled ScaFFold runs:")
        print(format_cross_check(rows, tol=args.tolerance))


if __name__ == "__main__":
    main()
