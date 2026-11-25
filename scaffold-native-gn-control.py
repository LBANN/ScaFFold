"""Diagnostic only: retain native ATen GroupNorm inside Inductor graphs.

Run from the ScaFFold checkout, with its Python interpreter:
  python /path/scaffold-native-gn-control.py --smoke
  MPI4PY_RC_INITIALIZE=0 SCAFFOLD_TEST_DEVICE=cuda python \
      /path/scaffold-native-gn-control.py --log-dir /path/new-diagnostic-logs

The parent launches two ranks through the existing harness. This does not edit
ScaFFold or weaken its parity assertions. Private PyTorch 2.15 APIs are used.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")

import torch

LOWERED_TARGETS = []
GLOBAL_DECOMPOSITIONS = []
NATIVE_NAMES = (
    "aten.native_group_norm.default",
    "aten.native_group_norm_backward.default",
)


def native_gn_backend(graph, inputs):
    from torch._inductor.codegen.common import init_backend_registration
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner
    from torch._inductor.decomposition import select_decomp_table

    init_backend_registration()
    original = select_decomp_table()
    GLOBAL_DECOMPOSITIONS.append((original, dict(original)))
    decompositions = dict(original)
    for operation in (
        torch.ops.aten.native_group_norm.default,
        torch.ops.aten.native_group_norm_backward.default,
    ):
        decompositions.pop(operation, None)

    def observe_lowered(lowered, examples, **kwargs):
        LOWERED_TARGETS.extend(
            str(node.target)
            for module in lowered.modules()
            if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes
            if node.op == "call_function"
        )
        return compile_fx_inner(lowered, examples, **kwargs)

    return compile_fx(
        graph,
        inputs,
        inner_compile=observe_lowered,
        decompositions=decompositions,
        config_patches={
            "emulate_precision_casts": True,
            # Ensure the observer sees the actual forward/backward FX graphs.
            "force_disable_caches": True,
        },
    )


def verify_control(*, require_native=True):
    counts = {name: LOWERED_TARGETS.count(name) for name in NATIVE_NAMES}
    unchanged = all(table == before for table, before in GLOBAL_DECOMPOSITIONS)
    print(
        json.dumps(
            {
                "native_gn_control": counts,
                "native_nodes_required": require_native,
                "global_tables_unchanged": unchanged,
            }
        ),
        flush=True,
    )
    if require_native:
        assert all(counts.values()), f"Native forward/backward not retained: {counts}"
    assert unchanged, "The global decomposition table changed"


def smoke():
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    values = (
        torch.randn(2, 8, 4, 4, 4).contiguous(memory_format=torch.channels_last_3d),
        torch.randn(8),
        torch.randn(8),
    )
    reference = [value.detach().clone().requires_grad_(True) for value in values]
    candidate = [value.detach().clone().requires_grad_(True) for value in values]

    def normalization(value, weight, bias):
        return torch.nn.functional.group_norm(value, 4, weight, bias)

    compiled = torch.compile(normalization, backend=native_gn_backend, fullgraph=True)
    expected = normalization(*reference)
    actual = compiled(*candidate)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for left, right in zip(candidate, reference, strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-5)
    verify_control()
    print("CPU FP32 native GroupNorm forward/backward smoke PASSED", flush=True)


def run_rank():
    from tests.helpers.compile_runtime import compile_device_type
    from tests.helpers.rank_scripts import compile_spatial_2rank as scenario

    original = scenario._compile_backend

    def select_backend(backend=None, **kwargs):
        if backend is None or backend == "inductor":
            return native_gn_backend
        return original(backend, **kwargs)

    scenario._compile_backend = select_backend
    require_native = compile_device_type() == "cuda"
    try:
        try:
            scenario.main()
        except BaseException:
            try:
                verify_control(require_native=require_native)
            except AssertionError as error:
                print(
                    f"Additional control verification failure: {error}", file=sys.stderr
                )
            raise
        else:
            # ScaFFold already uses its separate native opaque GN implementation
            # on CPU. The standalone FP32 smoke exercises this control.
            verify_control(require_native=require_native)
    finally:
        scenario._compile_backend = original


def summarize(rank, metrics, **extra):
    selected = {
        name: {key: value[key] for key in ("max_absolute", "relative_l2")}
        for name, value in metrics.items()
        if name.startswith(("output ", "input ", "all parameter gradients "))
    }
    worst = max(metrics, key=lambda key: metrics[key]["relative_l2"], default=None)
    print(
        json.dumps(
            {
                "rank": rank,
                **extra,
                "metrics": selected,
                "worst": {worst: metrics[worst]} if worst else {},
            }
        ),
        flush=True,
    )


def run_parent(log_dir):
    from tests.helpers.compile_runtime import run_compile_ranks

    log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Launching two ranks with compiler caches bypassed. Logs: {log_dir}",
        flush=True,
    )
    rc, out, err = run_compile_ranks(
        __file__,
        n=2,
        timeout=900,
        env={
            "PYTHON_EXEC": sys.executable,
            "SCAFFOLD_GN_CONTROL_CHILD": "1",
            "SCAFFOLD_TEST_COMPILE_BACKEND": "inductor",
            "SCAFFOLD_TEST_BF16": "1",
            "SCAFFOLD_TEST_TRANSITIONS": "0",
            "SCAFFOLD_TEST_EVAL_FIRST": "0",
            "SCAFFOLD_TEST_NUM_SHARDS": "[2,1,1]",
            "SCAFFOLD_TEST_SHARD_DIMS": "[2,3,4]",
            "SCAFFOLD_TEST_LAYERS": "2",
            "SCAFFOLD_TEST_LOCAL_SHAPES": "[[1,8,8,12],[2,8,12,8],[2,8,12,8],[1,8,8,12]]",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    (log_dir / "ranks.stdout").write_text(out)
    (log_dir / "ranks.stderr").write_text(err)
    for line in out.splitlines():
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                print("Unparsed rank stdout remains in ranks.stdout", file=sys.stderr)
                continue
            if "metrics" in result:
                summarize(
                    result["rank"],
                    result["metrics"],
                    pending_collectives=result.get("pending_collectives"),
                )
            elif "native_gn_control" in result or "compile_validation" in result:
                print(line, flush=True)
    if rc:
        for rank in range(2):
            text = "\n".join(
                match.group(1)
                for line in err.splitlines()
                if (match := re.match(rf"\[rank{rank}\]: (.*)", line))
            )
            start = text.find('{\n  "errors"')
            if start >= 0:
                try:
                    result = json.JSONDecoder().raw_decode(text[start:])[0]
                    summarize(rank, result["local_metrics"], errors=result["errors"])
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    print(f"Rank {rank} summary unavailable: {error}", file=sys.stderr)
        print(err[-4000:], file=sys.stderr)
    print(json.dumps({"returncode": rc, "log_directory": str(log_dir)}), flush=True)
    return rc if 0 <= rc <= 255 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--source", default=os.environ.get("SCAFFOLD_GN_CONTROL_SOURCE", os.getcwd())
    )
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return 0
    source = Path(args.source).resolve()
    if not (source / "tests/helpers/rank_scripts/compile_spatial_2rank.py").is_file():
        raise RuntimeError(f"Not a ScaFFold checkout: {source}")
    sys.path.insert(0, str(source))
    os.environ["SCAFFOLD_GN_CONTROL_SOURCE"] = str(source)
    if os.environ.get("SCAFFOLD_GN_CONTROL_CHILD") == "1":
        run_rank()
        return 0
    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="scaffold-native-gn-"))
    return run_parent(log_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
