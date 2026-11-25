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

"""Check precision-policy dispatch without requiring GPU compilation."""

import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.backends import registry

from ScaFFold import worker


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", [None, "inductor"])
@pytest.mark.parametrize("preserve", [False, True])
def test_inductor_precision_options_are_scoped(
    monkeypatch, device_type, backend, preserve
):
    fallback, compiled = object(), object()
    looked_up, calls = [], []

    def lookup(name):
        looked_up.append(name)
        return fallback

    def compile_graph(graph, inputs, **kwargs):
        calls.append((graph, inputs, kwargs))
        return compiled

    monkeypatch.setattr(registry, "lookup_backend", lookup)
    monkeypatch.setattr(torch._inductor, "compile", compile_graph)
    original_setting = torch._inductor.config.emulate_precision_casts
    selected = worker._compile_backend(
        backend, device_type=device_type, preserve_eager_precision=preserve
    )
    if device_type == "cpu" or preserve:
        graph, inputs = object(), []
        assert selected(graph, inputs) is compiled
        assert calls == [
            (graph, inputs, {"options": {"emulate_precision_casts": True}})
        ]
        assert looked_up == []
    else:
        assert selected is fallback
        assert looked_up == ["inductor"]
        assert calls == []
    assert torch._inductor.config.emulate_precision_casts == original_setting


@pytest.mark.parametrize("backend", [None, "inductor"])
@pytest.mark.parametrize("preserve_precision", [False, True])
def test_native_group_norm_decompositions_are_scoped(
    monkeypatch, backend, preserve_precision
):
    compile_fx = importlib.import_module("torch._inductor.compile_fx")
    decomposition = importlib.import_module("torch._inductor.decomposition")
    common = importlib.import_module("torch._inductor.codegen.common")
    forward = torch.ops.aten.native_group_norm.default
    backward = torch.ops.aten.native_group_norm_backward.default
    other = torch.ops.aten.add.Tensor
    original = {forward: object(), backward: object(), other: object()}
    before = dict(original)
    compiled, calls, tags = object(), [], []

    def compile_graph(graph, inputs, **kwargs):
        calls.append((graph, inputs, kwargs))
        tags.append(torch.compiler.config.cache_key_tag)
        return compiled

    monkeypatch.setattr(common, "init_backend_registration", lambda: None)
    monkeypatch.setattr(decomposition, "select_decomp_table", lambda: original)
    monkeypatch.setattr(compile_fx, "compile_fx", compile_graph)
    selected = worker._compile_backend(
        backend,
        device_type="cuda",
        preserve_eager_precision=preserve_precision,
        preserve_native_group_norm=True,
    )
    graph, inputs = object(), []
    with torch.compiler.config.patch(cache_key_tag="caller"):
        assert selected(graph, inputs) is compiled
        assert torch.compiler.config.cache_key_tag == "caller"
    options = {"emulate_precision_casts": True} if preserve_precision else {}
    assert len(calls) == 1
    actual_graph, actual_inputs, kwargs = calls[0]
    assert actual_graph is graph and actual_inputs is inputs
    assert set(kwargs) == {"config_patches", "decompositions", "inner_compile"}
    assert kwargs["config_patches"] == options
    assert kwargs["decompositions"] == {other: original[other]}
    assert kwargs["decompositions"] is not original
    assert kwargs["inner_compile"].func is worker._compile_native_group_norm_graph
    assert kwargs["inner_compile"].keywords == {"cache_tag": tags[0]}
    assert json.loads(tags[0]) == ["caller", "scaffold-native-group-norm-v1"]
    assert original == before


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("preserve_group_norm", [False, True])
@pytest.mark.parametrize(
    "custom_callable", [False, True], ids=["aot_eager", "callable"]
)
def test_precision_policy_preserves_other_backends(
    monkeypatch, device_type, custom_callable, preserve_group_norm
):
    def custom_backend(graph, inputs):
        return graph

    requested = custom_backend if custom_callable else "aot_eager"
    looked_up = []

    def lookup(backend):
        looked_up.append(backend)
        return custom_backend

    monkeypatch.setattr(registry, "lookup_backend", lookup)
    monkeypatch.setattr(
        torch._inductor,
        "compile",
        lambda *args, **kwargs: pytest.fail("Other backend entered Inductor"),
    )
    compile_fx = importlib.import_module("torch._inductor.compile_fx")
    monkeypatch.setattr(
        compile_fx,
        "compile_fx",
        lambda *args, **kwargs: pytest.fail("Other backend entered Inductor"),
    )
    selected = worker._compile_backend(
        requested,
        device_type=device_type,
        preserve_eager_precision=True,
        preserve_native_group_norm=preserve_group_norm,
    )
    assert selected is custom_backend
    assert looked_up == [requested]


@pytest.mark.parametrize("route", ["fullgraph", "blocks", "spatial"])
@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("compile_loss", [False, True])
@pytest.mark.parametrize("amp_enabled", [False, True])
def test_native_adapters_request_scoped_compiler_numerics(
    monkeypatch, route, device_type, compile_loss, amp_enabled
):
    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=torch.device(device_type))]),
        modules=lambda: [],
        forward=lambda value: value,
        down_list=[SimpleNamespace(forward=lambda value: value)],
        up_list=[SimpleNamespace(forward=lambda value: value)],
    )
    strategy = SimpleNamespace(
        device_mesh={"spatial": object()},
        distconv_dim_names="spatial",
        num_shards=(1, 1, 1),
    )
    selected, requested, compile_calls = object(), [], []

    def select_backend(backend, **kwargs):
        requested.append((backend, kwargs))
        return selected

    def compile_forward(forward, **kwargs):
        compile_calls.append(kwargs)
        return forward

    monkeypatch.setattr(worker, "_compile_backend", select_backend)
    monkeypatch.setattr(torch, "compile", compile_forward)
    options = dict(
        backend="inductor", compile_loss=compile_loss, amp_enabled=amp_enabled
    )
    if route == "spatial":
        worker.compile_spatial_forward(model, strategy, **options)
    else:
        worker.compile_unsharded_forward(
            model, parallel_strategy=strategy, fullgraph=route == "fullgraph", **options
        )
        assert compile_calls
        assert all(call["backend"] is selected for call in compile_calls)
    expected = {
        "device_type": device_type,
        "preserve_eager_precision": route == "spatial"
        or (compile_loss and amp_enabled),
    }
    if route == "spatial":
        expected["preserve_native_group_norm"] = device_type == "cuda"
    assert requested == [("inductor", expected)]


def test_native_group_norm_reaches_inductor_forward_and_lazy_backward(monkeypatch):
    from torch._inductor import config
    from torch._inductor.codegen.common import init_backend_registration
    from torch._inductor.decomposition import select_decomp_table

    compile_fx = importlib.import_module("torch._inductor.compile_fx")
    original_inner = compile_fx.compile_fx_inner
    observed = []

    def observe_lowered(graph, inputs, **kwargs):
        targets = {
            node.target
            for module in graph.modules()
            if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes
            if node.op == "call_function"
        }
        assert kwargs["get_decomp_fn"] is worker._native_group_norm_decompositions
        observed.append(
            (
                kwargs.get("is_backward", False),
                targets,
                config.emulate_precision_casts,
                torch.compiler.config.cache_key_tag,
            )
        )
        return original_inner(graph, inputs, **kwargs)

    monkeypatch.setattr(compile_fx, "compile_fx_inner", observe_lowered)
    init_backend_registration()
    decomposition_table = select_decomp_table()
    before = dict(decomposition_table)
    generator = torch.Generator().manual_seed(1234)
    values = (
        torch.randn(2, 8, 4, 4, 4, generator=generator).contiguous(
            memory_format=torch.channels_last_3d
        ),
        torch.randn(8, generator=generator),
        torch.randn(8, generator=generator),
    )
    reference = [value.detach().clone().requires_grad_(True) for value in values]
    candidate = [value.detach().clone().requires_grad_(True) for value in values]

    def normalization(value, weight, bias):
        return torch.nn.functional.group_norm(value, 4, weight, bias)

    # Verify the GPU policy's compiler wiring with CPU FP32 inputs. CUDA autocast
    # also promotes this GroupNorm to FP32; actual GPU parity is tested separately.
    selected = worker._compile_backend(
        "inductor",
        device_type="cuda",
        preserve_eager_precision=True,
        preserve_native_group_norm=True,
    )
    compiled = torch.compile(normalization, backend=selected, fullgraph=True)
    original_precision = config.emulate_precision_casts
    original_tag = torch.compiler.config.cache_key_tag
    torch._dynamo.reset()
    try:
        with config.patch(force_disable_caches=True, emulate_precision_casts=False):
            expected = normalization(*reference)
            with torch.compiler.config.patch(cache_key_tag="forward-caller"):
                actual = compiled(*candidate)
                assert torch.compiler.config.cache_key_tag == "forward-caller"
            assert config.emulate_precision_casts is False
            assert observed and not any(backward for backward, _, _, _ in observed)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            expected.square().mean().backward()
            # Lazy backward must retain the options after the forward callback's
            # precision context has returned to the ambient False setting.
            with torch.compiler.config.patch(cache_key_tag="backward-caller"):
                actual.square().mean().backward()
                assert torch.compiler.config.cache_key_tag == "backward-caller"
            assert config.emulate_precision_casts is False
            for left, right in zip(candidate, reference, strict=True):
                assert left.grad is not None and right.grad is not None
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-5)
            assert any(
                not backward and torch.ops.aten.native_group_norm.default in targets
                for backward, targets, _, _ in observed
            )
            assert any(
                backward
                and torch.ops.aten.native_group_norm_backward.default in targets
                for backward, targets, _, _ in observed
            )
            assert all(precision is True for _, _, precision, _ in observed)
            assert all(
                json.loads(tag) == ["forward-caller", "scaffold-native-group-norm-v1"]
                for _, _, _, tag in observed
            )
            assert decomposition_table == before
            assert select_decomp_table() == before
    finally:
        torch._dynamo.reset()
    assert config.emulate_precision_casts == original_precision
    assert torch.compiler.config.cache_key_tag == original_tag


def test_native_group_norm_cache_is_scoped_and_reused():
    from torch._dynamo.utils import counters
    from torch._functorch import config as aot_config
    from torch._inductor import config
    from torch._inductor.codegen.common import init_backend_registration
    from torch._inductor.decomposition import select_decomp_table
    from torch._inductor.utils import fresh_cache

    init_backend_registration()
    table = select_decomp_table()
    table_before = dict(table)
    original_tag = torch.compiler.config.cache_key_tag
    original_precision, previous_threads = (
        config.emulate_precision_casts,
        torch.get_num_threads(),
    )
    generator = torch.Generator().manual_seed(1234)
    values = (
        torch.randn(2, 8, 4, 4, 4, generator=generator).contiguous(
            memory_format=torch.channels_last_3d
        ),
        torch.randn(8, generator=generator),
        torch.randn(8, generator=generator),
    )

    def normalization(value, weight, bias):
        return torch.nn.functional.group_norm(value, 4, weight, bias)

    torch.set_num_threads(1)
    try:
        with (
            fresh_cache(),
            config.patch(
                force_disable_caches=False,
                fx_graph_cache=True,
                fx_graph_remote_cache=False,
                emulate_precision_casts=False,
            ),
            aot_config.patch(
                enable_autograd_cache=True, enable_remote_autograd_cache=False
            ),
        ):
            for iteration, preserve_native in enumerate((False, True, True, False)):
                torch._dynamo.reset()
                before = {
                    kind: counters[kind].copy() for kind in ("aot_autograd", "inductor")
                }
                reference = [
                    value.detach().clone().requires_grad_(True) for value in values
                ]
                candidate = [
                    value.detach().clone().requires_grad_(True) for value in values
                ]
                backend = worker._compile_backend(
                    "inductor",
                    device_type="cuda",
                    preserve_eager_precision=True,
                    preserve_native_group_norm=preserve_native,
                )
                compiled = torch.compile(normalization, backend=backend, fullgraph=True)
                with torch.compiler.config.patch(
                    cache_key_tag="native-gn-cache-test-caller"
                ):
                    expected, actual = normalization(*reference), compiled(*candidate)
                    assert (
                        torch.compiler.config.cache_key_tag
                        == "native-gn-cache-test-caller"
                    )
                    assert config.emulate_precision_casts is False
                with torch.compiler.config.patch(
                    cache_key_tag="different-backward-caller"
                ):
                    expected.square().mean().backward()
                    actual.square().mean().backward()
                    assert (
                        torch.compiler.config.cache_key_tag
                        == "different-backward-caller"
                    )
                    assert config.emulate_precision_casts is False
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                for left, right in zip(candidate, reference, strict=True):
                    assert left.grad is not None and right.grad is not None
                    torch.testing.assert_close(
                        left.grad, right.grad, rtol=1e-4, atol=1e-5
                    )
                changes = {kind: counters[kind] - before[kind] for kind in before}
                assert not any(
                    count
                    for counts in changes.values()
                    for name, count in counts.items()
                    if "bypass" in name
                ), changes
                cold = iteration < 2
                assert changes["aot_autograd"]["autograd_cache_miss"] == int(cold), (
                    changes
                )
                assert changes["aot_autograd"]["autograd_cache_saved"] == int(cold), (
                    changes
                )
                assert changes["aot_autograd"]["autograd_cache_hit"] == int(not cold), (
                    changes
                )
                assert changes["inductor"]["fxgraph_cache_miss"] == 2 * int(cold), (
                    changes
                )
                assert changes["inductor"]["fxgraph_cache_hit"] == 2 * int(not cold), (
                    changes
                )
                assert table == table_before and select_decomp_table() == table_before
                assert torch.compiler.config.cache_key_tag == original_tag
    finally:
        torch._dynamo.reset()
        torch.set_num_threads(previous_threads)
    assert config.emulate_precision_casts == original_precision
