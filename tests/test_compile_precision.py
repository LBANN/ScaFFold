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


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "custom_callable", [False, True], ids=["aot_eager", "callable"]
)
def test_precision_policy_preserves_other_backends(
    monkeypatch, device_type, custom_callable
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
    selected = worker._compile_backend(
        requested, device_type=device_type, preserve_eager_precision=True
    )
    assert selected is custom_backend
    assert looked_up == [requested]


@pytest.mark.parametrize("route", ["fullgraph", "blocks", "spatial"])
@pytest.mark.parametrize("compile_loss", [False, True])
@pytest.mark.parametrize("amp_enabled", [False, True])
def test_native_adapters_request_gpu_precision_only_for_amp_compiled_loss(
    monkeypatch, route, compile_loss, amp_enabled
):
    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda"))]),
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
    assert requested == [
        (
            "inductor",
            {
                "device_type": "cuda",
                "preserve_eager_precision": compile_loss and amp_enabled,
            },
        )
    ]
