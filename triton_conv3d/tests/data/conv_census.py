#!/usr/bin/env python3
# SPDX-License-Identifier: (Apache-2.0)
"""Record every convolution one ScaFFold run issues, measured rather than assumed.

    python triton_conv3d/tests/data/conv_census.py --out cens_A.json -- \\
        benchmark -c <config.yml> [more scaffold args]

Everything after ``--`` is handed to the ``scaffold`` CLI unchanged, so the
capture is of a real benchmark run.  When that run ends the record is written
to ``--out`` with its suffix replaced by ``.census.json`` (``.rN.census.json``
per rank under a multi-rank launch), with the top-level keys ``rank``,
``world``, ``env``, ``modules``, ``kernels`` and ``autograd_fn``.

Three layers of hook, so the transformation between them is visible:

* ``modules`` -- ``FastConv3d.forward`` / ``FastConvTranspose3d.forward``: the
  logical convolution as the model states it (the module's own padding, the
  local shard's shape for a ``DCTensor``), and the rung that served it.
* ``autograd_fn`` -- ``_TritonConv3dFn`` / ``_TritonConvTranspose3dFn``: what
  survives the halo exchange, i.e. the padding the kernel is actually given.
* ``kernels`` -- the six ``triton_conv3d`` entry points: the call itself plus
  the config each resolver answers with (for backward-weight also the default
  and the tuned row it chose between).  A transposed site's backward
  directions are ordinary strided convolutions and appear twice, as the
  transposed entry point and as the plain one it delegates to; the ``via`` tag
  tells the two apart.

Every record names its site (the module's path within the UNet), is keyed by
site and shape, and carries a call count ``n``.  Nothing tracked is modified and
no timing is claimed: run it separately from a timed run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from ScaFFold.unet import conv3d as c3  # noqa: E402
from ScaFFold.unet import unet_model as um  # noqa: E402
from triton_conv3d import bwd_data as bd  # noqa: E402
from triton_conv3d import gather_gemm as gg  # noqa: E402
from triton_conv3d import reduce_gemm as rg  # noqa: E402
from triton_conv3d import transposed as tp  # noqa: E402

_modules: dict[str, dict] = {}  # module level: the logical convolution
_kernels: dict[str, dict] = {}  # entry-point level: what the kernel got
_fn: dict[str, dict] = {}  # autograd-node level: post-halo padding
_ctx = threading.local()


def _site():
    return getattr(_ctx, "site", "?")


def _via():
    return getattr(_ctx, "via", None)


def _via_tag():
    return getattr(_ctx, "inner", None)


def _triple(v):
    if isinstance(v, (tuple, list)):
        return [int(x) for x in v]
    return [int(v)] * 3


def _memfmt(t):
    try:
        if t.is_contiguous(memory_format=torch.channels_last_3d):
            return "channels_last_3d"
        if t.is_contiguous():
            return "contiguous"
    except Exception:  # noqa: BLE001
        pass
    return "other"


def _bump(table, key, rec):
    e = table.setdefault(key, dict(rec, n=0))
    e["n"] += 1
    return e


# --------------------------------------------------------------------------
# 1. Site names.  Tagged on the model after it is built, so every record can
#    name the layer rather than an id().
# --------------------------------------------------------------------------
def _tag_sites():
    orig = um.UNet.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        for name, mod in self.named_modules():
            if isinstance(mod, (c3.FastConv3d, c3.FastConvTranspose3d)):
                mod._census_site = name

    um.UNet.__init__ = init


# --------------------------------------------------------------------------
# 2. Module level: the logical convolution, as the model states it.
# --------------------------------------------------------------------------
def _wrap_module(cls, op):
    orig = cls.forward

    def spy(self, input, *a, **k):
        site = getattr(self, "_census_site", f"<{op}>")
        dc = c3._dctensor_ops(input) is not None
        local = input._tensor if dc else input
        rec = {
            "site": site,
            "op": op,
            "dctensor": dc,
            "local_shape": list(local.shape),
            "global_shape": list(input.shape),
            "weight_shape": list(self.weight.shape),
            "module_padding": _triple(self.padding),
            "stride": _triple(self.stride),
            "dilation": _triple(self.dilation),
            "groups": int(self.groups),
            "bias": self.bias is not None,
            "param_dtype": str(self.weight.dtype),
            "input_dtype": str(local.dtype),
            "autocast_dtype": str(c3._autocast_dtype(local)),
            "input_memfmt": _memfmt(local),
            "weight_memfmt": _memfmt(self.weight),
        }
        key = (
            f"{op} {site} x={rec['local_shape']} w={rec['weight_shape']} "
            f"pad={rec['module_padding']} dc={dc}"
        )
        prev_site, prev_via = _site(), _via()
        _ctx.site, _ctx.via = site, None
        try:
            out = orig(self, input, *a, **k)
        finally:
            rec["rung"] = _ctx.via or "miopen"
            _bump(_modules, key, rec)
            _ctx.site, _ctx.via = prev_site, prev_via
        return out

    cls.forward = spy


# --------------------------------------------------------------------------
# 3. Autograd-node level: the padding that survives the halo exchange.
# --------------------------------------------------------------------------
def _unbound(fn):
    return fn.__func__ if hasattr(fn, "__func__") else fn


def _wrap_fn(cls, op, pad_index):
    orig_fwd = _unbound(cls.forward)
    orig_bwd = _unbound(cls.backward)

    def fwd(ctx, x, weight, bias, *rest):
        padding = _triple(rest[pad_index])
        site = _site()
        rec = {
            "site": site,
            "op": op,
            "x_shape": list(x.shape),
            "weight_shape": list(weight.shape),
            "kernel_padding": padding,
            "stride": _triple(rest[pad_index - 1]),
            "dtype": str(x.dtype),
            "memfmt": _memfmt(x),
            "padded": any(padding),
        }
        _bump(_fn, f"{op} {site} x={rec['x_shape']} pad={padding}", rec)
        _ctx.via = "triton"
        out = orig_fwd(ctx, x, weight, bias, *rest)
        ctx._census_site = site
        return out

    def bwd(ctx, *grads):
        # The backward runs outside the module's forward, so the site has to be
        # carried across on the context or every gradient call is anonymous.
        prev = _site()
        _ctx.site = getattr(ctx, "_census_site", "?")
        try:
            return orig_bwd(ctx, *grads)
        finally:
            _ctx.site = prev

    cls.forward = staticmethod(fwd)
    cls.backward = staticmethod(bwd)


# --------------------------------------------------------------------------
# 4. Entry points, and the config each resolver answers with.
# --------------------------------------------------------------------------
def _rec_kernel(
    direction,
    op,
    x_shape,
    w_shape,
    padding,
    stride,
    dilation,
    groups,
    dtype,
    memfmt,
    config,
    padded,
    extra=None,
):
    rec = {
        "site": _site(),
        "op": op,
        "direction": direction,
        "x_shape": list(x_shape),
        "weight_shape": list(w_shape),
        "padding": _triple(padding),
        "stride": _triple(stride),
        "dilation": _triple(dilation),
        "groups": int(groups),
        "dtype": str(dtype),
        "memfmt": memfmt,
        "config": str(config),
        "padded": bool(padded),
    }
    if extra:
        rec.update(extra)
    key = (
        f"{op} {direction} {rec['site']} x={rec['x_shape']} "
        f"w={rec['weight_shape']} pad={rec['padding']} "
        f"s={rec['stride']} via={rec.get('via')}"
    )
    _bump(_kernels, key, rec)


def _wrap_conv_fwd():
    orig = gg.conv3d_forward

    def spy(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1, **k):
        pad, st, dil = _triple(padding), _triple(stride), _triple(dilation)
        kk = tuple(int(v) for v in w.shape[2:])
        m = int(x.shape[0])
        for i in range(3):
            extent = int(x.shape[2 + i]) + 2 * pad[i] - dil[i] * (kk[i] - 1) - 1
            m *= extent // st[i] + 1
        try:
            cfg = gg.select_config(m, int(w.shape[1]), int(w.shape[0]), kk, x.dtype)
        except Exception as exc:  # noqa: BLE001
            cfg = f"?: {type(exc).__name__}: {exc}"
        _rec_kernel(
            "fwd",
            "Conv3d",
            x.shape,
            w.shape,
            pad,
            st,
            dil,
            groups,
            x.dtype,
            _memfmt(x),
            cfg,
            any(pad),
            {"via": _via_tag(), "gemm_m": m},
        )
        return orig(x, w, bias, stride, padding, dilation, groups, **k)

    gg.conv3d_forward = spy


def _wrap_bwd_data():
    orig = bd.conv3d_backward_data

    def spy(
        grad_output,
        weight,
        input_shape,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        **k,
    ):
        pad, st, dil = _triple(padding), _triple(stride), _triple(dilation)
        try:
            cfg = str(
                bd.bwd_data_config(
                    tuple(int(v) for v in grad_output.shape),
                    int(weight.shape[1]),
                    tuple(int(v) for v in weight.shape[2:]),
                    grad_output.dtype,
                    padding=padding,
                    dilation=dilation,
                )
            )
        except Exception as exc:  # noqa: BLE001
            cfg = f"?: {type(exc).__name__}: {exc}"
        _rec_kernel(
            "bwd-data",
            "Conv3d",
            input_shape,
            weight.shape,
            pad,
            st,
            dil,
            groups,
            grad_output.dtype,
            _memfmt(grad_output),
            cfg,
            any(pad),
            {"via": _via_tag(), "grad_output_shape": list(grad_output.shape)},
        )
        return orig(
            grad_output, weight, input_shape, stride, padding, dilation, groups, **k
        )

    bd.conv3d_backward_data = spy


def _wrap_bwd_weight():
    orig = rg.conv3d_backward_weight

    def spy(
        input,
        weight_shape,
        grad_output,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        **k,
    ):
        pad, st, dil = _triple(padding), _triple(stride), _triple(dilation)
        ws = [int(v) for v in weight_shape]
        m = 1
        for v in grad_output.shape:
            m *= int(v)
        m //= ws[0]
        padded = any(pad)
        try:
            cfg = str(
                rg.bwd_weight_config(
                    ws[0], ws[1], tuple(ws[2:]), m, input.dtype, padded=padded
                )
            )
            dflt = str(
                rg.default_bwd_weight_config(
                    ws[0], ws[1], tuple(ws[2:]), m, input.dtype, padded=padded
                )
            )
            tuned = rg._TUNED_BWD_W.get(
                rg.tune_key(input.dtype, ws[1], ws[0], tuple(ws[2:]))
            )
        except Exception as exc:  # noqa: BLE001
            cfg, dflt, tuned = f"?: {type(exc).__name__}: {exc}", None, None
        _rec_kernel(
            "bwd-weight",
            "Conv3d",
            input.shape,
            ws,
            pad,
            st,
            dil,
            groups,
            input.dtype,
            _memfmt(input),
            cfg,
            padded,
            {
                "via": _via_tag(),
                "grad_output_shape": list(grad_output.shape),
                "default_config": dflt,
                "tuned_row": str(tuned) if tuned is not None else None,
            },
        )
        return orig(
            input, weight_shape, grad_output, stride, padding, dilation, groups, **k
        )

    rg.conv3d_backward_weight = spy


def _wrap_transposed():
    o_fwd = tp.conv_transpose3d_forward
    o_bd = tp.conv_transpose3d_backward_data
    o_bw = tp.conv_transpose3d_backward_weight

    def fwd(
        x,
        w,
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        **k,
    ):
        pad = _triple(padding)
        m = int(x.shape[0]) * int(x.shape[2]) * int(x.shape[3]) * int(x.shape[4])
        try:
            cfg = str(
                tp.transposed_config(
                    m,
                    int(w.shape[0]),
                    int(w.shape[1]),
                    tuple(int(v) for v in w.shape[2:]),
                    x.dtype,
                )
            )
        except Exception as exc:  # noqa: BLE001
            cfg = f"?: {type(exc).__name__}: {exc}"
        _rec_kernel(
            "fwd",
            "ConvTranspose3d",
            x.shape,
            w.shape,
            pad,
            _triple(stride),
            _triple(dilation),
            groups,
            x.dtype,
            _memfmt(x),
            cfg,
            any(pad),
            {"output_padding": _triple(output_padding)},
        )
        prev, _ctx.inner = _via_tag(), "convT-fwd"
        try:
            return o_fwd(
                x, w, bias, stride, padding, output_padding, dilation, groups, **k
            )
        finally:
            _ctx.inner = prev

    def bwd_d(
        grad_output,
        w,
        input_shape,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        **k,
    ):
        _rec_kernel(
            "bwd-data",
            "ConvTranspose3d",
            input_shape,
            w.shape,
            _triple(padding),
            _triple(stride),
            _triple(dilation),
            groups,
            grad_output.dtype,
            _memfmt(grad_output),
            None,
            any(_triple(padding)),
            {"grad_output_shape": list(grad_output.shape)},
        )
        prev, _ctx.inner = _via_tag(), "convT-bwd-data"
        try:
            return o_bd(
                grad_output,
                w,
                input_shape,
                stride,
                padding,
                output_padding,
                dilation,
                groups,
                **k,
            )
        finally:
            _ctx.inner = prev

    def bwd_w(
        x,
        weight_shape,
        grad_output,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        **k,
    ):
        _rec_kernel(
            "bwd-weight",
            "ConvTranspose3d",
            x.shape,
            [int(v) for v in weight_shape],
            _triple(padding),
            _triple(stride),
            _triple(dilation),
            groups,
            x.dtype,
            _memfmt(x),
            None,
            any(_triple(padding)),
            {"grad_output_shape": list(grad_output.shape)},
        )
        prev, _ctx.inner = _via_tag(), "convT-bwd-weight"
        try:
            return o_bw(
                x,
                weight_shape,
                grad_output,
                stride,
                padding,
                output_padding,
                dilation,
                groups,
                **k,
            )
        finally:
            _ctx.inner = prev

    tp.conv_transpose3d_forward = fwd
    tp.conv_transpose3d_backward_data = bwd_d
    tp.conv_transpose3d_backward_weight = bwd_w


def install():
    """Install every hook; once per process, a second call would count twice."""
    _tag_sites()
    _wrap_module(c3.FastConv3d, "Conv3d")
    _wrap_module(c3.FastConvTranspose3d, "ConvTranspose3d")
    _wrap_fn(c3._TritonConv3dFn, "Conv3d", 1)  # rest = (stride, padding, dilation)
    _wrap_fn(c3._TritonConvTranspose3dFn, "ConvTranspose3d", 1)
    _wrap_conv_fwd()
    _wrap_bwd_data()
    _wrap_bwd_weight()
    _wrap_transposed()
    # ``transposed.py`` binds these at import time and both of its backward
    # directions go through them.  Rebinding them to the spies makes that
    # second, underlying problem visible too; the ``via`` tag keeps the two
    # apart.  ``bwd_data.py``'s own ``conv3d_forward`` is deliberately left
    # alone, so the ordinary backward-data is counted once rather than twice.
    tp.conv3d_forward = gg.conv3d_forward
    tp.conv3d_backward_weight = rg.conv3d_backward_weight


def dump(out):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    p = Path(out)
    # ``with_suffix`` after appending ``.rN`` would eat the rank again, so the
    # name is built by hand.
    p = p.with_name(p.stem + (f".r{rank}" if world > 1 else "") + ".census.json")
    p.write_text(
        json.dumps(
            {
                "rank": rank,
                "world": world,
                "env": {
                    k: os.environ.get(k)
                    for k in (
                        "SCAFFOLD_CONV_TRITON",
                        "SCAFFOLD_GROUPNORM_TRITON",
                        "PYTORCH_MIOPEN_SUGGEST_NHWC",
                        "HIP_VISIBLE_DEVICES",
                    )
                },
                "modules": _modules,
                "kernels": _kernels,
                "autograd_fn": _fn,
            },
            indent=1,
        )
        + "\n"
    )
    print(
        f"[conv census] {len(_modules)} module calls, {len(_kernels)} kernel "
        f"calls -> {p}",
        flush=True,
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        usage="%(prog)s --out FILE -- <scaffold subcommand and arguments>",
    )
    parser.add_argument("--out", required=True, help="capture path stem")
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    scaffold_argv = argv[split + 1 :]
    if not scaffold_argv:
        parser.error("the scaffold subcommand and its arguments must follow a '--'")

    from ScaFFold import cli

    install()
    sys.argv = ["scaffold", *scaffold_argv]
    try:
        return cli.main()
    finally:
        dump(args.out)


if __name__ == "__main__":
    sys.exit(main())
