# SPDX-License-Identifier: (Apache-2.0)
"""The convolution problems the kernels have to serve.

Three sources:

``scaffold_corpus()``
    The 57 distinct convolutions that occur in real ScaFFold runs, extracted
    from ``model-analysis/unet_shapes.py`` dumps for the three profiled
    configurations (scale 7 on 1 GPU, scale 8 sharded over 2 and over 4 GPUs)
    and joined with the measured MIOpen cost and roofline efficiency of each.
    This is the tuning target and the priority list.

``census_corpus()``
    Every convolution an **instrumented ScaFFold training step actually
    issued**, at the four configurations the current benchmark harness runs.
    Recorded by wrapping the entry points inside a running step, so it is a
    measurement rather than a model.  It carries no MIOpen timings; it exists
    to say what the shapes *are*.

``edge_cases()``
    Synthetic problems chosen to break addressing, masking and tiling
    assumptions: channel counts that are not multiples of the MFMA granularity,
    prime spatial extents, anisotropic volumes, batches, and volumes whose
    linear element index exceeds 2**31.

All return :class:`ConvProblem`, which knows how to derive the tensor shapes,
the FLOP and byte counts, and the implicit-GEMM shape for each of the three
directions.  Nothing here imports torch, so it is cheap to introspect and can
drive test parametrization at collection time.

One problem, three forms
========================

A single ScaFFold convolution reaches a kernel in three different shapes
depending on *who* issues it, and they are three different tuning problems --
MIOpen keys its find database on the padding, this package's ``bwd_data_config``
derives ``M`` from it, and every one of them is a different *measurement*.
Confusing them has already cost this project one wrong projection, so
:attr:`ConvProblem.form` names which one an instance is and every corpus
accessor says which it returns.  (``bwd_weight_config`` used to change its
answer on the padding as well; that clause went on 2026-08-05 and the forms are
still three problems without it.)

``"logical"``
    The convolution as ``unet_parts.py`` states it: the local shard at the
    module's own padding, ``k // 2`` on every axis.  This is what
    :func:`scaffold_corpus` stores.

``"distconv"``
    What upstream DistConv hands the backend: it concatenates a ``k // 2`` halo
    on **every dimension listed in ``dc_shard_dims``**, split or not, and zeroes
    the padding there.  :attr:`ConvProblem.halo_variant`.  This is the form the
    MIOpen baseline in ``measured`` was profiled in, and it is the incumbent's
    problem.

``"adapter"``
    What ``ScaFFold/unet/conv3d.py`` -- the shipped Triton rung -- hands the
    kernel: it exchanges a halo only on axes that are *genuinely split*, so at
    ``dc_num_shards = (1, 1, 1)`` nothing is exchanged and the convolution is
    padded on all three axes, and at ``(2, 1, 1)`` or ``(4, 1, 1)`` only D is
    halo'd while H and W stay padded.  :attr:`ConvProblem.production_variant`.
    **This is the form production actually runs today**, at every
    configuration, and it is padded at every one of them.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import pathlib
from typing import Iterator, Literal, Sequence

_CORPUS_PATH = pathlib.Path(__file__).resolve().parent / "scaffold_corpus.json"
_CENSUS_PATH = pathlib.Path(__file__).resolve().parent / "scaffold_census.json"

Direction = Literal["fwd", "bwd-data", "bwd-weight"]
DIRECTIONS: tuple[Direction, ...] = ("fwd", "bwd-data", "bwd-weight")

#: Which of the three statements of a problem an instance is.  See the module
#: docstring; the short version is that they differ in the padding and in the
#: input extent, and that ``"adapter"`` is the one production runs.
Form = Literal["logical", "distconv", "adapter"]
FORMS: tuple[Form, ...] = ("logical", "distconv", "adapter")

#: Empirical MI300A roofline constants, supplied rather than derived.  They are
#: measured ceilings: a kernel that exceeds one is a fact about the measurement,
#: not necessarily an error.
HBM_BYTES_PER_S = 3.3e12
PEAK_FLOPS = {"fp32": 82.6e12, "bf16": 600e12, "fp16": 600e12}
ELEM_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}

#: Largest linear element index representable in int32.
INT32_MAX = 2**31 - 1

#: Largest *storage*, in bytes, that an AMD buffer instruction can address.
#: Triton's runtime specializer tags a pointer argument ``tt.pointer_range = 32``
#: -- the flag that lets the backend emit ``buffer_load_dwordx4`` rather than
#: ``global_load_dwordx4`` -- from ``arg.untyped_storage().size() <= 2**31 - 1``.
#: That check reads the whole *storage* and it counts *bytes*, so it is a
#: different question from whether an element index overflows int32, and the two
#: answers differ by ``elem_bytes``.
BUFFER_OP_MAX_BYTES = 2**31 - 1


def _prod(xs: Sequence[int]) -> int:
    return math.prod(xs)


@dataclasses.dataclass(frozen=True)
class ConvProblem:
    """One convolution, in the form a kernel is tuned for.

    ``spatial`` is always the *input* volume, ``(D, H, W)``.  For the transposed
    case ``cin``/``cout`` keep their logical meaning (the operator maps ``cin``
    channels to ``cout``), which is the transpose of how PyTorch stores the
    weight -- :meth:`weight_shape` accounts for that.
    """

    name: str
    cin: int
    cout: int
    spatial: tuple[int, int, int]
    kernel: tuple[int, int, int] = (3, 3, 3)
    stride: tuple[int, int, int] = (1, 1, 1)
    padding: tuple[int, int, int] = (1, 1, 1)
    n: int = 1
    transposed: bool = False
    bias: bool = False
    dtype: str = "bf16"
    #: Where this problem came from: ScaFFold site tags, or "synthetic".
    sites: tuple[str, ...] = ()
    #: Measured MIOpen results, if any: one dict per (config, direction).
    measured: tuple[dict, ...] = ()
    #: Set for problems that need a lot of memory or a long time; opt-in.
    large: bool = False
    #: **Upstream DistConv's** halo width per spatial dim: ``k // 2`` on every
    #: dim listed in ``dc_shard_dims``, whether or not that dim is actually
    #: split.  Non-zero means the *MIOpen* rung -- which goes through
    #: ``distconv_forward`` -- runs the convolution unpadded at a larger extent;
    #: see :meth:`halo_variant`.  It says nothing about the Triton rung, which
    #: exchanges only what :attr:`shard_halo` records.  Zero for synthetic
    #: problems.
    halo: tuple[int, int, int] = (0, 0, 0)
    #: **The ScaFFold Triton adapter's** halo width per spatial dim: ``k // 2``
    #: on the dims that are genuinely split (``dc_num_shards > 1``) and zero
    #: elsewhere, because ``ScaFFold/unet/conv3d.py``'s ``_halo_plan`` skips an
    #: unsplit axis and leaves the module's own padding on it.  This is what
    #: separates the production form from DistConv's -- see
    #: :meth:`production_variant`.
    shard_halo: tuple[int, int, int] = (0, 0, 0)
    #: Which of the three statements of the problem this instance is.  Set by
    #: :meth:`halo_variant` and :meth:`production_variant`; ``"logical"``
    #: otherwise.  Carried so that a table row, a benchmark cell and a JSON
    #: record all say which shape they measured instead of leaving it to be
    #: inferred from the padding.
    form: Form = "logical"

    # -- derived shapes ---------------------------------------------------

    @functools.cached_property
    def out_spatial(self) -> tuple[int, int, int]:
        if self.transposed:
            return tuple(
                (i - 1) * s - 2 * p + k
                for i, k, s, p in zip(
                    self.spatial, self.kernel, self.stride, self.padding
                )
            )
        return tuple(
            (i + 2 * p - k) // s + 1
            for i, k, s, p in zip(self.spatial, self.kernel, self.stride, self.padding)
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (self.n, self.cin, *self.spatial)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.n, self.cout, *self.out_spatial)

    @property
    def weight_shape(self) -> tuple[int, ...]:
        """PyTorch's storage order, which differs between the two operators."""
        if self.transposed:
            return (self.cin, self.cout, *self.kernel)
        return (self.cout, self.cin, *self.kernel)

    @property
    def elem_bytes(self) -> int:
        return ELEM_BYTES[self.dtype]

    @property
    def halo_variant(self) -> "ConvProblem":
        """The same convolution in the form **upstream DistConv** issues it.

        ``distconv_forward`` never lets PyTorch pad a convolution on a dimension
        it manages.  It concatenates a halo slab of width ``k // 2`` onto both
        faces -- neighbour data, or zeros at the mesh boundary and at one shard
        -- and then sets that dimension's padding to zero (``distconv.py``).
        It does this for **every dim in ``dc_shard_dims``**, including dims with
        a single shard, where the slab is provably zeros; that is why
        :attr:`halo` is ``(1, 1, 1)`` on every ``k = 3`` corpus problem even at
        one GPU.  So the tensor MIOpen sees is two voxels larger per listed axis
        and the convolution is unpadded.

        **This is the incumbent's form, not the shipped one.**  ScaFFold routes
        its convolutions through ``ScaFFold/unet/conv3d.py``, which performs the
        exchange itself and only on axes that are genuinely split -- see
        :meth:`production_variant`.  The MIOpen numbers in :attr:`measured` were
        profiled through DistConv, so they *are* timings of this form, and it is
        the right form to compare an MIOpen baseline in; it is not the form the
        Triton kernels are handed.

        This distinction is not cosmetic.  MIOpen keys its find database on the
        whole problem descriptor, padding included: ``64-128-128-128-...-1x1x1``
        and ``64-130-130-130-...-0x0x0`` are two different problems that tune
        independently and can land on different kernels.  This package used to
        change its own answer on it too -- :func:`~triton_conv3d.reduce_gemm.
        bwd_weight_config` declined a tuned row with ``TAP_BLOCK > 1`` on a
        padded convolution until 2026-08-05 -- and no longer does; but the two
        forms remain different *problems*, they are timed differently, and
        :func:`~triton_conv3d.bwd_data.bwd_data_config` still reads the padding
        because it derives ``M`` from it.

        The cost model follows the shape rather than being special-cased: the
        halo'd form genuinely reads a slightly larger input and genuinely
        produces a slightly larger input gradient, and :meth:`flops` and
        :meth:`bytes` say so because they are derived from ``spatial``.
        Returns ``self`` when there is no halo -- the three forms genuinely
        coincide there, and returning an unequal copy would only make a
        distinction the problem does not have.
        """
        if not any(self.halo):
            return self
        return dataclasses.replace(
            self,
            name=f"{self.name}+halo" if self.name else "halo",
            spatial=tuple(s + 2 * h for s, h in zip(self.spatial, self.halo)),
            padding=tuple(0 if h else p for h, p in zip(self.halo, self.padding)),
            halo=(0, 0, 0),
            shard_halo=(0, 0, 0),
            form="distconv",
        )

    @property
    def production_variant(self) -> "ConvProblem":
        """The same convolution in the form **ScaFFold runs it today**.

        ``ScaFFold/unet/conv3d.py``'s ``_halo_plan`` walks the parallel
        strategy and ``continue``s past any axis with a single shard, so it
        exchanges a halo *only* on axes that are genuinely split and leaves the
        module's own padding on every other one.  ScaFFold ships
        ``dc_shard_dims: [2, 3, 4]`` with ``dc_num_shards`` of ``[1,1,1]``,
        ``[2,1,1]`` or ``[4,1,1]``, so:

        * unsharded, nothing is exchanged and the convolution reaches the kernel
          at its logical extent with ``padding = (1, 1, 1)``;
        * sharded, D is halo'd and H and W are still padded --
          ``padding = (0, 1, 1)`` at ``(D_loc + 2, H, W)``.

        **Every production convolution with ``k > 1`` is therefore padded, at
        every configuration.**  Measured inside running steps at all four, not
        inferred: 18 of the 19 distinct ordinary convolutions at scale 7 on
        one GPU arrive with ``padding = (1, 1, 1)``, and the nineteenth is the
        ``k = 1`` head, which has no padding to begin with.

        Dropping the zero slabs on the unsplit axes is a deliberate and
        separately verified decision -- ``cat(zeros, x, zeros)`` at
        ``padding = 0`` is the same arithmetic as ``padding = k // 2`` on ``x``,
        and it is measured bitwise identical through these kernels -- so this is
        not a divergence to be repaired but the shape to be tuned for.

        Returns ``self`` when nothing is split, for the same reason
        :meth:`halo_variant` does: unsharded, the logical statement *is* what
        the adapter issues, and there is no distinction to record.
        """
        if not any(self.shard_halo):
            return self
        return dataclasses.replace(
            self,
            name=f"{self.name}+shard" if self.name else "shard",
            spatial=tuple(s + 2 * h
                          for s, h in zip(self.spatial, self.shard_halo)),
            padding=tuple(0 if h else p
                          for h, p in zip(self.shard_halo, self.padding)),
            halo=(0, 0, 0),
            shard_halo=(0, 0, 0),
            form="adapter",
        )

    # -- cost model -------------------------------------------------------

    @property
    def tap_count(self) -> int:
        return _prod(self.kernel)

    def flops(self, direction: Direction = "fwd") -> int:
        """Multiply-accumulate count x2.

        All three directions perform the same contraction with different operands
        held fixed, so the count differs only in which volume indexes it.  For a
        forward convolution each *output* voxel gathers ``taps`` contributions;
        backward-data is the same contraction over the *input* volume.

        The transposed operator scatters instead of gathering, so every direction
        is indexed by the *input* volume -- and with ``kernel == stride`` that
        makes the tap factor illusory: each output voxel receives exactly one
        contribution, because the windows tile rather than overlap.
        """
        if self.transposed:
            vol = _prod(self.spatial)
        else:
            vol = _prod(self.spatial if direction == "bwd-data" else self.out_spatial)
        return 2 * self.n * vol * self.cin * self.cout * self.tap_count

    def bytes(self, direction: Direction = "fwd") -> int:
        """Compulsory traffic: each tensor the direction touches, read once.

        This is the denominator of the memory roof.  It credits the kernel with
        perfect reuse -- no im2col materialization, no partial spilling -- which
        is exactly the standard a fused implicit-GEMM kernel should be held to.
        """
        eb = self.elem_bytes
        x = self.n * self.cin * _prod(self.spatial) * eb
        y = self.n * self.cout * _prod(self.out_spatial) * eb
        w = self.cin * self.cout * self.tap_count * eb
        return {"fwd": x + w + y, "bwd-data": y + w + x, "bwd-weight": y + x + w}[
            direction
        ]

    def arithmetic_intensity(self, direction: Direction = "fwd") -> float:
        return self.flops(direction) / self.bytes(direction)

    def roofline_flops(self, direction: Direction = "fwd") -> float:
        """Attainable FLOP/s: whichever of compute and bandwidth binds first."""
        return min(
            PEAK_FLOPS[self.dtype],
            self.arithmetic_intensity(direction) * HBM_BYTES_PER_S,
        )

    def efficiency(self, ms: float, direction: Direction = "fwd") -> float:
        """Fraction of the roofline achieved by a measured time in milliseconds."""
        return (self.flops(direction) / (ms * 1e-3)) / self.roofline_flops(direction)

    # -- implicit-GEMM decomposition -------------------------------------

    def gemm_shape(self, direction: Direction = "fwd") -> tuple[int, int, int]:
        """``(M, N, K)`` of the GEMM this direction reduces to.

        Forward and backward-data tile over a volume with the channel count as N
        and the taps folded into K.  Backward-weight is the transpose of that
        situation: a tiny output reduced over the whole volume, which is why it
        needs split-K and why determinism is a live question there.

        The transposed operator with ``kernel == stride`` and no padding is a
        special case -- a pointwise GEMM producing ``cout * taps`` channels,
        followed by a voxel shuffle -- so its forward K carries no tap factor.
        """
        taps = self.tap_count
        if self.transposed:
            if self.kernel != self.stride or set(self.padding) != {0}:
                raise NotImplementedError(
                    "transposed convolutions are only decomposed for "
                    f"kernel == stride and no padding; got kernel={self.kernel}, "
                    f"stride={self.stride}, padding={self.padding}"
                )
            in_vol = self.n * _prod(self.spatial)
            if direction == "fwd":
                return (in_vol, self.cout * taps, self.cin)
            if direction == "bwd-data":
                return (in_vol, self.cin, self.cout * taps)
            return (self.cin, self.cout * taps, in_vol)
        out_vol = self.n * _prod(self.out_spatial)
        in_vol = self.n * _prod(self.spatial)
        if direction == "fwd":
            return (out_vol, self.cout, self.cin * taps)
        if direction == "bwd-data":
            return (in_vol, self.cin, self.cout * taps)
        return (self.cout, self.cin * taps, out_vol)

    # -- indexing ---------------------------------------------------------

    @property
    def max_elements(self) -> int:
        """Element count of the larger activation.

        The largest linear index a pointer into it will see is therefore
        ``max_elements - 1``.
        """
        return max(
            self.n * self.cin * _prod(self.spatial),
            self.n * self.cout * _prod(self.out_spatial),
        )

    @property
    def max_activation_bytes(self) -> int:
        """Storage of the larger activation, in bytes.

        This -- not :attr:`max_elements` -- is the quantity the AMD backend
        cares about, and it is ``elem_bytes`` times larger.
        """
        return self.max_elements * self.elem_bytes

    @property
    def index_exceeds_int32(self) -> bool:
        """The kernel's *element* offsets must be widened to int64.

        Counted in elements because that is what a Triton offset holds: the
        largest one is ``max_elements - 1``, so the boundary sits at ``2**31``
        elements and not at ``INT32_MAX``.  (The old form compared
        ``max_elements > INT32_MAX``, which fires one element early -- harmless,
        but it made the predicate hard to reason about at the boundary the
        edge cases exist to pin.)

        This is emphatically **not** the 2 GiB cliff.  No corpus problem
        reaches it in either shape mode, including the cliff cell itself:
        ``conv 128->64 k3 @ 130x258x258`` holds 1.108e9 elements -- half of
        int32's range -- in 2.22 GiB of storage.  What that shape loses is
        buffer ops, which is :attr:`buffer_ops_eligible`.
        """
        return self.max_elements - 1 > INT32_MAX

    #: The name ``bench/baseline.py`` records this predicate under, and so the
    #: name it carries in every row of ``baseline.json``.  Kept as an alias
    #: rather than renamed in place, because the field is published data.
    needs_int64 = index_exceeds_int32

    @property
    def buffer_ops_eligible(self) -> bool:
        """The larger activation still fits the buffer-load fast path.

        False costs about 4.5% on the shapes we measured it on (M1), and it is
        the property that separates the corpus's one cliff cell from the rest of
        it: a 2.22 GiB activation is 3.2% over the byte limit while being
        nowhere near the *element* limit.  Modelled from the shape, so it
        assumes a freshly allocated tensor -- a narrowed view keeps its parent's
        storage and ``conv_bench`` therefore measures the same predicate off
        ``untyped_storage().size()`` instead.
        """
        return self.max_activation_bytes <= BUFFER_OP_MAX_BYTES

    # -- reporting --------------------------------------------------------

    @property
    def label(self) -> str:
        k = "x".join(map(str, self.kernel))
        s = "x".join(map(str, self.spatial))
        op = "convT" if self.transposed else "conv"
        return f"{op} {self.cin}->{self.cout} k{k} @ {s}"

    @property
    def qualified_label(self) -> str:
        """:attr:`label` plus the two things that make it a *different problem*.

        ``label`` names the operator, the channels, the kernel and the extent,
        and every published table is keyed on it -- so it stays exactly as it
        is.  It does not name the padding, and the padding is what separates the
        three forms of the module docstring: ``conv 64->64 k3 @ 128x128x128``
        alone does not say whether it is the padded convolution ScaFFold runs or
        an unpadded one, and MIOpen and this package both answer differently on
        that.  Use this wherever a reader could otherwise take a halo'd cell for
        a production one.
        """
        p = ",".join(map(str, self.padding))
        return f"{self.label} p{p} [{self.form}]"

    def measured_for(self, direction: Direction, config: str | None = None):
        """The MIOpen measurements for one direction, most expensive first."""
        hits = [
            m
            for m in self.measured
            if m["direction"] == direction and (config is None or m["config"] == config)
        ]
        return sorted(hits, key=lambda m: -m["ms_per_call"])


# ---------------------------------------------------------------------------
# The ScaFFold corpus
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def scaffold_corpus() -> tuple[ConvProblem, ...]:
    """Every distinct convolution in the three profiled ScaFFold configurations.

    Ordered by measured cost, so truncating the list keeps the problems that
    matter.  Loaded from ``scaffold_corpus.json``, which is generated from the
    profiled shape dumps rather than written by hand.
    """
    raw = json.loads(_CORPUS_PATH.read_text())
    problems = []
    for entry in raw["problems"]:
        w = entry["weight_shape"]
        transposed = entry["op"] == "ConvTranspose3d"
        cin, cout = (w[0], w[1]) if transposed else (w[1], w[0])
        n, _, *spatial = entry["in_shape"]
        k = tuple(w[2:5])
        problems.append(
            ConvProblem(
                name="+".join(s.split(":", 1)[1] for s in entry["sites"][:1]),
                cin=cin,
                cout=cout,
                spatial=tuple(spatial),
                kernel=k,
                stride=(entry["stride"],) * 3,
                padding=(entry["padding"],) * 3,
                n=n,
                transposed=transposed,
                bias=entry["bias"],
                dtype="bf16",
                sites=tuple(entry["sites"]),
                measured=tuple(entry.get("measured", ())),
                halo=tuple(entry.get("halo_dhw") or (0, 0, 0)),
                shard_halo=tuple(entry.get("shard_halo_dhw") or (0, 0, 0)),
            )
        )
    return tuple(problems)


@functools.lru_cache(maxsize=1)
def halo_corpus() -> tuple[ConvProblem, ...]:
    """The corpus as **upstream DistConv** issues it: halo'd input, no padding.

    This -- not :func:`scaffold_corpus` -- is what the ``measured`` MIOpen
    timings in the corpus are timings *of*, because the profile that produced
    them ran through ``distconv_forward``.  It is therefore the right form to
    hold an *MIOpen baseline* in, and the wrong one to hold a Triton result in:
    the shipped Triton rung is handed :func:`production_corpus`'s form instead.
    :func:`scaffold_corpus` keeps the logical, unhaloed statement of each
    problem because that is what the shape dump records and what the FLOP model
    is naturally expressed in; the three differ by
    :meth:`ConvProblem.halo_variant` and :meth:`ConvProblem.production_variant`.
    """
    return tuple(p.halo_variant for p in scaffold_corpus())


@functools.lru_cache(maxsize=1)
def production_corpus() -> tuple[ConvProblem, ...]:
    """The corpus in the form **ScaFFold runs today** -- padded, mostly.

    :meth:`ConvProblem.production_variant` of every corpus problem: the local
    shard with a halo on the genuinely split axis only, and the module's own
    padding still in place on the others.  At the unsharded configuration this
    is identical to :func:`scaffold_corpus`; at the sharded ones it is a third
    shape, in neither :func:`scaffold_corpus` nor :func:`halo_corpus`.

    Verified against an instrumented run rather than asserted -- see
    ``test_infra.py::test_the_production_variant_matches_the_measured_census``,
    which joins this against :func:`census_corpus`.
    """
    return tuple(p.production_variant for p in scaffold_corpus())


@functools.lru_cache(maxsize=1)
def census_corpus() -> tuple[ConvProblem, ...]:
    """Every convolution an instrumented ScaFFold step actually issued.

    Recorded by a census harness that wraps ``FastConv3d`` /
    ``FastConvTranspose3d`` and the six kernel entry points and
    runs three real training steps at each of the four configurations the
    benchmark harness uses (A = scale 7 / 1 GPU, B = scale 8 / 1 GPU, C = scale
    8 / 2 GPUs, D = scale 8 / 4 GPUs).  Every problem here is in
    :attr:`ConvProblem.form` ``"adapter"`` by construction: it is the shape and
    padding the kernel was handed, read off the call.

    Why this exists beside :func:`scaffold_corpus`, rather than being folded
    into it:

    * it covers a configuration the profiled corpus does not (scale 8 on one
      GPU), and a *network depth* the profiled corpus does not -- the shape
      dumps behind :func:`scaffold_corpus` were taken at
      ``unet_bottleneck_dim = 4`` at scale 8, giving a four-layer model topping
      out at 1024 channels, while every step-level measurement in this project
      runs the shipped default of 3, i.e. a five-layer model topping out at 2048;
    * it carries no ``measured`` MIOpen data and no cost ordering, so it is not
      a priority list and must not be used as one;
    * and :func:`scaffold_corpus`'s ordering, indices and contents are the key
      every stored capture in this project refers to, so they do not move.

    ``large`` is set from the activation size, so a caller that iterates this
    without opting in does not try to allocate the 2 GiB scale-8 unsharded
    activations.
    """
    if not _CENSUS_PATH.exists():  # pragma: no cover - shipped with the package
        return ()
    raw = json.loads(_CENSUS_PATH.read_text())
    out = []
    for entry in raw["problems"]:
        w = entry["weight_shape"]
        transposed = entry["op"] == "ConvTranspose3d"
        cin, cout = (w[0], w[1]) if transposed else (w[1], w[0])
        n, _, *spatial = entry["in_shape"]
        out.append(
            ConvProblem(
                name=entry.get("name", ""),
                cin=cin,
                cout=cout,
                spatial=tuple(spatial),
                kernel=tuple(w[2:5]),
                stride=tuple(entry["stride"]),
                padding=tuple(entry["padding"]),
                n=n,
                transposed=transposed,
                bias=entry["bias"],
                dtype=entry.get("dtype", "bf16"),
                sites=tuple(entry["sites"]),
                large=bool(entry.get("large")),
                form="adapter",
            )
        )
    return tuple(out)


def hot_corpus(top: int = 12) -> tuple[ConvProblem, ...]:
    """The most expensive distinct problems -- the fast loop during development."""
    return scaffold_corpus()[:top]


# ---------------------------------------------------------------------------
# Synthetic edge cases
# ---------------------------------------------------------------------------


def edge_cases(include_large: bool = False) -> tuple[ConvProblem, ...]:
    """Problems chosen to break assumptions rather than to be fast.

    Each one targets a specific way an implicit-GEMM kernel goes wrong: tile
    remainders in every dimension, masking at volume faces, anisotropy, batching,
    and the int32 offset overflow that MIOpen itself gets wrong.
    """
    cases: list[ConvProblem] = [
        # Channel counts that are not multiples of any plausible BLOCK_K.
        ConvProblem("cin_tiny", 3, 64, (16, 16, 16), sites=("synthetic",)),
        ConvProblem("cin_odd", 5, 32, (8, 8, 8), sites=("synthetic",)),
        ConvProblem("cin_prime", 17, 24, (8, 8, 8), sites=("synthetic",)),
        ConvProblem("cout_tiny", 64, 6, (8, 8, 8), (1, 1, 1), padding=(0, 0, 0),
                    bias=True, sites=("synthetic",)),
        ConvProblem("cout_odd", 32, 7, (8, 8, 8), sites=("synthetic",)),
        # Spatial extents that do not divide any plausible tile.
        ConvProblem("spatial_prime", 32, 32, (13, 13, 13), sites=("synthetic",)),
        ConvProblem("spatial_one", 32, 32, (1, 8, 8), sites=("synthetic",)),
        ConvProblem("spatial_thin", 32, 32, (2, 31, 3), sites=("synthetic",)),
        ConvProblem("spatial_aniso", 64, 64, (5, 40, 96), sites=("synthetic",)),
        # Smaller than the kernel in one axis: every tap is masked somewhere.
        ConvProblem("smaller_than_kernel", 16, 16, (2, 2, 2), sites=("synthetic",)),
        # Padding variants: unpadded shrinks the output, k=1 removes the gather.
        ConvProblem("unpadded", 32, 32, (16, 16, 16), padding=(0, 0, 0),
                    sites=("synthetic",)),
        ConvProblem("pointwise", 64, 6, (16, 16, 16), (1, 1, 1), padding=(0, 0, 0),
                    bias=True, sites=("synthetic",)),
        ConvProblem("kernel_aniso", 32, 32, (8, 8, 8), (1, 3, 3), padding=(0, 1, 1),
                    sites=("synthetic",)),
        # The padding a *sharded* ScaFFold convolution actually reaches the
        # kernel with: a symmetric ``k = 3`` with the split axis halo'd (so
        # ``p = 0`` there) and H and W still padded.  Anisotropic padding under
        # an isotropic kernel is a combination nothing else here produces --
        # ``kernel_aniso`` gets its zero from ``kd = 1``, where the boundary
        # predicate on D is dead for a different reason -- and it is the form
        # every k=3 site runs at ``dc_num_shards = (2,1,1)`` or ``(4,1,1)``.
        ConvProblem("shard_padded", 32, 32, (8, 8, 8), padding=(0, 1, 1),
                    sites=("synthetic",)),
        # Batch > 1: ScaFFold never does this, but the M decomposition must.
        ConvProblem("batched", 32, 32, (8, 8, 8), n=3, sites=("synthetic",)),
        # The transposed upsample, at a size that is quick to check.
        ConvProblem("transposed", 64, 32, (8, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0),
                    transposed=True, bias=True, sites=("synthetic",)),
        # fp32, for more_determinism and for exact-arithmetic tests.
        ConvProblem("fp32", 32, 32, (8, 8, 8), dtype="fp32", sites=("synthetic",)),
        ConvProblem("fp16", 32, 32, (8, 8, 8), dtype="fp16", sites=("synthetic",)),
    ]
    if include_large:
        # The 2**31 *element* boundary, bracketed.  ``1 x 128 x 258^3`` =
        # 2.198e9 elements is the unsharded scale-8 activation that makes MIOpen
        # assert; ``255^3`` is the largest volume of the same shape family that
        # still fits an int32 index, at 2.122e9 elements.  The pair differs only
        # in spatial extent so that what it brackets is the boundary and not a
        # change of channel width or kernel as well.
        #
        # The previous ``int32_below`` was ``64 -> 64 @ 512^3``: 8.59e9
        # elements, four times *above* the boundary it was meant to sit below,
        # so the pair pinned nothing and the low case needed 16 GiB per
        # activation.  Both of these are 4.2-4.4 GiB in bf16 and both are past
        # the buffer-op byte limit -- see :attr:`ConvProblem.buffer_ops_eligible`
        # for why that is a different question from this one.
        cases += [
            ConvProblem("int32_below", 128, 64, (255, 255, 255), large=True,
                        sites=("synthetic",)),
            ConvProblem("int32_above", 128, 64, (258, 258, 258), large=True,
                        sites=("synthetic",)),
        ]
    return tuple(cases)


def all_problems(include_large: bool = False) -> Iterator[ConvProblem]:
    yield from scaffold_corpus()
    yield from edge_cases(include_large=include_large)


def problems_in_form(form: Form) -> tuple[ConvProblem, ...]:
    """The corpus in one of the three forms, chosen by name.

    A driver that takes a ``--form`` flag wants exactly this, and wants it in
    one place: the mapping from the word a user typed to the shape a kernel is
    handed is the thing this whole distinction exists to keep honest.
    """
    return {
        "logical": scaffold_corpus,
        "distconv": halo_corpus,
        "adapter": production_corpus,
    }[form]()


if __name__ == "__main__":  # pragma: no cover - a human-readable dump
    hdr = (f"{'ms/step':>9} {'logical':38s} {'adapter (production)':46s} "
           f"{'AI':>7} {'i64':>4}")
    print(hdr)
    print("-" * len(hdr))
    for p in scaffold_corpus():
        ms = sum(m["ms_per_step"] for m in p.measured)
        print(
            f"{ms:9.3f} {p.label:38s} {p.production_variant.qualified_label:46s} "
            f"{p.arithmetic_intensity():7.0f} {'yes' if p.needs_int64 else '':>4}"
        )
    padded = sum(1 for p in production_corpus() if any(p.padding))
    print(f"\n{len(scaffold_corpus())} ScaFFold problems, "
          f"{len(edge_cases(include_large=True))} synthetic edge cases, "
          f"{len(census_corpus())} measured by census")
    print(f"{padded}/{len(production_corpus())} of the production forms are "
          f"padded; {sum(1 for p in halo_corpus() if any(p.padding))} of the "
          f"DistConv forms are")
