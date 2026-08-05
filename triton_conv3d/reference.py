# SPDX-License-Identifier: (Apache-2.0)
"""Reference implementations and the tolerance policy.

The hard part of testing a reduced-precision kernel is deciding what "correct"
means.  Three standards are used here, in decreasing order of strictness:

1. **Exact.**  Inputs drawn so that every partial sum is exactly representable in
   the working dtype (small integers, bounded reduction length).  The kernel must
   then match the reference *bitwise*.  This is the standard that catches
   indexing, masking and boundary bugs, which a tolerance would hide -- a kernel
   that reads the wrong voxel usually reads a plausible one.

2. **No worse than the incumbent.**  Error against an fp64 reference must not
   exceed MIOpen's error on the same problem by more than a small factor.  This
   is the honest bar for a replacement: we do not have to be better, but we must
   not be worse, and it adapts automatically to shape and reduction length.

3. **Absolute tolerance.**  A dtype- and K-derived bound, used where an fp64
   reference is impractical.  Weakest, and only a backstop.

Everything takes and returns NCDHW tensors in PyTorch's usual convention; the
NDHWC memory format is a layout question, not a semantic one, and is handled by
``contiguous(memory_format=...)`` at the boundary.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F

from .shapes import ConvProblem, Direction

_TORCH_DTYPE = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

#: Mantissa bits, including the implicit leading one.
_MANTISSA_BITS = {
    torch.float64: 53,
    torch.float32: 24,
    torch.bfloat16: 8,
    torch.float16: 11,
}


def torch_dtype(problem: ConvProblem) -> torch.dtype:
    return _TORCH_DTYPE[problem.dtype]


def unit_roundoff(dtype: torch.dtype) -> float:
    """One half ulp, relative -- the classic ``u`` of error analysis."""
    return 2.0 ** -_MANTISSA_BITS[dtype]


# ---------------------------------------------------------------------------
# Operand construction
# ---------------------------------------------------------------------------


def make_inputs(
    problem: ConvProblem,
    device: torch.device | str = "cuda",
    *,
    seed: int = 0,
    exact: bool = False,
    channels_last: bool = True,
    dtype: torch.dtype | None = None,
    density: float | None = None,
) -> dict[str, torch.Tensor]:
    """Input, weight, bias and upstream gradient for one problem.

    With ``exact=True`` the values are small integers chosen so that every
    partial sum of the contraction is exactly representable in ``dtype``; see
    :func:`is_exactly_representable` for when that is possible and
    :func:`exact_density` for the knob that makes it possible at real widths.

    ``density`` thins the *activations* -- ``input`` and ``grad_output`` -- to
    that fraction of nonzeros, and only has an effect under ``exact=True``.  It
    exists because at a real ScaFFold channel width the dense ``{-1,0,1}`` draw
    is not exactly representable in bf16 at all: the forward reduces over
    ``Cin * taps``, which is 27 648 terms at ``Cin = 1024`` regardless of how
    small the volume is made, and a sum of that many random signs runs to a few
    hundred while bf16 holds integers only to 256.  Thinning is the one lever
    that shortens the *realized* reduction without touching the shape, so the
    channel widths, the tile selection and the 512-byte row strides under test
    all stay exactly as ScaFFold runs them.

    The weight is deliberately left dense.  Every one of the ``K`` gather
    addresses then contributes to every output element, so a wrong address is
    masked only by the sparsity of the value it happens to read -- independently
    per element, over millions of them.  Thinning the weight instead would
    multiply whole ``(tap, Cin)`` rows by an exact zero for a whole output
    channel, which is a hole in precisely the coverage this draw exists for.
    """
    dtype = dtype or torch_dtype(problem)
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    thin = exact and density is not None and density < 1.0

    def draw(
        shape: tuple[int, ...], offset: int, activation: bool = False
    ) -> torch.Tensor:
        g = torch.Generator(device=device).manual_seed(seed + offset)
        if exact:
            # {-1, 0, 1}: products are exact and sums stay small.
            t = torch.randint(
                -1, 2, shape, generator=g, device=device, dtype=torch.int8
            ).to(dtype)
            if thin and activation:
                # A separate stream, offset far enough that it cannot collide
                # with any operand's *value* stream: those are seed + 0..3, and
                # a mask drawn from one of them would correlate the zeros with
                # the signs of another tensor.
                gm = torch.Generator(device=device).manual_seed(
                    seed + offset + (1 << 20)
                )
                t = t * (torch.rand(shape, generator=gm, device=device) < density).to(
                    dtype
                )
        else:
            t = torch.randn(shape, generator=g, device=device, dtype=torch.float32)
            t = t.to(dtype)
        return t

    del gen
    fmt = torch.channels_last_3d if channels_last else torch.contiguous_format
    out: dict[str, torch.Tensor] = {
        "input": draw(problem.input_shape, 0, True).contiguous(memory_format=fmt),
        "weight": draw(problem.weight_shape, 1).contiguous(memory_format=fmt),
        "grad_output": draw(problem.output_shape, 2, True).contiguous(
            memory_format=fmt
        ),
    }
    out["bias"] = draw((problem.cout,), 3) if problem.bias else None
    return out


def exact_density(
    problem: ConvProblem,
    direction: Direction = "fwd",
    *,
    dtype: torch.dtype | None = None,
    headroom: float = 4.0,
) -> float:
    """Activation density that keeps the realized result inside the mantissa.

    The arithmetic.  Draw the activations from ``{-1,0,1}`` and then zero all
    but a fraction ``q`` of them, against a dense ``{-1,0,1}`` weight.  Each
    product then has variance ``(4/9) q``, so a reduction over ``K`` terms has
    standard deviation ``(2/3) sqrt(qK)``, and the largest of ``M*N`` such sums
    is about ``sqrt(2 ln(M*N))`` deviations out.  Setting that equal to
    ``2**mantissa / headroom`` and solving for ``q`` gives what is returned.

    Two things fall out that are worth stating.  ``q*K`` -- the number of terms
    that actually contribute to an output element -- comes out at a few hundred
    and is nearly independent of ``K``, so the draw is not "mostly zeros" in the
    sense that matters: every output element is still a sum of hundreds of
    genuine gathers, and 96% of them are nonzero.  And the shape is untouched,
    which is the whole point -- this is what lets the forward's bitwise standard
    run at ``Cin = 1024`` instead of skipping there, which is where it had no
    coverage at all.

    ``headroom`` is against the order-statistic estimate, which is an estimate:
    measured over the eleven corpus forward shapes the realized maxima land at
    58-71 against bf16's limit of 256, so 4.0 buys a genuine 3.6-4.4x rather
    than a nominal 4x.  Returns 1.0 -- no thinning at all -- wherever the dense
    draw already fits, so a caller can pass this unconditionally.
    """
    dtype = dtype or torch_dtype(problem)
    m, n, k = problem.gemm_shape(direction)
    if k <= 0 or m * n <= 0:
        return 1.0
    limit = 2 ** _MANTISSA_BITS[dtype] / headroom
    spread = math.sqrt(2.0 * math.log(max(m * n, 2)))
    return min(1.0, (1.5 * limit / spread) ** 2 / k)


def is_exactly_representable(result: torch.Tensor, dtype: torch.dtype) -> bool:
    """Whether every value in an fp64 reference survives ``dtype`` unchanged.

    With operands in ``{-1, 0, 1}`` every product is exact and every partial sum
    is an integer, so the only question is whether the *realized* magnitudes fit
    in the mantissa.  Asking that of the actual result rather than of the
    worst-case reduction length matters a great deal: bf16 has 8 mantissa bits,
    so a worst-case bound rejects any reduction longer than 256 and would skip
    almost the whole corpus, while a sum of a few hundred random signs is in
    practice tens.  The bitwise standard is the only one that reliably catches an
    off-by-one gather, so it is worth keeping applicable.
    """
    finite = result[torch.isfinite(result)]
    if finite.numel() == 0:
        return True
    integral = torch.equal(finite, finite.round())
    return bool(integral and finite.abs().max().item() < 2 ** _MANTISSA_BITS[dtype])


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _conv(problem: ConvProblem, x, w, b):
    op = F.conv_transpose3d if problem.transposed else F.conv3d
    return op(x, w, b, stride=problem.stride, padding=problem.padding)


def reference(
    problem: ConvProblem,
    operands: dict[str, torch.Tensor],
    direction: Direction = "fwd",
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """The trusted answer, computed in ``dtype`` (fp64 by default).

    fp64 3-D convolution has no fast path on any backend, so this is slow by
    construction -- it is for correctness, not for benchmarking.  Callers that
    need it at ScaFFold's real sizes should reach for ``direction``-specific
    tiling or simply use a smaller problem; every bug we are trying to catch
    reproduces at small sizes.
    """
    device = torch.device(device) if device is not None else operands["input"].device
    x = operands["input"].to(device=device, dtype=dtype)
    w = operands["weight"].to(device=device, dtype=dtype)
    b = operands["bias"]
    b = b.to(device=device, dtype=dtype) if b is not None else None

    if direction == "fwd":
        return _conv(problem, x, w, b)

    gy = operands["grad_output"].to(device=device, dtype=dtype)
    # Ask for only the gradient wanted.  fp64 convolution has no fast path on
    # any backend, so the unwanted one is not a rounding error in the test
    # suite's runtime -- differentiating both roughly doubled it.
    x = x.detach().requires_grad_(direction == "bwd-data")
    w = w.detach().requires_grad_(direction != "bwd-data")
    y = _conv(problem, x, w, b)
    (grad,) = torch.autograd.grad(y, (x if direction == "bwd-data" else w,), gy)
    return grad


def incumbent(
    problem: ConvProblem,
    operands: dict[str, torch.Tensor],
    direction: Direction = "fwd",
) -> torch.Tensor:
    """What MIOpen produces today -- the thing we have to be no worse than."""
    x = operands["input"]
    w = operands["weight"]
    b = operands["bias"]
    if direction == "fwd":
        return _conv(problem, x, w, b)
    gy = operands["grad_output"]
    x = x.detach().requires_grad_(True)
    w = w.detach().requires_grad_(True)
    y = _conv(problem, x, w, b)
    grad_x, grad_w = torch.autograd.grad(y, (x, w), gy)
    return grad_x if direction == "bwd-data" else grad_w


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ErrorReport:
    max_abs: float
    max_rel: float
    rms_rel: float
    #: Number of elements differing at all, and the total.
    n_diff: int
    n_total: int

    @property
    def bitwise(self) -> bool:
        return self.n_diff == 0

    def __str__(self) -> str:
        return (
            f"max_abs={self.max_abs:.3e} max_rel={self.max_rel:.3e} "
            f"rms_rel={self.rms_rel:.3e} diff={self.n_diff}/{self.n_total}"
        )


def compare(actual: torch.Tensor, expected: torch.Tensor) -> ErrorReport:
    """Error of ``actual`` against a higher-precision ``expected``.

    Relative error is normalized by the RMS of ``expected`` rather than
    elementwise, because a convolution output legitimately contains
    near-cancellations whose elementwise relative error is unbounded and
    uninformative.
    """
    a = actual.detach().to(torch.float64)
    e = expected.detach().to(device=a.device, dtype=torch.float64)
    if a.shape != e.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(e.shape)}")
    diff = (a - e).abs()
    scale = e.pow(2).mean().sqrt().item()
    scale = scale if scale > 0 else 1.0
    return ErrorReport(
        max_abs=diff.max().item(),
        max_rel=(diff.max() / scale).item(),
        rms_rel=(diff.pow(2).mean().sqrt() / scale).item(),
        n_diff=int((a != e).sum().item()),
        n_total=a.numel(),
    )


def error_bound(
    problem: ConvProblem,
    expected: torch.Tensor,
    direction: Direction = "fwd",
    *,
    roundings: float = 1.0,
) -> float:
    """Absolute bound on ``max |actual - expected|``.

    Two error sources, and they do not scale with the same quantity:

    - **Accumulation** in fp32 over ``K`` terms.  Rounding there behaves like a
      random walk rather than a worst case, so ``u * sqrt(K)``, and it scales
      with the *typical* magnitude of the result -- its RMS.  A random walk is
      an average, not a bound, so this term carries the 8x safety factor.
    - **The final store** down to bf16, which is up to one ulp of each element
      and so scales with the *largest* element, not the typical one.

    Conflating the two is a real trap, and one this code fell into: measuring
    error relative to the RMS while bounding it in per-element ulps understates
    the bound by the tensor's peak-to-RMS ratio, which for a convolution result
    is comfortably 5x.  MIOpen itself failed that bound on the transposed
    backward-weight, which is how the mistake surfaced -- the tolerance was
    wrong, not the incumbent.

    The 8x used to sit on *both* terms, and that was the opposite mistake.  The
    store is a single deterministic rounding, bounded by half an ulp of the
    element and so by ``u_dtype * peak`` outright -- there is no walk to take a
    safety factor against, and charging four ulps of the peak for it made the
    static bound 12-17x MIOpen's measured error and left the "no worse than the
    incumbent" clause of :func:`assert_close` dead in 46 of 48 cells.  Measured
    here over 78 (problem, direction) cells, MIOpen's own ``max_abs`` in bf16
    and fp16 lands at **0.24-0.66 ulps of the peak** in the forward and
    backward-data -- both of which are bitwise reproducible, i.e. genuinely one
    rounding -- so ``roundings=1`` (a full ulp of the peak) covers a
    single-store kernel with 1.5-4x to spare.  fp32 is the exception and is
    covered by the other term: there ``u_dtype`` is 2**16 smaller, the
    accumulation dominates, and MIOpen sits at 7-8 ulps of a very small ulp.

    ``roundings`` is the number of times a value is rounded into the working
    dtype on its way out, and it is a knob because the incumbent is not always
    1: MIOpen's backward-weight reduces with atomics, so two identical calls
    differ bitwise (verified) and its error *wanders* -- over eight calls on one
    cell it ranged 0.61-1.05 ulps of the peak, disagreeing with itself by 0.72,
    where a single rounding would repeat exactly.  Our backward-weight reduces
    its split-K partials in fp32 and stores once, so it stays at 1.

    What this bound still does not cover, and no tolerance can: a ``tl.dot``
    silently running at ~10-11 mantissa bits sits *under* one bf16 ulp of the
    peak and passes.  Only the bitwise standard rejects that.
    """
    dt = _TORCH_DTYPE[problem.dtype]
    k = problem.gemm_shape(direction)[2]
    e = expected.detach().to(torch.float64)
    rms = e.pow(2).mean().sqrt().item()
    peak = e.abs().max().item()
    accum = unit_roundoff(torch.float32) * math.sqrt(k) * rms
    store = unit_roundoff(dt) * peak
    return 8.0 * accum + 2.0 * roundings * store


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    problem: ConvProblem,
    direction: Direction = "fwd",
    *,
    incumbent_error: ErrorReport | None = None,
    margin: float = 4.0,
    roundings: float = 1.0,
) -> ErrorReport:
    """Apply the strictest standard the situation supports.

    If ``incumbent_error`` is supplied, the bar is "no worse than MIOpen by more
    than ``margin``"; otherwise :func:`error_bound` applies.  The two are combined
    with ``max`` so that a shape where MIOpen happens to be unusually accurate
    cannot make the test stricter than the numerics justify.

    That ``max`` is only worth writing if both arms can win, and for a long time
    only one could: with four ulps of the peak charged for the final store the
    static bound won 46 of 48 cells and the documented standard was never the
    one applied.  With the store term at one ulp (see :func:`error_bound`)
    ``margin * incumbent`` is operative in 71 of the 75 bf16/fp16 cells
    measured.  Which arm wins is closest to a coin toss in fp32, where
    ``u_dtype`` is 2**16 smaller: the store term stops dominating, the bound
    collapses onto MIOpen's own accumulation error, and the two arms come out
    within about 1.2x of each other in either direction.
    """
    report = compare(actual, expected)
    bound = error_bound(problem, expected, direction, roundings=roundings)
    if incumbent_error is not None:
        bound = max(bound, margin * incumbent_error.max_abs)
    if not (report.max_abs <= bound):
        raise AssertionError(
            f"{problem.label} [{direction}]: {report}, bound max_abs <= {bound:.3e}"
            + (f" (incumbent {incumbent_error})" if incumbent_error else "")
        )
    return report
