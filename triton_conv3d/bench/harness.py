# SPDX-License-Identifier: (Apache-2.0)
"""Timing that survives this machine -- and says how well it survived it.

**What this node does.**  Measured on GPU 2 on 2026-08-03 -- 47,686 consecutive
10-call blocks of ``conv 64->64 k3 @ 130^3`` over 14 minutes, with ``rocm-smi``
sampled concurrently from a sibling process:

* steady-state dispersion is **CoV 0.56%**, IQR/median 0.39%, p99/p01 = 1.022;
* the *sequential* protocol -- measure A for a while, then B -- simulated out of
  that stream, where both windows are the same kernel and the true ratio is
  therefore exactly 1, has a **worst error of 0.32%** over 206 replications at
  2 s separation, and 0.14 / 0.24 / 0.18% at 5 / 15 / 60 s;
* a deliberate **90-second idle gap** moves the measured time by **0.13%**, so
  there is no thermal ramp to catch;
* ``sclk`` under load stays in 1396-1451 MHz and junction temperature rises
  49 -> 66 C, and neither correlates with the time (r = -0.10 and -0.05).  The
  device is power-capped at 550 W and clocks *stably* at that cap.

So the clock is not what threatens a comparison here, and neither is elapsed
time between two measurements in one process.  What does threaten one is a
**neighbour**: a foreign tenant on the device inflated three cells **2.6x**,
with per-cell spreads 20-100x their quiet-device values.  That is common-mode
within a round -- every arm pays it at once -- so :func:`interleaved` survives it
where a between-run comparison does not.  It costs nothing, so it stays.  It is
not, however, a substitute for an error bar.

**What actually goes wrong here is the instrument, twice.**

1. Below about 0.15 ms the per-iteration event pair is a material part of what
   it reports.  Measured: ``hipEventRecord`` costs **9.5 us of host time**, and
   a block with an event between every iteration reports **16-26% more** than
   the same kernel's wall-clock throughput (``convT 1024->512 @ 8^3`` forward:
   0.0718 ms with events, 0.0617 bracketed, 0.0568 by wall clock).  The tax is
   per-arm, not common-mode, so it does *not* cancel in a ratio.  This harness
   now widens the event interval to cover several calls whenever the kernel is
   short enough for that to matter -- one common width for every arm of a
   comparison, from a rule that depends only on the measured duration -- and
   reports the residual it measures against an event-free bracket.  Checked
   against a wall-clock throughput reference at 512^2 / 1024^2 / 2048^2 / 4096^2
   bf16 GEMM: the harness now reads 0.96-1.08x of it across three orders of
   magnitude, where at the smallest of those it used to read **1.52x**.
2. ``iters`` was never a neutral knob.  The first call after a synchronize pays
   a queue restart, so ``iters=1`` over-reports by 3% at 1.4 ms and by **42%**
   at 0.07 ms.  The adaptive path picks ``iters`` from the measured duration and
   never leaves it at 1 for a small kernel.

**What the timed region contains, and what it does not.**  Below about 0.15 ms
the *host* is the pacer: a launch costs 2-37 us of Python and dispatch on this
node, against kernels of 28-68 us, so an event-timed loop of ``fn()`` measures
the launcher and the kernel together and cannot separate them.  :func:`capture`
puts ``chunk`` back-to-back calls behind one CUDA graph, which contains the
device work and none of the host work, so replaying it measures the kernel
alone.  Measured, per call, on ``convT 1024->512 @ 8^3``:

=============================  =======  ========  =============
arm                            eager    kernel    host launch
=============================  =======  ========  =============
``convT`` fwd, Triton          0.0421   0.0282    13.9 us
``convT`` fwd, MIOpen          0.0699   0.0683    **1.6 us**
``convT`` bwd-data, Triton     0.0539   0.0350    18.9 us
``convT`` bwd-data, MIOpen     0.0761   0.0395    **36.6 us**
``convT`` bwd-weight, Triton   0.0712   0.0542    17.0 us
``convT`` bwd-weight, MIOpen   0.0839   0.0504    33.4 us
=============================  =======  ========  =============

Those six host costs differ by **23x**, so *leaving* the launcher in is as much
a per-arm instrument as taking it out asymmetrically would be: the MIOpen
control for a backward direction pays an autograd-engine walk the Triton arm
does not, and the Triton arm pays a tuned-table lookup MIOpen does not.  At
these sizes the two do not cancel -- the same three cells read 1.659x / 1.414x /
1.176x with the launcher in and **2.420x / 1.130x / 0.931x** without it, i.e.
the launcher-inclusive number is 1.5x too *low* on the forward and 1.3x too
*high* on the weight gradient.  Hence the rule this module enforces: a graph is
chosen for the whole comparison or for none of it, and ``chunk`` -- like
``group`` -- is a function of the shortest arm's duration alone, never of a
per-arm measurement.  Above ~40 ms per call the host cost is under 0.2% of
either arm and no graph is used.

**And every number now carries its precision.**  ``rounds`` and ``iters`` are
chosen online from what has already been observed, stopping when the reported
statistic reaches a stated relative precision or when a wall-clock budget is
exhausted, and :class:`Measurement` says which of the two happened.  A cell that
stopped on the budget with a wide interval is visibly different from one that
converged.  That distinction is worth more than the time saved: four of this
project's five retracted results were numbers quoted without one.

Everything here is in-process.  Sub-process benchmarking adds interpreter start,
allocator state and MIOpen database warmth as confounders, none of which the
kernel controls.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
import warnings
from typing import Callable, Iterable, Mapping, Sequence

import torch

#: Big enough to evict MI300A's 256 MiB infinity cache.
_FLUSH_BYTES = 512 * 1024 * 1024
_flush_buffer: torch.Tensor | None = None

#: Fraction of a kernel's own time the event instrument is allowed to add before
#: :func:`interleaved` starts grouping iterations behind one event interval.
#: 2% because the smallest published per-cell differences are around 3%.
_TAX_BUDGET = 0.02

#: Wall time one timed block should aim for.  Large enough that the queue-restart
#: transient on the first iteration is a small fraction of the block, small
#: enough that a round is cheap.
_BLOCK_TARGET_MS = 15.0
_MAX_ITERS = 512

#: Student-t 97.5th percentile by degrees of freedom, so an interval can be
#: quoted without a scipy dependency.  Index 0 is unused.
_T975 = (
    math.nan,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
)

#: Asymptotic ratio of the standard error of a sample median to that of the
#: sample mean, for normally distributed data: ``sqrt(pi/2)``.  Round values are
#: already medians over ``iters`` calls, so they are close to normal; where they
#: are heavier-tailed this factor is conservative (the median's true SE is then
#: smaller than the formula says), which is the direction to err in.
_MEDIAN_SE_FACTOR = 1.2533141373155003


def _t975(n: int) -> float:
    if n < 2:
        return math.inf
    return _T975[n - 1] if n - 1 < len(_T975) else 1.96


def _half_width(values: Sequence[float]) -> float:
    """95% half-width for the *median* of ``values``, in the same units.

    Closed form rather than a bootstrap: it is deterministic (this project
    re-runs its numbers and compares them), it needs no RNG seed to be
    reproducible, and at the round counts in use (4-16) a percentile bootstrap
    of a median cannot produce an interval wider than the observed range, which
    understates exactly when it matters most.
    """
    n = len(values)
    if n < 2:
        return math.inf
    sd = statistics.stdev(values)
    return _t975(n) * _MEDIAN_SE_FACTOR * sd / math.sqrt(n)


def flush_caches(device: torch.device | str = "cuda") -> None:
    """Evict the cache hierarchy so a measurement starts cold.

    Matters for the memory-bound directions: a 16 MiB working set measured hot
    reports bandwidth the same kernel will never see inside a real step, where
    everything upstream has already flushed it.

    Two things to know before trusting it.  It **works** -- the first iteration
    after a flush is 1.50-1.54x the hot time at ``2048^2`` bf16 GEMM and at the
    transposed sites -- but a caller that flushes once per block and then reports
    the *median* over ``iters`` calls throws that one cold sample away: measured
    ``median_moved_by_flush`` is 0.99-1.01 at every real workload.  Use
    ``iters=1`` (what the adaptive path does when ``flush=True``) or read
    :attr:`Measurement.cold`, which this module records for exactly this reason.
    """
    global _flush_buffer
    want = torch.device(device)
    if want.index is None and want.type == "cuda":
        # ``torch.device("cuda")`` carries no index but a tensor created on it
        # does, so a naive ``!=`` is always true and this function used to
        # reallocate 512 MiB on every call -- one extra 512 MiB block live at a
        # time, and an allocation on the critical path of every timed round.
        want = torch.device("cuda", torch.cuda.current_device())
    if _flush_buffer is None or _flush_buffer.device != want:
        _flush_buffer = torch.empty(_FLUSH_BYTES, dtype=torch.uint8, device=want)
    _flush_buffer.zero_()


@dataclasses.dataclass(frozen=True)
class Measurement:
    """Per-round times for one variant, in milliseconds per call.

    The headline statistic is still :attr:`median`.  What is new is that it
    comes with :attr:`half_width` -- a 95% interval -- and with :attr:`stop`,
    which says whether the measurement reached its precision target or ran out
    of wall clock.  Print it; do not quote the median alone.
    """

    name: str
    rounds: tuple[float, ...]
    #: Per round, ``block mean / per-iteration median``.  See :func:`_time_block`:
    #: greater than 1 means the host failed to keep the queue full and the GPU
    #: idled between launches.  Reported rather than hidden, because the previous
    #: version of this harness folded that idle time into the kernel time and
    #: produced spreads of up to 2363% that were then misdiagnosed twice.
    #:
    #: It is a *skew* statistic and it is blind to the uniform case: if every
    #: iteration is inflated by the same launch gap it reads exactly 1.00.  Use
    #: :attr:`tax_frac`, which is measured against an event-free bracket and is
    #: therefore independent, for that question.
    stalls: tuple[float, ...] = ()
    #: Calls per timed block, and calls per event interval within it.
    iters: int = 0
    group: int = 1
    #: Per round, the *first* iteration of the block.  With ``flush=True`` that
    #: is the only cold sample there is.
    firsts: tuple[float, ...] = ()
    #: ``per-iteration-event time - event-free bracket time``, ms per call,
    #: measured for this variant during calibration.  The instrument's own cost.
    tax_ms: float = 0.0
    #: Why the measurement stopped: ``fixed`` (caller pinned ``rounds``),
    #: ``converged``, ``budget`` or ``max_rounds``.
    stop: str = "fixed"
    #: True when every variant occupied every position, and every ordered
    #: adjacency occurred, equally often -- i.e. ``rounds`` was a multiple of
    #: ``2 * len(variants)``.
    balanced: bool = True
    seconds: float = 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.rounds)

    @property
    def best(self) -> float:
        return min(self.rounds)

    @property
    def cold(self) -> float:
        """Median first-iteration time; the cold number when ``flush`` is on."""
        return statistics.median(self.firsts) if self.firsts else self.median

    @property
    def spread(self) -> float:
        """Relative range across rounds.  **Grows with** ``rounds`` by construction.

        Kept because every recorded result JSON has it, but it is not a measure
        of how much the machine moved: the expected range of ``n`` samples grows
        like ``d2(n)`` even on a perfectly stationary device.  Measured on this
        node with one kernel held constant for 14 minutes, the median of this
        statistic runs 0.23% at 2 rounds, 0.63% at 6, 0.98% at 20 and 2.70% at
        100 -- all of it arithmetic.  Compare :attr:`rel_half_width` instead,
        which is an interval and does not have that defect.
        """
        return (
            (max(self.rounds) - min(self.rounds)) / self.median if self.rounds else 0.0
        )

    @property
    def cov(self) -> float:
        """Coefficient of variation across rounds; comparable between runs."""
        if len(self.rounds) < 2:
            return 0.0
        return statistics.stdev(self.rounds) / statistics.fmean(self.rounds)

    @property
    def half_width(self) -> float:
        """95% half-width on :attr:`median`, in ms."""
        return _half_width(self.rounds)

    @property
    def rel_half_width(self) -> float:
        m = self.median
        return self.half_width / m if m > 0 else math.inf

    @property
    def converged(self) -> bool:
        return self.stop in ("converged", "fixed")

    @property
    def tax_frac(self) -> float:
        """Instrument cost as a fraction of the reported time.

        Independent of the numbers the median came from -- it is the gap between
        an event-per-iteration block and an event-free bracket of the same
        kernel -- which is precisely what :attr:`stall_ratio` cannot see.
        """
        m = self.median
        return self.tax_ms / m if m > 0 else 0.0

    @property
    def stall_ratio(self) -> float:
        """Worst launch-gap inflation seen in any round; 1.0 is a clean queue."""
        return max(self.stalls) if self.stalls else 1.0

    def __str__(self) -> str:
        stall = f", stall {self.stall_ratio:.2f}x" if self.stall_ratio > 1.05 else ""
        tax = f", instrument {self.tax_frac:+.1%}" if abs(self.tax_frac) > 0.02 else ""
        mark = "" if self.converged else f" [{self.stop}]"
        return (
            f"{self.name}: {self.median:.4f} +-{self.rel_half_width:.1%} ms "
            f"({len(self.rounds)}x{self.iters}{mark}, best {self.best:.4f}"
            f"{stall}{tax})"
        )


@dataclasses.dataclass(frozen=True)
class Ratio:
    """A paired ratio of two variants, with the interval that makes it a claim.

    Paired **per round**, not median-over-median: the two arms of a round were
    measured seconds apart under the same device state, so a common-mode
    excursion divides out of every pair before anything is averaged.  That is
    the property :func:`interleaved` exists to buy, and taking a ratio of two
    independently-reduced medians throws it away.
    """

    numerator: str
    denominator: str
    point: float
    lo: float
    hi: float
    n: int

    @property
    def rel_half_width(self) -> float:
        return (self.hi - self.lo) / (2 * self.point) if self.point > 0 else math.inf

    @property
    def significant(self) -> bool:
        """Does the interval exclude 1.0?  If not, there is no measured win."""
        return self.lo > 1.0 or self.hi < 1.0

    def __str__(self) -> str:
        star = "" if self.significant else "  (consistent with no difference)"
        return (
            f"{self.numerator}/{self.denominator} = {self.point:.3f}x "
            f"[{self.lo:.3f}, {self.hi:.3f}], n={self.n}{star}"
        )


def ratio(numerator: Measurement, denominator: Measurement) -> Ratio:
    """Paired ratio ``numerator / denominator`` with a 95% interval."""
    n = min(len(numerator.rounds), len(denominator.rounds))
    pairs = [
        numerator.rounds[i] / denominator.rounds[i]
        for i in range(n)
        if denominator.rounds[i] > 0
    ]
    if not pairs:
        return Ratio(numerator.name, denominator.name, math.nan, math.nan, math.nan, 0)
    logs = [math.log(p) for p in pairs]
    point = math.exp(statistics.median(logs))
    hw = _half_width(logs)
    if not math.isfinite(hw):
        return Ratio(numerator.name, denominator.name, point, 0.0, math.inf, len(pairs))
    return Ratio(
        numerator.name,
        denominator.name,
        point,
        point * math.exp(-hw),
        point * math.exp(hw),
        len(pairs),
    )


def _time_block(
    fn: Callable[[], object], iters: int, group: int = 1
) -> tuple[float, float]:
    """Return ``(median ms per call, stall ratio)`` for ``iters`` calls.

    Events rather than the wall clock: the launch is asynchronous, so a wall
    clock measures the host's ability to enqueue until something forces a
    synchronize, and the forced synchronize is then part of the measurement.

    But bracketing the *whole block* with two events, as this used to, has the
    same disease one level up: if the host cannot keep the queue full the GPU
    goes idle between launches, and that idle time is silently attributed to the
    kernel.  On a contended node it produced per-round spreads of 250-2363% that
    were diagnosed as host jitter, then as a rogue tenant, before turning out to
    be a duplicate driver process of our own.

    So time each iteration separately and return the **median**, which rejects a
    stalled launch instead of averaging it in, alongside the ratio of the old
    block-mean to that median.  A ratio near 1 means the queue stayed full and
    the two agree; a large ratio means the measurement is launch-bound and the
    number should not be read as kernel time.

    ``group`` widens the event interval to ``group`` calls.  An event costs
    ~9.5 us of host time and shows up in the reported number below ~0.15 ms of
    kernel, where it inflated ``convT`` sites by 16-26%; grouping divides that
    by ``group`` while keeping enough samples per block for the median to still
    reject a stall.  ``group=1`` is the historical behaviour and is what a large
    kernel gets, because there the tax is already under a tenth of a percent.
    """
    marks = _blocked_events(fn, iters, group)
    n = len(marks) - 1
    per_iter = [marks[i].elapsed_time(marks[i + 1]) / group for i in range(n)]
    median = statistics.median(per_iter)
    block_mean = marks[0].elapsed_time(marks[n]) / (n * group)
    return median, (block_mean / median if median > 0 else 1.0)


def _blocked_events(fn: Callable[[], object], iters: int, group: int) -> list:
    groups = max(1, iters // group)
    marks = [torch.cuda.Event(enable_timing=True) for _ in range(groups + 1)]
    torch.cuda.synchronize()
    for g in range(groups):
        marks[g].record()
        for _ in range(group):
            fn()
    marks[groups].record()
    torch.cuda.synchronize()
    return marks


def _time_block_full(
    fn: Callable[[], object], iters: int, group: int
) -> tuple[float, float, float]:
    """``(median, stall ratio, first sample)`` -- the first is the cold one."""
    marks = _blocked_events(fn, iters, group)
    n = len(marks) - 1
    per = [marks[i].elapsed_time(marks[i + 1]) / group for i in range(n)]
    median = statistics.median(per)
    block_mean = marks[0].elapsed_time(marks[n]) / (n * group)
    return median, (block_mean / median if median > 0 else 1.0), per[0]


def _bracket(fn: Callable[[], object], iters: int) -> float:
    """Per-call ms with two events around the whole block and none inside.

    The event-free control the instrument tax is measured against.
    """
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


@dataclasses.dataclass(frozen=True)
class Plan:
    """What calibration decided for one variant, so it can be reported."""

    per_call_ms: float
    iters: int
    group: int
    warmup: int
    tax_ms: float


#: Above this per-call time the event instrument is provably under 1%, so the
#: tax probe -- ten extra blocks -- is not worth running.  It would cost 450 s
#: on the 45 s cliff cell alone.
_TAX_PROBE_MAX_MS = 1.0

#: Floor on the cost of one event interval, in ms.  Measured on this device with
#: ``torch.cuda._sleep`` at four durations spanning 10 us to 285 us per call:
#: **2.85, 2.85, 2.85 and 3.09 us**, i.e. flat in the kernel size, which is what
#: makes it a property of the instrument rather than of the workload.  A real
#: kernel can cost *more* than this, because the host also pays ~9.5 us per
#: ``record()`` and a launch path expensive enough to make the host the pacer
#: turns that into device idle -- so this is a floor, and the per-arm probe
#: raises it.  It exists so that a probe which happens to measure near zero
#: cannot leave a tiny kernel ungrouped.
_EVENT_INTERVAL_MS = 0.00285

#: Group only once the instrument is worth more than twice its budget, i.e. more
#: than 4% of the kernel by default.  The band below that is left alone because
#: grouping is not free: it averages ``group`` calls behind one event interval,
#: so it also *reduces* the median's ability to reject a stalled launch.  At
#: 0.08 ms, forcing a group of 2 moved the reported time 7% the wrong way.
_GROUP_TRIGGER = 2.0


def _measure_tax(
    fn: Callable[[], object], iters: int, group: int = 1, reps: int = 5
) -> float:
    """Per-call cost of the event instrument: events minus an event-free bracket.

    Measured as a *paired* difference -- ev, br, ev, br, ... -- rather than as a
    difference of two separately-collected medians.  The two arms of each pair
    are adjacent in time, so a device-wide excursion cancels inside the pair
    instead of landing in the estimate; taken unpaired, this estimate was noisy
    enough to pick wildly different groups for byte-identical work.
    """
    diffs = []
    for _ in range(reps):
        ev = _time_block(fn, iters, group)[0]
        br = _bracket(fn, iters)
        diffs.append(ev - br)
    return statistics.median(diffs)


def _probe(
    fn: Callable[[], object],
    *,
    pinned_warmup: int | None,
    warmup_s: float,
    warmup_min: int,
    warmup_max: int,
    warmup_hard_s: float,
    block_ms: float,
    max_iters: int,
    need_duration: bool,
) -> tuple[float, int]:
    """Warm one variant and return ``(per-call ms, warmup calls issued)``.

    The first call absorbs whatever one-off the variant has -- MIOpen's find is
    **8.2 s** on ``conv 64->64 k3 @ 130^3`` against a 1.65 ms steady state, and
    Triton's JIT is a compile -- so it is never the call that decides anything.

    A pinned ``warmup`` issues exactly that many calls and nothing else, so a
    caller that also pins ``iters`` gets precisely the pre-adaptive call
    sequence and a re-capture stays comparable with what is on disk.
    """
    if pinned_warmup is not None:
        for _ in range(pinned_warmup):
            fn()
        torch.cuda.synchronize()
        if not need_duration:
            return 0.0, pinned_warmup
        probe = max(4, min(max_iters, 8))
        return max(_bracket(fn, probe), 1e-6), pinned_warmup

    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    rough = max((time.perf_counter() - t0) * 1e3, 1e-4)

    # Warm to a settled state.  Measured: Triton needs ~5 calls to come within
    # 0.5% of steady, MIOpen ~4 after its find; both are cheap when the kernel
    # is small and unaffordable when it is 45 s, which is what the hard cap is.
    n = min(warmup_max, max(warmup_min, int(warmup_s * 1e3 / rough)))
    if n * rough > warmup_hard_s * 1e3:
        n = max(0, int(warmup_hard_s * 1e3 / rough))
    for _ in range(n):
        fn()
    torch.cuda.synchronize()

    # ``rough`` is one call bracketed by two synchronizes and it over-states a
    # small kernel badly -- 0.37 ms for a 0.017 ms GEMM, because the sync and
    # the queue restart are most of it.  Sizing ``iters`` off that number gave
    # blocks 20x too short and left the instrument tax at 6.9% instead of 2%.
    # So re-estimate from an event-free bracket, which is the same quantity the
    # tax is measured against.
    d = rough
    if rough < block_ms:
        probe = max(4, min(max_iters, round(block_ms / rough)))
        d = max(_bracket(fn, probe), 1e-6)
    return d, n + 2


def per_call_ms(
    fn: Callable[[], object],
    *,
    warmup_s: float = 0.05,
    warmup_min: int = 5,
    warmup_max: int = 200,
    warmup_hard_s: float = 2.0,
    block_ms: float = _BLOCK_TARGET_MS,
    max_iters: int = _MAX_ITERS,
    settle_s: float = 0.25,
    settle_calls: int = 5,
) -> float:
    """One warmed, event-free estimate of a callable's per-call time.

    The same probe :func:`interleaved` runs internally, exposed because a caller
    that is about to decide *how* to time something -- whether to put it in a
    graph, and how many calls to put in one -- has to know roughly what it costs
    first, and that decision has to be made from a quantity measured the same
    way for every arm.

    With one difference, and it cost an afternoon: :func:`_probe` absorbs *one*
    one-off call before it estimates anything, and one is not always enough.
    The first ``256^2`` bf16 ``matmul`` in a fresh process measured **653 ms** on
    its *second* call, because rocBLAS/hipBLASLt loads its kernel library
    lazily and does it after the first launch.  A decision made from that number
    puts a 0.02 ms kernel on the eager path with ``iters=1``.  So this settles
    first, bounded by time rather than by a call count so that a 45-second
    kernel is called once and a 20-microsecond one five times.

    It is a probe, not a measurement: one bracketed block, no rounds, no
    interval.  Do not publish it.
    """
    t0 = time.perf_counter()
    for _ in range(settle_calls):
        fn()
        torch.cuda.synchronize()
        if time.perf_counter() - t0 > settle_s:
            break
    return _probe(
        fn,
        pinned_warmup=None,
        warmup_s=warmup_s,
        warmup_min=warmup_min,
        warmup_max=warmup_max,
        warmup_hard_s=warmup_hard_s,
        block_ms=block_ms,
        max_iters=max_iters,
        need_duration=True,
    )[0]


def _common_group(
    durations: Sequence[float], *, tax_budget: float, max_iters: int
) -> int:
    """One event-interval width for **every** arm of a comparison.

    Two things are load-bearing here and both were bought with a wrong answer.

    *The group is a function of the duration alone*, not of a per-arm
    measurement of the instrument tax.  A group derived from a per-arm probe
    made two byte-identical arms pick 12 and 2 in the same call -- a 4%
    difference in residual instrument cost, and therefore a 4% bias in a ratio
    whose true value was exactly 1.000.  That is the failure this whole
    exercise is about, reintroduced by the fix for it;
    ``test_a_paired_ratio_of_two_identical_arms_covers_one`` caught it.

    *And it is common to the whole call*, taken from the **shortest** arm, so
    that even when two arms are far apart in duration -- or merely far enough
    apart to straddle a power-of-two boundary, which byte-identical arms did --
    they are measured with the same ruler.  ``iters`` stays per-arm, because
    that is what the five orders of magnitude in this corpus need; the *width
    of the event interval* is what has to match.
    """
    d = min(durations)
    need = _EVENT_INTERVAL_MS / (tax_budget * d)
    if need <= _GROUP_TRIGGER:
        return 1
    return max(1, min(max_iters // 4, 1 << math.ceil(math.log2(need))))


def _size_block(d: float, group: int, block_ms: float, max_iters: int) -> int:
    """Calls per timed block: about ``block_ms`` of work, at least 4 samples."""
    if group == 1:
        return max(1, min(max_iters, round(block_ms / d)))
    samples = max(4, min(max_iters // group, round(block_ms / (d * group))))
    return group * samples


def _williams(n: int) -> list[int]:
    """``0, 1, n-1, 2, n-2, ...`` -- the first row of a Williams square.

    Its successive differences are ``1, -2, 3, -4, ...`` mod ``n``, which are
    all distinct, and that is what makes the rotations of this row
    *row-complete* for even ``n``: every ordered pair of variants occurs
    adjacent equally often instead of only the cyclically adjacent ones.
    """
    out, lo, hi = [], 0, n
    while lo < hi:
        out.append(lo)
        lo += 1
        if lo < hi:
            hi -= 1
            out.append(hi)
    return out


def _order(names: list[str], r: int) -> list[str]:
    """A position- and adjacency-balanced order for round ``r``.

    The old rule rotated by one position per round.  That balances *positions*
    but it preserves *adjacency*: with three or more variants, B always ran
    immediately after A, so whatever A left in the caches was a constant charged
    to B and averaged out of nothing.  Measured on the adversarial case -- a
    1 GiB cache-polluting arm plus two arms doing byte-identical work, 40
    replications -- cyclic rotation reported the two identical arms **2.8%
    apart** (sd 4.3%), while this rule reported them 0.2% apart and a uniformly
    random order 0.7%.  2.8% is larger than several of the per-cell differences
    this project publishes.

    Rotating a Williams row every *second* round and reversing it on odd rounds
    gives, over ``2 * len(names)`` rounds, every variant in every position
    exactly twice and every ordered adjacency equally often -- and for an even
    number of variants (``conv_bench`` runs four) *every* ordered pair occurs,
    not just the cyclic ones.  Deterministic, so a capture is reproducible.
    """
    n = len(names)
    base = _williams(n)
    k = (r // 2) % n
    seq = [(i + k) % n for i in base]
    if r % 2:
        seq = seq[::-1]
    return [names[i] for i in seq]


# ---------------------------------------------------------------------------
# Taking the launcher out of the timed region
# ---------------------------------------------------------------------------

#: Cost of one ``cudaGraphLaunch``, in ms per replay, measured on this device by
#: fitting ``per_call(chunk) = kernel + cost / chunk`` to graphs of 1, 2, 4, 8,
#: 16 and 32 calls at ``convT 1024->512 @ 8^3``:
#:
#: ===========================  ============
#: arm                          fitted cost
#: ===========================  ============
#: ``convT`` fwd, Triton        3.9 us
#: ``convT`` fwd, MIOpen        **12.8 us**
#: ``convT`` bwd-data, Triton   4.4 us
#: ``convT`` bwd-data, MIOpen   4.1 us
#: ===========================  ============
#:
#: The constant below is the **worst** of those, not their mean, because the
#: quantity that has to be bounded is the residual on whichever arm pays most --
#: and because a per-arm estimate is exactly the mistake :func:`_common_group`
#: exists to document: two byte-identical arms picked different instruments and
#: read 4% apart.  It is a property of the launcher, not of the workload.
_REPLAY_COST_MS = 0.0128

#: Fraction of the *shortest* arm's per-call time the residual replay cost is
#: allowed to reach.  1% because the smallest published per-cell differences are
#: around 3% and the target precision is 2%.
_REPLAY_BUDGET = 0.01

#: A graph holding this many calls is already 128 x 0.0128 us = 0.1 us per call,
#: below the event instrument's own floor; more would only cost capture time.
_MAX_CHUNK = 128

#: Above this per-call time no graph is used: the largest host launch cost
#: measured on this node is 0.08 ms (the autograd engine's, on the MIOpen
#: backward control), which at 40 ms per call is 0.2% of either arm -- smaller
#: than the harness's own target precision, so the exclusion is not worth the
#: capture.  Below it the host cost reaches 190% of the kernel and decides the
#: answer.
_GRAPH_MAX_MS = 40.0

_capture_stream: torch.cuda.Stream | None = None


class CaptureError(RuntimeError):
    """A callable could not be put in a CUDA graph, or the graph came out empty.

    Raised rather than swallowed: a caller that silently fell back to eager for
    *one* arm would be comparing a launcher-exclusive number against a
    launcher-inclusive one, which at these sizes is worth up to 1.4x.  The
    decision to fall back belongs to the comparison, not to an arm.
    """


def capture_stream() -> torch.cuda.Stream:
    """The one side stream every capture in this process uses.

    It has to be shared, and callers have to build their autograd graphs on it.
    ``torch.autograd.grad`` refuses to be captured with *"autograd node
    ``ConvolutionBackward0`` has a stale reference to the default stream"* when
    the forward that created the node ran on the default stream while the
    capture runs on another -- so the MIOpen control for a backward direction,
    which is a real forward plus :func:`torch.autograd.grad`, has to have its
    forward built here.  Handing every caller the same stream is what makes that
    possible without each of them inventing one.
    """
    global _capture_stream
    if _capture_stream is None:
        _capture_stream = torch.cuda.Stream()
    return _capture_stream


class on_capture_stream:  # noqa: N801 - a context manager, spelled like one
    """Run every timed call of one comparison on :func:`capture_stream`.

    Not a nicety.  The MIOpen control for a backward direction is
    :func:`torch.autograd.grad` over a forward graph that had to be built on the
    capture stream (see :func:`capture_stream`), and the autograd engine
    synchronizes when the node's recorded stream is not the caller's current
    one.  Measured, ``convT`` backward at ``1024->512 @ 8^3``: the MIOpen arm
    read **0.148 ms** when timed from the default stream and **0.113 ms** from
    the stream its graph was built on -- a **35 us** cross-stream tax that the
    Triton arm, which has no autograd graph, does not pay.  At those sizes that
    alone is worth 1.3x, and it is per-arm.

    So the stream is a property of the *comparison*, exactly like ``group`` and
    ``chunk``: one stream, entered once, for every arm and both launcher
    policies.  Above ~0.3 ms the tax is under 1% and the two agree, which is why
    it was invisible until the transposed cells.
    """

    def __enter__(self):
        self._stream = capture_stream()
        self._ctx = torch.cuda.stream(self._stream)
        self._stream.wait_stream(torch.cuda.current_stream())
        self._ctx.__enter__()
        return self._stream

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        torch.cuda.current_stream().wait_stream(self._stream)
        torch.cuda.synchronize()
        return False


@dataclasses.dataclass(frozen=True)
class Captured:
    """``chunk`` back-to-back calls of one callable, behind one graph replay.

    Calling this runs ``chunk`` calls' worth of device work and *no* host work
    beyond one ``cudaGraphLaunch``, so a harness that times it is timing the
    kernel.  :attr:`chunk` is the divisor a caller needs to get back to ms per
    call -- deliberately not hidden, because every relative quantity the harness
    reports (``rel_half_width``, ``cov``, :func:`ratio`) is scale-invariant and
    only the absolute times need it.
    """

    graph: object
    chunk: int
    #: Per-call ms of the eager callable, as measured before capture.  Kept so a
    #: caller can report what the launcher was worth.
    eager_ms: float = 0.0

    def __call__(self) -> None:
        self.graph.replay()


def capture(fn: Callable[[], object], chunk: int = 1, *, warmup: int = 3) -> Captured:
    """Put ``chunk`` calls of ``fn`` in a CUDA graph, or raise :class:`CaptureError`.

    Everything -- the warmup and the capture -- runs on :func:`capture_stream`,
    for the autograd reason given there.

    The warmup is not optional and it is not only PyTorch's lazy-init
    requirement: with ``cudnn.benchmark`` on, MIOpen's find is **8.2 s** and
    happens on the first call, and a find inside a capture would synchronize and
    abort it.  Three calls are enough (measured: MIOpen settles by call ~4,
    Triton by ~5, and the JIT compile is on call 1).

    An *empty* graph is treated as a failure.  PyTorch only warns -- "The CUDA
    Graph is empty.  This usually means that the graph was attempted to be
    captured on wrong device or stream" -- and a caller that ignored the warning
    would publish the cost of ``cudaGraphLaunch`` as a kernel time, which is the
    fastest wrong answer available.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    s = capture_stream()
    try:
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.cuda.graph(g, stream=s):
                for _ in range(chunk):
                    fn()
        torch.cuda.synchronize()
    except CaptureError:
        raise
    except Exception as exc:  # noqa: BLE001 - any capture refusal
        torch.cuda.synchronize()
        raise CaptureError(f"{type(exc).__name__}: {exc}") from exc
    for w in caught:
        if "graph is empty" in str(w.message).lower():
            raise CaptureError("the captured graph is empty")
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return Captured(g, chunk)


def common_chunk(
    durations: Sequence[float],
    *,
    cost_ms: float = _REPLAY_COST_MS,
    budget: float = _REPLAY_BUDGET,
    max_chunk: int = _MAX_CHUNK,
) -> int:
    """Calls per graph, for **every** arm of a comparison.

    Same shape and same reasoning as :func:`_common_group`, one level up: a
    function of the shortest arm's measured duration alone, so that two arms are
    never measured with two different rulers.  A graph replay costs 3.9-12.8 us
    of device time whatever is inside it, which is 45% of a 0.028 ms kernel at
    ``chunk = 1``; dividing it by ``chunk`` brings it under ``budget`` of the
    arm it hurts most.
    """
    d = min(durations)
    if d <= 0:
        return 1
    need = cost_ms / (budget * d)
    if need <= 1.0:
        return 1
    return max(1, min(max_chunk, 1 << math.ceil(math.log2(need))))


def graph_is_worthwhile(
    durations: Sequence[float], max_ms: float = _GRAPH_MAX_MS
) -> bool:
    """Is the host launch cost big enough to be worth capturing away?

    Above :data:`_GRAPH_MAX_MS` the measured worst-case host launch cost
    (0.08 ms, the autograd engine's) is under 0.2% of either arm, so eager
    timing is already launcher-exclusive to within a fifth of the target
    precision and the capture would only cost wall clock.
    """
    return min(durations) <= max_ms


def time_callable(
    fn: Callable[[], object],
    *,
    warmup: int | None = None,
    iters: int | None = None,
    rounds: int | None = None,
    flush: bool = False,
    **kwargs,
) -> Measurement:
    """Time a single callable.  Prefer :func:`interleaved` for comparisons."""
    return interleaved(
        {"fn": fn}, warmup=warmup, iters=iters, rounds=rounds, flush=flush, **kwargs
    )["fn"]


def interleaved(
    variants: Mapping[str, Callable[[], object]],
    *,
    warmup: int | None = None,
    iters: int | None = None,
    rounds: int | None = None,
    flush: bool = False,
    target_rel: float = 0.02,
    budget_s: float = 20.0,
    hard_budget_s: float = 300.0,
    min_rounds: int = 4,
    floor_rounds: int = 3,
    max_rounds: int = 64,
    block_ms: float = _BLOCK_TARGET_MS,
    max_iters: int = _MAX_ITERS,
    tax_budget: float = _TAX_BUDGET,
    warmup_s: float = 0.05,
    warmup_min: int = 5,
    warmup_max: int = 200,
    warmup_hard_s: float = 2.0,
    measure_tax: bool = True,
) -> dict[str, Measurement]:
    """Time several variants against each other, and say how precisely.

    Each round runs every variant once, in a position- *and* adjacency-balanced
    order (see :func:`_order`), and the arms are compared round by round so that
    a neighbour process -- which inflates every arm of a round at once -- divides
    out of the comparison instead of deciding it.

    **Online sizing.**  With ``iters`` and ``rounds`` left at ``None`` this
    picks both from what it has already measured, per variant:

    * ``iters`` so a block lasts about ``block_ms``.  The corpus spans five
      orders of magnitude -- 0.06 ms at the transposed sites, 45,241 ms at the
      2 GiB cliff -- and a fixed ``iters=10, rounds=6`` means 60 calls either
      way: microseconds for one cell and **45 minutes** for another.
    * ``group``, the number of calls behind one event interval, so the
      instrument costs under ``tax_budget`` of the *shortest* arm -- **one width
      for every arm**, never per-arm; see :func:`_common_group` for the ratio
      that got biased by 4% when it was per-arm.
    * ``rounds``, growing until the 95% half-width on every reported quantity --
      each variant's median and each variant's paired ratio against the first --
      is within ``target_rel``, or until ``budget_s`` of wall clock is spent, or
      ``max_rounds``.  It only ever stops on a **balanced block boundary**
      (a multiple of ``2 * len(variants)`` rounds) so that stopping early cannot
      reintroduce the position bias the ordering exists to remove -- except
      against ``hard_budget_s``, which is checked every round because a cell
      whose single round costs minutes must be able to stop without first
      completing a design it cannot afford.  ``Measurement.balanced`` then says
      the design did not close.

    **Arms are never stopped individually.**  If one arm's interval tightens
    first, it keeps running: dropping it would leave the other arm measured
    against a different stretch of wall clock, which is precisely the sequential
    comparison this function exists to avoid.  Stopping is a property of the
    block, not of an arm.

    Passing ``iters`` and/or ``rounds`` as integers pins them exactly, which is
    what every existing caller does and what a deliberately reproducible capture
    should keep doing.  ``warmup`` likewise.
    """
    names = list(variants)
    if not names:
        return {}

    # Warm and size in two passes, because the second decision belongs to the
    # comparison rather than to any one arm: every variant is probed first, and
    # only then is one common event-interval width chosen for all of them.
    need_duration = iters is None and not flush
    probes = {
        name: _probe(
            variants[name],
            pinned_warmup=warmup,
            warmup_s=warmup_s,
            warmup_min=warmup_min,
            warmup_max=warmup_max,
            warmup_hard_s=warmup_hard_s,
            block_ms=block_ms,
            max_iters=max_iters,
            need_duration=need_duration,
        )
        for name in names
    }
    if flush:
        # One flush per block reaches only the block's first sample, so a cold
        # measurement has to have exactly one sample per block.
        group = 1
        sizes = {name: iters or 1 for name in names}
    elif iters is not None:
        group = 1
        sizes = {name: iters for name in names}
    else:
        group = _common_group(
            [probes[n][0] for n in names], tax_budget=tax_budget, max_iters=max_iters
        )
        sizes = {
            name: _size_block(probes[name][0], group, block_ms, max_iters)
            for name in names
        }
    plans: dict[str, Plan] = {}
    for name in names:
        d, warmed = probes[name]
        tax = 0.0
        if measure_tax and sizes[name] >= 4 and 0 < d <= _TAX_PROBE_MAX_MS:
            tax = _measure_tax(variants[name], sizes[name], group)
        plans[name] = Plan(d, sizes[name], group, warmed, tax)
    torch.cuda.synchronize()

    times: dict[str, list[float]] = {name: [] for name in names}
    stalls: dict[str, list[float]] = {name: [] for name in names}
    firsts: dict[str, list[float]] = {name: [] for name in names}

    block = 2 * len(names)
    fixed = rounds is not None
    limit = rounds if fixed else max_rounds
    stop = "fixed" if fixed else "max_rounds"
    t0 = time.perf_counter()
    r = 0
    while r < limit:
        order = _order(names, r)
        for name in order:
            if flush:
                flush_caches()
            ms, stall, first = _time_block_full(
                variants[name], plans[name].iters, plans[name].group
            )
            times[name].append(ms)
            stalls[name].append(stall)
            firsts[name].append(first)
        r += 1
        if fixed:
            continue
        elapsed = time.perf_counter() - t0
        # The hard cap is checked every round, not only on a block boundary:
        # a cell whose single round costs minutes must be able to stop without
        # first completing a balanced design it cannot afford.  When it does,
        # ``balanced`` says so.
        if elapsed > hard_budget_s and r >= floor_rounds:
            stop = "budget"
            break
        if r % block:
            continue
        if r >= min_rounds and _precise_enough(names, times, target_rel):
            stop = "converged"
            break
        if r >= floor_rounds and elapsed + elapsed / r * block > budget_s:
            stop = "budget"
            break

    seconds = time.perf_counter() - t0
    # With one variant there is no order to balance, so the question does not
    # arise; with more, the design only closes on a multiple of ``2 * n``.
    balanced = len(names) < 2 or (r % block) == 0
    return {
        name: Measurement(
            name,
            tuple(times[name]),
            tuple(stalls[name]),
            iters=plans[name].iters,
            group=plans[name].group,
            firsts=tuple(firsts[name]),
            tax_ms=plans[name].tax_ms,
            stop=stop,
            balanced=balanced,
            seconds=seconds,
        )
        for name in names
    }


def _precise_enough(names, times, target_rel: float) -> bool:
    """Every reported quantity within ``target_rel``: the medians and the ratios."""
    for name in names:
        vals = times[name]
        med = statistics.median(vals)
        if med <= 0 or _half_width(vals) / med > target_rel:
            return False
    ref = names[0]
    for name in names[1:]:
        logs = [math.log(a / b) for a, b in zip(times[name], times[ref]) if b > 0]
        if not logs or _half_width(logs) > target_rel:
            return False
    return True


def format_table(
    rows: Iterable[Sequence[object]],
    headers: Sequence[str],
    aligns: str | None = None,
) -> str:
    """A plain fixed-width table; the reports are read in a terminal."""
    rows = [[str(c) for c in row] for row in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]
    aligns = aligns or "l" * len(headers)

    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(
            c.rjust(w) if a == "r" else c.ljust(w)
            for c, w, a in zip(cells, widths, aligns)
        )

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)
