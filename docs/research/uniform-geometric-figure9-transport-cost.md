# Figure 9 conservative transport slowdown

2026-09-20. Dawn Metal, default Uniform Geometric, 128 x 128 x 64,
dt=1/30 s. Diagnostic work only; no production shader changes in this
investigation. User clarified that downstream work must remain bounded even
when pressure fails to meet tolerance.

## Root cause of the performance cliff

`uvAddDonor` in `uniform-volume.wgsl.ts` implements floating-point atomic
addition with `atomicCompareExchangeWeak` inside an unbounded retry loop.
`uvSumDonors` calls it for every nonzero edge contribution. The host runs
this sum four times per step: once to find uncovered donors, then once in
each of three row/donor normalization rounds.

The traced departure is clamped to the simulation box. With extreme input
velocities, many different destination cells therefore address the same
boundary/corner donor. This preserves a small number of edges per destination
but creates enormous contention on a few donor accumulators. Losing threads
retry their floating-point sum. Neither the number of attempts nor execution
time has a fixed per-edge bound.

Captured edge-table fan-in and hardware timestamps from one run:

| Frame | Time | Active transport tiles | Nonzero edges | Maximum donor fan-in | Four donor sums |
|---|---|---:|---:|---:|---:|
| 95 | 3.167 s | 16,384 | 7,543,110 | 27 | 6.554 ms |
| 98 | 3.267 s | 16,384 | 7,567,216 | 29 | 6.816 ms |
| 99 | 3.300 s | 16,384 | 2,662,869 | 268,436 | 4,284.613 ms |

The most contended donor is `(127,127,63)`, the upper corner. Several other
boundary donors each receive approximately 26,000 contributions. Counts
come from the final normalized edge table; all transport tiles were active
for these samples, so stale skipped-tile edges do not affect this comparison.
They measure fan-in, not the actual number of failed CAS attempts.

Frame 99's four donor passes took 1794.179, 1754.857, 502.923 and 232.653 ms.
Its entire advance took 4417.409 ms wall time. Fewer nonzero edges produced
about 629 times more donor-sum time. A separate run had a much smaller
27.07 ms donor-sum spike at the same frame: the severity depends strongly on
the resulting trajectory distribution and GPU contention. It is not a
fixed extra set of dispatches or a newly enabled balancing schedule.

This is contention-driven, potentially superlinear retry work; the data do
not establish an exponential complexity law.

## What is already bounded

- `uvTrace` and `uvEmbeddedAir` clamp endpoints before deriving half-cell
  walk length. On this grid, a trace takes at most 256 half-cell steps, even
  when the raw velocity displacement is millions of cells.
- `uvTwoLevelTransportReach` caps dilation reach at 16 tiles. The active set
  was already the whole domain before the cliff.
- Each destination has nine edge slots. The three normalization rounds and
  four donor-sum dispatches are fixed with default capacity balancing off.

The longer clamped walks do cost more: edge construction rose from about
2.16 to 9.11 ms in the fan-in run. They do not explain the four-second stall.
The GPU CAS loop is the missing work bound. It would be incorrect to claim
that the trace loop literally executes millions of cell steps.

## Trigger, separately from the transport defect

Multiple runs encounter a pressure instability at frame 98. In one captured
run, incoming advected velocity was 19.41 m/s, projected velocity reached
7.84 million m/s, and negative-domain velocity reached 11.59 million m/s.
The finest residual was 58.84 million /s and the coarse solver hit its cap.
In the next step the recorded raw displacement was 3.37 million cells.

An instrumented repeat also failed at frame 98: fine residual 1.33 million
/s, coarsest iteration cap 4096, failing coarse invocation 11. Restricting
only frame 98 to one Full-Cycle still gave fine residual 3304.6 /s and
approximately 597 m/s projected velocity, although each coarse solve met its
own tolerance. Thus the coarse iteration-cap warning is not the first bad
correction. This investigation does not establish the precise hierarchy
cause and does not propose tighter tolerance as the transport remedy.

Pressure quality and bounded downstream work are separate requirements.
An accepted loose residual, a missed tolerance, or a catastrophic finite
velocity must not turn one donor accumulation into an unbounded retry storm.

## Recommended direction

Replace the contended floating CAS scatter with a bounded donor reduction.
The edge table already provides immutable donor indices for a step. Group
contributions by donor and reduce them in parallel bounded chunks/tree levels;
reuse that grouping across the four sums, since normalization changes weights
but not donor indices. A fixed-pass radix grouping plus segmented reduction
is one concrete design, with explicit edge-buffer capacity and no
contribution dropped on retry exhaustion. Its ordinary-case cost needs a
Dawn comparison before adopting it globally.

Sharding the CAS accumulators may reduce contention but does not establish
a strict work bound. Simply capping retries and dropping a contribution would
invalidate donor normalization and conservation. Fixed-point integer atomics
are another possible design, but require explicit overflow and tiny-weight
accuracy treatment; they are not an interchangeable replacement for the
current floating sums.

Validate the bounded path against frozen high-fan-in edge tables, including
all contributions targeting one donor, then replay figure 9 through 3–3.5 s
with the same loose pressure settings. Measure per-pass hardware time, mass
conservation and normalization error independently of whether pressure meets
its residual tolerance.

## Evidence

- `tools/probe-uniform-geometric-transport-cost-dawn.ts`: hardware timestamps
  for individual volume passes and optional donor fan-in readback.
- Reproduce: `node --import tsx tools/probe-uniform-geometric-transport-cost-dawn.ts --capture=90,95,98,99,100`.
- `uniform-geometric-figure9-transport-cost-2026-09-20.json`: both timing
  traces, default control values, pressure-onset statistics and one-cycle
  diagnostic.
- `/tmp/figure9-fanin`, `/tmp/figure9-transport`, `/tmp/figure9-fields`,
  `/tmp/figure9-coarse`, `/tmp/figure9-onecycle`: detailed diagnostic files.
