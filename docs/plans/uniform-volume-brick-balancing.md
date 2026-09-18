# Brick-local liquid capacity balancing

## Recommendation

Prototype 4×4×4 scheduling bricks over the existing dense fields. Keep direct
dispatch and uniform workgroup early exits. Restrict receiver work to bricks
whose incoming liquid can have changed; restrict normalization to donors whose
outgoing weights changed. Use actual transport edges to wake receivers.

Do not independently normalize bricks, use a fixed geometric neighbor halo,
or omit inactive receivers from a donor's sum. Those changes break conservation
or overlook long-CFL dependencies. Keep the lattice, phi/V fields, pressure,
redistancing, sharpening, and rendering unchanged.

## Measured opportunity

Production mini64, bounded MacCormack, 0.1% capacity tolerance, 64-round limit.
Captured the actual forward transport edges, source V, and open capacity at
frames 2, 8, 22, 30, before rounds 1, 4, 16, 64. For every violating receiver,
mark its positive-liquid donors, then all receivers sharing those donors.

| Frame | Time | Receiver bricks affected, 4³ | Receiver bricks affected, 8³ |
|---|---:|---:|---:|
| 2 | 0.067 s | 9.5–13.8% | 12.5–18.4% |
| 8 | 0.267 s | 38.3–39.1% | 43.4–45.7% |
| 22 | 0.733 s | 41.7–42.1% | 50.0–50.6% |
| 30 | 1.000 s | 37.9–38.6% | 44.3–45.5% |

These are dependency-expanded work fractions, not just overfull-cell counts.
4³ matches the existing 64-thread workgroups and wastes less work than 8³.
The measured snapshots suggest roughly 58–90% of repeated receiver grid work
can be avoided. They do not establish a matching GPU speedup: reverse-index
construction, divergent donor degrees, launch/clear overhead, and memory traffic
must be charged. Census readbacks distort timing and are not a benchmark.

An initial target of 2–3× faster balancing during impact is plausible, but
unproven. A 10× whole-step speedup is not supported. Even a 3× improvement in a
stage occupying 55% of total time gives only 1.58× overall. The earlier dispatch
benchmark used the then-default semi-Lagrangian velocity method; its milliseconds
must not be presented as current bounded-MacCormack timings.

## A convergence problem, too

At frame 2 there were no individually infeasible donors. At frames 8, 22, and
30 there were respectively 2,237, 310, and 2,087 before the first balancing
round. A necessary feasibility condition for donor j is:

    V[j] <= sum(K[i] for each distinct reachable receiver i)

Violating this proves that no reweighting on the frozen stencil can fit all its
liquid. Passing it does not prove feasibility: donor subsets can still compete
for insufficient receiver capacity.

At frame 30, the single-donor bound already implies at least 101.12 relative
overfill (about 10,112%). The measured maximum approaches that bound and stays
there through round 64. This is impossible placement under the current stencil,
not merely slow iterative convergence. It helps explain why a global maximum
keeps the whole grid awake. The current bounded-MacCormack simulation also has
known phi-volume drift; this scheduling proposal does not claim to fix it.

Publish infeasibility and residuals separately. Never subtract the bound from
the UI error, discard V, expand the stencil silently, or call stagnation
convergence. Initially retain the user round cap for unresolved components.
A single-donor certificate alone is not permission to skip an entire brick:
other receivers in it may still benefit from correction.

## Proposed algorithm

1. **Build a reverse edge index once per transport step.** After geometric
   coupling, construct donor→incoming-forward-edge CSR: offsets plus packed
   forward-edge IDs. Reuse the current forward donor/weight array. Include every
   positive edge belonging to a donor with exactly positive source V; zero-V
   donors cannot affect liquid this step. Rebuild next step, including new
   sources/topology. No spatial hashes, tree, or persistent scene topology.

2. **Seed receiver activity.** All receiver bricks participate in the first
   scan. Later rounds use GPU-owned brick activity epochs. If a brick's epoch
   is inactive, its workgroup returns before fetching its cell stencils.

3. **Receiver pass.** Compute incoming V and relative overfill for active
   bricks. Scale only violating receivers' edge weights, as today. Mark the
   affected donors and their scheduling bricks with the current epoch.
   Retain each brick's maximum error for truthful global diagnostics.

4. **Donor pass.** Each dirty donor reads *all* incoming edge weights via CSR,
   including edges from sleeping receiver bricks. Sum once and normalize every
   edge in that donor's list. Each forward edge belongs to exactly one donor,
   so its owner can write it without float scatter atomics. Wake every receiver
   brick whose edges this donor normalizes for the next receiver pass.
   A dirty-donor bit/epoch inside the donor brick avoids needlessly processing
   its other 63 donors.

5. **Stop locally, wake by dependency.** A receiver brick sleeps when no donor
   can have changed its incoming V. An error below tolerance allows sleep only
   if no donor correction schedules it again. Newly woken bricks recompute
   their residual; active sets can grow as well as shrink. The user tolerance
   and maximum-round controls retain their meaning. Global termination is
   safe only when no corrective dependency remains; a cached brick-error
   reduction can retain the existing global status without rescanning edges.

There is a whole-dispatch boundary between receiver writes and donor reads,
and another before receiver work resumes. Do not normalize a donor while
other workgroups are still accumulating its corrections.

Use direct dispatch, as measured faster on this Metal device. Inactive groups
exit on a uniform flag. Epochs avoid clearing the donor flags and receiver
masks every round. Handle epoch wrap with an explicit reset. Keep the existing
control dispatch if needed; this design does not rely on a CPU round trip.

Unlike an incremental floating-point donor-total cache, recomputing dirty sums
from reverse incidence avoids cancellation drift. Unlike accumulated row and
column factors, it avoids factors shrinking/growing through dozens of rounds.
It also removes the current repeated CAS float scatter of every edge in the
entire domain. Summation order changes, so numerical parity is tolerance-based,
not bitwise.

## Storage and setup cost

For 64³, there are 4,096 scheduling bricks at 4³. Reverse edge IDs need at most
9N uints (9 MiB); N+1 donor offsets need about 1 MiB. Activity masks are only
kilobytes. Reuse the existing conditioning scratch for count/cursor/epoch
storage where lifetimes permit. The measured positive-liquid edge count is
about 0.75–1.13 million, below the 2.36 million worst-case bound.

Count/scan/fill adds one-time per-step setup and roughly 10 MiB worst-case
persistent reverse-index storage. Include both in the benchmark. Use a separate
small balancing bind layout if needed; do not enlarge the pressure layout.
The inverse index is transport incidence, not adaptive spatial topology.

## Implementation and acceptance sequence

- Preserve the current full-grid algorithm as the reference arm.
- Add reverse incidence and brick epochs as a benchmark arm; compare identical
  initial scenes, bounded MacCormack, tolerance, cap, and simulation times.
- Check donor conservation after every normalization, V positivity, final
  overcapacity, and phi unchanged by balancing. Test a cross-brick shared donor,
  a characteristic spanning multiple bricks, a sleeping brick that must wake,
  a resting pool, cut capacities, and an infeasible donor. Never skip a donor's
  inactive outgoing edges.
- Measure entire-frame and stage GPU/CPU time, including CSR setup and memory,
  on mini32 and mini64: rest, pre-impact, impact and late motion. Compare 4³ and
  8³ grouping, 1/8/64 caps, and several tolerances. Tiny round limits may favor
  the existing dense path because reverse-index construction cannot amortize.
- Ship only if total time improves without degrading conservation or residual
  quality. Prefer a simple measured crossover between dense and brick paths
  over another universal policy. Any later work on unplaceable V is a separate
  numerical change with separate validation.

Raw census: `docs/benchmarks/uniform-volume-brick-work-2026-09-19.json`.
Reproduce with `node --import tsx tools/probe-uniform-volume-brick-work-dawn.ts`.
Only diagnostic readback support was added to the solver for this investigation;
the proposed scheduling algorithm is not implemented yet.
