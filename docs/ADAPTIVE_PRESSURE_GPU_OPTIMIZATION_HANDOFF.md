# Adaptive pressure GPU optimization handoff

## Purpose

This document captures the octree pressure-solve optimization completed in
July 2026 and translates it into a concrete starting point for the quadtree
tall-cell solver. It is intended to let another agent reproduce the reasoning,
avoid the unsuccessful branches, and validate a quadtree implementation
without rediscovering the performance model.

The central lesson is that this was not a workgroup-size problem. The leaf
solver was dominated by device-wide iteration boundaries and insufficient work
between them. The successful change restructured the numerical method and the
submission pipeline together.

## Results at a glance

Balanced `dam-break-ui`, 61x46x41 grid, 128-equivalent-sweep pressure effort:

| Path | Pressure GPU time | Wall time per step | Global solve passes |
| --- | ---: | ---: | ---: |
| Compacted weighted Jacobi | 3.28 ms | 4.40 ms | 128 |
| Row-parallel Chebyshev | 0.72 ms | 2.06 ms | 32 |

This is a 78% pressure-stage reduction and a 53% end-to-end step reduction.
Kinetic energy stayed within 0.03%, RMS liquid divergence within 0.2%, and the
2.2-second stability gate passed.

Balanced `dam-break-boxes`, 61x46x41, with the dynamic rigid-coupling operator
enabled:

| Path | Pressure GPU time | Wall time per step | Coupling policy |
| --- | ---: | ---: | --- |
| Exact compact Jacobi | 117.96 ms | 98.65 ms | Refresh `K^T p` every iterate |
| Chebyshev + lagged rigid exchange | 0.79 ms | 3.14 ms | Apply impulse next batch |

The lagged path reduced pressure time by 99.3%. At 0.2 seconds, kinetic energy
differed by 0.07% and RMS liquid divergence by 0.11%. A 0.5-second coupled
stability envelope, large prescribed-displacement test, and batched two-way
dynamic-body smoke test pass without WebGPU validation errors.

These numbers are workload and adapter specific. Their useful meaning is the
shape of the result: removing global dependencies dominates local shader
tuning when each dispatch contains only a small sparse leaf set.

## What the old pipeline did

The compact octree path already avoided the worst dense-grid work:

1. Find wet leaf origins.
2. Prefix-sum compact them into a row list.
3. Assemble the finite-volume diagonal, right-hand side, and merged neighbor
   coefficients once.
4. Launch one indirect weighted-Jacobi dispatch per sweep.
5. Project the dense velocity field from the final leaf pressure.

Step 4 was still expensive. A balanced solve issued 128 dispatches. Each
dispatch took roughly 17 microseconds even though its arithmetic was small.
The GPU repeatedly crossed a global storage/dispatch boundary, performed a
short row walk, and stopped. Increasing workgroup size could redistribute the
same small amount of work but could not remove 128 global boundaries.

A single-workgroup persistent megakernel was also tested. It removed dispatch
boundaries but serialized the entire sparse solve onto one compute unit. It
measured 22.7 ms on the transient dam-break solve and was rejected as the
default. It remains a useful A/B path for very small systems that converge in
only a few iterations.

## Successful numerical restructuring

The default solve is now a Chebyshev semi-iteration over the diagonally scaled
operator `D^-1 A`.

The topology and finite-volume discretization did not change. The key property
is that one polynomial iteration needs only a sparse matrix-vector product and
per-row recurrence state. It has no dot-product reduction, convergence
readback, workgroup barrier, or single-workgroup owner.

Each compacted row stores two additional values in existing header padding:

- the previous polynomial search correction;
- the previous Chebyshev recurrence scalar.

The spectral interval is currently `[0.01, 2.2]`. Balanced pressure effort of
128 Jacobi-equivalent sweeps maps to 32 polynomial passes. High and ultra map
their larger effort budgets by the same 4:1 policy. This is an empirical
quality mapping, not a claim that one Chebyshev pass is algebraically identical
to four Jacobi sweeps.

The final hot loop is therefore:

```text
compact rows once
assemble A and b once
repeat 32 times:
    one row-parallel cached SpMV
    one local Chebyshev recurrence update
project velocity
```

All rows remain available to the scheduler on every pass. No attempt is made
to manufacture occupancy by forcing a single persistent workgroup.

## Submission and readback restructuring

The solver optimization exposed a second problem: a fast GPU step can still be
starved by requestAnimationFrame and queue-fence cadence.

At the default 4 ms simulation clock, octree execution now prepares five
advances per 60 Hz presentation quantum and permits a second bounded batch in
flight. This gives the queue up to ten ordered advances without letting CPU and
GPU simulation time drift without limit.

Topology rebuild, compaction, pressure, projection, and surface transport stay
in one ordered GPU command stream. Queue-completion promises update telemetry;
they are not awaited by step encoding. Statistics readbacks are asynchronous
and pending-gated.

Rigid impulse readback uses a reusable slot pool. Every encoded coupled step
copies its small exchange record into its own idle staging slot. Overlapping
`mapAsync` operations therefore do not drop later impulses, and no submission
waits for an earlier map to complete. The controller merges returned impulses
and amortizes them over the next fixed rigid substeps.

## Rigid-body coupling: exact versus lagged

The original exact coupled pressure system includes the low-rank response

```text
(A + K M^-1 K^T) p = b
```

where `K` maps pressure to generalized body force and `M` is body mass and
inertia. The exact Jacobi ladder recomputes `K^T p` before every pressure
iterate. In the current shader, that reduction scans finest-grid faces with
one workgroup per body. It is both globally dependent and poorly parallel at
small body counts. On `dam-break-boxes`, it dominates the solve by two orders
of magnitude.

The default accelerated path now uses a weakly coupled partitioned split:

```text
presentation batch n
    upload body transform and velocity after consuming impulse J[n-1]
    assemble pressure RHS using that prescribed solid velocity
    solve A p[n] = b[n] with row-parallel Chebyshev
    project fluid velocity
    accumulate pressure and tangential reactions J[n]
    copy J[n] to an asynchronous pooled readback slot

presentation batch n+1
    distribute J[n] over the fixed rigid substeps
```

This preserves two-way momentum exchange but accepts a bounded temporal lag.
It removes `K^T p` from the inner pressure loop and removes the per-step CPU/GPU
feedback fence. The exact `compact` and `dense` paths remain selectable for A/B
validation and cases where same-step body response is more important than
throughput.

The tradeoff should be monitored most closely for very light bodies, stiff
contacts, large pressure impulses, and time steps near the coupling stability
limit. If those cases ring, prefer targeted stabilization—impulse blending,
an added-mass predictor, or one outer correction per presentation batch—over
restoring a global reduction to every pressure iterate.

## Code map

- `lib/webgpu-octree.ts`
  - compact row assembly;
  - `iterateChebyshev` recurrence;
  - exact rank-six A/B path;
  - lagged rigid gating and pressure-impulse publication.
- `lib/webgpu-uniform-eulerian.ts`
  - ordered step encoding;
  - pooled rigid exchange readbacks;
  - solver telemetry and labels.
- `lib/simulation/gpu-clock.ts`
  - presentation batch depth;
  - bounded two-batch preparation window.
- `lib/simulation/controller.ts`
  - consumption and amortization of the previous GPU impulse batch.
- `lib/performance-stage-model.ts`
  - user-visible description of the pressure and synchronization model.
- `tools/run-webgpu-smoke.ts`
  - `FLUID_OCTREE_LEAF_SOLVER=compact|chebyshev` A/B control.
- `tools/run-octree-lagged-rigid-smoke.ts`
  - batched, genuinely moving, two-way rigid feedback regression.
- `tools/run-octree-displacement-smoke.ts`
  - large prescribed-solid displacement and volume-complement regression.

## Quadtree transfer plan

Do not copy the octree shader mechanically. The quadtree solver is PCG with
several preconditioners, partial reductions, an optional persistent
single-workgroup path, dynamic iteration budgets, and topology packing. Apply
the same performance principles at its actual dependency boundaries.

### 1. Establish a dispatch ledger

For one representative solve, record:

- DOF and face counts;
- dispatches in setup, each PCG iteration, each preconditioner application,
  coupling, mapping, and projection;
- time per dispatch and total pressure time;
- number of global scalar dependencies per converged solve;
- whether the queue has useful independent work after each boundary.

Use `lib/webgpu-quadtree-tall-cell.ts` around the `iterations(...)` ladder and
the polynomial-preconditioner kernels as the starting point. Do not infer ALU
limitation from total pressure time; prove it with the dispatch ledger.

### 2. Keep geometry work outside the iterative loop

The octree win depended on assembling coefficients once. Verify the quadtree
path refreshes only genuinely time-dependent face fractions and RHS values,
while row connectivity, coefficient structure, line tables, and polynomial
spectral data remain cached. Avoid rebuilding or walking packed topology inside
every Krylov phase.

### 3. Reduce communication without sacrificing parallel rows

Candidate order:

1. Benchmark the existing polynomial/Jacobi parallel ladder separately from
   the one-workgroup megakernel.
2. For dispatch-bound systems, test a fixed-degree Chebyshev solve or smoother
   that keeps every DOF row parallel and eliminates PCG dot-product reductions.
3. If PCG convergence quality is required, investigate pipelined CG or an
   `s`-step/communication-avoiding chunk so several useful matrix operations
   occur per global scalar boundary.
4. Use a fixed polynomial as the preconditioner only if the surrounding PCG
   reductions do not remain the dominant cost.
5. Retain the exact ladder as a reference, not as an automatic fallback for
   every coupled scene.

The existing megakernel is appropriate only below its measured crossover. A
single workgroup is a latency optimization for tiny systems, not a general
solution for GPU utilization.

### 4. Port the lagged rigid split before optimizing exact coupling

The quadtree packed system currently treats dynamic coupling as a reason to
include low-rank body terms and to leave some GPU-resident fast paths. For the
throughput mode:

1. Build the fluid pressure operator without `K M^-1 K^T`.
2. Use the previous batch's body transform and velocity in solid-boundary RHS
   terms.
3. Solve the fluid rows with the same uncoupled parallel path.
4. Compute the current pressure impulse once, after convergence.
5. Deliver it through the existing pooled asynchronous exchange for the next
   presentation batch.
6. Allow coupled quadtree batching only after topology construction and body
   rasterization no longer introduce a host dependency.

The main architectural prize is not merely fewer coupling dispatches. It is
making dynamic bodies eligible for the resident topology and parallel pressure
paths that are currently restricted to uncoupled systems.

### 5. Validate in layers

Required A/B gates:

- shader and bind-layout validation;
- no CPU map or queue fence in the step encoder;
- compact/exact versus accelerated pressure timing at fixed topology;
- pressure residual or divergence quality;
- volume and level-set conservation;
- kinetic-energy and maximum-speed envelopes;
- static immersed geometry;
- prescribed large displacement;
- freely moving heavy and light bodies;
- contact/stack stability;
- impulse and angular-impulse delivery count under two batches in flight;
- long-horizon impact, rebound, and settling.

Report both stage time and end-to-end wall time. A faster pressure solve that
creates queue starvation, dropped impulse snapshots, or a new CPU topology
handshake is not complete.

## Reproduction commands

These require `WEBGPU_NODE_MODULE` to point at the local Dawn/WebGPU module.

```sh
# Uncoupled accelerated regression
npm run test:webgpu:dam-octree

# Exact versus accelerated coupled A/B
FLUID_SCENE=dam-break-boxes FLUID_METHOD=octree \
  FLUID_OCTREE_LEAF_SOLVER=compact FLUID_TARGET_S=0.2 \
  FLUID_CPU_ORACLE=0 node --import tsx tools/run-webgpu-smoke.ts

FLUID_SCENE=dam-break-boxes FLUID_METHOD=octree \
  FLUID_OCTREE_LEAF_SOLVER=chebyshev FLUID_TARGET_S=0.2 \
  FLUID_CPU_ORACLE=0 node --import tsx tools/run-webgpu-smoke.ts

# Batched two-way moving-body feedback
npm run test:webgpu:octree-lagged-rigid

# Large geometric displacement and free-surface complement
npm run test:webgpu:octree-displacement
```

## Decision summary

The optimization succeeded because it changed the unit of useful GPU work:
from one tiny globally ordered relaxation sweep to one fully parallel
polynomial pass, then grouped enough independent advances to keep the queue
fed. Rigid-body performance required the same move at a higher level: relax
time coherence by one bounded presentation batch so a low-rank global response
does not re-enter every pressure iteration.

For the quadtree port, optimize dependencies first, arithmetic second, and
workgroup dimensions last.
