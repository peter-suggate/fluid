# Quadtree tall-cell pressure solve — performance plan

Handoff plan for optimizing the Narita et al. quadtree tall-cell ICCG pressure
projection (`lib/webgpu-quadtree-tall-cell.ts`). Target hardware: Apple
M1 Max (32 GPU cores, unified memory), Chrome/Metal via WebGPU. Companion to
the restricted tall-cell multigrid optimization committed in `fe445f6`, whose
measured lessons directly shape the priorities below.

Fidelity ground rule: the solve is CG with a tolerance-based stop
(`relativeTolerance`, paper uses 1e-4). Anything that preserves the operator,
the RHS, and the stop criterion preserves the converged solution — including
**changing the preconditioner**, which only changes the path to convergence.
Optimizations below are grouped by whether they are bit-path-neutral or
tolerance-neutral.

---

## 1. Current architecture (as of 2026-07-16)

One `encode()` builds a single compute pass per solve:

- `refreshFaces` — re-derives free-surface scaling from the resident level
  set; now also caches the per-slot liquid mask into `face.packed` bits 22+
  and the face flux (`face.flux`), so per-iteration kernels no longer sample
  phi. (Landed during the current uncommitted refactor.)
- `refreshRows` — caches per-row activity into `state[row].activeFlag`
  (avoids the per-iteration `dofActive` face-graph walk). (Landed.)
- `initialize`, `precondition` | `preconditionJacobi`, `startDirection`,
  `reduceInitial`, then **`this.iterations` encoded CG iterations**, each:
  `updateDispatch` (1 thread) → `multiply` (indirect) → [`coupleReduce`,
  `coupleApply`] → `applyStep` (indirect) → `precondition*` (indirect) →
  `finishIteration` (indirect); finally `coupleImpulse` + `project`.
- Convergence early-exit is GPU-side: `updateDispatch` zeroes the indirect
  args once `‖r‖² ≤ tol²·‖b‖²`, so converged iterations become zero-workgroup
  dispatches — but they are still *encoded and processed*.
- `this.iterations = max(options.pressureIterations, min(2048, ⌈4√dof⌉))`
  (`lib/webgpu-quadtree-tall-cell.ts:696`). A 100k-dof scene encodes ~1,265
  iterations ≈ **~9,000 dispatch commands per step**.
- Topology rebuilds run GPU construction (`WebGPUQuadtreeBuilder`) + CPU
  decode/tall-grid/variational assembly/IC(0) factorization/level scheduling/
  system packing, now in a Web Worker (`quadtree-topology-worker.ts`).

### Baseline numbers (dam-break-ui, balanced, Dawn/Metal, M1 Max)

Measured 2026-07-16 mid-refactor — re-baseline before starting (§4.0):

| Metric | Value |
|---|---|
| liquid dof | 5.5k → 12.4k as the dam falls |
| iterations used (mean) | ~213 |
| encoded iteration budget | max(96, 4√dof) ≈ 445 |
| gpu `pressure_ms` per step | ~318 ms mean, 505 ms p90 (!) |
| CPU topology pack per rebuild | ~163–173 ms (in worker) |
| `gpuConstruction_ms` per rebuild | ~370 ms (needs investigation — likely readback stall, not kernel time) |
| topology reuse rate | 4–83% depending on flow phase |

~318 ms / 213 iterations ≈ **1.5 ms per CG iteration on a 12k-dof system** —
around 100× above any reasonable floor. This is the headline problem.

---

## 2. Transferable lessons from the multigrid work (fe445f6)

These were measured, not guessed; assume they hold here:

1. **Serialized-dispatch floor ≈ 13 µs/dispatch** on M1 Max for dependent
   dispatches within one pass (measured at two grid scales). 9,000 encoded
   dispatches ≈ 120 ms of pure command-processing even if every dispatch is
   empty. Encoded-but-converged iterations are *not* free.
2. **Single-workgroup kernels are barrier-latency-bound**, not load-bound
   (~6 µs per barriered round at 256 threads). The IC(0) `precondition`
   kernel does `2 × factorLevelCount` barriered rounds *per CG iteration* in
   one workgroup — if the level count is O(hundreds), that alone is ~1 ms per
   iteration and fully explains the 1.5 ms/iteration observation.
3. **Cache frozen-geometry work out of the iteration loop** (already largely
   done here via `refreshFaces`/`refreshRows`).
4. **M1 clock ramping**: micro-workloads run at low clocks; expect ±10% noise
   and ~1.5× hot-vs-cold differences. Interleave A/B runs, report medians.
5. **Validate shaders before trusting any benchmark.** During this session
   every quadtree pipeline was silently invalid for a while (`active` is a
   reserved WGSL word) and all GPU work was discarded while the sim appeared
   to "run". `npm run test:quadtree-shaders` first, always.

---

## 3. Where the time goes (hypothesis ranking, to be confirmed by §4.0)

Per CG iteration at ~12k dof:

| Suspect | Est. share | Why |
|---|---|---|
| `precondition` (IC(0) level-scheduled, **1 workgroup**, 2×levels barriered rounds) | dominant | lesson #2; `factorLevelCount` is packed in `params.counts.w` — log it, likely O(100s) |
| `applyStep` + `finishIteration` (**1 workgroup each**, two full sweeps of `state[]` at 256 threads) | large | each loops `dof/256` strided over a 32-byte AoS struct, twice |
| ~7 encoded dispatches/iteration × 13 µs incl. dead post-convergence iterations | ~120 ms/step at 4√dof budget | lesson #1 |
| `multiply` (parallel, rowGroups) | small after refresh caching | reads faces CSR + state |
| `updateDispatch` (1 thread, every iteration) | ~13–20 µs/iter | unavoidable per-iteration sync point today |

---

## 4. Workstreams (in order)

### 4.0 Measurement first (½ day) — do not skip

- Re-baseline after the in-flight refactor settles: `pressure_ms` per step,
  `pressureIterationsUsed` vs budget, dof, and `factorLevelCount`.
- Add an opt-in intra-solve breakdown: split `encode()`'s single pass into
  per-phase passes with `timestampWrites` behind a debug flag (setup /
  first-N-iterations / remaining iterations / project). The multigrid session
  used a monkey-patched probe for this; a first-class debug flag is better.
- Harness: `tools/run-webgpu-smoke.ts` with
  `WEBGPU_NODE_MODULE=<path>/webgpu/index.js`,
  `FLUID_SCENE=dam-break-ui FLUID_METHOD=quadtree-tall-cell FLUID_CPU_ORACLE=0`.
  A per-step probe pattern (advanceTo → onSubmittedWorkDone → readStats)
  gives per-step `gpuTimings` + quadtree info fields.
- Acceptance for every later item: dam-break physics stats match to ≥4
  significant figures at 0.3 s (unless the item is tolerance-neutral-only,
  then: final relative residual ≤ tolerance and stats match to ~2–3 figures),
  `npm run test:quadtree-shaders`, unit tests, and
  `test:webgpu:dam-break-regression`.

### 4.1 Stop encoding dead iterations (1 day, bit-path-neutral)

The budget is a worst case; the *encoded* count should track reality.

- Feed back `pressureIterationsUsed` (already read via
  `readSolveDiagnostics`, scalars[9]) into the next step's encoded count:
  e.g. `encoded = clamp(1.5 × EMA(used), 32, budget)`. The
  `iterationBudgetHint`/`iterationEmaHint` option fields exist for exactly
  this — wire them through `rebuildFromState` and the per-step path.
- Safety: if a step exits at the encoded cap without hitting tolerance
  (scalars expose this), bump the next step's count (e.g. ×2). The tolerance
  stop still governs correctness; this only trims dead encoded work.
- Expected: at 213 used vs 445 encoded, saves ~half the command stream
  (~10–20 ms/step at the 13 µs floor); much more on large-dof scenes where
  the 4√dof floor dominates (2048 encoded ≈ 14k dispatches ≈ ~180 ms).

### 4.2 Parallelize the per-iteration reductions (1–2 days, tolerance-neutral, fp-order changes)

`applyStep` and `finishIteration` run on 1 of 32 cores. Split each into:

1. multi-workgroup partial reduction (`dof/256` workgroups → partials buffer),
2. 1-workgroup finalize (reduce partials, write α/β into `scalars`),
3. multi-workgroup axpy/update using the scalar.

That is +2 dispatches per iteration but turns the two O(dof) single-workgroup
sweeps into full-GPU sweeps. At 12k dof this may be a wash (dispatch floor);
at 100k+ dof it is mandatory. Gate on dof if needed. While here: convert
`SolverState` from the 32-byte AoS struct to SoA arrays so reductions and
axpys read densely packed f32 streams.

### 4.3 The preconditioner (the big one, 3–5 days)

IC(0) with a level-scheduled triangular solve is intrinsically hostile to
this GPU: one workgroup, `2 × factorLevelCount` barriers *per iteration*, and
a large CPU factorization + level-schedule + aux-texture upload *per topology
rebuild* (a big slice of the 163 ms `cpuTopologyPack_ms`).

Options, in recommended order:

a. **Vertical line-Jacobi (tridiagonal Thomas per column segment).**
   The tall-cell operator's stiff direction is vertical (tall segments couple
   a whole column through one face). A per-segment tridiagonal solve is
   embarrassingly parallel (one thread or subgroup per segment; segments are
   already first-class: `pressureGrid.segments`), needs no CPU factorization,
   no level scheduling, no aux texture — and it is SPD when the tridiagonal
   blocks are taken from the SPD system. Expected: iteration counts between
   Jacobi and IC(0), each application fully parallel and ~as cheap as a
   matvec. This also deletes most of the CPU rebuild cost (§4.5).
b. **Polynomial (Chebyshev/Neumann) preconditioner**, degree 2–4: fully
   parallel, just repeated matvecs — trivial once the matvec is cheap, no
   assembly. Worth benchmarking against (a); composes with it (poly over
   block-diagonal).
c. **Keep IC(0) but restructure the triangular solve**: multi-workgroup
   per level via one dispatch per level is a non-starter (levels × 13 µs ×
   iterations); a subgroup-cooperative single-workgroup version helps at most
   ~2×. Only pursue if (a)/(b) regress iteration counts badly on the
   deep-water scenes (the paper's headline case — test `dam-break-boxes` and
   the 20 m column scene the comments mention).

Measure: iterations-to-1e-4 and wall time per solve for ic0 / jacobi / line /
poly on: dam-break-ui (balanced+ultra), a deep-water tank, and a rigid-body
scene. The option plumbing (`preconditioner: "ic0" | "jacobi"`) already
exists — extend the union rather than replacing it, and keep `ic0` as the
paper-conformance reference path.

### 4.4 Cheapen the matvec further (1–2 days, optional after 4.3)

`rowProduct` still walks rows → faces → 4 slots with gather indirection.
`refreshRows` can additionally bake, per row, the actual CSR coefficient per
neighbor dof (coefficients are frozen per step once `refreshFaces` fixes the
liquid masks and fluid scales): matvec becomes a flat CSR SpMV. Storage:
`rowEntries`-shaped buffer of (dof, coeff). This also gives (a) in §4.3 its
tridiagonal coefficients for free.

### 4.5 CPU rebuild path (1–2 days)

- `cpuTopologyPack_ms` ≈ 163–173 ms per rebuild, in-worker (browser) but
  still bounding rebuild cadence, and it runs *synchronously* in Node smoke
  runs (no global `Worker`), which poisons those benchmarks — worth a
  `worker_threads` shim for measurement honesty.
- Profile the split that already exists (`quadtreeDecode_ms`, `tallGrid_ms`,
  `variationalAssembly_ms`, `systemPack_ms`) — plus IC factorization inside
  `packSystem`. If §4.3(a/b) lands, factorization + level scheduling +
  factor-aux packing disappear outright.
- `gpuConstruction_ms` ≈ 370 ms/rebuild as measured is suspicious (the
  kernel-only number `gpuConstructionKernel_ms` should be ms-scale). Check
  whether the wall number is sitting behind the giant pressure pass in queue
  order — if so, rebuild kicks should be submitted before/independently of
  the solve submit, or simply reported separately.

### 4.6 Micro items (opportunistic)

- `project` runs over the full fine grid with per-cell face lookups — fine;
  but it re-reads `levelSetIn` per axis; single load + reuse.
- `updateDispatch` could run every K iterations (check-every-4) to cut its
  dispatch count 4× at the cost of ≤3 extra live iterations after
  convergence; net win only after 4.1 — measure.
- Bind-group is rebuilt per projection object; cached fine. No action.

---

## 5. What NOT to do

- Don't lower `relativeTolerance` or cap iterations below convergence to
  "win" benchmarks — that trades fidelity, which is out of bounds.
- Don't replace CG with the multigrid from the restricted path — different
  operator (variational, T-junctions, MLS sub-faces); the paper's solver is
  ICCG and the conformance tests pin its structure.
- Don't benchmark with shaders that haven't passed
  `npm run test:quadtree-shaders` (see §2.5 incident).
- Don't trust a single timing run (§2.4); interleave and use medians.

---

## 6. Verification checklist (per landed change)

1. `npm run test:quadtree-shaders` (Dawn validation of every entry point).
2. `npx tsc --noEmit`, `npm run test:unit`.
3. `npm run test:webgpu:dam-break-regression` (0.2 s field regression).
4. A/B physics: 0.3 s dam-break-ui, compare `maxSpeed`, divergence extrema,
   `pressureRelativeResidual`, volume drift, wet/mixed cell counts,
   centroid. Bit-path-neutral changes should match to 5–6 significant
   figures; preconditioner changes should match residual ≤ tolerance and
   stats to ~2–3 figures with identical stability flags.
5. Perf: median of ≥3 interleaved runs of per-step `pressure_ms` at
   balanced + ultra, plus `pressureIterationsUsed` (regressions in iteration
   count are a red flag even if wall time improves).

---

## 7. Expected end state

| Stage | pressure per step (12k dof, balanced) |
|---|---|
| today | ~318 ms |
| + 4.1 budget feedback | ~150–200 ms |
| + 4.2 parallel reductions | ~100–150 ms |
| + 4.3 line/poly preconditioner | **~5–20 ms** (iteration cost → ~2–4 dispatches × parallel work; iterations ~1–2× ic0) |
| + 4.4/4.5 | rebuild hitches largely gone; large-dof scenes scale by bandwidth, not barriers |

The end state should put the quadtree method's solve in the same class as the
optimized restricted multigrid (~3–7 ms/substep), with the tolerance stop —
not a fixed budget — as the governing accuracy control.
