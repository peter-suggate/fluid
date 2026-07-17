# Plan: persistent-CG megakernel for the quadtree tall-cell pressure solve

Handoff plan. Everything referenced here was measured/validated on 2026-07-17 on
dam-break-ui (61x46x41, ~15k pressure DOF) with the Dawn node harness.

## 0. Context and preconditions (read first)

- **This builds on uncommitted work.** The working tree (not HEAD) contains the
  "fused CG iteration" rework of `lib/webgpu-quadtree-tall-cell.ts` (kernels
  `multiplyPartial`, `applyStepAlphaUpdate*`, `reduceInitialNorm`, warm start,
  asymmetric iteration-budget EMA, per-step solve-feedback readback). The
  megakernel replaces that fused *ladder* on eligible solves; do not implement
  against HEAD. See `docs/QUADTREE_TALL_CELLS.md` (section starting "Measured on
  dam-break (16k DOF)...") for what already landed.
- **A second session is concurrently editing this tree** (volume-controller /
  pack-scan work in `lib/webgpu-quadtree-pack-builder.ts`,
  `lib/webgpu-eulerian.ts`, `lib/methods/tall-cell.ts`, tall-cell tests). Do not
  touch those files; do not `git stash` or commit without coordinating with the
  user. Known mixed-tree smoke failures NOT to chase: garden-pond /
  dam-break-boxes represented-volume drift ~-9 % (theirs), one failing
  restricted-tall-cell unit test ("remeshed cell assignment ... Section 8",
  theirs), deep-water "GPU pack matrix" 257 MB > 128 MB binding limit
  (pre-existing capacity bug, out of scope here).
- Harness: every GPU command needs
  `WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js`.
  If that path is missing (machine rebooted), recreate via
  `tmp/tall-cell-audit/post-reboot.sh`.

## 1. Motivation (measured)

The CG loop is dispatch-count-bound, not ALU-bound: each compute dispatch costs
~15-20 µs of fixed decode/usage-scope overhead on Dawn/Metal regardless of
workload, and post-convergence no-op indirect dispatches cost near-full price.
After the fused-ladder work, a poly-2 iteration is 6 dispatches ≈ 0.12 ms; a
~42-iteration dam-break solve spends ~5 ms/step in the loop while the actual
math for 15k unknowns is microseconds. A single-workgroup kernel that runs the
*entire* CG loop in ONE dispatch removes all per-iteration dispatch overhead
and the entire encoded-budget/no-op-tail mechanism for eligible solves.
Expected: pressure loop ~5 ms → ~1.5-2.5 ms/step on dam-break (memory-bound on
one GPU core: ~2-3 MB of state traffic per iteration), and near-zero cost on
settled scenes (warm start already converges those in 0-1 iterations; the
megakernel makes the remaining fixed cost one dispatch instead of a ladder).

## 2. Scope and gating

Implement a new WGSL entry point `solveMegakernel` in the big solver shader
string in `lib/webgpu-quadtree-tall-cell.ts` (same module as `multiply`,
`applyStep*`; bind group layout unchanged — it uses only existing bindings).

Eligible when ALL hold (CPU-side gate, mirror of `inlineSupported` at
~line 1660 plus a size threshold):
- `preconditioner` is `poly` or `jacobi` (default is poly-2);
- not rigid-coupled (`couplingBodyCount === 0`) — the coupled path inserts
  `coupleReduce`/`coupleApply` between SpMV and the dot product and stays on
  the ladder;
- `dofCount <= megakernelThreshold` (start at 32768; tune in §7). A stale
  dofCount (inline rebuilds refresh it via the async monitor readback) is
  safe — the threshold is purely a perf heuristic, the kernel is correct at
  any size.

Everything else (ic0/blockic/line, coupled scenes, dof above threshold) keeps
the existing fused ladder unchanged.

Controls: option `megakernelSolve?: boolean` on
`QuadtreeTallCellProjectionOptions` (default **true**), env
`FLUID_QUADTREE_MEGAKERNEL=0|1` wired in `tools/run-webgpu-smoke.ts` next to
the other `quadtree-tall-cell` overrides (search `FLUID_PRESSURE_WARM_START`
for the pattern: env const near top, `values.*` mapping near line ~620, then
`lib/methods/quadtree-tall-cell.ts` maps `values` → options in BOTH the
`createSolver` and `createSolverAsync` blocks — grep `pressureWarmStart` there
and mirror it).

## 3. Contracts the kernel must honor

State slots (per-row, `stateF(row, FIELD)`): `PRESSURE`, `BEST_PRESSURE`,
`RESIDUAL`, `DIAGONAL`, `PRECONDITIONED`, `DIRECTION`, `MATRIX_DIRECTION`,
`ACTIVE_FLAG`. Matrix access: `matrixStart/matrixNode/matrixCoefficient`;
rows with `!dofActive(row)` are identity (`rowProduct` returns DIRECTION).

Scalars (buffer is `clearBuffer`ed at the top of every `encode()`, so nothing
persists across solves): on exit the kernel must leave the same telemetry the
ladder leaves, because `readSolveDiagnostics`/`applySolveFeedback`
(~line 2470+) and the smoke invariants consume them:
- `scalars[3]` = |b|² (clamped ≥ 1e-30) — **relative to b, not to the warm
  residual**; this is the paper's stop-test semantics.
- `scalars[2]` = final residual² ; `scalars[7]` = minimum residual² seen.
- `scalars[9]` = iterations actually executed (drives the budget EMA — harmless
  on the megakernel path but keeps telemetry truthful).
- `scalars[4]`/`scalars[6]` = last alpha/beta (diagnostic only).
- `BEST_PRESSURE` snapshots whenever current rr ≤ min-so-far (the best-iterate
  guard that later stages consume — `mapPressure` reads BEST_PRESSURE).
- Do NOT touch `factorColumns` control words 12-33 or `dispatchArgs` — no
  ladder is encoded on this path so there is nothing to publish/zero.

Stop test: `rr <= params.solve.x * scalars[3]` where `params.solve.x` = tol²
(tolerance = max(scene tol, 1e-4)).

Warm start: reuse the existing `initializeRow(row)` helper — it already reads
x₀ from `mappedPressureIn` via the aux dof-sample table when
`params.couplingCounts.z != 0`, computes r₀ = b − A·x₀, and stages b in
`MATRIX_DIRECTION` for the |b|² reduction. The megakernel must compute |b|²
from `MATRIX_DIRECTION` *before* its first SpMV overwrites it (same reason
`reduceInitialNorm` exists on the ladder).

## 4. Kernel design

`@compute @workgroup_size(256)` (see §7 for trying 512/1024 via a device-limit
request), dispatched as ONE workgroup. All phases separated by
`storageBarrier(); workgroupBarrier();` — single-workgroup execution is what
makes storage-buffer coherence sufficient; this pattern is already used by the
`precondition` (IC0) and `applyStep`/`finishIteration` single-workgroup
kernels in this module — copy their style.

Pseudocode (poly-2; jacobi is the degenerate z = r/D case):

```
let lid = local_invocation_id.x; let n = dofs();
// Phase 0: init (warm start inside initializeRow)
for (row = lid; row < n; row += WG) { initializeRow(row); }
barrier;
// Phase 1: |b|^2 from staged MATRIX_DIRECTION, rr0 from RESIDUAL,
//          z0 = poly(r0), d0 = z0, rz0 = r0.z0
//   (poly(r): z = 0.5*r/D; repeat degree-1 times: t = A z; z += 0.5*(r-t)/D
//    — t can live in MATRIX_DIRECTION *after* bb is reduced)
reduce bb, rr  -> scalars[3] = max(bb,1e-30); scalars[2] = rr; scalars[7] = rr;
barrier; poly passes with barriers; set DIRECTION = PRECONDITIONED;
reduce rz -> workgroup var (and scalars[0]);
// Loop: cap = hard iteration budget (see below)
for (it = 0; it < cap; it++) {
  if (workgroupUniformLoad(&converged)) { break; }
  // q = A d ; pAp = d.q          (one fused pass, like multiplyPartial)
  // alpha = rz/max(pAp,1e-30); p += alpha d; r -= alpha q; iterations += 1
  // z = poly(r)                  (reuses MATRIX_DIRECTION as scratch)
  // rz' = r.z ; rr = r.r         (fused into the last poly pass)
  // beta = rz'/rz; rz = rz'; if rr <= min: BEST_PRESSURE = PRESSURE, min = rr
  // d = z + beta d
  // converged = rr <= tol2 * bb   (write by lid 0 into var<workgroup>)
}
// Exit: scalars[2]=rr, scalars[7]=min, scalars[9]=f32(iterations), [4]/[6].
```

WGSL pitfalls the implementer must respect:
- **Uniform control flow for barriers**: the loop-break condition must come
  from `workgroupUniformLoad(&convergedFlag)` (a `var<workgroup>` written by
  lane 0 in the previous phase) — a plain read will trip the uniformity
  analyzer or, worse, validate but hang.
- Reductions: reuse the module's `reducePair` pattern (`reductionA/B`
  `var<workgroup>` arrays, 256 entries, stride halving). Per-thread strided
  accumulation first (`for (row = lid; row < n; row += 256)`), then tree
  reduce, then lane 0 stores to a `var<workgroup>` scalar; barrier; all lanes
  read the scalar. alpha/beta never need to round-trip through `scalars[]`
  inside the loop.
- Iteration cap must be a *uniform* value. Suggested plumbing: pack it into the
  upper bits of `params.couplingCounts.z` (bit 0 is already the warm-start
  flag; the cap ≤ 2048 fits in bits 1..12). It is written once in the
  constructor (`writeBuffer(this.params, 80, ...)`, ~line 1527 — search
  `couplingCounts.z doubles as the warm-start flag`). Use
  `min(hardBudget, 2048)` from `quadtreeIterationBudget(...).hardBudget`
  (helper at ~line 185). Do NOT use `params.counts.z` (that is the ladder's
  encoded budget and the feedback loop rewrites it).
- Metal watchdog: worst case (2048 iterations × 32k rows) is tens of ms of GPU
  time in one dispatch — safe, but do not raise the 2048 cap.

## 5. CPU-side integration (`encode()`, ~line 2185)

In `encode()` where the pass structure is built:

1. Compute `const megakernel = this.megakernelEligible` (gate from §2,
   computed in the constructor next to `inlineSupported`, ~line 1660).
2. When megakernel is active, the pass becomes:
   `refreshFaces` (indirect 112 / direct), `refreshRows` (indirect 100 /
   direct), then `direct("solveMegakernel", 1)`. Skip `setup`'s
   initialize/precondition/reduceInitial dispatches, skip the `iterations(...)`
   ladder, and skip the pre-loop `dispatchPipeline` (`updateDispatch`) — the
   kernel handles init, |b|², the loop, and convergence internally.
   `mapPressure` + `project` + clamp/extrapolation/divergence stay exactly as
   they are.
3. Phase timings (`debugPressureTimings`, ~line 2295): put the whole megakernel
   dispatch in the "firstIterations" window (writes(2,3)) and leave the
   "remainingIterations" window empty — document that in the one-line comment.
   Do not drop the timestamps; the perf protocol below uses them.
4. Register the entry point in BOTH pipeline name lists (sync list ~line 1636
   `const names = [...]`, async list in `compilePipelinesAsync` ~line 1690 —
   grep `"reduceInitialNorm"` and add alongside).
5. `applySolveFeedback` keeps working unchanged (it only reads scalars). The
   budget EMA will track the megakernel's true iteration counts; that is fine
   because the ladder still uses it whenever the megakernel is ineligible.

## 6. Validation protocol (in order; do not skip)

All commands from repo root with the `WEBGPU_NODE_MODULE` env from §0.

1. `npx tsc --noEmit` — 3 pre-existing errors mention
   `volumeCorrectionDivergenceRate_s`; anything else is yours.
2. `node --import tsx tools/validate-quadtree-shaders.ts` → "All quadtree
   shader pipelines are valid".
3. `node --import tsx --test tests/quadtree-webgpu.test.ts` → 25+ pass
   (add a test asserting the megakernel source contains
   `workgroupUniformLoad` and that `solveMegakernel` never references
   `dispatchArgs`/control words — cheap regex tests in the existing style).
4. Numerical A/B, megakernel off vs on (`FLUID_QUADTREE_MEGAKERNEL=0|1`):
   ```
   FLUID_SCENE=dam-break-ui FLUID_METHOD=quadtree-tall-cell,uniform \
   FLUID_TARGET_S=1.5 FLUID_CHECKPOINT_EVERY_S=0.25 FLUID_CPU_ORACLE=0 \
   FLUID_FIELD_STATS=1 FLUID_DISABLE_TIMESTAMPS=1 \
   node --import tsx tools/run-webgpu-smoke.ts
   ```
   Acceptance (compare to the off-run, and to this session's reference values):
   - checkpoint `wetIntersectionOverUnion` vs uniform within ±0.05 of the
     off-run at every checkpoint (reference trace ≈
     .72/.56/.55/.57/.71/.71; the established uniform-vs-tall parity bar is
     ~0.6-0.68 at t=1 s);
   - `peakKineticEnergyProxy` ≈ 0.45 ± 0.02;
   - `maximumPressureRelativeResidual` ≤ 1e-4 (hard requirement);
   - quadtree `simulationWall_ms/steps` ≤ the off-run (target ≤ ~4.5 ms/step;
     off-run reference 5.77);
   - no NEW invariant-failure lines vs the off-run. Pre-existing failure
     classes on this scene: represented-volume drift >1 %, "wall exceeds 2x
     GPU 0.00" (artifact of FLUID_DISABLE_TIMESTAMPS), level-set drift peak
     ~0.095, dominant-component/components counts.
5. Calm-scene check (exercises warm start + zero-iteration solves):
   ```
   FLUID_SCENE=settled-tank FLUID_METHOD=quadtree-tall-cell FLUID_TARGET_S=1.0 \
   FLUID_CPU_ORACLE=0 FLUID_FIELD_STATS=0 node --import tsx tools/run-webgpu-smoke.ts
   ```
   `quadtreePressureIterationsUsed` ≤ 1, `maxSpeed_m_s` ≈ 1.15e-3 ± 10 %,
   wall/step ≤ the off-run.
6. Ladder regression guard (must be byte-identical code paths): rerun step 4
   with `FLUID_QUADTREE_PRECONDITIONER=ic0` (forces ladder+async path) and
   confirm behavior matches the pre-change tree.
7. All-scenes sweep: `FLUID_SCENE=all FLUID_METHOD=quadtree-tall-cell
   FLUID_TARGET_S=0.2 FLUID_CPU_ORACLE=0` — accept the pre-existing failures
   listed in §0; anything new is yours.

## 7. Perf tuning (after correctness)

- Timing tool: add `FLUID_PRESSURE_PHASE_TIMINGS=1` to a 0.5 s dam-break run
  and read `quadtreePressurePhaseTimings` (setup/first/remaining/project) plus
  `gpuTimings.pressure_ms`. Reference (ladder, this session): setup 0.39,
  iterations ≈ 7.2, project 3.2 ms on the last step.
- Threshold sweep: dam-break-ui gives ~15k DOF. For larger systems either use
  `FLUID_SCENE=hose-tank`/`garden-dam-break` or temporarily lower the threshold
  to force ladder-vs-megakernel on the same scene. Pick the crossover, set the
  default threshold, and record the numbers in `docs/QUADTREE_TALL_CELLS.md`.
- Workgroup size: try 256 first. Optionally request
  `maxComputeInvocationsPerWorkgroup`/`maxComputeWorkgroupSizeX` ≥ 512 in
  `requiredLimits` (device creation is in `tools/run-webgpu-smoke.ts` ~line 595
  and the browser path in `lib/webgpu-renderer.ts`) and template the size into
  the shader string; keep 256 as the fallback when the adapter refuses.
- If the SpMV is the bottleneck, consider sorting rows so each thread's strided
  rows are contiguous — but measure first; do not restructure `matrixWords`.

## 8. Risks / fallbacks

- Hang risk (barrier in non-uniform flow) → the `workgroupUniformLoad` rule in
  §4; if a run wedges, Dawn usually device-losts after ~2-5 s; the kill switch
  is the env/option gate.
- Numerical drift vs ladder: expected at FP-reordering level only; CG converges
  to the same tolerance, and the acceptance bars in §6 are trajectory-level.
  If IoU diverges beyond ±0.05, suspect a phase-ordering bug (e.g. beta using
  post-update rz on the wrong side), not FP noise.
- Single-SM latency: if the megakernel is *slower* than the ladder at 15k DOF,
  check that per-thread row loops are strided (`row = lid; row += WG`) so loads
  coalesce, and that poly scratch reuses `MATRIX_DIRECTION` rather than adding
  a new state slot.
- Keep the ladder fully intact. The megakernel must be deletable by flipping
  one boolean.

## 9. Out of scope (do not do here)

- Band-restricted dense tier (mapPressure/project/extrapolation over 115k
  cells — separate ~3 ms/step opportunity).
- Deep-water resident-pack 257 MB binding-limit fix.
- ic0-in-megakernel (its level-scheduled solve is already single-workgroup and
  could fold in later — note it, don't build it).
- Anything in `lib/webgpu-quadtree-pack-builder.ts` (concurrent session owns it).
