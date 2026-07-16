# Quadtree tall-cell remediation plan — level-set fidelity, topology staleness, solver throughput

Handoff document. Scope: the `quadtree-tall-cell` method (Narita et al., SIGGRAPH 2025,
`tmp/pdfs/sg2025narita.txt`). Prior audit: `docs/quadtree-paper-conformance-review.md` — its
R1–R5 items (per-step φ advection, per-step `refreshFaces`, φ-authoritative classification,
worker-based CPU prep, per-step rigid impulses, per-run quarter-depth optical layers, cached
builder buffers) are **already implemented** in the current tree. Do not redo them.

This plan addresses what the dam-break capture at t = 0.52 s shows is still wrong:
broken/patchy cells in the debug overlay, an over-large tall region, a stranded cubic band
bisecting the tall cells, and very slow steps.

---

## 1. Root-cause summary (why these exact changes)

Everything downstream — sizing oracle, cubic/tall segmentation, ghost-fluid boundary,
DOF wet-gating, the water render, and the debug overlay — now trusts a single resident
GPU level set (`WebGPUQuadtreeSurfaceState`, `lib/webgpu-quadtree-builder.ts`). That field
is being systematically degraded:

1. **The JFA redistance destroys sub-cell interface positions.** `finalizeDistance` writes
   `±(|x_cell − x_seed| + 0.5h)` clamped to `5h`. A seed (interface) cell is distance 0 from
   itself, so every interface cell is snapped to `|φ| = 0.5h` regardless of where the surface
   actually crosses it. Every 4th step (redistance cadence) all sub-cell information is
   erased: the surface staircases, the ghost-fluid θ used by the Eq. 25 free-surface scale
   quantizes (first-order BC, popping), and `|∇²φ|` in the sizing oracle spikes along the
   whole surface — fake curvature → near-maximal quadtree refinement → DOF blowup → slow.
2. **φ transport is first-order and uses a divergent velocity.** `advectLevelSet` is a
   single backward-Euler trilinear trace, and the surface state binds
   `resources.velocityIn` = `velocityB` = this step's *pre-projection* u\* (gravity applied,
   not divergence-free). Compressive transport plus first-order smearing shreds thin sheets
   — which is now directly visible because the renderer contours this φ
   (`volumeTexture` → `levelSetTexture`, `lib/webgpu-uniform-eulerian.ts`).
3. **No volume feedback of any kind.** VOF reconciliation was (correctly) retired, but
   nothing replaced it. The restricted tall-cell method got a narrow-band volume controller
   back in commit `27f2fe2`; the quadtree resident φ has none, so drift is unbounded.
4. **Topology lags the surface by 8–32 steps.** The cubic "optical band", tall segmentation,
   face graph, and IC factor are rebuilt at minimum every 8 steps
   (`quadtreeRebuildMinimumInterval`, `lib/webgpu-uniform-eulerian.ts`). At dam-break speeds
   (~3 m/s, CFL ≈ 0.6, ~1 cell/step) the band is placed where the surface *was* up to
   8+ cells ago — the stranded band bisecting tall cells — while the live surface sits inside
   a tall cell with only 2-sample (hydrostatic) vertical resolution. The paper rebuilds
   every step (Alg. 1).
5. **Segmentation over-triggers on non-SDF φ.** `populateTallPressureGrid`
   (`lib/quadtree-tall-cell-grid.ts:377`) marks a cell as interface when the leaf-centre
   profile flips sign **or** `|φ| ≤ min(h)`. Between redistances the advected φ is not
   unit-gradient, so interior values can dip through the magnitude test in swaths with no
   surface nearby — phantom cubic bands.
6. **Solver throughput.** (a) the iteration budget has a `min(2048, ⌈4√dof⌉)` floor with
   ~5–6 dispatches encoded per iteration every step (converged iterations become zero-size
   indirect dispatches but still pay encode/driver cost); (b) the IC(0) level-scheduled
   triangular solve runs in **one 256-lane workgroup** twice per iteration; (c) `dofActive`
   re-samples φ (4 texture loads per face slot) per row entry **per CG iteration**; (d)
   `faceVelocity` for vertical faces loops the leaf's full `span²·height` footprint per row
   entry during `initializeRow`.

The "broken cells" in the overlay are cell-scale wet/dry sign noise: the overlay colors each
voxel by live φ (`fluidSample > 0.5` via the mode-3 occupancy ramp,
`lib/webgpu-grid-overlay.ts:125`) inside segments classified at rebuild time — two time bases
in one image. Fixing φ removes the noise; item P2.4 makes the overlay honest regardless.

---

## 2. File map (current behavior, for orientation)

| File | Role |
|---|---|
| `lib/webgpu-quadtree-builder.ts` | `quadtreeSurfaceShader` (per-step φ advect + JFA redistance, `WebGPUQuadtreeSurfaceState`); `quadtreeConstructionShader` (sizing, refine, 2:1 smooth, `sampleLeafProfiles` leaf-centre φ readback); `WebGPUQuadtreeBuilder.build` |
| `lib/webgpu-quadtree-tall-cell.ts` | Pressure engine: WGSL (`refreshFaces`, `faceSamplePhi`, `faceVelocity`, `dofActive`, ICCG kernels, `precondition`/`preconditionJacobi`, MLS `project`, rigid coupling), `packSystem`, `prepareQuadtreeProjectionCPU`, rebuild orchestration |
| `lib/quadtree-tall-cell-grid.ts` | CPU/worker: `quadtreeFromPackedCells`, `populateTallPressureGrid` (cubic/tall segmentation), `buildVariationalSystem` (faces, IC(0), levels), `buildMlsProjectionRows`, `adaptivePressureCellTopology` (debug texture) |
| `lib/quadtree-topology-worker.ts` | Web Worker wrapper around `prepareQuadtreeProjectionCPU` |
| `lib/webgpu-uniform-eulerian.ts` | Orchestrator: dense advection, rebuild cadence (`shouldKickQuadtreeRebuild`), `encodeSurface` + `encode` per step, `volumeTexture` getter |
| `lib/webgpu-grid-overlay.ts` | Debug slice overlay (teal = tall∧wet, grey = cubic∧dry, blue = cubic∧wet, dark teal = tall∧dry; edges at topology-id transitions) |
| `lib/tall-cell-kernels.ts` | Restricted-method kernels to mirror: `traceDeparture` (RK2), `boundedMacCormack`, `volumeCorrectedPhi`, `strainMagnitude`, `curvature` |

Reference kernels for several items below already exist in `lib/tall-cell-kernels.ts` — copy
the math, not the bindings.

---

## 3. Phase 1 — restore the level set (correctness; do first)

### P1.1 Sub-cell-preserving redistance

**Where:** `quadtreeSurfaceShader` in `lib/webgpu-quadtree-builder.ts` — `seedDistance`,
`finalizeDistance`. (The construction shader has a near-duplicate JFA used at rebuild time;
apply the same change there, or better, delete it and make construction consume the resident
texture's already-redistanced φ — it already reads the resident texture for profiles.)

**Change:**

- `seedDistance` (unchanged criterion): seed = cell with a 6-neighbour sign change.
- `finalizeDistance`: replace the `+0.5h` floor with the seed's own sub-cell distance,
  gradient-normalized:

```wgsl
// Sub-cell distance at the seed cell: phi / |grad phi| (Chopp-style), clamped
// to the cell so a bad gradient cannot eject the estimate from the cell.
fn seedSubCellDistance(seed: vec3i) -> f32 {
  let h = params.cellAndDt.xyz;
  let g = vec3f(
    (loadPhi(seed + vec3i(1,0,0)) - loadPhi(seed - vec3i(1,0,0))) / (2.0 * h.x),
    (loadPhi(seed + vec3i(0,1,0)) - loadPhi(seed - vec3i(0,1,0))) / (2.0 * h.y),
    (loadPhi(seed + vec3i(0,0,1)) - loadPhi(seed - vec3i(0,0,1))) / (2.0 * h.z));
  let hMin = min(h.x, min(h.y, h.z));
  return clamp(abs(loadPhi(seed)) / max(length(g), 1e-6), 0.0, 0.87 * hMin);
}

@compute @workgroup_size(4, 4, 4)
fn finalizeDistance(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let word = distanceSeedsIn[index3(gid)];
  let hMin = min(params.cellAndDt.x, min(params.cellAndDt.y, params.cellAndDt.z));
  var distance = 5.0 * hMin;                       // far-field clamp unchanged
  if (word != 0xffffffffu) {
    let seed = vec3i(unpackSeed(word));
    if (all(seed == vec3i(gid))) {
      distance = seedSubCellDistance(seed);        // seed keeps its own sub-cell offset
    } else {
      distance = min(5.0 * hMin,
        sqrt(seedDistanceSquared(gid, word)) + seedSubCellDistance(seed));
    }
  }
  textureStore(phiOut, vec3i(gid),
    vec4f(select(distance, -distance, loadPhi(vec3i(gid)) < 0.0), 0.0, 0.0, 0.0));
}
```

No buffer-layout change: the seed word already carries the seed's coordinates, and `phiIn`
(the advected field) is already bound, so `seedSubCellDistance` costs 7 texture loads per
finalized cell. Sign still comes from the advected φ at the cell itself (unchanged — this is
what preserves thin features' topology).

**Note on the metric:** the JFA Voronoi search still compares distances to seed cell
*centres* (offset added only at finalize). That is the standard approximation; error is
sub-cell and acceptable. Do not try to fold the offset into `seedDistanceSquared` — it breaks
the JFA's metric consistency.

**Accept when:** a settled-tank probe shows the surface height constant to ≪ 0.5h across
redistance steps (today it snaps to the half-cell lattice every 4th step), and
`tests/quadtree-webgpu.test.ts` gains a case: initialize φ to an analytic tilted plane with
sub-cell offset, run one redistance, assert max |φ − exact| < 0.1h within the 3h band.

### P1.2 φ advection: divergence-free velocity + RK2 + bounded MacCormack

**Where:** `WebGPUQuadtreeSurfaceState` (constructor + `encode` + `advectLevelSet` in
`quadtreeSurfaceShader`), plus one call-site change in `lib/webgpu-quadtree-tall-cell.ts:644`.

**Change 1 — velocity source (one line, do first, biggest ratio of value to risk):**
construct the surface state with `resources.velocityOut` (= `velocityA`) instead of
`resources.velocityIn` (= `velocityB`). Ordering makes this correct: `encodeSurface` runs
before the projection writes `velocityA`, so at trace time `velocityA` holds the *previous
step's projected (divergence-free) velocity* — which is what the paper advects with
(Alg. 1 uses the saved previous grid). `velocityB` is this step's divergent u\*.

**Change 2 — RK2 midpoint trace** (mirror `traceDeparture`, `lib/tall-cell-kernels.ts:178`):

```wgsl
fn departurePoint(p: vec3f, dt: f32) -> vec3f {
  let cellsPerSecond = 1.0 / params.cellAndDt.xyz;   // component-wise
  let first = centredMacVelocityAt(p);               // trilinear MAC-centred sample
  let midpoint = p - 0.5 * first * dt * cellsPerSecond;
  return p - centredMacVelocityAt(midpoint) * dt * cellsPerSecond;
}
```

`centredMacVelocityAt` = trilinear interpolation of the existing `centredMacVelocity`
cell-centre values (or equivalently: per-component face-sample interpolation with the
half-index offset; the cell-centre trilinear form is what the restricted method uses and is
fine here).

**Change 3 — bounded MacCormack (BFECC) correction** (mirror `boundedMacCormack`,
`lib/tall-cell-kernels.ts:197-205`):

- `predict`: `phiPredicted(x) = phi(departurePoint(x, +dt))` — texture → scratchA.
- `reverse`: `phiReversed(x) = phiPredicted(departurePoint(x, −dt))` — scratchA → scratchB.
- `correct`: `phiNew = phiPredicted + 0.5·(phi − phiReversed)`, then clamp each cell to the
  min/max of the 8 φ corners surrounding the *forward departure point*; where the clamp
  trips, fall back to `phiPredicted`. Reads texture + scratchA + scratchB → writes scratchC.

Texture plumbing: the state currently owns `texture` + `scratch`; add one more `r32float`
3-D texture (n³ × 4 B — cheap) so the sequence is
`texture →(predict)→ scratchA →(reverse)→ scratchB`, `correct → scratchC`,
`redistance: scratchC → texture`. Rebuild the six cached bind groups accordingly (they are
built once in the constructor — extend the `group(...)` helper, no per-frame cost).

Dispatch cost: 3 advection-class dispatches instead of 1, same footprint as the VOF
MacCormack the dense path already runs. All per-step.

**Accept when:** the Zalesak-style column test (rotating notched column of φ in a fixed
velocity field for one revolution — add to `tests/quadtree-webgpu.test.ts`) retains > 95 %
of the notch volume vs ~70 % with the current first-order trace; visually, thin dam-break
sheets stop dissolving into voxel debris.

### P1.3 Redistance every step

**Where:** `WebGPUQuadtreeSurfaceState` constructor default `redistanceInterval = 4` →
`1` (`lib/webgpu-quadtree-builder.ts:148`).

With P1.1 the redistance is non-destructive, so running it per step is safe and removes the
4-step "staleness sawtooth" that P2.1's magnitude test relies on. Cost: `log2(maxDim)` JFA
passes (e.g. 7 at 128³) at 27 loads/cell/pass — same cost class as one MacCormack stage;
acceptable. If profiling later shows it matters, the optimization is a *banded* JFA (skip
cells whose incoming best distance already exceeds 5h — early-out at the top of `jumpFlood`),
not a longer interval.

### P1.4 Narrow-band volume control on the resident φ

Mirror the restricted method's controller (commit `27f2fe2`; kernels
`lib/tall-cell-kernels.ts:180-187`, controller math `lib/webgpu-eulerian.ts:830`).

**Mechanism:** one global normal-speed correction `w` (cells/s), applied only in the
interface band during advection:

```wgsl
// In the MacCormack `correct` entry point, after the clamp:
let hMin = min(params.cellAndDt.x, min(params.cellAndDt.y, params.cellAndDt.z));
let corrected = select(value, value - params.correctionSpeed * hMin * params.cellAndDt.w,
                       abs(value) < 1.5 * hMin);
```

**Controller (CPU, once per `readStats`):**
`w = clamp(0.5 · (V_ref − V_now) / max(interfaceCellCount, 1) / (1/30), −30, 30)` — verbatim
the restricted method's constants. `V_ref` = occupancy sum at scene start.

**GPU reduction for `V_now` and `interfaceCellCount`:** add a small kernel to
`quadtreeSurfaceShader` dispatched after redistance:

- per cell compute `occupancy = clamp(0.5 − φ/(4h_y·…), 0, 1)` (same ramp the renderer uses,
  `occupancyFromPhi`) and `isInterface = |φ| < 1.5·hMin`;
- workgroup-reduce (the 4×4×4 = 64-lane groups reduce in shared memory), then
  `atomicAdd` fixed-point (`u32`, ×2048 — same convention as the existing diagnostics
  reduction in the uniform solver) into a 2-word buffer;
- copy to a staging buffer read in `readStats` with the existing async, one-frame-lag
  pattern. Write `w` into the surface params UBO each step (`params` is already rewritten
  per `encode`; widen it by one float).

**Accept when:** `representedVolumeCellSum` drift computed *from φ* stays within ±0.5 % over
a 10 s dam break (add this as a smoke gate in `tools/run-webgpu-smoke.ts`); the retired
φ-vs-VOF mismatch diagnostic (still cheap to compute) stays near zero as a passive check.

---

## 4. Phase 2 — segmentation robustness and topology staleness

### P2.1 Sign-change-only interface detection

**Where:** `populateTallPressureGrid`, `lib/quadtree-tall-cell-grid.ts:375-378` (runs in the
worker; plain TS).

Replace the `|φ[y]| ≤ min(h)` OR-term: a cell is interface **only** if the profile sign flips
against a vertical neighbour, or its magnitude is small *and* a sign flip exists within ±1
cell. With P1.1–P1.3, φ is a real SDF every step, so sign changes alone place the band; the
guarded magnitude test only patches bilinear leaf-centre sampling at exact tangencies.

### P2.2 Staleness margin on the cubic band

**Where:** `populateTallPressureGrid` (band loop at `:382-394`), threaded from the
orchestrator through `prepareQuadtreeProjectionCPU`'s input.

Add `stalenessMarginCells: number` to the segmentation input. Band per surface crossing
becomes `[surfaceY − depthCells + 1 − margin .. surfaceY + margin]` (clamped to the column).
The upward extension is nearly free (air samples carry no DOF — they are dropped by
`buildVariationalSystem`'s liquid filter); the downward extension is the purchase: the live
surface can fall `margin` cells before it exits cubic coverage.

Orchestrator computes it at kick time
(`kickQuadtreeRebuild`, `lib/webgpu-uniform-eulerian.ts`):

```
margin = clamp(ceil((maxSpeed + g·T)·T / h), 1, 8)   // T = minInterval · typical dt
```

using `info.maxSpeed_m_s` (already tracked). This is the principled compromise for not
rebuilding every step: the topology is stale, but the surface never leaves the resolved band
while it is.

### P2.3 Faster rebuild cadence for fast scenes

**Where:** `lib/webgpu-uniform-eulerian.ts:88-89`.

Make the minimum interval speed-aware instead of the fixed 8:
`minInterval = clamp(round(0.5h / (maxSpeed·dt)), 2, 8)` — i.e., aim to rebuild after ~½ cell
of surface travel, floored at 2 to keep pipelining. Keep the hard max 32 and the existing
displacement/body triggers. P2.2 covers whatever staleness remains. (Rebuilds are pipelined
and off-main-thread already; the risk is worker throughput, which `quadtreeRebuildWall_ms`
telemetry will show — if the worker can't keep up at interval 2, the effective cadence simply
stretches, which is safe.)

### P2.4 Debug overlay: separate the two time bases

**Where:** `adaptivePressureCellTopology` (`lib/quadtree-tall-cell-grid.ts:85-100`) and the
overlay shader (`lib/webgpu-grid-overlay.ts:238-270`).

The packed topology voxel `.y` currently holds `firstY | (lastY+1) << 10`; bits 20+ are free.
Set **bit 20 = segment classified liquid at rebuild time**. Overlay: default mode colors
wet/dry from bit 20 (pure rebuild-time view — what the *solver topology* believes); keep the
live-φ coloring behind a toggle (a second `debug.w` flag value). Also draw a distinct outline
(e.g. dashed/red) where bit 20 disagrees with live φ — that band is exactly the staleness
visualization, which turns the current confusing picture into a diagnostic.

---

## 5. Phase 3 — paper-conformant sizing oracle

**Where:** `evaluateSizing` in `quadtreeConstructionShader`
(`lib/webgpu-quadtree-builder.ts:333-353`); CPU mirror
`quadtreeSizingFromVelocityAndSurface` (`lib/quadtree-tall-cell-grid.ts:1044-`).

The paper (Sec. 4.1, Sec. 6) uses **surface curvature + velocity variation** near the
surface (Ando et al. 2013), deliberately ignoring interior turbulence. Current code uses
`4·|∇²φ| + 3·|∇·u|`: the Laplacian is acceptable as curvature *once φ is an SDF* (P1), but
`|∇·u|` of a pre-projection field is mostly gravity/projection noise, not scene complexity.

**Change:**

1. Replace the velocity term with strain magnitude — the formula already exists as
   `strainMagnitude` (`lib/tall-cell-kernels.ts:176`): symmetric-gradient Frobenius norm of
   the MAC-centred velocity. This is translation-invariant (rigid translation scores 0),
   which is the paper's "variation in velocity" intent. Term: `w_vel · h · strainMagnitude`.
2. Keep `|∇²φ|` for curvature but evaluate it on the post-redistance φ (already true — the
   construction reads the resident texture) and optionally switch to the normalized
   `∇·(∇φ/|∇φ|)` form (`curvature`, `lib/tall-cell-kernels.ts:174`) for robustness at weak
   gradients.
3. Keep the `2·max(h)` evaluation band and the crossing-admittance unchanged.
4. **De-duplicate the weights**: export `QUADTREE_SIZING_WEIGHTS = { curvature: 4, velocity: 3 }`
   from `lib/quadtree-tall-cell-grid.ts` and consume it in both the GPU `staticData` upload
   (`lib/webgpu-quadtree-builder.ts:544`) and the CPU reference, so they cannot diverge.
   Retune after P1 lands (expect the curvature weight to need lowering once fake curvature
   is gone; do this against the DOF-count telemetry, target: dam-break liquid DOFs
   concentrated at crest/impact regions, not a uniform surface blanket).

**Accept when:** top-view leaf visualization of the dam break shows coarse leaves in calm
regions mid-collapse (paper Fig. 6 rightmost); `quadtreeLiquidDofCount` drops materially
(expect ≥ 2× on this scene) with no loss in the smoke matrix's front-progression gates.

---

## 6. Phase 4 — solver throughput (independent of Phases 1–3; can parallelize)

### P4.1 Hoist per-iteration φ/velocity work into per-step precompute

**Where:** pressure WGSL + `packSystem` in `lib/webgpu-quadtree-tall-cell.ts`.

Today, per CG iteration, `rowProduct` → `dofActive` re-derives liquid-ness by bilinear φ
sampling (4 texture loads) per face slot per row entry, and `faceGradient` samples φ again
per slot. Per solve, `initializeRow` additionally calls `faceVelocity`, which loops the
leaf's full `span²·height` footprint per row entry (so a face shared by N rows re-loops
N times).

**Change — all φ/velocity texture access moves to two per-step kernels; the iteration loop
becomes pure buffer math:**

1. Extend `refreshFaces` (`:329-339`) to also write, per face:
   - `liquidMask`: bit per slot, `faceSamplePhi(face, slot) < 0` — store in spare bits of
     `face.packed` (bits 22..28 are free: 16 span + 2 axis + 3 nodeCount + 1 ghost = 22 used;
     `faceNodeCount ≤ 7` fits 7 mask bits).
   - `flux`: `weights.x · faceVelocity(face) + solidFlux` — new `f32` field on the `Face`
     struct (update the WGSL struct, the `packSystem` stride, and the CPU packer in
     lockstep; struct is std430 storage, adding one 4-byte word after `solidFlux` keeps
     alignment trivial).
   This makes `refreshFaces` the *only* kernel that touches `levelSetIn`/`velocityIn`,
   once per face per step.
2. New tiny kernel `refreshRows` (dispatch ⌈dofs/128⌉ after `refreshFaces`): writes
   `rowActive[row] : u32` (new storage buffer, one word per row) by scanning the row's faces
   once and testing the precomputed `liquidMask`.
3. Rewrite the iteration-path readers:
   - `dofActive(row)` → `rowActive[row] != 0u` (one load).
   - `faceGradient`: replace `faceSamplePhi(...) < 0` with `(face.packed >> (22u + slot)) & 1u`.
   - `initializeRow`: `rhs += item.coefficient * face.flux`.
   - `faceSamplePhi` remains only inside `refreshFaces` and the MLS `project` path (which
     runs once per solve — fine).

Expected effect: the `multiply` kernel's memory traffic drops by an order of magnitude, and
`initializeRow` stops duplicating footprint loops. This is the single largest per-iteration
win available without touching the preconditioner.

### P4.2 Iteration budget from telemetry, not `4√n`

**Where:** `lib/webgpu-quadtree-tall-cell.ts:688-689` and `encode` (`:1017-1044`).

Replace `largeSystemFloor = min(2048, ⌈4√dof⌉)` with an adaptive budget:
`iterations = clamp(2 × EMA(pressureIterationsUsed), pressureIterations, 2048)` where the EMA
(α ≈ 0.25) is updated from the existing `pressureIterationsUsed` diagnostic each solve, and
the first solve after a topology change uses `max(budget, 4·previousUsed)` as a safety.
Rationale: the dam break converges in ~25 iterations but encodes 96+ (and up to 2048 on deep
scenes) at ~6 dispatches each; encode/driver cost of thousands of empty indirect dispatches
is pure waste. The GPU early-out (zeroed indirect args) stays as the correctness backstop —
if the budget is ever short, the best-iterate guard already reports `relativeResidual`, and
the next step's EMA doubles the budget. Keep the deep-water regression
(`docs/QUADTREE_TALL_CELLS.md` "whole tank in free fall") in the smoke matrix as the gate.

### P4.3 Preconditioner strategy

**Where:** `precondition`/`preconditionJacobi` (`lib/webgpu-quadtree-tall-cell.ts:409-449`).

Facts to respect: the operator is SPD but not an M-matrix (paper Sec. 5), MIC and plain
Jacobi have already measured worse than IC(0) on the deep-water system
(`docs/QUADTREE_TALL_CELLS.md`), and the IC(0) triangular solves are inherently
level-serialized — one 256-lane workgroup, twice per iteration.

Ordered plan:

1. **Re-measure after Phases 1–3 + P4.1/P4.2**, because the DOF count and iteration counts
   both drop; IC(0)'s single-workgroup latency may stop being the bottleneck at the new
   sizes. Use `gpuTimings.pressure_ms` and `pressureIterationsUsed` on: dam break (this
   scene), deep tank, settled tank.
2. If the triangular solve still dominates: **block-Jacobi-IC** — factor each *level-scheduled
   independent subtree/column block* separately so `precondition` can dispatch one workgroup
   per block (the level schedule already exists; the change is partitioning the factor by
   connected component / column group in `buildVariationalSystem` and dispatching
   `numBlocks` workgroups). Tall-cell grids are strongly column-structured, so per-column
   tridiagonal blocks capture most of the coupling — the earlier "per-column tridiagonal
   block-Jacobi measured worse" note was against the *full* system; as a *block* precondition
   inside CG at post-Phase-3 sizes it deserves one re-measurement, cheaply behind the
   existing `preconditioner` option flag.
3. Endgame (only if needed): **geometric multigrid on the quadtree** — the paper's own
   suggestion. V-cycle over the leaf hierarchy: smoother = damped Jacobi (fully parallel),
   restriction/prolongation = the same volume-weighted variational stencils the face builder
   already computes (Galerkin coarse operators come free from the variational form). This is
   a multi-week item; do not start it before 1–2 above are measured.

### P4.4 Rebuild critical path (only if telemetry says so)

`quadtreeCPUTopologyPack_ms` / `variationalAssembly_ms` are already worker-side and
pipelined; with P2.3 shortening the cadence, watch them. If the worker becomes the limiter,
the two GPU-able stages are segmentation and face enumeration — see Appendix A. Do not move
them preemptively.

---

## 7. Suggested execution order and hand-off sizing

| Order | Item | Size | Risk | Depends on |
|---|---|---|---|---|
| 1 | P1.2 change 1 (velocity binding) | XS | none | — |
| 2 | P1.1 redistance | S | low | — |
| 3 | P1.3 per-step redistance | XS | low | P1.1 |
| 4 | P1.2 changes 2–3 (RK2 + MacCormack) | M | medium (bind-group plumbing) | — |
| 5 | P1.4 volume control | M | low (pattern exists) | P1.1–P1.3 |
| 6 | P2.1 segmentation | XS | low | P1.* |
| 7 | P2.2 staleness margin | S | low | — |
| 8 | P2.4 overlay honesty | S | none | — |
| 9 | P2.3 cadence | XS | low | P2.2 |
| 10 | P3 sizing oracle | M | medium (retuning) | P1.* |
| 11 | P4.2 iteration budget | S | low | — |
| 12 | P4.1 face/row precompute | M–L | medium (struct stride) | — |
| 13 | P4.3 preconditioner | L | high | P1–P3, P4.1–2 |

Items 1–3 alone should visibly change the screenshot (surface stops staircasing and
shredding; sizing stops blanket-refining). Items 11–12 are safe to run in parallel with
Phase 1 by a second person — they touch disjoint code.

## 8. Verification

- **Unit:** new redistance sub-cell test, Zalesak rotation test (P1); keep
  "packed GPU leaf maps reconstruct the CPU quadtree exactly" and the dam-break column tests
  green (`tests/quadtree-tall-cell-grid.test.ts`, `tests/quadtree-webgpu.test.ts`,
  `tests/tall-cell-paper-conformance.test.ts`).
- **Smoke:** `tools/run-webgpu-smoke.ts` matrix (settled tank, dam break + boxes, hose,
  sphere jet, deep water). Add gates: φ-volume drift ±0.5 % (P1.4); settled-tank kinetic
  energy monotone decay (no redistance-cadence sawtooth); front-progression unchanged
  within tolerance after P3.
- **Probes:** `tmp/tall-cell-audit/probe-*.ts` — collapse and boxes probes for energy and
  popping at rebuild boundaries.
- **Visual A/B on this exact scene** (dam break, adaptive, t ≈ 0.5 s, overlay on): cubic
  band tracks the surface (and the P2.4 disagreement outline is thin); teal ends at the
  true surface; no voxel-noise holes; `quadtreeLiquidDofCount` and `gpuTimings.pressure_ms`
  recorded before/after each phase.

---

## Appendix A — full-GPU residency map (requested: how every remaining function could run on the GPU)

Current split after this plan: everything per-step is GPU; per-rebuild work is GPU
(construction kernels) + worker (symbolic/sparse). If/when the worker becomes the
bottleneck, this is the porting path, in order of feasibility:

| Stage (worker today) | GPU formulation |
|---|---|
| `quadtreeFromPackedCells` (decode + validate) | Already GPU-native upstream: the packed leaf map *is* the GPU format. Decode exists only to build CPU-side arrays; with the stages below on GPU it disappears rather than ports. Keep the three validation sweeps as a debug-only path. |
| `populateTallPressureGrid` (cubic/tall segmentation) | One thread per leaf column (the builder's `sampleLeafProfiles` already iterates y per leaf — same shape). Two passes: (1) walk the profile, count segments and samples per leaf; (2) exclusive prefix-scan over leaf counts (standard workgroup scan + block offsets), then re-walk and emit samples/segments at scanned offsets. Output: the same packed sample/segment arrays `packSystem` consumes. |
| `buildVariationalSystem` — face enumeration | Per leaf: neighbours via the leaf-owner map (one `textureLoad`/word fetch per side at the finer of the two sizes — T-junction faces emerge naturally as one face per finer neighbour). Horizontal faces: thread per (leafPair, segmentPair) with interval intersection of the two columns' segment lists; vertical faces: thread per column segment boundary. Same count-scan-emit pattern as above. Coefficients (Eq. 2–4 interpolation weights, `±1/Δx`) are closed-form per face — no global state. |
| CSR assembly (rowOffsets/rowEntries) | Faces emit (row, entry) pairs; radix-sort by row (GPU radix sort, 32-bit keys) + run-length scan → `rowOffsets`. Or atomically increment per-row counters then scan — simpler, adequate at these sizes. |
| IC(0) factorization + level schedule | **Do not port.** Incomplete factorization is sequential by nature; a GPU port buys nothing (this is why P4.3 moves toward block-Jacobi-IC — blocks factor independently, so per-block factorization *is* GPU-parallel: one workgroup per block, sequential within — or multigrid, which needs no factorization at all). Until then, factorization stays in the worker; it is off the frame path. |
| `buildMlsProjectionRows` | Thread per eligible face: gather ≤ 32 sub-face sample positions, build the 4×4 normal-equation matrix in registers, solve by Cholesky in-register (fixed 4×4 — ~60 FLOPs), emit weights + the conservation shift. The 150k-row cap then becomes unnecessary; if kept, emit with an atomic cursor and `log`/flag truncation instead of silently breaking. |
| `solidFieldsFromBodies` (8-corner voxelization) | Thread per (cell, body) with an analytic SDF per body shape; area fractions by corner counting exactly as the CPU does. Trivial port; only worth it with many/large bodies. |
| Upload of packed buffers | Disappears for GPU-produced stages (they write storage buffers directly); the projection constructor then only rebinds. |

Ordering note: the value of Appendix A is gated on P2.3 telemetry. The per-step path
(advection, redistance, volume control, `refreshFaces`/`refreshRows`, ICCG, MLS projection,
rigid impulses) is already fully GPU-resident after Phases 1 and 4; the rebuild path is
pipelined and does not block physics. Port rebuild stages only when
`quadtreeRebuildWall_ms` demonstrably limits the achievable cadence.
