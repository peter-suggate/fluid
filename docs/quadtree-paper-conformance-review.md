# Handoff: Critical review of the quadtree tall-cell implementation vs. Narita et al., "Quadtree Tall Cells for Eulerian Liquid Simulation" (SIGGRAPH 2025)

## 1. Goal and scope

**High-level goal:** the repo implements the Narita et al. SG2025 quadtree tall-cell liquid simulator on WebGPU (`quadtree-tall-cell` method). This document is a paper-conformance and performance audit answering four questions:

1. Where does our approach differ from the paper?
2. What still runs on the CPU that should run on the GPU?
3. Where does the math differ (notably: conservative VOF vs. level set)?
4. What differences cause **poor remeshing** and **poor performance**?

It ends with a prioritized remediation plan with enough implementation detail to execute.

**Scope note:** the repo has TWO tall-cell solvers. The *restricted* tall-cell method (`lib/webgpu-eulerian.ts` + `lib/tall-cell-kernels.ts`, Chentanez-style fixed band) is mid-migration from VOF to level set **by the user, in the working tree right now — do not touch it**. This review covers only the quadtree method:

- `lib/methods/quadtree-tall-cell.ts` — method registration, defaults (ICCG max 96–240 iters, rel. tol 1e-4, `maximumLeafSize` 8/16, `opticalDepthFraction` 0.25, `adaptivityStrength` 1).
- `lib/webgpu-uniform-eulerian.ts` — the orchestrator. A *dense uniform fine-grid* solver (MacCormack advection, VOF transport, optional sharpening) that delegates pressure projection to the quadtree engine and kicks asynchronous topology rebuilds.
- `lib/webgpu-quadtree-builder.ts` — GPU Sec. 4.1 construction: φ advection, jump-flood redistance, sizing scan, quadtree refine/balance/dilate; one compact readback (φ field + packed leaf words).
- `lib/quadtree-tall-cell-grid.ts` — CPU machinery: quadtree decode, tall pressure grid population (Sec. 4.2), variational system assembly (Sec. 4.3), MLS projection rows (Alg. 1 line 10), sizing reference, redistancing reference.
- `lib/webgpu-quadtree-tall-cell.ts` — projection engine: CPU sparse pack + IC(0) factorization + level scheduling, GPU ICCG solve in one compute pass with convergence-driven indirect dispatch, monolithic rigid coupling, MLS-corrected velocity update.

Paper text: `tmp/pdfs/sg2025narita.txt`. Related references the code leans on: Ando & Batty 2020 (octree liquid, Eq. 25 SPD free-surface scale, Eq. 38 adaptivity strength, MLS Eq. 33–35), Batty et al. 2007 (variational coupling), Irving et al. 2006 (optical layer = ¼ depth).

## 2. Paper summary (what conformance means)

Per-step loop (Alg. 1), **executed every step**:

1. Save current grid + variables.
2. Evaluate sizing function on previous cells → vertical max-scan → flatten to 2D (Sec. 4.1). Sizing = surface curvature + velocity variation near the surface only (inner turbulence deliberately ignored).
3. Subdivide quadtree coarse-to-fine (`split if demand > 1/width`), avoid rapid coarsening via Ando–Batty adaptive smoothing (2:1 + dilation), not full tree balancing.
4. Extend 2D leaves vertically into columns; advect **velocity and level set** semi-Lagrangian at the *new adaptive sample locations*, interpolating from the saved previous grid (Klingner et al. 2006); extrapolation as Ando–Batty.
5. Advance rigid bodies.
6. Concatenate vertical cells a few cells from the surface into tall cells, downward AND upward; multiple tall runs per column for splashes/bubbles. Pressure samples: cubic centres; tall-cell top + bottom (Fig. 2).
7. Solve pressure variationally: −[∇]ᵀ[V][A][F][∇]{p} = −[∇]ᵀ[V][A]{u*} (Eq. 1); gradient via vertically interpolated endpoint pairs over the face-midpoint (Eq. 2–4, Δx = distance between interpolated samples, e.g. 1.5Δx at a 2:1 transition); face velocity = area average over the shared boundary (Eq. 5, via "collapse and average"); [F],[A] exactly as Ando–Batty (second-order free-surface BC); two-way rigid bodies monolithically via Eq. 14 (Δt[∇ᵀVAF∇ + JᵀMs⁻¹J]{p} = ∇ᵀVA{u*} − Jᵀ{w*}).
8. Map pressure back onto cubical cells (split tall cells, MLS interpolation per Ando–Batty), update velocity and rigid bodies.

Solver: ICCG (plain IC(0); MIC performs poorly because the operator is not an M-matrix), relative residual 1e-4. Surface: level set (+ EXNBFLIP particles for visual enrichment), dual contouring for meshing. Their implementation is CPU (Ryzen 5950X); GPU is listed as future work — so "should be on GPU" below is about *our* real-time WebGPU goals, not paper conformance.

## 3. What already conforms well (verified, no action needed)

- **Variational gradient/divergence** (`buildVariationalSystem`, `lib/quadtree-tall-cell-grid.ts:675`): per-neighbor-pair faces with one face per touching fine neighbor at T-junctions (:741), vertical interpolation of endpoint samples at the face midpoint (:754), coefficients ±1/Δx with Δx = (left.size+right.size)·h/2 (:743) — exactly Eq. 2–4. Divergence is the transpose (same coefficients reused row-wise).
- **[V]/[A]/[F]**: face-cell volumes including inner ghost volumes between stacked samples (:798, flagged as an authors' hindsight correction), solid open-fraction, and the Ando–Batty Eq. 25 SPD fluid scale with θ-floor `maximumFluidScale = 100` (:425–433).
- **Eq. 5 face velocity**: simple sub-face averaging on GPU (`faceVelocity`, `lib/webgpu-quadtree-tall-cell.ts:292`), CPU equivalent in assembly.
- **ICCG choice**: plain IC(0) — the MIC dropped-fill compensation is explicitly multiplied by 0 with a comment citing the paper's non-M-matrix finding (`lib/webgpu-quadtree-tall-cell.ts:822`); rel. tol 1e-4 (`lib/methods/quadtree-tall-cell.ts:35`); convergence early-exit without CPU readback via indirect zero-workgroup dispatch (`quadtreeDispatchShader`, :527).
- **Quadtree build**: coarse-to-fine sizing test with Eq. 38 pseudo-width (`pseudoCellWidth` :164 / GPU `refine` `lib/webgpu-quadtree-builder.ts:156`), strict 2:1 balance + 3 dilation passes (`buildQuadtree` :272; GPU `smoothTopology` :179), demand = max over the candidate leaf footprint (correct — a centre sample would miss sub-leaf features).
- **Tall-cell population** (`populateTallPressureGrid` :358): φ sampled at the leaf horizontal centre (per authors' clarification, :336 comment), interface band both sides of each sign change, multiple tall runs per column, samples at bottom/top replaced cube centres.
- **MLS pressure mapping** (Alg. 1 line 10): linear MLS with trilinear-hat weights per Ando–Batty Eq. 33–35 (`mlsWeightsAt` :873), conservation shift so sub-face corrections average to the solved face value (`buildMlsProjectionRows` :935), constant prolongation fallback where the reconstruction crosses air samples.
- **Monolithic rigid coupling**: rank-6 K = ∇ᵀV(1−A)L per body, K M⁻¹ Kᵀ added to the matrix, solid constraint flux (A·u_fluid + (1−A)·u_solid) in the RHS (:696–:857) — structurally and unit-wise equivalent to Eq. 8–14 (ρ folded into M⁻¹, impulse = −ρKᵀp̂; verified dimensionally).

## 4. Findings

### A. Approach differences → the main remeshing/quality gaps

**A1 (CRITICAL). The pressure solve's free surface is frozen between topology rebuilds; the paper refreshes it every step.**
The orchestrator rebuilds topology at an *adaptive cadence*: minimum every 8 steps, forced at 32, earlier only if a conservative speed bound says the interface moved ≥ 0.5h (`shouldKickQuadtreeRebuild`, `lib/webgpu-uniform-eulerian.ts:320`). Between rebuilds:
- `levelSetTexture` is only re-uploaded during a rebuild (`uploadLevelSet` at construction / `rebuildFromState` reuse path, `lib/webgpu-quadtree-tall-cell.ts:999`).
- `refreshFaces` (which recomputes the Eq. 25 fluid scale from the level-set texture) only runs when `facesDirty` is set — which happens **only** on a topology-reuse rebuild (:953, :1000). On a topology-change rebuild the new projection gets fresh weights once; then nothing until the next rebuild.
- The liquid/air DOF classification, cubic-vs-tall segmentation, and cell→face projection map are all baked at rebuild time.

Consequence: for up to 8–32 steps, incompressibility is enforced over a stale liquid region with a stale ghost-fluid boundary. A settling or rising surface gets a systematically misplaced Dirichlet boundary → pressure errors that read as surface noise / energy injection / "poor remeshing" (the band no longer tracks the interface). The paper's Alg. 1 does sizing→subdivide→advect→tall-cells→solve **every step**; the entire staleness architecture is our deviation.

**A2 (CRITICAL). φ is advected once per rebuild across the whole accumulated interval, with the end-of-interval velocity.**
`kickQuadtreeRebuild` accumulates `quadtreeDtSinceKick` over all steps since the last kick and passes it as one dt (`lib/webgpu-uniform-eulerian.ts:348`) to a single semi-Lagrangian trace (`advectLevelSet`, `lib/webgpu-quadtree-builder.ts:78`). One first-order step over 8–32 substeps of physics, using only the *latest* velocity field, is a large transport error: curved trajectories, accelerating fronts, and anything rotational are integrated wrongly. The paper advects φ every step on the saved previous grid. This error feeds the sizing function, the tall-cell cut, and (via A1) the pressure BC — a second independent driver of bad remeshing.

**A3 (MAJOR, architectural). Advection and transport run on the dense fine uniform grid; the paper's advection savings never materialize.**
The paper's Table 1 shows advection dropping ~4.5s → 0.77s (and extrapolation, FLIP ops similarly) because *all* fields live on the adaptive grid. Ours keeps full N³ `rgba32float` velocity ping-pong textures + VOF textures + fp16 transport textures and runs occupancy/transport/flux-scale prep, MacCormack predict/reverse/correct on **every fine cell every step** (`advanceTo`, `lib/webgpu-uniform-eulerian.ts:400–432`). The quadtree only accelerates projection. That is a defensible architecture for a GPU (dense = coalesced), but it caps speedup and memory at uniform-grid scaling; the paper's headline win (5–10× projection AND 3–6× advection) becomes projection-only. Flag as a strategic decision, not a bug.

**A4 (MAJOR). Rigid two-way coupling feedback happens once per rebuild, not per step.**
`coupleImpulse` computes Kᵀp on GPU every solve, but the scalars are only copied/read back inside `rebuildFromState` (`lib/webgpu-quadtree-tall-cell.ts:1032–1045`), so `onRigidLoads` receives one impulse per rebuild with `couplingInterval_s = advectDt` (`lib/webgpu-uniform-eulerian.ts:363`). Additionally the solid fractions [A], owners, and the K rows are frozen between rebuilds while bodies keep moving (they're CPU-built per rebuild: `solidFieldsFromBodies`, `lib/webgpu-quadtree-tall-cell.ts:37`). The paper advances the rigid body and solves the monolithic system every step. Expect laggy, springy body response and asymmetric coupling at our cadence.

**A5 (MINOR, deliberate omissions).**
- No EXNBFLIP particle enrichment (paper Sec. 5 uses it for visual detail; they note pure level set on coarse cells is the fallback — which is what we do).
- No dual-contouring surface extraction (renderer has its own raster pipeline).
- Optical layer thickness is a single global constant `ceil(ny · fillFraction · opticalDepthFraction)` (`lib/webgpu-quadtree-tall-cell.ts:586`) rather than ¼ of the **local** column depth (Irving/paper). Deep-and-shallow scenes get the wrong band on one side.

### B. Math differences

**B1 (the VOF issue, quadtree edition).** Conservative VOF is still the authority in the quadtree path:
- `effectiveWet` overrules advected φ with VOF wherever they decisively disagree (>0.5·h_min) (`lib/webgpu-quadtree-builder.ts:61`).
- `evaluateSizing` detects wet/crossing cells from `loadVolume` (VOF), not φ (:129–133).
- `seedDistance` seeds the redistance from VOF interface cells (α ∈ (0.001, 0.999)) (:91).
- CPU mirror: `reconcileLevelSetWithVolume` / `signedDistanceFromVolume` (`lib/quadtree-tall-cell-grid.ts:617`, :515), with `mismatchFraction` tracked as a drift diagnostic.
The paper transports only the level set. This is the same class of issue the user is currently fixing for the restricted tall-cell method; the quadtree path needs the equivalent migration once per-step φ transport exists (see A2/R1 — today the reconciliation is *load-bearing* because φ is advected so rarely that VOF is genuinely more trustworthy. Fix the cadence first, then demote VOF.)

**B2 (BUG). The GPU φ advection ignores the staggered (MAC) layout.**
`advectLevelSet` uses `textureLoad(velocityIn, gid).xyz` directly as the cell-centre velocity (`lib/webgpu-quadtree-builder.ts:80`). The CPU reference correctly averages the two adjacent face samples per axis (`advectLevelSetSamples`, `lib/quadtree-tall-cell-grid.ts:599–601`), and the projection shader itself treats `velocityIn[axis]` as the face-(axis) sample of the cell (`faceVelocity`). So the GPU trace uses u_{i−½} as if it were u_i → half-cell velocity bias → systematic interface drift, worst in shear. One-line fix: average `velocityIn[q][axis]` with `velocityIn[q − e_axis][axis]` before backtracing.

**B3 (minor).** Sizing curvature term is ∇²φ (fine near a redistanced interface, matches paper intent), but the weights (curvature 4, velocity 3) are duplicated in CPU (`quadtreeSizingFromVelocityAndSurface`, `lib/quadtree-tall-cell-grid.ts:1044`) and GPU (staticData at `lib/webgpu-quadtree-builder.ts:298`) and can silently diverge.

### C. CPU-resident work (question 2)

The whole rebuild critical path after the GPU construction kernels runs **on the JS main thread** (async in the promise sense only — it still blocks the event loop and, via the lag limit, eventually blocks physics):

| # | Stage | Where | Cost class |
|---|-------|-------|-----------|
| C1 | Full 3D φ readback (nx·ny·nz·4B) then full re-upload | `lib/webgpu-quadtree-builder.ts:356`, `uploadLevelSet` | Transfer; the CPU only *needs* per-column crossings + leaf words |
| C2 | `populateTallPressureGrid` per-column segmentation | `lib/quadtree-tall-cell-grid.ts:358` | O(leaves·ny), with bilinear φ sampling per (leaf, y) |
| C3 | `buildVariationalSystem` face enumeration; vertical faces iterate every fine sub-cell in every leaf column | :675 (vertical loop :804) | O(N³) worst case per rebuild |
| C4 | `packSystem`: CSR assembly, IC(0) factorization, level scheduling, aux-texture packing | `lib/webgpu-quadtree-tall-cell.ts:728–942` | O(nnz), sequential factorization |
| C5 | `buildMlsProjectionRows`: MLS with 4×4 normal-equation solves per cell, 150k-row cap | `lib/quadtree-tall-cell-grid.ts:935` | Large constant; cap silently truncates |
| C6 | `quadtreeFromPackedCells`: Map churn + three full-grid validation sweeps | :301 | O(nx·nz) ×3, allocation-heavy |
| C7 | `solidFieldsFromBodies`: 8-corner CPU voxelization per body per rebuild | `lib/webgpu-quadtree-tall-cell.ts:37` | Small unless many/large bodies |

Recommended split: C2–C6 into a Web Worker first (pure data-in/data-out, no GPU handles needed); C1 shrink the readback; longer term C2/C3 are column-parallel and GPU-amenable (the paper's "vertical scan" structure maps to one thread per column, same trick `evaluateSizing` already uses).

### D. GPU solve performance

**D1 (MAJOR). The IC(0) preconditioner application runs in ONE 256-thread workgroup.**
`precondition` is dispatched with a single workgroup (`direct("precondition", 1)` / indirect with 1, `lib/webgpu-quadtree-tall-cell.ts:956,961`) because the level-scheduled triangular solves need global ordering (`applyPrecondition` loops `params.counts.w` levels with storage barriers, :347–377). For the >50k-DOF systems that trigger the O(√n) iteration floor (:612 raises the budget up to 2048), the whole GPU idles on ≤256 lanes twice per iteration. This is very likely the dominant projection cost at scale and undermines the point of GPU ICCG. Alternatives, in increasing effort: (a) Jacobi/Chebyshev polynomial preconditioner — no triangular solve, fully parallel, usually 1.5–2.5× more iterations but each far cheaper; (b) multi-workgroup level-sync via device-scope atomics-based spin (fragile on WebGPU); (c) geometric multigrid on the quadtree — the paper itself names this as the natural future work.

**D2. Fixed-length command encoding.** `encode` unrolls `this.iterations` × ~9 dispatches (up to ~18k dispatches at the 2048 floor) every step; converged iterations become zero-workgroup indirect dispatches but still pay encoding/driver cost. Batch iterations (e.g. encode 64 at a time with a tiny readback-free loop across frames) or clamp the floor with telemetry (`pressureIterationsUsed` is already tracked).

**D3. Buffer churn.** `WebGPUQuadtreeBuilder.build` creates and destroys ~10 buffers per rebuild, including two N³×4B seed buffers and the readback (:267–275, :377). Cache them on the builder (it is already cached via `gpuCache.construction`).

**D4. Rebuild latency gates physics.** With `quadtreeTopologyLagLimit = 3` (`lib/webgpu-uniform-eulerian.ts:89`, enforcement :382), a slow CPU pack blocks `advanceTo` (`quadtreeRebuildBlockedFrames` counts this). Everything in (C) directly becomes frame stutter.

**D5 (small).** GPU `refine` re-scans its leaf's full footprint per thread — O(size²) redundant work per cell for large leaves (`lib/webgpu-quadtree-builder.ts:161`); a per-leaf max image (mip-style scan) would remove it. Low priority.

## 5. Remediation plan (priority order, with reasoning)

**R1. Per-step surface refresh — fixes A1 + A2, biggest quality win for least work.**
- Move φ advection into the *step* encoder: run the (fixed, see R2) `advectLevelSet` kernel every substep with that substep's dt against the current velocity texture, writing to a resident φ texture ping-pong owned by the projection (replace `levelSetTexture` upload-from-CPU with GPU-to-GPU).
- Run the existing jump-flood redistance every K steps (K=4–8 is plenty; per-step is unnecessary since φ stays near-SDF over a few steps).
- Dispatch `refreshFaces` **every step** (drop `facesDirty` gating) so the Eq. 25 fluid scale and liquid-term classification track the moving interface between topology rebuilds. `refreshFaces` already reads φ at sample locations from the texture — no CPU involvement.
- The topology rebuild then consumes the resident GPU φ (no advection inside `WebGPUQuadtreeBuilder.build`; delete the accumulated-dt path and `quadtreeDtSinceKick`).
- Reasoning: with a fresh BC every step, topology staleness (which cells are cubic/tall, leaf sizes) is comparatively benign — the paper's own damping discussion says coarse cells already smooth detail; a slightly-late *subdivision* is tolerable, a wrong *boundary condition* is not.

**R2. Fix the staggered-velocity bug in `advectLevelSet` (B2).** Average adjacent MAC faces per axis before backtracing, mirroring `lib/quadtree-tall-cell-grid.ts:599–601`. Do this first/with R1 since R1 makes the kernel run 8–32× more often.

**R3. Per-step rigid impulse readback (A4).** Copy the 6-per-body scalar slice to a small staging buffer each step and mapAsync with the same non-blocking pending-flag pattern the uniform solver already uses for `rigidExchangeBuffer` (`lib/webgpu-uniform-eulerian.ts:458,466`). Rebuild [A]/K when any body has moved > 0.5·h since the last rebuild (add to `shouldKickQuadtreeRebuild`).

**R4. Finish the level-set migration in the quadtree path (B1).** After R1, φ is fresh every step; make it authoritative: `effectiveWet` → `phi < 0` only, `evaluateSizing`/`seedDistance` keyed off φ, delete the VOF reconciliation and `mismatchFraction` (or keep as a pure diagnostic). Coordinate with the user's in-flight restricted-tall-cell migration for shared conventions (narrow-band clamp 5 cells, `occupancyFromPhi` shims in `lib/tall-cell-kernels.ts:89`).

**R5. Cut the rebuild critical path (C, D4).**
- Shrink the readback: leaf words (nx·nz·4B) + per-column interface data (crossing ys + φ at crossings) instead of the full 3D φ (C1). `populateTallPressureGrid` only consumes column φ profiles at leaf centres — compute those on GPU (one thread per leaf column) and read back just that.
- Move C2–C6 (tall grid, variational assembly, CSR/IC pack, MLS rows, decode) into a Web Worker; they are pure functions of (leaf words, column profiles, bodies). Main thread only uploads the packed buffers.
- Cache the builder's transient buffers across rebuilds (D3).

**R6. GPU-friendlier preconditioner (D1, D2).** Prototype a Chebyshev/Jacobi-polynomial preconditioner behind a flag; compare `pressureIterationsUsed` and wall time vs IC(0) on the dam-break and deep-tank scenes. If iterations blow up on tall-cell anisotropy, fall back to investigating quadtree geometric multigrid (paper's suggestion). Independently, batch the iteration encoding.

**R7 (strategic, optional). Adaptive advection (A3).** Only if paper-scale memory/advection wins are a goal: advect velocity/φ at adaptive sample locations with saved-previous-grid interpolation (Klingner), retiring the dense N³ textures. Large rewrite of the orchestrator; decide after R1–R6 land, since R1–R6 may already hit the app's perf targets.

## 6. Verification

- **Smoke scenarios:** `tools/run-webgpu-smoke.ts` with `tools/webgpu-smoke-scenarios.ts` (dam break, boxes). Watch `quadtreeLevelSetMismatchFraction` (should → ~0 after R4), `quadtreeRebuildBlockedFrames` (→ 0 after R5), `pressureIterationsUsed`, `quadtreeCPUTopologyPack_ms`, `gpuConstruction_ms`, `gpuTimings.pressure_ms`.
- **Energy/settling probes:** `tmp/tall-cell-audit/probe-*.ts` (esp. `probe-collapse.ts`, `probe-boxes.ts`) — the A1/A2 fix should flatten kinetic energy in a settled tank (the "stirring" symptom) and remove surface popping at rebuild boundaries.
- **Unit baseline:** keep `tmp/levelset-migration/baseline/unit.txt` green — notably "packed GPU leaf maps reconstruct the CPU quadtree exactly" and the dam-break column tests.
- **Visual A/B:** run the same scene with `rebuildTopology` on/off and quadtree vs uniform method; after R1 the quadtree method's surface should stop lagging/jumping relative to uniform on identical steps.
- **Rigid coupling:** after R3, a floating box in a calm tank should bob at a period independent of the rebuild cadence (today it visibly locks to the 8–32-step impulse cadence).
