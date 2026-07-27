# Structured cutover · dam-break physics parity — session handoff

Copy this file's contents as the opening prompt of a fresh session.

## Mission

On `perf/structured-cutover`, make the octree structured lane's
`minimal-power-dam-break` physically match `main` (reference worktree:
`/private/tmp/fluid-main.4laMrM`, commit 424b8e3, node_modules symlinked,
harness works there). Success is defined by measured Dawn criteria (below),
NOT by eyeballing the browser. Iterate in Dawn; verify in the browser last.

Memory notes `structured-dissipation-root-cause` and
`ui-epoch-freeze-root-cause` hold the full forensic detail; older
perf memories describe deleted code — grep before trusting.

## What is already fixed and validated (all uncommitted in the working tree)

1. **Wall-row staggered eligibility** (`webgpu-octree-air-velocity-support-gpu.ts`
   `emitAirSupportCandidates` + gates in `webgpu-octree-structured-dynamics.ts`):
   regular-closure eligibility is geometric (boundary-clamped 27-uniformity),
   not `caseId==0`; wall rows publish BOTH cube tags and their tet fan
   (`needsSelectors=!regular||g.z!=0u`). Census: 100% of wet faces staggered.
2. **Band-gated carried advection** (`advect` in structured-dynamics):
   main's DELTA_CARRIED semantics — deep-interior liquid faces carry their
   projected value exactly; the interface band re-traces. CRITICAL TRAP: the
   accepted row set is LIQUID-ONLY, so an interface face has
   `neighbor(handle)==INVALID` — the gate must trace those. This fix removed
   a dt-independent ~3.5%/step KE loss (retention 0.61→~1.0 by t=0.40).
3. **GFM crossing tolerance + θ floor** (`resolveStructuredBoundarySlots`,
   `webgpu-octree-structured-boundary.ts`): a crossing face with a dry owner
   centre no longer hard-fails the whole topology epoch (that froze the
   browser's velocity authority at gen 14 = t=0.056 forever). θ from the wet
   side with floor 1e-2; no wet side → scale 0. Browser went t=0.06 → t≈2.49.
4. **Euclidean CPT metric in the Section 5 march** (`extendFace`):
   carrier distance = straight-line to the ORIGINAL seed (candidate.z),
   not accumulated axis-hop path length (L1 → diamond-shaped dam front).
   **LANDED BUT NOT YET VALIDATED — first thing to measure.**

The 500-step Dawn gate (`npm run test:webgpu:minimal-power-dam-break`)
passes with 1-3: 500/500 steps, 0 validation errors, volumeDrift 0.
Unit suites updated; `tests/webgpu-octree-air-velocity-support-gpu.test.ts`
test "producer proves generations..." fails at HEAD too (pre-existing).

## Success criteria vs main (captured; all green at 150 steps)

A `front-footprint` probe was added to BOTH trees' `run-webgpu-smoke.ts`
(per checkpoint: floor-layer radial extents rx/rz/rdiag from the reservoir
corner + `circularity` = rdiag·√2/max(rx,rz); ~1 circular, ~0.71 diamond).
Reference lane command (run in the main worktree, then identically in the
branch; GPU is exclusive — one Dawn process at a time, and no browser WebGPU
tab while Dawn runs):

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
FLUID_WEBGPU_ADAPTER='Apple M1 Max' FLUID_WEBGPU_DAWN_FEATURES=skip_validation \
FLUID_SCENE=minimal-power-dam-break FLUID_METHOD=octree FLUID_QUALITY=balanced \
FLUID_TARGET_S=0.6 FLUID_MAX_DT=0.004 FLUID_ORACLE_STEPS=150 FLUID_EXPECT_EXACT_STEPS=150 \
FLUID_REQUIRE_SPATIAL_FIELD=1 FLUID_CHECKPOINT_EVERY_S=0.02 FLUID_VOXEL_CELL_SIZE=0.05 \
FLUID_EXPECT_GRID=16,16,16 FLUID_STABILITY_ENVELOPE=1 FLUID_CPU_ORACLE=0 FLUID_FIELD_STATS=1 \
FLUID_RASTER_CHECKPOINTS=1 FLUID_GLOBAL_FINE_GENERATION_TRANSITION=1 FLUID_MAXIMUM_LEAF_SIZE=2 \
FLUID_OCTREE_INTERFACE_BAND=3 FLUID_OCTREE_GLOBAL_FINE_FACTOR=4 \
FLUID_WEBGPU_SMOKE_TIMEOUT_MS=240000 node --import tsx tools/run-webgpu-smoke-isolated.ts
```

Extract per run: `compactMechanicalEnergyCheckpoints` (KE + retention),
`globalFineGenerationCheckpoints[].raster.frontInterfaceBounds_m` (front
arrival = maxX hits 0.4; far-wall jet = maxY after t≈0.30), and the
`front-footprint` lines (branch needs `FLUID_ENERGY_EVERY_STEPS=5` to
activate the block that reads the cubic field; main samples on checkpoints
without it).

**MEASURED REFERENCE + CURRENT BRANCH (2026-07-27, 150 steps to t=0.6,
logs `main-ref-150.log` / `branch-ref2-150.log` in the session scratchpad;
all criteria currently GREEN):**

| criterion | main | branch (all fixes) |
|---|---|---|
| retention t=0.22 / 0.30 | 0.981 / 0.952 | 0.992 / 0.990 |
| KE t=0.38 / peak | 0.405 / 0.472 | 0.451 / 0.479 |
| jet maxY t=0.38 / lid (0.80 m) reached | 0.647 / t≈0.44 | 0.712 / t≈0.44 |
| floor circularity t=0.10→0.22 | 1.16→1.04 | 1.16→1.04 |
| retention proxy late overshoot | 1.42 @ t=0.58 | 1.37 @ t=0.58 |

The late retention overshoot is a SHARED proxy artifact (main's is larger)
— do NOT chase it as a branch defect. Branch runs uniformly a few percent
hotter/earlier than main; acceptable bracketing, keep an eye on it at other
scenes. The Euclidean-CPT march fix (fix 4) is validated by the circularity
row; the user's visual "smooth circular front like main" check in the
browser is the remaining confirmation. Re-confirmed reproducible 2026-07-27
(`branch-fix4-150b.log`, this session's scratchpad): identical footprint
trajectory, retention 0.992/0.990, KE@0.38 0.451, jet maxY@0.40 0.774.

Known remaining raster deviation (the only smoke invariant that fails at
150 steps): a single 203-pixel (20×19) enclosed missing patch in the
FORWARD BACK surface at t=0.36, during jet rise. Main's run only shows
small (≤34 px) patches at t=0.54. Surface-raster artifact, not an energy
defect — triage after the late-time bug.

## Prioritized next steps

1. Browser visual confirmation of the circular front + full-run behavior
   (user drives; reload picks up all fixes).
2. ROOT-CAUSED AND FIXED 2026-07-27 — the browser "0 LIVE PRESSURE ROWS /
   WATER UPDATE REJECTED" failure was a DIAGNOSTICS SAMPLING RACE, not a
   solver failure. `readSolveDiagnostics` copies the LIVE `solveStats`
   buffer concurrently with in-flight steps; a mid-step sample legally
   reads the cleared row counter as 0. The UI's `powerHealthy` gate keyed
   on that racing counter, flipped the §4.1–4.2/§6 stage to REJECTED, and
   the sticky WATER UPDATE REJECTED banner + retained-mesh presentation
   made the run look damped/terminal. The queue-fenced structured receipt
   (velocity/boundary controls) stayed valid gen-current through every
   episode (frontier dumps in this session's console captures). Dawn never
   sees it because its readStats always follows a completed await; Dawn
   free-run is bit-identical to lockstep (phiBitXor matched), ruling out
   pipeline-depth physics divergence entirely.
   FIXES: `paper-pipeline-diagnostics.ts` powerHealthy now trusts the
   fenced receipt (racing counter is display-only);
   `webgpu-uniform-eulerian.ts` readStats publishes stabilityFlags —
   `structured-authority-lag N gen` (lag > 4) and
   `structured-publication-invalid` (fenced receipt bad) — so the GPU
   stability card ALERTs on real desync (user request), plus episodic
   console dumps ([structured-lag], [structured-epoch-freeze] re-armed
   per epoch, [structured-publication-failure] with full frontier decode,
   [structured-probe] heartbeat w/ counterRaces tally). Verified in
   browser: STABLE past t=1.94 where two runs previously "failed".
   STILL OPEN (real, now visible via the lag flag): transient
   structured-authority lag episodes — one observed run had the authority
   11 generations stale (gen 70 vs 81) with velocity stats unpublished for
   80+ steps → genuinely stale transport velocities → the visible damping
   vs main the user reported. Proper fix for the counter race is a
   step-coherent GPU snapshot (copy solveStats at end-of-step encode,
   read the snapshot), mirroring the smoke harness's audit-snapshot
   pattern. Vite trap: a hot edit is NOT picked up by an already-loaded
   page — hard-reload the tab and verify the module `?t=` stamp before
   trusting silent instrumentation.
   650-step findings that matter:
   - MAIN FAILS ITS OWN INVARIANTS LATE: authoritative power CFL 1.134,
     liquid disconnected (7 components, 0.984 dominant), Section 5 max
     displacement 5 fine cells. Main's late series churns (maxSpeed climbs
     monotonically to 14.1 m/s, KE ~0.5 at t=2.6) while the branch settles
     (KE 0.075, maxSpeed 4.7) — settling is physical; late-time (t>1.0)
     parity vs main is NOT a valid target.
   - Branch-specific invariant: `transportDepartureOutsideBand` nonzero
     intermittently from t≈1.5 (66..678 per checkpoint, 252 at final gen;
     main is 0 throughout). Late-settling band-classification issue in the
     branch's band-gated advection; fails the smoke gate's final-generation
     check. Needs a fix or a tolerance decision before merge.
3. Re-run the full acceptance set (`npm run test:webgpu:minimal-power-dam-break`
   passed with fixes 1-3; re-run with fix 4 included) and the other scenes
   (dam-break-ui, garden) for cross-scene sanity.
4. Commit the work in reviewable pieces (dissipation fixes / GFM boundary
   fix / CPT metric / instrumentation), including the memory docs.
5. Cleanup before merge: decide fate of temporary diagnostics — the
   `structuredFreezeDumped` dump in `webgpu-uniform-eulerian.ts readStats`,
   boundary fail sub-bits (128/256/1024 — keep, they're cheap and named),
   the per-step `structured-stage-energy` probe + sampler census (always-on
   cost; consider gating), dead X2 branch in `staggeredPlaneValue` (air rows
   don't exist as rows; measured no-op).

## Traps that cost hours — do not relearn

- The accepted row set is LIQUID-ONLY (`coarseNegativeEntries==rowCount`).
  Air = Section 5 support arena. `neighbor==INVALID` = free surface.
- WebGPU per-stage limit: 10 storage buffers. `skip_validation` masks the
  pipeline error and SIGSEGVs at bind-group time instead. Debug by running
  the 2-step lane WITHOUT skip_validation. Advect fits only because it no
  longer reads solidNormalVelocities (forceFamily re-imposes solid).
- WGSL helper calls change auto-layouts: update every `entries` list.
- Do NOT eyeball raw `mgpcgControl` words as residuals; use
  `octreeMGPCGDiagnostics` (pressure converges ~1e-5; it is NOT the issue —
  and per the task rules, don't chase pressure without direct evidence).
- Segfaulted Dawn runs leave `/tmp/fluid-webgpu-exclusive.lock`; verify the
  recorded pid is dead, then `rm -rf` it. NEVER `git stash` in this repo.
- Browser debugging rig: `http://localhost:3000/?gpu=on&method=octree&scene=
  minimal-power-dam-break&quality=balanced` (+`panel=diagnostics`), a second
  Fluid tab holds Web Lock `fluid-lab:webgpu-exclusive` (ask the user to
  close theirs). "structured OK gen N" stuck while fine gen advances = the
  epoch-freeze tell. The readStats one-shot dump prints
  `[structured-epoch-freeze]` with `readPowerFrontierFailure()` forensics;
  decode `epoch` via `struct Epoch` + `OCTREE_TOPOLOGY_EPOCH_ERROR` in
  `webgpu-octree-topology-epoch.ts` (error 512 = coefficientPositivity =
  boundary publisher; boundary flags OR the poison echo on top of its own
  codes 2/4/8/16/32/64).
- Dawn runs lockstep (stability envelope awaits per step); the browser runs
  an ~80 s deep pipeline with fractional rAF dts — trajectory divergence is
  EXPECTED; validators must tolerate legitimate geometry, not depend on
  cadence luck.
- The main worktree at `/private/tmp/fluid-main.4laMrM` has the footprint
  probe patched in (uncommitted). Keep it for comparisons.

## Measurement rig

- Per-step stage energies: add `FLUID_ENERGY_EVERY_STEPS=1` to the lane →
  `structured-stage-energy` JSON lines (start/postAdvection/preProjection/
  postProjection + wet-only + staggered-path census). Seam(N) =
  start(N) − postProjection(N−1) attributes remap/publication.
- 32-word `projectionEnergyStats` layout (8/stage); decode
  `decodeStructuredProjectionEnergy` in structured-dynamics.ts; the smoke
  audit snapshot stride is 336 bytes (webgpu-smoke-structured-audit.ts).
