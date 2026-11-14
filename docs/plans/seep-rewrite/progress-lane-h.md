# Lane H progress (harness and integration)

Entries newest last: STARTED, MILESTONE, READY TO MERGE, DECISION NEEDED, COORDINATION.

## 2026-11-14 STARTED (lane H)

Plan: (1) factor the NB-FLIP zone measurement into `tools/seep-observe-nbflip.ts` (same
owner-tetra fill and zone rules as probe-tier-budget, plus 16 right-chamber x bins, x
centroid, jet column); (2) `tools/seep-observe-seep.ts` over `SeepObservationSource`
with a CPU oracle unit test; (3) `tools/seep-bench.ts`; (4) NB-FLIP reference runs;
(5) `tools/seep-compare.ts`; (6) `seep-m1max` method registration and rendering.

## 2026-11-14 COORDINATION for lane S: export, observation, rendering contract

What lane H's method wrapper (`seep-m1max`, in `lib/methods/seep-method.ts`, lane H's file) and
`tools/seep-bench.ts` need from `lib/methods/seep/index.ts`:

1. **Export** `createSeepSolver(device: GPUDevice, scene: SceneDescription, options: { onProgress?:
   GPUInitializationReporter; signal?: AbortSignal }): Promise<SeepSolver>`, where `SeepSolver`
   implements `GPUSolverInstance` (lib/core/method-contract.ts) **plus**
   `readonly observation: SeepObservationSource`. The wrapper already refuses any
   `scene.sceneId !== "reservoir-seep"` before calling you; your bake should still fail explicitly
   on geometry it does not cover. Until index.ts exists the wrapper is not registered (the bench
   says so explicitly for `--method=seep`).
2. **Observation buffers** must be the ones the solver keeps for its whole life (the harness binds
   them once). `seepVertexPhi` in cells, negative in liquid. The harness counts only
   `seepCellOpen` cells; it also reports the wall zone over all cells (`wallAll`) because NB-FLIP's
   owner fill does not subtract solids: NB-FLIP reads **0.140 L in the wall zone at t = 0**
   (throat open cells hold only 64 cells = 0.008 L), so expect your t = 0 wall to be ~0.008 L and
   total ~368.65 L against NB-FLIP's 368.78 L. Gates use left/right/deltas, so this constant
   offset does not enter them (G7 drift is against each solver's own t = 0).
3. **Rendering (plain dense path of lib/core/webgpu-water-pipeline.ts):**
   - `surfaceFieldTexture`: **r32float 3D, 513 x 257 x 65 (nodal, dims + 1), phi in METRES,
     negative in liquid**, usage `TEXTURE_BINDING` (+ `STORAGE_BINDING` if your kernel writes it).
     Read with textureLoad only. The renderer recognises nodal phi by `textureDimensions == dims+1`
     and contours the mean of each cell's 8 corners; normals come from the same texture.
   - **phi must be positive (air) inside solid voxels (the dividing wall)**: the plain path does no
     solid clipping, so negative phi in the wall draws the wall as water.
   - `volumeTexture`: required by the type; the same texture is fine.
   - Do **not** set `denseLevelSetVolumeSource` (or, if you do, no `mixedOwnership`/
     `coarseVertexPhi`, and return the same object every time). `fluidDomain` optional (the
     lattice equals the container: 2.56 x 1.28 x 0.32 m). `velocityTexture` (rgba32float, cells)
     only for the speed/CFL overlays: optional.
   - A buffer-to-texture copy needs bytesPerRow % 256 == 0 (513*4 = 2052 is not), so write the
     render texture from a kernel (e.g. fold it into the redistance's last pass), or keep phi in
     that texture.
4. **Lifecycle the app relies on:** `info` is one stable mutable object (the renderer writes into
   it: completedTime_s, initialRasterSurface*, simulationPipelinesReady), with `gridKind:
   "uniform"`, `nx/ny/nz`, `storedNy`, `cellCount`, `cellSize_m`, `quality`, `allocatedBytes`, and
   the other required GPUEulerianInfo fields. `advanceTo(t)` must synchronously encode exactly
   one 1/60 s step when `t >= submittedTime_s + dt`, set `info.submittedTime_s`, and bump
   `info.encodedSteps` (the mesh re-extracts only when it changes); return false otherwise.
   `awaitFrameCompletion()` fences; `framePending` true only while you refuse a new step (the
   pipelined bench retries on it). Latched GPU failures: set `info.simulationPipelineError`
   (checked every step by the bench) and/or reject in `assertSimulationHealthy()` (called at
   sample steps). `readStats()` may return `{...info}`. No rigid bodies/edits: leave those optional
   members absent; the host gates decline them.

## 2026-11-14 MILESTONE H1a: observers, bench, compare; NB-FLIP repeat spread

Files: `tools/seep-observables.ts` (schema, shared WGSL zone accumulator, CPU oracle),
`tools/seep-observe-nbflip.ts` (zone probe factored from probe-tier-budget: owner tetra fill,
canonical load, BULK16 regions), `tools/seep-observe-seep.ts` (dense observer over
`SeepObservationSource`), `tools/seep-bench.ts`, `tools/seep-compare.ts`,
`tests/seep-observer.test.ts` (oracle vs closed forms; WGSL observer vs oracle on a synthetic
source, GPU-only: passed under fluid-gpu-run check).

t = 0 validation (NB-FLIP observer): total 368.780 L, left 368.640, **wall 0.1404 L**, right 0 —
identical to the baseline probe. Steps 60/120 reproduce the baseline (left 368.402/368.135,
wall 0.149/0.248, throat u 2.65/2.77).

NB-FLIP reference, two 1200-step check runs (`--nb-stats`), mean ± S (S = |a - b|):

| step | left L | right L | wall L | floor band L | q_left L/s | throat u m/s | wet cells | centroid x | jet L | particles | drift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 367.309 ± 0.000 | 1.104 ± 0.003 | 0.250 ± 0.000 | 0.945 ± 0.008 | 0.271 ± 0.000 | 2.640 ± 0.000 | 14820 ± 153 | 427.7 ± 0.3 | 0.0781 ± 0.0002 | 67632 ± 996 | -0.032% ± 0.001% |
| 600 | 365.951 ± 0.000 | 2.090 ± 0.012 | 0.253 ± 0.000 | 1.924 ± 0.011 | 0.271 ± 0.000 | 2.629 ± 0.000 | 16116 ± 195 | 368.9 ± 2.3 | 0.0798 ± 0.0007 | 129381 ± 3402 | -0.132% ± 0.003% |
| 900 | 364.604 ± 0.000 | 2.890 ± 0.021 | 0.256 ± 0.000 | 2.733 ± 0.015 | 0.269 ± 0.001 | 2.636 ± 0.017 | 16198 ± 96 | 363.0 ± 1.3 | 0.0746 ± 0.0058 | 186235 ± 451 | -0.279% ± 0.006% |
| 1200 | 363.267 ± 0.001 | 3.420 ± 0.011 | 0.258 ± 0.000 | 3.262 ± 0.004 | 0.267 ± 0.000 | 2.598 ± 0.080 | 16361 ± 33 | 369.8 ± 3.7 | 0.0769 ± 0.0005 | 220021 ± 1727 | -0.498% ± 0.003% |

Repeat profile L1 (16 bins, normalised): 600: 0.067, 1200: 0.075 (so G6's floor 0.15 binds).
G5 90% step: a 298, b 300. NB-FLIP's own volume drift (readStats) at 1200: -0.48% both runs
(the baseline's -0.55% was a third realisation). Step time in these check-slot runs (not a fair
timing): median 36.3 / 34.1 ms.

Harness self-check (`seep-compare --candidate=nbflip-check-b` against run a): G1-G6, G8 pass; G7
fails (-0.503% vs -0.496%): as written, G7 asks the candidate to conserve volume at least as
well as NB-FLIP, so an NB-FLIP repeat fails it half the time. Not loosened; noted only.

Notes:
- `npm run check:types` has pre-existing errors in tracked `tools/experiments/lane-n/*.mts`
  (they import `~/code/me/fluid-scratch/lane-n/tree/...`, which no longer exists); everything
  else is green. Not lane H's files; left alone.
- The main index shows the plan/progress files as staged deletions plus untracked (the
  coordinator's commit used a private index). Harmless; lane H commits with a private index.

## 2026-11-14 MILESTONE H1b: NB-FLIP timing reference (timing slot, exclusive GPU)

`tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/seep-bench.ts --method=nbflip --steps=1200 --every=60 --mode=<fenced|pipelined> --slot=timing --out=artifacts/seep-rewrite/nbflip-ref --name=nbflip-timing-<mode>` (rc 0 both)

| NB-FLIP, reservoir-seep, 1200 steps | value |
|---|---|
| fenced median / p95 / mean / max ms (21 sample steps and their successors excluded) | **33.69** / 37.65 / 33.34 / 42.17 |
| fenced median by thirds (steps 13-408 / -804 / -1200) | 29.7 / 33.7 / 36.5 ms (cost grows with the right chamber's particles) |
| pipelined, submit all then one fence | **32.22 ms/step** (38664 ms total) |
| construction (pipeline cache warm) | 10880 ms fenced run, 10809 ms pipelined run |
| peak GPU bytes requested (profile proxy) | 1,244,171,964 (runner saw 2.80-2.97 GB process peak) |

So §5's 3x pass needs Seep fenced median <= 11.2 ms and pipelined <= 10.7 ms/step; 5x stretch
<= 6.7 / 6.4 ms. The fenced timing run's observables (right 3.388 L, particles 213990 at 1200)
sit just outside the two check runs' S on right L (S 0.011, the third run is -0.032 from the mean)
but well inside every gate's 5% term: two runs understate the noise a little. seep-compare now
takes S from check-slot runs only, as the plan says, so a timing run never widens it.
Report: artifacts/seep-rewrite/nbflip-ref/nbflip-reference.md (performance verdict "incomplete"
until Seep timing runs exist).

URL method switch (for the coming `seep-m1max` method): the app had no method picker and every
path forced `defaultMethodId()`, so `?method=` was parsed but discarded. `lib/core/url-state.ts`
now honours `?method=<registered id>` (unknown or retired ids still fall back to the default and
are not serialised) and writes `method=` back only for a non-default method;
`lib/core/compare/compare-model.ts` keeps the pane's method. Opening a scene card still resets
to NB-FLIP. The seep under the new solver will be `?scene=reservoir-seep&method=seep-m1max`.

## 2026-11-14 COORDINATION for lane S: first seep-bench of the uncommitted solver (16:32)

`tools/gpu/fluid-gpu-run.sh check node --import tsx tools/seep-bench.ts --method=seep --steps=120 --every=60 --out=artifacts/seep-rewrite/dev --name=seep-dev-120`
(log artifacts/seep-rewrite/dev/seep-dev-120.log; working tree as of 16:30, not a judged run)

- The observer contract works against your source as is. t = 0: total 368.648 L, left 368.640
  (NB-FLIP 368.640, exact), open wall cells 0.008 L (as predicted), right 0. `wallAll` (every wall-zone
  cell, solids included) is 0.395 L against NB-FLIP's 0.140, so your vertex phi inside the wall is
  negative over more of it: harmless for the observer, but worth a look if the renderer or
  extrapolation reads phi in solids.
- Construction 2.6 s; 915 MB allocated (NB-FLIP 1.24 GB).
- **Failed at step 60**: `Seep solver latched DEEP_SURFACE, POOL, SEGMENTS (max backtrace segments
  4294967040, pool overflow 171020823)`. 4294967040 is 0xFFFFFF00, which looks like an uninitialised or
  negative counter rather than a real count. Step times while it ran: median 64 ms, thirds
  276 / 116 / 61 ms (the early steps are slow).
- Method wrapper `lib/methods/seep-method.ts` (`seep-m1max`, "Seep · M1 Max (experimental)") is
  registered in `lib/methods/index.ts` in the working tree. NB-FLIP stays the default; the wrapper
  refuses any scene but reservoir-seep before touching the device, and maps your onProgress stages
  to the app's phases. Types and unit tests are green with your files. **I cannot commit the
  registration until your `lib/methods/seep/*.ts` are committed** (HEAD would import untracked
  files). Please post a COORDINATION line when they land, and I will commit it straight after.
- App switch: `?scene=reservoir-seep&method=seep-m1max` (landed in fb671839).

## 2026-11-14 MILESTONE H2: `seep-m1max` registered (after lane S's 75bf4433)

`lib/methods/seep-method.ts` wraps `createSeepSolver`, and `lib/methods/index.ts` installs it beside NB-FLIP, which stays
the default. `tests/seep-method.test.ts` covers registration and the default, refusal of any other scene (before the
device is touched), and the `?scene=reservoir-seep&method=seep-m1max` parse/serialise round trip. Host gates:
`takesScheduledDrops` and `takesLiveSolidEdits` both test `methodId === "uniform-narrow-band-flip"`, so both are false
under Seep. The solver also declares no optional edit or injection members, so the host declines those features.

## 2026-11-14 COORDINATION for lane S: 1200-step check run of 75bf4433 stops at step 180

`tools/gpu/fluid-gpu-run.sh check node --import tsx tools/seep-bench.ts --method=seep --steps=1200 --every=60 --out=artifacts/seep-rewrite/seep-75bf4433 --name=seep-check`

**Failed at step 180**: `Seep solver latched SEGMENTS (max backtrace segments 478, pool overflow 0)`. No verdict
is possible, because seep-compare judges only complete runs. Observables up to that point (lane H observer; NB-FLIP mean in brackets where sampled):

| step | left L | right L | floor band L | throat u m/s | q_left L/s | wet cells | jet L | particles | drift |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 368.396 [368.40] | 0.288 | 0.232 | 2.52 [2.65] | 0.244 | 7373 [3559] | 0.0554 | 132k | +0.012% |
| 120 | 368.145 [368.14] | 0.297 | 0.218 | 2.56 | 0.252 | 9287 | 0.0788 | 368k | -0.054% |

- Right L barely rises from 60 to 120 (0.288 to 0.297) while the left loses 0.25 L. Your smoke run reported 0.396 at
  120. Total drift is only -0.05%, so the liquid is not lost. It may be sitting in cells my right zone does not count
  (x >= 256 open cells with vertex-fill > 0), or in particles that the surface field does not show. Worth a check.
- The open-cell wall zone stays at 0.008 L. That is the throat full (2x2x16 cells, 64 x 0.125 mL), as expected.
- Step time: median 46 ms, thirds 34 / 46 / 63 ms (rising with particles, 368k at 120 against NB-FLIP's ~30k).
  Construction 1.8 s, 1.05 GB.

## 2026-11-14 MILESTONE H3: independent verdict on lane S's ed896292 (S1b), 1200 steps, check slot

`tools/gpu/fluid-gpu-run.sh check node --import tsx tools/seep-bench.ts --method=seep --steps=1200 --every=60 --out=artifacts/seep-rewrite/seep-ed896292 --name=seep-check` (rc 0)
`node --import tsx tools/seep-compare.ts artifacts/seep-rewrite/nbflip-ref artifacts/seep-rewrite/seep-ed896292 --out=artifacts/seep-rewrite/seep-ed896292 --name=compare`

This run is of the committed tree. Lane S's compare-s-m3 ran on a dirty bc78ac9c.

| gate | verdict | detail (Seep vs NB-FLIP mean) |
|---|---|---|
| G1 | pass | q_left mean 0.2596 vs 0.2697 L/s (-3.74%); 18/18 windows |
| G2 | pass | left loss -2.99 / -3.81 / -3.79 / -3.51% at 300/600/900/1200 |
| G3 | **FAIL** | right 1.756 / 3.160 / 4.416 / 5.237 L; upper bounds 1.282 / 2.705 / 4.117 / 5.518 (1200 passes) |
| G4 | pass | throat u 2.581 vs 2.639 m/s (-2.2%) |
| G5 | pass | 90% wet at step 277 vs 299; 16384 vs 16361 at 1200 |
| G6 | **FAIL** | 600: centroid 375.1 vs 368.9 ok, **L1 0.213 > 0.202**; 1200: centroid 378.6 vs 369.8, L1 0.173 <= 0.226 |
| G7 | pass | drift +0.01% at 1200 |
| G8 | **FAIL** | jet column 0.0541 vs 0.0770 L (-29.7%, limit 25%) |

Verdict: **FAIL (5/8)**. Lane S's own run had G6 pass at 600 (L1 0.177), so G6 at 600 sits on the noise edge.

Notes for lane S:
- **G7 passes by cancellation.** Total drift is +0.16% at 600 (369.222 L, the chamber 0.57 L above
  delivery) and +0.01% at 1200. The chamber first gains about 0.6 L, then loses it back. It does not
  "track delivery" after step 360: q_right at 1200 is 0.174 against q_left 0.257 L/s. So both the early
  gain and the later loss are real volume errors. G7 judges only step 1200, as written.
- Jet column: 0.054 L against NB-FLIP's 0.077, while q_left and throat u match to within 4%. So the
  falling sheet is thinner or faster (more of it already on the floor) than NB-FLIP's.
- Particles 205k at 600 and 407k at 1200 (NB-FLIP 129k / 220k).
- Step time (check slot, not a fair timing; another process held 2.0 GB of GPU memory at start): median 50.4 ms,
  thirds 88 / 50 / 49 ms. Construction 2.7 s, 1.05 GB requested. A timing-slot run waits until the gates pass or
  S2 lands.

## 2026-11-14 MILESTONE H4: verdict on bef5feb6 (S2a speed), 1200 steps, check slot: physics regressed

Same commands as H3 with `--out=artifacts/seep-rewrite/seep-bef5feb6` (rc 0; the committed tree was clean for lane S's files).

| gate | ed896292 | **bef5feb6** | detail (bef5feb6 vs NB-FLIP mean) |
|---|---|---|---|
| G1 | pass | pass | q_left 0.2570 vs 0.2697 L/s (-4.69%, limit 5%: close to the edge) |
| G2 | pass | pass | left loss -3.95 / -4.75 / -4.73 / -4.47% |
| G3 | FAIL | **FAIL** | right 2.005 / 3.647 / 5.350 / **6.629** L; upper bounds 1.282 / 2.705 / 4.117 / 5.518 |
| G4 | pass | pass | throat u 2.582 vs 2.639 |
| G5 | pass | **FAIL** | 90% wet at step 209 vs 299 (-30%, limit 25%) |
| G6 | FAIL | **FAIL** | centroid 379.9 / 385.4 vs 368.9 / 369.8; L1 0.308 (limit 0.202) / 0.232 (limit 0.226) |
| G7 | pass | pass | drift **+0.41%** at 1200 (+0.29% at 600), against NB-FLIP's 0.50% |
| G8 | FAIL | pass | jet 0.0756 vs 0.0770 L |

Verdict: **FAIL (4/8; ed896292 was 5/8)**.

**COORDINATION for lane S:** the chamber now *creates* liquid all the way through. At 1200 the right
chamber holds 6.63 L against 5.13 L delivered (+1.5 L, +29%), and total volume rises steadily: +0.29%
at 600, +0.41% at 1200. So q_right (0.218 L/s at 1200) runs well ahead of what the throat supplies. The
gain wets the floor 90 steps early and pushes the profile right. Your step-120 check could not see it,
because the gain accumulates over the run. G7 still passes only because NB-FLIP loses 0.50%; at this
rate it fails just past 1200. The left side is unaffected: left loss and q_left sit 4-5% under
NB-FLIP, as before. So it is the chamber or the coarse/fine interface, not the reservoir. Suggest a
per-stage volume ledger over 600 steps (you had one for S1b).

Speed (check slot, GPU otherwise idle: 0.02 GB in use at start; indicative only): fenced median **21.2 ms**,
p95 24.0, thirds 19.7 / 21.1 / 22.4. Construction 1.8 s, 1.05 GB. That is 1.59x NB-FLIP's 33.69 ms timing
median (3x needs 11.2 ms). No timing-slot run yet: a timing run of a gate-failing build would only
measure a solver that does not yet match.

## 2026-11-14 MILESTONE H5: independent verdict on e20a907a (S2g), section 5 timing, and the S2g audit

**Gates** (my own 1200-step check run, committed tree: `seep-bench --method=seep --steps=1200 --every=60 --out=artifacts/seep-rewrite/seep-e20a907a --name=seep-check`; report `artifacts/seep-rewrite/seep-e20a907a/compare.md`). This reproduces lane S's compare-s2g to the third digit.

| gate | verdict | detail |
|---|---|---|
| G1 | pass | q_left 0.2580 vs 0.2697 L/s (-4.33%, limit 5%); 18/18 windows |
| G2 | pass | left loss -3.43 / -4.30 / -4.31 / -4.09% |
| G3 | **FAIL** | step 300 only: 1.293 against an upper bound of 1.282; 600/900/1200 pass |
| G4 | pass | throat u 2.588 vs 2.639 (-1.95%) |
| G5 | pass | 90% wet at step 366 vs 299 (+22%, limit 25%) |
| G6 | **FAIL** | centroid 384.5 / 395.0 vs 368.9 / 369.8; L1 0.277 / 0.342 (limits 0.202 / 0.226) |
| G7 | pass | drift +0.01% |
| G8 | **FAIL** | jet column 0.0371 vs 0.0770 L (-51.9%) |

**Section 5 timing.** Timing slot, the four runs back to back, all rc 0; dir `artifacts/seep-rewrite/timing-e20a907a`:

| | NB-FLIP | Seep | ratio | pass (>= 3x) |
|---|---|---|---|---|
| fenced median ms | 33.62 | **10.58** | **3.18x** | yes (5x stretch: no) |
| fenced p95 / mean ms | 37.38 / 32.97 | 11.76 / 10.56 | | |
| fenced thirds ms | 29.2 / 33.6 / 36.1 | 9.8 / 10.8 / 11.1 | | |
| pipelined ms/step | 32.40 | **9.91** | **3.27x** | yes |
| construction ms | 10768 | 1796 | | |
| peak GPU bytes requested | 1.244e9 | 1.049e9 | | |

NB-FLIP's rerun reproduces H1b to within 0.6% (33.69 fenced / 32.22 pipelined). Lane S's compare showed '-' for NB-FLIP pipelined only because it was not given the pipelined run.

**Audit.** Runs `seep-bench --steps=600 --every=60 --audit=0,60,120,300,600` for both methods (new `--audit` option,
tools/seep-audit-cells.ts); report `tools/seep-audit-report.ts`, written to `artifacts/seep-rewrite/audit-s2g/audit.md`.

(a) G3's "delivered".
- NB-FLIP's open throat cells hold 0.0080 L at every audit step (0 to 600), which is the full 64 cells, all h owners.
- Its h solid cells hold about 0.002 L and do not grow.
- The whole wall-zone growth (+0.109 L by step 120, +0.115 by 600) is in **4h owners overlapping the wall zone**. No throat cell is a 4h owner, so that growth is phi inside solid cells.
- Redefining delivered as left loss minus the change in open-cell wall liquid moves G3's upper bound at step 300 from 1.282 to 1.397 L. Seep's 1.293 then passes at every checkpoint (upper bounds 1.397 / 2.823 / 4.238 / 5.641). The gate is unchanged pending your decision.
- Seep, for contrast: its solid wall cells carry 0.39 L at step 0 and 0.87 L from step 60 of negative extrapolated phi. That is uncounted, and harmless to the gates.

(b) Jet trajectory. NB-FLIP: throat u0 2.64. Seep: throat u0 2.60; its jet's horizontal u is 2.0-2.25.
- **NB-FLIP's jet is ballistic.** Its mean y follows the parabola to within about 2 cells down to y = 30, then falls 2-4 cells below it. Its horizontal u stays at 2.3-2.8, which is u0. It reaches y = 10 at x = 380, against a ballistic floor at 383.
- **But NB-FLIP's jet carries about 2.4x its flux in volume.** A free jet with constant horizontal u holds Q/u per unit length, so 0.032 L over 64 cells. NB-FLIP's level set holds 0.076 L there, about 0.010 L per 8-cell slice moving at 2.5 m/s, which is about 0.6 L/s against the 0.26 L/s the throat delivers. So the excess is a fat level set around a sub-resolved 2x2 jet, not a slowed and thickened jet. The hypothesis that NB-FLIP's jet slows, thickens and lands short is refuted.
- **Seep's jet volume is flux-consistent:** about 0.0045 L a slice at 2.2 m/s is about 0.25 L/s, equal to Q. Its 0.037 L is close to the 0.032 L ballistic column.
- **But Seep's jet lands short.** It leaves the throat face at about 2.0 m/s rather than 2.6, and falls faster than g. At x = 344-352 its mean y is 13.5, where the parabola gives 34 from u0, or about 26 from its own 2.2 m/s. It reaches the floor at x = 354, against 381 ballistic. Both steps 300 and 600 agree.
- So, plainly: G8 as written rewards NB-FLIP's level-set bloat. Seep's trajectory is wrong in the other direction: too slow at the exit and too steep a fall.

(c) Right-chamber profile, Seep minus NB-FLIP.
- NB-FLIP's chamber is missing 0.60 L of its delivered volume at step 600 (22%) and 1.95 L at step 1200 (36%), with delivered computed as in (a).
- If that liquid is restored evenly over the floor, NB-FLIP's centroid moves 369.7 to 374.9 at step 1200, which closes **21%** of the 25.3-cell gap (22% at step 600).
- If it is restored only where NB-FLIP's film thins (x >= 400), it closes **124%**.
- Where NB-FLIP actually loses liquid cannot be located from these observables, so its film loss explains somewhere between about a fifth and all of the gap.
- The shape at step 1200: NB-FLIP's film is about 2 cells deep out to x = 400, then thins to about 1 cell. Seep's is 2.0-2.5 cells deep, then **piles at the far wall**: 3.7-4.4 cells at x = 464-512, which is +1.20 L of its +1.77 L excess.
- Seep's jet lands *short* of NB-FLIP's (354 vs 380), which by itself would pull Seep's centroid left. So the centroid gap is not where the jet lands; it is the far-wall pile, NB-FLIP's thinning film, or both.

## 2026-11-14 MILESTONE H6: corrected G3 and G8 (plan §4.1) in seep-compare; e20a907a re-judged

The observer (`tools/seep-observables.ts`, both observers) now also records, at every sample step:
- `wallOpen_L`: wall-zone h owners inside the throat patch;
- `jetSlices`: 24 slices of 8 cells from x = 256, right owners at y >= 8 only. Each slice gives liquid, the liquid-weighted owner-centre y, and the liquid-weighted mean x-face u.

These sums are computed in the same fixed-point pass as before. The observer runs only at sample steps, which timing excludes, so it costs nothing in timed steps. A GPU test checks them against the CPU oracle, including a synthetic descending jet. seep-compare changes:
- G3 now uses open-cell delivered. G8 is the trajectory gate at 300 and 600: impact x within max(8, 3S), and RMS height over x 256-352 within max(3, 3S).
- The §4 forms print as informational rows, "G3 (as written)" and "G8 (as written)".
- Runs that predate §4.1 are refused.

**Definition note (please confirm).** The slices hold only liquid at y >= 8, so NB-FLIP's slice mean y often ends at
10.5-11.0 in its last jet slice and never falls below 10. That happened in 3 of the 4 reference realisations;
the audit run happened to reach 9.9. Where no slice crosses 10, the impact is taken from the last two slices' line
extended to y = 10, by at most one slice. The report says when this happens ("k of n extended").

**New references** (`artifacts/seep-rewrite/nbflip-ref2/nbflip-check-{c,d}`): check-a/b predate the new fields. Both are 1200-step `--nb-stats`
check runs. NB-FLIP's code is identical between check-a/b's commit and e20a907a. To avoid lane S's uncommitted
`seep-kernels.ts`, these runs and the e20a907a re-run used a read-only export of e20a907a's lib/ and tools/ (plus the new
harness files) in ~/code/me/fluid-scratch/lane-h/tree-e20a907a. Their meta commit therefore reads "unknown".

e20a907a re-judged (`artifacts/seep-rewrite/seep-e20a907a/compare-g8.md`): G1, G2, **G3 (corrected)**, G4, G5, G7 pass.
G6 fails: centroid 384.1 / 396.8 vs 367.0 / 370.4. **G8 (corrected) fails**: impact x 354.3 / 354.1 vs NB-FLIP 380.8 / 382.8, and
RMS 9.97 / 10.43 cells against a limit of 3. G3 (as written) and G8 (as written) fail, informational only.

Explaining Seep's jet beyond x = 320 (lane S's 6bc9a3fa self-check vs H5):
- There is no discrepancy and no contamination: the slices exclude y < 8, and Seep's jet volume per slice stays flat at about 0.004 L through x = 352.
- A parabola fitted to Seep's slices at x < 320 implies u_x = 2.26-2.30 m/s from its curvature, matching the measured 2.25. It also implies an exit vertical velocity of -0.31 to -0.34 m/s.
- Extended past 320, that parabola predicts the measured heights to within 1.3 cells (e.g. 12.2-12.7 at x 344-352, measured 13.1-13.5).
- So Seep's e20a907a jet is ballistic for its own exit state (slow, angled down). H5's "falls faster than g" is withdrawn: H5 compared it against the parabola from the throat's u0 with a level exit.

## 2026-11-14 MILESTONE H7: combined gate round on S2h (bbc7828d)

Build: HEAD dd8cd428, whose lib/ is identical to bbc7828d (dd8cd428 adds only the lane H tools). The tree was clean at the check run
(`dirtyLibOrTools: false`). The timing runs read dirty only because of a one-line formatting fix in `tools/seep-compare.ts`
(G6 now prints ">" when L1 exceeds its limit), which the bench does not load. Lane S's current uncommitted
`lib/methods/seep` edits are all stamped 21:38:18 (+13), 20 s after the last timing run ended (21:37:58). The Seep timing runs
also end within 0.003 L and 1 particle of the check run at step 1200, so all runs measured bbc7828d.

Runs (all rc 0, no Metal aborts in the log), in `artifacts/seep-rewrite/final-bbc7828d/`:
- check: `seep-bench --method=seep --steps=1200 --every=60 --name=seep-check`
- timing, back to back: `seep-bench --method=seep --mode=fenced`, `--method=seep --mode=pipelined`, `--method=nbflip --mode=fenced`
  (`--slot=timing --steps=1200 --every=60`). NB-FLIP pipelined is reused from timing-e20a907a (32.40 ms; NB-FLIP code unchanged).
- `seep-compare artifacts/seep-rewrite/nbflip-ref2 timing-e20a907a/nbflip-timing-pipelined.jsonl final-bbc7828d/*.jsonl --candidate=seep-check`
  gives `compare.md`. The same judged on the timing-fenced run gives `compare-sensitivity-timing-fenced.md` (informational).

| gate | verdict | detail |
|---|---|---|
| G1 | pass | q_left 0.2590 vs 0.2700 L/s (-4.07%); windows 18/18 |
| G2 | pass | -2.76% / -3.87% / -3.92% / -3.74% at 300/600/900/1200 |
| G3 | pass | 1.278 in [1.111, 1.394]; 2.589 in [2.071, 2.821]; 3.902 in [2.779, 4.237]; 5.187 in [3.215, 5.643] L |
| G3 (as written) | pass, not gated | 1.278 in [1.111, 1.279] (0.001 under the bound) |
| G4 | pass | throat u 2.597 vs 2.639 m/s (-1.57%) |
| G5 | pass | 90% wet at step 271 vs 296; 16379 vs 16369 wet at 1200 |
| G6 | **FAIL** | 600: centroid 370.7 vs 367.0, L1 0.094 <= 0.327; 1200: centroid 385.9 vs 370.4 (limit 16), **L1 0.227 > 0.214** |
| G7 | pass | drift 0.00% vs NB-FLIP 0.51% |
| G8 | pass | 300: impact 375.8 vs 380.8 (limit 8), RMS 0.78 (limit 3); 600: impact 375.3 vs 382.8, RMS 1.40 |
| G8 (as written) | fail, not gated | jet column 0.0331 vs 0.0765 L (Seep is at the free-jet 0.032 L) |

§5 (timing slot):

| | NB-FLIP | Seep | ratio |
|---|---|---|---|
| fenced median ms | 34.72 | 10.74 | **3.23x** |
| fenced p95 ms | 38.90 | 11.98 | |
| pipelined ms/step | 32.40 (timing-e20a907a) | 9.88 | **3.28x** |
| construction ms | 11242 | 2140 | |
| peak GPU bytes | 1244173500 | 1049314004 | |

Performance: PASS (stretch 5x not reached). NB-FLIP's fenced median is 3% slower than at e20a907a (33.62 ms), which is run-to-run noise.

G6 reading:
- G6 is borderline, not a clear failure. Seep's own run-to-run spread at 1200 is a centroid of 381.0-386.6 and an L1 of 0.170-0.227. The timing-fenced run passes G6.
- The limit is narrow because NB-FLIP's two references agree closely (repeat L1 0.071, times 3 is 0.214).
- The profile difference at 1200, normalised:
  - Seep holds less in bins 4-8 (x 320-400): -0.015 to -0.026 per bin;
  - Seep holds more in bins 10-15, mostly at the far wall: bin 15 is +0.035, bin 14 +0.029.
- Seep's right chamber also holds 5.19 L where NB-FLIP's holds 3.39 L, because NB-FLIP loses about 1.8 L of floor film (plan §4.1 notes this part of the gap).
- More liquid in the film carries more momentum to the far wall, where it piles up. This would shift the profile even with an identical jet.
- Whether G6 should count NB-FLIP's film loss (as G3 now does) is the coordinator's call. Under G6 as written, S2h fails narrowly and nondeterministically at step 1200.

## 2026-11-14 MILESTONE H8: corrected G6 (plan §4.1 at 70351e00) in seep-compare

- `restoreLostVolume` and `normaliseProfile` (tools/seep-observables.ts, unit-tested):
  - each NB-FLIP reference's step-600/1200 right bins get lost = left loss − open-wall change − right volume, added uniformly over the 16 bins;
  - the restored centroid is the volume-weighted mean of the measured centroid and the chamber centre (x = 384).
- G6 now gates on two things:
  - L1 of the candidate's normalised profile against the mean of the references' normalised restored profiles, with the unchanged limit max(3× repeat L1, 0.15);
  - the centroid within 16 cells of [NB, NB_restored] (each the mean over references).
- "G6 (as written)" is printed and informational.

The plan's formula gives NB-FLIP losses of 0.50/0.51 L at step 600 and 1.96/2.02 L at 1200. That is more than lane S's 1.71/1.77 L, which is NB-FLIP's total drift. The difference is the roughly 0.25 L of ghost wall-zone liquid outside the open cells, which the open-wall term counts as lost.

Re-judged without new GPU runs:

| run | G6 1200: L1 vs restored (limit 0.214) | centroid in [354.4, 391.4] | G6 | as written | other gates |
|---|---|---|---|---|---|
| final-bbc7828d/seep-check | 0.155 | 385.9 | pass | fail (L1 0.227) | all pass, §5 PASS (`compare-g6.md`) |
| final-bbc7828d/seep-timing-fenced | 0.100 | 381.0 | pass | pass (0.170) | all pass (`compare-g6-sensitivity-timing-fenced.md`) |
| lane-s/seep-s2i | 0.200 | 388.2 | pass | fail (0.272) | all pass (`final-bbc7828d/compare-g6-s2i.md`) |

At step 600, all three pass G6 (L1 0.049 / 0.073 / 0.107 against 0.327). With this G6, bbc7828d's gate round passes every gate and §5.

## 2026-11-14 MILESTONE H9: gate round on S2j (d5d3ae2c)

**Build provenance.**
- Every run used HEAD 7ebdc7a6. Its lib/ is identical to d5d3ae2c, since 7ebdc7a6 is docs only. The tree was clean and every run reports `dirtyLibOrTools: false`.
- Lane S committed b5d7ea2d (docs only) at 22:22:01, during the NB-FLIP fenced run. That is why the NB-FLIP pipelined meta reads b5d7ea2d. lib/ is unchanged.
- All runs exited rc 0, and the system log shows no Metal aborts.
- Same-build cross-check, at step 1200:

  | Seep run | left L | particles | centroid |
  |---|---|---|---|
  | check | 363.4749 | 332248 | 382.0 |
  | timing fenced | 363.4737 | 332248 | 381.1 |
  | timing pipelined | 363.4745 | 332248 | 382.5 |

**Runs**, in `artifacts/seep-rewrite/final-d5d3ae2c/`:
- check: `seep-bench --method=seep --steps=1200 --every=60 --name=seep-check`;
- timing slot, back to back (22:21–22:23), `--slot=timing --steps=1200 --every=60`: Seep fenced, Seep pipelined, NB-FLIP fenced, NB-FLIP pipelined. This time both NB-FLIP modes were rerun.
- `compare.md` is the gated report against nbflip-ref2, including §5. `compare-sensitivity-timing-fenced.md` is informational.

| gate | check run | timing-fenced run (informational) |
|---|---|---|
| G1 | pass, -4.23%, windows 18/18 | pass, -4.21% |
| G2 | pass, -2.90 / -3.92 / -3.98 / -3.89% | pass |
| G3 | pass, 1.278 / 2.579 / 3.906 / 5.198 L in band | pass |
| G3 (as written) | pass, not gated (1.278 vs 1.279) | pass |
| G4 | pass, 2.597 vs 2.639 m/s | pass |
| G5 | pass, step 271 vs 296 | pass, step 268 |
| G6 | pass: at 1200, L1 vs restored 0.093 <= 0.214, centroid 382.0 in [354.4, 391.4]; at 600, 0.075 | pass: at 1200, 0.098, centroid 381.1 |
| G6 (as written) | pass, not gated (L1 0.154) | pass (0.163) |
| G7 | pass, 0.01% | pass, 0.00% |
| G8 | pass: impact 375.8 / 375.4 vs 380.8 / 382.8; RMS 0.74 / 1.33 | pass |
| G8 (as written) | fail, not gated (0.0331 vs 0.0765 L) | fail, not gated |

§5 (timing slot):

| | NB-FLIP | Seep | ratio |
|---|---|---|---|
| fenced median ms | 34.42 | 9.61 | **3.58x** |
| fenced p95 ms | 39.24 | 10.75 | |
| fenced mean ms | 33.89 | 9.51 | |
| pipelined ms/step | 32.19 | 8.87 | **3.63x** |
| construction ms | 10794 | 1848 | |
| peak GPU bytes | 1244173500 | 1049379540 | |

PASS. Measured against this round's NB-FLIP, the 5x stretch needs ≤6.88 ms fenced and ≤6.44 ms pipelined, so Seep is 2.73 ms (28%) and 2.43 ms (27%) short. Against the plan's 6.7 / 6.4 ms figures, it is 2.91 / 2.47 ms short.

Seep's fenced step time grows through the run: the medians of the three thirds are 8.75 / 9.66 / 10.11 ms, rising with particle count (166k at step 600, 332k at step 1200). NB-FLIP's thirds are 29.4 / 34.3 / 38.0 ms. My 9.61 ms is a 1200-step timing-slot median, which is slower than lane S's interleaved check-slot estimate of about 9.05 ms.

## 2026-11-15 MILESTONE H10: gate round on S2k + S2l (979d770b)

**Build provenance.**
- The check run used the main tree at 979d770b (`dirtyLibOrTools: false`, 08:29:49-08:30:01).
- Lane S then started uncommitted `lib/methods/seep` edits. The first changed file is stamped 08:32:39, after the first timing set.
- So the rerun timing set (r2) ran from a read-only `git archive 979d770b` export (`~/code/me/fluid-scratch/lane-h/tree-979d770b`, with my tools at HEAD identical). It used the main repo's runner, so its meta commit reads "unknown".
- All runs exited rc 0. The system log has no Metal command-buffer errors or GPU restarts.
- Same-build cross-check at step 1200 (left L / particles / centroid):

  | Seep run | left L | particles | centroid |
  |---|---|---|---|
  | check | 363.4611 | 332336 | 382.5 |
  | r2 timing fenced | 363.4623 | 332336 | 385.4 |
  | r2 timing pipelined | 363.4603 | 332336 | 377.9 |
  | contended timing fenced | 363.4623 | 332336 | 382.9 |

**Contention.**
- The first timing set (moved to `contended/`) gave NB-FLIP fenced 53.91 ms and pipelined 38.27 ms. NB-FLIP's code is unchanged, and it measured 33.6-34.7 ms in every earlier round. So the GPU was shared.
- A 2 s `ps` sampler saw no other harness or node GPU job, so the load came from outside node, most likely a Chrome tab (its GPU helper was busy).
- Seep in that set measured 8.12 ms fenced and 7.42 ms pipelined, the same as the rerun.
- The rerun (`fluid-scratch/lane-h/timing-round.sh`) starts each run only after the device has been under 15% utilisation (ioreg) for 5 s, and logs utilisation and other jobs every second.
- During the rerun, lane S's check-slot A/B jobs were interleaved. Their guard windows were 08:34:23-44, 08:35:01-10, 08:35:25-44 and 08:36:43-52. My bench windows were Seep fenced 08:34:46-57, Seep pipelined 08:35:11-22, NB-FLIP fenced 08:35:45-36:39 and NB-FLIP pipelined 08:36:58-37:50. They did not overlap, because the timing slot holds every check lock.

| gate | check run (gated) | r2 timing-fenced (informational) |
|---|---|---|
| G1 | pass, -3.98%, windows 18/18 | pass, -4.00% |
| G2 | pass, -2.61 / -3.67 / -3.72 / -3.63% | pass |
| G3 | pass, 1.295 / 2.592 / 3.905 / 5.174 L in band | pass |
| G3 (as written) | fail, not gated (1.295 > 1.279 at 300) | fail, not gated |
| G4 | pass, 2.597 vs 2.639 m/s | pass |
| G5 | pass, step 275 vs 296 | pass, step 268 |
| G6 | pass: at 1200, L1 vs restored 0.118 <= 0.214, centroid 382.5; at 600, L1 0.122 | pass: at 1200, 0.113, centroid 385.4 |
| G6 (as written) | pass, not gated (L1 0.198) | pass (0.181) |
| G7 | pass, -0.00% | pass, 0.01% |
| G8 | pass: impact 376.0 / 375.6 vs 380.8 / 382.8; RMS 0.79 / 1.29 | pass |
| G8 (as written) | fail, not gated | fail, not gated |

§5 (timing slot, r2, quiet GPU):

| | NB-FLIP | Seep | ratio |
|---|---|---|---|
| fenced median ms | 34.92 | 8.15 | **4.28x** |
| fenced p95 ms | 38.48 | 9.12 | |
| fenced mean ms | 34.34 | 8.05 | |
| pipelined ms/step | 34.22 | 7.42 | **4.61x** |
| construction ms | 11435 | 1865 | |
| peak GPU bytes | 1244173500 | 1049379540 | |

PASS.
- NB-FLIP pipelined at 34.22 ms is 6% above the earlier rounds' 32.19-32.40. Against 32.19 the pipelined ratio would be 4.34x.
- The 5x stretch against this round's NB-FLIP is ≤6.98 ms fenced and ≤6.84 ms pipelined. Seep is 1.17 ms (14%) and 0.58 ms (8%) short.
- Against the plan's 6.7 / 6.4 ms it is 1.45 / 1.02 ms short.
- Seep's fenced thirds are 7.34 / 8.16 / 8.63 ms, still rising with particle count. Lane S's check-slot A/B figure was 7.6-7.9 ms.

## 2026-11-15 MILESTONE H11: gate round on S2m + S2n (6fbe1174)

**Build provenance.**
- All five runs used a read-only `git archive 6fbe1174` export (`~/code/me/fluid-scratch/lane-h/tree-6fbe1174`) and the main repo's runner, so the meta commit reads "unknown".
- The main tree was clean at HEAD 6fbe1174 before and after the round.
- Every run exited rc 0, and the system log has no Metal command-buffer errors or GPU restarts.
- Lane S held off GPU runs for the round. The 1 s sampler logged no other harness or node GPU jobs.
- The quiet-GPU gate passed after 5 s before each timing run, with device utilisation at 0% between runs.
- Same-build cross-check at step 1200:

  | Seep run | left L | particles | centroid |
  |---|---|---|---|
  | check | 363.4711 | 332339 | 380.5 |
  | timing fenced | 363.4691 | 332344 | 383.4 |
  | timing pipelined | 363.4702 | 332340 | 383.7 |

Runs, in `artifacts/seep-rewrite/final-6fbe1174/`:
- check run, 09:17:06-17;
- `timing-round.sh`: Seep fenced 09:17:29-40, Seep pipelined 09:17:48-59, NB-FLIP fenced 09:18:06-59, NB-FLIP pipelined 09:19:07-58.

| gate | check run (gated) | timing-fenced (informational) |
|---|---|---|
| G1 | pass, -4.12%, windows 18/18 | pass, -4.07% |
| G2 | pass, -2.87 / -3.94 / -3.91 / -3.81% | pass |
| G3 | pass, 1.289 / 2.573 / 3.907 / 5.192 L in band | pass |
| G3 (as written) | fail, not gated (1.289 > 1.279 at 300) | fail, not gated |
| G4 | pass, 2.597 vs 2.639 m/s | pass |
| G5 | pass, step 276 vs 296 | pass, step 266 |
| G6 | pass: at 1200, L1 vs restored 0.083 <= 0.214, centroid 380.5; at 600, 0.112 | pass: at 1200, 0.156, centroid 383.4 |
| G6 (as written) | pass, not gated (L1 0.154) | fail, not gated (0.231) |
| G7 | pass, 0.01% | pass, 0.01% |
| G8 | pass: impact 375.7 / 375.6 vs 380.8 / 382.8; RMS 0.80 / 1.32 | pass |
| G8 (as written) | fail, not gated | fail, not gated |

§5 (timing slot, quiet GPU):

| | NB-FLIP | Seep | ratio |
|---|---|---|---|
| fenced median ms | 34.43 | 7.75 | **4.44x** |
| fenced p95 ms | 38.82 | 8.71 | |
| fenced mean ms | 34.03 | 7.71 | |
| pipelined ms/step | 32.61 | 6.91 | **4.72x** |
| construction ms | 11102 | 1925 | |
| peak GPU bytes | 1244173500 | 1049641156 | |

PASS.
- Against 979d770b (H10), Seep fenced is 0.40 ms faster (8.15 to 7.75) and pipelined is 0.51 ms faster (7.42 to 6.91). This matches lane S's estimate of about -0.4 ms.
- The 5x stretch against this round's NB-FLIP is ≤6.89 ms fenced and ≤6.52 ms pipelined. Seep is 0.86 ms (11%) and 0.39 ms (6%) short.
- Against the plan's 6.7 / 6.4 ms it is 1.05 / 0.51 ms short.
- Seep's fenced thirds are 7.09 / 7.82 / 8.24 ms.

## 2026-11-15 MILESTONE H12: gate round on S2o + S2p (1209eb14)

**Build provenance.**
- All five runs used a read-only `git archive 1209eb14` export (`~/code/me/fluid-scratch/lane-h/tree-1209eb14`) and the main repo's runner, so the meta commit reads "unknown". `dirtyLibOrTools` is false.
- The main tree was clean at 1209eb14 before and after the round.
- Every run exited rc 0, and the system log has no Metal command-buffer errors or GPU restarts.
- Lane S stayed off the GPU, and the 1 s sampler logged no other jobs. The quiet-GPU gate cleared in 5 s before each run.
- Same-build cross-check at step 1200:

  | Seep run | left L | particles | centroid |
  |---|---|---|---|
  | check | 363.4697 | 332344 | 383.7 |
  | timing fenced | 363.4688 | 332344 | 383.4 |
  | timing pipelined | 363.4659 | 332340 | 384.6 |

Runs:
- check, 10:33:35-46;
- Seep fenced 10:33:54-34:04, Seep pipelined 10:34:12-22, NB-FLIP fenced 10:34:30-35:22, NB-FLIP pipelined 10:35:30-36:19.

| gate | check run (gated) | timing-fenced (informational) |
|---|---|---|
| G1 | pass, -4.11%, windows 18/18 | pass, -4.07% |
| G2 | pass, -2.84 / -3.93 / -3.81 / -3.79% | pass |
| G3 | pass, 1.276 / 2.585 / 3.901 / 5.196 L in band | pass |
| G3 (as written) | pass, not gated (1.276 <= 1.279) | fail, not gated (1.286) |
| G4 | pass, 2.597 vs 2.639 m/s | pass |
| G5 | pass, step 272 vs 296 | pass, step 277 |
| G6 | pass: at 1200, L1 vs restored 0.134 <= 0.214, centroid 383.7; at 600, 0.123 | pass: at 1200, 0.122, centroid 383.4 |
| G6 (as written) | pass, not gated (L1 0.207) | pass (0.181) |
| G7 | pass, 0.01% | pass, 0.01% |
| G8 | pass: impact 375.5 / 375.8 vs 380.8 / 382.8; RMS 0.80 / 1.32 | pass |
| G8 (as written) | fail, not gated | fail, not gated |

§5 (timing slot, quiet GPU):

| | NB-FLIP | Seep | ratio |
|---|---|---|---|
| fenced median ms | 34.39 | 7.03 | **4.89x** |
| fenced p95 ms | 37.56 | 7.94 | |
| fenced mean ms | 33.50 | 6.92 | |
| pipelined ms/step | 32.16 | 6.32 | **5.09x** |
| construction ms | 10874 | 1872 | |
| peak GPU bytes | 1244173500 | 1049641156 | |

PASS.
- **Pipelined reaches the 5x stretch.** 6.32 ms is under this round's ≤6.43 and under the plan's 6.4.
- Fenced is 0.15 ms (2%) above this round's ≤6.88, and 0.33 ms above the plan's 6.7.
- Against 6fbe1174 (H11), Seep is 0.72 ms faster fenced (7.75 to 7.03) and 0.59 ms faster pipelined (6.91 to 6.32). This is about lane S's -0.65 ms.
- My fenced 7.03 ms matches lane S's check-slot A/B figure of about 7.03 ms.
- Seep's fenced thirds are 6.30 / 7.07 / 7.39 ms.
