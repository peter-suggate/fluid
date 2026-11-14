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
