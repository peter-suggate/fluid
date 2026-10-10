# Narrow-band FLIP only: purge plan (2026-10-11)

## Execution results

Stages 1–8 are implemented. Narrow-band FLIP, the plugin architecture, SVO,
AdvanceLab, Shape Lab and UI remain. The final solver census is 74 modules and
223 eager compute pipelines (73/222 without timestamp-query). Shader text and
pipeline descriptors are identical across dam32, dam64 and torus; all WGSL parses.
Nine live native continuation/setup kernels were retained after tracing their
default-frame callers; the roughly 201-pipeline estimate below was not exact.

Paper PDFs and `.txt` sources are preserved, including papers about retired methods.
All extracted paper images have been deleted; source papers are exempt from stale-doc cleanup.

The subsequent instruction to delete all skipped test code supersedes the planned
GPU-test ports below. Conditional GPU cases and ignored Rust cases, their orphaned
helpers, and the obsolete Dawn runner were deleted. Remaining unit tests pass
847/847 with zero skips; retained Rust kernel tests pass 3/3 with zero ignored.
Fifteen final GPU cases passed using temporary copies of applicable checks.

First-scene setup measured 87.3 s; a forced shader cache-miss probe measured 96.7 s,
then 0.189 s for the torus with the same shaders. Runtime geometry averaged about
7.1% frame overhead against a temporary baked-geometry probe. Measurements,
scope corrections and validation are in
[the verification report](narrow-band-flip-purge-verification-2026-10-11.md).
Deletion manifests accompany this plan. The sections below preserve the original
stage specification and estimates; their old line references are historical.

## Goal

Main supports one simulation method, narrow-band FLIP at its app defaults, plus the
parameters that are free at run time (a uniform write). Every other method, every
non-default mode of the uniform solver, and the tests, tools and scripts that exist
only for them are removed. All pipelines a scene can reach are built before the first
frame; nothing compiles while the simulation runs.

Line references below are as of `c0aa66cd`.

## Scope

| Stays | Goes |
|---|---|
| Narrow-band FLIP (`uniform-narrow-band-flip`) at app defaults | Losasso, Power Liquids, octree-shared |
| SVO renderer (`lib/svo`) and the `lib/core` files it needs | Adaptive mass, adaptive volume, `lib/sparse-world` |
| 2-D lab: `advance-lab`, `app/advance-lab`, `lib/physics-wasm`, `tools/wasm`, `rust` | SPH, APIC, MAC (`particle`, `mac-shared`, `uniform/mac`) |
| `SimulationMethod` contract and registry (one entry) | Uniform Geometric and legacy Uniform as methods, and code only they run |
| Compare mode (parameters and scenes) | `app/cm12-hole-probe`, `app/api/cm12-probe` |
| Shape lab | Method picker, per-method URL state, compile-by-need |

History is on `archive/all-methods-2026-10-11` (local and origin), which is `c0aa66cd`.

## Expected result

| Measure | Now | After |
|---|---|---|
| Pipelines a default scene can reach | 432 | 239 (stage 6), about 201 (stage 8) |
| Other method packages | about 210k lines | 0 |
| `lib/methods/uniform` | 40,231 lines | about 30k |
| Test files | 234 | about 190 |
| Tools | 417 files | about 190 |
| `package.json` scripts | 243 | about 60 |

The 239 figure is arithmetic from the CPU census (`tools/capture-uniform-wgsl.mts`), not
a GPU run. Cold compile time for 239 pipelines built up front is unmeasured; the last
everything-up-front figure was 110-129 s at about 360 pipelines.

## Working rules for this purge

- `npm run check:types` passes at the end of every stage. `tsconfig.json` includes every
  `.ts`/`.mts`, so a tool or test that imports deleted code goes in the same stage.
- Unit tests that fail because their subject was removed are deleted, not rewritten.
- Dawn runs are targeted lanes only, and only for stages that change shader text (5-8).
- Deletion is by file list. Each stage writes its list, checks it against the import
  graph (nothing kept may import it), then removes it.

## Stage 0: done

- Archive branch created and pushed; 91 local and 4 origin branches deleted; 33
  worktrees removed. Backups outside the repo: `~/code/me/fluid-deleted-branches-2026-10-11.bundle`
  and `~/code/me/fluid-deleted-worktrees-2026-10-11/`.
- `lib/methods/particle/scan.ts` moved to `lib/methods/uniform/particle-bin-scan.ts`
  (the default method imported it from the APIC package).

## Stage 1: make the default independent of what will be deleted

1. Flatten the method object. `uniformNarrowBandMethod` is built by spreading
   `uniformVolumeMethod`, which spreads `uniformMethod`. Write it out as one object in
   `uniform-narrow-band-method.ts`; drop parameters it pins (`sharpeningSweeps`,
   `sharpeningDistance`, dust thresholds, `phiDrain`, surface-volume parameters).
2. `lib/methods/index.ts` registers narrow-band FLIP only. The octree lane and row
   capacity registrations go with it.
3. Stored scenes. A saved scene naming a retired method throws on open
   (`stores/method-store.ts:25-28,64-67`, `simulation/controller.ts:787`). Migrate any
   unknown method id to the default in `scene-library.ts:109` and in URL hydration.
4. `gpu=safe` is pinned to Losasso (`WebGPUViewport.tsx:1161`, `gpu-startup.ts:21,66-74`).
   Point it at the default.
5. `LIVE_SOLID_EDIT_METHODS` (`method-contract.ts:215`) omits narrow-band FLIP, and
   `studioRegionSpaceForMethod` (`editor-refinement-region.ts:271`) matches only
   `uniform-volume`. Add narrow-band FLIP to both, then remove the lists once there is
   one method.
6. `lib/features/ui/FeatureSlot.tsx` drops the adaptive-mass and adaptive-volume views.
7. Detach the 2-D lab from adaptive volume. The lab imports four files from that
   package: `features/advance-slice/{definition,advance-work,loop}.ts` and
   `sparse-cm12-stages.ts` (2,636 lines). The first three have type-only imports. Move
   all four into `advance-lab/`, with the three small stage-control definitions the
   stage table imports (`correction-controls.ts`, `features/adaptivity/definition.ts`,
   `features/algorithms/definition.ts`), trimmed to what the lab reads. The exact
   closure of those three is not yet read; if it is large, the lab's Adaptive half is
   reduced to the stage table it displays.
8. The lab's Uniform half and `lib/physics-wasm` import `uniform-geometric-parameters.ts`,
   `uniform-volume-initial.ts`, `uniform-volume-pipeline.ts` and `uniform-paper.ts`.
   These are pinned: stage 5 may not delete them.
9. `gpu-frame-profile` skill: `tools/profile-mini-dam-xctrace.ts` imports
   `power-dam-lane-environment.ts`. Re-point it at a narrow-band FLIP dam.

## Stage 2: delete the other method packages

One change, because they share `octree-shared` and the type check couples them.

- `lib/methods/{losasso,power,octree-shared,adaptive-mass,adaptive-volume,sph,particle,mac-shared,verification}`,
  `lib/methods/uniform/mac`, `lib/sparse-world` (includes the 14 MB
  `lib/methods/power/generated/octree-power-catalog.bin`).
- `app/cm12-hole-probe`, `app/api/cm12-probe`.
- `lib/harness` files that drive retired methods (`webgpu-smoke-executor.ts` and its
  callers, `sparse-cm12-dawn-*`). Harness files the SVO tests use stay.
- Every test and tool that imports the above: 18 test files, about 205 tools, 159 scripts
  (6 already dangling). `tools/benchmark-apic.ts` and similar go whole.
- `lib/core/{method-contract,webgpu-eulerian,webgpu-renderer}.ts` lose their
  `sparse-world` types.

## Stage 3: prune `lib/core` and the method machinery

Unregistering alone frees almost nothing here: 277 of 301 core files stay reachable
through the renderer hubs. The cuts are in `webgpu-renderer.ts`,
`webgpu-water-pipeline.ts`, `visualization-catalog.ts`, `method-contract.ts` and
`url-state.ts`.

- Delete after the hub cuts (about 25 files, 11k lines): global-fine surface, octree
  debug views, stage lens, face-velocity overlay, tracer overlay, secondary particles,
  CM12 dirty visualisations, tall-cell grid, cell-trace probe (octree only, and the
  widest UI cut).
- Delete outright (13 files, 2.9k lines): core files only the retired packages imported.
- Delete `lib/features/pressure-inspection` and `lib/features/topology-freeze` if no
  kept method declares them.
- Collapse: solver picker (`SceneToolstrip.tsx:170-187`), `controller.setMethod`,
  `interactiveMethodId`, URL `method=`, per-method override buckets, contract hooks only
  retired methods implemented.
- Must stay despite the names, because SVO or the default uses them:
  `webgpu-sparse-scene-proxies.ts`, `webgpu-fluid-brick-residency.ts`,
  `webgpu-pass-broker.ts`, `adaptive-sparse-brick-plan.ts`, `sparse-scene-domain.ts`,
  `webgpu-sparse-brick-topology-mutation.ts`, `webgpu-voxel-debug.ts`,
  `webgpu-solid-world-pages.ts`, `cm12-numerics.ts`, `tall-cell-diagnostics.ts`,
  `gpu-performance-activity.ts`, `levelset-consumer-abi.ts`,
  `marching-cubes-lookup.wgsl.ts`, `webgpu-particle-overlay.ts`,
  `method-parameter-variants.ts`, `method-harness-contract.ts`.

The lab reaches core through its own imports; anything it needs stays, and is listed
by the import graph before each deletion.

## Stage 4: pin non-default settings

| Setting | Pinned to | Removes |
|---|---|---|
| `detailPolicy` | dynamic | Full and Requested, host region planner (`uniform-detail-planner.ts`, `uniform-detail-requests.ts`) |
| `adaptiveSurface` | on | fixed-band arm; declared default is "off" today, app default "on" |
| `coarseParticleMode` | off | separate particle render surface |
| pressure full cycles / V-cycles / sweeps | 3 / 4 / 6 | three shader-baked values |
| QA setters, `uniformAbOn` (73 call sites), env flags | off | `FLUID_UNIFORM_AB_OFF`, `FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT`, `FLUID_MIXED_HOST_DIAGNOSTICS`, `FLUID_MIXED_PRESSURE_TRACE`, `FLUID_UNIFORM_WINDOW_LAG_PAD`, `FLUID_UNIFORM_WINDOW_DISPATCH`, `FLUID_UNIFORM_PRESSURE_SETUP_AUDIT`, `FLUID_UNIFORM_ACTIVE_REGION` |

Options with no reference anywhere go: `pressureSmoothingForQA`, `pressureAuthorityForQA`,
`systemBuildForQA`, `extensionWorkForQA`, `pressureStorageForQA`, `geometricRedistance`,
`mixedCoarseningReach`, `mixedCoarseningHysteresis`, `pressureCycleDispatch`.

Stay as run-time parameters: detail criteria and thresholds, adaptive budget and fade,
fine-grid padding, pressure tolerance, extension sweeps, time step, rigid coupling,
gravity, density, viscosity, surface tension, cubic phi advection.

Tests: ten narrow-band tests run only `detailPolicy: "full"` and three mix arms. Each
is re-pointed at `appDefaults`; one that cannot hold there is deleted.

## Stage 5: delete uniform-solver code the default never runs

About 10,300 of 40,231 lines. "Certain" means no narrow-band path constructs or
compiles it; "defaults-only" means reachable only through a setting pinned in stage 4.

Certain (about 3,200):
- Native `uv*` kernels in `uniform-volume.wgsl.ts` (1,113; keep `uniformVolumeTargetWGSL`
  and the byte constants).
- Page generation, support, redistance, phi-band and stencil modules (775, test-only).
- Alternative detail-storage placements and uncalled QA methods (about 690); `audit`,
  `siteWords`, `currency` in `uniform-detail-fields.ts`.
- Paged-storage arms (about 180; `paged` is always false at `:938`).
- `uniform-detail-pool.ts`, orphans `uniform-mixed-pressure-topology.wgsl.ts`, small dead
  WGSL files (`uniform-volume-pages`, `uniform-volume-donor-sum`, `uniform-source`,
  `uniform-sharpen-budget`, `uniform-volume-correction`).

Defaults-only (about 7,100):
- Native kernels in `webgpu-uniform-reference.wgsl.ts` (1,476) and the legacy dense path
  in `webgpu-uniform-reference.ts` (about 900, `:3339-3657` plus members).
- Stages only Uniform Geometric builds: `UniformMixedTransportStage` (744),
  `UniformMixedSharpening` (659), `UniformMixedCleanup` and
  `UniformMixedSolidDisplacement` (259), Section 3.3 extrapolation (about 766), CM11a
  native cycle (506 WGSL, about 400 host), non-narrow-band entries in
  `uniform-mixed-surface.ts` (308).

Pinned by the lab (stage 1.8) and therefore kept even if the default does not need them.

Coverage that exists only as Uniform Geometric tests is ported to narrow-band FLIP, not
deleted: rigid bodies, live voxel edit, solids promoted to h, dynamic relayout,
surface-distance filter, domain faces, hydrostatic rest, force cache, cubic phi advection.
Narrow-band FLIP omits the solid displacement stage, so its live-edit path has no test
today; that test is written in this stage.

## Stage 6: remove pipeline twins (432 to 239)

| Cut | Pipelines | How |
|---|---|---|
| Solid-free twins (`umSolidsPresent`) | -144 | flag in the existing `umSolidParams` uniform, early-outs kept |
| FLIP coarse-only set (`nbCoarseOnly`) | -32 | delete; Dynamic holds fine capacity |
| Inline surface-tension forces | -3 | cached path always |
| Band pressure variants (`measure@1-3`, fused small band, 64/256-lane coarse) | -9 | keep 1024-lane and `coarseSweep` |
| Diagnostics, trace marker, census `budget`, duplicate momentum cache | -5 | delete |
| General h-store form (`udWindow`) | 0 now, -292 avoided | pin direct form |

Gate: one A/B on one scene for the solid flag. The only prior figure is +5.6% on a
solid-free dam under Uniform Geometric with the solid library always compiled.

## Stage 7: remove compile-by-need

With the twins gone, a first voxel edit or first body is a uniform write.

- `uniform-pipeline-needs.ts`; `pipelinesPrepared`, `warmPipelines`, the "preparing"
  status and held-change replay (`webgpu-uniform-reference.ts:771-779,2232-2309,3205-3269`).
- Throwing selectors in `uniform-mixed-solid.wgsl.ts:142-148,293,318` and
  `uniform-detail-fields.ts:598-602,1292`.
- Host opt-ins: `method-contract.ts:486-496`, `webgpu-renderer.ts:1049-1051,2596,2669,2680,4073-4077`,
  `webgpu-render-worker.ts:121`, `solver-detail.ts:60-63`, `UniformCoarseControl.tsx:249,258-259`.
- Tests that pin it: `uniform-mixed-live-solid-edit-dawn:61-124`,
  `uniform-mixed-rigid-body-dawn:118,170`, `uniform-dynamic-coarsening-dawn:112`,
  `uniform-force-cache-dawn:31`; about 30 others only await `pipelinesPrepared()`.

Gate: cold setup time and pipeline count on the torus and one dam, against the 120 s
ceiling in `tests/falling-water-torus-dawn.test.ts:6`.

## Stage 8: make shader text independent of the tank

This is the original complaint: 68 of 78 shader modules differ between two scenes, so a
new tank misses the Metal cache for nearly everything.

- Move to a uniform: `UM_D`, `UM_T`, `UM_TILES`, `UM_SOLID_COUNT` offset, band `SLOTS`,
  `UM_H`, scratch-arena offsets, layout builder / detail ring / FLIP dimensions,
  `umWallReachGroups`, `ROOT_NX/NY/NZ`. Candidate home: extend `umCounts` or add one
  layout-words uniform per solver.
- Cannot move as they stand: pressure-root level tables (`UM_LD/LP/LB/LM/LC`, array
  sizes), the `umIterate` workgroup array, per-level pipeline counts. Moving them means
  a launch word and tables padded to a maximum (239 to about 201).
- Measured cost of the arithmetic form is +3.1% on dam64; per-axis masks were
  unmeasurable. Gate: frame time on one scene, then warm-cache load time on a second
  tank after loading the first.
