# Method decoupling handoff — fluid-core + three method plugins

**Goal.** Decouple the fluid methods into a physically separated layout: a `fluid-core`
of truly general-purpose code, and one self-contained folder per method — **uniform**,
**losasso adaptive**, **power liquids** — with every method-specific visualization,
panel contribution, harness hook, and shader colocated with its method and reached only
through a registered plugin contract. Everything that belongs to no surviving method is
deleted.

This doc is the implementation plan. Every claim below was verified against HEAD
(c1f4c51 + the current uniform mass-conservation working set) by a five-way audit of the
import graph, the god files, and the UI/harness seams. File:line references are to that
state and will drift as phases land — re-verify before acting on any single one.

## 0. Status (2026-08-14, uncommitted working tree)

| phase | state |
|---|---|
| 0 — guardrails, deletions, decisions | done |
| 1 — cut the SCC, shrink the contracts | done |
| 2 — physical layout + uniform first | done |
| 3 — WGSL inversion + planner extraction | done |
| 4 — split the god class; lane strategies | done |
| 5 — registry promotion + UI plugins | done |
| 6 — harness plugins | in progress |
| 7 — sweep | in progress |

Landed results worth carrying:

- **Boundary violations 43 → 9** (`node --import tsx tools/check-module-boundaries.ts`).
  The nine survivors are all `octree-shared → method-*` or `octree-shared → svo` and are
  listed in Phase 7 below.
- **`webgpu-octree.ts` 10,805 → 3,992 lines**, with the lane strategies split out into
  `octree-coarse-dynamics-lane.ts` (836), `power/octree-power-lane.ts` (2,145) and
  `losasso/octree-losasso-lane.ts` (2,068). The 66-member `OctreeCoarseDynamicsLane` is
  the Phase 4 contract.
- **WGSL byte-identical across the 6,800-line move.** `octreePowerProjectionShader`
  190,220 bytes sha256 `2b17d6ae…f46f6`; `octreeLosassoProjectionShader` 190,340 bytes
  sha256 `3d944b68…5184d5`. Re-check these after any lane edit.
- **Three registered methods** — `losasso`, `power-liquids`, `uniform`; all interactive;
  `uniform` the default; `coarseBackend` deleted everywhere.
- Dawn: `symmetric-expansion:power2017` and `symmetric-expansion:fine` both PASS with 0
  failing findings. The 250-step losasso main lane is 10-of-46, **identical to the
  pre-split baseline** — i.e. the split moved nothing. Those ten are pre-existing physics
  and are parked; see §0.2.

### 0.1 The app has three module graphs, not one

This bit the refactor and is not obvious from the layout. `vinext` / `@vitejs/plugin-rsc`
gives the app **three module graphs that share no evaluation**:

1. **server / RSC** — `app/**` and its static imports.
2. **client** — each `"use client"` module the server graph hands the router, plus its
   imports; evaluated in the browser *and* in the separate SSR pass. Today
   `components/AppShell.tsx` and `components/SceneLibrary.tsx` (the library arrives as
   `children`, so it is a **sibling** entry, not a descendant of the shell).
3. **worker** — `lib/core/webgpu-render-worker.ts`, spawned by `new Worker(new URL(…))`.

Each is a composition root and each does its own `import "@/lib/methods"`. Because
`lib/core/stores/method-store.ts` resolves `defaultMethodId()` at module scope, dropping
any one of them 500s every page with *"No simulation methods are installed"*.
`tools/check-method-install.ts` originally modelled a single graph — its comment
reasoned that "the router evaluates the root layout around every route, so wiring it
wires the whole browser app" — and reported green while every page was broken. It also
credited the worker's install to the page that spawns it, because the reach walker
followed `new URL(…)` edges. A `new URL(…)` specifier is a **thread** boundary, not an
import: `reach(roots, { evaluated: true })` in `tools/import-partition.ts` is the reading
any question about module-scope side effects wants.

### 0.2 Parked: the losasso symmetry findings

Peter parked further fluid-correctness work on 2026-08-14 in favour of finishing the
re-architecture. Recorded here so the next pass does not re-derive it:

- Three symmetry fixes did land during Phase 4/5 and are in the tree: exact
  superaccumulator reductions replacing 11 CompensatedF32 folds (+9.0%), page-level band
  membership in the extension band (−2.2%), and a `seedOrderKey` JFA tie-break in the
  fine level-set redistance (+0.4%). The main lane stayed at 10-of-46 findings.
- **The factor-1 (genuinely adaptive coarse-band) lane and the factor-4 lane have
  different D4 generators.** At factor 1, `swap-xz` is clean and the *reflections* break:
  velocity loses bit-exact D4 first, alone, at step 18, born in the pressure projection's
  axis-face velocity update at 1 ulp. Advection, the predictor and the pressure rows
  themselves stay bit-exact; the extension is a passenger. The unverified next step is
  that one of `faceMetrics.z` (`deltaYCells`), `face.openFraction` or
  `face.inverseDistance` is not itself mirror-exact. At factor 4 the polarity is
  reversed, so a fix for one lane should not be assumed to help the other.
- The factor-1 lane does **not** fail-closed — it runs all 250 steps and then throws in
  the terminal `Losasso cutover oracle`, before diagnostics are evaluated, which is why
  it normally emits no findings at all. `FLUID_LOSASSO_CUTOVER_REPORT_ONLY=1` demotes the
  oracle and yields 41 findings, 9 failing.
- `fluid-symmetry-losasso-projected-velocity` is a **broken diagnostic** (~108k
  mismatches with null values at step 1), not a physics signal.

---

## 1. Ground truth (what the audit established)

### 1.1 The three methods are not three methods today

- `lib/methods/index.ts:10-14` registers `octreeMethod`, `uniformMethod`,
  `cpuReferenceMethod`. **"Losasso adaptive" and "power liquids" are one registry entry**
  (`id: "octree"`) forked by a construction-time param `coarseBackend: "losasso" |
  "power2017"` (`lib/methods/octree.ts:42`, resolved into a frozen policy object by
  `resolveOctreeCoarseDynamics` in `lib/octree-coarse-backend.ts:139`).
- "Power liquids" = the `power2017` backend: Aanjaneya et al. 2017, *Power Diagrams and
  Sparse Paged Grids for High Resolution Adaptive Liquids* (paper vendored at
  `docs/papers/aanjaneya-2017-power-liquids.pdf`). Power-diagram faces at T-junctions,
  SPGrid paged pyramid, §4.3 hybrid preconditioner, separate factor-4 fine band.
  `normalizeOctreeMethodValues` pins `globalFineLevelSetFactor="4"`,
  `topologyCadenceAdvances=1` for it (`lib/methods/octree.ts:70-77`).
- "Losasso adaptive" = the default backend: `coarseBackend=losasso` with
  `globalFineLevelSetFactor=1` (`coarseOnlySurfaceTracking`, `lib/webgpu-octree.ts:2113`).
- `cpuReferenceMethod` (+ `lib/eulerian-solver.ts`) is not in the survivor set → delete.

### 1.2 Footprints (verified exclusive by reverse-import scan)

| Lane | Exclusive lib code | Notes |
|---|---|---|
| Uniform | 8 files, ~4.2k lines | `lib/methods/uniform.ts`, `webgpu-uniform-reference{,.wgsl}.ts`, `webgpu-uniform-pressure-multigrid{,.wgsl}.ts`, `webgpu-uniform-velocity-extrapolation{,.wgsl}.ts`, `uniform-host-allocation.ts` |
| Losasso | 41 files, ~17.3k lines | all `webgpu-octree-losasso-*` + `octree-losasso-operator.ts`; **zero imports of `webgpu-octree.ts`** — flow is one-way into the lane |
| Power | ~30 files, ~22.9k lines | `webgpu-octree-{spgrid-vcycle,air-velocity-support*,structured-*,power-*,section43-preconditioner}.ts` + CPU oracles/audits + `lib/generated/octree-power-catalog.{ts,bin}` (14.2 MB binary, 8,083 configs) |
| Octree-shared | ~130 files total octree-private; the shared subset both lanes need | owner-pages (1,609), pipelined-mgpcg (2,297, both lanes), fine-levelset family, worksets, topology-epoch, coarse/fine level-set exchanges, `webgpu-quadtree-builder.ts` (misnamed; resident level-set transport used by both lanes — rename, don't delete) |
| Shared core | 83 files (octree-reach ∩ uniform-reach) | includes 5+ octree-owned modules pulled in only via `lib/methods/types.ts` type imports — see §3.2 |
| Renderer/scene-only | 59 files | SVO stack (62 files, ~43.5k lines) is **method-agnostic — zero mentions of losasso/power2017/coarseBackend**; consumes only `SparseVoxelSceneRenderSource` |

### 1.3 The dependency graph is one SCC with exactly two load-bearing cycle edges

`reach(uniform) == reach(octree) == reach(renderer)` — an identical 281-file set. The
cycle:

```
methods/uniform.ts → methods/types.ts → resource-readiness.ts
  →(1) webgpu-renderer.ts (import type { GPUStatus, EffectiveRendererStatus })
  →(2) ./methods (renderer imports getMethod + the whole registry)
  → methods/index.ts → methods/octree.ts → webgpu-octree-eulerian.ts → webgpu-octree.ts
```

Cut (1) by moving the status types into core; cut (2) by injecting the registry into the
renderer. Then the partition in §1.2 falls out mechanically.

### 1.4 The god file, measured

`lib/webgpu-octree.ts` (12,229 lines) decomposes as:

- **Lines 168–1686**: types, ABIs, env gates, and *pure CPU capacity planners* — no GPU
  state; the natural first extraction.
- **Lines 1687–8091**: `WebGPUOctreeProjection` — 6,405 lines, ~200 flat private fields,
  857-line constructor, **~25 `backend === "losasso"` conditional sites**, and a single
  `destroy()` touching ~55 subsystems. Clean per-lane regions exist:
  `initializeLosassoAuthority` (2880–3162), `encodeLosasso` (5328–5411) vs
  `initializeNativePowerAuthority` (3756–4220), `encodeNativePowerAssembly` /
  `encodeStructuredProjection` (5103–5182); the shared topology-candidate/frontier engine
  is 4547–4820 + 5183–5229; `encodeSurface` (5583–5904) is the shared fine-band step with
  scattered lane branches.
- **Lines 8213–12032**: one ~3,800-line WGSL template. Losasso arena/mass/phi WGSL is
  embedded at 8377–8495 and 9578–9670 alongside Power's directory authority; both funnel
  into `correctedCoarsePhi`.
- **Three chained string transforms** mutate that WGSL: `octreeGradingFixpointShader`
  (env-decided **at module eval**, 12038), `octreeLosassoSurfaceGradingShader` (12054 —
  two exact-string `.replace()` calls that derive the Losasso topology shader from the
  Power one and throw if the anchors move), and the regex-driven activity instrumenter
  (12091). **The lanes cannot own separate WGSL files until this is inverted into
  composition.**
- Only **one production file imports the module**: `lib/webgpu-octree-eulerian.ts:52`.
  Everything else is tests (35 files) and 4 tools.
- Dead exports (zero references anywhere): `expandOctreePressureRowAffectedOneRing`,
  `enumerateOctreeFrontierCandidateLattice`, `mergeOctreePowerRowIdentities`,
  `OCTREE_PRESSURE_ROW_ONE_RING_DIRECTION_COUNT`, `OctreePowerRowIdentity/DeltaOracle`;
  ~25 more exports have no external importer and should drop to module-private.

### 1.5 Existing plugin seams (build on these, don't invent new ones)

Already registry-driven and healthy:

- **`SimulationMethod` / `GPUSolverInstance`** (`lib/methods/types.ts`) — identity, param
  schema, quality presets, `supportedFieldModes`, `resource` plugin, solver factory.
- **`ResourcePluginDefinition`** (`lib/resource-readiness.ts:27`) — capability-lane
  readiness protocol; declarations already colocated with owners (methods, SVO scene,
  renderer platform).
- **`lib/fluid-pipelines.ts`** — lazy `methodId → dynamic import` registry for pipeline
  graphs (currently only `uniform`).
- **Renderer render-source seam** — the renderer never branches on `methodId` for field
  selection; it reads optional `GPUSolverInstance` fields (`volumeTexture`,
  `sparseVoxelSceneSource`, `globalFineLevelSetSource`, …) with fallbacks, and builds a
  `WebGPULiveSvoScene` **sidecar** when a solver publishes no sparse scene
  (`lib/webgpu-renderer.ts:2069-2081`). This is the cleanest seam in the codebase; the
  uniform method already gets full scene rendering through it with zero renderer code.
- **Harness diagnostic packs/hooks** — string-id registries in
  `lib/scene-diagnostic-pack-implementations.ts:572` and
  `lib/scene-custom-diagnostic-implementations.ts:760`.

Where colocation is violated today (the work of §4): `GridOverlayMode` splices an
octree-owned union into the renderer (`lib/webgpu-renderer.ts:223`);
`lib/octree-technique-debug.ts` contributes ~19 octree-only catalog entries;
`components/PerformanceDials.tsx` imports `octree-runtime-dials` directly;
`DiagnosticsPanel`/`MethodPanel`/`PerformancePanel`/`FluidFieldFlyout`/`TransportBar`
carry ~30 literal `methodId === "octree"` / backend branches (full inventory in §9.3);
`GPUSolverInstance` carries ~25 QA-only per-lane debug fields; `methods/index.ts:7`
re-exports octree-private `octree-coarse-backend` from the registry barrel;
`tools/webgpu-smoke-executor.ts` (7,172 lines) imports methods directly and reaches into
12 octree/losasso/power internal modules.

### 1.6 Packaging reality

Single package, no workspaces, one tsconfig (`@/*` alias). `lib/` is flat and 100%
relative-import (1,407 edges); `tools/` + `tests/` hold 1,826 `../lib/…` imports;
`components/` already uses `@/`. **~140 of 407 test files assert on raw source text via
`readFileSync`** (plus `tools/audit-octree-production-source.ts:118` hardcoding the
filename `webgpu-octree.ts`). Per standing direction these are ceremonial and get
**deleted in Phase 0**, not preserved — especially shader source-text assertions —
which removes what would otherwise be the largest mechanical cost of the refactor. The
CPU suite is already red at HEAD (~124 pre-existing failures) — baseline by diffing
failing *names* against a HEAD archive, not by expecting green.

---

## 2. Target architecture

### 2.1 Physical layout

Folders + tsconfig path aliases, enforced by the boundary invariant test. **This is the
end state** — no npm workspaces (decided 2026-08-12). The alias step is 95% of the
value at 20% of the churn, and `components/` needs almost no edits.

```
lib/
  core/                     # @fluid/core — truly general-purpose only
    methods/                #   SimulationMethod contract, registry, resolveMethodValues
    simulation/             #   controller, gpu-clock, recording
    render/                 #   webgpu-renderer, water pipeline, SVO stack, grid overlay,
                            #   visualization registry/catalog core, resource-readiness
    scene/                  #   model, scenes, scene-lattice, terrain, rigid-body, scenery
    gpu/                    #   pass-broker, compilation manager, device limits, radix sort,
                            #   exact reduction, gpu-initialization, performance-trace/activity
  methods/
    uniform/                # @fluid/method-uniform  (solver + wgsl + pipeline graph + harness plugin)
    octree-shared/          # @fluid/octree-shared — shared by losasso+power ONLY, not core
                            #   (topology engine, planners, owner pages, fine-levelset family,
                            #    pipelined MGPCG, worksets, surface-state [ex quadtree-builder],
                            #    octree-coarse-backend policy, projection-core WGSL)
    losasso/                # @fluid/method-losasso  (41 lane files + lane strategy + lane WGSL
                            #   + adaptive-velocity visualization + dials + harness plugin)
    power/                  # @fluid/method-power    (~30 lane files + generated catalog +
                            #   power visualizations + harness plugin)
  harness/                  # @fluid/harness — smoke catalog, diagnostic packs/hooks, evidence
                            #   (node-only; imports method harness plugins, never vice versa)
```

Dependency rules, enforced (dependency-cruiser or a repo invariant test that walks
imports — this repo's culture is invariant tests, use one):

1. `core` imports nothing outside `core`.
2. `methods/*` import `core` and (losasso/power only) `octree-shared`. **Never each
   other.**
3. `octree-shared` imports only `core`. It exists because losasso and power genuinely
   share an adaptive-topology engine that is *not* general-purpose; putting it in core
   would violate "only truly general purpose code is shared".
4. `harness` may import method harness plugins via the registry; nothing imports
   `harness` except `tools/` and `tests/`.
5. `components/`, `app/`, `worker/` import `core` only — method specifics arrive via
   registered plugin data, never via direct imports.

### 2.2 Three registered methods

Promote the backend switch to the registry: `uniform`, `losasso`, `power-liquids` as
three `SimulationMethod` entries. `lib/methods/octree.ts` splits into two thin method
modules over `octree-shared`; `coarseBackend` dies as a param. Compatibility shims:

- URL state: map legacy `method=octree` + `param.octree.coarseBackend=power2017` →
  `method=power-liquids`, else → `losasso`, during hydration (`lib/url-state.ts`); keep
  writing only new ids.
- Scene catalog: rewrite authored `MethodProfile`s (`lib/scenes.ts:58-116,529-547,96`)
  to the new ids; `frozenPowerReferenceOverrides`
  (`lib/scene-webgpu-smoke-catalog.ts:384`) becomes the power method's profile.
- Safe-mode allowlist keys `param.octree.*` (`lib/gpu-startup.ts:67-82`) → new ids.
- Saved scenes / recordings with old profiles: hydrate through the same alias map.

**Power-liquids param surface (decided 2026-08-12): it only needs default factor-4.**
The method exposes no pinned params in the UI — today's forced tuple
(`globalFineLevelSetFactor=4`, `topologyCadenceAdvances=1`, and the frozen-reference
overrides) become constants inside `methods/power/`'s `createSolver`, `params` shrinks
to whatever genuinely remains user-tunable (possibly empty), and
`normalizeOctreeMethodValues`'s pinning logic is deleted along with the
`coarseBackend`-dependent disable branch in `MethodPanel.tsx:16-24`. Losasso keeps its
real param surface (velocity extension, cadence, leaf size, band cells).

### 2.3 The plugin contract (extend `SimulationMethod`, shrink `GPUSolverInstance`)

`GPUSolverInstance` keeps the universal ~10 members (`info`, `volumeTexture`,
`advanceTo`, `readStats`, `destroy`, `applySceneUniforms`, `reseed`,
`applyRuntimeValues`, `stageSceneUpdate`, `encodeSceneMaintenance`) plus the *render
source* optionals the renderer genuinely consumes (`surfaceFieldTexture`,
`sparseVoxelSceneSource`, `globalFineLevelSetSource`, `coarseLevelSetSource`,
`secondaryParticles`, rigid buffers/pickers). The ~25 QA-only per-lane fields
(`powerPressureBuffer`, `losassoVelocityDebug`, `globalFine*Control`,
`ownerLatticeDebug`, …) move to one opaque `debug?: Record<string, unknown>` bag that
only the owning method's harness plugin knows how to read. This shrinks the core
contract ~60% and removes every lane name from `core`.

`SimulationMethod` gains (all optional, all data-or-lazy-import — keep the UI generic):

```ts
interface SimulationMethod {
  // existing: id, labels, params, presetFor, normalizeValues, resource,
  //           supportedFieldModes, createSolver(Async), runtimeParamKeys ...
  capabilities?: {
    volumeRendering?: boolean;        // replaces methodId === "octree" in FluidFieldFlyout/PerformancePanel
    fencedInitialPresentation?: boolean; // replaces gpu-t0-presentation.ts:33 + TransportBar/renderer gates
  };
  visualizations?: readonly VisualizationFieldEntry[];  // method-contributed catalog entries
                                                        // (octree technique-debug entries move here;
                                                        //  dense-grid structure/cfl/speed/phi stay core)
  overlayPipelines?: Record<string, OverlayPipelineFactory>; // technique overlay/audit; renderer
                                                        // instantiates by registration, not import
  pipelineGraph?: () => Promise<FluidPipelineGraph>;    // absorbs lib/fluid-pipelines.ts
  dials?: readonly RuntimeDialSpec[];                   // absorbs PerformanceDials' direct import
  diagnosticRows?: (info: GPUEulerianInfo, values: MethodParamValues) => DiagnosticRow[];
                                                        // absorbs DiagnosticsPanel/paper-pipeline branches
  harness?: () => Promise<MethodHarnessPlugin>;         // node-only: readbacks, env overrides,
                                                        // debug-bag decoders, oracle metrics
}
```

`GridOverlayMode` becomes an open string type in core; the mode *catalog* is the union
of core entries + registered method entries, so `url-state`'s inlined union
(`lib/url-state.ts:462`) reads from the catalog instead.

`GPUEulerianInfo` (`lib/webgpu-eulerian.ts`) is the other shared ABI to slim: keep the
universal counters, move the ~120 `power*/globalFine*/structured*/quadtree*` optionals
behind per-method info extensions (typed in the method package, surfaced to panels via
`diagnosticRows`). `gridKind` keeps `"uniform" | "octree"`; drop
`"restricted-tall-cell" | "quadtree-tall-cell"`.

### 2.4 The octree split: core engine + lane strategies

Inside `octree-shared`, `WebGPUOctreeProjection` becomes an `OctreeTopologyEngine`
owning: topology/frontier buffers, owner pages, candidate graph
(`encodeInactiveTopologyCandidate`), frontier rows (`encodeFrontierRows`), allocation
planners, fine-band surface step, pipeline plumbing/caches, `destroy` of core resources.
Each lane implements:

```ts
interface OctreeCoarseDynamicsLane {
  initialize(engine: OctreeTopologyEngine): GPUInitializationTask[]; // ex initializeLosassoAuthority /
                                                                    //    initializeNativePowerAuthority
  encodeAdvance(encoder: GPUCommandEncoder, ctx: OctreeAdvanceContext): void; // ex encodeLosasso /
                                                                    //    power assembly + structured projection
  topologyFlipHooks / surfaceStepHooks;   // the ~25 scattered backend branches become
                                          // explicit hook points on the shared paths
  wgsl: OctreeLaneShaderFragments;        // see below
  debugSources(): Record<string, unknown>;
  destroy(): void;
}
```

**WGSL inversion is the prerequisite** (Phase 3): break the 3,800-line template into
composed snippet modules (the repo already does this well — `fineLevelSetPackedSampleWGSL`,
`octreeCompensatedF32WGSL`): `octree-projection-core.wgsl.ts` (bindings, owner arena,
frontier ABI, structural delta, split/refine/balance/frontier/row-delta kernels) plus
lane fragment slots for coarse-phi authority (`losassoArenaLookup`/`losassoAdaptivePhi`
vs Power corrected directory), fine-leaf summary (`adaptiveLosassoLeafSummary` vs
`fineLeafSummary`), and the two Power-only clauses that
`octreeLosassoSurfaceGradingShader` currently splices out of `pressureRefinementEvidence`
and `repairPaperMixedNeighbors`. Each lane assembles its own shader from core + its
fragments; **delete the `.replace()` transform and the module-eval env mutation** (the
grading-fixpoint experiment becomes a construction option or dies — it is exercised by
one test). Binding 15's documented dual-ABI (bulk worklist vs coarse-phi directory vs
Losasso arena) stays a core "lane scratch" binding with per-lane bind groups.

### 2.5 Shape-lab splits out

Decided 2026-08-12: shape-lab leaves the fluid tree during the Phase 2 moves. It is a
CPU-only art-direction route with zero GPU and zero method coupling: `lib/shape-lab/*`
(pool, specimens, trace, worker), `components/ShapeLab.tsx` + `shape-lab-styles.ts`,
`app/shape-lab/page.tsx`. Target: a top-level `shape-lab/` module (route file stays a
thin re-export under `app/`). Boundary rules: only the `app/shape-lab` route imports
it; it may import `core/scene` (it feeds results back through
`lib/hero-garden-overrides.ts`, which stays in core/scene as the consumer-side seam).
Nothing in `core` or `methods/*` may import shape-lab.

Watch items called out by the audit: the shared `info` telemetry blob (both lanes write
it — becomes engine-owned with lane contributions), the device-keyed pipeline caches
(key already embeds `coarseBackend`; keep shared in `octree-shared`), the Power catalog
lazy singleton (moves to `methods/power/`), `octree-runtime-dials.ts` (Losasso-named but
imported by `methods/octree.ts` and pipelined-mgpcg — split: tuning types to
`octree-shared`, Losasso dial specs to `methods/losasso/`).

---

## 3. Phase plan

Each phase lands independently with gates green (per the baseline discipline in §5).
Moves and edits go in **separate commits** (reviewability + the source-text tests).

### Phase 0 — Guardrails, deletions, decisions (no moves)

1. Add the import-boundary invariant test (initially permissive, tightened per phase).
2. **Delete the ~140 source-text (`readFileSync`) tests** — they are ceremonial, and we
   avoid unit tests, particularly for shaders. The narrow exception: where one guards a
   real invariant (e.g. a binding-budget or reachability property currently asserted by
   text-matching WGSL), re-express it as a behavioral check — compile the shader and
   assert entry points/bindings/limits — rather than keeping any text match. Update
   `tools/audit-octree-production-source.ts:118` (hardcoded filename) or delete the
   audit with them. With these gone, file moves cost only import rewrites.
3. Delete the non-survivors (see §9.1 for the full disposition table):
   - `cpuReferenceMethod` + `lib/eulerian-solver.ts` + the ~11 `backend ===
     "cpu-reference"` controller branches + `SimulationBackend` cpu arm
     (`webgpu-renderer.ts:109,2739`) + the executor's CPU-oracle lane
     (`tools/webgpu-smoke-executor.ts:7055-7159`, `FLUID_CPU_ORACLE`).
   - Dead tall-cell/quadtree remnants: `WebGPUSmokeMethodId`'s `"tall-cell" |
     "quadtree-tall-cell"` ids and the executor's dead branches
     (`:3078-3094`, single-tall-cell probe remap), `cpu.worker.quadtree.*` activity ids,
     the `TallCellLayout/chooseTallCellBase` half of `tall-cell-grid.ts` **after**
     relocating `GPUQuality`, `initialLiquidPhi`, `representedInitialPhiVolumeCells` to
     core (verify the probe layouts' last importers die with the executor branches).
   - Zero-importer files: `lib/webgpu-octree-adaptive-nodes.ts`,
     `lib/wgsl-entrypoint-module.ts`, `lib/simulation/gpu-loads.ts`,
     `lib/gpu-advance-pacing.ts`, `lib/validation.ts` (+ its one test).
   - Dead exports in `webgpu-octree.ts` (§1.4 list).
   - **`webgpu-octree-persistent-mgpcg{,.wgsl}.ts`** (2,897 lines, test-only; decided
     2026-08-12: delete) + its two tests. Follow-ups in the same commit: correct
     `octree-coarse-backend.ts`'s `pressureExecutor: "persistent-power-mgpcg"`
     advertisement to the pipelined executor the runtime actually constructs, and
     re-point `power2017-method-selection.test.ts:37`. `isOctreePersistentMGPCGSolverLabel`
     lives in `webgpu-octree-section43-contract.ts` (not the deleted file) — audit
     whether the "POWER + SECTION 4.3" label paths in MethodPanel/DiagnosticsPanel still
     fire with the pipelined executor's solver label, and fix the label source if not.
   Hero-garden/scenery and the SVO render stack are **not** methods and are kept in
   core/render regardless; shape-lab is split out in Phase 2 (§2.5).

### Phase 1 — Cut the SCC, shrink the contracts (edits, no moves)

1. Cut edge (1): move `GPUStatus`/`EffectiveRendererStatus` (type-only) out of
   `webgpu-renderer.ts` into a core status module; `resource-readiness.ts` imports that.
2. Cut edge (2): renderer stops importing `./methods`; the registry (or resolved
   `SimulationMethod`) is injected at construction from the app shell/controller side.
   Also delete the `methods/index.ts:7` barrel re-export of `octree-coarse-backend`
   (its consumers import it directly until Phase 4 relocates it).
3. Shrink `GPUSolverInstance` to universal + render sources + `debug` bag (§2.3); move
   the QA readback call sites in `tools/webgpu-smoke-executor.ts` onto the bag.
4. Slim `GPUEulerianInfo` per §2.3 (mechanical: the panels that read the moved fields
   switch to `diagnosticRows` in Phase 5; until then the method packages re-export their
   info extension types).
5. Kill the label-regex lane sniffing in `resource-readiness.ts:85-89` by requiring
   `status.resource` from all producers.
6. Gate: partition measurement re-run (the reach sets must now differ); CPU suite diff
   clean; one GPU lane per method family (§5).

### Phase 2 — Physical layout + uniform first

1. Introduce tsconfig aliases (`@fluid/core/*`, `@fluid/method-*/*`, `@fluid/harness/*`)
   and the folder skeleton; boundary test learns the rules of §2.1.
2. Move the 83-file core set and renderer/scene set into `lib/core/…` (one mechanical
   commit; import rewrites only, since the source-text tests are gone).
3. Move uniform's 9 files (8 exclusive + `fluid-pipeline.ts` staying core as the
   framework, `UNIFORM_FLUID_PIPELINE` moving with the method) into
   `lib/methods/uniform/`; absorb `lib/fluid-pipelines.ts` into
   `SimulationMethod.pipelineGraph`. Uniform is the smallest and proves the pattern;
   `uniformPipelineFacts`/`uniformTargetCells` leave `webgpu-eulerian.ts` (the latter is
   dead — delete).
4. Split out shape-lab per §2.5 (same mechanical-move commit style).
5. Gate: uniform lanes (`test:webgpu:uniform-mini-dam-32-ceiling`,
   `uniform-hydrostatic-16ms`, `uniform-mini-dam-64-one-step`) + app boot on default
   scene.

### Phase 3 — WGSL inversion + planner extraction (inside `webgpu-octree.ts`, no behavior change)

1. Extract lines 168–1686 (types/ABIs/planners/env gates) into `octree-shared` modules;
   the 35 test files importing real symbols follow via import rewrites.
2. Invert the shader: core snippet modules + lane fragments per §2.4; delete
   `octreeLosassoSurfaceGradingShader` and the module-eval env branch. Aim for
   byte-identical assembled WGSL first, verified by a **throwaway scratch script**
   comparing new composition output against the old transform output (not a retained
   test — we don't keep shader unit tests); where composition intentionally diverges,
   fall back to the identity A/B discipline of §5.
3. Extract the host bootstrap rasterizers and diagnostic overlay shader.
4. Gate: WGSL compile validation (`tools/validate-water-shaders.ts` path), octree
   runtime lanes both backends. The `symmetric-expansion` source-text assertions are
   already gone (Phase 0).

### Phase 4 — Split the god class; lane strategies

1. Carve `WebGPUOctreeProjection` per the §2.4 interface: shared engine in
   `octree-shared`; `initializeLosassoAuthority` + `encodeLosasso` + Losasso getters
   (~1,600 class lines) into `methods/losasso/`; `initializeNativePowerAuthority` +
   power encode/readbacks (~1,300 class lines) into `methods/power/`. The ~25 scattered
   backend branches become named hooks; the constructor threads one lane object through
   its nine allocation stages instead of branching inline.
2. Move the 41 losasso files and ~30 power files (+ generated catalog + its npm
   scripts) into their method folders; `webgpu-octree-eulerian.ts` splits its lane
   branches (`:1216,1344,1992-2010,2365,2459,2573`) into the lane strategies and the
   remainder becomes the `octree-shared` solver shell both methods instantiate.
3. Rename `webgpu-quadtree-builder.ts` → `octree-shared/surface-state.ts` (or similar).
4. Gate: full octree lane matrix (§5) both backends; power catalog
   `verify:octree-power-catalog`.

### Phase 5 — Registry promotion + UI plugins

1. Three method ids (§2.2) with URL/scene/safe-mode compat shims; delete
   `coarseBackend` param; `octree-coarse-backend.ts` policies fold into the two lane
   packages.
2. Power-liquids drops its pinned param surface per §2.2 — the tuple becomes solver
   constants, `MethodPanel` renders whatever little remains from the schema alone.
3. Implement `capabilities`, `visualizations`, `overlayPipelines`, `dials`,
   `diagnosticRows`; burn down the component branch inventory (§9.3) to zero literal
   method names in `components/` (copy strings like "POWER LIQUIDS OBSERVATORY" become
   method-supplied copy). `gpu-t0-presentation.ts:33` and `TransportBar.tsx:50` read
   `capabilities.fencedInitialPresentation`.
4. `paper-pipeline-diagnostics.ts` and `viewport-failure-diagnostics.ts` become
   method-registered diagnostics.
5. Gate: `method-ui`, `url-state` (updated for new default/ids), `visuals-panel`,
   `performance-observatory-ui` tests; manual pass over both popovers (the FIELD flyout
   is the method switch surface).

### Phase 6 — Harness plugins

1. `MethodHarnessPlugin` (node-only, lazy): env-override parsing (`FLUID_UNIFORM_*`,
   `FLUID_LOSASSO_*`, `FLUID_POWER_*`, `FLUID_COARSE_BACKEND` compat), debug-bag
   readbacks, per-method terminal collectors, oracle metrics. The executor's ~60
   `method.id ===` branches and 12 deep octree imports move behind it; the executor
   shrinks to the generic run loop + scenario/evidence plumbing.
2. The smoke catalog keeps method ids as *data* (`methods.<id>.info.*` metric selectors
   are fine — they're strings in a DSL, not imports); lane entries update to the three
   new ids; `WebGPUSmokeMethodId` becomes the registry's id union.
3. Move `lib/scene-webgpu-smoke*.ts`, diagnostic packs/hooks, evidence collectors into
   `lib/harness/`; method-specific packs (octree-authority, structured-power,
   global-fine-publication) register from method harness plugins.
4. Gate: full smoke lane matrix + `npm run test:unit` name-diff clean.

### Phase 7 — Sweep

Boundary test goes strict (rules 1–5 hard-fail); delete compat shims once URLs/scenes
are migrated; final dead-code pass (imports-of-imports orphaned by the moves); update
CLAUDE-adjacent docs and the README architecture section.

---

## 4. What deliberately stays shared (and why)

- **SVO render stack + `webgpu-svo-dry-scene.ts` (10.8k lines)**: production render
  lane (constructed at `webgpu-renderer.ts:1248`, fed via the sidecar seam), verified
  method-agnostic. It is the "one pipeline: high-res voxels" destination — core/render.
- **Water pipeline**: shared three-lane surface consumer (dense volume / global-fine /
  coarse). Its imports of `octree-consumer-sampling` and the power coarse-levelset
  sample ABI are the *renderer-facing consumer contracts* — those two modules move to
  core as the neutral "level-set consumer ABI" (they are consumed by core render code;
  the octree lanes implement them).
- **`webgpu-pass-broker`, compilation manager, radix sort, exact reduction, device
  limits, performance trace/activity, gpu-initialization, resource-readiness**: zero
  method knowledge, high fan-in — core/gpu.
- **Hero garden / scenery / shape-lab**: scenes and a CPU route, not methods. Out of
  scope except for path moves.

---

## 5. Verification discipline (applies to every phase)

- **CPU suite**: red at HEAD (~124 failures). Before each phase: archive failing test
  *names*; after: diff names, not counts. Never chase pre-existing reds.
- **GPU identity A/Bs**: for refactor steps that must not change simulation output, run
  the lane with `FLUID_AWAIT_EVERY_STEPS=1` (mid-run checkpoints lie without it) and
  compare checkpoints old-vs-new. Byte-hash comparison is invalid across any change that
  can reassociate floats (loop restructuring, shader recomposition) — use
  checkpoint-envelope comparison there.
- **Minimum lane matrix per phase**: `test:webgpu` mini dam (uniform default) ·
  `uniform-mini-dam-32-ceiling` · `dam-octree` (losasso) ·
  `minimal-power-dam-32-runtime` + `power-hybrid-deep-ocean` (power) ·
  `svo-dry-render` (renderer untouched proof). Full matrix at Phases 4 and 6.
- **Worktree rules**: no stash/checkout/reset for baselines — get baselines from `git
  show HEAD:path` into scratch, or archive artifacts before starting.
- Land the current uncommitted uniform mass-conservation work (or park it deliberately)
  **before Phase 0**; this refactor must start from a committed tree.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Deleting source-text tests also deletes the few real invariants they guarded | Phase 0 triage: re-express real invariants behaviorally (compile + assert), delete the rest outright |
| WGSL recomposition silently changes shaders | Throwaway assemble-and-compare scratch check against the old transform output; identity A/Bs after |
| God-class split destabilizes one lane | Phase 3 (no behavior change) fully gated before Phase 4; lanes carved one at a time (losasso first — its TS is already one-way isolated) |
| Registry promotion breaks URLs/saved scenes | Hydration alias map, kept through Phase 7; `url-state` tests updated same commit |
| `octree-runtime-dials` straddle | Split types/specs explicitly (§2.4); it is the only module the audit found genuinely straddling the losasso/shared boundary |
| Hidden importer of a "dead" file (worker-URL pattern) | `webgpu-render-worker.ts` precedent: grep for `new URL(` and dynamic `import(` before every deletion |
| Binding-budget regressions from split bind groups | Keep bind group 0 layout ABIs in `octree-shared`; `webgpu-octree-projection-binding-budget.test.ts` stays authoritative |

## 7. Suggested order of first PRs

1. Phase 0.2 source-text test deletion sweep (pure mechanics, huge de-risk for every
   later move).
2. Phase 0.3 deletions (cpu-reference first — high confidence, medium blast radius).
3. Phase 1 SCC cuts + `GPUSolverInstance` debug bag.
4. Phase 2 uniform extraction (pattern-prover).
Then re-plan pacing for Phases 3–6 with fresh measurements.

---

## 8. Decisions (resolved by Peter, 2026-08-12)

1. `webgpu-octree-persistent-mgpcg` — **DELETE** (Phase 0.3, with the policy-label
   follow-ups noted there).
2. Packaging — **enforced folder boundaries are the end state**; no npm workspaces.
3. `power-liquids` param surface — **default factor-4 only**; pinned params leave the
   UI, tuple becomes solver constants (§2.2).
4. Shape-lab — **split out** (§2.5, Phase 2).

## 9. Appendices

### 9.1 Disposition table (summary; per-file lists live in the audit reports)

| Disposition | Contents |
|---|---|
| DELETE (Phase 0) | cpu-reference method + `eulerian-solver.ts` + branches; tall-cell/quadtree dead ids + layout half of `tall-cell-grid.ts`; `webgpu-octree-adaptive-nodes.ts`; `wgsl-entrypoint-module.ts`; `simulation/gpu-loads.ts`; `gpu-advance-pacing.ts`; `validation.ts`; 5 dead `webgpu-octree.ts` exports; `uniformTargetCells`; `webgpu-octree-persistent-mgpcg{,.wgsl}.ts` + 2 tests (policy-label follow-ups per Phase 0.3); the ~140 source-text tests |
| core | contracts, controller, renderer + SVO stack + water pipeline, scenes/model/scenery, resource-readiness, gpu utilities, level-set consumer ABIs, dense-grid visualizations, `GPUQuality` + `initialLiquidPhi` (ex tall-cell-grid) |
| methods/uniform | 8 exclusive files + pipeline graph + harness plugin + 5 exclusive tests + `probe-uniform-collapse` |
| methods/octree-shared | topology engine (ex god class core), planners/ABIs, owner pages, pipelined MGPCG, fine-levelset family, worksets, topology-epoch, surface-state (ex quadtree-builder), coarse/fine exchanges, projection-core WGSL, solver shell (ex octree-eulerian core) |
| methods/losasso | 41 lane files, lane strategy (ex ~1,600 class lines), lane WGSL fragments, adaptive-velocity visualization, dials, 18 exclusive tests, losasso probes/benchmarks |
| methods/power | ~30 lane files, generated catalog (+ generator/check scripts), lane strategy, power visualizations (power-cells/faces/operator/…), ~46 tests, ~16 power tools |
| harness | smoke contracts/catalog, diagnostic packs/hooks, evidence collectors, executor (slimmed), readbacks |
| split out (Phase 2) | shape-lab → top-level `shape-lab/` module (§2.5) |
| keep, out of scope | hero garden/scenery, editor, stores, recording |

### 9.2 Numbers worth keeping in head

Uniform 8 files / 4.2k · Losasso 41 / 17.3k exclusive + ~400 refs inside
`webgpu-octree.ts` (incl. inlined WGSL) + 126 in `octree-eulerian` · Power ~30 / 22.9k +
14.2 MB catalog · SVO stack 62 / 43.5k, method-agnostic · shared-core partition 83 files
· octree-private 130 · renderer-only 59 · one production importer of `webgpu-octree.ts`
· 2 cycle edges · ~25 backend branch sites in the god class · ~140 source-text tests ·
1,826 `../lib` imports in tools+tests · CPU suite ~124 pre-existing reds.

### 9.3 Component method-branch inventory (burn-down list for Phase 5)

`FluidFieldFlyout.tsx:46,68,101` (volumeCapable) · `MethodPanel.tsx:9,16-24,94-96`
(power pin + PCG label) · `PerformancePanel.tsx:47,210,251,301,353` ·
`PerformanceDials.tsx:4-10,73,84-91,100,117-120,144` · `DiagnosticsPanel.tsx:13,55-67,
89-131,157-173` · `TransportBar.tsx:50` · `VisualsPanel.tsx:7,54` ·
`WebGPUViewport.tsx:26,84-86,297,872,1479` · `FluidCellTraceHud.tsx:75` (copy) ·
`VisualPanel.tsx:446,673,679` (copy). Lib-side: `gpu-t0-presentation.ts:33`,
`gpu-startup.ts:21,67-82`, `simulation/controller.ts:207`, `webgpu-renderer.ts:2087,
2784-2788`, `paper-pipeline-diagnostics.ts` (19), `viewport-failure-diagnostics.ts:58`,
`webgpu-grid-overlay.ts` (12), `url-state.ts:4,462`, `physics-step-program.ts:112-120`,
`performance-trace.ts:29` (`power-topology` stage id — becomes a method-registered stage).
