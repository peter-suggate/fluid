# WYSIWYG Scene Editor — Implementation Plan (Agent Handoff)

Status: **Phase 2 landed** (editor shell). Phases 0/1 deliberately deferred — see
§7. Phases 3–6 not started.
Audience: implementing agents. Each phase is independently executable; phases list their
dependencies. All file:line references verified against the working tree as of 2026-07-26
(branch `main`, post fine-band/structured-cutover work).

## 1. Goal

Turn the web UI into a full WYSIWYG world editor with extreme iteration speed:

- Define scene geometry by direct manipulation (the world model is analytic/SDF + voxel lattice).
- Define terrain regions and interactively raise/lower them.
- Define and modify fluid enclosures, water bodies, and hoses (inflow jets).
- Place rigid bodies intuitively (point-and-click, drag, gizmos).
- Forms exist only behind progressive disclosure for precision entry; the primary interface
  is direct manipulation in the viewport.
- **Every edit resets the simulation to t=0 and it is immediately runnable. Target: a
  committed edit is simulating again in well under ~300 ms, ideally ~100 ms. The user makes
  many consecutive edits with no perceptible wait.**

The rendered world is the SVO raymarched scene (`SparseVoxelDrySceneRenderer`); arbitrary
fluid bodies can be added/modified and simulated by the octree/power solver.

## 2. Current architecture — facts the plan is built on

### 2.1 Scene model (already editor-friendly)

- **Single schema, plain JSON**: `SceneDescription` in `lib/model.ts:37` (schemaVersion
  "1.0.0"). Serialization/validation already exist: `serializeScene` (`lib/model.ts:169`),
  `parseScene` (`:173`), `validateScene` (`:184` — the complete authoring constraint list),
  `canonicalScene` (`:158`), `cloneScene` (`:154`).
- The world model is **closed and analytic** — no baked volumes, no meshes:
  - Container: axis-aligned box + open/closed top + wall slip mode (implicit walls; not objects).
  - Terrain: analytic heightfield = `baseHeight_m` + ≤8 elliptical basin/mound features,
    p-norm smooth union (`lib/terrain.ts:15-42`, `MAX_TERRAIN_FEATURES = 8` is a WGSL
    uniform-buffer capacity). Single evaluator `terrainHeightAt` (`terrain.ts:62`); solvers
    consume it **baked to column heights** via `terrainColumnHeights` (`terrain.ts:93`) →
    r32float texture (`lib/webgpu-uniform-eulerian.ts:951`).
  - Initial fluid: `"dam-break" | "tank-fill"` analytic fills + `initialBrickSeeds_m: Vec3[]`
    (each seed wets exactly one 8³ brick; `lib/initial-fluid.ts`). This is the only way to
    author arbitrary water bodies today, and it is already in the solver rebuild key.
  - Inflow: exactly **one** optional `FluidInflow` cylinder jet (`lib/model.ts:117`,
    `lib/inflow-boundary.ts`). It also carves solids (`insideInflowChannel`,
    `lib/webgpu-octree.ts:4463`). No sinks/outlets exist.
  - Rigid bodies: ≤12 (`MAX_BODIES`, `lib/simulation/controller.ts:352`; GPU
    `array<RigidBody, 12>` in `lib/webgpu-octree.ts` ~`:4190`), 4 primitive shapes,
    `motion: "dynamic" | "static"`. Static bodies are the only authored solid obstacles.
  - Environment props (trees, mushrooms, lampposts): **procedural code keyed by
    `EnvironmentId`** (`buildEnvironmentProxyCatalog`, `lib/voxel-environments.ts:357`) —
    render-only, invisible to the solver, and not scene data (not editable).
- Scene catalog is code: `ScenePreset.create()` factories in `lib/scenes.ts:252-421`.
  Parallel headless catalog: `tools/webgpu-smoke-scenarios.ts:11` (newer entries already
  reuse `getScenePreset()` to avoid drift).
- SVO primitive ABI (`lib/svo-primitive-abi.ts:14-95`): sphere/box/capsule/cylinder/
  ellipsoid/terrain-heightfield, 64-byte records, **union-only** (no subtraction — documented
  workarounds at `lib/voxel-environments.ts:161,299`).
- `planVoxelScene` (`lib/voxel-scene.ts:495`) already produces a hashed editor-side model
  with **static-topology vs dynamic-transform separation**
  (`revisions { sceneHash, staticHash, dynamicTopologyHash, dynamicTransformsHash }`).

### 2.2 UI / interaction (good insertion points exist)

- Shell: `components/FluidLab.tsx` — CSS grid: left (ScenePanel + MethodPanel), center
  viewport with overlay chrome, right utility panel (Visual/Bodies/Diagnostics/Performance
  tabs), transport bar.
- All viewport input is one pointer state machine: `pointerRef` discriminated union in
  `components/WebGPUViewport.tsx:48-54`, dispatch in `pointerDown` (`:344-377`).
  **This is where editor tools plug in** — dispatch on an `activeTool` before the existing
  slice-grab → pick → orbit fallback.
- Ray casting exists: `pointerRay` (`WebGPUViewport.tsx:303-308`), `planeHit` (`:309`).
  ⚠ The FOV half-tangent `0.72` is duplicated in `WebGPUViewport.tsx:307` and
  `components/RigidBodyTray.tsx:27` and must be unified with `CAMERA_TAN_HALF_FOV`
  (`lib/webgpu-camera.ts`) before adding more ray tools.
- Picking exists and is production-quality:
  - GPU G-buffer pick: `FluidLabRenderer.pickRigidBody` (`lib/webgpu-renderer.ts:587-616`) →
    `pickGBuffer` (`lib/webgpu-svo-dry-scene.ts:2074`) → 1×1 readback ring
    (`lib/webgpu-svo-picking-readback.ts:96-158`). Returns world position, geometric normal,
    depth, material, owner index. Only available in SVO smooth mode.
  - CPU analytic pick: `pickSvoScene` (`lib/svo-picking.ts:197`) over primitives + terrain +
    glass, with a ready-made WGSL adapter (`svoPickingWGSL`, `:286`). Terrain hits currently
    return `{kind:"none", reason:"terrain"}` deliberately.
  - Terrain ray intersection exists: `intersectSvoTerrainHeightfield`
    (`lib/webgpu-svo-dry-scene.ts:368`).
- Existing gizmo precedent: the grid-overlay slice gripper (`sliceGrabHit`,
  `WebGPUViewport.tsx:317-334`). Drag-to-spawn precedent: `RigidBodyTray.tsx:43-63` +
  `simulation.addBodyAt`.
- State: zustand stores (`lib/stores/*`) hold pure data; the imperative seam is the
  `simulation` singleton (`lib/simulation/controller.ts`) — its doc comment (`:40-46`) states
  the rule: anything that rebuilds runtime state is a controller method.
  `dragBody` (`controller.ts:442` area) already demonstrates the **live-manipulate /
  commit-on-release** split: dragging touches runtime `RigidBodyState` only; descriptions
  change on commit.
- Persistence today: URL query sync (`lib/url-state.ts`, whitelist `sceneQueryPaths`
  `:27-52`) + JSON import in `TransportBar.tsx:83-88` + localStorage load.
  **`terrain`, `environment`, `lighting`, `systems`, `initialBrickSeeds_m`,
  `voxelDomain.bounds_m` are NOT in the URL whitelist.**
- No undo/redo, no transform gizmos, no terrain editing of any kind exists.

### 2.3 Simulation rebuild path (the bottleneck to fix)

- Single decision point: `gpuSceneSolverKey` (`lib/webgpu-renderer.ts:221-223`), compared
  every frame in `currentGPUFluid` (`:963-974`). Any mismatch → transactional full rebuild
  (`beginGPUFluidInitialization`, `:861-960`): construct a brand-new
  `WebGPUUniformEulerianSolver` + `WebGPUOctreeProjection` from zero. **No partial
  invalidation path exists.**
- Known key defects (fix regardless of the editor):
  - `scene.terrain` is **absent** from the key — live terrain edits would not rebuild.
  - `simulationEpoch` is in the key — pressing Reset on an identical scene does a full rebuild.
  - `JSON.stringify(scene.rigidBodies)` is in the key — dropping a sphere = full rebuild.
  - Pure uniform scalars (density, viscosity, surface tension, gravity) are in the key.
- Rebuild cost is dominated by **scene-independent** work rebuilt every time:
  - ~350 uncached, synchronously-compiled compute pipelines across ~25 subsystems created
    inside `initializeNativePowerAuthority` (`lib/webgpu-octree.ts:1870-2166`); ~30 shader
    modules, ~600 KB WGSL re-parsed per build. The AB/BA fine-level-set pairs
    (`lib/webgpu-octree-fine-levelset-topology.ts:574-638` — 47×2 pipelines;
    redistance 25×2; transport 7×2; volume 10×2) compile **identical WGSL twice**.
  - Only ~50 base pipelines are cached (`octreePipelineCache`, `webgpu-octree.ts:144`), and
    that cache key includes `denseSolidField` — adding the first rigid body busts it.
  - 14 MB power catalog (`lib/generated/octree-power-catalog.bin`, device-independent
    constant) is re-fetched, re-decoded, re-memcpy'd, and re-uploaded per build
    (`loadGeneratedOctreePowerCatalog`, `webgpu-octree.ts:124-134`; upload in
    `lib/webgpu-octree-power-topology.ts:266-332`). No memo anywhere.
  - Non-analytic scenes (any body, terrain, or brick seed — i.e. **every editor scene**)
    run a dense CPU level-set seed + exact EDT on the main thread
    (`initialOctreeLevelSet`, `webgpu-octree.ts:4144`; EDT in
    `lib/quadtree-tall-cell-grid.ts:734-775`) plus a dense texture upload. The analytic GPU
    bootstrap (`analyticInitialPhi`, `webgpu-octree.ts:4352`) requires no seeds, no bodies,
    no terrain (`:1036`).
  - 4 fenced warm-up phases (`OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES`,
    `webgpu-octree.ts:94-99`; runner `webgpu-uniform-eulerian.ts:690-760`) + 2 CPU
    readbacks — this is the irreducible "physics setup" and it is small.
- **Warm-path extension points exist but are unused**: `applyRuntimeValues` is an empty
  method (`webgpu-uniform-eulerian.ts:934`); `runtimeParamKeys` (`lib/methods/types.ts:188`)
  is declared by no method; `ParamBase.update: "runtime"` is honored by
  `setMethodParam` (`controller.ts:333`) but set by no octree param.
- **The octree already rebuilds topology incrementally every substep**:
  `encodeInactiveTopologyCandidate` (`webgpu-octree.ts:2331+`) has a `coldFullRebuild` mode
  (full-domain refine/balance from a phi seed against existing allocations) and a recurring
  dirty-tile delta mode. The fine band has mark/rank/scatter + carry + transactional
  rollback (`lib/webgpu-octree-fine-levelset-topology.ts:389`). A "warm reset" is
  **"re-run the cold branch against existing allocations with a new phi seed"**, not
  "build a new solver". See `docs/WEBGPU_OCTREE_M1_MAX_IMPLEMENTATION_PLAN.md` Phase 7.
- Contract tests that constrain any lifecycle refactor (update them deliberately, don't
  fight them): `tests/gpu-initialization.test.ts:68-75` (drain→destroy→create ordering,
  cache presence), `tests/webgpu-renderer-lifecycle.test.ts:106-206` (epoch in key, warm-up
  phase order), `tests/webgpu-power-ui-construction.test.ts`.
- Init profiling harness exists: `tools/run-webgpu-smoke.ts:2646-2657` logs
  `construction_ms` + per-task `solver-initialization` snapshots;
  `tools/run-webgpu-bringup-stage-worker.ts` is the stage-by-stage bring-up harness.
  **No init timings have ever been captured into `artifacts/` — measuring is step zero.**

### 2.4 Other landmines

- Paused-repaint gate: when paused, the viewport skips frames unless
  `presentationStateChanged` sees a change in the `pausedPresentation` tuple of store
  snapshots (`WebGPUViewport.tsx:235-241,271`). **Every new editor state that affects the
  image must be added to that tuple** or the paused viewport won't repaint.
- Safe bring-up mode (`gpu=safe`, `lib/gpu-startup.ts`) hard-fails on unapproved UI state
  and query keys (`safeBrowserGPUBringupViolations`). New editor state/query params must be
  threaded into that allowlist.
- Renderer↔React: two independent rAF loops; the render loop reads stores imperatively via
  `getState()` (`WebGPUViewport.tsx:218-273`) — never re-create the loop from React renders.
- `voxelDomain.brickSize_cells === 4` is renderer-only; fluid requires 8 (`lib/model.ts:211`).

## 3. Design principles

1. **Direct manipulation first.** Every property that has a spatial meaning is edited in
   the viewport (drag handles, brushes, drop targets). `SceneConfigPopover`-style forms
   remain, demoted to progressive disclosure ("precision" flyouts on the selection).
2. **Live-manipulate, commit-on-release.** During a drag/brush stroke, mutate only a
   lightweight preview (renderer-side); commit the `SceneDescription` patch on release.
   This is the existing `dragBody` pattern generalized.
3. **Commit = warm reset, not rebuild.** A committed edit re-seeds the existing solver in
   place (new phi/solids/params against existing pipelines, catalogs, arenas). Full
   rebuild only when the lattice shape or method changes.
4. **The scene document stays the single source of truth** (`SceneDescription`, extended).
   Undo/redo, persistence, and headless reproduction all operate on the document.
5. **Everything the editor authors must round-trip** through `serializeScene`/`parseScene`
   and be reproducible headlessly via the smoke harness.

## 4. Phases

Dependency graph: P0 → P1 → {P2, P3, P4, P5} (parallelizable after P1+P2 land the shared
infra); P6 last. P1 (fast reset) and P2 (editor shell) can proceed in parallel.

---

### Phase 0 — Measure the rebuild (small; do first)

**Goal:** a recorded baseline of solver construction cost so P1 has a target and a
regression guard.

- Add a per-stage init timeline: wrap `WebGPUUniformEulerianSolver.createAsync` task list
  (`webgpu-uniform-eulerian.ts:649-688`) and `WebGPUOctreeProjection` constructor sections
  with named timing marks (planner / brick-world / surface-state / pipelines-per-subsystem /
  catalog-fetch / catalog-upload / warm-up-phase-N). Reuse the existing
  `solver-initialization` record stream (`tools/run-webgpu-smoke.ts:2650-2657`).
- Capture baselines into `artifacts/` for: `minimal-power-dam-break` (analytic),
  `garden-pond` (terrain + bodies, non-analytic), `ocean-seiche` (large grid).
- Also measure the browser path once (DevTools or `performance.mark` around
  `beginGPUFluidInitialization`).

**Acceptance:** a doc section (append to this file) with the measured breakdown, confirming
or correcting the predicted split (pipeline compilation dominant, catalog upload medium,
warm-up fences small).

---

### Phase 1 — Fast reset foundation (the enabler; largest solver-side phase)

**Goal:** committed scene edits apply in ~100 ms. Three sub-workstreams, in order of
leverage:

**1a. Device-scoped caches for scene-independent state.**
- Shader-module + pipeline cache: extend the `octreePipelineCache` pattern
  (`webgpu-octree.ts:144`) into a device-scoped cache used by **all** subsystem
  constructors under `initializeNativePowerAuthority` (list and per-subsystem pipeline
  counts in §2.3 / the init exploration). Key = WGSL source hash + entry point + pipeline
  constants. Convert synchronous `createComputePipeline` calls to `Async` where the
  construction flow allows. Deduplicate the AB/BA twin compiles by sharing one module +
  pipeline set per pair (the two generation instances differ only in bound buffers, not
  WGSL).
- Catalog memo: module-level memo of the decoded catalog in
  `loadGeneratedOctreePowerCatalog` (`webgpu-octree.ts:124-134`); device-scoped GPU
  buffers for the catalog in `WebGPUOctreePowerTopology`
  (`webgpu-octree-power-topology.ts:266-332`) shared across solver instances (they are
  immutable constants — version-keyed by the manifest sha).
- Remove `denseSolidField` from the base pipeline cache key by making the solid-field
  binding unconditional (bind a 1-cell dummy when absent) so adding the first body no
  longer busts the ~50 cached pipelines.

**1b. Tier the solver key** (`gpuSceneSolverKey`, `webgpu-renderer.ts:221`). Split into:
- `structuralKey` — device/method/quality/structural method values/`voxelDomain`/container
  dims/top/wall mode (grid shape + arena capacities): mismatch → full rebuild (current path).
- `seedKey` — terrain (**add it — currently missing**), rigidBodies, initialCondition,
  fillFraction, brick seeds, inflow, environment-affecting-solids: mismatch → **warm reset**
  (1c).
- `uniformKey` — density, viscosity, surface tension, gravity, dt, wall-mode bit: mismatch →
  hot uniform write via `applyRuntimeValues` (implement it — currently a no-op,
  `webgpu-uniform-eulerian.ts:934`), no reset unless the user asked for one.
- Remove `simulationEpoch` from the structural key; an epoch bump with unchanged
  structural+seed keys should also take the warm path (reset-to-t0 = re-seed).
- Update the contract tests deliberately: `tests/webgpu-renderer-lifecycle.test.ts`,
  `tests/gpu-initialization.test.ts`.

**1c. Warm reset (`solver.reseed(scene)`)** — new method on the octree solver:
1. Write new params/uniforms (`writeParams`, `webgpu-octree.ts:2200+`).
2. Refresh scene-derived GPU inputs in place: terrain texture
   (`webgpu-uniform-eulerian.ts:951`), rigid body buffer, solid vertex SDF, inflow params.
3. Produce the new phi seed **on GPU**: extend `analyticInitialPhi`
   (`webgpu-octree.ts:4352`) to compose tank/dam fills **+ brick-seed box SDFs + terrain
   (via the terrain texture) + static body SDFs** so editor scenes never take the CPU
   dense-EDT path (`initialOctreeLevelSet` becomes a fallback/oracle only). The WGSL
   already evaluates terrain and bodies per-cell (`currentSolidAt`, `webgpu-octree.ts:4483`);
   the CPU mirror for brick seeds exists (`initialFluidBrickSignedDistance`,
   `lib/initial-fluid.ts:78`).
4. Re-enter the existing cold path against existing allocations:
   `encodeColdBootstrapRebuild` (`webgpu-octree.ts:2258-2282`) → structured authority →
   `encodeSurface(encoder, 0)` → sparse render world, i.e. the same 4 fenced phases,
   without reallocating anything.
5. Reset host-side clocks/diagnostics (existing `reset()` path in
   `lib/simulation/controller.ts:260-284` minus the solver teardown).
   Capacity guard: if the new seed exceeds a planned arena (e.g. fine-band brick capacity,
   pressure rows), fall back to full rebuild transparently — the planners
   (`planOctreePressureCapacity` etc., `webgpu-octree.ts:568-618`) already compute the
   bounds to check. Keep headroom generous (plans already carry 1.25× growth).
- Beware `lib/octree-structured-reject-carry.ts` freeze-on-reject: a bad re-seed can
  silently no-op the solver — surface topology-epoch rejection reasons in diagnostics
  during development.

**Acceptance:**
- Smoke harness: new `reseed` scenario — construct once, apply N scene edits (add body,
  move terrain feature, add brick seed), assert per-edit wall time vs the Phase 0 full
  baseline (target ≥5× faster; no shader compilation, no catalog upload in the trace) and
  assert first post-reseed steps stay bit-stable vs a fresh-build control on the same
  scene (reuse the fingerprint machinery from the smoke harness / IoU parity bar).
- Existing `test:webgpu:*` suites green.

---

### Phase 2 — Editor shell (tools, selection, gizmos, undo)

**Goal:** the interaction chassis every later phase plugs into.

- **Tool state**: `activeTool` in `lib/stores/ui-store.ts`:
  `"select" | "terrain-raise" | "terrain-lower" | "fluid-paint" | "fluid-erase" |
  "body-place" | "inflow"` (extensible). Toolbar in the viewport (left edge vertical strip,
  consistent with `.utility-panel-tabs` styling in `app/globals.css`). Keyboard: `Q/W/E/R…`
  cycle, `Esc` → select.
- **Pointer dispatch**: in `WebGPUViewport.tsx` `pointerDown` (`:344`), dispatch on
  `activeTool` **before** the slice-grab/pick/orbit fallback; orbit/pan/zoom stay always
  available (RMB-drag orbit — RMB is currently free; `onContextMenu` already prevented).
  Extend the `pointerRef` union (`:48-54`) with tool-specific drag states.
- **Unify ray math**: single `viewportRay(camera, ndc)` in `lib/webgpu-camera.ts` using
  `CAMERA_TAN_HALF_FOV`; replace the duplicated `0.72` in `WebGPUViewport.tsx:307` and
  `RigidBodyTray.tsx:27`.
- **Cursor/world feedback**: use G-buffer pick (`pickGBuffer`) for hover position + normal
  (already returns both); fall back to `pickSvoScene` + `intersectSvoTerrainHeightfield`
  when SVO picking is unavailable. Make terrain hits first-class (today
  `svoPickingInteractionForHit` maps terrain → `none`, `lib/svo-picking.ts:154`).
- **Overlay/gizmo layer**: a small immediate-mode 3D overlay pipeline (lines/discs/handles,
  depth-tested against the G-buffer depth) — model it on `GridOverlayPipeline`
  (`lib/webgpu-grid-overlay.ts`). DOM overlays (like `viewport-failure-marker`,
  `WebGPUViewport.tsx:433-451`) are fine for labels/handles-as-hotspots in v1.
- **Selection**: generalize `selectedBodyId` (ui-store) to
  `selection: {kind: "body"|"terrain-feature"|"inflow"|"solid"|"prop", id}`; a selection
  shows its gizmo + a compact floating "precision" flyout (the progressive-disclosure form:
  numeric fields from `components/controls.tsx`).
- **Undo/redo**: history of `SceneDescription` snapshots in a new
  `lib/stores/history-store.ts` — scenes are small JSON (`cloneScene` already exists);
  coalesce per gesture (one undo entry per drag/brush stroke, not per pointermove).
  `Cmd+Z`/`Shift+Cmd+Z`. Undo application = `simulation.reset(snapshot)` → warm path from P1.
- **Draft/commit plumbing**: a `simulation.beginEdit()/updateEdit(patch)/commitEdit()`
  triple on the controller; `updateEdit` mutates preview-only state (renderer-side), commit
  patches the scene store → seedKey change → warm reset.
- **Bookkeeping**: add all new ui-store fields to the paused-presentation tuple
  (`WebGPUViewport.tsx:235-241`) and the safe-bringup allowlist (`lib/gpu-startup.ts`).

**Acceptance:** select/move a rigid body entirely in-viewport with gizmo + undo/redo;
tool switching; hover feedback on terrain and bodies; all existing interactions
(orbit, slice drag, body drag-force) unchanged.

---

### Phase 3 — Terrain editing (raise/lower)

**Decision: introduce a sampled heightfield variant** alongside analytic features. The
8-feature cap cannot support brush sculpting, but every solver consumer already ingests
terrain as a baked column-height grid (`terrainColumnHeights`), so a grid representation
is the natural authoring target; only the renderers and contact solver evaluate the
analytic form directly.

- **Schema** (`lib/terrain.ts`, `lib/model.ts`): extend `TerrainDescription` with an
  optional grid form, e.g.
  `{ kind: "grid"; origin_m: {x,z}; spacing_m: number; size: {nx,nz}; heights_m: number[] }`
  (keep the existing analytic form valid — presets stay untouched). `terrainHeightAt`
  gains a bilinear-sample branch; `terrainNormalAt`, `terrainColumnHeights`,
  `terrainCellSolidFraction`, `validateTerrain` follow. On first sculpt of an analytic
  terrain, bake features → grid at ~2× finest-cell resolution (one-way, with undo).
- **Consumers to extend for the grid form**:
  - Solvers: already grid-based via `terrainColumnHeights` — no kernel changes.
  - SVO renderer: `SvoTerrainHeightfieldPrimitive` (`lib/svo-primitive-abi.ts:89`) +
    `lib/svo-scene-primitives.ts` currently evaluate features analytically in the ray
    march — add a heights-texture sampling path (bind the same texture the solver uses).
  - Raster renderer terrain packing (`lib/webgpu-renderer.ts:1243-1250`): same texture path.
  - Rigid contacts (`lib/rigid-body.ts:367`): comes free via `terrainHeightAt`.
- **Brush tool**: `terrain-raise`/`terrain-lower` (modifier inverts). Pointer ray →
  `intersectSvoTerrainHeightfield`; apply a smooth radial falloff delta to the grid each
  pointermove; brush radius/strength on scroll-wheel + flyout. Live preview: during the
  stroke, update the terrain texture + renderer only (draft mode); commit on pointerup →
  warm reset. Clamp heights to [0, container.height]; revalidate brick seeds/bodies
  (settle bodies with the existing `settleBodiesOnTerrain` pattern,
  `lib/garden-scene.ts:51`).
- **Feature-handle editing** (cheap, ship first): for analytic terrains, make each
  `TerrainFeature` selectable (disc gizmo at center: drag = move, ring = radius, vertical
  handle = amount, rotation handle) — pure `SceneDescription` edits, no schema work, good
  for validating the P1/P2 loop end-to-end before the grid lands.
- **Solver key/persistence**: add `terrain` to the seed key (P1b does this) and handle
  persistence via P6 (grids are too large for the URL whitelist).

**Acceptance:** sculpt a pond into a flat scene, water fills it after commit
(tank-fill or painted fluid), stroke-commit latency within the P1 budget; garden-pond
preset still renders identically (analytic path untouched); headless smoke can load a
grid-terrain scene.

---

### Phase 4 — Fluid authoring (bodies, enclosures' water, hoses)

- **Fluid paint** (`fluid-paint`/`fluid-erase`): brush adds/removes
  `fluid.initialBrickSeeds_m` entries (8³ bricks — already in the solver key and the
  warm-path phi seed via P1c). Snap ray hits to brick centers; paint on terrain/solid
  surfaces or on a height-locked plane (modifier). Preview: ghost boxes via the overlay
  layer during the stroke; commit on release. `initialBrickSeedsAdditive` semantics: the
  editor always writes additive seeds over the base fill.
- **Fill-level handle**: a horizontal plane gizmo on the container — drag vertically to
  set `container.fillFraction` (tank-fill) live; this is a seed-key change → warm reset on
  release.
- **Hoses = inflow jets**: promote `fluid.inflow` to `fluid.inflows: FluidInflow[]`
  (schema + `validateScene` + WGSL loop in `lib/inflow-boundary.ts` +
  `insideInflowChannel` union + URL whitelist blob). Keep `inflow` parsing for back-compat
  in `parseScene`. Cap small (e.g. 4) for uniform-buffer simplicity.
  - Placement tool: click a surface → nozzle placed at hit point + normal; gizmo: drag body
    = move, arrow = direction+speed (arrow length), ring = radius; timing/ramp in the
    precision flyout. Render nozzles via the existing static-cylinder pattern
    (`lib/paper-scenarios.ts:50-57`) or a dedicated prop.
- **Enclosures (walls/tanks)**: v1 = static solids via Phase 5's `scene.solids` (below) —
  draw a box/wall by click-drag on the ground plane, extrude height with a vertical handle.
  Water stays inside because solids carve the domain. (True custom container shells /
  subtractive CSG are out of scope — the SVO ABI is union-only.)

**Acceptance:** paint an arbitrary elevated water blob, it falls and settles on commit;
place two hoses filling a drawn enclosure; all authored via viewport only; scene
round-trips through JSON and runs headlessly.

---

### Phase 5 — Solids, rigid bodies, and props

- **Static solids as first-class scene data**: add `scene.solids?: SolidPrimitive[]`
  (shape/dimensions/position/orientation, reusing `RigidBodyDescription` geometry fields
  minus dynamics — or keep them as `motion:"static"` rigid bodies if the 12-cap is raised;
  decide at implementation time, but the solver already rasterizes static bodies via
  `bodySolidFraction`, `webgpu-octree.ts:4472`, so the fastest path is: raise the GPU array
  cap (e.g. 32) + split the UI concept). They participate in `currentSolidAt`, the SVO
  render (via `svoPrimitiveForRigidBody`, `lib/svo-primitive-abi.ts:307`), and picking.
- **Placement UX**: extend the existing tray drag-drop (`RigidBodyTray.tsx`) to drop onto
  the **picked surface** (G-buffer hit position + normal) instead of the container-center
  plane; ground-snap + rest on terrain. Post-placement gizmo: translate arrows/plane,
  rotate ring, uniform + per-axis scale handles mapping to `dimensions_m`; duplicate
  (`Alt-drag`), delete key.
- **Live drag without rebuild**: while gizmo-dragging a body, go through
  `simulation.dragBody`-style runtime state (and the existing `rasterizeSolidsDelta`
  incremental path, `webgpu-octree.ts:5010`); commit description on release (seed-key →
  warm reset). This kills today's "editing a slider rebuilds the world" behavior
  (JSON(rigidBodies) in the key) once P1b lands.
- **Environment props → scene data** (render-only, low risk): add
  `scene.props?: PropPrimitive[]` mirroring `EnvironmentProxyPrimitive`
  (`lib/voxel-environments.ts:32-49`: box/cylinder/ellipsoid + material). Environment IDs
  become factory defaults that populate `props` on preset load instead of procedural
  hard-coding (`buildGarden` etc. become data generators). Props are placeable/movable
  with the same gizmos; they never enter the solve (unchanged behavior — they only extend
  the sparse domain via `planSparseSceneDomain`, `lib/sparse-scene-domain.ts:156`).

**Acceptance:** build a small scene from an empty tank — terrain, a wall enclosure, two
mushroom props, a dynamic ball on a ledge — entirely in-viewport; per-edit latency within
budget; props render in SVO with correct materials.

---

### Phase 6 — Persistence, presets, polish

- **Named local scenes**: localStorage/IndexedDB scene library (list/save/rename/delete) +
  file export/import (JSON already works via `TransportBar`). The URL keeps working for
  small scalar diffs; add a `scene=<id>` reference for library scenes rather than growing
  the whitelist to hold terrain grids.
- **URL whitelist**: add `terrain` (analytic form only), `fluid.inflows`, `solids`,
  `props` as atomic blobs where size permits (`lib/url-state.ts:27`).
- **Catalog unification**: route `tools/webgpu-smoke-scenarios.ts` through `getScenePreset`
  for all entries + add a "load scene JSON from path" mode so authored scenes are directly
  reproducible headlessly (`FLUID_SCENE_FILE` env var).
- **Form demotion**: fold `SceneConfigPopover` content into the per-selection precision
  flyouts + a single "Scene" inspector (container dims with viewport resize handles,
  numerics, fluid constants — fluid constants become hot-applied via P1b uniformKey).
- **Editor QoL**: hover highlighting, snapping toggles (grid = brick size), camera focus-on
  -selection (`F`), safe-bringup allowlist finalization, docs.

## 5. Verification strategy

- **Per-phase smoke tests** (headless, Dawn): extend `tools/run-webgpu-smoke.ts` with a
  scripted-edit mode — load scene, apply a JSON patch list, assert reseed (not rebuild) was
  taken, assert step fingerprints match a fresh-build control. Wire into `package.json`
  as `test:webgpu:editor-reseed`.
- **Unit tests**: terrain grid evaluator parity (analytic-baked-to-grid vs analytic within
  tolerance), scene round-trip with new fields (extend the S2-02 byte-identical test,
  `lib/validation.ts:18`), url-state round-trip, undo coalescing, solver key tiering.
- **Interactive verification** (Claude in Chrome or manual): tool-by-tool checklist per
  phase acceptance; record GIFs of the terrain brush and fluid paint loops.
- **Perf regression guard**: Phase 0 baseline numbers re-captured after P1 and kept in
  `artifacts/`; per-edit budget asserted in the editor-reseed smoke.
- **Existing suites**: `npm run test` + the relevant `test:webgpu:*` scenarios after each
  phase; the lifecycle/init contract tests are updated (not deleted) in P1.

## 6. Risks / open questions

- **Warm-reset numerical parity**: re-seeding in place must produce the same trajectories
  as a fresh build (the reseed smoke's fingerprint check is the gate). Watch the
  freeze-on-reject carry (`lib/octree-structured-reject-carry.ts`) and topology-epoch
  rejection.
- **Arena headroom vs. edits**: an edit can exceed planned capacities (paint a huge water
  volume). The transparent fallback-to-full-rebuild must be reliable; consider planning
  arenas from container volume rather than initial fluid volume for editor sessions.
- **Terrain grid memory in the document**: a 320×80 grid is ~100 KB of JSON numbers —
  fine for files/localStorage, not for the URL. Accepted; covered by P6.
- **12-body GPU cap**: raising `array<RigidBody, 12>` touches WGSL + uniform layout +
  `MAX_BODIES`; sizing decision deferred to P5 (32 suggested).
- **Union-only SVO ABI**: no subtractive shapes; hollow enclosures are built from wall
  slabs. Documented limitation, not a blocker.
- **Two catalogs drift** (UI vs smoke): addressed in P6; until then new schema fields must
  be exercised in both.
- **MG candidate hierarchy rebuild is single-threaded per step**
  (`docs/OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md:127-215`) — warm reset inherits its cost at
  t=0; acceptable, but don't lean on the dirty-delta path for reseed until that's fixed.

## 7. Implementation log

### Phase 2 — Editor shell — LANDED (2026-07-26)

**Why this phase first.** Phases 0 and 1 are solver-side and land in
`lib/webgpu-octree.ts`, `lib/webgpu-uniform-eulerian.ts`, and
`lib/webgpu-renderer.ts` — the exact files carrying the in-flight structured-cutover
physics work (235 files, ~31 k insertions in the working tree). Phase 2 is entirely
UI-side and touched no file the physics change set had modified, so it was taken first
per the plan's own note that P1 and P2 parallelize.

**New modules**

- `lib/editor-tools.ts` — `EditorTool` union, per-tool specs (label/shortcut/hint/
  status/phase), shortcut resolution, `EditorSelection`. Tools whose pointer behaviour
  has not landed carry `status: "planned"` plus the phase that implements them, and the
  toolbar renders them disabled rather than accepting dead clicks.
- `lib/editor-gizmo.ts` — translate-gizmo geometry: screen-constant handle length
  (world length scaled by eye depth cancels the perspective divide), pixel-space handle
  hit test with centre priority, and the closest-approach axis constraint. Returns
  `undefined` for near-parallel rays instead of letting a drag diverge.
- `lib/editor-hover.ts` — analytic CPU hover over bodies (bounding sphere), terrain
  (`intersectSvoTerrainHeightfield`), and the container floor. Deliberately not the
  G-buffer pick: that is a fenced 1×1 readback, wrong for a per-pointermove cursor. It
  stays the authority for click-to-select. **Terrain is a first-class hover hit here**,
  which is what §2.2 asked for; `svoPickingInteractionForHit` still maps terrain to
  `{kind:"none"}` because only rigid bodies are draggable — that contract is unchanged.
- `lib/stores/history-store.ts` — undo/redo over whole `SceneDescription` snapshots
  (entries record the document *before* an edit), gesture coalescing by key + time
  window, bounded at 64 entries, deep-cloned on record.
- `lib/use-editor-shortcuts.ts` — `⌘Z`/`⇧⌘Z`/`⌃Y`, `Q`/`W` tool arming, `Esc`,
  `Delete`, `F` focus-on-selection; inert while focus is in a form control.
- `components/EditorToolbar.tsx`, `components/SelectionFlyout.tsx` — left tool strip
  (mirroring `.utility-panel-tabs`) and the per-selection precision flyout, collapsed to
  a chip by default per design principle 1.

**Changed**

- `lib/webgpu-camera.ts` now owns `viewportRay`, `viewportRayForPointer`,
  `projectToViewport`, and `viewportAspect`. The duplicated `0.72` half-tangent is gone
  from `WebGPUViewport.tsx`, `RigidBodyTray.tsx`, and `viewport-failure-diagnostics.ts`
  (the third copy the plan had not recorded).
- `lib/stores/ui-store.ts` — `activeTool`, `selection` (generalizing `selectedBodyId`,
  which is retained as its body-only projection because the renderer and roster address
  bodies directly), `placementShape`.
- `lib/simulation/controller.ts` — `beginEdit`/`commitEdit`/`cancelEdit`, `undo`/`redo`,
  private `recordHistory` (a no-op inside an open gesture, so one gesture is one entry),
  history recording on `addBody`/`addBodyAt`/`removeBody`/`updateBody`/`loadPreset`/
  `importScene`, `manipulateBody` (the `dragBody` kinematic constraint refactored out and
  reused without forcing run state — authoring geometry is an edit, not a throw), and
  `addBodyAt(..., {autoRun:false})` for editor placement.
- `components/WebGPUViewport.tsx` — pointer dispatch order is now gizmo → armed tool →
  slice grab → pick → orbit. Two new `pointerRef` variants (`gizmo-axis`, `gizmo-free`)
  implement live-preview/commit-on-release. SVG gizmo overlay, hover chip, and the
  selection flyout are DOM/SVG overlays, which the plan sanctions for v1.
- `lib/gpu-startup.ts` — `activeTool` added to `SafeBrowserGPUBringupConfig`; any armed
  authoring tool is a safe-mode violation, since an authoring gesture mutates the pinned
  workload mid-session. `tests/gpu-startup.test.ts` updated deliberately.
- No addition to the paused-presentation tuple was needed: `WebGPUViewport.tsx:235-241`
  already carries the whole `ui` store snapshot, so any new ui-store field changes its
  identity and repaints. Gizmo/hover state is React-side and needs no GPU repaint.

**Tests** — `tests/editor-tools.test.ts`, `tests/editor-gizmo.test.ts` (ray/projection
inverse, exact axis recovery, parallel-ray rejection, no-jump grab offset, screen-constant
handles, behind-camera rejection), `tests/editor-history.test.ts`. 75 assertions green
across the new plus affected existing suites.

**Known gaps, deliberately left for later phases**

- A committed edit still takes the existing full-rebuild path
  (`gpuSceneSolverKey` carries `JSON.stringify(scene.rigidBodies)`), so per-edit latency
  is the Phase 0/1 problem, unchanged. Undo/redo applies via `reset()` for correctness.
  This is pre-existing behaviour for body edits, not a Phase 2 regression.
- Clicking a body (rather than a gizmo handle) still starts the physics drag and the
  clock, as before — the plan required existing interactions to be unchanged.
- `body-place` drops at the hovered surface; ground-snap refinement, rotate/scale
  handles, and duplicate/delete gizmo affordances are Phase 5.
