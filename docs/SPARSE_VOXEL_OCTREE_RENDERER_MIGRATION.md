# Sparse voxel octree renderer migration

Status: **planned**
Created: 2026-07-19
Last updated: 2026-07-19

This document is the implementation plan and completion ledger for migrating
Fluid Lab from its hybrid analytic/raster presentation to a direct sparse voxel
octree (SVO) renderer. Update it as work lands. A milestone is complete only
when its implementation tasks, acceptance gates, and evidence entry are all
complete.

## How to use this plan

- Change a milestone from `not started` to `in progress` when work begins.
- Check a task only when its code or test has landed.
- Do not mark a milestone complete from a screenshot or successful build alone.
- Record tests, benchmarks, screenshots, deviations, and design decisions in
  the ledgers at the end of this file.
- Put newly discovered required work into the relevant milestone rather than an
  external TODO list.
- Reopen affected milestones when a later decision invalidates completed work.
- `[ ]` means not implemented or not demonstrated; `[x]` means implemented and
  demonstrated. Mark excluded work `N/A` with a decision-ledger explanation.

## Fixed decisions

Changing one of these requires a decision-ledger entry and review of downstream
milestones.

1. The primary renderer will trace the sparse voxel octree directly.
2. Dual contouring and other production surface-mesh extraction are out of
   scope for the SVO path.
3. The current raster renderer remains user-selectable and is the automatic
   fallback.
4. Raw voxels and brick bounds remain inspection modes, not production modes.
5. Terminal SVO payloads may be heterogeneous: exact implicit primitives,
   sampled smooth SDFs, or feature-aware sampled SDFs.
6. Analytic rigid descriptions remain valid physics and authoring sources. The
   migration does not replace rigid dynamics or collision with voxel physics.
7. Fluid and solid signed-distance fields remain logically separate.
8. Rendering stays GPU-resident. CPU readback must not control a production
   frame's topology, visibility, or allocation decisions.
9. Dynamic publication is generation-safe: rendering sees a complete old or
   complete new generation, never a partial rebuild.
10. Raster and SVO geometry share one material and lighting contract.
11. Simulation behavior and acceptance results do not depend on the selected
    presentation renderer.

## Scope

### In scope

- Direct WebGPU traversal of the sparse-brick octree.
- Smooth fluid rendering from the authoritative level set.
- Exact implicit rendering of spheres, boxes, capsules, cylinders, ellipsoids,
  and terrain heightfields.
- Sampled and feature-aware SDF support for voxel-native solids.
- Dynamic-solid residency, dirty-region updates, and complete generations.
- PBR opaque lighting, SVO shadows, water/glass media, temporal stability, and
  bounded indirect lighting.
- SVO picking with stable owner/material identity.
- Raster compatibility, debug views, diagnostics, benchmarks, and fallback.

### Out of scope

- Replacing the fluid solver, pressure topology, or analytic rigid physics.
- Dual contouring, marching cubes, or marching tetrahedra in SVO mode.
- Deleting the raster renderer.
- Unbounded recursive path tracing.
- A general imported-mesh pipeline unless a shipped scene later requires it.
- Silent allocation overflow, missing pages, or traversal exhaustion.

## Existing foundation

- `SparseBrickOctreeGPU` publishes node, leaf, geometry, velocity,
  material/owner, count, generation, and indirect-work buffers.
- Sparse geometry already stores fluid signed distance, a solid-distance
  estimate, solid fraction, and pressure.
- The octree solver owns an authoritative fluid level set.
- `WebGPUSparseSurfaceBand` supplies dynamically paged fine `phi` samples with
  core/halo residency and complete coarse fallback.
- `GPUFluidBrickResidency` supplies GPU-owned activation, swept support,
  hysteretic retirement, and active/retired worklists.
- `SparseSceneProxyVoxelizer` evaluates box, cylinder, and ellipsoid implicit
  distances for scene proxies.
- Stable material and owner IDs already connect proxies, rigid bodies, and
  inspection rendering.
- The raster path already provides HDR composition, water interfaces,
  Beer-Lambert absorption, Fresnel response, tone mapping, GPU timings, and an
  A/B control.

## Current gaps

1. Production voxel presentation consumes expanded debug records rather than
   binding the structural octree directly.
2. Fluid `phi` is presentation-ready, but dynamic solid distance is often
   reconstructed from solid fraction and is not an exact SDF.
3. Renderer-specific surface refinement and complete dynamic generation
   semantics are not yet a production acceleration structure.
4. Smooth presentation still uses analytic scene intersections and rasterized
   water triangles.
5. The shared lighting closure is color-consistent but not fully PBR.
6. Water optics depend on raster front/back interface buffers.
7. Picking is tied to rigid-system intersections rather than the visible SVO.
8. Temporal history does not understand topology generation or fluid motion.

## Target architecture

```text
Scene authoring and simulation
  |-- static environment catalog
  |-- rigid transforms and primitive descriptions
  |-- terrain heightfield
  |-- coarse fluid phi and velocity
  `-- sparse fine fluid surface pages
                 |
                 v
Sparse scene publication
  |-- static layer
  |-- dynamic-solid layer
  |-- fluid layer
  |-- material/owner identity
  |-- complete generation and dirty worklists
  `-- conservative node/brick summaries
                 |
       +---------+----------+
       |                    |
       v                    v
Direct SVO renderer     Raster compatibility
  |-- primary hits        |-- current analytic scene
  |-- G-buffer            |-- current water extraction
  |-- PBR lighting        |-- shared PBR closure
  |-- shadow/media rays   `-- automatic fallback
  `-- temporal resolve
```

Final user-facing controls are orthogonal:

```ts
type SceneRenderMode = "svo" | "raster";
type InspectionMode = "off" | "raw-voxels" | "brick-grid";
```

Raster optics is the only water presentation path; inspection modes remain
orthogonal debug views rather than alternate optical renderers.

## Data contracts

### Distance convention

- Distances are world metres.
- Negative is inside, zero is the boundary, and positive is outside.
- Fluid and solid fields have separate validity and revision state.
- Sample position is explicit: cell centre, lattice corner, or analytic.
- Every sparse lookup validates both its logical page and physical slot.

### Proposed production render source

The final packing may change, but the semantics must include:

```ts
interface SparseSceneRenderSource {
  control: GPUBufferBinding;
  nodes: GPUBufferBinding;
  leaves: GPUBufferBinding;
  geometry: GPUBufferBinding;
  velocity: GPUBufferBinding;
  materialOwners: GPUBufferBinding;
  brickStates: GPUBufferBinding;
  activeBricks: GPUBufferBinding;
  retiredBricks: GPUBufferBinding;
  materials: GPUBufferBinding;
  primitives: GPUBufferBinding;
  lights: GPUBufferBinding;
  worldOrigin_m: readonly [number, number, number];
  finestCellSize_m: readonly [number, number, number];
  brickSize: 4 | 8;
  maximumDepth: number;
  topologyRevision: number;
  staticRevision: number;
  dynamicRevision: number;
  fluidRevision: number;
  publishedGeneration: number;
  sparseFluidSurface?: SparseSurfaceBandGPUSource;
}
```

This source exposes structural buffers. Production frames must not materialize
the 48-byte records used by raw-voxel inspection.

### Terminal representations

```ts
type SurfaceRepresentation =
  | "empty"
  | "implicit-primitive"
  | "sampled-smooth-sdf"
  | "sampled-feature-sdf";
```

- `implicit-primitive`: exact sphere, box, capsule, cylinder, ellipsoid, or
  heightfield evaluation referenced by stable primitive/owner ID.
- `sampled-smooth-sdf`: bounded root refinement and continuous gradient, used
  by fluid and smooth voxel-native surfaces.
- `sampled-feature-sdf`: sampled field plus one to three local plane/normal
  constraints when a non-primitive face, edge, or corner must remain sharp.

### Minimum primary-hit record

```ts
interface SparseSceneHit {
  distance_m: number;
  worldPosition_m: readonly [number, number, number];
  geometricNormal: readonly [number, number, number];
  shadingNormal: readonly [number, number, number];
  velocity_m_s: readonly [number, number, number];
  materialId: number;
  ownerId: number;
  surfaceKind: number;
  mediumBefore: number;
  mediumAfter: number;
  topologyGeneration: number;
}
```

### PBR material semantics

The common material table eventually includes scene-linear base color and
emission, roughness, metallic response, specular weight or IOR, transmission,
absorption, scattering, thin-wall behavior, optional procedural material
function ID, stable material ID, and revision.

## Milestone summary

| ID | Milestone | Status | Depends on |
| --- | --- | --- | --- |
| M0 | Baselines, invariants, evidence harness | not started | — |
| M1 | Structural production render-source ABI | not started | M0 |
| M2 | CPU oracle and GPU traversal core | not started | M1 |
| M3 | Fluid level-set visibility | not started | M2 |
| M4 | Dynamic renderer residency and generations | not started | M1–M3 |
| M5 | Exact implicit rigid bodies | not started | M2, M4 |
| M6 | Static scene, terrain, container, thin features | not started | M2, M4 |
| M7 | Opaque composition, G-buffer, interaction | not started | M3, M5, M6 |
| M8 | Shared PBR lighting and SVO visibility rays | not started | M7 |
| M9 | Water and glass media | not started | M3, M7, M8 |
| M10 | Temporal stability and bounded indirect light | not started | M8, M9 |
| M11 | UI, URL state, fallback, observability | not started | M3–M10 |
| M12 | Default flip and migration completion | not started | M0–M11 |

## M0 — Baselines, invariants, evidence harness

Objective: make visual, numerical, resource, and performance regressions
measurable before introducing the renderer.

### Tasks

- [ ] Record sign, units, sample positions, and validity for every existing
  fluid and solid field.
- [ ] Record sparse-brick offsets and adapter-limit assumptions.
- [ ] Add deterministic CPU SDF/normal helpers for every supported primitive.
- [ ] Capture raster reference frames for all acceptance scenes.
- [ ] Capture current raster GPU timings, memory, and internal resolution.
- [ ] Name artifacts by scene, quality, renderer, adapter, resolution,
  simulated time, and revision.
- [ ] Add durable outputs for depth, geometric normal, material, owner, and
  medium ID.
- [ ] Select baseline adapters and document required WebGPU limits.
- [ ] Define tolerances for primitives, sampled fluid, water thickness, and
  temporal rejection.
- [ ] Define balanced/high/ultra frame and memory budgets.
- [ ] Confirm simulation state and diagnostics are renderer-independent.

### Acceptance scenes

- [ ] Settled tank.
- [ ] Dam break, with and without bodies.
- [ ] Sphere and cube under a strong moving highlight.
- [ ] Partially and fully submerged rigid body.
- [ ] Garden terrain and small props.
- [ ] Night laboratory with emissive fixtures and interior occlusion.
- [ ] Thin glass at normal and grazing angles.
- [ ] Deep-water sparse-page stress.
- [ ] Forced sparse allocation overflow/fallback.

### Exit gates

- [ ] Reference artifacts and timing records exist for every scene.
- [ ] Tolerances and budgets are explicit rather than implied.
- [ ] Existing unit, shader, and real-GPU smoke tests pass unchanged.
- [ ] Evidence entry `E-M0` is complete.

Likely files: this document, `tools/run-webgpu-smoke.ts`, a new
`tools/benchmark-svo-renderer.ts`, new CPU reference tests, and retained
artifacts under `artifacts/svo-baseline/`.

## M1 — Structural production render-source ABI

Objective: bind actual sparse topology and payload without expanding debug draw
records.

### Tasks

- [ ] Add `SparseSceneRenderSource` to the solver/renderer boundary.
- [ ] Expose control, node, leaf, geometry, velocity, and material/owner
  bindings with offsets and capacities.
- [ ] Expose brick states plus active and retired worklists.
- [ ] Define topology, static, dynamic, fluid, and generation revisions.
- [ ] Expose fine fluid pages without coupling to the solver class.
- [ ] Add primitive, material, and light bindings or explicit temporary
  fallbacks.
- [ ] Retain `SparseVoxelRenderSource` as inspection-only compatibility.
- [ ] Skip expanded debug publication when inspection is off.
- [ ] Account for memory/time saved by skipping debug publication.
- [ ] Validate alignment and maximum storage-binding limits.
- [ ] Define replacement, destruction, and device-loss ownership.

### Tests and gates

- [ ] Unit-test offsets, strides, capacities, and revision semantics.
- [ ] GPU structural lookup agrees with debug material/owner/origin/extent.
- [ ] Active and retired worklist lookup is validated on a real adapter.
- [ ] Production mode performs no debug-record materialization.
- [ ] No CPU readback controls publication or rendering.
- [ ] Raw-voxel and brick-grid modes still work.
- [ ] Evidence entry `E-M1` is complete.

Likely files: `lib/methods/types.ts`, `lib/sparse-brick-octree.ts`,
`lib/webgpu-octree-sparse-bricks.ts`, `lib/webgpu-uniform-eulerian.ts`,
`lib/webgpu-voxel-debug.ts`, and their tests.

## M2 — CPU oracle and GPU traversal core

Objective: prove robust, bounded traversal before surface and lighting work.

### Tasks

- [ ] Implement scene-domain slab intersection.
- [ ] Implement Morton/node decoding through the structural ABI.
- [ ] Choose and record bounded stack versus stackless/rope traversal.
- [ ] Implement front-to-back child ordering.
- [ ] Implement terminal-leaf intervals and brick-local voxel DDA.
- [ ] Handle origins inside the scene, leaf, and occupied material.
- [ ] Handle zero direction components and boundary-aligned rays.
- [ ] Add conservative per-node/per-brick occupancy summaries.
- [ ] Protect invalid indices, stale generations, stack exhaustion, and
  iteration limits.
- [ ] Return distance, voxel, material, owner, leaf, and counters.
- [ ] Implement a CPU traversal oracle with identical ownership rules.
- [ ] Visualize leaf level, node visits, and DDA steps.

### Tests and gates

- [ ] Empty, single-leaf, nested, mixed-level, and 2:1 fixtures.
- [ ] Boundary, corner, axis-aligned, and inside-origin rays.
- [ ] Deterministic nearest hit among multiple candidates.
- [ ] Missing, retired, invalid, and stale-generation leaves.
- [ ] CPU and GPU agree on deterministic fixtures.
- [ ] Exhaustion is bounded and visible in diagnostics.
- [ ] Full-screen traversal heat map has no validation errors.
- [ ] Evidence entry `E-M2` is complete.

Likely new files: `lib/svo-traversal-reference.ts`,
`lib/webgpu-svo-traversal.ts`, `lib/webgpu-svo-renderer.ts`, and corresponding
CPU/real-GPU tests.

## M3 — Fluid level-set visibility

Objective: render authoritative fluid `phi` directly without triangles.

### Tasks

- [ ] Resolve coarse fluid `phi` from sparse-brick payload.
- [ ] Resolve fine `phi` only from valid sparse surface pages.
- [ ] Fall back to coarse `phi` for missing/overflowed fine pages.
- [ ] Define coarse/fine ownership so a surface is not hit twice.
- [ ] Detect the nearest sign-changing interval during DDA.
- [ ] Refine `phi = 0` with bounded secant/Newton/bisection updates.
- [ ] Compute world gradients from valid neighbor/apron samples.
- [ ] Define degenerate-gradient fallback.
- [ ] Output depth, position, normal, velocity, material, medium, generation.
- [ ] Add opaque diagnostic fluid shading first.
- [ ] Visualize coarse/fine source, root iterations, and gradient validity.
- [ ] Avoid unchanged-frame work while paused.

### Tests and gates

- [ ] Planar, spherical, diagonal, and disconnected analytic level sets.
- [ ] Cross-brick and cross-fine/coarse surfaces.
- [ ] Missing-page and forced-overflow fallback.
- [ ] Ray starting inside fluid and degenerate gradients.
- [ ] Depth/normal comparison against analytic and current render paths.
- [ ] No holes, duplicate sheets, non-finite hits, or mesh extraction.
- [ ] Fluid depth/normals meet M0 tolerances.
- [ ] Evidence entry `E-M3` is complete.

## M4 — Dynamic renderer residency and generations

Objective: update affected support only, without exposing partial scenes.

### Tasks

- [ ] Separate static, dynamic-solid, and fluid update layers.
- [ ] Generalize active/core/halo/retired semantics where needed.
- [ ] Generate fluid dirty bricks from revisions/residency.
- [ ] Generate solid dirty bricks from swept old/new bounds.
- [ ] Pre-activate support from velocity and displacement.
- [ ] Apply retirement hysteresis.
- [ ] Enforce renderer-required 2:1 balance.
- [ ] Define refinement from surface proximity, curvature/normal variation,
  material boundaries, and optional projected error.
- [ ] Allocate strongest surface requests before halos.
- [ ] Keep complete coarse fallback under pressure.
- [ ] Publish complete generations by double buffering or atomic equivalent.
- [ ] Invalidate only changed ancestor summaries.
- [ ] Clear retired dynamic payload while preserving static geometry.
- [ ] Surface overflow and stale-generation counts.
- [ ] Prove renderer refinement cannot affect simulation topology/physics.

### Tests and gates

- [ ] Fluid and solid activation/retirement follow motion without trails.
- [ ] Stationary scenes perform no dynamic rebuild.
- [ ] Rapid direction reversal leaves no gap.
- [ ] Retired payload contains no stale material, owner, or SDF.
- [ ] Forced overflow stays complete at coarse resolution.
- [ ] Renderer never observes a mixed generation.
- [ ] Work scales with changed support, not total scene volume.
- [ ] Simulation output is identical across render modes/qualities.
- [ ] Evidence entry `E-M4` is complete.

## M5 — Exact implicit rigid bodies

Objective: render smooth and sharp bodies equally well without presentation
triangles or fraction-derived distance.

### Tasks

- [ ] Define GPU primitive table with stable primitive and owner IDs.
- [ ] Pack current/previous transforms for rendering and motion vectors.
- [ ] Implement exact sphere, oriented box, capsule, capped cylinder, and
  ellipsoid distance/intersection and normal.
- [ ] Add conservative primitive bounds to SVO coverage.
- [ ] Traverse candidates only after SVO spatial rejection.
- [ ] Resolve nearest fluid-versus-solid hit deterministically.
- [ ] Preserve box face normals; never interpolate across edges.
- [ ] Define edge/corner tie behavior for shading and picking.
- [ ] Update coverage from swept old/new bounds.
- [ ] Keep rigid physics and collision unchanged.

### Tests and gates

- [ ] CPU/GPU distance, hit, and normal comparison for every primitive.
- [ ] Identity/rotated, inside-origin, grazing, tangent, edge, corner, and
  overlapping-owner cases.
- [ ] Moving body generation and stale-coverage regression.
- [ ] Visible hit agrees with existing analytic picker.
- [ ] Sphere silhouette/highlight are continuous.
- [ ] Box faces remain planar and edges sharp.
- [ ] Exact-primitive tolerances pass with no ghost geometry.
- [ ] Evidence entry `E-M5` is complete.

Likely new file: `lib/webgpu-svo-primitives.ts`, with updates to sparse scene
proxies, rigid publication, the SVO renderer, and CPU/real-GPU tests.

## M6 — Static scene, terrain, container, thin features

Objective: make every shipped environment complete without analytic raster
geometry in SVO mode.

### Tasks

- [ ] Compile static box/cylinder/ellipsoid references and SVO coverage.
- [ ] Implement terrain heightfield implicit intersection and normal.
- [ ] Preserve terrain regions and procedural world-space materials.
- [ ] Represent container floor/walls with correct orientation.
- [ ] Choose and implement thin-walled glass representation.
- [ ] Define minimum refinement/analytic treatment for thin furniture and
  fixtures.
- [ ] Preserve room-shell occlusion from interior cameras.
- [ ] Preserve stable environment material/owner keys.
- [ ] Compile static geometry only on static revision changes.
- [ ] Add sampled smooth solid SDF fallback for non-primitives.
- [ ] Add feature SDF records only where exact primitives cannot preserve a
  required face, edge, or corner.
- [ ] Define overlap/CSG ownership among scene layers.

### Tests and gates

- [ ] Coverage test includes every visible authored catalog entry.
- [ ] Terrain height/depth/normal/material comparison passes.
- [ ] Default cameras see the correct room interior.
- [ ] Thin features survive balanced/high/ultra quality.
- [ ] Container top/panes and static revision caching pass.
- [ ] SVO mode invokes no analytic dry-scene presentation.
- [ ] Every shipped environment has complete SVO coverage.
- [ ] Evidence entry `E-M6` is complete.

## M7 — Opaque composition, G-buffer, interaction

Objective: deliver a complete interactive opaque SVO presentation.

### Tasks

- [ ] Choose formats for depth, position/reconstruction, geometric/shading
  normal, material, owner, velocity, medium, and topology generation.
- [ ] Implement full-screen visibility at explicit internal scale.
- [ ] Add environment miss shading.
- [ ] Resolve nearest static, dynamic-solid, and diagnostic fluid hits.
- [ ] Implement SVO picking using shared traversal.
- [ ] Map owner IDs to existing rigid interaction state.
- [ ] Preserve drag position, plane, and grab offset.
- [ ] Implement owner-based selection as a material override.
- [ ] Composite scientific overlays over either renderer.
- [ ] Preserve secondary-particle composition or log an accepted deferral.
- [ ] Add split-screen and depth/normal difference views.
- [ ] Repaint paused scenes on camera/selection changes.

### Tests and gates

- [ ] G-buffer miss/hit encoding and format precision tests.
- [ ] Visible owner equals picked owner and hit position/normal pass tolerances.
- [ ] Dragging works for every body shape.
- [ ] Selection repaints paused scenes.
- [ ] Overlays and difference views are deterministic.
- [ ] Complete opaque scene uses no analytic presentation geometry.
- [ ] Opaque depth/material/owner parity passes.
- [ ] Evidence entry `E-M7` is complete.

## M8 — Shared PBR lighting and SVO visibility rays

Objective: shade authored materials rather than illuminated cubes while keeping
raster and SVO response consistent.

### Tasks

- [ ] Expand material ABI with PBR and dielectric parameters.
- [ ] Implement energy-conserving diffuse, GGX, Smith visibility, and Schlick
  Fresnel.
- [ ] Define geometric-versus-shading normal rules.
- [ ] Supply per-environment diffuse irradiance and prefiltered specular data.
- [ ] Publish directional, point, area, and emissive-fixture lights.
- [ ] Implement SVO shadow rays with conservative opacity.
- [ ] Add temporally sampled soft area shadows.
- [ ] Add bounded short-range ambient/contact visibility.
- [ ] Preserve procedural variation through material function IDs.
- [ ] Keep lighting linear HDR and apply display transfer once.
- [ ] Use the same closure in raster compatibility.

### Tests and gates

- [ ] White-furnace/energy, Fresnel, roughness, metallic, and emission tests.
- [ ] Sphere highlight continuity and cube face/edge reflection tests.
- [ ] Known blocker visibility and closed-wall leak tests.
- [ ] Raster/SVO closure parity and display-transfer tests.
- [ ] Smooth/sharp normals produce intended distinct response.
- [ ] No direct-light leak crosses an opaque closed wall.
- [ ] Evidence entry `E-M8` is complete.

## M9 — Water and glass media

Objective: replace raster interface composition with bounded direct medium
traversal in SVO mode.

### Tasks

- [ ] Define air, water, glass, and solid media and transition rules.
- [ ] Trace fluid entry/exit from `phi`.
- [ ] Compute dielectric Fresnel and Snell refraction.
- [ ] Traverse water to fluid exit or opaque solid contact.
- [ ] Apply Beer-Lambert absorption and bounded in-scattering.
- [ ] Handle total internal reflection.
- [ ] Implement thin-walled glass and required thick dielectrics.
- [ ] Define ordering at coincident/near-contact surfaces.
- [ ] Terminate submerged solid contacts as solid.
- [ ] Bound reflection, transmission, and internal-transition rays.
- [ ] Use environment fallback at the secondary-ray limit.
- [ ] Leave raster optics unchanged in raster mode.
- [ ] Explicitly defer caustics unless a bounded requirement is accepted.

### Tests and gates

- [ ] CPU Fresnel, Snell, Beer-Lambert, and TIR mirrors.
- [ ] Settled thickness, underwater camera, submerged body, glass/water order,
  grazing glass, coincident boundary, and ray-budget tests.
- [ ] Thickness comparison against raster interface buffers passes.
- [ ] No optical loop is unbounded.
- [ ] Thin glass remains stable at grazing angles.
- [ ] Raster optics remains stable as the sole water renderer.
- [ ] Evidence entry `E-M9` is complete.

## M10 — Temporal stability and bounded indirect lighting

Objective: stabilize stochastic work across camera, rigid, fluid, and topology
motion.

### Tasks

- [ ] Publish previous/current camera matrices and rigid transforms.
- [ ] Generate fluid surface motion from published velocity with documented
  limitations; static motion is zero.
- [ ] Key history by depth, normal, material, owner, medium, and generation.
- [ ] Reject disocclusion, identity change, and affected topology history.
- [ ] Add decorrelated sampling and first/second moments.
- [ ] Add small edge-aware spatial filtering.
- [ ] Stabilize soft shadows before indirect light.
- [ ] Add at most one stochastic diffuse bounce initially.
- [ ] Add bounded glossy reflection only if budget permits.
- [ ] Define paused convergence and debug visualizations.

### Tests and gates

- [ ] Camera reprojection, moving rigid, fast fluid, topology activation,
  identity change, and paused convergence tests.
- [ ] Cube edges and thin geometry survive filtering.
- [ ] No persistent trails follow bodies or fluid.
- [ ] Unchanged regions retain history across local topology updates.
- [ ] Indirect lighting remains fixed-bounce/fixed-budget.
- [ ] Evidence entry `E-M10` is complete.

## M11 — UI, URL state, fallback, observability

Objective: expose the migration safely and make failure/performance attributable.

### Tasks

- [ ] Add `SceneRenderMode = "svo" | "raster"`.
- [ ] Add orthogonal `InspectionMode`.
- [ ] Migrate existing UI state without conflating water optics and scene
  representation.
- [ ] Preserve old URLs by mapping legacy parameters.
- [ ] Add development-only split/difference modes.
- [ ] Keep raster default until M12.
- [ ] Fall back on pipeline compilation, adapter-limit, allocation, or fatal
  initialization failure and display the reason.
- [ ] Timestamp publication, topology, visibility, direct light, shadows,
  media, indirect light, temporal resolve, and upscale.
- [ ] Report active/dirty/retired/overflowed bricks.
- [ ] Report node visits, DDA steps, root iterations, secondary rays, history
  rejection, memory categories, and generation.
- [ ] Add `benchmark:svo` with warmups, alternating A/B, timestamps,
  distributions, metadata, and raw JSON.
- [ ] Reset timing context when render/inspection mode changes.

### Tests and gates

- [ ] UI accessibility, URL round-trip/migration, and inspection combinations.
- [ ] Forced pipeline and adapter-limit fallbacks.
- [ ] Timing decode, stale-sample reset, and benchmark equivalence.
- [ ] Raster can always be selected and fallback is explained.
- [ ] Every material stage has timestamp/diagnostic coverage.
- [ ] Evidence entry `E-M11` is complete.

## M12 — Default flip and completion

### Completion checklist

- [ ] M0–M11 are complete with evidence.
- [ ] Every shipped environment has complete interactive SVO coverage.
- [ ] SVO mode contains no analytic dry-scene presentation or mesh extraction.
- [ ] Fluid, terrain, bodies, props, container, water, and glass all render.
- [ ] Material/owner picking works for all interactive objects.
- [ ] Dynamic topology never exposes a partial generation.
- [ ] Sphere smoothness and cube sharpness pass.
- [ ] Water/refraction/absorption/contact gates pass.
- [ ] PBR energy, visibility, and display transfer pass.
- [ ] Temporal gates pass for camera, rigid, fluid, and topology motion.
- [ ] Balanced/high/ultra meet M0 budgets on baseline adapters.
- [ ] Long/soak simulation tests remain renderer-independent.
- [ ] Raster passes its retained suite and remains selectable.
- [ ] Automatic fallback is exercised on a real or limited adapter.
- [ ] Documentation describes SVO default and raster compatibility.
- [ ] Old hybrid-default claims are updated or marked historical.
- [ ] SVO becomes the UI/URL default.
- [ ] Evidence entry `E-M12` is complete.
- [ ] Document status changes to **complete**.

## Validation matrix

| Layer | Purpose | Requirement |
| --- | --- | --- |
| CPU unit | SDFs, normals, traversal, optics, state | deterministic; exact/tolerance-based |
| Shader | WGSL portability and bindings | minimum-limit compatible where declared |
| GPU structural | buffers, addresses, generations | real adapter; invalid access surfaced |
| GPU depth/image | visibility and shading | retain raw outputs, not screenshots alone |
| Interaction | picking, selection, dragging | visible result and owner agree |
| Simulation | renderer independence | identical solver state across modes |
| Performance | frame and memory budget | timestamps, median/p95, raw JSON |
| Soak | retirement, history, resource lifetime | dynamic scenes without growth/leaks |

## Quantitative acceptance categories

M0 must replace every `TBD` before M12 begins.

| Category | Metric | Target |
| --- | --- | --- |
| Exact primitive depth | max/p99 world error | TBD in M0 |
| Exact primitive normal | max/p99 angle | TBD in M0 |
| Sampled fluid depth | p50/p95 finest-cell error | TBD in M0 |
| Sampled fluid normal | p50/p95 angle | TBD in M0 |
| Fine/coarse continuity | uncovered/duplicate pixels | zero above declared epsilon |
| Owner/material identity | valid-hit mismatches | zero |
| Dynamic retirement | stale occupied/material cells | zero after hysteresis |
| Generation safety | partial generations | zero |
| Water thickness | p50/p95 versus reference | TBD in M0 |
| Energy conservation | furnace excess | TBD in M0 |
| Temporal history | retained invalid pixels | TBD in M0 |
| Traversal safety | silent invalid access/exhaustion | zero |
| CPU render readback | per-frame control decisions | zero |
| Balanced frame time | median/p95 by adapter | TBD in M0 |
| Renderer memory | peak by adapter | TBD in M0 |

## Performance rules

- Report SVO and raster internal resolutions with comparisons.
- GPU timestamps are authoritative; CPU encoding remains separate.
- Separate unchanged-frame and changed-topology cost.
- Separate primary, shadow, media, indirect, temporal, and upscale work.
- Report overflow/exhaustion beside timing.
- Retain distributions and raw metadata, not only averages.
- Quality may reduce secondary/indirect work but cannot change simulation state.

## Fallback contract

Raster compatibility is used when the user selects it, required pipelines fail
to compile, adapter limits cannot be met, structural allocation fails, fatal SVO
initialization validation occurs, or device recovery requires it. Runtime sparse
pressure should first use explicit coarse/quality fallback inside SVO mode.
Every automatic renderer switch must report its reason.

## Risk register

| ID | Risk | Impact | Mitigation | Status |
| --- | --- | --- | --- | --- |
| R1 | Fraction distance rounds sharp geometry | high | exact primitives; feature records for non-primitives | open |
| R2 | Thin glass/props vanish | high | thin-surface encoding or mandatory refinement | open |
| R3 | Mixed dynamic generations | critical | double buffer or atomic generation publication | open |
| R4 | Fine/coarse fluid holes/double sheets | critical | explicit ownership and complete coarse fallback | open |
| R5 | Traversal divergence misses budget | high | summaries, ordering, telemetry, quality scaling | open |
| R6 | Media exceed ray budget | high | fixed limits and environment fallback | open |
| R7 | Temporal filter smears water/edges | high | motion/identity/generation rejection | open |
| R8 | Procedural environments lose detail | medium | material functions and targeted refinement | open |
| R9 | Lighting leaks through coarse geometry | high | conservative opacity and closed-wall tests | open |
| R10 | Renderer changes simulation | critical | mode-independence tests; no controlling readback | open |
| R11 | Inspection breaks after optimization | medium | retain inspection ABI and tests | open |
| R12 | WebGPU limits reject bindings | high | arenas/offsets, limit audit, raster fallback | open |
| R13 | Resource/history memory grows | high | stable pools, lifetime tests, memory telemetry | open |
| R14 | Raster rots after default flip | high | shared closure and retained acceptance suite | open |

## Open design decisions

| ID | Question | Recommendation | Needed by | Status |
| --- | --- | --- | --- | --- |
| O1 | Stack or ropes? | start bounded stack; measure before topology cost | M2 | open |
| O2 | Fluid corners or centres plus apron? | reuse current convention; require explicit gradient tests | M3 | open |
| O3 | Renderer topology separate from simulation? | separate policy, shared fields/residency signals | M4 | open |
| O4 | Generation publication? | double-buffer dynamic payload plus atomic switch | M4 | open |
| O5 | Primitive candidate packing? | per-leaf ranges into shared index arena | M5 | open |
| O6 | Thin glass representation? | oriented thin-walled implicit surface | M6 | open |
| O7 | G-buffer formats/internal scale? | choose from measured precision/bandwidth | M7 | open |
| O8 | Environment irradiance? | prefiltered environment plus dynamic direct lights | M8 | open |
| O9 | Initial caustics? | defer from base optical acceptance | M9 | open |
| O10 | First indirect method? | one stochastic diffuse bounce plus filtering | M10 | open |

## Decision ledger

Add entries chronologically; supersede rather than rewriting old decisions.

| Date | ID | Decision | Reason | Milestones |
| --- | --- | --- | --- | --- |
| 2026-07-19 | D1 | Direct SVO traversal is the target renderer. | Removes production scene triangles and uses sparse state directly. | all |
| 2026-07-19 | D2 | Skip dual contouring. | It still produces a mesh and does not meet the primary goal. | M2–M12 |
| 2026-07-19 | D3 | Keep raster as toggle/fallback. | Enables A/B validation, adapter coverage, and reversible rollout. | M0, M8–M12 |
| 2026-07-19 | D4 | Use heterogeneous terminal surfaces. | Exact primitives preserve spheres/cubes; SDFs suit fluid/general geometry. | M1–M10 |
| 2026-07-19 | D5 | Reuse authoritative fluid `phi` and fine pages. | The smooth GPU-resident fluid representation already exists. | M1, M3, M9 |
| 2026-07-19 | D6 | Fraction-derived solid distance is not final presentation geometry. | It cannot preserve exact curves and sharp edges. | M5–M7 |

## Evidence ledger

Each entry should name committed artifacts, raw benchmark files, commands,
adapter metadata, and accepted deviations.

| ID | Milestone | Date | Evidence | Result |
| --- | --- | --- | --- | --- |
| E-M0 | M0 | — | — | pending |
| E-M1 | M1 | — | — | pending |
| E-M2 | M2 | — | — | pending |
| E-M3 | M3 | — | — | pending |
| E-M4 | M4 | — | — | pending |
| E-M5 | M5 | — | — | pending |
| E-M6 | M6 | — | — | pending |
| E-M7 | M7 | — | — | pending |
| E-M8 | M8 | — | — | pending |
| E-M9 | M9 | — | — | pending |
| E-M10 | M10 | — | — | pending |
| E-M11 | M11 | — | — | pending |
| E-M12 | M12 | — | — | pending |

## Final definition of done

The migration is complete when direct SVO rendering is the default; every
shipped scene is complete and interactive through it; fluid and curved
primitives remain smooth; sharp solids remain sharp; realistic lighting and
bounded media pass quantitative gates; dynamic publication is generation-safe;
simulation evidence is unchanged; and raster compatibility still passes its
retained suite as a selectable and automatic fallback.
