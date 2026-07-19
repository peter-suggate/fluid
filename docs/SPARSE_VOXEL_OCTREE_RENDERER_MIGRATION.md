# Sparse voxel octree renderer migration

Status: **in progress**
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

1. A production dry-scene path now binds the structural octree directly, but
   fluid visibility still uses extracted raster interfaces and current small
   authored proxy catalogs use a direct primitive-table fast path.
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

### Audited existing field inventory

This is the M0 contract recorded from the implementation on 2026-07-19. It
describes what exists now, not the final render-source ABI.

| Field/source | Sign and units | Sampling and quality | Current validity/revision contract | Production use |
| --- | --- | --- | --- | --- |
| Coarse fluid `phi` | negative in liquid; world metres | finest transport-cell centres; transported near zero and rebuilt/capped to `+/-5h` in the far field | stable `r32float` texture identity; update sequence is private | authoritative near-interface root field |
| Sparse-brick `geometry.x` | copied coarse fluid `phi`; world metres | finest solver-leaf cell centres | residency/domain membership is the real validity gate; never-activated leaves can contain false zero; retirement writes `FLT_MAX`; render revision is fixed at `1` | unusable without explicit validity and revision semantics |
| Fine sparse fluid pages | negative in liquid; world metres | fine cell centres aligned to coarse cell centres; missing pages fall back to coarse trilinear `phi` | page and physical-slot checks required; band generation is exposed, but is not paired with a coarse-field revision or GPU-completion generation | authoritative refinement after validated lookup |
| Dynamic rigid solid estimate | negative when occupancy exceeds `0.5`; world metres | eight-sample volume fraction converted to a bounded local estimate | no dynamic-solid revision | occupancy/support only; never final presentation geometry |
| Terrain solid estimate | same fraction-derived convention as dynamic solids | cell occupancy from the terrain heightfield | no terrain-field revision in the sparse render ABI | use the heightfield implicit for presentation instead |
| Static box/cylinder/sphere proxy | negative inside; world metres | analytic implicit sampled at terminal-voxel centres; exact for these primitives | static proxy payload is encoded once; positive support can be overwritten where no material ID is retained | usable as sampled support; prefer exact primitive reference for final hits |
| General ellipsoid proxy | negative inside; world metres | zero-set/sign-correct distance approximation | same static limitations | use the exact CPU/GPU ellipsoid primitive implementation |
| Container boundary | material identity only on some boundary cells | no corresponding negative wall field is guaranteed | synthetic raster boundary currently closes water | requires an explicit presentation implicit in M6 |

Audit consequences that are requirements, rather than optional cleanup:

- Add separate topology, static, dynamic-solid, coarse-fluid, and fine-fluid
  revisions plus a published complete generation.
- Add explicit per-field validity. A finite value, and especially `phi == 0`,
  must never imply that a field is valid.
- Resolve the existing solid estimate scale mismatch: sparse bricks use
  `(0.5 - fraction) * 2h`, while fine-band sizing uses
  `(0.5 - fraction) * h`.
- Do not promote the fine-band sizing expression `fluidPhi + solidPhi` into a
  render contract; fluid and solid fields remain separate.
- Treat `phi == 0` as the boundary. Liquid classification remains `phi < 0`.

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
| M0 | Baselines, invariants, evidence harness | in progress | — |
| M1 | Structural production render-source ABI | in progress | M0 |
| M2 | CPU oracle and GPU traversal core | in progress | M1 |
| M3 | Fluid level-set visibility | in progress | M2 |
| M4 | Dynamic renderer residency and generations | in progress | M1–M3 |
| M5 | Exact implicit rigid bodies | in progress | M2, M4 |
| M6 | Static scene, terrain, container, thin features | in progress | M2, M4 |
| M7 | Opaque composition, G-buffer, interaction | in progress | M3, M5, M6 |
| M8 | Shared PBR lighting and SVO visibility rays | in progress | M7 |
| M9 | Water and glass media | in progress | M3, M7, M8 |
| M10 | Temporal stability and bounded indirect light | in progress | M8, M9 |
| M11 | UI, URL state, fallback, observability | in progress | M3–M10 |
| M12 | Default flip and migration completion | in progress | M0–M11 |

## M0 — Baselines, invariants, evidence harness

Objective: make visual, numerical, resource, and performance regressions
measurable before introducing the renderer.

### Tasks

- [x] Record sign, units, sample positions, and validity for every existing
  fluid and solid field.
- [x] Record sparse-brick offsets and adapter-limit assumptions.
- [x] Add deterministic CPU SDF/normal helpers for every supported primitive.
- [x] Define stable acceptance case IDs, cameras, fixed-step checkpoints, and
  raster capture defaults.
- [ ] Capture raster reference frames for all acceptance scenes.
- [ ] Capture current raster GPU timings, memory, and internal resolution.
- [x] Name artifacts by scene, quality, renderer, adapter, resolution,
  simulated time, and revision.
- [ ] Add durable outputs for depth, geometric normal, material, owner, and
  medium ID.
- [x] Select the primary performance adapter and portable required-limit
  contract; a second physical adapter capture remains outstanding.
- [x] Define tolerances for primitives, sampled fluid, water thickness, and
  temporal rejection.
- [x] Define balanced/high/ultra frame and renderer-memory target budgets.
- [ ] Confirm simulation state and diagnostics are renderer-independent.

Current implementation note (2026-07-19): deterministic CPU references now
cover sphere, oriented box, capsule, capped cylinder, exact Euclidean
ellipsoid, and the terrain heightfield. The terrain value is the signed
vertical height residual, so it preserves sign and the zero set in metres but
is not Euclidean distance away from slopes. Ambiguous gradients deliberately
return no normal. The acceptance catalog defines 13 deterministic variants
across all nine acceptance areas, and the artifact-path contract encodes every
required comparison dimension in a traversal-safe path. `npm run baseline:svo`
now emits 26 paired raster/hybrid jobs and can ingest externally captured color
and timing observations into canonical scene, camera, raw timing, signal, and
manifest paths. It is a deterministic planner/validator/ingester, not an
automated browser capture: live color capture and renderer readback still have
to supply its observation input.

The renderer benchmark observation contract is now schema version 2. A frame
is admissible only when its accepted timestamp sample ID advances; cached
telemetry, mode/epoch drift, and unequal raster/SVO camera or checkpoint state
fail closed. Every accepted frame carries scene, temporal, and total
presentation timing together with adapter identity, requested/effective mode,
timing context/epoch, and output/internal resolution. This makes a future M0
distribution reproducible without treating a UI repaint or repeated cached
sample as another GPU observation. No 120-frame distribution is claimed by
this contract change alone.

The captured hybrid renderer identity is explicit: direct SVO dry-scene output
feeds the existing raster water extraction/interface/optical path. The primary
performance adapter is Apple M3 Max/Metal. A portable compatibility record must
meet at least 10 storage buffers per shader stage, 128 MiB storage bindings,
256 MiB buffers, three color attachments and exactly 32 attachment bytes per
sample, 8192/2048 2D/3D texture dimensions, and 65,535 compute workgroups per
axis. Timestamp queries are required for performance evidence but not rendering
correctness; `float32-filterable` remains optional. The pinned sparse ABI is
32-byte nodes, 16-byte leaves/geometry/velocity, 4-byte material-owner words,
128-byte control storage, leaf payload base `leaf.topology.y`, and indirect
offsets 80/96 bytes.

Durable bundle names are now fixed for display color, raw timings, canonical
scene/camera inputs, linear depth, geometric normal, material/owner/media
identity, linear-energy statistics, and the manifest. Only color/timing and
input metadata are ingestible from current public hooks; depth/normal/identity
and energy readbacks remain deliberately reported as outstanding signals.

### Quantitative acceptance targets

All spatial tolerances are world-space and use `h = min(cellSize_m)`. A result
must satisfy both finiteness/identity rules and the listed numeric bound.

| Signal | Required tolerance |
| --- | --- |
| Exact primitive depth | `max(1 mm, 0.05 h)`; analytic tangent/inside ownership must also match |
| Sphere/ellipsoid normal | angular error at most 1 degree away from ambiguous singularities |
| Box/capped-cylinder feature | exact face/edge/corner feature ID and one non-interpolated feature normal |
| Coarse sampled fluid depth | `max(2 mm, 0.35 h)` with no invalid-corner interpolation |
| Fine sampled fluid depth | `max(1 mm, 0.20 hFine)` once fine pages are published |
| Fluid normal | angular error at most 8 degrees coarse and 4 degrees fine |
| Terrain depth/normal | `max(1 mm, 0.05 h)` and at most 2 degrees |
| Water thickness | `max(5 mm, 0.50 h)`; linear-RGB transmittance absolute error at most 0.03/channel |
| Temporal static depth/reprojection | `max(2 mm, 0.20 h)` / `max(1 mm, 0.15 h)` |
| Temporal normals | reject below `cos(20°)` geometric or `cos(35°)` shading agreement |

Performance targets are GPU timestamps at the renderer's explicit internal
resolution after 30 warm frames. They are gates, not claims about current
completion; distributions require alternating raster/SVO runs and p95 values.
All numeric tolerances and budgets in this section are provisional until the
complete paired capture matrix validates that they discriminate regressions
without rejecting stable adapter variance.

| Quality | SVO visibility + direct light p95 | Total presentation p95 | Renderer-owned memory ceiling |
| --- | ---: | ---: | ---: |
| Balanced | 1.0 ms | 4.0 ms | 192 MiB |
| High | 1.75 ms | 6.0 ms | 384 MiB |
| Ultra | 3.0 ms | 8.0 ms | 768 MiB |

No quality may exceed 16.67 ms total presentation at the acceptance output
resolution, expose a partial generation, or trade simulation-field fidelity
for rendering performance. Secondary work remains bounded by each milestone's
published caps even when a time target is missed.

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

Baseline health note: the latest full unit suite is green at 699 passing, zero
failing, and 15 hardware-dependent skips (714 total). TypeScript, focused lint,
water-shader validation (including octree projection), and live browser WebGPU
compile/render are green. Native solver smoke, a second physical adapter,
complete paired color/timing captures, and renderer signal readbacks are still
required for the M0 exit gate.

The restricted tall-cell Section 8 hardware gate is also green after restoring
single-phase level-set endpoint transfer during remeshing: the Metal dam-front
transient at `t=0.224 s` reports `0/1768` wet columns with their surface outside
the cubic band, `0` dry stores beneath wet band cells, and `0` Eq. 10 neighbour
violations. A clean-HEAD A/B reproduced the former `84/1971` failure exactly,
proving it was a latent restricted-remapper invariant bug rather than a change
to quadtree sparse surface execution or volume correction.

Likely files: this document, `tools/run-webgpu-smoke.ts`, a new
`tools/benchmark-svo-renderer.ts`, new CPU reference tests, and retained
artifacts under `artifacts/svo-baseline/`.

## M1 — Structural production render-source ABI

Objective: bind actual sparse topology and payload without expanding debug draw
records.

### Tasks

- [x] Add a structural production source to the solver/renderer boundary.
- [x] Expose control, node, leaf, geometry, velocity, and material/owner
  bindings with offsets and capacities.
- [x] Expose authoritative brick states plus active/core/halo/retired
  worklist views, counters, offsets, capacities, state bits, and list
  generation from `GPUFluidBrickResidency`.
- [x] Define topology, static, dynamic, fluid, and generation revisions.
- [x] Add explicit fluid, dynamic-solid, static-solid, and fine-page validity;
  never infer validity from zero or finiteness.
- [x] Expose a typed renderer-owned fine-fluid capability without coupling to
  the solver class: authoritative sparse-surface buffers are consumed
  read-only, structural/fine generations are fenced independently, and each
  staged page carries an accepted fine-generation stamp. Production visibility
  consumes it through the combined publication/fine arena while legacy raster
  water retains presentation ownership.
- [x] Add primitive, material, and light bindings or explicit temporary
  fallbacks.
- [x] Retain `SparseVoxelRenderSource` as inspection compatibility while its
  nested structural source feeds production traversal.
- [x] Skip expanded voxel/brick debug count copies, compute passes, and
  dispatches in smooth production mode while structural publication remains
  unconditional; paused inspection toggles use explicit pending publication.
- [ ] Account for memory/time saved by skipping debug publication.
- [x] Validate alignment and the eight read-only storage-binding path on the
  baseline browser adapter.
- [x] Define replacement, destruction, and device-loss ownership.

### Tests and gates

- [x] Unit-test offsets, strides, capacities, and revision semantics.
- [x] GPU structural lookup agrees with the CPU oracle on authoritative
  material/owner, payload offset, local voxel, anisotropic leaf origin/extent,
  and nearest entry distance without consulting expanded debug records.
- [ ] CPU/WGSL worklist layout, state decoding, lifecycle, and generation
  semantics are validated; the real-adapter structural lookup now covers valid,
  stale, unpublished, retired, malformed-topology, and bounded-work outcomes,
  while a complete end-to-end worklist lifecycle oracle remains open.
- [x] Production smooth mode performs no debug-record materialization; raw and
  brick inspection modes re-enable it before encode.
- [x] No CPU readback controls publication or rendering.
- [x] Raw-voxel and brick-grid modes still work.
- [ ] Evidence entry `E-M1` is complete.

Likely files: `lib/methods/types.ts`, `lib/sparse-brick-octree.ts`,
`lib/webgpu-octree-sparse-bricks.ts`, `lib/webgpu-uniform-eulerian.ts`,
`lib/webgpu-voxel-debug.ts`, and their tests.

## M2 — CPU oracle and GPU traversal core

Objective: prove robust, bounded traversal before surface and lighting work.

### Tasks

- [x] Implement scene-domain slab intersection.
- [x] Implement Morton/node decoding through the structural ABI.
- [x] Choose and record bounded explicit-stack traversal (32 entries, 256
  visits) versus stackless/rope traversal.
- [x] Implement front-to-back child ordering.
- [x] Implement terminal-leaf intervals and brick-local voxel DDA.
- [x] Handle origins inside the scene, leaf, and occupied material.
- [x] Handle zero direction components and boundary-aligned rays.
- [ ] Add conservative per-node/per-brick occupancy summaries.
- [x] Protect invalid indices, unpublished generations, stack exhaustion, and
  iteration limits.
- [x] Return distance, voxel, material, owner, leaf, and counters.
- [x] Implement a CPU traversal oracle for the packed topology with identical
  near-to-far and boundary ownership rules.
- [ ] Visualize leaf level, node visits, and DDA steps.

### Tests and gates

- [x] Empty, single-leaf, nested, mixed-level, and coarse/fine fixtures.
- [x] Boundary, corner, axis-aligned, and inside-origin rays.
- [x] Deterministic nearest hit among multiple candidates.
- [x] Missing, retired, invalid, and stale-generation leaves.
- [x] CPU and GPU agree on deterministic fixtures.
- [x] Exhaustion is bounded and returned as a distinct status.
- [ ] Full-screen traversal heat map has no validation errors.
- [ ] Evidence entry `E-M2` is complete.

Likely new files: `lib/svo-traversal-reference.ts`,
`lib/webgpu-svo-traversal.ts`, `lib/webgpu-svo-renderer.ts`, and corresponding
CPU/real-GPU tests.

## M3 — Fluid level-set visibility

Objective: render authoritative fluid `phi` directly without triangles.

### Tasks

- [x] Resolve coarse fluid `phi` from the packed structural sparse-brick
  payload with bounded node lookup, publication/residency checks, anisotropic
  world mapping, strict cross-brick interpolation, production geometry and
  leaf-state bindings, and explicit invalid/miss/exhausted results.
- [x] Resolve fine `phi` only from valid sparse surface pages. The renderer
  staging arena, apron-safe trilinear/gradient WGSL, per-page generation stamp,
  mandatory exact coarse fallback, and production renderer resource-owner
  lifecycle now feed the structural primary/root/normal sampler. A combined
  publication/fine arena reuses fragment binding 8 and keeps the portable
  ten-storage-buffer ceiling unchanged.
- [x] Define the validity rule which falls back to coarse `phi` for a missing
  or invalid fine sample (GPU page binding remains open).
- [x] Define exclusive coarse/fine ownership so a surface is not hit twice.
- [x] Detect the nearest sign-changing interval during a bounded ray walk.
- [x] Refine `phi = 0` with safeguarded bounded secant/bisection updates.
- [x] Compute anisotropic world gradients from valid central or one-sided
  neighbor samples.
- [x] Define a finite deterministic degenerate-gradient fallback.
- [ ] Output depth, position, normal, velocity, material, medium, generation;
  the production coarse diagnostic now writes depth, smooth gradient normal,
  stable fluid material/source, oriented air/water media, and generation, while structural
  velocity is intentionally marked unavailable until its binding is integrated.
- [x] Add opt-in opaque diagnostic fluid shading first, behind an explicit
  primary-ownership mode which defaults to the legacy water compositor.
- [ ] Visualize root iterations and gradient validity. Coarse/fine root source
  identity now uses the existing compact G-buffer `fluidCoarse`/`fluidFine`
  field-source lane without expanding the MRT ABI.
- [x] Avoid unchanged fine-field compaction/allocation/staging work while
  paused by queueing the production chain at most once per published fine
  generation; presentation shading still executes when a repaint is required.

### Tests and gates

- [x] Planar and nonlinear analytic root fixtures (spherical, diagonal, and
  disconnected GPU fixtures remain).
- [x] Cross-brick coarse lookup, conservative trilinear interpolation, root,
  inside-exit, and anisotropic gradient fixtures; continuous cross-fine/coarse
  surface fixtures remain open.
- [x] Invalid, stale-generation, missing-owner, missing-page, non-finite, and
  partial fine ownership falls back to valid coarse `phi`; forced production
  GPU page overflow evidence remains.
- [x] Ray starting inside fluid and degenerate gradients.
- [ ] Depth/normal comparison against analytic and current render paths.
- [ ] No holes, duplicate sheets, non-finite hits, or mesh extraction.
- [ ] Fluid depth/normals meet M0 tolerances.
- [ ] Evidence entry `E-M3` is complete.

## M4 — Dynamic renderer residency and generations

Objective: update affected support only, without exposing partial scenes.

### Tasks

- [x] Define separate static, dynamic-solid, and fluid update layers and
  uint32 revision comparisons (fine-payload consumers remain open).
- [x] Define active/core/halo/retired renderer-residency semantics.
- [x] Generate bounded fluid dirty-brick requests from revisions/residency
  inputs in the CPU contract.
- [x] Generate bounded solid dirty-brick requests from swept old/new bounds in
  the CPU contract.
- [x] Define velocity/displacement pre-activation.
- [x] Define deterministic retirement hysteresis and reactivation.
- [ ] Enforce renderer-required 2:1 balance.
- [ ] Define refinement from surface proximity, curvature/normal variation,
  material boundaries, and optional projected error.
- [x] Allocate core/surface requests before active support, halos, and retired
  detail in the deterministic residency contract.
- [x] Keep complete coarse fallback publishable under detail pressure.
- [x] Define and instantiate the complete-generation publication gate in the
  production renderer: consumer, owner allocation/retirement, and fine staging
  execute in queue order and publish only after their GPU fences complete.
- [ ] Invalidate only changed ancestor summaries.
- [ ] Clear retired dynamic payload while preserving static geometry.
- [x] Surface source overflow, renderer exhaustion, invalid-entry, coarse-
  fallback, and stale-generation counts in a GPU-owned control arena.
- [x] Keep the GPU worklist consumer unable to affect simulation
  topology/physics: all three producer arenas are immutable bindings and all
  compaction stamps, requests, releases, counters, and dispatches are
  renderer-owned. End-to-end render-quality simulation identity remains a
  separate test gate.
- [x] Consume the validated renderer worklist with a renderer-owned physical
  owner-page arena and deterministic free list: activation precedes retirement,
  accepted generation publishes only after zero-fill/lifecycle completion, and
  overflow/invalid/stale/unchanged state remains GPU telemetry with no renderer
  CPU readback. Fine `phi` staging and visibility consumption are separately
  fenced so owner allocation alone never advertises fine-field validity.
- [x] Stage authoritative fine `phi` into bounded renderer owner slots with a
  one-sample apron, per-page fine-generation validity, source/owner generation
  fences, deterministic retired-tile air scrubbing, and explicit partial-page
  telemetry. The typed capability requires coarse fallback and explicitly
  cannot enable direct-water ownership.
- [x] Embed the solver-local residency and fine-field domains in the padded
  structural scene lattice: active and retired brick IDs are translated by the
  aligned solver origin before owner allocation, staging subtracts that origin
  only for producer-page lookup, and strict source/refinement/residency/bounds
  validation rejects malformed layouts before resource creation.
- [x] Own the consumer, owner allocator, and fine stager transactionally across
  solver attach/replacement, unchanged frames, source-buffer replacement,
  renderer teardown, and device loss; detach replacement bindings before the
  old solver retirement fence and expose only a read-only future capability.

### Tests and gates

- [ ] Fluid and solid activation/retirement follow motion without trails.
- [x] An unchanged completed generation emits zero GPU compaction work; the
  broader stationary-scene producer-revision gate remains to be measured.
- [ ] Rapid direction reversal leaves no gap.
- [ ] Retired payload contains no stale material, owner, or SDF.
- [x] Forced source/output overflow emits explicit telemetry and zero missing
  coarse coverage; fine staging capacity degrades explicitly and invalid pages
  retain coarse ownership, while live production overflow evidence remains.
- [x] The publication contract never exposes a mixed generation; immutable
  structural-source bindings and GPU fence validation now consume the real
  producer ABI, and fine capability publishes only after source, owner, apron
  staging, and retirement clearing complete. Production visibility rechecks
  the same structural, fine, per-page, and owner-slot fences before sampling.
- [ ] Work scales with changed support, not total scene volume.
- [ ] Simulation output is identical across render modes/qualities.
- [ ] Evidence entry `E-M4` is complete.

## M5 — Exact implicit rigid bodies

Objective: render smooth and sharp bodies equally well without presentation
triangles or fraction-derived distance.

### Tasks

- [x] Define a 64-byte GPU primitive table with stable primitive, material, and
  owner IDs.
- [x] Define a same-index 128-byte primitive-motion sidecar with normalized
  previous/current transforms, revisions/generations, exact surface velocity,
  conservative swept bounds, and fail-closed continuity rules.
- [x] Pack GPU-resident current/previous rigid transforms, exact surface
  velocities, revisions/generations, and continuity flags for rendering and
  motion vectors in the established 128-byte sidecar.
- [x] Complete exact sphere, oriented box, capsule, capped cylinder, and
  ellipsoid distance/intersection/normal parity. Ellipsoid Euclidean distance
  now uses the same active-axis closest-point construction on CPU and WGSL,
  with a fixed 64-step f32 bisection ceiling; malformed dimensions return the
  invalid sentinel and ambiguous medial-axis normals remain invalid.
- [x] Add conservative authored-proxy bounds to SVO coverage.
- [x] Reject static authored primitives through a conservative GPU-resident
  balanced BVH before exact intersection; append its 64-byte nodes to the
  primitive storage arena, retain the SVO leaf-payload path for catalogs over
  64 entries, and fail to raster on missing, corrupt, or exhausted metadata.
- [ ] Resolve nearest fluid-versus-solid hit deterministically.
- [x] Preserve box face normals; never interpolate across edges.
- [x] Define deterministic edge/corner feature-normal ties for shading.
- [x] Publish conservative swept old/new bounds from the same GPU motion record
  for active coverage preactivation; spatial candidate rejection remains a
  separate traversal task.
- [ ] Keep rigid physics and collision unchanged.

### Tests and gates

- [x] CPU descriptor/packed-record and real-GPU hit/normal comparison for every
  finite primitive kind.
- [x] Identity/rotated, inside-origin, grazing, tangent, edge, corner, and
  overlapping-owner cases.
- [x] Moving-body generation, discontinuity/teleport, roster-compaction, and
  conservative swept-coverage regression.
- [ ] Visible hit agrees with existing analytic picker.
- [x] CPU primitive evidence proves continuous sphere normals and single hard
  box feature normals.
- [x] Box faces remain planar and edges sharp in the primitive contract.
- [x] Exact-primitive CPU/GPU tolerances pass at `3e-4` metres/components;
  tangent hits, grazing misses, invalid dimensions, inside exits, and hard
  edge/corner ties produce no contract-level ghost geometry.
- [ ] Evidence entry `E-M5` is complete.

Likely new file: `lib/webgpu-svo-primitives.ts`, with updates to sparse scene
proxies, rigid publication, the SVO renderer, and CPU/real-GPU tests.

## M6 — Static scene, terrain, container, thin features

Objective: make every shipped environment complete without analytic raster
geometry in SVO mode.

### Tasks

- [x] Compile static box/cylinder/ellipsoid references and SVO coverage with
  owner/material identity preserved exactly.
- [x] Implement terrain heightfield implicit intersection and normal.
- [x] Mirror the existing raster garden terrain regions and seeded world-space
  procedural material exactly in CPU/WGSL, with stable terrain/sub-region IDs,
  packed metadata, revision, boundary ties, production dry-uniform composition,
  PBR base-color/roughness evaluation, and a pending-G-buffer identity adapter.
- [x] Represent container floor/walls/top as correctly oriented finite panes;
  the post-dry-scene vessel compositor owns their primary-camera application
  until M9 replaces that compositor, while SVO shadow rays still see them.
- [x] Define the 80-byte oriented finite-pane ABI, two-sided feature/normal
  rules, optics, conservative bounds, CPU/WGSL reference, deterministic
  authored pane catalog, and production revision-cached upload/binding.
- [x] Keep thin furniture/fixtures as exact analytic primitives and publish
  separate conservative coverage/collision-proxy bounds below 1.5 nominal
  cells, preserving owner/material identity without altering rigid or solver
  collision physics.
- [x] Preserve room-shell occlusion from interior cameras while explicitly
  omitting the front/open shell owner.
- [x] Preserve stable environment material/owner keys.
- [x] Publish every selected environment's stable `32 + ownerIndex` material
  into the production 96-byte direct-index PBR table; reject invalid owner/ID
  bounds before upload instead of allowing valid primitive hits to shade black.
- [x] Publish stable semantic material-function IDs for architectural, wood,
  stone, foliage, ceramic, brushed-metal, and organic authored surfaces, and
  evaluate bounded seeded world-space color/roughness variation without a
  texture, readback, owner-local coordinate, or additional storage binding.
- [x] Split the night-lab back wall into four exact analytic boxes around the
  authored city-window pane, removing its whole-environment opaque-cutout
  fallback without adding CSG or mesh geometry.
- [x] Add a stable finite research-station observation-port backing/frame and
  analytic thin pane with deterministic owner/material/pane identities; retain
  the raster wall's circular/curved transmission as an explicit unsupported
  detail rather than approximating it with invented CSG.
- [x] Add deterministic environment/preset coverage reporting (`npm run
  report:svo-scenes`) with explicit visible, collision, and lighting ownership,
  stable owner/material IDs, default-camera priority, and typed degraded or
  unsupported fallbacks.
- [x] Content-hash and bounded-cache primitive, pane, material, light, and
  coverage publications so unchanged solver/world rebuilds reuse the same
  immutable packed-record identities and only repack on static revision
  changes.
- [x] Define a separate bounded 80-byte thick-glass sphere/ellipsoid ABI with
  exact oriented entry/exit roots, outward normals, medium-transition optics,
  authored globe/station-lens identities, and fail-closed revision status;
  production binding remains deferred to M9.
- [ ] Add sampled smooth solid SDF fallback for non-primitives.
- [ ] Add feature SDF records only where exact primitives cannot preserve a
  required face, edge, or corner.
- [ ] Define overlap/CSG ownership among scene layers.

Current deterministic catalog audit (`npm run report:svo-scenes`):

| Environment | Complete | Degraded | Unsupported | Exact remaining typed gaps |
| --- | ---: | ---: | ---: | --- |
| `default` | 7 | 0 | 0 | none |
| `conservatory` | 34 | 3 | 1 | `pendant-{0,1,2}/globe` thick-glass contract is authored but production-unbound; botanical raster foreground |
| `courtyard` | 28 | 0 | 1 | citrus raster foreground |
| `night-lab` | 46 | 2 | 1 | `desk-lamp/bulb` thick-glass contract is authored but production-unbound; `counter/monitor-screen` emissive-display optics; raster vignette |
| `concrete-gallery` | 21 | 0 | 1 | raster slab-dust particles |
| `bathhouse` | 28 | 0 | 1 | raster post-cloth foreground |
| `research-station` | 32 | 2 | 2 | `console-{left,right}/monitor` emissive-display optics; authored elliptical thick lens awaits production binding for curved porthole transmission; raster frame-drift particles |
| `garden` | 17 | 0 | 1 | raster grass/sun-bloom foreground |

The report merges optical degradation into the existing analytic proxy entry,
so keys are one-to-one rather than double-counted. All fixture lights fit the
bounded publication and every finite authored pane is complete. Direct analytic
visibility preserves default-camera subcell props. Bathhouse lantern cords,
conservatory pendant cords, night-lab monitor/keyboard/lamp/troffer/shelf
thicknesses, and the station observation-port backing now retain exact analytic
hit geometry plus separately audited conservative bounds below 1.5 nominal
cells. Frames already wider than that threshold retain exact bounds. These
coverage bounds do not mutate rigid-body or solver collision physics.

### Tests and gates

- [x] Coverage test includes every authored analytic primitive, finite pane,
  terrain source, light, shipped preset rigid body, and typed procedural or
  optical gap.
- [x] Terrain height/depth/normal comparison passes.
- [x] Terrain region/procedural material CPU parity, production shader-source
  composition, packed-uniform/cache, and binding-budget comparisons pass.
- [x] Authored material-function classification, revision invalidation,
  CPU/WGSL policy/seed parity, bounded PBR inputs, and cross-primitive/cell
  seam continuity pass for every shipped environment and all garden props.
- [ ] Default cameras see the correct room interior.
- [ ] Thin features survive balanced/high/ultra quality.
- [x] Container top/panes, camera-orbit compositor ownership, upload lifecycle,
  and static pane revision caching pass.
- [x] Static primitive/pane/material/light/coverage cache identity and content
  invalidation pass; unchanged publications reuse packed arrays by identity.
- [x] SVO mode replaces rather than overlays the legacy analytic dry-scene
  presentation pass.
- [ ] Every shipped environment has complete SVO coverage.
- [ ] Evidence entry `E-M6` is complete.

## M7 — Opaque composition, G-buffer, interaction

Objective: deliver a complete interactive opaque SVO presentation.

### Tasks

- [x] Choose a baseline-valid three-MRT/32-color-byte G-buffer plus separate
  reversed-Z depth and optional non-MRT diagnostic sidecar, with explicit
  depth/normal/velocity quantization and temporal-key rules.
- [x] Implement full-screen dry-scene visibility at the water pipeline's
  explicit internal scale.
- [x] Add environment miss shading.
- [x] Resolve nearest static, dynamic-rigid, and opt-in structural coarse-fluid
  primary hits; full water media traversal remains M9 work.
- [x] Implement a shared bounded picking oracle/adapter over exact primitives,
  terrain, thin glass, and structural coarse fluid, plus production one-pixel
  GPU dispatch through a reusable three-slot asynchronous readback ring.
- [x] Define exact owner mapping to existing rigid IDs while environment,
  fluid, terrain, and no-owner hits remain noninteractive.
- [x] Preserve body position, camera-facing drag plane, and the exact picked
  surface grab offset across SVO and raster picking paths.
- [x] Implement owner-based selection as a material override.
- [ ] Composite scientific overlays over either renderer.
- [ ] Preserve secondary-particle composition or log an accepted deferral.
- [ ] Add split-screen and depth/normal difference views.
- [x] Repaint paused scenes on camera/selection and renderer-mode changes.
- [x] Apply the same conservative BVH rejection to primary, picking-identity,
  and hard-shadow queries, with deterministic lower-index equal-depth ties and
  dynamic-body AABB rejection before exact body intersection.

### Tests and gates

- [x] G-buffer miss/hit encoding, format-byte-budget, quantization, identity,
  reconstruction, temporal-key, and hard-feature precision contract tests.
- [ ] Visible owner equals picked owner and hit position/normal pass tolerances.
- [ ] Dragging works for every body shape.
- [x] Selection repaints paused scenes.
- [ ] Overlays and difference views are deterministic.
- [ ] Complete opaque scene uses no analytic presentation geometry.
- [ ] Opaque depth/material/owner parity passes.
- [ ] Evidence entry `E-M7` is complete.

## M8 — Shared PBR lighting and SVO visibility rays

Objective: shade authored materials rather than illuminated cubes while keeping
raster and SVO response consistent.

### Tasks

- [x] Define a 96-byte direct-index material ABI with base/emission,
  roughness/metallic/specular/IOR/transmission, absorption/scattering,
  thin-wall/closure flags, stable ID/revision, and material-function ID, and
  consume it directly at production binding 6; the compact 32-byte table is
  retained only by raw-voxel/brick-grid inspection.
- [x] Gate production encoding on an exact material publication contract
  (96-byte stride, bounded uint32 count/revision, sufficient binding size),
  validate direct-index identity/revision/opaque flags in WGSL, and expose a
  typed raster fallback when the publication is absent or malformed.
- [x] Implement energy-conserving Lambert diffuse, GGX, height-correlated
  Smith visibility, and Schlick
  Fresnel.
- [x] Define initial geometric/shading normal rules for smooth primitives and
  hard box/cylinder features.
- [x] Define raster-derived image-free diffuse irradiance and roughness-
  prefiltered specular/key-light data for every environment in a revisioned
  96-byte CPU/WGSL contract, publish the selected record, and consume diffuse,
  roughness-prefiltered opaque, miss, and thin-glass reflection response in
  production.
- [x] Publish a scene-specific, revisioned 112-byte directional/point/sphere-
  area/rectangle-area light table with deterministic capped emissive-fixture
  selection, and consume up to eight deterministic direct-light samples in
  production without exceeding the ten-storage-buffer adapter limit.
- [x] Bind authored night-lab/research-station monitor, task-lamp, troffer, and
  indicator emission to same-owner local lights with exact surface position,
  color and radiance; explicit face-direction tags keep screens room-facing
  and troffers downward-facing, while low-power decorative emitters carry a
  documented surface-only exception.
- [x] Integrate one hard SVO shadow ray with conservative opacity across
  authored primitives, analytic terrain, dynamic rigid bodies, and the SVO
  payload fallback. Clip directional light to the scene exit, bias by the
  geometric-normal-projected anisotropic cell width, fail closed, and permit
  at most four bounded thin-pane transmission events.
- [x] Add temporally stable two-sample soft sphere/rectangle-area shadows with
  a shared eight-sample hard upper bound; stochastic temporal sampling remains
  intentionally deferred until M10's history resolve exists.
- [x] Add a production-compiled, explicitly gated short-range contact-
  visibility term over the shared bounded visibility traversal: two stable
  feature-oriented hemisphere rays, a cell/scene-scaled finite radius,
  smooth/hard-feature origin biases, fixed per-ray work ceilings, and
  fail-closed invalid/exhausted behavior. It attenuates indirect diffuse only
  and adds no binding. The shipped gate remains off until live enabled-path
  timing and visual acceptance prove it fits the dry-scene budget.
- [x] Preserve procedural variation through stable material-function IDs:
  exact garden terrain remains its dedicated policy, while authored shells and
  props use continuous seeded world-space architectural, wood, stone, foliage,
  ceramic, brushed-metal, or organic base-color/roughness variation. The
  binding-free policy leaves emission, metallic/specular inputs, owner/material
  identity, and raster output unchanged.
- [x] Keep current shared lighting linear HDR and apply display transfer once.
- [x] Use the same current lighting closure in raster compatibility and SVO
  dry-scene presentation; full PBR expansion remains open.

### Tests and gates

- [x] White-furnace/energy, Fresnel, roughness, metallic, and emission CPU
  tests.
- [x] Every visible emissive environment owner resolves to a finite non-black
  direct-index material and a matching stable light, or to an explicit audited
  low-power surface-only exception; fixture position/orientation/color/power,
  light/material revision, and bounded-table coverage are deterministic.
- [x] Sphere/ellipsoid highlight continuity and cube face/edge reflection
  tests use exact analytic hits and the shared CPU lighting mirror.
- [x] Known blocker visibility and closed-wall leak tests cover every non-open
  authored shell element in conservatory, night-lab, and research-station.
- [x] Raster/SVO closure parity and display-transfer source/CPU tests prove
  both dry paths remain scene-linear and the final compositor applies the
  single shared tone-map/gamma transform exactly once.
- [x] Smooth/sharp normals produce the intended distinct response: adjacent
  curved samples remain continuous while cube edges retain axis-only,
  discontinuous face normals.
- [x] No direct-light leak crosses an opaque closed authored wall in the exact
  primitive/visibility oracle.
- [ ] Evidence entry `E-M8` is complete.

## M9 — Water and glass media

Objective: replace raster interface composition with bounded direct medium
traversal in SVO mode.

### Tasks

- [x] Define air, water, glass, and solid media and transition rules.
- [x] Trace the nearest fluid entry or inside-fluid exit from structural coarse
  `phi` for both the opaque diagnostic and gated production direct-media branch.
- [x] Compute dielectric Fresnel and Snell refraction.
- [x] Traverse water to fluid exit or opaque solid contact in the binding-free
  CPU/WGSL oracle and production dry-shader callbacks; exact refracted
  behind-water opaque shading remains open.
- [x] Apply Beer-Lambert absorption and bounded single in-scattering in the
  direct-media oracle and gated production environment-radiance composition.
- [x] Handle total internal reflection.
- [x] Implement first-pass production thin-walled glass with exact finite-pane
  hits, two-sided normals, custom IOR, Schlick Fresnel, absorption, stable
  identity, one collinear transmitted-scene result, and environment fallback.
- [x] Define the bounded thick dielectric sphere/ellipsoid record, CPU/WGSL
  entry/exit/normal oracle, Beer/Fresnel/TIR handoff, stable authored
  globe/station-lens publication, and malformed/stale fail-closed status.
- [ ] Bind thick dielectric publications and replace the legacy vessel
  compositor with authoritative direct medium traversal.
- [x] Define ordering at coincident/near-contact surfaces.
- [x] Terminate submerged solid contacts as solid.
- [x] Bound reflection, transmission, and internal-transition rays.
- [x] Use environment fallback for bounded first-pass thin-glass reflection and
  transmitted primary misses; full fluid secondary-limit fallback remains open.
- [x] Leave raster optics unchanged in raster mode.
- [ ] Explicitly defer caustics unless a bounded requirement is accepted.

### Tests and gates

- [x] CPU Fresnel, Snell, Beer-Lambert, and TIR mirrors.
- [x] Thin-pane two-sided/grazing, custom-IOR/absorption, identity, environment
  fallback, one-query cap, shadow-transmission cap, and CPU coincidence tests.
- [x] Thick-glass sphere/ellipsoid outside, inside, tangent, rotation,
  owner/material/revision, Beer/Fresnel/TIR, malformed/stale CPU tests and
  binding-free Naga validation pass; the opt-in Metal numeric oracle is added.
- [ ] Quantitative CPU thickness, underwater/submerged-contact, TIR,
  glass/water coincidence, stale/nonresident, and budget tests pass; equivalent
  production frames and full GPU-budget evidence remain open.
- [ ] Thickness comparison against raster interface buffers passes.
- [x] No optical loop is unbounded.
- [ ] Thin glass remains stable at grazing angles.
- [ ] Raster optics remains stable as the sole water renderer.
- [ ] Evidence entry `E-M9` is complete.

## M10 — Temporal stability and bounded indirect lighting

Objective: stabilize stochastic work across camera, rigid, fluid, and topology
motion.

### Tasks

- [x] Publish previous/current camera position and orthonormal projection bases
  plus GPU-resident previous/current rigid transforms to the production resolve.
- [ ] Generate fluid surface motion from published velocity with documented
  limitations; static motion is zero.
- [x] Key history by depth, geometric/shading normal, material, owner, medium
  transition, and local topology generation.
- [x] Define deterministic rejection for disocclusion, reprojection/motion
  invalidity, identity/medium changes, sharp-normal changes, and affected local
  topology history, and run those exact gates in production for static and
  bounded rigid-valid motion.
- [x] Add bounded first/second luminance moments and signed validity/sample
  counters; decorrelated sampling remains open.
- [x] Add a 3x3 current-frame neighborhood clamp plus two-sigma history clamp;
  a wider edge-aware filter remains intentionally absent.
- [x] Audit the production resolve's exact work/resources and remove the
  redundant center fetch from the 3x3 clamp (nine to eight neighborhood loads
  per accepted pixel); reject invalid moments and exact
  identity/media/generation mismatches before fetching or decoding the
  remaining previous history.
- [ ] Stabilize soft shadows before indirect light.
- [ ] Add at most one stochastic diffuse bounce initially.
- [ ] Add bounded glossy reflection only if budget permits.
- [x] Define 64-sample accumulation and paused-stability counters, saturating
  storage at 255; debug visualizations remain open.

### Tests and gates

- [ ] Camera reprojection, moving rigid, fast fluid, topology activation,
  identity change, and paused convergence tests; deterministic CPU rejection
  plus production camera/static-valid shader and ping-pong lifecycle fixtures
  have landed, while real-GPU motion captures remain.
- [ ] Cube edges and thin geometry survive filtering.
- [ ] No persistent trails follow bodies or fluid.
- [x] The history contract retains unchanged regions across local topology
  updates; the production resolve and a real-Metal nonzero-HDR/stable-history
  readback have landed, while moving-edge captures remain open.
- [ ] Indirect lighting remains fixed-bounce/fixed-budget.
- [ ] Evidence entry `E-M10` is complete.

## M11 — UI, URL state, fallback, observability

Objective: expose the migration safely and make failure/performance attributable.

### Tasks

- [x] Add `SvoRenderMode = "svo" | "raster"`.
- [x] Keep the existing smooth/raw-voxel/brick-grid inspection control
  orthogonal to production renderer selection.
- [x] Migrate existing UI state without conflating water optics and scene
  representation.
- [x] Make the hybrid direct-SVO-dry plus raster-water presentation the URL/UI
  default; serialize only the explicit compatibility selection as
  `render=raster`.
- [ ] Add development-only split/difference modes.
- [x] Keep raster user-selectable and automatic as a typed fallback while the
  default uses SVO only for the dry-scene replacement and retains current
  raster water extraction/interfaces/optical composition.
- [x] Fall back on pipeline compilation, missing/unpublished structural data,
  unsupported terrain, adapter-limit, allocation, or fatal initialization
  failure.
- [x] Display the effective renderer and typed automatic fallback reason in the
  Render panel.
- [ ] Timestamp publication, topology, visibility, direct light, shadows,
  media, indirect light, temporal resolve, and upscale.
- [x] Split the production SVO dry visibility/shading pass from temporal
  resolve/copy timing, include temporal cost in total GPU render time only when
  encoded, and expose a default-preserving `svoShadowVisibility=0` diagnostic
  gate to isolate primary versus shadow cost without changing shipped output.
- [x] Fence asynchronous render-timestamp readback by the complete renderer
  mode plus a monotonic context epoch, reset per-stage fallbacks on a mode
  change, and exclude older raster/SVO epochs from performance averaging.
- [x] Fence timestamp readback across play/pause transitions, reject all-zero
  resolves as unavailable rather than measured zero-time frames, and request
  at most one paused retry before returning to on-change idle.
- [ ] Report active/dirty/retired/overflowed bricks.
- [ ] Report node visits, DDA steps, root iterations, secondary rays, history
  rejection, memory categories, and generation.
- [x] Add `benchmark:svo` with warmups, alternating A/B, timestamps,
  distributions, metadata, and raw JSON.
- [x] Reset timing context when render/inspection mode changes.
- [ ] Implement the one-shot, epoch-fenced stage/resource capture contract in
  `docs/GPU_STAGE_DIAGNOSTICS_CAPTURE_PLAN.md`: clean timings remain
  authoritative, resource versions are deduplicated, previews/reductions run
  on the GPU, and only bounded artifacts are read back asynchronously.

### Tests and gates

- [x] UI accessibility, URL round-trip/migration, and inspection combinations.
- [ ] Forced pipeline and adapter-limit fallbacks.
- [x] External-observation timing decode, stale-sample/reset rejection, exact
  frame continuity, effective-renderer/fallback validation, and raster/SVO
  scene/camera/solver-state equivalence.
- [x] Raster can always be selected; automatic fallback is retained (visible
  fallback-reason UI remains open).
- [ ] Every material stage has timestamp/diagnostic coverage.
- [ ] Evidence entry `E-M11` is complete.

`npm run benchmark:svo -- --revision <immutable-revision> --adapter-id
<adapter-id> --reset-token <unique-session> --not-before-ms <unix-ms>` emits a
deterministic external capture plan. Four A/B pairs per acceptance case are the
default, alternating raster-first and SVO-first order. Each run requires 30
warmup plus exactly 120 measured frames and a distinct reset token. Aggregation
retains every raw frame and reports per-run and combined p50/p95/max CPU frame,
GPU total presentation, GPU dry-scene, GPU SVO-temporal, and
renderer-owned-memory distributions, plus paired ratios, adapter limits,
revision, quality, output/internal resolution, timestamp context/epoch/sample
identity, requested/effective renderer, and fallback identity. Capture must
append a frame only when `renderTimingSampleId` advances.

The focused dry-scene comparison is generated with `--quality balanced --case
dam-break--bodies --raster-resolution <WxH> --svo-resolution <same-WxH>`.
Raster and SVO must use the same canonical checkpoint and camera; raster records
the temporal stage as explicitly idle while SVO records it independently from
scene and total presentation cost.

This is deliberately not an automated browser capture. `--observation` accepts
the typed externally captured bundle and rejects missing runs/frames, stale
revision or reset identity, pre-reset/out-of-order samples, adapter/quality or
resolution mismatches, mixed timestamp availability, automatic fallback, and
unequal raster/SVO scene, camera, checkpoint, or renderer-independent solver
state. No performance distribution is claimed until such a bundle is supplied.

## M12 — Default flip and completion

### Completion checklist

- [ ] M0–M11 are complete with evidence.
- [ ] Every shipped environment has complete interactive SVO coverage.
- [ ] SVO mode contains no legacy raster dry-scene presentation or mesh
  extraction; exact implicit primitives remain first-class SVO terminals.
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
| Exact primitive depth | max/p99 world error | `max(1 mm, 0.05 h)` |
| Exact primitive normal | max/p99 angle | 1 degree away from declared singularities |
| Sampled fluid depth | p50/p95 finest-cell error | coarse `max(2 mm, 0.35 h)`; fine `max(1 mm, 0.20 hFine)` |
| Sampled fluid normal | p50/p95 angle | 8 degrees coarse; 4 degrees fine |
| Fine/coarse continuity | uncovered/duplicate pixels | zero above declared epsilon |
| Owner/material identity | valid-hit mismatches | zero |
| Dynamic retirement | stale occupied/material cells | zero after hysteresis |
| Generation safety | partial generations | zero |
| Water thickness | p50/p95 versus reference | `max(5 mm, 0.50 h)` and transmittance error <= 0.03/channel |
| Energy conservation | furnace excess | no energy gain above the M8 white-furnace tolerance |
| Temporal history | retained invalid pixels | zero across failed depth/normal/identity/generation gates |
| Traversal safety | silent invalid access/exhaustion | zero |
| CPU render readback | per-frame control decisions | zero |
| Balanced frame time | median/p95 by adapter | visibility + direct light p95 <= 1.0 ms; total p95 <= 4.0 ms |
| Renderer memory | peak by adapter | balanced <= 192 MiB; high <= 384 MiB; ultra <= 768 MiB |

## Performance rules

- Report SVO and raster internal resolutions with comparisons.
- GPU timestamps are authoritative; CPU encoding remains separate.
- Separate unchanged-frame and changed-topology cost.
- Separate primary, shadow, media, indirect, temporal, and upscale work.
- Report overflow/exhaustion beside timing.
- Retain distributions and raw metadata, not only averages.
- Keep instrumented capture timings separate from clean performance samples;
  resource snapshots explain a result but never establish production speed.
- Treat `GPU busy` as queue-time utilization. Claims about shader-core
  occupancy, bandwidth, cache, or stalls require a matching external hardware
  capture because portable WebGPU does not expose those counters.
- Quality may reduce secondary/indirect work but cannot change simulation state.
- Latest same-paused-frame garden spot check (2026-07-19): the bounded terrain
  solver reduced the SVO scene pass from 5.47 ms to 0.524 ms (about 10.4x),
  versus 0.459 ms raster at the same frame and settings. The remaining gap was
  0.065 ms / 14%; this is a spot measurement, not median/p95 evidence.

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
| R8 | Procedural environments lose detail | medium | stable binding-free world-space material functions now cover terrain, shells, and semantic prop groups; targeted refinement remains | open |
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
| O5 | Primitive candidate packing? | static balanced BVH nodes appended to the existing 64-byte primitive arena for authored catalogs up to 64 entries; larger/generated catalogs retain SVO leaf payload traversal | M5 | decided |
| O6 | Thin glass representation? | oriented thin-walled implicit surface | M6 | open |
| O7 | G-buffer formats/internal scale? | three core MRTs at the active internal render scale: `rgba16float`, `rgba32uint`, `rgba16uint`, plus `depth32float` reversed-Z | M7 | decided; temporal resolution remains open |
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
| 2026-07-19 | D7 | Use direct analytic-table intersection for authored catalogs of at most 64 proxies, retaining SVO payload DDA for larger/generated catalogs; use a capped bracket/secant terrain solver for ordinary rays and a graded fallback for shallow or unresolved rays. | Restarting root traversal across many empty fluid leaves measured about 10 ms; the bounded table path measured 0.066 ms for the default scene versus 0.197 ms raster. In the same paused garden frame, the terrain fast path (at most 12 height evaluations including the normal) plus a 20-step shallow fallback reduced the SVO scene pass from 5.47 ms to 0.524 ms, versus 0.459 ms raster. These are interim primary-hit optimizations until node occupancy summaries provide equally fast spatial rejection; the garden comparison is a spot measurement rather than median/p95 evidence. | M2, M5–M7 |
| 2026-07-19 | D8 | Keep the existing water-compositor `rgba16float` dry target as G-buffer location 0 and renderer-own the two compact identity/surface MRTs plus reversed-Z depth. | This preserves the established HDR/compositor path without a copy while meeting the baseline three-target, 32-color-byte/sample limit exactly. SVO zero-depth misses are decoded alongside raster's retained `65504` compatibility sentinel. | M7, M9–M10 |
| 2026-07-19 | D9 | Default to direct SVO dry rendering feeding the existing raster water pipeline; retain full raster as the explicit toggle and automatic fallback. | Ships the largest dry-scene performance win immediately without replacing the validated raster water interfaces and optics before M9 is complete. | M7, M9, M11–M12 |
| 2026-07-19 | D10 | Bind structural coarse fluid into the production dry renderer now, but keep its opaque primary diagnostic behind `legacy-compositor` / `coarse-opaque-diagnostic` ownership. | This validates the real GPU field, publication, residency, root, and G-buffer path without rendering the same fluid twice. The owner must suppress legacy water interfaces before opting in; raster and the existing SVO-plus-raster-water path remain unchanged until media handoff. | M3, M7, M9 |
| 2026-07-19 | D13 | Use a bounded static balanced BVH for small authored analytic catalogs, stored after primitive records in binding 7, and keep the existing SVO leaf-payload traversal for catalogs above 64 entries. | This removes the primary and shadow full-table loops without adding a fragment storage binding or CPU readback. Conservative float-padded bounds retain thin/rotated features; exact intersections still own normals, identities, and deterministic ties. Missing or corrupt candidate publication produces the typed `missing-primitive-candidates` raster fallback, and invalid/exhausted GPU traversal fails closed. | M5, M7, M8 |
| 2026-07-19 | D11 | Make direct structural-fluid ownership one atomic handoff that disables legacy extraction, interface rendering, and composition together, and fail back to the complete legacy-water path until end-to-end optics are explicitly validated. | Prevents double-rendering and also prevents a half-disabled raster-water pipeline while the bounded direct media path is integrated and measured. | M9, M11–M12 |
| 2026-07-19 | D12 | Compile the direct structural-media branch into the production dry shader and water-pipeline owner now, but require an explicit per-scene end-to-end validation gate before mode word `2` or legacy-stage suppression can activate. | Exercises the real resource, shader, and presentation seams without changing the shipped SVO-dry plus raster-water default or permitting double-rendering. | M9, M11–M12 |

## Evidence ledger

Each entry should name committed artifacts, raw benchmark files, commands,
adapter metadata, and accepted deviations.

| ID | Milestone | Date | Evidence | Result |
| --- | --- | --- | --- | --- |
| E-M0 | M0 | 2026-07-19 | Field-contract audit; `lib/svo-implicit-reference.ts`; `tools/svo-baseline-cases.ts`; `tools/svo-baseline-contract.ts`; `tools/capture-svo-baselines.ts`; `npm run baseline:svo`; paired 26-job raster/default-hybrid plan, canonical bundle paths and manifest schema, raw timing summarizer, observation ingester, explicit sparse layout/adapter/tolerance/budget contracts, and focused tests in `tests/svo-implicit-reference.test.ts` and `tests/svo-baseline.test.ts`; TypeScript, ESLint, water-shader validation, live WebGPU renderer compile, and full unit suite (699 pass, 0 fail, 15 hardware skips) | in progress; the planner/validator/ingester is durable but is not an automated browser capture, so no complete acceptance matrix is claimed; color/timing observations, depth/normal/identity/energy GPU readbacks, second physical adapter evidence, and native solver smoke remain outstanding; tolerances and budgets remain provisional pending those captures |
| E-M1 | M1 | 2026-07-19 | `SparseVoxelStructuralRenderSource`, complete-generation publication, authoritative `fluidResidency` worklist views, producer-owned `inspectionPublication` controller, immutable GPU renderer-consumer binding adapter, renderer-owned owner-page consumer binding, and `SvoFineFluidGpuCapability`; deterministic zero-debug-dispatch plan, paused pending publication, structural/residency/inspection/fine-staging tests, and live smooth/raw/brick toggle; inspection materialization now translates expanded leaf-order records through each leaf's authoritative `topology.y` payload offset with bounds guards; a known-active-voxel GPU readback requires more than 400 material-colored interior pixels and cannot pass from residency lines or tank panes; `tests/webgpu-svo-structural-lookup.test.ts` performs direct Metal readback from the production packed node/leaf/material-owner ABI and rejects stale, unpublished, retired, malformed, and truncated payload publications | in progress; exact structural/worklist ABI, GPU-native renderer request/release compaction, generation-fenced owner-page allocation, and an independently fenced fine-fluid capability backed by read-only authoritative sparse-surface inputs landed while structural publication remains unconditional; cache-busted Reset-to-Front evidence measures the filled dam crop at mean RGB `[35.2, 98.5, 65.4]` with 89.0% green pixels versus `[19.0, 29.3, 28.7]`/3.4% in the empty tank and `[6, 7, 9]`/0% in the background; the hardware structural oracle agrees with CPU identity and world bounds without debug expansion; measured time/memory accounting and live motion/overflow evidence remain |
| E-M2 | M2 | 2026-07-19 | `lib/webgpu-svo-traversal.ts`; `lib/svo-structural-payload-traversal.ts`; CPU and composable WGSL traversal plus binding-free material/owner leaf DDA; `tests/webgpu-svo-traversal.test.ts`; `tests/webgpu-svo-structural-lookup.test.ts`; baseline-browser shader compile, Naga validation, and explicit Dawn/Metal compute readback via `WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js"` | in progress; deterministic two-leaf packed fixtures prove CPU/GPU agreement on anisotropic leaf bounds, node/leaf/payload identity, x-major local voxels, material/owner, entry distance, nearest forward/backward hit, face/corner ties, inside starts, misses, stale/unpublished/retired/malformed publications, and distinct traversal/DDA exhaustion statuses. Conservative occupancy summaries, heat maps, and full-screen validation remain |
| E-M3 | M3 | 2026-07-19 | `lib/svo-fluid-visibility.ts`; `lib/svo-fluid-structural-sampling.ts`; `lib/svo-fluid-structural-visibility.ts`; `lib/webgpu-svo-fluid-primary.ts`; `lib/webgpu-svo-dry-scene.ts`; `lib/webgpu-svo-fine-phi-stager.ts`; `lib/webgpu-svo-fine-phi-resources.ts`; packed topology/geometry/residency/publication-aware coarse lookup, staged fine-page ABI, one-sample owner-tile aprons, generation-stamped manual trilinear and anisotropic-gradient WGSL, production renderer resource ownership and visibility binding, leaf-aware bounded ray/root/inside-exit oracle, focused CPU/mock/real-GPU tests, TypeScript, ESLint, Naga, and live coarse-path Chrome validation | in progress; continuous authoritative coarse `phi` reaches production, while authoritative fine `phi` stages GPU-to-GPU into bounded renderer pages and fails every stale/missing/partial lookup to exact coarse ownership. The fine capability is bound into the production structural sample, root-refinement, and normal path: valid fine data exclusively owns a sample, invalid fine data produces no second hit, fine gradients require all six generation-valid neighbours, and coarse gradients remain the deterministic fallback. The arena mirrors the structural publication in its first eight words and packs fine controls, per-page generations, owner slots, and payload behind aligned offsets, so binding 8 is reused and the fragment stage remains at the baseline ten-storage limit. Coarse/fine root identity uses the existing G-buffer field-source bits with no MRT expansion. CPU seam/ownership values and gradients pass, and Naga validates the complete dry and staging shaders. The production renderer queues `consumer -> owner pages -> fine stager` and mirrors publication even for unchanged fine generations. The padded-scene seam remains `61×46×41` solver cells inside `352×232×296` structural cells at origin `144,8,128`. `directWaterOwnership` remains false, direct fluid stays opt-in, and default water remains raster. Live fine-root depth/normal and overflow captures remain open |
| E-M4 | M4 | 2026-07-19 | `lib/svo-render-residency.ts`; `lib/svo-render-residency-source-adapter.ts`; `lib/webgpu-svo-render-residency-consumer.ts`; `lib/webgpu-octree-owner-pages.ts`; `lib/webgpu-svo-fine-phi-stager.ts`; `lib/webgpu-svo-fine-phi-resources.ts`; `lib/webgpu-renderer.ts`; bounded dirty/residency planner, exact structural-source adapter, CPU consumer/owner/fine-fence oracles, renderer-owned compaction and deterministic owner free list, apron-tile fine payload staging, per-page fine-generation validity, source/owner fence telemetry, indirect dispatch, deterministic retired-tile air scrubbing, and production attach/replacement/device-loss ownership; focused tests, TypeScript, ESLint, and both staging/sampling Naga validation | in progress; immutable producer bindings, activation-before-retirement reuse, complete coarse fallback, and GPU-only fine staging are connected to production visibility without changing raster-water ownership. Solver-local active and retired IDs are remapped to full-scene structural owner IDs before allocation; the stager retains full-scene sampling coordinates and applies the fine-resolution solver origin only when reading source pages. Exact nonzero-origin CPU coverage exercises active allocation, retirement, and bounded page lookup; the live dam-break configuration and malformed refinement/residency/bounds cases are regression fixtures. Publication and fine data now share one aligned renderer arena at the ten-storage adapter limit; unchanged fine generations skip compaction/staging but still refresh the publication mirror required by dry rendering. New presentation bindings and fine resources replace old bindings before solver retirement; source-buffer changes and device loss detach the capability, reverse-order teardown is idempotent, and capability invariants require coarse fallback with `directWaterOwnership:false`. 2:1 renderer refinement, ancestor invalidation, live motion/overflow scaling, and production performance evidence remain |
| E-M5 | M5 | 2026-07-19 | `lib/svo-primitive-abi.ts`; `lib/svo-primitive-motion.ts`; `lib/svo-primitive-candidates.ts`; `lib/webgpu-rigid-body.ts`; `lib/methods/types.ts`; `lib/webgpu-eulerian.ts`; `lib/webgpu-uniform-eulerian.ts`; `tests/webgpu-svo-primitive-exact.test.ts`; 64-byte primitive ABI plus same-index 128-byte GPU-authored motion sidecar; exact descriptor/packed CPU and real-GPU ray hits for sphere/box/capsule/capped-cylinder/ellipsoid with stable feature normals and identities; exact active-axis ellipsoid Euclidean closest-point distance on CPU and bounded WGSL; shortest-arc previous/current transforms, exact rigid surface velocity, conservative swept bounds/preactivation publication, bit-exact command generation, modular revision continuity, roster compaction, fail-closed teleport/discontinuity, and a conservative 64-leaf/127-node static candidate BVH appended to the existing binding-7 arena; focused ABI/ray/exact-parity/GPU/motion/candidate/production integration tests, TypeScript, ESLint, composed WGSL Naga, production build, and cache-busted live Chrome validation | in progress; the resident rigid solver publishes all twelve motion records without readback, while the dry renderer now performs bounded dynamic AABB and static BVH rejection before exact primary and shadow intersections. Smooth, sharp, rotated, subcell, overlapping, equal-depth, every-shipped-catalog, missing-publication, byte-corruption, work-bound, and open-front-shell cases pass; invalid/exhausted traversal fails closed and incomplete candidates report the typed raster fallback. `WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" node --import tsx --test tests/webgpu-svo-primitive-exact.test.ts` passes 2/2 on the local Metal adapter with zero validation errors; deterministic compute readback covers non-unit directions, rotation, inside exits, tangent/grazing, box-corner and cylinder-rim feature ties, all five finite kinds, and invalid dimensions against CPU at `3e-4` tolerances. The combined exact/implicit/ABI/ray/GPU suite passes 30/30; the candidate/status/primary/shadow suite passes 20/20 and the wider candidate-backed dry integration suite passes 41/41; TypeScript, ESLint, isolated and composed production WGSL Naga pass. A fresh default frame reports `Active: Sparse voxels`, visibly renders authored environment and bodies, retains raw inspection drawing, and has zero fresh warning/error logs after startup. Numeric real-GPU sidecar readback, post-candidate live timing distribution, picker/live ghost captures, and distribution evidence remain |
| E-M6 | M6 | 2026-07-19 | `lib/svo-scene-primitives.ts`; `lib/webgpu-svo-dry-scene.ts`; `lib/webgpu-renderer.ts`; `lib/svo-thin-glass.ts`; `lib/svo-scene-glass.ts`; `lib/svo-terrain-material.ts`; `lib/svo-procedural-material.ts`; `lib/svo-scene-coverage.ts`; `tools/report-svo-scene-coverage.ts`; deterministic authored props/panes, analytic garden heightfield, exact raster-parity liner/shore/grass seeded material function, selected-environment PBR material publication, focused scene/terrain/glass/coverage/status/production-material integration tests; ESLint, composed WGSL Naga validation, live Chrome compilation, and same-paused-frame A/B | in progress; authored props, garden terrain geometry plus exact world-space procedural material, correctly oriented container/environment panes, revision-cached pane upload/binding/lifecycle, deterministic visible/collision/lighting ownership reports for every environment and preset, and a four-box night-lab back wall around its now-unblocked city-glass pane landed. The production direct-index table now includes every referenced environment material at stable ID `32 + ownerIndex`; prior out-of-range hits could resolve to invalid black, while regression coverage now requires finite non-black opaque closure data and fail-closed uint16 bounds. Revision two adds stable semantic function IDs and binding-free, seeded, continuous world-space base-color/roughness variation for every room shell plus wood, stone, foliage, ceramic, brushed-metal, and organic props; CPU/WGSL policies are generated from one seed table, primitive/cell seam tests pass, and emission plus material/owner identity remain unchanged. Garden metadata is packed and content-cached in the existing dry uniform with no new binding, and stable region/variation identity has an explicit pending-G-buffer adapter; clean post-atlas-repair garden timing measured 0.459 ms for both SVO and raster dry passes on the same paused build, with no dry-scene shader errors. The research station now has a stable five-primitive observation-port frame/backing plus one finite analytic pane, raising complete coverage from 26 to 32 while curved circular transmission remains explicitly unsupported. Emissive monitor optics, curved glass globes, raster-only foreground effects, conservative collision coverage for subcell cords/shelves, full static-geometry revision caching, and distribution-level performance evidence remain |
| E-M7 | M7 | 2026-07-19 | `lib/webgpu-svo-dry-scene.ts`; `lib/webgpu-svo-fluid-primary.ts`; `lib/webgpu-svo-gbuffer-targets.ts`; `lib/svo-gbuffer.ts`; `lib/svo-picking.ts`; `lib/webgpu-svo-picking-readback.ts`; `lib/webgpu-water-pipeline.ts`; `lib/webgpu-renderer.ts`; `components/WebGPUViewport.tsx`; `lib/webgpu-voxel-debug.ts`; replacement/lifecycle; exact 3-MRT/32-color-byte production allocation and writes plus `depth32float` reversed-Z; stable analytic/terrain/glass/coarse-fluid material-owner-media-feature-generation output; explicit motion validity; bounded exact picking plus a reusable three-slot `MAP_READ` ring copying three aligned one-pixel rows on click only; exact primary-ray metric-depth reconstruction, resource/source-epoch invalidation, identity/generation validation, SVO surface-anchored dragging, raster picker fallback, all four rigid-shape gates, and paused selection repaint; focused tests; TypeScript, ESLint, Naga, and live Chrome shader/pipeline/render validation | in progress; direct dry replacement/lighting/glass, compact production MRT allocation/lifecycle/writes, compositor-compatible zero-depth miss decoding, authoritative shared picking, opt-in structural coarse-fluid G-buffer population, and asynchronous production G-buffer picking landed; raw inspection renders exact occupied records two-sided with bounded Lambert/ambient material preview; fluid is tagged `fluidCoarse` with stable water material plus air/water media and fluid-surface/inside-start flags. The picking/G-buffer suite passes 28/28 with TypeScript, targeted ESLint, and composed Naga validation; live visible owner/normal tolerance, structural fluid velocity, raster compatibility G-buffer writes, live direct-fluid diagnostic visual evidence, and distribution evidence remain |
| E-M8 | M8 | 2026-07-19 | `lib/webgpu-lighting.ts`; `lib/svo-visibility-rays.ts`; `lib/svo-contact-visibility.ts`; `lib/svo-material-abi.ts`; `lib/svo-procedural-material.ts`; `lib/svo-light-abi.ts`; `lib/svo-environment-lighting.ts`; `lib/svo-terrain-material.ts`; `lib/webgpu-svo-dry-scene.ts`; `tests/svo-lighting-surface-correctness.test.ts`; `tests/svo-procedural-material.test.ts`; typed material/light/environment producer publication; production binding-6 consumption of the 96-byte material ABI; renderer-owned 3,696-byte light/environment uniform mirror; production direct PBR/shadows, bounded contact visibility, and procedural garden integration; focused publication/PBR/visibility/contact/material/light/environment/terrain/dry-scene tests; TypeScript, ESLint, composed WGSL Naga, and live Chrome validation | in progress; the dry renderer consumes producer-owned direct-index base color, emission, roughness, metallic, specular weight, and F0 derived from IOR. Exact garden terrain plus seven stable semantic authored-surface functions are selected by the record's material-function ID; continuous world-space uint-seeded value noise varies only bounded base color and roughness, preserves emission/PBR energy inputs and identity, needs no texture/readback/binding, and has deterministic CPU/WGSL policy, seed, bound, revision, and seam coverage; material and lighting publication count/revision/stride/cache identity fail closed before encoding, and record identities/revisions/flags are validated; the light/environment CPU mirrors are rebuilt from the same canonical scene contracts and packed into one uniform binding, so directional/point/sphere/rectangle-area lighting, a deterministic eight-sample direct-light ceiling, stable two-sample area softness, environment diffuse/prefiltered specular, miss radiance, and thin-glass reflection land without increasing the live adapter's ten fragment-storage bindings; night-lab and research-station monitors, task lamp, ceiling troffers, and indicators now publish same-owner lights whose position/color/radiance exactly match their emissive material, with explicit screen/troffer face orientation and audited low-power surface-only exceptions; hard-shadow work skips back-facing samples, rejects dynamic bodies with a conservative world-space sphere before local transforms, and terminates immediately on any exact opaque body, static-BVH, SVO-payload, or terrain blocker, while transmissive panes retain bounded spectral transmission; static candidate traversal has a shadow-only any-hit mode so dense authored catalogs avoid nearest-identity work without changing primary ties; short-range contact visibility reuses the same traversal with two stable bounded rays and open/corner/closed-wall CPU oracles, modulates indirect diffuse only, fails invalid/exhausted work closed, and is production-compiled behind a zero-cost-before-traversal gate that is intentionally off in shipped scenes pending enabled-path timing acceptance; deterministic exact-hit tests now prove continuously moving sphere/ellipsoid highlights, planar axis-only cube-face response with a discontinuous edge, and blocker rejection for every non-open authored conservatory/night-lab/research-station shell element; raster and SVO dry shaders both call the same binding-free scene-linear closure, and `sceneLinearToDisplay`/`unifiedDisplayTransfer` now define the compositor's sole final tone-map/gamma operation; malformed lighting metadata resolves to the typed `missing-lighting-publications` raster fallback without throwing through solver initialization; focused lighting/surface tests, TypeScript, and composed Naga pass; the current native Metal harness compiled the parity shader but returned a zeroed sentinel buffer shared by other native readback oracles, so no numeric Metal parity is claimed for this slice; the last cache-busted Chrome frame reports `Active: Sparse voxels` with no fresh binding-13 or dry-scene errors; post-optimization live timing, enabled contact-visibility visual/timing acceptance, opaque closed-wall visual captures, temporal soft-shadow filtering, numeric native parity after harness recovery, and distribution evidence remain |
| E-M9 | M9 | 2026-07-19 | `lib/svo-media.ts`; `lib/svo-fluid-media-path.ts`; `lib/svo-thin-glass.ts`; `lib/svo-scene-glass.ts`; `lib/webgpu-svo-fluid-primary.ts`; `lib/webgpu-svo-dry-scene.ts`; `lib/webgpu-water-pipeline.ts`; `lib/webgpu-renderer.ts`; authoritative structural-visibility adapter, bounded CPU/direct-WGSL water media path, production structural-water/solid/glass callbacks and environment composition, atomic raster-water stage planner, nearest coarse `phi` entry/inside-exit, primary pane composition and shadow transmission, twelve focused media/integration tests, ESLint, TypeScript, full composed WGSL Naga, and live Chrome default-path validation | in progress; the direct path follows air-to-water entry through a smooth water exit or submerged opaque contact, records thickness and entry/exit normals, applies spectral Beer-Lambert absorption plus bounded single in-scattering, handles Snell/Schlick/TIR under the shared 16-query/8-transition/4-reflection/8-transmission caps, resolves near-coincident water/glass boundaries atomically, and fails closed on stale generation, nonresidency, malformed fields, or exhausted work. The production dry shader now composes those callbacks and writes a coarse-fluid G-buffer surface, while one validated ownership branch renders directly to the presentation texture and returns before legacy extraction, caustics, interfaces, or optical composition; an inconsistent mixed ownership throws. No shipped scene supplies the validation gate, so the required SVO-dry plus raster-water default is unchanged; a fresh Chrome frame reports `Active: Sparse voxels` with no shader/pipeline/WebGPU warnings. Exact refracted behind-water opaque shading/contact identity, thick dielectric support, structural velocity, experimental-path visual captures, raster thickness comparison, and eventual compositor removal remain |
| E-M10 | M10 | 2026-07-19 | `lib/svo-temporal-history.ts`; `lib/svo-primitive-motion.ts`; `lib/webgpu-rigid-body.ts`; `lib/webgpu-svo-temporal-accumulator.ts`; `lib/webgpu-svo-dry-scene.ts`; `lib/webgpu-svo-gbuffer-targets.ts`; compact G-buffer-derived history key, deterministic CPU rejection/convergence and surface-velocity oracles, renderer-owned ping-pong HDR/key/moment targets, previous/current camera bases and rigid transforms, production local-generation/identity/normal/depth/exact-surface-velocity/motion-validity writes, static and bounded rigid-valid velocity reprojection/rejection, 3x3 neighborhood and variance clamps, exact resource/command tests, focused temporal/G-buffer/hybrid tests, TypeScript, ESLint, composed WGSL Naga, production build, cache-busted live Chrome validation, and an explicit Dawn/Metal seeded-HDR/stable-history readback | in progress; production dry-only temporal resolve runs after direct SVO visibility and before legacy raster water. GPU-authored motion is copied into a 1,536-byte renderer-owned uniform mirror, preserving the live ten-fragment-storage limit; owner/material/current-transform/generation validation fails closed, while accepted rigid pixels back-project by packed surface velocity and the published frame delta under the existing world/cell motion ceiling. Live split telemetry measured the temporal stage at roughly `2.8–3.4 ms`. An attempted 8x8 shared-tile compute rewrite compiled under Naga and Metal but zeroed the presentation HDR both live and in a seeded real-GPU readback without emitting a validation error, so it was rejected and fully rolled back; no tiled-compute path is active. The retained fragment path reuses the already-loaded center texel, reducing exact 3x3 neighborhood texture loads from nine to eight per accepted pixel, checks previous-moment validity before the other three history fetches, and rejects exact identity/media/generation mismatches before normal conversion without changing acceptance, clamp, moment, or accumulation semantics. WebGPU forbids sampling the current dry target while attaching it for output, so the current command graph remains one full-screen resolve pass plus one required alias-breaking texture copy; no pass/copy-count reduction is claimed. Logical allocation is exactly `160 + width * height * 64` bytes (`132,710,560` bytes at 1920x1080); driver-private texture alignment is not observable. Focused temporal/G-buffer checks pass 9/9 including nonzero RGB, exact depth, and stable second-frame sample-count readback on Metal; TypeScript, ESLint, Naga, and the production build pass. Quantitative moving-rigid trail/edge captures, fluid velocity, memory/performance distributions after the safe optimization, temporal soft-shadow sampling, debug views, and bounded indirect lighting remain |
| E-M11 | M11 | 2026-07-19 | `lib/svo-render-mode.ts`; `tools/svo-benchmark-contract.ts`; `tools/benchmark-svo.ts`; `npm run benchmark:svo`; typed effective-renderer/fallback status; diagnostics store, default hybrid SVO-dry/raster-water URL state with explicit `render=raster`, Render panel control, viewport integration, lifecycle/fallback tests; deterministic external-observation A/B planning and aggregation with 30 warmups/120 measured frames, alternating order, raw samples, p50/p95/max timing and memory distributions, pair ratios, complete capture metadata, stale-reset/revision/order rejection, and scene/camera/checkpoint/solver-state equivalence; production SVO G-buffer click routing with retained raster fallback and paused selection repaint; independent hardware timestamp intervals for SVO visibility/shading and temporal resolve, with a default-preserving hard-shadow isolation query gate; renderer-mode plus monotonic-epoch readback fencing and history filtering; focused benchmark/timing tests; cache-busted live Chrome raw/brick verification plus quantified Reset-to-Front crop comparison | in progress; requested hybrid default, full-raster toggle, automatic fallback and visible reasons landed; SVO-dry clicks now read the rendered owner/depth/material/generation asynchronously while raster keeps its resident rigid picker, and selection remains part of the paused presentation key. The durable benchmark planner/validator/aggregator now fails closed on incomplete, stale, mismatched, fallback, or inequivalent observations and explicitly records unavailable timestamp queries without inventing GPU values; production diagnostics now report `SCENE` and `TEMPORAL` independently and include temporal cost in render totals only when encoded. Those split counters exposed the current `2.8–3.4 ms` temporal bottleneck and also showed the rejected compute experiment as `TEMPORAL` idle while the viewport was black; the permanent seeded-HDR Metal regression now prevents a shader that merely compiles from being accepted as presentation-safe. Pending timestamp promises are rejected if either renderer mode or epoch changed before resolution; controller fallbacks reset rather than carrying the old mode's numbers, and rolling averages only consume the live epoch. The focused timing/benchmark/averaging/lifecycle/pacing suite passes 27/27; targeted ESLint and `git diff --check` pass. It does not automate browser capture, and no post-optimization distribution evidence is claimed until an external observation bundle is supplied. Paused raw mode visibly draws filled material-colored occupied voxels plus residency outlines, while brick-grid still draws the full sparse topology and smooth mode keeps zero expanded publication work; visibility/direct-light/shadow sub-counters, external benchmark captures, and distribution-level gate evidence remain |
| E-M12 | M12 | — | — | pending |

E-M11 paused-attach addendum (2026-07-19): successful warmed-solver
attachment increments a renderer-owned `presentationRevision` exactly once,
after the SVO source is installed into the already-initialized dry/temporal
renderer. The paused viewport polls that scalar as part of its immutable
presentation key, consumes the change after a submitted frame, and therefore
does not create a render loop. The same one-shot repaint attaches Raw/brick
inspection, services its pending GPU materialization in that command, and then
returns to idle. The focused lifecycle/frame-pacing/inspection/temporal suite
passes 28 tests with one expected hardware skip; TypeScript, targeted ESLint,
and `git diff --check` pass. A clean-server live retest now presents the fresh
paused default SVO geometry without camera interaction, and toggling Raw voxels
immediately draws filled material cells plus inspection bounds.

E-M0/M11 timing-integrity addendum (2026-07-19): benchmark schema version 2
records adapter, output/internal resolution, requested/effective mode, timing
context/epoch, accepted sample ID, and independent scene/temporal/total values
for every raw observation. The validator rejects cached sample IDs, epoch drift,
fallback, unequal camera/checkpoint state, incomplete 30-warmup/120-measured
runs, and raster observations whose explicitly idle temporal stage is nonzero.
`--case dam-break--bodies --quality balanced` produces the focused equal-camera,
equal-resolution dry-scene pair; no distribution is claimed until its external
observation bundle is supplied.

Renderer timing epochs now cover play/pause as well as mode switches, so an
in-flight running query cannot become a paused measurement. An all-zero query
resolve remains unavailable rather than being presented as `0.00 ms`; paused
mode permits one bounded retry and then returns to on-change idle. Conversely,
a paced callback skipped because a prior presentation is pending preserves the
current context, epoch, accepted sample ID, and cached timing instead of
publishing a synthetic `:legacy` context that hides active history. The
single-query suppression guard is itself epoch-owned: an unresolved readback
from a previous run state or renderer mode cannot starve the first query in the
new epoch, and its eventual completion cannot clear the new epoch's guard. The
performance panel exposes those identities as capture-readable attributes and
shows `awaiting timestamp…` until a positive hardware interval is accepted.

Clean live transition evidence first showed awaiting state, then one paused
retry published `0.33 ms` total with `SCENE 0.131 ms` and `TEMPORAL idle`;
resuming for more than 120 history frames retained active telemetry at
`SCENE 0.524 ms` and `TEMPORAL 0.262 ms`. Earlier fresh paused same-camera spot
checks measured raster at `0.39 ms` total / `0.131 ms` scene and SVO at
`0.26 ms` total / `0.131 ms` scene. These are lifecycle spot checks, not
p50/p95 claims.

E-M6 addendum (2026-07-19): `lib/svo-static-publication-cache.ts` now backs
bounded content caches for primitive, glass, direct-index material, authored
light, environment coverage, and shipped coverage publications. Unchanged
world rebuilds reuse immutable packed arrays by identity; source revision
changes invalidate before repacking. Exact analytic cords, shelves, controls,
and the station port backing also publish conservative audit/collision-proxy
bounds below 1.5 nominal cells. Focused identity/invalidation and
owner/material/collision coverage checks pass 47/47 with TypeScript green.
Curved emissive globes and the station observation lens now have a separate
authored analytic thick-glass contract, but remain degraded/unsupported until
its production media binder is explicit. Remaining exact M6 scene gaps are
that binder, emissive-display optics, explicitly raster-only foreground
effects, default-camera quality sweeps, and distribution evidence.

E-M9 addendum (2026-07-19): `lib/svo-thick-glass.ts` and
`lib/svo-scene-thick-glass.ts` add a separate bounded 80-byte analytic
sphere/ellipsoid volume ABI and deterministic authored publication for three
conservatory globes, the night-lab bulb, and one elliptical station observation
lens. CPU/WGSL mirrors provide exact oriented entry/exit roots and outward
normals, inside/tangent handling, Beer absorption plus Fresnel/refraction/TIR
handoff, stable owner/material/glass/revision identity, and typed malformed or
stale failure. Focused CPU/coverage tests and standalone/composed Naga pass; an
opt-in Metal numeric oracle covers hit distances, normals, tangent/inside,
stale, and malformed records. The remaining production seam is a renderer-owned
bounded buffer/binder, nearest-boundary composition with water/thin panes,
G-buffer medium identity, and validated replacement of the existing opaque
globe proxies/thin station pane. Until then the new records are explicitly
`deferred-media-binder` and the shipped SVO-dry plus raster-water path is
unchanged.

## Final definition of done

The migration is complete when direct SVO rendering is the default; every
shipped scene is complete and interactive through it; fluid and curved
primitives remain smooth; sharp solids remain sharp; realistic lighting and
bounded media pass quantitative gates; dynamic publication is generation-safe;
simulation evidence is unchanged; and raster compatibility still passes its
retained suite as a selectable and automatic fallback.
