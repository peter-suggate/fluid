# Live Empty-Scene Authoring Acceptance Plan

## Goal

Start with an empty bounded scene and author it while the viewport keeps
rendering: add, edit, move, and remove terrain, rigid bodies, lights, and fluid.
The same SVO topology and fixed GPU arenas remain attached throughout. There is
no scene-content bake, solver restart, or render-world replacement.

"Instant" means the exact raster/analytic path shows the accepted edit in the
first submitted presentation frame. The SVO is a mutable acceleration cache: it
repairs only affected bricks/pages, and dirty data is unavailable until its
generation completes.

## Shared authority

One mutable SVO owns addressing, topology, terminal lifecycle, and generation
publication. Payload ownership is split into non-overlapping lanes inside the
same arena:

- live scene SDF/coverage and scene material/owner;
- dynamic-solid, fluid SDF/fraction, pressure, and fluid material/owner;
- velocity;
- derived opacity and radiance pages with per-page validity.

Scene editing and physics may update the same terminal in one frame, but never
write the same lane. Terminal occupancy is the conservative union of all lanes.
The final publication fence advances only after every scheduled lane write and
derived invalidation is ordered.

## Authoring transaction

Every edit produces a monotonically increasing `SceneRevision` and a compact
delta containing stable object IDs plus old/new AABBs.

1. Publish exact primitive, material, lighting, and BVH arenas immediately.
2. Convert old/new AABBs to a deduplicated brick mutation worklist.
3. Activate missing topology paths from fixed node/leaf arenas.
4. Mark affected terminals `DIRTY | QUEUED`; traversal rejects them.
5. Recompose only the scene payload lane using brick-local candidates.
6. Update conservative union occupancy without touching fluid payload.
7. Invalidate and rebuild only affected opacity/radiance pages.
8. Clear lifecycle flags and advance the completed generation only if every
   capacity and completion receipt succeeds.

While steps 2-8 run, exact raster/BVH traversal is authoritative in dirty
regions. An overflow remains visible as a diagnostic and cannot publish a false
complete generation.

## Acceptance sequence

### 1. Empty scene

- Create a scene with bounds, resolution, and explicit capacity budgets, but no
  terrain, bodies, or fluid.
- The first viewport frame renders without waiting for SVO content work.
- SVO buffers are allocated once; terminal and derived page counts may be zero.
- Camera, lighting, and environment edits do not reconstruct the world.

### 2. Terrain

- Add a terrain primitive; it appears in the next submitted frame.
- Sculpt continuously at pointer rate. Each event carries only the old/new
  sculpt bounds; unchanged bricks and pages receive no work.
- Move, resize, change material, and remove terrain.
- Verify fluid lanes are byte-identical when no fluid operation was requested.

### 3. Rigid bodies

- Add bodies into previously empty regions, forcing absent-leaf insertion.
- Move and rotate bodies continuously; exact bounds refit immediately and the
  SVO converges behind them.
- Change shape/material/motion mode and delete bodies.
- Run rigid motion while terrain is edited. Confirm topology is shared and scene
  and dynamic-solid payload writes remain disjoint.

### 4. Fluid

- Enable the fluid system without replacing the scene SVO.
- Add a fluid volume, then an inflow and outlet, through the same revision/delta
  protocol used by other scene content.
- Seed and evolve fluid only in the fluid/velocity lanes of existing or newly
  activated terminals.
- Edit terrain and move bodies while fluid advances. Collision/coupling consumes
  one completed topology generation; rendering consumes current scene and fluid
  lanes from that same structure.
- Pause physics and continue editing; scene maintenance must still run from the
  presentation encoder.
- Remove all fluid and continue authoring without reconstructing the SVO.

## Capacity policy

An empty scene declares an editable bounds/resolution envelope and explicit
budgets for nodes, leaves, primitives, dirty bricks, derived pages, bodies, and
fluid pages. These are capacity declarations, not scene content and not a bake.

The acceptance scene chooses budgets large enough for the complete scripted
session. Exceeding a budget must:

- keep the last completed SVO generation;
- continue exact rendering of the current scene;
- expose which arena overflowed and the requested/available counts;
- never silently omit content or trigger an implicit blocking rebuild.

Later, capacity growth can use a background candidate arena and atomic
generation swap, but it is not allowed to masquerade as an instant in-place
edit.

## Performance gates

- No GPU allocation, pipeline creation, queue fence, or CPU readback after the
  empty scene attaches.
- Unedited frames encode zero scene-maintenance work.
- Edit cost scales with dirty bricks/pages and local candidates, not total scene
  primitives, total leaves, or total fluid volume.
- Exact fallback covers dirty regions only.
- Record p50/p95 CPU encode time, GPU scene-maintenance time, dirty bricks,
  inserted leaves/nodes, rebuilt derived pages, candidate overflow, and exact
  fallback pixels.
- Pass target: continuous single-object manipulation does not reduce the
  established presentation frame rate; maintenance stays within a configurable
  per-frame budget and never exposes stale geometry.

## Automated checks

- CPU contract tests for delta coalescing, stable IDs, capacity receipts, and
  lane ownership.
- Metal/WebGPU tests for absent-leaf insertion, simultaneous scene/fluid writes,
  deletion cleanup, dirty lifecycle rejection, and derived-page validity.
- A scripted empty-to-terrain-to-bodies-to-fluid run that asserts buffer
  identities remain unchanged throughout.
- Pixel probes during edits proving either the new exact result or the completed
  SVO result is visible—never the superseded scene.
- A performance capture comparing idle, continuous transform, terrain sculpt,
  and coupled fluid/body/terrain editing against the current reference frame.

## Delivery phases

### A. Empty document and edit transaction

- Add `createEmptyScene(bounds, resolution, capacities)` as the only new-scene
  constructor. It creates document identity and resource budgets, not geometry.
- Give every authored entity a stable ID. Route create/update/delete through one
  `SceneEditTransaction` containing the new scene revision, changed IDs, and
  old/new bounds.
- Keep the current renderer and solver attached when a transaction is accepted.
  Remove any editor action that substitutes a preset scene or restarts the
  solver as a way of applying content.
- Add an editor action log so the scripted acceptance run and undo/redo exercise
  the exact same transaction path as pointer tools.

### B. Exact live render authority

- Hot-publish primitive/BVH, material, light, glass, and rigid-motion records
  into fixed-capacity renderer arenas.
- Make the accepted render revision visible in the first submitted frame,
  including while physics is paused.
- Report capacity rejection per arena and retain the prior complete acceleration
  generation; never retain superseded exact scene content.

### C. Mutable terrain field

- Replace analytic-feature-only terrain publication with a GPU height/SDF field
  owned by the scene transaction. The raster path, SVO scene-lane voxelizer,
  collision sampling, and fluid boundary conditions all sample that one field.
- Brush edits publish rectangular texel ranges and old/new world bounds. They
  invalidate affected SVO terminals before rebuilding them.
- Do not mark derived opacity or radiance pages valid until their terrain field
  revision matches the transaction. This is the required gate before enabling
  interactive terrain tools in the empty-scene test.

### D. Bodies and topology growth

- Route body placement, transform, shape/material change, and deletion through
  the same scene transaction.
- Exercise fixed-arena node/leaf insertion in previously empty coordinates and
  reclaim terminals only when the conservative union of scene and fluid lanes
  is empty.
- Refit the exact BVH immediately; rebuild optional traversal accelerators from
  GPU dirty worklists, never from a replacement world.

### E. Fluid lifecycle on the shared structure

- Treat fluid paint/erase, sources, drains, and boundary-condition edits as
  stable-ID transaction records with dirty bounds.
- Activate topology through the same mutation worklist, then write only fluid
  and velocity lanes. Terrain and bodies write only their scene/dynamic-solid
  lanes.
- Schedule coupling after all input lanes for a revision are complete. Publish
  one generation receipt consumed by rendering, collision, pressure, and fluid
  advection.

### F. Scripted acceptance harness

- Start from `createEmptyScene`, attach once, and record all GPU buffer and
  texture identities.
- Replay: add/sculpt terrain; add/move/delete bodies; paint/erase fluid; add an
  inflow; edit terrain and bodies while fluid advances; pause and keep editing;
  remove everything.
- After every transaction, probe exact pixels immediately and wait only in the
  test harness for SVO convergence. Assert unchanged lanes and resources by
  identity/hash, then enforce the performance gates above.

## Cutover rule

The empty-scene workflow becomes the primary application path once phases A-F
pass. Preset scenes become ordinary transaction scripts that populate an empty
document. No preset-only initialization, fluid-free render world, compatibility
publication, or scene-content bake remains on the shipping path.
