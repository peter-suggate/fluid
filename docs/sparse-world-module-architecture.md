# Sparse world module architecture

Status: proposed production boundary, 2026-08-24.

## Decision

Sparse simulation is one module, not a collection of shader entry points exposed
through the solver.

The module owns compilation, physical tile residency, topology mutation, frame
execution, pressure data, presentation publication, and detailed diagnostics. Its
public contract describes only a sparse world's lifecycle:

1. acquire a device-wide implementation;
2. create a world from resident seed tiles and a physical capacity policy;
3. encode steps;
4. obtain an accepted presentation view and a compact status receipt;
5. destroy the world.

World extent is metadata. No allocation, compile key, loop bound, or work queue may
be sized from logical-domain volume or finest-domain volume. Empty coordinates have
no record.

## Public API

The public package is `lib/sparse-world`. It exports semantic types, not pipeline
names, arenas, phases, indirect buffers, or shader modules.

```ts
export interface SparseWorldDevice {
  createWorld(config: SparseWorldConfig): Promise<SparseWorld>;
  readonly status: "loading" | "ready" | "fault";
  readonly fault?: SparseWorldFault;
}

export interface SparseWorld {
  edit(edit: SparseWorldEdit): SparseWorldEditReceipt;
  encodeStep(encoder: GPUCommandEncoder, input: SparseWorldStepInput): SparseWorldStep;
  presentation(): SparseWorldPresentation;
  status(): SparseWorldStatus;
  destroy(): void;
}

export interface SparseWorldConfig {
  readonly profile: "cm12-b8";
  readonly capacity: SparseWorldCapacity;
  readonly seeds: readonly SparseTileSeed[];
  readonly boundaries: SparseWorldBoundaries;
  /** Optional simulation bounds; never an allocation dimension. */
  readonly extent?: SparseWorldExtent;
}

export interface SparseWorldStepInput {
  readonly time: number;
  readonly dt: number;
  readonly gravity: readonly [number, number, number];
  readonly rigidBodies?: readonly RigidBodyState[];
  readonly liquidInflow?: SparseWorldLiquidJet;
}

export interface SparseWorldStep {
  readonly generation: number;
  readonly submittedTime: number;
}
```

Application features submit semantic edits to the world. The edit union covers
transient fluid additions and complete authored-scene revisions. Rigid-body roster
changes and semantic hose boundary conditions travel with the authoritative step input.
Callers do not translate metres to cells, derive solid occupancy, pack refinement
controls, or invoke implementation passes.

`SparseWorldPresentation` is the renderer's stable, read-only resource contract. It
contains the accepted generation and the few GPU bindings needed to extract the
surface. It does not expose topology construction, pressure, activity, allocator,
or candidate-generation resources.

`SparseWorldStatus` is intentionally small:

```ts
export interface SparseWorldStatus {
  readonly state: "ready" | "running" | "saturated" | "fault";
  readonly acceptedGeneration: number;
  readonly residentTiles: number;
  readonly capacityTiles: number;
  readonly lastAcceptedTime: number;
  readonly fault?: SparseWorldFault;
}
```

The normal UI may show only `loading`, `ready/running`, `capacity reached`, or a
stable fault code. Pipeline labels, queued jobs, phase names, arena offsets, and
internal receipts belong to a developer trace, not application state.

## Internal composition

The module has six internal owners.

### 1. Device library

`SparseWorldDeviceLibrary` owns the scene-independent shader and pipeline bundle.
It is acquired once per GPU device and profile.

- Its key is numerical profile plus device features, never scene dimensions,
  capacity, resident count, or coordinates.
- Layout offsets and capacities are read from runtime headers.
- Construction compiles one predefined family. Creating another world clones
  resources and uploads seeds; it does not generate WGSL.
- Future predefined topology programs live here. Runtime tile cloning copies a
  descriptor and patches neighbor handles; it never compiles a seam.

The library may internally use many compute pipelines during migration. That is an
implementation detail behind one readiness promise and one failure value. The
target is a handful of transform pipelines selected by constants or command data,
not one externally named pipeline per bookkeeping phase.

### 2. Physical tile store

`TileStore` owns all capacity-shaped storage:

- coordinate-to-slot hash directory;
- free and retired slot queues;
- complete cell, face, boundary, and presentation pages;
- sparse pressure aggregate and hierarchy pools;
- accepted/candidate generation selectors.

Every array is sized from explicit physical capacity. A coordinate exists only when
the accepted directory maps it to a slot. Signed coordinates are never linearized
through world dimensions.

A physical tile page is a complete unit of simulation. It contains fields and all
local execution descriptors required to run transport, pressure, projection,
activity, and presentation. A geometry-only or presentation-only resident tile is
invalid.

### 3. Topology transaction

`TopologyTransaction` is the only owner allowed to create, rerung, or retire tiles.
It consumes a compact set of intents produced by the previous step:

```text
existing coordinate -> update/rerung intent
absent coordinate   -> clone intent
unreferenced tile   -> retirement intent
```

One transaction performs compact deduplication, capacity preflight, slot allocation,
field initialization/transfer, neighbor and pressure patching, validation, and one
accepted-generation flip. Failure publishes none of the candidate generation.

Changed tile IDs are appended directly to one compact transaction list. No bitmap
tree, domain scan, global owner plane, or consumer-specific topology journal is
allowed.

### 4. Frame executor

`FrameExecutor` owns the fixed frame graph. Its semantic stages are:

```text
predict -> transport -> assemble/solve pressure -> correct/classify
        -> derive topology intent -> transact -> publish presentation
```

Stages operate on the accepted physical tile set or on compact changed-tile lists.
They may produce transient packet masks, but they do not create persistent
begin/seal/finalize authorities around individual kernels.

The executor exposes only `encodeStep`. It owns every internal pass break, indirect
argument, binding change, and generation receipt. The solver cannot dispatch an
internal entry point by string.

### 5. Presentation bridge

`PresentationBridge` publishes a read-only view from accepted resident pages. It
cannot allocate physics residency and cannot interpret candidate topology. Its page
count follows resident presentation closure, not logical extent.

The renderer keys extraction by accepted simulation generation. Rendering neither
drives simulation compilation nor observes an incomplete topology transaction.

### 6. Trace and validation

`SparseWorldTrace` is an optional developer-only sink. It may record internal stage
times, pipeline compilation, allocator receipts, generation transitions, and first
fault locations. These details do not appear in `SparseWorld`, `GPUEulerianInfo`, or
the normal transport UI.

Validation is expressed at module boundaries:

- device library ready or faulted;
- transaction atomically accepted or rejected;
- step generation accepted or faulted;
- presentation generation matches the accepted world.

Internal subphase failures map to a stable `SparseWorldFault` code plus an optional
trace record.

## UI facade

Core UI and renderer code receive one optional `SparseWorldUI` companion. It is
split into three semantic ports:

```ts
export interface SparseWorldUI {
  readonly control: SparseWorldUIControl;
  readonly overlays: SparseWorldUIOverlays;
  readonly diagnostics: SparseWorldUIDiagnostics;
}
```

- `control` exposes user actions such as enabling/reseeding tracers and arming a
  pressure film.
- `overlays` exposes only method-neutral, read-only render publications.
- `diagnostics` exposes a bounded physical-storage/world snapshot and explicit
  pressure-film readback.

Adding a UI capability extends one of these ports; it does not add another field
to the generic solver and cannot reveal a CM12 resident, pipeline, queue, stage,
arena, or transaction receipt. Detailed acceptance and QA receipts remain on the
internal developer trace and are never normal UI state.

## Ownership rules

1. The device library owns compilation; a world never calls pipeline compilation.
2. The tile store owns physical addresses; no consumer maintains a second owner
   directory.
3. The topology transaction owns structural mutation; numerical stages emit intent
   only.
4. The frame executor owns ordering; external code cannot invoke sub-stages.
5. Accepted generation is the sole visibility boundary.
6. Presentation is a consumer of accepted pages, never a parallel residency model.
7. Diagnostics observe receipts; they do not become synchronization authorities.

## Scaling laws

For `R` resident tiles, `C` physical capacity, and `D` tiles changed in a step:

| Concern | Required bound |
| --- | --- |
| persistent field/topology memory | `O(C)` |
| coordinate directory | `O(C)` |
| ordinary step work | `O(R)` plus numerical iterations |
| topology mutation | `O(D)` plus changed seams/ancestors |
| presentation | `O(surface/receiver closure)` |
| shader source and pipeline count | `O(1)` per device profile |
| dependency on logical world volume | `0` |

Translating the same resident set to coordinates billions of tiles apart must leave
GPU bytes, pipeline keys, construction work, and per-step work unchanged.

## What disappears from current public architecture

- the resident's map of hundreds of externally addressable pipeline names;
- simulation readiness inferred from a queue of entry-point compilations;
- UI strings for individual shader compilation;
- solver-visible topology, PCM, PEI, PTR, FPL, and allocator phases;
- caller-managed indirect-buffer copies between those authorities;
- scene-shaped WGSL and scene-specific simulation pipeline cache keys;
- global logical-domain owner directories and finest-domain field rasters;
- separate physics and presentation notions of residency.

These concepts may temporarily exist inside the module during cutover, but adding a
new external reference to one is prohibited.

## Cutover plan

### A. Freeze the boundary

Create `lib/sparse-world/index.ts` with the API above. Wrap the existing Sparse CM12
resident behind it. Adapt the solver and renderer to depend only on `SparseWorld` and
`SparseWorldPresentation`. Move detailed compilation progress to `SparseWorldTrace`.

Exit: no UI, solver, renderer, or test dispatches or names an internal pipeline.

### B. Device-wide predefined implementation

Move shader construction, layouts, and pipeline acquisition into
`SparseWorldDeviceLibrary`. Make every layout capacity-independent and acquire the
library before world creation. Remove scene source text from pipeline cache keys.

Exit: a second Long Dam world on the same device performs zero shader/pipeline
compilation; a first world has one bounded library readiness operation.

Current cutover: device/profile readiness and faults now have one cached owner;
worlds, renderer, transport UI, and solver scheduling consume that semantic owner.
The legacy resident still constructs some scene-shaped WGSL behind the owner and
emits that fact only as migration debt on developer trace. Therefore Phase B is
started, but its zero-compilation second-world exit is not claimed yet.

### C. Complete physical tiles

Make `TileStore` the physics address space, not only the clone/presentation address
space. Remove the 1,152-key compatibility atlas, global stable cell/row ownership,
and domain-shaped pressure structures. Seed only the 80 authored-fluid Long Dam
tiles.

Exit: construction and memory depend only on the 80 residents plus configured slab
headroom.

### D. One topology transaction

Route activation, rerung, seam changes, pressure ancestry, and retirement through the
single compact transaction. Delete PCM/PTR-style structural rediscovery and their
begin/seal/finalize pipeline families.

Exit: advancing the front clones complete tiles with `O(D)` mutation work and one
generation flip.

### E. Consolidate the frame graph

Replace consumer-specific catalogues with shared packet masks and predefined local
topology programs. Collapse bookkeeping kernels into their owning transforms.

Exit: compilation is a small scene-independent bundle and the Long Dam first load is
bounded by resource creation/upload, not shader topology.

## Acceptance gates

The production gate uses only the public module.

1. Cold device acquisition and first world creation have explicit time limits and
   run through the same managed Dawn device path as the UI.
2. Generation zero contains exactly the 80 authored-fluid Long Dam tiles.
3. No dry coordinate exists before a swept-front intent requests it.
4. Every requested receiver becomes a complete accepted B8 tile before transport
   targets it.
5. The rendered liquid front advances monotonically and reaches the far end.
6. There are no world, transaction, pressure, or presentation faults.
7. A vast-coordinate translation produces identical allocation sizes and pipeline
   keys.
8. A second world on the same device compiles zero pipelines.

The Dawn gate must create the same managed device capability used by the UI, disable
persistent blob caching for the cold-library test, await `SparseWorldDevice`, run the
Long Dam through the public `encodeStep` API, and classify the published presentation
view. Tests may attach `SparseWorldTrace` after a failure, but may not construct or
compile internal shader entry points directly.
