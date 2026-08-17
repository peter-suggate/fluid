# Sparse CM12 bounded pressure coherence

**Status:** implementation design for the 16³-brick Sparse CM12 resident path.
This replaces the idea that a topology change should trigger a complete pressure
topology or MGPCG rebuild. There is one production algorithm: mutate persistent
pressure state from generation-stamped 4³ dirty frontiers, close the mutation
over incidence and hierarchy dependencies, and continue the existing solve.

The amount of work may approach the whole pressure graph when the physical
change really has that reach (for example, deleting a bridge that splits a
domain-sized unanchored component). There is no dirty-ratio threshold, scene
predicate, or alternate full-rebuild solver. Capacity exhaustion, missing write
provenance, and an incomplete closure are initially **fail-closed** conditions:
the frame does not publish pressure or advance parity, and the receipt identifies
the first broken invariant.

This document covers pressure topology and the solve. The same 4³ tile identity
and generation vocabulary should be shared with face preparation, transport,
surface conditioning, and presentation, but those stages are outside the
implementation described here.

## 1. What the current path actually does

The current code has already made one important sparse transition:
`classifyPressureCells` and `classifyRows` build compact liquid-cell and active-row
lists, and fine PCG/vector kernels dispatch indirectly over those lists. Air is
not a PCG unknown.

The remaining repeated work is structural:

- `advance()` in
  `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts:2483` classifies every
  accepted cell and row, counts and compacts both lists, bakes every live cell's
  effective edges, then dispatches every structural brick, brick edge, and
  hierarchy group/edge.
- `bakeBrickAggregateDiagonal` in
  `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts:1473` walks the
  accepted cell range of every brick and tests `isLiquid` inside the loop.
- `restrictBrickAggregateResidual` at line 1639 repeats the full accepted range
  walk during every preconditioner application.
- `bakePressureHierarchy*`, `restrictPressureHierarchyResidual`,
  `refinePressureHierarchyCorrection`, and
  `combinePressureHierarchyCorrection` dispatch over structural capacity rather
  than live aggregate state.
- The host loop at
  `webgpu-sparse-cm12-resident.ts:2537` encodes the complete iteration ceiling.
  Fine pipelined kernels obey `pipelinedPressureActive()`, but aggregate and
  hierarchy kernels do not consistently obey it, so convergence leaves a real
  preconditioner tail, not merely encoded zero-count commands.
- `candidateState` is cleared at
  `webgpu-sparse-cm12-resident.ts:2461`. It therefore cannot own persistent
  generations, worklists, coefficients, or recycled vectors. Its pressure
  scratch/transfer lifetime may remain unchanged.
- `compactPressureTopology()` at
  `webgpu-sparse-cm12-resident.ts:1404` already constructs immutable fine-edge,
  brick-edge, and hierarchy incidence on the host. It is the right place to add
  reverse dependency tables; it is not a per-frame operation.

The pressure optimization is therefore not “remove air unknowns”; that has
already happened. It is “stop rediscovering those unknowns, stop walking air to
aggregate them, and stop refreshing structural ancestors that did not change.”

## 2. Non-negotiable invariants

The incremental path is allowed to publish generation `g` only when all of the
following are true.

1. **Write provenance.** Every accepted density, solid-open-fraction, accepted
   topology, or row-geometry write that can change pressure has stamped at least
   one pressure tile or row in generation `g`.
2. **Phase closure.** Every cell whose pressure density or liquid predicate may
   differ has been reclassified. `liquid(cell)` is identical to a from-scratch
   classification for the accepted generation.
3. **Incidence closure.** Every row incident to a changed cell, and every row
   whose acceptance/geometry changed, has current `theta`, active membership,
   and row generation.
4. **Operator closure.** Every directed fine edge affected by an endpoint,
   theta, or solid-weight change has current weight; every affected fine
   diagonal has been recomputed from current incident rows.
5. **Aggregate closure.** Every affected brick scalar and brick edge is current.
   Every affected hierarchy group and hierarchy edge is current at every level,
   bottom-up.
6. **Membership equivalence.** The persistent live tile, row, wet-brick,
   brick-edge, hierarchy-group, and hierarchy-edge worklists contain exactly the
   records whose current live predicates are true. Inverse slot maps agree with
   the forward lists.
7. **Solver continuity.** Retained unknowns retain pressure/history. Entering
   unknowns receive a defined local seed. Retired unknowns are absent from every
   live worklist before the solve. No topology event zeroes unrelated pressure,
   hierarchy state, or recycled modes.
8. **Component authority.** Each pressure component has current anchor/gauge
   metadata. A merge, split, gained anchor, or lost anchor is resolved before
   PCG starts.
9. **Generation monotonicity.** A consumer may read only records whose
   `resolvedGeneration == pressureGeneration`. A producer stamp newer than its
   consumer stamp is a hard error, not permission to use stale data.
10. **Convergence authority.** The final published result still carries a fresh
    global `b - Ap` receipt. Temporal coherence may improve the seed and
    preconditioner, but never substitutes a recursive residual for the final
    authority.

These are algorithmic invariants, not debug aspirations. Production kernels
check cheap header-level forms; the diagnostic build records the first concrete
record that violates each one.

## 3. Stable 4³ pressure tiles

### 3.1 Identity

A 16³ accepted brick contains 64 fine work tiles. A tile contains 4×4×4
accepted cells, so one 64-lane workgroup maps one lane to one cell. Coarser
accepted rungs use the same rule in their own accepted-cell lattice:

| accepted brick rung | cells | 4³ work tiles |
|---:|---:|---:|
| 16³ | 4096 | 64 |
| 8³ | 512 | 8 |
| 4³ | 64 | 1 |
| 2³ / 1³ | 8 / 1 | 1 partial tile |

`PressureTileId` is a stable host-packed ID for `(brick, template-rung,
tile-local)`. Only tiles belonging to the currently accepted rung may be live.
The immutable tile record contains:

```text
firstCell             u32   // contiguous template cell base
acceptedResolution    u16
validLaneCount        u8
tileLocal             u8
brick                 u32
incidentRowBegin/end  u32,u32
```

The current template cells are already brick/rung-contiguous, so fine and 8³
records do not need a 64-word cell indirection. Partial/coarse tiles use
`lane < validLaneCount`. The host packer asserts that the cell ordering still
matches `x + r*(y + r*z)`; failure to prove that is a construction error.

Each mutable tile state stores:

```text
liquidMaskLo/Hi             u32,u32  // one bit per lane
dirtyReasonMask             u32
producerGeneration          u32
classifiedGeneration        u32
rowsClosedGeneration        u32
operatorClosedGeneration    u32
activeListSlot              u32      // INVALID when liquidMask == 0
lastPressureDensity[64]     f32      // persistent; not candidateState
```

The two-word mask is the common pressure/non-pressure ABI. Pressure does not
invent a different granularity when the rest of the resident pipeline adopts
temporal coherence.

### 3.2 Persistent arena

Enlarge and version binding 15, currently `pressureWorklists`, into a persistent
`pressureCoherenceArena`. This avoids another storage binding (the resident bind
group is already designed around the portable storage-buffer limit) and keeps
the construction-time neighbour/extrapolation planes already stored there.
The arena contains:

1. header, generations, fail-closed receipt, queue counters, and indirect counts;
2. immutable tile records and reverse dependency offsets;
3. persistent tile/row/edge/aggregate/group stamps and values;
4. double-buffered generation-stamped frontier queues;
5. paged live worklists and inverse slot maps;
6. existing neighbour IDs and extrapolation weights;
7. pressure history and recycled-subspace metadata not stored in the ordinary
   state buffer.

`candidateState` remains transient edge/preconditioner scratch until those
planes move into the persistent arena; it must no longer be the only copy of a
coefficient that temporal coherence claims to preserve.

### 3.3 Generation-stamped queues

There are distinct queues for tiles, rows, fine directed edges, fine cells,
bricks, brick edges, component repairs, and `(level, group/edge)` ancestors.
Enqueue is idempotent:

```wgsl
if atomicExchange(&record.queuedGeneration, generation) != generation {
  let slot = atomicAdd(&queue.count, 1u);
  if slot < queue.capacity { queue.items[slot] = id; }
  else { failClosed(QUEUE_OVERFLOW, kind, id); }
}
atomicOr(&record.reasonMask, reason);
```

Every capacity is the size of the stable universe for that record kind, so a
deduplicated generation cannot legitimately overflow. Overflow therefore means
bad metadata or a broken deduplication invariant; there is no larger emergency
queue and no scan of the universe.

Queues alternate input/output storage while a closure stage is running. A
stage continues while the GPU-written output count is nonzero. Host encoding
uses a conservative maximum number of graph levels/rings, but each pass is an
indirect zero-count dispatch after the frontier empties. Exceeding the proven
maximum closure depth fails closed with the nonempty queue head captured.

`u32` generation wrap is also fail-closed initially. At 60 pressure epochs per
second it is a multi-year event; later it can become a separately proved epoch
renormalization transaction, never an implicit whole-topology rebuild.

## 4. Persistent paged worklists

The active pressure tiles, active projection rows, wet brick aggregates, live
brick edges, and live hierarchy records are not rebuilt by count/prefix/compact.
Each is a dense vector implemented as 32-entry pages:

```text
page: liveMask u32, count u32, entries[32]
list: pageCount u32, tailCount u32, activePageIds[], inverseSlot[stableId]
pool: freePageStack[], freeCount
```

Insertion appends at the tail. Removal swaps the last live entry into the hole
and updates that entry's inverse slot. An empty tail page returns to the free
stack. Parallel mutations first deduplicate into add/remove queues, then reserve
tail ranges with one prefix operation; removals claim tail slots with atomics.
The repair moves at most one record per removal and allocates at most one page
per 32 additions. Thus repair is proportional to membership delta, all pages
except the tail are full after commit, and fragmentation cannot accumulate.

The mutation journal stores `(list, stableId, oldSlot, newSlot, displacedId)`
until forward/inverse validation succeeds. Any double membership, absent remove,
out-of-range slot, exhausted page pool, or mismatched inverse fails closed and
retains the pre-transaction generation. It does not fall back to compaction.

One tile workgroup is addressed by a packed tile rank; one row workgroup uses a
64-row page variant or two 32-entry pages. Reductions emit one partial per live
tile, so `pressureCellWorkgroups()` becomes `activePressureTileCount()`. List
order may change after local swap-tail repair; numerical acceptance therefore
uses residual and physical envelopes rather than bitwise reduction identity.

## 5. Dirty producers and bounded closure

### 5.1 Producers

All writers use one of the following reasons. The reason is retained for the
overlay even if several producers coalesce into one queue item.

| reason | producer | initial pressure frontier |
|---|---|---|
| `DENSITY_WRITE` | transport, gamma/surface finalization, injection | exact 4³ tile containing a value that changed |
| `SOLID_WRITE` | rigid voxelization/open-fraction update | swept-solid tiles and directly modified rows |
| `ACCEPTED_RUNG_EXIT/ENTER` | topology commit | old-rung and new-rung tiles for only the committed bricks |
| `ROW_GEOMETRY` | reconstructed/changed face topology | exact changed rows |
| `BOUNDARY_WRITE` | boundary-condition mutation | touched boundary rows and their incident tiles |
| `SEED_REPAIR` | liquid membership transition | entering cell plus one-ring liquid neighbours |
| `COMPONENT_REPAIR` | active-edge/anchor mutation | affected component endpoints |

Density writers compare the value already in registers with the old destination
value before storing and call `markPressureTileDirty` only on a real bit/value
change. The persistent `lastPressureDensity` is updated only when pressure
classification consumes the tile.

`validateAndCommitShadowTopology` is currently the atomic accepted-generation
publication point. Before it clears a scheduled brick at lines 3501–3516, it
must append `(brick, oldRung, newRung, shadowGeneration)` to the pressure topology
delta queue. The pressure mutation consumes that queue at the start of the next
pressure epoch. Publication is rejected if the queue entry cannot be reserved.

Dynamic topology pages are not pressure-ready until they own rows, incidence,
fine edges, tile records, and all reverse dependency tables. The existing
cell-only dynamic-page prototype remains ineligible. Attempting to accept such a
page is `MISSING_PRESSURE_PROVENANCE`, not a reason to scan or rebuild globally.

### 5.2 Closure algorithm

The transaction is deterministic in dependency order, although work inside one
frontier is parallel.

1. **Apply accepted-rung deltas.** Retire old-rung tiles/rows from persistent
   membership; activate new-rung records; preserve transferred pressure already
   written by `writeCandidateCellsToShadow`; enqueue new/retired cells, all rows
   incident to either rung, affected brick edges, and component endpoints.
2. **Classify dirty tiles.** For each valid lane, recompute `pressureDensity`,
   compare with `lastPressureDensity`, update the liquid bit, and emit entering
   or leaving cell mutations. A separating-minimum cell also enqueues the tiles
   read by its continuation stencil. Update active-tile membership only when the
   64-bit mask crosses zero.
3. **Close rows.** Walk the immutable tile-to-row incidence and deduplicate row
   IDs. Recompute row acceptance, liquid/air weighted phi, theta, active-row
   membership, and anchor contribution. If theta, activity, geometry, or solid
   weight changed, enqueue every directed fine edge owned by that row and every
   incident fine cell diagonal. `compactPressureTopology()` adds the required
   `rowEdgeOffsets/rowEdges` reverse table.
4. **Patch the fine operator.** Recompute only queued directed weights and fine
   diagonals. Each changed edge enqueues its owning brick edge (if cross-brick),
   both endpoint bricks, and component endpoints when its nonzero predicate
   changes.
5. **Patch brick aggregates.** Recompute a queued brick diagonal by traversing
   only set bits in its active 4³ tile masks. Recompute only queued brick edges
   from their immutable fine-edge contribution list. Update wet-brick and live
   brick-edge membership. No kernel walks a 16³ range merely to reject air.
6. **Propagate hierarchy deltas bottom-up.** A changed brick scalar enqueues its
   parent group at every required next level; a changed brick edge enqueues the
   hierarchy edge containing it and both endpoint groups. Recompute queued
   group diagonals/edges from current child contributions, update live
   membership, and enqueue the next ancestor only when the numeric value or live
   predicate changed. Stop at the fixed point, not at a preselected dirty ring.
7. **Repair components.** Edge additions merge persistent component metadata.
   Deleting a non-forest edge is local. Deleting a spanning-forest edge launches
   an interleaved flood from both endpoints looking for a replacement edge; if
   none exists, it relabels the smaller discovered side and updates anchor/gauge
   sums. A true large split can visit a large component, but it remains the same
   delta algorithm and visits the physically affected component—not unrelated
   components or air.
8. **Validate and publish.** Check queue exhaustion, generation equality,
   membership journals, hierarchy parent closure, and component authority.
   Atomically advance `acceptedPressureGeneration` only after all pass. On
   failure, retain the previous pressure generation and do not solve/project or
   advance frame parity.

There is deliberately no “if dirty coverage > X then rebuild” branch. A large
frontier simply contains more records.

## 6. Solve shape after the topology transaction

### 6.1 Fine work

Replace cell-list kernels with 4³ tile kernels. `wid.x` resolves a live tile and
`lid.x` resolves its cell; lanes whose liquid bit is clear return. This applies
to pressure preparation, PCG initialization/update/operator applications,
preconditioner application, true-residual measurement, pressure history, and
projection diagnostics. The operator remains the exact composite fine operator
and retains ghost-fluid liquid-air boundary stiffness.

RHS preparation still visits every active liquid cell each frame because face
velocity and volume correction are frame state, not topology. This is useful
work. It no longer pays cell compaction or air lanes outside a partially wet
4³ tile.

Projection uses the persistent active-row list. Rows leave or enter it only via
the incidence closure above.

### 6.2 Eliminate full-brick air walks

Each active tile reduction writes one `tileResidualPartial[tileId]`. A
`restrictWetBrickResidual` workgroup is dispatched only for the persistent
wet-brick list and sums at most 64 tile partials selected by the brick's active
tile mask. This replaces `restrictBrickAggregateResidual`'s repeated 4096-cell
walk at 16³.

Brick diagonal refresh follows the same mask, but only on the dirty-brick queue
and only once per coefficient generation. Brick-edge and hierarchy coefficient
refreshes similarly use dirty/live lists rather than structural counts.

During each V-cycle/preconditioner application:

- tile residuals: one workgroup per active pressure tile;
- brick restriction/refinement: one workgroup per wet brick;
- hierarchy restriction/refinement: one workgroup per live group;
- hierarchy edges: only live edge lists;
- correction prolongation: wet bricks or active tiles, never all structural
  bricks.

### 6.3 Eliminate the encoded convergence tail

First, every aggregate/hierarchy kernel with workgroup barriers receives a
uniform gate before its first barrier. That immediately removes the current
post-convergence body work and is safe because `pipelinedPressureActive()` is
uniform for the dispatch.

The final form also makes the tail's indirect workgroup counts zero. Use two
iteration-dispatch buffers, A and B, with records for tile, wet-brick, live-group,
live-edge, and scalar dispatches. Iteration `k` consumes A as `INDIRECT` and a
small, separately laid-out gate pipeline writes B as `STORAGE`; iteration
`k+1` swaps them. The compute pass ends between iterations so the WebGPU usage
scope never treats one buffer as writable storage and indirect in the same
pass. The gate publishes the real live counts while active and all zeros after
the guarded true-residual reduction closes the solve. All host-ceiling
iterations remain encoded, but the GPU launches zero workgroups for the tail.

This is a single solver mode, not a convergence fallback. The cost is a compute
pass boundary per active iteration. Measure that cost explicitly on Metal; do
not weaken the design to a full active tail if it is visible. If pass boundaries
dominate, use a proven small fixed batch and report its bounded tail separately,
but that is an interim milestone rather than the intended endpoint.

The final true-residual application is outside the gated iteration ring and
always runs over all active pressure tiles. Curvature recovery uses the same
live worklists and indirect gate; it never reconstructs topology or clears the
seed.

## 7. Solver continuity and recycling

Topology coherence saves setup work. Pressure temporal coherence must also save
iterations without assuming an ocean or a half-full surface brick.

### 7.1 Seed continuity

Keep two accepted pressure histories on stable cell IDs. For a retained liquid
cell, seed with a timestep-aware linear predictor:

```text
p* = p[n] + clamp(dt[n+1] / dt[n], predictorMinimum, predictorMaximum)
              * (p[n] - p[n-1])
```

The predictor bounds are numerical-stability constants shared by every scene,
not occupancy heuristics. The initial fresh residual remains authoritative, so
an unhelpful predictor costs iterations but cannot silently publish error.

For an entering liquid cell, use transferred pressure when a rung transition
supplied it. Otherwise seed by a bounded local harmonic extension from retained
liquid neighbours in the entering tile plus one incidence ring. Leaving cells
are removed from worklists but their history is retained until the stable
template slot is reused. No local event zeros retained cells.

After a component merge/split or anchor transition, adjust only the affected
component gauges. Anchored components preserve their physical pressure; a newly
unanchored component subtracts its volume-weighted mean. No unrelated component
is touched.

### 7.2 Recycled low modes

Maintain a small fixed number `K` (initially 4) of recent accepted correction
modes `U`, their images `C = A U`, and the small Gram matrix `E = U^T C` on
stable pressure IDs. Before MGPCG:

```text
y = E^-1 U^T r
x <- x + U y
r <- r - C y
```

When the operator changes locally, recompute `C` only on rows in the fine
operator closure. Update `E` by subtracting the cached old local contributions
and adding the new ones. Entering IDs receive locally interpolated mode values;
retired IDs contribute zero. The global `U^T r` reduction is normal solve work,
not topology recovery.

A successfully converged frame may rotate its normalized correction into one
column. Computing that new column's image is one ordinary active-tile operator
application. Loss of independence deterministically evicts the offending mode;
it does not reset pressure, topology, hierarchy, or the other columns. A
non-finite/indefinite Gram receipt fails closed.

The multigrid hierarchy itself is persistent. Coefficients and live membership
are patched by the closure algorithm; correction scratch is cleared only on
live groups at the start of a V-cycle. Abrupt local topology therefore changes
the preconditioner only in its ancestor cone while the outer true residual
continues to protect correctness.

## 8. Dirtyness/provenance visualization

Temporal coherence is not acceptable as invisible scheduler state. Publish a
read-only `GPUSparseCM12CoherenceSource` that aliases the exact arena used by
physics. Do not build a CPU shadow representation for the overlay.

The overlay needs these selectable layers:

- **producer dirty tiles:** density, solid, accepted-rung, boundary, and seed
  reasons as independently toggleable colors;
- **closure stage:** classified tile, incident row, fine edge/diagonal, brick,
  hierarchy ancestor, and component-repair frontiers;
- **age:** `currentGeneration - lastResolvedGeneration` for every record kind;
- **phase:** 64-bit liquid occupancy within each 4³ tile, including partial wet
  tiles versus all-liquid tiles;
- **membership:** active pressure tiles, active rows, wet bricks, live brick
  edges, and live groups per hierarchy level;
- **blast radius:** origin generation/reason plus closure distance, so a user can
  see which producer caused a distant ancestor/component visit;
- **allocator:** live page occupancy, tail page, swap repairs, free-page count,
  and inverse-map faults;
- **fail closed:** the first offending record in opaque red, with expected and
  observed generations and the provenance chain that should have reached it.

Keep a small ring (for example 16 generations) of tile/reason/closure-distance
stamps rather than snapshots of all fields. The UI can freeze an epoch and scrub
those generations. Counters published beside the view include, per stage:

```text
producer records, unique queued records, duplicate enqueues, records resolved,
maximum closure distance, pages touched, entries moved, active/live counts,
indirect active iterations, zero-dispatch tail iterations, fail-closed code/id
```

Implementation touchpoints are a new pressure-coherence overlay beside
`lib/core/webgpu-pressure-journal-overlay.ts`, registration in
`lib/core/visualization-catalog.ts`, and an optional source in
`lib/core/method-contract.ts`. The existing journal overlay remains the field
and residual view; the coherence overlay explains why work ran.

## 9. Exact code touchpoints

### `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts`

- Extend `compactPressureTopology()` with stable tile records,
  tile-to-row incidence, row-to-directed-edge incidence, fine-edge-to-brick-edge,
  brick-edge-to-hierarchy-edge, and component adjacency records.
- Replace `pressureWorklistAndNeighbors()` with the versioned persistent arena
  initializer. Preserve neighbour/extrapolation planes.
- Replace `pressureCellIndirectArguments` / `pressureRowIndirectArguments` with
  persistent paged-list counts plus the A/B iteration-dispatch buffers.
- Remove the per-frame classify/count/finalize/compact chain from the
  `pressure-topology` stage. Encode the dirty transaction and its fail-closed
  publication instead.
- Stop clearing persistent pressure coefficients/generations with the
  `candidateState` clear. Keep only truly transient transfer/V-cycle planes in
  that clear range.
- Dispatch dirty coefficient kernels by queue, and iteration kernels by live
  paged lists. End/swap iteration passes for legal indirect gating.
- Expose coherence-source buffers and receipts with no production readback.

### `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts`

- Replace `pressureCellInvocation` / `pressureRowInvocation` with paged tile and
  row resolvers.
- Add producer stamping, queue enqueue/drain, paged membership mutation/repair,
  and fail-closed helpers.
- Split `classifyPressureCells`, `classifyRows`,
  `bakeEffectivePressureEdges`, and `preparePressure` into dirty-topology and
  every-frame RHS tile kernels.
- Replace `bakeBrickAggregateDiagonal` and
  `restrictBrickAggregateResidual` with active-tile-mask/wet-brick kernels.
- Make hierarchy coefficient kernels consume dirty ancestor queues and V-cycle
  kernels consume live lists.
- Add component repair, pressure-history prediction/local extension, recycled
  mode projection/update, and the A/B indirect gate publisher.
- Uniform-gate every barrier-bearing preconditioner/recovery kernel immediately.
- Add the pressure generation to journal records so residual films and dirty
  overlays correlate exactly.

### Topology and external writers

- `validateAndCommitShadowTopology` must journal old/new rung deltas before
  accepted publication.
- Every density finalizer, injection path, rigid voxelizer, boundary writer, and
  topology row reconstruction must stamp the shared tile/row provenance ABI.
- GPU-created dynamic topology must publish full pressure reverse incidence
  before it becomes eligible for acceptance.

### Diagnostics

- Add `GPUSparseCM12CoherenceSource` to `lib/core/method-contract.ts`.
- Add `lib/core/webgpu-sparse-cm12-coherence-overlay.ts` and catalog entries.
- Extend performance receipts with dirty/live counts and active/zero iteration
  counts; do not make readback a scheduling dependency.

## 10. Phased landing order

Each phase leaves one runtime path. Temporary comparison/oracle code is probe
only and is removed or compile-time diagnostic; production never chooses an old
path by dirty ratio.

1. **Coherence ABI and fail-closed provenance.** Add tile IDs, persistent arena,
   generation queues, reason stamps, paged-list primitives, overlay, and receipts.
   Keep existing numerical kernels temporarily fed through the new live lists.
2. **Air-walk removal.** Convert fine kernels to 4³ tile dispatch, replace
   brick residual/diagonal walks with tile partials and wet-brick reduction, and
   convert projection to the persistent row list.
3. **Incremental fine topology.** Land density/topology/solid producers, dirty
   tile classification, incident-row closure, fine-edge/diagonal patching, and
   entering/leaving worklist mutations. Delete full accepted classification and
   compaction.
4. **Incremental aggregates/hierarchy.** Land dirty brick/edge propagation,
   bottom-up ancestor fixed point, live hierarchy worklists, and component
   repair. Delete structural brick/group/edge coefficient dispatches.
5. **True zero tail.** First gate all bodies uniformly, then land A/B indirect
   dispatch buffers and per-iteration pass swaps. Publish active versus
   zero-dispatch iteration counts.
6. **Prediction and recycling.** Land two-field pressure prediction, local seed
   extension, gauge-local adjustment, then the `K=4` recycled low-mode space.

No phase adds a global rebuild escape hatch. Until a phase's closure is complete,
its unsupported mutation fails closed with a visible reason.

## 11. Regression and measurement lanes (no unit tests)

Use executable Dawn/Metal probes and scene-level property checks. Do not add
isolated unit tests for queue helpers or manufactured tiny topology cases as the
acceptance authority.

### 64³ dam-front lane

The standalone
`tools/run-sparse-cm12-temporal-regressions.ts` lane is configured for the 16³
brick ladder and 16-wide presentation pages. It carries forward the physical
properties previously exercised by `tests/sparse-cm12-mini64-front-dawn.test.ts`:

- mass drift at most 0.2%;
- surface front reaches at least `x=56` by frame 5 and never retreats;
- adaptive/all-fine surface and liquid fronts agree within one fine cell each
  frame;
- final density relative L1 at most 0.06 and maximum density at most 2;
- every prepared topology transition commits, with live same-level coarse and
  mixed-seam rows;
- final fresh relative residual and maximum divergence remain inside the
  captured pre-change envelope.

Add coherence-specific gates:

- no fail-closed receipt;
- every committed brick appears as an `ACCEPTED_RUNG_*` origin;
- dirty rows equal the incidence closure oracle; dirty fine/brick/hierarchy
  edges and groups equal their reverse-incidence closure oracle;
- unchanged distant tiles/rows/groups retain their generations and coefficients;
- page repair moves no more records than membership removals plus additions;
- no full accepted-cell/row or structural brick/group coefficient dispatch is
  encoded;
- zero-dispatch tail count equals encoded minus active iterations.

The closure oracle may recompute the entire graph in the diagnostic probe and
compare it to the incremental publication. It is evidence, never a production
fallback.

### Weakened symmetric-expansion lane

Run the symmetric expansion scene without the exact D4 authority that copies
one wedge into the others. Preserve independently evolved symmetry properties:

- reflected density L1/L∞, front-radius spread, center-of-mass drift, projected
  face D4, and divergence stay within a baseline envelope captured before this
  work;
- opposing dirty-frontier counts and maximum closure distances agree within the
  same weakened envelope, rather than requiring identical list slots or
  reduction bits;
- recycled-mode projection does not introduce a preferred axis;
- topology events in one wedge produce only the geometrically corresponding
  ancestor/component blast radius, with no unrelated-domain generation bump.

Run both lanes across quiescent frames, moving-front frames, rung enter/exit,
liquid enter/exit, and at least one component anchor transition. Report GPU
times for pressure topology, RHS, coefficient patching, V-cycle, fine SpMV,
reductions, iteration pass boundaries, and final receipt. Also report work
counts so a faster time cannot conceal accidental skipped closure.

## 12. Completion criteria

This work is complete only when:

- a quiescent pressure epoch performs no classification, compaction, coefficient,
  aggregate, or hierarchy-topology work;
- a local topology event mutates only its incidence/ancestor/component closure;
- no pressure kernel walks a 16³ brick to reject air;
- no converged-tail pressure workgroup launches;
- pressure/history/recycled state survives unrelated topology events;
- the dirtyness overlay can explain every visited record back to a producer;
- all invariant gaps fail closed with actionable GPU receipts; and
- the 64³ dam-front and weakened symmetric-expansion property lanes pass without
  a unit-test-derived or scene-specific exception.
