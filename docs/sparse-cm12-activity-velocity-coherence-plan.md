# Sparse CM12: incremental activity and velocity extension

**Scope.** This is the next performance slice after the scalar/row temporal
worklists. The current ocean B16/P16 trace reports about **2.88 ms** for compact
brick activity measurement and **1.90 ms** for transport-velocity extension.
Both stages still revisit the accepted domain even when the surface and
topology have a small local blast radius.

The design below is scene-independent. It preserves the current numerical
definitions, never schedules a whole-domain recovery, and rejects a candidate
generation when provenance, capacity, or closure is incomplete. The existing
accepted generation remains untouched so a fault is diagnosable.

## 1. Resident facts that constrain the design

- `buildTemporalCellWorklist` already performs the one accepted-cell census at
  the end of an advance. It publishes a stable per-cell flag plus scalar-cell,
  scalar-row, and face-row lists in the tail of `pressureWorklists` (binding
  15 of `pressureBindGroup`). The next slice must enrich that census, not add a
  second one.
- The shared CMD1 dirty-provenance publication uses stable logical base-brick
  keys and 4^3 local tiles. Its causes already distinguish density, phase,
  gamma, velocity/CFL, solid/boundary, topology, and generation faults.
- `measureBrickActivity` currently launches one 64-lane workgroup per physical
  brick. It scans every accepted cell, then every cell incidence and opposing
  row term. The density moments are fixed-point sums; deformation, predicted
  motion, detail error, and velocity travel are maxima; surface/support state
  is an OR reduction. These are all exactly decomposable into persistent 4^3
  tile summaries.
- Velocity extension currently initializes every accepted cell and performs
  eight accepted-cell Jacobi dispatches. A cell becomes valid at its first
  reachable graph layer and is thereafter copied. Therefore its exact result
  is determined by liquid seeds and accepted pressure-template edges no more
  than eight graph edges away.
- The compute layout is already at the portable storage-buffer limit. Activity
  state must remain in the `activity` arena (binding 12), extension metadata and
  worklists in `pressureWorklists` (binding 15), and extension vector payloads
  in the `state` arena (binding 2). No new bind-group slot is required.

## 2. Slice A: persistent 4^3 activity summaries

### 2.1 Dirty law

An activity tile is dirty when any contribution to its current summary may
have changed:

1. final density, open fraction, accepted rung, or cell ownership changed in
   the tile;
2. a density/phase change in one incidence neighbour can change the tile's
   interface, exposed-side, deformation, or thin-feature tests;
3. a density change in the same 2^3 accepted-cell sibling group can change its
   restriction-detail error;
4. a projected cell velocity change can change velocity travel, deformation,
   or swept support in its tile and one accepted-incidence ring;
5. a projected face-velocity change can change predicted motion in every
   incident cell tile;
6. topology creation/retirement, moving-solid openness, or a changed accepted
   row dirties the affected owner tiles and the same one-ring dependency
   closure.

All six causes already arise in existing producer kernels or in the existing
tail census. Producers call one helper, conceptually
`markActivityOwnerTile(cell, cause, closureDepth)`. It maps through the stable
logical 4^3 identity used by CMD1, stamps the tile once for the candidate
generation, ORs the cause/depth, appends it to the compact tile worklist, and
stamps/appends its physical activity brick once. Duplicate events are benign.

The crucial additions are folded into work that already exists:

- density/phase/sibling and incidence closure is marked by the mass and surface
  producers that already visit those cells/rows;
- `projectFaces` compares its new accepted face value with the prior face bank
  and marks incident owner tiles;
- `collocateAndDiagnose` compares the new collocated velocity with the prior
  projected cell bank and marks its tile plus the accepted-incidence ring;
- topology commit marks only created, retired, or ownership-changed tiles;
- `buildTemporalCellWorklist` verifies/finalizes these flags while it performs
  the already-paid accepted census.

No activity-only cell or brick scan is introduced.

### 2.2 Activity-arena tail

Append this versioned region after the existing activity records and current
presentation-private tail. It is part of binding 12, not a new buffer:

```text
ActivityIncrementalHeader (32 u32)
  0 magic = "A4D1"             8 recomputedTileCount
  1 version = 1                9 recomputedBrickCount
  2 candidateGeneration       10 reusedTileCount
  3 acceptedTopologyGen       11 maximumClosureDepth
  4 dirtyTileCount            12 uncoveredWriteCount
  5 dirtyBrickCount           13 firstFaultTile
  6 directTileCount           14 firstFaultBrick
  7 closureTileCount          15 faultFlags
 16..19 tile indirect args    20..23 brick indirect args
 24 tileCapacity              25 brickCapacity
 26 tileSummaryWords = 16     27 scoreHistogramBins = 256
 28 producerGeneration       29 consumerGeneration
 30 executedCellContribs      31 skipped/reusedCellContribs

ActivityTileSummary[logicalTileCount] (16 u32 each)
  0 committedGeneration       8 max velocity travel (f32 bits)
  1 topologyGeneration        9 flags: surface axes/occupied/thin/cut
  2 density fixed sum        10 support mask (27 bits)
  3..5 fixed moments xyz     11 swept-support mask (27 bits)
  6 max deformation          12 accepted contribution count
  7 max predicted motion     13 max detail error (f32 bits)
                              14 cause/depth receipt
                              15 validity/check word

tileVisitGeneration[logicalTileCount]
dirtyTileList[logicalTileCount]
brickVisitGeneration[physicalBrickCount]
dirtyBrickList[physicalBrickCount]
scoreHistogram[256]
```

Signed fixed-point words 2..5 retain the current `ACTIVITY_FIXED` arithmetic.
Maxima and masks retain the current formulas. A tile owns an accepted cell
when the cell's finest-coordinate minimum lies in that tile. This assigns a
coarse cell exactly once. A 64-lane tile kernel enumerates the 4^3 finest
positions and only the lane whose position equals the resolved owner's minimum
evaluates that owner, so no dynamic tile-to-cell table or scan is needed.

### 2.3 Kernels and order

1. **`beginIncrementalActivity`** (one workgroup) advances the activity clock,
   publishes the candidate generation, and clears only compact counters/faults.
   Visit arrays are generation-stamped and are never globally cleared.
2. Existing mass, surface, projection, solid, and topology kernels call
   `markActivityOwnerTile`. A first-frame bootstrap marks every accepted tile;
   topology change marks only its affected tiles.
3. **`rebuildDirtyActivityTiles`** is dispatched indirectly over
   `dirtyTileList`. One 64-lane workgroup produces the exact 16-word persistent
   summary for each dirty tile. It walks incidence only for the accepted cell
   owners in that tile. The 2^3 detail group is evaluated once per owner rather
   than redundantly by all eight siblings.
4. **`reduceDirtyBrickActivity`** is dispatched indirectly over
   `dirtyBrickList`. One workgroup reduces at most `(B/4)^3` persistent tile
   summaries: 1, 8, or 64 for B4/B8/B16. It then runs the existing score,
   reason, temporal-moment, support, and velocity-floor formulas unchanged.
   A rung/topology generation change requires all owner tiles to carry that
   generation; a stale tile faults instead of triggering a brick/domain scan.
5. **`commitActivityCensusDelta`** compares each changed brick's prior and new
   class. It atomically updates persistent surface/hot/quiet totals and moves
   one count between the 256 score bins.
6. **`finalizeIncrementalActivity`** (one workgroup) finds the highest nonempty
   score bin, finalizes counts/indirect receipts, and validates generations,
   capacities, and executed counts.

The current hot/quiet history update should move into `planBrickResolution`.
That kernel already visits every physical brick and already owns the topology
epoch predicate. Updating history there only on a topology epoch preserves the
current semantics for unchanged bricks without making activity measurement
scan them. Planning itself remains a separate later optimization.

### 2.4 Activity fail-closed invariants

Publication is rejected if any of the following holds:

- dirty-tile or dirty-brick append overflow;
- a tile was written without a current-generation direct/closure cause;
- rebuilt tile count differs from the compact worklist count;
- a dirty brick observes a stale tile after a rung/topology change;
- fixed contribution count differs from the accepted owner count for the
  rebuilt tile;
- census decrement would underflow or the histogram total differs from the
  active brick total;
- producer, consumer, CMD1, and accepted topology generations disagree.

There is deliberately no `measureBrickActivity` fallback. During bring-up the
old kernel can write a shadow receipt for regression comparison, but a mismatch
is a hard fault and never selects the shadow result.

## 3. Slice B: exact local eight-edge velocity extension

### 3.1 Preserve the numerical operator

The optimization is not “usually use two sweeps.” It reproduces the existing
eight-sweep result on the only cells that can change.

Persist, for every template cell, the accepted extension vector and its first
valid graph depth (`0` for liquid seed, `1..8` for extrapolated air, `255` for
unresolved). Direct roots are:

- an interface liquid seed whose projected velocity changed;
- a cell whose liquid/air class changed;
- an accepted pressure-template edge/row whose acceptance or extrapolation
  weight changed;
- a created, retired, or ownership-changed cell;
- a moving-solid or boundary source that changes one of those authorities.

Interior liquid velocity changes are not extension roots: transport samples
the current projected liquid velocity directly. Only liquid cells incident to
the air band seed the persistent air extension.

The dirty blast is the exact accepted pressure-template graph dilation of
those roots through eight edges. At layer `d`, a dirty air cell averages
neighbours whose depth is less than `d`. A dirty neighbour reads its current
candidate depth/value; a clean neighbour reads its persistent accepted
depth/value. Liquid neighbours read the current projected source bank at depth
zero. This is the same validity recurrence as the existing Jacobi sweeps, but
clean values beyond the eight-edge blast are reused.

### 3.2 Reuse the activity arena and append a dedicated state tail

The implemented standalone `VEX1` ABI appends to the binding-12 activity arena
after ACT1. It deliberately does not share pressure coefficient or candidate
storage: those regions are transient across topology transfer. Persistent
extension vectors occupy a dedicated aligned tail of the resident state after
the optional pressure journal.

```text
ExtensionHeader (32 u32, 256-byte aligned)
  magic/version/header/flags, accepted and candidate generations
  root/frontier-A/frontier-B/blast counts and indirect triplets
  fault count, first-fault cell/depth, maximum closure depth
  executed/reused/uncovered-write counts, topology generation, capacity

rootStamp[cellCapacity]       blastStamp[cellCapacity]
blastDepth[cellCapacity]      candidateDepth[cellCapacity]
rootCause[cellCapacity]       rootList[cellCapacity]
frontierA[cellCapacity]       frontierB[cellCapacity]
blastList[cellCapacity]       acceptedDepth[cellCapacity]
acceptedOwner[cellCapacity]   reuseStamp[cellCapacity]
```

Append `extensionAcceptedVelocity: vec4f[cellCount]` to the existing `state`
layout. XYZ is the persistent accepted air vector; W is reserved/check data.
The ordinary `destinationCellVelocity` bank is dead scratch before pressure
projection and is the per-frame candidate vector. No second persistent vector
bank and no new binding are needed.

Transport sampling becomes one accessor:

```text
liquid cell                       -> current sourceCellVelocity
dirty air, current candidate gen -> destinationCellVelocity candidate
clean air                         -> extensionAcceptedVelocity
invalid/unresolved               -> zero
```

This removes the current full liquid copy as well as the eight full recurrence
dispatches. Pressure projection may overwrite the candidate bank only after
transport has committed dirty air vectors to the persistent cache.

### 3.3 Kernels and order

The existing tail census authors the next frame's roots:

1. Existing scalar, projection, topology, injection, and moving-solid producers
   call `cm12ExtensionRecordRoot`. Each cause has an explicit required closure
   in `SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT`; missing coverage faults
   instead of selecting a global recovery path.
2. **`seedVelocityExtensionRoots`** stamps roots at depth zero and initializes the root
   and blast lists.
3. **`expandVelocityExtensionFrontier`** runs for graph depths 1..8 on the compact prior
   frontier. It walks the existing pressure CSR, applies `rowAccepted`, stamps
   each neighbour once, and appends it to both the next frontier and blast
   list. Each frontier dispatch is indirect. Small device-side copies of the
   next-frontier dispatch triplet create the required pass boundary; no
   accepted-domain dispatch is substituted when a frontier is empty.
4. At the next frame's extension stage,
   **`initializeVelocityExtensionCandidates`** dispatches indirectly over the
   persisted blast list: current liquid roots get depth zero/current velocity;
   dirty air starts unresolved.
5. Eight specializations of **`advanceVelocityExtensionCandidates`** dispatch
   over that same compact blast list. Pipeline constants supply the depth, so
   there is no per-frame uniform mutation. Each cell freezes on first validity,
   matching the current recurrence and source/destination bank parity.
6. **`commitVelocityExtensionCandidates`** copies only dirty air candidates and
   accepted depths to the persistent cache, validates the blast generation,
   and publishes executed/reused counts. **`finalizeVelocityExtensionCandidate`**
   publishes the generation only after depth eight and only when no fault is
   latched. Face preparation and characteristic sampling use the accessor above.

Frontier construction can initially live after the current temporal-list
authoring pass. It adds work proportional to the eight-edge blast, not another
domain census. If command-buffer pass/copy overhead becomes visible, the same
ABI permits a later tile-frontier compactor; correctness must not be traded for
a fixed full-capacity dispatch.

### 3.4 Extension fail-closed invariants

- every direct root has a physical CMD1 cause and current producer generation;
- every positive-depth visit has a prior-depth accepted CSR witness;
- root, frontier, and blast capacities do not overflow;
- blast depth never exceeds eight and every executed cell belongs to the
  current blast generation;
- topology generation on every reused cache entry equals its accepted owner;
- no vector/depth changes outside the declared blast in shadow comparison;
- all dirty air cells commit before pressure reuses the destination bank.

An unresolved cell after depth eight is a valid zero result, as today. Missing
provenance or stale cache identity is not: it rejects the candidate instead of
running the old nine full accepted-cell dispatches.

## 4. Recommended implementation order

1. Land CMD1 allocation/publication and add generation-stamped generic helpers
   for stable logical tile marking.
2. Implement incremental activity first. It is the larger measured cost, its
   reductions are associative, and it immediately validates that 4^3 dirty
   closure reaches density, topology, solid, and projection consumers.
3. Implement the persistent extension cache and exact eight-edge frontier.
   Reuse the projection comparisons and topology causes already proven by the
   activity slice.
4. Only after shadow equivalence is clean, delete the old full activity and
   extension dispatches. Keep the fail-closed validation and receipts.

## 5. Risks and required evidence

| Risk | Containment/evidence |
|:--|:--|
| Coarse or macro cells span several logical tiles | Attribute by finest-cell minimum; fan any intersecting change to that owner tile. Topology generation mismatch rejects stale attribution. |
| Interface classification changes because only the neighbour changed | Mandatory accepted-incidence one-ring closure; picked CMD1 witness shows the parent cell/row and depth. |
| Maximum contributor disappears from a tile/brick | Rebuild the whole dirty 4^3 tile, then re-reduce all persistent tiles of only its dirty brick. Never subtract a maximum. |
| Activity histories age while a brick is clean | Update histories in the already-existing all-brick planning kernel at topology epochs, not in activity measurement. |
| Cached air extension is treated as an early Jacobi seed | Store first-valid depth and allow a clean cache neighbour only when its depth is less than the current recurrence depth. |
| A projected change in deep liquid makes the blast huge | Liquid is sampled directly; only liquid cells incident to the air band are extension seeds. This is an operator dependency, not a scene heuristic. |
| Topology changes invalidate stable cell IDs | Cache entries carry accepted topology/owner generation; only the created/retired/ownership blast is rebuilt. A mismatch faults locally. |
| Append contention or duplicate work dominates | Generation-stamped CAS deduplicates tiles/cells; receipts publish raw origin events, unique roots, closure amplification, and high-water capacity. |
| High dirty share makes incremental work slower | It remains the same local algorithm at a large blast radius. There is no alternate full path; stage receipts expose the cost honestly. |

Acceptance uses the non-unit-test lanes requested for this work:

```sh
node --import tsx tools/run-sparse-cm12-temporal-regressions.ts \
  --lane=dam-front --dam-steps=5

node --import tsx tools/run-sparse-cm12-temporal-regressions.ts \
  --lane=symmetric-expansion-weakened --symmetry-steps=5

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx tools/probe-sparse-cm12-stage-cost.ts \
  --scene=ocean-seiche --brick-fine=16 --presentation-page=16 \
  --warmup=8 --frames=24
```

The regression receipt must additionally report activity dirty/reused tiles
and bricks, extension roots/blast/depth, executed/reused cells, capacity high
water, producer/consumer generations, and zero uncovered-write faults. The
performance claim is the quiescent and changed-frame median for the two named
hardware stages, not merely a skipped-work percentage.

## 6. Resident ACT1 activity-summary ABI

The integrated activity slice appends a 64-word-aligned `ACT1` arena after the
temporal scalar worklist. Its exported layout is the only offset authority:

```text
header[16]
stableTileStamp[stableTileCount]
stableTileCause[stableTileCount]
brickStamp[physicalBrickCount]
brickList[physicalBrickCount]
brickTopologyState[physicalBrickCount]
brickCensusState[physicalBrickCount]
brickTileMaskLow[physicalBrickCount]
brickTileMaskHigh[physicalBrickCount]
scoreHistogram[256]
```

The header is `[ACT1, version, 16, flags, generation, dirty tiles, dirty
bricks, uncovered writes, dispatch x/y/z, stable tiles, physical bricks,
measured bricks, reused active bricks, dirty active bricks]`. All stamps use
the current activity generation; arrays are not globally cleared. Invalid
headers, generation exhaustion, list overflow, invalid tile/brick identity,
initialization races, or a supposedly clean retained surface brick increment
the uncovered-write fault and reject reuse.

Each physical brick owns a 64-bit mask of dirty 4-fine-cell tiles. For ladder
size `B`, `tilesPerAxis = B/4`, valid bits are `tilesPerAxis^3`, and bit
`tx + tilesPerAxis*(ty + tilesPerAxis*tz)` identifies the tile. Thus B4 uses
one bit, B8 eight bits, and B16 all 64 bits. Presentation size does not alter
the mask: P4 consumes one bit per page; P8 consumes 2^3 bits per page; P16
consumes 4^3 bits per page. Topology activation, retirement, or rung/span
change forces every valid bit for only the changed physical leaf and expands
CMD1 identity across only that leaf's logical span.

The final projected velocity producer compares its collocated value with the
transport value before overwrite and marks the owner plus accepted-incidence
closure on any bitwise change. The already-compact scalar slow list marks
density/gamma work; the physical-brick topology census contributes local
created/retired/coefficient causes. The first producer authors the bounded
coverage journal, later producers merge their causes and direct/closure class
into the same packet without consuming another journal slot. CMD1 publication
still visits stable records to stamp explicit executed versus skipped state,
but it never schedules a full-domain physics fallback.
