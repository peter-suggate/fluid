# Sparse CM12 B16/P16 GPU frame architecture

## Scope and non-negotiable constraints

This is the cutover architecture for the production Sparse CM12 lane. The
primary specialization is a `16^3` brick ladder with `16^3` presentation pages.
Other supported sizes remain valid, but they must not shape the hot data path.

The target is a fully GPU-resident runtime with stable-frame non-pressure work
below 10 ms in ocean seiche, no performance-destroying global fallback during
large topology changes, and no physics regression through at least two seconds
of dam-break and weakened-symmetry evolution.

Every optimization is constrained by the following rules:

- A topology, capacity, generation, or provenance failure is fail-closed at the
  affected tile or brick. It may reject a candidate generation; it must not
  launch a whole-domain recovery path.
- Runtime scheduling decisions, topology worklists, pressure membership,
  presentation worklists, and publication generations remain device-owned.
  CPU readback is diagnostic only and never schedule input.
- Dirty visualization reads the same records that schedule physics. A second
  observability-only reconstruction is not authoritative.
- Physical fields are bit-exact across an optimization cutover unless the
  change has an explicit physics-equivalence proof and is admitted by the
  two-second regression gates below.
- The default contract is B16/P16 at every public and direct helper entry point.

## Why B16/P16 is the native execution shape

A B16 brick contains exactly `4^3 = 64` stable `4^3` dirtiness tiles. This gives
the GPU a useful one-workgroup mapping:

- one workgroup owns one logical or physical brick;
- one lane owns one stable dirty tile;
- one dirty tile workgroup owns 64 cells, one cell per invocation;
- one P16 presentation page owns one ordinary B16 brick;
- one presentation workgroup emits 4096 samples as 64 iterations per lane.

These are layout facts rather than scene assumptions. Coarser rungs use
construction-time cell and row packet maps so that a coarse authority is still
visited exactly once even when it covers several stable tiles.

## Current resident buffer and access map

The source of truth for this table is
`lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts` and
`lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts`.

Let `C` be physical template cells, `R` physical template rows, `E` directed
pressure edges, `B` physical bricks, `L` logical B16 bricks, `T = 64L` stable
tiles, and `M` tracers.

| Arena | Current size or record | Spatial order and access |
| --- | --- | --- |
| Frame parameters | 448 bytes | CPU-written uniform. It currently includes CPU-owned field parity. |
| Compact ownership topology | `4 * (hashCapacity + 2B + 2)` bytes | Open-addressed brick hash plus an 8-byte brick record. Every characteristic corner can repeat hash probes. |
| Physical cell template | 32 bytes/cell | AoS center, volume, widths, and packed brick/rung. Cell IDs are dense inside a brick/rung range. |
| Physical row template | 36 bytes/row | Nine SoA planes: packed term range, packed requirements/axis/kind, dual weight, area, distance, exterior phi, and center. Coalesced only for ordered row IDs. |
| Terms | 8 bytes/term | Row-contiguous cell ID and coefficient. |
| Incidence | 4 bytes/cell offset plus 8 bytes/incidence | Cell CSR containing row ID and global term index. |
| Accepted/shadow worklists | 8 bytes/cell and 8 bytes/row | Double-buffered IDs. Shadow construction uses global atomic append, so order is not guaranteed after a commit. |
| GPU topology page | 131,088 bytes/page at B16 | Four-word header plus 4096 eight-word cell records. Device LIFO allocation. |
| Physics state, non-rigid | approximately `4 * (23C + 3R + 4M)` bytes | Field SoA. Gamma intermediate images alias pressure fields. Several later phases reuse sharpening/D4 planes. |
| Conditioning | `4 * max(4C, C + R + 8)` bytes | Entire buffer is atomic. Reused by transport, gamma, sharpening, D4, and pressure compaction. |
| Pressure template | CSR offsets plus 8 bytes/static edge, then coarse/hierarchy records | Read-only and cell-contiguous. This is the cleanest current hot topology binding. |
| Pressure worklist/cache | approximately `4 * (21 + 3C + 4R + 2E + B)` bytes | Bootstrap is stable; incremental append and swap-delete progressively randomize cell and row order. |
| Candidate/pressure scratch | maximum of pressure scratch and 104,448 bytes/B16 candidate brick | Effective pressure edges and hierarchy data alias candidate topology transfer. Gamma clears this pressure region each frame. |
| Activity base | 112-byte header plus 156 bytes/brick | History, policy, topology state, counters, and dirty scheduling share one atomic arena. |
| Temporal lists | 32 bytes plus 12 bytes/cell and 4 bytes/row | Two flags, one unordered atomic-append cell list, and one unordered atomic-append row list. |
| Dirty scheduler/publication | approximately 244 bytes/stable tile, plus headers | CMD1/PKT1, CMS1 packet/journal/frontiers, and ACT1 tile stamp/cause. At B16 this is about 15.6 KiB/logical brick before brick history. |
| Presentation samples | 4 bytes/sample | B16/P16 is 16 KiB per ordinary brick page. All pages are currently dispatched before clean samples return early. |
| Partial/scalar reductions | 16 bytes/pressure workgroup and 80 scalar bytes | Reused by pressure and divergence reductions. |
| Indirect snapshots | 12 bytes/list, 24 bytes for pressure bootstrap | Device-authored but copy-isolated to satisfy WebGPU storage/indirect usage scopes. |

### Current hot-access problems

1. Immutable cell, row, term, incidence, and accepted-list data live in a
   storage-atomic `topologyArena` because the same binding also owns mutable
   worklists and topology pages. Non-pressure physics therefore pays atomic
   load semantics for immutable topology.
2. `state.pressure/rhs/diagonal/liquid` are gamma scratch before they become
   pressure state. `candidateState` is both persistent-looking pressure data
   and candidate-transfer storage. These aliases force clears and destroy
   cross-frame pressure reuse.
3. Temporal, pressure, activity, and shadow lists use global append or
   swap-delete. They lose the brick/rung locality established by template
   packing.
4. `ownerCellAt()` resolves a logical brick through a hash for every adaptive
   sampling corner. Face preparation, mass transport, sharpening, tracers, and
   coarse presentation amplify that random access.
5. The current dirty scheduler publishes rich provenance after a mostly global
   path. It is not yet the compact record that actually dispatches each heavy
   stage.

## Target resident domains

The cutover separates storage by lifetime and access mode. A phase-specific
shader and bind-group layout must expose only the buffers that phase uses.

### `StaticTopology`

Immutable, ordinary read-only storage:

- a dense logical-brick owner directory;
- cell geometry and brick/rung ranges;
- row records, terms, and requirements;
- cell incidence packets;
- directed extrapolation/pressure edges;
- brick aggregate and hierarchy topology;
- brick-to-row and brick-to-ancestor closure maps.

The dense owner directory maps every logical B16 coordinate directly to its
physical brick, span, accepted descriptor slot, and range table. Macro bricks
publish the same owner into every logical coordinate they cover. Ownership does
not change during runtime, so no frame performs hash reconstruction.

Initial hot packet shapes are:

- retain the exact 32-byte cell geometry record;
- use a 16-byte cell-incidence packet containing row, coefficient, static flux
  weight, and packed flags;
- use a 16-byte directed edge packet containing neighbor, row, base weight, and
  packed flags;
- retain row SoA for wide row passes, with construction-time tagged inline
  two-term and bounded variable-term records. A mixed row's variable record is
  normal local work, not a global fallback.

### `FramePlanCurrent` and `FramePlanNext`

Double-buffered GPU-owned scheduling state:

- one brick packet per logical/physical brick;
- 64-bit tile masks for face, mass, gamma, surface, pressure coefficient, and
  presentation stages;
- deterministic cell and row packet ranges;
- active pressure brick/group masks;
- indirect dispatch triplets;
- cause, closure depth, packet assignment, generation, and fault fields used
  directly by Grid structure dirty overlays.

One lane owns one B16 tile record. No producer competes to append that tile.
Cross-brick dependency closure writes bounded neighbor masks and is resolved in
fixed closure rounds. Capacity or generation failure invalidates only the
candidate brick/tile generation.

### `PhysicsState`

Persistent density, gamma, collocated velocity, face velocity, pressure, RHS,
diagonal, Krylov vectors, and divergence fields. Ping-pong bank selection is
derived from a GPU frame-generation word rather than host parity.

### `ScalarScratch`

Transport beta/deficit receipts, gamma images, sharpening delta/receipts, and
optional D4 scratch. It has no lifetime overlap with `PressureCache` and never
forces pressure membership or coefficients to be cleared.

### `PressureCache`

Persistent device state:

- liquid membership;
- ghost-fluid theta;
- stable per-brick segmented pressure cell and row lists;
- effective directed-edge weights;
- brick aggregate weights/diagonals;
- hierarchy group and edge weights;
- coefficient generations and dirty ancestor masks.

One workgroup repairs one dirty brick in stable local order. Large topology
changes therefore increase the number of parallel brick packets rather than
entering a singleton or whole-domain recovery path.

### `TopologyCandidate`

Candidate pages, prepared-brick list, cell/face transfer scratch, conservation
receipts, and next accepted descriptors. It is writable only by topology
phases and never aliases pressure coefficients.

### `Presentation` and `Diagnostics`

Presentation metadata and samples are updated from dirty P16 brick packets.
Diagnostics consume published receipts and may be read back explicitly, but
neither diagnostics nor visualization feeds scheduling decisions.

## Full-frame phase graph

### 1. Heavy physics on the accepted generation

Heavy kernels read `StaticTopology`, `FramePlanCurrent`, and the required
physics fields. They do not write topology or presentation metadata.

- Velocity extension keeps its ordered ping-pong sweeps. The final destination
  sweep also emits per-tile velocity validity and CFL evidence.
- Face preparation runs only face packets and applies body force after the
  characteristic result, removing a second row traversal without changing the
  value seen by pressure.
- Conservative mass keeps its three true global dependencies: trace/beta
  scatter, deficit scatter, then gather. Interior tile receipts may later be
  workgroup-local; cross-tile receipts retain fixed-point atomics.
- Gamma becomes two cell-owned incidence gathers. Each cell recomputes the
  signed integer row receipt and writes itself once, eliminating row atomics
  and accumulator clears while preserving the exact integer sum.
- Surface conditioning retains prepare/scatter/finalize ordering on dirty tile
  packets. Atomics are restricted to cross-tile receipt spill.

### 2. Pressure coefficient repair and solve

Scalar dirtiness and topology/solid closure update persistent pressure state:

- classify only dirty cells;
- reclassify only incident dirty rows;
- repair stable per-brick pressure segments;
- update only directed edges incident to changed rows;
- update affected brick aggregates and hierarchy ancestors;
- dispatch the solve over liquid cell, active row, active brick, and active
  hierarchy packets.

`bakeEffectivePressureEdges` and `preparePressure` are one cell traversal once
theta is published. Clean membership, theta, and coefficient generations are
reused without a clear or validation sweep.

### 3. Projection and current-frame measurement

Face projection and collocation remain physics work. Collocation also emits:

- divergence partials;
- exact changed-velocity tile masks;
- pressure/projection closure required by the next frame.

Activity measurement reduces one dirty brick at a time from deterministic tile
masks. It does not create a second global provenance transaction.

### 4. Topology update

Resolution planning produces a prepared-brick worklist. Grading closure is
bounded and brick-local. Candidate synthesis, cell/face transfer, row
reconstruction, and validation dispatch only prepared bricks and incident
rows.

Per-brick receipts validate in parallel. A final singleton may reduce receipt
flags and flip the accepted generation, but it must not loop over all bricks or
perform repairs. A rejected brick retains the prior accepted descriptor and
publishes its local fault.

### 5. Presentation and next-frame publication

For B16/P16, one dirty brick means one dirty page. Only dirty pages emit their
4096 samples. Presentation closure includes density/phase change, wetness
transition, rung/topology change, activation/retirement, and moving boundary or
solid change.

After topology commit and presentation publication, `FramePlanNext` is sealed
with complete generation and coverage receipts. It becomes current only after
all required packets are covered. A changed next-frame `dt` runs a small GPU
CFL revalidation that expands affected tile masks locally.

## Synchronization and atomic policy

Allowed global ordering points are limited to mathematical dependencies:

- velocity extension sweep boundaries;
- conservative transport scatter/gather boundaries;
- gamma first/second image boundary;
- surface scatter/finalize boundary;
- pressure row classification before coefficient assembly;
- Krylov reductions;
- topology receipt reduction before generation flip.

Atomics are appropriate for:

- fixed-point contributions crossing independently owned tiles;
- one counter per indirect packet family;
- rare topology page allocation;
- final diagnostic histograms or fault latches.

Atomics are not appropriate for:

- immutable topology reads;
- appending every dirty cell or row;
- claiming every tile from many cells;
- rebuilding accepted/shadow lists;
- serial pressure membership repair;
- ordinary D4 scratch loads and stores.

Writable scheduling arenas are copied to small indirect-only buffers only at a
phase boundary required by WebGPU usage scopes. Packet families should share a
single packed indirect snapshot so one copy publishes all heavy dispatches for
that generation.

## Physical bit-exactness contract

"Bit-exact" means identical physical state bits for the same backend, device,
initial state, controls, and accepted topology generation. It covers density,
gamma, cell and face velocity, pressure, membership, theta, accepted topology,
and conservation receipts. Diagnostic counters may change only where their
definition is explicitly versioned.

The following invariants must hold at every cutover:

1. Every accepted cell and row is executed exactly once by a stage that owns
   it, or is covered by an exact reuse certificate for that stage and
   generation.
2. Temporal closure covers every donor, receiver, incidence, characteristic,
   and scatter target reachable under the published CFL bound.
3. Fixed-point conservative transfers debit and credit the same integer once.
   Replacing atomic scatter with cell gather is permitted only when it sums the
   same signed integer receipts.
4. Row and incidence traversal order remains unchanged for floating reductions
   unless an explicit equivalence gate admits the new order. Packet compaction
   must not silently reorder a physical stencil.
5. Pressure membership contains exactly active, open liquid cells and no
   duplicates. Active pressure rows and theta match the same scalar/topology
   generation.
6. Effective directed edges, cell diagonals, brick aggregates, and hierarchy
   coefficients all carry the same coefficient generation. A partial mix is a
   local fail-closed fault.
7. The pressure operator remains symmetric positive definite under the same
   ghost-fluid and solid rules. Skipping an inactive aggregate is valid only
   when its RHS and couplings are exactly zero.
8. Candidate topology preserves 2:1 grading and passes mass, gamma, momentum,
   and face reconstruction receipts before publication.
9. Presentation samples correspond to the accepted topology and destination
   density generation. A clean page is reused only when every possible source
   is certified clean.
10. No overflow, stale generation, or missing provenance can select a broader
    performance path. The affected candidate stays unpublished.

## Cutover sequence and gates

1. **Arena separation.** Split immutable topology, scalar scratch, persistent
   pressure cache, and topology candidate storage. Keep numerical dispatches
   unchanged and require full physical bit identity.
2. **Actual B16 tile plan.** Replace unordered temporal/activity compaction and
   observability-only scheduler work with deterministic tile-owned packets.
   Require complete Grid structure dirty overlays and physical bit identity.
3. **Persistent pressure topology.** Remove liquid/theta clears and persistent
   member scans. Add per-brick repair and coefficient generations. Require
   pressure membership, theta, operator, residual, and projected velocity bit
   identity.
4. **Compiled owner and hot stencils.** Replace hash and atomic topology reads
   while preserving stencil order. Require full physical bit identity.
5. **Safe traversal coalescing.** Fold final velocity classification, face
   force, pressure coefficient/RHS preparation, and collocation diagnostics.
   Convert gamma to integer-equivalent cell gather. Require full physical bit
   identity for the folds; gamma gather additionally requires exact receipt
   and two-second physical gates.
6. **Blast-radius topology update.** Prepared-brick and incident-row packets,
   parallel receipts, and local generation publication. Require bit-identical
   accepted topology and transfer receipts across large dam-front changes.
7. **Dirty P16 publication.** Dispatch one page per dirty brick. Require exact
   presentation sample identity and generation coverage.
8. **GPU frame control.** Move parity and frame generation off the host. The
   CPU supplies only external controls and encodes a fixed schedule.

Each step is gated with GPU runs, not unit tests:

- ocean seiche B16/P16 stage trace and the combined non-pressure p95 target;
- mini dam break 64 front position, mass ledger, topology commits, and
  performance against its recorded baseline;
- weakened symmetric expansion density D4, velocity symmetry, mass, and
  topology evolution;
- continuous evolution for at least two simulated seconds, including the
  final rather than only an intermediate capture.

No step is accepted merely because its stable-frame benchmark improves.

## Source touchpoints

- Resident packing, arenas, defaults, and frame encoding:
  `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts`
- Resident topology access, kernels, and presentation:
  `lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts`
- Dirty packet ABI and current scheduler:
  `lib/methods/adaptive-mass/sparse-cm12-dirty-scheduler.ts` and
  `lib/methods/adaptive-mass/sparse-cm12-dirty-scheduler.wgsl.ts`
- Incremental activity:
  `lib/methods/adaptive-mass/sparse-cm12-incremental-activity.ts` and
  `lib/methods/adaptive-mass/sparse-cm12-incremental-activity.wgsl.ts`
- Pressure membership control:
  `lib/methods/adaptive-mass/sparse-cm12-pressure-membership.ts`
- Public B16/P16 option defaults:
  `lib/methods/adaptive-mass/method.ts`,
  `lib/methods/adaptive-mass/sparse-brick-atlas.ts`, and
  `lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts`
- Grid structure dirty visualization ABI:
  `lib/core/sparse-cm12-dirty-visualizations.ts` and
  `lib/core/webgpu-grid-overlay.ts`

