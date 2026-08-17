# FPL1 presentation packet executor (FPP1)

FPP1 is the standalone presentation consumer for the GPU-authored FPL1 frame
plan. Its native specialization is B16/P16. One workgroup owns one physical
presentation page; lanes 0..63 own its 4³ tiles and each lane evaluates the 64
samples in its tile. B8/P8 uses lanes 0..7 and B4/P4 uses lane 0. The page
kernel is indirectly dispatched from a compact list, so a clean page has no
presentation workgroup at all.

The scheduling scan is not presentation work: one 64-lane workgroup per
physical brick validates FPL1 and compacts only pages with presentation stage
bit 5. It performs one atomic append per dirty page, never per tile or sample.
The resulting three-word indirect header is GPU-authored and copy-isolated;
the CPU uploads neither count nor frame parity.

## Accepted causes

Every scheduled tile must name at least one exact cause and no cause outside
this set:

- density changed;
- phase crossed the liquid isovalue;
- topology owner created or retired;
- presentation page activated or retired;
- embedded/open boundary source changed;
- dependency closure, only on a positive-depth closure tile.

A direct tile must also carry a non-closure cause. Topology or page lifecycle
causes require all valid tiles in the page. Unsupported, missing, stale,
uncovered, or incomplete provenance omits only that page and publishes its
local fault. A malformed global ABI or compact-list overflow dispatches zero
pages.

## Transaction and receipts

`PreparePage` cooperatively constructs the same per-page cache used by the
baseline kernel. `ExactSample` returns the packed baseline sample and a status.
Samples go to a non-visible candidate bank. Only after all dirty tile masks
equal their executed masks may the dedicated commit kernel copy candidate
words to the accepted page. `CommitCandidate` publishes its generation after
that copy; a partial page never becomes renderer-visible. Successful lanes
then write the FPL1 stage-5 execution receipt. FPP1 publishes page/tile
execution censuses, a generation receipt, exact-sample count, and first local
fault.

## Byte-exact integration checklist

Do not integrate this executor until the 2 s HEAD oracle is stable. Integration
is acceptable only when every published presentation word matches HEAD.

1. Allocate FPP1 in a dedicated 256-byte-aligned storage arena. Initialize it
   with `createSparseCM12FramePlanPresentationInitialWords`; do not overlap the
   pressure coefficient, candidate topology, activity, or FPL1 arenas.
2. Expose a copy-isolated `INDIRECT | COPY_DST` buffer. After packet finalization,
   copy exactly the 12-byte `layout.indirectBinding` into it. The host records a
   fixed copy and indirect dispatch but never reads or uploads the page count.
3. Preserve FPL1 frame authority: use Current slot, accepted generation,
   accepted topology generation, and accepted parity. Reject mixed generations;
   never substitute the host `parity` field.
4. Populate the six presentation causes at the original producer writes:
   final density/phase classification, topology commit/retirement, page
   activation/retirement, and boundary mutation. Lifecycle events force the
   complete valid page mask. Do not reconstruct causes with a later scan.
5. Build the immutable physical-brick-to-page table in the FPP1 initializer
   from accepted fine metadata. Implement `PageMatches` to require exact
   physical brick, logical key, span, page key, topology generation,
   activation and retirement identity. A mismatch omits that page. Static
   construction metadata is not runtime scheduling authority.
6. Extract the existing `publishSparseLevelSet` cache setup into `PreparePage`
   without changing expression order, precision, clamp points, or loop nesting.
   This includes `cacheFirst`, `cacheDimensions`, coarse clamping, and
   `restrictedPresentationDensity` accumulation order for macro leaves.
7. Extract the existing per-sample body into `ExactSample` verbatim. Preserve:
   page/source decoding; sample scale; brick/rung range lookup; wet predicate;
   embedded-boundary test; scale-one direct cell address; coarse trilinear loop
   order `dz,dy,dx`; `presentationPhi`; half packing; and flag construction.
8. Change only enumeration: lane owns one 4³ tile and walks local sample 0..63.
   The tile/sample-to-page index function is a bijection onto the same P³
   indices, so independent writes land on the identical words. Do not fuse or
   reassociate any density or interpolation arithmetic.
9. Expand the existing `fineSamples` allocation to accepted and candidate
   regions without changing accepted offsets. `StoreCandidate` writes only the
   alternate region. After exact dirty/executed equality, the commit kernel
   copies precisely those dirty tile words back to the accepted prefix, then
   `CommitCandidate` publishes the generation. Clean pages retain their
   accepted words and receive no execution or commit dispatch.
10. Schedule fixed GPU work in this order: begin packet; brick validation and
    compaction; finalize packet; copy indirect triplet; indirect candidate-page
    execution; indirect accepted-page commit; FPL1 stage-5 coverage
    verification; finalize FPP1 execution. No CPU readback, conditional encode,
    or runtime-owned count is permitted.
11. Before replacing the baseline, shadow both outputs for B4/P4, B8/P8 and
    B16/P16. Compare metadata and every packed sample word after each substep.
    Report the first mismatch as frame/generation, page, brick/key/span, tile,
    sample coordinate, cause mask, baseline word and FPP1 word.
12. Gate 2 s dam-front and weakened symmetric expansion on exact density,
    gamma/internal state, velocity, pressure, divergence, and presentation
    sample hashes. Topology/workset hashes remain diagnostic when authoritative
    physical hashes match. Any presentation mismatch or uncovered receipt leaves
    FPP1 fail-closed. It must not select the baseline publisher at runtime.

## Current resident cutover map

The current resident has one mutable storage arena at binding 12 (`activity`)
and one presentation payload at binding 15 (`fineSamples`). The least invasive
cutover appends FPL1 after `canonicalMembershipLayout`, then appends the
256-byte-aligned FPP1 subarena after FPL1. `initialActivity` grows through the
FPP1 end and receives both initializer slices at their absolute bases. This
preserves the dirty scheduler, temporal, incremental-activity, canonical
membership, and pressure layouts already in the prefix and requires no new
resident bind-group entry.

`fineSamples` keeps its accepted prefix byte-for-byte and grows to two payload
capacities. Renderer and `globalFineLevelSetSource.samples` continue to expose
only the accepted prefix semantics. FPP1 hooks address the candidate bank at
`payloadCapacityWords + page * P^3 + localIndex`; commit copies only the sealed
dirty tile words to the accepted prefix. The existing four-byte `fineWorkA`,
`fineWorkB`, and rollback placeholders are not candidate storage.

The immutable `brickPages` initializer table is derived once from
`fine.metadata[4 * page + 3]` using the exact source decoding in
`publishSparseLevelSet`. For B16/P16 there is one page per physical brick,
including one macro page for a span greater than one. Duplicate, missing, or
out-of-range brick mappings are construction errors; a generation/key/span
mismatch later is a local GPU page fault.

Two copy-isolated indirect buffers are required: 72 bytes for FPL1's six fixed
packet triplets and 12 bytes for the compact FPP1 page count. The command graph
always encodes the storage-to-indirect copies and indirect dispatches. It never
branches on a mapped page count. FPP1 selects the density bank from
`cm12FramePlanAcceptedParity()` and does not read `p.frame.w` or the resident's
host `parity` field.

Replace all three runtime publication sites together:

- the normal `presentation-publication` stage;
- `encodeInitialPresentation`;
- `encodeLiquidInjection` after topology commit and density injection.

Each site must author/accept FPL1, build and seal FPP1, copy the 12-byte
indirect triplet, execute candidates indirectly, commit candidates indirectly,
verify FPL1 presentation coverage, and finalize FPP1. Initial publication
bootstraps accepted FPL generation/parity zero and every visible page with its
complete valid mask. The first physics frame therefore advances to parity one,
matching the existing A-to-B field update without consulting host parity.
Injection marks
the exact density, phase, topology, activation, retirement, and boundary
causes at their producer writes before it enters the same packet path.

Keep `publishSparseLevelSet` compiled only behind an explicit QA capture path.
That path binds a separate oracle payload, runs on explicit test request, and
compares every accepted FPP1 word with the oracle, reporting the first mismatch.
It is never encoded by normal advance, initialization, or injection and is not
a fault fallback. The Grid-structure source publishes FPL1 by adding
`framePlan: { kind, plan: { buffer: activity }, indirectSource, indirect,
layout }` to `sparseAdaptiveGridSource`; missing or invalid publication remains
magenta through the already optional core ABI.

The standalone implementation is in
`sparse-cm12-frame-plan-presentation.ts` and
`sparse-cm12-frame-plan-presentation.wgsl.ts`. The Naga checker validates all
three B/P specializations without a GPU run.
