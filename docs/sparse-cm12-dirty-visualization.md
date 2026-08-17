# Sparse CM12 temporal-dirty observatory

**Purpose.** Make every use of temporal coherence in Sparse CM12 visible and
auditable. The observatory is part of the correctness design, not a late debug
view: incremental publication is forbidden unless every eligible 4³ work tile
has generation-matched provenance. There is no whole-frame/full-MGPCG recovery
path. A change expands only through its declared dependency closures; missing
provenance rejects the candidate generation and preserves the prior accepted
generation.

This design covers face preparation, mass transport, gamma transport, surface
conditioning, pressure coefficients/topology, and 16³ presentation pages. It
does not prescribe the dirty-set implementation itself; it specifies the
evidence that implementation must publish.

## 1. Existing seams to reuse

The repository already has the right pieces, but they currently answer
different questions:

- `lib/core/visualization-registry.ts` declares field views beside their data
  producer and gives each view its authoritative source and legend.
- `lib/core/webgpu-grid-overlay.ts` already binds Sparse CM12 topology, state,
  activity, fine metadata/worklists/samples, and the topology arena. Dirty views
  should be method-contributed field modes using the same camera slice/volume
  controls, not a parallel renderer.
- `lib/core/fine-flood-provenance.ts` is the precedent for generation-aware
  provenance and closure-depth histograms.
- `lib/core/webgpu-fluid-blast-radius.ts` owns a diagnostic field, floods the
  real sparse row graph, and keeps the result on the GPU. The dirty observatory
  should reuse its visual grammar while showing the *actual scheduled closure*,
  rather than recomputing a hypothetical cone.
- `lib/methods/adaptive-mass/adaptive-mass-frame-pipeline.ts` already has stable
  stage seams and GPU timestamp labels. Dirty census counts must use these same
  seams so work and time remain joinable.
- `lib/methods/adaptive-mass/sparse-cm12-pressure-receipt.ts` establishes the
  versioned, JSON-safe, fail-fast artifact pattern.

The prior page-level experiment in
`docs/sparse-cm12-non-pressure-temporal-coherence-handoff.md` is an important
constraint: it skipped 84.4% of publication lanes but saved nothing because the
skipped dry pages were already cheap. The observatory therefore counts 4³ work
tiles, eligible/executed/skipped stage-native work items, and time. A clean-tile percentage by
itself is never presented as a performance result.

## 2. One provenance model, six stage views

The common spatial identity is a 4³ fine-cell work tile. A 16³ storage or
presentation brick contains 64 such tiles. Stable brick identity plus the
six-bit local tile coordinate forms `tileId`; physical slots are not identity.

Each stage owns a compact list of `DirtyTileRecord`s:

```text
tileId
candidateGeneration
stage
originStage
directCauseMask
inheritedCauseMask
closureDepth
parentStage
parentTileId              // one deterministic witness, not all parents
producerGeneration
consumerGeneration
inputGenerationDigest
flags: DIRECT | CLOSURE | UNKNOWN | FORWARDED | PROCESSED | UNCOVERED_WRITE
```

The full dependency set remains in the stage worklist. `parentTileId` is only a
witness that lets a picked tile explain one real path back to a direct cause.
It must never be used to reconstruct or schedule the closure.

The stages are fixed in receipt version 1:

| Stage | Direct dirt | Required local closure | Stage-native work item |
|:---|:---|:---|:---|
| face preparation | changed face velocity/solid geometry, created or retired ownership, boundary source | incident faces and transport donors required by the characteristic bound | oriented face row |
| mass transport | changed density, characteristic/CFL growth, ownership/boundary change | swept donor/receiver footprint and conservative debit/credit partners | cell or transfer row |
| gamma transport | changed gamma support plus all mass events that carry gamma | mass footprint, gamma diffusion neighbours, conservation partners | gamma row |
| surface conditioning | transported density/gamma change or phase crossing | sharpening stencil, scatter destinations, local-return owners | interface tile/cell |
| pressure coefficients | phase crossing, theta/open-fraction/solid change, row creation/retirement | incident pressure edges, connected-component metadata, brick aggregates, hierarchy ancestors | pressure row/edge/group |
| presentation | changed conditioned density or accepted topology | only the filter/restriction footprint of affected 4³ subtiles | explicit presentation subtile |

“Topology changed” is not a stage-wide cause. It is a set of created, retired,
or ownership-changed tiles. Each one enters at depth zero and grows through the
same declared closure law as any other cause.

Cause bits are deliberately physical or structural, not scene-specific:

```text
topology-created       topology-retired       phase-crossing
density-changed        gamma-changed           velocity-characteristic
cfl-growth             moving-solid-sweep      boundary-source
coefficient-changed    dependency-closure      generation-mismatch
capacity-or-provenance
```

A tile can have multiple direct and inherited causes. Cause census counts may
therefore overlap and are labelled as attributions, never as a partition.

### CMD1 v1 GPU layout and optional PKT1 pass packing

The renderer-facing CMD1 ABI stays version 1. Its 16-word header and one
16-word record per stable 4³ tile are unchanged, so an existing producer and
capture remain readable. The byte immediately after the last CMD1 tile record
may begin an optional **PKT1** tail. Absence is valid for the original stage
views, but the pass-packing view renders it as unavailable magenta; it never
infers physical sharing from adjacent stage masks.

PKT1 begins with 16 `u32` words:

```text
0  magic = 0x504b5431 ("PKT1")
1  version = 1
2  headerWords = 16
3  packetWords = 4
4  tileWords = 4
5  packetCount (0..32)
6  tileCount (must equal CMD1 logical tile count)
7  epochCount
8  acceptedGeneration
9  candidateGeneration
10 provenanceGeneration
11 publicationFlags (same COMPLETE/ACCEPTED/REJECTED/FAULT bits as CMD1)
12 uncoveredPackingFaultCount
13 firstFaultTile
14 firstFaultPacket
15 reserved = 0
```

The header is followed by `packetCount` four-word physical packet descriptors:

```text
0 stable packet identity
1 physical dependency epoch
2 logical stage mask (low six bits)
3 execution generation
```

The remaining `tileCount` records are four words each:

```text
0 assigned physical-packet mask
1 executed physical-packet mask
2 stages 0..4 packet index + 1, five packed six-bit lanes
3 stage 5 packet index + 1 in low six bits; remaining bits reserved zero
```

Zero in a stage lane means that logical stage was proven skipped for the tile;
values 1..32 address packet descriptors 0..31. Assigned and executed masks must
match in an accepted publication. Reconstructing the mask from the six stage
lanes must reproduce that mask exactly, each referenced descriptor must contain
the logical stage bit, and all PKT1 generations and publication flags must
match CMD1. Any missing field, stale epoch, uncovered packing write, capacity
overflow, or inconsistent mapping is magenta/red and rejects acceptance. This
is a description of physical dispatch sharing, not permission to merge stage
semantics or their individual provenance records.

## 3. Generations and fail-closed publication

Every candidate frame publishes three identities:

- `acceptedGeneration`: immutable state from which the candidate began;
- `candidateGeneration`: state being prepared;
- `provenanceGeneration`: generation stamped on every dirty list and census.

Each stage also publishes the generations of the authorities it read—for
example topology, density bank, gamma bank, solid motion, pressure coefficients,
and presentation atlas. The stage's `inputGenerationDigest` is a cheap device
comparison; the receipt expands it into named generation values.

`originStage` is the stage where the reason first arose (`external` covers
solid motion, boundaries, injection, and frame controls). `stage` is the
derived consumer currently doing work. This distinction prevents a pressure
coefficient tile inherited from a surface phase crossing from being counted as
a new pressure-origin event. The compact receipt publishes the complete
origin-stage × derived-stage census.

Before any candidate state becomes accepted, a GPU validation pass proves:

1. every eligible tile is exactly one of clean/reused, direct dirty, closure
   dirty, or unknown;
2. processed tiles equal direct plus closure tiles;
3. every dirty record carries the candidate provenance generation;
4. each positive-depth tile has a valid prior-depth parent in the declared
   dependency relation;
5. stage worklist capacity, deduplication, and emitted indirect counts agree;
6. all downstream stage handoffs consumed the upstream generation they name;
7. pressure ancestor closure terminates at the existing hierarchy root without
   turning unrelated fine tiles dirty;
8. presentation subtiles cover every conditioned-density change visible to its
   filter footprint;
9. a shadow write-watch reports zero writes outside the declared dirty closure.

That last check is the **uncovered-write fault**. During bring-up, every stage
may execute its current full path into a shadow bank while the dirty scheduler
predicts what could change. Any bitwise-changed output outside the predicted
closure records the first tile/stage/generation witness, increments the compact
fault counter, and rejects publication. Once skipping is enabled, canary/watch
subsets retain the check without a full readback. A zero fault count is required
for acceptance; absence of the counter is incomplete provenance.

Initially, any failure rejects the candidate generation and stops the run at
the regression harness boundary. Production may later retain the last accepted
generation for display, but it must not silently execute a full-frame path or
publish a partially updated candidate. The observatory draws all unknown,
wrong-generation, capacity-overflow, and missing-parent tiles in magenta/red
from the rejected candidate's diagnostic bank. Rejection evidence must survive
until explicitly acknowledged; rollback must not erase it.

This is “fallback limited to the blast radius” in the strong form: known tiles
outside the blast remain reusable, and an unknown tile prevents publication
rather than expanding dirt to the domain. Once an omitted dependency is found,
the closure law is corrected and the same regression becomes ordinary local
work.

## 4. Spatial overlays

All six views share a single selection control and can draw a slice or volume.
The default composition encodes orthogonal dimensions rather than folding them
into one ambiguous heat map:

- **hue = stage**: face cyan, mass green, gamma violet, surface gold, pressure
  blue, presentation coral;
- **fill = provenance class**: solid for direct, diagonal hatch for closure,
  no fill for proven reuse;
- **alpha = closure depth**: direct is strongest, then decreases by depth;
- **outline = cause family**: structural white, material yellow, motion cyan,
  boundary orange;
- **blinking magenta = unknown or generation mismatch**. This alarm cannot be
  disabled while a rejected candidate is selected.

The main modes are:

1. **Stage dirt.** One stage at a time, with direct/closure/reuse/unknown.
2. **Cause overlay.** Filter by one or more cause bits; inherited causes remain
   hatched and a picked tile names both direct and inherited masks.
3. **Closure depth.** Sequential ramp by dependency depth. Depth zero is white;
   overflow beyond the declared maximum is red.
4. **Generation coherence.** Accepted, candidate, and provenance generations;
   stale inputs and mixed generations are magenta.
5. **Blast radius.** Direct roots are points, closure tiles are filled boxes,
   and deterministic parent witnesses form sparse arrows back to roots. For
   pressure, hierarchy ancestors are shown as nested boxes rather than
   pretending their influence dirties all descendant fine tiles.
6. **Cross-stage handoff.** Selecting a tile shows the same stable tile across
   all stages and draws only the actual upstream/downstream forwarding edges.
7. **Pass packing.** Within each 4³ tile, logical-stage stripes use the stable
   colour of the physical packet and epoch that executed them. Equal saturated
   stripes mean two or more logical stages shared one packet; muted colour is a
   single-stage packet. Different colours show a split across physical passes.
   A v1 producer without PKT1 is magenta rather than guessed from stage timing.

Picking a tile opens a compact explanation:

```text
frame 184 · tile 0x… · pressure coefficients · closure depth 3
accepted/candidate/provenance 183/184/184
direct: none
inherited: phase crossing, dependency closure
witness: surface tile 0x… → pressure edge 91 → brick aggregate 17 → L2 group 3
processed: yes · emitted rows 12 · generation inputs topology=9 density=184 solid=27
```

The overlay reads resident diagnostic buffers. It never requests a full CPU
readback and never changes physics scheduling.

The GPU publication has two layers. Exact per-tile records, bitsets, and first
fault witnesses remain device-resident for the overlay and for scheduling. A
small double-buffered census header is copied to a map-readable receipt ring at
the existing observatory cadence and resolved asynchronously after queue
completion. Neither the next frame nor a dirty-work decision waits for that
mapping. A picked witness may be read asynchronously while paused; there is no
automatic full-record readback.

## 5. Timeline and census

Spatial pictures find *where*; the observatory panel explains *how much* and
*when*.

### Timeline

Rows are the six stages and columns are accepted or rejected candidate frames.
Each cell is a stacked bar of proven reuse, direct dirt, closure dirt, and
unknown. Its tooltip includes stage time, work-item count, maximum closure
depth, generation tuple, and maximum physical blast distance. Rejected frames
have a magenta border and remain selectable.

Event lanes above the grid mark direct topology changes, phase crossings,
moving-solid sweeps, boundary/injection changes, CFL band growth, and capacity
or provenance faults. This makes a wide pressure closure visibly traceable to a
specific local event instead of looking like random scheduling noise.

### Census

For the selected window, publish:

- eligible, proven-reused, direct-dirty, closure-dirty, unknown, and processed
  4³ tile counts by stage;
- direct and inherited cause attributions by stage;
- closure-depth histogram and `closureDirty / directDirty` amplification;
- stage GPU time and time per processed tile/work item;
- root count, unique touched count, forwarded count, axis-aligned fine-cell
  bounds, maximum dependency depth, and maximum Manhattan fine-cell distance;
- accepted/candidate/provenance and named input generations;
- producer/consumer generation by stage and the origin-stage × derived-stage
  matrix;
- eligible, executed, and skipped stage-native work items, kept distinct from
  work-tile counts;
- uncovered-write fault count and first fault witness;
- physical packet and epoch identities, logical-stage masks, single/coalesced/
  split tile partition, and logical stage-tile work saved by shared packets;
- high-water capacities, tombstones, overflow, duplicate, and missing-parent
  counts.

Always show tile share beside GPU time. The earlier presentation experiment
proved that “84% skipped” can mean “nothing saved.” A stage is considered a
successful optimization only when both processed work and measured time fall.

## 6. Pressure blast-radius semantics

Pressure needs two different notions of reach, shown separately:

- **coefficient dirt:** the exact rows, edges, brick aggregates, and hierarchy
  ancestors whose operator data changed;
- **solution influence:** the elliptic response of the pressure field, which
  can be global even when coefficient maintenance is local.

The dirty overlay is the first notion. It must not reuse the existing pressure
blast-radius label without qualification. A topology event may touch three
fine tiles, twelve incident edges, two brick aggregates, and one ancestor at
each hierarchy level while leaving every other coefficient bit-identical. The
nested ancestor boxes make that local maintenance visible. Krylov iterations
may still read the global solution vector; that is solution influence, not a
reason to rebuild global topology.

Receipt `blast` fields record direct roots, unique touched tiles, forwarded
tiles, maximum closure depth, maximum Manhattan distance in fine cells, and an
exclusive fine-cell bounding box. The spatial buffer retains exact membership;
the receipt is only its census.

## 7. Artifact contract and standalone reporter

`tools/report-sparse-cm12-dirty.ts` defines receipt version 1 and validates it
without simulator imports. It accepts a JSON receipt and writes Markdown, JSON,
or a dependency-free SVG dashboard:

```bash
node --import tsx tools/report-sparse-cm12-dirty.ts \
  --input=artifacts/sparse-cm12-dirty.json --format=markdown

node --import tsx tools/report-sparse-cm12-dirty.ts \
  --input=artifacts/sparse-cm12-dirty.json --format=svg \
  > artifacts/sparse-cm12-dirty.svg
```

The strict checks intentionally make partial instrumentation painful:

- all six stages are required;
- the four-way tile census must equal eligible tiles;
- processed must equal direct plus closure;
- depth zero must equal direct and positive depths must equal closure;
- blast root/touched/depth fields must match the census;
- every stage must name at least one input generation;
- executed plus skipped native work must equal eligible native work;
- producer and consumer generations are explicit, and every accepted consumer
  generation must equal the candidate generation;
- every accepted stage must report zero uncovered-write faults;
- an accepted frame may contain no unknown tile and must have
  `provenanceGeneration == candidateGeneration`;
- a rejected frame must carry a reason.

Receipt version 1 also accepts an optional `frame.packing` object, versioned
independently as packing version 1. It contains the same five-way tile
partition shown by the overlay (`reused`, `singleStage`, `coalesced`, `split`,
`unknown`), physical packet descriptors with per-logical-stage tile counts,
and an explicit dependency-epoch DAG. Packet indices are dense 0..31 to match
PKT1 masks; each packet belongs to exactly one epoch; packet/epoch generations
must agree; and an accepted packing census has no unknown tile or uncovered
packing fault. Omitting `packing` preserves compatibility with an original v1
receipt and is reported as **unavailable**, never as zero packets or zero
coalescing.

Malformed receipts exit 1. A well-formed receipt containing a fail-closed
rejection exits 2 after emitting the report. This lets the dam-front and weak
symmetric-expansion harnesses preserve the artifact while still failing the
gate.

## 8. Regression use, without unit tests

Instrumentation lands with the aggressive incremental path, not in advance as
mock-only tests. The two regression lanes are:

1. **64³ dam-front properties.** Track the advancing front's connected dirty
   components, maximum upstream/downstream reach, closure amplification by
   stage, mass/gamma conservation, pressure true residual/divergence, and the
   absence of unexplained dirt behind the front. Local topology changes may
   grow ancestor boxes but may not turn the pressure coefficient tile census
   domain-wide.
2. **Weakened symmetric expansion.** Require reflected dirty occupancy and
   closure-depth histograms to agree within the weakened tolerance, while
   retaining existing mass, gamma, surface, and pressure authority gates.
   Compare direct-cause masks separately from inherited closure so symmetric
   physics cannot be “passed” by symmetric over-dirtying.

The initial fail-closed campaign runs with retained rejected diagnostics and
no full-frame fallback. Every rejection is classified as a missing cause,
missing closure edge, generation mismatch, capacity issue, or real physics
regression. Fixes expand only the declared local closure and add a named receipt
cause. Once both lanes run without unknown provenance, the same receipts become
performance evidence: dirty share, closure amplification, work emitted, and
GPU time must all move together.

## 9. Implementation order

1. Reserve generation-stamped per-stage dirty records, origin/derived-stage
   fields, census headers, uncovered-write witness, rejected diagnostic bank,
   and stable 4³ tile identity.
2. Instrument the current full-work path as `processed` while computing direct
   and closure provenance in shadow. Fail closed on any unknown, but do not yet
   skip work.
3. Add the six method-contributed field views and the timeline/census panel.
4. Turn on skipping stage by stage, beginning with presentation subtiles and
   pressure coefficient maintenance; keep the same receipts before and after.
5. Run the 64³ dam-front and weakened symmetric-expansion lanes continuously;
   preserve rejected receipts and selected spatial diagnostics as artifacts.
6. Remove shadow/full-work instrumentation once every scheduled work item is
   generated from the provenance lists and the two lanes demonstrate stable
   locality. Retain the observatory and fail-closed validation permanently.

The critical invariant is simple: a tile is either proven reusable or carries
a complete, inspectable reason for work. There is no invisible third state.
