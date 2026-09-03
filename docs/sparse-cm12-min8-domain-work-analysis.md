# Sparse CM12 minimum-cell-size performance analysis

## Scope

This investigation reproduces the mini dam-break 64 scene under Dawn/Metal with
Sparse CM12 B8/P8 in three configurations:

1. ordinary adaptive resolution;
2. a minimum cell size of 8 over the complete fluid domain (the best-case work
   reduction); and
3. a minimum cell size of 8 over the authored initial dam only (the realistic
   local-region case).

Each result is the median of 24 hardware-timestamped frames after 8 warm-up
frames. Captures are spaced by 110 ms. The artifacts contain the exact scene
bounds, resolved method values, git provenance, authority receipts, terminal
work counts, phase samples, and validation errors.

## Result

The region feature is working. It reduces the numerical CM12 graph by two to
three orders of magnitude in the full-domain experiment, but frame time falls
by only 1.69x.

| measurement | ordinary | full-domain min-8 | retained | reduction |
|---|---:|---:|---:|---:|
| accepted cells | 112,976 | 1,000 | 0.89% | 113.0x |
| accepted rows | 327,128 | 2,415 | 0.74% | 135.5x |
| pressure cells | 46,951 | 183 | 0.39% | 256.6x |
| pressure rows | 141,222 | 549 | 0.39% | 257.2x |
| temporal scalar cells | 108,315 | 1,035 | 0.96% | 104.7x |
| temporal scalar rows | 315,609 | 2,517 | 0.80% | 125.4x |
| total GPU frame | 51.12 ms | 30.28 ms | 59.2% | 1.69x |
| non-pressure GPU work | 39.52 ms | 20.84 ms | 52.7% | 1.90x |
| pressure solve | 11.53 ms | 9.96 ms | 86.4% | 1.16x |

The initial-dam-only region reaches 40.50 ms. It retains 77,701 accepted cells
and 222,991 rows, so its smaller 1.26x speedup is consistent with its smaller
graph reduction.

## The fixed work shape

The new work-shape receipt shows that all three runs construct the same physical
domain:

| capacity | value |
|---|---:|
| finest domain | 64 x 64 x 64 = 262,144 cells |
| logical brick domain | 8 x 8 x 8 = 512 bricks |
| packed physical bricks | 512 |
| immutable template cells (all B8 rungs) | 299,520 |
| immutable template rows | 938,688 |
| full template cell scan | 4,680 workgroups |
| full template row scan | 14,667 workgroups |
| conditioning clears | 10.78 MB/frame |
| pressure scratch clear | 12.80 MB/frame |
| pressure hierarchy groups / edges | 73 / 312 |
| scalar-result tile capacity | 4,096 |
| face-preparation authority leaves | 3,667 |
| presentation pages | 512 |
| resident allocation | 361.8 MB |

The full-domain region changes the selected rung *inside* each B8 brick. It does
not merge the 512 logical bricks, shrink the immutable multi-rung template
library, reduce pressure hierarchy capacity, or reduce the one-fine-page-per-
brick presentation allocation. The terminal activity census reports 398
retained/active bricks, but host-visible `packed.brickCount` remains 512 and is
the width used by the recurring resident-brick and presentation schedules.

That is the central reason the speedup does not track cell count.

## Phase evidence

| phase | ordinary | full min-8 | retained |
|---|---:|---:|---:|
| pressure solve | 11.53 ms | 9.96 ms | 86.4% |
| conservative transport | 9.44 ms | 3.47 ms | 36.8% |
| face preparation | 5.31 ms | 0.79 ms | 14.8% |
| pressure topology/cache | 3.41 ms | 3.21 ms | 94.2% |
| surface conditioning | 3.41 ms | 2.10 ms | 61.5% |
| pressure projection | 3.28 ms | 0.66 ms | 20.0% |
| shadow topology + transfer | 3.21 ms | 2.10 ms | 65.3% |
| velocity extension | 2.88 ms | 1.57 ms | 54.5% |
| presentation | 2.23 ms | 1.90 ms | 85.3% |
| activity measurement | 1.38 ms | 0.72 ms | 52.4% |
| resolution planning | 0.46 ms | 0.59 ms | 128.6% |

The row/cell numerical phases do respond: face preparation and projection fall
to 15-20%, and gamma diffusion falls to one third. Pressure topology,
presentation, pressure solve, candidate topology, velocity extension, surface
authority, and activity retain large floors. This is a multi-stage domain-shape
problem, not one pressure dispatch problem.

## Static inventory

### Template-capacity work

- Three conditioning clears cover 36 bytes per template cell every frame.
- Candidate pressure scratch is cleared at its construction capacity.
- Shadow topology generation scans every template cell and row when a topology
  transaction is prepared. In this scene that is 4,680 + 14,667 workgroups to
  publish about 1,000 cells and 2,415 rows.
- Several authority structures retain template-derived tile/leaf capacities,
  including the 4,096-tile scalar ingress and 3,667-leaf face authority.

### Packed-domain / resident-brick work

- Pressure aggregate smoothing/restriction/correction repeatedly walks
  `packed.brickCount` for every iteration. Its cells are sparse, but its
  aggregate graph is still the 512-brick construction graph.
- Activity topology marking/aging and resolution planning traverse the packed
  brick domain.
- Candidate synthesis, cell transfer, six-sided face transfer, shadow writes,
  and velocity-root publication are encoded over packed bricks and branch out
  inside the shader when a brick is not scheduled.
- Topology scheduling and commit validation contain serial or 64-lane loops
  over every packed brick.

### Presentation-domain work

Presentation owns one P8 fine page per B8 packed brick. Planning, closure,
sealing, receipt marking, packet construction, verification, and rejection all
traverse this 512-page/brick domain. Coarsening CM12 physics does not coarsen the
render publication lattice, which explains the 85% retained presentation cost.

### Fixed transaction work

The frame-control, scalar-result, face-preparation, pressure-topology,
persistent-cache, and presentation protocols use many singleton begin/finalize/
seal/verify kernels and storage-to-indirect copy boundaries. These transactions
are important correctness authorities, but their cost does not scale with the
accepted graph. They become visible once accepted physics collapses.

## Changes made

1. The Dawn stage-cost probe now accepts `--minimum-cell-size` and
   `--region-scope=domain|initial-dam`, records the exact authored bounds, and
   emits a work-shape receipt. This makes the regression directly reproducible
   and prevents accepted-count improvements from hiding unchanged capacity.
2. `reconstructShadowFaces` already consumed the compact shadow-row worklist,
   but the host launched it at full template-row width. The host now snapshots
   the GPU-authored shadow indirect arguments and launches the shader at the
   compact row count. For the measured min-8 frame this changes the width from
   14,667 workgroups to roughly 38. It preserves row IDs, arithmetic, and
   ordering. The whole candidate phase moves by at most one 65.5 us timestamp
   quantum because the upstream template scans and packed-brick transfer remain.
3. The topology transaction now publishes a compact accepted-leaf manifest.
   It is double-buffered by the existing accepted topology slot, and therefore
   has the same generation and commit point as the accepted cell and row lists.
   It is not an independently maintained activity structure.
4. The same stable compaction pass also publishes a transaction-local topology
   delta: the leaf IDs whose rung is being prepared in this transaction. The
   accepted manifest describes current structure; the delta describes lifecycle
   work which cannot safely be inferred from the current active set.
5. Candidate-cell transfer is the first non-pressure consumer. On the terminal
   full-domain-min-8 frame it launches exactly 20 workgroups from the delta,
   rather than 512 packed-brick workgroups which immediately test the same
   scheduled bit. Injection retains the broad entry point because it may create
   receiver work inside its command buffer.

## The unifying representation

The useful minimum representation is a generation-stamped leaf ID, not a new
fat leaf record:

- the leaf ID indexes the existing activity record for active state and accepted
  rung;
- the same ID and rung index the immutable template cell range;
- accepted cells are therefore the concatenation of leaf ranges and may remain
  a derived cache during staged migration;
- shared face rows remain an explicit compact list, because a row may have two
  leaf owners and cannot be obtained by blindly concatenating per-leaf ranges;
- the topology delta is authored by the same transaction for created,
  rung-changed, or retired lifecycle work.

The Dawn receipt validates this model rather than assuming it. For the
full-domain-min-8 capture:

| invariant | result |
|---|---:|
| accepted leaf IDs | 398 |
| expected active leaves | 398 |
| leaf ID mismatches / order violations | 0 / 0 |
| topology-delta leaf IDs | 20 |
| expected scheduled leaves | 20 |
| delta mismatches / order violations | 0 / 0 |
| accepted cells / cells reconstructed from leaf ranges | 1,000 / 1,000 |
| missing, unexpected, duplicate, or overlapping cells | 0 |
| accepted rows / rows reconstructed from owner requirements | 2,415 / 2,415 |
| shared two-owner rows | 1,383 |
| missing, unexpected, duplicate, or invalid rows | 0 |

This is deliberately smaller than the earlier attempted architecture: consumers
receive IDs and one generation, while field storage, template ranges, activity,
and row ownership remain where they already are. There is no second copy of
rung, bounds, offsets, neighbors, flags, and lifecycle state to reconcile.

## Leaf-ID locality

Both lists use a stable prefix scan over physical brick IDs. Atomic append was
avoided because it would make list order scheduler-dependent and scatter nearby
bricks between workgroups. Physical brick IDs follow the atlas's logical linear
order, so ascending IDs keep adjacent bricks and their activity records close,
make accepted rung lookups sequential, and give deterministic captures.

The next locality step should be measured, not assumed: compare ascending
physical ID against a Morton-ordered leaf manifest. Morton order should improve
three-dimensional neighbor reuse for stencil consumers, but it can worsen
sequential access to the current linear activity and template metadata. A more
promising later layout is a small structure-of-arrays hot cache, indexed in
manifest order, containing only accepted rung and cell-range start/count. That
should be introduced only when several consumers amortize its build; copying a
fat per-leaf record would recreate the duplicate-authority problem.

## Reusable execution-graph audit

The repository already contains two relevant construction-time ABIs which are
not yet the resident shader's physical access path:

- LOD1 is an eight-byte dense logical-brick owner directory. It replaces the
  resident's hash probe used under every `ownerCellAt` query.
- HTP1 contains 32-byte cell geometry, tagged rows with inline common two-term
  stencils, variable terms, cell-incidence CSR, directed-edge CSR, and row rung
  requirements.

The resident currently exposes HTP1-compatible function names over its older
template arrays rather than binding HTP1 itself. The low-risk unifying design
is therefore not another topology format. It is a compact accepted-generation
overlay of leaf/cell/row ordinals over LOD1/HTP1, with separate frame-local
dynamic stencil results.

HTP1's current constructor deliberately admits literal open-domain geometry.
Moving rigid bodies and evolving embedded-boundary coverage must remain dynamic
state (or carry a separate geometry epoch); they must not be frozen into the
accepted topology generation. This limitation is useful because it keeps the
shared graph structural rather than turning it into another physics authority.

### Existing Dawn A/B evidence

A three-replay, five-frame mini-dam-64 audit exercised the existing compiled QA
specializations. Every arm passed its lane receipts and was physics-bit-exact
to baseline. The short lane is variable, so these are architectural signals,
not final acceptance numbers.

| specialization | face preparation | mass transport | interpretation |
|---|---:|---:|---|
| baseline | 9.9615 ms | 8.5197 ms | reference |
| characteristic cache | 10.0925 ms | 8.5197 ms | cache alone did not help |
| face row packets | 12.1242 ms | 9.8304 ms | bitmap packets alone hurt occupancy/locality |
| face packets + characteristic cache | 8.4541 ms | 10.2236 ms | face improved 15.1%; unrelated mass variance regressed |
| mass rung packets | 10.6824 ms | 6.4225 ms | mass improved 24.6% |
| mass local atomics | 12.4518 ms | 7.2745 ms | mass improved 14.6% |
| rung packets + local atomics | 10.4858 ms | 10.1581 ms | naive composition regressed 19.2% |

The important result is compositional: predecoded structure is useful when it
removes repeated owner resolution, and dynamic reuse is useful when paired with
an execution layout that preserves occupancy. Independently reasonable caches
cannot simply be stacked.

### Mass and gamma transport

Each active scalar cell currently resolves its tile coordinate back to an owner
cell in all three transport kernels. `traceGammaAndBeta` and
`gatherConservativeDensity` then recompute the same departure characteristic and
eight-corner stencil. `scatterDensityDeficit` builds a separate arrival
characteristic and stencil. Each velocity sample performs a probe plus eight
corner owner queries; RK2 characteristic tracing repeats that sample up to two
times per substep for as many as sixteen substeps.

The shared graph can remove tile-to-cell rediscovery and accelerate every
dynamic corner query through LOD1. A frame-local cache should then store each
cell's departure and arrival endpoint/stencil after velocity preparation. The
departure record is consumed by both trace/beta and final gather. Scatter
transpose work still needs an ownership design; the existing local-atomic arm
shows that reducing global atomics can help, but its regression when combined
with rung packets rules out simply layering the two implementations.

### Candidate transfer

Launch compaction is not the candidate opportunity. Compacting candidate cells
and then candidate faces changed the whole phase by at most one timestamp tick.
The scheduled work itself repeatedly reconstructs static operators:

- cell transfer derives dyadic parent/child overlap for every local cell;
- face transfer walks boundary-cell incidence, row terms, claimant selection,
  and patch mapping for all six sides;
- shadow face reconstruction rediscovers changed row ownership and row terms.

All rung pairs are powers of two and finite in number. Construction can publish
an immutable `(old rung, new rung)` cell transfer catalog and
`(old rung, new rung, side)` boundary-face sparse operators. The accepted
generation supplies compact cell/row ordinals; the lifecycle delta selects
which brick/operator instances execute. Dynamic density, velocity, and
conservation reductions remain transaction data.

### Face preparation

For every live row, `prepareTransportFaceRow` first scans row terms to determine
liquid and extended-velocity contact, then traces a departure characteristic
and samples velocity again at its endpoint. HTP1 can make the common two-term
row a single contiguous packet and LOD1 can replace the hash-based owner lookup
inside every characteristic sample. A cause-keyed frame cache may reuse the
characteristic only when its velocity dependencies are unchanged.

The A/B result says the packet must be a compact contiguous live-row stream.
The existing 256-row authority-bitset packet makes each lane test four possible
rows and regresses by 21.7% on its own. Coupling it to valid characteristic
reuse recovers a 15.1% improvement, demonstrating both reuse and the cost of a
poor packet shape.

### Gamma diffusion

HTP1 already contains the row terms, incidence CSR, and directed edges needed
for a canonical-pair representation. The accepted overlay should publish active
pair packets plus compact cell-to-pair incidence. One pass computes each
quantized pair receipt; a cell-owned gather applies signed receipts. This removes
four global atomics per active pair and permits direct output without clearing a
template-capacity accumulator. A row-pair lookup cache without this ownership
change would optimize only secondary metadata work.

## Implemented owner-directory cut

The first reusable cut is now resident. Construction builds and exhaustively
validates LOD1 once, then appends a specialized logical-owner plane to the
existing immutable topology binding. No new binding, mutable authority, or
per-frame publication phase was introduced.

Production stores two owner IDs per `u32` (a `0xffff` invalid sentinel) when
the atlas has fewer than 65,535 physical bricks. The hot path is one directory
load, one shift/mask, and the existing physical-brick/rung lookup. Larger
atlases automatically retain the validated 32-bit LOD1 accessor. The legacy
hash/span ladder and the self-describing eight-byte LOD1 record are retained as
construction-selected QA oracles, not uploaded together in ordinary production.

The first generic direct accessor was bit-exact and measured 10.2% faster mass,
8.8% faster face preparation, and 4.2% faster non-pressure work than the hash
path in the three-replay paired capture. Removing per-query header and
origin/span validation strengthened the measured hash-to-production gap to
27.3% in mass and 21.3% in face preparation in the next capture. These short
Metal timings are quantized and thermally noisy, so the result is evidence of
direction and attribution rather than a stable product benchmark.

Packing the already-trusted owner plane from eight bytes per logical brick to
two bytes produced a further, smaller five-replay gain: 1.2% mass, 1.9% face
preparation, and 2.3% total non-pressure. This is consistent with mini64's
directory already fitting in cache; the larger benefit should be on domains
whose logical-owner plane competes with field data for cache residency.

A four-word frame-local mass tile descriptor was also implemented and tested.
It was physics-bit-exact, but its compiler pass and repeated arena loads made
mass 27--57% slower across the paired arms. It was removed completely. This
rejects materializing medium-sized descriptors that save only an already-cheap
owner lookup; future packets need to cache expensive characteristic/stencil
results or provide a contiguous no-atomic ownership traversal.

Receipts:

- `artifacts/sparse-cm12-min8-analysis/logical-owner-mass-audit.json`
- `artifacts/sparse-cm12-min8-analysis/logical-owner-packed-mass-audit.json`
- `artifacts/sparse-cm12-min8-analysis/packed-owner16-audit.json`

## Implemented adaptive-structure ABI

The resident template now builds one immutable adaptive graph at construction
and serializes several indexed views over the same stable cell and row IDs:

- brick/rung to owned-row CSR;
- row to canonical negative/positive gamma pairs and cell-to-pair incidence;
- brick/accepted-rung/side/boundary-cell to claimed exterior rows; and
- the existing cell ranges, row terms, cell incidence, and row requirements.

The frame transaction adds only mutable views: compact accepted/shadow leaf and
row lists, a topology-delta leaf list, and a two-bit row-membership word keyed by
accepted/shadow slot. Rebuilding the shadow slot clears membership by traversing
the previous compact shadow-row list, not by scanning the template row domain.
The fused leaf workgroup then publishes cell IDs, owned row IDs, and slot
membership together.

This is deliberately a graph plus views rather than a universal fat record.
Immutable adjacency is stored once. Current membership is slot-qualified.
Dynamic characteristic results are per-frame cell packets. Consumers whose
transaction phase is not yet explicit retain requirement-based `rowAccepted`;
globally replacing it with one current-membership predicate faulted the velocity
extension authority because some consumers observe different commit phases.

### Packed size receipt (mini64/B8)

| view | bytes |
|---|---:|
| row-owner CSR | 3,762,948 |
| gamma pair graph | 22,681,352 |
| candidate boundary-face graph | 2,508,288 |
| accepted/shadow row membership | 3,754,752 |
| dynamic departure-cache capacity | 16,773,120 |

The candidate-face experiment first indexed every candidate rung and patch. It
was exact but added 13.5 MB and did not change changed-frame time. The retained
view stores each claimed boundary row once per accepted rung and performs only
the cheap candidate-patch test dynamically, reducing that graph to 2.51 MB.
Gamma pair records omit row IDs because row ownership is already encoded by the
row-pair CSR; records contain only their two endpoint IDs.

### Reused access paths

- Gamma diffusion computes one fixed-point rho/gamma packet per accepted pair,
  stamps it with the current topology generation, and gathers through compact
  cell-to-pair incidence. Production performs no gamma accumulator clears and
  no pair atomics. The two iterations remain separate because iteration two
  consumes the completed cell gather from iteration one.
- Conservative mass transport stores the backward characteristic's eight-cell
  stencil once in `traceGammaAndBeta`; the final gather reuses it. Eight stable
  cell IDs are packed as 24-bit values into six words and the eight f32 weights
  remain exact. The intervening deficit scatter is a real global dependency
  boundary and cannot be fused with the gather.
- Candidate transfer consumes the topology delta for cells, all six face sides,
  shadow writes, and velocity-root publication. Face receipt initialization is
  fused into the cell-transfer packet. Boundary rows come directly from the
  immutable face view rather than rediscovering incidence, axis, plane, and
  claimant ownership.
- Shadow construction is one leaf traversal. Cell ranges, row-owner ranges,
  row-list publication, and slot membership share the leaf/rung lookup and make
  one output reservation per leaf.

### Short Dawn measurements

The eight-frame min-8 receipt is a directional measurement, not the 24-frame
acceptance gate. Compared with the first exact owner-row catalog capture:

| stage | before | adaptive-structure ABI | change |
|---|---:|---:|---:|
| conservative scalar transport | 3.0802 ms | 1.6384 ms | -46.8% |
| gamma diffusion | 0.4588 ms | 0.3277 ms | -28.6% |
| quiescent candidate transfer | 0.1311 ms | 0.0655 ms | one timestamp quantum |
| changed candidate transfer | 10.6168 ms | 10.6824 ms | unchanged/noisy |

The changed candidate result is important: after topology discovery is removed,
its cost is the actual multi-field conservative transfer, reductions, and state
publication. More adjacency metadata will not remove that field bandwidth.

Production conditioning clears fell from the previously reported 10.78 MB to
3.59 MB per frame; pressure/candidate scratch clears fell from 12.80 MB to zero.
The structure increases resident allocation from 365.6 MB in the first owner-row
capture to 398.5 MB. Future views must therefore replace enough rediscovery to
justify their cache footprint.

The legacy recomputation arms remain construction-only QA oracles. Both the mass
and gamma packet paths pass the five-frame physics gates but introduce an f32
materialization boundary and are not JSON-bit-identical to recomputation. The
B8 mini-front mass-drift test currently reports the identical 0.003909 drift
with production packets and with both caches disabled, so that pre-existing gate
failure is not caused by this structure cut.

Receipts:

- `artifacts/sparse-cm12-min8-analysis/min8-all-final-structure.json`
- `artifacts/sparse-cm12-min8-analysis/structure-cache-split-equivalence.json`
- `artifacts/sparse-cm12-min8-analysis/structure-cache-equivalence.json`

## Rejected experiments

The following were measured and removed rather than shipped:

- Replacing template-sized buffer clears with accepted-cell clear shaders moved
  work into the compute phases, regressed dense transport substantially, and
  did not improve total min-8 frame time.
- Rebuilding shadow lists from scheduled cell incidence preserved counts and
  receipts but did not reduce the candidate phase; atomic incidence discovery
  merely exchanged one full-library cost for another.
- Compacting pressure-bearing bricks and launching aggregate kernels from that
  list preserved terminal counts and receipts but made pressure slower (13.30
  ms versus 9.96 ms in the matched min-8 capture). The persistent aggregate and
  hierarchy data path, not the direct launch width, remains the limiting work.
- Substituting the accepted active-leaf list for candidate transfer faulted:
  candidate transfer is lifecycle work and needs changed leaves, including
  states not represented by the post-change active set. This directly motivated
  the transaction-local delta.
- Classifying presentation from only accepted leaves failed its byte oracle
  because inactive/retired pages must be revisited to clear stale wet state.
  Its future compact domain is accepted visible leaves union presentation
  lifecycle delta, not accepted leaves alone.

## Plan

### P0: make every capacity explicit and gate regressions

- Keep the work-shape receipt in the Dawn probe.
- Add a CI comparison that reports accepted work, packed work, template work,
  presentation pages, fixed clear bytes, and their ratios together.
- Split pressure topology/cache and presentation into internal timestamp scopes
  so authority planning, numerical execution, verification, and publication
  are separately attributable without relying on source-level guesses.

### P1: migrate lifecycle consumers to the topology delta

- Convert candidate face receipts, six-sided face transfer, shadow cell writes,
  and topology-root publication to the same delta, one at a time.
- Preserve broad injection entry points until injection authors or extends the
  delta in the same command buffer.
- Add per-kernel timestamp scopes around candidate transfer so the gain from
  each cut is visible below the current whole-stage aggregate.
- Require list/predicate equality receipts and existing transaction receipts at
  every cut. Do not add another activity or topology authority.

The first cell-transfer cut is intentionally small. Candidate phase time changes
from 2.0972 ms to 2.0316 ms in the matched min-8 capture because the remaining
candidate kernels are still domain-shaped. The value of this cut is that it
proves the representation and lifecycle semantics before widening the migration.

The one-step B8 production presentation remains byte-exact against its immutable
publisher oracle with the compact path. The existing five-step paired oracle
diverges at presentation byte zero with both the original 512-brick candidate
dispatch and the compact dispatch; the paired runs already have different
velocity hashes after one step. It is therefore tracked as a surrounding B8
oracle issue rather than attributed to this list conversion.

### P1: migrate accepted-structure consumers

- Start with brick-local non-pressure kernels which can use `leaf ID -> accepted
  rung -> template cell range` without row ownership or stale-state concerns.
- Keep cell/row lists as derived caches until at least two consumers can be
  converted without changing arithmetic or ownership.
- For stale-state writers, dispatch accepted leaves union the relevant lifecycle
  delta. For shared-row writers, continue to use the explicit accepted row list.
- Publish separate indirect triplets for linear-invocation and one-workgroup-per-
  leaf kernels; a single count does not imply a single dispatch geometry.

### P2: compact the physical aggregate graph

The leaf manifest eliminates recurring scans over absent work, but a B8 brick at
the 1-cell rung is still a physical aggregate node. Build compact aggregate
adjacency and hierarchy indices from the same accepted generation so pressure
aggregate work, activity, and planning follow active leaves. Only after this is
stable should coarse regions merge physical B8 ownership spans.

### P2: decouple presentation resolution from physics brick allocation

Publish variable-resolution pages (or macro pages) for coarse bricks and build
fine pages only for visible interface/receiver closure. The renderer can sample
the coarse CM12 authority directly or reconstruct on demand. One P8 page per
logical B8 domain brick makes a large speedup impossible even when physics has
only one accepted cell per brick.

### P2: incremental topology list publication

Maintain accepted/shadow lists as persistent generations and patch changed
brick ranges plus seam closure. Do not rescan 299,520 cells and 938,688 rows for
20 changed bricks. The topology delta now supplies a stable changed-brick
prefix. The remaining problem is deterministic row ownership/deduplication; the
measured incidence-atomic prototype is not sufficient.

### P2: generation-stamped sparse scratch

Replace capacity-sized scratch clears only after accumulators carry generation
stamps or compact owner lists. A scatter clear over accepted cells is not enough
for dense performance and can miss receiver closure. The representation must
make stale entries unobservable without trading bandwidth for atomics.

### P3: coalesce authority transactions

Once work domains are compact, fuse compatible begin/finalize/verify operations
and author several indirect families in one kernel. Do this after data shape is
fixed; transaction fusion alone cannot recover the missing two orders of
magnitude.

## Acceptance target

For mini64/B8/P8/full-domain-min-8, each compact-list migration should satisfy:

- its invoked leaf IDs exactly match the shader's former branch predicate;
- leaf IDs remain stable and spatially ordered;
- dense mini64 does not regress by more than 2%; and
- all FCA/SRR/FPA/PTR/PCF/FPP receipts and refinement-region tests remain clean.

The later physical-graph cut should satisfy:

- packed physical bricks and presentation pages fall with the coarse region;
- no recurring full template-cell or template-row scan on a 20-brick topology
  transaction;
- pressure aggregate/hierarchy work is proportional to live aggregate nodes;
- pressure topology and presentation each retain less than 25% of ordinary
  cost when pressure cells/rows retain 0.4%.

## 2026-09-02 ocean-seiche B8/P8 full-domain min-8 follow-up

The large ocean case makes the remaining execution-shape defect more explicit
than mini64. Its accepted topology contains 4,800 cells and 15,400 rows across
2,560 accepted leaves: 2,400 B1 leaves, 140 B2 leaves, and 20 B4 leaves. The
transport authority emits 2,560 live 4-cubed packets. Those packets launch
163,840 lanes for 4,800 accepted cells, so only 2.93% of conservative-transport
lanes carry accepted work. The packet list has removed absent packets, but its
lane layout has not adapted to coarse rungs.

Velocity extension has a separate defect. B8 resident capacity exposes 24,576
direct packet ordinals and dispatches that domain for initialization plus eight
Jacobi sweeps. A construction-isolated QA arm now compiles accepted packets by
traversing the accepted-leaf manifest and expanding only the packet count
implied by each rung. It preserves the stable TEI packet address and runs the
same numerical kernels through an indirect dispatch.

Matched 16-frame ocean samples measured:

| arm | VEX median | VEX p95 | frame median | non-pressure median |
|---|---:|---:|---:|---:|
| direct capacity | 2.2938 ms | 4.0632 ms | 37.3555 ms | 31.9816 ms |
| accepted packets | 1.5073 ms | 1.8350 ms | 36.8968 ms | 31.2607 ms |

The VEX arm is a useful 34% stage win but only a 0.46 ms frame win. On a
fine-biased mini64 sample it was neutral within run variance (3.1457 versus
3.2768 ms VEX median). The production path therefore remains unchanged. The
right production form is a topology-generation-cached accepted packet image,
with a density/occupancy policy that keeps direct brick-major dispatch for
dense fine rungs.

The larger opportunity is conservative transport: pack B1/B2/B4 accepted cells
from multiple leaves into full warps while retaining packet-local TEI staging
for B8. That execution image must be deterministic and spatially ordered; a
per-frame atomic append would trade lane waste for unstable locality. Ocean is
only the stress instrument for this work—the selection rule must depend on rung
and accepted work, never scene identity or refinement-region identity.

### Hybrid coarse-cell transport experiment

A second construction-isolated QA arm implements that split directly. Accepted
B1/B2/B4 cells are replayed in stable accepted-cell order, 64 cells per
workgroup. B8 packets remain on the ordinary staged packet kernel. Both arms
consume the same TPA dirty masks; the compiler retains coarse masks for cell
selection but omits coarse packets from the fine-packet indirect list.

On ocean min8 every accepted leaf is B1/B2/B4, so conservative transport's
packet indirect count falls from 2,560 to zero and the coarse stream executes
75 workgroups. The first timing arm incorrectly reused that same empty indirect
list for sharpening, however, so its 27.4596 ms result omitted physical work
and is not a valid optimization result.

Raw Phase-1 instrumentation localized the error. The two arms were bit-exact
through four complete steps; the first field difference appeared only after
surface sharpening on step five. Conservative departure, stencil cells,
stencil weights, beta, deficit, and gathered mass were still identical. The
fix is a separate sharpening packet schedule rebuilt from the masks published
by packed coarse transport. This preserves packet-local directory staging for
sharpening while allowing conservative transport to use cross-leaf cells.

With sharpening restored, ten consecutive ocean steps are raw-bit identical
for every diagnostic field and Phase-1 receipt. The 24-advance terminal work is
also identical: 2,670 pressure cells, 8,797 rows, and relative residual
2.527815047e-4 in both arms.

| arm | conservative transport | frame median | non-pressure median |
|---|---:|---:|---:|
| staged packets | 10.4202 ms | 37.3555 ms | 31.9816 ms |
| packed coarse + restored sharpening | 2.8836 ms | 28.5082 ms | 23.4619 ms |

This established the opportunity, but not yet a safe default. The extra packed
dispatch can make conservative transport itself slower when most accepted work
is already B8, so the production form now makes the decision on the GPU from
the current frame's actual work. It counts dirty B1/B2/B4 packets and selects
accepted-cell packing only when their 64-lane packet domain is at least four
times the complete accepted-cell scan, with a four-packet minimum. Otherwise
it retains the complete packet schedule and emits zero packed workgroups. This
predicate uses only accepted topology and dirty transport masks; it has no
scene, refinement-region, or rung-name special case.

The selector authors separate indirect schedules for conservative transport
and sharpening. When packing is selected, B1/B2/B4 packets leave only the
transport list and are restored to sharpening's packet-local list from the
masks emitted by packed transport. B8 always stays packet-local. When packing
is rejected, both consumers use the original complete packet stream.

A second experiment packed sharpening sources too. It regressed the ocean
sharpening stage from 3.5389 to 4.7186 ms and introduced small field drift: its
iterative gradient trace repeatedly needs neighboring owner lookups, and the
cache-free cross-leaf form loses more locality than it gains in occupancy. Keep
sharpening packet-local until a locality-preserving source clustering scheme
exists.

With compact candidate planning held constant, matched 24-frame production
samples measured:

| arm | dirty coarse packets | packed workgroups | conservative transport | frame median | non-pressure median |
|---|---:|---:|---:|---:|---:|
| complete packet control | 0 (capability off) | 0 | 8.9784 ms | 36.3725 ms | 30.5398 ms |
| dynamic production selector | 2,560 | 75 | 3.8666 ms | 30.8019 ms | 25.1003 ms |

The production selector cuts conservative transport by 56.9%, the full frame
by 15.3%, and non-pressure work by 17.8% in the paired ocean measurement. Both
arms finish with exactly 4,800 cells, 15,400 rows, 2,400 pressure cells, 7,900
pressure rows, and residual 1.3529102959654814e-4. Ten consecutive steps are
raw-bit identical for every diagnostic field, transport receipt, and work
count.

On the fine-biased mini64 case, the selector sees 184 dirty coarse packets,
rejects packing, emits zero packed workgroups, and preserves all 1,016 ordinary
transport/sharpening packets. Its transport median is effectively neutral
(2.2282 ms production versus 2.1627 ms paired packet control), while the frame
median improved within run variance (35.9137 versus 37.4211 ms). This dynamic
hybrid is now the ordinary construction path.

The focused WGSL and topology tests pass, as do Dawn's symmetric expansion,
hydrostatic adaptivity, mini32 correctness, both min8 surface gates, mini32 and
mini64 performance, live liquid insertion, and outside-tank symmetric collapse.
No regression lane or timing ceiling was weakened.

### Candidate-topology / resolution-planning experiment

The UI's **Candidate topology build** label is the resident
`resolution-planning` stage. On ocean min8 its two dominant chunks were
refinement-policy classification (0.6554 ms) and resolution grading/validation
(0.8520 ms). The latter ran three policy-tile closures and three face-grading
closures even though every leaf in the fully covering min8 region had a hard
B1 cap. At that cap, `planBrickResolution` has already clamped every candidate
to B1; both closures are mathematical no-ops.

The initial hypothesis—that thousands of non-leader workgroups were the main
cost—was incomplete. A compact GPU leader list reduced 3,072 candidate launches
to 180 policy-tile leaders, but the leaders still cooperatively visited almost
the entire 2,560-leaf volume and planning remained about 2.03 ms. The useful
general changes were instead:

- compile policy-tile leaders on the GPU and dispatch only that stable-ID list;
- reuse the immediately preceding per-leaf six-face enclosure receipts when
  reducing tile enclosure, rather than repeating directory and SolidWorld
  queries for every sibling face; and
- elide policy-tile and face-grading closure for a leaf whose authored hard cap
  is B1, because no neighbor request can raise it above that cap.

These predicates depend only on authored policy metadata and the coarse rung,
not on the ocean scene or a region identifier. Ten consecutive ocean steps are
bit-identical to the control for every diagnostic field, the complete activity
and topology snapshot, all work counts, and the pressure residual.

Matched 24-frame samples measured:

| arm | policy classification | grading/validation | planning stage | frame median |
|---|---:|---:|---:|---:|
| full leaf/tile closure | 0.6554 ms | 0.8520 ms | 1.9661 ms | 36.4380 ms |
| receipt reuse + B1 closure elision | 0.3277 ms | 0.1311 ms | 1.0486 ms | 35.7171 ms |

The stage falls 46.7%, saving 0.9175 ms; the paired frame median falls 0.7209 ms
despite larger whole-frame variance. Candidate-default Dawn lanes pass
hydrostatic adaptivity, symmetric expansion, mini32 correctness, both min8
surface tests, and the mini32/mini64 performance ceilings. The compact planning
path is now the ordinary construction path. The canonical aggregate Dawn run
again reached the pre-existing long-dam generation-zero assertion (276 pages
versus the test's expected 80), then exhausted its 180-second budget in the
tall-cells lane. Individually, live liquid insertion and outside-tank symmetric
collapse also pass; no lane or timing ceiling was weakened.

Holistically, very little *resolution choice* remains for this stage to improve
in a fully covering min8 region: the authored B1 cap has already decided the
answer. The physically necessary work is tile membership/lifecycle closure,
which prevents partially retiring the B8 siblings that jointly represent one
coarse physical tile, plus transition grading when a hard-cap boundary exists.
The fact that a full-domain B1 policy still owns 2,560 resident B8 leaves for
only 180 elected physical policy tiles is the deeper representation problem.
Making those coarse tiles first-class resident topology—not groups of fine leaf
pages—would make candidate topology, lifecycle, presentation, and other
leaf-domain stages scale with coarse-cell proportion instead of authored B8
page count.

### Presentation, sharpening, and face-preparation follow-up

After coarse transport and planning, ocean's largest non-pressure stage was
presentation publication at 6.0948 ms. The fixed P8 representation publishes
2,560 pages, 20,480 4-cubed tiles, and 1,310,720 exact samples every frame from
only 4,800 accepted physics cells: 273.1 presentation samples per physical
cell. The existing P8 executor also used only eight lanes per workgroup, with
each live lane evaluating 64 samples serially.

The retained general change flattens each page's sample domain over all 64
lanes in both execute and commit. It changes no page format, reconstruction,
fault rule, or scene predicate. A source-toggle A/B produced identical hashes
for the complete 6,291,456-byte accepted/candidate sample buffer and 49,152-byte
metadata buffer. The complete FPP header, allocator, page/tile/sample receipts,
accepted work, and topology also match exactly, with no faults or omissions.

| median | serial page lanes | flattened page samples | change |
|---|---:|---:|---:|
| presentation publication | 6.0948 ms | 5.0463 ms | -17.2% |
| current full frame | — | 29.8844 ms | — |
| current non-pressure total | — | 24.6415 ms | — |

This fixes the immediate execution-shape defect but not the 273x representation
amplification. Scaling presentation with coarse-cell proportion ultimately
requires variable-resolution or procedural macro pages rather than materialized
P8 samples for every resident B8 leaf.

Two additional general experiments were implemented and measured but rejected:

- Splitting physical and sparse-air face addresses halved seam workgroups from
  340 to 170, yet face preparation regressed 3.7356 to 3.9977 ms. The hot path
  is trace/latency constrained, not rejected-lane constrained. See
  [the face-preparation experiment](sparse-cm12-face-preparation-seam-family-experiment-2026-09-02.md).
- A destination-bit fixed-point gate proved that ocean's density-capacity
  repair is physically inert after the first round, but branch-only suffix
  dispatches regressed capacity repair from 1.3107 to 1.3763 ms. The rejected
  specialization remains production-unreachable for QA. See
  [the sharpening experiment](sparse-cm12-ocean-min8-sharpening-capacity-early-exit-experiment.md).

The combined focused suite passes 42 tests and all 226 compiled WGSL entry
points. Dawn passes symmetric expansion, hydrostatic adaptivity, mini32
correctness, both min8 surface lanes, live liquid insertion, outside-tank
collapse, and the mini32/mini64 performance ceilings. The canonical aggregate
run again reproduces the pre-existing long-dam generation-zero assertion (276
pages versus 80) and then exhausts its 180-second budget in tall-cells; no lane
or ceiling was weakened.
