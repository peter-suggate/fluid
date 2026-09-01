# Sparse CM12 performance analysis — 2026-09-01

## Scope and performance objective

This is a source-level audit of the complete resident Sparse CM12 advance, from
frame-control and velocity extension through presentation publication. It covers
every stage in `SPARSE_CM12_RESIDENT_STAGES`, the persistent/transient data those
stages touch, indirect and indexed traversal, locality, packing, stale paths,
shader structure, and repeated work. It is paired with the repository's existing
hardware timestamp evidence; a fresh post-cleanup Dawn result is recorded below
when the shared WebGPU lease is available.

The optimization objective is deliberately not “make the finest case faster.”
It is:

> Make frame cost follow accepted cells, active rows, changed topology and dirty
> presentation pages, so an aggressively coarse adaptive world receives the
> performance benefit that its smaller physical problem deserves.

The surface/thin-feature safeguards, accepted-output B8-to-B4 proof, one-rung
demotion, urgent refinement lane, 2:1 closure, conservative transfer, and pressure
tolerance remain correctness constraints. A timing win that weakens one of these
is not a win.

Pressure-solver optimization is explicitly out of scope. Its timing is retained
below only so the non-solver percentages can be interpreted. Dispatch count and
dispatch overhead are also not optimization targets. A dispatch is interesting in
this audit only when it exposes duplicated memory traversal, a capacity-wide scan,
atomic serialization, or another GPU-unfriendly data path.

## Evidence and limitations

The checked-in mini32 B8/P8 control trace
`artifacts/sparse-cm12-mini32-b8-p8-20260827-control3-stage-cost.json` used GPU
hardware timestamps with a 65.536 us quantum. Its all-frame medians were:

| Stage | Median (ms) | p95 (ms) | Comment |
|---|---:|---:|---|
| pressure-solve | 10.1581 | 14.1558 | Dominant fixed-budget iterative traffic |
| surface-sharpening | 2.7525 | 4.0632 | Includes 24 accepted-cell capacity-repair dispatches |
| pressure-topology | 2.4904 | 39.3216 | Large tail/outliers; generation repair and cache publication |
| transport-velocity-extension | 1.9005 | 2.8180 | Eight fixed extension sweeps plus packet compilation |
| conservative-transport | 1.5729 | 1.8350 | Cached departure trace/scatter/gather |
| activity-measurement | 1.3763 | 1.7695 | Several capacity scans despite sparse acceptance |
| resolution-planning | 1.1141 | 1.3107 | Policy, grading, allocation and shadow worklists |
| pressure-rhs | 0.7209 | 1.3107 | Seed, reductions and solve-gate publication |
| face-preparation | 0.6554 | 0.7864 | Dense fine-face support plus sparse packets |
| gamma-diffusion | 0.6554 | 1.7039 | Row scatter and cell finalize |
| velocity-projection | 0.4588 | 0.9175 | Excludes one isolated 9.175 ms outlier in p95 |
| candidate-transfer | 0.4588 | 1.5729 | Cheap when delta is empty, much dearer when changed |
| presentation-publication | 0.3932 | 0.4588 | Older trace predates the latest receipt detail |
| symmetry-authority | 0.1311 | 0.3277 | Frame-control-gated D4 work |
| body-forces | 0.1311 | 0.2621 | One accepted-row traversal |
| brick-retirement | 0.0655 | 0.1311 | Capacity scan; D4 can make this much larger |
| tracer-advection | 0 | 0 | Disabled in the control capture |

These numbers rank opportunities, but are not a current before/after comparison.
The trace is older than several topology, presentation and refinement-policy
changes. In particular, the old empty `projection-diagnostics` timing row did
not own the diagnostic work: those dispatches were in `velocity-projection`.

### External xctrace campaign

The cleanup was also measured with Metal System Trace rather than only the
instrumented stage probe. The capture keeps the shipping graph unisolated for
the wall-clock control, then uses encoder isolation and GPU counters only for
stage attribution. Pressure is shown for frame closure but was not optimized.

| Metric | Original baseline | Frontier-cache pass | Final pass | Original → final |
|---|---:|---:|---:|---:|
| clean shipping wall / advance | 12.97 ms | 11.74 ms | 11.33 ms | -12.6% |
| isolated frame wall | 15.35 ms | 12.95 ms | 13.04 ms | -15.0% |
| GPU busy / advance | 12.65 ms | 10.23 ms | 9.92 ms | -21.6% |
| activity measurement | 2.430 ms | 0.844 ms | 0.847 ms | -65.1% |
| resolution planning | not isolated in the original report | 1.268 ms | 0.947 ms | -25.3% after frontier pass |
| surface sharpening | 1.395 ms | 0.820 ms | 0.814 ms | -41.6% (not caused solely by this patch) |
| compute occupancy | about 0.8% | about 0.9% | about 0.8% | still latency/irregularity limited |

Receipts:

- `artifacts/xctrace-sparse-cm12-nonpressure-baseline-2026-09-01-r9/report.html`
- `artifacts/xctrace-sparse-cm12-frontier-cache-post-2026-09-01/report.html`
- `artifacts/xctrace-sparse-cm12-nongpu-cleanup-final-2026-09-01/report.html`

The final capture attributes 100% of retained Metal interval time to exact
stage labels. Mean compute occupancy is 0.8% and ALU utilization 2.6%. That
falsifies a saturation model: the non-pressure problem is dominated by tiny,
branchy kernels and indirect topology/data access. Dispatch overhead is not a
target and is not used to explain these results.

## Data layout and access audit

| Data family | Current shape | Access behavior | Finding |
|---|---|---|---|
| Physical cell record | 8 words / 32 B per template cell | Stable-ID indexed; accepted worklists turn this into indirect random reads | Compact enough for identity/topology, but unrelated consumers often load the whole record through helpers. Split hot identity/level/brick facts from cold diagnostics only after load profiling. |
| Cell state | Roughly 24 f32 planes per template cell before optional fields | SoA gives coalesced single-field sweeps; multi-field kernels issue many independent streams | Appropriate for PCG and scalar passes. Field lifetime aliasing is incomplete, so total residency and cache pressure exceed the live set. |
| Physical rows | 9 SoA word planes / 36 B per row | Accepted row list -> row ordinal -> endpoints/geometry; endpoints then index cell fields | Sequential row-plane loads are good; endpoint field reads are indirect. Row-kind-specific payload is only partly packed, so regular rows pay for mixed/seam capability. |
| Cell-row incidence | 4 B offset per cell + 8 B per incidence | Per-cell loops gather row ids and signs, then row endpoints and face/state fields | The central irregular access. Accepted cell order inherits topology/atomic construction order rather than a cache-aware brick/level order. Reordering immutable incidence by accepted brick and local cell is a high-value experiment. |
| Dense fine-face support | vec4 / 16 B per finest-grid cell | Direct fine-lattice address; face packet kernels gather it | Large, but predictable and cache-friendly. Do not replace it with a more compact hash/indirection without proving a bandwidth win; it is intentionally a locality cache. |
| Transport departure cache | 6 u32 packed id words + 8 f32 weights = 56 B per accepted/template cell slot | Trace writes once; scatter and gather replay the same stencil | Expensive but removes repeated geometry/search. IDs are efficiently 24-bit packed; f32 weights preserve conservative reproduction. Focus on making allocation/worklist proportional to transport packets before quantizing weights. |
| Pressure templates/cache | Immutable coefficient/endpoint images plus mutable topology/cache headers | PCG repeatedly streams a compact pressure cell list and row/coefficient image | Duplicating immutable hot pressure data away from atomic topology storage is sound. Stale aggregate/hierarchy storage and receipt fields remain after production moved to Jacobi. |
| Conditioning arena | Sized for max(7 * cells, cells + rows + 8), atomic words | Reused across transport/sharpening/pressure membership epochs | Good phase aliasing, but atomic typing forces integer/atomic access patterns and full clears in some branches. Continue phase-arena cutover so disjoint lifetimes do not reserve simultaneous buffers. |
| Candidate arena | Max of pressure scratch, departure cache, candidate pages; persistent fine-edge image also resides nearby | Different phases use different interpretations | Alias is valuable. Memory diagnostics must distinguish reserved peak from live bytes and persistent subregions. |
| Worklists and indirect arguments | Stable-id lists plus copied 12-byte dispatch triplets | Efficient zero-work gating after the copy; construction commonly uses atomic append | Dispatch is sparse, but appended order can destroy spatial locality. Preserve zero-count indirect dispatches while producing lists in brick-major order where possible. |
| Presentation pages/directory | Sparse page slab plus directory and frame-plan receipts | Planner scans leaf capacity; execution is compact/indirect; verification/proof scan capacity | Execution is sparse but planning and proof are not. This prevents coarse/quiet worlds from realizing the full benefit. |

### Cache conclusions

1. The good locality paths are the SoA state planes, immutable pressure image,
   dense fine-face support cache, and replay of a departure stencil across
   transport phases.
2. The bad locality paths are all variants of `accepted id -> physical record ->
   incidence -> row -> other endpoint -> state plane`, especially when the
   accepted list was atomically appended in topology order rather than emitted
   brick-major.
3. A coarse cell reduces accepted-cell arithmetic but may have more mixed-seam
   incidence and a larger spatial stride between neighbor state addresses.
   Therefore the coarse policy needs compiled same-level fast paths and compact
   seam exception lists, not merely fewer cells.
4. Capacity scans are predictable and cacheable but scale with reserved world
   size, not physical work. They are the main architectural enemy of aggressive
   coarsening.

## Repeated approaches across the frame

| Repetition | Where | Cost/risk | Cleanup direction |
|---|---|---|---|
| begin / append-or-classify / finalize transaction protocol | Frame control, scalar masks, candidate effects, IBO, frame plan | Header atomics and repeated reads/writes of the same generation/count facts | Keep transactions where correctness needs generation sealing; carry already-proved facts in one cache-compact receipt rather than re-deriving them. |
| Capacity scan followed by early return | Activity, resolution, brick retirement, presentation allocation/plan/proof | Coarse and quiet scenes still pay for reserved leaves | Publish compact dirty/accepted brick packets once and share them across consumers. |
| Atomic append then indirect replay | Accepted/shadow/topology-delta/transport packets | Good work suppression, weak ordering/locality | Count per brick/tile, prefix, then emit in spatial order for large lists; retain atomic path for tiny deltas. |
| Per-cell incidence traversal | Gamma scatter/finalize, pressure operator/RHS, projection diagnostics, forces | Repeated indirect row and endpoint gathers | Compile row-kind-specialized images and fuse only consumers with matching lifetime and bind group. Do not build a second topology authority. |
| D4 preserve then commit | Scalars, velocity, activity, post-topology repair | Two complete passes per authority event | Frame-control gating is good. For post-commit, compile a changed-orbit packet so the second authority application does not scan template capacity. |
| Eight density capacity relay passes, each three kernels | Sharpening | 24 full accepted-cell traversals | Produce an overflow frontier and relay only affected cells, preserving the eight-cell support bound and exact debit/credit conservation. |
| Similar face families: interior, seam, sparse-air, dynamic | Prepare and project | Four pipelines/dispatch families repeat common code and fields | Retain specialization for locality; share helpers and consider a compact exception stream rather than merging divergent row kinds into one shader. |
| Proof/validation at multiple candidate seams | Candidate transfer and presentation | Necessary transactional safety, but some predicates are re-derived | Carry generation-stamped receipts forward when the exact producer already proved the predicate; never substitute a stale generation. |

## Stage-by-stage audit

### 1. `transport-velocity-extension`

- **Data and access:** Three singleton frame-control kernels write headers, then a
  copied indirect table gates solid/D4 families. Velocity extension traverses a
  stable leaf-by-local-cell transport packet grid for eight depth sweeps. Each
  sweep reads the previous extension bank, face/solid validity and neighbor
  addresses, then writes the next bank. Final masks compile a compact transport
  packet list and copy its indirect triplet.
- **Indirect/cache:** The direct B-profile packet address is predictable and avoids
  a catalog lookup. Neighbor reads are short-range within a B8 page until a seam,
  so brick-major packet order is favorable. Eight global sweeps repeatedly stream
  the same masks/banks eight times, even after most cells have converged.
- **Stale/packing/shader:** `velocityExtensionDepths` is a separate allocation and
  was missing from byte accounting. Depth-specialized pipelines have already been
  collapsed into one pipeline plus depth bind groups, which is the right direction.
  Root-cause/stamp/depth banks should be checked for production readers before
  retaining them all.
- **Win:** Compile a frontier packet per extension depth (or a bitset with a compact
  nonempty tile list) so later sweeps do not revisit converged cells. Preserve the
  maximum eight-hop reach. First measure whether packet compilation costs less than
  full sweep bandwidth at B4/B2/B1-heavy acceptance.

### 2. `face-preparation`

- **Data and access:** Retired/published support scans activity bricks and updates
  the dense fine-face support cache. Interior tiles are contiguous; seam and
  sparse-air packets gather topology/face data; dynamic rows use the accepted-row
  indirect list.
- **Indirect/cache:** The dense cache is 16 B per finest cell but converts later
  reconstruction into direct addresses. Interior tile locality is excellent.
  Seam/dynamic paths pay stable-id and endpoint indirection.
- **Stale/packing/shader:** Four face families repeat aperture, parity and velocity
  decoding. Combining them blindly would introduce branch divergence; common helper
  loads can still be audited for redundant physical-record reads.
- **Win:** Drive support publication from the final scalar/topology dirty brick
  packet, with a fallback full scan only on generation reset. Build a compact seam
  exception list for coarse same-level interiors. Keep the dense support cache until
  an end-to-end alternative beats it.

### 3. `conservative-transport`

- **Data and access:** Clears receipts over accepted cells, traces gamma/beta for
  compact transport packets, writes the 56 B departure stencil, scatters density
  deficit atomically/indirectly to receivers, then gathers conservative density.
  Optional QA capture adds another accepted-cell pass.
- **Indirect/cache:** Trace performs the expensive spatial gather; replay gives
  scatter/gather stable stencil addresses. Receiver writes are irregular and may
  contend. Packet order is only as local as the compiler emits it.
- **Stale/packing/shader:** The 24-bit id packing is space-efficient but creates a
  hard addressability limit and unpack ALU. Eight f32 weights dominate the stencil;
  reducing precision risks conservation and repeatability. QA is correctly absent
  unless armed.
- **Win:** Sort/emit transport packets brick-major and bin scatter receivers by
  destination brick to improve cache and reduce atomic contention. Test f16 or
  shared-exponent weights only as an opt-in experiment with bitwise mass and long-run
  drift gates; the default should remain f32 until proved safe.

### 4. `tracer-advection`

- **Data and access:** Optional dense marker array; each marker gathers velocity from
  sparse/dense face support and writes its new position/color. Seeding is one extra
  pass on enable/reseed.
- **Indirect/cache:** Marker order is not guaranteed to match spatial order, so
  velocity gathers become increasingly random as mixing proceeds.
- **Stale/packing/shader:** Correctly encodes zero GPU work when hidden. This is view
  instrumentation, not simulation authority.
- **Win:** No production priority. If visible-marker cost matters, bin markers by
  brick every several frames or render a sampled subset; never charge hidden frames.

### 5. `gamma-diffusion`

- **Data and access:** Clears per-cell receipts, traverses accepted rows to scatter a
  snapshot, then accepted cells finalize one Jacobi diffusion step. Row endpoints
  index gamma/state planes indirectly.
- **Indirect/cache:** Row SoA is sequential but endpoint state reads and receipt
  writes are scattered. Cell finalization is coalesced if accepted ids are local.
- **Stale/packing/shader:** The former redundant second all-axis pass is already
  removed. Receipt fields share large conditioning storage and may use atomics wider
  in lifetime than required.
- **Win:** Emit accepted rows brick/level-major and split regular same-level rows from
  seam exceptions. Consider fusing receipt clear with the first writer using an
  epoch stamp only if stamp traffic beats the clear on measured coarse scenes.

### 6. `surface-sharpening`

- **Data and access:** Clears conditioning/delta banks, prepares/scatters sharpening
  through transport packets, finalizes cells, then performs eight rounds of
  initialize/scatter/finalize capacity repair. Final scalar masks scan leaf capacity
  between singleton begin/seal kernels.
- **Indirect/cache:** The transform benefits from the compiled transport packet.
  Capacity repair is the worst repeated non-pressure pattern: 24 accepted-cell
  dispatches repeatedly stream density, capacity and debit/credit fields even when
  overflow is sparse.
- **Stale/packing/shader:** Native clears are bandwidth-efficient but force pass
  boundaries. The eight-pass bound is a correctness response to floor-impact mass
  concentration and must not simply be reduced.
- **Win:** After sharpening finalize, compact only over-capacity cells into an
  overflow frontier. Each relay should publish the next frontier and dispatch
  indirectly; empty later frontiers become zero work. Preserve eight relays, exact
  paired debit/credit and mass receipts. This is the clearest non-pressure win.

### 7. `symmetry-authority`

- **Data and access:** Frame-control indirect families gate scalar D4 preserve and
  commit; a singleton publishes scalar output.
- **Indirect/cache:** Symmetric scenes traverse D4 partner cells, which can be far
  apart in memory; asymmetric scenes issue only zero-count/no-op authority work.
- **Stale/packing/shader:** The no-op bypass keeps a sealed command schedule, but it
  is not itself an optimization target.
- **Win:** Keep correctness and gating. Improve partner locality by emitting D4
  orbit members together, especially for the post-topology changed set. Do not
  weaken D4 authority.

### 8. `body-forces`

- **Data and access:** One accepted-row traversal reads row kind/endpoints, body force,
  aperture/mass facts, and writes predicted face velocity.
- **Indirect/cache:** Sequential row records with indirect endpoint data. Mixed seams
  and rigid coupling add divergent branches.
- **Stale/packing/shader:** A row-kind switch in the hot path likely loads fields that
  regular rows do not need.
- **Win:** Compile regular same-level force rows separately from seam/solid exceptions.
  Keep a single canonical row authority; specialization should be execution images,
  not duplicated topology.

### 9. `pressure-topology` — audit only, no optimization focus

- **Data and access:** Singleton transaction setup, accepted/dirty cell classification,
  canonical cell repair, row compilation, frozen cell/membership/coefficient
  publication, pressure execution image finalization and indirect copies. It reuses
  conditioning storage for membership worklists. Fine coefficients are the complete
  production cache.
- **Indirect/cache:** Dirty repair is sparse, but row compilation can scan a fixed
  membership word range. Coefficient publication is compact by pressure cell; row
  construction follows incidence/endpoints and is irregular. Multiple pass closures
  and 12–48 B copies serialize transaction seams.
- **Stale/packing/shader:** Production preconditioning is cell-local Jacobi. Coarse
  aggregate/hierarchy entry points are retired/no-op, yet source helpers, layouts,
  headers, QA receipts and cache storage still describe them. A 144 B
  `persistentPressureCacheIndirectArguments` buffer is allocated/read by QA but is
  not authored or consumed by production dispatch. Stage prose still called the
  solver/cache aggregate + hierarchy in places. This is the largest stale subsystem.
- **Disposition:** The stale aggregate/hierarchy surface is recorded as cleanup debt,
  but pressure work is not part of the optimization plan requested here.

### 10. `pressure-rhs` — audit only, no optimization focus

- **Data and access:** Builds compatible divergence/RHS, initializes PCG and Jacobi
  direction over pressure cells, performs several singleton reductions, initializes
  the pipelined image, publishes/copies a device solve gate, and optionally journals.
- **Indirect/cache:** Pressure-cell lists are compact. RHS formation gathers incident
  rows/faces; vector initialization streams multiple SoA planes. Single-workgroup
  reductions repeatedly revisit partial/header storage.
- **Stale/packing/shader:** UI prose still referenced an aggregate + hierarchy
  preconditioner although production uses Jacobi. Solver vectors have overlapping
  lifetimes that should be mapped before further arena allocation.
- **Disposition:** No optimization work planned.

### 11. `pressure-solve` — excluded from optimization

- **Data and access:** Each encoded iteration updates pipelined vectors, applies the
  pressure image and reduces. Every eight iterations it measures a fresh residual,
  may restart/recover, republishes a zero-count gate and copies indirect arguments.
  A final true residual always closes the solve.
- **Indirect/cache:** This repeatedly streams pressure vectors and coefficient/row
  images; it is bandwidth-dominant. Same-level coarse cells have fewer total cells
  but the generic incidence operator still chases row ids/endpoints.
- **Stale/packing/shader:** “MGPCG” names survive despite a uniform Jacobi
  preconditioner. Generic row-kind helpers in the inner SpMV can cause redundant
  loads and branches. Several f32 vector planes may be live only across one
  iteration phase.
- **Disposition:** Dead-end for this effort. No solver, preconditioner, convergence,
  vector-traffic or command-schedule optimization is proposed.

### 12. `velocity-projection`

- **Data and access:** Projects interior/seam/sparse-air tiles and accepted dynamic
  rows, enforces inflow rows, then collocates velocity per accepted cell by traversing
  incidence. The same incidence loop now accumulates divergence maxima into
  workgroup partials; one singleton reduces them. D4, rigid reaction and output
  publication follow.
- **Indirect/cache:** Face tile paths are local; dynamic rows and collocation are
  indirect. Before cleanup, a second accepted-cell kernel repeated the entire
  incidence traversal solely for diagnostics.
- **Stale/packing/shader:** The empty `projection-diagnostics` stage and separate
  `measureDivergenceDiagnostics` entry point were stale/repeated architecture.
- **Cleanup completed:** Divergence measurement is fused into
  `collocateAndDiagnose`, eliminating one accepted-cell dispatch and one full
  indirect incidence traversal. The empty stage was removed; the reduction remains.
- **Next win:** Apply the pressure operator's same-level/seam execution split to
  projection, sharing the compiled row classification rather than re-deriving it.

### 13. `activity-measurement`

- **Data and access:** Scalar dirty marking dispatches one workgroup per leaf capacity;
  topology/history scans use capacity/64; brick measurement scans the activity brick
  slab. D4 folds activity when enabled. Optional sparse-world growth scans 26 neighbor
  directions per leaf and synthesizes page capacity.
- **Indirect/cache:** These are mostly linear, cache-friendly scans, but they ignore
  how few leaves are accepted/dirty. The 26-neighbor frontier scan is especially
  expensive at large reserved capacity.
- **Stale/packing/shader:** Scalar, topology, history and measurement kernels load
  overlapping brick headers/masks in separate passes. The former world-growth path
  also repeated signed directory and SolidWorld reachability probes for every
  accepted leaf's 26-neighbour apron on every advance.
- **Cleanup completed:** Frontier allocation now dispatches over the accepted-leaf
  manifest and stores a 26-bit resolved-neighbour cache in activity word 42. Existing,
  unreachable and successfully allocated neighbours become persistent evidence;
  allocator exhaustion remains retryable. Retirement re-arms reciprocal neighbour
  bits and a SolidWorld edit clears only this derived cache. The accepted manifest
  publishes its frontier indirect domain atomically with topology acceptance.
- **Measured result:** On the mini16 paper-cadence probe, frontier allocation fell
  from 0.983 to 0.590 ms median (-40%), with late quiescent samples at
  0.262–0.328 ms. External xctrace reduced the complete activity stage from
  2.430 to 0.844 ms (-65%).
- **Next win:** Share the existing dirty/accepted evidence with measurement and
  history aging where it removes substantive memory traversal. Do not introduce
  compact-list lookup for small one-record-per-leaf kernels unless the capacity
  ratio is large enough to win a hardware A/B.

### 14. `resolution-planning`

- **Data and access:** Two leaf-capacity classifiers feed brick-level planning,
  activation and retirement scan capacity, and each ladder rung runs tile-policy
  closure plus 2:1 brick closure. It validates, schedules, allocates pages,
  synthesizes authored leaves, builds shadow row/leaf/structure worklists, then makes
  four indirect copies.
- **Indirect/cache:** Classifiers are linear but capacity-bound. Brick closure reads
  neighboring planned levels and writes atomically/iteratively; coarse policies can
  therefore reduce later cells yet still pay almost the same planning scan. Shadow
  appends can produce non-spatial order.
- **Stale/packing/shader:** Timing metadata had omitted two new refinement-policy
  entry points; that manifest is now repaired. Diagnostic sub-seams proved the
  branch-heavy initial resolution request was the hot part, not repeated grading.
  For ordinary leaves it serially repeated six directory-neighbour lookups to
  establish deep enclosure after the preceding frontier classifier had already
  visited all 26 neighbours cooperatively.
- **Cleanup completed:** The frontier classifier now publishes a deep-enclosure bit
  from its existing six face lanes. Policy-tile classification overwrites that bit
  only for true multi-leaf policy tiles; scale-one planning consumes it directly.
  The old serial `brickDeeplyEnclosed` function was deleted. All surface proof,
  recovery locks, velocity floors, urgent refinement, direct deep-bulk coarsening,
  one-rung surface demotion and 2:1 closure remain unchanged.
- **Measured result:** Mini64's initial planner fell 0.7864→0.3277 ms (-58%). The
  classifier rose 0.0655→0.1966 ms, leaving the full resolution stage
  1.6384→1.3763 ms (-16%). Mini16 fell 1.2452→1.1141 ms (-10.5%). External
  xctrace measured 1.268→0.947 ms (-25%). Work-shape receipts remained identical:
  mini64 ended with 347 resident leaves and 85 topology commits; mini16 retained
  eight leaves and zero commits.
- **Next win:** Seed a “policy changed or neighbour of changed” closure frontier and
  iterate only it, with a full-scan validation oracle in QA. Emit shadow worklists in
  brick-major/local-cell order. Preserve the complete aggressive adaptive policy.

### 15. `candidate-transfer`

- **Data and access:** Twelve timed transaction seams cover delta cell transfer, face
  reconstruction/validation, effects census, IBO build/validation, TEI compilation,
  authorization, PTR publication, state publication and replay. Most large kernels
  use topology-delta or shadow indirect lists; singleton seal/finalize kernels always
  run.
- **Indirect/cache:** Empty deltas correctly produce near-zero large work. Changed
  deltas gather old cells and write shadow pages, then validate/replay the same ids
  several times. Atomic worklist order affects locality. Dynamic SolidWorld branches
  scan topology page capacity.
- **Stale/packing/shader:** Validation is intentionally repeated across independent
  authorities, but some generation/count predicates are re-read rather than carried
  in the transaction receipt. Formatting/indentation around IBO/PTR blocks obscures
  the actual ownership and invites drift.
- **Win:** Share one brick-major delta packet across transfer, effects, IBO and TEI;
  attach compact per-brick validation receipts so later kernels need not rescan
  unchanged members. Keep independent semantic validation and fail-closed generation
  checks. Optimize the changed case without adding cost to empty deltas.

### 16. `brick-retirement`

- **Data and access:** On D4 scenes it re-applies velocity preserve/commit over
  `templateCellCount`, because the new accepted indirect list is not promoted yet.
  It then scans leaf capacity to mark post-topology activity and finalizes masks.
- **Indirect/cache:** Direct template/capacity scans are contiguous but ignore delta
  size. The second D4 application can be much larger than the stage name suggests.
- **Stale/packing/shader:** This stage no longer decides retirement; it publishes
  post-commit authority. The label has been corrected in the UI, but source concepts
  and probes should consistently use the new meaning.
- **Win:** During candidate authorization compile changed D4 orbits and post-topology
  dirty leaves. Dispatch this stage indirectly over those packets, preserving the
  final-accepted D4 guarantee without scanning all template cells.

### 17. `presentation-publication`

- **Data and access:** Allocation scans leaf capacity and singleton-sorts the page
  directory. Frame planning performs eight brick-capacity passes plus begin/finalize
  singletons, copies two indirect bindings, executes compact presentation packets,
  verifies all bricks, then publishes/rejects surface proof over all bricks. Retirement
  scans capacity, compacts the directory and commits frame control.
- **Indirect/cache:** Page execution is properly compact and page-local. Planning,
  verification and proof are linear capacity scans, so quiet/coarse worlds retain a
  fixed floor. The singleton directory sort is a serial algorithm whose cost will
  grow with resident pages.
- **Stale/packing/shader:** Several plan/proof kernels read the same brick generation,
  dirty and surface facts. The current presentation receipt timing additions are
  useful because this stage's old aggregate number hid that repetition.
- **Win:** Make plan construction consume accepted and dirty-page packets; verify only
  executed pages plus explicit retirement/topology deltas; retain prior
  generation-stamped proof for untouched bricks. Replace singleton insertion/sort
  behavior with stable slot ownership or a parallel radix/prefix scheme before page
  counts grow. Surface proof must remain over the accepted topology logically, but
  unchanged bricks may reuse a proof only when topology, scalar generation and page
  generation all match.

## Cleanup round completed

The first patch intentionally removes repeated work without changing topology,
adaptivity decisions or numerical policy:

1. Fused divergence diagnostics into the existing collocation incidence loop.
2. Removed the separate accepted-cell diagnostic dispatch and its WGSL entry point.
3. Removed the empty `projection-diagnostics` resident stage and corrected timing/
   migration contracts to place diagnostic partials in `velocity-projection`.
4. Repaired the resolution-planning timing manifest for the two refinement-policy
   kernels already present in the encoder.
5. Updated the velocity-extension timing check for the current shared pipeline.
6. Corrected `allocatedBytes` accounting for the pressure-cache indirect buffer and
   velocity-extension depth buffer. This exposes memory; it does not allocate more.
7. Split activity and resolution timing into hardware-attributable substages. This
   first disproved the guess that activity D4/history was dominant: frontier
   allocation was the real hot path. It later disproved the guess that repeated 2:1
   grading dominated resolution planning: the initial request shader did.
8. Replaced repeated SparseWorld frontier directory/reachability questions with an
   accepted-leaf frontier domain and a persistent, explicitly invalidated 26-bit
   resolution cache.
9. Reused the cooperative frontier classifier's six-face evidence for deep enclosure
   and deleted the planner's repeated serial directory walk.

### Failed approaches and updated world model

Failure was treated as evidence rather than hidden:

1. **Accepted-leaf compaction alone did not speed frontier allocation.** The first
   implementation reduced the invocation domain, but repeated directory and
   reachability probes still dominated. Direct readback proved the compact indirect
   triplet was valid (`[4,1,1]` on mini16). Adding persistent resolved-neighbour
   evidence was what produced the 40% stage win.
2. **Compact accepted-leaf resolution planning was rejected.** It first added a
   capacity initializer and was neutral/slower on mini16 because both paths occupied
   one workgroup. The initializer was removed and mini64 was tested, where six
   indirect accepted groups still failed to beat eight direct capacity groups:
   0.8520 ms compact versus 0.7864 ms direct for the initial-plan bucket, while the
   complete resolution stage was identical at 1.6384 ms. The compact planner,
   helper and semantic accommodations were removed completely.
3. **The updated rule:** compact domains pay when they eliminate substantial
   irregular traversal or persistent repeated questions. They do not automatically
   help modest capacity ratios, especially when a direct one-record-per-leaf kernel
   gains a list lookup. Re-review therefore targeted duplicated topology evidence,
   yielding the retained cooperative enclosure win.
4. **Exhaustive final QA inside the timing probe was not a useful timing path.** It
   became host-JavaScript bound for minutes after GPU sampling. Timing A/Bs now use
   `--final-qa=0`; correctness is established with focused static tests and the Dawn
   regression lanes instead of conflating a CPU catalogue audit with GPU timing.

The pressure aggregate/hierarchy deletion is deliberately not bundled into this
first patch. Although production kernels are no-op/retired, layout and QA removal is
broad enough to deserve its own regression-tested change.

## Ranked win plan

| Rank | Work | Expected leverage | Adaptivity/correctness guard |
|---|---|---|---|
| P0 | Sparse overflow frontier for eight-pass density capacity repair | Removes up to 24 full-cell traversals when overflow is localized | Eight-hop bound, paired debit/credit and mass conservation unchanged |
| P0 | Spatially ordered accepted/row/delta/transport worklists | Better cache use in every indirect row/cell consumer | Stable ids and authority semantics unchanged |
| P1 | Incremental presentation plan/proof and scalable directory maintenance | Final xctrace's largest non-solver consumer is FPP1 execution at 1.160 ms; capacity verification/proof remains a fixed sparse-world floor | Reuse proof only on exact generation match; surface coarsening proof remains mandatory |
| P1 | Changed-frontier 2:1 closure plus brick-major shadow worklists | Makes planning follow actual adaptive change after the initial-request cleanup | Full-scan QA oracle; complete hard-region caps, urgent refinement and 2:1 validation unchanged |
| P2 | Frontier-driven velocity extension depth sweeps | Benefits settled coarse regions | Maximum extension reach and root-cause semantics unchanged |
| P2 | Phase-arena/lifetime aliasing for transport, sharpening, adaptivity and publication scratch | Lowers residency/cache pressure | No overlapping lifetime; byte accounting and debug capture remain truthful |
| P3 | Experimental transport weight compression | Potentially reduces 56 B/cell stencil bandwidth | Off by default until conservation, symmetry and long-run drift gates pass |

## Measurement matrix for the next round

Each optimization should be evaluated on work shape as well as milliseconds:

- mini32 B8/P8 for rapid stage attribution and timestamp stability;
- mini64 for scaling and pressure/vector bandwidth;
- long-dam left/right far-wall fronts for topology change and presentation deltas;
- floor-only symmetric collapse for both D4 authority passes;
- hydrostatic and ocean/seiche scenes for coarse deep bulk, surface proof and pressure
  iteration behavior;
- live rigid/liquid insertion for frontier and candidate transaction spikes.

Report accepted cells/rows, leaf capacity, dirty leaves, topology-delta cells, compact
presentation pages, pressure iterations executed/encoded, bytes reserved/live, and
stage median/p95. A speedup accompanied by a finer accepted topology is not evidence
that the kernel improved; normalize against the same topology generation/work shape.

## Verification of the cleanup round

Completed on the working tree:

- the resident WGSL compiled under Dawn/Metal in both the hardware probe and the
  external xctrace harness;
- 70 focused stage-partition, timing-manifest, activity-mask, surface-policy,
  capacity-dispatch, topology-transaction and xctrace-segmentation tests passed
  after the final source changes;
- the phase-migration check passed, including the now-provable collocation plus
  divergence-partial coalescing;
- the stage-timing contract passed;
- canonical Dawn lanes passed for symmetric expansion, coarsening-biased
  hydrostatic adaptivity, four-second mini32 conservation, min8 regional surface,
  mini32 performance (23.1997 ms median under 40 ms), and mini64 performance
  (37.8798 ms median under 50 ms).

The complete canonical suite is not green on the current shared working tree:

- `mini64-min8-surface` reported a 17.1683-cell complete-cell ridge;
- `long-dam-far-wall` observed 276 generation-zero leaves where its fixture expects
  80;
- `tall-cells-hills-far-wall` timed out, after which the suite exhausted its 180 s
  budget before the three live-edit/D4 lanes.

The skipped lanes were then run individually after the final enclosure cleanup.
`live-liquid-injection` and `outside-tank-symmetric-collapse` passed.
`live-rigid-body-coupling` reproduced the pre-cleanup line-58 `false !== true`
assertion and the Dawn child subsequently reported `SIGSEGV` while unwinding.

Those failure signatures are in presentation/world-growth behavior, while the first
cleanup changes collocation diagnostics and stage ownership. The working tree also
contains concurrent presentation/world-growth edits that were deliberately
preserved. The result is therefore reported as a shared-tree gate failure, not
silently attributed to or fixed inside this performance cleanup.

TypeScript's repository-wide check is not green on the shared tree because of
unrelated existing errors in the Losasso harness, a ceiling-slab test and older
activity-policy tool literals. Those errors were not edited or masked during this
round.

## Large-scene re-review and second cleanup round

The mini16 FPP1 aggregate was split under the existing diagnostic-only pass-isolation
switch without changing the production command graph. On mini16, the 1.160 ms
aggregate resolved to 0.900 ms of accepted-output surface proof and 0.258 ms of
packet publication. That made the proof look like the next target, but the required
larger-scene capture changed the priority:

| large power-dam B8/P8 stage | ms/advance | share of attributed GPU time |
|---|---:|---:|
| candidate transfer | 7.753 | 41.6% |
| surface sharpening | 1.556 | 8.4% |
| conservative transport | 1.024 | 5.5% |
| transport velocity extension | 1.013 | 5.4% |
| resolution planning | 0.973 | 5.2% |
| activity measurement | 0.844 | 4.5% |
| FPP1 surface proof | 0.744 | 4.0% |
| FPP1 packet publication | 0.195 | 1.0% |

The scene carried 30,080 active samples at a 0.367 active compression ratio. Its
18.81 ms of GPU-busy time had only 5.4% mean compute occupancy. Surface proof did
not scale with the larger topology, while candidate transfer became 42% of the
frame. This is why the proof was left intact: weakening or approximating it would
sacrifice aggressive coarse-surface safety to optimize the wrong workload.

Timestamp receipts on a moving long-dam front localized the candidate spike to
coarse-to-fine scalar/momentum transfer. Each of the eight B8 children of a B4
parent independently repeated the same eight-child open-volume mass census. One
refined brick therefore performed 4,096 redundant limited-linear density
reconstructions. The retained cleanup computes that parent-invariant correction
once per B4 parent into workgroup memory, barriers once, and lets each child consume
the exact cached value. The conservative reduction order, generation-local
transaction, topology decisions and B4-to-B8 output field are unchanged.

Matched B8/P8 long-dam receipts show the result on refinement frames:

| metric | before | parent cache | change |
|---|---:|---:|---:|
| candidate field transfer, advance 3 | 4.391 ms | 1.180 ms | -73.1% |
| candidate field transfer, advance 7 | 4.915 ms | 1.376 ms | -72.0% |
| whole candidate transfer, advance 3 | 5.898 ms | 2.753 ms | -53.3% |
| whole candidate transfer, advance 7 | 6.357 ms | 2.884 ms | -54.6% |

Non-refinement frames remained at the same timestamp floor. In the production
shipping pass layout, the identical 536-step large power-dam run improved from
17.87 to 14.47 ms/advance (-19.0%). Both runs ended with 30,080 active samples,
0.258 total compression and 0.367 active compression; the optimized run reported
zero represented-volume drift, no rejected advances and no validation errors.

Two failed measurements updated the profiling model rather than being discarded:

1. A 174-step xctrace run ended before Instruments' attach latency and counter
   window. Large-scene captures must size the *stepping phase* independently of
   variable shader construction time; 536 advances supplied a valid steady window.
2. A timestamp probe requested samples faster than the instrumentation cadence and
   captured only 3/8 hydrostatic frames. The corrected 120 ms cadence captured 8/8,
   and moving-front topology was required because hydrostatic candidate transfer
   only exposed its 0.33--0.59 ms fixed floor.

Receipts:

- `artifacts/xctrace-sparse-cm12-large-components-2026-09-01/report.html`
- `artifacts/sparse-cm12-long-dam-candidate-stage-cost-2026-09-01.json`
- `artifacts/sparse-cm12-long-dam-candidate-parent-cache-2026-09-01.json`
- `artifacts/sparse-cm12-large-parent-cache-wall-2026-09-01.log`

Post-change validation kept the coarse and dynamic guards green: focused
hydrostatic adaptivity, live liquid insertion and outside-tank symmetric collapse
all passed; the canonical mini64 timing lane passed at 38.7318 ms in isolation and
35.7171 ms in the full matrix. The full canonical matrix again passed its first six
lanes, then reproduced the pre-change 17.168268-cell mini64 surface ridge, the
long-dam 276-versus-80 generation-zero assertion, and the tall-cells timeout before
exhausting its 180 s budget. No ceiling or correctness assertion was changed.

### Post-candidate face-preparation experiments

Re-ranking the post-change long-dam receipt put dirty oriented face-row preparation
at 6.488 ms median. Two plausible cleanups were measured and removed:

1. **Fusing physical-seam and sparse-air seam consumers regressed.** The two kernels
   walk the same packed address list and select disjoint family halves, so one fused
   traversal appeared to remove repeated indirect loads. Instead, dirty-face median
   rose from 6.488 to 7.275 ms and p95 from 8.651 to 13.566 ms. The mixed address
   branch and both inlined row resolvers increase register pressure/divergence across
   the much heavier RK2 trace. Projection saved only one timestamp quantum. The
   specialized kernels were restored.
2. **Caching `rowCenter` in a live `vec3` regressed.** This removed six repeated
   atomic row-word loads, but kept the vector live across the complete RK2 trace.
   Dirty-face median rose to 8.520 ms and p95 to 11.600 ms. The source was restored
   to short-lived row metadata reads.

The updated model is that face preparation is register/live-range constrained once a
row enters the characteristic trace. Its next useful optimization must remove whole
row traces—using a generation-stamped compact extended-face domain or a conservative
temporal characteristic receipt—without widening the specialized hot shader. Any
compact domain must include dynamic and sparse-air lifecycle rows and prove equality
against the accepted BFA1 row set; an accepted-row list alone is not sufficient.

Rejected-experiment receipts:

- `artifacts/sparse-cm12-long-dam-fused-seam-face-stage-cost-2026-09-01.json`
- `artifacts/sparse-cm12-long-dam-face-center-cache-2026-09-01.json`
