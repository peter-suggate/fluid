# Uniform Geometric compiled topology and surface performance

6 October 2026. Design analysis for the current dynamic h/4h Uniform Geometric implementation, including the working tree changes. The original analysis used source inspection only. The user subsequently authorized GPU measurement and set a target of 25% lower frame time on `cm12-figure-9` in Dynamic mode. The implementation and measurement record below distinguishes completed changes, rejected experiments, and that still-unmet target.

This is a chronological research record. The latest retained implementation and
validation are in **Phase 3: search locality and specialized extension launches**
below; earlier pending-work statements and representation tables describe their
respective checkpoints. In particular, stencil bits 24–26 are now the Phase 2
far-positive flags, not spare bits.

## Decision

Compile ownership-dependent vertex and face relationships when building a layout, and reuse them across simulation stages. Keep regular interiors arithmetic. Refresh values when their source fields change, while preserving the compiled geometric recipes. Begin with a shared topology contract used by both CPU and GPU builders, surface vertex traversal, phi resolution, surface-volume correction and seam faces. Prepare one shared surface sampling field per phi epoch and read it directly during fine redistance, eliminating per-workgroup windows.

This is a structural replacement of repeated topology discovery with reusable data. It does not change the h/4h ownership policy, pressure tolerances, characteristic integration, interpolation formulas, sharpening sweeps, conservation rules or work admission.

## Existing reuse and the remaining opportunity

The implementation already amortizes substantial work:

| Existing mechanism | Work already shared |
| --- | --- |
| Frozen 27-bit tile stencil | Neighbor widths |
| Coarse velocity cache | Restriction of fine face patches |
| Hanging velocity taps | Mixed reconstruction shared by surface and momentum |
| Resolved phi | Hanging-vertex reconstruction before subsequent consumers |
| Sharpening geometry cache | Admission flags across eight sweeps |
| Pressure-band rows and neighbor slots | Coefficients and connectivity across iterations |
| Fine redistance workgroup window | Field samples across Newton iterations |

A surface tile is classified from changing fields; a seam tile is classified from ownership. Their lifetimes differ. A mixed 3-by-3-by-3 stencil includes edge and corner proximity, whereas sharpening's exceptional coarse faces require an h neighbor directly across a face. A single universal seam worklist would overprocess some stages.

The default field storage is domain placement, with no atlas directory lookup. Packed atlas and haloed patch implementations are QA alternatives. Production still pays address arithmetic, field-home selection and some ring/store bookkeeping. Optimizing a generic page lookup would target the wrong default path. See `lib/methods/uniform/uniform-detail-fields.ts`.

## Compile vertex ownership and reconstruction

Vertex operations repeatedly decide which owner may write a canonical vertex, whether the vertex hangs, which coarse owner supplies its reconstruction, and which corners and weights to use. These decisions depend on ownership and position, not current phi.

Surface advection and redistance revisit canonical ownership. Surface-volume correction also resolves authorities. `uniform-mixed-phi-resolve.ts` and `uniform-mixed-surface-volume.ts`'s `resolveScale` contain almost identical staged incident-owner selection and interpolation. They should consume one shared reconstruction recipe with different field loaders.

The recipe is:

```
tile-local vertex
  -> stored or reconstructed
  -> relative coarse authority
  -> coarse corner offsets and interpolation weights
```

The second stencil word currently holds minimum width in bits 27-31 and the detail-ring flag in bit 0. Its 26 spare bits can hold eight coarse-incident bits for the negative octant, seven coarse-neighbor bits for the positive octant, and eight bits identifying a coarse tile's owned canonical corners. That is 23 bits without increasing topology allocation. CPU and GPU builders must produce the same encoding during their existing classification.

For an unstored vertex, eligible incident tiles depend only on which local coordinates are zero. There are eight coordinate categories. Intersect the corresponding constant eligibility mask with the compiled coarse-incident mask and select the first eligible coarse tile. All coarse owners are numbered in tile-key order, so the selected tile is the same lowest-index coarse authority as the current walk. Use the original interpolation formulas, positive-weight handling and D4 sum.

The implementation tests enumerate all 256 ownership layouts on a 2-by-2-by-2 tile domain, comparing every vertex with an independent incident-owner walk. Additional CPU cases cover thin domains, interior seams and remote owner renumbering. A Dawn test compares CPU and GPU topology words and checks authority, face patches and bit-exact reconstructed synthetic vertex values across those 256 layouts. This establishes the finite topology contract; integrated field behavior needs the existing surface and conservation lanes as well.

The compiled recipes are stable under compact-owner renumbering because they contain relative tile choices, not owner indices. Fine canonical ownership can use the positive-octant mask with the existing local vertex traversal. Coarse canonical writers can use an eight-bit mask rather than repeating the eight-owner authority search.

## Redistance preparation: original proposal and final representation

Fine redistance already stages a 14-cubed window: 2,744 floats, or 10,976 bytes of workgroup storage. It also stages the widths of the window's 4-cubed tiles. `umResolvedVertex` fills each coarse portion by independently loading eight coarse corners for each window point. Adjacent points repeatedly fetch the same corners.

The entire window spans at most 5-cubed aligned coarse vertices: 125 floats, or 500 bytes. Stage those once, fill fine portions from the resolved h field, and fill coarse portions by evaluating the same interpolation from workgroup memory. Then perform the existing Newton searches unchanged. Preserve the existing direct certified-fine path.

This reduces source-level texture requests and address evaluation. It is not an eightfold bandwidth claim: the current texture cache can serve repeated requests. Extra synchronization and shared memory may affect occupancy. Field values cannot survive an intervening phi write; only the geometric recipe can be shared across stages.

The window spans tile offsets -1 through +2 on each axis. A cache covering this footprint has a larger dependency region than an immediate-neighbor topology mask.

**Final implementation:** the per-window coarse cache was superseded twice: first by shared coarse samples, then by a complete prepared fine/coarse sampling field. Fine Newton searches now read that field directly. The 14-cubed window, its width masks, its admission reduction and its staging barriers are gone. The preparation pass stages only eight coarse corners, and copies fine values. Existing deferred-list storage holds the field during this epoch, so no allocation was added.

## Share seam-face recipes

`umFace` derives an owner origin, probes a neighbor, resolves width and compact index, determines subdivision and constructs an anchor. A coarse face beside h has sixteen patches. In sharpening, proposal, limiter and commit repeat face construction across eight sweeps, even though admission geometry is already cached.

A coarse owner with s fine face-neighbors has F = 6 + 15s incident patches. Limit and commit can each traverse these every sweep, in addition to positive-face proposal work. There is meaningful reuse even when ownership changes every frame.

Separate geometric recipes from current storage addresses:

| Shared geometry | Stage-local resolved data |
| --- | --- |
| Side type and boundary status | Current neighbor owner bases |
| One or sixteen patches | Sharpening budget and flux addresses |
| Anchor and local-lane formulas | Extension or pressure-band slots |
| Neighbor tile relationship | Current field bindings |

Tile words can be loaded once for an executing tile or face and their local lanes derived arithmetically. A sixteen-patch face has one neighboring tile relationship. Regular fine interiors retain index offsets 1, 4 and 16. Negative domain planes and RGBA single-writer packing remain explicit. Each stage preserves its summation order.

A table containing only neighboring tile IDs is unhelpful: IDs are already arithmetic. Useful compiled information removes repeated classification and decoding. A six-u32 descriptor per tile costs 0.75 MiB at 128 cubed and 6 MiB at 256 cubed. Six u32 indices per finest cell cost 48 MiB and 384 MiB respectively. Prefer compact tile rules and cooperative staging over an expanded per-cell graph.

Sharpening also repeatedly maps its packed lane to side and patch. The six side bits define only 64 configurations, so this schedule can be derived from a shared finite recipe rather than six-side discovery per lane. Its benefit must include any table loads it introduces.

## Compile fixed hanging-velocity tap recipes

Surface and momentum already share `UniformMixedHangingTaps.unitVelocity`. Coarse seam taps are currently produced through the general mixed sampler. Their locations are fixed lattice sites, so under fixed ownership each can be classified as a canonical face load, coarse-cache interpolation or negative-boundary read. Source relationships and interpolation fractions can be compiled with ownership; values refresh for each extended-velocity epoch.

This optimizes cache production. Consumers already avoid much of this reconstruction, so the scope is narrower than the shared vertex contract. Arbitrary characteristic positions still depend on current velocity and timestep, and cannot be frozen with ownership.

`umVelocitySamplingWeights` enumerates fine neighbor boxes. Some boxes are geometrically redundant: within the current tile a fine face-neighbor can dominate a fine edge or corner neighbor extending in the same direction. An exact reduced candidate mask could shorten repeated weight evaluation. This requires a proof including boundaries; an approximate sampled weight field changes the method.

## Cache lifetimes and dynamic adoption

| Cached information | Invalidated by |
| --- | --- |
| Width relationships and relative vertex authority | Ownership changes within its footprint |
| Face subdivision and canonical writer | Ownership changes within its footprint |
| Compact owner indices and seam slots | Generation renumbering |
| Physical field addresses | Extent, allocation or binding changes |
| Hanging velocity values | Extended velocity or boundary changes |
| Resolved phi | Canonical phi writes |
| Sharpening admission flags | Geometry, solids or policy changes |
| Pressure coefficients and band neighbors | Setup and band membership changes |
| Transport donor recipes | Departure geometry changes |

The GPU builder places all fine owners first and coarse owners afterward, and independently compacts seam slots. One local edit can renumber distant owners and slots. A local changed-tile list therefore cannot invalidate an arbitrary cache of compact indices. Persistent geometry uses stable tile keys, relative offsets and local lanes; current compact addresses are refreshed separately.

The existing radius-one dilated change list covers immediate incident topology, but not the redistance window or radius-three detail ring. Start by compiling the small bitfields inside the existing full classification pass. Incremental maintenance is a separate optimization.

Persistent recipes belong in ownership topology or dedicated persistent storage, not the scratch range overwritten by successive non-pressure stages. Adoption must publish matching topology and recipes, including CPU-built layouts, GPU generations, unchanged generations, capacity growth and failed/deferred admissions.

## Boundaries of the shared representation

The pressure band already bakes neighboring slots and coefficients. Its active membership and numerical coefficients depend on fields, so shared topology can feed setup but cannot replace its prepared operator. `uniform-pressure-band.ts`'s `bCellNear` is an existing example of tile-local arithmetic plus baked cross-tile relationships.

Transport donors depend on current departures. `donorAt` already uses one compact tile word and shifts. Caching donor indices within an advance is possible, but increases metadata in potentially large rows and must outperform the existing compressed representation. It cannot persist with ownership alone.

Compiled topology does not remove Newton iterations, characteristic substeps, cubic interpolation or surface-volume trial evaluation. These retain arithmetic and dependency chains. Source inspection cannot determine the fraction of frame time spent on topology or establish a speedup.

## Implementation and verification

Implement the shared vertex and face contract first, use it across consumers, and stage the unique coarse corners in the redistance window. Prefer a few coherent structural changes to independent micro-tweaks.

CPU reference checks should enumerate incident widths and local positions, test domain faces/edges/corners, compare both ends of every canonical face, and change distant ownership to exercise compact renumbering. GPU validation should compare CPU/GPU builder topology and preserve the numerical lanes for surface resolution, sharpening, volume conservation, symmetry, solids and dynamic churn. Shader compilation must be checked in actual consumers because common source is assembled and rewritten for multiple namespaces and storage modes.

The clean-repository gate remains `npm run check:types`, `npm run test:unit` and `npm run test:dawn`. Dawn runs serially under the WebGPU lease. Do not weaken a lane or raise timing ceilings. Correctness checks are separate from performance experiments; no performance gain is claimed until a controlled comparison includes topology construction, value refresh, all consumers and dynamic adoption.

## Implementation record

All outstanding repository work was checkpointed as `fe587ebb` before integrating these consumers. The checkpoint includes the first version of this analysis and the compiled-topology module; it also includes substantial earlier repository work. It is a recovery point, not evidence that the full test gate passed at that revision.

The first implementation replaces repeated discovery in these paths:

| Path | Structural change | Remaining cost |
| --- | --- | --- |
| CPU and GPU layout builders | Compile incident, positive-octant and canonical-writer masks during existing stencil classification | Fixed mask arithmetic per tile when building a generation |
| General vertex authority | One compiled incident-mask selection instead of up to eight owner probes | Resolve the selected owner through its current tile word |
| Phi and volume-scale resolve | Shared reconstruction body; remove staged incident tile words and each vertex's candidate scan | Stage coarse field values, calculate the same weights and sum |
| Fine/coarse surface writers and wall admission | Read compiled authority masks | Field-dependent admission and numerical work |
| General face traversal | Resolve the first patch once; derive further patch anchors and owner lanes arithmetically | One cross-tile neighbor resolution per face invocation |
| Fine interior faces | Direct owner-index strides | Bounds and interior classification |
| Sharpening seam classification and schedules | Extract six side bits; use a compiled side order for all 64 configurations | Stage-specific admission, compaction and neighbor resolution |
| Sharpening proposal geometry | Retain the open-face result in a third bit of the existing admission word; reuse through eight sweeps | Current volume, budgets and limiter values |
| Fine redistance | Prepare one shared sampling field per phi epoch, then read it directly | Once-per-epoch preparation, cached global samples and Newton searches |

The stencil word allocation is now explicit:

| Bits | Meaning |
| --- | --- |
| 0 | Existing detail-ring flag |
| 1–8 | Coarse incident tiles in the negative octant |
| 9–15 | Coarse neighbors in the positive octant, excluding the current tile |
| 16–23 | Canonical corners owned by this coarse tile |
| 24–26 | Spare |
| 27–31 | Existing minimum width |

Relative vertex recipes remain stable under owner renumbering. Sharpening now caches generation-specific compact neighbor addresses during each geometry preparation; these are reused for that encode's eight sweeps and rebuilt before the next encode. They are not reused after an adoption without refresh. The persistent cross-stage representation is the compiled stencil contract. The same face helpers also serve velocity sampling, frame planning, face dispatch and presentation, so those generated shader compositions must remain part of integration validation.

The initial 125-float coarse lattice and packed width masks were useful intermediate versions. They have been superseded by the direct shared-field representation described above. The final fine redistance consumer reserves no shared field window and performs no per-window interpolation or staging. Its field values are refreshed after phi changes, never reused across epochs.

The additional structural implementations are now present in the working tree:

| Change | Contract and memory cost |
| --- | --- |
| Reduced velocity blend candidates | A fifth topology word per tile stores the nondominated fine-neighbor mask. CPU and GPU builders agree; clipped box-distance weights are unchanged. +4 bytes per tile. |
| Fixed hanging velocity taps | Integer eighth-cell recipes use the existing coarse cache and compiled face-side mask. No owner/face discovery for each fixed tap; no new allocation. |
| Shared surface preparation | Fine values and interpolated coarse h samples are prepared once per redistance phi epoch. The producer stages eight coarse corners. All fine Newton searches read the dead deferred-advection scratch directly, before coarse redistance reuses it as a list. No new allocation; the entire 14-cubed workgroup window is gone. |
| Prepared sharpening neighbors | Six packed links per listed tile contain neighbor base, width, and listed admission. Fine interiors use lane strides; regular boundaries and seam patches reuse the links throughout all sweeps. +24 bytes per tile, rebuilt with geometry. |
| Compact hanging records | Removed an unused 125-float vertex payload from every hanging slot. Phi already lives in the resolved field. Saves 500 bytes per slot (8.192 MB at Figure 9's maximum 16,384 slots). |
| Cached momentum blends | The prepared h and 4h velocity fields also handle seam blends in the fast characteristic path. The actual local owner width still controls characteristic substeps. Negative h boundary-plane samples retain the general fallback. |

Compiled extension requests now replace general owner/face searches with one coarse face, two bounding faces, or four tied fine patches. The independent reference comparison is bit-exact across all 75,648 searching requests in the 256-layout fixture. Once this geometry became cheap, the shared request queue could be removed: 3,844 bytes of workgroup storage, per-patch atomic reservations and two barriers are gone. Each lane now plans and consumes a patch immediately, without carrying private arrays across a queue barrier. The same staged liveness pruning, nearest-distance selection, tied-value order and Godunov combination remain.

Private gradient corner caching was implemented and tested in three source forms. All retained the source interpolation formula but failed the attempted bit-exact comparison on arbitrary finite f32 fields, with small floating-point differences. That candidate was withdrawn; production retains the original six-sample gradient and its arithmetic, now reading the prepared field. An initial direct-field version used atomic loads and changed compiler contraction, producing up to 1.9e-6 difference in the new synthetic gradient test. Splitting the scratch representation into atomic header counters and uniquely written ordinary data words restored bit-exact parity across all 32,768 comparisons without relaxing the test. The whole Figure 9 run also restored the earlier candidate’s recorded mass and represented-volume results. This is why storage semantics must be included when examining apparently identical arithmetic. The independent fixed-tap and topology comparisons remain bit-exact. The new momentum comparison uses the general prepared-field sampler as reference, bounds sample differences by 1e-6 for source velocities in [-4,4], and characteristic differences by 1e-5 cells, over all 256 layouts, anisotropic spacing, and short/long characteristics.

A 12-cubed sub-window split remains an evaluated alternative rather than a selected implementation: eight jobs stage roughly five times the data and reduce active vertex lanes per group. Its lower shared-memory reservation alone does not justify that duplication. Lossy phi compression and altered reduction order remain outside the numerical contract.

Validation below records earlier integration evidence; the full final gate must be rerun after all edits. The current working tree is **not yet landed or clean-gate certified**.

| Check | Result |
| --- | --- |
| TypeScript | `npm run check:types` passed |
| Complete unit suite | 890 passed, 80 skipped, zero failures |
| Compiled CPU topology checks | All five tests passed, including exhaustive layouts, window octants, seam schedules and remote renumbering |
| GPU builder, vertex and face parity | Passed across all 256 layouts, including bit-exact staged reconstruction and packed window masks |
| Sharpening seam correctness | Passed the independent dense-reference lane across layouts, solids and policies, with conservation and per-sweep checks |
| Presentation integration | `uniform-volume-dawn.test.ts` passed (2 tests); this file checks presentation and does not exercise a volume-correction advance |
| Long-dam coarse-surface correctness | Passed its existing conservation, advancement and impact assertions |
| Complete Dawn gate | Pending a fresh complete run after the final changes |

## Reducing workgroup storage without giving up reuse

The optimization hypothesis should survive an early disappointing result. Fewer lookups can expose another bottleneck, and adding a cache can reduce arithmetic while lowering occupancy. The useful response is to inspect the changed cost and representation, not immediately discard sharing. None of the following source-level savings proves an occupancy improvement; allocation granularity, registers, generated machine code and the device still matter.

The development sequence included these representation changes (the final surface field supersedes the first row):

| Working data | Previous representation | New representation | Exact saving |
| --- | --- | --- | --- |
| Redistance window widths | 64 u32 widths, loaded from 64 tile words | Eight 8-bit octant masks in eight u32 words, extracted from eight existing stencils | 224 bytes of workgroup memory; 56 fewer topology reads per staged mixed window |
| Sharpening seam terms | 192 pairs of float value and presence flag | 192 float values; absent contributions are zero | 768 bytes of workgroup memory |
| Volume seed and reduction | 125 corner floats plus 64 two-float rows | One 128-float scratch array reused after the existing uniform barrier | 500 bytes of workgroup memory |

The width masks partition the 4-cubed tile footprint into eight 2-cubed octants. Each octant comes from one already compiled 3-cubed stencil at T plus that octant's corner. The extractor clips the upper domain and preserves the full footprint; this is not a radius-one approximation. Eight independently written words avoid both atomic packing and a new synchronization point. Packing those eight bytes into two words would save only another 24 bytes while adding coordination or concentrating preparation into fewer lanes, so that is not the current design.

Sharpening's term flag is redundant for finite numerical state: an absent face adds zero to the serial sum. Full f32 contributions and their order are retained. Removing the flag can add zero-valued arithmetic where the previous loop skipped a term, so the benefit is a smaller representation and fewer shared loads, not a claim that every instruction count falls. The side schedule is separately compiled into six 3-bit side IDs for each of 64 masks, a 256-byte constant table. It replaces repeated six-side scans in proposal, limit and commit. Its indexing cost still needs eventual measurement.

The existing geometry cache also now carries an open-face bit. Before this change, `umFaceFlags` checked owner solids and face aperture during geometry preparation, but `umProposal` repeated those checks on every sweep. The third cached bit preserves the open result even when both directional admission bits are zero, which matters for the orphan policy. Proposals reject zero transfer budgets before reading the cache: quiet owners intentionally have no prepared geometry. For a face that can transfer mass, the lower owner is active and has prepared the cache. Geometry, solids and policy remain fixed through the encoded sweeps; rebuilding geometry after changes is already the stage contract. This removes repeated solid/aperture discovery without enlarging scratch storage.

The volume seed's lifetimes are disjoint. All lanes finish corner classification before the `workgroupUniformLoad` of `measureBand`. A seeded group returns uniformly. An unseeded group overwrites the same storage with its reduction rows, whose publication and tree barriers already exist. The reduction uses the same pairwise addition order. This is a useful pattern for other kernels: prove last reads and first writes against existing barriers before merging storage, rather than merely declaring an alias.

The original dominant redistance allocation was 14-cubed f32 values (10,976 bytes). Aliasing its coarse lattice while filling the window would be a data race; moving the lattice to private arrays would shift pressure into registers. The selected design removes the window instead. It accepts cached global reads during Newton iteration in exchange for preparing each field value once and no per-tile shared reservation.

The larger design decisions are now resolved:

1. **Shared surface sampling field:** implemented, including its full dependency footprint and refresh after phi writes.
2. **Prepared seam neighbor bases:** implemented for regular and seam sharpening tiles, refreshed during each geometry preparation.
3. **Smaller window jobs:** evaluated but superseded by removing the window. Eight 12-cubed jobs would stage roughly five times the data; the chosen design shares one field across all jobs.
4. **Lossless packing:** implemented for topology, schedules, redundant flags, scratch lifetimes and hanging records. Lossy f16 phi or reordered reductions remain outside this work.
5. **Compiled extension geometry:** implemented and used to remove queue storage, synchronization and per-patch private state that its former expensive search required.

Another candidate is sharing corners within a Newton gradient evaluation. `umWindowGradient` evaluates six trilinear samples at q plus or minus 0.25 along each axis: 48 source-level workgroup reads. If no sample crosses a cell boundary, all six use the same eight corners. Each axis can cross at most one boundary and adds at most four distinct corners, so the union contains at most 20. A private corner cache could preserve each sample's weights and summation order while reusing those values. Its cost is registers, selection logic and possible dynamic-array spills; an analytic trilinear derivative would be a different numerical operation and is not a drop-in replacement for the current centered difference. This candidate was subsequently implemented, tested, and withdrawn as described above.

Persist with the objective of reducing total repeated work across an epoch. Evaluate each representation by preparation, reuse count, field lifetime, shared memory, registers and critical-path dependencies together. A local kernel improvement that shifts more work into every other stage is not the intended result.

## Controlled Figure 9 measurements — in progress

These are simulation-frame measurements, not browser render FPS. Both arms use
`uniform-volume`, the production solver factory, the UI defaults with
`detailPolicy=dynamic`, a 128×128×64 grid, 1/60 s steps, and 120 consecutive
frames (two simulated seconds). There is no artificial inter-frame delay;
frames 1–4 are excluded. Wall time includes advancing and waiting for submitted
GPU work; rendering and statistics readbacks are excluded. GPU times use
hardware stage timestamps. Physics controls, detail thresholds, pressure
accuracy, and sweep counts are identical.

The baseline is the `fe587ebb` checkout with the unrelated current pressure
changes copied into it, so those pressure changes are held constant. Their
hashes are checked against the live checkout. None of the topology changes in
this document is included in that baseline. An earlier 30 Hz declared-default
probe was exploratory and is not used for the UI comparison.

| Arm | Mean wall ms | Mean GPU ms | Change |
| --- | ---: | ---: | --- |
| Baseline A | 21.420 | 19.136 | Fixed comparison state |
| Candidate A | 20.745 | 18.414 | Compiled topology, fixed taps, reduced blend masks, seam cache, workgroup packing; original gradient |
| Candidate B | 21.358 | 18.216 | Shared surface preparation and removal of unused hanging payload |
| Candidate C | 20.838 | 18.191 | Prepared momentum blend path |
| Candidate D | 20.277 | 17.958 | Neighbor links reused by regular sharpening tiles as well as seams |
| Candidate E | 20.094 | 18.045 | Compiled extension requests, retaining queue |
| Candidate F | 20.093 | 17.825 | Remove extension request queue and its barriers |
| Candidate G (superseded) | 19.434 | 17.286 | Direct prepared surface field, atomic data reads; rounding difference |
| Candidate H | 19.393 | 17.109 | Ordinary uniquely written scratch data; exact gradient parity restored; no per-tile window/admission barrier |

Candidate H is about 9.5% faster in wall time and 10.6% in GPU time than the initial baseline run. The surface stage declined from 3.056 to 2.366 ms; the queue-free extension stage in F was 1.955 ms versus 2.332 ms at baseline. **This does not meet the 25% target.** These development runs still need repeated interleaved baseline/candidate measurements. Candidate G's rounding-changing representation was replaced, not accepted by weakening a numerical lane.

The normal runs are saved under `/tmp/fluid-figure9-ui-*.json`. A separate
per-kernel diagnostic splits dispatches into timestamped passes and disables
Dawn's timestamp quantization. That diagnostic changes command overhead and is
never included in the headline comparison. At frame 100 it identified seam extension sweeps (1.827 ms isolated) and fine redistance (1.802 ms isolated) as the largest individual kernel totals outside pressure, motivating the subsequent changes. It exists to identify which
preparation and consumer kernels actually dominate the remaining cost.

Recent checks: all four surface integration files passed (coarse redistance,
coarse-surface long-dam impact, cubic surface, and detail surface). Prepared
momentum passed its 256-layout comparison. Prepared regular/seam sharpening
passed its dense-reference/conservation test. Type checking and the complete
unit suite passed (890 passed, 80 skipped); the production build passed. The full Dawn gate completed with 55 of 58 files passing. All three failing files reproduce on the control checkout with the same unrelated pressure work and without the topology changes: raster AO spotlight specialization differs by two half-float steps; coarse-solid-rest has the same two drift failures and one pressure-convergence failure; pressure-local-visit fails its reference coarse-visit assertion. The new interior extension and prepared-gradient comparisons also pass separately. These failures still prevent a clean-repository claim; no thresholds were relaxed. The retained implementation is checkpointed on local `main` as `6c7f1712`, with its production build passing. Final interleaved measurements and the 25% target remain outstanding. No remote deployment is claimed. Benchmark reports now record the commit and SHA-256 of each Uniform source file at startup, so dirty candidates are identifiable independently of their branch name.


## Prepared surface storage lifetime

| Dispatch phase | Scratch data interpretation | Access proof |
| --- | --- | --- |
| Advection and wall gathering | Packed deferred vertex IDs | Atomic header allocates disjoint ranges; data entries have one writer |
| Deferred advection | Vertex list | Producer dispatches have completed before reads |
| Surface preparation | Full-precision phi sample bits | Positive tile owns each vertex; upper domain closure has one writer |
| Regular and mixed fine redistance | Prepared sampling field | All consumers read the same immutable phi epoch; no window or topology lookup per sample |
| Coarse band collection | Packed band vertex IDs | Every fine consumer has finished; atomic claims allocate disjoint ranges |
| Coarse band redistance | Band list | Reads only entries published by band collection |

The four header words keep their original offsets and atomic types. Data begins
at byte 16 and uses ordinary u32 accesses. The allocation size and bindings are
unchanged. There are no concurrent list and field interpretations. Dispatch
ordering supplies the required global visibility; no host readback or new CPU
synchronization was introduced.

Development summaries and preserved controls are saved in
[measurements.json](uniform-compiled-topology-2026-10-06/measurements.json).
The full final comparison will include per-frame timings, quality values and
source hashes. A 25% reduction from the original 21.420 ms baseline would require
16.065 ms, still about 3.33 ms below candidate H. That remaining gap is a
performance objective, not a reason to reduce numerical work or relax tests.


## Focus: the splash, not a wider solver rewrite

The user narrowed the next optimization work to topology, cache reuse and
compaction in Figure 9's splash. Keep the existing whole-run measurement and
also report frames 91–120 (1.5–2 seconds at 60 Hz). This diagnostic window does
not replace the original whole-run target.

| Late-splash mean | Baseline A | Candidate H |
| --- | ---: | ---: |
| Simulation wall time | 32.058 ms | 26.854 ms |
| GPU time | 28.604 ms | 24.818 ms |
| Surface transport and redistance | 5.140 ms | 3.650 ms |
| Extension and hierarchy | 4.004 ms | 3.246 ms |
| Sharpening | 3.244 ms | 3.032 ms |

These single-run results indicate a 16.2% wall-time reduction in the late
splash. The next interleaved comparison must confirm it, including active fine
owners and changed tiles so reduced work is distinguishable from cheaper work.

Priorities within this scope, in execution order:

1. **Extension support and address reuse.** First pack support membership into
   spare bit 30 of the already staged topology words. Staging already reads
   support to load finite masks, so this adds neither a global load nor shared
   storage. Slot decoding masks out the bit. Compare this against the fused,
   queue-free search, including both sweep costs. If useful, examine compact
   slot bases and recipe types shared across both sweeps; refresh addresses
   after ownership renumbering. Values and finite-distance masks remain
   sweep-specific. Do not expand a six-neighbor graph for every h face.
2. **Surface search compaction.** Keep the shared full-precision prepared field.
   Measure prepared samples, admitted Newton searches and active lanes per
   tile before deciding whether to compact individual searches. Empty lanes
   alone do not justify a list: include classification, writes, indirect
   dispatch and lost spatial locality in the measured surface stage. If cache
   coverage is wasteful too, derive the union of the full sampling footprint
   from admitted searches, including gradient offsets and the entire search
   reach. Do not shrink the search radius or reintroduce per-tile windows.
3. **Sharpening reuse, only if the measurements justify it.** The six side
   links are shared already, but the splash gain is small. Measure
   admitted/listed/active tile ratios and separate link preparation, proposal,
   limiter and commit. Compact preparation to the consumers' exact dependency
   closure only if it is broad today; otherwise examine repeated data loads
   inside the eight sweeps.

For each candidate, compare the same frames 91–120, with matching source
hashes for unrelated code, fine/coarse owner counts, changed tiles and quality
values. Preserve a whole-run comparison too. First use a local A/B to reject
regressions, then baseline/candidate/candidate/baseline to confirm the selected
combined change. These are simulation-and-fence timings, not rendered UI FPS.

Pressure, rendering and transport redesigns are outside this focused next
iteration. The long repository checks are correctness validation, not a new
optimization direction.


### Focused follow-up: observed demand and rejected packing

The fused queue-free extension consumer (I) and subsequent variants used the
same production defaults, 120 frames at 60 Hz, with the same unrelated
pressure source hashes. Their recorded per-frame quality and ownership
counts match exactly.

| Variant | Whole-run wall / GPU | Splash wall / GPU | Splash extension |
| --- | --- | --- | --- |
| I: fused extension, existing staged words and widths | 19.433 / 17.184 ms | 27.095 / 25.002 ms | 3.268 ms |
| J: support membership packed in staged words, rejected | 19.598 / 17.206 ms | 28.642 / 25.061 ms | 3.395 ms |
| K: remove duplicate staged widths, current | 19.273 / 17.100 ms | 27.831 / 24.882 ms | 3.262 ms |

Packing support alone added no storage, but did add cache-selection logic at
slot reads. It did not improve this run; the variant was withdrawn. The width
array duplicates the topology word's fine bit. Removing it saves 216 bytes
per workgroup at pack size two and the staging writes, with essentially
unchanged extension time. Keep this as a memory simplification, not a claimed
speedup. Every width consumer either asks whether a staged neighbor is fine
or has already established that the neighbor lies inside the domain. The
outside sentinel therefore needs no separate width entry.

A separate diagnostic appends work counters to the existing claims allocation
and injects atomic counts into the surface shader. It does not change the
production shader or binding layout, and emits no timing summaries. Its
recorded quality matches K at every frame. In frames 91–120, per-frame means
were:

| Surface work | Count |
| --- | ---: |
| Prepared fine samples | 447,702 |
| Prepared coarse samples | 445,684 |
| Owned fine vertices | 409,773 |
| Admitted fine Newton searches | 238,715 |
| Newton iterations | 776,442 |

Thus 58.3% of owned fine vertices enter a search, with 3.25 iterations per
search. These are work counts, not hardware occupancy measurements. The next
substantial candidate should compact the expensive searches, keeping the
shared field and the original admission predicate. Classification writes
unchanged values for rejected vertices and emits packed vertex coordinates
for accepted ones; dense consumer groups run the original search and write
unique output vertices. A compact list needs storage separate from the live
prepared field. Include its allocation, prefix/count work, list traffic,
dispatch and spatial-locality effects in the comparison. Do not assume the
41.7% admission gap converts directly into time saved.

For extension, the queue and duplicated width array are now gone. A narrowly
scoped next experiment can pack more than two seam tiles into a workgroup,
using the recovered shared-memory budget to amortize staging and fill more
lanes. Measure stage time and per-group useful patches; larger packs may
increase serial scheduling work or register pressure. Cache more addresses
only if measurement shows repeated decoding still dominates.

Keep sharpening out of this next implementation step. Confirm the selected
combined implementation with interleaved comparisons before claiming a frame
time improvement. The original 25% whole-run target remains unmet.

Latest type checking and full unit suite pass (890 passed, 80 skipped). Focused
compiled-topology, interior extension, symmetry, and nearest-extension GPU
checks pass after the width-array removal. The full Dawn failures described
above remain reproduced on the control; no clean-gate claim is made.


### Testing the compaction hypothesis

The next three experiments were implemented and measured, rather than assumed
beneficial. All recorded per-frame quality values match K exactly.

| Variant | Whole-run wall / GPU | Splash wall / GPU | Relevant splash stage |
| --- | --- | --- | --- |
| L: compact accepted fine searches | 19.261 / 17.092 ms | 26.982 / 24.899 ms | Surface 3.668 ms versus K 3.655 ms |
| M: four seam tiles per extension workgroup | 19.482 / 17.218 ms | 28.160 / 24.917 ms | Extension 3.218 ms versus K 3.262 ms |
| N: compile packed patch ranges once per workgroup | 19.441 / 17.104 ms | 26.873 / 24.853 ms | Extension 3.242 ms versus K 3.262 ms |

L used packed ten-bit vertex coordinates, one atomic range reservation per
64-lane classification round, and separate dense regular/general consumers.
Their lists grew from opposite ends of one bounded allocation, preserving
sampling specialization and unique output ownership. The queue followed
persistent claims data and remained separate from the live prepared field.
It added 4,326,668 bytes at the final frame. Surface stage time was neutral,
so the allocation and compaction path were withdrawn. Initial cubic/detail
surface checks passed, and the final specialized variant completed Figure 9
with unchanged quality and no validation errors.

At frame 100, L's isolated classification kernels totalled 0.064624 ms, its
search consumers 0.720446 ms, and field preparation 0.049165 ms. These are
instrumented dispatch timings, not directly comparable to normal stage
brackets. They establish that shaving classification overhead alone cannot
produce the missing multi-millisecond gain. The earlier 58.3% admission ratio
must not be treated as a hardware occupancy measurement. A future compaction
attempt needs per-workgroup admission/iteration distributions and a locality
strategy; repeating a flat global list is not supported by this result.

M and N showed small extension differences but no convincing total GPU-time
improvement. M raised staging and scheduling work with the pack size. N added
a barrier and 24 bytes of shared prefix data while removing per-patch repeated
job/live/width probes. Both were withdrawn pending stronger evidence. The
retained production implementation is K: the shared surface field, compiled
extension geometry, queue-free fused consumers, and deduplicated staged widths.

The next focused topology hypothesis is to separate immutable request geometry
from sweep-dependent liveness. Direct/search classification and seam address
recipes depend on ownership; finite masks and distance/value state change each
sweep. A compact boundary-only recipe prepared once per ownership generation
could serve both extension sweeps. First account for its construction, metadata
bandwidth and storage against the topology work actually removed. Avoid a full
six-neighbor graph for every fine face. Absolute compact indices must refresh
on adoption; relative recipes can survive unchanged neighborhood topology.
Keep the shared full-precision surface field, avoid repeating the unsuccessful
flat-list experiment, and leave pressure and rendering outside this work.


## Phase 2: shared boundary request geometry

The next phase implements the boundary-recipe hypothesis without a per-face
allocation. The existing first stencil word supplies the 27 fine-neighbor
bits. Three previously unused bits (24–26) in the second word encode whether
the tile two positions along each positive axis is fine. CPU layout creation
and GPU classification compile exactly the same representation. The metadata
stays at 20 bytes per tile; there is no new buffer or preparation dispatch.

A positive canonical patch's extension requests lie in that 3-cubed stencil,
except the upper incident tile of a width-4 request along the positive normal.
The three added bits cover exactly those exceptions. A unit request is direct
when either incident tile is fine. A width-4 request is direct when neither
in-domain incident tile is fine. Clipped stencil bits represent the domain
exterior without synthetic owners. The GPU parity test enumerates canonical
faces, requests, and domain cases against the frozen width-lookup reference.

Each seam job reads its two-word recipe once; requesting lanes retain it
while processing their patches. Both sweeps use the layout's same immutable
recipe. Compact owner indices are absent, so renumbering does not make a
relative recipe stale. Adoption still refreshes recipes whenever widths
change. The declarations add 24 bytes of workgroup state for the mixed seam
launch, with no change to allocated storage or scratch buffers.

Planning and evaluation are now one request loop. That loop computes one
integer anchor and shares it among the domain test, geometry classification,
finite-mask test and direct slot load. A searching request reconstructs its
MAC point once for the existing compiled-neighbor evaluator. Current-sweep
finite masks and value/distance slots are never cached in the layout. The
candidate selection, tie handling, spacing and axis reduction order remain
unchanged.

The strengthened GPU test adopts 256 changing 3-cubed layouts, compares all
CPU/GPU topology words, checks direct/search and domain decisions, compares
reconstructed request points bit-for-bit, and verifies both direct and search
value/distance/spacing against the old traversal with anisotropic cell sizes.
Symmetry and nearest-extension integration checks pass as well.

Measurement uses the previous retained implementation at `aa09a695`, with the
same six unrelated pressure edits copied into its isolated control checkout.
The first ABBA sequence tested fused planning with floating request points.
The final sequence brackets two shared-integer-anchor runs with controls B
and C. All runs use Figure 9, Dynamic, app defaults, 120 frames at 60 Hz, and
report frames 91–120 separately. Final numbers and gate status follow below.


| Final ABBA mean | Previous implementation | Phase 2 | Reduction |
| --- | ---: | ---: | ---: |
| Whole-run simulation wall time | 19.423 ms | 19.064 ms | 1.85% |
| Whole-run GPU time | 17.133 ms | 17.022 ms | 0.65% |
| Splash simulation wall time | 27.855 ms | 26.656 ms | 4.30% |
| Splash GPU time | 24.994 ms | 24.675 ms | 1.28% |
| Splash extension | 3.259 ms | 3.146 ms | 3.49% |
| Splash census/layout/remap | 1.510 ms | 1.456 ms | 3.55% |

The wall-time result is noisy: control B's splash mean was 28.847 ms versus
26.862 ms for control C; candidates were 26.682 and 26.630 ms. The additional
earlier control A was 26.775 ms. Do not present the 4.30% wall mean as a robust
frame-time guarantee. Extension was much more consistent across controls
(3.255–3.262 ms), and both final candidates measured 3.146 ms. The layout
stage includes more than the three new bit computations; its observed change
must not be attributed to those computations as an isolated improvement.
The retained change reduces repeated topology work and has a small measured
GPU benefit. The original 25% frame-time objective is still unmet.

Every recorded per-frame quality value, ownership count and allocation size
matches the control. The final source hashes match both selected candidate
runs. The earlier two-loop recipe probe and fused floating-point version are
also preserved in [boundary-phase2.json](uniform-compiled-topology-2026-10-06/boundary-phase2.json),
with per-frame evidence in
[boundary-phase2-frames.jsonl](uniform-compiled-topology-2026-10-06/boundary-phase2-frames.jsonl).


Phase 2 validation is complete: types pass, unit tests report 890 passed and
80 skipped, and the production build passes. The full serial Dawn run passes
56 of 59 files, including the expanded compiled-extension comparison and all
extension, dynamic-layout, force-cache, geometric-boundary and live-edit
integration files. The same three baseline files still fail: raster AO
spotlight specialization, coarse-solid-rest (two drift assertions and one
pressure-convergence assertion, plus its existing TODO), and pressure-local-visit's
reference coarse-visit assertion. All five failing assertions were independently
reproduced on `fe587ebb` with the same six pressure edits and without the
topology changes; the final run reproduces their messages and numerical values.
No thresholds were relaxed. The repository therefore does not pass its full
clean-repo gate.

The selected implementation is active in the normal Uniform Geometric
production path, including the UI's Dynamic mode, without a feature flag.
All 111 Uniform source hashes match both selected benchmark runs. Measurements
cover simulation advance and its GPU fence, excluding statistics readbacks and
rendering; they are not rendered frame-rate measurements. No remote deployment
is claimed. This phase keeps the existing full-precision surface cache and
compaction behavior, and changes only the shared boundary request geometry and
its consumers.


## Phase 3: search locality and specialized extension launches

7 October 2026. Outstanding pressure work was checkpointed as `11ea9c04`
before this phase. All controls use that checkpoint, including those pressure
changes. The final production change is the extension launch split described
below. Surface scheduling, sampling arithmetic and storage remain unchanged
from the checkpoint after the unsuccessful candidates were withdrawn.

### Measured surface demand

The extended `--surface-work-audit` records a per-vertex Newton iteration count
and sampling specialization, and reconstructs the original tile/round/lane
mapping, including negative domain closure. Aggregate counters independently
check every recorded owner, admission and iteration. This distribution mode
requires one redistance epoch per sample: multiple visits cannot be represented
by one per-vertex record and fail the consistency assertions. A GPU buffer clear
resets diagnostic storage; a large host write encountered a native Dawn upload
abort during development. No production buffer or shader changes are needed
for this diagnostic. Its timings are deliberately excluded from comparisons.

During splash frames 91–120, means were 409,773 owned fine vertices, 238,715
admitted searches and 776,442 iterations. Of logical 32-lane groups containing
owned vertices, 26.8% contain no searches. Taking each group's longest search
as its loop duration, useful iterations are 44.1% of available lane-iterations.
This is a software work model, **not measured hardware occupancy**.

| Logical packing model | Mean lane-iterations | Useful fraction |
| --- | ---: | ---: |
| Current tile/round mapping | 1,760,773 | 44.1% |
| Compact within one tile | 1,616,461 | 48.0% |
| Compact within 2×2×2 tiles | 1,461,037 | 53.1% |
| Compact within 4×4×4 tiles | 1,445,267 | 53.7% |

Spatial models retain separate regular/general sampling classes. They account
for observed loop lengths but exclude queue construction, barriers, address
arithmetic, register pressure and cache behavior. The 17% modeled reduction
for two-tile-wide blocks is consequently a hypothesis, not a predicted speedup.

### Implemented candidates and withdrawals

Two unmodified controls both measured 3.670 ms for the splash surface stage.
All rows below except the explicitly invalid branched layout matched the
control's recorded quality at every frame.

| Surface candidate | Splash surface GPU time | Outcome |
| --- | ---: | --- |
| Local search queue within 2×2×2 tiles; existing tile jobs elect leaders | 4.273 ms | Slower; withdrawn |
| Same queue; block jobs published during existing field preparation | 4.253 ms | Slower; withdrawn |
| Prepared field in 4×4×4 bricks, branched boundary indexing | 4.028 ms | Numerical parity failed; invalid comparison; withdrawn |
| Same bricks with branchless boundary indexing | 4.111 ms | Exact gradient parity restored, but slower; withdrawn |
| Contiguous x rows grouped into 4×4 y/z slabs | 3.795 ms | Exact parity, but slower; withdrawn |

The local queue held at most 729 packed coordinates (2,916 declared workgroup
bytes) and retained the original search and admission predicate. Publishing
block jobs used the existing preparation dispatch and 16,392 additional global
bytes at Figure 9's dimensions. Removing redundant tile launches barely changed
the regression, so launch duplication alone did not explain the cost. The
local queue's construction, synchronization, execution shape and storage costs
must be evaluated together; these runs do not isolate register pressure or
cache misses as the cause.

Both blocked field layouts fit the existing allocation exactly, including
upper-domain closure, and changed only addressing. The branched version failed
31,804 of 32,768 synthetic gradient comparisons by small floating-point amounts
and changed recorded quality from frame 6. Branchless indexing restored exact
gradient parity and full-run quality, but neither layout improved performance.
These results do not prove the existing field is cache-optimal; they reject
these particular layouts and indexing costs.

A final surface prototype reused the seven axial values sufficient for the
first Newton gradient at an integer vertex. A sparse arithmetic form failed
2,130 of 2,744 integer-gradient comparisons; retaining the weighted-product
loop reduced this to 1,252, still unacceptable. Neither version was enabled in
the production search or benchmarked. Original tests and thresholds remain
intact; the extra experimental tests were removed with their rejected code.

### Retained extension launch specialization

The old sweep entry point handled fine seam tiles, regular coarse tiles and
packed coarse seams. A three-way split preserved quality but increased
whole-run extension time despite a small splash benefit. The retained version
uses two launches: fine seams, then all coarse work. Regular coarse owners
still share a launch with coarse seams, avoiding a third dispatch.

The fine pipeline compiles with `ueMixedJobs=false`, has no packed-job base to
resolve, and reserves one staged neighborhood. Its word and finite-mask arrays
shrink from 648 to 324 declared workgroup bytes. Coarse jobs retain the two-tile
pack and their original recipes. These are combined changes: the measured gain
cannot be attributed solely to the 324-byte reduction or called a measured
occupancy improvement. Compiled geometry is still shared across sweeps; masks
and values still come from each sweep. There is no new global
buffer, scratch growth, admission change or altered reduction order. Two sweeps
add two dispatches per advance. A coarse-only capacity keeps its existing
regular-list path.

A differential GPU test compares every word of both extension slot arrays
after each of three sweeps over 66 changing layouts, including all-fine and
all-coarse cases, anisotropic spacing, empty support, open/closed upper walls
and different grid widths. The merged entry-point specialization remains as
the test reference; production compiles the fine/coarse specializations. All
198 state comparisons pass bit-for-bit.

The final comparison is control/candidate/candidate/control, with matching
Uniform source hashes for every file except the intended extension change.
Each run uses the production factory, Figure 9, Dynamic, app defaults, 60 Hz
and 120 frames. Whole-run means exclude frames 1–4.

| Final interleaved mean | Control | Retained split | Reduction |
| --- | ---: | ---: | ---: |
| Whole-run simulation wall | 19.970 ms | 19.334 ms | 3.19% |
| Whole-run GPU | 17.349 ms | 17.253 ms | 0.56% |
| Whole-run extension | 1.929 ms | 1.881 ms | 2.50% |
| Splash simulation wall | 29.002 ms | 26.936 ms | 7.12% |
| Splash GPU | 25.111 ms | 24.846 ms | 1.06% |
| Splash extension | 3.198 ms | 2.979 ms | 6.86% |

The extension result is consistent across the final arms: controls 3.185 and
3.211 ms; candidates 2.978 and 2.980 ms. Wall timing varies substantially across
the broader session (an earlier control's splash mean was 27.062 ms), so the
7.12% wall mean is not a reliable standalone frame-time guarantee. Measurements
exclude rendering and statistics readbacks. The 25% whole-run frame-time target
remains unmet.

All final per-frame quality metrics, mixed ownership counts and allocation
sizes match. The selected source is enabled in the normal UI/production
Dynamic path, without a user setting. No remote deployment is claimed.

[spatial-phase3.json](uniform-compiled-topology-2026-10-06/spatial-phase3.json)
records source hashes, controls, development variants, aggregate demand and
validation. Per-frame timing/quality evidence is in
[spatial-phase3-frames.jsonl](uniform-compiled-topology-2026-10-06/spatial-phase3-frames.jsonl);
per-frame demand distributions are in
[spatial-demand-frames.jsonl](uniform-compiled-topology-2026-10-06/spatial-demand-frames.jsonl).
[Archived prototype patches](uniform-compiled-topology-2026-10-06/rejected-spatial-variants/README.md)
preserve the withdrawn implementations against the checkpoint; measured source
patches were verified against the captured hashes.

Types, unit tests (892 passed, 82 skipped) and the production build pass.
The full serial Dawn gate finishes at **58/61 files passed**. The three failing
files reproduce the earlier baseline failures: `svo-raster-ao` (channel 4630,
14238 versus 14240), `uniform-coarse-solid-rest` (seam-floor/toggle-floor drift
and mixed pressure convergence), and `uniform-pressure-local-visit` (fine
reference must execute coarse visits). Those failures were previously reproduced
on `fe587ebb` plus the then-current pressure edits; this run matches those
records. The existing coarse-rest-seam-open TODO is not an additional failure.
No tests or thresholds were weakened. The repository clean gate remains unmet.
