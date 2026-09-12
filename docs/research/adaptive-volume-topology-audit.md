# Adaptive-volume topology audit: mini-dam 32

## Scope and constraints

This code was inspected on 2026-09-13. The audit covers the current `adaptive-volume` Sparse CM12 implementation, with the `minimal-power-dam-break-32` geometry as the concrete scale model. It covers topology construction, resident data access, the complete frame stage graph, candidate publication, generation replacement, generation transfer, and generation-specific shader/pipeline work. It does not evaluate the pressure algorithm or geometric transport numerics except where they consume or reconstruct topology; those are separate research tracks.

The working tree already contained extensive uncommitted Sparse CM12 changes. This audit describes the files as they exist in that tree and does not treat old documentation as current authority. In particular, the incremental proposals in `docs/sparse-cm12-compiled-topology-handoff.md` are not carried forward. The architecture proposed here always recompiles the complete accepted topology and every topology-derived execution view after an accepted topology change. There is no dirty-row repair, delta replay, or partial schedule patching in the target design.

No browser, Dawn, or GPU process was launched for this audit. Chrome was already running, so the repository's exclusive Dawn gate was not started. The mini32 census below used the production CPU resource-recorder path documented as CPU-only at `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6349-6389`. Historical Dawn receipts are explicitly stale evidence for this audit.

## Executive finding

The current implementation does not have one accepted compiled topology. It has an authoritative object grid, an all-rung packed template catalogue, a mutable atomic topology arena, independent IBO/TEI/face-address/pressure images, a duplicated read-only pressure template, capacity-sized state planes, and accepted/shadow/delta worklists. The same topology is repeatedly rediscovered or republished by several frame stages.

For the current mini32 seed, the input composite grid contains 5,755 cells and 16,920 rows across all 64 authored leaves. The runtime accepts 4,887 cells and 15,006 rows across 36 active leaves, while the resident is sized for 244,288 cells and 845,040 rows. That is 50.0 times the accepted cell count and 56.3 times the accepted row count. The resource recipe allocates about 263.25 MiB before any live GPU execution. The principal cause is architectural: 37,440 cells and 146,928 rows are retained in the all-rung host catalogue, then another 404 B8 topology pages are reserved and every dynamic field plane is sized to the resulting capacity.

The highest-leverage change is a generation compiler that consumes exactly one accepted atlas and emits one immutable, accepted-only `CompiledTopologyGeneration`. Every stage reads a view from that generation. A topology change builds the complete next generation off the frame path, conservatively transfers fields, then swaps the whole generation at one publication boundary. Shader ABI must move generation counts and offsets into a small header/uniform so the same B8/P8 pipelines can be reused for all generations.

## Current mini32 CPU census

The census used the current adaptive-volume normalized defaults: B8/P8, coarse-first, one authored surface ring, and the current initial coarsening policy. The defaults are defined at `lib/methods/adaptive-volume/method.ts:285-324`; B8/P8 is also a hard resident requirement at `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:4281-4312`. The scene geometry is the 32-cubed, 0.025 m mini dam at `lib/core/scenes.ts:969-979` and `lib/core/scenes.ts:2613-2622`.

The CPU path built the actual composite grid, ran `createConfigured` through the fake GPU-shaped recorder, waited for the recorded simulation pipeline family, and inspected buffer descriptors and resident layouts. The recorder emits resource commands without a live device (`lib/methods/adaptive-volume/sparse-cm12-resource-recipe.ts:3-21`, `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6349-6389`). Counts and byte totals below are therefore allocation/capacity evidence, not runtime or compiler timing. The exact receipt is `docs/research/adaptive-volume-mini32-topology-census.json`; reproduce it with `node --import tsx docs/research/adaptive-volume-mini32-topology-census.ts`.

| Item | Current mini32 value |
|---|---:|
| Authored atlas leaves | 64 |
| Initially active leaves | 36 |
| Input atlas resolution histogram | B1: 11, B2: 22, B4: 23, B8: 8 |
| Active-leaf resolution histogram | B1: 7, B2: 10, B4: 11, B8: 8 |
| Input composite cells | 5,755 |
| Input composite rows | 16,920 |
| Input composite row terms | 34,530 |
| Input composite row kinds | 14,328 intra; 806 brick-face; 619 mixed; 1,167 sparse-air |
| Runtime accepted cell worklist | 4,887 |
| Runtime accepted row worklist | 15,006 |
| All-rung template cells | 37,440 |
| All-rung template rows | 146,928 |
| Reserved dynamic topology pages | 404 |
| World leaf capacity | 468 |
| Physical cell capacity | 244,288 |
| Physical row capacity | 845,040 |
| Cell capacity / accepted cells | 50.0x |
| Row capacity / accepted rows | 56.3x |
| TEI packet capacity | 29,952 |
| TEI spatial-tile capacity | 3,744 |
| Fine pressure edges | 337,440 |
| CPU resource-recipe operations | 602 |
| Total resident allocations | 276,033,788 bytes (263.25 MiB) |

Largest allocations:

| Buffer | Bytes | MiB |
|---|---:|---:|
| Resident state | 171,540,400 | 163.59 |
| Atomic topology arena | 61,636,416 | 58.78 |
| Candidate state | 15,029,888 | 14.33 |
| Conditioning | 10,749,052 | 10.25 |
| Read-only pressure topology duplicate | 4,516,892 | 4.31 |
| Effective velocity | 3,908,608 | 3.73 |
| Activity/planning | 3,187,756 | 3.04 |
| Pressure aggregate/execution | 2,363,136 | 2.25 |
| Fine presentation samples | 1,916,928 | 1.83 |
| TEI2 | 1,108,576 | 1.06 |

The capacity equations are explicit: 512 cells and 1,728 rows per B8 page are added to the host template counts at `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:4423-4429`, and `residentStateLayout` receives those physical capacities at `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:4455-4472`. The page budget is based on twelve non-wet pages per initially wet brick, bounded by a default ceiling, at `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:4355-4378`. This explains why accepted sparsity does not translate into field-plane sparsity.

An exact-sized design does not imply that every byte falls by 50-56x: fixed headers, presentation payloads, pressure scratch, transfer overlap metadata, and double-buffering do not scale uniformly with accepted cell/row counts. It does establish a hard direction. The 171.5 MB (163.59 MiB) state plane and much of the 61.6 MB (58.78 MiB) topology arena are capacity-proportional, so accepted-only compilation should remove tens to hundreds of MiB on this scene. The implementation should publish a byte map from the new compiler and gate measured totals rather than claim a ratio-linear final number.

## Current topology authorities and their duplicated derivations

### Composite object grid

`SparseAtlasCompositeGrid` is documented as the topology epoch identity and contains arrays of rich cell and row objects plus maps (`lib/methods/adaptive-volume/sparse-atlas-composite-projection.ts:83-92`). `buildSparseAtlasCompositeGrid` is the sole authoritative G-row builder (`lib/methods/adaptive-volume/sparse-atlas-composite-projection.ts:366-371`). It sorts bricks if required, materializes every cell and its geometric arrays in nested loops (`:375-454`), then materializes row objects and per-row term objects (`:456-522`). Intra-brick rows are enumerated cell-by-cell (`:524-586`); boundary and mixed-seam construction continue through the rest of the builder. A reusable object pool exists (`:99-124`), but the result is still a pointer-rich JavaScript graph that later compilers repeatedly traverse.

The grid correctly excludes solid geometry from static topology (`lib/methods/adaptive-volume/sparse-atlas-composite-projection.ts:13-15`). That distinction must remain in the new ABI: solid open fractions and moving-boundary state are dynamic overlays, while cell/face incidence and static geometric measures belong to the generation.

### All-rung SCMT template catalogue

The current packed format stores an eight-word cell record, nine structure-of-arrays row planes, variable row terms, incidence, row ownership, candidate face data, and pressure edges (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:1215-1301`). `packResidentTopologyTemplates` explicitly promises runtime rung switching without a host rebuild (`:1818-1824`), which is the opposite of the requested architecture.

Building this catalogue performs several full derivations:

- It preserves accepted identifiers and creates string/map based row keys (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:1928-1987`).
- It repeatedly calls the complete composite grid builder for rung variants, including 18 alternating 2:1 builds (`:2021-2071`).
- It scans all cells in six directions to synthesize sparse-air fallbacks (`:2074-2145`).
- It sorts row ownership, builds incidence counts/CSR, and independently builds pressure-edge counts/CSR (`:2147-2175`).
- It later builds incidence records again (`:2239-2245`), then walks row ownership and terms again for pressure edges (`:2263-2273`).
- Candidate cell and row worklists are produced with full-catalogue `flatMap` scans (`:2282-2289`).

For mini32 this turns 5,755 accepted cells and 16,920 accepted rows into 37,440 template cells and 146,928 template rows before page capacity is added.

### AEI, IBO, TEI, face address, pressure, and host-incidence views

`compileSparseCM12FactoredAEIPackedTemplate` reads the SCMT image yet again. Its canonical pass iterates every leaf/rung cell, validates it, filters owner intervals by axis, and hashes rows, terms, and geometry (`lib/methods/adaptive-volume/sparse-cm12-factored-aei-packed-template.ts:117-213`). Its patch compiler then visits every canonical item in six directions, collects rows in sets, rescans terms, computes mappings and weights, and hashes the result (`:216-370`).

Resident construction independently builds stable neighbors, IBO catalogs/ref lookup, geometry neighbors, and semantic authority at `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:5031-5132`. It then assembles the IBO image (`:5108-5119`), compiles a separate face-address program from that IBO (`:5191-5199`), and copies immutable host incidence beside the mutable arena (`:5213-5235`). TEI is constructed separately at `:4740-4752`. Pressure topology and its execution image are separately compacted from template data at `:4755-4798`.

The pressure path even keeps a separate immutable topology buffer because SpMV wants a read-only binding while the main topology arena is mutable atomic storage (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:3521-3523`). The mini32 copy is 4.5 MiB. This is a direct symptom of putting immutable cell/row topology and mutable publication state in one atomic arena.

The older `SparseCM12HotTopology` demonstrates useful packing ideas: one header covers cells, rows, variable terms, incidence CSR, directed-edge CSR, and requirement metadata (`lib/methods/adaptive-volume/sparse-cm12-hot-topology.ts:16-101`), and its builder counts and packs those sections in a few explicit passes (`:177-355`). It is currently used by the `advance-slice` runtime authority, where it is remapped to stable leaf slots (`lib/methods/adaptive-volume/advance-slice/slice-runtime-authority.ts:339-396`), not by the production resident. It should not be revived as-is: it lacks the production accepted-only schedules, WDR/dynamic-world rules, pressure membership overlays, and complete stage views, and its builder immediately performs another exhaustive validation pass (`lib/methods/adaptive-volume/sparse-cm12-hot-topology.ts:353-355`, `:432-518`). It is evidence that one contiguous generation image is feasible, not an implementation ready for adoption.

`SparseCM12TopologyGenerationStore` is another incomplete abstraction. It owns only a topology copy plus a membership buffer and explicitly says all other consumers must be prepared separately (`lib/methods/adaptive-volume/sparse-cm12-topology-generation-store.ts:7-15`). Its membership builder uses sets and rescans every selected row term (`:35-72`), and staging clones the complete supplied topology words (`:123-190`). No production resident use was found. The target generation owner should subsume its sound lease/reclamation concept (`:206-241`) while owning every compiled view and every field binding needed for an atomic swap.

## Runtime topology access

The resident row-access library documents the central ABI cost: production shaders use atomic `ta`/`taf` access because the arena is writable, while read-only lenses use a separate path (`lib/methods/adaptive-volume/sparse-cm12-row-access.wgsl.ts:1-16`, `:32-42`). Host and dynamic cells then diverge. Host cells load an eight-word record through atomics, while dynamic-page cells derive addresses and geometry arithmetically (`:69-137`). Row access also branches between host and dynamic representations and maps nine row planes (`:141-201`). Incidence traversal branches again: host cells load CSR offsets and records, while dynamic pages calculate arithmetic incidence with override checks (`:202-257`).

The library already contains one useful local optimization: packed row-term range access avoids separate loads (`lib/methods/adaptive-volume/sparse-cm12-row-access.wgsl.ts:292-305`). A shared immutable generation extends that principle across the whole ABI: one representation and one accepted dense identifier domain eliminate host/dynamic branches, atomic loads for static topology, accepted membership indirection, and pressure's duplicate read-only copy.

The target does not require one physical GPU buffer. WebGPU binding-size limits and access classes can justify a small set of immutable storage buffers. “One compiled topology” means one owner, one generation header/hash, one ID space, one compiler invocation, and views generated from the same canonical traversal. Mutable fields and per-frame masks must be separate buffers.

## Every production stage: topology work and data access

The canonical fifteen-stage order is defined at `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:365-382`; its substage contract exposes the amount of planning/publication work at `:403-473`. Accepted cells/rows, shadow lists, topology deltas, leaves, frontier neighbors, transport packets, and pressure cells each use separate direct or indirect dispatch helpers (`:6529-6608`). Every stage closes its compute pass before the failure gate (`:6665-6688`), so extra publication seams also add command/pass overhead.

| Stage | Current topology/data work | Compiled-generation opportunity |
|---|---|---|
| Transport velocity extension | Publishes frame control, copies indirect arguments, seeds/refreshes/extends accepted cells, invokes a topology edit transaction, compiles the accepted VEX schedule, initializes packets, and performs eight packet sweeps (`webgpu-sparse-cm12-resident.ts:6706-6772`). | Compile the static extension graph, level/order, packet-to-row/cell spans, and indirect counts once per full generation. Per frame only seed dynamic masks/values and execute required levels. Replace the legacy in-place pre-transport transaction; if support admission is required for imminent transport, the dependent stage waits for and swaps a complete rebuilt generation. |
| Face preparation | Seeds the destination a second time; clears and republishes support over brick/page capacity, then prepares accepted rows (`:6773-6783`). | Store accepted face rows and static support ownership in compact row tiles. Clear/publish only dynamic support masks for accepted tiles; fuse with the first consumer when legal. |
| Body forces | Applies accepted-row force, initializes accepted cells, and for inflow repeats row connection plus cell compression 32 times before reducing source state (`:6785-6801`). | Static row/cell ownership and source-neighborhood edges come from the generation. The iterative source algorithm is dynamic, but it should consume compact source/frontier lists rather than reconstruct connection addressing. |
| Pressure topology | Refreshes and extends the geometric interface again, opens many topology/membership epochs, runs dirty canonical repair, republishes frozen cell IDs/membership/coefficients, publishes PEI, commits PTR, and prepares pressure (`:6803-6907`). Aggregate/hierarchy substages are already empty (`:6875-6883`). | Compile static accepted pressure graph, row operands, cell/row tiles, hierarchy connectivity, and deterministic reduction order. Per frame classify liquid/free-surface membership and evaluate dynamic theta/open-fraction coefficients into compact arrays. Do not republish static IDs or reconstruct row topology. |
| Pressure RHS | Initializes PCG and Jacobi and performs several reductions/gates (`:6909-6927`). | Consume generation pressure-cell/row tiles directly. This stage remains dynamic numerical work. |
| Pressure solve | Each iteration updates state, applies SpMV, and reduces; every eighth iteration adds true-residual/recovery/Jacobi work and buffer copies (`:6928-6959`). | Reuse immutable read-only topology without the pressure template duplicate. Numerical/reduction redesign belongs to the numerics audit. |
| Velocity projection | Projects compiled interior/seam/sparse-air addresses, then accepted dynamic rows and collocates cells; it invokes the topology transaction a second time, recompiles projected VEX, and performs another eight sweeps (`:6960-7032`). | Interior/seam/sparse-air row classes and projection address packets are one compiled face program. Projected VEX uses the same generation graph with a dynamic receiver mask. Replace the legacy in-place post-projection transaction; newly required projected support triggers a complete generation rebuild and swap before any stage that consumes that support. |
| Conservative transport | Compiles geometric subfaces and cell-face incidence every frame, then gathers accepted cell/row envelopes (`:7295-7305`). The direct path can encode 512 packet iterations (`:7325-7358`). | Compile subface records, per-cell face ranges, packet order, and transport tile lists once. The dynamic envelope remains, but topology compilation and accepted-ID lookups leave the frame path. |
| Tracer advection | Optionally seeds and advances a fixed tracer lattice (`:7036-7046`). | Little topology compilation opportunity unless tracer ownership lookup is presently WDR/arena based; use the generation owner directory for that lookup. |
| Scalar publication | Builds final scalar masks over leaf capacity, publishes frame scalar output, then refreshes and extends the interface for a third topology consumer (`:7047-7066`). | Keep one authoritative interface refresh result per scalar epoch and share it with pressure, publication, and planning. Compile scalar packet/carrier-to-leaf mappings once. |
| Activity measurement | Scans leaf capacity for scalar/topology masks, measures brick capacity, ages history, and optionally scans the world directory and every page (`:7067-7094`). | Use accepted leaf/cell tiles and compact dynamic event outputs. Activity remains dynamic input to the next full topology request, but it must not mutate accepted topology in place. |
| Resolution planning | Classifies full leaf capacity, plans bricks/pages/faces, performs three B8 grading passes, validates/certifies, then builds shadow leaf/cell/row worklists and copies several indirect blocks (`:7096-7154`). Production's QA leader compaction flag defaults false at `:4281-4305`, so classification dispatch is not compacted. | Planning emits a desired atlas descriptor only. It does not build GPU shadow topology. CPU/worker full compilation validates 2:1 closure once and produces the complete next generation. |
| Candidate transfer | Transfers delta cells/faces, validates shadow faces, compiles and validates IBO deltas, compiles shadow TEI, authorizes effects, publishes PTR/deltas/pages/faces/selector, and replays retired TEI/IBO slots (`:7155-7230`). | Replace the entire stage with asynchronous whole-generation preparation plus the existing conservative old-to-new field transfer. Commit is a generation/bind-group swap after validation. There are no delta, shadow, replay, or retired-slot kernels. |
| Brick retirement | Reopens post-topology activity masks over leaf capacity (`:7233-7245`). | Retirement is represented in the desired next atlas. Old buffers retire as a leased generation after queue completion. |
| Presentation publication | Allocates and sorts pages, executes the frame plan, retires and compacts pages, commits control, then copies more indirect state (`:7247-7287`). | Compile topology-to-presentation source mapping and accepted page order. Dynamic scalar publication remains; topology page allocation/retirement moves to generation creation/destruction. Detailed renderer orchestration belongs to the presentation audit. |

The same long shadow/delta transaction also exists in the live-edit path. It performs planning/grading/certification, builds shadow worklists, transfers cells/faces, compiles IBO and TEI deltas, authorizes and publishes, connects pages, and replays retired slots at `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:8169-8255`. This duplicates the ordinary frame path and makes topology semantics depend on which entry point initiated the change. A full-generation compiler gives frame adaptivity, injection, terrain, and live edits one path.

## Replacement generations and field transfer

There is already a whole-resident replacement path, but it rebuilds far more than topology. `createReplacement` sends the next atlas, active set, solid world, source geometry/IDs, large source resource descriptors, and settings to a worker (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:3605-3648`). The worker calls `recordPreparedGeneration`, which builds the complete composite grid and complete resident, waits for every simulation pipeline, and only then prepares transfer (`:6349-6389`). The realized result is a whole new `WebGPUSparseCM12Resident` object (`:3650-3667`). The local fallback explicitly retains all-rung candidate backing to avoid repeated rebuilds (`:3669-3674`), which is obsolete under the requested always-full-rebuild policy.

Commit discipline is sound and reusable. `prepareGenerationReplacement` prepares a complete next resident, checks the source generation is still current, rebuilds target solid capacity at the copied pose, encodes transfer and presentation, submits, validates conservation/health, and leaves only a short publication boundary (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:9079-9133`). Keep that prepare/validate/swap model, but make the prepared object a compact compiled topology plus exact-sized fields and reusable pipeline handles rather than a complete independently compiled solver.

The transfer compiler already implements the right whole-generation semantics. It builds dyadic cell and coplanar-face overlap indices without finest-grid expansion (`lib/methods/adaptive-volume/sparse-cm12-generation-transfer.ts:22-85`), compiles source contributions for every target cell and face (`:118-173`), and performs conservative device-to-device field transfer (`:207-218`). However, preparation reconstructs several items that the unified topology compiler already knows: target cell volumes, face areas/axes, row-term offsets/cells/weights, source/target geometry, overlap grouping, and sorted contribution ownership (`:220-289`). Fold transfer-map compilation into `compileTopologyTransition(oldGeneration, newGeneration)`, reusing both generation descriptors rather than walking the object grids again. The four transfer pipelines are currently WGSL-specialized with source/target counts and every metadata/state offset (`:297-370`, `:453-500`) and compiled anew at `:523-539`; move counts and offsets to a transition header so one transfer pipeline family is reusable.

The transfer execution is already bounded and easy to retain: four passes for source allocation, cells, faces, and audit, followed by a validation/conservation readback (`lib/methods/adaptive-volume/sparse-cm12-generation-transfer.ts:550-612`).

## Generation-specific WGSL and pipeline work

The resident shader source is generated with pressure, activity, membership, frame-plan, repair, VEX, IBO, topology-effects, scalar-mask, face-address, world-directory, solid, and state-layout offsets (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:5740-5805`). Separate pipeline constants include generation capacities such as brick, leaf, page, frontier, and directory workgroups (`:5701-5713`).

The cache key is the entire generated source (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6090-6103`). Any changed count, capacity, offset, optional image, or layout produces another pipeline family. The family contains hundreds of entry points (`:5859-6019`). Simulation compilation slices that source into bounded call-graph chunks and processes chunks sequentially (`:6112-6160`), then compiles presentation allocator variants separately (`:6181-6207`). Those are sensible mitigations for the present enormous family, but a replacement generation still records and realizes the construction commands. The CPU resource recipe serializes all resource creation operations and explicitly awaits every `createComputePipelineAsync` while realizing it (`lib/methods/adaptive-volume/sparse-cm12-resource-recipe.ts:175-218`). The worker transfers all array buffers in the recipe (`lib/methods/adaptive-volume/sparse-cm12-preparation-worker.ts:5-23`).

The target ABI should compile pipelines per algorithm/configuration, not per generation:

- Compile-time: B8/P8, workgroup sizes, enabled feature family, precision/numerical algorithm.
- Generation header/uniform: cell/row/term/leaf counts; section offsets; tile/packet counts; old/new transfer counts; optional-view flags.
- Generation bind group: immutable topology sections, dynamic fields, per-frame masks, indirect dispatch arguments.
- Transition bind group: old generation fields/topology descriptors, new generation fields/topology descriptors, overlap records, audit scratch.

This converts a topology update into CPU compilation of arrays, buffer allocation/upload, bind-group creation, and field transfer. It must not generate new WGSL or compile new simulation pipelines. A device-level pipeline cache can then be keyed by an explicit ABI version plus feature bits instead of the whole generation-specialized source.

## Proposed `CompiledTopologyGeneration`

One full compiler invocation should consume the accepted atlas, the authoritative G-row builder output (or a more direct equivalent), the solid-independent topology policy, and presentation resolution. It should emit the following immutable views in one dense accepted ID domain:

1. **Generation header and certificate.** ABI version, generation number, topology hash, section offsets/sizes, exact counts, and hashes for each derived view. Every view certifies the same canonical source hash.
2. **Leaf and owner directory.** Dense accepted leaf descriptors, signed logical coordinate lookup, rung/resolution, first cell/row/tile, and presentation page/source mapping. Dynamic-grown leaves are ordinary accepted leaves in the next full generation; shaders no longer distinguish host from page topology.
3. **Cells.** Dense accepted cell descriptors with leaf/local coordinates, static volume/width/center or compact arithmetic descriptors, and stable geometric identity used only by old-to-new transfer.
4. **Rows and terms.** Dense accepted rows in canonical order with kind/axis/static area/distance, packed term range, and exact operand order. Uniform interior rows may use an implicit block descriptor, while all seam/sparse-air/exception rows remain explicit. Any implicit encoding must reproduce the authoritative row operands and floating-point evaluation order.
5. **Incidence and directed neighbors.** Accepted-only cell-to-row CSR and neighbor/edge CSR, built in the same row traversal. Pressure, projection, transport, VEX, diagnostics, and transfer reuse them.
6. **Dispatch tiles and accepted manifests.** Dense cell, row, leaf, pressure-static, and packet tiles plus indirect argument seeds. No separate accepted membership worklists are needed when IDs are dense.
7. **Face/projection program.** Interior, seam, mixed, and sparse-air row classes and address packets derived directly from the canonical rows. This replaces independent IBO, face-address, and semantic-neighbor derivations.
8. **Transport program.** TEI packets/spatial tiles, cell-face ranges, subface records, and deterministic packet order. This removes per-frame `compileGeometricVolumeSubfaces` and `compileGeometricVolumeCellFaces`.
9. **Velocity-extension program.** Static graph, traversal levels, and packet ranges shared by pre-transport and projected extension. Per-frame masks select active seeds/receivers.
10. **Pressure-static program.** Accepted cell/row IDs, static graph, hierarchy edges, deterministic SpMV/reduction ordering, and exact static geometric operands. Dynamic liquid membership, ghost-fluid theta, solid-open fractions, and moving-boundary terms live in per-frame arrays.
11. **Presentation source program.** Accepted leaf/page order and sample-to-cell mapping for the renderer's scalar publication.
12. **Construction diagnostics.** Byte map, count map, maximum fan-outs, validation certificate, and provenance. Validation runs once before upload, not as independent validators in each image builder.

Separate mutable buffers should contain density/gamma parity banks, velocity/face parity banks, pressure/Krylov vectors, geometric-interface values, liquid membership/theta, activity/history, source ledger, tracers, presentation samples, and failure receipts. Static SolidWorld aperture geometry may be compiled as an immutable view keyed by the SolidWorld version. Moving-body apertures and any edited solid version remain overlays until a complete generation is rebuilt for that solid version. Static topology buffers use `read-only-storage`; only indirect argument and dynamic buffers need writable/atomic access.

## Full-rebuild lifecycle

1. A frame measures activity and planning intent against generation N. Planning emits a complete desired atlas descriptor and reasons, not row/cell deltas. A request needed by an imminent transport, projection, source, or edit is a dependency barrier: the consumer waits for the complete admitted generation rather than running against missing support.
2. A worker builds and validates the entire accepted composite topology for generation N+1, then emits every compiled view above in one invocation. There is no all-rung catalogue and no spare dynamic topology page slab in the generation image.
3. The transition compiler compares stable geometric keys from N and N+1 and emits full cell/face overlap maps. It reuses already compiled geometry/row descriptors.
4. The advancing worker allocates exact-sized N+1 static and dynamic buffers, uploads the immutable views, binds the reusable pipeline family, initializes dynamic masks, and transfers fields on the GPU.
5. Validation checks topology certificate, transfer conservation, field finiteness/capacity, pressure/static-view consistency, and initial presentation coverage.
6. At a short queue-safe boundary, the host swaps the current generation object/bind groups. N remains leased until all submitted commands complete, then every N buffer is destroyed together.
7. If generation N changes before N+1 is ready, discard N+1 and compile the newest complete desired atlas. Do not attempt to patch or rebase it.

This lifecycle intentionally accepts CPU rebuild cost at topology epochs to minimize every-frame GPU work and memory traffic. It also removes two semantic systems: in-place backed rerungs and replacement-only unbacked changes. All accepted changes use the same full compiler and transfer path.

## Work eliminated or coalesced

- Delete all-rung and alternating-seam template enumeration; build only the accepted topology.
- Delete capacity-sized dynamic page topology/state slabs; allocate the next exact accepted generation.
- Delete accepted/shadow selector, topology delta, IBO delta, TEI replay, candidate state, and retired slot repair.
- Coalesce cell/row/term/incidence/neighbor/pressure-edge construction into one canonical traversal.
- Coalesce IBO geometry/semantic authority, face-address compilation, projection row classes, and row ownership into one face program.
- Coalesce TEI, transport subfaces, and cell-face incidence into one transport program.
- Compile one VEX graph and reuse it for both extension sites.
- Refresh the geometric interface once per scalar epoch and share it among pressure, scalar publication, planning, and presentation consumers.
- Separate immutable topology from mutable state, removing atomic reads for static records and the read-only pressure duplicate.
- Move generation counts/offsets out of WGSL source, allowing all generations to share compiled pipelines.
- Reuse compiled generation geometry when building transfer overlaps; stop repacking target row terms and geometry into transition metadata.
- Group frame kernels by real data dependency. Multiple singleton begin/seal/publication kernels that only initialize adjacent header words should be one control kernel; adjacent compact tile passes with no intervening consumer should share a compute pass and, where semantics permit, one kernel.

## Complexity and memory model to gate

Let `L` be accepted leaves, `C` accepted cells, `R` accepted rows, `T` row terms, `I` incidence records, `E` pressure/directed edges, and `P` transport/VEX packets. The compiler should be `O(L + C + R + T + I + E + P)` time and space, apart from owner-directory hashing. There must be no multiplier for four rungs, alternating rung pairs, page capacity, or repeated consumers.

The build should use count/prefix/fill passes over typed arrays:

- Pass A: assign dense leaf/cell IDs and count local rows, seam ports, terms, incidence, and derived packets.
- Prefix: compute exact section offsets and allocate once.
- Pass B: emit canonical rows/terms and simultaneously fill incidence/neighbor/face/pressure/transport references.
- Pass C: stable-sort only where canonical determinism requires it, resolve cross-references, hash, and validate.

The resource budget must account for the publication peak, not only the accepted resident: generation N static+fields, generation N+1 static+fields, transition metadata/scratch, and any shared presentation double buffer. Budget admission happens before allocation. Report accepted bytes, staged bytes, transition bytes, and peak bytes separately.

Mini32 acceptance targets for the new architecture should include:

- `physicalCellCapacity == acceptedCellCount` and `physicalRowCapacity == acceptedRowCount`, except explicitly documented alignment padding.
- No all-rung cell/row catalogue and no topology page pool in the accepted generation.
- Zero per-frame dispatches named candidate/shadow/delta/replay or topology schedule compilation.
- Zero generation-dependent shader modules/pipelines after initial device warm-up.
- No static-topology atomic loads and no pressure topology duplicate.
- One interface refresh per scalar epoch.
- A compiler count/byte certificate whose derived views all share the same topology hash.

## Risks and validation

The full rebuild changes storage and publication, not the mathematical row authority. Preserve the row order, term order, and arithmetic operands emitted by `buildSparseAtlasCompositeGrid`; the opening contract ties G, divergence, and pressure identities to those exact gradient rows (`lib/methods/adaptive-volume/sparse-atlas-composite-projection.ts:3-15`). Do not precombine coefficients if that changes floating-point evaluation order.

Dynamic facts must not be accidentally frozen into the generation. Solid geometry is explicitly external to the static topology. Liquid membership, ghost-fluid theta, moving-solid capacity, source state, activity, and scalar/face parity are per-frame data. The generation may compile where these facts are evaluated and the static operands they use, but not their values.

The new owner directory needs signed sparse-world coordinates and exact failure behavior. Current production ownership is WDR1, while the dense logical owner upload exists only for QA (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:4379-4408`). Full generation rebuild does not justify falling back to a complete logical-domain array.

Field transfer must remain conservative and must validate incomplete/overlapping coverage. The existing transfer compiler rejects target cells with overlapping source coverage and admits missing coverage only inside explicit new-air regions (`lib/methods/adaptive-volume/sparse-cm12-generation-transfer.ts:130-170`). Keep that behavior and the compensated volume audit (`:474-499`, `:575-610`).

Because a browser is currently using WebGPU, this research did not run the repository's canonical Dawn regression. Once an implementation exists and no browser/other Dawn process is active, the required post-refactor gate is `npm run test:dawn:sparse-cm12` per `AGENTS.md`. Before that gate, add CPU compiler equivalence tests that compare every generated row/term/incidence/edge/program view against the authoritative grid for accepted, 2:1 mixed-seam, sparse-air, dynamic-world, and live-edit topologies. Those tests validate whole generations; they must not introduce a partial update mode.
