# Adaptive-volume performance plan

The recommended design is one fully compiled representation of the accepted simulation topology, shared by every numerical and presentation stage. Rebuild the complete representation whenever topology changes, transfer the live fields, validate it, and publish it atomically. Keep the shader programs stable across generations. Recompute changing numerical state over compact accepted cells and faces, with explicit global dependency boundaries.

This is a replacement of the present all-rung, capacity-shaped execution architecture. A collection of additional dirty lists or locally patched caches would retain its principal costs. The existing implementation already has several useful compiled representations, but they neither describe the same domain nor serve all consumers. Physical volume faces are still compiled per frame; generic row access remains inside pressure iterations; many arrays and scans are sized for a much larger capacity than the accepted scene.

## Evidence and scope

The source audit follows the production `adaptive-volume` GPU path, including its current uncommitted changes. The inspected base commit is `2b32bd38d66951d9e22d0b96080b5a2ce5b84b3d`; the captured source-content fingerprint is `1c15404b3310450f9570dea6d730eba5319f7eb2ee50a6d6c78a108c3eedefc8` over 1,276 source inputs. Separate detailed audits cover [topology construction and admission](/Users/petersuggate/code/me/fluid/docs/research/adaptive-volume-topology-audit.md), [velocity and pressure](/Users/petersuggate/code/me/fluid/docs/research/adaptive-volume-numerics-audit.md), [geometric volume transport](/Users/petersuggate/code/me/fluid/docs/research/adaptive-volume-transport-audit.md), and [orchestration, activity, and presentation](/Users/petersuggate/code/me/fluid/docs/research/adaptive-volume-orchestration-presentation-audit.md).

The reference is `minimal-power-dam-break-32`: a 32³ finest lattice in a 0.8 m closed free-slip tank, 0.025 m finest cells, no moving rigid bodies or inflow, and a 1/30 s physical step. Balanced production defaults select B8/P8, coarse-first adaptation, automatic macro span, RDF presentation, pressure tolerance 1e-6, and up to 128 pressure iterations. CM12 gamma diffusion and sharpening are off. A B8 domain has 4³ logical brick coordinates; accepted leaves can be mixed-rung, omitted, or macro leaves. It is not the eight B16 bricks described in the old handoff.[^1]

The present report contains current CPU construction measurements and code-derived findings, with older GPU timings clearly identified as historical. It contains no measured speedup for the proposed design. No fresh Dawn run was made while a browser was running, following the repository's exclusion rule. Production implementation and validation remain future work.

## What mini32 exposes

### Current construction footprint

The CPU-only production construction census gives the following initial scene shape. These are construction counts and recorded GPU buffer allocations, not hardware execution timings or an evolved scene census. The topology audit documents the construction path; the [CPU census receipt](/Users/petersuggate/code/me/fluid/docs/research/adaptive-volume-mini32-topology-census.json) and [reproducible census script](/Users/petersuggate/code/me/fluid/docs/research/adaptive-volume-mini32-topology-census.ts) separate the input grid, runtime accepted lists, and physical capacities.

| Quantity | Accepted initial topology | Backing capacity | Capacity / accepted |
| --- | ---: | ---: | ---: |
| Cells | 4,887 | 244,288 | 50.0× |
| Pressure/velocity rows | 15,006 | 845,040 | 56.3× |

The runtime accepted lists contain 36 active leaves (B1: 7, B2: 10, B4: 11, B8: 8). The broader CPU input grid contains 5,755 cells and 16,920 rows across all 64 authored leaves; those are not the runtime accepted counts. The all-rung catalog alone contains 37,440 cells and 146,928 rows. Page reservation expands the physical capacities further: 404 topology pages and 468 leaf slots. Recorded resident allocations total 276,033,788 bytes, about 263.25 MiB. The largest are state at 171,540,400 bytes and topology arena at 61,636,416 bytes, followed by candidate state at 15,029,888 bytes and conditioning at 10,749,052 bytes.

These ratios do not mean every kernel executes 50 times too much work: many use accepted indirect lists, and other kernels exit early. They do show that live data, scratch, metadata, uploads, and some frame traversals are designed around a much larger address space. Exact accepted storage is a primary design objective. Reserve allocation headroom independently of logical iteration bounds; a larger buffer need not imply more initialized records, scans, or shader specialization.

### Historical GPU costs

A production geometric mini32 RDF receipt recorded 24 samples after eight warmups. It used B8/P8 and a 1/30 s step, left topology unfrozen, but classified all measured frames as having no committed bricks. Its source fingerprint is `334d69e79d4d121196de5839a0e95106c7b1f6dfe1eef999072d6aa4b94d54d6`, different from the inspected source.[^2]

| Historical quantity | Median ms |
| --- | ---: |
| Complete-frame wall time | 174.3227 |
| Advance timestamp span | 167.1168 |
| Geometric transport latency | 105.9062 |
| Pressure solve | 20.9060 |
| Face preparation | 11.8620 |
| Presentation publication | 7.3400 |
| Velocity extension | 7.0124 |

Transport timing includes asynchronous continuation waits. Stage medians do not sum to the median frame. The final transport receipt records nine microsteps, 203 limiter passes, and 224 encoded packets; those counts describe its final frame, not all 24 samples. This evidence directs attention to repeated limiter passes and their execution schedule, but cannot quantify today's cost or the proposed savings.

The older mini32 “frame anatomy” and mass-transport receipts are unsuitable baselines: the numerical method, pressure tolerance, stage roster, and capacity layout changed. Likewise, a quiescent capture cannot price full topology rebuilding. The validation plan must measure both steady-topology frames and topology-update frames.

## Complete production-stage audit

The table follows actual host execution order. “Compile” means produce topology-dependent data once for the complete next generation. “Refresh” means recompute values from current fluid/geometry state. Detailed shader entry points and line references are in the four audits linked above.

| Stage | Current data access and repeated work | Shared compiled view and execution change |
| --- | --- | --- |
| Frame authority and velocity extension | Frame-control sealing and indirect snapshots; destination seed; interface refresh/extension; topology preflight; packet schedule begin/compile/seal; eight neighbour sweeps. Generic incidence/row walks remain at seams and wherever the global static-solid predicate excludes the interior path. | Compile accepted cell packets, neighbour endpoints, fixed weights, open-interior certificates, and local ownership once per generation. Refresh validity and velocities each frame. Keep eight propagation dependencies. Avoid dispatching topology schedule construction on an unchanged generation. |
| Face preparation / momentum advection | Seeds destination a second time, clears/publishes face-support cache at brick capacity, resolves trajectory and staggered sample support per accepted row. Source and trajectory fields have different semantics. | Compile accepted face descriptors, static support/overlap addressing, and interior sampling certificates. Preserve the contiguous support cache unless matched measurements justify replacement. Resolve moving departure positions through one shared sampler. |
| Body forces and sources | Another accepted-row pass; no-inflow mini32 still begins and initializes source state over accepted cells. | Fuse gravity with the face predictor's final store where operation order and bank writes match; retain inflow overrides. Omit repeated zero-source initialization after a known-zero state, with a full reset on source removal or generation replacement. |
| Pressure geometry, membership and coefficients | Refreshes interface again; builds and repairs multiple membership/coefficient authorities. Static voxel presence causes broad row invalidation. Generic incidence gathers compute row/cell predicates and diagonal operands. | Reuse interface values only for the identical input version. Fully refresh pressure membership and numeric coefficients over accepted cells/rows. Consume compiled signed incidence and static apertures; remove dirty-token and repair-tree machinery from the replacement path. |
| Pressure RHS / solver seed | Separate PCG initialization, Jacobi initialization, reductions, true-residual application, and pipelined-image initialization. Initial operator/precondition work overlaps. | Fuse diagonal and RHS preparation where they use the same finalized row coefficients; fuse cell-local seed products and partials. Keep neighbour-read and global reduction dependencies. Preserve a real initial residual and the same warm start. |
| Pressure iterations | Per-cell incidence → row terms → membership lookup in repeated operator applications; periodic recovery shaders are encoded even when their internal guard does no useful work. | Compile row opcodes, ordered term IDs/coefficients and cell incidence recipes. Refresh only numeric row weights per physical step. Consolidate conditional recovery work and use dedicated GPU-authored recovery dispatch arguments. Retain convergence rules and final true residual. |
| Projection and collocation | Accepted face-address program still denotes an all-rung superset with acceptance checks; row gradients and cell incidence are gathered again. Projected-velocity support may require a second topology preflight and extension. | Execute accepted-only faces and signed cell incidence. Share static seam flags and dynamic pressure facts with diagnostics. Rebuild once when a preflight actually changes topology; preserve any required extension after newly admitted support. |
| Geometric subfaces and transport setup | Every outer transport counts/emits physical subfaces, validates marginals, then counts/emits cell→subface CSR. IDs live in mutable float state slots. | Compile physical subfaces, row→subfaces, and ordered cell→subfaces once with the topology. Certify area and signed marginals before publication. Separate integer topology from evolving flux state. |
| Interface reconstruction / microstep flux setup | PLIC/RDF support repeatedly resolves owners/adjacency; geometry and capacities evolve per microstep. Pressure/preflight geometry can overlap the first transport reconstruction. | Compile fixed support IDs, overlap weights, LS geometry and extension reachability. Refresh normals, planes, capacities and swept fluxes for each required field version. Reuse microstep-zero values only when every input version matches. |
| Low-flux limiter | Each pass gathers the entire cell→subface neighbourhood, writes next factors, then runs another whole-cell factor-copy pass. Iteration count can be large even on small scenes. | Use compact compiled adjacency. Ping-pong limiter factor banks to remove the full-cell copy. Fuse static initialization and the first update where equivalent. Preserve global factor generations and the exact accepted-factor certificate. |
| FCT correction, validation, volume commit | Face/cell alternating gathers and bounds checks, followed by accepted volume publication; each packet includes control work and sometimes empty phase dispatches. | Share compact physical faces and cell-face recipes. Fuse owner-local producers and partial receipts where legal; keep face→cell→face synchronization and validate before commit. Remove control work through a coherent phase schedule, without reducing iterations or bypassing checks. |
| Marker advection | Optional trajectory sampling and position writes; no work when the view is off. | Share the arbitrary-point sampler and compiled owner directory. Low priority for default mini32. |
| Scalar publication | Final scalar-mask transaction, bank publication, then interface refresh/extension. | Publish scalar authority and compact frame facts once. Make this geometry output available to later consumers with an explicit version key. Remove obsolete masks only after consumers adopt full accepted work. |
| Activity and frontier census | Brick-capacity launch with dirty exits, contiguous local cells, then generic signed incidence/row-term gathers; curvature performs owner probes and virtual restrictions. Frontier support already has a shared 26-neighbour reduction. | Full accepted-leaf census over compiled signed incidence; compile curvature sampling support and neighbour directory. Fuse history updates into leaf-owned reduction. Keep neighbour-support reduction after census completion. |
| Resolution planning | Capacity/leader scans, policy classification, frontier activation, grading passes, support certification, and shadow-list construction. | Compile static region/neighbor/face support per topology or policy revision. Run complete policy evaluation over accepted leaves plus explicit candidate frontier. Keep grading to closure and independent candidate certificates. Produce one candidate for the full compiler. |
| Candidate transfer / topology publication | Transfers shadow fields/faces; constructs and validates IBO/TEI effects; authorizes, publishes, and replays retired images. Backing growth separately constructs a new resident/pipeline family. | Replace both topology mutation forms with one complete next-generation compiler and field-transfer transaction. No delta replay. Separate shader programs from generation data and reuse them across updates. |
| Post-topology activity mask / retirement | Marks changed leaves and republishes masks for next frame. Actual page/directory retirement also occurs in presentation. | Full next-generation execution views remove post-topology dirty repair. Preserve temporal history remapping and retiring presentation-page semantics. |
| Presentation and surface proofs | Page allocation/directory work, ten planning/packet dispatches, interface/RDF rebuild, page candidate writes/copies, verification, virtual coarsening proof, retirement and compaction. TEI staging and density/RDF caches already exist. | Compile page/sample support and proof restriction schedules. Publish all accepted pages in the baseline design; retain short coverage/fault receipts. Share geometry values only by exact input version. Fully rebuild page directory on membership change. Keep proofs camera-independent. |

Three registry entries—gamma diffusion, surface sharpening, and density-capacity repair—have no stage calls in the current geometric encoder. They contribute no production mini32 execution cost. Their legacy declarations, scratch dependencies, and compilation reachability should be audited for removal as part of the architecture replacement, rather than optimized as live kernels.[^3]

## Compiled topology design

### One authority, several access views

“Shared” should mean one canonical generation and one compiler, not one giant record loaded by every kernel. Different consumers need cell-major, row-major, and leaf-major views. Produce those views together from the same accepted graph, with cross-view identity checks. Small read-only structure-of-arrays tables are preferable to exposing the mutable construction arena through long accessor chains.

| Compiled product | Contents | Main consumers |
| --- | --- | --- |
| Generation header | Generation identity, counts, offsets, layout version, static-geometry/policy versions, validation status | Every binding and dispatch |
| Leaf directory | Signed origin, span, accepted rung, valid clipped extent, cell/row ranges, neighbour descriptors | Sampling, census, planning, presentation |
| Cell view | Leaf/local address, bounds/volume, static boundary classification, ordered incidence range | Pressure, transport, reconstruction, activity |
| Row view | Axis, location, structural area/distance, ordered signed terms, static aperture data, numerical summation opcode | Face prediction, forces, pressure, projection |
| Physical subface view | Exactly one shared face per physical patch, endpoints, parent row, signed orientation, bounds/area, row→face and cell→face ranges | Flux/CFL, limiter, FCT, remap certificates |
| Sampling support | Local neighbour IDs, uniform-interior certificates, fixed overlap donor/weight lists, corner/vertex connectivity | VEX, fixed reconstruction probes, RDF, activity, presentation |
| Execution lists | Dense accepted cells/rows, leaf/rung tiles, boundary/regular classes, VEX packet schedule, page list, indirect counts | All stages |
| Pressure structure | Row opcodes and signed incidence; aggregation/prolongation structure if used by the selected solver | Coefficient update and operator applications |
| Presentation structure | Page mapping, sample ownership/restriction schedules, proposed-rung proof support | Publication and adaptivity proofs |

Preserve exact ordered reductions initially. A sparse row can have a specialized two-, three-, or five-term summation tree; replacing it with an arbitrarily ordered edge sum can change symmetry and pressure results. A pressure cell operator is not automatically the same as a pairwise physical-face Laplacian. Share the underlying connectivity and geometry while retaining the operator's algebra.

Regular interior cells can use arithmetic neighbours and a compact stencil class. Irregular, clipped, mixed-rung, sparse-air, and solid-boundary cells use compiled recipes. The current global “has static voxels” branch should become local certified geometry. Compile the actual signed coordinate and clipped overlap geometry; nominal adjacent bricks are insufficient proof of a face.

There is a separate operator experiment worth measuring once the shared graph exists. Current cell-major SpMV recomputes a whole row gradient for each incident pressure cell: potentially twice for a two-term row and five times for a five-term seam. A row-major `G*x` pass could calculate each gradient once, followed by a cell-major transpose gather. It adds a dispatch and row scratch traffic, so it is not an unconditional win on mini32. Compare it with the direct compiled cell-major operator while preserving the exact row sum tree and cell accumulation order. [Operator audit](/Users/petersuggate/code/me/fluid/docs/research/adaptive-volume-numerics-audit.md).

Arbitrary advection departure points cannot be fully precomputed at topology update because velocity changes them. Compile the owner directory and local support descriptors, then calculate the dynamic position and interpolation weights at runtime. Fixed RDF vertices, cell-centred restriction probes and presentation lattice samples have stronger compile-time opportunities. Do not merge these cases under a claim that all sampling becomes one lookup.

### Values that must remain dynamic

The following are not topology: density/volume, velocity, pressure, liquid/air membership, free-surface theta, numeric pressure coefficients, current interface planes/RDF values, limiter factors, source amounts, activity history, and moving-solid apertures/capacity. Full topology rebuilding does not make any of these constant.

Use explicit version tuples for derived numerical values: `(topology generation, scalar version, velocity version where used, geometry pose/version, source/policy version, physical substep)`. A producer writes a complete next value plane, consumers read the completed plane, and publication flips its version only after validation. This avoids incremental dirty tracking while allowing exact reuse of an already-produced result within the same dependency graph.

Static SolidWorld edits invalidate static geometry and any support/operator view derived from it. Live refinement-region edits invalidate policy support; renderer layout changes invalidate presentation schedules. Treat connectivity changes as full topology rebuilds regardless of whether they came from ordinary adaptation, frontier admission, live edits, or backing replacement.

### Full rebuild transaction

1. Finish the preceding accepted physical state at a valid dependency boundary. Evaluate the full refinement/frontier policy and seal a complete candidate description.
2. Enumerate all candidate accepted leaves and cells in deterministic spatial order; create the sparse signed owner directory and physical clipped bounds.
3. Count rows, ordered row terms, physical subfaces and incidences. Prefix-sum sizes and emit compact arrays. This is a full pass over the candidate, with no dirty-region patching.
4. Emit every dependent view from those shared arrays: packets, face addresses, VEX support, sampling support, pressure structure, presentation/proof support, and indirect counts.
5. Certify exact accepted coverage, endpoint ownership, duplicate-free faces, positive geometric measures, row/subface signed marginals, boundary semantics, and version consistency.
6. Transfer extensive fluid quantities from old to new cells using physical overlap and the existing geometric transfer rules. Transfer velocity/pressure/history with their respective rules, preserving the preceding accepted state until validation succeeds.
7. Publish the new fields and all compiled views with one generation selector. Retain old buffers until simulation, presentation, readback, and transfer leases are released. On refusal, retain the preceding accepted generation and its physical clock.

The same transaction must handle in-frame frontier support. Current code performs preflights before transport and after projection; neither can be removed merely because end-frame adaptation exists. If a preflight changes support, stop the dependent stage, fully rebuild and transfer, and resume against the new generation. Record such rebuilds separately to expose multiple topology changes within one physical frame. Predicting adequate support earlier is a separate correctness-tested policy change.

Shader source and pipeline keys must depend on stable algorithm options and brick/tile specialization, not generation-specific array lengths or arena offsets. Put changing counts/offsets in headers/uniforms and runtime storage arrays. Allocation growth should bind new buffers to the same pipelines. Shader compilation on each topology update would defeat the full-rebuild design.

### Cost and memory model

Let `C`, `R`, `T`, `F`, `I`, `L`, and `S` denote accepted cells, rows, row terms, subfaces, incidences, leaves, and compiled sampling-support entries. A compact full compiler should target work proportional to `C + R + T + F + I + L + S`, plus sorting/prefix-scan and geometric intersection costs. It is not automatically O(surface), and full sampling support can be large. Count/prefix/emit is an acceptable full rebuild even if it visits records more than once.

For `K` frames between updates, compare `build_new/K + steady_new` against the complete old build and steady costs. When every frame changes topology, use `K = 1`; the design must still be measured there. Memory peak includes old topology/fields, complete candidate topology/fields, transfer scratch, and any retained presentation consumer. Steady-state memory alone is insufficient.

Illustratively, `8(C+1)` bytes for two-word cell ranges plus roughly `8F` bytes for two endpoint incidence entries is small beside the present capacity-shaped state. That is only the CSR slice; face geometry, sampling support, pressure structure and transfer workspace add to it. Publish an exact per-product byte ledger instead of extrapolating a 50–56× ratio into an overall memory or speedup claim.

## Coalescing priorities and boundaries

| Candidate | Why it removes work | Required boundary / condition |
| --- | --- | --- |
| Full topology compiler for subfaces, incidence, VEX, BFA and TEI | Stops independent accepted-topology discovery and repeated per-frame topology builds | All consumers must switch to the same fully validated generation |
| Limiter factor ping-pong | Removes one full-cell copy dispatch per limiter pass | Each update reads only the complete previous bank; certificate pins the exact accepted bank |
| Static limiter initialization + first update | Shares first cell-face traversal | Static-solid branch only until moving-solid equations are separately proven |
| Pressure diagonal + RHS | Shares cell incidence and finalized row numeric operands | All row coefficients and membership must already be complete |
| PCG seed products and local partials | Avoids repeated initial preconditioning/operator work | Keep global reductions and later neighbour reads separate |
| Row-major pressure gradient followed by cell transpose gather | Computes each row gradient once instead of once per incident cell | Benchmark added dispatch/scratch traffic against direct compiled SpMV; preserve algebra and summation order |
| Face prediction + gravity store | Removes a row traversal | Same floating operation order, dry-face semantics, source/destination bank writes and inflow behavior |
| Shared interface-value producer | Avoids repeated refresh/extend for unchanged input versions | Topology, scalar bank, pose and reconstruction policy must match exactly |
| Census + history output | Shares cell/leaf loads and reductions | Exclusive leaf ownership; preceding history remains an immutable input |
| Presentation geometry support + proof support | Removes owner/overlap rediscovery | Proposed-rung proof values are distinct unless their actual sampling query matches |
| Owner-local planning/packet metadata | Removes full-capacity clear/populate/repack chains | Retain coverage/fault provenance; global acceptance still follows all producers |
| Full directory reconstruction on membership change | Removes steady-frame compaction/bounds work | Retirement must still publish the required all-air state before releasing pages |

Do not fuse a limiter's neighbour update loop into one multi-workgroup kernel, a global reduction into its producer, or a face→cell→face FCT chain with only workgroup barriers. WGSL synchronization does not provide a grid-wide barrier.[^4] A dedicated indirect producer can share a compute pass with later work only when each dispatch's actual bindings satisfy resource usage rules; do not bind the indirect argument buffer for writable storage in its consumer.[^5]

Also avoid making each kernel enormous to save launches. Staged neighbourhoods and more fused live values consume registers and workgroup memory, potentially reducing occupancy. Measure the resulting kernel rather than assuming more caching is always faster. Apple explicitly identifies exhausted thread/threadgroup memory and memory access patterns as occupancy concerns.[^6]

## Implementation sequence and acceptance

These are reviewable slices of one architecture, not a plan to layer more incremental maintenance onto the current runtime.

### 1. Establish a reproducible baseline and exact execution inventory

Capture current source bytes and resolved defaults. Run isolated mini32 with natural adaptation across initial release, front propagation, reflection, and later relaxation; additionally use a frozen evolved topology to isolate steady execution. Separate unchanged-topology frames, in-resident topology changes, backing replacements, and multiple preflight rebuilds. Report initial construction separately.

Record accepted/capacity counts and bytes; dispatches and nonzero dispatches; pressure iterations; microsteps and limiter passes; continuation submissions/map latency; CPU encoding; GPU kernel time; frame latency; topology planning/build/upload/compile/transfer/admission; and peak retained generations. Compare traces at the same accepted physical time. Keep normal production instrumentation and heavily instrumented kernel profiles separate.

Existing stage-cost tooling can supply the first lane:

```bash
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/probe-sparse-cm12-stage-cost.ts \
  --scene=mini32 --production-defaults=1 --brick-fine=8 \
  --presentation-page=8 --warmup=8 --frames=24 \
  --final-qa=1 \
  --out=artifacts/adaptive-volume-performance-research/mini32-baseline.json
```

Create the output directory first. The short probe does not guarantee a topology update or cover late evolution; add a physical-time checkpoint/replay lane for that purpose. Retain all current tolerance and timing limits.

### 2. Build the accepted-generation compiler and stable pipeline ABI

Implement the canonical compact topology and its cross-view certificates, then construct the current numerical inputs from that graph. First preserve existing state IDs through a compatibility mapping if needed for comparison; the final path must use accepted-sized field planes. Use full rebuilds for every candidate. Separate static shader family keys from runtime counts/offsets.

Acceptance: exact coverage and geometry certificates; canonical mixed-ratio and clipped-domain cases; no new shader source or pipelines merely because generation counts/offsets changed; bounded old/candidate/retained memory; matching numerical inputs by physical identity. Deterministic IDs can change raw buffer hashes, so compare canonical physical order and field values as well as version/coverage receipts.

### 3. Cut over transport and reconstruction

Make geometric subfaces and cell-face CSR generation products. Move fixed reconstruction support into shared views. Change limiter factors to ping-pong banks and remove the corresponding commit scan. Remove duplicate seeding/history storage only after all generation-transfer and preflight consumers are accounted for. Reuse interface values by exact version identity.

For fused limiter initialization, the first pass must use an explicit constant-one factor rule or a previously initialized complete bank. It cannot initialize a cell's factor and immediately read neighbouring factors in the same multi-workgroup dispatch. Later passes read a complete prior bank and write the next bank; convergence publication must retain the exact factor generation that passed validation.

Acceptance: per-face signed fluxes, per-cell volume, microstep clocks, sources/outflow/roundoff, limiter termination and accepted-factor certificates. Preserve physical operation order for the representation-only changes. Use the translation, corner reconstruction, mixed-subface, and moving-solid controls in addition to mini32; unchanged volume alone cannot establish surface or momentum fidelity.

### 4. Cut over momentum, pressure, activity, and output

Move VEX and pressure operators to compiled accepted adjacency, replace global static-solid penalties with local certificates, and fully recompute pressure numeric membership/coefficients. Apply proven local pressure/force fusions. Replace activity dirty maintenance with a complete accepted-leaf census and retain policy history. Replace presentation's general dirty frame-plan pipeline with direct full accepted-page execution and compact receipts, using shared sampling support and versioned geometry.

Acceptance: identical pressure algebra, true residuals, projection fields, hydrostatic equilibrium, symmetry, adaptation decisions, and camera-independent surface proofs. Measure ordinary interior kernels and boundary kernels separately. If a changed arithmetic order is necessary, review it as a numerical change with dedicated fidelity tests, not as an incidental performance optimization.

### 5. Remove competing authorities and gate the complete system

Delete the now-unused all-rung runtime catalogs, topology delta/shadow replay chains, redundant membership/rank/dirty structures, legacy scratch, and generation-specialized shader construction from production reachability. Keep construction/reference code only where it serves explicit oracles. A new compiled image that leaves all the old structures active is not completion.

Run the unchanged full `npm run test:dawn:sparse-cm12` after the major implementation changes. Its mini32 ceiling is 40 ms and mini64 ceiling is 110 ms; do not raise them to pass. Use the longer ladder and analytic controls where the affected mechanisms require them. Run no Dawn process concurrently with a browser or another Dawn process.[^7]

Performance acceptance must include complete-frame p50/p95 and topology-update latency, not only steady GPU stage totals. Require repeated interleaved baseline/candidate runs on the same device and equivalent physical trajectories. Report known baseline failures separately; a historical passing receipt does not certify the current tree. No numerical tolerance, physical timestep, solver ceiling or topology policy should be relaxed as part of this program.

## Decision

Start with the accepted-generation compiler and stable shader ABI, then make transport its first major consumer. This directly addresses the measured capacity disparity and repeated physical-subface/adjacency construction, while enabling the limiter and pressure optimizations to operate on compact data. The high-value unit of reuse is a validated generation of connectivity plus explicitly versioned numerical planes. The number of tiny cache updates is not the target.

## Sources

[^1]: Repository source, [mini32 scene](/Users/petersuggate/code/me/fluid/lib/core/scenes.ts:970), [base scene](/Users/petersuggate/code/me/fluid/lib/core/scenes.ts:864), and [production defaults helper](/Users/petersuggate/code/me/fluid/lib/harness/sparse-cm12-dawn-defaults.ts:1), inspected 13 September 2026. Exact code evidence and construction measurements are expanded in the four companion audits.
[^2]: Repository measurement artifact, [mini32 RDF stage-cost receipt](/Users/petersuggate/code/me/fluid/artifacts/sparse-cm12-rdf-presentation-performance-final/mini32-rdf.json), historical source fingerprint `334d69e79d4d121196de5839a0e95106c7b1f6dfe1eef999072d6aa4b94d54d6`. Not a current-source benchmark.
[^3]: Repository source, [resident encoder](/Users/petersuggate/code/me/fluid/lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6706), [geometric transport encoding](/Users/petersuggate/code/me/fluid/lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:7295), and [retained stage registry](/Users/petersuggate/code/me/fluid/lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:402).
[^4]: W3C, [WebGPU Shading Language: synchronization builtins](https://www.w3.org/TR/WGSL/#synchronization-builtin-functions), Candidate Recommendation Draft 31 August 2026, accessed 13 September 2026.
[^5]: GPU for the Web Working Group, [WebGPU specification: resource usages](https://gpuweb.github.io/gpuweb/#resource-usages), accessed 13 September 2026.
[^6]: Apple, [Finding your Metal app's GPU occupancy](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy), accessed 13 September 2026.
[^7]: Repository guidance, [AGENTS.md](/Users/petersuggate/code/me/fluid/AGENTS.md:1) and [Sparse CM12 Dawn regression gate](/Users/petersuggate/code/me/fluid/docs/SPARSE_CM12_DAWN_REGRESSION.md:1).
