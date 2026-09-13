# Adaptive-volume numerical pipeline performance audit

This audit inspects the live production Sparse Geometric / adaptive-volume path on 13 September 2026, with `minimal-power-dam-break-32` as the concrete scene. It covers velocity extension, face preparation, body forces, pressure membership and row classification, coefficient publication, right-hand-side construction, PCG, projection, collocation, tracer advection, and the numerical topology accessors they share. It does not describe the old adaptive-mass implementation or the CPU advance slice as production.

## Executive finding

The numerical pipeline already has several compiled representations, but they stop one layer above the facts most hot kernels need. TEI2 compiles leaf, packet, and point-owner data; BFA1 compiles all-rung projection addresses; PEI1 compacts pressure cell IDs and membership. The hot numerical kernels still repeatedly decode mutable authored/dynamic rows, walk term and incidence lists, re-evaluate acceptance, reconstruct dynamic-page addresses, and recompute static solid geometry.

The primary architectural change should be one **complete compiled topology**, with a numerical view emitted by the same shared compiler that emits transport and presentation views. Rebuild all views after every accepted topology generation, and after any static SolidWorld geometry generation that changes precomputed boundary factors. Do not patch records incrementally. The numerical view should publish immutable accepted cell and row records, row stencils, cell-major incidence recipes, velocity-extension adjacency, accepted projection addresses, accepted packet lists, and indirect counts under the shared generation seal. Per-frame liquid membership, ghost-fluid theta, operator scale, diagonal, and RHS remain a separate fully rebuilt numerical image because they depend on current state.

This split removes connectivity work from every velocity-extension sweep and every pressure matrix-vector product without pretending that liquid membership is topology-invariant.

## Evidence and interpretation for mini32

`minimal-power-dam-break-32` is the 0.8 m dam at 0.025 m finest resolution and installs a static voxel shell (`lib/core/scenes.ts:969-979`, `lib/core/scenes.ts:2613-2622`). Although its scene catalogue profile is the older coarse-only profile, every sparse CM12 regression simulation lane overrides that with the balanced adaptive-volume production defaults (`docs/SPARSE_CM12_DAWN_REGRESSION.md:63-65`). The canonical current gate is B8/P8, 3 warm-up frames, 12 measured frames, a 24.576 ms reference and a 40 ms median ceiling (`tools/sparse-cm12-dawn-regression-manifest.ts:141-154`).

A historical receipt reports 24 quiescent samples, 167.1168 ms median advance, 20.906 ms pressure, and 11.862 ms compact face preparation (`artifacts/sparse-cm12-rdf-presentation-performance-final/mini32-rdf.json:12797-12829`, `artifacts/sparse-cm12-rdf-presentation-performance-final/mini32-rdf.json:15322-15425`, `artifacts/sparse-cm12-rdf-presentation-performance-final/mini32-rdf.json:15459-15521`). Its source fingerprint is stale relative to this audit, so those values are directional evidence only, not a current baseline. The checked-in Dawn gate is the acceptance authority.

Mini32 exposes two global static-solid penalties:

1. Velocity extension takes its generic incidence path for every cell whenever any static solid voxels exist, so the room shell disables the arithmetic interior fast path throughout the domain (`lib/methods/adaptive-volume/sparse-cm12-velocity-extension.wgsl.ts:336-376`).
2. Pressure row repair marks every accepted row dirty whenever static solid voxels exist, and the reuse branch is disabled by the same global flag (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:6041-6072`).

Both costs arise from representation, not from changing physics. The shell is static. A full compiler can resolve which rows actually touch static geometry and precompute their static aperture/closure factors while leaving moving-solid overlays dynamic.

## The two required lifetimes

### Compiled numerical topology view

Key the shared compiled image by `(acceptedTopologyGeneration, staticSolidWorldGeneration)`. Rebuild every transport, numerical, and presentation record and list in one compilation transaction, then atomically publish the completed selector and indirect counts. The CNX1 name below denotes its numerical plane family, not a second compiler or independent topology authority. Never retain or repair individual records across generations.

Recommended contents:

| Plane | Contents | Main consumers |
| --- | --- | --- |
| Accepted cells | stable state ID, brick ID, centre/width/volume descriptor, incidence offset/count, packet/lane | membership, RHS, SpMV, collocation, VEX |
| Accepted rows | stable state ID, axis/kind/opcode, term offset/count, centre/distance, static area/dual, static SolidWorld open/pressure-open flags and values | face prep, row classification, force, operator, projection |
| Row terms | state cell ID and exact coefficient, in the existing canonical term order | theta, pressure gradient, VEX |
| Cell incidence recipes | compiled row-record index, own term/coefficient, axis/sign, static base weight | diagonal, RHS, SpMV, collocation |
| VEX adjacency | six side buckets of neighbour ID and exact subface weight; generic overflow only for non-production QA rows | eight VEX sweeps |
| Accepted face programs | accepted interior tiles, seam packets, sparse-air packets, dynamic rows | projection |
| Accepted packets/leaves | TEI descriptors and compact packet lists with final indirect triples | transport and VEX |

Strong 2:1 production rows have the bounded 2-, 3-, and 5-term cases already encoded in `pressureRowGradient`; ungraded QA topologies retain a generic fallback (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:308-362`). The compiler should emit an opcode for those exact sum trees. This keeps the current reflection-invariant arithmetic rather than expanding the operator into cancellation-prone matrix coefficients.

### Per-frame numerical image

Rebuild this image in full after interface geometry and current scalar state are ready:

- pressure-cell membership bitset and stable ascending cell list;
- pressure-row membership and theta;
- dynamic row scale, such as `staticDual * pressureOpen / theta`;
- cell diagonal and RHS;
- indirect counts and solve gates.

This is intentionally a full frame build. Membership depends on density, previous submerged membership, source rates, and moving-solid prediction (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5344-5363`, `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5450-5508`). Theta depends on current geometric interface and membership (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5908-6012`). Neither belongs in the topology-generation image.

### Data lifetime matrix

| Numerical work | Topology/static inputs suitable for CNX1 | Per-frame inputs that stay outside CNX1 | Outputs |
| --- | --- | --- | --- |
| VEX schedule/init/sweep | accepted packet list; cell-to-packet map; neighbour IDs; subface weights; static closure | source wetness and velocity; depth validity; moving-solid row openness | effective transport velocity and accepted depth |
| Face preparation | accepted row order; row axis/centre; term IDs; cell widths | support flags; source face/cell velocity; `dt`; refinement policy | predicted destination face velocity |
| Body force | row axis/kind; endpoint and static boundary facts | acceleration; `dt`; predicted velocity; separation state; inflow | forced source/destination face velocity |
| Pressure-cell classification | accepted cell IDs; open-volume static part; compiled neighbour graph | density; previous sealed membership; moving-solid capacity; source rate | new membership/list |
| Pressure-row classification | row kind/axis/centre/distance; ordered terms and coefficients; static dual/open factors | new cell membership; interface planes/density; moving-solid state; gravity and region policy | row membership, theta, dynamic row scale |
| Diagonal and RHS | cell incidence recipes; own coefficient/axis/sign; static base weight | row membership/scale; face velocity; moving wall velocity; capacity/source rates | diagonal and RHS |
| SpMV / PCG | row opcodes and terms; cell incidence recipes | vector plane; pressure membership; dynamic row scale; solver scalars | operator image, recurrence vectors and reductions |
| Projection | accepted face addresses; row opcode | pressure vector; row theta/open/separation; predicted face velocity | projected face velocity |
| Collocation | cell incidence recipes | projected faces; moving-solid openness; pressure membership | cell velocity and divergence receipts |
| Tracers | TEI point-owner/packet image | marker positions; effective velocity; density; `dt` | marker positions/visibility |

## Live stage audit

The host encodes the production sequence in `webgpu-sparse-cm12-resident.ts:6706-7066`. The following audit follows that actual path.

### 1. Transport velocity extension

The stage seeds destination velocity, refreshes and extends interface data, performs topology preparation, optionally handles rigid bodies, then runs VEX schedule begin/compile/seal and eight breadth sweeps (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6706-6772`). The schedule shader detects an unchanged generation and avoids appending packets, but the host still submits begin, compile, seal and the indirect-argument copy every frame (`lib/methods/adaptive-volume/sparse-cm12-velocity-extension.wgsl.ts:124-167`). Fold this packet list into the full topology compiler and publish the final VEX indirect arguments with the generation. Stable frames should begin at initialization.

The initialization pass is packet-local but reloads the four-word TEI descriptor in every lane (`lib/methods/adaptive-volume/sparse-cm12-velocity-extension.wgsl.ts:267-303`). Load it once into workgroup storage only if a measured result justifies the barrier; a prior workgroup metadata-cache experiment regressed, so the safer initial optimization is a smaller compiled packet record and direct lane arithmetic.

Each unresolved mini32 cell currently traverses incidence and row terms because the static shell makes `hasStaticSolidVoxels()` true globally (`lib/methods/adaptive-volume/sparse-cm12-velocity-extension.wgsl.ts:312-389`). Compile the neighbour/weight stencil once. Static closed rows can be absent from that stencil; moving-body openness remains a per-frame multiplier or exception. The eight sweeps cannot be fused into one ordinary WebGPU dispatch because each depth needs globally visible results from every workgroup at the preceding depth.

The stage also performs a full interface reconstruction and one-ring support extension before topology preparation (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6725-6732`). Pressure repeats both over accepted cells after face and force work (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6803-6807`). The reconstruction reads destination density, topology incidence, and solid openness (`lib/methods/adaptive-volume/geometric-interface-resident.wgsl.ts:443-468`, `lib/methods/adaptive-volume/geometric-interface-resident.wgsl.ts:479-520`); face prediction and ordinary body force do not change those inputs. On a stable mini32 frame with no rigid body, the pressure refresh is therefore the same expensive reconstruction. Publish the first result with its scalar, topology, and static-solid generations. Reuse it for pressure when those generations still match; if topology preparation adopts a new generation or rigid geometry changes, fully rebuild the interface caches over the new accepted image. This is generation-level reuse of a complete image, not record-level repair.

### 2. Face preparation

The stage seeds the geometric-volume destination a second time, republishes face support, and then dispatches one invocation per accepted row (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6725-6727`, `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6773-6784`). On mini32 there is no rigid coupling and VEX writes its dedicated effective-velocity plane rather than the destination state, so the second seed appears to repeat the first cell copy. Prove that no topology-prepare hook writes accepted destination state, then remove the second mini32 pass or move the one seed to its last required point. `prepareTransportFaceRow` walks every row term to read the dense four-float support cache, then may walk the terms again when a partial refinement region is active (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:3977-4027`). The mini32 scene does not author a partial refinement region; that second traversal is absent unless the runner adds such an override. The first traversal is required because wet/extended flags change each frame.

Retain the dense support cache. The compiled row record should supply term IDs, axis, centre, and static width facts without generic row decoding. Load each term record once and accumulate support flags and minimum width in the same loop. Face RK2 and staggered sampling remain dynamic; cubic sampling can touch up to 64 nodes and falls back to linear at uncertified boundaries (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:3918-3974`). The topology image helps owner lookup, but it cannot eliminate these velocity samples.

### 3. Body forces and continuous source

`forceFaces` immediately rereads every accepted row after face preparation, rechecks acceptance, evaluates static/dynamic boundary openness, adds acceleration, evaluates inflow coverage, and writes both face parity banks (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5289-5329`). For mini32 there is no inflow, and the encoder executes only force plus the source begin/cell initialization; the 32 connect/compress rounds and source publication are conditional on inflow (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6785-6801`).

The force arithmetic can share face preparation's row metadata and predicted value, but it cannot publish both parity banks in that same dispatch. Face prediction samples neighbouring `sourceFaceVelocity` values, while `publishForcedFace` writes `sourceFaceVelocity`; combining them creates a cross-workgroup read/write race. A safe experiment is one prediction-plus-force pass that writes destination only, followed by a simple accepted-row destination-to-source publication pass after the global dispatch boundary. This retains two dispatches but turns the second into a contiguous copy and removes its topology, boundary, and inflow work. Keep the prescribed-inflow semantics in the first pass. Closed-world separation remains dynamic because it reads density, velocity, gravity, and the prior row membership (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:1698-1734`); the compiled record can still provide row kind, orientation, endpoint, and distance.

### 4. Pressure-cell membership

The current transaction opens PCM cell/row, PCF, membership planning, and PTR state; copies indirect arguments; may run a bootstrap classification; always scans accepted cells again with `classifyDirtyPressureCells`; finalizes a dirty frontier; repairs membership leaves; and finalizes cells (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6803-6844`). Bootstrap and dirty classification can both visit the same accepted cells. The dirty pass recomputes the membership predicate just to compare it with the old bit (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5520-5548`).

Replace that transaction with one complete accepted-cell classification followed by deterministic count/scan/emit. Emit the membership bitset and stable ascending cell list directly. Read submerged-history tests from the prior sealed membership image and write the new image to the other slot, so parallel classification never observes partially published current-frame bits. The current PCM allocation carries a full-capacity candidate-token plane, dirty stamps/list, and a multi-level rank tree (`lib/methods/adaptive-volume/sparse-cm12-canonical-membership.ts:69-101`, `lib/methods/adaptive-volume/sparse-cm12-canonical-membership.ts:123-177`); leaf repair walks candidate tokens and updates every ancestor count (`lib/methods/adaptive-volume/sparse-cm12-canonical-membership.wgsl.ts:102-153`). Those structures have no role in a full rebuild.

The predicate itself is dynamic and must remain: it checks active/open cells, current density, previously submerged cells, moving-solid predicted fill, and source rates. Its neighbour walks should use compiled cell incidence and row-term records rather than generic access.

### 5. Pressure-row classification

The current path marks dirty row tiles over accepted rows, scans the row-tile capacity to compact the dirty list, seals it, copies an indirect triple, classifies compacted tiles, and finalizes (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6845-6859`). Because mini32 has static solid voxels, every accepted row is dirty (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:6041-6053`). It therefore pays the dirty-mark and compaction machinery before doing effectively full classification.

Classify every compiled accepted row once per frame and publish the row bitset/theta directly. Remove dirty tile stamps, dirty lists, repair controls, and scalar-change graph walks. The classification body currently:

- rereads acceptance and requirement metadata for diagnostic counters;
- evaluates row dual/open state;
- walks terms once to combine geometric planes;
- walks terms again to validate sign and open fractions;
- walks terms a third time to compute liquid/air phi sums and gradients (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5908-5975`).

Use the compiled row opcode and bounded local arrays to load each term's ID/coefficient once. Reuse those registers across the geometric validation and phi calculations. Keep the exact reduction order. Partial-refinement hydrostatic classification scans five vertical columns (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5387-5437`), but it is gated to a partial authored region and does not apply to the native mini32 scene unless the runner adds such a region (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5365-5377`, `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5984-6001`).

### 6. Pressure execution image and coefficients

After row publication, the encoder rank-selects PCM cells into PEI, builds the full membership image, publishes coefficients, finalizes PCF, finalizes PEI, copies two indirect blocks, commits PTR, immediately reopens PTR, and prepares pressure (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6861-6907`). The coarse and hierarchy substages are retained only as no-op phase boundaries.

The cell-ID publisher descends the PCM rank tree for every output cell (`lib/methods/adaptive-volume/sparse-cm12-canonical-membership.wgsl.ts:55-99`). The membership publisher then scans every capacity word, binary-searches the just-built list, and performs entered/retired callbacks (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5827-5860`). Direct count/scan/emit during full classification eliminates both transformations.

Production uses Jacobi only; PEI explicitly reports zero wet-brick and hierarchy work (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5671-5680`). Yet its layout still allocates wet-brick and hierarchy planes plus four retired indirect triplets (`lib/methods/adaptive-volume/sparse-cm12-pressure-execution-image.ts:24-54`, `lib/methods/adaptive-volume/sparse-cm12-pressure-execution-image.ts:68-105`). Its finalizer retains generic loops for those zero-count domains (`lib/methods/adaptive-volume/sparse-cm12-pressure-execution-image.wgsl.ts:139-217`). Reduce PEI to the frame cell list, membership, row state, and solve gates.

`publishPressureCoefficientCell` has two costly forms. If the world has any dynamic leaves, all authored and dynamic cells use incidence traversal (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5743-5758`). Otherwise it may update a directed-edge cache and then independently traverse incidence to build the diagonal (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:5760-5825`). The iterative operator never consumes that fine directed-edge cache; it reconstructs `G^T W G` from rows. Coarse/hierarchy consumers are retired. Remove the persistent fine-edge change tracking and compute only the Jacobi diagonal from compiled incidence recipes.

The persistent cache currently allocates brick aggregate edges, brick diagonals/ranges, hierarchy edges/diagonals, and four families of candidate generations, active bits, dirty lists/stamps, and count trees (`lib/methods/adaptive-volume/sparse-cm12-persistent-pressure-cache.ts:158-227`). Most of that memory and transaction state can disappear with the retired aggregate path.

Fuse diagonal construction with `preparePressure`. Both execute after row theta is final and traverse the same pressure-cell incidence. `preparePressure` currently loads `pressureDensity` into an unused local (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:6157-6184`); delete that dead work. The fused pass should compute diagonal and divergence/source RHS in the current axis/sign accumulation order.

### 7. Pressure RHS and PCG initialization

The encoder currently performs:

1. `initializePCG`: one `A*p`, residual, and a Jacobi result;
2. `initializeJacobiDirection`: recompute Jacobi, set `z` and direction, reduce gamma and RHS norm;
3. `measureTrueResidual`: a second identical `A*p`, reduce true residual;
4. `initializePipelinedImage`: `A*z`, reduce curvature (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6909-6927`).

`initializePCG`'s first Jacobi write is overwritten before use, and the immediate true-residual pass computes the same `b-Ap` before pressure changes (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:6126-6135`, `lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:6347-6370`). Fuse the first three items into one pressure-cell pass: compute `A*p`, residual, Jacobi `z`, direction, gamma, RHS squared, residual squared, and residual max. A four-component partial holds the four reductions; one reduction initializes solver scalars and the convergence gate. `A*z` remains necessary.

This is the highest-confidence arithmetic coalescing opportunity because it removes a complete SpMV without changing values or crossing a dependency boundary.

### 8. PCG iterations and recovery

Each ordinary iteration requires state update, `A*z`, and a global reduction (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6928-6934`). Do not fuse state update with SpMV: neighbouring directions must all be updated before any row gradient reads them. WebGPU has no grid-wide barrier within a dispatch. Likewise, consecutive PCG iterations remain separate because their recurrence consumes the global dot product.

Every eighth iteration, the host unconditionally encodes true residual, reduction, restart, Jacobi recovery, recovery reduction, recovery image, and recovery reduction before updating the gate (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6934-6947`). Shader predicates suppress arithmetic, but the cell workgroups and reduction dispatches still launch. Publish a distinct recovery indirect triple immediately after the guarded residual reduction, copy it once, and use zero-work indirect dispatches when recovery is unnecessary. When recovery is required, fuse residual restart and Jacobi-direction initialization; then retain the required `A*z` and curvature reduction.

The final true residual is a correctness receipt and must remain (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6953-6958`).

### 9. Operator access inside every SpMV

`applyOperator` walks a cell's incidence. For every active row it loads theta and membership, obtains its own coefficient, calls `pressureRowGradient`, and then loads row dual/axis (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:6139-6155`). `pressureRowGradient` then loads the row term range, term coefficients and cells, performs a PEI membership-bit lookup per endpoint, and reads the vector plane (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:308-362`). This nested graph traversal is repeated for initialization, every PCG iteration, every true-residual guard, recovery, and final residual.

The generic row accessor makes that more expensive for live pages. Authored rows use nine structure-of-arrays planes; dynamic rows branch by ID, divide/modulo into a page, remap semantic planes, and derive area/exterior facts (`lib/methods/adaptive-volume/sparse-cm12-row-access.wgsl.ts:140-176`, `lib/methods/adaptive-volume/sparse-cm12-row-access.wgsl.ts:258-307`). Dynamic incidences similarly branch, derive page/cell/side, test boundary overrides, and reconstruct interior row/term IDs (`lib/methods/adaptive-volume/sparse-cm12-row-access.wgsl.ts:202-257`). These are excellent construction semantics and poor repeated inner-loop semantics.

The compiled topology should expose both row-major stencils and cell-major incidence recipes. The per-frame row image should expose membership/theta/dynamic scale by compact row-record index. SpMV then follows contiguous compiled incidence entries and invokes a row opcode over contiguous term IDs. It still checks dynamic pressure membership, but it avoids mutable topology atomics, authored/dynamic branches, requirement walks, page arithmetic, and repeated static geometry loads.

There is a second, potentially larger operator experiment after that cutover. Today every incident pressure cell recomputes the complete row gradient. A two-term row is evaluated twice, while a mixed five-term row can be evaluated as many as five times. Benchmark a row-major `G*x` pass that writes one gradient per active row, followed in the same compute pass by a cell-major `G^T*(rowScale*gradient)` gather. This adds one dispatch and a row scratch stream but removes repeated endpoint membership/vector loads, especially at seams. It preserves the exact row sum tree and cell accumulation order. Keep the direct cell-major opcode path as the comparator; choose from timestamps rather than assuming the extra scratch traffic wins on unified memory.

Preserve stable ascending pressure-cell order at first. Reordering by brick could improve locality, but it changes reduction order and should be a separate numerical change with symmetry and hydrostatic evidence.

### 10. Velocity projection

BFA1 is already an immutable compiled projection address program, but it is all-rung rather than accepted-generation-specific (`lib/methods/adaptive-volume/sparse-cm12-brick-tile-face-address-program.ts:1-8`, `lib/methods/adaptive-volume/sparse-cm12-brick-tile-face-address-program.ts:75-163`). Every projected row still resolves through ITR and rechecks `rowAccepted` and pressure row membership (`lib/methods/adaptive-volume/sparse-cm12-brick-tile-face-address-program.wgsl.ts:30-65`). The host dispatches three static BFA families and then a fourth accepted dynamic-row family (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6965-6977`).

Emit accepted interior, seam, sparse-air, and live-page addresses together in the generation compiler. The projection kernels can trust the sealed accepted address list and check only dynamic pressure-row membership. `projectPressureRow` should use the same compiled row opcode and per-frame row scale as SpMV, preserving a single definition of the gradient.

Do not fuse projection and collocation. Projection is row-owned and writes face velocities; collocation is cell-owned and must observe all incident projected faces after a global dispatch boundary.

### 11. Collocation and divergence diagnostics

The cell kernel already coalesces velocity collocation and postprojection divergence in one incidence walk, then the host reduces diagnostics (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6977-6980`). Keep that fusion. Replace its generic incidence/acceptance/axis/static-geometry reads with the compiled cell recipe. Remove the unused raw-density load in the collocation body (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:6641-6716`).

### 12. Projected velocity extension

After projection, the host prepares a topology edit transaction, runs another VEX begin/compile/seal sequence, copies arguments, runs a separate gate publisher, initializes VEX, and performs eight more sweeps (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:6988-7029`). The second extension is semantically needed for projected transport velocity when projected topology is active. Its schedule construction should consume the same candidate generation compiler and published packet list. If projected topology equals the accepted topology, reuse the accepted generation's schedule; if it differs, build the complete candidate image once. No record-level patching is needed.

### 13. Tracer advection

Tracer work is omitted when tracers are disabled or the lattice is empty (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:7033-7046`). When enabled, its TEI directory stages a 27-leaf window for a 64-marker group. The TEI structure and point-owner lookup are already compiled (`lib/methods/adaptive-volume/sparse-cm12-transport-execution-image.wgsl.ts:51-121`, `lib/methods/adaptive-volume/sparse-cm12-transport-execution-image.wgsl.ts:123-179`). Spatially divergent markers can fall outside the first marker's staged window and take the fallback. Compact or sort live tracers spatially only if tracer profiles show material cost; it is outside mini32's default critical path.

## Repeated access patterns to eliminate

### Accepted invocation followed by acceptance validation

Accepted list access itself loads the topology worklist base, count, selector-dependent offset, and stable ID (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:1378-1420`). Many consumers immediately call `rowAccepted` again. That predicate either performs live-page arithmetic or walks every host row requirement and rechecks brick activity/resolution (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:2043-2082`). A sealed generation image should guarantee that its accepted IDs are valid for its generation; hot consumers need only a generation-level fail-closed gate and final-lane bounds check.

### Static boundary openness

`physicalRowHasClosedEndpoint` walks every row term and samples cell openness; both physical and pressure open-fraction helpers can invoke it, and `rowDualWeight`/`rowArea` layer more loads over it (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts:1639-1693`). Precompute the static SolidWorld part per compiled row. Moving rigid geometry remains in its own dynamic plane.

### Same connectivity represented several ways

The runtime binds immutable pressure templates at binding 14, PEI/worklists at binding 15, and the mutable topology arena at binding 16 (`lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts:5515-5531`). Hot pressure kernels therefore mix compiled immutable topology, frame lists, and mutable semantic access. The unified numerical image should become the sole immutable topology input for numerical consumers. Mutable IBO/ITR construction authority can remain behind the compiler and generation transaction.

## Coalescing decisions

| Change | Decision | Reason |
| --- | --- | --- |
| Face preparation + ordinary body-force arithmetic | Split publication | Compute prediction and force together into destination, then copy destination to source in a second row pass. Writing source while neighbouring workgroups still sample it is a race. |
| Pressure membership + row classification | Do not fuse | Row theta requires completed cell membership for every endpoint. Use two full passes with a global boundary. |
| Row classification's three term walks | Coalesce locally | Bounded production term count permits one load into local arrays while retaining the exact arithmetic trees. |
| Pressure diagonal + RHS | Do | Same pressure-cell incidence after theta; independent accumulators fit one pass. |
| PCG seed residual + Jacobi direction + initial true residual | Do | Same `A*p`; current second SpMV is mathematically identical and the first Jacobi result is overwritten. |
| PCG state update + SpMV | Do not fuse | Cross-workgroup reads require all direction updates to complete first. |
| Guard recovery passes | Gate, then partially fuse | Zero-work indirect dispatch when unused; fuse restart and Jacobi only on the recovery arm. |
| Projection + collocation | Do not fuse | Row writes must be globally complete before cell gathers. |
| VEX depth sweeps | Do not fuse | Each depth has a grid-wide dependence on the prior mask/value image. |
| Pre-topology and pre-pressure interface refreshes | Reuse by sealed generation | Density is unchanged by face/force work; reuse on stable non-rigid frames, fully rebuild if topology/static-solid/moving-geometry generation changes. |
| Post-transport interface refresh | Keep | Scalar transport changed density, so publication/adaptivity needs a fresh reconstruction. |

## Recommended implementation order

1. **Add counters before changing layout.** Record accepted cells/rows, average and maximum row terms, average cell incidence, row-access path counts, SpMV count, VEX generic/interior lane counts, PCM dirty/accepted ratio, recovery-fired blocks, and bytes allocated to PCM/PCF/PEI. Attribute by the existing stages.
2. **Add CNX1 numerical planes to the one shared compiled-topology ABI.** Use the existing shared compiler transaction to emit double-buffered full views with one generation tuple, validated counts/offsets, and a selector published only after full validation. Compile accepted cell/row/term/incidence/VEX/projection/packet planes in deterministic stable order. Give the interface cache the same topology/static-solid generation receipt plus its scalar epoch.
3. **Cut VEX and projection over first.** These changes remove mini32's global static-solid fallback and per-frame VEX schedule transaction while using comparatively isolated consumers. Keep the eight sweep barriers.
4. **Replace PCM/PCF incremental machinery with a full PEI frame compiler.** Full classify cells, count/scan/emit membership/list, full classify rows, compute dynamic row scales, and rebuild the Jacobi diagonal. Keep PTR only while it remains the shared topology-effects journal. Delete retired aggregate/hierarchy storage once receipts prove it has no live numerical consumer.
5. **Cut the iterative operator and projection gradient to CNX1.** Preserve the current 2/3/5-term sum trees, axis-separated accumulation, stable pressure-cell order, and final true residual.
6. **Benchmark shared row-gradient SpMV.** Compare direct compiled cell-major application with row-major gradient plus cell-major transpose gather on mini32 and mixed-ratio scenes.
7. **Coalesce proven work.** Keep the fused PCG initialization, split face-force arithmetic from its source-bank publication, fuse recovery restart/Jacobi, and add a dedicated recovery indirect gate.
8. **Remove old access paths from production compilation.** Keep generic topology access only for compiler validation and deliberately ungraded QA topology. Mark alternate CPU slice and old adaptive-mass paths explicitly so their data structures do not constrain CNX1.

## Alternate paths excluded from the design authority

The files under `lib/methods/adaptive-volume/advance-slice/` are CPU/reference and lab-slice implementations. They are useful numerical oracles but are not the resident production dispatch path. Older adaptive-mass material and historical transformation documents describe another solver architecture. CNX1 should be specified from the live encoder and resident WGSL cited above, then checked against those alternate paths only for mathematical consistency.

## Validation and acceptance

For each cutover, compare the production result against a preserved baseline capture. Validate exact stable IDs, term order, coefficients, incidence reciprocity, row kinds, static geometry factors, VEX neighbour weights, and accepted projection coverage. Do not retain duplicate production authorities for comparison. Because the proposal intentionally changes dispatch grouping but should not change arithmetic order inside rows/cells, require:

- pressure operator symmetry and finite diagonal receipts;
- hydrostatic first-step stability;
- mini32 correctness and volume retention;
- live rigid and liquid insertion;
- far-wall dam fronts and floor-only symmetry;
- mini32 and mini64 timestamped performance.

After the complete implementation, run the repository's canonical `npm run test:dawn:sparse-cm12` gate described in `AGENTS.md`; diagnosis can use individual lanes, but do not raise timing ceilings. This research pass did not run Dawn, a browser, or any GPU workload.

## Expected priority for mini32

The most likely wins, in order, are:

1. compiled VEX adjacency that restores a cheap interior path despite the static shell;
2. compiled row/cell recipes for every repeated pressure SpMV;
3. removal of full-dirty PCM/row/PCF transactions in favour of one full frame build;
4. deletion of the duplicate seed SpMV and precondition;
5. zero-work recovery dispatches on ordinary eight-iteration guards;
6. face-preparation/body-force arithmetic with a separate source publication barrier;
7. accepted-generation BFA lists without `rowAccepted` rechecks.

The historical receipt suggests transport still dominates overall, so numerical-topology work should be measured beside the geometric transport work rather than treated as the only bottleneck. Within this audit's scope, pressure and face preparation are large enough to justify the structural cutover, while the shared compiler benefits both them and VEX.

## Production cutover completed in this change

The first numerical cutover now uses CNX as the pressure and projection connectivity authority. PEI2 rebuilds the complete pressure-cell membership, stable ascending cell list, pressure-row membership, theta field, and Jacobi diagonal after scalar conditioning. Cell and row membership use inactive/active slots; seal changes the selector only after CNX generation, stage order, capacity, and sticky failure checks succeed. Theta and diagonal remain shared numerical planes, so a failed build is terminal for the frame rather than a rollback to the old numerical image.

The old adaptive-volume PCM, PCF, PCA, frozen-pressure publication, immutable pressure-template upload, directed-edge cache, coarse pressure cache, hierarchy cache, and their production WGSL entrypoints have been removed. PTR remains only as the topology-effects journal and can accept its generation only after `peiFullImageAccepted()` proves the full pressure image and CNX generation agree. BFA projection, geometric air projection, transport diagnostics, pressure SpMV, projection, and collocation now read PEI membership.

PCG initialization now evaluates the seed `b-Ap` once and publishes the Jacobi direction, gamma, RHS norm, true-residual norm, maximum residual, and initial convergence gate through the same reduction tree. The final true-residual pass remains authoritative. The authored uniform-interior Jacobi diagonal retains the prior source incidence order and two-term off-diagonal recurrence through CNX, while mixed and dynamic cells retain the axis-grouped generic order.

Source-only validation constructs the mandatory compiled topology rather than substituting legacy accessors. CPU/source contract tests cover disjoint PEI slots, stable compaction, full-image journal gating, CNX generation fencing, and exact seed reduction behavior. GPU differential hashes and the canonical Dawn regression remain the acceptance gate owned by the serialized test run.
