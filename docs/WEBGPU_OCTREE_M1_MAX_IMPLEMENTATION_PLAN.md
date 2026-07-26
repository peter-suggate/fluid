# WebGPU power-octree implementation plan for Apple M1 Max

## Scope

This plan applies only to the existing TypeScript/WGSL WebGPU octree method.
The native Metal solver is out of scope.

The target is the architectural intent of the supplied recommendations:

- explicit sparse page pools with dense hot loops;
- table-driven power geometry;
- the paper's stronger same/finer-or-same/coarser local topology invariant;
- topology-class-specialized kernels;
- synchronization-poor MGPCG;
- destination-owned Eulerian velocity transport;
- GPU-driven, double-buffered topology publication;
- bandwidth-first layouts and scheduling.

Three literal Metal-specific mechanisms do not exist in WebGPU and therefore
have explicit WebGPU substitutions:

1. WebGPU does not expose `MTLHeap`, `MTLBuffer` storage modes, or concurrent
   CPU access to an in-use `GPUBuffer`. The replacement is a small number of
   large `GPUBuffer` arenas, with all recurring allocation, directory, and
   topology work performed on the GPU.
2. A CPU helper cannot safely mutate topology buffers concurrently with the
   WebGPU queue. The replacement is a double-buffered GPU topology transaction
   prepared for the next substep and published by an epoch flip.
3. A CPU coarse solve would require `mapAsync`, a completed submission, and a
   second submission. The production replacement is a persistent one-workgroup
   GPU coarse solve. CPU coarse solving may exist only as an offline oracle.

These substitutions preserve the performance principles—no hot-path
readbacks, no sparse lookup in regular kernels, and minimal synchronization—
without touching the native Metal application.

**This document is a specification, not a status report.** Phase sections
below describe intended end states. For what is actually implemented, and for
the phases whose exit gates are not yet met, read
[Implementation status and open findings](#implementation-status-and-open-findings)
at the end of this document first.

## Reconciled architectural decisions

The two recommendation sets overlap but are not identical. The implementation
uses these explicit resolutions:

1. **The power diagram remains implicit.** Offline tables may contain polygon
   geometry used to generate coefficients, but the per-timestep simulation
   representation contains no polygon vertices, general face list, or general
   adjacency graph. Runtime state is structured page metadata, a dense case ID,
   fixed neighbor handles, six velocity families, dynamic boundary fractions,
   and compact worksets.
2. **The pressure operator remains matrix-free.** “Stamping” means resolving a
   compact case ID, fixed neighbor handles, scale, and dynamic boundary terms
   once per topology epoch. It does not mean emitting a general CSR matrix for
   regular or transition rows.
3. **The hybrid preconditioner is retained.** Matching damped-Jacobi sweeps of
   the accurate second-order operator remain around the first-order adaptive
   V-cycle. Chebyshev is used as the regular multigrid smoother, not as an
   asymmetric replacement for the hybrid pre/post shell.
4. **V-cycle schedules are fixed; outer PCG may stop early.** Every application
   of the preconditioner uses a fixed linear schedule. Only the outer Krylov
   iteration writes zero indirect counts after convergence.
5. **Fine topology uses compact A/B rebuilds.** Fine bricks are allocated by
   mark/rank/scatter into the next compact pool, with no recurring linked free
   list. Pressure/MG pages may retain stable carried physical IDs where that
   prevents expensive value remapping.
6. **The Eulerian face field is the sole momentum-transport authority.** The
   paper-style full-vector face-advection path is the production transport:
   reconstruct a full vector, trace with the previous projected and extended
   field, interpolate with the catalog basis, and project once onto the
   destination family normal. Primary FLIP/APIC is not part of the solver.
   Existing secondary particles remain one-way presentation data and never
   feed momentum, topology, pressure, or the level set.

## Current source baseline

The plan is grounded in the current source tree, including the present
uncommitted changes.

| Recommendation | Current implementation | Required end state |
| --- | --- | --- |
| Explicit sparse page pool | Fine phi already uses a physical page pool, 4³ bricks, compact worklists, direct logical-page lookup, and six stored neighbor IDs. Pressure ownership uses 8³ pages but performs sorted-record binary searches. Other topology identities have separate directories. | One reusable page/workset ABI. Hot kernels receive physical page or compact row indices directly. Every page has precomputed halo adjacency. Sorted or hashed lookup is restricted to topology publication. |
| Power geometry as tables | The generated catalog already contains direct descriptor lookup, face geometry, volumes, tetrahedra, and 19 coefficient channels. Runtime code still reconstructs/clips faces, resolves neighbor rows, builds incidence, and emits a compact operator each changing generation. | Extend the catalog with complete implicit case templates. At topology publication, resolve dense case IDs, bounded neighbor handles, scales, and live boundary terms once. Solver iterations perform no geometry, directory search, deduplication, or CSR assembly. |
| Topology-class kernels | The operator identifies bulk versus cut rows, but both emit kernels are dispatched over the complete live row set and branch per row. Fine-band kernels similarly mix regular and exceptional paths. | GPU-built, disjoint worksets for regular interior, transition interior, physical boundary, and transition-plus-boundary rows; equivalent block classes for fine-phi work. Each workset owns a separately compiled indirect kernel. |
| Sync-poor MGPCG | The default method is Galerkin. The alternate MGPCG uses damped Jacobi and GPU early-out, but the large-domain path retains several globally reduced stages per PCG iteration. SPGrid already has compact per-level worklists and a persistent tiny-domain executor. | Pipelined MGPCG is the production default. Chebyshev smoothers operate on class-specialized applies. One combined hierarchical reduction is used per outer iteration. Global scalars use compensated f32. Small coarse levels run persistently in one workgroup. |
| Momentum transport | The octree method is Eulerian, but recurring face/incidence records and mixed topology branches still inflate the path. Existing secondary particles are one-way presentation data. | Six structured velocity families are the sole momentum authority. Destination-owned full-vector advection uses regular trilinear or transition tetrahedral interpolation. No primary FLIP/APIC graph exists. |
| Parallel topology rebuild | Current topology rebuilding is GPU-resident and delta-aware, but it is encoded before each pressure substep and still carries several sorted publications and capacity-shaped stages. | Topology candidate generation is built from compact fine-band, solid, inflow, and authored refinement deltas into the inactive epoch, with bounded grading passes, precomputed adjacency, worksets, and indirect args. Publication is one validated epoch flip. |
| Bandwidth-first execution | Storage is mostly SoA and many stages are indirect, but there are duplicate velocity forms, capacity-shaped passes, repeated validation loads, runtime bind-group construction in some paths, and few full brick+halo kernels. | One authority per physical quantity, active-content dispatch everywhere, cached bindings, brick-resident halo kernels, deliberate fusion, optional packed-f16 storage, and counters proving bytes/lanes/iterations. |
| Strong topology restriction | The descriptor/catalog path already distinguishes same-or-finer and same-or-coarser configurations, and topology tests exercise the paper grading rule. The invariant is not yet the central contract of every new workset and page publication proposed below. | Every live cell's relevant face/edge 1-ring is exclusively same-or-one-finer or same-or-one-coarser. Mixed finer/coarser rings reject the candidate before case assignment. |
| Implicit power storage | The catalog is implicit, but recurring runtime state still materializes general power-face records, incidence streams, centroids, normals, quadrature, and compact operator rows. | Six structured velocity-family channels plus fixed case-local handles replace the general face graph. Geometry exists in the generator and topology publisher only. |
| Destination-owned transfers | Several current paths already use parent-owned gathers, but this is not yet a repository-wide rule. | Every bounded-fan-in transfer is destination-owned: parent restriction/ghost accumulation, face advection, row publication, and topology carry use deterministic gathers rather than floating scatter. |

The primary orchestration owner is `lib/webgpu-octree.ts`, reached through
`lib/methods/octree.ts` and `lib/webgpu-uniform-eulerian.ts`. The implementation
should become less monolithic as the workset, page-pool, solver, and structured
velocity ABIs are introduced.

## Target substep

The final command graph is:

1. validate and flip a ready topology candidate;
2. advect the fine level set over active bricks with the previous substep's
   projected and extended divergence-free field, before gravity or other
   current-substep forces are applied;
3. advect the coarse octree level set with the same field and correct it from
   transported fine phi wherever the fine band is valid;
4. destination-advect all six structured velocity families with that field,
   then apply forces and solid constraints;
5. build divergence/RHS while classifying cut rows;
6. run pipelined MGPCG over class-specialized operator applies;
7. project faces and seed closest-point velocity extension in one pass;
8. complete fine-band extrapolation/redistance and re-correct coarse phi from
   the newly validated fine publication;
9. derive the next topology candidate from fine interface blocks, solid
   deltas, inflows, authored sizing, and refinement policy;
10. validate the inactive topology epoch and leave its flip ready for the next
    substep.

The current topology remains immutable throughout steps 2–8. Candidate
support must cover at least the maximum encoded CFL displacement, interpolation
halo, redistance width, and one grading ring. If that proof cannot be made,
candidate publication fails closed; kernels must not widen their searches or
fall back to a domain scan.

## Canonical GPU data model

### Channel ownership and layout

Use structure-of-arrays by default. Do not reproduce SPGrid's CPU virtual-page
channel interleaving in the explicit GPU pool.

Separate buffers own:

- pressure and Krylov vectors;
- fine level set;
- six face-velocity families;
- cell flags and masks;
- topology/case IDs;
- dynamic free-surface and solid fractions;
- static solid geometry;
- temporary multigrid vectors.

Small AoSoA bundles are allowed only when every consumer always reads the
fields together and a benchmark demonstrates fewer bytes/transactions. A
pressure stencil must never fetch phi, velocity, solid geometry, or full
topology metadata merely because those channels share a page.

Organize state into three update-frequency groups:

1. immutable or topology-epoch metadata: case ID, neighbor/parent/child page
   handles, static solid geometry, transfer case;
2. per-substep geometry: active masks, boundary fractions, dynamic cut
   coefficients;
3. high-frequency iterative vectors: pressure, residual, preconditioned
   residual, direction, product, and correction.

The inner PCG/V-cycle loop may touch group 3 plus compact case/coefficient data
from groups 1–2. It must not traverse the complete topology or boundary state.

### Page arenas

Add `lib/webgpu-octree-page-pool.ts` as the common planner and runtime ABI for
pressure-owner pages, fine-phi pages, and multigrid pages.

Each level/field domain owns:

- a fixed-capacity physical page arena;
- two topology tables, active and candidate;
- a compact Morton-ordered active-page worklist;
- a logical-page-to-physical-page directory used only at publication and
  cross-page lookup;
- six face-neighbor physical page IDs, plus an optional 27-page halo record
  for trilinear/3×3×3 kernels;
- page flags, topology class, active element count, and epoch;
- indirect dispatch records adjacent to the worklist header;
- an allocation policy:
  - compact rank into the next A/B pool for fine phi;
  - stable carried IDs for pressure/MG pages when remapping would cost more.

Keep the existing 4³ fine brick as the first production shape: 64 samples map
exactly to a 64-lane workgroup and current code/tests already assume it.
Introduce an 8×8×4 experimental shape only after a threadgroup-memory and
occupancy benchmark proves it wins on the M1 Max. Do not make block shape a
runtime branch inside kernels; use pipeline constants or distinct modules.

Pressure and multigrid page shape is a compile-time, benchmark-selected choice,
not a runtime branch. Start with 4³ as the conservative portable lane, but
benchmark it against at least 8×8×4 on the M1 Max. Both candidates must fit
their page, halo, and live f32 channels within the 32 KiB portable
workgroup-memory budget. Selection must include measured halo amplification,
occupancy, and useful repeated local smoother work; the pressure/MG shape is
not frozen until that gate passes. One workgroup processes one page.

The fine level-set pool is rebuilt in bulk. Interface/support bricks mark a
logical occupancy bitset; prefix rank both deduplicates and produces canonical
Morton order; scatter writes a compact next pool; overlapping old values are
gathered by logical key; new bricks initialize from the coarse signed-distance
authority. This is equivalent to sort/unique without performing a radix sort
when the bounded logical bitset is available.

### Independent fine-surface authority

Retain the fine level set as a uniform sparse narrow-band grid independent of
the adaptive pressure/velocity hierarchy. It has one resolution, no
multiresolution pyramid, and a configured factor such as 4× or 8× relative to
the finest physics cells. Its compact brick pool covers only the interface
band, characteristic/interpolation halo, redistance width, and solid support.

The adaptive octree supplies the velocity characteristics; the fine grid owns
the geometric zero crossing, redistance, volume measurement, and surface
sampling. Regridding either structure must not silently transfer authority to
the other.

### Worksets

Add `lib/webgpu-octree-worksets.ts`. A workset is:

```text
header:
  epoch, count, capacity, flags
  dispatchX, dispatchY, dispatchZ
payload:
  compact physical page IDs or row IDs
```

Every recurring kernel consumes a workset or a proven live prefix. Direct
dispatch over `dims³`, row capacity, face capacity, or maximum page capacity is
for bootstrap and diagnostics only.

Topology publication creates these disjoint pressure row sets:

- `regularInteriorRows`: same-resolution, full-open, non-boundary rows;
- `transitionInteriorRows`: catalog transition rows with no free/solid/world
  cut;
- `physicalBoundaryRows`: free-surface, solid, or world-boundary rows with no
  refinement transition;
- `transitionBoundaryRows`: rows containing both a refinement transition and
  a physical boundary;
- `invalidRows`: diagnostic-only rejection records, never a simulation input.

It also creates fine-grid block sets:

- `interfaceBlocks`;
- `transportBlocks`;
- `redistanceBlocks`;
- `rareTransportBlocks`;
- `solidBoundaryBlocks`.

The lists are constructed with mark, prefix rank, and scatter. Their order is
`(level, Morton, local index)` so determinism does not require sorting.

### Topology invariant and dense case IDs

Ordinary 2:1 grading is necessary but insufficient. For every live cell, all
relevant face and edge neighbors must be exclusively:

- the same level or one level finer; or
- the same level or one level coarser.

A cell must never observe both finer and coarser neighbors in its relevant
1-ring. Candidate validation enforces this before descriptor lookup. This
bounded invariant is what permits a 19-bit-or-smaller local descriptor, fixed
stencils, bounded Delaunay tetrahedra, direct velocity reconstruction, and no
tree traversal in hot kernels.

Topology classification maps the sparse descriptor space to consecutive dense
case IDs. Direct 18/19-bit descriptor tables may be used by the rebuild stage,
but hot kernels index only compact, contiguous tables of valid cases. Case zero
is the regular Cartesian specialization and requires no table access.

Each case contains or references:

- presence/orientation of six face and twelve edge neighbors;
- any additional 3-D power-face relationships;
- the fixed Laplace/divergence channels;
- local Delaunay tetrahedra and selectors;
- the least-squares/pseudoinverse velocity reconstruction map;
- parent/child restriction, prolongation, and ghost-transfer behavior;
- class flags and dynamic-boundary slots.

### Resolved implicit power rows

Add a packed `ResolvedPowerRow` arena:

- row class and catalog entry;
- physical cell volume;
- fixed neighbor row/page handles for the case's bounded slots;
- cell scale used with normalized table coefficients;
- fixed dynamic-boundary slots and fractions;
- six structured velocity-family slot/orientation IDs;
- topology epoch.

Regular interior rows use an implicit 7-point layout and store only their base
row/page identity. Transition rows use the catalog's bounded neighbor channels.
Boundary rows use the same fixed case layout plus dynamic fractions; they do
not create a general compact sparse row.

All internal face coefficients are resolved symmetrically from one canonical
structured face-family owner. Both incident rows reference the same channel
value. This prevents independent rounding or cut evaluation from breaking SPD.

### M1 Max execution profile

The representation uses WebGPU `GPUBuffer` pools rather than Metal sparse
resources or Apple-only residency APIs. Production is the measured M1 Max
profile: one 128-lane subgroup implementation. Construction fails closed when
the required subgroup feature or limits are absent. No portable, non-subgroup,
or alternate-lane shader is retained as an executable path.

Kernels remain moderately specialized. Regular, transition, physical-boundary,
transition-boundary, topology, and reduction logic do not share a giant entry
point. Threadgroup memory is a cache for the current brick field, its halo,
reduction scratch, and a genuinely shared compact table—not storage for every
channel owned by the brick.

Compact integer metadata is packed where it reduces traffic without adding hot
unpack overhead: two 16-bit case/local IDs may share a `u32`, and masks,
presence bits, and tetrahedron selectors use the narrowest packed
representation supported by WGSL storage-buffer layout. Authoritative
pressure, Krylov vectors, velocity, level-set distance, cut coefficients, and
dot partials remain f32 until an individual precision experiment passes its
numerical gates.

### Six structured velocity families

At every octree level, store six separate SoA velocity families:

- three ordinary x/y/z face relationships;
- three extra power-face families induced by edge-neighbor transition
  configurations.

A case ID determines whether a family slot is present, its orientation, its
neighbor, and its coefficient. Uniform regions touch only the three ordinary
families. Transition kernels touch the bounded additional families. No
per-timestep polygon vertices, face list, or incidence graph is authoritative.

The catalog also stores the fixed pseudoinverse that maps incident
normal-velocity family values to one cell-centered full vector. Regular cells
use opposing-face averages without a table.

### Eulerian momentum authority

The six structured face-velocity families are the only production momentum
store. Each destination face reconstructs the prior projected and extended
full-vector field, traces its own centroid, samples a regular cube-trilinear or
transition tetrahedral-barycentric interpolant, and projects exactly once onto
the destination normal. Every destination family slot owns its write; no
floating scatter or general incidence graph participates.

The fine level set consumes this divergence-free field before current-substep
forces. The coarse level set is transported by the same field and corrected
from fine phi wherever the fine band is valid. Forces then update the face
families before divergence and pressure projection.

No primary FLIP/APIC particle store, binning graph, P2G, G2P, reseeding, or
particle-driven topology activation exists in production. The existing
`lib/webgpu-secondary-particles.ts` path remains a one-way presentation
feature; it cannot write solver velocity, pressure, topology, fine phi, or
coarse phi.

## Detailed implementation phases

### Phase 0 — lock the contract and measurements

Files:

- `lib/methods/octree.ts`
- `lib/webgpu-octree.ts`
- `lib/webgpu-uniform-eulerian.ts`
- `tools/benchmark-power-dam.ts`
- new `tests/webgpu-octree-work-accounting.test.ts`

Changes:

1. Add per-stage counters for scheduled lanes, active lanes, active pages,
   workset count, topology epochs, catalog stamps, solver iterations, reduction
   passes and bytes allocated by authority.
2. Record free-running wall time and GPU timestamps for the existing mini and
   UI dam lanes before changing numerical defaults.
3. Add a quiescent still-water lane and a moving-interface two-level lane.
4. Make the current uncommitted active-volume and validation-hoisting changes
   pass their focused tests before building on them.
5. Add source-level guards: production octree kernels may not introduce
   full-domain recurring dispatches, unbounded lookup loops, a primary
   FLIP/APIC graph, or floating scatter into face velocity.

Exit gate:

- all current octree tests pass;
- baseline artifacts include stage time, dispatch count, active/scheduled lane
  ratio, memory, residual, volume, and energy;
- the still-water lane exposes fixed-cost work clearly.

Regression attribution is implemented by the fail-closed artifact and
comparator contract in `docs/OCTREE_REGRESSION_ATTRIBUTION.md`. Production
baselines may not contain missing-metric blockers; the minimal dam lane is
accepted only after exactly 500 steps reach 2.0 simulated seconds.

### Phase 1 — common worksets and direct page adjacency

Files:

- new `lib/webgpu-octree-worksets.ts`
- new `lib/webgpu-octree-page-pool.ts`
- `lib/webgpu-octree-fine-levelset-bricks.ts`
- `lib/webgpu-octree-fine-levelset-topology.ts`
- `lib/webgpu-octree-owner-pages.ts`
- `lib/webgpu-octree.ts`

Changes:

1. Define one workset header and indirect-dispatch ABI.
2. Move fine worklist indirect records into that ABI.
3. Replace pressure owner sorted-record lookup with logical-brick direct lookup
   or brick-mask+rank lookup. Preserve stable physical page IDs for pressure/MG
   pages when profitable; the fine A/B pool remains compactly reranked.
4. Publish six physical neighbor IDs for every pressure/fine/MG page and an
   optional 27-page halo record for interpolation kernels.
5. Convert recurring owner, fine topology, redistance, volume, summary, and
   restriction consumers to indirect workset dispatch.
6. Keep binary search only in failure diagnostics until those diagnostics can
   consume the direct directory.
7. Make bounded transfers destination-owned. Fine ghosts gather their unique
   coarse parent; coarse parents gather child/ghost contributions in a fixed
   order. Prolongation and restriction use the same stored transfer case and
   are verified adjoints.

Exit gate:

- zero binary/hash probes in regular fine-brick and pressure-page hot loops;
- every empty workset emits `(0,1,1)` and performs zero element work;
- page carry/add/remove tests prove stable IDs and exact adjacency;
- ghost propagation/accumulation uses no atomics and has deterministic
  floating-point order;
- scheduled lanes scale with active pages, not logical domain size.

### Phase 2 — extend the offline power catalog

Files:

- `lib/octree-power-catalog.ts`
- `tools/generate-octree-power-catalog.ts`
- generated `lib/generated/octree-power-catalog.bin`
- generated `lib/generated/octree-power-catalog.ts`
- `tests/octree-power-catalog-artifact.test.ts`
- `tests/octree-power-catalog.test.ts`

Changes:

1. Extend each entry with a row template:
   canonical neighbor selector, transformed face slot, normalized coefficient,
   normalized area/inverse distance, volume, and row-class hints.
2. Add a direct template for the regular 7-point interior case.
3. Add boundary template metadata identifying which terms require live
   free-surface or solid fractions.
4. Store the fixed least-squares pseudoinverse from incident normal-velocity
   families to cell-centered velocity for every irregular case.
5. Store compact cube/tetrahedron interpolation selectors and parent/child
   ghost-transfer cases.
6. Deduplicate all valid cases and assign consecutive hot case IDs. Preserve
   the existing direct same/finer and same/coarser descriptor tables only as
   rebuild-time maps into those compact IDs.
7. Version the binary ABI and verify its content hash.

Exit gate:

- exhaustive descriptor tests reconstruct the current CPU power operator;
- every internal template is reciprocal and coefficient-bit-identical from
  both incident rows;
- reconstruction maps reproduce constant and affine vector fields;
- transfer cases are exact adjoint pairs;
- table size stays within the existing binding limits;
- no runtime geometry is needed to apply an uncut row.

### Phase 3 — resolve implicit rows and build topology classes once per epoch

Files:

- `lib/webgpu-octree-power-topology.ts`
- `lib/webgpu-octree-power-descriptor.ts`
- `lib/webgpu-octree-power-faces.ts`
- `lib/webgpu-octree-power-operator.ts`
- new `lib/webgpu-octree-power-resolved-rows.ts`
- `lib/webgpu-octree.ts`

Changes:

1. Resolve the descriptor with the direct catalog table.
2. Reject a mixed finer/coarser 1-ring before case assignment, even when the
   ordinary 2:1 ratio check passes.
3. Resolve neighbor row IDs once through the page/rank directory.
4. Publish the `ResolvedPowerRow`, six velocity-family handles, and dynamic
   boundary slots.
5. Build all four disjoint row worksets and their indirect args during the
   same prefix pipeline.
6. Retain dynamic polygon clipping only inside topology/boundary publication
   for actual solid cuts. Free-surface
   ghost-fluid scaling becomes a boundary-row scalar update, not a new
   topology/geometry build.
7. Replace global face sort/merge with canonical structured family ownership
   plus prefix/rank publication. Carry exact unchanged family values through
   old-to-new row remaps.
8. Remove general CSR from all production row classes. Boundary state is fixed
   case-local slots plus dynamic fractions, not a variable row stream.
9. Store polygon, centroid, normal, and quadrature scratch only for the
   topology/boundary publisher; do not commit them as timestep authority.

Exit gate:

- solver iterations call no geometry, sort, binary search, or descriptor code;
- a coefficient-only free-surface update does not rebuild topology geometry;
- resolved implicit operator action is bitwise/differentially equal to the old
  compact operator on the same published topology;
- no live row has a mixed finer/coarser relevant 1-ring;
- no production timestep buffer is a general face list or incidence graph;
- SPD, constant-field, affine-field, and flux-reciprocity tests pass.

### Phase 4 — class-specialized brick and row kernels

Files:

- `lib/webgpu-octree-power-operator.ts`
- `lib/webgpu-octree-power-velocity.ts`
- `lib/webgpu-octree-face-closest-point.ts`
- `lib/webgpu-octree-fine-levelset-fused-transport.ts`
- `lib/webgpu-octree-fine-levelset-redistance.ts`
- new `lib/webgpu-octree-brick-stencils.ts`

Changes:

1. Implement four disjoint operator applies:
   branchless 7-point regular, table-driven transition, physical boundary, and
   transition-plus-boundary.
2. Implement equivalent four-way specialized divergence and pressure-gradient
   kernels over the six velocity families.
3. Stage one page plus halo into workgroup memory. Perform repeated smoother
   operations locally when dependency boundaries permit.
4. Split common fine transport from rare air/no-simplex/solid cases. The common
   pass appends rare samples to `rareTransportBlocks`; the rare kernel is
   separately compiled and indirectly dispatched.
5. Cache bind groups by immutable resource tuple. No per-round bind-group
   creation remains.
6. Hoist publication validation to one workgroup-uniform check or the workset
   publisher.
7. Precompute and consume irregular velocity pseudoinverses. Regular cells use
   opposing-family averages; transition cells execute one fixed
   matrix-vector reconstruction.
8. Split characteristic interpolation into regular trilinear and transition
   tetrahedral worksets. A regular trajectory must not load tetrahedron tables.
9. Make the current full-vector face-advection algorithm the sole production
   momentum lane: reconstruct at the destination face centroid, trace the
   centroid, interpolate advecting and transported full vectors, then project
   once onto the destination normal. Every destination family slot owns its
   write.
10. For fine-grid refinement factor `m`, execute `m` velocity-reinterpolated
    characteristic microsteps while sampling fine phi only once at the final
    departure point. Dispatch only over `transportBlocks`.
11. Keep redistance wholly parallel. Seed from sign changes, run a fixed
    band-width-derived JFA/CPT plus local Eikonal/Jacobi sweep schedule over
    `redistanceBlocks`, and retain no priority queue or host convergence check.
12. Make velocity extrapolation destination-owned and GPU-resident: build
    discrete interface-distance layers or ping/pong frontier masks, then let
    each newly activated sample gather from already valid closer neighbors.
    Transition samples use preencoded case tetrahedra.

Exit gate:

- regular kernels contain no topology-class branch;
- transition-plus-boundary code is absent from both regular and transition-only
  entry points;
- rare transport code is absent from the common shader entry point;
- halo load count is measured once per page per fused local stage;
- regular interpolation performs no tetrahedron-table access;
- a factor-`m` fine backtrace performs `m` velocity samples and exactly one phi
  sample;
- redistance has a fixed GPU-resident iteration bound derived from band width;
- extrapolation uses no scatter atomics or host-visible frontier check;
- validation-injected failures still reject the epoch before it becomes live.

### Phase 5 — pipelined MGPCG

Files:

- delete the pre-pipelined `lib/webgpu-octree-mgpcg.ts` executable path
- `lib/webgpu-octree-spgrid-vcycle.ts`
- new `lib/octree-pipelined-pcg.ts`
- new `lib/webgpu-octree-pipelined-mgpcg.ts`
- `lib/methods/octree.ts`

Changes:

1. Implement a float64 CPU oracle for the selected pipelined/merged PCG
   recurrence before writing WGSL.
2. Retain the symmetric hybrid preconditioner exactly:
   fixed damped-Jacobi sweeps of the accurate second-order operator on the
   compact transition/boundary shell; residual formation; one first-order
   adaptive V-cycle; the identical second-order sweeps in reverse.
3. Use degree-2/4 Chebyshev as the fixed first-order MG smoother. At topology
   publication, compute safe spectral bounds from the scaled first-order
   operator; reject invalid/non-positive bounds. Weighted Jacobi is deleted
   from the production shader graph.
   Every coarse operator is the rediscretized first-order Setaluri operator for
   that level. Production contains no Galerkin RAP/triple-product construction
   at startup, topology publication, or solve time.
4. Make the matrix-free second-order apply order explicit:
   gather coarse values into fine ghosts; regular page stencil; transition
   table stencil; physical and transition-boundary modifications; gather fine
   ghost contributions into coarse parents; publish the result vector.
5. Use one class-specialized operator apply per level.
6. Merge required PCG dot products into one hierarchical reduction per outer
   iteration: optional subgroup reduction, then workgroup partials, then one
   finishing workgroup.
7. Store global Krylov scalars as compensated `(hi, lo)` f32 pairs. Do not use
   emulated f64 in WGSL.
8. Give every MG level its own active-brick workset. Run every level fitting
   the bounded state budget in a persistent workgroup;
   the final coarse solve is an exact GPU workgroup operation.
9. Use a fixed pre/post smoothing and coarse-operation count inside every
   V-cycle. Encode the worst-case outer chain once; convergence writes zero
   indirect counts only for remaining outer PCG iterations.
10. Require subgroup WGSL and the benchmark-selected 128-lane configuration.
    Reject unsupported devices before pipeline construction; do not compile a
    portable reduction alternative.
11. Cut over directly to pipelined MGPCG and delete the Galerkin runtime,
   benchmark, selector, and production-facing tests in the same change. Frozen
   numerical fixtures may remain; executable old solver code may not.

Exit gate:

- one global reduction synchronization per executed outer iteration;
- no Galerkin RAP or triple-product construction is reachable from the
  production octree graph;
- every V-cycle is a fixed linear, symmetric map with destination-owned ghost
  gathers and no transfer atomics;
- the required subgroup configuration fails closed when unavailable;
- symmetry/positivity, manufactured Poisson, mixed transition/boundary, and
  long-run residual tests pass;
- convergence makes the encoded tail do zero row work;
- the UI exposes no automatic solver fallback.

### Phase 6 — Eulerian momentum specialization and cutover

Files:

- `lib/webgpu-octree-power-face-advection.ts`
- `lib/webgpu-octree-power-velocity.ts`
- `lib/webgpu-octree-brick-stencils.ts`
- `lib/webgpu-octree.ts`
- `lib/webgpu-uniform-eulerian.ts`
- `lib/methods/octree.ts`

Changes:

1. Publish disjoint regular, transition, physical-boundary, and
   transition-boundary worksets for every live structured velocity-family
   slot.
2. Reconstruct the prior projected and extended velocity at each destination
   face from opposing-family averages or the catalog pseudoinverse.
3. Backtrace the destination centroid and interpolate the transported
   full-vector field with the matching cube-trilinear or
   tetrahedral-barycentric basis.
4. Project once onto the destination family normal and write only that
   destination slot.
5. Apply gravity, authored forces, and solid constraints only after fine and
   coarse phi have been transported with the previous divergence-free field.
6. Delete any primary particle transport code, shader, buffer, counter, test,
   selector, or hidden compatibility fallback. Secondary presentation
   particles remain outside the solver graph.

Exit gate:

- no primary FLIP/APIC module or execution edge is reachable from the
  production octree method;
- constant and affine full-vector fields reproduce to tolerance across uniform
  and 2:1 regions;
- destination-owned advection is deterministic and contains no floating
  scatter;
- momentum and transport dissipation are reported for regular and transition
  worksets;
- secondary-particle rendering remains behaviorally separate and one-way.

### Phase 7 — next-epoch GPU topology pipeline

Files:

- `lib/webgpu-octree.ts`
- `lib/webgpu-fluid-brick-residency.ts`
- `lib/webgpu-octree-owner-pages.ts`
- `lib/webgpu-octree-fine-levelset-topology.ts`
- `lib/webgpu-octree-power-topology.ts`
- `lib/webgpu-octree-worksets.ts`

Changes:

1. Move recurring candidate construction from the beginning of the current
   substep to the tail of the prior substep.
2. Union interface blocks, solids, inflow, authored sizing fields, and
   persistent hysteresis.
3. Enforce grading with bounded mark/dilate passes over compact frontier lists.
   Enforce both ordinary 2:1 balance and the exclusive
   same/finer-or-same/coarser 1-ring invariant to closure.
4. Build page allocation, row rank, adjacency, resolved implicit power rows,
   velocity-family handles, class
   worksets, MG hierarchy deltas, and indirect args in the inactive epoch.
5. Validate counts, reciprocal adjacency, 2:1 balance, catalog coverage,
   coefficient positivity, and support closure in one reduction.
6. At the next substep, flip active epoch only when the validation word and
   expected generation agree. Otherwise retain the prior topology and expose a
   terminal reason code.
7. Replace host-selected CFL substep counts with a maximum encoded schedule and
   GPU-resident active flags, while preserving the exact displacement bound
   used to size the candidate support closure.
8. Apply temporal hysteresis to refinement/coarsening and retain protective
   refinement bands around the interface, moving solids, inflows, and authored
   regions.
9. For fine phi, rebuild the next compact pool from logical occupancy
   mark/rank/scatter, gather overlapping values from the current pool, and
   initialize newly exposed bricks. Do not run a recurring per-brick allocator.
10. Reclassify case IDs only for dirty pressure regions and their dependency
    ring; carry clean compact cases unchanged.

Exit gate:

- no recurring CPU topology decision or simulation-sized readback;
- unchanged topology performs no row/page rebuild work;
- a changing topology never becomes partially visible;
- no accepted topology contains a mixed finer/coarser relevant 1-ring;
- active page count scales with interface area × fixed support width;
- moving-interface endurance tests never outrun the prebuilt closure.

### Phase 8 — bandwidth and precision pass

Files:

- `lib/gpu-startup.ts`
- `lib/webgpu-renderer.ts`
- all octree WGSL modules touched above
- `tests/webgpu-device-limits.test.ts`

Changes:

1. Require `subgroups` for the production octree method. Reject `shader-f16`
   storage for authoritative and solver channels until an independent
   numerical experiment proves it; no f16 production module is retained.
2. Benchmark 64- and 128-lane candidates offline, select the measured M1 Max
   winner, and retain only that executable shader. The current gate selects
   128 lanes.
3. Pack eligible case IDs, local indices, masks, neighbor-presence fields, and
   tetrahedron selectors into `u32` words with measured unpack cost.
4. Keep pressure, fine-interface phi, face velocity, diagonals, finest
   residuals, cut coefficients, and reduction partials in f32.
5. Evaluate packed f16 for far-band phi, coarse MG residuals, and
   non-authoritative summaries. Every precision change lands independently
   behind differential gates.
6. Fuse divergence with RHS construction.
7. Fuse pressure-gradient projection with closest-point extension seeding.
8. Fuse mask generation with producers where the mask has one consumer.
9. Remove duplicate persistent velocity caches unless measured reuse exceeds
   on-demand reconstruction over the active workset.
10. Reuse scratch arenas across non-overlapping stages and keep all major fields
   SoA.

Exit gate:

- bytes read/written per active row/page are recorded for each hot stage;
- no precision change moves the zero crossing, residual, volume, or energy
  beyond its stated gate;
- unsupported subgroup configurations are rejected before allocation;
- every surviving cache names its consumers and demonstrates a measured win.

### Phase 9 — production cutover and deletion

Files:

- `lib/methods/octree.ts`
- `lib/webgpu-octree.ts`
- superseded octree modules and tests

Changes:

1. Make the GPU page/workset/resolved-row/Eulerian-face-advection/
   pipelined-MGPCG graph the sole production octree path.
2. Remove old runtime face/incidence graphs, general CSR assembly,
   capacity-shaped recurring dispatches, duplicate topology directories,
   priority-queue redistance, and retired solver selection code.
3. Keep small CPU mathematical oracles and the generated catalog tooling.
4. Delete prior executable WebGPU solver/operator paths in the cutover change.
   Keep only frozen data fixtures and small CPU mathematical oracles.

Exit gate:

- no compatibility fallback can silently revive the old path;
- all production quantities have one named authority;
- mini, UI, hydrostatic, moving-solid, open-top, factor-4, and factor-8 lanes
  pass;
- the M1 Max performance report demonstrates improvement by stage and active
  work, not only total wall time.

## Verification matrix

### Structural

- page allocation/free/carry and epoch publication;
- exact six/27-neighbor adjacency;
- workset prefix/scatter and zero-count indirect dispatch;
- exclusive same-or-finer/same-or-coarser face-and-edge 1-rings, with mixed
  rings rejected;
- exhaustive descriptor-to-template lookup;
- sparse-descriptor to compact dense-case-ID mapping;
- four disjoint pressure worksets with no cross-class rows;
- reciprocal face ownership and symmetric coefficients;
- six structured velocity families with no authoritative timestep face or
  incidence graph;
- destination-owned ghost propagation/accumulation and transfer adjointness;
- separate topology-epoch, per-substep geometry, and iterative-vector SoA
  groups;
- no hot-loop binary/hash lookup;
- no primary FLIP/APIC execution graph;
- no floating scatter into structured velocity families;
- no recurring domain/capacity dispatch.

### Numerical

- constant and affine velocity transfer;
- pseudoinverse reconstruction and regular-trilinear/transition-tetrahedral
  interpolation;
- divergence-free transfer across 2:1 transitions;
- matrix symmetry and positive energy;
- manufactured Poisson convergence;
- hydrostatic pressure with free/solid/world boundaries;
- pressure residual versus an independent CPU apply;
- level-set plane/sphere transport and redistance;
- factor-`m` fine backtrace with `m` velocity reinterpolations and one phi
  sample;
- fixed-schedule redistance and destination-owned layered extrapolation;
- zero-crossing, volume, and energy across repeated regrids;
- Eulerian face-transport momentum and dissipation across regular and
  transition regions.

### Lifecycle

- cold bootstrap;
- unchanged epoch;
- add/remove/carry pages;
- topology rejection with old epoch retained;
- moving interface crossing page and refinement boundaries;
- moving solid and inflow activation;
- factor-4 and factor-8 support closure;
- required M1 Max subgroup/limit contract and fail-closed rejection.

### Performance

- free-running wall time;
- timestamped stage time;
- scheduled versus active lanes;
- executed versus encoded solver/redistance/grading iterations;
- active pages versus logical pages;
- directory probes in hot kernels (target zero);
- geometry evaluations per solver iteration (target zero);
- float atomic operations in transfers (target zero);
- authoritative and scratch bytes;
- estimated bytes moved per active element.

## Definition of complete conformance

The work is complete only when all of the following are true:

1. Regular hot kernels iterate dense physical pages or compact worksets and use
   precomputed adjacency.
2. Power geometry and transition coefficients are generated offline and
   resolved to compact cases and fixed handles only on topology change; no
   general face graph or CSR row is a timestep authority.
3. Every accepted topology satisfies the exclusive same-or-finer or
   same-or-coarser face/edge 1-ring rule and maps to a compact dense case ID.
4. Regular interior, transition interior, physical boundary, and
   transition-plus-boundary work are separate indirect kernels.
5. Six structured velocity families and precomputed reconstruction,
   tetrahedral interpolation, and parent/child transfer tables cover every
   valid irregular case.
6. Production pressure is a matrix-free second-order operator using the
   symmetric damped-Jacobi shell / first-order Chebyshev-MG /
   damped-Jacobi-shell preconditioner, direct positive direction-curvature PCG,
   compensated f32 reductions, fixed V-cycles, destination-owned transfers, and a
   persistent GPU coarse solve.
7. Eulerian full-vector face advection is the sole momentum authority,
   destination-owned, deterministic, and free of floating scatter; no primary
   FLIP/APIC execution graph exists.
8. Topology is GPU-built into an inactive epoch and published atomically with
   no hot-path CPU readback.
9. The uniform fine surface grid uses compact A/B bulk rebuilds, factor-`m`
   multi-step characteristics, fixed GPU redistance, and destination-owned
   extrapolation; residency scales with interface support rather than volume.
10. SoA frequency groups, halo staging, moderate kernel specialization, the
    measured 128-lane M1 pipeline, measured fusion, selective integer packing,
    authoritative f32, and scratch reuse make bandwidth the explicit budget.
11. The old production representations and fallback paths are deleted at
   cutover; they do not remain as hidden alternate authorities.

## Implementation status and open findings

> **Superseded — see [`OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md`](./OCTREE_M1_MAX_IMPLEMENTER_HANDOFF.md).**
> A second audit later on 2026-07-26 found that the Section 6.3 operator layout,
> the solve tail, the fine-summary merge ladder, and the dead-module category
> below have all been closed. Two new items replaced them: a single-threaded MG
> candidate rebuild and an unreachable topology-delta comparison, plus five
> reproducible cutover-suite failures. The handoff carries the current, actionable
> list. The section below is retained as the record of the first pass.

Recorded 2026-07-26 from a static read of the working tree. No lane was
benchmarked for these findings: a `vinext dev` server and a Dawn smoke run held
the exclusive GPU lock, and `POWER_LIQUIDS_PERF_HANDOFF.md` requires no
concurrent session before timing. Every cost figure below is derived from
encoded dispatch counts and the measured link taxes in
`EXECUTION_MULTIPLIERS_HANDOFF.md`, not from a run.

Five items are open. Suggested order is 1 → 2 → 4 → 5 → 3.

### 1. Phase 0 exit gate is not met — no baselines exist

`docs/baselines/octree-regression/` and `artifacts/octree-regression/` are both
absent. Every other part of the evidence path is complete: `--artifact=` already
forces `FLUID_STABILITY_ENVELOPE` and `FLUID_PERFORMANCE_TRACES`, the four lane
capture scripts exist, and the comparator is implemented. Nobody has run a
capture and committed the result.

What closing this requires:

- four captures taken in a clean tree with no concurrent GPU session;
- zero `blockers` in each artifact. Two metrics need explicit confirmation
  rather than assumption: `energyRatio`, which is now sourced from the stability
  envelope rather than the deleted energy-ledger module, and
  `activeScheduledRatio`, which blocks the whole artifact if any stage still
  reports a `null` scheduled or active counter;
- the `mini` lane captured as the exact 500-step, 2.0-second run, which is the
  only form the contract accepts.

Done when four baselines with empty `blockers` are checked in and the comparator
runs green against a re-capture.

This is first because roughly twelve substantive changes have landed since the
last recorded numbers — band restriction, convergence gating, the transport
owner-map rewrite, coarse-Eikonal deletion, the tolerance clamp, direct axis
neighbour handles — and not one has an attributed measurement. Every other item
here is a judgement about where time goes, and none of them can be ranked until
this exists.

One decision to settle before the baselines are frozen: the artifact has no
field for the quantity the deleted energy ledger used to own. A baseline defines
what the project will be blind to, so if those per-stage energy taps were
load-bearing for catching dissipation regressions, that gap belongs in the
artifact first.

### 2. The encoded solve tail costs more than the solve

The convergence gate is complete and correct — `prepareCorrectionDispatches`
zeroes both the row dispatch and all five band-class records once stopped, so
there is zero lane work after convergence. The *launches* are not gated. The
encode loop runs the full budget of 16 iterations, each encoding 118 dispatches
(4 + 4 apply + 110 correction, where 110 is 16 sweeps × 5 plus the V-cycle's
26). At the current 1e-4 tolerance converging in roughly 5, that is about 1,300
zero-work launches, or ~4.7 ms of dependent-dispatch latency per advance.

The constraint is already established and rules out the obvious fixes: batching
cycles behind extra pass boundaries regressed on M1 (4-cycle 11.60 ms, 8-cycle
11.21 ms, against 10.88 ms direct). More indirect zeroing and more pass
structure are both dead ends. The encoded chain has to get shorter.

Three independent levers, by ratio:

- **Collapse the four band class-applies into one dispatch over a merged band
  workset.** The classes exist to avoid per-row branching, but the band is by
  construction the transition-and-boundary region, where the classes are nearly
  degenerate and the register-ceiling argument is weakest. Takes the correction
  count from 110 to roughly 50 and the tail to about 2 ms. Needs a measurement
  to confirm the branch cost does not consume the gain.
- **Derive the iteration budget from the previous frame's executed count**
  rather than fixing it at 16. The paper's envelope at this tolerance is 4–10
  iterations; the encoded cost of the worst case is currently paid every frame.
- **Re-tune the shell depth `k`.** Eight was chosen when the shell ran over the
  whole domain and has not been revisited since the band restriction landed. The
  paper uses about 8 for its band, 10 for thin river geometry, 3 elsewhere. This
  is a one-parameter sweep against convergence count and it multiplies directly
  into the 110.

Done when encoded pressure dispatches per advance are under ~600, executed
convergence is unchanged at the same residual target, and the wall-time change
is attributed on the mini and UI lanes.

### 3. The Section 6.3 operator layout is not the paper's

This is the deepest architectural gap and the one that decides scaling. It is
also smaller than it looks, because the generator half is finished.

`catalog.coefficientData` is exactly the paper's Section 6.3 table — diagonal
plus eighteen canonical face-slot coefficients per entry, 19 channels across
1,608 cases, bit-exactness asserted in `tests/octree-power-catalog-artifact.test.ts`.
It is uploaded as `catalogCoefficients` at
`lib/webgpu-octree-power-topology.ts:304` and exposed on the topology source at
`:407`. **It is bound to no pipeline.** Meanwhile the runtime rebuilds
equivalent information every epoch as up to 36 explicit neighbour row IDs plus
36 coefficients per row (`maximumNeighborRows: 36`) and applies the operator as
random-index gathers. The task is to consume a table that already exists and
passes its tests, not to implement Section 6.3.

The ghost machinery also already exists, at the wrong level: the first-order
V-cycle carries `restrictAndGhostAccumulate`, `prolongAndGhostPropagate`, and
`SPGRID_CELL_FLAG.ghost`, with validated adjoint transfers. The paper's point is
that the same mechanism carries the cross-level spokes for the second-order
operator, which is what caps a fine-level row at 19 coefficients over same-level
neighbours.

What closing this requires:

- **a decision about where cross-level spokes live.** They are in the row today;
  the paper puts them in the ghost transfer. Adopting the paper's answer is what
  caps the row at 19 coefficients and makes neighbour addresses implicit, which
  is the entire bandwidth argument. Everything else follows from this one choice;
- **same-level neighbour addressing that is derivable rather than stored.** The
  paper streams because a neighbour's address is an offset, not an index to be
  loaded and chased. The current flat `(level, morton)` row array cannot provide
  that; whatever replaces it must make "my +x neighbour" computable from a row's
  own identity within its page;
- **adjointness and symmetry preserved across the change.** SPD is guaranteed
  today by resolving each internal coefficient once from a canonical owner and
  having both incident rows reference it. A ghost-transfer formulation must
  establish the same property through the propagate/accumulate pair being exact
  adjoints, as the first-order path already proves for itself;
- **a differential gate against the current operator** on the same published
  topology before the resolved-row path is retired, per the Phase 3 exit
  criterion.

This touches the resolved-row ABI, the four class-apply kernels, the band
worksets, and the L2-to-L1 interaction. It is the largest piece of work
remaining and it deserves its own plan rather than being folded into this
cutover.

It will also not show up in any lane currently run. At 1,372 rows the whole set
fits in cache and gather-versus-stream barely registers. The paper's own
measurement of this exact difference is 8× — 3.95 GB/s optimized against
0.49 GB/s for the explicit-mesh variant, Table 1 — and it appears when the row
set stops fitting in cache. The honest framing is that this is the difference
between correct at 16³ and the paper's envelope at 512³. A lane large enough to
make it measurable should be added alongside the work, or it will be done with
no way to prove it paid.

### 4. The fine summary merge ladder needs replacement, not deletion

The summary directory is not diagnostics. `fineSummarySizingGroup` is bound into
octree refinement sizing and frontier classification at `lib/webgpu-octree.ts`
lines 2295, 2298, 2331, 2354, and 2365. It is a physics authority, and deleting
it changes topology.

The cost is unchanged: `sortFineSummaryTiles`, an N-deep merge ladder, three
binary searches (`recordLowerBound`, `committedFineAt`, and the merge
partitions), and roughly seven uncached bind groups per advance.

What closing this requires:

- **an enumeration of the consumers and what each one actually needs** — a
  per-tile min/max, a nearest-interface distance, an occupancy count. Almost
  certainly not the full sorted record stream;
- **a representation chosen from that requirement rather than from the current
  shape.** A rank-indexed mip over the active set answers sizing queries
  directly with no sort, no merge ladder, and no search. That was already the
  Section 5.4 conclusion; the missing piece was the consumer list justifying it;
- **bind-group caching regardless**, as already applied to the coarse level set
  and topology paths.

Done when every consumer is named and pointed at a representation answering its
query in O(1), no sort or merge remains in the recurring path, and a differential
run produces identical topology decisions.

Because this feeds refinement it carries real regression risk, so it should land
after the baselines exist rather than before.

### 5. Four modules are unreachable, and the phase list reads as complete

| Module | Status | Phase intent |
| --- | --- | --- |
| `lib/webgpu-octree-topology-epoch.ts` | 0 importers in `lib/` | Phase 7 fail-closed A/B epoch flip |
| `lib/webgpu-octree-structured-velocity.ts` | 0 importers in `lib/` | Phase 6 planner; runtime is the `-gpu.ts` variant |
| `lib/webgpu-octree-page-pool.ts` | constants only, via brick-stencils | Phase 1 common page/workset ABI |
| `WebGPUOctreePressureMGPageChebyshev` | never constructed | Phase 4 halo-staged page-resident smoother |

All four pass their unit tests, which is why the phase list reads green.

Each needs a separate decision, because the correct answer differs and depends
on whether production covers the intent by other means:

- **`topology-epoch.ts`** — the production flip is `ownerPages.encodeReadyCommit`,
  which validates and publishes both authority headers atomically. Either that
  satisfies the Phase 7 contract and this module is redundant, or it does not
  and this module is the specification production should adopt. Determine which.
  Leaving it in place is the one option that is wrong under either answer.
- **`structured-velocity.ts`** — a planner whose runtime counterpart diverged.
  Either it is the oracle `-gpu.ts` is validated against, in which case it needs
  a differential test saying so, or it is stale.
- **`page-pool.ts`** — production reached the Phase 1 outcome by other routes,
  through direct logical-to-physical directories and brick-mask-plus-rank
  lookup. The unified ABI never materialized. Closing this as "intent met
  differently" is reasonable, but it should be recorded rather than left
  implicit.
- **`PressureMGPageChebyshev`** — genuinely unfinished work rather than
  redundancy. The first-order V-cycle has its own page-resident Chebyshev; this
  second one is unwired. It is also downstream of item 3, since a page-resident
  second-order smoother is only meaningful once rows are page-addressable, so
  this decision belongs in that plan.

Done when every module in `lib/` is either reachable from production or is a
named oracle with a test comparing it to production. There is no third category.

### Minor

The `unbounded-lookup` source guard in `lib/webgpu-octree-work-accounting.ts`
matches only `while (true)` and `for (;;)`. It does not match WGSL `loop {}`,
which is the form the shaders actually use.

### Verified closed this round

Recorded so the next audit does not re-derive them: band-restricted second-order
applies via `bandWorksets`; convergence gating of both row and band dispatch
records; direct six-axis neighbour handles replacing the O(slots) scan in
structured dynamics; live-row indirect dispatch sizing in the pipelined MGPCG;
the pressure tolerance clamp to 1e-4; alias-following capacity-dispatch source
guards; deletion of the eight coarse Eikonal passes; the transport owner-map
epoch cache and direct `ownerAtPosition`; bind-group caching on the coarse
level-set path; and the paper's barycentric transition interpolation now
consuming the sampling catalog.
