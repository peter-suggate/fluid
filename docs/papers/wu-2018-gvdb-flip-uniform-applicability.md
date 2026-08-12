# Wu et al. 2018 GVDB FLIP: applicability to the uniform GPU method

> Scope note: this is a supplemental analysis of a constant-resolution sparse
> backend. For the paper's applicability to the activity-adaptive
> multiresolution goal and its current uniform-region successor, see
> [wu-2018-gvdb-flip-activity-adaptive-challenge-analysis.md](./wu-2018-gvdb-flip-activity-adaptive-challenge-analysis.md).

Date: 2026-08-12

Source: [Wu, Truong, Yuksel, and Hoetzlein, *Fast Fluid Simulations with
Sparse Volumes on the GPU*](./wu-2018-gvdb-flip.pdf), EUROGRAPHICS 2018,
Computer Graphics Forum 37(2), DOI
[10.1111/cgf.13350](https://doi.org/10.1111/cgf.13350). A layout-preserving
text extraction is stored at [wu-2018-gvdb-flip.txt](./wu-2018-gvdb-flip.txt).

Repository snapshot: `c4b861c`, plus the uncommitted uniform-pressure work that
was present on 2026-08-12. That work includes residual-gated cycles, in-place
pressure/minimum storage, fused descent and smoothing experiments, projected
SOR, and a workgroup-resident coarse hierarchy. It is important not to confuse
"workgroup-resident" coarse data with spatially sparse GVDB residency.

## Executive conclusion

The paper is **strongly applicable as a systems design for a new
sparse-uniform backend**, **weakly applicable as a numerical method**, and
**inappropriate as an in-place redefinition of the existing `uniform`
method**.

The useful idea is not "adopt FLIP" or "replace CM11a with CG." It is:

> Keep one globally aligned, constant-resolution Cartesian lattice, but store
> and dispatch only fixed-size pages that are required by liquid, stencils,
> characteristics, inflows, and moving solids.

That would preserve uniform cell geometry while removing the bounding-box tax.
It should be introduced as a separate experimental method, tentatively
`sparse-uniform`, because the current method is intentionally a dense,
matched-lattice oracle. `docs/UNIFORM_GPU_REFERENCE.md:5-12` and
`lib/methods/uniform.ts:53-83` make dense storage and the absence of adaptive
topology part of its contract. Making that method sparse would eliminate the
independent baseline used to detect page, seam, and topology errors.

The recommendation is therefore:

1. Keep `uniform` dense and unchanged as the authority.
2. First add a **shadow page-residency census** to the dense method. It should
   compute required pages and worklists without changing any reads or writes.
3. Proceed to a separate sparse-uniform prototype only if the full-stage
   support closure, not merely the wet set, remains materially sparse on the
   target scenes.
4. Reuse the paper's dense-brick/worklist/incremental-lifecycle concepts, but
   retain this repository's surface-density transport, MAC ownership,
   cut-cell/ghost-fluid operator, CM11a LCP multigrid, moving-solid coupling,
   and residual gates.
5. Do not adopt the paper's particle-triggered topology, particle-to-grid
   subcell lists, unpreconditioned CG, CUDA warp assumptions, or one-cell apron
   as universal design decisions.

An overall applicability score is useful only if the dimensions are kept
separate:

| Dimension | Assessment | Why |
| --- | --- | --- |
| Sparse storage and work scheduling | High | Directly attacks the dense 3D allocation and full-grid dispatch tax |
| Incremental topology lifecycle | High, after adaptation | The reuse principle transfers; particles cannot be the activity oracle here |
| Dense bricks and compute subcells | High | Matches regular stencil kernels and the existing 4x4x4 workgroup shape |
| Apron voxels | Medium | Useful for some kernels, expensive and insufficient for others |
| Matrix-free pressure | Already present in stronger form | The current operator is matrix-free/baked-coefficient LCP multigrid, not a stored sparse matrix |
| Paper's unpreconditioned CG | Low/negative | Hundreds of iterations and no support for the current separating-boundary LCP contract |
| FLIP particles and particle rasterization | Not applicable | The uniform method is Eulerian and has no particle authority |
| Direct modification of `uniform` | No | It would destroy the dense reference lane |
| Separate sparse-uniform method | High-potential, evidence-gated | Preserves the oracle and isolates topology risk |

## 1. What the paper actually contributes

The paper combines several ideas that should not be treated as one indivisible
method.

### 1.1 Sparse GVDB storage

GVDB represents a large logical voxel domain with a hierarchy whose leaves
point to dense 3D bricks in a texture atlas (paper Section 2.2 and Figure 2).
The evaluated configuration uses large bricks for storage and 4-cubed logical
subcells for GPU work. Results describe 32-cubed bricks; the introductory GVDB
configuration illustrates 16-cubed leaves. Brick size is therefore an
implementation parameter, not a theorem.

The important separation is:

- **logical address space:** potentially very large;
- **physical storage:** only allocated bricks;
- **compute granularity:** smaller subcells within a brick; and
- **neighbour access:** cached apron values around the brick.

This separation is more applicable than the exact GVDB tree.

### 1.2 Full and incremental topology construction

For a full rebuild, the paper emits a `(level, x, y, z)` key for every particle
at every required tree level, radix-sorts the keys, marks unique values,
prefix-sums them, allocates nodes, and then links children (Section 5.1 and
Figure 3). A second pass adds bricks touched by each particle's influence box.

For incremental rebuilds, it emits only missing nodes and marks/removes nodes
that are no longer required (Section 5.2). Physical pools retain released
storage for reuse. The paper reports incremental topology updates 80-179x
faster than its CPU rebuild and substantially faster than its full GPU rebuild
(Table 1).

The transferable result is **stable pooled storage plus incremental set
difference**, not the particle-key producer or the exact tree.

### 1.3 Subcells and gather rasterization

The paper divides storage bricks into 4x4x4 subcells. It builds compact lists of
particle positions and velocities overlapping each subcell, then lets a
workgroup gather from a coherent local list rather than scatter particle values
with conflicts (Sections 6-7 and Figure 4). The subcell list is also the largest
transient memory consumer in several results.

The uniform method has no particles, so the list contents do not transfer. The
**two-granularity architecture** does: use a page large enough to amortize
directory metadata and a smaller work tile sized for GPU execution.

### 1.4 Sparse matrix-free CG

The paper applies a seven-point pressure matrix by reading solid/fluid/empty
voxel classifications and six neighbours on demand. It updates the search
direction apron inside CG and uses hierarchical reductions for inner products
(Section 8, Figures 5-6, Algorithm 2).

This demonstrates that an elliptic solve can operate over sparse bricks. It
does not establish that the paper's solver is a good replacement for the
current one. Table 2 reports 229-511 CG iterations for the dam-break ladder and
243-456 for the water-drop ladder. The authors explicitly list a sparse
multigrid preconditioner as future work.

### 1.5 End-to-end GPU residence and sparse rendering

Topology construction, particle transfers, projection, advection, and
raytracing remain on the GPU. This is directionally aligned with the repository,
which already keeps its simulation-sized authority on the GPU and uses
asynchronous diagnostics rather than host scheduling.

## 2. What the current uniform method is

The current method is not a basic dense FLIP solver. It is a dense Eulerian
reference implementation with a more demanding numerical contract:

- a scene-authored finest lattice in all three axes;
- cell-centred conservative surface density `rho` and persistent `gamma`;
- packed positive MAC faces plus separate negative boundary faces;
- current and predicted velocity extrapolation using a causal FIM front and a
  hierarchy fill;
- conservative density scatter/gather, seven ordered gamma-diffusion
  iterations, local sharpening, and partial-solid excess redistribution;
- selectable one-pass semi-Lagrangian or bounded MacCormack velocity transport;
- moving rigid bodies, terrain, cut-cell apertures, ghost-fluid free surfaces,
  surface tension, and open/closed domain boundaries;
- the CM11a separating-boundary LCP multigrid hierarchy; and
- optional presentation-only density reconstruction.

The stage order is visible in `lib/webgpu-uniform-reference.ts:816-972`.
The numerical contract is documented in `docs/UNIFORM_GPU_REFERENCE.md:14-115`.

The current allocations are also much heavier per cell than the paper's phrase
"thirteen channels" might suggest. The constructor creates four
`rgba32float` velocity textures, two haloed `rgba32float` transport textures,
six haloed `rgba32float` FIM value/distance textures, eight scalar textures,
three fixed-point conditioning fields, boundary-face buffers, and the complete
pressure pyramid (`lib/webgpu-uniform-reference.ts:312-382`). The pressure
hierarchy currently owns pressure, two rhs fields, two phi fields, two
`rgba32float` topology fields, a minimum, baked `rgba32float` coefficients,
and a finest backup (`lib/webgpu-uniform-pressure-multigrid.ts:172-214`).

The paper therefore addresses a real problem in this method: every extra
physical channel multiplies the cost of empty domain space.

## 3. Contribution-by-contribution applicability

| Paper element | Verdict for the uniform method | Required adaptation |
| --- | --- | --- |
| Sparse hierarchy over a large logical domain | Adopt concept | Prefer a page directory/two-level map over a literal pointer tree for hot stencil lookup |
| Dense voxel bricks | Adopt | Benchmark 8, 16, and 32 interior widths; do not inherit the paper's result blindly |
| 4x4x4 compute subcells | Adopt as a starting point | It matches the current 64-invocation workgroups, but must not assume a 32-lane CUDA warp |
| 3D texture atlas | Evaluate, not default | Current formats are often unfilterable and already use manual reconstruction; buffer page pools may be simpler |
| Persistent one-cell aprons | Selective | Good for radius-one stencils; use direct directory reads or stage-specific packed halos elsewhere |
| Full topology rebuild from particles | Reject | There are no particles; dense-field or analytic producers must mark required pages |
| Incremental topology reuse | Adopt | Stable physical page IDs, candidate/accepted generations, hysteresis, and fail-closed capacity |
| Particle influence boxes | Adapt | Replace with characteristic reach, FIM reach, inflow support, solid swept bounds, and stencil closure |
| Subcell particle lists | Reject | No particle transfer stage exists |
| Gather rather than scatter | Mixed | Current density/sharpening intentionally use fixed-point scatters for conservation; page-local gathers cannot silently change the operator |
| Matrix-free seven-point pressure | Preserve principle, reject exact stencil | Keep the current cut-cell/ghost-fluid/LCP coefficients and global solve |
| Unpreconditioned CG | Reject | Retain CM11a multigrid and its residual acceptance |
| Sparse reductions | Adopt where measured useful | Reduce over active page/subcell worklists, then perform a compact final reduction |
| GPU-only pipeline | Already satisfied | Topology and page publication must also remain GPU-driven in steady state |
| Direct sparse-volume raytracing | Potentially adopt | Requires a sparse-fluid renderer source or a bounded dense presentation cache |
| 16-bit particle packing | Not applicable | There is no particle payload |
| Static collision volume | Insufficient | Current moving solids must preactivate and update affected pages |

## 4. The architectural boundary: do not make the oracle sparse

The dense method answers a specific experimental question: what does the same
finest lattice produce when every cell is present? A spatially sparse method
answers a different question: can page lookup, activation, retirement, and seam
handling reproduce that result for less work?

Those questions need two independent implementations. Otherwise a missing
page can make both the simulated result and its supposed reference omit the
same state.

Recommended method split:

| Method | Spatial resolution | Storage | Role |
| --- | --- | --- | --- |
| `uniform` | Constant, scene-authored finest | Dense bounding box | Numerical and seam-free oracle |
| `sparse-uniform` | The same constant cell size | Sparse fixed-size pages | GVDB-inspired experiment/product path |
| adaptive Losasso/Power | Multiple cell sizes | Sparse adaptive topology | Resolution adaptivity research/product path |

The sparse-uniform method would be **sparse but not adaptive**. A page being
absent changes storage and work, not physical cell size or the discrete
equations on resident cells.

## 5. Proposed sparse-uniform data model

### 5.1 Logical coordinates and page directory

Every logical cell retains its global integer `(i, j, k)`. A page key is the
integer floor division by page width. The page directory maps that key to a
stable physical page ID for the accepted generation.

For bounded current scenes, a dense logical-page directory is a reasonable
first implementation. For the paper's "virtually unbounded" claim it is not:
a dense directory still scales with the bounding domain even if payload pages
do not. A production extension would need either:

- a two-level directory with sparse second-level slabs;
- a compact sorted key/page table plus a small hash accelerator; or
- a GPU hash table with deterministic overflow/failure semantics.

The hot path should not traverse a general tree for every neighbour. The paper
uses aprons to avoid that cost; the repository already demonstrates another
option: a direct logical-brick-to-physical-page directory in
`lib/webgpu-octree-owner-pages.ts:201-207`.

### 5.2 Page pools should be structure-of-arrays

Do not build one giant per-page struct containing every channel. Different
stages need different support domains:

- `rho`/`gamma` and conditioning deposits need conservative transport support;
- velocity needs characteristic and extension support;
- pressure/topology/coefficient fields need all liquid rows plus a stencil
  closure at every multigrid level;
- render-only surface fields need an interface band; and
- solid/terrain data may be immutable, analytic, or separately paged.

Separate page pools let a render-only halo avoid allocating pressure scratch,
and let deep-liquid pressure pages avoid allocating surface post-processing.
The simplest prototype may use one conservative union residency set, but the
telemetry must expose how much that simplification costs.

### 5.3 Face ownership

The current packed MAC convention stores positive x/y/z face values with the
lower-coordinate cell and stores the three negative domain slabs separately.
That convention can remain sparse:

- a cross-page interior face is owned by the lower-coordinate logical cell;
- both adjacent divergence rows read that same value;
- projection writes it exactly once; and
- page retirement cannot delete it while either adjacent cell still needs it.

Duplicating face values into both pages without one authoritative owner would
create exactly the mass and pressure seam that the dense oracle is meant to
detect.

### 5.4 Missing values are field-specific

A generic "missing means zero" rule is incorrect. At minimum:

| Field | Implicit missing value or action |
| --- | --- |
| `rho` / conservative density | zero, only outside proven support |
| `gamma` | one in untouched air, matching current initialization |
| pressure | zero only for legitimate Dirichlet air/exterior, never for a missing liquid row |
| velocity | boundary-, solid-, or inflow-dependent; not universally zero |
| pressure minimum | `-FLT_MAX` for unconstrained rows, zero for the relevant solid/domain constraints |
| topology/aperture | derive from analytic boundary/solid authority or require residency |
| FIM distance | infinity for unknown, zero for seeded faces |

New pages must be initialized from these rules before any consumer sees the
new generation.

## 6. Residency production without FLIP particles

Particles give the paper an external set of points that can request storage
before grid state exists there. A pure Eulerian sparse method has a bootstrapping
problem: an absent cell cannot inspect its own field and request activation.

The required page set for an accepted step must be the union of:

1. every page containing nonzero conservative density;
2. the donor and receiver footprint of the accepted density trace;
3. backward and forward velocity-characteristic footprints, including the
   midpoint samples used by RK2;
4. the current and predicted FIM front plus hierarchy support;
5. pressure rows for every connected liquid cell and their face-neighbour
   classification support at every multigrid level;
6. sharpening, solid-excess, and gamma-diffusion neighbours;
7. authored inflow support before density arrives;
8. the swept bounds of moving rigid bodies and their coupling quadrature;
9. open/closed domain boundary support; and
10. presentation support if rendering reads directly from the sparse fields.

For a characteristic stage, a conservative cell reach is

`ceil(maximum sampled speed * dt / minimum cell spacing) + interpolation radius`.

The current repository already implements useful pieces of this policy in the
adaptive path. `lib/webgpu-fluid-brick-residency.ts:530-557` widens support by
`|v| dt`, adds a safety factor, and preserves hysteresis; its downstream
expansion activates a face-adjacent brick into which velocity points. Those
ideas can be generalized, but the uniform method should not import octree
physics or make an 8-cubed owner page an unquestioned ABI.

Topology publication should be transactional:

1. build candidate page keys and stage worklists;
2. sort/deduplicate and allocate stable IDs;
3. initialize new pages and carry unchanged pages;
4. validate capacities, references, and generation counts;
5. publish directory, fields, and worklists atomically; or
6. keep the previous accepted generation intact on any failure.

## 7. Aprons, page size, and break-even

### 7.1 Persistent apron cost

With a one-cell apron on all sides, the stored samples per interior sample are:

| Interior width | Stored width | Ratio |
| --- | --- | --- |
| 8 | 10 | 1.953x |
| 16 | 18 | 1.424x |
| 32 | 34 | 1.199x |

With a two-cell apron the ratios become 3.375x, 1.953x, and 1.424x. This cost
multiplies every apron-backed channel. For the current method's many velocity,
FIM, density, and pressure fields, a universal apron can erase the sparse win.

The corresponding idealized one-cell-apron break-even resident fractions are
about 51%, 70%, and 83%, before directory, topology, worklist, fragmentation,
and update costs. A practical threshold must be materially lower.

### 7.2 Direct lookup versus cached apron

Use a per-stage choice:

- direct directory reads for globally coupled, radius-one stencils when page
  size is small and face ownership matters;
- workgroup-shared tile halos for smoothing/diffusion;
- packed stage-specific halos for repeated read-heavy kernels; and
- persistent aprons only when measurements show that their update and memory
  cost is lower than lookup cost.

The paper's apron result is evidence that cached neighbours can win on a Quadro
GP100 texture atlas. It is not evidence that persistent aprons win for WebGPU
buffers, Apple GPUs, unfilterable formats, or 8-cubed pages.

### 7.3 Current-scene initial occupancy audit

The following local audit reproduced the dense method's t=0 wet-cell
initialization, marked every page containing a wet cell, then conservatively
dilated that set by one complete 26-neighbour page ring. `core -> halo1` is a
percentage of logical pages. It is **not** a performance result and the dilation
is deliberately more conservative than a one-cell apron. It shows how quickly
page granularity and support closure can consume nominal sparsity.

| Scene | Grid | Wet cells | B=8 core -> halo1 | B=16 core -> halo1 | B=32 core -> halo1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Water-box dam | 24x16x16 | 23.4% | 33.3% -> 100% | 50% -> 100% | 100% -> 100% |
| Water-box tank fill | 24x16x16 | 25.0% | 50% -> 100% | 100% -> 100% | 100% -> 100% |
| High-resolution dam | 128x128x128 | 36.0% | 36.6% -> 47.3% | 39.1% -> 56.3% | 56.3% -> 100% |
| Ocean seiche | 320x96x80 | 75.4% | 75.4% -> 84.0% | 83.3% -> 100% | 100% -> 100% |
| Garden dam migration | 120x40x88 | 0.1% | 0.1% -> 3.3% | 0.7% -> 18.8% | 4.2% -> 75.0% |
| Garden hose | 120x40x88 | 0.2% | 2.3% -> 9.9% | 5.6% -> 33.3% | 16.7% -> 75.0% |
| CM12 Figure 9 dam | 128x128x64 | 23.4% | 23.4% -> 30.5% | 28.1% -> 43.8% | 37.5% -> 75.0% |
| Minimal dam 64 | 64x64x64 | 36.0% | 39.1% -> 56.3% | 56.3% -> 100% | 100% -> 100% |
| Large-power dam | 64x20x64 | 1.8% | 4.2% -> 14.1% | 3.1% -> 25.0% | 25.0% -> 100% |
| Symmetric expansion | 32x16x32 | 12.5% | 12.5% -> 100% | 100% -> 100% | 100% -> 100% |

Implications:

- garden migration/hose and large empty-domain dams are excellent candidates;
- the 128-cubed dam is plausible with 8- or 16-cubed pages;
- the 64-cubed mini-dam is a useful parity test but a weak performance target;
- the deep ocean is a negative control because pressure must retain the wet
  bulk, not only the surface; and
- no single page width wins every scene.

The shadow census must repeat this measurement at runtime peak, separately for
each stage support set. Initial wet occupancy is not sufficient.

## 8. Memory opportunity

A local allocation model based on the current constructors gives the following
approximate dense footprint for cubic grids. It includes the host allocation
plan, the current in-place pressure hierarchy, its finest backup, and the FIM
down/up hierarchy. It excludes device row/alignment padding, rigid-system
storage, audit-only fields, pipeline objects, and renderer allocations.

| Grid | Approximate dense fields | Bytes per logical cell |
| ---: | ---: | ---: |
| 32 cubed | 11.7 MiB | 374 |
| 64 cubed | 87.4 MiB | 350 |
| 128 cubed | 676 MiB | 338 |
| 256 cubed | 5.19 GiB | 332 |
| 512 cubed | 41.2 GiB | 330 |

The estimate reflects the actual current constructor, which allocates the
MacCormack-sized A/B/C/D and transport resources even when one-pass
semi-Lagrangian transport is selected. The public `allocatedBytes` value does
not currently add `WebGPUUniformVelocityExtrapolator.scratchBytes` for the
down/up hierarchy, so UI telemetry is slightly lower than this physical model.

At large sizes, the asymptotic cost is roughly 330 bytes per logical cell
before rendering. This makes sparse storage strategically relevant even if
page lookup slows individual samples: a dense 256-cubed field set is already
multi-gigabyte.

The paper's Table 4 is not a direct predictor. Its peak totals range from
900 MB to 9.7 GB and include particles, subcell lists, topology construction,
collision objects, and thirteen GVDB channels. In the 65-million-particle
512-cubed dam result, subcell lists consume 2.53 GB, particle data 1.49 GB,
topology-build scratch 0.99 GB, and GVDB channels 4.66 GB. Our method avoids
all particle/list cost but has a larger Eulerian per-cell state. The useful
comparison is qualitative: empty-domain channel storage dominates once many
fields are carried.

## 9. Stage-by-stage migration analysis

### 9.1 Initialization and page discovery

Applicability: **high after replacing the producer**.

The paper's particle-centre and influence-box passes should become analytic/dense
initial-condition scans at t=0 and field-derived swept scans thereafter.
Terrain and authored inflows can publish page keys directly. The full sort,
deduplicate, allocation, and linking pattern remains useful.

### 9.2 Velocity extrapolation

Applicability: **very high for work compaction, high risk for support**.

The current FIM already uses a GPU active counter and indirect dispatch, but
while the front is live each update covers the entire dense domain
(`lib/webgpu-uniform-velocity-extrapolation.ts:304-367`). A compact list of
active 4-cubed subcells is the closest direct analogue to the paper's subcell
work scheduling.

Requirements:

- active faces must enqueue destination subcells/pages without races;
- the source distance/value epoch must be coherent across page seams;
- newly reached pages must already exist; the FIM cannot allocate a page after
  the same pass has attempted to write it;
- the hierarchy fill needs a sparse parent closure or a dense coarse fallback;
  and
- current and predicted extrapolation may require different support because
  they see density at different stages.

This is likely the first simulation stage where sparse worklists show a clear
compute win without changing the physical equations.

### 9.3 Conservative density and gamma transport

Applicability: **high, but conservation is the gate**.

The trace, deficit scatter, gather, paired gamma diffusion, sharpening scatter,
and solid-excess scatter all cross prospective page seams. Every donor/receiver
pair must use one logical destination and one fixed-point deposit address.

Activation must include all trace destinations before the conditioning buffer
is cleared. A deposit to a missing page is not a small visual error; it destroys
mass. Retirement must wait until no conservative field, pending deposit, or
next-step trace needs the page.

The paper's gather lesson is useful for contention analysis, but converting
the existing fixed-point scatter/gather into a different gather operator would
change numerical results. First sparse the current operator; optimize it only
after dense parity exists.

### 9.4 Velocity advection and forces

Applicability: **high**.

Semi-Lagrangian RK2 and MacCormack require arbitrary trilinear samples across
page boundaries. Because the current shader already reconstructs samples
manually from unfilterable textures, a buffer page accessor does not forfeit a
large hardware-filtering advantage. It does add several directory lookups per
sample, so caching the departure stencil's pages or staging page tiles is a
measurement target.

Surface tension, terrain, wall mode, inflows, and moving-solid velocities must
use the same authority as dense execution. A sparse backend should share pure
geometry helper logic where possible, not copy a simplified paper boundary
model.

### 9.5 Pressure setup and projection

Applicability: **high for sparse rows, low for the paper's solver**.

The paper validates matrix-free sparse neighbour access. The current solver
already reconstructs the operator from baked positive-face coefficients and a
liquid flag (`lib/webgpu-uniform-pressure-multigrid.wgsl.ts:45-48,110-126`).
The sparse task is therefore storage/work compaction, not a change from a
stored matrix.

Do not solve each brick independently. Pressure projection is globally coupled
over each connected liquid component. A page seam must be algebraically
invisible, and the global projected residual must control acceptance.

A credible sparse CM11a hierarchy is:

1. active fine pressure pages for all liquid rows plus one face-neighbour
   classification closure;
2. a parent-page closure at every multigrid level;
3. direct cross-page reads for red/black smoothing or a halo refresh between
   colour dependencies;
4. sparse restriction/prolongation over active parent/child page pairs;
5. densification once a coarse level is small enough; and
6. one global residual reduction and cycle gate.

The current dirty-worktree pressure program is orthogonal and complementary:
in-place storage, fused descent, resident coarse levels, projected SOR, and
residual-gated cycles reduce work *within a dense hierarchy*. Stabilize and
measure those changes independently before using their output as the sparse
reference. `FLUID_UNIFORM_FIXED_SCHEDULE=1` should remain the control arm.

### 9.6 Rigid bodies and terrain

Applicability: **paper coverage is insufficient**.

The paper voxelizes collision geometry into another sparse volume and lists
moving rigid objects and better level-set boundaries as future work. The
repository already has moving bodies, cut-cell fractions, boundary velocities,
two-way impulses, and partial-solid excess redistribution. Solid swept bounds
must preactivate pages before face geometry or coupling is evaluated.

Static terrain can remain a separate 2D heightfield, which is already much
cheaper than paging it as 3D solid voxels.

### 9.7 Diagnostics and reductions

Applicability: **high and low risk**.

Page/subcell partial reductions followed by a compact final reduction are a
natural replacement for full-domain atomic diagnostics. Required global values
include conservative volume, represented volume, maximum speed, front extent,
pressure residual, FIM terminal activity, capacity errors, and page-generation
health.

Diagnostics must report logical missing-page errors separately from numerical
zeros. Otherwise a topology failure can look like good volume or residual
telemetry.

### 9.8 Presentation

Applicability: **medium**.

The paper raytraces GVDB directly. The current renderer expects dense
renderer-facing textures. Options are:

- add a sparse-fluid sampling source to the renderer;
- publish a dense texture only for a bounded visible/active presentation box;
  or
- keep presentation dense during early physics migration.

Presentation-only Section 3.8 fields are a good first sparse payload because
they do not feed simulation. They are not a useful final benchmark if dense
simulation allocations still dominate.

## 10. Pressure-specific correctness traps

1. **Global connectivity:** a long liquid component spanning many pages remains
   one pressure system.
2. **Null spaces:** closed fully-Neumann components require the same gauge/null
   treatment across page split/join events.
3. **Ghost-fluid theta:** both orientations of a liquid-air face must read the
   same coefficient; missing air support cannot silently imply theta=1.
4. **Separating solid bounds:** `p_min` is part of the LCP. It cannot be dropped
   because the paper's CG solves an unconstrained SPD system.
5. **Cross-page red/black ordering:** cached halos must be refreshed at the
   dependency boundary or the smoother becomes block Jacobi/Schwarz. That may
   be acceptable only as an explicitly re-blessed numerical change.
6. **Coarse coverage:** restricting only the interface band is wrong for deep
   liquid pressure. Parent closure must include the complete wet bulk.
7. **Residual reduction:** acceptance is over all active rows, not a per-page
   maximum that omits failed or missing pages.
8. **Generation coherence:** coefficients, rhs, minimum, pressure, and the page
   directory must refer to the same topology epoch.

## 11. WebGPU-specific design choices

### 11.1 Buffer pages versus a texture atlas

Start with buffer-backed structure-of-arrays pages unless a measured kernel
requires texture sampling:

- current `rgba32float`/`r32float` paths are frequently unfilterable;
- manual interpolation already exists;
- buffers support integer atomics needed by conservative deposits;
- stable physical page IDs make offsets simple; and
- a directory lookup is explicit and testable.

A texture atlas may still win for read-heavy velocity sampling or presentation,
but it introduces apron maintenance, atlas packing, format constraints, and
copy/update complexity. It should be an A/B backend, not the starting premise.

### 11.2 Workgroups and subgroups

The paper chooses 4-cubed subcells partly because 64 threads are two 32-thread
CUDA warps. In WebGPU, 64 invocations are portable but subgroup width is not.
Correctness must use workgroup barriers/reductions and tolerate different
subgroup sizes.

### 11.3 Indirect dispatch and readback

The repository already uses GPU-written indirect args for FIM and pressure
cycle gating. Sparse page worklists should use the same pattern. Host readback
may observe telemetry asynchronously, but it must not decide current-frame
residency or convergence.

### 11.4 Capacity is part of correctness

WebGPU resources are preallocated. A sparse system therefore needs explicit
page, key, worklist, and adjacency capacities. On overflow it must keep the last
accepted generation or fall back to dense execution; partial publication is not
permitted.

## 12. Performance interpretation of the paper

The paper's reported numbers establish feasibility, not a speedup forecast for
this repository:

- hardware is a 2018-era Quadro GP100 and the comparison CPU is a four-core,
  eight-thread i7-6700K;
- dense CPU comparisons cover the domain while the GPU path is sparse, which
  intentionally combines representation and hardware advantages;
- GPU pressure totals remain 146 ms to 3.475 s in Table 2 because CG takes
  hundreds of iterations;
- Table 3 frame times range from 179 ms to 3.79 s, not real-time; and
- the paper's largest memory costs are often particle subcell lists, which do
  not exist here.

The relevant claims to reproduce locally are instead:

1. stored bytes scale with required pages rather than the logical box;
2. stage work scales with active subcells/pages;
3. topology churn is small enough for incremental reuse to pay;
4. directory/apron overhead does not erase the saved bandwidth; and
5. sparse results match the dense method within explicit numerical gates.

## 13. Recommended implementation program

### Phase 0: shadow residency census, no numerical changes

Add a GPU or deterministic CPU diagnostic that computes, per advance and per
stage:

- core page count;
- support/halo page count;
- activated, carried, and retirement-candidate counts;
- page fill ratio;
- per-field estimated bytes;
- workgroups avoided versus dense;
- maximum characteristic reach;
- missing-support attempts; and
- directory and sort cost estimates.

Run widths 8, 16, and 32. Do not use one union set only; publish pressure,
transport/FIM, conservative scalar, solid, and presentation sets separately.

Decision gate: proceed only if at least the target large sparse scenes retain a
clear end-to-end margin after support closure. A reasonable initial gate is
less than 35-50% page residency for the dominant fields, low single-digit page
churn after startup, and a modeled memory reduction above 2x. These are
prototype gates, not permanent magic constants.

### Phase 1: generic page directory and presentation-only payload

Build a method-neutral page pool/directory with:

- stable physical IDs;
- candidate/accepted double buffering;
- allocation/free lists;
- deterministic failure receipts;
- indirect active-page and subcell worklists; and
- dense-to-page and page-to-dense test adapters.

Migrate Section 3.8 render-only density reconstruction first. This proves
lookup, page seams, renderer integration, and lifecycle without affecting
physics.

### Phase 2: conservative scalar transport, dense velocity and pressure

Move `rho`, `gamma`, conditioning deposits, diffusion, sharpening, and solid
excess to pages while continuing to sample dense velocity and run dense
pressure. Compare each stage against the existing audit textures. Do not retire
the dense scalar copy until cross-seam conservation and activation are proven.

### Phase 3: sparse velocity sampling and FIM worklists

Move velocity page storage, cross-page MAC ownership, transport shells, and FIM
active subcell lists. Keep pressure dense. This phase should demonstrate the
paper's work-efficient subcell idea most directly.

### Phase 4: sparse projection work with a global hierarchy

First dispatch pressure setup/projection over active fine pages while retaining
a dense coarse solve. Then make fine multigrid levels sparse with parent
closure. Densify at the first level where the active page set or directory
overhead is no longer favorable.

Never introduce per-page independent pressure solves as an intermediate
"approximation" without a separate method ID and error budget.

### Phase 5: retire dense mirrors and optimize representation

Only after full-step parity:

- compare direct directory reads, staged halos, and persistent aprons;
- compare buffer pages and texture atlases;
- tune storage page versus compute tile sizes;
- split per-field residency sets where the union is wasteful; and
- evaluate hash/two-level directories for logical domains whose dense page map
  is itself too large.

## 14. Required validation matrix

### 14.1 Exact seam and storage tests

1. Insert artificial page seams through a dense state and run each kernel.
2. Verify one authoritative MAC face and equal/opposite divergence use.
3. Verify trilinear samples at every corner/edge/face seam orientation.
4. Verify field-specific missing defaults, especially `gamma=1` and FIM
   distance infinity.
5. Verify new-page initialization before publication.
6. Verify carried pages retain exact state and retired IDs are not reused early.

### 14.2 Conservative transport tests

1. Translate a slab across one and many page seams.
2. Advect a diagonal droplet through page corners.
3. Exercise sharpening and partial-solid excess across seams.
4. Compare every donor/receiver fixed-point sum and total mass with dense.
5. Force activation at the final possible trace destination.
6. Force capacity overflow during a scatter and require fail-closed behavior.

### 14.3 Pressure tests

1. Dense-versus-sparse artificial-seam equivalence.
2. A long connected channel spanning many pages.
3. A one-cell neck between pools.
4. Split and rejoin components.
5. Closed fully-Neumann and open-top cases.
6. Liquid-air faces in both ownership orientations.
7. Separating moving solids crossing a page seam.
8. Per-cycle projected residual and final divergence versus dense.

### 14.4 Activation tests

1. Maximum accepted CFL displacement.
2. MacCormack forward and reverse footprints.
3. Fast inflow into previously empty logical space.
4. Moving rigid swept bounds.
5. FIM front reaching a new page.
6. Retirement immediately followed by reverse flow.
7. Topology capacity exhaustion and recovery.

### 14.5 Scene benchmark set

Use both positive and negative controls:

| Scene | Purpose |
| --- | --- |
| 64-cubed mini dam | Dense parity and expected weak speedup |
| 128-cubed high-resolution dam | Moderate sparsity and practical crossover |
| CM12 Figure 9 dam | Paper-contract scalar transport with good initial sparsity |
| Garden dam migration | Extreme sparse movement and page churn |
| Garden hose/inflow | Pre-activation into empty space |
| Large-power dam | Large empty domain |
| Ocean seiche | Deep-wet negative control; sparse pressure should not fake a win |
| Symmetric expansion | D4 symmetry and worst-case small-domain halo expansion |
| Moving-solid oracle | Cut cells, ownership, and two-way impulse across seams |

Report memory, page counts, directory lookup time, topology time, every existing
pipeline stage time, end-to-end wall time, residuals, divergence, volume drift,
maximum speed, symmetry, and renderer cost. A sparse kernel microbenchmark is
not sufficient evidence.

## 15. Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Sparse method replaces dense oracle | Topology bugs become self-validating | Separate method ID and retain dense field/stage comparisons |
| Support underestimation | Lost mass, undefined samples, unstable pressure | Predictive union of characteristics, stencils, inflows, and solid sweeps; hard missing-page counters |
| Page seam duplicates face state | Divergence and projection mismatch | Global logical face ownership by lower-coordinate cell |
| One universal residency set | Sparse fields allocate like dense pressure bulk | Per-field/stage census, then split pools where justified |
| Universal persistent aprons | Memory blow-up and repeated copy work | Direct lookup/staged halo baseline; apron only by measurement |
| Small pages | Directory and halo overhead dominate | 8/16/32 sweep with storage pages decoupled from 4-cubed work tiles |
| Large pages | Internal waste and coarse activation | Fill-ratio telemetry and mixed scene benchmark set |
| Independent page pressure solves | Reflections, divergence seams, wrong long modes | One global sparse multigrid hierarchy and residual |
| Import paper CG | Hundreds of iterations and lost LCP physics | Keep CM11a operator and projected solve |
| Particle-derived lifecycle copied literally | No way to activate absent Eulerian state | Analytic/field/swept producers and transactional preactivation |
| Dense page directory in huge worlds | Metadata still scales with bounding box | Two-level/hash directory after bounded MVP |
| Moving solids omitted from residency | Missing cut cells and impulses | Swept solid bounds and generation-coherent face geometry |
| Current pressure WIP used as an assumed baseline | Sparse comparison moves underfoot | Stabilize/record dense control and fixed-schedule arm first |

## 16. Go/no-go decision

**Go** for Phase 0, the shadow residency census. It is low risk, preserves the
reference, and will answer the real applicability question with local data.

**Conditional go** for a separate sparse-uniform prototype if runtime peak
support remains clearly sparse and modeled bytes/work beat dense by a wide
enough margin to pay for directory and lifecycle overhead. Garden migration,
garden hose, large empty-domain dams, and possibly the 128-cubed dam are the
best candidates.

**No-go** for:

- changing the existing `uniform` method into a sparse method;
- adding FLIP particles solely to imitate this paper;
- replacing CM11a LCP multigrid with the paper's unpreconditioned CG;
- treating one-cell aprons or 32-cubed bricks as fixed truths;
- solving pressure independently per page; or
- claiming the paper's 6-28x CPU comparisons as an expected browser speedup.

The defensible target is a **numerically matched sparse-uniform method** whose
absence of pages is observationally equivalent to unused dense storage. If
that condition cannot be demonstrated at artificial seams, during activation,
and in the global pressure residual, the paper remains useful inspiration but
not an applicable implementation path.

## 17. Relationship to the earlier repository critique

`wu-2018-gvdb-flip-critique.md` analyzed sparse pages and regional/domain-
decomposed pressure ideas, then recorded a scope correction toward adaptive
fine/coarse regions. This document answers a different and narrower question:
applicability to the current **dense uniform reference**. Its central decision
is therefore deliberately different:

- sparse uniform pages are a valid new execution method;
- regional pressure decomposition is not required by GVDB-style storage;
- one global uniform pressure system should remain the starting point; and
- the dense uniform implementation remains the oracle rather than the migration
  target to overwrite.

## 18. Verification notes

- The downloaded PDF has 11 A4 pages, title *Fast Fluid Simulations with Sparse
  Volumes on the GPU*, and SHA-256
  `3af396b1bd3d7239f2f1a32d88dfea4d130c9f3baeb4cae02631289354254b26`.
- A fresh `pdftotext -layout` extraction was byte-identical to
  `wu-2018-gvdb-flip.txt` (741 lines, 9,483 words).
- All 11 pages were rendered and visually checked. Figures 2-5 and Tables 1-4
  require the PDF for layout; the text extraction is complete but cannot
  preserve their spatial diagrams.
- The current worktree was dirty before this analysis. No existing source or
  handoff change was overwritten.
- A focused static test run on the dirty snapshot reported 43 passes, 5
  failures, and 1 skipped GPU test; repository-wide `tsc --noEmit` also has
  unrelated pre-existing failures. This document does not certify the
  in-progress pressure branch as stable. Its sparse recommendation is based on
  the declared/current architecture and explicitly requires a stabilized dense
  control before implementation.
