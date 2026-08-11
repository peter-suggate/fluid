# Sparse uniform Eulerian regions: GVDB FLIP applicability and risks

> Scope correction: the intended design was subsequently clarified as
> **fine grids in energetic regions and coarser grids in calm regions**, not
> uniform-resolution regions with activity-dependent iteration counts. GVDB's
> sparse bricks remain relevant for storage and work scheduling, but this
> document's uniform-lattice recommendation is not the final physical
> discretization. See `liu-2016-schur-complement-fluids-notes.md` for the
> corrected multiresolution assessment.

## Decision summary

Wu et al.'s storage and work-scheduling ideas are a strong fit for a sparse
set of **fixed-size, globally aligned uniform bricks**. They are not, by
themselves, a safe numerical design for independently simulated Eulerian
regions that happen to exchange values at their boundaries.

The conservative interpretation is:

- one global Cartesian lattice and one physical cell size;
- sparse brick/page residency only changes where storage and work exist;
- a face shared by two resident bricks has one authoritative flux/normal
  velocity;
- pressure coupling across every face-connected wet interface is represented,
  even when the interiors are solved in separate regional jobs;
- halos/aprons are caches, never additional physical degrees of freedom; and
- topology, fields, face adjacency, and worklists publish as one coherent
  generation.

Under those constraints, "interconnected regions" can serve as useful solver
subdomains, but should not duplicate ownership of seam state. Separate regional
interior solves plus a coupled interface treatment are a domain-decomposition
method. The GVDB paper does not validate that method, although the Schur-
complement fluid work it cites does.

## What transfers well

1. **Dense computation inside sparse pages.** The paper separates large
   storage bricks from 4-cubed, 64-voxel GPU subcells. The repository already
   has the more portable foundation: stable 8-cubed physical owner pages and a
   direct logical-brick directory in `webgpu-octree-owner-pages.ts`.

2. **Active worklists rather than dense-domain dispatch.** Compact core and
   halo brick lists are directly applicable to Eulerian transport, projection,
   redistance, rendering publication, and reductions. The current
   `webgpu-fluid-brick-residency.ts` already distinguishes core, halo,
   activated, and retired bricks and supplies indirect dispatches.

3. **Matrix-free pressure application.** Reconstructing a row from cell/face
   state avoids materializing a seven-point matrix. The current Losasso path
   already goes further by storing compact, reciprocal faces and applying the
   operator through row-to-face incidence.

4. **Predictable neighbour access.** A direct page directory or a cached halo
   can remove hierarchy traversal from stencil inner loops. Which one wins on
   WebGPU should be measured; the CUDA texture-atlas result is not sufficient
   evidence for either choice.

5. **Incremental topology lifecycle.** Activation, stable carried page IDs,
   retirement hysteresis, capacity checks, and candidate/accepted publication
   are useful regardless of whether particles or an Eulerian field discover
   active space.

## Principal risks

| Risk | Why it matters | Required control |
| --- | --- | --- |
| Treating a storage seam as a physical boundary | Creates reflections, mass leakage, pressure discontinuity, or duplicated flux | One shared face record and one flux evaluation per face |
| One-shot disconnected regional pressure solves | Pressure projection is elliptic; uncorrected interface flux leaves divergence and visible seams | Coupled interface solve, converged overlapping iterations, or a measured inexact-projection error budget |
| Stale or contradictory halos | CG, advection, redistance, and cut-cell operators can each read a different epoch | Producer-specific halo schedule or direct lookup; generation stamped publication |
| Late activation | Pure Eulerian state has no particles outside the allocation from which to rebuild support | Predictive activation from characteristic reach, inflows, moving solids, and interface/velocity halos |
| Premature retirement | Deletes velocity or scalar state needed by the next backtrace or pressure stencil | Field-specific support closure plus hysteresis and conservative handoff |
| Brick padding and halo memory | Sparse geometry can still allocate mostly empty voxels; halos multiply every field | Measure live-cell/page occupancy and page-size sweeps before migration |
| Topology/field split epochs | A solver can combine new adjacency with old values or reuse a page before consumers finish | Double-buffer candidate state and fail-closed atomic commit |
| CUDA-specific assumptions | Warp size, texture atlas behaviour, atomics, and readback costs differ in portable WebGPU | Benchmark WebGPU buffers/directories and indirect convergence on target adapters |
| Basic seven-point physics | The paper's solid/fluid/empty stencil does not cover this repository's complete cut-cell, ghost-fluid, moving-solid, and free-surface contract | Preserve the existing face metrics and boundary classification in the sparse operator |
| Surface faceting | Sparse uniform bricks save empty-domain work, not resolution within a resident brick | Retain interface-resolution rules and the separate presentation/detail path |

## Boundary interconnection is the critical design point

### Storage seams must be invisible

For two face-adjacent bricks, the seam should be observationally equivalent to
an interior plane in the dense uniform reference. In particular:

- the face-normal velocity/flux is stored once or has a single declared owner;
- divergence in both adjacent cells uses that same value with opposite signs;
- projection updates the face once from the pressures on both sides;
- VOF/density transport applies one paired donor/receiver flux;
- solid aperture and moving-boundary velocity are identical from both sides;
- scalar interpolation may use cached duplicates, but authoritative writes do
  not race through both copies.

Face connectivity also needs a precise meaning. A seven-point pressure stencil
connects cells only across faces. Edge- or corner-touching wet bricks may be
neighbours for interpolation or morphology, but they are not automatically one
pressure component.

### One-cell aprons are not universally sufficient

The paper's apron is suitable for immediate-neighbour stencils and is refreshed
inside CG for the search direction. This solver also has characteristic
backtraces, MacCormack prediction/correction, VOF transfer, redistance,
velocity extension, moving solids, and topology sizing. Their required support
is controlled by maximum trace distance and stencil radius, not just by the
pressure Laplacian. A safe residency closure is therefore field- and stage-
specific, or conservatively at least

`ceil(maximum characteristic distance / dx) + stencil radius`.

Dispatching a fixed one-brick halo is only correct when the brick width exceeds
that bound for every accepted substep.

### Halo storage can erase much of the memory win

If a full one-cell apron is stored on every side of every brick, the stored
samples per logical sample are approximately:

| Interior brick | With one-cell apron | Storage ratio |
| --- | --- | --- |
| 8-cubed | 10-cubed | 1.953x |
| 16-cubed | 18-cubed | 1.424x |
| 32-cubed | 34-cubed | 1.199x |

That cost repeats for every haloed channel. The paper reports thirteen volume
channels and identifies subcell particle lists as its largest transient cost.
An Eulerian-only method avoids the particle-list cost, but not the multi-field
halo cost. A direct logical-page lookup plus shared face buffers may be better
for 8-cubed pages; this is an empirical WebGPU decision.

## Adjacent regional solves and wave propagation

Solving adjacent regions together can propagate a large wave across a vast
scene. Conservative transport, gravity, and the moving free surface carry the
wave from one region to the next over successive physical time steps. That is
not the same as fully coupling the pressure projection.

In an incompressible method the gravity wave has finite phase/group velocity,
but the pressure Poisson equation used to enforce zero divergence is elliptic.
A pressure correction can therefore have nonlocal support within one connected
liquid component during a single substep. If each local cluster accepts its
first solution using lagged pressure or velocity at the outer boundary, wave
motion will still travel, but the method has deliberately become an inexact
projection. Likely failure modes are seam reflection, incorrect long-wave
phase speed, hydrostatic drift, volume error, and a divergence layer that moves
with cluster boundaries.

There are three credible coupling levels:

1. **Exact algebraic coupling, region-local work.** Solve interiors in parallel,
   then solve a Schur-complement system containing only interface unknowns and
   finish the interiors with those boundary values. The computation is regional
   even though the result is the global pressure solution. Liu et al. (2016)
   demonstrate this pattern on large sparse uniform grids.
2. **Iterative neighbouring clusters.** Use overlapping Schwarz/block solves,
   exchange seam values, and repeat until global divergence and interface-flux
   residuals pass. Information moves roughly through the region graph as the
   iterations proceed; a coarse region-graph correction prevents iteration
   count from growing with scene diameter.
3. **Bounded local projection.** Run a fixed small number of neighbour exchanges
   and accept the residual. This is potentially fastest and still allows waves
   to propagate over physical time, but it is a new approximation that needs
   explicit visual and quantitative error budgets. It should not be described
   as equivalent to the dense incompressible reference.

If "solve adjacent regions together" means repeatedly merging every touching
wet region, one large connected ocean eventually becomes one vast solve. To
retain bounded work, keep fixed-size subdomains and couple their comparatively
small interface graph instead of making connected-component identity define a
monolithic allocation.

## The AVBD parallel: adaptive work where constraints are active

The useful parallel with Augmented Vertex Block Descent is not the physical
equation; it is the scheduling and constraint architecture:

| AVBD | Regional Eulerian projection analogue |
| --- | --- |
| Body/vertex block | Uniform fluid region or page cluster |
| Contact/joint constraint | Shared-face flux and divergence constraint |
| Graph coloring | Parallel solve of nonadjacent regions |
| Lagrange multiplier | Persistent interface pressure/flux correction |
| Progressive penalty/stiffness | Increase interface coupling only where residual persists |
| Warm start | Carry decayed interface multipliers/pressure from the prior substep |
| Constraint violation | Seam flux mismatch and post-projection divergence |

This suggests an augmented regional solve:

1. solve every active region once, processing a color of the region-adjacency
   graph in parallel;
2. evaluate cell-divergence and interface-flux residuals;
3. mark violating interfaces and both incident regions, then dilate that hot
   set by one graph ring;
4. update persistent interface multipliers and, if useful, a bounded penalty;
5. iterate only the hot induced subgraphs;
6. stop when the hard residual passes, or escalate by widening the hot cluster,
   running a coarse interface correction, or substepping.

The warm-start idea is particularly attractive for slowly evolving water: most
seams should begin near their prior solution, while an impact or newly connected
region creates a localized burst of extra work. Decay is important so obsolete
large corrections do not remain after a wave or topology event has passed.

The analogy has a hard limit. AVBD inherits unconditional stability from its
variational VBD step; its paper explicitly says that low iteration counts can
leave large constraint error and excess momentum without breaking numerical
stability, and adds gradual correction to avoid a later momentum spike. A
regional fluid projection does not automatically inherit that guarantee.
Unresolved divergence can amplify transport and free-surface error. Therefore
"iterations only when stability demands" needs two gates:

- a **hard acceptance gate** for finite state, bounded VOF/density, no missing
  stencil support, normalized divergence, paired seam flux, and correction
  impulse; and
- a **soft accuracy gate** for pressure residual, volume drift, wave phase,
  seam reflection, and hydrostatic balance.

Extra regional iterations can be demand-driven by the hard gate. Soft error is
not always locally observable: a long-wavelength mode can have small error at
every individual seam while being wrong across the full region chain. A cheap
coarse interface-graph residual or periodic global audit is the safeguard. If
the design forbids even that small global operation, iteration count and error
must be tested explicitly against region-graph diameter.

## Pressure-solver concerns

The paper's unpreconditioned CG is useful evidence for matrix-free sparse
execution, not a solver target. Its Table 2 iteration counts rise from 229 to
511 for the dam-break sequence as the dense-equivalent resolution grows from
256-cubed to 512-cubed. Its GPU advantage is primarily per-iteration throughput.
The repository's persistent MGPCG/V-cycle and GPU-side convergence tail should
be retained.

A one-shot pressure solve per uniform region is especially risky. Long
connected bodies, thin necks, or separated pools joined during a frame transmit
pressure through their interfaces. Independent solves with lagged Dirichlet or
Neumann guesses can violate conservation at the seam. Regional decomposition
is much stronger when used as block Jacobi/Schwarz or Schur-complement
preconditioning beneath an outer Krylov iteration, with a coarse region-graph
space to communicate low-frequency pressure modes. A deliberately truncated
outer iteration is also viable, but its remaining divergence is part of the
method and must be exposed in diagnostics.

Connected-component handling must also be explicit. A fully Neumann component
has a constant-pressure null space and needs a gauge or null-space projection;
a component touching a free surface receives an atmospheric Dirichlet
condition. Dynamic splits and joins can change this classification at the same
time as page residency changes.

## Topology and lifecycle concerns

The paper discovers required space from FLIP particles. A pure Eulerian method
cannot rely on that mechanism: state outside resident pages does not exist and
therefore cannot request its own allocation. The active-set producer must look
ahead from current interface cells, velocity characteristics, inflows, moving
rigid bodies, force bounds, and all operator halos. A fast jet crossing more
than the retained halo in one substep is a correctness failure, not a graceful
loss of detail.

Variable-sized "regions" are also a poor ownership primitive. Merging and
splitting boxes causes large copies, unstable indices, overlapping ownership,
and difficult partial failure. Fixed-size pages should own state. Connected
regions can be recomputed as cheap labels over the page/face graph when they
are useful for gauges, diagnostics, or dispatch batching.

The repository's existing candidate/accepted discipline is the right model:
stable physical page IDs while resident, inactive candidate banks, exact
generation clocks, and fail-closed publication. A sparse-uniform path should
reuse that substrate rather than recreate the paper's mutable pointer tree.

## WebGPU-specific caveats

- GVDB's 3D texture atlas and hardware trilinear filtering were tuned for
  NVIDIA CUDA-era hardware. WebGPU texture limits, storage-texture formats,
  binding budgets, and adapter differences make a buffer page pool plus direct
  directory the safer starting point.
- The paper's 4-cubed subcell is justified partly as two 32-thread CUDA warps.
  A 64-invocation workgroup is portable, but subgroup size is not universally
  32 and should not be embedded in correctness logic.
- Reading the residual to the CPU every ten CG iterations introduces a queue
  synchronization point. Preserve the current GPU-zeroed indirect dispatch
  convergence mechanism.
- The reported Quadro GP100 timings compare against CPU implementations from
  2018. They establish feasibility, not a current-browser speedup estimate.

## Recommended architecture

1. Keep a single global lattice and call the allocation units **pages** or
   **bricks**, not simulation regions.
2. Reuse the existing 8-cubed owner-page arena, stable physical page IDs,
   direct logical directory, active/halo worklists, and candidate-ready commit.
3. Make face records global and reciprocal. Page-local cell payloads may be
   dense, but normal velocity and flux ownership must survive page seams.
4. Keep globally unique compact pressure rows/faces, but allow regional
   interior solves. Couple them with an interface Schur complement or an
   augmented/overlapping outer iteration. Persist and decay interface duals,
   schedule extra work from residual-marked hot subgraphs, and retain the
   current MGPCG as the comparison and likely subdomain solver.
5. Start without stored aprons. Measure direct-directory neighbour fetches
   against one-cell and stage-specific packed halos on representative adapters.
6. Migrate one conservative scalar field behind a paged accessor before moving
   pressure or velocity authority. This isolates residency and seam semantics
   from adaptive-operator changes.
7. Retain dense uniform execution as the exact matched-lattice oracle.

## Minimum acceptance experiments

1. **Interior-seam equivalence:** insert an artificial page seam through a
   dense state; one full step must match the dense uniform reference within the
   established f32 tolerance.
2. **Conservative seam transport:** advect a planar slab and a diagonal droplet
   repeatedly across page boundaries; report total volume and paired face-flux
   mismatch.
3. **Regional projection:** project a velocity field spanning two, then many,
   regions. Compare one-shot local, fixed-hop Schwarz, converged Schwarz, and
   Schur-coupled results against the dense reference. Repeat with a long channel
   and a one-cell neck; plot error and iteration count against region-graph
   diameter.
4. **Split/join:** let liquid components separate and reconnect while pages
   activate and retire; verify pressure gauge handling, finite state, and no
   impulse or volume jump.
5. **Fast-front activation:** use the maximum accepted CFL/substep displacement
   and an inflow near nonresident space; no characteristic may sample a missing
   page.
6. **Moving-solid seam:** move a rigid boundary across a page boundary and
   compare aperture, reaction impulse, and no-penetration error with dense.
7. **Capacity rejection:** force page/adjacency exhaustion; the entire candidate
   must fail closed while the previous accepted generation remains coherent.
8. **Page-size/lookup sweep:** compare 8, 16, and 32 interior widths, direct
   directory lookup, and stored halos using occupied-cell ratio, bytes, pass
   count, neighbour bandwidth, projection time, and end-to-end frame time.
9. **Adaptive-iteration pile:** create a deep/wide connected region graph with
   a localized impulse, analogous to the AVBD block pile. Compare fixed one,
   two, and four sweeps against residual-driven hot-subgraph scheduling. Record
   hot-region count, total regional solves, maximum divergence, seam residual,
   wave phase, volume drift, and correction impulse over time.
10. **Long-mode adversary:** initialize a low-frequency pressure/velocity mode
    over a long chain with small per-seam residual. Verify that the coarse audit
    triggers work that a purely local threshold would miss.

## Go/no-go criterion

Proceed if a paged scalar prototype shows a material memory/work reduction on
the sparse scenes while seam equivalence and conservative transport pass. A
regional pressure prototype should additionally show either converged interface
coupling at lower cost than current MGPCG, or a fixed-hop approximation whose
divergence, volume, phase, and reflection errors stay inside an explicitly
accepted budget as scene diameter grows. Stop or narrow the scope if savings
require duplicated face authority or a halo allocation whose multi-channel
overhead approaches the dense baseline. The most defensible first target is
sparse storage and work compaction for the uniform reference method, followed
by regional projection as a separately gated experiment.

## References

- Wu, Truong, Yuksel, and Hoetzlein, *Fast Fluid Simulations with Sparse
  Volumes on the GPU*, 2018: <https://doi.org/10.1111/cgf.13350>
- Liu, Mitchell, Aanjaneya, and Sifakis, *A Scalable Schur-complement Fluids
  Solver for Heterogeneous Compute Platforms*, 2016:
  <https://doi.org/10.1145/2980179.2982430>
- Giles, Diaz, and Yuksel, *Augmented Vertex Block Descent*, 2025:
  <https://doi.org/10.1145/3731195>
