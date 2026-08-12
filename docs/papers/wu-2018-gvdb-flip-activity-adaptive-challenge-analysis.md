# GVDB FLIP against the activity-adaptive multiresolution goal

Source paper: [Wu, Truong, Yuksel, and Hoetzlein, *Fast Fluid Simulations with Sparse Volumes on the GPU*](./wu-2018-gvdb-flip.pdf) (2018). The layout-preserving extraction is [wu-2018-gvdb-flip.txt](./wu-2018-gvdb-flip.txt).

Plans evaluated:

- [activity-adaptive-multiresolution-fluid-plan.md](../activity-adaptive-multiresolution-fluid-plan.md), retained as a research and test reference but explicitly superseded.
- [uniform-region-fluid-handoff.md](../uniform-region-fluid-handoff.md), the current implementation direction.

## 1. Bottom line

GVDB FLIP provides strong answers to the **sparse GPU execution** challenges around the goal, but almost no direct answers to the **activity-adaptive physical-resolution** challenges that make the goal difficult.

The paper demonstrates that a very large, moving, sparsely occupied Eulerian domain can be represented as pooled dense bricks, updated incrementally on the GPU, traversed with predictable neighbor access, processed with GPU worklists, and solved by one global matrix-free pressure iteration. Those are valuable implementation results.

It does **not** demonstrate a fluid discretization in which adjacent active regions have different cell sizes. Its GVDB hierarchy is an addressing and allocation hierarchy. The physical fluid samples in the active leaf bricks remain uniformly spaced. Consequently, it does not supply the core mechanisms required by either multiresolution plan: 2:1 seams, coarse/fine flux authority, conservative refine/coarsen transfer, composite free-surface pressure, transition-wave validation, or an activity policy that selects physical resolution.

The most precise conclusion is:

> GVDB FLIP is evidence that sparse, dynamic, GPU-resident block infrastructure is practical. It is not evidence that activity-selected multiresolution fluid physics is solved.

For the current uniform-region architecture, the paper is most applicable to shared storage, work scheduling, same-resolution halo exchange, global sparse operator execution, and the conditional P5 idea of sparse residency inside large regions. It does not de-risk P1, the load-bearing coarse/fine seam.

## 2. The word "adaptive" hides three different problems

Distinguishing these problems prevents the paper from being credited with results it does not contain.

| Kind of adaptation | Question | GVDB FLIP | Our goal |
| --- | --- | --- | --- |
| Allocation adaptation | Where does voxel storage and compute exist? | Directly addressed | Required for scale, but not sufficient |
| Resolution adaptation | Where does physical cell size change? | Not addressed; active leaf voxels have one physical spacing | Central requirement |
| Activity and lifecycle adaptation | What evidence requests accuracy, how is it predicted, budgeted, hysteretic, woken, or retired? | Particle coverage drives allocation, not an accuracy policy | Central requirement |

The GVDB tree contains multiple index levels, but this must not be read as a hierarchy of coarse and fine fluid control volumes. Internal nodes locate children. Leaf nodes point to dense voxel bricks. The pressure, velocity, markers, and density are evaluated on the uniform leaf voxel lattice.

This distinction changes the applicability verdict from "a solution to adaptive fluids" to "a useful substrate for sparsely allocated fluid blocks."

## 3. Verdict vocabulary

The challenge ledger uses four verdicts:

- **Answers**: the paper implements and evaluates substantially the same problem we need to solve.
- **Partially helps**: the paper supplies a reusable mechanism or evidence, but important requirements remain.
- **Sidesteps**: the paper avoids the challenge because FLIP particles or uniform voxels make it unnecessary in that design.
- **Does not address**: the necessary mechanism and validation are absent.

"Sidesteps" is not a criticism. A sidestep can be a useful architectural option. It is not, however, evidence that our Eulerian handoff problem has been solved.

## 4. Executive challenge ledger

| Challenge in our goal | Verdict | What the paper actually contributes | What remains unanswered |
| --- | --- | --- | --- |
| Sparse, virtually unbounded occupancy | **Answers** | Hierarchical sparse addressing, pooled dense bricks, GPU traversal | Our allocation and capacity policy |
| Dynamic GPU topology | **Answers** for occupancy topology | Full and incremental GPU construction; reuse and removal of nodes/bricks | Transactional multiresolution regrid with physical state transfer |
| Efficient work granularity | **Answers** for its workload | Storage bricks plus smaller 4^3 subcells for particle work | Best compute granularity for our uniform solver kernels |
| Neighbor access across sparse bricks | **Answers** at equal cell size | One-voxel apron duplicates neighboring values | Different-resolution halo semantics and conservative seam ownership |
| Global pressure over sparse storage | **Partially helps** | One matrix-free CG solve spans all active sparse voxels | Composite 2:1 operator, free-surface complementarity, robust preconditioning, global coarse correction |
| Activity selects required physical spacing | **Does not address** | Particle influence determines which bricks exist | Accuracy indicators, requiredDx, thresholds, budgets, hysteresis |
| Predictive support for moving features | **Partially helps** only as analogy | Particle influence boxes allocate touched neighborhoods | Velocity/acceleration reach, stencil bands, late-activation prevention |
| Conservative refine/coarsen | **Sidesteps** | Persistent particles are re-rasterized after the grid is cleared | Exact Eulerian mass, flux, momentum, and energy transfer between resolutions |
| Coarse/fine 2:1 coupling | **Does not address** | All active fluid voxels have equal spacing | T-junction incidence, atomic subfaces, restriction/prolongation, SPD proof |
| Free-surface composite solve | **Does not address** | Basic FLIP pressure projection with fluid markers | Our LCP/cut-cell/ghost-fluid semantics across seams |
| Long-wave preservation through LOD | **Does not address** | No resolution transitions or reflection tests | Wavelength floor, phase error, reflection and transmission criteria |
| Sleep/wake and many-body lifecycle | **Does not address** | Allocation follows particles frame to frame | Analytic asleep bodies, wake transactions, merge/split, temporal cadence |
| Coherent generations and rollback | **Does not address** | Incremental node mutation and buffer resize/clear | Candidate/accepted generations, validation gates, fail-closed commit |
| GPU memory scaling | **Answers** for its data model | Pooled channels and measured topology memory | Memory cost of our multilevel pressure state and regional solver metadata |
| Real-time target | **Does not establish** | Large throughput gain over CPU, but reported frames are generally hundreds to thousands of milliseconds | Product-specific frame budget and bounded worst case |
| Sparse volume rendering | **Answers** for its representation | Direct volume ray tracing without polygon conversion | Compatibility with our surface representation and desired rendering |

## 5. What the paper genuinely answers

### 5.1 Sparse storage can be separated from dense local computation

GVDB stores active voxel data in dense bricks pooled into a texture atlas. A tree maps world-space positions to those bricks. This is a strong architectural result for us: sparse global occupancy does not require every numerical kernel to operate on an irregular pointer-rich data structure.

That supports a design in which:

1. world-space occupancy is sparse;
2. each allocated block has a regular local layout;
3. kernels are launched from compact active-block or active-subcell lists; and
4. storage pools are reused as activity moves.

This is compatible with the current uniform-region plan, but the reusable unit should probably be a pooled block at a particular resolution level, not necessarily one complete solver object and multigrid hierarchy per small region. If fixed per-region overhead becomes material, GVDB suggests flattening same-level regions into shared pools and shared dispatch lists while retaining region identity as metadata.

### 5.2 GPU topology updates need not round-trip through the CPU

Sections 5.1 and 5.2 construct the sparse hierarchy on the GPU. The incremental path identifies missing and unused nodes, reuses allocations, and relinks the hierarchy. The reported topology update is a small portion of frame time in the tested scenes and uses much less temporary memory than a full rebuild.

This directly answers whether moving sparse occupancy can be maintained without rebuilding a CPU-side spatial structure every frame.

It does not answer whether a proposed topology is physically safe to accept. The paper's mutation problem is "which uniform bricks cover the particles?" Our regrid problem is additionally "can this new combination of physical resolutions preserve all conserved state and form a valid symmetric operator?" The latter needs candidate construction, validation, receipts, and atomic commit.

### 5.3 Storage granularity and compute granularity should differ

The paper uses relatively large bricks for storage and 4^3 logical subcells for particle binning and kernel work. This is a particularly useful lesson. The ideal allocation unit is governed by metadata and fragmentation costs; the ideal dispatch unit is governed by occupancy, cache behavior, and GPU workgroup size. They need not be identical.

For our implementation, this suggests separately tuning:

- allocation block extent;
- pressure/transport workgroup tile;
- activity-reduction tile;
- halo-copy tile; and
- seam work item, which should remain the atomic fine subface where resolutions differ.

The paper does not determine the numerical values for WebGPU or for our solver. Its CUDA-era 32-thread warp assumptions and texture behavior are not portable constants.

### 5.4 Equal-resolution sparse neighbors can be made stencil-friendly

GVDB's apron voxels duplicate values from neighboring bricks so stencil kernels can access immediate neighbors locally. This is strong support for a same-resolution halo ABI and for explicit halo-update stages in the frame graph.

An apron is not a coarse/fine interface. At a 2:1 boundary, a ghost value is not merely a copied neighbor sample: it depends on restriction or prolongation, geometry, quantity type, and the authority rule for flux. The paper supplies the scheduling pattern but not the seam mathematics.

### 5.5 Sparse storage does not require independent local pressure solves

The matrix-free CG solve runs over the complete active sparse voxel set. Global reductions compute inner products, and sparse stencil application crosses brick boundaries through updated aprons. This is important evidence for a principle already present in our plans: storage blocks do not define pressure domains.

The paper therefore argues against treating each region or pairwise seam as an independently projected fluid. A connected fluid component needs one coupled pressure problem, whether implemented monolithically or by a mathematically equivalent domain-decomposition method.

The contribution stops short of our pressure requirements. The reported solver is unpreconditioned matrix-free CG for a simpler FLIP projection, with residual checks read back periodically. It does not include:

- a 2:1 composite discretization;
- cut-cell/free-surface complementarity semantics matching our uniform method;
- FAC or block multigrid across physical resolution levels;
- a global coarse correction designed for long waves; or
- convergence and symmetry tests at coarse/fine transitions.

The paper itself identifies improved preconditioning or sparse multigrid as future work. Its iteration counts and pressure timings reinforce that sparsity alone does not solve the global-pressure bottleneck.

## 6. The crucial FLIP sidestep: topology changes without Eulerian regrid transfer

Algorithm 1 rebuilds or incrementally updates topology, resizes and clears volume data, bins particles, and transfers particle state back to voxels before projection. Persistent Lagrangian particles carry the fluid state while the Eulerian grid is disposable working storage.

This elegantly avoids much of the state-transfer problem in the superseded plan. If a brick disappears and another appears, the authoritative particle set persists and the new grid is rasterized from it. There is no need to conservatively restrict an old fine Eulerian density field into a new coarse Eulerian field because the paper does not treat that field as the sole persistent authority.

That is a **sidestep**, not a solution to our proposed Eulerian transaction. Our current uniform method has graph-owned surface mass, face velocity/flux, level-set geometry, and a specific pressure formulation. Changing its persistent authority to particles would be a major method change with its own questions:

- particle quadrature noise and bias;
- reseeding and deletion conservation;
- surface reconstruction;
- thin-feature preservation;
- particle memory and sort cost;
- consistency with exact graph-owned mass; and
- transfer of the current wall, inflow, and LCP semantics.

The paper reports particle sorting/subcell lists as a major performance and memory cost. Therefore, "use particles to avoid regrid transfer" is a possible alternative research branch, not a free import into the uniform method.

## 7. Mapping to the superseded activity-adaptive plan

The old plan remains useful because it states the difficult requirements explicitly.

### 7.1 Outcome and research interpretation

The outcome requires fine physical resolution in violent regions and coarser physical resolution in calm regions. GVDB provides sparse occupancy at a fixed physical voxel resolution, so it cannot establish the outcome.

The old plan's existing interpretation is correct: GVDB supports sparse GPU-owned hierarchical storage, active-region worklists, and topology/data locality, but not the multiresolution coarse/fine fluid operator. That qualification should remain attached to every citation of this paper.

### 7.2 Core invariants

| Invariant family | Paper coverage |
| --- | --- |
| Dyadic physical spans and strict 2:1 balance | None |
| Candidate/accepted topology generations | No transactional equivalent |
| Fail-closed topology acceptance | None |
| One authoritative flux per atomic subface | None |
| Symmetric positive operator across T-junctions | None |
| Exact surface-mass conservation during regrid | Sidestepped by particles |
| No positive energy injection during transfer | Not evaluated |
| Activity changes spacing, not solver tolerance | No activity-selected spacing |
| Predictive expansion and hysteretic coarsening | No equivalent policy |

The paper's internal hierarchy consistency is an implementation invariant, but it is not the same as physical-generation coherence across topology, geometry, velocities, pressure coefficients, and surface state.

### 7.3 Activity model

The paper allocates coverage from particles and their influence boxes. This can inform `supportRadiusCells`: an active sample must allocate all storage touched by its transfer stencil. It does not inform `requiredSpan`, because it never asks whether the local physics warrants a finer or coarser cell size.

No equivalents are provided for curvature, strain/vorticity, velocity defect, surface defect, compression, forcing, solver defect, camera importance, wavelength constraints, global budgets, or quiet-history hysteresis.

The influence box is reactive stencil coverage, not the plan's predictive reach. It does not extrapolate velocity and acceleration several steps forward so a fast feature cannot outrun refinement. A GVDB allocation miss might lose a transfer destination; our policy failure can place a physical phenomenon onto an inaccurate resolution before the next accepted regrid.

### 7.4 Topology selection

The GPU full/incremental builders are useful implementation references after the desired resolution map is known. They do not produce that map. In particular, they supply no answer for:

- converting continuous evidence to a dyadic target spacing;
- 2:1 closure;
- authored-box oracle comparisons;
- budget arbitration;
- one-level-per-epoch transitions; or
- rejection of a numerically invalid candidate.

The incremental builder also should not be copied as an in-place physical regrid. Our safer abstraction remains build candidate, validate incidence/state/operator, then commit or reject.

### 7.5 Coarse/fine state transfer

The paper contains no restriction or prolongation between physical grid levels. It therefore does not answer the old plan's requirements for exact mass overlap, reconstruction of phi, flux restriction, divergence-aware velocity prolongation, momentum/energy receipts, or pressure warm starts.

Particle-to-grid and grid-to-particle transfer are different operations. They cannot be cited as evidence that an Eulerian fine-to-coarse transfer conserves the exact quantities owned by our method.

### 7.6 T-junction operator requirements

There are no physical T-junctions in the paper's fluid lattice. Aprons cross allocation-brick boundaries at equal spacing. Consequently, there is no evidence for reciprocal coarse/fine incidence, exact subface areas, symmetric coefficients, constant/linear reproduction, hydrostatic balance at a seam, or composite divergence cancellation.

This is the largest gap between the paper and the goal.

### 7.7 Long-wave propagation

The paper's global CG solve is directionally consistent with preserving globally coupled pressure modes, but there is no change in physical resolution and no long-wave transition experiment. It provides no wavelength floor, reflection coefficient, transmission phase, or amplitude-decay result.

The plan's global coarse correction remains necessary because local or pairwise solves can lose long modes. The GVDB result supports global coupling in principle but does not provide the correction scheme.

### 7.8 Risk register

The paper materially reduces only two old risks:

- **Sparse capacity cliff:** pooled sparse bricks and incremental reuse are credible mechanisms, although capacity exhaustion policy remains ours.
- **Activity work consumes the saving:** topology maintenance can be cheap relative to a large particle solve in the tested conditions, but this does not measure our richer activity estimator or WebGPU implementation.

It does not reduce the physics risks: resolution feedback, transition reflection, parasitic hydrostatic currents, topology chatter, fast features outrunning refinement, coarsening energy injection, global pressure cost, long-wave under-resolution, or velocity/front fidelity regression. It actually confirms that global pressure may remain dominant.

## 8. Phase-by-phase impact on the superseded plan

| Old phase | Applicability | Assessment |
| --- | --- | --- |
| Phase 0 - baselines and receipts | Low | Paper offers performance categories, not our conservation baselines |
| Phase 1 - observational activity | Very low | No physical-resolution activity estimator |
| Phase 2 - activity topology, fixed state | Medium infrastructure value | GPU allocation/worklists help after topology selection; no 2:1 closure |
| Phase 3 - conservative handoff | None for Eulerian design | Particle persistence sidesteps rather than solves it |
| Phase 4 - coarse/fine physics | None | No multiresolution operator or T-junctions |
| Phase 5 - moving activity and waves | Low | Incremental occupancy motion helps; predictive refinement and wave fidelity absent |
| Phase 6 - controls and failure policy | Low | Memory/performance observations help telemetry; no fail-closed policy |
| Phase 7 - regional decomposition | Low to medium | Global sparse CG and reductions are useful context; no Schur/FAC decomposition |

The paper is therefore not a reason to revive the superseded octree implementation. Its strongest lessons transfer naturally to block-structured infrastructure in the successor.

## 9. Mapping to the current uniform-region handoff

### 9.1 P0 - uniform-lane prerequisites

GVDB does not help establish exact conservation, bounded dissipation, inflow/wall correctness, or low per-region overhead in our method. It does suggest a way to address overhead if measurement finds that many solver instances are too expensive: use shared per-level pools and dispatch tables instead of duplicating all GPU resources per region.

This should be a measured optimization, not a precondition for correctness.

### 9.2 P1 - one load-bearing 2:1 seam

The paper contributes almost nothing to the mathematical seam:

- no authoritative fine-subface flux ledger;
- no scalar restriction/prolongation;
- no 2:1 velocity extension;
- no symmetric composite pressure row;
- no free-surface seam case; and
- no reflection/hydrostatics validation.

Equal-resolution aprons can inform the mechanics on either side of the seam, but not the seam itself. P1 remains the decisive research and implementation gate.

### 9.3 P2 - many bodies

The paper strongly supports batching active blocks and compacting worklists rather than serially dispatching tiny bodies. Its global sparse solve also supports the rule that connected bodies/regions must be solved as a component, not as isolated objects.

It does not address asleep analytic bodies, connection detection, merge/split transactions, or independent temporal cadence for disconnected components.

### 9.4 P3 - dynamic regrid

GPU incremental allocation is relevant here, but only after the physical regrid transaction is specified. The correct order for reuse is:

1. determine desired region levels from activity;
2. close and validate topology;
3. compute conservative state transfer and receipts;
4. validate the composite operator;
5. commit accepted state; and
6. update pooled GPU allocation/worklists incrementally.

The paper mainly helps with step 6. It must not be used to justify skipping steps 2-5.

### 9.5 P4 - controls and overlays

GVDB's occupancy and memory statistics suggest useful instrumentation: active bricks, brick fill ratio, topology additions/removals, pool high-water mark, apron cost, and worklist duplication. The paper provides no activity-quality visualization or user-facing fidelity control.

### 9.6 P5 - conditional scale-out

This is the strongest fit. The current plan lists sparse residency inside large regions as an optional measured optimization. GVDB directly demonstrates the feasibility of sparse brick residency and shared GPU work over a large logical domain.

Even here, adoption requires care:

- preserve one physical spacing inside each region/level;
- distinguish unallocated exterior from liquid/air state required by the surface method;
- allocate all stencil, extension, pressure, and rendering support, not only occupied liquid;
- keep same-level sparse brick boundaries invisible to conservation tests; and
- treat pool exhaustion as an explicit policy event.

## 10. Architectural lessons worth importing

### 10.1 Use one block graph, not one numerical universe per block

Regions are ownership and fidelity units. They should not automatically become independent pressure domains or fully duplicated runtime stacks. A global block graph can retain region and connected-component IDs while batching common kernels.

### 10.2 Pool by physical resolution level

A practical synthesis is one or more shared atlases/buffer pools per dyadic `dx`, with each allocated block carrying:

- world-space origin and level;
- accepted topology generation;
- region and connected-component IDs;
- neighbor/seam incidence ranges;
- channel offsets;
- active bounds and support flags; and
- lifecycle state.

Same-level neighbors use direct halo copies. Different-level neighbors use the explicit seam operators from our plan. This preserves the useful GVDB storage idea without confusing its addressing hierarchy with physical AMR.

### 10.3 Separate allocation lists from physics lists

The paper's subcells show that one list need not serve every kernel. We likely need distinct compact lists for:

- allocated blocks;
- liquid/active pressure cells;
- transport and extension support;
- surface reconstruction;
- same-level halos;
- coarse/fine atomic subfaces;
- connected-component reductions; and
- activity/coarsening candidates.

This also makes empty-block cost visible instead of hiding it in region dispatch.

### 10.4 Make halo updates explicit and typed

Apron updates are frequent in the paper. Our version should type them by quantity because mass, signed distance, face flux, pressure iterate, marker state, and activity evidence do not share identical transfer rules. At coarse/fine seams the operation must be named restriction, prolongation, or flux synchronization rather than generic "apron update."

### 10.5 Keep pressure global across a connected component

GVDB is a useful counterexample to the idea that sparse blocks force local projection. Its matrix-free application and global reductions span the sparse domain. Our implementation can similarly use block-local kernels while preserving a single component-wide operator and coarse correction.

## 11. Claims the paper cannot support

The following claims would overstate the evidence:

1. "GVDB demonstrates activity-adaptive fluid resolution." It demonstrates occupancy-adaptive allocation at a fixed voxel spacing.
2. "The GVDB hierarchy is an AMR hierarchy." It is a sparse addressing hierarchy whose leaf bricks contain the simulated voxels.
3. "Aprons solve coarse/fine seams." They duplicate equal-resolution neighbor data.
4. "Incremental topology solves conservative regridding." Particle authority avoids the Eulerian transfer problem.
5. "Matrix-free CG solves our composite pressure problem." It establishes a sparse global solve pattern, not our operator or preconditioner.
6. "The paper validates real-time performance." It reports substantial acceleration, but its large examples take far more than a real-time frame budget.
7. "Particle influence boxes implement predictive activity." They cover interpolation support, not future accuracy demand.
8. "Sparse allocation alone makes calm water cheap." A large connected pressure solve, surface support, and long waves can require work far beyond currently occupied liquid voxels.

The decision record in the current handoff should therefore use wording such as "GVDB demonstrates GPU sparse-block storage and execution" rather than "GVDB demonstrates multiresolution adaptation."

## 12. Concrete implications for our plan

### 12.1 Keep the current architecture decision

Nothing in the paper argues for returning to the superseded per-cell octree plan. Dense uniform blocks connected by explicit 2:1 seams are at least as compatible with GVDB's successful execution model and much easier to reconcile with the existing uniform solver.

### 12.2 Treat P1 as still unresolved

The paper supplies no reason to lower the P1 kill gate. The one-seam vertical slice must independently prove:

- exact scalar conservation;
- one authoritative seam-normal flux;
- symmetric composite pressure coefficients;
- acceptable hydrostatic error;
- bounded wave reflection and phase error; and
- stability through repeated transfers.

### 12.3 Move sparse residency behind measurement

GVDB makes sparse residency credible, not mandatory. It belongs where the current plan places it: conditional scale-out after the uniform lane and seam are correct. Adding a tree, atlas, and worklist system before measuring region occupancy would increase implementation risk without addressing the load-bearing physics.

### 12.4 Prototype shared pools before a hierarchy

If per-region overhead is the immediate concern, the smallest experiment is not a full GVDB clone. It is a shared buffer pool and indirect dispatch list for many equal-resolution blocks. A tree becomes necessary only if world-to-block lookup or highly sparse large-region residency is actually a bottleneck.

### 12.5 Preserve the particle route as a separate alternative

If conservative Eulerian regrid transfer proves intractable, a particle-authoritative topology route deserves a separate design study. That study must compare the entire method, not just topology cost. It would need acceptance tests for exact mass, noise, surface quality, memory, sorting, reseeding, walls/inflows, and compatibility with the current pressure lane.

It should not be mixed incrementally into the existing method under the assumption that FLIP transfer is equivalent to our current ownership model.

## 13. Focused experiments suggested by the paper

These experiments extract the useful claims without prematurely adopting its whole architecture.

### Experiment A - pooled equal-resolution blocks

Run many disconnected uniform blocks from shared GPU buffers and compact dispatch lists. Compare against one solver allocation/dispatch stack per region.

Measure:

- fixed bytes per block/region;
- dispatch and bind overhead;
- active-cell throughput;
- empty-cell tax;
- pool fragmentation; and
- maximum practical body count.

This tests the paper's most applicable result for P2.

### Experiment B - same-level halo ABI

Implement explicit typed halos between two same-resolution uniform blocks. Require bitwise or tolerance-level equivalence with one monolithic grid for advection, divergence, projection, and surface updates.

This isolates the reusable apron pattern before adding 2:1 mathematics.

### Experiment C - global sparse pressure dispatch

Apply the existing matrix-free operator over a compact active-block list while maintaining global component reductions. Compare convergence and result against the current dense-domain reference.

This tests sparse execution without changing the operator.

### Experiment D - storage tile versus compute tile

Sweep allocation-block and workgroup-tile sizes independently. Measure fill ratio, memory overhead, halo traffic, cache behavior, and kernel occupancy on target WebGPU devices. Do not import the paper's 32^3 brick and 4^3 subcell choices as constants.

### Experiment E - P1 seam remains independent

Run the current plan's one-seam hydrostatic, constant/linear reproduction, divergence, mass, pulse transmission, and oblique-wave tests. GVDB infrastructure should be considered successful only if same-level sparse boundaries are invisible and it does not weaken these seam gates.

## 14. Go/no-go assessment

### Go

Use the paper as a design reference for:

- pooled sparse blocks;
- GPU-resident topology maintenance;
- compact active worklists;
- separating allocation and compute granularity;
- explicit same-resolution halos;
- matrix-free global sparse stencil execution;
- batched reductions; and
- optional sparse residency inside large uniform regions.

### No-go

Do not use the paper as the technical basis for:

- the activity-to-resolution policy;
- 2:1 coarse/fine coupling;
- conservative Eulerian regrid transactions;
- the composite free-surface pressure operator;
- wave-fidelity claims;
- sleep/wake/component lifecycle; or
- the P1 acceptance decision.

### Final applicability judgment

GVDB FLIP answers a supporting scalability question: **how can a GPU fluid method operate over a large, dynamically sparse set of regular voxel blocks?**

It does not answer the defining scientific question in our goal: **how can activity change physical resolution while conservation, stability, global pressure coupling, and long-wave plausibility survive the moving coarse/fine boundaries?**

The paper is therefore highly relevant infrastructure literature and weak evidence for the core multiresolution method. Its best role in the project is to shape the execution substrate after, or alongside, an independently validated uniform-region seam - not to substitute for that validation.

## 15. Paper evidence index

Use these locations when tracing a claim back to the paper:

- Figure 2 and Section 3: sparse hierarchy, node pools, dense leaf bricks, texture atlas, and aprons.
- Algorithm 1 and Section 4: per-frame topology update, volume clear, particle binning, transfers, apron updates, divergence, pressure, and particle advance.
- Sections 5.1 and 5.2: full and incremental GPU topology construction.
- Section 6 and Figure 4: particle influence boxes, subcells, and the split between storage and compute granularity.
- Section 7: sparse particle-to-voxel and voxel-to-particle transfer.
- Section 8 and Algorithm 2: matrix-free sparse CG, reductions, and apron-dependent stencil application.
- Tables 1 and 2: topology and solver timing comparisons.
- Tables 3 and 4: per-frame phase costs and memory usage.
- Section 10: limitations and future work, including collision constraints, particle sorting cost, and solver improvements.
