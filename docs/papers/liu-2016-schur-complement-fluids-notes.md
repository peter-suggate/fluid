# Resolution-adaptive regional fluids: reading notes on Liu et al. (2016)

Source extraction: `liu-2016-schur-complement-fluids.txt`

## Goal being evaluated

Support very large, sparse fluid scenes in which localized high-energy activity
uses fine Eulerian grids, while calmer connected regions use progressively
coarser grids. Resolution changes introduce nonconforming coarse/fine faces and
T-junctions, but deliberately trade local accuracy and dissipation for lower
work and memory.

## Verdict

The paper is **consistent with decomposing and coupling regional pressure
solves**, but **does not provide the required multiresolution physical
discretization**.

Its most useful idea is static condensation: solve or approximate each
subdomain interior independently, eliminate those interior degrees of freedom,
and couple the subdomains through a much smaller Schur-complement interface
system. This supplies a possible higher-level mechanism for joining regional
solves without one monolithic voxel solve.

However, every physical subdomain in the paper is cut from one large uniform
grid. Its Section 6.3 octree-like adaptivity only approximates the harmonic
interior used to apply the Schur interface operator; it is not the grid on which
fluid pressure, velocity, or transport are discretized. The authors explicitly
leave adaptive discretizations and dynamically changing adaptive partitions to
future work.

## What matches the intended approach

### 1. Region-local interior solves with explicit interface coupling

Section 4 reorders pressure unknowns into independent interiors
`Omega_1 ... Omega_k` and an interface `Gamma`. One Schur application is:

1. solve all subdomain interiors independently;
2. accumulate their flux contribution on `Gamma`;
3. solve the interface system; and
4. solve the interiors again using the corrected interface values.

The regional work is parallel and communication is restricted to the
interface. A one-cell-thick separator is sufficient for a seven-point Cartesian
stencil when it completely decouples the two interiors.

### 2. Regional interiors can be represented by their boundary response

Section 6 interprets `A_Gamma,i A_i,i^-1 A_i,Gamma` as:

1. impose interface pressure as a Dirichlet condition;
2. harmonically extend it through the subdomain interior; and
3. return the resulting Laplacian/flux on the interface.

This is a discrete Dirichlet-to-Neumann response. For the intended system, it
suggests that a regional solve can expose a smaller coupling surface to the
outer scheduler. The interior still participates through an accurate or
bounded approximation of its interface response.

### 3. The interface operator remains matrix-free and SPD

The exact Schur complement would be dense. The paper avoids constructing it,
uses an assembly-free fixed-point smoother and multigrid hierarchy on the
interface, and retains a symmetric-positive-definite preconditioner suitable
for outer CG. This is compatible with the repository's existing matrix-free
row/face authority and fail-closed solver discipline.

### 4. Sparse uniform storage is already part of the design

Section 7 uses SPGrid blocks over sparse active domains. This aligns with the
GVDB-inspired uniform page proposal: fixed Cartesian interiors, sparse
residency, compact active worklists, and regular streaming operations.

### 5. The hierarchy can recurse

The future-work discussion proposes using the entire preconditioner as a
subdomain solver inside a deeper hierarchy. That is a natural model for a vast
scene: pages form local regions, regions form clusters, and only condensed
interface information reaches the next level.

## What does not match yet

### 1. Physical resolution is not driven by fluid activity

The paper partitions for hardware locality and memory capacity. Every physical
subdomain retains the background grid's cell size. Its Section 6.3 adaptivity
coarsens a harmonic interior approximation farther from the partition
interface; it does not identify calm flow or lower the physical grid resolution
used by pressure, velocity, or transport.

### 2. It remains a globally converged pressure solve

The regional solves form a preconditioner for top-level PCG over the complete
pressure problem. The interface solve is also global over `Gamma`. The method
therefore avoids a monolithic *interior computation*, but it does not eliminate
global algebraic coupling.

This is useful rather than contradictory: the interface level can be far
smaller than the voxel problem. If the target forbids even a coarse/global
interface operation, the resulting method is an inexact local projection and
must carry an explicit divergence and wave-error budget.

### 3. Extra local work without boundary exchange is ineffective

Section 2 explicitly warns that several additional smoothing iterations inside
a partition provide little convergence benefit without synchronization at the
partition boundary. This is directly relevant to residual-driven hot-region
iterations: every extra regional solve must update or consume its interface
state, not merely refine an interior against stale boundary values.

### 4. The method only addresses pressure projection

It does not reduce the cost of advection, VOF/density transport, redistance,
velocity extension, rigid coupling, topology publication, or rendering. Those
stages need operators on the same multiresolution topology and conservative
coarse/fine transfer rules.

### 5. Its performance crossover is at very large scale

For examples around 100 million degrees of freedom or less that fit in one GPU,
the authors report homogeneous GPU MGPCG as 3-5x faster than DDPCG. DDPCG wins
when the problem no longer fits accelerator memory and the alternative becomes
CPU-only MGPCG. A single-adapter WebGPU implementation must therefore prove a
different benefit: sparse residency plus strongly uneven regional resolution,
not the paper's CPU/GPU offload economics.

### 6. Thin connectors and topology changes are difficult

The paper reports that narrow connectors reduced interface size but increased
coarsening error and required more PCG iterations. It also lists disappearing
Neumann gaps, merging regions on coarse grids, adaptive interfaces, and dynamic
partitioning as unresolved concerns. These cases are central to interconnected
fluid regions and should be first-class tests.

## Corrected multiresolution interpretation

The desired structure is closer to an octree or block-structured AMR method:

- hot regions contain small cells;
- calm regions contain larger cells;
- dyadic, globally aligned regions meet at nonconforming faces;
- pressure and face-normal velocity remain coupled across those faces; and
- refinement follows activity with a predictive transition band and
  hysteresis.

Liu et al. can still supply a regional Schur-complement layer above this
discretization, but it does not solve the T-junction problem. The direct
references are Losasso et al. (2004), Aanjaneya et al. (2017), and Ando and
Batty (2020). The repository's current adaptive pressure rows, reciprocal
faces, transition subfaces, Power option, constrained T-junction nodes, and
candidate/accepted topology transaction already implement much of that
foundation.

### Dissipation is acceptable; broken conservation is not

Coarse transport will dissipate short wavelengths, vorticity, and fine surface
features. That can be an intentional level-of-detail policy in a calm region.
But a naive T-junction can also cause errors that are not benign dissipation:

- asymmetric pressure coefficients and loss of the SPD solve;
- unequal flux on the two sides of one geometric face;
- divergence or volume creation at resolution transitions;
- parasitic currents in hydrostatic water;
- pressure/velocity discontinuities and visible wave reflection; and
- topology-dependent rigid-body impulses.

Aanjaneya et al. specifically report that the simple symmetric Losasso
gradient becomes first-order at T-junctions and can generate hydrostatic
parasitic currents. Their Power-diagram construction and the later Ando-Batty
operator address accuracy while preserving symmetry. Whether the remaining
transport dissipation is visually acceptable should be measured only after the
pressure and conservation invariants pass.

### Required coarse/fine face contract

For a coarse face incident on finer cells:

1. partition the coarse face into geometrically matching fine subfaces;
2. give every subface one authoritative normal velocity and open fraction;
3. use the same subface flux with opposite signs in the adjacent divergence
   equations;
4. assemble pressure coefficients from shared area and centre-distance metrics
   so opposite row incidences are bitwise reciprocal;
5. project each subface once, then area-average only when a coarse consumer
   explicitly needs a coarse face value; and
6. distinguish storage/dispatch region boundaries from actual solid, free
   surface, or world boundary conditions.

Hanging scalar or nodal velocity samples should be constrained by their coarse
edge/face masters, not treated as independent degrees of freedom. Refinement
ratios should initially be dyadic and 2:1 balanced, matching the existing
octree topology and limiting the number of transition configurations.

### Activity should select resolution, not solve tolerance

The pressure system should still converge to its normal residual tolerance on
the accepted multiresolution grid. Activity changes the grid and therefore the
number of unknowns; it should not initially change how accurately those
unknowns are solved.

An activity/refinement score should combine:

- vorticity, strain, and kinetic-energy density;
- free-surface curvature, normal velocity, and transported volume;
- rigid contacts, inflows, and solid motion;
- characteristic/CFL reach over the next topology epoch;
- pressure/divergence and coarse/fine interface residuals; and
- recent maximum activity with spatial dilation and temporal hysteresis.

Energy alone is not sufficient. A low-amplitude long wave can have low local
energy but a phase-sensitive wavelength, and a slow moving solid can require
fine cut-cell geometry. Refinement should be predicted ahead of a moving wave,
not triggered only after high-frequency content has entered a coarse region and
already been lost.

### Dynamic refinement/coarsening contract

- Refinement prolongation and coarsening restriction must conserve liquid mass
  exactly to the chosen arithmetic precision.
- Face velocity transfer must preserve integrated normal flux; momentum/energy
  changes introduced by coarsening must be measured rather than hidden.
- Pressure, surface, velocity, rigid-boundary state, topology, and transition
  faces must publish as one accepted generation.
- A refine/coarsen hysteresis band should prevent the activity boundary from
  oscillating around a travelling wave.
- Backtrace and stencil support must be included before the new topology
  commits; a fast front cannot request missing fine cells after sampling them.

## Relevance of the Schur-complement paper after the correction

The Schur layer is optional rather than foundational for a single connected
adaptive grid. It becomes useful when the scene is partitioned into large
regional allocations or multiple devices and the adaptive pressure operator is
too large for one monolithic execution. Each region may then have its own
multiresolution interior operator, while a Schur interface system couples their
boundary unknowns.

That extension is nontrivial. The paper assumes uniform subdomain stencils and
aligned interface degrees of freedom. With different regional resolutions, the
interface must be expressed in shared geometric subfaces or in a common mortar/
variational space. The interface approximation must remain fixed, symmetric,
and positive definite within PCG. The paper itself identifies adaptive
interfaces and dynamic partitioning as future research.

## Recommended prototype order

1. Use the existing adaptive Losasso/Power grid as the physical discretization;
   do not start by creating independent uniform solvers with interpolated
   boundary conditions.
2. Define activity-driven regional minimum leaf sizes and preserve 2:1 closure.
3. Verify manufactured coarse/fine pressure symmetry, hydrostatic balance, and
   exact paired-face flux before judging appearance.
4. Run a wave through a stationary fine-to-coarse-to-fine corridor and measure
   phase error, amplitude loss, reflection, volume, and divergence.
5. Make the refinement corridor move with the wave using predictive reach and
   hysteresis; compare against an all-fine reference.
6. Only after the adaptive grid is sound, test whether a regional Schur layer
   improves memory locality or scaling beyond the current global compact MGPCG.

## Acceptance scenes and metrics

### Scenes

- huge coarse basin with one fine localized breaking wave;
- travelling wave crossing fine/coarse transitions at multiple angles;
- low-amplitude, long-wavelength swell over a mostly coarse domain;
- narrow connector straddling a refinement boundary;
- hydrostatic tank with an intentionally complex T-junction sheet;
- moving rigid body crossing coarse/fine regions; and
- repeated refine/coarsen cycle around a stationary interface.

### Metrics

- cell/face/node counts and time by leaf size;
- pressure-matrix symmetry/SPD and reciprocal-face errors;
- maximum/RMS divergence and paired subface-flux error;
- total volume, momentum, and rigid coupling impulse;
- hydrostatic parasitic velocity;
- wave phase speed, amplitude dissipation, and reflected energy;
- topology churn and remap error; and
- memory/work reduction relative to the all-fine reference.

## Go/no-go interpretation

The approach succeeds when compute scales with the fine high-activity volume
plus a narrow graded transition, while coarse regions preserve the important
low-frequency motion and remain conservative. Additional dissipation in calm
regions may be acceptable. Pressure asymmetry, seam volume error, hydrostatic
currents, unstable topology remaps, or strong artificial reflection are not.

Liu et al. remain relevant for coupling large regional solves, especially at
multi-device scale. They are not the numerical justification for the
coarse/fine T-junction discretization itself.

## Reference

Haixiang Liu, Nathan Mitchell, Mridul Aanjaneya, and Eftychios Sifakis. *A
Scalable Schur-complement Fluids Solver for Heterogeneous Compute Platforms*.
ACM Transactions on Graphics 35(6), 2016.
<https://doi.org/10.1145/2980179.2982430>
