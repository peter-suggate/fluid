# Adaptive Mass-Conserving GPU Liquids

Implementation research and plan, 2026-08-14

Status: Milestone 1 implementation in progress. The existing uniform method
remains the behavioral oracle and is not the implementation substrate.

Current implementation checkpoint (2026-08-14):

- CM12 constants and pure formulas live in `lib/core/cm12-numerics.ts` and are
  consumed by the production uniform shaders and the adaptive CPU oracles.
- One row builder covers the `8+8`, `4+4`, `8+4`, and reflected `4+8`
  two-tile pressure topologies on x/y/z. The mixed seam has one authoritative
  five-cell port per coarse face patch and derives divergence as the weighted
  negative transpose of its gradient.
- Focused executable receipts cover algebraic A/B symmetry, a matrix-free
  pressure projection, volume-weighted conservative transport with persistent
  gamma, and WGSL pressure-operator parity. `npm run
  acceptance:adaptive-mass:m1` runs this checkpoint without adding a unit-test
  matrix.
- Free-surface pressure (using one SPD ghost fraction per composite row),
  composite gamma diffusion/sharpening, GPU transport-operator parity, and a
  symmetric density-slice A/B are now executable companion receipts.
- This is not yet the completed production method. GPU-authored arbitrary
  characteristics, velocity advection/extension, a GPU pressure solve, solid
  coupling, and a fully coupled rendered CM12 step remain open.

Working method id: `adaptive-mass`

## 1. Decision

Build a new, GPU-resident, sparse multiresolution liquid method around fixed
world-space tiles whose internal cell resolution can vary. The authoritative
state is a compact set of leaf control volumes and staggered face samples. It is
not a dense texture with sparse dispatch, and it is not an octree traversal in
the inner loop.

The first milestone is deliberately much smaller than the eventual sparse
system, but it attacks the highest-risk numerical problem first:

> Two face-adjacent logical tiles, one `8 x 8 x 8` and one `4 x 4 x 4`, with a
> fixed 2:1 resolution transition. Run the CM12 step across that seam with
> conservative mass transport, a globally coupled pressure projection, stable
> velocity sampling, and no visible seam artifact.

Topology is frozen in Milestone 1. There is no allocator, camera policy,
refinement transaction, sleeping, or general sparse topology yet. Those pieces
are valuable only after this seam is numerically credible.

The end-state has three non-negotiable properties:

1. Empty tiles are absent from every physics workset. An empty world produces an
   indirect dispatch of `(0, 1, 1)` and no field clears, copies, or dense census.
2. Regular tile interiors use branch-free, level-specialized GPU kernels.
   Coarse/fine work is isolated into compact seam worksets rather than paid by
   every cell.
3. Resolution is selected independently from residency. A tile can be resident
   because a characteristic or pressure solve needs it while being coarse
   because its physics and screen-space error permit it.

## 2. Why the first milestone uses equal-size logical tiles

“Two adjacent bricks at different resolution” has two possible geometries.

- In conventional block AMR, every storage brick has the same cell count and a
  fine brick has half the world-space width. In 3D, four fine bricks are needed
  to cover one coarse-brick face. Exactly two bricks can only meet over one
  quarter of that face.
- In the proposed method, logical tiles have equal world-space extents while
  their internal resolution varies. An `8^3` tile and a `4^3` tile therefore
  share one complete face. One coarse face sample covers a `2 x 2` patch of fine
  faces.

The second interpretation is used here. It matches the product requirement that
a tile become lower resolution while remaining the same spatial region, gives a
literal two-tile test, and makes the eventual tile directory a simple bounded
3D array. A tile at level `l` has

```text
cells per axis: B_l = B_0 / 2^l
cell width:     h_l = H / B_l = 2^l h_0
cell volume:    C_l = h_l^3
```

where `H` is the fixed tile width and the initial maximum `B_0` is 8. The useful
levels are therefore `8^3`, `4^3`, `2^3`, and `1^3`. Neighbor levels may differ
by at most one.

This is a sparse, adaptive multiblock grid rather than a classical octree. It
retains the parts of block AMR that matter—composite operators, conservative
restriction/prolongation, graded transitions, and synchronization—without
putting pointer-chasing tree traversal in every shader invocation.

## 3. Existing repository baseline

### 3.1 What the uniform method actually does

The current reference is
[`lib/methods/uniform/webgpu-uniform-reference.ts`](../lib/methods/uniform/webgpu-uniform-reference.ts)
and its WGSL modules. Its physics order follows the locally downloaded primary
paper, [Chentanez and Müller (CM12)](papers/massConservingLiquids.txt):

1. FIM-based velocity extension into an air band.
2. Persistent-gamma conservative density advection.
3. Axis-by-axis gamma diffusion.
4. Density sharpening and local mass return.
5. Partial-solid excess removal.
6. Semi-Lagrangian or bounded MacCormack velocity advection and body forces.
7. CM11a separating-boundary LCP projection using multigrid.
8. Rigid coupling, presentation reconstruction, and diagnostics.

This order, the liquid threshold, the CM12 sharpening constants, pressure
classification from `rho / openVolume`, the force placement, and the default
time step are the behavioral contract for `adaptive-mass`.

### 3.2 Why the current “active region” is not sparse

The uniform solver's active mode reduces dispatch to one rolling axis-aligned
box, but all authoritative fields remain full-domain 3D textures. It cannot
skip holes inside the box, it cannot release memory for empty space, and it
cannot represent two resolutions. External sources can also force a dense
full-domain census, and several full-volume texture copies remain.

That is a useful optimization for the uniform oracle, not a foundation for the
new method.

At cube resolutions, the currently planned uniform allocations are roughly:

| Resolution | Uniform host fields | Pressure hierarchy | Total before rigid extras |
|---:|---:|---:|---:|
| `64^3` | 50.9 MiB | 41.9 MiB | 92.8 MiB |
| `128^3` | 395.4 MiB | 317.3 MiB | 712.7 MiB |
| `256^3` | 3,117.3 MiB | 2,470.8 MiB | 5,588.2 MiB |

The new method must eliminate allocation proportional to the finest enclosing
box, not merely reduce thread count within that allocation.

### 3.3 Infrastructure worth reusing

Reuse concepts and generic core facilities, not method-specific numerical code:

- GPU compilation and pass scheduling.
- Exact/compensated reductions and asynchronous diagnostics.
- Indirect workset headers with a canonical zero dispatch.
- Generation-stamped, fail-closed A/B publication.
- Compact brick residency flags and retirement hysteresis.
- GPU radix sort and mark/rank/scatter building blocks.

`lib/core/webgpu-fluid-brick-residency.ts` already demonstrates a useful
transactional residency model. The workset ABI in
`lib/methods/octree-shared/webgpu-octree-worksets.ts` is also good, but the new
method must not import a sibling method package. Truly generic pieces should be
moved or reimplemented in `lib/core/`, with compatibility re-exports if needed.

The method catalog explicitly forbids sibling-method imports. `adaptive-mass`
imports `core`, not `uniform`, `losasso`, `power`, or `octree-shared`.

### 3.4 The uniform oracle needs its regression net restored

The package split removed many method-level uniform tests, including active
region, FIM, gamma, coefficient, pressure-boundary, and spherical-container
tests. Smoke lanes remain, but Milestone 1 needs small numerical comparisons.

Do not “clean up” the uniform shaders while building the new method. First
capture CPU vectors and GPU readbacks from the current code. A same-resolution
two-tile configuration must match those vectors before a 2:1 seam is enabled.

## 4. Research conclusions

The design is based on primary papers plus direct inspection of this repository.
The downloaded papers are source material. Existing local handoffs,
implementation maps, and design documents were treated as historical input,
not authority.

### 4.1 Sparse GPU layout

- [GVDB FLIP](https://people.csail.mit.edu/kuiwu/GVDB_FLIP/gvdb_flip.pdf)
  shows the practical value of dense voxel bricks, compact topology, GPU-side
  updates, and smaller compute tiles inside larger storage bricks. It also
  exposes the central failure mode: one wet voxel can make a large brick pay
  for every voxel, and iterative apron refreshes are expensive.
- [DCGrid](https://doi.org/10.1145/3522608) uses small uniform blocks, a GPU
  directory, precomputed neighbor/apron indices, parent closure, and a dense
  coarse solve. Its reported break-even is not automatic: sparse overhead can
  lose to a uniform grid at high occupancy. Sparse is a measured regime, not a
  universal speedup.
- [NanoVDB](https://research.nvidia.com/labs/prl/nanovdb/nanovdb2021.pdf) is a
  strong compact read-only traversal design. Its pointerless hierarchy is good
  inspiration for publication and lookup, but mutable simulation authority
  should use stable page slots and transactional directories rather than
  rebuilding a read-optimized tree every step.
- [Cirrus](https://wang-mengdi.github.io/proj/25-cirrus/) is recent evidence
  that GPU-resident adaptive grids with `8^3` tiles can scale well for sparse
  fluid workloads. Its smoke/particle formulation is not a liquid seam
  discretization and is not copied here.

Implication: start with `8^3` high-resolution tiles and `4^3`/64-thread compute
chunks, but benchmark `4^3`, `8^3`, and `16^3` storage tiles before freezing the
production ABI. Keep regular and seam work in separate lists.

### 4.2 Adaptive pressure is one global problem

- [Ando and Batty 2020](https://cs.uwaterloo.ca/~c2batty/papers/Ando2020/Ando2020.pdf)
  gives a practical staggered octree pressure operator at 2:1 T-junctions. Its
  gradient uses one coarse face velocity and an average of four fine pressure
  samples in 3D. The divergence is the negative transpose of that gradient,
  giving a variational, symmetric operator. Their SPD free-surface treatment
  was visually indistinguishable from their strictly second-order but
  non-symmetric variant in their tests.
- [A Conservative Adaptive Projection Method](https://ccse.lbl.gov/Publications/almgren/abchw.pdf)
  and modern block-AMR practice show why independently projecting bricks is
  wrong: coarse/fine fluxes and the pressure correction must be synchronized in
  one composite solve.
- [Lai et al. 2020](https://uwspace.uwaterloo.ca/items/d5b88286-d408-47a0-b8d7-d7dcb2be4adc)
  provides substantially better multigrid strategies for the separating-solid
  LCP than the original CM12-era scheme. This is the likely post-Milestone-1
  direction once the composite SPD operator is trusted.

Implication: there is one pressure unknown per active liquid leaf cell and one
authoritative face/port flux. A tile is never projected in isolation. The first
milestone uses a tightly converged solve so pressure-solver error cannot disguise
a bad seam discretization.

### 4.3 Conservation must use physical volume

CM12's matrix is column-stochastic because every uniform cell has the same
volume. On a variable-resolution grid, conserving `sum(rho)` is wrong. The
invariant is

```text
M = sum_i V_i rho_i
```

where `V_i` is the cell's open physical volume. The fully conservative
[Lentine–Grétarsson–Fedkiw method](https://physbam.stanford.edu/~fedkiw/papers/stanford2010-01.pdf),
[adaptive conservative semi-Lagrangian work](https://epic.awi.de/id/eprint/15119/1/Beh2005a.pdf),
and block-AMR synchronization all lead to the same rule: transport integrated
mass and reconcile coarse/fine fluxes. Refinement and coarsening are also
transfers of integrated mass, never averages of density without a volume
factor.

### 4.4 WebGPU constraints

The current [WebGPU specification](https://www.w3.org/TR/webgpu/) supports
storage buffers that are also indirect-dispatch buffers, which is sufficient
for a completely GPU-authored work graph. The current
[WGSL specification](https://www.w3.org/TR/WGSL/) exposes optional subgroups,
but subgroup size and invocation mapping are implementation-selected.

Implication: the portable baseline is 64-thread workgroups, workgroup memory,
and buffer atomics. Subgroup reductions may be an optional pipeline variant;
the numerical result and correctness cannot depend on a 32-lane warp.

## 5. Milestone 1 contract

### 5.1 Domain

The initial executable is a dedicated seam harness, not yet an interactive
product method.

```text
world tile 0                    world tile 1
+---------------------------+  +---------------------------+
| fine: 8 x 8 x 8           |  | coarse: 4 x 4 x 4         |
| h = H / 8                 |  | h = H / 4 = 2 h_f         |
| 512 pressure cells        |  | 64 pressure cells         |
+---------------------------+--+---------------------------+
                         complete shared x face
```

The seam may be placed on any axis and the fine/coarse sides may be swapped.
The other world faces are periodic for pure transport tests or grid-aligned
static walls for liquid tests. There are no moving rigid bodies, cut solids,
inflows, topology changes, or surface tension in the first full-step gate.
Gravity, free surface, high-CFL transport, and CM12 sharpening are in scope.

These exclusions isolate the resolution seam. They are not exclusions from the
eventual method.

### 5.2 “Minimal divergence from uniform” rule

The method retains:

- CM12 stage order and persistent-gamma lifecycle.
- The same liquid threshold (`rho / openVolume > 0.5`).
- The same sharpening strength and local mass-return semantics.
- The same body-force placement.
- The same semi-Lagrangian velocity path for the initial comparison.
- The same pressure ghost-fluid rule on regular faces.
- The same time step and large-CFL trace integration.

Only four things may differ in Milestone 1:

1. Addressing an authoritative leaf sample.
2. Interpolation/restriction/prolongation when a trace crosses the seam.
3. Volume factors in conservative transport and gamma exchange.
4. The pressure gradient/divergence pair on a 2:1 seam face.

At equal resolution, the seam adapters disappear algebraically. A two-tile
`8^3 + 8^3` run and a two-tile `4^3 + 4^3` run must match the corresponding
dense uniform domain to the agreed floating-point tolerance.

### 5.3 Submilestones

#### M1.0 — Static geometry and algebra oracle

- CPU construction of the two-tile cell and face graph.
- Stable global IDs for cells, regular faces, and seam ports.
- Restriction/prolongation test vectors.
- Matrix assembly for the regular and seam pressure operator.
- Proof by test that `D = -G^T` under the selected volume inner product.
- Constant and linear field reproduction tests.

No WebGPU code is accepted before this oracle can report the exact coefficient
row for every seam port.

#### M1.1 — GPU state and composite projection

- Two fixed tile descriptors uploaded once.
- Level-specialized cell chunks: eight 64-cell chunks for the fine tile and one
  64-cell chunk for the coarse tile.
- Separate regular-face and seam-port worksets.
- Matrix-free application of the composite operator.
- A tightly converged SPD solve for static walls/free surfaces.
- Projection writes one authoritative seam-port velocity consumed by both
  tiles.

The first solver can be PCG with a simple diagonal or two-level preconditioner.
Production multigrid and the separating-solid inequality are intentionally not
allowed to complicate this gate.

#### M1.2 — Conservative CM12 surface transport

- Persistent gamma on variable-volume leaf cells.
- Backward and forward characteristics crossing the seam in both directions.
- Volume-weighted beta.
- Exact integrated-mass receipt.
- Seam-aware gamma diffusion.
- Sharpening and mass return using integrated mass.

#### M1.3 — Complete CM12 step and visual gate

- Composite velocity extension near the free surface.
- Semi-Lagrangian staggered velocity advection across the seam.
- Gravity and volume-correction divergence.
- Projection.
- Dense diagnostic materialization only for side-by-side rendering.
- Multi-step stability and visual comparisons against fine and coarse uniform
  controls.

Milestone 1 is complete only after M1.3. A pressure-only demo is evidence, not
the milestone.

## 6. Milestone 1 data model

### 6.1 Tile descriptor

```ts
interface AdaptiveMassTileDescriptor {
  key: number;                 // logical fixed-footprint tile key
  generation: number;          // state/topology generation
  level: 0 | 1;                // M1: 0 => 8^3, 1 => 4^3
  flags: number;
  originFineCells: [number, number, number];
  cellBase: number;
  faceBaseX: number;
  faceBaseY: number;
  faceBaseZ: number;
  neighborKeys: [number, number, number, number, number, number];
}
```

The production descriptor will support levels 0–3 and absent neighbors, but M1
uses two immutable records. World positions are derived from integer finest-grid
coordinates; do not accumulate floating-point tile origins.

### 6.2 Field storage

Use storage buffers, structure-of-arrays, level-segregated and tile-major:

- Cell integrated mass: positive fixed-point integer, logically 64-bit.
- Derived density `rho = mass / openVolume`: `f32` scratch/publication.
- Gamma A/B: `f32`.
- Pressure, RHS, residual, search vectors: `f32`.
- Cell open volume and free-surface phi/occupancy: `f32`.
- Staggered face velocity A/B and face-open geometry: `f32`.
- Transport beta and deposit receipts: integer/fixed-point scratch.

Buffers are preferable to a 3D texture atlas here because offsets and strides
vary by level, conservative scatters require atomics, and a direct page slot is
cheaper than atlas coordinate reconstruction. A read-only texture cache is a
future measured optimization, not authoritative state.

Within an `8^3` tile, use tile-major `4^3` microtiles: eight contiguous blocks
of 64 cells. A workgroup reads one block. A `4^3` tile is exactly one block.
Production `2^3` and `1^3` tiles are packed across a 64-cell linear work chunk
so low resolution does not leave most lanes idle.

### 6.3 Seam ports

The shared face has `4 x 4 = 16` seam ports. Each record identifies:

- one coarse cell;
- four fine cells sharing the same coarse face patch;
- one coarse-resolution normal velocity degree of freedom;
- the shared physical face area and center distance;
- optional free-surface and solid fractions;
- stable IDs for the transpose scatter into pressure rows.

This follows the practical Ando–Batty choice: fine child faces inherit the one
parent-face normal velocity at the transition. It deliberately filters
sub-coarse tangential variation exactly at the seam, but gives a compact,
stable, symmetric coupling and has a strong published visual precedent.

If visual gates show unacceptable loss, a later experimental mode can promote
the four fine subfaces to independent flux DOFs with a coarse flux constraint.
That is not the default because it changes the pressure space and substantially
increases solver complexity.

## 7. Conservative density transport across the seam

### 7.1 Authoritative variable

The authoritative liquid quantity is cell-integrated mass

```text
m_i = V_i rho_i
```

not `rho_i`. `V_i` includes physical cell volume and the open fraction. Density
is derived when CM12 needs its `0.5` surface threshold or its sharpening law.

This makes the following operations exact by construction:

- coarsen: sum child masses;
- refine: split one parent integer mass among children, distributing the final
  integer remainder deterministically;
- transport: distribute every donor's integer mass exactly once;
- sharpening and solid excess: debit before credit and close the receipt.

### 7.2 Volume-weighted CM12 beta

Let destination cell `i` trace backward to donor `j`. Let `w_ji^-` be the
nonnegative interpolation weight and `g_i` the backward-advected persistent
gamma used by CM12. The raw density operator coefficient is

```text
a_ij = g_i w_ji^-
```

Uniform CM12 computes `beta_j = sum_i a_ij`. With unequal cell volumes, the
correct donor mass fraction is

```text
beta_j = sum_i (V_i / V_j) a_ij
```

The backward mass fraction from donor `j` to receiver `i` is then

```text
t_ij^- = (V_i / V_j) a_ij / max(1, beta_j)
```

For `beta_j < 1`, forward tracing supplies the missing donor mass fraction

```text
d_j = 1 - beta_j
t_kj^+ = d_j w_jk^+
```

and the column sum of mass fractions is one. When all `V` are equal, these
expressions reduce to the existing CM12 scheme.

If expressed as density instead of mass, the corresponding deposit is

```text
delta rho_i = t_ij m_j / V_i = (V_j / V_i) t_ij rho_j
```

but the shader should never form that expression as the conserved operation.
It deposits `t_ij * m_j` integer mass, then derives density.

### 7.3 Seam sampler

Density/gamma interpolation must have nonnegative weights. Do not use an
unconstrained MLS interpolant for conservative transport; negative weights
would defeat positivity and the beta argument.

Use a uniform-compatible composite sampler:

- Within one level, return the exact eight trilinear donors and weights used by
  the uniform method.
- Sampling a coarse field at a fine query uses slope-limited, partition-of-unity
  prolongation. The expanded coefficients name authoritative coarse cells.
- Sampling a fine field at a coarse query first forms virtual coarse samples by
  volume-weighted restriction of `2^3` fine cells, then applies the ordinary
  coarse trilinear stencil. The coefficients are expanded back to authoritative
  fine donors.
- Duplicate donors are reduced in a fixed order. The maximum M1 stencil is
  bounded and asserted by the CPU oracle before a shader ABI is frozen.

The implementation recomputes this compact stencil in beta and deposit passes
instead of storing a global transfer matrix. A small explicit edge buffer is
permitted in the seam harness for forensic comparison, but it is not the
production design.

### 7.4 Exact integer receipt

WGSL core atomics are 32-bit. Represent positive mass totals as `(lo, hi)`
atomic `u32` words. An add atomically updates `lo`; overflow increments `hi`.
All reads occur after the dispatch boundary, so no consumer observes a torn
in-flight total.

For each donor:

1. Floor every weighted deposit to integer mass quanta.
2. Accumulate the deposited amount in a donor receipt.
3. Add `sourceMass - depositedMass` to a canonical receiver selected during the
   beta pass (lowest stable receiver ID with positive weight).
4. Fail the step if there is no receiver, the arithmetic overflows, or a trace
   reaches an unrepresented tile. Never silently drop or clamp mass.

The finest-cell quantum is selected from capacity and worst-case level so a
coarsest full cell and all expected temporary excess fit. Quantization error is
measured against the uniform `f32` oracle; conservation itself is exact in the
chosen integer measure.

External inflow and injection are separate source receipts. Open-boundary
outflow is a separate sink receipt. Neither is hidden inside transport error.

### 7.5 Gamma diffusion

The published axis diffusion assumes equal cell volumes. At a level transition,
equal and opposite changes in `rho` do not conserve mass.

Use a composite finite-volume exchange:

- regular same-level faces execute the existing equation exactly;
- a seam port evaluates four fine subface conductances from one immutable
  snapshot;
- gamma capacity is cell volume, and liquid transfer is represented as
  integrated mass;
- the interface relaxation is bounded by the sum of incident conductances so
  no coarse cell can be overdrawn by four simultaneous fine neighbors;
- deposits are pairwise conservative and resolved through the integer receipt.

M1.0 must compare this against two alternatives: volume-weighted star
equalization and an explicit finite-volume Jacobi step. Select the one that
simultaneously (a) reduces exactly to CM12 on equal cells, (b) remains positive,
(c) conserves mass, and (d) gives the closest gamma envelope to the fine
uniform control. This is one of the milestone's two genuine numerical research
items; it should not be buried in GPU plumbing.

### 7.6 Sharpening and local mass return

Compute the same CM12 `delta rho` from the local reconstructed density and local
cell width. Convert the debit to integrated integer mass before scattering.
Across a seam, gradient destinations are leaf cells located in physical space;
credits are scaled by receiver volume. The receipt must close before the new
mass bank is published.

No global “fix the total afterward” pass is allowed. A global correction can
hide a local seam leak while moving liquid to an unrelated component.

## 8. Composite pressure projection

### 8.1 One gradient and its transpose

Let `G` map cell pressures to authoritative face/port pressure gradients. Let
`W` be the diagonal dual-face volume, open-area, and free-surface weighting. The
projection system is

```text
G^T W G p = G^T W u*
u = u* - (dt / density) F G p
```

Signs may be arranged differently in code, but divergence must be the negative
transpose of the same gradient coefficients. Do not separately invent a
coarse-fine divergence stencil.

For a regular face, `G` is the existing centered difference.

For a 3D seam port with fine cell width `h`, one coarse pressure `p_c`, and four
fine pressures `p_f0...p_f3`, use

```text
G_port p = (0.25 (p_f0 + p_f1 + p_f2 + p_f3) - p_c) / (1.5 h)
```

up to orientation sign. The dual volume is the coarse face area times the
center distance:

```text
W_geometry = (2h)^2 (1.5h) = 6h^3
```

This is the Ando–Batty 2:1 T-junction operator. It is aligned with the face
normal, constant-preserving, and its transpose couples the one port flux back
to all five incident cells.

### 8.2 Free surface

Regular faces keep the current ghost-fluid fraction. On a seam port intersected
by the free surface, use the Ando–Batty SPD weighting constructed from the same
`G` row and leaf phi values. Clamp a negative degenerate face weight to the
documented positive floor and count it in a diagnostic histogram.

The strictly second-order non-symmetric variant is retained in the CPU oracle
as a comparison, not used for the first GPU solve. Milestone acceptance requires
the SPD result to be visually and quantitatively close enough to the fine
uniform control; published precedent is not a substitute for our own CM12
density tests.

### 8.3 CM12 volume correction

Compute `rho' = mass / openVolume` on each leaf. Preserve CM12's calibrated
small-excess slope using the leaf's local `h`:

```text
min(0.5 max(rho' - 1, 0), 1) / h
```

Add it to the cell RHS with the cell's physical volume measure. The fine and
coarse rows must therefore receive the same physical divergence, not the same
unscaled scalar RHS.

### 8.4 Solver strategy

M1 uses the simplest solver that makes operator error observable:

- matrix-free SPD apply over regular-face and seam-port worksets;
- diagonal scaling;
- PCG with deterministic compensated dot products;
- residual-based termination, not a fixed iteration count;
- a small dense CPU solve as the oracle for M1.0 cases.

Separating solid boundaries turn the system into a bound-constrained convex
problem/LCP. They are deferred from the first seam gate. After M1, implement a
projected FAS or multilevel active-set solver informed by Lai et al., retaining
the same composite operator. Do not force coarse/fine seam validation to share
its failure budget with a new LCP solver.

## 9. Velocity representation, extension, and advection

### 9.1 Authoritative staggered samples

Regular faces retain ordinary MAC samples. The normal component at a 2:1 seam
is the shared seam-port DOF. Fine child-face queries read that port value;
coarse queries read the same value. There is no duplicated coarse and fine copy
to drift out of sync.

Tangential components near the transition use component-aware restriction and
limited prolongation. Area-weighted restriction preserves integrated flux.
Same-level interpolation remains the exact uniform trilinear staggered sampler.

### 9.2 Velocity advection

Milestone 1 starts with the uniform method's default semi-Lagrangian path.
Trace integration is in world space and uses local leaf `h` only for locating
samples and clipping against boundaries. Crossing a tile or level boundary does
not reset or split the characteristic.

Bounded MacCormack is added only after the semi-Lagrangian seam passes, because
its limiter needs a trustworthy composite donor set. The limiter bounds against
exactly the authoritative donors used by the predictor, including expanded
coarse/fine donors.

### 9.3 Velocity extension

The interface authority is still `rho / openVolume`. Construct the narrow-band
graph from regular leaf neighbors and seam-port adjacency. An eikonal update at
the seam uses physical center distances and the same 2:1 connectivity as the
pressure graph.

M1 may use a compact frontier over all 576 cells. The full method creates an
extension workset only for interface/support tiles and stores coarser fallback
velocity farther into air. It never fills the whole logical domain.

## 10. Milestone 1 tests and acceptance gates

### 10.1 Algebraic invariants

All are hard gates:

- Constant scalar interpolation reproduces the constant on either side.
- Constant normal velocity produces zero divergence at the seam.
- A linear pressure field normal to the seam produces its exact gradient.
- CPU matrix symmetry error is at roundoff scale.
- `x^T A x >= 0` for generated liquid masks; at least one Dirichlet/free-surface
  constraint makes the test system positive definite.
- The GPU operator matches the CPU row application.
- A same-level two-tile run matches the dense uniform oracle.

### 10.2 Conservation gates

- Closed-domain integrated mass is bit-identical in the integer measure after
  every transport, diffusion, sharpening, and complete-step receipt.
- Refinement/coarsening test transfers close exactly even though dynamic
  topology is not yet enabled.
- Missing receiver, overflow, duplicate ownership, and stale generation are
  explicit failures; tests inject each fault.
- Forward and backward seam crossings conserve equally.

### 10.3 Physics cases

Run each with the seam on x/y/z and with fine/coarse sides swapped:

1. Uniform translation of a density slab normal to the seam.
2. Diagonal translation so characteristics cross the seam obliquely.
3. High-CFL translation matching the uniform method's intended large-step use.
4. Solid-body rotation/vortex straddling the seam.
5. Hydrostatic column whose free surface intersects the seam.
6. Drop crossing from fine to coarse and coarse to fine under gravity.
7. Dam-break front crossing the seam.
8. Thin sheet crossing the seam, with sharpening on and off.
9. Long settling run to expose a stationary kink or pressure leak.

Controls are a finest-uniform domain and a coarsest-uniform domain with the
same world extents, time step, and scene.

### 10.4 Quantitative go/no-go rule

Do not choose absolute visual-error thresholds before collecting oracle data.
Use these fixed relationships:

- Mass: exactly closed in the integer receipt.
- Stability: no NaN/Inf, negative mass, failed pressure convergence, or
  generation/receiver error at every time step accepted by the uniform control.
- Divergence: mixed-grid post-projection norms must be within the greater of
  the solver tolerance and two times the finest-uniform residual.
- Accuracy: for density, centroid, interface position, and kinetic energy, the
  mixed result must be closer to the finest control than the all-coarse control.
  The initial target is at least 25% reduction in each normalized all-coarse
  error. Cases that are analytically invariant must be exact instead.
- Direction bias: rotating the domain and swapping fine/coarse sides may not
  change a normalized metric by more than the mixed-grid error itself.
- Visual: no stationary crease, pinned surface, persistent reflection,
  checkerboard, or one-cell gap may identify the seam in density, velocity, or
  divergence views.

Store images, metrics, adapter info, commit, and method parameters in one
regression artifact. “Looks okay on my machine” is not a gate.

### 10.5 Performance observations, not M1 gates

Record GPU time and counts per stage, but do not reject a numerically correct
two-tile seam for being slower than the dense oracle. M1 is too small for useful
throughput conclusions. It must nevertheless demonstrate the intended shape:

- regular interior dispatch and seam dispatch are distinct;
- no dense full-domain texture is authoritative;
- no CPU readback occurs in a time step;
- every recurring count is GPU-consumable indirect state;
- fields swap descriptors/banks instead of copying whole volumes.

## 11. Milestone 1 implementation slices

### Slice A — CPU seam oracle

Add:

```text
lib/methods/adaptive-mass/adaptive-mass-grid.ts
lib/methods/adaptive-mass/adaptive-mass-seam.ts
lib/methods/adaptive-mass/adaptive-mass-pressure-oracle.ts
lib/methods/adaptive-mass/adaptive-mass-transport-oracle.ts
tests/adaptive-mass-seam.test.ts
tests/adaptive-mass-pressure-oracle.test.ts
tests/adaptive-mass-transport-oracle.test.ts
```

The oracle is small, deterministic, and intentionally readable. It emits the
exact donor and pressure coefficients used in GPU comparison tests.

### Slice B — Fixed GPU authority and worksets

Add:

```text
lib/methods/adaptive-mass/webgpu-adaptive-mass-state.ts
lib/methods/adaptive-mass/adaptive-mass-abi.ts
lib/methods/adaptive-mass/webgpu-adaptive-mass-worksets.ts
lib/methods/adaptive-mass/webgpu-adaptive-mass-addressing.wgsl.ts
tests/webgpu-adaptive-mass-state.test.ts
tests/webgpu-adaptive-mass-worksets.test.ts
```

Upload only the two immutable descriptors. Hard-code neither their order nor
the seam axis in WGSL.

### Slice C — Composite projection

Add regular-face/seam-port operator application, PCG vectors/reductions, ghost
fluid weighting, projection, and readback tests. First run all-liquid linear
fields, then free-surface masks, then hydrostatics.

### Slice D — Conservative transport

Add composite sampling, volume-weighted beta, integer mass deposits, gamma,
diffusion, and sharpening. Validate each receipt independently before composing
the step.

### Slice E — Full step and rendering artifact

Add velocity extension and advection, the CM12 stage coordinator, diagnostics,
and a test-only dense materializer. Register a non-interactive harness lane only
after all previous slices pass. Register the interactive method after M2 sparse
residency, not for the two-tile lab.

Suggested final package shape:

```text
lib/methods/adaptive-mass/
  method.ts
  harness.ts
  adaptive-mass-abi.ts
  adaptive-mass-grid.ts
  adaptive-mass-seam.ts
  webgpu-adaptive-mass-solver.ts
  webgpu-adaptive-mass-state.ts
  webgpu-adaptive-mass-worksets.ts
  webgpu-adaptive-mass-topology.ts
  webgpu-adaptive-mass-transport.ts
  webgpu-adaptive-mass-transport.wgsl.ts
  webgpu-adaptive-mass-extension.ts
  webgpu-adaptive-mass-extension.wgsl.ts
  webgpu-adaptive-mass-pressure.ts
  webgpu-adaptive-mass-pressure.wgsl.ts
  webgpu-adaptive-mass-sampling.wgsl.ts
  webgpu-adaptive-mass-presentation.ts
  adaptive-mass-diagnostics.ts
```

## 12. From the seam to the full sparse method

### M2 — Sparse single-resolution residency

Keep every resident tile at `8^3` and add GPU-authored occupancy:

1. Seed source tiles and swept external-source tiles.
2. Classify wet, interface, pressure, extension, and characteristic-support
   requirements from current resident tiles only.
3. Expand support by the actual swept characteristic reach, not one neighbor
   ring. A CFL-25 trace can cross several tiles.
4. Allocate candidate slots, initialize new state, validate capacity and all
   required destinations, then atomically publish a new generation.
5. Compact stage-specific worksets. Empty worksets dispatch `(0, 1, 1)`.
6. Retire dry tiles only after hysteresis and after no stage claims support.

M2 proves the first user goal: two disconnected droplets cost their occupied
and support tiles, not the enclosing AABB. An empty scene has zero physics work.

### M3 — Dynamic per-tile resolution

Enable levels 0–3 using the already-proven M1 seam:

- Refine immediately when physics requires it.
- Coarsen one level per topology epoch after hysteresis.
- Enforce 2:1 face-neighbor grading before publication.
- Sum child mass exactly on coarsening.
- Use limited density prolongation plus deterministic integer remainder on
  refinement.
- Restrict face velocity by area-weighted flux; prolong with a correction so
  child integrated flux equals the parent flux.
- Restrict gamma by volume and keep pressure only as a warm start.
- Run the composite pressure solve over all leaf cells and seam ports.

Topology is double-buffered. A candidate generation is invisible to physics
until directory entries, descriptors, fields, worksets, and all receipts agree.
Capacity failure keeps the last valid generation; it never publishes a partial
mesh.

### M4 — Physics-driven resolution policy

Separate support from accuracy.

Support score answers “must this tile exist?” and includes wet cells, interface
band, pressure connectivity, solid/inflow reach, and swept characteristics.

Accuracy score answers “which level may represent it?” and uses maxima, not
averages, of dimensionless indicators:

- interface presence and projected curvature;
- normalized strain, vorticity, and velocity-gradient variation;
- divergence and pressure residual;
- solid/cut/inflow proximity;
- unresolved mass variance under trial restriction;
- protected wavelength/detail history.

Raw speed alone does not request refinement: uniform translation needs a larger
support closure but not smaller cells.

Initial safety floors:

- interface, moving solid/cut cells, and inflow: finest;
- one graded guard tile around those regions;
- deep calm liquid may coarsen;
- no surface coarsening until M3 pressure/transport artifacts pass.

Use per-tile maximum error and temporal hysteresis. One high-curvature corner
must not disappear because 511 quiet cells dilute an average.

### M5 — Camera-driven resolution

Add an optional method-contract view hint containing camera generation,
position, frustum, viewport, and a target projected cell size. The renderer has
this information at the call site that advances the solver; the solver contract
currently does not.

Camera resolution is advisory:

```text
requestedLevel = min(physicsAllowedLevel, cameraRequestedLevel)
```

when level 0 is finest. Physics floors always win. Missing or stale camera data
means physics-only selection. Camera changes use hysteresis and a per-frame
refinement budget so a cut cannot cause a topology storm.

Enable camera LOD in rungs:

1. Deep liquid only; no surface seam.
2. Fully submerged surface-adjacent tiles after seam tests.
3. Far-camera free-surface coarsening after silhouette and reflection gates.

### M6 — Separating solids, rigid coupling, and production presentation

- Extend the composite pressure operator to cut/open volumes and separating
  inequalities.
- Adopt a projected FAS or multilevel active-set solver.
- Couple seam face impulses to rigid bodies with equal-and-opposite receipts.
- Add prescribed inflow/outflow receipts and swept source support.
- Replace the M1 dense diagnostic materializer with sparse adaptive surface
  extraction and a renderer-facing multiresolution surface ABI.

The existing `globalFineLevelSetSource` assumes a uniform fine brick scale and
is not sufficient for far-surface coarse tiles. Do not encode level in an
unrelated flag. Add a generic adaptive scalar/surface publication contract or
publish an adaptive mesh generated from interface worksets.

## 13. Full GPU execution model

### 13.1 Resident state versus stage work

Keep four distinct concepts:

1. Directory entry: whether a logical tile has a stable physical slot.
2. Leaf level: the tile's current cell resolution.
3. Support flags: why state must remain resident.
4. Stage worksets: the exact chunks/ports a pass executes this step.

A resident calm tile need not run sharpening. A dry support tile may run
velocity extension but not pressure. An interface tile participates in all
surface stages. One global “active” bit cannot express this efficiently.

### 13.2 Directory and slots

For a bounded scene, use a direct logical tile-key to slot table. Every lookup
validates key, generation, and level. A missing entry returns a field-specific
virtual value only where that value is physically defined; a missing required
transport receiver is an error.

State slots remain stable while resident. Worklists are sorted by level and
Morton tile key for locality and may change every step without moving state.
All capacities are allocated up front from the method memory plan.

### 13.3 Fast and slow paths

Do not branch on seam type in every cell:

- interior regular microtiles;
- same-level tile-boundary microtiles;
- 2:1 seam cells/ports;
- external/cut boundary records;
- arbitrary-characteristic sampling.

Each gets its own compact workset and pipeline where profitable. Regular
interiors dominate and should look like the uniform shader with buffer
addressing. Seam work is more expensive but proportional to transition area,
not volume.

Immutable advection inputs may use generated one-cell read aprons after
profiling. Iterative pressure vectors must not copy aprons on every iteration;
pressure ports hold direct neighbor indices and read the current vector bank.

### 13.4 Work publication

Every workset uses a small canonical header:

```text
count, capacity, generation, status, dispatchX, dispatchY, dispatchZ
```

The last three words are consumed directly as indirect dispatch arguments.
Mark/rank/scatter validates capacity and uniqueness before setting `READY`.
Physics consumes only `READY` worksets matching the accepted topology
generation.

No step performs a synchronous CPU readback. Diagnostics copy compact receipts
asynchronously after physics publication.

## 14. Performance plan

### 14.1 Measurements

Measure per adapter and scene:

- resident tiles and cells by level;
- work chunks and seam ports by stage;
- useful cells per dispatched lane;
- directory lookups and arbitrary trace samples;
- atomic contention/transport stencil width histogram;
- topology, transport, extension, pressure, and presentation GPU time;
- pressure applications and residual reduction;
- bytes allocated, resident, and touched;
- occupancy fraction and sparse-versus-uniform break-even.

Primary targets should include the repository's Apple M1 Max path and at least
one discrete NVIDIA or AMD WebGPU adapter. Tune pipeline variants from measured
adapter limits; do not hard-code a CUDA warp model.

### 14.2 Required performance behaviors

- Empty scene: zero physics chunks and ports.
- Work scales with resident/support tiles plus transition surface area.
- No full-domain clear, copy, mip build, or occupancy scan.
- No per-cell hash lookup on the regular interior path.
- No pressure solve per disconnected brick; disconnected liquid components may
  be separate global components, but each component's coarse/fine cells remain
  one system.
- High occupancy is reported honestly. If sparse overhead loses to uniform,
  telemetry should make the crossover visible rather than masking it.

### 14.3 Optimization order

1. Remove unnecessary work and copies.
2. Separate regular and seam paths.
3. Improve memory layout and work ordering.
4. Cache uniform directory/neighbor data per workgroup.
5. Fuse passes only where it preserves immutable-snapshot semantics.
6. Add optional subgroup reductions.
7. Consider read-only texture caching after buffer traces identify bandwidth as
   the limiter.

Never trade exact mass receipts or `D = -G^T` for a pass-count win.

## 15. Diagnostics and failure policy

Each accepted step publishes compact receipts:

- topology generation/status/capacity;
- resident and per-stage counts;
- missing/stale/duplicate directory references;
- transport source, deposited, source/sink, and remainder mass;
- sharpening and solid-excess debits/credits;
- pressure initial/final residual and iteration count;
- post-projection volume-weighted divergence norms;
- negative/degenerate free-surface seam weights;
- refinement/coarsening mass and flux closure;
- first failing tile/cell/port ID.

Failures are fail-closed:

- A bad candidate topology leaves the previous generation active.
- A transport receiver miss rejects the step; it does not delete mass.
- Integer overflow rejects the step.
- Pressure non-convergence rejects publication of projected velocity.
- Diagnostic overflow sets an explicit truncated bit; zero is never interpreted
  as “no failures” after overflow.

## 16. Risks and explicit experiments

### Highest risks

1. **Gamma at unequal volumes.** The volume-weighted CM12 extension and seam
   diffusion are new. M1.0 must compare formulations before GPU optimization.
2. **Free-surface pressure at the seam.** The Ando–Batty operator is strong
   prior art, but CM12's density-derived phi and volume correction need our own
   hydrostatic and settling tests.
3. **Shared seam velocity filters detail.** This is the stable first choice;
   compare against a constrained four-subface research variant only if visible
   damping fails the gate.
4. **Conservative arbitrary traces are expensive.** Expanded donor stencils and
   integer deposits may dominate. Measure seam incidence and stencil widths
   before introducing cached transfer edges.
5. **Sparse overhead at high occupancy.** DCGrid's results show the crossover
   can be high. The method is justified by empty space and LOD, not by assuming
   every sparse representation is faster.
6. **Large-CFL support.** A one-tile halo is insufficient. Candidate residency
   must cover swept traces before the step.
7. **Presentation can re-densify the method.** A dense volume publication is
   test-only; production rendering must consume sparse adaptive surface data.

### Approaches rejected up front

- Expanding the uniform active AABB feature into a brick list while retaining
  dense textures.
- Independent per-brick pressure projections.
- Stale pressure halos refreshed once per V-cycle.
- Conserving `sum(rho)` instead of `sum(V rho)`.
- Using negative-weight MLS for mass transport.
- Dropping mass when a characteristic leaves the resident set.
- A global end-of-step mass rescale.
- Averaging refinement scores over a tile.
- Letting camera distance override interface/solid/inflow physics floors.
- Requiring a fixed subgroup/warp size.
- Mutating the uniform method and thereby losing the oracle.

## 17. Definition of done

### Milestone 1

Milestone 1 is done when the complete CM12 step runs for the frozen `8^3 + 4^3`
two-tile domain, all hard algebra/conservation/stability gates pass, the mixed
result beats the all-coarse control against a fine-uniform reference, and no
seam is visible in the prescribed renders or diagnostics.

### New method

The new method is ready for interactive registration when:

- M1 seam gates remain green under the production composite operator;
- M2 proves zero physics work for empty tiles and sparse memory authority;
- M3 dynamically refines/coarsens with exact mass and flux transfer;
- the pressure solve is globally coupled and convergent across all seams;
- M4 physics LOD is stable under hysteresis;
- production presentation consumes sparse/adaptive output;
- memory and GPU timing scale with resident cells rather than the finest domain;
- failure receipts are surfaced in the harness;
- the uniform method remains available as the unchanged reference.

Camera-driven surface LOD and separating rigid coupling can then graduate in
their own gated rungs rather than delaying validation of the fundamental sparse
multiresolution liquid method.

## 18. Immediate next actions

1. Check in this numerical contract and name the working method package.
2. Restore/capture the smallest uniform CPU/GPU oracle vectors needed by M1.0.
3. Implement the two-tile graph and print every one of the 16 seam-port rows.
4. Prove the composite `G^T W G` operator against a dense assembled CPU matrix.
5. Implement and compare the two volume-aware gamma seam exchanges.
6. Freeze the M1 ABI only after those results.
7. Implement GPU state/worksets and projection before transport.
8. Add transport, exact receipts, complete stepping, and regression artifacts.

The critical path is numerical seam correctness. Dynamic topology, camera
policy, and aggressive kernel optimization begin after that evidence exists.
