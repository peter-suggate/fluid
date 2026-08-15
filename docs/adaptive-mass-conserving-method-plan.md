# Adaptive Mass-Conserving GPU Liquids

Implementation research and plan, 2026-08-14

Status: Sparse CM12 is interactive and the fixed-topology composite method is
GPU-resident. Construction still packs the initial compact topology on the
host, but accepted transport, surface conditioning, pressure projection,
diagnostics, and presentation remain on the device. Dynamic topology is the
next production target. CPU implementations are test oracles only and may not
become frame authority. The existing uniform method remains the behavioral
oracle and is not the implementation substrate.

Current implementation checkpoint (2026-08-15):

- CM12 constants and pure formulas live in `lib/core/cm12-numerics.ts` and are
  consumed by the production uniform shaders and the adaptive CPU oracles.
- One row builder now covers arbitrary strongly graded sparse
  `1^3`/`2^3`/`4^3`/`8^3` atlases as well as
  the original `8+8`, `4+4`, `8+4`, and reflected `4+8` two-tile controls. The
  mixed seam has one authoritative five-cell port per coarse face patch and
  derives divergence as the weighted negative transpose of its gradient.
- Focused executable receipts cover algebraic A/B symmetry, a matrix-free
  pressure projection, volume-weighted conservative transport with persistent
  gamma, and WGSL pressure-operator parity. `npm run
  acceptance:adaptive-mass:m1` runs this checkpoint without adding a unit-test
  matrix. Forced-all-fine Sparse CM12 is now the primary behavioral reference;
  Uniform remains a diagnostic cross-check, not the adaptive physics target.
- Free-surface pressure (using one SPD ghost fraction per composite row),
  composite gamma diffusion/sharpening, GPU transport-operator parity, and a
  symmetric density-slice A/B are now executable companion receipts.
- The interactive solver omits empty bricks, activates face-local transport
  receivers, conservatively transports density/gamma/momentum, conditions the
  CM12 surface, and projects one globally coupled composite pressure system.
  The short mixed-resolution symmetric-expansion receipt is stable and the
  matched mini-dam performance lane is inside the `1.20x` median target; the
  canonical two-second blockers are recorded below.
- Logical residency is now adaptive for the long-tank rung: construction packs
  bounded dormant receiver slots, but a GPU activity transaction alone decides
  when each slot becomes active. Empty slots outside the exact 26-neighbor air
  support ring now retire on the GPU without deleting any positive-density
  receipt. Resolution inside an allocated slot remains fixed; conservative
  level mutation, generic rolling allocation/freeing, solid coupling, and
  compact sparse surface publication remain open.
- The resident GPU graph now owns CM12 transport, gamma diffusion/sharpening,
  one composite Jacobi-PCG projection, and dense diagnostic publication. A
  resident activity/history arena measures every active compact brick after
  projection, advances its own four-step epoch clock, and publishes score,
  reason, hot/quiet, and D4 receipts. Its logical active bit now gates packed
  owner lookup and physics work; packed topology and accepted CM12 fields remain
  immutable. This is the first GPU-published residency transaction, not yet a
  resolution-transfer transaction.
- From this checkpoint onward, every production adaptation increment is fully
  GPU-resident. No per-frame CPU field readback, classifier, planner, transfer,
  dispatch decision, topology rebuild, or fallback may participate in an
  accepted step. Readback remains permitted only for explicit diagnostics and
  acceptance tests.
- The no-change pass is proven non-perturbing through a dispatch-on/off Dawn
  A/B: through 30 mixed-resolution steps, mass, velocity D4 error, pressure D4
  error, and maximum divergence are identical. The canonical 250-step baseline
  is not yet green independently of this pass: mixed topology begins losing
  velocity/pressure D4 symmetry at step 22, and forced-all-fine later reaches a
  `3.05e-5 s^-1` float32 divergence spike. These remain guardrails and are not
  hidden by symmetry projection or relaxed thresholds while sparse residency
  work proceeds.
- The read-only GPU rung turns accepted activity/history into an epoch-gated
  requested `1^3`/`2^3`/`4^3`/`8^3` resolution for every resident brick.
  Surface and predicted receivers request `8^3` immediately; changes otherwise
  move one rung per epoch, and three ordered refine-only closure passes enforce
  a maximum 2:1 face ratio. It still cannot mutate accepted topology until
  conservative candidate transfer and rollback exist.
- Accepted and candidate levels now have separate device records. The closed
  plan is validated for legal one-rung changes and 2:1 adjacency without
  mutating accepted fields. A max-`8^3` isolated candidate arena performs exact
  overlap transfer for density and gamma, mass-weighted cell momentum, and a
  pressure warm start; six candidate exterior faces area-average the
  authoritative accepted normal flux. Device receipts gate mass, gamma, XYZ
  momentum, and all exterior-flux integrals. This is still deliberately
  non-authoritative: row patching, global reprojection, coupled validation, and
  atomic publication remain required before accepted levels can change.
- The canonical end-to-end sparse scene is `sparse-cm12-long-dam-break`: a
  `96x24x16` tank whose full-width reservoir occupies the first two of twelve
  brick columns. The front must traverse the ten initially dry columns and
  reach the opposite wall without a residency gap, a 2:1 discontinuity, mass
  loss, or an unexplained departure from forced-all-fine Sparse CM12.
  The original long Dawn receipt was red: at `1.004 s`, Uniform reached fine
  cell `x=95`, while Sparse CM12 and its resident allocation stopped at `x=23`.
  The receiver transaction is green in the current Adaptive/All-fine A/B. At
  `0.324 s` both fronts are at `x=53`, Adaptive retains 38 of 72 bounded slots
  with a ten-cell receiver lead, and its mass drift is `2.74e-6`; at `1.0 s`
  its liquid front reaches `x=95` only `0.004 s` after All-fine. Dawn reports
  no validation error. Dormant cells stay at `rho=0`, `gamma=1`, and zero velocity;
  transport, diffusion, sharpening, pressure, and presentation exclude them
  until the GPU atomically publishes their active bit. Occupied tiles and their
  one-tile Moore air ring remain active; unsupported empty tiles atomically
  retire after activity measurement. There is no global mass
  rescale, front injection, CPU frame decision, or D4-specific correction in
  this residency change. The harness now hard-fails both a pinned active
  boundary and failure to match All-fine's far-wall arrival.

Working method id: `adaptive-mass`
User-facing method name: `Sparse CM12`

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

### M2 — GPU sparse single-resolution residency

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

### M3 — Dynamic two-level resolution controller

The detailed first-rung specification is colocated with the method at
[`surface-activity-resolution-policy.md`](../lib/methods/adaptive-mass/surface-activity-resolution-policy.md).
This section is the architectural contract and takes precedence if an older
prototype or benchmark policy disagrees with it.

#### M3.1 Keep residency, accuracy floors, and activity separate

Three independent questions produce the next topology:

1. **Residency:** must this logical brick exist for wet mass, pressure
   connectivity, velocity extension, a swept characteristic, or a prescribed
   source? If not, it is absent and receives zero physics or classification
   work.
2. **Accuracy floor:** what is the coarsest representation that is safe even if
   the flow is currently calm? Free surface, unresolved fine detail,
   solid/inflow proximity, and a predicted interface arrival can impose a hard
   `8^3` floor.
3. **Activity:** does accepted state show enough deformation or temporal change
   to promote, retain fine resolution, or begin cooling toward `4^3`?

Do not fold these into one average. A nearly stationary thin sheet is calm but
not safely coarse. A uniformly translating saturated block is moving but does
not need fine interior cells. A dry swept receiver must exist even before it
contains mass, but it need not enter the pressure workset.

The first production rung has only `4^3` and `8^3` leaves. More levels wait
until repeated two-level promote/demote cycles pass every conservation,
symmetry, stability, and frame-time gate.

#### M3.2 First-rung state machine

Every resident brick has compact persistent policy state:

| State | Meaning | Allowed next states |
|---|---|---|
| `CoarseCalm` | `4^3`, below all floors and thresholds | `CoarseWatch`, `PromotePending` |
| `CoarseWatch` | `4^3`, activity rising but below a hard floor | `CoarseCalm`, `PromotePending` |
| `FineActive` | `8^3`, protected by a floor or recent activity | `FineCooling` |
| `FineCooling` | `8^3`, quiet but still inside demotion hysteresis | `FineActive`, `DemotePending` |
| `PromotePending` | candidate `4^3 -> 8^3` transaction | `FineActive` or rollback |
| `DemotePending` | candidate `8^3 -> 4^3` transaction | `CoarseCalm` or rollback |

Absence is directory state, not a resolution state. Transitions are decided
from one immutable accepted snapshot; one brick's decision cannot mutate the
inputs used by another brick in the same epoch.

Run the cheap measurement/history pass after every accepted step. Run topology
planning every four accepted steps, or earlier for a hard prediction/emergency.
Initial thresholds are:

```text
promote: score >= 160 / 255 for 2 topology epochs
emergency promote: score >= 224 / 255 immediately
demote: score <= 96 / 255 for 8 topology epochs
demotion veto: fine-to-coarse restriction error > 0.08
```

Promotion wins every conflict. Demotion is at most one level per epoch. A
topology or history discontinuity resets quiet history rather than treating
missing evidence as calm. No brick may reverse its last transition before its
cooldown expires unless a hard safety floor requests promotion.

#### M3.3 Conservative surface-first rollout

The rollout intentionally starts more conservative than the eventual policy:

1. **Rung A — fine interface, adaptive bulk.** Every brick containing
   `0.05 < rho < 0.95` or a composite row crossing `rho = 0.5` is `8^3`.
   Bricks reachable by the interface before the next topology epoch are also
   `8^3`. Deep saturated liquid may demote after hysteresis and a restriction
   veto. This replaces component-size bootstrap coarsening first.
2. **Rung B — activity-sized surface.** A planar, slowly changing surface may
   become `4^3` only after a composite restriction estimator predicts bounded
   density, normal, flux, and silhouette error. Curved, thin, breaking, or
   rapidly moving surface remains fine. This rung cannot relax Rung A gates.
3. **Rung C — additional levels.** The operator, scalar transfer oracle,
   activity estimator, and GPU candidate planner now accept the complete
   `1^3 / 2^3 / 4^3 / 8^3` ladder. Candidate changes move one rung per epoch
   and close face-neighbor grading by refinement. Accepted runtime level
   changes still wait for GPU candidate transfer, row patch, reprojection,
   validation, and atomic publication. Never jump an accepted field across a
   level.

New face-local receivers start fine in Rung A. This is deliberate: the long
tank showed that a shallow front entering a pre-published `1^3` receiver was
volume-averaged away at that first tile. Receiver activation therefore writes
an `8^3` GPU candidate request before publication and grading closes outward
through `4^3`, `2^3`, and `1^3`. Omitted air has no physical level; the current
bounded all-fine dormant backing is temporary scaffolding for the residency
transaction and must be replaced by free-slot allocation. Rung B may create a
coarse receiver only when the swept-interface predictor proves no surface can
enter it before the following epoch.

A method-level seam sentinel may pin one fine brick, or a complete D4 orbit, for
diagnostic A/B scenes. It is explicit in policy telemetry and is off for
production and physics comparisons. The present `fineHalf` and component-size
bootstrap are deleted once the sentinel and real classifier land.

### M4 — Physics-driven measurement, planning, and transfer

#### M4.1 Compact activity measurement

Classification reads accepted compact leaves and composite rows. It never
materializes or scans the finest presentation volume. Per-brick reductions use
maxima and reason bits, not averages:

- **surface shape:** density-normal variation and projected curvature;
- **deformation:** normalized strain, vorticity, and velocity-gradient
  variation;
- **temporal change:** overlap-matched density and normal change since the last
  accepted observation;
- **transport prediction:** characteristic reach before the next topology
  epoch and local CFL;
- **solver trouble:** post-projection divergence, pressure residual, or an
  iteration spike localized to the brick/incident ports;
- **fine-detail veto:** error produced by trial volume restriction followed by
  the exact current prolongation operator;
- **hard reasons:** interface, solid/cut/inflow, protected detail, unknown
  history, and diagnostic seam sentinel.

Raw speed is not a refinement signal. It affects swept support and arrival
prediction, while strain/variation determine accuracy. All channels are
dimensionless, axis symmetric, byte quantized with round-to-nearest-even, and
published with a policy version and threshold receipt.

#### M4.2 Budgeted, symmetry-preserving planning

A `4^3 -> 8^3` promotion adds 448 leaves. Plan in leaf deltas rather than brick
counts and cap ordinary promotion per epoch to:

```text
clamp(max(448, floor(0.10 * acceptedLeafCount)), 448, 8 * 448)
```

Consume complete equal-score/equal-reason buckets. If a bucket does not fit,
defer the whole bucket; do not split an exactly symmetric D4 orbit by brick key.
Emergency and predicted-surface requests may exceed the soft frame budget,
because knowingly advecting an interface into an under-resolved receiver is a
correctness failure. A hard slot-capacity miss rejects the step and keeps the
accepted generation.

Before transfer, close the plan under face-neighbor grading. With only `4^3`
and `8^3` this is validation; with later rungs it becomes a fixpoint promotion
pass. Omitted air is not assigned a level merely to satisfy grading.

#### M4.3 Exact transition transaction

Build a candidate generation beside the accepted state:

- `8^3 -> 4^3`: volume-average density and gamma; sum liquid momentum then
  divide by mass; area-average authoritative normal face flux; volume-average
  pressure only as a warm start.
- `4^3 -> 8^3`: first use conservative constant injection for density, gamma,
  and momentum; inject parent flux to child subfaces; initialize internal faces
  from the injected field. Limited-linear prolongation is a later accuracy
  improvement, never a prerequisite for conservation.
- Transfer history by exact overlap. Children inherit hot/protected evidence
  and zero quiet age; a parent receives the maximum child score, union of
  reasons, minimum quiet age, and zero hot age.
- Rebuild only affected brick/seam descriptors, remap the pressure warm start,
  and run the same global composite projection. Pressure itself is not a
  conserved quantity.

Validate mass, gamma integral, XYZ momentum, exterior flux, finite bounds,
grading, unique ownership, pressure convergence, and post-projection
divergence. Conservation is checked on the transferred candidate before
projection. The subsequent pressure impulse is reported separately so a
legitimate boundary/free-surface impulse is not mislabeled as transfer error.
Also record the kinetic-energy change caused solely by transfer: coarsening may
remove unresolved energy, but it may not create energy beyond roundoff and its
loss must be bounded by the measured restriction error. The zero-time
projection then has its own energy/impulse identity.

Only after every receipt passes does one generation word atomically publish
directory entries, levels, fields, worksets, policy history, and diagnostics.
Any failure leaves the previous accepted state fully usable. There is no
partial resolution change and no global mass rescale.

#### M4.4 Adaptive acceptance matrix

Every case has forced-all-fine Sparse CM12, forced-all-coarse where meaningful,
adaptive Sparse CM12, and Uniform CM12 controls at the same finest resolution
and exact time step:

1. **Settling pool:** surface stays fine, saturated bulk demotes, transition
   count reaches zero, and hydrostatic pressure remains quiet for two seconds.
2. **Translating slab:** predicted receivers promote before the interface
   arrives; bricks behind the slab cool and demote without a mass or momentum
   step.
3. **Wake/sleep pulse:** one compact impulse promotes the affected region; it
   remains fine through cooldown and returns to coarse exactly once.
4. **Slosh:** repeatedly changing normals do not chatter levels or accumulate a
   pressure/energy pulse at mixed seams.
5. **Thin sheet and droplet:** restriction-error veto prevents irreversible
   loss even when instantaneous velocity is small.
6. **Dam break and impact:** rapid activation may use emergency budget but must
   keep mass, pressure convergence, and frame timing explicit.
7. **D4 symmetric expansion, at least two seconds:** resolution, activity,
   reasons, and history are D4 equivariant at every epoch; density, velocity,
   pressure, divergence, topology, and connectivity retain their existing
   long-run gates.
8. **Empty/disconnected:** empty has zero policy work; separated bodies do not
   force resolution or residency in the space between them.

Hard transition gates are exact CPU/double closure and `<= 5e-7` GPU/float32
relative error for pre-projection density, gamma, XYZ momentum, and exterior
flux; no nonfinite or invalid density/gamma value; no topology-only
kinetic-energy increase above roundoff; a closed projection impulse/energy
identity; unchanged pressure residual/divergence tolerances; and no reversed
transition inside cooldown. A transition pressure solve may not exceed the
greater of eight extra iterations or `1.5x` the preceding steady-topology
iteration count without publishing a performance failure.

Adaptation must also be useful: after the settling interval, at least 60% of
eligible saturated bulk bricks are coarse, while every Rung A interface and
predicted receiver is fine. A controller that passes by retaining everything
at `8^3` fails.

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

### 13.5 Resolution epochs on the GPU

The production controller is a method-colocated frame-pipeline plugin, not a
CPU policy loop hidden inside `advanceTo`:

```text
accepted compact state
  -> per-brick measure + history update                 every accepted step
  -> mark requests + score buckets                      topology epoch only
  -> scan/scatter budgeted requests + grading closure   topology epoch only
  -> allocate candidate slots + conservative transfer   changed bricks only
  -> patch affected regular/seam blocks and worksets     changed neighborhood
  -> global composite projection                         candidate topology
  -> validate receipts -> atomic publish or rollback
```

Use stable brick slots and per-brick row blocks so a one-brick transition does
not renumber every leaf or rebuild the complete pressure graph. Unchanged
regular interiors and seam blocks retain their descriptors; only the changed
brick and its face neighbors are regenerated. PCG vectors use stable leaf slots
plus compact active worklists, and the pressure warm start is overlap-remapped
without a host map.

The current CPU oracle may rebuild a complete candidate graph while proving the
transaction. That is not acceptable as the final GPU execution path. It must
remain visibly separated in timing telemetry so an oracle convenience cannot
silently become production topology cost.

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
- occupancy fraction and sparse-versus-uniform break-even;
- activity score/reason histograms and bricks in each lifecycle state;
- requested, budgeted, deferred, promoted, demoted, and grading-closure counts;
- classifier/history, planning, transfer, row-patch, and post-transition
  projection time;
- pressure iterations and warm-start residual before/after every resolution
  epoch;
- topology-induced mass, momentum, flux, and kinetic-energy deltas.

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
- A no-change activity epoch performs no allocation, field transfer, or
  pressure-topology rebuild.
- Transition cost scales with changed bricks plus their seam neighborhood, not
  all resident leaves.
- Resolution budgets bound ordinary frame spikes. Predicted activation should
  make emergency budget overruns rare and independently visible.

### 14.3 Frame-time contract

At a given finest authored resolution and time step, compare complete captured
physics frames against Uniform CM12 after construction and warm-up:

- target median Sparse CM12 frame `<= 1.20x` uniform;
- hard median cap `<= 1.30x` uniform;
- p95 Sparse CM12 frame `<= 1.30x` uniform;
- classifier plus history `<= 5%` of Sparse frame and `<= 0.25 ms` median on
  the reference adapter;
- no-change topology epoch `<= 2%` of frame and performs zero allocation;
- candidate planning/transfer/patch work `<= 10%` amortized over the default
  four-step cadence;
- after a settling scene becomes calm, transition count must converge to zero.

The hard comparison includes pressure-iteration effects caused by the adaptive
topology; reporting a fast classifier beside a much slower solve is a failure.
Publish p50/p95 stage timings on the pipeline graph and keep a forced-all-fine
Sparse arm to distinguish sparse bookkeeping overhead from useful coarsening.

### 14.4 Optimization order

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
- fine/coarse counts, lifecycle-state counts, and exact mixed-level adjacency;
- activity score/reason histogram, history age, hard-floor count, and policy
  version/thresholds;
- requested/deferred/promoted/demoted/sentinel/balance-closure keys or compact
  counts, plus ordinary/emergency leaf-budget use;
- missing/stale/duplicate directory references;
- transport source, deposited, source/sink, and remainder mass;
- sharpening and solid-excess debits/credits;
- pressure initial/final residual and iteration count;
- post-projection volume-weighted divergence norms;
- negative/degenerate free-surface seam weights;
- refinement/coarsening mass and flux closure;
- refinement/coarsening momentum and kinetic-energy deltas;
- pressure warm-start residual and iteration delta across a topology epoch;
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
8. **Resolution chatter.** Threshold-only selection can rebuild topology every
   few frames and inject transfer/projection noise. Use asymmetric hysteresis,
   cooldown, immutable decisions, and a no-ping-pong acceptance lane.
9. **Late promotion loses detail permanently.** Constant prolongation cannot
   recreate a sheet or curvature already averaged into `4^3`. Predict interface
   arrival, preserve unknown/history evidence, and veto demotion with trial
   restriction error.
10. **Topology changes destabilize pressure.** Changed conditioning or a poor
    warm start can make an otherwise cheap adaptation dominate the frame or
    produce visible boiling. Reproject every candidate globally, gate pressure
    iterations/residual/divergence, and retain the previous generation on
    failure.
11. **Averaging velocity creates energy.** Transfer authoritative normal flux
    and liquid momentum, not presentation/collocated velocity alone. Record the
    topology-only energy delta and reject artificial energy gain.

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

### Production GPU method

The interactive CPU authority can be replaced by the production GPU authority
when:

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

### Dynamic-resolution rung

The first adaptive calm/active rung is done only when:

- component-size bootstrap coarsening and implicit `fineHalf` policy are absent
  from production selection;
- the same accepted snapshot produces byte-identical score, reason, history,
  and resolution maps under x/z D4 transformations;
- interface/predicted receivers promote before mass arrives, while deep liquid
  demonstrably demotes after settling;
- every transition closes density, gamma, momentum, and exterior flux and is
  followed by a converged global projection;
- a two-second symmetric expansion, translating slab, slosh, droplet impact,
  and settling pool show no boiling, seam crease, pressure spike, or topology
  ping-pong;
- the adaptive result remains between the all-coarse and finest-uniform
  controls for interface shape, centroid/spread, and kinetic energy, and moves
  closer to finest-uniform as the fine region grows;
- mini dam break and symmetric expansion satisfy the complete p50/p95 frame
  contract, including topology epochs and pressure-iteration spikes;
- a settled pool reaches zero transitions and a materially coarse bulk, rather
  than passing stability by remaining all fine.

## 18. Immediate next actions

1. Keep the now-green long-tank dam break as the canonical residency gate. Add
   stored intermediate receiver-lead and active-column receipts (the final
   front, far-wall, mass, and active-boundary receipts are already hard gates),
   then compare the same checkpoints with forced-all-fine Sparse CM12.
2. Replace the bounded all-fine dormant backing with a device free-slot pool.
   Allocate `8^3` only in the swept-interface band, retain the exact one-tile
   air-support ring, return unsupported empty slots to the pool, and add compact
   allocation-failure/capacity receipts plus face-local row patching. Preserve
   the current no-readback frame dependency and 2:1 closure.
3. Keep the Rung A CPU policy and transfer as an offline oracle only. Replace
   construction-time component-size selection through GPU-authored measurement,
   planning, conservative candidate transfer, global reprojection, validation,
   and rollback; do not route accepted frames through the CPU oracle.
4. Add translating-interface, settling-pool, slosh, and repeated wake/sleep CPU
   receipts. Capture resolution/history maps at every epoch, not just final
   fluid fields.
5. Extend the now-live isolated GPU cell/flux transfer with candidate row
   patches and candidate owner/directory records, then run the global candidate
   projection so topology costs and pressure iteration spikes are visible.
6. Keep resolution requests read-only until score/reason/history/active maps are
   exactly D4 equivariant and adaptive remains inside the forced-all-fine Sparse
   CM12 comparison gates at every tested cadence. Logical residency may change only
   through the accepted GPU active-bit transaction; `4^3 <-> 8^3` field transfer
   still requires the full conservation and rollback gate.
7. Implement GPU candidate allocation and exact `8^3 <-> 4^3` transfer, patch
   only changed descriptor neighborhoods, reproject, validate, and atomically
   publish.
8. Remove `fineHalf` from production policy; retain only an explicit off-by-
   default seam sentinel for diagnostics.
9. Run two-second symmetry, long-tank dam, and mini-dam matched A/B lanes with
   Adaptive and forced-all-fine Sparse CM12. Tune thresholds only from stored
   receipts; use Uniform only for explicitly diagnostic cross-checks.
10. Only after Rung A is stable, useful, and inside the frame contract, research
   Rung B calm-planar-surface coarsening and then camera advice.

The critical path is now stable, incremental resolution change—not additional
seam algebra. Correctness requires conservative transfer and global projection;
performance requires compact classification and local topology patching.
