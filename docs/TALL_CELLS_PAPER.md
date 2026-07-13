# Restricted Tall-Cell Paper: Technical Reference

This is a structured, equation-oriented set of notes for:

> Nuttapong Chentanez and Matthias Müller, "Real-Time Eulerian Water
> Simulation Using a Restricted Tall Cell Grid," 2011.

- Paper: <https://matthias-research.github.io/pages/publications/tallCells.pdf>
- Length: 10 pages
- Scope here: the simulation algorithm, discretization, pressure solve, and
  stability-relevant implementation details
- Reading convention: equation and algorithm numbers below match the paper

This document is a technical paraphrase, not a verbatim transcription. It is
intended to make implementation audits possible without repeatedly navigating
the two-column PDF.

## Reproducible example scenes

The application exposes three paper-derived presets that the current rectangular
container, inflow, and rigid-body systems can represent:

- Figure 3: a continuous bucket/hose jet filling a tank;
- Figure 4: a dam break striking a stack of rigid boxes; and
- Figure 6: an inlet jet flowing past a fixed sphere into a tank.

The Figure 1 flood and Figure 5 lighthouse/beach scenes require uneven terrain
plus particle spray, mist, and foam systems that are not implemented here. They
are therefore not labelled as reproduced scenarios.

The paper reports a `1/30 s` step for its examples. These presets use `1/180 s`
for inlet scenes and `1/360 s` for the rigid-body dam break because this
repository's flux-form VOF transport and bounding-proxy rigid
contacts do not yet share all of the paper implementation's large-step
stabilization. The smaller step keeps the live component-CFL diagnostic below
one during jets and impacts.

## 1. Purpose and central restriction

The method accelerates large 3D Eulerian liquid simulations by representing
each vertical `(x, z)` column with:

1. terrain;
2. exactly one bottom tall cell; and
3. a fixed number `B_y` of ordinary cubic cells above it.

The tall cell covers an integer number of cubic subcells. Its height and the
terrain height are integer multiples of the cubic spacing `Delta x`. The
regular band follows the liquid surface, focusing storage and computation near
the region where 3D motion matters.

The paper's three simplifying restrictions relative to earlier generalized
tall-cell work are:

- one tall cell per water column;
- the tall cell is always the bottom fluid cell; and
- regular-cell quantities are cell-centered, while tall-cell quantities are
  represented only at its bottom and top subcells.

These choices give a constant-size packed representation and constant stencil
size. They also make a GPU implementation practical.

## 2. Governing equations

The simulated liquid is inviscid and incompressible. With velocity
`u = (u, v, w)^T`, pressure `p`, density `rho`, and body force `f`:

```text
du/dt = -(u . grad)u + f/rho - grad(p)/rho                 (1)
div(u) = 0                                                 (2)
```

The liquid domain is the negative region of a signed-distance level set
`phi`. Its evolution is:

```text
dphi/dt = -u . grad(phi)                                   (3)
```

Free-surface Dirichlet conditions and solid Neumann conditions enter the
pressure solve.

## 3. Packed grid and sampling

For horizontal dimensions `(B_x, B_z)` and `B_y` regular layers, every scalar
or vector field is stored as a packed 3D array of size:

```text
(B_x, B_y + 2, B_z)
```

The extra two entries are the bottom and top endpoint samples of the tall
cell. Two 2D arrays store terrain height `H[i,k]` and tall-cell height `h[i,k]`.

For a packed sample `q[i,j,k]`, its uncompressed vertical coordinate is:

```text
y[i,j,k] = H[i,k] + 1                  if j = 1 (tall bottom)
           H[i,k] + h[i,k]             if j = 2 (tall top)
           H[i,k] + h[i,k] + j - 2     if j >= 3 (regular)  (4)
```

The paper uses one-based packed indices in this expression. Implementations
using zero-based indices must translate it consistently.

Sampling at an arbitrary uncompressed point treats the packed structure as a
logical cubic grid:

- below terrain: use the below-terrain boundary value;
- inside the tall cell: linearly interpolate between its two endpoint values;
- inside the regular band: look up the corresponding packed regular sample;
- above the stored band: use the above-air boundary value.

Inside a tall cell spanning from terrain height `H` to `H + h`, the interpolation
is:

```text
q(x,y,z) = ((y-H)/h) q_top + (1-(y-H)/h) q_bottom          (5)
```

Regular-band lookup is a vertical index shift:

```text
q(x,y,z) = q_packed(x, y-H-h+2, z)                         (6)
```

Once this mapping is implemented, the rest of the solver can evaluate fields
as though they live on an ordinary cubic grid. Trilinear interpolation is then
applied in logical uncompressed space.

## 4. Time-step algorithm

The paper's Algorithm 1 performs these stages in order on every time step:

```text
1. Velocity extrapolation
2. Level-set reinitialization
3. Advection and external-force integration
4. Remeshing
5. Incompressibility enforcement
```

This order matters. In particular, the grid hierarchy is rebuilt at the
incompressibility stage and reused for velocity extrapolation on the next
frame, because remeshing happens after extrapolation.

The paper's examples use a relatively large fixed time step of `1/30 s` and
report real-time runs. The advection and remeshing modifications below are
presented specifically as safeguards for large time steps. The paper does not
state that arbitrary time steps are stable and does not replace empirical CFL
monitoring.

## 5. Velocity extrapolation

Before advection, velocity is extrapolated from liquid into air. For the
`x` component:

```text
du/dtau = -(grad(phi)/|grad(phi)|) . grad(u)                (7)
```

Equivalent equations apply to the other components.

Because water can cross many cells in one `1/30 s` step, the paper does not
limit extrapolation to a tiny surface band. It extrapolates across the full
domain, using a hierarchy of collocated tall-cell grids.

### 5.1 Hierarchy construction

The number of levels is:

```text
L = log2(min(B_x, B_y, B_z))
```

At level `l`, spacing doubles relative to the next finer level. Terrain height
is downsampled with a minimum over the corresponding fine columns. The coarse
tall height is based on the maximum fine water-column top, averaged into coarse
units and offset by the coarse terrain height:

```text
H^l[i,k] = floor(min H^(l+1) over the 2x2 horizontal block / 2)   (8)

h^l[i,k] = ceil(max(H^(l+1)+h^(l+1)) over the block / 2)
           - H^l[i,k]                                             (9)
```

The exact typesetting uses the four fine horizontal columns under the coarse
column.

### 5.2 Extrapolation procedure

1. Sweep fine-to-coarse and mark a coarse velocity known if at least one
   corresponding finer velocity is known.
2. Trilinearly interpolate known values from the previous level, renormalizing
   weights to ignore unknown values.
3. Traverse coarse-to-fine and fill remaining unknown fine values from known
   coarse values.

All components use the same collocated layout and can be processed together.
After the hierarchy pass, every finest-grid sample has a velocity.

## 6. Level-set reinitialization

Advection destroys the signed-distance property of `phi`, so the paper
periodically reinitializes it. Its implementation uses a high-resolution
surface-tracking grid and performs these stabilizations:

- reinitialize only two or three cells away from the surface;
- while reinitializing, do not change `phi` on grid points adjacent to the
  interface;
- clamp interface motion during one reinitialization to at most one grid
  spacing; and
- clamp every `|phi|` to at most `5 Delta x`.

The last clamp supports a narrow signed-distance representation. The authors
report no significant visual artifact from these safeguards.

## 7. Advection and external forces

Velocity uses the bounded modified MacCormack method of Selle et al. If the
corrected value leaves the range of interpolation-source values, the method
falls back to first-order semi-Lagrangian advection.

The level set uses semi-Lagrangian advection rather than MacCormack because the
authors observed noisy surfaces near the interface with MacCormack level-set
transport.

All three velocity components are collocated. They are traced together and
reuse interpolation weights. External forces are integrated after advection.

## 8. Remeshing constraints

After advection, cells with `phi <= 0` are liquid. The tall height is recomputed
for every column. The new grid must satisfy three constraints:

1. At least `G_L` regular cells exist below the lowest liquid surface, so
   three-dimensional liquid motion stays in the cubic band.
2. At least `G_A` regular cells exist above the highest liquid surface, so a
   surface moving upward remains in the cubic band for subsequent steps.
3. Adjacent tall-cell heights differ by no more than `D` cubic units, reducing
   volume-gain artifacts caused by the collocated pressure/divergence scheme.

The algorithm first finds per-column minimum and maximum surface coordinates
that satisfy constraints 1 and 2. It initializes a temporary split location
from their average, then smooths across the horizontal grid. Preference is
given to the air-halo constraint when both cannot be met. Finally, one or two
Jacobi-style passes enforce the neighbor-height constraint:

```text
y_tmp[i,k] = min(y_tmp[i,k],
                 max over horizontal neighbors(y_tmp[neighbor] - D))  (10)
```

The paper's examples use:

```text
8 <= G_L <= 32
G_A = 8
3 <= D <= 6
```

and one or two Jacobi passes.

After choosing new tall heights:

- regular samples copy or trilinearly interpolate old values at their new
  world positions; and
- tall-cell endpoints are obtained by a least-squares fit through the old
  values covered by the new tall cell.

The paper warns that constraints can conflict and are not guaranteed to all be
satisfied. This makes representability and constraint-violation diagnostics
important in any implementation.

## 9. Pressure projection

Let `u*` be velocity after advection, forces, and remeshing. Projection seeks:

```text
div(u* - (Delta t/rho) grad(p)) = 0                       (11)
```

For constant density:

```text
laplacian(p) = (rho/Delta t) div(u*)                      (12)
```

### 9.1 Collocated divergence

At a stored sample, divergence is the sum of directional differences:

```text
div(u) = du/dx + dv/dy + dw/dz                            (13)
```

The positive `x` face velocity is the average of the current and positive
neighboring cell-centered velocities, except that a solid neighbor supplies
the solid velocity:

```text
u_x+ = (u_current + u_neighbor)/2   if neighbor is not solid
       u_solid                       otherwise             (14)
```

Negative faces and the other axes are analogous.

An important consequence is that divergence inside a tall cell is not sampled;
only its bottom and top endpoint samples constrain it. The paper explicitly
identifies this as a source of gradual volume gain where adjacent cubic cells
flow into a tall cell. The `D` remeshing bound mitigates, but does not eliminate,
this error.

### 9.2 Laplacian, free surface, and solids

The Laplacian is a conventional second-difference sum:

```text
laplacian(p) = d2p/dx2 + d2p/dy2 + d2p/dz2                (15)
```

For a positive-face pressure sample `p_x+`:

```text
p_x+ = p_neighbor                                         liquid neighbor
       s p_current + (1-s) p_neighbor                     mixed solid cell
       ghost-fluid pressure                               air neighbor       (16)
```

Here `s` is the solid volume fraction. The ghost-fluid construction places
zero pressure at the reconstructed liquid surface rather than at the center of
the air cell. This improves free-surface accuracy. Solid-fraction blending
handles partially occupied cells and is also required on coarse multigrid
levels.

The pressure gradient is:

```text
grad(p) = ((p_x+ - p_x-)/Delta x,
           (p_y+ - p_y-)/Delta x,
           (p_z+ - p_z-)/Delta x)^T                       (17)
```

and velocity is corrected by:

```text
u = u* - (Delta t/rho) grad(p)                            (18)
```

The discrete Laplacian is deliberately not the composition of the paper's
collocated divergence and gradient. Consequently, projection is not
idempotent and may not eliminate divergence completely.

The tall-cell system is nonsymmetric, so ordinary conjugate gradients are not
applicable. Its constant-size stencil is nevertheless suitable for a parallel
multigrid solver.

## 10. Multigrid pressure solver

### 10.1 Hierarchy and coarse fields

The pressure hierarchy shares the tall-cell layout used by velocity
extrapolation. Coarse right-hand side values are ordinarily 8-to-1 averages.
Tall endpoint values use a least-squares fit to the corresponding fine values.

When downsampling `phi`:

- use the ordinary 8-to-1 average when all eight values have the same sign, or
  on levels coarser than the finest `C` levels;
- otherwise average only positive `phi` values, preserving bubbles on the
  finest levels.

The paper uses `C = 2`.

All levels recompute coefficients with the ghost-fluid and solid-fraction
rules. Subgrid boundary locations are therefore represented at every level;
the authors describe this as crucial to robust convergence.

### 10.2 Operator on a packed tall cell

For restriction, the logical trilinear representation is special inside a
tall cell: interpolation is zero everywhere except its bottom and top
subcells. This prevents an endpoint residual from being smeared through the
unrepresented tall interior:

```text
r(x,y,z) = r_bottom    at tall bottom
           r_top       at tall top
           r_regular   in represented regular cells
           0           elsewhere                       (19)
```

Prolongation uses trilinear interpolation. Out-of-domain values are ignored and
the remaining weights are renormalized; if every sample is out of bounds,
pressure is zero.

### 10.3 Cycles and smoothing

Algorithm 2 builds matrices coarse-to-fine, computes the finest RHS, clears
pressure, performs one or more full cycles, then optional V-cycles.

A V-cycle (Algorithm 3) performs:

```text
pre-smooth
compute residual
restrict residual
solve/cycle on the next coarser level
prolongate and add correction
post-smooth
```

A full cycle (Algorithm 4) restricts the finest RHS down to the coarsest level,
solves there, then prolongates upward with a V-cycle at each level. This is
followed by any configured top-level V-cycles.

The smoother is parallel red-black Gauss-Seidel. The paper emphasizes three
requirements for reliable convergence:

1. use full cycles;
2. preserve air bubbles on the finest levels; and
3. apply ghost-fluid and solid-fraction boundary treatment on all levels.

Omitting any one caused stagnation or divergence in the authors' tests.

## 11. GPU-specific optimizations

The paper describes these implementation optimizations:

- perform vertical interpolation first so every trilinear lookup fetches only
  two consecutive packed samples per horizontal corner;
- in the smoother, obtain pressure below a tall top endpoint through the
  Laplacian/interpolation mapping rather than an extra conditional fetch;
- stop coarsening when the top grid fits in shared memory, then solve it with
  multiple Gauss-Seidel iterations in one kernel; and
- build the hierarchy once during the incompressibility stage and reuse it for
  velocity extrapolation on the next frame.

## 12. Extensions outside the paper core

The paper also sketches:

- rigid-body coupling through a volume-of-solid representation, alternating
  fluid and body solves, with cells above `s > 0.9` treated as solid in
  divergence;
- a separately diffused level set inside solids to repair semi-Lagrangian
  traces through rigid bodies;
- sampled buoyancy and drag forces on rigid bodies;
- particle thickening for thin sheets and volume retention; and
- spray, mist, and foam particles for rendering detail.

These extensions are not required to reproduce a no-rigid-body tank-fill
instability and should be removed from a minimal reproduction.

## 13. Results and explicit limitations

The examples run with a `1/30 s` time step. Reported simulation grids range
from `64 x (64+2) x 64` to `128 x (32+2) x 128`, with separately higher
surface-tracking grids. The authors use a single precision GPU implementation.

For the flood convergence plot, the residual decreases rapidly and then
plateaus due to single precision. The multigrid iteration count is described
as nearly resolution-independent.

The paper explicitly calls out two limitations relevant to stability audits:

- because divergence is measured only at tall endpoints, not inside a tall
  cell, adjacent cubic-cell inflow can create gradual volume gain; and
- because the Laplacian is not `div(grad)`, projection may leave divergence and
  is not idempotent.

The proposed mitigation for the first issue is the adjacent-height bound `D`.
Neither limitation implies that sudden kinetic-energy or velocity blow-up is
expected. Such an eruption should therefore be treated as an implementation or
time-integration defect until measurements show otherwise.

## 14. Audit checklist for this repository

For each simulation step, a comparison to the paper should answer:

- Was velocity extrapolated far enough into air, and were unknown samples
  explicitly distinguishable from valid zero velocity?
- Was `phi` (or the VOF-derived pressure proxy) reinitialized without moving
  the interface by more than one cell and clamped to `5 dx`?
- Did velocity use bounded MacCormack with a first-order fallback?
- Were forces applied only to physically relevant liquid/face samples?
- Was remeshing executed this step, and were `G_L`, `G_A`, and `D` satisfied?
- Was remapping conservative per column, and did any clamp discard residual
  volume?
- What were maximum divergence before and after projection, and where did each
  occur?
- What were maximum pressure-system residual and relative residual?
- What were liquid and extrapolated-air maximum speeds?
- What was the maximum component CFL, `max(|u_i| Delta t / Delta x_i)`?
- Did any velocity, pressure, VOF, base height, or residual become non-finite?
- Did the first anomalous signal occur during extrapolation, advection/forces,
  remeshing, pressure solution, or projection?

The diagnostics should preserve extrema locations in both packed and
uncompressed coordinates. A scalar maximum without a location is insufficient
for distinguishing an interface, tall endpoint, wall, remesh boundary, or
multigrid artifact.

## 15. Current implementation mapping

The primary files to audit are:

- `lib/tall-cell-grid.ts`: packed layout and initial bases;
- `lib/tall-cell-kernels.ts`: extrapolation, advection, forces, remeshing,
  divergence, and projection;
- `lib/tall-cell-multigrid.ts`: `phi` reconstruction, hierarchy, boundary
  coefficients, smoothing, restriction, prolongation, and cycles;
- `lib/webgpu-eulerian.ts`: stage ordering, time-step selection, dispatches,
  and readback diagnostics; and
- `docs/TALL_CELL_WEBGPU.md`: declared implementation departures.

The implementation-comparison table and measured instability findings belong
in `docs/TALL_CELL_STABILITY.md` so this paper reference can remain stable.
