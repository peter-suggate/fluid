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
                 max over horizontal neighbors(y_tmp[neighbor]) + D)  (10)
```

(An earlier revision of this document printed `- D` inside the max; the
paper's form is `+ D` outside it — see Appendix A.5.)

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

---

## Appendix A. Verbatim transcription from the tall-cell PDF

Source: `docs/papers/tallCells.pdf` (archived copy of
<https://matthias-research.github.io/pages/publications/tallCells.pdf>).
This appendix records the paper's exact equations and parameter statements.
Where the paraphrase above disagrees, THIS APPENDIX IS AUTHORITATIVE and the
disagreement is flagged.

### A.1 Packed layout (Sec 3.1, Eq 4–6)

Quantities are stored in a compressed array `q_{i,j,k}` of size
`(B_x, B_y + 2, B_z)`. Terrain height `H_{i,k}` and tall-cell height `h_{i,k}`
are 2D arrays. The uncompressed y of array element `q_{i,j,k}` (one-based j):

```text
y_{i,j,k} = H_{i,k} + 1              if j = 1 (tall cell bottom)
            H_{i,k} + h_{i,k}        if j = 2 (tall cell top)
            H_{i,j} + h_{i,k} + j-2  if j >= 3 (regular)             (4)
```

Evaluating `q` at uncompressed `(x,y,z)`:

- `y <= H_{x,z}`: value below terrain.
- `H_{x,z} < y <= H_{x,z}+h_{x,z}` (inside the tall cell):

```text
q_(x,y,z) = ((y - H_{x,z})/h_{x,z}) q_{x,2,z}
            + (1 - (y - H_{x,z})/h_{x,z}) q_{x,1,z}                  (5)
```

- `H+h < y <= H+h+B_y`: `q_(x,y,z) = q_{x,(y-H_{x,z}-h_{x,z}-2),z}`   (6)
  (paper's one-based j; the `-2`→`+2` shift depends on index convention).
- otherwise: the above-air value.

NOTE (A.1a): Eq 4 places the endpoint samples at the CENTERS of the bottommost
(`H+1`) and topmost (`H+h`) subcells — separation `(h-1)Δx` — but Eq 5's
interpolation parameter `(y-H)/h` treats the endpoint values as if they sat at
`y=H` and `y=H+h`. The paper's interpolation is therefore not endpoint-exact
at the stored sample positions (at `y=H+1` it returns
`(1/h)q_top + (1-1/h)q_bottom`, not `q_bottom`). Any implementation must pick
one convention and use it consistently in reconstruction, divergence,
Laplacian, gradient, restriction, and prolongation.

### A.2 Velocity extrapolation (Sec 3.3)

Eq 7 (per component, fictitious time `tau`): `du/dtau = -(∇φ/|∇φ|)·∇u`.

Exact procedure: "we apply the algorithm proposed in [Jeong et al. 2007] only
in a narrow band of two cells. Outside this region we use a hierarchical grid
for extrapolating the velocity field."

Hierarchy (Sec 3.3.1): `L = log2 min(B_x, B_y, B_z)` levels; finest level L is
the simulation grid. Coarser levels:

```text
H^l_{i,k} = floor( min_{i'=2i..2i+1, k'=2k..2k+1} H^{l+1}_{i',k'} / 2 )      (8)
h^l_{i,k} = ceil( max_{i'=2i..2i+1, k'=2k..2k+1} (H^{l+1}_{i',k'} + h^{l+1}_{i',k'}) / 2 ) - H^l_{i,k}   (9)
Δx^l = 2Δx^{l+1};  B^l_x = B^{l+1}_x/2, B^l_y = B^{l+1}_y/2, B^l_z = B^{l+1}_z/2
```

Sweep down then up: on the finest level a cell's velocity is "known" if the
cell is liquid or already extrapolated (by the 2-cell PDE band). Fine→coarse:
tri-linear interpolation using only known values, renormalizing weights; a
coarse cell is known if at least one corresponding finer cell is known.
Coarse→fine: fill fine unknowns by tri-linear interpolation from coarser
grids. "After these two passes every cell of the finest grid has a known
velocity."

### A.3 Level set reinitialization (Sec 3.4) — exact stabilizations

1. reinitialization runs "only every ten frames";
2. during reinitialization, φ values of grid points next to the surface are
   not modified (avoids moving the interface);
3. "in every frame we clamp the value of φ next to the liquid surface to not
   exceed the grid spacing Δx";
4. all |φ| clamped below `5Δx`.

### A.4 Advection (Sec 3.5)

Velocity: modified MacCormack [Selle et al. 2008], reverting to
semi-Lagrangian "if the new velocity component lies outside the bound of the
values used for interpolation". φ: semi-Lagrangian. One ray trace shared by
all quantities (collocated). External forces forward-Euler after advection.

### A.5 Remeshing (Sec 3.6) — exact text

Liquid cells: `φ <= 0`. Constraints: (1) at least `G_L` regular cells below
the bottom-most liquid surface; (2) at least `G_A` regular cells above the
top-most liquid surface; (3) adjacent tall-cell heights differ by ≤ `D`.

Procedure: per column compute the maximum and minimum y of the tall-cell top
satisfying (1) and (2) respectively; init `y_tmp` to their average; run
several smoothing passes on `y_tmp`, clamping during smoothing to satisfy (1)
and (2), "giving preference to condition (2) by enforcing it after condition
(1)"; finally enforce (3) Jacobi-style:

```text
y'_tmp[i,k] = min( y_tmp[i,k],  max_{|i'-i|+|k'-k|=1} y_tmp[i',k'] + D )   (10)
```

Parameters: `8 <= G_L <= 32`, `G_A = 8`, `3 <= D <= 6`, one to two Jacobi
iterations. Then `h_new = y_tmp - H`.

FLAG (A.5a): the paraphrase in §8 above printed Eq 10 as
`min(y, max_neighbors(y - D))`, which is a different (and collapsing) bound.
The paper's form only lowers a column that pokes more than `D` above its
HIGHEST neighbor. Verify what `limitNeighboringTallCellBases` /
`smoothRemesh` actually implement before changing anything.

Transfer: regular cells copy values at corresponding locations from the old
grid "or interpolate linearly if the location was occupied by a tall cell in
the previous time step". Tall cells: "we do a least square fit to obtain the
values at the bottom and the top of the cell".

The paper does NOT state any per-step limit on how far `h` may move between
steps (no temporal clamp), and remeshing runs every step (Algorithm 1).

### A.6 Pressure (Sec 3.7, Eq 11–18) — exact operators

```text
∇·(u* - (Δt/ρ)∇p) = 0                                                (11)
∇²p = (ρ/Δt) ∇·u*                                                    (12)
(∇·u)_{i,j,k} = (∂u/∂x) + (∂v/∂y) + (∂w/∂z)                          (13)
(∂u/∂x)_{i,j,k} = (u+_{i,j,k} - u-_{i,j,k}) / Δx
u+_{i,j,k} = (u_{i,j,k} + u_{(i+1,y,k)})/2   if cell (i+1,y,k) not solid
             u_solid                          otherwise               (14)
(∇²p)_{i,j,k} = (∂²p/∂x²) + (∂²p/∂y²) + (∂²p/∂z²)                    (15)
(∂²p/∂x²)_{i,j,k} = (p*+_{i,j,k} - 2 p_{i,j,k} + p*-_{i,j,k}) / Δx²
p*+_{i,j,k} = p_{i,j,k} · φ_{(i+1,y,k)} / φ_{i,j,k}     if cell (i+1,y,k) is air
              s_{(i+1,y,k)} p_{i,j,k} + (1 - s_{(i+1,y,k)}) p_{(i+1,y,k)}  otherwise  (16)
(∇p)_{i,j,k} = [(∂p/∂x), (∂p/∂y), (∂p/∂z)]^T                          (17)
   with (∂p/∂x)_{i,j,k} = (p*+_{i,j,k} - p*-_{i,j,k}) / Δx
u_{i,j,k} -= (Δt/ρ)(∇p)_{i,j,k}                                       (18)
```

`s_{i,j,k}` is the solid fraction; Eq 16's first line is the ghost-fluid
method (p = 0 on the liquid surface, φ ratio form); the second line is valid
for any s in [0,1].

NOTE (A.6a): Eq 14's `u+` and `u-` are averages at ±Δx/2, so Eq 13 is a
proper Δx-span centered divergence. Eq 17's `p*+` and `p*-` are (for liquid
neighbors) the neighbor cell-center pressures at ±Δx, yet the printed
denominator is `Δx`, not `2Δx` — the printed gradient is twice the centered
estimate and is dimensionally inconsistent with Eq 14's construction. The
repository's `2Δx` interior / `Δx` wall reading treats this as an erratum.
This must be settled empirically (hydrostatic + projection-idempotency tests)
after the endpoint-wetness fixes land; the paper text offers no further
disambiguation.

NOTE (A.6b): the paper states "the divergence is only measured at the top and
the bottom of tall cells, in the center, the solver is only aware of water
flow in adjacent cubic cells, not inside the tall cell, which results in
slight water gain over time. …we chose speed over accuracy in this trade off.
To mitigate the problem, we make sure that the heights of adjacent tall cells
do not differ too much, using parameter D." And: "our pressure projection
operator is not idempotent because the Laplacian is not a composition of
gradient and divergence and hence may not eliminate divergence completely."

### A.7 Multigrid (Sec 3.7.1, Eq 19, Alg 2–4)

Algorithm 2: build `A^L`; for l = L-1 down to 1: downsample φ, s; build
`A^l`. `b^L = -(Δt/ρ)(∇·u)` [as printed]; `p^L = 0`; run `num_Full_Cycles`
full cycles then `num_V_Cycles` V-cycles.

Downsampling: `s`: 8-to-1 average for regular cells and "a least square fit of
the 8-to-1 averages of the sub cells for the tall cells". φ: 8-to-1 average if
all 8 values share a sign or `l < L - C`; otherwise average of the positive
φ-values only. `C = 2` in all simulations.

Coefficients of `A^l` are recomputed from Eq 16 on EVERY level (sub-grid
ghost-fluid + solid fraction on all levels).

Smoother: Red-Black Gauss-Seidel, two parallel passes. Restriction:
tri-linear interpolation of r where

```text
r_(x,y,z) = r_{x,1,z}                     if y = H_{x,z} + 1
            r_{x,2,z}                     if y = H_{x,z} + h_{x,z}
            r_{x,(y-H-h-2),z}             if H+h <= y < H+h+B_y
            0                             otherwise                   (19)
```

"r_(x,y,z) is zero everywhere inside a tall cell except at the top and
bottom." Prolongation: tri-linear; out-of-grid samples ignored with weight
renormalization; if all samples are outside, pressure = 0.

Three critical convergence requirements (verbatim list): (1) full-cycles;
(2) preserving air bubbles in the finest levels; (3) ghost fluid and solid
fraction methods. "Not considering any one of these leads to either stagnation
or even divergence."

Algorithm 3 V_Cycle(l): if l==1 solve `A¹p¹=b¹`; else num_Pre_Sweep smooths,
`r = b - Ap`, restrict to `b^{l-1}`, `p^{l-1}=0`, V_Cycle(l-1),
`p += Prolong(p^{l-1})`, num_Post_Sweep smooths.

Algorithm 4 Full_Cycle(): save `p_imp = p^L`; `r^L = b^L - A p^L`; restrict r
down to level 1; `b¹ = r¹`; solve level 1; for l = 2..L: `p^l = Prolong(p^{l-1})`,
`b^l = r^l`, V_Cycle(l); finally `p^L = p_imp + p^L`.

Results (Sec 4): Δt = 1/30 s everywhere; "executing two V-cycles and one full
multigrid in the pressure solver is sufficient"; benchmark used
`num_Pre_Sweep = num_Post_Sweep = 2`. Grids up to 128×(32+2)×128 sim.

### A.8 Optimizations (Sec 3.8, verbatim highlights)

- interpolate along y first (2 consecutive packed samples per column);
- Gauss-Seidel obtains the pressure below a tall top via the compressed
  neighbor `p_{i,1,k}` by modifying the Laplace stencil implicitly;
- clamp the hierarchy at the level fitting GPU shared memory; solve the top
  level with multiple Gauss-Seidel iterations in a single kernel;
- build the hierarchical grid once per frame at the incompressibility stage
  and reuse it for velocity extrapolation next step (remeshing happens after
  extrapolation).

### A.9 Rigid coupling (Sec 3.9.1)

Voxelize bodies into solid fraction `s`; blend fluid/solid velocities by s;
"the divergence calculation treats a cell as solid if s > 0.9". In-solid
level set `φ^s` diffusion:

```text
φ^s_{i,j,k} = (1/S) Σ_{|i'-i|+|j'-j|+|k'-k|=1} (1 - s_{(i',j',k')}) φ_{(i',j',k')}   if S > 0
φ^s_{i,j,k} = (1/6) Σ φ_{(i',j',k')}                                                 otherwise
S = Σ (1 - s_{(i',j',k')})
```

Mixed cells blend `s φ^s + (1-s) φ`. Buoyancy from s and relative density;
drag proportional to s and relative velocity.

---

## Appendix B. Verbatim transcription: Mass-Conserving Eulerian Liquid Simulation

Source: `docs/papers/massConservingLiquids.pdf` (archived copy of
<https://matthias-research.github.io/pages/publications/masscon_sca.pdf>,
Chentanez & Müller, SCA 2012). This is the volume-loss-correction companion
paper. Equation numbers below are THAT paper's.

### B.1 Setting and discretization

The liquid domain is `ρ > 0.5` of a *surface density* field ρ (not mass
density), advected by `∂ρ/∂t = -u·∇ρ` (Eq 3) and "periodically sharpened to
prevent the 0.5 iso-contour from being blurred by numerical damping."

IMPORTANT (B.1a): this paper uses a REGULAR STAGGERED (MAC) grid — velocity
components at face centers, p and ρ at cell centers (Sec 3.1). The tall-cell
paper is collocated. Combining the two therefore requires adaptation by
construction; the sharpening/correction machinery below is defined on cell
centers and carries over, but face-based quantities (Eq 18–19 solid area
fractions) need collocated translation.

Algorithm 1 (time step): 1. Velocity extrapolation; 2. Density advection and
density sharpening; 3. Velocity advection and external force addition;
4. Incompressibility enforcement.

Velocity extrapolation (Sec 3.3) cites the tall-cell paper's method verbatim
(Jeong et al. a few cells from the interface, then the grid hierarchy).

### B.2 Conservative density advection (Sec 3.4)

Based on Lentine et al. [LGF11]/[LAF11] conservative semi-Lagrangian
advection (`ρ^{n+1} = A ρ^n`, backward-trace weights `w-_{ij}`, forward-trace
weights `w+_{ij}`, row sums γ_i, column sums β_j), modified to need only 3
scatter passes. The paper's exact modified scheme:

1. Advect γ_i using the backward semi-Lagrangian method (γ = 1 at first step).
2. Initialize β ← 0.
3. β_l += w-_{li} γ_i (backward trace, tri-linear).
4. ρ^{n+1}_i = Σ_l (γ_l / max(1, β_l)) w-_{li} ρ^n_l ; γ'_i computed likewise.
5. γ ← γ'.
6. For each j with β_j < 1: ρ^{n+1}_k += ρ^n_j (1 - β_j) w+_{jk} (forward trace).
7. Similarly γ^{n+1}_k += γ^n_j (1 - β_j) w+_{jk}.
8. Apply diffusion as in the original approach [1 to 7 iterations: for
   neighbors i,j with γ_j > γ_i move ρ_i(γ_j-γ_i)/(2γ_j) from j to i and set
   both γ to (γ_j+γ_i)/2; does not change β].

The scheme is unconditionally stable and fully conservative (paper runs
CFL 25 at Δt = 1/30 s).

NOTE (B.2a): this repository instead uses a flux-form donor/receiver-limited
VOF transport (also conservative by pairwise cancellation, but CFL-limited).
This is a DIFFERENT advection operator from the paper's. It is a departure to
either replace or justify; at the repository's CFL ≤ 1 time steps the schemes
serve the same role, but the departure must be recorded, not silent.

### B.3 Density sharpening (Sec 3.5, Eq 4–17, Algorithm 2) — the missing stage

Mass change of cell i due to unit velocity along ±x (ΔT = 3 × simulation Δt):

```text
δ^{x+}_i = ∫_{C_i} ∇·(ρ[1,0,0]^T ΔT) dV ≈ -(ρ_i - ρ_{i-(1,0,0)}) Δx ΔT     (4,6)
δ^{x-}_i = ∫_{C_i} ∇·(ρ[-1,0,0]^T ΔT) dV ≈ -(ρ_{i+(1,0,0)} - ρ_i) Δx ΔT    (5,7)
```

(y and z analogous — upwind differences.) Maximum mass increase / decrease
under any unit velocity:

```text
ΔT|∇ρ|+_i = (1/Δx²)( max(max(δ^{x+},0)², min(δ^{x-},0)²)
                   + max(max(δ^{y+},0)², min(δ^{y-},0)²)
                   + max(max(δ^{z+},0)², min(δ^{z-},0)²) )^{1/2}          (8–10)
ΔT|∇ρ|-_i = (1/Δx²)( max(min(δ^{x+},0)², max(δ^{x-},0)²)
                   + max(min(δ^{y+},0)², max(δ^{y-},0)²)
                   + max(min(δ^{z+},0)², max(δ^{z-},0)²) )^{1/2}          (11–13)
```

Sharpening weight and density correction:

```text
w_i(ρ) = (ρ_i - 0.5)³ (1 - min(1, max_{j∈N(C_i)}(|ρ_i - ρ_j|) / τ))        (14)
Δρ_i = w_i(ρ) ΔT|∇ρ|+_i   if w_i(ρ) >= 0
       w_i(ρ) ΔT|∇ρ|-_i   if w_i(ρ) < 0                                    (15)
ρ_i ← ρ_i + Δρ_i                                                           (16)
```

`N(C_i)` = adjacent cells; τ = 0.4 (limits the max density difference between
adjacent cells; larger τ visually resembles surface tension).

Local mass conservation (their novel contribution — Mullen et al.'s global
redistribution moves mass across the whole domain and deletes small features):

```text
Δρ_i ← -ρ_i   if ρ_i + Δρ_i < 0 or ρ_i < ε        (ε = 1e-5)
       0      if ρ_i > 0.5
       Δρ_i   otherwise                                                     (17)
```

(second line: cells with ρ > 0.5 are not modified — "mass only moves from the
air side to the liquid side"). Update ρ with the modified Δρ (Eq 16), then
add back `-Δρ_i` locally via Algorithm 2:

```text
Algorithm 2: for each cell i:
  p = TraceAlongField(Position(i), ρ, ∇ρ, D·Δx)
  ScatterValue(p, -Δρ_i)
```

`TraceAlongField` starts at the cell center and follows ∇ρ (multiple forward
Euler sub-steps) until it reaches the 0.5 iso-contour, a distance `D Δx` is
covered, or a solid boundary is crossed. `ScatterValue` deposits `-Δρ_i` to
nearby grid points with tri-linear weights; weights of solid grid points are
zeroed and the rest renormalized. `D` between 1.1 and 3.1 (results use
D = 2.1); increasing D visually resembles surface tension.

### B.4 Solid boundaries (Sec 3.6, Eq 18–19)

With face non-solid area fractions `V^f` and cell non-solid volume fraction
`V_i`, the δ estimates become e.g.

```text
δ^{x+}_i ≈ -(ρ_i V^f_{i+(½,0,0)} - ρ_{i-(1,0,0)} V^f_{i-(½,0,0)}) Δx ΔT    (18)
```

If ρ_i grows beyond V_i: for partially solid cells compute excess
`d = ρ_i - V_i`, follow the gradient of the solid signed-distance AWAY from
the solid for distance `S Δx` (S = 1) and scatter `d` there, then subtract `d`
from ρ_i. Fully solid cells (V_i = 1 non-valid case) are handled by the
incompressibility step (below).

### B.5 Incompressibility with volume correction (Sec 3.7, Eq 20) — the missing correction term

Liquid fraction for the pressure solve (cannot use ρ directly, since a cell
with non-solid fraction V < 0.5 would read as air):

```text
ρ'_i = 0            if V_i = 0 (fully solid)
       ρ_i / V_i    otherwise                                              (20)
```

ρ' is extrapolated from cells with V > 0 into V = 0 cells so they can join
the linear system. Signed distance for the ghost-fluid boundary:

```text
φ_i = -(ρ'_i - 0.5) Δx
```

**Volume-gain correction (verbatim):** "To handle the cells with ρ'_i > 1
(whether or not V = 1 or V < 1), we add min(λ(ρ'_i - 1), η)/Δx to the
divergence, where we use λ = 0.5 and η = 1 in all our examples. This
artificial divergence pushes the excess density away from the cells whose
ρ' > 1. Mullen et al. [MMTD07] also added this term to the divergence but
with λ = 1 and η = ∞ which can cause stability problems when ρ' is much
larger than 1. … Adding additional divergence is important because in our
case, ρ' > 1 results in visual volume loss. With the method described above,
this problem gets gradually corrected over time."

Pressure is solved with the multigrid of [CM11a] (separating solid
boundaries), then velocity is corrected.

### B.6 Density post-processing for rendering (Sec 3.8, optional)

γ_i = 2 min(ρ_i, 0.5); Gaussian-blur γ (σ = 2Δx); ρ''_i = ρ_i / min(max(γ_i, θ), 1)
with θ = 0.01; render the 0.5 iso-surface of ρ''. Purely a rendering-side
enhancement to reveal sub-grid thin features; does not feed back into
simulation.

### B.7 Parameters and results

Δt = 1/30 s, Δx = 0.05 m, gravity 10 m/s², D = 2.1, grids up to
256×128×128, CFL up to 32. Mass conserved to arithmetic error in all
examples; volume (0.5 iso-contour) stays close but can dip when features
thin below grid spacing.

---

## Appendix C. Repository gaps identified against the transcriptions

1. Eq 10 paraphrase error (A.5a) — verify `limitNeighboringTallCellBases` /
   `smoothRemesh` against the paper's `min(y, max_neighbor + D)` form.
2. Endpoint sample semantics: the paper stores POINT samples at tall-cell
   endpoints (A.1); the repository stores a column-average VOF in the bottom
   sample and a band-max guide in the top sample and gates gravity, pressure
   wetness, projection, and φ on those values as if they were point samples.
3. Eq 5 parameterization (A.1a): `(y-H)/h` vs endpoint-center interpolation —
   pick and enforce one convention everywhere.
4. Eq 17 printed denominator (A.6a): settle by experiment after (2) lands.
5. Extrapolation (A.2): implemented 2026-07-15 — two narrow-band neighbor
   passes plus the Sec 3.3.1 hierarchical sweep
   (`lib/tall-cell-extrapolation.ts`).
6. Remeshing (A.5): repository seeds crossings only from the stored band and
   rate-limits base movement per step; the paper scans the full column via φ
   and has no temporal clamp.
7. Density sharpening (B.3): implemented 2026-07-15 on both solver paths
   (sharpenCompute/sharpenScatter/sharpenResolve kernels).
8. Volume-gain correction divergence term (B.5): implemented 2026-07-15
   (`volumeCorrectionDivergence`, λ=0.5, η=1 expressed as a rate against the
   paper's 1/30 s step).
9. Advection operator (B.2a): flux-form VOF vs the paper's conservative
   semi-Lagrangian — departure must be recorded/justified or replaced.
10. Grid layout (B.1a): mass-conserving paper is staggered; tall-cell paper
    is collocated. The composition uses the tall-cell collocated layout; the
    sharpening and correction terms are cell-centered and translate directly.
11. Section 5 middle faces: the isolated probe shows error growing with tall
    depth even when all height transitions are removed. Production currently
    caps `maximumTallHeight=3` as a paper-compatible parity boundary; lifting
    it is blocked on a coherent middle-face pressure representation.

### Appendix C update — 2026-07-16 full-sweep audit (items 12–22)

Status refresh of items 1–11: (1) RESOLVED as a recorded deviation, see item
12. (2) root cause of the 07-15/16 instabilities; migration plan in
`TALL_CELL_LEVELSET_MIGRATION.md`. (3) resolved: the repository uses the
endpoint-exact convention (`t = y/(base-1)`) consistently in pressure/solid/φ
reconstruction; velocity remains a separate departure (item 13). (5,7,8)
implemented. (6) resolved 07-15 (full-column scan, no rate limit). (11)
superseded: the parity cap was removed 07-16 after the constant-density and
control-volume fixes; the flow gate now bounds the same error dynamically.

New items found by sweeping Appendices A/B against the current tree:

12. **Eq 10 smoothing operator (closes the A.5a "verify" flag).** The paper
    lowers a column only when it exceeds its HIGHEST neighbor + D
    (`min(y, max_neighbor + D)`); `smoothRemesh` and
    `limitNeighboringTallCellBases` apply `min` against EVERY neighbor + D —
    the strict pairwise form. Ours is STRONGER than the paper's operator and
    exactly matches the paper's stated constraint (3); the paper's own Eq 10
    does not actually guarantee pairwise ≤ D. Recorded as a deliberate
    strengthening; cost is extra remesh churn on steep staircases.
13. **Tall-interior velocity reconstruction is piecewise-constant, not Eq 5
    linear** (top world cell = top dof, all others = bottom dof;
    `validVelocityCell`). Chosen 07-15 to make the pressure stencil an exact
    div∘grad pair; it is the deviation that made the VOF store mass leak
    possible (see stability doc) and it biases advection traces inside deep
    tall cells. Under the level-set migration it should be revisited against
    Eq 5.
14. **Narrow-band extrapolation method.** Paper: Jeong et al. PDE-form
    directional extrapolation `du/dτ = -(∇φ/|∇φ|)·∇u` in a TWO-cell band,
    then the hierarchy. Ours: isotropic valid-neighbor averaging run
    `max(2, airHalo) = 8` passes, then the hierarchy. Direction-blind
    averaging smears tangential liquid velocity into air along all axes and
    over a 4× wider band; this is adjacent to the air-velocity igniter class
    of failures and should be re-examined (a φ field makes the paper's form
    implementable directly).
15. **Ghost-fluid θ floor.** Eq 16's ghost coefficient is the raw φ ratio;
    the repository floors the interface fraction at 0.05 in
    `pressureTerm`/`interfaceFraction`/`ghostFraction` (up to 20× gradient
    amplification bound). A stabilizer absent from the paper; flagged as a
    suspect in the (since fixed) episodic blow-up. Keep, but recorded.
16. **Multigrid restriction weighting.** Paper Sec 3.7.1: coarse RHS is
    ordinarily an 8-to-1 average with least-squares fits for tall endpoints;
    the repository trilinearly point-samples the Eq 19-masked residual at
    the coarse sample position (`restrictResidual`). Same masking, different
    quadrature. Also the smoother uses 0.8 under-relaxed RBGS (paper: plain
    RBGS).
17. **planRemesh split initialization.** Paper initializes `y_tmp` from the
    AVERAGE of the two feasibility bounds; the repository clamps the OLD base
    into the feasible interval (hysteresis). Deliberate (reduces base churn)
    but unrecorded until now. The paper also re-clamps constraints (1)/(2)
    during every smoothing pass; the repository re-applies only the
    representability floors.
18. **No separate high-resolution surface-tracking grid.** The paper carries
    φ on a finer grid than the simulation grid (their reported results pair
    e.g. 128³ simulation with higher surface grids); this repository tracks
    the surface at simulation resolution. Feature gap with direct visual
    impact; independent of the VOF/φ choice.
19. **No terrain height `H`.** The packed layout hard-codes H = 0 (flat
    floor). The paper's tall cells sit on per-column terrain. Feature gap.
20. **Rigid-body trace repair missing (A.9).** The paper diffuses a separate
    in-solid level set `φ^s` so semi-Lagrangian traces that land inside
    bodies sample repaired values; the repository has no equivalent, so
    traces through rigid bodies sample coupled-solid velocities directly.
    Affects `dam-break-boxes`/`sphere-jet` class scenes only.
21. **Mass-conserving paper solid handling absent (B.4/B.5).** Sharpening
    ignores face non-solid fractions `V^f`; there is no `ρ > V_i` excess
    scatter along the solid gradient (S = 1), and pressure classification
    uses raw ρ rather than `ρ' = ρ/V_i` with extrapolation into fully solid
    cells. Matters at walls and rigid bodies; moot after the level-set
    migration retires ρ.
22. **Correction-divergence units (B.5).** The paper adds
    `min(λ(ρ'−1), η)/Δx` — with an explicit 1/Δx — to the divergence; the
    repository expresses it as the rate `min(0.5·excess, 1)·30 /s` with no
    1/Δx. At the current grid (h ≈ 0.067 m) the paper's literal reading is
    ~15× stronger. The repository's rate form is deliberate (dt-independent)
    but the magnitude discrepancy is unresolved; revisit when calibrating
    the global volume control in the migration (its λ_v inherits this
    question). Also unimplemented and optional: B.6 render-side density
    post-processing (γ-blur thin-feature reveal).
