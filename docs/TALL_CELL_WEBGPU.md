# Restricted Tall-Cell WebGPU Solver

## Reference and scope

This implementation follows Chentanez and Müller, “Real-time Eulerian Water
Simulation Using a Restricted Tall Cell Grid” (SIGGRAPH 2011).

Reference: <https://matthias-research.github.io/pages/publications/tallCells.pdf>

Each x/z column contains one variable-height bottom cell and a moving band of
ordinary cubic cells near the liquid surface. The browser uses density-based surface
tracking and immersed rigid bodies while implementing the paper's endpoint
reconstruction, bounded MacCormack velocity advection, remeshing, ghost-fluid
pressure boundary, collocated operators, and full-cycle multigrid hierarchy.

The previous uniform cubic WebGPU solver remains available through the A/B
selector at the same x/z and cubic-equivalent y spacing.

## Packed layout and representability

For a cubic-equivalent grid `nx × fineNy × nz`, each column stores two tall-cell
endpoint samples followed by `regularLayers` cubic samples. Textures therefore
have dimensions `nx × (regularLayers + 2) × nz`; a 2D texture stores each
column's tall height.

The requested `regularLayers` is the paper's construction-time `B_y`. The
planner increases it when the vertical crossings within a column and their
liquid/air halos do not fit. A vertical dam face is represented horizontally
between neighbouring tall endpoint fields, so the balanced Figure 4 scene
keeps 24 regular layers and starts with 2,408 tall columns rather than entering
the ordinary-grid limit.

The planner retains liquid and air halos, includes moving-body bounds, limits
neighboring height differences by `D`, treats a wet band ceiling as unresolved
surface, and limits temporal height changes. It maps one-subcell tall cells to
the ordinary-cell limit to avoid coincident endpoint unknowns.

## Pressure translation

The pressure path:

- reconstructs a narrow signed-distance field from VOF and clamps it to `5 dx`;
- retains mixed-sign air on the two finest coarse levels (`C=2`);
- evaluates ghost-fluid and solid-fraction coefficients on every level;
- downsamples solid fractions by 8-to-1 averaging and tall-cell endpoint least
  squares;
- zeros residuals through tall interiors as in Equation (19);
- ignores out-of-grid prolongation samples and renormalizes their weights;
- uses two damped red-black Gauss-Seidel pre/post sweeps;
- clamps the hierarchy when its top level fits one 256-thread workgroup; and
- executes one full cycle followed by two V-cycles; uses 32 coarsest RBGS
  iterations for ordinary-depth initial solves and 192 for extreme-depth
  initial solves, with 16 and 144 iterations respectively for correction
  visits; and
- caches recurring bind groups and records the multigrid dispatch sequence in
  one compute pass.

Divergence uses Equation (14)'s average of adjacent collocated velocities, or
solid velocity when the adjacent cell is more than 90% solid. Projection uses
Equation (16)'s ghost-fluid air pressure and solid-fraction pressure blending.
For stability, the pressure difference is divided by the physical span between
its samples: `2 dx` for an interior centered pair and `dx` at a one-sided wall.
The paper prints `dx` in Equation (17) even though the samples are two cell
centers apart. The literal form reflected each hydrostatic gravity impulse and
caused the deep tank eruption even after the remesh constraints were enforced.
The `2 dx` denominator is treated as a correction to that printed stencil, not
as a consequence of using a different storage layout; the paper and this
implementation both store collocated velocities.

## Transport, remeshing, and conservation

Surface density uses the same strictly donor/receiver-limited conservative VOF
face flux as the matched cubic solver. Receiver capacity excludes speculative
outflow: including raw outflow can permit more inflow than the eventually
limited outgoing faces carry, after which a density clamp silently destroys
mass. A regular sample has unit control volume and the
bottom tall sample has volume equal to its covered cubic-cell height. Every
ordinary face contribution is shared by its two adjacent control volumes, so
the update conserves mass by pairwise cancellation rather than a post-step
rescaling.

Pairwise transport is restricted to faces whose donor and receiver world cells
are both represented by their packed columns. Treating an above-band location
as an empty receiver lets the donor lose flux with no packed sample available
to own it. The air halo and subsequent remesh move representable faces; the
packed boundary itself remains conservative.

This transport is not the complete density method from *Mass-Conserving
Eulerian Liquid Simulation*. That paper follows conservative advection with a
local conservative density-sharpening stage. The current solver does not yet
implement that stage, so a moving front can still diffuse into fractional VOF
values even when its total represented mass is conserved.

A deep face shared by two tall cells uses at most 12 stratified samples. Both
columns evaluate the identical oriented face integral, so its approximation
still cancels pairwise. Only the portion where one side is tall and the other
is regular is expanded cell-by-cell; the paper's neighbor-height bound `D`
keeps that work bounded. Dispatch and deep shared-face work therefore remain
independent of full domain depth.

The Tall method routes to the cubic backend when the construction-time surface
range of vertical crossings makes a height-two tall cell geometrically
impossible. Horizontal liquid/air faces do not trigger this limit.

The band is remeshed after advection on every step, matching Algorithm 1.
Regular values are copied or interpolated at their world positions
and endpoint velocities use a least-squares fit. For surface density, the
shader computes the old represented column amount and assigns the residual
after copying the new regular band to the new tall cell. This preserves every
representable column integral without a global volume correction. The residual
is no longer clamped to one: density above one is temporary stored mass, not
volume to erase.

The bottom density sample is the authoritative conservative average for the
tall control volume. The top density endpoint is a zero-mass topology sample:
advection and remapping fit it from the maximum density in the overlying stored
regular band. Keeping these roles separate preserves the column integral while
allowing the two tall endpoints to represent a vertical crossing. It prevents
a sub-threshold average from appearing as a missing tall cell directly beneath
threshold-liquid regular cells.

Temporary bottom averages above one remain valid through the next advection;
they are not clamped before bounded face fluxes can redistribute the remap
residual. In a restricted packed layout every column retains a height-two or
taller bottom cell. Base zero belongs only to the separately allocated full
uniform backend: using it with a fixed surface band would discard the column
residual above that band.

Remesh surface bounds use liquid/air sign changes, matching the paper's
reinitialized `phi` test, rather than treating every fractional VOF sample as a
separate surface. When `G_L` and `G_A` conflict, the air constraint wins as in
the paper. The neighbor-height bound `D` is applied to the newly proposed
field with eight ping-pong Jacobi passes. A live dam-break readback showed that
four passes left adjacent `22/16` bases when `D=4`; eight removes that incomplete
propagation. The smoke runner reads back every split and fails if any adjacent
delta exceeds `D`.

The scientific grid overlay interpolates the bottom and top endpoint topology
values and uses the same `0.5` liquid threshold as the pressure domain and
surface extractor. The smoke runner separately checks the conservative bottom
average and fails the dam case if both endpoints are dry while a stored regular
cell above is liquid.

An inflow is a one-sided face connected to a virtual upstream reservoir. Its
bounded receiver source is separate from internal pairwise transport, and the
receiver/nozzle velocity samples remain known during extrapolation. This avoids
both density accumulation at the outlet and erasure of a newly entering jet.
The prescribed reservoir occupies the open channel through the displayed
nozzle: inlet cells are excluded from the tall-cell solid mask, and the inlet
velocity is restored after immersed-body coupling. The visible nozzle tip and
analytic boundary face share the same position.

External force integration is restricted to the liquid domain. Extrapolated
air samples support characteristic tracing but do not accumulate their own
gravity impulse.

The bounded deep-face quadrature and the ordinary-cell fallback are covered by
layout tests. Long-run hose and deep-water measurements are required whenever
the transport or remeshing rules change.

## Rigid-body coupling

Rigid bodies are voxelized with eight subcell samples per stored cubic sample.
The solid fraction blends fluid and rigid velocities before projection,
participates in divergence and every multigrid level, and contributes an
opposite drag impulse and torque to the body. Approximate buoyancy is computed
from displaced volume. Body bounds constrain remeshing so solids stay in the
regular band. The UI no longer fabricates a zero momentum-closure error for
buoyancy that is not present in the explicit GPU exchange buffer.

## Deliberate remaining departures

- Surface density is persistent. A narrow signed-distance field is reconstructed
  for pressure; the paper advects level set `phi` directly and periodically
  reinitializes it.
- Velocity extrapolation uses repeated fine-grid neighbor passes rather than
  the paper's multigrid known/unknown hierarchy.
- The separately diffused in-solid `phi_s` field and resolved pressure traction
  are not implemented.
- Terrain cut cells, particle level-set tracking, particle thickening, foam,
  spray, and mist are outside this browser core.
- The renderer continues to use the persistent VOF `0.5` isocontour. In the
  paper's Tank example, surface tracking uses twice the simulation resolution
  along each axis, and the optional particle-thickening extension protects thin
  features. Those mechanisms are not reproduced here, so matching pressure and
  inflow transport does not imply identical small-scale surface detail.
- The compact pressure path remains substantially more expensive per stored
  sample than the legacy Uniform Jacobi comparison. Balanced dam-break runs
  are currently slower; deep aspect-ratio scenes are where compression
  amortizes this overhead.

## Stability diagnostics

The GPU readback reports liquid and extrapolated-air speed extrema, divergence
before and after projection, exact finest-level pressure residual, relative
residual, pressure maximum, component CFL, finite-state count, volume drift,
and cubic-equivalent locations for each important extremum. It also separates
requested UI time from encoded GPU simulation time and reports lag rather than
silently discarding clamped time.

See [`TALL_CELL_STABILITY.md`](TALL_CELL_STABILITY.md) for the full paper/code
audit and the before/after deep tank measurements.

Deep-scene performance must be remeasured whenever the smoother or cycle
budget changes. At ordinary depth, multigrid overhead can make the tall path
slower; this is not presented as a universal optimization.
