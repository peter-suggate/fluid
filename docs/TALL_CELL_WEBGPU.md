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

The planner retains liquid and air halos, caps a tall store below a nearby
moving body's projected bottom without lifting the surface band, limits
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

Divergence and the applied pressure gradient use the one-sided face pairing:
each collocated sample's velocity acts as its positive-face value, divergence
is the backward difference of those faces (solid velocity when a neighbor is
more than 90% solid), and projection subtracts the forward difference
`(p*(q+e) - p*(q)) / (h·θ)` with Equation (16)'s ghost-fluid and
solid-fraction blending. The compact Laplacian solved by the multigrid is
exactly the composition of this pair, so projection is idempotent.

This replaces the paper's printed Equation (13/14) centered-average divergence
and Equation (17) gradient. That pair is inconsistent with the paper's own
compact Equation (15/16) Laplacian: the printed `dx` denominator (samples two
cells apart) doubles every correction and erupts hydrostatic columns, while
the corrected `2 dx` reading leaves a compact-solve/wide-apply mismatch that
measurably pumped energy at ghost-fluid faces — the 24-layer dam break grew
~1.2×/step at the dam-face bottom endpoints and destroyed all liquid by
t≈1.8 s (see `TALL_CELL_STABILITY.md`, 2026-07-15 audit, and the verbatim
transcription in `TALL_CELLS_PAPER.md` Appendix A.6a). The authors' successor
paper (*Mass-Conserving Eulerian Liquid Simulation*, Sec 3.1) abandons the
collocated pairing for a staggered grid outright.

Tall-cell endpoint samples take their wetness from the settled point-sample
reconstruction (paper Eq 4 stores point samples; the packed bottom texel is
the conservative column average): the fill height is `alpha·base` subcells,
the bottom endpoint is liquid when the fill covers it, and gravity, the
pressure right-hand side, projection, extrapolation seeding, and the
multigrid φ all read this view. Mass transport keeps the average as the
column integral. The renderer and readback reconstruct the tall interior the
same way, so a partially filled tall cell draws as bottom-settled water.

Remeshing follows paper Section 8 over the full column: a settled surface
inside the tall region is a crossing at the fill height, and there is no
per-step limit on base movement. Two conservative-VOF safeguards go beyond
the paper (whose level set silently deletes water a column cannot represent):
a column may never take a base too low to represent its own water
(representability outranks the neighbor bound `D` at such cliffs), and remap
residuals beyond one full tall cell settle upward into the band's remaining
capacity. Any remaining excess density is drained by the *Mass-Conserving
Eulerian Liquid Simulation* Section 3.7 correction: cells holding more than
they represent add `min(λ(ρ'-1), η)` artificial divergence (λ=0.5, η=1,
expressed as a rate against the paper's 1/30 s step).

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

The density method also implements the *Mass-Conserving Eulerian Liquid
Simulation* Section 3.5 sharpening stage after conservative advection, on both
the tall and cubic paths (Eq 4–17 with τ=0.4, ε=1e-5, fictitious step 3Δt, and
the Algorithm 2 local mass return tracing D=2.1 cells along ∇ρ; verbatim
transcription in `TALL_CELLS_PAPER.md` Appendix B.3). Corrections are
non-positive per Eq 17 — mass only moves from the air side to the liquid
side — and the removed mass is deposited near the 0.5 iso-contour through a
fixed-point atomic buffer, so the stage conserves mass locally to within
rounding of 2⁻¹⁶ per deposit. The `densitySharpening` method parameter (env
`FLUID_SHARPENING=0` in the smoke runner) disables it as a diagnostic
departure.

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
- Velocity extrapolation follows the paper's Section 3.3.1 hierarchy: two
  fine narrow-band neighbor passes (standing in for the Jeong et al. Eikonal
  band), then Eq 8/9 coarse grids swept fine-to-coarse and back so every
  sample carries a velocity (`lib/tall-cell-extrapolation.ts`). The
  `hierarchicalExtrapolation` option (env `FLUID_HIERARCHY=0`) reverts to the
  legacy repeated neighbor passes as a diagnostic departure.
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
