# Restricted Tall-Cell WebGPU Solver

## Reference and scope

This implementation follows Chentanez and Müller, “Real-time Eulerian Water
Simulation Using a Restricted Tall Cell Grid” (SIGGRAPH 2011).

Reference: <https://matthias-research.github.io/pages/publications/tallCells.pdf>

Each x/z column contains one variable-height bottom cell and a moving band of
ordinary cubic cells near the liquid surface. The browser retains VOF surface
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

`regularLayers` is a minimum, not permission to discard geometry. If the
initial vertical free-surface range and its available halos do not fit, the
layout increases the band up to the uniform-grid limit. The default dam break
therefore stores 104% of the cubic sample count (the two endpoint slots are
inactive), whereas a deep tank with a mostly horizontal surface remains
strongly compressed. A tall-cell interior cannot represent a vertical free
surface.

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
- executes one full cycle followed by two V-cycles, with 256 shared-memory
  red-black Gauss-Seidel top iterations per visit.

Divergence uses Equation (14)'s average of adjacent collocated velocities, or
solid velocity when the adjacent cell is more than 90% solid. Projection uses
Equation (16)'s ghost-fluid air pressure and solid-fraction pressure blending.
For stability, the pressure difference is divided by the physical span between
its samples: `2 dx` for an interior centered pair and `dx` at a one-sided wall.
The paper prints `dx` in Equation (17) even though the samples are two cell
centers apart. The literal form reflected each hydrostatic gravity impulse and
caused the deep tank eruption. This compatibility correction is therefore a
deliberate departure from the paper.

## Transport, remeshing, and conservation

VOF transport exchanges top vertical flux with the first regular cell and
integrates horizontal tall faces over every covered cubic subcell. Exposed
height differences next to shorter columns are summed explicitly.

On remesh, regular values are copied or interpolated at their world positions
and endpoint velocities use a least-squares fit. For VOF, the shader computes
the old represented column amount and assigns the residual after copying the
new regular band to the new tall cell. This preserves every representable
column integral without a global volume correction.

External force integration is restricted to the liquid domain. Extrapolated
air samples support characteristic tracing but do not accumulate their own
gravity impulse.

The default moving-dam browser sample measured 0.07% raw VOF drift at 2.24 s
after the adaptive uniform fallback. The matching uniform path measured
effectively zero drift. These are interactive measurements, not unit-test
acceptance thresholds.

## Rigid-body coupling

Rigid bodies are voxelized with eight subcell samples per stored cubic sample.
The solid fraction blends fluid and rigid velocities before projection,
participates in divergence and every multigrid level, and contributes an
opposite drag impulse and torque to the body. Approximate buoyancy is computed
from displaced volume. Body bounds constrain remeshing so solids stay in the
regular band. The UI no longer fabricates a zero momentum-closure error for
buoyancy that is not present in the explicit GPU exchange buffer.

## Deliberate remaining departures

- VOF is persistent. A narrow signed-distance field is reconstructed for
  pressure; the paper advects level set `phi` directly and periodically
  reinitializes it.
- Velocity extrapolation uses repeated fine-grid neighbor passes rather than
  the paper's multigrid known/unknown hierarchy.
- Remeshing is amortized every 60 encoded steps instead of every step.
- The separately diffused in-solid `phi_s` field and resolved pressure traction
  are not implemented.
- Terrain cut cells, particle level-set tracking, particle thickening, foam,
  spray, and mist are outside this browser core.
- Tall-face VOF evaluation is currently linear in tall-cell height. Packed
  pressure, projection, coupling, and advection dispatch counts remain
  independent of full domain depth.

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
