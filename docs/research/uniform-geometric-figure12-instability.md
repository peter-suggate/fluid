# Figure 12: instability introduced by the separating-boundary change

2026-09-20. Investigation and proposal; experimental shader edits were restored.
This supersedes the earlier boundary review's implementation confidence.

## Finding

The new closed-domain dual-volume rule creates a fictitious fluid face where
Figure 12's voxel vessel overlaps the domain floor. The pressure matrix and
velocity projection then disagree on the pressure and ghost-fluid treatment
of that face. An unphysical floor velocity accumulates and feeds the vessel
interior. This is a demonstrated cause of the reported runaway, not evidence
that separating boundary conditions are inherently unstable.

The earlier diagnosis of suppressed wall separation was valid but incomplete.
I assumed that enabling solid pressure rows and retaining their face velocities
would preserve the existing solver/projection correspondence. It does not at
an embedded-solid/domain intersection. Flat-wall separation tests, conserved
mass, and an algebraic pressure residual did not test that correspondence.
The rendered-volume regression in the prior validation was also a reason to
withhold confidence in surface/volume consistency, even though it is not the
identified source of this particular feedback loop.

## Controlled Dawn reproduction

Scene: `cm12-figure-12`; user confirmed all defaults. Dawn Metal, balanced
Uniform Geometric defaults, 128 cubed, cell size 0.05 m, dt=1/30 s. No browser
or concurrent Dawn process. Baseline restores only the three changed uniform
shader files to HEAD `8750e8b5`; it uses identical scene and control values.
The experiment changes only the closed-domain face weight from 0.5 to
`0.5 * cellOpenFraction(interiorCell)`. All the new phi contact/release,
velocity extension, solid-pressure and solver settings remain enabled.

Peak absolute component on stored positive MAC faces (m/s):

| Frame | Time (s) | Before our changes | Current changes | Geometry-only experiment |
|---|---:|---:|---:|---:|
| 27 | 0.900 | 28.77 | 15.53 | 22.90 |
| 30 | 1.000 | 28.88 | 17.86 | 17.42 |
| 31 | 1.033 | 21.37 | 35.89 | 14.13 |
| 32 | 1.067 | 14.68 | 79.80 | 10.66 |
| 33 | 1.100 | 8.67 | 135.72 | 8.45 |
| 34 | 1.133 | 6.28 | 166.06 | 7.69 |
| 40 | 1.333 | 2.96 | 101.58 | 4.71 |

The geometry-only repeat was extended through frame 60 (2 seconds), where
peak stored-face speed is 2.17 m/s and maximum cell volume is still 16.68.
The runaway remains absent over that observed interval.

The baseline has a transient impact spike too; its peak subsequently decays.
It is not a correct reference solution: its maximum conserved cell volume
reaches about 440 at frame 40 (capacity is 1). The changed implementation
reaches about 20 by frame 30, so overfill is real but does not by itself
explain the new runaway. The geometry-only experiment also retains substantial
overfill. Removing the explosive feedback is not proof of transport quality.

A 64-cubed run of the changed implementation peaks near 23 m/s around impact
and does not reproduce the same runaway over 45 frames. Low-resolution
separation checks were insufficient for the default scene.

## Concrete mechanism

1. `pressureFaceData` in `webgpu-uniform-reference.wgsl.ts` handles domain
   boundaries before voxel solids. My new branch assigns 0.5 to every closed
   nonsymmetry domain face. At cell `(56,0,56)`, both the exterior and interior
   sides of the negative Y domain face are solid. Its actual fluid dual volume
   is zero. The separate embedded-solid branch would correctly return zero,
   but the domain branch returns first.
2. The solid cell `(56,0,56)` is next to liquid at `(56,1,56)`. The new
   `pressurePhi` continuation makes the solid cell a pressure unknown. The
   domain halo `(56,-1,56)` is two cells from the open liquid and stays air.
   This is also what `mgExtrapolatePhiOneCell` produces: it only extends from
   non-solid neighbours.
3. `mgCoefficientRaw` uses a ghost-fluid coefficient for the solid-liquid
   unknown / air-halo pair. `mgApply` and smoothing use **zero** pressure for
   a non-liquid neighbour. However, an inactive air texel can retain a
   nonzero value from multigrid transfers; smoothing passes it through.
4. The low-domain branch in `project` reads `projectPressureValue(halo)`
   without checking the halo's pressure membership and uses no ghost-fluid
   theta. It therefore applies a different gradient from the matrix. Its
   boundary velocity persists between steps through `carryBoundaryVelocity`.
5. The next RHS includes that false floor flux through `divergenceAt`.
   The solve responds by pushing on the adjacent liquid. The newly enabled
   extension preserves the resulting solid-contact velocity, making the
   feedback visible in the fluid.

A repeat run captured the scalar-packed negative-boundary buffer as well as
pressure and positive MAC velocities. Pressure lattice was `[130,130,130]`
with origin `[0,0,0]` throughout, so these are direct simulation coordinates.
At the same `(56,0,56)` column:

| Frame | False floor velocity | Upper solid/liquid velocity | Published weighted divergence in solid cell | Reported fine residual infinity |
|---|---:|---:|---:|---:|
| 30 | 111.236 | 10.788 | -1004.474 | 1.535 |
| 31 | 144.806 | 35.889 | -1089.169 | 1.526 |
| 32 | 151.893 | 79.801 | -720.922 | 1.111 |

Velocities are m/s, divergence and residual are /s. Tangential dual weights
are zero at this solid cell; with the erroneous 0.5 weights and h=0.05,
divergence is `10*(upper-lower)`. At frame 32, stored pressures are about
103157 Pa in the inactive halo, 92545 Pa in the solid cell, and 26295 Pa in
the open liquid cell. The algebraic solver ignores the inactive halo's
103157 Pa; the boundary velocity projection reads it. The solid pressure is
positive, yet its published flux is far from the constrained solution.
The release mask consequently also cannot be treated as a reliable contact
certificate while this mismatch exists.

The single geometry-only ablation breaks the false connection and removes
the 1.0–1.3 s runaway without tightening convergence, capping velocities,
changing the timestep, or disabling the new phi release logic. That isolates
the trigger more strongly than a broad revert or multiple simultaneous fixes.

## Revised proposal

1. **Make boundary geometry compositional.** Intersect the domain exterior
   with voxel, terrain and rigid occupancy before determining the fluid
   fraction of any dual cell. For this binary voxel case, a closed domain
   face has half the adjacent interior open fraction; two solid sides give
   zero. Preserve authored open boundaries and symmetry explicitly. Audit
   positive and negative domain planes and solid/domain intersections.
2. **Make projection implement exactly the solved operator.** Use the same
   pressure membership, dual weights, boundary pressures and theta as the
   finest matrix for every face, including domain faces. Canonical air
   pressure is zero regardless of scratch-texture contents. Avoid another
   local clamp that hides the discrepancy. Keep nonnegative solid pressure
   and derive release only from a consistent projected contact state.
3. **Then revisit phi and conserved volume together.** Keep proper separating
   contacts, but measure actual liquid departure as well as phi opening.
   Neither the baseline's pile-up nor the ablation's overfill is acceptable
   evidence of a complete wall-treatment solution. Do not turn on all phi/V
   agreement experiments as a substitute for diagnosing that transport issue.
4. **Use Dawn scene diagnostics for acceptance.** Figure 12 at default 128
   cubed through impact and rebound; figure 8 for wall detachment; hydrostatic
   and flat/embedded ceiling fixtures. Measure published divergence and
   contact complementarity alongside algebraic residual, include negative
   domain faces in velocity extrema, and track V/phi disagreement. Measure
   kinetic energy with the correct dual weights if using it as a stability
   criterion—the probe's owner-volume energy is only a diagnostic proxy.
   After implementation, run the canonical Sparse CM12 regression gate and
   report failures without weakening lanes or ceilings. No unit tests needed.

The next implementation should cover points 1 and 2 together, followed by
these scene checks. The investigation initially left the earlier production changes in place.
The subsequent authorized implementation is recorded below.

## Evidence and references

- `uniform-geometric-figure12-dawn-2026-09-20.json`: saved control values and
  per-frame traces for baseline, changed 64/128 and geometry-only 128 runs.
- `tools/probe-uniform-geometric-boundary-energy-dawn.ts`: reproducible Dawn
  stage probe; `--n=128 --steps=40 --out=/tmp/figure12 --capture=31,32,33,34`.
  Binary captures live under the recorded `/tmp/figure12-*` directories.
- CM11a, `docs/papers/A_Multigrid_Fluid_Pressure_Solver_Handling_Separat.txt`,
  section 3.1 and equation 11: one-cell solid continuation, non-solid dual
  volumes and pressure/normal-velocity complementarity. The paper's argument
  requires the variational operator; a nonnegative pressure clamp alone does
  not repair mismatched geometry or gradients.
- CM12, `docs/papers/massConservingLiquids.txt`, sections 3.6–3.7: solid-aware
  mass transport and pressure correction. Conservation alone is insufficient
  when mass piles up in cells inconsistent with the visible interface.
- `uniform-geometric-solid-boundary-review.md`: original source comparison
  with uniform density CM12 and Sparse CM12, and earlier validation limits.


## Implementation follow-up

The authorized correction is now in `webgpu-uniform-reference.wgsl.ts`:

- Domain dual weights include the adjacent cell's voxel/terrain/rigid open
  fraction. Closed solid/solid contacts have zero fluid weight; open-top
  ambient space and depth symmetry retain explicit treatment.
- `geometricProjectedFace` applies the same pressure membership, air-zero
  convention, ghost-fluid theta (including its numerical epsilon) and
  zero-weight cutoff as the finest matrix to interior and both domain sides.
- Release decisions use canonical pressure and exclude fully blocked faces.
  The density-based uniform branch keeps its previous projection path.

Default figure 12 at 128 cubed was run for 60 frames (2 seconds) in Dawn.
At frame 34, peak stored-face velocity is 7.71 m/s instead of 166.06 m/s;
at frame 60 it is 1.84 m/s. The former false floor face at `(56,0,56)` is
exactly zero in all captured frames 31–34. The adjacent liquid-facing
velocity is below 5e-7 m/s in magnitude.

Published weighted divergence was independently reconstructed from captured
MAC velocities on pressure-supported solid contacts. Its maximum is
2.379, 1.925, 1.520 and 1.044 /s at frames 31–34, below the corresponding
reported global residuals 8.489, 6.870, 5.128 and 3.874 /s. This replaces the
previous discrepancy of about 721 /s versus a reported 1.11 /s.

Volume/phi consistency remains unresolved: maximum V is 10.62 at 2 seconds,
and total V drift is approximately -0.000237%. The correction removes the
identified explosive feedback; it does not claim a complete transport fix.

The existing focused Dawn run passes ceiling separation, both embedded wall
orientations, figure-8 upper-shell detachment (386 initially wet vertices to
0 after one second), hydrostatic stability, translation and pressure-cycle
checks. Mini32's final capacity assertion fails; mass conservation still
passes and represented-volume drift is -3.93% at frame 30 and -4.39% at 90.
The failing Dawn assertion is retained, with no threshold relaxation.

Readbacks and logs: `/tmp/figure12-fixed128`,
`/tmp/fluid-boundary-correction-focused.log`.


Final focused rerun (including the new default figure-12 floor regression):
18/18 reported tests passed. Mini32's maximum V was 1.05716 at frame 90,
below the unchanged 1.1 bound; its represented-volume drift was -3.16%.
This differs from the earlier failing run, so that result remains recorded
above rather than being silently erased. The capacity result is sensitive
to the evolving numerical trajectory. Final log:
`/tmp/fluid-boundary-correction-focused-final.log`.
The canonical sparse gate passed 4/17 lanes in 437.2 seconds; the rest failed
on timeouts, the mini64 performance ceiling and topology assertions.
Repository type-checking still reports errors in other sparse tests/tools,
with none reported in the changed files.
