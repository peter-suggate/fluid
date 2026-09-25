# Hose-off hero pond: Dawn investigation

The reproduction uses the catalog's `hero-garden-hose-x10`, Uniform Geometric,
144×96×96 cells at 12.5 mm, 1/30 s steps, and inflow disabled before solver
construction. The reported browser configuration had pressure residual
tolerance **0**, which executes the configured 3 Full-Cycles + 4 V-Cycles;
it does not mean that the solve reaches zero residual. Surface tension,
sharpening, redistancing, cubic advection and total-volume correction stay on
unless a named control arm explicitly disables one.

## Reproduce

Close or navigate away from WebGPU browser scenes first. The probe acquires
the repository-wide exclusive GPU lease and releases it in `finally`.

```sh
node --import tsx tools/probe-uniform-pond-rest-dawn.ts --arm=rest --frames=90
node --import tsx tools/probe-uniform-pond-rest-dawn.ts --arm=no-forces --frames=10 --sigma=0 --gravity=0
node --import tsx tools/probe-uniform-pond-rest-dawn.ts --arm=tight --frames=90 --values='{"pressureResidualTolerance":0.05}'
npm run test:dawn:uniform-pond-rest
```

JSON receipts go to `artifacts/pond-rest/`. The probe measures the vertical
zero crossing at fixed initially wet columns: 2,589 columns overall, including
1,999 with two cells of clearance from nearby solids. Heights are relative to
the authored waterline; RMS therefore includes mean-level drift. Missing
crossings are counted rather than silently dropped. Velocity is measured in
cells carrying liquid, and conservative volume is summed directly from V.

## Causes and controls

1. **Inaccessible initial mass.** The initial volume used fractional solid
   occupancy, while the solver's binary static mask closes every touched cell.
   This seeded 876.404 cell-volumes in 1,179 closed cells. Total-volume
   correction raised the visible water by 4.102 mm on step one to represent
   this mass. The rise persists with gravity and surface tension off and
   disappears with total-volume correction off. Initialization now uses the
   geometric solver's actual static capacity; the density reference path is
   unchanged.
2. **Contact continuation erased wet vertices.** At a submerged terrain
   corner, copying phi from a cell above the vertex could replace a negative
   wet value with positive air. Gravity then acted on a surface containing
   artificial holes. Increasing pressure work from 3+4 cycles/6 sweeps to
   5+8 cycles/8 sweeps lowered the first-step residual from 0.0351 to 0.00108
   s⁻¹ but left peak velocity at 0.0707 m/s. Contact now only adds wetting,
   matching domain-wall continuation; the explicit separating-air term still
   releases water. With sigma=0 this reduced first-step speed to 0.000196 m/s.
3. **Curvature read solid air sentinels.** After those fixes, enabling sigma
   raised first-step speed to 0.00862 m/s even though phi was planar. Geometric
   normals now differentiate live vertices within each open cell. Curvature
   uses one-sided differences at solids, and capillary acceleration does not
   interpret a solid face as a liquid/air interface. The initial capillary
   impulse disappears. A separate analytic sphere oracle checks that genuine
   curvature remains, with the correct sign and magnitude.
4. **Cubic interpolation reached buried vertices.** The 4³ stencil includes
   points outside the enclosing open cell. Their construction air sentinel is
   not physical phi. Cubic interpolation now falls back to the enclosing
   trilinear cell when a nonzero-weight sample is buried. Cubic reconstruction
   remains active elsewhere.
5. **Dust cleanup removed surface mass.** Deleting every deposit below 0.001
   cell-volumes slowly drained the pool; global correction faithfully lowered
   the surface to match. Turning both dust floors off preserved V to about
   2e-8 relative over three seconds. The regular floor now protects positive
   deposits in the existing 4h phi band, which already keeps those cells live.
   Far-air residue and negative roundoff cleanup remain enabled.

## Measurements during isolation

Each row includes all preceding fixes. These are diagnostic measurements, not
thresholds to be copied into unrelated lanes.

| State | Time | Interior RMS height error | Maximum sampled height error | Peak liquid-cell speed |
|---|---:|---:|---:|---:|
| Original reproduction | 1 s | 4.093 mm | 9.923 mm | 0.788 m/s |
| Correct initial mass | 1 s | 2.101 mm | 11.014 mm | 0.715 m/s |
| Contact fixed, sigma enabled | 3 s | 2.282 mm | 17.879 mm | 0.605 m/s |
| Curvature fixed | 3 s | 0.177 mm | 6.274 mm | 0.128 m/s |
| Cubic stencil fixed | 3 s | 0.0747 mm | 0.305 mm | 0.0222 m/s |
| Dust-off control | 3 s | 0.00682 mm | 0.200 mm | 0.0277 m/s |
| All fixes, tolerance 0 (also identical at 0.05) | 3 s | 0.00818 mm | 0.270 mm | 0.0156 m/s |

## Remaining numerical limit

The pond is not mathematically motionless: finite pressure residuals still
seed small currents. The normal 3+4 full schedule leaves approximately 0.038
s⁻¹ divergence residual here. A requested 1e-4 target exhausts the current
cycle budget and correctly withholds projection.

The global default tolerance of **5 s⁻¹** is substantially looser: a separate
three-second run with all geometric fixes still developed roughly 6.66 mm
interior RMS error. A 0.05 target works for this pond, but changing the global
default violated existing pressure-work budget checks in impact fixtures.
That experiment was reverted; no budget assertion or timing ceiling was
relaxed. Improving the general pressure accuracy/work tradeoff remains
separate from the five defects fixed here. The pond regression explicitly
pins the user's tolerance-zero configuration and does not claim the loose
default is calm.

## Why the pressure solve is demanding

The following separates verified implementation facts from hypotheses about
convergence. The experiments above establish accuracy sensitivity, but do not
yet apportion the remaining residual among hierarchy, boundary and smoother
errors.

* **Hydrostatic balance is reconstructed every step.** `mgBuildFinestRhs`
  initializes pressure to zero. Gravity remains -9.80665 m/s² with the hose
  disabled, so every solve must rebuild the pressure gradient that cancels
  it. At a 1/30 s step the unprojected gravitational velocity increment is
  about 0.327 m/s. Cold starting explains repeated work; a warm start is a
  plausible experiment, not a measured improvement yet.
* **The actual water is shallow and boundary dominated.** CPU enumeration
  of the corrected initial volume gives 15,340 occupied cells in 2,680
  columns. Column depths range from 0.6 to 8.6 cell-volumes per horizontal
  cell, with median 5.6 (70 mm at h=12.5 mm). These counts describe volume
  support, not the solver's larger set of pressure unknowns. The earlier
  2,589-column surface metric intentionally excludes the shallowest columns.
* **The hierarchy loses vertical detail quickly.** The actual planner returns
  144×96×96 → 72×48×48 → 36×24×24 → 18×12×12 → 9×6×6 → 9×3×3.
  Median water depth is only 0.7 cells by the fourth grid. Topology and
  interface distances are restricted separately; the last levels apply a
  sign-aware interface rule. Thus coarse correction sees an approximation
  of the thin curved basin. Reduced fidelity near the shoreline is a leading
  hypothesis, not proof that shallow pools are intrinsically ill-conditioned.
  The final level also stops coarsening x while halving y/z, giving unequal
  spacings and 275 cells including halo. This is supported behavior, not
  evidence by itself of a defect.
* **Free-surface coefficients are nonuniform even for this flat surface.**
  The authored waterline is y=20.6h, just 0.1h above the top liquid cell
  centre. Away from shore, ghost-fluid theta is therefore 0.1, and the
  vertical surface coefficient is 10/h² versus 1/h² for a full interior
  face. Near terrain, solid continuation and changing face volumes add more
  variation. Coefficient contrast is real; its contribution to iteration
  count still needs a waterline-offset control.
* **Solid contact is a constrained solve.** One layer of solid pressure
  unknowns participates, with a zero lower pressure bound. Projected updates
  enforce separating contact, and coarse corrections must preserve these
  bounds. This adds active-set decisions around the stepped basin to the
  usual free-surface pressure problem.

The acceptance metric is the maximum projected divergence residual in s⁻¹,
not mean pressure error or visible surface motion. A few difficult boundary
rows can set the stopping time. The measured ~0.038 s⁻¹ fixed-budget result
and the much worse default-5 run establish that accuracy matters here; they
do not establish where the worst rows lie. The most useful next diagnostic
is to record those locations and residual after each cycle, then compare
the authored basin with a flat basin at matched depth and waterline offset.
