# Stationary bowl 2×: zero-gravity control and corrections

**Scope correction:** These captures use zero gravity and 1/60 s. The user
subsequently clarified the acceptance case is gravity −9.80665 m/s² at 1/30 s,
with both gamma diffusion and sharpening on. The results below are isolation
controls, not evidence that the gravity-on physics problem is resolved.

The corrected fully adaptive scene remains stationary through eight simulated
seconds. Its surface differs from min1/max1 by **0.0427 mm RMS and 0.0773 mm
maximum over the entire tank**. No force, damping, conditioning setting,
refinement region, or physics timestep was changed to obtain that result.

![Before and after](../artifacts/stationary-bowl-2x/ab-before-after.png)

## Configuration and measurements

Both arms use the actual `stationary-bowl-2x` UI scene and its balanced
adaptive-mass profile: coarse-first, scene timestep 1/60 s, macro span 1,
gamma diffusion on, sharpening on, live topology. A adds a whole-domain
min1/max1 region; B retains the scene's empty refinement-region list. Gravity,
viscosity, surface tension and initial velocity are zero. Therefore the
manufactured curved interface should remain at rest.

The baseline matrix runs six arms for 120 steps: A, B, frozen B, conditioning-off
B, gamma-only B and sharpening-only B. The final matched A/B runs 480 steps per
arm. The probe captures reset, every step through 10, then every 30 steps,
including native density, velocity, pressure, published heights, integrated
columns, topology, frame receipts, resolved options and compiled shader hashes.
All GPU runs are serial and acquire the repository-wide WebGPU lease.

Native fields are compared after **restricting both to identical width-8
control volumes**. Repeating a coarse average at fine diagnostic coordinates
is not a fine point field: the raw maximum density difference after a topology
change is not itself a physics error. The comparison JSON uses the stationary
fine reset as its reference. The final height comparison separately compares
both evolved arms at exactly eight seconds.

Interior metrics use x/z samples at least eight finest cells from each wall,
including derivative neighbours outside that interior. The final regression
also measures the maximum height error across the **whole tank**, so excluding
walls cannot conceal the boundary defect. Curvature is the second height
difference divided by h², in 1/m; the analytic x/z curvatures are 0.24/0.168.

| Measurement | A at 8 s | Original B at 2 s | Corrected B at 8 s |
| --- | ---: | ---: | ---: |
| Interior height error vs analytic, RMS mm | 0.04205 | 17.94145 | 0.00284 |
| x curvature error, RMS 1/m | 0.002718 | 3.20500 | 0.002766 |
| z curvature error, RMS 1/m | 0.002741 | 3.71973 | 0.002525 |
| Common width-8 density error vs fine reset, RMS | 1.87e-9 | 0.0080743 | 8.65e-8 |
| Interior column drift, RMS mm | 0 | 13.8678 | 0 |
| Final maximum speed, m/s | 2.30e-7 | 3.67e-5 | 3.60e-13 |
| Maximum density reflection error | 0 | 0.04405 | 0 |
| Relative mass change | 5.75e-10 | -2.685e-4 | 2.706e-8 |
| Accepted topology generation | 2 | 82 | 4 |

Corrected B uses 820 active simulation cells: four width-2 bricks, 64 width-4
bricks and 52 width-8 bricks. A uses 46,080 active fine cells. B's surface is
not obtained by forcing its simulation to the finest rung. Final whole-tank
analytic height error is 0.00316 mm RMS / 0.02884 mm maximum. The slightly
larger fine-grid analytic error includes the original cell-average/quadrature
bias; the table does not claim a general accuracy ranking of coarse and fine
fluid simulations.

## First mechanism: contradictory topology decisions

The initial adaptive field is a correct restriction of the fine field. Frozen
B retains exactly the same density through two seconds. Its surface nevertheless
has a 2.114 mm interior reconstruction error already at reset.

In live B, the first demotion occurs at step 3. The raw diagnostic columns
change by 5.155 mm RMS, but the common-volume density error remains approximately
4e-9: this first change is conservative loss of resolution, not lost mass.
The proposed coarser representation then requests refinement again. With
conditioning disabled, those topology changes continue to generation 82 while
the native common-volume state remains stable.

Refinement below the finest transition copies the parent density into its
children. The resulting vertically diffuse interface is eligible for numerical
sharpening. With sharpening enabled, common-volume density begins changing at
step 5, and the initially stationary bowl develops the large trough in the
figure. Gamma-only reproduces the stable conditioning-off result;
sharpening-only reproduces the sustained drift. Gamma diffusion is therefore
not the initiating mechanism in this fixture. The observed generated speeds
are small: substantial density redistribution need not be accompanied by
substantial physical fluid motion.

**Correction:** the generation-stamped surface demotion proof now evaluates
normal variation in its existing virtual restricted-density cache at the
proposed rung. It rejects a demotion that exceeds the configured coarse-first
curvature tolerance. This uses the existing physical tolerance and existing
proof pass; it adds no damping or simulation pass. The single-cell rung uses
halo normals, as its ordinary activity census does. Failure bit 16 records
this prospective curvature rejection.

With this correction alone, topology settles at generation 4, interior column
drift is zero, and common-volume density error stays at roundoff through the
two-second replay. The original 2.114 mm surface error remains, which separates
the physical fix from the presentation fixes.

## Second mechanism: treating volume averages as point heights

The previous volume-node interpolation was trilinear. For a broad quadratic
surface it produces piecewise slopes and recurring curvature jumps, and it
retains the width-dependent bias between a column average and its centre
height. This is measurable in the published scalar before meshing or lighting.

**Correction:** horizontal volume reconstruction now uses a quadratic B-spline
quasi-interpolant. A quadratic B-spline contributes variance 1/4 and the native
cell average contributes 1/12. Subtracting one sixth of the second difference
removes that combined quadratic bias. The resulting five-node weights reproduce
constant, affine and quadratic column heights and have a continuous first
derivative. Vertical interpolation retains the existing compact volume scalar.
The demotion proof uses the same reconstructed scalar.

This is a presentation reconstruction, not a density smoothing operation. It
does not write density, gamma, pressure or velocity. The five-node weights can
be signed; they are not used as mass-transport coefficients.

## Third and fourth mechanisms: mixed support and boundaries

Applying a finer density reconstruction to a coarser horizontal neighbour
invented a vertical profile at the seam. Coarse pages now restrict their
horizontal face-neighbour support to a common width before reconstructing the
surface. The next dyadic demotion's volume width covers that support under the
existing face-neighbour grading constraint. Fine simulation pages retain their
fine presentation path.

The wider interpolation stencil also exposed the old boundary continuation:
repeating the edge column imposes a flat contact angle on a curved or tilted
surface. The bounded-tank presentation apron now continues the column geometry
quadratically from the first/last three columns. SolidWorld still clips the
actual boundary; unbounded world leaves keep their existing world support.
Neither operation changes liquid mass or wall physics.

The original static curvature gate is unchanged. Its 25% analytic-curvature
budget now passes for fixed width 4, width 2/4 mixed support and frozen width 8.
Flat and tilted controls and 4,100 planar subcell/rung queries also pass.

## Validation and limits

The focused Dawn run passes all four tests, including the new 480-step live
stationary regression and the previously red coarse-curvature gate. Six
scene/proof unit tests pass. Targeted ESLint has no errors; it reports two
pre-existing unused-variable warnings in the resident shader generator.
Repository type checking still fails in other existing files; it reports no
errors in the new probe or regression test.

The full canonical gate was run on the unmodified staged snapshot in
`/tmp/fluid-bowl-baseline-20260907`, after the topology-only correction, and on
the final changes. The baseline already fails expansion density symmetry and
multiple existing per-lane timeouts, then exhausts its 180-second total budget.
No lane, threshold or timing ceiling has been relaxed. See the retained gate
receipts for final status; a focused success is not a claim that the canonical
gate is green.

These tests establish the stationary-bowl result and the manufactured surface
contracts. They do not establish resolution-independent dynamics for moving
liquid, or solve the separate transport/remapping issues discussed in the
existing adaptive-mass mechanism report.

## Reproduction and artifacts

```bash
node --import tsx tools/probe-stationary-bowl-ab-dawn.ts \
  --arm=adaptive --steps=480 --output=artifacts/stationary-bowl-2x/regression/adaptive
node --import tsx tools/probe-stationary-bowl-ab-dawn.ts \
  --arm=fixed1 --steps=480 --output=artifacts/stationary-bowl-2x/regression/fixed1

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx --test --test-concurrency=1 \
  tests/stationary-bowl-ab-dawn.test.ts \
  tests/sparse-cm12-native-volume-surface-dawn.test.ts \
  tests/sparse-cm12-surface-grid-imprint-dawn.test.ts

npm run test:dawn:sparse-cm12
```

The probe also accepts `--frozen=true`, `--conditioning=off|gamma-only|sharpen-only`,
and `--arm=fixed2|fixed4|fixed8`. Defaults preserve the actual UI scene profile.
Run `tools/analyze-stationary-bowl-ab.py` with Python, NumPy and Matplotlib after
the captures to regenerate the numerical comparison and figures.

- [Control matrix plot](../artifacts/stationary-bowl-2x/ab-diagnosis.png)
- [All numerical comparisons](../artifacts/stationary-bowl-2x/comparison.json)
- [Matched eight-second A/B](../artifacts/stationary-bowl-2x/regression/matched-comparison.json)
- [Original baseline gate](../artifacts/stationary-bowl-2x/receipts/baseline-regression.log)
- [Topology-only gate](../artifacts/stationary-bowl-2x/receipts/proof-regression.log)
- [Focused test receipt](../artifacts/stationary-bowl-2x/receipts/focused.log)
- [Final canonical gate](../artifacts/stationary-bowl-2x/receipts/final-regression.json)
