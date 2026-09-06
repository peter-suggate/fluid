# Symmetric expansion: coarse-first physics A/B

The reported shape difference is present in the accepted simulation density,
independently of the surface renderer. The runtime curvature estimator was
silently losing its signal. Restoring that signal and making its stencil
consistent across cell widths recovers the collapse without changing energy,
curvature, prediction, persistence, or mass thresholds.

## Reproduction

The fixture uses the native `sparse-cm12-symmetric-expansion` scene document,
balanced quality and its production method profile: 32 × 16 × 32 finest cells,
0.05 m spacing, a centred 0.8 × 0.4 × 0.8 m liquid body, and 0.256 m³ initial
liquid. Both arms run 13 encoded steps of 1/30 s, reaching the screenshot's
0.4333 s. The reference adds a full-domain maximum cell size of 1³ to the same
Sparse CM12 solver. It is not the separate Uniform solver.

The probe awaits deferred topology work and asserts each actual encoded step.
It captures accepted density, velocity, pressure, divergence, open fractions,
activity, integrated column heights, and stats at reset and after every step.
Run arms sequentially, with Fluid browser tabs unloaded, under the repository
WebGPU lease:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
SYMMETRIC_MAX_CELL=1 SYMMETRIC_OUTPUT=artifacts/symmetric-coarse-first-ab/fine \
node --import tsx tools/probe-symmetric-coarse-first-ab-dawn.ts

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
SYMMETRIC_OUTPUT=artifacts/symmetric-coarse-first-ab/corrected \
node --import tsx tools/probe-symmetric-coarse-first-ab-dawn.ts

uv run --with numpy --with matplotlib python tools/analyze-symmetric-coarse-first-ab.py
```

The plot also requires `baseline`, captured by the same probe with the original
curvature functions from commit `d5cb3502`. `accepted-owner` preserves the
intermediate experiment that corrected only the owner lookup. The output
directory, step count, timestep, maximum cell size and method overrides are
configurable through the `SYMMETRIC_*` environment variables in the probe.

## Causal chain

1. `coarseFirstDensity` called `tracerCellAt`. That helper reads a workgroup
   transport-directory cache. Tracer entry points stage the cache; the activity
   entry point does not. Around the default origin it reads zero-initialized
   leaves; elsewhere it can query the default topology slot rather than the
   accepted slot. Neither result is valid curvature evidence.
2. CPU initialization correctly preserves the box corners at B8, but at step one
   the runtime reports curvature floor B1 for all four central wet bricks. By
   step three it demotes the collapsing body to B4, then to B2 by step six.
   Mass conservation and D4 symmetry still hold while the collapse changes.
3. Direct accepted-owner lookup restores the missing curvature. Alone, it also
   exposes a representation defect: adjacent coarse and fine cell averages are
   treated as point samples. Even a flat surface has different average fills
   at different widths, creating fictitious horizontal gradients. The unchanged
   mini32 top-corner test catches this unnecessary B2-to-B4 refinement.
4. Each normal now chooses a common dyadic support width from its accepted
   neighbourhood, aligns the stencil, and volume-restricts finer donors to that
   width before differencing. Missing sparse liquid is air; a solid cut-cell
   stencil is unavailable liquid-curvature evidence and remains governed by the
   existing solid geometry and coupling floors. Solid boundaries must not be
   mistaken for liquid-air curvature.

The solid-stencil distinction matters in the rigid-body gate. Without it,
restored curvature sees moving solid interfaces as surface features and changes
the refinement/retirement trajectory. The intermediate run lost 1.0182 finest
cell equivalents, crossing the unchanged 1-cell bound. With the proper stencil
validity rule it loses 0.6151 cells and passes. Transfer receipts remain near
roundoff; the large intermediate loss increments coincide with explicitly
receipted sparse residue retirement.

## Physical comparison

At 0.4333 s, against the 1³ reference:

| Measurement | Original coarse-first | Corrected estimator |
| --- | ---: | ---: |
| Density relative L1 difference | 56.68% | 1.03% |
| Integrated height relative L1 difference | 23.74% | 0.70% |
| Integrated height RMS difference | 35.16 mm | 1.88 mm |

The direct-owner-only experiment reaches 2.60% density and 2.06% height L1
difference, but has a 7.95% transient density difference at step five. The
equal-volume stencil brings that checkpoint to 0.25%. These comparisons measure
agreement with the requested numerical reference, not error against an exact
continuum solution. They do not imply identical fields or universal accuracy.

![Accepted density and height comparison](../artifacts/symmetric-coarse-first-ab/comparison.png)

![Error through the collapse](../artifacts/symmetric-coarse-first-ab/error-history.png)

## Regression coverage

`tests/sparse-cm12-coarse-first-expansion-dawn.test.ts` executes both production
arms in isolated sequential processes. It checks exact initial-density equality,
the four box-corner curvature floors, finite fields, mass, and density/height
agreement at steps 3, 5, 8 and 13. The physical envelope is 2.5% density L1 and
2% height L1 at each checkpoint, plus 0.1 finest cell (5 mm) height RMS at the
final checkpoint. The original run fails these checks despite conserving mass.

`tests/sparse-cm12-coarse-first-curvature-dawn.test.ts` runs the production WGSL
estimator on exact planar control-volume averages across 8:4:2:1 widths, for all
three normal axes. It also checks that cut-solid stencils do not supply liquid
normals. The unchanged mini32 corner test checks that fixing curvature does not
reintroduce first-step overrefinement. Both new tests are included in
`npm run test:dawn:sparse-cm12:coarse-first`.

The still-pool, impact-anticipation and moving-surface coarsening controls remain
required: the fix must preserve distant B1 water and allow disturbed surfaces
to become coarse again. No existing physical or performance ceiling is raised.

The final focused run passes all 11 tests. The still pool retains all 128 sampled
far-side B1 surface bricks and has zero measured mass drift. The impact run
retains its anticipation and distant-coarse checks, with 0.00411% mass drift;
the disturbed pool returns to coarse coverage while moving, with 0.0213% drift.
Both CPU coarse-first tests and lint of the new TypeScript files pass. The
repository-wide TypeScript check reports 50 errors outside the changed files.
