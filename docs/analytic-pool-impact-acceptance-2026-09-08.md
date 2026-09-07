# Analytic pool-impact surface acceptance

The acceptance uses the actual `coarse-first-pool-impact-quarter` and
`coarse-first-pool-impact-half` scene definitions, including their suspended
liquid sphere. It reads the original URL region through the production parser:
`regions=0_0_0_25_66.6667_100_8_8`.

| Catalog scene | Finest lattice | Pool height | Sphere center | Radius | Parsed region bounds |
| --- | --- | --- | --- | --- | --- |
| Quarter | 32 × 24 × 32 | 0.4 m | (0, 0.9125, 0) m | 0.25 m | (-0.8, 0, -0.8) to (-0.4, 0.8, 0.8) m |
| Half | 64 × 48 × 64 | 0.8 m | (0, 1.825, 0) m | 0.5 m | (-1.6, 0, -1.6) to (-0.8, 2.0, 1.6) m |

Both use 0.05 m finest cells. The half-scene region reaches 2.0 m because the
literal rounded percentage is slightly above the parser's lattice tolerance,
so its outward snap grows one 0.4 m cell. This test preserves that behavior;
it does not quietly substitute an idealized two-thirds region.

## What is measured

`tests/pool-impact-analytic-oracle.test.ts` checks catalog dimensions and region
coordinates, then checks the compiled retained field against independent
plane/sphere equations at arbitrary physical points. One thousand sphere
points per scene measure the zero surface and normals; four thousand offset
points measure the fixed-width diffuse density. Canonical float32 coefficients
have an explicit 0.2 micrometre zero-surface allowance.

`tests/sparse-cm12-pool-impact-analytic-dawn.test.ts` starts each actual scene
with the original region, then runs two paused editing cycles through global
minmax1, minmax4 and minmax2 partitions, returning to the original local
minmax8 region after each change. It checks:

- Native cell widths actually change and the original region contains native
  width-eight cells alongside other widths.
- Native means equal restrictions of the fine retained-density integral
  receipt, total density amount is retained, and no simulation step or velocity
  is introduced by a paused edit.
- Every vertical column retains its pool crossing and both sphere crossings.
- Published interface samples equal the independent analytic quadratic/plane
  defining field within binary16 storage and float32 arithmetic precision.
- A fresh run of the production classifier, scan and emitter produces the
  pool and ball at startup and after each complete edit cycle. The test measures
  pool area, planarity, upward normals, sphere vertex and triangle-interior
  distance, sphere normals and nonmanifold edges. It cannot pass by retaining
  an old renderer mesh.
- Resuming one physical step changes velocity and published scalar values.

The implicit sphere is exact up to coefficient precision. Its published
cell-center samples and linear triangles have finite discretization error.
The radial geometry budgets are declared from finest spacing and physical
radius before any candidate is measured; accepted native coarse widths never
enter those budgets. Pool planarity has a 2 micrometre allowance. Sphere
vertex and triangle-interior errors are reported separately so exact sphere
vertices cannot conceal chord error. Native density amount refers to the
retained diffuse field, not the sharp sphere volume.

## Analytic ladder receipt

The ten independent plane, curved-surface and sharp-edge fixtures now measure
actual zero-set crossings along analytic normals after one hundred mixed
split/merge cycles. Smooth normal measurements exclude sharp branch ties,
which retain their exact half-density residual check.

The CPU receipt `artifacts/implicit-density/ladder-analytic-surface.json` and
its PNG show maximum zero-set displacement `9.40e-14`, maximum normal-vector
error `5.08e-15`, and maximum half-density residual `1.12e-14`.
These are standalone exact-family algebra results, not evidence that the
production renderer or dynamics have passed.

## Production result

The actual quarter and half scenes both passed all thirteen paused partitions
and the physical resume check in the exclusive Dawn run recorded in
`/tmp/fluid-retained-pool-production-6.log` (two tests, 64.1 seconds total).
Published scalar samples were identical across paused partitions; every pool
and upper/lower sphere crossing remained present. Native means agreed with
restrictions of the same retained density to below `8.4e-8`.

Fresh production meshes at startup and after both editing cycles had no
interior open edges or nonmanifold edges. The worst measurements across all
three captures of each scene were:

| Metric | Quarter | Half |
| --- | ---: | ---: |
| Pool height error | 5.96e-9 m | 1.19e-8 m |
| Sphere vertex radial error | 3.773 mm | 1.879 mm |
| Whole-triangle radial error | 4.779 mm | 2.737 mm |
| Sphere normal-vector error | 1.92e-4 | 2.35e-4 |

The whole-triangle measurement finds the closest point anywhere on each
triangle, not merely its centroid. Curved adaptive fans project their center
onto the represented scalar and subdivide when interior error exceeds 1/32 of
a finest cell. Quadratic normal reconstruction selects a one-sided published
stencil near a support boundary, avoiding an artificial air value above the
quarter scene's sphere. Exact planar fans keep their coarse triangulation.

`artifacts/retained-density-production/analytic-mesh-sections.png` plots the
saved GPU triangles intersected with the physical z=0 plane against independent
plane and circle references. These are finite triangles at 0.05 m sampling;
the retained implicit zero set is exact up to coefficient precision. The
broader Dawn matrix and browser capture are separate verification work.

## Commands

CPU checks:

```sh
node --import tsx --test tests/implicit-density-field.test.ts tests/pool-impact-analytic-oracle.test.ts
```

Production acceptance (exclusive Dawn lease; do not run beside a browser):

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
POOL_ANALYTIC_OUTPUT=artifacts/pool-impact-analytic \
node --import tsx --test --test-concurrency=1 tests/sparse-cm12-pool-impact-analytic-dawn.test.ts
```

`POOL_ANALYTIC_SCENE` can select one exact catalog scene while diagnosing a
failure. Output includes actual emitted `mesh.bin`, published `phi.bin` and
metric receipts. The original retained-field-partition test remains an
independent negative baseline; this acceptance does not replace or weaken it.
