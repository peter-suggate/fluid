# Analytic contour reproduction

The reported outliers and mismatched cell contours reproduce without garden
terrain, neighbouring primitives, or the raster lighting pass. The new Dawn test
sweeps a translated 24³ lattice through a sphere, ellipsoid, cylinder, capsule and
box, each unrotated and rotated. It runs the production contour fitter, oct8
normal codec, and packed cap polygonizer. Independent CPU shape equations check
cap vertices, retained cube corners, sampled solid containment, and the gap
between adjacent planes on their shared face.

This is a geometry-level GPU reproduction, not an image comparison of the full
raster renderer. The test supplies analytic fields and finite-difference normals;
it does not exercise the production scene candidate builder or material winner.
Errors below are in cell widths. Sphere/capsule/cylinder/box residuals are signed
distances; ellipsoid residuals are a conservative distance lower bound, not exact
Euclidean distance. Shared-face gaps measure separation along a face coordinate
where both planes intersect that face, before neighbour face coverage closes it.

## Findings

1. **Empty and full-cube results were conflated.** Coverage can retain a cell
   outside the actual solid. When every fitting subcell proves empty, `!found`
   returned zero, which means “retain the cube” to the consumer. That resurrected
   a whole cube. The producer now returns a distinct empty result, consumed before
   publication to clear fraction, identity and occupancy together. No reserved
   empty marker is stored in scene geometry. This fix is shared across shapes.
2. **The fitted support is a conservative occupied-volume bound.** Bounding whole
   4³ subcells adds a lattice-dependent offset. It is not an estimate of the
   analytic zero surface. Even valid contours therefore stand outside the shape.
3. **Independent planes have no continuity constraint.** There are no shared edge
   intersections or shared vertices between cells. Better individual fits cannot
   guarantee that adjacent planes meet. The mesh closes the difference with side
   faces, which makes steps rather than a connected smooth surface.
4. **Sharp features need more than one plane.** Box corners and cylinder rims
   expose a representational limitation separate from smooth-surface fitting.

| Unrotated shape | Worst cube residual before | After empty fix | Worst cap residual | Shared-face gap |
| --- | ---: | ---: | ---: | ---: |
| Sphere | 1.692 | 0.425 | 0.483 | 0.262 |
| Ellipsoid | 1.569 | 0.384 | 0.439 | 0.307 |
| Cylinder | 1.700 | 0.376 | 0.910 | 0.660 |
| Capsule | 1.697 | 0.409 | 0.505 | 0.181 |
| Box | 1.574 | 0.298 | 0.781 | 0.374 |

The cap errors and gaps are unchanged by the empty-result fix. Of 1,406 partial
sphere cells, 103 now return empty rather than a cube. Sampled analytic solid
points remain contained. This fixes the extreme outliers, not all protrusion.

## Rejected experiments

Per-primitive support functions were removed. An experimental shared adaptive
fit at 1/32-cell subdivision reduced smooth cap residuals to 0.11–0.15 cells,
but rotated smooth shapes still had shared-face gaps up to 0.264 cells. A garden
smoke run increased startup from approximately 2.7 to 5.1 seconds. That experiment
was also removed: it was a more expensive partial improvement, not a continuity
fix. The renderer retains the original 4³ fit plus the empty-result correction.

The next reconstruction change needs a shared-boundary representation: derive
edge intersections and normals from the common field, give neighbouring cells
shared constraints/vertices, then generate a connected triangle surface. That is
compatible with raster primary but is a larger change than tightening independent
Laine-style slabs. It must explicitly handle sharp features and sparse LOD joins.

## Running the reproduction

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  node --import tsx --test tests/svo-cell-contour-analytic-dawn.test.ts
```

Containment and the empty-result regression are active assertions. The stricter
geometric quality subtest is an explicit known-failing TODO, not a claim of a
quality pass. Set `FLUID_SVO_CONTOUR_ENFORCE_QUALITY=1` to make it fail normally.
Its 0.18-cell smooth error/seam target is provisional, not a user-specified
acceptance requirement. Full per-shape numbers and rejected-experiment logs are
in `artifacts/svo-cell-contours-analytic/`.

Targeted producer/polygonizer, scheduler and shader-variant Dawn checks passed;
the quality TODO remains failing as described above. The broader CM12 gate was
attempted again and did not pass: a symmetry assertion, a page-budget timeout,
and GPU lease conflicts prevented a clean result. Full TypeScript checking still
reports errors outside the modified files. No gate thresholds were changed.
