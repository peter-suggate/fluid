# Raster cell contours

Enable **Contour geometry** under primary rasterization, or load a scene with
`svoPrimary=mesh&svoMeshContours=1`. Keep **Smooth surface** off: that older
view-dependent reconstruction is separate. Contours default off.

**Contour inflation** expands each contoured cell before slicing it with the
original plane. It ranges from 0 to 0.50 cell widths **per side**, in steps of
0.01, and defaults to zero. For example, 0.25 changes a unit cell's clipping box
from `[0,1]` to `[-0.25,1.25]`; the plane stays in the same world-space position.
Use `svoMeshContourInflation=0.25` in a saved URL. Changing the slider rebuilds the
mesh without refitting the source or restarting the simulation; reset restores
the original bounds. Unsupported cube cells are not expanded.

Overlapping patches can cover indents, but this does not impose continuity or
guarantee a smooth surface. Positive inflation retains closed cell faces instead
of applying native neighbour-face rejection, increasing geometry and overdraw.
Inflation is baked into each triangle record so a pending rebuild cannot distort
the old front mesh; triangle bounds and back-face tests use the expanded geometry.
This option affects raster geometry only. Temporary traced primary fallback and
secondary visibility still use the original cell domains and may differ around
the expanded edges. Physics and the coverage-based lighting hierarchy are unchanged.

This is a native-resolution, one-sided variant of Laine and Karras §5.2/§7.2.
It clips each supported occupied cell by an outward-facing plane; the cell's
opposite support plane supplies the other side of the slab. It bounds solid
volume and does not implement hierarchical surface slabs or dual contouring.

The authored dry-scene voxelizer fits the contour before source information is
lost. It splits a partial cell into 4³ subcells, discards only subcells proved
empty by a primitive distance lower bound or the authoritative render-terrain
height samples, and bounds all remaining subcell boxes. The bound includes
all candidate primitives, not just the material winner. Ellipsoids use an
explicit conservative lower bound. The fit uses the decoded oct8 normal and
rounds the support outward, with an additional offset step for vertex rounding.
A cell whose bound reaches the whole cube writes no contour.
If every subcell proves empty, a separate producer-only result clears the cell's
occupancy; it must not be confused with the zero code that retains a cube.

The [analytic Dawn reproduction](svo-cell-contour-analytic-reproduction.md)
documents remaining surface-offset and neighbour-continuity failures. Contours
are still experimental; the empty-result correction does not fix those failures.

The high byte of packed `f16-unorm8` scene geometry stores this offset; the
existing identity contains its normal. No extra per-cell allocation is needed.
Full-precision/solver-owned payloads do not acquire this attachment. The render
source selects the packed format when contours are requested. Generic payload
writers clear the attachment, and the banded encoder preserves it. Edits refit
it within the existing scene publication transaction.

Density-threshold foliage and canonical voxel-only solids retain cubes. Terrain
subcells intersecting fill/clear overlays are bounded conservatively as boxes;
distant edits no longer disable fitting across the whole terrain. Full cells, degenerate normals and inconclusive fits also
retain cubes. Terrain fitting, coverage and normals share a continuous bilinear reconstruction
of the published centre-height samples. Contour-enabled native detail now builds
this renderer-owned heightfield too. Physics still uses its canonical voxel field. Source detail not represented
by the occupied cell set cannot be recovered by this feature.

The mesh count/emit passes triangulate the clipped cube, including its new cap.
Covered faces are omitted; faces against a contoured neighbour are clipped to
the portion outside that neighbour. Triangles use tagged 32-byte mesh records with three packed local vertices. Cube
neighbours expose their faces against clipped cells; closed clipped-cell faces
cover resolution transitions. Existing range allocation, rollback, incremental
neighbour updates and replacement publication remain in use. A contour toggle
forces replacement; contours suspend coarse mesh LOD. The culler uses conservative
cell bounds and geometric back-face tests for triangles. Four strip vertices draw one triangle (the fourth
repeats the third), with ordinary reverse-Z hardware depth and no fragment depth.

Temporary primary ray fallback and exact secondary visibility intersect the same
slab. The radiance/opacity hierarchy remains coverage-based, and fluid collision
and water reconstruction remain unchanged. This is not a change to simulation
resolution. Contour shading currently uses geometric triangle normals, so it
also avoids asking the existing axis-face metadata codec to represent a separate
arbitrary geometric normal.

Construction has extra field evaluations and polygon clipping. At fixed voxel
resolution, contour cells usually require more records and reduce greedy merging.
The possible performance win from using a coarser render representation is not
implemented. Do not infer a production frame-rate improvement from this prototype.

Validation commands:

```sh
node --import tsx --test tests/svo-cell-contour.test.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  node --import tsx --test --test-concurrency=1 \
  tests/svo-cell-contour-dawn.test.ts tests/svo-surface-mesh-scheduler-dawn.test.ts \
  tests/svo-filter-controls-dawn.test.ts
npm run test:dawn:sparse-cm12
```

The garden benchmark accepts `FLUID_SVO_DRY_FRAME_MESH_CONTOURS=1` alongside
`FLUID_SVO_DRY_FRAME_SURFACE_MESH=1` and raster-primary traversal. It reports a
GPU attachment census and total world startup time, separately from warm frame
timings. Run it through `tools/run-webgpu-exclusive.ts`; never run it alongside
a browser or another Dawn process. Compare against the same flags with contours
set to zero, and confirm the receipt says ready with no fallback or overflow.
Set `FLUID_SVO_DRY_FRAME_CONTOUR_INFLATION=0.25` to exercise inflation. The Dawn
scheduler test verifies expanded vertices, unchanged plane depth, replacement
publication, cache reuse and restoration to zero inflation.

## Initial validation — 2026-09-10

Apple M1 Max/Metal, `hero-garden-hose`, 400×240, split raster primary,
GI/shadows/AO off. Four warm frame samples per run are smoke measurements,
not a performance acceptance test. Captures, receipts and logs are in
`artifacts/svo-cell-contours-2026-09-10/`.

| Run | Contoured cells | Required mesh records | Drawn records | Wall median |
| --- | ---: | ---: | ---: | ---: |
| Native, off | 0 | 274,885 | 44,221 | 5.85 ms |
| Native, on | 23,381 | 276,826 | 118,956 | 4.67 ms |
| Refinement 1, on | 193,020 | 1,433,912 | 737,626 | 3.80 ms |

All three receipts report mesh ready, zero overflow and no mesh fallback reason.
The legacy `*Quads` diagnostic fields count records, including contour triangles.
The off run also stores unused coarse LOD meshes, so required-record totals are
not an equal-LOD comparison. Native terrain remains canonical cubes; refinement
1 supplies the heightfield render source. Startup wall times were 2.50, 2.70 and
2.47 seconds respectively; compilation and run-to-run noise prevent attributing
these differences solely to contour fitting. No speedup claim follows from these
short runs.

Seven CPU tests passed (one GPU-only test skipped in that invocation); the three
targeted Dawn tests passed, covering conservative fitting, closed clipped-cell
geometry, shader variants, and mesh replacement/dirty updates/overflow rollback.
The canonical CM12 gate was attempted but did not pass: symmetric expansion hit
a D4 density assertion, topology-page-budget timed out, and eleven lanes were
blocked by another task's GPU lease. Four lanes passed. No thresholds were
changed. Full TypeScript checking also reports errors outside this change's files.


## Terrain source correction

Terrain previously combined piecewise-constant column heights with normals from
neighbouring columns. Its vertical-only coverage test also misclassified cells
crossed by steep slopes as full or empty. At native detail it received canonical
voxel boxes instead of the render heightfield.

The render source now uses the same centre-aligned bilinear field for heights,
gradients and conservative min/max bounds. Bounds split at sample knots and check
patch corners; coverage marks all height-range crossings partial. The refinement
hierarchy includes a sample's influence across brick boundaries. This changes
render reconstruction only, including the refined terrain path without contours;
it does not resample or modify the simulation's collision field.

Validation: `tests/svo-terrain-contour-dawn.test.ts` covers flat terrain, gentle and
steep ramps, bilinear curvature, knot crossings, distant edits and all three
producer payload encodings. `tests/svo-terrain-contour.test.ts` checks refinement
across a brick boundary. The native garden-hose smoke run at inflation 0.25
reported 73,599 contour cells, 460,898 drawn records, mesh ready, no overflow and
no mesh fallback. This compares with 185,512 drawn records before the terrain
correction; additional contoured terrain raises geometry and overdraw cost.
Startup was about 2.6 seconds in this short run, not a timing acceptance result.
Receipts and capture: `artifacts/svo-terrain-contours/`.
The refinement-1 run also completed without mesh overflow or fallback, with
263,034 contour cells and 1,530,191 drawn records at inflation 0.25. Eleven
relevant CPU tests and the terrain Dawn test passed. The CM12 gate did not pass:
it encountered a symmetry assertion and lane timeouts, then exhausted its
180-second suite budget. No timing ceilings were changed. Repository-wide
TypeScript checking still reports errors outside these changes.
