# Smooth, watertight surfaces across the set and its backdrop

Status 2026-10-02: implemented, uncommitted. Verified headless on
`hero-garden-hose-x10` (Dawn/Metal, 1600x920); not yet looked at in the app.
A performance pass followed (see below): the voxel path is unchanged from HEAD
and smooth is within about 1 ms of it at the app's cone scale.

## What "Smooth surface" now draws

With the dual-marching-cubes mesher (`dry.meshFilterNormals.w == 3`) every
surface in the frame is fitted from a continuous field, never from voxel
occupancy:

| Region | Field | Drawn as |
| --- | --- | --- |
| Set terrain | render terrain height field (`rtSurface`) | dual mesh |
| Set scenery | primitive SDFs | dual mesh |
| Backdrop, stored rings | `backdropDetailField` | dual mesh, same octree |
| Backdrop, past the rings | `backdropTerrainSurfaceAt` | analytic height-field walk |

One octree holds the set and the rings, so ring boundaries and the footprint
seam are ordinary mixed-resolution dual cells: no skirts, no overlapping
patches. Grading up to 4:1 is meshed; anything steeper raises
`meshDmcUnsupported()` and the frame falls back to the trace.

## Pieces

- `backdropDetailField(world, overlap)` (`backdrop-detail.ts`): ground
  `(y - h) / sqrt(1 + |grad h|^2)` unioned with the scatter SDFs, intersected
  with the outside of the set footprint. Same seeds, placement and seam height
  as the voxel backdrop. `sampleBackdropDetail` derives occupancy from this
  field under DMC, so what shadows is what is drawn.
- Classifier `dualGrid` option: wider vertical reach and one apron brick past
  the outer ring, so the last ring has neighbours to close against.
- `dcField` (`webgpu-sparse-scene-proxies.ts`) unions the backdrop field into
  the set's field. A renderer-only world with DMC now always builds the render
  terrain; before, a wet depth-0 sidecar fitted the solver's stepped voxel SDF
  and the set's own terrain came out shattered.
- `backdropTerrainSmoothTrace` (`backdrop-terrain-tiles.ts`): the tile walk
  with a per-column bracketed secant on the analytic height. Used for the
  background pass and for backdrop shadow rays when smooth. It overlaps the
  mesh apron and is depth-composited, so the join needs no stitching.
- Mesh vertex normals are stored by the mesher: each dual triangle carries its
  three vertices' fitted normals in the high halves of its origin words. A
  vertex takes the normal of the solid leaf on its dual edge.

## Deviation from the handoff

The handoff proposed cached raster height-field tiles for the far ground. The
analytic walk was used instead: it is the exact surface the rings are fitted
to, needs no boundary-vertex matching against the mesh, adds no geometry
memory, and reuses the tile table already bound. If the walk's cost shows up
at grazing angles, raster tiles remain the next step; the worst test ray used
318 of the 384 work budget.

## Two defects found on x10 and fixed

1. **Upright "shield" plates on the footprint edge.** Thin vertical triangles
   in the clip plane (a quarter cell inside the footprint), 3 to 14 cells above
   the ground. The backdrop field returned `-outside` one overlap inside the
   footprint, which above the ground is a jump from a quarter cell to the
   ground clearance. The field is now `max(solid, -outside)` out to four cells (16 overlaps)
   inside and the plates are gone from the x10 frame. The isolated fit fixture
   does not reproduce them with the old field, so which consumer of that jump
   drew them (fit or stored occupancy) is not pinned.
2. **Dark band beside the footprint and dark flecks on bush tops.** A dual
   vertex can lie in the air cell of its edge. The vertex shader read that
   cell's identity word (`0xffff0000`) as a normal. The normal now comes from the
   solid end of the vertex's dual edge, stored in the triangle record at build.

## Performance (x10, 1600x920, M1 Max, GPU pass timestamps, median of 9)

Two cameras: the preset one, and a low wide one that sees to the horizon.
"HEAD" is 98394421; "voxel" and "smooth" are this tree.

| Frame, ms | preset | low |
| --- | --- | --- |
| Cone scale 0.5 (app default): HEAD voxel | 11.34 | 9.50 |
| Cone scale 0.5: voxel | 11.47 | 9.76 |
| Cone scale 0.5: smooth, before the pass | 13.30 | 12.45 |
| Cone scale 0.5: smooth | 12.39 | 10.75 |
| Cone scale 1: HEAD voxel | 24.71 | 17.17 |
| Cone scale 1: voxel | 20.58 | 14.68 |
| Cone scale 1: smooth, before the pass | 27.7 | 24.8 |
| Cone scale 1: smooth | 27.07 | 20.38 |
| Lighting off, cone scale 0.5: voxel | 4.46 | 4.13 |
| Lighting off, cone scale 0.5: smooth | 4.39 | 4.65 |

The voxel path has not regressed from HEAD. Smooth is still slower than
voxel: +0.9 / +1.0 ms at cone scale 0.5, +6.5 / +5.7 ms at cone scale 1.

What the pass changed:

- **Per-tile continuous bounds for the smooth walk.** The walk used the
  stepped tile bounds widened by a global Lipschitz margin, so it entered
  nearly every tile. Tile word 3 now holds the continuous ground's own low and
  high (16-bit fractions of the stepped band widened by two cells, rounded
  outward, from `backdropTaylorHeightBounds`), and `ceilingWord + 2` holds the
  largest excess of the smooth level maximum over the stepped one, in cells.
  Low camera, lighting off, cone scale 1: 8.6 to 5.2 ms.
- **One secant step** per bracketed column instead of three.
- **Vertex normals stored in the mesh.** The vertex shader descended the octree
  up to three times per vertex to find its normal. Low camera: -0.65 ms.
- **Half-rate lighting reuse for smooth receivers.** A curved mesh failed the
  receiver test tuned for voxel faces (triangle normals within 26 degrees,
  eighth-power normal guide) and fell to full-rate cones. The crease test is
  now 60 degrees and the guide for reconstructed receivers is
  `(0.5 + 0.5 dot)^4`. Mean error against full-rate lighting is 0.0138 /
  0.0120 (voxel path: 0.0192 / 0.0121). -0.8 / -1.4 ms.

Tried and dropped: an inward-first single normal lookup (faceted, per-triangle
offsets disagree at shared vertices); crossing the stored rings' square in one
step in the walk (no gain).

What is left of the gap:

- Cone scale 1 has no half-rate pass, so every smooth pixel traces the sun
  cone that a voxel face reads from the voxel light cache. Smooth receivers
  stay off that cache on purpose: it is flat per voxel face.
- Cone scale 0.5, preset camera: AO and GI +0.4 ms, sun +0.3 ms.
- Cone scale 0.5, low camera: the far-ground walk +0.5 ms (an analytic height
  per column against a stored column top), AO and GI +0.3 ms.

## Other measurements

| | value |
| --- | --- |
| Triangles built | 920,459 |
| Triangles drawn, preset / low camera | 127,444 / 310,729 |
| Mesh build | ~350 ms |
| Allocated | ~379.5 MB |
| Fallback | none |

## Tests

`tests/svo-dual-marching-cubes-dawn.test.ts` gained a ring-corner fixture
(zero open interior edges, consistent winding, vertices on the ground) and a
168-ray check of the smooth ground walk against the full tile table.

## Open

- Refinement depth 3 without refined rings exceeds 4:1 and falls back.
- Cone fan-out lighting still walks the stepped far ground.
- Rebuild scope under live edits is unmeasured.
- The pond rim still shows ridged walls and the foreground has spiky tufts;
  both are set content, not backdrop, and were not investigated.
