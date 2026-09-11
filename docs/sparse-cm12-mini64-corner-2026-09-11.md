# Mini64 Dawn front trajectory — 2026-09-11

The default mini64 UI had visible surface holes, reproduced and repaired below. Dawn measurements show that liquid reaches the far corner by 0.333 s with zero world allocation and insertion failures. The repair fixes macro/fine surface meshing; it does not change simulation transport or allocation.

## Reproduction

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal node --import tsx --test tests/sparse-cm12-mini64-corner-dawn.test.ts
```

This uses the catalog `minimal-power-dam-break-64` scene with balanced Sparse CM12 production defaults, no method overrides, and the paper timestep of 1/30 s. Only the authored duration is extended to one second. Frames are submitted in pairs and sampled after each pair. A preceding 60-step run sampled every six steps also reached the corner and passed. GPU lifetime is explicitly retained because the first unretained Dawn process crashed during startup/early advancement.

The initial reservoir is 0.5 × 0.736 × 0.5 m in a 0.8 m cube, at 0.0125 m finest-cell width. The initial x/z faces are at 0.5 m from the near walls. Coordinates below use the near tank wall as zero.

## Measurements

The axis front is the furthest cell with density >0.05. The diagonal samples x=z in the bottom eight cell layers. Positions report the outer edge of that cell, with one-cell uncertainty; this is not an interpolated zero-density tip. The far-corner volume census uses x,z ≥60 over all heights. Published corner samples count negative level-set samples in the x=z=7 brick column.

| Time (s) | Axis front (m) | Diagonal x=z coordinate (m) | Published corner samples | Allocation failures |
|---:|---:|---:|---:|---:|
| 0.000 | 0.5000 | 0.5000 | 0 | 0 |
| 0.067 | 0.5250 | 0.5125 | 0 | 0 |
| 0.133 | 0.6375 | 0.5500 | 0 | 0 |
| 0.200 | 0.7875 | 0.6125 | 0 | 0 |
| 0.267 | 0.8000 | 0.7000 | 0 | 0 |
| 0.333 | 0.8000 | 0.8000 | 267 | 0 |
| 0.400 | 0.8000 | 0.8000 | 1350 | 0 |

Both axis-wall cells are reached in (0.200, 0.267] s. The far-corner cell is reached in (0.267, 0.333] s, also at the stronger density >0.5 threshold. At 0.333 s the corner volume census contains 47.85 fine-cell volumes, about 0.0935 litres, and rendering publishes 267 liquid samples in the far-corner brick column. The corner remains represented through 1 s.

All sampled allocation, insertion, simulation-fault and failed-commit checks pass. The maximum sampled live directory occupancy is 491/785 leaves. Final represented mass loss is 0.00649% at one second.

## Analytic comparison

[Ritter's dry-bed solution](https://wolf.hece.uliege.be/tutorials/wolfgpu_ritter.html) gives

`x_front(t) = x_dam + 2 sqrt(g H) t`

and within the rarefaction fan

`h(x,t) = (2 sqrt(g H) - (x-x_dam)/t)^2 / (9g)`.

Here H=0.736 m, g=9.80665 m/s², and x_dam=0.5 m. The ideal unbounded front speed is 5.3732 m/s, giving an axis-wall arrival of 0.05583 s. At the first nonzero sampled time (0.0667 s), Ritter's unbounded front is at 0.8582 m, already beyond the tank wall; Dawn is at 0.525 m. The measured axis arrival is roughly 3.6–4.8 times later than this reference time.

This is a substantial difference from Ritter, but **not an analytic correctness verdict for mini64**. Ritter assumes one-dimensional hydrostatic shallow-water motion with a semi-infinite reservoir, while mini64 releases a tall finite column in two horizontal directions and soon encounters reflecting walls. H is larger than the initial reservoir width and more than twice the dry runout distance. Ritter has no corner coordinate; applying its speed diagonally would be a heuristic, not an exact 3D solution. Its unbounded solution should not be treated as the post-impact solution by clipping its front at the wall.

The capture establishes delayed arrival relative to this idealized reference, not a stopped front or page-allocation failure. A numerical-accuracy diagnosis would need a matched one-dimensional shallow-water fixture or a converged 3D reference for this exact geometry.

## Artifacts and checks

- Raw trajectory and geometry: `benchmarks/results/sparse-cm12-mini64-corner-2026-09-11.json`.
- Regression: `tests/sparse-cm12-mini64-corner-dawn.test.ts`; checks real liquid and renderer publication in the corner by step 12, finite nonnegative density, and zero allocator/simulation faults through step 30.
- Dawn test passed (30 steps; about 38 seconds including construction).
- Repository typecheck reports existing errors in other files; this test's introduced type errors were corrected.
- No solver implementation, timing ceiling, or accepted behavior baseline was changed. Renderer repair validation is recorded below.

## UI surface defect reproduced and repaired

The front-arrival result above did not establish that the UI was correct. A fresh browser run with Sparse CM12 / Coarse first and no overrides exposed holes in the upper remnant of the column at step 10 (0.3333 s). The first Dawn check counted published liquid but never inspected mesh closure, so it missed this failure.

A mesh capture at that step found 470 interior open edges above 0.1 m and six non-manifold edges. The defect also occurred with the unsimplified contour and with mesh refinement ×1, ×2 and ×4. It is a native macro/fine contour mismatch, not a page-allocation failure: the coarse face's interpolated contour did not share the fine neighbour's sample positions and values.

The renderer now samples native pages by interpolation at intermediate lattice points and meshes the surface-crossing portions of macro cells on the shared unit lattice. Its boundary census includes intermediate face samples so a thin crossing between coarse corners is not discarded. Homogeneous interior cells retain sparse native storage; no simulation pages or solver rungs are added.

On the captured failure, the revised mesh has zero non-floor interior open edges and zero non-manifold edges. The remaining 853 open edges lie on the existing floor clipping plane. Geometry rises from roughly 63,058 to 66,926 triangles (about 6%). The new `mini64 moving native macro contours remain closed` regression in `tests/sparse-cm12-adaptive-mesh-dawn.test.ts` checks the actual mesh at step 10, including the unsimplified path and UI refinements ×2 and ×4.

The brief early clock pause was observed while required topology preparation completed; advancing resumed. This repair addresses the reproduced surface holes and does not remove required topology/shader preparation.

Repair validation:

- Existing mixed-resolution mesh fixtures and the new production mini64 closure regression pass. Mini64 reports zero non-floor interior open edges and zero non-manifold edges for ratios 0, 2 and 4.
- The full `npm run test:dawn:sparse-cm12` run passed every lane except the topology page-budget lane, which timed out. Its Dawn instance was not retained across asynchronous work in direct invocation. Retaining it fixes the harness lifetime issue; the unchanged lane passes in 7.8 s against its existing 30 s limit. Every canonical lane therefore passed across the full run and focused rerun.
- Mini32 and mini64 median advance times are 28.6 ms and 83.9 ms, below their unchanged 40 ms and 110 ms ceilings.
- Repository typecheck still reports existing errors outside the changed renderer and mesh-test files.

Final UI check: Chrome loaded the production scene with Sparse CM12, Coarse first, column height off, ×2 mesh refinement, 33.3 ms step, and no settings overrides. Single-stepping reached the reproduced 0.3333 s frame with the repaired surface and liquid at the far corner. Paused presentation was about 38–40 FPS. The isolated cold startup took over two minutes; this remains a startup limitation, not a repaired performance claim. The tab was closed and the GPU lease released afterward.

The final sampler keeps background-octree fallback out of native interpolation: compact missing samples are authoritative air. The final mixed-resolution and production mini64 Dawn mesh tests both pass (about 50.5 s total); mini64 has 66,956 triangles and zero non-floor interior open edges or non-manifold edges at each tested ratio.
