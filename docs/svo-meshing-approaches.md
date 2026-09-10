# GPU raster meshing approaches

The raster-primary meshing selector now offers **Voxel faces**, **Clipped contours**, **Hermite DC**, and **Dual Marching Cubes**. The default remains voxel faces. Select **Primary visibility → Meshing method → Hermite DC**, or use `svoMesher=dual-contouring` with mesh primary visibility. Old `svoMeshContours=1` URLs migrate to clipped contours; an explicit `svoMesher` takes precedence.

See [GPU Dual Marching Cubes](svo-dual-marching-cubes.md) for the fourth strategy, its validation and comparison measurements.

## Strategy boundary

`lib/svo/features/meshing/plugins.ts` registers the built-in strategies and their construction requirements. Extraction lives in `voxels.ts`, `contours.ts`, and `dual-contouring-mesh.ts`. The DC fitting kernel lives in `dual-contouring.ts`.

The common surface-mesh scheduler owns count/emit dispatches, capacity and overflow handling, dirty-region invalidation, the two mesh arenas, publication, culling, indirect drawing and G-buffer output. Strategies emit through `meshAppend`; they do not own a second renderer. Clipped contours still use greedy voxel faces for unclipped cells. The registry is a built-in strategy boundary, not a dynamic third-party plugin loader.

| Approach | Construction input | Geometry | Strength | Principal limitation |
| --- | --- | --- | --- | --- |
| Voxel faces | Coverage/material | Greedy axis-aligned quads, existing LOD | Cheap, conservative solid coverage | Stepped silhouettes and edges |
| Clipped contours | Coverage, normal, conservative support plane | Independent clipped cells | Retains thin occupied features; compact attachment | Neighbouring patches need not agree; inflation overlaps patches |
| Hermite DC | Signed field sampled at grid corners and edge roots | Connected surface from fitted cell vertices | Smooth shape approximation and sharp edge placement | Corner sampling misses unresolved sheets; one vertex cannot represent arbitrary topology |
| Dual Marching Cubes | Values and gradients throughout cells | Marching Cubes on a fitted dual grid | Can resolve thin features between primal corners; sharp function features | More construction samples; unresolved function features and folded dual cells remain possible |

## Hermite DC implementation

All new geometry construction runs in GPU compute: corner signs, edge-root bisection, finite-difference gradients, QEF accumulation, bounded vertex fitting, sign-edge connectivity, triangle packing, counting and emission. Normal interpolation and drawing run on GPU too. No mesh geometry is read back to the CPU in production. The existing host-side scene authoring, sparse topology planning, buffer allocation and dispatch scheduling remain in place; this change does not make those systems GPU-only.

The construction callback evaluates the shared procedural primitive distance field and renderer-owned bilinear terrain field with ordered solid patches. It does not fit from the coverage fraction or choose special algorithms for particular primitives. Canonical-only solid-world inputs without a continuous field retain a discrete sign fallback; DC cannot invent smooth source detail from those inputs.

Each active cell finds the crossings of its twelve grid edges (twelve bisection iterations per crossing). A regularized QEF is solved in local cell coordinates. Enumerating the interior, faces, edges and corners gives a bounded solution without simply clamping an unconstrained answer. Vertices are quantized to 1/256 cell. Neighbouring faces use the same stored vertices, so packing them relative to different face origins preserves matching positions.

Each sign-changing grid edge connects its four incident cell vertices into two triangles. The initial implementation requires equal cell sizes: render content is refined to the selected uniform depth and planar-terminal substitutions are disabled. Mixed-size connections are deliberately not implemented. Grid/domain boundaries need incident cells; unresolved or missing cells can leave openings.

A 16-byte record per allocated voxel stores the fitted position, corner signs, normal and crease marker in the producer's GPU maintenance arena. The final raster mesh copies positions into the existing 32-byte triangle record, so an old mesh never refers to vertices being rewritten by a new construction transaction. Smooth cells interpolate field normals; cells whose Hermite normals disagree substantially mark incident triangles for geometric-normal shading. This is an initial crease classification, not a guarantee of feature-aligned shading for every CSG junction.

Dirty regions include a one-cell halo for DC's diagonal dependencies and two-cell face footprints. Mesher changes rebuild presentation data without changing the fluid solver allocation key. Source voxel coverage remains available to existing traced/shadow/lighting paths; those paths do not intersect the DC triangles, so secondary visibility can differ from raster primary.

## Costs and limits

- DC adds 16 bytes per allocated voxel, including inactive capacity. The garden-hose native build allocates 7,174,656 voxel slots: approximately 109.5 MiB of DC construction attachment, in addition to the triangle arenas and existing scene data. The existing “Mesh memory” readout covers the triangle arenas, not this attachment.
- Root finding and QEF fitting happen during construction/edits, not each warm frame. This prototype repeats corner and edge samples between cells; a shared Hermite-edge cache is a future optimization.
- Uniform refinement can be expensive. Allocation fails explicitly if the enlarged producer arena exceeds the device's storage-binding limit. There is no silent downgrade to a different mesher.
- No adaptive DC transitions, QEF simplification, multi-component/manifold DC, sub-cell feature recovery or self-intersection guarantee is included yet. Use finer resolution when corner signs miss a feature. Clipped contours remain useful for comparison on thin geometry.

## Validation, 10 September 2026

Dawn tests execute the production fitting and extraction kernels on a 16³ grid, then read back only for test assertions. Sphere, rotated box and bounded wavy-terrain fixtures have two owners per undirected edge, outward total winding, and these maximum vertex errors against their analytic fields:

| Fixture | Triangle records | Maximum error, cells |
| --- | ---: | ---: |
| Sphere | 816 | 0.07530 |
| Rotated sharp box | 508 | 0.00226 |
| Bounded wavy terrain | 560 | 0.07988 |

These are vertex-error checks, not bounds on error throughout a triangle or a general topology proof. Tests also compile the construction producer for dense, occupancy and banded scene formats. GPU scheduler rollback/replacement, filter pipelines, mesher URL migration and solver-allocation isolation pass (10 tests).

A native garden-hose run on Apple M1 Max, Metal, raster primary, 400×240, with GI/shadows/AO disabled, produced 417,230 triangle records, 187,381 after culling, ready publication and zero mesh overflow/fallback flags. Construction took 2.54 seconds. Four warm submit-to-fence frame samples were 7.82, 1.83, 1.65 and 1.64 ms; the reported median was 1.83 ms. This is a smoke measurement with an outlier, not a comparative performance claim. Captures and the full report were written to `/tmp/svo-dual-contouring/native/`.

Reproduce the focused tests:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
node --import tsx --test --test-concurrency=1 \
  tests/svo-dual-contouring-dawn.test.ts \
  tests/svo-surface-mesh-scheduler-dawn.test.ts \
  tests/svo-filter-controls-dawn.test.ts \
  tests/svo-cell-contour.test.ts tests/svo-meshing-plugins.test.ts
```

The frame harness accepts `FLUID_SVO_DRY_FRAME_MESHER=dual-contouring` together with `FLUID_SVO_DRY_FRAME_SURFACE_MESH=1`. Keep the legacy `FLUID_SVO_DRY_FRAME_MESH_CONTOURS` flag disabled for this mode. Run Dawn exclusively, without a concurrent browser or another Dawn process.

The native capture and report are also retained under `artifacts/svo-dual-contouring/native/` (ignored build artifacts).

The required `npm run test:dawn:sparse-cm12` gate was also run. It did **not** pass: symmetric expansion reported D4 error 0.0149062 against 0.006; hydrostatic-adaptivity, mini32-correctness and the mini32/mini64 performance lanes timed out. The 180-second suite budget expired before the remaining lanes. No lane or ceiling was changed. Full log: `/tmp/dc-cm12-gate.log`. Repository-wide TypeScript checking also remains red on errors outside the changed files; no errors were reported in the meshing implementation or modified benchmark after correction.
