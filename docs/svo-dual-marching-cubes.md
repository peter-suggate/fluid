# GPU Dual Marching Cubes

Implemented as the fourth built-in raster meshing strategy. Select **Primary visibility → Meshing method → Dual Marching Cubes**, or `svoMesher=dual-marching-cubes` with mesh primary visibility. Voxel faces, clipped contours and Hermite DC remain selectable.

## Plan and delivered scope

1. Fit the implicit function rather than directly fitting its zero surface.
2. Connect those fitted samples into a dual grid and contour that grid on GPU.
3. Integrate with the existing meshing registry, raster-primary scheduler, allocation and publication.
4. Validate analytic smooth/sharp/thin geometry, ambiguous faces, existing strategies, and garden lighting.

The algorithm follows the function-graph construction in Schaefer and Warren's [Dual Marching Cubes: Primal Contouring of Dual Grids](https://www.cs.rice.edu/~jwarren/papers/dmc.pdf). It is a **uniform sparse-grid variant**, not an implementation of the paper's adaptive octree traversal. The final contouring uses a Marching Cubes face graph, with a consistent face ambiguity decision and occasional interior polygon vertices, rather than the fixed Lorensen–Cline triangle table.

## GPU implementation

`lib/svo/features/meshing/dual-marching-cubes.ts` samples values and finite-difference gradients on a 3×3×3 lattice in each resident cell. It fits tangent hyperplanes to the function graph in four dimensions: three position coordinates and the scalar value. Eliminating the scalar coordinate analytically gives a three-dimensional quadratic system. Centred covariance avoids subtractive cancellation on planar fields; the shared bounded QEF solver selects a position within the cell, with regularization toward its centre.

The producer stores a fitted position **and estimated scalar value** for every cell. Cells with identical primal-corner signs are still fitted, allowing a dual sample to enter a thin feature between those corners. This is the substantive difference from Hermite DC's one surface vertex per sign-changing cell. Both use the same generic construction field, including procedural primitives, bilinear terrain and ordered solid patches. There are no primitive-specific meshing branches.

The restricted zero-value QEF provides sliver elimination. A normalized squared residual below `1e-6` snaps the fitted scalar to exactly zero. Its RMS tolerance is 0.001 characteristic cells, below the 1/256-cell position quantization. This avoids treating numerical noise around planar features as alternating inside/outside samples. Quantization can make several crossed edges share a vertex; the extractor removes repeated packed points before triangulating each polygon.

`dual-marching-cubes-mesh.ts` visits each primal-grid vertex and reads the eight incident cell records. They form a dual hexahedron. Sign-changing dual edges are interpolated; the six cube faces connect those crossings into contour loops. Alternating-sign faces use the same bilinear determinant and tie rule on both owners. Winding is established in cube parameter space, so collapsed or warped world-space triangles cannot independently reverse neighbouring faces.

Ordinary polygons use an n−2 triangle fan. If a fan diagonal would lie on a shared cube face, an interior fan avoids duplicating the neighbouring cell's boundary triangles. Near-zero fitted samples also mark nearby triangles for geometric-normal shading to retain planar creases; other triangles use the existing smooth-normal path.

Construction, fitting, connectivity, ambiguity decisions, packing, counting, emission and drawing all run on GPU. Production does not read mesh geometry back to the CPU. Existing host-side scene authoring, sparse topology planning, allocation and command scheduling remain unchanged in ownership.

## Plugin and publication contract

The strategy registers its fitting/extraction kernels and requirements in `meshing/plugins.ts`. The allocator, two raster mesh arenas, count/emit transaction, culling, dirty-region halo and indirect drawing remain shared. The 16-byte attachment has a different fourth word from Hermite DC: DMC stores a float scalar, while DC stores signs/normal/crease metadata. The source therefore publishes its attachment kind. The GPU scheduler checks that tag before extracting, preventing one method from interpreting the other's records during a selector/source transition.

DMC claims a one-cell sampling halo, including face/edge/corner neighbour bricks, and expands candidate indexing and edit invalidation by the same support distance. This provides positive field samples outside solid bounds without inflating the SDF. Its terrain input has a finite exterior continuation of the bounded heightfield, so dual edges crossing the terrain footprint interpolate meaningful values rather than an absent-data sentinel.

Both methods copy final positions into the existing immutable 32-byte raster triangle records. An old mesh never references a construction attachment being rewritten by an edit. Switching methods changes the presentation key but does not change the fluid solver allocation key.

## Validation

Dawn executes production fitting and extraction WGSL on a 16³ grid. Readback is used only for test assertions. All fixtures have two triangles per undirected edge, opposite directed edge winding, positive enclosed volume, and maximum analytic vertex error below 0.06 cells:

| Fixture | DMC triangles | Max vertex error (cells) | Hermite DC comparison |
| --- | ---: | ---: | --- |
| Sphere | 1,020 | 0.03108 | 816 triangles, 0.07530 |
| Rotated sharp box | 434 | 0.00762 | about 508 triangles, 0.00226 |
| Bounded wavy terrain | 702 | 0.01842 | 560 triangles, 0.07988 |
| 0.16-cell wall between integer x coordinates | 280 | 0.02938 | No primal-corner sign crossings |
| Synthetic dual-grid checkerboard | 562 | Not an analytic-error test | Exercises ambiguous shared faces |

The wall retains both sheets and encloses approximately 6.918 cubic cells (analytic volume 6.746). This demonstrates one sub-cell configuration, not arbitrary thin-feature recovery. The checkerboard is a direct dual-grid extraction fixture; it intentionally tests connectivity independently of fitting.

The producer compiles for dense, occupancy and banded scene payloads, with and without the terrain field. A GPU occupancy test also verifies that the sampling halo retains all 26 neighbouring bricks around an isolated claim. Eleven focused tests pass, including the existing Hermite DC fixtures, scheduler incremental rebuild/rollback/replacement, filter pipelines, URL migration and solver-allocation isolation.

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
node --import tsx --test --test-concurrency=1 \
  tests/svo-dual-marching-cubes-dawn.test.ts \
  tests/svo-dual-contouring-dawn.test.ts \
  tests/svo-surface-mesh-scheduler-dawn.test.ts \
  tests/svo-filter-controls-dawn.test.ts \
  tests/svo-cell-contour.test.ts tests/svo-meshing-plugins.test.ts
```

The final garden-lighting capture on Apple M1 Max / Metal at 1200×800, with GI, shadows and AO enabled, produced 206,602 triangle records (106,115 after culling), ready publication, zero mesh overflow and zero mesh fallback flags. The source has 3,146,752 voxel slots: approximately 48.0 MiB of DMC attachment. The earlier Hermite DC capture at the same resolution had 159,778 triangle records and 2,952,192 slots. DMC's outside-sample halo also closes additional boundary geometry, so this is not just a tessellation comparison.

The final DMC run took 1.52 seconds to construct; four warm submit-to-fence frame samples were 6.35, 9.00, 6.54 and 6.05 ms (reported median 6.54 ms). These are smoke measurements, not a paired performance benchmark against DC. The source report and screenshot are retained together for reproduction.

The frame harness accepts `FLUID_SVO_DRY_FRAME_MESHER=dual-marching-cubes` with `FLUID_SVO_DRY_FRAME_SURFACE_MESH=1`; disable the legacy contour flag. Garden-lighting captures and receipts are retained in `artifacts/svo-dual-marching-cubes/garden-lighting/`. Run Dawn exclusively, without another Dawn process or browser GPU session.

## Costs and current limits

- The attachment costs 16 bytes per allocated voxel, equal to Hermite DC's attachment, plus the emitted mesh. The attachment kind check is metadata, not another geometry allocation.
- Function fitting samples every resident cell, and therefore costs more field evaluations away from the surface than DC's corner-sign early exit. Samples are not cached across neighbours yet. All fitting is construction/edit work, not warm-frame work.
- Initial topology is uniform at the selected render resolution. No adaptive transitions, adaptive error refinement or manifold/multi-component fitting guarantee is supplied.
- Shared-face consistency and the tested closed meshes do not prove that arbitrary fitted dual cells cannot fold or self-intersect. A cell containing multiple unresolved function features still needs refinement.
- The garden capture still has visible artifacts around some intersections. Secondary shadow/GI visibility continues to use the existing voxel representation, which can disagree with the raster mesh. This variant is available for comparison rather than being a universal quality replacement.

## Seam visibility investigation

The largest perceived defect is the dark treatment of seams and intersections.
Four exclusive Dawn captures of `garden-svo-lighting` at 1200×800, using the
same DMC raster mesh and camera, are retained in
`artifacts/svo-seam-visibility/`: `all`, `no-direct-shadows`, `no-visibility`
(shadows, AO and GI disabled), and `direct-shadows-only`. Each directory has
the full-resolution `reference.png` and its benchmark receipt. These are
diagnostic ablations, not proposed rendering settings or a completed fix.

Disabling direct shadows alone leaves dark patches at intersections and on
curved surfaces. Disabling AO and GI as well makes those patches much lighter,
but leaves visible surface discontinuities. Thus direct shadows alone do not
explain the defect; secondary visibility amplifies an existing surface issue.
The GI-off benchmark also withholds the radiance source, so the direct-only
capture must not be treated as a purely additive decomposition of the all-on
frame.

Two representation inconsistencies are visible in the production code:

- Raster primary hits lie on fitted triangles, while shadow, contact and GI
  visibility still use voxel coverage. Their receiver and occluder boundaries
  need not coincide.
- `dryRasterPrimaryFacedSurface` publishes both normals to the packed G-buffer,
  but deferred lighting does not read that packed plane. `dryOpaqueFaceWord`
  preserves a separate geometric normal only for exactly axis-aligned faces;
  otherwise `dryGeometricNormal` falls back to the shading normal. Triangle
  geometric normals therefore do not survive that handoff in general.

Neither observation alone establishes the cause of every visible patch.

### Implemented receiver correction

Reconstructed receivers now carry their oct8 geometric normal through both
the full-resolution split identity and reduced lighting identity. Static mesh
receivers have no rigid owner, so the tagged encoding reuses those owner bits;
rigid ownership and ordinary voxel axis normals retain their old encoding.
The reduced metadata tag fits exactly in its existing f16 channel. No texture,
buffer, extra construction pass, or CPU geometry work is added.

Shadow, contact, cone fan-out and GI ray escapes use the geometric normal.
Material shading and the GI sampling hemisphere retain the shading normal.
Reduced receiver compatibility additionally rejects normals across a crease.
The reduced visibility pass samples the published raster hit and its matching
camera ray when reconstructed geometry is present, instead of retracing a
different voxel surface at boundary pixels. Reconstructed receivers bypass
the visibility cache whose representative position is a voxel centre.

`artifacts/svo-seam-visibility/fixed/` retains the final full and reduced
captures and benchmark receipt. The cap junctions and lamp base lose much of
their bright/dark scalloping while contact shadows remain. Some patches on
upper surfaces remain. Secondary **occluders still use voxel coverage**; this
change corrects receiver consistency, not mesh-accurate secondary traversal
or remaining folded/missing geometry. Global bias values are unchanged.

The final 1200×800 garden run measured 7.72 ms median submit-to-fence time
versus 6.36 ms in the preceding baseline, four samples each. These are separate
smoke runs, not an interleaved performance comparison. Cache rejection and
stricter crease compatibility can increase per-frame visibility work.

Eight focused tests pass, including Dawn shader/pipeline validation, mesh
scheduling, and the new `svo-reconstructed-receiver-dawn.test.ts`. The latter
executes the production packing/decoding on GPU and checks arbitrary normal
directions, reduced f16 metadata round trips, rigid ownership, and crease
rejection. Repository-wide TypeScript checking still reports unrelated errors;
none were reported in the changed shader, mesh or receiver tests.

The post-correction CM12 gate remains red: D4 symmetry error 0.304505 exceeds
0.006; performance probes cannot find their baseline artifact; the far-wall
lane reaches the remaining time limit and subsequent lanes exhaust the
180-second suite budget. No lane or threshold was changed. The report is
retained at `artifacts/svo-seam-visibility/fixed/cm12-gate.log`.

## Broader regression status

The required `npm run test:dawn:sparse-cm12` was run again after the final implementation changes. It remains red: symmetric expansion reports D4 error 0.0149062 against 0.006; mini32 performance reports 43.1227 ms against the 40 ms ceiling; topology-page-budget, hydrostatic-adaptivity and mini64 performance time out. The 180-second suite budget expires before the remaining lanes finish. No thresholds or lanes were weakened. Final log: `/tmp/dmc-post-change-cm12.log`.

Repository-wide `tsc --noEmit` also remains red on errors outside the modified files. No TypeScript errors were reported in the modified meshing/construction/presentation files, the new tests, or the benchmark. `git diff --check` passes.
