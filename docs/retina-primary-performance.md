# Retina primary visibility

Measured on 2026-10-25 with an Apple M1 Max, Dawn/Metal and GPU timestamps.
The pool uses its preset overview camera, 25 mm visual cells (refinement 1),
50 mm fluid cells, dual marching cubes, raster primary, and raster AO.
These are warmed dry-render measurements: fluid simulation and water compositing
are excluded. They do not reproduce the screenshot's running simulation state.
The baseline source is `767fc07f`.

## Why the primary band is large

In the supplied capture, primary rasterization accounts for 10.9 of the primary
band's 11.5 ms. Mesh maintenance is 0.07 ms and culling is 0.51 ms. The shared
raster pass draws background/planes, native imported meshes, voxel triangles and
rigid surfaces; its label does not mean all that time is octree traversal.

Resolution changes both fragment work and geometry admission:

- A 3448×1892 framebuffer has four times the pixels of 1724×946.
- Sample-coverage culling retains more small voxel triangles at the larger size.
- Native asset LOD selection uses projected error in pixels, so increasing
  resolution can select finer index streams.
- The four primary colour attachments and depth occupy 60 bytes per pixel:
  approximately 391 MB at the Retina size. This is attachment storage, not a
  measurement of physical memory traffic; tile storage and compression matter.

The final resolution sweep reuses one scene, alternates the sizes twice, and
reruns production culling and LOD selection after each resize. Each measurement
has eight warmups and 64 isolated GPU timestamp samples:

| Viewport | Pixels | Drawn voxel records | Cull median, both visits | Raster median, first / second visit |
| --- | ---: | ---: | ---: | ---: |
| 1724×946 | 1,630,904 | 274,981 | 0.262 ms | 2.687 / 3.408 ms |
| 3448×1892 | 6,523,616 | 493,066 | 0.262 ms | 4.719 / 4.653 ms |

The draw is resolution dependent, but does not scale purely with pixel count.
It also pays for geometry, and desktop load affects absolute timings. The
retained voxel mesh is unchanged at 1,482,200 records.

Draw-removal experiments identify native imported assets as the largest marginal
cost, followed by voxel triangles. In the initial run, omitting native draws
reduced the pass from 4.850 to 2.490 ms; omitting voxel triangles measured
3.408 ms. The ending baseline had drifted to 6.095 ms, so these are diagnostic
attributions, not an additive breakdown or a precise optimization result.

A separate interleaved diagnostic kept native geometry and replaced only its
material/normal sampler with a trivial value. The pass measured 5.112 ms with
the trivial sampler versus 6.095 ms with production sampling. Sampling matters,
but removing it would still leave most of the total pass cost.

## Change and validation

Native asset indices previously retained extraction/simplification order. Each
LOD now receives vertex-cache triangle ordering once during asset preparation.
The encoder's vertex remap is inverted so every level and instance still shares
the original vertex buffer. Faces, winding, vertex positions, index counts,
LOD errors, normals, materials and render resolution are preserved.

The benchmark captures the previous native index buffers, then alternates old
and new buffers in ABBA order on the same device and published scene. Shader,
vertex buffers, draw ranges, attachments and camera stay the same. The final
3448×1892 comparison uses 16 warmups and 256 samples (128 per arm):

| Primary rasterization | Median | p95 |
| --- | ---: | ---: |
| Original order | 4.784 ms | 4.915 ms |
| Vertex-cache order | 4.653 ms | 4.981 ms |

This is a **2.7% median improvement**, with no demonstrated tail improvement.
An earlier, noisier comparison measured 7.733 → 7.340 ms. Neither result implies
that the screenshot's 10.9 ms becomes 4.65 ms. The screenshot also has a separate
13.2 ms lighting-visibility cost that this change does not address.

Overview, close-up and return captures are byte-identical in RGBA16F. The orbit
view differs at one of 6,523,616 pixels by at most 0.0000611 linear intensity;
its displayed RGB remains [75, 90, 62]. All captured depths match exactly.
GPU validation reports no errors.

Tests verify oriented triangle multiplicity, sparse shared vertex addresses,
original/reference buffer replay and resource lifetime. `npm run check:types`
passes. `npm run test:unit` passes with 1,201 passing, 127 skipped and no failures.

Sampler-loop unrolling and duplicate material-vote rejection did not improve
the paired median. Reusing adjacent decoded normals regressed 4.653 → 4.784 ms.
Those experiments were removed.

## Reproduction

```sh
export WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal
export FLUID_SVO_DRY_FRAME_SCENE=backyard-pool-party
export FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/pool-party-live.ts
export FLUID_SVO_DRY_FRAME_WIDTH=3448 FLUID_SVO_DRY_FRAME_HEIGHT=1892
export FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT=1
export FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary FLUID_SVO_DRY_FRAME_SURFACE_MESH=1
export FLUID_SVO_DRY_FRAME_MESHER=dual-marching-cubes
export FLUID_SVO_DRY_FRAME_SPECIALIZED_LIGHTING=1
export FLUID_SVO_DRY_FRAME_CONE_TRACING=raster-ao
export FLUID_SVO_DRY_FRAME_CONE_FANOUT=0 FLUID_SVO_DRY_FRAME_CONE_SCALE=1
export FLUID_SVO_DRY_FRAME_TIMING=gpu FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=1
export FLUID_SVO_DRY_FRAME_WARMUPS=8 FLUID_SVO_DRY_FRAME_CYCLES=64
export FLUID_SVO_DRY_FRAME_OUT=artifacts/retina-primary/repro.json
export FLUID_SVO_DRY_FRAME_PRIMARY_RESOLUTIONS=1724x946,3448x1892,1724x946,3448x1892
node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

`FLUID_SVO_DRY_FRAME_PRIMARY_ABLATION=1` omits each draw category in turn.
For paired ordering comparisons, capture the original implementation with
`FLUID_SVO_DRY_FRAME_PRIMARY_INDEX_CAPTURE=<directory>`, then run the reordered
implementation with `FLUID_SVO_DRY_FRAME_PRIMARY_INDEX_REFERENCE=<directory>`.
This comparison requires identical vertex buffers and LOD ranges. The benchmark
rejects missing reference buffers and unequal index buffer sizes.

Local evidence is in `artifacts/retina-primary/`: `final.log`,
`final.json.primary-resolutions.json`, `indices-comparison.json`,
`index-baseline.rgba16f`, `indices.rgba16f` and their camera variants.
