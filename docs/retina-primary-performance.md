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
- The four primary colour attachments and depth occupy 52 bytes per pixel:
  approximately 339 MB at the Retina size. The separate HDR output adds another
  8 bytes per pixel, bringing these textures together to about 391 MB.
  This is attachment storage, not a
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

## Adaptive geometry investigation

The native selector already adapts per instance: it chooses the coarsest stored
index stream whose estimated world-space simplification error projects to at
most **0.25 physical pixels**. It projects using the nearest depth of the asset's
bounding sphere, which is conservative for elongated assets or a camera close to
a large asset. The sphere is also used for frustum culling. This is independent
of the normal/material field sampler; coarser geometry still evaluates the
original source field at each surviving full-resolution fragment.

The voxel mesh has a different policy. `dryMeshLodPixels` scales its reference
pixel threshold with viewport height, but returns zero for dual contouring and
dual marching cubes. In these smooth modes, the existing “Filtered detail”
control cannot lower geometry resolution. Sample-coverage culling only removes
triangles that cannot cover a raster sample; it is not a perceptual LOD.

The first experiment is therefore a native LOD error sweep, keeping the viewport,
materials, source meshes, shadows and voxel geometry fixed. The benchmark-only
`FLUID_SVO_DRY_FRAME_NATIVE_LOD_SWEEP=0.25,0.5,1,2` compiles selector variants and
alternates each candidate with the shipped quarter-pixel selector in ABBA order.
It records native triangle/instance counts, isolated primary GPU time and
display/depth/coverage differences in overview, orbit and close-up views. PNGs
and a `.native-lod.json` report are written alongside the requested output.
The sweep does not change the production selector.

### Full-resolution native LOD sweep (2026-10-25)

The pool sweep used 3448×1892, minimum AO 0, the preset overview, an azimuth
offset of 1.1 radians, and a close camera at one quarter of the preset distance.
Each comparison used eight warmups and 64 ABBA samples (32 per arm). The scene
is static and dry, with no simulated water. These timings isolate the primary
raster pass and exclude culling, lighting and presentation.

| Camera | Error allowance | Native triangles, reference → candidate | Raster ms, reference → candidate | Display RMSE / 255 | Surface pixels differing >8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overview | 0.5 px | 8,712,025 → 6,921,427 | 5.898 → 4.981 | 10.68 | 5.73% |
| Overview | 1 px | 8,712,025 → 6,564,109 | 4.850 → 4.063 | 11.99 | 7.35% |
| Orbit | 0.5 px | 8,812,259 → 6,943,205 | 4.850 → 4.522 | 11.15 | 6.30% |
| Orbit | 1 px | 8,812,259 → 6,564,109 | 4.784 → 4.063 | 13.12 | 8.15% |
| Close | 0.5 px | 8,944,639 → 5,306,922 | 4.260 → 4.194 | 5.20 | 1.72% |
| Close | 1 px | 8,944,639 → 4,569,572 | 4.260 → 3.932 | 7.04 | 2.68% |

Two pixels selected the same native index streams as one pixel in the overview
and orbit views. Close-up it selected 4,314,838 triangles, measured 3.998 ms and
increased RMSE to 8.91. Absolute clocks drifted in the first overview comparison;
ABBA samples are retained in the report, and the rows are independent paired
comparisons rather than a shared fixed baseline. Visible instance counts are
unchanged in every comparison. GPU validation reported no errors.

This is evidence of both geometry and pixel costs: cutting close-up native
triangles by 41% at 0.5 px barely changed raster time, while the overview gained
more. **No looser native error limit is shipped.** The half-pixel result already
shows objectionable speckling on the umbrella. Estimated geometric error does
not bound shading error when simplified geometry samples the original normal
and categorical-material fields. A future LOD policy needs an appearance-error
constraint as well as a silhouette/position constraint. These image errors are
substantially larger than the accepted adaptive AO changes.

Local evidence: `artifacts/primary-adaptive/pool.log`,
`pool.json.native-lod.json` and its overview/orbit/close PNGs. Type checking and
the unit suite passed (1,221 passed, 131 skipped).

Unlike the AO minimum, geometry error should not be normalized so that some
object must always select the coarsest mesh: that could remove a silhouette
even when every visible object needs detail. A screen-space error allowance,
with tighter treatment for thin features and stable transitions as the camera
moves, is the appropriate quality constraint. Smooth voxel geometry would need
compatible simplified meshes and seam handling before it can use that policy.

A larger alternative is to separate full-resolution visibility (depth and
surface identity) from material/normal evaluation, then share those evaluations
inside smooth regions with exact fallbacks at boundaries. That targets fragment
work and attachment traffic rather than triangle count, but requires a new
representation and reconstruction pass; its net benefit has not been measured.

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
