# Pool party rendering and picking

The pool uses a 50 mm fluid lattice and a separate 6.25 mm visual octree. Imported assets keep their native signed-distance samples and their complete PBR palettes; grass retains its 3.125 mm source sampling. These changes do not reduce render resolution, visual refinement, shadow samples, or material detail.

## Missing mesh asset during picking

The `Mesh asset diving-board-ladder has not loaded` exception came from the window's synchronous scenery picker. The render worker had loaded the volume, but the window had only received the scenery catalog.

The worker waits until all of the scene's scalar fields are loaded, then publishes each unique asset's field before publishing the catalog that makes it pickable. This also covers draws during initialization. In cross-origin-isolated browsers, brick views share the decoder's existing immutable `SharedArrayBuffer`; there is no second download or copy of the distance samples. Other hosts clone only the scalar channel once per asset. Spatial material IDs and normal channels remain in the renderer. Picks still intersect the actual surface, including gaps within the asset bounds.

CPU marching now evaluates only distance until it accepts a hit, then computes its normal. Already-occluded scenery is bounded by the nearest hit found so far.

## Rendering changes

- Imported proxy mesh faces were allocated, culled, transformed and then discarded in the fragment shader. The mesh extractor now rejects them in both its count and emit phases. Native sampled surfaces still draw them, and the voxel field remains available to lighting. The separate raster-shadow-map mode retains its proxy casters.
- Distance-only GPU queries skip source normal decoding and material voting. The accepted surface still uses the complete PBR sampler.
- Native opaque shadow rays stop after their first exact blocker. Camera rays continue searching for the nearest surface.

## Measured bottlenecks and results

Measured on an Apple M1 Max with 32 GiB unified memory, using Metal through Node 25.8.1 and Dawn, at 1000×720. Both captures use the same pool scene and camera, 6.25 mm visual cells, 50 mm fluid cells, surface meshing, split raster primary visibility, full PBR, and cone lighting at scale 0.5. Baseline is commit `86afa154`; the comparison includes the rendering changes above. Each run warms up four frames and measures a 12-second loop.

These are **dry scene rendering measurements**, excluding fluid simulation and water compositing. They are not live simulation FPS. WindowServer competed for GPU time in both captures; the results establish the improvement under these desktop conditions, not an isolated hardware ceiling.

| Metric | Before | After |
| --- | ---: | ---: |
| Median render frame | 187.34 ms | 136.69 ms |
| 95th percentile | 310.19 ms | 168.72 ms |
| Median JavaScript encoding | 0.568 ms | 0.565 ms |
| Live mesh records | 32,414,185 | 15,823,181 |
| Allocated GPU mesh arena | 2 GiB | 1 GiB |

Median frame time decreased **27.0%**, equivalent to **37.1% greater rendering throughput**. The mesh arena saves **1 GiB**, without reducing the retained native volume data.

The dominant bottlenecks are GPU work:

- Deferred dry lighting remains the largest isolated stage: **83.78 → 77.52 ms** average. Native shadow intersection and distance queries are part of this work.
- The composite bucket beginning at mesh rasterization fell from **95.77 → 39.02 ms**. This supported removing faces that the fragment shader already discarded.
- The composite bucket beginning at imported native surfaces fell from **32.58 → 24.98 ms**. Native volume tracing remains a substantial cost.
- CPU frame encoding is under 0.6 ms; GPU submission-to-completion accounts for almost all measured frame time.

Composite buckets include every stage until the next Metal pass boundary; they are **not isolated kernel timings**, and overlapping GPU intervals must not be added as exclusive costs. The reducer now restricts analysis to the final measured frame starts, excluding initialization and warmup with identical labels. It retains 51 complete baseline frames and 83 complete optimized frames after clipping boundary frames.

Local evidence: [baseline report](../artifacts/pool-timing-before/report.html), [optimized report](../artifacts/pool-timing-after/report.html), [baseline frame](../artifacts/pool-timing-before/frame.png), [optimized frame](../artifacts/pool-timing-after/frame.png), and [image comparison](../artifacts/pool-timing-after/image-comparison.json). Generated captures are ignored by Git.

Reproduce from the repository root (Metal and Xcode Instruments required):

```sh
FLUID_SVO_DRY_FRAME_SURFACE_MESH=1 \
  FLUID_SVO_DRY_FRAME_MESHER=dual-marching-cubes \
  FLUID_SVO_DRY_FRAME_SPECIALIZED_LIGHTING=1 \
  node --import tsx tools/profile-svo-render-xctrace.ts \
  --scene=backyard-pool-party --scene-module=tools/preview/pool-party-live.ts \
  --resolution=1000x720 --refinement=3 --traversal=raster-primary \
  --shading=split --screen-space-pixels=0 --cone-scale=0.5 --warmups=4 \
  --timing-only --out=artifacts/pool-timing-repro
```

## Verification

The worker-handoff regression test picks the actual diving board with no full asset/PBR volume registered in the window and verifies that empty space inside its bounding box remains a miss. A shared-buffer test covers brick seams and confirms that structured cloning shares the scalar allocation.

`node --import tsx tools/blender/check-volume-distance-gpu.ts` compares 32,768 GPU queries across the diving board, travertine, turf and parasol. The largest scalar/shaded distance difference was 3.73e-9 metres (compiler floating-point rounding). Both the normal cone-lighting and raster-shadow shader variants compile.

The settled before/after images have **99.989% identical pixels** after the production AgX display transform: 78 of 720,000 pixels changed, and only one differed by more than two channel levels out of 255. Neither capture reported GPU validation errors. Chrome loaded the live pool and successfully selected `DIVING/ASSEMBLY`; [selection screenshot](../artifacts/pool-performance-live/diving-board-selected.jpg). `npm run check:types` passed; `npm run test:unit` passed with 900 passing tests and 45 skipped.

Full GPU-counter captures consumed excessive profiler memory and were discarded as performance evidence. Final measurements use Metal timing-only captures. The benchmark now reports actual visual cell size separately from fluid cell size, includes mesh memory and validation status, and can capture a final frame after the timing window for image comparison.

## Primary rasterization follow-up

An isolated replay of each production render pass, with GPU timestamps and warmed fixed-camera resources, confirms that the smooth mesh draw itself is expensive: **39.98 ms**, separately from **12.26 ms** for imported native fields. The earlier composite-bucket timing was not sufficient attribution on its own.

Removing all 128 turf instances leaves **7,907,840** drawn mesh records versus **7,891,528** with turf. Grass is therefore not the source of the mesh explosion. The separate imported-surface pass drops to **7.54 ms** without turf; the mesh pass remains expensive (49.15 ms in that run, with desktop contention). Removing grass alone does not restore the old rasterization cost.

A GPU census of the actual culled draw list attributes 3,939,763 records to foliage, 2,208,005 to SolidWorld pool/deck geometry, 935,754 to tree/shrub structure, 336,866 to the house wall and 297,672 to guests. No turf records occur in this draw; imported turf has its own surface pass.

The renderer previously submitted one four-vertex instance per record. The new triangle-list stream submits all visible records in one instance, with three vertices for each DMC triangle and six corners for ordinary quads. It preserves record order, strip winding, quad diagonals, quantized vertex positions, shading normals and material identities. The original indirect header remains intact for shadow-map consumers and diagnostics; a separate four-word argument block serves the continuous primary stream. Culling and the unculled diagnostic path both populate it.

The GPU regression test compares instanced strips against the continuous stream for triangles, quads, both windings and mixed quad/triangle records. It checks exact pixel coverage, interpolants and identities. The profiling helper also rebuilds untimed prerequisite passes for every sample, so a pass using depth `load` receives the real preceding depth buffer rather than its own previous output.

Sampled boundary queries now reject brick coordinates outside exact occupied bounds before probing the sparse hash table. Three previously reserved header words store these bounds, without increasing the field allocation. Empty, legacy or unrepresentably large fields keep the original lookup. This avoids unnecessary normal-sampling lookups across geometry far from the localized diving-board boundary.

With the corrected pass replay, the same M1 Max, 1000×720 camera, DMC mesher, 6.25 mm visual cells, 50 mm fluid cells and full PBR give:

| Isolated mesh draw | Median | 95th percentile |
| --- | ---: | ---: |
| Before continuous stream | 39.65 ms | — |
| Continuous stream | 16.19 ms | 16.32 ms |
| Stream and bounded boundary lookup | 15.60 ms | 15.79 ms |

The combined improvement is **60.7%**. Each isolated timing uses four warmups and twelve GPU timestamp samples. The additional 0.59 ms boundary-lookup difference is small relative to desktop variability; the continuous stream accounts for the clear majority of the gain. Final dry frame median is 111.13 ms over a short three-second window; this is still GPU-limited and excludes water and fluid simulation. It does not establish live simulation FPS or a return to the earlier two-millisecond mesh cost. The 7,891,528 drawn records, 15,823,181 live records and 1 GiB mesh arena remain unchanged.

GPU validation was enabled for the final captures and reported no errors. The final image is **bit-identical in linear color and depth**, with zero changed pixels after the display transform. Evidence: [final timing](../artifacts/pool-primary/final.log), [frame](../artifacts/pool-primary/final.png), and [comparison](../artifacts/pool-primary/final-comparison.json). Explicit GPU tests verify stream coverage/interpolation and bounded versus legacy boundary samples/normals. `npm run check:types` passed; `npm run test:unit` passed with 918 passing and 46 skipped tests. Grass appearance is unchanged; its separate roughly five-millisecond cost remains a possible later target.

For direct pass attribution, run `tools/benchmark-svo-dry-frame-gpu.ts` with the same scene/mesher settings and:

```sh
export FLUID_SVO_DRY_FRAME_PRIMARY_TIMING=1
export FLUID_SVO_DRY_FRAME_PRIMARY_TIMING_LABELS='Voxel surface mesh rasterization,Imported native voxel field surfaces'
export FLUID_SVO_DRY_FRAME_TIMING=gpu
export FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=3
```

The result JSON includes both isolated pass timings. Local diagnostic evidence is in `artifacts/pool-primary/`.

## Sample coverage culling

The 7.9 million records passing back-face and frustum checks are not all raster-visible. Most fine DMC triangles in the overview fall between the single-sample primary pass's pixel centres. The compute culler now projects each candidate triangle with the draw's camera transform and rejects it only when its expanded bounds or edge tests exclude every possible sample. Small triangles that cover even one sample remain drawable. Bounds spanning more than two by two samples, near-plane crossings and numerically large projections go to hardware without the edge test.

A 1/16-pixel guard in both axes conservatively accounts for rasterizer vertex snapping and floating-point projection differences. Vulkan requires at least [four fractional bits of subpixel precision](https://docs.vulkan.org/refpages/latest/refpages/source/Required_Limits.html); the regression also checks the actual Metal rasterizer used for these measurements. This is not an area threshold or mesh simplification. Each frame re-evaluates coverage for the current camera and viewport, including after movement or zoom. The full 6.25 mm SVO, DMC geometry, normals, PBR identities and shadow casters remain available. Neither the 50 mm fluid grid nor the mesh's 1 GiB retained allocation changes.

The overview's actual draw census changes as follows:

| Material group | Before | After |
| --- | ---: | ---: |
| Foliage | 3,939,763 | 213,322 |
| Pool/deck | 2,208,005 | 225,239 |
| Tree/shrub structure | 935,754 | 49,636 |
| House wall | 336,866 | 52,102 |
| Guests | 297,672 | 16,199 |
| Other | 173,468 | 11,975 |
| **Total drawn records** | **7,891,528** | **568,473** |

The first validated overview draw measured **1.64 ms** (95th percentile 1.70 ms), with bit-identical linear color and depth against the previous implementation. The retained live mesh still contains 15,823,181 records. This reduces the draw list rather than permanently deleting geometry. The SVO's spatial hierarchy and per-brick mesh ranges remain possible accelerators for candidate discovery; no coarse geometric LOD is enabled by this change.

The final paired runs use the same settings and validation-enabled Metal device as above, with four warmups and twelve timestamp samples per view:

| View / stage | Reference median | Optimized median | Optimized drawn records |
| --- | ---: | ---: | ---: |
| Overview mesh draw | 15.53 ms | 1.70 ms | 568,473 |
| Orbit mesh draw | 22.61 ms | 1.70 ms | 584,026 |
| Close mesh draw (camera distance 6.25 m) | 6.62 ms | 1.64 ms | 566,829 |
| Return to overview mesh draw | 15.27 ms | 1.70 ms | 568,473 |
| Overview culling compute | 2.69 ms | 2.69 ms | — |

The overview submits **92.8% fewer records** and its draw is **89.0% faster**. Culling cost did not increase measurably. The reference orbit has desktop-contention outliers; the return measurement confirms the overview result. All four paired captures have **zero changed linear-color, display-color or depth pixels**, and both runs report no GPU validation errors. [Paired comparison](../artifacts/pool-coverage/comparison.json), [reference timing](../artifacts/pool-coverage/reference.log), [optimized timing](../artifacts/pool-coverage/optimized.log), [overview](../artifacts/pool-coverage/optimized.png), [close view](../artifacts/pool-coverage/optimized.close.png).

Short three-second dry frame windows measured 119.38 → 99.38 ms median under variable desktop load. Lighting and imported-field rendering remain substantial; these measurements exclude fluid simulation and water compositing, and the roughly two-millisecond result refers specifically to the primary mesh draw at 1000×720. It is not a claim of two-millisecond total frames or all-camera/all-resolution performance.

The GPU coverage regression checks 37,548 triangles across 17×19, 64×64 and 1000×720 viewports against actual hardware fragment coverage. It includes both windings, perspective, tiny sample-covering triangles, triangles between samples, raster snapping boundaries, degenerate edges and near-plane/eye crossings. Every hardware-covered triangle must survive the compute test.

For paired checks, `FLUID_SVO_DRY_FRAME_MESH_SAMPLE_CULLING=0` keeps the reference draw and `=1` enables sample rejection. `FLUID_SVO_DRY_FRAME_MESH_CAMERA_CHECKS=1` saves orbit, close and return captures beside `FLUID_SVO_DRY_FRAME_CONFIGURED_RAW_OUT`. The isolated pass profiler also accepts compute labels; include `Voxel mesh culling` alongside `Voxel surface mesh rasterization` to account for the culling cost. Local evidence is in `artifacts/pool-coverage/`.

`npm run check:types` passed. `npm run test:unit` passed with 920 passing tests and 47 skipped. Explicit GPU checks for sample coverage, stream parity and raster AO passed six tests with no skips.

## Raster + AO attribution and shadow proxy culling

The live panel's roughly 31 ms Primary visibility total with only 5 ms attributed to primary rasterization was a reporting defect. Imported native fields execute in a separate pass, timed as `scene-primitive-visibility`. The mesh configuration marked that stage's row unavailable, suppressing its displayed duration while the band still included it. The panel now labels this row **Imported asset surfaces** and reports its cost. A regression test reproduces the hidden 25.6 ms interval. The former analytic-tier toggle does not control this pass and is no longer offered on this row.

Raster + AO also differs from the cone-lighting measurements above: its shadow cache retains native-asset proxy mesh triangles. Camera fragments discard these proxies because native fields provide their visible surfaces, but they previously still incurred camera culling and vertex work. The camera culler now rejects these records immediately. Shadow collection continues to use the complete arena, preserving its casters.

Validated Metal runs on the same M1 Max at 1000×720, 6.25 mm visual cells, 50 mm fluid cells and full PBR measured:

| Metric | Before | Final |
| --- | ---: | ---: |
| Camera draw records | 1,512,559 | 568,473 |
| Isolated mesh draw median | 3.41 ms | 1.84 ms |
| Isolated culling median | 5.57 ms | 5.11 ms |
| Isolated native imported surfaces median | 11.93 ms | 16.25 ms |
| Dry frame median | 25.21 ms | 27.64 ms |

Each isolated pass uses four warmups and twelve samples. The final run had substantial variability: culling ranged from 4.06 to 9.57 ms, and the unchanged native pass from 10.94 to 19.40 ms. The mesh draw reduction is clear, but these short runs do **not** demonstrate an overall frame-rate improvement. Imported field rendering remains the largest measured pass and needs further optimization; fixing its reporting does not remove that cost. These dry captures exclude water and fluid simulation and use a different viewport/refinement configuration from the user's screenshot.

The final capture is bit-identical to the baseline in linear color, display color and depth, with zero GPU validation errors. Visual resolution, source normals, PBR and shadows are unchanged. The retained 32,412,747 live mesh records and 2 GiB arena also remain unchanged; this improvement reduces camera work, not shadow storage. Local evidence: [baseline](../artifacts/pool-native/before.log), [final](../artifacts/pool-native/final.log), [comparison](../artifacts/pool-native/comparison.json), [frame](../artifacts/pool-native/final.png).

The benchmark now accepts `FLUID_SVO_DRY_FRAME_CONE_TRACING=raster-ao`, selecting raster AO and disabling the incompatible voxel light cache. Use mesh primary, split shading and `FLUID_SVO_DRY_FRAME_CONE_FANOUT=0` to match production. Include `Voxel surface mesh rasterization,Voxel mesh culling,Imported native voxel field surfaces` in `FLUID_SVO_DRY_FRAME_PRIMARY_TIMING_LABELS` to expose all three costs. `npm run check:types` and `npm run test:unit` passed (921 passing, 48 skipped).
