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
