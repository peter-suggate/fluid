# DMC switching on the ten-times garden

Measured 2026-09-11 on Apple M1 Max / Dawn Metal, `hero-garden-hose-x10`, requested refinement 3, raster primary, flat surfaces, 400×240, GI/shadows/AO disabled. These are construction and mesh-publication timings, not end-to-end browser interaction timings. Shader compilation is separate.

| Phase | Before | After |
| --- | ---: | ---: |
| World construction and initial GPU publication | 42,965 ms | 8,403 ms |
| Mesh extraction through ready raster receipt | 1,579 ms | 485 ms |
| Actual refinement built | 1 | 1 |
| Required triangles | 2,677,320 | 2,677,320 |

Full image, packed surface, identity, and hardware-depth hashes match exactly. Both runs finish with one build, no overflow and a ready raster publication. Results and captures are under `artifacts/svo-dmc-x10-performance/{before,after}`.

## Causes and changes

Uniform dual meshing cannot use the ordinary mesher's coarse terminal leaves. At refinement 3 the old path built millions of nodes before failing its node-index capacity check; at refinement 2 it built the topology and payload buffers before failing the maintenance storage limit. Both attempts were discarded. The existing resolution fallback eventually chose refinement 1.

The new preflight counts the already selected finest occupancy keys, stopping as soon as fitted samples, candidate slots, dirty records and the edit reserve exceed the binding limit. Impossible rungs fail before planning or allocating the octree. This is a lower bound; the final complete-arena check remains authoritative. The existing fallback policy and resolution limits are unchanged. Geometry sampling and fitting remain GPU work.

DMC extraction previously walked the octree eight times for every dual cell, in both count and emit passes. A slice now resolves its eight neighbouring brick payloads once, then directly addresses the cells inside them. Only the required slice jobs are dispatched. Missing/coarser neighbours are still rejected; the face decider, fitted samples, triangle packing and shading are unchanged. Tests cover both a single brick and eight tiled bricks, including seam-crossing sharp and thin geometry.

A separate transition bug applied the requested mesher to the old producer while the replacement was still building. The old producer had no matching fitted attachment, so the shader invalidated its ready mesh while the status retained the previous completed brick count. The renderer now retains the current meshing mode until the replacement producer is ready; other render controls remain live.

## Reproduce

Run exclusively, with no active browser GPU scene:

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
FLUID_SVO_DRY_FRAME_SCENE=hero-garden-hose-x10 \
FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT=3 \
FLUID_SVO_DRY_FRAME_ALLOW_RESOLUTION_FALLBACK=1 \
FLUID_SVO_DRY_FRAME_SURFACE_MESH=1 \
FLUID_SVO_DRY_FRAME_MESHER=dual-marching-cubes \
FLUID_SVO_DRY_FRAME_SURFACE_STYLE=voxel-flat \
FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary \
FLUID_SVO_DRY_FRAME_SHADING=split \
FLUID_SVO_DRY_FRAME_WIDTH=400 FLUID_SVO_DRY_FRAME_HEIGHT=240 \
FLUID_SVO_DRY_FRAME_WARMUPS=12 FLUID_SVO_DRY_FRAME_CYCLES=4 \
FLUID_SVO_DRY_FRAME_GI=0 FLUID_SVO_DRY_FRAME_SHADOWS=0 FLUID_SVO_DRY_FRAME_AO=0 \
FLUID_SVO_DRY_FRAME_OUT=/tmp/dmc-x10/result.json \
node --import tsx tools/run-webgpu-exclusive.ts --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

Resolution fallback is opt-in in this benchmark and the result records requested and actual refinement separately. The benchmark now waits for a ready raster receipt, services capacity growth between frames, and fails on a blocked mesh rather than timing an unfinished fallback after a fixed warmup count.

## Validation

Eight focused tests pass (DMC/DC Dawn shapes, scheduler, reconstructed lighting receivers, mesher persistence/transition and capacity preflight). The five G-buffer/normal contract tests also pass. TypeScript reports existing errors outside the changed files.

The required `npm run test:dawn:sparse-cm12` gate is not green: symmetry, hydrostatic adaptivity and the min8 region surface lane fail, and the suite exhausts its 180-second budget before completing the remaining lanes. Mini32 correctness/performance pass. Full output is retained in `artifacts/svo-dmc-x10-performance/cm12-gate.log`; no lane or timing ceiling was changed.

## Browser switch

The live in-app browser was opened at the same x10 scene, requested refinement 3, voxel faces, raster primary. Its initial ready mesh had exactly **604,871 bricks**, build 3, matching the reported starting state. Clicking **Dual Marching Cubes** retained the ready old mesh while construction ran. The replacement reached **Mesh ready**, **73,998 / 73,998 bricks**, with DMC selected, by the observation at **36.654 seconds** after the click. This is an observed upper bound, not an instrumented exact completion time. Browser console errors/warnings were empty at the readiness check. The temporary tab was closed after capturing the result to release the GPU.
