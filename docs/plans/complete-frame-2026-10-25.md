# Complete-frame performance checkpoint

The 8.33 ms target is **not yet met during camera motion**. All retained changes
keep full-resolution material shading, water rendering, AO, the same authored
lights and their configured sample budget. The shipping visibility lattice stays
at half width and height. Neither fewer lights nor lower output resolution is
used to obtain the results below.

## End-to-end measurements

Apple M1 Max, macOS 26.6.2; isolated in-app browser at **3474 × 1882** backing
pixels. Chrome tabs are not used. Scene `reservoir-seep`, camera distance
2.3710241428319203 m, azimuth 0.22, elevation 0.28, target
(0.13258204727927497, 0.5496532146494025, -0.009815988207894755).
The scene is paused at time zero, matching the supplied screenshot. Its water
is rendered every frame. These are presentation results, not physics throughput.

| Configuration | Completed FPS | Mean time per completed frame |
| --- | ---: | ---: |
| Starting implementation, stationary | 62.70 | 15.95 ms |
| Exact visibility reuse + queued exceptions, stationary | 119.37 | 8.38 ms |
| Queued exceptions, moving camera | 65.12 | 15.36 ms |
| Compact atlas + bounds + shared mapped samples, moving camera, repeat | 64.36 | 15.54 ms |

Each run warms 120 completed frames and then measures 360 completions over wall
time. The count comes from actual presentation completion, including worker
delivery, GPU work, and browser scheduling. It does not measure physical display
scan-out. Completion callbacks can arrive in batches; the saved interval
distribution is therefore not a GPU frame-time distribution. Aggregate
throughput uses total completions divided by elapsed time.

The stationary result is a **1.90× throughput improvement** from exact reuse.
It is close to the display cadence but does not prove a strict sub-8.33 ms mean.
The moving-camera results do not establish a material end-to-end improvement.

An initial 116.7 FPS moving run overlapped asynchronous pipeline compilation and
is explicitly excluded. The repeat with the completed scene was 64.36 FPS. The
benchmark now requires an active renderer, ready mesh, matching effective
lighting mode, and ready GPU status before beginning its warmup; losing readiness
discards the current sample window. Results include effective renderer state and
quality settings. Earlier measurements predate that automatic readiness gate.

## Changes retained at this checkpoint

- Full-resolution reconstruction queues receivers with no valid reduced sample.
  Only that compact queue runs exact AO and light visibility. The common resolve
  shader no longer carries exact traversal's private stacks and register demand.
  The queue covers every pixel, handles more than 65,535 workgroups, and resets
  each frame. A reference inline path remains available for comparison.
- Unchanged paused, body-free receiver inputs reuse completed visibility. The
  complete view uniform, render parameters, scene/source publication, shadow
  generation, and settled mesh receipt govern reuse. Resize, changed publications,
  camera changes, or withdrawn eligibility invalidate it. Running simulation,
  body corrections, water-shadow variants, and stage ablations refuse reuse.
  Geometry, water, material shading, optical composition, and presentation still
  execute.
- A bijective atlas permutation places reduced samples in a contiguous rectangle.
  Other pixels retain unique locations for exact exceptions, including odd-sized
  viewport edges. RGB10 values and sample positions are unchanged.
- Ordinary lattice receivers copy their own AO/sun guide directly. Other pixels,
  seam receivers, and clamped even border pixels retain bilateral reconstruction.
- Mapped spot emitters with multiple samples share their map filter only when
  no body or water correction can make it sample-dependent. Exact lights retain
  independent rays. The reservoir's rectangular softbox is **not** a mapped spot
  and does not receive this shortcut.
- Raster AO now enables existing conservative occupied-brick bounds. A separate
  full-image comparison found identical HDR/depth bytes with bounds on and off
  in the authored and mixed-light fixtures, stationary and orbiting.

The light count/sample budget is unchanged: support for each light kind does
not imply constant cost for unlimited lights. The existing eight-sample budget
still applies, and point/sphere/excess/unsupported mapped lights can require
exact traversal.

## Remaining bottlenecks

A 31-sample instrumented moving-browser run before the last layout changes had
a median complete GPU span of 12.98 ms. Median stage contributions included
5.77 ms visibility/AO, 2.56 ms material shading, 0.98 ms mesh drawing, 0.46 ms
mesh culling, and approximately 1.1 ms across water interfaces, optical
composition, overlays, and presentation. Per-stage medians do not sum to the
median overall span; render-pass timestamp boundaries also overlap GPU work.

The isolated dry renderer makes the exact-light problem explicit. At the closer
fixture camera, mixed directional/rectangle/point/spot/sphere lighting cost about
143 ms with the compact atlas and occupancy disabled, versus about 115 ms with
occupied-brick bounds. Images were byte-identical, but this remains far outside
budget. The visibility producer alone accounts for about 102 ms with bounds.
These dry figures exclude water and browser scheduling and must not be presented
as complete-frame results.

The next optimization needs to avoid repeated exact visibility work for these
finite lights while retaining their occluders and edge quality. Smaller dispatch
grids and more parallel emitter dispatches did not demonstrate a general win.
The coarse three-pixel lattice and parallel-emitter experiment were removed.

## Reproduction and evidence

Raw browser runs, dry paired timings, configuration snapshots, and excluded-run
reasons are in `complete-frame-2026-10-25.json`. Temporary full HDR/depth captures
and PNGs are under `/tmp/fluid-complete-frame/`.

Open the scene with `?frameBenchmark=checkpoint&frameBenchmarkMotion=orbit`
alongside its normal scene/camera parameters. The visible benchmark button starts
the capture; the second button switches stationary/orbit motion. Opening the
Frame instrument adds the existing stage traces, so use separate uninstrumented
runs for throughput.

The dry comparison uses `tools/benchmark-svo-dry-frame-gpu.ts` with
`FLUID_SVO_DRY_FRAME_VISIBILITY_COMPARE=1`,
`FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=1`,
`FLUID_VISIBILITY_QUEUE_COMPARE=1`, and
`FLUID_VISIBILITY_REUSE_CHECK=1`. Select raster primary, surface mesh,
`dual-marching-cubes`, raster AO, specialized lighting, GI off, occupancy bounds,
3474 × 1882, and `FLUID_VISIBILITY_CAMERA_SCALE=0.624`. The benchmark reports
both its scope and exact camera; its camera target differs from the browser
screenshot camera, so those timings are not interchangeable.

The required repository checks are `npm run check:types` and
`npm run test:unit`. GPU regressions additionally cover the queue, RGB10 packing,
specialization eligibility, and atlas bijection. Paired image tests require
identical depth, receiver display RMSE below 1/255, and differences above 8/255
on fewer than 1% of receivers. Exact cache reuse is checked for HDR/depth bit
equality and camera-change invalidation.
