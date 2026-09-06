# Hero garden primary throughput — 1920×1080

The 2× primary-throughput target was **not achieved**. The current renderer
measures 45.88 ms for primary visibility in the initial baseline. Compute
prototypes measure 36–38 ms, about **1.2–1.3× throughput** in matched runs,
before integration costs. A priori tile ordering adds at most 1.8% in this study. The
shipping renderer is unchanged. The retained changes are opt-in profiling tools.

Measured 2026-09-05 on Apple M1 Max / Metal, macOS 26.6.2, with the catalog scene
`hero-garden-hose` (the requested garden-hose hero), its default camera,
1920×1080 physical pixels, authored refinement depth 3, canonical parametric
traversal, split shading, occupied-brick bounds and contour slabs enabled.
Cone scale is 0.5, full-resolution relighting, cone fan-out off. No resolution,
scene-detail, or lighting reduction earns any reported gain. This is the dry
renderer; simulation and the separate water rendering are outside its scope.

## Measured experiments

| Experiment | Primary GPU median | Throughput | Result |
| --- | ---: | ---: | --- |
| Production baseline, 40 samples | 45.88 ms | 45.20 Mpix/s | Reference |
| Predicated voxel-DDA axis updates | 45.74 ms | 45.33 Mpix/s | Noise; removed |
| Existing one-bit occupancy payload | 44.11 ms | 47.01 Mpix/s | Small gain; no default change |
| Node-index-only continuation stack, full 32-entry capacity | 59.57 ms | 34.81 Mpix/s | Regression; removed |
| Production recheck, 60 samples | 45.81 ms | 45.27 Mpix/s | Baseline reproduced |
| Compute, 8×4 workgroup, recheck | 38.73 ms | 53.54 Mpix/s | Prototype |
| Compute, 8×8 workgroup, recheck | 37.88 ms | 54.74 Mpix/s | Best confirmed prototype |
| Compute, 16×8 workgroup, recheck | 40.24 ms | 51.53 Mpix/s | Prototype |
| **2× target** | **22.94 ms** | **90.40 Mpix/s** | **Not reached** |

The first compute experiment measured 36.24 ms (1.26×), but the final probe also
models raster reverse-Z miss rejection and its repeated result is 37.88 ms.
Use approximately 1.2× as the supported gain, rather than selecting the fastest
single run. Workgroup-size ordering varies across runs; none approaches 2×.

Baseline complete-frame GPU time was 77.46 ms; the recheck was 78.51 ms. These
are measurements of the production frame. The compute prototype is **not**
integrated into that frame, and no complete-frame compute gain is claimed.

## Measurement and correctness boundaries

`FLUID_SVO_DRY_FRAME_PRIMARY_TIMING=1` routes the actual production primary
render pass into its own command buffer after warmup. All surrounding passes
are recorded but not submitted. Every sample retraces the primary rays against
the warmed, unchanged scene and entry seed. The one-pass command buffer makes
its GPU timestamp span attributable, avoiding Apple's overlapping multi-pass
render timestamp windows. It excludes entry preparation and lighting. Each
submission completes before the next sample; samples are retained in JSON.

`FLUID_SVO_DRY_FRAME_PRIMARY_COMPUTE_PROBE=1` captures public WebGPU shader,
pipeline, and binding descriptors, then calls the production primary entry's
body from a compute wrapper. It tests 32-, 64-, and 128-thread workgroups. The
probe writes all outputs to a 48-byte-per-pixel buffer; the raster attachments
use 28 bytes per pixel. Its timing excludes the depth bridge and production
storage-texture integration. It is an architectural experiment, not a shipping
renderer mode.

## Hardware utilization

A three-second Instruments capture attached **after** construction recovered
42 complete frames. Primary visibility has exact encoder attribution and
averages 45.67 ms, agreeing with the isolated-pass benchmark.

| Primary-pass counter | Measurement |
| --- | ---: |
| Fragment occupancy | 15.61% |
| ALU utilization | 26.21% |
| GPU last-level cache utilization | 29.69% |
| GPU read bandwidth | 3.99 GB/s |
| GPU write bandwidth | 13.31 GB/s |
| Peak/mean partition occupancy | 1.049× |

The frame's GPU busy fraction is 96.61%, with 2.70 ms mean gaps. The GPU is
being kept busy, but its primary shader is not filling execution capacity.
Near-balanced partition occupancy does not support simple distribution across
partitions as the missing 2×. Combined with the experiments, this is consistent
with dependent traversal and limited latency hiding; it does **not** isolate
register pressure as the proven cause. The index-stack regression is especially
important evidence against treating smaller private storage as a sufficient fix.

These are sampled counters, not a fully uncontended machine benchmark.
WindowServer, Grok Bot, and Codex also issued GPU work. The report excludes
overlapping activity from counter attribution: 341 primary samples cover
8.72% of primary GPU time; exclusive coverage across the frame is 18.73%.
The aggregate counter values therefore characterize those uncontended windows,
not every instant of the primary pass. Absolute timings are under this desktop
workload; the repeated production baseline remains close (45.88 → 45.81 ms).

The earlier construction-inclusive capture could not finalize and was not used
for these counter claims. The successful attached trace, HTML report, and
summary are in `artifacts/hero-utilization-2026-09-05/counters-1080/`.

## Output comparison

The final probe compares every pixel against the configured raster pass:

- Hit/miss coverage: **0 differences in 2,073,600 pixels**.
- Packed surface: 423 differing pixels (0.0204%).
- Material/owner/media identity: 55 differing pixels (0.00265%).
- Hardware depth: 477,390 differing values; maximum absolute difference
  0.00000539422 in reverse-Z depth units.

Compute constructs UVs explicitly while raster interpolates them. Shader-stage
floating-point evaluation can therefore change crossings and surface choices;
the measured differences must be resolved or visually bounded before promoting
this path. Zero missing surfaces alone is not proof of identical shading.
The probe models the raster pass's greater-than-zero depth test, and compares
the identity attachment as four 16-bit integers.

Predicated DDA and occupancy-mask experiments both preserve the full reference
image and all reference G-buffer hashes. The index-stack experiment changes
34,070 depth values but loses/gains no hit pixels; it is rejected on performance
regardless. Failed shader experiments were reverted, with their patch retained
under the artifact directory.

## What the experiments imply

Moving the same work into compute improves throughput, but does not explain a
2× opportunity by itself. Removing three DDA branches has no measurable payoff;
replacing empty-cell material reads with occupancy bits buys little. Reducing
stack storage by recomputing intervals costs more than it saves. These results
argue against shipping local loop or stack changes on intuition alone.

The current work map points primarily at hierarchy traversal: 48,857,116 node
visits versus 4,040,705 voxel-cell steps (92.36% versus 7.64% of their combined
count). These operations do not have equal cycle cost, but the counts weaken
the case for prioritizing voxel-DDA queues or occupancy skipping. Investigate
sharing hierarchy work across neighboring rays and tighter entry bounds next.
Neither is demonstrated to achieve 2×. The confirmed compute result still
requires another 1.65× throughput gain to reach the target.

## Long rays and a priori scheduling

At 1920×1080 the combined node-plus-cell work per ray has mean 25.51, p99 72,
and maximum 199. The longest ray does about 7.8× the mean counted work, but the
most expensive 1% of rays account for only 3.21% of useful work. Expensive rays
are spatially clustered: an 8×4 footprint achieves 89.87% modeled SIMD lane
efficiency (sum of lane work divided by 32 times each group's maximum).
This is a work-count model, not measured hardware occupancy.

A zero-overhead, stable survivor-compaction model saves only 4.04% of scheduled
work with an eight-operation quantum, or 1.94% with a sixteen-operation quantum.
The former requires 25 rounds and 5.72 million ray continuations. Even impossible
perfect packing has only a 1.113× bound under this model. This does not justify
indirect dispatch and state traffic for this scene. Globally shuffling individual
rays reduces modeled lane efficiency to 38.22%; preserve local ray coherence.

The benchmark-only scheduling experiment keeps coherent 8×8 tiles and uses a
129,600-byte permutation table inside one direct dispatch. It compares raster,
Morton, spatially spread, and expensive-first orders, then repeats raster to
check drift. All schedules use the same compiled pipeline and trace all rays.
The expensive-first order uses this fixed-camera work map as an offline oracle;
map capture and table construction are excluded from timing. This tests an
optimistic cost hint, not a production predictor for moving views. Hardware
may execute workgroups out of index order, so the table supplies a hint rather
than enforcing a start order.


| A priori 8×8 tile order | Primary GPU median |
| --- | ---: |
| Raster | 36.700 ms |
| Morton | 36.045 ms |
| Spatially spread | 38.207 ms |
| Expensive-first, offline oracle | 36.635 ms |
| Raster repeated | 36.700 ms |

All five schedules produce byte-identical compute outputs across every pixel;
the previously documented compute-versus-raster differences remain unchanged.
Each case used 12 warmups and 60 samples. Morton improves 1.8% over its matched
raster-order control; expensive-first improves only 0.18%, while spreading tiles
regresses 4.1%. The bracketing control is stable. These are small gains, not a
route to 2×: even the oracle fails to uncover a substantial scheduling benefit.
This run's raster primary measured 47.251 ms, so the best scheduled compute
prototype is 1.31× within this run, still excluding production integration.
Cross-run compute variation makes approximately 1.2–1.3× the supported range.

Reproduce scheduling by adding
`FLUID_SVO_DRY_FRAME_PRIMARY_SCHEDULE_MAP=artifacts/hero-utilization-2026-09-05/work-map-1080/report.json`
to the command below. Capture a corresponding work map first using
`FLUID_SVO_DRY_FRAME_PRIMARY_WORK_MAP=1` with the same camera, scene and resolution.
Results and raw outputs are in `artifacts/hero-utilization-2026-09-05/apriori-1080/`.


## Reproduction

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal \
FLUID_SVO_DRY_FRAME_SCENE=hero-garden-hose \
FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT=3 \
FLUID_SVO_DRY_FRAME_WIDTH=1920 FLUID_SVO_DRY_FRAME_HEIGHT=1080 \
FLUID_SVO_DRY_FRAME_BRICK_OCCUPANCY=bounds \
FLUID_SVO_DRY_FRAME_TIMING=gpu \
FLUID_SVO_DRY_FRAME_PRIMARY_TIMING=1 \
FLUID_SVO_DRY_FRAME_PRIMARY_COMPUTE_PROBE=1 \
FLUID_SVO_DRY_FRAME_WARMUPS=12 FLUID_SVO_DRY_FRAME_CYCLES=60 \
FLUID_SVO_DRY_FRAME_OUT=/tmp/hero-primary/report.json \
node --import tsx tools/run-webgpu-exclusive.ts \
  --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

Artifacts: `artifacts/hero-utilization-2026-09-05/`. `comparison.json` summarizes
the five runs; each run retains all timing samples. `compute-confirm-1080`
contains the final full-population comparisons and raw output buffers.

The earlier 800×460 study is not a current-source baseline: today's authored
depth-3 reconstruction allocates capacity for 230,388 leaves and 117,958,656
voxels, versus the earlier report's 14,153 leaves and 7,246,336 voxels. Keep the
current scene, camera, refinement, and source fixed when comparing timings.

Seven targeted renderer tests pass. The GPU benchmark and compute variants
complete without validation errors. Repository-wide TypeScript checking is
blocked by unrelated solver/harness errors; none were reported in these tools.
No simulation, sparse-world topology, or presentation publication code changed.
