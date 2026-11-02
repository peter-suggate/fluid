# Pool Party: main SVO rendering experiments

2026-11-02, Apple M1 Max / Dawn Metal. This study uses the ready surface mesh,
raster AO, split shading, and specialized opaque lighting of
`backyard-pool-party`, at 1280×720 with four camera poses. Environment refinement
is the renderer default (1). It excludes fluid simulation and water composition.
There is no brick fallback benchmark in this study. The earlier fallback changes
were removed.

## Method

`tools/benchmark-svo-main-kernels.ts` runs the actual renderer and replaces only
selected active compute pipelines. It asserts that the surface mesh is ready
and the mesh rasterization and raster AO passes execute. Scene geometry,
materials, and quality settings are identical between A and B. Lighting is
invalidated each frame to measure a moving-camera workload rather than cached
idle presentation. Final rgba16float color and depth are compared bit for bit
at four poses. Candidates that change pixels are not eligible for promotion.

Use the shared timing lock, paired ABBA/BAAB ordering, two warmup rounds, and
hardware timestamps with quantization disabled. The guarded run rejects rounds
that overlap other repository GPU benchmark processes. Eight frames per timed
submission reduce submission noise; ready-state telemetry is suppressed equally
for both variants in these batches because that telemetry assumes one encode per
submission. Section 10 uses normal one-frame submissions. Its CPU encode and
submission-to-completion measurements include instrumentation overhead and are
comparative, not application frame latency. No claim about register counts,
spills, cache misses, or physical DRAM bytes is made without hardware counters.

The initial 20-way screen below used mean within-round candidate/baseline
elapsed-time ratios with a 2,000-resample bootstrap 95% interval. Its samples
showed an alternating submission warmup effect. The final confirmation and
current runner cross each camera pose over both ABBA and BAAB orders, pool
eight-round blocks, and bootstrap those blocks instead. At least 32 rounds are
needed for the runner to report an interval; use 64 or more for confirmation.
Intervals measure sample variation, not independence from every system effect. Full-frame elapsed time is
measured directly; render-pass attribution windows can overlap on this backend
and must not be summed. Separate compute attribution is diagnostic only.
Earlier exploratory runs overlapped a fluid benchmark outside the shared lock;
their small timing differences are not evidence of improvements.

```sh
FLUID_SVO_MAIN_KERNEL_STUDY=artifacts/svo-main-kernels-new-run \
FLUID_SVO_KERNEL_CYCLES=64 FLUID_SVO_KERNEL_BATCH=8 FLUID_SVO_KERNEL_GUARD=1 \
tools/gpu/fluid-gpu-run.sh timing \
node --import tsx tools/benchmark-svo-main-kernels.ts
```

Set `FLUID_SVO_KERNEL_VARIANTS` to comma-separated experiment names for a repeat.
Set `FLUID_SVO_DRY_FRAME_WIDTH` / `HEIGHT` to exercise viewport boundaries.
Results include raw samples, image hashes/errors, actual pipeline labels,
per-pass attribution, and readback counts. Captures and generated WGSL sit beside
`results.json` in the selected artifact directory.

## Results

The guarded full-frame run completed **20 alternatives covering all ten sections**.
Seventeen retained bit-identical color and depth at all four poses. Three were
rejected for color changes; no experiment changed depth. No candidate has a
statistically established whole-frame speedup in this run. Positive percentages
below mean slower; intervals are paired bootstrap 95% intervals.

| Section | Experiment | GPU time change | 95% interval | Exact image parity |
| --- | --- | ---: | ---: | --- |
| 1 | `layout-aos` | +0.25% | -2.73 to +3.76% | yes |
| 1 | `layout-quad-tiled` | +0.24% | -3.83 to +4.76% | yes |
| 2 | `bandwidth-late-shadow` | +0.68% | -2.80 to +4.33% | yes |
| 2 | `bandwidth-representative` | +0.77% | -2.26 to +3.91% | yes |
| 3 | `active-tiles` | +3.95% | +1.83 to +6.60% | yes |
| 4 | `divergence-row-coherent` | -1.89% | -5.50 to +2.10% | yes |
| 5 | `fuse-exact-contact` | +0.08% | -2.24 to +2.63% | **no** |
| 5 | `fuse-range-classification` | +1.41% | -1.88 to +4.76% | yes |
| 6 | `split-inline-control` | +9.79% | +6.51 to +13.50% | **no** |
| 7 | `atomics-subgroup-range` | +0.27% | -3.69 to +4.66% | yes |
| 7 | `atomics-global-per-receiver` | +8.07% | +3.24 to +15.16% | yes |
| 8 | `reuse-reload-neighbors` | +0.59% | -2.82 to +4.45% | **no** |
| 9 | `occupancy-8x8` | +2.19% | -0.94 to +5.99% | yes |
| 9 | `occupancy-32x8` | -0.17% | -3.29 to +2.88% | yes |
| 9 | `occupancy-8x32` | +0.56% | -1.82 to +3.45% | yes |
| 9 | `occupancy-exact-32` | +1.63% | -0.38 to +3.95% | yes |
| 9 | `occupancy-exact-128` | +0.58% | -2.70 to +3.88% | yes |
| 9 | `occupancy-exact-256` | -0.78% | -2.98 to +1.09% | yes |
| 10 | `coordination-no-receipts` | +4.56% | -5.77 to +16.78% | yes |
| 10 | `coordination-sync-receipt` | +7.51% | -1.97 to +16.72% | yes |

Raw evidence: `artifacts/svo-main-kernels-guarded/results.json`. Batched frame
baseline medians are approximately 4.5 ms; the single-submission coordination
trials were much noisier, with baseline medians of 10.35–10.84 ms. Compare each
candidate only to its paired baseline, not across those workloads. This is SVO
opaque rendering, not total simulation/application frame time. Active tile counts are 926–930 of
3,600 tiles (about 26%), with 44,303–46,764 exact receivers. The same counts
are preserved by standalone and fused tile classification.

The repeated compute attribution study (`artifacts/svo-main-kernels-attributed/`)
found useful candidates despite inconclusive whole-frame times. Across 32 rounds,
row-coherent scheduling reduced the summed resolve/exact compute mean from
0.4470 to 0.4056 ms. Exact workgroup size 128 reduced the two exact consumers
from 0.3092 to 0.2769 ms; 256 was similar. Range shape 32×8 did not improve its
kernel (0.0506 versus 0.0507 ms). These measurements force encoder boundaries to
make compute timestamps trustworthy; they are not whole-frame speedup claims.
The combination uses 128 lanes to test the two promising changes together.

The combined row-coherent/128-lane candidate preserved all color and depth bits
at 1280×720. Its final 96-round confirmation pooled 12 balanced camera/order
blocks: **−0.52% GPU time**, 95% interval **−1.62% to +0.56%**. Baseline/candidate
medians were 4.5380/4.5038 ms. The interval includes a regression, so this does
not establish a frame-time improvement. Raw evidence is in
`artifacts/svo-main-kernels-combined/results.json`.

**Decision: retain the production renderer defaults.** The 21 tested alternatives
(including the combination) remain benchmark-only. The useful compute reductions
are recorded for future work, but none justifies claiming a faster main render
path from these measurements. No scene representation, geometry, material,
editing, collision, save/load, or undo code changes are introduced.

## 1. Memory layout

Mesh culling follows a compact 4-byte index stream into 32-byte quad records,
then emits 4-byte visible indices. The persistent quad is an AoS record used by
both culling and rasterization. Rewriting that geometry ABI is unnecessary to
experiment with the active screen kernels.

AO range lane `x + 16*y` reads a 16-byte geometry texel at `(2*x+1,2*y+1)`
inside its screen tile, writes contiguous half-resolution importance texels,
and stages position/validity and normal in two separate workgroup arrays.
Neighbor indices are `lane^1`, `lane^16`, and `lane^17`. Thus adjacent lanes
read every other full-resolution texel but write adjacent reduced texels.
Texture physical tiling is backend-defined; these are logical addresses.

The SoA baseline uses two arrays with 16-byte element strides (8 KiB total).
`layout-aos` interleaves each position and normal into a 32-byte receiver record,
keeping 8 KiB. `layout-quad-tiled` permutes the same shared arrays so the four
reused receivers occupy adjacent slots. Both are tested against the actual
neighborhood access pattern, rather than assuming SoA always wins.

## 2. Bandwidth

Logical traffic, before caches and compression:

- Range production: one 16-byte geometry read and one 4-byte importance write
  per reduced sample, or 5 bytes/full-resolution pixel, plus mip construction
  and two global reduction atomics per workgroup.
- Reduced AO/shadow: 8-byte contact and 4-byte shadow outputs per reduced sample;
  horizon geometry taps and shadow traversal add variable reads.
- Visibility: 4 bytes per pixel per allocated light layer (RGB10 already packs
  three channels into one word); the reconstruction guide is 8 bytes/pixel.
- Each exact receiver carries a 4-byte coordinate and 4-byte unquantized AO.
  The queue entails a coordinate write, two coordinate reads, AO write and AO
  read: 20 logical bytes/receiver across the split passes, excluding counters,
  geometry reads, traversal, and final outputs.

`bandwidth-late-shadow` delays each 4-byte shadow tap until its reconstruction
weight is positive. `bandwidth-representative` moves the 16-byte geometry load
below the representative-pixel shortcut: up to one quarter of screen pixels can
avoid it. Mesh geometry and packed material identity are retained. There is no
constant bytes/ray figure: visits, light count, and horizon tap count vary.

## 3. Unnecessary work

`active-tiles` enables the existing occupied-tile worklist and indirect dispatch
for reduced light visibility and edge resolve. Classification scans all four
pixels in each reduced cell so thin geometry is included. Its full scan,
reservation, finalizer, argument copy, and consumption are inside the measured
frame. Mesh culling and exact receivers already use GPU-generated active counts;
allocated quad capacity is not the visible draw count. This scene has 704,453
allocated quad slots, 163,859 live quads, and 33,535 drawn quads in the baseline.
The experiment asks whether skipping empty screen tiles pays for the extra scan.

## 4. Divergence

`divergence-row-coherent` changes each lane's four-pixel schedule. The baseline
processes a 2×2 block per lane (adjacent lanes begin two texels apart). The
candidate processes four 8×8 quadrants of the same 16×16 tile, so adjacent lanes
visit adjacent texels at each iteration. It adds no sorting pass or worklist;
the existing exact queue preserves spatial grouping. Geometry tests, bilateral
acceptance, horizon taps, and traversal lengths still diverge. Image comparisons
include the downstream consumers, since queue order changes too.

## 5. Pass fusion

`fuse-exact-contact` removes the exact AO producer and evaluates it inside the
shadow consumer. It eliminates the 4-byte AO write/read round trip and duplicate
receiver setup, but increases the state and work carried by the traversal
kernel. This candidate changes some final pixels and is rejected regardless of
its speed; the numerical cause has not been established.

`fuse-range-classification` instead classifies occupied tiles inside AO range
production, reusing its representative geometry read. It retains the global
finalizer because a workgroup barrier cannot finalize a device-wide list. The
additional three geometry reads per reduced receiver and local flags are
included. Compare it to both the default and standalone `active-tiles` variant.

## 6. Kernel splitting

`split-inline-control` disables the exception queue and performs exact horizon
and shadow work inline in the screen resolve. The baseline includes every cost
of queue generation, finalization, indirect argument transfer, and both exact
consumers. The inline control changes pixels and is not eligible for promotion.
This measures the existing common/exception split rather than proposing another
queue without accounting for its construction.

## 7. Atomics

The shipping exact queue stages coordinates in a 1 KiB workgroup array, reserves
one global range per nonempty tile, then scatters contiguous entries. AO range
already reduces to two global atomics per workgroup. `atomics-subgroup-range`
adds subgroup min/max reduction before updating the two workgroup atomics; it
reduces local contention but cannot further eliminate those two global updates.
`atomics-global-per-receiver` replaces tile staging with a global reservation per
exceptional receiver, directly measuring the existing global aggregation benefit
and the cost of its barriers on this scene.

## 8. Neighborhood reuse

`reuse-reload-neighbors` removes the shared position/normal tile and its first
barrier, fetching and reconstructing each of the three neighboring receivers
instead. That adds up to 48 geometry bytes per valid reduced receiver but saves
8 KiB of shared storage. The 2×2 reuse groups already fit within every tested
workgroup; no halo is needed for this stencil. Long horizon taps have varying
screen radii and do not fit a cheap fixed halo. The reload candidate changes a
few pixels and is rejected. Fluid-cell stencil work is outside this rendering
study.

## 9. Occupancy

Range workgroups 8×8, 32×8, and 8×32 are tested against 16×16. The 8×8 version
reduces receiver storage to 2 KiB but quadruples the number of global reductions.
The other shapes retain 8 KiB and change lane locality. Each retains identical
2×2 neighborhoods and updates the dispatch dimensions.

Exact AO and shadow consumer sizes 32, 128, and 256 are tested against 64.
The GPU finalizer changes with them, so the tests include the real rounded-up
active dispatch rather than incorrectly dropping or repeating receivers. These
are occupancy hypotheses tested through elapsed time; no occupancy percentage
or spill reduction is inferred from timing alone.

## 10. CPU–GPU coordination

`coordination-no-receipts` suppresses ready-state mesh status receipts while
leaving GPU revision checks, culling, and indirect work active.
`coordination-sync-receipt` adds a diagnostic copy/map dependency every frame.
The production receipt is asynchronous, at most every 250 ms once ready.
Build receipts remain prompt. Host-only or unknown publications can rely on
that bounded poll, so removing it universally would be incorrect even if a
fixed-scene benchmark got faster. Dispatch dimensions and visible counts remain
GPU-owned; the 12-byte indirect copies are GPU-to-GPU usage transitions.

In 32 measured frames per arm, the no-receipt comparison recorded 12 baseline
copies versus zero candidate copies; the synchronous control recorded 8 versus
40 (32 forced copies plus 8 ordinary polls). Median encode times were about
0.36–0.37 ms. Median submission-to-completion times were 12.23/11.78 ms without
receipts and 11.77/12.35 ms with synchronous receipts, but the broad GPU intervals
and instrumented waits do not establish an application latency improvement.

## Source and validation

Candidates are isolated in `tools/svo-main-kernel-variants.ts`; capture, parity,
and paired timing live in `tools/svo-main-kernel-study.ts`. Production shader
sources are `lib/svo/features/lighting-visibility/raster-ao-budget.ts` and
`raster-light-visibility.ts`; encoding is in `svo-raster-ao.ts` and
`lib/svo/pipeline/webgpu-svo-dry-scene.ts`.

All 21 alternatives were also exercised at **801×451** with four poses to cover
odd rows, odd columns, and partially occupied workgroups. The same 18 candidates
retain exact color/depth parity at both resolutions; the same three fail color
parity. No candidate changes depth, and both runs finish without GPU validation
errors. The small odd-viewport run is a correctness check, not timing evidence;
its results are in `artifacts/svo-main-kernels-odd/results.json`.

`npm run check:types` passes. `npm run test:unit` passes: 1,447 passed,
264 skipped, zero failures. `git diff --check` passes. The checks do not measure
other GPU architectures, fluid simulation, animated bodies, editing workloads,
or the application's full composition frame.
