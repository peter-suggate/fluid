# SVO primary-ray traversal study — `hero-garden-hose`

Measured 2026-09-04 on Apple M1 Max / Metal. The current-source captures use
commit `cc42fe79734f050f7fffc1ff6399abffb7e0f670` plus the dirty render-source
state recorded in each JSON artifact. The renderer used canonical parametric
traversal, split shading, the primary-entry prepass, brick occupancy `off`, and
refinement depth 3.

## Result in one sentence

Primary visibility is the largest frame stage, and its most expensive internal
unit is the serial fine-cell DDA loop in `traceLeafVoxelPayload`: it performs
11.62 million cell iterations for the 800×460 view, versus 6.01 million SVO
node visits, while a DDA-span-reduction experiment removes 1.90 ms (13.9%) from
the complete frame.

## What was measured

The `primaryWorkMap` diagnostic instruments the actual shipping primary trace,
not a CPU replica. Every fragment writes one `rgba32uint` record containing:

- SVO hierarchy node visits;
- leaf visits;
- fine voxel-cell DDA iterations;
- planar tests, candidate-BVH node visits, and analytic primitive tests;
- entry-prepass seed state, terminal state, and hit-field source.

The counters are exact loop counts. A node visit and a voxel-cell step are not
assumed to have identical instruction cost; their sum is a useful dependent-work
indicator, not a cycle model. The instrumented pass is used only for spatial
analysis. All wallclock numbers come from clean, uninstrumented runs.

Scene structure for this capture: 288×192×192 fine grid, 8³ bricks, maximum
depth 9, 306,972 nodes, 14,153 leaves, 7,246,336 voxels, five rigid bodies, and
385,392,152 allocated renderer bytes.

## 1. The expensive piece

### Exact stage attribution at 1600×920

The 308-frame Instruments capture reports:

| Stage | GPU time/frame | Share of GPU intervals | Attribution |
| --- | ---: | ---: | --- |
| Primary visibility | 22.297 ms | 60.85% | exact isolated encoder |
| Deferred dry lighting | 7.392 ms | 20.17% | exact isolated encoder |
| Primary entry depth | 0.786 ms | 2.14% | exact isolated encoder |
| Other exact and composite intervals | 6.168 ms | 16.84% | partly merged |

Median frame wallclock was 38.90 ms (p10 36.69, p90 41.57). GPU busy time was
36.43 ms, or 93.4% of the 39.00 ms mean frame wallclock. Primary visibility is
therefore 57.2% of elapsed frame wallclock and 60.8% of attributed GPU intervals.

### Internal traversal work at 800×460

| Actual loop | Total iterations | Mean/pixel | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Fine voxel-cell DDA | 11,615,313 | 31.56 | 125 | 186 |
| SVO hierarchy node visits | 6,013,762 | 16.34 | 53 | 75 |
| Leaf visits | 2,023,605 | 5.50 | 20 | 28 |
| Planar tests | 0 | 0 | 0 | 0 |
| Candidate-BVH nodes | 0 | 0 | 0 | 0 |
| Analytic primitive tests | 0 | 0 | 0 | 0 |

The normal primary path therefore spends 65.9% of its counted node-plus-cell
iterations in the in-brick DDA. The scene contains 295 authored primitive
records, but this production path sees their unified voxel representation;
the separate analytic candidate traversal contributes exactly zero work here.

![Node plus voxel work map](../artifacts/svo-primary-experiments/maps/work-items.png)

The white/red bands follow grazing silhouettes: the underside of the upper
garden object, the front pond rim, and thin projecting geometry. These rays
visit many fine cells serially before finding a solid cell or proving a miss.

![Fine voxel-cell DDA iterations](../artifacts/svo-primary-experiments/maps/voxel-cells.png)

### Causal timing check

Enabling the existing occupied-sub-AABB bounds experiment shortens the interval
walked by the same DDA loop:

| 800×460 whole-frame run | Median | p95 |
| --- | ---: | ---: |
| Baseline | 13.697 ms | 16.581 ms |
| Occupied-sub-AABB bounds | 11.796 ms | 12.845 ms |
| Difference | **−1.901 ms (−13.9%)** | **−3.736 ms** |

This is the strongest A/B evidence that DDA span, rather than node-record load
shape, is the dominant removable traversal cost. Earlier in-source controls at
2488×1256 found vectorized node records within noise (24.183→23.921 ms) and a
lean parametric expansion slightly slower (24.183→24.379 ms).

A previous same-scene 800×460 Metal-counter capture gives the matching hardware
signature: 17.0% fragment occupancy, 26.6% ALU utilization, 5.10 GB/s reads,
and 20.10 GB/s writes. That is not a saturated arithmetic or read-bandwidth
kernel; it is consistent with a long dependent fragment loop with poor latency
hiding. This older counter capture is used qualitatively because it predates the
current source snapshot.

## 2. Are a few pixels much slower?

Yes at the individual-pixel and tile level, but not in the form of isolated
one-pixel outliers that alone dominate the frame.

For `node visits + voxel-cell iterations`:

- mean 47.91, standard deviation 40.82, coefficient of variation 0.85;
- p50 46, p90 101, p95 134, p99 174, maximum 257;
- the maximum is 5.36× the mean and 1.48× p99;
- the top 1% of pixels contributes 4.05% of total counted work;
- the top 10% contributes 28.68%, and the top 20% contributes 46.27%;
- the hottest 32×32 tile carries 4.23× the mean tile workload;
- mean active 2×2-quad efficiency is 96.4% when measured as
  sum(work) / 4×max(work), excluding all-zero quads where that ratio is undefined.

The worst pixel is `(461,146)`: 71 node visits and 186 fine-cell steps, for 257
counted work items. Its neighbouring pixels are almost as expensive. Node and
voxel work are also strongly correlated (`r=0.91`), while leaf and voxel work
correlate at `r=0.94`.

![Entry-depth seed and top-one-percent work tail](../artifacts/svo-primary-experiments/maps/entry-seed-and-p99-tail.png)

This distinction matters: the tail is severe, but it forms broad coherent
silhouette bands. Fine-grained SIMD divergence inside 2×2 quads is small;
coarse tile/screen-region imbalance is the more credible scheduling problem.
The entry prepass seeds 77.0% of pixels directly at a leaf and classifies 23.0%
as empty; no pixels remain unseeded.

### What the slowest pixels do to throughput

The slowest pixels look dramatic on a heat map, but they are not poisoning the
whole GPU. To separate useful work from SIMT lane waiting, I replayed the exact
800×460 work map through four plausible 32-lane spatial footprints. Each group
is charged for its slowest lane: `32 × max(node visits + voxel cells)`. This is
an explicit scheduling model, not a new GPU timestamp measurement.

| 32-lane footprint | Baseline lane efficiency | Primary saving if every ray were clipped to p99 | Modeled 1600×920 primary time saved |
| --- | ---: | ---: | ---: |
| 8×4 | 89.5% | 0.645% | 0.144 ms |
| 4×8 | 88.4% | 0.694% | 0.155 ms |
| 16×2 | 86.8% | 0.694% | 0.155 ms |
| 32×1 | 80.7% | 0.781% | 0.174 ms |

So a perfect optimization that removed **all work above the p99 threshold of
174** would improve primary-ray throughput by only about **0.65–0.79%**. Against
the measured 38.90 ms large frame, that is about **0.37–0.45% whole-frame**.
The work numerically above p99 is just 0.416% of all useful node-plus-cell work.

The reason is spatial coherence. In the 8×4 model, p99-tail pixels touch only
199 of 11,500 groups (1.73%), and a touched group contains 18.9 tail lanes on
average. Neighbouring pixels are usually slow together, so few fast lanes wait
for one isolated straggler. Keeping the same work distribution but randomly
shuffling pixel positions collapses modeled 8×4 lane efficiency from 89.5% to
30.2%. The image-space arrangement of the work is therefore helping a great
deal.

As an intentionally impossible upper bound, deleting every top-one-percent ray
entirely removes 4.05% of useful work but only 2.01–2.68% of scheduled group-max
work, depending on footprint. That would save about 0.45–0.60 ms of the 22.297
ms large primary pass, or roughly 1.2–1.5% of the whole frame. The single worst
pixel, at 257 work items beside a 256-work neighbour, raises one group maximum
by only one iteration—about 34 ns when mapped proportionally to the measured
primary time.

Coarse scheduling can matter more than individual stragglers. For 32×32 tiles
distributed across 32 abstract workers, greedy work-queue scheduling finishes
1.50% above the ideal balanced load; static raster-order round-robin assignment
finishes 37.3% above ideal. This does not claim Apple's exact hardware scheduler
behaves like either endpoint. It says the actionable risk is **how coherent hot
regions are distributed**, not a handful of pathological rays.

## 3. M1 Max throughput

### Measured

| Resolution | Pixels | Primary time | Primary throughput | Whole-frame time | Whole-frame throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| 800×460 | 0.368 M | 8.513 ms | 43.2 Mpix/s | 13.697 ms | 26.9 Mpix/s |
| 1600×920 | 1.472 M | 22.297 ms | 66.0 Mpix/s | 37.421 ms timestamp median / 38.90 ms Instruments median | 39.3 / 37.8 Mpix/s |

The 800×460 primary time is from the earlier exact Instruments capture; the
800×460 whole-frame time is current. The current 1600×920 capture is the clean
large-resolution reference. Quadrupling pixels increases exact primary time by
about 2.62×, so the larger workload achieves about 1.53× more primary pixels per
second. The small render does not fill the GPU as effectively.

At the large-resolution rate and same view/work mix, the primary pass processes
approximately 1.08 billion SVO node visits/s plus 2.08 billion fine voxel-cell
steps/s. This is about 3.16 billion counted dependent loop iterations/s; it is
not an instruction/s figure.

### Extrapolated primary-only budgets

- 16.67 ms (60 Hz): about 1.10 million pixels, roughly 1400×788 at 16:9;
- 33.33 ms (30 Hz): about 2.20 million pixels, roughly 1980×1114;
- 1920×1080: about 31.4 ms for primary visibility alone;
- 2560×1440: about 55.8 ms for primary visibility alone.

Those extrapolations exclude lighting and every other frame stage. The actually
measured 1600×920 complete frame is 25.6 fps by mean wallclock.

## Artifacts and reproduction surfaces

- Work-map report: `artifacts/svo-primary-experiments/work-map-800x460.json`
- Raw per-pixel records: `artifacts/svo-primary-experiments/work-map-800x460-primary-work-map.rgba32uint.bin`
- Heat maps: `artifacts/svo-primary-experiments/maps/`
- Clean timestamp runs: `baseline-800x460.json`, `baseline-1600x920.json`
- DDA-span A/B: `bounds-800x460.json`
- Current large Instruments report: `xctrace-1600x920/summary.json` and
  `xctrace-1600x920/report.html`

`tools/render-svo-primary-work-map.ts` regenerates the PNG views from a work-map
JSON report and the reproducible slow-tail scheduling model in
`maps/work-map-scales.json`. `tools/benchmark-svo-dry-frame-gpu.ts` enables the diagnostic with
`FLUID_SVO_DRY_FRAME_PRIMARY_WORK_MAP=1`; the diagnostic remains disabled by
default and does not affect production rendering.

In the app, press **Tab** to enter scene interaction mode. Primary-ray work is a
row in the scene's existing Field / Tank / Fluid solver toolstrip: click its
chevron to choose node-plus-voxel work, voxel DDA cells, SVO node visits, leaf
visits, or entry seed plus slow tail. Its glyph behaves like the field-view
glyph: click the lit mark to hide the visual, then click it again to restore the
last choice.
The first work view compiles the instrumented primary variant; moving between
work views reuses that counter plane. Turning the tool off restores the normal
uncounted renderer, so diagnostic timing is never mixed into presentation
timing.

## Interpretation boundary

The evidence identifies `traceLeafVoxelPayload` as the dominant internal unit,
and the cell-advance/search span as the best-supported target. It does not yet
separate the cost of each load, comparison, and DDA arithmetic instruction.
Doing that honestly requires compile-time micro-variants that preserve the same
hit result while selectively replacing one operation group at a time.
