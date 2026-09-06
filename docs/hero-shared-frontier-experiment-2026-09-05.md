# Shared upper-tree frontier experiment

Result: the cooperative shared frontier is **16% slower** in a paired comparison.
It is not promoted to production.

This benchmark-only prototype tests whether sharing hierarchy work across an
8×4 primary-ray tile improves garden-hose hero throughput at 1920×1080 on M1 Max.
Production rendering is unchanged.

One lane traverses the upper hierarchy against a conservative tile frustum.
The five half-space tests enclose all pixel-center primary rays and expand the
bounds for floating-point classification error. A 64-entry workgroup frontier
contains terminal nodes or subtrees at the selected cutoff depth (2, 3, or 4).
The shared pending stack holds 64 node indices. Together these arrays occupy
2,304 logical bytes per workgroup, plus counters. A workgroup barrier publishes
the frontier before any per-pixel bounds check or early return.

Each ray intersects the frontier boxes, sorts its candidates into the existing
32-entry continuation stack, and executes the unchanged fine traversal. There
is one direct dispatch. Timings include frontier construction, candidate sorting,
workgroup synchronization, output writes and any retries. No previous-frame
result is reused.

If shared construction exceeds a capacity or encounters invalid topology, the
tile uses ordinary root traversal. Per-ray frontier overflow likewise uses root
traversal. A fine-traversal stack overflow after accepting a shared frontier
causes a complete primary retrace with shared traversal disabled. A separate
forced-frontier-overflow case exercises the tile fallback.

This changes the arithmetic used to enter subtrees: frontier AABB intervals
replace the corresponding sequence of upper-tree parametric intervals. It must
therefore be checked against every pixel, including depth and identity, even
though the geometric candidate set is conservative. Full-population comparisons
are recorded separately from timing. Floating-point differences and pathological
views remain a promotion concern until validated.

Run the existing dry-frame benchmark with the same scene/refinement/resolution
settings as the main throughput report and enable:

```bash
FLUID_SVO_DRY_FRAME_PRIMARY_COMPUTE_PROBE=1
FLUID_SVO_DRY_FRAME_PRIMARY_SHARED_SWEEP=1
```

The sweep includes native controls before/after, depths 2/3/4, and forced fallback.
Reports and all raw outputs are under
artifacts/hero-utilization-2026-09-05/shared-frontier-1080/.


## Serial frontier construction result

| Variant | Primary GPU median | Relative to first native control |
| --- | ---: | ---: |
| Native 8×4 | 74.252 ms | 1.000× time |
| Shared depth 2 | 85.066 ms | 1.146× time |
| Shared depth 3 | 99.156 ms | 1.335× time |
| Shared depth 4 | 107.741 ms | 1.451× time |
| Forced frontier overflow | 89.784 ms | Correctness control |
| Native repeat | 75.235 ms | 1.013× time |

All cases use 12 warmups and 40 samples. Absolute native time is substantially
higher than earlier sessions (roughly 38 ms); its cause was not established.
Use the bracketing controls in this run, not historical absolute timings, to
assess these variants. No second Dawn benchmark ran concurrently with this one.

Every seeded ray (1,580,336 pixels) used the shared frontier in all three normal
variants. There were no frontier overflow tiles and no fine-stack retries.
Mean frontier size across all tiles was 3.87 / 6.61 / 9.27 nodes at depths 2/3/4;
maximum sizes were 8 / 12 / 20. More aggressive sharing increased candidate
filtering/sorting and preparation costs, and was slower.

All normal cases preserve hit/miss coverage. They are not byte-identical to
native compute: depths 2/3 differ in 3,150 pixels, depth 4 in 6,061 pixels
(comparing packed surface, identity and depth; excluding diagnostic counters).
Their surface/identity mismatch counts against raster also increase slightly.
Forced frontier overflow reproduces native compute exactly in these outputs.
This exercises the tile fallback, not the fine-stack retry path, which no ray
required in this run.

## Cooperative construction refinement

A second variant expands the upper hierarchy breadth-first with all 32 lanes.
It uses two workgroup-local pending lists, workgroup atomics for slot allocation,
and fixed-count level barriers. It introduces no indirect dispatch or device
queue. Per-ray candidate sorting and exact fine traversal are unchanged.
The generated WGSL passes validation. Enable PRIMARY_SHARED_COOPERATIVE=1 in
addition to PRIMARY_SHARED_SWEEP=1, with the full FLUID_SVO_DRY_FRAME_ prefix.
Reports are in artifacts/hero-utilization-2026-09-05/shared-cooperative-1080/.


The sequential cooperative sweep measured native 38.470 ms, shared depth 2
68.092 ms, shared depth 3 66.847 ms, and native repeat 58.524 ms. This large
control drift makes exact speedup/regression percentages unreliable. Both shared
variants used the frontier on all 1,580,336 seeded pixels, with no overflow or
retry, no hit/miss changes, and 3,150 differing pixels versus native compute.

A final paired mode alternates native/shared dispatches each sample and reverses
the order on alternate samples. Both pipelines compile before measurements.
Each dispatch has its own GPU timestamps; only the final sample copies its
output for correctness checks. This improves the comparison under changing
desktop load. Enable FLUID_SVO_DRY_FRAME_PRIMARY_SHARED_PAIRED=1 together with
PRIMARY_COMPUTE_PROBE=1; it selects the depth-2 cooperative variant and native
8×4 control. Outputs: artifacts/hero-utilization-2026-09-05/shared-paired-1080/.


## Final paired result and decision

| Variant | GPU median |
| --- | ---: |
| Native 8×4 compute | 38.666 ms |
| Cooperative shared frontier, depth 2 | 44.827 ms |

Twelve warmup pairs and sixty measured pairs were run. Shared traversal is slower
in **all 60 pairs**; the median per-pair time ratio is **1.1631**. The ratio of
individual medians is 1.1593. Use approximately **16% regression** as the result.
There are no frontier overflows or fine-stack retries, so fallback traffic does
not explain this regression.

Compared directly with native compute across 2,073,600 pixels:

- Hit/miss differences: 0.
- Packed surface differences: 2 pixels.
- Identity differences: 2 pixels.
- Depth differences: 3,149 pixels; maximum absolute reverse-Z difference
  0.0000000013969838619232178.
- Any output difference: 3,150 pixels, excluding diagnostic fields.

Both serial and cooperative construction failed to improve throughput. The data
supports rejecting this particular frontier-construction/filtering design; it
does not establish that all coherent packet traversal is ineffective. This
prototype still retains a full private per-ray continuation and adds candidate
sorting and workgroup synchronization. It does not demonstrate improved hardware
occupancy, which was not recaptured for these variants.

The forced tile-overflow control is byte-identical to native compute. Fine-stack
retry was implemented but never exercised by these scene runs. Additional edge
and overflow validation would be needed before production use regardless of
performance. Since the performance gate fails, no production changes are made.

Generated serial/cooperative WGSL passes Dawn validation, and the completed GPU
benchmarks report no validation errors. Seven targeted renderer tests pass.
Repository-wide type checking still fails in unrelated solver/harness files;
there are no reported errors in the changed profiling tools.
