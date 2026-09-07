# Primary traversal utilization: research and experiment priorities

Research date: 2026-09-05. Target: garden-hose hero, M1 Max/Metal through
Dawn/WebGPU, 1920×1080, unchanged visibility, 2× primary throughput. This note
proposes experiments; it reports no additional measured speedups.

## Interpret the counters first

Fragment occupancy 15.61%, ALU utilization 26.21%, and LLC utilization 29.69%
are consistent with insufficient latency hiding, but do not identify its cause.
LLC utilization is neither cache hit rate nor cache capacity occupied. Improving
locality could reduce LLC utilization while making traversal faster. Apple's
occupancy guidance explicitly requires correlating occupancy with other counters.
[Apple occupancy documentation](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy)

Collect shader theoretical occupancy, registers, spill traffic, and the available
ALU/buffer/LLC limiter counters alongside utilization. A limiter substantially
above utilization indicates inefficiency or stalls within that subsystem.
Apple's M1-era guidance discusses dynamically indexed private arrays and register
allocation thresholds: a small source-level reduction need not change occupancy.
[Metal Compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/)

Use the counters actually available on M1. M3/A17 and M5/A19 talks introduce
additional architecture-specific diagnostics; their availability and occupancy
management behavior must not be assumed on M1. Recapture compute counters for
compute experiments: the existing fragment percentages characterize the raster
baseline, not the 36–38 ms compute prototype.
[Apple M3 profiling tools](https://developer.apple.com/videos/play/tech-talks/111374/)

## Ranked experiments

### 1. Seed-leaf fast path and tighter conservative entry proxies

This is a renderer-specific proposal inferred from the current code. The entry
prepass emits nodeIndex+1, but dryPrimaryEntrySeedLoad retains only its empty
state and depth. The main cursor then starts a hierarchy traversal. Exploit the
identity to test that leaf directly before constructing a continuation.

A seed proxy hit is not an exact nearest surface certificate. Padded proxies can
overlap; empty space inside a leaf and competing proxy entries matter. For a
safe first prototype, use the candidate hit as a tighter upper bound and run the
normal traversal to certify it. Only bypass traversal when an exact frontmost
interval or equivalent certificate proves there cannot be a nearer hit. Misses,
ambiguous overlap and stale publication must fall back without losing surfaces.

Tighten raster proxies to conservative occupied bounds/contours where possible.
Measure seed-leaf hit rate, certifiable hits, fallback rate, removed node visits,
and prepass plus primary time. The shader should initialize the full stack only
on fallback, but check compiled resource allocation: one function containing a
fallback can still reserve resources for the worst path. Two fixed full-screen
dispatches with per-pixel flags are an alternative if specialization wins enough
to pay for another pass; no indirect dispatch is required.

Code: lib/svo/features/primary-visibility/webgpu-svo-primary-entry-prepass.ts:245;
lib/svo/pipeline/webgpu-svo-dry-scene.ts:3338 and :5449.

### 2. Share upper-tree traversal across coherent rays

Laine and Karras describe beam acceleration of SVO ray casts. Apply the spatial
coherence principle to a tile's common hierarchy work, with exact per-ray
traversal after the rays separate.
[Efficient Sparse Voxel Octrees technical report](https://research.nvidia.com/publication/2010-02_efficient-sparse-voxel-octrees-analysis-extensions-and-implementation)

Our proposed adaptation: a fixed 4×4 or 8×4 packet generates a conservative
frontier of candidate subtrees once, then each ray traverses its valid intervals.
A single workgroup can construct and consume this frontier in one direct dispatch.
Bounded shared storage needs a correctness-preserving overflow fallback. Keep
collectives in uniform control flow, including lanes whose rays have finished.
Check the adapter's subgroup feature before choosing subgroup operations;
workgroup memory is an alternative with synchronization costs.
[WGSL subgroup and uniformity rules](https://www.w3.org/TR/WGSL/)

First measure common-node overlap by tree depth. Similar ray lengths do not
prove shared node identities. Stop sharing where divergent child masks create
more work than they save. Avoid a large shared stack that merely moves the
occupancy constraint from private memory to workgroup memory.

A coarse depth-only beam pass is lower priority: this renderer already has a
per-pixel entry-depth seed. It must improve that bound or avoid root descent to
provide something new. A center ray, or corner samples alone, cannot certify
empty space throughout a tile without conservative geometric bounds.

### 3. Compact continuation state without replaying AABB work

The current continuation includes 32 entries of {nodeIndex,tEnter,tExit}, or
384 bytes of logical stack state before other fields. Logical bytes are not a
measurement of registers or physical spills. Our index-only stack experiment
regressed, so do not repeat index compression plus interval recomputation.

Research alternatives include compressed pending-child masks and stackless
backtracking with auxiliary hierarchy metadata. Binder and Keller show constant
state/time next-node selection in a BVH algorithm. Adapting it to this octree
requires its own design and measurement; the paper does not prove an M1 gain.
[Stackless hierarchy traversal](https://research.nvidia.com/publication/2016-06_efficient-stackless-hierarchy-traversal-gpus-backtracking-constant-time)

Instrument stack high-water marks, then compare a short stack with exact
fallback, or a parent/child-mask continuation with preserved interval arithmetic.
Keep full precision for ray bounds. Compile out optional features and shorten
live ranges of node, leaf and hit records. Apple specifically identifies shader
specialization as a way to reduce register pressure.
[Apple silicon shader specialization](https://developer.apple.com/videos/play/wwdc2021/10153/)

### 4. Reduce dependent node loads through a traversal-specific layout

Compressed wide BVHs combine compact nodes, compressed stacks and fixed child
ordering. Their reported performance is for NVIDIA BVHs and mostly incoherent
rays; it is not a forecast for this scene.
[Compressed wide BVHs](https://research.nvidia.com/publication/2017-07_efficient-incoherent-ray-traversal-gpus-through-compressed-wide-bvhs)

Our canonical nodes are 32 bytes; children are already contiguous, and parametric
expansion already avoids reading every child record. Existing compact/wide modes
should be benchmarked on the current scene before designing another hierarchy.
Potential new work: separate hot traversal fields from cold metadata, store small
subtrees together, and collapse multiple levels only where this reduces dependent
loads without expensive child tests. Include derived-publication cost for edits.

### 5. Hide latency within a thread or change the load path

A bounded two-ray-per-thread experiment could issue independent traversal loads
before consuming either result. It may improve instruction-level parallelism,
but doubles ray state and can worsen occupancy. Try only after the state audit,
with nearby rays and one direct dispatch. This is a proposed experiment, not a
result established by the cited BVH studies.

Apple also suggests textures as an alternative to buffers when buffer accesses
are limiting, since their access paths use different caches. Test exact integer
texture loads for read-only node fields only if buffer limiter evidence supports
it; format conversion and coordinate arithmetic may erase the benefit.
[Apple GPU counter optimization](https://developer.apple.com/videos/play/wwdc2020/10603/)

## What not to prioritize

Global compaction/indirect queues and static expensive-first ordering have little
support from our work-map and timing results. DDA-only arithmetic tuning targets
7.64% of counted node-plus-cell operations. Half-precision ray intervals risk
visibility changes. Temporal G-buffer reuse is explicitly retired in this repo.
Hardware RT is not an M1 performance lever; Apple introduced Mac hardware ray
tracing with M3. The existing raster-primary proxy path is another relevant
baseline, but must include overdraw and exact in-brick visibility costs.
[Apple M3 GPU architecture](https://developer.apple.com/videos/play/tech-talks/111375/)

## Decision gate

Start with compiler/register/spill evidence and seed-leaf success statistics.
Then test certified seed bypass and shared upper-tree work, followed by compact
continuations. Accept only reduced primary time with unchanged visibility; count
new preparation passes and output bridging. Retain fixed 1920×1080, the same
camera/publication, bracketing baselines, and full G-buffer comparisons. Profile
counters in a separate run from instrumented work counting. No paper establishes
that any individual change will deliver our missing 2×.


## Follow-up measurements

The seed/stack diagnostic at the same 1920×1080 view found:

- Seeded pixels: 1,580,336; voxel-hit pixels: 1,278,232.
- Seed node equals the final hit node: only 6,738 pixels (0.527% of voxel hits).
- Seeded pixels with no voxel hit: 302,104.
- Maximum observed stack occupancy: 18 entries; 533 pixels exceed 16 entries.
- Full-population output comparison matches the prior compute baseline:
  no coverage changes, 423 packed-surface differences, 55 identity differences,
  and maximum reverse-Z depth delta 0.00000539422 relative to raster.

This measures identity agreement, not the success rate of a certified shortcut.
Its low value deprioritizes direct seed-leaf bypass. The seed remains useful as
an empty-ray certificate and conservative near distance. The stack histogram
supports a scene-specific 20/24/32 capacity experiment preserving interval values;
it does not establish a safe global capacity reduction.

Instrumentation writes seed/winner IDs and stack/work counters into unused
compute-output components. Instrumented timings are excluded from speedup claims.
Use PRIMARY_DIAGNOSTIC=1 with PRIMARY_COMPUTE_PROBE=1 (full environment prefix
FLUID_SVO_DRY_FRAME_). The report is in
artifacts/hero-utilization-2026-09-05/diagnostic-1080/report.json.

The installed Xcode Metal compiler launcher cannot run because its Metal Toolchain
component is absent. Consequently no compiled register count or spill-byte result
has been obtained, and register pressure remains a hypothesis. The existing
Instruments capture was reprocessed for additional limiter counters without
another GPU workload.


### Recovered primary-pass limiter counters

| Counter | Limiter | Utilization |
| --- | ---: | ---: |
| ALU | 29.50% | 26.52% |
| Buffer reads | 2.81% | 2.74% |
| Buffer writes | 4.30% | 4.28% |
| GPU LLC | 30.48% | 30.44% |
| MMU | 5.25% | 5.16% |

F32 utilization is 2.84%, F16 is zero, and fragment occupancy is 15.40%.
This extraction retains more counter kinds and therefore uses a larger sampling
stride at the same reduction setting: 109 primary samples, drawn from the same
8.72% uncontended primary coverage. Small differences from the previous counter
means reflect sampling, not a shader change. Raw report:
artifacts/hero-utilization-2026-09-05/limiters-1080/summary.json.

The small LLC/buffer limiter-utilization gaps do not support large cache-stall
penalties in these windows. This does not rule out serial load latency, low
memory-level parallelism, or spills. It also does not prove register pressure.
Low F32 utilization motivates inspecting integer/control work, not subtracting
counters to obtain an exact integer-instruction fraction.

### Capacity-only stack sweep

| 8×8 compute variant | Median GPU time |
| --- | ---: |
| 32-entry stack | 37.224 ms |
| 24-entry stack | 38.994 ms |
| 20-entry stack | 36.962 ms |
| 32-entry repeat | 37.159 ms |

All four variants produce byte-identical compute output over all 2,073,600 pixels.
Each case uses 12 warmups and 60 timed samples. All entries still store node ID
and both interval endpoints. Twenty entries improve only about 0.7% versus the
first control, and 0.5% versus its repeat. Twenty-four regress about 4.8%.
This does not support shipping smaller stacks. A short stack without fallback is
not safe for arbitrary views; this probe is restricted to the measured scene.
The compiler may have other allocation constraints, and this experiment alone
cannot locate them. Results: artifacts/hero-utilization-2026-09-05/stack-sweep-1080/.

The next algorithmic experiment should measure common-node overlap across ray
tiles and share a conservative upper-tree frontier where that overlap is high.
The seed-ID shortcut and capacity-only stack reduction are now lower priority.
Production shaders remain unchanged. Seven renderer tests and seventeen trace
parser/report tests pass; repository-wide type checking still reports unrelated
solver/harness errors, with none in the changed profiling tools.


The shared upper-tree frontier experiment has now been run in serial and
lane-cooperative forms. A paired 60-sample comparison finds the cooperative
variant 16% slower (38.67 ms native versus 44.83 ms shared), with no overflow
or retries. This design is rejected. See
[the complete experiment](hero-shared-frontier-experiment-2026-09-05.md).
