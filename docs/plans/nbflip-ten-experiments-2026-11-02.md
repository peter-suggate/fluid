# Narrowband FLIP: ten executed optimization experiments

Follow-up: [actual hot-kernel experiments and full-scene validation](nbflip-hot-kernels-2026-11-02.md) cover particle advection, transfer, pressure and surface reconstruction.

All ten questions now have executable comparisons, including preparation costs
and correctness checks. The first matrix finds workload-specific improvements,
not a universal replacement for the current kernels. No runtime change is
promoted from these measurements. The full-solver experiment supports using the
existing asynchronous receipt ring instead of awaiting every frame.

## Method and scope

Hardware: Apple M1 Max, Metal 3, macOS 26.6.2, Node/Dawn. The shared GPU runner
serializes timing against other repository GPU work. The initial matrix uses
11 rotating trials of 32 complete chains, with an unchanged second baseline.
GPU timestamps have Dawn timestamp quantization disabled. Compilation,
allocation and validation readbacks are excluded; result clearing, conversion,
worklist construction, classification, dispatch argument generation and the
consumer are included wherever needed. Every iteration clears both output
planes, so repeated atomic reductions are not measured against saturated results.

Sections 1–9 use the actual production surface shaders on a controlled 8³-tile,
33³-vertex domain. Directory lookup is replaced by a regular-grid fixture.
These are surface-chain experiments, not whole-frame speedups or measurements
of every hot kernel. Previous real-scene pressure, movement and surface replays
are documented in [the kernel audit](nbflip-kernel-audit-2026-11-02.md).
Section 10 includes both a controlled chain and the complete current solver.

| Fixture | Particles | Occupied tiles | 512-particle jobs | Hot particles |
| --- | ---: | ---: | ---: | ---: |
| Empty | 0 | 0 | 0 | 0 |
| Sparse | 256 | 32 | 32 | 9 |
| Mixed cost | 173,184 | 512 | 768 | 2,720 |
| Dense | 524,800 | 512 | 1,536 | 175,104 |
| Clustered | 262,208 | 64 | 576 | 1,865 |

Mixed tiles contain 8/64/256/1,025 particles; clustered tiles contain 4,097.
The fixture permutes the occupied list, sorts particles spatially, includes
boundary vertices and coherent cold regions, and allocates about 50% spare
particle capacity. Packing processes the active count, not that spare capacity.
Auxiliary experiment storage is allocated for all variants; the effect of extra
persistent memory on a complete solver's working set is outside this test.

Every candidate compares all 71,874 distance/hot-distance output words against
the scalar reference. All explicit tile-job queues are additionally checked as
multisets against the expected jobs, catching missing/duplicate work even when
a reduction happens to give the same minima. Empty queues, partial jobs,
multiple jobs per tile and more jobs than launched workgroups are exercised.
All five fixtures passed; no GPU errors were reported.

## Experiments and decisions

1. **Memory layout.** Compare compact 16-byte AoS, four SoA component planes,
   and 64-record AoSoA blocks. Consecutive lanes in AoS read byte addresses
   `base + 16*i + {0,4,8,12}`; SoA uses `base + 4*i + 4*capacity*component`;
   AoSoA uses `base + 1024*floor(i/64) + 256*component + 4*(i%64)`.
   The latter two make each component contiguous. Both include conversion from
   the existing producer's AoS stream. They slow nonempty chains by 13–20% and
   13–18% respectively in the initial run. Keep the existing stream. This does
   not establish that a producer writing SoA directly would lose.

2. **Bandwidth.** Retain exact xyz floats and pack the hot predicate into one
   bit instead of a float. Stream storage falls from 16 to 12.125 bytes/sample
   (24.2%); positions are not quantized. Packing and flag construction are timed.
   Chains slow by 18–35%. Each consumer still issues a 32-bit flag-word load;
   adjacent particles can reuse that word in cache, but packed storage size is
   not measured DRAM traffic. Reject this extra conversion pass.

3. **Unnecessary work.** Compare the fixed 1,024-group scatter with a GPU-built
   indirect launch of `min(jobCount,1024)` groups, preserving striding when
   jobs exceed the cap. Includes the argument-builder dispatch and Dawn's
   indirect validation. It saves 992 launched groups in the sparse case yet
   slows its chain by 22%; other nonempty cases slow by 2–3%. Existing compact
   planning already avoids scanning particle capacity. Keep fixed launch here.

4. **Divergence and scheduling.** Build three compact queues by job population
   (384–512, 128–383, 1–127), then consume heavy jobs first. Construction and
   queue lookup are timed. Mixed-cost time falls 18%; dense rises 7%; sparse
   and clustered gains are negligible. This tests workgroup cost distribution,
   not a claim that instruction-level divergence was measured. A conditional
   policy needs complete-scene evidence before adoption.

5. **Pass fusion.** Scatter directly from the occupied-tile list, consuming the
   full tile instead of a bounded particle chunk. Removes the planning dispatch
   and two-word job write/read round trip (16 logical bytes/job). Sparse time
   falls 16%; mixed and clustered time more than doubles. Dense improves 4%.
   Removing the job boundary also removes parallelism for dense outliers.
   Keep bounded jobs as the default.

6. **Kernel splitting.** Classify each job by whether any sample is hot; enqueue
   no-hot and exceptional jobs separately and dispatch both indirectly. The
   common kernel removes the second 343-word shared window, hot loads and hot
   flush. The classifier, queues and argument generation are timed. Despite
   29/32 sparse and 718/768 mixed jobs taking the common path, chains slow
   15–118%. The duplicate scan and extra dispatches outweigh the saved state.

7. **Atomics.** Replace one global job-range reservation per occupied tile with
   one per nonempty 64-lane planner workgroup; use local aggregation and barriers
   to place each tile's jobs. This reduces reservations from 512 to 8 in the
   dense fixture. Total chain time is essentially unchanged there and sparse
   time increases 3%. Exact job-set checks pass despite changed queue order.
   Do not promote a global-atomic count reduction without a chain improvement.

8. **Neighborhood reuse.** Compare scalar per-vertex reads against the existing
   cooperative per-cell kernel on exactly the same selected cell list and
   indirect arguments. Up to 64 vertices reuse each particle; the cooperative
   kernel stages 64 positions and flags in 1,280 bytes plus its range, with two
   barriers per batch. It is 9% faster on clustered cells, but 3–10% slower on
   the other nonempty cases. Cache reuse in the scalar path already matters.
   This pair must be compared to each other, not to the tile-scatter baseline.
   The previous actual-scene pressure tile/halo experiment also preserved exact
   outputs without establishing a consistent improvement.

9. **Occupancy.** Compare 64/128/256 scatter lanes with matching loop strides,
   identical 512-particle jobs and both shared windows. 64 lanes slows nonempty
   chains by 5–28%; 256 is near the baseline except an 8% clustered regression.
   Keep 128. These are workgroup-size timings, not hardware occupancy or spill
   measurements. Previous actual-scene 64-lane replays also lost; shortening
   separation-kernel private state was measured separately in the kernel audit.

10. **CPU–GPU coordination.** Run (a) CPU-map a 4-byte count before dispatch,
    (b) GPU argument generation with one completion fence per chain, and
    (c) GPU arguments with asynchronous diagnostic receipts, four submitted
    chains per batch. Include every map/fence and verify every counter and final
    output. Use 24 chains across seven rotating rounds. Separately compare
    per-step completion waits against the solver's existing two-frame receipt
    ring, with a second fenced control. Results below distinguish GPU chain
    timing from CPU wall time and preserve the current lagged planning policy.

## Logical traffic model

These are shader-derived logical accesses, not hardware DRAM counters. For a
tile job with P samples, the main scatter reads 12P position bytes and up to
4P hot-state bytes (cold samples skip the latter), plus its 8-byte job record
and address/bin metadata. Planning writes that 8-byte record. Both 343-word
windows require 2,744 bytes of shared initialization per job. Each sample tests
64 vertices and conditionally performs shared minima; publication performs up
to 343 global reductions per output plane. Reduction traffic depends on support,
hotness, overlap and cache behavior, so a fixed bytes-per-output-vertex number
would be misleading.

SoA/AoSoA conversion adds 16 bytes read plus 16 written per active sample before
the consumer. Bit packing reads the same 16 bytes and writes 12.125 bytes per
sample (apart from the final partial word). The conversion costs explain why
smaller or more contiguous storage is not automatically a faster complete path.
The full 48-byte FLIP/112-byte APIC record ABI is not changed by these tests.

## GPU chain measurements

Initial medians in milliseconds, including reset and required preparation.
The second baseline differs from the first by at most 0.2% across fixtures.

| Variant | Empty | Sparse | Mixed cost | Dense | Clustered |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.012065 | 0.021574 | 0.134998 | 0.305756 | 0.164167 |
| layout-soa | 0.014349 | 0.024288 | 0.156505 | 0.365319 | 0.196569 |
| layout-tiled | 0.014286 | 0.024290 | 0.154912 | 0.358950 | 0.193605 |
| bandwidth-packed-hot | 0.014854 | 0.029204 | 0.159779 | 0.375610 | 0.200708 |
| active-indirect | 0.012277 | 0.026345 | 0.139607 | 0.311370 | 0.168639 |
| cost-buckets | 0.012228 | 0.021719 | 0.110627 | 0.326836 | 0.162257 |
| fused-tile | 0.009334 | 0.018090 | 0.290105 | 0.294394 | 0.334145 |
| split-hot | 0.018650 | 0.047038 | 0.193019 | 0.350940 | 0.212109 |
| aggregated-atomics | 0.012457 | 0.022318 | 0.136299 | 0.306533 | 0.163025 |
| reuse-scalar | 0.012272 | 0.015965 | 0.113927 | 0.244589 | 0.121386 |
| reuse-shared | 0.012240 | 0.016387 | 0.124709 | 0.267988 | 0.110563 |
| lanes64 | 0.011896 | 0.022570 | 0.145830 | 0.351357 | 0.210244 |
| lanes256 | 0.012969 | 0.021378 | 0.136067 | 0.309148 | 0.177877 |
| baseline-repeat | 0.012091 | 0.021548 | 0.135153 | 0.305917 | 0.163871 |

## Balanced confirmation

The final repeat uses 14 rotating trials, so every one of the 14 variants
occupies each execution-order position exactly once. All five fixtures and
all four coordination cases pass again. All shader hashes match the initial
run. Baseline A/A differences are at most 0.3%. The principal conclusions
repeat; the table gives percentage change in total GPU chain time, so negative
means faster. Neighborhood reuse uses its own scalar reference.

| Candidate | Sparse | Mixed cost | Dense | Clustered |
| --- | ---: | ---: | ---: | ---: |
| layout-soa | +13.8% | +16.4% | +19.3% | +19.1% |
| layout-tiled | +13.7% | +15.3% | +17.1% | +17.5% |
| bandwidth-packed-hot | +36.9% | +18.1% | +22.7% | +22.2% |
| active-indirect | +22.6% | +3.5% | +1.8% | +3.0% |
| cost-buckets | +0.7% | -17.9% | +6.9% | -1.1% |
| fused-tile | -16.0% | +114.5% | -3.6% | +103.7% |
| split-hot | +119.3% | +42.9% | +15.7% | +29.8% |
| aggregated-atomics | +3.5% | +1.3% | +0.3% | -0.4% |
| reuse-shared / scalar | +3.6% | +10.2% | +7.8% | -8.9% |
| lanes64 | +5.5% | +8.0% | +15.2% | +30.9% |
| lanes256 | -0.2% | +0.4% | +1.1% | +9.2% |
| baseline-repeat | -0.1% | -0.1% | +0.0% | -0.0% |

Confirmation CPU wall medians in milliseconds/chain:

| Fixture | CPU count readback | GPU count + fence | GPU count + async receipts |
| --- | ---: | ---: | ---: |
| sparse | 0.7723 | 0.4011 | 0.2237 |
| mixed-cost | 1.1512 | 0.8888 | 0.6157 |
| dense | 1.2837 | 0.9561 | 0.7075 |
| clustered | 1.1978 | 0.9905 | 0.6890 |

These retain the same scheduling ranking while showing the expected wall-time
variation between runs. The final repeat ran after type checking and unit tests
finished. An earlier balanced repeat overlapped a CPU type check and is excluded
from the reported confirmation; its raw data remains under `confirm-v2/`.

## CPU scheduling measurements

Initial controlled-chain CPU wall medians in milliseconds/chain. These include
Node/Dawn submission, polling and synchronization, not rendering. Round-to-round
drift is visible in the raw samples; compare magnitudes rather than fine digits.

| Fixture | CPU count readback | GPU count + fence | GPU count + async receipts |
| --- | ---: | ---: | ---: |
| sparse | 0.6742 | 0.3724 | 0.1774 |
| mixed-cost | 1.1914 | 0.8094 | 0.5986 |
| dense | 1.4826 | 1.0459 | 0.6672 |
| clustered | 1.3361 | 1.0415 | 0.7026 |

The full solver uses `sparse-cm12-long-dam-break`, 1/60 s, 120 steps, excluding
the first 24 from timing. Twelve-step blocks rotate among three solver instances.
Construction, rendering and final diagnostics are excluded. Results:

| Mode | Median block ms/step | Mean ms/step | Completion fences over 96 measured steps | Mapped bytes |
| --- | ---: | ---: | ---: | ---: |
| Fenced A | 24.707 | 26.184 | 96 | 62,976 |
| Existing two-frame ring | 21.208 | 21.636 | 8 | 62,976 |
| Fenced B | 25.325 | 25.260 | 96 | 62,976 |

All modes submit 384 command batches and perform 192 maps: four submissions,
two maps and 656 mapped bytes per step. Pipeline throughput improves about
14–16% by median, while preserving the readback volume. Mapped buffers are the
pressure/accounting receipt and mixed-relayout receipt. Fixed-lag work counts,
pressure/cold planning and failure information influence later frames; these
are not all disposable diagnostics. The experiment changes admission/waiting,
not their contents or the lag. Production already exposes this pipelined path.

All three runs finish without pipeline/GPU errors and agree on volume sum
40,956.546875 and volume drift −0.00008430481. Final particle counts are
152,013 / 152,684 / 149,500; maximum speeds are 2.507 / 2.690 / 3.043 m/s;
accepted pressure residuals are 0.03234 / 0.02100 / 0.01888. Independent atomic
simulations therefore do not give bitwise parity, including between the two
fenced controls. This supports a scheduling conclusion, not a proof of identical
trajectories. The controlled-chain coordination experiment does have exact
field/counter parity.

## Reproduction and validation

```sh
bash tools/gpu/fluid-gpu-run.sh timing \
  node --import tsx tools/benchmark-nbflip-ten-sections.ts \
  --repeats=14 --inner=32 \
  --out=artifacts/nbflip-ten-sections-2026-11-02/confirm-final
bash tools/gpu/fluid-gpu-run.sh timing \
  node --import tsx tools/benchmark-nbflip-coordination.ts
npm run check:types
npm run test:unit
```

`--sections=1,2,...,10` and `--cases=empty,sparse,mixed-cost,dense,clustered`
can select a smaller study. The baseline controls always run. The shader
transformations fail if expected source fragments disappear, rather than
silently timing an unchanged candidate. Each results file records generated
shader hashes, variant definitions, dispatch sequences and individual timings;
the fixture run also writes each generated shader. The full solver report
records a hash of its uniform-method sources and all method values.

Evidence: `artifacts/nbflip-ten-sections-2026-11-02/results.json`,
`confirm-final/results.json`, and `coordination/results.json`. The first two run
directories contain the exact generated WGSL. The initial reports predate the
explicit `completed` field; their five fixture rows/four coordination rows and
three full-solver summaries, together with successful process exits, establish
completion. Confirmation reports set `completed` only after all checks pass.
An intervening confirmation attempt failed shader compilation after a typo in
the source-transformation guard refactor; it produced no timing results. The
typo was fixed, and all 14 generated shader hashes were checked against the
successful initial run before repeating the matrix.

- `npm run check:types`: passed after the tooling edits.
- `npm run test:unit`: 1,447 passed, 264 skipped, zero failures (1,711 total).
- Initial and final GPU matrices: all field/job/counter comparisons pass; zero GPU errors.
- Full solver: all three 120-step runs complete; no pipeline/GPU errors.
- The final matrix reports `completed: true` and preserves all 14 initial shader hashes.
