# Particle ordering: moderate occupancy and density scaling

Measured on 2026-10-19 with the production ordering stage on Apple Metal. Other GPU jobs were running throughout; these are contention-limited measurements, not isolated frame-rate claims.

The retained change removes the quadratic cost of sorting an arbitrarily crowded cell. The follow-up reduces the moderate-case regression: the million-particle, 128-per-cell diagnostic moved from 2.094 ms to 1.609 ms (23%). Those are separate runs with identical per-pass instrumentation, so the percentage is indicative. The final paired measurement with exactly two timestamps per arm is 1.688 ms versus 1.635 ms for the original sorter: a remaining 3% median overhead.

## Why the moderate case was slower

The first scalable version sorted 64-key runs, then merged them. A 128-particle cell therefore paid for an intermediate write and a full merge despite fitting comfortably in the original rank kernel. Merges initially followed source indices back into shuffled particle records for position comparisons; that random traffic developed a cache-size cliff. Retaining compact xyz/source keys and loading bounded merge tiles once into shared memory removed most of it.

The remaining regression came from work scheduling and data movement. Building merge lists in the packing pass increased its million-particle time from about 0.28 ms to 0.33 ms. Moving that work into the initial sort reuses cell metadata already loaded there and restores the original packing kernel.

Several plausible alternatives were measured and rejected: packed shared-memory sorting networks, subgroup exchanges, statically expanded network stages, a separate record gather, inverse-permutation scatter, and shared-memory rank sorting. The network variants reduced comparisons but did not recover the moderate-case timing; separating record movement exposed roughly 0.75 ms of random full-record gather at a million particles. None of those experimental paths remains in production.

## Retained implementation

- Every cell uses the same bounded-run sort followed by parallel tiled merges. Initial runs contain at most 128 keys; there is no occupancy-triggered return to the unbounded algorithm.
- Merge work is split into 256-key output tiles. Two diagonal searches locate the input slices, which are loaded once into shared memory. Per-key rank searches then stay within that tile.
- Both banks retain the compact position and source index. Full particle payloads are fetched for exact-position tie comparisons and the final record move.
- The initial sort builds merge work lists while its cell metadata is available. GPU-generated indirect counts activate only populated levels; no CPU count readback or per-frame scratch allocation is needed.
- All direct and indirect grids stride their work. A quiet-frame grid remains correct when an unexpectedly large population activates.

For cell populations k and total N, the former position-ranking work was Σk², with a further full-record comparison scan on ties. Initial ranking is now bounded by 128N position comparisons plus a similarly bounded tie scan. Each merge level performs at most eight shared-key binary-search comparisons per particle, plus two logarithmic diagonal searches per 256-key tile. Increasing concentration adds merge levels instead of unbounded per-particle scans. This is not a claim of strictly linear work for arbitrarily large cells.

At capacity 1,048,576 the order stage reports 17,340,232 allocated bytes (about 16.54 MiB), including scan resources. The main addition is one 16-byte compact-key bank per particle plus bounded work lists. It does not allocate another full particle-record buffer.

## Paired final measurements

GPU milliseconds include prefix scan, packing, and ordering. Each arm uses exactly two timestamp queries, nine alternating rounds and three repetitions per round. First dispatches are recorded separately. Exact output records are checked against a CPU oracle. The table retains p95 values to show the timing limitation.

| Workload | Old median / p95 ms | New median / p95 ms | Old/new median |
|---|---:|---:|---:|
| 1,048,576 particles, 128/cell | 1.635 / 3.264 | 1.688 / 3.494 | 0.97× |
| 32,768 particles, 128/cell | 0.114 / 0.142 | 0.163 / 1.775 | 0.70× |
| 32,768 particles, 1,024/cell | 0.415 / 0.644 | 0.218 / 0.926 | 1.90× |
| 32,768 particles, 8,192/cell | 2.285 / 4.589 | 0.253 / 2.411 | 9.02× |
| 8,192 in one cell, distinct positions | 1.238 / 3.101 | 0.185 / 1.857 | 6.67× |
| 8,192 in one cell, position ties | 8.371 / 11.229 | 0.294 / 1.849 | 28.49× |
| 8,192 in one cell, identical records | 9.247 / 11.576 | 0.296 / 1.834 | 31.20× |
| 8,192 in one cell, last-field differences | 9.026 / 11.678 | 0.310 / 1.507 | 29.13× |

For one cell with distinct positions, new median times at 8,192 / 16,384 / 32,768 / 65,536 / 131,072 particles were 0.185 / 0.217 / 0.391 / 0.570 / 0.991 ms. Position-tied records at 131,072 particles took 2.271 ms. The explicit work counts support the expected scaling; concurrent GPU activity prevents a claim of a smooth measured frame-time curve.

## Activation and remaining overhead

Small workloads still pay approximately 30–50 μs for the extra dispatch sequence, even when every merge count is zero. Eliminating that fixed scheduling cost is the next target; speculative host-side skipping would risk missing an unexpected activation. Moderate 128→129 occupancy adds a merge stage, so a small step remains even though there is no quadratic density cliff.

The quiet-grid run used 64 direct workgroups, the same million-particle capacity, and 15 alternating rounds. First/median/p95 below are for the new implementation; the repeated rows preserve the activation sequence.

| Active particles in one cell | First dispatch ms | Median ms | p95 ms |
|---:|---:|---:|---:|
| 0 | 2.401 | 0.099 | 1.008 |
| 64 | 0.163 | 0.159 | 0.220 |
| 127 | 0.203 | 0.199 | 0.609 |
| 128 | 0.179 | 0.200 | 0.906 |
| 129 | 0.208 | 0.208 | 0.224 |
| 128 | 0.162 | 0.162 | 0.417 |
| 4,096 | 0.325 | 0.255 | 0.656 |
| 128 | 0.895 | 0.161 | 0.222 |
| 0 | 0.082 | 0.080 | 0.746 |

The quiet-grid run did show first-dispatch spikes: 2.401 ms on the initial empty workload and 0.895 ms when returning to 128 particles. The old arm also had first-dispatch outliers (2.796 ms at 64 particles), and its initial empty sample returned an invalid timestamp. These results cannot separate contention from startup effects, and do not establish an activation-latency guarantee. The standard-grid run's 128→129 transition was 0.161→0.169 ms median, with first dispatches of 0.165→0.173 ms. Pipeline compilation is completed before measurement.

Both final data files contain one invalid first timestamp in the old arm; none of their repeated samples used for medians or p95 was invalid. The benchmark now explicitly marks missing or reversed timestamp ranges invalid and excludes them from aggregates. Real positive outliers are retained. A follow-up quiet-grid run explicitly rejected one invalid first sample and retained all 810 valid repeated samples; it still showed variable positive first-dispatch delays. The table above deliberately preserves the original observed spikes.

## Correctness and validation

- `npm run check:types`: passed.
- `npm run test:unit`: 972 passed, 65 skipped, zero failures.
- GPU ordering and APIC transfer suite: four passed, zero failures. Exact unsigned record ordering covers FLIP and APIC, both buffer parities, dense and sparse offsets, dropped particles, coincident positions, identical records, every payload component, partial runs, 127/128/129 and 255/256/257 boundaries, and repeated empty/dense transitions. Capacity stress reaches 131,072 with a one-workgroup launch.
- The previously attempted whole-solver pool integration test failed during compilation of a separate transfer/advection shader in `uniform-narrow-band-flip.ts` because of a non-uniform workgroup barrier. This report does not claim a passing whole-solver run.

## Reproduction and evidence

Use Node 22 and the local Dawn package:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/benchmark-particle-order-scaling.ts --capacity=1048576 --rounds=9 --repetitions=3 --out=artifacts/particle-order-scaling/retained-final.json
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/benchmark-particle-order-scaling.ts --capacity=1048576 --grid=64 --case=activation --rounds=15 --repetitions=3 --out=artifacts/particle-order-scaling/retained-quiet-activation.json
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test tests/uniform-particle-order-merge.test.ts tests/uniform-narrow-band-apic.test.ts
```

Use `--phases` only for diagnostic pass attribution; it adds timestamps at every pass. Timestamp quantization is disabled in the benchmark. The fixture keeps generated positions inside their intended cells so float rounding cannot silently alter nominal occupancy.

Local raw samples and rejected experiment snapshots are under `artifacts/particle-order-scaling/` (ignored artifacts). Principal evidence: `retained-final.json`, `retained-quiet-activation.json`, `retained-quiet-validation.json`, `fused-scheduling-population.json`, and `tile128-population.json`. The checked-in benchmark and this table preserve the method and findings without depending on those generated files.
