# Uniform velocity extension — 2026-09-24

Figure 9 (`cm12-figure-9`), Uniform Geometric, 128×128×64, Apple M1 Max / Dawn Metal.
The complete extension stage fell from **7.67 ms to 4.78 ms median (38%)**.
The requested **3 ms target is not met**. The optimized p90 is 4.98 ms; the
slowest measured frame was 5.37 ms, so this is not a strict sub-5-ms bound either.

## Measurement

Production simulation at 30 Hz, 180 advances, first 120 excluded, 60 measured
samples. GPU hardware timestamps, queue-fenced execution, rendering excluded.
The complete extension measurement sums the three production trace phases for
each sample before computing its median. Per-pass instrumentation is a separate
run: summing per-pass medians would omit work attributed to the production phase
boundaries and is not the headline measurement.

| Production trace phase | Before median (ms) | After median (ms) |
| --- | ---: | ---: |
| Interface authority | 0.655 | 0.655 |
| Narrow-band FIM front | 4.522 | 2.097 |
| Hierarchy fill + transport shell | 2.425 | 2.032 |
| Complete extension, median of sample sums | 7.668 | 4.784 |
| Complete extension, p90 | 8.061 | 4.981 |

Whole-advance CPU wall medians were 47.44 ms and 47.02 ms. Other stages varied
between these runs and the uninterrupted trajectories differ in FP32 roundoff;
do not interpret the extension saving as an established equal whole-frame saving.

Both arms use the same working tree, including independent existing changes to
other stages. Only the four extension switches below differ. Pressure iteration
budgets, tolerances, multigrid kernels, and the two front sweeps are unchanged by
this work.

## Causes and changes

- Inactive faces repeatedly recomputed a neighboring active face's convergence,
  including its Godunov distance solve. A preparation pass now calculates each
  active face once; its update and adjacent faces share that result.
- The first sweep checked for converged neighbors even though all active seed
  distances are infinite. A specialized first update skips this impossible case.
- Seed work repeatedly reconstructed source/open classifications for six neighbors
  of each of three components. One classification pass supplies these masks.
- Front sweeps reloaded geometry that was already represented in the seeded
  distance and known-value state. Closed faces remain infinity/unknown and sources
  remain zero distance during an invocation, allowing those repeated reads to be
  removed. Inactive activation still checks face openness.
- Recipient distance solves now occur only after finding an eligible converged
  upwind neighbor.
- The final hierarchy transfer repeatedly loaded the same coarse velocity and
  source bounds. A 4³ fine workgroup now shares its 4³ coarse stencil (about 3 KiB).
  This path is restricted to native, unwindowed, source-aware grids with dimensions
  divisible by four. Other layouts retain direct sampling.

Representative individual-kernel medians: first front update 1.31 → 0.33 ms;
second update 2.16 → 0.79 ms, with a 0.13 ms convergence preparation; final hierarchy
transfer 1.44 → 1.05 ms. Source classification plus seeding total about 0.52 ms,
versus 0.85 ms for the previous seed pass.

The front cache costs **16 MiB** at this resolution. It is reused for source masks
and convergence results within each invocation. The two-sweep schedule adds two
compute passes: source classification and one convergence pass. Scratch byte and
encoded-pass reporting include these additions.

Large shared-memory caching of the front stencil was slower and was discarded.
Loop unrolling and a brick-planar convergence-buffer layout did not improve the
measurement and were also discarded.

## Validation

Passed all three Dawn tests (49 seconds total):

- `tests/uniform-extension-symmetry-dawn.test.ts`: reflections, X/Z transpose, closed
  walls, interior/wall-adjacent sources; now exercises 1, 2, and 4 front sweeps.
- `tests/uniform-nearest-extension-dawn.test.ts`: remote-pool/drop isolation,
  predicted velocities, fused/unfused packing, dense/full/empty shells.
- `tests/uniform-system-extension-dawn.test.ts`: partial atlas dimensions, terrain,
  live solid insertion/removal, moving bodies, and per-step field comparisons.

No test tolerances were relaxed. In the system/extension comparison, extrapolated
velocity, volume, and vertex phi matched exactly on each shared physical input.
The existing combined system comparison's maximum velocity difference was
1.52e-6 (maximum relative L2 1.10e-7), within its existing tolerance.

After the uninterrupted six-second Figure 9 run, raw volume drift was -0.241%
before and -0.255% after; represented-volume drift was -0.582% and -0.196%.
Full-trajectory field hashes differ; bit identity is not an acceptance criterion.

`npm run check:types` still reports pre-existing errors in unrelated sparse tests
and probe tools; none reference the changed extension, benchmark, or test files.
`git diff --check` passed. No Sparse CM12 code was changed, so the large sparse
refactor gate is outside this change's scope.

## Reproduce

Stop browser simulation and other Dawn runs first; the benchmark requires the
repository's exclusive WebGPU lease.

```sh
# Baseline production stage timestamps
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_UNIFORM_AB_OFF=frontactivate,frontreceipts,frontstate,extensioncache \
UNIFORM_BENCH_STAGE=extension UNIFORM_BENCH_OUTPUT=/tmp/extension-before-stages.json \
node --import tsx tools/benchmark-uniform-sharpening-dawn.ts

# Optimized production stage timestamps
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
UNIFORM_BENCH_STAGE=extension UNIFORM_BENCH_OUTPUT=/tmp/extension-after-stages.json \
node --import tsx tools/benchmark-uniform-sharpening-dawn.ts

# Add UNIFORM_BENCH_PASSES=on for individual extension kernel timestamps.

WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx --test --test-concurrency=1 \
tests/uniform-extension-symmetry-dawn.test.ts \
tests/uniform-nearest-extension-dawn.test.ts \
tests/uniform-system-extension-dawn.test.ts
```

## Next work toward 3 ms

The front and hierarchy phases are now both about 2 ms. Getting below 3 ms needs
another ~1.8 ms across the complete stage. The next candidates are fusing the
initial front update with seeding, reducing the numerous small coarse hierarchy
passes, and reducing the final nearest-source interpolation's arithmetic. Review
the falling drop and pool interaction before considering a cheaper interpolation
rule; preserving their source isolation is covered by an existing regression test.
