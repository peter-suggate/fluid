# Paper pressure solver: Dawn before/after profile

Date: 22 July 2026. Host: Apple M1 Max, 32 GB. Backend: Dawn/Metal through
`node_modules/webgpu`.

## Compared configurations

`Before` is the retained `aggregate-galerkin` rollback hierarchy. `After` is
the opt-in `paper-pyramid` hierarchy. Both use the same matrix-free L2 PCG,
relative tolerance, cap of 128, eight symmetric boundary-band sweeps, scene,
fixed timestep, and current one-compute-pass command batching. This isolates
the hierarchy change rather than comparing two different command-recording
implementations.

For historical context, the unbatched implementation recorded 15,357
compute-pass transitions at cap 128. Both current configurations record one;
the tables therefore compare remaining ordered dispatch and hierarchy cost.

Runs used `FLUID_PERFORMANCE_PROFILE=1`. This keeps the solver-control,
convergence, and queue-completion measurements, but omits compact cubic field
reconstruction, raster checkpoints, generation audits, and scene quality
gates. `simulationWall_ms` is the production stepping wall time with deliberate
QA sampling removed. Each pair ran serially under the Dawn exclusive lock.

The timestamp-query feature was requested, but this Dawn configuration
published zeroes for every per-stage GPU timestamp bucket. No shader-duration
claim is made from those counters; the evidence below is command encoding,
dispatch count, convergence, and queue-completion wall time.

## Results

| Scenario | Steps | Rollback wall | Paper wall | Wall reduction | Rollback pressure encode | Paper pressure encode | Encode reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hydrostatic two-level, 16³ | 50 | 7.173 s | 4.261 s | **40.6%** | 28.622 ms | 16.860 ms | **41.1%** |
| Hydrostatic large-offset, 32×24×16 | 1 | 53 ms | 43 ms | **18.9%** | 32.161 ms | 20.226 ms | **37.1%** |
| UI dam reference, 24×18×16 | 1 | 51 ms | 38 ms | **25.5%** | 32.660 ms | 19.820 ms | **39.3%** |
| Minimal dam break, 16³ | 100 | 21.637 s | 16.281 s | **24.8%** | 28.412 ms | 16.379 ms | **42.4%** |
| Minimal dam endurance, 16³ | 550 | 102.439 s | 72.653 s | **29.1%** | 27.978 ms | 16.742 ms | **40.2%** |

The one-step wall measurements are construction/readback sensitive, so their
pressure-encode and dispatch columns are more useful than their total wall
ratios. The 100- and 550-step runs are the meaningful end-to-end results.

| Scenario | Rollback dispatches | Paper dispatches | Reduction | Rollback full pressure projection | Paper full pressure projection | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hydrostatic two-level | 18,712 | 10,730 | **42.7%** | 34.735 ms | 23.207 ms | **33.2%** |
| Hydrostatic large-offset | 20,905 | 12,410 | **40.6%** | 38.871 ms | 26.828 ms | **31.0%** |
| UI dam reference | 20,905 | 12,410 | **40.6%** | 39.773 ms | 26.554 ms | **33.2%** |
| Minimal dam, 100 steps | 18,712 | 10,730 | **42.7%** | 34.729 ms | 22.337 ms | **35.7%** |
| Minimal dam, 550 steps | 18,712 | 10,730 | **42.7%** | 33.586 ms | 22.652 ms | **32.6%** |

The reported dispatch count is the fixed host-authored schedule at cap 128;
GPU early convergence prevents numerical work in later kernels but cannot
remove already encoded dispatch commands.

## Convergence and physical parity

| Scenario | Rollback final iterations / residual | Paper final iterations / residual | Result |
| --- | ---: | ---: | --- |
| Hydrostatic two-level | 10 / 3.77e-5 | 10 / 8.43e-5 | Both converged; speeds remained below 3.6e-6 m/s |
| Hydrostatic large-offset | 0 / 0 | 0 / 0 | Both accepted the zero-residual solve |
| UI dam reference, one step | 0 / 0 | 0 / 0 | Both remained at the initialized rest state |
| Minimal dam, 100 steps | 24 / 9.15e-5 | 10 / 4.13e-5 | Paper required fewer iterations at the final generation |
| Minimal dam, 550 steps | 8 / 3.51e-5 | 21 / 8.09e-5 | Paper required more iterations at the final generation |

Every paired run converged below `1e-4`, published no non-finite value, and
completed its exact step count. At 100 steps, final maximum speeds were
0.849953 and 0.849954 m/s. At 550 steps they were 0.0279477 and 0.0279467 m/s.
The long-run iteration reversal is material: the paper hierarchy is
substantially cheaper overall, but it is not uniformly a stronger
preconditioner for every topology generation. Cap 128 must remain the default.

A separate reconstruction-enabled 100-step A/B quality run measured one
connected liquid component on both paths, about 0.108% volume drift, and only
3.3e-8 absolute difference between their drift values. The standard quality
harness still encounters the pre-existing coarse/fine publication-generation
mismatch before some scenarios can be graded; performance mode deliberately
does not conceal or reclassify that issue.

## Conclusion

Across all measured Dawn pressure scenarios, `paper-pyramid` reduced the
fixed dispatch schedule by 40.6-42.7% and pressure encoding by 37.1-42.4%.
The dynamic runs reduced production stepping wall time by 24.8-29.1%. This is
a large, repeatable improvement and preserves convergence at the safe cap.

Keep the solver as an opt-in A/B until the following are complete:

1. cache the native pyramid by accepted topology generation instead of
   rebuilding it for every pressure solve;
2. resolve the upstream coarse/fine generation mismatch so the full quality
   matrix can run without performance mode;
3. characterize the topology generations where the paper hierarchy takes
   more PCG iterations, especially the 550-step final generation;
4. repair or replace the zero-valued Dawn per-stage timestamp publication so
   shader execution time can be separated from queue/driver overhead.
