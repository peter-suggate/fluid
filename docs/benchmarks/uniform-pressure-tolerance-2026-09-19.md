# Uniform pressure cycle tolerance

The pressure stage now checks the fine-grid projected residual infinity norm (s⁻¹) after each complete Full-Cycle and top-level V-Cycle. A residual at or below tolerance stops all subsequent cycles through direct-dispatch shader exits. Nested coarse solves retain their own existing convergence rule. No CPU readback or new indirect dispatch is required.

The Multigrid cycles stage exposes a live residual tolerance slider, default 0.0001 s⁻¹; zero disables early stopping. Existing cycle counts become caps. Each checkpoint measures the residual and runs a one-thread decision kernel; pressure is canonicalized to texture A before checking, so skipped ping-pong passes cannot invalidate the final pressure. Setup and final diagnostics always execute. Status resets every solve. The norm is the existing projected LCP residual, not the unconstrained linear residual.

## Short Metal benchmark

`tools/benchmark-uniform-pressure-tolerance-dawn.ts` runs fresh mini64 solvers in A/B/B/A order, 12 frames each, excluding the first three from medians. Hardware timestamps bracket the pressure hierarchy (setup, cycles, and finish); wall time includes the whole step. Nonempty marker dispatches are required for reliable Metal timestamps. Balancing is off and velocity transport is semi-Lagrangian. These short runs establish mechanism and initial cost, not long-run quality.

| Case | Tolerance | Median pressure time across arms | Completed cycles |
|---|---:|---:|---|
| Dam | 0 | 52.95–53.94 ms | 3 Full + 4 V |
| Dam | 0.001 | 50.79–54.46 ms | Initial frames skip V cycles; later frames use all cycles |
| Resting pool | 0 | 47.45–48.63 ms | 3 Full + 4 V |
| Resting pool | 0.001 | 38.14–38.86 ms | 3 Full + 0 V throughout |
| Resting pool, stricter comparison | 0 | 48.89–52.23 ms | 3 Full + 4 V |
| Resting pool, stricter comparison | 0.0001 | 51.51–53.81 ms | 3 Full + 4 V |

The resting pool benefits by about 20% in the pressure stage at 0.001 s⁻¹. At the conservative default, its measured residual stays around 0.0008–0.0011 s⁻¹, so it does not stop early and pays checking overhead. The dam has no convincing aggregate speedup over this short window. Keep the tolerance explicit rather than silently relaxing accuracy to report a speedup.

Reproduce with `PRESSURE_TOLERANCE=0.001 node --import tsx tools/benchmark-uniform-pressure-tolerance-dawn.ts`, adding `--rest` for the pool. Raw frame timings, residuals and cycle counts are in the adjacent JSON.

## Validation

All 11 uniform-volume Dawn checks pass, including default-tolerance convergence on a zero-gravity pool, exact nonzero pressure equality with explicit one-Full-Cycle and one-V-Cycle schedules, live disabling/reset, hydrostatic stability, and mini32 far-wall impact. All 32 scene/configuration tests pass. Typechecking still reports pre-existing errors in unrelated tests; none are in the changed pressure files or benchmark.
