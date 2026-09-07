# Test audit — 8 September 2026

`npm run test:unit`: **1,200 passed, 0 failed, 170 skipped** (47.2 seconds).

## Deleted

Two obsolete tests prohibited any scene profile from choosing a timestep other than the CM12 paper default. Analytic motion, standing-wave and stationary-bowl study scenes intentionally select their authored timestep. The method-default tests and analytic-scene tests remain. The broad “all authored scenes use 1/30 s” assertion now checks the default scene document; transport synchronization still rejects incompatible pane timesteps.

Removed the runtime skip that treated an uninitialized accepted GPU generation as a skipped test after submission. This now fails explicitly after releasing its resources.

## Retained skips

All 170 skipped cases are GPU regressions gated on an enabled Dawn runtime. There are no permanently disabled test declarations to delete. They cover current simulation, topology, publication, capture, rendering and editing behavior. They remain available through the explicit GPU scripts. No GPU case ran in this CPU verification; another task held the browser QA WebGPU lease.

The SolidWorld boundary regression previously started a GPU probe unconditionally from the CPU suite. It now uses the same Dawn-environment guard as the other GPU regressions, preventing accidental contention with a browser run.

## Repaired relevant failures

- SVO URL persistence exercises a nondefault refinement depth; depth three is now canonically omitted from URLs.
- Resizing the open tank keeps five solid walls and an open top.
- Global-fine payload accounting uses the packed one-word sample ABI. Volume accounting includes correction/residual indirect buffers and the full reference control.
- Power catalog accounting includes the compiled interpolation sampler in addition to the serialized artifact. Exact byte accounting and row-capacity scaling replace the obsolete pre-compilation 16 MiB assertion; no production allocation or performance limit changed.
- Pipeline ABI tests await explicit asynchronous initialization before examining reflected bind groups.
- Sparse transport retains existing source rungs and allocates new support at the coarse rung.
- Timing labels are checked against registered dispatch/copy counts rather than an obsolete snapshot.
- Height publication assertions follow the continuous-column helper and coarse-page fallback.
- Pressure aperture assertions account for separating closed-world rows; transport apertures remain SolidWorld-owned.
- Host catalogue assertions follow the bounded active/apron selection. Resource recipes reserve every B1/B2/B4/B8 variant, not just the accepted cell.
- Transaction checks use the vector correction cache, mode-specific planning entry point and current receipt scope; ordering and rejection checks remain.
- The rigid-body host may synchronize body buffers; physics dispatch still crosses the public world boundary.
- Fine-page lookup tests retain loop-free direct addressing checks and admit the explicitly tagged compact signed lookup.
- Factor-eight orchestration tests read the current Power lane owner and recognize packed writes and the shared box-SDF helper.
- Reviewed atomic functions now include compact boundary-census publication, its count reader, and the all-parent summary reducer. Numerical values are not accumulated through new floating-point atomics.
- Effective transport assertions follow relative-coordinate sampling while retaining region-span and cached-policy checks.

One fresh run also caught an undefined shader variable while another task was editing rigid coupling. That owner corrected its work during this audit; this task did not stage or commit the concurrent implementation.

## Limits

The global TypeScript check still reports unrelated existing/in-progress errors, with none in the files modified by this audit. No numerical oracle, simulation tolerance, Dawn lane or timing ceiling was weakened. Changes in this audit are limited to tests and this record. Only this task's patches were staged and committed.
