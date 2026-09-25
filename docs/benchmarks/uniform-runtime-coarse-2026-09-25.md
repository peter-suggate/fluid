# Uniform Geometric: live coarse activation proof

The solver can switch an evolving h simulation to a full 4h simulation without
resetting its liquid or clock, constructing resources, or calling shader/pipeline
compilation APIs at activation. This is opt-in. The original measurements below cover one-way activation; the
subsequent UI and return-to-fine validation is recorded in the round-trip section.
Arbitrary solids/terrain and render performance are not certified by this proof.

Design and remaining work: [runtime resolution plan](../plans/uniform-runtime-resolution.md).

## What the GPU tests prove

Dawn/Metal, the mini64 dam-break box and a 128³ version of the same physical box:

1. Prepare both configurations, sharing stage scratch and rigid-system ownership.
2. Advance six fine steps and verify the fine liquid has actually changed.
3. Request coarse mode and process the request at the current simulation time.
   Activation consumes no timestep and preserves the existing step count.
4. Compare every coarse V cell with an independent CPU sum of its 64 current fine
   cells. Verify total physical volume, finite moving velocity, and a pressure
   residual within the configured tolerance.
5. Advance six coarse steps. Verify continued evolution, conserved liquid,
   continuous time/counters, and a byte-identical inactive fine volume field.
6. Intercept buffer/texture creation and shader/pipeline compilation calls: none
   occurs in the transition, diagnostics read, or first/subsequent coarse steps.

Both fixed and production adaptive (`lagged`) pressure schedules pass. A separate
GPU fixture verifies nonuniform cell restriction, all three MAC components,
negative domain face planes, and signed-distance values in physical metres.
A CPU test verifies that activation cannot take the arena from a pending fine
pressure frame.

The tests use the configured production pressure tolerance, 5 s⁻¹ in these runs,
without changing it at activation. Accepted transfer residuals were approximately
0.0691 s⁻¹ at 64³ and 0.5106 s⁻¹ at 128³. An additional exploratory 10⁻⁴ run failed:
the fixed transfer reached 1.52×10⁻⁴, and the adaptive fine simulation failed to
converge before the switch (0.00725 after seven cycles). This milestone does not
certify that tighter setting. Those exploratory results preceded concurrent
pressure-solver work elsewhere in the workspace.

## Memory

Actual live GPU buffer and texture payload bytes were counted through device
allocation and destruction calls, before and after coarse preparation. These are
not process RSS or driver-private pipeline allocations. The existing solver info
census reports a slightly smaller estimate; the independent count is authoritative
for the figures below.

| Fine → coarse | Fine resources | Additional resources | Increase |
| --- | ---: | ---: | ---: |
| 64³ → 16³ | 54,891,244 B | 703,512 B | 1.282% |
| 128³ → 32³ | 430,328,112 B | 4,760,776 B | 1.106% |

Preparation enforces the planned 3% limit using the solver census, and the GPU
proof independently asserts the 3% limit against actual payload allocation.
The fine allocations remain resident after switching. There is no claim that
coarse mode reduces resident memory or supports a scene exceeding fine capacity.

## Timing observations

Queue-fenced wall times include CPU encoding/submission and GPU execution;
startup compilation and rendering are excluded. Fine medians use steps 2–6;
coarse medians use steps 7–12. These are different segments of the evolving flow,
so they are workload observations rather than matched-trajectory speedup claims.

After adding startup transfer/projection warmup:

| Fine size | Pressure schedule | Fine median | Coarse median | Transition |
| --- | --- | ---: | ---: | ---: |
| 64³ | Fixed | 30.115 ms | 16.029 ms | 10.172 ms |
| 64³ | Adaptive | 15.453 ms | 7.078 ms | 10.515 ms |
| 128³ | Fixed | 87.419 ms | 20.407 ms | 15.543 ms |
| 128³ | Adaptive | 63.716 ms | 11.699 ms | 15.659 ms |

Before startup warmup, the first fixed-schedule 128³ transition took 67.131 ms.
Warmup runs restriction, surface correction and projection over disposable
coarse startup state; it does not advance the fine simulation. It adds no
persistent storage or fine-frame passes. The first complete coarse physics step
at 128³ still took 49.925 ms in the fixed case, so a guaranteed interactive frame
budget for every first-use path is not established. Raw samples retain that
outlier; it is not hidden by the median table.

The first fine-mode ABBA comparison, before transfer warmup, passed the planned
2% tolerance: 19.871 ms baseline versus 19.900 ms with prepared coarse resources
(+0.150%). Each arm is a fresh solver instance, using 30 frames with the first
five excluded, in unprepared/prepared/prepared/unprepared order. The benchmark
asserts the mean of the two prepared medians does not exceed the corresponding
baseline by more than 2%.

The first post-warmup comparison failed that unchanged threshold: 22.427 ms
baseline versus 23.855 ms prepared (+6.37%). An exact repeat passed, measuring
27.099 ms baseline versus 20.596 ms prepared (−24.00%). Control-arm medians
across these two runs ranged from 20.049 to 34.148 ms. That variability prevents
a conclusive no-regression claim; the apparent speedup on the repeat is not
evidence that preparation accelerates fine simulation. Both results are retained
below. Concurrent pressure-solver revisions also limit comparisons with the
earlier pre-warmup run. A stable performance validation remains outstanding.

Raw proof samples include pre-warmup and post-warmup runs, explicitly tagged:

- [64³ results](uniform-runtime-coarse-64-2026-09-25.json)
- [128³ results](uniform-runtime-coarse-128-2026-09-25.json)
- [Fine-mode ABBA](uniform-runtime-coarse-abba-2026-09-25.json)
- [Post-warmup ABBA: failed threshold](uniform-runtime-coarse-abba-postwarm-first-2026-09-25.json)
- [Post-warmup ABBA: repeat](uniform-runtime-coarse-abba-postwarm-repeat-2026-09-25.json)
- [Latest workspace activation recheck](uniform-runtime-coarse-current-2026-09-25.json)

After the concurrent pressure changes in base commit `46a6842f`, both size-specific
GPU suites were rerun successfully (3/3 tests each). The last link preserves those
four fixed/adaptive activation samples separately from the timing table above.

## Other validation

All 50 non-Dawn Uniform CPU tests pass. Typechecking still reports errors outside
this change; none are in the runtime-coarse implementation, test or benchmark.
`git diff --check` passes.

The unchanged canonical Sparse CM12 suite was run: **5 of 17 lanes passed**.
It is not a passing regression certificate. Nine lanes exceeded their existing
process timeouts; mini64 performance measured 211.092 ms against its unchanged
110 ms ceiling; the terrain lane halted at frame 35 in
`addWholeFrameUncoveredDonorFallbacks`; and outside-tank collapse halted at frame 1
with `MISSING_COMPILED_TOPOLOGY_FACE`. These are in the separate adaptive-volume
implementation. This work did not change those numerical paths or relax any
lane, timeout or ceiling. No baseline reproduction is claimed.

[Complete canonical gate receipt](uniform-runtime-coarse-sparse-gate-2026-09-25.json).

## Reproduction

Run sequentially, with no browser or other Dawn workload holding the GPU lease:

```sh
npm run test:dawn:uniform-runtime-coarse
UNIFORM_COARSE_TEST_SIZE=128 npm run test:dawn:uniform-runtime-coarse
npm run benchmark:uniform-runtime-coarse
```

The solver opt-in is `prepareCoarseSimulation: true` during `createAsync`.
While fine physics is running, call `requestCoarseSimulation()`; the next
completed-step boundary activates it. Inspect `simulationCellScale` (1 or 4)
and the public active textures/info. An idle caller can instead call
`activateCoarseSimulation()`, which returns false while a fine frame is pending.
Call `requestFineSimulation()` to reconstruct the current coarse solution on
the retained fine lattice at a completed-step boundary.


## UI and return-to-fine follow-up

In the simulation panel (`S`) or solver setup, select **Prepare coarse switch
(restart)** once. Then **Switch to coarse** and **Switch to fine** operate on the
current liquid without restarting. The button reports a pending request until
the active grid changes. **Disable preparation (restart)** explicitly restores
an unprepared fine simulation. Preparation stays off by default.

The return reconstructs V from current coarse cell fractions, interpolates MAC
velocity along each face normal (preserving each parent face's integrated flux),
and trilinearly interpolates physical vertex phi. It then corrects surface volume,
rebuilds authority and projects velocity on the fine grid. It cannot recreate
fine detail discarded by coarsening. Additional retained GPU storage beyond the
one-way version is only 408 bytes of fine diagnostic/pressure receipts.

Both the 64³ and 128³ suites pass all three tests. Each evolving test now advances
six fine steps, six coarse steps, returns to fine without changing time or step
count, advances six fine steps, and performs three further round trips. It checks
volume conservation, projected finite velocity, configured pressure tolerance,
and no allocation or compilation API calls during either switch or subsequent
steps. The independent fixture also checks positive/negative MAC flux round trips
and exact interpolation of a linear signed-distance field. Ten targeted CPU
control/lifecycle/scratch tests pass. Typecheck retains unrelated Sparse errors.

| Fine size | Additional GPU bytes | Increase | Fine → coarse | Coarse → fine |
| --- | ---: | ---: | ---: | ---: |
| 64³ | 703,920 | 1.282% | 10.27–11.20 ms | 26.12–28.31 ms |
| 128³ | 4,761,184 | 1.106% | 15.33–15.39 ms | 63.76–70.43 ms |

These are queue-fenced fixed/adaptive samples, not guaranteed frame budgets.
Returning to fine touches fine-sized fields and runs fine pressure projection;
no-rebuild switching does not imply a sub-16-ms return. Return residuals were
0.760 and 1.588 respectively, below the configured 5 s⁻¹ threshold. Per-cell V
prolongation error was zero in all four samples.

[Round-trip raw measurements](uniform-runtime-roundtrip-2026-09-25.json).


The final round-trip version's fresh-instance ABBA fine-mode benchmark passed the
unchanged 2% threshold: 20.269 ms baseline, 20.463 ms prepared (+0.960%).
[Raw final ABBA samples](uniform-runtime-roundtrip-abba-2026-09-25.json).
The earlier noisy results remain above; this is one passing comparison, not a
universal performance guarantee.


Browser verification on the 64³ mini dam used the actual buttons: preparation,
Play, Switch to coarse (16³), Switch to fine (64³), and continued fine evolution.
Time advanced continuously through both live switches. Both directions were also
verified while paused with unchanged time. A paused transfer now publishes its
new grid diagnostics without waiting for a physics advance. The retained fine
surface renders after return. The complete non-Dawn Uniform test selection passes
52 tests; the unrelated untracked work-topology test was excluded.


Final GPU reruns also pass 3/3 at each size after the paused-control fix. Their
three repeated round trips call `applyRuntimeValues` exactly as the UI does,
without calling `advanceTo`, and preserve time, step count and liquid volume.
Raw measurements distinguish these from the earlier request-API samples.
