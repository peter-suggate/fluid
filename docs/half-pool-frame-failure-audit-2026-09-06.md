# Half-pool frame failures and rejected topology candidates

The native Dawn run of `coarse-first-pool-impact-half` covers every frame from
reset through 3 seconds: balanced quality, coarse-first, B8 ladder, 0.05 m finest
spacing, no enforcement region or topology freeze, 180 steps at 1/60 second.
The shader hash is `a53e787d93300db2f8f46d5d941387a9babac5bb737d9d1a3fab03a46e32bc0b`.
This is a reproducible default-scene audit, not a capture of the user's browser.

`POOL_AUDIT_FAILURES=1` adds read-only receipts after every encoded step. These
include pressure diagnostics, accepted-frame control, topology candidate effects,
final-scalar publication, velocity extension and renderer-facing publication.
The complete receipts and summary are in the ignored capture directory
`artifacts/pool-impact-ab/adaptive-uniform-ab/`.

## Completed-frame and pressure results

- All 180 pressure solves converged. None reached its iteration cap or reported
  curvature breakdown. The maximum iteration count was 56; the maximum true
  relative residual was 0.0009970253, below the configured 0.001 tolerance.
- All 180 accepted-frame commit receipts advanced exactly once per step.
- Every surface publication generation matched its step, and every dirty page
  was executed and published. There were no omitted pages, coverage faults or
  publication faults.
- No topology transaction commit failed. Candidate-effects, final-scalar and
  velocity-extension fault receipts were also zero.

These results do not support a failed solve or full-frame rollback explanation
for this run's surface changes. Successful frames can still contain a local
rejected topology candidate, or a representation discontinuity in a successfully
published surface.

## Local topology rejection is a real planning bug

Activity fault bit 1 occurred on steps 38–42 (0.6333–0.7 seconds) and step 91
(1.5167 seconds). This bit covers candidate validation; it is not exclusively a
2:1 grading failure. Inspecting the actual candidates distinguishes the cause:

| Steps | Rejected leaves | Accepted rung | Requested rung |
| --- | ---: | ---: | ---: |
| 38, 40, 41, 42 | 4 each | 8 | 5 |
| 39 | 4 | 8 | 6 |
| 91 | 8 | 8 | 6 |

Rungs 5 and 6 are invalid. `measureBrickActivity` stores a numeric curvature rung
in bits 16–20 of the reason word. `preserveActivityHorizontalD4` combines the
whole reason words with bitwise OR. Consequently numeric requests 1 and 4
become 5; requests 2 and 4 become 6. The downstream validator correctly rejects
these invalid requests. Valid updates elsewhere still commit (8, 20, 12 and 16
leaves on steps 40, 41, 42 and 91 respectively).

The topology agent is correcting this aggregation to take the maximum of the
numeric rung field while ORing the independent boolean reasons. A subsequent
run must demonstrate the invalid requests disappear. The failed requests delay
coarsening of those already-fine leaves; their exact contribution to visible
popping is not established by this diagnostic alone.

## Reproduction

Unload Fluid browser tabs and run Dawn sequentially under its exclusive lease:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
POOL_STEPS=180 POOL_AUDIT_FAILURES=1 \
POOL_OUTPUT=artifacts/pool-impact-ab/failure-audit \
node --max-old-space-size=12288 --import tsx tools/probe-pool-impact-ab-dawn.ts
```

## Correction and full-run verification

`mergeActivityReasonWords` now takes the numeric maximum of bits 16–20 and
unions every other bit. The curvature measurement, selector thresholds, proof
requirements and 2:1 closure are unchanged.

The regression `tests/sparse-cm12-activity-d4-rung-dawn.test.ts` executes the
production `preserveActivityHorizontalD4` function over eight symmetry members.
It tests differing rung pairs, unrelated low/high flags, and the existing
score/history reductions. The saved original production function fails on
rungs 1 and 4 (`5 !== 4`); the corrected function passes all cases. This checks
the actual aggregation, not a duplicated CPU implementation.

The 180-frame half-pool rerun in
`artifacts/pool-impact-ab/adaptive-d4-only-fixed` retained the preceding
presentation implementation so this comparison isolates the D4 correction
from the concurrent fine-column display experiment. All 180 frames committed.
There were **zero activity fault flags and zero rejected candidates**, replacing
six faulty epochs and 28 rejected leaf candidates. Frame, presentation,
candidate-effects, scalar-mask and velocity-extension fault receipts also
remained zero.

The four central leaves now successfully coarsen at step 38, when the former
planner requested rung 5 and committed nothing. Their successful transition
advances the accepted generation rather than repeatedly deferring the request.
Subsequent accepted topology and wave trajectories change. At 3 s, the original
run had 43,428 cells and diagnostic kinetic energy 275.805; the corrected run
had 48,000 cells and energy 500.775 (same diagnostic units). Final mass differed
by -0.00605% between the runs. This demonstrates the invalid requests were
physically consequential, but the resulting energy difference is not an
isolated measure of dissipation: the subsequent discretization, wave phase and
ordinary solver history all differ.

A separate combined D4-plus-seven-sample-presentation capture in
`artifacts/pool-impact-ab/adaptive-d4-fixed` also completed 180 frames with zero
activity faults or rejected candidates. It is not the isolated planning A/B.
The canonical combined regression gate is still required after all concurrent
physics and display changes settle.

## Activity telemetry bookkeeping

The audit also exposed impossible aggregate activity counts (for example,
4,294,966,754 hot bricks in a historical frozen capture). The D4 publication
replaced per-brick scores and reasons without replacing their contributions in
the incremental histogram. The next measurement subtracted the new score's
contribution from a histogram that still contained the old one, so unsigned
counts could underflow.

`commitActivityHorizontalD4` now replaces registered census contributions before
publishing the new score/reason word. Unmeasured leaves remain unregistered.
This changes diagnostics only; these aggregate counters are not solver or
selector inputs. The production-WGSL test alternates hot/quiet scores forty
times and checks the final histogram and registered population. The previous
commit function fails that test; the corrected function passes.
