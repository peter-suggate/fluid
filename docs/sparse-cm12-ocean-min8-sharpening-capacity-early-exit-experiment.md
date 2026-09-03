# Ocean min8 sharpening: capacity early-exit experiment

Status: rejected as a performance optimization; retained as an isolated QA arm.

## Question

The ocean-seiche B8/P8 full-domain min8 profile attributed about 4.46 ms to
surface sharpening. Its two dominant components were the sharpening transform
(about 2.88 ms) and eight rounds of conservative density-capacity repair
(about 1.38 ms). This experiment asked whether the repair reaches an exact
floating-point fixed point before its eight-round ceiling.

The experiment is general adaptive logic. It contains no scene, region ID, or
resolution-rung condition.

## Fixed-point evidence

`tools/probe-ocean-seiche-sharpening-capacity-rounds-dawn.ts` runs independent
fresh arms, truncating only the sampled final frame after one through eight
repair rounds. At step 12, every arm produced the same destination density hash
and the same destination gamma hash:

- density: `4899946b3bf36f6e467b1f76669cd4d6a237c44e19c5605d47d4fb045ee15383`
- accepted cells: 4,800
- accepted rows: 15,400
- validation errors: none

The integer sharpening-receipt scratch hash continued changing. Those receipts
are overwritten before their next physical consumer, and none changed a
destination density bit after the first round. The full receipt is preserved
in [the round probe artifact](../artifacts/ocean-seiche-sharpening-capacity-rounds-20260902.json).

## Persisted QA arm

`--capacity-early-exit=1` selects a construction-only solver specialization.
It keeps the first two rounds unconditional. The second round publishes one
continuation bit when any finalizer changes a destination density bit. Rounds
three through eight retain the accepted-cell dispatch ordering and exact
integer arithmetic, but return after one gate load when their predecessor was
bit-inert. The eight-round ceiling remains unchanged.

This shape was deliberately branch-only. Using the activity arena itself as an
indirect source would violate WebGPU's writable-storage/indirect aliasing rule;
a copy-isolated design would introduce a pass break and GPU copy between every
dependent round.

## Paired result

Both arms used 8 warmup and 24 timestamped frames, B8/P8, a full-domain min8
region, production coarse transport, production compact policy planning, and
disabled final readback QA.

| Median | Control | Early-exit QA | Change |
|---|---:|---:|---:|
| Density-capacity repair | 1.3107 ms | 1.3763 ms | +0.0656 ms |
| Surface sharpening | 4.2598 ms | 4.4564 ms | +0.1966 ms |
| Non-pressure total | 24.6415 ms | 24.7726 ms | +0.1311 ms |
| Frame | 29.8844 ms | 30.3432 ms | +0.4588 ms |

Terminal cell/row counts, pressure work, and both pressure residuals were exact
between the arms. Both had empty validation-error arrays and passed the probe's
diagnostic closure.

- [Control artifact](../artifacts/ocean-seiche-min8-capacity-control-20260902.json)
- [Early-exit artifact](../artifacts/ocean-seiche-min8-capacity-early-exit-20260902.json)

## Conclusion

The physical field reaches a fixed point early, but dispatching the accepted
domain and loading a gate in every lane costs at least as much as letting this
small 4,800-cell repair finish. This arm must not be promoted.

The remaining high-value sharpening direction is locality-preserving work
reduction in the transform. A particularly safe next experiment is to move
only `prepareSharpeningField` to the accepted-cell stream: it walks incidence
but performs no TEI owner sampling, while the expensive trace/scatter can keep
its packet-local staged directory. That avoids repeating the previously
rejected cache-free accepted-cell scatter experiment.
