# Matched full-fine field comparison

The isolated current-quadratic GPU prototype preserves the translated sphere
in the same two-step experiment that breaks the shipping retained field.
This validates the bounded affine field representation and its GPU evaluation;
it does not replace production physics or validate general fluid deformation.

Both captures use 32³ supports, h=.05 m, a sphere of radius .25 m centered at
(-.15,.8,0), density width .05 m, velocity (.75,0,0) m/s, and dt=1/30 s.
The old arm is the actual full-fine resident capture described in
[the root-cause diagnosis](retained-imposed-flow-diagnosis-2026-09-08.md).
The new arm transports its accepted current numeric quadratic coefficients
on the GPU. Only initialization compiles the analytic quadratic. Motion is
prescribed in this experiment; it is not yet compiled from resident velocity.

`tools/capture-quadratic-imposed-flow-dawn.ts` captures the current coefficient
banks, actual GPU point/gradient/Hessian queries, ray roots, and support amounts.
It checks complete wet coverage before each coefficient publication. Requested
worklists shrink only across independently proven dry columns, retaining the
complete positive-density sphere and every destination donor. The general GPU
coverage checks also pass their missing-center, missing-forward-corner and
uncertain-zero negative controls.

The default capture command runs Dawn in a separate short-lived child, which
exits before independent CPU analysis starts. `--gpu-only` and `--analyze-only`
allow the GPU lease to be returned before analysis when coordinating other
tasks. Run-start source snapshots and hashes accompany the raw binary data.

## Measured field results

| Quantity | Initial | Step 1 | Step 2 |
|---|---:|---:|---:|
| Maximum actual GPU phi error, m | 5.35e-7 | 5.15e-7 | 6.49e-7 |
| Maximum actual GPU density error | 6.84e-7 | 7.07e-7 | 6.92e-7 |
| Maximum density jump across sampled support faces | 1.28e-7 | 2.32e-7 | 3.64e-7 |
| Maximum support mean error | 2.25e-6 | 1.13e-6 | 2.25e-6 |
| L1 support amount error, m³ | 2.08e-8 | 1.70e-8 | 2.14e-8 |
| Relative total amount error | 1.89e-7 | 1.20e-7 | 1.93e-7 |
| Maximum six-axis ray-root error, m | 0 | 0 | 0 |

The global analytic amount is .06577759120626807 m³, computed by an independent
radial primitive. Cell reference amounts use exact vertical integration and
adaptive transverse quadrature; their accumulated estimated error is
5.50e-10 to 5.78e-10 m³. These local estimates are not interval certificates.
The measured support-mean errors are reported, not hidden behind the GPU's
nested-quadrature error estimate. The latter stayed below the unchanged 2e-6
budget and used at most 16,704 exact-axis slices per support in this capture.

Point measurements cover support centers and the same off-center fraction
(.37,.63,.29) throughout the valid grid. The face measurements use nine points
per shared face; coefficient-level global-quadric coherence is separately
checked by the GPU. These observations are not claims of bit-exact continuity
or an exhaustive bound on every possible f32 query.

The old shipping field has density jumps .3613 and .4152 after the two steps,
diffusive native amount errors, and published zeros inconsistent with its own
density. The new field's half-density condition is directly its quadratic
zero; no endpoint inversion or initial-shape lift is used.

## Images and validation

`tools/compare-imposed-flow-fields.py` makes two comparisons from saved data:

* `artifacts/retained-imposed-flow/comparison/density-comparison.png` evaluates
  both captured coefficient fields at identical dense physical points. Rows
  are old production, isolated prototype, and independent analytic density.
* `artifacts/retained-imposed-flow/comparison/gpu-phi-comparison.png` compares
  actual GPU scalar queries at identical finest-cell centers. Linear contours
  of these equally sampled values have the same sampling limitations.

No mesh is consumed, constructed, projected, smoothed, or repaired. The dense
section's new density error is 7.70e-8, 1.25e-7 and 2.29e-7; the old field's
section errors are 0, .407 and .540. Both rows use the same coordinates and
plot scale. The graphics are diagnostic plots, not shipping renderer output.

Validation on Metal:

* Quadric gate: all 15 cases pass, `/tmp/fluid-quadratic-pullback-gpu-8.log`.
* Matched GPU capture: three states completed in .51 s including compilation
  and QA readbacks, `/tmp/fluid-quadratic-imposed-flow-1.log`.
* Independent CPU analysis: 6.80 s, all existing acceptance assertions pass,
  `/tmp/fluid-quadratic-imposed-flow-analysis-1.log`.

These are isolated-process wall times, not production frame benchmarks. The
capture above uses coefficient publication followed by a separate integration
query. The later atomic gate below closes that transaction gap. Compiling motion
from actual immutable resident velocity and non-affine/branch/solid handling
remain required before production adoption. The
[departure-map decision](current-density-departure-map-decision-2026-09-08.md)
describes the next gates and the unresolved bounded-history problem.

## Subsequent atomic field-and-amount gate

`advanceConservative` now writes candidate support amounts into the same GPU
record bank as the candidate potential. Coherence and complete wet coverage
validate first; candidate integration follows. Only a successful receipt
publishes the bank and generation. Words 12–15 hold the amount, quadrature
estimate, work count, and amount generation. Consumers must require the amount
generation to match the coefficient generation: initialization and the older
geometry-only `advance` API do not advertise accepted amounts.

An independent review also reproduced a host race: the old operation released
its lock before the caller flipped the bank. A second microtask could encode
against the same old source while both operations advanced the host epoch.
Completeness validation and bank publication now occur inside the locked
operation, which returns its captured generation. Three CPU interleaving tests
verify sequencing, pending-operation exclusion, and failed-receipt recovery.

Metal passes all 17 quadric cases in 2.87 s, including same-generation field
and amount publication, complete radial mass, coverage-failure rollback,
retry, explicit amount invalidation by geometry-only updates, and a deliberately
unachievable integration budget. The integration-failure case preserves an
initial field with empty amount slots; the coverage-failure case preserves
nonzero accepted amounts. Log: `/tmp/fluid-quadratic-pullback-gpu-9.log`.

No field-sized CPU readback is part of this transport operation. Its fixed-size
receipt is still read on the host in this research API. Quadrature estimates,
the f32 map determinant envelope, and same-field integration remain numerical
contracts, not certified exact conservation or a production physics coupling.
