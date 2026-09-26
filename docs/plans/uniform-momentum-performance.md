# Mixed Uniform momentum performance investigation

## Scope and result

The all-fine mixed pipeline spent several times longer than native Uniform in
momentum, even with every tile certified regular. The change retains the same
h/2h/4h ownership, characteristic integration, interface interpolation, face
eligibility, wall conditions, force terms and pressure acceptance. It adds no
persistent GPU allocation and no fallback or alternate solver.

Controlled source variants on `minimal-power-dam-break-64`, 24 frames with the
first eight excluded, identified two major avoidable costs:

- The default momentum sampler still used `array<vec3f,8>` interpolation and
  vector reductions for velocity/physical-velocity/phase-weight, although only
  velocity was required. Returning `vec3(value,0,0)` and reading `.x` did not
  eliminate its runtime cost on this Metal implementation. Generate a genuinely
  scalar sampler for the already-specialized default numerical options.
- Certified fine interpolation still performed per-tap boundary tests and
  transverse clamps. For an interior unit stencil, use the ordinary eight
  scalar texture loads. The clamped upper-edge out-of-range tap has exactly zero
  weight. Negative boundary samples and uncertified/interface samples retain
  their existing handling.

The final regular-tile dispatch uses 192 lanes: 64 owners times three MAC
components. Each lane evaluates one characteristic, then a workgroup barrier
allows the first 64 lanes to pack the three components into one RGBA store.
This avoids evaluating three large characteristic call graphs per invocation.
The workgroup uses 768 bytes of transient shared storage. The same regular
pipeline is used for certified fine tiles in mixed scenes. Coarse, transition
and uncertified fine tiles continue through the ownership-aware face traversal.

Exact register spilling has not been established with a GPU compiler profiler;
the source A/B timings establish the cost of the vector payload and sampling
control flow, not a particular compiler mechanism.

## Diagnostic captures

Median GPU compute-pass time, milliseconds. The combined column is the median
of each frame's momentum-plus-forces sum, not the sum of stage medians.

| Variant | Momentum + forces |
| --- | ---: |
| Original mixed | 3.473408 |
| Constant certified unit-face geometry | 3.145728 |
| Constant default characteristic iteration limit | 3.112960 |
| Scalar default sampler | 1.474560 |
| Scalar sampler, component lanes | 1.376256 |
| Unconditional regular scalar taps, component lanes | 1.114112 |
| Direct interior loads, original single-owner lanes | 0.753664 |
| Direct interior loads, component lanes | 0.557056 |
| Native Uniform, fresh capture | 0.524288 |

The final diagnostic is about 84% cheaper than the original mixed stage and
within 6.25% of native stage time. These were sequential per-pass-instrumented
runs, not a whole-frame throughput acceptance result. Timestamp granularity,
thermal variation and slightly different floating-point trajectories limit
precision; do not infer bit-exact frame equivalence or the 98% overall target.
Force time was already roughly 0.13 ms; its numerical implementation is unchanged.

Reproduce the all-fine diagnostic with:

```sh
node --import tsx tools/probe-uniform-mixed-pressure-dawn.ts \
  --arm=fine --scene=minimal-power-dam-break-64 --tolerance=5 --steps=24
```

Use `--arm=native` for native. Both acquire the repository GPU lease; run
serially and never alongside browser simulation or another Dawn process.

## Correctness coverage

The momentum matrix compares the general and scalar paths with non-constant
velocity on fine, coarse and graded layouts. It separately compares certified
scalar dispatch against the general path. Existing analytical characteristic,
wall carry, phase filtering and bounded-correction checks retain their limits.
Force tests cover endpoint viscosity, flat-interface balance and gravity.

Validation: momentum and force matrix passed (2/2, including all seven
momentum layouts), and the complete frame/remapping/fail-before-projection
check passed (1/1). No numerical tolerances were changed. Repository-wide
TypeScript checking still fails in unrelated Sparse/SVO work; these changed
files have no reported errors.

A subsequent production ABBA run confirmed combined stage medians of 0.590 ms
mixed and 0.557 ms native (eight distinct hardware traces per arm). Whole-frame
medians were 22.112 ms mixed and 20.398 ms native: 92.25% native throughput,
versus the earlier 71.97% checkpoint. Resident bytes remain 1.01977 times
native. This remains below the overall 98% throughput acceptance target.
Other outstanding pressure, scene support and whole-frame performance work is
tracked in `uniform-mixed-performance-bridge.md` and is outside this change.
