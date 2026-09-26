# Minidam64 extension and multigrid performance

Benchmark: `minimal-power-dam-break-64`, production Uniform Geometric defaults.
This follows [the momentum investigation](uniform-momentum-performance.md).
Architecture and overall acceptance targets remain in
[the mixed performance bridge](uniform-mixed-performance-bridge.md).

## Causes and implementation

Velocity extension used dynamic-width ownership and general face traversal for
seed, sweep and publication even on certified fine tiles. It now reuses the
frame's frozen regular/interface dispatch lists and width-specialized pipelines.
Certified fine faces have constant unit geometry and a direct neighbor stencil;
interfaces retain the same patch traversal and interpolation. The two sweeps,
negative boundary values, far hierarchy and publication semantics are unchanged.
No persistent GPU storage is added.

Coarse simultaneous multigrid visits copied global pressure into a shared
snapshot, updated it, and wrote it globally on every Jacobi iteration. Where a
whole level fits in one workgroup with one cell per lane, each lane now keeps
its current value privately, publishes the same shared snapshot, and writes
global pressure once at the end. Two workgroup barriers separate snapshot
publication and consumption. The same arithmetic, sweep count, pressure floor,
convergence gate and hierarchy are retained. Larger visits keep their existing
multi-cell scheduling; red/black smoothing is unchanged. This scheduling
improvement applies to native Uniform and the mixed continuation alike.

Experiments with regular pressure neighbor addressing and liquid/air theta
arithmetic did not show a reliable benefit and were removed.

## Measurements

Per-pass GPU timestamp diagnostic, 24 frames, first eight excluded. These are
sequential captures, not an overall throughput acceptance result.

| Work | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Extension seed | 0.262 | 0.066 |
| Extension sweeps (two total) | 0.786 | 0.328 |
| Extension publication | 0.393 | 0.262 |
| Coarse multigrid fused visits | 0.786 | 0.557 |

Extension restriction and its native hierarchy are additional, unchanged work.
The frame-24 accepted residual was exactly `1.7166051864624023` in these all-fine
captures before and after. Full-field parity and complete-frame acceptance are
separate checks; matching a residual alone is not sufficient.

Reproduce per-pass diagnostics:

```sh
node --import tsx tools/probe-uniform-mixed-pressure-dawn.ts \
  --arm=fine --scene=minimal-power-dam-break-64 --tolerance=5 --steps=24
```

Run production ABBA (native, mixed, mixed, native) with stage timestamps:

```sh
node --import tsx tools/benchmark-uniform-mixed-production-dawn.ts \
  --mode=all --diagnostic --profile
```

`--diagnostic` reports the existing 98% fine/one-air throughput and 3% memory
limits without asserting them. It does not redefine acceptance.

## Correctness checks

- `tests/uniform-mixed-extension-dawn.test.ts`: constant and varying supported
  velocity, live fine/mixed/coarse/fine layouts; specialized versus general
  dispatch must agree within the existing `1e-6` tolerance.
- `tests/uniform-pressure-local-visit-dawn.test.ts`: run minidam64 through native,
  all-fine mixed and one-air-tile mixed layouts; substitute the previous texture
  visit at pipeline compilation and compare volume, velocity, vertex phi,
  negative boundary velocity and accepted residual exactly after four frames.
- `tests/uniform-mixed-frame-dawn.test.ts`: complete-frame mass, remapping and
  withholding velocity publication after failed pressure acceptance.

All Dawn runs must hold the repository GPU lease and run without a browser
simulation or another Dawn process.

## Final production capture and remaining work

The stable all-fine ABBA repeat measured these median hardware stage times
(deduplicated by trace timestamp):

| Full stage | Previous mixed (ms) | Updated mixed (ms) | Updated native (ms) |
| --- | ---: | ---: | ---: |
| Velocity extension | 2.458 | 1.770 | 1.770 |
| Pressure solve | 5.145 | 4.751 | 6.849 |
| Surface/volume transport | 8.225 | 8.290 | 4.358 |

Whole-frame medians were 21.921 ms mixed and 19.741 ms native: 90.06% native
throughput, so the 98% overall target is **not met**. Resident memory remains
1.01977 times native, within the unchanged 3% limit. Native also benefits from
the shared coarse-visit improvement; a ratio alone hides that improvement.
Surface/volume transport is the largest remaining all-fine stage gap.

An earlier all-mode run was unstable (its identical native arms measured
36.14 ms and 23.40 ms for the fine case), so its timings are not used to claim
a gain. All modes completed, but stable coarse-layout timing is reported
separately. Raw logs are `/tmp/fluid-minidam64-stage-final-abba.log` and
`/tmp/fluid-minidam64-stage-repeat-fine.log`; these temporary files are not a
portable record, hence the commands and results above.

Extension parity passed. The complete-frame test and the minidam64 native,
all-fine and coarse-air visit comparison passed; the latter matched all checked
fields and residuals bit for bit. Repository TypeScript checking still reports
unrelated Sparse/SVO errors; none were in these changed files.

## Live-UI discrepancy: unresolved

The user reported 8–10 ms extension and 14–15 ms multigrid in the default
minidam64 URL. Chrome UI inspection reproduced approximately 7.8 ms and
14.6 ms. The standalone Dawn figures above therefore do not establish live-UI
performance. Controlled Chrome-only captures also showed expensive fine-grid
pressure sweeps, but source edits occurred concurrently and the comparison did
not establish a browser/backend cause. Do not attribute the discrepancy to
Chrome or renderer interference on this evidence.

The six-neighbour unrolling and cached-theta experiments were removed; neither
is part of the retained implementation. The temporary browser harness was
removed. Further investigation must pin the source revision and compare exact
settings, frame identities, dispatch counts and measured boundaries before
changing kernels. The full live-UI performance target remains unverified.
