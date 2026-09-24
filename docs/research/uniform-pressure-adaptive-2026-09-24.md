# Current-frame adaptive Uniform Geometric pressure

The 3D Uniform Geometric adapter now enables a staged pressure solve. The
reference solver and the 2D parity path retain their existing execution mode.
`adaptivePressure: false` selects the previous controller for QA comparisons.

## Algorithm

- Build the pressure system and try a V-cycle first. Continue with configured
  V-cycles while each correction reduces the residual by at least half. Skip
  remaining V-cycles for Full-Cycles after slower progress, or use Full-Cycles
  when the V-cycle budget is exhausted. If no V-cycles are configured, start
  with a Full-Cycle. The total configured caps are unchanged.
- Read a compact GPU diagnostic receipt before projection. A finite,
  non-increasing iterate is not enough: its accepted fine projected residual
  must also meet the current outer tolerance. Continue the same physical frame
  if it does not. The previous frame no longer decides the adaptive prefix.
- Use loose, RHS-dependent coarse accuracy. With outer tolerance T, incoming
  coarse RHS infinity norm B in divergence units, and accuracy scale s, the
  inner threshold is max(1e-4, min(T*s, max(0.1*T*s, 0.1*B*s))). Start
  with s=1 and retain it while progress is good. A stalled correction tightens
  to s=0.1; another stall uses s=0 (strict). Thus the default outer T=10 normally permits coarse thresholds
  between 1 and 10 s^-1. No global outer tolerance was tightened.
- Encode the original bounded recovery only after a current-frame rejection.
  The normal finish has three dispatches; the idle 160-dispatch recovery tail
  is absent. Retain the accepted-pressure restore and final residual.
- If the configured cycle/recovery limits cannot satisfy tolerance, latch an
  error and withhold projection and presentation. A subsequent advance cannot
  quietly step past the failure. Zero tolerance and explicit fixed mode retain
  full-plan diagnostic execution.

The coarse solver retains compensated pressure arithmetic, projected bounds,
its 4096-sweep cap and diagnostics. Simultaneous/Jacobi updates retain reflection
symmetry. No spatial approximation, new grid resolution, warm start, or direct
coarse solver was introduced in this change. Those are separate follow-on
optimizations; the measured unnecessary cycles and inner accuracy were addressed
first, with a current-frame correctness check.

## Frame and presentation ownership

`framePending` covers all pressure continuations and the final submission.
`awaitFrameCompletion()` waits for that complete frame and reports a latched
failure. Runtime parameter writes cannot mutate an in-flight solve. The renderer
recognizes deferred publication and shows each completed frame before starting
the next; it never samples intermediate pressure/transport fields. Existing
solvers whose publication is already submitted on return retain their cadence.

Stage instrumentation spans all submissions. Queue-wall fallback includes the
whole staged frame, and CPU trace records receipt waits separately. GPU stage
seam intervals can include dependency and host continuation gaps; they are not
isolated shader times. The pipeline panel identifies V-first adaptive mode,
actual dispatch counts, on-demand recovery, and projected Jacobi smoothing.

## Validation notes

The GPU acceptance test runs 90 frames each of the authored half slab and
Figure 9 plus 150 frames of minidam64, checks every completed residual against the default 10 s^-1 target,
asserts same-frame continuation on impact, and verifies that a deliberately
impossible target withholds publication. A fault-injection case exercises
same-frame recovery. CPU presentation tests cover the deferred-publication
cadence and preserve the existing immediately-published solver behavior.

The existing Figure 9 conserved-mass gate failed with both controllers in the
current working tree: the old controller failed at frame 10 and the first
adaptive run at frame 13. Its 1e-4 threshold was not changed. The remainder of
that pressure-suite run passed 27 checks, including coarse manufactured systems,
pressure layouts and corruption/recovery. This workspace has concurrent unrelated
fluid changes; the baseline failure is not treated as a newly passing gate.

The required Sparse CM12 gate was run unchanged. It passed 5 of 17 lanes;
8 lanes timed out, mini64 exceeded its performance ceiling (213.91 ms versus
110 ms), both far-wall cases halted in `addWholeFrameUncoveredDonorFallbacks`,
and outside-tank collapse halted in `certifyGeometricTopologyFaces`. These
headless lanes exercise the separate sparse backend, not the changed uniform
pressure or renderer path. A clean baseline for those failures was not run.
No lane, limit, or timing ceiling was relaxed.

The final targeted Dawn run passes all 23 checks: both 90-frame scenes,
rejected-correction recovery, exhausted-budget publication blocking, manufactured
coarse systems, live wall/sphere edits, and moving solid coupling.

All 11 targeted CPU pressure-plan/budget/page-domain/presentation tests pass.
Repository type checking still reports errors in existing sparse tests and
probe tools; it reports none in the files changed for this pressure work.

## Matched final measurements

Serial Dawn/Metal runs, 30 physical frames, default scene parameters, first four
frames excluded. These are instrumented simulation times without rendering.
GPU cycle stages include submission/dependency gaps. Per-frame receipts and
source hashes are in [the measured data](uniform-pressure-adaptive-2026-09-24.json).

| Scene | Controller | Cycles median | Pressure including setup/finish | Whole simulation median |
| --- | --- | ---: | ---: | ---: |
| coarse-first-pool-impact-half-slab | legacy | 16.06 ms | 18.78 ms | 31.76 ms |
| coarse-first-pool-impact-half-slab | adaptive | 3.44 ms | 4.85 ms | 17.22 ms |
| cm12-figure-9 | legacy | 11.86 ms | 16.94 ms | 59.09 ms |
| cm12-figure-9 | adaptive | 5.96 ms | 8.45 ms | 47.75 ms |

The slab uses one V-cycle, one coarse sweep, and 143 total pressure dispatches
instead of 464. Figure 9 uses one or two cycles (167 or 523 dispatches), adding
a correction when the impact needs it. The adaptive fine residual maxima are
2.116 and 9.918 s^-1 respectively. The old Figure 9 controller reaches 12.386
s^-1 in this window despite the requested 10 s^-1 tolerance.

Browser smoke: restored the user’s original half-slab URL and overrides, played
through 4.9 s, and paused successfully. The live panel showed V-first adaptive,
1 of 7 cycles, 143 dispatches, and a 3-dispatch finish; presentation and the
simulation clock continued advancing.

## Reproduction

```
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx \
  --test tests/uniform-pressure-adaptive-dawn.test.ts

node --import tsx tools/profile-uniform-geometric-dawn.ts \
  --scene=coarse-first-pool-impact-half-slab --frames=60 \
  --pressure-mode=adaptive --out=/tmp/slab-adaptive.json
```

Use `--pressure-mode=legacy` for the previous controller with the same scene,
parameters, instrumentation and working tree. Run all GPU work serially under
the repository lease, with browser simulation/rendering unloaded.
