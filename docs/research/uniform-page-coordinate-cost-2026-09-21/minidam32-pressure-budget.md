# MiniDam32: evidence-sized pressure encoding

The production `minimal-power-dam-break-32` scene now meets the main-relative
performance target. This follows the single-page coordinate specialization;
it measures production scheduling defaults, without a forced one-cycle schedule
or a replacement miniature fixture.

## Measured result

Sequential ABBA on the same Metal GPU, main `5d4d31f2` versus working branch
`fddb0977` plus the recorded changes, using 124 advances per arm (four warmup,
120 measured). Both branches use balanced quality, 1/30 s advances, the same
32³ scene and pressure hierarchy, configured 3 Full-Cycles + 4 V-Cycles,
6/6 sweeps, and residual tolerance 10. Each advance is queue-fenced; rendering,
initial compilation, and final diagnostics polling are excluded. Hardware stage
instrumentation is enabled in both. These are simulation timings, not UI FPS.
The ABBA runs do **not** call `readStats` between advances: the dedicated
asynchronous demand readback must supply the pressure evidence.

| Run | Median advance | Pressure Full-Cycle stage median |
|---|---:|---:|
| Main A1 | 20.773 ms | 4.719 ms |
| Branch B1 | 16.377 ms | 3.473 ms |
| Branch B2 | 16.458 ms | 3.604 ms |
| Main A2 | 21.587 ms | 4.719 ms |
| Pooled main | 21.218 ms | |
| Pooled branch | 16.422 ms | |

The branch uses **22.6% less frame time**, or **129.2% of main throughput**,
above the original 90% floor. This is a local measurement with observed run
variation, not a performance guarantee for other GPUs or multi-page scenes.

A preceding 44-frame diagnostic pair measured main at 22.167 ms and the branch
before this pressure change at 33.437 ms. The updated branch measured 17.740 ms.
All three ended at the identical fine residual, 0.3748006522655487, with one
executed Full-Cycle, no rejected cycles, and no recovery sweeps.

## Cause and implementation

Paging had disabled the existing lagged host budget and forced all seven cycles
into the command buffer. GPU convergence gates prevent numerical work after
convergence, but do not remove the cost of encoding and submitting those passes.
On MiniDam32, the branch encoded 1,621 pressure passes to execute one cycle.
Main normally encodes two cycles (816 passes), reserving one cycle of headroom.

The geometric path now starts with a one-cycle prefix and zero headroom:

- The entire configured plan, pipelines and bind groups remain prebuilt.
- Without a sample, encode one cycle; after convergence, use observed demand.
- After unmet tolerance, increase to `max(2 * executed, executed + 2)`, capped
  by the configured schedule. The real GPU test observed **1 → 3 → 6 → 7**.
- Keep setup, acceptance checks, rejected-iterate restoration, recovery and final
  residual measurement in every encoded prefix.
- Explicit fixed mode and zero tolerance still encode the full configured plan.
- Preserve the nonpaged reference solver's existing startup/headroom defaults.

MiniDam32 now encodes **499 passes** per pressure solve, while executing the
same one Full-Cycle. All 124 frames of each measured branch arm reported
convergence. Tolerance, sweeps, pressure cells, stencils and storage are unchanged
by this pressure change. Runtime parameter application preserves zero default
headroom, and telemetry now truthfully reports host scheduling readback use.

Evidence is a twelve-byte asynchronous copy after submission. The frame does
not wait for it; while a map is outstanding, the next step uses the latest sample.
A sudden increase in demand can therefore leave a step above tolerance before a
later step expands its budget. The GPU acceptance/recovery gates still reject
nonfinite or worsening pressure iterates. This is a lagged scheduling policy,
not a guarantee of reaching tolerance within the same step that demand changes.

## Validation

- 11 CPU checks pass, including startup, growth, shrinkage, caps and preserved
  reference policy.
- The new Dawn test compares adaptive and fixed seven-cycle pressure schedules
  bit-for-bit for pressure, velocity, volume and vertex phi at frames 1, 30 and
  60; every intervening adaptive frame reports convergence. It reapplies runtime
  defaults each frame and never supplies demand through `readStats`.
- Tightened tolerance produces real asynchronous budget escalation without
  rebuilding the plan; explicit fixed and disabled-convergence modes retain
  all seven cycles.
- The new test and existing corrupt-pressure tests pass all 10 reported checks,
  including finite, NaN, infinity and post-acceptance corruption on direct and
  indirect dispatch paths.
- All three page compatibility checks pass: overlay publication, live garden-hose
  insertion, and 64-frame paged-versus-dense backing equivalence with matched
  fixed pressure schedules. The tests now expect the restored pressure readback;
  the overlay check uses the actual 32-cell domain page instead of the stale
  16-cell physical-tile expectation. Numerical equality assertions are unchanged.
- Type checking retains 15 existing errors outside changed files; none names
  the new tests or modified solver files.

The canonical Sparse CM12 gate **fails: 5/17 lanes pass in 373.2 s**.
See `minidam32-pressure-sparse-regression.json`. Failures include sparse expansion,
authored re-rung, clipped transfer, far-wall/collapse assertions, correctness and
live-edit timeouts, and mini64 performance (212.075 ms versus its unchanged
110 ms ceiling). These areas also failed in the preceding coordinate-only run;
this is not a clean broad regression gate or proof of attribution for every
failure. No lane or timing threshold was weakened. This suite targets the
separate `adaptive-volume` implementation; the measured A/B and focused tests
here target `uniform-geometric`.

## Reproduction and receipts

`minidam32-pressure-measurements.json` retains revision/source hashes, adapter,
raw frame and hardware-stage samples, per-frame encoded/executed/converged
counters, final diagnostics, and hierarchy/work censuses. Invocation metadata
identifies the asynchronous-only runs (older capture schema omitted that flag).

```sh
UNIFORM_BENCH_SCENE=minimal-power-dam-break-32 ASYNC_DEMAND=1 FRAMES=124 \
  ARMS=production WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  UNIFORM_BENCH_OUTPUT=/tmp/minidam32.json \
  node --import tsx tools/benchmark-uniform-long-dam-paging-dawn.ts
```

Run the identical probe sequentially in a detached main checkout with the same
node dependencies. Never overlap Dawn runs or browser simulation. Use
`tools/compare-uniform-page-performance.ts main.json branch.json` for the matched
fixture/hierarchy and 90% throughput gate.
