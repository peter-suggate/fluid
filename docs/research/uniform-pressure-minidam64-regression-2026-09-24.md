# Minidam64 pressure regression

The V-first adaptive implementation regresses cycle cost on
`minimal-power-dam-break-64`. This was missed by testing the slab and Figure 9.

Matched 150-frame Dawn/Metal runs (5 seconds simulated, first four frames
excluded), default authored scene and residual target 10 s^-1. Browser unloaded,
GPU lease held, no concurrent GPU runs. HEAD was exported to a separate tree;
the same profiling harness was used. Times are stage seams, including host
continuation gaps, not pure shader timings.

| Arm | Cycle median | Pressure incl. setup/finish | Whole simulation | Fine residual median / max |
| --- | ---: | ---: | ---: | ---: |
| head | 6.29 ms | 8.52 ms | 36.34 ms | 5.62 / 17.66 |
| legacy | 8.06 ms | 10.55 ms | 36.44 ms | 7.49 / 19.01 |
| adaptive | 10.75 ms | 11.86 ms | 33.62 ms | 1.11 / 9.42 |
| vonly | 6.82 ms | 8.00 ms | 31.94 ms | 3.81 / 9.28 |

`legacy` disables the new controller but retains all other dirty working-tree
changes. `vonly` is a diagnostic parameter override (Full-Cycles=0, V-Cycles=7),
not a production-default change.

The current plan tries one V-cycle then immediately escalates to a Full-Cycle.
142 of 146 measured frames take this path. A six-level Full-Cycle contains six
coarse invocations and nested smoothing visits; it drives the median residual
to 1.1, unnecessarily far below the requested 10. The first cheap V-cycle is
therefore an additional cost on almost every frame in this scene.

Each cycle also ends in a GPU-to-CPU mapAsync receipt and a new submission.
These waits are included in the cycle stage seam intervals. Their independent
contribution was not isolated in this experiment. The old controller could
accept a finite improving iterate above 10 and adjust the next frame's budget;
the new one insists on meeting 10 in the current frame. This is a stricter
acceptance policy even though the numerical tolerance setting did not change.

Repeating cheap V-cycles instead of immediately running a Full-Cycle largely
removes the regression while every measured completed residual remains <=10.
The next correction should use V-cycles while progress is useful and reserve
Full-Cycles / tighter inner accuracy for stalls. A subsequent optimization can
keep convergence decisions on the GPU to avoid per-cycle host waits. Looser
outer tolerances are a separate quality/performance choice, not required to
show the scheduling mistake here.

No production solver code was changed during this diagnosis. The whole advance
is not slower than HEAD in these headless runs; the regression is specifically
in the pressure-cycle stage the user identified. Other working-tree changes
also matter: legacy-on-current differs from clean HEAD.

Per-frame evidence: [JSON](uniform-pressure-minidam64-regression-2026-09-24.json).

## Implemented correction

Adaptive plans now put all configured V-cycles before the Full-Cycles. After
an unconverged correction, retain loose inner accuracy and the next V-cycle if
the residual fell by at least half. Otherwise skip unused V-cycles and tighten
inner accuracy by one step (1 → 0.1 → strict). Compare the first cycle with its
actual initial residual, not infinity. Exhausting the V-cycle allowance also
makes the Full-Cycle budget available. Physical-frame acceptance, recovery and
failure handling remain unchanged.

Final default-settings benchmark: minidam64 cycle median **6.88 ms** (formerly
10.75 ms), slab **3.64 ms**, Figure 9 **5.93 ms**. Minidam64 used no Full-Cycles
in 150 frames and never exceeded residual **9.28** against target 10. These
runs preserve the original configured 4 V / 3 Full caps; the diagnostic
V-only override is no longer necessary.

[Final measurements and source hashes](uniform-pressure-v-progress-2026-09-24.json).
The progress policy has unit coverage for useful progress, stalled escalation,
accuracy tightening, and exhausted/absent cycle types. The Dawn acceptance
suite now includes 150 mini64 frames and a stalled-progress injection that
checks jumping over unused V-cycles into a Full-Cycle with a real final fine
residual measurement. No renderer, live-edit, or sparse-backend changes were
made in this follow-up, so the previously recorded broad gate failures are
unchanged validation limitations.

Validation: all 7 Dawn test reports pass, including the three scenes (330
physical frames), stalled-progress skip, recovery and exhausted-budget cases.
All 8 targeted CPU tests pass. Type checking retains the previously recorded
unrelated sparse test/tool errors, with none in this change’s files.
