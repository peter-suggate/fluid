# Lane C progress

Written by the lane C agent; read by Peter and the coordinator. The format and rules are in `../tiered-adaptivity-handoff-2026-10-25.md` §6. Newest entry first.

## Status
- **Tip:** e73cbfa0 (C3/C4, construction locking and measurement tooling ready)
- **Doing now:** Candidate43 letters cost complete; seep is last queued pair (PID 83841)
- **Next:** Complete C1 basic scenes; hold C5 and wide ABBA while core A/B proofs progress
- **Waiting on:** Exclusive timing slots; A acceptance repair; B1/B2 storage/work separation; coordinator review of ready commits

## Log

### 2026-10-25 · COORDINATION · next exact C3 deposit candidate
- C will edit only `pbAdd` in `uniform-particle-brick.wgsl.ts`: keep the existing round/cast, then independently skip momentum or weight atomicAdd when its integer operand is exactly zero. B confirms no overlapping deposit/scatter edits. No allocation, dispatch, quantization, classification or nonzero sum changes.
- Root approved this narrow candidate with focused arithmetic/overflow GPU cases and the registered gate/performance evidence, reusing existing full-suite evidence. Existing older letters profile identifies transfer as material (~1.80 ms/step), but gives no current isolated estimate for this guard. No saving is claimed before measurement.
- Root released C5 after the current seep pair. Coordinator 11166 waits for a valid C1 completion, then runs the existing guarded length ladder sequentially. Wide43 ABBA remains held pending A discharge attribution or a replacement candidate.

### 2026-10-25 · MILESTONE · candidate43 letters cost
- Letters completed 100 steps/arm, measuring frames 40–100: median paired candidate added cost **0.052292 ms**; mean wall 39.8347 → 39.9020 ms. Effectively flat in one run. Separate medians (40.081 → 38.666 ms) are not the paired comparison and should not be used to claim a saving.
- No GPU errors; all final field hashes differ. Durable `final-43b1aa3c/a1-nbflip-figure-8-letters.{json,log}` and before/after process snapshots preserve the evidence. `basic-cost-summary.json` consolidates completed scene results, source hashes, clocks and the failed-acceptance caveat.
- Seep is the final queued C comparison. Driver stops after it. C5/wide remain held, and no new full suite is planned. A’s failed seep gate/conservation acceptance remains unresolved.

### 2026-10-25 · PROGRESS · remaining C1 resumed; acceptance remains failed
- B’s direct RHS fixture passed 1/1 and its requested manual hold cleared; its strict batch also finished. A requested resume under the coordinator’s existing priority decision. Guarded continuation PID 83841 runs only candidate43 letters and seep, sequentially. No active full suite was present. C5 and wide ABBA remain held.
- A reports candidate43 A/B splash and letters gates pass, but seep fails the bandTiles window at frames 31–40 (ratio 1.080976). Candidate 20s and fast-jet runs also have conservation failures. These costs are evidence for `43b1aa3c`; they cannot make A1 READY or override those failures.
- A’s separate rest100 physical check passed: maximum sampled speed 1.9559011e-6 m/s, zero solver/raw drift, no fatal/capped traces. This supports the unchanged rest surface in C’s matched phi/volume hashes; pressure/velocity differences remain disclosed. A’s evidence is in its durable `artifacts/tiered-lane-a-2026-10-25/rest-footprint*` files.

### 2026-10-25 · MILESTONE · final43 splash cost; C queue held
- Final splash pair reached 100 steps/arm, measured frames 40–100: median paired candidate added cost **0.040458 ms**; mean wall 16.3892 → 16.4091 ms. Effectively flat in this single run, with no GPU errors. All final field hashes differ; the A repeat-spread gates, not these hashes, determine correctness. Durable evidence: `final-43b1aa3c/a1-splash.{json,log}` plus process snapshots.
- To free B’s implementation proof queue, the remaining C continuation was canceled. Splash had just begun when cancellation was requested, so its GPU child ran to completion untouched. No C GPU/CPU job or pending marker remains. Letters/seep continuation now requires explicit coordinator release after B’s pressure400 audit and focused shadow/RHS batch. C5 and wide ABBA remain held as well.
- B reports the pressure400 audit finished without errors; focused proof batch is queued. C’s completed rest/splash results and all previous ready evidence remain preserved. No new full suite is planned.

### 2026-10-25 · MILESTONE · final43 rest cost, remaining C1 queued
- `43b1aa3c` against exact `c8b25c39`, controller off, 100 steps/arm and frames 40–100 measured: median paired saving **0.375126 ms**; mean wall 10.9975 → 10.6648 ms. One paired run, not a guaranteed or isolated-kernel gain. No GPU errors. Final phi/volume hashes match; velocity/pressure differ, and live histories first differ at frame 5. A owns the physical rest speed/drift check.
- Evidence is durable `final-43b1aa3c/a1-rest.{json,log}` with before/after process snapshots in the C artifact directory. Candidate frame100 tiers: 64 COLD; baseline: 32 COLD + 32 WARM. No CPU suite overlapped the measured window according to the timing-entry/completion checks.
- Coordinator priority now holds C5 length pairs and wide ABBA until core A/B proofs progress. The original scheduler was replaced without touching its already queued splash runner 11858. Continuation 17866 validates that result, then runs letters/seep only. The old combined driver is disabled; a separate C5 script remains held. No extra full unit suite is planned.

### 2026-10-25 · PROGRESS · final candidate cost queue
- A confirmed `43b1aa3c`: focused GPU 2/2 pass; full unit 1222 pass / 134 skip / 0 fail. Shared immutable snapshots exist. C launched durable final driver PID 788 with one sequential timing waiter at a time; first rest pair is waiting for the exclusive GPU slot.
- Final wide ABBA CPU-only preflight passes for the same exact snapshots: 834 blocks per arm, 1/60 s, controller off, only coldSurface warm→still differs. Plan is `final-43b1aa3c/a1-wide-abba/plan.json` inside the durable artifact directory. Actual wide ABBA remains unlaunched.
- Per user steering, no new full unit suites for incremental work. C reuses its completed full-suite evidence; subsequent tooling edits will use focused fixtures/types. This cost driver launches no CPU test suites.

### 2026-10-25 · PROGRESS · durable evidence and final cost preparation
- All completed C logs, fixtures, plans and C3 gate results are preserved in ignored durable `artifacts/lane-c-2026-10-25/` in `/Users/petersuggate/.codex/worktrees/tiers-lane-c/fluid` (1.8 MiB). Its README maps the previous temporary paths and records snapshot recovery commits. New final cost outputs will be written there directly.
- Prepared `final-a1-costs.sh` now targets candidate `43b1aa3c` and exact baseline `c8b25c39` shared snapshots. Candidate focused GPU tests passed 2/2 with no skips; A’s full suite and shared snapshot are still pending. No C timing or GPU job has launched.
- The driver covers C1 rest/splash/letters/seep (100 steps, frames 40–100) and C5 rest/cold adjacent 128→256→512 (180 steps, frames 120–180), preserving existing CPU guards and whole-process timing locks. A owns the separate rest physical-stat check. Wide ABBA remains separately coordinated.

### 2026-10-25 · READY TO MERGE · C1/C5 measurement tooling d905ea99..e73cbfa0
- Tooling commits `d905ea99`, `62c72a70`, `9025d008` and `e73cbfa0` are ready together. They add explicit wide geometry and same-source length comparisons, CPU-only plans, and memory-safe serialized wide ABBA measurements. They do not change solver physics.
- Both small GPU harness checks PASS under the check runner: same-source/same-scene 3 steps with matching final hashes; same-source cold 128→256 3 steps with separate geometry, both live-work histories and a computed slope. Different-geometry field equality is correctly reported as null. Logs/artifacts `/private/tmp/lane-c-20261025/harness-{same,length}.{log,json}`; wrapper reports `HARNESS GPU SMOKES PASS`. These short check-mode wall times are **not performance evidence**.
- Full unit suite: 1221 pass / 131 skip / 0 fail. Typecheck: only the two documented baseline import errors. Eleven plan fixtures and six ABBA orchestration/failure fixtures pass. Final CPU-wait fix anchors the actual Node executable, excludes stopped suites and ignores queued lock wrappers so an exclusive run cannot wait on its own queued followers. Evidence remains in `unit-paired-harness.log`, `types-abba.log`, `paired-plan-fixture.log`, `abba-fixture.log` in that artifact directory.
- No real wide ABBA run or C5 acceptance run has happened. Candidate `82326cc7` includes the reviewed direct velocity-footprint repair; final cost work waits for A’s focused GPU/full unit checks and an immutable agreed snapshot. Prior `787d5cdd` cost results remain superseded checkpoint evidence. No lane C GPU job remains queued.

### 2026-10-25 · PROGRESS · serialized wide timing and harness checks
- Tooling `62c72a70` adds `benchmark-seep-abba.ts`: one exclusive runner lock, four fresh processes in A/B/B/A order, at most one wide solver resident. Same-scene preflight permits only the A1 coldSurface difference; both arms use controller off and 1/60 s. Source hashes are checked before/after each child. Frames 40–100 are measured, with final diagnostic-layout frame 101 excluded.
- Review follow-up `9025d008` waits for active CPU unit suites after taking the lock and before each block, records before/after process snapshots, and rejects a block with a unit suite still active afterward. The command notes document the boundary-check limitation and require CPU-heavy work to remain paused.
- Fake runner/probe/process fixtures PASS: one lock, ABBA order, fresh processes, paths with spaces, equal windows, excluded diagnostic frame, active/stopped CPU handling, overlap rejection, child nonzero exit, probe exit-zero failure, GPU errors and truncated horizons. Logs `/private/tmp/lane-c-20261025/abba-fixture.log`; `check-abba.py` reproduces them. Real-source CPU preflight confirms 834 blocks per arm and warm/still defaults; no wide GPU run was launched.
- Required full unit rerun PASS: 1221 passed / 131 skipped / 0 failed, 307.95 s (`unit-paired-harness.log`). Latest typecheck only the two documented baseline errors (`types-abba.log`). Same-source cold GPU harness check PASS for 3 steps with identical final hashes and zero history differences (`harness-same.json`). Check-mode wall times are not performance evidence.
- One tiny 128→256 cold check remains queued under the runner (wrapper PID 70765), behind other shared GPU work; no C timing jobs are queued. Final A1 costs remain deferred until A supplies the repaired candidate and regression evidence. The new measurement tooling is not marked ready until the remaining GPU check clears.

### 2026-10-25 · PROGRESS · C1/C5 measurement preparation d905ea99
- Separate tooling checkpoint adds explicit `--seep-width`, `--seep-depth`, `--seep-cell` to the paired benchmark, using each snapshot's production factory. `--candidate-length` compares different lengths only with the same source and method values; normal scene equality remains strict. Both arms' live-work rows and a paired cost slope per doubling are saved.
- `--plan` resolves both scenes, clocks and parameters before any GPU initialization. Eleven CPU-only geometry/argument checks pass: base/wide seep, rest/cold 128→256→512, and invalid identity/method/geometry/scale inputs. Wide plans confirm 834 blocks in each arm. Typecheck has only the two baseline import errors. Logs: `/private/tmp/lane-c-20261025/paired-plan-fixture.log`, `types-paired-harness.log`; plans in `prepared/` there.
- Commands and acceptance interpretation are in the branch file `laneC-cost-commands.md`. C5 uses current final-A1 defaults, 180 frames with frames 120–180 measured, adjacent slopes ≤0.2 ms/doubling, with actual tiers/live work reported. No forced cold tier or altered physics.
- The old-layout wide pair would hold roughly 26 GiB, so it must **not** run on this 32 GiB host. Await integrated memory measurements, or implement separately labeled serialized ABBA blocks. No wide pair or new timing was launched.
- This tooling commit is **not yet ready**: a full unit rerun is deferred behind active A/B suites, and small same-scene/different-length GPU harness checks remain pending. Earlier C3/C4/lock-tooling ready evidence is unchanged. Final A1 SHA and repaired regression are still required before cost runs.

### 2026-10-25 · READY TO MERGE · C3 77c275dc
- Exact merge-scan short circuit: stop reading later records when accumulated integer mass reaches the cap of 8. Every live record has positive mass, so no later record can be accepted; accepted arithmetic, sample order and retirement diagnostics stay unchanged. No new dispatch, allocation or physics parameter.
- Required repeat-spread bundle against `c8b25c39` PASS: splash 70 ×2/arm (worst ratio 0.4124), seep 60 ×3 (0.1382), letters 60 ×3 (0.2159). No fatal. Artifacts: `/private/tmp/fluid-gates/gates/c3-merge-cap-{splash,seep,letters}.{json,log}`.
- Paired alternating letters run completed 100 steps/arm, measured frames 40–100 with identical default parameters (`coldSurface:warm`, `volumeControlSeconds:0.1`, dt 1/60). Mean wall time 37.549 → 36.704 ms; median paired saving **0.539 ms**, with 49/61 measured pairs faster. Separate arm medians: 36.829 → 36.459 ms. Artifacts: `/private/tmp/lane-c-20261025/merge-short-circuit.{json,log}`.
- Performance scope: one clean paired run, not an isolated kernel attribution. Atomic-scatter histories first differ at frame 36 and final field hashes differ, as expected for this registered-scene comparison; the repeat-spread gate above is the correctness evidence. Do not read the 0.539 ms as a guaranteed speedup independent of the evolved workload.
- Checks: full unit suite 1221 pass / 131 skip / 0 fail; final typecheck only the two documented baseline `app/app/*-lab` import errors. All A/C timing is finished; lane B was notified to resume its deliberately paused suite. No lane C GPU jobs remain.
- C4 `efda8410` and construction-lock tooling `49c20310` remain independently ready. C1 final-default/wide cost, C2 16h scale ladder and C5 length ladder remain unfinished dependencies; this entry completes the first C3 optimization only.

### 2026-10-25 · PROGRESS · A1 candidate cost evidence (787d5cdd)
- Paired alternating runs against `c8b25c39`, 80 steps each, measured steps 40–80. Both arms use `volumeControlSeconds:0`; baseline `coldSurface:warm`, candidate `coldSurface:still`. Every scene completed without a fatal. This is evidence for candidate `787d5cdd`, not approval of final A1: lane A is still checking ballistic-particle reach.

| Scene | Baseline median ms | Candidate median ms | Median paired candidate cost change |
|---|---:|---:|---:|
| Splash | 10.701 | 10.591 | +0.047 ms |
| Letters | 32.477 | 32.488 | +0.364 ms |
| Seep | 28.008 | 28.496 | +0.715 ms |

- The paired differences, not subtraction of separate medians, are the comparison. These basic scenes show small added cost rather than savings. Lane A owns rest and its command attribution. Wide seep remains pending the final A1 candidate.
- Artifacts: `/private/tmp/lane-c-20261025/a1-cost-{splash,nbflip-figure-8-letters,reservoir-seep}.json` and matching logs. Pre-run process snapshots are `timing-processes-*.log` in the same directory; no active unit suite ran during these samples. Lane B's suite remains explicitly stopped until the final C3 pair finishes.

### 2026-10-25 · MILESTONE · C3 physics gate PASS · tip 49c20310
- Solver candidate `77c275dc` against baseline `c8b25c39`: full `c3-merge-cap` bundle PASS. Splash 70 frames ×2 repeats per arm: worst ratio 0.4124 (bandTiles). Seep 60 ×3: 0.1382 (fineTiles). Letters 60 ×3: 0.2159 (occupancy). No fatal; all scenes reached their full lengths.
- Gate artifacts: `/private/tmp/fluid-gates/gates/c3-merge-cap-{splash,seep,letters}.{log,json}`; bundle summary `/private/tmp/lane-c-20261025/c3-bundle.log`, exit 0. Whole-process construction locking was enabled, so no registered-scene GPU allocation happened before its exclusive slot.
- Still not ready for the C3 optimization: paired cost evidence remains pending. A1 cost pairs now use immutable candidate `787d5cdd` and baseline `c8b25c39`, both with volume controller off; the earlier `352645f6` candidate was superseded before any measurements.
- Timing hygiene: CPU suites finished except lane B's explicitly paused process group 6825; all its processes were verified stopped. The cost launcher excludes stopped suites, checks again after acquiring the timing lock and saves a process snapshot immediately before each measured run. Lane B resumes after A/C costs finish.

### 2026-10-25 · READY TO MERGE · C4 efda8410; gate tooling 49c20310
- C4 `efda8410` fixes the letters benchmark clock without changing solver physics. Default and explicit 0.0166667 s smokes both pass 3 steps with no fatal, ending at 0.05 s and 0.0500001 s respectively. Logs/artifacts: `/private/tmp/lane-c-20261025/clock-{default,explicit}.log` and `nbflip-letters-clock-{default,explicit}.json`. A conflicting explicit clock/method mode rejects before GPU setup (`invalid-clock.log`).
- Separate tooling commit `49c20310` adds opt-in `FLUID_GATE_LOCK_CONSTRUCTION=1`: acquire the runner slot before constructing solver instances. Default frame-only locking is preserved; whole-process mode omits the inner lock. This avoids pre-lock allocation of six seep instances.
- Tooling validation: `bash -n` and a fake runner/node fixture pass default and whole-process forwarding (including paths/arguments containing spaces), modes gate/timing/timing, absence of nested locks, node failure propagation (bundle exit 1) and invalid option rejection (exit 2). `/private/tmp/lane-c-20261025/gate-lock-fixture.log`; fixture script `check-gate-lock.py` in that directory.
- Checks: full unit suite 1221 passed, 131 skipped, 0 failed; typecheck has only the two documented app/app errors. These two tooling commits need no physics gate because solver behavior is unchanged.
- **Do not merge the whole branch yet:** intervening C3 solver commit `77c275dc` still needs paired performance evidence and the full bundle. The two ready commits can be applied individually.

### 2026-10-25 · PROGRESS · tip 77c275dc
- C4 repair committed separately as `efda8410`. Default letters smoke passes: 3 encoded steps, completed time 0.05 s at 1/60 s, no fatal. `/private/tmp/lane-c-20261025/nbflip-letters-clock-default.json`. Explicit 0.0166667 s smoke is queued; incompatible `--dt`/`--set=timeStep` combinations reject before GPU construction. Check-mode wall times are not benchmark evidence.
- C3 exact candidate `77c275dc`: stop both crowded-cell scans when accumulated integer mass reaches 8. No later positive-mass sample can fit, so accepted merges, arithmetic order and retirement counters stay unchanged. GPU bundle and paired cost remain pending; this is not ready to merge.
- Checks: `npm run test:unit` PASS (1352 tests, 1221 passed, 131 skipped, 0 failed), `/private/tmp/lane-c-20261025/unit.log`; final typecheck has only the two documented `app/app/*-lab` TS2307 errors, `/private/tmp/lane-c-20261025/types-c3.log`. Targeted seeding tests 7/7 pass.
- Paired current-default A/A rest/splash/letters/seep and C3 letters A/B are scripted after the old verification gate, clock smoke and CPU suites finish. Baseline is the coordinator's immutable `tiers-baseline/fluid` worktree; logs will be in `/private/tmp/lane-c-20261025/`.

### 2026-10-25 · COORDINATION · tip c8b25c39
- C3 will edit `uniform-narrow-band-flip.ts` functions `nbMergeRun` and `nbMergeAbsorbed` only. Stop scanning once accumulated mass reaches 8: every remaining sample carries positive integer mass, so none can fit. The accumulation order and accepted merges stay unchanged.
- Lane B confirmed its bin-map constructor/constants and fatal-report edits do not overlap. Lane A is editing heat and region dynamics.
- Validation: benchmark profile/pair and full physics-neutral gate bundle required before proposing this optimization for merge.

### 2026-10-25 · PROGRESS · tip c8b25c39
- Worktree: `/Users/petersuggate/.codex/worktrees/tiers-lane-c/fluid`, branch `codex/tiers-lane-c`.
- C4 diagnosis: benchmark advances at the paper output interval (1/24 s), while the default solver admits only 1/60 s steps. The requested clock and completed clock therefore disagree. The repair resolves the method step and makes an explicit `--dt` select the scene step mode.
- Checks: type and unit checks running; GPU smoke waits behind the existing verification gate. No solver files changed.
- Next: validate default, explicit 60 Hz and paper-interval benchmark runs, then profile letters and current-default paired timing. C1 after A1 and C2 after B1/B2 remain dependencies.

### 2026-10-25 · HANDOFF
- The lane was handed over from the Claude lane C agent. The previous owner's note is `laneC-*.md` in this directory.
