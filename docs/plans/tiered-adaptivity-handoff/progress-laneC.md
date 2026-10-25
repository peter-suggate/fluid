# Lane C progress

Written by the lane C agent; read by Peter and the coordinator. The format and rules are in `../tiered-adaptivity-handoff-2026-10-25.md` §6. Newest entry first.

## Status
- **Tip:** 49c20310 (C4 `efda8410`; C3 candidate `77c275dc`; gate tooling `49c20310`)
- **Doing now:** C4 and gate tooling ready for review; current-default paired timing and C3 evidence queued
- **Next:** C4 GPU smoke, current-default paired baseline and particle-path attribution (C3)
- **Waiting on:** CPU suites and active GPU checks before uncontended timing; A1 for C1, B1/B2 for C2

## Log

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
