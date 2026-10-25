# Lane A progress

Written by the lane A agent; read by Peter and the coordinator. The format and rules are in `../tiered-adaptivity-handoff-2026-10-25.md` §6. Newest entry first.

## Status
- **Tip:** 5d974c9c (branch codex/tiers-lane-a; solver snapshot 787d5cdd)
- **Doing now:** Verifying candidate default still with current-step certification and post-failure halo propagation.
- **Next:** A1 registered A/B and new-arm A/A gates plus rest/cost evidence; then A2 candidate 1200-step seep and accumulated-volume diagnosis.
- **Waiting on:** Queued final rest/letters diagnostics and GPU gates; clean paired timing waits for all CPU suites. Main remains warm.

## Log

### 2026-10-25 16:28 · PROGRESS · tip 5d974c9c
- Candidate code now defaults to still. The current-step certificate runs after velocity extension/remap and before transport; failures are classified first, then a separate halo pass reads immutable post-failure tiers. This closes the previous one-step gap when a newly failing region was still COLD in the snapshot. The coordinator reviewed that ordering. Diagnostic record readers use exported strides, including the pool audit.
- First current-certificate trial fixes the exact letters event: fine-tile coordinate sets at frames 30/31/32 equal warm (1379/964/1034); frame-32 particles 9775 in both. At frame 36 particles are 40680 vs warm 40694. The 60-step trial completes without fatal or reach breach; final travel is 4.47555 h. Final post-failure-halo trial is queued, so this is diagnostic evidence, not a gate claim.
- Full unit suite: 1221 pass, 0 fail, 132 skip. Focused default tests pass; six focused GPU tests pass. Type check has only the two known app/app symlink resolution errors. Logs: `/private/tmp/fluid-lane-a/{unit,default-unit,types}.log`; numerical/event evidence `letters-current.log` and `current-tile-diff.json`.
- The coordinator completed the current-main warm 1200-step / 20 s seep baseline with volumeControlSeconds=0. Left level fell 14.6436 mm vs Torricelli 14.8605 mm at measured Cd=0.81327 (−1.46%). Left lost 5.62314 L, right gained 2.03384 L, wall gained 0.32166 L; the summed phi zones lose 3.26764 L. Solver drift −0.90915%, raw −0.97358%; no fatal/capped traces. One sampled right-fill interval is negative, though overall fill is positive. This does not establish A2 because the accumulated deficit is unexplained.
- Baseline deletion/control counters do not explain the deficit: cumulative dust, orphan dust, outflow and control rate are zero; volumeShift is only an unapplied proposal; all four sampled phi forms agree. Retirement counters alone are not proof of physical-volume loss. Sampled maximum speed 3.8078 m/s, throat 2.8383–2.8648 m/s; the dedicated fast-jet fixture remains unrun. Root is isolating stage loss with a 660-step phi-stage probe.
- Baseline evidence: `/Users/petersuggate/.codex/worktrees/tiers-baseline/fluid/artifacts/tiered-baseline-2026-10-25/seep5-main-1200{.log,.json,-analysis.json}`. Final candidate diagnostics: `/private/tmp/fluid-lane-a/{rest-current,letters-final}.log`; C owns splash/letters/seep paired costs against immutable 787d5cdd, A owns rest. A premature rest timing was stopped before samples and discarded; corrected coordinator waits for CPU suites. No READY TO MERGE claim.

### 2026-10-25 16:20 · PROGRESS · tip c8b25c39
- Original letters divergence reproduced: frame 30 has identical 1379 h tiles; at frame 31 warm has 964 and still 1004. Exactly 40 extra tiles lie in eight regions x=6..9, y=2, z=3..4. They were already WARM in both arms, so this is not simply those regions failing to wake.
- Pressure-local scale warms roughly 125 additional regions but leaves the same 40 extra frame-31 tiles. Frame-32 particles: warm 9775, original still 20801, pressure-only still 22574. Tightening the head alone does not resolve the divergence.
- Implemented an opt-in current-step certificate after extension/remap and before transport, a deterministic halo snapshot, and a one-way thaw preserving the stored velocity term. Removed the residual floor from still-mode discarded correction accounting. Disabled the two-frame-lag host skip for still because it can suppress a required wake; region-level COLD freezing remains.
- Validation: six focused GPU tests pass, including local-pressure round-off, accumulation, nonfinite rejection, and existing freeze contracts. The first test attempt failed because a test shader used a nonfinite compile-time constant; runtime-input form now passes. Full unit and type checks are running; two known app/app symlink errors are excluded.
- Evidence: `/private/tmp/fluid-lane-a/letters-{warm,still,pressure}.log`, `original-tile-diff.json`, `tile-diff.json`; new reader `tools/gpu/probe-readers/fine-tile-diff.py`. Current 60-step trial: `letters-current.log`. No default flip, no gate or merge claim yet.

### 2026-10-25 · VERIFIED · main 6992ac44 (coordinator)
- Lane A's last merge (59ce5a32 → 6992ac44: still-surface wake rule, head-scaled bound, reach bound; opt-in) leaves the default path within repeat noise.
- `tools/gpu/gate-bundle.sh verify-6992 9785c8fe 6992ac44`: PASS on all three scenes.
  - splash: worst ratio 0.38 (particles)
  - seep: worst 0.26 (fineTiles)
  - letters: worst 0.94 (particles, close to the limit, 0 windows over); every other letters metric is ≤ 0.23
- Logs: /private/tmp/fluid-gates/gates/verify-6992-*.log

### 2026-10-25 16:15 · COORDINATION · tip c8b25c39
- Additional files/functions: `uniform-mixed-frame.ts` advance immediately after `plan.encodeCertificate`; `uniform-cold-freeze.ts` one-way thaw; `uniform-region-heat.ts` current-step reach/omission check.
- Found a proof gap: head heat reads the previous pre-projection velocity certificate. Its maximum cannot prove this step's reach. The new check will use the current extended velocity immediately before transport, after remap/source promotions, and thaw only when the stillness/halo certificate fails.
- Pressure-only trial stays isolated in `/private/tmp/fluid-lane-a/pressure`; current default remains warm. Warm 36-step baseline completed. Other probes are queued behind the existing timing gate.

### 2026-10-25 16:08 · COORDINATION · tip c8b25c39
- Worktree: `/Users/petersuggate/.codex/worktrees/tiers-lane-a/fluid`, branch `codex/tiers-lane-a`.
- Files/functions: `uniform-region-heat.ts` (heat stillness bound), `uniform-mixed-region-dynamics.ts` (pressure publication and exported record stride), `tools/probe-tier-budget.ts` (optional fine-tile coordinate diagnostic). B and C notified; no shared arena/frame/particle edits planned.
- Diagnosis: current head uses all liquid along gravity; disconnected airborne letters inflate a quiet surface's allowance. Testing a pressure-local bound so only round-off at that region's actual pressure can qualify.
- GPU: sequential 36-step letters warm/still probes, full-process check locks, explicit volume control off; logs `/private/tmp/fluid-lane-a/letters-{warm,still}.log`. No gate bundle started while old six-instance gate is active.

### 2026-10-25 · HANDOFF
- The lane was handed over from the Claude lane A agent. The previous owner's note is `laneA-*.md` in this directory.
