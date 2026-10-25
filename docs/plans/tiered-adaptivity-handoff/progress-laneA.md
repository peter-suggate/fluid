# Lane A progress

Written by the lane A agent; read by Peter and the coordinator. The format and rules are in `../tiered-adaptivity-handoff-2026-10-25.md` §6. Newest entry first.

## Status
- **Tip:** c8b25c39 (branch codex/tiers-lane-a)
- **Doing now:** A1 letters tile diagnosis and local pressure round-off bound.
- **Next:** A1 gate/flip evidence, then A2 candidate 1200-step seep.
- **Waiting on:** shared GPU; existing coordinator gate. Root is running the current-default 1200-step seep baseline.

## Log

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
