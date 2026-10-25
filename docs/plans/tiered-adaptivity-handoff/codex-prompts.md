# Prompts for the tiered-adaptivity agents (2026-10-25)

Start three agents, one per lane, each with its prompt below. Each agent runs in its own git worktree of `/Users/petersuggate/code/me/fluid`, on a new branch from main (`codex/tiers-lane-a`, `codex/tiers-lane-b`, `codex/tiers-lane-c`). The lanes share one GPU through the runner in `tools/gpu/`, so keep to three agents.

---

## Lane A prompt: seep fidelity

You own lane A of the tiered-adaptivity work in the fluid repo (`/Users/petersuggate/code/me/fluid`). It is a WebGPU NB-FLIP fluid solver in TypeScript and WGSL. Your lane's outcome: the `reservoir-seep` test scene behaves with correct physics at 5 mm and later at 2.5 mm.

Before you write any code:
1. Read `docs/plans/tiered-adaptivity-handoff-2026-10-25.md` in full. It has the goal, Peter's physics rules, the state of main, the GPU runner, the gates and how to report.
2. Read `docs/plans/tiered-adaptivity-handoff/laneA-fidelity.md` in full. It has the previous owner's findings, probes and the exact commands.
3. Skim `docs/plans/tiered-adaptivity-huge-grids-2026-10-24.md` §0, §4 and §6 for the plan.
4. Read `AGENTS.md` (checks: `npm run check:types`, `npm run test:unit`).

Your tasks, in order (handoff doc §5, lane A):
- **A1.** Make `coldSurface: "still"` the default, with the wake fix and a proved halo reach bound.
- **A2.** Milestone A: the 5 mm seep over 20 s, volume controller off.
- **A3.** The rest scene settles below 1e-4 m/s.
- **A4.** The 2.5 mm seep, then Phase 4 velocity-projected reach.
- **A5.** Triage the analytic suite's base failures.

Rules that matter most:
- **Correct physics only.**
  - No special cases, flux patches, volume injection, damping or speed clamps.
  - The time step stays 1/60 s, with no time substeps. Splitting one backward trace into segments is allowed.
  - Particles carry thin features, and isolated particles are never deleted.
  - Physics changes go behind NB-FLIP params.
  - Default flips need the evidence listed in handoff doc §3.
- **The main checkout is shared.** Never stash, checkout, reset or rebase in it; work in your worktree.
- **GPU work.** Every run goes through `tools/gpu/fluid-gpu-run.sh check|gate|timing`, with at most 2 of your jobs queued. Detach long runs with nohup.

Report as handoff doc §6 says: keep `docs/plans/tiered-adaptivity-handoff/progress-laneA.md` current and commit only that file on main from the main checkout. Write a `READY TO MERGE` entry with SHA and evidence when a milestone is ready. Don't merge into main yourself. Write `DECISION NEEDED` when Peter must choose, then keep working on something else. Lanes B and C work in parallel: check their progress files, and write a `COORDINATION` entry before large edits to the shared files the handoff lists.

---

## Lane B prompt: memory follows tiers

You own lane B of the tiered-adaptivity work in the fluid repo (`/Users/petersuggate/code/me/fluid`). It is a WebGPU NB-FLIP fluid solver in TypeScript and WGSL. Your lane's outcome: memory follows the tiers. Submerged bulk lives at 16h on small slots, 4h blocks shrink toward the census estimate, and a 4×-widened seep runs with memory reported per tier.

Before you write any code:
1. Read `docs/plans/tiered-adaptivity-handoff-2026-10-25.md` in full. It has the goal, Peter's physics rules, the state of main, the GPU runner, the gates and how to report.
2. Read `docs/plans/tiered-adaptivity-handoff/laneB-memory.md` in full. It has the 16h design in build order, the arena's level lists, the 10-storage-buffer limit, the pool A/B accessors, the per-block byte table and the Milestone B commands.
3. Skim `docs/plans/tiered-adaptivity-huge-grids-2026-10-24.md` §2 and §8 for the design and the h-storage inventory.
4. Read `AGENTS.md` (checks: `npm run check:types`, `npm run test:unit`).

Your tasks, in order (handoff doc §5, lane B):
- **B0.** Cut the 4h block's h-shaped stores, starting with NB-FLIP's per-block bins.
- **B1.** 16h members leave the level-2 lists and the 4h launches, with one coarse row per member region.
- **B2.** 16h members release their block onto a 16h slot budget, with promotion back to 4h.
- **B3.** Milestone B: the 4×-widened seep with 16h on, memory per tier against the census estimate, and frame time.
- **B4.** Pools past 4096 blocks, if needed.
- **B5.** Flip `fineBlocks` to the default.

Extend the CPU twin (`uniform-bulk-tier-reference.ts`) first, and test the compact 16h form against it.

Rules that matter most:
- **Correct physics.**
  - The 16h bulk must not change the jet. The decisive test is the seep A/B with 16h on vs off, within repeat noise.
  - Rest with a 16h bulk must hold at ≤ 1e-5 m/s with zero drift.
- **Allocation.** Allocate upfront, with fixed budgets that fail fast and name the tier. No per-frame allocation, no indirect dispatch.
- **Pressure pools.** Any kernel touching n*, k*, x*, y*, r* or d* channels uses pool B.
- **The main checkout is shared.** Never stash, checkout, reset or rebase in it; work in your worktree.
- **GPU work.** Every run goes through `tools/gpu/fluid-gpu-run.sh check|gate|timing`, with at most 2 of your jobs queued.

Report as handoff doc §6 says: keep `docs/plans/tiered-adaptivity-handoff/progress-laneB.md` current and commit only that file on main from the main checkout. Write `READY TO MERGE` with SHA and evidence per milestone. Don't merge into main yourself. Write `DECISION NEEDED` when Peter must choose. You will edit `uniform-pressure-arena.ts`, `uniform-brick-arena.ts`, `uniform-page-pool.ts` and `uniform-mixed-frame.ts`, which lanes A and C also touch: write a `COORDINATION` entry naming the functions before each large edit, and check their progress files first.

---

## Lane C prompt: compute follows tiers

You own lane C of the tiered-adaptivity work in the fluid repo (`/Users/petersuggate/code/me/fluid`). It is a WebGPU NB-FLIP fluid solver in TypeScript and WGSL. Your lane's outcome: frame time follows live h and 4h work, never pool size or resident bulk, and the cost of the new correct-physics defaults comes down by exact means.

Before you write any code:
1. Read `docs/plans/tiered-adaptivity-handoff-2026-10-25.md` in full. It has the goal, Peter's physics rules, the state of main, the GPU runner, the gates and how to report.
2. Read `docs/plans/tiered-adaptivity-handoff/laneC-compute.md` in full. It has what landed, the measured costs, the verdicts already closed (don't redo them) and the measurement pitfalls.
3. Skim `docs/plans/tiered-adaptivity-huge-grids-2026-10-24.md` §1.4, §2.5 and §3.2.
4. Read `AGENTS.md` (checks: `npm run check:types`, `npm run test:unit`).

Your tasks, in order (handoff doc §5, lane C):
- **C1.** Re-measure and attribute the defaults' cost after lane A's A1 lands. Until then, measure today's costs as the baseline.
- **C2.** The wide-seep scale ladder with 16h on, after lane B's B1/B2.
- **C3.** Exact savings in letters' particle paths.
- **C4.** Fix or retire `tools/benchmark-nbflip-letters.ts`.
- **C5.** The length ladder: slope ≤ 0.2 ms per doubling.

Rules that matter most:
- **Exact savings only.** A compute change must be physics-neutral (unchanged inputs give unchanged outputs) and pass the gate bundle (`tools/gpu/gate-bundle.sh`) against main. Never trade accuracy for speed.
- **Repo rules.** One call site per heavy WGSL function, no indirect dispatch, zero per-frame allocation, lean split kernels over compact lists.
- **Timing hygiene.** Timing runs use `tools/gpu/fluid-gpu-run.sh timing` and paired or in-run alternation (`tools/benchmark-region-tier-paired.ts`). Single runs vary up to 2.4×. Check `ps` for other GPU users before trusting a number.
- **The main checkout is shared.** Never stash, checkout, reset or rebase in it; work in your worktree.

Report as handoff doc §6 says: keep `docs/plans/tiered-adaptivity-handoff/progress-laneC.md` current and commit only that file on main from the main checkout. Write `READY TO MERGE` with SHA and evidence. Don't merge into main yourself. Write `DECISION NEEDED` when Peter must choose. Lanes A and B change the physics and storage under you: re-baseline on main after each of their merges, and write a `COORDINATION` entry before editing the shared files the handoff lists.
