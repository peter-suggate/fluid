# Tiered adaptivity: completion handoff

**Paused at Peter's request, 27 October 2026. The goal is not complete.** No lane agents or GPU probes are running at handoff. This document supersedes historical “running”, “next” and parallel-lane instructions in the original handoff and progress files. Resume only when Peter requests it.

## 1. Outcome and priorities

Deliver expandable fluid scenes with high resolution around surfaces, jets and active regions, evolving 4h/16h bulk, and memory and work that follow those tiers. Fluid must remain physically conservative, including detached particles, during transport, tier changes and world-window movement. The reservoir-seep scene must visibly work in the UI.

Peter's latest priorities are **one agent, one focus; finish the main functional goals, then pause**. Use focused checks. Do not repeat full unit suites or spend time chasing optional performance improvements. Reducing allocations enough to prevent GPU/desktop failure is necessary stability work, not optional tuning.

The original [handoff](tiered-adaptivity-handoff-2026-10-25.md), especially §§1, 3 and 5, remains the source for original acceptance criteria. This document defines the current sequence and distinguishes verified components from unfinished end-to-end goals.

**Main completion gates:** safe execution; conservative 5 mm seep for 20 s; working compact 16h bulk in the wide scene; conservative automatic expansion and readmission; real particle mass/momentum preservation; visible accepted surface in the UI. Preserve the original rest, interface and wake requirements. The original finer-resolution A4 path remains a required correctness follow-on where capacity permits, not a reason to weaken the 5 mm gate.

**Deferred:** C5 timing slope, C3 tuning, further optional pressure/momentum packing, and chasing B0's historical sub-0.5 MiB estimate. Report actual memory and compute behavior, but do not keep optimizing after functional acceptance. A5's pre-existing analytic failures need a disposition, not an unbounded unrelated repair campaign.

## 2. Physical and execution constraints

- Fixed physical dt = 1/60 s; no physical substeps. Segments of one frozen backward characteristic trace and chunks of numerical solver work are allowed.
- Preserve the authored gravity, viscosity and surface tension. Measure with `volumeControlSeconds: 0`.
- No artificial inflow/outflow, volume injection, damping, speed clamp, residual floor or deletion of positive phases. Preserve liquid and its momentum; particle retirement requires proof that the level set holds the same liquid.
- Keep original pressure divergence (1e-4/s), phase, marginal, closure and interface gates. Do not turn a rejected run into a pass by loosening them or discarding positive coefficients.
- 16h is liquid-only bulk: no free surface, cut solids or particles there. Boundary wall faces must receive the correct treatment. Coarse tiers evolve; COLD is only for provably still regions.
- The still-surface certificate allows only round-off-sized per-step omission, accumulated omission ≤ 0.01 cell, and sufficient wake reach. Preserve the original halo/travel requirements.
- Allocate simulation buffers and pipelines upfront, no simulation-time compilation or indirect dispatch. Explicit fixed capacities; reject exhaustion or stale identities. Never silently truncate or resize.
- GPU work is exclusive and bounded. Bounded host command recording does not permit new simulation GPU resources each frame. A rejected step must preserve the last accepted simulation and rendered state.

## 3. Where the work is

| Checkout | Branch / handoff tip | Role |
|---|---|---|
| `/Users/petersuggate/code/me/fluid` | `main`, `9962636d` before this handoff | Documentation and user's working checkout. Contains unrelated uncommitted implementation work. |
| `/Users/petersuggate/.codex/worktrees/tiers-rounded-still/fluid` | `codex/tiers-transport-map`, `71f31539`; latest implementation `43bee750` | Current single-agent conservative solver and GPU stability work. **Resume implementation here.** |
| `/Users/petersuggate/.codex/worktrees/tiers-baseline/fluid` | `codex/tiers-integrated-acceptance`, `39525158` | Isolated reviewed integration of earlier A/B/C work and its acceptance ledger. |

Other lane worktrees are historical sources, not instructions to restart those lanes. Use `git worktree list` and verify the current tips before work. The current root worktree has a pre-existing untracked `node_modules` link; do not commit it.

Never stash, reset, checkout or rebase the main checkout. Its uncommitted renderer/particle work belongs to another effort. Commit only owned paths. No implementation merge into main is authorized by this handoff; prepare a reviewable packet for the coordinator. Documentation-only commits on main are the existing reporting convention.

Read these in order, without replaying their entire history:

1. This document and the original handoff's physical rules.
2. Current root `docs/plans/tiered-adaptivity-handoff/conservative-seep-capacity.md`, starting at its final GPU incident section. Earlier sections preserve failures and repairs.
3. Root `docs/plans/tiered-adaptivity-handoff/laneA-pressure-tree.md` when working on pressure/transport.
4. Integration checkout `docs/plans/tiered-adaptivity-handoff/integrated-acceptance-ledger.md`, especially “Compact16h wide functional milestone VERIFIED and integrated” and “Expandable world and compact-state shift VERIFIED”.
5. Older handoff documents and lane progress files were removed from main at Peter's request. If historical detail is needed, retrieve its exact path from commit `30feacbf` using `git show 30feacbf:<path>`. The separate worktree ledgers above remain available; their old parallel assignments are superseded by this pause/single-agent handoff.

## 4. What is actually verified

| Area | Verified evidence | What it does not establish |
|---|---|---|
| Automatic fine storage | B5 `36630727`, integrated `caa8129e`, passed the frozen splash/seep/letters comparison bundle with independent source and occupancy audits. Combined storage `913d27ef` also passed its scoped neutrality bundle. | Conservation or final memory target. Its measured marginal was 781.32 KiB/block, above the old B0 target. |
| Compact 16h storage and pressure | Runtime `b48d3425`, packet `097fa6f6`, integrated `8b1f423d`: wide 120 frames passed, final 267 native + 287 compact pages, 968 exact storage transactions, unchanged pressure gates. Live/peak allocations 2.359/5.623 GB. | Full conservative-carrier evolution, long-horizon physics or a pressure-memory saving at that declared reserve. |
| Window movement | C packet `43e1ac81`, integrated `deafd7db`: compact state survives a shift and promotion with exact transaction checks. Travel120 crosses the original 24 m edge and reaches 46.75 m with fixed pool/allocation. | Repeated recentering, readmission or conservation. Travel lost 2.694% represented volume; its inherited 6% travel gate is not acceptable proof of conservative fluid. |
| Conservative transport/forces | Multiple saved donor/geometry/pressure failures have focused repairs. Small full-force evolution passes; one historical coarse 30-step pass had LIQ/AIR changes about 4.23e-7/4.26e-7 and closure 1.93e-6. Later coarse trajectories exposed additional failures. | No clean current-source 20 s pass. Do not treat individual saved-input repairs as a trajectory pass. Runtime momentum is still grid-based; final particle coupling is unfinished. |
| Authored 5 mm topology | `3f660afd`, complete-box control: 234,560 h + 92,080 4h + 616 16h open owners, 197,004 mesh vertices, exactly unchanged phase totals, closure 8.65e-9. | A physical timestep or sparse-default AIR support. |
| Captured 5 mm pressure | `1f7424db`: complete GPU-built hierarchy from physical inputs, no CPU hierarchy packets. Wet/AIR 544/27 iterations within original 2048 budgets. Independent all-owner divergence max 4.857e-5/s; all 348,539 frozen wet faces bit-identical through AIR extension. | A live timestep or full-scene acceptance. |
| Live hierarchy integration | `790a7052`: both native-frame and owner-candidate paths use it; two small live constructors pass, including wet walls and a 16h owner. | Safe large-scene execution. The later large run ended in a desktop/GPU incident. |
| GPU safeguards | `43bee750`: focused CPU/stream/presentation checks, tiny GPU bit-equivalence, runner exclusion and an actual small native-wake physical step pass. LIQ/AIR changes below 3.89e-9 in that step. | A proven fix for every GPU hang or safe full-scale execution. |
| Still default | A's `43b1aa3c` is a scoped READY recommendation under deliberate-default comparison rules. | A2 conservation, or permission to silently change the conservative probe's current `coldSurface: warm` configuration. Final intended defaults still need explicit validation. |

**The latest authored 5 mm run has zero accepted full physical steps.** Its startup half kick completed. Initialization, topology transactions and captured pressure solves must not be counted as completed physical evolution.

## 5. GPU incident: do not repeat the unsafe run

The `790a7052` run requested **17,931,138,268 bytes** of buffers/textures. Its process reached **23.73 GB** in a WindowServer spin report. WindowServer watchdog reports on 27 October at 14:43/14:44 NZDT record 40 s missed check-ins; the terminated process subsequently remained in IOGPU/AGX cleanup. The machine rebooted at 14:55:30. The full physical step never returned a verdict. The exact shader, driver failure or OOM cause is **not proved**.

Investigation found giant command streams and startup/full-step overlap. A guarded small constructor also exceeded four million recorded GPU API calls before the new stage boundaries were introduced. Requested GPU bytes omit substantial command/driver overhead.

Safeguards already implemented on `43bee750`:

- Serialize conservative frames. Wait at existing topology, transport, momentum and force/projection boundaries.
- Native command buffers contain at most eight complete passes/copies; wait for completion before constructing/submitting the next batch. Recording caps are four million API calls and 64 dispatches per pass.
- A 10 s batch timeout stops subsequent submissions and identifies the batch. **It cannot cancel an already hung kernel.**
- Publication/presentation waits for the full accepted frame receipt. Destroy cancels pending recording.
- All GPU runner modes acquire exclusive locks.
- The large constructor probe caps requested resources at `min(12 GiB, 3/8 of host memory)`, including buffers and textures before allocation. On this machine the ceiling is 12 GiB. `--max-bytes` can lower it, not raise it.

**Do not raise this guard or rerun the old 17.93 GB configuration.** The cap currently protects the constructor probe; do not assume every replay or UI route inherits it. Older captured-pressure replays submit directly and must be made bounded/admitted before another large run. The exclusive runner cannot prevent an independently opened UI tab from consuming the GPU: keep one task-owned simulation active, and do not terminate unrelated user processes.

Incident evidence is root `artifacts/single-focus-seep/gpu-stability-incident.json`, including scoped OS report paths/hashes. Earlier “pending run, preserve process” instructions are obsolete. There is no old process to resume.

## 6. Ordered completion plan

### Gate 1 — fit the required work safely

**Immediate next task: CPU-side allocation/lifetime audit, then fixed arena reuse.** The four live preconditioner clients (native wet/AIR and candidate wet/AIR) currently retain substantial duplicate storage despite sequential use. Share only resources whose lifetimes are proved disjoint. Keep physical inputs, outputs and outer solver ownership distinct.

A proposed shared fixed hierarchy arena plus sequential sparse-product/prolongation scratch reuse projects roughly 12.63 GB requested resources, down from 17.93 GB. **This is an unimplemented estimate, not an admission or stability result.** It is close to the ceiling and does not account for all driver overhead. Preserve the proven required capacities and all positive coefficients; do not shrink budgets merely to get past admission.

Required design checks: exact layout compatibility; full rebuild before use; a shared monotonic generation plus each client's private prepared generation; reject old-client apply after another client prepares; reject partial/unready builds; clear scratch on reuse; one allocation owner and correct borrower destruction; count shared bytes once. Existing builders destroy their owned buffers, so unsafe aliasing is not a solution.

**Accept:** independent resource census fits the host reserve, focused alternating-client/stale/overflow/smaller-rebuild/destruction controls pass, no per-frame simulation allocation, and small live physical checks preserve their original gates. Then run one bounded admitted constructor and first step, observing process memory and batch completion. A timeout/device loss stops the campaign; save the last stage and do not immediately relaunch. Do not claim full-scale safety merely because admission passed.

### Gate 2 — demonstrate actual conservative evolution

Use the authored reservoir-seep with fixed physics. First obtain one accepted 5 mm physical step, then short continuation, then the 20 s / 1200-step A2 milestone once prerequisite checks pass. Each failure should produce an immutable stage capture; repair the first failing stage with a focused regression before extending the horizon. Do not rerun a successful unchanged test without a dependency change.

Track LIQ and AIR totals, closure/marginals, phase positivity, force and divergence gates, accepted epochs, visible surface, throat flux, chamber volumes, wet speed/location and all exhaustion/retirement counters. Attribute wet-jet peaks using actual liquid-bearing faces, direction, mass and force stages; unfiltered AIR velocity maxima are insufficient. Do not hide physical speed peaks with a cap.

**A2 accepts only when:** 1200 full steps complete without fatal; the right chamber fills; left-level fall agrees within 10% of Torricelli at measured Cd; drift is small and explained; the fast-jet fixture stays bounded near 3.96 m/s. Report `overCap`, `mergedMassRetired` and any deep/held-heat retirement. Keep existing numerical conservation bounds; the old milestone's “small and explained” wording does not authorize inventing a looser tolerance after seeing a result.

The complete-box probe is a diagnostic control. Its success alone cannot close the user-facing milestone. Final acceptance must use the intended scene/default settings after Gates 3–5 are incorporated; short control runs can proceed earlier to isolate failures.

### Gate 3 — sparse support and real particles

Repair default sparse AIR publication: it still rejects unsupported unresident AIR. Define physically correct coverage/collars and admission for the scene; do not emulate an inflow, delete AIR or substitute permanent complete-box residency as the expandable solution.

Integrate particles as real liquid with conservative mass and momentum exchange. The current experimental grid-momentum path is not the final particle answer. Preserve isolated droplets through band exit, merging, splitting, coarse transitions and readmission; retirement is only valid after equivalent liquid is represented in the level set. Focus regressions on the original donor-loss situations and true thin/tiny positive phases.

**Accept:** sparse authored initialization and evolution, no unsupported boundary, conserved mass/momentum through explicit particle-grid transitions, no unexplained retirement, and unchanged positive-phase/closure gates. Repeat only affected end-to-end milestones on the resulting source. Also retain A3 rest speed < 1e-4 m/s with no volume drift; any repair must identify its physical/numerical cause rather than damp it.

### Gate 4 — integrate conservative compact 16h evolution

Reuse the accepted compact storage design, not a fresh implementation. Bring it together with the conservative carrier, force/pressure and particle path. Verify that 16h members leave 4h work and backing, evolve at 16h, and wake/promote from fixed storage with correct wall and h/4h/16h interface treatment.

**Accept:** original B1 rest bound ≤ 1e-5 m/s and zero drift; 16h-on/off seep discharge and wet jet within repeat noise; correct generation and poisoned-slot reuse/exhaustion rejection; wide 5 mm seep (`probe-scale-ladder.ts --width=2 --depth=2`) with active compact bulk and real physical evolution. Report memory by tier and work attribution. Reuse historical wide120 evidence only for unchanged components; it does not validate the new conservative combined path. Run 2.5 mm when it fits fixed safe capacities, with the original A4 fast-front reach/discharge requirements. Capacity expansion is explicit and still subject to the host reserve.

### Gate 5 — conservative expandable world and readmission

Reuse C's verified world-coordinate/window work. Join native backing, compact owners, particles, carrier, pressure and surface into one accepted generation. Exercise automatic repeated recenters in both directions, crossing boundaries and returning to previously visited world coordinates. Preserve retained fluid; a departure from the active window must not silently erase liquid.

**Accept:** fixed resource allocation, correct world identities and nonoverlapping maps, current/presented agreement, conserved phase and momentum through shifts/readmission, no stale-slot resurrection, and exhaustion/rejection preserving the last accepted state. A single forced shift or the old travel test's 6% drift allowance is insufficient. “Infinite expandable” means a moving bounded active simulation with correct world-state semantics, not unbounded GPU allocation.

### Gate 6 — final integration, UI and review packet

Freeze the combined source and intended defaults. Validate `reservoir-seep` through the actual UI, including loading, continuous level-set visibility, accepted-frame presentation, pause/resume and scene teardown/reload. Confirm device loss stops work cleanly without silently continuing on invalid resources. The earlier reload/reboot recovery is not proof of a solver/rendering fix.

Run focused affected regressions and required scene comparisons on this exact source. Reuse unchanged audited results instead of repeatedly running full suites. Record any remaining baseline analytic failures and their relevance. Review all original acceptance items as passed, deferred by Peter's priorities, or still open; do not silently drop an item.

**READY TO MERGE packet:** exact commit(s), source hashes, resolved settings/capacities, commands, raw receipts, independent audits, resource totals, UI result and a compact goal-by-goal verdict. The coordinator verifies and handles main integration; acceptance on an isolated branch is not a main release. Once the main functional gates are met and reported, **pause**. Do not start another optional optimization campaign.

## 7. Checks, commands and evidence discipline

Repository checks are `npm run check:types` and unit tests. Peter specifically asked for focused checks instead of repeated full `npm run test:unit` runs. Select affected test files and named cases. Types currently have only the two known missing imports in `app/app/advance-lab/page.tsx` and `app/app/shape-lab/page.tsx`; preserve the exact output and do not classify new errors as baseline automatically.

After resume, the small stream/memory/presentation tests are relevant starting points if these implementations change:

```sh
node --import tsx --test tests/gpu-memory-profile.test.ts tests/uniform-conservative-command-stream.test.ts tests/uniform-presentation-completion.test.ts
npm run check:types
```

Use Node 22 at `/Users/petersuggate/.nvm/versions/node/v22.22.1/bin`. GPU jobs must use root's updated `tools/gpu/fluid-gpu-run.sh`, with `WEBGPU_NODE_MODULE=/Users/petersuggate/code/me/fluid/node_modules/webgpu/index.js`. The wrapper form is `bash tools/gpu/fluid-gpu-run.sh check node --import tsx <focused test or probe>`. Verify admission and bounded submission for that specific probe first.

The authored 5 mm control's proven capacity requirements are 512 bricks, 393216 owners, 4194240 transport edges, 14409728 strain rows, 67108864 strain entries and 524288 fits. They are not a ready-to-run large job: the current allocation does not fit the new guard. `--resident-box` is a diagnostic control, `--wake-preflight` is topology-only, and `--steps=N` requests physical steps. Preserve source/settings manifests before launch; count completed physical receipts, not submitted calls.

Evidence lives in worktree-local `artifacts/` directories and may not be tracked by Git. Preserve it or archive with checksums before removing worktrees; source commits alone do not preserve raw receipts. Do not rely on `/tmp` surviving reboot. Freeze failed evidence as well as successes.

| Evidence location | Purpose |
|---|---|
| Root `artifacts/single-focus-seep/gpu-stability-incident.json` | Watchdog incident, scoped system evidence and hashes. |
| Root `gpu-stability-all-focused-cpu.log`, `gpu-stability-final-stream.log`, `gpu-stability-presentation.log`, `gpu-stability-lock-test.json` in that directory | Safety regression and exclusivity checks. |
| Root `gpu-stability-large-admission.log`, `gpu-stability-live-staged.log`, `gpu-stability-committed-types.log` | Rejected oversized override, accepted small live step, types. |
| Root `seep5mm-live-pressure-first-step*` in that directory | Terminated large run, startup success and source audit; no full-step verdict. |
| Root `seep5mm-pressure-live-hierarchy*` | Captured complete GPU pressure pass and independent/source audits. |
| Root `seep5mm-all-coarsening-funded*` | Complete-box topology publication, not evolution. |
| Integration `artifacts/tiered-review/pressure-live/` | Compact storage closed/open/wide audits, including `wide120-E256-independent-audit.json`. |
| Integration `artifacts/tiered-review/expandable/` | Exact compact-state shift and travel evidence with conservation limitations. |
| Integration `artifacts/tiered-review/b5-default/` and `reuse-history/` | Accepted fine defaults and combined storage neutrality. |

## 8. Resume instruction

> Resume the tiered-adaptivity goal as one agent. Read the 2026-10-27 completion handoff and current root capacity ledger. Verify worktree tips, uncommitted changes and GPU process state. Continue in `codex/tiers-transport-map`; do not touch unrelated main changes or restart lanes. Start with the CPU allocation/lifetime audit and fixed shared-arena design needed to fit required 5 mm resources within the existing host reserve. Keep all physics, capacities and acceptance gates intact. Use focused checks, exclusive bounded GPU work, immutable failure captures and independent audits. Work through conservative evolution, sparse support/particles, compact 16h and repeated world-window readmission to final UI acceptance. Prepare a review packet without merging implementation into main. Pause when the main functional goals are met; do not chase optional performance targets.
