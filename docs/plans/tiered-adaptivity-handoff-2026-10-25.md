# Tiered adaptivity: handoff to the next agents (2026-10-25)

This doc hands the tiered-adaptivity work (`tiered-adaptivity-huge-grids-2026-10-24.md`) from the Claude lanes to a new set of agents.
- The new agents implement.
- Claude (the coordinator session) plans and verifies.
- Peter decides.

**Read §1–§4 before starting. §5 is your lane. §6 says how to report.**

## 0. Status board (the coordinator keeps this current)

**Current execution mode (2026-10-26): one agent, one focus.** A/B/C agents remain stopped with their worktrees preserved. Root alone continues conservative reservoir-seep evolution in `codex/tiers-transport-map` (live integration `0a82fcd3`, physical-input guards `adbbd6c2`, ledger `fc9c6634`). A fixed GPU spanning forest now selects pressure-difference coordinates; the transformed operator preserves every original h/4h/16h whole-face incidence and fixed massless-face RHS term. Signed factored Krylov state and compensated reductions solve the same pressure objective without mass floors, clamps, force reapplication or physical substeps. Main has no implementation merge. No optional performance tuning or full unit suite ran.

**Captured pressure blocker repaired.** The exact step29 failure now passes the complete GPU pressure path: liquid1376iterations, shared native wet maximum6.29425e−5/s; dry545iterations, final maximum9.17062e−5/s, all within the original1e−4/s gate and2048 budgets. Root independently checks all12480 active owners and bit-identical velocities on all11916 frozen wet faces. Correction energy agrees with the independent reference to7.18491e−10 relative difference. The original matrix's126213 tree-coordinate entries pass an independent incidence-path audit; no term is dropped. Focused topology2/2, operator2/2, solve1/1 and bridge/guard2/2 checks pass, including all17 coarse/fine incidences, mass2^-160, fixed-flux forcing, startup modes, diagnostic pressure, native-verifier rejection, stale input and capacity rejection. Capacity checks4pass/1GPUconstructor skipped; types retain only the two known lab imports. Evidence: `newton29-gpu-tree-full.*`, `newton29-gpu-tree-full-independent.*`, `newton29-gpu-tree-full-objective.json` and `docs/plans/tiered-adaptivity-handoff/laneA-pressure-tree.md` in the root checkout.

**Verified actual30-step checkpoint (2026-10-27).** The h=.04m live run on `0a82fcd3` passed startup and all30 physical1/60steps, including the former step29 pressure failure. Root's independent sequential log audit checks steps0..30: LIQ/AIR maximum relative changes4.22794818e−7/4.26357276e−7, maximum closure1.93015148e−6, and a nonempty surface throughout. Final epoch72,22848 mesh vertices,12480 active owners (12352h+128×4h),234 native fine blocks,92733unique/284562raw transport support within1048576 fixed edges. Evidence: `artifacts/single-focus-seep/tree-pressure-live30.log` and `tree-pressure-live30-audit.json`; log SHA256 `e6ab81b1cedf11704bd6bc1b62974573b6b11ef7417955776c8eb53a3a3f4a83`. This is a0.5s coarse checkpoint, not20s5mm/A2, conservative16h or moving-window acceptance. The later guard-only commit has focused validation; the full run is attributed to0a82fcd3. Next single focus remains conservative seep evolution; required viscosity-capacity and window drafts stay paused.

**Continuation (2026-10-27), one agent / one focus.** The previous60-step check accepted36steps, then rejected37 on transfer marginal2.62260437e-6 versus the original2e-6 gate; the later force/pressure failures were prerequisite failures. Root repaird6aaeae4 preserves small Newton potential increments and removes irrelevant common offsets before row normalization. The captured transfer passes at1.07288361e-6 after3corrections with unchanged budgets. Independent audit retains all96379positive edges, donor error1.64775e-7, published closure9.74498e-7, and LIQ/AIR relative changes1.73094e-8/2.12680e-8. Focused7/7 pass; types have only the two existing lab imports. This is captured-transfer evidence, not a live-step or geometry pass.

A new60-step live run on d6aaeae4 is underway in root's `artifacts/single-focus-seep/gauge-transport-live60.log`, with accepted-carrier chamber/discharge/jet measurements enabled and complete rejection capture. Read-only measurements34469aa9/1451b12c have focused2/2 and strict startup1/1 checks; full-liquid throat flux requires exact absence of stored AIR. Root remains the only active agent, focused on actual conservative seep evolution. Other lanes remain paused; no optional performance tuning or full unit suite. Details: root branch ledger7ebf4c50, `docs/plans/tiered-adaptivity-handoff/laneA-pressure-tree.md`.

**Current user priority (2026-10-26).** Complete the main functional goals—expandable scenes, working16h simulation with compact state, and conservative fluid behavior—then pause. Performance tuning, C5 slope chasing, optional storage micro-optimizations and census-estimate targets are deferred when they do not block those functions. Keep focused correctness checks and explicit failures; do not rerun full unit suites. The older performance targets below remain reference work, not prerequisites for this pause.

| Lane | Owner | Branch | Last tip | State | Next milestone |
|---|---|---|---|---|---|
| A: seep fidelity | Codex A | `codex/tiers-lane-a` | `20bed391` | Agent paused; native wake checks preserved | Root alone continues actual conservative seep |
| B: memory follows tiers | Codex B | `codex/tiers-compact-owner-live` | `1c95f575` | Agent paused; compact constructor checks preserved | Adaptive strain storage capacity for5mm, deferred |
| C: expandable scenes | Codex C | `codex/tiers-native-divergence` | `ca60cf2e` | Agent paused; pressure checkpoint integratededd6f63c | Moving-window transaction draft preserved, deferred |

**UI investigation closed by user, 2026-10-26.** The user restarted the machine and confirms reservoir-seep loads correctly. All lane GPU holds are released; work returns to the functional goals. No runtime crash fix or root cause is claimed. Mainfbafe938 retains additional loss diagnostics; main types and focused worker shutdown2/2 passed. Earlier Chrome reproductions and standalone startup measurements remain diagnostic evidence only.

**Coordinator verification, 2026-10-26.** Implementation remains in the isolated `codex/tiers-integrated-acceptance` branch; main's unrelated working changes are preserved. User instruction supersedes repeated full-suite runs: development uses focused checks and milestone gates. The isolated candidate's float-default A/B and A/A bundles, pressure/frame-storage bundle, solid-metadata bundle, and combined generation/node-gather bundle pass independently audited gates. The node gather saves paired medians of 7.271 ms on letters and 2.420 ms on seep; these are workload costs, not isolated-kernel measurements.

**Current correctness checkpoint.** The experimental directional map, compensated transfer sums and duplicate-pair consolidation retain original2e−6 marginal/closure gates; moving h/4h/16h and mixed wake proofs remain scoped as recorded below. The full-force dry20 failure is now diagnosed and its exact captured candidate is repaired: the old whole-face dry graph manufactured an isolated AIR pocket despite an open PLIC AIR path. Joint massless compatibility constrains the same liquid kinetic objective; it adds no AIR inertia, floor, clamp or time subdivision. Fixed GPU topology builder82549348/6d707b41 plus C pressure67698841 pass the actual capture with the shared canonical RT0 verifier: liquid136iterations, maximum8.24105e−5/s; dry1051iterations, maximum6.83020e−5/s; original1024/2048 budgets and1e−4/s limits. Root independently checks all20544 active owners, bit-identical frozen positive-mass faces and liquid correction objective3.28575e−10. A bounded topology builder rejects irreducible adaptive hyperedges instead of treating connectivity as the full nullspace. Runtime/candidate integrationc7643848 passes an actual automatic mixed wake and full-force frame1/1; types retain only the two known imports. Live coarse30 with explicit4h collar REJECTED during startup at step1 on that exact source, before transport: liquid failure0,403iterations, canonical wet divergence4.40764e−5/s but accepted0; the subsequent dry stage rejects its liquid prerequisite256. C643fdd94 repairs recursive early convergence that omitted the gauge equation, whose actual residual was0.0092473/s. Only existing reliable replacements now authorize mapped convergence; original budgets/tolerances remain. Captured startup passes576/415iterations; root independently verifies12480owners, mapped maximum7.52896e−5/s, final physical maximum9.33256e−5/s, and frozen wet faces bit-exact. Live30 on exact6dd11e1b passed startup but failed physical step1 at the unchanged1024 liquid budget. Its omitted gauge residual was independently2.80282e−4/s. Cca60cf2e solves reliable corrections against the exported velocity, avoiding loss of small increments in accumulated f32 pressure. Captured step now passes640 liquid/497 dry iterations; independent final maximum9.76066e−5/s, frozen wet faces bit-exact. Focused9/9 pass; root reviewed/pickededd6f63c. No horizon pass is claimed. The5mm seed audit finds85,322 active open+solid owners with explicit4h initialization versus4,980,800 open owners in default all-h initialization; merely packing unused slots cannot make all-h fit. B865ca80e compact graph and joint publication pass focused checks:128×4h→65logical including16h→128×4h, fixed1024slots versusnative32768, exact spatial carrier return and failed-mesh rollback of all joint buffers. B’s public fixed-capacity constructor51ca821d and normal full-force check pass1/1. Root merged the reviewed B checkpoint through1c95f575 into the experimental branch; a new probe-only profile-nullability type error was corrected atbd54fcc7. The5mm compact constructor and adaptive strain capacity remain unverified. A adaptive collar fixcdeedb3d is reviewed and integrated8a3f4fcd: exact-owner full-force2/2 passes at epoch5, liquid relative change8.52651e−10, AIR0, and all24,576 classified solid records preserved. Tiny boundary controls reject missing/malformed solids, open exterior and arbitrarily small positive liquid. The zero-measure proof does not depend on velocity fitting. A now works on transactional native fine backing; merely removing the native-h guard would import incorrect coarse momentum and is not accepted. General force-compatible coarsening, native fine backing/wake, moving conservative state, particle-momentum reconciliation and the20s conservation milestone remain open.

**Goals still open.** A2's earlier1200-step run lost2.551L of represented phi volume and is not a conservation pass. B5 automatic fine storage `36630727` is independently VERIFIED and integrated as `caa8129e`: required splash, seep and letters gates pass, including canonical owner-phi occupancy and source identity. B0 remains over budget: the combined storage candidate `913d27ef` measures742.386482/875.149902MiB at226/400blocks, or781.320KiB per additional block; the complete splash/seep/letters neutrality bundle is independently VERIFIED. Compact16h storage and the scoped wide120 functional milestone now pass. A4/adaptive conservative coupling remain unfinished; C5 and the historical B0 estimate target are deferred under the current user priority. Scoped primitive checks do not complete these milestones.

**Latest functional verification.** B wide120 remains VERIFIED at8b1f423d. C packet43e1ac81 is independently VERIFIED and integrated atdeafd7db: travel120 crosses the original24m boundary at61 and reaches46.75m at fixed2,883,260,024B, with−2.694% represented drift within the inherited6% travel bound. Combined compact16h shift60 retains fixed2,771,103,268B;64 compact pages survive shift40 and promote safely at41. All60 pressure closures pass and3,870,720 pressure words verify exactly. Root checks all923 runtime files against exact2ab1e492 Git source and identical B-control scene/settings. This proves bounded-world travel and safe compact promotion, not exact conservation, repeated automatic shifts or compact re-admission. Types retain only the two known lab import errors. No full unit suite was rerun.

**COORDINATION / deferred work.** The earlier C5 ladder, cache timing and B0 estimate targets above are historical, deferred by the current user priority. Root's optional shared-certificate change is reverted on the integration branch at369f9a5c; its focused checks and partial allocation receipts are retained without a READY claim. B owns required pressure/compact momentum, A conservative evolution, and C functional world expansion. No further performance tuning is scheduled.

Full verification records and exact raw-evidence paths are in `docs/plans/tiered-adaptivity-handoff/integrated-acceptance-ledger.md` on the isolated integration branch. Lane progress files on main retain each lane's newest evidence and coordination entries.

Main at handoff: see the commit that adds this doc. The defaults are listed in §2.

## 1. Goal and the rules Peter has set

**Goal (Peter):** "double down on adaptivity to ensure extremely high fidelity physics and fluid surface treatment at areas of high heat and much less elsewhere in order to save compute … support extremely large (high resolution) grids that are mostly 4h or 16h with only h at certain areas. both the memory and compute should follow this pattern with enough memory buffer to allow for some growth … a plan that allows a huge hi-res scene to render fast with only isolated regions of full res".

**Test scene:** `reservoir-seep` (`lib/core/reservoir-seep-scene.ts`). Deep water on the left and an empty catch chamber on the right, with a wall between them pierced by a **2×2-cell square throat**. Peter accepted 2×2 in place of the 1-cell hole: "we can accept a larger hole (2x2)".

**Physics standard (Peter, non-negotiable):**
- "we want no hacks/inflows/outflows in the scene. we want correct physics". Every fix must be the standard, physically or numerically correct treatment. That rules out:
  - special-case exceptions;
  - flux patches;
  - volume injection;
  - damping, clamps or speed limits to hide an instability;
  - the volume controller doing the real work. Measure conservation with `volumeControlSeconds:0`.
- If a scene can't be resolved, change the scene; don't special-case the solver.
- "particles are our answer for: disappearing fluid and also correct behavior at the seep hole". Isolated particles are real liquid: never delete or retire them, or the liquid they carry, unless the level set provably holds it.
- Coarse tiers must still evolve with the solved velocity at their resolution. COLD is allowed only where it is provably still (§5 A1).
- **Time step.** It stays 1/60 s, with no time substeps and no smaller scene dt. Splitting one advection's backward trace into segments on fast faces IS allowed (Peter, 2026-10-25); that is `momentumTrace: "substep"`.

**The four plan decisions (2026-10-24):**
1. 16h is liquid-only bulk (no surface, cut solids or particles at 16h).
2. 1/60 s, with fast-front reach velocity-projected later (Phase 4).
3. Wall faces on 16h boundaries.
4. Phase 2 is the arena migration, not a parallel effort.

**Repo rules:**
- Allocation and compiles:
  - No indirect dispatch.
  - Zero per-frame allocation.
  - No compile during simulation.
  - Allocate upfront and fail fast; no silent truncation (count and report any cap that is hit).
  - One call site per heavy WGSL function.
  - Lean split kernels over compact lists.
- Physics changes go behind NB-FLIP params (`lib/methods/uniform/uniform-narrow-band-method.ts` and its param tables).
- No step limits.
- Git:
  - **Never** stash, checkout, reset or rebase in the main checkout `/Users/petersuggate/code/me/fluid`. Other sessions have uncommitted work there (renderer, UI, svo).
  - Work in your own git worktree.
  - Commit only your own paths: `git commit -- <paths>`.
- Checks: `npm run check:types` (ignore the two `app/app/*-lab` TS2307 errors from a stray symlink, and errors in files you did not touch) and `npm run test:unit`.

## 2. State of main at handoff

**Defaults now on** (all NB-FLIP params; the old values stay selectable):

| Param | Default | What it does |
|---|---|---|
| `momentumTrace` | `substep` | Splits momentum's backward trace into RK2 segments of ≤ 1 cell on every tier, capped at 128. A capped trace is counted in claim word 4 (`UNIFORM_MOMENTUM_TRACE_CAPPED_WORD`). This fixed the seep jet ratchet: 35 m/s → 3.7 m/s, under the 4.34 m/s energy bound. |
| `coldSurface` | `warm` | Free-surface regions are never COLD; COLD stays for submerged bulk, air and solid. `still` (opt-in) lets a provably still surface freeze (§5 A1). |
| `seedScoring` | `every` | The census scores every surface tile, so "need" seeding sees jets and films that the shape criterion already holds at h. Before this the seep had 0 particles. |
| `outerSeeding` | `uncovered` | Seeds outer-layer cells with no sample in their 3×3×3, so liquid leaving the throat carries samples. |
| `heatExpiry` | `held` | Heat expiry retires only samples deep in the level set. |
| `sparseNeighbours` | `0` | No isolated-particle expiry. |
| `sampleBound` | `walls` | The particle union's bound fills from walls, so a film shallower than half a cell counts. |
| `nativeSmoothing` | `composite` | Red-black relaxation of the native 4h rows against the band's real iterate. This fixed a period-2 non-convergence of the band solve at h/4h seams. |
| `bandCycles` | `4` | Reaches the 0.05 /s residual target on the seep. |
| `crowdedCells` | `merge` | Mass-conserving merge and split of samples in crowded cells, instead of retiring them. Per-sample mass m is capped at 8; an excess sample stays live and is counted as overCap. |

**Opt-in:** `fluid.tiers.fineBlocks` (h stores compacted to the detail ring's blocks), `bulkTier: "16h"`, `coldSurface: "still"`.

**Seep 5 mm, 300 steps, current defaults:**
- **Throat:** u ≈ 2.87–2.89 m/s, Cd ≈ 0.82 (the textbook short square-edged tube).
- **Left chamber:** drains at 0.81 Q_T.
- **Right chamber:** holds 1.28 L at step 300.
- **Drift:** −0.022% with the volume controller off.
- **Health:** no fatal, 0 capped traces, about 110k particles.

**Costs** (paired medians, before the merge default; merge adds ~0.5 ms):

| Scene | Before the package | With it | Main contributor |
|---|---|---|---|
| Rest | 3.2 ms | 8.6 ms | coldSurface warm, 6.3 ms; `still` brings rest back to 3.1 ms |
| Letters | 29.7 ms | 34.3 ms | seedScoring 2.4 ms (state-driven, more particles and h tiles), bandCycles 4 0.9, nativeSmoothing 0.6, substep 0.6 |

Splash and seep each add +0.2–1.2 ms.

**Wide seep (4× domain, 5 mm, pool 834):** about 44 ms a step. 65% of held blocks are submerged COLD bulk, and roughly 4 ms a frame of 4h pressure work runs over it. No exact proof can skip that work; its fix is the 16h tier (lane B).

**Memory:** about 1570 KiB a 4h block at start "4h", after lane B's byte cuts (1773 before the last cut), against a census estimate of about 48 KiB. 497 KiB of that is NB-FLIP's per-block bins (DEPTH_A, DEPTH_B, surfaceHot). 16h members still keep full 4h storage, so 16h saves nothing yet.

## 3. Infrastructure

**Machine.** Apple M1 Max, 32 GiB unified memory, a single GPU shared by every agent.

**Disk.** It has run out before. Keep snapshots few, and delete yours when done.

**Dawn.** Runs need `WEBGPU_NODE_MODULE=/Users/petersuggate/code/me/fluid/node_modules/webgpu/index.js`. Node is `/Users/petersuggate/.nvm/versions/node/v22.22.1/bin/node`, used with `--import tsx`.

**GPU runner** (`tools/gpu/fluid-gpu-run.sh`). Every GPU run goes through it:
- `check <cmd…>`: correctness only (hashes, fatals, counts). Up to 3 run at once, never alongside a timing run.
- `gate <cmd…>`: a merge-gate check. It takes priority over ordinary checks.
- `timing <cmd…>`: exclusive. Use it only for measurements whose milliseconds you will report.
- It shares its locks in `/private/tmp/claude-501` (`FLUID_GPU_LOCK_DIR`) with the Claude sessions. Don't change that directory, and don't call `lockf` on its files yourself.
- At most 2 GPU jobs queued per lane.
- Long runs: detach with `nohup … &` and poll the log. Harnesses can reap foreground shells (exit 144).

**Memory per run.** A seep instance is several GiB. Two or three instances at once is the limit.

**Nondeterminism is by design.** Atomic slot allocation makes repeats diverge after a few frames. Merge gates are therefore "within repeat noise", never bit identity:
- `tools/gate-repeat-spread.ts` scores 10-frame window means.
- Tolerance = max(noise band, 1.66 × the largest A/A window spread).
- Registered scenes (seep, letters) need ≥ 3 repeats; splash and region-tier fixtures may use 2.
- `--set=key:value` applies a param to both arms. `--differ=key` allows a deliberate default change.

**Gate bundle** (`tools/gpu/gate-bundle.sh NAME BASE_REV CAND_REV [args]`):
- What it runs: splash p20 70×2, seep 60×3 and letters 60×3.
- The baseline is a `git archive` of the merge-base on main, never a stale scratch tree; the script builds the snapshots for you.
- `FLUID_GATE_DIR` sets the output directory (default `/private/tmp/fluid-gates`).
- **Physics-neutral change:** the bundle must pass.
- **Deliberate behaviour change (a default flip):** the bundle against main may fail on the metrics the change targets. Report instead:
  1. every scene runs its full length with no fatal (seep 60+ frames);
  2. volume drift with the controller off, change vs main;
  3. paired frame cost per scene (`tools/benchmark-region-tier-paired.ts`);
  4. the gate ratios, for information;
  5. A/A within the new arm, which must pass.

  The coordinator takes that evidence to Peter, or applies the standard he has already set.

**Metal shader cache.** Each process uses its own shard. For a warm-construction timing, set `FLUID_METAL_SHADER_CACHE=<private dir>`.

**Useful probes** (all in `tools/`):
- `probe-tier-budget.ts`: the seep and region-tier probe. Flags `--flowStages`, `--phiStages`, `--retire`, `--retireSplit=256`, `--heatState`, `--census`, `--set=`. Every sample also carries the band (arena) header with its residual history, the `still` words and the discharge block. Lane A's note has the commands; `tools/gpu/probe-readers/*.py` summarise its logs.
- `probe-reservoir-bulk.ts`: 16h bulk-tier probe (`--closed` for rest, member counts).
- `probe-poison-dispatch.ts --audit` with `tools/gpu/perblock-bytes.py`: allocation audit and bytes a block.
- `probe-scale-ladder.ts`: bytes by class, live counts and per-phase GPU ms (`--timestamps`, `--tiers=FILE`, `--width/--depth`). Use `--every=1000` for timing.
- `benchmark-region-tier-paired.ts`: paired A/B timing (`--set`, `--set-candidate`).
- `profile-solver-construction.ts` and `probe-pipeline-compile.ts`: construction time and cold compiles.

## 4. How to work

- **Lane setup.** One long-lived branch per lane, in your own worktree (`git worktree add`). Merge main into it often.
- **Gating.** Gate at milestones or after each large change, not per commit.
- **Big swings over narrow streams.** Each lane owns a whole outcome and lands large changes at milestones. Don't spawn sub-agents.
- **Coordination on shared files.** Before a large edit to `uniform-mixed-frame.ts`, `uniform-pressure-arena.ts`, `uniform-mixed-remap.ts`, `uniform-narrow-band-flip.ts` or `uniform-particle-brick.wgsl.ts`, write a `COORDINATION` entry in your progress file (§6) naming the functions you will touch. Check the other lanes' progress files first.
- **Landing.**
  1. Write a `READY TO MERGE` entry (§6) with the tip SHA, gate or flip evidence, and check results.
  2. The coordinator verifies the claims and merges into main with a `--no-ff` merge commit after `check:types` and `test:unit`.
  3. The coordinator records the verdict in the board above and in your progress file.
  4. Don't merge into main yourself.
- **Decisions.** When you need one from Peter, write a `DECISION NEEDED` entry and continue with other work in your lane until it is answered.

## 5. Lanes

The previous owners' detailed notes are in `docs/plans/tiered-adaptivity-handoff/`:
- `laneA-fidelity.md`: seep physics, probes, open problems.
- `laneB-memory.md`: storage, the arena, the 16h design.
- `laneC-compute.md`: live lists, measured costs, closed verdicts.

Read your lane's note in full before starting.

### Lane A: seep fidelity (the test scene must behave correctly)

**A1. Make `coldSurface: "still"` the default.**
- **The rule:**
  - A surface region may stay COLD only while its per-step omitted transport is within f32 round-off, k·dt²·ε·g·H/h², k = 256.
  - Its accumulated omission since freezing must stay ≤ 0.01 cell, with no residual floor.
  - A free-surface region beside any non-COLD neighbour is WARM. This is the wake-lag fix in `rhSpread`; it was committed ungated on the old lane branch (see the lane A note).
- **Required:** prove the one-region halo is enough. The fastest neighbouring flow must travel less than one region per step; check this against speeds the heat pass already has. If it can be exceeded, widen the halo by ceil(travel / region width), or count and report the breach.
- **Where it stands:**
  - The neighbour rule, a head scaled by the liquid's extent along gravity, and the halo reach bound are all on main (opt-in), but ungated.
  - The reach bound warms every still surface on a frame whose fastest liquid travels ≥ 16 h; there have been 0 breaches so far. Seep is the closest, at about 13 h a step.
  - Without the wake fix, splash and seep passed against warm and rest dropped from 9.4 to 3.1 ms, but **letters failed**.
  - With the neighbour rule letters still diverges at frames 31/32 (h tiles, then particles).
  - Some letters regions sit at about 7e-4 h/step of omission, which is likely real flux. They stay COLD because airborne liquid inflates the head.
- **Next** (lane A note, open problem 1):
  1. Diff the extra frame-31 h tiles tile by tile.
  2. Base the head on liquid connected to the region's column, or the region's own pressure, so those regions warm.
  3. Gate again, using the flip evidence rules in §3.

**A2. Milestone A: seep 5 mm over 20 s (1200 steps), volume controller off.**
- **Pass conditions:**
  - The left level falls within 10% of Torricelli at the measured Cd.
  - The right chamber fills.
  - No fatal.
  - Drift stays small and explained.
  - The fast-jet fixture's throat speed stays bounded near 3.96 m/s.
- Report the overCap and mergedMassRetired counters and explain any mass that leaves through deep or held-heat retirement.
- No 1200-step run exists on the current defaults yet. At 300 steps the left drains at cdInstant 0.81 and the right chamber receives about 80% of the throat flux. The exact command is in the lane A note.
- Open questions from the previous owner, about whether these are correct:
  - should a merged sample crossing the band depth split rather than retire with its whole mass?
  - should clustered deep mass (overCap) spread into neighbouring under-filled cells?

  Bring evidence before changing either.

**A3. The rest scene settles.** Max speed is below 1e-4 m/s with no volume drift (it used to plateau near 5e-3 m/s). Find the source with the stage probes; the fix must be correct, not damping.

**A4. 2.5 mm seep** (23 h a step at the jet), then **Phase 4**: velocity-projected h promotion and residency reach for the fast front, so 2.5 mm and later 1.25 mm run 20 s without a speed or band fatal. Discharge should match 5 mm within the resolution trend.

**A5. Analytic suite.** `nbflip-analytic-correctness` has base failures that predate this work: free-fall, dam-break, mound and tilted-free-fall. Triage whether they are test or solver problems, and report before fixing.

### Lane B: memory follows tiers (16h must save memory and compute)

The lane B note has the design in build order, the arena constraints and the exact commands.

**B0. Cut the 4h block's h-shaped stores.** Start with NB-FLIP's per-block bins (497 KiB a block): replace them with a tile→band-slot map under a band-tile budget, fatal on overflow. Then work down the per-block table in the lane B note. Measure with `tools/probe-poison-dispatch.ts --audit` at two pool sizes and `tools/gpu/perblock-bytes.py`. Target: a 4h block well under 0.5 MiB, heading for the census estimate.

**B1. 16h members leave the 4h work.**
- Member pages leave the level-2 page lists (`ubaLevelCount(2)`) and the 4h launches.
- Each member region gets one coarse row at the 16h level, coupled at its boundary faces, and stays in the 8h/16h/32h hierarchy so the tail keeps it.
- The 10-storage-buffer binding limit applies to arena rows and native pipelines (§ lane B note).
- **Gates:**
  - hydrostatic rest with 16h bulk: max speed ≤ 1e-5 m/s and zero drift;
  - seep discharge and jet with 16h on vs off, within repeat noise. This is the decisive test that the 16h bulk doesn't change the jet;
  - the bundle.

**B2. 16h members drop their 4h block storage.**
- Whole member pages release their block slot onto a 16h-sized slot from a fixed, upfront budget.
- Promotion back to 4h on heat comes from the same budget, and the budget fails fast when exhausted.

**B3. Milestone B: the 4×-widened seep with 16h on.**
- Run `probe-scale-ladder.ts --width=2 --depth=2`, at 5 mm, and at 2.5 mm if it fits.
- Report memory per tier against the census estimate (within 50%) and frame time against today's ~44 ms.
- Expect most of lane C's ~4 ms of bulk pressure work to go.

**B4.** A pool past 4096 blocks (the arena pool A/B split allows ~4470 with solids), if Milestone B or 2.5 mm needs it. Also:
- W7's 4096-slot superBlocks budget;
- particle banks sized by h capacity (W10's companion sizing at `5f890d09` is unfinished, with an unrun fix for "GPU-adopted ownership has no host mirror").

**B5. Flip `fluid.tiers.fineBlocks` (h compaction) to the default** once the bundle and the memory numbers support it.

### Lane C: compute follows tiers

**C1.** Re-measure the defaults' cost after A1 lands: rest, splash, letters, seep and the wide seep, paired. Attribute what remains of coldSurface's cost (before A1 on the wide seep: grid surface advection +1.2 ms, momentum +0.5, classifyCoarse +0.4, velocity advection +0.3). Find exact savings only, i.e. a change that is physics-neutral and passes the bundle.

**C2.** After B1/B2: the wide-seep scale ladder and per-phase split with 16h on. Show frame time following live h and 4h work, not resident bulk.

**C3. Letters cost.** 34 ms against 29.7 ms. Exact savings in the seeding and transfer paths that the extra particles exercise (P2G/G2P, merge, brick deposit).

**C4.** Fix or retire `tools/benchmark-nbflip-letters.ts`. It fails on main at line 578 ("complete exactly one requested clock step").

**C5. Length ladder (Phase 1 gate).** Rest/cold length ×1, ×2, ×4: slope ≤ 0.2 ms per doubling on the current defaults.

## 6. Reporting progress (every lane must do this)

Each lane owns one progress file and nobody else edits it:
- lane A: `docs/plans/tiered-adaptivity-handoff/progress-laneA.md`
- lane B: `docs/plans/tiered-adaptivity-handoff/progress-laneB.md`
- lane C: `docs/plans/tiered-adaptivity-handoff/progress-laneC.md`

**Where to write it:** in the main checkout, so Peter and the coordinator see it without merging your branch. Commit only that file, directly on main, from the main checkout, and touch nothing else there:

```bash
cd /Users/petersuggate/code/me/fluid
# edit docs/plans/tiered-adaptivity-handoff/progress-laneX.md
git commit -m "Lane X progress: <one line>" -- docs/plans/tiered-adaptivity-handoff/progress-laneX.md
```

If the commit fails because the index holds someone else's staged change or a merge in progress, don't touch that state. Retry a few minutes later.

**When to write:**
- at least every 2–3 hours of work;
- when a milestone or gate finishes, pass or fail;
- when you hit a blocker;
- when you need a decision;
- before a large edit to a shared file;
- when your lane goes idle.

**Format:** newest entry at the top, under the `## Log` heading:

```markdown
### 2026-10-26 14:05 · <kind> · tip <sha>
- What changed: …
- Numbers: … (scene, steps, metric, value vs baseline)
- Gates: <bundle name> PASS/FAIL, worst ratio per scene; logs at <path>
- Next: …
```

`<kind>` is one of:
- `PROGRESS`
- `MILESTONE`
- `READY TO MERGE`: give the exact SHA and the evidence §3 asks for.
- `DECISION NEEDED`: state the question, the options and your recommendation.
- `BLOCKED`
- `COORDINATION`: list the files and functions you're about to change.
- `IDLE`

Also keep the `## Status` block at the top of the file current. It holds four fields:
- **Tip**
- **Doing now**
- **Next**
- **Waiting on**: a gate, a decision or another lane.

Write plainly: what you measured and what it means. A failed gate or a fatal is reported as such, with the log path.

**Coordinator side:** Claude reads the progress files, verifies `READY TO MERGE` claims (re-running checks or gates as needed), merges, and updates the §0 board. Verdicts go in the lane's file under a `### <date> · VERIFIED` or `### <date> · REJECTED` entry.
