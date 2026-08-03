# Power Liquids — scenario leap: status & handoff (2026-08-03)

Fresh whole-program audit of `docs/POWER_LIQUIDS_SCENARIO_LEAP_PLAN.md` against the
tree at `97a6fa7` + working tree. Detail lives in `BET1_DISPATCH_SHAPE_AUDIT.md`
and `STRUCTURED_CUTOVER_LEDGER.md`; this doc is the compact verdict and the queue.

## TL;DR

The goal identity (`wall ≈ O(interface changed) + O(live rows) + small constant`)
and the four-cell matrix are still the right program. The implementation is
**ahead of the plan text** (air support, corridor, hybrid apply, quiescence,
exact carries: landed, default-on) but **behind on cutover and measurement**:

1. **Three finished fast paths sit default-OFF**, one policy is written but
   unwired, and no toggle's non-default arm has ever been run on a GPU.
2. **The unit of measure was broken; it is now half-fixed and half-open.**
   *Resolved:* the "26 MGPCG dispatches/advance" is a **label-attribution
   artifact**, not 26 solve dispatches — `POWER_DAM_MGPCG_SOLVE_STAGE`
   (`tools/power-dam-performance-report.ts:476-482`) folds `SPGrid V-cycle`,
   `Section 4.3 preconditioner`, `SPGrid accurate A2` and `SPGrid Section 6.3
   apply` into the solve stage and then counts every dispatch inside those
   passes. The persistent solve is still literally **one** dispatch
   (`WebGPUOctreePersistentMGPCG.encodedDispatchCount = 1`,
   `lib/webgpu-octree-persistent-mgpcg.ts:292-293`; artifacts record
   `computePassesByLabel["Octree persistent MGPCG - whole solve in one
   workgroup"] = 1`). The other ~25 are the SPGrid candidate rebuild, which now
   opens two labelled passes (`… · capture plan L1 delta`, `… · candidate commit
   changed L1`) where the 07-29 capture opened none — hence 1 → 26 with no
   change in solve work. *Also resolved:* the missing environment. Artifacts now
   carry a resolved-`environment` record (`tools/power-dam-run-environment.ts`)
   with `clean`, `contaminants` and a `comparisonKey`.
   *Still open:* the wall. `FLUID_MAXIMUM_LEAF_SIZE` went 2→32 on mini and 16→32
   on large in `065219a` (08-01), so no pre/post artifact pair is comparable.
   Measured at HEAD on the same 240-step mini lane: **245.3 ms/adv at leaf 32**
   vs **211.1 at leaf 2**, against **69.6** for the 07-29 leaf-2 capture. Leaf
   size explains ~14%; a **real ~3× regression remains unexplained** and is the
   live item (ledger §4 C7).
3. **The wall's structure didn't move where we said it would**: passes/adv is
   flat at 80 (target ≤25) because the indirect-authorship conversions traded
   `copyBufferToBuffer` for `broker.fence()` — fewer bytes, same barriers. And a
   still step dispatches within 4 of a churning step (499 vs 503) — Bet 2's
   thesis, measured, still unbanked.

## Scoreboard

| Bet | Verdict | One-line evidence |
|---|---|---|
| **1 — existence is free** | **~80% landed, default-on** | Air-support domainVolume dispatches gone (test-locked); compact demand/cells flags *deleted* (unconditional); indirect frontier gate default-ON; corridor + unreached-face tripwire landed; capacity planner fluid-footprint-derived; arena diet done (36×→2.5× rows). Residue: fine-topology recurring ladders (~22 capacity-shaped dispatches — now the **largest O(domain) term left**), SPGrid dirty-epoch directory sweep (replacement written, flagged off), `listCapacity` frontier sort ladder (+13 launches bought by the 8k row override). |
| **2 — change is the only work** | **carries landed; the loop isn't closed** | Descriptor/topology/row-geometry exact carries: landed, default-on, zero-workgroup on clean steps. Transport quiescence: landed, default-on (global bit, not per-brick). **Holes:** structured boundary has *zero* delta machinery yet computes an exact identity proof (`acceptedBoundary.pad`) that **nothing reads**; velocity reconstruction runs full-width ×2/substep unconditionally; result: still ≈ churn in dispatch count. |
| **3 — shorten the spine** | **partial; thesis not yet cashed** | Direct indirect-authorship landed in 7 modules (copy bytes −22%). But passes flat at 80: converted modules kept one fence per record family (12 in fine-topology alone); air support still stages **13 copies into a non-STORAGE indirect buffer, encoded twice per advance** (~16–18 of the 80); the fence-attribution audit is built (`PassBrokerBoundaryAudit`) but wired to nothing, and 25 throwaway `new PassBroker` instances discard it. Persistent maintenance kernel (3.2): not started. |
| **4 — hybrid discretization** | **Stage 1 shipped; Stage 2 + coarsening blocked on scenes** | 5-class GPU row classification + class-dispatch apply: default-on, with 6 full-power oracles flagged off; f64 CPU differential harness green (bit-identity, seam symmetry, D4×48). Not done: `applyRegularRow` still loads stored coefficients & walks `pageSlot` (explicit "second cut"); classification producer still does the full 19-channel power walk for every row; **the authored large scenes are geometrically incapable of interior coarsening** (1-cell-deep hydro slab → 1.004 rows/cell; dam interior ≈5 cells deep), so Bet 4.2's cost model is untestable on the current lanes. |

## The cutover gap (the direct answer to "have we cut over fully?")

Mostly yes on the big families — and the hatches that matter were *deleted*, not
defaulted. What has **not** cut over:

| Item | State | Flip cost |
|---|---|---|
| `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN` | written, argued in-source, **default OFF** | Gate A, one line; narrows one dispatch **~2.3–4×** on large, *not* 10× — see below |
| `FLUID_SPGRID_TOUCHED_RADIX_SORT` | multi-WG stable LSD radix sort **exists** (plan's "missing primitive" is stale), Dawn-validated, wired, **default OFF**, ships its own differential tripwire | Gate B (adds storage round-trips); deletes the 6-kernel capacity-shaped directory sweep — the scene-size-tax term |
| `FLUID_COARSE_SUMMARY_INDIRECT_DISPATCH` | written, **default OFF** — the whole module is **factor-1-only**, unreachable on every matrix lane (guard chain below) | deprioritize to the coarse-only track |
| Predicted solve tail | policy written + tested, **no production caller** — and **its premise is dead on this branch**: there are no encoded outer dispatches left to zero (below) | not a launch win; a semantics change worth zero measured ms at HEAD |
| Hierarchical executor | `compileHierarchicalExecutor: false` hard-coded; drags 4 flags + the `HYBRID_REGULAR_ROW` fast path as dead code | delete or ticket |
| Air-support compact paths at factor 4 | frontier gate & compact fine demand require `fineFactor === 1` / a fine slot — **they do not engage on the factor-4 matrix lanes**; the still lane marches *more* face items than the churn lane (43,776 vs 31,584) | corridor engagement at factor 4 is the real Bet 1.3 remainder |

Cross-cutting: **no toggle's non-default arm is exercised by any GPU test** —
every "a zero restores the former behaviour" comment is unverified; treat
rollback hatches as untrusted until run once.

## Lane / measurement state

- **Matrix: FILLED, 2026-08-03.** All four cells captured at HEAD `f4d11e7`,
  artifacts in `artifacts/measurement-floor/`, each with a resolved
  `environment` record. ms/advance: tiny-still **361.6**, large-still **143.6**,
  small-churn **245.3**, large-churn **306.9**. **The wall is anti-correlated
  with scene size** — the 16³ still scene is the slowest cell in the matrix, and
  the "scene-size tax" the plan wanted comes out **−218 ms**. Band 3 vs band 1
  is ruled out as the cause (tiny-hydro at band 1: 360.6 vs 361.63). What tracks
  the wall is the persistent solve's executed iteration count (4/4/2/1 for
  tiny-still/mini/large-churn/large-still) and that solve's own pass measures
  **84.9 ms/advance = 38% of the mini frame**. Full table, caveats and the
  consequence for Bet 1 are in the plan's §2.
- **Large-dam death regimes, disambiguated.** The `runtime-150` smoke lane is
  **green**: 150 steps, 0 tripwires, volume drift 0, pressure residual 1.3e-5,
  all diagnostics passed. The **benchmark** lane, once given
  `FLUID_PRESSURE_ROW_CAPACITY=8192` (see plan §5 — the lane table is missing
  the scene's authored capacities, which is what makes the un-patched lane die
  at t=0), is clean through **200 steps**, 0 validation errors, all gates pass.
  The old "dies at ~step 413" figure describes **neither** of those: it is the
  post-`bb862de`/`3ebffc7` class-4-fix configuration run past 200 steps
  (alternating empty-band/rejected-rebuild; first domino is air-support seeding
  of the settled thin film). Name the configuration whenever quoting it — the
  physics decision it implies (seed from the fine band per paper §5, or scope
  the regime out of the lane) is still pending, but it does not gate the
  ≤200-step measurement window.
- **Symmetric-expansion re-run on the current tree, and it held.** Volume,
  velocity, pressure and rhs first lose bitwise D4 at **step 68** (the
  contract); diagonal/topology at **step 69**, one step *later* than the oracle
  doc recorded. `docs/SYMMETRIC_EXPANSION_ORACLE.md` is updated. **The 250-step
  run exits non-zero and that is the expected outcome** — the gate is the
  divergence step, not a passing process.
- **Ocean:** still a 2-step frame-capture lane; the plan's 500-step lane is
  unstarted. Ocean is also the *only* scene with a real coarsenable interior.
- The working tree (~88 files) is an **async-pipeline-compilation cutover +
  render work** — zero dispatch-shape changes. Commit it separately so perf
  A/Bs aren't contaminated.

## Ranked next steps

0. **NEW, and it displaces the ordering below: attack the solve constant.**
   The filled matrix says the wall tracks the persistent MGPCG's executed
   iteration count, and that solve's single pass is 84.9 ms/advance — 38% of the
   mini frame — because it runs the entire outer loop inside **one workgroup**.
   Bets 1–3 are all launch-structure work, and launch structure is now measured
   flat across the whole matrix (80 passes / ~500 dispatches on every lane while
   the wall moves 143.6 → 361.6). Perfecting `wall ≈ O(interface changed) +
   O(live rows) + small constant` does not help while the constant is 85 ms.
   First questions, cheapest first: what is the per-iteration cost as a function
   of live rows (the mini leaf-2 run does **6** iterations in **211 ms** while
   leaf-32 does **4** in **245 ms**, so per-iteration cost is not constant); and
   what is the row-count threshold above which the hierarchical/parallel solve
   should be selected instead? The persistent-vs-hierarchical selector is the
   MGPCG playbook the plan's Bet 3.2 already describes — it was applied in one
   direction only.
1. **Finish restoring the unit of measure.** The 26-vs-1 MGPCG discrepancy is
   explained (attribution, TL;DR #2), the environment record has landed, and the
   four cells are captured. What is left is the wall itself: bisect the ~3× mini
   regression that survives the leaf-32 correction (245.3 leaf-32 / 211.1 leaf-2
   at HEAD vs 69.6 at 07-29). Use `--leaf-size=` to hold the discretization
   fixed and `comparisonKey` to reject mismatched pairs. Re-capture the
   large-churn cell **clean** once the lane table carries the scene's authored
   capacities — 306.9 ms currently comes from a `clean: false` run.
2. **Bank the remaining free greens.** `runtime-150` and symmetric-expansion are
   now run and green (above). Still unbanked: scoring the class-4 fix on the
   large lane past step 200, and the `deep-hydrostatic` Bet-4.2 lane.
3. **Close the Bet 2 loop on the boundary lane** (biggest cheap win): bind
   `powerRowDelta` in `structured-boundary`, publish zero row/slot dispatches
   when the already-computed identity receipt says clean — the same
   `publishExactRowDispatch(…, 0)` move used two files over. Then gate the ×2
   velocity reconstruction the same way.
4. **Flip the finished cutovers, one A/B each:**
   `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN` (Gate A) → then
   `FLUID_SPGRID_TOUCHED_RADIX_SORT` (Gate B, judge on wall not dispatch count,
   per the plan's own trap #2). **The predicted solve tail is removed from this
   queue** — see below; it is not a launch win on this branch.

### The predicted-solve-tail item, honestly

The "~31% of mini launches / ≈388 zero-workgroup dispatches" prize belongs to
the **hierarchical** MGPCG, which no longer exists here. `webgpu-octree.ts:2790`
hard-codes `compileHierarchicalExecutor: false` on the only production
`WebGPUOctreeSPGridVCycle` construction, and that is test-locked at
`tests/gpu-initialization.test.ts:153`.

The persistent kernel runs the **whole outer loop inside one workgroup**, with
`storageBarrier()`/`workgroupBarrier()` where dispatch boundaries used to be
(`…persistent-mgpcg.wgsl.ts:1236`), and it already breaks early on convergence
(`:1257`, `:1274` — `halt = workgroupUniformLoad(&wHalt); if(halt!=0u){break;}`).
The `accountZeroAll`/`accountZeroRemaining` counters at `:1004-1011` say so in
their own comment: they exist *only* to keep the published
`zeroedDispatches` word identical to a hierarchical run so a lockstep A/B still
compares. **They simulate dispatches that are not encoded.**

So lowering `encodedOuterIterations` removes no dispatch and shortens no
converged solve. On the persistent path it can only move `p.shape.x`, the
non-convergence threshold at `:1066-1068` — i.e. it *tightens a fail-closed
bound*. Two further facts make even that unreachable today: the loop trip count
is a WGSL literal baked at shader build (`octreePersistentMGPCGWGSL`'s
`maximumIterations`, `…persistent-mgpcg.wgsl.ts:136`), and the params buffer is
written exactly once, in the constructor
(`…persistent-mgpcg.ts:495`), from `this.solveTailPolicy.encodedOuterIterations`
fixed at `webgpu-octree.ts:1607`. A per-step selection cannot take effect
without reconstructing the solver.

**Rewritten queue item:** *not* a ~31% launch win — a semantics change with
**zero measured win at HEAD**. If anyone revives it, two blockers come first:
`selectOctreeFactorOneEncodedSolveTail` refuses unless `factorOne`
(`globalFineLevelSetFactor === 1`, while every shipping mini/dam lane runs
factor 4), and it refuses on non-adjacent history
(`observation.step + 1 !== nextStep`) — but the browser polls `readStats` on a
250 ms cadence (`lib/webgpu-renderer.ts:2042`), so the selector would always
return `non-adjacent-history`. Keep the selector and its tests as the written
spec; do not budget GPU minutes against it.
5. **Spine surgery — the audit now reports, and it names its own target.**
   `PassBrokerBoundaryAudit` is aggregated into the benchmark artifact
   (`summary.passBoundaries`, schemaVersion 1, `exact: true`). On
   `hydrostatic-tiny`, 240 measured advances, warm-up excluded:
   **80.0 pass closures/advance** out of **123.0 fence requests** (43.0
   idempotent), **61 copy commands**, 5 clears, 36 KB/advance, across 18 command
   encoders. Ranked by closures/advance:

   | reason | closures/adv | copies/adv | share |
   |---|---:|---:|---:|
   | `stage indirect args` | 19.0 | 29 | 23.7% |
   | `copy buffer` | 14.0 | 32 | 17.5% |
   | `raw command encoder access` | 3.0 | 0 | 3.7% |
   | `structured indirect arguments published` | 3.0 | 0 | 3.7% |
   | *(remainder: per-family publication fences)* | ~41 | 0 | ~51% |

   **33 of the 80 closures — 41% — are staging/copy boundaries, not algorithmic
   ones.** That is direct measured support for 5b below: give the indirect
   buffers the STORAGE bit + per-phase arenas (pattern:
   `fine-levelset-summary-direct.ts`) so args are authored by kernels instead of
   staged by copies, then merge the per-family publication fences into
   per-phase-group boundaries. This is what moves 80 → ≤25. Note the ordering
   caveat from item 0: at 84.9 ms of solve in a ~360 ms advance, removing
   barriers is worth doing but is not where the wall is.
6. **Fine-topology recurring ladders** — the last true O(domain) recurring
   term: shape the 5-dispatch `logicalBrickCount` band ladder from the compact
   seed count `publishRecurringSparseBand` already authors (Gate B).
7. **Give Bet 4.2 a scene that can coarsen.** Author a deep large-hydro variant
   (or promote ocean to a ≥150-step lane) — the current 20× scenes cannot
   express interior coarsening at all; then decide the dam thin-film seeding
   physics question, which is the same "settled shallow regime" territory.

## Plan-doc errata — **APPLIED 2026-08-03** to `POWER_LIQUIDS_SCENARIO_LEAP_PLAN.md`

All of the following are now corrected in the plan itself. Kept here as the
audit trail, with the two items that were *themselves* wrong flagged.

- §3 B1.2: `FLUID_OCTREE_AIR_SUPPORT_INDIRECT_FRONTIER_GATE` is default-**ON**
  (`…air-velocity-support-gpu.ts:346`);
  `FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_DEMAND` **no longer exists** (path
  unconditional, gated on `fineSlot` alone).
- §3 B1 status: the "missing radix-sort primitive" exists
  (`lib/webgpu-radix-sort-u32.ts:23`) and is wired behind
  `FLUID_SPGRID_TOUCHED_RADIX_SORT`.
- The "~94k face patches" corridor figure is retracted by in-tree censuses
  (31,584 on mini; 43,776 on the *still* tiny-hydro lane) — corridor
  *engagement at factor 4* remains open.
- §1 "solve is 1 dispatch" is **TRUE and was wrongly doubted**. The fresh
  artifacts' "26/adv" is the label-attribution metric described in TL;DR #2, not
  a solve dispatch count. *Recommended (not done — `tools/power-dam-performance-
  report.ts` is owned elsewhere): rename or split `mgpcgDispatchesPerAdvance`,
  e.g. `pressureStageDispatchesPerAdvance` plus a separate
  `persistentSolveDispatchesPerAdvance`, so it stops reading as the solve.*
- Coarse-summary's 12 `domainVolume` dispatches are factor-1-only → coarse-only
  / Bet 4.3 track. Guard chain, confirmed unreachable on every matrix lane:
  `coarseOnlySurfaceTracking = globalFineLevelSetFactor === 1`
  (`webgpu-octree.ts:1638`) → `if (this.coarseOnlySurfaceTracking)` (`:2819`) →
  the only `new WebGPUOctreeCoarseSummary` (`:2825`); all eleven other
  `coarseOnlySummary` references are optional-chained, and the factor default is
  4 (`lib/methods/octree.ts:7`, `:108`; `webgpu-octree.ts:2155`).
- §1 fine-topology scans are shaped by **`maximumResidentBricks`**, not
  `logicalBrickCount` (`…fine-levelset-topology.ts:1476-1477`,
  `Math.ceil(plan.maximumResidentBricks / 64)`).
- §1's air-support encode sites: **four** call sites today
  (`webgpu-octree.ts:3774`, `:4004`, `:4342`, `:4380`), two per steady-state
  advance — the artifacts carry every Section 5 label twice, suffixed
  `- topology-commit` and `- settled-fine`.
- §2 P0.4's `FLUID_OCTREE_ROW_DELTA_CENSUS` does not exist (removed in
  `9de199a`); §2's P0.2 lane key is `--lane=large-hydrostatic`; §3 B1's commit
  credit is `a56ddd0`, not `a0a2247` (rebased).
- §1/§2 walls: the leaf-size discontinuity and the surviving ~3× — TL;DR #2.

### Errata that were themselves wrong

- **"The `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN` dispatch is 10× narrower."** It
  is `Math.ceil(plan.maximumResidentBricks / 64)`, and the *capacity* moved. At
  07-30 the large lane's `fineBrickCapacity` was 81,920 with 8,126 desired fine
  bricks (`recheck-persistent-1.json` `terminalCounters`) → 1,280 vs 127
  workgroups, so 10× was right *then*. The large lane now carries an **authored**
  budget — `LARGE_POWER_DAM_FINE_BRICK_CAPACITY = 32_768` (`lib/scenes.ts:73`,
  applied `:86` and `lib/scene-webgpu-smoke-catalog.ts:351`, consumed at
  `webgpu-octree.ts:2199`) → 512 workgroups, against 227 for the 14,474 desired
  pages in `artifacts/scene-size-overhead/large-current/traced.log`. So **~2.3×
  at that state and ~4× at the sparser one — never 10× again.** (And the
  mechanism is the authored budget, *not* the owner-page planner in
  `lib/webgpu-octree-owner-pages.ts`; the fine-band capacity comes from
  `planFluidFootprintFineNarrowBandBrickCapacity`, `webgpu-octree.ts:881-907`.)
- **"The predicted solve tail is worth ~31% of mini launches."** Dead premise —
  see "The predicted-solve-tail item, honestly" above.

### Audit gate

`audit:octree-production-source` now scans **61** sources and reports **119**
violations (was 108 in 59). Three changes landed 2026-08-03:

- Scope widened to `lib/webgpu-fluid-brick-residency.ts` (**6** capacity-shaped
  launches) and `lib/webgpu-sparse-brick-topology-mutation.ts` (**clean**).
- New rule `capacity-indirect-args` closes the `dispatchWorkgroupsIndirect`
  exemption: **4** hits, all real —
  `webgpu-octree-spgrid-vcycle.ts:3884-3886` (`prepareCandidateSchedules`
  publishing `plan.brickCount` / `pageDirectoryWords` into the candidate
  schedule) and `webgpu-octree-coarse-summary.ts:638` (the
  `ceil(domainVolume/256)` record 0).
- Capacity vocabulary now matches the bare lowercase `this.capacity`, which had
  hidden 4 of brick-residency's 6.

This is the gate's **third** measured blindness. A green result from it has
twice been quoted as evidence while wrong; `docs/BET1_DISPATCH_SHAPE_AUDIT.md`
now carries a "what the gate still cannot see" list. Read it first.
