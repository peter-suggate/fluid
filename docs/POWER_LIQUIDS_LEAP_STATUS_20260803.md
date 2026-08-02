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
2. **The unit of measure is broken.** Fresh baselines (mini 241 ms, tiny-hydro
   352 ms/adv, 26 MGPCG disp/adv) are 6–9× the plan's "today" column (~40 ms,
   1 disp) with no environment recorded in the artifact. Nothing can be scored
   until this is reconciled (ledger §4 C7 agrees).
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
| `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN` | written, argued in-source, **default OFF** | Gate A, one line; narrows one dispatch 10× on large |
| `FLUID_SPGRID_TOUCHED_RADIX_SORT` | multi-WG stable LSD radix sort **exists** (plan's "missing primitive" is stale), Dawn-validated, wired, **default OFF**, ships its own differential tripwire | Gate B (adds storage round-trips); deletes the 6-kernel capacity-shaped directory sweep — the scene-size-tax term |
| `FLUID_COARSE_SUMMARY_INDIRECT_DISPATCH` | written, **default OFF** — but the whole module is **factor-1-only**, unreachable on every matrix lane | deprioritize to the coarse-only track |
| Predicted solve tail | policy written + tested, **no production caller** (`physics-step-program.ts` predicates frozen empty); meanwhile 10 outer iterations are encoded, ~5 execute → ≈388 zero-workgroup dispatches ≈ 31% of mini launches | a wiring job, not a flip |
| Hierarchical executor | `compileHierarchicalExecutor: false` hard-coded; drags 4 flags + the `HYBRID_REGULAR_ROW` fast path as dead code | delete or ticket |
| Air-support compact paths at factor 4 | frontier gate & compact fine demand require `fineFactor === 1` / a fine slot — **they do not engage on the factor-4 matrix lanes**; the still lane marches *more* face items than the churn lane (43,776 vs 31,584) | corridor engagement at factor 4 is the real Bet 1.3 remainder |

Cross-cutting: **no toggle's non-default arm is exercised by any GPU test** —
every "a zero restores the former behaviour" comment is unverified; treat
rollback hatches as untrusted until run once.

## Lane / measurement state

- **Matrix:** tiny-hydro + mini have interleaved A/A pairs (tight, but
  unreconciled — see TL;DR #2). Large-hydro: the air-support/row-capacity fix
  **is committed** (`lib/scenes.ts:78`, authored 4,096-row budget) and went
  green 10+240 steps in-session — **no artifact captured yet**. Large-dam: the
  class-4 producer fix landed (`bb862de`/`3ebffc7`), the old step-249 death is
  survived, and the lane now dies in a **new regime at ~step 413** (alternating
  empty-band/rejected-rebuild; first domino is air-support seeding of the
  settled thin film — physics decision pending: seed from the fine band per
  paper §5, or scope the regime out of the lane).
- **Free greens sitting unbanked:** the 150-step `runtime-150` large-dam lane
  (150 < 249, inside the clean window) has never been run; the large-dam
  benchmark clean window (≤ step 200) can produce the missing churn cell now.
- **Symmetric-expansion 68/spread-0** predates the last 4 commits and the
  entire working tree — re-run before trusting any Gate A claim.
- **Ocean:** still a 2-step frame-capture lane; the plan's 500-step lane is
  unstarted. Ocean is also the *only* scene with a real coarsenable interior.
- The working tree (~88 files) is an **async-pipeline-compilation cutover +
  render work** — zero dispatch-shape changes. Commit it separately so perf
  A/Bs aren't contaminated.

## Ranked next steps

1. **Restore the unit of measure** (blocks everything). Re-capture mini +
   tiny-hydro explaining 241/352 ms vs ~40 ms and 26 vs 1 MGPCG dispatches;
   record the resolved environment in the artifact schema so this class of
   ambiguity can't recur. Then capture large-hydro (now green) and large-dam
   (≤200 steps) → compute the plan's three decision numbers for the first time.
2. **Bank the free greens.** Run `runtime-150`; re-run symmetric-expansion on
   the current tree; score the class-4 fix on the large lane.
3. **Close the Bet 2 loop on the boundary lane** (biggest cheap win): bind
   `powerRowDelta` in `structured-boundary`, publish zero row/slot dispatches
   when the already-computed identity receipt says clean — the same
   `publishExactRowDispatch(…, 0)` move used two files over. Then gate the ×2
   velocity reconstruction the same way.
4. **Flip/wire the finished cutovers, one A/B each:**
   `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN` (Gate A) → predicted-solve-tail wiring
   (~31% of mini launches) → `FLUID_SPGRID_TOUCHED_RADIX_SORT` (Gate B, judge
   on wall not dispatch count, per the plan's own trap #2).
5. **Spine surgery, audit first:** aggregate `PassBrokerBoundaryAudit` across
   the 25 broker instances into the benchmark report (~20 lines) so boundaries
   are ranked by measured cost; then give air-support's indirect buffer the
   STORAGE bit + per-phase arenas (pattern: `fine-levelset-summary-direct.ts`),
   then merge the per-family publication fences into per-phase-group boundaries.
   This is what actually moves 80 → ≤25.
6. **Fine-topology recurring ladders** — the last true O(domain) recurring
   term: shape the 5-dispatch `logicalBrickCount` band ladder from the compact
   seed count `publishRecurringSparseBand` already authors (Gate B).
7. **Give Bet 4.2 a scene that can coarsen.** Author a deep large-hydro variant
   (or promote ocean to a ≥150-step lane) — the current 20× scenes cannot
   express interior coarsening at all; then decide the dam thin-film seeding
   physics question, which is the same "settled shallow regime" territory.

## Plan-doc errata (stale claims in `POWER_LIQUIDS_SCENARIO_LEAP_PLAN.md`)

- §3 B1.2: `FLUID_OCTREE_AIR_SUPPORT_INDIRECT_FRONTIER_GATE` is default-**ON**;
  `FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_DEMAND` **no longer exists** (path
  unconditional).
- §3 B1 status: the "missing radix-sort primitive" exists
  (`lib/webgpu-radix-sort-u32.ts`) and is wired behind a flag.
- The "~94k face patches" corridor figure is retracted by in-tree censuses —
  but corridor *engagement at factor 4* remains open.
- §1 "solve is 1 dispatch": the fresh artifacts record 26/adv — unexplained,
  part of item 1.
- Coarse-summary's 12 domainVolume dispatches are factor-1-only → move to the
  coarse-only/Bet 4.3 track.
- `audit:octree-production-source` now catches capacity-shaped launches it
  previously missed (108 violations / 68 capacity-dispatch), but still skips
  `webgpu-fluid-brick-residency.ts` and `webgpu-sparse-brick-topology-mutation.ts`
  (scope filter) — widen it.
