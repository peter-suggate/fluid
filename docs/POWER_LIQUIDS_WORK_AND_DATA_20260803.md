# Power Liquids — work, data, and where the frame actually goes (2026-08-03)

Session record against `docs/POWER_LIQUIDS_LEAP_STATUS_20260803.md`'s ranked
queue. That doc's item 1 was "restore the unit of measure, it blocks
everything". The unit of measure turned out to be sound; the tree had regressed
3×, and the cause was the Bet-1 scaffolding itself. Everything below follows
from that.

Companion docs updated in the same session: `POWER_LIQUIDS_SCENARIO_LEAP_PLAN.md`
(§2 matrix, errata), `STRUCTURED_CUTOVER_LEDGER.md` (§3 protocol, results),
`BET1_DISPATCH_SHAPE_AUDIT.md` (what the gate still cannot see),
`SYMMETRIC_EXPANSION_ORACLE.md` (68/69 window).

## TL;DR

1. **The baselines were right.** Mini at 240 steps reads **245.3 ms/advance**
   today against **69.6 ms** on 2026-07-29 for the same lane and step count.
   `065219a` (mini `FLUID_MAXIMUM_LEAF_SIZE` 2→32) accounts for 245→211. The
   rest is a real regression, traced to `a56ddd0`.
2. **Bet 1 as executed traded parallelism for launch count.** `a56ddd0`
   correctly removed capacity-shaped *launches* from two kernels and
   accidentally serialized their *work*. Those two kernels are 54% of the mini
   frame.
3. **The wall is anti-correlated with scene size.** The tiny still scene is the
   slowest cell in the four-cell matrix; a scene 20× larger is 2.5× cheaper.
4. **Capacity is not inert — it changes correctness.** The large lane fails at
   t=0 with the planner's default row capacity and is green with the scene's
   authored one. This is the strongest form of the Bet-1 invariant breaking.
5. **Deleting dispatches, copies and bytes did not move the wall or the pass
   count.** Three independent confirmations this session. The program's
   dispatch-shape lens is measuring something that is not currently the cost.

## 1. The four-cell matrix, measured for the first time

All at HEAD `f4d11e7`, Apple M1 Max / Dawn Metal, `environment:
measurement-clean`, artifacts in `artifacts/measurement-floor/`.

|          | small (16³)                     | large (64×20×64)                  |
|----------|---------------------------------|-----------------------------------|
| **still** | **361.6** (`hydrostatic-tiny`, 240) | **143.6** (`large-hydrostatic`, 240) |
| **churn** | **245.3** (`mini`, 240)         | **306.9** (`large`, 200)          |

ms/advance. The mini cell is at 240 steps rather than the lane default 500 so it
is directly comparable to `artifacts/scene-size-overhead/baseline-mini.json`;
the 500-step lane reads ~241 ms.

Ranked: tiny-still > large-churn > small-churn > large-still. **The scene-size
tax is −218 ms.** The plan's framing — that large costs more than small — is
inverted by every pair in the table.

**The obvious confounder is ruled out.** The small lanes are authored at
`FLUID_OCTREE_INTERFACE_BAND: 3` and the large lanes at `1`, so "small vs large"
also meant "band 3 vs band 1". Re-running `hydrostatic-tiny` at band 1:
**360.60 ms vs 361.63 ms** — no difference. Band reach is not what makes the
tiny still lane expensive. (The authored matrix still confounds size with band;
this single paired run is the only thing currently separating them.)

**What does predict the wall** is the persistent MGPCG's executed iteration
count: tiny-hydro 4 → 361.6, mini-dam 4 → 245.3, large-dam 2 → 306.9,
large-hydro 1 → 143.6. Not a law — mini at leaf 2 does 6 iterations in 211 ms
against leaf 32's 4 in 245 ms — but the direction is unambiguous.

## 2. Where the frame goes

Per-pass GPU timestamps, mini lane, 120 steps, label isolation on (the wall
reads ~19% high; the ranking is honest). Summed pass occupancy **224.37
ms/advance** over a **232.73 ms** GPU span, coverage **0.964** — the frame is
GPU-bound and well attributed. This is not a lost host/GPU-overlap story.

| pass | ms/advance | share |
|---|---:|---:|
| `Octree persistent MGPCG - whole solve in one workgroup` | **84.93** | 38% |
| `SPGrid V-cycle - publish validated exact level deltas` | **37.10** | 16% |
| `Advect structured families` | **36.68** | 16% |
| `March Section 5 sparse changed frontier` (×2) | 21.76 | 10% |
| `Advect fine phi rare` | 8.69 | 4% |
| everything else | < 3.1 each | |

Top three = 158.7 ms = **71%**. The whole 2026-07-29 advance was 69.6 ms, so at
least one of these three grew by more than the entire old frame.

## 3. The measurement floor (ledger C7): resolved, differently

- **Step count was not the explanation.** Both captures are 240 steps.
- **`065219a` "fix(octree): restore factor-one scene parity"** changed mini's
  `FLUID_MAXIMUM_LEAF_SIZE` 2→32 and large's 16→32. Every pre/post artifact pair
  in the tree has been compared across different discretizations. Measured at
  HEAD: leaf 32 = 245.3, leaf 2 = 211.1. Worth ~14%.
  It also flips `smallTwoLevelProfile` (`lib/octree-solve-tail-policy.ts:297`),
  doubling the Section 4.3 shell depth k=4→8 and adding one outer iteration.
- **The residual ~3× is `a56ddd0`** — see §4.
- **"26 MGPCG dispatches/advance" is a label-attribution artifact, not a
  regression.** `POWER_DAM_MGPCG_SOLVE_STAGE`
  (`tools/power-dam-performance-report.ts:476-482`) folds `SPGrid V-cycle`,
  `Section 4.3 preconditioner`, `SPGrid accurate A2` and `SPGrid Section 6.3
  apply` into the solve stage, then sums every dispatch in every such pass. The
  solve is still literally one dispatch
  (`lib/webgpu-octree-persistent-mgpcg.ts:292`, `encodedDispatchCount = 1`), and
  `computePassesByLabel["…whole solve in one workgroup"] = 1` in every capture
  including the 07-29 one. The 1→26 step is that `baseline-mini.json` carries no
  `SPGrid V-cycle` pass label at all. **Plan §1's "the solve is one dispatch"
  was true; the status doc's doubt was wrong.**

**The fix that prevents recurrence.** `tools/power-dam-run-environment.ts`:
every artifact now carries an `environment` record — every `FLUID_*`/`WEBGPU_*`
variable the run received, which were *inherited from the shell* rather than
authored, a `contaminants` list of wall-affecting knobs off their
measurement-clean value, a `clean` boolean, and a `comparisonKey` digest over
scene + solver configuration. Printed next to the ms/advance, not buried.
`--leaf-size=` makes the leaf size an explicit A/B axis, since the lane's own
leaf size has already changed under the program once.

## 4. Performance shortfalls

### 4.1 Bet 1 traded parallelism for launch count

`a56ddd0` ("perf(power-liquids): revive the coarse-only tracker and land Bet 1/4
scaffolding") independently caused both top passes:

- **`commitCandidateLevels`** went from `@workgroup_size(64)` data-parallel with
  a capacity-shaped `dispatchFor(...)` to `@workgroup_size(1)` with an indirect,
  live-count-shaped launch — **one thread per level, ~3–5 threads total**, with
  all twelve per-item loops moved inside. The launch-shape conversion was
  correct Bet-1 work. Collapsing the work onto one lane was not.
- **The persistent MGPCG's LDS page-blocked smoother was deleted.** 18 neighbour
  reads per cell per sweep moved from workgroup memory to scattered global at a
  cache-hostile stride, plus a dynamically-indexed `array<f32,18>` that Metal
  spills to scratch — inside a single-workgroup kernel with nothing to hide the
  latency. Diagnosed as **latency, not bandwidth**: one sweep moves ~295 KB and
  the whole solve ~50 MB, i.e. ~0.12 ms at M1 Max bandwidth.

**The lesson generalizes.** Bet 1's invariant is "capacities may size buffers,
never launches". Nothing in it protects occupancy, and
`audit:octree-production-source` scores a `@workgroup_size(1)` indirect launch
as *more* compliant than a wide capacity-shaped one. A launch-shape rule without
an occupancy rule rewards exactly this trade.

### 4.2 Capacity is not inert

| | rows |
|---|---:|
| live at t=0, `large-power-dam-break` | **1,185** |
| scene's authored budget (`lib/scenes.ts:86`) | **8,192** (6.9× headroom, deliberate) |
| `planOctreePressureCapacity` default | **33,536** |
| domain | 81,920 cells |

The default is `max(nx·ny, nx·nz, ny·nz) × bandLayers` plus a closed-wall strip
(`lib/webgpu-octree.ts:1059-1085`) — container geometry, zero fluid input, **28×
the live row count**.

**The lane fails at t=0 with the 33,536 default and is green to 200 steps with
8,192.** Capacity changes correctness, not just footprint. This compounds a known
case where raising this same capacity added ~13 launches while the audit reported
clean.

Two defects, not one:
- (a) the *default* is container-derived and must become fluid-footprint-derived,
  the way `planFluidFootprintFineNarrowBandBrickCapacity` already is;
- (b) the **benchmark lane cannot express the scene's authored capacities at
  all** — it resolves method values from `FLUID_*` env, and
  `globalFineLevelSetMaximumBricks` has no env path
  (`lib/methods/octree.ts:58-62`). The smoke catalog applies
  `largePowerDamOverrides` (`lib/scene-webgpu-smoke-catalog.ts:346-352`); the
  benchmark path does not. That is the whole reason `--lane=large` dies at cold
  topology while `runtime-150` is green on the same scene.

### 4.3 Deleting dispatches, copies and bytes does not move the wall

Three independent confirmations:

- **`FLUID_SPGRID_TOUCHED_RADIX_SORT=1`** — the ledger's Q2, "the biggest
  structural prize". Interleaved A/B, mini, 120 steps, 3 rounds, A/A noise floor
  5.54 ms: **+2.72 ms, inconclusive.** It deletes six dense-directory dispatches
  and does not move the wall. The ON arm also *adds*
  `appendCandidateDirectoryIdentities`, itself `@workgroup_size(1)` — the same
  disease as §4.1.
- **Air-support direct indirect args** — converting the indirect buffer to
  `STORAGE|INDIRECT` and authoring args in place deletes **22 copies and ~384
  B/advance and exactly zero pass closures.** Every staging fence sits on a real
  producer→consumer link in a serial 8-stage pipeline. The status doc's item 5
  premise — that these are "staging boundaries, not algorithmic ones" — does not
  hold for air support.
- **80 compute passes/advance is not staging-bound.** The pass-boundary audit
  (below) shows 33 of 80 closures are staging/copy *causes*, but removing the
  copies moves them into named per-family buckets rather than deleting them. The
  only lever left is dispatching indirectly *within* a pass from a buffer an
  earlier dispatch in that pass wrote — a separate, unproven bet.

### 4.4 The pass-boundary audit, now wired

`PassBrokerBoundaryAudit` was built and connected to nothing; ~33 throwaway
`PassBroker` instances discarded it. It now aggregates process-wide through the
single `boundaryBucket` funnel and reports per advance. On `hydrostatic-tiny`:

```
80.0 closures from 123.0 fence requests (43.0 idempotent — already batched)
61.0 copies · 5.0 clears · 18.0 command encoders
  stage indirect args        19.0 closures / 29.0 copies
  copy buffer                14.0 closures / 32.0 copies
  raw command encoder access  3.0 closures
  … then ~40 named semantic reasons at 1–2 closures each
```

`requests − passClosures` = 43/advance is the "already batched" signal;
`passClosures` is the serial spine. The flat semantic tail is where the real
merges are: single source lines multiplied by velocity classes, hierarchy levels
and transport chunks.

### 4.5 Bets that cannot currently fire

- **The structured-boundary exact row carry.** Landed, handles all three traps
  (advance `control.published`, an identity arm for the in-place workset scan,
  and the persistent solver's per-step workset revalidation). Measured:
  dispatches unchanged at 499.0 on the still lane — **it never fires**, because
  the gate refuses to carry while a live `fine` source is passed and there is no
  bitwise fine-phi receipt to check against. It also costs when declining.
  Unblocking it needs a fine-side delta receipt.
- **The predicted solve tail.** Its premise is dead on this branch: the ~388
  zero-workgroup dispatches belonged to the hierarchical MGPCG, and
  `compileHierarchicalExecutor: false` is hard-coded
  (`lib/webgpu-octree.ts:2790`, test-locked). The persistent kernel runs the
  outer loop inside one workgroup and already breaks early; the `accountZero*`
  counters only *simulate* the old count so an A/B stays comparable. Lowering
  the budget removes no dispatch — it only tightens a fail-closed threshold. It
  is also not per-step mutable: the loop bound is a WGSL literal and the params
  buffer is written once in the constructor.

## 5. What landed

All default-OFF. The default emitted shader is byte-for-byte identical to
`f4d11e7` (verified against `git show HEAD:` for both MGPCG variants).

| flag | reduces | gate | measured |
|---|---|---|---|
| `FLUID_OCTREE_AIR_SUPPORT_DIRECT_INDIRECT_ARGS` | 22 copies, 384 B/advance | A | 0 closures saved |
| `FLUID_STRUCTURED_RECONSTRUCTION_IDENTITY_GATE` | `2×ceil(rows/64)` workgroups/substep | A (provable) | not yet |
| `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN` | capacity→live launch, ~2.3× | A | D4-clean |
| `FLUID_SPGRID_PARALLEL_LEVEL_COMMIT` | *wall only* | A (no float arithmetic) | **−41.0 ms, −16.1%** |
| `FLUID_OCTREE_MGPCG_STAGED_SMOOTHER` | *wall only* | A (term-by-term) | **−9.1 ms, −3.6%** |
| both together | | | **−47.6 ms, −18.7%** |
| `FLUID_STRUCTURED_BOUNDARY_EXACT_ROW_CARRY` | 9 of 11 boundary dispatches | B | never fires |
| `FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS` | 12 loop iterations + adjoint per class-0 band row | B | predicted 1–3 ms on mini |

**Caveat on the −41/−9/−47.6 figures:** only round 1 of the interleaved A/B
completed — the lane went red mid-run when an unrelated scene refactor landed.
Control and control-aa agreed to **0.29 ms** within that round, so the effects
are two orders of magnitude outside the within-round noise, but they are single
samples per arm and want a 3-round median before they are quoted as final.

**The D4 gate holds.** `symmetric-expansion` with
`SPGRID_PARALLEL_LEVEL_COMMIT` + `MGPCG_STAGED_SMOOTHER` +
`STRUCTURED_RECONSTRUCTION_IDENTITY_GATE` + `FINE_TOPOLOGY_INDIRECT_ASSIGN` all
on: volume/velocity/pressure/rhs first diverge at **step 68**, diagonal/topology
at **69** — identical to the control window. The staged smoother's bit-identity
claim is therefore witnessed on a device rather than argued.

Also landed: `deep-power-hydrostatic` (64×48×64, **67.8% interior fraction**
against today's large-hydro 0%, so Bet 4.2's coarsening is expressible at all —
**not yet validated on GPU**); a widened Bet-1 audit (108/59 → 119/61) with a
third vocabulary hole closed (`\w*Capacity` never matched `this.capacity`) and a
new `capacity-indirect-args` rule that catches capacity-shaped launches wearing
an indirect costume (`p.totals.y` at
`lib/webgpu-octree-spgrid-vcycle.ts:3884-3886`); and a real stale-args bug fixed
on air support's topology-reuse path, where a schedule copy moved words that
`prefixAirSupportBlocks` had not recomputed.

## 6. Learnings

1. **Launch count is not the wall.** Three independent confirmations (§4.3). The
   program has been optimizing a proxy. Score work and data directly — the
   pass-boundary audit and the dispatch census now make that possible.
2. **Bet 1 needs an occupancy invariant.** "Capacities may size buffers, never
   launches" is satisfied by a `@workgroup_size(1)` indirect launch, which is
   how 54% of the frame got serialized while the audit stayed green.
3. **A capacity that changes behaviour is a bug class, not a tuning knob.** Test
   for it: a lane that passes at one capacity and fails at another is reporting
   a Bet-1 violation, whichever direction it fails in.
4. **Negative results need to be as loud as positive ones.** The three items in
   §4.3 and the two in §4.5 are the session's most valuable output, and every
   one of them would have been reported as a landed cutover under a
   count-the-dispatches lens.
5. **A number without its environment is not a baseline.** Two artifact sets
   disagreed 3.5× and neither recorded what it ran with. Now they do, and the
   verdict prints next to the number.
6. **Measurement hygiene, two hazards worth knowing.** A crashed GPU run leaves
   `/tmp/fluid-webgpu-exclusive.lock` behind and every subsequent run then fails
   with an unrelated-looking `EEXIST` — twelve A/B arms reported as failures
   from one root cause. And a tree edited under a running benchmark silently
   changes what is being measured: `npx tsc --noEmit` does not catch a module
   that throws at import, so `node --import tsx -e "import('./lib/…')"` belongs
   in the loop.

## 7. Next steps

1. **Fix the mini lane.** It is red with no flags set — `air-support-failure`,
   rejections latching from generation 4, `topologyDesiredBricks: 0`, dying
   inside 20 steps — since the `scene: { definitionId }` refactor replaced
   hand-rolled suite factories with `sceneDocument(...)`. That changes the scene
   being simulated for the eleven suites the refactor names. Blocks the churn
   cell entirely.
2. **Make pressure row capacity fluid-footprint-derived** (§4.2a) and **make the
   benchmark lane consume the scene's authored method profile** (§4.2b). Largest
   data-size item in the tree, and it currently gates lane correctness.
3. **Run the work/data census** on a green lane with the §5 keepers on vs off:
   dispatches, pass closures, copy bytes, launched workgroups. Predictions are
   pinned so a wrong result is visible — air support should read `stage indirect
   args` 19→5 closures and 29→7 copies with **total closures unchanged at 80**.
   Then flip defaults for whatever passes with the D4 window still ≥68.
4. **Re-run the parallelism A/B for a 3-round median** once mini is green, and
   score both flags on `--lane=large`, where the serialization should be worse.
5. **Validate `deep-power-hydrostatic` on GPU** — 1-step cold gate, then a
   20-step sizing run, then the authored 240-step lane. Predicted terminal
   counters are recorded with the scene so the first run checks against a
   prediction rather than accepting whatever appears.
6. **Run the band-row `census` mode before the routing A/B.** It publishes the
   class-0 share of band rows and the coarse share. If `coarseRegularBandRows`
   is ~0 on mini, do not spend GPU minutes there — the change should pay on
   `large`/`deep-hydrostatic` where the octree has real coarse rows.
7. **Occupancy sweep.** `@workgroup_size(1)` kernels found so far:
   `appendCandidateDirectoryIdentities`
   (`lib/webgpu-octree-spgrid-vcycle.ts:4392`) and, until this session,
   `commitCandidateLevels`. Worth an audit rule of its own.

### Chores

- One line in `tools/webgpu-smoke-executor.ts` beside the `power-hybrid-census`
  block to print `readPersistentBandCensus()`; it is published and reachable but
  nothing logs it.
- `octreePersistentMGPCGWGSL` has never been in `tools/validate-water-shaders.ts`
  — the persistent MGPCG is not in the naga gate at all.
- `tests/large-power-hydrostatic.test.ts:93` expects `9,512,436` against an
  actual `9,512,524`; an air-support arena layout change landed without it.
- Pre-existing reds from `f4d11e7`/`bb862de`, unrelated to this work:
  `octree-power-hybrid-shipping-contract`, `webgpu-synchronous-pipeline-caches`,
  `webgpu-large-family-pipeline-cache`.

## 8. Corrections to claims made during this session

Recorded because each was asserted with confidence and then refuted by evidence:

- "The capacity planner `webgpu-octree-owner-pages.ts:190-222` halved
  `maximumResidentBricks`" — **false**. That module plans *owner* pages. 32,768
  is an authored scene budget, `LARGE_POWER_DAM_FINE_BRICK_CAPACITY`
  (`lib/scenes.ts:73`).
- `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN` is "10× narrower" — **~2.3×** on the
  current tree (227 vs 512 workgroups against 14,474 live pages).
- "Six recurring launches consume the candidate schedules" — **four**;
  `rankCandidateBricks` and `compactCandidatePages` are level-count shaped.
- "Restore the deleted LDS smoother" — **wrong remedy**. The old smoother used a
  sequential running subtraction (different summation tree from
  `canonical18Sum`) *and* staged from a frozen pre-sweep snapshot, so cross-page
  neighbours were stale for sweeps 2..N. It was a different iteration, not just
  different rounding. Restoring it would have moved the D4 window.
- "`FLUID_PRESSURE_ROW_CAPACITY=8192` papers over a domain-sized live row count"
  — **inverted**. 8,192 is the fluid-footprint budget; 33,536 is the
  container-derived default it replaces.
- "Air support's staging copies are staging boundaries, not algorithmic ones"
  (status doc item 5) — **false for air support**; all 13 sit on real
  producer→consumer links.
