# Bet 1 dispatch-shape audit — "existence is free"

Status: audit, 2026-08-03. Branch `perf/structured-cutover`. Static analysis only
(no GPU; the exclusive device lock was held elsewhere). Read-only on `lib/`.

This is the audit step `POWER_LIQUIDS_SCENARIO_LEAP_PLAN.md` §3 Bet 1 item 1 asks
for and that nobody had done. Anchor on **symbol names**; line numbers are from
the working tree of 2026-08-03 and `lib/webgpu-octree*.ts` moves daily.

## 0. The rule and the scoring convention

> Every recurring dispatch must be shaped by a GPU-published **live count**
> (compacted worklist + indirect args), or be a `(1,1,1)` control singleton.
> Allocation-time capacities may size *buffers*, never *launches*.

Classes used below:

| class | meaning | verdict |
|---|---|---|
| **A** | `dispatchWorkgroupsIndirect` off a GPU-published record whose value is a live count | compliant |
| **B** | literal `(1)` / `(1,1,1)` / a small fixed constant control singleton | compliant |
| **C** | shaped from an allocation-time capacity (`rowCapacity`, `maximumResidentBricks`, `brickCount`, `pageDirectoryWords`, `totalBrickCount`, arena byte lengths) | **violation** |
| **D** | shaped from `domainVolume`, a domain edge, cross-section, or `logicalBrickCount` | **violation** |
| **E** | shaped from a host JS number tracking fluid | violation if it came from a readback |

Two refinements the plan's rule does not distinguish, and which this audit found
to matter more than the raw A/B/C/D split:

- **A-over-C / A-over-D.** A dispatch can be syntactically indirect (class A) and
  still be shaped by a capacity, because the *GPU* wrote `ceil(capacity/64)` into
  the indirect record. The letter of the rule passes; the identity
  (`O(interface area) + O(live rows)`) fails. Marked **A(C)** / **A(D)**.
- **Launch *count* vs launch *width*.** Most known violations are wide launches
  with the same dispatch count on every lane. A separate and previously
  unrecorded family shapes the *number of encoded dispatches* from a capacity or
  from `log2(domain edge)`. Those are what actually move `dispatches/advance`
  between lanes.

## 1. Classification of the recurring per-advance path

### 1.1 What the recurring path is

Per substep, from `lib/webgpu-uniform-eulerian.ts:1877` (`for substep`):

1. `WebGPUOctreeProjection.encodeReadyTopologyFlip` (`lib/webgpu-octree.ts:3731`)
2. `.encodeSurface` (`lib/webgpu-octree.ts:4125`)
3. `.encode` (`lib/webgpu-octree.ts:3909`)
4. `.encodeInactiveTopologyCandidate` (`lib/webgpu-octree.ts:3423`)

and once per advance, outside the substep loop,
`.encodeSparseBrickWorld` (`lib/webgpu-octree.ts:6010`).

`this.hostAllocation` gates the dense uniform-Eulerian kernels; the octree lanes
never take them.

### 1.2 Module-level census (default env, `globalFineLevelSetFactor = 4`, no rigid bodies)

| module | A | A(C)/A(D) | B | C | D | per substep | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `webgpu-octree-air-velocity-support-gpu.ts` | 168 | 0 | 24 | 0 | 0 | **192** | 96 per encode × 2 encode sites (`webgpu-octree.ts:3790` `"topology-commit"`, `:4020` `"settled-fine"`) |
| `webgpu-octree-fine-levelset-topology.ts` (`encode`) | 14 | 0 | 7 | **15** | **5** | ~41 | the single largest violation block |
| `webgpu-octree-fine-levelset-topology.ts` (`encodeFinalizePublication`) | 2 | 0 | 1 | 0 | 0 | 3 | |
| `webgpu-octree-fine-levelset-redistance.ts` | 32 | 0 | 4 | 0 | 0 | 36 | warm/recurring JFA; 6 strides on both dam lanes |
| `webgpu-octree-fine-levelset-transport.ts` | 13 | 0 | 6 | 0 | 0 | 19 | |
| `webgpu-octree-fine-levelset-summary-direct.ts` | 12 + `maximumLevel` | 0 | 4 | 0 | 0 | 16 + L | **L is domain-shaped** (§1.4) |
| `webgpu-octree-fine-levelset-volume.ts` | 2 | 0 | 6 | 0 | 0 | 8 | cadence-gated by `FLUID_FINE_VOLUME_CADENCE` (default 1) |
| `webgpu-octree-fine-seed-adapter.ts` | 2 | 0 | 1 | 0 | 0 | 2–3 | **count depends on `rowCapacity <= 4096`** (§1.4) |
| `webgpu-octree-fine-to-coarse-levelset.ts` | 2 | 0 | 4 | 0 | 0 | 6 | ×2 (`encodeCoarsePhiCorrection` runs twice) |
| `webgpu-octree-power-coarse-levelset.ts` | 6 | 0 | 4 | 0 | 0 | 10 | ×2; the 8-sweep coarse redistance never fires on fine-band lanes |
| `webgpu-octree-spgrid-vcycle.ts` (`encodeCandidateSetup` + `encodeReadySetupCommit`) | 21 | **4** | 5 | 0 | 0 | **30** | `encodedSetupDispatchCount`, `:1824` |
| `webgpu-octree-power-descriptor.ts` | 4 | 0 | 3 | 0 | 0 | 7 | |
| `webgpu-octree-power-topology.ts` | 4 | 0 | 3 | 0 | 0 | 7 | |
| `webgpu-octree-structured-velocity-gpu.ts` | 6 | 0 | 4 | 0 | 0 | 10 | |
| `webgpu-octree-structured-dynamics.ts` | 10 | 0 | 3 | 0 | 0 | 13 | one of the 10 is dead work (§1.5) |
| `webgpu-octree-structured-boundary.ts` | 9 | 0 | 3 | 0 | 0 | 12 | |
| `webgpu-octree-persistent-mgpcg.ts` | 0 | 0 | 1 | 0 | 0 | **1** | whole solve, `dispatchShape = [1,1,1]` |
| `webgpu-octree-topology-epoch.ts` | 1 | 0 | 3 | 0 | 0 | 4 | |
| `webgpu-octree-owner-pages.ts` | 0 | 0 | 3 | 0 | 0 | 3 | |
| `webgpu-octree-solid-vertex-sdf.ts` | 1 | 0 | 2 | 0 | 0 | 3 | |
| `webgpu-fluid-brick-residency.ts` (`encodeFineSeedCandidates`) | 0 | 0 | 1 | **3** | 0 | 4 | dense arm on both dam lanes |
| `webgpu-octree.ts` (topology/frontier chain) | ~30 | 0 | ~10 | 0 | 0 | ~40 | **frontier sort ladder count is capacity-shaped** (§1.4) |
| `webgpu-octree-coarse-summary.ts` | — | — | — | (4) | (12) | **0** | unreachable on both dam lanes (§3c) |

Nothing in the audited set is class **E**. Every `mapAsync` on the recurring path
(`webgpu-octree-power-descriptor.ts` `readCandidateFailure`,
`webgpu-octree-structured-dynamics.ts` `censusTick`,
`webgpu-fluid-brick-residency.ts` `readStats`,
`webgpu-octree-fine-levelset-transport.ts` `censusTick`) terminates in host
diagnostics and never feeds a dispatch argument. `hostSchedulingUsesReadback:
false` holds.

`lib/webgpu-octree.ts` itself is clean: all 39 dispatch sites are either
`dispatchWorkgroups(1)` or indirect off `solveDispatch` / `coldDispatch` /
`topologyCandidateDispatch` / `structured.source.liveRowDispatch`.

### 1.3 The width violations (same dispatch count on every lane, wrong width)

**V1 — fine-topology capacity/domain identity ladder.**
`WebGPUFineLevelSetTopology.encode`, `lib/webgpu-octree-fine-levelset-topology.ts`,
recurring (`publication.kind === "delta"`) branch. All go through the direct
helper `runIdentity` (`:1348`, body `pass.dispatchWorkgroups(x, y)`).

Class **C**, shaped by `plan.maximumResidentBricks`:
`:1464` `classifyIdentity` `Math.ceil(plan.maximumResidentBricks / 64)`;
`:1466`/`:1467`/`:1470`/`:1471` and again `:1496`/`:1497`/`:1500`/`:1501` and
again `:1514`/`:1515`/`:1517`/`:1518` (`identityBlocks` /
`identitySuperBlocks` = `pageDeltaLayout.identityScanBlockWords` =
`ceil(maximumResidentBricks/256)`, `:1462`);
`:1474` `compactIdentity`; `:1477` `assignIdentity`. **15 dispatches.**

Class **D**, shaped by `plan.logicalBrickCount` (the *whole domain's* fine brick
lattice, `webgpu-octree.ts:933` `logicalBrickCount = x * y * z`):
`:1437` `const recurringBlocks = Math.ceil(plan.logicalBrickCount / 256)`, used at
`:1439` `scanRecurringDesired`, `:1442` `scanSparseGroups`, `:1447`/`:1449`
`offsetSparse*`, `:1454` `scatterRecurringSparseBand`. **5 dispatches.**

Live counts already on the GPU that could shape these: `this.identityDispatch`
(published by `finalizeIdentityPipeline`, `:1480`) already drives `assignIdentity`
and `classifyPageDelta` when `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN=1`;
`this.haloDispatch` (published by `publishRecurringSparseBandPipeline`, `:1425`)
already carries the compact seed count that `scatterRecurringSparseBand` wants.

**V2 — SPGrid dense brick/page directory sweep (A(C)/A(D)).**
`prepareCandidateSchedules` (`lib/webgpu-octree-spgrid-vcycle.ts:3877`) publishes
the whole candidate chain's indirect records — genuinely live for every phase
except:

```
lib/webgpu-octree-spgrid-vcycle.ts:3912
 if(topologyLevelItems!=0u){brickItems=p.totals.y;logicalPageItems=p.totals.z;
  physicalPageItems=p.totals.z;}
```

`p.totals.y` is `totalBrickCount` (`:3563`), `p.totals.z` the page-directory
cardinality. Four dispatches consume those records on any dirty epoch:
`markCandidateBrickOccupancy` (`:2005`), `scatterCandidateRankedSlots` (`:2009`),
`markCandidatePageOccupancy` (`:2011`), `linkCandidatePageNeighbours` (`:2015`).
The in-file comment at `:3909` already admits it. Clean epochs publish zero, so
this is *change-gated but not size-gated*.

**V3 — brick-residency fine-seed candidate scan.**
`GPUFluidBrickResidency.encodeFineSeedCandidates`,
`lib/webgpu-fluid-brick-residency.ts:1454`, dense arm on both dam lanes:
`publish.dispatchWorkgroups(dispatches[index]!)` for prepare/mark/resolve, where
`dispatches` (`:1452`) derives from `worklistByteLength`, `tileWorklistByteLength`,
`currentAllocationPlan.stateBytes`, `publicationCapacity` — all allocation-time.
3 class-C dispatches per substep. `GPUFluidBrickResidency.encode` (5 C + 1 B)
is bootstrap-only (`webgpu-octree.ts:3337`) and is fine.

### 1.4 The count violations (these are what move `dispatches/advance` between lanes)

These are new to this audit; the plan's residue list does not contain them.

**V4 — distributed frontier merge-sort ladder is sized from `pressureRowCapacity`.**

```
lib/webgpu-octree.ts:1703  this.useLocalFrontierCandidateSort = this.frontierAllocation.listCapacity <= 4096;
lib/webgpu-octree.ts:1994  const frontierSortStageCount = this.useLocalFrontierCandidateSort
                             ? 0 : Math.ceil(Math.log2(Math.max(1, this.frontierAllocation.listCapacity))) + 1;
lib/webgpu-octree.ts:3639  for (const group of this.frontierSortGroups) { … dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0); }
```

`listCapacity = min(cellCount, rowCapacity)`
(`planOctreeLeafFrontierAllocation`, `:716`). Below 4096 rows the sort is **one**
`(1,1,1)` dispatch; above it, `ceil(log2(rowCapacity)) + 1`. Each dispatch *is*
indirect (class A by letter), but the encoded **count** is a pure function of an
allocation-time capacity — the archetypal Bet-1 violation, and it is currently
invisible because everyone measures the width, not the count.

Concretely, with the lane profiles as authored today:

| lane | `listCapacity` | local sort? | frontier-sort dispatches |
|---|---:|---|---:|
| mini / tiny-hydro | 3,072 | yes | **1** |
| large (authored `pressureRowCapacity: 8_192`) | 8,192 | no | **14** |

**V5 — fine summary pyramid depth is domain-shaped.**
`lib/webgpu-octree-fine-levelset-summary-direct.ts:416`
`for (let level = 1; level <= this.plan.maximumLevel; level += 1)` — one indirect
dispatch per pyramid level, and `maximumLevel = levelOffsets.length - 1` is
`log2` of the fine brick lattice edge. mini = 4, large = 6.

**V6 — fine-seed adapter branch is capacity-shaped.**
`lib/webgpu-octree-fine-seed-adapter.ts:456` `encode` takes the compact
2-singleton arm when `this.plan.rowCapacity <= OCTREE_FINE_SEED_PERSISTENT_ROW_CAPACITY`
(4096, `:35`) and the 3-dispatch planner+indirect arm otherwise. mini 2, large 3.

### 1.5 Incidental findings on the recurring path

- **Dead dispatch.** `lib/webgpu-octree-structured-dynamics.ts:837`
  (`transferStructuredTopologyCandidate`) reads `liveRowDispatch` word 18
  (`OCTREE_STRUCTURED_TOPOLOGY_TRANSFER_DISPATCH_OFFSET_BYTES = 72`). The only
  writer of word 18 in the tree is `finalizeStructuredPublication`
  (`lib/webgpu-octree-structured-velocity-gpu.ts:852`), which assigns `0u` in
  **both** arms. As written it dispatches zero workgroups every substep while
  still costing a bind group and (with the compact pass off) a pass boundary.
  Worth confirming on-device before deleting.
- **Dead entry points.** `OctreeSparseBrickWorld.encode`
  (`lib/webgpu-octree-sparse-bricks.ts:1461`) has no caller anywhere;
  `WebGPUStructuredBoundaryCoefficients.encode` / `encodeAcceptedCandidate`
  (`lib/webgpu-octree-structured-boundary.ts:408`/`:420`) are unreachable, so the
  "Prepare structured boundary accepted recurring transaction" label never runs.
- **The in-tree capacity-dispatch lint is green and blind.**
  `npm run audit:octree-production-source` reports *27 violations in 59 sources,
  none of them `capacity-dispatch`* — while V1 above is 20 capacity/domain-shaped
  dispatches per substep. Two independent holes in
  `lib/webgpu-octree-work-accounting.ts`:
  1. `CAPACITY_AUTHORITY` (`:1045`) is
     `…|maximumResidentBricks|logical(?:Cell|Brick|Page|Domain)|dims\.n[xyz])\b`.
     The trailing `\b` after the alternation means `logicalBrickCount`,
     `totalBrickCount`, `brickCount` and `pageDirectoryWords` **do not match**
     (verified: only `plan.maximumResidentBricks` of that list tests true).
  2. `capacityDispatchViolations` (`:1055`) only inspects the literal argument
     text of `dispatchWorkgroups(` and follows `const/let/var` aliases *in the
     same file*. Capacity passed as an argument into a helper —
     `runIdentity(pipeline, entries, Math.ceil(plan.maximumResidentBricks / 64), …)`
     with the helper body `pass.dispatchWorkgroups(x, y)` — is invisible.

  Both are one-line fixes and should land before any Bet-1 work, or the gate will
  keep certifying violations as clean.

## 2. Per-lane accounting: where 470 vs 442 comes from

### 2.1 Lane geometry (computed from the pure planners; no GPU)

`tools/power-dam-lane-environment.ts`: mini/tiny-hydro `16,16,16` band 3 factor 4
leaf 32; large `64,20,64` band 1 factor 4 leaf 32.

| quantity | mini | tiny-hydro | large |
|---|---:|---:|---:|
| domain cells | 4,096 | 4,096 | 81,920 |
| `pressureRowCapacity` (derived) | 3,072 | 4,096 | 3,072 |
| `pressureRowCapacity` (authored override, `lib/scenes.ts:68`) | — | — | **8,192** |
| `logicalBrickCount` | 4,096 | 4,096 | **81,920** |
| `maximumResidentBricks` (derived) | 4,096 | 4,096 | 6,600 |
| `globalFineLevelSetMaximumBricks` override (`lib/scenes.ts:69`, constant at `:56`) | — | — | 32,768 |
| `recurringBlocks = ceil(logicalBrickCount/256)` | 16 | 16 | **320** |
| `classifyIdentity` groups `= ceil(maximumResidentBricks/64)` | 64 | 64 | 512 |
| SPGrid `levelCount` / `brickCount` / page-directory words | 5 / 75 / 21 | 5 / 75 / 21 | 7 / **1,511** / **461** |
| fine summary `maximumLevel` | 4 | 4 | 6 |
| warm JFA strides | 6 | 6 | 6 |
| frontier sort dispatches (today) | 1 | 1 | **14** |

**tiny-hydro is dispatch-identical to mini** (same lattice, same band, same
factor; only `rowCapacity` differs, 3,072 vs 4,096, both under the 4,096 sort
threshold). So the tiny-hydro cell of the matrix cannot show a Bet-1 win — it is
a **Bet 2** instrument, not a Bet 1 one. The measured artifacts agree:
`artifacts/scene-size-overhead/fresh-20260802-mini-a.json` = 503.0 dispatches,
`…-hydrostatic-tiny-a.json` = 499.0. Four dispatches separate a churning dam from
still water.

### 2.2 The 470-vs-442 number is stale, and stale in the direction that flatters us

From `artifacts/scene-size-overhead/`:

| artifact | lane | captured | disp/adv | indirect | passes |
|---|---|---|---:|---:|---:|
| `baseline-mini.json` | mini | 2026-07-29 23:56 | 442.08 | 252 | 80.03 |
| `large-after.json` | large | 2026-07-30 00:20 | 470.08 | 281 | 80.03 |
| `recheck-persistent-{1,2}.json` | large | 2026-07-30 00:22 | 470.17 | 281 | 80.05 |
| `fresh-20260802-mini-{a,b}.json` | mini | 2026-08-02 10:00 | **503.02** | 328 | 80.00 |
| `fresh-20260802-hydrostatic-tiny-{a,b}.json` | tiny-hydro | 2026-08-02 10:02 | **499.03** | 328 | 80.00 |

The plan's §1 table quotes mini = 442, which is **61 dispatches stale**; the
current mini figure is 503. There is no post-2026-07-30 large baseline (blocked on
the class-4 red), so the "470 vs 442" pair is internally consistent but both
halves predate the current tree.

Worse: `pressureRowCapacity: 8_192` on `LARGE_POWER_DAM_METHOD_PROFILE` landed on
**2026-08-02** (`git log -S "pressureRowCapacity: 8_192" -- lib/scenes.ts` →
`a56ddd0`, the same commit that landed the Bet-1 scaffolding). At the time 470 was
measured, `listCapacity` at large was 3,072 and the frontier sort was the local
1-dispatch arm. **Today the large lane crosses the 4,096 threshold and pays +13
launches that were not in the 470 figure.** A Bet-1 capacity-planning change has
itself introduced a capacity-shaped launch count.

Diffing `computePassesByLabel` between the mini and large artifacts yields
**zero differences** — 80.025 passes, identical label multiset. That is expected,
not informative: `PassBroker.compute()` silently drops the label when a pass is
already open unless `FLUID_GPU_ISOLATE_PASS_LABELS=1`
(`lib/webgpu-pass-broker.ts:116`, and its own doc comment at `:2`). Pass labels
therefore cannot attribute dispatches in the captured artifacts.

### 2.3 Static attribution of the gap

| contributor | mini | large (as measured 2026-07-30) | large (today) | Δ today |
|---|---:|---:|---:|---:|
| V4 frontier sort ladder | 1 | 1 | 14 | **+13** |
| V5 fine summary pyramid levels | 4 | 6 | 6 | **+2** |
| V6 fine-seed adapter arm | 2 | 2 | 3 | **+1** |
| everything else audited | identical | identical | identical | 0 |

So **+2 of the historical +28 is attributable statically**, and **+16 is
attributable for the current tree**. The residual ~26 of the 2026-07-30 gap is not
explained by any encode-time loop whose trip count depends on scene geometry that
I could find: `refinementSizes` / `coarseRefinementSizes` / `balanceRounds`
(`lib/webgpu-octree.ts:1617`–`:1625`) depend only on `maxLeafSize` (32 on both);
JFA strides and frontier stages are 6 and 7 on both; air-support frontier waves
are the fixed 12; SPGrid `encodedSetupDispatchCount` (`:1824`) is level-independent
at 30; `encodeCorrectionBody`'s per-level ladder is behind
`assertHierarchicalExecutorCompiled` and production passes
`compileHierarchicalExecutor: false` (`lib/webgpu-octree.ts:2775`).

**The measurement that closes this.** Re-run the A/A pair with
`FLUID_GPU_ISOLATE_PASS_LABELS=1`. With isolation on, every labelled pass brackets
exactly its own dispatch, so `computePassesByLabel` **becomes** dispatches-by-label
and a mini↔large diff attributes the whole gap by kernel in one run per lane. The
broker's own note measures the cost of isolation at 39.07 vs 39.09 ms/advance on
mini, i.e. free for a counting run. Do this before spending any effort on the
remaining ~26; it is one benchmark invocation per lane and it removes all guessing.

Second-priority measurement: capture a fresh `large` baseline at all (P0.4's two
missing cells). The whole §1 table of the plan rests on a pair that is now four
days and one capacity-policy change old.

## 3. The residue list, corrected

The plan's status block (§3 Bet 1, "Status 2026-08-02") lists four items and
asserts they are all blocked on one missing radix sort. Item by item:

**(a) SPGrid dense brick/page directory sweep — CONFIRMED, but NOT blocked.**
The four kernels and their `p.totals.y` / `p.totals.z` shaping are exactly as
described (§1.3 V2). What the plan does not say is that **the entire compact
replacement is already written, wired, and tested-adjacent in-tree** — see §4. It
is behind `FLUID_SPGRID_TOUCHED_RADIX_SORT` (default off).

**(b) fine-topology `maximumResidentBricks` / `logicalBrickCount` scans —
CONFIRMED and understated.** It is 20 dispatches per substep (15 C + 5 D), not a
handful, and it is the largest violation block on the recurring path. It is also
**not blocked on a sort**: `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN`
(`lib/webgpu-octree-fine-levelset-topology.ts:1689`, `=== "1"`, default **off**)
already converts `assignIdentity` from `ceil(maximumResidentBricks/64)` to an
indirect launch off `this.identityDispatch`, with the correctness argument written
out at `:1483`–`:1488`. That is a finished, unshipped, zero-new-machinery fix.

**(c) 12 `domainVolume` dispatches in `webgpu-octree-coarse-summary.ts` — count
right, symbol wrong, lane wrong.** The 12 dispatches exist
(`lib/webgpu-octree-coarse-summary.ts:255`–`:277`) but are shaped by
`this.air.layout.ownerDirectoryCellCapacity`, not by a symbol named
`domainVolume`; the host-side `domainVolume` occurrences in that file
(`:81`, `:96`, `:127`, `:303`) are allocation and readback only. More importantly
**the module is unreachable on every fine-band lane**: `coarseOnlySummary` is
constructed only inside `if (this.coarseOnlySurfaceTracking)`
(`lib/webgpu-octree.ts:2844`) and
`coarseOnlySurfaceTracking = options.globalFineLevelSetFactor === 1`
(`lib/webgpu-octree.ts:1636`). Both dam lanes and both hydrostatic lanes run
factor 4. **Optimising this file buys exactly zero on the matrix.** It should be
struck from the Bet-1 residue and moved to the coarse-only track.

**(d) grow-on-reject capacity reallocation — the remainder is smaller than
stated.** Bet 1 item 4's *derivation* half has landed and works:
`planFluidFootprintFineNarrowBandBrickCapacity` (`lib/webgpu-octree.ts:885`) is
the production default (`:2194`), and `planOctreePressureCapacity` (`:1040`) takes
`Math.min(sceneShapedRequest, footprintRequest)` (`:1089`). Measured effect: the
81,920-cell large domain plans **3,072** pressure rows and **6,600** fine bricks
from the authored 1,500-cell reservoir — genuinely fluid-shaped. What remains is
only the grow-on-reject reallocation path, and it does not exist yet. Note the
new hazard: the *authored overrides* (`pressureRowCapacity: 8_192`,
`globalFineLevelSetMaximumBricks: 32_768`, `lib/scenes.ts:68`/`:69`) now exceed
the derived plan by 2.7× / 5× and, via V4, buy 13 extra launches.

**(e) "the four `ceil(domainVolume/256)` air-support dispatches are gone and
test-locked" — CONFIRMED.** No dispatch in
`lib/webgpu-octree-air-velocity-support-gpu.ts` is shaped by anything but an
indirect record or the literals `1` and `3`. All 11 `domainVolume` occurrences are
allocation fields or WGSL bounds checks. Locked by
`tests/webgpu-octree-air-velocity-support-gpu.test.ts:746`
(`assert.doesNotMatch(encode, /maximumResidentBricks|domainVolume\/OCTREE_AIR_SUPPORT_GPU_WORKGROUP_SIZE|dilateFineBandAirSupportDemand/)`)
and `tests/octree-air-support-compact-authority.test.ts:26`–`:27`, `:108`. Caveat:
the lock is a regex over `encode.toString()`, so it catches the removed spellings,
not the concept — a `this.domainGroups` precomputed in the constructor would pass.

**Also confirmed:** `FLUID_OCTREE_AIR_SUPPORT_INDIRECT_FRONTIER_GATE` is
default-**ON** (`lib/webgpu-octree-air-velocity-support-gpu.ts:343`,
`!== "0"`), as the task suspected; the plan's §3 Bet 1 item 2 text calling it
"currently default-off" is stale. It additionally requires
`changedFrontier && fineSlot !== undefined && fineFactor === 1` (`:1089`–`:1092`),
so on the factor-4 dam lanes it does not engage at all.

### 3.1 What the plan's residue list misses

- **V4 / V5 / V6** (§1.4) — the three places where the encoded *dispatch count*,
  not width, is shaped by capacity or `log2(domain)`. V4 is the single largest
  scene-size term in today's tree.
- **V3** — three class-C dispatches per substep in
  `webgpu-fluid-brick-residency.ts:1454`.
- **The lint hole** (§1.5) — the gate that is supposed to prevent all of this
  cannot see `logicalBrickCount` and cannot see capacity passed through a helper.
- **Two dead lanes and one dead dispatch** (§1.5) — free launches to delete.
- **Two dead env levers.** `FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_DEMAND` and
  `FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_CELLS` are still exported and
  default-tested, but `encode` hard-codes `compactFineDemand = true` (`:1088`) and
  `compactFineCells = fineSlot !== undefined` (`:1093`). Setting either to `0`
  changes nothing on the production path. The plan's item 2 asks to "finish the
  family"; the family is finished and the switches are vestigial.

## 4. The radix-sort verdict

**What is implemented.** `lib/webgpu-radix-sort-u32.ts` is a complete
**multi-workgroup, stable, atomic-free LSD radix sort** over u32 identities: four
8-bit passes of `countRadixDigits` → `scanRadixDigits` → `scatterRadixDigits`,
plus `prepareRadixSort` (the fail-closed header validator that authors the one
indirect record), `compactSortedRuns` (emits `(unique value, first sorted index)`
pairs — exactly the touched-directory form), and `publishRadixSort`.

- **Multi-workgroup:** yes. `countRadixDigits` and `scatterRadixDigits` are
  `@workgroup_size(256)` launched via `dispatchWorkgroupsIndirect(this.dispatch, 0)`
  with `dispatch[0] = min(65535, ceil(count/256))` and Y-folding
  (`foldedBlock`, `:195`). Only `prepare`, `scan`, `compactSortedRuns` and
  `publish` are `(1,1,1)`.
- **Stable:** yes, and argued in-source. `scanRadixDigits` rewrites per-block
  digit counts into digit-major exclusive bases *in block order*; a lane's rank
  counts only earlier same-digit elements *of its own block*
  (`for(var t=0u;t<lane;t+=1u)`, `:217`). No atomics anywhere — asserted by test.
- **Bet-1 compliant itself:** every recurring launch is indirect off the
  GPU-published live count; capacity sizes only buffers. Test-locked at
  `tests/webgpu-radix-sort-u32.test.ts:25`–`:38`.
- **Tested:** yes, well. CPU-oracle equivalence against `stableRadixSortU32` /
  `buildSPGridTouchedDirectory` in `lib/octree-spgrid-touched-directory.ts`
  (`tests/webgpu-radix-sort-u32.test.ts:56`), plus a Dawn test
  (`:105`, gated on `WEBGPU_NODE_MODULE`) covering 40k full-range keys,
  duplicate-heavy runs, every block boundary `{1,2,255,256,257,4095,4096,4097}`,
  the empty publication, and three fail-closed rejections (count past capacity,
  wrong magic, zero epoch) with the producer's bank-0 words asserted untouched.

**So the plan's premise is wrong in both directions.** The blocking primitive is
not missing — it exists, it is correct, it is multi-workgroup and stable, and it
is tested. And it is not the blocker for most of the residue.

**What is actually blocking each residue item:**

| item | needs the sort? | actual blocker |
|---|---|---|
| (a) SPGrid directory sweep | it uses it, and the consumer is **already written** | validation + a soundness question (below), not authorship |
| (b) fine-topology identity ladder | **no** | `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN` is written and default-off; the remaining 19 need `haloDispatch`/`identityDispatch` plumbing, which is prefix-sum work, not sorting |
| (c) coarse-summary | **no** | unreachable on every lane in the matrix; delete from the list |
| (d) capacity planner | **no** | derivation landed; only grow-on-reject remains, which is a host allocation policy |
| V4 frontier sort ladder | **no** | either raise the 4,096 threshold, or make `frontierSortStageCount` follow the GPU-published live candidate count instead of `listCapacity` |
| V5 summary pyramid depth | **no** | the pyramid is genuinely `log2(domain)` deep; the fix is to skip levels with zero live keys via an indirect-zeroed record, not to sort |
| V6 fine-seed adapter arm | **no** | pick one arm; the compact arm is already live-count-only |

**The SPGrid cutover is further along than the plan says — and has one thing to
check before it is enabled.** `encodeSetupCandidate`
(`lib/webgpu-octree-spgrid-vcycle.ts:2002`–`:2030`) already swaps the four dense
kernels for `appendCandidateDirectoryIdentities` → `brickSort.encode` →
`markCompactCandidateBrickOccupancy` → `rankCompactCandidateBricks` →
`pageSort.encode` → `buildCompactCandidatePages` →
`linkCompactCandidatePageNeighbours`, plus `commitCandidateTouchedBricks`
(`:2054`). A dual-run oracle exists: `FLUID_SPGRID_TOUCHED_RADIX_TRIPWIRE=1`
(`:1173`) keeps the dense kernels running and makes each compact kernel assert
bit-equality against them, reporting `0x2000` / `0x4000` / `0x10000` / `0x20000`
on mismatch (`:4508`, `:4521`, `:4541`, `:4557`). `tools/webgpu-smoke-executor.ts:3359`
already reads the tripwire. That is a Gate-A-ready A/B.

Three caveats before flipping it:

1. **Launch-width soundness of the two compact consumers.** The radix
   `liveDispatch` publishes `ceil(count/256)` workgroups, where `count` is the
   *pre-sort key count*. `markCompactCandidateBrickOccupancy` (`:4498`) and
   `linkCompactCandidatePageNeighbours` (`:4548`) are `@workgroup_size(64)` and
   index `run < control.runCount` via
   `boundedLinearIndex(g) = g.x + g.y*65535*64` (`:3507`). Total threads is
   therefore `64·ceil(count/256) ≈ count/4`, so coverage requires
   `runCount ≤ count/4` — i.e. an average run length of at least 4. For bricks
   (7 ordinals × up to 64 cells per brick) that is comfortable. For **pages** it
   is not obviously true: `touchedPageKeys` gets one key per occupied brick and a
   page spans 2×2×1 bricks, so a sparse band where most pages hold one occupied
   brick gives `runCount == count` and roughly three quarters of the pages would
   never get their 27 neighbour links written. This is a static reading; confirm
   on-device with the tripwire before trusting the cutover. If real, the fix is a
   second live record shaped from `runCount` rather than reusing `liveDispatch`.
2. **Dispatch-count regression.** `encodedSetupDispatchCount`
   (`lib/webgpu-octree-spgrid-vcycle.ts:1824`) goes from 30 to **62** with the
   cutover on: `−6` dense kernels, `+6` compact ones, `+30` for the two sorts
   (15 each). The trade is 4 wide domain-shaped launches for 30 narrow live ones.
   Per §5 of the plan ("dispatch-count deletion alone does not move the wall"),
   that is acceptable *for the ocean scale it unblocks*, and probably a small
   regression on mini. A/B both.
3. **Three of the compact kernels are `@workgroup_size(1)` serial walks.**
   `appendCandidateDirectoryIdentities` (`:4392`) is one lane looping
   levels × selected slots × 7 ordinals; `rankCompactCandidateBricks` (`:4512`)
   and `buildCompactCandidatePages` (`:4534`) are one lane looping over all runs
   for every level. They trade O(domain) parallel width for O(live) serial depth
   — exactly the shape Bet 3 item 3 says to give the classify→prefix→scatter
   treatment. Enabling the cutover without widening these may be a net loss on
   small scenes even though it is the only thing that scales to ocean.

## 5. Ranked work list

Cost is dispatches/substep unless noted. Gate A = launch shape and pass structure
only. Gate B = anything that adds or removes a storage round-trip between kernels
(plan §5: a round-trip is a rounding step, so it is never Gate A).

| # | violation | cost today | scales with | fix | gate |
|---|---|---|---|---|---|
| **1** | Lint holes in `webgpu-octree-work-accounting.ts` (`CAPACITY_AUTHORITY` misses `logicalBrickCount`/`totalBrickCount`/`brickCount`/`pageDirectoryWords`; `capacityDispatchViolations` cannot see capacity passed into a helper) | 0 (meta) | — | extend the regex; resolve one level of helper-parameter binding, or forbid non-literal dispatch arguments in audited modules | **not a gate** — tooling; do it first or every later landing re-certifies itself green |
| **2** | **V4** frontier merge-sort ladder count from `listCapacity` (`webgpu-octree.ts:1994`, `:3639`) | **+13 at large**, 0 at mini | `log2(rowCapacity)`; steps at 4,096 | shape `frontierSortStageCount` from the GPU-published live candidate count (`topologyCandidateDispatch` already carries it) and zero the unused stages via the indirect record, so a 1,500-cell reservoir encodes the local-arm cost regardless of the authored budget | **Gate A** (launch shape only; stages already indirect) |
| **3** | **V1** fine-topology `assignIdentity` capacity launch (`fine-levelset-topology.ts:1477`) | 1 | `maximumResidentBricks` | flip `FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN` to default-on after an A/B — the indirect arm is already written (`:1482`–`:1490`) with its correctness argument | **Gate A** (same record, same round-trips) |
| **4** | **V1** fine-topology `logicalBrickCount` recurring band ladder (`:1437`–`:1455`) | 5, width `ceil(81,920/256)=320` at large vs 16 at mini | **O(domain volume)** | shape from `haloDispatch` (the compact seed count `publishRecurringSparseBand` already authors) instead of the dense logical lattice; the scan/offset pair becomes a live-count prefix | **Gate B** — replacing a dense scan with a compacted one changes what each lane accumulates and therefore the reduction order |
| **5** | **V1** fine-topology identity scan/offset/compact ladder (14 remaining C dispatches) | 14, width `ceil(maximumResidentBricks/64)` | capacity (32,768 authored at large) | drive all of them from `identityDispatch`; this is the same prefix-sum machinery as #3 repeated over three ladders | **Gate B** for the scan reductions (order changes), **Gate A** for the pure `classify`/`compact` sweeps that only mask |
| **6** | **V3** brick-residency fine-seed candidate scans (`webgpu-fluid-brick-residency.ts:1454`) | 3 | arena byte lengths | publish a live worklist count from `publishFineSeedCandidateResidency`; or take the sparse-pool arm on the dam lanes (`fineSeedCandidatesOnly` + a real budget) | **Gate A** if the sparse arm is bit-identical, else Gate B |
| **7** | **V2** SPGrid dense brick/page directory sweep (`spgrid-vcycle.ts:3912`) | 4, width `ceil(1,511/64)` + `ceil(461/64)` at large; `ceil(1,511×…)` at ocean | `O(domain)` per **dirty** epoch | enable `FLUID_SPGRID_TOUCHED_RADIX_SORT` after (i) resolving the page-launch coverage question in §4 caveat 1, (ii) an A/B on dispatch count (30→62), (iii) widening the three `@workgroup_size(1)` compact kernels | **Gate A** by construction (the tripwire asserts bit-equal directory contents), *provided* caveat 1 is a non-issue; if the launch has to change shape it stays Gate A, but any re-ordering of the ranked-slot write becomes Gate B |
| **8** | **V5** fine summary pyramid depth (`summary-direct.ts:416`) | +2 at large | `log2(domain edge)` | give each level an indirect record that is zero when the level has no live keys; the pyramid stays `log2` deep but costs nothing above the live band | **Gate A** (indirect-zeroing only) |
| **9** | **V6** fine-seed adapter arm (`fine-seed-adapter.ts:456`) | +1 at large | `rowCapacity` crossing 4,096 | keep only the compact arm — it is already live-count-only and needs no planner pass | **Gate A** |
| **10** | Dead work: `structured-dynamics.ts:837` zero-workgroup dispatch; `sparse-bricks.ts:1461` and `structured-boundary.ts:408`/`:420` dead entry points; the two vestigial air-support env levers | ~1 dispatch + 2 bind groups | — | delete after confirming word 18 really is always 0 on device | **Gate A** |
| **11** | Bet 1 item 4 remainder: grow-on-reject reallocation | 0 today | — | reallocate at a generation boundary on the existing fail-closed overflow receipt, so the authored overrides in `lib/scenes.ts` can come *down* toward the derived plan (which would also retire #2 and #9) | **Gate B** — capacity changes the arena layout the kernels address |
| — | ~~coarse-summary `domainVolume` dispatches~~ | **0 on every matrix lane** | — | strike from Bet 1; move to the coarse-only track | — |

### Ordering note

#1 and #2 are the highest value per unit of risk: one is tooling that stops the
program from lying to itself, the other is the single largest scene-size term in
today's tree and is pure launch shape. #3 is a flag flip on already-written,
already-argued code. #4 and #5 together remove `O(domain volume)` from the
recurring path and are the real prize, but they are Gate B and need the
`symmetric-expansion` bitwise-D4 window (≥ 68) plus the mini 500-step envelope.

Before any of it: run the two counting lanes with `FLUID_GPU_ISOLATE_PASS_LABELS=1`
(§2.3) so the remaining ~26 unattributed dispatches stop being a guess, and
capture a fresh `large` baseline so the program stops planning against
2026-07-30 numbers.
