# Power Liquids — the two-scenario leap plan

Status: handoff proposal, 2026-08-02. Branch `perf/structured-cutover`.
Companions: `POWER_LIQUIDS_10X_DISCOVERY_RESULTS.md` (evidence ledger),
`POWER_LIQUIDS_ULTIMATE_M1MAX.md` (incremental catalogue + refuted-ideas list),
`POWER_LIQUIDS_TEMPORAL_COHERENCE_HANDOFF.md`, `POWER_LIQUIDS_COARSE_ONLY_PLAN.md`,
`papers/aanjaneya-2017-power-liquids.txt`.

Anchor on **symbol names**, not line numbers — this tree moves fast.

## Mission

GPU utilization is poor and the sim is too slow, and incremental kernel work has
stopped moving the wall. This plan restructures the attack around two scenario
lenses that factor the cost into orthogonal axes:

1. **`large-power-dam-break`** ("20× dam break", exists: 64×20×64, water block
   is 1.8% of the domain — `lib/scenes.ts` `createLargePowerDamBreakScene`) —
   isolates every cost that scales with **scene size** while fluid is tiny.
2. **`hydrostatic-power-two-level`** ("tiny hydrostatic", exists: 16³, fill
   0.75, `sceneId: tiny-hydrostatic-two-level`) — isolates every cost that is
   paid **when nothing changes**.

A solver that is fast on both — and still fast on `benchmark:power-dam-mini`
churn — has the property we actually want:

> **wall(advance) ≈ O(interface area changed this step) + O(live liquid rows in
> the solve) + a small constant. Never O(domain volume), never O(capacity),
> and near-zero when the state is steady.**

Every bet below is scored against that identity. The bets deliberately include
departures from Aanjaneya 2017; each departure names what it changes, why the
pressure-boundary treatment stays correct, and which gate proves it.

---

## 1. Where we actually are (do not plan against stale numbers)

The `166 passes / 1,513 dispatches / hierarchical MGPCG` numbers in older docs
are obsolete on this branch. Measured on this branch
(`artifacts/scene-size-overhead/`, `artifacts/measurement-floor/`):

| lane | captured | leaf | passes/adv | dispatches/adv | indirect | "MGPCG disp" | solve iters | ms/adv |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mini (persistent) | 07-29 | **2** | 80 | 442 | 252 (57%) | 1 | — | **69.6** |
| mini (persistent) | 08-03 | **32** | 80 | 503 | 328 | 26 | 4 | **245.3** |
| mini (persistent) | 08-03 | **2** | 79 | 469 | 294 | 26 | 6 | **211.1** |
| tiny-hydro | 08-03 | 32 | 80 | 499 | 328 | 26 | 4 | **361.6** |
| tiny-hydro, band 1 | 08-03 | 32 | 80 | 499 | 328 | 26 | 4 | 360.6 |
| large-hydro | 08-03 | 32 | 80 | 501 | 328 | 26 | 1 | **143.6** |
| large 20× churn | 08-03 | 32 | 81 | 515 | 328 | 26 | 2 | **306.9** |
| large 20× (persistent) | 07-30 | 32 | 80 | 470 | 281 (60%) | 1 | — | 122.7 |
| large (old hierarchical, for contrast) | — | — | 170 | 2,144 | 1,852 | 1,675 | — | — |

Note what the structure columns do *not* explain: passes/advance is 80 and
dispatches/advance is ~500 on **every** current lane, from 16³ still water to
the 20× dam, while the wall moves 143.6 → 361.6. Launch structure is flat across
the matrix; the wall is not. See §2's filled matrix for what does track it.

**Read that table with two corrections in hand.**

- **The leaf column is not decoration.** Commit `065219a` (2026-08-01,
  "fix(octree): restore factor-one scene parity") moved the mini lane's
  `FLUID_MAXIMUM_LEAF_SIZE` from 2 to 32 and the large lane's from 16 to 32.
  Every pre-`065219a` artifact therefore describes a **different
  discretization** from every post-`065219a` one, and they have been compared as
  if they did not. Measured this session on the same 240-step mini lane at HEAD:
  245.3 ms/adv at leaf 32 vs 211.1 ms/adv at leaf 2
  (`artifacts/measurement-floor/mini-240-clean-a.json`, `…/mini-240-leaf2.json`).
  So the leaf change accounts for ~14% of the gap to the 07-29 leaf-2 capture's
  69.6 ms/adv — and a **real ~3× regression remains, unexplained**. See §5.
- **"MGPCG dispatches" is a label-attribution metric, not a solve dispatch
  count.** `POWER_DAM_MGPCG_SOLVE_STAGE`
  (`tools/power-dam-performance-report.ts:476-482`) resolves a compute pass to
  its owning stage and folds `SPGrid V-cycle`, `Section 4.3 preconditioner`,
  `SPGrid accurate A2` and `SPGrid Section 6.3 apply` into the solve, then sums
  the dispatches in every such pass. The 07-29 capture recorded 1 because **no
  pass in it carried an `SPGrid V-cycle` label at all**; the current captures
  carry two (`… · capture plan L1 delta`, `… · candidate commit changed L1`), so
  the candidate rebuild's ~25 dispatches are now counted as the solve.

The pressure *solve* itself is still literally **one dispatch**:
`WebGPUOctreePersistentMGPCG.encodedDispatchCount = 1`
(`lib/webgpu-octree-persistent-mgpcg.ts:292-293`), and every fresh artifact
records `computePassesByLabel["Octree persistent MGPCG - whole solve in one
workgroup"] = 1`. It is not the launch problem. *(Recommended, not done here —
that file is owned elsewhere: rename or split `mgpcgDispatchesPerAdvance` so it
stops reading as the solve. It is currently "dispatches inside solve-owned
passes", and the SPGrid candidate rebuild dominates it.)*

What remains, with measured evidence:

1. **Scene-shaped maintenance.** The SPGrid candidate rebuild is ~21
   capacity-shaped dispatches per advance (`webgpu-octree-spgrid-vcycle.ts`
   `encodeSetupCandidate`), sized from `rowCapacity` / `brickCount` /
   `pageDirectoryWords` — all O(domain-edge²) allocation-time constants. They
   now run *through* the GPU-authored `CANDIDATE_SCHEDULE`, which does not make
   them live-shaped: `prepareCandidateSchedules` writes those same capacities
   into the indirect args on the GPU (`:3878-3879`, `brickItems = p.totals.y`
   = `plan.brickCount`), so four of them are capacity-shaped launches wearing an
   indirect costume. The four `ceil(domainVolume/256)` air-support dispatches
   are **gone** (test-locked); `domainVolume` survives in that module only in
   capacity planning and WGSL bounds checks. There are now **four**
   `airVelocitySupport.encode` call sites (`lib/webgpu-octree.ts:3774`, `:4004`,
   `:4342`, `:4380`), of which two run per steady-state advance — the artifacts
   show every Section 5 label twice, suffixed `- topology-commit` and
   `- settled-fine`. Fine topology identity scans run over
   **`maximumResidentBricks`**, not `logicalBrickCount`
   (`…fine-levelset-topology.ts:1464-1517`, `Math.ceil(plan.maximumResidentBricks / 64)`).
2. **Change-independent maintenance.** Topology candidate + descriptor + power
   topology + boundary + SPGrid setup + air support + full transport + full JFA
   redistance run **every step**, while X-1 measured 197 of the first 200
   still-water generations with **zero affected rows**, and X-2 measured
   98.7–99.1% of descriptor/topology/row-geometry/velocity records exactly
   identical frame-over-frame. X-4's freeze probe bounds this tax at **41.5%**
   of the advance; the X-9 clean-room packed band pipeline ran the fine lane at
   **0.43 ms/advance** on real page populations.
3. **A serial spine.** X-6: longest RAW/WAW path = 94.5% of wall, implied
   parallelism 1.33. The ~80 pass boundaries are almost all forced by
   `copyBufferToBuffer`/`clearBuffer` staging into INDIRECT buffers
   (`webgpu-pass-broker.ts` — every copy calls `fence()`); the mini xctrace
   shows ~40% of attributed GPU time below 10% occupancy.
4. **Two regimes.** Mini/large are **launch/latency-bound**; `ocean-seiche`
   (320×96×80, the real end-goal scale) is **work-bound** (occupancy 50–90%,
   dominated by V-cycle level-0 and JFA floods). Bets 1–3 attack the first
   regime; Bet 4 (hybrid discretization + interior coarsening) is the only one
   that attacks the second.

Also: the correctness instrument the team calls "symmetry dam break, up to step
63" is the **`symmetric-expansion`** scene (centred 2×1×2 brick block, bitwise
D4 oracle — `docs/SYMMETRIC_EXPANSION_ORACLE.md`). On this working tree the
bitwise-symmetric window now extends to **step 68** with wall-contact spread 0
(`docs/SYMMETRIC_EXPANSION_FRAME_PROFILE.md`); the 63 figure is stale. The
62-step `--lane=symmetric-expansion` perf window sits entirely inside the clean
window, which is exactly what makes it usable as a combined perf+correctness
lane.

---

## 2. The scenario matrix and its missing cells

Factor cost as `wall = f(scene size, fluid size, change rate)`. The lanes:

| | small scene | large scene |
|---|---|---|
| **still** | `hydrostatic-power-two-level` 16³ (add benchmark lane) | **missing — add `large-power-hydrostatic`** |
| **churn** | `minimal-power-dam-break` 16³ (`benchmark:power-dam-mini`) | `large-power-dam-break` 64×20×64 (`benchmark:power-dam-large`) |

Phase 0 fills the matrix:

> **Status 2026-08-02:** P0.1 (`hydrostatic-tiny` lane) and P0.2 exist in
> `tools/power-dam-lane-environment.ts`. P0.2's **lane key is
> `--lane=large-hydrostatic`**; `large-power-hydrostatic` is the *scene* id and
> `--lane=large-power-hydrostatic` throws. npm script:
> `benchmark:power-dam-large-hydrostatic`.
> P0.4 is half-captured: mini + tiny-hydro fresh baselines live in
> `artifacts/scene-size-overhead/fresh-20260802-*`; the two large cells are
> blocked on red lanes — the large-dam class-4 stage-34 trip from step 249 and
> the large-hydrostatic restriction bootstrap loop from step 3 (see the
> dry-identity RHS contract note in `webgpu-octree-persistent-mgpcg.wgsl.ts`).
> P0.5 landed (oracle doc now says 68/spread-0). P0.3 landed as the
> `runtime-150` lane on `large-power-dam-break` in
> `lib/scene-webgpu-smoke-catalog.ts` (150 steps, exhaustive power
> diagnostics, volume-drift ≤ 0.01) — it inherits the same step-249 red as the
> benchmark lane until the class-4 stall is fixed.

- **P0.1 — tiny-hydrostatic benchmark lane.** Add a `hydrostatic-tiny` entry to
  `tools/power-dam-lane-environment.ts` running `hydrostatic-power-two-level`
  for 240 steps at dt 0.004, band/factor matched to the mini lane. The existing
  `hydrostatic` lane (32×24×16 `hydrostatic-power-large-offset`) stays as the
  quarter-cell-cut GFM oracle; the tiny lane is the *cost* instrument.
- **P0.2 — large still-water scene.** Author `large-power-hydrostatic`: the
  20× container (3.2×1.0×3.2 m, 64×20×64) with the same absolute water block
  volume as the mini/large dam but **at rest** (tank-fill slab, e.g. 0.25 m
  deep so fluid ≈ the large dam's 1,472 wet cells in footprint terms). This
  completes the matrix: size axis with zero change, change axis at fixed size.
- **P0.3 — multi-step correctness lane for the 20× dam.** The smoke catalog
  has only a 1-step lane for `large-power-dam-break`
  (`lib/scene-webgpu-smoke-catalog.ts`). Add a 150-step lane with the same
  invariant set as the mini gate (validation errors 0, volume drift 0,
  publication/topology tripwires) so scene-size work has a correctness gate at
  depth, not just step 1.
- **P0.4 — fresh baselines + censuses on all four cells.** Clean walls via
  `benchmark-power-dam.ts` (A/A pairs, interleaved, per the measurement
  protocol in `POWER_LIQUIDS_ULTIMATE_M1MAX.md`), plus one run each with
  `FLUID_WORKSET_CENSUS=1`, plus one xctrace stage capture each
  (`profile:mini-dam-xctrace` / `--lane=large`). Record: ms/advance, dispatches
  (direct vs indirect), passes, per-family attributed ms,
  zero-affected-generation fraction, arena bytes. **`FLUID_OCTREE_ROW_DELTA_CENSUS`
  does not exist** — it was removed in `9de199a` and survives only in docs;
  setting it yields a silently census-free artifact. Every artifact now carries
  a resolved-`environment` record (schemaVersion 2) — check `clean` and
  `comparisonKey` before comparing two runs at all.
- **P0.5 — refresh `SYMMETRIC_EXPANSION_ORACLE.md`** divergence table (63→68,
  spread 0) so nobody gates against stale numbers.

**Decision numbers the matrix must produce** (they size every later bet):
scene-size tax = wall(large-hydro) − wall(tiny-hydro); change tax =
wall(large-dam) − wall(large-hydro); existence floor = wall(tiny-hydro).

### The matrix, filled in for the first time (2026-08-03, HEAD `f4d11e7`)

All four cells captured this session on Apple M1 Max / Dawn Metal, artifacts in
`artifacts/measurement-floor/`, each carrying the resolved-`environment` record.
ms/advance:

| | small (16³) | large (64×20×64) |
|---|---:|---:|
| **still** | **361.6** (`hydrostatic-tiny`, 240 steps) | **143.6** (`large-hydrostatic`, 240 steps) |
| **churn** | **245.3** (`mini`, 240 steps) | **306.9** (`large`, 200 steps) |

> The mini cell is 240 steps, not the lane's default 500, so it is directly
> comparable to the July `baseline-mini.json` capture; the 500-step default lane
> reads ~241 ms.

**The headline: the wall is anti-correlated with scene size.** Ranked —
tiny-still 361.6 > large-churn 306.9 > small-churn 245.3 > large-still 143.6.
**The tiny still scene is the slowest cell in the matrix**, on a domain 20×
smaller than the fastest. The plan above is written on the assumption that large
costs more than small. On this tree it does not, and the three decision numbers
come out with signs nobody planned for: the "scene-size tax"
(143.6 − 361.6) is **−218 ms**.

**The confounder, and how far it is ruled out.** The authored matrix confounds
size with interface band: the small lanes run `FLUID_OCTREE_INTERFACE_BAND: 3`
and the large lanes `1` (`tools/power-dam-lane-environment.ts`). So "small vs
large" has always also meant "band 3 vs band 1". Re-running `hydrostatic-tiny`
at band 1 gives **360.60 ms vs 361.63 ms**
(`hydrostatic-tiny-240-band1.json` vs `hydrostatic-tiny-240.json`, both
`clean: true`) — no difference. Band reach is not what makes the tiny still lane
expensive. **That single measurement is the only thing currently separating the
two axes**; the lane table still confounds them, and any other size-vs-band
question needs its own paired run.

**What does predict the wall: the persistent solve's executed iteration count.**
From `terminalCounters.pressureIterationsExecuted` in the same artifacts:

| lane | executed iterations | ms/adv |
|---|---:|---:|
| tiny-hydro | 4 | 361.6 |
| mini-dam | 4 | 245.3 |
| large-dam | 2 | 306.9 |
| large-hydro | 1 | 143.6 |

Combined with the per-pass profile — `Octree persistent MGPCG - whole solve in
one workgroup` measured at **84.9 ms/advance, 38% of the mini frame** — the
honest reading is that **the wall is set by a serialized single-workgroup
pressure solve and tracks its iteration count, not the domain**. All four lanes
encode 10 outer iterations and hard-limit at 16; none is iteration-starved.

So **Bet 1's identity is not currently the binding constraint. The "small
constant" is.** `wall ≈ O(interface changed) + O(live rows) + small constant`
can be perfectly satisfied and still leave a 361 ms still-water advance, because
one dispatch that runs the whole solve inside one workgroup is the constant.
Bet 3.2 (persistent kernels) built that constant; making it *parallel again for
large row counts* is now a first-order item, not a Bet-3 refinement.

Two caveats that must travel with these numbers:

- **The large-churn cell is not measurement-clean.** `large-200.json` reports
  `clean: false` with `FLUID_PRESSURE_ROW_CAPACITY` as its one contaminant — it
  had to be set by hand to make the lane run at all (see §5). A pressure-row
  capacity plausibly affects the wall, so 306.9 ms is provisional until the lane
  table is fixed and it is re-captured clean.
- **Iteration count is a correlate, not a law.** The mini leaf-2 run executed
  **6** iterations at **211.1 ms** while the leaf-32 run executed **4** at
  **245.3 ms** — more iterations, lower wall. Whatever sets the per-iteration
  cost (row count, occupancy, the serialized workgroup) matters at least as much
  as the count.

### Targets (accept/reject thresholds for the program, measured clean)

| lane | today (measured, 08-03, leaf 32) | target | stretch |
|---|---:|---:|---:|
| tiny hydrostatic, settled steps | **361.6** | **≤ 4 ms** | ≤ 1.5 ms |
| large hydrostatic, settled steps | **143.6** | **≤ 1.5× tiny-hydro** — *already met, in the wrong direction* | ≈ tiny-hydro |
| large 20× dam churn | **306.9** (200 steps, `clean: false`) | **≤ 1.5× mini churn** | ≤ 1.25× |
| mini dam churn | **245.3** (240 steps; ~241 at the 500-step default) | **≤ 15 ms** | single digit |
| ocean-seiche | 2-step only | 500-step lane exists and completes | ≤ 250 ms/adv |

The old "~38–44 ms" column matched no artifact in the tree and is deleted. Every
number above is from `artifacts/measurement-floor/` with a resolved
`environment` record; check `clean` and `comparisonKey` before comparing any
two. **These are not a settled floor**, for two independent reasons: the mini
lane's ~3× gap to the 07-29 capture is unexplained (§5), and the
"large ≤ 1.5× tiny" target is being met only because tiny is 2.5× *slower* than
large — see the filled matrix above. A target that passes for the wrong reason
is not a passing target; re-derive these thresholds once the solve constant is
understood.

Rationale: X-9 proves the fine lane can run at 0.43 ms; the persistent solve is
1 dispatch converging in 3–5 iterations; what stands between the measured wall
and single digits is maintenance that ignores change and dispatch shapes that
ignore fluid — plus, on the current mini number, whatever the unexplained ~3×
turns out to be. Memory: O(fluid + interface) arenas, not O(domain) — required for ocean
(current air-support allocator is ~42 KB/pressure-row and domain-volume
dispatch-shaped; it blocks >115³ today).

---

## 3. The bets

Ordered so that each produces its own measured verdict on one axis of the
matrix. Bets 1–3 are restructuring (mostly Gate A); Bet 4 is the sanctioned
physics departure (Gate B + oracles). Everything stays inside the existing
fail-closed epoch discipline: **GPU-resident predicates, indirect-zeroed work,
no host readback scheduling** (`hostSchedulingUsesReadback: false` stays true).

### Bet 1 — Existence is free: no dispatch shaped by domain or capacity
*(primary lens: 20× scenarios; also the memory unlock for ocean)*

The rule: every recurring dispatch must be shaped by a **GPU-published live
count** (compacted worklist + indirect args), or be a `(1,1,1)` control
singleton. Allocation-time capacities may size *buffers*, never *launches*.

> **Status 2026-08-03:** items 1–3 below mostly landed in `a56ddd0`
> ("perf(power-liquids): revive the coarse-only tracker and land Bet 1/4
> scaffolding" — the older `a0a2247` hash was rebased away; anchor on the
> message) and read stale as written. The candidate chain runs off the
> GPU-authored `CANDIDATE_SCHEDULE`/`runCandidateIndirect` indirect schedule;
> the four `domainVolume` air-support dispatches are gone (test-locked); the
> indirect frontier gate is default-on; the proven-reach corridor exists with
> the out-of-corridor census/reject split (the "~94k faces" figure is retracted
> by the in-tree censuses). The honest residue:
> (a) the SPGrid dense brick/page **directory sweep** — its replacement
> primitive is **not missing**: `lib/webgpu-radix-sort-u32.ts` is a genuinely
> multi-workgroup stable LSD radix sort, Dawn-validated against the CPU oracle
> `octree-spgrid-touched-directory.ts` by `tests/webgpu-radix-sort-u32.test.ts`,
> landed and wired behind `FLUID_SPGRID_TOUCHED_RADIX_SORT=1`, awaiting its
> Gate-B A/B. (`sortSparseCandidates` *is* a single-workgroup bitonic; that half
> of the old claim was right.) Meanwhile the default arm's schedule is authored
> from capacity **on the GPU** — `prepareCandidateSchedules` writes
> `plan.brickCount` and `pageDirectoryWords` into the indirect args
> (`…spgrid-vcycle.ts:3878-3879`), which is why an indirect launch here is not
> yet a live launch. (b) the fine-topology **`maximumResidentBricks`** recurring
> scans — `logicalBrickCount` does not shape them. (c) the 12 `domainVolume`
> dispatches in `webgpu-octree-coarse-summary.ts` are **unreachable on every
> matrix lane**: the module is constructed only inside
> `if (this.coarseOnlySurfaceTracking)` (`lib/webgpu-octree.ts:2819`,
> construction `:2825`), `coarseOnlySurfaceTracking = globalFineLevelSetFactor
> === 1` (`:1638`), every other touch is optional-chained, and the factor
> defaults to 4 (`lib/methods/octree.ts:7`, `:108`; `webgpu-octree.ts:2155`).
> Move it to the coarse-only/Bet 4.3 track. (d) grow-on-reject capacity
> reallocation (item 4's remainder — note the large-dam capacity override in
> `lib/scenes.ts` is the *authored budget* precedent, not a wart to retire: the
> floor-spanning collapse needs more bricks than any initial-footprint
> estimate).

1. **Indirect-ify the SPGrid candidate chain.** `encodeSetupCandidate`'s ~21
   phases move from `dispatchFor(rowCapacity/levelStride/brickCount/pageWords)`
   to indirect args published by the existing delta machinery
   (`probeCandidateSkip` / `applyCandidateSkip` already retire clean levels —
   extend them to *author the dispatch shapes* for the dirty remainder). The
   audit exists: grep every `dispatchFor(` and `workgroupPerItemDispatch(` in
   `spgrid-vcycle.ts` and classify each against a live count that already
   exists in the delta/compaction control words.
2. **DONE — do not re-run.** The four `ceil(domainVolume/256)` air-support
   dispatches are gone and test-locked. The compact-demand indirect mechanism is
   **unconditional**: `FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_DEMAND` was dead
   and is now deleted (ledger §2.1) — `encode` gates the fine-demand schedule on
   `fineSlot` alone, and setting that env var does nothing because nothing reads
   it. `FLUID_OCTREE_AIR_SUPPORT_INDIRECT_FRONTIER_GATE` is **default ON**
   (`…air-velocity-support-gpu.ts:346`, `!== "0"`, test-locked at
   `tests/webgpu-octree-air-velocity-support-gpu.test.ts:67-75`); the A/B this
   item asks for has already been taken. What is left of item 2 is the
   `maximumResidentBricks`-shaped demand marks, and corridor engagement at
   factor 4 (item 3).
3. **Air-support corridor (the structural one).** Topology reuse landed, but
   support membership is still ≈ the whole air partition — **not** the retracted
   "~94k face patches": the in-tree census is
   `terminalCounters.airSupportFaceItems = 31,584` on mini
   (`artifacts/scene-size-overhead/fresh-20260802-mini-a.json`). The stronger
   argument is in the same artifact set: the *still* scene marches **more** faces
   than the churn scene (43,776 vs 31,584, seed faces 30,528 vs 17,564, on the
   same 16³ domain — `fresh-20260802-hydrostatic-tiny-a.json`). Air support
   scales with the air partition, not with change, exactly as this bet predicts;
   it scales with domain, catastrophic at 20×/ocean. Replace
   "all air near liquid" with a **proven-reach corridor**: the union of (a)
   faces within the advection backtrace reach of any wet/interface cell
   (transport bound + interpolation stencil — the same radii the fine-band
   topology already authors as dependency cones), and (b) faces the structured
   candidate/GFM path reads. Fail-closed: consumers already reject a
   generation on missing support (`supportPublicationValid`); add a tripwire
   that counts out-of-corridor reads so under-coverage rejects rather than
   corrupts. This is not a paper departure — the paper's fast-marching
   extrapolation is inherently band-limited; marching the whole air region is
   our implementation artifact. Extrapolated air velocities never enter the
   pressure system (the accepted row set is LIQUID-ONLY), so pressure-boundary
   correctness is untouched by construction.
4. **Capacity planner from fluid footprint.** `planOctreePressureCapacity` and
   `planGlobalFineNarrowBandBrickCapacity` derive from domain cross-section
   area. Derive instead from the **authored initial fluid + inflow budget**
   (the hidden override `globalFineLevelSetMaximumBricks` in
   `lib/methods/octree.ts` is the precedent), with the existing capacity
   tripwires (fine-band 0xFFFFFFFF sentinel, wrapped pair counts) as the
   fail-closed guard, and a headroom policy (grow-on-reject reallocation at a
   generation boundary is acceptable; silent overflow is not).
5. **Arena diet** (prerequisite for ocean): re-derive air-support arenas from
   the catalog's real incidence cap instead of the 36× multiplier
   (`POWER_LIQUIDS_ULTIMATE_M1MAX.md` E3 item 3).

**Gate:** Gate A bit-exactness on mini + large lanes for items 1–2 (launch
shape only); Gate B + forced-rejection fault injection for 3–4. **Verdict
metric:** wall(large-hydro) − wall(tiny-hydro) → ~0; large-dam dispatch count
becomes fluid-shaped (470 → ≈ mini's, then both fall).

### Bet 2 — Change is the only work: delta-repair maintenance
*(primary lens: tiny hydrostatic; X-2/X-4 fund it)*

X-2's measured redundancy (98–99% exact-identical maintenance records) says the
maintenance program should be an **exact delta-repair engine**, not a rebuild
engine with a skip probe bolted on.

1. **Persistent exact carry** for power descriptors, power topology metrics,
   structured row geometry, boundary liquid masks, and structured cell
   velocities: fingerprint-gated on GPU (the `probeCandidateSkip` /
   `committedInputs` pattern is the template), repair only the rows named by
   the exact row-delta (`rowDeltaNewToOld` already exists for the pressure
   warm start). A clean step's descriptor/topology/boundary stages execute
   zero workgroups via GPU-zeroed indirect args.
2. **Do NOT hash-carry phi** — X-2 measured 0% redundancy. Phi and CPT flags
   get the lean packed-stream path (the X-9 kernel shapes: classify → carried
   worksets → staged advection → packed B4 floods → summarize). The JFA warm
   start already landed (6.6% wall); the remaining step is packing the
   transport/topology/redistance lane to the X-9 shapes under the normal
   generation contracts.
3. **Quiescent-region gating (new, and the hydrostatic end-game).** Per fine
   brick and per structured row, a GPU-resident activity predicate: max face
   speed under ε **and** zero membership delta in its dependency cone ⇒ the
   brick's transport is the identity; carry phi/CPT/velocity exactly and skip
   its worksets. This is rigid-body-style sleeping for grid regions, authored
   as worklist membership (a sleeping brick simply isn't in the transport /
   redistance / summary worklists). Waking is conservative: any neighbor
   activity, inflow, body motion, or pressure-row change in the cone re-lists
   it; a stale wake costs work, never correctness; the existing
   generation/tripwire set (transport governor, band residency, publication
   receipts) fails closed on any miss. In hydrostatic equilibrium the entire
   domain sleeps: the advance degenerates to forces + 1-dispatch solve
   (converging in ~0–1 iterations off the warm seed) + projection + snapshot.
4. **Settled solve tier stays authored, not adaptive:** keep the encoded
   iteration envelope (B4 taught us shrinking the encode is a regression);
   quiescence only zeroes *maintenance*, the solve's own GPU convergence gate
   already zeroes its tail.

**Gate:** Gate A for 1 (exact carry must be bit-identical); Gate B for 2–3
with the 500-step mini gate, the new 150-step large gate, and the
`symmetric-expansion` bitwise-D4 window ≥ 68 (quiescence must not fire
asymmetrically — the predicate is symmetric because it reads symmetric fields;
assert via the one/two/three-step stage-audit lanes). **Verdict metric:**
tiny-hydro settled wall → ≤ 4 ms; mini's settled tail (`quiescent` window,
measured via the progress heartbeat, not the two-process differencing) drops
proportionally.

### Bet 3 — Shorten the spine: one submission, few barriers, fat dispatches
*(primary lens: mini + both hydrostatics; X-6 funds it)*

1. **GPU-authored indirect args, no staging copies.** The dominant pass-break
   driver is `copyBufferToBuffer` staging into INDIRECT-usage buffers. Make
   producers write dispatch words **directly into the indirect buffer as
   storage** (the air-support prepare pattern), and batch the unavoidable
   storage→indirect transitions: one "publish worklists" pass boundary per
   phase group instead of one per family. Target: passes/adv 80 → ≤ 25.
   Instrument first: extend the pass-broker audit to attribute every `fence()`
   call site, so the top copy-driven boundaries are ranked by measured cost
   before surgery. (Remember the refuted item: this frame is not
   dispatch-*count* bound — the win here is removing **barriers**, i.e.
   critical-path stalls at near-zero occupancy, not launches.)
2. **Persistent maintenance kernel for small systems.** The persistent MGPCG
   precedent (2,144→442 dispatches; mini 107→58 ms/10 advances) generalizes:
   for systems under an authored capacity, collapse the *maintenance* chains —
   SPGrid candidate rebuild, frontier classify/sort/merge, boundary rebuild —
   into one or two persistent cooperative kernels with
   `storageBarrier()`/`workgroupUniformLoad` phase gates
   (`marchAirSupportFacesToFixedPoint` is the in-tree structure to copy).
   Selection by authored threshold, hierarchical path retained above it, both
   paths A/B-able — exactly the MGPCG playbook. This is what turns "80 passes
   of mostly-empty launches" into "a handful of dispatches" on 16³–64³ scenes.
3. **Widen what remains.** The `@workgroup_size(1)` singletons and
   one-thread-per-level phases in the candidate chain
   (`buildCandidateLevelDeltas` et al.) get the classify→prefix→scatter
   treatment (Part C recipes in `POWER_LIQUIDS_ULTIMATE_M1MAX.md`, C1/C2/C6
   remain valid on this branch — re-verify each against the persistent-era
   graph before building).

**Gate:** Gate A throughout (launch shape and pass structure only; recall the
hard lesson — any change that removes/adds a storage round-trip between
kernels is Gate B, never Gate A, because the round-trip is a rounding step).
**Verdict metric:** X-6 rerun: path/wall falls from 0.945 toward the
attributed-GPU floor; mini clean wall approaches its attributed GPU time.

### Bet 4 — Hybrid discretization: power diagram only where it pays
*(primary lens: ocean + 20×; the sanctioned paper departure)*

The paper itself proposes this in §8: *"use the non-graded approach of Losasso
et al. [2006] in regions deep interior to the liquid, and our power diagram
discretization only near the free surface to obtain a lower cost Laplace
operator that still yields a second order accurate pressure field"* — tested on
a prototype, never productized. Our version, in increasing order of departure:

1. **Machinery hybrid (no discretization change).** On uniform-resolution
   liquid interior, the power diagram *is* the regular grid: descriptors,
   catalog lookups, tet fans, and per-row explicit neighbor tables encode a
   7-point stencil expensively. Classify rows once per epoch into
   `regular-interior` vs `power-band` (free-surface band, solid cut cells,
   level transitions — the classification already exists as workset classes in
   structured dynamics and the staggered/wall-row gates). Regular-interior
   rows: implicit addressing (arithmetic neighbors within the SPGrid page, no
   `pageSlot` chains, no stored coefficients — unit coefficients scaled by
   face area/dx). Power-band rows: today's full path, untouched. The operator
   values are bit-questionable only at the seam; make the seam rule exact:
   **any face incident to a power-band row uses the power coefficient on both
   sides** — symmetry of the matrix is preserved because the coefficient is a
   property of the face, evaluated once. Pressure-boundary correctness is
   untouched: GFM θ, cut-cell fractions, and the liquid-only row set live
   entirely in the power-band class.
2. **Interior coarsening actually engaged (discretization change, paper-
   internal).** The 20× and ocean scenes should carry coarse cells through the
   deep interior (the machinery exists — `FLUID_MAXIMUM_LEAF_SIZE=32` on the
   large lane), with the power band pinned to the interface/boundary shell.
   The cost model to validate: V-cycle level-0 work and JFA support shrink
   from O(fluid volume at finest) to O(interface area). This is where ocean's
   2.4 s of level-0 V-cycle goes to die. Requires Bet 1.3/1.5 (corridor +
   arena diet) so air support doesn't reinflate it.
3. **Regional fine-band overlay (the `POWER_LIQUIDS_COARSE_ONLY_PLAN.md`
   end-state).** Factor-1 band as the always-on baseline tracker; 4×/8× fine
   band only where it earns its cost (high-curvature/high-energy regions).
   That plan is already written and consistent with this one; treat it as the
   Bet 4 extension track rather than duplicating it here.

**Gates (this is the physics bet, so the full battery):**
- `symmetric-expansion` bitwise D4 through the current window (68) at factor 1
  and the factor-4 one-step gate — the hybrid operator classification is
  D4-equivariant by construction (classes derive from symmetric geometry);
  assert it with `FLUID_SYMMETRY_STAGE_AUDIT=1` which pins the Section 6.3
  diagonal, case ids, rhs and all preconditioner stages at tolerance 0.
- Mini 500-step envelope: retention ≈0.99 @ t=0.22/0.30, KE ≈0.45 @ t=0.38,
  circularity 1.16→1.04, intentionally-red gates not redder.
- New 150-step large-dam lane (P0.3) green.
- A differential operator harness: hybrid vs full power operator applied to
  identical vectors on identical published topology — **bit-identical where
  both claim the regular stencil**, and matrix symmetry asserted across the
  seam (r·(A·s) == s·(A·r) on random vectors, in f64 on the CPU oracle
  `octree-pipelined-pcg.ts`).

**Verdict metric:** ocean-seiche per-advance attributed pressure+JFA falls by
the interior/interface volume ratio; 20× dam approaches mini wall.

---

## 4. Sequencing and effort split

```
P0  lanes + baselines + censuses            (small; unblocks everything)
B1  existence-is-free (indirect + corridor + capacity)   ← 20× lens
B3.1 spine: indirect authorship + pass batching          ← all lanes
B2  delta-repair carry + quiescent regions               ← hydrostatic lens
B3.2 persistent maintenance kernel (small systems)
B4.1 machinery hybrid (regular-interior fast path)
B4.2 interior coarsening at 20×/ocean
B4.3 regional overlay (separate plan)
```

Rationale for order: B1 and B3.1 are Gate-A restructurings that make every
later measurement cleaner (dispatch shapes stop lying about work). B2 needs
B1's live-count worklists to express "sleeping = not in the worklist". B4
rides on all of it and carries the physics risk, so it goes last and lands
behind the differential harness.

Keep per-change scoring against the matrix, not just mini: a change that helps
mini but scales with domain is a regression under this plan.

## 5. Traps carried forward (measured; do not relearn)

- **The discretization moved under the artifacts.** `FLUID_MAXIMUM_LEAF_SIZE`
  changed on 2026-08-01 in `065219a`: mini 2→32, large 16→32, hydrostatic
  16→32. Pre- and post-`065219a` artifacts are different simulations and were
  compared as if they were not. Measured at HEAD on the same 240-step mini lane:
  **245.3 ms/adv at leaf 32, 211.1 ms/adv at leaf 2**, against **69.6 ms/adv**
  for the 2026-07-29 leaf-2 capture. Leaf size explains ~14%; a **real ~3×
  regression is still unexplained and is the open item** — treat every
  cross-date comparison in the older docs as void until it is closed.
  Recurrence prevention now in tree: benchmark artifacts carry a resolved
  `environment` record (`tools/power-dam-run-environment.ts`; diagnostic
  artifact at schemaVersion 2, plus the octree regression artifact) naming every
  `FLUID_*`/`WEBGPU_*` variable the run received, which were inherited from the
  shell rather than authored, a `contaminants` list of wall-affecting knobs off
  their measurement-clean value, a `clean` boolean, and a `comparisonKey` digest
  over the scene+solver configuration. `tools/benchmark-power-dam.ts` takes
  `--leaf-size=` so the leaf is an explicit A/B axis. **Compare only equal
  `comparisonKey`s.**
- **The `large` benchmark lane is red for a lane-table reason, not a physics
  one.** `tools/benchmark-power-dam.ts --lane=large` dies at t=0 with *"Initial
  sparse authority cold-topology published no liquid-row frontier"*, while the
  smoke lane on the **same scene** (`npm run
  test:webgpu:large-power-dam-runtime`, `runtime-150`) is green for 150 steps.
  The difference is capacity, not the solver: the smoke catalog applies
  `largePowerDamOverrides` (`lib/scene-webgpu-smoke-catalog.ts:349-355`) with
  `pressureRowCapacity: 8_192` and `globalFineLevelSetMaximumBricks:
  LARGE_POWER_DAM_FINE_BRICK_CAPACITY` (32,768); the benchmark lane table
  (`tools/power-dam-lane-environment.ts`, `large:`) sets **neither**. Passing
  `FLUID_PRESSURE_ROW_CAPACITY=8192` by hand
  (`tools/webgpu-smoke-executor.ts:454-458`) makes it run green to 200 steps —
  which is how the large-churn cell above was captured, and why that cell is
  `clean: false`. **`globalFineLevelSetMaximumBricks` has no env override at
  all**: `lib/methods/octree.ts:58-62` reads it only from authored method
  values, so the benchmark lane cannot express its own scene's authored capacity
  even in principle. Fix: add `FLUID_PRESSURE_ROW_CAPACITY: "8192"` to the
  `large` lane, and decide separately whether the brick capacity gets an env
  override or the lane table gets a method-override channel. **Do not read a
  cold-topology publication failure on this lane as a solver red.**
- **A green `audit:octree-production-source` is not evidence of a live-shaped
  frame.** The gate has been green-and-blind three times (capacity vocabulary by
  enumeration; detection stopping at the literal `dispatchWorkgroups(` call; and
  the `dispatchWorkgroupsIndirect` exemption, which missed a GPU kernel writing
  `plan.brickCount` straight into the indirect args). All three are closed and
  pinned by regressions, but `docs/BET1_DISPATCH_SHAPE_AUDIT.md` §"What the gate
  still cannot see" lists what remains invisible — read it before quoting a
  clean run.
- **`environment.clean: false` is not always contamination.** An *unset*
  variable whose measurement-clean value is `"0"` is currently listed as a
  contaminant with `resolved: null` — `mini-240-clean-a.json` reports six of
  those and is otherwise a clean run. Read the `contaminants` entries, not just
  the boolean; a `resolved: null` entry means "not proven clean", a non-null one
  means "actually set to something else".
- **A/A before A/B; no second WebGPU client on the GPU** — contamination
  silently changes physics (`POWER_LIQUIDS_ULTIMATE_M1MAX.md`).
- **Dispatch-count deletion alone does not move the wall** (B4-experiment:
  −231 dispatches = +2 ms). Barriers and occupancy move it.
- **Removing/adding a storage round-trip is never Gate A** — it is a rounding
  step (fusion experiment: 6e-5 residual drift → different simulation by gen
  502).
- **Naive membership dirty oracles are unsound** — X-3's ring probe rejected;
  only per-reason dependency cones with fail-closed tripwires.
- **Host may only delete provably-zero work, from GPU-published evidence,
  staleness-conservative** (A5 contract carve-out). Quiescence and skips are
  GPU-resident predicates zeroing indirect args, never host readback control.
- **The quiescent two-process differencing lane is untrustworthy**; use the
  in-process progress heartbeat for settled windows.
- **Uniform workgroup-memory sweeps beat divergent early-exit searches** on
  this GPU; and the q-dependent `pageSlot`/`brickRecord` scattered-load chain
  is the real cost in the merged band, not page resolution.
- Segfaulted Dawn leaves `/tmp/fluid-webgpu-exclusive.lock`; never
  `git stash` in this repo; hard-reload browser tabs before trusting hot
  edits.

## 6. What "done" looks like

- The four-cell matrix + ocean all green on their gates, with clean walls at
  or under the targets in §2.
- `wall(large-hydro) ≈ wall(tiny-hydro)` (scene size decoupled).
- `wall(tiny-hydro settled) ≤ 4 ms` (change decoupled).
- `wall(large-dam) ≤ 1.5× wall(mini-dam)` (fluid-proportional churn).
- `symmetric-expansion` bitwise window ≥ 68 preserved at every landing.
- Memory O(fluid + interface); ocean-seiche runs a 500-step lane.
- Every skip/quiescence decision auditable via the step-snapshot ring and
  failing closed under fault injection.
