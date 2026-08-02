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
(`artifacts/scene-size-overhead/`):

| lane | passes/adv | dispatches/adv | indirect | MGPCG dispatches |
|---|---:|---:|---:|---:|
| mini (persistent) | 80 | 442 | 252 (57%) | **1** |
| large 20× (persistent) | 80 | 470 | 281 (60%) | **1** |
| large (old hierarchical, for contrast) | 170 | 2,144 | 1,852 | 1,675 |

The pressure *solve* is now one dispatch (`webgpu-octree-persistent-mgpcg.ts`
`encodeSolve`); it is no longer the launch problem. What remains, with measured
evidence:

1. **Scene-shaped maintenance.** The SPGrid candidate rebuild is ~21
   capacity-shaped dispatches per advance (`webgpu-octree-spgrid-vcycle.ts`
   `encodeSetupCandidate`), sized from `rowCapacity` / `brickCount` /
   `pageDirectoryWords` — all O(domain-edge²) allocation-time constants, zero
   indirect. Air support dispatches four kernels at `ceil(domainVolume/256)`
   (`webgpu-octree-air-velocity-support-gpu.ts`, `plan.domainVolume`), at **two
   encode sites per advance**. Fine topology identity scans run over
   `logicalBrickCount` (whole-domain fine bricks) and `maximumResidentBricks`.
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
  `FLUID_OCTREE_ROW_DELTA_CENSUS=1` and `FLUID_WORKSET_CENSUS=1`, plus one
  xctrace stage capture each (`profile:mini-dam-xctrace` / `--lane=large`).
  Record: ms/advance, dispatches (direct vs indirect), passes, per-family
  attributed ms, zero-affected-generation fraction, arena bytes.
- **P0.5 — refresh `SYMMETRIC_EXPANSION_ORACLE.md`** divergence table (63→68,
  spread 0) so nobody gates against stale numbers.

**Decision numbers the matrix must produce** (they size every later bet):
scene-size tax = wall(large-hydro) − wall(tiny-hydro); change tax =
wall(large-dam) − wall(large-hydro); existence floor = wall(tiny-hydro).

### Targets (accept/reject thresholds for the program, measured clean)

| lane | today (approx) | target | stretch |
|---|---:|---:|---:|
| tiny hydrostatic, settled steps | ≈ mini wall (~38–44) | **≤ 4 ms** | ≤ 1.5 ms |
| large hydrostatic, settled steps | unmeasured | **≤ 1.5× tiny-hydro** | ≈ tiny-hydro |
| large 20× dam churn | ~38–47 | **≤ 1.5× mini churn** | ≤ 1.25× |
| mini dam churn | ~38–44 | **≤ 15 ms** | single digit |
| ocean-seiche | 2-step only | 500-step lane exists and completes | ≤ 250 ms/adv |

Rationale: X-9 proves the fine lane can run at 0.43 ms; the persistent solve is
1 dispatch converging in 3–5 iterations; what stands between ~40 ms and single
digits is maintenance that ignores change and dispatch shapes that ignore
fluid. Memory: O(fluid + interface) arenas, not O(domain) — required for ocean
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

1. **Indirect-ify the SPGrid candidate chain.** `encodeSetupCandidate`'s ~21
   phases move from `dispatchFor(rowCapacity/levelStride/brickCount/pageWords)`
   to indirect args published by the existing delta machinery
   (`probeCandidateSkip` / `applyCandidateSkip` already retire clean levels —
   extend them to *author the dispatch shapes* for the dirty remainder). The
   audit exists: grep every `dispatchFor(` and `workgroupPerItemDispatch(` in
   `spgrid-vcycle.ts` and classify each against a live count that already
   exists in the delta/compaction control words.
2. **Kill the four `ceil(domainVolume/256)` air-support dispatches** and the
   `maximumResidentBricks`-shaped demand marks. The compact-demand indirect
   mechanism already exists (`FLUID_OCTREE_AIR_SUPPORT_COMPACT_FINE_DEMAND`,
   default on) — finish the family: demand mark, clears, directory stages all
   driven from the compacted demand list. Promote
   `FLUID_OCTREE_AIR_SUPPORT_INDIRECT_FRONTIER_GATE` (GPU-published wave gate,
   currently default-off) after an A/B, so quiet steps zero the 12 frontier
   waves on-GPU.
3. **Air-support corridor (the structural one).** Topology reuse landed, but
   support membership is still ≈ the whole air partition (~94k face patches
   marched on mini; scales with domain, catastrophic at 20×/ocean). Replace
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
