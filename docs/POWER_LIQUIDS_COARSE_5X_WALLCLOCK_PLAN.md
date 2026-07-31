# Coarse-Only 5× Wallclock Plan — mini dam 16³, 26.3 → ~5 ms/advance

Status: implementation plan for handoff (2026-07-31). Companions:
`docs/POWER_LIQUIDS_COARSE_ONLY_PLAN.md` (the cutover this optimizes),
`docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md` (refuted-moves register),
`artifacts/xctrace-mini-dam-16-coarse-2frames-v2-2026-07-31/` (the evidence).

## 0. Diagnosis (read before touching anything)

The profiled coarse-only mini dam frame is **not compute-bound anywhere**:
mean ALU 2.26%, read bandwidth 2.18 GB/s, mean occupancy 2.55%. Intrinsic
arithmetic per advance is on the order of 1–2 ms; the 26.32 ms clean wall is
machinery shaped for the large lane running on a 4,096-cell problem:

| Fact | Value | Source |
|---|---|---|
| Clean shipping wall | 26.32 ms/advance | `baseline-500-clean.log` (500-advance control) |
| Shipping passes / dispatches per advance | 166 / 1,513 (81% indirect) | `gpuCommandAudit` in the same log |
| Attributed compute (isolated graph) | 23.207 ms | `advance2-summary.json` |
| Pressure share | 14.004 ms, 60.4%, 1,112 encoded dispatches | family breakdown + `lib/webgpu-octree-pipelined-mgpcg.ts:730-742` |
| Passes ≤20 µs | 81.4% of passes, only 32.3% of GPU time | interval stats |
| Encoders/submits/readbacks per advance | 1 / 1 / 0 | `lib/webgpu-uniform-eulerian.ts:2005,2310`; `lib/physics-step-program.ts:125,130` |
| Implied graph parallelism | 1.33 (path/wall 0.945) | `docs/POWER_LIQUIDS_10X_DISCOVERY_RESULTS.md:161-166` |

Consequences that shape the whole plan:

- The frame graph is already optimal at the submission layer. **Do not** hunt
  CPU↔GPU sync (there is none inside an advance) and **do not** attempt
  subgraph overlap (parallelism ceiling 1.33 — it cannot pay for its
  submission boundaries).
- The wins are (a) doing less provisioned-capacity work, (b) collapsing
  launch count, (c) re-parallelizing a handful of near-serial kernels.
- Everything below preserves the Stage 2 regional fine-band overlay
  (`POWER_LIQUIDS_COARSE_ONLY_PLAN.md §5`): band machinery untouched,
  `fineFactor`-parameterized expressions stay parameterized, capacity gates
  keep large/fine lanes on their existing paths.

Budget to target (attributed-ms, with launch overhead collapsing alongside):

| Family | Today | Target | Phase |
|---|---:|---:|---|
| Pressure / multigrid | 14.0 | ~2 | 1 |
| Air-velocity support | 3.5 | ~0.8 | 2 |
| Structured velocity / boundaries | 2.3 | ~1.2 | 3 |
| Factor-1 surface pipeline | 1.7 | ~1.5 | — (already cheap; leave alone) |
| Octree topology / refinement | 1.4 | ~0.8 | 3–4 |
| Coarse publication | 0.3 | 0.3 | — |

Wall target: **≤ 6 ms/advance** on the mini coarse lane (5× gate at 5.3 ms;
accept ≥ 4.4× if Phase 1 lands only its fallback tier).

---

## Phase 0 — Close the measurement gap and pin baselines (½ day)

No shipping-mode artifact attributes CPU vs GPU (`advance2-summary.json` has
`cpu.samples = 0`; the 23.2 ms figure is from the label-isolated graph).

1. Re-run the clean 500-advance control with `GPUStageTimestampRecorder`
   enabled and label isolation **off** (`lib/webgpu-uniform-eulerian.ts:2006-2025`;
   do NOT set `FLUID_GPU_ISOLATE_PASS_LABELS`). Record: timestamp-bracket sum
   vs the 26.32 ms wall.
   - If GPU busy ≈ 15 ms → plan order below is correct.
   - If GPU busy ≈ 8 ms → CPU encoding (~1,500 Dawn dispatch records + 166
     passes + 24 bind groups/advance) is co-dominant; Phase 1 still leads
     (it deletes most dispatch records), but re-weight Phase 4 upward.
2. Split air-support site 1 vs site 2 in attribution (encode sites
   `lib/webgpu-octree.ts:3491` and `:3730`). Site 2 reuses topology nearly
   free (`s(47)`/`s(50)` reuse latches,
   `lib/webgpu-octree-air-velocity-support-gpu.ts:1011-1030`), so today's
   per-call means average a heavy call with a near-free one. Label the two
   sites distinctly (suffix the pass labels) so Phase 2 progress is measurable.
3. Record baseline quality numbers once: 500-step mini gates, hydrostatic
   lane, free-fall drop oracles, volume-drift per 100 steps, terminal
   pressure iterations (`readSolveDiagnostics`, `lib/webgpu-octree.ts:4128`).
   These are the Phase 1–4 regression denominators.
4. Keep the capture recipe: the `gpu-frame-profile` skill regenerates the
   xctrace report; every phase's exit criterion re-runs it.

Deliverable: one-page baseline table committed next to this doc or in the
artifact directory.

---

## Phase 1 — Pressure: persistent single-dispatch MGPCG for factor-1 (~10 ms)

### 1.1 The situation

A complete single-dispatch, single-256-lane-workgroup MGPCG already exists
and is validated: `lib/webgpu-octree-persistent-mgpcg.ts` (contract
`lib/webgpu-octree-section43-contract.ts:24-25`, `encodedDispatchCount: 1`),
with a threadgroup-memory page smoother and V-cycle
(`lib/webgpu-octree-persistent-mgpcg.wgsl.ts:582-604` `smoothOnePage`,
`:669` `vcycleCorrection`) and a real early `break` on convergence (`:1066`).

It is disabled for exactly this lane at two gates:

- construction: `lib/webgpu-octree.ts:2485-2486`
- selection: `lib/webgpu-octree.ts:3681-3682`

both `fineFactor !== 1 && …`. The recorded reason (`:2478-2484`) is
numerical, not architectural: the kernel's **compact Section 4.3 recurrence
loses preconditioner positivity after repeated factor-1 authority
generations on the hydrostatic plane**. The hierarchical A/B path stays
positive on the same rows.

Evidence of the payoff: with the persistent solver active, mini ran at
80 passes / 442 dispatches per advance with `mgpcgDispatchesPerAdvance = 1`
(`artifacts/scene-size-overhead/baseline-mini.json`) and mini dam wall went
107 → 58 ms/10 advances (`lib/webgpu-octree.ts:395`).

### 1.2 Primary work item

Fix positivity of the compact §4.3 recurrence under factor-1 authority, then
delete `fineFactor !== 1` from both gates.

1. **Reproduce first.** Build a failing test before changing arithmetic: a
   hydrostatic factor-1 lane run under the persistent executor
   (`FLUID_OCTREE_PERSISTENT_MGPCG=1` with the gate temporarily bypassed)
   for enough advances that repeated authority generations accumulate.
   Assert the failure signature (non-positive preconditioner energy /
   diverging or stalling residual). This test becomes the Phase 1 gate.
2. **Diagnose the divergence between the two representations.** The
   hierarchical path's §4.3 arithmetic
   (`lib/webgpu-octree-section43-preconditioner.ts`, sweeps at `:333-358`,
   merged-band pairing at `:369-390`) is the known-good reference. Diff the
   compact recurrence in the persistent WGSL against it term-by-term on the
   hydrostatic configuration; the plane is exactly representable, so any
   drift is a bug with a findable first divergent term (suspects: f32
   rounding order in the compact fold, shell-depth handling `k=4`
   `lib/octree-solve-tail-policy.ts:19`, or the factor-1 boundary-band
   image differing from what the compact recurrence assumes).
3. **Prefer transcription over invention**: if the compact form resists,
   port the hierarchical sweep ordering verbatim into the persistent kernel
   (barriers where dispatch boundaries were). SPD symmetry is legitimate in
   one workgroup via `workgroupBarrier()` — the objection at
   `lib/webgpu-octree-spgrid-vcycle.ts:1598-1601` applies to *frozen-halo
   page-local* folding, not to a globally synchronized single workgroup.
4. Keep every existing gate except the factor test:
   - capacity: `WebGPUOctreePersistentMGPCG.selects(rowCapacity)` ≤ 4,096
     (`OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY`, contract `:18`).
     The persistent path is **known-catastrophic at large capacity**
     (74.82 vs 38.3 ms, `docs/POWER_LIQUIDS_10X_DISCOVERY_RESULTS.md:278-281`).
     This gate is also the regional-overlay compatibility story: fine/large
     lanes never select it.
   - env kill-switch `FLUID_OCTREE_PERSISTENT_MGPCG=0`
     (`lib/webgpu-octree.ts:403`) stays functional for A/B.

### 1.3 Fallback tier (if positivity resists a timebox of ~3 days)

Independent, ordered by value; all remain useful for the large lane even if
1.2 lands:

- **(a) Stop encoding the dead iteration tail.**
  `lib/octree-solve-tail-policy.ts:131` hard-wires 10 encoded outer
  iterations; the profiled advance executed 6, the 500-advance control 4.
  4 dead applies ≈ 388 zero-workgroup dispatches ≈ 31% of lane launches.
  **Constraint (verified):** the policy comment at `:124-130` records that
  transient mini-dam steps genuinely need the tail (wall-climbing jet), so
  this must be an encode-side, stale-safe predicate — e.g. the P0.5
  conductor-conditional carve-out (`lib/physics-step-program.ts:127`,
  `predicates` currently frozen empty at `:137`) fed by the **previous
  advance's published iteration count + safety margin (predict high, clamp
  to the envelope on any topology change)** — never a lowered cap, and the
  GPU residual gate remains the convergence authority.
- **(b) Fold V-cycle levels 1–4 into one single-workgroup kernel.**
  Levels 1–4 hold ~5 workgroups of work but cost 30 of 50 dispatches per
  cycle → 330 dispatches/advance. `smoothOnePage`/`vcycleCorrection` in the
  persistent WGSL are a working implementation to lift. Level 0 (~24 wg)
  keeps the existing row-parallel path.
- **(c) Shape the three capacity-direct dispatches by live rows.**
  `clearCorrection`/`seedRhs`/`publish`
  (`lib/webgpu-octree-spgrid-vcycle.ts:1797,1799,1809`) launch
  `ceil(rowCapacity/64)` = 64 wg for ~1,500 live rows, 33×/advance. Reuse
  the `gatedRowDispatch` indirect pattern from
  `lib/webgpu-octree-section43-preconditioner.ts:434`. No arithmetic change.
- **(d) Merge the two per-iteration reduction drains**
  (`lib/webgpu-octree-pipelined-mgpcg.ts:832-847`) into one combined finish
  kernel: 20 single-workgroup device drains → 10.

### 1.4 Verification (Phase 1 exit)

- The new hydrostatic-positivity test passes under the persistent executor
  for ≥ 500 advances at factor 1 (bit-stable pressure — hydrostatic is
  exactly representable, any drift is a bug, per
  `POWER_LIQUIDS_COARSE_ONLY_PLAN.md §4.6`).
- A/B trajectory vs the hierarchical path on the 500-step mini lane: same
  quality gates, terminal iteration counts within +1, no tripwire
  (frozen-surface, zero-iteration) fires.
- Free-fall drop oracles pass (memory: wall-sticking oracles — contact
  behavior is solver-sensitive).
- `gpuCommandAudit`: `mgpcgDispatchesPerAdvance = 1`; advance total
  dispatches ≤ ~500, passes ≤ ~80.
- Wall: mini coarse lane ≤ 12 ms/advance after this phase alone.
- Fallback-tier items (a)/(c)/(d) additionally require: byte-identical
  command-stream semantics test where one exists (the encode stringify test
  referenced at `lib/webgpu-octree.ts:3678-3680` must be updated
  deliberately, not silently), and for (a) a lane where iterations spike
  (impact transient) proving the predicate never under-encodes: assert
  `executedIterations < encodedIterations` on every advance.

---

## Phase 2 — Air support: fix the factor-1 degeneracies (~2.5 ms)

All in `lib/webgpu-octree-air-velocity-support-gpu.ts`. Three independent
work items; each is order-free ("Gate A shaped") with the proofs already in
the file.

### 2.1 Demand-neighborhood marking (highest ratio, lowest risk)

At factor 1, `markFineBandDemandNeighborhood` (`:1131-1135`) computes
`radius = ceil(maxDisplacementFineCells / fineFactor)` = 3 → a 343-tap
`atomicOr` loop per lane; and the uniform-brick fast path (`:1192-1200`)
is dead because `sampleBase = q / fineFactor` is the identity, so
`markFineBaseSplit` (`:1185`) is always set. Net: ~10⁶ atomics to set at
most 4,096 flag bits, in a `@workgroup_size(64)` kernel (`:1142`).

Fix (either; first is simpler):
- Factor-1 uniform path: reduce over the brick's own base-cell set (a 4³
  brick covers exactly 64 known cells) and emit the neighborhood once per
  distinct base cell, not once per sample.
- Or separable dilation: 3 axis sweeps of width `2r+1` over the domain
  (~28k taps vs ~10⁶). `publishFineDemand` is idempotent/commutative
  (`:948-956`, proof at `:1163-1175`) so both are order-free.

**Constraint:** keep `radius` an expression of `fineFactor` (`:1132`) — the
Stage 2 overlay runs factor 4/8 concurrently and needs the divide.

### 2.2 Capacity-shaped arena sweeps → live-shaped

`clearAirSupportCandidates` (`:629`), mark/scan + prefix + scatter
(`:648-650`) each traverse the full provisioned extent `R·231 + 4096`
(~1M slots on a 4,096-cell domain); `clearAirSupportTags` (`:1106-1110`)
stores `R·283` INVALID words. This is item 2 of E3
(`docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md:1082-1088`), pre-analyzed:
- Fold the candidate clear into `emitAirSupportCandidates`' first phase —
  each row owns the contiguous block `[row·231, (row+1)·231)`.
- Shape mark/scan from the emitter's live count instead of the provisioned
  extent.
- Clear only the tag ranges the emitter will write.

### 2.3 March structure

The sparse changed-frontier march (`:727-743`):
- 12 host-side waves run **unconditionally** past convergence
  (`OCTREE_AIR_SUPPORT_GPU_PARALLEL_FRONTIER_WAVES = 12`, `:63`); each wave
  includes `advanceAirSupportChangedFrontier` at **1 thread** (`:739`,
  `:1897-1900`).
- The tail `marchAirSupportFacesChangedFrontier` is a 3-workgroup persistent
  kernel — 768 threads on a 32-core GPU (`:743`, `:1901-1933`).

Fixes:
- Fold the wave counter roll-over into the last workgroup of
  `commitAirSupportChangedFrontier` (atomic last-workgroup-done latch):
  deletes 12 dispatch boundaries per march.
- Route the GPU convergence flag (`faceFrontier[10]`, `:1866`, `:1900`) into
  the indirect args so post-convergence waves dispatch **zero** workgroups —
  the mechanism exists (`finalizeRetainedAirSupportMarchSchedule` zeroes a
  schedule at `:1755-1759`).
- Widen the tail (more workgroups per axis); `betterFace` (`:1764-1767`) is
  a min over `(squaredDistance, seedItem)`, so wider execution is order-free.

### 2.4 Forbidden moves (both previously refuted — do not revisit)

- **Do not shrink the march destination set.** Whole-air-partition marching
  is a correctness invariant; demanded-only destinations islanded thin films
  and froze them (`:1303-1311`, restated in E3). This invariant is also what
  Stage 2 partial coverage depends on.
- **Do not merge the two per-advance encode sites** (`lib/webgpu-octree.ts:3491`
  and `:3730`). B2 is BLOCKED: live same-step consumer, different field
  states, dependency cycle; measured +45% mechanical energy
  (`docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md:549-588`).

### 2.5 Verification (Phase 2 exit)

- Existing suite `tests/webgpu-octree-air-velocity-support-gpu.test.ts`
  green; update the wave-constant assertions (`:505-508`) deliberately if
  the wave structure changes.
- New unit test: factor-1 demand mask equals the brute-force
  `(2r+1)³`-per-sample reference mask bit-for-bit (idempotence makes this
  exact, not approximate).
- March fixed point identical pre/post change on a captured lane state:
  assert equal face carrier fields (the march is deterministic given seeds).
- Energy gate: `peakLiquidSpeed` and mechanical-energy trajectories within
  the established envelope on the 500-step mini lane (the B2 failure mode is
  energy injection — watch for it even on the allowed changes).
- Per-site attribution (Phase 0 item 2): site-1 identities + march + face
  stages ≤ 1.0 ms combined on the re-profiled frame.

---

## Phase 3 — Re-parallelize the near-serial kernels (~1.5 ms)

Long single-call stages at near-zero occupancy (from `stage-breakdown.csv`):

| Stage | ms | Occupancy | Family |
|---|---:|---:|---|
| Advect structured family class 7 | 0.464 | n/a (~serial) | structured dynamics |
| Advect structured family class 8 | 0.409 | 0.7% | structured dynamics |
| Inactive pressure-row candidate one-ring publication | 0.516 | 2.0% | topology |
| Octree resident grading closure | 0.373 | 0.0% | topology |
| Finalize direct structured publication | 0.218 | 0% | structured dynamics |

A 400 µs kernel at ~0% occupancy is a handful of threads looping. These were
not root-caused by the discovery work — **first task is a 1-day read of each
kernel** (`lib/webgpu-octree-structured-dynamics.ts` for the advect classes;
grep the labels for the topology pair) answering: what does the per-thread
loop iterate over, and can it be flattened to one thread per element with an
order-free combine? Only then implement.

Verification: trajectory A/B on the 500-step mini lane (advection classes
touch the velocity field — hold to the same envelope gates as Phase 2), plus
per-stage time in the re-profiled frame: each listed stage < 100 µs or a
written note explaining why it is irreducible.

---

## Phase 4 — Pass-boundary reduction (~1–2 ms; do last)

Sequenced last because Phase 1 deletes most pressure-lane boundaries and
this phase's targets shift accordingly. Measured price: ~0.12 ms per removed
boundary (structured-publication packing precedent,
`docs/POWER_LIQUIDS_10X_DISCOVERY_RESULTS.md:288`).

1. **Storage→INDIRECT staging copies: 60/advance**, each forcing a fence at
   `lib/webgpu-pass-broker.ts:154` (via `updateIndirectBuffer` `:159-167`),
   3.5 KB total moved. Cluster copies at boundaries that must exist anyway —
   e.g. `lib/webgpu-octree.ts:3141-3158` already has 3 adjacent copies the
   broker collapses to one fence, but `:3141` sits apart for no stated
   reason. Target: ≤ 25 forced boundaries.
2. **Singleton gate chains**: `webgpu-octree-topology-epoch.ts:173-188`
   fences 3× for 3 kernels; the first fence (`:178`) separates two
   plain-storage kernels and can be dropped (WebGPU orders storage accesses
   within a pass — the broker header `webgpu-pass-broker.ts:15-17` says
   exactly this). Same shape at `webgpu-octree-owner-pages.ts:1425,1441`.
   ~20 singleton+fence pairs per advance run ~1 µs of work against ~26 µs of
   boundary.
3. **Broker handoffs**: 24 `new PassBroker(encoder)` sites in
   `webgpu-octree.ts` each force a boundary. Where consecutive sites share
   the encoder with no intervening copy/clear, thread one broker through
   instead.

Verification: `gpuCommandAudit` pass count per advance ≤ ~50; the
step-sequence conformance check (`webgpu-uniform-eulerian.ts:2340`) and the
encode-stringify test still pass; full mini quality gates unchanged (fences
here have no semantic content, so any behavior change is a bug).

---

## 5. Global verification harness (applies to every phase)

Every phase lands behind the same regression wall, in this order:

1. **Unit/contract tests** touched by the phase (named per phase above).
2. **500-step mini quality gates** (the set from
   `POWER_LIQUIDS_TEMPORAL_COHERENCE_HANDOFF.md:57-73`, as pinned by
   `POWER_LIQUIDS_COARSE_ONLY_PLAN.md §4.6`): volume drift per 100 steps,
   θ-histogram L1 delta, surface metrics.
3. **Hydrostatic lane**: bit-stable pressure. Non-negotiable — exactly
   representable, so any regression is a bug.
4. **Free-fall drop oracles** (ceiling/seam sticking — see memory note).
5. **Tripwires** stay fatal in tests: frozen surface with kinetic energy,
   terminal iterations = 0 on a churn lane, sentinel band counts, volume
   budget. The known failure family here is *"faster frame, frozen surface,
   still prints PASS"* — a perf phase that turns a tripwire off has failed.
6. **Re-profile** with the `gpu-frame-profile` skill (2-advance xctrace,
   same recipe as `artifacts/xctrace-mini-dam-16-coarse-2frames-v2-2026-07-31/`)
   and diff `family-breakdown.json` against the Phase 0 baseline. Also
   re-run the clean 500-advance control for the wall number — the isolated
   trace never substitutes for it.
7. **Command audit diff**: passes, dispatches, indirect share, bind groups,
   copies per advance from `gpuCommandAudit`. Structural phases must move
   these numbers; if wall improves but the audit doesn't, the improvement is
   not the one you engineered — investigate before crediting it.

Acceptance for the plan as a whole:
- Clean mini coarse lane ≤ 6 ms/advance (5× gate: 5.3 ms).
- All quality gates at or better than Phase 0 baseline.
- Large lane unregressed (persistent solver stays capacity-gated; run the
  large-lane control once after Phases 1 and 4).
- Factor-4 lane unregressed (the factor-parameterized code paths in Phase 2
  still serve the existing fine-band configuration — run one factor-4 mini
  control after Phase 2).

## 6. Refuted / out-of-scope register (do not spend time here)

| Idea | Status | Evidence |
|---|---|---|
| Merge air-support encode sites (B2) | BLOCKED | `POWER_LIQUIDS_ULTIMATE_M1MAX.md:549-588`, +45% energy |
| Shrink march destination to demanded set | BLOCKED (correctness) | `webgpu-octree-air-velocity-support-gpu.ts:1303-1311` |
| Fuse §6.3 A2 apply into band smoother | Gate B only, previously diverged | `POWER_LIQUIDS_ULTIMATE_M1MAX.md:324-357` |
| Multi-page Chebyshev folding with frozen halos | Not SPD | `webgpu-octree-spgrid-vcycle.ts:1598-1601` |
| Subgraph overlap / multi-submit pipelining | Parallelism ceiling 1.33 | `POWER_LIQUIDS_10X_DISCOVERY_RESULTS.md:161-166` |
| Lowering the encoded iteration cap numerically | Breaks transient jets | `octree-solve-tail-policy.ts:124-130` |
| Persistent MGPCG above 4,096 rows | Catastrophic | `POWER_LIQUIDS_10X_DISCOVERY_RESULTS.md:278-281` |
| Hunting CPU↔GPU sync inside the advance | None exists | `physics-step-program.ts:125,130`; audit: 0.018 fences/advance |
