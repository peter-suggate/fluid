# Power-liquids M1 Max restructure — implementation handoff

Branch `perf/structured-cutover`, written 2026-07-27. This document
describes the changes to make, in enough detail to implement without the
analysis context. Anchor on **symbol names** — a second session edits this
tree and line numbers rot. The physics conformance record
(`docs/papers/aanjaneya-2017-implementation-audit.md`) is unaffected: none
of these changes alter the discretization, the preconditioner algebra, the
band physics, or any fail-closed publication semantics.

Recommended order: Part A (groundwork) first — several later changes are
illegal or unmeasurable without it. Then B (encode gating), C (kernel
widening), D (persistent solve), E (fine lane), F (scale items), roughly
independent within each part.

> **2026-07-29 performance correction.** The Part-D conclusion below that the
> persistent solver was slower is superseded. That result came from a
> label-isolated ~1,470-row case and attributed tracing overhead as shipping
> wall time. Clean current-tree A/Bs show the opposite for small scenes:
> ceiling drop is 113 -> 53 ms/10 advances and mini dam is 107 -> 58 ms/10,
> with converged residuals and zero tripwires. The persistent executor is now
> the default up to its constructed 16,384-row capacity; setting
> `FLUID_OCTREE_PERSISTENT_MGPCG=0` retains the hierarchical oracle.
>
> Section 5 also now separates topology from values across the two publications
> in one accepted epoch. The second publication reuses identities, tags, owner
> directory, catalog topology and face adjacency after a GPU-side VALID receipt
> check, while refreshing fine-demand flags, seeds, the closest-face fixed
> point and reconstructed velocities. Its metric is squared Euclidean distance
> in the candidate loop (one square root per reconstructed row), and its wide
> envelope is 12 waves plus an exact persistent convergence tail.

---

## How to run and verify (used by every change below)

Environment: `WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js`
prefixed to every command (reinstall after reboot:
`mkdir -p /private/tmp/fluid-webgpu && cd /private/tmp/fluid-webgpu && npm
install webgpu --no-save`). The GPU is exclusive: one Dawn process at a
time, no browser WebGPU tab. After a SIGSEGV, delete
`/tmp/fluid-webgpu-exclusive.lock` only if the recorded pid is dead.

Throughput (clean — the only walls you may compare across builds):

    npm run benchmark:power-dam-mini              # full 2 s, churn then settle
    npm run benchmark:power-dam-moving-interface  # 62-step all-churn window
    npm run benchmark:power-dam-quiescent         # settled window, steps 501–560
    npm run benchmark:power-dam-ui                # larger scene, all churn

**Measured baseline, 2026-07-27, working tree (HEAD 5354e25 + uncommitted).**
mini full-2 s **108.36**; mini churn window **96.18**; ui churn **136.05**.
Pre-instrumentation commit c8ee6aa measures 107.12, so the three
instrumentation commits are not responsible for the level.

**There is no settled regime, and `--lane=quiescent` must not be trusted.**
One 560-step run, per-25-step wall from the harness's own `record: "progress"`
heartbeat (`windowWallPerStep_ms`, no differencing) — through step 60 / 120 /
180 / 240 / 300 / 360 / 420 / 480 / 540: 99.8 / 105.3 / 111.3 / 114.1 / 113.5 /
112.8 / 111.6 / 133.9 / 112.0 ms. The cost is flat at ~100–115 ms for the whole
run; the full-2 s average of 108.36 corroborates it. The quiescent lane runs
500 steps and 560 steps as two INDEPENDENT processes and differences their
cumulative walls over the 60-step tail; that tail is ~11 % of the wall, so the
divide amplifies run-to-run drift ~9× — a 5 % drift moves the reported number
~45 %. It reported 159.65 today; an earlier session recorded 10.27 from the
same lane. Both are drift. Until the runner can window a trailing sample inside
ONE process, use the progress heartbeat.

Consequences: change **B5 (settled tier)** loses its headline — there is no
~10 ms floor to defend and no 1–2 ms end state to reach; a settled tier can
still delete work but its ceiling is the same ~110 ms every step pays. Against
that, the churn attribution applies to ALL 500 steps rather than only the first
60, so Parts B–E are worth more than a churn-only reading suggests. Score every
change against the ~110 ms flat cost.

Attribution (traced — serializes the frame; use only to see where time
goes, never as a wall): `npm run profile:power-dam-mini`,
`profile:power-dam-mini:fine`, `profile:power-dam-ui`.

Correctness gates:

- **Gate A (bit-exact).** For changes that claim to be
  restructuring-only. Run `npm run test:webgpu:minimal-power-dam-break`
  on the parent commit and on your change; the per-step field
  fingerprints (`phiBitXor` and the final field stats in the smoke
  output) must be identical. Any difference means the change was not
  restructuring-only: stop, find why.
  **Run the A/A pair (the same build twice) alongside it — see "A second
  WebGPU client on the GPU changes physics results" below. A/A must be zero
  before A/B means anything.**
- **Gate B (envelope).** For changes that legitimately reorder float
  operations or change eligibility logic. The 500-step gate must pass
  500/500 steps with 0 validation errors and volumeDrift 0, and the
  checkpoint physics must stay inside the recorded bracketing in
  `docs/STRUCTURED_DAM_BREAK_HANDOFF.md` (retention ≈0.99 at t=0.22/0.30,
  KE ≈0.45 at t=0.38, floor circularity 1.16→1.04). The intentionally-red
  energy gates must not get redder. Do not chase late-time (t>1.0) parity
  with main; main violates its own invariants there.
- **Tripwires** (Part A adds these as hard failures; until then check
  them manually in the smoke output): topology rollback on any
  generation; nonzero `fineRestrictionControl[3]` (unaccepted restriction
  rows); terminal pressure iterations of 0 on a churn lane; fine-band
  active count 0xFFFFFFFF (capacity overflow — silently no-ops the solver
  and still prints PASS); solver non-convergence at the encoded budget.
  A change that gets faster while tripping any of these is not a speedup;
  a silent topology rollback once manufactured a fake 1.66× win.

Test-pinning rule: several tests deliberately pin the current
architecture. Where a change must break one, this document names it;
change exactly that assertion in the same PR, with a comment referencing
this document. Never loosen a regex beyond the named assertion.

---

## A second WebGPU client on the GPU changes physics results — measured 2026-07-28

**This section replaces an earlier one that claimed the lane is not run-to-run
deterministic. That claim was wrong, and the retraction is the point: the
divergence was contamination, not a solver bug.**

What was actually seen. Four runs of
`npm run test:webgpu:minimal-power-dam-break`, two of each of two builds,
produced three distinct physics results, grouping across builds rather than
along them. Every one of those runs was taken while the fluid sim was also open
in a browser WebGPU tab.

What a quiet machine gives, same commit, same lane, same harness:

- three A/A runs at the lane's exact configuration (500 steps, raster on,
  envelope on, `FLUID_CHECKPOINT_EVERY_S=0.1`, `skip_validation`): all three
  byte-identical across all 21 `compact-octree-field-readback` records;
- two A/A runs of the npm script itself: 0 differing readback lines;
- A/A at 300 and 500 steps, raster on and off, dense checkpoints: 0 differing.

So the lane IS deterministic, and Gate A is a sound instrument — **provided
nothing else is using the GPU.** The repo's existing "close every browser WebGPU
tab" warning is not only about crashes and throughput: a shared GPU can silently
change the values a run publishes. First divergence in the contaminated pairs
was at generation 227, in `mgpcgControl` residual low bits and `volumeControl`
word 6 (`correction`, an f32: ~-7e-9 against exactly 0.0), with executed
iterations and row count agreeing — small enough to look exactly like benign
float reassociation, which is what made it convincing.

Consequences for anyone measuring here:

1. A contaminated run can produce a false Gate-A **pass** as easily as a false
   failure. Two of the four contaminated runs agreed with each other across
   different builds.
2. Always run the A/A pair (the same build twice) alongside the A/B pair, in the
   same session, and report both. A/A must be zero before A/B means anything.
   This is cheap and it is the only thing that catches contamination.
3. Do not gate on the raster checkpoints
   (`globalFineGenerationCheckpoints[*].raster`, `frontInterfaceHash`,
   `terraceEdges`). Under contamination an A/A pair differed at 12 of 20 of
   them, against 12 of 21 for the physics readbacks — they amplify, not isolate.
4. The physics instrument to diff is the `compact-octree-field-readback` line
   set. It excludes every timing field, so it needs no normalisation.

Verified with this protocol on a quiet machine: `f9b599b` (candidate hash claim)
is bit-identical to its parent across all 21 records — Gate A, confirmed rather
than assumed. The single-page V-cycle coarse tail was **not**: 21 of 21 records
differed against the same parent, which is why it is reverted rather than
downgraded to Gate B.

## Measured results, 2026-07-27

Interleaved A/B (alternating runs between two worktrees, paired median —
blocked A-then-B designs produce false verdicts on this machine, see below).
All rows below are **Gate A bit-identical**: the 500-step field fingerprint and
the invariant-failure set are unchanged from the parent.

| lane | before | after | delta |
|---|---|---|---|
| mini churn | 97.55 | 69.56 | **−27.97 ms (−28.7 %)** |
| ui churn | 136.66 | 85.69 | **−51.04 ms (−37.3 %)** |

The ui lane gains more because C6 and E3 both scale with domain size. ui
counters after: 1,284.1 dispatches, 168 passes, attribution complete, fine-band
occupancy 93.9 %, terminal pressure iterations 5/10, 0 validation errors.

Landed: A1, A2, A3, A4, A5, B1, B3, B6, C1, C2, C3, C4, C6, E1, E3, F4.
Intermediate checkpoints on the same lane: 96.71 → 95.55 after
A1/B1/B3/B6/E1; → 88.95 adding C2/C3/C4/F4; → 84.56 adding C1;
→ 79.59 adding E3; → 69.56 adding C6.

Work counters, mini churn: 1,199 → 1,205.1 dispatches (UP — see item 9 below),
169 → 168 passes, 0 validation errors, `topologyRolledBack` false, fine-band
occupancy 99.8 %, terminal pressure iterations 4/10. Air-support producer
allocation 112.3 MB → 19.6 MB (28.07 → 4.89 KB per pressure row; at 128³,
56.15 GB → 9.77 GB).

**Measurement protocol — required.** This machine drifts on ~hour timescales
(one build measured 88.95 and 92.80) and has periodic episodes that slow both
arms at once. Never compare builds measured in separate blocks. Alternate runs
between two worktrees and take the median of per-round deltas; within a paired
design the resolution is ±0.1–0.4 ms on a ~90 ms lane, so 1 ms changes are
detectable. Discard rounds where both arms exceed 1.3× their own minimum.
A blocked design once made a stable 5.18 ms win (C1) look like a 60 %
nondeterminism bug.

**`npm run test:water-shaders` is NOT a uniformity checker.** naga 30.0.0
validates `if(lid<10u){workgroupBarrier();}` and a barrier after a
storage-derived loop bound as successful, exit 0. Verify barrier uniformity
structurally instead: every barrier's enclosing construct must be bounded by a
`var<uniform>` value, a literal, or a `workgroupUniformLoad` — never a storage
read or `local_invocation_id` — and every early `return` must sit after all
barriers so per-lane barrier counts match. Tint enforces this for real, so a
violation surfaces as pipeline-creation failure on Dawn.

## Things that look like wins and are not (do not implement)

These were each proposed and then refuted by measurement or analysis.
Recorded so the implementer does not rediscover them.

1. **Shrinking redistance and topology as if they were one radius.** Physical
   redistance now ends at `transport`, but topology must independently
   cover `transport + backtrace + interpolation`. Its required radius credits
   the mandatory final safety ring only after proving that complete residency
   floor. Under-covering it rejects and rolls back topology — the surface
   freezes, the frame gets faster, and a run without per-generation tripwires
   can still print PASS. Immutable page capacity deliberately retains the
   former conservative envelope; reserved pages do not become active work.
2. **Staging transport field data into workgroup memory.** The
   backtrace reach is 8 fine cells + 1 interpolation cell, so one staged
   φ channel needs a 22³ tile = 42,592 bytes against the 16,384-byte
   workgroup-storage budget; and total staged traffic per output sample
   (~666 words) exceeds what the kernel currently fetches (~160 words).
   The right variant is staging the *addressing* only — see change E2.
3. **Choosing the solve iteration budget from the previous frame's
   executed count.** Executed iterations jump by more than one across
   impact frames, and when the budget is exhausted without convergence,
   `finalizeAndPublish` publishes the *seed* pressure and nothing fails
   the step (`pressureConverged` is only read by a 250 ms telemetry
   poll). A trailing budget parks the encode exactly on that cliff.
   Budget selection must be authored (change B4), and non-convergence
   must become a tripwire (change A3).
4. **Deleting the 12-wave Jacobi prefix before the Section 5 march.**
   The march's ping-pong `readA` initializer is derived from the prefix
   wave count, and the prefix offloads work the 768-thread persistent
   march would otherwise do at 19 % occupancy. Keep the prefix. The later
   production result uses 12 additional wide waves, then the persistent tail;
   the former 48-wave envelope encoded 36 redundant sweeps per publication on
   the ceiling graph.
5. **Enabling the dead `var topologyChanged = true` branch in
   `planL1CaptureDelta`.** Its comparison is unsound three ways (compares
   2 of 4 relevant words, indexes captured geometry with new-generation
   row indices, mixes generations). The *live* skip mechanism is
   `probeCandidateSkip` / `applyCandidateSkip`, which compares 4 words
   per row against `committedInputs` and already retires every level's
   dirty word — measured working (topology candidate cost collapses when
   the scene settles).
6. **Parallelizing the recurring seed sort.** `sortSeedRecords` output is
   provably unread on recurring steps; the fix is not encoding the chain
   at all (change B1). Its measured cost is ~1 ms — an older 13.75 ms
   figure was an artifact of the serialized tracing regime.
7. **Removing pass-boundary fences wholesale.** In
   `webgpu-octree-structured-velocity-gpu.ts` exactly two fences are
   provably removable (the two between `classify` and `section63`,
   marked by the in-file comment "on paper these need no boundary"). The
   fence after `finalize` guards a genuine hazard: `finalize` writes
   `liveRowDispatch` as storage and `encodeCandidateReconstruction`
   reads the same buffer as indirect args. Storage→indirect transitions
   force pass breaks in WebGPU; that is why ~120 of the frame's pass
   boundaries exist and cannot be removed by refactoring alone.
8. **Batching solve iterations behind extra pass boundaries** — measured
   as a regression on this GPU in an earlier round; the encoded chain
   must get shorter (Part D), not deeper.
9. **Shortening the encoded solve budget — change B4 itself.** MEASURED
   2026-07-27 and refuted. Deriving `E` from the discarded
   `sceneComplexityScore` (mini scores 7) removed **231 dispatches and 24
   compute passes per advance** (1,199 → 968) and the churn wall got
   *slower*: 98.74 / 98.71 ms against a 96.55 / 96.71 / 96.77 baseline.
   The GPU convergence gate already zeroes the unused tail, and a
   zero-work indirect dispatch costs so little that 231 of them are below
   the noise floor. B4 is a ~2 ms **regression**; do not implement it.
10. **Replacing a uniform workgroup-memory sweep with per-lane
    first-hit searches.** MEASURED 2026-07-28 and refuted, on
    `linkCandidateParentChains`. Its predecessor/last scan visits all 256
    `chunkKey[t]`; the "obvious" improvement is for each lane to scan down
    from `lane-1` for the predecessor and up from `lane+1` for `last`,
    breaking at the first hit. That is *exactly* equivalent — verified over
    1,024,000 lane cases across three key distributions, and it is integer
    index arithmetic, so there is no reassociation to worry about — and it
    measured **0.903 → 1.134 ms/advance (+26%)** under label isolation while
    every neighbouring SPGrid kernel stayed flat in the same capture.
    The reason is the transferable part: in the uniform sweep every lane
    reads the *same* `chunkKey[t]` index, which is a workgroup-memory
    broadcast; giving each lane its own address turns one broadcast into 32
    distinct bank accesses, and the early exits buy nothing because a SIMD
    group still runs to its slowest lane. Generally on this GPU, prefer a
    uniform full sweep over a divergent early-exit search when the operand
    lives in workgroup memory. `tests/webgpu-octree-spgrid-vcycle.test.ts`
    pins the shared index so this cannot be reintroduced silently.
11. **Memoizing `pageSlot`'s page resolution on the ordinal.** MEASURED
    2026-07-28 and refuted. `applyRow` and `finerAdjoint` walk eighteen
    stencil directions from one origin, and `pageSlot` re-derives
    `pageNeighbour` + `decode` + an origin compare on each; because a page
    is 8x8x4, most of those directions land on ordinal 13, so caching the
    resolution per distinct ordinal looked like it should delete most of
    the walks (up to 8 children x 18 directions = 144 per row above L0 in
    `finerAdjoint`). It is exact — `topology` is bound read-only, so the
    resolution is a pure function of (level, page, ordinal) — and it bought
    **0.02 ms**: the merged band went 2.796 -> 2.776 ms over two interleaved
    captures while `buildCandidateLevelSets`, which the change does not
    touch, moved by the same 0.02 in the same pair.
    The lesson is where the cost actually is. The page resolution is two
    loads of a single address that stays hot in L1 across all eighteen
    iterations, so it was already nearly free. What costs is the
    **q-dependent** tail of `pageSlot`: `brickRecord(l,q)`, the two
    brick-mask loads and the ranked-slot indirection
    `topology[rankedSlotsBase()+levelBase(l)+topology[record+3]+rank]` — a
    dependent chain of scattered loads, one per direction, that no
    origin-side memo can remove. Any future attempt on the §6.3 merged band
    has to attack that chain, not the page lookup.
   Keep the authored E=10 envelope.
   **The general lesson, which reprices several items in this document:**
   this frame is NOT dispatch-overhead-bound. Any change justified mainly
   by a dispatch-count reduction should be re-scored on measured kernel
   time before it is built; changes that reduce *work inside* kernels
   (C4's table, F4's hoists, C1/C2's parallelization) are where the wall
   actually moves.

10. **Deleting a storage round-trip between two kernels — including the
    §4.3 merged-band apply / band sweep fusion.** MEASURED 2026-07-28 and
    refuted. Fusing `SPGrid Section 6.3 · destination-owned merged band`
    into `smoothAtoB`/`smoothBtoA` removes 165 dispatches per advance and
    was argued Gate A from a rendered-shader diff: every float op, both
    18-channel loops, `max(0.0, diagonal-sum)`, `finerAdjoint` and all
    eleven report codes byte-identical, same row set, no other reader of
    `operatorImage`, no hazard. **The argument is wrong, and the way it is
    wrong generalizes.** A/A was 0 on both arms (uncontaminated
    instrument); A/B differed at 20 of 21 records in both pairings.

    Divergence starts at the first solve checkpoint, in `mgpcgControl`
    residual words only, at ~6e-5 relative — 173846.97 vs 173847.06,
    4.3749e-4 vs 4.3721e-4. By generation 502 that has grown into a
    different simulation: 212,981 vs 230,785 fine samples, 4,075 vs 4,096
    active pages, 1,473 vs 1,470 rows, and **peakLiquidSpeed 7.826915 ->
    8.046456**, which is the intentionally-red gate the protocol says must
    not get redder. Note that is the *identical* end value the reverted
    Part D reduction-tree patch produced; slightly-different rounding in
    the operator appears to fall into the same alternative trajectory.

    **The mechanism: the round-trip IS a rounding step.** Storing
    `applyRow`'s result to an f32 storage buffer and reloading it forces a
    round to f32 and ends the expression. Computing it inline leaves the
    value in a register, where the backend contracts the multiply-add the
    split form could not. This is the same effect the `applyRow`
    row-x-channel experiment hit from the other direction — staging terms
    through workgroup memory turned one fused multiply-accumulate into a
    bare multiply plus a bare add and moved peak speed 7.8269 -> 7.5066.

    So: **on this backend, removing (or adding) a memory round-trip between
    two kernels is never a restructuring-only change.** Any fusion or
    de-fusion across a storage buffer must be proposed as Gate B with the
    envelope run, never as Gate A, however identical the two shaders read.
    Worth salvaging from the attempt regardless of the fusion: it extracted
    `applyRow` and its addressing into shared WGSL, deleting a ~4.7 KB
    duplicate transcription of the §6.3 addressing that the §4.3 shell
    carried. That deduplication is a correctness win and can land alone.

---

# Part A — groundwork

## A1. Turn off shader instrumentation in the product default

`lib/stores/performance-instrumentation-store.ts` currently initializes
`mode: "activity", enabled: true, shaderActivityEnabled: true`
("Detailed profiling is the product default"). In activity mode,
`createGPULogicalActivityAdoptionContext` rewrites every entry point of
adopted modules — including all 12 MGPCG entry points — inserting an
enter/exit heartbeat that performs a global `atomicAdd` on a single
counter plus a 16-word record store per sampled workgroup, and a
`.replace(/\breturn;/g, …)` that injects exit records on every early-out
path. Change the initial store state to `mode: "off"`. The Dawn harness
sets its own state explicitly (`usePerformanceInstrumentationStore
.getState().setEnabled(performanceTraceRequested)` in
`tools/run-webgpu-smoke.ts`), so benchmark behavior is unchanged; the
browser product stops paying per-dispatch atomics inside the solve.
Update whatever UI/store tests assert the old default.

## A2. Make the profiler's attribution complete

Two defects make the current reports untrustworthy and block the
regression-capture artifacts:

1. `POWER_DAM_COMPUTE_PASS_OWNERSHIP` in
   `tools/power-dam-performance-report.ts` is a closed label→owner table
   with no fallback; ~78 of 169 passes per advance carry labels not in
   the table and are reported "unattributed" (they include most of the
   hot path: the Section 4.3 shell passes, the A2 class passes, the §5
   march, the air-support identity publication). Run any profile with
   `--json`, list the unowned labels, add ownership rules for every one.
2. `mgpcgDispatchesPerAdvance` matches pass labels that no longer exist
   ("Octree MGPCG solve"), so the report prints "MGPCG: 0.0 dispatches"
   while the solve is ~75 % of all dispatches. Re-key it to the current
   MGPCG pass labels.

Also note for anyone adding labels: `PassBroker.compute(descriptor)`
**ignores the descriptor when a pass is already open** — the label that
matters is the one on the call that *opens* the pass. Labels passed by
later dispatches in the same pass are silently dropped. If a phase needs
its own attribution, it needs a `fence()` before it, or its label must be
on the opening call.

## A3. Promote the silent failure modes to hard gates

Today every one of these is observable only through a ~250 ms telemetry
poll or not at all. Add them to the smoke harness's per-step audit
(`tools/run-webgpu-smoke.ts`) and the benchmark summary
(`tools/benchmark-power-dam.ts`) as run-failing assertions:

- **Topology rollback:** the fine topology control word already encodes
  `rolledBack` (decoded in `applyGlobalFineDiagnostics`). Assert it is
  zero for every accepted generation during benchmark/acceptance runs.
- **Unaccepted restriction rows:** `fineRestrictionControl[3]` is
  written GPU-side by `publishRestriction`
  (`select(diagnosticCounts[0], 0u, sourceRejected)`) and already read
  back in the QA path. Assert zero per generation.
- **Solver convergence:** the MGPCG control buffer's converged/error
  words (the same 64-byte segment the step-snapshot ring already
  copies) must show converged for every step that executed iterations.
  This is the guard for the seed-publish cliff described in "do not
  implement" item 3.
- **Terminal pressure iterations > 0** on churn lanes, and **fine-band
  active count ≠ 0xFFFFFFFF** (read from the worklist header — note the
  count is header **word 1**; a prior consumer bug read word 0, the
  generation, and printed nonsense).

## A4. Widen the step-snapshot ring; fix its capacity policy

`StructuredStepSnapshotRing` (`lib/structured-step-snapshot.ts`) is the
one correct readback mechanism in the codebase: 3 slots × 336 bytes,
each record copied by the *step's own encoder* as its last commands, so a
mapped record is torn-free and step-final; `readLatest` marks a slot
"mapping" synchronously before `mapAsync`, and the producer skips slots
that are mapping. It is the substrate for every host-side scheduling
decision in Part B. Changes:

1. **Record layout** (extend `lib/structured-authority-audit.ts` and the
   copy list in `webgpu-uniform-eulerian.ts`): append copies of
   (a) the topology-epoch state buffer (64 B — `ready`, `error`, the
   clean-commit token; from `webgpu-octree-topology-epoch.ts`), (b) the
   SPGrid per-level `levelDelta` words (generation/dirty per level),
   (c) the air-support producer's `scratch[0..1]` failure words, (d) the
   fine-transport `governor` words 0–3 (schedule validity + active
   substeps). The MGPCG control words and fine worklist header are
   already in the record. Every source buffer's end-of-encode value is
   the step's final value because the ring copy is encoded last — keep
   that invariant when adding copies (append after all producers).
2. **Slot count:** the browser keeps up to 8 advances in flight, and a
   record's map resolves only after its step completes. With 3 slots, a
   per-step consumer can find all slots mapping; the producer then
   *skips* the record — and a skipped record means the required
   `step-snapshot` stage is missing from the step program, which
   **latches a permanent sequence fault**. Make the slot count
   `maxPendingAdvances + 1`, or give the conductor a
   single-outstanding-map discipline. Either is fine; pick one and
   assert the producer never skips.
3. **Decode fix:** the fine active-brick count is worklist header word 1
   (see A3).

## A5. Version the contracts that currently forbid host scheduling

Three places assert "the host never shapes an advance" in absolute
terms. Part B breaks that deliberately (for launch *shape*, never
physics values), so amend the contracts first:

1. `lib/physics-step-program.ts` — the step program validates a fixed
   stage list per advance; a skipped stage is a latched deviation, and
   `optional` is reserved for conditions "the driver does not control".
   Add **conductor-conditional stages**: a stage may declare a predicate
   id; the recorder accepts its absence when the driver records that
   predicate true for the step, and stores the predicate value in the
   step record.
2. Add a **prediction record**: for each step the driver records the
   encoded solve budget and the list of skipped stages with the snapshot
   step each predicate was read from. When snapshot N later maps,
   compare its GPU counters (executed iterations, epoch verdict, delta
   counts) against the prediction for step N. Mismatch = hard failure in
   the harness, ALERT in the browser diagnostics card. This is the
   mechanism that keeps every Part-B skip honest: a skip may only ever
   delete work the GPU itself reports as zero.
3. `docs/PHYSICS_STEP_SEQUENCING.md` — the driver contract text
   ("drivers are forbidden from affecting step content";
   `hostSchedulingUsesReadback: false`) gets the carve-out: drivers may
   delete provably-zero-work encodes from snapshots; they may never
   change a value the physics reads, and any predicate that could skip
   live work must be structured so staleness makes it *conservative*
   (encode more, never less).
4. `lib/structured-authority-audit.ts` requires the structured epoch and
   fine generation to advance every step. This forbids the frozen-step
   settled tier (B5b). Design an authored "settled epoch" record kind:
   a step may declare itself settled, in which case the audit checks
   that generations are *unchanged* and that the settled predicate's
   snapshot evidence is present. Do this as its own reviewed change —
   it relaxes a safety net.

Also known and worth fixing while in there: the sequence-neutrality
clause ("a traced advance submits the same command graph as an untraced
one") is currently false — tracing threads `productionBoundary` through
`webgpu-octree.ts`, which inserts extra fences and swaps the `PassBroker`
at ~11 sites. Either make the boundary hooks no-ops that ride existing
pass breaks, or amend the clause; today's docs claim something the code
does not do.

## A6. Capture the missing baselines

`npm run capture:octree-regression-{mini,ui,quiescent,moving-interface}`
in a clean tree (A2 unblocks the unowned-label blocker), committed with
empty `blockers`. Additionally capture one *traced* profile of the
settled window (mini steps 501–560): every existing per-pass artifact is
a churn window, so no settled attribution exists at all.

---

# Part B — encode gating and deletions

## B1. Stop encoding the recurring fine-seed chain

**What exists.** Every advance, `encodeSurface` in `lib/webgpu-octree.ts`
calls `globalFineSeeds.encode(...)`, which encodes an 8-dispatch chain
(`clearSeedState`, `classifySourceBlocks`, `scanSourceBlocks`,
`emitSourceRecords`, `sortSeedRecords` — a single-workgroup bitonic sort
of 8,192 records with 91 barriers — `classifySeedRuns`, `scanSeedRuns`,
`emitSeedRuns`). On every recurring step the output is unread: all four
consumers (`externalSeedPhi`, `externalAffineInterfaceBrick`,
`insertExternalSeeds`, and `initializeDesiredSamples` via
`externalSeedPhi`) begin with an early-out on
`currentFinePopulated()` (`currentFinePublished() && worklist[1] > 0`),
and the seed buffers are bound in no other module.

**Change.** Maintain a host boolean `finePopulated`, set when the
bootstrap fine publication is accepted (the host already distinguishes
the `bootstrap` vs `delta` publication kinds when it encodes). When set,
do not call `globalFineSeeds.encode` on the recurring path. Keep the
`fineSeedAdapter` bootstrap encodes untouched, and keep encoding the
chain during any window where a publication has not yet been accepted
(the code comments in `encodeSurface` describe the pre-acceptance retry
window — preserve it).

**Why the predicate is safe at any staleness:** populated is monotone
after bootstrap (a rejected publication retains the previous populated
generation; teardown is the only reset). A stale *false* merely encodes
the chain harmlessly; a true can never become false. Gate A applies and
must show bit-identical fields.

## B2. Encode the air-support publication once — BLOCKED, DO NOT ATTEMPT

**Investigated 2026-07-27 and refuted. The premise below ("site 2 exists as
insurance") is false.** Site 2 has a live same-step consumer:
`transferStructuredTopologyCandidate`
(`lib/webgpu-octree-structured-dynamics.ts`) binds the air-support arena and
reads the publication on its primary path via `extendedOwnerVelocity` →
`taggedVelocity`, gated by `supportPublicationValid()`, which is satisfied at
that point — so the read is live, not dead. An in-file comment records that
this input was measured at **+45 % mechanical energy** when it is wrong.

Nor are the two encodes the same bytes: `seedAirSupportFaces` reads the
structured authority face values, which this step's advection / forces /
projection rewrite between site 1 and site 2, and the two encodes pass
different fine slot / generation (site 2 runs after settlement flipped the
current bank). Site 1 is separately consumed by fine transport and
`encodeAdvection`. Both encodes are load-bearing for different consumers
needing different field states.

Worse, B2's own ordering requirement is unsatisfiable. Three constraints all
hold: site 2 must precede the transfer candidate (which needs the fresh
post-projection field); the transfer candidate must precede reconstruction,
which must precede validation (`validateInactiveTopologyEpoch` reads the
structured candidate control that reconstruction writes); and B2 requires
validation to precede site 2. That is a cycle.

Also correcting the brief: the "clean-commit token" `epoch.reserved[0]` is NOT
available at the tail of step N — it is written by `beginReadyTopologyCommit`
at the next step's head. Only `ready` / `error` / `readyGeneration` /
`activeGeneration` are same-step-fresh. The gate predicate, if it were ever
reachable, is `error==0 && ready==1 && readyGeneration!=0 && age!=0 &&
age<0x80000000` with `age = readyGeneration - activeGeneration`.

**The ~7.5 ms is real but must be attacked as E3** — make one encode cheap
(widen the 768-thread march, widen the 17-dispatch identity chain, cut the
~127 MB arenas) — not by deleting an encode. The machinery for a gate exists
if the cycle is ever broken: `beginAirSupportPublication` already has a
preserve branch that zeroes the four publication indirect records and returns
before any arena write, and every downstream stage honours it.

### Original (refuted) text follows

**What exists.** The Section 5 air-velocity-support build
(`webgpu-octree-air-velocity-support-gpu.ts`) is encoded **twice per
advance** from `lib/webgpu-octree.ts`: site 1 inside the ready-topology
flip, site 2 after `encodePendingFineSettlement` (gated
`powerTimestep_s > 0`). Each encode is 5 passes / 34 dispatches: a
17-dispatch identity/candidate/prefix/scatter chain, a 12-wave Jacobi
extrapolation prefix, then `marchAirSupportFacesToFixedPoint` (a
768-thread persistent fixed-point march) and reconstruct/commit. Site 2
exists as insurance: if the *next* step's topology flip rejects the
candidate epoch, consumers fall back to the most recent publication, and
site 2's refresh keeps that publication no more than one generation
stale.

**Why a host-side skip is wrong.** The verdict site 2 insures against is
produced by `validateInactiveTopologyEpoch` at the **tail of the same
step** (after site 2's current encode position), then merely re-checked
by `beginReadyTopologyCommit` at the next step's head. Any host decision
made from a previous step's snapshot removes the insurance in exactly
the step whose candidate fails.

**Change.**
1. Move site 2's encode to *after* the candidate-validation encode in the
   same step, so the epoch verdict words exist in GPU memory when site
   2's work would run.
2. Bind the topology-epoch state buffer into the producer's existing
   1-workgroup `prepareAirSupportFaces`-style prepare stage, and have it
   zero the producer's indirect dispatch args when the verdict is clean
   (`ready == 1` and the clean-commit token set). The producer already
   drives most of its dispatches through an indirect args buffer
   refreshed by that prepare stage — extend the same mechanism.
3. Four dispatches in the chain are fixed-size
   (`dispatchWorkgroups(ceil(domainVolume/wgSize))` clears/marks).
   Either source them from the same zeroed indirect args, or give their
   kernels a first-instruction early-out on the verdict word. Both are
   acceptable; indirect is cleaner.
4. Do not touch site 1, and do not touch the Jacobi prefix (see "do not
   implement" item 4).

**Verification.** Gate A on the normal path. Additionally, force a
topology rejection (the smoke harness has fault-injection hooks used by
the stability-envelope tests) and assert the fallback publication still
occurs and the run fails closed exactly as before. On clean steps the
second march/identity/extrapolate chain must execute zero workgroups —
assert via the A5 prediction record.

## B3. Give the coarse-φ schedule a restriction-only entry point

**What exists.** `encodeCoarsePhiCorrection` in `lib/webgpu-octree.ts`
runs twice per advance: once on the recurring path with the real dt, and
once inside `encodePendingFineSettlement` with **dt = 0**. Both run the
full chain: the fine→coarse restriction (3 dispatches) then the coarse
schedule `prepare → advect → correct → publish → commit`
(`webgpu-octree-power-coarse-levelset.ts`). At dt = 0 the advect is
value-neutral by an explicit guard, but the second run is not pure
waste: it folds the post-redistance restriction into the coarse field,
republishes the **coarse delta directory** that `globalFineSummaries
.encode` consumes immediately afterwards, and recomputes the
`PHI_INTERFACE` flags and physical-volume tallies.

**Change.** Add an entry point to the coarse schedule that keeps
`correct → publish → commit` (including the delta-directory publication
and flag/volume recomputation) and drops `prepare`'s gradient staging
and the `advect` dispatch. Use it at the dt = 0 call site only. Gate A.

## B4. Author the solve budget instead of hardcoding 10

**What exists.** `planOctreeSolveTail` (`lib/octree-solve-tail-policy.ts`)
computes a scene-complexity score from immutable authored scene facts,
then discards it: the encoded outer-iteration budget is always
`OCTREE_SOLVE_TAIL_MAXIMUM_ENCODED_OUTER_ITERATIONS` (10). The solve
executes 3–5 and the remaining iterations run as zero-workgroup indirect
dispatches (the contract `convergenceTail: "gpu-zero-indirect"` requires
launches to stay and lanes to zero).

**Change.** Use the score: encoded budget
`E = clamp(sceneComplexityScore, 4, 10)`. Add one sticky safety valve:
if any step-snapshot reports non-convergence (the word A3 now trips on),
encode 10 for the next `maxPendingAdvances` steps before returning to
the authored value. Sticky-increase-only is the property that makes the
valve safe at any snapshot staleness.

**Tests to renegotiate by name** (`tests/octree-webgpu.test.ts`): the
assertion that the encode contains no `updateSolveBudget` (bans
observation-driven mutation — keep banning it; the valve only ever
*raises* the budget toward the old constant) and the construction-time
`iterationBudget` check (now equals the authored score, not the
constant). Verification: Gate B, plus assert per-step *executed*
iteration counts are identical to the parent run — only encoded-but-idle
launches may disappear.

## B5. Settled tier

**B5a — without contract changes.** Add a conductor rule in the driver:
when the latest mapped snapshot shows executed-iterations == 0 AND the
fine page-delta count == 0, AND no host-side input changed this step
(bodies and inflow are `advanceTo` arguments — compare them), AND
`movingRigidBodyCount == 0` in the authored scene (bodies integrate on
the GPU, so a resting body can move with zero host input — force the
full tier whenever moving bodies exist), select a settled template:
the full graph minus the B1/B2/B3 gated work, with solve budget E = 1.
Entering late (stale snapshot) only costs savings; exiting is immediate
because host-input changes are known synchronously. Gate A over a
60-step settled window: fields must be bit-identical to the worst-case
encode.

**B5b — the frozen tier (requires A5.4).** Once the audit ABI has an
authored settled record kind: gate `[forces + divergence + solve +
projection]` as one unit and the fine-surface chain as another. They
must gate *together* because gravity is applied by the forces stage and
cancelled by projection every step — skipping only the solve injects
g·dt of momentum per frame; skipping the whole unit leaves every buffer
unwritten, which cannot drift. Before enabling, run the settling
experiment: a 60-step settled window with the unit frozen vs encoded
must produce bit-identical fields; if it does not, the fixed point is
not stable and the tier stays at B5a.

## B6. Remove the two provable fences

In `webgpu-octree-structured-velocity-gpu.ts`, remove exactly the two
fences between `classify` and the `section63` consumer that the in-file
comment already marks as theoretically unnecessary ("on paper these need
no boundary…"). Leave all others; in particular the fence after
`finalize` guards the storage-write → indirect-read hazard on
`liveRowDispatch`. Gate A. (This is hygiene, kept because the comment
records a measured-under-1 ms collapse attempt that coincided with an
unrelated rejection — retire the superstition properly, with Gate A
evidence.)

---

# Part C — widening the serial kernels

The codebase's standard parallel shape already exists in
`lib/webgpu-octree-fine-levelset-topology.ts`: a classify dispatch
(64–256 lanes over items), a block prefix-sum
(`scanSparseSeedRecords` / `scanSparseGroups` / `scanSparseSuperGroups` /
`offsetSparseGroups` / `offsetSparseRecords`), and a scatter/compact
dispatch. Every change in this part is "replace a single-thread loop
with that shape, preserving the exact output byte layout and ordering".
Ordering note: all these serial loops iterate pages/slots in index
order; a deterministic scatter keyed by the same index preserves the
output ordering, which is what makes Gate A achievable.

## C1. The fine-transport publication trio

In `lib/webgpu-octree-fine-levelset-transport.wgsl.ts` three entry
points are `@workgroup_size(1)` dispatched `[1,1,1]`:

- `publishStructuredFineTransportWorksets` — two serial loops over all
  (~4,096) live pages, classifying each into one of 4 workset classes
  (2 of which are permanently empty — see C5) and building the class
  headers + page lists.
- `summarizeStructuredFineTransport` — one serial loop over page
  statuses (3 dependent loads each) producing the transport summary and
  retaining the first-rejected position.
- `publishStructuredFineDelta` — a serial clear of `8 + 2·pageCapacity`
  words followed by a serial compaction of changed-page keys into the
  delta stream with a count in `delta[0]`.

Replace each with classify → prefix → scatter. The delta stream's order
(ascending page index, as the loop produces today) and the class-header
layout must be byte-identical. The summary's "first rejected position"
becomes an atomicMin over (position) — same result, order-free. Gate A.

## C2. The recurring band scatter

`publishRecurringSparseBand` (same file family, dispatched as ONE
256-lane workgroup) walks the interface-seed list and performs a cubic
`(2·rings+1)³` = 15³ = 3,375-tap `atomicOr` scatter per seed into the
desired-brick membership mask. Replace with an indirect dispatch over
(seed × halo-offset): thread = (seedIndex, offsetIndex), one `atomicOr`
each. The seed count is already available in the compacted seed header
the current kernel reads; halo volume is a compile-time constant.
`atomicOr` into a mask is idempotent and order-free, so Gate A holds.

Related, same family: `changedNeighborRadii` (called per resident page
from `classifyAffectedPages`) linearly scans the whole changed-key
stream (≤8,192 entries) per page to compute Chebyshev distances. Invert
it: scatter from changed keys into per-page min-distance/flag words
(atomicMin), then classify pages from their own word. Same output
values; Gate A.

## C3. Collapse the four class-apply solve kernels

In `lib/webgpu-octree-spgrid-vcycle.ts` the five entry points
`applyRegularInterior`, `applyTransitionInterior`,
`applyPhysicalBoundary`, `applyTransitionBoundary`, `applyMergedBand`
are textually identical except a workset-class literal 0–4; all call the
same `applyRow`. A full A2 apply today encodes 1 gate dispatch + 4
class-indirect dispatches; `applyMergedBand` proves the alternative — a
single indirect dispatch over a union workset (class 4), which the
band-workset compaction already emits alongside the per-class lists.
Change `encodeAccurateWorksets` to encode the single union-workset
dispatch. Per-row behavior is identical because `applyRow` never
branches on the class — the class only selected which list the row came
from. Tests to renegotiate by name: the "four disjoint row classes"
assertions in `tests/webgpu-octree-spgrid-vcycle.test.ts` and the
matching structural check in `tests/octree-webgpu.test.ts`. Gate A.

## C4. Replace the coefficient-direction scan with a table

`coefficientForDirection` in `spgrid-vcycle.ts` finds which of the 18
stored Section 6.3 channels corresponds to a world direction by scanning
all 18 candidates, evaluating `worldDirection` (a 6-way permutation +
3 sign selects) per candidate. It is called from `finerAdjoint`'s
8-child × 18-direction loop — up to 144 calls × 18 scan steps per
coarse row per apply — and from a second site in the Section 4.3
preconditioner's band dilation. The channel index is a pure function of
`(metric.transformAndFlags & 63, direction)`. Precompute the inverse
table host-side (mirror `worldDirection`'s math in TypeScript over all
64 transform codes × 18 channels), producing 1,152 entries; bake it as
a WGSL `const` array (or a small uniform buffer if const-array size is
an issue), and replace both scan sites with one indexed load. Gate A —
it is a pure lookup replacement; add a unit test that exhaustively
checks table[code][dir] == scan result for all 1,152 combinations.

## C5. Delete the dead transition-transport entry points

`lib/webgpu-octree-fine-levelset-transport.ts` compiles pipelines only
for `transportRegularCommonPhi` / `transportRegularRarePhi`, but the
WGSL module still carries `transportTransitionCommonPhi`,
`transportTransitionRarePhi`, their body functions and departure
helpers, and the workset machinery still allocates and prefix-sums 4
class headers of which the two transition classes are permanently
empty. Delete the dead entry points/functions and shrink the class set
to 2 (this simplifies C1). Gate A.

## C6. Parallelize the multigrid candidate rebuild

In `spgrid-vcycle.ts`, the topology-candidate build runs ~21 dispatches
of which `buildCandidateLevelDeltas` is `@workgroup_size(1)` at
`[1,1,1]` (a single GPU thread looping all levels × slots with ≤256-probe
`cInsert` open addressing — up to millions of serial iterations when
topology churns), and six more phases (`buildCandidateLevelSets`,
`insertCandidateGhosts`, `scanCandidateTransfers`,
`linkCandidateParentChains`, `rankCandidateBricks`,
`compactCandidatePages`) run at `[levelCount,1,1]` with
`@workgroup_size(1)` — one thread per level. The settled case is already
cheap (the `probeCandidateSkip` gate retires clean levels); this change
targets churn.

Parallelize per level: one lane per slot/row with block prefix sums for
the compactions; hash insertion (`cInsert`) parallelizes because probe
sequences are per-key independent, but *insertion arrival order* into a
bucket is not deterministic under concurrency. Preserve determinism by
two-phase insertion: phase 1 claims slots with `atomicCompareExchange`
keyed on the cell key (arrival order irrelevant — the winner is keyed,
not ordered); phase 2 writes payloads. If any consumer depends on
bucket-scan order rather than key identity, fix that consumer to key
lookups (grep for the `cLookup` scan). Prefer Gate A; if reduction to
bit-exactness proves impractical because slot indices differ, fall back
to Gate B plus a stronger structural check: the *accepted topology*
(level sets, ghost sets, transfer lists as key sets) must be identical
per step vs the parent run.

---

# Part D — IMPLEMENTED; 2026-07-29 CLEAN A/B SUPERSEDES THIS RESULT

**Historical result; do not use it for current selection.** The kernel described below
was implemented in `lib/webgpu-octree-persistent-mgpcg{,.wgsl}.ts` and wired
into `WebGPUOctreeProjection.encode`. It works: on the mini churn lane it
collapses the solve from **1,125 → 413 dispatches** and **168 → 78 compute
passes** per advance, executes the same 4/10 iterations, and produces zero
validation errors. Interleaved A/B, same binary, flag on vs off:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| hierarchical | 79.74 | 79.68 | 79.68 |
| persistent | 94.79 | 94.90 | 94.79 |

That isolated experiment measured **+15.1 ms**, but a clean current-tree A/B
shows that its conclusion does not transfer to small live systems. On the
ceiling case (roughly 459 live pressure rows), the hierarchical command graph
cost 113 ms/10 advances and the persistent executor cost 53 ms/10. On the
mini case the corresponding result was 107 -> 58 ms/10. One workgroup is one
GPU core, but for these small systems eliminating the fixed launch graph wins.

The historical interpretation was: one workgroup gives up ~31/32 of the
machine to buy back dispatch overhead this frame does not pay — the same
premise the B4 experiment independently refuted (deleting 231 dispatches per
advance measured as a 2 ms *regression*). The 862-dispatch/91-pass count was
never the cost; it was the symptom of a wide problem being solved widely.

The current production policy is default-on whenever the executor was
constructed (capacity <= 16,384). `FLUID_OCTREE_PERSISTENT_MGPCG=0` is the
explicit full-hierarchy A/B. The level-zero band also retains one extra
redistance/restriction shell; this repaired the pre-existing step-19
interface-coverage failure exposed by the persistent trajectory without
doubling the transported band.

**The reduction-tree patch was reverted with it.** Replacing `subgroupAdd` with
an explicit width-halving tree existed only to make the two paths bit-comparable.
Measured alone (Gate B) it reddened the physics: peak speed 7.827 → 8.046 m/s
(the intentionally-red gate the protocol says must not get redder), projected
residual 3.10e-6 → 3.69e-6, plus three new failures — fine transport invalid,
and top-two-layer-wet at t=1.60 and t=2.00. Retention and volume drift were
unchanged, and projection energy ratio and dominant-component fraction actually
improved, so this is a genuine reassociation trade rather than a bug — but with
the persistent path unselected there is no benefit to buy it with. Restore the
tree only alongside a persistent path that has been measured to win.

### Original (unrealised) design follows

## Part D — the persistent small-domain pressure solve

**What exists.** One solve = 862–890 encoded dispatches across ~91
compute passes: an outer CG loop (fixed budget, B4) where each iteration
encodes ~77 dispatches — the Section 4.3 correction (k=8 boundary-band
Jacobi sweeps → first-order V-cycle → k=8 symmetric sweeps ≈ 66
dispatches, of which the 5-level V-cycle is 26, twelve of them
one-workgroup coarse-level launches), an A2 apply, and two reductions
each split into a wide partial pass and a 1-workgroup finish
(`reduceMergedPartials`/`finishMergedReduction`,
`reduceDirectionCurvaturePartials`/`finishDirectionCurvature`) with
storage↔indirect fences between. The system being solved has ~1,470
unknowns. Neighbor addressing inside `applyRow` resolves each of up to
18 neighbors through `pageSlot`, a 5–7-deep dependent-load chain, and
`finerAdjoint` adds up to 152 more chains per coarse row.

**The change: implement `persistentMGPCG`** — the contract symbol
already declared in `webgpu-octree-section43-contract.ts` and
implemented nowhere. One dispatch, one workgroup of 256 lanes, runs the
entire solve for row counts up to an authored threshold (start: 8,192 —
covers both production lanes with headroom).

Design requirements, each of which was independently verified feasible:

1. **Algebra unchanged.** The kernel runs exactly today's iteration:
   k=8 band sweeps → first-order V-cycle over all 5 levels → k=8
   symmetric sweeps as the preconditioner application, A2 applies, CG
   vector updates, two dot products per iteration, same tolerances,
   same fail-closed error semantics. Coefficients come from the existing
   `section63Coefficients` double-banked store; addressing initially
   stays `pageSlot` (Part F upgrades it — do not couple the two
   changes).
2. **Synchronization.** Phases are separated by
   `storageBarrier(); workgroupBarrier();` — with a single workgroup,
   workgroup-scope acquire/release fully orders storage traffic between
   phases. Every loop exit (convergence, error, budget) must branch on
   a `workgroupUniformLoad` of a flag word so control flow stays
   uniform. The shipping in-tree proof of this exact pattern is
   `marchAirSupportFacesToFixedPoint` (persistent 256-lane loop,
   `storageBarrier` + `workgroupUniformLoad` exit); copy its structure.
   Millisecond-scale single dispatches are far under Metal's watchdog.
3. **Work distribution.** Rows striped across lanes
   (`for (row = lane; row < rowCount; row += 256)`), per level. The
   coarse V-cycle levels (capacities 128/16/2) leave most lanes idle
   for microseconds — that is fine and still beats twelve dependent
   dispatch launches. The band sweeps iterate the band workset (which at
   these domain sizes is nearly all rows — also fine).
4. **Bit-exact reductions.** Today's dot products are: per-row
   compensated products → per-subgroup `subgroupAdd` → an explicit
   width-halving shared-memory tree over the workgroup's slots → one
   partial per 128-lane workgroup (12 live partials at mini) → a
   1-workgroup strided sequential merge with the same tree. Reproduce
   this bit-for-bit inside the persistent kernel by emulating 128-lane
   virtual workgroups (lanes 0–127 compute the same partials the
   separate dispatches did, sequentially over the 12 virtual groups),
   then the same merge. Additionally replace `subgroupAdd` with an
   explicit shuffle/shared-memory tree in **both** the persistent kernel
   and the retained hierarchical path — `subgroupAdd`'s association is
   implementation-defined, and pinning the tree in both paths is what
   makes Gate A applicable to the whole change.
5. **Bindings.** The kernel wants the 7 CG row-vectors, the V-cycle
   state arena (26 channels), coefficients, topology/descriptor tables,
   band worksets, and control words — more distinct buffers than the
   portable per-stage storage-buffer limit. Consolidate the 7 CG
   vectors into one arena buffer with fixed row-capacity-sized offsets
   (they are all f32, all rowCapacity-length) and fold the control +
   dispatch words into one. Aim for ≤ 8 storage bindings. Do not raise
   device limits; the 10-buffer ceiling has bitten this codebase before
   and `skip_validation` turns the error into a SIGSEGV.
6. **Outputs.** Write the same MGPCG control words (executed
   iterations, converged, error) to the same buffer — the snapshot ring
   (A4) and tripwires (A3) then work unchanged.
7. **Selection.** The encode chooses persistent vs hierarchical by row
   count against the authored threshold. The hierarchical path remains
   fully supported (it is the big-domain path and keeps C3/C4 and Part
   F improvements). Both paths share the reduction-tree pinning from
   item 4.

**Tests to renegotiate by name:** in `tests/octree-webgpu.test.ts`, the
encode assertion `doesNotMatch(/persistentMGPCG/)`; in
`tests/webgpu-octree-spgrid-vcycle.test.ts`, the "resolved-row
persistent executor is absent at every production capacity" block; and
the Section 4.3 persistent-correction deletion test — that one pins the
deletion of an *older, different* kernel (a 128-lane megakernel that
walked pages serially and measured catastrophically); keep banning that
specific shape (its `var<workgroup> persistentPage` page-walk), allow
the new row-striped kernel. Also revise the contract constant
`OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY` (64) — an artifact of
the dead design's one-row-per-lane layout.

**Verification.** Gate A (enabled by item 4). Additionally run a
lockstep A/B on identical published topology comparing per-iteration
residual words between the persistent and hierarchical paths — they
must match bit-for-bit before the persistent path becomes the default
for small domains. Keep the hierarchical path selectable by env for
A/B at any time.

---

# Part E — fine-surface lane

## E1. Fix the JFA stride ladder

`planFineLevelSetJFAStrides` is called by the redistance encode as
`planFineLevelSetJFAStrides(bandCells, bandCells)` — the second
argument (maximum displacement) is passed as the band width (23),
which triggers the round-up branch and yields the ladder
32,16,8,4,2,1 plus two +1 collar repairs = 8 flood passes, each a
27-tap gather over every support-marked sample. (An in-file comment
claiming "a 16-cell band starts at stride 16" is stale — fix it.)

Two steps, separately gated:

1. **Free fix (Gate A):** change the first-stride rule from "round band
   up to a power of two" to "smallest power of two such that the
   stride-sum covers the band": for band 23 that is 16+8+4+2+1 = 31 ≥ 23
   ⇒ ladder 16,8,4,2,1,+1,+1 (7 floods). Keep both +1 collar repairs —
   they fix a recorded boundary failure (the "generation-280" note in
   the file). JFA seeded from scratch each generation only needs
   coverage, not the power-of-two-of-band start.
2. **True-displacement ladder (Gate B, blocked on a warm start):** the
   per-step interface displacement is bounded by the transport bound
   (2m = 8 cells), so a warm-started JFA only needs 8,4,2,1,+1,+1 = 6
   floods. Today `seedClosestPoints` cold-seeds from φ sign changes
   every generation, so this is invalid until the closest-point field
   is carried into the next generation's page transaction (add the CP
   channels to the fine-page carry/commit path). Implement the carry,
   then the ladder.

## E2. Stage the page directory for transport (and the JFA gather)

Every fine-transport sample resolves its owner/page repeatedly during
the m-substep backtrace: `transitionSample` calls
`airOwnerAtPosition` per substep, `regularSampleExact` does 8 corner
lookups each with its own owner resolution and tagged-velocity fetch,
and the terminal `sampleFine` does 8 more sparse double-indirections —
every one a dependent pointer chain through the page directory in
global memory, no reuse across the 64 samples of a brick that share the
same neighborhood.

Change: at workgroup start, cooperatively load the page/owner directory
entries for the brick's 7³ logical neighborhood (radius 3 covers the
9-fine-cell reach at 4-cell bricks; ~343 entries, ~1.4 KB) into
`var<workgroup>` arrays; resolve all per-sample owner/page lookups from
shared memory. Field data (φ, velocities) continues to be read from
global — the neighborhood's field footprint is L2-resident and staging
it does not pay (see "do not implement" item 2).
`commitStructuredFineTransport` already stages a 6-neighbor directory
the same way — generalize that code. Apply the same staging to the JFA
flood kernel's 27-tap gather (it already caches per-page sample
indices; extend to the directory). Address staging changes no
arithmetic and no evaluation order: Gate A.

## E3. Restructure the air-support producer — topology/value split landed

**Landed 2026-07-29.** The two same-epoch publications now share immutable
support identities, tags, owner directory, resolved catalog topology and face
adjacency. Reuse is decided on GPU from the preceding VALID receipt, accepted
epoch/bank/boundary and capacity; an invalid or partial receipt rebuilds. Only
generation-dependent fine flags and velocity-dependent seed/march/vector work
are refreshed. Fresh catalog resolution is dispatched over support capacity,
not the much larger candidate capacity. Set
`FLUID_OCTREE_AIR_SUPPORT_TOPOLOGY_REUSE=0` for the full-rebuild oracle.

The inner closest-face ordering now carries squared Euclidean distance, removing
a square root from every candidate visit and recovering physical distance once
per reconstructed row. The schedule is 12 prefix + 12 wide waves with the
three-axis persistent kernel as the exact tail. On the 20-frame ceiling run the
final fine/raster hashes remain bit-identical; dispatches fell 10,121 -> 7,962
after shortening the envelope.

**Still open:** support membership is still almost the whole air partition, so
both publications still march about 94k face patches. The next structural win
is a compact exact seed-to-demand corridor/frontier; topology reuse alone does
not solve the domain-wide graph expansion.

Even encoded once, the air-support build is a heavy chain with three
structural problems:

1. **The march is 768 threads.** `marchAirSupportFacesToFixedPoint`
   runs the whole fixed-point relaxation on 3 workgroups. Restructure
   as: wide per-layer sweeps (indirect dispatch over the face-row
   worklist, one lane per face-slot) for the first N layers — the
   existing 12-wave Jacobi prefix is exactly this shape, so this is
   "raise the prefix and widen it" — with the persistent 3-workgroup
   march kept only as the convergence tail. The copy rule
   (`betterFace` selects min (layer, item)) is order-free, so widening
   preserves results.
2. **The identity chain is 17 mostly-narrow dispatches** (clears,
   candidate emission, mark/scan/prefix/scatter, tag resolution).
   Apply the Part-C recipe: classify/prefix/scatter shapes at
   64–256 lanes, and fold the three separate clear dispatches into
   their producers' first phase.
3. **The arenas are the largest allocation in the engine** (~127 MB:
   ordinary-face ping/pong sized `(R + 36R) × 12 slots × vec4`,
   adjacency `(R+36R) × 55 words`, plus candidate scratch). The 36×
   multiplier vastly over-provisions actual face incidence (the
   catalog's own cap is `maximumFaceIncidence = 30` per row, and real
   incidence is far lower). Re-derive the arena sizes from the catalog
   cap, drop the second bank where the consumer only ever reads the
   committed bank, and size candidate scratch from the measured
   candidate counts (published in the producer's control words). This
   is also the prerequisite for any domain larger than ~115³ — the
   current allocator scales at ~42 KB per pressure row.

Gate B (eligibility logic untouched ⇒ aim for Gate A on 1–2; the arena
resize is Gate A trivially), plus B2's forced-rejection test.

## E4. Size the support-shaped dispatches from the changed set

The JFA "support" dispatches and several topology dispatches are shaped
by the support/dirty page masks, which the halo radii saturate at
today's domain sizes; the *mechanism* (indirect args from compacted
page lists) is already in place. Audit every dispatch sized by
`maximumResidentBricks` or `rowCapacity` (grep those symbols in the
fine-lane modules) and convert the capacity-shaped ones to the
compacted-count indirect form. No behavior change (Gate A); this is
what makes all of Part E scale with interface area on larger domains.

---

# Part F — operator storage and the big-domain path

## F1. Bind the Section 6.3 coefficient catalog

The paper's §6.3 stencil table already exists in this repo:
`catalog.coefficientData` — 19 coefficients per topology case, all
1,608 cases, bit-exactness asserted by
`tests/octree-power-catalog-artifact.test.ts` — is uploaded as
`catalogCoefficients` by `webgpu-octree-power-topology.ts` **and bound
to no pipeline**. Meanwhile every topology epoch, the runtime
re-derives equivalent per-row data (up to 36 explicit neighbor row ids
+ 36 coefficients per row) in `rebuildStructuredBoundaryRows`, and the
operator gathers through those explicit indices.

Change, in two stages:

1. Make the operator's stencil *coefficients* come from the catalog:
   row → case id (already in the row metadata) → 19 catalog
   coefficients, scaled by the row's geometric factors (aperture /
   boundary scale — the geometry-dependent part stays per-row; the
   topology-determined part moves to the shared table). Delete the
   per-epoch coefficient derivation from the rebuild once the
   differential gate passes.
2. Make same-level neighbor *addresses* derivable rather than stored:
   within a page, neighbor offsets are arithmetic; across pages, the
   staged directory (E2's mechanism, applied to the solve's pages)
   replaces the per-neighbor `pageSlot` chain. This is what
   `applyRow`'s 19 chains and `finerAdjoint`'s 152 chains become: one
   shared-memory directory hit each.

Verification: a differential harness that applies both operators to the
same vectors on the same published topology and requires bit-identical
results (stage 1) before the rebuild is deleted; Gate A end-to-end.
This item is invisible at 16³ (everything is cache-resident) — its gate
is the F3 lane.

## F2. Page-blocked A2 for the hierarchical path

`smoothPage` in `spgrid-vcycle.ts` already demonstrates the pattern:
stage a page + halo (~12 KB of the 16 KB workgroup budget) once, run
multiple sweeps from shared memory. Apply the same structure to the
hierarchical A2 apply for large domains. Before building it, measure
page occupancy (live rows per touched page) from the `dispatchMeta`
words already read back — the staging only pays above roughly 21 %
occupancy; below that keep the gathered form. Test to renegotiate by
name: the A2-shader ban on
`pageSlots|applyAccuratePages|ownerHead|ownerNext` in
`tests/webgpu-octree-spgrid-vcycle.test.ts` (it pinned a cutover-era
decision). Gate A.

## F3. A lane where the scale story is measurable

None of F1/F2 (and little of E4) changes anything measurable at 16³.
Add a benchmark lane at ≥128³ coarse (the widened-tank harness in
`tools/benchmark-octree-leaf-sizes.ts` / `FLUID_BENCH_TANK_SCALE` is
the starting point). Prerequisite: E3's arena diet — the current
allocation model (~26.6 MB + ~42 KB per pressure row, dominated by the
air-support producer) exceeds 64 GB long before 128³. Reference target
for the pressure stage at scale exists in-repo:
`benchmarks/results/sparse-fluid-ocean-surface-2026-07-19.json`
(pre-cutover lane, 213,844 rows, pressure 8.32 ms/step ≈ 39 ns/row).

## F4. The structured-dynamics lane

After Parts B–E land, the advection/boundary/dynamics family kernels
become the largest remaining subsystem; nothing above touches them.
The same three mechanisms apply and should be executed as a follow-on
with its own measurement round: (1) collapse the per-family
micro-dispatch chains into class-merged indirect dispatches (the C3
mechanism — `encodePrepare` alone is encoded three times per advance);
(2) hoist repeated publication-header re-validation out of per-sample
hot loops (the `ownerAtCached` pattern already proves the hoist is
sound); (3) directory-stage the owner lookups in the staggered
samplers (the E2 mechanism). Keep the wall-row eligibility and carry
gates exactly as they are — they encode hard-won physics fixes
(`DELTA_CARRIED` semantics; the accepted row set is LIQUID-ONLY and an
interface face has `neighbor == INVALID` and must trace — see
`docs/STRUCTURED_DAM_BREAK_HANDOFF.md` traps).

---

## Verification wrap-up — what "done" looks like

- All Part-A tripwires active in every benchmark/acceptance lane and
  green at HEAD.
- Every Gate-A change: bit-identical 500-step fingerprints, and the A5
  prediction record shows zero mismatches between conductor predictions
  and GPU counters over the full run.
- Every Gate-B change: 500-step gate green, checkpoint physics inside
  the recorded brackets, per-step executed-iteration counts unchanged
  where the change claims launch-only effect.
- The four regression-capture artifacts recommitted after each part,
  with empty blockers, plus the settled-window traced capture.
- The pinned tests renegotiated only by name, each with a comment
  pointing at the relevant section of this document.
