# Uniform pressure 5× handoff — residual-driven CM11a

**Scope.** The dense `uniform` method's CM11a pressure solve
(`lib/webgpu-uniform-pressure-multigrid.ts` + `.wgsl.ts`) is 20.30 ms of the
33.25 ms clean advance on the Dawn mini dam break 64³ lane
(`npm run profile:uniform-mini-dam-64-xctrace`, archived at
`artifacts/xctrace-uniform-mini-dam-64`). Goal: ~5× frame-time reduction,
generalizing to every scene. This document is the analysis and the
implementation plan. Every claim carries a verdict: **CONFIRMED** (verified in
code or the archived trace), **ESTIMATED** (cost model, not yet measured), or
**OPEN** (settled by WP0).

**Direction constraints from Peter (do not re-litigate):**

- Do **not** optimize dispatch/pass counts as an end in itself — launches are
  cheap on M1 Max. The lens is *preventing needless work*.
- No per-scene tuning: no magic cycle counts, no dispatch shapes fitted to this
  scene. Mechanisms must be work-proportional on every scene.
- Structural do-less-work before cheaper launches (standing direction).
- Transport is not a target: density + velocity transport combined is 10.1% of
  attributed GPU work. CONFIRMED (trace).

---

## 1. Evidence base

All numbers are from `artifacts/xctrace-uniform-mini-dam-64` (captured
2026-08-12, scene `minimal-power-dam-break-64`, grid 64³, 2 retained frames).
Do not re-profile to re-derive them.

- Clean, non-instrumented wall: **33.25 ms/advance** (`summary.json .wall`).
  Full pass isolation distorts 4.17×; isolated GPU busy is 29.05/29.44 ms with
  occupancy 12.26%/11.96% over 1,811/1,812 passes. CONFIRMED.
- Attributed stages: CM11a pressure **20.30 ms (76.1%, 1,468 passes)**; FIM
  extension ×2 **3.43 ms (290 passes)**; density transport 1.64 ms; velocity
  MacCormack 1.05 ms; projection 0.22 ms. CONFIRMED.
- `mgSmoothColour` alone: **16.03 ms, 1,040 dispatches**. CONFIRMED.
- Final fine residual: **7.79e-5 s⁻¹** against the code's cited tolerance
  `UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE = 1e-4` s⁻¹ (TallCells' GPU
  absolute L∞ figure, `webgpu-uniform-pressure-multigrid.wgsl.ts:9`). CONFIRMED.
- FIM telemetry: 13 executed front updates against 64 encoded slots per
  invocation (`uniformFIMExecutedPasses`). CONFIRMED.

**Pass-count model (reconciles the trace, so the cost model can be trusted).**
64³ builds a 6-level hierarchy (66³, 34³, 18³, 10³, 6³, 4³ with halos;
`levelCount = log2(64)`). From `buildPlan()`
(`webgpu-uniform-pressure-multigrid.ts:294`), with the fixed schedule
3 Full-Cycles + 4 V-Cycles and 4 pre/4 post sweeps:

| Group | Passes | Derivation |
|---|---|---|
| Setup | 19 | topology+rhs (2), downsample ×5, extrapolate+bake ×6 levels |
| Full-Cycles ×3 | 1,020 | 340 each: 15 spine + 5 prolongate-assign + nested V-cycles (22+43+64+85+106) |
| V-Cycles ×4 | 424 | 106 each: 80 smooth + 20 residual/restrict/clear/min + coarse + 5 prolongate |
| Finish | 2 | parity copy + fine residual measure |
| **Total** | **1,465** | trace says 1,468 (± clears) — **model CONFIRMED** |

`mgSmoothColour` per level per advance: L0(66³) **112** passes, L1 160,
L2 208, L3 256, L4 304 = 1,040 exactly. In cell-writes that is L0 32.2M,
L1 6.3M, L2 1.2M, L3 0.26M, L4 0.07M — **~80% of smoother cell-work is the
finest level, carried by only 11% of the smoother's passes.** CONFIRMED
(arithmetic from the plan).

Cost split of the 16.03 ms smoother: 928 sub-finest passes ride the ~5 µs
small-kernel floor (≈4.6 ms); the 112 finest passes are ~100 µs each
(≈11.4 ms), consistent with 287k cells × ~20–60 B of texture traffic each at
memory-bound rates (ALU utilization is 8.2%). ESTIMATED (split), CONFIRMED
(total).

---

## 2. Findings ledger

### F1 — The schedule is open-loop: convergence is measured but never consulted. CONFIRMED

`buildPlan()` bakes a static dispatch list at initialize; `encode()` replays it
verbatim every advance (`webgpu-uniform-pressure-multigrid.ts:207-239`). The
projected fine residual is computed **once, after everything, for telemetry
only** (`mgMeasureFineResidual`, finish stage). Nothing — host or GPU — ever
skips work because the solve is done. A vestige of flag-gating exists
(`mgSmoothColour` reads `mgConvergence[1]` when `control.w & 2u`,
`.wgsl.ts:291`) but the host never sets that bit; it is dead code.

### F2 — Every solve starts from zero. The fixed schedule is correctly sized *for that*, which is the trap. CONFIRMED

`mgBuildFinestRhs` stores `p = 0` over the whole finest level every advance
(`.wgsl.ts:141`). The final residual 7.79e-5 lands **just** under the 1e-4
tolerance — the paper's 3F+4V is not gratuitously long from a cold start; a
gate alone would trim roughly nothing, and shrinking the schedule blind would
break tolerance. The waste is the cold start itself: the dominant pressure
content (the quasi-hydrostatic column) is nearly identical frame to frame at
dt = 1/30, and the solver rebuilds it from scratch ~30 times a second. The
finest level receives ~130 full-texture writes per advance to reproduce a
field that differs from the previous one by a small local correction.
(Cold-start sizing: CONFIRMED by the residual margin. Frame-to-frame
correlation magnitude: OPEN — WP0 measures it directly.)

### F3 — The ping-pong makes every smoother pass write 100% of cells at every level. CONFIRMED

`mgSmoothColour` writes both textures of a pressure pair alternately, so
wrong-colour, air, and solid cells all take a load+store pass-through path
(`.wgsl.ts:297`) purely to move data to the other texture. Per colour pass at
L0 that is 287k writes to update at most ~12–13% of cells (half of the ~25%
liquid fraction here — but the mechanism wastes proportionally on **every**
scene: the pass-through exists regardless of liquid fraction). The pass-through
is also what forces the parity bookkeeping (`p[level] ^= 1` threading through
`buildPlan`) that makes any skip-a-cycle scheme state-inconsistent today.

### F4 — WebGPU permits fixing F3 exactly for this format. CONFIRMED

`r32float` is precisely the storage-texture format WebGPU allows `read_write`
access for. Red–black GS is in-place-safe by construction: a colour-c update
reads only opposite-colour neighbours (never its own old value —
`.wgsl.ts:300-301`), and pass-through projection `max(old, min)` is the
identity for every cell the update path doesn't already project (air has
`min = -∞`; solid cells adjacent to liquid are on the update path via the
extrapolated phi). In-place smoothing is therefore **bit-identical**, deletes
the pass-through write entirely, and collapses each level's pressure pair to
one texture.

### F5 — An encoded-but-empty pass is cheap but not free: ~2.5 µs. CONFIRMED (in-repo measurement)

The FIM front already uses the target pattern — GPU-side counter + indirect
dispatch, "zero-work calls once the active list empties"
(`webgpu-uniform-velocity-extrapolation.ts:131-135`). Its measured tail:
2×51 exhausted update slots plus their 1-workgroup `prepare` chasers cost
~0.67 ms. That floor means *gating alone cannot deliver the win*: 1,400
converged-but-encoded pressure passes would still cost ~3.5 ms. The encoded
schedule itself must shrink to what recent frames needed, with the GPU gate as
the exact in-frame guarantee. (This is not dispatch micro-optimization — it is
refusing to encode work known to be unnecessary, with a correctness backstop.)

### F6 — FIM encodes for the grid diameter; the front depth is physical and small. CONFIRMED

`activeFrontPasses = max(dims) = 64` slots per invocation, twice per advance;
telemetry reports 13 executed. The extension depth tracks how far the surface
moved (CFL-bounded), not the grid diameter, on every scene. ~102 empty
update passes + 128 `prepare` dispatches ≈ 0.67 ms, plus the fixed full-domain
seed/resolve/hierarchy/pack cost. FIM is the secondary target (12.8%).

### F7 — The async readback path needed for lagged sizing already exists. CONFIRMED

`readStats()` (`webgpu-uniform-reference.ts:972`) maps the diagnostics buffer
without stalling the frame (`mapAsync`, `readbackPending` guard). Lagged
schedule sizing rides this exact mechanism — **no new synchronous readback is
introduced anywhere**. (Peter's readback veto in the dam-boiling context was
about mid-frame stalls/substepping as correctness mechanisms; a one-frame-lagged
telemetry hint that only sizes the next encode is neither.)

---

## 3. The program

The multiplier is the *product* of two facts: warm start makes most cycles
unnecessary (F2), and residual-driven execution is what lets us not pay for
them (F1) without ever violating the tolerance the code already cites. Neither
works alone: gating a cold-start solve saves ~nothing (F2), and a warm start
without a gate still runs the full schedule. In-place storage (F3/F4) is the
enabler that makes skipping state-consistent, and is a standalone ~2× on the
dominant kernel even if everything else stalls.

### WP0 — Per-cycle residual trajectory (instrumentation + one capture). Do first.

Add per-cycle convergence telemetry: extend the diagnostics buffer with one
projected-fine-residual slot per cycle (7 max), written by an
`mgMeasureFineResidual` variant encoded at each cycle boundary (`stage`
boundaries already exist in the plan). Extend `readStats()` and the profile
lane's report to dump the trajectory. Then capture the mini-dam-64 lane once
(plus one still-water scene) over a few hundred advances, both with and
without the WP2 warm-start seed.

This is the program's decision gate, and it is cheap (~7 × 40 µs of measure
passes, telemetry only). It settles the two OPEN questions:

- **k_typ**: after how many cycles is the projected residual ≤ 1e-4, per
  frame, warm vs cold? (Expected: cold ≈ 7 — the current margin says so; warm
  ≈ 1–2 with spikes at impact frames.)
- **No-stall check**: does the residual plateau (LCP active-set churn) instead
  of contracting? A plateau above tolerance kills WP3's payoff and reduces
  this program to WP1 (+~1.3× frame); say so and stop there.

Acceptance: trajectory table archived in the artifact dir; k_typ and the
warm/cold ratio quoted in this doc's status line.

### WP1 — In-place pressure/minimum storage. Bit-identical, lands independently.

Convert `mgSmoothColour`, `mgProlongateAdd`, `mgAddPressure`,
`mgShiftMinimum` (and the clears) to `texture_storage_3d<r32float, read_write>`
on a single pressure texture and a single minimum texture per level. Update
path: liquid cells of the active colour compute and store `max(p_new, min)`;
everyone else **returns without writing**. Delete the parity arrays
(`p[]`, `min[]`), the pass-through branch, the finish-stage parity copy, and
the dead `coarseDone` flag path (F1). `pressureTexture` keeps meaning "the"
L0 pressure texture. Keep `fullCycleBackup` (Algorithm 3 p_tmp) as is.

- Expected: smoother 16.03 → ~8–9 ms (pass-through was ~half the finest-level
  traffic and all of the wrong-colour cost); frame ~33 → ~25 ms. ESTIMATED.
- Acceptance: **bit-identical** end-of-step state vs HEAD on the mini-dam lane
  (checksum A/B under `FLUID_AWAIT_EVERY_STEPS=1` — mid-run checkpoints lie
  without it), existing uniform test suites green, allocatedBytes drops by one
  r32float texture per pair converted.
- This WP also deletes the aliasing hazard class the `emit()` guard checks for
  on those bindings; keep the guard for the remaining ping-pong users
  (rhs/residual scratch selection in `vCycle`).

### WP2 — Warm start: previous pressure as the initial guess.

`mgBuildFinestRhs` seeds `p = select(0.0, max(p_prev, minimum), liquidNow)`
instead of 0, where `p_prev` is the same L0 texture's value from the previous
advance (after WP1 it is trivially still there; the kernel reads its own texel
before writing — no cross-texel hazard). Add a finish-stage cleanup that zeroes
non-liquid cells so `p_prev` is well-defined (today air cells accumulate
prolongation residue), one ~30 µs pass. Coarse levels stay cold — every cycle
re-derives them from the fine residual, so nothing else changes.

- Correctness envelope: the seed is only an initial guess to the same LCP with
  the same projection applied at every sweep; a bad seed (newly-liquid cell)
  costs iterations, never the answer. With the WP3 gate, the tolerance
  criterion is *enforced* rather than hoped for — strictly stronger than
  today's open-loop schedule.
- Alone (before WP3) this changes cost not at all and results only within
  solver-iteration noise; it exists to make cycles 2..7 skippable. Its effect
  is read off the WP0 warm-vs-cold trajectory.
- Acceptance: stability lanes green (ceiling-separation GPU test, dam-boiling
  observations at 7a5ef89 not regressed); warm-start trajectory strictly below
  cold at cycle 1 on the captured lanes.

### WP3 — Residual-driven cycle execution (the multiplier).

Two layers over the same fixed plan, which becomes an **upper bound** instead
of a constant:

1. **Exact in-frame gate (GPU, house pattern from FIM).** Tag every planned
   dispatch with its cycle index. All dispatches of gateable cycles read their
   workgroup counts from an indirect-args buffer (one 3-u32 slot per
   level-size per cycle; `mgSolveCoarsest`'s (1,1,1) included). After cycle
   k's boundary measure (WP0's kernel), a 1-workgroup `mgGateCycle` writes
   cycle k+1's slots: real counts if the projected residual > 1e-4, zeros
   otherwise — and zeros its own subsequent measure so a converged solve costs
   only empty passes. Skipping is state-consistent because after WP1 a cycle
   that doesn't run leaves the single canonical pressure texture exactly as
   its predecessor left it, and every cycle rebuilds all coarse-level state it
   consumes (`mgClearPressure`/restrict chain) before use. No readback, no
   stall; worst case = today's schedule + ~0.4 ms of measures/gates.

2. **Lagged encoded ceiling (host, rides F7).** Per advance, encode only the
   first `min(ceil(k_recent) + 1, full schedule)` cycles of the plan —
   a prefix, so the ceiling remains the paper's exact 3F+4V. `k_recent` comes
   from the async per-cycle telemetry (grow immediately to the full schedule on
   a miss — residual above tolerance at the encoded prefix's end — decay by
   one cycle per quiet frame). This is what removes the F5 empty-pass floor.
   A missed frame is no worse than today's guarantee class (the fixed schedule
   never checked the residual at all), lasts one frame, and is visible in
   telemetry. The growth/decay policy is scene-independent — it is a control
   law on the measured residual, not a tuned constant.

- Prefix composition is initially the paper order (F,F,F,V,V,V,V truncated).
  If WP0 shows warm-started frames converging inside the first F-cycle's cost
  for the wrong reasons (F-cycle ≈ 340 passes vs V-cycle ≈ 106), evaluate a
  V-first prefix *by measurement*; the ceiling stays the full paper schedule
  either way.
- Escape hatch: `FLUID_UNIFORM_FIXED_SCHEDULE=1` restores today's open-loop
  behaviour exactly (full prefix, gate off) for A/B and for paper-fidelity
  runs.
- Acceptance: with the flag on — bit-identical to HEAD. With adaptation on —
  projected fine residual ≤ 1e-4 on ≥ the same set of frames as HEAD across
  the mini-dam lane and one still scene; miss-rate telemetry present;
  pipeline-panel/pass-count consumers (`tests/fluid-pipeline-graph.test.ts`,
  `planStageCounts`) updated to tolerate a variable encoded prefix.

### WP4 — FIM ceiling from observed front depth (secondary, independent).

Encode `min(2 × maxRecentExecuted + 4, maxDim)` update+prepare slots using the
same lagged telemetry (`uniformFIMExecutedPasses` already exists), with a
GPU tripwire: if the front is still non-empty when the last encoded slot runs,
set a flag; the host escalates to the full-diameter ceiling next frame and the
frame's extension falls back to the hierarchy fill it already runs (the
current + predicted fields still get far-field values; one frame of narrow-band
truncation is bounded by the same reasoning the existing ceiling comment makes
about settled chains). The ×2+4 headroom is a control law, not a tuned number:
front depth is CFL-bounded and frame-coherent on every scene.

- Expected: 3.43 → ~2.3 ms (kills the 0.67 ms tail and most prepare chasers).
  Deeper FIM cuts (shared seeding across the two invocations, banding the
  hierarchy fill) are real but out of scope until pressure lands. ESTIMATED.

---

## 4. Projection — and what 5× actually requires

Per-advance model (ESTIMATED, anchored to §1 splits; "quiet" = warm-started
frame converging in ≤1 cycle, "heavy" = impact/topology frame needing several):

| Component | HEAD | WP1 | +WP2/3 quiet | +WP2/3 heavy (3 cycles) |
|---|---|---|---|---|
| Pressure setup+finish | ~1.0 | ~1.0 | ~0.8 | ~0.8 |
| Cycles (smooth+spine) | 18.6 | ~10.5 | ~1.5 (1 V, in-place) | ~5.5 |
| Measures + gates | — | — | ~0.4 | ~0.4 |
| FIM ×2 | 3.4 | 3.4 | ~2.3 | ~2.3 |
| Transport + projection + diag | 3.0 | 3.0 | 3.0 | 3.0 |
| Unattributed/serialization | ~6.5 | ~5.5 | ~1.5 | ~2.5 |
| **Advance** | **33.3** | **~23** | **~9.5** | **~14.5** |

- WP1 alone: ~1.4×, bit-identical, no open questions.
- If WP0 confirms k_typ ≤ 1 warm (the F2 physics argument says it should):
  median advance ~7–10 ms → **3.5–5×**, still frames better than 6×, heavy
  frames ~2.5×. The 5× claim is a *median* claim and it lives or dies on the
  WP0 trajectory — which is why WP0 is first and costs a day, not a month.
- If k_typ ≈ 2–3 warm: median ~12–14 ms → ~2.5×. The remaining road to 5×
  is then per-cycle cost (V-first prefix, banded/active-tile smoothing at the
  finest level — work ∝ liquid, another structural do-less-work rung that this
  architecture makes easy: the gate buffer generalizes to per-level tile
  lists). Named here so the follow-on is obvious, deliberately not designed
  now.

---

## 5. Risks and kill-switches

- **LCP residual plateau (WP0).** If PRBGS cycling stalls above 1e-4 with
  active-set churn, gating never fires and WP3 is dead weight. Kill-switch:
  ship WP1 only; re-plan. The measure kernel reuses the *existing* projected
  LCP residual definition (`mgMeasureFineResidual`), so the gate can never
  disagree with the telemetry humans already read.
- **Warm-start pathology.** A seed can place cells at the wrong active set;
  PRBGS projects every sweep, and the gate refuses to stop early while the
  projected residual says otherwise. Observable as k spikes in the new
  telemetry rather than as physics artifacts. Stability lanes in §3/WP2
  acceptance are the backstop; `FLUID_UNIFORM_FIXED_SCHEDULE=1` isolates it
  in one variable.
- **Under-encoded frame (WP3 layer 2).** One frame at above-tolerance
  residual, then immediate growth to the full ceiling. Equal to today's
  guarantee class; visible in telemetry; bounded by design.
- **Consumers of fixed pass counts.** The pipeline panel groups by plan stage
  (`planStageCounts`), tests snapshot pass counts, and the xctrace tooling
  labels the fixed schedule (`CM11a Full-Cycles ×3`). All must learn
  "≤ ceiling, gated" semantics — mechanical, but do it in the same PR as WP3
  or the SIM tab lies.
- **Sub-tolerance drift.** Stopping at 1e-4 instead of 7.79e-5 admits ~20%
  more residual on frames that gate at exactly one cycle. This is inside the
  tolerance the code already cites as the GPU-precision figure; the
  volume-drift and max-speed telemetry on the mini-dam lane are the acceptance
  monitors (no worse than HEAD over the capture window).

## 6. Non-goals

- No pass fusion, no megakernel, no encoder restructuring for its own sake
  (the ~5 µs small-pass floor and 12% occupancy are known and deliberately
  not the lens).
- No timestep, substep, or synchronous-readback mechanisms (vetoed).
- No changes to transport, projection, or the FIM hierarchy-fill math.
- No scene-derived constants anywhere: tolerance is the paper-cited 1e-4 s⁻¹
  already in the code; everything else is measured-per-frame control flow with
  the paper schedule as a hard ceiling.

## 7. Validation protocol

1. Every A/B under `FLUID_AWAIT_EVERY_STEPS=1`; end-state checksums, not
   mid-run checkpoints.
2. WP1 gate: bit-identical vs HEAD, mini-dam-64 + one still scene.
3. WP2/3 gate: residual-tolerance satisfaction ≥ HEAD's, volume drift and max
   speed within HEAD envelope over the same advances, `uniform-*` suites and
   `tests/uniform-ceiling-separation-gpu.test.ts` green.
4. Final: re-run `npm run profile:uniform-mini-dam-64-xctrace`; publish the
   before/after table in this doc; the clean (non-instrumented) wall is the
   scoreboard, not the isolated capture.
