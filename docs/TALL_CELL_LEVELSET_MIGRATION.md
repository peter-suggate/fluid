# Tall-cell method: VOF → level-set migration plan (handoff)

Status: approved plan, ready to execute (2026-07-16).
Prerequisite reading: `TALL_CELLS_PAPER.md` (Sections 3, 5–9 + Appendix A),
the two "2026-07-16" sections of `TALL_CELL_STABILITY.md`, and
`tests/tall-cell-assignment.test.ts` for the assignment invariants.

---

## 0. Goal, non-goals, decisions

**Goal.** Replace the tall-cell method's conservative-VOF surface field with
the signed-distance level set the tall-cells paper actually prescribes
(point samples at Eq 4 positions, Eq 5 linear reconstruction, semi-Lagrangian
advection, Sec 6 reinitialization, Sec 8 remeshing without representability
floors), while keeping volume drift bounded by a global correction term.

**Non-goals.** The uniform solver stays VOF (it is the A/B reference). The
quadtree method is untouched. No spray/particle system. No new UI beyond one
method parameter.

**Decisions adopted (change only with new evidence):**
1. Volume strategy: **global correction divergence** — measure reconstructed
   liquid volume vs the layout reference each frame; distribute the error as
   a uniform correction divergence over interface cells through the existing
   RHS `c`-term plumbing. Not per-cell VOF, not CLSVOF.
2. Uniform path unchanged; cross-method comparisons run on reconstructed
   occupancy fields (φ≤0 → 1).
3. The interior-velocity remesh gate (`storeSpeed > 0.05·rail` descent)
   stays, behind `FLUID_FLOW_GATE=0/1` (default on), re-evaluated in Phase 3.
4. Transition flag: `FLUID_SURFACE=vof|levelset` env override plus a
   `surfaceField` method param (default `vof` until Phase 3 exit, then
   `levelset`). Both paths must build and pass their own gates until Phase 4
   removes the VOF path.

**Environment protocol for every measurement below:**

```
WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js
FLUID_DISABLE_TIMESTAMPS=1 FLUID_CPU_ORACLE=0
# per-step telemetry when a trajectory is needed:
FLUID_REPORT_EVERY=1
```

---

## 1. Baseline capture (do this FIRST, before any code change)

Record into `tmp/levelset-migration/baseline/` (jsonl + the dump output):

| # | Command | Record |
| --- | --- | --- |
| B1 | `npm run test:unit` (with env) | count (146 as of 2026-07-16), all pass |
| B2 | `FLUID_METHOD=tall-cell FLUID_SCENE=settled-tank,deep-water,dam-break-boxes npx tsx tools/run-webgpu-smoke.ts` | pass/fail + failure strings |
| B3 | `npm run test:webgpu:dam-tall-active` | envelope numbers + failure strings |
| B4 | `FLUID_METHOD=tall-cell FLUID_SCENE=dam-break-ui FLUID_TARGET_S=3 FLUID_REPORT_EVERY=1 npx tsx tools/run-webgpu-smoke.ts` | KE/residual/gaps trajectory |
| B5 | `FLUID_TARGET_S=0.224 npx tsx tools/dump-tall-slice.ts` | slice + counters |
| B6 | Same as B4 for `FLUID_METHOD=uniform` | uniform KE reference |

Expected baseline (2026-07-16 evening; investigate first if these moved):

- B4 tall KE at t = 0.5/1.0/1.5/2.0/3.0 s ≈ 0.68 / 1.65 / 1.20 / 0.97 / 1.20;
  pressure relative residual ≤ 0.015 after startup; **no blow-up**;
  `dryTallWithWetRegularAbove` mid-run 45–105.
- B6 uniform KE ≈ 0.47 / 0.11 / 0.04 / 0.02 / 0.007.
- B5: `dryUnderWetBand = 0`, `wetStores ≈ 1100`, `unexcusedDelta = 0` (via
  the assignment test's excuse rules).
- B2: dam-break-boxes passes; settled-tank fails with flags
  `[pressure-residual]`; deep-water fails with
  `[pressure-residual, advective-cfl]` (the known y0-conditioning gap —
  this migration is expected to FIX it in Phase 2).
- B3: energy criteria pass (projRatio 1.0, peakCFL ~2.7); fails on
  `dry tall columns underneath wet regular` (~79) and wet-IoU
  (~0.25 vs 0.35 floor / ~0.38 vs 0.40 final).

---

## 2. Phase 1 — shadow level set (no behavior change)

### Work items
1. `lib/tall-cell-grid.ts`: emit `initialPhi: Float32Array` alongside
   `initialVolume` (keep both until Phase 4). φ = signed distance to the
   initial surface in metres, sampled at each packed sample's Eq 4 world
   position (y0 → world y 0.5·h; y1 → (base−0.5)·h; band → cell centers),
   negative inside liquid, clamped to ±5·min(h). Same for the probe layouts.
   Keep `initialVolumeCellSum` and add `referenceLiquidVolume_cells` (same
   number; new name survives Phase 4).
2. `lib/webgpu-eulerian.ts`: allocate `phiA/phiB` (r32float, packed dims);
   upload `initialPhi`; add two pipeline stages per substep, after velocity
   advection:
   - `advectPhi`: semi-Lagrangian (RK2 trace reusing `tracedVelocity`),
     sampling via the new `samplePhi` (Eq 5/6 logical-cubic trilinear). NOT
     MacCormack (paper Sec 7).
   - `reinitializePhi`: port from `tall-cell-multigrid.ts` with the Sec 6
     safeguards — only act where |φ| ≤ 3 cells; freeze samples adjacent to a
     sign change; per-pass change ≤ 1 cell spacing; clamp |φ| ≤ 5Δx. Two
     passes per step (ping-pong).
   - On remesh steps: remap φ like other band fields (trilinear at new
     positions) and tall endpoints by direct `samplePhi` at the new Eq 4
     positions (least-squares fit upgrade lands in Phase 3).
3. `lib/tall-cell-kernels.ts`: add (do not yet consume) `phiCell(q)` (Eq 5
   linear interior / Eq 6 band lookup / boundary values outside) and
   `samplePhi(p)`.
4. Telemetry: in `readStats`/reductions, add
   `phiSignMismatchFraction` = fraction of active samples where
   (φ≤0) ≠ (alpha-classified wet), and `phiReconstructedVolume_cells` =
   Σ occupancy(φ) over the fine grid (occupancy: 1 if φ≤−h/2, 0 if φ≥h/2,
   linear between). Surface both in the smoke `running` records.
5. `FLUID_SURFACE` flag plumbed (method param `surfaceField`, env override in
   `tools/run-webgpu-smoke.ts`) — in Phase 1 it only gates the telemetry
   consumers, not physics.

### Testing criteria (Phase 1 exit)
- T1.1 All baseline suites (B1–B5) unchanged within noise — the shadow field
  must not perturb physics: B4 KE at each half-second within ±10% of
  baseline, B5 counters identical.
- T1.2 New unit test `tall-cell-phi.test.ts`:
  - initial φ: sign matches `initialVolume` classification at every packed
    sample for dam-break-ui and settled-tank layouts; |φ| ≤ 5Δx; endpoint
    samples differ across a dam face column (point samples, not averages).
  - reinit safeguards present structurally (regex pins like the existing
    conformance tests): freeze-adjacent, 5Δx clamp, ≤1-cell motion.
- T1.3 Shadow agreement, dam-break-ui, 1 s, per-step:
  `phiSignMismatchFraction < 0.02` at every step after step 5, and
  `< 0.005` at t=1 s for settled-tank.
- T1.4 Shadow volume: |`phiReconstructedVolume` − reference| / reference
  < 3% at t=1 s dam-break (uncorrected drift measurement — record it; it
  calibrates the Phase 3 correction gain).
- T1.5 GPU cost: `runtime_ms` for B4 grows < 15%.

Rollback: the flag defaults to `vof`; deleting the two dispatches restores
baseline exactly.

---

## 3. Phase 2 — consume φ (classification, pressure, remesh, render)

All changes below are inside `if (surfaceField === "levelset")` branches or
shader variants selected at pipeline-build time. VOF transport still runs and
alpha remains the volume monitor.

### Work items
1. Classification: `pointSampleAlpha(id) ≥ 0.5` → `phi(id) ≤ 0` at: gravity
   MAC gate (either-adjacent via `phiCell`), `buildPressureRhs` wet gate,
   `jacobi` liquid gate, extrapolation seeding flags
   (`lib/tall-cell-extrapolation.ts`), diagnostics reductions.
2. Pressure ghosts: `pressureTerm`/`interfaceFraction` switch to the
   multigrid's φ form (`ghostFraction(|φown|,|φother|)`, θ floor unchanged at
   0.05); `pressureGradientAt` likewise. Multigrid: delete `buildFinePhi`
   usage for the levelset path — feed `phiA` directly as the fine level.
3. **Revert the control-volume pressure operator** (levelset path only):
   `divergenceAt` returns the paper point divergence at both endpoints;
   `jacobi`/`project`/multigrid stencils use the pre-CV endpoint forms
   (vertical coupling `1/(distance·h)`, single-row laterals at the sample
   row). The CV forms remain compiled for the VOF path until Phase 4.
4. Remesh: `planRemesh` crossings from φ (band sample signs; interior
   crossing when endpoint signs differ, located by the Eq 5 zero of the
   linear profile). Representability floors REMAIN in this phase (alpha is
   still transported and must stay representable). Flow gate unchanged.
5. Render/overlay: `webgpu-water-pipeline.ts` `fieldCell` and
   `webgpu-grid-overlay.ts` read φ (surface at φ=0; overlay wet = φ≤0) when
   the flag is on. `tools/dump-tall-slice.ts` gains a `--field=phi` view.
6. Curvature/surface tension: CSF from ∇φ (normalize `phiGradient`), same
   coefficient wiring.

### Testing criteria (Phase 2 exit) — run everything twice (flag off/on)
- T2.1 Flag OFF: byte-for-byte baseline behavior (B1–B5 unchanged).
- T2.2 Flag ON, equilibrium: settled-tank and deep-water invariants **pass
  with zero stability flags** — this is the acceptance test for the CV
  revert (baseline fails on `pressure-residual`). Additionally
  post-projection relative residual < 0.05 every step and equilibrium liquid
  speed < 0.05 m/s after 0.5 s.
- T2.3 Flag ON, dam-break 3 s: no blow-up (KE < 3.0 at every sample);
  KE at t≥1.5 s ≤ baseline B4 at the same times (expect improvement from the
  cleaner classification); residual ≤ 0.02 throughout.
- T2.4 Flag ON, `dump-tall-slice` at 0.224 s: `dryUnderWetBand = 0`
  (redefined as φ>0 store top under φ≤0 band bottom), `unexcusedDelta = 0`.
- T2.5 `npm run test:webgpu:dam-tall-active` flag ON: energy criteria pass
  (projRatio ≤ 2.0, peakCFL ≤ 32, no non-finite); record IoU (expected to
  improve vs 0.25 — do not gate yet, re-baseline in Phase 4).
- T2.6 Assignment tests pass with the φ-based crossing definitions (update
  `tall-cell-assignment.test.ts` to read φ when the flag is on).
- T2.7 Shadow-consistency inversion: alpha (still transported) vs φ
  classification mismatch < 3% through the 3 s run — now alpha is the shadow.

Rollback: flip default flag to `vof`.

---

## 4. Phase 3 — retire VOF transport; volume correction; LSQ remap

### Work items
1. Delete from the levelset path (kernels + eulerian pipeline):
   conservative flux stack (`rawVolumeFlux`, limiters, integrated face
   fluxes, `advectedVolume`, `advectedTallVolume`, `advectedTallTopGuide`),
   transport-preparation shader/dispatch, density sharpening (3 dispatches +
   deposit buffer), `volumeCorrectionDivergence` per-cell drain/refill,
   `tallConnectedToBand`, `tallStoreAlpha` consumers, `tallFillCells` /
   `columnWaterCells` / `columnHighestWetCell`.
2. Remesh floors: delete the water-integral floor and `wetTopFloor` from
   `planRemesh`/`smoothRemesh` (levelset deletes unrepresentable liquid —
   paper Sec 8). Eq 10 becomes strict. Flow gate: measure with
   `FLUID_FLOW_GATE=0` vs `1` (T3.4) and keep whichever passes; default per
   result.
3. Remap: tall endpoint φ and velocity by least squares through the old
   values covered by the new tall cell (port `coarseTallFit`); band samples
   trilinear as today.
4. **Global volume correction**: after `readStats`-independent reduction
   (GPU sum of occupancy(φ)), compute
   `c_global = λ_v · (V_ref − V(φ)) / (V_interface · Δt_paper)` clamped to
   ±1/s (λ_v = 0.5, Δt_paper = 1/30 s — same constants family as Sec 3.7)
   and add `−c_global` to the RHS of every interface cell
   (|φ| < 1.5Δx). Uniform-per-interface-cell, sign pulls volume back toward
   `V_ref`. Wire `FLUID_VOLUME_CONTROL=0` to disable for A/B.
5. Inflow: sources set φ (carve negative φ in the inlet region) instead of
   depositing alpha; `V_ref` integrates the inflow volume rate.
6. Gates/tools: `exactVolumeCellSum` path reads occupancy(φ);
   `representedVolumeDrift` becomes drift-vs-`V_ref`; `inspectTallVolumeGaps`
   φ-based; smoke invariants switch the tall path from "conservation exact"
   to "drift bounded" (see table §6).

### Testing criteria (Phase 3 exit)
- T3.1 Unit suite green after test migration (see §5): conformance tests pin
  SL-φ advection + reinit safeguards + Eq 4/5 sampling and ASSERT ABSENCE of
  the deleted machinery on the levelset path.
- T3.2 Volume: dam-break-ui 5 s with `FLUID_VOLUME_CONTROL=1`:
  |drift| ≤ 1% at every sample and ≤ 0.3% at t=5 s. With `=0`, record the
  free drift (expect 2–6%; document).
- T3.3 Stability soak (`test:webgpu:dam-tall-soak` equivalent, 5 s): no
  non-finite, peak CFL ≤ 8, KE at t=4 s ≤ 0.1 and monotone-decaying trend
  after the last slosh (uniform reaches 0.004; allow 25× while the Sec 5
  interior-face gap remains).
- T3.4 Flow-gate A/B (3 s dam): choose the config whose KE trajectory
  dominates (lower at ≥80% of samples) AND passes T3.3; record both.
- T3.5 Equilibrium: settled-tank/deep-water still zero-flag (regression
  guard on the deleted floors); deep-water compression < 0.5 preserved.
- T3.6 Assignment test strict mode: `unexcusedDelta` computed with NO floor
  excuses == 0 at t=0.224 AND t=2.0 (add the second sample point).
- T3.7 Spray audit (expected visual change, bounded): component count at
  t=3 s ≤ 20 (baseline B4 shows ~30–100 VOF droplets; level set will
  delete small ones — verify it doesn't delete the POOL: dominant component
  fraction ≥ 0.99 and volume drift within T3.2 bounds).
- T3.8 Performance: levelset-path B4 `runtime_ms` ≤ VOF baseline (deleting
  sharpening + transport-prep should more than pay for reinit).

Rollback: flag back to `vof` (path still intact in this phase).

---

## 5. Test migration inventory (Phase 3/4)

| File | Change |
| --- | --- |
| `tests/tall-cell-paper-conformance.test.ts` | Tall path: replace sharpening block with reinit pins (freeze-adjacent regex, `5.0*` clamp, SL trace for φ — `assert.doesNotMatch` MacCormack corrector in the φ path); delete correction-divergence and constant-density blocks; add Eq 4 (endpoint point samples at `0.5`/`base-0.5`) and Eq 5 (`mix(phi0,phi1,t)`) pins; keep uniform-path sharpening tests unchanged. |
| `tests/tall-cell-assignment.test.ts` | Crossings from φ; delete the representability excuses (strict Eq 10); add t=2.0 s second GPU sample; keep printed slices. |
| `tests/tall-cell-grid.test.ts` | `initialPhi` construction: signs, 5Δx clamp, endpoint distinctness at the dam face, `referenceLiquidVolume_cells` equals the old cell sum. |
| `tests/tall-cell-phi.test.ts` (new, Phase 1) | See T1.2/T1.3. |
| `tests/tall-cell-velocity-transport.test.ts` | Drop conservative-flux φ cases; velocity MacCormack cases unchanged. |
| `tests/webgpu-smoke-scenarios.test.ts` | Unchanged (scenario metadata only). |
| Smoke gates | See §6. |

---

## 6. Gate recalibration table (Phase 4)

| Gate (tools/run-webgpu-smoke.ts) | VOF semantics | Level-set semantics |
| --- | --- | --- |
| `representedVolumeDrift ≤ 1%` (non-inflow scenes) | exact conservation check | drift vs `V_ref` with volume control ON; same 1% number, now a *control* check |
| dam-break-boxes `exactVolumeDrift ≤ 1e-3` | exact | ≤ 1e-2 with control ON (recalibrate from T3.2 data) |
| `volume maximum ≤ 1.5` | density overfull bound | delete (no density) |
| mixed-cell fraction vs uniform | sharpening regression | delete (no sharpening); replace with interface-thickness check: fraction of cells with \|φ\| < 0.5Δx adjacent to sign change ≥ 0.9 |
| `dryTallWithWetRegularAbove == 0` | classification gap | φ-sign inconsistency, keep == 0 |
| `unexcusedDeltaViolations == 0` | with floor excuses | strict (no excuses) |
| wet-IoU floors 0.35 / 0.40 | calibrated 2026-07-15 | re-baseline after Phase 3 (expect higher; set floor at 0.9× measured min over 3 runs) |
| KE/projRatio/CFL envelope | unchanged | unchanged numbers; re-verify |
| equilibrium zero-flags | currently failing (CV conditioning) | must pass from Phase 2 on |

---

## 7. Phase 4 — cleanup

- Remove the VOF path from the tall method (kernels, pipeline stages, flag
  branches), `initialVolume` from the layout (uniform solver keeps its own),
  the CV operator code, `surfaceField` param collapses to informational.
- Delete `densitySharpening` method param; keep `FLUID_SHARPENING` for the
  uniform path only.
- Update `TALL_CELL_WEBGPU.md` (storage/semantics section), append the
  migration record to `TALL_CELL_STABILITY.md`, refresh
  `tools/dump-tall-slice.ts` default view, re-run and commit new baselines
  for §6 gates.
- Exit: B1–B5 suite fully green INCLUDING the two equilibrium scenes and
  `dam-tall-active` (all criteria, with re-baselined IoU); `npm run build`
  green; `docs` updated; a `dump-tall-slice` capture at 0.224 s and 2.0 s
  archived next to the baselines.

---

## 8. Risk register

| Risk | Detection | Response |
| --- | --- | --- |
| Volume drift exceeds budget | T3.2 per-step drift telemetry | raise λ_v toward 1, widen interface band to 2Δx; if still failing, revisit CLSVOF decision |
| Spray/droplet loss reads as visual regression | T3.7 + user review of the UI dam break | accept (paper behavior) or schedule particle spray follow-up; do NOT re-add sharpening |
| Reinit disturbs a calm surface (settled tank ripples) | T2.2 / T3.5 equilibrium speeds | safeguards order: freeze-adjacent first; reduce reinit cadence to every 2–4 steps |
| CV revert re-exposes an old instability the CV form was masking | T2.3 trajectory vs baseline | the mass-leak motivation is VOF-only; if dam KE regresses >2×, bisect classification vs operator by toggling only the operator branch |
| Air-velocity igniter recurrence under new classification | `maxAirSpeed_m_s` precursor (bounces >2× liquid max) in any 3 s run | re-run the igniter fix validation (see stability doc); the fix is classification-adjacent |
| Eq 10 strictness breaks splashy remesh (floors deleted) | T3.6 at t=2.0 s | the level set deletes above-band liquid during remap — verify remap actually clamps φ>0 above the band before suspecting Eq 10 |
| f32 precision in the 5Δx band | interface-thickness gate §6 | acceptable per paper; do not widen the band beyond 5Δx |

## 9. Effort estimate

Phase 1: ~1 session. Phase 2: 1–2 sessions (the CV revert + equilibrium
validation is the bulk). Phase 3: 1–2 sessions. Phase 4: ~1 session.
Each phase is independently landable and independently revertible via the
flag until Phase 4.
