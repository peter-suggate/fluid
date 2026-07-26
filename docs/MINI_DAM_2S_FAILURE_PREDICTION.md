# Mini dam break, 2 s / 500 steps — predicted failure map

Static analysis, 2026-07-26. **Nothing here was measured** — the lane was held by
another session. Every claim is anchored to a file:symbol you can re-grep.

Target: `npm run test:webgpu:minimal-power-dam-break`
(`FLUID_SCENE=minimal-power-dam-break FLUID_METHOD=octree FLUID_TARGET_S=2
FLUID_MAX_DT=0.004 FLUID_ORACLE_STEPS=500 FLUID_EXPECT_EXACT_STEPS=500
FLUID_EXPECT_GRID=16,16,16 FLUID_MAXIMUM_LEAF_SIZE=2 FLUID_OCTREE_INTERFACE_BAND=3
FLUID_OCTREE_GLOBAL_FINE_FACTOR=4 FLUID_STABILITY_ENVELOPE=1
FLUID_POWER_GENERATION_AUDIT=1 FLUID_REQUIRE_SPATIAL_FIELD=1`)

This gate **passed on the pre-cutover architecture** (`POWER_LIQUIDS_PERF_HANDOFF.md`:
"full 500-step / 2 s minimal-dam gates pass… generation 501, 1,372 power rows,
converged MGPCG, no validation errors"). So this is a *regression hunt*, not a
bring-up. Prioritise what the structured cutover replaced.

---

## Scene arithmetic (do this once, it rules several things out)

| Quantity | Value |
| --- | --- |
| Domain | 16³ cells × 0.05 m = 0.8 m cube |
| Levels | `maximumLeafSize=2` → leaves are 1–2 finest cells, **only two levels** |
| Fine band | factor 4 → h = 0.0125 m, 64³ samples, 4³ bricks → 4096 bricks |
| dt | 0.004 s fixed, 500 steps = 2.000 s exactly |
| Peak speed | dam break ≈ √(2gh) ≈ 3.7 m/s |
| Coarse CFL | 3.7 × 0.004 / 0.05 ≈ **0.30 finest cells/step** |
| Fine CFL | 3.7 × 0.004 / 0.0125 ≈ 1.2 → substeps m = max(4, 2) = **4** |

**Rules out:** the 6-layer extension depth (`STRUCTURED_VELOCITY_EXTENSION_LAYERS`)
is ≥ 6 finest cells against a 0.3-cell displacement — ~20× margin. Do not spend
time there for *this* scene. Same for the 64-substep bound (m = 4).

**Rules out:** the three `storage buffers (12) exceeds max per-stage limit (10)`
failures in `test:webgpu:octree-m1-cutover` cannot fire here. That suite clamps
to 10 deliberately (`tests/webgpu-power-ui-construction.test.ts`
`PORTABLE_STORAGE_BUFFER_LIMIT`); the smoke harness requests the adapter's
advertised limits (`tools/run-webgpu-smoke.ts:2504` →
`requiredFluidDeviceLimits`). Those three are a **portability-contract** failure,
not a functional one. Different axis — don't let them block this lane.

---

## The complete gate list for this lane

`tools/run-webgpu-smoke.ts` `invariantFailures`, lines 3514–3560:

1. `steps === 500` **and** `info.encodedSteps === 500` **and**
   `submittedTime_s == 2.0 ± 1e-9` **and** `completedTime_s == 2.0 ± 1e-9`
2. `powerGenerationAuditedSteps === steps`
3. `envelope.sampledSteps === steps` — a pressure+volume sample **every** step
4. `invalidVolumeSampleCount === 0` — per-cell fraction stays in [−0.01, 1.5] (:3050)
5. `nonFiniteVelocityCount === 0`
6. `maximumPressureRelativeResidual <= 1e-4` (`webgpu-smoke-pressure.ts:56-65`)
7. `maximumExactVolumeDrift <= 0.01` — **cumulative** vs initial (:3047)
8. spatial QA field full, partially wet, with mixed fine/coarse evidence

Gates 6 and 7 are `Math.max` / cumulative over 500 steps: **one bad step fails the
whole run.**

---

## Ranked predictions

### P1 — Pressure residual gate has literally zero margin ★ most likely

The solver's own target and the test's threshold are **the same number**.

- Gate: `maximumRelativeResidual <= 1e-4` (`webgpu-smoke-pressure.ts:60`)
- Solver: `relativeTolerance = max(requested, OCTREE_SOLVE_TAIL_RELATIVE_TOLERANCE)`
  = **1e-4** (`octree-solve-tail-policy.ts:96-99`, `:13`)
- Budget for this scene, from `planOctreeSolveTail` (`:62-92`), all terms:
  `maximumLeafSize=2` → depth 1 → allowance `min(2, ceil(1/2))` = **1**;
  dam-break **+1**; closed-top ∧ dam-break **+1**; aspect 16/16 = 1 → +0.
  `score = 4 + 3` → **E = 7 encoded outer iterations**, fixed for the whole run.

So the solve aims at exactly the bar, with a fixed 7-iteration budget derived from
*immutable scene facts* — it cannot adapt when a step is hard. The paper's own
envelope is 5–10, and it uses 10 for its hardest geometry.

**Trigger:** the impact transient (~t ≈ 0.25–0.5 s when the front reaches the far
wall), thin-sheet run-up, and air-pocket closure. Also the first step after any
topology flip that changes operator conditioning.

**Signature:** `octree per-step pressure residual peaked at relative=1.0xe-4`
— a number *just above* 1e-4.

**Fix:** give the solver margin against its own gate. Request 1e-5 so the encoded
tail actually drives below the 1e-4 bar. The post-convergence tail is already
zero-work (`prepareCorrectionDispatches` zeroes both dispatch records), so the
cost is bounded by the extra iterations actually executed, not by the encode.
Raising `E` alone is weaker — it does not change *what the solver is aiming at*.

**Diagnose before fixing:** the per-step JSON at `:3137` already prints
`pressureIterationsUsed`, `pressureIterationBudget`, `pressureConverged`. If
`used < budget` on the failing step, the budget is **not** the cause and this is
an operator/BC bug — go to P4/P5.

---

### P2 — Capacity overflow masquerades as a residual failure ★ highest triage value

`webgpu-octree.ts:3115-3129`:

```ts
if (!overflow && liquidRows > 0) { … this.relativeResidual = √(rr/bb); }
else { this.relativeResidual = undefined; }
```

and `tools/run-webgpu-smoke.ts:3131-3134`:

```ts
sample.pressureRelativeResidual ?? (steps <= 2 ? 0 : Infinity)
```

A `pressureCapacityOverflow` (or `liquidRows === 0`) at **any step ≥ 3** yields
`undefined` → `Infinity` → gate 6 fails, reporting
`relative=undefined rms=undefined`.

**This is a different bug wearing the residual gate's clothes.** A dam break
spreads liquid from a column across the whole floor, so the live liquid row count
peaks far above what the initial condition sized
(`planOctreePressureCapacity`, `planOctreeLeafFrontierAllocation`).

**Triage rule: on any residual failure, read `relative=` first.**
`undefined` → capacity, not convergence. `info.pressureCapacityOverflow` and
`info.frontierCapacityOverflow` are already published (`:3108-3110`) — print them.

---

### P3 — Error carries turn a real fault into a silent freeze ★ instrument first

`structured-dynamics.ts` `rejectSample(stage, index)` does
`atomicOr(&accepted[0], ERROR_SAMPLE)` + `atomicMin(&accepted[1], (stage<<24)|index)`.
Then `prepareStructuredDynamics` gates on `acc(0u)==0u` and publishes
`indirect[…] = 0` for **all nine classes**. Every downstream stage dispatches zero
workgroups.

Consequence: one rejected face silently converts the rest of the step — and every
later step — into a no-op. The sim does not error; it **freezes**. What you then
observe is gate 1 (step count), gate 7 (volume drift), or a flat spatial field,
several hundred steps away from the cause.

`accepted[1]` already holds the **first** error's stage code and index. Decode it:

| Stage | Kernel | Meaning |
| --- | --- | --- |
| 1 / 2 | `advect` | advecting / transported sample invalid |
| 3 | `advect` | non-finite aperture, area, or prior |
| 10 / 11 | `forceFamily` | bad aperture/solid; non-finite forced value |
| 20–23 | `divergenceRow` | slot count, handle, geometry, volume/dt |
| 30–33 | `projectFamily` | row range, normal, aperture/solid, non-finite |
| 40–45 | `neighborAverageA/B` | slot count / handle / neighbour range |
| 50–54 | `reconstructRow` | header, handle, neighbour, coefficient, non-finite |

Parallel carries exist in `structured-boundary.ts` (`control.flags` bits
2/4/8/16/32/64 + `firstError`) and in the solver
(`control[0]`, `control[6]`=stage, `control[7]`=row via `reportAt`).

**This is the single highest-leverage 30 minutes on the whole list.** It converts
"the sim stopped moving at step 213" into "stage 22, row 1044, step 213".

---

### P4 — All-or-nothing fine transport commit

`fine-levelset-transport.wgsl.ts` `summarizeStructuredFineTransport`:
`control.committed = (control.nonfinite == 0)`, and
`commitStructuredFineTransport` writes **nothing** unless `committed != 0`.

One unresolvable characteristic anywhere in the band discards **the whole
domain's** surface advection for that step. The surface freezes for a step while
velocity keeps evolving → level set desyncs from the flow → liquid
classification drifts → shows up 100+ steps later as gate 7 (volume) or a
component-count change.

Triggers here: the front reaching a wall (`departureOutsideBand`), thin sheets,
and — after the air-support cutover — any row where `regularVectorAt` returns
`w=0` so `reconstructAirSupportVectors` bails and never publishes that air row's
velocity.

`unpackFineLevelSetGPUTransportControl` (`fine-levelset-transport.ts:98`) already
decodes `departureOutsideBand`, `nonfiniteVelocity`, `committed`,
`extrapolatedVelocity`, `maximumDisplacementFineCells`, `velocityUnavailable`.
**Assert `committed === true` every step as its own gate** rather than letting it
surface as volume drift.

---

### P5 — Air-support epoch handshake (newest code, highest prior)

`supportPublicationValid()` (`structured-dynamics.ts:442-453`) is a **12-condition**
cross-module agreement: `acc(3u)` generation, `bank()`, `boundaryControl[4]`,
`acc(2u)` row count, `SUPPORT_VALID` magic, `SUPPORT_LAYOUT_VERSION`, capacity,
and seed/face count ordering. `advect` calls it at `:592` and rejects the sample
outright on any mismatch → straight into P3's silent freeze.

`webgpu-octree.ts:2695` publishes air support *after* `dynamics.encodeProjection`,
so it is refreshed against the just-projected field — correct — but that also means
any generation flip between publication and the next step's advect invalidates the
whole handshake. On a dam break the topology changes **every step**, so this path
is exercised 500 times.

Also `ERROR_CAPACITY` in `prepareAirSupportFaces` (`:679`) when
`faceRows > faceCellCapacity` or `count > faceCapacity` (rows × 12 owned slots).

This is the least-aged code on the path. Suspect it early.

---

### P6 — Cumulative volume drift ≤ 1 % over 2 s

`exactVolumeDrift = (cellSum − initialVolumeCellSum) / |initialVolumeCellSum|`
(`:3047`) — **cumulative**, and `Math.max|·|` over all 500 steps, so it also
catches transient bulges.

Contributors, in expected order: dropped transport commits (P4), first-order
semi-Lagrangian dissipation, JFA-CPT redistance not being mass-preserving, and
whatever `webgpu-octree-fine-levelset-volume.ts` is or isn't correcting.

Related and stricter: gate 4 requires the per-cell fraction field to stay within
`[−0.01, 1.5]` **every** step (`:3050`). A cell reading > 1.5 is a volume-field
bug, not drift — check that first if both fire.

---

### P7 — Step-count failure is usually a *symptom*

Gate 1 (`accepted N steps; expected exactly 500`) is the first message you see
when anything terminates the run early. The real cause is at step N. Do not debug
the step count — debug step N. Same for the `submittedTime_s`/`completedTime_s`
equality checks: they fail as a consequence, not a cause.

---

### P8 — Fix cutover failure #39 before trusting any long run

`rejected packed A/B publication preserves every accepted byte` is the A/B carry
path. On a lane with per-step topology churn, a rejected candidate **must** leave
the accepted bank byte-identical. If it does not, one rejected epoch corrupts
state permanently and every subsequent step is garbage — which looks like a slow
drift failure, not a publication bug.

This is a long-run correctness precondition. It is cheap relative to chasing its
downstream symptoms.

---

## Recommended order

Each step below makes the next one cheaper. Resist reordering.

1. **P3** — decode and print the three error carries. Turns silent freezes into
   named stage + row. ~30 min, unblocks everything else.
2. **P2 + P4** — print `pressureCapacityOverflow`, `frontierCapacityOverflow`, and
   the transport `committed` flag per step. Separates capacity and transport
   failures from convergence failures. ~20 min.
3. **P8** — fix the A/B carry preservation. Precondition for believing any 500-step
   result.
4. **P1** — only now attack the residual gate, with the instrumentation from (1)
   telling you whether the budget saturates. Give the solver 1e-5 against a 1e-4
   gate.
5. **P5/P6** — by this point the carries will name them directly.

## Anti-recommendations

Do not spend time on: extension depth, substep count, or the 12>10 storage-buffer
failures. The arithmetic and the harness config above rule all three out for this
lane.
