# P1: lagged encoded cycle budget for the Uniform CM11a pressure solve

2026-09-19 · branch `codex/main-unpushed-20260918` · measured on Dawn/Metal, one process at a time.

The Uniform method's pressure solve encodes a prebuilt CM11a schedule — 3 Full-Cycles
then 4 V-Cycles — every single step, and lets a GPU-side residual gate early-return the
kernels of any cycle the solve no longer needs. That gate saves the *body* of a pass and
nothing else: the `beginComputePass`, the bind-group sets and the dispatch are still in the
command stream, still cost their launch floor on the GPU, and still cost ~10 µs each of CPU
encode. At 128³ that is 2661 passes a step whether the solve converged in cycle one or cycle
seven.

P1 makes the *encoded prefix itself* shrink. The solver reads how many cycles the last
observed step actually executed and whether it met tolerance, and encodes only that many
cycles plus a headroom. A cycle that is never encoded costs neither its launch floor nor its
encode. The GPU gate stays exactly where it was, as the inner gate inside the encoded prefix.

## Headline

All numbers are paired, per frame, from a single lockstep capture per scene: both arms are
advanced one frame each per round with the arm order rotating, so an arm-to-arm delta is
measured against the same machine state frame by frame. (The uniform Dawn lane is bimodal —
identical code runs 48-86 ms — so two medians from two separate captures would measure
nothing.) `pressure` is the GPU timestamp seam around the whole multigrid solve; `cpu` is the
host encode time for the advance; warmup frames 1-3 are excluded.

| scene | arm | pressure GPU (med) | CPU encode (med) | passes/step | cycles enc | cycles exec | converged | worst residual |
|---|---|---|---|---|---|---|---|---|
| cm12-figure-7 (128³, 7 cycles, 70 f) | **lagged** | 43.88 ms | 27.15 ms | 1307 – 2661 (med 2483, mean 2113) | 2 – 7 (mean 4.78) | 1 – 7 | 74.6% | 5.78e-2 |
| | fixed | 45.65 ms | 29.40 ms | 2661 (all) | 7 | 1 – 7 | 73.1% | 1.98e-3 |
| | **paired Δ** | **−5.54 ms** (p25 −10.09, ratio med 0.871 / p25 0.632) | **−2.58 ms** (p25 −13.80) | −548 mean | | | | |
| minimal-power-dam-break-64 (60 f) | **lagged** | 21.36 ms | 21.84 ms | 2008 (all) | 7 | 7 | 0% | 4.48e-2 |
| | fixed | 20.97 ms | 22.06 ms | 2008 (all) | 7 | 7 | 0% | 3.92e-2 |
| | **paired Δ** | −0.07 ms | −0.08 ms | 0 | | | | |
| large-power-dam-break (60 f) | **lagged** | 7.14 ms | 16.46 ms | 1412 – 1710 (mean 1488) | 3 – 5 (mean 3.51) | 2 – 4 | 100% | 9.55e-5 |
| | fixed | 9.31 ms | 21.92 ms | 2008 (all) | 7 | 2 – 4 | 100% | 9.55e-5 |
| | **paired Δ** | **−2.29 ms** (p25 −2.49, ratio med 0.784) | **−5.11 ms** (p25 −5.95) | −520 mean | | | | |

Wall-clock per advance, paired median: fig7 **−7.86 ms** (p25 −22.49), mini64 −0.19 ms,
large-power **−7.38 ms** (p25 −8.45).

Read the three scenes as three regimes:

* **large-power-dam-break** is the case P1 was built for. The solve converges in 2-4 cycles
  *every step of the run*, so the fixed schedule was encoding 3-5 dead cycles 100% of the
  time. Lagged cuts a steady 26% of passes, 25% of pressure GPU time and 23% of CPU encode,
  with residuals indistinguishable from fixed (worst 9.55e-5 on both arms) and the final
  volume agreeing to 6 significant figures (1471.519 vs 1471.512).
* **cm12-figure-7** is the mixed case. During free fall (frames 4-24) the solve converges in
  one cycle and lagged encodes 2 of 7: pressure −10 ms, CPU −14 ms, wall −23 ms a step, with
  the reported residual *identical to fixed to every printed digit*. From the impact onward
  the demand is genuinely 5-7 cycles and the two arms converge on the same schedule, which is
  why the whole-run median only moves 1.8 ms while the p25 moves 10 ms.
* **minimal-power-dam-break-64** is the null case and the safety proof. This scene never meets
  tolerance at any step on either arm, so the rule pins the budget at the full 7 cycles and
  encodes 2008 passes, bit for bit the fixed stream. The measured delta is −0.07 ms, i.e.
  noise. **P1 cannot make a hard scene slower; it can only decline to shrink.**

### Where the passes go

| | setup + finish | one Full-Cycle | one V-Cycle | full schedule |
|---|---|---|---|---|
| cm12-figure-7 (128³) | 23 | 642 | 178 | 2661 |
| 64³ scenes | 23 | ~463 | 149 | 2008 |

So on fig7 the first cut (7 → 2 cycles) removes 1354 passes, and the measured saving is
~10 ms GPU and ~14 ms CPU: **~7.4 µs of GPU and ~10.3 µs of CPU per pass that was never
encoded**, consistent with the known launch floor. On large-power the 596 removed passes cost
3.8 µs GPU and 8.6 µs CPU each.

## Fidelity

The honest summary: **on two of three scenes the fidelity is unchanged, and on fig7 the cost
is confined to the two steps around impact.**

* **large-power-dam-break**: 100% of steps meet tolerance on both arms. Worst residual
  identical (9.55e-5). 14 of 57 frames agree on the residual to within 1e-12; the rest differ
  in the last digits, which is the ordinary float-CAS transport non-determinism between two
  independent solver instances, not a budget effect.
* **minimal-power-dam-break-64**: neither arm converges at any step (this is pre-existing; the
  scene's tolerance is not reachable in 7 cycles). Lagged worst residual 4.48e-2, fixed
  3.92e-2, medians 1.57e-3 vs 1.64e-3 — lagged is *better* at the median. Since both arms
  encode and execute the identical 2008-pass stream, this difference is entirely CAS ordering
  and bounds how much of any other scene's residual difference is attributable to P1.
* **cm12-figure-7**: lagged converged on 74.6% of steps, fixed on 73.1% — lagged converged
  *more often* over the run. Worst residual is the one real regression: 5.78e-2 on lagged vs
  1.98e-3 on fixed, both at frame 26, the impact frame. Final `volumeCellSum` 33463.617
  (lagged) vs 33463.667 (fixed): a difference of 0.05 cells in 33464, i.e. 1.5e-6 relative,
  three orders of magnitude below the run's own frame-to-frame spread (33463.51 - 33464.45).

### The ramp at impact, frame by frame (fig7)

```
frame | lagged enc/exec  residual | fixed enc/exec  residual
 4-24 |   2/1  Y  ~1e-5           |   7/1  Y  ~1e-5      identical to every printed digit
   25 |   2/2  n  1.58e-4         |   7/3  Y  3.06e-6    <- impact; lagged is capped at 2
   26 |   4/4  n  5.78e-2         |   7/7  n  1.98e-3    <- doubled to 4, still short
   27 |   7/7  n  1.37e-3         |   7/7  n  1.42e-3    <- full schedule, matches fixed
   28+|   7/7     tracks fixed    |   7/7
```

This is the rule's worst case and it is worth stating plainly: **the step that first needs
more cycles than the last step needed will be under-solved, and the recovery takes two steps.**
The growth arm is deliberately aggressive (double, or +2, whichever is larger) precisely to
bound that to two: 2 → 4 → 7. A second, milder instance occurs at frame 41 (4/4 unconverged,
back to 7 the next step) and recovers in one step.

Peter's stated position is that raising the residual tolerance to its maximum (10 s⁻¹) changed
the look of no scene he tried, so a single under-solved impact step is within the fidelity he
has already accepted. If that ever proves wrong in a scene, the headroom control is the dial:
headroom 2 makes the free-fall budget 3 instead of 2 and costs one Full-Cycle (642 passes at
128³) a step, and `fixed` is always one click away.

## What changed

Five files, no behaviour change on the `fixed` arm.

**`lib/methods/uniform/webgpu-uniform-pressure-multigrid.ts`** — the truncation mechanism.
* `:42` `UNIFORM_CM11A_MINIMUM_CYCLE_BUDGET = 1`, `:44` `UNIFORM_CM11A_DEFAULT_BUDGET_HEADROOM = 1`.
* `:309` `activeResidualTolerance` and `:385` the `residualTolerance` getter — `setResidualTolerance`
  previously wrote straight to the uniform buffer and retained nothing, so the host had no way
  to ask whether the gate was even armed.
* `:73` `uniformCM11aCycleBudget(input)`, the whole rule as a pure function (see below).
* `:306` `cycleBoundaries`, `:308` `finishStart` — plan indices recorded while the plan is
  built (`:690`-`:701`). Entry 0 is the end of setup; entry *k* is the end of cycle *k*.
* `:419`-`:429` `encode()` takes a 4th `cycleBudget` argument and `continue`s past every
  dispatch between the budget's boundary and `finishStart`. Setup and finish are always
  encoded.
* `:464` `cycleCount`, `:467` `planPassCount`, `:470` `encodedPassCount(budget)`,
  `:476` `clampCycleBudget`.

**`lib/methods/uniform/webgpu-uniform-reference.ts`** — host policy, telemetry, UI.
* `:179`/`:181` options `pressureCycleBudget`, `pressureBudgetHeadroom`; `:587` constructor;
  `:1167` `applyRuntimeValues` (both are live-toggleable, and flipping the mode clears the
  stale sample).
* `:1436` `planPressureCycleBudget()` — computes the budget and publishes the six `info`
  fields; `:1878` passes it into `encode`. It stands the budget down whenever the residual
  tolerance is zero (see the rule below).
* `:1468` `readPressureCycleDemand()` and `:1876`-`:1883`/`:1923` — a 12-byte
  `copyBufferToBuffer` of the cycle counters into the solver's own readback buffer, mapped
  asynchronously and never awaited in the frame path. It adds no compute pass, so it cannot
  move a stage seam, and it is skipped entirely while an earlier map is outstanding or the
  arm is `fixed`.
* `:1985` `readStats()` refreshes the same sample (guarded on `encodedSteps > 0`), so a caller
  that polls stats keeps the signal fresh too; `:2007` publishes the counters.
* `:2146` destroys the readback buffer.
* `:2443`-`:2480` the `pressure-cycles` panel stage gains the `Cycle budget` choice, the
  `Budget headroom` range (disabled under `fixed`), and the `Cycles encoded` / `Passes encoded`
  readouts. `:2528`-`:2556` rewrites the stage chip.

**`lib/methods/uniform/method.ts`** — `:32` `pressureCycleBudget` (Lagged | Fixed, default
**Lagged**, runtime), `:38` `pressureBudgetHeadroom` (default 1, 0-4, runtime), `:165`-`:166`
plumbed into the solver options. Both are inherited by Uniform Geometric through
`uniform-volume-method.ts`'s filter, so one edit gives both methods the control and both
panels the stage.

**`lib/core/webgpu-eulerian.ts`** `:414`-`:421` — eight new optional `GPUEulerianInfo` fields.

### The rule

```
if no sample yet              -> encode the configured schedule   (pre-P1 behaviour)
else if the sample converged  -> lastExecuted + headroom
else                          -> max(2 * lastExecuted, lastExecuted + 2)
clamped to [minCycles=1, configured]

and, ahead of all of it:
if residualTolerance <= 0         -> encode the configured schedule
```

**The zero-tolerance guard is load-bearing.** `pressureResidualTolerance = 0` is documented in
its own control as "runs every configured cycle", and it disables the GPU gate outright. With
the gate off, the executed-cycle counter reports the *schedule*, not the demand — it carries no
information at all — so a lagged budget reading it would silently cap a solve the operator had
just explicitly asked to run in full. The existing Dawn test
`pressure cycle convergence preserves parity, resets, and supports live tolerance` caught
exactly this: it converges a step at a huge tolerance, drops the tolerance to zero live, and
asserts the next step executes all 3 Full-Cycles; the first cut of P1 encoded 2 and failed it.
The guard is the fix, and the test passes on both arms without its pinned values being touched.

Asymmetric on purpose. The signal lags the encoded step by at least a frame, so shrinking is
done one cycle at a time (headroom, default 1) while growing is done by doubling: the cost of
shrinking too slowly is a few wasted passes, and the cost of growing too slowly is an
under-solved step, so they are not symmetric errors. The floor of 1 means a step whose
estimate is stale still projects against a coarse-corrected pressure rather than against the
previous step's field.

**Why the cycle order is unchanged.** Truncation is only legal at a checkpoint boundary. The
plan's bind groups are baked with fixed ping-pong parities; `checkpoint()` canonicalises
`p[0]` to parity A and `fullCycle()` restores `min[0]` to 0, so the always-encoded finish
section's bind groups are valid at every boundary and only at a boundary. Reordering the
schedule — V-Cycles first, or interleaved — would change which corrections the first cycles
carry and would need the plan rebuilt; truncating the existing Full-then-V order needs
nothing rebuilt. Sweep counts are untouched.

**Why the trace partition survives.** `encode()` still walks the whole plan and still fires
every stage boundary in order, skipping only the dispatches. A truncated section therefore
reads as a zero-length phase in the advance trace rather than vanishing from it, and the four
pressure phases stay exhaustive.

## Existing Dawn tests

Run one process at a time, foreground, on both arms. The off arm was produced by an in-memory
preload that flips the shared param default (`force-fixed-budget.mts`, beside this report), so
no repository file was edited to run it — another session is working in this tree today:

```
node --import tsx --import docs/research/uniform-pressure-granularity-2026-09-19/force-fixed-budget.mts \
  --test --test-concurrency=1 tests/uniform-volume-dawn.test.ts
```

| suite | lagged (default) | fixed |
|---|---|---|
| `tests/uniform-volume-dawn.test.ts` + `tests/volume-levelset-overlay-dawn.test.ts` | 9 pass, 1 subtest red | 9 pass, 1 subtest red |
| `tests/uniform-volume-tile-work-dawn.test.ts` | 1 pass | 1 pass |

The one red is `mini32 retains its liquid region through far-wall impact`, which fails on
**both** arms with the same number — frame 30 phi volume drift −0.0616753 (fixed) against
−0.0616575 (lagged), agreeing to four significant figures. The fixed arm is the pre-P1 command
stream, so this is pre-existing and P1 neither causes nor worsens it. No bound was weakened
and no pinned value was changed.

Note that `pressure cycle convergence…` constructs the solver directly through
`WebGPUUniformReferenceSolver.createAsync` rather than through the method params, so it runs
on the constructor default (lagged) on *both* arms — which is why it failed on both before the
zero-tolerance guard, and passes on both after it.

`npx tsc --noEmit -p .`: 15 errors, all pre-existing and all in `tests/sparse-cm12-*` and
`tools/*` — none in any file this work touched, and none in `lib/methods/uniform` at all.

## Checklist for the app

1. Open a scene on the Uniform or Uniform Geometric method and go to **SIM → Pressure cycles**.
   The chip reads `lagged · N of 7 cycles · M of 2638 passes`.
2. Watch `Cycles encoded` and `Passes encoded` while the water is still falling. On a scene
   that converges early they should settle to 2-4 of 7 and roughly 50-70% of passes.
3. Flip **Cycle budget** to **Fixed**. `Passes encoded` jumps to 100% and stays there; the
   frame time should rise by roughly the amount the readout says you gave back. Flip back —
   both directions are live, no reset.
4. Drop something in, or let the dam hit the far wall. The budget should visibly ramp back to
   7 within one or two steps and the water should not visibly jolt.
5. If anything looks wrong, set **Budget headroom** to 4 (nearly always the full schedule) or
   **Fixed** (exactly the pre-P1 stream) and compare.

## Caveats and what is not verified

* **The `fixed` arm is byte-identical by construction, and checked two ways**: the budget
  equals `cycleCount`, so the truncation `continue` never fires and the demand copy is never
  encoded; only host-side `info` fields are added. Empirically, mini64 encoded 2008 passes on
  both arms and the probe asserts the encoded pass count against the plan on every frame.
  It is *not* proven bit-identical by a hash comparison of the output field.
* **The demand signal is stale by at least one frame, by design.** It is a lagged controller,
  not a predictor. The frame-26 miss above is the shape of every worst case it can have.
* **Another live session was running its own Dawn probe during part of this work**, and it
  edited `webgpu-uniform-reference.ts` (params buffer 192 → 208 bytes) mid-session. All three
  captures in this report were re-taken *after* those edits settled, on one consistent tree,
  and all three report zero Dawn validation errors. Absolute medians may still be inflated by
  machine load; the paired deltas are the numbers to trust, because both arms saw the same
  load frame by frame.
* **Residual differences between arms are not all attributable to P1.** mini64 encodes the
  identical stream on both arms and still shows a 14% difference in worst residual; that is
  the float-CAS transport ordering floor for two independent solver instances.
* **The zero-tolerance guard was found by a test, not by design.** It is a reminder that the
  budget interacts with every other pressure control; the combinations that are *not* covered
  are listed below.
* **No unit tests.** Peter's instruction mid-task was not to add any. `uniformCM11aCycleBudget`
  is extracted as a pure function so the rule *can* be tested later without a device.
* **Not measured**: the lagged budget's interaction with rigid coupling (no rigid scene in the
  matrix), or with a non-default `pressureFullCycles`/`pressureVCycles` schedule. The rule
  reads `cycleCount` from the built plan, so a different schedule is handled, just unmeasured.
  All three captures ran at the default tolerance of 1e-4; a **zero** tolerance is now an
  explicit stand-down (verified by the Dawn test), but tolerances *between* zero and the
  default, and the maximum of 10 s⁻¹, are untested with the budget on. A very loose tolerance
  makes every step converge in one cycle, which is the regime P1 helps most and is also where
  it is most exposed to a sudden change in demand.
* **Not measured**: any scene at headroom other than 1.

## What P2 and P3 are now worth

P1 removes the passes that were never going to do work. What remains is the cost of the passes
that *do* run, and the picture changed:

**P2 — a resident coarse sub-hierarchy from the 16³ level down.** Still the largest remaining
item, and P1 makes it *more* attractive rather than less. After P1, large-power encodes 1488
passes a step, of which setup + finish is 23 and the rest are cycle passes; the coarse levels
are the ones that survive truncation, because truncation removes whole cycles from the tail
and every remaining cycle still walks the full depth. A single resident dispatch that runs the
bottom of the hierarchy in one launch would fold roughly the coarsest 4-5 levels of every
remaining cycle into one pass each. On fig7's 642-pass Full-Cycle the levels below 16³ are a
large minority of the passes and almost none of the arithmetic, so this is close to pure
launch-floor recovery: at the ~7.4 µs/pass measured here, folding ~200 passes a step is ~1.5 ms
GPU plus ~2 ms of CPU encode, on top of P1. This is the same shape as the Losasso resident
coarse-band solve, which is precedent that it works.

**P3 — fusing smoother / residual / restrict into one pass.** P1 does not reduce the *count*
of these inside a cycle at all, so P3's arithmetic is untouched by P1: at 6 pre + 6 post
sweeps, the smoother-residual-restrict triple is the bulk of every level of every encoded
cycle. Fusing three passes into one on the levels where the workgroup can hold the tile would
cut roughly a third of the remaining cycle passes. On fig7 post-P1 (mean 2113 passes/step)
that is ~700 passes, ~5 ms GPU and ~7 ms CPU encode — the largest single remaining number in
this report. It is also the most invasive: the fused kernel needs the level's tile plus halo
resident, and the projected red-black sweep's dependency pattern has to survive the fusion
exactly or the solve's convergence rate changes, which would then show up as a *higher* cycle
demand and partly cancel itself. Worth doing after P2, and worth measuring the cycle demand
before and after, not just the per-step time.

Ranking on measured evidence: P2 first (lower risk, the precedent exists, and it compounds
with P1 because it shortens every surviving cycle), P3 second and only with a convergence-rate
check in the same capture.

## Raw data

* `p1-cm12-figure-7.json` — 70 frames, both arms, full per-frame series.
* `p1-minimal-power-dam-break-64.json` — 60 frames.
* `p1-large-power-dam-break.json` — 60 frames.
* `force-fixed-budget.mts` — preload that runs any existing suite on the `fixed` arm.
* `probe-cycle-budget-dawn.mts` — the lockstep two-arm harness.
  `FLUID_P1_SCENE`, `FLUID_P1_FRAMES`, `FLUID_P1_OUT`, `FLUID_P1_ARMS`.
