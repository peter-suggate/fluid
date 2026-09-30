# Uniform Geometric at 60 Hz: temporal coherence plan (2026-09-30)

The app now steps at 1/60 s. A step costs about the same as a 1/30 s step
([retuning report](../benchmarks/uniform-sixtieth-retuning-2026-09-30.md)), so
the cost per simulated second nearly doubles. This plan decides which work can
follow the physics rather than the step count, and lands it.

## Finding: tile-level dirty tracking does not pay in a moving dam

We measured `high-resolution-dam-break` at 128³ and 1/60 s: consecutive
awaited steps, six sample times from 0.10 to 2.0 s, final canonical GPU
state. The probe lives in the session scratchpad. Shares of liquid tiles:

| Quantity, step k → k+1 | Range over the run |
|---|---|
| phi changed > 1e-4 h | 95–100% |
| surface sign pattern changed | 12–42% |
| V changed > 1e-5 | 97–99.8% |
| \|Δu\| < 1e-2 h/dt | 0.4–9% |
| all-4h pressure changed < 1% | 0.3–22% |
| skippable with a one-tile halo | **0–5%** |

- The median flow moves 5–13 h cells per step even at 60 Hz. Each step is
  not a small perturbation.
- Gravity and the global projection change u throughout the liquid.
- Bulk V changes by 1.5e-3 to 5e-3 per step (departure-map compression, see
  [local volume recovery](uniform-local-volume-recovery.md)).
- Far-air phi is an unclamped distance field that follows the surface.
- Only far air is static, and the active lists already skip it.
- k → k+2 gives the same fractions: the change saturates within one step.

## Where the waste actually is

A fit of per-frame stage times against work counts in the retuning JSON gives:

- **Fixed GPU cost per step:** 9–14 ms at 128³, 17–21 ms at 256³.
- **Marginal cost:** about 3.5 µs per h tile.
- **Layout churn per step halves at 60 Hz,** so churn per simulated second is
  flat. The waste is layout bookkeeping whose cost does not follow churn,
  plus dose-type passes, both now run twice per simulated second.

Three code audits (frame head and extension; surface, volume and momentum;
pressure) produced the workstreams below.

## Prerequisite: 60 Hz pressure reliability

- **128³ fails closed with the default lagged plan.** At frame 66 it
  accepted 6.59 after 3 cycles against tolerance 5.
- **256³ fails at frame 52:** residual 5.03 after 2 cycles.
- **Why:** frame N encodes the slot list that frame N−2 planned (cycles run +
  1 spare). An impact frame whose initial residual jumps about 100× has no
  lagged signal. The full 4V+3F envelope costs +5.47 ms/step at 256³ and is
  diagnostic only.
- **The h band does not converge at 128³:** median band residual 7.3 s⁻¹,
  p90 148. The band always runs 4 cycles with target 0, and the frame verdict
  does not check the band residual.

Constraints (Peter, 30 September): fail closed, no fallbacks, no loosened
tolerance, no added work without measured justification.

## Workstreams

### WS-P: pressure reliability (prerequisite)

1. **In-frame coarse accuracy.**
   - The slot-0 gate already knows this frame's initial residual
     (`state[2]`), and coarse accuracy is a GPU word
     (`uniform-mixed-pressure-schedule.ts`).
   - When the initial residual jumps relative to the last accepted frame's
     initial residual (a new persistent control word), start at accuracy 0.
   - No launches are added. It works only if the slow contraction comes from
     the inexact coarse solve.
2. **If (1) is insufficient,** find what else the gate can decide inside the
   encoded envelope:
   - send a stalled V phase to the Full slots sooner;
   - pick the slot kind by initial residual.

   The rule must be derived from observed contraction, not a timer.
3. **Band convergence at 128³:** diagnose first (where the band residual
   lives; whether the 1-workgroup coarse solve or theta rows cause it). Land a
   fix only if it does not add net frame time. Record the finding either way.

**Gates:**
- 128³ runs 120 frames at 1/60 with the default plan.
- fig7-256 runs 60 frames at 1/60.
- No added frame time on frames that don't jump.

### WS-S: dose cadence and solver exits

1. **Dose cadence by simulated time.**
   - Sharpening (8 sweeps, no dt factor) and global surface-volume correction
     (a projection) run when at least 1/30 s of simulated time has
     accumulated since their last run.
   - At 1/30 s nothing changes. At 1/60 s they run on alternate steps.
   - The clock is host-side, with no GPU gating and no empty launches.
   - **Quality check:** excess/total V, max V/capacity, V in air, band
     thickness, at equal simulated time against the base. At CFL > 1 the
     resampling diffusion per step does not fall with dt, so sharpening's
     dose may genuinely be needed per step. Measure before claiming it.
2. **Redistance Newton tolerance exit.** The loop breaks only when |phi|
   stops shrinking (`uniform-mixed-surface.ts`, the Newton search). Add an
   exit at |phi| ≤ 0.1·tol.
3. **Surface-volume secant warm start.** Seed from the last applied shift ×
   dt ratio instead of 0. Drop to one round only if the secant still hits
   its target (the receipt's relative residual).
4. **Drain per simulated time.** `min(value+0.5h, 0.5h)` retreats 0.5h per
   step. Scale it by dt·30 so the 30 Hz rate is preserved.

### WS-H: head work driven by the list of changed tiles

1. The layout builder already knows which tiles changed width. Have it emit
   a compacted list on the GPU (a counted-list mode, direct grid-stride
   launch).
2. Drive these from that list, dilated by the 3³ neighbourhood where the
   pass needs it:
   - remap `markChanged` (today a lattice scan × 27 loads);
   - geometry `markChanged`/`compactChanged` (two lattice scans);
   - solid `encodeSimulation`;
   - the head phiResolve.
3. Stop `adoptGpu` copying support words [0, 4n): the plan overwrites them
   (`uniform-mixed-ownership.ts`). Skip solid `encodeSimulation` when the
   scene has no cut tiles and no bodies.
4. **Stretch:** patch the tail plan and phase within the dilation reach
   instead of re-running them at the head.

**Gate:** the relayout head cost follows churn. Expected 1–2 ms at 128³.

## Rejected (with reasons)

- **Extension as a cached gather:** phase flips every step at the surface,
  and owners are renumbered on every relayout.
- **Relayout every other step:** saves 1.3–1.9 ms and costs 2.7–3.6 ms of
  extra h tiles.
- **Root pressure warm start in a dam:**
  - The root already runs one cycle.
  - Impact frames have no previous pressure to seed from.
  - It cuts the planner's headroom.
  - Revisit for calm pools, with a p=0 reference bound.
- **Setup dirty bits:** launch-bound, gain ≤ 0.3 ms.
- **Hydrostatic split:** equivalent to a warm start.
- **Momentum skip:** fails in a moving pool.
- **Newton closest-point cache:** 270 MB at 256³.
- **Saturated-transport certificate:** blocked until face-traced boxes remove
  the departure-map defect.
- **Stable owner addressing (64·tile + lane):** the enabler for any
  cross-step owner cache, but a broad refactor. Deferred.

## Method

- **Target scene:** `high-resolution-dam-break`, 1/60 s. Instrumented: 60
  frames, discard 0.133 s, `--split-stages --quality-census`. Throughput:
  `--throughput`, 120 frames.
- **Reliability scenes:** 128³ for 120 frames, and fig7-256 for 60 frames.
- **Isolation:** each workstream works in its own scratch tree, a snapshot of
  the real tree. Patches are merged back with `git merge-file`.
- **GPU budget:** at most 3 runs per workstream, serialised on the lease.
  CPU type check and shader preflight come before each run.
- **Tests:** no suites. Only targeted lanes at the end
  (`uniform-long-dam-front`, pond-rest, dynamic coarsening where touched).

## Results (2026-09-30, merged into the working tree, uncommitted)

All three workstreams were merged with `git merge-file` without conflicts. The
lanes then showed that the surface-volume warm start (WS-S) broke
`uniform-long-dam-front-dawn`, so it was reverted; see the WS-S section. The
**final** tree is base + WS-P + WS-H + the rest of WS-S. Measurements on
Dawn/Metal, M1 Max:

| 128³ dam, 1/60 s | Base | Combined (with warm start) | Final |
|---|---:|---:|---:|
| Throughput wall, ms/step (60 frames) | 39.87 | 38.72 | 40.03, 40.24 (two runs) |
| Instrumented GPU mean, ms/step | 41.14 | 38.50 | 40.98 |
| Mean h tiles per step | 8123 | 7715 | 8393 |
| 120 frames, default pressure plan | fails frame 74 | passes (33.41 GPU) | passes (33.93 GPU) |
| fig7-256, 60 frames, GPU ms/step | 35.22 | 33.84 | 34.11 |

Frame time is not better on the target scene.

- **The combined tree's apparent 2.6 ms gain was mostly a trajectory
  effect.** The warm start ran 5% fewer h tiles, and every stage fell with
  it.
- **The final tree runs 3.3% more h tiles than base**, about 1 ms at
  3.5 µs/tile. That cancels the structural saving.
- **What does hold:**
  - the relayout-head saving (census + layout build + remap, 2.93 → 2.13 ms);
  - fig7-256 at −1.1 ms;
  - the 60 Hz pressure reliability fix, which is the real deliverable.

Stage changes on the dam (GPU ms/step):

| Stage | Base | Combined | Final |
|---|---:|---:|---:|
| Census + layout build + remap | 2.93 | 2.05 | 2.13 |
| Redistance | 3.48 | 3.17 | 3.49 |
| Global surface volume | 2.74 | 2.44 | 2.82 |

The Newton exit's redistance saving does not show once the h-tile count is
back up.

Final-frame quality is excess 9.00% (base 8.84%) and max V/capacity 191
(base 142). The max V/capacity figure is a single-frame number, noisy across
arms.

Lanes on the final tree:

- `uniform-pond-rest-dawn`: test 1 passes. Test 2 ("resting pond conserves
  mass") fails at base too.
- `uniform-long-dam-front-dawn`: fails with base's front-position failure,
  not a new failure.

### WS-P: pressure plan window

- **Rule.** `umPlan` in `uniform-mixed-pressure-schedule.ts` now plans at
  least the largest cycle count of the last four accepted frames, plus the
  spare. The last three counts are packed in control word 11. Everything
  stays within the existing 4V+3F cap: no launches added, no tolerance
  changed.
- **Why it failed before.** After the impact, the cycles a frame needs
  swing between 1 and 3 from one frame to the next. The lag-2 "ran + 1" plan
  under-planned whenever a dip was followed by a rise.
- **Cost.** About 0.1 extra encoded slots per frame, which is noise in frame
  time. fig7-256 is bit-identical to base.
- **Null: coarse accuracy.** Tightening coarse accuracy changed nothing.
  The coarsest solve takes 1–2 iterations at any accuracy. The binding limit
  is root V-cycle contraction: about 25× on the first cycle, then 0.25–0.48
  per cycle.
- **Diagnostic trace.** `FLUID_MIXED_PRESSURE_TRACE=1` logs each frame's
  initial residual, per-slot residuals, coarse iterations and band history.
- **Band diagnosis, no fix.** The band's contraction degrades each cycle
  (0.11, 0.26, 0.34, 0.68). At 128³, 53 of 120 frames end above 5 s⁻¹. This
  is the signature of unsmoothed piecewise-constant aggregation
  (coarseScale 0.6). The fix is a better band coarse correction, not more
  cycles.
- **Risk.** The four-frame window is empirical. A larger swing still fails
  closed, as before.

### WS-S: surface-volume warm start, Newton exit, drain rate

What landed:

- **Redistance Newton exit** at |phi| ≤ 0.1 × the acceptance tolerance.
- **Drain rate per simulated time.** The drain retreats 15·dt·h per step,
  which equals 0.5h per 1/30 s. At 1/30 s it is identical to before.

Merged, then reverted after the lanes:

- **Surface-volume warm start with one round.**
  - What it did: the search was seeded from the last applied shift × the dt
    ratio, using 48 bytes of persistent memory, and the default rounds
    dropped from 2 to 1.
  - On the dam at 1/60 s:
    - The seed left 0.22–0.28% mismatch, against 1.25–1.37% unseeded.
    - Round one then left 2e-4.
    - The stage cost fell by 0.3 ms.
  - Why it was reverted: `uniform-long-dam-front-dawn` then failed at frame 12
    in the fine arm at dt 1/30. The error was "pressure did not converge
    (accepted 8.54 after 4 cycles)". The base fails that lane differently:
    wrong front positions, no fatal.
  - Bisect: the warm start alone reproduces the fatal. The Newton exit alone
    matches base exactly in this lane and in `uniform-pond-rest-dawn`.
  - The pond shoreline error (0.518 mm against the 0.5 mm limit) appeared
    only with both changes together. It is gone after the revert.
  - Lesson: one seeded round is not equivalent at 1/30 s, where one step's
    mismatch is twice as large. A retry would need a certificate: keep
    solving until the residual is below the two-round level. It is not a
    fixed-round saving.

Rejected on measurement:

- **Surface-volume cadence every 1/30 s.** Redistance erodes phi by about
  1.4% of volume per step. Two steps' worth exceeds the one-cell shift limit,
  so phi fell 1.9% below V. The frame got faster only because phi got
  thinner.
- **Sharpening cadence every 1/30 s.** The stage saved 2.1 ms. But h tiles
  rose 3.4%, excess went from 8.84% to 9.01%, and max V/capacity from 142 to
  151. Throughput worsened by 0.45 ms. So a per-simulated-time dose is not
  right for these passes at CFL > 1.

Quality at t = 1 s, base vs combined (measured before the warm-start revert):

| Measure | Base | Combined |
|---|---:|---:|
| Excess | 8.84% | 8.14% |
| Max V/capacity | 141.9 | 146.4 |
| V drift | −0.13% | −0.09% |
| V in centre-phi air | 4708 | 6622 |

V in centre-phi air is a single end-frame number that is noisy across arms.

### WS-H: changed-tile list drives the relayout head

- **The list.** The builder's `widths` kernel appends each tile whose width
  changed. A `dilate` kernel produces the deduplicated 3³ neighbourhood.
- **What now runs over the list instead of the lattice:** remap
  `markListed` (plus exact early exits, and five passes merged into two),
  geometry's changed launch, solid `widthsListed`, and the head's
  `resolveListed` phi resolve.
- **`adoptGpu` no longer copies** the support words [0, 5n+16) that the
  frame plan rebuilds.
- **Fail-fast guard.** `ownership.revision` makes the geometry changed
  launch throw unless exactly one adoption intervened.
- **Exact.** Every per-frame count and the final quality match base on the
  dam and on fig7. `uniform-dynamic-coarsening-dawn` passes.
- **Gain.** The head drops from about 3.08 to 2.35 ms at 128³ and by 0.85 ms
  at 256³.
- **Stretch not attempted: patching the head plan.** The plan seed reduces
  a global maximum face speed, and remap can lower it, so a patch limited to
  the dilation reach cannot be exact.
- **What remains.** Per-listed-tile remap work, about 0.9 µs per changed
  tile.

### What this means for 60 Hz

The durable results:

- 128³ dam at 60 Hz runs 120 frames on the default pressure plan.
- The relayout head is 0.8 ms cheaper.
- fig7-256 is 3% faster.

Throughput on the 128³ dam is flat within trajectory noise. Moving scenes do
not offer the large temporal-coherence wins. The next levers are
structural, all listed above:

- root and band multigrid contraction (fewer cycles and cheaper spare slots);
- the transport departure-map defect, which blocks interior skipping;
- per-step fixed launch cost.
