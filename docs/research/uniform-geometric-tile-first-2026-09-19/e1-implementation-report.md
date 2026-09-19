# Volume dust floor and the E1 two-level velocity sampler

2026-09-19. Uniform Geometric (`uniform-volume`). Both features are live runtime
controls on the pipeline panel; neither is committed.

Raw measurements: `dust-arms.json`, `e1-arms.json` (both beside this file).
Scripts: `/private/tmp/claude-501/-Users-petersuggate-code-me-fluid/4dd2ff94-6fba-4892-a1dc-4a2c63f35fd8/scratchpad/{dust-arms,e1-arms,check-uniform-wgsl}.mts`.

---

## Part 1 — where the dust comes from, and the floor

### Where it is created

The census at `fig7-census.json` (70 steps of `cm12-figure-7`, captured before
this work) already held the answer, so no extra Dawn run was spent finding it.

The negatives' magnitudes are the tell: `minCellV` is **-5.96e-8 = -2^-24** and
**-1.49e-8 = -2^-26** at every sample. Those are float32 ULPs of numbers near 1
and near 0.25 — the residue of a subtraction of two nearly equal values, not a
physical undershoot.

- `uvGather` (`uniform-volume.wgsl.ts:236`) cannot make a negative: every
  `weight[k]` and every `volume()` it sums is non-negative. What it *can* make
  is a **tiny positive**: a trilinear corner weight of ~1e-8 times a full cell
  lands ~1e-8 of liquid in a cell the characteristic barely grazed. That is the
  dust *population*.
- `uvCommitSharpen` (`:352`) writes `volume(id)+d4Sum6(terms)`, a signed flux
  sum. When the six limited fluxes very nearly cancel the cell's own V, the
  result is one ULP either side of zero. That is the dust's **negative half**,
  and the census confirms the order of appearance: 0 negatives at step 1, 583 at
  step 5, 280,098 at step 55.
- `uvAddDonor`'s float CAS and `semi-Lagrangian`/`project` never write V, so
  they are not sources. The CAS *is* why two identical runs differ (below).

Once created, nothing removes dust: a 1e-8 cell is a legal donor for the next
step's gather, so the population ratchets. The mass involved is negligible
(1.3e-7 of the frame at step 60) but the **cell and tile count is not**, and
that is what any work map pays for.

### The floor

`uvDustFloor` (`uniform-volume.wgsl.ts:228`), applied at the three places V is
finally stored for a step: the transport gather (`:240`), the sharpening commit
(`:352`) and the commit's inactive-tile copy path (`:346`, so the 4h work map
stays bit-identical to the dense control). Threshold is `params.tuning.z` —
previously written as a literal `0` and read by nothing.

```wgsl
if(value==0.0||!(abs(value)<params.tuning.z)){return value;}
```

The negated comparison makes **threshold 0 an exact no-op**: `abs(x) < 0` is
false for every float including NaN, so the control arm stores the untreated sum
bit for bit and no pipeline variant is needed. Tiny negatives are covered by the
same `abs()` test — there is no separate clamp and no redistribution scheme.

Cost telemetry: `reductions[5]` counts cells zeroed, `reductions[6]` accumulates
the discarded mass in sixty-fourths of the threshold (words 5–7 were dead; the
"4..6 are transient volume-control totals" comment at
`webgpu-uniform-reference.wgsl.ts:46` was stale). `readStats` now copies 32
bytes instead of 20 (`webgpu-uniform-reference.ts:1720`) — bytes 20..31 of the
208-byte readback were free.

### Measured, threshold 0 vs 1e-6

`cm12-figure-7`, 128³, 32,768 tiles of 4³, 70 steps:

| step 70 | off | on (1e-6) |
|---|---|---|
| V≠0 cells | 530,099 | **127,841** (−75.9%) |
| cells \|V\|>1e-6 | 132,448 | 127,841 |
| negative cells | 189,125 | **0** |
| occupied 4h tiles | 9,411 (28.7%) | **3,159 (9.6%)** — 2.98× fewer |
| V sum | 33463.9246 | 33463.9184 |
| cumulative relative drift from t=0 | +7.4e-7 | **−2.4e-6** |
| cells zeroed on the last step | — | 11,474 (4.2e-3 cell volumes) |

`minimal-power-dam-break-64`, 64³, 4,096 tiles, 60 steps:

| step 60 | off | on (1e-6) |
|---|---|---|
| V≠0 cells | 248,904 | **153,538** (−38.3%) |
| negative cells | 512 | **0** |
| occupied tiles | 4,001 (97.7%) | **2,828 (69.0%)** |
| cumulative relative drift | +1.05e-6 | **+2.0e-7** |

So the floor is a **3× reduction in live tiles on figure 7** and a 1.4× one on
the dam break, for a drift that stays in the parts-per-million. On figure 7 it
turns a +7.4e-7 spurious *gain* into a −2.4e-6 *loss*; on the dam break it
reduces the drift magnitude five-fold. Neither arm has a mass problem — the
point of the floor is the population, not the mass.

Zero validation errors in either arm.

### UI

`volumeDustThreshold`, number, tier `fine`, runtime update, default **1e-6**,
range 0..1e-3 (`uniform-volume-method.ts:33`). Surfaced on the **Conservative
volume transport** stage (`uniform-volume-pipeline.ts:73`) as a slider plus a
"Dust discarded" readout reading `N cells · M cell volumes` from the published
words, and the stage chip reads `dust floor 1e-6` / `dense finest lattice`.

Test: `tests/uniform-volume-initial.test.ts`, "the volume dust floor is a live
number surfaced on the transport stage".

---

## Part 2 — E1, the two-level velocity sampler

### What was built

Everything lives in **words [N, 2N) of `conditioningScratch`** — a region the
geometric path never addresses. No new binding, no new texture, no new bind
group, so the compute group stays at its ten storage buffers.

To make that region survive the step, the two **unranged**
`encoder.clearBuffer(this.conditioningScratch)` calls in `encodeGeometricVolume`
are now ranged to the N-word donor-sum region
(`webgpu-uniform-reference.ts:1295-1298`). This is safe: `uvSumDonors`,
`uvFallback` and `uvNormalizeDonors` only address `[0,N)`; the liquid-balance
header at `2N..2N+6` is explicitly `atomicStore`d by `uvBeginLiquidBalance`
every round; and the 4h sharpening map at `2N+8..` is fully rewritten by
`uvClassifySharpenTiles` after those clears every step. The audit flagged these
clears as the trap that would wipe a live map; narrowing them is the fix.

Four passes, encoded once per step immediately after
`encodeVelocityExtrapolation` and before the first velocity sample
(`webgpu-uniform-reference.ts:1444-1452`), dispatched over the ceil(n/4)³ tile
grid:

1. **`uvTwoLevelRestrict`** (`uniform-volume.wgsl.ts:394`). Restricts the
   *extended transport field* — the same padded `transportIn` the fine path
   samples — onto the 4h faces: a coarse upper face is the area mean of the
   sixteen fine upper faces it covers. Then SEEDs the tile if any cell has
   `|V| > dust` (1e-6 when the dust param is 0), or `uvOpen(cell) < 0.99999`, or
   `dropSource`/`inflowSweptPlugSource` is positive this step, or any of its
   5³ owned vertices has `|phi| < 4h`.
2–4. **`uvTwoLevelDilateX/Y/Z`** (`:426`, `:434`, `:442`). Chebyshev dilation by
   k tiles, separated into three axis scans of 2k+1 taps each (a single-pass
   (2k+1)³ scan would be 4,913 taps at k=8). Two single-word planes above the
   table are the ping-pong; the z scan lands the final class back in the table
   and adds the fine count to `reductions[7]`.

The sampler branch is the single choke point, `sampleVelocityComponent`
(`webgpu-uniform-reference.wgsl.ts:265`):

```wgsl
if(params.physical.z>=0.0&&!uvTwoLevelFineAt(p)){return uvCoarseVelocityComponent(p,component);}
```

`uvCoarseVelocityComponent` (`uniform-volume.wgsl.ts:381`) is the fine formula
with `dims()→uvCoarseDims()` and `p→0.25*p`. I verified the face-lattice
convention algebraically: `q=p-offset`, `textureLoad(...,base+o+1)` and the
one-texel shell together put `velocity(cell i)[c]` at lattice coordinate i+1 on
axis c, i.e. cell i's **upper** face — so coarse face t is fine face 4t+3,
lattice 4(t+1), and a plain quarter of the lattice coordinate carries the
convention across.

### Decisions you should know about

- **Rigid bodies and sources seed rather than force the toggle off.** Partial
  `cellOpenFraction` covers rigid bodies, terrain and cut cells in one test, and
  the drop/inflow source functions are evaluated directly. Nothing forces E1 off
  for a scene.
- **No blending at the level interface.** A sample just inside a fine tile gets
  the fine trilinear; just outside, the coarse one. They differ by the
  restriction error, so there is a jump. E1 exists to *bound* that jump; a blend
  would hide exactly the number being measured. If the numbers below justify
  shrinking anything, a blend band is the obvious next knob.
- **The closed wall is the same at both levels.** `uvCoarseFace` returns 0
  outside `[0, coarseDims)`, which is the `sampledFaceVelocity` rule the design
  doc asked for, and it reproduces the fine path's padded zero shell exactly
  (the `clamp` bounds `base+o` to `[-1, dims]`, and index `dims` is the shell).
- **The off arm is a uniform branch, not a pipeline variant.** `physical.z` is
  −1 with the experiment off, so the branch is never taken and the returned
  value is bit-identical; the four map passes are not encoded at all. This keeps
  the toggle **live** (one `writeBuffer`, no rebuild) instead of doubling every
  advection pipeline. If the branch's issue cost ever matters, the
  `override`+variant pattern is a drop-in follow-up.

### (a) Same-input replay, `cm12-figure-7`

From identical code over the prefix, one step off vs one step on. The **off/off**
row is the float-CAS noise floor: `uvAddDonor` accumulates by
compare-exchange, so two runs of identical code are not bit-identical and *every
on/off number must be read against it*.

| prefix | arm | φ, band \|φ\|<4h: max / mean | V, cells>dust: max / mean | u, band: max / mean | fine tiles |
|---|---|---|---|---|---|
| 10 | off/off (noise) | 4.99e-7 / 4.22e-9 | 7.27e-6 / 3.31e-8 | 9.54e-7 / 2.28e-8 | — |
| 10 | on k=2 | **5.66e-7 / 4.92e-9** | **7.21e-6 / 3.88e-8** | **9.54e-7 / 2.60e-8** | 3,516 (10.7%) |
| 25 | off/off (noise) | 8.05e-7 / 7.04e-8 | 2.867e-2 / 4.30e-6 | 3.09e-5 / 3.06e-7 | — |
| 25 | on k=1 | 8.05e-7 / 7.04e-8 | 2.866e-2 / 4.31e-6 | 3.45e-5 / 3.07e-7 | 2,092 (6.4%) |
| 25 | on k=2 | 9.09e-7 / 7.10e-8 | 2.866e-2 / 2.72e-6 | 2.19e-5 / 3.03e-7 | 3,156 (9.6%) |
| 25 | on k=3 | 7.73e-7 / 6.92e-8 | 2.866e-2 / 4.30e-6 | 3.60e-5 / 3.13e-7 | 4,436 (13.5%) |
| 40 | off/off (noise) | 6.92e-2 / 5.93e-6 | 5.52e-1 / 7.10e-5 | 8.00e0 / 4.89e-4 | — |
| 40 | on k=2 | **3.98e-1 / 9.06e-5** | **1.31e1 / 1.87e-3** | 8.00e0 / 3.60e-3 | 4,184 (12.8%) |

Unrestricted ("everything") differs sharply from the band, and that is the most
informative row in the file. At prefix 10, **φ max over the whole lattice is
7.56e-2 while φ max inside the band is 5.66e-7**. The two-level sampler moves
the far field — where φ is metres from zero and nothing reads it — and leaves
the surface at the noise floor. Same at 25 (7.94e-2 vs 9.09e-7).

Reading:

- **At steps 10 and 25 (free-fall and approach) E1 at k=1..3 is
  indistinguishable from the CAS noise floor** everywhere that is read: band φ,
  liquid V, band velocity. The effect is confined to far-field φ.
- **At step 40 (the first frames after impact) it is not.** Band φ max is 5.7×
  the noise floor and band φ mean 15×; V max over liquid cells is 24× (13.1 vs
  0.55 cell volumes). Caveat: the *noise floor itself* is enormous at step 40 —
  0.55 cell volumes of V difference from a float-CAS reordering alone — so the
  flow is in a regime that amplifies a ULP into a visible difference within one
  step. E1's perturbation is larger than the noise, but "larger than an already
  chaotic noise floor" is as strong a statement as this experiment supports.
- **k is not the lever at step 25.** k=1, 2 and 3 all sit at the noise floor
  while the fine set moves 6.4% → 13.5%. The reach matters for what a future
  shrink would cost, not for one step's numerics here.

### (b) Trajectory sanity, 70 steps

| step | off: V sum / drift / maxSpeed | on k=2: V sum / drift / maxSpeed | fine tiles (on) |
|---|---|---|---|
| 10 | 33463.9994 / +2.00e-6 / 3.334 | 33463.9994 / +2.01e-6 / 3.334 | 3,484 (10.6%) |
| 20 | 33463.9996 / +9.76e-6 / 6.668 | 33463.9996 / +9.65e-6 / 6.668 | 3,552 (10.8%) |
| 30 | 33464.0005 / −1.40e-6 / 20.120 | 33464.0005 / −1.42e-6 / 20.120 | 2,156 (6.6%) |
| 40 | 33463.9850 / −2.96e-6 / 15.513 | 33463.9846 / −3.36e-6 / 14.766 | 4,180 (12.8%) |
| 50 | 33463.9697 / −7.12e-6 / 19.140 | 33463.9695 / −9.47e-6 / 18.198 | 6,253 (19.1%) |
| 60 | 33463.9527 / −1.91e-5 / 4.472 | 33463.9492 / −1.75e-5 / 4.422 | 7,507 (22.9%) |
| 70 | 33463.9191 / −2.26e-5 / 8.326 | 33463.9148 / −2.71e-5 / 7.699 | 7,178 (21.9%) |

- **Mass is unaffected.** The two V sums agree to 1.3e-7 relative at step 70;
  both drift by ~2.5e-5 over 70 steps, which is the dust floor's own cost (both
  arms run it at 1e-6), not E1's.
- **maxSpeed tracks to three significant figures until impact** and then differs
  by 5–8% (15.51 vs 14.77 at step 40; 8.33 vs 7.70 at step 70). Directionally
  the on arm is slightly slower, which is what a restricted velocity field does,
  but the noise floor above says a chaotic run would show differences of this
  order anyway.
- **Liquid extent is identical through step 40** (the free-fall column and the
  floor spread agree cell for cell) and diverges in splash height afterwards:
  the top of the liquid at step 70 is y=58 off vs y=67 on. Same caveat.
- **Fine tiles run 6.6%–22.9% of the 32,768-tile grid** at k=2, mean around 14%.
  That is the number a future shrink would be buying against.

### (c) Cost

Wall clock per step over 70 steps, in-process, readbacks excluded from the
timer: **off 205.9 ms/step, on 220.0 ms/step — +6.9%**. E1 shrinks nothing, so
this is pure added work: four dispatches over 32,768 threads plus a
`conditioningScratch` load per velocity sample. **Per-stage GPU timestamps were
not captured** — the stage probe is a separate harness and this is a
numerics-only experiment, so the frame-level number is what I measured. If you
want the stage split, `probe-fig7.mts` still works against this tree.

### UI

- `twoLevelVelocity`, select on/off, runtime, default **off**
  (`uniform-volume-method.ts:36`).
- `twoLevelFineReach`, number 0..8, runtime, default **2** (`:39`).
- Both on the **velocity extension** stage (`uniform-volume-pipeline.ts:123`),
  beside the sweep budget that produced the field E1 restricts. The stage chip
  appends `E1 two-level · 14% fine`; the "Fine tiles" readout reads
  `4184 / 32768 (13%)` from `reductions[7]`. The existing `extensionFrontSweeps`
  slider and "Sweeps with work" readout are preserved and asserted in the test.

Test: `tests/uniform-volume-initial.test.ts`, "the two-level velocity sampler is
a live experiment on the velocity-extension stage".

---

## Caveats and what is not verified

1. **Transport is not bit-reproducible.** `uvAddDonor`'s float CAS means two
   runs of identical code differ; the off-arm V≠0 count at step 20 was 57,057
   here against 56,924 in the pre-change census. Every on/off comparison above
   carries an off/off control for exactly this reason, and the step-40 rows in
   particular should not be read as "E1 changes the answer by 13 cell volumes"
   without the 0.55 beside it.
2. **The step-40 replay is one sample.** I did not repeat it; a second off/off
   pair would say how reproducible that noise floor is.
3. **k was swept only at prefix 25**, where all three values sit at the noise
   floor — so the sweep says nothing about whether k=2 is enough *at impact*.
   That is the experiment I would run next.
4. **No blend at the level interface**, by choice (above).
5. **Per-stage GPU timings not captured** (above).
6. **The dust floor's default is on at 1e-6.** That changes the shipped
   behaviour of Uniform Geometric. Set it to 0 in the panel for the exact
   previous numerics — it is bit-identical there, not approximately so.
7. **The floor discards mass and is not conservative.** ~4e-3 cell volumes per
   step out of 33,464 on figure 7 at step 70. Over 70 steps that is the −2.4e-6
   relative drift in the table. A conservative variant would donate the residue
   to a neighbour; the brief asked for the simple win, and the simple win is a
   3× tile reduction for two parts per million.
8. **`minimal-power-dam-break-64` was run for Part 1 only.** E1 was measured on
   `cm12-figure-7` alone.
9. **Unverified:** `maxCellV` reaches 55.2 cell volumes in the off arm around
   step 50 (it is in `fig7-census.json` too, so it predates this work). That
   overfill is a separate pathology and it is why the step-40 V differences are
   measured in whole cell volumes.

## Checks run

- `npx tsc --noEmit`: 15 errors, all pre-existing in `tests/sparse-cm12-*` and
  `tools/probe-sparse-cm12-*`. **None in any file touched here.**
- `tests/uniform-volume-initial.test.ts`: 7/7 pass (5 pre-existing + 2 new).
- `tests/host-transport-status.test.ts`, `tests/cm12-paper-step-default.test.ts`,
  `tests/oak-initialization-cancellation.test.ts`: 11/11 pass.
- `tests/uniform-volume-tile-work-dawn.test.ts`: pass — the 4h work map is still
  bit-identical to the dense control with the floor applied on both paths.
- `tests/uniform-volume-dawn.test.ts`: 11/11 pass.
- WGSL: both module variants compile on Dawn with zero messages, and all 26
  geometric entry points build a compute pipeline with no validation error
  (`check-uniform-wgsl.mts`).
- Both measurement runs asserted an empty `uncapturederror` list.

## What to look at in the app

1. **Uniform Geometric → SIM tab → Conservative volume transport.** Run
   `cm12-figure-7` past the impact (~step 40) and drag **Dust floor** between 0
   and 1e-6 while it runs. The "Dust discarded" readout moves live; the
   structure field should look unchanged. It is the *occupancy* that changes —
   at step 70 the liquid occupies 9,411 tiles at 0 and 3,159 at 1e-6.
2. **Velocity extension stage → Sampler.** Flip **Two-level** on and off while
   running. No rebuild, no reset to t=0. Watch the chip and the "Fine tiles"
   readout (expect 7–23% on figure 7). Then move **Fine reach** 0→8 and watch
   the fine set grow; at k=0 only seed tiles stay fine, which is the most
   aggressive setting and the one most likely to show an artefact.
3. **The thing to judge:** whether the surface under Two-level looks the same as
   under All fine. The numbers say the band is untouched before impact and
   perturbed after it — but perturbed inside a regime where a float reordering
   is also perturbing it. Your eye on the splash is the measurement I cannot
   make.
