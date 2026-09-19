# Uniform Geometric: making phi agree with V, locally

Status: PLAN, VALIDATED ON THE SCENE 2026-09-19 — see "Validation" directly below. Nothing is implemented in
production code; every stage was run as a shader rewrite injected by the probe (`PATCHES` in
`probe-surface-noise.mts`), with `volumePressureRows` OFF (its default since `abandoned` did not work out in the app).
Evidence: `docs/research/uniform-geometric-thin-film-2026-09-19/` — `probe-surface-noise.mts`,
`wp0-tile-shift-*.json`, `surface-noise-dam-break*.json`, `thin-ledger-*.json`. Predecessor commit: `a0348a5c`.

## Validation (2026-09-19, one Dawn run per arm; single runs of a chaotic scene, so treat ±25% on roughness as noise)

**water-box-dam-break, means over t = 3.3–6 s; shells at t = 6 s**

| arm | deep-interior mean V | phi volume − ΣV | roughness (cells) | band jitter (m/s) |
|---|---:|---:|---:|---:|
| HEAD (rows off) | 0.34, no cell full | **+136 (+8.1%)** | 0.0223 | 0.041 |
| + compaction (WP1) | **1.00, all full** | −107 (−6.4%), 92% of it now in the band | **0.0129** | **0.032** |
| + shift, gain 0.25 / clamp 0.1 cell | 1.00 | −12 | 0.0398 ✗ | 0.067 ✗ |
| + shift, gain 0.05 / clamp 0.01 cell | 1.00 | −43 | 0.0166 | 0.042 |
| + seed (all three, slow shift) | 1.00 | **−21 (−1.2%)** | 0.0201 | 0.043 |

Resting pool (`hydrostatic-power-large-offset`, all three): bit-still for 60 frames, max speed 0.

**corner-brick-drop, rows off, t = 3 s**

| arm | liquid centres | phi volume / 512 | max V | state |
|---|---:|---:|---:|---|
| HEAD | 0 | 165 | 10.0 | frozen from frame 30 |
| + compaction | 0 | 170 | 10.8 | frozen |
| + shift (0.25 / 0.05) | 0 | 156 | 8.8 | frozen by frame 60 |
| + seed | **512** | **472 (92%)** | 1.17 | alive, V > 1.5h from any phi surface: 289 → 5.5 |

What that settles:
- **WP1 is a pure win and should land first.** It fills the interior, *smooths* the surface (fewer Sec. 3.7
  excess events), and flips the residual to the right sign and into the band. It also reveals that HEAD's +8% phi
  volume was inflation driven by surplus V, not bubble deletion alone: compact V and phi *loses* 6% in 6 s.
- **The shift works but must be slow.** The drift it has to cancel is ~0.002 cell/step. A 0.1-cell clamp lets
  transient regional V/phi mismatch (±0.4 cell, smooth) drive phi at 0.15 m/s, which makes real waves. At
  0.01 cell/step it is inside the baseline's noise and removes 60–80% of the drift.
- **Disjoint 4h tiles are dead; overlapping cell tents are required.** Lateral roughness of the implied shift per
  unit gain: disjoint tiles 0.13, [1,2,1]³-blurred tiles 0.07, cell tent radius 2: 0.017, radius 4: **0.005**
  (phi's own roughness is 0.01–0.02). Tiles split R from A wherever the surface runs near a tile face.
- **Seed is the film's cure, alone sufficient for dynamics with pressure rows off** — a seeded cell has a liquid
  centre, so it owns an ordinary phi row. `volumePressureRows` can be deleted once seed lands.
- Principle 2 confirmed: without compaction the same shift field reads +0.97 cell while phi is 8% OVER.

Not validated yet: the no-pumping gate (no two-body scene run), the drain half of WP3, the sub-half-cell publish
question, run-to-run variance, and the production form of the gather (the prototype is a brute-force 8³ gather
per band vertex).

## Landed behind toggles (2026-09-19, uncommitted) — all default OFF

| Toggle (method param) | Panel stage | What it gates |
|---|---|---|
| `volumeCompaction` | Volume sharpening → Compaction | WP1. `uvPrepareSharpen` admits every phi-liquid cell and offers all of its V; `uvProposeSharpen` caps that at surplus-over-fill except toward a deeper liquid neighbour. `uvClassifySharpenTiles` also admits a tile holding an under-full liquid cell or the liquid neighbour of one, so the work map stays on and deep tiles drop out again once full. |
| `phiSeedFromVolume` | Vertex level set → Seed from V | WP3 seed half, `uvSeedPhi` in `uvAdvectPhi`. |
| `phiAgreement` (+ `phiAgreementGain` 0.05, `phiAgreementClamp` 0.02) | Vertex level set → Follow V | WP2. New pass `uvAgreementResidual` packs `r + 4a` into the gamma scratch half from start-of-step V/gamma/phi; `uvAdvectPhi` binds `densityGatherGroup` (gamma halves swapped) and gathers 8^3 single loads. Pass and rebind are only encoded while the gain is non-zero. |

Plumbing: a 13th `Params` word `agreement` (buffer 192 → 208 bytes); x compaction, y seed, z gain, w clamp, all zero unless Geometric. With every toggle off the 4h work-map bit-identity test (`tests/uniform-volume-tile-work-dawn.test.ts`) and `tests/uniform-volume-initial.test.ts` are green.

Re-measured THROUGH the toggles (`probe-surface-noise.mts` arms `t-compact`, `t-compact-dense`, `t-seed`, `t-seed-only`, `t-shift`, `t-all`; `--set=key:value` overrides any method value; ledgers `toggles-dam-break.json`, `toggles-corner-brick-drop.json`):

| Dam break, t = 3.3–6 s | deep-interior mean V | phi fill − V at 6 s | roughness | jitter |
|---|---|---|---|---|
| off | 0.34 | +8% | 0.0227 | 0.041 |
| compaction | 1.00 | −6% | 0.0131 | 0.032 |
| compaction, dense schedule | 1.00 | −6% | 0.0130 | 0.033 |
| all three (gain 0.05, clamp 0.01) | 1.00 | −1.2% | 0.0195 | 0.046 |
| all three, clamp 0.02 / 0.05 | 1.00 | −1.3% / −1.3% | 0.0195 / 0.0188 | 0.045 / 0.046 |

These reproduce the shader-rewrite numbers. The resting pool stays bit-still with all three on (max speed 0 over 90 frames).

**New, from running the film to 10 s instead of 3 s (corner-brick-drop, V = 512):**

| Arm | phi volume at 3 s / 6 s / 10 s | Reading |
|---|---|---|
| off | 165 frozen from 1 s | dead |
| seed only | 328 / ~270 / — | alive, phi erodes |
| compaction + seed | 421 / 240 / 216 | alive, phi erodes: the seed only fires where NO liquid centre is near, so it cannot hold a film that merely thins |
| all three, clamp 0.002 | 374 / 302 | shift too slow to beat the erosion |
| all three, clamp 0.01 | 567 / 779 / 668 | OVER-grows to +52% and takes seconds to come back (two runs agree) |
| all three, clamp 0.02 | 428 / 436 / 441 | holds at −14% |
| all three, clamp 0.05 | 458 / 478 / 445 | holds at −6…−13% |
| all three, no compaction | 361 / 382 | compaction matters to the film too |

So the 3× roughening recorded above for "gain 0.25, clamp 0.1" was the GAIN: at gain 0.05 the dam break cannot tell clamp 0.01 from 0.05. The default clamp is therefore 0.02, not the 0.01 first validated. The clamp-0.01 overshoot is NOT explained — the patch residual was strongly negative (tent mean −0.2…−1.0 cell) the whole time phi grew, so something adds phi at ~+0.014 cell/step that a 0.01 clamp cannot beat and a 0.02 clamp can. The missing DRAIN half of WP3 is the first suspect (the seed adds half a cell at once wherever V has visited; only the clamped shift ever removes it). Treat the film as alive-but-unfinished and the drain as the next piece of work.

## Dissipation (Peter, 2026-09-19: "this is dissipative … when we reinitialize, we're not setting the correct velocity")

Probe `probe-energy.mts` (per frame: V centre of mass, KE over phi-liquid cells, cells that lose their row with V ≥ ½ aboard, cells seeded; `--patch=keep|keepab` shader rewrites). Ledgers `energy-*.json`.

**The structure behind the hypothesis is real and is in HEAD, not in the toggles.** Momentum lives only on faces that touch a phi-liquid cell: `project` sets every face between two rowless cells to zero, and at the head of the next step the extension overwrites every face that does not touch an authority (`0.5 − pressurePhi/h > ½`) cell with the nearest phi-liquid's velocity. V has no velocity of its own. So mass that leaves phi loses what it was carrying, and a seeded cell starts from the extension's value, never from the velocity its mass arrived with. Gravity too is gated on published phi within 2h, so V farther than that from phi does not even fall.

| Dam break, 6–8 s | KE | rows lost with V ≥ ½ | momentum aboard them |
|---|---|---|---|
| off | 113 | 676 /s | 73 %/s of the liquid's |
| compaction | 63 | 410 /s | 35 %/s |
| seed only | 150 | 920 /s | 104 %/s |
| all three | 152 | 388 /s | 29 %/s |

("Momentum aboard" is an upper bound on the loss: faces shared with a cell that is still liquid stay projected.) Slosh amplitude (V centre-of-mass peak-to-peak) is the same in every arm to 8 s. So on the pool, all three together are LESS dissipative than HEAD; **compaction alone is the arm that reads lower** (KE −45%), most likely because it removes the over-full surface cells whose Sec. 3.7 excess divergence was feeding the flow — not verified.

**On the film the flicker is the dissipation.** Seed only: ~1100 rows lost and ~500 cells seeded per second from 4 s on — the seed writes `h(½ − mean V)`, which for a film under half a cell is still a non-liquid centre, so the film blinks in and out of phi and swaps its own velocity for the extension's each time. All three: 45–76 rows lost /s, 5–12 seeded /s.

**Causal test (`--patch=keepab`):** a rowless cell with V ≥ ½ and no phi-liquid face neighbour becomes an extension source and `project` stops zeroing its faces (they stay advected + gravity). Film KE per liquid cell 0.04–0.06 → 0.21–0.23 (×4–5): keeping the velocity is what it takes. But phi coverage collapses (523 → 117 liquid cells, 1200 seeds/s): ballistic faces are not divergence-free, V piles, the film tears. Unrestricted (`keep`, any V ≥ ½ cell) destroys the dam break's phi (1736 → 625 liquid cells). `volumePressureRows: all` with compaction on still bubbles (roughness 0.059, jitter 0.14) — compact V does not rescue it.

So: the velocity matters (×4–5 on the film), it cannot be had by simply not zeroing, and the working lever so far is to stop the row loss (all three on cut it 2× on the pool and 15–20× on the film).

## What WP0 found: there are two different disagreements, and neither is "phi lost volume at the surface"

I expected one disease — phi leaks volume, V's surplus piles up beside the surface, so shift phi outward by the
local surplus. Measured by depth from phi's surface (`abandoned` rows, defaults):

**water-box-dam-break, t = 6 s** (ΣV = 1680, exact)

| shell (centre phi) | cells | ΣV | Σ phi fill | mean V |
|---|---:|---:|---:|---:|
| air, > 1.5h | 3790 | 36 | 0 | |
| air band, 0…1.5h | 522 | **306** | 38 | |
| liquid band, −1.5h…0 | 628 | 614 | 568 | 0.98 |
| liquid, 1.5–4h deep | 828 | 600 | 828 | **0.72** |
| liquid, > 4h deep | 376 | **124** | 376 | **0.33** |

Not one cell deeper than 4h is full; 88 of 376 hold less than a quarter. **phi encloses 7.7% MORE volume than V
(1809 vs 1680).** The surface surplus is not volume phi lost — it is the volume displaced from interior voids. The
voids are entrained air: the level set deletes an under-resolved bubble (phi says liquid), V correctly keeps it
empty, and then nothing lets it rise. Sharpening only admits cells with `|phi| < 2.1h`
(`sharpeningDistance`), Sec. 3.7 only expels *excess*, and the velocity field sees no bubble at all, so the void is
advected passively and smeared (already 0.57 mean at 2 s). Compact V under phi and the true surface is ~0.25 cell
**below** phi's — the opposite sign to what the band residual (+316) says. A local shift driven by the band
residual, which is what I first drafted, would have inflated every pool: tile `R/A` reads +0.8 cells, with
0.45-cell steps between neighbouring tiles.

**corner-brick-drop, t = 2 s** (ΣV = 512)

| shell | cells | ΣV | Σ phi fill |
|---|---:|---:|---:|
| air, > 1.5h | 8169 | **415** | 0 |
| air band | 1043 | 96 | **178** |
| liquid band | 4 | 1.5 | 2 |

81% of the liquid is more than 1.5 cells from any phi surface, and what is left of phi sits where V mostly is
not. Here phi and V have parted company *in position*: phi keeps a ghost where the mass has left, and has
nothing where the mass went. A shift cannot fix either half.

## What has been tried here before (so this plan does not repeat it)

| attempt | what | why it died |
|---|---|---|
| `correctWholeFrameVolumePhi` (adaptive, f756fd47 → removed 54806ba6) | one GLOBAL damped phi offset from Σ(V − target)/area | "distributed a coarse disk/pool SDF mismatch onto a disconnected resting pool, inflating it every frame" — 60 cm by frame 8. Guarded by `tests/geometric-volume-phi-feedback-dawn.test.ts` |
| per-native-cell volume shift (replay arms B/C, `HANDOFF_FLUID_SURFACE_REVIEW_2026-09-08`, deleted) | each interface cell's phi matched to its own V | "transport error is a per-cell fact and a per-cell constraint copies it into the geometry … interface cells alone cannot absorb bulk mismatch" |
| sub-cell fill test (same doc) | fill phi where V says liquid | "fired on the whole pool surface every step" |
| signed `V − C·H(phi)` pressure source (sparse diagnosis doc) | deficit as negative divergence | declined: "would force phi to hide the transport error" |
| `volumePressureRows: all` (this week) | per-cell V as the pressure surface | every surface bubbled: roughness 0.021 → 0.059 |
| arm D (replay only, never built) | sign-test fill/drain, redistance, one constant shift **per connected body** by bisection | held mass to 0.00 L; never ported because component labelling is "a new serial dependent chain" |
| 2D plan S2 (never built) | σ = residual/area, smoothed tangentially, + per-region multiplier | same objection |

"Interface cells alone cannot absorb bulk mismatch" is exactly what WP0 re-measured. The common thread of the
failures: **V was read per cell, or phi was moved for an error that moving phi could not close.**

## Principles

1. **V says how much; phi says where.** phi may be moved only along its own normal, by a patch-integrated amount.
2. **Make V compact before believing its residual.** A residual measured while V has interior voids has the
   wrong sign.
3. **Read V as geometry only where phi offers none to disagree with**, and then write it INTO phi (seed), so
   pressure, extension and render all see one surface. `volumePressureRows: abandoned` applied the first half to
   the pressure interface alone; it measured better than `all` but did not hold up in the app and is off.
4. **Never ask phi to close an error it cannot close.** Every correction is driven only by residual its own
   movement reduces, inside its own patch. That, not component labelling, is the defence against pumping.

## Work packages

### WP1 — inward compaction of V (fixes the pool's real defect; touches neither phi nor pressure)

Extend the sharpening flux from the `|phi| < 2.1h` band to the whole phi-liquid region: a liquid cell with
`V < open` may draw from any face neighbour with larger phi, whether that neighbour has surplus or is itself only
part-full — a pour rule, deeper cell first. It reuses the propose/limit/commit kernels (conservative face fluxes,
blocked where the face midpoint is air, so it cannot cross between bodies); only `uvPrepareSharpen`'s admission
and the need/surplus definitions change. Eight sweeps a step move a void eight cells a step. Work map: a 4h tile
is live if it has a band cell **or** a liquid cell below capacity, so cost is ∝ deficit and returns to today's
once the pool is full. *(Coordinate with the E3 tile session: same classify pass.)*

Live toggle `volumeCompaction` (off / on) on the sharpen stage. Gates:
- shells table: every liquid shell ≥ 0.98 mean V by t = 3 s; deep histogram all "full"; ΣV exact.
- no bubbling: dam-break roughness ≤ 0.026, band jitter ≤ 0.10 (today: 0.026 / 0.103; off-rows floor 0.021 / 0.044).
- planar pool stays bit-still; `uniform-volume-dawn` suite no new reds.

Risk: deep phi is not redistanced beyond 4h, so "larger phi" is only approximately "toward the surface". A local
phi maximum deep inside can stay part-full. Expected to be isolated cells; the histogram will say.

### WP2 — slow normal shift from an overlapping-patch residual

Per band vertex (`|phi| < 2h`): `R = Σ w·(V − uvTarget)`, `A = Σ w·[cut cell]` over band cells
(`|phi_c| < 1.5h`, `open ≥ 0.99999`) within ±4 cells, tent weights; `phi −= h·clamp(gain·R/A, ±δ)`, zero when
`A < 1` or `|R/A| < 0.02`. **gain 0.05, δ = 0.01 cell/step** (validated; 0.25/0.1 roughens the surface 3×).
Evaluated from start-of-step V and `gammaA`, which are mutually consistent with start-of-step phi, inside the phi
advect pass; redistance follows in the same step. Production form: two r32 scratch fields (w·r, w·a) blurred
separably — three box passes twice over the band tiles only — then one vertex pass; **not** 4h tile sums.
Toggle `phiVolumeAgreement` (off / shift).

Gates: phi volume within 2% of ΣV on the dam break at 6 s; roughness and jitter within the baseline's noise
(≤ 0.026 / ≤ 0.05); resting pool bit-still (validated); `uniform-volume-dawn` mini32 frame-90 drift green;
**no pumping** — port the scenario of `tests/geometric-volume-phi-feedback-dawn.test.ts` (drop + disconnected
resting pool) to uniform: pool height within 2 cm. A vertex is driven only by residual within 4 cells of it, so a
resting pool more than 4 cells from anything cannot be moved; the test covers the near case.

### WP3 — seed and drain where phi and V have parted (fixes the film Peter reported)

Both directions, both under principle 3, both on a 3³-tent-smoothed V̄ so no single cell decides:
- **seed** (validated): a vertex whose 4³ cell neighbourhood has no phi-liquid centre, and whose 8 adjacent cells
  average more than a quarter full, takes `phi = min(phi, h(0.5 − V̄))`. On the dam break it barely fires and the
  surface stays at baseline roughness.
- **drain**: a phi-liquid region whose patch holds V̄ ≈ 0 for N consecutive steps is a ghost;
  `phi = max(phi, +0.5h)` there. (After WP1 a real interior is full, so V̄ ≈ 0 under phi-liquid is unambiguous.)

Once seeded, a tile has a band and WP2's shift maintains it. Toggle value `shift+seed`. Gates: corner-brick-drop
at 2 s — V farther than 1.5h from any phi surface 415 → < 50, phi volume/ΣV ≥ 0.85 (0.36 today), liquid centres
≥ 300 (4 today); an additive brick seeded over `tank-fill` (V but no phi — it hangs in mid-air today) gets a
surface and falls; dam-break gates unchanged (seed/drain must almost never fire beside a healthy surface —
count the firings).

Open question to measure first, not assume: a *perfect* flat phi film of depth d < h/2 still publishes
occupancy d/h < 0.5 at the floor cell centre (`uvPublish`). Whether the mesher draws it depends on what it takes
below the floor. Check with a synthetic 0.3h film before WP3's gates are trusted.

### WP4 — defaults and clean-up

Default the three toggles on; delete `volumePressureRows: all`; re-measure how often `abandoned` still grants
(expected: rare once phi tracks V); write the port note for adaptive-volume, which has the same gap and already
reduces `phiVolumeResidualFine3` / `phiInterfaceAreaFine2` with no consumer.

## Order and why

**WP1 → WP3 (seed) → WP2 → drain → WP4.** WP1 is small, touches only V, improved every metric it was measured on
and changes what every later measurement means. Seed is what makes corner-brick-drop visible and alive with the
pressure rows off, so it outranks the shift. The shift is last of the three because it is the only one with a
noise/accuracy trade and the only one still missing a gate (no pumping).

Every gate is a Dawn/CPU probe already in the research directory; one run per arm per scene; A/B against the
toggle's off value, which must stay bit-identical.
