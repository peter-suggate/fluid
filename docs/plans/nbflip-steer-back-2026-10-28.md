# NB-FLIP steer-back: review and plan (2026-10-28)

## Verdict

The fear is justified. The uncommitted tree passes the seep ±0.1% gate, but it got there by adding new
machinery tuned to the seep, and that machinery made the canonical scene worse:

| Measure | 04f725d1 (07:27 run) | Working tree (13:12-13:38 runs) |
|---|---|---|
| Letters phi drift, 240 steps | −14.1% | −31.8% when first measured, now −26.7% |
| Seep worst drift, 1200 steps | −0.3%, gate fails | +0.07% / −0.08%, gate passes |
| Seep median step | 35.2-35.8 ms (exclusive timing) | 49-54 ms (check mode, unconfirmed) |
| Letters median step | 65 ms (check mode) | 89-122 ms (check mode, unconfirmed) |

The "32% → 28% improvement" in the agent's summary is measured from its own regression. Both runs
were check mode and may have overlapped, so the step-time numbers need exclusive repeat-gated timing.
The letters regression does not depend on timing.

Why it drifted: the gate asks for more than the method claims. Ferstl 2016 §3.3: "the volume
conservation of our method is equivalent to regular FLIP, i.e., we do not conserve mass exactly …
significant changes in mass can only originate right at the liquid's surface." Holding ±0.1% over 1200
steps on one scene rewarded bookkeeping that conserves the seep, not a faithful NB-FLIP. Our own commit
started this with mass-carrying particles (graded mass-2 seeds, `crowdedCells: "merge"`).

## Mechanism review

Verdicts:

- **keep**: a measured bug fix, or a paper method;
- **earn**: it stays only if the ablation (Phase 1) shows it is needed and regresses nothing;
- **revert**: not in Ferstl or Sato, and a paper-standard replacement exists.

### Uncommitted working tree

| # | Mechanism | Papers | Verdict |
|---|---|---|---|
| U1 | One-sided thin-film depth against walls (`umDepthSample`). It fixes the measured 2× film-thickness bug | Bug fix | keep |
| U2 | P2G expected face mass of 8 under merge, matching graded seeds | Consistency | keep (then re-checked in Phase 1) |
| U3 | Telemetry: measuring volume before each reported step, letters dt and volume reporting, allocation caps | Tooling | keep |
| U4 | Reconstruction-ownership fixture repair | Test | keep |
| U5 | Inviscid, zero-tension water in the presets | Planned (−3 to −4 ms) | keep |
| U6 | RT0/Pollock flux trace replacing RK3 for every coupled sample and every seed trace | No. Ferstl uses RK4, Sato RK2. A per-axis linear field cannot rotate within a cell. Letters with RK3 restored: −23.3%, against −26.7% | revert |
| U7 | Solenoidal air extension: 32 Jacobi sweeps over all fine cells, every step (34 extra dispatches), unconverged | No. Sato step 8 is standard extrapolation (Bridson) | revert |
| U8 | Material-arrival and departure reseeding: forward traces of Eulerian sites into the band, plus a backward trace per seed candidate | No. Ferstl reseeds only where a cell holds fewer than n, in −R ≤ φ ≤ −h, at grid velocity | revert |
| U9 | Physical mass cached apart from record count, uncapped mass, a mass bound on erasure | No. They exist to serve the mass-carrying scheme | earn |
| U10 | Thin tiles exempt from deep retirement; heat held in every thin tile | Special cases | revert |
| U11 | 23 Picard re-solves at t=0 | No. This hides seam heads that depend on pressure. The same lag hits every dynamic frame | keep for now; replace in Phase 5 |
| U12 | BULK16 pages, allocator and reference (not wired in) | Plan bet A | keep, isolated |

### Committed in 04f725d1

| Mechanism | Verdict |
|---|---|
| Cold-tile union fix | keep (real bug) |
| Contact radius r(m) against all solids | keep (standard particle radius) |
| Separation pre-pass | keep (−9 ms) |
| Jacobi-averaged pair separation (Sato §3.2.2, bounded) | keep |
| `activationSeeding: "uncovered"`, `innerReseed: "cells"` | keep; re-check against Ferstl's rule of never seeding within h of the surface |
| `crowdedCells: "merge"`, graded density, mass-2 seeds (mass-carrying particles) | earn: ablate against Ferstl's 2n cap with uniform density |

## Goals

1. **Seep**: plausible physics. The right tank fills slowly, with no unphysical gain.
2. **Splash scenes** (letters first): volume loss no worse than FLIP's on the same solver.
3. **Realtime**: step time ranks first. Fixed dt of 1/60 s, one step per frame, no substeps.
4. **Vast scenes**: BULK16 as the full design, with band-sized particle storage.
5. **Simplicity**: paper methods first. A new mechanism needs a measured disagreement with the paper.

## Gates

- Closed rest: drift 0 at 1×-2.75× (passes now).
- No gain beyond +0.1% in any scene.
- **Letters, 240 steps: no worse than −14.1% (04f725d1).** The target is the FLIP-oracle loss from Phase 2.
- **Step time:** exclusive, 3+3 interleaved repeat gate on seep and letters. No regression beyond
  spread against 04f725d1 plus U5.
- Seep loss bound: **decision for Peter.** ±0.1% over 1200 steps is stricter than either paper claims.
  - Recommendation: keep ±0.1% for gain.
  - For loss, require "no worse than the FLIP oracle, with the right tank ≥ 95% of delivered by step 1200".
  - Until Peter decides, ±0.1% stays, but it may not be met by reverted-class mechanisms.

## Plan

### Phase 0: preserve (no GPU)

- Save the solver part of the working tree as a patch, with copies of the untracked files, under
  `artifacts/nbflip-handoff/working-tree-2026-10-28/`.
- Do not stash, reset or check out.
- Leave the other review session's files alone (pane-activity, progress-history, gate-repeat-spread,
  water-window-pages).

### Phase 1: ablation matrix (one agent, sequential GPU jobs, timing runs exclusive)

- Baseline: 04f725d1 plus U1-U5.
- Add each "earn" item on its own, then each "revert" item on its own, so the evidence is on record.
- One table row per configuration:
  - seep: worst drift and right tank at 1200 steps;
  - letters: drift at 240 steps;
  - median step time for both scenes.
- Landing rule:
  - The item improves the scene it targets.
  - It does not regress the other scene beyond spread.
  - It costs no more than spread in step time.
- Expected result: U6, U7, U8 and U10 go; U9 and merge are decided on the numbers.

### Phase 2: FLIP oracle

- Run letters at half resolution (128×96×64) twice:
  - once with particles everywhere: band depth set to the domain, for this configuration only;
  - once with NB-FLIP at the same resolution.
- This gives the FLIP-equivalent loss target. It also splits letters loss into what FLIP itself loses
  and what is specific to NB-FLIP.
- The `bandDepth` 2-4 guard stays for scenes; the oracle uses its own path.

### Phase 3: root cause of the letters loss

- Apply the seep ledger to letters, per step:
  - particle mass against phi volume;
  - union and erosion;
  - reseed and retire;
  - remap;
  - clearAir;
  - h and 4h exchange.
- Find where loss beyond the oracle's starts. Fix it at the mechanism, paper first.

### Phase 4: seep

Close the remaining seep gap using only Phase 1 survivors and paper-standard fixes. The film-resolvability
fix (U1) and the h exchange balance from the handoff's open issue 1 come first.

### Phase 5: step time and vast scenes

In the order of `nbflip-handoff-2026-10-28.md` §3–4:

1. BULK16 as the full design, with band-sized storage;
2. a seam operator independent of pressure, which retires U11;
3. symmetry, whenever `uniform-mixed-extension.ts` is next touched.

## Rules for the implementing agent

- **Mechanisms:**
  - No new mechanism without one of two grounds: a citation (Ferstl 2016, Sato 2018, or a standard
    reference such as Bridson for extrapolation), or a measured case where the code disagrees with the paper.
  - Each new option must be ablatable.
  - A change that helps one scene and hurts another is reverted, not compensated.
- **Every report carries the fixed table:**
  - seep worst drift and tank volume;
  - letters drift at 240 steps;
  - median step time for both scenes;
  - the baseline row.
- **GPU jobs:**
  - only through `tools/gpu/fluid-gpu-run.sh`, under 6 GiB;
  - one simulation at a time when timing;
  - check runs never reported as timing.
- **Fixed constraints:** `volumeControlSeconds: 0`; no clamps, damping, injection or speed limits; no
  substeps; no commits without asking.

## Phase 1 result (2026-10-28)

All artifacts are under `artifacts/nbflip-steer-back/`. Rows marked "timing" are exclusive repeat-gated runs. Every other row is a check-mode run, so its milliseconds are not timing.

Peter relaxed the gates mid-phase:

- seep |drift| ≤ 3% with a plausible fill;
- letters target |drift| ≤ 3% at step 240;
- step time ranks first;
- seep receiving ratio (right-tank phi volume over the volume that left the left tank) within 0.9–1.1.

The original ±0.1% seep gate fails on every row.

### Fixed table

Timing is the median of the three runs' medians, with `--pipelined` on the seep and `--plain` on letters.

| Row | Seep worst drift (step) | Receiving ratio | Letters drift at 240 | Seep median (timing) | Letters median (timing) |
|---|---|---|---|---|---|
| (a) HEAD + U5 | −0.241% (1200) | 0.84 | −14.44% | 36.95 ms | 64.93 ms |
| (b) cleaned (HEAD + U1–U5, U9, U11–U14) | +0.267% (1176) | 1.17 | −18.58% | 36.29 ms | 68.91 ms |
| (g) b with HEAD's no-erasure guards | +0.855% (1071) | 1.52 | −12.88% | 37.89 ms | 68.82 ms |
| **(n) the main checkout now** (scratch row) | +1.128% (929) | 1.60 | −14.39% | 36.89 ms | 64.47 ms |
| (n) applied in the main checkout | +1.183% (948) | 1.56 | −14.68% | 36.85 ms (t2 runs) | 65.95 ms (t2 runs) |
| n + `particleSeeding: activity` (Sato's seeding rule, below) | +1.696% (1200) | **2.11** | **−3.40%** | 36.03 ms | **83.17 ms** |

Timing artifacts are in `timing/`:

- the a/b/g/n set: `seep-t-*`, `nbflip-letters-letters-t-*`;
- the n vs activity set: `seep-t2-*`, `nbflip-letters-letters-t2-*`.

Repeat-gate verdicts. The tolerance is 1.66 × the larger run range.

- Seep: every variant is within spread of a. g vs b is +1.60 ms against a tolerance of 0.69: slower. n+activity vs n is −0.82 ms against 2.05: within spread.
- Letters median:
  - b vs a: +3.97 ms against 3.64, slower;
  - g vs a: +3.89 ms, slower;
  - n vs a: within spread;
  - n vs b: −4.44 ms, faster;
  - n+activity vs n: +17.22 ms against 2.81, slower.
- Letters p95:
  - a 73.75 ms, b 80.40 ms, g 84.40 ms, n 74.38 ms;
  - g vs a: +10.65 ms against 8.94, slower;
  - n+activity vs n: +17.75 ms against 4.33, slower.

Across its three timing runs, n's letters drift at 240 was −15.36, −14.38 and −15.13%, and n+activity's was −3.40, −3.48 and −3.33%. Both solvers are nondeterministic by design.

### What the main checkout is now (row n)

The main checkout holds HEAD plus the following:

- U1, the one-sided wall depth;
- U2, expected face mass 8 under merge;
- U3, telemetry and probes;
- U4, the fixture repair;
- U5, the inviscid presets;
- U11, the startup re-solves;
- U12, the BULK16 hooks, inert unless enabled;
- U13, far-open extension and closed-owner phase −1;
- the symmetry capture.

Removed, hunk by hunk, with nothing restored wholesale:

| Removed | Why | Saved in |
|---|---|---|
| U6 RT0/Pollock flux trace | Not paper: Ferstl uses RK4, Sato RK2. Re-added alone (e6): letters −16.9% against b's −18.6%, but receiving ratio 1.27 | `removed.patch` |
| U7 solenoidal air extension | Not paper: Sato step 8 is standard extrapolation. Only feeds U6 | `removed.patch` |
| U8 arrival/departure reseeding | Not paper. Needs U6, U9 and U10 | `removed.patch` |
| U10 thin-tile exemptions | Not paper. Re-added alone (e10): no letters gain (−18.7%) | `removed.patch` |
| Sample-probe diagnostic | Reads U6/U7 state | `removed.patch` |
| U9 mass plane, uncapped mass, erasure bound, merge depth | Peter's ruling (option 2). It costs about 4 ms on letters (b vs a, slower) through about 30% more particles, and buys only 1.5 points (g −12.9% against n −14.4%). Its erasure bound is the cleaned tree's extra 4-point loss (ledger below) | `removed-u9.patch` |
| U14 3³ separation stencil | Only needed for U9's heavier merged samples; n is built without it | `removed-u9.patch` |

Files in `removed.patch` (702 lines):

- `uniform-narrow-band-activity.wgsl.ts`: the U8 NEW bit;
- `uniform-narrow-band-air-velocity.wgsl.ts`: deleted;
- `uniform-narrow-band-flip.ts`: 19 hunks;
- `uniform-narrow-band-flux-trace.wgsl.ts`: deleted;
- `tools/probe-tier-budget.ts`: the sampledDivergence/sample-probe hookup;
- `tests/uniform-narrow-band-air-velocity.test.ts` and `tests/uniform-narrow-band-flux-trace.test.ts`: deleted.

Files in `removed-u9.patch` (463 lines):

- `uniform-narrow-band-flip.ts`: 12 U9/U14 hunks; the two U2 call sites stay;
- `uniform-narrow-band-surface.wgsl.ts`:
  - `RECONSTRUCTION_DEPTH`/`MERGE_DEPTH` are removed;
  - `if(particle<bulk)`, `nbBinCount` and `min(bulk,…)` are restored;
  - U2's `narrowBandExpectedMassWGSL(…, merge)` and `transferProperties` merge parameter stay;
- `uniform-narrow-band-splat.wgsl.ts`: `nbSurfaceFootprintHeld`;
- `uniform-particle-slab.wgsl.ts`: the merge mass plane;
- `uniform-narrow-band-mass.wgsl.ts`: deleted (untracked, U9 only; its full text is appended to the patch);
- `tools/reservoir-particle-cost.ts`;
- tests:
  - `reservoir-particle-cost`;
  - `uniform-narrow-band-seeding-params`;
  - `uniform-particle-slab`;
  - `uniform-narrow-band-transfer-mass`: the "reseeding fills physical mass" test is removed; U2's expected-mass test stays.

The only difference between the scratch n tree and the main checkout in `flip.ts`/`surface.wgsl.ts` is the two U2 lines.

Merge mode (`crowdedCells: "merge"`, mass-2 graded seeds) is HEAD's and stays. Row (d), Ferstl's 2n retire cap with uniform density, gives no letters gain (−18.8%). It also loses the seep film: receiving ratio 0.42, worst drift −0.88%.

Candidate patches that were measured but not applied:

- `candidate-g-no-erasure-bound.patch`;
- `candidate-nu9-head-u1-u11-u13.patch`: row n as a patch on HEAD.

### Gates on the main checkout (n applied)

| Gate | Result |
|---|---|
| `npm run check:types` | clean |
| `npm run test:unit` | 1394 tests, 1259 pass, **0 fail**, 135 skipped |
| Focused GPU tests (slab, transfer-mass, reconstruction-ownership) | 3/3 (`gpu-tests-n.log`) |
| Closed rest, steps 30/60/90/120 | drift **0** at every step, maxSpeed 7.5e-6 m/s (`rest-n-applied.json`) |
| Seep reproduces n | +1.183% (948) against +1.128% (929); ratio 1.56 against 1.60 (`seep-n-applied.json`) |
| Letters reproduces n | −1.18 / −5.21 / −10.27 / −14.68% against −1.19 / −5.17 / −10.24 / −14.39% (`nbflip-letters-n-applied.json`) |

### Ablation matrix (check runs)

| Row | Seep worst drift (step) | Right tank at 1200 | Ratio | Letters 60 / 120 / 180 / 240 | Letters particles (mean) |
|---|---|---|---|---|---|
| (a) HEAD + U5 | −0.241% (1200) | 4.73 L | 0.84 | −1.19 / −5.25 / −10.20 / −14.44% | 341k |
| (b) cleaned | +0.267% (1176) | 6.59 L | 1.17 | −1.18 / −5.89 / −12.36 / −18.58% | 461k |
| (c) b − U13 − U14 | +0.274% (1200) | 6.64 L | 1.18 | −1.19 / −5.66 / −12.80 / −18.12% | 453k |
| (d) b + retire, uniform, cap 16 | −0.878% (1198) | 2.36 L | 0.42 | −1.18 / −6.36 / −12.46 / −18.76% | 422k |
| (e6) b + U6 | +0.416% (1198) | 7.12 L | 1.27 | −1.24 / −5.65 / −11.54 / −16.94% | 469k |
| (e10) b + U10 | +0.414% (1180) | 7.06 L | 1.26 | −1.18 / −5.84 / −12.81 / −18.74% | 467k |
| (e7), (e8) | not separable: U7 only feeds U6; U8 needs U6, U9 and U10 | | | | |
| (h1) HEAD + U5 + U1 | | | | −1.14 / −5.10 / −10.03 / −14.02% | 342k |
| (h11) HEAD + U5 + U11 | | | | −1.19 / −5.18 / −9.65 / −14.31% | 333k |
| (nu2) b − U2 | | | | −1.19 / −6.12 / −13.54 / −18.76% | 446k |
| (g) b + no-erasure guards | +0.855% (1071) | 8.50 L | 1.52 | −1.18 / −4.86 / −8.87 / −12.88% | 448k |
| (n) HEAD + U5 + U1 + U11 + U13 | +1.128% (929) | 9.00 L | 1.60 | −1.19 / −5.17 / −10.24 / −14.39% | 347k |
| n applied | +1.183% (948) | 8.75 L | 1.56 | −1.18 / −5.21 / −10.27 / −14.68% | 331k |

Delivered volume is 5.6 L in every row. The artifacts are `seep-<row>.json` and `nbflip-letters-<row>.json`, named as in the rows.

### Letters ledger (where the volume goes)

The ledger is `probe-tier-budget --scene=nbflip-figure-8-letters --ledger=1-240`, run in scratch copies with the absent dividing wall at x=0. The scheduled drops are subtracted from the advect stage. Units are m³; the target is about 12.8 m³.

| Steps 1–240 | HEAD (a) | cleaned (b) | b − a | no-erasure (g) |
|---|---|---|---|---|
| Loss against target | −1.794 (−14.0%) | −2.283 (−17.8%) | **−0.489** | −1.717 (−13.4%) |
| Advect, without drops | −3.667 | −3.758 | −0.091 | −3.698 |
| … of which h bare cells | −3.653 | −4.163 | | −3.930 |
| Reconstruct (union) | **+1.838** | **+1.446** | **−0.392** | +1.944 |
| … of which h bare cells | −0.044 | **−0.432** | −0.388 | −0.019 |
| Every other stage | within ±0.03 | within ±0.03 | | within ±0.03 |

Logs: `lledger-a-head.log`, `lledger-b-cleaned.log`, `lledger-g-no-erasure-bound.log`.

Findings:

1. The cleaned tree's extra 4 points come entirely from U9's erasure bound.
2. HEAD's 14% is grid advection of φ in h cells that hold no samples, which the union cannot restore because no particles are there.

### Heat test (letters) and the comparison with Sato §3.3

| Run (on n unless marked) | Letters 60 / 120 / 180 / 240 | Particles mean / max | Artifact |
|---|---|---|---|
| n | −1.18 / −5.21 / −10.27 / −14.68% | 331k / 633k | nbflip-letters-n-applied |
| `adaptiveFadeSeconds: 2` (cooling 1/s) | −1.15 / −4.70 / −7.15 / −8.45% | 503k / 1.03M | nbflip-letters-n-heat-fade2 |
| `particleSeeding: activity` | −0.99 / −2.58 / −3.04 / **−3.40%** | 824k / 1.23M | nbflip-letters-n-heat-activity |
| both | −0.98 / −2.70 / −3.16 / −3.40% | 825k / 1.25M | nbflip-letters-n-heat-fade2-activity |
| thin trigger ungated, shape still gated (scratch) | −1.19 / −5.33 / −9.99 / −14.67% | 335k / 657k | nbflip-letters-n-thin-ungated |
| g, no cooling (`w[0]=0`) + activity (ledger) | −0.91 / −2.56 / −3.00 / −3.40% | about 1.4M | lledger-g-heat-nocool-activity |

With the whole band hot, the loss stops at about 3.4%. The remaining loss is in the union (reconstruct +3.80 m³; h-bare reconstruct +0.29). So the 14% is a seeding gap: the letters' surface is left without particles. Cooling is not the cause: fade 2 s alone recovers 6 points, and adds nothing on top of activity.

The comparison, using the paper text in `docs/papers/narrow-band-flip-2026/sato-2018-exnbflip/paper.txt`:

| Sato 2018 §3.3 | Ours | Disagreement | Measured |
|---|---|---|---|
| γ = max(Au·min(1,‖q‖) − Tu, Ag·\|c\|/Δx − Tg), with q a velocity high-pass (Eq. 3) and c = φ − G∗φ, a pseudo-curvature (Eq. 4). The geometric term is evaluated regardless of motion: it "acts to retain sharp corners during the liquid's free-fall" (Fig. 1). Only q is blind to translation. | Binary census triggers per 4h tile. In the default `particleSeeding: "need"`, the geometric criteria (shape, thin) heat only when per-tile deformation exceeds the transport budget, and the code cites Sato for "translation is no detail" | **Yes, decisive.** The gate applies Sato's remark about q to the geometric term | Ungating every criterion (activity): −14.7 → −3.4%. Ungating thin alone: no change, so the shape criterion is the one that matters for the falling letters |
| Velocity term q (σ=1, Au=5, Tu=0.2) | No direct analogue. Strain and rotation exist but are off by default. Impact and approach are anticipatory, which matches Sato's "future collisions" | Partial | Not measured separately |
| Seeding wherever γ*>0 | `seedCells`: wherever target heat > 0. The same rule, fed by different triggers | No | — |
| Cooling γp −= αΔt, α=10; γ ≤ about 4.8 from the velocity term (about 0.48 s) | heat 2.0, cooled at 2/fade = 4/s (0.5 s) | Lifetime is equivalent for strong triggers; weak ones live longer in ours | fade 2 s: −8.5%, so cooling is not the main gap |
| Propagation: 4 upwind iterations of ∂γ/∂τ = ‖∇γ‖ into the interior, Δτ = 0.75Δx (about 3 cells, decaying) | `activitySpread`: max over a 1-tile (4-cell) neighbourhood at 0.5 | Comparable reach; ours also spreads laterally | Not measured |
| Ballistic particles are not cooled | Spray keeps its heat (`nbMoveHeat` is skipped when ballistic) | No | — |
| Per-cell heat | Per-4h-tile census | Coarser | Not measured |

**Proposed paper-faithful heat fix:** let the geometric criteria heat regardless of motion, as Sato's Eq. 4 term does. The smallest form is the existing `particleSeeding: "activity"` setting as the NB-FLIP default (one default value; no new mechanism).

The fix is measured but **not applied**:

- letters −14.7 → −3.4% (passes ≤ 3% within noise; the three timing runs gave −3.40, −3.48 and −3.33%);
- letters +17.2 ms median (65.95 → 83.17 ms, slower beyond spread) and 2.5× the particles;
- seep time within spread, but the receiving ratio worsens from 1.56 to 2.11 (worst drift +1.70%).

Step time ranks first and the seep ratio gets worse, so this needs Peter's ruling. A cheaper paper form would heat on Sato's actual geometric measure (c = φ − G∗φ, thresholded at Tg) instead of the shape census. That is a new trigger: write-up only.

### Seep receiving-film diagnostic (n)

Source: `film-diag-n.json` (the tool was run as `tools/film-diag.ts` in a scratch tree; a copy is saved as `film-diag.ts.txt`).

Method:

- right-tank floor columns, y < 8 cells;
- binned by column film thickness (Σ phi fraction, in h);
- phi volume against particle mass, in L, at eight mass units per cell;
- per-cell phi/mass histogram;
- excess (phi > mass) split into cells holding merged samples, unit-only cells, and cells with mass < half a cell.

| Step (drift) | Right floor phi / particle mass | Where the film is | phi/mass by bin | Mass per occupied cell (records) |
|---|---|---|---|---|
| 600 (+0.24%) | 3.53 / 4.19 L | 1.5–3 h | 0.84–0.85 (thin < 1 h: 0.9–1.1) | 6.5–8.3 (6.5–7.6) |
| 750 (+0.57%) | 5.45 / 9.83 L | 2–4 h | **0.51–0.55** | **13.4–14.8** (7.2–7.6) |
| 900 (+1.06%) | 7.94 / 11.01 L | 3–6 h | 0.71–0.73 | 11.6–13.6 (6.9–7.5) |
| 1050 (+0.93%) | 8.18 / 5.59 L | 4–6 h | 3.03 (4.28 L of phi in cells with no samples: the cooled film) | 11.5 (7.5) |
| 1176 (+0.74%) | 8.05 / 5.69 L | 4–6 h | 2.59 (4.05 L with no samples) | 10.8 (6.9) |

Answers to the two hypotheses:

- **The sparse film drawn at splat-radius thickness is not supported.** Columns thinner than 2 h hold at most 2.0 L (at step 600) and far less later, and phi/mass there is 0.69–1.3. Excess phi in those bins is ≤ 0.11 L per bin at every step.
- **A merged-sample radius ≠ r(m) is not supported.** Excess in cells holding merged samples is ≤ 0.09 L per bin. Excess in unit-only cells is 0.5 L at 900 and 4.1 L at 1050–1176. The late figure is cooled film with no samples, where phi is the grid's by design.
- **What the data shows.** The drift rises between steps 600 and 900, the hot period. In that period the film's particles carry 1.3–1.9× the nominal mass per occupied cell (11–15 units against 8). Records stay about 7 per cell, so the extra is heavy samples (mass-2 graded seeds and merges), not crowding by count. Phi sits *below* the particle mass (0.5–0.75), so the union is not drawing phi past the samples cell by cell.
- **How the gain scales.** Across rows, the gain follows particle presence in the film:
  - d (retire, uniform, fewer samples): 0.42;
  - a (HEAD, film at 4h): 0.84;
  - n: 1.56–1.60;
  - n+activity (more samples): 2.11.
- **The seep ledger.** It shows a large exchange: right floor advect −61.1 L against reconstruct +66.7 L by step 750 (the ledger is in mL; an earlier draft misread it as cells). The advect figure includes Ferstl's shrink of φ0 by h. The net gain is the small difference between grid-advection loss and union restore in the film.

**Proposed paper-faithful fix: none yet; this is the next measurement.** The per-cell histogram rules out the two suspected representation errors. The candidate left is the union's per-step restore exceeding the advection loss in a 2–4 h hot film. Ferstl φ ← min(φ0 + h, φp) lets particles overrule, so any systematic offset of φp's zero contour beyond the outermost samples (sample radius against sampling inset) is restored every step.

The next measurement is a per-step, per-thickness ledger of advect against reconstruct in the right film, together with the φp contour offset above the topmost sample per column. Untested aside: Ferstl caps every band cell at 2n by removal, and we cap or merge only cells deeper than one cell. Mean records per cell is about 7, so that cap would rarely bind here; it is not proposed.

### Decisions (by the landing rule)

- **Base = n** (Peter's ruling, option 2):
  - letters median 64.5–66.0 ms, within spread of HEAD and 4.4 ms faster than cleaned;
  - letters −14.4 to −15.4%;
  - seep ratio 1.56–1.60.
- U6, U7, U8, U10, U9 and U14 removed; patches saved.
- U1, U2, U3, U4, U5, U11, U12 and U13 kept. Merge mode (HEAD) kept: row d fails the seep.
- Not landed, for Peter to decide:
  - `particleSeeding: "activity"` (Sato's geometric heat): letters −3.4%, but +17 ms and seep ratio 2.11;
  - g (the erasure-guard fix of U9): moot with U9 removed.
- Open bugs:
  1. Receiving-film gain: ratio 1.56 on n against the 0.9–1.1 gate. Every row fails it (a fails low at 0.84).
  2. Letters −14.7% on n against ≤ 3%.

### Contradictions to the review

- U1 does not change letters (−14.0% against −14.4%). On the seep it keeps the receiving film at h, which turns HEAD's film loss (ratio 0.84) into a gain (1.56–1.60). The film gain is not a U9 effect: U9 rows sit lower (1.17), but only because of its erasure bound.
- Under U9 the seep's ±0.1% failure is a gain (+0.27%), not the −0.3% loss the review records for 04f725d1.
- U6 alone helps letters by 1.6 points against cleaned (−16.9%), but adds seep gain (ratio 1.27). The review measured the reverse for the full working tree.
- Merge against Ferstl's 2n cap (row d) gives no letters benefit and loses the seep film (ratio 0.42).
- The letters loss is a heat/seeding gap. Sato's geometric term, which the `need` gate suppresses for translating liquid, recovers 11 points. The review attributed the loss to advection order and flux tracing (U6), which recover at most 1.6 points.

### Follow-up: ungating shape (Sato Eq. 4) and the per-step film ledger

Peter ruled that `activity` does not become the default (+17 ms on letters). The paper-faithful form was tried instead.

**The change.** In `need` mode, the shape criterion heats regardless of the transport budget. The velocity and impact parts keep their gate, and thin stays gated. Patch: `candidate-shape-ungated.patch`, two lines in `nbActivityTriggered`.

**Is our shape criterion Sato's c = φ − G∗φ at Tg?** No. With the app's displacement metric at a 0.5 h tolerance, it is the largest of:

- the h-tile surface displacement when the tile is carried by its 4h corners;
- 4 × the mean signed displacement;
- on moving tiles only, 4 × (4h)²/12 × |∇²φ| over interior vertices, plus the same Laplacian at the tile corners.

All of these are high-pass measures of φ, as Sato's c is. The differences:

1. **Kernel.** Ours is the error of a 4h trilinear interpolant and a Laplacian at the 4h scale. Sato's is a Gaussian with σ = 1 cell at h.
2. **Threshold.** For a small Gaussian, c ≈ −(σ²Δx²/2)∇²φ, so Sato heats when Δx|∇²φ| > 0.4:
   - a cylinder of radius < 2.5 h;
   - a sphere of radius < 5 h.

   Ours triggers on a cylinder of radius < 10.7 h, as the code's own comment says. That is about 4× more sensitive, and it adds the flat-displacement terms.
3. **Fast tiles.** Tiles faster than the surface-travel limit skip shape entirely. Sato has no exemption.
4. **Granularity.** Ours is per 4h tile and binary; Sato's is per cell and graded.

**Results, n against shape ungated** (check runs unless marked timing):

| Run | Letters 60 / 120 / 180 / 240 | Letters particles mean | Seep worst drift | Receiving ratio |
|---|---|---|---|---|
| n | −1.18 / −5.21 / −10.27 / −14.68% | 331k | +1.183% (948) | 1.56 |
| shape ungated | −1.00 / −2.64 / −3.01 / **−3.25%** | **825k** | +1.797% (1200) | **2.18** |
| `activity` (reference) | −0.99 / −2.58 / −3.04 / −3.40% | 824k | +1.696% (1200) | 2.11 |
| shape ungated + `detailShapeTolerance: 2` (about Sato's threshold, but it also coarsens refinement) | −1.59 / −3.66 / −4.85 / −5.42% | 623k | — | — |
| n + `detailShapeTolerance: 2` (control) | −1.88 / −5.94 / −10.66 / −15.34% | 265k | — | — |

Timing (repeat gate, 3+3 interleaved, `timing/*-t3-*`):

- **Letters median:** n 67.60 ms (runs 68.96 / 67.60 / 66.85) against shape ungated 83.74 ms (87.53 / 83.74 / 83.50). That is **+16.14 ms**, tolerance 6.69: slower.
- **Letters p95:** 80.58 against 94.19 ms, +13.61 ms (tolerance 8.86): slower.
- **Seep median:** 37.20 against 36.80 ms, −0.40 ms (tolerance 8.15; round 1 was noisy for both): within spread.
- **Letters drift in the timing runs:** n −14.62 / −14.61 / −15.62%; shape ungated −3.40 / −3.44 / −3.10%.

Shape ungated *is* `activity` in effect. Its particle count is the same to within 0.1%, and so are its letters and seep results. Earlier, ungating thin alone changed nothing. So shape is the entire cost of activity, and the entire gain.

Not landed at first: the step-time cost is not clearly below activity's, and the seep receiving ratio gets worse (2.18).

**Landed later (ruling, 2026-10-28).** The ≤3% letters gate is a correctness gate, so shape ungated is now in the main checkout (`uniform-narrow-band-activity.wgsl.ts`; the unused `UNIFORM_BODY_GEOMETRY_REACH` import is removed). Gates on the applied tree:

- `npm run check:types`: clean.
- `npm run test:unit`: 1394 tests, 1259 pass, 0 fail, 135 skipped.
- Closed rest drift at step 120: 0 (`rest-shape-applied.json`).

A version with Sato's actual threshold (a heat-only threshold separate from the refinement tolerance) would heat fewer tiles. That is a new parameter, so it is reported, not built. Raising the shared tolerance to 2 h, as a proxy, kept −5.4% at 623k particles (−25% particles). It also changes refinement, so the proxy is not a clean measure of a heat-only threshold.

**Per-step film ledger** (seep, n). Run with a scratch copy of `probe-tier-budget` (`diag-film-ledger-probe.patch`) and logged to `seep-n-filmledger.log`. It reproduced n: +1.03% at 900, ratio 1.58.

By interval, right tank gained against left tank lost:

| Steps | Left lost | Right gained | Excess | Ratio | Drift at end |
|---|---|---|---|---|---|
| 0–150 | 0.70 L | 0.69 L | −0.01 L | ≈1.0 | −0.004% |
| 150–300 | 0.71 L | 0.81 L | +0.11 L | 1.15 | +0.03% |
| 300–450 | 0.70 L | 0.94 L | +0.24 L | 1.33 | +0.09% |
| 450–600 | 0.70 L | 1.23 L | +0.53 L | 1.75 | +0.23% |
| 600–750 | 0.70 L | 1.80 L | +1.11 L | 2.59 | +0.53% |
| 750–900 | 0.70 L | 2.53 L | **+1.83 L** | **3.62** | +1.03% |
| 900–1050 | 0.70 L | 0.71 L | +0.01 L | 1.01 | +1.03% |
| 1050–1200 | 0.69 L | 0.14 L | −0.55 L | 0.21 | +0.88% |

So the gain is not steady. It grows through the film's hot phase, peaks at 3.6× delivery over steps 750–900, stops abruptly at about 900, and reverses as the film cools to grid only (particle mass on the floor halves between 900 and 1050).

Stages in the right floor band: only advect (which includes Ferstl's shrink of φ0 by h) and reconstruct (the union) are material. Over steps 750–900, for example, advect was −9.97 L net (gain 21.0, loss −30.9) and reconstruct +12.46 L. Every other stage summed to ≤ 0.13 L over the whole run. By pre-stage thickness bin, the net mostly shows the film spreading: thin columns gain and thick ones lose. It cannot isolate the gain without inflow by column.

**Offset of the surface above each column's topmost sample** (column liquid height minus the top sample's y). For 8 samples per cell, the expected gap is about 0.125 h. Mean by bin, over the populated 1.5–6 h bins:

| Steps | Mean offset | Gain at that time |
|---|---|---|
| 300–600 | 0.04–0.09 h | small |
| 700–900 | 0.15–0.21 h; 0.34 h in the few thin columns at 900 | gain peaks |
| 1000–1050 | 0.23–0.37 h (3–6 h bins) | stopped |
| 1150–1200 | 0.15–0.27 h | reversing |

The top sample's radius is 0.50 h in every bin: tops are unit-mass samples, not merged ones. The offset is highest when the gain has already stopped. So φp's contour offset above the samples does not track the gain, and the offset hypothesis is not supported.

**What remains.** The gain tracks the hot, particle-carrying phase of the film, and reverses when it cools.

- Candidate 1: union and reseeding acting while the jet's particles accumulate. Floor particle mass rises from 4.2 L to 11.0 L between steps 600 and 900, about 1.5× nominal density.
- Candidate 2: the hot-to-cold handover.
- Next measurement: seeded mass on the right (`--retireSplit=1120,896` counts it) against the excess per interval.

No fix is proposed.

### Follow-up 2: seeded mass, mass-aware reseeding, Sato's heat threshold

**Seeded mass on the right** (main = n + shape ungated). Run with scratch counters in `seedCandidate`, logged to `seep-shape-seedbins.*` and `seep-shape-seedmass.*`; seep 1200 steps, x ≥ 1120.

- (a) Seeded mass is 3–20× the excess. For example, 750–900 seeded 17.4 L against an excess of +1.80 L. Most of it is churn: deep retirement removes 14.5 L over 750–900 and 29.5 L over 900–1050.
- (b) In the floor band every seed is at φ ≤ −h. Film thickness is estimated as height above the floor minus φ. During the gain phase (600–900), 2–4 h films at [−2h, −h] take 3.3–3.6 L per interval; later seeds go to films of 4 h or more.
- (c) Floor-band seeds within h of the surface: ≤ 80 records per interval. Films under 2 h: 0.04–0.06 L per interval. So Ferstl's [−R, −h] rule holds, and seeds in thin hot films do not explain the excess. One caveat: the thickness estimate uses the same φ, so a φ that is itself too thick would not show as thin.
- Exception: 9–12k unit seeds per interval within h of the surface at the seep-hole exit (x ≈ 1124, y ≈ 959). They come from `outerSeeding`/`activationSeeding` = `"uncovered"`, the defaults in `uniform-narrow-band-method.ts`. That breaks Ferstl's rule. They add a steady 0.15 L per interval from step 0, when the excess is still 0. **Open item.**

**The disagreement with Ferstl.** Seeded cells held 2.7 records on average but 9–10 units of mass before seeding, against a nominal 8. Merge mode lowers the record count, and the reseed then tops the count back up.

- 71% / 57% / 45% of seeded cells were already at 8 units or more (600–750 / 750–900 / 900–1050).
- Mass seeded beyond 8 units per cell: 3.8 / 8.6 / 10.7 / 8.6 L per interval after step 600, which is 50–70% of all seeded mass.

Ferstl §3.3 reseeds cells with fewer than n = 8 unit particles. With mass-carrying records, that test is on mass.

**Landed: mass-aware reseeding** (`uniform-narrow-band-flip.ts`). `nbCellMass` sums the filed samples' mass over `nbCellRun`. `seedCells` skips a cell at 8 units or more, and `seedCandidate` seeds only while the cell's mass before the seed is below 8.

Gates:
- types clean;
- unit tests 1398 / 1259 pass / 0 fail / 139 skipped;
- rest drift 0 (`rest-mfix-applied.json`).

Seep at 1200 steps (`seep-mfix-1200.*`):
- Ratio 1.72 (main 2.09 / 2.19). Worst drift +1.10% (main +1.66 / +1.80%). Right volume 9.65 L (main 11.75 / 12.25).
- Excess per interval: −0.02, +0.05, +0.10, +0.30, +0.74, +1.09, +1.02, +0.78 L. Main: −0.03, +0.11, +0.20, +0.38, +1.11, +1.90, +1.48, +0.98 L.
- Particle peak 400k (main 427k). Seeded mass at 750–900 is 5.6 L (main 16.1).
- Ratio over 300–600 in a 600-step run (`seep600-mfix.*`): 1.30 (main 1.41 / 1.52).

Letters, check run: −0.99 / −2.70 / −3.39 / −3.84% at 60 / 120 / 180 / 240.

**Per-stage ledger, 300–600, mfix** (`seep600-mfix-ledger.*`). On the right, advect (which includes the h shrink) moves −16.10 and −13.89 L. Reconstruct moves +16.90 and +14.89 L. The net is the right-side gain (0.80 and 1.00 L, against 0.70 L lost on the left). Redistance is ≤ 0.03 L and every other stage is 0. The union restores more than advect plus the shrink removed: the particle envelope extends past the advected surface.

**Sato's heat threshold: dropped.** The scratch tree `trees/sato` replaced need-mode shape heat with Sato's geometric test, with refinement unchanged:

- c ≈ −(σ²Δx²/2)∇²φ, σ = 1 cell, heat if |c|/Δx > 0.2;
- evaluated on crossing tiles' cells within h;
- no fast-tile exemption.

Letters at 120 steps: −3.47% (main −2.64%), with the same mean particle count (503k against 496k) and slower (54.5 against 46.6 ms over 120 steps, check runs).

Falling letters are all corners, so any faithful geometric test heats them. The letters cost is the particles correctness needs, not a bad trigger. Shape ungated costs +12.5 ms (check run) for about 385k more mean particles over 240 steps (825k against 331k for n). That cost is to be won back on the per-particle and per-tile side.

## Follow-up 3: the shared loss is the free-surface energy bound

**Probe fix first.** `tools/probe-tier-budget.ts` `particleMassAt` read `particleBins[binCells+…+bin]` under merge. U9 removed that mass plane, so it read tile coverage words. It now reads `particleBins[bin]`. The bare/covered split of the in-flight `seep600-sepoff-ledger.*` (about 99% "bare") is therefore **invalid**; its stage totals stand. The per-cell ledgers below come from a scratch probe that already read `bins[bin]`.

**Where the loss is.** Per-cell ledgers, separation off:
- `lcensus-sepoff.*`: letters, 120 steps.
- `seep600-sepoff-cells.*`: seep, 600 steps.

Both scenes lose in grid φ advection, and the union gives back only part of it:
- Seep right tank, 300–600: advect −24.4 L, union +25.5 L.
- Letters 0–120: advect −899 L (net of the drops), union +395 L, other stages +11 L, so −493 L (−3.99%).

Bare cells are not where the union fails. Over 120 steps, the union's own change in cells bare at the union is:
- outer [−1,0), where the −h rule refuses reseeding: −33 L;
- with the neighbourhood uncovered: −0.9 L;
- [−1.5,−1): −0.1 L;
- deeper bare or emptied cells: about 0;
- never heated: +2.3 L.

So bare-cell erosion explains under 10% of the loss. The advect deltas per cell are transport-dominated (cov→bare −3299 L against bare→cov +2680 L). I tried four variants against the loss (letters 120 / seep ratio):
- surfaceAdvection best: −3.77% / 0.80;
- phiFineAdvection cubic: −3.73% / 0.79;
- bandExit strict: −3.96% / 0.82;
- activationSeeding band: −3.65%.

All four are within noise. innerReseed sites gives −3.50% / 1.16: a gain, at 310k particles.

**Cause.** `surfaceImpulse:"energy"`, the default since 6b9fb237, is the culprit. `bSurface` capped the outward speed of every liquid→air face at sqrt(u0²+2p/ρ). It ran after the projection, so it removed outflow from a field the solve had just made divergence-free:
- surface cells converge;
- the advected surface moves less than the liquid it bounds;
- that volume is lost.

The bound is a clamp. It appears in neither Ferstl 2016 nor Sato 2018: both use the ordinary ghost-fluid free surface. Setting `"full"` (the standard GFM impulse) moved both scenes the right way.

**Provenance.** Commit 6b9fb237 added the bound to stop 25–40 m/s slivers in the pour at 1/24 s (CFL ≈ 20), and moved the closure-cap fatal from frame 43 to frame 90. Rerun now with `tools/probe-nbflip-speed.ts` and a requested-bytes cap (scratch `loss/pour/*.log`). "travel" is the census's largest tile face speed:

| pour | median travel | max | steps > 15 / > 20 m/s | fatal |
|---|---|---|---|---|
| 1/24 s, energy, 120 steps | 11.6 m/s | 60.8 m/s | 13 / 6 | none |
| 1/24 s, full | 21.8 m/s | 51.5 m/s | 99 / 73 | none |
| 1/60 s, energy, 300 steps | 9.2 m/s | 15.4 m/s | 1 / 0 | none |
| 1/60 s, full | 9.4 m/s | 26.0 m/s | 29 / 5 | none |
| 1/60 s, full, θmin 1e-2 (diagnostic) | 9.4 m/s | 33.6 m/s | 23 / 9 | none |

- The closure-cap fatal is gone under both settings.
- At the fixed 1/60 s, "full" brings back transient face-speed spikes: up to 26 m/s, against a free fall of about 9.6 m/s.
- Raising θmin 1e-3 → 1e-2 does not remove them, so tiny-θ amplification is not the cause.
- Letters 240 at 1/60 s (`loss/letters-speed-*.log`): energy max 7.6 m/s, 0 steps above 10 m/s. Full max 16.2 m/s, 17 steps above 10 m/s, none above 20 m/s. No fatal; drift at 240 is −1.0% / +0.8%.
- The upstream source of these spikes is **open**. The bound hid them by breaking the divergence-free condition.

**A/B** (check runs; medians are contended wall time):

| run | seep worst drift | seep receiving ratio (0–600) | letters 120 | step median seep / letters | particles max seep / letters |
|---|---|---|---|---|---|
| sep off, energy | −0.144% | 0.80 | −3.95% | 33.9 / 43.4 ms | 173k / 1189k |
| sep off, full ×2 | −0.022 / −0.019% | 0.970 / 0.974 | −2.72 / −2.66% | 38.6, 39.6 / 45.6, 47.0 ms | 215k, 222k / 1084k, 1091k |
| sep on, energy | +0.125% | 1.163 | −2.75% | 35.8 / 47.2 ms | 210k / 1038k |
| sep on, full ×2 | +0.390 / +0.425% | 1.533 / 1.582 | −1.51% | 37.8, 40.1 / 49.0 ms | 265k, 275k / 963k |

With separation off, "full" clearly improves both scenes. With separation on, letters improve and the seep gets worse: separation's outward volume source (4:1) is no longer offset by the loss.

**Landed: the paper's free surface** (no new knob):
- `surfaceImpulse` and the `bSurface` energy branch are deleted from `uniform-pressure-arena.ts`. The open face and the liquid/air band face return the plain ghost-fluid value.
- The schedule field is removed from `uniform-near-settled.ts`, `uniform-mixed-frame.ts` and the method parameter in `uniform-narrow-band-method.ts`.

Gates:
- types: the touched files are clean. The only `tsc` errors are in `lib/features/persistence.test.ts` (`"reflection-a"` PaneId), another session's in-flight edit;
- unit tests: 1408 / 1268 pass / **0 fail** / 140 skipped;
- closed rest: drift **0** at steps 30/60/90/120, maxSpeed 7.5e-6 m/s (`rest-full-applied.json`).

Confirmation, production defaults (separation on), snapshot of main with the change landed:
- Letters 240 (`letters-letters240-full.*`): **−0.18%** (main −3.90%). By step: −0.90 / −1.50 / −1.00 / −0.18% at 60 / 120 / 180 / 240. Median 86.3 ms (main 82.9, check runs); mean 737k particles (806k).
- Seep 1200 (`seep1200-full.*`): ratio **2.18** (mfix 1.72), worst drift **+1.72%** (+1.10%), right 11.72 L (9.65). Median 48.5 / p95 89.4 ms (39.8 / 61.0, contended). Particles peak 384k (400k).

**Remaining letters loss, separation off, full** (`lcells-full-sepoff.*`). Over 0–120: advect −815 L net of the drops, union +467 L (57%), other stages +11 L, so −337 L (−2.73%). The union's change in cells bare at the union is again about 0 (outer [−1,0) −35 L, never heated +1.8 L). By coverage transition:
- cov→cov cells: advect −803 L, union +279 L;
- bare→cov: advect +2634 L, union +170 L;
- cov→bare: advect −3198 L.

The unrecovered part sits in particle-covered cells, where the union does not reach the advected-away surface. It does not sit in the bare cells that the −h rule leaves. Next: compare φp's own volume against the target to learn whether the particle envelope itself lags.

**Separation**: not yet made volume-neutral. Sato §3.2.2 cites [ATT12, FAW*16] for the pair correction, with no surface treatment given. Our pairs push a surface sample outward with nothing to push back, which is the measured 4:1 outward bias.

## Follow-up 4: separation, placement and the trace (2026-10-28)

Task: make `sampleSeparation:"pairs"` volume-neutral in a way the papers support.

**Result**: the shell's normal projection is replaced by jittered sub-cell seeding (Zhu and Bridson 2005; Bridson 2008), and it is landed. Separation stays on. Separation is *not* made volume-neutral. Its seep volume source answers clumps that the RK3 trace makes during transport; that is open item O1 below. No gate was landed and there is no volume correction.

Fixed table (main before this follow-up):

| config | seep worst drift | seep receiving ratio | letters 120 / 240 | step medians, seep / letters |
|---|---|---|---|---|
| production, separation on, seep 1200 / letters 240 | +1.72% | 2.18 | −1.50% / −0.18% | 48.5 / 86.3 ms |
| separation off, seep 600 / letters 120 | −0.02% | 0.97 | −2.7% | 39 / 46 ms |

### Sources
- **Sato 2018 §3.2.2**: d_ij = m_j/(m_i+m_j)(r_i+r_j−|p_i−p_j|), cites [ATT12, FAW*16]. It gives no radii and says nothing about the free surface.
- **Ando, Thürey and Tsuruno 2012** (ATT12, TVCG, §5.2, Eq. 18):
  - Δp_i = −Δt γs d_i Σ (p_j−p_i)/|p_j−p_i| W(p_j−p_i, d_i), with W = 1−r²/d²;
  - d0 = Δx/2, the same 0.5h reach as ours. γs = 50, Δt = 0.006, described as "very weak";
  - they note the isotropic form "can lead to a thickening of thin surfaces" and replace it with the anisotropic kernel C⁻¹ (Eqs. 19–22);
  - it exists to keep samples evenly spread, not to restore volume.
- **Um, Baek and Han 2014** (UBH14, sub-grid particle correction):
  - density-error pressure kp(ρ/ρ0−1);
  - their free-surface artefact (outward push where neighbours are missing) is cured by clamping that pressure at zero or by ghost particles.
- **Ferstl 2016**: the light-weight NB-FLIP solver uses no position correction; only their ATT12 port does.

### Where the expansion comes from (letters 120, seep 600; diagnostic build)
- Separation's normal displacement, outward:inward:
  - interior 0.70;
  - within h of the surface 3.65 (letters) and 3.6 (seep);
  - outside φ, 10–12.
- About 60% of the letters on/off volume difference is the outward sweep within h. The φp union turns 15–20% of it into volume.
- **Separation off is clumpy.** Share of samples with a neighbour closer than h/8, Poisson ≈ 6% at 8 per cell:
  - letters: within h 46% (on: 5.7%), interior 15% (2.4%);
  - seep: 43–45% (on: 2.6–4.9%);
  - contacts within 0.5h: 18–20 against 4.2 expected.

**Divergence**, separation off, dt·div per step. Columns: φ < −h / [−h, 0) / outside.

| field | letters | seep |
|---|---|---|
| point div of the interpolated velocity | −0.0001 / +0.0009 / −0.005 | ≈ MAC |
| RK3 trace Jacobian det J − 1 | −0.0067 / −0.024 / −0.063 | within h +0.009 overall, −0.12 in the jet phase; outside −0.13 |
| MAC cell divergence | −0.0001 / +0.0009 / −0.003 | within h −0.002 |

The grid and the interpolated field are divergence-free to within 1e-3. The compression is the trace's (fixed dt 1/60):
- In free fall det J − 1 = 0, although 14% of stage points sample air.
- Near the pool, stage-3 along-flow speed is 1.6–5% slow and det J − 1 is −0.01 to −0.066, consistent with the nearest-source extension handing stage points the pool's rest velocity.

### Placement
The outer shell's history: c942473a fitted sites shallower than 0.875h along ∇φ to depth √(0.875²−0.125)h, "to avoid jitter-induced rest-state ripples". That fit lands the two lattice sites along the normal on one point. Those are coincident pairs that neither paper creates; separation then splits them and raises the upper member. Two variants were A/B'd:
- **(a)**, a stacked pair seeded as one mass-2 sample: no effect. A merge within one cell misses pairs that straddle two cells, which happens whenever the surface lies mid-cell.
- **(b)**, jittered sub-cell sites with no normal fit, for the initial band and the inner reseed alike: chosen.

Replicates ×2, seep 600 / letters 120:

| variant | seep worst drift | ratio | median | peak particles | letters 60 / 120 | median | mean particles | nn<h/8 within h / interior (letters) |
|---|---|---|---|---|---|---|---|---|
| base, on | +0.35 / +0.41% | 1.484 / 1.563 | 63.7 / 43.1 | 263k / 271k | −0.91 / −1.57, −0.85 / −1.27% | 89 / 78 | 469k / 472k | 0.057, 0.055 / 0.024 |
| (a), on | +0.38 / +0.42% | 1.526 / 1.573 | 42.7 / 46.2 | 267k / 277k | −0.90 / −1.49, −0.89 / −1.50% | 113 / 56 | 472k | 0.053, 0.051 / 0.024 |
| **(b), on** | +0.37 / +0.38% | 1.506 / 1.520 | 43.6 / 43.2 | 265k / 269k | −0.78 / −1.02, −0.78 / −0.99% | 62 / 61 | 497k / 494k | 0.042, 0.041 / 0.024 |
| base, off | −0.04% | 0.951 | 45.3 | 220k | −1.07 / −2.74% | 60 | 508k | 0.456 / 0.150 |
| (a), off | −0.02 / +0.04% | 0.968 / 1.043 | 44.3 / 43.6 | 224k / 235k | −2.62 / −2.82% | 85 / 83 | 508k / 494k | 0.350, 0.361 / 0.14 |
| **(b), off** | −0.03 / −0.04% | 0.968 / 0.952 | 39.1 / 42.2 | 218k / 217k | −0.91 / −2.09, −0.91 / −2.16% | 62 / 70 | 525k / 524k | 0.189, 0.189 / 0.14 |

Medians are contended check runs.
- (b) improves letters in both modes and halves the clumps at creation.
- Seep is neutral.
- Seep's separation-off time series shows why. (b) halves nn<h/8 at step 25 (within h 0.112 against 0.226). By step 50 base and (b) alike reach 0.33–0.42, and 0.45–0.50 from step 225 on. Seep's clumps are made by the trace in the jet, not by seeding.
- Separation's outward:inward ratio is unchanged (4.0).

UBH14 non-negative gate (A/B only, not landed): seep worst drift +0.18%, ratio 1.245; letters −0.98 / −2.01%. It helps one scene and hurts the other, so it is reverted.

### Landed (`uniform-narrow-band-flip.ts`)
- `seedSite` places site k at c + 0.5·corner(k) + 0.5·hash, inside its eighth of the cell.
- `nbJitter` / `nbSeedSalt`: the initial band's salt is 0, so the count pass and the emit pass agree. The reseed salt is the seed-list slot plus the list length.
- The `seedShell` fit and its include are removed. `uniform-narrow-band-seed.wgsl.ts` stays, still used by `tests/uniform-rigid-containment.test.ts`.
- φp's shell-calibrated slack in `uniform-narrow-band-surface.wgsl.ts` is unchanged.

Gates:
- `npm run check:types` clean;
- `npm run test:unit` 1429 / 1282 pass / **0 fail** / 147 skipped;
- closed rest: drift **0** at 30/60/90/120 steps, maxSpeed 7.55e-6 m/s, for (b) on and off (`rest-f4-b-on/off.json`). This is identical to the digit with base, so the closed rest state does not depend on seed positions and is not a ripple check. The c942473a wrinkle test no longer exists.
- Creation offset, seep: φ volume at steps 1–4 is +1.9e-6 / +5.1e-6 / +1.1e-5 / +2.2e-5 (base) against +1.8e-6 / +4.4e-6 / +1.1e-5 / +2.2e-5 ((b)), so none.

Confirmation, snapshot of main with (b), full length:

| config | seep 1200 worst drift | ratio | right | median / p95 | peak particles | letters 60 / 120 / 180 / 240 | median | particles mean / max |
|---|---|---|---|---|---|---|---|---|
| production before | +1.72% | 2.18 | 11.72 L | 48.5 / 89.4 | 384k | −0.90 / −1.50 / −1.00 / −0.18% | 86.3 | 737k / 1064k |
| **(b), on** | +1.73% | 2.19 | 11.76 L | 49.3 / 82.8 | 385k | −0.78 / −1.00 / −0.32 / **+0.68%** | 101.6 | 759k / 1088k |
| (b), off | −0.54% | **0.632** | 3.40 L | 43.9 / 49.5 | 436k | −0.90 / −2.11 / −2.58 / **−3.07%** | 93.3 | 891k / 1405k |

Medians are check runs, with seep and letters concurrent. Exclusive timing, separation on, letters 120 ×3 / seep 600 ×2, pre against landed (`f4t-*`): letters medians 68.8 / 48.8 / 65.9 ms against 192.4 / 117.1 / 59.4 ms; seep 43.4 / 145.6 against 37.1 / 38.2 ms. The landed letters runs 1–2 are slow at identical work: at frame 15 both have 416 fine tiles and 56k particles, yet take 218 and 77 ms against 22 and 17. That is contention. The clean third pair matches pre by quarter (19 / 40 / 74 / 94 against 20 / 48 / 73 ms). No step-time change is measurable beyond the 1.66x noise. Letters carries 1–10% more particles at frame 120 (978k–1064k against 956k–965k).

### Separation off as the default: rejected
Ferstl's light-weight solver has no position correction. With (b), off meets both short gates. At full length it fails:
- letters 240 reaches −3.07%;
- the seep receiving ratio falls to 0.63 by step 1200 (0.97 at 600) while the discharge is the same (5.4 L);
- the sampling cost: nn<h/8 at 43–46% in seep against 3–5% with separation on, and 17% more particles on average (29% at peak) in letters 240, because clumps leave cells short and the reseed refills them.

Separation therefore stays the default.

### Open items
- **O1, trace compression.** The RK3 trace at fixed dt 1/60 compresses the sample cloud wherever the velocity gradient over one step is large, and where stage points read extended air velocity:
  - in the jet: det J − 1 ≈ −0.12 per step;
  - outside φ: −0.13;
  - between the letters and the pool: −0.01 to −0.066.

  That is the source of the clumps separation answers, and so of seep's outward volume (ratio 1.5 at 600, 2.19 at 1200). The fix belongs in velocity sampling or extrapolation (the nearest-source extension's rest velocity at stage points in the air between letter and pool), not in separation. It is deliberately not fixed here.
- **O2, letters loss.** The −3% loss with separation off at 240 is the clumps' cost, and is the same mechanism.
- **O3, face-speed spikes.** After the bound's removal (task 5): not investigated in this follow-up.

### Closed (2026-10-30): the seep's receiving ratio above 1
The ratio above 1 was separation's outward source, not O1's compression.
In O1, compression made the clumps, but the volume came from the pair
push's normal part at the free surface. A surface sample has no neighbours
on the air side, so every overlap pushes it out, and the union keeps what it
pushes. Within h of the surface the push ran outward:inward 7.07 in a
channel dam's run-up and 3.48 on a wide dam. It was the measured 4:1 here.

The fix: separation keeps only its tangential part within a cell of the
surface, d − (d·n)n with n = ∇φ. This is particle shifting's free-surface
rule: Khayyer et al. 2017 cancel the normal shift at the free surface and
its neighbourhood; Sun et al. 2019 give it as Eq. 44, after Lind et al.
2012. Afterwards the ratio is 1.00 and the seep at 1200 is:

| | before | after |
|---|---|---|
| worst drift | +1.98% | +0.14% |
| right tank | 12.77 L | 5.93 L, against the left's 5.46 L loss |
| late right-volume Cd against throat Cd | 2.46 against 0.78 | 0.79 against 0.78 |

O1 and O2 stay open. Letters at 240 now read −1.38%, where they read
+0.86%: part of the old figure was this source hiding O2's loss. The clump
share within h is 0.07, against 0.46 with separation off. Details:
`docs/plans/bulk16-plan-2026-10-28.md`, M4b, "the dam's volume gain".
