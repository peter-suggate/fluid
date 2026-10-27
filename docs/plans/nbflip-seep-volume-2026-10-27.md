# NB-FLIP seep: right-tank volume loss (2026-10-27)

Worktree `fluid-nbflip-seep`, branch `claude/nbflip-seep-realtime`, base `ad40be44`, main defaults. Artifacts are under `artifacts/seep-baseline/` (gitignored).

## Baseline: 1200 steps, 5 mm, `seep5-milestone`
- **Throat:** Cd 0.816 throughout. The left level fell 14.6 mm, against about 15 mm from Torricelli.
- **Right tank:** ends at 2.08 L of about 5.8 L delivered. Late in the run it receives 9% of the throat flux.
- **Drift:** −0.90% of total volume (about 3.3 L).
- **Particles:** 97k → 403k. The step median is 47.6 ms, with p95 156 ms and 213 ms at the end; these are check-mode times.
- **Memory:** fixed at 3.31 GiB.

## Mechanism
- **The particles hold the water.** The new `--particleMass` diagnostic reads the live particle bank. At step 600 the right-chamber particle mass is 3.17 L (3.12 L of it in the 8-cell floor band), while the level set holds 1.74 L there.
- **Advection erodes; the union restores.** `--flowStages` shows that grid advection of phi removes about 50 mL per step in the right chamber, and the particle union adds back about 50–60 mL. The shortfall varies, and it is the drift.
- **The film saturates.** The right-tank level set plateaus at 0.86 cell over the whole floor, which is the sphere radius `NARROW_BAND_SURFACE_RADIUS` = 0.875. For a film thinner than h, Ferstl's `min(phi'+h, phi_p)` leaves only the sample spheres, and they hold no more than 0.875 h at a floor vertex.
- **Excess samples pile up.** Liquid beyond that depth exists only as samples. Pressure never sees it, so nothing lifts it; samples pile into the film cells, and all 1.18M crowded-cell merges are at x ≥ 256.

## Experiments, both reverted
1. **Union at full particle weight (theta = 1)** at gathered vertices: right tank 1.77 L versus 1.74 L. Heat blending is not the cause.
2. **Supply phi from uncapped sample counts** where a cell holds more than 8: volume **gains** 3.3 L in 8 s and samples reach 880k. Seeding creates samples wherever phi is liquid, so sample mass is not conserved, and feeding it back into phi forms a loop. This is the reason for the existing 8-sample cap.

## Conclusion
Neither phi nor the samples is a conserved carrier in the band: samples are created by seeding, and phi erodes in sub-cell films. A real fix has to make one of them conserved.

## Checked against the papers
- **Ferstl 2016 §3.3:** particles are massless markers. Resampling adds or removes markers only where −R ≤ φ ≤ −h, with at least n = 8 and at most 2n per cell. The paper says volume conservation is "equivalent to regular FLIP": not exact, with changes originating at the surface. The limitations section also notes that structures thinner than 2r are carried by particles alone.
- **Sato 2018:** keeps that model. §3.2.2 adds pair position correction, `d_ij = m_j/(m_i+m_j)(r_i+r_j−|p_i−p_j|)`, with a velocity Jacobian term, and still removes markers above a per-cell maximum.
- **Neither paper uses conserved particle mass.** The conserved-carrier plan would be an extension beyond both.

## Where the right-chamber samples sit (`rightY_quarterCells`, `rightClamped_m3`)
- **Squashed onto the floor.** At step 600, 164k of the 213k right-chamber mass units (77%) lie below 0.25 h. That is why the sphere union stops at 0.86 h.
- **Partly the wall clamp.** `nbTraceEndpoint` clamps trace ends to 0.01 inside fixed box walls. A single RK3 step at jet speed (CFL around 10) puts overshooting samples exactly on y = 0.01. That plane holds 24% of the right-chamber mass at step 300 (0.41 of 1.69 L) and 8% at step 600 (0.26 of 3.17 L).
- **Mostly between 0.01 and 0.25 h.** In the floor cell the interpolated v goes to 0 linearly at the floor, so samples that sink there barely rise again.
- **The bound pins the film.** With cell y = 1 empty, the walls bound keeps phi ≥ 0 at the y = 1 vertex, so the level-set film can never exceed one cell.

## Experiment 3: Sato §3.2.2 pair separation in the move (reverted)
- **Setup:** r = cbrt(m)/4 (eight samples a cell sit half a cell apart), computed over the current bank's 3×3×3 sorted runs and added to the trace end. No velocity term.
- **The layer lifts.** At step 600 the bottom quarter-cell holds 29k units, against 164k without separation.
- **Volume is gained, not conserved.**
  - The left tank delivered 2.82 L by step 600, yet the right chamber's level set held 5.66 L (1.76 L in the baseline).
  - Right-chamber particle mass was 5.0 L, and total drift +0.54%.
  - Records reached 818k at step 450.
- **Why:** separation spreads a disordered cloud below eight a cell. Ferstl's interior reseeding (eight a cell in [−R, −h]) then refills it, which pumps mass in.

## Continued in the main checkout (`/Users/petersuggate/code/me/fluid`, HEAD `ad40be44`)
All runs: `tools/probe-tier-budget.ts`, reservoir-seep at 5 mm, dt 1/60 with no substeps, `volumeControlSeconds` 0, check mode unless marked timing. Artifacts are in `artifacts/seep/<name>.log`. The split counters use `--retireSplit=1120,896`: the box origin is [864, 896, 992], so the right chamber starts at lattice x 1120 (`--retireSplit=256` counted the whole domain). "Delivered" is the left tank's loss; "right" is the level set's liquid at x ≥ 256.

### Experiment 4: separation with whole-cell inner reseeding (`sep15`, three repeats, 600 steps)
- **Setup:** separation as in experiment 3, plus a reseed condition: inner reseeding only in cells whose centre is 1.5 cells or more below the surface (`--reseedCentre=1.5`).
- **Right at 600:** 2.05 / 2.12 / 2.20 L against 2.82 L delivered (baseline 1.76 L).
- **Drift and particles:** drift −0.133 / −0.111 / −0.101%; 128k–138k particles (baseline 196k).
- **Throat:** Cd 0.813–0.816.
- **Particle mass:** the right chamber's mass equals its level set (2.06 L against 2.05 L).
- **Remaining loss:** about 65–72% of the throat flux arrives once past the first window.
- **Verdict:** whole-cell reseeding stops experiment 3's pumping. The remaining loss is elsewhere.

### Wall zone
- **Not real liquid:** the wall zone (x 240–256) jumps from 0.11 to about 0.43 L in the first 60 steps in every run, the baseline included. It is phi extrapolated into the solid.
- **Out of the drift:** the solver's drift excludes it.

### Right-chamber particle budget (`sep15c`, split counters)
- **It closes:** seeds offered (298k units) − deep-retired mass (156.5k) = 141k units = 2.21 L, which is the measured particle mass.
- **Where deep retirement happens:** on the far floor (x ≥ 496), where the water piles more than 3 cells deep. None of it is at the outlet.

### Throat seeding (150 steps, temporary counters, reverted)
- **No seeding at the outlet:** all 7168 throat-cell visits pass the reach test and none passes the heat test. The throat tiles are never hot ("need" seeding), so the jet leaves as level set only.
- **Phi is not a distance there:** about −30 at the centre.

### Ledger of the phi stages (`--ledger=A-B`, a new probe option)
`--ledger` sums each phi stage's change in volume per zone over a step range.
- **`sep15l` (steps 301–450):**
  - Right chamber: advect −13112 mL, reconstruct +13548 mL.
  - Whole domain: redistance −13 mL.
  - The right chamber gained 436 mL net against 705 mL delivered.
- **`sep15m` (steps 301–400):**
  - Jet zone (above the floor band, within 64 cells of the wall): advect −49 mL, reconstruct +50 mL. It balances.
  - Floor band: advect −10129 mL, reconstruct +10429 mL.
  - Whole domain: −170 mL.
- **Verdict:** the loss is not at the jet's handoff. It is in the floor film's advect/union exchange: advection erodes about 100 mL a step and the union restores 98%.
- **Film depth:** at step 450 the film covers 16250 of the 16384 floor cells, about 0.8 cell deep on average.

### Experiment 5: seeding diagnosis at 450 steps (split counters by kind)
Delivered at 450 is 2.11 L.

| Run | Change | Right at 450 (L) | pmR (L) | Drift | Particles |
|---|---|---|---|---|---|
| sep15n | sep15 | 1.65 | 1.62 | −0.068% | 101k |
| sep15act | + `particleSeeding:activity` | 1.44 | 1.38 | −0.074% | 85k |
| sep15d4 | + `particleBandDepth:4` | 2.26 | 2.48 | +0.066% | 150k |
| sep15uni | + `particleDensity:uniform` | 2.22 | 2.40 | +0.058% | 151k |

- **Heat coverage is not the loss:** heating everything (activity) does no better.
- **Graded density is:** graded seeds four unit-mass samples per deep cell. The union's sample bound counts records (eight to a full cell), so a graded cell reads as half its liquid once its samples rise into the film. Eight a cell (uniform, Ferstl's n) removes most of the loss.
- **What is left over-delivers:** with uniform density the right chamber runs +0.2 L above delivered from the first 150 steps.
- **Activated seeds match the excess:** they total 13.5k units (0.21 L) by step 150. A newly heated tile seeds its whole band, outer layer included, wherever a cell is under its count, on top of the samples already carrying that liquid.

### Experiment 6: activation seeds only where uncovered (`uniact`)
- **Setup:** a newly heated tile follows the outer-seeding rule (no sample within a cell) instead of seeding its whole band.
- **450 steps (`uniact`):**
  - Right 0.63 / 1.37 / 1.96 L against 0.70 / 1.41 / 2.11 L delivered.
  - Drift −0.012%.
  - Activated seeds fell from 22k to 2.6k.
- **`uniact10` (inner reseeding back to 1.0):** gains. Right is 3.71 L against 2.82 L at 600, drift +0.208%. Whole-cell reseeding is required.
- **`uniact-nosep` (no separation):**
  - Right level set is 1.60 L at 600, while the particle mass is 2.67 L.
  - 87% of the samples sit in the bottom quarter cell.
  - Separation is required: without it the samples pack onto the floor and the union rebuilds too little liquid.

### Experiment 7: the candidate at 1200 steps
The candidate is separation, whole-cell reseeding, uncovered activation and uniform density.

| Run | Right at 600 / 750 / 900 / 1050 / 1200 (L) | Delivered (L) | Drift at 1200 | Particles max |
|---|---|---|---|---|
| cand1200 | 2.81 / 3.44 / 3.70 / 4.00 / 3.99 | 2.82 / 3.52 / 4.22 / 4.92 / 5.62 | −0.252% | 231k |
| candl (1000 steps) | right 3.81 at step 1000 | 4.70 at step 1000 | −0.178% (at 1000) | 229k |

- **Tracks, then stalls:** the right chamber tracks delivered volume within 1–2% to step 750, then stalls.
- **The stall coincides with heat churn:**
  - Heat retirement goes from 3k units at step 600 to 669k at 1200.
  - Activated seeds go from 2.7k to 278k.
  - The calm, two-cell-deep pool cools, its samples retire, and it reheats.
- **Ledger, steps 901–1000:** the floor band shows advect −6073 mL and reconstruct +6280 mL, net −264 mL over 100 steps.

### Experiment 8: is the heat cycle the late loss? (diagnostics, not fixes)
| Run | Change | Right at 1200 (L) | Delivered (L) | Drift | Particles max | Verdict |
|---|---|---|---|---|---|---|
| candfade2 | `adaptiveFadeSeconds:2` (heat lasts 4× longer) | 5.34 | 5.62 | −0.032% | 297k | Heat retirement halves (323k); the late loss mostly goes. |
| uniband1200 | Whole-band activation | 4.34 | 5.62 | −0.195% | 234k | The late loss stays, so it does not depend on the activation mode. |
| candhold | Shallow samples held at heat 1 everywhere | 6.44 | 5.62 | +0.129% | 272k | Gains, so the exchange errs both ways. |

- **Verdict:** the late loss is the cool/retire/reheat cycle of a calm film whose tiles are not thin.
- **Not used as fixes:**
  - The fade time is a user control, and tuning it would be scene tuning.
  - Holding shallow samples keeps samples over every calm surface. Memory and time would then scale with the whole surface, not the active region.
- **Unresolved:** the loss happens at the handoff between the level set and the samples when a calm film's tile cools and reheats.

### Experiment 9: graded mass fix and octant separation
- **Graded mass fix (replaces old behaviour, not an option):** under merge mode each of the graded four seeds in a deep cell now carries two units (`nbWithMass(...,select(1,2,deep))`, initial seeds `8/nbSeedTarget(d)`), so the record-count bound reads a graded cell as a full cell. Seed restate and the first-move reset keep the mass bits (`nbParticleRestate`), and the split seed counter counts mass, not records.
- **Octant separation:** Sato 2018 §3.2.2 pair separation now gathers only the 2×2×2 cells around the sample (the pair reach r_i+r_j ≤ 0.5 cell for unit masses, ≤ 0.63 for two-unit pairs), instead of 27 cells. The pair formula is unchanged.
- **Defaults:** `sampleSeparation:pairs`, `innerReseed:cells`, `activationSeeding:uncovered`, `particleDensity:graded` (with the mass fix). The `narrowBandExperiment` global is gone; the three toggles are select options in `uniform-narrow-band-method.ts` and parsed in `uniform-near-settled.ts`.

### Experiment 10: step-time A/B (timing mode, exclusive runner, 1200 steps, interleaved)
Base is `sampleSeparation:off innerReseed:sites activationSeeding:band particleDensity:graded`. It cannot switch off the graded mass fix.

| Series | Variant | Median (ms) | p95 (ms) | Drift at 1200 | Right at 1200 (L) | Particles |
|---|---|---|---|---|---|---|
| t | base ×3 | 33.0 / 33.2 / 33.1 | 36.5 / 37.1 / 40.8 | −0.81 / −0.78 / −0.73% | 2.08 / 2.11 / 2.24 | 378k / 385k / 435k |
| t | uniform density + 27-cell separation ×3 | 36.0 / 36.9 / 35.5 | 40.0 / 40.3 / 39.5 | −0.21 / −0.25 / −0.26% | about 4.2 | about 240k |
| — | uniform density, no separation ×3 | 32.2 / 31.4 / 31.9 | — | −0.82 to −0.85% | about 1.9 (particle mass 5.6) | — |
| v | base ×3 | 32.6 / 33.9 / 36.0 | 35.0 / 40.5 / 40.5 | — | — | — |
| v | uniform + octant separation ×3 | 36.3 / 36.3 / 37.0 | 39.2 / 38.8 / 39.7 | about −0.25% | — | — |
| w | base ×3 | 33.7 / 33.7 / 33.2 | 38.9 / 36.5 / 35.8 | −0.77 / −0.78 / −0.79% | — | — |
| w | **defaults** (graded mass + octant + cells + uncovered) ×3 | 36.0 / 35.5 / 35.5 | 39.0 / 40.2 / 41.4 | −0.086 / −0.060 / −0.117% | 4.65 / 4.92 / 4.64 (delivered 5.62) | about 200k |

- **Separation cost:** the 27-cell gather cost about 4 ms (12%) a step; the octant gather about 2–2.5 ms. Without separation the step is about 1 ms faster than base, but the level set holds a third of the delivered liquid.
- **Defaults vs base (w):** median +2.3 ms (+7%), p95 about +3 ms. **The step-time gate fails.** The volume gain (drift 0.78% → about 0.09%) is bought with separation's time.
- Cd 0.813–0.816 with the defaults. The right chamber tracks delivered volume within about 2–7% to step 900, then lags 13–17% by 1200 (the heat-cycle loss of experiment 8).
- One octant run (oct3) stopped at step 47 with "a particle arrived in page (42,31,32), which holds no storage; the residency closure is too small". Two reruns and all three w candidates passed. Separation moves a sample by at most half a cell beyond its trace, which the residency prediction does not include; the residency file was also changing under another agent at the time. Unattributed.

### Experiment 11: residency gate (`tests/uniform-stage-residency.test.ts`, "long slide")
| Tree | Result |
|---|---|
| main tree as found | FAIL, "empty bottom-left page must retire" (14 resident at step 30, against 12) |
| main tree, my toggles off and graded mass 1.0 | FAIL |
| clean HEAD af0e5dea + only my hunks | PASS |
| clean copy + the uncommitted `uniform-mixed-surface.ts` | FAIL |

- **Verdict:** it fails only with the foreign `uniform-mixed-surface.ts` hunk (another agent's surface advection diagnostics). It passes on a clean copy plus my hunks.

### Experiment 12: closed rest gate (`probe-reservoir-bulk.ts --closed --steps=120 --every=30`)
| Variant | Max speed at 30 / 60 / 90 / 120 (m/s) | Drift | Median / p95 (ms, check mode) |
|---|---|---|---|
| off (`sampleSeparation:off innerReseed:sites activationSeeding:band`) | 0.0219 / 0.0174 / 0.0150 / 0.0131 | 0 | 13.1 / 16.4 |
| defaults | 0.0219 / 0.0174 / 0.0150 / 0.0131 | 0 | 13.3 / 16.4 |

- Identical to every printed digit, with zero drift in both. That is consistent with the resting tank holding no narrow-band samples, so the changed code has nothing to act on; the sample count was not read directly.

### Experiment 13: letters splash gate (Ferstl Fig. 8 letters, dt 1/60, 240 steps, `volumeControlSeconds:0`)
- **Probe:** `tools/benchmark-nbflip-letters.ts` cannot serve: its `volumeDrift` is one `readStats` at the end, which returns the budget's last requested measurement (both variants printed the identical −0.0864), and the scene profile runs `volumeControlSeconds:0.1`. `--dt=0.0166667` also trips its "complete exactly one requested clock step" assert; `--dt=0.016666666666666666` passes. A scratch probe (scratchpad `probe-letters-volume.mts`) builds the scene the same way at dt 1/60 with volume control 0 and calls `readStats` every step.

| Run | Phi drift at 60 / 120 / 180 / 240 | Dense V drift at 240 | Median / p95 (ms, check mode) |
|---|---|---|---|
| off | −1.17 / −4.96 / −9.10 / −13.28% | −5.08% | 61.4 / 75.1 |
| off2 | — / −4.79 / — / −12.82% | −4.58% | — |
| defaults | −1.14 / −5.19 / −10.09 / −14.49% | −6.41% | 62.6 / 74.1 |
| defaults2 | — / −5.12 / — / −14.26% | −6.15% | — |

- **No volume gain:** the gate as written holds. Both variants lose heavily without volume control (the scene relies on it), and the defaults lose about 1.3 points more by step 240, repeatably across two pairs.

### Experiment 14: where the excess sample mass sits (crowd census, scratch `probe-crowd4.mts`)
- **Piles are volumetric, not flat on a face.** At step 450, cell (1120, 896, 1035) held 1158 records of total mass 1325, spread through the cell.
- **The excess explains the drift.** Mass above 8 a cell is invisible to the union, because the bound counts `min(count, 8)`. In the right tank it was 0.25 / 0.91 / 1.05 / 1.37 L at steps 300 / 600 / 900 / 1200, about the drift at the same steps (−0.36%, about 1.3 L).
- **Sato ω = 1.5 over-relaxation was rejected.** At 1200 it gave 1.11 L of excess against 1.29 L, but the right sample mass fell to 4.09 L against 4.37 L.
- **Not a ghost-velocity sink.**
  - The grid's normal velocity goes to 0 at every face.
  - Seeds landed in crowded cells 0 times over steps 400–415.
  - The crowds are clumps travelling along the corner edge.
- **They read as air.** The LSQ divergence of pic in the wall-foot crowded cells was −0.28 to −0.69 cell volumes a step, while their phi depth was +0 to +0.2.

### Experiment 15: the cold-tile union skip (landed, unconditional)
- **Cause.** The right tank's wall-face vertices at x = 1120 belong to the left tank's COLD wall tile (page 34). `coupleFine`, `surfaceClear` and `coarseCorners` returned early on `nbTileCold(owner.tile)`, so the union never ran on vertices that a hot tile's cells use. The dump showed those vertices at −0.85 to +2.2 on the floor row and +2.4 to +3.9 one row up, next to −0.5 in the interior.
- **Fix.** The tile-level early-out now applies only where the COLD tile's cells are all open (`umTileOpen >= 1`). There, every vertex the tile owns touches only COLD cells, so skipping is exact. A COLD tile with solid cells falls through to the per-vertex `nbVertexCold` test. `coarseCorners` keeps the couple flag for such tiles and drops retirement.
- **Measured.** The wall-face vertices read −0.4 to −0.8, and the x = 1120 crowds are gone. Excess was 0.50 against 0.78 L at step 600, and 1.18 against 1.29 L at 1200. Drift at 1200 over 3 runs was −0.43 / −0.53 / −0.54%.

### Experiment 16: wall-plane trap and contact radius (landed, one rule for every solid)
- **Mechanism.** The box clamp put trace ends at 0.01 inside the walls. The grid's normal velocity is zero on the wall plane, so a sample there rides along the face and never leaves it.
- **Rule.** A sample's centre keeps its radius r(m) = cbrt(m)/4 from every solid, measured along the solid's normal. It is one table, `nbSampleRadius`, applied to:
  - the box walls: `nbTraceEndpointIn`, which replaces the 0.01 clamp;
  - voxel and SOLID page faces, in `nbFinish`'s walk: the leading edge is tested and the centre stops at face ∓ r;
  - the procedural field (`nbProceduralContact`): pushed out to r along the normal whether or not the segment crossed the surface;
  - terrain (`nbTerrainContact`): vertical clearance r·min(h)·|(s_x/h_x, 1/h_y, s_z/h_z)|;
  - rigid bodies (`nbRigidContact`): skin r·min(h). The central-difference step stays at 0.01 cell.
  - The trace hull grows by r.
- **Removed.** The old skins 0.01 (box), 0.001 (terrain), 0.001·h (field) and 0.01·h (rigid) are gone, and so is the 2-argument `nbTraceEndpoint`.
- **Measured** (with the cold fix). At step 1200 only 388 records sit below 0.25 cell height. Right/delivered was 2.90–3.03/2.82 L at step 600 and 4.39–4.49/4.22 L at 900.

### Experiment 17: accounting for the mid-run gain and the late stall (ledger)
- **R − delivered equals the solver's own rawDrift in every row.** For example, at step 600 R − D was +0.165 L and rawDrift +0.044% (0.163 L). So the gain is real, not a measurement redistribution. The probe's wall zone adds a constant +0.32 L in every run, baseline included, and the solver's drift excludes it.
- **Steps 450→750:**
  - advect −26.25 L and reconstruct +26.89 L;
  - clearAir, retire, redistance and coarsePhi together about −0.06 L;
  - net +0.58 L, the union restoring more than advection erodes.
- **Steps 900→1200** (mL):
  - inflow +1408; the right tank +192 net;
  - the coarse 4h region receives +549 from h by remap and loses −633 in 4h advection;
  - the h region's advect/reconstruct exchange is about −500, and clearAir −202.
- **Heat retirement starts with the stall.** It was 1.7k records cumulative at step 600, then 206k / 523k / 851k / 1.21M at 750 / 900 / 1050 / 1200, about 2200 a step. Particles plateau at about 215k from step 600.
- **Reading.** The calm pool is 2.2 h deep. Its tiles cool and coarsen, its samples retire, and the 4h level set erodes the film.
- **Why the thin criterion misses it.** `detailThin` measures twice the deepest phi over 4-aligned vertices. The floor vertex reads −2.2, so the film measures 4.4 h, against a threshold of 4 h, and coarsens.
- **Diagnostic, not a fix** (`--set=detailThinThickness:6`, one run):
  - the film stays at h;
  - right/delivered 6.17/4.22 L at step 900 and 6.33/5.62 L at 1200;
  - drift +0.53% at 900 and +0.19% at 1200; 325k particles.
  - Keeping the film at h removes the late sink, but exposes a gain in the h exchange, as `candhold` did in experiment 8. A coarsening-resolvability rule would therefore not converge on its own, so none landed.

## Final state (HEAD af0e5dea plus uncommitted work, 2026-10-28)
Landed, as defaults with no env switches:
- the cold-tile fix (experiment 15);
- contact r for all solids (experiment 16);
- Jacobi-averaged pair separation (Macklin 2014 §4.2), bounded by `NARROW_BAND_SEPARATION_REACH` in the residency travel;
- graded mass (experiment 9);
- the select options `sampleSeparation:pairs`, `innerReseed:cells` and `activationSeeding:uncovered`.

Removed: `NB_COLD_SOLID`, `NB_CONTACT_R`, `NB_SEP_OMEGA` and the TEMPSEP counter.

### Gate table
| Gate | Result | Verdict |
|---|---|---|
| Seep 1200, check ×3 (`fin-a1/b1/c1`) | Right/delivered at 600: 2.98–3.18/2.82 L; at 900: 4.36–4.60/4.22 L; at 1200: 4.46–4.57/5.62 L (−19%). Drift at 1200 −0.29 / −0.32 / −0.29%, peak +0.12 / +0.20 / +0.12% (step 600–900). Cd 0.816. Particles at most 249k. | **Fail:** tracking and \|drift\| ≤ 0.1%. Cd and particles pass. |
| Timing A/B, exclusive, 1200 steps (`fab-*`) | Candidate median 35.2 / 35.8 / 35.5 ms (p95 39.1–40.0). Same tree with old options: 36.2 / 36.6 / 36.8 ms (p95 40.3–42.3). | Faster than the old options on today's tree. **+1.7–2.3 ms over the morning baseline (33.2–33.7)**; not attributed between my changes and the other uncommitted work. |
| Seep A/B candidate rows | Drift at 1200 −0.31 / −0.32 / −0.27%; old options −0.43% (base1). | As above. |
| Closed rest, 120 steps (`rest-fin`) | Drift 0 at 30/60/90/120. Max speed 0.0219 / 0.0174 / 0.0150 / 0.0131 m/s, identical to experiment 12. | Pass. |
| Letters splash, 240 steps (`lv-fin`) | Phi drift −1.15 / −5.22 / −10.15 / −14.81% at 60/120/180/240; dense V −6.76% at 240. | No gain, so it passes as written. About 0.3–0.5 points more loss than experiment 13's defaults. |
| Long slide residency ×3 | Pass ×3 (resident 56 / 56 / 57 at step 169). | Pass. |
| Crowd census at 1200 (`cr4fin`) | Excess over 8 a cell 1.43 L; 780 cells over 16 (19.6k records). Base was 1.29 L / 450; cold-only 1.18 L / 478. | **Fail:** it did not collapse. |
| `npm run check:types` | Clean. | Pass. |
| `npm run test:unit` | 1253 pass, 0 fail, 132 skipped. | Pass. |
| GPU unit tests | Pass: particle-endpoint, nbflip-wall-heat, rigid-impact "overtaken", terrain-edge-contact, `probe-procedural-solid`. reconstruction-ownership fails on HEAD's fixture (`NB_ACTIVITY_THETA`, `bins`, from 1e0f90f2), not on this change. | Pass, with one failure that predates this work. |

### Open items, with evidence
1. **Late stall 900→1200** (experiment 17): the h→4h remap of a 2.2 h calm film, followed by 4h erosion, together with heat retirement. Keeping the film at h removes the sink but gains (+0.53% at 900). The h advect/reconstruct exchange has to balance before a resolvability rule can land.
2. **Mid-run gain** of +0.12 to +0.20% at steps 600–900 is over the ±0.1% gate. It comes from the union restoring more than advection erodes (experiment 17).
3. **Excess sample mass over 8 a cell** is 1.43 L at 1200, and the union cannot see it. Contact r did not collapse it.
4. **Step time** is 35.2–35.8 ms against the morning baseline of 33.2–33.7 ms. The cost of coupleFine on more tiles after the cold fix is not isolated.
5. **Test fixtures.** `tests/uniform-reconstruction-ownership.test.ts` fails at HEAD. The terrain-edge and procedural-solid fixtures lacked `umSolidFieldCutAt`/`umPointPresent` from 9ed83e12; those stubs were added.
