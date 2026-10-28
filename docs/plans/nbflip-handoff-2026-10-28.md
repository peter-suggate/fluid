# NB-FLIP seep, step time and vast-scene handoff (2026-10-28)

What landed on 2026-10-27/28, what is still open (with evidence), and the
remaining big bets, ranked by measured value. Detailed evidence is in:

- `nbflip-seep-volume-2026-10-27.md`: seep volume mechanisms, experiments 1-17, final gate table.
- `nbflip-particle-scaling-2026-10-27.md`: particle-count scaling, separation pre-pass, move register cap, symmetry bisect.
- `nbflip-step-time.md`: per-pass breakdown, the advectDeferred bound, the BULK16 pressure bound, the intercept and gap study, the occupancy audit.
- `nbflip-16h-progress.md` §4: the 16h baseline, the corrected BULK16 estimate, the depth ladder.

Constraints in force: water only (inviscid, no tension), fixed dt 1/60 s, one
step per frame, no substeps, no hacks (clamps, damping, volume injection or
speed limits), `volumeControlSeconds: 0`, step time ranked above memory,
GPU jobs only through `tools/gpu/fluid-gpu-run.sh` under 6 GiB requested.

## 1. Landed

### Seep volume mechanisms (all defaults, no switches)

| Mechanism found | Fix |
|---|---|
| Graded deep seeds carried unit mass, so the 8-a-cell bound read deep cells half full | Merge-mode graded seeds carry 2 units and initial seeds 8/count; restate, reset and the split counter keep mass |
| Activation seeded a whole band over liquid that samples already held | `activationSeeding: "uncovered"` |
| Inner reseeding pumped volume into shallow cells | `innerReseed: "cells"` (reject by cell depth as well as site depth) |
| Samples packed onto floors and walls, so the level set held about a third of the delivered liquid | `sampleSeparation: "pairs"`: Jacobi-averaged pair separation (Macklin 2014 §4.2 on Sato 2018 §3.2.2), bounded below 1 cell and counted in the residency reach. The plain sum moved samples up to 68 cells and was rejected |
| Wall-face vertices owned by a COLD tile inside the wall were never unioned (coupleFine, surfaceClear and coarseCorners skipped the whole tile), so full wall-foot cells read as air, were never projected and became sinks | The tile early-out applies only to fully open tiles; others use the per-vertex `nbVertexCold` |
| The box clamp at 0.01 left samples on the wall plane, where normal velocity is 0, so they could never leave | Contact radius r(m) = cbrt(m)/4 along the normal for every solid: box walls, voxel/page faces in the walk, procedural, terrain, rigid. The trace hull grows by r |

### Step time

| Change | Measured |
|---|---|
| Separation as a pre-pass before the move (64-record jobs, 12 B/record scratch) | Letters-uniform about 1.1M particles: steps 150-300 about −9 ms (80→70), p95 −14.6 ms, both beyond spread. Neutral on the seep. Pre-pass plus move 13.0 ms against 17-31 ms in-move |
| surfaceHeld: a whole held region per item, one binary search | Wide seep pooled median 48.6→43.2 ms (with the next row) |
| pCopyNative: 64 tiles per workgroup | (with the row above) |

Tooling: `tools/gpu-kernel-timer.ts`; `probe-tier-budget.ts` gains `--particleMass`, `--cells`, `--kernels` and `--retireSplit`;
`probe-scale-ladder.ts` gains `--kernels`, `--water`, `--stats-every` and the audit `--max-gib` cap;
`probe-reservoir-bulk.ts` gains `--tiers`, `--water`, `--kernels` and `--pipelined` (app-like overlapped timing).

### Gates on the committed tree

- `npm run check:types` clean; `npm run test:unit` 1253 pass, 0 fail, 132 skipped.
- Closed rest at step 120: drift 0, identical digits.
- Long slide residency: 3 of 3 pass.
- GPU focused tests (endpoint, wall heat, rigid impact, terrain edge, procedural solid, slab and canonical) pass.

## 2. Open issues (correctness first)

1. **Seep gate not met.** At 1200 steps the right tank holds 4.46-4.57 L of 5.62 L delivered, drift −0.29 to −0.32%. That's up from 2.08 L and −0.78 to −0.90% this morning.
   - Up to step 900 the tank tracks within about 6%, but with a real gain peaking at +0.12 to +0.20% (over the ±0.1% gate).
   - Late stall at steps 900-1200, ledger in mL: inflow +1408, right tank net +192. The 4h coarse region receives +549 by remap and loses −633 to 4h advection; the h advect/reconstruct exchange is about −500; clearAir −202.
   - Cause: `detailThin` measures 2× the floor vertex depth, so the 2.2 h film reads 4.4 h against the 4 h threshold, coarsens and is eroded by construction.
   - Keeping it at h (`detailThinThickness: 6`, one run) removes the sink but gains +0.53% at step 900. So the h exchange's net gain has to be found and balanced first, then the resolvability criterion fixed.
2. **Hidden crowd mass.** 1.43 L of sample mass sits above 8 a cell at step 1200 (780 cells over 16), invisible to the union. These are floor-jet cells (liquid, depth −0.6 to −0.95, 1.6-2.9 m/s).
   - The deep-cell merge does not drain them.
   - Find why liquid cells accumulate samples when the grid field is divergence-free: measure the per-cell divergence of the sampled field and the merge rate.
3. **Rest drift grows with depth** (from the depth ladder; shared 4h path, identical with `bulkTier: "off"`).
   - Step-120 drift: 0 at 1×, −9.0e-7 at 1.5×, −2.28e-5 at 2×, −9.01e-5 at 2.5×, −1.70e-4 at 2.75×.
   - All of it is lost by step 60, then flat.
   - This will matter for vast scenes.
4. **z-mirror symmetry test fails on HEAD itself** (`tests/uniform-nbflip-symmetry.test.ts`, 2.026e-4 m at step 6).
   - Bisect: 24d05d06 breaks it at step 21; 049221ca (its 4h corner curve term) exposes it at step 5.
   - The first asymmetry is the velocity extension into air at step 5, at h tiles next to the z walls (u 0.078 against 0.215 at the mirror).
   - Next suspect: `umNeighbor` in `uniform-mixed-extension.ts`, where the low-side `max(below, wall)` clamp and `lowWidth`/`highWidth` look one-sided.
   - Repro: `node --import tsx tools/probe-nbflip-symmetry.ts --steps=6 --runs=1 --capture-extended=true`.
5. **Seep step time** is 35.2-35.8 ms median at 1200 steps, against the morning's 33.2-33.7 ms. The old options on the same tree run 36.2-36.8 ms, so the new defaults are faster than the old ones. The rest of the gap isn't split between the cold-tile fix (more coupleFine tiles), separation and other work.
6. **Letters splash** loses 0.3-0.5 points more phi volume than before (−14.8% at 240 steps without volume control). There's no gain. The benchmark's own `volumeDrift` read is stale (a single end read) and `--dt=0.0166667` trips its one-clock-step assert; `scratchpad/probe-letters-volume.mts` was used instead.
7. `tests/uniform-reconstruction-ownership.test.ts` fixture lacks `NB_ACTIVITY_THETA`/`bins` since 1e0f90f2 (it fails on HEAD).

## 3. Big bets, ranked by measured value

### A. BULK16, the full design (vast scenes, required)

- Depth ladder (closed rest, pipelined timing): step = 10.5 ms + **25 µs per whole-16h page**. That cost comes from about 20 per-block 4h passes, none above about 4 µs/page:
  - project 4.2, V-cycle 2.6, localSpeed 2.6, rhs 2.1, census 2.1;
  - advectWalls, pageMark, frame plan and surface volume about 1.6 each.
- So it only pays as the whole design that takes whole pages out of every sweep. A pressure-only slice was measured at about 3 µs/page and is not worth building.
- At 100k bulk pages: about 2.5 s/step and about 159 GiB at 4h. BULK16's state is 2-4 KiB/page, so about 0.2-0.4 GiB.
- Wide seep upper bound: about 7.5 ms/step. That replaces the earlier 12.7 ms claim, which came from timer-on serialised costs.
- The reader audit (seven choke points, about 25 neighbour sites, the lifecycle) is in `nbflip-16h-progress.md` §4.

### B. Memory per pool block

Construction-time live memory grows **14.9 MiB per pool block**, not 1.6 MiB. NB-FLIP buffers sized per pool block account for most of it:
- particles A and B, 1.5 MiB each;
- cell starts and samples, 1.6;
- counts and metrics, 0.9;
- stage scratch, 2.0;
- arena pools, 1.7.

Bulk pages hold no particles, so sizing particle storage by the band, not the pool, is the memory lever that capped the ladder at 2.5× depth under 6 GiB.

### C. Fine-band cost (today's scenes)

Letters fit: GPU = 13.8 ms + 13.7 µs/1k particles + **5.9 µs per fine tile**. The per-tile cost is spread over about 200 kernels, each at most 0.34 µs/tile:
- by group: pressure 0.73, surface metric/publication 0.65, surface advect 0.34, transfer 0.32, census 0.30, phi resolve 0.30.

Levers are fewer fine tiles (band thickness, coarsening rules; this ties to open issue 1) or fusing per-tile sweeps.

### D. Register-capped kernels (12.5% occupancy)

- 42 of 453 pipelines are capped at 384 threads.
- On the wide seep these are latency-bound with every limiter at 36% or less: advectDeferred 4.05 ms, project 2.02, coupleFine 1.27, cacheNormals 1.22, classify 1.19, arenaPublishCanonical 1.13.
- In advectDeferred and coupleFine the cap is the solid-aware path as a whole: stubbing `umSolidPresent` lifts 384→1024.
- The only fix is routing open-water tiles to the existing `umSolidFree` twins per tile. That needs per-tile twins, which e4041e9a removed for repeatability, and a correct ±12 cell certificate measured +1.1 ms. The no-solid best case was −3.4 ms at 3000 h tiles. Revisit only if the twin policy changes.
- The move kernel is capped at 384 by its RK3 sampler (the walk, rigid contact and particleDepthIn too); there's no non-twin split.

### Measured and closed (don't repeat)

| Idea | Result |
|---|---|
| advectDeferred fast-path certificate | +1.1 ms; best case −3.4 ms |
| Dispatch gaps and pass merging | Gaps 0.3 ms; Dawn already packs about 450 passes into about 9 command buffers |
| Indirect dispatch | About 12 µs each on this stack, against 4.5 µs for an early-exit direct dispatch |
| CPU encode | 3.1/4.2 ms, overlapped by the app's 2-frame receipt ring |
| Probe serial wait | About 2 ms/step at rest that the app doesn't pay. Use `--pipelined` for app-like timing |
| Device-wide radix sort | Sort is 1.9 ms at 1.14M, near its 0.65 ms floor; at most 0.3 ms to gain |
| Dropping canonical within-cell order | Faster sort, but cells pile to 828 and the move gets 35-60% slower; symmetry depends on it |
| Move chunk 256→64 | Within spread on both scenes; reverted |
| Inviscid, no-tension water in the realtime scenes | Measured −3.3 to −4.3 ms on the wide seep (`--water=inviscid`). Not yet authored into the scene presets |

## 4. Suggested order

1. Open issue 1 (the h exchange gain, then the thin-film resolvability criterion), with issue 2 alongside. They decide the seep gate.
2. Open issue 3 (rest drift with depth), before any vast-scene work.
3. Author inviscid, no-tension water into the realtime presets (−3 to −4 ms).
4. BULK16, the full design, with band-sized particle storage (bet B) in the same pass.
5. Symmetry (issue 4) whenever `uniform-mixed-extension.ts` is next touched.

## 5. Implementation pass (2026-10-28, incomplete)

This pass does **not** close the seep gate or implement BULK16. The numerical
prerequisites in §4 remain open; no thin-thickness increase or volume
correction was introduced.

Implemented:

- Corrected a remaining graded-mass mismatch in P2G. Under `crowdedCells:
  "merge"`, four deep seeds hold two mass units each, but the expected face
  mass still fell to four below depth −1. All three transfer paths now expect
  eight physical mass units there, as uniform seeding does. Legacy unit-mass
  `retire` mode retains its four-unit expectation. This changes the blend of
  partially covered deep faces, without changing particle mass or positions.
- Authored inviscid, zero-tension water into the shared NB-FLIP scene factory,
  reservoir seep, churning tank, backyard pool (including drift), and forest
  waterfall factories. Existing physical scales and schedules are retained.
- Repaired the reconstruction-ownership fixture for the shared heat arrays
  and the current surface-function signatures. Its GPU comparison passes
  all three mixed-layout patterns.
- Repaired letters benchmark time-step selection and volume telemetry. It
  defaults to 60 Hz and zero volume control, explicitly selects the scene
  clock, requests phi measurement before each reporting step and the final
  step, and reports those measurements per row. `--volume-every` defaults to
  60. Explicit time-step overrides must agree with `--dt`.
- Reservoir and tier probes also request volume measurement **before** the
  reported step. Previously their `volume.drift` could belong to the step
  after the previous report. The tier probe's independent `liquid_m3.total`
  measures the current surface. `rawVolumeDrift` / `volume.rawDrift` instead
  measure the pressure fill; they are not independent phi measurements.
- Added live and construction allocation caps (default/maximum 6 GiB) to
  letters, reservoir and tier probes, using the existing allocation audit.
  The tier probe accepts `--tiers` and `--water=inviscid`; `--cells` now counts
  physical mass over eight per cell as well as record populations. Its
  `--retire` counters continue to report merges, refused merges and splits.

Validation, all GPU jobs through `tools/gpu/fluid-gpu-run.sh check`:

- Type check clean; unit suite 1253 pass, 0 fail, 133 GPU-skipped.
- GPU regression for graded transfer mass: pass. Four mass units on a deep
  face give half coverage in merge mode; graded and uniform mass thresholds
  agree. Reconstruction ownership: pass.
- Closed rest, 120 steps, 5 mm, `tiers={"start":"4h","bricks":512}`:
  drift 0 at 30/60/90/120; endpoint maximum speed 0.01720 m/s. This is the
  newly authored inviscid scene, so its speed is not the earlier viscous
  baseline's identical-digit comparison.
- Letters, 240 steps at exactly 1/60 s: current phi drift at 60/120/180/240
  is −1.177 / −5.077 / −9.390 / −14.064%. Peak requested allocation 3.005
  GiB. No gain, but substantial absolute loss remains. These are correctness
  runs, not exclusive timing evidence.
- Seep with corrected transfer mass, inviscid, 1200 steps, same tier budget:
  right tank 3.415 L at 600, 4.622 L at 900 and 4.612 L at 1200 (about 5.62 L
  delivered). The independent spatial probe, relative to its initial total,
  peaks at +0.2013% in the sampled rows and ends at −0.2736%: **gate still
  fails**. Construction peak 2.047
  GiB. The run predates the reporting-step telemetry repair; use its
  `liquid_m3.total` relative to the initial total, not its stale
  `volume.drift` or the pressure-fill `volume.rawDrift`.
- Reporting fixes: a 150-step tier-probe run completed with physical-mass
  census output, and the letters benchmark completed two exact requested
  steps with `--dt=0.0166667` (the previously failing decimal spelling).
- The same inviscid configuration before the mass fix reached +0.2575% at
  600 and stopped at step 1079: a transient build needed 4142 fine tiles,
  beyond the 4096-tile reservation. It is not a completed A/B gate.
- Symmetry failure reproduced at step 6: reflected surface error
  2.026297e-4 m. Restricting the direct neighbor fast path to unit requests,
  disabling the cached fine requests, and admitting one-ulp Godunov cutoff
  roundoff each left that error unchanged. These experiments were reverted;
  no speculative extension change remains.

Local evidence is under `artifacts/nbflip-handoff/` (gitignored):
`handoff-mass*`, `rest.json`, `nbflip-letters-handoff.json`, and
`symmetry-before/report.json`. The seep invocation was:

```sh
tools/gpu/fluid-gpu-run.sh check env \
  WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  node --import tsx tools/probe-tier-budget.ts handoff-mass \
  --steps=1200 --every=150 --water=inviscid \
  --tiers=artifacts/nbflip-handoff/tiers.json \
  --set=volumeControlSeconds:0 --particleMass --cells --retire \
  --out=artifacts/nbflip-handoff
```

Still required: balance the h exchange, measure sampled-field divergence in
crowded cells, fix thin-film resolvability, resolve depth-dependent rest
loss, implement the full BULK16 lifecycle and band-sized storage across its
readers, and fix symmetry. The existing optional `fluid.tiers` particle
budget is usable by the probes; it is not a new BULK16 implementation.

## 6. Continuation checkpoint: seep gate and deep rest

The seep gate now passes three independent 1200-step runs at exactly 1/60 s,
with inviscid water, zero tension and `volumeControlSeconds: 0`. Every step
requests and checks a fresh phi measurement against the ±0.1% gate:

| Run | Worst drift (step) | Receiving phi / particle mass at 1200 | Throat Cd |
|---|---|---|---|
| `forward-material-arrival` | +0.057253% (811) | 5.459 / 5.496 L | 0.81603 |
| `forward-material-arrival-r2` | −0.050277% (1200) | 5.420 / 5.417 L | 0.81582 |
| `forward-material-arrival-r3` | +0.074792% (857) | 5.528 / 5.464 L | 0.81639 |

These are correctness runs, not timing evidence. The third run ends with
339,053 particles; the three endpoint counts span 334,308–339,053.
Construction peak is 2,764,034,484 requested bytes. Two seep checks fit below
6 GiB together. Artifacts and the numerical-source checkpoint are in
`artifacts/nbflip-handoff/` (gitignored).

The mechanisms addressed are:

- Physical mass is cached separately from record count. Mass bounds and
  reseeding use the former; sorting and work enumeration use the latter.
- Particle trajectories integrate the normal-affine face field within each
  cell, with exact face-crossing events over the one frozen fluid step.
  The projected liquid MAC divergence is therefore also the trajectory
  field's divergence. Stable log1p/expm1 evaluations avoid a roundoff-induced
  missed crossing at large lattice coordinates.
- The liquid projection alone does not constrain the extrapolated air
  field: measured near-surface air divergence was −8 to −14/s. An auxiliary
  extension potential adjusts only air–air faces, preserving liquid and
  solid boundary fluxes. It greatly reduces the receiving-film piles.
- Material transfer snapshots the old Eulerian/particle ownership and
  liquid metric. Quarter-cell material sites owned by Eulerian liquid are
  transported forward and handed to the active particle band. Existing
  particle-owned liquid is not seeded again. A known-liquid material site
  must be allowed to arrive where the advected grid says air: rejecting it
  there removed genuine transported material. The rejected arrival-clipped
  trial failed at −0.1611%.
- Reconstruction applies the physical-mass bound to erasure as well as
  addition, under the existing seeded-footprint and heat checks. Merging
  preserves the vertices used by the reconstruction band.
- Thin-film resolvability uses one-sided depth against a solid wall and
  two-sided depth in open water. A floor film no longer measures twice its
  actual thickness and prematurely coarsens.

The rest drift was a startup pressure transient. The pressure-dependent
seam heads now receive 23 Picard re-solves at t=0 (24 solves total), without
advancing positions or the clock. In the `bulkTier: "off"` ladder, **all 120
per-step measurements are zero drift** at heights 1×, 1.5×, 2×, 2.5× and
2.75×. Maximum speeds over those runs are respectively 1.09e−5, 1.90e−5,
1.93e−5, 1.93e−5 and 4.03e−5 m/s. This adds startup work; it adds no re-solves
to ordinary frames.

The remaining work is still active: regression checks, wall symmetry,
the complete BULK16 lifecycle and consumers, band-sized storage, and
exclusive performance measurements. The three seep passes above precede
the subsequent wall-contact interpolation correction and must be repeated
if that correction changes this scene's transport.
