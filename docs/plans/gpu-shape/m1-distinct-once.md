# GPU-shape M1: each distinct brick of a stencil looked up once (lanes A, A2, 2026-11-24)

Plan: docs/plans/gpu-shape-2026-11-23.md (M1). Tools: tools/experiments/gpu-shape/m1/ (m1fix.mts frozen-state A/B,
variants.mts text variants, dump.mts). Run logs: ~/code/me/fluid-scratch/gpu-shape/m1a2/ (round*.log, per-run logs and
jsonl; gitignored here).

## Hypothesis

m0e: the brick kernels are control-bound. A tap's slot is a directory walk (fgBrickSlot: page, then child, 2 dependent
loads), and the tap branches on the walk (NB_NONE), so each tap's loads wait on the previous tap's branch: lAdvect's
departure cubic is 64 serial walks of at most 8 bricks. A stencil over cells lo..hi (hi - lo < 8 an axis) lies in
bricks lo >> 3 and hi >> 3 an axis. Walk each distinct brick once, independently, then read every tap with its slot
picked by `select`: the same words, no walk and no branch a tap.

## Method

m1fix.mts: warm the scene up (census settings: brick engine, fixed dt 1/60, presentation off, moored bodies on
paddleboats), save the engine state at STEPS (100; seep 300, where phiSolid and pSolid have work), then per variant of
the kernel's compiled text:
- exactness: restore, run the base placements, read the written buffers; restore, run the variant's; compare word for
  word (base against base first). pMove: PRE=1 (the step's placements before pMove run before the save, so pMove
  replays on its own inputs; replaying it off a finished step latches FAIL.REACH) and fpart not compared (atomic rank
  order is nondeterministic, 900-36000 words differ base against base).
- timing: R batched copies (12; seep 40) untimed then timed, base and variants interleaved, 5 samples, median; the
  clock probe (clock-probe.ts) after every sample. A sample whose probe is far above 75.5 µs is at low DVFS clock; the
  runs below whose probes sat at 100-250 µs throughout are marked invalid and were rerun.
- REF=file: a variant returning HEAD's dumped text, for a built candidate tree (TREE=) against HEAD in one process.
- caps: Metal's thread cap of each text (fluid-scratch/brick-tune/regcap.sh).

All runs `tools/gpu/fluid-gpu-run.sh timing`, rc 0, no GPU errors; fail/info words 0 at the frozen state and after the
base placements in every PRE run. HEAD 734dd9e2 (c1d8c24e has the same lib/).

## Results (frozen state, kernel ms median, change vs base in the same process)

lAdvect (cap 384 for every variant):

| variant | dam | paddleboats | seep (step 300) | exactness |
|---|---|---|---|---|
| cubic (64-tap cubic only, all 8 bricks walked) | | −12.5% (2.07 → 1.82) | | exact |
| cubic-distinct | −12.0% (lane A) | −11.8% | | exact |
| cubic+grad-distinct | −13.7% (1.64 → 1.42), −14.2% | −13.4% (2.07 → 1.80), −14.3% | +3.3% (0.787 → 0.813), +5.6% | exact |
| cubic+grad+heat-distinct | −12.5% | −12.8% | | exact |
| all-distinct (+ lCorners0) | −14% (lane A) | −19.5% | | 1 ulp, 32-66k φ words |
| cg + two trilinear copies (tri2) | −16.4% | | | 1 ulp, 32k φ words |
| cg + per-corner select (cornsel) | −11.6% | | | 1 ulp, 32k φ words |
| cached (one copy, lPhi0's per-tap rule with ALU slot) | +7.1 to +8.3% | −7.9 to −9.3% | +16.7 to +20.3% | 1 ulp, 13-66k φ words |

phiSolid (seep step 300, cap 384): cubic-distinct +2.2%, cubic+grad-distinct +5.2% then −8.1% (two runs), cg+heat+buried
+3.6%, cached +21.6 to +22.8% (1 ulp); exact except corners and cached (264-350 φ words, 1 ulp). Step-100 phiSolid
timings ran at low clock (probe 100-150 µs): invalid.

pMove (PRE=1, buffers set, fcell, fmoved; caps base 512, depth 576, corners 512, depth+corners 512):

| variant | dam | paddleboats | exactness |
|---|---|---|---|
| corners (fgCornersOf of NB_PHI_F_PHI, 8 slot-hit words) | −7.7%, −9.0% (1.73 → 1.59) | −8.9%, −9.2% | exact |
| depth (fgDepth, two loop copies) | −12.7% (1.73 → 1.52) | −13.0% | 1 ulp, 141k / 73k fmoved words |
| depthc (fgDepth one loop, fgDepthVertex's rule with the ALU slot) | **−15.8% (1.68 → 1.42)** | −7.3% (noisy; min −7.2%) | **exact** |
| depthc+corners | −11.9% | −10.3% | exact |

PRE=0 pMove runs at step 100 (round 1) ran at low clock (probe 248 µs): timings invalid, exactness as above.
seep (step 300) pMove: depthc, corners and depthc+corners exact; timings at low clock (probe 125-245 µs), invalid.

Landed source, built (TREE = HEAD + patch) against HEAD's text in one process (REF, or `depth-head`: the landed
fgDepth turned back into HEAD's text, verified to round-trip HEAD's dump), every written buffer bit-identical:

| kernel × scene | HEAD ms | landed ms | change | cap |
|---|---|---|---|---|
| lAdvect dam | 1.805 | 1.534 | −15.0% | 384 / 384 |
| lAdvect paddleboats | 2.155 | 1.814 | −15.8% | 384 / 384 |
| lAdvect seep (300) | 0.762 | 0.787 | +3.3% | 384 / 384 |
| phiSolid seep (300) | 0.545 | 0.511 | −6.2% | 384 / 384 |
| pMove dam | 1.691 | 1.439 | −14.9% | 512 / 576 |
| pMove paddleboats | 0.834 (min 0.820) | 0.911 (min 0.772) | −5.8% by min (bimodal samples) | |
| pMove seep (300) | 0.352 | 0.310 | −11.9% | |
| uTransfer dam (C page fgDepth) | 1.627 | 1.586 | −2.5% | |
| uTransfer paddleboats | 1.687 | 1.526 | −9.5% (cbb5f5b5's message says −8.3%) | |
| pSolid seep (300) | 0.884 (min 0.873) | 0.893 (min 0.829) | −5% by min, +1% by median | |

Clock probes 75.9-77.2 µs in these rows except single samples (81-409 µs).

## Findings

- The lever holds where the stencil is in the band's interior (dam, paddleboats: −12 to −16% on lAdvect and pMove)
  and is flat to slightly negative on seep, where the lStencil walk misses more often and the slow arm runs as well
  (inference: not measured per lane).
- 1 ulp: any change of control flow around lCorners0's trilinear or fgDepth's weighted sum that duplicates the
  arithmetic changes φ / depth by 1 ulp in some vertices although the loaded words are identical (corners-verify:
  the 8 corner words agree). Inference: Metal fast-math contraction differs between the copies; restoring the summation
  order in the source does not reach it. A single copy whose per-tap rule is unchanged (depthc) stays exact.
- One copy with a per-tap branch on a select-picked slot (cached) is exact for depth but 1 ulp and slower for lAdvect's
  cubic: the two-copy form (fast cubic behind one stencil test) is what wins there.
- Source form matters as much as the variant: a first lAdvect source (lStencil with an early return, the cubic called
  through a wrapper function) was exact but +4.4% on dam and +25% on seep; the landed form mirrors the measured
  variant's text and timed −15.0% dam, −15.8% paddleboats, +3.3% seep (lAdvect), −6.2% seep (phiSolid).

## Landings

Each landing: HEAD + patch in a full source archive passed `tsc --noEmit` (only the other session's lane-n tools
fail, missing modules) and `tools/run-feature-tests.ts` (2019 pass, 0 fail, 564 skipped; the lAdvect landing updates
brick-solid-walk.test.ts's expected phiSolid cubic text). The main checkout's own check:types fails on other sessions'
untracked files (lane-n, m4.mts), not on these.

1. **cbb5f5b5 fgDepth** (fluid-grid.wgsl.ts, particles-kernels.wgsl.ts): fdBricks / fdVertex; psDepth from the same
   generator, text unchanged (checked).
2. **e8a33481 lAdvect** (levelset-kernels.wgsl.ts, levelset-formulas.wgsl.ts, brick-solid-walk.wgsl.ts + test):
   lStencil / lStencilPhi, lStencilCubic from nbfCubicLimitedWGSL (nbfPhiAdvectWGSL's text unchanged, checked);
   the cubic and the gradient only, lCorners0 untouched.

Timing rounds (`tools/nbflip-timing-round.sh`, base = export of 734dd9e2, lib/ equal to c12d19cd; candidate = export
+ patch; all rc 0 except paddleboats, 0 Metal aborts). Step ms, base → candidate:

| round | scene | fenced median | fenced mean | pipelined |
|---|---|---|---|---|
| fgDepth | seep | 27.27 → 26.15 (−4.1%) | 26.49 → 25.70 (−3.0%) | 25.02 → 25.86 (+3.4%) |
| fgDepth | dam | 33.80 → 34.27 (+1.4%) | 38.20 → 37.98 (−0.6%) | 35.86 → 35.95 (+0.3%) |
| fgDepth | letters | 54.73 → 53.94 (−1.5%) | 47.28 → 46.54 (−1.6%) | 45.68 → 44.54 (−2.5%) |
| lAdvect 1 | seep | 26.32 → 26.54 (+0.8%) | 25.65 → 26.02 (+1.5%) | 24.83 → 25.56 (+3.0%) |
| lAdvect 1 | dam | 33.91 → 34.37 (+1.3%) | 38.08 → 38.07 (0.0%) | 35.97 → 36.44 (+1.3%) |
| lAdvect 1 | letters | 54.36 → 54.84 (+0.9%) | 47.74 → 47.45 (−0.6%) | 45.23 → 44.99 (−0.5%) |
| lAdvect 2 | seep | 29.66 → 28.15 (−5.1%) | 29.19 → 28.11 (−3.7%) | 25.18 → 25.79 (+2.4%) |
| lAdvect 2 | dam | 33.45 → 33.95 (+1.5%) | 37.96 → 37.39 (−1.5%) | 36.51 → 35.85 (−1.8%) |

Drift at the end (fenced / pipelined): seep base −0.60/−0.59, −0.63/−0.68, −0.46/−0.55%; candidates −0.59/−0.54
(lAdvect 1), −0.57/−0.68 (fgDepth), −0.79/−0.71% (lAdvect 2). letters base +0.36/+0.57, +0.42/+0.43%; candidates
+0.68/+0.54 (lAdvect 1), +0.52/+0.63% (fgDepth). The same tree differs by up to 0.2% between its own runs (atomic
order), and both changes are bit-identical at the frozen state.

Reading: the expected whole-step gains (dam ≈ 0.27 ms lAdvect + 0.25 ms pMove, under 1% each) are below the round's
resolution; base and base-like rounds swing by ±3-5% (seep fenced median 26.3 / 27.3 / 29.7 ms for the same base
tree). Seep pipelined came out +2.4 to +3.4% for every candidate in all three rounds, the fgDepth one included,
whose seep kernels are all faster at the frozen state; inference: an order or thermal bias of the round (the
candidate always runs second), not either change. Not verified.

paddleboats: `nbflip-gate-bench.ts` fails at step 0 for base and candidate alike ("Brick box observer: the box
512x32x192 is not the engine's box 512x64x192", tools/nbflip-observe-brick.ts): a tool defect, not these changes.
Its numbers above are the frozen-state ones.

## Not done

- lCorners0 (lAdvect, phiSolid) and pMove's fgCornersOf: the corners win (−4 to −9%) is either 1 ulp (lAdvect) or
  smaller than depth alone and lives in the shared FG_PHI_WGSL (pMove, exact); not landed.
- nbOperator and pSeed: not looked at (pSeed already stages its depth vertices in workgroup memory).

## Gates for small exact wins (lane A2, 2026-11-24, second task)

The timing round cannot resolve a change under ~1% (above). Three tools changed, commit 67c09e96 and the stepab
commit:

- **Paddleboats observer** (tools/nbflip-observe-brick.ts). The engine's box under an open top is the lattice's
  y: the container plus `fluid.brickTiers.openTopHeadroom` (brick-scene.ts; a scene setting since 9265994e). The
  observer compared that with the container (sceneLatticeDimensions): 512x64x192 vs 512x32x192, and failed at step
  0 for every tree. Now x and z must match, and y must hold the container under the headroom (open top only). The
  bins are x/z only, so liquid splashed over the rim is counted, as the engine's box counts it.
- **Balanced timing round** (tools/nbflip-timing-round.sh, nbflip-gates.ts `--order`). Per scene, fenced and
  pipelined run their base/candidate pair in opposite orders, and the scenes alternate which order comes first:
  scene 1 runs fenced base then candidate, and pipelined candidate then base; scene 2 the reverse. OUT/order.log
  records the order. The table adds each pair's row with its order and the **order-balanced change** (the mean of
  the two pairs' changes). No extra runs: four a scene, as before.
- **Frozen whole-step A/B** (tools/experiments/gpu-shape/stepab.mts). The landing gate for bit-identical changes.
  It builds the engine from BASE and warms it to step 100 (census settings, as m1fix). A child process builds
  CAND's engine and dumps its schedule. Placement for placement, the two schedules must agree in everything but
  the WGSL (label, entry, bindings, dispatch, z/w); otherwise it fails explicitly, so host-side changes need the
  round.
  - Exactness: per changed placement, it snapshots the state just before it (S0, then base placements 0..i-1). From
    that snapshot it runs base, candidate and base again, then compares every read-write binding word for word.
    Where base and base again differ (atomic order: pSeed's claimed particle slots, pMove's fpart ranks), the buffer
    is compared as a multiset of records instead: NB_PSTRIDE words a particle in `parts`, one word elsewhere, by two
    order-free hash sums. It passes "identical up to order".
  - Timing: the whole step from S0, with base, candidate and an A/A set (base again), rotated each sample. S0 is
    held on the GPU (a mirror of every buffer, copied back inside the submit before each replay). A sample is R=4
    replays untimed, then R timed with gpu-kernel-timer, then R GPU walls. S=9 samples. A sample is dropped when the
    clock probe reads more than 1.3x the top clock's. The figures are per-family minima.
  - The gate figure is the **changed families' Σ Δ**. Unchanged families run the same text, and their Δ, like
    every A/A, is the floor.
  - First version, superseded: a host restore (GBs uploaded) before each single replay. Per-kernel times spread
    1-3x, and the whole-step A/A was +1.9 ms on dam.

stepab, cbb5f5b5 + e8a33481 (export of 7e6a6ac8) vs c12d19cd (lib of 734dd9e2), step 100, one run a scene.
Measured; ms are per-family minima (Σ placements is serialised, one pass a dispatch):

| scene | changed placements: exact | changed families Σ Δ (A/A) | unchanged Σ Δ (A/A) | GPU wall min (A/A) |
|---|---|---|---|---|
| seep | 8 of 8 (fpart, parts up to order) | −0.037 ms, −0.19% of 19.57 (−0.032) | +0.126 (+0.015) | +0.004 ms, +0.02% (−0.200) |
| dam | 6 of 6 | **−0.466 ms, −1.49%** of 31.22 (+0.042) | +0.067 (+0.025) | −0.589 ms, −1.88% (−0.016) |
| letters | 6 of 6 | **−1.128 ms, −2.48%** of 45.55 (−0.010) | +0.033 (−0.061) | −1.452 ms, −3.18% (−0.539) |
| paddleboats | 6 of 6 | **−0.554 ms, −1.60%** of 34.64 (+0.028) | +0.024 (−0.022) | −0.564 ms, −1.63% (+0.032) |

Per family (base → candidate ms):
- dam: lAdvect −10.7%, pMove −12.9%, pSeparate −8.0%, pSeed −0.8%, uTransfer +0.1%.
- letters: pMove −16.6%, uTransfer −10.2%, lAdvect −13.3%, pSeparate −11.8%, pSeed +2.2%.
- paddleboats: lAdvect −11.4%, uTransfer −9.1%, pMove −13.5%, pSeparate −10.6%, pSeed +0.9%.
- seep: uTransfer −10.1%, phiSolid **+9.1%** (0.882 → 0.962, A/A −0.010), pMove −9.4%, pSeparate −5.2%, lAdvect
  +1.2%, pSeed +1.6%, pSolid −3.8%.

Reading:
- The changed-family A/A is ≤ 0.04 ms on every scene, so the tool resolves about 0.1% of a step.
- All four scenes are exact.
- Three scenes gain 1.5-2.5%. Seep is flat: in the step, phiSolid loses what uTransfer and pMove gain. The
  isolated frozen-state A/B (m1fix, R batched copies of phiSolid alone) had phiSolid −6.2% on seep. Inference:
  phiSolid's in-step cost depends on what runs before it (its cache state), and only the whole step sees that.
  Not verified. A later lane could revert the phiSolid half of e8a33481 on seep's evidence; the lAdvect half
  holds on every scene.
- pSeparate and pSeed changed through fgDepth (cbb5f5b5) and were not timed before.

Balanced round, paddleboats only. The base is 67c09e96 with lib reverted to before cbb5f5b5 + e8a33481
(lib = HEAD minus the two landings, lane B2's M4 commits included); the candidate is 67c09e96. Both trees carry the
fixed observer. Measured:
- All four 300-step runs completed, rc 0, no Metal aborts. The observer fix holds.
- Fenced (base first): 36.41 → 36.27 ms median (−0.4%).
- Pipelined (candidate first): 34.53 → 34.56 ms/step (+0.1%).
- Order-balanced change: −0.2%. stepab's −1.6% is below this round's resolution, as expected.

The round's final table step had a zsh quoting slip (`${CAND:+a b}` is one word); it was fixed before the stepab
commit, and the table above comes from rerunning nbflip-gates.ts on the round's runs.

check:types (`tsc --noEmit`):
- At commit 7e6a6ac8 (a clean archive) it fails on lane-n's files only:
  - tools/experiments/lane-n/dump-msl.mts (6,46) and (7,43);
  - tools/experiments/lane-n/features.mts (3,46);
  - each imports from fluid-scratch/lane-n/tree/lib/..., which no longer exists.
- In the main working tree at 09:2x it passes. The uncommitted lane-n edits fix it there.
- The untracked tools/experiments/gpu-shape/m4.mts error seen earlier (388,103: `unknown[]` vs `GPUBuffer[]`) no
  longer reproduces. m4.mts is lane B2's (it ran as `m4.mts paddleboats` in B2's process tree).

## phiSolid revert, nbOperator and pSeed (lane A2, 2026-11-24, third task)

Every row below comes from stepab.mts (step 100, S=9, written buffers compared per changed placement). The Δ is
in-step (the family's min over samples) with the run's A/A.

**phiSolid (a89eff60, landed).** phiSolid runs on seep only: dam, letters and paddleboats schedule no phiSolid (no
static solid), so they are neutral by construction. On seep, reverting phiSolid's text to its pre-e8a33481 form was
exact in every run:

| candidate | phiSolid in step |
|---|---|
| both halves reverted | 0.932 → 0.841 ms (−9.8%) |
| landed text, run 1 | −2.3% |
| landed text, run 2 (S=18) | −9.6% |
| landed text, run 3 (S=18) | −9.4% |
| foot taps only | −2.4% |
| cubic only | +6.5% |

- The landed text is "both halves" plus two comment lines.
- Whole step: −0.48% (GPU wall −0.44%).
- The halves are not additive. All four texts cap at 384 threads (regcap), so the cap is not the cause.
- Inference, not verified: Metal's codegen of the combined text, or the lStencil misses.

**nbOperator.** Ablations on the frozen state (m1fix, min ms):

| ablation | paddleboats | dam |
|---|---|---|
| base | 2.431 | 0.933 |
| stage run twice (stage ≈ 0.49 / 0.25) | +20% | +27% |
| rows skipped | −70% | — |
| face loop skipped (faces ≈ 1.12 / 0.2) | −46% | −21% |
| starts skipped | −1.6% | −0.8% |

Variants tried:
- **frac, landed 9c82f1df.** With cuts, a row's own FRAC is read once a row (openC → poBandFaceO), not once a face
  plus twice more. Exact. Operator: paddleboats −3.3%, seep −6.4%. dam and letters have no cuts, so their text is
  unchanged. Cap 512 = 512.
- **pre, not landed.** The face plane velocities (and W) are loaded ahead of the loop. Exact, but paddleboats +8.5%
  and dam +3.0% on the frozen state.
- **roots, not landed.** poTiles' roots are always staged. Exact; dam −2.4% (noisy), paddleboats +2.4%.
- **page, not landed.** The row's own page word is read once a lane for poAcross. Exact, but mixed in step:

  | scene | operator in step |
  |---|---|
  | dam | +1.2% |
  | seep | +1.4% |
  | paddleboats | −1.2% |
  | letters | −0.6% |

  Paddleboats' cap fell 512 → 448.

**pSeed (landed 9b8f98db).** A cell's centre and its sites share one floor cell. psCellGate loads that cell's reach
bit and 8 staged depth vertices once (private), and the count pass's sites read them (psDepthCell).
- "depth" also cached the write pass. z0 parts DIFFERS as a multiset on all four scenes, though set (counts) stayed
  identical.
- The generated MSL carries `#pragma METAL fp math_mode(relaxed)`, so the same sum in another context may round
  differently.
- "depth-w" took the write pass's depth through `select(psDepth, fgPhiAt)`, which evaluates both. It was exact but
  lost the win (letters +2.4%), because the write pass then reads fgPhiAt for every site.
- "depth-x" (landed) keeps psSiteDistance in the write pass. Exact on all four scenes, z1 and z0, pSeed in step:

  | scene | pSeed in step |
  |---|---|
  | letters | 2.415 → 2.190 ms (−9.3%) |
  | dam | 1.256 → 1.134 ms (−9.7%) |
  | paddleboats | 0.646 → 0.599 ms (−7.3%) |
  | seep | 0.421 → 0.382 ms (−9.2%) |

- Cap 576 → 512.
- The landed lib text, dumped, equals the gated candidate's pSeed text, apart from comments and one ABI constant that
  lane B2 added meanwhile.
- Caveat: exactness here is measured, not by construction. Under relaxed math the count pass's cached site depths
  could differ from psDepth's in the last bit. That would flip a seed decision only at a threshold, and none
  flipped on the four gate states.

## The ulp tier, its re-gates, and the uExtend reshape (lane A2, 2026-11-24, fourth task)

**The tier (82edb8b1).** stepab now takes `TOL=ulp4` or `--tol=ulp4`.
- f32 words may differ by ≤ 4 ulp; the tool reports the count and the max.
- Every other word stays exact.
- Which words count as f32:
  - a binding's declared `array<f32>`: vel, pres, pmgF, pmgRed, xfer, fext4, pEnergy;
  - the phi pool's PHI and PHI1 channels, from the layout's own bases (FLAGS exact; a new channel fails);
  - particle records (parts, fmoved), every word but before.w, which packs mass, mode and age.
- pmgC stays exact: its PDELTA words hold u32 patterns as bitcast f32.
- Order-dependent particle buffers: bit-identical records cancel, and the rest pair by exact words and by floats
  with their low 8 bits cleared.

**Re-gates on the tier (HEAD 82edb8b1, step 100).** None landed.

| variant | where | result |
|---|---|---|
| pSeed write-pass caching ("depth") | dam | fails: 2444 seeded records differ in word 7 (velocity.w, the site depth); max 59 ulp, 15 over 4 |
| ls-corners (lCorners0 through the landed lStencil) | dam / paddleboats | fails: 419 / 109 φ words over 4 ulp (e.g. 32 ulp near φ = 0), 31-68k within; lAdvect −1.8% / −5.4% |
| ls-tri2 (trilinear in two copies) | dam / paddleboats | fails: 409 / 134 φ words over 4 ulp; lAdvect −5.7% / −6.8% |
| pMove corners (fgCornersOf) | dam / paddleboats | exact, but pMove +6.0% / +5.7% |

- pSeed: the excess is cancellation near zero, where the trilinear sum is small against its corners.
- lAdvect: the earlier "1 ulp" for all-distinct (above) was measured against a different base text. On the landed
  form the corner paths miss the tier, and they win only 2-7% on top of the landed −15%, not the −16 to −19.5% the
  earlier table had against the old base.

**uExtend shape (43e10865, landed).** 256-lane groups now sweep the brick's two z halves in turn, each over a
12 x 12 x 8 window. The dispatch is unchanged: one group a brick.
- Workgroup memory, ~15.4 KB a group (was ~23 KB):
  - wV/wD 9.2 KB, not 13.8;
  - wD is u32 with the distances bitcast, so the FC_LIQ words (wLiqW, removed) live in it until the first seed;
  - wS4 is three f32 a state, 4.5 KB, not 6.
- Cap 768 (regcap).
- Per-workgroup clock trace (cpages.mts MODE=clock, dam):
  - before (m3-cpages.md §5): 26–32 groups in flight;
  - now: 85–95 groups in flight through the pass, about three a core.
- stepab, exact (vel and fext4 identical) on all four scenes:

  | scene | uExtend in step | GPU wall |
  |---|---|---|
  | dam | 1.663 → 1.286 ms (−22.7%) | −1.49% |
  | paddleboats | 1.924 → 1.526 ms (−20.7%) | −1.20% |
  | seep | 0.653 → 0.530 ms (−18.9%) | −0.43% |
  | letters | 2.253 → 1.755 ms (−22.1%) | −1.11% |

- The go line (≥ 15% on dam and paddleboats) is met.
- velocity-formulas.test's "volume control" fails identically at HEAD without the change.
- The same shape likely applies to uExt4 (m4-clist.md: "the uExtend pattern"). Not tried.

## uExt4 and the uExtend pattern (lane A2, 2026-11-24, fifth task)

**Verdict: no-go.** uExt4's time is not set by residency, so the uExtend pattern does not help it. No lib change.

uExt4's time in the step (stepab family minima, base at 8a50eb74):

| scene | uExt4 |
|---|---|
| dam | 0.23–0.33 ms (it varies between processes; the A/A within a process is 0.000) |
| paddleboats | 0.33 ms |
| seep | 0.25 ms |
| letters | 0.44 ms |

So uExt4 is above the 0.2 ms floor on every scene.

**Variants.** All are in m1/variants.mts, kernel uExt4. Every one is exact: m1fix BUFS=all gives 0 words, and
stepab gives set, vel and fext4 identical.
- half: 256 lanes over the page's two z halves, with a 12 x 12 x 8 window (wV/wD 13.8 KB, not 20.7). Cap 576.
- pub: half, with one axis loop that publishes each axis's state as soon as it is swept (yPublish). Cap 640.
- seam: pub, with the seam's vLiquidOwner(c) read from the held slot. Cap 704.
- seam2: seam, with the seam's face tests ahead of its loads. Cap 768, so three groups a core.
- axis: the page over C_ACT4_G1's four workgroups, one axis a workgroup (the fourth leaves). Run with m4-stepab
  for the dispatch. Cap 640.

The base is 512 lanes at cap 640: one group a core.

**stepab, dam:**

| variant | uExt4 in step |
|---|---|
| seam | 0.267 → 0.320 ms (+20.0%) |
| seam2 | 0.226 → 0.232 ms (+2.8%) |
| axis | 0.250 → 0.294 ms (+17.9%) |

Each misses on dam, so none can land ("wins everywhere"). The other three scenes were not run.

**Attribution.**
- noflag: the base with the page sweep switched off, timing only. uExt4 went 0.326 → 0.246 ms (−24.6%). So the
  sweeps, which are what the pattern reshapes, are about a quarter of the kernel. The rest is the per-tile flag
  (x4Support's z taps) and the publish (C faces and ufxPublish4's 16-subface seam).
- Clock trace (t2/cpages-dam.log, dam):
  - 124 groups (C_ACT pages);
  - 32 in flight for the first quarter of the pass, falling to 1 over the second half;
  - group time p50 22 µs, p90 106 µs.
- So the pass is a tail of a few long pages, not a residency limit:
  - three groups a core (seam2) gains nothing;
  - halving a page's lanes doubles each page's serial work;
  - the axis split runs x4Support's flag three times over.

**Other F-side kernels with the signature** (≥ 512 lanes, wgmem ≥ 12 KB, ≤ 1 group a core). From a static scan
of the four scenes' schedules, regcap, and lane B2's M6 table (fluid-scratch/gpu-shape/m6/table.md;
m6-residency.md has not landed):

| kernel | lanes | wgmem | cap | groups a core | in step | worth reshaping? |
|---|---|---|---|---|---|---|
| nbOperator | 512 | 15.7 KB | 512–576 | 1 | dam 1.14 ms, paddleboats 2.34 ms | no (see below) |
| pmgCoarsenC | 512 | 18–20 KB | 1024 | 1–2 (peak 32 dam, 63 paddleboats) | 57–88 µs | no |

- nbOperator is register-bound, not memory-bound. Its cap is 512–576. At 256 lanes the same per-thread code gives
  two groups, still 512 threads a core. The uExtend win came from cap 768: three groups of 256 against one of 512.
- pmgCoarsenC is below the 0.2 ms floor, and B2's table classes it few/full+tail.
- pmgSetupF and pmgUpdate already run two groups a core (B2: "full").
- pmgCoarsenTop and pmgTopCycle are 1024-lane, few-group passes. Their cost is serial latency, not residency.

No candidate remains for this pattern.

## Register diet: lAdvect, nbOperator, pMove (lane A2, 2026-11-24, sixth task)

**Verdict: no exact variant crossed a cap step, so nothing was timed and there is no lib change.**

The baselines come from regcap on the 8a50eb74 dumps:

| kernel | cap |
|---|---|
| lAdvect | 384 |
| nbOperator | 576 (dam), 512 (paddleboats) |
| pMove | 576 (dam), 512 (paddleboats) |

The work is in fluid-scratch/gpu-shape/m1a2/t4/: rc.py (one substitution, one cap) and edit-*.py.

**lAdvect: the departure cubic holds the peak, through two independent copies.** Removing either copy alone leaves
384; removing both gives 448.
- lStencilCubic: the eight lsSlot values are live across the 64 taps, with a three-level select a tap.
  - edit-lpack.py packs them two to a u32 (16 bits: an F slot is under NB_FCAP ≤ 10240 in the four scenes).
  - With the fallback removed, this gives 448 on dam and paddleboats (384 without the packing).
- nbfCubicLimited, the fallback when lStencil misses: 64 lPhi0 taps, each a directory walk plus fgPhi's tiered
  path inlined. Stubbing only fgPhi in lPhi0 gives 448. Inside it, the absent vertex's eight-corner C interpolant is
  what counts (Read4 alone: 448).
- These exact rewrites all stay at 384:
  - lPhiAbsent: fgPhi as it resolves where no slot holds v, without the repeated walks;
  - a pairwise interpolant sum;
  - wy and wz recomputed where used;
  - opaque loop bounds;
  - the fallback read through the packed stencil slots.
- So 448 needs the fallback out of the kernel (fission). Its frequency was not measured, and +17% threads at the
  ~0.6 elasticity is about the 10% go line.

**nbOperator: no single region.**
- The six-face loop, the eight-vertex guard, and poStart at both sites (own cell and face halo) each hold 576 on
  their own.
- The pieces only cross together, and not exactly. On a trivial face loop:
  - guard and halo start removed: 704;
  - plus the own start: 832.
- These exact rewrites all stay at 576:
  - one poStart call site (a loop over halo then own);
  - K/KU stored in the loop instead of two live vec3;
  - pairwise poCentreG and poRootPhi4 with no array of 8;
  - the rate load moved late.

**pMove: the peak is spread over the post-G2P part.**
- Truncations, with p filed: cut before G2P 832; cut before the drift 640; cut before the endpoint 576.
- No call family crosses alone:
  - all three fgSampleVel;
  - fgDepth and the separation shift;
  - pHeat;
  - pMassDepth;
  - the merge;
  - the split;
  - pCount.
- Reloading `sep` late: 576.

**Caveat.** The caps move against what each edit removes (for example, pMove's merge removed: 512, and an
early-return probe: 512). So single-cap readings near a step are allocation noise. The conclusions above rest on
consistent multi-edit results.

## lAdvect fallback fission (lane A2, 2026-11-24, seventh task)

**Verdict: no-go.** The fallback is rare, and the fissioned pair is exact. But lAdvect at 448 threads is barely
faster than at 384, and the second pass costs more than that saving. Nothing landed in lib.

### Frequency

Method:
- 300 steps a scene: brick engine, fixed dt 1/60, census settings, moored bodies on paddleboats.
- Atomic counters in a scratch tree only: trees/fbprobe, made from t5/probe.patch.
- The counters live in nbSet's unused list-slot-31 padding. Header words 249..255 are taken: they are lane E's
  REGION_H (brick-census.wgsl.ts, brick-abi.md §5.1), which censusBegin, regionSpread, regionDeep and regionHeat
  write each step.

Fallback share by step range (1–100 / 101–200 / 201–300):

| scene | of lAdvect threads | worst step | bricks touched | worst step |
|---|---|---|---|---|
| dam | 0.079 / 0.093 / 0.071% | 0.21% | 2.3 / 3.2 / 2.3% | 6.6% |
| paddleboats | 0.015 / 0.003 / 0.003% | 0.04% | 0.6 / 0.35 / 0.3% | 1.25% |
| seep | 0.049 / 0.051 / 0.116% | 0.15% | 2.3 / 2.8 / 6.1% | 7.9% |
| letters | 0.007 / 0.041 / 0.032% | 0.09% | 0.4 / 1.8 / 1.9% | 3.3% |

Every fallback on every scene is a departure stencil past the lattice's low walls (cell − 1 < 0). None comes from a
missing brick.

So the go/no-go bar of ≤ 2% of threads and ≤ 10% of bricks passes on every scene.

### Fission

The patch is fluid-scratch/gpu-shape/m1a2/t5/fission.patch (tree trees/fis):
- **Main pass, lAdvect:** the eight stencil slots are packed into 16 bits. A vertex whose stencil misses returns at
  once; its PHI1 word is set to L_DEFER and its brick is listed once, using a mark word an F slot.
- **lDeferArgs:** one lane. It reads and zeroes the count and writes the indirect arguments.
- **lAdvectF:** one workgroup per listed brick, dispatched indirect. It runs lAdvect's full text on the vertices
  marked L_DEFER, then clears the mark and the list word. It needs a workgroupBarrier before the clear: without
  it, a later subgroup read a cleared list word, and about 15% of deferred vertices were left unadvected.

regcap gives the main pass 448 on all four scenes and lAdvectF 384.

### Timing

Measured with stepab, using a group exactness check: lAdvect + lDeferArgs + lAdvectF together, compared on set and
phi. The base was the same tree with lAdvect's text reverted.

The pair is identical word for word on all four scenes.

Family minima (ms):

| scene | lAdvect, HEAD | lAdvect, 448 | lAdvectF | pair vs HEAD |
|---|---|---|---|---|
| dam | 1.759 | 1.715 (−2.5%) | 0.298 | +14.7% |
| paddleboats | 2.133 | 2.146 (+0.6%) | 0.114 | +6.2% |
| seep | 0.639 | 0.639 (0%) | 0.187 | +30% |
| letters | 1.764 | 1.732 (−1.8%) | 0.170 | +8.1% |

lDeferArgs costs 5 µs.

Even with a free second pass, the main pass misses the 8% go line. lAdvect's time is not set by its thread cap.
Inference: its time is the trace's latency per lane, not occupancy. lAdvectF runs about 100 nearly empty workgroups
that each wait on a few full traces, a latency-bound tiny pass.
