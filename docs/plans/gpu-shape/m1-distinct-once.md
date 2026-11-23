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
