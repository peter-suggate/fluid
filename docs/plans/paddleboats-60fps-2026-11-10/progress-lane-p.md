# Lane P: pressure launch study (read-only, CPU prototypes)

2026-11-10. No file under `lib/` was touched and nothing is committed. Added:
`tools/experiments/lane-p/pressure-launch-proto.ts` and this file. Ledgers and prototype output are in
`~/code/me/fluid-scratch/lane-p/`.

Every number below is labelled **measured** (this lane or a cited earlier lane), **model** (the CPU
prototype) or **assumed**.

## 0. Verdict

**NO-GO against the stated bar** (the same residual with at least 2x fewer launches on every case, with no
per-scene tuning).

- The best candidate that leaves convergence alone takes a cycle from 43 launches to 26: **1.65-1.73x** on
  all three prototype cases (model).
- Variants that also cut the tail to one cycle reach about 2.0-2.9x on the shallow tank and the jet but only
  **1.6-2.0x on the deep tank** at the tighter target and from a cold start (model). The GPU already showed the same thing: `tailCycles 1` left letters
  25-40% looser (measured, 2026-11-03).
- In milliseconds the prize is smaller than the launch count suggests. The solve pass was measured at
  5.8-6.0 ms of GPU time over 171 launches, about 34 us each, not 70. Removed pressure launches have been
  worth 9-30 us each in every earlier A/B. So the 26-launch cycle is an estimated **0.7-2.0 ms** on
  paddleboats, and 4.7 ms only under the 70 us assumption, which the earlier measurements contradict.

## 1. Launch anatomy

### Sources

- Paddleboats: Lane B's ledger, step 30 (`fluid-scratch/lane-b/ledger-paddleboats-30.json`, rc 0).
- Seep and letters: this lane's two GPU runs, step 60, both rc 0, check mode (not timing gates):
  `tools/probe-reservoir-bulk.ts --pipelined --dispatches --steps=60 --every=60`, and the same with
  `--scene=nbflip-figure-8-letters --dt=0.016666666666666666`. GPU runs used: 2 of 2.
- The per-cycle split is `uniform-pressure-arena-launches.ts`, which
  `tests/uniform-pressure-arena-launches.test.ts` holds to the encoder. The ledgers' per-kernel counts agree
  with it on all three scenes.

### Per step (measured)

| | paddleboats (step 30) | seep (step 60) | letters (step 60) |
|---|---|---|---|
| Launches a step, all stages | 489 | 467 | 464 |
| Pressure family | 215 | 211 | 210 |
| - arena solve pass | 178 | 178 | 177 |
| - arena prepare + unmark / projection / verdict | 9 / 5 / 1 | 9 / 5 / 1 | 9 / 5 / 1 |
| - surface pressure authority | 16 | 12 | 12 |
| - rhs, mixed project, bulk faces, 2 transfers, root owners | 6 | 6 | 6 |
| Cycles encoded / run | 4 / 3 | 4 / 3 | 4 / 4 (the cap) |
| Band tiles (64 h rows each) | 1,169 | 238 | 2,657 |
| Resident pages / BULK16 live slots | 192 / 0 | 82 / 72 | 107 / 0 |
| Workgroups in the solve pass | 71,368 | 46,071 | 118,866 |

The structure is identical on the three scenes. Only widths differ. Letters has no solid, so it lacks the
one settle sweep.

### Where 178 comes from

178 = 4 (init) + 40 (cycle 0) + 3 x 43 (cycles 1-3) + 5 (final).

One cycle, 43 launches. Workgroups a launch are paddleboats / seep / letters (measured):

| Level | Launches a cycle | Workgroups a launch | Notes |
|---|---|---|---|
| 0, h band: pRelax x6 (3 pre, 3 post) | 6 (+1 settle in cycle 0 with a solid) | 738 / 512 / 1,677 | A launch is one red-black sweep, both colours. Workgroup = 2 tiles x 64 lanes |
| 0: pRestrict0, pRebaseC, pProlong0 | 3 | 1,475 / 1,024 / 3,354 | |
| Native 4h rows: pNativeRelax x4 (2 colours, pre and post) | 4 | 240 / 103 / 133 | 512 lanes a page |
| Composite residual pResidual2, pNativeOperator | 2 | 240 / 103 / 133 | |
| 1, 2h: pDownA, pDownR, pUpA | 3 | 73 / 67 / 173 | Overlapping box, 3 sweeps inside the launch |
| 2, 4h: pDownB, pDown2, pBulkLift, pUpB | 4 | 240 / 103 / 133 | Box, 6 sweeps inside the launch |
| 3, 8h: pDown3 x3, pUp3 x3 | 6 | 240 / 103 / 133 | Three tail cycles |
| 4, 16h: pDown4 x3, pUp4 x4 | 7 | 242 / 193 / 135 | The last tail cycle's up leg is two colour passes |
| 5, 32h: pTail x3 | 3 | **1** | 256 lanes, 32 sweeps, serial by design |
| Krylov GCR(1): kApplyT, kApplyNB, kReduce, kBound, kStep | 5 (cycle 0: kStart only) | 1,475; 312; **1**; 240; 240 (paddleboats) | |

So a cycle is: level 0, 9 + natives and residual, 6 + level 1, 3 + level 2, 4 + the tail, 16 + Krylov, 5.
The formula asked for:

- sweeps x colours is already 1 launch a sweep at level 0, and 1 launch a multi-sweep box at levels 1 and 2;
- the tail is `tailCycles` x 5 + 1 = 16, the largest single term;
- dot products and the step are 5;
- the convergence check is 1 (pVerdict). The per-cycle measure rides on pResidual2, which the restriction
  needs anyway.

Once a step: pInit, pNative, pBulkRows, pRebase; pResidual2F, pFinal, pWrite, pBulkProject, pBulkCommit.

### Single-workgroup and tiny launches a step (measured)

| Launch | Count a step | Workgroups |
|---|---|---|
| pTail | 12 | 1 |
| kReduce | 3 | 1 |
| pVerdict | 1 | 1 |
| pBulkCommit | 1 | 1 (12 on seep) |
| pBulkRows, pBulkProject | 2 | 2 (90 on seep) |
| ubaLevel1, ubaRelease | 2 | 5 / 3 / 13 |
| Authority bodyArgs, bodyReset (paddleboats; Lane B's ledger) | 2 | 1 |

That is 19 single-workgroup pressure launches a step on paddleboats (17 in the arena, 12 of them pTail) and 4 tiny ones.

### Idle launches

Paddleboats and seep run 3 cycles and encode 4 (`min(cycles, wideCycles + 1)`). The fourth cycle's 43
launches exit on the kernels' own converged test. Letters runs all 4.

## 2. What a pressure launch costs: the measurements on file

The brief's figure is 70 us a latency-bound launch (the 34 ms floor over about 490 launches). For the
pressure solve the earlier measurements give less:

| Evidence (all measured, earlier lanes) | Value of a launch |
|---|---|
| Solve pass 5.8-6.0 ms GPU time over 171 launches at 3.1 cycles (`small-scene-step-2026-11-03/progress-pressure.md`, P12) | 34 us mean, work included |
| Tail launches, xctrace (`small-scene-step-2026-11-03/pressure.md`) | 13-33 us each |
| Krylov kernels fused, 11-14 launches removed (`bulk16-plan-2026-10-28.md` :4340) | about 0.11 ms, 9 us a launch |
| `tailCycles 1`: 32 launches a step removed, with their work | 0.9 ms GPU, 28 us a launch |
| pDown4 folded into pTail: 12 launches removed | pTail 0.35 -> 0.77 ms: **slower** |
| Fused page tail (pTailV) | 1.46 -> 0.71-1.06 ms, and letters' residual worse |
| Far half of a tank frozen: 0.96 ms of arena kernels and 10 of 34 cycles removed (`activity-step-2026-11-05/progress-pressure.md`) | **step time unchanged** |
| Launches of a converged cycle | 5-15 us each (estimate in `pressure.md` P5) |

The earlier conclusion stands: "the arena is bound by each kernel's critical path, not by the number of
launches". I therefore price a removed pressure launch at **10-30 us (measured basis)** and show the
70 us figure beside it as **assumed**. The whole solve pass is about 6 ms, which caps any restructuring of
the cycle. The rest of the pressure family (rows, projection, authority, rhs; about 5.5 ms, estimated as the
11.5 ms family less the solve) is 37 heavy launches and is not launch-bound.

## 3. Options, ranked

Launches are the solve pass a step on paddleboats, 4 cycles encoded, today 178.

| # | Option | Solve launches after | Saved: measured basis / at 70 us (assumed) | Convergence | Physics risk |
|---|---|---|---|---|---|
| 1 | **Lean cycle, tail x3**: level 0 as one overlapping-box launch a leg (the level 1-2 smoother), prolongation staged into the post box, tail legs fused per page, Krylov 5 -> 3, each native relax pair in one launch, pRestrict0 with pRebaseC. 26 a cycle | 111 | 0.7-2.0 ms / 4.7 ms | Model: unchanged on all three cases | Low to medium. See "unverified" |
| 2 | **Do not encode the idle cycle**: grow from the verdict's final residual instead of from a spare cycle | 135 (paddleboats, seep); letters unchanged | 0.2-0.6 ms / 3.0 ms | A frame that needs one more cycle ends looser until the receipt returns (about 2 frames) | Medium: residual left in the velocity on the frames where the flow gets harder |
| 3 | 1 + 2 | 85 | 0.8-2.4 ms / 6.5 ms | As 1 and 2 | As 1 and 2 |
| 4 | **Lean cycle, tail x1**. 20 a cycle | 87 | 0.9-2.7 ms / 6.4 ms | Model: deep tank's late contraction falls from 9.8x to 4.3x a cycle. Measured: letters 25-40% looser | High: fails equal-or-better on a gate scene |
| 5 | (a) Dense or more exact coarsest solve | No change in cycles | 0 | Measured: tailCycles 5 and 9, coarsestSweeps 12 and an exact CPU coarsest solve change nothing | None, and no gain |
| 6 | (b) Chebyshev or more sweeps | None removed | 0 or negative | Level 0 already runs both colours in one launch. Measured: contraction is the same at sweeps 3, 4, 5; the coarse correction is the limit | - |
| 7 | (c) Warm start, fixed cycle count | - | - | Already there: the solve starts from the presented pressure, and the cycles follow the last frames' count | - |
| 8 | (d) Column or height-field coarse space, 3D work only near bodies | Unknown | Unknown | Not prototyped. Measured: the calm bulk is already 4h and BULK16 rows sitting 2-3 decades under the band's residual, and the worst rows are deep-interior band rows near the action. Freezing the calm half lost volume (drift x90) and did not shorten the step | High |
| 9 | Galerkin or PtAP coarse operators, to reach 2 cycles | Fewer cycles, more launches a cycle | Measured and priced on 11-03: -0.9 to +0.3 ms | CPU 1.4-1.6 decades a cycle, about half of it transferring to the GPU | Rejected on file |

Notes on the items.

**(a) Coarse levels in one launch.** Only the 32h rows fit one workgroup, and that single workgroup is the
form the user forbids. Folding more into it serialises work that page workgroups do side by side: measured
slower. What remains is fusing the tail's legs per page (inside option 1), which removes launches but not
pTail itself.

**(b) Block smoothers.** Levels 1 and 2 already are multi-sweep overlapping boxes. The untried part was
level 0, and it is the prototype's main result: one box launch of 3 sweeps does what three launches do
today. In milliseconds I expect it to be about neutral: pDownA, the same kernel shape at level 1, costs
74-133 us a launch against 3 x 51 us for three pRelax launches, and the band has 5-10x more bricks than
level 1.

**(e) One fused launch per level per phase** is option 1. Its floor is 20-26 a cycle, not 10, because a
residual needs its neighbours' finished values (a launch boundary a level) and a dot product needs every
workgroup's partials.

**pTail and kReduce.** Options 1 and 4 keep one pTail a tail cycle. Removing it means solving the 32h rows
redundantly in every page workgroup of the up leg. That fits only while the rows fit workgroup memory, so
it does not generalise to large pools. Not prototyped. kReduce (a single workgroup) folds into kStep in
option 1.

## 3a. Option 1's launch budget a cycle

| Item | Today | Lean | Arithmetic |
|---|---|---|---|
| Level 0 pre-smoothing | 3 | 1 | Changes (box). Model: neutral |
| Native relax, pre | 2 | 1 | Changes: the second colour reads other pages' first colour a launch late. Not in the model |
| pResidual2 | 1 | 1 | |
| pRestrict0 + pRebaseC | 2 | 1 | Unverified that they can share a launch |
| pNativeOperator | 1 | 1 | |
| Level 1 | 3 | 3 | |
| Level 2 | 4 | 4 | |
| Tail | 16 | 9 (3 with one tail cycle) | The down and up legs fused per page; pTail stays |
| pProlong0 + level 0 post-smoothing | 4 | 1 | Changes (box) |
| Native relax, post | 2 | 1 | As above |
| Krylov | 5 | 3 | Unchanged if the reduction order is pinned |
| **Total** | **43** | **26 (20)** | |

## 4. CPU Float64 prototype

`npx tsx tools/experiments/lane-p/pressure-launch-proto.ts [--case=shallow|deep|jet] [--json=path]`
(about a minute for all three; `npm run check:types` is clean).

**What it shares with the arena:** six levels h to 32h; sum restriction; trilinear renormalised
prolongation; lumped 7-point aggregates (x0.5, x0.75 at 16h and 32h) with the Dirichlet row sums kept;
level 0 red-black sweeps that are Jacobi across two-tile workgroups; overlapping 8^3 boxes at levels 1 and
2 (halo 2, core kept, 3 and 6 sweeps); three tail cycles with page-block smoothing and a 32-sweep 32h
solve; flexible GCR(1) from the second cycle.

**What it leaves out:** the band and native composite (every liquid cell is an h row), seams, the wall
active set, BULK16 and f32.

**Cases** (h = 1, ghost-fluid free surface with theta in [0.1, 1], a moving solid box, a splash lump with
random face velocities, gravity):

- shallow tank 128 x 40 x 48, 24 deep, 147k rows;
- deep tank 48 x 72 x 48, 56 deep, 129k rows;
- thin jet: a radius 2.5 column falling into a 12-deep pool, 21k rows.

Each is solved from the previous step's converged pressure (the wave, the lump and the box all moved) and
from zero. Launches a cycle use the real composite's count (section 3a), so the model measures cycles and
the count comes from the encoder.

**Calibration (model against measured).** The model's late contraction is 9.8-13.7x a cycle for today's
scheme. The GPU measures about 10x on paddleboats. The model's first cycle is easier than the GPU's (it
reaches 1/400 in 1.0-1.3 cycles from the warm start; paddleboats needs about 3), so the 1e-5 column is the
representative one.

**Results.** "x" columns are today's launches over the candidate's, at the same max-norm reduction.

| Scheme | Launches a cycle | Shallow: late x/cycle | 1/400 warm | 1e-5 warm | 1/400 cold | Deep: late x/cycle | 1/400 warm | 1e-5 warm | 1/400 cold | Jet: late x/cycle | 1/400 warm | 1e-5 warm | 1/400 cold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Today | 43 | 12.5 | 1.00x | 1.00x | 1.00x | 9.8 | 1.00x | 1.00x | 1.00x | 13.7 | 1.00x | 1.00x | 1.00x |
| Today, tail x1, launches unfused | 33 | 8.5 | 1.31x | 1.23x | 1.14x | 4.3 | 1.31x | 0.99x | 0.95x | 12.8 | 1.30x | 1.29x | 1.26x |
| Today's arithmetic, launches fused | 31 | 12.5 | 1.39x | 1.39x | 1.39x | 9.8 | 1.39x | 1.39x | 1.39x | 13.7 | 1.39x | 1.39x | 1.39x |
| **Lean, level 0 box k3, tail x3 (option 1)** | 26 | 13.2 | **1.65x** | **1.67x** | **1.66x** | 9.8 | **1.73x** | **1.67x** | **1.67x** | 17.4 | **1.70x** | **1.65x** | **1.73x** |
| Lean, box k6, tail x3 | 26 | 12.7 | 2.25x | 1.78x | 1.79x | 9.2 | 2.01x | 1.72x | 1.77x | 17.3 | 1.93x | 1.75x | 1.95x |
| Lean, box k3, tail x1 (option 4) | 20 | 8.7 | 2.15x | 2.05x | 1.97x | 4.2 | 2.26x | 1.65x | 1.60x | 16.6 | 2.21x | 2.12x | 2.16x |
| Lean, box k6, tail x1 | 20 | 8.6 | 2.92x | 2.19x | 2.12x | 4.4 | 2.61x | 1.81x | 1.70x | 16.4 | 2.51x | 2.25x | 2.45x |
| Lean, box k6, boxes 6/12 sweeps, tail x1 with 6 sweeps | 20 | 11.5 | 2.95x | 2.35x | 2.25x | 5.6 | 2.64x | **1.98x** | **1.88x** | 24.6 | 2.50x | 2.42x | 2.55x |
| The same, tail x2 | 23 | 15.1 | 2.56x | 2.14x | 2.07x | 7.7 | 2.29x | **1.93x** | **1.95x** | 25.7 | 2.17x | 2.11x | 2.25x |

Readings:

1. **The level 0 box costs no convergence** on any case: late contraction 13.2 / 9.8 / 17.4 against
   12.5 / 9.8 / 13.7. One launch replaces three, and the post leg absorbs the prolongation.
2. **One tail cycle fails on the deep tank**: 9.8x -> 4.3x a cycle. It holds on the shallow tank's first
   cycles and on the jet. This is the GPU's own result (paddleboats held its cycles, letters ended looser).
   More sweeps inside each launch recover part of it (5.6x), not all.
3. **No row is at least 2x in every column.** The closest (20 a cycle with doubled in-launch sweeps) is
   1.88-1.98x on the deep tank, and doubling the sweeps doubles those kernels' critical paths, which the
   launch count does not see.
4. The warm start did not lower the starting residual here (0.6-1.2x). I made the step-to-step change
   large, so this says nothing about warm starts in the solver.

**Verdict by the stated rule:** option 1 is a consistent 1.65-1.73x, below 2x: NO-GO. Option 4 is over 2x
on two cases and under on the third: NO-GO.

## 5. Summary for the user

- **What reopening pressure could save.** Solve-pass launches a step from 178 to about 111 (option 1), or
  85 with the idle cycle also gone (letters runs all 4 cycles, so it keeps 111). Estimated 0.7-2.0 ms and 0.8-2.4 ms on
  paddleboats at the 10-30 us a launch that earlier A/Bs measured in this pass; 4.7 and 6.5 ms if a launch
  were worth 70 us (assumed). The whole solve pass is about 6 ms of the 52.
- **Best option.** Option 1: level 0 smoothing as one overlapping-box launch a leg, per-page fused tail
  legs, a 3-launch Krylov step, one launch per native relax pair. It removes kReduce's single-workgroup
  launch and 6 launches of every 16 in the tail. It keeps pTail.
- **Risk.** The level 0 box changes the arithmetic on the rows that matter most (theta rows, seams, the
  wall active set), and the model has none of them. The earlier lane recorded that its CPU model
  over-predicted GPU gains (+12% predicted, +4% measured). The fused page tail was built once before and
  came out at 0.71-1.06 ms against 1.46 with a worse letters residual.
- **Size of the change.** `lib/methods/uniform/uniform-pressure-arena.ts` (a new level 0 box kernel
  replacing pRelax and pProlong0, fused pDown3/pDown4 and pUp4/pUp3, the Krylov reduction, the native
  relax pair, the schedule), `uniform-pressure-arena-launches.ts`, `uniform-mixed-frame.ts` if the cycle
  budget rule changes, and `tests/uniform-pressure-arena-launches.test.ts`,
  `tests/pressure-tail-red-black.test.ts`, `tests/pressure-tail.test.ts`, `tests/pressure-krylov.test.ts`
  and the residual-history test. Every kernel the solve launches per cycle changes except levels 1 and 2.
- **It does not clear the bar.** Nothing reaches 2x fewer launches at the same residual on all cases. The
  cycle count is set by the coarse correction (about 10x a cycle), and every way to raise that is already
  on file as measured and rejected. The remaining gain is launch bookkeeping, worth an estimated 0.7-2.4 ms by
  the measurements on file.

### Unverified

- The value of a removed launch in milliseconds. This lane ran no timing runs. The 10-30 us range is from
  earlier lanes' A/Bs on older trees.
- The level 0 box on the real band: its thread cap, its time a launch against three pRelax launches (my
  estimate is about equal), and its convergence with seams, the active set and f32.
- Fusing pRestrict0 with pRebaseC, and the two native colours in one launch: I did not check them against
  the WGSL for data races. An earlier lane found pResidual2 with pRestrict0 to be one.
- Whether a verdict-driven cycle budget (option 2) holds drift on seep and letters.
- The ledgers are single runs at step 30 (paddleboats) and step 60 (seep, letters). Later steps have wider
  launches and the same counts.
- Option (d), a column or height-field coarse space, was not prototyped. The case against it rests on
  earlier measurements, not on a test of that design.
