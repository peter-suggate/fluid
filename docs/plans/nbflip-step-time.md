# NB-FLIP step time on large calm scenes (2026-10-27)

Goal: cut the per-resident-block cost of the NB-FLIP step (wide seep: about
13 ms fixed + 43 µs per resident block + 10.6 µs per h tile; see
`nbflip-16h-progress.md` section 4). No physics change, fixed dt 1/60,
`volumeControlSeconds: 0`. All GPU work went through
`tools/gpu/fluid-gpu-run.sh`. Artifacts: `artifacts/nbflip-step-time/`.

## 1. Which dispatches scale with resident blocks

Kernel ladder (`tools/probe-scale-ladder.ts --kernels=4 --steps=14`, 16h,
tiers `4h-b512`, 5 mm, steps 4-14, h tiles held about equal at 157-202 so
the slope is blocks, not h tiles). Artifacts `kl-w{1d1,1d2,2d1,2d2}.{json,log}`.

| scene | resident blocks | h tiles | kernel sum (ms) |
|---|---|---|---|
| w1 d1 | 128 | 201 | 41.6 (first-run outlier, excluded) |
| w1 d2 | 252 | 193 | 24.8 |
| w2 d1 | 240 | 157 | 24.2 |
| w2 d2 | 476 | 202 | 34.1 |

The GPU is not saturated below about 250 blocks (w1d2 and w2d1 sum to the
same 24 ms as each other), so the per-block cost is the marginal cost
between the 246-block mean and 476 blocks: **44.2 µs per block**, which
matches the 43 µs fit. By pass (µs per block, marginal; dispatches below
about 0.4 µs per block are at the timer's 65.5 µs quantum):

| pass | µs/block | main dispatches (µs/block) |
|---|---|---|
| pressure arena solve (V-cycles) | 7.7 | pUpB 1.5, pDownB 1.1, pUp3 0.8, pUp4 0.75, pNativeRelax 0.6, pDown2 0.6, pTail 0.5 |
| surface pressure authority | 3.9 | surfaceHeld 2.9-3.1, wetRegions 0.4 |
| pressure project | 3.4-3.6 | project 3.4 |
| dynamic ownership census | 2.2-2.6 | classifyCoarse 1.0, solidPromote 0.8-0.9, decide 0.3 |
| global surface volume (diagnostic) | 2.1-2.4 | dilate 1.3-1.4 |
| extension | 2.1 | pair_sweepSeams 0.7-0.9, prolong 0.6, seedList 0.4-0.6 |
| layout build | 1.6-1.9 | mirror 0.6, classify 0.4 |
| arena projection | 1.6-1.7 | pCopyNative 1.55 |
| residency | 1.5-1.6 | pageMark 1.5 |
| pressure rhs | 1.3-1.4 | buildRhs 1.3 |
| local speed certificate | 1.2-1.7 | localSpeed 1.2 |
| momentum tiers | 1.0 | momentumAffineResidual 0.8 |
| body forces | 0.8-1.1 | cacheNormals, forcesRegularCoarse |
| 4h sampling cache | 0.8-1.0 | cache 0.8 |
| bulk tier selection | 0.75 | bulkSelect 0.75 |
| surface advect deferred | 0.5-1.6 | general twin; ~1.2-1.5 ms at every width (not per block) |

Pressure (arena solve + project + rhs + arena projection) is about 15 µs
per block, a third of the slope. It is real work on every 4h row; only a
coarser bulk pressure (BULK16) removes it.

### The global surface volume pass is a diagnostic

`readStats()` (webgpu-uniform-reference.ts) asks the next step for
NB-FLIP's volume probe, which runs the "global surface volume" pass (seed,
dilate x4, grow, metric, measure). The app calls readStats only at pause
(`webgpu-renderer.ts` setSimulationRunning), so live frames never run it.
`probe-scale-ladder.ts` called readStats every step, which put this pass
(2.1-2.4 µs per block, 2-4 ms late) inside every measured step. Added
`--stats-every=N` (default 1, unchanged behaviour); the A/B runs below use
`--stats-every=10`, so 9 of 10 measured steps are app-like.

## 2. Changes and measurements

Method for every change: two snapshots of the same working tree that differ
only in the change's files (`$S/mksnap.sh`), run interleaved base, cand per
repeat, 3 repeats, timing mode, clean (no kernel timer):

- wide seep: `probe-scale-ladder.ts --cell=0.005 --width=2 --depth=2
  --steps=70 --every=35 --stats-every=10 --tiers=tiers-4h-b512.json
  --values='{"volumeControlSeconds":0,"bulkTier":"16h"}'`. The seep is
  nondeterministic (h tiles differ between repeats), so the comparison is
  the per-repeat least-squares fit of step time on h tiles, read at 1000 and
  3000 h tiles, plus window medians/p95.
- closed rest: `probe-reservoir-bulk.ts --closed --cell=0.005 --steps=120
  --every=30` with the same values; steps 21-90 median/p95, and step 120
  max speed, volume and drift (must equal the baseline exactly).

### Change 1: surfaceHeld and pCopyNative lane use (kept)

Both kernels spent most of their lanes idle on the 4h tiles that make up
calm bulk:

- `surfaceHeld` (NB-FLIP surface pressure authority,
  `uniform-mixed-pressure-authority.ts`): an item was an eighth of a held
  region (8 of 64 lanes active) and every item paid `umHeldPage`'s binary
  search. Now an item is a whole held region, its 64 tiles a lane each (one
  binary search per region), then the h-tile list, an owner a lane. The same
  owners run `umSurfaceOwner` with the same stamps; nothing else changes.
- `pCopyNative` (arena projection, `uniform-pressure-arena.ts`): a workgroup
  per tile, where a 4h tile used 3 of 64 lanes. Now a workgroup takes 64
  tiles, a lane a tile for the 4h faces, then each h tile's 64 cells across
  the lanes. The same texels are copied.

Artifacts `c1-{seep,rest}-{base,cand}-r{1,2,3}.{json,log}`.

| wide seep | base r1 / r2 / r3 | cand r1 / r2 / r3 |
|---|---|---|
| fit at 1000 h tiles (ms) | 43.9 / 45.1 / 46.6 | 41.7 / 40.6 / 40.8 |
| fit at 3000 h tiles (ms) | 65.8 / 65.7 / 66.0 | 63.8 / 59.1 / 59.0 |
| steps 56-70 median / p95 (ms) | 58.6/62.0, 61.0/64.8, 64.5/70.7 | 58.5/61.8, 53.7/57.4, 58.4/61.5 |
| steps 56-70 h tiles | 2437 / 2670 / 3050 | 2677 / 2488 / 2990 |
| all steps 5-70 median / p95, pooled (ms) | 48.6 / 65.4 | 43.2 / 58.6 |
| resident blocks (late) | 548 / 548 / 552 | 550 / 546 / 549 |

The fitted ranges do not overlap at either point: about -4.2 ms at 1000 h
tiles and -5.2 ms at 3000 (pooled p95 -6.8 ms).

| closed rest | median / p95 steps 21-90 (ms) | step 120 max speed, volume, drift |
|---|---|---|
| base r1 / r2 / r3 | 14.44/16.30, 14.65/18.51, 25.94/29.68 (r3 outlier) | 0.013471915386617184, 2949119.8727798667, 0 (all) |
| cand r1 / r2 / r3 | 13.97/14.98, 13.82/15.56, 13.74/15.58 | identical, every digit |

Rest: about -0.6 ms median against the two normal base repeats (the base r3
run was slow throughout, 24 ms median; it is reported, not dropped, and the
conclusion does not rest on it). Rest physics identical to every digit.

### Change 2: pageMark lane use (reverted)

`lrResidentLane` (residency pageMark, `uniform-local-residency.ts`) gave a
lane one neighbour block of the search box, so a 3^3-block box kept 27 of 64
lanes busy, each reading eight pages' records in series. The candidate gave
a lane one (block, page) pair instead; the OR of stage bits is
order-independent, so the marks should be identical. Same method, baseline =
change 1 applied, artifacts `c2-*`:

| wide seep | base r1 / r2 / r3 | cand r1 / r2 / r3 |
|---|---|---|
| fit at 1000 h tiles (ms) | 40.9 / 40.3 / 49.1 | 38.9 / 40.2 / 40.5 |
| fit at 3000 h tiles (ms) | 57.4 / 58.0 / 71.3 | 54.7 / 58.0 / 59.9 |
| steps run | 70 / 70 / 70 | **65 / 70 / 68** |

| closed rest | median / p95 steps 21-90 (ms) | step 120 |
|---|---|---|
| base r1 / r2 / r3 | 13.77/15.29, 13.70/15.16, 20.14/34.49 (slow run) | identical to change 1 |
| cand r1 / r2 / r3 | 13.83/15.84, 13.77/15.22, 14.19/15.07 | identical to change 1 |

No win beyond repeat noise (the seep fits overlap; rest is unchanged), and
two of three candidate seep runs stopped on the same fatal, at step 65 and
step 68: "Narrow-band FLIP: a particle arrived in page (54,29,30), which holds
no storage; the residency closure is too small". No baseline run failed,
here or in change 1's six seep runs. I could not find a semantic difference
by reading the kernel. But pageMark feeds the residency certificate, so I
cannot rule it out. **Reverted** (the file is byte-identical to HEAD again).
The GPU certificate test that compares every pageMark flag with an
independent CPU enumeration (`tests/uniform-stage-residency.test.ts`, long
slide) was run on the candidate snapshot; see below.

Also note that change 1 and the current seep-agent working tree (st-base2)
ran 3 of 3 wide seeps to 70 steps without that fatal.

Certificate test (`tests/uniform-stage-residency.test.ts`, "long slide",
check mode; `c2-stage-residency-test{,-base}.log`): on the change 2
candidate, every pageMark flag equalled the independent CPU enumeration at
steps 1 and 30. Both the candidate and its baseline (change 1 plus the
current seep-agent tree) then fail the same later assertion at step 169,
"empty bottom-left page must retire". That failure does not come from
change 2. I did not run it on HEAD, so I have not shown whether it was
already there before this working tree.

## 3. Gates for change 1

- Splash: `tools/gate-repeat-spread.ts step-time-c1-splash --scene=splash
  --steps=70 --repeats=3 --set=volumeControlSeconds:0`, baseline = current
  tree with the two change 1 files from HEAD, candidate = current tree
  (`gate-splash-c1.log`, `gate-splash-c1/`): **pass**. Fine tiles, band
  tiles, band bricks and occupancy are identical in all six instances.
  Particles differ by at most 2 between repeats of one arm (797 vs 795 at
  frame 70; A/B 0). Volume drift A/B 7.9e-8, worst window ratio 4.0e-5.
- Closed rest: max speed, volume and drift at step 120 equal the baseline to
  every digit in all six c1 runs (section 2).
- Registered seep gate (`--scene=reservoir-seep`, 3 repeats, 6 instances of
  about 3 GiB): **not run**. It is over the 6 GiB rule, the same reason the
  previous campaign gave. The six wide-seep timing runs of change 1 all ran
  70 steps without a fatal.

Checks: `npm run check:types` reports no errors in the touched files (the
only errors are in `lib/svo/...` files another session is editing). Focused
tests (`pressure-tail`, `uniform-local-residency`, `uniform-stage-residency`,
`uniform-solid-refinement`, `uniform-narrow-band-seeding-params`): 20 pass,
3 GPU-skipped. Full `npm run test:unit`: 1372 tests, 1240 pass, 0 fail, 132
skipped.

## 4. Next levers (evidence, not yet measured as changes)

1. **Coarse bulk pressure (BULK16).** Pressure is about 15 µs per resident
   block (arena solve 7.7, project 3.4, rhs 1.3, arena projection 1.6
   before change 1), a third of the per-block slope. Every one of those
   dispatches does real work on 4h rows. Only a coarser bulk system removes
   it. At 557 blocks that is about 8 ms.
2. **advectDeferred in walled scenes (4.85 ms late on the wide seep, 1.31 ms
   at rest).** From the code: the coarse merged advect launch is compiled
   with `umDeferTraced`, so every non-COLD 4h vertex is queued as a code-2
   entry. The fast twin skips code-2 entries by construction (`if(code==2u&&
   umFastSurface){continue;}`, because their preflight, including the
   buried test, needs the solid sampler). So every WARM/HOT 4h vertex of the
   basin goes through the general solid-aware twin. A locally certified
   preflight for code-2 entries in the fast twin (clear-tile certificate in
   place of the buried test) is the candidate. I did not count the list's
   composition (no seep work-counter probe exists yet;
   `tools/investigate-pool-work.ts` is pool-scene only), so this is a
   reading, not a measurement. **Measured in section 5: no correct
   certificate wins. The lever is closed.**
3. Smaller per-block items, each under 1 µs per block at the timer's
   quantum: classifyCoarse (already a replay), solidPromote (27-neighbour
   scan per non-band tile), localSpeed, bulkSelect, layout mirror.

## 5. advectDeferred code-2 certificate (measured, nothing kept)

Every number here comes from a scratch snapshot of the tree as it stood that
afternoon, with the seep agent's flip work in flight. Seep runs use the
standard wide-seep args (5 mm, w2 d2, 70 steps, `tiers-4h-b512`,
`bulkTier:16h`, `volumeControlSeconds:0`). Rest runs use `probe-reservoir-bulk
--closed`, 120 steps. Timing mode, base and candidate interleaved. The
artifacts are in `artifacts/nbflip-step-time/` (`ad-reach*`, `ko-seep-*`,
`koc-*`).

### Counts: the merged list's 4h (code 2) entries

These come from temporary flags.z-gated counters in `advectDeferred`
(`surface-deferred-counters.surface.patch`, plus the saved
`surface-deferred-work.ts.saved` reader and `--surface-work` probe hooks, all
removed again). With the counters on, closed rest stayed identical to every
digit at step 120 (volume 2949119.8727798667, max speed 0.013471915386617184,
drift 0).

| | listed | traced by general twin | support valid (2w+7) | reach-clear, h margins | reach-clear, hull ±12 |
|---|---|---|---|---|---|
| rest, step 120 | 7343 | 7343 | 0 | 6258 (85%) | 4075 (55%) |
| seep, step 63 | 53975 | 45299 | 0 | (40706 of 44928 in the other run: 91%) | 22310 (49%) |
| seep, steps 49-63 | 46-56k | 43-49k | 0 | 89-91% | 47-49% (about 22.5k flat) |

- All code-2 entries go to the general twin. The fast twin admits 0 of them
  (the code-2 skip). `umLocalSolidAdmission` admits 0 of any code.
- The existing support certificate (`umCanonicalSupportAt` at reach 15) is
  never valid for a 4h vertex here, so a support-based preflight can admit
  nothing.
- `umReachSolidClear`'s margins (cubic ±2, foot ±3) are h-cell margins. A 4h
  vertex's general reads go further: foot smoothing taps at ±2w = ±8, cubic
  taps at ±8 around the end, and burial/capacity reads ±4 at those taps.
  The correct box is the trace hull ±12. Only about half the traced entries
  pass it.
- My first candidate (the fast twin takes code-2 entries, then holds and
  kept air) changed rest physics: volume 2949119.9382 against 2949119.8728,
  max speed 0.0134827 against 0.0134719. Every mismatching entry came
  through `umHoldCold`. `umVertexTier` reads `umVertexCellCapacity`, which a
  solid-free compile folds to "open", so the fast twin holds vertices the
  general twin traces. Holds and other preflight steps must stay in the
  general twin.

### Timing bounds

All seep figures are from a least-squares fit of `wall_ms` against h tiles,
steps 5 and up, per repeat (`fit.py`). Raw step medians are confounded by
how many h tiles each run reaches.

| arm (seep) | what it does | @1000 h tiles | @3000 h tiles | per repeat @3000 |
|---|---|---|---|---|
| base (`ko-seep-diagbase`, 2 reps) | current | 43.0 | 61.2 | 61.0, 61.6 |
| koa | general merged twin compiled solid-free (wrong) | 49.3 | 65.5 | 63.9, 67.0 |
| kob | merged general compiled fast + free, every code-2 entry admitted (wrong: ignores solids) | 43.2 | 57.8 | 58.0, 57.8 |
| base (`koc-seep-base`, 3 reps) | current | 42.2 | 59.8 | 59.0, 60.4, 60.2 |
| koc | fast twin traces every code-2 entry, retires it only when the hull ±12 box is certified, otherwise leaves it to the general twin | 42.7 | 60.9 | 60.8, 60.9, 61.0 |

Late seep (steps 36+), step time median / p95 in ms per repeat:
- koc base: 58.9/62.7, 57.0/62.1, 57.0/59.9
- koc candidate: 57.9/63.1, 57.5/61.8, 56.1/63.3

Closed rest, koc, step time in ms:

| arm | median per repeat | p95 per repeat | late-half median |
|---|---|---|---|
| base | 14.31, 14.13, 14.44 | 22.47, 16.10, 17.50 | 14.39, 14.18, 14.63 |
| koc | 14.46, 14.23, 14.25 | 17.22, 18.36, 16.37 | 14.61, 14.30, 14.51 |

Rest physics in all six koc runs was identical to the baseline to every digit
(volume, max speed, drift 0), with about 55% of the 4h traces taken by the
fast twin. That is evidence the ±12 certificate preserves results.

### Reading

- The most any code-2 fast path could save, even ignoring solids
  altogether, is 3.4 ms at 3000 h tiles (kob, 5.5%).
- The correct certificate admits only about half the traces. Speculating
  the rest (traced twice) costs more than it saves: koc is +1.1 ms at 3000 h
  tiles, slower in all three repeats, and flat at rest.
- koc is an optimistic bound on the correct design. The correct design
  also needs the general preflight to run first: a code-1 "preflight done"
  re-tag and two more launches.
- The only way to avoid the speculation would be to certify before the
  trace, which means a box of p ± (8 + travel + 12), over the proof's
  31-cell limit.
- Conclusion: no change kept. The lever is closed.

The koc candidate's third seep repeat stopped at step 68 with "a particle
arrived in page (54,29,33), which holds no storage; the residency closure is
too small". The base snapshot (`ko-seep-diagbase`, both repeats) hit the
same fatal at step 64. It belongs to the in-flight flip work, not the
surface stage. The fit uses the steps before it.

### The long-slide residency failure (bisected for the coordinator)

`tests/uniform-stage-residency.test.ts`, "long slide", fails with "empty
bottom-left page must retire" at its step-169 check. It passes at steps 1
and 30.
- HEAD af0e5dea plus the frozen uncommitted files: the smallest failing set
  is the particle sort/splat group (flip, splat, slab, canonical; they only
  compile together) plus `uniform-mixed-frame.ts` plus
  `uniform-near-settled.ts`.
- With near-settled forced to keep only one of its three new defaults:
  `sampleSeparation:"pairs"` alone fails; `innerReseed:"cells"` alone and
  `activationSeeding:"uncovered"` alone pass.
- Culprit: the `nbSeparation` pair push applied in the move,
  `if(!ballistic){end+=nbSeparation(...)}` in
  `uniform-narrow-band-flip.ts`. Frame forwards the option, and
  near-settled makes "pairs" the default.
- Logs: `artifacts/nbflip-step-time/bisect/`.

## 6. BULK16 step-time slice: whole-16h pages out of the 4h V-cycle (measured, nothing built)

The idea: keep every whole-16h page's 4h block resident with its page word
unchanged (no lattice or reader changes). Take the page out of the level-2
V-cycle lists, solve it at arena level 4 with the analytic coarse operator
(`nbflip-16h-progress.md` section 2), and prolong its 4h faces
conservatively. Before building it, I bounded what it can save.

**Upper-bound knock-out (snapshot only, `st-pko` against `st-pbase`).**
- pNative sets page flag bit 8 when all 8 region records are bulk members.
- `tPageLive` and the `rFlags` gate then skip those pages in every V-cycle
  page kernel: pDownB, pDown2, pDown3, pDown4, pUp4, pUp3, pUpB,
  pResidual2 and pNativeRelax.
- pNative and pWrite still run.
- This removes more work than the design would: the design keeps the
  page's level-4 rows and adds a prolongation.
- It changes physics, because a knocked-out page gets no correction. At
  rest, step 120 volume was 2949119.9809 against base 2949119.8728, max
  speed 0.012781 against 0.013472, and drift was 0. It is a timing bound
  only.

**Closed rest** (548 member regions, about 68 whole pages; 3+3 interleaved,
timing mode, median ms over steps 21-90):

| arm | r1 | r2 | r3 |
|---|---|---|---|
| base | 13.78 | 13.69 | 13.81 |
| KO | 13.54 | 13.54 | 13.75 |

The bound is 0.15 to 0.25 ms, about 3 µs per whole page, for the entire
V-cycle.

**Wide seep** (296 whole pages from step 35 on; 2+2 interleaved, timing
mode, per-repeat fit of step time on h tiles):

| arm | @1000 h tiles (ms) | @3000 h tiles (ms) | steps 5-20 median (ms) |
|---|---|---|---|
| base r1, r2 | 41.9, 40.9 | 60.1, 60.2 | 33.6, 33.1 |
| KO r1, r2 | 41.0, 41.3 | 56.4, 58.0 | 34.5, 34.7 |

- The intercept did not fall: 32.0 ms base against 33.2 ms KO. That is
  where a whole-page saving would show, since the page count is flat late.
- The difference at 3000 h tiles comes from the slope, and the KO repeats
  ran different h-tile paths, so it does not read as a whole-page effect.
- **Kernel timer** (check mode, steps 20-40, `pko-kern-{base,ko}`): the
  level-2 page kernels sum to 2.4-4.4 ms per step for all ~520 resident
  pages, timer on. So the whole-page share is at most about 1.7 ms with the
  timer on, and less in clean (overlapped) timing.
- On steps 20-32 the KO arena cost *more* (6-6.7 ms against 4-5 ms):
  uncorrected whole pages leave seam residual, so the convergence-gated
  cycles stop later. On steps 33-40 the two arms were equal.

**Reading.**
- The 43 µs per resident block (section 1) is a timer-on, serialised
  marginal from a width ladder. It is not per-page work that a whole-16h
  page can shed in clean timing.
- Removing whole pages from the largest per-block pass (the V-cycle, 7.7 of
  44 µs) bounds at about 3 µs per page clean at rest. The design would also
  pay its own level-4 solve and prolongation.
- Every other per-block pass in section 1 is 3.5 µs per block or less with
  the timer on. At 296 pages that is 1 ms or less each with the timer on,
  and less clean. The seep spread is about 1.6 ms at 3000 h tiles.
- So no single redundant pass can show a 3+3 win beyond spread, and the
  pressure slice was not built. The 12.7 ms BULK16 estimate in
  `nbflip-16h-progress.md` section 4 does not hold as a step-time number.
- **Later depth ladder** (`nbflip-16h-progress.md` section 4, "Depth
  ladder"). A whole page costs about 25 µs clean when every per-block pass
  is counted. No single pass costs more than about 4 µs.
  - So BULK16 is worth step time only as the whole design: up to about
    7.5 ms on the wide seep, and the difference between about 2.5 s and
    realtime at 100k bulk pages.
  - It cannot be landed pass by pass.
- Artifacts: `pko-rest-*`, `pko-seep-*` and `pko-kern-*`. The knock-out
  edits are saved as `pko.py.saved` (three replacements in the arena).

## 7. The fixed intercept: GPU busy against wall (measured)

The question: how much of the ~13.7 ms closed-rest step is kernel work,
and how much is gaps, launch overhead, readback stalls or CPU encode.

**Method.**
- A Metal System Trace (xctrace, launch mode) of the probes, run in a
  scratch copy with `use_user_defined_labels_in_backend`.
- One 1-blit marker command buffer is submitted just before `advanceTo` and
  one just after `awaitFrameCompletion`.
- Per step: the GPU-busy union of the solver's intervals between the
  markers, the head (start marker to first GPU work), the tail (last GPU
  work to end marker), and the gaps inside the GPU span.
- Medians over steady steps.
- Artifacts are in `artifacts/nbflip-step-time/xctrace/`: traces,
  `*-steps.json`, the scripts `xrec.sh`, `xan.py` and `xsteps.py`, and the
  probe patches.

| scene | wall | GPU busy | head (CPU encode) | tail (completion wake) | gaps inside |
|---|---|---|---|---|---|
| closed rest, 38 steps | 14.40 | 11.66 | 1.69 | 0.72 | 0.29 |
| wide seep, 21 steps (1000-1600 h tiles) | 45.0 | 40.4 | 2.80 | 0.77 | 0.33 |

All values are ms per step.

- **Command buffers.** A step is about 9 command buffers, and Dawn packs
  its ~390 dispatches into about 5 Metal encoders. The gaps between them
  are 15-60 µs, and nothing stalls mid-frame on a readback. Merging passes
  buys nothing measurable.
- **CPU per step.** JS record plus Dawn encode is 3.1 ms at rest and 4.2 ms
  on the seep (median). That is far below GPU time.
- **The app hides the head and tail.** It keeps up to
  `UNIFORM_MIXED_RECEIPT_RING` = 2 frames in flight, through `framePending`
  and `submitNextPreparedGPUAdvance`, so the next frame is encoded while
  the current one runs.
  - Measured with the opt-in `--pipelined` mode added to
    `probe-reservoir-bulk.ts`. Closed rest, 120 steps, interleaved, timing
    mode, median over steps 21-90:

    | mode | r1 (ms) | r2 (ms) |
    |---|---|---|
    | serial | 14.03 | 13.82 |
    | pipelined | 11.96 | 12.22 |

  - Step-120 physics was bit-identical in all four runs: volume
    2949119.8727798667, max speed 0.013471915386617184, drift 0.
  - So about 2 ms of the probe's rest intercept is the probe fencing every
    frame, not app frame time.
- **Dispatch floor.** In a microbench (`dispatch-bench.ts.saved`), a direct
  serial dispatch costs about 4-5 µs even with 1 workgroup or an
  immediate early exit.
  - Kernel-timer counts at rest: 389 dispatches per step, 106 of them in
    the arena solve. At ~4.5 µs each that is about 1.75 ms of the 11.7 ms
    GPU busy.
  - Only kernel fusion removes it. Arena fusion would be worth at most
    about 0.5 ms, and was not pursued.
- **Indirect dispatch is slower on this stack.** 450
  `dispatchWorkgroupsIndirect` calls with 0 workgroups took 5.4 ms (12 µs
  each, Dawn's indirect validation), against about 2 ms for 450 direct
  dispatches that exit early. The solver uses none, and converting
  empty-list launches to indirect would cost time.
- **Verdict.** No gap or launch overhead that can be fixed reaches 2 ms. No
  solver code changed.
- **Not pipelined.** The scale-ladder probe reads the page-pool receipt
  every step, so it stays serial.

## 8. Occupancy audit of the wide seep (measured, nothing built)

- **Method.**
  - One Metal GPU Counters capture, attached at steps 64-100 of the wide
    seep (5 mm, w2 d2, `bulkTier:16h`, `volumeControlSeconds:0`), at about
    2500-2800 h tiles. 37 steps were in the window.
  - The capture ran in split mode (`FLUID_PROFILE_SPLIT_DISPATCHES=1`, with
    `coalescePasses` off in a scratch snapshot only): one labelled pass per
    dispatch. Each pass then pays its own launch, so the GPU total was
    47.3 ms/step, against 40.4 ms GPU busy when coalesced (section 7).
  - Caps are Metal's `maxTotalThreadsPerThreadgroup`. They come from an
    offline compile of Dawn's dumped MSL (`dump_shaders,disable_symbol_renaming`)
    with the dumped `max_total_threads_per_threadgroup` attribute removed,
    which is the particle agent's method.
  - A cap of 384 is 12.5% occupancy (3072 threads per core); 1024 is the
    limit.
  - 42 of the 453 pipelines are capped at 384.
- **Top 20 by GPU time.**
  - Occupancy and limiters are mean percentages over each kernel's
    intervals. A pair in the cap column means two pipelines of that entry
    point.
  - Every row with occupancy at 16% or less has all of its limiters at or
    below 36%. Apart from pageMark, these are latency-bound on the register
    cap rather than throughput-bound.
  - pageMark reads at 85% of read bandwidth, so it is memory-bound.
  - Only pUpB, pDownB and pageMark have a buffer or MMU limiter above 25%.

  | kernel | ms/step | n/step | cap | occ | ALU lim | LLC | MMU | flag |
  |---|---|---|---|---|---|---|---|---|
  | advectDeferred (surface) | 4.05 | 1 | 384 | 9.3 | 14.8 | 20.4 | 4.4 | latency |
  | project (pressure) | 2.02 | 1 | 384 | 11.7 | 21.7 | 27.7 | 10.3 | latency |
  | coupleFine (surface) | 1.27 | 1 | 384/512 | 11.3 | 27.1 | 14.7 | 1.2 | latency |
  | cacheNormals (forces) | 1.22 | 1 | 448 | 14.2 | 25.5 | 18.1 | 2.4 | latency |
  | classify (census) | 1.19 | 1 | 448/1024 | 13.3 | 20.3 | 18.3 | 0.7 | latency |
  | advectParticlesDeferred (particle, not mine) | 1.18 | 1 | 384 | 7.2 | 10.3 | 9.9 | 1.9 | latency |
  | arenaPublishCanonical (surface) | 1.13 | 1 | 384 | 10.4 | 9.6 | 5.0 | 0.5 | latency |
  | localSpeed | 1.00 | 1 | 512 | 16.0 | 30.7 | 21.4 | 22.0 | latency |
  | arenaDistance (surface) | 0.93 | 1 | 576 | 16.6 | 35.7 | 8.2 | 2.7 | latency |
  | classifyCoarse (census) | 0.93 | 1 | 576/1024 | 12.3 | 21.7 | 21.2 | 14.9 | latency |
  | pUpB (arena) | 0.84 | 4 | 1024 | 28.7 | 40.4 | 30.3 | 9.7 | |
  | advectFine (surface) | 0.81 | 1 | 384 | 11.8 | 25.1 | 14.7 | 1.4 | latency |
  | pageMark (census) | 0.80 | 1 | 768 | 13.2 | 26.0 | 26.5 | 25.7 | memory |
  | cache (4h sampling) | 0.80 | 1 | 576/896 | 15.0 | 34.1 | 28.0 | 16.5 | latency |
  | separationPrepass (particle, not mine) | 0.78 | 1 | 832 | 3.2 | 7.3 | 1.3 | 0.4 | small grid |
  | pDownB (arena) | 0.75 | 4 | 832/1024 | 28.2 | 30.5 | 27.1 | 6.9 | |
  | advectFinish (surface) | 0.72 | 1 | 384 | 11.0 | 32.5 | 16.6 | 2.8 | latency |
  | pair_sweepSeams (extension) | 0.68 | 1 | 384/512 | 10.3 | 17.4 | 16.0 | 3.7 | latency |
  | pRelax (arena) | 0.64 | 25 | 1024 | 6.7 | 3.7 | 3.1 | 1.3 | small grid |
  | pUp3 (arena) | 0.63 | 12 | 896 | 18.8 | 18.6 | 7.9 | 2.2 | |

  - Rows marked "small grid" are capped at 832-1024 yet run at 3-7%
    occupancy, so they launch too few threads to fill the GPU. Their time is
    launch-shaped and the cap is not what limits them.
- **Register-cap leaves.**
  - To find them, each reachable function in the dumped MSL was stubbed on
    its own (`return {};`) and the kernel recompiled offline.
  - **advectDeferred (384, 267 functions).**
    - Stubbing `umSolidPresent` alone gives 1024.
    - That function is the scene's solid gate, `umSolidParams.lean.x<0.5`.
      With the gate false, the compiler removes every solid-aware branch.
    - Of the other leaves, only `umAdvected` moves the cap, to 512.
    - With all 14 large leaves stubbed together the cap reaches only 576. So
      no one cold leaf sets it: it is the solid-aware general path as a
      whole.
  - **coupleFine (384/512).**
    - `umSolidPresent` again gives 1024.
    - Many single leaves give 448-640.
  - **project (384).**
    - Stubbing `umProjectWithSource` (the inflow source average) or
      `umProjectV` gives 448.
    - The core lattice addressing (`umPageWord`, `umDirectoryPageWord`, the
      `umFace*` helpers) gives 832-896.
    - The geometry helpers (`umCellBodyFraction`, terrain, bodies),
      `umBulkRow` and `umDetachedMass` do not lift it past 448, alone or
      together with `umProjectWithSource`.
  - **arenaPublishCanonical (384).**
    - Many leaves give 448. `nbStoreAt` and `ubClass` give 704.
    - Stubbing `nbArenaUniform` gives 1024, but only because it folds the
      whole kernel to constants. That says nothing about a split.
- **Why nothing was built.**
  - The largest lift (to 1024) comes only from removing the solid path.
    The solver already has that split: the `umSolidFree` / `umSolidTwinned`
    pairs (coupleFine, advectDeferred and others). The GPU picks one of each
    pair per step from the scene-wide gate.
  - The wide seep is walled, so its gate is always on and the general
    pipeline does all the work. Only a per-tile solid-free twin would help
    here, and the coordinator rules twin pipelines out (e4041e9a): they
    compile different arithmetic.
  - The remaining single-leaf lifts are 384 to 448 (14.6% occupancy).
    - In project, the best one moves the inflow-source average into its own
      pass. But `value` feeds the release test in the same kernel, so the
      split would need a second face pass, plus a store and reload of every
      face value.
    - At these limiter levels, the gain is at most a few tenths of a ms,
      against a new pass and new traffic.
    - It does not meet the bar for a change, so it was not built.
  - **Standard fix, if twins are ever allowed.** Route solid-free tiles
    (not scenes) to the `umSolidFree` pipelines from a per-tile solid
    certificate. advectDeferred and coupleFine would then run at a 1024 cap
    over the open water. This is a twin by tile, and its repeatability would
    need the same audit as e4041e9a.
- **Artifacts.**
  - Directory: `artifacts/nbflip-step-time/xctrace/`.
  - `occ.trace` and `occ/*.ndjson` hold the capture.
  - `msl-audit/` holds the reducer (`xocc2.py`) and its output
    (`top-kernels.txt`). It also has the caps for all pipelines
    (`stats.txt`, `index.tsv`), the stub tools (`mstub.py`, `mdrop.sh`,
    `mloo.sh`), `mslstats.swift`, and the four kernels' MSL.

## Files changed (uncommitted)

- `lib/methods/uniform/uniform-mixed-pressure-authority.ts`: the
  `surfaceHeld` kernel (change 1).
- `lib/methods/uniform/uniform-pressure-arena.ts`: the `pCopyNative`
  kernel (change 1).
- `tools/probe-scale-ladder.ts`: `--stats-every=N` (default 1, as before).
- `lib/methods/uniform/uniform-local-residency.ts`: change 2, reverted to HEAD.
- `lib/methods/uniform/uniform-mixed-surface.ts`: the section 5 counters
  and candidates, all reverted. The file equals HEAD.
- `tools/probe-scale-ladder.ts`, `tools/probe-reservoir-bulk.ts`: their
  `--surface-work` hooks are removed. `tools/surface-deferred-work.ts` is
  deleted; a copy is in the artifacts.
- `tools/probe-reservoir-bulk.ts`: opt-in `--pipelined` (section 7). The
  default is unchanged.
- Section 8 changed no solver or tool file. Its probe and frame-plan
  instrumentation exists only in scratch snapshots.
