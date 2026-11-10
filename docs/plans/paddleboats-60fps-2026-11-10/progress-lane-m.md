# Lane M progress: multi-rate bookkeeping (2026-11-10)

**Verdict: NO-GO.** Running the bookkeeping every K steps cannot reach 5 ms net on paddleboats. The whole
deferrable set measures 6.3 ms of GPU time a step at step 160 and 6.7 ms at step 300, so even skipping it on
three frames in four saves at most 4.7 to 5.0 ms before the enlarged horizons are paid for, and it makes the worst
frame slower. No solver file was
changed. Step 2 was not started.

## 1. What was measured

GPU run 1 of 1 (rc 0, exclusive `timing` slot, no abort):
`tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/probe-reservoir-bulk.ts --scene=paddleboats --bodies --pipelined --dispatches --kernels=100 --steps=160 --every=160 --out=~/code/me/fluid-scratch/lane-m/kernels-paddleboats-160.json`

- Kernel timer on from step 100; kernel means over steps 100–160.
- State at step 160: 2,780 h tiles, 257k particles, 489 launches, drift −0.0085, max speed 2.99 m/s.
- The timed kernels sum to 49.3 ms a step. Wall time with the timer on was 40–47 ms over steps 60–100
  (pipelined), so the timed spans account for the whole frame: there is no large untimed gap between launches.
- The timestamps are quantised at about 65 µs (kernel medians fall on 0.786, 0.852, 1.049 ms). Per-kernel
  means under 0.1 ms are averages of 0-or-65 µs readings over 61 frames; family sums are sound, single small
  kernels are not.

**Second capture (the coordinator's, not a new run):** the 300-step paddleboats `--kernels` capture at
`/private/tmp/claude-501/-Users-petersuggate-code-me-fluid/091189a7-d1c8-43a2-8cbd-bd52f1e1bf21/scratchpad/boats-moored.json`
(3,526 h tiles, step median 51.9 ms, kernel sum 52.1 ms). The same family sums on it: relayout head 6.32 ms, region
tiers 0.34 ms, every-step 32.5 ms, pressure 10.6 ms. It has no per-pipeline launch counts, so the shared labels are
split with this run's counts.

The launch classes come from Lane B's ledger (`~/code/me/fluid-scratch/lane-b/ledger-paddleboats-30.json`, 489
launches) and from reading where each pass is encoded in `UniformMixedFrame.advance`, `encodeRelayoutHead`,
`encodePageCommit` and `mixedRelayout().encode`. Scripts: `~/code/me/fluid-scratch/lane-m/classify.py`,
`analyze.py`.

## 2. Classification of the launches

| Class | Launches | GPU ms a step (timer on) | What it is |
|---|---|---|---|
| Pressure (off-limits) | 203 | 9.65 | arena solve, rows, project, rhs, transfers, bulk tier, root owners |
| Diagnostic | 17 | 2.35 | global surface volume: 13 are the probe's last-step measurement, 4 are the gauge |
| (a) every step | 147 | 31.0 | move, transfer, reconstruction, surface advect, extension, forces, bodies, certificate, seed, slabs |
| (b) relayout head | 111 | 5.94 | census, page commit, layout build, remap, and what follows an adoption |
| (b) region tiers | 11 | 0.32 | region heat, region dynamics classify and publish, regional frame plan |

My pressure prefix list gives 203 where Lane B counted 215; the 12 between are the authority and transfer launches
that sit on the boundary. The post-relayout share of the geometry (5) and authority (4) launches is read from the
code, not counted separately, so class (b) is 111 ± 4.

### (b) the relayout head: 111 launches, 5.94 ms

It is one causal chain, encoded only when a relayout is due (`encodeRelayoutHead`). A frame without one already has
a code path (the `else` at `uniform-mixed-frame.ts:1119`), but it throws on a live page pool.

| Family | Launches | ms | Invariant it protects | Horizon change for K steps |
|---|---|---|---|---|
| Census (classify, classifyCoarse, decide, importance, interface, solid promote, page seed/request) | 13 | 2.38 | every tile a sample, a crossing, a body or the shape criterion needs at h is h before transport reads it | sample join box travel × K; body margin × K; `approachSteps` × K |
| Body tile mark (`markTiles`) | 1 | 0.06 | body-cut tiles are h | `bodyHorizon` = K dt (the parameter exists) |
| Page pool commit, BULK16 transfer and lifecycle, residency, block neighbours | 24 | 0.52 | a sample never arrives in a page that is not resident (a latched fatal) | residency travel bounds × K |
| Page initialise, slot init, new-block solid record, page words | 14 | 0.11 | a claimed block holds valid fields before any reader | none (follows the commit) |
| NB-FLIP activity and spread | 2 | 0.08 | heat and the seeded collar | cooling by K dt; collar wide enough for K steps of sample travel |
| Layout build | 12 | 0.42 | the generation's owners, slots and widths | none (follows the census) |
| Detail slots (claim, ring enter, release) | 6 | 0.12 | h storage for the generation's tiles | none |
| Remap, publish, adopt | 12 | 1.26 | fields carried onto the new generation | none |
| Frame bricks, phi twins, listed resolve, remap revisions, deep heat, widths | 13 | 0.21 | derived tables of the adopted generation | none |
| Post-adoption plan, geometry, authority | 13 | 0.78 | support, geometry and phase on the new layout | none |

### (b) region tiers: 11 launches, 0.32 ms

Region heat, region dynamics and the regional frame plan decide which 16h regions work this frame. They could run
every K steps with the HOT set dilated by K steps of the region's speed bound. At 0.32 ms the family is not worth a
mechanism.

### (a) every step, and (c) cannot be deferred

- **Physics (a), 31.0 ms:** particle move (4.4), momentum, P2G/G2P transfer, surface advect, reconstruction,
  redistance, extension, forces, body record and coupling, rigid integrate (3.7 ms, one workgroup), live copies
  (pressure warm start).
- **(c) Local speed certificate (8 launches) and the support plan (4):** they bound this step's transport from this
  step's velocity and phi. Both change in every pressure solve. A K-step version needs an acceleration bound that
  the solver does not have. Lane B's window already takes 16 to 7 here.
- **(c) NB-FLIP seed (9), slabs, placement, sort (7):** the seeds fill holes that this step's transfer reads; slabs
  are sized from this step's move.

## 3. Estimate

The brief priced a launch at about 70 µs whatever its work. The coordinator has since withdrawn that figure (the
pressure study measured 10 to 34 µs a removed launch), and **this measurement agrees.** The 70 µs row below is kept
only to show what the assumption would have claimed; the verdict uses measured kernel time. The 111 relayout launches cost 5.94 ms, 54 µs each on average, and the cost sits in a few kernels with
real work (classify 0.81, remapFaces 0.74, classifyCoarse 0.65, decide 0.37, pageMark 0.23 ms). The 46 pool,
BULK16, neighbour, initialise and slot launches total 0.75 ms, about 16 µs each. Both figures are below.

| | K = 2 | K = 4 |
|---|---|---|
| Launches removed a step, mean (122 × (K−1)/K) | 61 | 91.5 |
| Saving at 70 µs a launch (the assumption) | 4.3 ms | 6.4 ms |
| **Saving from measured kernel time, step 160 (6.26 ms × (K−1)/K)** | **3.1 ms** | **4.7 ms** |
| **Saving from measured kernel time, step 300 (6.66 ms × (K−1)/K)** | **3.3 ms** | **5.0 ms** |
| Extra h tiles from the enlarged horizons (estimate, not measured) | +150 to +400 | +600 to +1,500 |
| Cost of those at 3 µs a tile | 0.5 to 1.2 ms | 1.8 to 4.5 ms |
| Extra particles | few: join tiles hold none unless hot | collar ring 2 in fast tiles, unsized |
| **Net, mean** | **1.9 to 2.8 ms** | **0.2 to 3.2 ms** |
| Worst frame (the census frame) | today + 0.5 to 1.2 ms | today + 1.8 to 4.5 ms |

Where the tile estimates come from (reasoned from the code, not measured):
- **Sample joins.** `nbRefineBox` is the sample's reconstruction radius (0.875 cells), swept by one step of its last
  accepted motion, plus padding. K steps sweeps K times the motion plus g (K dt)²/2 (0.11 K² cells). At 1 m/s that
  is +0.7 cells for K = 2 and +2 cells for K = 4 on a 4-cell tile, so about a sixth and a half of the rim's
  neighbour tiles join. The rim today is 751 tiles.
- **Bodies.** `markTiles` dilates every body shape by (|v| + |ω| r) × horizon + 1 cell, in all directions. The wheels
  turn at 11.4 rad/s, 10.9° a step, with eight blades 45° apart. At K = 4 the sweep is 43.6°: the whole wheel
  annulus is body tiles. 754 tiles carry the solid reason at step 300.
- **Shape trigger.** A trigger that fires on a census frame only is up to K−1 frames late. That is a fidelity
  change, not a conservative horizon, unless `approachSteps` grows with K, which adds more tiles.

The verdict does not rest on these estimates. The gross saving is already under 5 ms.

**GO needs ≥ 5 ms net on paddleboats: the measured ceiling is 4.7 to 5.0 ms gross at K = 4, and 6.3 to 6.7 ms if
the bookkeeping never ran at all.**

## 4. Why staggering and event-driven skipping do not rescue it

- **Worst frame.** A cadence cuts the mean only. The census frame still runs all 122 launches and pays for the
  larger tile set, so the worst frame gets slower. For a 60 fps target that is a regression.
- **Staggering.** The relayout head is one chain: census → commit → build → remap → adoption. Splitting it over two
  frames (census on one, build and remap on the next) gives every frame half the chain, a 3.0 ms saving on every
  frame. But the census then reads a state one step older, so the horizon is three steps for a K = 2 saving. Net is
  under 2 ms.
- **Event-driven.** On paddleboats the census changes the layout on 90 of 90 frames, with 158 to 272 remap jobs a
  frame (`small-scene-step-2026-11-03/progress-relayout.md`). There is no quiet frame to skip. The solver already
  skips the census on an idle advance, decided on the CPU from lagged receipts without a stall.
- **Scene independence.** The mechanism itself is not scene-keyed, and seep and letters run the same head each
  frame. Their share was not measured (the one GPU run went to paddleboats).

## 5. Findings worth more than this idea

1. **The 70 µs-a-launch floor is not a per-launch cost for small launches.** With the timer on, 191 kernel keys
   under 0.1 ms sum to 5.3 ms; 89 keys under 0.02 ms sum to 0.9 ms. Removing tiny bookkeeping launches saves their
   measured time, about 10 to 20 µs each, not 70. This matches Lane B's fusion saving 9 launches for no visible
   time. Caveat: the 65 µs timestamp quantum makes each small mean noisy, and the timer breaks passes apart, so a
   timer-off cost per pass boundary is not excluded by this run.
2. **`GPU rigid-body integrate` is 3.7 ms as a single-workgroup launch**, the second largest kernel in the frame
   (median 3.74 ms, stable). That is 7.5% of the frame in one serial lane, more than the whole census.
3. Class (a) is 31 of 49 ms. The frame is physics work, with the particle move (4.4 ms) and rigid integrate on top.

## 6. Verified and assumed

- **Measured:** per-kernel GPU time with the timer on, one run, steps 100–160; launch counts per pipeline (Lane B's
  ledger at step 30, 489 launches; this run 489 at step 160).
- **Read from code:** which passes are encoded only under a relayout; the join box; the body margin; the fatal on a
  sample arriving in a non-resident page.
- **Assumed:** the extra-tile ranges in §3; 3 µs a tile (plan §1); that timer-on kernel sums rank the families as a
  timer-off frame would. The step-300 capture is the earlier lane's run on an older tree state; its 6.66 ms for
  class (b) puts K = 4 at exactly 5.0 ms gross, and under 3.2 ms net.
- **Not run:** seep, letters, any cadence prototype. No code changed, so types and unit tests were not re-run.

GPU runs used: 1 of 1 for step 1. Step 2's four were not used.
