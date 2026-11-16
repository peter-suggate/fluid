# Speed-of-light review: the brick engine on the high-resolution dam (2026-11-16)

**Question (user):** how far is each slow stage from what an M1 Max can do for this scene, and what bold swings close
the gap? This review uses only measured numbers: lane profiles, frozen replays and the M0 microbenchmarks
(nbflip-structural §4.1). Estimates are marked as estimates.

## 1. What the scene actually asks of the GPU

Counts on main (attr4, tiered dam, 4913-brick lattice = 136³ h cells):

| step | resident F bricks (cells) | h-owned tiles (cells) | owned share of resident cells | particles |
|---|---|---|---|---|
| 60 (splash) | 4164 (2.13M) | 7670 (491k) | 23% | 642k |
| 165 | 1831 (937k) | 2808 (180k) | 19% | 369k |
| 300 (calm) | 1435 (735k) | 2291 (147k) | 20% | 337k |

At step 300 only 477 of the 1435 F bricks own any h tile. 854 are ring bricks and 74 are margin bricks: they exist
for stencil and trace reach.

**The work is small.** The live h band is 150–500k cells, plus about 60k C vertices and 340–640k particles. The
working set is about 100–300 MB of traffic a step, or 0.3–1 ms at M1 Max bandwidth.

## 2. Measured speed of light on this machine (M0 microbenchmarks, §4.1)

| operation | dense | through NB-FLIP's address chain | staged in workgroup memory |
|---|---|---|---|
| semi-Lagrangian MAC advect, 3 components, 96 taps | 1.1 ns a cell | 1.8 ns | 1.9 ns (cache already catches reuse) |
| RK3 particle move, 72 taps | 3.5 ns a particle | 4.2 ns | 2.9 ns |
| rebin (count sort) | about 1 ns × 3 passes a particle | | |

## 3. Per stage: today against speed of light

Today is the area profile at 06d4ca89 (steps 18–23, ms a step). Since then: lcAdvect −0.3 (3090d99e), the ring skip
−0.8 to −1.1 (71d4acc5). The fenced median is about 43–44 ms. Speed of light is the operation count × §2's dense
cost at step 100 (about 300k h cells, 450k particles), plus 2 sweeps of redistance and 10 pressure cycles.

| area | today | speed of light (estimate) | gap | measured cause |
|---|---|---|---|---|
| level set | 10.6 | about 1.5 | 7x | per-tap tier authority (nbVertexCoarse: 8 tiles through the directory a tap), the ±5 lNear walk family, 4 lResolve materialisations |
| velocity | 10.5 | about 1.0 | 10x | per-tap owner and face resolution in fgSampleVel (k6, one slot a sample: 2x on uAdvect), register-capped occupancy 13–20% |
| transfer and seeding | 7.5 | about 1.5 | 5x | per-record lookup chains, two seeding passes |
| move | 4.5 | about 1.5 | 3x | separation's 8 fgRun lookups, sampler taps |
| pressure solve | 7.0 | about 2.0 | 3.5x | first-touch reloads, iteration count (4 cycles against the seep's 1–2) |
| pressure setup and projection | 3.2 | about 0.5 | 6x | |
| brick set | 2.7 | about 0.4 | 7x | |
| **sum** | **46** | **about 8–9** | **5x** | |

**Every profiled kernel is latency-bound, not bandwidth-bound.**
- Read bandwidth is 1–3% of peak in lAdvect, lDistance, lcAdvect, uExtend, uAdvect and uTransfer.
- ALU looks busy (20–50%), but F32 is about 1%: it is integer and address work.
- Occupancy is 7–23%, set by registers. Code that never runs still costs. F's trace skip made uAdvect 2.3 ms slower
  with nothing taken.

So the cost is the shape of the data structure, not the amount of work:
1. Sub-brick (4³ tile) ownership means every tap of every stencil asks who owns it, often through the directory.
2. Generic kernels carry every tier's code path, so registers cap occupancy.
3. 80% of launched fine lanes have no owned work: ring bricks exist only to hold data for reach.

## 4. Bold swings, ranked by bound

### A. Resolve once, read flat (velocity and level set; bound about 6–9 ms)

The rule: tier authority is resolved **once per stage per value**, into the slot's own storage plus an apron. Every
kernel after that reads plain brick-local words with one slot lookup per sample, and carries no tier code.
- **Velocity samples:** F's FC_UNIT apron is in its gate now. Bound net of fills: 4.9 / 2.8 / 1.6 ms at steps
  20 / 100 / 300.
- **φ reads:** the same for lUnion, lDistance, lNear, uExtend and uTransfer's owner tests. Store resolved φ and the
  owner bit per vertex, so the 8-tile authority walk never runs in a hot loop.
- **Kernel specialisation by brick class:** compile two variants of each hot kernel, "interior" (all 8 tiles h, no
  tier code) and "seam", and dispatch each over its own list. Interior bricks are the majority of owner bricks. This
  attacks the register cap directly.
- **Go/no-go:** k6 is the velocity half's bound (measured). For level set, price the interior variant as a frozen
  knockout: tier code stripped, run on interior bricks only.

### B. A fused, brick-resident particle pipeline (bound about 6–8 ms)

Particles cost about 12 ms for 340–640k records, or about 25 ns a record. §2 says about 8–10 ns is reachable.
- Records stay brick-major between steps. Each step migrates only the records that left their brick, and does not
  re-sort all of them.
- One workgroup per brick chunk does G2P → move → separation → deposit:
  - the brick's velocity plus its apron is staged once in workgroup memory;
  - deposits accumulate in workgroup memory for the brick plus its apron;
  - one merge pass folds the apron deposits into neighbours, with no global float atomics.
- Heavy bricks (up to about 4k records) split into chunks that share one staging.
- **Go/no-go (cheap, from snapshots):** the per-step migration fraction and the per-brick record histogram. Then
  price the fused kernel with §2's staged costs.

**Result (lane E, 2026-11-16): no-go.**

How it was measured:
- Frozen tiered dam on tree 3090d99e, with no Metal aborts.
- One pMove ran from each snapshot, and the census was read from the records it moved and filed.

| census | step 20 | step 100 | step 300 |
|---|---|---|---|
| live records | 388k | 443k | 374k |
| records leaving their brick in a step | 0.90 | 0.39 | 0.12 |
| records leaving their cell in a step | 1.00 | 0.98 | 0.76 |
| travel in a step (cells): p50 / p99 / max | 6.3 / 10.1 / 19.0 | 1.9 / 5.8 / 8.7 | 0.6 / 1.4 / 5.4 |
| records travelling at most 2 cells | 0.1% | 58% | 99.4% |
| records a brick: mean / p99 / max | 629 / 2569 / 6421 | 568 / 2199 / 2932 | 845 / 1968 / 2274 |
| records a non-empty cell: mean / p99 / max | 6.0 / 32 / 217 | 5.8 / 17 / 155 | 6.1 / 15 / 81 |
| separation neighbours (the 2³ cells): mean / p99 | 69 / 289 | 53 / 133 | 51 / 95 |
| records whose deposit stencil leaves their brick | 0.66 | 0.70 | 0.67 |
| deposit nodes in other bricks | 0.24 | 0.26 | 0.24 |
| bricks touched over bricks owning | 2.20 | 2.33 | 2.22 |

The bound, using the frozen per-kernel times at steps 20 and 100:
- **Staged move.** The brick-plus-apron velocity stage misses exactly when the move is expensive. At step 20, 99.9%
  of records travel more than 2 cells, so the RK3 trace reads through the address chain anyway. Where it does hold,
  it saves the address chain's 4.2 ns against a staged 2.9 ns a record: at most about 0.5 ms.
- **Incremental migration.** Records leave their cell 76–100% of steps, and the separation needs cell runs. The most
  migration can save is the rebin itself: pScan + pScatter + pTiles + pArgs, about 0.36 ms.
- **Deposit fused into the move.** The deposit tests support on the new φ, and seeds deposit after the level set.
  Fusing the deposit ahead of the level set is a physics deviation. Without the deviation, the only saving is the
  record re-read, about 0.1 ms. The deposit's own cost is its liquid staging (0.65 of 1.0 ms), which fusion leaves
  in place.
- **pDelta fused with the next G2P** (the same q): about 0.2 ms.
- **Fusion-only total:** about 0.6–1.0 ms. Even with the standalone staging levers added, the total is about 3 ms,
  under the 4 ms bar.

Standalone levers, measured with frozen2 in the timing slot:

| lever | step 20 (ms) | step 100 (ms) | exactness | verdict |
|---|---|---|---|---|
| Staged separation in pMove: each workgroup stages its records' cells ±1 and their records as (position, mass) in 28 KB of workgroup memory | 3.09 → 7.00 | 3.92 → 8.29 | fmoved bitwise equal | parked: occupancy collapse, agreeing with lane P's sepF–I |
| Staged pSeed (lever S): see below | main 2.62 → 1.65 | main 2.55 → 1.56 | counts and probe words equal; 1 ULP in site depth | approved; lands on its gate |

Lever S, the staged pSeed:
- A brick that can seed stages its tile gates, its 4h bits and its 9³ FC_DEPTH vertices once.
- The count pass returns a site mask, so the write pass no longer re-tests each site.
- uncovered and seam are evaluated only near the band.

Counters at step 100:
- pMove: occupancy 18%, ALU 41%, last-level-cache limiter 22%.
- pSeed: occupancy 16–18%, ALU 25%, read bandwidth 1.7 GB/s, TLB miss rate 24%. pSeed is latency-bound on its
  dependent lookups, which is the latency lever S removes.

### C. Seep-style redistance (bound about 2 ms, and removes the 2-tile reach)

Replace lTiles, lDilate, lClear, lNear (±5 walk), lDistance and lPublish (2.8 ms) with the seep's redistInit plus 2
brick-local sweeps. Redistance is numerical maintenance, not the flow's physics. Its ±5 reach is also what forced
the two-tile halo that killed lever 1.
- **Go/no-go:** a scratch swap on frozen states, then the surface metric (rough, dimples025), D rows and drift.
- **Result (P, 87e777ce): no-go, and the premise was wrong.**
  - Only lNear, lDistance and lPublish are redistance. lTiles, lDilate and lClear are band, activity and collar
    bookkeeping that residency, seeding, extension and the C tier read.
  - The 2- and 3-tile reaches come from lDilate's crossing distance, not from lNear's ±5 walk.
  - A lower-bound probe (init plus 2 launches of 8 Jacobi Godunov sweeps) against lNear + lDistance + lPublish:
    - step 100: 2.21 ms with lNear dropped, 1.44 ms with lNear kept;
    - step 300: 1.38 ms and 0.87 ms.
  - Under the 1.5 ms bar at step 300, even in the best case.
  - Residual lever: lDistance alone is 1.0–1.7 ms. It could be cheaper with its semantics unchanged. Not pursued.

### D. Fewer particles where nobody looks (bound about 2.5 ms per 100k records)

At about 25 ns a record, particle count is step time. Keep particles only within about 2 cells of the surface, plus
sheets and splash, and seed fewer per cell in calm h tiles.
- **Go/no-go:** the surface metric, D rows and a visual check.

**Result (lane E, 2026-11-16): go, landed as b3b8cc2c.** The dam's particle band is 2 cells deep, through the scene
field scene.fluid.brickTiers.particleBandDepth (default 3).
- D8 is 20–33% lower: 4.35e5 / 3.12e5 / 3.01e5 against 6.54e5 / 4.04e5 / 3.69e5.
- Fenced medians are 35.6 ms against 39.1 ms (−3.5 ms, about 1.6 ms per 100k records).
- dam-300 passes, the surface metric is within the noise, and the visual check matches.

| census (frozen dam, h cells) | step 20 | step 100 |
|---|---|---|
| mass per cell, depth −2.5..−1 / −1..0 | 10–11 / about 8 | 10–11 / about 8 |
| max records in a shallow cell (band 3) | 152 | 125 |
| records deeper than −2 (band 3) | 20% | 24% |
| band 2, depth −1..0: records past the 2n cut (of all in the band) | 9.3k (of 86k) | 2.4k (of 134k) |
| band 2, depth above 0: records past the 2n cut, ballistic | 642, 216 | 341, 18 |

The shallow cells are uncapped (the old depth < −1 gate), and the census shows the uncapped hole is small.
- Extending the 2n cut to every cell drops 414 / 10 air records a step (90 / 0 ballistic). That is mass the step
  would lose, so a dropping cap is ruled out.
- A mass-conserving fold at depth −1..0 in liquid-mode cells would remove at most the excess stock: 9.3k / 2.4k
  records, 2.1% / 0.8% of the step's records. That is about 0.2 / 0.06 ms at the bound.
- **Parked**: the cut is small. The mass-8 cap is parked as well.

### The 4h tier's machinery (bound 4.83 ms at step 100; no-go against the 1.5 ms bar)

**Result (lane E, 2026-11-16): no-go.** The brick set, census, transfer and C tier cost 4.33 / 4.83 / 3.94 ms at
steps 20 / 100 / 300. The C kernels are latency-bound: each fits in one wave, so its time is its slowest lane's
chain. The exact options price at about 0.6 ms realistic and 0.9 ms upper bound. The full bound is in
progress-lane-e.md (2026-11-16).
- **Landed:** lcCouple and lcRetire at X8 (eight 64-lane workgroups a page, a vertex a lane, as lcAdvect). This is
  bit-identical against the G1 kernels on the frozen dam, and saves 0.34 / 0.33 / 0.39 ms at steps 20 / 100 / 300.
- **Tier churn is a quality signal as well as a cost one.** About 900–1000 tiles change width a step at step 100,
  and about 500 at step 300. Of each step's changes, 10–30% reverse the previous step's, and **about 45% reverse
  within six steps**. A tile that flickers between widths gets resampled each time: its level set, velocity and
  particles cross a seam twice for no lasting change in resolution.
- **Hysteresis stays parked.** It deviates from the tier rule. Its direct cost share (the transfer, about 0.35 ms,
  plus claims and leaves) is under the bar. Revisit it if the flicker shows in surface quality.

### E. Pressure (bound about 5–7 ms, needs the user's word)

The user said on 2026-11-08 that pressure was not worth optimising further (old engine). Here it is 10 ms of 46. The
seep's solver converges in 1–2 cycles where this one takes 4. Listed, not started.

## 5. Expected end state

| | today | after A | after A + B | after A–D |
|---|---|---|---|---|
| dam fenced median, steps 13–165 (estimate) | 43–44 | about 36–38 | about 29–31 | about 24–27 |

Speed of light is about 8–9 ms. What remains after A–D is pressure, brick-set machinery and the 4h tier's own C
level set.

## 6. Lanes

| swing | lane | first step |
|---|---|---|
| A, velocity | F | land the apron, then class-specialised uTransfer, uExtend and pDeposit |
| A, level set | E (after B's census) | resolved owner and φ words; interior/seam variants |
| B | E | migration and record census from snapshots, then a fused-kernel bound |
| C | P | frozen-state swap in scratch; F reviews (levelset-kernels.wgsl.ts is F's file) |
| D | E | after B's census; policy, judged by the surface metric |
| E | none | waits for the user |
