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

### C. Seep-style redistance (bound about 2 ms, and removes the 2-tile reach)

Replace lTiles, lDilate, lClear, lNear (±5 walk), lDistance and lPublish (2.8 ms) with the seep's redistInit plus 2
brick-local sweeps. Redistance is numerical maintenance, not the flow's physics. Its ±5 reach is also what forced
the two-tile halo that killed lever 1.
- **Go/no-go:** a scratch swap on frozen states, then the surface metric (rough, dimples025), D rows and drift.

### D. Fewer particles where nobody looks (bound about 2.5 ms per 100k records)

At about 25 ns a record, particle count is step time. Keep particles only within about 2 cells of the surface, plus
sheets and splash, and seed fewer per cell in calm h tiles.
- **Go/no-go:** the surface metric, D rows and a visual check.

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
