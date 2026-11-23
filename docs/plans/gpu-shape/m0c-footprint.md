# GPU shape M0c (lane B): clock, footprint, achieved rate and slot locality in the real step

Lane B, 2026-11-23, for [§M0c](../gpu-shape-2026-11-23.md). The question: do the heavy kernels, which hold 63–96% of the step, sit on the collapsed side of lane A's random-gather curve? Lane A found 14.5 G lane-loads/s up to about 8 MB, a cliff at 32–40 MB, and 0.68 G/s at 256 MB and above. Lane A's M0c (m0a-shape.md §M0c) adds that 16 KB scattering moves the cliff down to about 1 MB, while workgroup-local 16 KB windows keep 6.6 G/s even at 1 GB.

**Short answer: no.**

- **Clock.** Real back-to-back steps run at the top clock.
- **Footprint.** The heavy kernels reach tens of MB at the live slots and 0.1–0.45 GB at the slots' high water. They never reach GB.
- **Hardware counters** put them far from any memory limit:
  - buffer-read limiter 0.5–8%, against 97–99% for every calibration chase;
  - MMU limiter 0.3–12%, against 76–84%;
  - L2-miss read traffic 0.2–22 GB/s;
  - compute occupancy 5–28%.
- **Slot order is spatially random.** A brick's 27-neighbourhood spans as many slots as the whole live range. Renumbering the slots in Morton order would cut its 16 KB pages 1.4–2.6×.
- *Inference:* working-set shape is not the first-order lever on the step's time. It is at most a latency term, through 10–39% TLB miss rates in some kernels.

## Method

### Tools

All three are in `tools/experiments/gpu-shape/`:

- **`footprint.mts`, MODE=probe.** Runs 160 normal back-to-back steps (`advanceTo`, no fence, no replay; fixed dt 1/60; paddleboats with `mooredMotorBodies`).
  - A clock probe is submitted before every step's part 1, between part 1 and part 2, and after part 2. On paddleboats the last one comes after the rigid integrator too.
  - The probe is lane A's kernel (`clock-probe.ts`): one 32-lane workgroup, a 2048-step dependent FMA chain, about 75.5 µs at the top clock, alone in a timestamped pass.
  - It binds the engine's `set` buffer read-only. Every part writes `set`, so Metal's hazard tracking orders the probe after the part before it, and the next part waits for it.
  - The probes are placed by wrapping `queue.submit` and the engine's `encode` (to tag its command buffers) from the tool. lib/ is unchanged.
  - Then, fenced, it dumps every engine's `set` and `dir` buffers, the layout, the buffer sizes and the schedule's kernels (hash and bindings).
- **`footprint-analyze.mts`.** Offline, no GPU.
  - **Locality.** For each live brick on F_ALL, it takes the slots of the brick and its live neighbours from record words 0–26. For C pages the neighbours come from the page window, because C records keep no neighbour words. It computes:
    - the slot spread, max − min;
    - the distinct 16 KB pages that those slots' 2080-byte regions cover in one channel (vel U0);
    - the same two figures after a hypothetical Morton renumbering of the live slots by their keys.
  - **Footprint per kernel.** It builds a call graph over the module's WGSL from the entry point, and lists the pool channels and lane-F regions (`NB_VEL_F_U0`, `FC_UNIT`, …) that the reachable functions name. A vector's channel 0 indexed with an axis term counts as 3 channels.
    - It sums those regions' bytes at the high-water slot (HIGH_F / HIGH_C / HIGH_K, the span the addresses can fall in) and at the live counts (LIVE_*).
    - Buffers that are not slot-indexed (`set`, `dir`, `census`, …) count whole.
    - Particle buffers count at the live particle count.
    - It also reports the allocated bytes of the kernel's bindings.
- **`footprint-ctr.mts`.** Hardware counters (Instruments' Metal GPU Counters, attached for 3 s; adapted from brick-tune/counters.mts).
  - It runs `footprint.mts` in MODE=counters: 100 warm-up steps, then split-pass steps. The split comes from `tools/gpu-kernel-timer.ts`: each dispatch is its own encoder, labelled.
  - After every step come four calibration chases. They use lane A's `chaseWGSL` (L256, D16, M1): random over 8, 32 and 256 MB, and coalesced over 8 MB, each in its own labelled, timed pass.
  - It reduces the trace with `tools/xctrace-hardware-limiters.ts`, which keeps only samples where one of our encoders runs alone, and time-weights per kernel entry.

### Runs

- Every GPU run went through `fluid-gpu-run.sh timing`, with rc 0 and 0 gpuErrors. Probe runs covered seep, dam, letters, paddleboats and teaser (detail 2). Counter runs covered dam and paddleboats.
- A first dam counter run was **discarded**. Its `collect()` destroyed resolve buffers that a step already encoded ahead still held, so those command buffers failed validation. The fix: the timer is off for one step before each collect.
- Commit: HEAD 226ac783. Other sessions' uncommitted files (webgpu-water-pipeline.ts, lane-n/*, water-caustic-receiver*, probe-inflow.mts) are untouched and are not on the engine path.

Data:

- Kept here: `docs/plans/gpu-shape/m0c/<scene>/`, holding `probes.tsv`, `analysis.txt` and, for dam and paddleboats, `counters.tsv` and `counters-loop.json`.
- In scratch: the set/dir dumps, traces and `meta.json` are in `~/code/me/fluid-scratch/gpu-shape/m0c{,-ctr}/`.

## 1. Clock state in a real step

Probe µs at steps 21–160 (139 steps × 3 probes; top clock = 75.5):

| scene | start med | mid med | end med | p90 (all) | > 1.1× top | > 1.5× top | worst | part 1 / part 2 GPU span (med) |
|---|---|---|---|---|---|---|---|---|
| seep | 76.5 | 76.7 | 77.2 | ≤ 77.8 | 6 / 417 | 0 | 112 | 4.36 / 14.24 ms |
| dam | 76.1 | 76.8 | 76.9 | ≤ 77.3 | 5 / 417 | 0 | 111 | 8.59 / 24.03 ms |
| letters | 75.8 | 76.8 | 76.9 | ≤ 77.2 | 3 / 417 | 0 | 111 | 11.44 / 31.08 ms |
| paddleboats | 76.3 | 76.6 | 77.1 | ≤ 77.7 | 8 / 417 | 2 | 139 | 8.45 / 25.88 ms |
| teaser parent | 76.5 | 76.8 | 76.8 | ≤ 77.2 | 12 / 834 (both engines) | 2 | 112 | 17.23 / 52.16 ms |
| teaser fine | 76.1 | 77.0 | 77.1 | ≤ 77.6 | (above) | | 131 | 2.07 / 9.24 ms |

- **Free-running steps hold the top clock.** The probe runs 0.3–2% above 75.5 µs at the median, and at most 3% of probes are more than 10% slow. The GPU gap between steps is 41–44 µs. For the teaser, the fine engine's gap is the parent's step.
- The rare slow probes (up to 139 µs, 1.8×) are isolated single readings.
- Lane A's 3.3× low state (about 250 µs) never appears, so the census times are top-clock times.
- *Caveat:* the probes add about 0.23 ms of ALU per step (about 1%). Whether they help hold the clock is not tested; lane A found ALU load holds it. The app paces steps on the frame clock, and these steps run back to back. All five steps are longer than a 16.7 ms frame anyway.

## 2. Footprint per heavy kernel

Census times are at step 100 (m0b). The state is at step 160.

- **Dam:** F live 1569, high 3793, cap 4913; C 97/125; particles 274k. Engine buffers total **862 MB**.
- **Paddleboats:** F live 1703, high 7122, cap 10240; C 274/274; K 276/676; particles 169k. Engine buffers total **1688 MB**.

Column meanings:

- **reach live / high:** the MB that the regions the kernel can name span, at the live counts and at the high-water slot.
- **alloc:** the allocated MB of all its bindings.
- **pages/nbhd:** the 16 KB pages that one brick's 27-neighbourhood covers over the kernel's brick-sized F channels.

The counter columns are the time-weighted means from the counter runs. They use Instruments' units (%, GB/s).

| dam kernel | ms | reach live / high MB | alloc MB | pages/nbhd | occ % | ALU lim | buf-read lim | LLC lim | MMU lim | TLB miss | read GB/s | MB/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lAdvect | 1.96 | 62 / 144 | 255 | 358 | 8.4 | 19.6 | 3.0 | 9.8 | 0.8 | 20.8 | 2.3 | 4.5 |
| uExtend | 1.63 | 45 / 98 | 260 | 228 | 22.1 | 27.8 | 1.2 | 6.0 | 1.9 | 33.2 | 5.1 | 8.3 |
| uTransfer | 1.59 | 77 / 143 | 436 | 288 | 9.5 | 16.7 | 1.9 | 9.2 | 3.0 | 17.6 | 6.3 | 10.0 |
| pMove | 1.22 | 109 / 194 | 445 | 378 | 13.0 | 25.3 | 7.9 | 23.9 | 12.5 | 13.4 | 21.8 | 26.5 |
| uAdvect | 1.21 | 56 / 128 | 255 | 316 | 16.5 | 30.9 | 6.9 | 11.3 | 1.8 | 20.2 | 4.8 | 5.8 |
| pSeed | 1.20 | 89 / 171 | 375 | 358 | 12.9 | 11.0 | 0.9 | 7.6 | 0.9 | 15.2 | 2.8 | 3.4 |
| nbOperator | 1.17 | 100 / 236 | 422 | 608 | 16.3 | 21.7 | 0.5 | 6.4 | 1.9 | 38.6 | 2.5 | 2.9 |
| pSeparate | 1.13 | 60 / 79 | 351 | 83 | 5.2 | 10.6 | 3.3 | 8.2 | 2.4 | 3.6 | 7.1 | 8.1 |
| censusDecide | 1.11 | 16 / 21 | 44 | 21 | 6.1 | 5.4 | 1.0 | 8.5 | 0.3 | 4.4 | 0.7 | 0.8 |
| censusFine | 0.94 | 26 / 44 | 138 | 83 | 18.6 | 30.5 | 1.8 | 17.7 | 0.8 | 25.9 | 2.3 | 2.2 |
| lDistance | 0.92 | 16 / 36 | 161 | 83 | 24.2 | 56.9 | 4.9 | 5.2 | 2.9 | 15.4 | 7.6 | 7.0 |
| pmgUpdate | 0.75 | 80 / 181 | 305 | 443 | 27.6 | 34.2 | 4.1 | 12.7 | 5.8 | 37.9 | 15.4 | 11.5 |

| paddleboats kernel | ms | reach live / high MB | alloc MB | pages/nbhd | occ % | ALU lim | buf-read lim | LLC lim | MMU lim | TLB miss | read GB/s | MB/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nbOperator | 2.46 | 118 / 455 | 1037 | 340 | 15.6 | 25.2 | 0.7 | 6.8 | 2.3 | 26.1 | 3.1 | 7.6 |
| lAdvect | 2.40 | 76 / 272 | 537 | 200 | 13.4 | 28.5 | 5.0 | 17.2 | 1.1 | 10.1 | 3.2 | 7.7 |
| uExtend | 1.90 | 62 / 189 | 552 | 127 | 24.1 | 34.0 | 1.9 | 6.9 | 2.5 | 19.1 | 6.5 | 12.4 |
| censusDecide | 1.75 | 41 / 53 | 104 | 12 | 13.5 | 42.3 | 7.0 | 3.3 | 0.3 | 4.3 | 0.7 | 1.2 |
| uTransfer | 1.73 | 76 / 234 | 709 | 161 | 14.1 | 24.2 | 3.6 | 14.9 | 5.1 | 12.2 | 9.9 | 17.1 |
| uAdvect | 1.30 | 69 / 242 | 537 | 177 | 20.5 | 42.4 | 7.1 | 12.7 | 1.8 | 9.0 | 4.5 | 5.9 |
| lcAdvect | 1.14 | 73 / 258 | 537 | 188 | 13.4 | 22.1 | 1.9 | 16.9 | 0.5 | 6.8 | 1.7 | 1.9 |
| lcRetire | 1.00 | 19 / 55 | 339 | 35 | 12.2 | 21.1 | 1.8 | 11.9 | 0.7 | 2.8 | 1.2 | 1.2 |
| censusFine | 0.97 | 52 / 97 | 302 | 46 | 19.6 | 39.2 | 2.5 | 19.4 | 1.7 | 12.2 | 4.8 | 4.7 |
| bodyStamp | 0.86 | 14 / 31 | 221 | 12 | 17.9 | 29.1 | 2.4 | 5.8 | 0.4 | 13.8 | 0.2 | 0.2 |
| project | 0.80 | 84 / 320 | 1037 | 236 | 19.9 | 25.2 | 1.8 | 10.7 | 3.9 | 21.5 | 9.4 | 7.5 |
| lDistance | 0.73 | 22 / 69 | 339 | 46 | 23.4 | 54.3 | 3.9 | 5.7 | 2.4 | 7.3 | 6.2 | 4.5 |

The calibration chases ran in the same windows, with their timed rates in G lane-loads/s:

| chase | rate, dam / paddleboats run | occ % | ALU lim | buf-read lim | LLC lim | MMU lim | TLB miss | read GB/s |
|---|---|---|---|---|---|---|---|---|
| random 8 MB | 10.5–13.6 / 11.6–13.5 | 88–90 | 2–3 | 97–98 | 63–68 | 76–78 | 1–2 | 175–194 |
| random 32 MB | 6.2–6.7 | 86–87 | 1 | 98–100 | 47–48 | 82–84 | 78 | 101–104 |
| random 256 MB | 0.75–0.76 | 33–39 | 0.1 | 98 | 51–53 | 77–78 | 96–97 | 12 |
| coalesced 8 MB | 191–196 | 83–86 | 36–37 | 90–93 | 62–65 | 69–72 | 4 | 169–173 |

Rates are given as median–min. They agree with lane A's 14.5 / 6.8 / 0.68 / 212 G/s.

What the footprint tables show:

- **Footprint: tens of MB live, hundreds of MB of address span, never GB.**
  - Live-slot reach is 6–109 MB on dam and 14–118 MB on paddleboats. That is in or around lane A's 32–40 MB cliff, if the gathers were random.
  - The high-water span is 11–236 MB on dam and 31–455 MB on paddleboats. Paddleboats' HIGH_F is 4.2× LIVE_F, and dam's is 2.4×.
  - The biggest single regions are:
    - vel, at 73 MB on dam and 138 MB on paddleboats at high (9 F + 9 C channels × 2080 B × slot);
    - pmgF (106 / 198 MB);
    - pres (87 / 164 MB);
    - fcell's UNIT (8.7 KB a slot).
  - The bindings' allocations are 34–1037 MB, and the engines total 0.86 / 1.69 GB. Most of the allocation is capacity that no step reaches.

What the counters show:

- **Achieved rate: the memory system is mostly idle.**
  - Every heavy kernel's buffer-read limiter is 0.5–8% and its MMU limiter 0.3–12.5%. All four chases sit at 90–100% and 69–84%, whether at 8 MB or 256 MB.
  - Their GPU read bandwidth is 0.2–22 GB/s. In the calibrations this counter reads about 14–16 bytes per random lane-load (194 GB/s at 13.6 G/s; 12 GB/s at 0.76 G/s). So it measures traffic past the GPU L2, not payload.
  - *Estimate:* if all of a kernel's L2-miss traffic were random 4-byte gathers, it would be 0.01–1.5 G missing lane-loads/s. That is below even the collapsed 0.68 G/s for every kernel except pMove (about 1.5). Per call, the traffic past L2 is 0.2–27 MB, against a reach of 6–118 MB.
- **What the kernels look like instead** (*inference*): latency- and issue-bound at low occupancy.
  - Compute occupancy is 5–28%, against 83–90% for the chases.
  - The ALU limiter is 5–57%, and nothing else is near a limit. This fits the register-bound caps (384–832 threads; brick-register-bound memory) and dependent gathers with few loads in flight.
- **The TLB miss rate is high in some kernels** (*inference*): 15–39% in nbOperator, uExtend, pmgUpdate, censusFine, lAdvect, uAdvect and project, against 1–4% for the 8 MB random and coalesced chases.
  - The MMU is not saturated (≤ 12%), so this is a per-miss latency cost on dependent loads, not a throughput collapse.
  - Fewer pages per neighbourhood would lower it. How much time that buys is not measured.

### Loads per lane (static)

Not counted. A static count of the WGSL is not meaningful for these kernels. For example, lAdvect returns early for kept far or deep air, and runs a 10-tap gradient, a trace, a trilinear or cubic sample and wall probes only near the surface. So its loads per lane depend on the data.

`analysis.txt` lists the static load sites per binding (each indexing once, loops not unrolled) only as an inventory. The counter-based rate above replaces this count.

## 3. Slot-order locality

### Histograms

**F bricks.** F_ALL at step 160. "nbhd" is the brick and its live record neighbours (16–21 bricks on average). The pages are 16 KB pages of one channel.

| scene | bricks | slot spread p10 / p50 / p90 | HIGH_F | pages / nbhd mean (p50, p90) | pages a brick | Morton: spread p50 / p90 | Morton: pages mean (p90) |
|---|---|---|---|---|---|---|---|
| seep | 1169 | 692 / 1036 / 1196 | 1281 | 15.3 (15, 21) | 0.79 | 105 / 442 | 7.3 (11) |
| dam | 1569 | 2591 / 3354 / 3620 | 3793 | 20.7 (20, 28) | 1.01 | 231 / 760 | 8.0 (11) |
| letters | 4408 | 2720 / 3764 / 4378 | 4739 | 19.9 (20, 27) | 0.93 | 287 / 2279 | 8.4 (12) |
| paddleboats | 1703 | 448 / 3891 / 5999 | 7122 | 11.6 (12, 15) | 0.55 | — | 8.2 |
| teaser parent | 5484 | 3128 / 5644 / 6219 | 6451 | 20.3 (20, 28) | 0.95 | 192 / 1060 | 8.2 (12) |
| teaser fine | 255 | 197 / 361 / 453 | 464 | 11.7 (11, 16) | 0.76 | 31 / 50 | 4.9 (6) |

Page histograms (bins ≤ 4 / 8 / 12 / 16 / 20 / 24 / 27 / 32 / 40 pages):

- dam: 0 / 16 / 122 / 269 / 401 / 270 / 257 / 223 / 11; Morton: 143 / 763 / 616 / 47 / 0 …
- paddleboats: 11 / 332 / 654 / 617 / 80 / 9 / 0 …
- The rest are in `m0c/<scene>/analysis.txt`.

**C pages.** C_ALL; the neighbours come from the page window.

| scene | pages | spread p50 / p90 | pages / nbhd mean (p90) | Morton: pages mean (p90) |
|---|---|---|---|---|
| seep | 237 | 152 / 208 | 8.3 (13) | 6.2 (10) |
| dam | 97 | 85 / 119 | 7.9 (11) | 5.7 (8) |
| letters | 198 | 126 / 181 | 9.3 (13) | 6.3 (10) |
| paddleboats | 274 | 192 / 256 | 8.6 (12) | 5.5 (8) |
| teaser parent | 316 | 179 / 322 | 10.8 (16) | 6.5 (10) |

### Reading

- **The slot order is spatially random.** A brick's neighbours span 80–90% of the live slot range (dam p50 3354 of HIGH_F 3793). On dam, letters and the teaser, every brick of a neighbourhood lands in its own 16 KB page (0.93–1.01 pages a brick). That is the "16 KB scattered" regime of lane A's table.
- **Morton renumbering** of the live slots would cut the pages per neighbourhood 2.4–2.6× on dam, letters and the teaser, 2.1× on seep, and 1.4× on paddleboats (whose fresh allocations are already partly sequential).
- Per kernel, one brick's neighbourhood touches the following 16 KB pages over all its brick-sized F channels, with the Morton figure in brackets:
  - dam: 83–608 (32–235);
  - paddleboats: 12–340 (8–236).
- *Inference:* each workgroup gathers from a few hundred scattered pages, not from a handful of windows. The counters above still show no MMU or LLC saturation.

## 4. How slots are assigned (allocator)

- **Free stack, LIFO.**
  - `nbPop` (brick-set.wgsl.ts:54) is an `atomicSub` on TOP and returns `FREE[top−1]`; it also does `atomicMax` on HIGH.
  - `nbPush` (:59) is an `atomicAdd` on TOP and stores the slot at `FREE[top]`.
  - The boot fills `FREE[i] = CAP−1−i` (brick-engine.ts:779–783), so a fresh engine pops 0, 1, 2, ….
- **Claim order is candidate order, which is atomic-append order.**
  - Candidates are appended by `atomicAdd` on `N_CAND`. This happens in censusDecide (brick-census.wgsl.ts:1585), the residency request (brick-residency.wgsl.ts:135) and the static need (brick-static-need.wgsl.ts:173).
  - `setClaimPages` (brick-set.wgsl.ts:264) and `setClaimBricks` (:293) walk the candidates in that order and pop one slot each.
  - So the first fill follows the census's workgroup race order. That order is spatial only to the extent the census lists are.
- **Freed slots go back in race order.**
  - `setLeave` (:324, push at :335) and `setLeavePages` (:347, push at :358) push leaving slots, from F_ALL / C_ALL order, onto the stack.
  - The next entering brick takes the most recently freed slot, wherever it is. After churn, a slot says nothing about position. At step 160 the next F pops on dam are 3041, 3280, 3269, 2023, 3409, ….
  - Holes accumulate below HIGH: dam has 2224 holes for 1569 live bricks, and paddleboats 5419 for 1703.
- **Lists are in near-slot order.** `setLists` (:501) scans slots 0..HIGH and appends with `atomicAdd`, so F_ALL has only 15–100 descents in 255–5484 entries.
  - A kernel therefore walks bricks in slot order, which is random in space. Consecutive workgroups do not share neighbourhoods.
- **C records hold no neighbour words.** `setClaimPages` sets words 0..26 to NONE, so a page's neighbours come only through the page window.

## 5. Read: is working-set shape the lever?

*Inference, from the measurements above.*

- **Not the first-order lever on the heavy kernels' time.** They gather over tens of MB live and hundreds of MB of span, through a spatially random slot order: the shape lane A showed collapses a pure random chase. Yet the counters show them nowhere near the memory system's limits:
  - buffer-read limiter ≤ 8%;
  - MMU limiter ≤ 12.5%;
  - LLC limiter ≤ 24%;
  - traffic past L2 ≤ 22 GB/s.
  Meanwhile occupancy is 5–28%. The step is latency- and issue-bound per resident lane, as the brick-register-bound memory already said. It is not bound by the throughput of gathers from DRAM or by translation.
- **A second-order latency term is plausible.** TLB miss rates are 15–39% in the vel- and pres-heavy stencils (nbOperator, uExtend, pmgUpdate, uAdvect, lAdvect). With few loads in flight, each miss's page walk lands on the critical path. Morton or blocked slot order (2.4–2.6× fewer pages a neighbourhood) and compacting HIGH toward LIVE (2.4× on dam, 4.2× on paddleboats) are the cheap ways to shrink it.
- **A go/no-go that would measure it directly** (not run): permute the frozen state's F slots into Morton order offline, then rerun the census replay on the same state. That means the pools, fcell and fdep regions, pmgF, the set's per-slot arrays, dir records, child words and lists, and the particles' slot fields. If the heavy kernels gain less than about 10%, drop working-set shape and attack occupancy and latency per lane instead.
- **Clock is not a factor.** Back-to-back steps stay at the top clock, so the census's times need no DVFS correction.

## Caveats

- The counter runs use split-pass steps (one encoder per dispatch, as in the census replay), and counters are device-wide. The reducer keeps only samples taken while one of our encoders runs alone, with a full-sample guard. Its percentages are time-weighted means over each kernel, so short phases inside a kernel are averaged away.
- "GPU Read Bandwidth" is read here as traffic past the GPU L2, which includes SLC hits. This reading comes from the calibrations, not from documentation.
- Reach is an upper bound on what the reachable code can name. Which cells and neighbours a kernel actually touches depends on the data.
- The page counts treat one channel's slot region as a virtually contiguous run. The physical layout and the real MMU page size are not observed (see lane A's M0c caveats).
- The census times are from step 100. The dumped state and the counters are from steps 100–330: the 160-step state dump, and about 230 counter-loop steps after a 100-step warm-up.
