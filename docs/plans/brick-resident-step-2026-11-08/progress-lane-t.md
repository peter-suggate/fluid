# Lane T progress: B1 transport, B2 surface, B3 momentum to RHS

Design: `docs/plans/brick-resident-step-2026-11-08.md`. Lane E owns ubaStage, B4, the census and T.

## LEARNINGS (wrap-up, 2026-11-08)

Unless stated otherwise, numbers are kernel ms from single runs with raw timestamps.
- "drop" means 8x8x4 drop=2 at step 29.
- "seep" means the median of steps 196-199.
- Seep timestamps are quantized to 0.066 ms.

The primary metric is the switch-on cost: drop's non-pressure excess over rest. Seep is the steady-state guard.

### What landed

| commit | change | drop, before → after | seep, before → after |
|---|---|---|---|
| 1cbebf88, S3 | Particle sort as one counting sort per brick | sort family 0.225 → 0.126 ms (slabSort 0.047) | slabSort 0.197 ms |
| 8822b2d8, S2 | Momentum as one claimed launch, the COLD retain kept | momentum 0.297 → 0.116 ms | advectMomentum 1.18 ms |

- **Together at drop:**
  - B1 family: 2.177 → 1.823 ms.
  - Non-pressure excess: 8.870 → 8.706 ms.
  - Pipelined drop16 step: 16.47 → 15.69 ms.
- **Other scenes:**
  - seep: 31.6 → 31.0 ms.
  - letters: 58.0 → 54.0 ms.
  - rest16: 3.79 → 3.28 ms.
- Drift stayed in band.
- Cold compile: advectMomentum 1.1 s, slabSort 0.2 s.

### What failed, and why

- **S1, one surface-advect launch.** It cut the surface family from 0.780 to 0.666 ms at drop, but took 640 s to compile cold.
  - Mechanism, known: Metal inlines every call site, and umAdvected's solid body grows superlinearly. See Lane C.
  - It is withdrawn until Lane C's structure lands.
- **S1-S3 seep regression (+3.8 ms, 5% fewer particles, phi source flipped).** S1's mechanism was not isolated. S3 alone and S2+S3 are clean.
- **S4, surface compaction queue.** The coordinator ruled it launch tuning and it was reverted. It was never measured at seep.
- **m1, m3 and m4, windowed or staged momentum, slower than S2.**

  | variant | seep | drop |
  |---|---|---|
  | S2 | 1.18 ms | 0.116 ms |
  | m1 | 1.51 ms | not measured |
  | m3 | 1.38 ms | not measured |
  | m4 | 1.51 ms | not measured |

  - **The mechanism is unknown.**
  - The window-only bound (P8: every tap from threadgroup memory, misses read 0) took the h jobs only from 0.72 to 0.59 ms. So the h jobs are not bound by global load latency.
  - Hardware counters exist only for S2 at drop: momentum repeated alone, `kernel-counter-bench`, at 188 µs a launch.

    | counter | mean | p90 |
    |---|---|---|
    | compute occupancy | 7.8% | 14.6% |
    | ALU limiter | 11.9% | 26.8% |
    | LLC limiter | 3.8% | |
    | texture filtering limiter | 2.5% | |
    | buffer read limiter | 0.6% | |
    | threadgroup load and store | below 1% | |
    | MMU TLB miss rate | 24% | |

    Its h-jobs-only probe gave about the same: occupancy 8.7%, ALU 12.9%. No unit is near its limit. At drop, momentum is starved: about 117 h jobs of 192 lanes, about 22k threads against 98k.
  - Threadgroup memory per 192-lane group, from the WGSL declarations (static, not measured):

    | variant | momentum's own arrays |
    |---|---|
    | S2 | about 1.3 KB |
    | m3 | about 1.8 KB |
    | m4 | about 7.9 KB, adding umWin's 6 KB |

    At 32 KB per core, m4 holds at most 4 groups, 768 threads, per core. That would explain m4 at seep if threadgroup memory is the limiter, but this is not measured. It cannot explain m3, whose extra 0.54 KB is small.
  - **Hypotheses still open:**
    - (a) occupancy from threadgroup memory or registers;
    - (b) both paths staying live on a miss;
    - (c) barrier stalls, or few traced faces per 512 staged texels;
    - (d) bank conflicts;
    - (e) divergence in the trace length;
    - (f) compiler scheduling.
  - Tools for the open hypotheses are built but were not run:
    - counter trees for S2, m3 and m4 at drop and seep;
    - a census probe of segments per traced face, the longest trace per SIMD group, and traced faces per job. It is `trees/m4c`, umProbe 2.

### Two regimes

| | drop | seep |
|---|---|---|
| bound by | chain length | throughput |
| why | every B1 kernel runs about one wave | about 6 residency waves |

- **advectParticlesDeferred:** 175 jobs of 64 records against 1024 workgroups at drop; 61k particles at seep.

  | | drop | seep |
  |---|---|---|
  | time | 0.18 ms | 1.51 ms |
  | per particle | 16 ns (11.2k particles) | 25 ns (61k particles) |

- **What a fix needs:** a shorter dependent chain per job helps drop; fewer registers or more occupancy helps seep.
- **Particle-move split at seep:** measurement launches q1-q3 before the real move. The real move was 1.44 ms in this run.

  | part | ms |
  |---|---|
  | q1, job and window staging only | below 0.066 |
  | q2, + Heun's three velocity samples | 0.131 |
  | q3, + depth, heat, cold, support, blocked, endpoint | 0.262 |

  - **82% of the move lies outside q3:**
    - G2P and the retire logic;
    - the separation read;
    - nbHullUncut and nbFinish's solid walk (seep has solids);
    - the rigid contact;
    - the appends;
    - or the occupancy cost of the full kernel's registers.
  - These were not separated, so that 82% is unknown.
  - **At drop, step 29:** one sample a kernel, quantized to 16 µs.

  | part | ms |
  |---|---|
  | q1 | 0.033 |
  | q2 | 0.016 (noise: below q1) |
  | q3 | 0.082 |
  | real move | 0.180 |

  About 55% of the move at drop also lies outside q3.

### Drop excess by kernel (S2+S3 tree)

- **Totals:** excess 11.99 ms, of which pressure 3.29 ms and non-pressure 8.71 ms.
- **Spread:** 192 kernels run only at drop.

  | kernel size | count |
  |---|---|
  | 100 µs or more | 19 |
  | 50-100 µs | 36 |
  | 20-50 µs | 54 |
  | 10-20 µs | 37 |
  | under 10 µs | 46 |

- **Non-pressure excess by module:**

  | module | ms |
  |---|---|
  | extension | 0.97 |
  | ownership census | 0.72 |
  | surface advect (+ deferred) | 0.79 |
  | remap | 0.37 |
  | NB transfer | 0.36 |
  | surface reconstruction | 0.29 |
  | layout build | 0.29 |
  | canonical metric | 0.29 |
  | geometric fill | 0.29 |
  | pressure authority | 0.29 |
  | coarse retirement | 0.28 |
  | particle advect | 0.28 |
  | local speed | 0.27 |
  | phi resolve | 0.24 |
  | seed | 0.24 |
  | body forces | 0.22 |
  | others | about 50 modules under 0.2 each |

- **B1 family at drop: 1.82 ms over 31 kernels, about 60 µs each.**

  | kernel | ms |
  |---|---|
  | advectDeferred | 0.29 |
  | resolve | 0.20 |
  | advectParticlesDeferred | 0.18 |
  | localSpeed | 0.15 |
  | advectCubic | 0.12 |
  | advectMomentum | 0.12 |
  | advectWalls | 0.08 |
  | advectFine | 0.07 |
  | separationPrepass | 0.07 |
  | advect | 0.07 |
  | advectFinish | 0.05 |
  | slabSort | 0.05 |
  | the rest | under 0.05 each |

### Cold share at drop (step 32, 8x8x4)

- **Regions:** 492 COLD, 48 WARM, 4 HOT, so 90% COLD.
- **Level-0 bricks of the fused pass:** 144 in all.

  | tier | bricks |
  |---|---|
  | COLD | 20 (14%) |
  | WARM | 92 |
  | HOT | 32 |
  | deep-COLD | 0 |

  No particles are in cold bricks.
- **Level-2 pages:** 68 in all.
  - 52 have all 8 regions COLD (76%).
  - 15 are partly COLD.
  - 46 are deep-COLD (68%).
- **What a cold skip would save.** The level-2 jobs already skip deep-COLD owners: the momentum 4h jobs enumerate only the active regions. COLD tiles already retain per tile.
  - What is left is the level-0 COLD share, 14%. That is at most about 0.1 ms of the B1 family at drop.
  - The level-2 share is largely skipped already.
- 16x8x4 was not measured: the run was stopped at wrap-up.

### What remains

- The fused B1 pass on the real traces: walls, seams, freeze, the substep trace, and the particle walk. It needs Lane C's surface structure. Measure it at drop against the sum of the launches it replaces.
- The counter-based mechanism for m3 and m4, and the census of trace length per SIMD group.
- The 82% of the particle move not yet attributed.
- B2 and B3 were not started.

### Recommendations, ranked by expected gain at drop

1. **Fuse B1 into one claimed pass on the real traces, after Lane C lands.**
   - At drop, the 31 kernels cost about 60 µs each of mostly chain and launch floor, against single-digit-µs work: 117 h tiles and 11k particles.
   - The real cost of one pass would be the longest job chain, plus a floor of about 40 µs. The fused bench, with simplified traces, put both levels at 0.28-0.35 ms.
   - Expected: 1.82 → about 0.5 ms, so **-1.3 ms at drop**. The risk is the surface trace's compile time and length; hold it to 5 s or less per pipeline.
2. **Treat the switch-on long tail as a structure problem, not per-kernel tuning.** 192 kernels switch on at drop, a mean of 42 µs and 8.7 ms in all. Every stage family pays a launch and wave floor at drop.
   - The biggest families outside B1: extension 0.97 (Lane E, B4), ownership census 0.72, remap 0.37, transfer 0.36, and the surface reconstruction and metric family, about 0.9.
   - Each family collapsed to one claimed launch has the same expected shape as S2: about -60% of its drop ms. Expected **-2 to -3 ms at drop** across them.
3. **Attribute the particle move's other 82%, then shorten its chain.**
   - Split G2P and retire, separation, hull and walk, and append, the same way as q1-q3, and read the counters with `kernel-counter-bench` to separate chain length from occupancy.
   - Expected: **-0.1 ms at drop** (from 0.18 ms) and **-0.5 to -1 ms at seep** if the walk or registers dominate.
4. **Do not pursue momentum windows alone.** The bound shows at most 0.13 ms at seep, and about 0 at drop, where momentum is 0.116 ms and occupancy-starved.
5. **A brick-granularity cold skip is worth at most about 0.1 ms at drop.** Fold it into recommendation 1 rather than building it alone.

### Scratch: kept, not landed

Everything is under `~/code/me/fluid-scratch/lane-t/`.
- `saved/`:
  - `uniform-brick-transport.ts`, the fused B1 kernel;
  - `brick-transport-bench.ts`, its bench with the cold census;
  - `probe-reservoir-bulk-brick-transport.patch`, the hook;
  - `kernel-counter-bench.ts`, one kernel's GPU counters under Instruments attached to its own pid;
  - `mc_infra.py` and `mc_census.py`, the momentum counter probes and census;
  - `pp_probe.py`, the particle-move probes q1-q3.
- `trees/`:
  - `m1`, `m3` and `m4`: the momentum variants;
  - `mp`, `mp2` and `mp3`: momentum probes P1-P8;
  - `pp`: the particle probes;
  - `s23c`, `m3c` and `m4c`: the counter trees, with `m4c` also holding the census;
  - `s1` and `s4`: surface S1 and S4.
- `edits/`: the scripts that rebuild each tree from its base.
- `out/`: every run's logs and JSON, with `gate.log` as the index.

## Slot space (coordinated 2026-11-08)

B1 runs on what exists today and has no store of its own:
- surface advect and momentum use the paged textures and owner lists;
- the particle sort uses NB-FLIP's per-frame brick table.

Lane E builds the persistent level-0 frame brick store. Lane T adopts it, with ubaStage, when B2 needs halos.

## Baseline: HEAD f1da3003, 8x8x4 drop=2, steps 29-32, raw timestamps

| | drop=2 | rest |
|---|---|---|
| kernel total | 15.604 ms | 3.530 ms |
| pressure | 5.630 ms | 2.425 ms |

- Non-pressure excess: 8.870 ms.
- B1 family: 2.177 ms.

B1 family by group (ms):

| group | ms |
|---|---|
| surface advect (15 dispatches) | 0.780 |
| particle advect (prepass, move, walk) | 0.278 |
| local speed certificate | 0.276 |
| phi resolve (4 calls a frame) | 0.242 |
| particle sort (slabSort, 22 merge passes, finish) | 0.225 |
| momentum and its tiers | 0.297 |

Cross-scene baseline (Lane E, HEAD 334745cb):

| scene | step median | drift |
|---|---|---|
| seep 300 | 31.5 ms | +3.2e-4 |
| letters 120 | 57.7 ms | -0.0092 |
| rest 16x8x4 | 4.11 ms | 0 |
| drop=2 16x8x4 | 16.50 ms | -6e-8 |

## Landed (2026-11-08)

- **S3, the particle sort: 1cbebf88.** A counting sort per brick, landed alone without S1. The cross-scene table is in the commit message.
- **S2, momentum: 8822b2d8.** One claimed launch with the COLD retain.
  - Cold compile: advectMomentum 1.1 s, slabSort 0.2 s.
  - HEAD+S2 against the S3 commit:

| scene | before | after |
|---|---|---|
| seep | 31.6 ms | 31.0 ms |
| letters | 58.0 ms | 54.0 ms |
| rest16 | 3.79 ms | 3.28 ms |
| drop16 | 16.47 ms | 15.69 ms |

  - Drift stays in band.
- **S1, surface advect: withdrawn.** It took 640 s to compile cold. umAdvected's solid body is superlinear under Metal's inlining (see Lane C's progress). S1 comes back on Lane C's structure, held to 5 s or less per pipeline.

## Momentum attribution at seep (HEAD+S2, steps 196-199)

These were measurement-only launches, run before the real one and overwritten by it, so the simulation is unchanged.

| part | ms |
|---|---|
| whole launch | 1.18 |
| 4h jobs | 0.26 |
| h tiles | 0.72 |
| of which: tile context (claim, sides, liveness) | 0.07 |
| of which: traces | 0.59 |
| trace, sample addressing removed (tile home, texture fetch only) | 0.46 |
| trace, wall bounds removed | 0.66 |

The trace is a latency chain of dependent global loads and filtered fetches, each sample depending on the last. It is not ALU.

The first window (m1: a per-tile 8³ window, each texel staged per tile-home, with per-tap validity and umPointBounds still per sample) made it worse, 1.18 → 1.51 ms.

Two refinements of the window, as single runs of advectMomentum against the 1.18 ms of HEAD+S2:

| variant | what it does | advectMomentum |
|---|---|---|
| m3, staged addressing | Per job, threadgroup copies of the 27 tile homes and the 27 page kinds around the tile. umSampleTile and umPointBounds become threadgroup lookups; the texture fetch stays. | 1.38 ms |
| m4, m3 plus a window | An 8³ window of the job tile's own halo block: 512 texel loads, one home, exact taps. A miss falls back to the sampler. | 1.51 ms |
| bound on m4 | The h jobs sample only from the window, and a miss reads 0. A measurement launch. | 0.59 ms (the h jobs alone, today 0.72) |

**Finding:** with every tap in threadgroup memory, the h jobs save at most 0.13 ms.

**Why m3 and m4 are slower is unknown.** The explanations first written here are hypotheses, not attributions:
- that the trace's sample count dominates;
- that the staging preamble lengthens each job's critical path.

The bound itself argues against a load-latency story. Serving every tap from threadgroup memory should have collapsed the h jobs if they were latency-bound, and it removed only 0.13 of 0.72 ms. See LEARNINGS for what was and was not measured.

The fused bench's momentum stage (530 µs for 2.1M faces) used one midpoint step without walls, seams or freeze. That is not today's trace, so it overstated the win.

**So momentum alone does not justify a brick-job form, and it is not wired.** Its only possible win is sharing the staged windows with phi and particles in one pass.

## B1 steps (S1 and S4 withdrawn, S2 and S3 landed)

Each step replaces one field's classify, list and twin machinery with one claimed launch and deletes the old path.

- **S1, surface advect.** One launch, `advect`. Each job (an h tile, 8 seam 4h tiles, or 64 regular 4h owners) evaluates every vertex's whole path in place.
  - Deleted: the deferred, cubic, trace and wall lists, their twins, and `uniform-mixed-surface-capacity.ts`.
  - The surface advect family went from 0.780 to 0.666 ms. The single launch costs 0.642 ms: the long general path diverges inside SIMD groups, behind the quick exits.
- **S2, momentum.** One launch, `advectMomentum`, covers h tiles and the merged 4h jobs.
  - Deleted: the retain/affine tiers (`uniform-mixed-momentum-tiers.ts`), the trace list and its stage scratch. Every non-frozen face takes the exact unit trace.
  - Momentum went from 0.297 to 0.112 ms.
- **S3, particle sort.** It is now a counting sort by cell inside each brick's workgroup (Bridson 2015, section 5.3), and the scatter writes the compact positions.
  - Deleted: the canonical in-cell merge sort, its 22 merge passes, `slabSortFinish`, `uniform-particle-canonical.wgsl.ts`, and its tests and probes. No bit parity is required.
  - Sort went from 0.225 to 0.044 ms.
- **After S3:**
  - B1 family: 1.680 ms.
  - Non-pressure excess: 8.316 ms.
  - Drift at step 32: -5.0e-6, against -4.7e-6 at baseline.
  - Step median: 15.88 ms, against 16.68.
- **S4, surface advect compaction.** The quick exits and the Euler transport ran in the enumerating lane. Unsettled vertices went into a 256-entry workgroup queue, which all lanes drained through the general path (Laine, Karras and Aila 2013).
  - Reverted: the coordinator ruled it launch tuning. It is acceptable only as the in-pass work distribution of the fused kernel.

## S1-S3 gates (trees/s3 = f1da3003 + S1-S3, not yet landed)

- `npm run check:types` passes.
- `npm run test:unit`: 1776 pass, 1 fail. The failure is `reservoir-particle-cost`'s merge cap test, which belongs to the peer's uncommitted NB_CELL_MASS hunks.
- Focused GPU tests (particle slab, momentum trace, surface canonical): 5 of 5 pass.

| scene | baseline (334745cb) | S1-S3 | drift (baseline → S1-S3) |
|---|---|---|---|
| rest 16x8x4 (60, pipelined) | 4.11 ms | 3.17 ms (steady 3.08) | 0 → 0 |
| drop=2 16x8x4 (60, pipelined) | 16.50 ms | 16.09 ms (steady 15.73) | -6e-8 → -4.9e-7 |
| letters 120 | 57.7 ms | 56.6 ms | -0.0092 → -0.0091 |
| seep 300 | 31.5 ms | 34.4 ms | +3.2e-4 → +2e-4 |

**Seep A/B, paired under the same load:**

| | base (f9ae8760) | S1-S3 |
|---|---|---|
| step median | 31.1 ms | 34.9 ms |
| p95 | 33.6 ms | 37.6 ms |
| particles | 104,896 | 99,429 |
| phi source the probe matched | canonical/field | raw/field |

The +3.8 ms is confirmed. The 5% fewer particles and the phi source flip point to a behaviour change. One lead is S1's 4h base/twin write-through no longer matching h. Kernel attribution is underway (seep steps 196-199, base against S1-S3). S1-S3 land once seep is clean.

## Fused B1 pass (the design, accepted as GO)

- **Kernel:** `lib/methods/uniform/uniform-brick-transport.ts`, kept unlanded at `fluid-scratch/lane-t/saved/`. One dispatch with one 256-lane group per brick, level 0 and level 2 in the same launch.
  - It stages the 27-neighbour record and the level-0 brick's parent 4h region (5³ values per channel) once.
  - It fills halo-2 windows (phi and the 3 face components) through the record, prolonging from the staged parent.
  - Then it runs, with barriers between them:
    - particle RK2, appended to the destination brick's inbox with one atomic per destination per round;
    - phi RK2 departure, trilinear, or a Catmull-Rom cubic within 2 widths of the interface;
    - momentum RK2 with MAC sampling;
    - the speed max.
- **Bench:** `tools/brick-transport-bench.ts`, also in `saved/`, run as `probe-reservoir-bulk --brick-transport`.
  - An untimed gather from NB-FLIP's brick table and the frame's textures fills a dense store.
  - Each `btStages` variant runs as 9 timestamped single-dispatch passes. A stage's µs is the difference from the variant before it.

drop=2 at step 32, 8x8x4: 144 level-0 bricks, 68 level-2 pages, 11.4k particles, 0 spills.

| µs | parent staged per fetch | parent staged once |
|---|---|---|
| one-group floor | 77 | 40.5 |
| record + windows (level 0) | 218 | 107 |
| particles | 3.6 | 0.5 |
| phi | 22 | 26 |
| momentum | 62 | 80 |
| speed max | 2.3 | 2.2 |
| level-0 total | 308 | 216 |
| both levels, min (median) | 326 (432) | 276 (348) |

Against the B1 family's 2.18 ms at baseline and 1.68 ms after S3. The median is noisy because other agents' runs were contending.

**Not yet in it:** the certificate chain, resolve, solids and walls, the separation prepass and walk, and the exact seam rules (ubaStage). Nothing is wired into the frame.

**Wiring order:**
1. Momentum. Its stage must skip COLD or still faces by the existing heat and freeze state.
2. Surface phi.
3. Particles.
4. Speed, certificate and resolve.

Each stage deletes its old launch in the same commit. The slot space is Lane E's `UniformFrameBricks` (26c969e4), widened to the particle bricks and the level-2 pages.
