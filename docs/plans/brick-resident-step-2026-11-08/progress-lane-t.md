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

## Phase D (2026-11-08, in the working tree, not yet gated)

### The move as one launch

- **Dead fast-sample code deleted.** `nbFastSample` was a constant false,
  so the slow status (2), `nbListSlow`, state word 10's slow list, the
  routed walk list (`nbListWalk`, word 23) and the `advectWalk` entry never
  did any work. advectWalk was still a launch every step. The flat lane's
  slow-list index (`jobs.w`, always 0) is gone too.
- **separationPrepass folded into the move.** sampleSeparation "pairs" now
  takes each record's pair correction inline (`nbSeparation`) from the
  current bank, which the move does not write. That frees the prepass
  launch and its 3 words a record past the arena windows (about 40 MB at
  capacity 3.39M). The advect batch is the single `advectParticlesDeferred`.
- **What the combined gate should check.** Whether the inline pair loop
  raises the move's register pressure. If it does, the fix is to compute
  the pair corrections into threadgroup memory ahead of the trace, inside
  the same job, not a separate launch.
- **Files touched:** uniform-narrow-band-flip.ts,
  uniform-particle-slab.wgsl.ts (comments only), tools/nbflip-hot-variants.ts,
  tools/probe-nbflip-variant-replay.ts and tools/experiment-nbflip-hot-frame.ts
  (dropped references to the deleted entries), and the plan's B1 note.

### G closed with no code change

- **Remap.** It runs only in encodeRelayoutHead, which only runs when
  `relayout.due()`. On a census frame the generation is built on the GPU,
  and remapFaces, copyFaces, planeSums and remapCells stride markListed's
  dilated changed-tile list, so an unchanged generation is empty
  workgroups. A host gate would need a readback or a heuristic.
- **The remaining fixed cost of an unchanged census frame** is the adopt
  copy (proportional to live tiles) and the ownership revision bump. Both
  need proof that the generation is word-identical. That proof is E's job:
  a deterministic builder with an exact "identical" bit.
- **NB transfer** is P2G. It reads the moved particles every step and is
  already skipped when idle.

### Momentum stays out of B1

momentumGroup reads this frame's centerPhi, volume and phase (`umCullAir`),
so it runs after the geometry. That is recorded under the plan's pass table.

### The move's cold compile: 6.9 s, accepted

The one-launch move compiles cold in 6.9 s against 6.5 s at HEAD. The
coordinator accepted it unstaged. If the gate shows the inline pair loop
costs registers, the pair corrections go to threadgroup memory inside the
same job (above), which would also bring the compile back down.

### E-a: a repeated build is exact, and skips its adoption

- **same** (the builder's first launch, a workgroup of 256 a group) compares
  three sets of words against copies the last encode held, and refreshes
  them. The last group to finish decides, through one ticket word.
  - The band words of the resident blocks.
  - The window's header and block records. Header word 8 is skipped: it is
    the boundary decision's counter, which every commit advances and no
    address reads. A first GPU check found it was why no build ever repeated.
  - The resolved page words of each resident block's 27-page neighbourhood.
    Every build read is within two tiles of a resident tile, so within one
    page. This covers AIR, SOLID and BULK16 page kinds, which block records
    do not show.
- **A build repeats** when those inputs match, the host epoch matches and the
  last build was CLEAN: no fatal bit, and (claims) nothing refused and no
  slot cut, or (admission) the need within the capacity. A clean build left
  nothing to claim, so its buffers are already the adopted generation.
- **The host epoch** (params word 6) is bumped by every host write a build
  reads that `same` cannot see: the region table, the capacities and the
  admission. It is also bumped by a live ownership revision that this
  builder's own adoption did not make.
- **On a repeat** every later launch exits at once (`repeated`,
  `repeatedUniform`). The receipt's counters are restored from the build it
  repeats, with changed 0. The adopt (`uniformMixedAdoptWGSL`) reads the
  builder's SAME word and copies nothing.
- **Checked by** tests/census-region-replay.test.ts (the second test). On the
  still tank, after each repeated build, every word the adoption would copy
  must already be the live ownership's.

### E-b: the census's deep COLD cache, by region

- **regionList** (one workgroup) walks the resident regions in pool order. A
  region replays when surface authority holds, it is deep COLD, and its stamp
  and generation are the ones its last classification wrote. Its records are
  then current. Every other region classifies and takes this census's stamp.
  The two lists are compacted by a scan, not by atomics.
- **replay**: a workgroup a replayed region, a lane a tile. It publishes
  each record, gathers the counters, the wet and near bits and the audit per
  region, and flushes them once. It replaces the per-tile replay inside
  classify and classifyCoarse.
- **classifyCoarse**: a workgroup a classified region, a lane a 4h tile. It
  was a fixed grid over the whole 4h list, which mixed deep and live regions
  in a workgroup and ran each group over several list chunks in series.
- **classify (h)** skips a replayed region's tile.
- **decide** now maps a workgroup to a region, so a deep COLD region's lanes
  take the short path together.
- **The replay is kept, not dropped**, because importance read-modify-writes
  classify's importance words in place. The tile record format (10 words) is
  unchanged.
- **Checked by** tests/census-region-replay.test.ts (the first test), on the
  drop frame. It runs the census twice from saved state. Run A has every
  region classified. Run B replays the deep regions from A's records. Every
  census word and bounds word must match, except the modes and lists. The
  reference is the census without the cache, which the cached census is
  contracted to equal. tests/census-coarse-bounds.test.ts now launches
  regionList before classifyCoarse.
- **uniform-mixed-frame.ts is not touched by E**, so it needs no
  COORDINATION entries.

### E checks (single GPU checks, 2026-11-08, no timing)

- **Drop frame, step 32, 8x8x4 drop=2.** The frame's census had 132 regions
  classified and 380 replayed (512 resident). Run A, every region
  classified: 512/0. Run B: 132/380. Every census word matches, and so does
  every bounds word outside the modes and lists. Running the frame's own
  census again, with the frame's own records, also gives census words
  identical to A's. That held only in this working tree's run: on HEAD plus
  the patch alone, the same run differed in 32440 velocity-key words (see
  the key check below).
- **Repeated builds, still tank.** 5 of the 6 builds the frames ran were
  repeats. That tank stops building once its layout is no longer due. After
  each repeat, every word the adoption would copy was already the live
  ownership's. Before the header word 8 fix, 0 of 24 frames repeated.
- **census-coarse-bounds** (the spread classifyCoarse against the serial
  reference, now launched per region) passes.
- **Unit tests** are at baseline. The one failure is the peer's known
  reservoir-particle-cost test.

### E-b key check: HEAD's replay was stale in velocity

- **The latent staleness.** HEAD's replay already kept a tile's velocity
  keys in place ("the velocity keys are classify's alone"), and so did E-b.
  The deep epoch certifies phi, widths and the open-cell masks. It does not
  certify the field the census reads: velocityB, the extension's output.
  UniformColdFreeze freezes f.velocity, but the extension's air faces carry
  lagged state that no 27-region certificate bounds. HEAD's extension has
  held slots, regional marks and a frozen bit taken from its own marks. B4
  keeps the last frame's far field wherever no sweep reaches.
- **The evidence.** On HEAD plus the patch, the frame's own census and a
  full classification differed in 32440 key words, all in replayed tiles.
  The first went from 0 to about 5e-10 m/s. In this working tree's run (B4
  extension, 512 rather than 544 resident regions) they matched by
  coincidence of the field.
- **Every reader of the keys reaches replayed tiles.** importance covers
  every live tile (flow activity, approach, speed). boundCubes covers every
  tile, and decide's departure trace and pageSeed read the cubes. decide's
  movingShapeNear reads neighbours' speed. Only decide's deep short path
  reads no keys.
- **Every velocity word of a record is a function of the keys.** That
  covers speed, reach, the fast hysteresis, the shape score, gap, directed
  travel and umBoundary. So a record is exact when its keys are.
- **The fix: a key-checked replay.**
  - replay takes each live tile's keys afresh (umOwnerBounds, the velocity
    half of umClassifyOwner, now its own function).
  - A 4h tile's keys come from its lane's serial loop. An h tile's come
    from the workgroup, a lane an owner, one tile at a time, and the check
    stops at the first key that differs.
  - A region whose keys all match publishes its records, and its keys are
    the stored ones by construction. A region with any moved key is demoted
    to mode 2 and publishes nothing.
  - classify (h) then classifies its h tiles, since ccReplayed is mode 1
    only. classifyCoarse's jobs now run on past the classify list into the
    replay list, and a demoted region classifies there. Every other
    replayed region's job exits at once.
  - The region's stamp is unchanged, so its new records replay next census.
- **The saving shrinks.** E-b no longer saves classify's velocity loads (6
  face loads for a regular 4h tile, up to 96 beside h tiles, 64 owners for
  an h tile). It still saves the phi corners, volume, shape error, depth
  samples, boundary and thin work. The velocity loads can't be skipped,
  because the census contract is equality with a full classification and
  velocity is the one input the deep epoch doesn't certify.
- **coldRegions "fine".** There's no UniformColdFreeze, and a COLD region is
  re-solved every step, yet the reference attached the heat for replay on
  coldCensus alone. It now attaches only when coldRegions isn't "fine"
  (webgpu-uniform-reference.ts, patch-only). That also drops decide's
  deep-COLD keep under "fine", whose premise ("nothing around it moved")
  fails in the same way.
- **The test.** The first test now asserts C (the frame's own census and
  records again) against A, word for word, and logs how many replayed
  regions C demoted. It is not yet run: it goes into the combined round's
  correctness checks. The census WGSL passes naga in the CPU preflight
  (tools/capture-uniform-wgsl.mts, bulk-rest-tank). The three naga
  failures there are the known umUnitTrace and nbBoundaryCubicHit
  returns.

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

## 2026-11-09: the letters particle move (advectParticlesDeferred)

The scene is letters (nbflip-figure-8-letters, dt 1/24, balanced, about 960k particles). The move takes 16.5-16.7 ms per step.

### Counters

One capture covered frames 101-160 and 22 move windows of 15-19 ms. It was reduced by `fluid-scratch/lane-t/tools/advect-counters.mts`.

| Counter | Value |
|---|---|
| Compute occupancy | flat at 12.5%, a hard ceiling |
| ALU | 21% |
| F32 | 3.8% |
| LLC | 27% |
| Buffer read | 6.6% |
| Buffer write | 2.8% |
| Texture sample | 2.6% |
| Threadgroup memory | about 0.1% |
| MMU | 8% |
| TLB miss rate | about 18% |
| DRAM read / write | 21 / 6 GB/s |

- **No limiter.** The kernel is latency-bound at a register-capped occupancy.
- **A 2-3.5 ms tail.** Each move ends with occupancy falling away. This is load imbalance from the static round-robin job stride.
- **Per-particle cost.** It is 17 ns per particle in letters against 29 ns in seep, so the cost scales with particles and is not a fixed floor.

### Occupancy probe

The probe compiles only. It dumps the MSL from Dawn, strips the 64-thread cap and reads Metal's `maxTotalThreadsPerThreadgroup`.

- **Whole kernel: 384 threads.** That is 384/3072 = 12.5% occupancy, which matches the counter.
- **nbAdvect stubbed out: 1024.**
- **No single section is to blame.** These sections each limit the kernel on their own:

| Section alone | Threads |
|---|---|
| One velocity sample | 448 |
| Phi depth | 576 |
| gridSupported | 768 (384 together with the loop skeleton) |
| nbParticleBlocked | 704 |
| Sparse age | 832 |
| Separation | 768 |
| Merge prefix | 512 |
| Bare Heun skeleton | 576 |

- **umSolidFree and a rolled axis loop do not help.**
- **Consequence:** raising occupancy would take register work across the whole kernel. This batch instead removes dependent load chains and the tail.

### Changes

All four changes are exact: they do not change any result.

- **A. Ticketed jobs.** Each workgroup takes its next job from `state[NB_MOVE_TICKET_WORD=23]`, which the host clears before the move. The flat spill jobs come first. This removes the end-of-move imbalance tail.
- **B. Window bins for cell runs.** The window stages each tile's first bin in `nbWinBin`. nbRun then skips the page-key, stored, coverage and slot reads: 2 dependent hops per run, 8 runs a separation plus sparse age.
- **C. Window fast path for the move sample.** In a fine tile, nbMoveSample reads the tile class and halo home from the window (`nbWinFlags`/`nbWinHome`), skipping the page walk, tile word and halo lookups for each of the 3 Heun samples. The value is umSampleVelocity's own interpolant through the same umSampleVelocityAxis call. This is not the separate path that e4041e9a removed.
- **D (reverted). Separation reads the compact stream.** It read positions from SORT's stream at 16 bytes a sample. Because `links` is `array<u32>`, each candidate cost three loads against the record's single vec4 load. Buffer load utilisation rose from 5.4% to 9.5%, which cancelled the locality gain.

Probe after the changes: still 384 threads, so occupancy did not fall. The slab GPU test passes. The census, slab and NB-FLIP unit tests pass.

### Measurement

My earlier main-checkout capture is not a fair baseline. That checkout held a peer's uncommitted NB_CELL_MASS retirement: 798k particles at frame 128, against 1.037M on clean HEAD. My after capture there is not fair either, because Lane E's WIP did not compile. The coordinator's 16.5 ms letters figure is unaffected. It came from a clean `git archive` tree (combined/h3-tree) at steps 116-119 with about 960k particles. The 1.04M here is a later frame.

So both runs were taken from trees built by `git archive 83e48800` into scratch (`fluid-scratch/lane-t/tree-head` and `tree`). Each is one counter run with the same gated window. Frame 128 had 1,036,693 particles on HEAD and 1,036,201 with the patch. The 31 move intervals pair index by index.

**Move time.** These numbers are for A+B+C+D.

| Frames | HEAD (ms) | Patch (ms) |
|---|---|---|
| All 31, mean | 28.31 | 28.54 |
| First 10 | 34.0 | 36.6 |
| Last 15 | 24.6 | 23.0 |

The result is neutral overall: early frames are slower and late frames faster. Each figure is a single run.

**Counters, HEAD → patch.**

| Counter | HEAD | Patch |
|---|---|---|
| End-of-move occupancy tail | about 12 bins | mostly gone; holds 12.5% to the last few bins |
| TLB miss rate | 18-20% | 4.5% |
| Read bandwidth | 18 GB/s | 13 GB/s |
| Buffer load utilisation | 5.4% | 9.5% (the D cause) |
| ALU | 25% | 26-31% |

**What is untimed.** A+B+C without D has had no GPU timing yet. The combined gate measures it.

## 2026-11-09: slab pool overflow (submerged sphere, HEAD 83e48800)

**Symptom.** `uniform-body-contact-refinement`'s submerged prescribed sphere stopped at frame 66 with "particle slabs exceed the pool: 52224 records sized of 52224". The live count peaked at about 45-46k (Lane E, `fluid-scratch/lane-e/r7`).

**Cause.**
- slabAllocate and slabPlace sized each brick's slab as its count plus slack:
  - scratch slabs: a quarter, at least 32;
  - current slabs: an eighth, at least 16, plus last move's refused seeds.
- The run stopped (state[3] = 6) when the slab sums passed NB_SLAB_LIMIT = capacity − spill reserve.
- With many small bricks, the per-brick minimums made the slack alone about 6-7k records. The pool held every live record, but the run still went fatal.
- Seed deferral was correct: refused seeds go to the spill reserve, where optional seeds are clipped (state[2]) and reseeded later. The fatal check fired before deferral could act.

**Fix** (`uniform-particle-slab.wgsl.ts`, landed as 99dd6d64 together with the test's `UM_BULK16` stub):
- Slack is headroom, not records. Each pass first scans the demands and the slacks (`nbSlabPrefix` then `nbSlabSums`).
- Where demand plus slack passes the limit, every slab keeps its demand and the same share of its slack (`nbSlabSlack`). Its floor is `room / total slack`.
- Only demand past the limit stops the run, as does the overflow index passing capacity.
- Records past a slab's room overflow to the spill reserve, and seeds past it are refused there. Both paths already existed.

The limits stay separate: the spill reserve is the move's overflow tail, and folding it into the slabs would remove what absorbs a slab's overflow.

**Check** (clean tree: 83e48800 + lane-t.patch):
- The submerged sphere passes: step 120 displaced 0.00304 m³ with 0.712 Ns upward impulse, step 240 0.00281 m³ with 0.652 Ns.
- The slab test passes.
- The file's first test, "static and body contact retain wet or predicted contact detail", failed on 83e48800 as well. Its WGSL stub lacked `UM_BULK16` ("unresolved value 'UM_BULK16'", line 70). 99dd6d64 adds the stub, and all 3 tests pass on HEAD + the fix alone, with 1784/1784 unit tests.

## 2026-11-09: the move's register ceiling (compile-only probe)

**Tools.**
- `tools/spill.sh` gives the kernel's threads plus its compiler spill bytes, read from the trace's `graphics-compiler-spill-events`.
- A run sometimes logs no spill event and then reads low (0 or 32 B), so spill figures are repeated before they count.
- `tools/callgraph.py` counts each function's inlined copies from the entry point.

**Where the cap comes from.** These are the threads each section allows with nbAdvect cut down to it, inside the nbMove driver only:

| Section alone | Threads |
|---|---|
| Nothing (record in, record out) | 1024 |
| Bare Heun skeleton (every section stubbed) | 576 |
| One velocity sample | 448; 576 without the mixed-width tap path (`umVelocityTap1`'s nested `umSampleVelocity4`) |
| One depth | 576; with the skeleton, 384 (448 with `bandPhi` stubbed) |
| One rigid contact | 512 |
| nbFinish | 384 |
| nbFinish without the solid walk and the split | 1024 |
| nbFinish, walk only | 384 |
| nbFinish, procedural/terrain/rigid contacts only | 384 |
| nbFinish, `umCellOpen` only | 448 |

- With the skeleton and one section real, the sample, depth, rigid contact and finish each reach 384, the floor, on their own.
- Inlining: nbMoveDepth had 10 copies (merge 1, G2P 1, surface shift 7, finish split 1), each expanding to about 250k characters of WGSL, mostly `bandPhi`'s vertex sample. That is 80% of the kernel's inlined text.

**Changes** (both exact):
- particleDepthIn takes each key as it weighs its vertex, in one rolled loop, with no `keys` array.
- nbSurfaceShift reads its centre and six offsets at one nbMoveDepth site.
- Result for the move: 384 threads, spill 464 -> 352 B (repeated). Depth change alone: 416 B.
- seedCells drops 640 -> 576 threads. Keeping the single-block branch unrolled made both kernels worse (move spill 384 B).
- unit 1784/1784.

**Why 512 needs a different shape.**
- Four sections each reach the cap alone: the solid walk, the sample, the depth and the rigid contact. The skeleton by itself sits at 576.
- So no set of local edits brings the whole kernel to 512. Getting there takes:
  1. moving the rare paths into a second launch over a compacted list: the solid walk (taken only where the hull is not proven uncut), body contact, the mixed-width tap path and the unreached-depth `bandPhi`;
  2. splitting the move's common path so the sample and the depth do not share one kernel's live state.

## 2026-11-09: the rare-path launch (option 1), measured and not landed

**What was built.** This lives in `rare-split.diff` in the lane's scratch directory, not in lane-t.patch.
- The step's move (`advectParticlesDeferred`) runs a hot body, nbAdvectHot. It defers a record to a compacted list (`nbDefer`; state word 27 counts, the list follows the slab words, list overflow is fatal 12) at the point it would take any of these paths:
  - the unreached-depth `bandPhi`;
  - the general mixed-width tap (`umVelocityTap1`);
  - a body's candidate tile (checked before the trace, at the contact's own point);
  - the solid walk (hull not proven uncut);
  - the merged sample's split.
- A second launch in the same pass (`advectParticlesRare`) then moves each deferred record through the full body from its record, appending as a flat job does.
- `--rare` on probe-tier-budget prints the last move's deferred records against the live ones.

**Probe** (letters capture, compile only):

| Kernel | Threads | Spill | Inlined WGSL | Cold compile |
|---|---|---|---|---|
| lane-t.patch move (baseline) | 384 | 352 B | 1.63M chars | 3.2 s |
| hot move | 384 | 192 B (176 to 256 across variants) | 0.32M chars | 1.6 s |
| rare launch | 384 | not measured | 1.64M chars | 2.8 s |

- The cold compile was measured with a perturbed literal so Metal's cache cannot answer.
- The literal `hot` folds: the stubbed-cold variant reads the same threads.

**Why the hot body stays at 384** (variants of the hot body, every other section stubbed):
- The bare skeleton is 576, but only once the body-mask check is hoisted before the trace. An early return inside the trace loop alone costs it 64 threads (512).
- Skeleton plus one real velocity sample: 384, whichever path is left. Halo only, coarse 4h only (rolled or unrolled), and the fine paths deferred all read 384.
- Skeleton plus one depth: 384, even with only the single-block key path.
- A kernel that does nothing but one `nbMoveSampleHot` (or one depth, or one surface shift) in the nbMove frame reads 512.
- So no kernel that samples the field through the shared sampler reaches 576 in this frame. A stage-per-launch split (one Heun stage a kernel) would cap at 512 too.

**Rare fraction** (60-step check runs, one each):
- Letters: 3.3%, 7.7% and 4.7% at steps 20, 40 and 60.
- Seep: 51%, 73% and 56% of live records at steps 20, 40 and 60. Nearly all of them are the solid walk (2840/2855, 6932/7940, 8071/8362); the rest are unreached depth (969 and 148 at steps 40 and 60) and splits (15 to 143).
- At seep the walk is the common path, not a rare one. Most records would run the hot body and then the full body again.

**Conclusion.**
- Not landed. It doesn't lift occupancy (384 -> 384), and at seep more than half of the records would run twice.
- The cap is the shared sampler's and the depth's register cost on top of the move's live state, not the rare paths.
- The levers left are:
  - a move-specific sampler and depth with far less live state per call;
  - a smaller per-record live state across the trace: the record's 12 words, the hull, k1/pic/end/x.

## 2026-11-09: live state and move-specific fields (levers 1 and 2), the stop rule

**Method.** Compile-only probes of the lane-t.patch move: `advectParticlesDeferred` in the letters capture, baseline 384 threads with 352 B spilled.
- `tools/tgp.py` rewrites nbAdvect and nbFinish to hold the record in a per-lane workgroup slot instead of registers. It uses `let p=&nbLaneP[lane]`, with every use reading through the pointer. Dawn accepts the pointer member access.
- `tools/bound.sh` swaps in a best case for lever 2:
  - a sampler that reads only the window's staged halo (`nbWinHome`; 3 axes x 8 taps, no page walk, no 4h blend);
  - a depth that reads only the single-block key run (`umVertexRun`, 8 taps).
- Every figure below is a single probe.

**Lever 1 alone** (record in workgroup memory, everything else real): 384, spill 352 -> 304 B.

**Both levers at their best, from the skeleton up** (every section not named is stubbed):

| Kernel | Threads |
|---|---|
| Skeleton (all sections stubbed), record in workgroup memory | 576 |
| ... plus the slim sample and slim depth | 448 |
| ... and no hull (`nbHullInclude` / `nbHullUncut` stubbed) | 512 |
| that 512 kernel plus the real merge prefix | 512 |
| plus the real surface shift (7 slim depths) | 512 |
| plus the real endpoint clamp, gridSupported and umPointKind | 512 |
| plus the real nbParticleBlocked | 448 |
| plus the real nbSeparation | 448 |
| plus the real window lookups (age, heat, cold, thin) | 448 |
| plus the real rigid contact | 384 |
| plus the real nbFinish (solid walk, contacts, split; record in workgroup memory) | 384 |
| Whole move, slim sample and depth, finish stubbed | 384 |
| Whole move, slim sample and depth, record in workgroup memory | 384 |

**Register map.**
- Kernels that don't fit:
  - Even the best case only reaches 512 with no hull and with the blocked test, separation, window lookups, rigid contact and finish all stubbed.
  - The finish (the solid walk with its contacts) and the rigid contact each reach 384 on their own, whatever the trace holds.
  - nbParticleBlocked, nbSeparation and the window lookups each cost a bucket (512 -> 448).
- Inside the trace, the live state costs about one bucket each:
  - the record's 12 words: about 64 threads (448 -> 512 needs the hull gone as well);
  - the hull: 6 floats.
- Neither lever gets the whole move to 512, so per the stop rule nothing lands.
- What would still change the cap:
  - moving the finish's walk out of the move kernel. At seep the walk is the common path (51-73% of records), so this means a launch over the records, not a rare list;
  - an occupancy-independent design for the walk's contacts.

## 2026-11-09: the depth rewrite is the move's own

The combined gate (HEAD+E+T against 99dd6d64, letters steps 116-119) moved three kernels:
- the move was 0.66 ms faster;
- NB-FLIP seed was 0.39 ms slower;
- NB-FLIP transfer was 0.26 ms slower.

The rolled key loop went into the shared particleDepthIn, which seeding and transfer also call.

**Probe** (compile-only threads on e68655c3: E only, E + T without the depth change, E + T, E + T with the move's own depth):

| Entry | E | E+T no depth | E+T | E+T, move's own depth |
|---|---|---|---|---|
| seedCells | 640 | 640 | 576 | 640 |
| seedRank | 640 | 640 | 576 | 640 |
| seedInitial, transfer | 384 | 384 | 384 | 384 |
| transferDeposit, transferSeams, seed, seedSources | 512 | 512 | 512 | 512 |
| advectParticlesDeferred | 384 | 384 | 384 | 384 |

- Only particleDepthIn reaches seed and transfer. nbSurfaceShift and A+B+C are move-only: the "no depth" column equals E.
- The fix: particleDepthIn is HEAD's again, and the rolled loop is `particleDepthMoveIn`, which only nbMoveDepth calls. The value is the same: same keys, same weights, same order of the sum.
- Spill, same columns without the "no depth" one (E / E+T / move's own depth): seedInitial 64/80/64 B; the rest are unchanged at 48-64 B.
- The move: 480/352/352 B. The move's own depth keeps its whole cut.
- Transfer's threads and spill are the same in all four trees. Its gate rise is not a register cost; the move's own depth puts its code back as it was on HEAD in any case.
