# Lane T progress: B1 transport, B2 surface, B3 momentum to RHS

Design: `docs/plans/brick-resident-step-2026-11-08.md`. Lane E owns ubaStage, B4, the census and T.

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
- The cost is the trace's sample count. The substep trace takes 2 samples per segment of at most one cell, 3 components each, and a SIMD group runs as long as its longest lane.
- Each job's staging preamble adds to its critical path. It costs more than the addressing it removes.

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

- **Kernel:** `lib/methods/uniform/uniform-brick-transport.ts`, one dispatch with one 256-lane group per brick, level 0 and level 2 in the same launch.
  - It stages the 27-neighbour record and the level-0 brick's parent 4h region (5³ values per channel) once.
  - It fills halo-2 windows (phi and the 3 face components) through the record, prolonging from the staged parent.
  - Then it runs, with barriers between them:
    - particle RK2, appended to the destination brick's inbox with one atomic per destination per round;
    - phi RK2 departure, trilinear, or a Catmull-Rom cubic within 2 widths of the interface;
    - momentum RK2 with MAC sampling;
    - the speed max.
- **Bench:** `tools/brick-transport-bench.ts`, run as `probe-reservoir-bulk --brick-transport`.
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
