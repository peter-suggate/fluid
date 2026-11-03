# Dynamic coarsening census: why it costs ~8.9 ms on paddleboats, and a plan to get it under 1 ms

Scene: paddleboats, 18 rigid boxes (2 hulls and 16 motor paddles; every one moves every step).
The h tiles at step 300 number 4,389–4,852. Of these, only **52** are surface `crossing` tiles; 2,097 are `solid` (body proximity), ~2,320 `join`, 160 `travel` and 130 `closure` (scratchpad/occ.txt).
Resident pool: 192 blocks, so 98,304 tile keys and 1,536 16h regions. Region heat at step 300 was 562 hot / 850 warm / 124 cold, so cold replay barely applies.

Evidence sources:
- **B**: per-dispatch kernel timer, `probe-reservoir-bulk --scene=paddleboats --bodies --steps=150 --kernels=90 --dispatches` (scratchpad/exp/base.json).
- **X**: xctrace counters, artifacts/xctrace-paddleboats (xctrace-kernels.txt, scratchpad/census-xc.txt).
- **E**: one result-preserving experiment, patches/census2.patch, giving scratchpad/exp/patched.json. It was applied and reverted inside the GPU lock; the tree is clean.

Experiment E:

| change | kernel | base ms | patched ms | Δ |
|---|---|---|---|---|
| couple: 11 global atomics per (cell, body) → `var<workgroup>` integer sums, flushed once per workgroup (exact: integer adds commute) | couple | 2.262 | **1.377** | **−0.885** |
| buildRegions: `BODY_REGION_CHUNKS` 8 → 64 (512 → up to 1,536 live workgroups) | buildRegions | 1.379 | 1.193 | −0.186 |
| — | step mean (steps 90–150) | 73.08 | 71.49 | −1.6 |

The drift at step 150 was −0.00744 base and −0.00757 patched. I did not measure run-to-run variance, so bit-exactness of E is expected but **unverified**.

A first attempt using `enable subgroups` failed to compile, because the probe's device lacks the subgroups feature ("extension 'subgroups' is not allowed"). The kernels must not depend on subgroups unless the feature is guaranteed, so use workgroup memory.

## Per-kernel table

Column notes: occupancy, ALU and read bandwidth are xctrace means over the kernel's intervals; the M1 Max is ~400 GB/s. Every row is latency-bound: low occupancy, low ALU and almost no bandwidth. These kernels wait on dependent loads, serial lanes and contended atomics; they are not doing arithmetic or streaming memory.

| kernel (file:line) | ms (B) | occ / ALU / rd GB/s (X) | dispatch extent | useful items | per-thread work | confirmed cause | evidence |
|---|---|---|---|---|---|---|---|
| **couple** uniform-mixed-bodies.ts:126 | 2.26 | 15 / 16 / 0.7 | counted grid over every owner: ~300k h-cell lanes and 4h lanes that return at once | cut cells of 18 boxes: a few thousand (cells whose face fractions differ, or fraction > 0) | loops **all 18 bodies** (umBodyNear); per near pair: 6 × umBodyFaceFraction = 24 umInsideBody, then 8 more for umBodyFraction, then umBodyAmbient (6 probes, each umSolidVoxel + umBodyAt **all-body loop** + umOwnerAt + 14 texture loads); **7–12 global atomics per near pair even when area is 0**, all into 12 words per body (one cache line) | (1) contended same-line global atomics ≈ 0.89 ms, **measured** by E; (2) the remaining 1.38 ms is lanes × the all-body loop × the shape tests of ~130k near pairs over the 2,097 `solid` h tiles (~4M box evaluations), at 15% occupancy | E (−0.885 ms); X: 0.7 GB/s read means the cost is not bandwidth |
| **buildRegions** uniform-mixed-solid.wgsl.ts:770 (umBuildTile :659) | 1.38 | 8.4 / 18 / 0.3 | 8 chunks × 64 = 512 workgroups; each chunk walks 64-region rounds and builds tile `w&63` of **every dirty region in its round, serially** | dirty regions are clustered around 18 moving boxes, all in a few rounds; the cells actually swept are a few thousand | per cell: umCellOpen, 3 × umPressureFaceV → umFaceOpen = 4 samples × (umSolidVoxelAtWorld + terrain + umBodyAtCandidates) ≈ 30+ point queries, static voxel/terrain page-table lookups repeated every step; lane-0 serial loops inside umBuildTile; several barriers per tile | serialized tile builds per workgroup (low occupancy), each a long dependent chain that recomputes **static** geometry because a body moved. A wider grid alone buys only 0.19 ms (E), so the chain length and the static recomputation dominate | E; X occ 8.4%, rd 0.3 GB/s |
| **bodyMask** uniform-mixed-solid.wgsl.ts:514 | 0.40 | **0.6** / 2.9 / 0.1 | **one lane per region**: 1,536 lanes, 24 workgroups on a 98k-slot GPU | 1,536 regions × 64 tiles | 24-body loop with umBoxSegmentDistance, then for a near region **64 tiles × umBodyTileSwept (loops moved bodies, 2 × umBodyTileNear each), serially in one lane**: up to ~2.3k shape tests per lane | serial tail: a few lanes run ~2k dependent tests while the GPU sits idle | X occ 0.6% (lowest of the frame) |
| **classify** uniform-mixed-dynamic.ts:1062 | 1.18 | 11 / 15 / 2.8 | min(4096, tiles) workgroups striding the h list: one workgroup per h tile | 4.4–4.8k h tiles, of which **52 cross the surface** | per lane: umClassifyOwner (open words, corners, gaps/reach); 27-lane 4h-neighbour probe (umTilePresent/umTileAt); lane-0 resetGroupCensus / umFinishTile / flushGroupCensus while 63 lanes wait at barriers; 3 barriers per tile; resolution error and depth only on interface tiles | every h tile is fully reclassified each step although ~99% have no interface and their class did not change. The cost is ~4.8k serial lane-0 tails plus barrier chains (~40 µs of latency per tile at ~170 resident workgroups) | B; X occ 11%, rd 2.8 GB/s; occ.txt reasons |
| **classifyCoarse** :1120 | 0.66 | 12 / 27 / 7 | 256 workgroups striding the 4h list, 64 tiles per pass | 4h tiles; the crossing ones queue error/depth samples | crossing tile: 8 corner-laplacian samples + **64 umDepthSample**, each umLoadCorner + 8 umCellKind + up to 8 umTileKey → umCensusOpenCells (~20 dependent loads) | **64× redundant vertex work**: each 4h vertex's depth is recomputed by every tile whose 4×4×4 vertex window contains it; dependent page-table chains | code; X ALU 27% but occ 12% |
| **localSpeed** uniform-mixed-frame-plan.ts:259 | 0.83 | 14 / 29 / 7.6 | counted all-owner grid: h tiles one workgroup each, 4h tiles one lane each | all live tiles every step | 3 × umSignedFaceExtent (owner→face→texture chains), 8 corners, tuSurfaceRate/tuReaderClass/tuFineDefect, lane-0 64-iteration serial reductions per h tile, **6 global atomicMax per tile into regionVelocity[6·region]** (64 lanes in a 4h workgroup usually share one region) | dependent face chains for every tile, including far, dry, still 4h tiles; contended region atomics; serial lane-0 reduction | X occ 14%; code |
| **decide** uniform-mixed-dynamic.ts:1510 | 0.36 | 5.4 / 10 / 5 | 256 workgroups (16k lanes) striding **98,304 tile keys**: 6 visits each | the band candidates near the interface or bodies (a few thousand) | surfaceAllowed / movingShapeNear: 27–343-tile windows, umTileKey + umKeyStored + census per neighbour; RK2 sampledFlow boxes; departureMeetsSurface box scans; global atomicAdd(census[2,4,5]) per tile | grid is 1/6 of the GPU (launch shape), and each candidate walks neighbour windows of page-table lookups | X occ 5.4%; code |
| **seed** (frame plan) :163 | 0.29 | 13 / 30 / 15 | umResidentAllOwner | — | 125 vertex loads per h tile, many already loaded by localSpeed/classify | repeats loads that other kernels make | B, X |
| **solidPromote** :1615 | 0.15 | 17 / 34 / 15 | 16k lanes over 98k keys | non-band solid tiles | two 27-neighbour umTileKey windows (wall distance, active stencil) | 27 dependent page-table lookups per visited tile | B, X |
| importance, pageSeed, pageRequest, solidActive, interfaceBits/Blocks, boundCubes | 0.09 / 0.08 / 0.07 / 0.02 / 0.05 / 0.02 | low | key or page grids | — | — | small; mostly launch + grid-stride over keys | B |
| layout build (mirror 0.12, classify 0.11, + 9 more) | ~0.45 | classify 60% ALU | per key | — | — | the only ALU-heavy kernel; key-sized grids | B, X |
| frame plan dilate0–2, regional expand | ~0.18 | low | per key | — | — | three key-sized passes | B |
| markTiles uniform-mixed-bodies.ts:150 | < 0.03 | — | 98k keys × 18 bodies | — | ball + distance per body | cheap; it is the body-to-tile binning the other kernels should reuse | B |

**Not in this stage but the same pattern:** `NB-FLIP surface reconstruction / coupleFine umSolidTwinned` costs **1.36–1.47 ms** (occ 11%). It is a per-cell × all-bodies loop like couple, and the same fixes 1–3 below apply.

## Root cause in one line

Every census kernel visits **every** h tile, 4h tile or tile key each step and walks **all bodies** or a **neighbour window of page-table lookups** in its inner loop. The work that can change is a few hundred surface and body-cut tiles. The GPU spends its time waiting on serial dependent loads, serial lanes and contended atomics, at 1–17% occupancy.

## Plan to reach < 1 ms

There are no scene special cases. Every step is exact or a certificate, not a heuristic.

1. **Body-to-tile binning, once per step** (extend markTiles :150). Write a candidate-body mask per tile key (24 bits fit in one u32), with bodies swept over the step. couple, coupleFine, umBuildTile, bodyMask, umBodyAt and umBodyAmbient then loop only the tile's candidate bodies via `countTrailingZeros`. Typical tiles hold 1–2 candidates instead of 18, about 10× fewer shape tests.

2. **couple over a body-cut list, not over every owner.**
   - bodyMask/buildRegions already know which tiles a body cuts (`solidRecord[t].x>0`, body tile bits). Compact them into a list and launch couple and coupleFine over that list: a few hundred tiles instead of ~4.8k h tiles.
   - Reduce in workgroup memory, as experiment E did (exact).
   - Skip zero contributions: an area of 0 adds nothing.

   Expected: couple 2.26 → **≈0.08 ms**, and coupleFine 1.4 → ≈0.1 ms (outside the census).

3. **Split the solid record into a static part and a body part.**
   - Static voxel/terrain face openness changes only on edits. Cache it per tile at build time; it lives in the existing record slots or a sibling array sized like the record, so there is **no domain-sized new allocation**.
   - A body-only rebuild evaluates only the candidate bodies' shapes and combines them with the cached static fraction.
   - Launch one workgroup per **dirty tile** from a compacted list, instead of the chunked region rounds. This removes the serial chains.

   Expected: buildRegions 1.38 → **≈0.15 ms**.

4. **bodyMask: one lane per (region, tile).** Use 98k lanes, one wave; the region filter comes from candidate masks. Expected 0.40 → **≈0.02 ms**.

5. **classify: decide cheaply which h tiles need a full classification.** From the tile's 125 vertices, which localSpeed already loads into `planPhi`, publish a tile phi range and a "class inputs unchanged" certificate. A non-crossing tile whose range, open words and speed bounds are unchanged replays its last class, the same mechanism as ccValid cold replay but driven by the tile's own inputs instead of region heat. Only tiles failing the certificate, mainly the 52 crossing tiles plus tiles near moving bodies, run umClassifyOwner. Expected 1.18 → **≈0.15 ms**.

6. **classifyCoarse: compute each vertex once.** A vertex-centric pass computes depth (and the 4h corner laplacian) once per 4h vertex in a crossing tile's window. Tiles then take the max over their window as a separable 3-pass 4-wide max: 12 reads instead of 64 × ~20 dependent loads. Expected 0.66 → **≈0.10 ms**.

7. **localSpeed and seed: share the vertex loads; reduce the atomics.**
   - Reduce regionVelocity in workgroup memory: one atomicMax per workgroup per region word.
   - Feed classify (step 5) and seed from the `planPhi` that localSpeed already holds, so seed's 125-vertex reload goes away.
   - Restrict the face-extent work to tiles holding liquid or a physical reader. A dry, far tile's speeds are certified zero by its own empty face set, which is a certificate, not a skip.

   Expected: localSpeed 0.83 → ≈0.2 ms; seed 0.29 → ≈0.05 ms.

8. **decide, solidPromote and importance over a candidate list, with neighbour tables loaded once.**
   - Launch over the band-candidate list (interface-adjacent tiles from interfaceBits, plus body-candidate tiles) instead of 98k keys at 16k lanes.
   - Per block, load the 3×3×3 neighbour-block page words into workgroup memory once, and resolve every umTileKey in the windows from that table: one table load instead of 27–343 indirect lookups per tile.
   - Replace the global atomicAdd(census[2,4,5]) per tile with workgroup sums.

   Expected: decide 0.36 → ≈0.05 ms, solidPromote 0.15 → ≈0.02 ms, importance and pages ≈0.1 ms total.

9. **Key-sized small passes:** layout build (~0.45 ms) and frame-plan dilates (~0.18 ms). Run them over the changed-tile list (uniformMixedChangedTilesWGSL already exists) instead of all keys. Expected ≈0.15 ms total.

### Expected total

couple 0.08 + buildRegions 0.15 + bodyMask 0.02 + classify 0.15 + classifyCoarse 0.10 + localSpeed 0.2 + seed 0.05 + decide/promote/importance/pages 0.17 + layout/dilate 0.15 ≈ **1.07 ms**.

Reaching **< 1 ms** also needs localSpeed's liquid/reader restriction to land at ≈0.1 ms. That is the largest uncertainty, because localSpeed was not measured by experiment.

Cost scales with the cut, surface and changed tiles rather than with resident keys, which matches the memory-must-scale rule.

## Order of work, and how each step is verified

| # | change | expected Δ step | verify |
|---|---|---|---|
| 1 | couple workgroup reduction (E, ready) | −0.9 ms | rigidExchange words bit-equal to base over 150 steps (readback hash); seep/letters drift + step time |
| 2 | candidate-body mask + couple/coupleFine over the cut list | −1.3 ms (+ −1.3 coupleFine) | forces/torques/displaced equal to the all-body loop (unit test: random boxes, CPU reference); paddleboats drift; seep/letters unchanged |
| 3 | static/body solid split + per-dirty-tile build | −1.2 ms | solid record words equal to a full rebuild after random body moves (unit test); save/load/undo and edit regressions per AGENTS.md |
| 4 | bodyMask per (region, tile) | −0.38 ms | bodyRegions/tile bits equal (unit test) |
| 5 | classify certificate replay | −1.0 ms | census word and reason equality vs. full classify on recorded frames (seep, letters, paddleboats); drift + step time |
| 6 | vertex-centric depth/laplacian | −0.55 ms | tileDeep/tileError equal per tile (unit test on a wave scene) |
| 7 | localSpeed/seed sharing + restriction | −0.85 ms | speeds/updateError/regionVelocity equal on wet tiles; dry-tile certificate test |
| 8 | candidate-list decide + neighbour tables | −0.4 ms | census bits/reasons equal to the key-sweep version over 300 steps |
| 9 | changed-list layout/dilate | −0.45 ms | layout words equal |

Every step ends with `npm run check:types`, `npm run test:unit` and a single short GPU gate (seep 300 / letters 120 / paddleboats 150): report drift and step time each time.
