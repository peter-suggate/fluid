# Level set + volume on paddleboats: why each kernel costs what it does, and a plan to get the stage under 1 ms

Scene: paddleboats, `--bodies` (moored, paddles driven), M1 Max.

Sources:
- `scratchpad/boats.json`: the per-kernel timer, 300 steps, kernels timed from step 240.
- `xctrace-kernels.txt`: the parent's xctrace counters over 44 advances.
- `scratchpad/ls-counts.txt`: a census of the recorded step at frame 240 (workgroups per kernel, deferred-list headers, solver info).
- `scratchpad/occ.txt`: census reasons at step 300.

Scene facts at frame 240:
- 4845 h tiles, 93,459 4h tiles, 3152 + 4495 seam tiles.
- 327k particles, 1844 arena bricks.
- The h-tile census reasons are: join 2320, solid 2097, travel 160, closure 130, crossing 52. Only about 7% of h tiles are h because of the surface. The rest are h because of bodies or joins, so they are mostly submerged liquid or air well away from the surface.

## 0. Correction to the stage table: global surface volume is not part of the live frame

The 2.70 ms "Global surface volume" stage is the volume **diagnostic**, not a per-step cost.

How it runs:
- The full sweep (seed, 4× dilate, grow, measureGrow, metric, resolveScale, measure) is encoded only when a reader calls `requestNarrowBandVolumeProbe()`. See `lib/methods/uniform/uniform-mixed-frame.ts:1189` and `:466`.
- `probe-reservoir-bulk.ts:147` requests it every 60 steps (`--every`, default 60) and on the last step.
- The app requests it only from `readStats()` (`webgpu-uniform-reference.ts:3324`). The renderer calls `readStats()` only on pause (`lib/core/webgpu-renderer.ts:1873`).
- The live step instead runs the gauge path, `uniform-mixed-surface-volume.ts:555`: clearBand, measureInitial, reduce and gauge, about 0.07 ms in total. measureInitial is 4096 wg and 0.047 ms.

Evidence:
- The frame-240 census records only clearBand, measureInitial, reduce and solve in the volume pass.
- The xctrace table (44 live advances) has no seed, dilate, metric or measure rows.
- The 1.245 ms metric figure is a per-occurrence mean. `probe-reservoir-bulk.ts:432-434` averages each kernel over the steps where it ran, so a stage table that sums those means overstates any kernel that does not run every step. **The pressure, census and momentum tables built the same way should be checked for the same overstatement.**

Live stage total: about **13.3 ms**, not 15.9 ms.

The diagnostic should still be made band-proportional (P7), because the drift gates pay for it and it distorts the step time of any step that samples it.

## 1. Per-kernel table

Counts below are per step:
- "wg" is workgroups dispatched.
- "occ / ALU / rd" are the xctrace compute occupancy %, ALU % and read GB/s.
- The M1 Max has about 400 GB/s of bandwidth and about 98k resident threads.

### Surface advect, 3.06 ms, and surface advect deferred, 1.85 ms

Deferred-list header at frame 240, from `deferred.header`:

| near | cubicClear | cubicDeferred | trace | cubicSolid | deep |
|---|---|---|---|---|---|
| 26,317 | 36,601 | 0 | **248,778** | 4,375 | 41,487 |

| kernel | ms | dispatch extent | useful items | per-thread loads / loop trips | confirmed cause | evidence |
|---|---|---|---|---|---|---|
| advectFine `uniform-mixed-surface.ts:1798` | 1.18 | 2048 wg × 64 (dispatchBuffered "fine") over 4845 h tiles, about 310k owned h vertices, 1–3 rounds per tile | the band vertices: those within about 5h of the surface (crossing / travel / closure tiles, about 7% of h tiles) | Per vertex, in `umAdvectStore` (:1459):<br>• umHoldCold: tier load through page → block<br>• umKeptAir: 2 loads<br>• burial: umSolidClear plus the umBuried loop<br>• `umVertexFoot`: 1 velocity sample via `umFastVelocity` (:361) = umSampleTile page lookup + 3 axes × (8 taps + a 4h sample), about 30 dependent loads<br>• then an atomicAdd and 8 stores to the trace list | **Every non-cold h vertex is traced, however far it is from the surface.** The only far exits are air-only:<br>• `umFarAirKept` (:1351) needs phi ≥ +16h<br>• `umDeepAirKept` (:1394) needs phi ≥ +(6+3w)h<br>Submerged liquid (phi < 0) has no exit at all. The certificates `umEarlyFarAdvected`, `umCheapAdvected` and `umEulerAdvected` compile to `return 0` (:1300-1302) because the 16h surface evidence is off, so every vertex falls through to `umListTrace` (:1447). | trace header = 248,778, about 80% of h vertices; occ 12.0, ALU 25.9, rd 3.4 GB/s, so latency-bound on the dependent velocity-tap chain |
| advectTrace (:1746) | 0.21 | 1024 wg grid-stride over 248,778 entries | as above | 2 more RK stages × about 30 dependent loads | same: transport of out-of-band vertices | occ 15.7, LLC 43% |
| advectFinish (:1762) | 0.70 | 1024 wg over 248,778 entries | as above | umWalk (solid slide, one per half cell), `umVertexCell` (8 loads), `umStoreSurface` (compares with the old value and bumps the region revision atomic) | same | occ 11.6, ALU 24.7 |
| advectDeferred general, solid-twinned (:1526) | 1.70 | 1024 wg (deferredGrid) over near + deep = 67,804 entries, about 25 ns per entry | the near 4h band vertices (some of the 26,317) | Full general path: RK3 trace (up to 3·LIMIT+3 stages, `umSurfaceTrace` :407), `umWalk`, and `umCubicPhi` (:651) with 64 taps, a 7-probe umCellOpen loop and a per-tap `umBuried` loop | Coarse owners run with `umDeferTraced`: every non-cold 4h vertex is queued (:1474), and **41,487 "deep" vertices (|phi| ≥ 2w·h) are traced in full anyway**. The comment at :1419 records that skipping them was only tried as a SIMD sort. | deep header = 41,487; occ 10.1, rd 13.8 GB/s |
| advectCubic ×2, advect merged, advectWalls, advectOwners, wallReach | 0.24 + 0.21 + 0.20 + 0.20 + 0.11 + 0.04 | 1024–4096 wg each | cubic: 36,601 + 4,375 entries | 64-tap cubic per entry; advectWalls is 4096 wg over wall faces | cubic work is band-limited already; the walls kernel is a fixed full grid | advectCubic plain occ 2.1 |

### Global surface volume: 0.07 ms live, 2.70 ms when sampled

| kernel | ms | dispatch extent | useful items | per-thread loads / loop trips | confirmed cause | evidence |
|---|---|---|---|---|---|---|
| seed, dilate ×4, metric, measure ×2 (`uniform-mixed-surface-volume.ts:238/324/360/434`) | 0.175 + 0.68 + 1.245 + 0.33, sampled steps only | `dispatchAllCounted` with `umResidentAllOwner`: umCounts.x + umResidentPageCount() ≈ 4852 + 1536 = 6388 wg × 64 ≈ 409k lanes, **8 dense sweeps** | the cut / band tiles, a few hundred | dilate: 27-neighbour `svNearBand` (umTilePresent → umTileAt → scratch) per lane.<br>metric: per owner, a loop over umCounts.w corners, each with an authority test, up to 8 band reads and 3 axes × 2 `umLoadVertexW` | the band is rebuilt from nothing by global sweeps, though NB-FLIP already has it (`NB_BAND_REACH`, crossing masks) | only runs on probe steps (§0) |
| measureInitial (gauge, live) | 0.047 | 4096 wg, all owners | the cut owners | 8 loads + fills | global sweep | ls-counts |

### Phi resolve, 0.94 ms

| kernel | ms | dispatch extent | useful items | per-thread loads / loop trips | confirmed cause | evidence |
|---|---|---|---|---|---|---|
| resolve ×4 + resolveListed (`uniform-mixed-phi-resolve.ts`) | 0.85 + 0.08 | 5 × 4096 wg × 125 lanes = 20,480 wg; each launch strides over all 7,647 seam tiles | the seam tiles whose 4h lattice changed since the last resolve | per job: stage 27 4h values through the page chain, then 125 lanes interpolate and store | The frame resolves the whole seam set four times per step (`uniform-mixed-frame.ts:1153, 1163, 1172, 1175`): after advect into scratch, after couple, after clearAir (gated) and after redistance. Each is a full launch whether or not its producer touched the tile. A changed-list variant (`resolveListed`, :153) already exists. | occ 24, ALU 37, rd 5.0 GB/s |

### Geometric fill, 1.47 ms

| kernel | ms | dispatch extent | useful items | per-thread loads / loop trips | confirmed cause | evidence |
|---|---|---|---|---|---|---|
| geometryRegional (`uniform-mixed-surface-geometry.ts:355`) | 0.98 | 2 × REGIONAL_GRID = 4352 wg; jobs = every frame-plan h tile (4845) + 64-tile 4h jobs | the tiles whose phi or solid capacity changed this step | `gcOwner`: 8 corner loads, `gcEvaluate`, 2–4 textureStores per cell | Re-fills every planned tile. Most tiles' phi did not change, or changed only through the out-of-band transport above. | occ 11.8, rd 18.7 GB/s |
| geometryChanged + mirrors | 0.32 + 0.13 | 8192 wg + 3 × 4096 wg | the changed list | | full fixed grids | |

### NB-FLIP surface reconstruction, 2.09 ms

| kernel | ms | dispatch extent | useful items | per-thread loads / loop trips | confirmed cause | evidence |
|---|---|---|---|---|---|---|
| coupleFine (`uniform-narrow-band-surface.wgsl.ts:187`) | 1.36 | a workgroup per h tile (umAllOwner, certified loop), 4845 tiles × 125 vertices; ×2 pipelines (solid-free and solid-twinned) | the h tiles holding band particles: particles live within bandWidth 4 of the surface | per vertex: `particleSurfaceBulk` (a loop over neighbour cell bins, then the particles: page → block → bin → particle) and `particleSurfaceFinish` / `nbCachedSurfaceNearest` | Visits every non-COLD h tile. A deep-liquid or body tile with no particles still runs the bin loops to find nothing. | occ 10.9, ALU 20.6, rd 4.2 GB/s: latency-bound |
| surfaceSplatOccupied, couple, surfaceClear, coarseCorners | 0.27 + 0.20 + 0.15 + 0.08 | 1024 wg each | occupied / band cells | | surfaceClear clears per non-COLD tile | |

### NB-FLIP arena search, canonical metric and publication, coarse retirement: 0.85 + 2.53 + 0.36 ms

| kernel | ms | dispatch extent | useful items | per-thread loads / loop trips | confirmed cause | evidence |
|---|---|---|---|---|---|---|
| arenaDistance (`uniform-surface-brick-arena.wgsl.ts:306`) | 1.20 | 1024 wg × 256, grid-stride over 1844 bricks | the vertices in reach&2 tiles. Bank A is h-resolution everywhere (35937 keys per block, :40), and particle membership samples it in 4h tiles too (`membership.wgsl.ts:328`), so 4h bricks are *useful*. This corrects my earlier waste hypothesis. | Per brick:<br>• stage a 729-value window, 1000 nearest-cell loads from `NB_DEPTH_B`, 64 reach loads<br>• then 2–3 vertex rounds of 256, each with about 5 barriers<br>• up to 8 candidates per vertex queued, `nbArenaCellDistance` per candidate | **Occupancy is capped by workgroup memory.** The kernel holds window 2.9 KB + nearest 4.0 KB + queue 8.2 KB + found 1 KB + reach and records 0.7 KB ≈ 16.9 KB per 256-lane group, so a core can hold only about 2 groups. This matches the measured 17% occupancy. Per-brick work is a serial chain: staging, then barrier-separated rounds. | occ 17.1, ALU 32.6, rd 5.8 |
| arenaPublishCanonical (:729) | 1.05 | 1024 wg over the bricks | the canonical vertices of non-COLD h tiles and 4h corners | re-stages 64 tile records (`nbStoreTile`), 1000 capacities (`nbStoreCapacity`) and incident words, then publishes each vertex with a bank-A load (:767) | Re-resolves the same brick neighbourhood that arenaDistance just staged: the staging is paid again on a separate launch | occ 11.5, ALU 13.4, rd 2.6: the lowest throughput of the stage |
| arenaPublishFinish (:797), arenaNearest, arenaGather ×2, arenaCross, arenaFar, arenaMark | 0.28 + 0.39 + 0.41 + 0.16 + 0.15 + 0.06 | 1024 wg each | the same bricks | each stages the brick window or the tile records again | **The same brick is staged 5–6 times across launches** | arenaPublishFinish occ 11.3, ALU 7.3 |
| depthCoarse, redistanceCoarse | 0.27 + 0.07 | 1024 wg over `NB_COARSE_LIST` | the listed corners | | 1024 fixed groups over a short list; 4.1% occupancy means mostly idle or serial | occ 4.1 |

Across every row the confirmed pattern is the same. The kernels sit at 10–17% occupancy, 13–33% ALU and 1–14 GB/s of read bandwidth, so they are latency-bound on dependent load chains. Total time is therefore roughly (items visited) × (dependent-load depth). The plan attacks the item count first, and the depth second by resolving addresses once per tile or brick.

## 2. Plan to bring the stage under 1 ms

Every item below is general: no scene checks and no special cases.

### P1. Symmetric narrow-band transport: skip every out-of-band tile

This is Peng et al.: outside the band, phi carries only its sign plus a guarantee that it is "at least 5h away". The arena already enforces that guarantee: `arenaFar` writes sign·max(|v|, 5h), and arenaDistance writes exact distance inside reach.

**Certificate (per tile, in advectFine, before any per-vertex work).** A tile keeps every vertex value unchanged when all of these hold:
1. The previous step's `NB_BAND_REACH` for the tile and its 26 neighbours has no bit set. That means no crossing within two tiles, which is 8 or more cells.
2. The local speed certificate (the existing `localSpeed` kernel) bounds displacement to max|u|·dt < 8h − stencil reach (3h). Otherwise the tile is traced as today.
3. No source box, released wall or body-swept region touches the tile. That region comes from compactBodies and the swept list, which already exist.

When all hold, the tile skips foot, trace, store and revision bump. Nothing else changes. Its sign cannot change, because no crossing can enter the departure stencil. Its magnitude is "≥ 5h" by the arena contract. Every consumer reads only that: sign for volume and cut cells, < 4h for particle membership, crossings for pressure ghost cells.

**Code.**
- Replace the air-only `umFarAirKept` / `umDeepAirKept` exits with this tile test, run once per workgroup into workgroup memory.
- Apply the same test before `umQueueVertex` for coarse owners. That removes the 41,487 deep 4h vertices.

**Expected saving.** About 4.5 of the 4.9 ms, because traced vertices fall from about 290k to the band set. Band set: crossing, travel and closure tiles plus a 1-tile halo, about 10–15% of h tiles.

**Compounding.** Unchanged tiles no longer bump revisions, which shrinks P4, P5 and the arena's changed sets.

**Verify.**
- Unit tests (the AGENTS.md contract):
  - Far tile, liquid and air: values are bit-identical to before the step.
  - A surface entering a previously far tile within one step at the speed-certificate limit: falls back to a full trace and matches the old path bit-exactly in the band.
  - Source, released-wall and body-swept tiles are never certified.
- A/B on seep: identical sign field after 300 steps, and identical band values (bit-identical, since the band path is unchanged).
- Seep and letters drift plus step time, and paddleboats step time.

### P2. coupleFine, surfaceClear and splat over the particle-band tile list only

- Drive coupleFine, surfaceClear and surfaceSplatOccupied from the tile list that already marks particle tiles (`particleTiles` / `NB_BAND_REACH`&1) instead of `umAllOwner` over every non-COLD h tile.
- A tile with no band particles keeps its phi, which is P1's certificate again.
- Stage the address plan and the 27 neighbour bin heads once per tile in workgroup memory, as `umLoadAddressPlan` already does, instead of resolving per vertex inside `particleSurfaceBulk`.

**Expected saving.** 1.8 → about 0.25 ms.

**Verify.**
- Unit test: coupleFine output on a tile with particles is unchanged, and a tile without particles is untouched.
- Letters drift (particle-heavy) and seep drift, plus step time.

### P3. Arena: one fused per-brick kernel, with half the workgroup memory

**(a) Fuse.** Combine arenaNearest, arenaDistance, arenaPublishCanonical and arenaPublishFinish into one kernel per brick:
- Stage the window, the tile records, reach and capacities **once**.
- Compute nearest, then distance, then publish from workgroup memory.

This removes 4–5 re-stagings of the same neighbourhood and 3 launches with their tails.

**(b) Halve the workgroup memory.**
- Pack two candidate-queue entries per u32: an 8-bit vertex lane plus a cell code within the 10³ halo. This needs `nbArenaPack`'s code to fit 8 bits after re-basing, which is unverified; otherwise process the two queues in halves. Either way the queue drops from 8.2 to 4.1 KB.
- Reuse the 1000-word nearest array for the publish capacities, since their lifetimes are disjoint.

That brings the kernel to about 8–9 KB, doubling resident groups and occupancy.

**(c) Shrink the dispatch.** Use an indirect dispatch sized to the brick count, not a fixed 1024 × 256.

**Expected saving.** 3.7 → about 0.6–0.8 ms.

**Verify.**
- Unit test: a bank-A and publication parity test, bit-exact against the unfused kernels on a randomized brick set, including page-face vertices (217 per brick).
- Drift plus step time.

### P4. Phi resolve on change only

- Each producer (advect, couple, redistance) appends tiles whose 4h lattice changed to the existing dilated changed list.
- Run `resolveListed` over that list instead of four full `resolve` launches.
- With P1, the changed set is band-only.

**Expected saving.** 0.94 → about 0.1 ms.

**Verify.**
- Unit test: hanging values equal a full resolve after each producer.
- Seep drift plus step time.

### P5. Geometric fill on change only

- Drive geometryRegional and geometryChanged from tiles whose phi revision or solid capacity changed: the revision atomics bumped in `umStoreSurface`, plus the body-swept list.
- Use indirect dispatch sized to that list.

**Expected saving.** 1.47 → about 0.15 ms.

**Verify.**
- Unit test: geometry textures are bit-identical to a full fill after a local edit and after a body sweep (the AGENTS.md "local edits / untouched fidelity" contract).
- Drift plus step time.

### P6. Coarse retirement: indirect dispatch sized to `NB_COARSE_LIST`

Use one lane per entry and drop the fixed 1024 groups.

**Expected saving.** 0.36 → about 0.05 ms.

**Verify.** Existing retirement tests, plus drift.

### P7. Volume diagnostic and gauge driven by the band

- Seed from the crossing-mask tiles.
- Take dilation for free from `NB_BAND_REACH`, which removes the seed, 4× dilate, grow and measureGrow sweeps.
- Run metric and measure over the band list.
- Keep an incremental bulk sum, updated only on tiles in band ∪ solid-capacity-changed ∪ topology-changed.
- measureInitial uses the same list.

**Expected saving.** Probe steps 2.7 → about 0.15 ms; live gauge 0.07 → about 0.02 ms.

**Verify.**
- Unit test: the volume equals a full-sweep measurement on seep and letters at sampled steps (≤ 1e-6 relative).
- Save/load/undo of the bulk sum.

### Expected live stage after the plan

| item | now (ms) | after (ms) |
|---|---|---|
| Surface advect + deferred (P1) | 4.91 | 0.35 |
| Volume gauge (P7) | 0.07 | 0.02 |
| Phi resolve (P4) | 0.94 | 0.10 |
| Geometric fill (P5) | 1.47 | 0.15 |
| NB-FLIP surface reconstruction (P2) | 2.09 | 0.25 |
| NB-FLIP arena search + metric + publish (P3) | 3.38 | 0.6–0.8 |
| Coarse retirement (P6) | 0.36 | 0.05 |
| **Total** | **13.2** | **≈ 1.5–1.7** |

**Honest gap.** Under 1 ms needs the arena at about 0.3 ms. That depends on the brick count: 1844 bricks × 729 vertices is about 1.3M distance vertices. Reaching it needs one of:
- **P8 (a decision for you):** cell-granular reach. Skip candidate solves for vertices whose |phi| ≥ 6h, since membership only needs < 4h. Today's reach is tile-granular at 3 tiles (12 cells), so this is a contract change.
- The census agent's cut of the "join" h tiles (2320), which shrinks P1, P2 and P5 further.

### Order and gates

The order is P1, then P2, P4 and P5 (all fed by P1's unchanged tiles), then P3, P6 and P7.

The first step of P1 is a counter pass: band tiles vs h tiles per step on paddleboats, seep and letters. This confirms the 10–15% band fraction I estimated from census reasons. I did not measure phi magnitudes per vertex.

Gates follow the GPU gate policy: seep 300, letters 120 and the rest at 60 per commit, plus paddleboats `--bodies` step time. Each gate reports seep and letters drift and step time, together with `npm run check:types` and `npm run test:unit`.
