# Pressure stage on paddleboats: why it costs ~14.5 ms, and a plan to get it under 1 ms

## Sources

- **xctrace counters** (parent capture, `artifacts/xctrace-paddleboats/`). Paddleboats with `--bodies`, split dispatches, 43 advances in the window. My own reduction of the exported tables is `scratchpad/xc-reduce.mjs`. It matches the parent's `xctrace-kernels.txt` to within 2%.
- **Static reading** of `lib/methods/uniform/uniform-pressure-arena.ts` (A below), `uniform-mixed-pressure-velocity.ts`, `uniform-mixed-pressure-authority.ts`, `uniform-mixed-topology.wgsl.ts` and `uniform-brick-arena.ts`.
- **Dispatch census** (`scratchpad/pressure-count.mts`). A check run on paddleboats with the bodies driven, counting frames 241–300 per (pass, entry point): direct and indirect dispatches and workgroups, plus the arena header every 20 frames. Output: `scratchpad/pressure-count.txt`.
- **Two timing runs** (`tools/probe-reservoir-bulk.ts --scene=paddleboats --bodies --steps=120 --kernels=60`). Patches were applied and reverted inside the locked command. Results are in §3:
  - A = baseline, B = `patches/hoist.patch`, in one command;
  - C = `patches/nobody.patch`.

Scene numbers (parent's run and my census at frame 300):

| Quantity | Count |
|---|---|
| Band tiles | 2,713, i.e. 173,632 h rows |
| h tiles | ~4,852 |
| Tiles with census reason "crossing" | 52 |
| Wet 4h tiles | ~37.7k |
| Resident pressure pages | 192 |
| Cycles executed | 3–4 |
| Level-1 bricks / level-2 pages | 137–141 / 192–196 |

- **Residual history per cycle:** 20.8 → 2.15 → 0.34 → 0.062 → 0.015, about 6–10× per cycle. Krylov L2: 1.46e6 → 2.3e5 → 5.5e4 → 8.2e3.
- **Launch widths per dispatch (census):**

| Width | Workgroups | Live work | Over-launch |
|---|---|---|---|
| tileGroups | 3,273 | 2,549 band tiles | 1.28× |
| pRelax | 1,637 | — | — |
| level1Groups (pDownA, pUpA) | **1,024 × 512 lanes** | ~140 level-1 bricks | **7.3×** |
| pages (pNativeRelax, pDownB, kApplyNB) | 241 | 192 pages | — |
| kApplyT | 3,273 | — | — |
| mixed rhs, project, toPressureCoarse | 1,494 | — | — |
| toPressureFine, pMark | 4,096 | — | — |

Stage totals per advance (xctrace, ms):

| Pass label | ms | dispatches |
|---|---|---|
| Uniform pressure arena solve | 7.54 | 164 |
| Uniform pressure arena projection | 1.91 | 5 |
| NB-FLIP surface pressure authority | 1.88 | 12.7 |
| Uniform pressure arena marks and rows | 1.66 | 8 |
| Uniform mixed pressure project | 0.75 | 1 |
| Uniform mixed transfer to pressure | 0.30 | 2 |
| Uniform mixed pressure rhs (buildRhs) | 0.27 | 1 |
| Uniform pressure root owners (roKeys) | 0.10 | 1 |
| **Total** | **14.5** | **~200** |

**Headline: the GPU is idle, not busy.** Apart from pRelax, every kernel runs at:

- 2–25% compute occupancy;
- 5–30% ALU;
- 1–13 GB/s read bandwidth on a part with about 400 GB/s.

This is the signature of serial dependent-load latency, either inside per-lane loops or in chains of tiny dispatches. The two causes are (a) and (b):

**(a) Per-cell address resolution through 2–5 dependent loads.** It is repeated for every face of every row, in every kernel that touches a row:

- `pBandCell(q)` = `umCellPresent` (page table) → `umTileAt` (page table again) → `atomicLoad state[SLOTS+tile]` (A:554).
- `umOwnerAt(q)` = page table → tile word → `umTopology[tile]` → `phi[index]` (`uniform-mixed-faces.wgsl.ts:17`). `bNeighbour` calls it twice for an h neighbour (A:684–689).
- `pAt(q)` = `pBandCell` + `ubaBrickAt` (page table) + `ubaSlot` (atomic) (A:625).
- `nAcross` = `ubaBrick` (atomic) → `ubaOrigin` → page table → pool (A:645, used in pRelaxRow at A:880).
- Every index-table read (`pIndex`, `ubaSlot`, `ubaBrick`, `ubaLevelRecord`) is an `atomicLoad` (A:540, `uniform-brick-arena.ts:214–217`). The compiler can neither hoist nor merge these.

Most of these lookups are resolving addresses that are identical for all 64 cells of a tile:

- 75% of a cell's face neighbours lie inside its own 4³ tile.
- All 16 subfaces of a tile face lie in one neighbour tile.

**(b) About 200 dispatches, most of them tiny.** 57.5 dispatches a step go to the levels 3–5 tail, which runs 12k + 1.5k + 192 rows. They take 13–33 µs each at 2–13% occupancy.

## 1. Per-kernel table

Times are ms per advance from xctrace: count × µs per dispatch. Columns:

- **Extent:** the launch.
- **Useful:** the items that do work.
- **Loads/trips:** per thread.

### Marks and rows (1.66 ms)

**pRows: 1.52 ms (1 × 1,591 µs). Occupancy 10.9%, ALU 17%, read 9 GB/s.**

- **Extent:** `tileGroups` = min(workTiles, 4096) workgroups of 64 lanes, grid-striding over 2,713 band tiles. Code: A:466, A:1496–1640.
- **Useful:** 173.6k rows. All rows are rebuilt every frame.
- **Per-row work, once per row:**
  - `umOwnerAt(p)` + phi;
  - `umSolidRegionClearAtBuild`, `umCellIsOpen(p)`, `umCellInsideSolid(p)`, `umCellOpen(p)`;
  - `umBodyPoseNearTile`, a loop over every body (`uniform-mixed-solid.wgsl.ts:1259`);
  - `bHeadCapacity`.
- **Per-row work, for each of 6 faces:**
  - `bForcedField`: a texture load plus `umFrozenFace`;
  - `umCellKind(q)` at tile faces;
  - `bNeighbour(q)`: 2 × `umOwnerAt`, 4–5 dependent loads, plus `bCentrePhi`, which is 8 texture loads for 4h;
  - `umCellIsOpen(q)` and `bFacePhi` (4 texture loads) on wet/dry faces;
  - `pBandCell(q)` (3 dependent loads, 1 of them atomic) and `bNative(q)` (2);
  - aperture lookup;
  - 3 plane stores.
- **Load count:** about 6 × 10 + 15 ≈ 75 loads per row in dependent chains 3–5 deep. That is about 13M loads per frame.
- **Per-tile prologue:** 2 barriers, 6-face `umSolidValid` tests on every lane, and a 4-lane aperture verdict.
- **Confirmed cause, about 60%: rigid-body geometry evaluated per row.** In C, these were removed from pRows:
  - `umCellOpen(p)`: its body pose is never current here (16 motor paddles move every frame), so `umAnyBodyNear` is an 18-body loop and the geometry path runs `umCellBodyFraction`, another 18-body loop (`uniform-mixed-solid.wgsl.ts:1485–1500`);
  - `umCellInsideSolid(p)`: `umBodyAt`, an 18-body loop (:1523);
  - `umCellIsOpen(p)`;
  - `umBodyPoseNearTile`: an 18-body loop per row (:1259);
  - `bHeadCapacity`.

  pRows went from 1.50 to 0.58 ms with a band equal or larger (2,344–3,480 tiles against 2,293–2,463).
- **Remaining 0.58 ms:** latency of per-face chains at 10.9% occupancy:
  - `bForcedField` → `umFrozenFace`: 2 × (`umTileKey` → page table → cold texture);
  - `bNeighbour` and `pBandCell` across tile faces;
  - the aperture prologue.
- **The hoist of interior-face lookups saved nothing** (B, §3). Those chains run in parallel with the face's other chains and are not on the critical path.
- **The data volume is small:** about 20 MB a frame, which is 50 µs at 400 GB/s.
- **Evidence:** read bandwidth is 9 GB/s, 1/40 of peak; A/B/C in §3.

**pRowsG, pMark, pClear2, ubaUpdate, ubaLists, ubaRecord, ubaLevel1: 0.14 ms in total (7 dispatches).** Launch-bound; each takes 7–43 µs.

### Solve (7.54 ms)

**pRelax: 1.23 ms (23.9 × 51.5 µs). Occupancy 23%, read 60 GB/s, LLC 42%.**

- **Extent:** `relaxGroups` = 1,357 × 64 lanes. Each workgroup takes 2 tiles, one red cell then one black cell per lane, with a `storageBarrier` between colours (A:2275).
- **Useful:** 173.6k rows per sweep.
- **Loads per row:** atomic `TLIST` → kinds → RHS and DIAG → 6 × (X, plus C if the face is not unit). 0.75 of a cross-brick `ubaLevelRecord` (atomic) on average. Seams through `nAcross`, 4 dependent loads.
- **Walls:** the clamp loop runs 4 passes and re-reads `P_C` and `pLift` (poolB) for each wall face (A:885–897).
- **Data volume:** about 3.1 MB per dispatch reaches DRAM, about 40 B per row through the LLC.
- **Cause:** the sweep count (24 per step = 4 cycles × (3 pre + 3 post) + settle, Krylov-gated) × all 173.6k band rows. Each dispatch runs at about 2.5× the bandwidth floor because the address and atomic chain is 4 loads deep.
- **This is the only kernel doing proportionate work.** It scales with the band row count, which the census sets.

**pRebase, including pRebaseC: 0.69 ms (4.8 × 144 µs). Occupancy 21%, ALU 30%, read 30 GB/s.**

- **Extent:** `tileGroups`.
- **Per row:**
  - 6 faces, each with `pNear` plus `P_C`;
  - every seam face calls `bSeamHead`: 2 tangential axes × 2 sides of `pNear0` + kinds + base, i.e. 4–12 loads (A:599–622);
  - 2 atomicMax per lane;
  - an 8-lane level-1 bake: 7 stores each;
  - 3 barriers per tile.
- **Cause:** it repeats every cycle (`pRebaseC`), recomputing the seam heads and couplings from `P_BASE` and kinds. Both are constant for the whole solve. Only the active set (`P_CORRECT`) changes.

**pNativeRelax: 0.53 ms (15.3 × 34.6 µs). Occupancy 18%, ALU 6%.**

- **Extent:** `pages` workgroups of 512 lanes, over level-2 pages (A:2098).
- **Useful:** half the lanes per colour. Wet 4h rows are about 37.7k of 98k slots.
- **Seam faces** loop over 16 band cells, loading kinds + C + X each (48 loads).
- **Walls** use the same 4-pass clamp.
- **Cause:** 15 launches of a page-wide grid in which most lanes are idle (wrong colour, inactive or air), plus 16-way seam gathers.

**Levels 1 and 2 transfers: 2.0 ms.**

| Kernel | ms | Count × µs |
|---|---|---|
| pDownA | 0.51 | 3.8 × 133 |
| pDownB | 0.36 | 3.8 × 93 |
| pUpA | 0.28 | 3.8 × 74 |
| pUpB | 0.23 | 3.8 × 59 |
| pResidual2 | 0.27 | 4.8 × 56 |
| pDown2 | 0.16 | — |
| pNativeOperator | 0.13 | — |
| pNative | 0.33 | 1 × 341 |
| pRestrict0 | 0.19 | — |
| pProlong0 | 0.105 | — |

- **Extents:** pDownA, pUpA and pDownB use `g` / `pages` workgroups of 512 lanes over level-1 and level-2 slots.
- **Occupancy:** 14–30%, with read below 13 GB/s.
- **pNative:** every native row loops over 16 seam subfaces with `bSeamHead` on each (A:1730ff).
- **Cause:** latency, through one 512-lane workgroup per page and per-row gathers through records. Each kernel runs once per cycle.
- **Confirmed launch-shape waste in pDownA and pUpA:** they launch `level1Groups` = 1,024 workgroups × 512 lanes for about 140 level-1 bricks. 884 of the 1,024 workgroups exit after a count load.
  - **Why:** `observeWork` floors every width at the census reserve, `UNIFORM_WORK_RELAYOUT_RESERVE` = 1,024. That reserve is counted in h tiles but applied to bricks and level-1 widths too (`uniform-mixed-frame.ts:1043–1045`, A:470–476, `uniform-buffered-work.ts:10,30`).
  - The same floor sets tileGroups to 3,273 against 2,549 band tiles.

**Tail ladder, levels 3–5: 1.31 ms over 57.5 dispatches.**

| Kernel | ms | Count × µs | Occupancy |
|---|---|---|---|
| pDown3 | 0.25 | 11.5 × 22 | 6% |
| pDown4 | 0.15 | 11.5 × 13 | 4% |
| pTail | 0.37 | 11.5 × 33 | 1.9% |
| pUp4 | 0.22 | 11.5 × 19 | 7.5% |
| pUp3 | 0.32 | 11.5 × 27.5 | 13% |

- **Useful rows:** 192 pages × 64 = 12k (8h), about 1.5k (16h) and 192 (32h).
- **Schedule:** `tailCycles` = 3 inside every V-cycle × 4 cycles (A:3683–3755).
- **pTail** is 1 workgroup of 256 lanes running 32 red-black sweeps with workgroup barriers (A:2851). It is serial by design.
- **Cause:** a chain of dispatches with almost no parallel work. Each dispatch pays a pipeline drain and refill of about 10–30 µs.

**Krylov GCR(1): 1.06 ms.**

| Kernel | ms | Count × µs | Notes |
|---|---|---|---|
| kApplyT | 0.32 | 2.9 × 112 | |
| kApplyNB | 0.335 | 2.9 × 117 | 16-iteration loops in the BULK16 branch (A:1301) |
| kStep | 0.13 | — | |
| kStart | 0.11 | — | |
| kReduce | 0.106 | 2.9 × 37 | Occupancy 0.2%: one workgroup |
| kBound | 0.06 | — | |

- **Cause:** 5 dispatches per cycle. kReduce is a serial single-workgroup reduction.

**pInit, pFinal, pBulk\*, pWrite, pLaunch: 0.17 ms.** Launch-bound.

### Projection (1.91 ms)

**pProjectNative: 1.165 ms (1 × 1,220 µs). Occupancy 15.6%, read 3.2 GB/s, write 0.5 GB/s.** Hoisting `pAt` out of the 16-subface loop saved 0.107 ms (−10%, B).

- **Extent:** 6 × 2,713 = 16,278 jobs of 64 lanes (A:3123–3156).
- **Every job, including ones that are then skipped:**
  - an atomic `TLIST` load;
  - 2 workgroup barriers;
  - 7 lanes each running a `pBandCell`/frozen test (3–4 dependent loads).
- **Surviving jobs** (root tiles beside the band, once per root):
  - 6 lanes each run `bRootFace`, a serial 16-iteration loop;
  - each iteration is `pAt(q)` (5 dependent loads: page table ×2, atomic slot, page table, atomic `ubaSlot`) + `pVolume` + `bFace`;
  - `bFace` is about 8 loads, plus `nAcross` (4) and `bSeamHead` (4–12) on seam faces;
  - 58 of 64 lanes wait at the barrier.
- **Load count:** 16 × ~20 ≈ 300 dependent loads on one lane per face.
- **Cause:** serial loops on 6 of 64 lanes behind 2 barriers per job, for 16k jobs. Read is 3.2 GB/s, i.e. 1% of peak.
  - Confirmed: the per-subface `pAt` chain is about 10% of it.
  - Likely, not measured: the rest is each subface's `bFace` (`umFrozenFace`, 4 loads; `nAcross` + `bSeamHead`; `umSolidFaceVelocity`'s body broad phase on K_CLOSED faces) plus the 16k barrier pairs.
- **The work is also redundant.** The 16 subface fluxes are the same `bFace` values pProject computes for the band tile across that face.

**pProject: 0.56 ms (1 × 590 µs). Occupancy 10.6%, read 13 GB/s.**

- **Per row:**
  - 3–6 `bFace` calls, with `nAcross` + `bSeamHead` on seams;
  - `bReleased` × 3–6, each with `umCellIsOpen(p)` and `umCellIsOpen(q)` solid queries;
  - then 3 axes of `umCellPresent` + `pBandCell(q)` and an inner `pBandCell(r)` loop (A:3197–3204).
- **Wasted lookups:** for 3/4 of the rows on each axis, q lies in the row's own band tile, but the full chain still runs.
- **Cause:** dependent-chain latency at low occupancy. The `pBandCell` hoist below the row saved 0.019 ms (3%, B). The rest is `bReleased`'s per-face `umCellIsOpen` pair and `bFace`'s frozen and seam chains.

**pCopyNative: 0.144 ms.** About 98k-lane page jobs; occupancy 27%; reasonable.

**pCopy: 0.035 ms.**

### Mixed pressure: rhs, project and transfer (1.32 ms)

**Mixed :project: 0.75 ms (1 × 789 µs). Occupancy 11%, read 13 GB/s.**

- **Extent:** `umResidentAllOwner` over 1,492 workgroups × 64 = 95.5k lanes, one per resident 4h tile (`uniform-mixed-topology.wgsl.ts:186–194`, dispatch at `uniform-mixed-pressure-velocity.ts:603`).
- **Useful:** about 37.7k wet 4h tiles, plus the air faces beside them.
- **Per lane:**
  - `umResidentPageTile`, `umTileLive` and `umTileWidth`;
  - for each of 6 faces, `umFaceFirst` (`umOwnerAt` chain, 4–5 loads) plus `umPressureLiquid(neighbour)`;
  - bulk-row checks;
  - `umColdLow` and `umFrozenFace`.
- **Cause:** about 60% of the lanes are air or dead tiles, and each live face is a 4–5-load chain.

**buildRhs: 0.27 ms.** Same launch, same cause.

**toPressureFine and toPressureCoarse: 0.21 + 0.09 ms.** Latency: occupancy 10–13%, read 7–10 GB/s.

**roKeys: 0.10 ms.** Occupancy 8.5%, read 1.5 GB/s.

### NB-FLIP surface pressure authority (1.88 ms)

| Kernel | ms | Count × µs | Occupancy | Read |
|---|---|---|---|---|
| surfaceHeld | 0.605 | 1.9 × 317 | 10.3% | 4.9 GB/s |
| surfaceRegions | 0.495 | 1 × 507 | 10.3% | 1.1 GB/s |
| wetRegions | 0.40 | — | 7.7% | — |
| cutRegions | 0.25 | — | 9.8% | 3.1 GB/s |
| stamp + stampHeld | 0.08 | — | — | — |

**surfaceHeld** (`uniform-mixed-pressure-authority.ts:684`):

- Items: held regions + every h tile (umCounts.x ≈ 4,852), one 64-lane workgroup each, twice a step.
- Each item re-runs `umSurfaceOwner` unless its stamp proves it unchanged.
- **Cause:** in the wake, every h tile counts as changed, and each owner runs the closed-owner neighbour loop through `umOwnerAt`.

**surfaceRegions** (:722):

- Items: 8 × held pages, one workgroup each.
- Each item does a 27-lane `umTileKey` + `umBodySweptReach` scan, then 8 lanes each loop over 27 bits, then the h tiles run serially, 8 per workgroup.
- **Cause:** a 27-neighbour indirect scan repeated for every eighth of every region, with serial h tiles.

**wetRegions and cutRegions** (:474, :516): each item is a region with the same 27-neighbour stamp scan. Both launch over `min(REGION_GRID, pages)`, so they scan every page.

**Overall cause:** the change detection is redone per kernel through indirect neighbour scans over all regions and pages, and the authority is recomputed per owner even where nothing changed.

## 2. Arithmetic floor

At the current band size of 173.6k h rows:

| Item | Traffic | Time at about 300 GB/s achievable |
|---|---|---|
| Row assembly | ~110 B/row ≈ 19 MB | 63 µs |
| One sweep | ~40 B/row ≈ 7 MB | 23 µs |
| One V-cycle at level 0 (6 sweeps + residual + restrict + prolong + rebase) | ~9 sweep-equivalents ≈ 63 MB | 0.21 ms |
| 3 cycles + Krylov (2 applies per cycle) | — | about 0.75 ms |
| Projection | ~15 MB | 50 µs |

**The floor alone is about 0.9 ms with 173.6k band rows.** So the "under 1 ms" goal needs both of these:

1. Every kernel at bandwidth instead of latency (§4, P1–P7).
2. A band sized to the surface. 52 crossing tiles need about 52 × 27 = 1.4k tiles at most with their 3³ halo; the census now keeps 2,713. That is the census fork's lever. Pressure cost is linear in it, and a band of 1,000 tiles gives a floor of about 0.35 ms.

## 3. Exact-hoist A/B (measured)

Per-kernel means (gpu-kernel-timer, ms per step, steps 60–120):

| Kernel | A: baseline | B: exact address hoist | C: B + no per-row body geometry in pRows |
|---|---|---|---|
| pRows | 1.503 | 1.533 | **0.583** |
| pProjectNative | 1.074 | **0.967** | 1.064 |
| pProject | 0.564 | 0.545 | 0.576 |
| pRebase | 0.660 | 0.652 | 0.698 |
| pRelax | 1.183 | 1.207 | 1.234 |
| mixed project | 0.783 | 0.791 | 0.764 |
| band tiles, steps 60/90/120 | 2293/2457/2463 | 2224/2406/2542 | 2344/2495/3480 |

**What the patches changed:**

- **B (`patches/hoist.patch`):** same arithmetic, arithmetic addressing.
  - pRows: same-tile faces skip `bNeighbour`/`pBandCell` and read `phi[base ± stride]`.
  - pProject and pCopy: a cell below inside the tile is known to be band.
  - pProjectNative: one `ubaSlot(ubaBrickAt)` per root face instead of 16 `pAt` calls.
- **C (`patches/nobody.patch`), timing attribution only:** in pRows, `umCellOpen`, `umCellInsideSolid` and `umCellIsOpen(p)` are replaced by `umSolidValid`, and the bodies term is dropped. The physics is wrong by design: drift was −0.017 against −0.007.

**What the results say:**

- **The paddleboats run is not deterministic run to run.** A and B diverge by step 60 (band tiles 2,293 against 2,224), so B could not be checked as bit-identical end to end. Bit-identity of the patched kernels needs a unit test.
- **Address-chain hoisting is worth about 10% where it removes a serial in-loop chain** (pProjectNative). It is worth nothing where the chain runs in parallel with other per-face chains (pRows interior faces).
- **The confirmed large cause in pRows is the per-row rigid-body loops: 0.92 ms of 1.50.** This revises the "indirect loops" hypothesis for this kernel: the loops are over bodies, through geometry, repeated per row, and caused by a stale record.

## 4. Plan to under 1 ms

All items are general: no scene checks, no clamps, no fast paths. Each removes work or address-chain latency. Expected savings are per advance on paddleboats, scored by xctrace or per-kernel time, not by dispatch count.

### P0. Body geometry once per step, not per row (confirmed −0.9 ms in pRows)

**Change.** After rigid publication, run a record refresh over a compact list of body-swept tiles. The list is the tiles whose region `umBodySweptReach` ≤ reach, already computed by the record pass. One lane per cell evaluates `umCellOpenGeometry`, inside-solid, and the face apertures at the published pose, and writes them into the existing solid record rows. Set `umSolidBodyPoseCurrent` for those tiles.

Every downstream query then takes the record path:

- pRows' `umCellOpen`, `umCellIsOpen`, `umCellInsideSolid` and `bHeadCapacity`;
- pProject's `bReleased`;
- mixed project's `umRelease` (`umCellOpen`);
- K_CLOSED `umSolidFaceVelocity`, using the stored face velocity per body-cut face.

Cost is proportional to the cells beside moving bodies, evaluated once, instead of every band row running 18-body loops several times.

Also store per-tile body candidates as a u32 mask over at most 32 bodies, from the same broad phase. The remaining geometry callers then use `umBodyAtCandidates` (`uniform-mixed-solid.wgsl.ts:1465`), which already exists, instead of full loops.

**Expected savings.** pRows 1.50 → about 0.6 (measured by C). pProjectNative, pProject and mixed project: likely −0.1 to −0.3 more, not measured.

**Coordinate with the "all-4h solid record" stage** (buildRegions and bodyMask, 2.0 ms), which another fork owns. This should be the same refresh, not a second one.

**Verification:**

- A unit test that the refreshed record equals the geometry evaluation at the published pose for cells near a moving box.
- Seep and letters drift.
- Paddleboats drift unchanged against baseline (within run noise).

### P1. Tile frame: resolve addresses once per tile, not per cell-face

**Measured caution.** Hoisting the interior-face lookups alone gave 0 in pRows and 3% in pProject. It gave 10% in pProjectNative, where it removed a serial in-loop chain. The tile frame pays off only where a chain is serial: in-loop, or the per-row prologue. Do it as part of P2/P3/P6, not as a standalone pass. Estimates below are not measured.

**Change.** Every band-tile kernel loads its tile's 3³ neighbourhood into workgroup memory with one parallel 27-lane load at the start of each tile iteration. The kernels are pRows, pRowsG, pInit, pRebase, pRestrict0, pProlong0, pRelax, pProject, pCopy and pProjectNative. Each neighbour gets one compact descriptor:

- band slot or 0;
- brick slot and tile bit;
- owner base and width;
- native flag;
- page word.

Every per-cell neighbour or seam lookup then becomes in-tile arithmetic plus one data load:

- `pBandCell`;
- `umOwnerAt`;
- `pAt`;
- `pNear0` across bricks;
- `nAcross`;
- `bNative`;
- `umCellKind`.

The same tile frame carries the per-tile solid verdicts: `umSolidRegionClearAtBuild`, body-near, aperture rows.

**Load counts.**

| Kernel | Now | After |
|---|---|---|
| pRows | ~75 loads per row, chains 3–5 deep | ~25 loads, chains 1 deep |
| pProject | ~40 per row | ~20 |

`pIndex(TLIST)`, `ubaSlot` and `ubaBrick` move from atomic loads per lane to one load per tile, broadcast. A plain read-only binding of the index table, where the pass does not write it, would allow caching. That needs a second binding view; Dawn forbids aliasing a writable and a read-only binding in one pass, so this is done per pass.

This extends the existing page-level address plan (`uniform-address-plan.ts:27–75`, `umUseAddressPlan`, today off in every arena kernel) to tiles.

**Expected savings, unmeasured beyond B.** pRows 0.6 → about 0.35, mostly the frozen and cold chain per face. pProject −0.1. pRelax −0.2, from the per-tile `TLIST`/`ubaLevelRecord` broadcast instead of per-lane atomics.

**Verification:**

- The §3 A/B has to be bit-identical: the patch changes addressing only.
- Seep and letters drift and step time.
- New unit regressions comparing the pRows and pProject outputs against the current kernels on a randomized mixed band (`tests/`, as the arena tests do).

### P2. Seam fluxes from pProject; pProjectNative over a compact root list

**Change.**

1. pProject's tile workgroup already computes `bFace` on every band face. Add one workgroup reduction to it: on each of the tile's 6 faces, the volume-weighted sum and weight of its 16 subface fluxes. Write the result to a per-band-tile seam-flux record (6 × 2 floats per tile, a pool plane).
2. pRows, which already visits every band tile and face, appends each root tile beside the band once, using the existing first-band-face rule (A:3141–3144), to a compact `ROOTS` list.
3. pProjectNative becomes one lane per (root tile, face) over `ROOTS`. Each lane reads the seam-flux record, or `textureLoad(field)` on a non-band face, then interpolates.

This removes 16,278 jobs × 2 barriers and the 6 × 16-iteration `pAt` + `bFace` chains.

**Expected savings.** 1.165 → about 0.03 ms. pProject gains about 10 µs.

**Verification:**

- Output texture bit-equal on a unit test. The summation order changes, so allow ulp-level tolerance and pin the order (fixed tree reduction) for determinism.
- Seep and letters drift.

### P3. Constants of the solve computed once, not per cycle

**Change.** Within a step, `bSeamHead`, the seam couplings and `nAcross` targets depend only on `P_BASE` and kinds. pRebase stores the seam head per seam face, in a plane alongside `P_U`/`P_V`, and the native link per seam face. These are then read with one load each:

- pRebaseC;
- pRelaxRow seams (A:880);
- pRestrict0;
- pResidual;
- pNative and pNativeRelax seam loops;
- bFace.

pRebaseC then only re-derives the active-set parts. In the wall clamp, read `P_C` and `pLift` once into registers before the 4 passes (A:885–897, and native :2121–2131).

**Expected savings.** pRebase 0.69 → about 0.25; pNative and pNativeRelax about −0.3; pRelax about −0.1. Total about −0.8 ms.

**Verification:**

- Bit-identical residual history: header words `HISTORY` (h[8..12]) and `INITIAL` (h[26]) equal across A/B.
- Seep and letters drift.

### P4. Collapse the coarse tail: 57.5 dispatches → about 8

**Change.**

- **Levels 4–5** (about 1.7k rows) run fully inside the single tail workgroup: x and r live in workgroup memory, couplings come from L2, and there are no dispatch boundaries.
- **Level 3** (8h, 64 rows per page) runs as a page-local block smoother inside pDown3 and pUp3, which already run per page.
- **Each V-cycle's tail** becomes pDown3′ → pTail′ (levels 4–5, iterated inside the workgroup until its own residual drops) → pUp3′. That is 3 dispatches instead of 15.
- **tailCycles** goes from 3 to 1, but only if the measured cycle residual history is unchanged or better. This is gated on the header's `HISTORY` words, not assumed.

**Expected savings.** 1.31 → about 0.15 ms.

**Verification:**

- Per-cycle residual history and cycles executed (header h[4], h[8..12]) on seep, letters and paddleboats.
- Seep and letters drift.
- Unit tests of the arena's tail on the rest tank.

### P5. Krylov, launch shape and converged cycles

**Launch shape.** Dispatch pDownA, pUpA and pRestrict/prolong-level work indirectly from the GPU's own level counts (`ubaLevelCount`, written by ubaLevel1 in the same frame), through the existing `pLaunch` arguments. Do not use the fixed-lag width floored at the 1,024-tile census reserve. This removes 884 empty 512-lane workgroups per call on 8 calls a step, about −0.1 to −0.2 ms, to be confirmed. It is general: the GPU count is exact for every scene.


**Change.**

- Fold kReduce into kApplyT and kApplyNB. Each writes per-workgroup partial dot products; the last workgroup to finish, found by an atomic ticket (legal in WGSL), reduces them. This removes 2.9 dispatches of a single workgroup at 0.2% occupancy.
- Fold kBound into kStep.
- Route every per-cycle dispatch after a converged verdict through the existing `pLaunch` indirect arguments (already used for BULK16 live), with 0 workgroups once `bConverged`. Today, converged cycles still launch full grids that exit early.

**Expected savings.** About −0.25 ms, plus 5–15 µs for each dispatch skipped after convergence.

**Verification:**

- Residual history equal.
- Step time.
- Kernel census through `--dispatches`: the indirect launches go to 0 workgroups after the verdict.

### P6. Mixed project, rhs and transfer over a live 4h list

**Change.**

- buildRhs already visits every resident tile. Have it append the live 4h owners (liquid, a liquid neighbour across any face, or detached mass) to a compact list, using a subgroup or workgroup prefix sum and one atomic per workgroup.
- `project`, `toPressure*` and `roKeys` dispatch indirectly over that list. They use the tile frame (P1) for `umFaceFirst` neighbours.

This removes about 60% of the lanes (air and dead) and the per-face `umOwnerAt` chains.

**Expected savings.** project 0.75 → about 0.15; toPressure about −0.15. Total about −0.75 ms.

**Verification:**

- Face texture bit-equal on unit tests, comparing a residentAll run against the list run.
- Seep and letters drift.

### P7. Authority: one change list per step

**Change.**

- stamp and stampHeld already compute which regions changed and which bodies' swept reach they lie in. Make them emit one deduplicated tile list, `{tiles whose 3³ reach touches a changed or body-reached region}`. Use a per-tile stamp word as the dedup key; it already exists as `UM_TILE_BASE`.
- One dispatch, a lane per owner, runs `umSurfaceOwner` and the wet and cut fits over that list.
- This replaces four region scans (`surfaceHeld` ×2, `surfaceRegions`, `wetRegions`, `cutRegions`), each repeating 27-neighbour stamp lookups over every region or page.

**Expected savings.** 1.88 → about 0.4 ms. The rest is the genuinely changed owners in the paddle wakes, at about 64 µs per 1k h tiles of umSurfaceOwner at latency. P1-style neighbour framing inside `umSurfaceOwner` cuts that further.

**Verification:**

- Unit regression asserting that the list-driven authority produces the same root topology, phi masks and balance as the current scans over randomized change patterns.
- Seep and letters drift.

### Expected total

| Area | Now | After P1–P7 |
|---|---|---|
| pRows | 1.52 | 0.35 (P0 measured to 0.58, then P1) |
| Solve | 7.54 | about 3.3: pRelax 0.8, pRebase 0.25, levels 1–2 1.2 (P1 + P3), tail 0.15, Krylov 0.75, other 0.15 |
| Projection | 1.91 | 0.35 |
| Mixed rhs, project, transfer | 1.32 | 0.45 |
| Authority | 1.88 | 0.4 |
| **Total** | **~14.5** | **~5 ms** (of which about 1.1 ms of the savings is measured: P0 in pRows, plus the pAt hoist) |

**The rest is linear in the band row count and the cycle count.** Getting below 1 ms needs both:

- **P8 (census fork): size the band to surface need.** 2,713 → about 1,000 tiles is −63% on everything band-linear, which is about 3 ms of the 5. The census has 52 crossing tiles; the band holds tiles for other reasons (travel, join, closure). This is the census investigation's lever.
- **P9: fewer V-cycles, only from proven convergence.** With better smoothing at the seams from P3's exact seam heads, and the tail solved to convergence in P4, measure whether cycles drop from 3–4 to 2 at the same residual (0.007–0.018). Gate it on the residual history. Do not cap cycles.

With P1–P7 at bandwidth and P8 at about 1k band tiles, the band-linear part is about 1.6 ms × 0.37 ≈ 0.6 ms. Fixed costs come to about 0.3 ms: tail, Krylov reductions, authority over changed tiles, and the 4h list. **That gives about 0.9–1.0 ms**, which meets the target only with P8. **Without P8, the pressure stage bottoms out near 0.9 ms (floor) to 2.5 ms (realistic) at the current band size.**

## 5. Order and gates

The order is P0 → P2 → P3 → P5 (launch shape) → P6 → P7 → P4 → P5 (Krylov) → P1 (folded into the others), then P8 and P9 with the census. P0 is a record-path equivalence and needs its own regression. P2, P3 and P6 are pure reuse changes: bit-identical, or ulp-identical where a summation order changes.

For each item, at the commit:

1. `npm run check:types` and `npm run test:unit`.
2. The new regressions named above.
3. One short timing run (`tools/gpu/fluid-gpu-run.sh timing`, ≤150 steps, `--kernels`) on paddleboats with `--bodies`, reporting the touched kernels' ms and step ms.
4. Seep (300 steps) and letters (120 steps), reporting drift and step time.

At milestones, after P3, P7 and P5, take an xctrace capture to confirm occupancy and bandwidth moved, not only time.
