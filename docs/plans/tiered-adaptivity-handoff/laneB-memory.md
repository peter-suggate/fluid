# Lane B handoff: memory follows tiers (2026-10-25)

Previous owner: Claude lane B (W6, absorbing W1, W4, W7 and W10). Paths under `scratchpad/` are the previous owner's scratch and may be gone; the tools they relied on are now in the repo (`tools/probe-poison-dispatch.ts`, `tools/gpu/perblock-bytes.py`, `tools/gpu/fluid-gpu-run.sh`, `tools/gpu/gate-bundle.sh`).

## Branch and tip
- Branch: `worktree-agent-ac430bd9044c09622`. Worktree: /Users/petersuggate/code/me/fluid/.claude/worktrees/agent-ac430bd9044c09622.
- Tip: `88eebb1f`. That is main itself: my last merge, e032509d, was merged to main as 88eebb1f, and the branch fast-forwarded to it. There is nothing uncommitted and no GPU job is queued.
- Gate bundle (e032509d against a git-archive of b0bce708, `tools/gate-repeat-spread.ts` at 1.66 × spread, worst 10-frame window ratio; under 1 passes):
  - splash L128 p20, 70 frames ×2: pass, worst 0.32;
  - letters, 60 frames ×3: pass, worst 0.31;
  - seep 5 mm, 60 frames ×3, timing lane: pass, worst 0.29.
  - Under fluid.tiers start "4h", the reuse-slot commit 180129fb against f8d9f0ee: letters ×3 and seep ×3 both pass.

## What landed, and what is opt-in
Default on (bit-identical or within the gate):
- **Pool A/B arena split** (W1).
  - Pool A (`pool`, accessors `pF`/`pSet`) holds the level-0 channels x, x1, diag, base, warm, rhs, kinds and c0..5.
  - Pool B (`poolB`, accessors `qF`/`qSet`) holds u, v and every level ≥ 1 channel (xa.., xb.., nx/nb/nr/nd/nm, nk, nl, xt..), the super-block tail's rows, and the 16h bulk record (`bulkRecord`, at `bulkWord`).
  - Each pool is checked against `min(maxStorageBufferBindingSize, maxBufferSize)` (uniform-pressure-arena.ts:292).
- **W7 super-block directory**: uniform-window-address.ts and uniform-page-pool.ts, including the flat window and the narrow pool.
- **Byte cuts**:
  - the metric validity/plane words dropped (45eac62a);
  - the deferred list bounded at 729 a block + 125 an h tile (fc4d01fa);
  - the NB-FLIP reuse cache held in slots that the frame plan claims and releases, under a particle budget (180129fb, `nbReuseSlotUniform` for WGSL uniformity).
- **Arena fine slots** capped at the h ceiling (a33027e9), and a slotless block reads base (8625edfe).
- **f8d9f0ee**: pNativeRelax reads the n* rows from pool B (see Pitfalls).
- **probe-scale-ladder `--tiers=FILE.json`** (02b68407).

Opt-in:
- `fluid.tiers.start: "4h"`: start at 4h, with the fine ceiling F = min(poolTiles, 8 × brick capacity), a floor of 256 bricks and ×1.5. The seep gives F = 2048.
- `fluid.tiers.fineBlocks`: C2 compaction of the h store (304a9edd). Gated with it on and off.
- `fluid.tiers.bricks` and `fluid.tiers.particlesPerFineCell` (W10).
- `bulkTier: "16h"` (W4, default off): today a constrained 16h system on 4h storage. It saves no memory and no compute yet.

## Bytes per block now
These come from the allocation audit at e032509d: seep 5 mm, start "4h", `--steps=0`, check mode, pool 226 against pool 400 (`tools/probe-poison-dispatch.ts --audit`).
- Live: 888 MiB at 226 blocks, 1155 MiB at 400.
- **Marginal: 1570 KiB a block.** Before this session's cuts it was 1773 KiB. The target is well under 0.5 MiB, so it is not met.
- The census estimate is about 48 KiB per 4h block, 150 KiB per h brick and 32 KiB per 16h super-block. The 4h tier is about 30× over its estimate, because of h-shaped stores still sized per block.

Where the 1570 KiB goes (KiB a block):
| KiB | Store |
|---|---|
| 497 | NB-FLIP "cell counts, surface metrics and persistent regional crossing masks", mostly the per-block bins: DEPTH_A (V = 35937 words; 24 use sites), DEPTH_B (C = 32768 words; 11 sites) and surfaceHot (V words; 7 sites). This is the next cut: a tile→band-slot map under a band-tile budget, fatal on overflow. |
| 158 | Uniform mixed stage scratch |
| 104 | Uniform shared pressure scratch |
| 78 | all-4h solid record |
| 75 | pressure arena pool B |
| 74 | nearest-source origins 32×32×200 textures (×2) |
| 36 | shared frame support/certified work |
| 33, 32 | dynamic ownership census; tile update errors |
| 4 × 29 | the reference velocity A/B/D textures and the unit taps (the 4h texture parts; their h parts are capped by F) |
| 29 | 16h surface work evidence |
| 23, 22, 22, 21 | extension held values; ownership bound cubes; root topology; owners and worklists |
| ~18 each | §3.3 hierarchy up/down textures, presented pressure phi, layout builder support |
| the rest | under 15 each |

To reduce two audits to bytes a block, run `python3 tools/gpu/perblock-bytes.py <audit a> 226 <audit b> 400`.

## The 16h swing: design (not started)

### Facts that make it tractable
- The arena's levels:
  - level 2 is the native 4h rows, 512 a page (key 512·block + local);
  - levels 3, 4 and 5 are 8h, 16h and 32h per page: 64, 8 and 1 rows, at words 0/64/72 of a page in the kt channels;
  - the 32h rows go to `pTail` (TAIL_ROWS = 4096 pages), or through the super-block tail (64h, 128h).

  So **a member region already is a level-4 row**: 8 a page, one per region.
- Membership requires the region to be fully open liquid (open ≥ 1 − 1e-5), deep, not HOT and with no band tile in its 3³ region neighbourhood. Every face V is therefore 1, and a member's operator is the uniform 7-point stencil: its couplings are constants (`UNIFORM_BULK_COUPLING`: native 64, member 16, seam 25.6 per V/h4² of a 4h subface).
- Today's constrained scheme, which a compact form must reproduce (uniform-pressure-arena.ts ~573-640, 1136-1197, 1373-1404, 1432, 1998-2020):
  - member rows are excluded from the level-2 red-black smoother (`(m&N_BULK)==0u`, line ~1432);
  - their residual is reported as the region mean;
  - in `pUpB` they take the mean of the prolonged correction (`nBulkMean`), so a member changes only by coarse correction;
  - links inside a region carry 0, member-to-member links 16 per subface, and member-to-native seams 25.6 with the seam head (`nBulkHead`: the native start slope × tangential offset) written to the record's head table;
  - the operator links on the rows' start mean (`nBulkStart`), and each row keeps its offset (OFFSET), so other stages still read a hydrostatic profile.
- `uniform-bulk-tier-reference.ts` (`UniformBulkComposite`) is the CPU twin of all this (rows, residual, CG solve, projection). Extend it first, and test the compact form against it.

### (a) Member pages leave the level-2 lists: compute
1. **Whole pages only.** A page is a *16h page* when all 8 of its regions are members. Regions in mixed pages keep today's constrained form. The physics is the same, so the two forms must give the same rows.
2. **Lists** (uniform-brick-arena.ts `ubaLists`, line ~282, and `ubaLevelCount`/`ubaLevelSlot`, lines 227-228).
   - Split the page list. `UBA_PAGES` (level 2) holds 4h pages only. A new `UBA_BULK_PAGES`/count holds the 16h pages.
   - The coarse kernels (pDown3, pDown4, pTail, pUp4, pUp3, and the super tail's sDown45/sUp6) must stride **both lists**: n4 + n16, with level-3+ records for both.
   - Every level-2 kernel strides `UBA_PAGES` alone: pNative, pRhs, pNativeRelax*, the residual kernels, pDownB, pDown2, pUpB, pResidual2F and pWrite.
   - Order: the bulk selection (`UniformBulkTier.encodeSelect`) must run before `ubaLists`. Today it runs after the band's prepare, so either move ubaLists after it or take last frame's membership. Don't run a page-table walk in an inner loop: build the records once (brick-arena-core §3.9).
3. **New kernel pBulkRows** (one workgroup a 16h page, 8 lanes of rows, or 64 lanes doing 8 subfaces each).
   - It forms the 8 region rows directly. The RHS is −Σ over the region's six faces of 16 · 64 · s · ū*_face, using face means: interior shared faces cancel inside the page, so only boundary subface fluxes matter, and they equal the sum of the 64 native RHS.
   - Couplings, per face:
     - member neighbour: 16 × 16 subfaces × (p_n − p_r);
     - native neighbour: Σ over the 16 subfaces of 25.6 × (p_tile − p_r + head).
   - Write the heads to the record's head table, as pNative does.
   - **One call site** for the shared coupling code, used by both pNative's member path and pBulkRows.
4. **Coarse operator for a 16h page.** It must equal the aggregation of the constrained native operator, which is what pDownB/pDown2's bake would produce from the 64 rows. Because the couplings are constants, bake it analytically:
   - level-3 links inside a region are 0;
   - between regions, 0.5 × (sum of children links);
   - levels 4 and 5 use tailScale 0.75.

   Read `coarse()` and `bakeArrays` in uniform-pressure-arena.ts and reproduce them exactly. W4's item 3 (a rediscretised level-2 row made the coarse operator inconsistent, and the rest scene went HOT at step 60) is the warning. Keep the constrained algebra: the member is smoothed only by coarse correction plus an optional Jacobi on its 16h row, and the restriction is its region residual split over its 8 level-3 children. Test the split against the twin: at rest, it must give the same answer as the constrained form.
5. **Prolongation.** In pUpB's place for 16h pages, a 16h page's row takes the mean of its 8 level-3 corrections (`nBulkMean`'s rule).
6. **Gates.** bulkTier off must stay identical. For 16h, a rest probe (`tools/probe-reservoir-bulk.ts --closed`, drift 0, max speed as W4's) and the seep A/B: 16h constrained against 16h compact must agree within the A/A spread.

### (b) 16h pages release their block: memory
1. **Storage.**
   - A 16h page's page word gets a new kind (UM_BULK) plus a 16h slot index, and the block slot returns to the pool's free stack. Every per-block store, all 1570 KiB, is then freed.
   - 16h slots: `fluid.tiers.superBlocks` or a new `bulkPages` budget. The default is the t=0 census of eligible pages × 1.5. Allocate upfront and make exhaustion fatal, naming the tier.
   - Per slot: 8 pressures, 8 offsets, the means of 36 faces (3 axes × 3 planes × 4) for u A/B and u*, and the head table (6 faces × 16). That is about 1-2 KiB a page.
2. **Readers of a UM_BULK page.**
   - Grading (no h within a region of 16h) keeps the surface, particles, extension and the renderer away from it.
   - Stages that still read neighbours:
     - the pressure seams (above);
     - the 4h semi-Lagrangian advection: a departure point can land up to the reach, 8 h, inside a 16h page. Give it a sampler that prolongs from the face means (`uniformBulkProlong`), with one call site shared with the 16h advection;
     - region heat and COLD (`umColdFaceTerm`, `umColdCorrect`);
     - the census and volume drift: count a 16h page as fully liquid;
     - the selection itself, which must treat a UM_BULK page as passing.
   - `umTileKind(...)==UM_RESIDENT` is tested everywhere. Audit each use; don't map UM_BULK to RESIDENT blindly.
3. **16h momentum.** Advect the 36 face means semi-Lagrangian through the composite sampler, apply gravity (balanced by the hydrostatic offsets), then project from the region pressures: member-to-member gradients on face means, and at seams the subface values that the bulk faces texture already carries (`UNIFORM_BULK_FACE_TEXELS`).
4. **Transitions** (in the page pool commit, uniform-page-pool.ts).
   - Coarsen: restrict the 4h faces V-weighted into the slot, record MEAN and OFFSET, then release the block (with the existing one-commit quarantine).
   - Promote (the error test at 2× tolerance, a structural fail, or heat): claim a block, then prolong the faces and MEAN + OFFSET (hydrostatic) into fresh 4h storage before any reader runs that frame.
   - The CPU twin already proves both exact for uniform and axis-linear fields (tests/uniform-bulk-tier.test.ts).
5. **Pool sizing.** The block pool is then sized by the non-16h census × 1.5, not by every stored page.

## Arena level-list and binding constraints
- **10 storage buffers per stage.** pNative already sits at ten storage bindings. That is why the bulk record lives inside pool B, after its words at an aligned offset (`bulkWord`), instead of in its own binding. Any new 16h slot data read by pNative or pBulkRows must go into pool A or B, or into the arena index buffer. It cannot be an eleventh binding.
- **Pool size.** Each pool must fit `min(maxStorageBufferBindingSize, maxBufferSize)`. Pool B grows with blocks × UBA_STRIDE (520) × its level-1/2 channel count. Before the split, one pool capped the arena at about 1975 blocks (W1); the split roughly doubles that (my estimate, not measured).
- **Records.**
  - Arena records are 32 words a slot, and `ubaRecord` builds them for level 0 (live bricks), level 1 and level 2 (pages): n0 + n1 + n2 × 32 words.
  - A new 16h page list needs its own records at levels 3+, or the coarse kernels must read the 4h neighbours' records.
  - `ubaLevelRecord(level, slot, 27)` holds the parent.
- **Launch counts.** PAGE_GROUPS = 1024 workgroups grid-stride the page lists, TILE_GROUPS = 4096, and LIST_GROUPS and TABLE_GROUPS similarly. There is no indirect dispatch.
- **Coordination.** Lanes A and C edit uniform-pressure-arena.ts, uniform-mixed-frame.ts and uniform-mixed-remap.ts. Write a COORDINATION entry (handoff doc §6) before a large edit.

## The pool past 4096 blocks: not run
Limits to lift before going past 4096 blocks:
- `TAIL_ROWS = 4096`: the "pages" tail. Use `pressureTail: "superblocks"`, which "auto" already picks when the pool exceeds it.
- `UNIFORM_SUPER_BLOCK_SLOTS = 4096`: directory child tables.
- The pool A/B binding size above.
- Host limits printed by `probe-scale-ladder --plan`: page window, block grid and memory.

Milestone B at 4× seep, 5 mm, with 16h on may stay under 4096 resident blocks; check with `--plan` first. I never ran a pool past 4096.

## Milestone B: exact commands
Build a tree from main, then run in the timing lane. This is the scale ladder: the seep widened 2 × 2 = 4× at the same head, 120 steps, start "4h", bulkTier 16h.
```
echo '{"start":"4h"}' > /tmp/tiers-4h.json
# host plan first (no GPU):
node --import tsx tools/probe-scale-ladder.ts --cell=0.005 --width=2 --depth=2 --plan --tiers=/tmp/tiers-4h.json --values='{"bulkTier":"16h"}'
# then in the timing lane (keep the hold under ~3 min with --budget-s):
tools/gpu/fluid-gpu-run.sh timing env WEBGPU_NODE_MODULE=/Users/petersuggate/code/me/fluid/node_modules/webgpu/index.js \
  /Users/petersuggate/.nvm/versions/node/v22.22.1/bin/node --import tsx tools/probe-scale-ladder.ts \
  --cell=0.005 --width=2 --depth=2 --steps=120 --every=10 --budget-s=150 \
  --tiers=/tmp/tiers-4h.json --values='{"bulkTier":"16h"}' --out=<dir>/mb-w2d2-16h.json
# the same with --values='{}' for the 16h-off control; then --cell=0.0025 if it fits.
```
- What to report:
  - from the ladder JSON: bytes by class, per tier (h bricks, 4h blocks, 16h pages), against the census estimate (150 / 48 / 32 KiB), within 50%;
  - the 16h member count, from `UniformBulkTier.readCounts` (probe-reservoir-bulk prints it);
  - fenced frame ms (timing mode only) against lane C's 44 ms on the current defaults.
- Lane A's warning: the seep's 2×2 throat jet went fatal on page-pool exhaustion. Leave ×1.5 growth room in the pool (`--pool=`).

## Pitfalls I hit
- **Pool A/B accessors after the split.** Every native-row (n*) channel is in pool B, so read it with `poolB[...]`/`qF`/`qSet`. P_KINDS, P_C and P_X/P_X1 stay on `pool`/`pF`.
  - Lane A's merged pNativeRelax still read and wrote n* through pool A. The letters went +4% in particles at frame 40, with no fatal and no NaN.
  - Found by bisecting lane A's defaults: `nativeSmoothing: coarse` alone matched main, which pointed at the native relax. Fixed in f8d9f0ee.
  - Any new kernel that touches n*, k*, x*, y*, r* or d* channels must use pool B.
- **WGSL uniformity.** A value read from an atomic, or one that depends on one (e.g. a slot lookup), cannot guard a `workgroupBarrier`. Broadcast it through `workgroupUniformLoad`, as `nbReuseSlotUniform` does. The default path fails to compile too, because the analysis is not value-sensitive.
- **GPU test preludes go stale.** 5 of 7 GPU reuse tests fail on the base too, with "unresolved call target 'umVertexCellCapacity'". When a WGSL function is added, update the test preludes (crossing-frame-plan, crossing-list-fusion, nearest-address-plan).
- **Noise.** Main with composite smoothing shows instance-to-instance particle noise on letters (47k-50k at frame 40). Use ≥ 3 repeats, an A/A control and 10-frame windows. Don't use the pressure readback hash: it is constant.
- **GPU runs.** Run gates through `tools/gpu/fluid-gpu-run.sh gate|check|timing`, with at most 2 jobs queued.

## Scratch aids (previous owner's scratchpad/w6; may be gone, kept for reference)
- `rs5.sh <name> <base dir> <cand dir> <gate|timing> <tiers|none> <args>` runs b0bce708's gate tool.
- `gates-l.sh` is the landing bundle.
- `mktrees3.sh <tar> <name>` builds a tree plus a 4h tree that reads `W6_TIERS_FILE`.
- `tiersd.sh` runs the trace and audit probe. It needs tools/probe-poison-dispatch.ts copied from w6s2.
- `perblock2.py` gives the marginal bytes a block.
- `bulk-smoke.sh` runs probe-reservoir-bulk in the check lane.
- `until.sh` waits on a file without sleeping.
- Saved gate runs are in `rs3/*.json`; rescore them with `--rescore`.
