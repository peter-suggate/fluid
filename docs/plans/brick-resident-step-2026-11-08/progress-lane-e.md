# Lane E progress: ubaStage, B4 (projection to tail), census and T

Design: `docs/plans/brick-resident-step-2026-11-08.md`. Lane T owns B1-B3.

## ubaStage interface (for Lane T)

`lib/methods/uniform/uniform-brick-stage.ts`, `ubaStageWGSL(arena, K, options)`.
It works on top of the arena table (`arena.tableWGSL`) and pool (`arena.poolWGSL`).
There is one call for each fused kernel K, and any number of channels.

```ts
ubaStageWGSL(arena,"K",{
 level:0|2,                 // 0: h bricks (live slots); 2: 4h pages (stored blocks)
 workgroup:N,               // >= 32 lanes; record staging uses lanes 0..27
 resolve?:"fn",             // fn(word:u32, neighbourOrigin:vec3i)->u32: once a neighbour, the base a core fn needs
 fineOwned?:"fn",           // level 2: fn(c:vec3i)->bool; default: the 4h cell's tile is live in the level 0 table
 channels:[{
  name, layout:"cell"|"face"|"vertex", axis?:0|1|2 (face), halo:0..8,
  core:"fn" | {pool, component?, bank?},   // fn(word, base, l:vec3u)->f32, l = local owned index 0..7
  seam:"prolong" (level 0) | "restrict" (level 2) | {fn},  // {fn}: fn(p:vec3i, word:u32)->f32
  coarse?:"fn",   // prolong: fn(c:vec3i)->vec2f (value, known>0) at 4h owned index c
  fine?:"fn", weight?:"fn",  // restrict: fn(p:vec3i)->f32 at h owned index p
  outside?:"fn",  // fn(p:vec3i)->f32 where no page is stored (SOLID/AIR); default 0
 }]})
```

Generated WGSL:
- `ubaStage_K(slot, lane)` stages the record, then every channel.
- `ubaRecord_K(slot, lane)` stages the record only. `ubaStage_K_<ch>(lane)` stages one window, so a pass can stage channels one after another as it uses them.
- Inside a window, `ubaGet_K_<ch>(c)`, `ubaPut_K_<ch>(c, v)` and `ubaAt_K_<ch>(c)` read and write it. `c` is relative to the brick's first cell, `-halo .. 8+halo`.
- The record accessors are `ubaNeighbour_K(n)`, `ubaMask_K(n)`, `ubaBase_K(n)` and `ubaFirst_K()`, where `n = x+3y+9z` over `(-1..1)³`.
- Every entry contains barriers, so call it from uniform control flow. Each window is its own workgroup array. Their sum plus 128 words must fit in `UBA_STAGE_WORDS` (8192 words, 32 KiB).
- Owned indices follow the arena's owned-high layouts. A cell is p. A face is the +axis face of cell p. A vertex is lattice vertex p+1.

The seam rules are cited in the file header:

| level | layout | seam rule |
|---|---|---|
| 0 (prolong) | cell | 4h value plus the measured central, one-sided or zero slope (d7fa24db; exact for linear fields) |
| 0 (prolong) | face | linear between the 4h owner's two faces on the axis, constant across it. The h subfaces on a 4h plane carry the 4h face exactly, so the V-weighted sum is the coarse flux. |
| 0 (prolong) | vertex | trilinear in the known 4h corners (the hanging vertex rule) |
| 2 (restrict) | cell | V-weighted mean of the 64 h cells |
| 2 (restrict) | face | V-weighted mean of the 4x4 subfaces on the face plane (the reflux) |
| 2 (restrict) | vertex | the coincident h vertex |

Regression: `tests/uniform-brick-stage.test.ts`.
- The fixture is 4x2x2 pages with an AIR page, and half-live bricks whose odd id leaves only the low-z tiles live.
- Every staged value is checked against a CPU twin of its case: core through the record, core through a resolver, prolong, restrict, or outside.

## Baseline: HEAD 334745cb, clean export

| scene | step median (p95) | drift | notes |
|---|---|---|---|
| seep 300 | 31.5 ms (34.0) | +3.2e-4 | right tank 0.00145 m³ at step 300 |
| letters 120 | 57.7 ms (88.9) | -0.0092 | 995k particles |
| rest 16x8x4 (60, pipelined) | 4.11 ms (15.9) | 0 | steady median 3.80 |
| drop=2 16x8x4 (60, pipelined) | 16.50 ms (26.5) | -6e-8 | steady median 16.42 |
| rest 8x8x4 (60, pipelined) | 2.94 ms (11.6) | 0 | |
| drop=2 8x8x4 (60, pipelined) | 13.58 ms (18.9) | | |

Non-pressure excess for 8x8x4 drop=2 against rest. Source: Lane T's base run at 334745cb, 32 steps, `--kernels=29`, `lane-t/excess.py`.

| scope | excess |
|---|---|
| all non-pressure | **8.87 ms** (drop 15.60, rest 3.53; pressure 5.63 / 2.43) |
| extension, hierarchy, snapshot, recordDelta (B4) | 1.14 ms |
| plan tail, certificate, heat, dynamics publish (B4) | 0.80 ms |
| census, layout, remap, residency, fill, hanging, 4h cache, tiers, transfer (T) | 2.35 ms |

The B4 and T rows are regex groups of stage labels, so they are approximate.

## Decisions

- **ubaStage is a module of its own, beside `UniformBrickArena.stageWGSL`.** That older helper (used only by the pressure arena's level 2 octant passes) stages seams through a caller fn. It is pressure-side, so it is left alone. ubaStage adds the h↔4h rules, a level 2 restrict path and a per-kernel budget check.
- **B4 extension plan:** a fused 4h pass, then an h pass. Together they replace the paired retain/prepare/finish sweeps and the hierarchy restrict, sweep and prolong.
  - The 4h pass restricts the h faces and sweeps at 4h.
  - The h pass stages with halo 2 (prolonged 4h neighbours), sweeps in threadgroup memory, and publishes the canonical scratch.
  - Cited methods: Zhao fast sweeping (2005); Adalsteinsson & Sethian (1999) for the extension velocity; Bridson (2015), §4 for extrapolation.

## ubaExtend (B4's extension core)

> Superseded in B4 round 4: `uniform-brick-extension.ts` and its regression are deleted. The frame extension's per-block windows (`uniform-frame-extension.ts`) replace it; see "B4 round 4" below.

`lib/methods/uniform/uniform-brick-extension.ts`, `ubaExtendWGSL(arena, K, {level, workgroup, sweeps, spacing, reach, components, write})`.
- **Per component:** it stages the value window and the seed distance window with halo = sweeps. It runs `sweeps` Jacobi sweeps of today's `ueCombine` in threadgroup memory: the Godunov upwind root, the band cut-off at `reach` cells, and the upwind-weighted extension value. It hands each core face's (value, distance) to `write`.
- **Domain:** a seed distance of 0 marks a source, 1e20 an unreached face, and −1 (`UBA_BLOCKED`) a face outside the domain. A blocked face is never swept and is never an upwind neighbour. It is the default outside the stored pages. Seam faces are swept like any other face.
- **Exactness:** sweep s computes the window inset by s+1. The core therefore equals that many global Jacobi sweeps, which is today's slot-array result on uniform-width regions, with no seam jobs.
- **Memory:** 4 windows of (8+2·sweeps)³ words, reused across components. At sweeps=2 that is 6912 words plus the record.
- **Channel sharing:** `ubaStage` gained `window` (a channel reuses an earlier channel's array) so a kernel can stage one field after another.
- **Regression:** the third test in `tests/uniform-brick-stage.test.ts` compares each brick's core against two global Jacobi sweeps over the live faces, on anisotropic spacing.

## B4 integration plan (not landed)

**Slot space:** the frame brick store (below). The coordinator decided that Lane E builds it and that Lane T adopts it.

**Wiring the extension, once the frame store exists:**
1. **Level 2 pass** (pages, 4h spacing).
   - Seeds: `umSource`, the closed and low walls (the negative buffer).
   - Restrict seams wherever the 4h cell is h-owned.
   - It writes the 4h extended state at the base-block texels, the canonical format today's consumers read.
2. **Level 0 pass** (h bricks).
   - The core fn reads `f.velocity` through the resolver: the brick's texel origin is umFineSlot(block)<<5 plus 8 times its local index.
   - Prolong seams take the level 2 result (value and distance).
   - It writes `velocityScratch` and `negativeScratch`.
3. **Far faces** (past the 4h band): the design's lagged fixed point (§5 risk 1). They keep last frame's extended value.
   - This replaces the hierarchy pyramid. Its restrict, sweep and prolong (`webgpu-uniform-velocity-extrapolation*`) cost about 0.37 ms of round 12's excess.
4. **Paired FLIP delta:**
   - The old field runs the same kernel on `f.departure` into an old-extended texture.
   - `recordDelta` (`uniform-narrow-band-flip.ts`) reads the two textures instead of the companion slot arrays.
   - That file carries a peer's hunks, so the change lands as a patch through safe-commit.
5. **COLD regions:** a brick whose 16h region is COLD exits at once (the heat tiers). The held, regional and replay machinery has no counterpart. A fused pass is cheap enough to recompute.
6. **Deleted in the same commit:**
   - `uniform-mixed-extension.ts`, `-pair.wgsl.ts`, `-slots.wgsl.ts` and `uniform-compiled-extension.wgsl.ts`;
   - the hierarchy, where it has no other user;
   - the frame's retainPair/preparePair/finishPair/finishSnapshot branches and the reusableExtension/bodyExtension bookkeeping.

## B4 inventory (what the one commit must cover)

These are the old extension's semantics, which the fused passes must keep. They are read from `uniform-mixed-extension.ts`.
- **Sweep gate:** a face sweeps only where its owner tile's support has bit 1 (`umTileSupport(tile) & 2`) and it is not a closed wall (`umClosedWall`). Seed distance 0 is kept.
- **Seeds:** `umSource(face, owner)`, the phase > 0.5 of the owner or its across neighbour. A low face (`umFaceLow`) reads the negative buffer at `umLowFaceIndex`.
- **Cut-off:** root > 2·width·min(h) keeps the old value. Unreached faces take `umFarValue`, the trilinear value of known 4h cells from the pyramid (`webgpu-uniform-velocity-extrapolation*`). Under the B4 plan these hold last frame's value instead.
- **Outputs:** `velocityScratch`, which is canonical (h faces, plus 4h at the base-block texels), and `negativeScratch`. velocityScratch has readers in `webgpu-uniform-reference.ts`, `uniform-bulk-page-transfer.ts`, `uniform-cold-freeze.ts`, `uniform-detail-fields.ts`, the frame and NB-FLIP.
- **Modes the frame drives:**
  - plain `encode`, which is full, or regional when `same` holds (bodies only);
  - the census reuse (`reusableExtension`, `bodyExtension`);
  - the idle skip (`extensionIdle`);
  - the paired sequence `retainPair → preparePair → recordDelta → finishPair`, or `finishSnapshot → snapshot → finishPair`, chosen by NB-FLIP's flipDelta: "swept" is the default and "extended" the option;
  - `regularBulk` (direct restriction);
  - COLD freeze (`capacity.coldRegions`).
- **Other importers to move or delete:**
  - `uniform-mixed-momentum-cache.ts` imports `uniformMixedExtensionSnapshotWGSL` from `-slots.wgsl.ts`;
  - `uniform-velocity-extrapolation-twin.ts` documents umFarValue;
  - `tools/probe-uniform-extension-pyramid.ts` and `tools/probe-paired-extension.ts`;
  - the pyramid (`prepareMixedContinuation`) is also referenced by `webgpu-uniform-reference.ts` and `webgpu-uniform-pressure-multigrid.ts`, so it is deleted only after those are checked.
- **Stage scratch:** the extension's slot arrays are the stage scratch's prefix (`useSlots`, the scratch sizing in the frame at about line 696), and NB-FLIP's particle banks are sized by `extensionCompanionBytes`. Both size rules change with B4.

## Milestones

1. ubaStage. Generator, `baseExpression` made public, GPU regression.
   - No frame path changes, so the scene gates are the baseline rows above.
   - `npm run check:types` passes. `npm run test:unit` reports 1 failure, `reservoir-particle-cost.test.ts`, "merge cap protects neither the outer layer nor mass-saturated deep runs".
   - That test passes at clean HEAD. It fails only with the peer's uncommitted NB-FLIP merge hunks in `uniform-narrow-band-flip.ts`, so it is not this lane's.
   - Commit f1da3003.
2. ubaExtend, the fused extension core, with its regression. The stage window sharing landed with it.
   - Nothing in the frame uses it yet, so the scene gates are the baseline rows.
   - Commit d78a265f.
3. The frame brick store (`UniformFrameBricks`) with its GPU regression, which passes. Nothing in the frame constructs it yet, so the scene gates are the baseline rows.
   - `npm run check:types` passes. `npm run test:unit` fails only the same peer test as milestone 1.

## Frame brick store interface (for Lane T)

`lib/methods/uniform/uniform-frame-bricks.ts`, `new UniformFrameBricks(device, ownership, {capacity?, channels?})`, then `await initialize()`.
- **What it is:** a `UniformFineBrickStore` on the frame's ownership. Its arena is the step's level-0 slot space and records. It owns the table buffer (`table`), and no other buffer until a channel is given.
- **Marks:** a brick (8³ h cells, id 64·block + local) is marked `UNIFORM_ARENA_CLASS.fine` where it holds an h tile of the ownership's h list (`umTopology[UM_TILES + j]`, j < `umCounts.x`). Fine is not a core bit, so there is no dilation. A neighbour brick that is not live stages through ubaStage's seam rule.
- **Persistence:** a brick keeps its slot, record and channel values while it stays marked. Only enter and leave churn runs: `ubaUpdate`, `ubaLists`, `ubaLevel1`, `ubaRecord`, then `ubaRelease`.
- **Record word 28:** the brick's h-owned tiles (`ufbOwnedTiles(id)`, bit x|y<<1|z<<2 of its 2³ tiles). A 4h tile inside a live brick is a seam. Word 29 is the brick id.
- **Capacity:** authored, never grown. An h tile lies in exactly one brick, so the ownership's h tile ceiling (`capacity.fineTiles`, the scene's authored peak) bounds the live bricks. That ceiling is the default (`uniformFrameBrickCapacity`: at least 256, at most the lattice's bricks). An empty free stack sets the fatal word (`fatalOffset`). The frame's diagnostics must throw on it.
- **Per step:**
  - `encodeUpdate(encoder)` runs after the frame's topology is final: the step clear, the mark, then the table entries.
  - `encodeRelease(encoder)` runs after the last reader of leaving bricks (a churn scatter, if any).
- **Fused passes:**
  - Bind `table` as `array<atomic<u32>>` and include `tableWGSL(name)` after the ownership topology (`uniformMixedTopologyWGSL(capacity, group)`) and `uniformStageListsWGSL`.
  - Then call `ubaStageWGSL(store.arena, K, {level: 0, ...})` or `ubaExtendWGSL`.
  - Live bricks: `UBA_LIVE`, with count `UBA_LIVE_COUNT`. Held pages: the level-2 list.
- **Channels:** they move into the store's pool (the `channels` option, `store.moduleWGSL`) only where a fused pass reads them in slots. Elsewhere a core fn loads the paged texture by lattice coordinate (`textureLoad(velocity, origin + l, 0)` under `uniformDetailModule`). The detail rewrite resolves the page, so a fused pass needs no resolver.
- **Regression:** `tests/uniform-frame-bricks.test.ts`, on a 64³ lattice with two h boxes whose corners split bricks. It checks:
  - live bricks equal the h tiles' bricks, with their tile masks and keys;
  - a second frame has no churn and the same slots;
  - removing a box leaves exactly its bricks, while the others keep their slots;
  - restoring the box reuses the released slots;
  - the fatal word stays 0.
- **Not yet in the frame:** the store is constructed by its first consumer (B4's extension), so no step runs a table nobody reads. Lane T's B2 adopts the same instance, through a frame accessor added with B4.

## Status

- The frame brick store landed (above). B4 was built and measured, but did not land (below).
- The frame runs the old extension, so its share of the excess is unchanged.

## B4 build (not landed; saved as a patch)

`UniformFrameExtension` (`uniform-frame-extension.ts`), built on the frame brick store.
- **ufx4** (strided over held pages, the arena's level 2):
  - each 4h face is its canonical texel, or the liquid-subface mean of its 16 h subfaces;
  - it is a source where any subface is;
  - two Jacobi sweeps at 4h, reach 2;
  - each face's (value, distance) goes to a per-block buffer, then 4h tiles publish their texels.
- **ufxH** (strided over live bricks):
  - two sweeps at h, with seams prolonged from the 4h state;
  - a face past the h band takes the 4h value where that sweep reached it, else keeps its own.
- **Rewire:** the frame calls one `encodeExtension` for both the current field and the departure field (the paired FLIP delta), with NB-FLIP's flipDelta "swept" mode removed. In the same patch:
  - the slot extension, the pair/slots/compiled WGSL, the hierarchy pyramid and its CPU twin, and three probes are deleted;
  - in total, 24 files change, +442 −3977.
- **Regression:** `tests/uniform-frame-extension.test.ts`, on a dam break with an h box across the surface. It checks:
  - sources exact;
  - no stale face against a departure poison;
  - side walls kept;
  - both levels reach the band;
  - every extended face within its axis's source [min, max], the maximum principle of fast sweeping (Zhao 2005; Adalsteinsson & Sethian 1999).
  - Result: 13232 sources, 22368 h and 268 4h faces extended, 0 stale, 0 outside the range.
- `npm run check:types` reports 0 errors, and `npm run test:unit` 0 failures (2133 tests) on the B4 tree.

**Patch:** `~/code/me/fluid-scratch/lane-e/b4-final/b4-extension-rewire.patch` (against b759020c).
- It covers the whole change. It applies with `git apply`; the NB-FLIP hunks sit beside the peer's NB_CELL_MASS hunks without overlapping.
- Split patches for the two shared files are in `~/code/me/fluid-scratch/lane-e/patches/`. The new files are also copied in `b4-final/`.

**Cross-scene gates** (B4 tree c3 = 8822b2d8 + B4, one run each, all rc 0), against Lane T's latest at 8822b2d8:

| scene | B4 | before | |
|---|---|---|---|
| seep 300 | 43.7 ms, drift +5e-4 | 31.0 ms, +4e-4 | **+12.7 ms** |
| letters 120 | 75.8 ms, drift −0.0082 | 54.0 ms, −0.0090 | **+21.8 ms** |
| rest 16x8x4 | 3.66 ms | 3.28 ms | +0.4 |
| drop=2 16x8x4 | 24.78 ms | 15.69 ms | **+9.1 ms** |
| rest 8x8x4 | 2.92 ms | 2.94 (334745cb) | 0 |
| drop=2 8x8x4 | 16.28 ms | 13.58 (334745cb) | +2.7 |

Drift is in band. The fixed reach (2 sweeps at h plus 2 at 4h, unreached faces keeping their value) replaced the pyramid's far field without moving seep or letters drift.

**Non-pressure excess at drop=2** (8x8x4, 32 steps, `--kernels=29`, `excess.py`): **11.58 ms** with B4, against 8.71 on Lane T's s23 tree with the old extension.

| stage | old | B4 |
|---|---|---|
| extension (mixed extension + regions + retain) | 1.11 | — |
| ufx4 | — | **3.87** |
| ufxH | — | 0.62 |
| frame bricks update | — | 0.13 |
| NB-FLIP snapshot | 0.05 | 0.26 |

**Cold compile:** profile-solver-construction with a fresh shader cache, stopped at the wrap-up with 3 pipelines still pending.
- ufxH took 1.35 s cold.
- ufx4 and the frame bricks pipelines are not measured: they are not in the slowest-12 list, which starts at 1.35 s, but may have been among the 3 pending.
- The slowest pipelines in the frame were `Uniform mixed surface / advectDeferred`, at about 26 s each (Lane C's).

## LEARNINGS

**Landed (Lane E):**
- ubaStage, f1da3003;
- ubaExtend, d78a265f;
- the frame brick store, 26c969e4;
- the B4 inventory, 1a7daa91.

None of them changes the step, so the drop=2 and seep numbers are the baseline: seep 31.0 ms; drop=2 16x8x4 15.69 ms; non-pressure excess 8.71 ms.

**Did not land: B4.**
- **Step-time regression.** Mechanism measured:
  - ufx4's publish loop visits all 512·64 = 32768 h texels of every held page, and runs a coarse index lookup and a detail-rewritten load and store on each, to publish only the 4h tiles' 3 canonical faces a 4h cell (1536 a page).
  - It is a capacity-sized loop, 21x the useful work, and it costs 3.87 ms at drop=2.
  - It scales with held pages, which is consistent with the much larger seep and letters regressions (seep holds 1792 resident pages). Per-kernel times on seep and letters were not measured.
  - ufxH (0.62 ms) plus the bricks update (0.13 ms) alone is already below the old extension (1.11 ms).
- **The snapshot rose by 0.21 ms.** Unknown: it is likely that NB-FLIP's snapshot now reads the departure extension's output in a different pattern, but this was not attributed.
- **Two correctness bugs, found and fixed on the way:**
  1. *Lattice-coordinate accesses need the detail rewrite.* The rewrite runs only for the device object a `UniformDetailStorage` registered, which is the solver's initialization-wrapped device. A module built with the raw device compiles raw accesses that read zeros and drop every store. `initialize()` now throws in that case.
  2. *The maximum principle had two breaks* (616 faces, worst 0.22 of the span):
     - the 4h restriction averaged air subfaces, so it now takes the liquid-subface mean;
     - the 4h prolongation weighted unreached faces, so it now weights only reached ones.

**Share of the drop=2 non-pressure excess in Lane E's scope** (Lane T's attribution at 8822b2d8):
- extension 0.97 ms (B4);
- census 0.72 (T);
- remap 0.37 (T);
- NB transfer 0.36 (T);
- plan tail, certificate, heat and publish about 0.80 (B4, a regex group at 334745cb).

That is about 3.2 of the 8.7 ms.

**What remains:**
- B4's ufx4 publish fix, then landing the patch;
- the census as a reduction;
- T only on change frames;
- the plan tail, heat and publish fusion;
- one bricks update a step (it now runs 3 to 4 a step);
- the cold compile of ufx4 and the frame bricks pipelines.

**Recommendations, ranked by expected gain at drop=2 (estimates, not measured):**
1. **Land B4 with ufx4 publishing only its 1536 canonical texels a page, from one workgroup a page.**
   - ufx4 is expected to fall from 3.87 to about 0.2 ms, which puts the extension at about 0.95 against 1.11 today.
   - Also make the departure pass share the current pass's staging, or drop the snapshot's extra read: about −0.2 more.
   - Net: −0.2 to −0.4 ms, plus 3977 lines and the pyramid deleted. Re-gate seep and letters first, since their regression should vanish with the same fix.
2. **The census as a reduction over the brick store's enter/leave lists.** classifyCoarse alone costs 0.24 ms per step on every coarse tile. Expected −0.4 to −0.6 ms of the 0.72.
3. **Fuse the plan tail, certificate, heat and publish into one pass over the live list.** Expected −0.3 to −0.5 ms of the 0.80.
4. **T (remap and NB transfer) only on change frames.** Expected about −0.2 ms at drop=2, where most frames change topology, and up to −0.7 ms on settled scenes.
5. **One `encodeUpdate` of the frame bricks a step,** once the head topology is final. Expected −0.08 ms.

## Phase 2 (2026-11-08): B4 re-landed with the publish fix

The B4 patch is re-applied to the main tree, with the following changes. Nothing has been GPU-timed yet: the combined gate round will measure it.

**C. ufx4 (`uniform-frame-extension.ts`):**
- **Publication:** each page's 512 cells are visited once, not its 32768 texels. A 4h tile writes its three anchors (origin + 3e_a). On an axis whose +neighbour is an h tile, it also writes that plane's 16 subface texels. These are the texels the slot extension wrote (publishList's regular branch and publishSeams' uePublishCoarse).
- **Unsupported pages:** a page none of whose tiles has the extension support bit holds only sources and blocked faces (ufxSeed4). No sweep changes them: a source's root never beats 0, and a blocked face is never swept. Its coarse state is its seeds, written without staging the halo window. The result is identical to the full pass. Deep-liquid and far-air pages take this path; the plan sets support only on the surface band plus one tile.
- **Launch:** one workgroup per held page (the pool's blocks) and one per live brick (the store's slots). A group past the count exits at once, as before. The launches are no longer 64 and 256 groups striding the lists.

**What else scaled with held pages** (by code reading):
- **The publish loop** (fixed above).
- **ufx4 staged a 12³ halo window for every held page,** including deep-bulk ones, with about 7 loads per face per axis. The skip above removes it where no tile is supported.
- **The frame runs the extension up to three times a step:** at the head when the census extension isn't reusable, on the departure field, and post-pressure. Each run ran the bricks update and release too; that is now once a step.
- **The coarse and result buffers** are pool- and slot-sized (authored), not domain-sized.

**The NB-FLIP snapshot's +0.21 ms** (attributed from the c3 kdrop run's kernels):
- **The cost:** B4 removed the paired extension's companion slots, so the old field is no longer read from slots (HEAD's `finishSnapshot` then `snapshot(slots)`, or s23's `recordDelta` at 0.046 ms). `narrowBandFlip.snapshot(encoder)` now runs the general path:
  - a 4h sampling cache of velocityScratch, `cache`: 0.098 ms;
  - the unit-tap builder: `markUnitHalo` 0.033, `fillUnitHalo` 0.082 and `unitVelocityTaps` 0.016 ms;
  - then the per-sample `sampleVelocity`, `snapshot`: 0.033 ms.
- **The fix, not done:** a brick-native FLIP delta. The departure extension's ufxH would keep each h face's old extended value in its slot (the result buffer is per slot already). The post-pressure ufxH would then write current minus old, and G2P would read that delta through the same taps it samples u with, so no snapshot pass runs. This touches NB-FLIP's G2P, which carries a peer's hunks, so it is left for agreement with Lane T.

**Bricks update once a step (`uniform-mixed-frame.ts`, `uniform-mixed-invalidation.ts`):**
- `encodeBricks` runs `encodeUpdate` then `encodeRelease`, and only when `bricksCurrent` is false.
- `encodeRelayoutHead` clears the flag right after `remap.applyGpu` and updates once.
- A new invalidation cache, `"bricks"`, is reached by solids, scene, layout, capacity and relayout, so a host change of tiles also refreshes the table.
- encodeVelocityExtension no longer updates the table.

**G. Remap and NB transfer only on change frames** (moot, by code reading):
- `UniformMixedRemap.applyGpu` already strides the GPU worklist of the generation's dilated changed tiles (markListed). An unchanged generation lists no tile, and every launch exits. Its drop=2 time is real change work. The host can't know the count, and needs no host gate.
- "NB transfer" is P2G (transferDeposit, transferRefine, transferProperties, transferSeams, transfer). It runs every step because particles move every step. It is B3, Lane T's, not a topology stage.

**E and F, not implemented, with the reasons from the code:**
- **E (census as a reduction):** umClassifyOwner reads three kinds of evidence per owner:
  - 8 phi corners and the volume. That is B2's evidence, from a brick surface pass that does not exist yet (Lane T);
  - the extended face velocities, its 6 faces plus seam parts. B4 could reduce these per tile in ufx4 and ufxH: each tile's +plane (lo, hi) per axis, with the census combining its own and its −neighbour's. But on a regular 4h owner that is 6 of classifyCoarse's roughly 96 loads. The rest are phi corners, `umResolutionError` (8 per error tile) and `umDepthSample` (64 per deep tile), all phi evidence. So about −0.02 ms, not worth coupling the census to the extension;
  - for decide, importance, interfaceBlocks, boundCubes, pageSeed and pageRequest: neighbour tiles' classes. They become brick-local only as staged windows over tile classes in T.
  - classifyCoarse already replays unchanged tiles (`ccValid`, then `umReplayTile`).
- **F (one pass for the plan tail, certificate, heat and publish):**
  - the plan (seed, dilate0-2) gates the extension (support bit 2), and the certificate reads the extension's output. They are a chain with grid-wide data between links, so one pass over the live list cannot hold them;
  - the tail plan is already a no-op unless the ownership or policy changed (`encodeTail`);
  - the certificate's localSpeed (with the fine defect and surface rate) is B1's in the design, Lane T's;
  - the certificate cannot move to the tail either: on a census step it must read the adopted generation (its certify lists are per h tile), which exists only after the head's relayout;
  - the remaining fusible part is region heat and dynamics publish, about 0.15 ms.

## Phase 3 (2026-11-08): the extension's work follows the band

**The combined gate (HEAD + E + T + C):**

| Scene | ufx4 (ms) | ufxH (ms) | B4 total (ms) | Old extension family (ms) |
|---|---|---|---|---|
| drop | 3.54 | 0.64 | 4.18 | 0.97 |
| seep | 10.22 | 4.00 | 14.22 | 1.64 |

The cost scales with the scene, not with activity. The phase-2 publish fix was not the mechanism.

**The mechanism** (by code reading):
- **ufx4 restricted on the fly, per window cell and per channel:**
  - ubaStage called `ufxValue4` and `ufxSeed4` for every one of the 12³ = 1728 window cells, for each of 6 channels (value and seed, 3 axes). That is 10368 calls a page, 81 a lane.
  - On a face that touches an h tile, each call ran 16 subfaces. Each subface tested `ufxSource`: two `ufxPhase`, each a full `ufxTile` (two `umPageWord` lookups and a block word) plus a texture load.
  - So a face cost about 150 loads for its value and 130 for its seed, and the same face was restricted again in each of the 27 windows that contain it.
  - That came to about 1.4 M loads a page, or 11 k serial loads a lane, which fits the measured 39 µs a page at drop.
  - It scales with the h tiles of the held pages. That is the scene's refined volume (seep's 1229 h floor tiles), not the band.
- **ufxH swept every live brick:** staging, 2 sweeps × 3 components, and publication for every brick of the h ownership, deep liquid included. Its seed staging ran `ufxSeed` (about 20 loads) for each of 1728 × 3 window faces. It scales with the h tile count.
- **Rest** (+0.3 ms on rest16, by code reading): the old extension replayed held regions at rest, while B4 recomputed everything on every encode. The bricks update runs once a step on top of that.

**The fix (`uniform-frame-extension.ts`):**
- **ufxF:** one workgroup a held page flags it when any of its 512 tiles has the support bit (one load a tile).
- **ufxR:** a thread a face of each active page (flagged, or next to a flagged page), with launch (pages, 12). It restricts once, into the coarse buffer's seed half. Both tiles are looked up once a face, so a subface costs a velocity load and two phase loads.
- **ufx4:** stages each window face with one load from the seed half.
  - It sweeps only flagged pages.
  - An active page that isn't flagged copies its seeds as its state.
  - Every other page publishes its own texels. There each face is a source or blocked, which no sweep changes.
- **ufxRH:** a thread per tile row (4 cells) of a live brick in an active page. It packs the h seeds, 2 bits a face, after the results (128 words a slot). Each tile is looked up once a row.
- **ufxH:** sweeps only a brick with a supported h tile; the others publish their own. Each brick looks up its 8 tiles once, not 512 cells.
- **Regression:** the extension GPU regression gives the same counts as phase 2: 13214 sources, 22385 h and 268 4h faces extended, 0 of 3275 stale, 0 out of range.
- **Memory:** the result buffer is 6.5 KiB a slot and the coarse buffer 24 KiB a block.

**Letters +0.27 GiB** (B4 alone; +0.32 GiB combined), by code reading:
- **The extension's buffers, about 80 MiB.** The result buffer is sized by the brick capacity, which is 64 × blocks = 12288 slots: fineTiles defaults to the pool's tiles, so the cap is the lattice's brick count. That is against a measured peak of 8020 h tiles.
- **NB-FLIP's own stage unit taps, about 0.2 GiB:** a face atlas plus a 35³ halo a fine slot (`createTaps`). At HEAD it shared `hanging.unitVelocity` through the paired extension (`sharedStageUnit`), which B4 removed. The FLIP-delta proposal removes these taps and the snapshot kernels together.

### B4 round 3: work proportional to the band, and the tail-recorded FLIP delta

The B4-alone gate (seep 33.0 ms and 1.28 GiB against HEAD's 30.5 and 1.17; letters 65.3 and 2.26 against 52.6 and 1.98) put ufxH 2.687 + ufx4 1.442 = 4.59 ms at seep, against the old extension's 1.77, and the snapshot family at 0.655 against 0.197.

**Attribution (code reading):**
- **Both extensions published the whole live volume.** The departure extension (for the snapshot) and the tail extension each swept the band and republished every warm and frozen texel into velocityScratch, so the step paid two volume publications.
- **Frozen (COLD) regions were recomputed.** The old extension replayed them. B4 swept and republished them on every encode, though no stage changes them: the freeze holds their faces in velocity and velocityScratch.
- **Warm non-band volume has to be republished once a step.** Forces, projection and transfer overwrite velocityScratch there. An rgba32float face field has no read-write storage binding, so in-place publication of only the band's faces is not possible. The one remaining volume publication is the tail's.
- **The snapshot** (its 4h cache, the stage unit taps and halo, the kernel) cost 0.46 ms at seep. Its own taps were about 0.2 GiB at letters.

**Changes:**
- **Frozen regions are held** (`attachFreeze`, freeze regions at binding 7):
  - ufxF flags no tile of a frozen region;
  - ufx4 publishes no cell of a frozen octant (and skips a page that is frozen throughout);
  - ufxH neither sweeps nor publishes a brick of a frozen region.
  - The frozen set is the freeze's own (umFrozenRegion), fixed at encodeHead for the frame.
- **The old field** (`UniformFrameExtensionFields.old`) is the departure extension, now written into departure. Momentum, its last reader, has run, and departure rests until the next momentum.
  - It publishes only swept pages and bricks: the supported band.
  - It copies no low faces.
  - It writes no sentinel. The next step's transfer reads departure faces that momentum did not write, and an earlier UFX_NONE version leaked 3e38 into velocity at 1124 faces of the translation scene within two steps.
  - Both extensions keep an unreached face's own value, as the snapshot did.
- **recordDelta** (NB-FLIP, after the tail extension):
  - A workgroup an occupied tile stages its 3³ tiles' flags (live, supported outside a frozen region, h) and the 540 MAC stencil faces as tail minus old.
  - A 4h face is the canonical texel or, against an h tile, the subface texel.
  - Samples interpolate the known faces, renormalised. COLD tiles, slab seeds and spill take NB_DELTA_NONE on every axis.
  - G2P takes `v + d` per known axis, else PIC.
  - displayMotion captures display energy before before.xyz becomes the delta.
  - The census's travel inputs (transferRefine) still read the move's accepted velocity, because G2P rewrites before.xyz before the transfer.
- **Deleted:** the snapshot kernel, nbSampleCold, the snapshot batch, cache group and stage group, the stage unit taps (`createTaps`, about 0.2 GiB at letters), and NB_SNAPSHOT_PASS.
- **Result buffer** (from the previous round): 0.5 KiB a slot (h seeds packed at 2 bits), so 6.3 MB at 12288 slots instead of 81.8 MB. The h results stay in threadgroup memory.

**Expected (by code reading, to be gated):**
- **Seep:** one volume publication instead of two, the band sweeps, and no frozen-region work. The snapshot's 0.46 ms becomes one recordDelta pass over the occupied tiles.
- **Letters memory:** minus the stage taps (about 0.2 GiB) and minus about 75 MB of extension result buffer.

**Checks:**
- **Extension GPU regression:**
  - 13214 sources, 22385 h and 268 4h faces extended, 0 out of range, 0 of 22623 distinguishable faces stale.
  - The test now runs the extension twice, over a departure prior and then over a stored-field prior, and counts a face stale where the two outputs differ. The departure is no longer a poison, because it now holds an extension.
- **NB-FLIP Galilean seeds:** passes (0.75 m/s kept).
- **Types:** check:types is clean.
- **test:unit:** 1783 pass. The one failure (`reservoir-particle-cost`, the merge-cap shader string) is the peer's uncommitted NB_CELL_MASS hunk, not lane E's.

### B4 round 4: one workgroup a block, an axis at a time

The round-3 gate (HEAD e357848c + B4: seep 32.9 against 30.8 ms, letters 61.6 against 54.5, drop16 17.9 against 15.63) put the extension at 15.33 ms at letters (ufxH 10.88, ufx4 2.62, recordDelta 0.92) against the old extension's 4.65, 4.72 at seep against 1.77, and 2.23 at drop against 1.11.

**The mechanism, by counters** (gpu-frame-profile, letters frames 113 to 122, split dispatches, before: `fluid-scratch/lane-e/prof-split`):
- **ufxH, 11.22 ms a frame (two calls):**
  - Occupancy is 8.3%, the threadgroup-memory limit: four 12³ windows plus a stash, 31.6 KB per 128-lane workgroup, fit two workgroups a core, 256 of 3072 threads.
  - ALU 21%, F32 0.99%, LLC 10%, read 3.6 GB/s, write 2.9 GB/s, TLB miss 20%.
  - It was integer- and latency-bound: neither bandwidth nor float work. The counters do not report registers or spills.
- **The chain:** each brick ran three components in series, each doing stage, two sweeps and a write, about 18 barriers. Every texel load was a page-table walk.
- **Work at step 112:**
  - 1576 live bricks, all swept (every h tile of letters is in the band);
  - 2.75M core cells and 885k seam cells staged, three prolongs each.
- **First fix, tile octants** (eight 256-lane workgroups a brick, all three axes in an 8³ window):
  - ufxH fell to 8.26 ms a frame and occupancy rose to 18.6% (`prof-after`).
  - It staged 2.3 times the cells and repeated the record and its dependent loads eight times a brick.
  - Its ablation per call: the skeleton about 1.1 ms, staging about 1.5, sweeps about 1.0.

**The design:**
- **One workgroup a block for both passes,** a lane a cell, 512 lanes:
  - an h brick for ufxH, a held 4h page for ufx4;
  - one 12³ window an axis (`ufxBV`/`ufxBD`, 13.8 KB), staged an axis at a time;
  - two Jacobi sweeps, the first over the 10³ inset, the second the block's own cells: still exactly two global sweeps.
- **Loaded once, not per octant or axis:**
  - ufxH stages the 4h state of the brick's 5³ 4h cells per axis once (`ufxP4`), so a seam's prolongation reads threadgroup memory, not the coarse buffer and the page table;
  - core texels and seed words load once and stay in registers across the three axes.
- **The brick header:** ufxRH writes each live slot's header (mode, origin, and the 27-neighbour record with packed h-store texels). ufxH loads it in one round, with no tile-flag or frozen-region walk and no record chase.
- **The tail extension is seeded:**
  - the old field's extension of the same advance wrote the flags, the h seeds and the headers;
  - pressure changes no phase, topology or freeze;
  - so the tail runs only the restriction, the sweeps and publication (`encode(..., seeded)`, from `encodeVelocityExtension`), voided by a table update in between.
  - Advances with bodies or physics plugins seed afresh: the split's authority and the solids can change within the advance. Seeded, the submerged-sphere regression (`uniform-body-contact-refinement`) overflowed its particle pool at frame 66 (52224 records); it passes unseeded and on HEAD.
- **ufx4 publication:** each lane writes its cell's anchors; the page's seam cells are compacted into a list and shared out over the workgroup.
- **Scalar combine:** ufxCombine has no indexed arrays. The sweep helpers are generated (`sweep()`), with the six neighbours as fixed strides.
- **Fixed launch grid:** every launch strides its live count over UNIFORM_LIVE_GROUP_GROUPS (256), not the capacity. 12288 empty 512-lane workgroups had cost 0.42 ms a call at letters and 0.11 at seep.

**Timestamps, per call** (`fluid-scratch/lane-e/r4/probe-ufxh2.ts`, 10 dispatches a pass, 5 interleaved rounds, medians):

| | ufxH | ufx4 | ufxR | ufxRH | ufxF |
|---|---|---|---|---|---|
| letters, step 112 (1628 bricks) | 1.135 | 0.265 | 0.208 | 0.198 | 0.006 |
| seep, step 60 (331 bricks) | 0.250 | 0.124 | 0.091 | 0.055 | 0.006 |

- **Letters ufxH ablation:** with no sweeps 0.70, no seam prolong 0.94, no finish prolong 1.07, no store 1.06, skeleton 0.07.
- **Estimated frame extension** (the old call in full, plus the tail's ufxR + ufx4 + ufxH):
  - letters 3.42 ms + recordDelta 0.93, against the old extension's 4.65 including the snapshot;
  - seep 0.99 ms + recordDelta, against 1.77.
- **Correctness:** bit-identical to the round-3 brick kernels (a copy of them in the scratch probe) on dam-break with an h box, steps 3 and 10, both modes: 0 texels differ.

**After, by counters** (`fluid-scratch/lane-e/prof-r5`, letters frames 113 to 122):

| kernel | ms a frame | occupancy | ALU | F32 | LLC | read GB/s | TLB miss |
|---|---|---|---|---|---|---|---|
| ufxH | 2.357 (11.22 before) | 16.1% | 37.3% | 3.4% | 8.2% | 7.1 | 56.6% |
| ufx4 | 0.501 (3.04) | 18.6% | 24.2% | 0.6% | 15.4% | 16.4 | 30.6% |
| ufxR | 0.508 (0.50) | 22.1% | 33.9% | 0.3% | 37.0% | 65.1 | 38.8% |
| ufxRH | 0.205 (0.35) | 26.1% | 70.8% | | | 8.9 | 20.9% |
| recordDelta | 0.918 (0.93) | 26.2% | 48.6% | 2.6% | 28.1% | 34.3 | 43.5% |

- **Extension total:** 3.57 ms a frame, 4.49 with recordDelta, against 16.04 + 0.93 before and the old extension's 4.65.
- **ufxH occupancy:** 16% is about one 512-lane workgroup a core. Threadgroup memory (about 17 KB) would allow three, so registers are now the likely limit; the counters do not report them.
- **What is left:**
  - ufxH's sweeps (0.43 ms of 1.135 a call) and seam staging (0.2);
  - ufxR's restriction, a read-bound pass at 65 GB/s;
  - recordDelta's 540 rewritten face loads a tile, each a page walk.

**The frame-level questions:**
- **Can the old-field extension reuse the tail's work?** Partly, and that part is done: the tail reuses the old call's flags, seeds and headers.
  - Given the distances, the extension is linear in the values, so ext(C) − ext(O) = ext(C − O). One two-channel tail could replace the old call and write the delta directly.
  - What blocks it: the raw old field does not survive to the tail. Departure holds the pre-force velocity and must still hold it for the next step's transfer, and one dispatch cannot read and write it.
  - So it needs one more face field (the old field kept to the tail), or departure redesigned. That would save the old call's ufxR + ufx4 + ufxH (about 1.6 ms at letters) and fold recordDelta's face subtraction into publication. It is not done here: it is a proposal for the next round.
- **recordDelta (0.92 ms at letters):** it is not a subtraction pass. It runs per occupied tile: 540 faces, each two rewritten loads (page walks), then a per-particle interpolation. Fusing it into ufxH does not fit, because ufxH is per brick and recordDelta is per tile and per particle. The two-channel tail above is the route. It is unchanged: it lives in the peer-shared `uniform-narrow-band-flip.ts`.

**Checks:**
- **GPU regressions:** frame extension, NB-FLIP Galilean seeds, brick stage and the submerged-sphere body regression pass on HEAD + this patch.
- **GPU test files compared with a clean HEAD archive:** these fail on HEAD too and are not this patch's:
  - churning spatula, cloth clamp, pressure-seam-cycle, super-tail-bulk16;
  - body contact's "static and body contact", body query reuse's "face candidate reuse".
  - The DER cloth, grid overlay and merge-cap failures are unrelated.
- **Types:** check:types is clean.
- **test:unit:** 1783 pass. The one failure (`reservoir-particle-cost`, the merge cap) is the peer's NB_CELL_MASS hunk.
- **Deleted:** `uniform-brick-extension.ts` and its brick-stage regression, superseded by the per-block windows.

**The gate** (HEAD e357848c + B4 round 4, coordinator's run):

| job | B4 r4 | HEAD |
|---|---|---|
| seep 300 | 30.2 ms, p95 32.5, drift 0.0004, 1.16 GiB | 30.8, drift 0.0002, 1.17 |
| letters 120 | 54.8 ms, p95 81.2, drift −0.0083, 1.97 GiB | 54.5, p95 81.3, drift −0.0091, 1.98 |
| drop16 | 14.39 ms | 15.63 |
| rest16 | 3.57 ms | 3.53 |

- **Extension kernels, B4 against HEAD's old extension plus its snapshot or delta record:** drop 0.61 against 1.15 ms (means), seep 1.38 against 1.77, letters 4.52 against 4.78 (medians).
- **check:** all pass except the known work-control test.

**Next steps:**
- **The two-channel tail** (see the frame-level questions above). Keep the transferred field to the tail in one more face field, or redesign departure, so that one tail extension carries both the current and old channels and writes FLIP's delta as it publishes. That removes the old call (its ufxR + ufx4 + ufxH, about 1.6 ms at letters) and recordDelta's face subtraction.
- **recordDelta, 0.92 ms a frame at letters:** 540 faces a tile, each two rewritten loads (page walks), then a per-particle interpolation. It is the largest extension-side cost left, and the two-channel tail is the route to removing it.
- **Follow-up, phi resolve at letters: 2.425 against HEAD's 1.966 ms, in rounds 3 and 4 alike.** B4 does not write phi resolve's direct inputs: the phi field, the ownership's seam lists, the narrow band's gate word. Its held launch also reads the region heat tiers: it skips a seam tile whose 27-tile neighbourhood is COLD.
  - The deleted extension honoured heat. It held a COLD region at 16h and restored its held values, and it published the old FLIP payload only within one region of a non-COLD one, so G2P took no FLIP delta at COLD samples.
  - The B4 extension sweeps and publishes every live band brick, and recordDelta records a delta at every one. If those deltas wake particles or raise heat in regions that were COLD, phi resolve holds fewer tiles.
  - This is unconfirmed. To check it, compare at letters step 112, HEAD against B4: the heat diagnostics' COLD/WARM/HOT counts, the phi resolve held-word count, and the per-call phi resolve time.
  - If confirmed, the fix is to make the B4 extension honour COLD the way the deleted one did: skip a COLD brick's sweep and keep its published values (the freeze is already honoured), and limit the old payload to within one region of a non-COLD one.

## Phase 4 (2026-11-09): the two-channel tail (B4 round 5)

**Item 1, phi resolve at letters (+0.46 ms): the COLD wake hypothesis is not confirmed.** No code change follows from it.
- **Probe:** letters step 112, HEAD~ e357848c against B4 83e48800, means over the last 12 frames (`fluid-scratch/lane-e/r5/heat-*.log`):

  | | HEAD~ | B4 |
  |---|---|---|
  | COLD | 375 | 440 |
  | WARM | 387 | 371 |
  | HOT | 425 | 516 |
  | free | 365 | 224 |
  | gate openings (of 112) | 74 | 80 |
  | seam tiles, fine / fours | 6899 / 8911 | 6977 / 8878 |
  | phi resolve, held call | 0.472 ms | 0.471 ms |
  | phi resolve, full / scratch | 0.369 / 0.369 | 0.372 / 0.371 |

- **Variance:** an earlier pair of identical runs gave COLD 637 on HEAD~ against 481 on B4. The variance between runs of one tree is larger than the difference between the trees. Neither tree opened the gate in the last 12 frames.
- **Conclusion:** B4 changes neither phi resolve's per-call cost nor its inputs. B4 has more COLD regions, not fewer, so its deltas do not wake COLD regions. The +0.46 ms is run variance in the heat tiers, or an attribution artefact.
- **Kept as before:** the extension does not hold COLD bricks.
- **A COLD region's G2P is unchanged from HEAD.** recordDelta gives a COLD tile's samples NB_DELTA_NONE on every axis, and G2P skips the delta of a COLD sample (nbMoveCold).

**Item 2, the two-channel tail: implemented.**

The old-field call is gone. In its place are two calls. Frame order: momentum, transfer, **keep**, forces, pressure, **pair tail**, recordDelta.
- **Keep** (`mode:"keep"`, the transferred velocity into departure):
  - It runs ufxF, then ufxR in keep mode, which computes only the active bits (one launch, a page group a workgroup). Then ufxRH runs.
  - Then ufx4 and ufxH copy the raw transferred texel at the 4h anchors, the seam subfaces and the owned h cells of active, non-frozen pages and bricks.
  - It has no negative copy and no sweeps. Departure then holds O, the transferred field, and holds it to the tail.
  - With bodies, the kept texels are the keep's own (there is no separate old-field extension).
- **Pair tail** (`mode:"pair"`, a second module):
  - One sweep carries `UfxV = vec2f(C, O)`. C is the current field and O is departure, read through bindings 9 and 10 (`prior`, `priorTexels`).
  - Given fixed distances and seeds, the extension is linear in its values, so both channels share the distances, roots and flags.
  - At a frozen texel, O takes C: frozen faces stay put in both fields, so their delta is 0, as on HEAD. Frozen neighbour bricks are marked in ufxNM bit 31, and the restriction uses `umFrozenRegion(umPageOfTile(t))`.
  - Publication writes ext(C) to velocityScratch, as before, and writes the delta ext(C) − ext(O) to the **delta pool** (binding 11) at every swept face:
    - 4h canonical: `ufxDelta4Index(block, cell, axis)`.
    - h cells: `ufxDeltaRowIndex(rank, l&3, axis)`, where the rank is the h tile's list rank from header words 88+i.
    - Seam subfaces, into the across h tile's low row: `ufxDeltaLowIndex(rank, axis, u, v)`.
- **Delta pool** (`f32`), `deltaWords(blocks, fineTiles)`:

  | region | offset | size |
  |---|---|---|
  | 4h deltas | 0 | fours = 3·512·blocks |
  | prior seeds | fours | fours |
  | prior state | 2·fours | fours |
  | h rows | 3·fours | 240 words × fineTiles (192 +face values, then 48 low subfaces) |

  - Sizes: letters 16.3 MB, seep 8.2 MB.
  - Sizing: the rows are indexed by h list rank and sized by the ownership's h tile ceiling (`capacity.fineTiles`, the authored peak).
  - That needs no counter and cannot overflow. It is smaller than a per-brick pool at the measured peak band (letters 2048 bricks × 7.5 KB ≈ 15.7 MB).
  - No runtime growth.
- **recordDelta** now reads the pool (binding 19, read-only storage) in place of the two face textures (8 and 17). It no longer makes two rewritten face loads (page walks) per face.
  - `ndSlot[27]` holds an h tile's rank, or a 4h tile's block.
  - An h rank at or past `UFX_DELTA_FINE` is fatal 11 ("an h tile's rank lies past the frame extension's N FLIP delta rows").
- **Encode order:** in all modes it is flag, restrict, seed and RH, coarse, fine. Seed and RH moved before coarse; a seeded call skips flag and seed.
  - A keep is never seeded (it throws).
  - With bodies or physics plugins, the tail is unseeded.
  - APIC has no keep and no recordDelta; its tail is the single unseeded extend.
- **Detail rewrite:** `prior` (face) and `priorTexels` (atlas) are added to CLASS_BY_NAME. Bindings 8 and 10 are skipped.
- **Deleted:** `flipOldExtensionGroup`, the old-field extension call, recordDelta's face-texture bindings, and ufxS.

**Item 3, correctness:**
- **Dam-break before/after** (`fluid-scratch/lane-e/r6/probe-delta-eq.ts`, an h box [8,8,8] to [40,40,24], FLIP), comparing every live sample's FLIP delta against HEAD:
  - Step 1: 6049 samples on both trees, at identical positions.
  - No sample's delta is known on one tree and NONE on the other.
  - 48.5% of components are bit-exact. The max |Δ| is 3.0e-8 against a delta scale of 0.163, which is rounding: the pair sweep combines in a different order.
  - Steps 2, 3 and 8: 7513, 14330 and 26097 samples, against HEAD's 7516, 14342 and 26034. This is the small drift expected from rounding-level differences (no bit-parity).
- **GPU regressions on HEAD + patch** (`fluid-gpu-run.sh check`, `fluid-scratch/lane-e/r6/final-*.log`): the frame extension, NB-FLIP Galilean seeds and brick stage (2 of 2) all pass.
- **Body contact:** its "static and body contact" case fails, as it does on HEAD (known).
- **The submerged-sphere case fails on HEAD 83e48800 as well** (`fluid-scratch/lane-e/r7`). This batch does not cause it; round 4 does. It overflows at frame 66: "particle slabs exceed the pool: 52224 records sized of 52224". HEAD~ e357848c passes.
  - **Live samples with a non-binding budget** (`fluid.tiers.bricks` 256), peak at frame 66:

    | tree | peak live | at 70 | at 100 |
    |---|---|---|---|
    | HEAD~ | 46587 | 37892 | 24708 |
    | HEAD | 45234 | 39593 | 25096 |
    | HEAD + patch | 45242 | 39699 | 25273 |

  - So B4 does not make more particles, and the scene's peak demand exceeds its rule-sized pool in every tree. The rule sizes 56320 records, with a 52224-record slab limit.
  - **Reseed deferral** (`reseedClipped`, rule-sized pool):
    - HEAD~ clips 7034 seeds at frame 65 and 8232 at 66, holding at 36260 live, and survives.
    - HEAD clips 4903 at 65, then overflows the slab limit at 66 with 38.3k live.
  - **Cause:** the deferral is judged against the record capacity. The slabs' per-brick slack (a quarter more, at least 32; an eighth more, at least 16, plus refused seeds) can exhaust the slab limit first. B4's particle spread trips that limit where HEAD~'s did not.
  - **Owner:** the seed and slab code, in `uniform-particle-slab.wgsl.ts` and the seed deferral in `uniform-narrow-band-flip.ts`. Two fixes are open:
    - make reseed deferral respect the slab limit;
    - author the scene's particle budget from its measured peak.
  - This lane does not change it.
- **Types:** check:types is clean. The extracted tree (HEAD + patch) is also clean, apart from two app pages that import files git does not track.
- **test:unit on the extracted HEAD + patch tree:** 1784 pass, 0 fail.

**Timing, by counters.** Captures are in `fluid-scratch/lane-e/r7/prof-{letters,seep}-{verify-head,patch-check}`:
- Settings: `profile-mini-dam-xctrace --counters --split-dispatches --frame-labels`, 10 frames after warmup (letters 112 steps, seep 60).
- Kernels are split by their call within each frame (`r7/split.py`, `r7/ctr.py`).
- All times are ms a frame.

| | letters HEAD | letters patch | seep HEAD | seep patch |
|---|---|---|---|---|
| first call (HEAD: old field; patch: keep) | 1.900 | 0.429 | 0.586 | 0.167 |
| tail (HEAD: single; patch: pair) | 1.695 | 2.668 | 0.497 | 0.771 |
| recordDelta | 0.899 | 0.714 | 0.084 | 0.068 |
| **total** | **4.494** | **3.812** (−0.68) | **1.167** | **1.006** (−0.16) |

- **Keep:** it saves what was expected, 1.47 ms at letters.
  - Of its 0.43 ms, 0.21 is ufxRH, the headers the seeded tail reuses (the old call ran it too). The copies are ufx4 0.105 and ufxH 0.091.
- **Pair tail:** +0.97 ms at letters, mostly ufxH (1.888 against 1.175; ufxR +0.13, ufx4 +0.13).
  - ufxH counters, pair against single: occupancy 15.9% against 15.7% (one 512-lane workgroup a core; neither fits two in threadgroup memory); ALU 24.2% against 32.0%.
  - So ALU work grows 22% (1.888 × 24.2% against 1.175 × 32.0%) while time grows 61%.
  - **Attribution:** the pair ufxH is latency-bound. The doubled window staging loads (C and O for each of 1728 window cells, four a lane) and the second channel's 4h state reads (`ufxP4O`, from the pool) are not hidden at one workgroup a core.
- **recordDelta:** −0.19 ms. TLB misses fell from 43.6% to 13.1% and texture sampling to 0, but its cost is the per-sample interpolation (ALU 44%), not the face loads.
- **Pool memory:** letters +16.3 MB, seep +8.2 MB (no runtime growth).
- **Next:**
  - ufxH's occupancy. Single and pair alike run one workgroup a core: the pair uses about 25 KB of threadgroup memory, the single about 19 KB. Halving the workgroup, or the window staging, would let latency hide, and would save more than the second channel costs.
  - recordDelta's per-sample interpolation, now the whole of its cost.


### B4 round 6 (2026-11-09): the pair ufxH's window

Goal: the pair ufxH at no more than the single one's 1.175 ms (letters), so the keep's full saving lands.

**Attribution** (isolated and in-frame A/B of ufxH variants, `fluid-scratch/lane-e/r8/probe-{pairh,inframe,slim}.ts`; the in-frame harness swaps the tail's ufxH for a variant, timestamped in its own pass, a variant a frame after 100 warm frames). In frame, at letters step 112, against the round 5 pair (1.84) and single (1.36):
- the departure's atlas loads (`ufxLoadPrior`, the second texel of each window cell) cost about 0.32 ms; with no window loads at all the pair is 1.37 against single's 1.28, so the doubled sweep is about 0.1;
- the C loads are free in the single but not in the pair: the pair holds a second vec3 a window cell through all three axes, and the registers, not threadgroup memory, are what hide the loads' latency. The occupancy counter reads the same for both (15.9% against 15.7%), and padding the single's threadgroup memory to 9 KB or 14 KB more changes nothing (1.353, 1.349 against 1.33);
- no help: staging only the cells a seam reads (lazy), loading per axis, issuing the loads before the 4h stage, reading the kept value by h tile rank (1.72).

**Changes** (exact; both kept):
- **The kept field by brick slot.** The keep writes each h cell it copies into the departure and also into the pool's tail (`UFX_KEPT`, `ufxKeptIndex(w,l)`, 3 × 512 words a slot). The pair's window reads its second channel there: one buffer load a cell for the departure's atlas texel, still the field's own value in a frozen region. The keep's and the tail's slots are the same, from the same headers. The restriction and ufx4 still read the departure texture. `ufxLoadPrior` and the departure's h-store twin (binding 10, `priorTexels`) are gone.
- **A slimmer window.** A window cell is held as a vec3 texel, with its kind packed into the low 2 bits of its seed word: 9 fewer registers a lane in both modes. This is worth −0.09 ms in the pair and nothing in the single.

**Checks.**
- Dam-break deltas against round 5 (`r6/probe-delta-eq.ts`, `r6/cmp-r5.py`): step 1 has no NONE mismatches, and max |Δ| is 3.0e-8 of 0.163, the same transfer-atomic noise as round 5 against HEAD.
- GPU tests on HEAD 99dd6d64 + patch, all passing: frame extension, Galilean seeds, brick stage (2/2), body contact (2/2, including the submerged sphere after Lane T's slab fix).
- test:unit: 1784 pass, 0 fail. check:types is clean (apart from the two untracked app imports in the extracted tree).

**Counters** (`r7/prof-{letters,seep}-patch-check`; round 5 in `-r5`, the slot buffer alone in `-buf`). All ms a frame:

| | letters HEAD | letters r5 | letters r6 | seep HEAD | seep r5 | seep r6 |
|---|---|---|---|---|---|---|
| first call (HEAD old; keep) | 1.900 | 0.429 | 0.463 | 0.586 | 0.167 | 0.177 |
| tail ufxH | 1.175 | 1.888 | **1.678** | 0.252 | 0.425 | 0.375 |
| tail total | 1.695 | 2.668 | 2.517 | 0.497 | 0.771 | 0.713 |
| recordDelta | 0.899 | 0.714 | 0.709 | 0.084 | 0.068 | 0.068 |
| **total** | **4.494** | 3.812 | **3.689** (−0.81) | **1.167** | 1.006 | **0.958** (−0.21) |

- The pair ufxH is −0.21 at letters (the slot buffer alone −0.10); the keep's ufxH is +0.03 for the extra write.
- The letters tail's ufx4 reads 0.456 against 0.38–0.41 in the other captures. Its code is unchanged, so this is capture noise.
- ufxH counters (letters, pair): ALU 26.9% (from 24.2%), MMU miss 35.9%.

**Memory** (superseded by round 7, which removes the kept field). The kept field is 6 KB a brick slot of the authored capacity. The pool is letters 88.3 MB (12416 slots; round 5 had 16.3 MB) and seep 32.4 MB (4202 slots). There is no runtime growth. Indexed by h tile rank (192 words a tile) instead, it would be about 10 MB at letters, at the cost of one dependent rank load a window cell (in frame, 1.72 against 1.68: within noise of the slot layout). This is the coordinator's call if memory matters here.

**Gap to the target.** The pair ufxH is still about 0.5 ms over the single (1.678 against 1.175). What remains is the second channel's register footprint across the axis loop and the buffer load itself (1.52 with no second-channel load at all). An exact pair cannot drop either. The remaining lever changes semantics and needs the coordinator's decision:
- D = C − O stays exact at owned sources;
- extended h faces take D prolonged from the 4h extension, which already carries both channels;
- this drops the h window's second channel, so ufxH ≈ single.

A 256-lane half-brick workgroup was not built. Occupancy is not the limit (the same counter for single and pair, and padding is flat), and a half-brick's 12 × 12 × 8 window stages 4.5 cells a lane against 3.4.


### B4 round 7 (2026-11-09): the h window sweeps one channel (a deliberate approximation)

The coordinator decided round 6's two open questions.
- **Semantics (approved):** the h window drops its second channel.
- **Memory:** rank-indexed kept field. That field is now gone with the second channel, so the pool is back to round 5's size: letters 16.3 MB, seep 8.2 MB.

**The approximation.** The pair tail's h window (ufxH) sweeps the projected field alone (`Ufh`, `ufxHV`, the `ufxHw` sweep). The FLIP delta it writes to an h tile's row, by face (`ufxFinishH`):
- **The face holds its own value** (a source, a closed wall, a face nothing reached): its own values' difference, C − O. This is exact; O is the departure's own texel, one load an owned cell.
- **The h sweep extended the face:** the 4h band's delta, prolonged as the h seams prolong the 4h state (linear along the axis, constant across it). The 4h window still sweeps both channels, so a uniform delta is exact (a Galilean translation) and a linear one holds to the prolongation's order.
- **The h sweep reached the face and the 4h band did not:** unknown (`UFX_DELTA_NONE`, 3e38). recordDelta drops it from the sample's weights, and G2P takes PIC on an axis with none.
- **The h sweep did not reach the face but the 4h band did:** the 4h band's value and delta, as before.

So FLIP's delta is at the 4h band's resolution, but only in the h band's extrapolated faces. Source faces, and therefore every liquid sample's own faces, stay exact.

**Code.**
- The sweep and the combine are generated by state type: `sweep(name, V, D, w, T, S)` and `combine(T, S)`. These give the 4h `Ufx` (a value pair in a pair) and the h `Ufh` (the field's value).
- `ufxReached` takes a distance.
- The round 6 slot-indexed kept field and its keep write are deleted. `ufxLoadPrior` and binding 10 (`priorTexels`) are back for the one own-cell load.
- The slim window (kind packed into the seed word, vec3 texels) stays.

**Checks.**
- **New test,** `tests/uniform-frame-extension.test.ts` "pair tail: an h face's delta is exact at sources, a linear shear's to the 4h order where extended". It runs on the dam break with the h box.
  - A detail-rewritten pass writes the departure as the velocity less D = k (y, z, x): at every face of an h tile, and at a 4h tile's anchors and its subfaces against an h tile. The test checks that every h face holds the shear.
  - All 12608 source h faces' deltas are exact.
  - 22368 h faces are extended, 0 of them unknown.
  - The shear's error is at most 11.00 k (bound 12: the 4h band's two 4h cells, 8, plus 1.5 each for the constant prolongation across a 4h face and the liquid restriction's offset). The mean is 2.48 k (bound 4).
- **Galilean seeds:** passes (a uniform delta is exact).
- **Frame extension:** passes. Its h band is unchanged: the same 22368 h and 268 4h faces extended, none stale, none outside the source range.
- **Brick stage and body contact:** 2/2 each.
- **Dam-break sample deltas against round 5, step 1:** no NONE mismatches; max |Δ| is 3.2e-3 of 0.163, mean 5.1e-5. 44.6% of known axes are bit-exact; the rest are the extrapolated faces.
- **test:unit** on HEAD + patch: 1784 pass, 0 fail.
- **check:types:** clean (the extracted tree has only its two untracked app imports).

**Counters** (`r7/prof-{letters,seep}-patch-check`; HEAD 99dd6d64 in `r7/prof-*-verify-head99`). ms a frame:

| | letters HEAD | letters r7 | seep HEAD | seep r7 |
|---|---|---|---|---|
| first call (HEAD old field; keep) | 1.884 | 0.430 | 0.589 | 0.170 |
| tail ufxR / ufx4 / ufxH | 0.255 / 0.243 / 1.180 | 0.360 / 0.417 / **1.536** | 0.106 / 0.133 / 0.250 | 0.153 / 0.186 / 0.324 |
| tail total | 1.678 | 2.313 | 0.489 | 0.663 |
| recordDelta | 0.928 | 0.699 | 0.082 | 0.068 |
| **total** | **4.490** | **3.442 (−1.05)** | **1.160** | **0.901 (−0.26)** |

- **Pair ufxH:** 1.888 (r5), then 1.678 (r6), then 1.536.
  - The in-frame A/B, the same frame state with the single module swapped in (`r8/probe-r7.ts`, 12 rounds), gives single 1.41 against pair 1.57.
  - So on equal work the pair is +0.15 (11%). Of that, about 0.08 is the own-cell kept load, the delta row write and the prolongation at extended faces, and the rest is the 4h state's second channel staged for the seams.
  - The counter capture's larger gap against HEAD also includes the two trees' different band states at step 112.
- **What remains over HEAD's single tail is the 4h pair:** ufxR +0.10 and ufx4 +0.17 at letters, from both channels' restriction, sweep and publication. The next lever, if wanted, is the 4h window's second channel.
- **Declined: cheapening recordDelta** (hoisting the per-axis base and fraction work, plus an all-eight-known fast path): −0.087 ms at letters, under the 0.1 bar. Folding recordDelta into G2P was declined too: the taps differ, and the relayout invalidates the pool.

**Combined gate (coordinator, HEAD 99dd6d64 against HEAD + E + T).**

| | HEAD | HEAD + E + T |
|---|---|---|
| letters 120 step | 54.5 ms (p95 79.3) | 55.3 ms (p95 81.6) |
| letters drift | −0.0079 | −0.0080 |
| letters memory | 1.98 GiB | 1.97 GiB |
| seep 300 step | 29.6 ms | 29.6 ms |
| seep drift | 0.0004 | 0.0003 |
| rest16 | 3.44 | 3.44 |
| drop16 | 15.7 | 14.86 (within noise; the drop kernels are lower) |

- **Checks:** 16/17; the one failure is the known work-control test.
- **Kernel medians as reported** (letters steps 116–119, then seep):
  - letters extension 2.556 and 3.473; letters FLIP delta 0.590 and 0.852; seep extension 0.917 and 1.179.
  - These are listed in the coordinator's column order (HEAD, then HEAD + E + T), which makes the combined tree higher. Lane E's own counters show the reverse: total 4.490 → 3.442 at letters and 1.160 → 0.901 at seep. The columns may be swapped; this is not yet confirmed.
- **NB-FLIP seed +0.39 ms and transfer +0.26 ms at letters (follow-up, by code reading):** neither comes from the keep or from the change to departure.
  - **Order:** the keep runs after `encodeParticleMomentum`, in the slot of HEAD's old-field extension. Within the frame, the transfer reads departure as momentum leaves it (`["transfer",f.departure,…]`), before the keep writes it.
  - **Cross-frame:** momentum rewrites departure ahead of the next transfer. A frozen quiet frame skips momentum, but even then the transfer would only read different values: neither kernel branches on velocity values.
  - **Seed:** the seed (`compiled.seed`, on the advectParticles group) samples no velocity at all. A seed takes the grid's velocity at the next G2P (`nbSeedFirst`).
  - **Pair tail:** its first channel (velocityScratch) is HEAD's tail extension.
  - **Lane E alone:** the E-only captures at the same window (`r7/prof-letters-verify-head99` against `r7/prof-letters-r7`) show no rise. Seed's kernels sum to 1.670 → 1.645 ms and the transfer's to 4.508 → 4.486.
  - **Likely cause, Lane T's uncommitted `particleDepthIn` rewrite** (`uniform-narrow-band-membership.wgsl.ts`):
    - it moves the key selection inside the rolled tap loop;
    - `particleDepth` is called eight times for each seed site (`seedSite`) and at each transfer face (the depth blend);
    - so the rewrite reaches both kernels.

## COORDINATION

- **Frame (`uniform-mixed-frame.ts`), Lane E's hunks:**
  - the B4 rewire: encodeVelocityExtension, encodeExtensionOf, the keep (`flipKeepGroup`: the transferred field into departure) and the pair tail (`flipTailGroup`: velocity and departure, publishing into velocityScratch and the FLIP delta pool), `extension.attachFreeze`, `narrowBandFlip.recordDelta` after the pair tail (bound to `extension.delta`), and the paired-extension removal;
  - `bricksCurrent`, `encodeBricks`, the `"bricks"` reach case, and the update after `remap.applyGpu` in encodeRelayoutHead.
- **Lane T** owns the store's widening (particle bricks, level-2 pages), on hold with D's fused launch over the compile constraint. The particle move's jobs come from the particle slab table, so nothing reads leaving slots and the release stays beside the update in `encodeBricks`. If the widening resumes, Lane T's edits go beside that call as exact-string edits, announced first.
