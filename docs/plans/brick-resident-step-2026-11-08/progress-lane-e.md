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
