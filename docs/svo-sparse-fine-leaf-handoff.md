# Sparse fine-leaf bricks for the dry hero garden

Goal: leaf voxels fine enough that the hero garden's silhouette stops reading as
voxels at all — the plate's 29 mm florets, the coping's arris, the pebble grade —
without the allocation growing as `1/h³`.

Water is out of scope here. The hero garden is a **dry** world
(`scene.systems.fluid === false`) and everything below assumes it.

## 0. The wall, measured

Per finest voxel, a dry scene pays:

| item | bytes/voxel | where |
|---|---|---|
| brick payload `sceneGeometry` (solidSignedDistance, solidFraction) | 8 | `sparse-brick-octree.ts:128` |
| brick payload `sceneMaterialOwners` | 4 | `sparse-brick-octree.ts:129` |
| **node-mip opacity page** (4 000 B / 512) | **7.8** | `svo-node-mip-pyramid.ts:3` |
| **node-mip radiance, 4 lobes rgba16f** (32 000 B / 512) | **62.5** | `svo-node-mip-address-plan.ts:39` |
| ancestor levels (×8/7) | ×1.143 | |
| voxeliser candidate arena | 4 | `webgpu-octree-sparse-bricks.ts:304` |

≈ **12 B of brick against ≈ 80 B of pyramid.** The pyramid is the wall, and the
brick payload is not even the thing the renderer reads.

Halving the leaf multiplies every row by 8:

| lattice | occupied bricks | pyramid |
|---|---|---|
| 25 mm | 384 base pages | ~16 MB |
| 6.25 mm (today) | 10 625 | ~437 MB |
| 1.5625 mm | — | infeasible |

Four independent factors sit on top of that 8×:

1. **Apron.** Pages are 10³ physical for an 8³ interior — **1.953×** wasted so a
   linear sampler can read a neighbour.
2. **Fluid lanes on a dry world.** `SVO_NODE_MIP_LANES` reserves `fluidMean` and
   `fluidMaximum` unconditionally; the dry expansion hard-codes
   `fluidFraction = "0."` (`webgpu-svo-live-derived-builder.ts:227`). **Two of
   four RGBA8 bytes are always zero.**
3. **Constant radiance alpha.** `writeRadiance` stores `vec4f(rgb, 1.)` in every
   lobe — **8 of every 32 radiance bytes** are a literal 1.0.
4. **A dense address shape.** `planSvoNodeMipAddresses` prefers "every page the
   domain can hold" whenever it fits. At 25 mm that is 441 pages against 245
   occupied: **196 pages of pure air, ~7 MB.**

And nothing culls interiors. Two producers actively fill them:

- `sparseSceneTerrainBrickCoordinates` (`sparse-scene-terrain-field.ts:152`)
  emits `for (y = 0; y <= top; y++)` — every brick from the domain floor to the
  column height. The ground is a **solid volume**, not a shell.
- `liveSceneBrickCoordinatesForRegions` (`webgpu-octree-sparse-bricks.ts:480`)
  fills each primitive's whole **AABB**, interior and empty corners alike.

## 1. The three facts that make this tractable

**(a) The renderer does not read the per-voxel SDF.** The dry primary binds
exactly two things from the tree: `structural.structure` and
`structural.sceneMaterialOwners` (`webgpu-svo-dry-scene.ts:6828`). It never binds
`sceneGeometry`. Those 8 B/voxel exist only so the *derived builder* can
central-difference a normal and compute coverage.

**(b) The exact SDF of every primitive is already in the shader, indexed by
owner.** `svoPrimitiveWGSL` is included at `webgpu-svo-dry-scene.ts:4504`, and
`:4831` already builds a `DryHit` from an exact intersection with an exact
normal. Meanwhile the voxel DDA path returns `dryVoxelFaceNormal` (`:2403`,
`:4869`) — an **axis-aligned face normal**. That is precisely the stair-step
terracing in `artifacts/fine/cell3.125.png`.

> A voxel hit already carries its owner (`cellIdentity >> 16`). The owner already
> names a primitive whose exact SDF and exact normal are one function call away.
> The per-voxel SDF lane is redundant *and* worse than what it displaced.

**(c) The identity lane is not per-voxel data.** Censused on this scene at
6.25 mm (`sparse-brick-dry-payload.ts:6`): **91.67 % of occupied bricks hold
exactly one owner**, mean 1.092, max 4. A complete, tested, device-free packer
already exists — `lib/sparse-brick-dry-payload.ts` — with `empty` 8 B,
`uniform` 72 B, `palette` 216 B against 2 048 B raw. **It has no GPU
implementation and nothing outside its own test references it.**

## 2. Target state

A dry voxel stores **one bit**. Nothing else.

Per brick, classified into four kinds (three exist; `full` is new):

| kind | when | bytes | pyramid page |
|---|---|---|---|
| `empty` | no occupied voxel | 8 (header) | none |
| **`full`** | **all 512 occupied, one owner — an interior** | **8 (header)** | **none** |
| `uniform` | one owner over a mask | 8 + 64 = 72 | one |
| `palette` | 2–4 owners | 8 + 64 + 16 + 128 = 216 | one |
| `overflow` | >4 owners (unreached on this scene) | 8 + 2 048 | one |

- **No fluid payload.** Already true on `dry` — the lanes are absent, not zeroed
  (`sparse-brick-octree.ts:128`). Hold that line and extend it to the pyramid,
  where the two fluid lanes are still reserved.
- **No material payload.** Material is derivable from owner via
  `svoPrimitiveMaterialId(record)`; the shading is procedural from world position
  (`svo-procedural-material.ts`) and already costs zero per-voxel bytes. Store the
  owner, drop the material half of the packed word.
- **No SDF payload.** Normals come from the owner's exact primitive.

Interior (`full`) bricks are **classified, not deleted** — a buried voxel must
still report full coverage to every cone
(`webgpu-octree-sparse-bricks.ts:1109`). A header-only kind gives the cone its
answer for 8 bytes instead of 6 144.

## 3. Work packages

Order is load-bearing: **SP3 must land before SP2** removes the SDF lane, or
normals regress from smooth to voxel-face.

- **SP1 — brick classification and producer culling.** Add `full` to the
  classifier. Stop `sparseSceneTerrainBrickCoordinates` filling columns and
  `liveSceneBrickCoordinatesForRegions` filling AABB interiors: emit surface
  bricks, and mark the interior span `full` without allocating voxels.
  *No format change, no shader change.* Biggest single allocation win.
- **SP3 — owner→exact normals at voxel hits.** Refine the DDA hit against the
  owner's exact primitive for the normal (and the hit point where it is cheap).
  Kills terracing; unblocks SP2. Independent of SP1.
- **SP2 — GPU dry payload = occupancy bitmask + owner palette.** Wire
  `sparse-brick-dry-payload.ts` to the device; variable-length per-brick
  allocation replacing `voxelOffset = leafIndex * 512`. Drop the `sceneGeometry`
  lane.
- **SP4 — cheaper pages, and switch the elision on.** See §5: air elision is
  already correct and already happening at the base level. The remaining wins are
  per-page bytes (dead radiance alpha, dead fluid lanes, apron) and the *dead*
  `emptySpaceElision` traversal path.
- **SP5 — push the leaf down** and report how fine the scene actually goes.

## 5. Air pages and cone lighting — correcting §0

**Empty-air pages are not needed for accurate cone lighting**, and the code
already relies on that.

`dryNodeMipAt` (`webgpu-svo-dry-scene.ts`) returns
`SvoNodeMipSample(0,0,0,0)` with `valid = 1` when a page is non-resident. Air is
opacity 0 and premultiplied radiance 0, so **an absent page and an all-zero page
are the same answer**. Absence is a defined, correct result, not a hole.

**The level structure is what makes it sound, and it is an invariant, not a
policy.** `planSvoNodeMipPyramid` (`svo-node-mip-pyramid.ts:177-184`) walks every
occupied base page up through every level, halving the coordinate and inserting
an ancestor at each. Residency is therefore **ancestor-closed**: a non-resident
page provably has no resident descendants. A coarse page may only be elided if
its whole subtree is elided — which is automatic when the set is built bottom-up
from occupancy.

That invariant buys more than storage. `dryConeZeroRegionAt`
(`webgpu-svo-dry-scene.ts:1664`) turns a non-resident page into a whole **zero
region** the cone skips in one step, then does a **directory-only coarse upgrade
two levels up** — no texture fetch — to skip a box 4× larger, justified by
exactly that ancestor chain. It is gated behind an `emptySpaceElision` option
that **both construction sites omit** (`:4417`, `:6275`). It is dead code.

**So §0's fourth factor was overstated.** `planSvoNodeMipAddresses` selects the
dense "total" shape only when `pageCapacity >= domainPyramidPageCount` — i.e.
only for domains small enough that it is free. At 6.25 mm it is **already
occupied-only**; the 196 wasted pages are a 25 mm artefact and do not scale up.

The real remaining pyramid wins are per-page, not per-page-count:

| item | share of the pyramid | why it is dead |
|---|---|---|
| radiance alpha | **~22 %** | `writeRadiance` stores `vec4f(rgb, 1.)` — 8 of every 32 B is a literal 1.0, and radiance is 32 000 of the 36 000 B/page |
| `fluidMean` + `fluidMaximum` | half the opacity atlas | dry expansion hard-codes `fluidFraction = "0."` |
| apron | 1.953× | 10³ physical for an 8³ interior |

Plus turning `emptySpaceElision` on, which costs nothing and is a traversal win.

### 5a. All three are taken, and the apron was never load-bearing

- **Radiance alpha** is free the moment the lobes are `rg11b10ufloat`: a
  three-channel format simply discards the fourth `textureStore` operand.
- **The fluid lanes** are gone on a dry world. The opacity page follows the
  payload profile now — `SVO_NODE_MIP_OPACITY_STORAGE`, `rg8unorm` where the
  device has `texture-formats-tier1`, `rgba8unorm` otherwise. Nothing ever read
  them: the dry renderer's only reads of a node-mip sample are `solidMean` and
  `solidMaximum`, and the sole consumer of `.zw` was the parent reduction
  feeding them back into themselves.
- **The apron was a replica, not a neighbour.** `buildPages` derived the value
  it stored from `interiorCoordinate(physical) = clamp(physical,1,8)-1`, so the
  shell held a copy of the page's own edge texel. A hardware linear tap across a
  page boundary was therefore blending a value against a copy of itself, and
  `clamp(t, 0, interiorSize-1)` on a page with no apron returns the identical
  byte. Removing it is **bit-exact, not an approximation**, and there is no
  traversal cost to weigh against it: the read side is still exactly one
  `textureSampleLevel`, with a different clamp constant.

  It also removes 48.8 % of the derived builder's threads (a page dispatch is
  `physicalSize³`, 1 000 → 512), 48.8 % of the publish copy's traffic, and
  48.8 % of the directional voxel-light cache, which is sized from
  `plan.atlas.texels`.

  (`createSvoNodeMipPageWithApron` *can* fill true neighbour data and still
  will if the apron is put back. Nothing on the production path called it —
  only `tools/benchmark-svo-cone-gpu.ts`.)

Together: **4 000 B → 1 024 B per opacity page, 3.906×.** Measured on
`hero-garden-hose` at depth 0, all 21 checks green:

| | before | after |
|---|---|---|
| opacity page | 4 000 B (`rgba8unorm`, 10³) | **1 024 B** (`rg8unorm`, 8³) |
| pyramid, 12 222 pages | 47.3 MB | **12.3 MB** |
| per base page (10 377) | 4 779 B | **1 241 B** |
| settled frame hash | `0xb7c0a17b` | `0xf13eb954` |

The hash moves by **171 pixels of 368 000 (0.046 %), every one a single LSB,
max channel delta 1** — the float-rounding signature of a changed clamp
constant, not a changed image.

Both halves are levered so the arms can be held apart:
`FLUID_SVO_NODE_MIP_APRON=1` restores the 10³ page exactly (same clamp range,
same offsets, same 4 000 B), and `FLUID_SVO_NODE_MIP_NARROW_OPACITY=0` pins the
four-lane format. Together they reproduce the pre-SP12 layout byte for byte —
which is also the one-line revert if anything downstream goes red.

**Not validated on device, and worth knowing before trusting this further:**

- **The wet lane.** `test:webgpu:dam-ui-two-step` was queued and abandoned when
  the GPU lane was needed elsewhere. The narrow *format* cannot reach a wet
  world — `svoNodeMipOpacityFormat` returns the wide page whenever the payload
  profile is not `dry`, and the `full` expansion's store strings are character-
  identical to before — so the only wet exposure is the apron, which the dry
  lane measured at a maximum channel delta of 1. The dam lane's gated metric,
  `represented-volume-drift`, is a solver quantity that never reads this atlas.
  It is still unmeasured.
- **Depth 1 and above.** Only depth 0 was run.
- **Frame cost.** Unmeasurable in this lane today (an identical config measured
  27.45 ms and 16.22 ms with five agents contending), and the paired A/B the
  levers above make possible was never run. There is no reason to expect a cost:
  the read side is the same single `textureSampleLevel` with a different clamp
  constant, and everything else strictly shrinks — the page dispatch is
  `physicalSize³`, so it falls 1 000 → 512 threads.

## 6. The refinement ladder, walked — the real ceiling

Measured 2026-08-05, `FLUID_SVO_DRY_SMOKE_REFINEMENT=0..3` on `hero-garden-hose`.
This supersedes §0's framing of where the wall is.

| depth | leaf | node-mip base pages | growth | result |
|---|---|---|---|---|
| 0 | 6.25 mm | 10 370 | — | all 21 checks pass |
| 1 | 3.125 mm | 77 045 | 7.43× | renders; `terrain-coverage-solid` **fails**, 37 148 `terrain-partial` holes |
| 2 | 1.5625 mm | 591 761 | 7.68× | Dawn refuses a **9.99 GB** staging buffer (max 4 GB) → black frame |
| 3 | 0.78125 mm | — | — | **Node OOM** at a 4 GB V8 heap, before the device |

**Pages grow ~7.5× per level. Surface area grows 4×.** The tree scales with
*volume*. Bytes-per-page work (SP4, SP7) buys roughly **one** level; changing the
exponent is the only thing that buys three. Hence SP9.

### Three things pin detail even when the tree is fine

1. **The terrain bake was stuck at 3.125 mm at every rung. Fixed (SP10).**
   `heroGardenTerrainSample_m` asked for `min(6.25/4, detail/2)` mm and
   `MAX_TERRAIN_GRID_SAMPLES = 262_144` coarsened it back; the cap **already
   bound at the default**, so it could never stop binding. Worst ground error
   against the analytic vessel was 0.00 leaves at 6.25 mm, 0.73 at 3.125, 1.23 at
   1.5625, **2.75 leaves at 0.78 mm**.

   `scene.terrain` now carries a `TerrainProcedural` description — the vessel
   spec, the container and a spacing — and `terrainSampleGrid` derives the
   heightfield at the lattice in use. Spacing is `min(6.25 mm, detail) / 2` at
   every rung, so worst ground error is **0.000 mm / 0.00 leaves everywhere**,
   and the document went from 4.327 MB / 18.8 ms `structuredClone` to 10.2 KB /
   0.06 ms. No WGSL changed: the GPU still bilinearly samples one heightfield.

   > **Depth 1's coverage holes were *not* this.** With the ground provably
   > exact at every rung, `terrain-coverage-solid` at depth 1 went from 37 148
   > `terrain-partial` to 36 822 — a 0.9 % move. See the correction below.
2. **No generator refines below 6.25 mm.** Of eight, only `bonsai` reads
   `detailCellSize_m`, and its floret floor stops binding at exactly 6.25 mm. The
   proxy catalog is **byte-identical** at 6.25 / 3.125 / 1.5625 / 0.78 mm.
   Finer leaves still help — the proxies are exact analytic SDFs, so a finer leaf
   resolves them genuinely better — but no *new* authored detail appears.
3. **The LOD threshold rescales.** 3 px against a 460 px reference becomes ~13
   device px at the app's real target (`lib/svo-screen-space-termination.ts:41-54`),
   so a sub-mm leaf never satisfies the descent predicate. Drag it toward 0, or
   conclude wrongly that refinement did nothing.

### Correction: `terrain-coverage-solid` is broken above depth 0

`source.structural.domain` publishes `cellSize_m = renderCellSize` — the
**refined** cell — alongside `dimensionsCells = sceneDomain.sceneDimensionsCells`,
which is **unrefined** (`webgpu-octree-sparse-bricks.ts:1951`). At refinement
depth N the published domain therefore describes a box `2^N` times too small on
every axis. The same file already knows this and multiplies by `refineScale` for
the inspection lattice at `:1623-1630`, with a comment naming the exact failure
mode; the published `domain` block never got the same treatment.

The smoke lane bakes its own terrain replica from that published domain
(`run-svo-dry-render-smoke.ts:758`), which is why it reports **376x232 columns at
every depth** while `cellSize_mm` halves, and why the ground's height range
collapses (0.143..0.376 m at depth 0, 0.143..0.344 m at depth 1). Cells outside
the truncated footprint fall back to a clamped edge column, so the oracle calls
them buried when the real surface passes through them — and the voxeliser, which
uses the *correct* refined field, wrote them the partial fraction it should have.
That is the whole `terrain-partial` signature: material 2, fraction just under 1
(0.9957, 0.9752), all at scale 1, first example `cell 5,95,250` with `z = 250`
against an `nz` of 232.

So above depth 0 that check measures the oracle, not the tree. SP9's related
caveat: it only walks voxels inside *published* leaves, so it cannot see an
elided interior either — it proves nothing was corrupted, not that coverage is
sound. Fix the published domain before reading any refinement-depth coverage
number, including the ones in the table above.

### Do not raise `MAX_TERRAIN_GRID_SAMPLES`

The baked `number[]` grid is **4.32 MB of the 4.34 MB document** and 18.5 ms of
its 18.6 ms `structuredClone`; at 1.5625 mm it is 17.3 MB / 92 ms, at 0.78 mm
69 MB / 542 ms. At depth 0 the grid is *already* exactly `c/2` and coverage is
perfect, so an unconditional raise is 4× heavier for zero benefit. The structural
answer is to stop carrying a baked grid and evaluate `pondVesselHeightAt`
analytically in WGSL. A `Float32Array` payload will not work: `cloneScene` JSON
round-trips the document.

### `environmentRefinementDepth` is silently two numbers

The tree divides its cell by `2^depth` from `SvoRenderTuning`
(`webgpu-octree-sparse-bricks.ts:1244`); the scenery catalog reads
`voxelDomain.detailCellSize_m` off the document (`voxel-environments.ts:72`).
Nothing kept them in step. `rebuildSceneAtLattice` now sets both together, but the
one-way case — a user moving the Visual panel's slider — still leaves them
disagreeing.

## 4. Validation

Dawn smoke lanes only — no unit tests.

```
npm run check:scenery                        # CPU, seconds
npm run test:webgpu:hero-garden-hose         # the gate, ~5 s
npm run test:webgpu:svo-live-voxelization    # edit/republish path
npm run test:webgpu:hero-garden-hose-x10     # density headroom
FLUID_HERO_FIDELITY_BASELINE= npm run gate:hero-fidelity   # report, don't gate
```

Layout levers, all of which reproduce an earlier arm exactly (§5a, §7):

```
FLUID_SVO_NODE_MIP_APRON=1            # 10^3 pages, byte for byte
FLUID_SVO_NODE_MIP_NARROW_OPACITY=0   # four-lane rgba8unorm page
FLUID_SVO_OPACITY_LEVEL_FLOOR=0       # base level 0 at any refinement depth
FLUID_SVO_OPACITY_LEVEL_FLOOR=1       # base level 1 at the reference leaf
```

`run-webgpu-exclusive.ts` throws `EEXIST` immediately rather than queuing, so a
paired A/B under agent contention needs one lock acquisition around all the
arms: `tmp/sp24/arms.sh` polls for the lock, holds it, and invokes
`tools/run-svo-dry-render-smoke.ts` directly once per arm.

Traps, verified:

- The exclusive lock is a **directory**, `/tmp/fluid-webgpu-exclusive.lock`, and
  a crash leaves it. `run-webgpu-exclusive.ts` has no timeout.
- Two ratchets are at their floor and are meant to only ever go down:
  `SCENE_PER_BRICK_CEILING["hero-garden-hose"] = 67`,
  `SCENE_OVERFLOWED_BRICK_CEILING = 0` (`run-svo-dry-render-smoke.ts:213`).
- The smoke lane currently defaults to a **neutral clay albedo**
  (`run-svo-dry-render-smoke.ts:952`, uncommitted), while
  `docs/hero-fidelity-baseline.json` was blessed against **authored** materials.
  Pass `FLUID_SVO_DRY_SMOKE_ALBEDO=authored` when scoring colour, or re-bless.
- The baseline's `blessed` string names a camera (1.667 m / 0.50 rad) and grade
  the tree no longer has (now 1.40 / 0.40, exposure 0.2200). Re-bless before
  reading any regression from it.

## 7. The depth-3 pyramid, measured — and the opacity level floor

Measured 2026-08-05 with `tmp/sp24/census.ts`, a device-free replica of the
world constructor that plans **through the shipped functions**
(`liveSvoPlanBasePages` → `planSvoNodeMipAddresses` → `planSvoNodeMipPyramid`),
under the browser default `FLUID_SVO_REFINEMENT_MODE=surface`.

### The figure

| depth | leaf | leaves | pyramid pages | growth |
|---|---|---|---|---|
| 0 | 6.25 mm | 10 369 | 12 213 | — |
| 1 | 3.125 mm | 22 834 | 26 928 | 2.20× |
| 2 | 1.5625 mm | 86 006 | 100 888 | 3.75× |
| **3** | **0.78125 mm** | **441 370** | **512 893** | **5.08×** |

At depth 3, in bytes:

| | |
|---|---|
| opacity atlas (`rg8unorm`, 8³, 1 024 B/page) | **506.2 MB** |
| direct page table (dense over the domain page grid) | 90.9 MB |
| sampled directory (32 B/page) | 15.7 MB |
| radiance atlas (SP7's floor) | 0.8 MB |
| **total** | **613.6 MB** |

Two corrections to the figure this work started from. The extrapolation of
"~378 000 pages / ~470 MB" was low: the per-level growth **rises** (2.20×, 3.75×,
5.08×), so projecting depth 3 from two depth-2 points understates it by 36 %.
And the atlas is not the whole cost — the *direct page table* is another 90.9 MB,
because it is dense over the domain's page grid while the pages in it are sparse.

### Where the bytes are: the base level is 81 % of them

Pages by level at depth 3: **415 379** at level 0, 71 929 at level 1, 13 935 at
level 2, 9 882 at 3, then 1 498, 226, 35, 6, 2, 1. The base level is 81.0 % of
the pages and 405.6 MB of the 500.9 MB payload; *every level above it together*
is a fifth of it.

That base level is one texel per finest voxel — a level-0 page is exactly one
finest brick (`brickSize` 8 = `interiorSize` 8). Nothing reads it at that
resolution except a cone whose diameter is one voxel, and refining the leaf makes
those voxels smaller without making the cones any narrower. It is the same
observation SP7 made about radiance, one field over.

### `SVO_OPACITY_LEVEL_FLOOR`

The opacity pyramid now has a base level anchored to a **world size** rather than
to the leaf: the finest opacity texel stays at the reference 6.25 mm however fine
the tree gets. `liveSvoPlanBasePages` raises every seed to that floor, and
`dryNodeMipAt` clamps every sampled level up to it — the same redirect the
radiance floor already performs, published in `nodeMipAtlas.w` and *read off the
plan* (the finest level any page occupies) so the CPU and the shader cannot
disagree.

| depth | before | after | pyramid pages |
|---|---|---|---|
| 0 | floor 0 | floor 0 | 12 213 → 12 213 (identical) |
| 1 | — | floor 1 | 26 928 → **11 866** (−55.9 %) |
| 2 | — | floor 1 | 100 888 → **26 081** (−74.1 %) |
| 3 | — | floor 1 | 512 893 → **97 514** (−81.0 %) |

At depth 3 that is **613.6 MB → 112.3 MB**: opacity atlas 506.2 → 97.1 MB,
direct page table 90.9 → 11.4 MB, directory 15.7 → 3.0 MB. Note that depth 1's
floored pyramid (11 866 pages) is *smaller than depth 0's* — the first refinement
step now costs the pyramid nothing at all.

**No default configuration renders a coarser opacity field than today's.** At
refinement depth N the floored base texel is `6.25 / 2^(N-1)` mm — 6.25 mm at
depth 1, exactly the resolution the blessed depth-0 frame's cones sample; 3.125
mm at depth 2; 1.5625 mm at depth 3. Refining the leaf still refines the cone
field at every rung, one step behind the leaf instead of in lockstep with it.
Depth 0 keeps floor 0 and is unchanged in every byte.

**Why the pages above the floor are unchanged, exactly.** A raised seed lands on
a page the plan's ancestor walk would have inserted anyway, so the plan at every
level at or above the floor is identical — verified page-for-page at depths 1, 2
and 3, and pinned device-free by `tests/svo-node-mip-pyramid.test.ts`.

**Why the values are unchanged, to a rounding.** A level-1 page's eight children
are level-0 pages, and a level-0 page is one finest brick, so it is one leaf.
When the child page is absent the derived worklist already resolves the child
slot to that leaf (`deepestLeaf` of its centre) and the build reads the payload
directly; `childScale` is 1 at level 1, so `childCell` addresses **exactly the
same eight finest cells** the eight level-0 texels held. The only difference is
that the leaf path reads them as floats instead of through an 8-bit round trip,
so a level-1 texel can move by at most one LSB.

**Why it stops at one level.** At level 2 a child level-1 page spans eight
leaves and the record carries one slot for it, so `deepestLeaf` names one of the
eight and `leafLocal` clamps every cell outside it to that leaf's edge voxel. A
deeper floor needs the fallback to resolve a leaf *per sample* — a change to the
build shader, not to a plan. `SVO_OPACITY_LEVEL_FLOOR.maximumLevels` is 1 and an
override past it is clamped rather than silently wrong. Lifting that bound is
worth roughly another 5× (depth 3, floor 2: 25 585 pages) and is the obvious
successor.

**The GPU worklist needed no change.** `emitPage` asks the direct page table for
a slot; a floored plan gives levels below the floor a zero-depth slab, so the
lookup answers `INVALID` and the loop's own coordinate halving carries the seed
to the floor with the right page coordinate.

**Levers, both directions.** `FLUID_SVO_OPACITY_LEVEL_FLOOR=0` pins the shipped
pyramid at any refinement depth; `=1` forces the floor at the reference leaf, so
the fidelity cost can be measured against the blessed depth-0 frame.

### What else shrinks with it

Four allocations are sized *from* the pyramid rather than by it, and all four
follow the floor. At depth 3, in MiB:

| | before | after |
|---|---|---|
| opacity atlas | 506.2 | 97.1 |
| direct page table | 90.9 | 11.4 |
| sampled directory | 15.7 | 3.0 |
| radiance atlas | 0.8 | 0.8 |
| **pyramid residency** | **613.6** | **112.3** |
| derived builder's opacity scratch (`pageCapacityPerLevel`) | 407.0 | 71.6 |
| derived builder's worklist arena (12 words × capacity × levels) | 190.8 | 33.6 |
| radiance scratch | 0.6 | 0.6 |
| directional voxel-light cache (gated; 8 B/base texel) | 2 008.8 | 386.2 |
| **everything the pyramid sizes** | **3 220.8** | **604.3** |

The voxel-light cache is the surprise: it is one `rg32uint` texel per *base*
texel, four times the opacity page it parallels, and it is allocated at the
atlas's whole slot capacity. It is gated behind `voxelLightUserEnabled`, so it
is not always paid — but where it is, it is larger than the pyramid.

### Measured on device

Five arms, one lock acquisition, `tools/run-svo-dry-render-smoke.ts` on
`hero-garden-hose` at 800×460 (`tmp/sp24/arms.sh`):

| arm | pyramid pages | pyramid | settled hash |
|---|---|---|---|
| depth 0, default | 12 213 (base level 0) | 12.3 MB | `0x4318d060` |
| depth 0, `FLOOR=0` | 12 213 | 12.3 MB | `0x4318d060` |
| depth 0, `FLOOR=1` | 1 844 (base level 1) | 2.1 MB | `0x38a7417a` |
| depth 1, `FLOOR=0` | 26 021 (base level 0) | 25.8 MB | `0x853a2bd2` |
| depth 1, default (floor 1) | 11 869 (base level 1) | 11.9 MB | `0x0e5f4963` |

**Depth 0 default against depth 0 `FLOOR=0`: 0 differing pixels of 368 000.**
The shipping configuration is the pyramid that shipped, byte for byte, and the
lever agrees with the default exactly.

Depth 1 is where the floor engages: **26 021 → 11 869 pages, −54.4 %**, and the
lane's check set is unchanged. `terrain-pages-planned` passes in every arm, at
both floors — the oracle now asks at the level the plan stores.

**The fidelity cost is real and is not an LSB.** Forcing `FLOOR=1` at the
reference leaf moves 209 712 of 368 000 pixels (57.0 %), mean channel delta 3.7,
max 87, with 4 420 pixels past delta 16; at depth 1 the floored arm differs from
the unfloored one on 198 811 pixels (54.0 %), mean 4.9, max 122. Coarsening cone
opacity by one level is a visible change to ambient occlusion and contact
shadow — it is *not* being claimed as free. What is claimed is narrower: the
shipping frame is untouched, and at every refinement depth the floored field is
at least as fine as the shipping frame's. Both depth-1 arms are available for
Peter to compare (`sp24-d1-auto.png`, `sp24-d1-floor0.png`).

> Two checks fail in **every** arm, including the unfloored ones:
> `gpu-validation` ("Copy range … does not fit in the payload arena", 59.2 MB
> against a 44.4 MB buffer) and `terrain-coverage-solid` (0 of 4 061 223 buried
> voxels covered, every one `terrain-partial` at fraction 0.0000). Both name the
> **payload arena**, which SP20–SP22 are mid-surgery on; neither moves with the
> opacity floor, and every pyramid check (`derived-lighting`,
> `node-mip-pages-resident`, `terrain-pages-planned`) passes in all five arms.

### Consumers that assumed a level-0 base

Three, all fixed, all identities at floor 0:

- **The directional voxel-light cache** (`dryVoxelLightAddress`,
  `dryVoxelLightPopulateMain`) addressed `dryNodeMipFind(0u, …)` and skipped any
  page with `level != 0`. Under a floor that disables the cache outright — not
  wrong, but every static shadow re-traced per pixel. It now addresses the floor.
- **`terrain-pages-planned`** in the smoke lane asked whether a *level-0* ground
  page was in the plan. Under a floor the answer is no and the check measures the
  oracle, not the tree; the key is now raised to the published floor first.
- **`tools/benchmark-svo-cone-gpu.ts`** declares its own `DryParams` and had no
  `nodeMipAtlas` lane for the marcher to read. It has one now, holding zero.
