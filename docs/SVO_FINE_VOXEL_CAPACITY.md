# How fine the render SVO can go, and what stops it

Date: 2026-08-03 · M1 Max (32 GiB) · Dawn/Metal · `hero-garden-hose`, dry

Question this answers: the hero pond runs a 25 mm lattice, at which every
feature in its reference image — 8 mm cauliflower florets, 15 mm pebbles, 2 mm
grass — is sub-voxel. What would it actually take to reach 5 mm, or 2 mm, or
finer, and what fails first?

The short version, and it is one sentence: **the sparse octree is not sparse.**
Every brick of the container is enumerated, allocated and walked at the finest
level whether or not any geometry is in it, so the render SVO costs h⁻³ where
the scene in it costs h⁻². Everything below is a consequence of that. The
walls arrive in the order the dense cover makes them arrive, and the last of
them — a 4 GiB buffer at 2.5 mm — is reached by a payload arena that is mostly
empty air.

Three subsidiary findings, all measured:

- **The 7.5 mm wall recorded in the hero plan is a solver wall and the dry
  render path never touches it.** 7.5 mm renders, in 16.6 ms.
- **The first render wall is at 5 mm and it is not memory.** It is two textures
  that were one element wide, plus a `Math.max(...array)` that overflowed a call
  frame. All three are lifted here; **3 mm now renders**, and 2.5 mm is the next
  hard stop.
- **The lattice is already costing the scene objects.** At the shipped 25 mm,
  three bricks of the hero's pebble bands are over the voxelizer's
  64-primitives-per-brick limit, and the surplus is silently absent from the
  lighting hierarchy. One halving clears it.

---

## 1. What was measured, and how

`tools/preview/hero-cell-size.ts` renders the hero document at whatever
`FLUID_PREVIEW_CELL_MM` names — through the catalog, so the environment is the
art-directed one, and touching nothing but `voxelDomain`:

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/hero-cell-size.ts \
FLUID_PREVIEW_CELL_MM=5 FLUID_SVO_DRY_FRAME_WIDTH=800 FLUID_SVO_DRY_FRAME_HEIGHT=460 \
FLUID_SVO_DRY_FRAME_WARMUPS=1 FLUID_SVO_DRY_FRAME_CYCLES=1 \
FLUID_SVO_DRY_FRAME_OUT=$OUT/frame.json \
node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

`tools/svo-fine-voxel-capacity.ts` computes what a lattice will allocate before
a GPU is asked for it, by calling the same three functions the production path
calls in the same order — `createTallCellLayout`, `planSparseSceneDomain`,
`planAdaptiveSparseBrickOctree` — and comparing the result against this
adapter's limits. It exists because a 2 mm bring-up is five minutes of CPU
before the first buffer is created, and it is worth knowing in advance which
sizes are worth spending that on. Its byte model mirrors `SparseBrickOctreeGPU`
field by field; where the GPU has since reported a number, the two agree to the
byte (the 2.5 mm payload arena: model 4 552.68 MiB, Dawn 4 773 830 656 B).

`FLUID_SVO_CAPACITY_SURFACE_ONLY=1` on the same tool skips the layout walk and
reports the geometry-driven brick count and the per-brick primitive census
instead — seconds rather than minutes, at the cost of deriving the lattice
dimensions rather than planning them. The derivation is asserted against
`createTallCellLayout` on every row that does plan, so the two cannot quietly
disagree.

Device limits, from `adapter.limits` on this machine:
`maxBufferSize` and `maxStorageBufferBindingSize` **4 294 967 295**,
`maxTextureDimension2D` **16 384**, `maxTextureDimension3D` **2 048**,
`maxComputeWorkgroupsPerDimension` **65 535**.

---

## 2. The table

Structural counts and allocation are from the CPU model; `allocated` and
`frame` are what the GPU reported; the `outcome` column is the state after the
lifts in §3, and names the walls that had to be lifted to reach it. `leaves` and `voxels` are capacities — the
plan plus the fixed 4 096-brick in-place mutation reserve — because that is what
is allocated. All rows are 1.8 × 0.6 × 1.2 m with 8-cell bricks.

Each row's structural and GPU columns were captured in the same run, so a row is
internally consistent. The rows are not consistent *with each other* to the leaf:
the hero document's scenery grew by several hundred primitives while this sweep
ran, which moves the environment bricks and therefore the counts. Re-running the
25 mm row now gives 190 leaves against the 172 it had then. The walls and their
order do not move; the third significant figure does.

| cell | cells | bricks | leaves | voxels | payload arena | node-mip pages | GPU allocated | frame | bring-up | outcome |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 25 mm | 72×24×48 | 10×4×6 | 412 | 210 944 | 11.3 MiB | 211 | 0.028 GiB | 6.9 ms | 3.5 s | renders |
| 12.5 mm | 144×48×96 | 19×7×12 | 2 916 | 1 492 992 | 79.7 MiB | 1 531 | 0.206 GiB | 10.9 ms † | 4.3 s | renders |
| 7.5 mm | 240×80×160 | 32×11×20 | 10 147 | 5 195 264 | 277 MiB | 6 966 | 0.835 GiB | 16.6 ms | 10.5 s | renders |
| 5 mm | 360×120×240 | 47×16×30 | 24 434 | 12 510 208 | 668 MiB | 23 545 | 2.501 GiB | 23.6 ms | 25.5 s | renders after walls 1–2 |
| 4 mm | 450×150×300 | 59×20×38 | 45 339 | 23 213 568 | 1 240 MiB | 47 656 | 4.958 GiB | 34.0 ms | 46.7 s | renders after walls 1–2 |
| 3 mm | 600×200×400 | 78×26×50 | 98 096 | 50 225 152 | 2 682 MiB | 108 421 | 12.234 GiB | 39.5 ms | 125.7 s | renders after walls 1–3 |
| 2.5 mm | 720×240×480 | 94×32×60 | 166 498 | 85 246 976 | **4 553 MiB** | 185 943 | — | — | — | **fails — wall 4** |
| 2 mm | 900×300×600 | 117×39×75 | 326 522 | 167 179 264 | **8 928 MiB** | 370 069 | — | — | — | **fails — wall 4** |

† The 12.5 mm frame time predates two shader edits landed by other agents during
this investigation; its structural columns are unaffected. Cross-row timing
comparisons in this table are indicative, not a controlled series.

Two costs the `GPU allocated` column does *not* include, because
`OctreeSparseBrickWorld` does not account for them:

- **The CPU staging buffers** the initial plan is uploaded through — one
  geometry, one velocity, one material-owner array at 36 B per planned voxel.
  106 MiB at 7.5 mm, 1 652 MiB at 3 mm.
- **`WebGPULiveSvoScene`'s dense fluid-field 3D textures** — an `r32float`
  level set and an `rgba32float` velocity over the whole lattice, 20 B per cell,
  allocated for a scene with no solver. 59 MiB at 7.5 mm, 916 MiB at 3 mm,
  3 090 MiB at 2 mm.

And one that is not memory at all: `createTallCellLayout` walks every cell
building an initial liquid field the dry scene never reads. That is 0.19 s at
25 mm, 5.9 s at 7.5 mm, 93 s at 3 mm and **311 s at 2 mm** — most of the
bring-up column above, and by 3 mm it is the dominant cost of opening the scene.

---

## 3. The walls, in the order they are hit

### Wall 0 — 7.5 mm, and it is not on this path

`docs/HERO_GARDEN_HOSE_SCENE_PLAN.md` §6b records 7.5 mm as overrunning a device
limit: "Seed global fine bricks from every interface leaf dispatches one
workgroup per interface leaf on one axis: 65 536 against WebGPU's 65 535". The
dispatch is real and the shape of the diagnosis is right —
`lib/webgpu-octree-fine-levelset-topology.ts:725` is a one-axis
`pass.dispatchWorkgroups(workgroups)` and three of the eight passes it drives
are issued at `sortCapacity / 64`, where `sortCapacity` is rounded up to a power
of two (`:593`). That is exactly why the failure reads as 65 536 rather than as
some awkward number: the count steps 32 768 → 65 536 and lands one past the
ceiling on the first step over.

But it is on the **solver** path. `WebGPUFineLevelSetLeafSeeds` is reached only
when `systems.fluid === true`; the hero scene opens dry and the renderer
attaches `WebGPULiveSvoScene`, which owns no transport. **7.5 mm renders**, in
16.6 ms, with a complete node-mip pyramid. It was left unlifted here: the fix
needs a grid-width parameter threaded into five WGSL entry points that index
`scratch[blockBase() + wid.x]`, it belongs to whoever is bringing the solver up
at that lattice, and §6c already records that 7.5 mm now fails the solver
earlier and for a different reason.

### Wall 1 — 5 mm: the node-mip page directory was one page per row · **lifted**

```
lib/webgpu-octree-sparse-bricks.ts:972
  throw new RangeError("Live SVO derived-page capacity cannot cover the declared editable domain");
```

reached because `webGpuSvoNodeMipMaximumPages` returned `maxTextureDimension2D`
— 16 384 — against a plan needing 23 545 pages. The sampled directory was a
two-texel-wide `rgba32uint` texture with one *row* per page, so the 2D height
limit was the entire page ceiling.

> **This paragraph is history, not the current ceiling.** `webGpuSvoNodeMipMaximumPages`
> is now `limit * floor(limit / directoryTexelsPerPage)`
> (`lib/webgpu-svo-node-mip-pyramid.ts:71-75`), so the directory packs many pages
> per row and the smallest acceptable device offers on the order of 10^8 pages
> rather than 16 384. Read as current, the sentence above sends you looking for a
> texture-dimension wall that is no longer there — and `requiredLimits` cannot
> put it back, because dawn-node ignores a request for *less* than the adapter
> advertises. Whatever is limiting a pyramid today, it is not this.

This is not a graceful degradation: a page
the plan cannot address is a page the marcher samples as empty air, so
`OctreeSparseBrickWorld` refuses the whole pyramid rather than draw holes, and
cone lighting falls back to exact traversal for the frame.

The page count tracks the octree's node count almost exactly (6 966 nodes and
6 966 pages at 7.5 mm), which is to say it tracks the dense cover, which is why
it arrives before any memory limit does.

**Lifted** by wrapping pages across columns. Pages stay in slot order along the
row and then wrap, so the CPU-side buffer needs no repacking and the directory
stays sorted by (level, morton) in page-index order — which is what the
marcher's binary search over a level's run depends on. The shader derives the
column count from `textureDimensions(directory).x >> 1`, so a texture and a
shader cannot hold different opinions about the layout, and at one column the
addressing is `(0, pageIndex)` and `(1, pageIndex)` exactly, as before.

- `lib/svo-node-mip-sampling.ts` — `svoNodeMipDirectoryEntry`
- `lib/webgpu-svo-node-mip-pyramid.ts` — `webGpuSvoNodeMipDirectoryShape`,
  `webGpuSvoNodeMipMaximumPages`, both directory allocations and uploads
- Pinned by `tests/svo-node-mip-cpu-oracle.test.ts` ("the sampled directory
  keeps one column until the height limit forces a wrap", and the ceiling is now
  the directory's *area*) and `tests/webgpu-svo-node-mip-pyramid.test.ts` ("a
  directory past the 2D height limit wraps into columns without repacking").

New ceiling: 16 384 × 8 192 = 134 217 728 pages.

### Wall 2 — 5 mm, immediately behind wall 1: the page-validity texture was one texel per page · **lifted**

With wall 1 gone, Dawn refused the next allocation:

```
Texture size ([Extent3D width:23545, height:1, depthOrArrayLayers:1]) exceeded
maximum texture size ([Extent3D width:16384, height:16384, depthOrArrayLayers:256]).
 - While validating [TextureDescriptor "Unified live SVO node mips page validity generations"]
```

`WebGpuLiveSvoDerivedPageState` allocated `[capacity, 1]`, so the 2D *width*
limit was a second page ceiling of exactly the same size, and both the opacity
and the radiance atlases hit it.

**Lifted** the same way: slots wrap onto rows, with
`liveSvoDerivedPageValidityShape` choosing the width and a shared
`svoDerivedPageValidityWGSL` helper doing the addressing at all eight sites that
touch the texture — two reads in `lib/webgpu-svo-dry-scene.ts` and six
reads/stores across the three `lib/webgpu-svo-live-derived-builder.ts` shader
modules. Every one of them derives the width from `textureDimensions`. The
upload moved from contiguous slot runs to whole rows, because a run of slots
stops being a run of texels once the layout wraps; `generations` is row-major so
a row range is still one subarray, and rewriting the untouched slots in those
rows writes back the values they already hold.

- `lib/webgpu-svo-live-derived-cache.ts` — shape, helper, row upload
- Pinned by `tests/webgpu-svo-live-derived-cache.test.ts` ("a page-validity
  capacity past the 2D width limit wraps onto rows"), and exercised on real
  hardware by the two GPU-backed tests in
  `tests/webgpu-svo-live-derived-builder.test.ts` that build and certify pages
  through these shaders.

**5 mm renders after walls 1 and 2**, in 23.6 ms, with all 23 545 pages
resident. 4 mm renders too, in 34.0 ms.

### Wall 3 — 3 mm: an argument list, not a device · **lifted**

```
[svo] live derived lighting unavailable; exact visibility will be used
RangeError: Maximum call stack size exceeded
```

from `createWebGpuSvoNodeMipDirectPageTable` in
`lib/webgpu-svo-node-mip-pyramid.ts`, then line 151:
`Math.max(...plan.pages.map((page) => page.key.level + 1))`. 108 421 pages is
more arguments than a call frame holds. The world catches the throw and reports
it as "derived lighting unavailable", so a CPU limit arrives wearing a device
limit's clothes — worth naming, because the next one of these will look the same.

**Lifted** by folding instead of spreading. Pinned by "the direct page table
survives a page count that overflows an argument list" in
`tests/webgpu-svo-node-mip-pyramid.test.ts`, at 200 000 pages.

**3 mm renders**, in 39.5 ms, allocating 12.2 GiB.

### Wall 4 — 2.5 mm: the payload arena passes `maxBufferSize` · **hard**

```
Buffer size (4773830656) exceeds the max buffer size limit (4294967295).
 - While calling [Device].CreateBuffer([BufferDescriptor "Octree unified live sparse-brick world payload arena"])
```

from the `SparseBrickOctreeGPU` constructor in `lib/sparse-brick-octree.ts`. The
payload arena is one buffer holding five per-voxel lanes — geometry (16 B),
velocity (16 B), material owner (4 B), scene geometry (16 B), scene material
owner (4 B) — so **56 bytes per voxel**, and `maxBufferSize` on this adapter is
4 GiB − 1. That caps the arena at 76.7 M voxels, which the dense cover reaches
between 3 mm (50.2 M) and 2.5 mm (85.2 M).

Note what is *not* the wall: no single binding is near
`maxStorageBufferBindingSize`, because the lanes are bound with explicit offsets
and the largest is 16 B/voxel — 766 MiB at 3 mm. Splitting the arena into five
buffers would move this wall to about 1.7 mm — the largest single lane is
16 B/voxel against the same 4 GiB, so 268 M voxels — and 16 of the 56 bytes are a
velocity lane a dry scene never writes. Both are real levers. Neither is the
right one, for the reason in §5.

### Wall 5 — 0.879 mm: the lattice is silently clamped

`lib/tall-cell-grid.ts:337-339` clamps each axis to `maxTextureDimension3D`:

```ts
const nx = Math.min(maximumTextureDimension, Math.max(8, Math.round(c.width_m / requestedCellSize)));
```

At 2 048 and a 1.8 m container that is 0.879 mm. Below it the scene does not
fail — it quietly becomes a coarser scene than the one that was asked for, with
anisotropic cells, and nothing in the frame says so. That is a worse failure
mode than any of the four above, and it exists only because a dry scene is being
sized by a fluid field's texture.

---

## 4. What a scene costs per object

The lattice is not the only budget a scene library spends. Three numbers bound
how much generated detail a document can hold, and they behave differently.

| budget | value | shape | what happens when it is exceeded |
|---|---:|---|---|
| Live scene primitive arena (`OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY`) | 4 096 | a **total** | `throw new RangeError("Live scene primitive capacity exceeded")` (`lib/webgpu-sparse-scene-proxies.ts:920`) — loud, and the scene does not publish |
| Render candidate BVH (`SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES`) | 4 096 | a **total** | the candidate index is simply not built (`lib/svo-scene-primitives.ts:301`), and every consumer must then fall back or fail |
| Voxelizer candidates per brick (`OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK`) | 64 | a **density** | `CANDIDATE_OVERFLOW` is raised and the surplus primitives are **not written into that brick's voxels** (`lib/webgpu-sparse-scene-proxies.ts:652-656`) |

The third is the one nobody is counting, and it is the one that ties object
count to cell size. A brick is `brickSize × cell` on a side, so the ceiling is 64
primitives per (8h)³ of world:

| cell | brick | primitives per litre |
|---:|---:|---:|
| 25 mm | 200 mm | 8 |
| 12.5 mm | 100 mm | 64 |
| 7.5 mm | 60 mm | 296 |
| 5 mm | 40 mm | 1 000 |
| 3 mm | 24 mm | 4 630 |
| 1 mm | 8 mm | 125 000 |

**The hero scene already exceeds it, today, at 25 mm.** Measured over its 1 004
published primitives by `tools/svo-fine-voxel-capacity.ts`, which counts each
primitive's conservative bounds against the brick lattice:

| cell | brick | busiest brick | bricks over 64 |
|---:|---:|---:|---:|
| 25 mm | 200 mm | **70** | **3** |
| 12.5 mm | 100 mm | 36 | 0 |
| 7.5 mm | 60 mm | 26 | 0 |
| 5 mm | 40 mm | 21 | 0 |
| 3 mm | 24 mm | 19 | 0 |

Three bricks of the pebble bands are over the limit at the shipped lattice, and
whatever primitives lost the race are absent from those bricks' voxels — which
is to say absent from the opacity and radiance hierarchy that shadows, ambient
occlusion and GI read. Nothing reports it. **One halving of the cell size clears
it with room to spare**, and the curve then flattens — 36, 26, 21, 19 over four
more halvings, and not monotone, because from 12.5 mm down the count is set by
how tightly a *species* packs its own instances rather than by the lattice.

That is the shape of the answer for a library of parameterised species. The
ceiling is not "how many objects the scene has" — 1 004 primitives is a quarter
of the 4 096 arena and costs nothing in octree memory, because the dense cover is
the same size for an empty container as for a full one. The ceiling is **how many
objects any one species piles into one brick**, and the lattice sets the brick.

Projecting the budgets reported alongside this work (46 leaves per air plant, 61
per grass tuft, 1 500 for the bonsai): a 1 500-leaf canopy roughly
0.6 × 0.25 × 0.6 m spans about 18 bricks at 25 mm — 83 leaves per brick averaged,
far worse in the middle — and about 108 bricks at 12.5 mm, averaging 14. **The
bonsai canopy cannot be voxelised at 25 mm at all**, and that is a stronger case
for the finer lattice than sub-voxel florets ever were: the failure is not blur,
it is leaves that are silently missing from the lighting hierarchy. This
projection is arithmetic, not a render — the canopy species is still being
built.

---

## 5. The octree is not sparse, and that is the whole story

**Decoupling the render lattice from the solver's is possible and cheap. It is
also not the point.** The two are already partly separated —
`LiveSvoSceneOptions.renderBrickSize` (`lib/webgpu-live-svo-scene.ts:27`) exists
so a renderer-only experiment can change brick size without mutating
`scene.voxelDomain`, and says so in its comment. What is not separable today is
the *cell size*: `createTallCellLayout` reads
`scene.voxelDomain.finestCellSize_m` and `planSparseSceneDomain` derives
`cellSize = containerExtent / dimensions` from it. Adding a `renderCellSize_m`
beside `renderBrickSize` is about five lines, and for a dry scene it is
unconditionally safe — with `systems.fluid === false` there is no solver to
disagree with, and the renderer is the lattice's only consumer.

It buys nothing on its own, because a finer render lattice costs h⁻³. **The
octree must never allocate or walk its entire leaf dimension.** Here is where it
does, all of it CPU work done before the first buffer exists:

| site | what it enumerates | at 2 mm |
|---|---|---:|
| `lib/tall-cell-grid.ts:357` | every cell of the container, building an initial liquid level set a dry scene never reads | 162 M cells, **311 s** |
| `lib/sparse-scene-domain.ts:211` | every brick of the solver range, as an array of coordinate objects | 322 050 objects |
| `lib/adaptive-sparse-brick-plan.ts:118` | a `Set<bigint>` per level over every finest key, then a `Set<string>` of leaf keys, then a `Map` per level | ~2.6 M BigInt shifts, 2.4 s |
| `lib/webgpu-octree-sparse-bricks.ts:721-731` | one interned `"x,y,z"` string per finest brick, to remember what is covered | 322 050 strings |
| `lib/webgpu-octree-sparse-bricks.ts:782-789` | a `Uint32Array` of leaf indices over every local brick, plus a `Map` keyed by string | 322 050 entries |
| `lib/svo-node-mip-pyramid.ts:176` | a `Map<string, key>` over every page at every level | 370 069 entries, 4.3 s |

None of these are the octree being sparse. They are a dense grid wearing an
octree's interface, and the reason is one line: `planSparseSceneDomain` takes
the *solver range* as the candidate set, and `planAdaptiveSparseBrickOctree`
keeps every solver brick at `maximumDepth` by construction — solver bricks are
never coarsened. For a fluid scene that is correct; the solver needs every cell.
For a dry scene there is no solver, and it is pure waste.

How much waste, measured. `tools/svo-fine-voxel-capacity.ts` with
`FLUID_SVO_CAPACITY_SURFACE_ONLY=1` counts the bricks whose extent the sculpted
heightfield or a primitive's conservative bounds actually crosses — an upper
bound on what a geometry-driven octree would hold, since a primitive's bounds
include its interior and a real refinement would descend only to its surface:

| cell | dense bricks | geometry bricks | ratio | voxels | payload arena | fits 4 GiB? |
|---:|---:|---:|---:|---:|---:|---|
| 25 mm | 162 | 108 | 1.5× | 0.06 M | 3 MiB | yes |
| 12.5 mm | 1 296 | 591 | 2.2× | 0.30 M | 16 MiB | yes |
| 7.5 mm | 6 000 | 1 932 | 3.1× | 0.99 M | 53 MiB | yes |
| 5 mm | 20 250 | 4 946 | 4.1× | 2.53 M | 135 MiB | yes |
| 4 mm | 41 154 | 8 507 | 4.8× | 4.36 M | 233 MiB | yes |
| 3 mm | 93 750 | 17 723 | 5.3× | 9.07 M | 485 MiB | yes |
| 2.5 mm | 162 000 | 27 973 | 5.8× | 14.3 M | 765 MiB | yes (dense fails) |
| 2 mm | 322 050 | 48 423 | 6.7× | 24.8 M | 1 324 MiB | yes (dense fails) |
| 1.5 mm | 750 000 | 106 045 | 7.1× | 54.3 M | 2 900 MiB | yes (dense fails) |
| 1 mm | 2 531 250 | 303 674 | 8.3× | 155 M | 8 304 MiB | **no** |

Measured on the hero scene as it stands, with 1 004 published primitives — a set
that grew by 3–5× while this investigation ran, which is exactly the point about
species being instantiated freely, and which is why the ratios here are lower
than an empty container's would be. The dense cover grows as h⁻³ and the
geometry as h⁻², so the ratio grows without bound; at this scene's density the
crossover into "worth several halvings" is around 3 mm.

**A geometry-driven octree moves the reachable lattice from 3 mm to about
1.4 mm on this device** — the payload arena reaches 4 GiB at roughly 55 M
voxels — where the dense one stops at 2.5 mm and spends 12 GiB to get to 3 mm.
Going below that is then the arena split in §8.4, not another architecture: with
the five lanes in separate buffers the 16 B/voxel ceiling is 268 M voxels and
1 mm fits.

What it would take, honestly. The dense cover is not an oversight; it is the
fluid contract leaking into the renderer:

- `lib/sparse-scene-domain.ts:211` enumerates every brick of the solver range as
  a finest-level coordinate, and `planAdaptiveSparseBrickOctree` keeps every one
  of them at `maximumDepth` by construction — solver bricks are never coarsened.
- `lib/webgpu-octree-sparse-bricks.ts:789` then asserts it:
  `throw new Error(\`Fluid brick ${key} has no finest scene leaf\`)`. For a dry
  scene there are no fluid bricks, so the invariant is vacuous and the cover is
  pure waste — but the code has no way to say that today.
- `WebGPULiveSvoScene` allocates the dense `r32float` and `rgba32float` fields
  regardless, and `createTallCellLayout` spends the minutes computing an initial
  liquid field for them.

So a dry-scene geometry-driven path is three changes, none of them deep: a
domain plan that takes candidate bricks from geometry instead of from the
container, a residency object that tolerates a dry world, and a live-scene
constructor that does not build a fluid field for a scene with no fluid. Nothing
downstream needs to know: `planAdaptiveSparseBrickOctree`, the publication
shaders, the node-mip pyramid and the marcher all already work on whatever leaf
set they are handed — they are sparse. Only the *candidate set* is dense.

It is the single highest-value item this investigation found, and it is worth
more than the four walls above put together: those buy 25 mm → 3 mm at 12 GiB;
this buys 3 mm → 1.4 mm at under 4, and takes the bring-up from two minutes to
seconds.

---

## 6. On one pipeline: should the ground become voxels?

Asked alongside this work, and the measurements answer it.

Today the ground is a second pipeline. `traceTerrainHeightfield` marches the
sculpted grid ahead of the octree with a Lipschitz step derived from one
**global** slope bound (`lib/terrain.ts:188`,
`slopeBound = √2 · largestRise / spacing_m`), inside a 64-step budget
(`SVO_TERRAIN_GRID_MARCH_STEPS`). Steepening the pond's inner face took that
bound from 2.85 to 9.54 and mean height evaluations per camera ray from 46.7 to
70.7. The cause is structural: a single global worst case means the flat plaster
metres from the pond is stepped as finely as the pond's wall.

**Finer voxels do make retiring `traceTerrain` viable, but only together with
§5.** With the dense cover, voxelising the ground changes nothing about the
walls — the bricks are already all there, which is another way of saying the
ground is *already* costing what voxelising it would cost, and buying nothing
for it. With a geometry-driven octree the ground *is* the geometry, and the cost
is the column above: 485 MiB at 3 mm, 2.9 GiB at 1.5 mm. What it buys is exactly the property named
in the question: resolving a heightfield inside one owning voxel is a bracketed
root-find in a box of side h — a handful of bisections with a guaranteed bracket
— where the scene-wide march has no bracket at all and must be bounded by a
global Lipschitz constant. The global `slopeBound` stops existing, the 64-step
budget stops existing, and the ground gets per-voxel material ownership like
everything else.

What it costs beyond memory, and this is the part that is not free:
`SparseSceneProxyVoxelizer` has six analytic kinds and explicitly refuses
heightfields — `lib/webgpu-sparse-scene-proxies.ts:100`, "Terrain heightfields
are not finite live primitive updates". A heightfield proxy kind needs
conservative bounds per brick, a WGSL SDF the CPU mirrors, and an owner
assignment rule where the ground meets scenery. By the cone's own footprint in
`lib/svo-primitive-abi.ts` that is roughly a dozen files plus tests.

If instead the analytic trace is kept — which is the right call until §5 lands —
**the obvious next lever is a hierarchical slope bound**, and it is cheap. The
grid already computes one global bound from `largestRise`; a mip pyramid of
per-tile (minimum, maximum, slope) over the same grid costs 4/3 of one extra
value per sample — about 0.3 MB for the hero's 56 k-sample grid — and lets the
march take the *local* bound at the *current* tile. On the hero's own numbers
that is the difference between stepping the flat plaster at slope 9.54 and at
its own near-zero, which is most of the 70.7 evaluations. It also removes the
coupling that makes every future steepening of any one feature a tax on the
whole frame.

---

## 7. On G2 — one owner per voxel

`docs/HERO_GARDEN_HOSE_SCENE_PLAN.md` §3 calls one-owner-per-voxel "the real
limit on fine detail". A blade-width ladder rendered through both the megakernel
and `raster-primary` — which never consults an owner — agreed to 2/255 over
33 000 pixels with identical break-up in identical places. **That result is
consistent with everything found here, and the claim should be retired.**

Two structural facts consistent with it, from this side. The scene's primitives
are published as a BVH over analytic records — `buildSvoPrimitiveCandidates`,
4 096 leaves, built from `SvoFinitePrimitiveDescriptor` bounds and not from
voxels — and the ground is not in the octree at all: the sparse voxelizer has six
analytic kinds and refuses heightfields outright
(`lib/webgpu-sparse-scene-proxies.ts:100`). Whatever the per-voxel owner is doing
for this scene, it is not deciding where the ground is.

Where per-voxel ownership *does* decide the image is the derived hierarchy:
`liveSvoDerivedBuildWGSL` reduces each voxel to a solid *coverage* plus one
*material identity*, and that pair is what shadows, ambient occlusion and GI
sample. Sub-voxel geometry there is averaged rather than lost — it contributes
its coverage fraction and borrows the winning owner's material. That is a
softening, not a disappearance, and it is not what the plan doc was describing.

The per-object limit that *is* a hard drop, and *is* fixed by a finer lattice,
is the 64-candidates-per-brick ceiling in §4. That is where a floret disappears
rather than blurs.

---

## 8. What I would do next

1. **Stop enumerating the container.** (§5) The candidate brick set should come
   from geometry, not from the solver range — three contained changes, worth
   3 mm → 1.4 mm, 12 GiB → under 3, and two minutes of bring-up → seconds.
   Everything else on this list is smaller, and several items stop mattering if
   this lands.
2. **Do not build a fluid field for a scene with no fluid.** `createTallCellLayout`
   is 93 s at 3 mm and 311 s at 2 mm, and every second of it computes an initial
   liquid level set the dry path never reads. A dimensions-only path is small and
   removes most of the bring-up column in §2.
3. **Make the `maxTextureDimension3D` clamp fail rather than lie** (wall 5). A
   scene that silently becomes coarser than it asked for is the one failure mode
   in this document that produces a plausible wrong picture instead of an error.
4. **Split the payload arena** if the dense path must survive a while longer.
   Five buffers instead of one moves wall 4 from 2.5 mm to about 1.7 mm, and
   dropping the velocity lane for dry scenes saves 16 of 56 bytes per voxel.
5. **A hierarchical terrain slope bound** (§6), until the ground becomes voxels.
6. **Count candidates per brick in `check:scenery`.** The 64-per-brick ceiling
   is silent, it is the budget a scene library will actually hit, and it is a
   pure CPU calculation from primitive bounds — the same one in
   `tools/svo-fine-voxel-capacity.ts`.

## 9. What was deliberately not established

- **The solver at any lattice.** Wall 0 was read, not run. Nothing here says
  what the fluid path costs at 7.5 mm or whether it works.
- **What the finer lattice looks like.** No perceptual claim is made about what
  5 mm or 3 mm buys the image. The frames rendered; they were not judged.
- **A controlled before/after frame hash for the lifts.** Two other files
  (`lib/hero-garden-scene.ts` geometry, `lib/webgpu-svo-dry-scene.ts` shading)
  changed under this investigation, so the 25 mm hash moved for reasons that
  were not mine. The no-op claim for the lifts rests on the algebra — one column
  and one row reduce the addressing to the identity — and on the tests that pin
  it, not on a frame comparison.
- **The unit suite's 54 failures.** They were present before these changes and
  are unrelated to them, established by toggling each edit off surgically and
  re-running: the failing set is identical apart from the new tests. Several
  flap between runs because other agents are mid-edit in
  `lib/voxel-scenery/`.
- **How much of the "geometry brick" column is really surface.** It counts every
  brick a primitive's conservative bounds touch, interior included, so it is an
  upper bound: a real refinement would descend only where the surface is, and
  for the ~1 000 ellipsoids in this scene that is a substantial over-count. The
  terrain half of the column is honest. Treat §5's ratios as pessimistic.
- **Anything about a scene that is not this one.** The hero document's primitive
  count grew from ~760 to 1 004 during this investigation, and §5's ratios move
  with it. The h⁻³-against-h⁻² argument does not, but the crossover cell size
  does.
- **A green `minimal-power-dam-break`.** The lane ran, took its full 500 exact
  steps with `validationErrorCount` 0, 0 tripwires and 0 structured rejects — so
  the octree world constructed, published and ran through the node-mip pyramid
  and page-validity textures changed here without a single GPU validation error
  — but it fails one physics check, `hook.minimal-dam-motion.octree.ritter-celerity`
  (8.40 m/s against a 7.25 m/s bound). That is a solver assertion about liquid
  speed and is one of this lane's known reds; nothing here touches it. It was not
  established that it is red for the same reason it was red before.
- **4-cell bricks.** `FLUID_PREVIEW_BRICK_CELLS=4` exists in the preview module
  and was never swept. It trades eight times fewer voxels per leaf against eight
  times more leaves and pages, and on this evidence — where pages, not voxels,
  were the first two walls — it may well be a worse trade, but that is a guess.
