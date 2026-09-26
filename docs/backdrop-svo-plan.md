# Backdrop: one larger dry SVO world, rolling hills to the horizon

Status: plan, 2026-09-26 (revised the same day: unified world, not a second
world). Owner: garden art direction.

## Idea

The dry scene has **one** uniform representation: a single, much larger sparse
voxel octree that holds the set *and* the landscape around it — rolling hills
with sparse vegetation in the porcelain garden's white-clay idiom.

- **The fluid never enters the landscape.** Its geometry comes from a separate
  document field, `scene.backdrop`, that no solver, solid-world bake, collider
  scan or solver key enumerates. The solver lattice and container are unchanged.
- **Minimal coupling.** The backdrop is only a geometry source for the render
  world. Everything else — traversal, shading, shadows, cones, GI — is the
  world's, so the seam at the slab edge stops existing and the set's shadows
  fall on the hills for free.
- **One pipeline.** The analytic ground plane (`dryGroundOrSky`) is deleted at
  the end; nothing outside the SVO march draws the ground.

A second, independent backdrop world was considered first and rejected: it is a
second pipeline to maintain and eventually retire, with a seam and no
cross-shadowing. Its only advantage was avoiding the limits below — and those
limits are the implementation's, worth removing for their own sake.

## What stands in the way (measured from the code, 2026-09-26)

1. **Span-sized derived lighting.** The node-mip opacity pyramid must stay at
   <= 12 mip levels (`webgpu-svo-node-mip-pyramid.ts:23`,
   `webgpu-svo-live-derived-builder.ts:1540`) — a span of 16384 render cells,
   25.6 m at the in-app 1.5625 mm leaf — and its direct page table is dense over
   the span (192 MB cap, else <= 2048 pages per axis). Past either, the set's
   derived lighting is withdrawn (the 15x cliff). The record index is also
   span-sized (<= 262,144 cells, <= 1023 per axis).
2. **No distance LOD.** The planner supports mixed-level leaves
   (`adaptive-sparse-brick-plan.ts`) and traversal handles them
   (`svoNodeBounds`), but the in-app refined world applies no coarsening
   (`wetCoarsening = !dryWorld && refinementDepth === 0`), the wet ladder is
   capped at 8x and is feature-driven, and field programs / clusters pin to the
   finest level. A hill surface at the fine leaf is ~23 M leaves; with a
   pixel-footprint rule it is ~10^5.
3. **Container-bound terrain and shadows.** The render terrain field is clamped
   to the container footprint and height (`svo-render-solid-field.ts:36-45`,
   `webgpu-svo-render-terrain.ts`); directional shadows march only inside the
   container box (`directionalLightSceneExitDistance`, `program.ts:~4385`).
4. **Shared budgets.** 16,384 records, 4,096 cluster blocks (x10 uses 3,338),
   2^22 raster nodes, 512 candidates per brick.
5. **Rebuild coupling.** A scenery change rebuilds the whole world (and the
   solver, via `sceneryConstructionKey`). Static hills must not be re-voxelized
   on every set edit.

## Steps (each stands alone and is verified before the next)

### Step 1 — span-independent derived lighting
Make the node-mip pyramid and its page table sparse (sized by occupied pages,
not by span) and lift the 12-level cap, so a world spanning ~100-200 m keeps
cones/GI/AO. Also the record index. Gate: the hero frame is unchanged
(byte-identical or within the lane noise floor) and a synthetic wide-span world
keeps derived lighting.

**Landed (2026-09-26): page table is a hash, 20 levels.**

Why the limits were there:
- **Twelve levels.** The dense direct page table packed one Z slab per level,
  and its per-level slab offsets lived in fixed twelve-entry uniform arrays:
  DryParams words 100-111, the planner's `zOffsets`, the builder feedback's
  `zOffsets`, and the planner's twelve-entry `sections`/`capacities`. The
  sparse-bricks guard then refused a thirteenth level up front.
- **192 MB / 2048 per axis.** The table was one r32uint texel per page of the
  *domain's* page grid, dense over the span. Past either cap it was not
  ready, which was fatal at the sparse-bricks call site.
- Neither ceiling was about occupancy. The atlas, the radiance atlas and the
  directory were already sized by occupied pages plus the 48 MiB reserve.

What changed:
- `lib/svo/features/radiance/svo-node-mip-page-hash.ts` is an open-addressed
  linear-probe hash from `(level, x, y, z)` to slot + 1.
  - It lives in a 2D rgba32uint texture, so it costs 64 B per addressable page.
  - It is sized from the address plan's `pageCapacity` (occupancy plus
    reserve), at a load factor of at most 1/4.
  - At most 32 probes; the CPU build throws on a longer sequence rather than
    dropping a page.
  - The shape comes from `textureDimensions`, so readers need no uniform.
- The dry shader, cone fanout, derived planner and radiance feedback all read
  it through one WGSL `svoNodeMipPageHashFind`.
- The old z-offset words stay as reserved padding, so no param layout moved.
- Slots are unchanged, so the hero frame is bit-identical. The x10 dry smoke
  gives settled hash `0x04b79600` both before and after, and the raw rgba16f
  captures match byte for byte.
- Wide world: hero x10 plus blocks 120 m out, at 6.25 mm (a ~217 x 163 m
  span, bricks 4342 x 26 x 3262).
  - It needs 14 levels. The retired dense table would have been 2.8 GB, so it
    was withdrawn before.
  - The hash is 4 MiB, at 262,144 entries with a longest probe of 10. The
    opacity atlas is 54 MiB for 53k planned pages.
  - The Dawn smoke passes every check: derived lighting is ready with 38,861
    pages, all resident; 0 of 208 radiance pages are black; no validation
    errors; 12.27 ms against 12.3 ms for the hero alone.
- `SVO_NODE_MIP_MAXIMUM_LEVELS = 20`. The planner's worklist sections are
  sized to it.
- Tools:
  - `FLUID_SVO_WIDE_SPAN_M` (`withSvoWideSpanProxies`) adds two far scenery
    blocks in `tools/svo-fine-voxel-capacity.ts` and the dry smoke.
  - The capacity tool reports the hash beside the retired dense table's size.

Remainder of Step 1 (not needed to lift the ceiling):
- **Record index.** It can wait: it never withdraws. It coarsens
  `bricksPerCell` until it fits (<= 262,144 cells, <= 1023 per axis), so a
  200 m world only makes each record-maintenance cell hold more bricks. Make it
  a hash like the page table once Step 2's LOD world shows the maintenance cost
  in a profile.
- **Directory level starts.** DryParams `nodeMipLevelStart` has twelve
  entries, and only the ranged binary-search fallback uses it. It stays
  correct past twelve levels, because the last start bounds every coarser
  level. Retire that fallback, now that the hash is always ready.
- **Hash cost at small spans.** For hero x10 the hash is 2 MiB against a
  0.3 MiB dense table (and a 26 MiB opacity atlas). A load factor of 1/2
  would halve it if that ever matters.

### Step 2 — pixel-footprint LOD in the refined world
A distance-driven `refineEnvironmentLeaf` predicate (from the set centre: the
camera orbits the set), with environment leaves allowed to coarsen past the
solver cell (beyond `log2(brickSize)`). The set itself keeps its current leaf.

**Leaves must store only their surface.** Census (`tools/backdrop-capacity.ts`,
2026-09-26, hero lattice 12.5 mm, voxel clamp 6.25-250 mm): a heightfield puts
~54 surface voxels in each 512-voxel brick, so dense leaves are a ~10x tax.

| rule | leaves | surface voxels | dense @8 B/vox | surface-only |
|---|---|---|---|---|
| 2 px | 1.73 M | 92 M | 6.8 GiB | 704 MiB |
| 3 px | 0.94 M | 50 M | 3.7 GiB | 382 MiB |
| 4 px | 0.58 M | 31 M | 2.3 GiB | 238 MiB |

Ground is ~95% of it; vegetation is noise. So step 2 includes a surface-only
leaf payload (the existing `occupancy` / `banded` payload modes are the place
to start) and a surface-only claim for environment leaves. The finest backdrop
rung also bounds the span under step 1: +-72 m at 6.25 mm is 23,040 cells.

**Tiles carry the LOD (Peter, 2026-09-26).** Detail is chosen per large tile,
not per brick: a tile is an octree node (2^k bricks) built as a subtree at one
leaf level, chosen from the distance of the tile's nearest point to the
viewpoint. This does not cut voxel count (a tile takes the level its nearest
point needs, ~+20-30% over per-brick) — surface-only leaves and the px rule do
that. What it buys: LOD changes only at tile borders (one place to handle
cracks), a bounded per-tile budget (overrun is a loud per-tile error), and
**rebuild granularity** — moving the viewpoint rebuilds only tiles whose level
changed, with hysteresis. First cut measures from the set centre (static,
built once; identical in practice for the orbiting hero camera). Camera-relative
tiles with per-tile rebuilds follow once a tile builds within a frame or two.

Steps 2 and 3 land together: the backdrop ground (`backdrop-field.ts`) is the
geometry that exercises the tiles, so the LOD world is verified on the hills it
exists for.

### Step 3 — render terrain beyond the container
A render-only landscape field (never published to physics) spanning the
backdrop, meeting the slab at the seam height.

### Steps 2 + 3 — landed slice (2026-09-26)

**What landed.** The hills are ordinary voxel leaves of the one dry world on
renderer-only / dry worlds for scenes with `scene.backdrop`. A solver-owned
world never compiles the field, so no solver, SolidWorld or collider sees it.

- **Tile LOD** (`lib/svo/features/backdrop/backdrop-octree.ts`).
  - The planner's new `supplementalEnvironment` option runs a descent from the
    root. It is steered by `createBackdropNodeClassifier`, which answers
    empty / leaf / split per node.
  - Target voxel = `BACKDROP_PIXELS_PER_VOXEL * d * 2 tanHalfFov / 1744`,
    where `d` is the distance from the set centre to the node's nearest point.
    It is clamped to [the set's leaf voxel, 0.25 m].
  - Supplemental leaves never overlap planned leaves. A node that holds a set
    leaf splits around it, and supplemental leaves are always voxel terminals.
  - So the ground beside the slab is at exactly the set's resolution.
- **Surface-only claim.** A node is claimed only if the height bounds (envelope
  intersected with Lipschitz, tightened on a 4x4 sub-grid at the leaf decision)
  put the surface within half a voxel of it. Buried bricks are never claimed.
- **Terrain beyond the container**
  (`backdrop-voxelizer-wgsl.ts`, `webgpu-sparse-scene-proxies.ts`).
  - The field is baked into the voxeliser as WGSL literals (18 unrolled waves,
    analytic gradient and normal). It is unioned into `sampleSolidWorld`
    outside the footprint; the fuller sample wins, so set content never loses
    a voxel.
  - Material is `SOLID_WORLD_TERRAIN_MATERIAL_ID` (porcelain terrain). The
    cell law is the render terrain's.
  - The domain widens through `worldBounds_m`, which enumerates nothing.
  - The first publication dirties the backdrop box once; set edits never
    re-voxelise it.
  - Works at refinement 0 (SolidWorld sampler) and >0 (render terrain sampler).
- **Plane gating** (`program.ts`).
  - `groundPlane.w` carries `SvoGroundPlane.backdropContentRadius_m`.
    `sceneSvoGroundPlane` fills it for backdrop scenes and the dry-scene params
    pack it.
  - When it is non-zero, `dryGroundPlaneReplaces` is off, and the plane draws
    only past the content radius (the far fallback). Misses inside see sky.
  - `backdropSeamForScene` is unchanged.
- **Haze.** `dryBackdropHaze` applies `1-exp(-d/hazeDistance)` toward
  `dryHorizonColor(rd)`. It covers terrain-material hits outside the footprint,
  on backdrop scenes only.
  - It is applied at the end of `shadeDryOpaque` and `dryPrepassShadeNoGi`,
    before the reduced-rate cache stores radiance, so it is never applied twice.
  - Set scenery beyond the footprint was never hazed and still is not.

**Deviations, with numbers.**
- **Content radius is the haze cutoff, not the 72 m outer radius.**
  `svoBackdropContentRadius` = min(outer radius, hazeDistance x ln 256).
  - With the plane's 2.25 m haze that is 12.5 m. Past it a hill is the horizon
    colour to within an 8-bit step, and the hazed plane stands in.
  - A longer haze moves the cutoff out automatically. Art direction should
    decide whether the backdrop wants a longer haze than the plane's.
- **24 px, not 3 px.** Output of `tools/backdrop-capacity.ts --octree`
  (hero x10, 6.25 mm, backdrop alone):

| px | leaves | total | payload | cand. arena | opacity | radiance | plan + pages CPU |
|---|---|---|---|---|---|---|---|
| 4 | 326k | 2429 MiB | 1274 | 637 | 407 | 63 | 6.5 + 9.3 s |
| 8 | 137k | 1062 MiB | 534 | 267 | 172 | 66 | 2.5 + 3.9 s |
| 16 | 49k | 431 MiB | 193 | 97 | 63 | 72 | 0.7 + 1.2 s |
| 20 | 33k | 313 MiB | 129 | 64 | 42 | 72 | 0.4 + 0.8 s |
| **24** | **24k** | **250 MiB** | 93 | 47 | 31 | 76 | 0.3 + 0.5 s |

  - At 24 px the leaves per level are 6.25 mm 7.9k, 12.5 mm 5.0k,
    25 mm 6.4k and 50 mm 4.5k.
  - Without the haze cut (out to 72 m), 24 px is 43k leaves and 575 MiB. The
    far 200 mm clamp zone is 10k leaves, and every such leaf owns a radiance
    page.

**Measured (Dawn, M1 Max, hero x10 smoke, 800x460).**
- 38,074 leaves, against 14,258 without hills.
- Node-mip: 42,023 pages, all resident. 8,040 radiance pages, 0 black.
- Frame median 15.65 ms, against 12.11 ms without hills. About half the frame's
  pixels were cheap plane pixels and are now voxel hits.
- World build, warm: 2.17 s + 0.18 s first publication, against
  0.96 s + 0.08 s. That is +1.3 s (cold +1.6 s). Most of it is the node-mip
  stage (+0.7 s) and the plan (+0.4 s).
- The set's geometry and materials are unchanged. The remaining set-region
  difference is smooth, page-shaped indirect light: GI now sees hills where
  it saw sky, and the pyramid is deeper. Pond box mean |dL| is 0.46%, and 1%
  of its pixels are over 0.02.
- PNGs:
  - `artifacts/backdrop-svo/x10-baseline.png` (before)
  - `artifacts/backdrop-svo/x10-hills.png`
  - `artifacts/backdrop-svo/x10-diff.png` (|dL| x4)

**Remaining (toward 3 px and the full radius).**
1. **Surface-only leaf payload.**
   - About 54 of 512 voxels in a hill leaf are surface. `banded` still keeps
     the dense lanes because `encodeBandedLeaves` reads them, and its blob
     carries a fixed 1 KiB normal lane.
   - Needs: a record arena sized to actual records, no dense staging, and
     readers (DDA, hit, derived builder, brick occupancy) that go through
     occupancy rank.
2. **Candidate arena sized by the dirty budget, not `leafCapacity`.**
   - It is 2 KiB per leaf today, and hill leaves have zero primitive
     candidates.
   - This needs the first publication chunked instead of coalesced into one
     box.
3. **Radiance pages for coarse backdrop leaves.**
   - Every page at level >= 3 owns a radiance slot (~8 KiB). This is the floor
     that dominates at high px and in the far clamp zone.
   - A radiance floor per world region, or sharing the parent's page, would
     cut it.
4. **Planner scaling.** String keys and BigInt Mortons are ~12 us per leaf; at
   3 px (~1M leaves) the plan alone is >10 s.
5. **Cracks at LOD borders.**
   - Thin arcs are visible in `x10-hills.png` where the rung changes (the
     rings top-left).
   - Tiles currently pick a level per node. Handle this with skirts or a
     2:1-balanced border at tile edges.
6. **Record index** coarsens with the wider domain (now ~536x133x524 bricks).
   - It is fine at this size. Replace it with a hash per the Step 1 remainder.
7. **Render-terrain refinement arrays** (`createSvoRenderTerrainRefinement`)
   are allocated over the whole refined brick domain. At refinement >= 1 they
   should be restricted to the footprint.
8. **Frame cost.** Profile the +3.5 ms: primary DDA on hills versus cones/GI
   over hills.
9. **Vegetation** (Step 5) is not included.
10. **Camera-relative tiles with per-tile rebuilds** are not started.
    Shadows past the container (Step 4) are not started either: the set does
    not yet cast onto the hills beyond the container box.

### Virtual backdrop leaves — second slice (2026-09-26, later)

**What landed (code; CPU and GPU unit probes green).**
- **Surface-only payload: virtual leaves.** Backdrop leaves own no stored
  voxels. Their `voxelOffset` is `0x80000000 | ordinal*512`, and the planner
  gives dense offsets only to set leaves (`adaptive-sparse-brick-plan.ts`,
  `supplementalVirtualPayload`).
  - A table of 128 header words, then one record per leaf (origin xyz and
    voxel edge), sits in the owner lane's tail at word `voxelCapacity`
    (`backdrop-virtual-voxels.ts`, `packBackdropVirtualTable`).
  - The header holds the footprint, seam, envelope, waves, the content radius,
    K2 and the Lipschitz bound.
  - Readers route virtual voxels to analytic answers. These are the dry
    identity codec (`dryVoxelReadable`), the derived builder, brick occupancy
    (virtual leaves are left unsummarised), scene proxies (`markDirtyBrick`
    skips them), the surface mesh and the node-mip pyramid.
  - The backdrop owns no radiance or opacity pages. The capacity tool prints
    `node-mip: 0 pages` for the backdrop alone.
- **Exact primary.** A virtual leaf answers with the Lipschitz-stepped
  crossing of the height field over the leaf chord
  (`backdropVirtualCrossing`), refined by regula falsi. It is shaded with the
  analytic normal at the crossing.
  - The primary insertion is at the top of `traceLeafVoxelPayload` and its
    macro-HDDA twin. Visibility gets the same insertion, and the mesh
    background uses `dryBackdropTrace`.
  - No cell is stepped, so neither LOD-rung changes (the arcs, item 5) nor the
    voxel lattice (the contour moiré) can print on the surface.
- **Planner speed (item 4).** Numeric/BigInt keys and a column memo in
  `backdrop-octree.ts`. The plan takes 3.4 s for 708k leaves at 6 px, about
  4.7 µs per leaf against about 12 µs before.
- **Haze.** `backdrop.hazeDistance_m` is validated in `lib/core/backdrop.ts`
  and is 30 m for the hero.
  - It drives backdrop hits, the content radius (min(outer radius, haze × ln
    256), so 72 m for the hero) and the far-plane fallback. The horizon uses
    one haze throughout.
- **Hills retuned.** `HERO_GARDEN_BACKDROP.hills` is now amplitude 2.2,
  wavelength 10, rampWidth 3, valleyRise 4, with the flat ring still 0.4.
  - Relief is 0.27–0.97 m at 2 m from the footprint, up to 1.74 m at 3 m, and
    0.00–0.01 m at 0.5 m, so the seam stays flush.
  - The top of the hero frame meets the ground at r 2.4 m. At elevation 0.15
    it meets it at r 6.7 m, h 1.27 m. L = 4.30.
- **`BACKDROP_PIXELS_PER_VOXEL = 6`.** With an analytic primary, a finer px
  buys no primary fidelity. It adds leaves per ray, plan time and memory. The
  cone and shadow sampling of the backdrop is the only consumer of leaf
  resolution.

**Capacity** (`tools/backdrop-capacity.ts --octree --scene=hero-garden-hose-x10`,
retuned hills, 6.25 mm, backdrop alone, all leaves virtual):

| px | leaves | nodes | total | records | leaf slots | plan CPU |
|---|---|---|---|---|---|---|
| 4 | 1,363,670 | 1,676,985 | 282.5 MiB | 72.0 | 176.9 | 7.4 s |
| 5 | 983,303 | 1,208,988 | 207.3 MiB | 51.9 | 127.5 | 5.0 s |
| **6** | **708,217** | **871,577** | **153.0 MiB** | 37.4 | 91.9 | **3.4 s** |
| 8 | 432,794 | 532,486 | 98.5 MiB | 22.9 | 56.1 | 1.8 s |
| 12 | 218,768 | 268,909 | 56.1 MiB | 11.5 | 28.4 | 0.8 s |
| 16 | 129,837 | 159,699 | 38.6 MiB | 6.9 | 16.8 | 0.46 s |

Every row also carries a fixed 12.8 MiB of opacity and radiance atlas. The
virtual table is 2–21 MiB.

**Measured (Dawn, hero x10 smoke, 800x460, retuned hills, 6 px).**
- 722,089 leaf bricks. The shared rule now selects the canonical-parametric
  primary: 1.96 proxies per pixel.
- Frame median 21.89 ms, against 15.65 ms for the voxel-payload slice and
  12.11 ms without hills.
- PNGs: `artifacts/backdrop-svo/final-6px/{hero,low-orbit}.png`, with the log
  beside them.

**OPEN BUG — backdrop ground does not appear in the production frame.**
Everything outside the slab is the flat miss colour (218,214,209). The grey
ridges at the top of the low orbit are the environment, not hills: they are
identical before and after the hill retune.

What is ruled out:
- **The GPU table.** Header and records were read back after build and after
  settle, and they match the CPU pack word for word. The read base
  `payloadLanes1.y + payloadLanes1.z` equals the write base.
- **Topology.** The GPU structure words were dumped after settle. All virtual
  leaves are active and not dirty, and their backlinks are correct.
- **Traversal plus crossing on that exact topology.** A standalone Dawn
  compute probe ran the production `svoTraversalContinuation*` (parametric)
  with the production crossing WGSL over the dumped structure, for hero and
  low-orbit rays. Every ray hits, at t 3.1–27 m, after 4–42 leaves, with no
  status error.
- **The final-colour path.** A diagnostic that returned the first failing
  virtual leaf as a hit at its entry drew grey ground everywhere outside the
  slab. So virtual leaves are reached, and a virtual `DryHit` survives
  shading.

So the fault is between the megakernel's leaf loop and the crossing inputs.
The candidates are:
1. the leaf-visit budget (`tuningCounts0.x`, 24–48) running out before the hit
   leaf;
2. the entry-seed minimum;
3. `tEnter`/`tExit`/`ro` as the megakernel passes them.

The next step is one diagnostic render that colour-codes, per pixel, the
number of virtual leaves visited, the loop exit reason (budget, cursor
status or hit) and the crossing result of the last leaf. Delete the
diagnostic afterwards.

**Remaining after this slice.**
1. Fix the open bug above, then re-measure frame time. If the budget is the
   cause, virtual-leaf visits should not count against it; keep the 256 hard
   cap.
2. **Frame cost.** +6 ms over the payload slice.
   - Suspects: the entry-seed prepass rasterises about 710k leaf proxies every
     frame, and rays walk several shell leaves.
   - Measure these before choosing between a coarser px (16 px is 130k leaves)
     and a prepass that skips virtual leaves.
3. The candidate arena (item 2), record index (6), render-terrain arrays (7),
   vegetation (9), and tiles, rebuilds and shadows (10) are unchanged.

### Virtual backdrop leaves — third slice (2026-09-26, later still)

**Root cause.** The fault was not in the leaf loop. Two diagnostic renders
wrote a per-pixel work map (visibility state, virtual leaves visited, exit
reason). They showed:
- The hero view had 367,999 hits out of 368,000 pixels.
- In the low orbit, 419 pixels missed. 103 of those ran out of the 128 budget.
- Backdrop pixels visited 4.4 (hero) / 9.5 (low orbit) virtual leaves.
- The lighting pass received analytic terrain hits.

The hills were there. `dryPresentationHit` snaps every normal to its dominant
axis under `flatVoxelNormals`, which production turns on. Every slope under
45° became +Y, so the backdrop shaded as one flat tone the same as the miss
colour (218,214,209).

**Fix.**
- `program.ts`: `dryBackdropHit(hit)` recognises the virtual-hit signature, and
  `dryPresentationHit` keeps the analytic normal for it. The hills are an exact
  height field, not voxel scenery.
- `program.ts`: virtual leaf visits no longer spend the tuned budget
  (`budgetedVisits`). The 256 hard limit still bounds the loop.
- `webgpu-svo-primary-entry-prepass.ts`: backdrop leaves share one proxy box
  per coarse ancestor, 5 levels up (32 finest bricks, 1.6 m). The first leaf of
  each Morton run emits the box and later ones skip. At 6 px that is 12.6k boxes
  instead of 708k. The proxy is still a lower bound, so an empty texel is still
  a proof of absence.

Artifacts: `artifacts/backdrop-svo/fixed/{hero,low-orbit}.png`. Hills are
visible and flush with the pond slab, with no LOD arcs or contour moiré.

**Frame time** (hero x10, 800x460, 6 px):
- No hills: 12.11 ms. Payload slice: 15.65 ms. Before the fix: 21.89 ms.
- After the fix: 23.62 ms, then 24.13 ms in the profile run.

Interleaved stage-withholding profile (8 cycles, medians):

| withheld | frame | implied stage |
| --- | --- | --- |
| none | 24.45 | — |
| primary-entry-prepass | 23.88 | ~0.6 ms |
| primary-traversal | 1.19 | ~23 ms (includes downstream) |
| deferred-lighting | 19.93 | ~4.5 ms |
| reduced-shade / sky-lighting | 23.87 / 24.10 | ≤0.6 ms |

The prepass was never the +6 ms. After the proxy coarsening it costs about
0.6 ms. The cost is primary traversal over backdrop pixels:
- `backdropVirtualCrossing` marches the height field inside every shell leaf a
  ray enters: up to 96 Lipschitz steps plus 12 refinements.
- That cost repeats for 4–10 leaves per pixel.

**Remaining.**
1. **Traversal cost** (~+11 ms over no hills). Next lead: store a
   per-virtual-leaf height interval (min/max ground over its footprint) in the
   table, so a ray segment wholly above it rejects the leaf with one compare
   and no noise evaluation. A second option is a per-pixel early exit to the
   first straddling leaf.
2. **Lighting on the hills.** The hills show N·L plus environment only. They
   get no AO or shadows because virtual leaves have no opacity or node-mip
   pages, and cones over backdrop pixels march empty space. This belongs with
   Step 4.
3. Items 2–3 of the second slice's list are otherwise unchanged.

### Virtual backdrop leaves — fourth slice (2026-09-26, evening)

**Cost: what changed.**
1. **Per-leaf ground bounds.** Each leaf record grows from 4 to 8 words:
   origin, voxel, groundMin, groundMax, slope, curvature.
   - The bounds are a second-order Taylor bound on a 2x2 sub-grid:
     `h ± (½(|gx|sx+|gz|sz) + ⅛K(sx²+sz²))`, intersected with the envelope
     bound (`backdropTaylorHeightBounds`, backdrop-field.ts).
   - K is local. The global 14.83 /m (ramp) applies only inside the ramp;
     past it only the body term (~1.56 /m) is left.
   - Packing pads the bounds outward and fails fast on malformed records.
   - CPU sampling at 16 px: 0 violations, mean slack 80 mm. The earlier
     Lipschitz value grid had slack of about 1.4 leaf edges.
2. **Slab clip.** `backdropVirtualLeafCrossing` clips the ray chord to
   [groundMin, groundMax]. A ray above the slab rejects with one compare. An
   underground-entry guard catches chords that start below the slab. The flat
   ring has zero-thickness slabs, so the crossing accepts `t1 == t0`: `>=`,
   not `>`.
3. **Curvature-stepped crossing.** The fixed Lipschitz march is replaced:
   - Each step goes to the first root of `g + g'dt − ½K|rd.xz|²dt² = 0`, with
     g' = rd.y − ∇h·rd.xz (a safe Newton step), for at most 32 iterations.
   - CPU cost per leaf visit: 9.6 field evaluations down to 1.2.
   - Of 20k random leaf chords, 15 disagree with brute force. All are zero-gap
     touches at a chord end, on the content-radius rim or on the set seam
     (d = 0), which the neighbouring set geometry draws.
4. **16 px per voxel** (`BACKDROP_PIXELS_PER_VOXEL`, was 6). The exact
   crossing makes leaf size irrelevant to the image.

| px/voxel | CPU leaves (hills) | GPU leaf bricks (scene) | plan |
|---|---|---|---|
| 6, old Lipschitz claim | 708k | 722k | 3.4 s, 153 MiB |
| 6, Taylor claim | 368k | 382k | 2.65 s, 92 MiB |
| 16 | 73k | 87k | — |
| 24 | 39k | 53k | — |

**Frame time** (x10 lane, 800x460, hero; same harness):

| state | hero | low orbit (elev 0.15) |
|---|---|---|
| no hills (third slice) | 12.11 | — |
| third slice | 24.13 | — |
| slab reject, 6 px, Lipschitz march | 18.8–19.0 | 37.9 |
| **curvature crossing, 16 px (shipped)** | **18.66** | **27.7** |
| same at 24 px | 19.6 | 26.3 |

Stage withholding at 16 px (hero, then low orbit, in ms):

| withhold | hero | low orbit |
|---|---|---|
| none | 18.74 | 27.68 |
| primary-traversal | 1.25 | 1.41 |
| deferred-lighting | 14.75 | 23.91 |
| reduced-shade | 18.80 | 27.72 |
| primary-entry-prepass | 18.05 | 26.79 |

Caveat: another process was using the GPU at 50–65% between runs (ioreg
`Device Utilization`). Treat cross-run differences under ~1.5 ms as noise.

**Result.** Hills cost about +6.5 ms over no hills, down from +12. That is
not "a few ms":
- Primary traversal is still ~17 ms of the hero frame.
- 24 px (39% fewer leaves) was no faster.
- Field evaluations are already ~1 per straddled leaf.

So the rest is neither leaf count nor the crossing. The candidates are the
generic octree walk itself (descending from the root per leaf and
per-pixel restarts through the shell) and warp divergence between backdrop
and set lanes. The next measurement should count node visits per pixel
before any more optimisation.

**Contrast: why the hills read flat.**
1. **Framing.** The hero frame sees only the flat ring and the ramp: its top
   edge meets the ground at r ≈ 2.4 m. The low orbit reaches r ≈ 6.7 m. Real
   hills barely enter either view.
2. **Backlight.** The sun points toward −z at 40°, so the visible hill faces
   are the shaded sides.
3. **Tone ≈ haze.** Flat lit ground (~218) is essentially the horizon/haze
   colour (218,214,209). N·L itself works: on the set, lit tops read ~218
   and ambient-only faces ~103.

What was added (`dryBackdropTone`, program.ts; `backdropVirtualCavity`,
backdrop-virtual-voxels.ts):
- **Cavity AO from the analytic field.** The field is a sum of sines, and
  a Gaussian blur of radius σ = 0.2 wavelength scales each wave by
  `exp(−(kσ)²/2)`. So `Σ a(1−g)sin(phase)` is height minus its blur, at no
  extra samples. It maps to occlusion 0.55–1.0.
- **A warm hills albedo** (0.93, 0.89, 0.82), off the fog tone.
- Both weigh in by `smoothstep(0, 0.5, rampCoordinate)`, so they are exactly
  zero on the flat ring. The seam with the set's ground stays flush and
  untoned.

A sun-shadow march was **skipped** on measurement:
- The steepest slope is ≈ 0.90, and 99% of the area is under 0.6.
- tan(sun elevation) = 0.84.
- So the hills cannot self-shadow; a march would cost time and change
  nothing.

The result is subtle. The low orbit shows soft warm dune banding in the
haze, close in spirit to `output/imagegen/garden-pond-hose-fill-simplified.png`.
The hero frame barely changes because it contains no hills. Artifacts:
`artifacts/backdrop-svo/slice4/{hero.png, low-orbit.png, x10.log}`.

**Leads.**
1. Count octree node visits per backdrop pixel, then choose between:
   - inner-node height rejection: store max ground in a node word
     (`links.w`?). This touches the generic traversal and topology
     mutation;
   - a shell-local walk that skips the root descent;
   - splitting backdrop and set lanes to cut divergence.
2. Art direction: bring hills into the hero framing. Options are a nearer
   ramp or bigger amplitude near the set (`HERO_GARDEN_BACKDROP`), or a sun
   that side-lights the dunes.
3. Real AO and shadows on the hills still belong with Step 4.
4. The record's slope word is unused by the new crossing. Drop it or use it
   for a cheaper first step.

### Step 4 — world-bounded shadows
Shadow march bounds from the world's occupied extent, not the container.

### Step 5 — the backdrop as a geometry source; delete the ground plane
`scene.backdrop` (document field, validated, seeded for the garden hero scenes)
expands into the landscape field + vegetation primitives feeding the world.
Vegetation is a few primitives per plant (cloud tree = trunk capsule + puffs;
shrubs, tufts, pebbles), ~1-1.5k records, no clusters. Static backdrop bricks
survive set edits. `dryGroundOrSky` and `sceneSvoGroundPlane` are deleted; the
horizon beyond the voxel content is sky + haze in the ordinary miss path.

## Guardrails

- Fail fast: capacity overruns are loud errors, never silent withdrawal.
- CPU checks first (`tools/svo-fine-voxel-capacity.ts`, the backdrop capacity
  tool), naga validation, then one Dawn render per iteration. No browser gates.
- The set's frame cost must not regress; measure on the existing x10 lane.

### Tiled terrain to the horizon — fifth slice (2026-09-26, night)

Peter: "remember we want full terrain out to the distance. LOD based on
tiles. proceed."

**Design (written before the code).**

*Why the octree is the wrong accelerator for the ground.* Slice 4 left the
hills at +6.5 ms (hero) / +15 ms (low orbit), and the profile ruled out the
crossing (about one field evaluation per straddled leaf) and the leaf count
(24 px was no faster than 16 px). What is left is the generic walk: each
virtual leaf is reached by the cursor descending from the root through a
domain widened to the backdrop's ±72 m, with per-pixel restarts through the
shell. The ground is a height field, so a ray can find it with a 2D walk and
a height interval per cell; it never needs the third axis the octree pays for.

*The tile layer.* A 2D clipmap of terrain tiles, centred on the set:
- Level 0 is a 16x16 grid over the square `centre ± R0`, with R0 twice the
  footprint's larger half-extent (1.8 m for the hero, so 0.225 m tiles).
- Level l is a 16x16 grid over `centre ± R0·2^l`. Its central 8x8 belongs to
  level l-1. Tiles double in size per level, so a level is one ring of 192
  tiles, and the count grows with log2(reach), not reach².
- Levels are added until the ring reaches the haze horizon: the footprint
  distance where the haze leaves less than 1/256 of the ground,
  `hazeDistance · ln 256` (166 m at the hero's 30 m haze, so 8 levels and
  2,048 tiles, 32 KiB).
- Each tile stores the ground's conservative `[min, max]` over its footprint
  (the slice-4 Taylor bound on a 2x2 sub-grid, intersected with the envelope)
  and its local curvature bound K. Tiles wholly inside the footprint are
  empty. Each level also stores its cumulative maximum (over its whole square,
  finer levels included).
- LOD is the tile level: tiles are small near the set, where the camera
  orbits, and coarser with distance. The crossing inside a tile stays the
  exact analytic one (slice 4's curvature-stepped root find, refined to the
  surface), so a tile's size changes what a ray pays, never what it sees: no
  cracks at level borders, no popping.

*The walk.* `backdropTerrainTrace(ro, rd, tMin, tMax)`:
1. Find the finest level whose square holds the current point
   (`ceil(log2(max(|q.x|,|q.z|) / R0))`) and the tile in it.
2. If the ray stays above the level's cumulative maximum until it leaves the
   level's square, jump to that exit (a min/max pyramid of depth two: level,
   then tile).
3. Otherwise clip the ray's chord in the tile to the tile's `[min, max]` slab
   and, outside the footprint rectangle, run the curvature-stepped crossing
   with the tile's own K. A chord that stays above the slab costs one compare.
4. Advance to the tile's exit. Stop at tMax, on leaving the outermost square,
   or once a rising ray is above the global maximum.
Tile visits and crossing steps share one per-ray budget.

*One world, not a second pipeline.* The octree remains the accelerator for
what is sparse and 3D (the set). The tile layer is the terrain's accelerator.
The same traversal functions consult it beside the octree and keep the nearer
hit, so it lives inside the SVO renderer:
- `traceStaticFrom`, and so every primary and secondary ray, the mesh
  background and exact visibility, returns one `DryHit`. It has the same
  material, the one `shadeDryOpaque`, and the same haze.
- The table sits where the virtual-leaf table sat: in the owner lane's tail,
  so there is no new binding and no new parameter word.
- The fluid never sees it. Nothing about the document changes: the solver
  never compiles `scene.backdrop`.
- Cones and GI still see only the set. Backdrop virtual leaves never owned
  node-mip pages either, so this is not a regression. It is Step 4's
  business. Directional shadows consult the tiles, so the hills can occlude
  the set at a low sun.
- Backdrop leaves leave the octree. The world's domain shrinks back to the
  set, and the virtual-leaf machinery is retired: the planner's supplemental
  environment, the virtual table records, the entry-prepass proxy sharing,
  and the virtual branches in the identity codec, derived builder, brick
  occupancy, topology mutation and surface mesh. Leaving it half-live would
  keep a dual path; fail-fast says retire it.

*No content radius, no hard edge.* The ground is covered everywhere outside
the footprint. A ray that leaves the outermost square (where the haze is at
least 255/256) or rises above every hill is a miss. For a backdrop scene a
miss shades as the sky, clamped to the horizon for downward rays. That is
exactly the colour the haze fades the ground to, so the terrain melts into the
horizon with no edge. `dryGroundOrSky`'s plane never draws on a backdrop
scene; it stays only for the garden scenes that have no backdrop.

*Camera motion.* Tile bounds depend only on the field, not the view: they are
built once per scene on the CPU, in milliseconds, and written with the table.
A camera orbit rebuilds nothing. The rings are centred on the set, which the
orbit camera always faces from a few metres away. A camera zoomed far out
meets coarser tiles near itself, which costs steps but never accuracy.
Camera-centred rings with per-tile rebuilds are the follow-up if that ever
shows in a profile.

**Course correction (Peter, mid-slice): "the hills are too tall / steep.
Also, I expected an extension of the voxel terrain out several large tiles
beyond the pond."** Two changes followed, and they replace parts of the
design above:
- *Voxels, not an analytic surface.* The analytic field stays the source.
  What a ray meets is its voxelisation: one cubic column per cell, with the
  column's top the field at the column centre rounded to the lattice.
  - Level 0's cell is the set's own scene cell (0.0125 m for the hero), on
    the set's lattice. The clipmap centre is snapped to that lattice, R0 is
    rounded up to whole tiles of cells, and the vertical anchor is the seam
    snapped to the lattice. So the seam is voxel against voxel.
  - Each level doubles the cell, so voxels coarsen with distance: 1.6 m
    cells in the outermost ring. Every tile holds 18x18 whole columns of its
    level, level squares fall on the coarser grid, and the grids nest.
  - A hit is a `DRY_GBUFFER_FIELD_VOXEL` hit whose normal is the face it
    entered. It takes the ordinary presentation (six-axis normals, and cell
    seams drawn on that voxel's own lattice via `dryFaceLattice`). The
    analytic-normal exemption in `dryPresentationHit` is gone. The cavity
    tone is sampled at the voxel centre, so every face of a voxel wears one
    tone.
  - Inside a tile the curvature-stepped crossing is replaced by a 2D column
    DDA: one field evaluation per column. Each column top is clamped to the
    tile's interval.
  - Tile bounds are the exact min/max column top, evaluated in f64 on the
    CPU. The shader's clamp means f32 rounding can never put a column
    outside the bounds the walk trusts. The per-tile curvature and the cone
    tolerance are retired.
- *Low, gentle relief.* `HERO_GARDEN_BACKDROP`: amplitude 1.5 → 0.5 m,
  wavelength 6 → 9 m, ramp 1.5 → 3 m, valley rise 3 → 0.8 m. The largest
  slope anywhere drops from 1.24 to 0.21. The ground is 0.05–0.25 m up 2 m
  out, 0.15–0.45 m by 5 m, and about 1 m at the outer radius.

**Landed and measured (2026-09-26, night).** All runs are x10, 800x460,
median of 8. They are same-session arms against the same working tree, with
the GPU idle at launch. Numbers from other nights are not comparable: the
slice-4 tree measured 26.0 ms tonight against 18.7 ms in its own write-up.

| arm | hero | low orbit | wide (el 0.3, d 4.5) |
|---|---|---|---|
| no backdrop | 18.20 | 16.96 | 10.33 |
| slice 4 (virtual leaves, old hills) | 26.02 | 32.61 | — |
| **slice 5 (voxel tiles, gentle hills)** | **20.82** | **20.52** | **14.98** |

The backdrop now costs:
- hero: +2.6 ms, down from +7.8
- low orbit: +3.6 ms, down from +15.6
- wide view: +4.7 ms. Almost every pixel there is backdrop.

Withholding stages splits the cost between the primary and deferred passes:

| | hero, withhold deferred | hero, deferred | low orbit, withhold deferred | low orbit, deferred |
|---|---|---|---|---|
| no backdrop | 7.77 | 10.4 | 7.94 | 9.0 |
| slice 4 | 13.61 | 12.4 | 21.19 | 11.4 |
| slice 5 | 9.84 | 11.0 | 10.11 | 10.4 |

- The primary share of the backdrop is +2.1 / +2.2 ms, against +5.8 / +13.3 ms
  in slice 4.
- The deferred share (+0.6 / +1.4 ms) is the per-pixel terrain shadow ray and
  the cavity tone.
- Withholding the primary-entry prepass changes little in any arm.

Per-pixel visits, from the GPU work map (slice-5 terminal "backdrop" also
counts the set's own terrain voxels, which share the material):

| | hero | low orbit |
|---|---|---|
| slice 4 octree nodes / backdrop px | mean 35.0, p99 79 | mean 47.8, p99 99 |
| slice 5 octree nodes / backdrop px | mean 14.4, p99 54 | mean 15.5, p99 57 |
| slice 5 tiles / backdrop px | mean 13.9, p99 25 | mean 14.5, p99 29 |
| slice 5 columns / backdrop px | mean 2.4, p99 14 | mean 10.2, p99 79, max 182 |

- The octree nodes left on backdrop pixels are the walk up to the set's own
  bounds. The terrain itself costs no node visits.
- The walk never exhausted its budget (384) in the CPU mirror. The mirror
  counts over 5,800 rays per camera agree with the GPU means.

*Correctness (CPU).* `traceBackdropTerrain` (the f64 mirror, with the same
counts) was checked against a brute-force march over the voxelised field,
5,800 rays per camera. Of 19 disagreements, all are hits the mirror found
and the march stepped over: thin column corners, each confirmed by a gap
sign change at the reported t. No ray misses a voxel the march finds.

*Orbit cost.* Nothing is rebuilt. The table (8 levels, 33,344 B) is built
once per scene in about 0.12 s on the CPU, which is 2,048 tiles x 324
column tops.

*Images* (`artifacts/backdrop-svo/slice5/`: hero, low-orbit, wide, x10.log):
- The voxel terrain continues from the set's slab edge in shallow one-cell
  terraces. It runs out through every ring to the haze, which melts it into
  the sky at the horizon. No level border is visible.
- The terraces read as fine dark contour lines: their risers face away from
  the sun. That is the same look as the set's own terraced bowl.
- The wide view shows the footprint rectangle faintly. The set's scatter
  stops at it, and the set's detail runs finer there than the backdrop's
  0.0125 m cells. The ground heights are flush.

*Deviations from the design above:*
- Tile bounds are exact column tops, not Taylor sub-grids.
- The footprint split is gone. Columns whose centre is inside the footprint
  are air, and tiles that overlap the footprint skip the lower band clip.
- Exact visibility (`svoTraceVisibility`) and cones/GI do not see the
  terrain; only the directional shadow ray does. The shadow ray found no
  terrain shadows at the default sun on the gentle hills.
- The mesh background calls the same walk. A camera placed low and far
  enough to sit inside a hill sees a voxel wall at t = 0.

*Still to do:* vegetation (`backdrop-vegetation.ts`, `backdrop-lod-census.ts`
are groundwork); cones and GI over the terrain (Step 4); and the analytic
ground plane, which stays only for the garden scenes without a backdrop and
never draws on a backdrop scene.

### Extended detail tiles — sixth slice (2026-09-26, late night)

Peter: "good. but remember i want to actually extend the detailed part of the
scene. [...] this gives us shadows and a more vibrant scene". In his
screenshot, the set's footprint (1.8 m x 1.2 m) holds stored voxel content:
terrain, stones and shrubs, with sun shadows, AO and GI. Outside it the
slice-5 walk is flat and unlit, reads as contour lines, and ends at a hard
rectangle.

**Design (written before the code).**

*Detail rings are the walk's own inner levels, stored as octree leaves.*
- Ring `l` (for `l < L_d`, where `L_d = backdrop.detailRings`) is walk level
  `l`: the square `c0 +- R0 2^l`, minus ring `l - 1` (ring 0 minus the
  footprint).
- Its voxels are the walk's voxels: cell `h0 2^l`, one column per cell. On
  x10, `h0` is the scene cell, 6.25 mm (refinement depth 0), so the rings
  are 6.25, 12.5, 25 and 50 mm and ring 0 is at the set's own voxel size. The column top is `anchor + round((h - anchor)/cell)
  cell`, the same f32 evaluator (`backdropTerrainSurfaceAt`).
- So a stored ring is exactly what the walk drew at that level. The seam
  with the walk (ring `L_d - 1` against level `L_d`) is the seam the walk
  already had between levels, and ring 0 meets the set's slab at the same
  anchor. Both are flush by construction.
- Leaf level for ring `l` is `solverLevel - l`. A ring-`l` leaf is an
  8^3-voxel brick of `8 h0 2^l`.
- The walk starts at level `L_d` (`planBackdropTiles({ firstLevel })`):
  finer tiles are empty, so the level maximum is -inf and the walk skips
  those squares in one step.

*Alignment.*
- `c0` snaps to `worldOrigin + k G`, with `G = 8 h0 2^(L_d-1)` (one
  outermost-ring brick). `columnsPerTile` must be even.
- Then every ring boundary (`R0 2^(l-1)` from `c0`) falls on a node boundary
  of that ring's leaf level, and stored columns coincide with walk columns.
- The anchor is the seam at 0.30 m, a multiple of 50 mm. With the domain
  origin on the 50 mm brick lattice, every ring's voxel lattice holds the
  column tops exactly for `L_d <= 4`.

*Scatter (set-style, seeded, never in the fluid).*
- Evaluated in the voxelizer and never an authored primitive, so
  `scene.scenery`, the primitive arena and the solver never see it.
- Four jittered-grid classes, each item kept inside its own grid cell, so a
  voxel reads one hash per class:
  - gravel (0.10 m grid)
  - stones (0.3 m)
  - three-lobe shrub puffs (0.8 m)
  - large three-lobe bushes (2.4 m), from ring 1 outward only (see Budget)
- Each item is seated on the column top at its centre.
- Items smaller than 0.6 of their ring's cell are dropped. Density fades over
  the outer half of the detail square, so detail thins and coarsens outward
  instead of ending at an edge.
- Material is the terrain's, as it is on the porcelain set, so the backdrop
  tone, haze and face lattice all apply.

*Planner.*
- A supplemental descent runs after the set's plan, which it leaves
  byte-identical.
- From the root:
  - a node under a set leaf is skipped;
  - a node that is an ancestor of set leaves splits;
  - otherwise the backdrop classifier answers empty, leaf or split.
- Coarse nodes use Taylor height bounds plus the largest scatter rise. Leaf
  nodes use exact column tops over the brick plus a one-column margin, plus
  the scatter items that overlap it, with relaxed acceptance so the CPU
  claims a superset.
- A brick is a leaf when content straddles it, including a fully solid brick
  beside a lower column, so riser faces are stored.
- Supplemental leaves take the voxel terminal and stay out of the
  terrain-resampling and planar split ledgers.

*Domain, voxelizer, publication.*
- `worldBounds_m` widens to the detail square (padded by `G`), with y up to
  the highest ring top plus the scatter rise.
- `sampleSolidWorld` gains a union with `sampleBackdropDetail`: ground and
  scatter, only outside the footprint and inside the detail square. It uses
  full or empty voxels with the up normal, which is the set's voxel look.
- The initial publication's dirty box includes the detail square.

*Lighting (Step 4).*
- `directionalLightSceneExitDistance` is container-bounded, so a receiver
  outside the container reads fully lit and casts nothing. It moves to the
  world box (`nodeMipOrigin`/`nodeMipExtent`, the box the cones already
  use), so voxel shadows, AO and cone GI cover the rings.

*Budget.*
- Estimate: about 1k leaves per ring (clipmap area is constant in leaves)
  plus scatter, at 4 KiB per dense leaf.
- The ring count is chosen from a CPU census (leaves, voxels, node-mip pages,
  plan time) and one Dawn render per arm.

**Results.**

*The first estimate was wrong by 4x.* The census had assumed `h0 = 12.5 mm`
with one refinement level. x10 is refinement depth 0 at a 6.25 mm scene
cell, so every ring is one level finer than planned: four rings came to
39.9k leaves, not 7.6k. The CPU census now agrees with the Dawn planner to
within 1%. There was no planner bug.

*Bushes were most of ring 0.* A 0.5 m three-lobe puff voxelized at 6.25 mm
covers about a thousand 50 mm bricks. Per-class census, four rings:

| Classes | Leaves | Ring 0 |
|---|---|---|
| terrain + gravel | 18.6k | 5.5k |
| + stones | 20.8k | 6.6k |
| + shrubs | 24.3k | 8.0k |
| + bushes (R 0.25–0.55, every ring) | 39.6k | 18.9k |

Bushes now start at ring 1 (`minimumRing`), with R 0.22–0.45. The set
already has its own planting beside the pond. The CPU and WGSL rules share
the same ring threshold (0.75 of the ring cell, and ring cells are exact
powers of two).

*Plan time.* The leaf-level column span depends only on a node's xz
footprint, so it is now cached per `(level, x, z)`. Classifying four rings
dropped from 4.6 s to 1.0 s of CPU (about 223k nodes).

*Budget curve.* x10, one Dawn render per arm on an idle GPU, 800 x 460. The
baseline was re-run on today's tree.

| Rings | Detail leaves | Leaf bricks | Node-mip pages | Pyramid | Hero | Low orbit | Wide |
|---|---|---|---|---|---|---|---|
| 0 (slice 5 today) | 0 | 14,258 | 12,208 | 12.5 MB | 20.06 ms | 19.84 ms | 15.06 ms |
| 2 | 13,420 | 27,682 | 29,554 | 33.7 MB | 23.90 | 25.73 | 22.35 |
| 3 (CPU only) | 21,759 | ~36k | – | – | – | – | – |
| **4 (default)** | 28,054 | 42,283 | 48,264 | 118.3 MB | 25.48 | 29.03 | 25.44 |
| 4, before the bush and ring fix | 39,886 | 54,115 | 62,050 | 133.5 MB | 24.16 | 29.55 | 29.67 |

- Detail voxel memory is about 4 KiB per dense leaf: roughly 53 MiB for two
  rings and 110 MiB for four.
- Four rings cost +5.4 ms on hero, +9.2 ms on low orbit and +10.4 ms on
  wide against today's baseline.
- The cost follows lit screen coverage, not leaf count. Every stored pixel
  now pays for its shadow ray, AO and cone GI, where the walk it replaces was
  unlit. The pre-fix arm had 42% more leaves but was only 1–4 ms dearer.

*Choice: 4 rings* (`HERO_GARDEN_BACKDROP.detailRings`).
- With two rings, the unlit contour-line walk starts 3.6 m out and fills the
  upper half of the wide and low-orbit views. That is exactly the look
  Peter asked to replace.
- With four rings, lit content reaches 14.4 m and the walk is only visible as
  hazed horizon.
- Two rings is the fallback if the +10 ms on wide is not acceptable.

*Images* (`artifacts/backdrop-svo/slice6/`: hero, low-orbit, wide, x10.log).
- Against Peter's screenshot, the set-style content now continues past the
  footprint: the same porcelain terrain, gravel, stones and shrub puffs,
  casting and receiving sun shadows, with AO and GI.
- The footprint rectangle is gone. The join at the set slab is not visible
  in any view, and neither are the ring seams (flush by construction; cell
  size steps are hard to spot through the fade).
- Scatter thins outward, and the far bushes read as rounded masses in the
  haze.
- Weaknesses:
  - Terrace risers still read as contour lines inside the lit rings.
  - Near the camera, bushes are large and dominate the wide view.
  - Scatter has one tone, so the rings are not yet "vibrant" in colour.

*Left.*
- Frame cost: the +5 to +10 ms is in shading. The first thing to measure is
  whether the shadow and cone rays over the 30 m world box, or primary
  traversal of 42k leaves, dominate.
- Lighten far-field riser faces (`dryBackdropTone`) beyond the rings.
- Add colour variation per scatter class.
- A 3-ring Dawn arm was not run (render budget).
- Memory: the 118 MB node-mip pyramid is the biggest item. It is
  own-level pages over coarse leaves; ring 2–3 leaves could share coarser
  pages.

### Consistent sunlight — seventh slice (2026-09-26, late night)

Peter, on the slice-6 renders: "quite amazing! what explains why the sun only
shines on some of the scene? want it to be consistent."

**Cause: three things, and one of them is not about the backdrop.** Every
direct-sun exit (the inline deferred term, the reduced-rate prepass and the
voxel light cache) goes through one node-mip cone, `dryConeVisibility`.

1. **The sun cone ran out of steps at 0.79 m.** It has 48 steps at a
   0.065 rad aperture. The first ~15 steps are one finest cell each, and after
   that each step is 6.5% longer than the last. So the march covered about 126
   finest cells and reported the rest of the way as clear. At the x10 lattice
   (6.25 mm) that is 0.79 m. Any blocker further along the sun ray cast
   nothing, anywhere: the tree canopy never shadowed the ground, and tall
   bushes had cut-off shadows. The reach was also tied to the lattice: the
   12.5 mm hero lane reached twice as far.
2. **The cone could not see the coarse detail rings.** Ring `l` stores leaves
   of cell `h0 2^l`. A leaf owns pyramid pages only at its own level
   (`liveSvoLeafPage`, "own-level"), and nothing finer exists inside it. The
   cone samples at its own diameter, which starts at one finest cell. So in
   ring `l` it read air until its diameter grew to `h0 2^l`, at 0.19, 0.38 and
   0.77 m for rings 1, 2 and 3. Ring 3's contact shadows were therefore gone,
   and ring 2 kept only the far part of each shadow. The normal escape and the
   self-shadow ramp were also sized to the set's cell, not the receiver's.
3. **The hills wore a tone of their own.** `dryBackdropTone` multiplied every
   terrain-material hit past the flat ring (ground and scatter, stored or
   walked) by a warm albedo (0.93, 0.89, 0.82). It also applied an analytic
   cavity term of 0.55–1.0 that eased in with the ramp. That dimmed and
   yellowed the key light on the outer rings compared with the set. It dated
   from slice 4, when the hills had no real AO.

Ruled out:
- Haze (it is `1-exp(-d/30 m)` toward the horizon, the same everywhere).
- Walk hits: `dryBackdropOccludes` runs before the cone for every receiver,
  and the world-box exit (slice 6) already covered the rings.
- The reduced-shade paths: they call the same cone.
- Snapped normals: they are the same six-axis rule at every level.

**Diagnostic.** One temporary shader arm (since removed) output, per pixel,
R = the key light's visibility, G = N·L and B = the tone, on all three
framings (`diag-before-*.png`). It showed:
- no canopy shadow;
- contact shadows only in the inner rings;
- the far field tinted yellow by the tone.

`diag-diff-*.png` shows the change in visibility after the fix (blue = new
shadow, red = lighter):
- the canopy's shadow now lies on the ground right of the pond;
- the far bushes gain contact shadows at their feet;
- the tips of mid-range shadows soften slightly (see the cost below).

**Fix.**
- `program.ts` (`createSvoDryConeMarcherWGSL`) floors each cone sample's
  footprint in two ways, beyond the aperture:
  - *Reach.* When the remaining budget cannot reach the end of the segment at
    the aperture's rate, the footprint grows geometrically so the remaining
    steps end exactly there:
    `distance * ((end/distance)^(1/stepsLeft) - 1)`. Directional and AO cones
    now always cover their whole segment. Anchored (finite-emitter) cones keep
    their half-way split and the emitter ladder.
  - *Stored voxel.* With `storedVoxelFloor`, the footprint is never finer than
    the stored voxel where the cone samples (`backdropStoredVoxelWidth`). The
    cone therefore reads each ring at the pyramid's own resolution there. The
    start distance and the self-coverage ramp scale with the receiver's
    voxel, and so does the normal escape at both direct-light call sites
    (`dryStoredVoxelWidthAt`).
- `backdrop-terrain-tiles.ts`: `BackdropStoredLattice` /
  `backdropStoredVoxelWidth`. The ring count is read from the table: stored
  levels are the leading levels whose cumulative maximum is -inf. The cell is
  0 inside the footprint and the outermost ring's cell beyond the rings.
- `dryBackdropTone`, `dryBackdropHit` and the WGSL cavity term are deleted.
  Haze is now the only thing distance does to the ground. The table word that
  held the cavity radius is reserved (written 0).
- The cone fan-out experiment (off by default) gets the reach floor but not
  the stored floor: its layout has no payload arena.

**Images** (`artifacts/backdrop-svo/slice7-sun/`: hero, low-orbit, wide,
x10.log; `diag-*`; `x10-before.log`). The "before" frames are
`artifacts/backdrop-svo/slice6/`: the baseline run tonight hashed
0xdaabfa33, as slice 6 did.
- The far rings are the same porcelain white as the set.
- The canopy shades the ground behind the pond.
- Far bushes sit in their own shadows.

**Frame time** (x10, 800x460, cone scale 1, median of 8, same session):

| | hero | low orbit | wide |
|---|---|---|---|
| before | 23.97 | 29.37 | 24.64 |
| after | 29.18 | 37.60 | 29.88 |
| delta | +5.2 | +8.2 | +5.2 |

An intermediate arm (both floors, tone still on) measured
30.34 / 36.31 / 31.03, so the tone removal is free. The whole cost is the
cone. There was no profile (render budget). Likely contributors, in order:
- Every sun cone now spends its 48 steps sampling real pages all the way up
  to the world-box top. Before, it quit at 0.79 m, and in the rings it quit
  after page misses.
- AO cones in the rings now fetch resident coarse pages where they used to
  miss.
- `backdropStoredLattice()` runs once per cone call: an up-to-8-level loop of
  storage loads, plus about 8 more loads. It could be hoisted to once per
  pixel, or kept as a table header word.

**Left.**
- The fixed 48-step budget now trades crispness for reach. The footprint
  outgrows the aperture from about 5 cm out (0.14 d against 0.065 d for a 3 m
  segment), so mid-range shadow tips soften. Two ways to buy back aperture
  fidelity:
  - more steps, only where the segment is long;
  - a coarse empty-space skip above the tallest content, since the world box
    is taller than the ring content.

  Either belongs with the frame-time task.

### Frame time — eighth slice (2026-09-26, night)

Task: bring the slice-7 frame back toward the 0-ring baseline without
changing the look. Budget: 8 Dawn renders (5 used). All numbers are x10,
800x460, cone scale 1, median of 8 frames.

**Profile** (render 1, instrumented build; stage withholds and lighting
switches are runtime writes, so all arms ran in one process):

| arm | hero | low orbit | wide |
|---|---|---|---|
| base | 33.8 | 37.2 | 32.6 |
| deferred lighting withheld | 11.1 | 17.3 | 12.4 |
| sun shadows off | 18.9 | 24.6 | 19.2 |
| AO off | 26.9 | 31.2 | 26.8 |
| shadows and AO off | 12.7 | 18.9 | 13.6 |
| cone budget 24 (not 48) | 28.8 | 33.0 | 28.7 |

- Nearly all of slice 7's cost is the deferred-lighting cones. The sun cone
  costs 12.5–15 ms and AO 6–7 ms. The tile-walk shadow (0.3 ms), the prepass,
  the reduced shade and the sky are each negligible.
- Per-pixel counts: sun cones take p50 48 steps (the whole budget) on the set
  and rings 0–1. The means are set 34, ring 0 about 40, ring 1 31–40, ring 2
  23–30 and ring 3 12–20. AO takes about 16 steps on the set and ring 0, 12 on
  ring 1, 4 on ring 2 and 0 on ring 3. There are about 28 page finds and 45
  atlas samples per pixel.
- Halving the budget saves 4–5 ms, so a sun step costs about 0.2 ms per step
  per pixel across the frame. The rest of the sun cost is fixed: the lighting
  closure, the per-cone setup and the first near-field steps.

**Null result: empty-space skip.** Jump through non-resident node-mip
ancestors (up to four levels, one page-table probe each). Sun steps fell by
only 4–9, but page finds rose by about 16 per pixel, so the frame got
*slower* (38.7 / 42.4 / 40.2 ms). It also moved 2–4.5% of pixels. Why:
- The ancestors of almost every sample are resident, because the ground and
  the set lie inside them, so the empty boxes are small.
- The reach floor then re-spends the saved budget on the rest of the way to
  the world-box top.

Removed.

**Null result, instructive: clipping the cone's end.** A content ceiling
(below) ended each sun cone where its ray clears all content, instead of at
the world-box exit. It cut sun steps by 2–5 per pixel. But because the reach
floor is `distance*((end/distance)^(1/steps)-1)`, a nearer end also shrinks
every footprint along the way. So the image changed: 10–28% of pixels moved,
0.3–0.9% by more than 8/255, with crisper and darker shadow tips. The time did
not move (30.6 against 30.5 ms on the hero). Fewer steps at finer LODs cost
as much as more steps at coarser ones: a finer footprint means more page
switches, and so more finds.

**Changes kept** (bit-exact: the settled hash stays 0x9392fb5d, and all three
sweep frames are pixel-identical to `slice7-sun/`):
1. **Content ceiling, as a loop exit.** `BackdropContentCeiling`
   (`backdrop-terrain-tiles.ts`) is a 128×128 xz map of the highest leaf-box
   top per column, dilated by one column. It is built on the CPU from the
   octree plan's leaves: every node-mip page lies under a leaf, so no sample
   above a column's value reads anything. It is appended to the terrain table
   in the owner-lane tail, so it needs no new binding. The section index and
   the size go in header words 125–126, which the waves never reached (21
   waves at most, as before).
   - Edits raise their columns in `stagePrimitivePublication` and upload only
     the touched rows. The initial publication needs no raise, because the
     leaves already cover it. A removed body leaves its old top, which is
     conservative.
   - On x10, 16,129 of 16,384 columns hold content: p50 0.75 m, p90 1.15 m,
     highest 1.55 m, in a world box 1.85 m tall.
   - The marcher (`contentCeilingClip`) keeps the full footprint schedule to
     the world-box exit. It only stops the loop once the ray is past the last
     column it could still touch. The margin is 4 receiver voxels plus two
     footprints of distance, the larger of the aperture and the reach step.
     `backdropContentCeilingEnd` is a 2D DDA over the columns. Every sample
     it keeps is the full march's sample, and the skipped ones read zero.
     - Sun steps per pixel: hero 36.3 → 31.3, low orbit 29.8 → 25.5,
       wide 36.0 → 31.1.
     - Page finds drop by 2.5–3 per pixel.
2. **Page validity read once per page switch.** `dryNodeMipAt` reads the
   page-validity texture when the page cache switches pages, not on every
   sample, as a third `resident` state. Validity cannot change within a pass.
3. **The stored ring count is a header word** (`latticeWord + 1`), so
   `backdropStoredLattice()` no longer runs its up-to-16-level loop per cone.

**Measured gain** (render 3, arms interleaved in one process, two repetitions,
GPU otherwise idle):

| arm | hero | low orbit | wide |
|---|---|---|---|
| both changes | 30.8 / 30.8 | 34.8 / 36.5 | 30.4 / 30.7 |
| no ceiling stop | 31.5 / 32.8 | 35.4 / 37.1 | 31.7 / 34.5 |
| per-sample validity | 32.5 / 33.2 | 37.4 / 36.5 | 32.0 / 32.5 |
| neither | 33.8 / 33.9 | 37.4 / 37.3 | 33.1 / 33.4 |

Each change is worth about 1–2 ms, and together they are worth about 3 ms.
Within an arm the spread is about ±1 ms. The low orbit is the noisiest.

**Before and after** (clean build, no instrumentation; three repetitions of
median-of-8 in one process, `slice8-perf/x10-timing.log`):

| view | 0 rings (before sun fix) | slice 5 | slice 7 | slice 8 | vs slice 7 |
|---|---|---|---|---|---|
| hero | 20.1 | 20.8 | 29.18 | 28.9 | −0.3 |
| low orbit | 19.8 | 20.5 | 37.60 | 34.3 | −3.3 |
| wide | 15.1 | 15.0 | 29.88 | 29.3 | −0.6 |

The instrumented slice-7 base in this session measured 33.8 / 37.2 / 32.6,
so the like-for-like gain is about 3–5 ms. The slice-7 row came from an
earlier session. **The target (the 0-ring baseline plus a few ms) is not
met.** The slice-7 look has a real price: every sun cone now covers its whole
segment, and every ring receives AO.

**Images:** `slice8-perf/` (hero, low-orbit, wide, x10.log) are
pixel-identical to `slice7-sun/`. No look trade-offs were made.

**Left** (every remaining lever changes the image, so it needs Peter's call):
- **The sun cone's fixed cost.** With shadows off the frame is 18 ms. The
  shadow costs about 12 ms, of which the step count is only about 5. The
  rest is per-cone setup, near-field steps at the finest LOD (one voxel
  each, about 15 of them), and the closure. Leads:
  - start the march at a coarser LOD, once past the self-shadow ramp;
  - share one sun cone across a 2x2 quad (reduced-rate shading, as cone
    scale 0.5 does);
  - the voxel light cache, which the smoke lane turns off.
- **AO at 6–7 ms.** It runs 4 cones of about 16 steps on the set and ring 0.
  Fewer cones (2) or a coarser start would roughly halve it. Ring 3 is
  already free.
- **The budget.** 24 steps saves 4–5 ms but softens mid-range shadows (the
  slice-7 reach trade-off). With the ceiling stop in place, a budget sized to
  the clipped segment might keep crisp near-field shadows and cost less.
- **Node-mip memory** (118 MB; coarser pages for rings 2–3) and the primary
  traversal were not examined. The 0-ring runs show the primary costs little
  here.
