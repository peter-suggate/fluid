# Renderer consumes H: plan for step 1 (all-4h endpoint), Uniform Geometric

Read-only research, 2026-10-04. Line numbers are from the working tree as read today; other
agents are editing the solver, so each reference also names the symbol. Nothing was run
(no Dawn, no tests). Costs below are counted from code (texel loads, launches), not measured.

**Status: Option A steps 1-6 are implemented (same day).** The research text below is kept as
written; statements that turned out wrong are marked "Corrected" in place, and the section
"Implementation record" at the end has what was built, the measurements and what remains.
The follow-up on the polygonise pass (the normal at zero and partial detail) is the last
section, "Polygonise at zero and partial detail".

Notation: lattice n per axis (cubic for the byte counts), t = n/4 tiles per axis, T = t^3 tiles,
H = 4h. r32float = 4 B/texel, rgba32float = 16 B/texel.

| | 64^3 | 128^3 | 256^3 |
|---|---|---|---|
| t | 16 | 32 | 64 |
| (t+1)^3 r32f (H vertex phi) | 19,652 | 143,748 | 1,098,500 |
| (n+1)^3 r32f (h vertex phi) | 1,098,500 | 8,586,756 | 67,898,372 |
| n^3 r32f (h cell field) | 1,048,576 | 8,388,608 | 67,108,864 |
| n^3 rgba32f (h face field) | 4,194,304 | 33,554,432 | 268,435,456 |

## Headline findings

1. There is **no per-frame publication or expansion pass** for the renderer. Consumers bind the
   solver's own fields: `present()` (webgpu-uniform-reference.ts:484) and
   `UniformTexturePages.publication()` (uniform-texture-pages.ts:167-174) return the field
   itself for adopted detail fields. The only dense n^3 publication is the t=0
   `encodeInitialPresentationSurface` (webgpu-uniform-reference.ts:1762).
2. The "expanded h volume" the handoff forbids (handoff :40, :163-169) exists today in two forms:
   **identity-placement field extents** (the bound textures are n^3 / (n+1)^3) and the
   **water extraction classify scan** over (n+1)^3 cubes.
3. `UniformMixedPhiResolve` is seam-tile-scaled and the renderer's sampler does not depend on it
   (`resolved=false` in the presentation copy). The `presentation` pressure/phi buffers are
   T-scaled buffer copies.
4. The water surface is **marching cubes on the cell-centred h dual lattice**, not a ray march.
   It is watertight today because every tile is marched on one global h lattice over a C0 field.

## Q1. What each consumer reads per frame

Source object: `DenseLevelSetVolumeConsumerSource` (levelset-consumer-abi.ts:297-314), built at
webgpu-uniform-reference.ts:1421 (`mixedSource`), exposed by the getter at :490.
Solver field getters: `volumeTexture` :516 (volumeA), `surfaceFieldTexture` :517 (surfaceB,
centre phi), `velocityTexture` :523 (velocityA). Renderer wiring: webgpu-renderer.ts:2078-2091,
:2568, :3404-3415.

| Consumer (file:line) | Binding | Resource | Format, logical extent | Sampling rule | Read at zero detail? |
|---|---|---|---|---|---|
| Water extraction, classify + polygonise (webgpu-water-pipeline.ts:380-403) | 13 `denseVertexPhi` | solver vertex phi | r32f (n+1)^3 | `fieldCell` :462-466: `0.5 - umSampleVertex(cell+0.5)/h`; owner lookup, 8 width-strided loads of the owner's corners (uniform-mixed-vertex-sampling.wgsl.ts `umSampleVertex`) | yes, stride-4 texels only |
| | 14 topology | `mixedFrame.ownership.presentation` (the topology buffer, uniform-mixed-ownership.ts:220-222) | u32, 16 B/tile + detail table | `umOwnerAt`, `umTileStencil` | yes |
| | 1 `volume` | surfaceB (centre phi) | r32f n^3 | not read when `umPresentationEnabled()` (:466) | bound, unread |
| Water normals (:532-562 `uniformPhiNormal`) | 13, 14 | as above | | 27 taps of `umVertexValue`, Gaussian-weighted least squares; each hanging tap = 8 loads (`umVertexFrom4`) | yes |
| Water composite, rigid contact band (:1139, :1148, :1208-1240, gate :1368) | 8 `liquidField` | surfaceB | r32f n^3 | identity: raw `textureLoad` trilinear over 8 h cells, treated as mode-1 occupancy; packed: `udrLoadCell` | only when a rigid body is within the contact band |
| | 17 `liquidTopology` | topology | | `uniformDetailRuntimeWGSL` | |
| Grid overlay, fragment (webgpu-grid-overlay.ts:117-149, bind group :2299-2327) | 21 `sliceDensePhi` | vertex phi | r32f (n+1)^3 | `umSampleVertex` (grid-overlay-levelset-volume.wgsl.ts) | when a phi/volume layer is on |
| | 9 `densityField` | volumeA | r32f n^3 | `udrLoadCell(densityField, umOrigin(umOwnerAt(cell)))` (:857-861): owner-origin texel only | layer on |
| | 5 `velocityField` | velocityA | rgba32f n^3 | `udrLoadFace` + `umSampleVelocity` (:916-929, :1794) | layer on |
| | 2 `fluidField` | surfaceB | r32f n^3 | mixed path goes to `umSampleVertex` (:829-833); `textureDimensions(fluidField)` still used twice | dims only |
| | 22 `sliceDenseOpen` | gammaB | r32f n^3 | read only when presentation is disabled | bound, unread |
| | 12, 13, 14 | `mixedFrame.presentation.pressure`, `.phi` (+ stage grids), `ownership.support` | u32/f32 buffers | pressure/tiles/importance layers | layer on |
| | 17 `sparseTopologyArena`, 23 `viewRecords` | topology; shared method-view slot | | | |
| Secondary particles (webgpu-secondary-particles.ts:92-98, :711) | | | | `WebGPUSecondaryParticleSystem` is never instantiated (:538-560); uniform solver publishes no `secondaryParticles`; only the render half exists and gets `setSource(undefined)` (webgpu-renderer.ts:1496) | no |
| Fluid coverage / shadows (webgpu-renderer.ts:1989-2012) | | | | needs a brick publication; never created for uniform-volume; shadows are off | no |
| Smoke harness (webgpu-smoke-readbacks.ts:980-982, webgpu-smoke-executor.ts:212-216) | | same as the water pipeline | | binds `vertexPhi` + `mixedOwnership` exactly as the renderer | |

Per-frame solver-side work done *for* presentation (uniform-mixed-frame.ts):

| Work | Line | Scale |
|---|---|---|
| `copyBufferToBuffer` root pressure -> `presentation.pressure`, root phi -> `presentation.phi` | :689 | pressure words; all-4h layout = t^3 + 2(3t^2) slots |
| pressure stage grid, band slot map copy | :690, :693 | tile words |
| stage views (`recordStageView`/`recordStageGrid`) | :546, :552, :605, :851, :870 | tile words; only while layout views are enabled |
| `UniformMixedPhiResolve.encode` | :625, :626, :635, :788, :861 | one workgroup (125 lanes) per **seam** tile; zero seam tiles -> the GPU-counted launch runs empty |

Renderer-owned buffers (surface-scaled, webgpu-water-pipeline.ts:279-289, `ensureGeometry` :2263-2285):

| Buffer | Formula | 64^3 | 128^3, 256^3 (capped) |
|---|---|---|---|
| vertices | clamp(80 * 3n^2, 262,144, 2,097,152) * 32 B | 31,457,280 | 67,108,864 |
| activeCubes | cap/3 * 8 B | 2,621,440 | 5,592,408 |
| globalCubeValues (allocated, unused on the dense path) | cap/3 * 32 B | 10,485,760 | 22,369,632 |
| globalCubeOffsets | cap/3 * 24 B | 7,864,320 | 16,777,224 |

## Q2. Zero detail: what is h-sized, what is n^3 work

### Bound resources at zero detail

| Resource | Bound by | Identity (production) | Packed (QA; base block only, `uniformDetailLayout` :63-98, patch edge 32) |
|---|---|---|---|
| vertex phi | water 13, overlay 21 | (n+1)^3: 1,098,500 / 8,586,756 / **67,898,372** | folded base [33,33,17] / [66,66,9] / [132,132,17]: 74,052 / 156,816 / 1,184,832 |
| surfaceB centre phi | water 1, composite 8, overlay 2 | n^3: 1,048,576 / 8,388,608 / **67,108,864** | t^3: 16,384 / 131,072 / 1,048,576 |
| volumeA | overlay 9 | n^3: same | t^3: same |
| gammaB open fraction | overlay 22 (unread) | n^3: same | t^3: same |
| velocityA | overlay 5 | n^3 rgba32f: 4,194,304 / 33,554,432 / **268,435,456** | (2t)^3 rgba32f: 524,288 / 4,194,304 / 33,554,432 |
| topology | water 14/17, overlay 17 | 65,956 / 526,500 / 4,210,852 (T-scaled) | same |
| support | overlay 14 | (9n+36+2 pages) words, about 9.47 MB at 256^3 (T-scaled) | same |
| presentation pressure / phi | overlay 12 / 13 | 256^3: 1,146,880 / 8,421,408 (T-scaled; +64 words per h band tile) | same |

Under packed placement every load pays a directory read (`udrEntry`) and the base fold
arithmetic (`udrBase`); under identity the table mode word is 0 and `udrInit` returns early
(uniform-detail-abi.ts:52-56), so identity has **no tile dimensions in the table** and consumers
take the lattice from `textureDimensions` (water :403, overlay levelset wgsl).

### Per-frame work that scales with n^3

| Item | Scale at zero detail | Evidence |
|---|---|---|
| Water `extractMain` classify | **n^3**: ceil((n+1)/4)^3 workgroups of 4^3, one cube per thread, 8 corners * `umSampleVertex` = 64 texel loads + about 20 topology words per cube. 4,913 / 35,937 / 274,625 workgroups; 17.6 M / 137 M / **1,086 M** texel loads per extraction | :247-259 `surfaceExtractionDispatchPlan`, :712-715, :2572 |
| Polygonise + normals | surface-scaled (activeCubes worklist, indirect) | :662-709, :2531-2534 |
| `UniformMixedPhiResolve` | zero seam tiles: 3 empty launches per frame (+1 at relayout), no n^3 work | uniform-mixed-phi-resolve.ts; frame :625-635 |
| `presentation` buffers | T-scaled copies | frame :377-383, :689-693 |
| `encodeInitialPresentationSurface` | n^3, t=0 and post-upload only | reference :1762, :1981 |
| Anything in webgpu-uniform-reference.ts expanding coarse onto h per frame | **none found** | `present()` :484 is identity |
| Grid overlay | per pixel | fragment shader |

Extraction runs once per new solver revision (`shouldUpdateWaterSurface`, :63-74), so with
presents paired 1:1 with steps it is a per-frame cost.

Latent defect, noted in passing: under identity the composite `contactFluidValue` (:1215-1232)
reads surfaceB raw as occupancy, but surfaceB is centre phi and only owner-origin texels are
canonical for 4h tiles. Reached only with a rigid body in the contact band.

## Q3. How the surface is found, and what marching H needs

Today (webgpu-water-pipeline.ts):

- Lattice: cell-centred dual lattice with a wall halo. Lattice point p maps to cell p-1
  (`latticeValue` :496-502): value 0 (air) for p.x, p.z <= 0, p >= dims+1; y is clamped at the
  floor (no floor closure). `latticeWorld` :504-507 clamps to the walls. (n+1)^3 cubes.
- Stage 1 classify (`classifyCubeScaled` :631-652): load 8 corners, min/max against 0.5, append
  `vec2u(x | z<<16, y | scale<<16)` to `activeCubes`. The scale field exists already (compact
  coarse path) and `polygoniseMain` honours it (`loadCubeCornersScaled` :609-618).
- Stage 2 polygonise: `prepare` sizes an indirect dispatch, 64 threads per workgroup, private
  slices. Normals from `uniformPhiNormal`.
- No empty-space skipping, no acceleration structure, no inter-frame cache (the mesh is retained
  only between solver revisions).

Why it is crack-free today: adjacent cubes share lattice points evaluated by one function, and
the field is C0 across h/4h seams because a hanging vertex is owned by the coarsest incident
cell and is that cell's trilinear interpolant (`umVertexAuthority`, `umVertexFrom4`).

What changes to march an H-spaced phi with h only in fine tiles:

| Concern | Requirement | Cheapest satisfying form |
|---|---|---|
| Step by tile width | Do not visit h cubes where the H data proves no crossing | Tile-window test on the H vertices (below). Exact, no geometry change |
| | Emit H-pitch triangles in 4h tiles | Needs a second marching lattice (Option C): contours on a shared face differ between an H cube and 16 h cubes, so it cracks without transition cells |
| Seam continuity | Shared face samples must be identical from both sides | Keep one global h dual lattice and one sampling function; only the *visit set* changes |
| Wall closure | Halo air at x/z low and all high faces | Unchanged if the lattice is unchanged; an "all liquid" skip must exclude windows touching the halo |
| Normals | Gradient stencil should not cost 216 loads per vertex in coarse tiles | Unchanged in the smallest change (surface-scaled). A width-stepped stencil is a later quality/cost change with its own seam (shading line where the stencil width switches) |
| No inter-frame caching | Visit set recomputed each extraction from current H | The tile test is stateless |
| No transparency, shadows off | Cracks show the scene behind opaque water | Reason to keep the mesh identical in step 1 |

Exact skip test. A thread owns the window g in [0,t]^3 = cube bases 4g..4g+3 per axis (today's
workgroup alignment). Those cubes sample cells 4g-1..4g+3, i.e. tiles g-1 and g, whose 4h
interpolants read only H vertices g-1..g+1 (27 values, indices clamped to the lattice):

- all 8 tiles are 4h and all 27 H phi > 1e-6 h -> every corner < 0.5, halo is 0 -> no crossing, skip;
- all 8 tiles are 4h, all 27 H phi <= 0, and the window touches no halo (g.x > 0, g.z > 0, g < t
  on all axes) -> every corner >= 0.5 -> skip;
- otherwise run the unchanged `classifyCube` on the window's cubes.

The margin on the air side covers `0.5 - s/h` rounding to 0.5; the liquid side is sign-exact.
A 4h owner is always `regular` in `umSampleVertex` (owner width 4 >= stencil maximum), so the
convex-hull bound holds whatever the neighbours are. Result: the active cube set, and so the
mesh, is identical to the full scan.

## Q4. Options

Common numbers. Phi state bound by water extraction, bytes at 64^3 / 128^3 / 256^3.

| Option | Zero detail, identity | Zero detail, patch layout | Full detail, identity | Full detail, patch layout (P=32, 33^3 slots) | Per-frame publication | Classify loads at 256^3 zero detail |
|---|---|---|---|---|---|---|
| Today | 2,147,076 / 16,975,364 / 135,007,236 (vertex phi + surfaceB) | n/a | same | n/a | none | 1,086 M |
| **A. Coarse phi texture + tile-driven classify** | **19,652 / 143,748 / 1,098,500** | same (it is the vertex base block) | 1,118,152 / 8,730,504 / 68,996,872 | 1,169,636 / 9,343,620 / 74,697,476 | identity: 1 launch, (t+1)^3 texels (4,913 / 35,937 / 274,625). Patch layout: tile corners of resident patches, 9^3 = 729 texels each, or none if the solver writes the base directly | about 7.4 M test + 35 M in surface windows |
| B. Accessors only + tile-driven classify | 1,098,500 / 8,586,756 / 67,898,372 (still binds (n+1)^3, reads stride 4) | 74,052 / 156,816 / 1,184,832 (folded base) | same as today | 1,224,036 / 9,356,688 / 74,783,808 | none | as A plus a directory read per load when packed |
| C. H cubes in 4h tiles, h cubes in fine tiles | as A | as A | as A | as A | as A | about 0.3 M (one H cube per tile) |
| D. Solver-published surface tile list | as today or A | | | | list build in the solver | surface tiles only |
| Expanded h volume per frame | disallowed (handoff :169) | | | | n^3 | |

Surface-window estimate for A: a flat surface flags two window layers, about 2(t+1)^2 = 8,450
windows at 256^3, each 64 cubes * 64 loads = 34.6 M; plus (t+1)^3 * 27 = 7.4 M test loads.
About 26x fewer loads than today at 256^3, 15x at 128^3, 8x at 64^3.

### Option A (recommended)

Solver publishes a (t+1)^3 r32float `coarseVertexPhi` with texel g = phi at vertex 4g. Consumers
load 4-aligned vertices from it and every other canonical vertex from the detail field. At zero
detail no unaligned vertex is ever loaded (`umSampleVertex` regular path and `umVertexFrom4`
load only `origin + 4*corner`), so the detail binding is a 1^3 dummy.

Shader changes per consumer:

| Consumer | Change |
|---|---|
| `uniformMixedPresentationWGSL` (uniform-mixed-presentation.wgsl.ts) | new optional `coarsePhi` argument. `umLoadVertex(p)`: `if(all((p&vec3u(3u))==vec3u(0u))){return textureLoad(coarse,vec3i(p>>vec3u(2u)),0).x;} return udrStoredVertex(phi,p);`. `umDimensions()`: `4u*(textureDimensions(coarse)-vec3u(1u))` (placement independent; fixes identity having no tile dims). **Corrected:** the argument is required, and the per-vertex alignment branch measured about +10% classify time at Full, so the loader is decided by the load site instead (see the record) |
| Water extraction (webgpu-water-pipeline.ts) | new texture binding 15 `coarseVertexPhi`; `uniformPhiNormal` guard uses `umDimensions()+1` instead of `udrVertexDims(denseVertexPhi)`; new entry point `extractTilesMain` (one thread per window, ceil((t+1)/4)^3 workgroups; skip test, else a 64-iteration opaque-bound loop over **one** `classifyCube` call site); `volume` binding gets the pipeline's 1^3 fallback when the mixed source is bound. `classifyCube`, polygonise, normals unchanged. **Corrected:** the thread loop measured 3-10x slower than the full scan; the window test is kept but the windows it cannot skip are listed and scanned by a second launch, one workgroup per listed window. There was no 1^3 3D fallback texture; one was added |
| Water composite | none in the first cut (see step 6) |
| Grid overlay (webgpu-grid-overlay.ts, grid-overlay-levelset-volume.wgsl.ts) | new **texture** binding 25 (texture count 10 -> 11; no storage buffer, so the 10/10 budget and binding 23 are untouched); `uniformMixedPresentationWGSL(17,"sliceDensePhi",…,"sparseTopologyArena")` gains the coarse argument; `textureDimensions(sliceDensePhi)-1` goes. Volume and velocity stay on `udrLoadCell`/`udrLoadFace` |
| Secondary particles | none (not live) |

Under a later patch layout: `coarseVertexPhi` is the vertex-class base block kept as its own
unfolded (t+1)^3 texture (t+1 <= 513, no fold needed), `detailVertexPhi` is the atlas, and the
publish pass becomes the base sync over resident patches' tile corners. That gives the vertex
class the "base is current for every tile corner at each published revision" rule the storage
review asks for (storage design :83-123, item 3). The consumer shader does not change between
placements: aligned loads never touch the directory.

Risks:

| Risk | Note |
|---|---|
| Seam cracks | None by construction: same lattice, same `classifyCube`, exact skip. The lane asserts equal cube counts |
| Stale coarse texel at a fine tile's corner | Publish must follow the last phi write of every path that changes phi before a present: frame tail (after uniform-mixed-frame.ts:635, unconditionally), relayout remaps (:788, :861), t=0 and uploads (reference :1381-1430, :1996-2002), voxel edits |
| Overlay Metal compile | `umLoadVertex` gains a branch and a second `textureLoad` at every inlined site. All sampler loops are already opaque-bounded (`uniformMixedPresentationLoops`); keep the new loader a single function, add no call sites. Check with the CPU preflight before any Dawn run |
| Single call sites | The tile loop calls `classifyCube` from one site with a runtime bound (`min(arrayLength(&umTopology),64u)`), as the existing eight-tap loops do |
| Full-detail timing | One thread per window loops 64 cubes serially, same total loads as today but 64x fewer threads. If the one benchmark run shows a regression at full detail, the alternative is today's dispatch plus a per-workgroup skip (thread 0 tests, `workgroupUniformLoad`, others return): identical cost at full detail, but n^3 idle thread launches at zero detail. **Corrected (measured):** the thread loop regressed everywhere it had work (128^3 Full 8.9 vs 4.85 ms), and the named alternative was not identical at Full either (6.03 vs 4.85 ms). Neither was kept |
| Rebinds | `detailVertexPhi` appears/disappears with fine capacity; the source object must change identity so `setVolume` (:1983) and `setDenseLevelSetVolumeSource` rebuild bind groups. **Corrected:** `setVolume` runs only when the volume texture changes, and the water pipeline had no source setter; it now has `setDenseLevelSetVolumeSource`, called each frame beside the overlay's |
| Mesh density | A leaves h-pitch triangles over a trilinear H field (surface-scaled, n^2, capped 67 MB vertex buffer). That is not n^3 and not a bound h volume, but it is 16x the triangles the data supports. Reducing it is Option C |
| Composite contact | still binds an n^3 cell field under identity until step 6 |
| Benchmark tool | tools/benchmark-water-extraction-dawn.ts:44 calls `setVolume` without `vertexPhi`/`mixedOwnership`, so today it times the raw-`volume` path, not the Geometric one. Fix before using it as evidence |

### Option B

No new texture; add only the tile-driven classify and have the test read `udrStoredVertex(phi, 4g)`.
Zero extra bytes, no publish. Fails the stated goal under identity (binds (n+1)^3), and under
packed placement every aligned load pays `udrEntry` + `udrBase`. Its base-currency question
(a resident patch's aligned vertices live in the atlas, the base copy may be stale) is exactly
the reopened review item. Reasonable only as a half-step if the solver agents do not want a new
texture this week.

### Option C

H cubes on the primal H vertex lattice in 4h tiles (corners are stored values, zero interpolation),
h cubes in fine tiles. Triangles /16 and a T-scaled vertex buffer at zero detail. Costs: transition
cells or skirts at every h/H marching boundary (bilinear face contour against a chord), a new wall
closure (the primal lattice has no halo), a different mesh at every occupancy, normals stencil
rework. Opaque water makes any crack visible. Separate decision after A; A's coarse texture and
window test are prerequisites for it anyway.

### Option D

Have the solver publish the surface tile list (for example the importance `crossing` bit). Those
records exist only under Dynamic or while layout views are enabled, and the (t+1)^3 test in A is
already 7.4 M loads at 256^3. Not needed.

### Verifying lane

No existing `tests/*dawn*` file runs `surfaceExtractionShader` on the mixed presentation
(`water-surface-parallel-scan-dawn`, `water-wall-contact-alignment-dawn`,
`water-wireframe-depth-dawn` cover the global-fine shaders; `visual-layers-dawn` builds the
overlay without `mixedOwnership`). The single targeted lane is therefore new:

`tests/uniform-surface-extraction-dawn.test.ts`, run as `npm run test:dawn -- uniform-surface-extraction`.
One device, 64^3 `uniformVolumeMethod`, bound as webgpu-smoke-readbacks.ts:980-982, three
states (zero detail, one Requested region, Full). Asserts:

1. `coarseVertexPhi[g] == vertexPhi[4g]` after a step (readback of both);
2. count-only vertex totals (`override countOnly`, :404, :641) are equal between `extractMain`
   and `extractTilesMain` in all three states;
3. at zero detail the pipeline builds and extracts with the detail phi and `volume` bound to 1^3
   dummies, with no validation error and the same count.

Before it: CPU preflight (`tools/validate-water-shaders.ts` parses the extraction and overlay
shaders with naga; `tools/capture-uniform-wgsl.mts` for the solver publish kernel). Timing
evidence, if wanted: one run per arm of the corrected benchmark tool. `visual-layers-dawn` is the
existing check that the overlay still compiles and draws once step 5 lands.

## Recommendation and ordered steps

Option A. It is the only option that meets "binds and samples only T-scaled phi at zero detail"
under identity today, it does not depend on the reopened packed layout, the consumer shader is
the same under both placements, and the mesh is unchanged so there is no visual risk to argue
about. Option C is the follow-up that removes the h-pitch mesh.

| # | Step | Files |
|---|---|---|
| 1 | Solver: allocate `coarseVertexPhi` ((t+1)^3 r32float, STORAGE + TEXTURE binding) and one publish kernel `H[g] = phi[4g]`, ceil((t+1)/4)^3 workgroups, encoded unconditionally at the frame tail after the last phi write and after each relayout remap, t=0 and upload. Expose on the source: `coarseVertexPhi?: GPUTexture`, `detailVertexPhi?: GPUTexture` (absent while fine capacity is 0), keep `vertexPhi` until consumers move | lib/methods/uniform/uniform-mixed-frame.ts (:562 `advance`, :635, :788, :861), lib/methods/uniform/webgpu-uniform-reference.ts (:1381-1430, :1421, :490), lib/core/levelset-consumer-abi.ts (:297-314) |
| 2 | Sampler: `coarsePhi` argument, split `umLoadVertex`, `umDimensions()` from the coarse texture | lib/methods/uniform/uniform-mixed-presentation.wgsl.ts |
| 3 | Water extraction: binding 15, `extractTilesMain` with the exact skip test, dispatch ceil((t+1)/4)^3 in place of `plan.full` when the mixed source is bound, `setVolume` takes the source (coarse, detail-or-dummy, topology), `volume` dummy | lib/core/webgpu-water-pipeline.ts (:380-403, :532, :712, :1983, :2343-2360, :2572), lib/core/webgpu-renderer.ts (:2082), lib/harness/webgpu-smoke-readbacks.ts (:980), lib/harness/webgpu-smoke-executor.ts (:212), tools/benchmark-water-extraction-dawn.ts (:44) |
| 4 | Lane: CPU preflight, then the new Dawn lane once | tests/uniform-surface-extraction-dawn.test.ts |
| 5 | Overlay: texture binding 25, coarse argument, dims from the coarse texture | lib/core/webgpu-grid-overlay.ts (:117-149, :2192-2200, :2299-2327), lib/core/grid-overlay-levelset-volume.wgsl.ts |
| 6 | Composite contact: sample phi through the same presentation module (`0.5 - umSampleVertex/h`, one helper, opaque loop) instead of the n^3 cell field; this also fixes the identity-path defect. Until then `liquidField` stays the one n^3 binding in the water pipeline | lib/core/webgpu-water-pipeline.ts (:1139-1240, :2424) |
| 7 | Later, separate decisions: Option C (H-pitch mesh with transition cells); width-stepped normals; overlay volume/velocity become T-scaled when the solver's cell and face storage do (t^3 and (2t)^3 base blocks), with no overlay shader change because it already reads through `udrLoadCell`/`udrLoadFace` | |

Questions for the solver agents before step 1:

1. Is "the vertex base is current for every tile corner at each published revision" acceptable as
   the base-currency rule for the vertex class? Option A assumes it.
2. Under the patch layout, can the vertex base be its own unfolded (t+1)^3 texture rather than
   z slabs of the atlas texture? Option A's aligned load is then a plain `textureLoad`.
3. Which signal tells a consumer that fine capacity is 0 (so the detail binding may be a dummy)?
   Today `fineTiles` capacity is `fixed ?? (gpuLayout ? tiles : 0)` (reference :1395): Dynamic
   reserves every tile, so under Dynamic the detail texture is always bound even at zero detail.

## Implementation record, 4 October 2026

Option A, steps 1-6. Everything below was run: CPU preflight (types, naga over the water,
overlay and solver shaders), the new lane, and three one-process benchmark runs on Dawn/Metal.
Line numbers are the working tree at the end of the day.

### Answers to the three questions

1. Yes. "The vertex base is current for every tile corner at each published revision" is the
   base-currency rule of the vertex class. Under identity placement it is one launch.
2. Yes. The base is its own unfolded (t+1)^3 r32float texture; the patch layout need not fold it.
3. Fine capacity is `UniformMixedCapacity.fineTiles` on the mixed ownership. The source names
   `detailVertexPhi` only while it is above zero, and the source object changes identity when
   that flips. Requested and Full construct at C = 0 and grow; Dynamic holds C = T, so under
   Dynamic the detail field is always bound. One shader at every occupancy: at C = 0 the same
   code runs with a 1^3 dummy bound, which it never loads.

### What was built

| Step | What | Where |
|---|---|---|
| 1 | `UniformCoarseVertexPhi`: (t+1)^3 r32float, one launch `H[g] = phi[4g]` over ceil((t+1)/4)^3 workgroups | lib/methods/uniform/uniform-coarse-vertex-phi.ts |
| 1 | Encoded at the frame tail after the surface-volume apply (the last phi write), after a host relayout's remap, and at t=0; source fields `coarseVertexPhi`, `detailVertexPhi` | uniform-mixed-frame.ts :658, :813, :901; webgpu-uniform-reference.ts :490-504, :1427, :1436; levelset-consumer-abi.ts :317-323 |
| 2 | Presentation sampler takes the base: `umDimensions()` from the base, `umLoadCoarseVertex`, `umLoadFineVertex`, loader chosen by load site | uniform-mixed-presentation.wgsl.ts :33, :57-66; uniform-mixed-vertex-sampling.wgsl.ts :24-26, :80, :88, :125 |
| 3 | Water extraction: binding 15 (base), binding 16 (window list), `uniformWindowQuiet`, `collectWindowsMain`, `extractWindowsMain`; `setDenseLevelSetVolumeSource` per frame; 1^3 fallback for the unread cell field and absent detail field | webgpu-water-pipeline.ts :413-414, :746-793, :2122, :2171-2187, :2527, :2604-2611; webgpu-renderer.ts :2082, :3417; webgpu-smoke-executor.ts :216, :349; webgpu-smoke-readbacks.ts :982 |
| 4 | Lane | tests/uniform-surface-extraction-dawn.test.ts |
| 5 | Overlay: texture binding 25 `sliceCoarsePhi`, lattice from the base under mixed presentation | grid-overlay-levelset-volume.wgsl.ts :61-66; webgpu-grid-overlay.ts :2112, :2197-2202, :2334 |
| 6 | Composite contact: bindings 18 (detail) and 19 (base), `contactFluidValue` = `0.5 - umSampleVertex/h` under mixed presentation, one evaluation site in `refineContactSurface`; the n^3 cell field is no longer bound there (1^3 fallback) | webgpu-water-pipeline.ts :1230-1232, :1298-1345, :2677-2680 |
| | Benchmark tool drives the Geometric presentation; compute-only classify matrix by default, `--render` for the old presentation timing, `--wait` for the lease | tools/benchmark-water-extraction-dawn.ts |

### Where the plan was wrong

| Plan | Reality |
|---|---|
| `extractTilesMain`: one thread per window loops the window's 64 cubes | Measured 3-10x slower wherever a window has work (64^3 partial 15.5 vs 1.57 ms, 128^3 Full 8.9 vs 4.85 ms): serial loops over a divergent body are expensive on Metal. Not kept |
| Fallback: today's dispatch gated by a workgroup variable and `workgroupUniformLoad`, "identical cost at full detail" | 128^3 Full 6.03 vs 4.85 ms. Not kept |
| | Kept: `collectWindowsMain` (one lane per window, the plan's exact test) appends the windows it cannot prove empty to a list; `extractWindowsMain` is the same (t+1)^3 workgroup launch as the full scan, workgroup w classifying listed window w with its 64 lanes and returning when w is past the count. A fixed direct launch, no indirect dispatch, no readback. Several windows per workgroup (striding the list) was slower at Full and partial and was dropped. A per-lane variant of the test (each lane tests its own cube) was correct but slower than the list at zero and partial detail |
| `umLoadVertex` branches on 4-alignment | That branch at every load cost about 10% of classify at Full (5.37 vs 4.85 ms). The split sampler names two loaders and each site calls the right one: an h tile's vertices from the detail field, a 4h owner's corners from the base; the regular sample branches once on the owner width. `umLoadVertex` with the branch remains for the few generic sites |
| | Added: in an all-4h stencil `umVertexValue` skips its incident-tile walk (presentation text only; the solver's sampler text is byte-identical, hash-checked) |
| `setVolume` rebuilds the bind groups when the source changes | `setVolume` runs only when the volume texture changes. The water pipeline gained `setDenseLevelSetVolumeSource`, called each frame |
| "the pipeline's 1^3 fallback" | No 3D r32float fallback existed; `fallbackField` was added |
| Publish after uploads and "voxel edits" as separate sites | Phi's writers are the surface advect, the surface-volume apply and the relayout remap, all inside an advance or a relayout; the one upload is at construction, before the t=0 publish. Three publish sites cover them: frame tail, relayout, t=0 |
| Overlay: texture count 10 -> 11, storage budget untouched | As planned. The extraction layout, however, is now at 10 of 10 storage buffers (the window list), and the composite fragment stage is at 15 sampled textures of 16 |
| Lane: one 64^3 scene, count-only totals | Two cases (32^3, 64^3), one live solver each through Requested/no region, a Fine region, Full, Requested again; base texel == phi[4g] bitwise; count-only totals, sorted worklists and sorted triangle bits equal between the full scan and the window scan; `detailVertexPhi` named exactly while capacity > 0 |
| "about 26x fewer loads at 256^3" | Not measured at 256^3. Measured classify at 128^3 zero detail: 5.11 -> 0.72 ms |

### Measurements

One device, one process, compute only (no raster). Median of 40 passes, ms; timestamp quantum
0.066 ms. `legacy` is the sampler and scan before this work (every vertex from the h field,
full lattice scan), rebuilt from the shipped shader text; its mesh is bit-identical to the
production mesh in every phase. Classify includes the prepare pass.

| Scene, detail | classify legacy | classify full scan, new sampler | classify windows (kept) | polygonise legacy | polygonise kept |
|---|---|---|---|---|---|
| 64^3 zero | 0.66 | 0.59 | 0.20 | 2.88 | 2.56 |
| 64^3 Fine region | 1.70 | 1.51 | 1.05 | 5.31 | 4.65 |
| 64^3 Full | 0.66 | 0.66 | 0.66 | 0.46 | 0.46 |
| 128^3 zero | 5.11 | 4.33 | 0.72 | 8.85 | 7.27 |
| 128^3 Fine region | 8.13 | 7.01 | 4.06 | 12.12 | 9.96 |
| 128^3 Full | 5.37 | 5.11 | 5.31 | 1.84 | 1.84 |

Full does not regress against today (64^3 equal; 128^3 5.31 vs 5.37, p25 4.85 vs 5.18). The
128^3 Full row is within run-to-run noise across arms (an earlier run of the same row read
5.18 legacy, 5.24 windows, 6.55 full scan).

Phi bound by the water extraction at zero detail (Requested or Full policy, no h tile):

| | 64^3 | 128^3 | 256^3 (computed) |
|---|---|---|---|
| before: h vertex phi + cell field | 2,147,076 | 16,975,364 | 135,007,236 |
| after: 4h base (+ one 4-byte dummy) | 19,652 | 143,748 | 1,098,500 |
| renderer's window list (new, storage) | 19,656 | 143,752 | 1,098,504 |

With any fine capacity under identity placement the detail binding is the solver's whole
(n+1)^3 field (1,098,500 / 8,586,756 / 67,898,372), as before.

### What the measurements say next

At zero and partial detail the extraction is now dominated by polygonise, not classify: 7.3 ms
at 128^3 zero against 1.8 ms at Full for fewer vertices (about 24 ns against 3 ns per vertex).
The cost is `uniformPhiNormal`: 27 `umVertexValue` calls per vertex, each an eight-load
interpolation in a 4h tile where at Full it is one load. That predates this work (legacy 8.85)
but it is the remaining gap of the all-4h endpoint in the renderer. Leads, in order: a normal
for 4h owners taken from the gradient of the tile's own trilinear interpolant (8 loads, exact
for the field being contoured) or one shared 27-load block per cube rather than per vertex;
then Option C (H-pitch triangles).

Done the same day by the second lead, exactly: see "Polygonise at zero and partial detail".

### Not verified

- Nothing was drawn. The composite and overlay shaders compile and validate on Dawn/Metal, and
  the overlay's level-set module returned the right phi through a compute probe
  (`uniform-volume-dawn`), but no raster pass ran: overlay layers actually building and drawing
  in the app, and the visual mesh, need the app.
- Composite contact refinement with a rigid body in the contact band. Its behaviour changed
  for Uniform Geometric (it read centre phi as occupancy before; now it reads the level set),
  and `refineContactSurface` was restructured for every dense method (same arithmetic, one
  evaluation site).
- The per-frame rebind when capacity crosses 0 in the app (the lane crosses it both ways on one
  solver, through the QA entry point, not through the renderer's frame loop).
- Packed (QA) storage placement was not run on Dawn; it parses under naga.
- 256^3 was not measured.
- The overlay still binds n^3 volume, velocity and open-fraction fields (plan step 7), and falls
  back to the cell field for its phi binding at zero capacity; only its phi reads moved.

## Polygonise at zero and partial detail, 4 October 2026

Follow-up to the record above. Goal: polygonise at zero detail no slower per vertex than at
Full, the same vertex positions, normals continuous across tile faces and h/4h seams, Full not
slower than today. Result: the normal is unchanged bit for bit (so continuity is today's) and
polygonise is 8-10x faster at zero detail and about 4x at partial; Full is 8-14% faster. Zero
detail lands at today's Full cost per vertex, not below the new Full cost (table below).
Everything here was run: CPU emulation, CPU preflight, two one-process benchmark runs and the
lane on Dawn/Metal (three Dawn processes).

### What was built

| | Where |
|---|---|
| The normal's samples are classified once per cube, not per sample. A cube at base b samples lattice vertices b-2..b+2 only; every tile around them is the home tile b/4 or a neighbour, and the home tile's stencil word (one load) already says which are h. Three classes: all h (direct detail loads), base block all 4h, mixed | `uniformNormalCube`, `lib/core/webgpu-water-pipeline.ts:567` |
| A 4h-owned sample is the trilinear interpolant of 27 base vertices loaded once per cube: `umVertexFrom4`'s arithmetic, term for term, as two vec4 products | `uniformNormalBase`, `:595` |
| In a mixed cube a sample off the tile corners is 4h-owned exactly when a tile around it is 4h (read from the stencil word, no load); otherwise it is stored. A tile corner stays `umVertexValue` (it needs its own tile's stencil) | `uniformNormalWide`, `:616`; the sample site in `uniformPhiNormal`, `:649`-`:660` |
| A crossing is evaluated once per cube and reused by the cube's triangles, same emission order | `shareCubeVertices`, `polygoniseCube`, `:712` |
| Both are overrides, on by default; off together is the previous polygonise, which the lane and the benchmark use as the reference | `uniformNormalGather`, `:559`; `shareCubeVertices`, `:712` |
| Lane: the shipped mesh against the reference polygonise in every phase: position bits per triangle and normal bits per vertex, asserted separately; vertices well formed; cube classes counted on the host and required (all 4h at zero detail, 4h and mixed with a region, all h at Full) | `tests/uniform-surface-extraction-dawn.test.ts:58`-`:110`, `:141`, `:190`-`:202` |
| Benchmark: polygonise arms (`legacy`, `sample`, `sample+share`, `gather`, `gather+share`, `floor+share`), mesh comparison against `sample` (positions, normals, largest angle), median and trimmed mean per arm | `tools/benchmark-water-extraction-dawn.ts:85`-`:110` |

Why it is exact. Off the tile corners, `umVertexValue` returns the stored vertex when every
tile whose closure holds it is h, and otherwise `umVertexFrom4` in the coarsest incident
owner. Any 4h tile around the vertex gives the same eight terms (weights off the vertex's own
face or edge are zero, and the weights are exact dyadic products), in slots related by axis
flips, and `umVertexSum8`'s pairing is invariant under those flips; so the value can be formed
in whichever tile of the cube's own base block holds the vertex. A CPU emulation in float32
against a port of `umVertexValue` (split loaders, unresolved), on random h/4h layouts with
independent base and detail values, agrees on all 32.9 M samples in both the scalar and the
vec4 form. On
Dawn/Metal the benchmark found 0 differing position words and 0 differing normal words against
the reference in all six rows, and the lane asserts the same on the final code at 32^3 and
64^3 through zero, Fine region, Full and zero again.

### Measurements

Same tool and matrix as above: one device, one process, compute only, arms interleaved and
rotated in one encoder, 40 passes per arm. Median ms, then ns per emitted vertex. `legacy` is
the polygonise before the 4h base (today's normal on the old sampler), `reference` the
previous polygonise on today's sampler, `kept` is production, `floor` is the kept pass with no
nodal normal at all (the contour normal): what the pass costs besides the normal.

| Scene, detail (vertices) | legacy | reference | kept | floor | kept against legacy |
|---|---|---|---|---|---|
| 64^3 zero (72,306) | 2.62 / 36.3 | 2.49 / 34.4 | 0.26 / 3.6 | 0.13 / 1.8 | 10.0x |
| 64^3 Fine region (95,274) | 4.92 / 51.6 | 4.13 / 43.3 | 1.31 / 13.8 | 0.59 / 6.2 | 3.7x |
| 64^3 Full (118,146) | 0.46 / 3.9 | 0.46 / 3.9 | 0.39 / 3.3 | 0.20 / 1.7 | 1.17x |
| 128^3 zero (297,594) | 7.73 / 26.0 | 7.01 / 23.6 | 0.92 / 3.1 | 0.39 / 1.3 | 8.4x |
| 128^3 Fine region (390,126) | 10.22 / 26.2 | 9.11 / 23.4 | 2.16 / 5.5 | 0.92 / 2.4 | 4.7x |
| 128^3 Full (561,522) | 1.64 / 2.9 | 1.77 / 3.2 | 1.51 / 2.7 | 0.79 / 1.4 | 1.09x |

The timestamp period (0.066 ms) is 0.9 ns per vertex at 64^3 zero and 0.22 at 128^3 zero, so
the 64^3 per-vertex figures are coarse; the trimmed means are 3.97 (64^3 zero), 3.52 (64^3
Full), 2.99 (128^3 zero), 2.63 (128^3 Full).

Against the goal:

- Zero detail per vertex against today's Full per vertex: 3.6 against 3.9 at 64^3, 3.1 against
  2.9 at 128^3 (one timestamp period apart). Met to within the measurement at 64^3, 5% short at
  128^3.
- Zero detail against the new Full: 3.6 against 3.3 and 3.1 against 2.7. Not met: Full got
  faster too. The pass besides the normal costs the same per vertex at zero and Full (floor
  1.3 against 1.4 at 128^3); the normal itself is 1.8 ns per vertex in 4h cubes (27
  interpolations from the cube's base block) against 1.3 in h cubes (27 direct loads).
- In absolute terms zero-detail polygonise is now cheaper than Full (0.92 against 1.51 ms at
  128^3), and the whole extraction at 128^3 zero detail is 1.57 ms (classify 0.66 + polygonise
  0.92) against 12.5 ms before the base and 7.7 ms after the window scan alone.
- Full does not regress: 0.39 against 0.46 and 1.51 against 1.64 ms.
- No normal changed, so there is no angle distribution to report and nothing new to look for
  in the app.

Arms measured and not kept (all bit-identical to the reference; 128^3 zero / Fine region /
Full, ms):

| Arm | | Why not |
|---|---|---|
| A 5^3 block of vertex values filled once per cube (125 interpolations), all-4h cubes only | 1.57 / 6.03 / 2.03 without sharing, 1.44 / 4.19 / 1.51 with | Still 1.6x the kept arm at zero detail (the fill is 125 interpolations a cube whatever the cube needs), and without sharing h cubes at Full ran slower with it compiled in (2.03 against 1.77) |
| The same block for every cube, by `umVertexValue` | 1.38 / 3.80 / 1.70 with sharing | 125 calls a cube is more than the samples it replaces |
| Classified samples, scalar interpolation loop | 1.44 / 2.75 / 1.51 | The vec4 form is 0.92 at zero detail |
| Classified, all-4h cubes only; all-4h and all-h cubes only | 0.85 / 3.93 / 1.57; 0.85 / 3.67 / 1.31 | Mixed cubes stay on the per-sample walk: 1.5 ms more at partial detail. The mixed class costs the others 0.07 ms at zero and 0.2 ms at Full, which is the price paid here for partial detail |
| Classified, no sharing | 1.18 / 3.15 / 2.03 | Sharing wins at every row |
| Sharing only, reference samples | 4.39 / 5.64 / 1.44 | |

Candidate (a), a gradient at the tile's H vertices blended to the mesh vertex, was not built:
the exact arm reached the same cost class with no visible change.

### What remains

- 4h against h per sample. Closing the last 0.4 ns per vertex means cheaper interpolation, not
  fewer loads: per-axis terms hoisted out of the 27-sample loop, and one loop per cube class
  instead of a class test per sample (which would also return the 0.2 ms the mixed class costs
  Full). Both keep the arithmetic and should stay exact, but Metal's fusion of multiply-adds
  depends on kernel shape, so each needs a measured run with the bit comparison.
- The Gaussian weight is 27 `exp` calls per crossing at every detail level. It is separable
  (9 calls), at the cost of ulp-level changes to every normal.
- Partial detail is now bounded by cube corners, not normals: the floor arm is 2.4 ns per
  vertex at 128^3 partial against 1.3-1.4 at zero and Full, and classify is 3.8 ms against 0.66
  at zero. Read from the code, not decomposed by measurement: the extra is `umSampleVertex`
  for an h cell inside a mixed stencil (eight `umVertexValue` walks per cell), in both passes;
  classify also scans every window that touches an h tile. The same stencil-word
  classification applies to those cell samples; it is the largest remaining extraction cost
  with a region drawn.
- At Full, classify (4.7 ms at 128^3) is three times polygonise: the h lattice has no base to
  prove windows empty from.

### Not verified

- Nothing was drawn (compute-only runs). The mesh is bit-identical, so there is nothing new
  to see, but the raster path was not exercised.
- The kept shader is the measured arm with the unkept arms' constant-folded branches deleted.
  The lane ran on that final text (exact equality holds); its timing was not re-measured.
- `shareCubeVertices` also applies to the sparse and adaptive extraction, which share
  `polygoniseCube`; those were not run.
- The lane's cube-class counts were asserted, not captured (the runner prints nothing on a
  pass).
- 256^3 was not measured.

## Classify at Full and cube corners in mixed stencils, 4 October 2026

Follow-up to the two records above, on their "what remains" lists. Result: classify at 128^3
is 0.92 ms at Full (4.65 before), 0.79 ms with a Fine region (3.87) and 0.52 ms at zero detail
(0.66); polygonise with a Fine region is 1.70 ms (2.36) and at Full 1.38 ms (1.64). The whole
extraction at 128^3 is 2.29 ms at Full and 2.49 ms with a Fine region, against 6.3 and 6.2 ms
before this task. Worklists, counts and the mesh (position and normal bits) are unchanged.
Everything here was run: CPU emulation, CPU preflight, three one-process benchmark runs and
the lane on Dawn/Metal (four Dawn processes).

### What was built

| | Where |
|---|---|
| The listed-window scan shares a window's samples in workgroup memory. A window's 64 cubes read 125 lattice values, whose cells read 216 vertices and, where a tile is 4h, 27 base vertices; each is formed once per window instead of about nineteen times. Lane 0 classifies the window from the stencil words of its eight tiles (every tile's stencil all h / every tile 4h / mixed); the lanes then fill vertices, cells and cubes, a barrier between | `extractWindowsMain`, `lib/core/webgpu-water-pipeline.ts:1160`; `uniformWindowHeader`, `:1028`; `uniformWindowVertices`, `:1068`; `uniformWindowCells`, `:1129` |
| An all-h window is straight-line: four direct detail loads a lane, and a cell from its eight stored vertices in the regular sample's arithmetic at a cell centre | `uniformWindowCellsFine`, `:1106` |
| A listed window whose every nodal input is on one side of the contour stops after its vertices: the window list's two quiet tests, applied to the values the vertex phase has just formed (every cell of the window is a convex combination of them), for h and mixed windows too | `windowSigns`, `uniformWindowSign`, `:1023`; the test in `extractWindowsMain` |
| A cell's value by kind (4h cell from its tile's corners; h cell of an all-h stencil from its stored vertices; h cell with a 4h neighbour from `umVertexValue` of its vertices), in `umSampleVertex`'s expressions | `uniformCellValue`, `:684`; `uniformCornerValue`, `:663` |
| Polygonise: a cube's nodal values are formed once. A mixed cube asked `umVertexValue` for the same vertices repeatedly through the incident-tile walk (64 reads of 27 vertices for its corners, 27 a crossing of 125 for its normals); each is now formed at first use and kept for the cube. The corner cells keep `umSampleVertex`'s own text: only `umVertexValue`'s answer comes from the memo, through the sampler's existing `cacheLookup` hook | `uniformMemoVertex`, `:700`; `uniformCubeVertex`, `:712`; `SURFACE_EXTRACTION_VERTEX_CACHE`, `:381`; `polygoniseMain`, `:895` |
| The normal's 27 samples by one literal loop per cube class, generated from one template | `uniformNormalLoopWGSL`, `:368`; `uniformPhiNormal`, `:721` |
| Both polygonise changes are overrides, on by default; off (with the two earlier ones) is the reference polygonise the lane and the benchmark compare against | `uniformCubeMemo`, `:586`; `uniformNormalLoops`, `:589` |
| The presentation sampler takes an optional `vertexCache` (the sampling source's `cacheLookup`). Empty, its text is unchanged: the solver's sampler and the overlay's and the contact pass's presentation text hash the same before and after | `lib/methods/uniform/uniform-mixed-presentation.wgsl.ts:30` |
| Benchmark: scans `legacy`, `before`, `shipped`; polygonisers `legacy`, `sample`, `before`, `kept`, `floor`, `separable`; a `shipped` arm that pairs the shipped scan with the shipped polygonise. The `before` arms are built from the shipped text with the cube hook removed | `tools/benchmark-water-extraction-dawn.ts:81`-`:156` |

Exactness. The full scan (`extractMain`, every cube through `latticeValue`) stays the
reference for classify, and the lane holds its count and sorted worklist equal to the window
scan's in every phase. The quiet test is the list's own argument: a cell is `0.5 - s/h` with
`s` in the hull of its inputs, so inputs all above the 1e-6 h margin give no corner at or
above 0.5, and inputs all at or below zero with no corner in the wall halo give none below
it. A CPU emulation on structured fields and six layouts (729 windows each) found no quiet
window holding a crossing and no load of a vertex no tile stores. The memo returns what
`uniformNormalCube`'s reduction returns for the normal, which the earlier emulation covers
(every vertex of b-2..b+2 against a port of `umVertexValue`). On Dawn/Metal: every scan's
count-only total agrees in all six rows of all three runs; every kept and unkept polygonise
arm of runs 2 and 3 has 0 differing position words and 0 differing normal words against the
reference; the mesh behind the shipped scan is bit-identical to the reference's; the lane
passed on the final text.

One attempt was not exact and is not in the tree: forming a cube's eight corner values from
the classification in new code (same operations, different kernel shape). On Metal 4,698 to
34,866 triangles a row came out with different position bits (counts equal), and it was
slower at Full. Keeping `umSampleVertex`'s text and memoising only `umVertexValue` is what
made the corners both exact and cheaper.

### Measurements

Same tool and method as above (one device, one process, compute only, arms interleaved and
rotated in one encoder, 40 passes per arm, median ms; timestamp period 0.066 ms). `legacy` is
before the 4h base, `before` is the tree before this task, re-measured in the same run.

Classify + prepare:

| Scene, detail | legacy | before | kept | kept against before |
|---|---|---|---|---|
| 64^3 zero | 0.66 | 0.20 | 0.13 | 1.5x |
| 64^3 Fine region | 1.57 | 0.98 | 0.20 | 5.0x |
| 64^3 Full | 0.66 | 0.66 | 0.20 | 3.3x |
| 128^3 zero | 4.78 | 0.66 | 0.52 | 1.25x |
| 128^3 Fine region | 7.54 | 3.87 | 0.79 | 4.9x |
| 128^3 Full | 4.85 | 4.65 | 0.92 | 5.1x |

Polygonise, ms / ns per emitted vertex. `kept` is the shipped polygonise behind the shipped
scan; `floor` is the kept pass with no nodal normal:

| Scene, detail (vertices) | legacy | before | kept | floor |
|---|---|---|---|---|
| 64^3 zero (72,306) | 2.62 / 36.3 | 0.39 / 5.4 | 0.33 / 4.5 | 0.13 / 1.8 |
| 64^3 Fine region (95,274) | 4.78 / 50.2 | 1.38 / 14.4 | 0.79 / 8.3 | 0.33 / 3.4 |
| 64^3 Full (118,146) | 0.46 / 3.9 | 0.46 / 3.9 | 0.39 / 3.3 | 0.20 / 1.7 |
| 128^3 zero (297,594) | 7.73 / 26.0 | 0.92 / 3.1 | 0.85 / 2.9 | 0.39 / 1.3 |
| 128^3 Fine region (390,126) | 10.22 / 26.2 | 2.36 / 6.0 | 1.70 / 4.4 | 0.72 / 1.8 |
| 128^3 Full (561,522) | 1.64 / 2.9 | 1.64 / 2.9 | 1.38 / 2.5 | 0.79 / 1.4 |

Whole extraction (classify + prepare + polygonise), each pair run together:

| Scene, detail | legacy | before | kept |
|---|---|---|---|
| 64^3 zero | 3.28 | 0.59 | 0.46 |
| 64^3 Fine region | 6.36 | 2.36 | 0.98 |
| 64^3 Full | 1.11 | 1.11 | 0.59 |
| 128^3 zero | 12.52 | 1.57 | 1.38 |
| 128^3 Fine region | 17.76 | 6.23 | 2.49 |
| 128^3 Full | 6.49 | 6.29 | 2.29 |

Step by step at 128^3 (zero / Fine region / Full, ms), from run 2. Full does not regress at
any step:

| Classify step | |
|---|---|
| before | 0.66 / 3.87 / 4.65 |
| window samples shared in workgroup memory | 0.52 / 1.31 / 2.88 |
| + all-h window straight-line | 0.52 / 0.92 / 1.05 |
| + quiet listed windows stop after their vertices | 0.52 / 0.79 / 0.98 |

| Polygonise step (behind the `before` scan) | |
|---|---|
| before | 0.92 / 2.29 / 1.64 |
| + cube memo | 0.98 / 1.90 / 1.64 |
| + one loop per cube class (kept) | 0.85 / 1.77 / 1.44 |
| one loop per class without the memo | 0.85 / 2.16 / 1.38 |

Where the Full classify cost was: not in the list (every window was listed, and still is) but
in every cube forming its own eight corners through the owner-aware sampler: 64 vertex loads
and eight owner lookups a cube for values its neighbours also form. Shared, a window is 216
loads and 125 cells for 64 cubes. The same sharing removes the partial-detail cost, where the
per-cube samples of h cells in mixed stencils were eight `umVertexValue` walks each.

Measured and not kept (all exact unless said; 128^3 zero / Fine region / Full, ms):

| Arm | | Why not |
|---|---|---|
| Cube corner values re-derived from the classification | polygonise 0.98 / 1.84 / 1.70 | Not bit-identical on Metal (above), and slower at Full |
| The all-4h class's per-axis terms hoisted out of the 27-sample loop | polygonise 1.38 / 2.10 / 1.44 | Exact, and 0.5 ms slower at zero detail than the plain per-class loop (0.85) |
| The normal's samples gathered into an array by a class loop, then accumulated (opaque bounds) | polygonise 1.57 / 2.88 / 2.36 | Exact, slower everywhere |
| Each cube classifying itself in the window scan (no sharing) | classify 1.38 / 3.67 / 8.59 | Slower at Full than before |
| The list's quiet test extended to all-h windows (216 stored vertices), leaving early or reading all | whole extraction 1.31 / 2.62 / 2.16 and 1.38 / 2.69 / 2.29, against kept 1.38 / 2.49 / 2.29 | Takes 0.13 ms off classify at Full, but polygonise behind it is 0.2 ms slower with a Fine region (1.90 against 1.70) and at 64^3 (0.92 against 0.79): the worklist's order changes |
| The list as a word per window in lattice order, with and without that test | whole extraction 1.31 / 2.82 / 2.23 and 1.31 / 2.69 / 2.10 | Same trade: polygonise 1.97 and 1.90 with a Fine region |
| Separable Gaussian weight (nine `exp` a crossing, per-axis factors indexed from an array) | polygonise 1.11 / 2.23 / 1.64 against kept 0.85 / 1.77 / 1.44 | It saves nothing in this form: 0.2 to 0.5 ms slower. It changes the normal of 164,264 / 211,928 / 338,701 vertices, by at most 0.04 degrees. Not landed, as agreed |

### What remains

- Polygonise time depends on the order of the worklist. The same cubes through the same
  kernel took 1.70, 1.90 and 1.97 ms at 128^3 with a Fine region behind three different
  window lists. The cause was not established (the kept list happens to be the fastest of
  those tried). An ordered or class-partitioned worklist is an untried lever, and it is what
  stands between the list's all-h quiet test and a further 0.13 ms at Full.
- At Full every window is still listed, and its 216 vertices loaded, before the scan can stop
  (0.92 ms at 128^3 against 0.52 at zero detail). Rejecting an h window without reading its
  vertices needs a per-tile phi range or sign summary from the solver; none is published.
- The zero-detail classify (0.52 ms at 128^3) is the fixed launch of one workgroup per window
  slot plus the list pass; read from the code, not decomposed by measurement.
- With a Fine region, polygonise is 4.4 ns per vertex against 2.5 at Full and 2.9 at zero
  detail: mixed cubes pay the memo's first-use walk and the base interpolation.
- The normal is still 27 `exp` a crossing; the separable form did not pay as built.

### Not verified

- Nothing was drawn (compute-only runs). The mesh is bit-identical, so there is nothing new
  to see, but the raster path was not exercised.
- The kept shader is run 3's `append` scan and `kept` polygonise with the unkept variants'
  constant-folded branches deleted (the list is back to its text before this task). The lane
  ran on that final text; its timing was not re-measured, and the benchmark tool's final arm
  list (the unkept scans removed) type-checks and its shaders validate but was not run on
  Dawn.
- The benchmark runs the solver's default placement. The window scan's and the memo's direct
  detail loads go through `udrStoredVertex`, so a packed placement costs a directory read a
  load; that was not measured.
- The sparse and adaptive extraction share `polygoniseCube` and `uniformPhiNormal`; they never
  classify a cube (the memo and the class loops stay off for them) and were not run.
- 256^3 was not measured.
