# Uniform mixed: sparse tile residency

Status: design, 2026-09-28 (fluid-a1). Owner: fluid-a1, on Peter's approval.

Code is at HEAD cf051b46 plus WP2 of `uniform-mixed-gpu-resident-advance-plan.md`. The storage figures come from a read-only audit (CPU scripts that repeat the allocation formulas, and scene statistics from the real scene documents). No GPU numbers here are new.

## 1. Goal

Make memory and GPU time follow the **liquid**, not the container. A garden-sized domain, where water can travel anywhere but usually sits in a pond, should cost about the pond plus its reach margin. A hose jet crossing the garden should pull storage in ahead of itself.

Standing rules this design must obey:
- GPU resident: no blocking readback on the advance path.
- Fail fast: overflow is a loud fatal error, never growth or a fallback.
- Layout-general: everything survives dynamic coarsening, with no scene special cases.
- Structural cuts before cheaper launches.
- Launch count fixed and small: empty indirect dispatches cost about 12 µs each on Dawn/Metal.

## 2. Where the memory and time go today

hero-garden-hose is 144×96×96: N = 1.33 M cells, T = 20,736 tiles of 4³ cells. The mixed solver allocates about **423 MB (318 B/cell)**. fig-12 at 128³ comes to about 665 MB for the solver (a peer measured about 717 MB overall).

| Addressing | What | MB (hose) | Share |
|---|---|---:|---:|
| Lattice position (3D textures; a 4h owner uses its origin texel) | 5 rgba32f velocity-like fields, 8 r32f fields, 2 vertex phi, lattice-indexed extension ping-pong in the arena | ≈270 | 64 % |
| Owner index, sized to all-h capacity | transport stencils/limbs/sums, mixed pressure memory, surface-volume scratch | ≈90 | 21 % |
| Tile/band slot, sized to "every tile is h" | pressure band rows (84 B/cell with solids), iterate, aggregates | ≈124 | – |
| Dead on the mixed path | the n/2 extension hierarchy level (15.9 MB); transportA (22 MB, unconfirmed) | ≈16–38 | – |

Structural facts that shape every option:
1. **Owner numbering is already compact.** h owners come first, 64 per tile, then one owner per 4h tile (`uniform-mixed-layout.ts:159-170`). The fields are still dense lattice textures, and a 4h owner reads and writes only `umOrigin` (`uniform-mixed-topology.wgsl.ts:139`). So **63 of every 64 texels in a 4h tile are dead.**
2. **Capacity is reserved for all-h.** `UniformMixedFrame` throws unless built on the all-fine layout (`uniform-mixed-frame.ts:182`). The transport slices, band capacity (`uniform-pressure-band.ts:121`) and pressure memory (`uniform-mixed-pressure-memory.ts:15-30`) are all sized for T·64 owners, although pressure level 0 is always all-4h.
3. **No hardware filtering.** Every field access on the mixed path is a manual `textureLoad`/`textureStore` tap (there are 0 `textureSample` calls in `lib/methods/uniform`). Every access can therefore go through an indirection.
4. **The mixed tile word is already a GPU-maintained directory.** `umTileAt` is a direct linear index, and the word holds the h/4h bit and a compact base. The census and builder rebuild it on the GPU at every relayout. The general samplers already load it on every tap.
5. **In the garden, solids set the h population, not water.** 6,992 of 20,736 tiles (34 %, 447 k h owners) are forced h by solid promotion even when completely dry. The filled pond has 492 wet tiles, or 1,214 including a one-tile halo. The band certificate (`uniform-pressure-band.ts:376-377`) makes *any* cut tile held at 4h fatal, wet or dry. Every all-owner pass visits those 447 k dry owners on every frame.
6. **The existing paged atlas is slow.** It is used by the native path but bypassed by the mixed kernels. It measured 3–4× slower: long dam 46 → 119 ms per advance, Full-Cycle 6.4 → 30.3 ms. The cause is a per-tap catalogue lookup plus div/mod against the atlas. It reached parity only once neighbours were compiled to native coordinates at generation time. **Design rule inherited from that work:** no catalogue lookup and no div/mod in repeated stencil taps. Resolve placement and neighbours once per generation.

Device ceilings today:
- The band rows are the tightest binding (84 B/cell ≤ maxStorageBufferBindingSize). At the 128 MiB default the largest container is about 1.6 M cells: 128³ does not fit.
- At 4 GiB bindings, total VRAM (318 B/cell) is the real ceiling, at about 20–30 M cells on a 16 GB Mac.

## 3. Requirements for any residency scheme

- **R1: Per-tap cost.** One extra dependent load per *sample* (per 4³ block), never per tap. The atlas result above is the counter-example.
- **R2: Allocation on the GPU.** Allocation and free happen on the GPU as part of the census/builder generation. The host learns only counts, lagged, for diagnostics.
- **R3: Warm pool.** Size pools once, from a scene budget. Overflow sets a receipt word and the frame fails naming it, like the band's existing overflow word (receipt word 23).
- **R4: Relayout.** Relayout stays a GPU gather from old storage to new, with no host arrays on the critical path. This converges with WP4 of the GPU-resident advance plan.
- **R5: Absent semantics.** An absent region reads as air: phi = +far, u = 0, V = 0, p = 0 (Dirichlet). This is correct only where a certified halo guarantees no liquid is adjacent.
- **R6: Departure certificate.** Semi-Lagrangian departures, extension and redistance must never need data from an absent region. This must be certified on the GPU and fatal when violated.
- **R7: Readers outside the solver.** Water extraction, normals, overlays, particles, the harness and tests read either through a residency-aware loader or from an explicit dense publish.

## 4. Approaches considered

### A. Tile-word directory + brick pool (bitmasked pointer grid)

This is the SPGrid / Taichi `pointer→bitmasked→dense` family, specialised to our fixed 4³ tile.

- **Directory:** keep a flat dense T-entry directory, the existing tile word. Extend the word with a state (absent / 4h / h) and a pool slot.
  - h tiles own a **brick**: 4³ cells (plus the per-cell positive MAC faces we already store) in a brick pool, addressed as `slot·64 + local`.
  - 4h tiles own one **coarse texel** in T-resolution textures, 1/64 of today's size. The 4h sampling cache (`uniform-mixed-frame.ts:191`) is already this shape.
  - Absent tiles own nothing.
- **Neighbours:** the builder compiles a per-tile 26-neighbour slot table (or the 6 face neighbours plus a rule for edges and corners) when it builds a generation. Stencil kernels already run one workgroup per tile, so each workgroup loads its own slot and its neighbours' slots into workgroup memory once. Taps within the tile and its apron then cost nothing extra (R1).
- **Allocation:** a GPU free-list stack (atomic pop and push). The builder's `scatter` (`uniform-mixed-layout-builder.ts:190-213`) pops slots instead of assigning scan ranks. Slots freed in generation g become reusable in g+1 (deferred reuse, as in `uniform-page-generation.ts`), so the remap can still gather from the old slots during the adopt.
- **Memory:** 64·H·(bytes per h cell) + C·(bytes per 4h owner) + about 3 B/cell of directory, where H = resident h tiles and C = resident 4h tiles.
- **Pros:**
  - It reuses the indirection we already maintain and pay for.
  - Owner-indexed code (transport, pressure) is already compact and needs only capacity changes.
  - It generalises to absent tiles with one more state.
  - Relayout is already a GPU remap.
- **Cons:**
  - About 25 files touch lattice-position taps directly (the fast paths `umRegularFine`, `regularTexture` and `unitTexture`, plus remap, transport and phi-resolve).
  - Vertex phi and face sharing across brick boundaries must be designed; see 5.3.
  - The dense directory is O(T) and fine up to about 10⁶–10⁷ tiles. Beyond that it wants a second level (see F).

### B. Paged atlas (existing `UniformTexturePages` paged mode)

- **How it works:** 16³/32³ pages placed in an atlas texture. `rewritePressureTextureCalls` rewrites every `textureLoad` into `fieldLoad(p)`, which runs `mgPageAddress`: a bounds check, a page lookup, then div/mod.
- **Pros:** it exists; the shader rewrite covers kernels mechanically.
- **Cons:**
  - It measured 3–4× slower, because the cost is paid on every tap (it violates R1).
  - Pages are 16–32 cells, far coarser than our 4³ tile, so the granularity is wrong for a narrow liquid band.
  - The mixed kernels bypass it entirely.
  - Parity came only after "compiling neighbours per generation", which is approach A in all but name.
- **Verdict:** rejected. Keep only its generation/transaction pattern as a reference.

### C. GPU spatial hash (voxel hashing)

This is the Nießner et al. voxel-hashing family: a hash table from tile coordinate to slot, with open addressing.

- **Pros:** memory is independent of the container's extent, so there is no O(T) directory at all. It suits unbounded worlds.
- **Cons:**
  - Probe chains on every lookup: variable latency and divergent loops, which is the cost we must avoid per tap.
  - Insertion under contention needs a lock-free protocol and a rebuild on overflow.
- **Better use:** our dense directory costs about 3 B/cell (80 MB for a 268 M-cell garden). Hashing pays only past about 10⁸ cells. Even then it belongs at the *upper* level of a two-level scheme (F), resolved once per generation into A's neighbour tables. It should never sit in the tap path.
- **Verdict:** not now; possible directory for F.

### D. Sliding dense window

One dense box that re-centres on the liquid's bounding box, shifting the data when the liquid approaches an edge.

- **Pros:**
  - Trivial addressing (a dense box with an offset); every kernel is unchanged except for the origin.
  - Works well for a single compact body of water.
- **Cons:**
  - Cost follows the bounding box, not the liquid. A pond plus a hose jet landing on the far side of the garden spans the garden, which is exactly our target case.
  - Shifting means whole-field copies plus boundary semantics at the window faces: a hidden domain wall where no wall exists.
- **Verdict:** rejected for the target. It would be acceptable only as a test-bench trick.

### E. Multi-patch dense boxes (block-structured AMR patches)

A handful of dense boxes clustered around bodies of liquid, with ghost exchange between patches.

- **Pros:** kernels stay dense inside each patch, and it is a well-understood technique (Berger–Colella).
- **Cons:**
  - Regridding (clustering, splitting, merging) on the GPU each frame is complex.
  - Ghost exchange between patches is a second seam system on top of our h/4h seams.
  - Patch count varies, so the launch count varies (or empty launches pile up).
  - Pressure must couple the patches across air, including the all-4h far field.
- **Verdict:** rejected. It duplicates the seam machinery the mixed layout already provides at tile granularity.

### F. Two-level directory (tree of bricks; VDB-lite)

- **How it works:** a coarse directory of super-tiles (for example 8³ tiles = 32³ cells). Each super-tile is absent (uniform air, or uniform solid) or points to a dense tile-word block, which then works exactly as in A.
- **Pros:** directory memory scales with the resident super-tiles, which is what very large or unbounded gardens need beyond about 10⁶–10⁷ tiles.
- **Cons:** another indirection at generation time (not per tap, if A's neighbour tables are still compiled per generation), and a more complex builder.
- **Relation to GVDB/NanoVDB:** full VDB is a deeper, variable tree with bit-masked nodes and per-level indirection. That suits read-mostly rendering (NanoVDB is effectively read-only). Rebuilding it every frame for simulation costs more than our fixed two-level shape.
- **Verdict:** future extension of A, when containers exceed about 10⁷ tiles. Design A's directory so a second level can be added without touching kernels.

### G. Hardware sparse (tiled) textures

- **Verdict:** not available. WebGPU exposes no sparse or tiled resources, so it is noted only for completeness.

### H. Dense coarse, sparse fine (hybrid of A without absent tiles)

- **How it works:** keep every 4h owner resident in dense T-resolution textures, which cost 1/64 of today's lattice fields over the whole container. Pool only the h bricks. Absent tiles are never introduced.
- **Pros:**
  - Removes R5/R6 entirely: no absent reads, no departure certificate beyond today's.
  - Pressure level 0 is already all-4h over the container, so it is unchanged.
  - Captures most of the memory win for containers up to about 10⁷ cells: 64× on dry lattice fields.
- **Cons:**
  - Time still scales with T for all-tile passes: census, all-4h pressure, directory scans. These are O(T), not O(64·T), so they are cheap but not flat.
- **Verdict:** **the recommended first compact-storage step** (phase 3 below). Absent tiles (phase 4) are needed only when containers outgrow it.

### Comparison

| | Per-tap cost | Memory scales with | GPU update | Fits dynamic coarsening | Kernel churn | Verdict |
|---|---|---|---|---|---|---|
| A tile directory + brick pool | ≈0 (per-workgroup slot table) | liquid + O(T) directory | free list in builder | native (same word) | high (≈25 files) | **target** |
| B paged atlas | high (lookup + div/mod per tap) | pages (16–32³) | exists | poorly (coarse pages) | low (auto-rewrite) | rejected |
| C spatial hash | medium (probe chains) | liquid only | lock-free insert | yes | high | directory level of F, later |
| D sliding window | 0 | liquid bounding box | whole-field shift | yes | low | rejected |
| E multi-patch | 0 inside patches | patches | regrid + ghosts | second seam system | very high | rejected |
| F two-level | ≈0 (resolved per generation) | liquid | two-level builder | yes | A + builder | later extension of A |
| G sparse textures | – | – | – | – | – | unavailable |
| H dense coarse + sparse fine | ≈0 | liquid (h) + T (4h) | free list for h only | native | A minus absent | **first compact step** |

## 5. Recommended design (A, delivered as H, then absent tiles)

### 5.1 Tile word

Today the word holds `width bit | compact base`. It becomes `state(2) | slot`:
- absent: no slot.
- 4h: slot = the tile's texel in the T-resolution fields, i.e. the tile index itself under H. Under phase 4 it becomes a pool slot.
- h: slot = brick index in the h pool.

The compact owner index (`UMOwner.index`) stays as it is for owner-indexed buffers (transport, pressure), which already need only capacity changes.

### 5.2 Storage

- **h pool:**
  - One brick = 4³ cells × (the fields that are per-cell at h).
  - Velocity is stored as positive MAC faces per cell (today's rgba xyz layout). A brick's negative faces on its low boundary belong to its neighbour.
  - Physical layout options:
    - (i) a 3D brick-atlas texture, with slots laid out in a 3D grid of 4³ blocks. The address is `slotOrigin + local`, with no div/mod once the slot origin is cached.
    - (ii) flat storage buffers `slot·64 + local`.
  - Prefer (ii) for new owner-indexed data and (i) where kernels already take textures, so the conversion stays mechanical. Decide per field by its readers.
- **4h fields:** T-resolution textures (n/4 per axis). Several already exist, e.g. the 4h velocity cache.
- **Vertex phi:** each h brick stores its own 5³ vertex record, so the shared face and edge vertices are duplicated. This follows the hanging-record precedent (`uniform-mixed-ownership.ts:219`). A **resolve pass** (phiResolve's job today) makes the duplicates agree after each writer. The cost is 125/64 ≈ 2× vertex storage, but only on h tiles, and it removes all cross-brick vertex lookups from taps.
- **Arena:** the transport slices and pressure memory become sized to the pool (64·H + C) instead of 64·T.

### 5.3 Addressing inside kernels (R1)

- Stencil kernels already run one workgroup per tile. At workgroup start they load the tile's slot plus a 27-entry neighbour slot table, compiled by the builder into the generation, into workgroup memory. All taps then address `neighbour[face] + local` with no directory load and no div/mod.
- Samplers that follow a departure point into an arbitrary tile (surface trace, momentum, general taps) already load the tile word per sample (`umVelocityTap1`, `umUnitTaps`, `umVertexValue`). They keep one directory load per sample, the same as today.
- The current direct `base+bit` fast paths that assume a dense lattice switch to the workgroup slot table.

### 5.4 Allocation, relayout, pools (R2–R4)

- **Census → builder:** the builder assigns states from the census as today, then:
  1. Pops pool slots for tiles becoming h, and for tiles becoming resident under phase 4.
  2. Writes the neighbour tables.
  3. Pushes the slots of tiles leaving h onto a **pending-free** list, which is merged into the free list only after the next adopt, so the remap can still gather from them.
- **Remap:** a gather from old storage (old word) to new storage (new word). Today's remap already does old→new per tile.
- **Pool size:** the budget comes from the scene: an explicit `residency.hTiles`, or derived from the initial liquid plus inflow capacity times a margin. It is fixed at construction.
  - Overflow: the builder sets a receipt word, the frame's receipt check fails as `Uniform mixed frame N: h pool needs X tiles, budget B`, and nothing grows.
  - The band budget becomes the same number: band rows exist only for h tiles.
- **Host:** no host arrays are needed on the critical path. That requires WP4 (a host-free adopt), because today `builder.read()` returns T tile words to the host every relayout. Until WP4, the host mirror stays lagged, as WP2 left it.

### 5.5 Absent tiles (phase 4)

- **Residency set** = the census reach closure around liquid (the existing departure/velocity-bound pyramid, BOUND_RADII up to 16 tiles), plus a one-tile halo, plus solid-coupled tiles within that reach. Everything else is absent.
- **Departure certificate (R6):** extend the plan certificate (`encodeCertificate(dt)`) so that every departure footprint and every extension or redistance stencil lies inside the resident set. Violation is fatal and names the tile.
- **Pressure:** absent tiles are Dirichlet air, and the halo guarantees no liquid owner neighbours an absent tile. Enclosed dry cavities are fine, because absent means "air at p=0", which is exactly what a vented dry region is. A *sealed* cavity is not vented, but it cannot hold liquid without being resident.
- **Phi at the boundary:** redistance clamps resident vertices bordering absent tiles to +far, consistent with the absent reading, so the band never sees a false gradient.
- **Launches:** fixed count, dispatched indirectly over the resident-h, resident-4h and band lists. Pressure level 0 runs over resident 4h tiles only.

### 5.6 Readers outside the solver (R7)

- **Residency-aware loaders:** water extraction (`umSampleVertex`), normals, the overlay density/velocity/divergence/cell paths, the level-set slice, diagnostics and the census are already owner-aware. They need only a residency-aware `umLoadVertex` / `umOwnerAt` / face loader. Absent reads +far, 0, 0.
- **Dense-assuming readers:**
  - The composite `liquidField`: it reads centre phi at every lattice cell. **This is already a latent bug:** the non-origin texels of 4h tiles are stale today.
  - The overlay `levelSetSample`.
  - Secondary particles.
  - The initial upload.
  - Harness readbacks and the tests' `readMixedTexture`.

  Port these to the loaders, or give them an explicit on-demand GPU dense publish, which is only for diagnostics and tests and never on the advance path. `UniformTexturePages.publication()` is precedent for such a publish.
- **Tests that deepEqual whole textures** (e.g. `uniform-pressure-local-visit-dawn.test.ts:85`) compare dead texels today. Under this design they compare owners.

## 6. Phases

Every phase is layout-general, measured on one target scene, and gated by the Uniform Dawn files (`npm run test:dawn -- uniform`) at the end.

**Target scenes:**
- **hero-garden-hose** for phases 1–3.
- **A new validation scene for phase 4, "garden-wide":** the same garden with its container grown to cover the whole set (2–4× volume) and the hose aimed across it. The existing garden containers are cut tight around the pond, so they cannot show absent-tile wins.

### Phase 0 prerequisites

- WP2 (receipt ring; landed, pending commit).
- WP4-lite: the adopt needs no T-word host readback. This can overlap phase 2.

### Phase 1: right-size (small, low risk; about 5 files)

1. Plan the pressure memory on `uniformMixedAllCoarseLayout(layout)` instead of the fine capacity layout. Level 0 is always all-4h.
2. Size the band to a budget B, not T. The fatal overflow receipt word already exists.
3. Delete the n/2 extension-hierarchy level on the mixed path; it is dead.
4. Verify that transportA and the native CM11a L0/L1 arena ranges are unused on the mixed path, then stop allocating them.

- **Expected:** about −100 MB of band and −16 MB of hierarchy at the hose; more at 128³.
- **Measure:** `executionInfo.allocatedBytes`; the maximum band tile count (receipt word 22) over a hose fill run, to set B; hose frame time flat.

### Phase 2: liquid-conditional solid promotion (medium; about 5 files; largest time win for gardens)

- Force h only on solid-coupled tiles within the census's predicted liquid reach plus the halo. The builder's static mask becomes a dynamic input.
- Relax the band certificate from "any cut tile at 4h is fatal" to "any cut tile with liquid, or in reach, is fatal".
- Dry cut tiles at 4h keep a correct all-4h solid record for pressure.
- **Expected:** garden h tiles 6,992 → about 1.1–1.5 k, i.e. about 350 k fewer owners in every all-owner pass.
- **Measure:** builder `tierCounts` at t=0 and through a hose fill; per-phase GPU ms (physicsTrace) on the hose; mass drift; the solid-parity and trough Dawn lanes.
- **Risk:** promotion must land before liquid touches a solid. It relies on the reach bound, which is already what dynamic coarsening trusts. Refinement near solids must rebuild the stencils exactly (the remap-follows-phi rule).
- **Landed 2026-09-28.** Rule, computed on the GPU in the census after `decide`:
  - A coupled tile is *active* when a wet tile (any non-air owner) or a band tile lies within one tile of it. `solidActive` computes this.
  - Every active tile and its 26 neighbours join the band bits. `solidPromote` does this.
  - The builder's static mask now holds only fine-only regions. The CPU path (`updateMixedRegions` in dynamic mode, and `promoteMixedDrop` through `uniformMixedLiquidSolidPromotion`) mirrors the rule.
  - A dry cut tile may therefore run 4h, and its h texels go stale. The solid record's per-tile `y` flag ("simulated at h") is rewritten by a widths pass after every relayout, and `umSolidCut` means cut ∧ h. The all-4h cut vote and cut-face flux treat a 4h cut tile as uncut.
  - The band certificate bit 1 is now "a liquid row (pressure-authority phi < 0) in a cut tile held at 4h".
- **Result (hero-garden-hose, dynamic, 90 pipelined steps, setTimeout(1) polling, one Dawn run per arm):**
  - Mean h tiles: 7,125 → 2,258. Solid promotion accounts for 1,404 of them.
  - Frame time: 18.63 → 15.80 ms.
  - Relayouts, certificate failures and GPU errors: 46 relayouts in both arms; no certificate failures; no GPU errors.
  - Dust: 1,163 → 1,381 cells, with mass 0.56 → 0.62 cells.
  - Uniform Dawn files: 10/14 pass, the same four reds as HEAD (long-dam-front, pond-rest, trough-contact, volume). trough-contact now trips its wall-phi assertion at frame 90 instead of 30.

### Phase 3: dense coarse + h brick pool, approach H (large; about 25 files; high risk)

- **Steps:**
  1. The tile word carries the brick slot.
  2. The builder pops and pushes a free list and compiles per-tile neighbour slot tables.
  3. Lattice fields split into a T-resolution 4h texture plus an h brick pool.
  4. Vertex phi moves to 5³ per-brick records plus a resolve.
  5. The arena and band are sized to the pool.
  6. Readers outside the solver use loaders or an on-demand dense publish.
- **Order inside the phase:**
  - Port field by field, starting with the owner-indexed ones (already compact, so capacity only).
  - Then volume/phase/target/centre phi (r32f, simple).
  - Then velocity (MAC faces across bricks).
  - Last, vertex phi (duplicated vertices plus resolve).
  - Each field lands on its own, with the Uniform lanes green.
- **Expected:** at the hose, about 182 MB of lattice textures plus 90 MB of arena → tens of MB. Frame time flat or better, with no per-tap regression.
- **Measure:** allocatedBytes; an A/B of hose and long-dam frame time (per-tap cost check); owner-mass conservation.
- **Risks:** faces on h/4h planes; the resolved-phi (hanging texel) contract; Metal compile size of the samplers.

### Phase 4: absent tiles + warm pool for 4h (large; high risk)

- **Steps:**
  1. Add the absent state and pool the 4h texels as well.
  2. The census residency closure plus halo decides the resident set.
  3. Extend the departure/extension certificate to cover residency.
  4. Clamp phi at the resident boundary.
  5. Pressure level 0 runs over resident 4h tiles only, with indirect launches over the resident lists.
- **Expected:** memory and time flat in container size.
- **Measure:** garden-wide at 1×, 2× and 4× container with flat frame time and flat allocation; a forced overflow and a forced departure violation each fail loudly within the WP2 K-frame bound.

### Phase 5 (only if needed): two-level directory (F) for containers above about 10⁷ tiles.

## 7. Open questions

1. **Pool budget source:** an explicit scene field, or derived from liquid volume plus inflow? Explicit is simpler and more honest; derived keeps scene documents clean.
2. **Vertex phi:** duplicated 5³ records (≈2× vertex storage on h tiles, no cross-brick taps), or 4³ plus neighbour lookups (less memory, cross-brick taps)? This is decided by measurement in phase 3.
3. **Negative MAC faces on low brick boundaries next to 4h or absent tiles:** keep the existing negative-plane records generalised per tile face, or add an apron to each brick?
4. **The census reach horizon:** WP2 widened it to 3 steps, which grew the long-dam band by about 46 %. WP4 (a host-free adopt) brings it back to 1 step, and so directly shrinks H and the pool budget. It should precede phase 3.
5. **The native n/4 CM11a continuation and extension hierarchy** stay dense over the container. They are cheap in memory, but O(T) in time, and matter only in phase 4.

## 8. Coordination

- **fluid-d1 / fluid-a0:** own `uniform-mixed-surface.ts` advectDeferred/umAdvectStore, which is active. Phase 3 touches the surface kernels' taps and will be sequenced after their round.
- **fluid-55:** owns tier/offset layout changes (tier-1 removal landed in cf051b46). The tile-word extension in phase 3 is in their area, so it needs a joint design review before edits.
- Nobody else is working on residency or paging (confirmed by fluid-d1, 2026-09-28).
