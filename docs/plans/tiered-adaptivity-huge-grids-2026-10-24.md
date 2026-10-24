# Three-tier adaptivity: h / 4h / 16h storage and work for huge grids, 2026-10-24

Peter: "double down on adaptivity to ensure extremely high fidelity physics and fluid surface treatment at areas of high heat and much less elsewhere in order to save compute … support extremely large (high resolution) grids that are mostly 4h or 16h with only h at certain areas. Both the memory and compute should follow this pattern with enough memory buffer to allow for some growth."

Test scene: `reservoir-seep` (`lib/core/reservoir-seep-scene.ts`, another agent, uncommitted). A deep reservoir on the left and an empty catch chamber on the right, with an 8-cell wall between them. The wall has a one-cell (h) throat with rounded mouths. At 5 mm the lattice is 512×256×64; `buildAt` keeps the throat at h at any cell size.

## 0. Summary

**Today, memory follows resident pages, not h cells.**
- Every resident 32³ block pays about **10 MiB**: full h storage for every detail field, plus particles and the pressure arena. That holds whether its tiles are owned at h or 4h (`uniform-lattice-address.ts:754-758`).
- The default `dynamic` policy reserves h capacity for the whole pool on frame 0 (`webgpu-uniform-reference.ts:1165`).
- A second set of allocations and limit checks is dense over the box (§1.3). Those, the 62-page box limit and the 4096-block pressure tail rule out a 2048-cell axis.

**Today, compute partly follows pool capacity.**
- Storage is sparse, but heat, census, root owners, extension restrict, particle grid steps, the pressure brick update, page kernels and several `clearBuffer`s launch over every region, tile key or block of the pool, every frame (§1.4).
- Cold pages exit early but are still visited.

**The plan:** three storage tiers, each a fixed-capacity slot pool with headroom, all made of the same 8³-cell unit at its own width.

| Tier | Unit | Footprint | Holds | Where |
|---|---|---|---|---|
| h | brick: 8³ h cells (brick arena ℓ0, made persistent) | 8³ h | u, φ, p, solid cuts, particles | throat, jet, impact: high heat |
| 4h | block: 8³ 4h cells (today's page footprint, **without h fields**) | 32³ h | u, φ, p at 4h | free-surface band of quiet water, solid cuts that are exact at 4h, the collar around h |
| 16h | super-block: 8³ 16h cells (**new**) | 128³ h | u, p at 16h, liquid only | submerged bulk with smooth velocity |

**Memory:** a byte count per slot times live slots, plus a fixed directory. Slot capacities come from the scene's expected counts × a growth factor, allocated upfront, and fail fast as today. Nothing scales with the box.

**Compute:** every launch, clear and copy is sized by a live list of its tier. Cold slots cost nothing beyond the pressure rows they hold.

**Projected for `reservoir-seep` (estimates, §3):**

| h | Lattice | Today | Tiered |
|---|---|---|---|
| 5 mm | 512×256×64 | ≈ 2.0 GiB (202 blocks) | ≈ 0.2 GiB |
| 2.5 mm | 1024×512×128 | ≈ 10 GiB | ≈ 0.3 GiB |
| 1.25 mm | 2048×1024×256 | does not fit any limit | ≈ 0.5–0.7 GiB |

At 1.25 mm the frame should cost about what the pour's fine work costs at the same fine-tile count, plus at most about 4 ms for everything else (§3.2).

**Same direction as work in flight.** The persistent brick arena (`brick-arena-core-2026-10-20.md` §2.3: "bricks are the storage for live regions … the paged store holds only cold regions") is the h tier here. It also completes Phase 5 of `domainless-nbflip-2026-10-13.md` ("a lake is out of reach until all-4h pages stop paying for h storage … beyond that it needs coarser levels than 4h").

## 1. Where it stands (surveyed 2026-10-24, HEAD 7a4316df)

### 1.1 Lattice and tiers
- **Units:**
  - a tile is 4³ h, which is one 4h cell;
  - a brick is 8³ h;
  - a 16h region is 4³ tiles;
  - a page (block) is 32³ h, which is 8³ tiles, 4³ bricks or 8 regions.
- **Page table:** a 64³ toroidal window (1 MiB), word = block, AIR or SOLID. The box must be ≤ 62 pages per axis (1984 cells), and ≤ 56 pages in y for a walled box (`uniform-lattice-address.ts:732, 818, 843`).
- **Ownership tiers:** exactly two, `UniformMixedTier = 0|1` (h, 4h) (`uniform-mixed-ownership.ts:98`).
  - 16h exists only as the heat/work region and as pressure MG and extrapolation levels.
  - COLD/WARM/HOT is activity per region, not width (`uniform-region-heat.ts`).
- **An h hole in a wall already forces h.** `umSolidGeometryExact` (`uniform-solid-refinement.ts:50-65`) refuses the throat's tiles, so the census promotes them plus their 26 neighbours once liquid reaches them.

### 1.2 What a block's 10 MiB is

| Item | ≈ per block | Where |
|---|---|---|
| h detail fields (velocity faces A/B/D, volume, surface, γ, vertex φ, unit taps) | 3.05 MiB | `uniform-detail-fields.ts:276-284, 584-626` |
| particles A/B (capacity ∝ cell keys) | 3 MiB | `uniform-narrow-band-flip.ts:447-450` |
| pressure brick arena (ℓ0/ℓ1 slots = 0.45·64·B, plus ℓ2) | 1.7–2.1 MiB | `uniform-pressure-arena.ts:83, 210-231` |
| bins, next, slabs, halos, hanging cache, stage scratch, 4h fields | the rest | §B of the memory survey |

### 1.3 Dense over the box (must go)

| Item | Where |
|---|---|
| `boxCells·24` and `·40` limit checks against binding size (≈107M cells at 4 GiB) | `webgpu-uniform-reference.ts:708-709, 770-772` |
| solid mask bed-slope lane, 2 B per h cell, plus a 65534-unit image cap | `uniform-mixed-solid.wgsl.ts:91-107`, `solid-occupancy-mask.ts:45` |
| authority φ term `owners = 63·fineTiles + boxTiles` | `uniform-mixed-ownership.ts:248, 338` |
| column textures [nx,nz] and the bed-slope bake | `webgpu-uniform-reference.ts:776-781`, `uniform-bed-slope-bake.ts:110-177` |
| detail table at 16h over the box (×2); dynamic-classifier host plane `16·boxPages` (≤ 65535 pages) | `uniform-detail-fields.ts:29`, `uniform-mixed-dynamic.ts:359` |
| host arrays: initial volume nx·ny·nz (2 GiB at 1.25 mm), widths per box tile, `Float32Array(boxCells)` | `uniform-page-pool.ts:431`, `uniform-mixed-layout.ts:68`, `webgpu-uniform-reference.ts:2415` |

**Block-count caps:**
- 33³ vertex texels per block put the z of the block grid at ≤ 62, so B ≤ 3968;
- `TAIL_ROWS = 4096`, one workgroup over the 32h rows of every page (`uniform-pressure-arena.ts:194`);
- layout builder, extension hierarchy and region dynamics: B ≤ 8191.

### 1.4 Launches sized by capacity, every frame
- **Heat, spread, deep, cold freeze, region dynamics:** all regions (`uniform-region-heat.ts:303-470`, `uniform-cold-freeze.ts:273-280`, `uniform-mixed-region-dynamics.ts:408`).
- **Census:** importance, decide, solidActive, solidPromote and pageSeed over all tile keys, plus a work-array clear per encode (`uniform-mixed-dynamic.ts:1797-1811`). Root owners also run over all tiles.
- **Surface:**
  - claims clear and a 4·tiles copy;
  - deferred over vertexKeys/64;
  - `coarsePhi` over the pool's whole 4h vertex lattice (`uniform-coarse-vertex-phi.ts:83`).
- **Particles:** grid steps of 1024 groups striding `UM_TILES` / `UM_CELL_KEYS`, plus O(tiles) clears.
- **Extension:**
  - `restrictBand` over 8·blocks;
  - `finishSnapshot` over 3·tiles/64;
  - dense per-block launches at 4h–32h (`webgpu-uniform-velocity-extrapolation.ts:100-189`).
- **Pressure:**
  - `ubaUpdate` over 64·blocks;
  - every live block on the ℓ2 list;
  - page kernels visit non-live pages;
  - a one-workgroup tail.
- **Renderer:** window classification at 8 workgroups per pool block (`webgpu-water-pipeline.ts:2507`).

Measured floor: 350–450 dispatches. Doubling a cold pool adds about 0.2 ms (`region-tier-performance.md`). That is small at 128 blocks and not small at 4000.

## 2. Design

### 2.1 One unit, three widths
Every tier stores **8³ cells of its own width** in a slot of a fixed-capacity pool. The tiers nest: 4³ bricks per block and 4³ blocks per super-block, each with a 64-word child table.

- **Directory.**
  - The top table stays a 64³ toroidal window (1 MiB, same 6-bit packing), but its entries index **super-blocks**: 64 × 128 h gives 8192 h cells per axis.
  - Word = super-block slot, AIR or SOLID.
  - A super-block slot holds 64 child words (block slot or `NONE` = 16h-owned or empty).
  - A block slot holds 64 child words (brick slot or `NONE` = 4h-owned). This is today's `64·block + i` brick key made indirect.
- **Lookups:**
  - A raw lookup is three dependent reads, against one today.
  - Inner loops do not walk the tree. Kernels follow the brick-arena rule (`brick-arena-core` §3.9): the 27-neighbour record per slot is resolved once per step, and taps come from staged windows or records.
  - The tree is walked by the record builder, staging, particles and the renderer.
  - `umPageWord(page)` becomes `child[superSlot(page>>2)·64 + (page&3)]` during migration.
- **Keys:** tile keys become `512·blockSlot + local` (unchanged meaning). Cell keys become `512·brickSlot + local`: h data is only addressable where a brick exists.
- **Free stacks per tier, fail-fast,** as `uniform-brick-arena.ts` and `uniform-page-pool.ts` already do. Retire after idle steps with one-commit quarantine.

### 2.2 What each tier stores
Target per-cell bytes. Scratch is shared across tiers and sized by the largest live list, not by capacity.

| Channel | h brick | 4h block | 16h super-block |
|---|---|---|---|
| u faces ×3, A/B | 24 B | 24 B | 24 B |
| φ (vertex), banks | 8 B | 8 B | — (liquid-only tier; sign is implicit) |
| p, rhs, row/coef per MG level | ~20 B | ~20 B | ~20 B |
| apertures / fill (cut slots only) | 8 B | 8 B | — (cells are fully open or SOLID) |
| class, heat, revisions | per slot | per slot | per slot |
| **≈ per slot** | **≈ 32–48 KiB + particles** | **≈ 32–48 KiB** | **≈ 24–32 KiB** |

- **Particles live only in bricks:** capacity = particles per fine cell (pour: about 2; reserve 4) × brick capacity × 512 × 48 B, which is about 96 KiB per brick slot.
- **Sampled channels:**
  - The channels sampled through the texture unit (u, φ) are a 3D brick atlas with a 1-cell apron, which costs (10/8)³ ≈ 1.95× on those channels alone. That keeps the hardware-filtered taps that won 1.6× in the particle and surface traces.
  - The apron is refreshed by the writer's epilogue, as the unit-tap halo is today.

### 2.3 Rules for the tiers
- **16h is submerged bulk only.** A super-block cell is 16h-owned only if all of these hold:
  - no φ crossing within one 16h cell;
  - each 16h cell is fully fluid or fully SOLID (the `umSolidGeometryExact` test one level up);
  - its velocity passes the coarsening error test.

  The free surface, cut solids and particles never exist at 16h. That keeps the surface, splat, extension and particle stages two-tier, and 16h touches only momentum, forces and pressure.
- **Grading:** neighbouring owners differ by at most one tier, across faces, edges and corners. Each tier keeps a one-cell collar of the next coarser tier, so the h↔4h seam rules (ratio 4) are reused verbatim for 4h↔16h, including the reflux of the coarse face as the V-weighted sum of 4×4 fine subfaces and d7fa24db's seam pressure gradient.
- **Refinement criteria** extend region heat. Nothing is keyed on the scene name.
  - **h:** today's census (shape, thin, strain, impact, approach, solid contact, particles, bodies), unchanged.
  - **4h → 16h:** the region (16³ h, which is exactly one 16h cell) has been deep COLD or WARM-submerged for `holdSteps`. Its restrict→prolong velocity error is below `coarseTolerance` (Sato high-pass one level up, relative to the local speed scale). It is graded.
  - **16h → 4h:** the predicted error exceeds the tolerance; or a φ crossing, cut solid, h brick or particle is within reach. Reach is velocity-projected, as the departure-box trace already is (`uniform-mixed-dynamic.ts:1430-1519`).
  - Hysteresis is as in `uniform-region-heat.ts`: fade, hold, retire ratio.
- **Remap on a tier change** reuses the existing remap: prolong on refinement, V-weighted restrict on coarsening. Both preserve a uniform field exactly (bed-slope's guarantee, `bed9fcbf`).

### 2.4 Pressure
The reservoir is near-hydrostatic, and the seep is driven by the global head. Pressure stays globally coupled, but its cost follows rows.
- **Levels:** h bricks → 2h (scratch) → 4h blocks → 8h (sub-block) → 16h super-blocks → 32h / 64h (sub-super-block) → 128h, which is one row per super-block.
- **Tail:**
  - The coarsest solve is one workgroup over super-blocks: 4096 super-blocks covers 8.6G h cells.
  - `TAIL_ROWS` caps super-blocks, not blocks.
- **Operators:** rediscretised at every level, as agreed for the arena (§3.10). Composite seams at both tier boundaries.
- **Warm start:** p is persistent at every owner, so a quiet reservoir converges in one or two V-cycles. Its residual is local to the throat and jet.
- **Lists:** liquid rows per tier. A block or super-block with no rows is not visited.
- **Rows at 1.25 mm** (estimates): about 0.5M h, 0.5–0.8M 4h and about 0.07M 16h. One V(2,2) cycle is about 1 ms at streaming rate.

### 2.5 Compute: nothing sized by capacity
- **One rule:** no launch, clear or copy is sized by a pool's capacity. Each is a direct grid-stride over a live list whose count lives on the GPU. That keeps the no-indirect-dispatch rule; the fixed grid is small, about 1–4k groups, and its empty groups cost less than a microsecond.
- **Lists built once per step:** `live`, `hot`, `warm`, `surface`, `liquid`, `cut`, `enter`, `leave` per tier, from the frame plan (`uniform-regional-frame-plan.ts`).
- **Clears:** O(capacity) clears become dirty-list clears.
- **Cold slots** cost one census bit and their pressure rows. They cost no heat or freeze pass: those run over warm and hot lists plus a ring of their neighbours.
- **Replacements** for every §1.4 item:
  - census over live tiles;
  - `coarsePhi` over surface blocks;
  - extension hierarchy over the extend list;
  - particle grid steps over particle bricks;
  - renderer windows over surface blocks and bricks.

### 2.6 Dense-box removal (independent of tiers)
- **Initial conditions and solids** are evaluated per claimed slot on the GPU from the scene's shapes: box volumes, voxel patches, cylinders.
- **The solid mask** becomes sparse:
  - SOLID super-blocks and blocks are directory words;
  - cut bricks hold apertures;
  - the bed-slope lane exists only in cut slots.
- **Smaller items:**
  - column textures become a per-block column table;
  - the owners term becomes live slots;
  - the detail table moves into the super-block child words;
  - dense device checks are deleted.
- **Host:** nothing on the host is sized by nx·ny·nz.

### 2.7 Capacity and growth
- **Scene parameters:** `fluid.tiers = { bricks, blocks, superBlocks, particlesPerFineCell }`, or `memory_MiB` split by measured live ratios.
- **Default:** the t=0 census count × 1.5, plus a floor of 256 bricks.
- **Fixed:** allocated upfront. Exhaustion stays fatal, and the message names the tier.
- **Live counts and high-water marks** are published every frame. The UI shows them beside `detailBudgetPercent`, so headroom is visible before it runs out.

## 3. Budgets for `reservoir-seep`

### 3.1 Memory (estimates; the counts are what Phase 0 measures)

| | 5 mm | 2.5 mm | 1.25 mm |
|---|---|---|---|
| Lattice (h cells) | 512×256×64 (8.4M) | 1024×512×128 (67M) | 2048×1024×256 (537M) |
| Liquid (h cells) | 3.1M | 25M | 200M |
| Live bricks (throat + jet + impact/film) | ~500 | ~800 | ~1–2k |
| Live blocks (surface band + wall sheet + collar) | ~150 | ~400 | ~1000 |
| Live super-blocks | ≤ 8 | ≤ 32 | ~130 of 256 |
| Capacity ×1.5–2 at ~150 / 48 / 32 KiB per slot | ≈ 0.2 GiB | ≈ 0.3 GiB | ≈ 0.5–0.7 GiB |
| Today | 2.0 GiB | ≈ 10 GiB | over the window, B and tail limits (~60+ GiB) |

**Biggest unknown: the catch floor.**
- At 5 mm the seep delivers about 1.8 L in 20 s (Torricelli: Cd·h²·√(2g·0.64 m)). Over 0.41 m² of floor, that is a film about one h thick.
- The `thin` criterion would put the whole floor at h, about 256 bricks at 5 mm and more at finer h.
- Phase 0 measures it. If it dominates, the thin criterion needs a minimum film-area rule, or a sub-h film goes to particles.

### 3.2 Frame (M1 Max, 1/60 s)

| Part | Target | Scales with |
|---|---|---|
| Fine stages (surface, particles, RFE and momentum at h) | the pour's per-kT rates after the speed-of-light work | live fine tiles (8 per brick) |
| 4h active stages | ≤ 1 ms | warm/hot blocks |
| 16h momentum and forces | ≤ 0.3 ms | live super-blocks |
| Pressure, all rows, warm-started | ≤ 2 ms | rows |
| Floor (census, plan, lists, pool commit) | ≤ 1 ms, independent of box and capacity | — |
| Surface extraction | ≤ 1 ms | surface blocks and bricks |

**Scaling gates.**
- Doubling the reservoir length at a fixed h with the same throat adds ≤ 0.2 ms and no h storage.
- Halving h scales the frame with fine tiles, not with h⁻³.

### 3.3 The jet is fast in h units
The head is (¾ − ¼)·1.28 m = 0.64 m, so the jet leaves at about 3.5 m/s. That is 5.9 cm per step at 1/60 s:

| h | Jet travel per step |
|---|---|
| 5 mm | 12 h |
| 2.5 mm | 23 h |
| 1.25 mm | 47 h |

The known band-face speed fatal is at 25–47 cells a step (`domainless-nbflip-remaining`). The rules are no substeps and 1/60 s everywhere. So the h band and residency must reach the jet's next position by velocity projection, and the band must tolerate a stream that crosses several bricks a step. This is a physics decision (§6, item 2), not just plumbing.

### 3.4 Scale ladder (W9, measured)
`tools/probe-scale-ladder.ts` builds the seep at a cell size (throat at h, `--width`/`--depth` scale the box at the same head), constructs the production solver with an explicit pool and reports the first failure, bytes by class, live counts, jet speed in h a step and catch-floor h tiles. `--plan` evaluates every host-side limit without a GPU. Summaries: `artifacts/scale-ladder/`.

```
env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-scale-ladder.ts \
  --cell=0.0025 [--width=W] [--depth=D] [--pool=N] [--steps=10] [--hold=/private/tmp/claude-501/fluid-gpu-run.sh] [--plan]
```

**First blocker per rung (main at 8010cd35, after W1 and W8's tail; plan checks, no GPU).**

| Rung | Lattice | Pool (blocks) | First blocker | Owner |
|---|---|---|---|---|
| 5 mm (control) | 512×256×64 | 226 (160 resident pages) | none; runs | — |
| 2.5 mm | 1024×512×128 | 1442 (1008 resident) | memory: 22.5 GiB at 16 MiB a block | W6 |
| 2.5 mm, pool 64 | 1024×512×128 | 64 | seed holds more pages than the pool | W6 (pool sized by memory) |
| 2.5 mm, ¼ depth | 1024×512×32 | 362 (252 resident) | none at the plan; 5.7 GiB | — |
| 2.5 mm, ½ depth | 1024×512×64 | 722 (504 resident) | none at the plan; 11.3 GiB (timing lane only) | — |
| 2.5 mm ×1.5 width | 1536×512×128 | 2082 (1456 resident) | memory: 32.5 GiB | W6 |
| 2.5 mm ×2 width | 2048×512×128 | 2722 (1904 resident) | 64 pages on x, past the window's 62 | W7 |
| 1.25 mm | 2048×1024×256 | 10882 (7072 resident) | 64 pages on x, past the window's 62 | W7 |
| 1.25 mm, ¼ width, ⅛ depth | 512×1024×32 | 402 (260 resident) | none at the plan; 6.3 GiB | — |
| 1.25 mm, ¾ width, ⅛ depth | 1536×1024×32 | 1042 (676 resident) | memory: 16.3 GiB | W6 |

**Behind the first blocker** (each check on its own): 2.5 mm ×2 width: 42.5 GiB (W6). 1.25 mm: a block grid z of 171 > 62, i.e. more than 3968 blocks (W6/W7); 170 GiB (W6); the seed residency's host pass, 100–230 s over 537M cells on a loaded machine (W1). The box-sized device checks (3D texture extent, boxCells·24 and ·40 bindings, which also failed 1.25 mm and 2.5 mm ×2 width at 0deb44ee) went with W1's 8bf2039e, and the 4096-block page tail with W8's super-block tail (51f3b614).

**5 mm control on the GPU** (7e75a94d, check lane, 240 steps = 4 s; ms are not timing-lane numbers): no failure. Live 3533 MiB for 226 blocks (15.6 MiB a block): h detail fields 1165, particles 974, ownership/layout 751, pressure 561 MiB. The largest single buffers are sized by the whole pool, not by what is live: pressure arena 469 MiB, mixed stage scratch 440 MiB (fineCapacity 114688 tiles against ≤ 621 live h tiles), FLIP particles A and B 339 MiB each against ≤ 2792 particles (W10), and FLIP cell counts and starts 269 and 266 MiB. The jet runs at 13.7–14.1 h a step from step 40 (4.1–4.2 m/s); median step 19.6 ms (check lane); build 34.5 s outside the lock.

**Catch floor at 5 mm** (`seep-5mm-floor`, 120 steps = 2 s, the window-address decode): catch-chamber h tiles run 73–333 (peak 333 at step 80), and floor tiles (ty < 2) stay 0. By 2 s the jet has not landed. W0's 448 floor peak comes later in its 20 s run, so the floor count at a full horizon is still to be measured.

**GPU rungs below 5 mm (check lane, ms not authoritative)**: both fail on the same fatal, after a max-speed runaway.

| Rung | Lattice, pool | Live MiB | First failure | Max speed, m/s (h a step) |
|---|---|---|---|---|
| 2.5 mm, ¼ depth | 1024×512×32, 362 blocks | 5574 (15.4 a block) | step 8: departure closure 35 > 32 tiles, page (15,6,0), uniform-mixed-frame.ts:1305 | 4.14, 6.96, 9.78, 12.5, 15.5, 18.4, 21.2 (27.6 → 141) |
| 1.25 mm, ¼ width, ⅛ depth | 512×1024×32, 360 blocks | 5549 (15.4 a block) | step 3: closure 40 > 32 tiles, page (1,3,0) | 6.54, 12.19 (87, 162) |

The speed gains 2.8 m/s a step at 2.5 mm and 5.6 m/s a step at 1.25 mm. Gravity adds 0.16 m/s a step and Torricelli allows about 4 m/s, so this is an instability, not honest fast flow. The gain scales as 1/h. W5 traced it to grid momentum's single RK2 midpoint trace (umUnitTrace) at CFL ≥ 12 through the throat. Faces whose midpoint lands in the slow reservoir come back near zero, and the projection hands their flux to the few fast faces. The fix candidate is NB-FLIP `momentumTrace: "turn"` (W5 e2206321, not on main), which is unverified at these rungs. Both rungs are one page thick in z, and 5 mm runs two, so the 5 mm ½/¼-depth controls (`seep-5mm-d0.5`, `seep-5mm-d0.25`) were queued to rule out single-page-z halos and never ran. 1.25 mm bytes by class: h detail 1804, particles 1551, ownership 1198, pressure 893 MiB. The biggest buffers are still sized by the whole pool: pressure arena 746, stage scratch 703, FLIP particles A/B 540 each.

**With the page window and memory both fixed, the next binding limit is the block grid**: 3968 blocks is 62 GiB at today's 16 MiB a block, so memory binds first until W6 cuts the block's cost; after that, W7's directory must lift the grid before 1.25 mm (10882 blocks).

Before 877f9138 the 1.25 mm refusal came after 452 s of host passes; 52b9daca keeps it ahead of them now that the window check lives in the address (`uniformLatticeWindowBox`). c406a706 removed the per-cell allocations of the host initial-liquid passes (seep GC 9.3 s → 2.4 s at 5 mm, hashes identical).

## 4. Phases
Each phase ends with numbers, the letters A/B, the analytic suite (no regressions against `nbflip-analytic-correctness` round 5) and a commit. Physics changes sit behind NB-FLIP params, and the old path stays the default until the A/B passes.

**Phase 0: baseline and harness (no solver change)**
- **Runs:** `reservoir-seep` at 5 mm, plus 2.5 mm if it fits, under `benchmark-nbflip-ui.ts --memory --profile` and `probe-nbflip-peaks.ts`.
- **Record:**
  - live h tiles, blocks, regions by tier and heat, and particles over 20 s;
  - allocated bytes by class (`uniform-gpu-allocation-audit.ts`);
  - fenced step;
  - seep discharge against Torricelli;
  - volume drift.
- **Length ladder:** reservoir ×1, ×2, ×4 at 5 mm (a `buildAt` variant, as `region-tier-scenes.ts` does) to measure the capacity floor directly.
- **Output:** the real counts for §3, replacing the estimates.

**Phase 1: dense-box removal (§2.6) and capacity-sized launches (§2.5)**
- **Change:** pure work and memory; the fields are bit-identical.
- **Gates:**
  - identical field hashes against HEAD on letters and seep 5 mm;
  - the length ladder's slope ≤ 0.2 ms per doubling;
  - no allocation term in nx·ny·nz.
- **Unblocks:** 2.5 mm's dense checks. Its 10 GiB of blocks still needs Phase 2.

**Phase 2: h storage leaves the block (bricks are h storage)**
- **Change:** this is `brick-arena-core` §2.3's persistent arena, extended from pressure to every h field. Cell keys become brick keys. Blocks keep 4h data only. Particle capacity moves to bricks.
- **Ordering:** coordinate with the speed-of-light owners. This is the same migration (surface, then pressure, then RFE onto the arena), and its completion is what drops the h fields from blocks.
- **Gates:**
  - letters A/B and analytic suite in repeat noise;
  - seep 5 mm bytes ≈ the §3.1 tiered estimate;
  - 2.5 mm runs in under 1 GiB.

**Phase 3: hierarchical directory and the 16h tier**
- **Change:**
  - super-block slots and child tables (§2.1);
  - the 16h owner with grading and the 4h↔16h seam (§2.3);
  - pressure levels to 128h and the tail over super-blocks (§2.4).
- **Physics gates:**
  - hydrostatic rest with a 16h bulk: max speed ≤ 1e-5 m/s, zero volume drift;
  - seep discharge and jet trajectory, with 16h allowed against disallowed, within repeat noise. This is the decisive test that the bulk at 16h does not change the jet;
  - the analytic suite with 16h on, no regressions.
- **Scale gate:** 1.25 mm runs in under 1 GiB at the §3.2 frame.

**Phase 4: fast-front reach**
- Velocity-projected h promotion and residency for the jet at 23–47 h a step (§3.3), within the rule set Peter chooses.
- **Gate:** 1.25 mm runs 20 s without a speed or band fatal, with discharge matching 5 mm and 2.5 mm within the expected resolution trend.

## 5. Risks
- **Directory depth.** Paged loads already cost 2.5–2.7× box loads, and a third read adds more. Mitigation: keep the tree out of inner loops (records and staging). Measure with a knock-out before switching (`benchmark-nbflip-address-layers.ts`).
- **Seam artefacts at 4h↔16h.** Spurious currents along the seam would wake the bulk. Gate: the rest test above. The reflux makes the coarse flux exact by construction.
- **Wall alignment.** The 8-cell wall's faces sit on 4h boundaries but not on 16h boundaries (x = nx/2 ± 4). A full-height sheet of 4h blocks is then needed on both faces: about 512 blocks at 1.25 mm, roughly half the block budget. Ask the scene agent to place the wall faces on 16-cell boundaries, e.g. `[nx/2 − 16, nx/2)`, keeping the h throat.
- **Thin film on the catch floor** (§3.1).
- **Fast jet** (§3.3).

## 6. Decisions (Peter agreed all four, 2026-10-24)
1. **16h is liquid-only bulk:** no surface, cut solids or particles at 16h (§2.3).
2. **Jet speed:** keep 1/60 s. Develop at 2.5 mm (23 h a step) first, then make reach velocity-projected (Phase 4). No substeps and no smaller scene dt.
3. **Wall faces on 16h boundaries** in `reservoir-seep`, keeping the h throat (§5).
4. **Phase 2 is the speed-of-light arena migration,** extended to every h field, not a parallel effort.

## 7. Execution: parallel workstreams
Peter: "Implement all of it using parallel agents so that we can move fast." Each workstream runs in its own git worktree and branch. A workstream lands on main only after its gates pass, and the coordinator merges.

| Stream | Phase | Scope | Lands when |
|---|---|---|---|
| W0 baseline | 0 | measurement harness, seep counts, length ladder, §3 numbers | numbers recorded here |
| W1 dense | 1 (§2.6) | everything sized by nx·ny·nz, on device and host | field hashes identical; no box term in allocation |
| W2 lists-head | 1 (§2.5) | heat, freeze, region dynamics, census, root owners, frame plan, pool commit: live lists, no capacity clears | identical hashes; length-ladder slope |
| W3 lists-stages | 1 (§2.5) | surface, particles, extension, pressure page visits, renderer windows | identical hashes; length-ladder slope |
| W4 16h physics | 3 (physics half) | 16h owner tier, grading, 4h↔16h seam, criteria and pressure levels, on today's storage, behind a param | rest test; discharge A/B 16h on/off; analytic suite |
| W5 fast front | 4 | velocity-projected h promotion and residency for 23–47 h a step | 2.5 mm-class fixture runs 20 s without a speed or band fatal |
| W6 bricks | 2, then 3 (storage half) | h fields onto persistent bricks; blocks keep 4h only; then the super-block tree | letters A/B; bytes match §3.1 |

The W6 storage moves follow the W1–W3 merges, because they rewrite the same addressing.

### 7.1 W4 results (16h physics, branch `worktree-agent-a52ae86e35fe7c7ab`, 2026-10-25)
NB-FLIP param `bulkTier: "off" | "16h"`, default `"off"`. Commits: 28c303bd (the tier, CPU twin, probe), 2f712b94 (detail-field binding), e36d90e9 (row offsets, live frozen faces), 8a70b571 (glued level-2 rows) reverted by 70e81733.

**Design.**
- **Unit:** a 16h owner is one region (4³ tiles = one 16h cell). Selection is a per-region kernel (`uniform-bulk-tier.ts`) after region heat; its CPU twin and rules live in `uniform-bulk-tier-reference.ts`.
- **Storage (for W6 to move):** constrained, on today's storage. A member's data stays in its 64 tiles' 4h storage, plus a record buffer (flags, hold, error, counts, member list, seam heads, the member's 16h pressure and row offset) and an r32float faces texture (128 texels a region: six face means and each seam subface's value). W6 replaces the 64-tile constraint with a 16h slot holding the same record; the rules don't change.
- **Criteria:**
  - *Structural* over the 12³-tile neighbourhood (one 16h cell of margin): no h band tile, HOT region, air, cut tile, open tile above 5.5 h deep, or closed own tile, and reach `speed·dt ≤ 8 h`. SOLID pages and closed tiles beside it are walls. This also gives the grading: h is at least one 4h region away on faces, edges and corners.
  - *Error:* the Sato high-pass one level up, `max |u − prolong(restrict u)| < 0.1·speed + 0.002·h/dt`, on each axis's 5×16 4h faces, with restrict as the V-weighted face mean and prolong linear between the two face means.
  - *Hysteresis:* enter after 30 consecutive passing frames; leave at once on structure or at an error above twice the threshold.
- **Pressure:**
  - A member's 64 arena rows take one correction: the V-cycle update is the region mean on member lanes, so their sum is the 16h row. Each row keeps its offset from the region mean, so the hydrostatic profile stays in the rows other stages read.
  - The operator couples a member at its 16h pressure (its rows' start mean): member↔member at 16h spacing, member↔native at 10h (the 4h↔16h seam, reusing the h↔4h head form at ratio 4: native start slope × tangential offset).
  - Level-2 correction couplings stay native. Gluing them to the member's true couplings (8a70b571) sent the rest probe HOT at step 60, so it was reverted.
  - The MG levels, tail and warm start are unchanged (W8's).
- **Seam and faces:**
  - Inside a member, every 4h face is prolonged linearly from the six face means.
  - A face mean is the V-weighted sum of its 4×4 subfaces /16 (reflux). The natives across a seam see the subface values, and their sum is the 16h flux exactly.
  - Frozen (COLD) member faces are written live as `bulk − umColdFaceTerm`, with no correction charged to the freeze budget.
- **Remap:** a refine leaves the 4h faces and rows as they are (the prolonged field is already stored). A coarsen takes the V-weighted face means. A uniform or axis-linear field survives both exactly.
- **Tests** (`tests/uniform-bulk-tier.test.ts`): remap exactness, hysteresis, grading/structure, the hydrostatic composite rest state across seams, and a divergence-free projection with the reflux identity.

**Gates** (check mode unless marked).
- **First run:** bulkTier=16h first ran the closed-throat rest scene at 2f712b94: 248 members at step 45, zero drift.
- **1. Hydrostatic rest** (`reservoir-seep` 5 mm, throat closed, `tools/probe-reservoir-bulk.ts --closed`, 900 steps at e36d90e9 ≡ 70e81733):
  - Owners: 535–541 of 928 regions are 16h members from step 31 (about 34.5k of the 4h tiles' worth); the other ~390 regions stay 4h, with 54 h tiles.
  - Heat stays ≈ [922 COLD, 6 WARM, 0 HOT]. Volume drift is exactly 0 over 900 steps.
  - Max speed falls 2.55e-2 → 1.34e-2 (step 120) → 4.99e-3 (step 420), then holds.
  - Off over the same 90 steps: 2.55e-2 → 1.505e-2 at step 90, against 1.548e-2 with 16h. Its residual is 0.003–0.03; with 16h it is 0.01–0.08.
  - **Not met:** ≤ 1e-5 m/s is not reached by either arm. The scene's own start transient decays slowly; 16h doesn't cause it.
- **2. Discharge A/B** (`reservoir-seep` 5 mm, two repeats per arm, 600–1200 steps):

  | | escaped L (1 s) | escaped L (10 s) | mean x m (10 s) | front m (1–2 s) | h tiles (10 s) | max speed (10 s) | drift (10 s) |
  |---|---|---|---|---|---|---|---|
  | off a / b | 0.019 / 0.016 | 0.001 / 0.003 | 1.338 / 1.527 | 2.558 | 105 / 232 | 4.106 / 4.106 | −4.2e-6 / 3.8e-6 |
  | 16h b / c | 0.024 / 0.022 | 0.003 / 0.004 | 1.524 / 1.432 | 2.558 | 238 / 206 | 4.099 / 4.098 | 2.9e-6 / 4.3e-6 |

  - With 16h, 532–543 regions are members throughout.
  - Every 10 s value is inside the off repeats' spread.
  - The 1 s burst is about 25% larger in both 16h repeats (0.022–0.024 L against 0.016–0.019 L). It is the one systematic difference.
  - **Caveat:** in both arms the seep stops after about 2 s: the escaped volume falls to 0.001–0.004 L, the floor front vanishes, and the max speed holds at ≈4.1 m/s. That is baseline behaviour of e218ecfe, so this A/B only tests the first 2 s of jet.
  - **Gate-tool A/B** (repeat-spread, a 16h-default snapshot against e218ecfe, 120 frames, two repeats each):
    - The first run failed on particles only: 2 of 12 windows over, worst ratio 1.22 at frames 91–100 (A/B 124 against a spread of 102). h tiles, band, drift and occupancy passed.
    - A rerun of the same pair passed everything, with a particles ratio of 0.22.
    - So the particle count's two-repeat spread underestimates its noise. The 16h arm's slightly larger early burst, seen in both runs here and in the probe above, is the thing to watch with more repeats.
- **3. Analytic suite** (`--set=bulkTier:16h`, e36d90e9): identical to round 6 at e0d8994f, because no region engages on the suite's 64×64×32 grids.
  - rest, rest-step, standing-wave and translation pass. The figures below are the e0d8994f baseline's, not W4's.
  - free-fall: φ drift 5.84%.
  - dam-break: φ drift 3.79% (round 5: 3.3%).
  - mound: φ drift 4.80%.
  - So no case is worse than round 5 because of 16h. The suite doesn't exercise the tier.
- **4. bulkTier off is unchanged:**
  - Shader digests of `reservoir-seep` (1 cm, 103 modules) are identical between the merge-base e218ecfe and HEAD. So are the 3-step max speed (2.369669198989868) and drift (4.3594e-6).
  - Repeat-spread gate (`tools/gate-repeat-spread.ts` from f82ca90c, window means, two repeats per arm, e218ecfe against HEAD): splash 70 frames passes, with h tiles, band, particles and occupancy identical in all four instances and drift A/B 1.8e-7 against an A/A spread of 3.6e-7. `reservoir-seep` 5 mm, 120 frames, passes, the worst window ratio being 0.42 (band bricks).
  - Letters (`nbflip-figure-8-letters`), 120 frames, pass, the worst window ratio being 0.81 (particles: A/B 3452 against a spread of 4253).
  - `tools/benchmark-nbflip-letters.ts --track --plain` stops at frame 1 with bulkTier off ("complete exactly one requested clock step"). The off path is the merge-base's, so that is a harness/baseline issue, not W4; the gate tool runs the letters scene instead.
- **Letters with 16h on:** the repeat-spread gate of a 16h-default snapshot against e218ecfe (120 frames) passes. The worst window ratio is 0.70 (band tiles), and drift peaks at 0.18 of its band.

**Failures and diagnosis.**
- *"Destroyed texture velocity A used in a submit":* the selection bound a detail field's texture view directly, so every frame's submit was rejected once the physical texture was replaced. Fixed by binding through `uniformDetailGroup` and marking the canonical load (2f712b94).
- *Members thawed the reservoir* (248 members fell to 20, regions went HOT): resetting a member's rows to the region mean put a pressure step at the seams and kinks in the hydrostatic profile, and the frozen faces' corrections were charged to the freeze budget. Keeping the row offsets and writing frozen member faces live fixed it (e36d90e9).
- *Glued level-2 rows* (8a70b571): the coarse operator no longer matched the native rows' residual, so the V-cycle injected non-hydrostatic corrections and the rest scene went HOT at step 60 (max speed 0.257 m/s). Reverted.

## 8. Phase 2 inventory: who reads and writes h storage (W6, at e218ecfe)
Every reader and writer of every h-resolution field, read from the code (not traced). B = pool blocks with the sentinels; C = 32768·B cell keys; V = 35937·B vertex keys; F = the h-tile capacity. Paths are under `lib/methods/uniform/` unless they start with `lib/core/` or `tools/`.

**Domains:**
- **H:** h-owner tiles only.
- **4hS:** also touches h storage of 4h tiles (seam patches, hanging texels, ring texels, h anchors of cut 4h faces).
- **C:** canonical texels only (a tile origin, a face anchor, a 4h corner), which the base block serves.
- **L:** the whole lattice, or every block.

### 8.1 What holds h storage today
`UniformDetailStorage` gives each field two homes:
- a base block: 8³ tile texels a block, or 9³ corners for a vertex field;
- while F > 0, an h store shaped like the block grid: 32³ cells or 33³ vertices for **every** block, whatever its tier.

Twin bindings (+64) write canonical texels to both. Pinned (per-tile) fields are base-only at every F.

| Field (reference name) | Class | h store a block | Frame name |
|---|---|---|---|
| velocityA / B / D | face rgba32f | 512 KiB each | V `velocity`, S `velocityScratch`, D `departure` |
| unit taps U, NB-FLIP stage taps SU (`createTaps`) | atlas (face) rgba32f | 512 KiB each | `unitVelocity`, stage unit taps |
| volumeA, volumeB, surfaceA, surfaceB, gammaA, gammaB | cell r32f | 128 KiB each | volume, volumeScratch, phase, centerPhi, correction, target |
| vertexPhiField, vertexPhiScratch | vertex r32f, base adopted from the coarse vertex phi | 140 KiB each | `phi`, `phiScratch` |
| pressureGeometry.target / centerPhi | cell, pinned | base only (2 KiB) | — |

That is **3.5 MiB a block** of detail fields. Beside them sit the raw block-keyed unit halo (35³ rgba32f, 670 KiB) and its rgba16f half halo (335 KiB).

**Buffers keyed by h keys** (`uniform-narrow-band-flip.ts:435-481`):
- **Particles:** banks A and B at 48·cap each. cap = min(8C, max(2²¹, C), limit/64), which is C at B ≥ 64: 1.5 MiB a bank, **3 MiB a block**.
- **`bins`:** counts 2C (no named reader of [C, 2C)), NB_DEPTH_A max(V, C/2), NB_DEPTH_B C, surface-hot V.
- **`links`:** cell starts C, then 7 words a particle, then 729-word arena windows for every one of the 64·B bricks.
- **Stage scratch:** at least 4V (the surface deferred list, surface-volume `SV_P`).
- **Boundary velocity planes:** 4·LF each.
- **Already sized by bricks:**
  - brick table: 31 words;
  - slab records: 17 words;
  - pressure arena: fineCap = min(64B, max(2048, ⌈0.45·64B⌉)), about 1.8 MiB a block.

### 8.2 Cell scalars

| Field | Writers | Live readers in NB-FLIP | Reads |
|---|---|---|---|
| gammaA (correction) | t=0 zero upload; authority `resolve`, not compiled under `surfaceOnly` | **none**: legacy t=0 binding 24 and QA `physicsFieldsForQA` only | — |
| pressureGeometry.target | `geometry*` at tile origins (`uniform-mixed-surface-geometry.ts:299`) | **none**: the split authority's `targetFill` is dead | — |
| volumeB (volumeScratch) | `toPressureCoarse` / `toPressureFine`, tile origin only (`uniform-mixed-remap.ts:494/497`) | pressure-velocity `project` via `/*r4h*/` (`uniform-mixed-pressure-velocity.ts:374-438`) | **C**: never needs h storage |
| gammaB (target) | `geometry*` (h cells and 4h origins); t=0 `uvPublish` | `mirror*` (→ volumeA); `measureInitial` (budget capture, gauge steps); grid overlay `openFraction` | H + C |
| volumeA (volume) | `mirror*` from gammaB, at owner origins | `toPressureFine` (64 h cells); `toPressureCoarse` (`/*4h*/`); forces `umCoarseMass` (C); `momentumFine` (64 + 48 cells); `momentumTrace` / `momentumRegularStep`; `diagnostics`; renderer `volumeTexture`; tools | H + C, seam gathers |
| surfaceA (phase) | authority `surface` family (`umSurfaceOwner`); t=0 `buildDenseExtrapolationAuthority` (L, then `encodeBaseRefresh`) | extension `seed*` / `publish*` (`ueSeedOwner`, `ueHeldFace`); extension pair; NB `advectParticles` (`gridLiquidCell`, the trilinear footprint) | H + C, seam gathers |
| surfaceB (centerPhi) | `geometry*`; t=0 `uvPublish` | authority (origin, plus 6 neighbours on solid h cells); forces (`umCellPhi`, `umFineOccupancy`); momentum; renderer `surfaceFieldTexture` (water pipeline, overlay) | H + C, seam gathers |

**Bound but never read in NB-FLIP:**
- the volume bindings of the remap faces, `remapCells`, the frame plan and the dynamic classifier;
- momentum's phase.

### 8.3 Vertex φ
**Writers into h storage:**
- Surface advect (`advect*`, `solid*`, `wallReach`), writing S: H, plus h texels of 4h tiles.
- `coupleFine` and `airClearFine`: H.
- `arenaStore`: H, plus the seam-face vertices whose cell is 4h.
- **Phi resolve** (`uniform-mixed-phi-resolve.ts:123`): the hanging texels of mixed-stencil **4h seam tiles**, on P and S.
- **Remap** (`remapCells` and the plane kernels): new h texels in **former 4h tiles**.
- Page initialise: whole 33³ blocks, skirt included.
- The ring's `enter`: canonical corners.

**Canonical writers** (C, twinned to the base):
- `coarseCorners` / `couple`;
- `redistanceCoarse`;
- `arenaDistanceCanonical` / `arenaPublishCanonical` / `arenaPublishFinish`;
- `airClearCoarse`.

**Readers that gather h texels, the hanging texels of 4h tiles included:**
- surface advect (8-tap, and cubic 64-tap over raw 33-stride `udRun_phi` / `udTexel_phi`);
- pressure arena `pMark` / `pRows` (`bNeighbour`);
- particles (`bandPhi` / `umSampleVertex` in move, seed, transfer and depth);
- census and activity (`classify`, `umResolutionError`);
- frame plan `seed` / `localSpeed`;
- surface volume, geometry (`gcOwner`), membership (`nbCrossingVertex`), surface work control;
- `arenaGather`, which reads `udTexel_phi(33*umBlockCoord(block)+…)` raw;
- forces and diagnostics, through the unresolved authority walk;
- bodies `couple`, at corners of any width;
- presentation `denseVertexPhi`, for the water pipeline and the grid overlay.

**Buffers keyed by vertex key:**
- NB_DEPTH_A and NB_SURFACE_HOT(_NEAREST);
- the surface deferred and cubic lists;
- surface-volume `SV_P`;
- two inline 35937·block keys: `uniform-narrow-band-membership.wgsl.ts` about lines 309-321, and `uniform-surface-brick-arena.wgsl.ts:38-41`.

**The base must keep its layout:** it is the adopted coarse vertex phi, which the renderer reads directly as `coarseVertexPhi`.

### 8.4 Velocity faces (V, S, D) and taps (U, SU)
**Writers into 4h tiles' h storage (4hS):**
- extension `publishSeams` / `publishList`;
- NB `transferOwners` / `transferSeams`;
- `toPressureFine`;
- `momentumStep` (the merged 4h/seam dispatch) and the merged `forces`;
- `pProjectNative` and pressure-velocity `project`;
- `remapFaces` / `copyFaces`;
- the seam slots of `unitVelocityTaps`, which write U at 4h seam tiles;
- the ring's `enter`, and page initialise.

**H-only writers:**
- `publishFine`;
- `momentumFine` / `momentumTrace` / `momentumRegularStep`;
- `forcesFine` / `forcesRegular`;
- `transferProperties` / `transfer`;
- `pProject` / `pCopy`;
- the unit jobs of `unitVelocityTaps`;
- `toSimulation*`.

**Texture-filtered readers** (sampler, owners of any width, L):
- NB G2P: `advectParticles*`, `advectWalk`, seed, `countInitial` / `seedInitial`, and `snapshot` via SU;
- surface advection: S, U and the halo;
- momentum's cached sampler: the halo, the half halo and C4;
- the grid overlay's `udrLoadFace` (`lib/core/uniform-detail-abi.ts:143-152`), which tells the h store from the base by `textureDimensions`.

**Readers that cross blocks:**
- `fillUnitHalo` reads U up to 2 texels outside the block, and S faces at walls.
- The C4 cache reads a 1-tile halo around each block.

**The pressure side's ownership differs from the simulation's:** `buildRhs`, `project` and `toSimulation*` run on all-4h pressure owners and read the h faces of cut 4h faces (`fineVelocity`).

**Dense passes (L):**
- `buildDenseExtrapolationAuthority`, which writes D, then `encodeBaseRefresh`;
- `diagnostics`, `localSpeed`, `classify`;
- `copyFacesReference`;
- the host `copyWhole` (V → D) on the path without particle momentum.

### 8.5 Particles
Particles live in per-brick slabs: 17 words for each of the 64·B bricks. The capacity is one pool across all bricks, and a full pool is fatal (`noteReceipt`). Also sized by cap:
- `links`: 7 words a particle;
- display energy: 1 word a particle;
- seed list: up to 3·cap entries.

**Two couplings make shrinking cap unsafe without a check:**
- The deposit shares the source bank only while 1540·poolTiles ≤ the bank's bytes. Otherwise it moves into `bins`, adding 770 KiB a block.
- The paired extension and swept delta need the bank to be at least `companionBytesAt(F)`.

### 8.6 What the inventory decides
1. **h storage must exist beyond the h owners.** Several stages touch h texels of 4h tiles next to h tiles: phi resolve, seam publishing, transfer seams, momentum and forces seams, projection, remap, the U seam slots, and the pressure side's cut 4h faces.
   - So the fine brick set is the h bricks **dilated by one brick** (the uba dilation), plus the bricks of cut 4h pressure faces.
   - Open question: whether the detail ring (4h tiles within 3 tiles of an h tile) fits inside that dilation, or has to shrink first.
2. **Addressing is not in one place yet.** These compute h texels themselves:
   - the samplers (G2P, surface advect, overlay);
   - `umBodyOwnerVelocity`, `umLoadMixedFace`;
   - the raw 33-stride vertex users (`udRun_phi`, `udTexel_phi`, `arenaGather`, `refreshVertex`);
   - the inline vertex keys;
   - host upload and readback;
   - `readMixedTexture`, in about 20 tools and tests.

   Each must move to a brick-slot address in its channel's commit.
3. **A slot lookup must stay out of inner loops.** Without the kernel first staging its 27-neighbour record (`brick-arena-core` §3.9), a brick slot is a table read per texel.
   - The kernels that gather across seams need the records: authority, extension, `gridLiquidCell`, forces, momentum, `bNeighbour`.
   - The records need those kernels launched over brick lists. That launch code is what W2/W3 are rewriting.
4. **Migration order** (fewest readers and least addressing first):
   1. **Base-only, no addressing change:** gammaA (no reader) and volumeB (canonical only). pressureGeometry.target already is.
   2. **Particle capacity to bricks.** Host side; check the deposit and companion couplings.
   3. **Cell scalars onto fine bricks through the accessors,** in the order gammaB, surfaceA, volumeA, surfaceB. Needs items 1 and 3.
   4. **Vertex φ:** a 9³-a-brick store, with the vertex-keyed buffers re-keyed and the raw 33-stride users rewritten.
   5. **Faces last:** V, S, D, U and SU onto atlas slots with a 1-cell apron. These are the sampled channels with the most readers, and they carry the halo's and C4's block-crossing reads.
   6. **Drop the block-sized h textures:** retire the detail storage's resident generation.
