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
| **Measured (Phase 0)** live h tiles | **250–385** steady (≈ 30–50 bricks), 500 in the first second | pending | — |
| **Measured** live blocks (resident / pool) | **138–158 / 226** | pending | — |
| **Measured** 4h tiles | **73–80k** (left 53.7k, right 15–21k) | pending | — |
| **Measured** today, allocated | **3.53 GiB**: pool-at-h 3.30, pool-at-4h 0.11, dense-box 0.03, fixed 0.09 | fit: 22.1 GiB at the default 1442 blocks, 15.4 GiB at 1000 | — |

Measured per block: **15.45 MiB** (h 14.97, 4h 0.48), against the 10 MiB of §1.2. Per box cell 3.27 B, fixed 96.5 MiB (`artifacts/tier-budget/allocation-classes.json`). The pool is the memory: 93% of it is h storage reserved for every block.

**Biggest unknown: the catch floor.**
- At 5 mm the seep delivers about 1.8 L in 20 s (Torricelli: Cd·h²·√(2g·0.64 m)). Over 0.41 m² of floor, that is a film about one h thick.
- The `thin` criterion would put the whole floor at h, about 256 bricks at 5 mm and more at finer h.
- Phase 0 measures it. If it dominates, the thin criterion needs a minimum film-area rule, or a sub-h film goes to particles.
- **Measured (Phase 0, 5 mm, 20 s):** the right floor band (y < 8 h) holds 46–70 h tiles in the first second, then **1** from step 300 to step 1200. There is no h film. The jet and spray zone holds 130–300 h tiles, and the throat ±8 h holds about 81. The right chamber's liquid volume is not yet measured (see Phase 0 results), so this does not yet prove the film is thin rather than absent.

### 3.2 Frame (M1 Max, 1/60 s)

| Part | Target | Scales with |
|---|---|---|
| Fine stages (surface, particles, RFE and momentum at h) | the pour's per-kT rates after the speed-of-light work | live fine tiles (8 per brick) |
| 4h active stages | ≤ 1 ms | warm/hot blocks |
| 16h momentum and forces | ≤ 0.3 ms | live super-blocks |
| Pressure, all rows, warm-started | ≤ 2 ms | rows |
| Floor (census, plan, lists, pool commit) | ≤ 1 ms, independent of box and capacity | — |
| Surface extraction | ≤ 1 ms | surface blocks and bricks |
| **Measured (Phase 0) whole step today, 5 mm** | — | fenced `advanceTo` + `awaitFrameCompletion`, timing mode: **median 20.6 ms, p95 35.2, mean 22.9** over 1200 steps (steps 600–1200: median 20.0) |

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

### 3.4 Phase 0 results (2026-10-24)
**Setup.**
- Machine: Apple M1 Max, 32 GiB, Darwin 25.6.0.
- Node: v22.22.1 (`/Users/petersuggate/.nvm/versions/node/v22.22.1/bin/node`) with Dawn (`webgpu`).
- Solver: commit e218ecfe (main with 94960cdc and e0d8994f). Harness: `tools/probe-tier-budget.ts` (49d6e5c2, zone fix b964dbc6).
- Scene: `reservoir-seep` as committed.
- Profile: dynamic policy, detailBudgetPercent 5, regionTiering error, warm at h; quality balanced.
- ms are from the shared runner's timing mode only.

**Reports** (`artifacts/tier-budget/`):
- `seep5-r1-1200.summary.json`: 5 mm, 1200 steps, a sample every 30.
- `seep5-r2-check.summary.json`: 5 mm ×2 reservoir, 300 steps, check mode (counts and bytes only).
- `allocation-classes.json`: the bytes-by-class fit.

**What the harness reports per sample.**
- Live h and 4h tiles and resident pages, from the ownership work receipt.
- Pool status, and region heat (COLD/WARM/HOT/free).
- Particles; phi volume drift (the solver's `narrow-band-phi` truth).
- Allocated bytes split into pool-at-h, pool-at-4h, dense-box and fixed.
- A per-owner zone probe: h tiles left of the wall, at the throat ±8 h, on the right floor (y < 8 h) and in the right jet; 4h tiles left and right.

**Allocation classes.** Each allocation label is fitted as fixed + perBlock·blocks + perBoxCell·boxCells over three constructions (pool, pool + 64, a ×2 box). A per-block term of 64 KiB or more is pool-at-h.

**5 mm, 1200 steps (20 s).**
- Step ms: median 20.6, p95 35.2. The first second is about 2 ms slower (13–300: median 22.5).
- h tiles: about 500 for the first second, then 250–385. Bricks number 30–50, not ~500.
- Resident blocks: 138–158 of 226.
- Regions: about 1050 COLD, 20–80 WARM, 20–170 HOT, 600–700 free.
- Particles: peak 580, a steady 43 from step 390.
- Floor film: none at h (see §3.1).
- Construction: 252 s. Pipeline constants depend on capacity, so a new pool or box recompiles.

**Length ladder at 5 mm** (the tool's `--reservoir=N`: the left chamber N× as long, same throat, head and right chamber; the wall moves to x = 240N).

| | ×1 | ×2 | ×4 |
|---|---|---|---|
| Lattice | 512×256×64 | 752×256×64 | 1232×256×64 |
| Pool (scene) / resident | 226 / 138–158 | 370 / 236–246 | 670 / pending |
| Allocated (fit) | 3.53 GiB | **5.71 GiB** (fit 5.72): h 5.41, 4h 0.17, box 0.04, fixed 0.09 | pending (fit 10.27) |
| h tiles after the first second | 250–385 | **95–144** (throat 74–76, jet 19–68, floor 0) | pending |
| 4h tiles | 73–80k (left 53.8k) | **121–126k** (left 107.5k, right 10–15k) | pending |
| Regions COLD / WARM / HOT | ~1050 / 20–80 / 20–170 | 1857–1932 / 8–32 / 11–18 | pending |
| Particles | peak 580, 43 from step 390 | **0** throughout | pending |
| Step ms (timing mode) | median 20.6, p95 35.2 | pending (batched) | pending |

×2 ran 300 steps in check mode (`seep5-r2-check.summary.json`), so its ms do not count. Bytes per doubling are the pool's: +144 blocks, +2.18 GiB measured for ×1→×2. The fit predicts +300 blocks, +4.5 GiB for ×2→×4. The 4h tiles of the left chamber double with its length, while the h work does not grow; it shrinks, and ×2 has no particles at all. Whether ×2's throat flows at all is the discharge question below.

**2.5 mm.** Pending, batched behind the merge gates as a timing run (over 6 GiB). The fit predicts 22.1 GiB at the default 1442 blocks, which does not fit a 32 GiB machine beside the OS, and 15.4 GiB at an explicit pool of 1000.

**Open: discharge.** After the first commit, the probe's phi and velocity loads read empty (phi 0 in both ping-pong textures, throat face u 0). This happens even through the canonical-load form `UniformMixedDiagnostics` uses, so the right chamber's volume, the level and Cd against Torricelli (Q_T = h²·√(2g·0.6375 m) = 8.84·10⁻⁵ m³/s at 5 mm) are not measured yet. The window coordinates are fixed: owner origins are window lattice cells, and under an open top the frame moves after the first commit.

**Contradictions with the estimates.**
- A block costs 15.45 MiB, not 10, so today's 5 mm is 3.53 GiB, not 2.0.
- Live h work is about 10× smaller than estimated (30–50 bricks, not ~500).
- The catch floor does not go to h.
- The whole step today is about 20 ms. The §3.2 budget (≤ 1 + 0.3 + 2 + 1 + 1 ms plus fine stages) sits well below it.

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
