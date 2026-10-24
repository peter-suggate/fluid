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
