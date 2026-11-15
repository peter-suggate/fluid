# NB-FLIP structural redesign toward seep-m1max's architecture (lane X, 2026-11-15)

The user's direction: "i expect you to adapt the narrowband flip method at a structural level to be more like the
optimized version". The optimised version is seep-m1max (`lib/methods/seep/`). It runs the seep 5.06x faster than
NB-FLIP fenced, and 5.28x faster pipelined (`nbflip-port/seep-techniques.md`).

This doc is the plan the coordinator approves before any implementation lane starts. It covers:
- §1, a subsystem-by-subsystem structural diff with today's cost;
- §2, the target architecture and stage list;
- §3, the strategy;
- §4, gated migration milestones and the go/no-go results;
- §5, the lane split;
- §6, the expected end state;
- §7, decisions the coordinator has to make.

The data comes from:
- lane A's attribution (`nbflip-port/attribution.md`);
- lane G's gates and baselines (`nbflip-port-2026-11-15.md` §4);
- lane P2's move measurements (folded in from the coordinator, §1.4);
- this lane's addressing microbench (`~/code/me/fluid-scratch/lane-x/bench-addressing.mts`, §4.1);
- a feature inventory of the uniform tree (§2.9).

**The short version:**
- NB-FLIP is not slow because of any one kernel or its addressing. Its addressing chain costs 1.6x dense; a sparse
  brick pool would cost 1.3–1.5x.
- It is slow because one step is about 450 dependent dispatches over two representations of the same state:
  - 3D paged textures with h fine slots;
  - the brick arena, a copy for pressure and the surface.

  Around them sit about 100 dispatches of census, layout and residency to keep the two consistent. Every kernel runs
  at 12–17% occupancy.
- seep-m1max is one representation: brick-major 8³ buffers. It has a static brick set, about 70 real dispatches a
  step, kernels shaped for occupancy and a two-part submit.
- The plan is a new brick engine behind the NB-FLIP method id (strategy C, a hybrid). It has the seep's skeleton
  and NB-FLIP's physics formulas ported verbatim, onto one sparse brick pool with an incremental brick-set update.
  It reaches every gate scene through four gated milestones, then the uniform tree is deleted.

## 1. Structural diff, subsystem by subsystem

Costs are lane A's split-phase ms a step: seep 600, letters 120 and paddleboats 300 (attribution §2.1). Dispatch
counts are per counted step. "Seep" in the right-hand column means seep-m1max.

### 1.1 Storage and addressing

| | NB-FLIP today | seep-m1max |
|---|---|---|
| fields | 3D paged textures. A cell resolves through `umPageWord` → block → `umFineSlot` (the h slot) → a texel (`uniform-window-address.ts`, `uniform-fine-slots.ts`, `uniform-texture-pages.ts`). The 4h fields live in the block, the h fields in a fine slot of 4³ tiles. | Dense brick-major buffers: `cell = (brick << 9) \| local`, brick id by arithmetic (`seep-layout.ts`). |
| second copy | The brick arena (`uniform-brick-arena.ts`): 8³ bricks at levels 0–5 with 32-word records (27 neighbours, parent, owned-tile mask, key) and a channel stride of 520. Pressure and the surface metric run here, so every step copies texels in and publishes them back: `arenaPublishCanonical`, `arenaStore`, `ufx4`'s 32768-texel-a-page publish. | none |
| directory | Two-level toroidal page directory (a 64³ top table of 4³-page super-blocks, 64-word child tables) plus a flat 64³ window | none (dense box) |
| memory | 1.2–3.4 GB peak (lane G §4.4: 1187 / 1922 / 2023 / 3228 MB) | about 0.3 GB on the seep |

**Why the seep is cheaper.**
- It is one representation, so there are no copies and no publish passes.
- Its addressing costs no loads.

The addressing itself is the smaller part. §4.1 measures NB-FLIP's three-load chain at 1.6x dense on a
semi-Lagrangian advect, and a sparse two-level brick directory at 1.3–1.5x. The copies, the publishes and the stage
count they force are the larger part. They make up the canonical publication's 1.8 ms on the seep and 6.0 ms on
letters, and the 2–4 dispatches each that every consumer stage spends staging texels into bricks.

### 1.2 Tiers (h, 4h, 16h)

| | NB-FLIP today | seep-m1max |
|---|---|---|
| tiers | Each 4³ tile is owned at h (64 owners) or at 4h (1 owner). Pages are 32³ h = 8³ 4h cells. BULK16 pages hold 16h, 2³ cells a page (8 pressures a slot). The composite seam couples an h face to a 4h row over 2.5h. | Static composite h / 4h: classes DEEP, BULK, THROAT and CHAMBER fixed at scene build. The 4h bulk is a block of the dense box. |
| change | Every step: census, remap and promote/demote (§1.3) | never |

**Why the seep is cheaper.** Its tiers never move, so no step pays for tier bookkeeping.

The general method must move tiers: letters' drops, the dam's collapse and the paddle wakes all need h where the
liquid goes. So the target keeps NB-FLIP's tier rule and its granularity (4³ tiles, §7.1), and changes only how a
tier change is carried out (§2.3).

### 1.3 Activation, residency and census

| | NB-FLIP today | seep-m1max |
|---|---|---|
| work | About 100 dispatches a step (attribution §2.1, "layout / census / residency"). They cover census (13, 1.85 ms seep), page-pool commit (8), layout build (12), remap (9, 1.07 ms), frame-plan dilations, the local speed certificate (8, 0.57 ms), hanging taps (3), the residency certificate on 16³ pages with its closure, and the remap and phi-resolve twins. | none: the static classes and a whole-box `classify` (domain-sized, not allowed in the general method) |
| cost | seep 3.60 ms, letters 8.06, paddleboats 6.41 (10–14%). Remap, canonical publication and resolve add 1.07 + 1.79 + about 1 ms on the seep, and 2.43 + 5.99 + more on letters. | about 0.1 ms |
| defects | Lane D's dam residency failure (6 of 11 fenced 300-step runs stop at steps 178–247): a reader outside the certified closure | none |

**Why the seep is cheaper.**
- It has no dynamic set.
- NB-FLIP's dynamic set is expensive because it is rebuilt through several layers: page, block, fine slot,
  texture, then arena. Each layer has its own commit, remap, publish and certificate pass.
- NB-FLIP also has a separate arena enter/leave (`ubaUpdate`, `ubaLists`, `ubaLevel1`, `ubaRecord`, `ubaRelease`).
  That shows an incremental brick-set update works on this hardware. It just is not the only structure.

### 1.4 Particles

| | NB-FLIP today | seep-m1max |
|---|---|---|
| where | The narrow band (Ferstl et al. 2016, R = 4h, r = 2h) around every surface, plus sources | Only where needed: the chamber and the jet |
| storage | Slabs: two a brick (scratch and current), an overflow tail and a spill reserve. Stride 12 or 28 floats (APIC). | Binned per brick by a per-brick counting sort (clear, count, scan, scatter) |
| move | One launch, `advectParticlesDeferred`: RK3 trace, depth sampler, solid walk (`nbFinish`), rigid contact (`nbRigidContact`), pair separation and surface shift. 3.76 / 14.12 / 5.12 ms (11–19% of the step). About 27 ns a particle on the seep (137k particles) and 16 ns on letters (890k). | Split by class. Bins are bricks. |
| other | transfer and seeding: 44 dispatches, 2.93 / 9.79 / 4.60 ms | `p2g` and `g2p`: one launch each |

**P2's move measurements** (coordinator, 2026-11-15, letters step 120, about 930k particles, base f72b3ca6):
- The whole move is 15.1 ms (about 16 ns a particle).
- Without the separation it is 7.1 ms.
- A standalone separation pass is 3.6 ms (3.9 ns a particle).
- **Register cap.** The kernel is capped at 384 threads (12.5% occupancy). Four paths each set the cap on their own:
  - the velocity sampler;
  - the depth sampler;
  - `nbFinish`'s solid walk;
  - `nbRigidContact`.

  Stubbing all four lifts it to 512 threads. Removing `nbAdvect` as well gives 1024.
- **Separation is not a cap setter.** Pulling it out on its own is a net loss: −2.8 ms in the move against +3.6 ms
  for the extra pass (variant p1a: 12.34 + 3.59 = 15.92 ms against 15.13).
- **Latency-bound.** Every limiter is at 3–13%, so under the cap the kernel waits on memory latency.
- **History.** A pre-pass (04f725d1) saved about 9 ms at 1.1M particles and was folded back in b20f6e2d. That commit
  suggests computing the corrections into threadgroup memory within the same job.
- **Open hazard.** Variant p1b moves `nbSurfaceShift` into the pre-pass. With its check off, it collapses letters to
  about 7.9k particles, and the cause was not found. The move therefore has an unexplained dependency, and it must
  be root-caused before the move is split (M0, §4).

**Why the seep is cheaper.**
- Its move is split by class, so no thread carries the registers of a path it does not take.
- Its bins are bricks, so a workgroup's particles share one brick's neighbourhood.
- It has far fewer particles.

The first two transfer; the third does not, because the band is NB-FLIP's method.

### 1.5 Level set

| | NB-FLIP today | seep-m1max |
|---|---|---|
| stages | 49 dispatches, 6.29 / 18.08 / 9.10 ms, plus phi advection (21 dispatches, 3.49 / 5.76 / 4.10 ms) | `advectPhi`, `combine`, `massFit` ×4, `redistInit`, `redistSweep` ×2: 9 launches |
| what | Advection: semi-Lagrangian, cubic or trilinear. The solid-aware `umAdvected` walk; a deferred and staged trace with a tail (`advectDeferred`, `advectStagedTail`, `advectFinish`, `advectFine`, `advectOwners`). Reconstruction (`coupleFine`, `coarseCorners`, `surfaceClear`, splat, `couple`; Ferstl Eq. 4, φ ← min(φ′ + h, φ_p)). Then retirement, `clearAir` (airSeeds, airSpread 0–2, airGate, airClear), redistance within 3 tiles (`arenaMark`, `arenaGather`, `arenaCross`, `depthLists`, `arenaNearest`, `arenaDistance`, `arenaFar`, `arenaStore`), phi resolve ×5 and canonical publication. | One representation. The two-pass crossing-based redistance runs over the active bricks. |

**Why the seep is cheaper.**
- It has one copy of phi.
- Its stages are brick-local and fused.

Most of NB-FLIP's 49 dispatches move phi between texels, the arena and the twins, or rebuild work lists for each
sub-stage.

### 1.6 Pressure

| | NB-FLIP today | seep-m1max |
|---|---|---|
| system | Composite h / 4h / 16h ghost-fluid operator on the brick arena: band h rows, native 4h rows, BULK16 rows, and seams at 2.5h | Composite h / 4h, one brick system |
| solver | MGPCG: 43 dispatches a cycle, 4 encoded cycles, 178 dispatches. The tail is one-workgroup `pTail` (0.6–1.6% occupancy). Convergence is measured after pre-smoothing. The slow mode is at the band/native seam. Seep and dam are capped above target in 26–70% of steps. | MGPCG with a staggered Schwarz smoother and an exact coarsest solve (a 32-row Gauss–Jordan inverse once a step). The check runs before the V-cycle, and later slots are gated (14 a slot, 10 slots, mostly empty). |
| cost | solve 5.99 / 10.48 / 6.05 ms, setup 2.18 / 4.93 / 3.25 | about 2–3 ms on the seep |

**Why the seep is cheaper.**
- It converges in fewer cycles.
- Its coarsest solve is one mat-vec.
- Empty slots cost one validation dispatch each.

NB-FLIP's arena already has the right shape: 8³ bricks with 27-neighbour records. Port lane P1 is moving the
seep's smoother and coarsest solve onto it (§5).

### 1.7 Solids and rigid bodies

| | NB-FLIP today | seep-m1max |
|---|---|---|
| static solids | Cut cells with exact fractions and apertures. A 77-row slot record is held only where a fraction is not 1 (`uniform-mixed-solid.wgsl.ts`). The sources are the paged voxel mask, the bed-slope lane, the terrain heightfield and procedural SDF boundary bricks. Records are rebuilt on edits, relayout and new blocks. | Static voxel wall and throat folded into the classes |
| bodies | Two-way coupling through pressure. The body reaction is the transpose of the moving-wall flux in the RHS. Face fractions use 4 transverse samples; rounded plates use exact ray intervals. `couple` writes fixed-point `rigidExchange`, and the rigid integrator (`lib/core/webgpu-rigid-body.ts`) runs contacts, motors and integration. `markTiles` dilates the body tiles by swept reach. | none |
| cost | Spread through geometry (`geometryRegional`, MMU-limited at 41% on the dam), apertures and 4.78 ms of CPU `advanceTo` on paddleboats. The first submit waits 4.23 ms on paddleboats, because rigid CPU records come first. | none |

**Why the seep is cheaper.** It has no bodies, and its solids are static.

What transfers is the pattern: stamp geometry into brick channels once, when it changes, and let every kernel read
the channel.

### 1.8 Schedule and submission

| | NB-FLIP today | seep-m1max |
|---|---|---|
| dispatches | 445–477 a step, nearly all direct (attribution §3). 74 passes, 4 submits. The floor is about 7–25 µs for an early-exit launch: 184 seep dispatches of 20–50 µs come to 5.8 ms. | About 170 encoded, about 70 real. Gated slots, a two-part submit and encode-ahead. |
| CPU | `advanceTo` 3.2–4.8 ms. The first submit waits 2.2–4.2 ms. | sub-ms |
| occupancy | 12–17% time-weighted, ALU 17–22%, no kernel above a 44% limiter (attribution §5) | shaped per kernel |

**Why the seep is cheaper.**
- It encodes a third of the dispatches and runs a sixth of them.
- Its CPU work is ahead of the GPU.

A dispatch on this stack costs 3.5 µs of GPU even when empty (seep-techniques §1). At about 450 dispatches,
NB-FLIP pays at least 1.6 ms, and the measured early-exit floor puts it at 3–5 ms.

### 1.9 Where the gap is

Lane A's ranked list of tweaks totals 10–20% a step. The rest of NB-FLIP's cost against the seep is structural:

| cause | how much it explains | evidence |
|---|---|---|
| Two representations and their copies: texel ↔ arena, twins and publishes | canonical, publish and store 2–6 ms; remap and resolve 1–3 ms | §1.1, §1.5; brick-resident B4's 32768-texel-a-page publish |
| The tier and residency machinery's layers | 3.6–8.1 ms | §1.3 |
| The stage count | 450 dispatches, at least 1.6 ms of floor, and 5.8 ms in 20–50 µs launches on the seep | §1.8 |
| Kernel shape: register caps, latency at 12% occupancy | the move 11–19%; every other kernel at 12–17% occupancy | §1.4; attribution §5 |
| Pressure cycles | 4 cycles where the seep's solver needs 1–2 | §1.6 |
| Addressing | 1.6x dense on stencils, about 1.1x on the move (§4.1) | this lane's microbench |

Addressing is the one item a redesign cannot remove entirely: a sparse pool is not dense. §4.1 shows the
residual is small, and that staging a brick in workgroup memory removes most of it.

## 2. Target architecture

**One representation, one brick set, one schedule.**

Every field of every tier lives in a sparse pool of brick-major 8³ bricks. This is the brick arena's ABI, which
becomes the storage itself rather than a copy. One incremental brick-set update replaces census, layout, residency,
remap and publication. Particles are binned per h brick. The step is a fixed list of about 135 encoded dispatches
(about 75–85 real), shaped for occupancy and submitted in two parts.

### 2.1 Storage: a sparse brick pool sized per scene

**Brick kinds.** Each kind is 8³ cells, stored brick-major as `slot << 9 | local` with the channel stride of 520
(512 aliases cache sets; the arena measured 2.5x slower staging at 512).

| kind | cell | covers | holds | ownership |
|---|---|---|---|---|
| F (fine) | h | 8³ h = 2³ tiles | u, v, w; u′ and the FLIP delta; φ, φ′; open fraction and 3 apertures; the pressure rows' vectors (x, r, z, p, A p, rhs); flags | An 8-bit owned-tile mask (which of its 2³ tiles are h-owned) |
| C (coarse) | 4h | one page, 32³ h = 8³ tiles: a C cell is a tile | the same channels at 4h | A 512-bit mask, 16 words: which tiles are 4h-owned and which are refined into F |
| B (bulk) | 16h | 2³ 16h cells a page, packed 64 pages a slot (as `BULK16_PAGE` today) | faces as 36 means, 8 pressures, the 9³ vertex block | NB-FLIP's BULK16 rules unchanged |

The 4³ tile granularity of h / 4h ownership is kept exactly, because the gates are two-sided (§7.1). An F brick
whose 2³ tiles are not all h-owned keeps its non-owned tiles as the ghost / seam values the C rows supply, as the
arena does today with `UBA_SEAM`.

**Directory: two levels, one indirection per brick, not per tap.**
- The top level is a page word per resident page, giving the page's C slot and its child table.
- The child level is a 64-word table of F slots, one per 8³ h brick in the page (4³ bricks = 32³ h = one page).
- The page window is toroidal, as today's directory (`uniform-window-address.ts`): its size comes from the scene's
  lattice in pages, never the cell box, so no allocation is domain-sized. Vast or moving-window scenes use the
  existing toroidal window.

This is variant T2o in §4.1, and it costs 1.43x dense when every tap resolves.

Two access rules follow from §4.1:
- **Particle kernels stage.** A workgroup per bin stages its brick and a one-cell ring in workgroup memory once,
  through the brick's 27-neighbour record (the arena's 32-word record: 27 neighbours, parent, owned-tile mask,
  key). A tap is then a workgroup-memory read. Measured at 0.82x dense: faster than the seep's dense move.
- **Cell kernels address directly.** Advect, stencils and sweeps resolve through T2o, a tap at a time. Staging a
  cell kernel measured 1.73x, worse than T2o's 1.43x: the 16 KB of workgroup memory costs occupancy, and the
  cache already serves the reuse.

**Slot order matters 1.15–1.25x.** Shuffled slots cost that much over spatial ones (R and BT against T2o and BTo in
§4.1). The free stacks therefore hand out slots page-locally where possible, and M1 measures how far that holds up
under churn.

**Pool size.**
- Pools are authored per scene from the measured peak, as `pagePool.blocks` and `tiers.bricks` are today. There is
  no runtime growth.
- Exhaustion latches an explicit failure bit (FAIL_POOL, following the seep's PRESSURE_CAP bit), never a silent
  clamp.
- The M0 measurement (§4.1) fixes each scene's pool.

**Memory estimate (§4.1):**

| scene | F peak | C pages | particles | estimate | today |
|---|---|---|---|---|---|
| seep | about 1.5k | about 200 | 220k | about 0.15 GB | 1.19 GB |
| dam | about 4k (its 4096-brick ceiling) | about 64 | 1M | about 0.45 GB | 1.92 GB |
| letters | about 2.5k (9.8k fine tiles at step 120) | about 400 | 1M | about 0.4 GB | 2.02 GB |
| paddleboats | about 3k | about 300 | 310k | about 0.25 GB | 3.23 GB |

### 2.2 Tiers as composite bricks

- **h ↔ 4h.** A page's C brick always exists while the page is resident. F bricks exist where at least one of their
  tiles is h-owned, or within a stencil ring of one.
- **One system.** The composite operator, the momentum update and the extension read the owner of every tile from
  the masks. There is one system; no copy is kept per tier.
- **16h.** A page demoted to BULK16 frees its C and F slots back to the stacks, and its B record carries NB-FLIP's
  36-face and 8-pressure restriction. Promotion prolongs as `uniform-bulk-page-transfer.ts` does today.
- **Cold regions.** Region heat (16h: COLD, WARM, HOT) and cold freeze keep their rules. A frozen region's bricks
  drop out of the active lists (§2.3), so the work skip is the list itself, and no stage early-exits per brick.

### 2.3 The incremental brick-set update

This replaces the census, page-pool commit, layout build, remap, frame-plan dilations, both certificates, hanging
taps, phi resolve, canonical publication and the arena's separate enter/leave: about 100 dispatches and the copies.

It is 4 dispatches, placed after the particle rebin and before every grid stage (§2.7).

| # | kernel | does | shape |
|---|---|---|---|
| 1 | `setMark` | One lane per tile of the active F and C bricks, plus the ring. Runs NB-FLIP's census verbatim: the departure-box RK2 trace over the per-brick velocity bound; the shape, thin, impact, approach and solid-contact criteria with the app profile; heat and hold. Takes the particle bin counts, φ, and the body and source marks. Decides each tile's tier for the next step. Marks bricks to enter, within the CFL reach `ceil(bound·dt / 8h) + 1` of liquid or of a particle, and bricks to leave. | over the active list; 64-lane groups; no stencil; high occupancy |
| 2 | `setUpdate` | Pops and pushes the free stacks with atomics. Writes the child tables and page words. Rebuilds the 27-neighbour records of entering bricks and of their neighbours. | over the enter/leave lists |
| 3 | `setTransfer` | Tier changes with NB-FLIP's conservative remap math, verbatim (`uniform-mixed-remap.ts`): volume and φ in place; faces through the brick's own scratch; solids by capacity; only within one tile of a width change. Covers prolong 4h → h for entering tiles, restrict h → 4h for leaving tiles, and BULK16 promotion and demotion. | a workgroup per changed brick, staged in workgroup memory |
| 4 | `setLists` | Compacts the active lists and writes their indirect arguments. The lists are F, C, B, seam, near-solid, near-body, the surface band and cold. | one pass over the pool's live slots |

**Residency is the brick set.**
- A tap that misses the directory reads "not resident". By construction, a non-resident brick is air outside the
  CFL reach, so every reader agrees on it.
- Because there is no second structure to disagree with, lane D's class of defect (a reader outside a certified
  closure) cannot happen.
- Particles may go anywhere in a step: the rebin marks the bricks they reach, and the update runs before any grid
  stage reads them.

**What this removes.**
- the page pool and block layer;
- fine slots;
- 3D textures;
- the hanging-tap table;
- the speed and residency certificates;
- the phi-resolve twins;
- the arena's copy-in and publish.

The arena's enter/leave kernels (`ubaUpdate`, `ubaLists`, `ubaRecord`, `ubaRelease`) are the starting point for
kernels 2 and 4.

### 2.4 Particles binned per F brick, the move split by register budget

The particle record keeps NB-FLIP's layout and physics: position, velocity, `before`, APIC cx/cy/cz, lifetime and
mass bits, at a stride of 12 or 28 floats. Bins are F bricks; seep-m1max's per-brick counting sort replaces the
slabs, overflow tail and spill.

| # | kernel | does | why separate | shape |
|---|---|---|---|---|
| 1 | `pTrace` | A workgroup per F-brick bin. Stages the brick's velocity with a one-cell ring (11³ × 3 = 16 KB of workgroup memory). For each particle: G2P with the FLIP 0.95 blend, then the RK3 trace through the staged field, falling back to the record path outside the ring. Writes the new position to the compact-position stream and a class word (ballistic, interior, near-surface, near-solid, near-body). | Only the velocity sampler. P2: without the other three cap setters the kernel reaches 512–1024 threads. | 128 lanes; target ≥ 33% occupancy |
| 2 | `pBand` | A workgroup per bin. Stages the bin's and its 26 neighbours' compact positions and φ in workgroup memory. Runs the depth sampler, the surface shift (`nbSurfaceShift`) and pair separation (`nbSeparation`) against the bin and its neighbour bins. Writes corrections into the motion stream of the links buffer, which is free during the move (P2). | The high-occupancy pass P2 asks for. Pairs come from bins, not a hash. b20f6e2d's "corrections into threadgroup memory" is this. | 128 lanes |
| 3 | `pSolid` | Indirect over the compact near-solid and near-body list from `pTrace`. Runs `nbFinish`'s solid walk and `nbRigidContact`, verbatim. | The other two cap setters, paid only by the particles that need them | 64 lanes |
| 4–6 | `pCount`, `pScan`, `pScatter` | The seep's per-brick counting sort into the new bins. `pCount` also marks bricks to enter for `setMark`. Retirement (age, sparse) and the cell cap merge happen at `pScatter`. | – | 3 launches, high occupancy |

**The p1b hazard.**
- Splitting the move changes when `nbSurfaceShift` reads positions: after the trace, from the compact stream, rather
  than inside the same thread.
- P2's p1b, which made that move, collapsed letters to 7.9k particles with its check off.
- M0 root-causes it before `pBand` is written (§4).
- Until then the design assumes the dependency is real and keeps the shift reading what it reads today. The
  surface shift keeps the pre-move bank (`particles[]`, which the move does not write; attribution §6 item 1),
  which `pBand` reads unchanged, and only the positions it writes move to the motion stream.

**Seeding.**
- `pSeed` runs Ferstl's inner-overlap reseed, the sources' whole-band seed (drops and the inflow plug) and Zhu &
  Bridson jitter.
- A workgroup per bin appends into its own bin's tail.
- It is one launch, fused after the level-set reconstruction (§2.5).

### 2.5 Level set

| # | kernel | does |
|---|---|---|
| 1 | `phiAdvect` | Semi-Lagrangian φ, cubic or trilinear per NB-FLIP's fine and coarse settings, over the F and C lists in one launch (the tier comes from the list entry). A lane per cell, addressed directly through T2o (§2.1). |
| 2 | `phiSolid` | Indirect over the near-solid cells. Runs NB-FLIP's solid-aware walk (`umAdvected`'s wall reach, `solidClosed`, `solidClear`) on those cells only. Kept apart so Metal never inlines the walk into the main advect: brick-resident's S1 took 640 s to compile cold. |
| 3 | `phiUnion` | Reconstruction, Ferstl Eq. 4 φ ← min(φ′ + h, φ_p): a workgroup per bin, splatting into its own brick plus a ring with workgroup-memory atomic min, then one write. Also covers `coupleFine`/`couple` and the coarse corners. |
| 4 | `pSeed` | §2.4 |
| 5–7 | `redistInit`, `redistSweep` ×2 | Crossing-based redistance within 3 tiles, NB-FLIP's crossing guard "previous". Retirement and `clearAir` (the air-seed spread and gate) fold into the init. The second sweep writes the coarse φ for C bricks and presentation, and the volume gauge's sums by atomics. |

That is 7 dispatches in place of 49 + 21.

**Volume control** is unchanged in physics: a uniform divergence in the pressure RHS from a 4-cell gauge band, with
secant Newton rounds and a rate limit of 4/s.
- The gauge's per-round volume sums are taken in the second sweep's pass. The probe offsets are φ shifts, so every
  probe is summed in one pass.
- The rate is computed redundantly by every workgroup of the operator kernel, so no single-lane pass is needed.

### 2.6 Momentum, forces, extension

| # | kernel | does |
|---|---|---|
| 1 | `p2g` | A workgroup per bin. Transposed lanes accumulate into workgroup memory, then one brick write: weights, the FLIP keep, properties, and seam owners. Replaces `transferDeposit`, `transferRefine`, `transferProperties`, `transfer`, `transferOwners` and `transferSeams`. |
| 2 | `uAdvect` | Semi-Lagrangian momentum for owners without a particle estimate (the 4h interior, C bricks), NB-FLIP's sampler, addressed directly. Forces fused in: gravity on faces with occupancy, the inflow override, and viscosity and capillarity when they are not zero (`uniform-mixed-forces.ts`). |
| 3 | `uExtendPre` | NB-FLIP's FLIP-keep extension ahead of pressure |
| 4–5 | `uExtend` ×2 | Fast-sweeping extension after projection (Zhao 2005; `uniform-frame-extension.ts`): 2 sweeps a brick at h and at 4h in the same launch. A workgroup per brick, so `ufx4`'s per-page publish disappears. |
| 6 | `uDelta` | `recordDelta` / `flipTail`: the grid delta the next `pTrace` reads |

### 2.7 Pressure

The same composite discretisation:
- band h rows, native 4h rows and BULK16 rows;
- seams coupled over 2.5h, V / (2.5h²), with the native row taking the sum of its 16 h faces;
- ghost-fluid θ, with `UNIFORM_MIXED_THETA_MIN` unchanged;
- apertures, and body wall flux in the RHS;
- the volume-control divergence.

Only the solver changes, and it changes onto the seep's structure.

| # | kernel | does |
|---|---|---|
| 1 | `operator` | Builds rows and the RHS from the channels, fused with geometry and θ: replaces `geometryRegional`, `pRows` and the authority passes. Body face fractions and wall velocity (4 transverse and 8 dual-cell samples) come from the stamped channels. |
| 2–3 | `coarsen`, `coarsenTop` | Aggregation over brick blocks. Level 1 is 2h of F. Level 2 merges C rows: the seam is in-level from here up, so the slow seam mode is smoothed where it lives. Levels 3–4 are 8h and 16h, where B rows merge. Levels go up to 128h (the existing superblock tail). `coarsenTop` inverts the coarsest operator once a step by Gauss–Jordan (SPD, no pivoting, a FAIL_PIVOT latch). At 128h the coarsest level is 6–8 rows on all four scenes, and it is capped at 64 rows; a scene over that adds a level at build time, never at run time. |
| 4 | `pcgGuess` | warm start, as `warmStart` today |
| per slot | `update`, `check`, `down` ×4, `coarse`, `up` ×4, `precond`, `direction` (12) | The check runs before the V-cycle (S2q). The smoother is a staggered Schwarz over brick blocks that straddles the seam (P1 item 10). The exact coarsest step is one mat-vec. Later slots read their workgroup counts from the check's records (S2e). |
| last | `project` | Projection, fused with the per-brick velocity bound for the next step's `setMark` and the body pressure for `couple` |

**Slots.** 8 slots × 12 = 96 dispatches encoded. The seep's solver converged in 1–2 slots; NB-FLIP today runs
3.0–3.95 of 4 cycles. M2 measures the real figure; the estimate is 30–40 real.

### 2.8 Solids and bodies

**Static voxel solids.**
- The source geometry stays where it is: the voxel mask, procedural boundary bricks, the bed slope and terrain. It
  is still hit-tested and edited through the shared geometry path, so AGENTS.md holds by construction.
- `solidStamp` writes each brick's open fraction and 3 apertures channels from the source when a brick enters, on
  an edit (only the touched tiles plus their 26 neighbours, as `editMixedSolids` does), and on relayout.
- The 77-row slot record is replaced by those 4 channels, held only in F bricks that have a cut. This is the same
  "only where it is not 1" economy at brick granularity. A C brick's cuts follow the existing "4h owners never reach
  the helpers" rule: promotion keeps liquid one tile from cut cells, and is unchanged.

**Rigid bodies and motors.**
- `bodyStamp` (one launch over the near-body list) writes the swept body's face fractions and wall velocity into
  the F channels each step. It replaces `encodeBodies`/`encodeCoarse` and the bodyCuts atlas reads.
- `couple` reduces force and torque per body with atomics, in NB-FLIP's fixed-point 1e6 `rigidExchange` format.
  It reads the projected band pressure: `umBodyPressure` becomes a channel read.
- The rigid integrator (`lib/core/webgpu-rigid-body.ts`: contacts, motors, integration) is unchanged and shared.
  Its CPU records are cached and rewritten only when strengths change (P2 item 7). The prescribed-motion sample and
  roster stay on the CPU, written into the one param ring.
- Bodies must be at h wherever liquid can reach them: `setMark`'s body-contact criterion, as `markTiles` with its
  separating-axis test and swept dilation.

**Emitters and sources.**
- Drops (spheres, disks, letters) and the inflow plug stay in params, read inside `phiUnion`, `pSeed`, `uAdvect`'s
  force step and `setMark`'s source tiles, exactly as today.
- Pages are pinned one step ahead by the same host rule.

### 2.9 Feature inventory: everything must keep working

From the uniform tree's inventory: rigid bodies and motors; voxel solids and editing; emitters; volume control;
tier adaptivity; the narrow band; the level set; extension; presentation; the gate scenes.

| feature | where today | target |
|---|---|---|
| Two-way rigid coupling, motors, contacts, picking | `uniform-mixed-bodies.ts`, `lib/core/webgpu-rigid-body.ts`, `rigid-motor.ts` | §2.8. The integrator is unchanged. |
| Voxel solids, cut cells, procedural bricks, bed slope, terrain, editing, sculpt collision | `uniform-mixed-solid.wgsl.ts`, `uniform-solid-boundary.wgsl.ts`, HOST scene apply 3440–3523 | §2.8. Source geometry and the edit path unchanged; channels restamped. |
| Inflow nozzle, scheduled and interactive drops, `editFluid`, `injectLiquidBall` | `inflow-boundary.ts`, `uniform-source.wgsl.ts`, HOST 2708–2715 | params, unchanged |
| Volume control | `uniform-mixed-surface-volume.ts` | §2.5 |
| Census, region heat, cold freeze and cold frame, BULK16, remap | `uniform-mixed-dynamic.ts`, `uniform-region-heat.ts`, `uniform-cold-*`, `uniform-bulk-*`, `uniform-mixed-remap.ts` | §2.3. Criteria and remap verbatim. |
| Narrow-band FLIP or APIC, seeding, separation, surface shift, retirement | `uniform-narrow-band-flip.ts`, `-surface`, `-splat`, `-air`, `-apic` | §2.4, §2.5 |
| Moving window, open-world terrain | `uniform-moving-window.ts`, `uniform-open-world-terrain.ts` | The toroidal page window as today; terrain into `solidStamp` |
| Detail policy and focus, editor focus, tiers.start, refinement regions | `uniform-detail-*`, `editor-focus` | `setMark` inputs (pinned tiles) |
| Presentation | contract 241–308, `uniform-mixed-presentation.wgsl.ts` | §2.10 |
| QA stage audits, receipts, failure bits | HOST 472–504, `uniform-mixed-frame-status.ts` | the same status words; audits read the pool |

**Unported features fail explicitly.** Until M6, the new engine refuses any scene that uses a feature it has not yet
ported. The method then runs that scene on the old engine. This follows AGENTS.md: "unknown kinds must fail
explicitly".

### 2.10 Presentation

The renderer reads the pool directly:
- `volumePageSource` and `denseLevelSetVolumeSource` become views of the directory and channels;
- `surfaceFieldTexture` becomes an adapter that contours φ from F, with C where no F brick exists.

**Interim.** Where the renderer needs a 3D texture before its brick path exists, one copy a frame into a
presentation texture is allowed. It is presentation only, not simulation (M6 removes it). The per-step live copies
of pressure, φ and the band (FRAME 1307–1312) go.

### 2.11 Schedule, submission and CPU

**Target stage list** (one step; "real" means a launch that does work in a typical step):

| stage | kernels | encoded | real | occupancy shape |
|---|---|---|---|---|
| head | `bodyStamp` (bodies only), `solidStamp` (edits only, gated) | 2 | 0–1 | 64 lanes, list-indirect |
| move | `pTrace`, `pBand`, `pSolid`, `pCount`, `pScan`, `pScatter` | 6 | 6 | 128-lane bins, workgroup memory 16 KB; `pSolid` 64 lanes on a compact list |
| brick set | `setMark`, `setUpdate`, `setTransfer`, `setLists` | 4 | 4 | 64 lanes; `setTransfer` a workgroup per changed brick |
| level set | `phiAdvect`, `phiSolid`, `phiUnion`, `pSeed`, `redistInit`, `redistSweep` ×2 | 7 | 7 | a lane per cell, direct T2o; `phiUnion` a workgroup per bin, staged |
| velocity | `p2g`, `uAdvect`, `uExtendPre` | 3 | 3 | `p2g` a workgroup per bin, staged; the others a lane per cell |
| pressure setup | `operator`, `coarsen`, `coarsenTop`, `pcgGuess` | 4 | 4 | 128 lanes; `coarsenTop` one workgroup over 64 rows at most |
| pressure slots | 8 × 12 | 96 | 30–40 | brick-block Schwarz; the check is one workgroup |
| projection | `project` | 1 | 1 | 128 lanes |
| tail | `uExtend` ×2, `uDelta`, `couple` (bodies), rigid integrator (unchanged, about 5) | 9 | 3–9 | – |
| **total** | | **about 132** | **about 60–75** | |

That is about 450 dispatches down to about 132 encoded and 60–75 real: the seep's order of magnitude (about 170
encoded, about 70 real). At lane S's 3.5 µs, the floor falls from about 1.6 ms to 0.46 ms. Most of the 20–50 µs
launches (5.8 ms on the seep) go with the stages they belonged to.

**Submission.**
- A two-part submit: part 1 is the head through the rebin; part 2 is the brick set onward.
- Encode-ahead as the seep.
- **One param ring.** The CPU writes every per-step value (dt, the volume rate inputs, body poses, motor records
  when changed, source params) into the ring before part 1 is submitted. No `queue.writeBuffer` is issued between
  the parts. This removes P2's parked hazard (writes issued after the split land after the census), because the new
  schedule has no write after the head.
- Readbacks stay lagged by 2 frames (`UNIFORM_MIXED_RECEIPT_RING`).
- There are no CPU solves, fallbacks or per-step CPU stages.

**Two compile rules.**
- No kernel inlines the solid walk. `phiSolid` and `pSolid` carry it.
- Every kernel's WGSL is budgeted to stay under a cold compile of 30 s. This is checked in M1.

### 2.12 Physics changes (stated)

The target is the same discretisation. Nothing in §2 changes:
- a stencil, the tier rule or granularity;
- the Ferstl or seeding parameters;
- FLIP 0.95;
- the composite operator;
- θ;
- the remap;
- the coupling formulas;
- dt.

The changes below can still move observables, and lane G's gates judge each one:
1. **Pressure converges more often.** Seep and the dam end above target in 26–70% of steps today. A solver that
   reaches the tolerance changes the iterate within it: "iterate-changing" in seep-techniques' classes. It is the
   same discretisation, solved closer.
2. **Texture filtering becomes manual trilinear.** `textureSampleLevel` is used in `uniform-cached-momentum.wgsl.ts`,
   `uniform-fine-brick-store.ts` and `uniform-mixed-surface.ts`. Hardware filtering on Apple GPUs interpolates with
   reduced-precision weights. Manual f32 weights are at least as accurate, and the difference is below the gates'
   S, but it is stated here.
3. **Summation order** changes in p2g, the splat and the reductions. NB-FLIP is already nondeterministic.
4. **The particle order** within a bin changes the pair-separation order. Separation is order-dependent today
   through the slab order too.

Nothing here is a clamp, a special case or a substep.

## 3. Strategy: (C) a hybrid, a new brick engine behind the NB-FLIP method id

| | (A) restructure in place | (B) grow the seep core | (C) hybrid |
|---|---|---|---|
| what | Convert `lib/methods/uniform/` step by step to the target | Extend `lib/methods/seep/` into the general method | A new engine, `lib/methods/narrow-band/`, built on the seep's skeleton: record(), slots, two-part submit, counting sort, kernel shapes. NB-FLIP's formulas are ported verbatim as pure WGSL functions with no addressing in them. The arena ABI and P1's solver are adopted. |
| lines | 55k today (172 files), with an estimated 30–40k touched | 4.1k → about 20k, with the seep's chamber physics replaced | about 17–20k new, of which about 6k is ported formulas, plus about 3k of tests |
| risk | High. Every intermediate state keeps both representations consistent; this is the dual-representation tax. Two attempts gave 2–10%: brick-resident 31.6 → 31.0 ms; P1/P2 tweaks at 10–20%. B4's publish visited 32768 texels a page. S1 compiled for 640 s. | High. The seep is dense, with a domain-sized `classify`, static classes and a different chamber discretisation (particle-carried with a mass fit, van der Pijl 2005; outlet emission; drift about 0%). Two-sided gates fail a more conservative candidate (§7.1). Every NB-FLIP feature would be new code anyway. | Medium. Formula-level parity is testable against the old engine on frozen states. Each milestone is gated. The old engine stays as the fallback until M6. |
| first gated scene | weeks, with no clear point at which the copies go | the seep only, and with different physics | dam at M2, in about 2 weeks of lane time (§4) |
| end state | still two representations unless everything is redone | not NB-FLIP | NB-FLIP: the same method id, every scene, the same physics within the gates |

**Why C.**
- The cost is structural (§1.9), so a structure has to be replaced, not tuned.
- In place, every step of the replacement pays for both representations until the last one lands. The brick-resident
  attempt and P2's results are that tax measured.
- The seep is the right skeleton but the wrong physics.
- C takes the seep's skeleton and the uniform tree's formulas. The formulas are the hard-won part, so they are
  ported unchanged, and each is checked against the old engine on a frozen state.

**Reuse.** The engine reuses rather than rewrites:
- the brick arena's record and stride ABI (`uniform-brick-arena.ts`) and its enter/leave kernels;
- P1's seep-structured MGPCG on the arena;
- the rigid integrator;
- the solid sources and the edit path;
- the census criteria, remap math, Ferstl reconstruction, seeding, separation and surface shift, force and coupling
  WGSL as functions;
- the host contract (`method-contract.ts`).

**Switch.**
- The method id `narrow-band-flip` keeps its name. A method parameter `engine: "uniform" | "brick"` selects the engine.
- The default flips to `brick` per scene as each passes its gates, and for every scene at M6.
- Then the uniform tree is deleted, and the parameter with it.

## 4. Migration milestones and go/no-go experiments

### 4.1 Go/no-go results (M0, this lane, check slot)

**1. The cost of one-indirection addressing** (`~/code/me/fluid-scratch/lane-x/bench-addressing.mts`).

The setup:
- the seep's box of 64 × 32 × 8 bricks;
- a slab of 2048 active bricks (1,048,576 cells);
- 262,144 particles, 128 a brick, sorted by cell.

The kernels:
- `adv` is a semi-Lagrangian MAC advect a cell: 3 components, each a sampleU plus a backtrace sample, 96 taps;
- `pmove` is Ralston RK3 with 3 sampleU calls, 72 taps a particle.

The variants differ only in how a tap is addressed:
- **D:** dense brick-major, the seep.
- **BT:** a dense brick table with shuffled slots.
- **BTo:** the same table with spatial-order slots.
- **T2o:** a two-level brick directory with spatial slots; memory ∝ resident groups.
- **R:** a 27-neighbour record a slot (shuffled).
- **H:** an open-addressed hash.
- **T:** NB-FLIP's chain (super-block → child → fine slot → brick, 3 loads).
- **S:** the target, a workgroup a brick that stages an 11³ ring of the three components through R's record once,
  then reads taps from workgroup memory.

Six runs. The first four were on a shared slot, where the move column varied up to 2x between runs; the adv column
was stable to ±5%. The last two (D, S, R, T2o, T in both orders) were quiet, and their move figures (D 3.54, T2o
3.83, T 4.16) are the reliable ones:

| variant | adv ns a cell (×D) | move ns a particle (×D) |
|---|---|---|
| D | 1.11–1.12 (1.00) | 3.53–3.57 (1.00) |
| BT | 1.52–1.54 (1.36–1.40) | 3.77–5.60 (1.07–1.57) |
| BTo | 1.48–1.49 (1.30–1.33) | 3.73–4.46 (1.05–1.25) |
| T2o | 1.60–1.63 (1.34–1.46) | 3.83 quiet (1.08); 3.9–7.2 shared |
| R | 1.85–1.88 (1.66–1.68) | 3.73–4.94 (1.06–1.20) |
| H | 2.85–2.92 (2.55–2.62) | 4.70–6.99 (1.33–1.89) |
| T (NB-FLIP) | 1.77–1.80 (1.59–1.62) | 4.16–8.06 (1.18–2.26) |
| S (target, staged brick) | 1.92–1.93 (1.73) | 2.90–2.93 (0.82–0.83) |

**Verdict: go.**
- **Addressing is not NB-FLIP's gap.** Its own chain costs 1.6x dense on the stencil kernel. NB-FLIP's move at 16–27
  ns a particle is 4–7x a bare RK3 through that same chain (4–8 ns). The rest is stages, register caps and the
  dual representation.
- **A sparse pool costs 1.3–1.5x per tap** when every tap resolves.
- **Staging the brick wins for particles and loses for cells.**
  - For particles it measured 0.82x dense. Each brick serves 128 particles × 72 taps from one 1331-cell stage.
  - For cells it measured 1.73x, against T2o's 1.43x. The 16 KB of workgroup memory limits resident workgroups,
    and the cache already catches the cell stencil's reuse.
  - The design follows this: particle kernels stage, cell kernels address directly (§2.1).
- The last two runs were on a quiet slot (1.1 GB in use at start). Both variant orders agree to 1%.
- **Directory choice: T2o.**
  - Memory ∝ resident pages.
  - No domain-sized table.
  - It fits the existing toroidal window.
  - The dense brick table (BT/BTo) is equally fast but domain-sized: rejected under "memory must scale".
  - The hash costs 2.6x: rejected.

**2. Brick-pool peak memory per scene.** This is an estimate from measured counts, to be confirmed in M0 by reading
each baseline run's peak h-tile, page and particle counts from its receipts:
- letters had 9.8k fine tiles and 937k particles at step 120 (attribution §3);
- the dam ceiling is 4096 bricks and 1.71M slab records;
- particles are at the §2 peaks.

The pool's per-brick cost:
- an F brick of 520 cells × 26 channels × 4 B is about 54 KB;
- particles cost 48 B, or 112 B with APIC, double-buffered for the counting sort.

| scene | F bricks | F MB | C + B MB | particles MB | total | today |
|---|---|---|---|---|---|---|
| seep | 1.5k | 81 | 11 | 21 | about 0.15 GB | 1.19 GB |
| dam | 4.1k (ceiling) | 221 | 3 | 96–164 | about 0.4 GB | 1.92 GB |
| letters | 2.5k | 135 | 22 | 96 | about 0.3 GB | 2.02 GB |
| paddleboats | 3k | 162 | 16 | 30 | about 0.25 GB | 3.23 GB |

**Verdict: go,** with a 4–10x margin to today on every scene. Memory is not a constraint on the design, and frame
time takes priority (memory "step time over memory").

**3. The move kernel at seep occupancy.**
- P2's data (§1.4) answers the occupancy half. The four cap setters alone hold the move at 384 threads; the trace
  alone allows 1024.
- The microbench answers the throughput half. A bare RK3 runs at 3.5 ns a particle dense, and at 2.9 ns
  staged in workgroup memory through a churned (shuffled) pool.

The projection for letters at 930k particles:

| kernel | ns a particle | ms on letters |
|---|---|---|
| `pTrace` | about 4.5: the staged trace at 2.9, plus G2P's two staged samples | about 4.2 |
| `pBand` | separation's standalone 3.9 ns from P2, less what bins in workgroup memory save; say 2.5–3.9 | 2.3–3.6 |
| `pSolid` | about 10 ns × the near-solid fraction (est. 10%) | about 0.9 |
| rebin | about 1 ns × 3 | about 1–1.5 |
| **total** | | **about 8.5–10 ms, against 15.1 today** |

**Verdict: go,** with one condition: the p1b root cause (M0, below) before `pBand` lands.

### 4.2 Milestones

Each milestone is gated by:
- lane G's gates on the scenes enabled so far (`tools/nbflip-gates.ts`, the two-sided rule, §7.1);
- C0 on all four (the old engine still runs the scenes not yet enabled);
- an interleaved A/B on the timing slot against baseline tree 33b78ebb (`tools/nbflip-timing-round.sh`).

**Expected ms** are fenced / pipelined, from the per-area model in §6. A milestone more than 1.5x off its expected
time is a diagnosis checkpoint, not a stop (memory "understand, optimize, refine"): attribute the cost, fix the
cause, remeasure.

| milestone | content | gate | expected | lane-days |
|---|---|---|---|---|
| **M0** go/no-go and preconditions | (a) the addressing bench, done; (b) peak counts per scene from the baseline receipts, which fix the pools; (c) root-cause P2's p1b collapse on the old engine, in scratch; (d) P1 lands the staggered Schwarz and exact coarsest solve on today's arena (P1-1 to P1-3), giving the solver M2 reuses; (e) freeze the brick ABI doc (channels, records, lists, bins) | (c) is explained and reproduced; (d) passes the P1 gates on four scenes | (d): the old engine −1–3 ms on seep and dam | 3 |
| **M1** engine skeleton | Pool, directory, the brick-set update (`setMark` with the liquid / particle / CFL criteria only), lists, the param ring, the two-part submit, `operator` and P1's solver on the pool. Static liquid boxes, hydrostatic. Unit tests: enter/leave against a CPU oracle; pool exhaustion latches; hydrostatic pressure against Float64; cold compile under 30 s a kernel. | unit tests; hydrostatic equilibrium; no domain-sized dispatch (the 4x-wider-box test) | – | 5 |
| **M2** dam | Particles (§2.4) and the level set (§2.5), without solids beyond the box walls; momentum, extension; full `setMark` tier criteria with `setTransfer` (h ↔ 4h). Formula parity per kernel against the old engine on a frozen state. | dam D1–D8 and C0 over 165 steps, then over 300 (the new engine has no residency closure) | 18 / 17 against 45.7 / 49.7 | 6 |
| **M3** seep | Static voxel solids (`solidStamp`, `phiSolid`, `pSolid`), the open top, refinement regions, BULK16, region heat and cold freeze | seep G1–G8 over 1200 steps; dam re-gated | 13 / 12 against 34.6 / 33.0 | 4 |
| **M4** letters | Drops (sources), volume control, the high-particle regime | L1–L5 over 240 steps; seep and dam re-gated | 31 / 28 against 84 / 69 | 3 |
| **M5** paddleboats | Bodies (`bodyStamp`, `couple`, the rigid integrator), motors, no-slip, scenery. The AGENTS.md regressions for the stamped channels: nearest-surface hit test, local edit opening visual and collision together, untouched fidelity outside the edit, save/load/undo of an edited solid. | P1–P4 over 300 steps; all four re-gated | 19 / 17 against 49.7 / 44 | 6 |
| **M6** default and delete | Every remaining feature (moving window, open-world terrain, the inflow nozzle, interactive balls and `editFluid`, detail focus, QA audits, the presentation brick path). Flip the default for every NB-FLIP scene in `scenes.ts` (smoke run plus gates where references exist), then delete `lib/methods/uniform/` and the `engine` parameter. | all four gate scenes plus a smoke run of every NB-FLIP scene; `npm run check:types` and `npm run test:unit` | as M5 | 5 |

The total is about 32 lane-days, about 2.5–3 weeks with three lanes in parallel. The first gated scene (the dam,
M2) comes about 2 weeks in.

**Order.**
- The dam comes first: a closed box, one liquid body, no internal solids, no bodies or sources, and the
  sloshiest physics.
- The seep adds static solids.
- Letters adds sources and particle load.
- Paddleboats adds bodies.

## 5. Lane split (three long-lived lanes)

| lane | owns (files) | milestones |
|---|---|---|
| **E: engine core** | `lib/methods/narrow-band/` storage, directory, `set*` kernels, lists, the schedule and record, the param ring, submission, host and contract adapter, presentation adapter, `solidStamp` and the scene-apply bridge, the method-id switch; the brick ABI doc | M0(e), M1, the M2 tier transfer, M3 solids, M6 |
| **F: fluid kernels** (absorbs P2) | `lib/methods/narrow-band/particles-*`, `levelset-*`, `velocity-*`: the move split, rebin, seeding, reconstruction, redistance, the volume gauge, p2g, momentum and forces, extension, sources. P2's move findings and parked variants (`~/code/me/fluid-scratch/lane-p2/parked/`) are its starting point. | M0(c) the p1b root cause; M2 particles, level set and velocity; M4 |
| **P: pressure and bodies** (continues P1) | `lib/methods/narrow-band/pressure-*`, `bodies-*`; until M2, `uniform-pressure-arena.ts` and its relatives for P1's in-place items | M0(d) P1 on the old arena; M1 the solver on the pool; M5 bodies, `couple`, motors |

**How P1 folds in.**
- P1's arena is the new engine's pool layout: the same 8³ bricks, 27-neighbour records and stride 520.
- Its staggered Schwarz smoother, exact coarsest solve, check-before-V-cycle and CPU Float64 symmetry oracle carry
  over with only the addressing changed.
- P1 keeps landing on the old engine through M0, so its gains are real on their own, then moves the same kernels
  across at M1.

**Other lanes.**
- Lane G stays the referee.
- Lane D's dam residency defect becomes moot for the new engine. It is still needed for the old engine's references
  past step 165, so lane D finishes it if cheap.

**Interfaces.**
- E freezes the brick ABI in M0: channels, record words, list formats, the bin layout and the param ring layout.
- F and P write kernels against it from M1.
- Changes to the ABI go through E.
- Every landing follows the private-index guard and the tests.

## 6. Expected end state

**Per-area model** (ms a step, split phase). Today's figures are lane A's. Each target is derived as follows:
- **move:** §4.1(3).
- **brick set:** 4 launches over the active set, at about 3x the seep's `classify` and scatter per brick.
- **level set and phi:** 7 staged launches at seep-like cost per brick.
- **velocity:** 3 launches plus 2 extension sweeps.
- **pressure:** setup at the seep's ratio. Solve at 1.5–2 slots of 12 launches. Rows today are about 1.3–1.8x the
  seep's on these scenes.
- **floor:** about 132 encoded at 3.5 µs, plus empty-slot validation.

| area | seep today → target | letters today → target | paddleboats today → target |
|---|---|---|---|
| move | 3.76 → 1.5 | 14.12 → 9.0 | 5.12 → 2.5 |
| layout / census / residency, plus remap, canonical, resolve | 3.60 + about 3.9 → 0.6 | 8.06 + about 8.4 → 1.5 | 6.41 + about 5 → 1.0 |
| level set and phi advection (outside canonical and resolve) | about 5.9 → 2.0 | about 15 → 4.0 | about 10 → 3.0 |
| grid velocity (outside remap) | about 3.9 → 1.6 | about 11.5 → 3.5 | about 8 → 2.8 |
| transfer and seeding | 2.93 → 1.0 | 9.79 → 2.5 | 4.60 → 1.3 |
| pressure setup and projection | 2.18 → 1.0 | 4.93 → 2.0 | 3.25 → 1.3 |
| pressure solve | 5.99 → 3.0 | 10.48 → 5.0 | 6.05 → 3.0 |
| bodies | – | – | in geometry → 1.5 |
| floor and other | about 2 → 1.0 | about 3 → 1.5 | about 3 → 1.5 |
| **split sum** | **33.4 → about 11.7** | **84.5 → about 29** | **45.9 → about 17.9** |

**End state against lane G's baselines** (timing slot, quiet GPU; fenced median / pipelined):

| scene | baseline fenced / pipelined | target fenced / pipelined | speed-up |
|---|---|---|---|
| seep (1200) | 34.6 / 33.0 | 13 / 12 | 2.7x |
| dam (165) | 45.7 / 49.7 | 18 / 17 | 2.5–2.9x |
| letters (240) | 84 / 69 | 31 / 28 | 2.5–2.7x |
| paddleboats (300) | 49.7 / 44 | 19 / 17 | 2.6x |

**How fenced becomes close to pipelined.**
- The two-part submit and encode-ahead remove the 2.2–4.2 ms first-submit wait.
- Cached rigid records remove the 4.78 ms CPU `advanceTo` share on paddleboats.

**Further work.** These are not promised against the seep's 5x, because the band's particle count is the method and
stays:
- particles only where the band needs them;
- fewer slots from a better smoother;
- f16 particle storage.

## 7. Decisions for the coordinator

1. **Should the drift gates be two-sided?** Lane G's gates pass when |candidate − mean| ≤ max(floor, 3S). A candidate
   that conserves volume better than NB-FLIP fails the gates for that improvement:
   - G7 (floor 0.1 point; NB-FLIP drifts −0.51% on the seep);
   - L1 (0.5 point; −1.1% on letters);
   - P1 (0.25 point; −0.97% on paddleboats).

   §2.12 (1), better pressure convergence, is the change most likely to move drift. The recommendation is to make
   G7, L1, P1 and D4 one-sided on |drift|: the candidate may drift less, never more. That needs the user's approval,
   because it relaxes "same physics within the gates".
2. **Is "iterate-changing" pressure accepted as the same physics?** That means converging to the same tolerance
   where today's solver caps out. The recommendation is yes, judged by the trajectory gates.
3. **May the presentation texture copy (§2.10) live until M6?** It is a presentation cost only.
4. **Lane D's priority.** The new engine has no residency closure, so lane D matters only for the old engine's
   references past step 165.
5. **P2's parked work.**
   - p1a is a net loss; keep it as data.
   - p1b is invalid until its cause is known.
   - `item2-census-split.patch` is not needed: the new schedule has no census, and its submit split is §2.11. It
     should not be landed on the old engine.

### 7.1 Coordinator rulings (2026-11-15, before any engine code lands)

The design is **approved**: strategy C, the §2 architecture, the §4.2 milestones and the §5 lanes E, F and P.

1. **Drift gates are one-sided on |drift|** (G7, L1, P1, D4). The candidate passes if its |drift| ≤ |ref mean drift| + max(floor, 3S).
   - This is registered now, before any landing, so it is not a loosening after the fact.
   - Basis: the seep rewrite's §4 principle (the G6 ruling, 70351e00). NB-FLIP's measured volume loss is a defect the candidate need not reproduce.
   - **Rows that move with volume** (depth maps, column depth, PE, G3, G6) are judged against the union of two bands:
     - the reference band;
     - the reference with its own measured loss restored uniformly, as seep G6's NB_restored.

     Lane G implements both bands before M2's first gate.
   - The user is told and may veto.
2. **Iterate-changing pressure is accepted as the same physics.** It means converging to the same tolerance where today's solver caps out, and it is judged by the trajectory gates.
3. **The presentation texture copy may stay until M6.** It must be outside the simulation step's timing, or counted in it and reported.
4. **Lane D finishes.** The old engine's dam references must reach step 300 for M2's 300-step gate. If lane D's fix changes the old engine's physics, lane G re-takes the dam references at the fixed commit.
5. **P2's parked work is ruled as §7.5 recommends.** p1a is kept as data; p1b stays invalid until F explains it; the census-split patch is not landed.
