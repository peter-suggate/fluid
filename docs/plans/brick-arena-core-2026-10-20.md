# The brick arena core and the surface route, 2026-10-20

This fills two parts of `speed-of-light-plan-2026-10-20.md`: the surface section (§5) and the shared structure (§4). It folds in the needs of `pressure-rfe-speed-of-light-2026-10-20.md`.

Sources:
- the profiled pour `part2h` (frames 94–100, dt 1/60 s), with `hm2` (after 01769825) where it differs;
- the hardware ceilings in `scratchpad/ceil/ceilings.json`, including two new sections, `brick_arena` and `brick_halo`, measured for this doc (§2.1).

## Summary

**Surface.**
- Today: 18.16 ms over 40 pass entries, about 55 dispatches. After 01769825 the store is 0.86 ms rather than 2.06, which puts the stage at about 17 ms.
- Bound at the pour's counts: **≈0.40 ms** (§1.2).
- Arena projection: **≈0.92 ms** in about 10 passes. That meets the ≤1 ms φ target and the ≤1.2 ms stage target, and is 2.3× the tighter bound. What holds it above 2× is the exact redistance's ALU (§1.3).

**Lifetime: persistent, not step-authoritative.** Bricks keep their slots and values across steps, and the paged h store becomes cold storage.
- A step-authoritative copy costs 0.9–1.3 ms a step on its own, measured. That is about half of the three stages' combined 2× budget (≈2.05 ms).
- The persistent arena copies only bricks that enter or leave, which comes to ≈0.02–0.04 ms.
- Until every reader of paged φ and u is on the arena, a write-through scatter of changed bricks keeps the paged store valid, at ≈0.12 ms per channel.

**Core.** `UniformBrickArena` (§3) provides:
- three levels of 8³-cell bricks: h, 2h and 4h, where a 4h brick is one page;
- persistent fixed-capacity slots that fail fast;
- 27-neighbour records that encode seams;
- per-stage SoA channels in one pool;
- one halo-staging helper for halos of 1–4 cells, which resolves hanging and seam values;
- churn gather and scatter, plus write-through;
- the MG restrict/prolong hook.

Pressure and RFE map onto it as written in their analysis (§3.10).

## 1. Surface

### 1.1 Counts

The pour at frames 94–100:

| What | Count |
|---|---|
| Fine tiles | 6004 (384k h cells) |
| Band vertices advected | ≈490k |
| Coarse cells | 57.5k |
| Particles | 661k–695k |
| Arena slots (surface core plus ring) | 3000–3240 |
| Arena core (bricks with a crossing candidate) | 1750–1810 |
| Exact distance evaluations (R) | 1.47M |

The core holds about 3.3 fine tiles of a brick's 8. The slot set as a whole holds about 1.9.

### 1.2 Per pass: today, irreducible work, bound, limiter

Each bound is the pass's job at the counts above, using:
- bytes at 360 GB/s;
- gathers at 15–34 G/s;
- workgroup memory at 719 G/s;
- workgroup atomics at 218 G/s;
- 2.3 µs a pass.

The limiters were found by code reading. The phase 1/1b work (01769825) adds evidence for R and W.

| Group / pass | part2h ms | Irreducible work | Bound ms | Limiter today |
|---|---|---|---|---|
| **Advect** | **6.47** | 490k vertices: an RK trace through u (2 trilinear samples × 3 components) and a 64-tap cubic of φ, plus wall and solid rules. Reads 1800 bricks × 4 channels × 2 KB = 14.7 MB | **0.10** | |
| advectFine | 1.657 | burial, foot, Euler transport | | One lane per vertex over the h list. Owner and burial walks through the page table (umVertexSlot, umOwnerAt). Predicate registers set occupancy. |
| advect deferred [6] | 1.816 | the general twins: slow samples, released walls, seams | | Fat general kernels with the solid record and seam searches, run at 1/10 of the trace's own throughput. |
| advectFinish | 1.058 | walk, value, embedded air, source, store | | Detail-store slot walks per tap and per store. |
| advectTrace | 0.749 | the trace alone (fast sampler) | | Gathers through the page table, each face texel a slot walk. |
| advectCubic + solidFree | 0.983 | 64 taps per listed vertex | | 64 slot-walked gathers per entry. |
| advectWalls, advect, owners, wallReach | 0.281 | wall-face vertices, lists | | Lists rebuilt each step. |
| **Reconstruction (splat and couple)** | **3.18** | 680k particles × 16 B (11 MB), and a sphere per sample into ≈27 vertices (18M workgroup atomic-mins). Then φ = min(φ'+h, φ_p) over the band | **0.11** | |
| coupleFine | 1.348 | the union over the band | | Per vertex: heat interpolation, an incident solid-owner walk, page walks. |
| surfaceSplatOccupied | 0.609 | the splat | | Device atomics on paged banks (5.7 G/s against 218 for workgroup atomics). |
| refinement | 0.524 | per-particle refinement prediction | | A whole pass over all particles that only reads them. It can fold into the splat's particle read. |
| couple | 0.365 | the coarse union | | Page walks. |
| surfaceClear | 0.337 | — (scratch) | 0 | Clears both banks over every mapped page, a whole-page memset each step. |
| **Crossings and nearest** | **1.43** | crossing masks of candidate tiles, and the separable nearest crossing cell per cell, in workgroup memory over core bricks | **0.03** | |
| arenaNearest | 0.449 | | | Window re-reads per axis pass; reach lists. |
| depthSeeds, depthNearestY/Z, depthReuseSelect/Snapshot | 0.836 | | | The legacy path for non-arena tiles, whose lists are rebuilt each step. |
| arenaCross | 0.140 | | | |
| **Redistance and store** | **4.27** (3.07 in hm2) | exact distance: 1.47M evaluations of the nearest represented zero, at ≈200–400 flops each (0.3–0.6 GFLOP) | **0.10–0.15** | |
| arenaDistance (R) | 1.657 | the exact distance | | ALU and latency: about 5 Newton steps an evaluation, the same crossing cell solved again for every vertex it serves (≈15). Occupancy 576 threads. |
| arenaStore (W) | 2.060 (0.861 in hm2) | changed values back to paged φ: 3.7 MB | 0.01 | Writer predicates and slot-walked texture stores. |
| arenaGather ×2 | 0.421 | φ into windows, every step: 9.4 MB | 0.03 | Per-texel slot walks (2× the dense microbench). |
| arenaFar | 0.131 | far values outside the arena | | |
| **Coarse retirement** (depthCoarse, redistanceCoarse) | **0.55** | 57k coarse cells | **0.01** | Page strides. |
| **Work control** (16h) [3] | **1.21** | which regions are hot: class bits per brick from last step's φ range and particle occupancy, 3240 words | **0.01** | A 16h region model with revision scans and a geometry cache, separate from the brick table. |
| **phi resolve** [5] | **0.71** | 0, because hanging values are resolved when staged | **0** | A pass that exists only to write hanging texels into the paged store. |
| Lists (arenaMark, brickTable*, depthTiles, depthPlan*, depthLists) | 0.27 | in the table build | 0.01 | |
| global surface volume [4] | 0.06 | a reduction | 0.005 | |
| **Surface** | **18.16** | | **≈0.40** | |

The ≈0.40 ms bound is the sum of the group bounds (0.10 + 0.11 + 0.03 + 0.12 + 0.01 + 0.01 + 0 + 0.01 + 0.005) plus ≈0.02 ms for about 10 passes.

### 1.3 Arena-native surface kernels

These are the target set. Each kernel is one workgroup per brick on a direct grid-stride over a class list of the persistent arena (§3). There is no page table in any inner loop. φ is a two-bank vertex channel on the fine level.

| Kernel | Replaces (part2h ms) | Projected ms | Limiter |
|---|---|---|---|
| **aTable**: three passes (update, records, class lists) | work control 1.21, lists 0.27, wallReach and owners 0.08: **1.56** | 0.03 | Pass floor; table reads per brick. |
| **aEnter**: gather entering bricks | arenaGather ×2: **0.42** | 0.02 | Latency: few bricks (churn estimated at 150–300). |
| **sAdvect**: stage u faces (halo 1), trace feet into registers; restage the old φ bank (halo 3) into the same memory, then cubic, wall/solid rules and source; write the new φ bank. A foot leaving the staged box goes on a compact list. | all advect: **6.47** | 0.25 | Staging: 4 channel visits over ≈1800 surface bricks. Measured 0.118 ms per channel at 3240 bricks (scalar), so 0.26 at most and ≈0.13 with vec4 core rows. |
| **sLong**: lean trace over the long-foot list, sampling through records | (part of deferred) | 0.02 | Global gathers. The list is a few percent of vertices. Never a step limit. |
| **sSplat**: per particle brick (the brick-keyed particle slabs): spheres into φ_p in workgroup memory (halo 1) with workgroup atomic-min, then the union into the new φ bank. The refinement prediction rides on the same particle read. | clear, splat, coupleFine, couple, refinement: **3.18** | 0.20 | Workgroup atomics and slab reads. |
| **sCross**: crossing masks and the separable nearest, from φ staged once (halo 2) | **1.43** | 0.06 | Staging, once. |
| **sDistance**: exact distance, with Newton warm-started from a per-crossing-cell closest-point seed computed once per cell in workgroup memory | arenaDistance, arenaFar: **1.79** | 0.30 | ALU. The seed cuts ≈5 Newton steps to 1–2 at unchanged exactness. A patch-distance variant (toggle) would approach the 0.10 bound. |
| **sCoarse**: coarse redistance and retirement on the 4h level | **0.55** | 0.03 | Small. |
| Hanging resolve | phi resolve: **0.71** | 0 | Folded into staging (§3.5). |
| Store | arenaStore: **2.06** (0.86) | 0 persistent; 0.12 write-through during the transition | Store latency (§2.1). |
| Volume | 0.06 | 0.01 | A workgroup reduction in sDistance. |
| **Surface** | **18.16** | **≈0.92** (≈1.04 while write-through runs) | 2.3× of ≈0.40 |

What stops 2×:
- **R's exact Newton solve** (0.30 against 0.12). The warm-started seed is the in-exactness lever. The patch variant goes behind a method-param toggle (update:"solver"), per the physics-change rule.
- **The staging rate** (§2.1). A single channel visit runs at ≈5× streaming in the scalar form.

### 1.4 Effect on between-pass idle

The frame carries ≈7.9 ms of GPU idle outside the pass sum, over 331 passes: ≈24 µs per pass boundary on average.

Surface goes from ≈55 dispatches to ≈10. Pressure goes from 155 to ≈20 and RFE from 58 to ≈12.
- If idle scales with boundaries, surface's cut removes ≈1 ms of idle and pressure plus RFE remove ≈4 ms.
- The arena adds no copies or clears between stages: no per-step gather, scatter or surfaceClear.
- It does add one dependency chain, the three aTable passes, at the head of the step.

The first switch-over measures this directly: the probe's idle_ms per frame, before and after.

## 2. Lifetime: step-authoritative or persistent

### 2.1 Measured copy costs

These come from `ceilings.json` (`brick_arena`, `brick_halo`). Setup: NB random bricks of 8³ cells, an r32float texture of 33³ blocks, and random 27-slot neighbour records, matching the arena's atomicAdd slot order.

| Copy, one f32 channel | 3240 bricks | Rate | Note |
|---|---|---|---|
| gather9: texture → SoA 9³ windows | 0.098 ms | 192 GB/s | Dense texture, no page walk. Today's real arenaGather is ≈2× this. |
| scatter: SoA → texture, every texel | 0.125 ms | 151 GB/s | |
| scatter, a quarter of the texels (per-texel mask) | 0.118 ms | | **Store latency, not volume.** Skipping values saves nothing; only skipping whole bricks does. |
| halo1 stage (10³ from 27 slots) plus a 7-point read, 729 stride | 0.118 ms | 27 G loads/s | ≈5× a streaming read of the same core values; 0.092 at stride 520 (below) |
| halo2 stage (12³) | 0.131 ms | | A wider halo is nearly free once staging is paid |

The `brick_halo_stride` section stages a 1-cell halo from 8³ SoA slots at several slot strides (3240 bricks, ms per channel):

| Slot stride (words) | Scalar staging | vec4 core rows | Core-only stream |
|---|---|---|---|
| 512 | 0.229 | 0.269 | 0.026 |
| **520** | **0.092** | 0.131 | 0.013 |
| 528 | 0.138 | 0.164 | 0.026 |
| 576 | 0.098 | 0.105 | 0.013 |
| 729 (today's 9³ windows) | 0.138 | — | 0.020 |

Three findings:
- **A power-of-two slot stride costs 2.5×.** 27 neighbour slots 2 KB apart alias the same cache sets. The pool pads every slot to **520 words**: 8 spare words keep vec4 alignment and break the aliasing.
- **The naive vec4 variant does not help**, because its shell loop still walks all 1000 positions. A real vec4 core needs the shell enumerated directly (488 positions). That is a helper optimisation still to be made.
- These runs repeat the dispatch 10× over a 7 MB pool, which stays in the 48 MB system cache. The core stream at 0.013 ms is faster than DRAM. The real arena's rates come from M2 (§4).

Timer quantisation is 6.5 µs per dispatch here, so differences under ≈0.01 ms are noise.

### 2.2 Step-authoritative

Gather once at step start, every stage works on the arena, scatter once at the end. Hot channels that must round-trip:
- φ;
- u faces ×3;
- RFE's snapshot u ×3 and pressure's warm-start p, unless those persist anyway.

| | Per step |
|---|---|
| Gather φ, u (4 channels) | 4 × 0.098 = 0.39 ms (bench) to ≈0.8 ms (real gather with page walks) |
| Scatter φ, u | 4 × 0.12 = 0.50 ms |
| **Copy** | **0.9–1.3 ms**, plus 0.2–0.4 ms for p and the snapshot |

The passes it removes are arenaGather, arenaStore, phi resolve and every page walk in surface, pressure and RFE. But the persistent arena removes the same passes. So the copy is a net cost against persistence, not a saving against today. That cost is 0.9–1.3 ms per step against a combined 2× budget of ≈2.05 ms (surface 1.2, pressure 0.5, RFE 0.35). **Step-authoritative cannot reach 2×.**

### 2.3 Persistent (chosen)

A brick keeps its slot and channel values while it stays marked. The paged h store is cold storage.

| | Per step |
|---|---|
| Churn gather: entering bricks, estimated 150–300 of 3240, all channels they carry | ≈0.01–0.02 ms |
| Churn scatter: leaving bricks, the same count | ≈0.01–0.02 ms |
| Slot allocation (§3.2): a free stack instead of a rebuild | +1 pass, ≈5 µs |
| Write-through, transition only: changed bricks of channels still read from the paged store | ≈0.12 ms per channel (the full-scatter rate, since skipping values saves nothing) |

**Churn estimate.**
- The surface moves less than a cell per step on the pour at 1/60 s, so the arena's shell shifts at most ⅛ brick a step.
- Entering bricks ≈ ring bricks × ⅛ ≈ 150–300.
- This is the core's first measured number (§4, M1). If churn were 100%, persistence would degrade to step-authoritative and never cost more.

**Correctness requirement.** Every writer of a persistent channel must write the arena, not the paged store, while the brick is live.
- Transition: a stage still on the paged store marks the bricks it wrote dirty. aTable then re-gathers them, which goes through the churn path.
- Persistence for φ is switched on only once every h-region φ writer is on the arena: surface advect, couple, redistance, sources.

**Who still reads paged φ and u once surface, pressure and RFE are on the arena?**
- Momentum and particle transfer: mapped by the momentum agent.
- Particle advection: samples u. Its slabs are brick-keyed, so it reads arena channels.
- The renderer: φ. Until it reads the arena, φ write-through stays at ≈0.12 ms.
- The coarse 4h region: it is the arena's 4h level (§3.1), so there is no copy.

### 2.4 Staging rate

Staging cost sets every kernel's floor, so the helper is the first thing to optimise.

Levers, measured or to measure:
- **A non-power-of-two slot stride (520)**: measured, 2.5× against 512.
- **vec4 core rows** with the shell enumerated directly. Core loads drop 4×. The naive form measured no gain (§2.1).
- **Stage only channels that need a halo.** The others read core-only, a coalesced stream at about the streaming rate.
- **Two bricks per workgroup** when workgroup memory allows: halo 1 at 4 KB a channel. This hides the per-brick latency chain.

Pressure's projection (26–40 µs per fine visit) and RFE's (0.12 ms extension) assume staging close to streaming. The measured rate at stride 520 is 0.092 ms per channel at 3240 bricks, ≈28 ns a brick.

What that means per stage:
- **Pressure:** 1500 fine bricks give ≈43 µs per visit for p's halo, plus core-only streams for rhs and row. That is close to their 40 µs.
- **Surface's sAdvect:** 4 visits over 1800 bricks come to ≈0.2 ms, which is the figure used in §1.3.

## 3. Core spec: `UniformBrickArena`

**Files:**
- `lib/methods/uniform/uniform-brick-table.ts`: extended with persistence and levels;
- `lib/methods/uniform/uniform-brick-arena.ts`: new; channels, pool, staging, gather/scatter, MG indices.

**Conventions:**
- WGSL helpers are prefixed `uba`; table words keep `ub`.
- Everything is fixed capacity, allocated at attach.
- No pipeline is compiled after attach.

### 3.1 Levels and keys

A level-ℓ brick is 8³ cells of width 2^ℓ h. All three levels use the page lattice's block index, so no level needs a hash.

| Level | Cell | Brick footprint | Bricks a page | Key | Slots |
|---|---|---|---|---|---|
| ℓ0 fine | h | 8³ h | 4³ | 64·block + (bx \| by≪2 \| bz≪4) | allocated (§3.2) |
| ℓ1 | 2h | 16³ h | 2³ | 8·block + (bx \| by≪1 \| bz≪2) | allocated, ≤ ℓ0 live count |
| ℓ2 coarse | 4h | 32³ h = the page | 1 | block | **identity: slot = block**, capacity UM_BLOCKS |

- The ℓ0 key is today's brick id, so `ubBrickOfTile`, `ubBrickAt` and `ubBrickOrigin` stand.
- ℓ2 needs no allocation. Its channels are block-indexed, and the 4h region's cells are its rows.
- Levels coarser than 4h (8h, 16h, 32h: 4³, 2³ and 1 cell a page) are sub-blocks of an ℓ2 brick's channels (§3.7). The 32h level, with one cell a page, is solved over the page list in one workgroup.

### 3.2 Slots: persistent, fixed capacity, fail-fast

- Capacity: `fineCapacity` ℓ0 slots and `fineCapacity/4` ℓ1 slots, set at construction.
  - The default is 64·UM_BLOCKS, as today's table, which cannot overflow.
  - A smaller configured capacity bounds memory: 2 KB per brick per f32 channel.
- Each level has a **free stack**: a count word plus capacity words, filled 0..cap−1 at attach.
- **Fail-fast.** A pop from an empty stack sets the table's fatal word (with the brick id and level) and leaves the brick unslotted. The existing frame diagnostics read it and throw at the first rejection. There is no recovery path and no growth.

Per step, replacing today's clear-and-rebuild:
1. Clear the marks only. The markers run as today, and classes OR in.
2. **brickTableUpdate** (a thread per brick id; the dilation as today):
   - live now, unslotted: pop a slot, append to the enter list;
   - slotted, unmarked: append to the leave list (the slot is not yet freed);
   - otherwise: update the class word.
3. **brickTableRecord**: the 27 records of every live slot (≈87k words, a few µs), plus the parent words (§3.3). Records are rebuilt every step and are cheap; slots are what persist.
4. **brickTableLists**: compact class lists (§3.4) over the live slots, with workgroup-aggregated atomics.
5. After the step's scatter (§3.6), **brickTableRelease** pushes the leave list's slots onto the free stack.

Slot order is still atomic. It is not deterministic run to run, but slots are stable for a brick's life, which is all the channels need.

### 3.3 Records

The record is 32 words per slot per level:
- [0..26] neighbours n = x+3y+9z of (−1..1)³;
- [27] parent;
- [28] child mask and tile mask;
- [29..31] spare.

**Neighbour word encoding:**
- `slot` (bit 31 clear): a live brick of the same level.
- `0x80000000 | block` (seam): the neighbour region has no brick at this level but its page is stored. Values come from the coarser level, resolved by staging (§3.5). At ℓ0 that is the ℓ2 brick (the 4h region); at ℓ1 it is ℓ2.
- `UB_NONE`: outside stored pages. The channel's declared outside value applies (wall or air).

**Parent word:** `parentSlot | octant≪29`. The octant is the brick's position in its parent: (bx&1) | (by&1)≪1 | (bz&1)≪2 for ℓ0 → ℓ1, and the ℓ1 local index for ℓ1 → ℓ2.

**Masks:**
- ℓ0 tile mask, bits 0..7: which of the brick's 8 tiles are h-owned. Bits 8..15 are a crossing-candidate tile mask.
- ℓ1 and ℓ2 child mask: which octants have a live child.

### 3.4 Classes and lists

The class word per slot keeps today's bits. New bits:

| Bit | Name | Meaning |
|---|---|---|
| 128 | fine | the brick holds an h tile; marked by the ownership |
| 256 | surface | \|φ\| < band anywhere in the brick, from last step's per-brick φ range word, which sDistance writes |
| 512 | liquid | any pressure row |
| 1024 | dirty | a legacy writer changed the paged copy; re-gather |

Lists are compact slot lists, rebuilt each step by brickTableLists:
- `surface` (surface kernels);
- `fine` (pressure, RFE);
- `extend` (RFE extension: the air band);
- `particles` (slabs);
- `enter` and `leave`;
- `level1`.

Kernels stride a list with a fixed direct grid sized to its capacity. Counts come from the list header, never from the host.

### 3.5 Channels

**Pool.** One `array<u32>` storage buffer, the arena pool, holds every channel of every level at fixed offsets, so a stage binds one buffer. f32 data is bitcast; relaxed atomics where a channel is atomic. Offsets are compile-time constants:

`base(channel) + 520·slot + i`

where `i = x + 8(y + 8z)` with x fastest. The layout gives each brick's x rows as two aligned vec4s.

The slot stride is 520 words, not 512: 8 spare words per slot. A power-of-two stride aliases the 27 neighbours' cache sets and staging costs 2.5× (§2.1).

**Layouts.** Everything is owned high, matching today's tiles, which own their positive vertices (local 1..4):

| Layout | Index (x,y,z) ∈ [0,8)³ means | The low boundary value comes from |
|---|---|---|
| `cell` | cell (x,y,z) | the −neighbour's x = 7 cell |
| `faceX`, `faceY`, `faceZ` | the +x (+y, +z) face of cell (x,y,z) | the −neighbour's x = 7 face, which is cell 0's low face |
| `vertex` | the +++ corner of cell (x,y,z): lattice vertex 8g + (x,y,z) + 1 | the −neighbour, giving vertex 8g |

Every channel holds 512 values per brick (520 words of stride): no 9³ duplication, unlike today's windows.

**Declaration.** A stage declares the channels it reads and writes:
```ts
arena.channel({ name: "phi", level: 0, layout: "vertex", type: "f32", banks: 2, outside: "far", seam: "coarse-vertex", mirror: "phi" });
arena.channel({ name: "u", level: 0, layout: "face", type: "f32", banks: 1, outside: "zero", seam: "coarse-face", mirror: "velocity" });
arena.channel({ name: "p", level: [0, 1, 2], layout: "cell", type: "f32", outside: "zero", seam: "coarse-cell" });
```

Fields:
- `layout: "face"` declares the three components.
- `banks: 2` gives the double buffer a stage swaps by a uniform word, with no copy.
- `seam` and `outside` are the staging rules (§3.6).
- `mirror` names the paged field for churn gather, scatter and write-through.

Memory is allocated at attach for the union of declared channels. `arena.wgsl(stage)` emits that stage's offsets and the `uba` accessors for its channels only.

**Initial channel set, agreed by need:**

| Channel | Levels | Layout | Users |
|---|---|---|---|
| φ (2 banks) | 0, 2 | vertex | surface (advect, splat, redistance); RFE geometry; pressure θ |
| u | 0, 2 | face ×3 | momentum, advect trace, pressure projection, forces, extension |
| uSnap | 0 | face ×3 | RFE snapshot extension |
| p | 0, 1, 2 | cell | pressure (warm start persists) |
| rhs | 0, 1, 2 | cell | pressure; restriction target |
| row | 0, 1, 2 | cell u32 | pressure kinds/diag bits; unit-coefficient flags |
| coef | 0, 1, 2 | face ×3 | pressure non-unit face weights (V-weighted, θ) |
| fill | 0, 2 | cell | RFE geometric fill (volume fraction) |
| aperture | 0, 2 | face ×3 | RFE/pressure solid fractions |
| known | 0 | cell u32 | extension known mask |
| scratch0..1 | any | any | per-stage, no persistence promise |

### 3.6 Halo staging: one helper, one call site

The generator emits one staging function per kernel:
```ts
arena.stageWGSL("pDown", { level: 0, channels: ["p"], halo: 1, region: "brick" /* or "octant" */ })
```

It emits:
```wgsl
var<workgroup> ubaW_pDown: array<f32, E*E*E*nChannels>;  // E = n + 2·halo, n = 8 (brick) or 4 (octant)
fn ubaStage_pDown(slot: u32, lane: u32, origin: vec3i);  // origin: the box's first owned index
fn ubaAt_pDown(c: vec3i, k: u32) -> u32;                  // index of (c, channel k) in ubaW_pDown, c ∈ [−halo, n+halo)
```

Each kernel calls its stage function exactly once, following the one-call-site rule for heavy functions. The kernel may then loop sweeps over workgroup memory.

**Steps inside:**
1. Stage the brick's 27 neighbour words and their tile masks into workgroup memory (54 words).
2. Core rows as vec4 loads.
3. Shell cells scalar.
4. One barrier.

**Seam and hanging rules,** per channel, applied where the source is a seam word, a cleared tile-mask bit (a 4h-owned tile inside a live fine brick), or `UB_NONE`:

| Rule | Value | Today's equivalent |
|---|---|---|
| `coarse-vertex` | `umVertexFrom4` over the ℓ2 brick's φ (same weights and D4 summation) | phi resolve's hanging texels. The phi resolve pass goes. |
| `coarse-face` | the 4h face value for each of its 4×4 h subfaces | pressure's Neumann faces taking the transferred 4h velocity |
| `coarse-cell` | the 4h cell value (piecewise constant), or `coarse-cell-tri` for trilinear | band init's 4h start pressure |
| `far` | ±far with the sign from the class (liquid or air) | arenaFar |
| `zero` / `clamp` | 0 / the nearest owned value | walls, air p |

Notes:
- A seam source costs one extra ℓ2 load per staged value, and only on seam bricks.
- Hanging values are never stored. They are recomputed at each staging from the current ℓ2 values, so they cannot go stale.

**Workgroup memory per f32 channel:**

| Halo | Box | KB |
|---|---|---|
| 1 | 10³ | 4.0 |
| 2 | 12³ | 6.9 |
| 3 | 14³ | 11.0 |
| 4 | 16³ | 16.4 |

Extension's 3 components at halo 3–4 (33–49 KB) do not fit. Options: one component per staging, the three passes over one buffer, or a halo of 2 with an iterate-and-restage loop. RFE's analysis already names this fallback.

### 3.7 Gather and scatter

- **aEnter** (churn gather): a workgroup per enter-list brick, per level. Each channel with a `mirror` is gathered from its paged field by that field's formula (today's arenaGather value for φ, the umFace values for u). Channels without a mirror start at their `outside` value: p = 0, so the first solve starts from the prolonged coarse p.
- **aLeave** (churn scatter): a workgroup per leave-list brick, before brickTableRelease. Mirrored channels are written to the paged field. Leaving bricks are written whole: scatter is store-latency-bound, so value skipping buys nothing (§2.1).
- **aMirror** (write-through, transition only): a workgroup per brick of the channel's dirty list. Its last writer in the step sets a per-brick bit (a class-word bit per mirrored channel). It writes the brick's owned values to the paged field. This is today's W reduced to a plain stream.
- **dirty re-gather:** a legacy writer's bricks are re-gathered through aEnter's path.

The mirror's only predicate is ownership. Per tile mask, it writes h-owned tiles only; the 4h-owned tiles of a fine brick are the ℓ2 level's own values.

### 3.8 The MG hierarchy hook

Levels are brick lists. Every transfer is a workgroup per child brick, with no atomics: each parent octant has exactly one child.

**Indices:**
- `ubaParent(level, slot) -> vec2u(parentSlot, octant)`;
- `ubaChild(level, slot, octant) -> slot or UB_NONE`, from the child mask and a 2³ child record in the ℓ1 and ℓ2 records' spare words. The child record needs 8 words; ℓ1 and ℓ2 records extend to 40 words.
- `ubaOctantCell(octant, c) = 4·octant + (c >> 1)`: the parent cell of child cell c.

**Restrict (child → parent):**
- The child kernel holds its residual or values in workgroup memory, reduces each 2³ group, and writes the 4³ octant block of the parent's channel.
- Faces: a parent face takes the sum of its 2×2 child faces, V-weighted by the declared weight channel.
- Pressure's pDown does this in its tail.

**Prolong (parent → child):**
- The child kernel stages the parent's octant with a 1-cell halo, using `region: "octant"` (6³) with the same helper, which resolves the parent's seams.
- It interpolates trilinear or constant per the stage, masked by the row channel.

**Composite seam (h next to 4h).**
- An ℓ2 cell is *covered* if its ℓ1 octant child exists and the corresponding ℓ0 tile mask is set; otherwise it is a coarse row. Coverage is the child mask chain.
- Fine side of a seam face: the ghost is the coarse value (`coarse-cell` rule).
- Coarse side: the face flux is the sum of the 4×4 fine subface fluxes. The ℓ0 kernel writes it into an ℓ2 face scratch channel at restriction (reflux). That is today's root rule (coarse coefficient = V-weighted sum of fine faces) made exact and one-sided.
- The operator itself (coefficients, the p ≥ 0 projection, cycle shape) is pressure's.

**Below 4h.** 8h, 16h and 32h are the 4³, 2³ and 1³ sub-blocks of an ℓ2 brick's channel. The 512-entry block is used as 64 + 8 + 1, with offsets in `uba`. One workgroup per page smooths and restricts its tail with a 1-cell halo from neighbouring pages (ℓ2 records). The 32h level is one workgroup over the page list.

### 3.9 Rules every arena kernel keeps

- One workgroup per brick (or two), on a direct grid-stride over a class list. No indirect dispatch.
- No page-table walk inside a kernel. Positions come from the record's origin word; seams from record words.
- Staging is one call site per kernel; the WGSL that consumes it is lean. Split rather than fuse heavy code paths, since fat kernels cost about 10×.
- No JS allocation per step: lists, counts and offsets live on the GPU.

### 3.10 Mapping pressure and RFE (for the momentum agent)

| Their kernel | Lists | Channels | Staging | Hook |
|---|---|---|---|---|
| pRows | fine, ℓ2 coarse rows | in: φ, u, aperture, fill. Out: rhs, row, coef, p₀ | φ halo 1 (vertex) | Balance: workgroup reduction plus one atomic a brick |
| pDown ℓ | level ℓ | p, rhs, row, coef | p halo 1 (scalar shell) | restrict → parent rhs |
| pUp ℓ | level ℓ | p, row, coef; u on the last cycle | parent octant halo 1; p halo 1 | prolong; projection into u |
| pCoarse | ℓ2 tail, page list | p, rhs sub-blocks | ℓ2 halo 1 | |
| rGeometry | surface (ℓ0), ℓ2 | φ → fill, aperture | φ halo 1 | |
| rRemap | enter/leave of the fine class | all mirrored, p | — | A width change is a fine-class change: the gather prolongs, the scatter restricts V-weighted |
| rForces | fused into momentum's u writer | u | — | |
| rExtend | extend | u, uSnap, known | one component, halo 2–3 | Far faces from ℓ2 |

Open points for the agreement:
- whether `row` and `coef` are per level or rediscretised on the fly at ℓ1;
- whether p's ℓ1 slots need persistence. Proposed: no, ℓ1 is scratch between cycles.

## 4. Build order and measurements

**Core, unwired (new files):**
1. Table persistence: free stacks, update, records with seam words and parents, lists, release, fatal word.
2. Pool and channel declaration.
3. The staging generator with vec4 rows and seam rules.
4. aEnter, aLeave, aMirror.

**Measurements, at the fixed frame-94 state with the replay harness:**
- **M1:** live counts per class (fine, surface, particles) and churn per step: enter/leave over 10 steps. This sets capacity and the persistence gain.
- **M2:** staging rate on the real arena: one channel at halo 1 and halo 3, against the `brick_halo` bench.
- **M3:** aMirror for φ against today's W.

**Surface switch-overs.** Each is one short edit, then the letters A/B (`lp.mts`) and the pour, then a commit:
1. **Redistance on core channels.** X, nearest, R and far read the φ vertex channel through the staging helper instead of 9³ windows. The store becomes aMirror. Gather becomes the step's aEnter with every brick entering, since this is not yet persistent. Expected: W 0.86 → ≈0.15; G about the same.
2. **Persistence for the redistance's φ** once advect writes the arena. G → churn.
3. **sAdvect, sSplat** (with the momentum agent's u channel), then work control → aTable, and phi resolve deleted.

Judge each step by surface per thousand fine tiles (`fam.js`): 3.03 on part2h, 2.76 on hm2. The target is 0.15 per kT (≈0.92 ms).
