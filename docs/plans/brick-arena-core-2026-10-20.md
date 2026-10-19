# The brick arena core and the surface route, 2026-10-20

This fills two parts of `speed-of-light-plan-2026-10-20.md`: the surface section (§5) and the shared structure (§4). It folds in the needs of `pressure-rfe-speed-of-light-2026-10-20.md`.

Sources:
- the profiled pour `part2h` (frames 94–100, dt 1/60 s), with `hm2` (after 01769825) where it differs;
- the hardware ceilings in `scratchpad/ceil/ceilings.json`, including two new sections, `brick_arena` and `brick_halo`, measured for this doc (§2.1).

## Summary

**Surface.**
- Today: 18.16 ms over 40 pass entries, about 55 dispatches. After 01769825 the store is 0.86 ms rather than 2.06, which puts the stage at about 17 ms.
- Bound at the pour's counts: **≈0.40 ms** (§1.2). §5 revises this: staging does not pay in the thin band, advect's gather-rate floor is ≈0.9 ms, and the pass-by-pass persistence projection is ≈5.4 ms (0.90 per kT) against today's 15.0.
- Arena projection: **≈0.92 ms** in about 10 passes. That meets the ≤1 ms φ target and the ≤1.2 ms stage target, and is 2.3× the tighter bound. What holds it above 2× is the exact redistance's ALU (§1.3).

**Element counts (§6, 2026-10-20).** Every surface pass runs near its per-element rate, so the levers are fewer elements and cheaper elements:
- advect's taps through the texture unit, ≈0.9 ms;
- an advect band of k cells, ≈1.6 ms, but it misses fast fronts early in the pour;
- a coarse band, ≈0.95 ms;
- particle-free copy tiles in coupleFine, ≤0.5 ms;
- fineGridPadding 0, ≈0.7 ms;
- redistance reach of one tile, ≈0.3 ms.

The ranked list with params is §6.9. §5.5 corrects §5: P0 is null, Newton is 14% of R, P1 is ≈1 ms and shelved, and P3 is void.

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
| **aEnter**: gather entering bricks | arenaGather ×2: **0.42** | 0.02 | Latency: few bricks (churn measured at 29–73 a step, §2.3). |
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
| Churn gather: entering bricks, measured 29–73 of 3100–3340 (M1), all channels they carry | ≈0.005 ms plus the pass floor |
| Churn scatter: leaving bricks, measured 16–49 | ≈0.005 ms plus the pass floor |
| Slot allocation (§3.2): a free stack instead of a rebuild | +1 pass, ≈5 µs |
| Write-through, transition only: changed bricks of channels still read from the paged store | ≈0.12 ms per channel (the full-scatter rate, since skipping values saves nothing) |

**Churn estimate.**
- The surface moves less than a cell per step on the pour at 1/60 s, so the arena's shell shifts at most ⅛ brick a step.
- Entering bricks ≈ ring bricks × ⅛ ≈ 150–300 was the estimate.
- **Measured (M1, 2026-10-20).** The pour, frames 94–104 at 1/60 s, reading the brick table after each step:

| Frame | Slots | Core | Enter | Leave | Core enter | Core leave |
|---|---|---|---|---|---|---|
| 95 | 3107 | 1745 | 34 | 48 | 42 | 33 |
| 96 | 3150 | 1767 | 66 | 23 | 49 | 27 |
| 97 | 3188 | 1797 | 57 | 19 | 56 | 26 |
| 98 | 3215 | 1825 | 44 | 17 | 47 | 19 |
| 99 | 3243 | 1851 | 48 | 20 | 47 | 21 |
| 100 | 3276 | 1887 | 73 | 40 | 67 | 31 |
| 101 | 3309 | 1897 | 52 | 19 | 39 | 29 |
| 102 | 3317 | 1913 | 36 | 28 | 45 | 29 |
| 103 | 3335 | 1920 | 34 | 16 | 37 | 30 |
| 104 | 3315 | 1915 | 29 | 49 | 37 | 42 |

  - Churn is 1.5% entering and 0.9% leaving a step: 47 and 28 bricks on average, 3–6× below the estimate.
  - So a persistent arena moves about 2% of the bytes a step-authoritative one would.
  - The pour is still filling, so the slot count grows by about 20 a step. Capacity is set from the run's peak, which is fail-fast (§3.2).
  - If churn were 100%, persistence would degrade to step-authoritative and never cost more.

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

- Capacity: `fineCapacity` slots at ℓ0 and at ℓ1, set at construction. ℓ1 never exceeds the ℓ0 list it is built from.
  - The default is 64·UM_BLOCKS, as today's table, which cannot overflow.
  - A smaller configured capacity bounds memory: 2 KB per brick per f32 channel.
- ℓ0 has a **free stack**: a count word plus capacity words, filled 0..cap−1 at attach.
- ℓ1 is scratch, rebuilt each step. `ubaLevel1` gives the parent of every brick on one ℓ0 list (`level1From`, e.g. `fine`) a slot by atomic claim on a per-step table of ℓ1 keys. ℓ2 slots are blocks, listed in the page list.
- **Fail-fast.** A pop from an empty stack sets the table's fatal word (with the brick id and level) and leaves the brick unslotted. The existing frame diagnostics read it and throw at the first rejection. There is no recovery path and no growth.

Per step, replacing today's clear-and-rebuild (entries as built in `uniform-brick-arena.ts`: ubaUpdate, ubaLists, ubaLevel1, ubaRecord, then ubaRelease after the scatter):
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

**Declaration.** Channels are declared at construction, each at the levels it lives on:
```ts
new UniformBrickArena({ blocks, channels: [
  { name: "phi", levels: [0, 2], layout: "vertex", banks: 2 },
  { name: "u", levels: [0, 2], layout: "face" },
  { name: "p", levels: [0, 1, 2], layout: "cell" },
  { name: "coef", levels: [0, 1, 2], layout: "face" },
  { name: "flux", levels: [2], layout: "face" },
], lists: [{ name: "fine", any: 128 }, { name: "liquid", any: 512 }], level1From: "fine" });
```

Fields:
- `layout: "face"` declares the three components.
- `banks: 2` gives the double buffer. The current bank is a GPU word that a stage flips from one lane of a later pass, with no copy.
- Seam values are per staging (§3.6). Gather and scatter formulas are per copy entry (§3.7).

Memory is allocated at attach for every declared channel. In WGSL, bases are `UBA_<NAME>_L<level>_<component>_<bank>`, with `ubaLoad(base, slot, i)` and `ubaStore(base, slot, i, v)`. u32 data is bitcast. Persistence is not a property of a channel: a channel persists unless a stage overwrites it, and ℓ1 channels are scratch because ℓ1 slots are rebuilt every step.

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

The generator emits one staging function per kernel. Each channel has its own halo; pressure's smoother takes p at halo 2 and the row channels at halo 1 in one call:
```ts
arena.stageWGSL("pDown", { level: 0, region: "brick" /* or "octant" */, workgroup: 256, channels: [
  { name: "p", halo: 2, seam: "pSeam" }, { name: "rhs", halo: 1, seam: "zeroSeam" }, { name: "coef", component: 0, halo: 1, seam: "coefSeam" } ] })
```

It emits:
```wgsl
var<workgroup> ubaW_pDown: array<f32, Σ_j E_j³>;    // E_j = n + 2·halo_j, n = 8 (brick) or 4 (octant)
fn ubaStage_pDown(slot: u32, lane: u32, octant: u32); // octant: the region's octant ("octant" only)
fn ubaAt_pDown(c: vec3i, j: u32) -> u32;              // channel j at c ∈ [−halo_j, n+halo_j), relative to the region
```

A seam function is `fn(p: vec3i, word: u32) -> f32`:
- p is the absolute owned index in the level's cells;
- word is `UB_NONE`, `UBA_SEAM | block`, or a live slot whose tile or octant is dead.

At ℓ1 the mask is the live-child octants: a dead octant is a seam to ℓ2, by the same code as a dead tile at ℓ0.

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
- `ubaChild(level, slot, octant) -> slot or UB_NONE`, read from the ℓ0 table (ℓ1 → ℓ0) or the per-step ℓ1 key table (ℓ2 → ℓ1). Records stay 32 words.
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

Agreed with the pressure agent, 2026-10-20, from its CPU twin of the frame-94 band (97.5k rows, 636 fine bricks):
- **Coarse operators are rediscretised:** piecewise-constant aggregation × 0.5, giving a 7-point stencil at every level. Galerkin bought only 0–1 cycle. `row`, `diag` and `coef` are per level, and pDown writes the parent's while it restricts.
- **ℓ1 and sub-ℓ2 are scratch.** p persists at ℓ0, and at ℓ2 on coarse rows.
- **The smoother stages p at halo 2 and the row channels at halo 1.** Halo 1 frozen across 2 sweeps needed 8 V(2,2) cycles. Halo 2, with the first sweep updating ring 1, is exact and needs 5–6.
- **Reflux into a declared ℓ2 face channel,** `flux`.
- **Lists:** `fine` and `liquid` at ℓ0, ℓ1 from `fine`, and the ℓ2 page list. All counts come from the GPU.

## 4. Build order and measurements

**Core, unwired (new files):**
1. Table persistence: free stacks, update, records with seam words and parents, lists, release, fatal word.
2. Pool and channel declaration.
3. The staging generator with vec4 rows and seam rules.
4. aEnter, aLeave, aMirror.

**Measurements, at the fixed frame-94 state with the replay harness:**
- **M1:** live counts per class (fine, surface, particles) and churn per step: enter/leave over 10 steps. This sets capacity and the persistence gain. *Done (§2.3): 47 in and 28 out a step, of ≈3200.*
- **M2:** staging rate on the real arena: one channel at halo 1 and halo 3, against the `brick_halo` bench.
- **M3:** aMirror for φ against today's W.

**Surface switch-overs.** Each is one short edit, then the letters A/B (`lp.mts`) and the pour, then a commit:
1. **Redistance on core channels.** X, nearest, R and far read the φ vertex channel through the staging helper instead of 9³ windows. The store becomes aMirror. Gather becomes the step's aEnter with every brick entering, since this is not yet persistent. Expected: W 0.86 → ≈0.15; G about the same.
   - **Revised after the W knock-outs (2026-10-20): expect about zero.** W's cost is not the layout. It is the paged store's write rules, which decide which redistanced values are physics: the writer tile, COLD regions held, interface magnitudes kept, capacity. At frame 94 (≈3100 slots, replay minimums, 0.065 ms timer steps):

| Arm | ms |
|---|---|
| arenaStore | 0.79–0.85 |
| no capacity | 0.72 |
| no capacity, writer, cold or interface | 0.33 |
| staging only | 0.20 |
| no wall-page authority walk | 0.79 |
| also no near-cell walk at x/y/z = 8 | 0.72 |
| regular fast path: owned-high writer when all 64 tiles from one below are stored, h, regular and not COLD | 0.85 |

   - The fast path is exact. With every eligible vertex forced to a sentinel, the plain and fast arms store the identical 247,144 vertices. It is still null, because few bricks have a fully stored, regular neighbourhood: the core holds ≈3.3 fine tiles of a brick's 8.
   - aMirror keeps all of those rules while paged φ stays authoritative. So it costs what W costs, and G, X, nearest and R read the same bytes through staging as through windows.
   - W's rules go away only once φ is persistent (switch-over 2): then the arena is the authority, and the store is the write-through scatter of changed bricks, about 0.12 ms.
   - So switch-over 1 is infrastructure with no expected gain. It is worth landing together with 2, not alone.
2. **Persistence for the redistance's φ** once advect writes the arena. G → churn.
3. **sAdvect, sSplat** (with the momentum agent's u channel), then work control → aTable, and phi resolve deleted.

Judge each step by surface per thousand fine tiles (`fam.js`): 3.03 on part2h, 2.76 on hm2. The target is 0.15 per kT (≈0.92 ms).

## 5. Persistence design (a), pass by pass (revised 2026-10-20, after sAdvect)

### 5.1 What this session measured

- **sAdvect: a loss.** Built at near-exact answers, then dropped (patch `sadvect-2026-10-20.patch`). It cost 4.22 ms against the 2.4 ms of advect it replaced. The band is thin: about 103 of the 125 tiles in a brick's staged window are 4h, so staging a window loads mostly texels nobody advects. **Texel loads are not the bottleneck.** advectTrace already runs at the measured coherent-gather rate (below), so moving loads into workgroup memory gains nothing.
- **Lean 4h path: exact, but null.** 21542 values equal at frame 100, 18 differing by at most 1e-7, none missed. Net gain 0.06 ms. Dropped (patch `coarse-lean-2026-10-20.patch`).
- **Wall trace "clip" (402a3221).** One RK3 step plus an analytic contact clip, behind the `wallTrace` param. General deferred 1.432 → 1.123 ms; advect 6.10 → 5.85 ms. Letters within the perturbation floor.
- **List sizes at frame 100** (counters in umAdvectStore and the deferred headers):

| List | Entries |
|---|---|
| h vertices into umAdvectStore | 362.5k |
| 4h vertices into umAdvectStore | 61.3k |
| Euler fails, so the vertex goes to the trace list | 331.3k (91% of h) |
| cubic, solid-free | 57.5k |
| cubic, solid-aware | 40.1k |
| cubic, deferred (4h) | 5.8k |
| general deferred, twinned | 35.0k: 13.3k wall-plane, 21.7k other |

- **The §1.2 bound of ≈0.40 ms assumed staged reads**, at the 719 G/s workgroup rate. The thin band rules that out for advect. At the 22 G/s gather rate that advectTrace reaches, advect's own floor is about 0.9 ms: ≈200k vertices × 48 trace taps, plus ≈97k × 64 cubic taps.

### 5.2 Per pass: today, what replaces it, projected

Today is the clip build at frames 94–100 (split-all profile `wcb`). ns is ms over the element count named. Projections use the element rates measured on today's lean kernels, not staging rates. **Keeps** means the pass keeps its current structure.

The design has three structural levers. None of them is staging.

1. **The advect certificate: the vertex disappears.**
   - Every non-interface, non-COLD vertex in reach has its value replaced by the redistance store (`arenaStore`: exact distance, or sign·6h). So only the advected sign survives.
   - If |φ_old(p)| > s·dt + 1.5h, where s is the speed bound over p's 27-brick neighbourhood, the sign cannot flip and p cannot become an interface vertex. Its advect can be skipped, keeping φ_old.
   - The exception is particle coupling. There, min(φ'+h, φ_p) can differ when φ_p falls between the kept and the advected values, which is rare. So the certificate goes behind an NB-FLIP param (update "solver").
   - It does not apply in COLD regions, where redistance holds advected values.
   - **Unmeasured:** what fraction of vertices it certifies. The projections assume 45% of h vertices (≈200k still advected). One counter run settles it.
2. **The persistent arena is the authority for fine φ.**
   - Gathers become churn.
   - Hanging texels are filled when written, so phi resolve is gone and no sampler carries a width, held or seam predicate.
   - Store rules become per-brick class bits plus one compare per texel.
   - Paged φ is fed by write-through (≈0.12 ms) until its readers move.
3. **Workgroup-memory atomics for the splat** (218 G/s, against 5.7 G/s for device atomics), on the particle brick slabs.

| Pass | ms | Elements | ns/element | Replacement | Projected ms |
|---|---|---|---|---|---|
| **Advect** | **5.85** | | | | **2.41** |
| advectFine | 1.208 | 424k vertices | 2.85 | Persistent per-brick vertex masks replace the ownership and burial walks. The certificate early-outs. No Euler attempt (it fails 91%). | 0.35 |
| advectTrace | 0.721 | 331k | 2.18 | **Keeps.** Arena-record sampler; ≈200k entries | 0.30 |
| advectFinish | 1.086 | 331k | 3.28 | Walk skip and cubic routing kept. The paged store (umStoreSurface, umRecordRegion) becomes a bank write. | 0.30 |
| advectCubic (solid-free) | 0.281 | 57.5k | 4.9 | **Keeps.** Arena sampler | 0.20 |
| advectCubic (solid-aware) | 0.721 | 40.1k | 18.0 | Per-tap solid-record walk becomes one bit test on the record's open mask; otherwise the free kernel | 0.24 |
| advectCubicDeferred | 0.272 | 5.8k | 47 | **Keeps** (4h) | 0.27 |
| advectDeferred, twinned | 1.123 | 35.0k | 32 | Slow samples and seams (≈21.7k) disappear on the arena sampler. Walls (13.3k) stay in the general twin: no lean wall kernel, per the coordinator. | 0.45 |
| advectWalls | 0.131 | 13.5k walls | 9.7 | Persistent wall list per brick (walls do not move) | 0.02 |
| advect, fast and regular twins, owners, wallReach | 0.28 | 61.3k 4h | — | **Keeps** (4h lattice); wallReach and owners go to aTable | 0.28 |
| **Reconstruction** | **3.03** | | | | **0.68** |
| surfaceClear | 0.318 | mapped pages | — | **Disappears.** φ_p lives in sSplat's workgroup memory. | 0 |
| surfaceSplatOccupied | 0.543 | 590k particles | 0.92 | sSplat: a workgroup per particle brick. Reads the 27 neighbour slabs' particles within r (≈2× reads) and does workgroup atomic-min. | 0.15 |
| coupleFine (both twins) | 1.338 | 362.5k h vertices | 3.7 | sSplat's tail: min with φ' on the brick's owned texels, into the new bank. Heat and solid-owner walks become record bits. | 0.10 |
| transferRefine | 0.449 | 590k | 0.76 | Rides on sSplat's particle read | 0.05 |
| couple (coarse) | 0.384 | 57.5k coarse cells | 6.7 | **Keeps** | 0.38 |
| **Crossings** | **1.78** | | | | **0.84** |
| arenaGather (search) | 0.225 | 3159 slots | 71 | **Disappears.** aEnter of ≈47 entering bricks | 0.01 |
| arenaCross | 0.150 | 1793 core bricks | 84 | **Keeps.** Reads the channel | 0.10 |
| arenaNearest | 0.506 | 1793 | 282 | **Keeps.** No per-element win named. | 0.45 |
| depthSeeds, depthNearest\*, depthReuseSelect, depthTiles | 0.825 | candidate tiles | — | Fine candidates fold into arenaCross; 4h candidates and the reuse plan stay. **Split not yet counted.** | 0.25 |
| arenaMark, brickTable\* | 0.075 | | | aTable | 0.03 |
| **Redistance and store** | **2.86** | | | | **0.81** |
| arenaGather (metric) | 0.253 | 3159 slots | 80 | **Disappears** (persistent) | 0.01 |
| arenaDistance | 1.582 | 1.47M evaluations | 1.08 | **Keeps** (one lean kernel). Newton warm-started from a closest-point seed computed once per crossing cell, cutting ≈5 steps to 1–2 | 0.66 |
| arenaStore | 0.889 | 247k stored | 3.6 | The rule chain collapses. COLD becomes a brick class (the brick leaves the list). Writer becomes the record's static owned mask. Interface becomes R's crossing mask, already in hand. Capacity becomes a record bit. Then write-through. | 0.12 |
| arenaFar | 0.131 | | | Only for entering and leaving bricks | 0.02 |
| **Coarse retirement** (depthCoarse, redistanceCoarse) | **0.56** | 57.5k coarse cells | 9.8 | **Keeps.** No per-element win named. | 0.56 |
| **phi resolve** (resolve, resolveListed) | **0.80** | | | **Disappears.** Write-through writes resolved hanging texels. | 0 |
| **Volume** (measureInitial, clearBand, solve) | **0.13** | | | **Keeps** | 0.13 |
| **Surface** | **15.01** (2.50 per kT, 6004 fine tiles) | | | | **5.43** (0.90 per kT) |

Not counted above, and outside (a): the pressure authority (surfaceHeld, wetRegions, cutRegions, stamp; 1.45 ms). It is pressure's interface consumer of φ and keeps its structure.

### 5.3 What keeps the stage above the 0.15 per kT target

| What | ms | Why |
|---|---|---|
| Advect | 2.41 | Gather-bound per vertex. The floor at the gather rate is ≈0.9 ms. The rest is the wall twin (0.45) and the 4h passes (0.55). |
| The coarse level: couple, coarse retirement, 4h advect | 1.22 | Off the fine arena. The arena's 4h level (§3.1) could take it, but no per-element win is named. |
| arenaNearest and arenaDistance | 1.11 | Already lean. Only the Newton warm start is named. |

### 5.4 Build order: certain wins first, each behind a param where physics changes

| Phase | What | Gate | Saves |
|---|---|---|---|
| **P0** | Advect certificate on today's kernels, no arena. One counter run first: what share it certifies. | Letters A/B | ≈1.0 ms at 45% |
| **P1** | Fine φ persistent. advect and couple stores also write the arena bank (dual write, ≈0.2 ms). R, X and nearest read the bank; fine crossing candidates fold into arenaCross; R gets the Newton warm start. The store rules become class bits. Write-through feeds paged φ. Gathers, arenaFar and phi resolve go. | Letters A/B | ≈3.4 ms |
| **P2** | Advect on the arena sampler. advectFine becomes persistent masks; seams and slow samples leave the deferred list; solid cubic becomes bit tests. Needs the u channel over core and ring (the core-velocity channel extended to the ring). | Letters A/B | ≈2.4 ms |
| **P3** | sSplat on the particle slabs (particle agent) | Letters A/B | ≈2.3 ms |

### 5.5 Corrections after the knock-outs (2026-10-20)

§5.2–5.4 projected before the kernels were knocked out. The knock-outs and counter runs since then change five rows.

- **P0, the advect certificate: null.** It assumed |u| bounds how fast the zero can reach a vertex. At frame 100, 92% of the 332k traced h vertices have s·dt ≥ 1h from the vertex's own |u|, an optimistic stand-in for the 27-brick bound. So it certifies 17.5k (5%) at best, under 0.15 ms. Even with no speed cap (c > 2 + 2·s·dt) it reaches only 20%. Not built. Advect's 2.41 ms projection becomes ≈3.4 ms on the arena route.
- **R's Newton is 14% of R.** Split by knock-out: per-target setup ≈0.85 ms, edge sweeps ≈0.49 ms, Newton ≈0.21 ms. So the warm start saves ≤0.2 ms, not 0.92.
- **phi resolve stays** while paged readers of φ exist. It disappears only with persistence, and P1 is shelved.
- **depth\* is not a duplicate.** It is a separate crossing search at reconstruct time, on the advected φ, for coupling and membership. The end-step arena search runs on the coupled φ. Folding fine candidates into arenaCross does not apply.
- **The splat was already workgroup-atomic**: a 7³ window, then one device atomicMax a word. P3's lever is void.
- **P1 is ≈1.0 ms, not 3.4: shelved.** Gathers ≈0.25 net, store ≈0.5, Newton ≈0.2, for a large build (persistent fine φ, dirty tracking on topology change, fail-fast capacity).
- **The store's capacity rule is ≈0.1 ms, not 0.5.** §4's table already showed it: "no capacity" is 0.72 against 0.79–0.85. Staging capacity a tile at a time (exact, patch `store-capacity-2026-10-20.patch`) measured 0.880 ms at 6420 fine tiles against 0.871 at 6067: 4% per kT, inside run-to-run spread. Dropped. True per-brick caching is not available, because the brick table is rebuilt every step and its slots move. The ≈0.4 ms that "no capacity, writer, cold or interface" removes is in the per-vertex rule chain, which only persistence replaces.
- **The targeted surfaceClear landed:** 0.309 → 0.169 ms. It clears only the vertices coupleFine and couple read, instead of every mapped page's 33³ words.
- **The core (u, u\*) face channel is deleted** (a4149615). Nothing used it. The patch `core-velocity-2026-10-20.patch` restores it.

## 6. Element counts: what each pass processes, what the physics needs (2026-10-20)

Every surface pass already runs near its measured per-element rate (§5.2). So the levers left are processing fewer elements, or a cheaper element: the texture unit, which nothing on the surface uses yet. Sources:
- the pour at 1/60 s, frames 94–100: `sc1` (after the targeted clear, 6067 fine tiles, 53.3k coarse tiles, surface 14.39 ms, 2.37 per kT);
- counter runs `band1`–`band3`;
- knock-outs `reach2`, `reach1` and `pad0`.

Runs vary by ±6% in fine tiles from run to run, so per-pass deltas under ≈0.03 ms are noise.

### 6.1 Priors: what broke, and what still stands

The rule: a measurement more than 2× off its speed-of-light figure means the priors behind both the bound and the design get reassessed, and a different approach is tried. Four projections failed that test.

| Projection | Projected | Measured | The prior that broke |
|---|---|---|---|
| sAdvect: staged advect | 2.4 → ≈0.3 ms | 4.22 ms | **Reads can be staged.** The band is thin: ≈103 of a brick window's 125 tiles are 4h, so a staged window loads mostly texels nobody advects. |
| The advect certificate (P0) | −1.0 ms at 45% | ≤−0.15 ms at 5% | **|u|·dt is the front speed.** 92% of traced vertices move ≥1h a step, yet only ≈5% of them flip sign. The velocity is mostly tangential, or the fluid moves with the surface; it is not the zero's normal speed. |
| Newton warm start | −0.92 ms | ≤−0.2 ms | **Newton dominates R.** R is setup 0.85, edge sweeps 0.49 and Newton only 0.21. |
| sSplat | −2.3 ms | void | **The splat used device atomics.** It was already workgroup-atomic. |
| advect floor | 0.9 ms | — | **Advect taps run at the gather rate.** They are trilinear and could run on the texture unit (§6.2). This prior was never tested. |

**Priors still unchallenged**, with where this section challenges each:
1. Every vertex in the fine band is advected every step, whatever its distance from the zero (§6.3).
2. The redistance reaches three tiles past a crossing tile (§6.4).
3. The fine band is as thick as particle joins plus one cell of padding make it (§6.5).
4. The coarse level visits every 4h tile of every mapped page (§6.6).
5. Reconstruction runs the particle gather at every band vertex, including those no particle reaches (§6.7).
6. arenaNearest's 282 ns a brick is "lean". It was never compared with a bound at the workgroup-memory rate. Not challenged here.
7. The brick table is rebuilt every step, so nothing can be cached per brick. This stands while P1 is shelved.

### 6.2 The texture unit: advect's taps are filtering

**The prior.** "Advect taps run at the gather rate, so the floor is ≈0.9 ms." That assumed every tap is a load. But both of advect's interpolants are exactly what texture filtering computes:
- the trace's velocity is a trilinear blend of eight halo texels a component;
- the walk's φ is a trilinear blend of eight vertices;
- the cubic is a separable Catmull-Rom of 64 vertices.

The texture unit filters rgba32f at 13.5 G samples/s, against ≈22 G loads/s for the coherent gathers advect reaches today. One filtered sample replaces 8 loads, so per interpolant the unit is ≈5× faster.

**It is already proven in this codebase.**
- Momentum traces through `unitHaloFiltered` (`momentumSampler` texture32, the default).
- The particle trace does the same (5929ca1d): `advectParticles` 1.48 → 0.90 ms.

The surface module already binds `unitHalo` (binding 14), the same rgba32f atlas, and builds it each step before the surface stage. So the surface trace needs only:
- a second, filterable binding of that texture;
- a linear clamp sampler;
- a copy of the particle sampler's three lines in `umFastVelocity`.

**Velocity: what it saves.** Samples are counted per vertex. With the trace seed, advectFine's Euler sample is stage 0, and advectTrace takes stages 1 and 2.

| Pass | Velocity samples | Loads today | ms in loads at 22 G/s | Filtered at 13.5 G/s | Saves |
|---|---|---|---|---|---|
| advectTrace (0.730) | 331k × 2 | 15.9M | 0.72, so the pass is its loads | 2.0M: 0.15 | ≈0.55 |
| advectFine (1.198) | ≈362k × 1 | 8.7M | 0.40 | 1.1M: 0.08 | ≈0.30 |
| advectDeferred fast twin | ≈35k × 3 | 2.5M | 0.11 | 0.3M: 0.02 | ≈0.09 |
| **Total** | | | | | **≈0.9** |

Keeping it current costs nothing. unitHalo is already built every step for momentum and particles, with no copy and no new texture.

Coverage:
- unitHalo holds the halo blocks of h pages. A sample with fine weight f > 0 and a halo home takes it.
- A sample in a 4h-only region keeps the coarse cache's loads (umSampleVelocity4In). `coarseFiltered` (momentum's) could take those too, but they are a small share of the band.
- A sample with no halo home already ends the fast trace (umSlow) and is unchanged.

**φ through the texture unit.** φ is a paged atlas of 33³ vertex blocks with skirts.
- For an h cell, umVertexCell's eight vertices all lie inside one block, so one filtered r32f sample replaces them.
- For the cubic, taps −1..+2 lie inside the block for cell-local 1..30, which is 82% of cells. Catmull-Rom has negative outer weights, so only the middle pair of each axis merges into one bilinear-trick sample. That gives 3 samples an axis, 27 in 3D, against 64 taps.
- A 4h cell's corners are 4 vertices apart, so they cannot be filtered. Held, buried and solid-aware taps keep the loads.

| Pass | Taps today | Filtered | Saves |
|---|---|---|---|
| advectFinish walk (1.095) | 331k × 8 = 2.6M loads, ≈0.12 ms | 331k samples: 0.03 ms. But the cubic reuses the eight corners where it routes (≈30%), so those vertices reload them. | ≈0.07 |
| advectCubic, solid-free (0.281) | 57.5k × 64 = 3.7M, ≈0.17 ms | 82%: 27 samples; the rest taps. ≈0.10 ms | ≈0.08 |
| advectCubic, solid-aware (0.749) | the per-tap solid walk dominates, not the taps | — | ≈0 |

Keeping φ filterable costs nothing for the trilinear walk, since each vertex block already holds its cell's eight vertices. The cubic's 18% boundary cells fall back to taps rather than paying for an apron atlas, which would cost ≈0.15–0.2 ms a step to keep current and would cancel the gain.

**Precision.** The texture unit rounds filter weights to about 1/256 of a texel.
- For velocity, that is a trace endpoint error up to ≈|u|·dt/256 ≈ 0.06h at 15 cells a step. Momentum and particles already accept it.
- For φ, the trilinear error is ≈|∇φ|·h/512 ≈ 0.002h.
- The bilinear-trick cubic multiplies that error by the merged weights' ratio, so it is the riskiest of the three.

Each goes behind a param, on by default only if the letters A/B passes.

**The floor, restated.** At filtered rates:
- the trace's floor is 331k × 6 samples / 13.5 G/s ≈ 0.15 ms;
- the cubic's floor is 97k × 27 / 13.5 G/s ≈ 0.19 ms.

So advect's interpolation floor is ≈0.35 ms, not 0.9. What then dominates advect is per-vertex control: ownership walks, routing, the store and the wall twin. That is §5.2's persistent-mask lever, unchanged.

### 6.3 Advect band width k

**Today.** Every h vertex the band holds is advected: 364k stores a step at frame 100, of which 331k fail the Euler test and are traced. Let c = |φ_old|/h. The distribution, with sign flips counted per step (`band1`, frames 94–100):

| c | Vertices | Sign flips | New \|φ\| < 1.5h |
|---|---|---|---|
| < 1 | 64.4k | 14184 | 53.8k |
| 1–2 | 50.7k | 4867 | 25.5k |
| 2–3 | 49.6k | 466 | 5.7k |
| 3–4 | 40.8k | 15 | 1186 |
| 4–6 | 65.9k | 1 | 1028 |
| 6–9 | 40.4k | 0 | 251 |
| 9–15 | 7.1k | 0 | 11 |
| ≥ 15 | 45.3k | 0 | 3 |

**What the physics needs.** A vertex further than k cells from the zero needs its sign, not its value:
- the redistance overwrites its magnitude in reach (§5.5);
- past reach it is clamped to sign·6h.

So a band of width k holds φ_old beyond c ≥ k and advects the rest. The trace chain costs ≈11.9 ns per traced vertex (trace, finish, both cubics and the deferred twin over 331k).

| k | Vertices advected | Traced vertices removed | Saves | Missed flips a step, late (f100) | Missed flips a step, early (f50) |
|---|---|---|---|---|---|
| 3 | 165k (45%) | ≈177k | ≈2.1 ms | 8 | 688 |
| 4 | 206k (56%) | ≈133k | ≈1.6 ms | 2 | 409 |
| 6 | 271k (74%) | ≈71k | ≈0.85 ms | 0 | 186 |

**What breaks.** Missed flips are fast fronts. Early in the pour (f40–f70), droplets and the falling column move at up to ≈15 cells a frame and cross vertices 4–15 cells from the old zero in one step:
- At f40, 457 flips a step occur at c ≥ 3: 152 of them at 4–6 cells, 27 at ≥15 cells.
- Late (f100), flips at c ≥ 3 drop to 18 a step.

A missed flip leaves a stale sign behind or ahead of the front:
- **Air → liquid** (fluid arriving): couple's min(φ', φ_p) repairs it wherever particles arrive.
- **Liquid → air** (draining or detaching): only the redistance repairs it. The stale vertex now borders the true zero, gains a small |φ|, and is advected properly the next step. That is a one-step phantom fragment.

Thin sheets and droplets are where liquid → air happens fastest, so they carry the risk.

**A velocity-aware band** (also advect when |u|dt/h ≥ c − k) does not fix the early misses. At f40–f70 it misses 31–185 flips a step whether k is 3, 4 or 6, because the front reaches slow vertices: the vertex's own speed underestimates the trace that arrives there. Late, it keeps 66.7k extra vertices at k = 4 and saves ≈1.0 ms, against 1.6 blind.

**Recommendation.** A param `surfaceAdvectBand` (cells; 0 = all, today's behaviour), default 0. On by default only if the letters A/B passes at k = 4 or 6. The fronts the letters scene exercises are exactly the early-pour ones.

### 6.4 Redistance reach

**Today.** arenaCross gives:
- reach 3 (bits 1 and 2) to tiles within two tiles of a crossing tile;
- reach 2 (R only) to tiles at three tiles.

R evaluates 1.47M (target, crossing) pairs. The store writes exact distances in reach and sign·6h past it.

**What the consumers read.** All of them read bit 1 (the reach-1 ring):

| Consumer | Window |
|---|---|
| transfer | depth in [−2, 1.5] |
| particleDepth | seeds in [−4, 0], cells in (−4.5, 0.9) |
| particle kill | below −4 |
| census | retire below −4h |
| couple | the vertices near particles |

The pressure band reads the interface cells and the sign. A tile one tile from a crossing tile holds every cell within 4h of a crossing. So the consumers need reach 1 plus the boundary cases at exactly 4h, which the sign·6h clamp reads the same way (below −4 either way).

| Arm | arenaDistance | arenaFar | depthCoarse | redistanceCoarse | Saves |
|---|---|---|---|---|---|
| today (`sc1`, reach to 3 tiles) | 1.554 | 0.131 | 0.431 | 0.140 | — |
| `reach2` (drop the three-tile ring) | 1.554 | 0.075 | 0.421 | 0.140 | ≈0.06 |
| `reach1` (one tile only) | 1.404 | 0.066 | 0.365 | 0.122 | ≈0.30 |

arenaNearest and arenaStore do not move: the store still visits every vertex of its windows.

**What breaks.** Past reach, |φ| clamps to 6h, so true distances of 4–8h read as 6h. That changes:
- advect's φ_old there (harmless while §6.3's band is off; with a band of k ≥ 4 those vertices are held anyway);
- the advect band criterion itself;
- coarse retirement's metric at 4h corners near the band. depthCoarse builds only for reach-bit-2 corners (≈6k of 27k visited), so reach 1 builds fewer of them, and their metric bound changes.

A half-tile reach (6h, cell-level) would land near ≈0.2 ms.

**Recommendation.** A param `surfaceRedistanceReach` (tiles: 1, 2, 3; default 3). Set the default to 1 if the letters A/B passes.

### 6.5 The h band's thickness

**What sets the ≈6k fine tiles.** A tile is fine when it is joined, a source, required or traced. Particle joins dominate: each particle's box of ±0.875h, swept over the step's travel, plus `fineGridPadding` (default 1h), floored to tiles. Census criteria add tiles where the shape (0.5h), a thin feature (4h), strain (0.1), impact (1h a step) or approach (2 steps) asks, within a 50% budget. A fine tile retires below −4h depth after a 0.5 s heat fade.

**Is the ring wider than the physics needs?**
- Consumers need φ to 4.5h on the liquid side (§6.4).
- Particles live to −4h, so the liquid side of the band is set by particles, and particles are the fluid.
- The air side needs the zero and its advect reach.
- One cell of padding, floored to 4-cell tiles, adds up to one tile ring wherever a particle box sits near a tile face.

**Measured (`pad0`, fineGridPadding 0, same frames):**

| | `sc1` (padding 1) | `pad0` (padding 0) |
|---|---|---|
| fine tiles | 6067 | 5052 (−17%) |
| coarse tiles | 53.3k | 55.9k |
| surface | 14.39 ms (2.37 per kT) | 13.72 ms (2.71 per kT) |
| RFE, pressure, momentum | 8.28, 8.34, 0.92 | 8.01, 8.27, 0.86 |
| GPU frame | 44.3 ms | 43.3 ms |

The fine passes shrink with the tile count:

| Pass | `sc1` | `pad0` |
|---|---|---|
| advectFine | 1.198 | 1.077 |
| advectTrace | 0.730 | 0.599 |
| advectFinish | 1.095 | 0.918 |
| coupleFine | 1.311 | 1.123 |
| arenaStore | 0.871 | 0.777 |

The 4h passes grow: advectCubic 0.749 → 0.824, couple 0.374 → 0.421. So the stage saves ≈0.7 ms and the frame ≈1.0 ms.

The padding ring is ≈1000 tiles (17%). What it buys is margin: a particle within a cell of a tile face still samples, deposits and transfers at h, not across a hanging seam into 4h. Without it, more particles sit next to 4h tiles, and more of advect's work takes the coarse paths. That is a physics change, and it already has a param.

### 6.6 The coarse level (1.22 ms)

**Today.** The coarse level processes every 4h tile of every mapped page: 53.3k coarse tiles, ≈57.5k coarse cells and corners. Only the ones near the surface need it.

| Pass | ms | Elements today | What the physics needs (`band3`, per step) | ns per element |
|---|---|---|---|---|
| couple (coarse) | 0.374 | 57.5k corners | ≈3.5k corners have particle bulk; the rest copy φ' | 6.5 |
| depthCoarse | 0.431 | 57.5k corners visited | ≈6.0k distance builds (reach bit 2) | 7.5 per visit, 72 per build |
| redistanceCoarse | 0.140 | ≈27k corners | ≈6.0k in reach bit 2 | 5.2 |
| advect on the 4h lattice (twins, owners, wallReach) | 0.28 | 61.3k 4h vertices | the 4h vertices within the band (unmeasured; see below) | 4.6 |

**Elements the physics needs.**
- **Seam coarse tiles:** 5310, next to fine tiles.
- **Coarse corners near a crossing:** the ≈6k that depthCoarse builds.
- **Deep 4h corners:** retired. Sign only.
- **Air 4h corners far from fluid:** no particles reach them, and the fine band follows the fluid (particle joins), so their φ stays clamped.

So the need is ≈6–8k of 57.5k corners, about 12%.

**Projected**, at each pass's measured rate on the needed elements:

| Pass | ms | Projected |
|---|---|---|
| couple | 0.374 | ≈0.05 |
| depthCoarse | 0.431 | ≈0.10 (the visits go; the builds stay) |
| redistanceCoarse | 0.140 | ≈0.03 |
| advect on the 4h lattice | 0.28 | ≈0.05 |

That is ≈0.98 ms saved of 1.22.

**What breaks.** Holding a 4h corner's φ beyond N coarse widths of a crossing is §6.3's band at 4h, where a coarse cell is 4h. With N = 2 (8h), a front moving 15 cells a step could cross it. Deep-liquid and far-air corners change sign only when a free surface reaches them, and a surface arriving at h is fine-banded first. The risk is a 4h-only surface, a calm surface far from the action, moving more than 8h in a step.

**Recommendation.** The list of coarse corners within N coarse widths of a crossing, built in the crossing search, feeds all four passes. A param `surfaceCoarseBand` (coarse widths; 0 = all, today's behaviour). One counter run first: the 4h vertices advect visits, by c, as `band1` did at h.

### 6.7 Reconstruction without particles

coupleFine visits 342k h vertices a step and gathers particles at each (3.8 ns a vertex). 190k of them (56%) find particle bulk; the other 152k store φ' unchanged into the new bank. Air-side tiles of the band have no particle in reach. If the splat marked tiles with any particle within its reach, coupleFine's particle-free tiles would take a copy path at ≈0.5 ns a vertex. That is exact, with no physics change.

The saving is up to ≈0.5 ms if the 152k cluster in whole tiles. They likely do, because the band extends ≈2–3 tiles on the air side, but this is unmeasured. A one-counter run settles it: particle-free fine tiles a step. couple (coarse) is §6.6's lever.

### 6.8 Per pass: elements today, elements needed, projected ms

Each row uses the pass's measured per-element rate (`sc1`). "Needed" is what §6.2–6.7 argue the physics reads.

| Pass | ms | Elements today | Elements needed | Projected ms | Lever |
|---|---|---|---|---|---|
| advectFine | 1.198 | 424k vertices (362k h + 61k 4h) | same visits; velocity filtered | 0.90 | §6.2 |
| advectTrace | 0.730 | 331k traced | k = 4: ≈198k; filtered | 0.09 (0.18 filtered only; 0.44 band only) | §6.2, §6.3 |
| advectFinish | 1.095 | 331k | k = 4: ≈198k | 0.66 | §6.3 |
| advectCubic, free | 0.281 | 57.5k | mostly c < 2: kept; filtered | 0.20 | §6.2 |
| advectCubic, solid-aware | 0.749 | 40.1k | mostly kept | 0.70 | — |
| advectDeferred, twinned | 1.545 | 35k | k = 4: ≈60%; filtered fast twin | 0.85 | §6.2, §6.3 |
| advect on the 4h lattice | 0.28 | 61.3k | ≈12% | 0.05 | §6.6 |
| surfaceClear | 0.169 | reconstructed vertices | same | 0.169 | done |
| surfaceSplatOccupied | 0.524 | 590k particles | same | 0.524 | — |
| coupleFine | 1.311 | 342k vertices | 190k with particles, plus a copy of the rest | ≈0.8 | §6.7 |
| couple | 0.374 | 57.5k corners | ≈6–8k | 0.05 | §6.6 |
| arenaGather ×2, arenaCross, arenaMark | 0.69 | ≈3.1k slots | same | 0.69 | — |
| arenaNearest | 0.477 | 1793 core bricks | same | 0.477 | — |
| arenaDistance | 1.554 | 1.47M pairs | reach 1 | 1.40 | §6.4 |
| arenaStore | 0.871 | 247k stored | same | 0.87 | — |
| arenaFar | 0.131 | | reach 1 | 0.066 | §6.4 |
| depth\* (seeds, nearest, reuse, tiles) | 0.90 | candidate tiles | same | 0.90 | — |
| depthCoarse | 0.431 | 57.5k visits | ≈6k builds | 0.10 | §6.6 |
| redistanceCoarse | 0.140 | 27k | ≈6k | 0.03 | §6.6 |
| phi resolve | 0.749 | | same | 0.749 | — |

### 6.9 Ranked levers

Physics changes go behind NB-FLIP method params (update "solver"). Each is on by default only if the letters A/B passes.

| Rank | Lever | Saves | Physics risk | Param |
|---|---|---|---|---|
| 1 | Velocity through the texture unit in advectFine, advectTrace and the deferred fast twin (§6.2): a copy of 5929ca1d | ≈0.9 ms | Low. Weight rounding ≈1/256 texel, already accepted for momentum and particles. | `surfaceVelocitySampler`: texture32 / software |
| 2 | Advect band k = 4 (§6.3) | ≈1.6 ms (≈1.4 after rank 1, since the trace gets cheaper) | Medium-high. ≈400 flips a step missed early in the pour; liquid → air ones leave one-step phantom fragments. | `surfaceAdvectBand`: cells, 0 = all |
| 3 | Coarse band: couple, depthCoarse, redistanceCoarse and 4h advect only within N coarse widths of a crossing (§6.6) | ≈0.95 ms | Medium. A 4h-only surface moving more than 8h a step. Needs a 4h counter run first. | `surfaceCoarseBand`: coarse widths, 0 = all |
| 4 | Particle-free tiles take a copy path in coupleFine (§6.7) | ≤0.5 ms | None (exact). Needs a tile-occupancy counter first. | none needed |
| 5 | fineGridPadding 0 (§6.5) | ≈0.7 ms surface, ≈1.0 ms frame | Medium. Particles within a cell of a tile face meet hanging seams and 4h; letters A/B not run. | `fineGridPadding` (exists) |
| 6 | Redistance reach one tile (§6.4) | ≈0.30 ms | Low-medium. True 4–8h read as 6h; coarse retirement builds fewer corners. | `surfaceRedistanceReach`: tiles 1–3 |
| 7 | φ through the texture unit: trilinear walk and 27-sample cubic (§6.2) | ≈0.15 ms | Low for trilinear, medium for the bilinear-trick cubic | `surfacePhiSampler`: texture32 / software |

Ranks 1–4 and 6 together project ≈4.0 ms of the stage's 14.4 (2.37 → ≈1.7 per kT at today's tile count); rank 5 changes the tile count itself and adds ≈0.7. The 0.15 per kT target still needs persistence and the per-vertex control levers of §5.2.

**Next surface lead: the coarse level** (rank 3). Rank 1 comes first: it is the cheapest certain win, a copy of a measured change.
