# Brick-resident step: the non-pressure frame as a few fused brick passes (2026-11-08)

Goal (user): a disturbance costs work in proportion to the detail it
creates, so a vast scene with local activity runs at a high frame rate. A
fundamental change, not dispatch tweaks. **Pressure is out of scope.**

**Decision (user, 2026-11-08): "we don't need bit-parity. performance is
most important. proceed, be bold."**
- No stage needs bit or field parity with today's kernels. Execution order,
  sweep order and seam evaluation may all change.
- The gates are physical (volume drift, tank growth, stable rest) plus step
  time. Wherever this document says parity, read it as "physically
  equivalent under those gates".
- Storage migration is a means, not a goal. A fused pass may stage from the
  current paged textures through the record first, and move a channel into
  brick slots only where that removes latency. This
design keeps the arena solve exactly as it is and changes everything around
it.

## 1. The problem, as measured

All numbers come from `activity-step-2026-11-05/progress-stages.md`, rounds 2,
6, 7, 11 and 12. The table data is `round12-kernels.tsv`, 8x8x4 `--pure`,
with per-dispatch raw timestamps.

- **A 2-cell drop costs 12.7 ms of GPU time over rest. 8.6 ms of that is
  outside pressure.** The live work is 123 fine tiles, 115 bricks, 10k
  particles and 87 non-COLD regions: about 8k h cells. That is about
  1 µs a live h cell, roughly 1000x the bandwidth bound.
- **About 8 ms of it does not grow with the disturbance (round 2).** drop=8
  has 4x the fine tiles and 7x the particles, yet costs only +4.9 ms. 201
  kernels barely move. It is a switch-on tax.
- **It is latency, not work (round 6).** Counters show 9-12% occupancy,
  ALU around 10%, bandwidth 1-2% of peak and no limiter over 12%. The
  drains between dispatches are at most 14%. The other 85% is resident
  kernels stalled on dependent loads.
- **The frame is a real chain (round 7).** Its RAW depth is 350 of 475
  dispatches, so levelling or merging independent stages removes at most
  26%. Removing launch boundaries alone is worth about 0.5 ms (round 11).
- **Each stage's cost is its own dependent-load chain (round 12).** For
  example:
  - classifyCoarse does up to 96 serial loads a lane;
  - pUp4 makes 8 parent lookups;
  - advectDeferred runs 4 dependent sampling stages before its trace, and
    pays the list scan 5 times;
  - pair_sweepSeams does a neighbour search per tap;
  - the slab sort takes 22 global merge passes.

Non-pressure excess, by what it serves (µs, approximate):

| what it serves | µs | stages |
|---|---|---|
| adaptivity bookkeeping: census, residency, layout, remap, frame plan, local speed, hanging taps, 4h cache, momentum tiers, transfers, geometric fill | ~2870 | ~60 launches |
| level set: surface advect (4h + h, deferred twins), resolve, depthCoarse, redistance arena, air plan, volume | ~2040 | ~40 |
| extension velocity: paired sweeps, hierarchy restrict/sweep/prolong, retain | ~1060 | ~25 |
| particles: advect, slab sort (22 passes), seams, couple, separation, transfer | ~1020 | ~35 |
| momentum, forces | ~400 | ~7 |
| tail of kernels under 50 µs | ~1200 | ~80 |

**Root cause.** The step is organised by stage. Each of about 250
switched-on stages walks the active set and re-resolves sparse addresses
for every item:
- page table, then block, then fine slot, then texel;
- hanging taps, seam values and neighbour searches at every h/4h boundary.

Each stage then reloads its inputs from global memory, one item a lane.
With a small active set nothing hides that latency, so the frame costs
about (number of stages) × (one item's dependent chain), whatever the
amount of work. Seam and hanging logic is in 63 files of `lib/methods/uniform`.

The same machinery also exists three times:
1. NB-FLIP's brick table is rebuilt every frame.
2. The pressure brick arena is persistent and has staged halos.
3. The fine brick store (7a6cc25a) is persistent and has aprons, but it is
   still unwired.

On top of these sit the paged h textures and the detail fine slots. Every
frame pays to keep all of them consistent.

## 2. The change in one paragraph

**The brick becomes the unit of storage and of execution.** Every live cell
outside pressure lives in exactly one persistent brick slot: a level-0 brick
is 8³ h cells, and a level-2 brick is a page of 8³ 4h cells. These are the
slot space and records the pressure arena already uses
(`uniform-brick-arena.ts`).

The non-pressure frame becomes **four fused brick passes**:
- **Launch:** one dispatch each, one workgroup per active brick.
- **Inside a pass:**
  1. The group stages its 27-neighbour record and its core plus halo into
     threadgroup memory once.
  2. It runs its whole stage chain in place, with `workgroupBarrier`
     between stages.
  3. It writes only its own core.
- **Seams:** the h↔4h seam rule runs in one place, the halo staging, and
  no stage sees a seam.
- **Topology:** brick enter and leave stay incremental. They are the arena's
  existing `ubaUpdate`, and they run only when marks change. Unchanged
  bricks keep their data in place, so they need no remap and no re-layout.

This is the arena's execution model, which pressure already uses, extended
to the rest of the frame.

## 3. Design

### 3.1 One store

- **h data:**
  - Phi, MAC velocity and its banks, apertures and fractions, the cell
    scalars and particle slabs live in level-0 brick slots of one pool.
  - That pool is the arena's slot space, which `uniform-fine-brick-store.ts`
    already shares with pressure.
  - The block-sized h textures, the detail fine slots and the hanging-tap
    tables are deleted.
- **4h data:** the 4h block grid already is level-2 bricks (slot = block).
  It stays, and its readers move onto the same records.
- **Sampled channels:** u and phi keep the store's 3D atlas with a one-texel
  apron, so trilinear taps stay hardware-filtered (§2.2 of the huge-grids
  plan; worth 1.6x on the traces).
- **Particles:** they live only in their brick's slab, as today. A slab is
  kept in cell order by the brick's own pass (§3.4). The global 22-pass
  slab sort goes away.
- **Capacity:** pool sizes are authored per scene from the measured peak
  (`fluid.tiers.bricks`) and do not grow at run time. Overflow is the
  arena's fatal word.

### 3.2 One record, one seam rule

- **The record:** each slot's 32-word record (27 neighbours, parent, owned
  mask, key) is the only addressing a kernel ever does. It is resolved when
  the brick set changes, not per access.
- **Staging:** `ubaStage(level, channel, halo)` fills a threadgroup window
  of (8+2k)³ for a channel. Lane i loads one value, by one of these cases:
  - from its own core;
  - from a neighbour's core, through the staged record;
  - by the seam rule, when the neighbour word is `UBA_SEAM`:
    - for an h value under a 4h owner, it prolongs from the 4h page;
    - for a 4h value over h owners, it restricts with V-weights;
  - SOLID or AIR by the record's kind.
- **Seam semantics move into staging, unchanged.** These are:
  - the reflux of a coarse face as the V-weighted sum of its 4×4 fine
    subfaces;
  - the d7fa24db seam pressure gradient;
  - the hanging vertex rule.

  They are ported once, into staging, and the per-stage seam kernels are
  deleted. These include transferSeams, sweepSeams, fineSeamSweep, the
  hanging taps and the seam jobs.
- **Threadgroup budget:** 32 KiB. One scalar field at halo 1 is 10³ × 4 B =
  4 KiB, and at halo 2 it is 6.9 KiB. A pass stages only the fields its
  current stage reads. Read-only inputs go in as f16 where their tolerance
  allows (shader-f16 is present). An oversized window marches z-slabs:
  3 slices live, the 2.5D stencil blocking.

### 3.3 The frame: four fused passes around the unchanged pressure solve

The launch list is one dispatch per pass over the active brick list, at
both levels, with one group per brick:
- **Group shape:** level-0 and level-2 groups share each dispatch, and a
  small indirect list orders them.
- **Lanes:** 256 a group. The move's sampler already caps registers at 384
  threads, so 512 would spill.
- **COLD regions:** a group whose bricks' regions are all COLD exits at
  once. The global idle advance is then just the case of an empty list.

| pass | reads | stages inside (in order) | writes | why it can be one pass |
|---|---|---|---|---|
| **B1 transport** | last frame's u, phi, particles (read-only) | G2P taps; particle advect; phi semi-Lagrangian advect (4h and h, one code path with the cell width as a parameter); velocity advect (momentum); each brick's local-speed max by subgroup max and one atomic | particles appended to their destination brick's inbox (or kept in place); phi*, u* (core) | every input is the previous frame's, so no same-frame halo exists |
| **B2 surface** | B1's phi*, inboxes, neighbour slabs | inbox merge plus a counting sort by cell in threadgroup memory (512 bins); reconstruct (Eq. 4 union) over core+halo; clearAir and retire; redistance; apertures and fractions; restriction to the 4h parent | phi, slab, geometry | each stage reads the brick plus a halo of B1/B2 values; halo k covers k chained radius-1 stages |
| **B3 momentum → RHS** | B2's geometry, slabs, u* | P2G as a **gather**: each cell gathers from its own and its neighbours' slabs within the kernel's support, with no cross-group atomics; authority; forces; extension retain; the rows the pressure arena reads (rhs, coefficients) | u*, arena rhs and coefficient channels | P+1 stencils on B2 outputs |
| *pressure* | | *unchanged: arena solve and project* | | |
| **B4 projection → tail** | projected u | band project; extension sweeps in threadgroup memory; dynamics publish; support plan and certificate for the next head; region-heat evidence (omission, speed, phi revision) reduced per brick; receipts by atomics | extended u, the next frame's marks and heat inputs | P+sweeps on the projected field |
| **T topology** (frames whose marks changed) | B4's marks | `ubaUpdate` (enter/leave), records for changed slots only, prolong into entering slots, restrict out of leaving ones | slots, records | already incremental in the arena |

**Census.** Today the census is about 10 launches, including
classifyCoarse's 96-load lanes. It becomes a reduction of evidence that B4
already holds in threadgroup memory. The tier and owner decisions are
brick-local functions of that evidence, applied in T.

**Launches.** About 4–6 non-pressure launches a frame, plus T on frames
whose topology changed. Today the switched-on frame adds about 250.

### 3.4 Determinism by construction

- The in-brick sort of B2 orders by (cell, stable particle key), not by
  arrival.
- B3's gather P2G sums in a fixed order.
- Neither depends on atomic order across groups.

That removes the class of divergence the pressure lane chased in rounds 7
to 10: seed-budget clips, atomicAdd order and the launch-order rank. Its
fixes (2a17b693, 7f8d76c1) remain correct and become simpler.

### 3.5 What is deleted

- The paged h textures, the detail fine slots, `umFineSlot` addressing and
  the hanging-tap tables.
- NB-FLIP's per-frame brick table rebuild.
- Every seam kernel and seam job, outside the staging rule.
- The 4h/h twin pipelines: surface advect twins, phi twins, the deferred
  lists scanned once a code.
- The global slab sort and slab-allocate chain.
- The remap of h data for unchanged bricks, and the layout builder's
  h-slot allocation.
- The frame plan's dilation chains. Brick marks dilate once, in
  `ubaUpdate`.
- The census's own passes.

This is a large net code removal. Every stage becomes a device function on
a staged window, rather than a kernel with its own addressing.

## 4. Expected cost

- **The evidence** is `tools/bench-one-pass-page.ts` (49517a7d), at small
  N, which is the regime that matters here:
  - fused one-group-a-page: 48 µs at 64 pages;
  - 14 separate dispatches: 152 µs;
  - launch floor: 3–4 µs.
- **Real kernels are heavier.** They average 26 µs a dispatch against the
  bench's 11, which is why the per-stage chain is the larger lever. A fused
  pass pays one item's chain once per pass, from threadgroup memory, where
  today it pays it once per stage, through global dependent loads.
- **Estimate for 8x8x4 drop=2** (about 115 level-0 and about 15 level-2
  bricks):
  - each of B1–B4 takes 50–200 µs;
  - T takes about 50 µs on change frames;
  - **non-pressure excess 0.3–1.0 ms, against 8.6.**
- **At scale** the fused bench grew 0.5 µs a page per chain. For a vast seep
  with about 1.1k h tiles (about 600 bricks) that is about 0.3 ms a pass.
- **A pessimistic 3x** still leaves the non-pressure excess under 3 ms.

## 5. Risks and open questions

1. **Redistance and extension reach beyond a small halo.**
   - The concern: if redistance's band (up to 2 tiles) were sweeps of
     radius 1, an exact fused pass would need halo 8.
   - First, determine what the arena redistance actually does
     (`arenaDistance`, `arenaNearest`, `arenaFar`). A closest-point search
     over staged crossings needs only the band's crossings in the window.
   - Fallback: run it as a fixed point lagged across frames. Halo values
     are the neighbour's previous iterate, which is the standard
     incremental redistance, and the fixed point is unchanged.
   - This needs a regression on the result rather than the order: phi
     within its existing tolerance on the letters and dam bands. The same
     applies to extension sweeps beyond halo 2.
2. **Seam semantics in staging must be identical.**
   - The test: run staging against today's seam kernels on recorded
     frames (seep, letters, drop=2, dam), field by field.
   - Volume drift on seep 300 and letters 120 is the gate.
3. **Threadgroup memory and registers.**
   - The risks: fused chains want many live fields, and the move already
     spills past 384 threads.
   - The mitigations: per-stage staging, f16 read-only inputs, z-slab
     marching and 256-lane groups.
   - Measure occupancy with the `gpu-frame-profile` skill on the first
     pass built.
4. **Load imbalance.**
   - The risk: a splash brick can hold 10x the particles of its
     neighbours.
   - The mitigation: B1 and B2 stride the brick's particles over lanes,
     not one particle a group, and a brick over a slab threshold splits its
     particles across a second group by a flag in the indirect list.
5. **Compile time.**
   - Fewer, larger pipelines.
   - Gate them with `check:gpu-compilation-policy` as today.
6. **Contract with pressure.** B3 writes the channels the arena reads
   today (rhs, coefficients, band rows), and B4 reads its projected u. The
   arena's code is not touched. Any interface the arena reads from paged h
   textures moves to slot reads through the same records it already uses.

## 6. Order of work and gates

Each step lands on main only after:
- `npm run check:types` and `npm run test:unit`;
- seep 300 and letters 120 (drift and step time);
- the tank at rest, 60 steps;
- 16x8x4 rest against drop=2, one run each, pipelined.

Old paths are deleted in the same commit that replaces them. No dual paths
are left behind.

**Step 0: B1, the go/no-go.**
- **What:** build the transport pass for real on drop=2. It needs no
  same-frame halo, so it tests the execution model without the seam port.
- **What it replaces:** particle advect, the slab sort chain, surface
  advect and its deferred twins, resolve, momentum advect and local speed.
  That is about 2.0 ms of round 12's excess.
- **Correctness:** the cross-scene gates (seep 300 and letters 120 drift,
  the rest tank's drift), plus a visual sanity check of drop=2. No parity
  is required.
- **Go:** that family drops to 0.4 ms or less on drop=2 (5x), with drift
  inside the bands.
- **Checkpoint, not a stop:** if it is under 3x, attribute the pass stage
  by stage, find the mechanism, then optimize and remeasure. The user, on
  2026-11-08: "remember not to give up when an approach doesn't first
  work. understand, optimize and refine".

**Step 1: staging and the seam rule.**
- `ubaStage` with the ported seam semantics.
- Checked by its consumers' gates (drift, rest). It does not need
  field-by-field parity with today's seam kernels.

**Step 2: B2 surface.**
- Resolve the redistance question (risk 1) first.
- Then reconstruct, retire, redistance and geometry, and the in-brick sort
  replacing the global one.

**Step 3: B3 momentum → RHS.**
- Gather P2G, forces, authority, and the arena's input channels.

**Step 4: B4 and the census as a reduction.**
- Extension, publish, the support plan, and heat evidence.
- T on change frames only.
- Delete the census passes, the layout builder's h slots and the remap
  for unchanged bricks.

**Step 5: delete the paged h store.**
- The h textures, the detail fine slots, the hanging tables and the
  per-frame NB brick table.
- Memory falls with them.

**Acceptance:**
- 16x8x4 drop=2 within 2 ms of rest outside pressure (from about 9–10).
- drop=8 minus drop=2 tracks bricks and particles.
- Rest, seep and letters step times no worse, with drift inside today's
  bands.

## Outcome, 2026-11-08 (wrap-up)

Lane learnings: `brick-resident-step-2026-11-08/progress-lane-{t,e,c}.md`, each ending in a LEARNINGS section.

**Landed:**

- **S3** (1cbebf88): the particle sort is one counting sort per brick.
- **S2** (8822b2d8): momentum is one claimed launch, with COLD retain.

| | before | after |
|---|---|---|
| seep step | 31.6 ms | 31.0 ms |
| letters step | 58.0 ms | 54.0 ms |
| rest16 step | 3.79 ms | 3.28 ms |
| drop16 step | 16.47 ms | 15.69 ms |
| B1 family at drop=2, 8x8x4 | 2.18 ms | 1.82 ms |
| non-pressure excess at drop=2, 8x8x4 | 8.87 ms | 8.71 ms |

**Built but unwired (helpers):** f1da3003 (ubaStage), d78a265f (ubaExtend), 26c969e4 (UniformFrameBricks). The tools are in a4114158 (compile attribution).

**Not landed (in scratch):**

- **S1**, one surface-advect launch. It takes 640 s to compile cold.
- **B4**, the extension rewire:
  - Step time: seep 43.7 ms and letters 75.8 ms.
  - Cause: ufx4's publish visits all 32768 texels of each page to write 1536, 3.87 ms at drop=2.
  - Patch: `fluid-scratch/lane-e/b4-final/`.
- **The fused B1 kernel.** Its traces are simplified. Saved in `fluid-scratch/lane-t/saved/`.
- **The momentum windows m1, m3, m4.** They were slower at seep, and the mechanism is unknown.

**Key findings:**

1. **The drop=2 excess is 192 kernels with a mean of 42 µs.** At drop, every kernel runs about one wave, so its cost is the length of its dependent chain. At seep it is throughput, over about 6 waves.
2. **Compile cost.** Seep needs 327 s of compile work done one at a time. Of that, 164 s is the three solid-aware surface-advect pipelines, where Metal's back end is superlinear in the size of the inlined umAdvected. Dawn runs only 2 compiles at a time, while a plain Metal process manages about 7.
