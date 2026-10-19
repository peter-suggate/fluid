# Particle brick slabs: the NB-FLIP particle stage on brick-local storage

2026-10-19. Target: `nbflip-figure-7-pour` at 1/60 s, under 17 ms a frame on the M1 Max.
Today (run vg1) the frame is 59.0 ms. The particle stage takes about 10.1 ms of it, at 818k particles, 7688 fine tiles and 57,848 coarse tiles.

Companion plans: surface-brick-arena-2026-10-19.md (it owns the shared brick table) and momentum-velocity-atlas-2026-10-19.md.

## A. Today's passes

Each frame the particle stage runs these passes (vg1 means, in ms):

| Group | Pass | ms | What it does |
|---|---|---|---|
| move | advectParticles (fast) | 2.013 | Substepped RK3 trace through the unit-tap fast sampler, hull proof, in-place store, a cell-count atomic and a coverage atomic per particle |
| | advectParticlesDeferred | 0.946 | The ~71k listed samples through the general sampler and the voxel walk |
| | order: clear, offsets, pack | 0.441 | Clears the occupied bins, scans tile and cell offsets, scatters every record to its cell run in the other bank |
| | markParticleTiles + commitCount | 0.262 | Compact positions, surface-tile and theta marks, tile heat |
| transfer | classify | 0.393 | Ballistic bit, compact motion, census |
| | transferDeposit | 1.123 | Tiled quadratic MAC scatter, flushed to six words per fine owner |
| | transferProperties | 0.580 | Blend weights and finish per fine cell |
| | transferSeams + owners + transfer | 0.581 | 4h owner faces gathered from cell runs |
| snapshot | cache, unitVelocityTaps, snapshot | 1.423 | u* sampled at every particle into `before` |
| update | cache, halo, taps, update | 1.226 | u sampled, the FLIP blend, depth, sparse age, heat |
| resample + seed | resampleCount/Prefix/resample, seedCells, seedSources, seed | 0.861 | Compacts the survivors into the other bank, then appends seeds |
| census | surface refinement | 0.655 | Reads every record for the residency closure |

The trace spends most of advect's time on the substep machinery: the gradient probe, up to 128 base substeps and the per-span page-face clamps. The data spends the rest: a full record copy each frame for the order and another for the compaction, plus one global device atomic per particle per count.

## B. Speed of light

The measured M1 Max ceilings are in the arena plan, §2a. The standalone bench (`scratchpad/pbench/particlebench2.mts`) runs the whole particle job on brick-bucketed records. At 818k particles and 1860 bricks it measured:

| Kernel | Bound (ms) | Bench (ms) | Notes |
|---|---|---|---|
| FUSED: G2P + one-step RK3 + hull + append | 0.27 | 1.507 (TW16) | Knock-outs: no seed search 0.852, no tail 0.721, no hull 0.59, no samples 0.262, no window fill 0.131 (stream floor) |
| DEPB: deposit, a workgroup per (brick, axis) | 0.61 | 0.59 | 1100-node shared window, fixed point |
| FIN: normalise, blend, write | 0.18 | 0.197 | |
| Velocity atlas (if separate) | 0.08–0.18 | 0.197 | |
| **Chain** | **~1.15** | **2.36–2.62** | FTW32 2.36, FAW (exact) 2.62; the v1 baseline was 3.67 |

With slack 1536 the bench saw 0 spills, about 1100 samples on the slow lane and 95k seeded.

One RK3 step of any length against the substepped reference: p99 error 1.70 cells for fast samples, max 2.76. That is accepted under the never-limit-step-sizes rule.

The bound for this stage is therefore about **1.15 ms**, against about 10.1 ms today (9× off). Every phase reports its frame ms and particle-stage ms as a percentage of that bound. Anything more than 2× off names its limiter.

## C. Design

### Data

- **Brick.** 8³ cells of a page, id 64·block + (bx | by<<2 | bz<<4), as in uniform-brick-table.ts. The particle stage keys everything it keeps across a frame by brick id. Table slots are rebuilt by several stages a frame and are not stable.
- **Brick table.** The arena's shared table, reused as is. The particle stage clears it, marks the bricks it needs (class `particlesCore`: bricks holding particles or spilled particles; the build dilates them into `particles`, the ring that can receive a move) and builds it at the head of its move. It reads the slots only inside that move (ALLOC, the advect launch, SORT). Later builders, such as the arena's search, rebuild it freely.
- **Two banks, fixed roles.** `particles[1]` is the **current** bank: cell-sorted within each brick slab, read by transfer, splat, snapshot and update, and by the overlay and refinement. `particles[0]` is the **scratch** bank: advect appends into it, SORT drains it, and the paired extension borrows it for the rest of the frame. Parity no longer flips.
- **Slabs.** Each listed brick gets a slab [base, base+cap) at the same offsets in both banks.
  - cap = demand + max(32, demand/4). demand = the brick's live records last frame plus the appends it refused.
  - base is the exclusive prefix of cap over the table's slots, computed by one workgroup.
  - Σcap must fit the capacity minus the spill reserve; otherwise it is a fatal (state word 3 = 3).
  - Holes past a slab's live count hold dead records: position (−1, 0, 0, 0). The overlay and refinement already skip heat ≤ 0.
  - The extent (Σcap) is state word 0: every flat consumer strides it and skips dead records. State word 1 is the live count.
- **Per-brick record** (in the state buffer, by brick id): base, cap, cursor, stamp, previous count, refused appends. stamp = the move's serial, so a brick without a slab this frame is known as such.
- **Input list.** ALLOC snapshots last frame's slabs (id, base, live count) before it allocates the new ones. The advect launch strides this list.
- **Spill list.** A fixed reserve S at the tail of each bank, S = max(4096, capacity/16).
  - It takes the records that found no room: no slab at their brick, or a full slab.
  - Advect spills land in the scratch tail. SORT copies them to the current tail, where seeding spills join them.
  - The next move re-advects them first-come through the slow lane, and marks their bricks so the table lists them.
  - A spilled record misses one frame of transfer, splat and update.
  - A full reserve is a fatal.

### Kernels and frame placement

**Phase 1 (move at the head, as today):**

1. **slabMark → table → ALLOC.** Mark the input list's live bricks and the spills' bricks. Build the table. One workgroup allocates the slabs and snapshots the input list.
2. **advect.** A workgroup per input slab, 64 lanes, today's trace and collision code at one call site.
   - Each finished sample appends to its destination brick in the scratch bank, through a workgroup histogram over the 27 neighbour bricks and one device atomic per neighbour per turn.
   - A destination outside the 27, or a slab with no room, spills.
   - Samples the fast sampler cannot serve are listed. The deferred launch takes them, and the spills, with direct atomics.
   - No per-particle global count or coverage atomic remains.
3. **SORT** (replaces order's clear, offsets and pack, plus markParticleTiles and commitCount). A workgroup per output slab.
   - It counts the slab's samples per cell (512 cells in workgroup memory), scans, and scatters each record into its cell run of the current bank at the same slab offsets.
   - It writes the compact positions, the cell starts and counts, the tile coverage and occupied list, the surface-tile and theta marks (a 4³-tile window per brick) and the tile heat. Then it dead-marks the holes.
   - Every consumer of cell runs (nbRun, the tiled deposit, the splat, sparse age, seeding) reads exactly what it reads today.
4. **transfer.** DEPB and FIN replace classify's motion, transferDeposit and transferProperties.
   - DEPB: a workgroup per (listed brick, axis). The brick's slab and its neighbours' are read through the table, scattered into an 11×10×10 fixed-point window, and flushed to the fine owners' accumulators.
   - FIN: per fine owner. The seams keep their gather.
5. **update** strides the extent and skips dead records. A small **retire** pass applies the old resample's keep rule in place: depth < −4, heat ≤ 0, sparse-expired, or the crowded deep overlap (cell rank ≥ 16 below depth −1). It dead-marks the record and decrements its cell count.
6. **seed** appends each seed into its brick's slab cursor in the current bank. No room, or no slab, spills.

The compaction is gone: the next move drops dead records, because the advect does not append them.

**Phase 2 (the move moves to the tail).** The tail extension moves ahead of update, so velocityScratch at update time is the field the next head would advect through.

- **FUSED** replaces update, retire, advect and the deferred launch. A workgroup per current slab:
  1. It samples pic = u(x) (k1 of the trace) and applies the FLIP blend with `before`, or the ballistic kick with its axis stops.
  2. It computes depth, sparse age and heat, and the retire rule.
  3. It takes one RK3 step (k2, k3; no substeps, no NARROW_BAND_TRACE_LIMIT, no gradient probe).
  4. It proves the hull (`utcSolidClearance` and the open-cell bits).
  5. It appends to the scratch slabs.
- **The long lane** takes what the hull cannot prove, and the class bits that need it: blend sampler, umHold walls, nbSampleCold, rigid contact and bed slopes. It walks the chord in strides of at most ½ cell an axis, with today's per-axis stop semantics.
- SORT and seeding follow at the tail, so the next frame's transfer reads a sorted current bank.
- It runs after bodies.couple: solids and rigid bodies are updated first. The move uses this step's dt. With the fixed step every NB-FLIP scene uses, that is the next step's dt.
- Snapshot's per-particle pass folds into FUSED once the extension's u* companion has its own buffer. Today it borrows the scratch bank that FUSED writes.

**Phase 3, if time.** Seeding from a per-brick deficit list, launches over the occupied-brick list only, and subgroup appends.

## Phases and gates

Each phase:

1. `npm run check:types` and `npm run test:unit`. Unit tests of retired behaviour are deleted.
2. One profiled pour (`P/pr.sh partN`), compared with `node P/cmp.js vg1 partN`.
3. Commit.

| Phase | Retires | Expected particle stage |
|---|---|---|
| 1 | order (clear, offsets, pack), markParticleTiles, commitCount, resample/compaction, classify motion, transferDeposit, transferProperties; parity flips | ~10.1 → ~7.5 ms |
| 2 | update, retire, advectParticles(+Deferred), trace substeps and limit; snapshot pass (with the companion buffer) | → ~3 ms |
| 3 | seedCells scan, idle-brick launches | → ~2.5 ms |

## Risks

- **Slack.** A jet front entering an empty ring brick finds 32 slots. The rest spill and are re-advected next frame, now with demand-sized slabs. The spill reserve is sized for a source's first frame; past it, the run stops on a loud fatal.
- **Block reuse.** A brick id names a block. A block freed while its slab holds live records would drop them, so ALLOC makes a live input brick without a slab fatal.
- **k1 = pic.** The trace's first stage reuses the update's sample of the extended field. If air faces near the surface make that wrong, k1 falls back to its own sample (+0.1 ms).
- **Tail move and the head census.** The head census predicts one frame of travel from the positions the last tail produced. That is the same structure as today; a long jump still meets the arrival fatal.
- **Variable dt.** The tail move uses dt_n for the move of step n+1. NB-FLIP scenes run a fixed step; a varying dt would be a physics change.
- **APIC** (experimental) keeps working: its records are larger, and every pass moves whole records.

## Invariants

- No indirect dispatch. Every launch is a fixed direct grid striding a GPU count.
- No JavaScript allocation in the advance path. Parameter words are written in place.
- Every pipeline is built at initialize. Nothing compiles during simulation.
- Fail fast: slab extent over capacity, a full spill reserve, a live brick without a slab and the existing trace, nonfinite and arrival fatals each stop the run with a loud message.
- Memory is allocated upfront. The per-brick record covers every brick of the lattice.
- No blocking readback.
- One call site per heavy WGSL function: the trace and its sampler, the collision walk and the seed site.
- Cell runs are contiguous and in tile-local order inside a tile, so every nbRun reader is unchanged.
- Dead records have heat ≤ 0. The live test everywhere is position.w > 0.
