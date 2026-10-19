# Pressure and RFE: job analysis and speed of light, 2026-10-20

This fills the pressure and RFE sections of `speed-of-light-plan-2026-10-20.md`.
It was produced read-only from code and the profiled pour run `part2h`, with no new GPU runs.

## Inputs

**Scene.** `nbflip-figure-7-pour`, frames 94–100, dt 1/60 s, profiled split run `part2h`.

**Counts.**
- 6004 h tiles, which is 384,256 h cells and 1.15M h faces.
- 57,484 4h tiles, which is 172k 4h faces.
- 441,740 owners in all.
- The tank is 128×256×128 h, which is 128 pages and 65,536 4h tiles.
- 1143 remap jobs a frame.
- 695k particles.

**Ceilings** (`scratchpad/ceil/ceilings.json`):

| What | Measured |
|---|---|
| Streaming | 360 GB/s, i.e. 2.8 µs per MB |
| L1-window gathers | 34 G/s |
| 4 MB gathers | 15.5 G/s |
| Workgroup memory | 719 G/s |
| Device atomics | 5.7 G/s |
| Per pass | 2.3 µs |
| FMA | 3.8 TFLOPS (latency-limited) |

**Band size.** The band's slot count is not in the run's receipts. It is at most 6004 (every h tile with a wet cell). Bounds below use 6004 slots, i.e. 384k rows. That makes the band bounds upper estimates, so the ratios are, if anything, flattering.

**Timer resolution.** Per-dispatch times are quantised to 65.5 µs. Means over 7 frames and many dispatches are reliable; a single dispatch is not.

**Later changes.** Since part2h, b7c3f514 (regional geometry grid) measured geometryRegional at 1.24 ms against part2h's 1.68.

## Summary

| Stage | Today (part2h) | Bound: today's algorithms | Bound: minimum work | Arena projection | Arena vs minimum-work bound |
|---|---|---|---|---|---|
| Pressure | 8.24 ms (155 dispatches) | 1.5 ms | **0.26 ms** (2 cycles); **0.41 ms** at 4 cycles, see §1.6 | 0.40–0.50 ms; ≈0.6 ms after §1.6 | 1.5× of the 4-cycle bound |
| RFE | 8.71 ms (58 dispatches); 8.3 after b7c3f514 | 0.6 ms | **0.18 ms** | 0.30–0.35 ms | 1.7–1.9× |

The umbrella plan's rough bounds were pressure ≈0.6 ms and RFE ≈0.3 ms. With real counts they are 0.26 ms and 0.18 ms. Those figures assume the minimum-work algorithms in §1.4 and §2.4.

Three findings set the route.

1. **Pressure cannot reach 2× on today's solver, however lean its kernels.**
   - The band's fine red-black sweep already runs at **1.8× of its own bound**: 27 µs a half sweep against 15 µs.
   - Its problem is how many there are: 4 V(3,3) cycles, i.e. 48 half sweeps plus 2 settle sweeps, plus a separate 4h root V-cycle.
   - Today's algorithms bound the stage at 1.5 ms, above the 1.2 ms target.
   - The lever is the solver: one MG hierarchy over h and 4h bricks, with smoothers resident in workgroup memory, and 2–3 cycles.

2. **Everything outside the sweeps is 8–45× off its bound**, for the same reasons in both stages:
   - page-table walks per element (umOwnerAt, umCellKind, bCellAt, umVertexSlot);
   - predicate chains per face (wet, frozen, shut, inside, θ, V);
   - one lane per owner over a whole-pool list;
   - the same classification and transfer repeated by several passes.

   These are exactly what the arena removes.

3. **Both stages do large redundant jobs:**
   - the h band is solved twice (as 4h aggregates in the root, then at h);
   - h faces are projected twice, with a whole-texture copy between;
   - wetness is classified by four passes;
   - the extension runs twice (main and snapshot), each with its own 4h pyramid;
   - geometry is filled for two ownerships.

## 1. Pressure

### 1.1 Structure today

It is a two-stage solve, run every frame.

**Authority** (`uniform-mixed-pressure-authority.ts`), 12 dispatches over 3 invocations:
- Two invocations on the simulation ownership (stamp, stampHeld, surfaceHeld).
- One on the all-4h pressure ownership (stamp, stampHeld, cutRegions, rows, build, chunks, reduce, resolve, wet).

Together they decide:
- which owners are pressure rows (centre phi, plus V claims over capacity);
- detached mass;
- the volume-control balance, which is a reduction over all owners;
- the least-squares phi plane of cut tiles held at h;
- the root's wet topology (which h faces a cut 4h face keeps).

**Root: the all-4h solve** (`uniform-mixed-pressure-cycles.ts`, `-velocity.ts`).
- Owners: about 63.5k resident 4h owners. Levels are 4h, 8h, 16h and 32h.
- Flow:
  1. Transfer to pressure (restricts the simulation velocity to 4h faces; a cut face takes Σ V_h u_h).
  2. rhs.
  3. setup: the per-solve scratch and job list.
  4. measure.
  5. A CM11a LCP multigrid V-cycle.
  6. Gates 1–3: up to 3 Full-Cycles and 4 V-Cycles budgeted. The pour opens one V-cycle; the other gates are closed capacity slots.
  7. project (the 4h faces).
  8. Transfer to simulation.
- Tolerance: 5 s⁻¹.
- A smoothing visit is 9 Jacobi iterations (seven at weight 1, two at 0.5), 3 to a launch.
  - Each launch stages a 4³ octant core plus a 3-cell halo (10³) in workgroup memory.
  - The visit runs before and after each coarse correction: smoothPages ×6 at 4h, smoothTiles ×6 at 8h.
- The 16h and 32h levels and their 16 red-black SOR(1.6) sweeps share one single-workgroup launch (coarse).

**Band: the h re-solve** (`uniform-pressure-band.ts`).
- Band tiles: every h tile with a wet cell (not frozen, not WARM).
- The band list is rebuilt each frame: list, prefix, offsets, scatter.
- Rows (prep):
  - 10 colour-major planes of 4 B per cell: rhs, diag, kinds, 6 coefficients and base;
  - plus 240 packed faces per slot of u*, and V with solids.
- Coarser levels: 2h aggregates (8 a slot) and 4h aggregates (one a slot), baked by middleBake and coarseBake (Galerkin, piecewise constant, couplings ×0.6).
- Start: init and rebase. The start pressure is the liquid-weighted trilinear 4h pressure; Neumann faces take the transferred 4h velocity.
- Two settle half sweeps run under solids.
- Cycles: **4 V-cycles** of 3 pre and 3 post red-black h sweeps.
  - The 2h level gets 2 sweeps each way (middleSweep).
  - The 4h aggregates are solved in one workgroup: 8 red-black SOR(1.4) sweeps (coarseSolve).
- Exit rule "returns": stop when the worst residual is at most 0.05 s⁻¹ and the last cycle cut it by less than 1/ratio.
  - On the pour, all 4 encoded cycles do real work: the per-position sweep times are equal in every cycle.
- Then band projection: a copy of the velocity texture, then a projection of the band faces.

### 1.2 Per pass

Notes on the bound column:
- Every bound is today's job at today's iteration counts: bytes at streaming rate plus 2.3 µs a dispatch.
- Assumed row traffic for a red-black half sweep: rhs, diag, kinds and base (16 B) plus a 4 B write. The six neighbour reads are cached. Coefficients are read only on non-unit faces.

| Pass | ms | Disp. | Computes | Irreducible job | Bound ms | × | Limiter today |
|---|---|---|---|---|---|---|---|
| Surface pressure authority | 1.311 | 12 | Rows, detached mass, balance, cut planes, root wet topology | 441k owners: phi, V and fill in, a phi word out. 63.5k root topology vec4s. ≈9 MB | 0.03 | 44 | Three invocations re-derive one classification. The held passes stride every page. Neighbour phi goes through umOwnerAt. cutRegions runs serial least-squares loops. The balance reduction is over all owners. |
| Band list, rows, bakes (bandPrep 0.946 and 6 small) | 1.076 | 7 | Band list; 10 row planes; 2h/4h operator bake | 384k rows × (7 phi, 6 u*, 6 V in; 40 B rows and 240 faces per slot out) ≈ 25 MB | 0.08 | 13 | Per face: umCellKind and umFrozenCell at the tile face, bNeighbour (owner, width, phi), bFacePhi/θ, bCellAt (slot map through the page table), umPressureFaceV / umCellIsOpen / umCellInsideSolid. That is about 6 × 5 dependent gathers per lane. |
| bandInit and bandRebase | 0.346 | 2 | Start pressure from 4h; Neumann u* | 384k × 8 B, plus 1.4M packed faces × 4 B ≈ 9 MB | 0.03 | 12 | Per-cell trilinear 4h gathers through the page table; Neumann face rewrites by tile. |
| bandSweep | 1.339 | 50 | 48 red-black half sweeps plus 2 settle | 192k rows × 24 B = 4.6 MB per half | 0.76 | **1.8** | **Iteration count.** The six neighbour reads are L1-window gathers (34 G/s, about 34 µs of gathers per half unless cached). The pass floor is 0.12 ms. |
| bandMiddleSweep | 0.309 | 28 | 2h red-black sweeps | 48k aggregates × 24 B per half | 0.11 | 2.8 | Launch count: 28 launches of about 1.5 µs of work each. |
| bandRestrict | 0.234 | 4 | h residual, restricted to 2h | 384k × 24 B per cycle | 0.11 | 2.1 | Streaming plus gathers. |
| bandCoarseSolve | 0.197 | 4 | 4h aggregates: 8 red-black SOR sweeps | 6004 rows × 8 × 32 B ≈ 1.5 MB per solve | 0.03 | 7.5 | One workgroup, i.e. one of 32 cores (by design: it avoids 16 launches). |
| bandMeasure | 0.122 | 2 | Worst residual | 384k × 24 B each | 0.06 | 2.2 | Streaming. |
| bandMiddleProlong, bandMiddleRestrict | 0.168 | 8 | 2h ↔ h, 2h ↔ 4h | Small | 0.04 | 4 | Launch floor. |
| Band projection | 0.262 | 2 | Copy of the velocity texture, then the band faces | 1.15M faces × 16 B ≈ 18 MB. The copy is redundant. | 0.05 | 5 | A whole-texture copy so the projection can read the old faces. |
| Transfer to pressure | 0.328 | 2 | h and 4h velocity → all-4h faces | 441k owner texels in (7 MB), 1 MB out | 0.02 | 15 | One lane per 4h tile; seam owners walk split planes serially. |
| rhs | 0.197 | 1 | 4h divergence | 63.5k × ≈50 B ≈ 3 MB | 0.01 | 16 | Per-owner face patch walks; solid record. |
| setup | 0.150 | 8 | Scratch, job list, far values, changed geometry | Mostly unchanged per frame | 0.02 | 7.5 | 8 launches each frame for state that rarely changes. |
| smoothPages | 0.477 | 6 | 18 Jacobi iterations at 4h, 3 per launch | 63.5k × 40 B ≈ 2.5 MB per launch | 0.06 | 8.5 | **Halo amplification**: a 4³ core with a 3-cell halo loads 10³ texels per 64 rows (15.6×). The core's 64 lanes wait on that load chain. |
| smoothTiles, descendTiles, ascendTiles | 0.356 | 10 | 8h visits and transfers | 8k rows | 0.04 | 9 | Launch floor; the same halo shape. |
| coarse | 0.272 | 1 | 16h and 32h levels: 16 SOR sweeps, one workgroup | 1.1k rows | 0.01 | 27 | One workgroup with serial storage barriers. |
| measure, checkpoints, gates, reserveTraversal, root owners | 0.321 | 17 | Convergence and capacity slots | Words | 0.04 | 8 | Launch floor. Two closed capacity slots still launch their threads. |
| project | 0.768 | 1 | Projects 4h faces and wall slots | 63.5k × 3 faces × 16 B ≈ 3 MB, plus cut faces' 16 subfaces | 0.02 | 38 | One lane a face through uniformMixedFaceDispatchWGSL: umFace neighbour walks, cut-face subface loops, solid record reads. |
| **Pressure** | **8.24** | **155** | | | **1.5** | 5.5 | |

### 1.3 Redundancies

- **The h band is solved twice.**
  - The root solves it as 4h rows; transfer to pressure restricts the h faces into those rows.
  - The band then solves the same cells at h, starting from that answer. The restriction, the 4h rows of band tiles, and the root's projection of their faces are thrown away.
- **Faces are projected twice, with a copy between.**
  - Root project, then transfer to simulation (0.197 ms; profiled under "other"), then the band's velocity copy and band project.
- **Wetness is classified four times:**
  - authority (three invocations);
  - the band list (bWet per cell);
  - bandPrep (bNeighbour per face);
  - middleBake (bLiquid).
- **Neighbours are resolved per pass through the page table:**
  - authority (umOwnerAt);
  - prep (umCellKind, bCellAt);
  - root setup (block neighbour rows);
  - root project (umFace).
- **Lists are rebuilt per stage, every frame:**
  - the band list (4 launches);
  - the root job list (setup);
  - authority's page strides;
  - the band's coarse bakes.
- **Pass count.** There are 155 pressure dispatches; their floor alone is 0.36 ms.

### 1.4 The minimum-work solver for today's accuracy

**Accuracy to meet.**
- Band: worst-cell residual at most 0.05 s⁻¹, about 2.5 mm/s at h = 5 cm, with today's cycle ratios.
- Root: tolerance 5 s⁻¹.
- p ≥ 0 for inside-solid rows (the projected update).
- Ghost-fluid θ at the free surface.
- CM11a V-weighted faces.

**Proposal: one geometric MG hierarchy over the arena's bricks, run as a single solve.**

*Levels*
- h fine bricks (8³);
- 2h (each fine brick restricts to a 4³ block of its parent);
- 4h: the coarse bricks, where the simulation's 4h region already lives, plus the restriction of fine bricks;
- 8h, 16h, 32h, ending at a cell per page.
- The h/4h seam is a level boundary inside one hierarchy, not a Neumann wall between two solves. So the root solve, its transfer and its projection disappear.

*Operators*
- Rediscretised coarse rows: a coarse face's coefficient is the V-weighted sum of its fine faces, which is today's root rule.
- Liquid-weighted full-weighting restriction.
- Trilinear prolongation, masked to liquid rows.
- Today's piecewise-constant aggregation with a 0.6 scale is a weak coarse correction. It is the likely reason 4 band cycles are needed.

*Smoother*
- Red-black GS, projected (max(·, −base)) on inside rows.
- 2 sweeps per brick load, resident in workgroup memory. Each brick holds p over its 8³ cells plus a 1-cell halo: 4 KB.
- Between the two sweeps, the brick's halo keeps the previous iterate (block-Jacobi across brick faces, i.e. an additive-Schwarz smoother). That costs a little smoothing per sweep against one load per two sweeps.

*Cycles*
- Expected: 2–3 V(2,2) cycles from the warm start, against 4 V(3,3) band cycles plus 1 root V-cycle today.
- The residual history the band already records (header words, one per cycle) gives the evidence at the fixed state. It is the first measurement of the pressure work.
- If standalone MG stalls at the free surface, as reported in the literature for irregular Dirichlet boundaries, the fallback is MG-preconditioned CG at V(1,1). The bound's traffic per iteration is the same order.

**Minimum-work bound.** Two V(2,2) cycles over 441k rows, 6004 slots worst case.

| Piece | Job | Bound |
|---|---|---|
| Rows and rhs (classification, θ, coefficients, divergence, volume control) | One brick pass. ≈40 B per row in and out: 17.6 MB | 0.05 ms |
| Fine level, per cycle | Pass A: pre-smooth, residual, restrict. Pass B: prolong, post-smooth. Each ≈21 B per row: 9.3 MB, 26 µs | 52 µs |
| Coarser levels, per cycle | About 1/7 of the fine level | 8 µs |
| Passes, per cycle | About 8 × 2.3 µs | 18 µs |
| Two cycles | | 0.16 ms |
| Projection | 1.33M faces × 12 B, fusable with the last post-smooth | 0.04 ms |
| Measure | | 0.01 ms |
| **Pressure** | | **≈0.26 ms** |

### 1.5 Pressure on the arena: kernels and projected fraction of bound

Every kernel is one workgroup per brick on a direct grid-stride over the arena's slot list, with no page table inside.

1. **pRows** (fine and coarse bricks):
   - Stage φ (cells plus a 1-cell halo), u* faces and V/solid fractions from the arena channels. Class bits come from the gather.
   - Write rhs, diag, kinds (unit-coefficient bits) and the non-unit coefficients (surface and solid rows only).
   - Write p₀: last step's p, or the prolonged coarse p.
   - It replaces authority (all three invocations), the band list, prep, init, rebase, both bakes, root rhs, root setup's job list and transfer to pressure.
   - The balance reduction becomes a workgroup reduction plus one atomic per brick, about 1.5k device atomics.
   - Projection: 0.07 ms against 0.05 (70%).
2. **pDown (level ℓ):**
   - Load p, rhs and row; 2 red-black sweeps in workgroup memory; residual.
   - Restrict 2³ into the parent's rhs channel.
   - Projection: 26 → 40 µs per fine visit (65%). The halo of 10³ / 8³ = 1.95× hits L2.
3. **pUp (level ℓ):**
   - Prolong the correction, 2 red-black sweeps.
   - On the last cycle, project: u −= (dt/ρ) coef · ∇p into the arena's velocity channel, with p still resident.
   - This replaces root project, transfer to simulation, the band copy and band project.
4. **pCoarse:** 16h and 32h levels, one workgroup per coarse brick group. Small enough that latency does not matter (≈0.01 ms).
5. **pMeasure:** fused into pDown's residual (a workgroup max plus one atomicMax per brick). It needs no pass of its own.

**Projection.**
- About 0.40 ms for 2 cycles, 0.50 ms for 3, against the 0.26 ms bound: 1.5–1.9×.
- About 20 dispatches, down from 155.

**Limits to watch.**
- Workgroup-memory occupancy: a brick at 4 KB plus rows in registers is fine.
- The convergence factor of the Schwarz-type smoother.
- The p ≥ 0 projection on inside rows, which MG transfers must not break. Today's root already runs projected Jacobi with the same constraint.

### 1.6 Measured: MG de-risk at frame 94 (2026-10-20)

The minimum-work bound in §1.4 assumed **2–3 cycles**. The measurement below shows that this projected, nonlinear system needs **4–6**. The algorithmic cycle count is the real floor, so the bound is restated at 4 cycles.

**Capture** (the pour at 1/60 s, frame 94; scratchpad `MG/capture.mts`):
- 2.5–2.7k band slots, 92–97k liquid rows, 636 fine 8³ bricks (4.2 of 8 tiles occupied a brick), h = 2.2 cm.
- Today's band runs its cap of 4 V(3,3) cycles on every frame. The residual history in s⁻¹ is 30.6k → 1862 → 61 → 3.7 → 0.79 → 0.36. Another run ended at 0.69.
- **Today never reaches the 0.05 s⁻¹ "returns" threshold on the pour.** The tolerance word is 1e-4.
- A CPU twin of the band operator reproduces the GPU's start and final residuals to four digits. It includes the nonlinear wall halos and the p ≥ 0 bound.

**Cycles to 0.05 s⁻¹** on the captured system (CPU bench `MG/mg2.mjs`; levels h … 32h; trilinear prolongation). The two figures are from the 4h start and from a warm start with frame 93's p.

| Smoother and coarse operator | Cycles |
|---|---|
| Exact red-black GS V(2,2), rediscretised (aggregation × 0.5) | 5 / 5 |
| Exact GS V(2,2), Galerkin (27-point coarse) | 4 / 4 |
| Brick-resident halo 1, two sweeps a load (as proposed in §1.4) | 8 / 8 |
| Brick halo 1, partitions alternating with a 4-cell shift | 6 / 6 |
| Shifted halo 1, Galerkin, V(3,3) | – / 4 |
| Today's aggregation (constant prolongation, × 0.6), under GS | 7 |

Notes on the table:
- Block-Jacobi at the brick faces costs about 3 cycles. More sweeps per load make it worse.
- Shifting alternate partitions by 4 cells recovers most of that loss at no extra staging.
- **MG-preconditioned CG does not help.** With the p ≥ 0 active set (≈865 rows at the bound) and the wall clamps frozen, a V(2,2) preconditioner reduces the residual about 3.7× an iteration. A plain projected V-cycle reduces it 7–10×. CG also plateaus whenever the active set moves.

**Per-cycle cost** (Dawn microbench `MG/mgbench2.mts`; real brick topology at 603/157/42/12/2 bricks; 27-neighbour staging; rows in registers):
- Fine pDown 67 µs, pUp 64 µs.
- Coarser levels: 8–25 µs a dispatch, latency-bound.
- One V(2,2) cycle: **0.29 ms**. At twice the bricks (the 6k-tile case) it is 0.44 ms.
- Knock-outs on the fine pDown: a floor of 23 µs (barriers, restrict, store), staging 20 µs, row loads 30 µs, sweeps 17 µs.

**Restated bound.** At 4 cycles (6004 slots, as in §1.4):

| Piece | Bound |
|---|---|
| Rows | 0.05 ms |
| 4 × (52 µs fine + 8 µs coarse + 18 µs passes) | 0.31 ms |
| Projection and measure | 0.05 ms |
| **Pressure** | **≈0.41 ms** |

**Design planned** (before implementation):
- Smoother: brick halo 1 with shifted alternate partitions, V(3,3) with three sweeps a load.
- Coarse operators: Galerkin at ℓ1 and below. Prolongation: trilinear. Warm start from the persistent p.
- Default: early exit at 0.05 s⁻¹, with a cap of 4 cycles (method param).
- The coarse tail is fused: ℓ3 and below run as one resident kernel; ℓ2 is one dispatch a direction.
- Target: ≤ 0.15 ms a cycle, i.e. band pressure ≈ 0.6 ms for 4 cycles, 1.5× of the restated bound.

### 1.7 Built: the h band on the brick arena (UniformPressureArena)

`lib/methods/uniform/uniform-pressure-arena.ts` replaces `UniformPressureBand`. It owns a `UniformBrickArena` (one `fine` list, ℓ1 built from it) and solves the h band's liquid rows in 8³ bricks.

**Design as built** (changes from the plan in bold):
- **Smoother: a halo-2 box per brick**, three red-black sweeps a load, staged in workgroup memory. Halo 2 removes the brick-face Jacobi loss without the shifted partitions.
- **Restriction: piecewise constant** (8 children sum to the parent); prolongation trilinear.
- **Coarse operator: rediscretised, aggregation × 0.5.** Galerkin is rejected: with trilinear prolongation its coarse stencil is 125-point, and staging and registers for it cost more than the cycle it saves.
- Warm start from the persistent p (warm := base + x after projection); the 4h trilinear carry seeds bricks without one.
- **Coarse tail**: ℓ2 is one dispatch a direction, from zero. ℓ3 and ℓ4 are page-local sweeps from zero. ℓ5 (≤ 496 pages) is staged in shared memory and runs 40 red-black sweeps in one 256-lane workgroup. A serial ℓ3–ℓ5 tail cost 5.4 ms; the split costs 0.38 ms in all.
- **Up kernels ping-pong** between two iterate channels: one channel read and written in place double-added the correction at brick seams.
- Stop rule: the residual history word of the cycle (exact pairwise residual at each restriction). Encoded cycles = min(cap, last wide count + 1).
- Default stop 0.05 s⁻¹ ("returns"), cap 4 cycles, `bandCycles` (1..8, update:"solver"). "strict" keeps the 1e-4 tolerance.
- Overflow is fatal: the band's tile capacity (mapped[23] = 1) and the ℓ5 page bound (mapped[23] = 2).

**Memory**: the letters scene with solids allocates 890 MB to the arena (194 blocks, 12416 bricks; fine capacity is every brick). The frame totals 3.03 GB.

**Letters A/B** (1/60 s, 120 steps, old band vs arena):

| | Old band | Arena |
|---|---|---|
| Volume drift at f120 | −1.31% | −1.32% |
| Liquid cells at f120 | 1284241 | 1284224 |
| yMean / yMax at f120 | 20.12 / 128 | 20.13 / 133 |
| Occupancy L1 difference | | 0.4% (f60), 1.7% (f120) |
| Final residual | cap 4, does not reach 0.05 | 0.02–0.10 s⁻¹ at 3–4 cycles |

**Pour** (1/60 s; GPU stage means from the same frames):

| | Old band | Arena |
|---|---|---|
| Band solve | 2.91 ms | 6.36 ms |
| Band list/marks and rows | 1.12 ms | 1.51 ms |
| Band projection | 0.29 ms | (in the solve) |
| **Band total** | **4.32 ms** | **7.87 ms** |
| Pressure per k fine tiles | 1.34 | 1.85 |

Per kernel (arena, ms a frame): pUp0 1.48, pRows 1.43, pSmooth0 1.23, pProject 0.75, pInit 0.49, pDownA 0.41, pRestrict0 0.38, pTail 0.38, pRebase 0.34, pDownB 0.31, pUpA 0.31, pUpB 0.15, pUpT 0.13.

**The target is not met.** A cycle costs about 1 ms, not ≤ 0.15 ms. The arena converges further than the old band, but it is 3.5 ms slower on the pour. Level 0 dominates: pUp0 and pSmooth0 stage a halo-2 box for each brick, and pRows, pInit and pProject run per brick, not per tile.

Next:
- P2: retire the root solve, its transfers and project, the band velocity copy, the coarse bakes and the triplicated authority classification (≈ 3–4 ms on the pour).
- Fuse pUp0 with the next cycle's pSmooth0.
- Stage less of the halo.
- Run prepare, init and project per tile in 64-lane groups.
- Bound the fine capacity by the band, not by every brick.

## 2. RFE: remap, geometric fill, body forces, extension

### 2.1 Per pass

| Pass | ms | Disp. | Computes | Irreducible job | Bound ms | × | Limiter today |
|---|---|---|---|---|---|---|---|
| geometryRegional (1.24 after b7c3f514) | 1.676 | 2 | Planar-exact fill per owner from 8 corner phi (corner tetrahedra elsewhere); surface targets | 441k owners × ≈20 B ≈ 9 MB, plus ≈300 flops per owner (0.13 GFLOP) | 0.04 | 42 (31 after) | Per job: corner umVertexSlot walks, umSurfaceTarget plane fits inline (a fat kernel), 2 barriers. The old grid ran about 65 jobs serially per workgroup (fixed in b7c3f514). |
| geometryChanged, mirrorRegional, mirror, compactChanged, normalTags | 0.309 | 5 | Changed-tile fill; mirror into pressure ownership | Changed tiles only | 0.01 | 30 | The same values stored twice (simulation and all-4h pressure). |
| remapCells | 1.292 | 1 | Cells of tiles within one tile of a width change: volume, phi and authoritative vertices; donor split | 1143 jobs × 64 cells × ≈6 channels ≈ 2 MB | 0.01 | 130 | **Latency per job.** A chain of oldumOwnerAt lookups through the old page table, umSurfaceTarget donor splits, 125-vertex authority checks, serial planeCorner sums, many barriers. |
| remapFaces (twinned 0.992, free 0.075) | 1.067 | 4 | Faces of the same tiles: beside, slope, sample, released | 1143 × 240 faces × 8 B ≈ 2 MB | 0.01 | 100 | 4+ barriers per job; remapBeside/remapSlope neighbour walks through the old ownership. |
| remap publish, markListed | 0.159 | 3 | Copy back from scratch; worklist | 1143 tiles | 0.01 | 15 | Launch floor and copy-back. |
| Body forces | 1.133 | 4 | Gravity, inflow, viscosity (vector), capillarity; COLD retain | 441k owner texels read-modify-write (32 B) plus occupancy: 16 MB | 0.045 | 25 | One lane per owner over umAllOwner. Per lane: umFace neighbour walk (gravity test), neighbour umOccupancy, umCoarseMass, umForceRetain. Viscosity terms reach across tiles with owner lookups. 4 launches by tier. |
| Extension, main | 2.099 | 26 | Godunov extension on canonical patches (seed, sweeps, seam sweeps, publish) plus the 4h nearest-source pyramid (restrict, sweep blocks, prolong, replay) | ≈190k air h cells × 3 faces = 576k faces × 2–3 sweeps × 16 B ≈ 25 MB | 0.07 | 30 | Slot indirection per patch (umSlot by fine-tile rank); seams in separate passes (16 subfaces); 26 launches, a pyramid level at a time. |
| Extension, slot snapshot | 0.974 | 12 | The same extension of the snapshot field | The same faces, fusable as a second channel | 0.02 | 49 | A second full run with its own pyramid. |
| **RFE** | **8.71** | **58** | | | **0.6** (today's counts and launches) | | |

**What makes up the RFE bound.**

At today's structure, 0.6 ms:
- body forces 0.045;
- geometry 0.05;
- remap 0.03;
- extension 0.09 (two runs);
- the 58 dispatches' floor, 0.13;
- with headroom for the seams' separate passes.

At minimum work, 0.18 ms (§2.4).

### 2.2 Redundancies

- **Extension runs twice:** main and snapshot, 38 dispatches between them, each with a 4h pyramid. One brick pass can extend both fields as two channels.
- **Geometry is computed for two ownerships:** the regional fill, then mirrors into the all-4h pressure ownership, plus an h tile's first lane recomputing the 4h rule.
- **Remap and forces redo work other passes have done.**
  - Remap re-derives geometry (umSurfaceTarget) and surface authority (125-vertex checks) for each donor.
  - Forces re-derive occupancy and neighbours that the authority and geometry passes resolved.
- **Remap works through the old ownership's page table.** In the arena, a width change becomes a refinement event at gather.

### 2.3 Why geometry and remap are the worst ratios

Both are change-driven in principle, but latency-bound in practice.
- A job is a serial chain of dependent page walks and barriers on 64 lanes.
- 1143 remap jobs cost 2.4 ms, about 2 µs of GPU time each with the whole GPU sharing them.
- Each job's real work is about 4 KB of reads and writes.

### 2.4 Minimum work and the arena kernels

The minimum-work bound is 0.18 ms:

| Piece | Job | Bound |
|---|---|---|
| Forces | Read-modify-write of the velocity channel | 0.045 ms |
| Geometry | 8 corners per cell from staged φ; surface bricks only, 0/1 elsewhere from class bits | 0.04 ms |
| Remap | Changed bricks only, restriction/prolongation in workgroup memory | 0.02 ms |
| Extension | Both fields as channels, resident sweeps over a 3–4 cell halo | 0.05 ms |
| Passes | About 10 | 0.025 ms |

**Arena kernels**, each one workgroup per brick:

1. **rGeometry:**
   - Surface bricks (class bit: φ crosses zero in the brick or its halo) compute the fill and apertures from staged corner φ.
   - Other bricks write 0 or 1 from class bits.
   - Bricks whose φ did not change are skipped.
   - One store serves both ownerships: the pressure rows read the same channel.
   - Projection: 0.06 ms (65%).
2. **rRemap:**
   - A width change is a brick refinement event: a coarse brick region gains or loses fine bricks.
   - The gather prolongs velocity, φ and volume into new fine bricks. The scatter restricts V-weighted into the coarse brick, which is today's conservative rule.
   - Only changed bricks run: 1143 tiles is roughly 150–300 bricks.
   - Projection: 0.04 ms (50%: few bricks, so latency shows).
3. **rForces:**
   - Fused into the kernel that writes momentum's velocity channel. Gravity is an add; viscosity is a Laplacian from the staged faces; capillarity comes from staged curvature.
   - The COLD retain is a class bit.
   - Projection: 0.03–0.05 ms. It is near zero if fused into momentum's store.
4. **rExtend:**
   - Air-band bricks stage known masks, values and a 3–4 cell halo.
   - They run fast-sweep or closest-source iterations in workgroup memory, enough to cover the halo, for the main and snapshot fields as channels.
   - Far faces come from coarse bricks' extended values: the 4h pyramid becomes a coarse-brick pass.
   - Workgroup memory sets the halo: 12³ × 3 components × 4 B is 20 KB, which caps occupancy at about 1–2 bricks per core. That is the limiter; a component per pass is the fallback.
   - Projection: 0.12 ms (40–60%).

**Total projection:** 0.30–0.35 ms against the 0.18 ms bound (1.7–1.9×), in about 12 dispatches instead of 58.

## 3. Levers, in order of yield

| Lever | Pressure ms | RFE ms | Notes |
|---|---|---|---|
| One MG over h and 4h bricks; delete the root solve, the transfers and the second projection | −3.5 | | Authority, root, transfers, band copy |
| Brick-resident smoother, 2–3 cycles instead of 4 + 1 | −1.8 | | bandSweep, middle, coarse, measures |
| Rows from staged channels (no page walks) | −1.0 | | bandPrep, init, rhs |
| Change-driven brick remap at gather/scatter | | −2.4 | remapCells, remapFaces, publish |
| Geometry from staged φ, one store | | −1.9 | Regional, changed, mirrors |
| Extension as one resident pass over two channels | | −2.9 | Main, snapshot, pyramids |
| Forces fused into the velocity channel | | −1.1 | |

## 4. Measurements this analysis needs before implementation

These are cheap, and run at the fixed frame-94 state of the umbrella plan, §6:
1. **The band's header words:** slot count, completed cycles and the residual history. These confirm the band size and whether 4 cycles are needed under today's aggregation.
2. **A brick-resident red-black microbench:** 2 sweeps per load on 1.5k 8³ bricks with a 1-cell halo. It confirms the 26 µs fine-visit bound and the halo hit rate.
3. **The fine-brick count for the pour's 6004 h tiles** (8³ bricks holding them). This sets the arena's capacity and halo overhead.
