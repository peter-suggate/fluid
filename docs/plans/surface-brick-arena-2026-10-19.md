# Surface brick arena: the liquid-surface stage on an M1-shaped data layer

2026-10-19. Target: `nbflip-figure-7-pour` at 1/60 s, under 17 ms a frame on the M1 Max.
Today the frame is 59.0 ms, with 24.5 ms of it in the liquid-surface family (54 passes).

## 1. The job

Each step, the stage turns (φⁿ, u, particles) into φⁿ⁺¹ on the narrow band. φ is stored at h on fine tiles and at 4h corners elsewhere. The steps are:

1. **Advect.** φ*(v) = φⁿ(X(v)), where X is the RK3 backtrace of vertex v through u, sampled cubic (Catmull-Rom). Far and cold vertices take early exits: identity, clamp or copy.
2. **Particle surface.** φ_p(v) = min over particles within 2h of (|v − p| − r), from a splat.
3. **Couple.** φ = mix(bulk(φ*), particle(φ_p), θ), with θ the erasure heat (Ferstl Eq. 4).
4. **Crossings.** Find the cells where φ changes sign, and the closest points on the trilinear zero set within them.
5. **Redistance.** Every non-interface vertex within reach (3 tiles) gets signed distance to the zero set. Interface vertices keep their value. Outside reach, |φ| ≥ 6h.
6. **Publish.** Write the band masks (crossing mask, reach, nearest crossing) for transfer, update, the pressure band and the census.

Production runs this as about 54 passes, each walking the page table per tap (`umVertexSlot` with up to 7 skirt pages). The resulting costs:

- Occupancy is 11.8–18.8% and throughput 5–8 GB/s (near-settled-pool-handoff-2026-10-18.md:566-570). The stage is latency-bound on addressing, not on math or bandwidth.
- The work is duplicated:
  - two full crossing searches per frame;
  - depthGeometry re-reading corners that depthSeeds just held;
  - buildDistance followed by redistanceFine copying its bank;
  - 4–5 phiResolves, one after every in-stage writer;
  - coupleFine re-sampling φ* through the page table;
  - coarse vertices re-traced by the fat deferred chain.

## 2. What this GPU does well (measured)

The standalone bench (`scratchpad/sbench/surfacebench.mts`, 1M particles, 2249 bricks, 371k band nodes) runs the whole job, particles included, in **3.6 ms**:

| Kernel | ms |
|---|---|
| D: phi advect, cubic + couple | 0.78 |
| C: splat | 0.65 |
| E1: crossings | 0.07 |
| E2: redistance | 1.50 |
| A: particle g2p + trace | 0.56 |

It does this with the following layout:

- dense 8³-node bricks, with node = brick·512 + local;
- one brick-table resolve per workgroup;
- windows in workgroup memory;
- per-brick crossing lists of closest points;
- redistance as a broadcast nearest loop over the 27 neighbour bricks' culled lists;
- shared atomics with one flush for the splat.

Every heavy function has one call site.

The production lessons point the same way:

- **Lean kernels beat fat ones.** Fat merged kernels cost 4–10× per entry (fat-kernels-cost-10x).
- **Addressing is the cost.** Paging tripled the INT32 instructions in redistance. Partial fixes regressed while the paged addressing stayed in the inner loop: the bounded shared plan (+1.5 ms) and workgroup caches.

## 2a. Speed of light

These are the measured M1 Max ceilings (`scratchpad/ceil/ceilings.mts`):

| What | Measured |
|---|---|
| Streaming | 360–376 GB/s |
| Random 4 B gathers, within 4 KB windows | 34 G/s |
| Random 4 B gathers, over 4 MB | 15.5 G/s |
| Random 4 B gathers, over 32 MB | 6.9 G/s |
| Random 4 B gathers, DRAM | 0.75 G/s |
| Workgroup memory reads | 719 G/s |
| Device atomics | 5.7 G/s |
| Workgroup atomics + one flush | 218 G/s |
| Per-pass and per-dispatch floor | ~2.3 µs each |

The job's irreducible work, at about 490k fine-band vertices and about 2k bricks:

- **Taps.** About 170 per vertex (advect: 3 velocity samples × 24 taps, 64 cubic, 10 foot; crossings: 8; redistance: a broadcast loop that is FMA-bound). That is about 83M taps. Served from workgroup windows, they take about **0.12 ms**. Served as device gathers in the 4–32 MB working set, they take **5–12 ms**, which is where production sits.
- **Window fills.** Brick plus halo, coalesced, is about 60 MB a frame: about **0.17 ms** at streaming rate.
- **Redistance math.** About 490k × ~200 candidates × 3 FMA ≈ 0.3 GFLOP: about **0.1 ms**.
- **Passes.** About 10 kernels: about 0.03 ms of floor.

The speed of light for the φ job is therefore **about 0.5 ms**. Today it costs about 19 ms (advect, crossings, couple, redistance, resolve, work control), roughly 40× off. The bench reached 3.0 ms, about 6× off.

The design target is the bound: each kernel is reported as a percentage of it. Anything more than about 2× off must name its limiter: occupancy, latency, atomics, divergence or off-arena taps.

## 3. The data shape: a per-frame band arena

The arena is a dense working copy of the fine band, gathered once per step. Every in-stage kernel reads and writes it. It is scattered back to the paged h store once, at the end of the stage. Consumers outside the stage keep reading paged φ unchanged.

- **Brick.**
  - 8³ cells of a page: a page is 4³ bricks, a brick is 2³ tiles.
  - Vertex-owned, using the h store's convention: a brick owns its cells' high corners, h-store local 1+8k … 8+8k.
  - There is no skirt. Neighbours are reached through the brick record.
- **Brick table.**
  - One u32 per (block, brick local 0..63): an arena slot, or NONE.
  - Rebuilt each step from the band (fine tiles plus the 1-tile ring, minus COLD regions).
  - The brick list is compact with a fixed capacity, sized upfront from the fine-tile capacity. Exceeding it is a fail-fast fatal. There is no growth.
- **Brick record** (computed once per step, reused by every kernel):
  - block and local;
  - 27 neighbour slots (self included);
  - class bits: any h-owned vertex, any crossing (after X), all cold, solid-clear radius (from the tile context's `utcSolidClearance`), any coarse corner or hanging vertex.
- **Per-vertex payload** (SoA, index slot·512 + l):
  - φ bank A and φ bank B (f32);
  - vertex class, 2 bits: h-owned, hanging (derived from 4h corners), 4h corner, closed.
  - splat accumulator (u32, the d² key);
  - distance (f32).
- **Per-brick crossing list.**
  - Closest points, packed 16-bit fixed point (the bench's codec), CAP 384.
  - Overflow is fatal.
  - It doubles as the published nearest-crossing data that replaces banks A/B for transfer, update and census.
- **Velocity.** For the trace, the brick holds the three staggered face components that fall in it. This keeps umFastVelocity's interpolation, not a node average, so physics is unchanged. They are gathered once per step, after the extension. The same window serves particle advect, the census swept boxes and the work-control velocity bounds; adopting it there is a later phase.

**Hanging vertices are resolved inside the arena.** After any arena write, a brick-local pass re-derives the hanging vertices from the 4h corners it holds. This replaces the in-stage phiResolves; one paged phiResolve runs after the scatter.

## 4. Kernels

Every kernel runs one workgroup per brick, with 256 lanes and 2 vertices per lane. Each strides the brick list on a fixed direct grid.

| Kernel | Replaces | Does |
|---|---|---|
| **G** gather | — | paged φ → bank A (h-owned via udLoad; hanging via umVertexValue; 4h corners); vertex class; brick class |
| **A** advect | advectFine, Trace, Finish, Cubic, Walls, deferred (fine part), phiResolve(scratch) | bank A → bank B. Same early exits, foot, RK3, cubic and wall release. Taps come from the arena (neighbour slots), with the paged sampler only off-arena |
| **S** splat | surfaceClear, surfaceSplatOccupied | particles of the brick's tiles and halo → shared atomicMin window → one flush to the accumulator. The clear is part of the gather, so no whole-block clear |
| **X** crossings | depthTiles, depthSeeds, depthGeometry, depthNearest*, search #1 and #2 | per brick, crossing cells of a bank → closest points (nbSurfaceDistance's math: plane shortcut, else 12 edge crossings plus 6 Newton) → list; mask and reach to the band planes |
| **C** couple | coupleFine (fine part) | bank B, accumulator and X(bank B) nearest → coupled φ in bank A |
| **R** redistance | buildDistance, redistanceFine, 2 phiResolves | X(bank A) lists of the 27 neighbours, culled → nearest per non-interface vertex → bank A (the bench's E2 shape) |
| **W** scatter | nbStoreReconstructed / in-place stores | bank A → paged h store, writing only changed values and bumping region revisions as today; then one phiResolve |

Coarse 4h corners keep their current path in phase 1. In phase 4 they get a 4h-native kernel over pages, because the base block (9³ per page) is already dense.

## 5. Expected cost

The estimates scale from the bench, at about 490k fine vertices.

| Phase | Replaces (today) | Expected | Win |
|---|---|---|---|
| 1: G + X + R + W, the redistance chain | search #2 (~1.1), depthGeometry 0.26, buildDistance 2.43, redistanceFine 0.69, 2 resolves (~0.5) ≈ 5.0 | ~1.5 | ~3.5 |
| 2: A, fine advect | 5.6 + deferred share | ~1.5 | ~4–5 |
| 3: S + C, plus search #1 | splat 0.8, clear 0.3, coupleFine 1.6, search #1 ~1.1 | ~1.2 | ~2.6 |
| 4: 4h-native coarse advect and couple | deferred coarse ~1.9, couple 0.4 | ~0.6 | ~1.7 |

The surface family would drop from 24.5 ms to about 12 ms, with extension (3.4), authority (1.5) and work control (1.3) untouched. The velocity window then feeds particle advect and the census in a later phase.

## 6. Invariants the implementation must keep

- COLD regions write nothing. Region revisions bump only when bits change; retention stamps stay valid.
- Interface vertices are preserved by redistance. Outside reach, |φ| ≥ 6h. A vertex with no crossing in range gets today's fallback value.
- Exactness:
  - Redistance must be at least as exact as today: 27 bricks cover 8h, against today's radius-5 nearest.
  - Beyond 8h, the vertex takes the same clamp as today's no-crossing case.
- No indirect dispatch, no per-frame JS allocation, no compile during simulation, fail fast on capacity.
- One call site per heavy WGSL function (nbSurfaceDistance, the cubic, the trace).
- Physics unchanged by default. Anything that changes results goes behind an NB-FLIP method param.
