# Within 2× of speed of light: the three slowest stages on one brick arena

2026-10-20. Target: `nbflip-figure-7-pour` at 1/60 s, under 17 ms a frame on the M1 Max.
Project lead's direction: get the three slowest stages within 2× of their speed-of-light bounds, through one shared data structure, and implement it.

## 1. Where the frame is

Pour run part2h (HEAD at 99d4a0ef plus particle work; frames 94–100):
- 6004 fine tiles, i.e. 384k h cells;
- 57,484 coarse tiles;
- 695k particles;
- 331 passes and 404 dispatches per frame.

| Stage | ms (part2h) | ms (vg1, 2026-10-19) | Bound | 2× target | Gap today |
|---|---|---|---|---|---|
| Surface (advect, splat, couple, crossings, redistance, store, resolve, work control) | 18.2 | 20.2 | ~0.6 | ≤1.2 | ~30× |
| Remap / geometric fill / body forces / extension (RFE) | 8.7 | 10.2 | ~0.3 | ≤0.6 | ~29× |
| Pressure (rows, band solve, V-cycle, authority, rhs, project) | 8.2 | 9.3 | ~0.6 | ≤1.2 | ~14× |
| Particles | 6.5 | 9.7 | 1.15 (measured design) | ≤2.3 | 5.6× |
| Momentum | 1.1 | 3.4 | 0.39 | ≤0.8 | 2.8× |
| Other + work control | 6.1 | 6.3 | — | — | — |
| **Sum of pass time** | **48.8** | **59.0** | | | |

Notes:
- **Idle time.** The frame also carries about 7.2 ms of GPU idle between passes, outside the pass sum. The 331 passes' own floor is about 0.76 ms. The idle is its own lever: copies, clears and dependency bubbles.
- **Surface bound restated (2026-10-20).** The ~0.6 ms bound assumed taps served from staged workgroup windows. The h band is too thin for that: about 103 of every 125 window tiles are 4h, and the staged sAdvect lost (9.64 ms against 6.05 ms for advect). Advect taps run at the gather rate (about 22 G/s, 2.18 ns per 48-tap trace), which puts advect's floor alone at about 0.9 ms. The surface design (`brick-arena-core-2026-10-20.md` §5) projects 15.0 → 5.4 ms (0.90 per k fine tiles) through a sign certificate, persistent fine φ and a workgroup-atomic splat. That is not window staging. The coarse level (1.22 ms) has no named win yet.
- **Counts vary between runs.** Pour state differs a lot from run to run (580k–1.1M particles, 5.6k–7.7k fine tiles). Every claim below is per unit of work, and wins are timed against one captured state (§6).

If the three stages reach their 2× targets (≤3 ms together, against 35 ms today) and particles reach their measured design (2.4 ms), the pass sum is about 48.8 − 35.1 + 3.0 − 4.1 ≈ **12.6 ms**. That leaves room for idle and the untouched rest inside 17 ms.

### Late frames: the churn case, now the focus (2026-10-20)

Peter: "we need to be focusing our efforts on the churn case (later frames). focus on the stages that bring the most time there".

Profile `scratchpad/P/late1.json`: clean HEAD 843802bd, pour frames 200–212, 1.2M particles, 8981 fine tiles. GPU pass time is 54.9 ms, plus 7.1 ms of idle.

| Stage | ms | Per unit | Largest passes |
|---|---|---|---|
| Surface | 16.8 | 1.87 / kT | arenaDistance 1.84, coupleFine 1.83, advectFine 1.63, deferred 1.44, advectFinish 1.22, arenaStore 1.16, phi resolve 0.98 |
| Pressure | 10.3 | 1.14 / kT | pRelax 1.89, pRows 1.74, authority 1.54, project 0.74 |
| Particles | 10.2 | 8.42 / Mp | advectParticles 3.33, transferDeposit 1.51, sort 0.70, refine 0.67, seed 0.63 |
| RFE | 10.0 | 1.12 / kT | remap 2.89, extension 2.42 + snapshot 1.06, geometric fill 1.75, body forces 1.46 |
| Other | 6.4 | | ownership census 2.19, local speed certificate 1.25 |
| Momentum | 1.3 | | |

Particle churn defines these frames: about 7% of particles are replaced per frame, and about half the seeds spill. Seeding fills down to depth −4 and the move retires at −4.0, with no hysteresis between the two. RFE now has its own owner. Every stage measures at f200–212.

## 2. Bounds: the irreducible job at the pour's counts

The ceilings are measured (`scratchpad/ceil/ceilings.json`):

| What | Measured |
|---|---|
| Streaming | 360–376 GB/s |
| Random 4 B gathers, within 4 KB windows | 34 G/s |
| Random 4 B gathers, over 4 MB | 15.5 G/s |
| Random 4 B gathers, over 32 MB | 6.9 G/s |
| Workgroup memory reads | 719 G/s |
| Device atomics | 5.7 G/s |
| Workgroup atomics + flush | 218 G/s |
| Trilinear texture, rgba16f / rgba32f | 28.4 / 13.5 G samples/s |
| Per pass | 2.3 µs |

### Surface, ~0.6 ms
- **φ job:** about 490k band vertices, about 83M taps per step served from workgroup windows. That is 0.12 ms of taps, 0.17 ms of window fills, 0.1 ms of redistance math and 0.03 ms of passes: about 0.5 ms.
- **Work control** at 16h is a reduction over about 1k regions: below 0.05 ms.
- The standalone bench ran the φ job plus particles in 3.6 ms on dense bricks.

### Pressure, ~0.6 ms
About 440k unknowns: 384k h cells plus the 4h cells.
- **One fine smoother sweep:** about 36 B per cell (p, rhs, weights, write) is about 14 MB, or 0.04 ms streaming.
- **A V(2,2) cycle:** about 7 fine sweep-equivalents plus about 15% for the coarse levels plus about 25 passes is about 0.33 ms.
- **Two cycles plus rhs and project:** about 0.7 ms.
- Workgroup-resident smoothing (two sweeps per brick load with a 2-cell halo) roughly halves the fine-level traffic, giving a bound of about 0.5–0.7 ms.
- The iteration count is the main unknown. Today's band solve and V-cycle structure and residual target are being measured (`pressure-rfe-speed-of-light-2026-10-20.md`).

### RFE, ~0.3 ms
About 1.2M live faces (384k h cells × 3, plus the 4h faces).
- **Body forces:** one read-modify-write per face, about 10 MB, 0.03 ms.
- **Extension:** 2–3 sweeps over the air band faces, about 0.1 ms.
- **Remap:** proportional to tiles whose ownership changed (1143 remap jobs), not to all tiles. That is about 0.05 ms.
- **Geometric fill:** the solid is static. Only changed tiles need refilling (geometryChanged is 0.18 today), so it is near zero.

## 3. Why each stage is 14–30× off, and what closes each factor

The diagnosis is the same in all three stages. Every inner-loop tap is a device gather through the page table: `umVertexSlot` / umOwner / umSampleTile, plus hanging and seam resolution. Each kernel also re-derives classification through a chain of predicates (writer, cold, interface, buried, frozen, retained, drop). Measured consequences:
- occupancy of 12–19%;
- throughput of 5–8 GB/s against 360;
- per-vertex dependent-gather chains of about 30 loads (arenaStore);
- device-atomic traffic of about 5M per frame in redistance alone;
- lists and classifications rebuilt per stage, about 330 passes in all.

Each factor has a lever that closes it:

| Factor | Typical size | Lever |
|---|---|---|
| Page-table addressing per tap | 5–10× | Taps from workgroup-staged brick windows: resolve once per brick, not per tap |
| Predicate chains per vertex or face | 2–4× | Classes computed once per brick at gather, as class bits in the record; kernels branch per brick, not per element |
| Device atomics (reach, bins, splat) | 2–5× on affected kernels | Gathers instead of scatters; workgroup atomics plus one flush |
| Redundant passes and re-gathers | 1.5–2× | One gather per step; fused kernels where data is already resident |
| Solver work (pressure) | 2–4× | MG with workgroup-resident brick smoothers; the iteration count set by the residual target, not by the band split |
| Work on unchanged data (geometry, remap) | up to 10× on those passes | Change-driven lists |

## 4. The shared data structure: the brick arena

One structure serves surface, pressure, RFE, momentum and particles. Pieces already in the tree:
- `uniform-brick-table.ts`: one brick table, class bits {arenaCore, arena, particlesCore, particles, solidClear, blend, rigid}, 27-neighbour records;
- `uniform-surface-brick-arena.wgsl.ts`: windows, gather and store;
- the particle slabs, keyed by brick.

What the arena is:
- **Bricks.** Fine bricks are 8³ h cells; a page holds 4³ of them. Coarse bricks are 8³ 4h cells, one page. The bricks form one compact slot list with fixed capacity, allocated upfront; exceeding it is fatal.
- **Records.** Each brick record holds 27 neighbour slots (fine to fine, fine to coarse at seams) and class bits computed once per step.
- **Channels.** SoA per slot, declared by stages: φ banks, u faces ×3 (MAC, brick-owned high faces), p, rhs, face weights and solid fractions, class, and scratch. A stage binds only the channels it reads.
- **Halo staging.** One WGSL helper at one call site stages a brick and a 1–3 cell halo into workgroup memory from the neighbour slots. Hanging and seam values are resolved here, by the gather's formula. Every kernel is one workgroup per brick on a direct grid-stride, with no indirect dispatch and no page table in inner loops.
- **MG hierarchy.** Fine bricks restrict to 2h bricks and then to 4h bricks, the coarse bricks being the same slots the 4h region already uses. Pressure's levels are brick lists, not page walks.
- **Lifetime: step-authoritative or persistent.** This decision is being quantified in `brick-arena-core-2026-10-20.md`.
  - *Step-authoritative:* gather once at step start, all stages work in the arena, scatter once at the end. Today's 2.06 ms arenaStore shows the copy must be a plain changed-value stream (about 0.1 ms at streaming rate), not a predicate chain.
  - *Persistent:* bricks are the storage for live regions across steps, and the paged store holds only cold regions. That removes the copy entirely.

## 5. Route per stage (each section filled by its analysis)

- **Surface:** `brick-arena-core-2026-10-20.md` (surface agent).
- **Pressure and RFE:** `pressure-rfe-speed-of-light-2026-10-20.md` (momentum agent).
- **Particles:** `particle-brick-slabs-2026-10-19.md`. FUSED + DEPB is in progress, measured at 2.4 ms in the bench.

Each section must give, per kernel: the irreducible work at pour counts, the bound, today's ms, the arena kernel's projected ms, and its limiter. Anything over 2× of its bound must name what stops it.

## 6. Measurement

- **Fixed state.** Capture one pour state (frame 94) and time every change against it, at the same particle and tile counts.
- **Per unit.** Report µs per thousand fine tiles and ms per million particles alongside frame totals.
- **Physics.** Run the letters A/B (`scratchpad/L/lp.mts`; φ read through the live GPU page table since d32345b5) before every commit. Gates: particle counts, volume drift and occupancy within repeat noise (about 0.2% at f60, about 2% at f120).
- **Reassess priors outside 2×.** When a measurement lands more than 2× off its bound, name the priors behind both the bound and the design (tap rates, element counts, which work is needed, data layout, algorithm), find the one that broke, and try a different approach. Never accept the gap as the per-element rate of today's design, and never retune the same approach.
- **GPU discipline.** All GPU runs go under the lock: one run per arm, a repeat only to settle noise.

## 7. Order of work

1. **Analyses** (in progress, doc only): surface plus the arena core spec; pressure plus RFE.
2. **Arena core**, built unwired: channels, halo staging, gather and scatter, MG hook.
3. **Surface onto the arena**, phase by phase: redistance and store first (largest gap), then advect, splat, couple and work control.
4. **Pressure onto the arena**: MG brick smoother, rhs and project. Retire the band rows and authority passes that the arena makes redundant.
5. **RFE onto the arena**: change-driven geometry and remap, body forces fused into the arena's face channel, extension as brick sweeps.
6. **Particles** finish FUSED + DEPB on the same bricks; the velocity channel serves G2P.
7. **Idle**: attribute the 7.2 ms of between-pass idle and remove the bubbles.

Each switch-over is one short edit, followed by the letters A/B and the fixed-state pour, then a commit.
