# Uniform Geometric fig7-256: 3× assessment (domain-shaped work and paging)

Status: assessment, 2026-09-29, HEAD 5606a3f4 plus the uncommitted working tree. No code changed.

Target: `cm12-figure-7-256` (256³ = 16.8 M cells, T = 262,144 4³ tiles; a ball of about 1.6 % of the cells dropped into a dry, solid-free box). Goal: 3× frame time, with work proportional to the fluid. Regression indicator: `sparse-cm12-long-dam-break`.

## Baseline (Dawn/Metal, M1 Max, one run per arm)

The probe is `tools/.probe-f7-a1.ts` (the grid-counting copy is `tools/.probe-f7-grid.ts`). Default dynamic coarsening, steps 10–50 (fig-7) and 36–60 (long dam), each frame awaited.

| scene | untimed wall median | traced GPU | mean h tiles | dispatches/frame | workgroups/frame | relayouts |
|---|---:|---:|---:|---:|---:|---:|
| fig7-256 | **57.5 ms** (mean 63.0) | 56.9 ms | 12,899 (4.9 % of T) | 684 | 2.56 M | 50/50 |
| long dam | **12.3 ms** | 10.7 ms | 1,367 | 475 | 0.34 M | 60/60 |

fig7-256 is GPU-bound: wall ≈ GPU. 3× means about 19 ms.

Stage means for fig7-256 (ms): extension 5.9, momentum 5.0, transport 4.6 (+1.4 live set), band solve 4.5, redistance 4.3, advect 3.5, sharpening 3.0, global surface volume 2.7, remap 2.4 + 1.1 + 0.6, pressure V-cycles 2.0, frame plan 1.6, geometry fill 1.4, forces 1.2, census 1.1 + 0.55 + 0.37, and the rest under 1 ms each.

## Finding 1: container-shaped work is about 14–17 ms (25–30 %)

About 82 % of launched workgroups scale with the container. Most of them exit cheaply, so their share of time is smaller.

| class | examples (file) | measured ms | far-field result |
|---|---|---:|---|
| one workgroup per tile over all T (327,675 WG each) | `retirementEvidence` (uniform-mixed-surface.ts:465), `liveSeed` (uniform-mixed-transport.wgsl.ts:426), surface-band `classify` (uniform-pressure-surface-band.ts:29, 125 vertex loads per tile) | **3.9** | constant: no sign change, evidence FAR, V=0 means no donor |
| merged certified grids, 65,535 WG, which run a full RK2 trace per far-air 4h owner | advect, traceCells, redistance (identity copy), momentum (uniform-mixed-ownership.ts:225) | part of 7.3 | constant (phi plateau is a fixed point) or dead (far momentum is overwritten by the tail extension) |
| all-4h pressure over the container | rhs, project, transfers ×2, authority ×2, 17³ `mgSmoothColour`, residual/prolong/restrict/copy, 8 acceptance reduces | about 4.3 | constant: p=0, b=0 in air |
| T/64 bookkeeping | plan dilate/spread/certify, live gather/scatter, surface-volume grow/growMeasure (125 loads per unvisited 4h tile), evidenceDistance, census classifyCoarse/boundCube/decide, builder distance0-2 (16.7 M loads each), markChanged | about 3.3 | mostly constant; `decide` has no global-reach early-out |
| n/4 extension far field | publishFine `umFarValue` on every 4h owner, 64³ prolong, restrictBand, momentum cache 17³ | about 2.6 | **not constant**: nearest-source velocity, needed within the census and sampler reach |
| T-sized copies and clears | solid record 4.6 MB every advance in a solid-free scene, band slot map 2 MB, census work 9.5 MB, stage grids | <0.3 | trivial but pure container work |

Host work per census is also O(T): `uint8Key`, `fine.map`, the JSON region key, `tiles.every` in `adoptBuiltLayout`, and a 1 MB topology readback. It is not on the GPU critical path at this frame time.

**Removing all of this gives about 43 ms, or 1.3×.** That is necessary, but it is not the 3×.

## Finding 2: the other ~40 ms is h-list work, and it has two levers

1. **The h count.** At a mean of 12.9k h tiles, the liquid is roughly 4–5k tiles (the ball, then the floor sheet). The fig-9 analysis found the RK2 departure halo plus the 26-neighbour closure makes it about 2.2× the required count. The ball falls fast: travel ≈ 13 cells per step at 10 m/s, so `umCertificateRadius` ≈ 6 tiles. That inflates both the halo and the reach of every certificate.
2. **Cost per h tile.** The measured leads are:
   - **`momentumStep` 5.1 ms** is the *merged* launch. `momentumRegularStep` costs 0.07 ms, so almost no h tile passes the regular certificate. Any h tile within R ≈ 6 tiles of a 4h tile is disqualified. So most h faces take the general sampler with up to 32 substeps. The same mechanism probably inflates advect and traceCells. The first measurement is the merged launch split into h-general and 4h jobs.
   - **Band solve 5.1 ms**, over 218 dispatches (124k WG).
   - **Remap about 4.1 ms per frame**, because a relayout happens every frame. `remapCells`/`remapFaces`/`copy*` run on a fixed `min(1024,T)` grid (uniform-mixed-remap.ts:33). That is the small-cap shape that lost load balancing in the certified wrapper (fixed there by sizing to the bound).
   - Extension sweep 2.4, redistanceFine 1.8, advectFine 1.4, sharpening 3.0 (8 sweeps × 3), frame-plan `seed` 1.85 (2 dispatches).

## Finding 3: paging is a memory win; the time win comes from a residency certificate

Paging machinery in the tree:
- `uniform-page-generation.ts` holds the transactions and pools. It is serial, capped at 4096 pages, uses indirect dispatch and an AoS pool, and only a test uses it.
- `uniform-page-support.ts` (indirect) and `uniform-page-layout.ts` (host planner) are test- or tool-only.
- The paged half of `uniform-texture-pages.ts`/`uniform-pressure-pages.ts` is unreachable.
- `uniform-page-domain.ts` always makes every page resident.

History:
- Paged storage became mandatory in 785c2e20; long dam went 46 → 119 ms.
- The causes were per-tap `mgPageAddress` with runtime div/mod, audit atomics that serialised stages, padded coarse launches, a per-advance catalogue rebuild, and publication copies.
- "Parity" came from specialising back to a single all-resident native catalogue (91de0807/e49add19/6ebe8c29). **Multi-page residency was never measured at solver scale.**
- The only multi-page evidence is a 64³ stencil micro-probe: 32³ pages −3.2 %, 16³ pages +5.1 %.

**Locality.** Apple GPUs store 3D textures swizzled, so a 4³ tile is already local. The phi4 mirror, the 4h scratch layout and the fig-9 counters (19 % occupancy, 15 GB/s) all show the frame is bound by latency and occupancy, not bandwidth. **Pooled storage will not buy time by itself.** It must be made time-neutral:
- Keep 3D textures as an atlas of resident pages.
- Compile a 3³ page-slot window into workgroup memory per generation.
- A tap then costs `(p>>4)−myPage` → a window lookup → `origin + (p&15)`: no directory load, no div/mod, no bounds check.
- A 16³ page with a 3³ window covers R ≤ 4 tiles; the ball needs R ≈ 6. So either 32³ pages (R ≤ 8) or a 5³ window variant, and a fatal receipt when the certificate exceeds the window.

**Residency unit.** Use large pages (16³ or 32³) for residency and storage, and keep the h/4h 4³ tiles inside resident pages.
- 16³: about 66 % of a page's cells are interior to a 1-cell stencil (4³: 12.5 %), and there are 4,096 directory entries against 262k tiles.
- Estimated resident fraction on fig-7: 16³ 5.5–17 %, 32³ 10–19 %, against tile-exact 4–13 %.
- This disagrees with docs/plans/uniform-sparse-residency-plan.md, which makes 4³ tiles the residency unit and puts the brick pool (phase 3, time still O(T)) before absent tiles (phase 4). For this scene the order should be reversed.

**The absent-page contract.**
- V=0, p=0 (Dirichlet air, consistent with the free surface), velocity 0, phi = +cap.
- The closure is computed on the GPU in the census tail. It is the h, band and wet tiles, plus the certificate radius, the census departure boxes, the extension read reach, and one page of slack.
- Pages are initialised to air on entry and reset on eviction.
- Any read outside the closure is a fatal receipt word, never a substitution. The rejected finite-phi patch shows that clamping resident far phi changes results, so the clamp may only apply outside every reach.
- The extension's far states must stop at the resident boundary.

Memory at 256³: about **5.5 GiB** today. The two big contributors are:
- band rows sized to all-h capacity with the *solid* row layout in a solid-free box: 1.34 GiB;
- the arena: 1.06 GiB.

About 0.9 GiB is dead allocations: transportA, frontReceipts, CM11a L0 pressure A, the n/2 extension level, and wall C.
- Right-sizing alone brings this to about 3.1 GiB.
- Pooled 16³ pages bring it to about 0.5–0.8 GiB.

## Plan

Each step is measured with one fig7-256 run, then one long-dam run, using the probe above plus type check and CPU shader preflight. No full Dawn runs or regression suites: only targeted tests (Peter, 2026-09-29).

| step | what | fig7-256 expected | long dam |
|---|---|---|---|
| **S0 measure** | split the merged momentum/advect/traceCells launches into h-general / h-regular / 4h job time; count changed tiles per relayout | – | – |
| **S1 far-air skip** (no paging) | Three changes: (a) `retirementEvidence`, `liveSeed` and band `classify` run over a listed-tile prefilter from census bits, not one WG per tile; (b) a certified "far air" class, so merged grids skip far 4h owners, and the redistance ping-pong gets its plateau written once at adopt; (c) the census `decide` gets a global-reach early-out, builder `distance*` exits for width≠1, and surface-volume grow/growMeasure run over a list. | −5 to −7 ms | neutral |
| **S2 absent-page certificate on dense storage** | A GPU 16³/32³ page mask and resident list in the census tail. Every container pass (plan, census, builder, surface volume, n/4 extension, momentum cache, all-4h pressure L0 and native L2, transfers, acceptance) becomes a fixed direct grid-stride over the resident list, sized to its bound. Owner numbering excludes absent tiles, evicted pages reset to air, and a violated closure is a fatal receipt. | to about 43 ms | ≤ +0.2 ms (all pages resident) |
| **S3 h-tile cost** | Three parts: the momentum/advect general-path cause from S0 (a per-owner or per-direction regular certificate, not a whole-tile R), the remap grid sized to its bound and a cheaper per-frame relayout, and the band solve. | the largest unknown: needs about −15 ms | must improve too |
| **S4 h count** | Port the tight per-frame RK2 halo (fig-9 f7d6176: −16 % h) and re-check the census margins for fast-falling liquid. | −3 to −6 ms | small |
| **S5 right-size memory** | Delete the dead allocations, give the band a budget, use solid-free rows when there are no solids, and size the arena to live. | ~0 ms, 5.5 → 3.1 GiB | 200 → 120 MiB |
| **S6 pooled page atlas** | Per-generation window table, one address function behind the ~235 helper sites, remap first, and a dense publish for external readers. Reuse the transaction ABI, with a parallel builder and no indirect launches. | neutral (gate: long dam within noise), 3.1 → ~0.7 GiB | per-tap risk |

S1+S2 give about 1.3× on their own; that estimate comes from per-dispatch attribution and still needs a Dawn run to confirm. 3× also needs S3 and S4 to halve the h-list work (about 40 → 20 ms). S0 is the gate on whether that is achievable.
