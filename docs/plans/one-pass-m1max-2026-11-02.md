# One pass per page: the step redesigned for the M1 Max (2026-11-02)

Question (user): "think about the M1 MacBook Max architecture. What can we
find out in one pass/launch?" Target: the pure BULK16 tank's ~6 ms fixed
step floor and ~88 µs per surface column (2 resident 4h pages). Dispatch
trimming is ruled out; this is an architecture proposal. No code or GPU runs
went into it. It rests on the pure-tank measurements in
`artifacts/bulk16/pure/` and a reading of the encode path.

## 1. What the measurements say

Pure tank, 1 cm, 300 pipelined steps (`tools/probe-reservoir-bulk.ts --pure`,
kernel means from steps 241–300, every dispatch timed alone):

| | 8·8·4 (64 resident, 224 BULK16) | 16·8·4 (128 resident, 448 BULK16) |
|---|---|---|
| step, pipelined median | 8.89 ms | 11.62 ms |
| dispatches a step (direct + indirect) | 397 (360 + 37) | 398 |
| distinct kernels | 284 | 284 |
| workgroups a step | 249,515 | 259,461 |
| sum of kernel means | 10.46 ms | 13.67 ms |
| mean per dispatch | 26.3 µs | 34.4 µs |
| median kernel / p10 / p90 | 20.8 / 5.5 / 97 µs | 22.9 / 5.5 / 130 µs |

- 211 of the 284 kernels don't change when the box doubles. 73 grow by more
  than 10 µs. The +3.2 ms is about 11 µs for each of those 73 kernels, which
  is about 0.17 µs per added page per kernel. No single kernel is the cost:
  every kernel re-reads the resident set from memory.
- The real data is small. 64 pages × 512 4h cells is 32,768 cells. At about
  400 B touched per 4h cell a step (velocity, phi, pressure work arrays and
  extension; an estimate), that is 13 MB a step. That's 33 µs at 400 GB/s,
  and it would sit in the SLC. A surface column is 1,024 cells, about
  400 KB, about 1 µs of DRAM time against the 88 µs measured. **The step
  runs about 100x slower than its bandwidth bound and far slower than its
  compute bound.** It is bound by the number of passes and by latency.
- The code is shaped as one thread per tile and one dispatch per stage. 251
  WGSL entry points are `@workgroup_size(64)`, against 28 at 256. Each
  stage re-derives the address chain (umPageWord, block, tile:
  `uniform-address-plan.ts:52`, `uniform-lattice-address.ts:220`) and
  reloads its fields from global memory. bulk16-plan:4358 measured the
  result: "a launch costs about 25 µs, and most of that is the kernel's own
  chain of dependent loads".

## 2. M1 Max facts that matter

| fact | value | status |
|---|---|---|
| GPU cores / FP32 ALUs | 32 cores × 128 = 4,096 ALUs, about 10.4 TFLOPS | known (Apple spec) |
| SIMD-group width | 32 | known (Metal) |
| threadgroup memory | 32 KiB a threadgroup | known (Metal limits); Dawn exposes it as maxComputeWorkgroupStorageSize; check the adapter (`lib/core/webgpu-device-limits.ts:12`) |
| threads a threadgroup | 1,024 | known (Metal); already read for the arena's tail (`uniform-pressure-arena.ts:494`) |
| unified memory | about 400 GB/s LPDDR5; CPU and GPU share it, so GPU buffers are RAM | known |
| system level cache | about 48 MB | reported (die analysis), not an Apple figure |
| register file, L1/L2 sizes, threadgroups resident a core | not published | assumed: several 256–512-thread groups a core |
| forward progress across threadgroups | none guaranteed (Metal, WebGPU) | known: **no grid-wide barrier**, so no persistent megakernel with spin-waits |
| order between dispatches | WebGPU makes each dispatch see the last one's writes; Dawn/Metal serialises them with a drain between them | assumed implementation; consistent with the ≥5.5 µs floor of the cheapest kernels |
| indirect dispatch | Dawn validates each one with a dispatch of its own | known (bulk16-plan:4361) |
| subgroups, shader-f16 | requested when present (`lib/core/gpu-startup.ts:132`) | known: simd shuffles and reductions are available inside a group |

What this means:
- **The only sync that costs nothing is `workgroupBarrier()`** inside a
  threadgroup, a few tens of cycles.
- **Every global sync is a dispatch boundary**, about 5–25 µs of drain,
  launch and reload, and Dawn serialises them all.
- The fast design therefore does as many stages as it can between global
  syncs, inside one threadgroup and on threadgroup-resident data.
- **The natural unit is the page.** A 4h page is 8³ = 512 cells. With a
  one-cell halo (10³ = 1,000 cells), 8 f32 words a cell come to 32 KB: a
  page and its halo fit one threadgroup, at the limit. They fit comfortably
  with f16 staging or a z-slab march: keep 3 of the 10 slices resident and
  march along z, the 2.5D blocking of stencil codes.
- **Occupancy.** 64 resident pages give 64 groups for 32 cores. That is
  enough at 256–512 threads a group, and vast scenes fill the machine.

## 3. The step, stage by stage, and how far each stage's dependencies reach

Order as in `UniformMixedFrame.advance` (`uniform-mixed-frame.ts:987-1315`)
and `encodeRelayoutHead` (`:1563-1606`). Radius: **T** = within the 4h
tile; **P+k** = within the page plus a k-cell halo; **G** = global (needs
all pages, a prefix sum or an allocator).

| # | stage (encode site) | reads → writes | radius |
|---|---|---|---|
| 1 | solid coarse record (`:1040`), static | solids → cut map | P (once; skipped while static) |
| 2 | work heat, freeze head (`:1042-1045`) | lagged receipts, 16h regions → tiers | P+1 region |
| 3 | census: plan, geometry, authority, extension, `relayout.encode`, detail admit/retire/settle, remap, phi twins (`:1563-1606`) | phi, velocity → ownership, residency marks, pool claims, slot claims | marks P+reach; **compaction, page pool commit and slot allocator G** |
| 4 | support plan and certificate (`:1076, 1087`) | velocity → support planes, local speed | P+CFL reach; max speed is a reduction (atomics) |
| 5 | 4h sampling cache, hanging taps (`:1091`) | velocity → taps | P+1 |
| 6 | surface volume begin (`:1097`) | phi → budget | reduction (atomics) |
| 7 | NB-FLIP move: advect and slab sort (`:1104`) | particles, velocity → particles | particles move ≤ CFL, so P+1 if bucketed by page. **Today's slab sort is G** |
| 8 | surface advect, phi resolve (`:1108`) | phi (old) → phi* | P+CFL (old field, read only) |
| 9 | reconstruct, clearAir, retire, redistance, revisions (`:1116-1132`) | particles, phi* → phi | P+band width (≤ 2 tiles) |
| 10 | 4h phi base, pressure geometry (`:1145-1150`) | phi → fractions, apertures | P+1 |
| 11 | momentum, plan, authority, P2G (`:1317-1323`) | velocity (old) → u*; particles → grid | P+CFL (old field); P2G P+1 |
| 12 | extension retain or snapshot (`:1158-1167`) | u* → companion | P+sweeps |
| 13 | dynamics classify, forces (`:1170`) | u* → u* | T |
| 14 | transfer to pressure, root owners, authority, warm copy, RHS (`:1180-1203`) | u*, apertures → rhs | P+1 face |
| 15 | bulk select (`:1209`) | tiers, 12³-tile reach → member flags | P+4 tiles |
| 16 | **pressure solve** (`:1210`), about 32 kernels, about 2 ms | rhs → p | **G** |
| 17 | bulk faces, project, freeze accumulate, transfer back, band project, dynamics publish (`:1212-1220`) | p → u | P+1 face |
| 18 | extension prepare/finish (`:1221-1238`) | u → extrapolated u | P+sweeps (hierarchy within a page) |
| 19 | presentation copies, receipts (`:1243-1265`) | → presented fields, readback | T, plus a few copies |
| 20 | plan tail (`:1273`) | u → next head's support | P+CFL |

The structure follows directly:
- **Every stage but 3's compaction and allocation, 7's sort and 16 is local
  to the page plus a small halo.**
- The halos chain, but stages 8 and 11 read only the *previous* field, which
  is read-only global data. Their reach costs bandwidth, not sync.
- A chain within a launch needs redundant halo work only where a value made
  in this launch is read across a page face: u* on a page's high faces for
  the RHS (stage 14), and the extension's sweeps (18). With faces owned at
  the low side, that is 3 × 64 of 1,536 faces, about 12% extra a page.

## 4. Proposal: four global syncs a step

```
K1 census       (relayout frames only)  mark → scan/compact → allocate     3 launches  G
K2 page-pre     one group per resident page (+ one per 8³ h brick):
                heat/freeze → plan → taps → surface advect → reconstruct /
                redistance (band) → geometry → momentum → P2G → forces → RHS
                + bulk select (reads lagged tiers)                          1 launch
K3 pressure     see §5                                                      4–6 launches G
K4 page-post    project → band project → transfer back → extension →
                dynamics publish → plan tail → presented values in place
                + receipts by atomics                                       1 launch
(K0 particles   move + page-bucketed counting sort in threadgroup memory,
                migrants to an overflow list read by K2; an empty page
                group exits at once)                                        1–2 launches
```

- **Launches a step:** about 8–12 (3 more on census frames), against 397
  today.
- **Expected floor:** about 10 launches × about 10–25 µs of drain, plus the
  data. The data is 13 MB for 64 pages, about 35 µs at DRAM bandwidth and
  less from the SLC. That is **about 0.2–0.4 ms for the 64-page tank,
  against 8.9 ms**.
- **Expected cost a resident page:** global round trips drop from about 280
  to about 6, so 48 µs × 6/280 is about 1 µs plus compute: **about 1–5 µs a
  page, against 44**.
- **Ceiling at a 10 ms budget:** with about 3 µs a page and the simulation
  given 5 ms, that is about 1,500 resident 4h pages, about 75 m² of surface
  at 1 cm, before BULK16's 0.7 µs a page (as measured; it would fall too).
  That is the "vast realtime" regime.
- **Reductions without syncs:** the volume budget, max speed, residual norms
  and receipt counters go through atomics or subgroup reductions into a
  word a step. Their readers are the next step's head or the lagged
  receipt, as today (`UNIFORM_MIXED_RECEIPT_RING`).
- **Two group types share one dispatch:** 4h page groups and h brick groups
  (8³ h cells, the same 512 cells). A small indirect list orders them, so
  Dawn validates 4–6 indirect calls a step, not 37.
- **What it costs in code:** stages become device functions that run on a
  threadgroup tile in place of global textures. The entry points shrink
  from 284 to about 15. Pipeline compile, 122 s of construction today
  (`solver-pipelines`), should shrink with them, but each pipeline gets
  much larger. That is a risk (§6).

## 5. Pressure in one page launch plus a tiny coarse solve

- **Today:** a composite multigrid on the brick arena, with the band's h
  rows, native 4h rows, the 8h/16h/32h levels a page each, and a 32h tail
  in one workgroup (`uniform-pressure-arena.ts:24-46`). It is about 32
  kernels and 2.0 ms at a single V-cycle. Its single-group tail grew from
  0.15 to 0.43 ms when the box doubled. Level 2 already smooths with "an
  overlapping box two deep": a page-local Schwarz smoother, spread over
  kernels.
- **Proposed:** a two-level domain decomposition, with a coarse problem
  that BULK16 already provides.
  - **P1 (page launch, folded into K2's tail).** Each group holds its
    page's 512 4h rows, plus its h brick rows if a band page, in
    threadgroup memory. It runs a whole local V-cycle (4h → 8h → 16h, 512
    → 64 → 8 rows) with barriers, and red-black overlapping-Schwarz
    sweeps. Neighbour values are the previous iterate from global memory.
    It writes its 8 16h residuals.
  - **P2 (coarse).** The coarse unknowns are 8 16h rows a liquid page.
    Resident pages restrict to them, and BULK16 slots *are* them
    (`uniform-bulk-pages.ts`, 8 pressures a slot). The pure tank has 256
    pages, 2,048 rows; the vast seep has about 300 pages, about 2.4k rows.
    At or below about 2–4k rows, one 1,024-thread group solves it: about 2
    rows a thread, with 32h and 64h levels or CG on subgroup reductions,
    all in one launch. Past that (10k pages, 80k rows), the 16h level runs
    as a page-group launch and the tail moves to 64h (1.25k rows), one
    group.
  - **P3 (page launch, the head of K4).** Prolong the 16h correction,
    post-smooth locally, project.
  - **Launches:** 2–4 a cycle. The warm start (`:1192`) already makes rest
    a 0–1-cycle solve (pure tank: 1 cycle a step at 8·8·4, 0 at 8·2·4).
    Krylov dots need a global reduction each, so the outer iteration is
    stand-alone MG or flexible CG with at most 1–2 dots a cycle, carried in
    the P1/P3 launches by atomics.
- **What it buys:** about 2 ms becomes about 0.1–0.3 ms. The coarse solve
  scales with liquid pages ÷ 8, not with the box, so pTail's box growth
  goes away.
- **Risk:** block-Jacobi or Schwarz between pages converges more slowly
  than today's multiplicative sweeps. A 2-level additive Schwarz method
  with a coarse space is known to be scalable (Smith, Bjørstad and Gropp,
  *Domain Decomposition*, ch. 3), but its cycle count on the seep's thin
  throat and the dam is unknown. Measure it before committing.

## 6. Risks

1. **Threadgroup memory at the limit.**
   - Problem: 8 words × 1,000 cells is 32 KB, and the fused chain wants
     more fields live.
   - Mitigation: a z-slab march (3 slices live), f16 staging of read-only
     inputs (shader-f16 is present), and recomputing coefficients from
     apertures.
2. **Registers and occupancy.**
   - Problem: a large fused kernel may spill and run 1 group a core.
   - Note: with 64–1,500 groups that is still far better than 397 serial
     drains.
   - Measure: occupancy with the existing `gpu-frame-profile` skill
     (xctrace).
3. **Load imbalance.**
   - Problem: band pages carry h bricks while quiet pages do almost
     nothing.
   - Mitigation: a page group exits early on COLD or frozen regions (the
     `cold` and `freeze` state exists).
4. **Shader compile time.** It may get worse a pipeline but better in
   total. Gate it like today (`check:gpu-compilation-policy`).
5. **Correctness of a rewrite.**
   - Same physics, new schedule. Results won't be bitwise equal: the
     iteration order inside a group changes, and Schwarz replaces the
     arena's sweeps.
   - Gates as always: the rest ladder with drift 0, seep 300 and letters 120
     drift, pool rest, and the vast rung.
   - Port stage by stage behind the existing stage boundaries (the `trace`
     phases), page-pre first.
6. **The census stays global** (prefix sums, 3 launches). It runs every
   step today. Skipping it on quiet frames is a separate question (idle
   advance already does, `:1066`).
7. **Assumed facts** to check with a 10-line probe first: whether Dawn
   serialises every dispatch, and the adapter's workgroup storage limit.

## 7. First experiment: a standalone Dawn microbenchmark

`tools/bench-one-pass-page.ts`, node webgpu like the probes, with GPU
timestamps and no solver.
- **Synthetic set:** N ∈ {64, 128, 512, 1,536} resident 4h pages. Each is a
  page word into a block pool with an 8³ cell block (the real umPageWord
  indirection). Liquid everywhere, a flat surface at the top layer.
- **Chain:**
  1. semi-Lagrangian velocity advect (trilinear, previous field);
  2. gravity;
  3. RHS (divergence);
  4. 4 red-black sweeps of the page-local 7-point Poisson;
  5. gradient projection;
  6. 2 extension sweeps.
- **Variant A (today's shape):** one dispatch a stage and a sweep, 64-thread
  groups a tile, global textures. About 14 dispatches.
- **Variant B (proposed):** one dispatch, one 512-thread group a page,
  fields in threadgroup memory with a 1-cell halo, workgroup barriers
  between stages.
- **Numbers to beat:**
  - A should reproduce today's regime: about 20 µs a dispatch, about 0.17 µs
    a page a kernel.
  - B must reach ≤ 0.1 ms total at N = 64, and a slope ≤ 1 µs a page up to
    N = 1,536.
  - **Go/no-go:** if B is under 5x better than A at N = 64 and N = 512,
    stop. The latency model above would be wrong.
- **Second experiment, only if the first passes:** P1+P2+P3 (§5) against
  the arena, on the pure tank's own rest field dumped from the solver.
  Measure cycles to the arena's accepted residual on the tank, and on the
  seep throat's page set.

**Result (2026-11-02, `tools/bench-one-pass-page.ts`, timestamps unquantised, one exclusive run).**
Medians of 60 steps, µs. E: 14 dispatches of a one-load, one-store kernel
over A's grid. E1: the same, one workgroup each (the launch floor).

| pages | A (14 dispatches) | B (one pass) | A/B | E | E1 |
|---|---|---|---|---|---|
| 64 | 152 | 48 | 3.2 | 55 | 57 |
| 128 | 379 | 83 | 4.6 | 51 | 36 |
| 512 | 452 | 270 | 1.7 | 104 | 39 |
| 1,536 | 1,224 | 784 | 1.6 | 181 | 56 |

- **Go/no-go: no-go as written.** B is under 5x better at 64 and 512 pages.
  It does meet its absolute targets: 48 µs at 64 pages, 0.50 µs a page.
- **A launch costs 3–4 µs** (E1), not 20. Synthetic A spends about 11 µs a
  dispatch at 64 pages, so most of that is the kernel's own work and latency.
- **B is throughput-bound at scale.** Its 24 KB of threadgroup memory allows
  one 512-thread group a core, and its 20 barriers serialise the chain. At
  1,536 pages it is only 1.6x ahead.
- **The solver's mean is 26 µs a dispatch** on the pure tank's 64 resident
  pages (§1). That is about 2.4x this A's 11 µs over the same set and work
  shape. If every solver kernel cost what A's do, 397 dispatches would be
  about 4.4 ms against 8.9; the launch floor is about 1.4 ms of that.
- **So the per-kernel cost is the larger lever.** Fusing stages pays where a
  chain is latency-bound at small N (up to 3x). It does not pay a 5x rewrite
  of the step. Next: the pure tank's top kernels against A's equivalents
  (their address chains and loads a cell).

## 8. Addendum: particle and h tiles on the paddleboats wake (2026-11-02)

`tools/probe-reservoir-bulk.ts --scene=paddleboats --bodies --criteria`, 600
steps, wheels at 1/3 throttle, hulls moored, app profile (shape 0.5 h,
displacement metric). Steps 300-600 averaged. Two runs of the same config
spread 422k-610k particles and 7.5k-9.5k h tiles, so only differences
beyond that band count.

- **Who asks for particles:** the shape criterion, almost alone (99% of
  required tiles, 75-80% shape only). Impact about 170, thin about 85,
  approach about 1, strain and rotation 0.
- **Heated vs required:** heated tiles are about 1.9x required, from the
  26-neighbour spread at 0.5 and the 0.5 s fade.
- **Shape tolerance 1 h** (the Surface dial's low end) cuts average
  particles 35-50%, peak particles 52% and step time about 11%. Drift is
  unchanged.
- **Fade 0.25 s** cuts the peak only. **Sparse lifetime 0.5 s** is within
  noise: heat, not age, retires the wake's particles.
- **h tiles vs required:** h tiles were 7.5k against 0.7-0.9k required,
  for two reasons:
  - the census holds every tile a body's *bounding sphere* reaches
    (uniform-mixed-bodies.ts markTiles); a 1.4 m hull has a 0.76 m
    sphere;
  - the hold adds more.
- **Body tiles by shape distance:** they are now marked by the shape's own
  1-Lipschitz distance, with the sphere as a prefilter only. The bound per
  tile is:
  - centre distance minus the tile's half diagonal,
  - against (|v| + |w| r) · horizon plus one cell.

  Results:
  - paddleboats at 1 h: 6.6k → 4.2k h tiles, pressure band tiles
    4.6k → 2.5k;
  - churning tank: 1.1k → 0.5k h tiles, step 17.8 → 15.1 ms, drift and
    max speed on the same trajectory.

The h ceiling (uniformFineTileCeiling):
- It never falls below 1.25x the census seed. Before this,
  dam-break-boxes failed at construction (1078 seed tiles against 1024).
- A start's free-face tiles (uniformCensusReach.freeTiles) each get one
  slot's sheet (128 tiles).
- A start with a front is capped by its slots' whole 512 tiles, not 128 a
  slot.
- garden-dam-break: refused 560 → 0, peak 666/1380.
- dam-break-boxes: peak 1689/4096, refused 0.
- seep under `--pool=rule`: 0.969 GiB (was 0.965).

## 9. Why the wake's particles never retire (2026-11-02)

`--criteria --phi-profile` on paddleboats (app profile, shape 0.5 h, step
600) classifies crossing h tiles as core (heat), collar (spread heat only),
carried or cold, and replays the census's four terms on the CPU from the
end-step phi:

| class | plane-fit rms | worst | census median: worst / vol / curve |
|---|---|---|---|
| core | 0.21-0.25 h | 0.5-0.6 h | 0.41 / 0.51 / 1.05 h |
| collar | 0.03-0.05 h | 0.08-0.1 h | 0.08 / 0.07 / 0.29 h |
| cold h | 0.01 h | 0.03 h | 0.03 / 0.02 / 0.18 h |

- The core tiles' surface is the particles' union of spheres. It carries
  about 0.2 h rms of bumps on a nearly flat surface (slope 0.12), and 12%
  of core tiles hold spray blobs above it. The same water, once cooled,
  is flat to about 0.03 h. One binomial pass only halves the bumps, so
  they are not just grid-Nyquist noise.
- The census measures those bumps as 4h error at or above the 0.5 h
  tolerance. That keeps the tile hot and its particles alive, and the
  particles keep the bumps: a latch. The worst term's 0.41 h median is
  about what the maximum of ~50 crossings drawn from 0.2 h noise gives.
- The churning tank's core tiles read the same (rms 0.2 h). The small dam
  box cools completely, so the latch needs continued forcing.

**The curve term now uses the level set's curvature,**
div(grad phi/|grad phi|) (uniform-level-set-curvature.wgsl.ts), not
laplacian(phi). Interface vertices keep the magnitude that places their
zero (arenaStore), so phi_nn there is no geometry:
- Smooth h tiles above 1 h on the curve term: 12-16% → 3-5% (CPU replay).
- Wake core curve median: 1.05 → 0.53 h.
- tests/uniform-level-set-curvature.test.ts: a sphere's distance field, the
  same field scaled 3x, and r^2 - R^2 all give 2/r within 4%.

A/B, one exclusive run each (`tools/gpu/fluid-gpu-run.sh timing`), default
tolerance, base → curvature:

| scene | median ms | drift |
|---|---|---|
| seep (pool rule) | 30.5 → 32.2 | +2.6e-4 → −1e-5 |
| letters 120 | 52.6 → 54.5 | −1.60% → −1.54% |
| vast | 26.1 → 24.6 | −3e-5 → −4e-5 |
| dam | 13.1 → 12.9 | −0.27% → +0.05% |
| churning tank | 16.3 → 16.0 | −0.85% → −0.85% |

The paddleboats wake changes within run-to-run noise: 67.3 → 69.6 ms, core
tiles 1290 → 1363, 519k → 659k particles against a band of 422-610k. So
this is a correctness fix of the measure, not the wake fix. The remaining
cause is the particle surface's own roughness, which sits at the tolerance.

**Where the bumps come from** (paddleboats, step 600, same configuration):
- Particles alone (`--zb` replays the sort's positions on the CPU): the union
  of spheres reads 0.19 h median rms in core and 0.22 h in collar.
  Zhu & Bridson's averaged surface reads 0.135 h at R = 1.5 h and 0.105 h
  at R = 2 h. The particles' placement is irregular everywhere. The collar
  is smooth only because the coupling keeps the grid's phi' there.
- Widening the coupling's ripple slack (0.076 → 0.25 → 0.5 h, so particles
  correct phi' less) makes core rougher: 0.27 → 0.31 → 0.375 h. The
  particles damp the bumps; phi' carries them. So the source is the
  particle-transferred velocity.
- APIC transfer (particleTransfer "apic") halves core rms (0.21 → 0.11 h)
  and the worst crossing (0.41 → 0.30 h median). In that run particles and
  h tiles grew instead (1.28M, 12k h tiles at step 600, against ~0.4M and
  ~5k), which needs explaining before APIC can be judged.

**The analytic suite found a Galilean bug instead** (`probe-nbflip-analytic.ts`,
FLIP default against APIC). FLIP failed pure translation: centroid 15.7 h,
velocity error 0.62 m/s, 2.6 J of internal KE in 20 frames; APIC 0.05 h.
Bisected to 3500c506 (2026-10-28), which ungated shape heat: before it the
translating block had no particles and the grid carried it exactly. With
particles, the initial band's seeds (seedCell's emit, seedInitial) lacked
the nbSeedFirst mark, so swept FLIP read their before.xyz (their velocity,
kept as residency travel) as a delta: 0.75 → 1.46 m/s in one step. Fixed in
13b1996b (tests/nbflip-galilean-seeds.test.ts): translation 0.044 h, free
fall 0.006 h, six-scene sweep within noise. The wake is unchanged by it
(flipDelta "extended" on paddleboats: core rms 0.237 h).

**Shape heat and the wake.** 3500c506 made shape heat particles whatever the
motion (Sato Eq. 4); before, only near a moving body. A/B against the fixed
build (one exclusive run each):

| | ungated (HEAD) | body gate (pre-10-28) | shape starts heat on cold tiles only |
|---|---|---|---|
| boats particles / ms | 474k / 63.1 | 280k / 58.1 | 558k / (GPU contended) |
| letters drift / ms | −1.54% / 53.5 | **−4.21%** / 40.4 | −1.54% / (contended) |
| rest-step late speed, surface | 0.041 m/s, 0.04 h | 7e-6 m/s, 5e-5 h | 0.061 m/s, 0.50 h |
| free fall / dam energy rise | 4.3% / 3.4% | 5.75% / 5.5% | 4.0% / 3.9% |

The body gate removes the wake's particles but the falling letters lose
2.7% more volume: their corners need shape heat (Sato's point). Letting
shape only start heat does not cool the wake: tiles re-heat as they cool.
Neither is committed.

**Shape heat gated on surface speed** (dt·|u|_max/h > 0.02, the tile's max
component speed in the FLIP copy's unused defect word): rejected. rest-step
got worse (surface 0.04 → 0.28 h, late speed 0.041 → 0.069 m/s, 7.9k
particles stay): the particles' own stirring holds the tile above the
threshold. Boats particles were unchanged (467k).

**APIC's growth is retained wake, not a fault** (`--particle-census`,
paddleboats step 600): live 1.0M against FLIP's 0.58M, almost all of it slow
deep collar samples (heat ≤ 1, depth < −1: 640k against 323k; mean speed
0.156 against 0.216 m/s). Shape-hot tiles grow only 796 → 993; heated tiles
1643 → 2476. APIC dissipates the wake's shape less than FLIP 95%, so the
shape criterion keeps more of it hot. The pressure residual at step 600 is
no worse (0.013 against 0.015); the earlier 0.196 was a single report-step
spike, which FLIP shows too (1.23 at step 100). So the wake's cost is the
shape criterion's, under either transfer.

**A lattice-aligned corner at rest coarsens and loses volume** (parked).
The dam break loses 0.53% in frame 1 with no particles; held at zero
gravity, it still loses 0.38% (66 cells) in one step and then holds. The
free top-front corner tile coarsens at the first census: 4h carries a corner
on its lattice exactly, the displacement test reads zero, and the curve and
corner terms count only for moving surfaces (umClassMoving). One step later
its corner phi reads (16,28) −h/30, (12,32) −h/30, (16,32) +h/15 (all were
0). With the surface along both coarse faces, that swings the 4h zero set
across two thirds of each face: the owner's fill is 0.87. Counting the
curve and corner terms at rest too fixes the column (frame 1 +0.02%, frame 3
−0.14% against −0.95%) but is far too broad: boats 63 → 143 ms with 1.8M
particles, rest-step 0.29 m/s, dam energy rise 3.4 → 11.8%. The right fix
is whatever perturbs a resting coarse corner's phi (not yet found; not
nbBuildDistance, which keeps an exact zero). It is worth 0.5% of the dam
break's 3.6%: the rest is lost during the collapse, with particles.

## 10. Where the move and the vast seep's time goes (2026-11-03)

**Letters move: occupancy confirmed (xctrace counters, 35 advances).** The
frame schedule merges the move with its neighbours into one encoder label,
so the move was found as the interval right after the local speed
certificate whose length grows with particle count, from 9.2 to 17.0 ms. Its
counters:

| counter | value |
|---|---|
| compute occupancy | 13.2% |
| ALU utilisation | 18.9% |
| last-level cache | 27.7% |
| read bandwidth | 21 GB/s |
| partition placement | 1.0× (spread evenly over all four) |

That is about 400 threads a core, starved per core. It matches the
register cap of 384 threads that nbflip-particle-scaling-2026-10-27
follow-up 2 found offline.

**Solid-free twin of the move: no lever.** One run of `advectParticlesDeferred{umSolidFree=1}`:

- move 10.68 ms against 11.02 ms;
- the other kernels in that run were about 7% slower, so this is within noise.

The sampler, `particleDepthIn`, `nbFinish` and `nbRigidContact` each cap the
kernel at 384 threads on their own, and the body alone caps it at 512.
Folding away the solid code leaves the sampler's cap in place. Reverted; the
twin was also rejected in e4041e9a. The lever is the live state of all four,
which can raise occupancy by at most 33% (384 → 512 threads).

**Vast seep (`reservoir-seep-vast`, 80 steps, kernels from step 60).**

- **Step time:** median 22.8 ms; serialised kernel sum 33.5 ms over 286 kernels.
- **Workgroups:** 339k direct workgroups a step (`--dispatches` now saves
  `dispatchesByLabel`). At about 1 ns each on the §7 microbenchmark, that is
  about 0.3 ms, so empty launches are not the cost.
- **h tiles:** `probe-nbflip-occupancy` now maps the dominant census reason
  per column. The 1,108 h tiles at frame 60 lie between the dividing wall
  (reason solid) and the catch chamber's jet and splash (join, travel,
  crossing). That is real detail, not a census artefact.
- **Pressure arena:** the largest group, 5.8 ms a step. It hits its 4-cycle
  cap every step, at residual 0.069 against the 0.05 target. The header
  history (words 8–12) per frame:

  | frame | after cycle 1 | 2 | 3 | 4 | final |
  |---|---|---|---|---|---|
  | 20 | 90.1 | 3.61 | 0.819 | 0.153 | 0.052 |
  | 40 | 25.8 | 3.76 | 1.29 | 0.46 | 0.075 |
  | 60 | 25.9 | 3.47 | 1.05 | 0.41 | 0.069 |

  The cycle factor degrades from about 0.13 to 0.30–0.39 after the first
  cycle. The Krylov step then takes the last one by about 0.17. At a steady
  0.13 a cycle, three cycles would reach the target and save about 1.45 ms
  a step on the vast seep and about 2.5 ms on letters. That slow mode after
  cycle 1 (seams? the frozen active set?) is the next lever, measurable from
  the history words without any timing run.
