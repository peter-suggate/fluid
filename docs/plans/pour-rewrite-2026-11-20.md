# Pour rewrite: a pour-only optimal reference solver for `nbflip-figure-7-pour` (2026-11-20)

Goal (user, 2026-11-20): "creating a completely fine tuned optimal method just for the
pour scene. Same physics but completely different stages. Once we have this optimal
reference we can fine tune the general brick method."

The brick engine runs the pour at 54 ms a step median, and 79 ms over its worst 25-step window
(check slot, HEAD 9039d222, the reference runs of §0). 60 fps needs 16.7 ms. Local kernel trims are spent
(memory: pour-60fps-findings). This solver answers what the pour *costs* when every
stage is designed for it, stage by stage, so that the brick engine can be tuned toward
that number stage by stage. It is modelled on seep-m1max (docs/plans/seep-rewrite-2026-11-14.md),
which reached 5x the brick engine on its scene.

**Done means both, reproducibly:**
1. Physical match with the brick engine on the pour within measured tolerance (§4).
2. A clean median step of at most 16.7 ms over the whole scene on the M1 Max, in the
   same harness, loop and device as the brick engine (§5).

A faster but divergent solver is not done; nor is a matching but slower one.

## 0. Status board

| milestone | state |
|---|---|
| P0 harness (`tools/pour-bench.ts`) and brick reference runs (two, for the spread) | done 2026-11-20: `artifacts/pour-rewrite/runs/brick-ref-{a,b}` (625 steps, every 25) |
| P1 correct uniform-h solver (`lib/methods/pour/`, method `pour-m1max`), gates G1–G7 | in progress 2026-11-21: run `pour-p1-seed` passes 17 of 19. Fails G1 @100 (drift −6.25% against a limit of 6.15) and G2 pool @325 (66.2 against 69.6, tolerance 2.3; the bore is 0.7% low and its surface flatter just after the inflow stops). Ported since `pour-p1-vc2`: the mass rules (retire, merge, split); the buried vertex hold and smoothing guard (wall interiors are air); the seed's first move taking the grid's velocity (nbSeedFirst: plug seeds had kept the stale zero, the stream's first liquid fell off the nozzle's lip, landed short of the far wall and ran up it, and the film spilt over the rim: outside 17165 at 625, now 3163); the extension closing every face not between two open cells. MAX_CYCLES 16. Median 21.2 ms |
| P2 at most 16.7 ms median, gates held | **met 2026-11-21** (HEAD 33569b6a). Timing slot, fenced (`pour-p2j-timing`): median 14.09 ms, p95 15.99, worst 25-step window 16.11 (301–325), worst single step 16.87; pipelined (`pour-p2j-timing-pipe`) 12.65 ms a step. Check slot (`pour-p2j`): median 13.91, worst window 16.18; gates 18 of 19 (G2 pool @325 fails as in P1; G1 @100 drift sits on its tolerance edge, passing or failing by run-to-run noise). Landed: P2a weighted coarse corrections (10 cycles a step to 6); P2b one red-black sweep a fine smoothing pass; P2c the brick engine's support rule in P2G and G2P; P2d the active set as the liquid's reach (median 18.1 to 16.2); P2e velocityTrace's face-centre sample from 9 loads; P2f coarseSolve as 512 threads (0.865 to 0.22 ms); P2g the velocity samplers gather from U textures (velocityTrace 2.26 to 0.80, advectPhi 1.51 to 0.60, G2P 1.08 to 0.73); P2h the step in five submits (Dawn translates at submit: worst window 18.2 to 16.6); P2i the extension in one 12 KB tile, skipping bricks it cannot change (1.29 to 0.88); P2j plain reads for single-owner accumulators, fill's whole-cell answers (velocity 1.19 to 1.04). See §5.1 |
| P3 per-stage comparison table, brick against pour-m1max on the pour (feeds the brick tuning) | **table and attribution done 2026-11-21** (§5.2, §5.3): the direct transfers gain at most 0.13 ms each; brick's heavy kernels are register-bound (caps 384–832, occupancy 10–24% against pour's 30–57%), elasticity about 0.6 |

## 1. The scene (lib/core/nbflip-paper-scenes.ts, createNbflipPour)

- Lattice 128 x 256 x 128, cubic h = 22.18 mm, y up; closed box, free-slip walls; dt = 1/60 s
  (the method's timeStep "sixtieth"); gravity 9.81 m/s^2, i.e. 0.123 cells a step^2; inviscid, no tension.
- Glass: an open cylinder on the box's axis (64, 64), inner radius 51.2 cells, wall 2.6 cells,
  rim at y = 192, its floor the box's floor. Voxel collision: a half-solid cell is solid,
  exactly `brickStaticMask(scene).solid(x, y, z)`, the brick engine's own mask.
- Inflow: a cylinder of radius 19.2 cells released at (35.7, 213.4, 64) with velocity
  (1.40, -0.84, 0) cells a step, for steps 0..315 (5.25 s) (the swept plug: inflow-boundary.ts and
  brick-source.wgsl.ts nbInflowPhi / nbInflowForced / nbInflowProjected). The scene runs
  625 steps (10.42 s).
- The stream falls about 200 cells: about 7 cells a step at impact. No substeps:
  every trace is segmented (CFL about 7).

## 2. Design (pour-m1max)

Independent of NB-FLIP: its own state, kernels and execution path. It may import
solver-neutral infrastructure: the scene model, the brick static mask (the bake reads
it on the CPU, once), level-set-fill.wgsl.ts, the inflow geometry, device limits.
Construction refuses every scene id but the pour, and any change to the geometry the
bake covers. Unknown kinds fail.

### 2.1 Baked static structure (construction, once)
- Dense, brick-major storage: 8^3 bricks, a brick's 512 cells contiguous, brick table
  16 x 32 x 16 = 8192. About 17 MB a scalar field; step time outranks memory for this
  single-scene reference. **Work** follows active bricks only.
- Solid: a per-cell open bit from the brick engine's mask; faces of exactly 0 or 1.
  Bricks wholly solid are DEAD (never work).
- Static region bits per brick: INSIDE (bore), WALL (meets glass solid), OUTSIDE (outside
  the glass, below the rim), ABOVE (above the rim). Activity is solid-aware: reach never
  crosses a solid wall, so the outside of the glass stays dry and inactive unless liquid
  goes over the rim.
- Pools and capacities sized from the measured peak plus margin; no runtime growth.

### 2.2 Step (one per 1/60 s step; all GPU; no per-step readback)
The physics of the brick engine's NB-FLIP, reorganised:
1. Active set: compact active bricks (liquid, particles, or the inflow plug within the
   CFL reach along open paths) into indirect lists.
2. Level set: semi-Lagrangian advection of the vertex phi (cubic, segmented backtrace);
   union with the inflow plug; combine with the particles in the band; redistance near
   the zero set. Deep bricks (all phi < -band) skip level-set work.
3. Momentum: semi-Lagrangian advection of the MAC velocity off the particle band; P2G
   in the band (FLIP 0.95 / PIC 0.05 blend at G2P); gravity; the inflow's forced faces.
4. Pressure: ghost-fluid free surface (theta clamp 1e-3), voxel Neumann faces, MGPCG,
   warm start, the brick engine's tolerance; the inflow's projected faces.
5. Project; extend velocity into air by the reach the next step's traces need.
6. G2P and particle advection (segmented RK through the extended velocity, exact
   voxel collision); band resampling to depth 3 (fewer particles are welcome:
   count is an upper bound only).

### 2.3 M1 Max specialisation (P2)
- Few, wide dispatches; a brick plus halo staged once in threadgroup memory; several
  sweeps a launch; 32-wide SIMD sizing; subgroup reductions.
- Deep liquid bricks (the pool below the band) cost only pressure and velocity.
- Candidates after P1, measured before adoption: a 4h composite for the deep pool,
  f16 storage, fused advect+P2G, fewer V-cycles from a better warm start.

### 2.4 Not allowed
No playback or baked trajectories; no calls into NB-FLIP; no hidden volume injection or
deletion; no substeps or smaller dt; no clamps or damping to hide error; no CPU stage per
step.

## 3. Harness (`tools/pour-bench.ts`)

- `--method=brick|pour`, both through the method registry on the same
  `sceneDocument("nbflip-figure-7-pour")`, same Dawn device setup, same loop.
- **One observable schema, one measure:** both solvers expose `volumeTexture` (a cell's
  liquid fraction, the lattice). Every `--every` steps the harness reads it after a fence
  and computes, on the CPU, identically for both:
  - V_total, V_bore (r < 51.2, y < 192), V_outside (the rest below the rim), V_above (y >= 192);
  - the inflow's delivered volume (analytic) and drift V_total / V_in - 1;
  - pool level: the median liquid column height over bore columns;
  - height profile: V per 16-cell y bin (16 bins), normalised;
  - centroid (x, y, z) of the liquid;
  - stream: per 8-cell y slab above the rim, V and mean x;
  - first floor impact step (bore, y < 4, V > 1 cell).
- Timing: fenced per-step wall time with sample steps excluded; pipelined throughput.

## 3.1 Brick reference (P0, 2026-11-20, HEAD 9039d222, check slot)

Two runs, 625 steps, sampled every 25 (a / b):

| step | V_total | drift | pool level | particles (a) |
|---|---|---|---|---|
| 50 | 96856 / 96898 | +2.45% / +2.49% | 0 / 0 | 307k |
| 100 | 178637 / 179832 | -5.52% / -4.89% | 7.3 / 8.3 | 666k |
| 200 | 384578 / 383919 | | 39.1 / 37.6 | |
| 300 | | | 61.6 / 63.5 | 644k |
| 625 | 602020 / 601953 | +1.10% / +1.08% | 72.9 / 72.9 | 184k |

Delivered at 625: 595497 cells. V_outside stays about 2 cells; V_above about 24.8k (the
inflow plug above the rim). Centroid at 625 (64.4, 36.6, 64.0). First floor liquid by
step 75. Step time median 54.5 / 53.4 ms, p95 79.8 / 77.8; 25-step windows: about 71 ms
over 76–125, 79 ms over 126–200, 40 ms after 500.

## 4. Physical gates (full scene, 625 steps)

The tolerance is measured: the brick engine runs twice; its repeat spread S(x) is the
noise floor for each observable.

| gate | observable | pass if |
|---|---|---|
| G1 | V_total at 100/200/315/450/625 | within ±3% of brick's, and drift no worse than brick's |
| G2 | pool level at 200/315/625 | within ±2 cells or 3·S |
| G3 | height profile at 315/625 | normalised L1 within max(0.15, 3x brick's repeat L1) |
| G4 | centroid at 100/315/625 | each axis within ±4 cells or 3·S |
| G5 | stream slabs at 50/100/200 | mean x per slab within ±4 cells or 3·S; V within ±15% |
| G6 | first floor impact (floorV > 1 cell) | within ±5 steps (brick: by step 75) |
| G7 | V_outside, V_above at 315/625 | within max(1% of V_total, 3·S) of brick's |

Gates are not loosened after the fact. A gate found to measure an artefact is
corrected in a dated §4.1 note with both definitions kept in the report.

### 4.1 Notes

- 2026-11-20: the bench samples every 25 steps, so the gates at step 315 read the
  sample at 325 (`tools/pour-compare.ts` takes the first sample at or after the
  gate step). G6 compares the first sample with floorV > 1.

## 5. Performance gate

Same harness, timing slot (`tools/gpu/fluid-gpu-run.sh timing`), 625 steps each,
back to back. Pass: clean median ≤ 16.7 ms, the worst 25-step window median ≤ 16.7 ms
(the brick engine's is steps 176–200 at 79 ms), and pipelined ≤ 16.7 ms. Stretch ≤ 12 ms.
Report p95, per-stage kernel time (the stage table of §0 P3), construction time and
memory.

### 5.1 P2 findings (2026-11-21, worst window 276–325 unless noted)

- **Kernel time** (kernel timer, steps 301–325, after P2f): velocityTrace 2.2 ms, advectPhi 1.6, extend 1.3,
  velocity 1.2, g2p 1.1, p2g 1.1, redistSweep 1.0, separate 0.75, seed and splat 0.6 each. The pressure takes
  6.6 ms in all: about 0.9 ms a cycle over about 7 cycles, and each fine pass is bandwidth-bound at about 0.1 ms
  a cycle. Per cycle the residual falls 8 → 1 → 0.1 quickly. Cycles 3–7 then reduce it only 2–5 times each. The
  late maxima lie along the inner glass wall (r ≈ 50, the floor corners) and under the impact (r ≈ 32); none
  lies in the film outside the glass.
- **The film outside the glass is about 2.3 ms of the worst window.** A diagnostic-only run that killed every
  particle outside the glass (not landable: it deletes volume) took the 276–300 and 301–325 windows from 19.0 and
  19.5 to 16.9 and 17.2. In the full run, V_outside is 1702 cells at step 325 and about 24% of the active
  bricks lie outside the glass. Brick (tiered) holds about 1 cell there, because its outside is at 4h, which
  cannot carry a sheet. Brick at all-h (`BrickNBFlipSolver` without tiers) spills too: V_outside 2.6 / 39 / 85
  and 29 / 390 / 938 particles at steps 150 / 200 / 300, against pour's 21 / 172 / 1553 and
  543 / 3001 / 10963 particles.
  - **The growth is the same in both.** Particle counts outside grow about 20 times from step 150 to 300 in
    each. The mechanism: band seeding of a spreading sub-cell sheet's uncovered cells adds more mass than the
    sheet holds, and the union then keeps that mass.
  - **Heat does not stop it.** Every brick particle outside the glass is at heat 2 (triggered), as is the
    falling stream (heat 1–2, so theta = 1). Porting the activity heat would not dry the outside.
  - **The spill differs:** pour spills about 10 times more. Pour's falling head follows free fall: its
    level-set front reaches y ≈ 113 at step 30, as free fall does. Brick's lags by about 15 cells at step 30
    and about 55 at step 46. Pour's floor sheet therefore reaches the far wall about 15 steps earlier and with
    about 10 times the particles. Both methods throw corner jets up the wall at about twice the impact speed:
    about 20 cells a step, against 9.4 at impact. Pour's climb reaches the rim in every sector by step 90.
  - The spill is not a pour defect relative to the reference physics, so no verbatim port removes it.
- **Ruled out or worse:**
  - staging velocityTrace's halo (slower);
  - a 4h composite for the deep pool (93k of 578k liquid cells are core-deep at step 300, and they are not
    calm);
  - AGGREGATION_OC (1.6 is best; 1.4 and 1.8 are worse);
  - 8 block sweeps at the coarse levels (no fewer cycles);
  - no level 4/5 sweeps (10.3 cycles, against 6.6);
  - an extrapolated warm start, 2x_n − x_(n−1) (7–8 cycles, against 6–7);
  - a 128- or 256-thread coarseSolve (0.72 and 0.32 ms, against 0.22 at 512);
  - folding pcgCheck into pcgUpdate (WGSL has no cross-workgroup acquire/release).
- **Dawn validation** of the indirect dispatches costs 0.4–1 ms a step (`FLUID_WEBGPU_DAWN_FEATURES=skip_validation`
  in `tools/pour-bench.ts`). The gates and timing run validated, as a browser does.

- **After P2g–P2j, ruled out or worse:** P2G at 512 threads (1.03 to 1.22 ms: more shared-atomic contention);
  P2G one component at a time in 8.8 KB (1.06: atomic-bound, not occupancy-bound); a 10-cycle cap on the
  recorded PCG tail (no change: idle cycles' empty dispatches are cheap); nine submits (17.0 against 16.6 for
  five); hardware-filtered U (r32float trilinear filtering is off by up to 5.4e-3 on values in [-1, 1]; the
  gathers are exact).
- **For the brick tuning (P3), in order of measured gain here:** gathers from per-component U textures for
  every trilinear U sample (2.7 ms a step); several submits a step instead of one (1.4 ms fenced); the coarse
  solve's thread layout (0.65 ms); extension tiles at 12 KB and skipping bricks with nothing to extend (0.4 ms);
  RMW-free reads of single-owner accumulators (0.13 ms); whole-cell fill answers (0.06 ms).

### 5.2 P3: brick against pour-m1max, stage by stage (2026-11-21)

Kernel timer, steps 301–325, check slot, HEAD 096dd641 (runs `brick-prof-head`, `pour-prof-fill`).
Brick: 289 dispatches, 68.8 ms GPU a step (fenced wall 70.5 over 276–300). Pour: 252
dispatches, 15.8 ms.

| Stage | brick ms (kernels) | pour-m1max ms (kernels) | ratio |
|---|---|---|---|
| pressure (operator, project, MGPCG) | 19.2 (42) | 6.9 (19) | 2.8x |
| level set | 16.1 (20) | 3.3 (7) | 4.8x |
| momentum (advect, P2G, transfer, uCache) | 10.3 (4) | 2.9 (3) | 3.6x |
| particles (G2P, move, seed, rebin) | 9.9 (10) | 1.5 (4) | 6.5x |
| velocity extension | 5.8 (3) | 1.0 (2) | 5.5x |
| residency, census, lists | 6.5 (40) | 0.1 (3) | - |
| present | 1.1 (2) | 0 | - |

**The work is the same size.** At step 300 brick holds F_ALL 3855 bricks (F_LIQ 1267, F_SURF
1044; 184 C pages), pour's active list 4035 (liquid 1633). Particles: 644k against 637k at 300,
773k against 879k at 200. PCG: brick about 6 iterations a step (cap 8, none capped over 295–305),
as pour's 6 cycles. So brick is about 4–5x slower *per element* in nearly every stage. pMove takes
4.3 ns a particle against pour's g2p at 1.2. It is not a larger work set, a slower solver or more
particles.

**The pour's lessons, applied directly to brick (each A/B'd, then reverted):**

| Lesson | On brick | Result |
|---|---|---|
| several submits a step (P2h) | parts in chunks of 8, 24, 48, then 64 | none (72.5/73.5 against 72.4/71.8 ms, interleaved): brick is GPU-bound, the translation is hidden |
| skip bricks with nothing to extend (P2i) | uExtend: a brick whose cells are all liquid takes US = U, UP = U - UP directly | -0.11 ms: tiered, the deep pool is 4h, so few F bricks are wholly liquid |
| two workgroups a core (P2i) | lDistance's pair list packed two a word (16.7 to 12.7 KB) | -0.03 ms |
| one fine sweep (P2b) | MGPCG fineSweeps 3 to 1 | none (19.2 to 19.9 ms); iterations unchanged |
| gathers from textures (P2g) | uCache's UNIT as an r32float atlas, fsFine two gathers a component (patch kept in fluid-scratch/brick-tune/unit-texture.patch) | uAdvect -0.13 ms, the rest within noise. Doubling fsFine's loads had cost +2.5 ms, so the loads' latency is mostly hidden |
| (diagnostic) Dawn robustness off | not shippable, a bound only | -2.5 ms (4%) |

**What this says.** Pour's wins were in its bottlenecks: idle GPU between submits, sampler-bound
traces, one workgroup a core. Brick's time is not in those. It is spread over about 130 kernels,
each paying for the general engine's per-element machinery:

- owner and tier tests;
- slot and directory lookups;
- cut, wall and seam paths;
- solid walks for the 2326 F_SOLID bricks.

The next step for the brick tuning is attribution, not more transfers: hardware counters
(occupancy, limiter, ALU) per brick kernel against its pour counterpart, from the counter
profiler (`tools/profile-mini-dam-xctrace.ts --counters`) pointed at the pour scene. That decides
between a common cause (register pressure capping occupancy, divergence) and per-path costs. For
per-path costs the remedy is lean variants for bricks the general paths cannot touch, exact by
construction.

### 5.3 P3: counter attribution, brick against pour-m1max (2026-11-21)

Instruments' Metal GPU Counters were attached for 2 s inside a kernel-timer window, a pass a dispatch, so
each Metal encoder is one kernel. Register caps come from compiling each kernel's Dawn MSL through Metal with
the size attribute stripped (`maxTotalThreadsPerThreadgroup`: under 1024 means register-bound). The probes are in
`~/code/me/fluid-scratch/brick-tune/`:

- `counters.mts` and `ctrtab.py`: the counter capture and its table;
- `splitmsl.py`, `psoinfo.swift` and `regcap.sh`: the register caps, plus an offline loop that takes an edited
  WGSL file to its cap in seconds.

**The common cause is registers.** Brick's heavy kernels are register-bound. Their occupancy matches the cap
almost exactly (resident threads a core of 3072):

| kernel | ms | workgroup | register cap | occupancy % | pour counterpart: occupancy % |
|---|---|---|---|---|---|
| uExtend | 4.45 | 512 | 640 | 20.1 | extend 56 |
| nbOperator | 4.25 | 512 | **384** (spills at 512) | 15.6 | operator 49 |
| uTransfer | 3.64 | 256 | 576 | 15.6 | |
| uAdvect | 3.64 | 256 | 704 | 20.1 | velocity 34 |
| pSeed | 3.28 | 256 | 576 | 16.0 | seed 24 |
| project | 3.00 | 256 | 448 | 13.9 | project 56 |
| pMove | 2.74 | 128 | 576 | 18.4 | g2p 28 |
| censusFine | 2.44 | 64 | 640 | 19.6 | |
| lDistance | 2.20 | 256 | 832 | 24.1 | redistSweep 57 |
| lcAdvect / lAdvect | 1.34 / 0.97 | 64 / 256 | 448 / 384 | 9.9 / 11.3 | advectPhi 31 |

These 13 kernels (cap under 1024) take about 37 ms a step. Brick kernels at cap 1024 (lResolve, lNear, the
pmg* cycle) reach 30–50%, as pour's do. In every one, ALU sits at 1.2–1.7 times occupancy, as pour's does. So
brick's 4–5x per element is about 2x from residency and about 2x from instructions: the general paths (owners,
tiers, cuts, seams, walls) execute on every element.

**What the caps respond to.** Robustness lengths are not the cause: fixed-size storage arrays leave nbOperator
at 384. The peaks are the general paths:

- nbOperator: the cold start (poStart's 4h prolongation and carry) sets the peak, 384 against 512 without it;
  the six-face band loop is next (640 without both).
- uExtend: the seed, its coarse face interpolation foUFaceU, sets the peak (640 against 768 without it; the
  whole seed costs 832).

A branch does not lower a kernel's allocation, so the remedy is fission.

**Elasticity, measured.** A warm-only operator (inexact, for timing only) took nbOperator from 4.25 to 3.53 ms
(cap 384 to 512). The exact fission came next: tileRoots marks the cells that may be cold rows, a new
operatorStart writes their starts into B, and nbOperator reads them. It passes the live-against-F_ALL test
bit for bit, and nbOperator falls to 3.29 ms. But operatorStart costs 0.72 ms, so the net is only about -0.2 ms.

On the pour, cold rows are a common path, not a rare one: the moving surface leaves new wet cells in most
surface bricks every step. The precursor must also stage the full window: poStageLight's unstaged
reads differ from poStage's at some vertex, and rowsF reads the staged values. The change was not landed (patch
`operator-start-split.patch`).

Occupancy elasticity is about 0.6: +33% resident threads gave -17% to -23%. A register diet across the 13
kernels is therefore worth perhaps 4–6 ms of 69, kernel by kernel, each an exact fission, and only where the
split-off path is rare in every scene. The other 2x, instructions a cell, is the generality itself.

**What a cap buys (2026-11-21).** Caps are now measured on the kernels as built from source (brick-tune
`dumpk.mts`: the engine's pipelines' WGSL, then `regcap.sh`), not on edited dumps. A higher cap paid off for
256-lane kernels:

- project, 448 to 640: 3.00 to 1.59 ms, occupancy 13.9 to 19.7%;
- uAdvect, 704 to 768: 3.57–3.64 to 3.32–3.36 ms, occupancy 20.1 to 20.6%.

It paid nothing for uExtend at 512 lanes: 640 to 768 (moving its seed's coarse face read out of the seed loop)
measured 4.45 and 4.47 ms, twice. Its 22.7 KB of workgroup memory most likely holds it to one workgroup a core
whatever the registers. Packing uAdvect's 6 KB trace list two entries a word changed nothing either.

The uAdvect change (fsCoarse: one C slot for its eight taps where they share a page, as fsFine) is exact and
general, and lifts no other kernel's cap. nbOperator (512 lanes, cap 384, so it spilled) reaches 512 with its cold start reading only the
staged tile window: the window holds every tile the start reaches, and the scan that stages it now also covers the
face halo's rows. Its global fallback, inline at each tap, had set the peak. nbOperator went 4.18 to 3.86 ms with no
extra pass, where the earlier cold-start fission netted −0.2 ms. phiSolid and pMove are capped by their surface
paths (the trace's foot, cubic and contact; the separation's surface shift); stripping all of phiSolid's still leaves
it at 640, so those need restructuring, not a diet. The seep's GPU work is about 1.6 ms a step, so its pipelined step time
(26.8–27.7 ms) is host-bound and these kernels cannot move it.

**project landed (exact).** Its peak was poProjectFace (npBandFace), reached only by a dry seam face: a dry h side
facing a resident wet 4h tile. projectF now flags the brick and skips those faces, and projectDry (64 lanes, over
HP, empty bricks leave at once) projects them with the same U, bound and release bits. The live-lists test checks
it bit for bit against the unchanged F_ALL path; its fixture has 7 such bricks.

- Pour kernel times: project 3.00 to 1.59 ms, projectDry 0.37 ms, GPU total 68.84 to 66.34 ms.
- Pipelined timing slot, single runs:
  - seep: 27.16 to 26.79 ms a step;
  - pour: 54.05 to 55.09 ms (+1.9%, inside single-run noise, against the kernel times).
- Gates:
  - seep 300: drift +0.21% (HEAD) against +0.33% and +0.40% (two runs);
  - letters 120: drift +0.48% against +0.46% and +0.43%.

**The level set, later on 2026-11-21.** One exact win and three no-gos.

- **lClear landed (6f282eab):** 0.755 to 0.399 ms. Its crossing pass took lCleared at all eight corners of each
  cell, 4096 calls a brick for 729 distinct corners. Each now takes it once into workgroup memory. Gates: seep
  +0.376%, letters +0.509% (exact change, run-to-run spread).
- **phiSolid's cost is the advection, not the walk.**
  - Dropping its near bricks from step 300 takes it from 2.93 to 0.20 ms, so the far vertices lAdvect hands it
    are worth about 0.2 ms. A walk-free path for far traces whose bricks beyond are clear gained nothing.
  - An open fast path tried the reverse: a staged 16^3 closed-block window, with the plain text for each vertex
    whose trace reads only open cells. It was equal to the walked text within 1e-6, but slower (2.79 to 6.03 ms).
  - With the plain text on every vertex it still cost 2.83. The 2326 F_SOLID bricks hold most of the glass's
    liquid, and their traces (foot, smoothing, velocity trace, cubic) cost what lAdvect's do.
- **Residency at 128 lanes:** lAdvect and phiSolid at 128 lanes × 4 vertices (three groups a core under cap 384,
  against one at 256 lanes) were slower: phiSolid +0.65, lAdvect +0.10 ms.
- **lUnion's θ heats** staged once a brick (4096 tile reads to 125): 1.33 to 1.26 ms, inside noise; not kept.

### 5.4 Solid handling, explored (2026-11-21)

Census at step 300 (fluid-scratch/solid-explore: census.mts, leak.mts).

- **Residency is solid-blind, but that costs almost nothing.**
  - 1111 of 3745 F bricks are within 16 cells of liquid only through the glass. Through open cells only, 2442
    remain. The seep has 122 of 1524 behind its dam, and letters has none.
  - Nearly all of them are kept by the ring of an h tile.
  - About 2700 h tiles behind the glass have φ of 2 to 8: the level set measures through the 2.6-cell wall. The
    static need also crosses it (its one-tile reach, and a wet cut brick takes all 8 tiles).
  - Diagnostics (pour-only analytic tests, reverted) removed this one cause at a time:
    - ring bricks without an h or WANT tile beside them;
    - the open-vertex wet test;
    - no band outside the glass, after the deep keep as well;
    - no static promotion outside the glass.

    Only the last two together removed about 1000 behind-wall h tiles. F stayed at 3707, and the step was +1.6
    to -0.8 ms, inside noise.
  - The wall's cut bricks and their rings pin the residency, and dry bricks are cheap in every kernel. A
    geodesic band and residency is a no-go on its ceiling.
- **The NEAR_SOLID split is not the cost either.**
  - 37% of particles go to pSolid, but only 11.6% are within 2 cells of the glass.
  - pSolid runs about 7.7 ns a particle against pMove's 6.9. phiSolid's rate equals lAdvect's (§5.3).
  - An exact per-brick clearance in place of ring 8 would gain about 0.2 ms.
- **Compiling the solid paths out buys no registers.**
  - Without the glass, the plain variant differs only in lAdvect, pMove, pSeed and project. Their caps are
    384, 576, 576 and 704 (project 640 with the glass).
  - uExtend, uAdvect, uTransfer and uCache are the same text either way.
- **The one sizeable solid cost is the all-h cut brick.**
  - 5520 of 11107 h tiles are in cut bricks. 1722 of them are air (every open vertex φ ≥ 4) and 489 deep liquid
    (φ ≤ -4). Elsewhere only 219 air and 513 deep tiles are h.
  - A composite operator that takes 4h tiles inside a wet cut brick would free about 2200 h tiles (20%). That
    means cut faces with apertures at 4h, or the dry-row rule extended to wet bricks. It is a pressure
    discretisation change.

### 5.5 Lean kernels for plain elements, tried (2026-11-21)

The idea: pour-style lean code, exact by construction, for elements whose stencil is all h and uncut, with the
general kernel deferring the rest. Three tests bound it (fluid-scratch/solid-explore: plain.mts, fixstate.mts).

- **The tier code is the whole register difference.**
  - The tiered kernel text, with `FG_TIERED` folded to false, gives exactly the all-h build's caps: lAdvect 576,
    lDistance 1024, pSeed 704, phiSolid 512, uExtend 896, uTransfer 640.
  - The particle kernels pMove, pDeposit and pSolid have the same caps either way, so a lean particle variant
    buys no residency.
- **Ceiling at the real workload.** This is a fixed-state A/B at pour step 300: the frozen state, 12 dispatches
  batched, medians of 5. The folded text is inexact at seams, so this measures timing only.

  | Kernel | Text (ms) | Folded (ms) |
  |---|---|---|
  | uExtend | 4.42 | 3.17 |
  | uTransfer | 3.79 | 2.82 |
  | lDistance | 2.21 | 1.69 |
  | phiSolid | 2.90 | 5.02 |
  | lAdvect | 0.98 | 2.17 |
  | pSeed | 3.16 | 3.82 |

  - The best case is about 2.7 ms of the 66 ms step. The level-set kernels and pSeed get slower when folded,
    because their tier paths are shortcuts: a coarse tile answers from its 4h state in place of the h walk.
  - The tier code is not the pour gap. The all-h build is 83.5 ms against 66.3 ms tiered.
- **Almost no brick qualifies.**
  - uExtend's 12³ window and uTransfer's 5³ tiles reach the 4³ tiles about a brick. A brick qualifies only if
    none of those tiles is 4h.
  - Bricks that qualify: pour 15 of 3844, seep 3 of 2574, letters 60 of 3795.
  - The h band is a few tiles thick, so every band brick borders the 4h tier. An exact per-brick lean variant
    therefore recovers about 0–3% of a 2.7 ms ceiling.
  - Per-particle plain support (§5.4 census: 50% of the pour's particles at ±2) only helps kernels whose caps do
    not move.
- **Verdict:** a no-go for per-element lean kernels. The tested reasons are that the qualifying fraction is
  about 0 for the brick kernels, there is no cap change for the particle kernels, and the ceiling is small.
- **Where the ceiling is, within uExtend** (inexact variants, same A/B; 4.48 ms text):
  - the 4h state stage and its reach bits, 0.47 ms;
  - xProlong, 0.28 ms;
  - the swept bits, 0.15 ms.

  uExtend runs one workgroup a core: 640-thread cap and 22.7 KB of workgroup memory. Its stages before the
  first barrier are therefore exposed latency. Any gain has to come from making those paths cheaper in every
  brick, not from skipping them in plain bricks.

### 5.6 Fixed-state attribution, then exact skips (2026-11-21)

The method:
- A frozen state at pour step 300 is the baseline (fluid-scratch/solid-explore/fixstate.mts).
- Inexact text variants each remove one path, which locates a kernel's hot work.
- The exact change is checked on the pour's and the seep's frozen states, then the gates are run:
  - fixsep.mts compares records bit for bit;
  - fixseed.mts compares bank multisets and set words;
  - fixeq.mts compares buffers word for word.

Landed:

| Commit | Change | Frozen-state result |
|---|---|---|
| 395a590b | pSeparate: the separation fissioned out of pMove. It was 1.8 of pMove's 2.9 ms: 64 neighbour reads per record, at pMove's 576 cap. | pMove + pSolid 4.46 to 4.12 ms. Profile: pMove 2.75 to 1.35 ms, pSolid 1.80 to 0.94, pSeparate 1.74. |
| 1c23d844 | pSeed's site loop skips tests that cannot pass. psSeam takes a staged coarse-tile pre-test. A covered shallow cell tests the source first and seeds nothing without a source. | 3.08 to 2.49 ms (profile 2.88 to 2.12). |
| 73df23a8 | uTransfer reads and clears its deposit words plain. The returning atomicExchange was 0.67 ms. Atomic load then store was slower, at 4.85 ms. | 3.65 to 3.40 ms. |

- Pour profile 301–325: 66.27 to 64.21 ms GPU.
- Gates:
  - seep 300 drift +0.124% and +0.137%;
  - letters 120 drift +0.49% and +0.44%.

Tried and slower:
- reading the separation's 8 runs into a private array first: 2.55 against 2.07 ms;
- caching a lane's 2³ brick slots in workgroup memory: 2.5 ms, because the directory's reads already hit the cache.

Attributed, not yet changed:

| Kernel | Total (ms) | Attribution |
|---|---|---|
| uExtend | 4.3 | seed fill 1.56, Jacobi 0.7, S4 stage 0.47 |
| uTransfer | 3.4 | C pages 0.75; the staging is the rest, with no single hot path |
| censusFine | 2.5 | ncSurfaceDisplacement 0.88 |

uExtend's seed fill is inherent to the 12³ window that two sweeps need. Only the cells within L1 distance 2 of the brick are read, so about 20% of the seeds could be skipped.

## 6. Process

Single agent (memory: single-agent-only). Commit verified milestones with
`tools/safe-commit.sh`; `npm run check:types` and `npm run test:unit` before each.
GPU runs one at a time, short. Each commit that touches shared code also reports
seep 300 / letters 120 drift and step time; pour-only files do not change them.
