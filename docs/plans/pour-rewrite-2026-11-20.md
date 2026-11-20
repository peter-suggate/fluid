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
| P2 at most 16.7 ms median, gates held | median met 2026-11-21; worst window open. Run `pour-p2e` (HEAD 6f246aec, check slot, fenced): median 15.81 ms, p95 19.04, worst 25-step window 19.19 (301–325; steps 126–350 all sit above 16.7). Gates unchanged at 17 of 19 (G1 @100 drift, G2 pool @325, as in P1). Landed: P2a weighted coarse corrections (10 cycles a step to 6); P2b one red-black sweep a fine smoothing pass; P2c the brick engine's support rule in P2G and G2P; P2d the active set as the liquid's reach (rule B, dead-trace filter, splat and operator confined to active bricks; median 18.1 to 16.2); P2e velocityTrace's face-centre sample from 9 loads (bit-identical, -0.17 ms); P2f coarseSolve as 512 threads with two level-4 cells each (0.865 to 0.22 ms a step: the 1024-thread layout cost about 12 µs a red-black sweep). See §5.1 for what was measured and ruled out |
| P3 per-stage comparison table, brick against pour-m1max on the pour (feeds the brick tuning) | open |

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

## 6. Process

Single agent (memory: single-agent-only). Commit verified milestones with
`tools/safe-commit.sh`; `npm run check:types` and `npm run test:unit` before each.
GPU runs one at a time, short. Each commit that touches shared code also reports
seep 300 / letters 120 drift and step time; pour-only files do not change them.
