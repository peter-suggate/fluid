# Seep rewrite: an independent, M1 Max–specialised solver for `reservoir-seep` (2026-11-14)

Goal (user, 2026-11-14): rewrite the fluid solver from scratch as a fully 3D, drop-in,
experimental replacement for Narrow-band FLIP, supporting only the default
`reservoir-seep` scene. Optimise every stage for the M1 Max, sacrificing generality
and baking static geometry and structure. Compute the fluid online; no trajectory
baking or playback. The rewrite must be independent of NB-FLIP: its own state, kernels
and execution path.

**Success requires both of the following, demonstrated reproducibly:**
1. The new solver matches NB-FLIP's physical results within measured numerical
   tolerance (the gates in §4), using full-scene comparisons.
2. It runs substantially faster on the M1 Max, measured with a fair benchmark using
   the same harness, loop and device for both solvers (§5).

A complete but slower, or physically divergent, implementation is not done.

Lanes:
- **S (solver):** `lib/methods/seep/`. Progress in `seep-rewrite/progress-lane-s.md`.
- **H (harness and integration):** `tools/seep-*.ts`, method registration and rendering.
  Progress in `seep-rewrite/progress-lane-h.md`.
- **Coordinator (Claude):** plans, judges, runs the combined gate rounds, and keeps §0 current.

## 0. Status board

| item | state |
|---|---|
| NB-FLIP baseline, 1200 steps (`artifacts/seep-rewrite/baseline-table.txt`) | done, HEAD 095d2518 |
| H1: common observer and bench harness | done (7fdf4bac..358ea7b5); NB refs in artifacts/seep-rewrite/nbflip-ref |
| S1: correctness milestone | partial: particle-carried chamber (75cab37b) conserves (ledger within 0.6%, drift +0.01% at 1200) |
| S2: speed milestone | e20a907a: 10.65 ms fenced, 3.16x on lane S's check-slot run; lane H independent timing-slot check pending |
| Gates at e20a907a (lane S run) | G1 G2 G4 G5 G7 pass; G3 fails at step 300 only (1.290 vs 1.282); G6 centroid 392 vs 370; G8 jet 0.037 vs 0.077 L |
| Measurement audit (lane H) | in progress: NB wall zone counts solid ghost phi (0.140 to 0.248 L, throat holds 0.008 L); NB jet vs ballistic trajectory |
| Combined gate round | todo |

## 1. The reference: NB-FLIP on the seep, as measured

Scene (lib/core/reservoir-seep-scene.ts):
- The lattice is 512×256×64 at h = 5 mm. dt = 1/60 s, fixed: no substeps, ever.
- The solid wall spans x∈[240,256). The throat is 2×2 cells at y∈[64,66), z∈[32,34).
- Liquid starts at x<240, y<192, plus the wetted throat box. Its volume is 368.8 L.
- Walls are free-slip; the top is open. The fluid is inviscid, with no surface tension.

Baseline run (`probe-tier-budget.ts nbflip-base-1200 --scene=reservoir-seep --steps=1200 --every=60`):

| step | left L | right L | floor band L | throat u m/s | q_left L/s | drift | wet floor cells | particles |
|---|---|---|---|---|---|---|---|---|
| 60 | 368.395 | 0.288 | 0.147 | 2.65 | 0.246 | +0.0001 | 3559 | 13k |
| 300 | 367.315 | 1.158 | 0.998 | 2.64 | 0.271 | −0.0005 | 15147 | 70k |
| 600 | 365.954 | 2.084 | 1.920 | 2.64 | 0.271 | −0.0016 | 16034 | 129k |
| 900 | 364.608 | 2.874 | 2.722 | 2.64 | 0.268 | −0.0031 | 16228 | 183k |
| 1200 | 363.268 | 3.361 | 3.213 | 2.56 | 0.267 | −0.0055 | 16378 | 216k |

Reading:
- **Discharge is the robust physical result.** q_left is 0.27 L/s, against a
  Torricelli flow of about 0.35 L/s, so Cd ≈ 0.77. The left tank loses 5.37 L in 20 s.
- **The right tank lags delivery.** It holds 3.36 L where 5.37 L was delivered. This
  is NB-FLIP's own volume loss (drift −0.55%, about −2 L), mostly in the calm film
  about 1.6 h deep that covers the whole right floor from about step 360.
  (docs/plans/nbflip-seep-volume-2026-10-27.md attributes it to 4h coarsening plus
  heat retirement.)
- **Step time** rises with particles: 28→41 ms with a readback every 60 steps; the
  median is 35 ms. A clean run with readbacks every 10 steps measures about 31 ms
  (docs/plans/brick-resident-step-2026-11-08.md).

Effective numerics (survey, 2026-11-14; file citations in the lane-S brief):
- MAC grid. Ghost-fluid Dirichlet free surface with θ = φ_l/(φ_l−φ_a), clamped to
  [1e-3, ∞). Grid-aligned voxel walls are pure Neumann (aperture 0 or 1).
- Vertex level set, advected by RK3 backtrace with Catmull-Rom interpolation.
  Redistanced to exact distance near the zero set.
- Semi-Lagrangian momentum (RK2 segments of at most 1 cell, trilinear), extended
  into air 2+2 sweeps.
- FLIP particles (95% FLIP, quadratic B-spline P2G, Heun RK3 advection, 8 per cell,
  band 3h) only in "hot" tiles, in practice the right chamber's film and splash.
  The jet leaves the throat as level set only.
- Reservoir quiet water is coarsened to 4h and 16h (pressure and velocity). The
  throat ±4 cells is pinned at h. The left tank has no h tiles after step 60.
- Liquid volume is the six-tetrahedron fill of the 8 corner φ, summed per cell
  (lib/core/level-set-fill.wgsl.ts). This is the drift metric.
- Volume control rate 0.1 s (`volumeControlSeconds`).
- Effective problem size: about 130 h tiles at the throat plus about 1900 in the right
  chamber (4³ cells each), plus about 42k coarse owners. NB-FLIP's 31–35 ms is
  overwhelmingly machinery: about 300 kernels a step, adaptivity and residency.

## 2. Design of the new solver (lane S)

Name: **Seep solver** (`lib/methods/seep/`, method id `seep-m1max`, experimental).
It imports nothing from `lib/methods/uniform/`. It may import shared, solver-neutral
infrastructure: the scene model, `level-set-fill.wgsl.ts` for the volume metric, the
GPU compilation manager, and the device limits.

### 2.1 Baked static structure (construction time, once)
- The lattice is the seep box only. Construction fails explicitly for any other scene
  id, or for any change to the seep's geometry the bake does not cover. Unknown
  kinds fail.
- Solid geometry is a baked per-face aperture of exactly 0 or 1: box walls, the
  middle wall and the throat. The open top is a Dirichlet p = 0 row above the domain.
- **Brick layout:** 8³ cells, brick-major storage (a brick's 512 cells are
  contiguous), and a dense brick table of 64×32×8 = 16384 entries. Bricks entirely
  inside the solid wall are never allocated work.
- **Static region classes**, baked from the scene and asserted every step on the GPU
  (a violation latches a failure, never a silent clamp):
  - **DEEP:** reservoir bricks that can never meet the free surface. They sit more
    than k cells below the lowest level the reservoir can reach in the scene's 20 s,
    with k chosen from the measured level drop of 15 mm (3 h) plus margin. DEEP bricks
    skip level-set advection, redistance and particle work. They still advect
    velocity and take part in pressure.
  - **THROAT:** the throat and its approach. Always fine.
  - **CHAMBER:** the right chamber. This is the only region that can hold particles.
- Pools and capacities are sized up front from the measured peak plus margin. No
  runtime growth, no compilation at simulation time (memory: no-runtime-pool-growth).
- Dense storage over the seep box is accepted for this experimental, single-scene
  solver (about 34 MB per scalar field at f32). Step time outranks memory. **Work**
  must still follow active bricks, never the box.

### 2.2 Step (one per 1/60 s frame; everything on the GPU, no per-step readback)
1. **Active set:** compact active bricks (liquid, or within the band of liquid) into
   lists for indirect dispatch.
2. **Particles (CHAMBER band only):** Heun RK3 through the extended grid velocity.
   Exact collision with the baked axis-aligned solids.
3. **Level set:** semi-Lagrangian advection of φ, RK3 backtrace with cubic
   interpolation, on non-DEEP active bricks. In the band, combine with the particle
   surface (Ferstl 2016 Eq. 4, or a mass-based bound), then redistance near the zero
   set.
4. **Momentum:** semi-Lagrangian advection of the MAC velocity. Where particles are
   present, P2G (quadratic B-spline), blended as in Ferstl 2016 §3.
5. **Gravity** on liquid faces.
6. **Pressure:** ghost-fluid free surface (θ clamp 1e-3, as the reference), Neumann
   apertures. MGPCG (McAdams 2010) or a warm-started multigrid. Convergence is held
   to a stated tolerance, at least as tight as NB-FLIP's (h band 0.05 s⁻¹ L∞ on the
   divergence).
7. **Project,** then extend velocity into air (enough cells for the next step's
   backtrace and particle advection at CFL ≈ 12 in the jet).
8. **G2P:** FLIP 0.95 / PIC 0.05. Band resampling: seed to 8 per cell in band cells,
   delete particles deeper than the band. Isolated particles are real liquid and are
   never deleted for being isolated.

### 2.3 M1 Max specialisation (S2 and onwards)
- Minimise dispatches. Launches cost 10–34 µs each on this stack (paddleboats
  profile). Target well under 60 dispatches a step.
- Use threadgroup memory: a brick plus halo staged once, with several smoothing
  sweeps per launch.
- Fill the GPU: no single-workgroup serial passes on the hot path.
- Size workgroups for 32-wide SIMD groups. Use subgroup reductions for the PCG dot
  products.
- f16 storage where the measured accuracy allows.
- **Baked structure beyond §2.1:**
  - static coarse cells for the DEEP reservoir, a composite h/4h with conservative
    flux sums at the baked interface, if the uniform-h bulk dominates;
  - precomputed multigrid hierarchy and operators for the static solids;
  - DEEP bricks' level set never touched.

  Every such change must keep the §4 gates.

### 2.4 What is not allowed
- No trajectory baking or playback.
- No calls into NB-FLIP.
- No hidden volume injection or deletion beyond the paper rules.
- No substeps, and no smaller dt.
- No speed clamps or damping to hide errors.
- No CPU stages per step.

## 3. Harness and integration (lane H)

- **`tools/seep-bench.ts --method=<nbflip|seep>`:** constructs either solver through
  the method registry on the same `sceneDocument("reservoir-seep")`, on the same
  Dawn device setup, in the same loop.
  - **Timing:** fenced per-step wall time (advanceTo plus awaitFrameCompletion), with
    no readbacks in the timed steps (sample steps are excluded and reported apart).
  - **Throughput:** 1200 steps submitted and fenced at the end, total divided by 1200.
  - **Report:** median, p95, mean, peak GPU memory, construction time.
- **One observable schema for both solvers**, sampled every N steps. For NB-FLIP it
  comes from the probe-tier-budget zone measurement, extended; for the new solver,
  from its own fields. Both use the same fill formula and zone definitions:
  - `liquid_L`: total, left (x<240), wall (240..256), right (≥256), rightFloorBand (y<8);
  - reservoir level (left / footprint) and head;
  - q_left and q_right over the window;
  - throat middle-cell +x face u;
  - wet floor cells (right, y=0, fill > 1%);
  - the right chamber's liquid per 16-cell x bin (16 bins), and its x centroid;
  - the jet column volume (right, above the floor band, within 64 cells of the wall);
  - particle count;
  - drift against the t = 0 volume.
- **`tools/seep-compare.ts`:** reads two or more runs and prints the §4 verdict table
  and a markdown report.
- **Method registration:** `seep-m1max` appears in the method picker, accepts only
  `reservoir-seep` (refuses others explicitly) and renders through the existing water
  renderer: a dense `volumeTexture`, plus a `surfaceFieldTexture` if the renderer
  needs it.

## 4. Physical acceptance gates (full scene, 1200 steps = 20 s)

The tolerance is measured, not invented:
- NB-FLIP is nondeterministic, so it runs twice. Its repeat spread S(x) is the noise
  floor for every observable.
- Where NB-FLIP's own conservation error dominates an observable (the right tank),
  the tolerance band includes that measured error.

| gate | observable | pass if |
|---|---|---|
| G1 | q_left, per 60-step window, steps 120–1200 | mean within ±5% of NB-FLIP's, and every window within ±10% or 3·S |
| G2 | left tank volume (reservoir level) at 300/600/900/1200 | left loss within ±5% of NB-FLIP's loss |
| G3 | right tank volume at 300/600/900/1200 | within [NB_right − max(5%, 3·S), NB_delivered + 5%], where NB_delivered = NB left loss − NB wall change: the band between NB-FLIP's result and NB-FLIP without its measured volume loss |
| G4 | throat u, median over steps 120–1200 | within ±10% |
| G5 | right floor wetting: step at which wet cells reach 90% of the floor; wet cells at 1200 | within ±25% of NB-FLIP's step; final within 5% |
| G6 | right-chamber x profile (16 bins) and centroid, at 600 and 1200 | centroid within ±16 cells; normalised L1 profile difference within 3× NB-FLIP's repeat L1 (floor 0.15) |
| G7 | total drift at 1200 | \|drift\| ≤ NB-FLIP's \|drift\| (0.55%) |
| G8 | jet column volume, median over steps 120–1200 | within ±25% |

Every report carries the full table at both checkpoints, plus step time. A gate
loosened after the fact is a failure, not a pass.

## 5. Performance acceptance

- Fair benchmark: the same `seep-bench.ts`, the same device setup, a timing slot
  (exclusive GPU, `tools/gpu/fluid-gpu-run.sh timing`), and 1200 steps for each
  solver, back to back.
- **Pass:** clean median step time ≥3× faster than NB-FLIP, with the pipelined
  throughput also ≥3× faster. **Stretch:** ≥5×.
- Also reported: p95, construction time (cold and cached shaders), and peak GPU memory.

## 6. Process

- Lanes edit only their own files. Commit verified milestones on main, with exact
  file lists and the private-index guard (memory: temp-index-commit-guard).
- Before every commit: `npm run check:types` and `npm run test:unit`. Never with
  `WEBGPU_NODE_MODULE` exported.
- GPU runs go only through `tools/gpu/fluid-gpu-run.sh` (check, or timing for
  benchmarks). rc 76, 77 or 78 means an invalid run, not a regression.
- Implement, then test. Keep GPU runs short (60–300 steps) while developing. Full
  1200-step comparisons only at milestones.
