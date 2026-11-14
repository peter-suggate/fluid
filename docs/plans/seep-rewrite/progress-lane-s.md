# Lane S progress (solver)

Entries newest last: STARTED, MILESTONE, READY TO MERGE, DECISION NEEDED, COORDINATION.

## 2026-11-14 STARTED
Lane S (solver) begins S1. Design notes as decided (deviations from the plan logged here):
- Vertex level set (matches the observation contract and the fill metric exactly), cell-centre phi = mean of 8 corners for the ghost fluid.
- Dense brick-major storage (8^3 bricks, 64x32x8 table); per-step work over GPU-compacted active-brick lists only (indirect dispatch).
- Pressure: warm-started geometric multigrid V-cycles (sawtooth V(0,2), brick-staged block red-black GS, ping-pong at h),
  aggregation (piecewise-constant) coarse operators recomputed per step from the fine ghost-fluid operator, GPU-gated cycle count
  until L-inf divergence <= 0.05 s^-1 (h band tolerance of the reference), capped, cap hit latched in stats.
- Particles in x >= 240 (throat + right chamber). The throat is inside the band (no depth deletion there) so the jet leaves the throat
  as FLIP particles carrying the throat's volume flux; chamber band deletion at phi < -3h; seeding only where phi(octant centre) < -r.

## 2026-11-14 MILESTONE: solver runs the seep end to end (S1 in progress, not gate-ready)
`lib/methods/seep/` (index, seep-layout, seep-wgsl, seep-kernels, seep-solver; CPU tests in seep-layout.test.ts) and
`tools/seep-solver-smoke.ts`. Imports nothing from lib/methods/uniform. Construction ~1.8 s, 1005 MB allocated.
Decisions changed since STARTED:
- Pressure: PCG with a symmetric V(2,2) aggregation multigrid preconditioner (McAdams 2010 MGPCG): brick-block red-black
  GS smoothers (pre red-first from zero, post black-first), Galerkin coarse operators A_c = P^T A P / 1.8, palindromic
  coarsest sweeps. 9 indirect passes per iteration, GPU-gated by L-inf residual <= 0.05 h (2.5e-4 m/s); 4-6 iterations a step.
  The sawtooth V-cycle with OC 1.8 diverged and with OC 1 stalled at ~0.65 a cycle; flexible PCG on it stalled at 1e-2.
- Particles only in the right chamber (x >= 256), as the reference: the jet leaves the throat as level set. With FLIP in the
  throat, seeds took the Eulerian velocity in place, the throat lost its advection term and ran at 7.7 m/s (Torricelli 3.5).
- Seeds take one step of particle advection before transfer (Ferstl Alg. 1 resamples at the end of a step); without it
  freshly seeded particles froze the exit velocity and the jet piled up at the exit and ran down the wall face.
- Particle surface = union min(phi0, phi_p) (Ferstl Eq. 4 without the h shrink, which the reference applies only in heated
  tiles); the shrunk union lost a sheet's volume every step.
- Solid walls are not interfaces: redistancing ignores sign changes against solid-interior vertices and extends the signed
  distance into them. (Before: throat phi -0.7 everywhere, the wall face seen as surface.)
- Chamber bricks activate two bricks ahead (the jet moves 8.5 cells a step).

Smoke run, 120 steps (check slot; NB-FLIP baseline in brackets):

| step | left L | right L | floor L | throat u m/s | q_left L/s | drift | wet cells | particles |
|---|---|---|---|---|---|---|---|---|
| 60 | 368.395 [368.395] | 0.255 [0.288] | 0.182 [0.147] | 2.50 [2.65] | 0.254 [0.246] | +0.003% [+0.01%] | 7166 [3559] | 146k [13k] |
| 120 | 368.142 | 0.396 | 0.325 | 2.60 | 0.256 | -0.028% | 9485 | 351k |

Fenced median ~39 ms/step (NB-FLIP 33.7): not yet faster. Pressure solve ~45% of it.
Open (S1): wetting and particle count run ahead of the reference (particles seeded in the whole chamber band, 8 a cell);
1200-step gates not yet run. S2 speed work not started.

## 2026-11-14 COORDINATION for lane H
The solver files are committed (this commit): `createSeepSolver(device, scene, {onProgress?, signal?})`, `SEEP_METHOD_ID`,
`observation` (= `observationSource`), `surfaceFieldTexture` 513x257x65 r32float nodal phi in metres (negative liquid,
+PHI_MAX h in solid), `volumeTexture` 512x256x64. You can commit the method registration now. The 0xFFFFFF00 segment count you
saw came from the 7.7 m/s throat runaway (fixed above); SEGMENTS/POOL/DEEP_SURFACE latch in `info.simulationPipelineError`.

## 2026-11-14 MILESTONE S1b: chamber physics to 6/8 gates (1200 steps, check slot)

Supersedes the S1 notes on seeding and the union above. Changes, each a published treatment cited in the kernel comments:
- Surface = NB-FLIP's: nearest-sphere union (radius 0.875h) bounded by the particles' own volume (sampleBound "walls"
  axis rule at solids), with erasure toward the spheres at most h a step (Ferstl 2016 Eq. 4).
- Seeding after redistance (level set and survivors at the same instant): inner band topped up to 8; outer layer only
  where no survivor lies in the 27 cells; inflow liquid (one-step departure point behind x = 256) seeded in every octant.
  Seeds carry the semi-Lagrangian velocity u^n(departure) (Stam 1999). Sampling u^n at the landing point handed the
  jet the stale velocity already there: the exit jet ran at 1.1 m/s, falling at 45 degrees, instead of 2.6 m/s.
- Particle separation (Sato 2018 pairs, constraint-averaged, tangential within h of the surface) and ballistic flight
  for particles with no liquid cell in their 2x2x2 neighbourhood.
- Particle centres keep a quarter cell from solids. On the wall plane the floor film's particles were held at v = 0
  and packed 9-41 per cell into the bottom row, under liquid they no longer described: -1.2% drift by step 1200.
- Redistance freezes every cut-cell corner at its value (Adalsteinsson & Sethian 1995). Re-initialising cut corners
  from axis crossings added 0.2% of the film's volume a step (measured per stage with the tetrahedral fill: SL advection
  -300 cells/step, union +330, redistance now exactly 0).
- The crowding cap (16 a cell) applies only deeper than h; deletions are counted by reason (deep / cap / out).

seep-bench 1200 steps vs NB-FLIP (artifacts/seep-rewrite/lane-s/compare-s-m3.md):

| gate | verdict | detail |
|---|---|---|
| G1 | pass | q_left mean 0.2596 vs 0.2697 L/s (-3.75%) |
| G2 | pass | left loss within -3.9% at 300-1200 |
| G3 | FAIL | right 1.721 / 3.199 / 4.644 / 5.694 L at 300/600/900/1200; upper bounds 1.282 / 2.705 / 4.117 / 5.518 |
| G4 | pass | throat u 2.589 vs 2.639 m/s |
| G5 | pass | 90% wet at step 286 vs 299; 16384 vs 16361 at 1200 |
| G6 | pass | centroid 372.7 / 378.5 vs 368.9 / 369.8; L1 0.177 / 0.155 within limits |
| G7 | pass | drift +0.14% vs NB-FLIP -0.50% |
| G8 | FAIL | jet column 0.0571 vs 0.0770 L (-25.9%, limit 25%) |

G3: the chamber gains ~10 cells a step over delivery during the jet and splash phase (steps 0-360; the union restores
4% more than SL advection removes there), then tracks delivery (q_right ~ q_left). Fenced median 45.8 ms/step
(NB-FLIP 33.7): S2 speed work is next.

## 2026-11-14 MILESTONE S2a: 4h composite reservoir, liquid-only PCG, 45.8 -> 18.1 ms/step

Speed work, every mechanism a published treatment cited in the kernels:
- BULK: 5220 deep reservoir bricks (away from a 3x5x4-brick throat approach) run as a composite h/4h grid (Losasso,
  Gibou & Fedkiw 2004): coarse MAC velocity with RK2 semi-Lagrangian advection, one pressure per 4h cell, conservative
  coarse-fine flux sums over the fine faces, interface coupling at distance 2.5h, and hydrostatically well-balanced
  gravity across the interface. The coarse pressure is the MG level-2 cell, so level 2 on BULK is exact. Fine shadow
  velocities (trilinear in the coarse field) feed backtraces that start in fine cells.
- PCG and MG run over a liquid-brick list (~620 of ~2600 active bricks), built by the operator pass and published to
  indirect args by a direct-dispatch pass. Level 3 is red/black block Gauss-Seidel over level-4 blocks with a
  sparse block list; levels 4-6 stay in the one coarse solve.
- Exact shortcuts: velocity skips faces far in air (US = 0 there regardless), redistance and extension skip tiles
  with nothing to do.
- Level-set advection: one Ralston RK3 step per dt, as the reference's surfaceTraceSubsteps "off".

Step 120 (check slot): fenced median 18.1 ms (from 45.8), mean 7.7 PCG cycles, q 0.253 L/s, throat u 2.58 m/s,
drift 9e-4. Profile: coarse solve 3.1, velocity 3.1, level set (advect/redistance/extend) 3.3, PCG fine passes ~6.
Still above the 11.2 ms 3x target; next up are the coarse solve, velocity, and the level-set passes.

## 2026-11-14 MILESTONE S2b: particles carry the chamber's liquid (conservative to <1%), 5/8 gates

Per the coordinator's ruling, the chamber now uses particles as its conserved carrier. Each mechanism is cited in the kernels:
- **Outlet emission** is the only source of chamber particles. Each step, every throat face at x = 256 emits
  u_x dt h^2 times its wetted fraction, as layers of 2x2 particles h/2 apart (8 per cell volume). The unemitted swept
  length carries to the next step. Particles are placed along the projected face velocity and carry it. The seed
  kernel is gone: nothing seeds from phi.
- **No deletions.** There is no depth retirement and no crowding cap. Deletions are counted by reason, and all of
  them are 0 over 1200 steps.
- **Chamber phi comes from the particles alone.** It starts from their mass plane, NB-FLIP's sampleBound with crowded
  cells counted in full. There is no SL union and no sphere union: the 0.875h sphere cap removed 17% of a 0.6h film.
  Then six Jacobi passes of a mass-conserving level-set fit (van der Pijl et al. 2005; CLSVOF) match each cell's
  tetrahedral fill to its particle mass F = count/8. Before fitting, a crowded cell's excess over full is carried up
  its column. Interior and far-air vertices stay fixed. The throat-exit vertices keep the grid phi.
- **Ledger.** `tools/seep-solver-smoke.ts --ledger` runs sampled steps in stages and prints the chamber tetra-fill
  volume after census, advection, combine, fit, redistance, G2P and emission, beside the particle mass, the emitted
  mass, deletions and the delivered volume.

Ledger, 1200 steps (cells; delivered = left loss - wall change):

| step | delivered | emitted | particle mass | chamber phi | phi vs delivered | total drift |
|---|---|---|---|---|---|---|
| 300 | 10276 | 10365 | 10330 | 10359 | +0.8% | +2.8e-5 |
| 600 | 20581 | 20741 | 20706 | 20718 | +0.7% | +4.6e-5 |
| 900 | 30904 | 31084 | 31049 | 31024 | +0.4% | +4.0e-5 |
| 1200 | 41247 | 41395 | 41361 | 41388 | +0.3% | +4.8e-5 |

seep-bench 1200 steps vs NB-FLIP (artifacts/seep-rewrite/lane-s/compare-s2b.md), check slot:

| gate | verdict | detail |
|---|---|---|
| G1 | pass | q_left 0.2582 vs 0.2697 L/s (-4.3%) |
| G2 | pass | left loss -3.4 / -4.3 / -4.3 / -4.0% |
| G3 | FAIL (300 only) | right 1.295 / 2.573 / 3.883 / 5.165 L; upper bounds 1.282 / 2.705 / 4.117 / 5.518 |
| G4 | pass | throat u 2.588 vs 2.639 |
| G5 | pass | 90% wet at step 334 vs 299 (+12%); 16383 at 1200 |
| G6 | FAIL (1200) | centroid 380.4 / 392.7 vs 368.9 / 369.8; L1 0.178 ok / 0.315 > 0.226 |
| G7 | pass | drift 0.00% at 1200 (and within 5e-5 at every checkpoint) |
| G8 | FAIL | jet column 0.0371 vs 0.0770 L |

Fenced median 17.2 ms (check slot): 1.96x NB-FLIP.

## 2026-11-14 DECISION NEEDED (coordinator, lane H): the remaining fails are set by the reference's own volume errors

- **G8 (jet).** The particles in the jet zone (right, y >= 8, x < 320) carry 0.037 L, and so does phi (the fit matches
  them). Ballistics give the same figure. At 2.6 m/s the jet leaves x < 320 after ~0.12 s. Q/u = 1e-4 m^2 (the 2x2
  throat), and the ~0.36 m path in the zone holds ~0.036 L. NB-FLIP's 0.077 L is about 2.1x that: its 0.875h particle
  spheres, unioned onto a 2h jet, fatten it. bef5feb6 "passed" G8 the same way, while creating liquid. A conservative
  surface cannot match it. Suggest judging G8 on the jet's flux or mass (Q times the zone's flight time), or within 25%
  of the ballistic 0.036 L.
- **G3 at 300.** Our throat delivers 1.285 L by step 300. NB-FLIP delivers 1.221 L, its left loss less 0.11 L parked
  in its wall zone (ours stays at 0.008 L). An exactly conservative chamber would hold 1.2845 L, already above the
  1.282 L upper bound (NB delivered x 1.05). 600-1200 pass.
- **G6 at 1200.** NB-FLIP's chamber holds 3.42 L of the 5.11 L it delivered (-33%). Ours holds 5.17 L of 5.16 L.
  The near-flat profile of a conserved 2.5-cell pool (centroid of a uniform layer: 384) differs from a reference that
  has lost a third of its liquid. The L1 of the normalised profile fails at 1200 only.

Until the gates are re-based, I am continuing S2 speed work on this conservative build. Each change is gated on the
ledger and the 1200-step run.

## 2026-11-14 MILESTONE S2c: 13.3 ms fenced over 1200 steps (2.53x NB-FLIP), gates unchanged at 5/8

These are speed changes on the conservative build (75cab37b), with the physics the same. Each one leaves every value
anything reads unchanged, or only reorders the same reads:

- **Velocity, 2.9 to 1.4 ms.** In the chamber, a face with no particle weight between two all-air cells now gets no
  backtrace. Nothing read those values: the projection reads only faces of liquid cells, and G2P's FLIP difference
  reads the trilinear stencil around a particle. That stencil lies inside the particle's own B-spline P2G support, so
  every face it reads carries weight. Standard FLIP keeps no grid-advected velocity where particles carry the liquid
  (Zhu & Bridson 2005). These traces through the jet's air band were half the kernel. Its level-set reads now come
  from one staged 10^3 vertex tile.
- **advectPhi, 0.72 to 0.39 ms.** Chamber vertices that the particle surface overwrites are no longer traced.
- **velocityBulk, 0.92 to 0.26 ms.** One workgroup per 8^3 coarse block, with a staged coarse-velocity tile, using the
  same interpolation.
- **redistInit, 0.52 to 0.13 ms.** The 26-neighbour sign test reads a staged sign tile, instead of 27 scattered loads
  and 216 cell-open tests per vertex.
- **Level-2 multigrid.** mgDown1 (BULK), mgDown2 and mgUp2 pack 8 bricks into a 64-thread workgroup, instead of one
  8-thread workgroup per brick.
- **Smaller fixes.** PCG dot sums load their 64 partial slots in parallel. coarseSolve clears its set with 192
  threads. BULK membership is computed in closed form (the bake equals bulkBrick, as the layout test pins).
- **Mass fit and emission.** The mass fit skips idle bricks and emission runs in parallel (both uncommitted after
  75cab37b).

Ledger, 1200 steps (cells): chamber phi vs delivered is +0.6 / +0.5 / +0.75 / +0.5% at 300 / 600 / 900 / 1200.
Total drift is +5e-5, with no deletions.

seep-bench 1200 vs NB-FLIP (artifacts/seep-rewrite/lane-s/compare-s2c.md), check slot:

| gate | verdict | detail |
|---|---|---|
| G1 | pass | q_left -4.24% |
| G2 | pass | -3.4 / -4.3 / -4.3 / -4.0% |
| G3 | FAIL (300 only) | 1.291 vs bound 1.282 L (reference-bound; see DECISION NEEDED) |
| G4 | pass | throat u 2.588 vs 2.639 |
| G5 | pass | 90% wet at 336 vs 299 |
| G6 | FAIL (1200) | centroid 386.5 vs 369.8 (was 392.7); L1 0.230 vs limit 0.226 (was 0.315) |
| G7 | pass | drift 0.01% |
| G8 | FAIL | jet 0.0367 vs 0.0770 L (ballistic ~0.036 L) |

Fenced median 13.33 ms over the 1200-step check run (thirds 12.0 / 13.5 / 14.2 ms). In the s2b run it was 17.2 ms.
The 3x target (11.2 ms) is still ahead. What is left: PCG at ~7.4 cycles (~0.6 ms per cycle), extension (0.98 ms),
redistance sweeps, and the mass fit.

## 2026-11-14 MILESTONE S2d: 12.2 ms fenced over 1200 steps (2.76x NB-FLIP, gates 5/8 unchanged)

Gate-neutral speed work on top of acfee28b. Each change keeps the arithmetic the same.

- **Reductions.** pcgUpdate, pcgPrecond, pcgDirection and the velocity kernel's minphi/far reduction use a
  two-level fold (2 barriers, a conflict-free stride) instead of 512-thread trees.
- **Extension.** One barrier per sweep for all three components, with UNK/SHUT sentinels in the tile, instead of
  per-component passes. The semantics are unchanged.
- **Velocity.** The face loop runs in two passes. The first handles zero/transfer faces and counts the traces. The
  velocity tile is staged only when a brick has 96 or more traced faces, and sparse bricks read global memory.
- **Particles.** Each pool stores positions contiguously, with velocities at +CAP. separate, count and scatter
  then stream half the bytes. redistSweep sets its mixed flag once.
- `tools/seep-solver-smoke.ts --profile-steps=K` averages K profiled steps (the timestamps are coarse).

Ledger, 1200 steps (cells): chamber phi vs delivered is -0.5 / +0.2 / +0.3 / +0.6% at 300 / 600 / 900 / 1200.
Total drift is +7e-5, with no deletions.

seep-bench 1200 vs NB-FLIP (artifacts/seep-rewrite/lane-s/compare-s2d.md), check slot. The verdicts match s2c:
G1, G2, G4, G5 and G7 pass. G3 fails at 300 only (1.297 vs bound 1.282 L). G6 fails (1200 centroid 387.8 vs
369.8, L1 0.241 vs limit 0.226). G8 fails (jet 0.0370 vs 0.0770 L).

Fenced median 12.21 ms (thirds 11.15 / 12.39 / 13.11, p95 14.05) against NB-FLIP's 33.69 ms. The 3x target is
11.2 ms.

The plateau is in the pressure solve: about 7.5 cycles at ~0.8 ms each. A per-region residual probe (not
committed) shows every region below TOL by cycle 3-4 except chamber surface cells. Those are ghost-fluid rows with
small theta and diagonals up to 1/THETA_MIN, which converge about 3x per cycle. Three experiments did not pay:
- Capping the Dirichlet part of the coarse diagonal made no difference.
- More fine sweeps (8 instead of 4) cut 0.6 cycles but saved only ~0.1 ms.
- THETA_MIN 0.05 cut 1 cycle but changes the discretisation (the plan fixes 1e-3).

## 2026-11-14 MILESTONE S2e: 11.43 ms fenced over 1200 steps (2.95x NB-FLIP, gates 5/8 unchanged)

- **Shadow, 0.25 to 0.02 ms.** Each shadow brick's samples read 4^3 coarse faces per component. These are now
  staged once (192 cface calls instead of 12k) and interpolated in the same order.
- **Converged coarseSolve no longer launches.** coarseSolve k (k >= 1) is indirect. Its argument is written by
  coarseSolve k - 1 into alternating buffers (cgate0/1), because a dispatch may not read indirect arguments from
  a buffer it binds writable. On convergence the solve closes every later gate at once. An idle coarseSolve
  cost about 50 us, so this saves ~0.25 ms a step.
- **Preconditioner settings.** These change the convergence path only; the solution meets the same tolerance.
  - Aggregation over-correction 1.8 -> 1.6: 7.6 -> 6.7 cycles a step over 1200 steps. 1.4, 1.5 and 2.1 were
    measured too.
  - Fine smoother 2 -> 3 red-black sweeps: 6.8 -> 6.3 cycles. Net -0.15 ms in paired runs.
- **Measured: an empty gated PCG iteration costs ~0.09 ms** (14 zero-workgroup dispatches at ~6 us each).
  MAX_CYCLES 24 instead of 12 costs +1.05 ms. With ~6.4 cycles a step, the unused tail is still ~0.5 ms.
  Sampling every 10 steps over 1200 saw at most 9-10 cycles and no cap hits, so 12 stays.

Ledger, 1200 steps (cells): chamber phi vs delivered is +0.7 / 0.0 / +0.6 / +0.7% at 300 / 600 / 900 / 1200.
Mass (count / 8) is within 0.3% of delivered. There are no deletions, and G7 drift is 0.01%.

seep-bench 1200 vs NB-FLIP (artifacts/seep-rewrite/lane-s/compare-s2e.md). The verdicts are the same as s2c/s2d:
G1, G2, G4, G5 and G7 pass. G3 fails at 300 only (1.291 vs 1.282). G8 fails (jet 0.0370 vs 0.0770).
G6 fails as before, with 1200 centroid 393.7 and L1 0.305. Run to run, G6's 1200 values have been 392.7/0.315
(s2b), 386.5/0.230 (s2c), 387.8/0.241 (s2d) and 393.7/0.305 (s2e). This is the spread of the late film, not a
trend from this change.

Fenced median 11.43 ms (thirds 10.60 / 11.70 / 11.93, p95 12.60) against NB-FLIP's 33.69 ms. The 3x target is
11.23 ms.

## 2026-11-14 MILESTONE S2f: 10.93 ms fenced over 1200 steps (3.08x NB-FLIP; gates 5/8 unchanged)

- **The next step is encoded ahead.** Encoding and submitting the ~200 dispatches cost 0.70 ms of CPU a step,
  measured with a temporary timer around advanceTo. This came before the GPU could start and was serial with it.
  A step's commands depend only on its parity: the records are fixed, and every data-dependent count is an
  indirect argument. So advanceTo now submits a command buffer built during the previous step, and then encodes
  the next one while the GPU runs. If another path advances the step (profileStep or the ledger) and the parity
  no longer matches, the buffer is rebuilt. The GPU still runs one fenced step at a time; only the CPU encoding
  overlaps it. Paired 1200-step runs measured 11.49 / 11.49 ms before and 10.91 / 10.94 ms after.

Ledger, 1200 steps (cells): chamber phi vs delivered is +0.6 / +0.5 / 0.0 / +0.6%. There are no deletions.

seep-bench 1200 vs NB-FLIP (artifacts/seep-rewrite/lane-s/compare-s2f.md). The verdicts are unchanged:
G1, G2, G4, G5 and G7 pass. G3 fails at 300 only (1.290 vs 1.282). G8 fails (jet 0.0371 vs 0.0770).
G6 fails, with 1200 centroid 389.9 and L1 0.262, inside the s2b-s2e spread.

Fenced median 10.93 ms (thirds 10.12 / 11.20 / 11.47, p95 12.14) against NB-FLIP's 33.69 ms: 3.08x, under the
11.23 ms target. The stretch target is 6.7 ms.

Measured and not kept:
- Restricting red L3 blocks inside coarseSolve to drop mgRestrict3 (+0.1 ms; single-workgroup work is dear).
- Subgroup-shuffle coarsest sweeps (+0.2 ms, unexplained).
- Staging separate's neighbours in workgroup memory (no gain; the loop is not load-bound).
- Fewer coarsest sweeps (more cycles). More coarsest sweeps (41 is saturated).

## 2026-11-14 MILESTONE S2g: 10.65 ms fenced over 1200 steps (3.16x NB-FLIP; gates 5/8 unchanged)

- **Level-3 smoothing is block Jacobi.** Before, two colours of level-4 blocks ran multiplicatively, with a red
  dispatch and a black dispatch each way. Now every block smooths at once, as additive Schwarz.
  - Eight blocks share a 64-thread workgroup, and the dispatch count is n4 / 8.
  - Pre-smoothing from zero has a zero halo, so the blocks are independent. Post-smoothing reads each
    neighbour's pre-smoothed plus prolonged value, so the cycle stays symmetric.
  - Cycles rose 6.26 -> 6.35 a step, and the cycle lost two dispatches each way.
  - Paired 1200-step runs: 10.92 -> 10.69 ms.
- **D2 is fused into D3.** The level-2 residual restricted to level 3 is now formed per level-4 block, in the
  same workgroup that pre-smooths that block. This drops one dispatch per PCG cycle.
  - The indirect count is a new slot, ARG_L4U = n4. The coarseSolve gate zeroes it with the others once
    converged.
  - Same 300-step state: 6.62 vs 6.67 cycles, and 9.76 vs 9.94 ms.
- Paired 1200-step runs of the batch against S2f: 10.87 / 10.93 ms (A) vs 10.48 / 10.60 ms (B).

Ledger, 1200 steps (cells): chamber phi vs delivered is +0.1 / +0.1 / +0.6 / +0.5%. There are no deletions.

seep-bench 1200 vs NB-FLIP (artifacts/seep-rewrite/lane-s/compare-s2g.md). The verdicts are unchanged:
G1, G2, G4, G5 and G7 pass. G3 fails at 300 only (1.290 vs 1.282). G8 fails (jet 0.0371 vs 0.0770).
G6 fails, with 1200 centroid 392.3 and L1 0.285, inside the s2b-s2f spread.

Fenced median 10.65 ms (thirds 9.77 / 10.85 / 11.10, p95 11.87) against NB-FLIP's 33.69 ms: 3.16x.

## 2026-11-14 Jet self-check (coordinator item 1): the particles and phi agree, and the jet is ballistic

**Method.** A check-slot run of the S2g build, using tools/seep-observe-seep.ts plus a particle readback (script
kept at ~/code/me/fluid-scratch/lane-s/jetcheck.mts). It compares two volumes in the G8 zone (right chamber,
x 256-319, y >= 8):
- the particle-carried liquid (count / 8 cells);
- the harness's phi-fill volume (jet_L).

| step | harness phi jet_L | particles | phiC fill (approx.) | mean particle u_x | throat u |
|---|---|---|---|---|---|
| 300 | 0.0370 L | 0.0380 L | 0.0379 L | 2.13 m/s | 2.60 m/s |
| 600 | 0.0370 L | 0.0377 L | 0.0380 L | 2.12 m/s | 2.59 m/s |

**The volumes agree slab by slab.** Each 8-cell x slab holds 34-42 particle cells against 35-41 phiC cells.
The thin-jet reconstruction is not losing volume locally.

**The jet is ballistic.** Down the zone, the mean u_y falls by 0.18 m/s per 8 cells (0.04 m).
- At u_x ~ 2.1 m/s, the flight time per slab is 0.019 s, and g dt = 0.187 m/s, which matches.
- The jet centre falls from y 64.6 to 40.0.
- u_x is 1.98 m/s at x 256 and rises to 2.21 m/s by x 319. It is below the throat u because the throat value is
  measured inside the wall.

**The jet volume is what its flux implies.** For a steady jet, the volume per length is q / u_x =
0.258e-3 / 2.1 = 1.23e-4 m^3/m, or 39 cells per 8-cell slab, as measured. So the jet holds about 0.037 L.
NB-FLIP's 0.077 L at a similar q (0.270 L/s) implies its liquid moves at about half the speed through the zone,
or takes a longer path, which is consistent with lane H's slowing/thickening hypothesis. No Seep defect was
found, so item 2 (speed) continues.

## 2026-11-14 Stretch speed (item 2): measurements so far, nothing landed

- **Fine smoother passes**, 300 steps on the check slot (FINE_PASSES env experiment, reverted):
  - 6 passes: 6.56 cycles
  - 10 passes: 6.29 cycles
  - 16 passes: 6.15 cycles

  The fine level does not limit convergence. The ~3 tail cycles a step (L-inf going from ~1e-3 to 2.5e-4)
  come from small-theta ghost rows near the jet impact and the far-wall films (diag 200-1300, x 440-506,
  y 0-3).
- **Gated empty cycles still cost ~0.5 ms a step** (12 - 6.4 cycles x 11 zero-workgroup dispatches).
- **Fusions examined and not taken:**
  - redistInit into redistSweep.1: one more phi buffer is needed to avoid the PHIT read/write race, for
    ~0.1 ms.
  - Multi-iteration massFit in one tile: either the frozen halo over-corrects at brick-boundary cells, or an
    exact 2-halo tile changes the column carry.
  - Pipelined or Chronopoulos-Gear CG to merge pcgPrecond with pcgDirection: z on the halo is not available
    inside the dispatch.
