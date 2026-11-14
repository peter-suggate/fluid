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

## 2026-11-14 MILESTONE S2h: jet trajectory fixed at the outlet (emission velocity); 7/8 gates as written, G8 passes as amended

**The defect was in emission.** Instrumenting mean particle u and v per 8-cell slice of the jet showed:
- Along the flight, the slice-to-slice change was already (0, -g dt): u_y fell by 0.18 m/s per slice at
  u_x 2.1 m/s. So G2P, P2G, pressure and separate were not changing the flight.
- The jet *left* the outlet wrong: u_x 1.98 m/s against the throat face's 2.59 m/s, and u_y about -0.1 m/s.

emit gave each new particle velU(p0), a trilinear sample on the outlet plane x = 256. This has two effects:
- **u_x is diluted by the wall.** Across the 2x2 throat, half the samples took a 25% weight from the closed
  wall faces next to it (u = 0) on each axis. The mean weight on throat faces is 0.875^2 = 0.766, and
  2.59 x 0.766 = 1.98 m/s, which is the measured exit speed.
- **u_y is fed back from the jet.** The v and w samples took half their weight from the first chamber
  column, where the jet is already falling.

**Fix (emit only).** A particle crossing outlet face (256, fy, fz) now takes:
- the face's own MAC normal velocity (Harlow & Welch 1965);
- the tangential velocity of the upstream throat cell it leaves (donor cell; Courant, Isaacson & Rees 1952).

This is the same velocity whose u_x already set the swept volume and the placement, so the emitted mass is
unchanged. There are no clamps and no jet special case.

**After the fix** (300 steps, check slot): u_x is 2.55-2.61 m/s along the whole flight and u_y starts at -0.03 m/s.
The mean height per slice is within 1-5 cells of the ballistic parabola.

Audit (artifacts/seep-rewrite/lane-s/audit-s2h/audit.md; tools at HEAD e5239022, because the working-tree
observer was mid-edit by lane H):

| | NB-FLIP | Seep S2g | Seep S2h |
|---|---|---|---|
| impact x, step 300 / 600 | 379.8 / 379.2 | 354.1 / 353.7 | 375.8 / 375.3 |
| RMS mean-y difference vs NB, slices in [256, 352), step 300 / 600 | - | ~9 | 1.43 / 1.19 cells |

The amended G8 limits are impact within 8 cells and RMS within 3 cells.

**Far-wall pile, depth at x 448-512:**
- step 1200: 2.3-3.1 cells, against NB 1.0-1.8 (S2g: 3.7-4.4);
- step 600: 0.85-2.2 cells, against NB 0.68-1.69.

Ledger, 1200 steps (cells): chamber phi vs delivered is +0.0 / +0.4 / +0.5 / +0.4%. There are no deletions.
Drift is at most 6e-5.

seep-bench 1200 vs NB-FLIP, check slot (artifacts/seep-rewrite/lane-s/compare-s2h.md):
- **G1:** pass, q_left -3.91% (was -4.24%).
- **G2:** pass, all checkpoints within 4%.
- **G3:** pass even as written; 300: 1.281 vs bound 1.282.
- **G4:** pass, 2.597 vs 2.639 m/s.
- **G5:** pass, 90% wet at step 277 vs 299.
- **G6:** pass, now inside the limits. 600: centroid 370.6 vs 368.9, L1 0.092. 1200: centroid 380.1 vs 369.8,
  L1 0.178 (limit 0.226).
- **G7:** pass, drift 0.01%.
- **G8 as written** (jet column volume, now report-only): 0.0331 vs 0.0770 L.

Fenced median 10.74 ms on the check slot (thirds 9.96 / 10.91 / 11.34, p95 12.11), with lane H sharing the GPU.

## 2026-11-14 (late): G6 far-wall investigation, jet impact offset, and the S2i speed batch (not landed)

### G6: Seep conserves; NB-FLIP's lost 1.74 L is the gap

All numbers are at step 1200, from five Seep runs and NB-FLIP check-c/d. The Seep runs are: s2h (mine), lane H's check,
timing-fenced and timing-pipelined runs of bbc7828d, and s2i (below). Script: g6prof.mjs in lane S's scratch.

**Particles vs phi:** they agree per 16-cell bin. There is no near-wall reconstruction excess and no particle stacking.
Phi in the chamber matches delivered volume within 0.5%.

**Film momentum:** the pool at 1200 is close to rest, with |u_x| < 0.14 m/s in the film away from the impact. The far
bins show no bore at the sample.

**Volume:** Seep's right chamber holds 5.19-5.21 L, against the 5.17 L it delivered. NB's holds 3.39 L, against the
5.37 L it delivered. NB's total drift is -1.71 / -1.77 L, which is the whole right-chamber gap (1.82 L). NB's deficit
is far-weighted. Per bin, Seep minus NB is 0.05-0.10 L at x < 400 and 0.13-0.24 L at x >= 400. NB's far bins are its
lowest (0.13-0.15 L, against 0.24-0.27 L near), even though the jet drives liquid toward the far wall.

| run | centroid | L1 vs NB | L1 vs NB + lost 1.74 L, uniform | L1 vs NB + lost 1.74 L, levelled |
|---|---|---|---|---|
| s2h | 380.0 | 0.174 | 0.117 | 0.112 |
| H check | 385.8 | **0.227** | 0.160 | 0.148 |
| H timing-fenced | 380.9 | 0.170 | 0.103 | 0.118 |
| H timing-pipelined | 386.5 | 0.211 | 0.139 | 0.098 |
| s2i | 388.2 | **0.272** | 0.205 | 0.159 |
| NB (binned) | 370.4 | - | centroid 375.0 | centroid 384.0 |

The limit is 0.214 (3 x NB repeat 0.071). "Levelled" water-fills NB's lowest bins with its lost volume, which is what a
conserving pool would do with it.

With NB's lost liquid restored, every Seep run passes both L1 and centroid: L1 0.10-0.16 and centroid within 5 cells of
384. Against NB as measured, the systematic part alone, the mean Seep profile vs NB, is L1 0.196 of the 0.214 limit.

Seep's own run-to-run L1 is 0.10-0.14, against NB's 0.07. The pool is deeper (2.5 vs 1.7 cells), so sloshing is less
damped and its phase at the 1200 sample varies. Seep's centroid has sat 8-16 cells beyond NB's at every sample since
step 660. As a result, G6 at 1200 passes or fails on sample noise: 3 of 5 runs pass L1, and s2i also fails the centroid
(17.8 > 16).

I found no Seep defect behind the far-wall share. The gap is NB-FLIP's film loss plus Seep's larger, less damped pool.
G6 as written cannot pass reliably for a conserving solver. This needs the coordinator's ruling. I have not changed G6.

### Jet impact 4-5 cells short: decomposed, no defect specific to Seep

Audit of s2h and s2i, steps 300 and 600. The jet core stays thin (the separate step's displacement is 0 there), and
its volume matches the ballistic column exactly (0.0331 vs 0.0320 L).

**Throat speed:** u0 is 2.60-2.61 m/s, against NB's 2.64-2.65 (G4 -1.6%). Ballistic impact moves from 383.9 to 381.3,
**2.6 cells**.

**Sag below ballistic:** it grows linearly with x in both solvers, so it acts as a constant downward velocity.
- Seep: 5.1 cells over 112 cells, about 0.12 m/s.
- NB: 3.8 cells, about 0.09 m/s.
- Most of this is the first-order time error of advecting supported particles with u^{n+1} (symplectic Euler: ½ g dt =
  0.082 m/s). NB's coupled samples share it.
- Seep's extra 0.03 m/s is the outlet cell's vertical face velocity carried at emission, worth about **1.3 cells**.

The two parts sum to the 4 cells measured (375.5 vs 379.2). G8 passes (±8).

Removing the ½ g dt error would need trapezoidal advection with the pre-force velocity. That averages a field that is
not divergence-free into particle motion everywhere, and it would move impact and centroid further out, making G6 worse.
Not done.

### S2i speed batch: Jacobi initial guess + exact coarsest solve (gated, NOT landed)

Two changes:
- **pcgGuess:** one Jacobi sweep from the warm start before PCG (Saad 2003 §4.1). The bulk is copied.
- **Exact coarsest solve:** the L6 32x32 operator is inverted once per step in coarsenTop by Gauss-Jordan (SPD, no
  pivoting, FAIL_PIVOT on a non-positive pivot). coarseSolve applies it with one mat-vec in place of 41 red-black
  sweeps (Trottenberg et al. 2001; Golub & Van Loan 2013 §4.2).

The tolerance is unchanged.

**A/B, 600 steps, check slot, interleaved x3** (HEAD from a scratch tree via SEEP_IMPL; nothing written to
lib/methods/seep):

| | HEAD | S2i | change |
|---|---|---|---|
| fenced ms | 10.14-10.22 | 9.73-9.81 | -0.42 |
| mean cycles | 6.29-6.30 | 5.47-5.50 | |

u and q are identical at 600.

**seep-bench 1200, check slot** (artifacts/seep-rewrite/lane-s/compare-s2i.md; audit audit-s2i):
- G1-G5, G7 and G8 (trajectory) pass. Impact 375.8 / 375.5 vs 380.8 / 382.8; RMS 0.74 / 1.34.
- G6 fails at 1200: centroid 388.2 vs 370.4, and L1 0.272 > 0.214.
- Fenced median 11.75 ms (p95 17.1, max 43.9). The check slot was contended during this run; the interleaved A/B is
  the comparison.

Under "every speed change keeps G6 passing", S2i is not landed. Its patch is saved at
~/code/me/fluid-scratch/lane-s/s2i-pcgguess-exactcoarsest.patch, and the working tree still holds it.

### Scratch workflow

The old ab.sh, which swapped HEAD files into lib/methods/seep, is deleted; that swap is what lane H saw at 21:42.
ab2.sh builds the A side in ~/code/me/fluid-scratch/lane-s/ab-tree and loads it through SEEP_IMPL, and it never writes
the checkout.

## 2026-11-14 (night): S2i landed under the corrected G6 (4d90be9e)

S2i is unchanged from the entry above. The re-compare of artifacts/seep-rewrite/lane-s/seep-s2i.jsonl with lane H's
corrected seep-compare (4d90be9e) passes every gate (artifacts/seep-rewrite/lane-s/compare-s2i.md):
- **G1-G5, G7:** pass as before.
- **G3 corrected:** 1.280 / 2.585 / 3.894 / 5.197 L, all inside their bands.
- **G6 corrected:**
  - 600: centroid 371.3 in [351.0, 386.2]; L1 vs restored 0.107 <= 0.327.
  - 1200: centroid 388.2 in [354.4, 391.4]; L1 vs restored 0.200 <= 0.214.
- **G8 trajectory:** impact 375.8 / 375.2 vs 380.8 / 382.8; height RMS 0.74 / 1.34.

Audit: artifacts/seep-rewrite/lane-s/audit-s2i/audit.md.

**Ledger, 1200 steps, check slot:** chamber phi vs delivered is -0.1 / +0.2 / +0.8 / +0.3% at 300 / 600 / 900 / 1200.
There are no deletions, drift is at most 8e-5, and no step is capped.

**Cycle histogram over 1200 steps** (probe in lane S's scratch), cycles 3..12: 9 / 107 / 578 / 329 / 157 / 16 / 3 /
0 / 0 / 1. The single 12-cycle step comes before step 300. MAX_CYCLES therefore cannot drop without capping a step,
and the tolerance stays.

**Speed:** -0.42 ms fenced in the interleaved A/B above (check slot).

## 2026-11-14 (night): S2j, per-axis CFL activation band + lighter level-4 smoothing (-0.75 ms)

**What S2j changes:**
- **Activation band.** The chamber activated every brick within two bricks of any liquid or particle brick, in all
  directions. Most of that band was the floor film's layers 1-2, which a horizontally moving film never reaches.
  - G2P and emission now mark, per brick and per axis, any particle moving more than FAST_CELLS = 6 cells a step along
    that axis (bricks FAST, 3 bits; reset with PCOUNT).
  - Classification reaches two bricks along an axis only from a brick marked fast along it, and one brick otherwise.
  - Six cells keeps a particle leaving its brick inside the next brick together with its 1.5h P2G support.
  - This is the CFL band of a narrow-band method (Bridson 2015 §6.3; Ferstl 2016 §3). The reservoir is unchanged at
    reach one.
- **Smoothing.** Level 4 and level 5 in coarseSolve use 2+2 red-black sweeps instead of 4+4. The pairs stay
  symmetric (R,B pre; B,R post). Mean cycles go from 5.50 to 5.39: the aggregated level-4 correction was over-smoothed.

**Effect at step 600:**
- Active chamber bricks: 1451 -> about 906 (floor layer 2 goes from 232 to 126).
- Active list: 2432 -> about 2260.
- Particles whose trilinear stencil touches a brick inactive that step are unchanged against HEAD: 41.4k vs 40.7k, summed
  over 60 samples in 1200 steps. Most of these touch the solid wall bricks.

**A/B, 600 steps, check slot, interleaved x3** (HEAD 434fbdff from the scratch tree via SEEP_IMPL):

| | HEAD | S2j |
|---|---|---|
| fenced ms | 9.81-9.83 | 9.04-9.13 |

That is **-0.75 ms**.

**seep-bench 1200, check slot** (artifacts/seep-rewrite/lane-s/compare-s2j.md, audit-s2j): all gates pass.
- **G3 corrected:** 1.289 / 2.596 / 3.921 / 5.214 L.
- **G6 corrected:**
  - 600: centroid 373.4 in [351.0, 386.2]; L1 0.096.
  - 1200: centroid 378.6 in [354.4, 391.4]; L1 0.088.
  - G6 as written passes too on this run (L1 0.114).
- **G8:** impact 375.8 / 375.5 vs 380.8 / 382.8; RMS 0.78 / 1.36.
- **Bench timing:** fenced median 9.79 ms (thirds 8.89 / 9.83 / 11.73). The slot was contended late in the run, so the
  A/B is the comparison.

**Ledger, 1200 steps:** chamber phi vs delivered is -0.7 / -0.3 / +0.8 / +0.6%. There are no deletions, drift is at most
8e-5, and no step is capped.

## 2026-11-14 (night): cost attribution after S2j, and what did not pay

All runs are on the check slot, 600 steps, at d5d3ae2c (fenced 9.04-9.13 ms in the A/B). Profile per step:

| kernel group | ms |
|---|---|
| velocity | 1.11 |
| pressure | 4.49 (pcgUpdate 0.75, coarseSolve 0.66, pcgPrecond 0.51, pcgDirection 0.51, mgDown0/1 0.44 each, mgUp2 0.33, mgDown3 0.28, mgUp1 0.21, mgUp3 0.20, mgRestrict3 0.16) |
| massFit (4 passes) | 0.67 |
| redistSweep | 0.66 |
| extend | 0.54 |
| separate | 0.52 |
| advectPhi | 0.39 |

**Measured probes.** Each change below is a deliberately wrong-physics or capped variant, used only to price a cost;
none is kept.
- **Empty gated cycles:** MAX_CYCLES 12 -> 8 saves 0.28 ms fenced (44 dispatches, about 6.4 us per empty indirect
  dispatch). The empty tail is therefore about 0.46 ms a step. The cycle histogram has one 12-cycle step and three
  10-cycle steps, so the cap cannot drop without capping a step. Only fewer dispatches per cycle (fusion) reduce it.
- **Separation:** skipping the neighbour loop takes separate from 0.52 to 0.05 ms, so the pair search is the whole cost.
  It grows with the particle count, which doubles by step 1200. The phi-gradient projection costs nothing.
  - I tried staging the workgroup's neighbour box in workgroup memory: a box of up to 512 cells and 1536 particles,
    filled by a scan. It was slower (0.72 ms), because about 29 KB of workgroup memory leaves one workgroup per core.
    Reverted.
- **Velocity:** skipping the backtraces takes velocity from 1.11 to 0.38 ms, so the RK2 traces cost about 0.7 ms. They
  are mostly in the reservoir: the surface layer, the deep non-bulk bricks next to the throat and walls, and air faces
  next to liquid.

**Directions that look worth the next round:**
1. **Fuse the upward MG chain** (U3 -> U2 -> U1) into one tile-local pass per level-4 block. This saves 2 dispatches per
   cycle slot (26 a step, about 0.17 ms empty plus the full ones) and a round trip of the corrections. The open question
   is whether post-smoothing stays the adjoint of pre-smoothing, so that the preconditioner stays symmetric.
2. **Pair search** with a smaller footprint: f16-packed relative positions, or a per-cell walk in which a subgroup of a
   cell's ~8 particles shares each loaded neighbour. The target is the separate kernel's growth with particle count.
3. **Reservoir traces:** widen the coarse BULK treatment toward the throat-free walls, so that fewer deep non-bulk fine
   bricks trace every face.

## 2026-11-14 (late night): item 1 scoping, upward MG chain fusion (no code yet)

I read the up-chain kernels: mgUp3, upKernel(2), upKernel(1) and pcgPrecond (U0).

Each up-level post-smooths its block by block Gauss-Seidel, with the halo frozen at the current iterate
e_pre + P e_(l+1)^post. That makes it e <- e + M^-T (b - A e) with M = blockdiag(M_GS), the exact adjoint of the
zero-halo pre-smoothing. Two consequences:
- The residual b - A e at a block's boundary needs the neighbours' current iterate.
- The halo at level l needs the post-smoothed parents at level l+1 of the cells next to the block. Those belong to
  neighbouring level-(l+1) blocks, which other workgroups smooth.

There are therefore two exact options.

**(a) Recomputation cone.** One workgroup per brick fuses U_(l+1) into U_l by recomputing U_(l+1) for the 26 neighbour
blocks before smoothing its own.
- For U2+U1: 27 level-2 blocks (216 cells, each with its 4^3 halo) per brick, BULK neighbours included.
- The adjoint is unchanged, so symmetry is exact by construction.
- Saving: one dispatch per cycle slot (13 a step: about 0.04 ms empty, plus the U2 pass of about 0.05 ms a cycle),
  against roughly 27x the U2 loads. Estimate -0.1 to -0.2 ms. Fusing U3 as well extends the cone to level 4 blocks.

**(b) Symmetric additive multigrid** (BPX form): z = sum_l P_l S_l R_l r, with P_l = R_l^T piecewise constant, and each
S_l a zero-halo block symmetric Gauss-Seidel (forward then reverse, so block-diagonal SPD).
- M^-1 is SPD by construction.
- The up chain collapses into U0: each fine cell gathers its parents' corrections. The down chain becomes restriction
  of r plus independent smoothing.
- A cycle could drop from 11 passes to about 5: X + fine SGS, the restriction/smoothing chain, the coarse solve,
  U0 + dot, direction. This is also item 4's lever.
- Additive MG usually needs 1.5-2x the PCG iterations of the multiplicative V-cycle. A CPU Float64 model must show the
  cycle count before any GPU work. The same model gives the symmetry oracle the coordinator asked for:
  z^T M^-1 y = y^T M^-1 z on random vectors over a small baked mask.

**Recommendation:** build the CPU Float64 oracle first, modelling the current V-cycle and (b). It proves symmetry for
any variant and prices (b)'s cycle count cheaply. Take (b) if the cycle count stays under about 8. Otherwise take (a)
for U2+U1 only.

Item 2 (reservoir traces, about 0.7 ms of the velocity kernel's 1.11 ms, plus reservoir rows in every pressure pass) has
the larger headroom of the two.

## 2026-11-14 (night, lane S second agent): S2k, BULK raised to row 184 and a smaller throat block (-1.2 ms)

**What S2k changes** (seep-layout.ts only; the composite h/4h machinery of S2a is unchanged and was already general in
its extent):
- **DEEP_ROW_LIMIT 176 -> 184.** Brick row 22 (rows 176-183) becomes DEEP and BULK: 240 more 4h bricks. The reservoir
  drops 2.8-3 cells over 20 s, so the surface stays above row 189 and every DEEP vertex (rows <= 183) lies at least
  PHI_MAX = 6 cells below it. Its band-clamped distance is -6 for the whole run, exactly the value it is frozen at.
  The GPU invariant (row 184 stays liquid) still latches FAIL_DEEP; it never fired.
- **Throat approach block 3x5x4 -> 2x2x2 bricks** (x 224-239, y 56-71, z 24-39): 16 cells upstream and 6-8 cells
  beside the 2x2 mouth, against the reference's +-4 cells pinned at h. Radial convergence puts the speed 8 cells out at
  about 1% of the throat's. A 4-brick block (8 cells upstream) measured the same time, so the 8-brick block is kept.
- NBULK 5220 -> 5512. The fine reservoir is now brick row 23 (the surface) and row 24 (air band), plus the 8 throat
  bricks. tests/seep-layout.test.ts pins the new count and checks that no cell within 6 of the mouth, or 16 upstream,
  is BULK.

**A/B, 600 steps, check slot, interleaved x3** (HEAD d5d3ae2c from the scratch tree via SEEP_IMPL):

| | HEAD | S2k |
|---|---|---|
| fenced ms | 8.93-9.03 | 7.73-7.83 |
| mean cycles | 5.35-5.40 | 5.15-5.22 |

That is **-1.2 ms**. Row 22 alone gave -1.0 ms (8.99-9.01 vs 7.99-8.16); the throat block -0.25 ms more. u 2.599 is
identical at 600 and q rises 0.258 -> 0.259.

**seep-bench 1200, check slot** (artifacts/seep-rewrite/lane-s/compare-s2k.md, audit-s2k): all gates pass.
- G1 -3.98% (s2j -3.9%); G2 -2.6 / -3.7 / -3.7 / -3.6%; G4 2.597 vs 2.639 (-1.56%); G5 90% at 267 vs 296.
- G3 corrected: 1.286 / 2.600 / 3.889 / 5.183 L, all inside their bands.
- G6 corrected: 600 centroid 369.6, L1 0.051; 1200 centroid 380.2, L1 0.109. G6 as written passes too (L1 0.152).
- G7 drift 0.00%. G8: impact 375.8 / 375.5 vs 380.8 / 382.8; RMS 0.80 / 1.29.
- Bench timing: fenced median 8.49 ms (thirds 7.54 / 8.50 / 9.01, p95 9.46), check slot.

**Ledger, 1200 steps, check slot:** chamber phi vs delivered is -0.03 / +0.01 / +0.44 / -0.11% at 300 / 600 / 900 / 1200.
No deletions; total drift within 1e-6 at every checkpoint (bench drift 0.00%).

## 2026-11-14 (night, lane S second agent): after S2k, attribution at 600 and 1100, item 2 rejected, small wins

**Cost attribution at S2k** (check slot, profile per step; the 600-step run fenced 7.77 ms and its profile sums to 8.34;
the 1100-step run fenced 8.27 and sums to 9.32):

| kernel | 600 | 1100 |
|---|---|---|
| velocity | 0.90 | 0.90 |
| massFit | 0.67 | 0.66 |
| redistSweep | 0.52 | 0.46 |
| coarseSolve | 0.52 | 0.64 |
| separate | 0.51 | 0.77 |
| pcgUpdate | 0.51 | 0.56 |
| extend | 0.46 | 0.46 |
| pcgPrecond | 0.41 | 0.38 |
| mgDown1 / pcgDirection | 0.33 / 0.33 | 0.39 / 0.38 |
| velocityBulk / coarsenBrick | 0.31 / 0.30 | 0.30 / 0.28 |
| advectPhi / p2g | 0.28 / 0.26 | 0.28 / 0.34 |
| mgDown0 / g2p | 0.25 / 0.23 | 0.33 / 0.44 |

Pressure totals about 3.6 ms at 600, roughly 0.55 ms a cycle. Between 600 and 1100 the growth is almost all
per-particle work: separate +0.26, g2p +0.21, p2g +0.08, count/scatter +0.05. The grid-side kernels stay flat.

**coarseSolve is mostly fixed cost.** Running its V-cycle body twice gives an identical result at twice the work. That
variant costs +0.11 ms fenced (7.59 / 7.60 -> 7.70 / 7.72), so the body is about 21 us a call. The rest of the profile's
~85 us a call is the serialization point: a one-workgroup dispatch between two full-GPU ones. Only fewer cycles or fewer
passes reduce it.

**Item 2 (cycle structure): additive and hybrid multigrid rejected on a CPU Float64 model.**
`~/code/me/fluid-scratch/lane-s/mgmodel.mts` models our hierarchy on 64x32x32 and 128x64x32 masks with a free surface,
using aggregation at OC = 1.6, the same smoothers and the exact coarsest solve. It counts PCG iterations to residual
reductions of 1e-2 / 1e-3 / 1e-4 for three right-hand sides: white noise, smooth, and surface-concentrated (- = not
reached in 40 iterations).

| preconditioner | white | smooth | surface |
|---|---|---|---|
| V-cycle (ours) | 5 / 7 / 9 (128-grid 7 / 8 / 10) | 7 / 8 / 10 | 1 / 2 / 3 |
| BPX additive, OC 1.6 | - / - / - | - / - / - | 8 / 15 / 27 |
| BPX additive, OC 1.0 | 33 / 40 / - | 40 / - / - | 5 / 8 / 20 |
| brick-local V + additive coarse | 29 / 40 / - (128: 37 / - / -) | 40 / - / - | 3 / 7 / 14 |
| zero-halo S3 + V4 hybrid | - / - / - | - / - / - | 4 / 12 / 23 |

Every variant is symmetric to 1e-14 (z^T M^-1 y = y^T M^-1 z). Additive forms need 30-40 or more iterations against
the V-cycle's 7-10, so cutting a cycle from 11 passes to 5 cannot pay. The V-cycle stays. Option (a), the U2+U1
recomputation cone, is still open at an estimated -0.1 to -0.2 ms.

**Probes (wrong physics or rejected, none kept):**
- **BULK through row 23 (the surface row):** cycles 5.2 -> 11.8 and 8.65 ms. The composite coupling at the free surface
  ruins the preconditioner as well as the physics. Row 184 is the ceiling.
- **Separation without neighbour loads:** 0.377 -> 0.197 ms at 600. The loads are about half its cost, and the pair math
  and the pgrid lookups the other half.
- **Tiled per-brick G2P** (a workgroup per brick, the brick's 9^3 velocity faces in workgroup memory): 7.87 / 7.76 vs
  7.81 / 7.74, slower. Crowded bricks unbalance it. Reverted.

**Small wins (working tree, check slot, 600 steps, interleaved x2):** base 7.81 / 7.74; coarsenBrick packing eight BULK
bricks a 64-thread workgroup (was one brick per workgroup, 8 of 64 lanes live) 7.77 / 7.59; plus the separation
restructure (load the four x-rows' (start, end) up front, merge a row's two contiguous cells, two neighbours per
iteration, same accumulation order) 7.71 / 7.58. That is about -0.1 ms.

**S2l landed** (the coarsenBrick packing plus the separation restructure above; results are unchanged by construction,
since both keep every value and the accumulation order).
- A/B, 600 steps, check slot, 5 interleaved rounds: base 7.81 / 7.74 / 7.93 / 7.90 / 8.84 against S2l 7.71 / 7.58 /
  7.63 / 9.31 / 8.44. Two rounds were disturbed by other lanes' GPU tests sharing the GPU. Clean rounds give -0.1 to
  -0.3 ms.
- Gates, seep-bench 1200 on the check slot (compare-s2l.md, audit-s2l): all pass.
  - G1 -3.96%. G2 -2.56 / -3.67 / -3.69 / -3.61%. G4 2.597 (-1.56%). G5 90% wet at step 274 vs 296.
  - G3 corrected: 1.282 / 2.583 / 3.898 / 5.199 L, all in band.
  - G6 corrected: 600 centroid 370.5, L1 0.100; 1200 centroid 383.1, L1 0.104. G6 as written passes too.
  - G7 drift 0.01%. G8: impact 375.8 / 375.3, RMS 0.80 / 1.30.
  - The bench median, 9.06 ms (thirds 8.66 / 9.14 / 9.66), ran under GPU contention from other lanes' tests, so it is
    not a speed figure.
- Ledger, 1200 steps: chamber phi against delivered is -0.57 / -0.32 / +0.19 / +0.82% at 300 / 600 / 900 / 1200, with
  no deletions and drift at most 1.2e-4. HEAD's code (S2k), rerun the same hour, gives -0.60 / -0.15 / +0.82 / +0.52%
  with drift 8.7e-5, so the spread is run-to-run variance.

**Rejected or neutral (probes, check slot, 600 steps):**
- velocityBulk folded into the velocity dispatch as leading workgroups (its tile aliased onto vt): 7.68 / 7.73 vs
  7.59 / 7.77, no gain.
- Velocity tile halo 2 -> 1 (VT 12 -> 10), or red/redFar aliased into vt: within noise.
- Doubled dispatches, to price each kernel's marginal cost: velocityBulk about 0.26 ms, separate 0.2, advectPhi 0.24,
  one redistSweep pass 0.17. Inside velocityBulk, neither the coarse traces (-0.07) nor the throat-interface fine traces
  (-0.06) dominate. The rest is staging and the three one-workgroup-per-core waves.

**Lead for the next pressure round: the late cycles are spent on chamber-film cells at brick boundaries.** I recorded
every fine row with |r| > TOL at cycles k >= 3 (probe tree v/rloc). The 29-505 cells per step are almost all film
cells at y = 0-4. Their x and z sit at brick faces (x = 7 or 0 mod 8, z = 7 or 0 mod 8) or at the chamber walls. The
L0, L1 and L2 smoothers are all brick-aligned block Jacobi with a zero or frozen halo, so a 1-2 cell film converges
slowest across the brick faces. On the CPU model (mgmodel2.mts, a chamber-only film 1-2 cells thick, film rhs), the
iterations to 1e-4 are:

| L0 smoother | iterations to 1e-4 |
|---|---|
| block Jacobi, 3 or 6 sweeps (6 is identical to 3) | 4 |
| two-colour multiplicative bricks | 4 |
| a second, half-brick-shifted partition after the first ("AB"; adjoint order in post-smoothing, symmetric to 1e-15) | 2 |

The first-iteration reduction goes from 0.16 to 0.01. On the GPU, AB costs one more L0 pass each way per cycle (about
+0.16 ms a cycle, +2 empty dispatches per tail cycle). The shifted pass also needs a shifted-block list that skips
BULK bricks' coarse slots. It pays if cycles fall from 5.2 to about 3.5 or fewer.

## S2m: staggered L0 smoother (overlapping/staggered block Schwarz)

**Method:** Smith, Bjørstad & Gropp 1996, "Domain Decomposition", the multiplicative Schwarz method over two
staggered partitions.
- Pre-smoothing runs the brick partition (X_k), then a partition shifted by (4, 0, 4) cells (D0B, `pcgShiftDown`).
- Post-smoothing runs the exact reverse: the shifted partition with its colour passes reversed (U0B, `pcgShiftUp`),
  then the bricks reversed (U0). Each pass is e <- e + M^-1 (b - A e) with the halo frozen, so the V-cycle stays
  symmetric.
- Shifted blocks are listed by operatorKernel: every liquid L0 brick stamps the 4 xz-shifted blocks that overlap it
  (an atomicExchange flag on a 65 x 32 x 9 table). coarsenTop publishes the count to indirect-args slot 5.
- A 3D shift (y as well) took the same cycles over twice the blocks, most of them air, so the shift is in xz only.
- Two sweeps in the shifted passes give 3.35-3.44 mean cycles and no gain (7.25 / 7.30 / 7.32 against 3 sweeps at
  7.27 / 7.29 / 7.31). One sweep gives 3.96 cycles and is slower (7.38-7.50). Three sweeps stay.
- MAX_CYCLES 12 -> 9.

**Unit test `seep-smoother.test.ts`:** the mgmodel2 structure as a Float64 model on a 32 x 16 x 32 film-plus-pool mask
(aggregation with OC 1.6, block red-black GS, zero and frozen halos, the staggered L0 pair).
- The V-cycle is symmetric to 1e-12 and positive, with and without the staggered partition.
- It converges a thin film in fewer PCG iterations (3 against 4).
- A mutated post order (shifted pass not reversed) fails the symmetry check.

**Cycles (1200 steps, probe tree v/abmax9 at MAX 9):**
- Mean 5.17 -> 3.32. Max 12 -> 7, one step at 7; the histogram for k = 1..7 is 8 / 63 / 767 / 283 / 63 / 15 / 1.
- No capped step.

**A/B, 600 steps, check slot, 4 interleaved rounds (the GPU is shared with other sessions' tests):**

| variant | rounds | median |
|---|---|---|
| S2l | 7.68 / 7.65 / 7.58 / 7.66 | 7.655 |
| shift, MAX 12 | 7.51 / 7.47 / 7.45 / 7.45 | 7.46 |
| S2m (shift, MAX 9) | 7.35 / 7.43 / 7.32 / 7.28 | 7.335 |

That is -0.32 ms. The sweep-count rounds above add three more S2m-equivalent samples: 7.27 / 7.29 / 7.31.

**Gates, seep-bench 1200 on the check slot (compare-s2m.md, audit-s2m): all pass.**
- G1 -4.01%. G2 -2.81 / -3.86 / -3.79 / -3.73%. G4 2.597 (-1.56%). G5 90% wet at step 277 vs 296.
- G3 corrected: 1.275 / 2.595 / 3.914 / 5.198 L, all in band. The 300-step value now also passes G3 as written
  (1.275 <= 1.279; S2l was 1.282).
- G6 corrected: 600 centroid 372.1, L1 0.098; 1200 centroid 385.0, L1 0.140.
- G7 drift 0.01%. G8: impact 375.8 / 375.7, RMS 0.80 / 1.28.
- The bench median, 7.81 ms (thirds 7.04 / 7.88 / 8.24), is not a speed figure: it was run on the check slot with the
  GPU shared.

**Ledger, 1200 steps:** chamber phi against delivered is -0.13 / +0.31 / +0.89 / +0.51% at 300 / 600 / 900 / 1200.
There are no deletions, and drift is at most 9.3e-5.

**Profile at 600 steps (fenced 7.28, kernel sum 8.31 ms):**

| kernel | ms | kernel | ms |
|---|---|---|---|
| velocity | 0.90 | pcgDirection | 0.33 |
| massFit | 0.74 | velocityBulk | 0.31 |
| redistSweep | 0.59 | mgDown1 | 0.31 |
| pcgUpdate | 0.48 | pcgShiftDown | 0.30 |
| extend | 0.46 | pcgShiftUp | 0.30 |
| separate | 0.38 | p2g | 0.25 |
| coarseSolve | 0.38 | g2p | 0.23 |

The shifted passes cost about 0.09 ms per call. The time they save comes from fewer cycles: about 1.85 fewer cycles,
each with its 13 gated dispatches.

**Next:** the per-particle cost (separate, g2p, p2g), the late-run growth from 166k to 332k particles.

## S2n: per-particle batch (count, p2g, g2p atomics)

**Attribution at step 1100 (S2m, check slot):** separate 0.56, g2p 0.41, p2g 0.33, count 0.08, scatter 0.07 ms
(fenced 7.64). At step 600 the same three kernels sum to 0.85 ms; at 1100 they sum to 1.30 ms (166k -> 310k
particles).

Probe kernels (tree v/pp) split the time. Each is a reduced duplicate dispatched just before its kernel. It writes
only the free half of the source pool, so the state is unchanged, and it carries its own label:

| kernel | ms | breakdown |
|---|---|---|
| separate | 0.52 | candidate loop 0.41; range lookups 0.07; one candidate a cell 0.07; phi projection 0.03 |
| g2p | 0.39 | no brick atomics 0.34; RK k2/k3 and collide 0.15; support check 0.03 |
| p2g | 0.31 | shared-atomic splat 0.25; global flush ~0 |

**Landed (S2n):**
- **p2g:** lane l of SIMD group l/32 takes every eighth particle, so the lanes of one group fall in different cells.
  A cell's particles share one stencil per axis-half, so with the sorted order lanes in one cell hit the same
  shared-memory word at once. The fixed-point integer sums are order-independent.
- **g2p:** B_PCOUNT is read only as nonzero (classify, combine), so G2P now sets it to 1 once (load, then store)
  rather than an atomicAdd from every particle on one contended word per brick. Fast bits are OR'd only when one is
  missing.
- **count:** the octant atomicOr is skipped when the count's atomicAdd already returned that bit.

| kernel at 1100 | S2m ms | S2n ms |
|---|---|---|
| p2g | 0.33 | 0.23 |
| g2p | 0.41 | 0.36 |
| count | 0.08 | 0.07 |

**A/B, check slot, 1100 steps, 4 interleaved rounds (fenced median of the run):**

| variant | rounds | median |
|---|---|---|
| S2m | 7.80 / 7.76 / 7.82 / 7.73 | 7.78 |
| S2n | 7.74 / 7.70 / 7.73 / 7.69 | 7.715 |

That is -0.065 ms over the whole run, and S2n is faster in every round. The gain is larger late in the run, since it
scales with particles.

**Gates, seep-bench 1200 on the check slot (compare-s2n.md, audit-s2n): all pass.**
- G1 -4.09%. G2 -2.82 / -3.94 / -3.84 / -3.80%. G4 -1.55%. G5 90% wet at step 275.
- G3 corrected: 1.283 / 2.583 / 3.897 / 5.181 L.
- G6 corrected: L1 0.078 / 0.092; G6 as written passes too.
- G7 0.00%. G8: impact 375.5 / 375.7.

**Ledger, 1200 steps, two runs:** chamber phi against delivered is -0.99 / +0.43 / +0.56 / +0.21% in one and
-0.10 / +0.28 / +0.60 / +0.85% in the other. The step-300 spread is run-to-run. There are no deletions.

**Separate: tried and rejected (check slot, separate's kernel ms at 1100 against 0.52-0.56):**

| variant | ms | result |
|---|---|---|
| shared-memory tiles (workgroup a brick, eight 4^3 tiles each staging its 6^3-cell region; 30 KB) | 1.72 | serial tile phases at about one workgroup a core |
| one flattened loop over the eight ranges | 0.66 | worse |
| eight cells in lock step (masked loads, unrolled) | 0.64-0.89 | worse |
| the 3^3 cells around the own cell, so a cell's lanes share addresses (bit-identical) | 0.82 | worse |
| workgroup 64 / 128 | 0.59 / 0.54 | no gain |
| inverseSqrt form | 0.54 | no gain |
| candidates' x only, strided (4-byte loads) | 0.39 vs 0.41 | bytes don't matter |
| candidates' x only, packed contiguously | 0.54 | lines don't matter |

The candidate loop costs about 100 lane-cycles a candidate, and neither bytes, cache lines nor ALU explain it; that
looks like latency with limited occupancy, which needs hardware counters to confirm. The paired U/US sampler at x in
g2p was neutral (0.41).

## S2o: counter profile of the whole step at 1100, then the level-set cut (-0.34 ms fenced)

**Counters (gpu-frame-profile regime).** Tools are in ~/code/me/fluid-scratch/lane-s/tools: seep-xctrace.ts and
seep-xctrace-worker.ts.
- **Method:** xctrace attaches to a worker warmed to step 1100. The window is 94 steps, captured after a quiet-GPU
  wait (device utilisation under 15% for 5 s). Each dispatch is its own labelled pass in its own encoder, and GPU
  intervals are joined to encoders by encoder-id.
- **Coverage:** exclusive coverage is at least 98% for every stage.
- **Caveat:** split passes serialise the step (16.15 ms a step against 7.75 fenced). Read the occupancy and limiter
  columns, not the times. The capture is in xctrace/s2n-1100/summary.txt.

| stage | split us/step | occupancy | ALU | top limiter | reading |
|---|---|---|---|---|---|
| velocity | 665 | 22 | 35 | ALU 38 | latency, at low (register-limited) occupancy |
| g2p | 376 | 28 | 70 | ALU 82 | ALU |
| separate | 298 | 17 | 48 | ALU 61, LLC 29 | latency, at low occupancy |
| extend.2 / .1 | 247 / 81 | 31 / 28 | 53 / 41 | ALU 56 / 44, LLC 26 | mixed |
| redistSweep.2 / .1 | 166 / 114 | 41 | 59 | ALU 69 | ALU |
| massFit x4 | 98-104 each | 40-44 | 37-40 | ALU 42-45 | nothing saturated |
| coarseSolve.k | 58-79 per cycle | 1 | 1 | - | one workgroup, latency |
| velocityBulk | 228 | 5 | 13 | - | 96 workgroups, a tail |

**Attribution with a kernel microbench.** The tool is kbench.ts with the wrapper kb.sh.
- **Method:** the solver runs to step 1100 and then up to the dispatch before the target. The target dispatch (same
  cfg and launch) and its variants are then timed on that frozen state: 30 back-to-back dispatches a timestamped
  pass, 7 interleaved rounds, medians.
- **Output check:** for idempotent stages each variant's output is compared bitwise with the baseline's.
- **Knockout probes** (scratch only) split each kernel:
  - **massFit:** 148 us a pass. The tetrahedral fill was about 68 of them; the whole-tile copy-through costs 46, an
    early return 17.
  - **redistSweep.2:** 275 us. The 27-neighbour frozen test was about 96 of them (index arithmetic in a k/3, k/9 loop:
    ALU), eik 30, nearSign 20. The sweeps without the test cost 72, staging alone 57.
  - **extend:** about 125 us of staging and output, plus 2-110 us of sweeps, depending on the state.
  - **velocity:** about 830 us in the microbench. Traces take 475 of them; the reservoir has 302k traced faces (1
    segment each), the chamber 17k. The volume fill takes 70.

**Landed (S2o, seep-kernels.ts):**
- **massFit:**
  - The 27 neighbour bricks' chamber flags are staged with the busy test. Each tile vertex is then one device load,
    where it was a class load followed by a phi load.
  - The 9^3 particle counts are loaded at once, with the busy test. The column carry then reads workgroup memory, not
    81 threads making 9 serial device loads.
  - The mismatch is computed only for cells with a band corner (|phi| < 1.5), the only cells the vertex update reads.
    A cell with no negative corner fills 0, and one with all corners negative fills 1, as each of its tetrahedra
    does. A tetrahedron of zeros counts as empty in `tetra`, so a cell whose top corner is 0 takes the full path.
  - The cut cells are compacted into a workgroup list, so whole SIMD groups run `fill` together.
  - Against the baseline the output differs by at most 1 ulp (1.2e-7) in 0.17% of words: the shortcut gives exactly
    F - 0 or F - 1, where the full path rounded.
  - 148 to 109 us a pass, four passes a step.
- **redistInit and redistSweep:**
  - The cut test (an open vertex with an opposite-signed open 26-neighbour) reads a packed code tile at constant
    offsets: 0 not open, 1 open phi >= 0, 2 open phi < 0. The sweep's frozen test is the same test on its staged
    tile. Signs and openness are unchanged by the sweeps.
  - A tile that is all -PHI_MAX is skipped, as all-+PHI_MAX tiles already were. It is a fixed point:
    eik(6, 6, 6) = 6.58 > 6, so no vertex moves, and nearSign keeps the solid-interior signs.
  - Bit-identical outputs. sweep.2 275 to 223 us, sweep.1 173 to 140, init 100 to 92.
- **extend:** the sweeps are also skipped when the tile has no unknown face, since relax returns every known or
  closed value as it is. Bit-identical.

**Tried and rejected:**
- **extend:** staging the 27 neighbour DEEP flags made it slower, 238 to 255 us.
- **velocity:** issuing the 11 tile loads per thread up front saved 9 of 812 us.
- **velocity:** skipping reservoir air faces whose Us nothing reads saved 24 of 828 us. It is exact, but the argument
  is subtle; not landed.

**A/B, check slot, fenced median of the run, interleaved:**

| run | HEAD 6fbe1174 | S2o |
|---|---|---|
| 1100 steps, 3 rounds | 7.67 / 7.71 / 7.67 | 7.34 / 7.31 / 7.30 |
| 1200 steps, 3 rounds | 7.72 / 7.75 / 7.74 | 7.40 / 7.42 / 7.36 |

That is -0.37 ms at 1100 and -0.34 ms at 1200. Mean cycles are unchanged (3.30-3.31).

**Gates, seep-bench 1200 on the check slot (compare-s2o.md, audit-s2o): all pass.**
- G1 -4.19%. G2 -3.25 / -4.16 / -4.04 / -3.96%. G4 -1.55%. G5 90% wet at step 274.
- G3 corrected: 1.280 / 2.588 / 3.893 / 5.198 L.
- G6 corrected: L1 0.063 / 0.047.
- G7 0.01%. G8: impact 375.8 / 375.6.
- The bench's fenced median is 7.385 ms.

**Ledger, 1200 / 300, two runs:** chamber phi against delivered is +0.12 / +0.30 / +1.02 / +0.83% in the first and
-0.31 / +0.46 / +0.60 / +0.26% in the second. The first run's step-900 point is the run-to-run spread; S2n's two
runs differed by up to 0.9 points at a step. There are no deletions.

**Next (from the counters):**
- **velocity trace phase:** 475 us, about 20x its ALU count. It is latency-bound at 22% occupancy, and register
  pressure is the suspect; tools/experiments/lane-n/dump-msl.mts can dump the MSL.
- **coarseSolve:** about 75 us per cycle in one workgroup.
- **velocityBulk:** 5% occupancy over 96 workgroups.

## S2p: velocity backtraces as their own slab kernel, velocityBulk's interface faces spread (-0.31 ms fenced)

**Register pressure, confirmed from the compiled MSL.** Dawn dumps the MSL (dump_shaders,
disable_symbol_renaming). The `max_total_threads_per_threadgroup` attribute is stripped and the result is compiled
with Metal, so the pipeline's maxTotalThreadsPerThreadgroup shows the register limit. The tools are
~/code/me/fluid-scratch/lane-s/tools/mstat.sh and mtlstat.
- **velocity (HEAD):** the register limit is 832 threads, and the kernel also uses 28.7 KB of workgroup memory. One
  512-thread brick a core.
- **velocity pass 1 alone:** 1024.
- **Traces alone:** 704-832 in every form tried: tile or none, compacted, sorted, components in a loop.
  - The trilinear vec3 sample itself holds the registers (24 loads in flight).
  - Knockouts reach 1024 only with no sampling at all.
- **Counters for the split trace kernel:** occupancy 16%, ALU 36%, nothing saturated. It is latency-bound.

**Attribution (kbench).**
- **Method:**
  - Acc-preserving copies of pass 1 (atomicLoad in place of atomicExchange) make the pair rerunnable.
  - kbench now takes multi-kernel variants (`a+b`), and a y-split indirect launch (`k@y8`).
  - The checked slice is poisoned before each checked run.
- **Trace counts at 1100:** the reservoir has 485.5k traced faces (486.7k segments, max 9) on 3908 64-cell slabs. The
  chamber has 17.2k faces (26.6k segments, max 11) on 1315 slabs.
- **Costs:**
  - The trace kernel with no sampling costs 144 us (launch, compaction, stores).
  - Chamber sampling costs 146 us for 17k faces. Each chamber brick held a 512-thread workgroup a core while its few
    long traces ran.

**Landed (S2p):**
- **velocity, split:**
  - **Pass 1** stays in `seep.velocity`: PHIC, fill, B_MINPHI, zero, far-skip and particle transfers. It writes each
    cell's need mask into its KNOWN word. Projection rewrites that word for every LACT brick, and nothing reads it in
    between.
  - **`seep.velocityTrace`** runs the backtraces:
    - Each brick is TRACE_SLABS = 8 slabs of 64 cells (indirect x = active bricks, y = 8, from the new args slot
      ARG_TRACE, counted by classify and reset by g2p).
    - Each slab compacts its traced faces into a workgroup list, and full warps run `fineFaceSL` plus gravity.
    - There is no velocity tile. Without the 20.7 KB tile and with 64-thread workgroups, the global-read samples are
      as fast, and light bricks no longer hold a core.
  - **Shared code:** `fineFaceSL` and `gravityOn` are now one WGSL block (FACE_SL_WGSL) shared with velocityBulk.
    `slSegs` keeps the unclamped segment count for C_MAX_SEG.
  - **Check:** US (vel 3N..6N) is bit-identical to HEAD on frozen state.
  - **Time:** velocity plus trace takes 812 to 618 us at 1100 (interleaved kbench, 7 rounds).
- **velocityBulk:**
  - **Before:** an interface coarse face (fine cells below) made one thread trace its 16 fine faces serially, which
    was the kernel's tail.
  - **Now:** the block lists its interface faces, and 16 threads trace each one.
  - **Check:** US and the coarse BUS faces are bit-identical.
  - **Time:** 264 to 157 us.
  - **What is left:** staging the coarse tile costs about 45 us beyond the launch (cface reads 16 fine faces next to
    fine cells, with one workgroup a core), and the coarse traces about 70 us.

**Tried and rejected (velocity traces), velocity plus trace in us, same session:**

| variant | us |
|---|---|
| HEAD, merged | 812-852 |
| Split, per-brick 512 threads, with tile | 692-705 |
| Same, no tile | 795 |
| Compacted per brick, with tile | 645-655 |
| Compacted per brick, 256 threads, no tile | 633-649 |
| Compacted per brick, sorted by axis | 677-686 |
| Slabs of 32 / 64 / 128 threads | 655 / 618-636 / 636-638 |

- **Slabs:** 64 was chosen.
- **Sampling the components in a non-unrolled loop:** this did not lower the register limit (704).

**coarseSolve, attributed (not changed).** The 32x32 matvec is only 1024 MACs, as noted.
- **Method:** a probe keeps C_RES and never converges, so it is rerunnable at seep.coarseSolve.1.

| part | us |
|---|---|
| Whole call | 83-103 (kbench is noisy here) |
| Empty dispatch of the 1024-thread workgroup | 17.5 |
| Thread-0 preamble and gates | about 5 (dropping the gate writes changed nothing) |
| Each level-4 red-black sweep | about 8 |
| Level-5 sweeps | about 15 |
| Top solve | about 10 |

- **The sweep cost is the work:** with the sweep's offS replaced by a diagonal-only update, the cost falls to that of
  having no sweeps, so the cost is offS, not the barriers. Halving the cells per sweep halves it.
- **Spill hypothesis:** the kernel needs 704 threads' worth of registers and runs 1024. I moved level 4's x and y
  couplings to workgroup memory, which brought the uncapped limit to 896 threads.
  - **Speed:** 72 against 83 us (minimums 63 against 65) in noisy runs.
  - **Values:** they differ by 1 ulp (fma contraction).
  - **Verdict:** not landed; spilling is not the main cost.
- **Next candidate:** fold the solve into mgRestrict3's last workgroup. This needs a device-scope fence, which WGSL
  lacks.

**Not landed: the reservoir air-face skip (-24 us).** Projection reads Us when either cell's PHIC is negative. A
DEEP or inactive neighbour's PHIC is not written by this step's velocity (DEEP bricks skip it; inactive ones keep
clear's or an older value). So all 12 corners being non-negative is not on its own proof that projection skips the
face. It needs the neighbour brick's class in the predicate, plus a test, before it can land.

**A/B, check slot, 1200 steps, fenced median, interleaved (ab-s2p-1200.txt):**

| run | HEAD 307be056 (S2o) | S2p |
|---|---|---|
| round 1 | 7.39 | 7.03 |
| round 2 | 7.34 | 7.03 |
| round 3 | 7.32 | 7.03 |

That is -0.31 ms. Mean cycles are unchanged (3.30-3.34).

**Gates, seep-bench 1200 on the check slot (compare-s2p.md, audit-s2p): all pass.**
- G1 -4.04%. G2 -2.88 / -3.92 / -3.84 / -3.75%. G4 -1.55%. G5 90% wet at step 277.
- G3 corrected: 1.286 / 2.588 / 3.898 / 5.197 L.
- G6 corrected: centroid 372.6 / 387.0, L1 0.100 / 0.150. The limits are 0.327 / 0.214; the run-to-run spread of the
  chamber profile is seen at S2o too.
- G7 0.01%. G8: impact 375.8 / 375.4.
- The bench's fenced median is 7.08 ms.
- The audit matches S2o's (wall zone, G3 as written fails at 300 as before, ungated).

**Ledger, 1200 / 300, two runs (ledger-s2p.txt, ledger-s2p-b.txt):** chamber phi at g2p against delivered is
-0.55 / +0.57 / +0.58 / -0.01% in the first run and +0.21 / +0.07 / +0.65 / +0.86% in the second. Both are within 1%,
with no deletions.

## S2q: the empty cycle tail is Dawn's indirect validation; the convergence test before the V-cycle, and a two-part submit (-0.30 ms fenced)

**The tail, measured (check slot, 1200 steps, interleaved fenced medians, ab-mcyc-1200.txt).**

| MAX_CYCLES | round 1 | round 2 | round 3 |
|---|---|---|---|
| 9 (HEAD) | 7.02 | 6.98 | 7.02 |
| 12 | 7.15 | 7.15 | 7.11 |
| 15 | 7.30 | 7.24 | 7.34 |

- **Per empty slot:** about 50 us (13 dispatches), so HEAD's 5.7 empty slots a step cost about 0.28 ms.
- **Cycle histogram over 1200 steps (probe build):** 1: 5, 2: 70, 3: 762, 4: 284, 5: 71, 6: 7, 7: 1. There are no cap
  hits.

**What an empty slot costs.**
- **CPU:** queue.submit takes 0.152 ms with 9 slots and 0.228 ms with 15, about 13 us a slot. The implementation
  translates the command buffer for the GPU at submit, and a fenced step's GPU work waits for all of it.
- **GPU:** the rest is Dawn's indirect-dispatch validation.
  - **What Dawn does:** with validation on, every dispatchWorkgroupsIndirect is preceded by a one-workgroup
    validation dispatch that copies and checks the arguments.
  - **Diagnostic** (toggle `skip_validation`, smoke runner only, 600 steps): MAX_CYCLES 9 runs 6.15 / 6.17 ms and
    MAX_CYCLES 15 runs 6.14 / 6.19 ms. Empty slots become free, and the whole step is 0.4 ms faster than with
    validation.
  - **The toggle is not usable:** browsers validate, and the device is not ours.
- **Each slot dispatch costs about 3.5 us whatever its form.** Launching pcgCheck and coarseSolve directly (one
  workgroup that returns at once) instead of indirectly changed nothing: 6.86 / 6.87 / 6.82 against 6.84 / 6.84 /
  6.87 (ab-chkd-1200.txt). A tiny real dispatch costs what the validation dispatch costs.
  - **So with validation on, a slot's floor is its dispatch count.** Neither zeroing a later slot's arguments nor a
    GPU-computed budget removes that cost; only fewer encoded dispatches or fewer slots would. A dispatchWorkgroups(0)
    is free, but only the CPU could choose it, and that needs a readback.
- **Dispatch microbench (dispatch-bench.ts, 200 dispatches over 13 pipelines):**

  | dispatch | validation on | skip_validation |
  |---|---|---|
  | indirect, 0 workgroups | 12.8 us | 7.2 us |
  | direct, 1 workgroup | 4.6 us | 4.6 us |

**Landed (S2q):**
- **pcgCheck:** the convergence test is now its own one-workgroup kernel, `seep.pcgCheck` (P_k), run right after
  X_k.
  - **The waste removed:** before, coarseSolve.k held the test after iteration k's down sweep. That sweep (D0B, D0,
    D1, D3, D3r) ran under the previous iteration's gate, so the converged iteration still ran it.
  - **New order:** P_k opens record k, which gates the whole of iteration k (D0B..U0, direction) and X_(k+1), only
    while the residual exceeds the tolerance. This is the standard PCG order: the test comes before the
    preconditioner solve (Saad 2003, Alg. 9.1).
  - **Launch:** P_k and coarseSolve.k are launched directly with one workgroup. P_k (64 threads) only closes record k
    once C_CONV is set. coarseSolve returns at once when C_CONV is set.
  - **What this removes:** the cgate buffer pair and its binding rule, since no dispatch reads its arguments from a
    buffer it writes.
  - **Unchanged:** the math and the record layout.
- **Two-part submit:** buildStep returns two command buffers, the first SUBMIT_PREFIX = 4 dispatches (classify,
  clear, count, scan) and the rest, and advanceTo submits them one after the other. The GPU starts the prefix while
  the rest is translated.

  | split (600 steps, fenced) | ms |
  |---|---|
  | unsplit | 6.57 / 6.57 |
  | 4 | 6.46 / 6.41 / 6.37 / 6.43 |
  | 2 | 6.43 / 6.46 |
  | 16 | 6.50 / 6.46 |
  | 4 and 24 | 6.45 / 6.48 |
  | 60 | 6.57 / 6.57 (a long first part delays the GPU's start) |

  It changes nothing in pipelined mode, where the next step is already encoded and queued.

**A/B, check slot, 1200 steps, fenced median, interleaved (ab-s2q-1200.txt):**

| run | HEAD 1209eb14 (S2p) | S2q |
|---|---|---|
| round 1 | 7.10 | 6.76 |
| round 2 | 7.08 | 6.80 |
| round 3 | 7.10 | 6.80 |

- That is -0.30 ms: the check split is about -0.09 (ab-chk-1200.txt) and the split submit the rest.
- Mean cycles 3.29-3.30 against 3.31-3.33, with no cap hits.
- One pipelined bench run on the check slot gave 6.22 ms a step. This is not an A/B; lane H's figure for 1209eb14 is
  6.32.

**Gates, seep-bench 1200 on the check slot (compare-s2q.md, audit-s2q): all pass.**
- G1 -4.04%. G2 -2.84 / -3.95 / -3.86 / -3.76%. G4 -1.55%. G5 90% wet at step 279.
- G3 corrected: 1.289 / 2.591 / 3.923 / 5.183 L.
- G6 corrected: centroid 373.4 / 382.2, L1 0.099 / 0.089 (limits 0.327 / 0.214).
- G7 0.00%. G8: impact 375.5 / 375.6.
- The bench's fenced median is 6.76 ms.
- The audit matches S2p's.

**Ledger, 1200 / 300, two runs (ledger-s2q.txt, ledger-s2q-b.txt):** chamber phi at g2p against delivered is
+0.33 / +0.45 / +0.40 / +0.43% in the first run and -0.06 / +0.63 / +0.67 / +0.34% in the second. Both are within 1%,
with no deletions.

**What is left of the tail.**
- **The cost:** each empty slot is now 14 dispatches at about 3.5 us, about 0.28 ms a step.
- **Fewer slots:** MAX_CYCLES is 9 against a measured peak of 7, and each slot removed saves about 50 us. This is the
  coordinator's call; the pools follow the same measured-peak rule.
- **Fewer dispatches a slot:** folding mgRestrict3 into coarseSolve's start would remove one dispatch a slot, about
  35 us a step. Its sum order can be kept so the result stays bit-identical.

## Capped-step failure bit (2026-11-15)

A step whose pressure solve stops at MAX_CYCLES above tolerance used to be counted (`capHits`) but not reported as a
failure, so the bench and `assertSimulationHealthy` passed it silently. pcgCheck now also latches
`FAIL.PRESSURE_CAP = 16`, and the failure text carries the capped-step count (`seep-diagnostics.test.ts`).
- **Forced-cap run** (scratch tree with tolerance 1e-12, 30 steps, check slot): latched at step 1, "Seep solver
  latched PRESSURE_CAP (max backtrace segments 0, pool overflow 0, capped steps 1)", so the bench's
  simulationPipelineError check halts it.
- **Normal run** (`seep-bench --method=seep --steps=1200 --every=60 --name=seep-check`, check slot, rc 0): no failure
  bit. At step 1200, left 363.468 / right 5.194 / total 368.670 L against lane H's 2be18980 run at 363.472 / 5.203 /
  368.683; throat u 2.5825 against 2.5825 m/s; particles 332340 against 332336. The step time (13.3 ms median) came
  from a shared check slot and is not a timing measurement.
