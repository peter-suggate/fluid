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
