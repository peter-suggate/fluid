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
