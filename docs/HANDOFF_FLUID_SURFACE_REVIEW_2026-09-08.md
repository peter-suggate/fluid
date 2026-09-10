# Fluid surface review and handoff — 8 September 2026, evening

## Read this first

**The surface carriers built today are a dead end as production. The
diagnosis underneath them is the valuable result, and it points at a
standard method that has not been tried yet.**

Four carriers for "one field supplies both native amounts and the displayed
surface" were built and measured to failure in one day: the retained affine
lift, the latent tensor / Galerkin potential / weak-segment ladders, and the
current-map (characteristic map) composition. The last one was checkpointed
as `a8796cd3` and the quarter scene's default was restored to native CM12 in
`288d922a` the same evening; both are committed, and neither is a production
path. None of them survives the falling-ball impact, none runs at an
interactive cost, and every one of them rediscovered a problem the level-set
literature solved twenty years ago.

The recommendation is a CLSVOF-style coupling built **inside the Sparse CM12
resident on its own fine lattice**: a narrow signed-distance band is the
geometry authority, CM12's conserved density stays the amount authority, and
the two are tied together per connected liquid body, with sharpening pulling
the density toward the band. The first draft of this document tied them per
native cell; the shadow replay below shows that rule copies the solver's
transport error into the geometry, and it has been dropped. **Losasso is out
of scope.** Do not reuse its band, extension, redistance or octree kernels, and
do not cite its lane as precedent; it was too slow and that cost model is what
this program exists to escape.

Scene under test, unchanged: `coarse-first-pool-impact-quarter`, 32×24×32
finest supports at 0.05 m, sphere radius 0.25 m centred at y=0.9125 m, pool
height 0.4 m, paper timestep 1/30 s, region query `0_0_0_25_66.6667_100_8_8`.
Discrete reference after n steps: `y = 0.9125 - 9.80665 (1/30)^2 n(n-1)/2`.

Late the same evening the proposed band was replayed on the CPU against the
unmodified solver's captured fields, both arms, steps 0–20. That experiment
(section "Experiment: shadow band replay") is the evidence behind the design
and gates as they now stand.

The work itself is laid out in "Execution plan" (six work packages, each with
its files, seams, tests and exit criteria), "Success criteria tracker" (every
criterion with its threshold, baseline and current value) and "Status log".

**Handoff, 9 September early.** WP0 is done; WP1 onwards has not been
started and the plan's shape is unchanged. Three things were learned that the
rest depends on:

1. The gate reproduces the replay exactly (C0.1), but HEAD `ec5af4bb` is not
   the solver the afternoon capture came from (C0.2). Its density is a sharp
   native initialisation: the pool is exactly 1.0 up to a cell face and 0.0
   above, and the sphere holds 64.4 L against the analytic 65.45 L, a 1.6 %
   deficit from step 0. The shipping mesh is 7 mm RMS at rest and the
   canonical gate is red for reasons unrelated to the band (C0.3). The
   "Baseline" for every later criterion is the HEAD number, not the
   afternoon one.
2. The pure band (arm A) replayed on the HEAD captures is the afternoon
   result: coarse arm within 0.01 mm RMS through step 7 (0.13 mm in the
   step-7 max), fine arm within 0.05 mm. The velocity the band advects with
   did not change between the two builds, only the amounts did. WP2's
   oracle is `artifacts/surface-band/baseline/{coarse,fine}/replay-normal/phi-A-step-<n>.npy`.
3. The coupling rule as written on 8 September broke on HEAD's sharp density
   and was corrected. Its fill/drain test compared the solver's full/empty
   classes to the band's sub-cell fraction under the ramp amount model; a
   sharp density holds every interface-adjacent cell at exactly 1.0 or 0.0
   while the ramp puts them at 0.875 / 0.125, so the test fired on the whole
   pool surface every step and the pool moved by 27–75 mm. The test is now a
   sign test (solver full while the band's cell centre is in air, and the
   reverse), `--classes topology`, the replay default. With it arm D on HEAD
   holds mass to 0.00 L, keeps the pool flat to 0.014 mm and shifts it by
   nothing; the sphere sits 2–3 mm inside the analytic sphere because the
   solver's sphere is 1.0 L short, which the band reports honestly. A sharp
   amount model for the shift itself (`--width 0`) was tried and rejected:
   its 4³ sub-samples quantise the pool volume in 32 L steps and the shift
   staircases by ±6 mm. WP3's oracle is `replay-topology/phi-D` under the
   same baseline directories: steps 0–30, both arms, mass 0.00 L throughout,
   shifts ≤ 2.10 mm after step 1, two bodies through step 7 and one from
   step 8, no drains and at most 28 fills on any step.

Next, in order: WP1 (static band, C1.1–C1.4), WP2 against the arm-A oracle,
WP3 with the corrected rule against the arm-D oracle. Two solver-side items
sit beside the band work and are not band work: the canonical gate's reds
(C0.3) and the native initialisation's 1.6 % sphere deficit (it sets C4.2's
baseline of 0.21 at rest and the 1.3 mm radial bias the coupled band will
show; a WP5 seam or its own fix).

## What today established

### The root fact: CM12 transport smears shape by construction

The full-fine imposed-flow capture (`tools/capture-retained-imposed-flow-dawn.ts`,
`artifacts/retained-imposed-flow/sphere-full-fine/`) prescribed a uniform
0.75 m/s translation, exactly half a cell per step, and read the actual GPU
native means. They equal `0.5·rho[i] + 0.5·rho[i-1]` after one step and
`[0.25, 0.5, 0.25]` after two, to 4.47e-8. That is ordinary trilinear
semi-Lagrangian diffusion; it is what §3.4 of the paper does. The paper's
answer is §3.5 sharpening, which moves mass back toward the 0.5 contour along
∇ρ. Sharpening restores where the mass is, not what shape it had.

Consequences that the rest of the day kept re-confirming:

- "All-fine CM12" is not a shape reference. The all-fine arm's sphere is as
  blocky as the coarse arm's at 0.2 s
  (`artifacts/retained-surface-endpoint-fix/quarter/section-comparison.png`).
- No reconstruction layered on top of the transported density can recover a
  sphere the transport has already smeared. Geometry has to be carried by
  something that is advected sharply.
- The physics itself is fine. The reviewed real-gravity capture
  (`tools/capture-retained-falling-velocity-dawn.ts`) shows the fully fine
  sphere bulk following the discrete gravity reference to 1.2e-5 m/s through
  six steps. The defect is representation and scalar transport only.

### Why each carrier died

| Carrier | Commit / state | Fatal measurement |
| --- | --- | --- |
| Retained affine lift `q_K = a·q_seed + b` | shipped, `434dca06` era | No departure coordinate. Formerly dry supports stay spatially constant; face jumps 0.36/0.42 after one/two half-cell steps; sphere blocky by 0.2 s in both arms |
| Latent 27-DOF tensor hierarchy | `1f208ea8` | Saturated face targets infeasible; counterexample recorded |
| Galerkin C2 potential fitting | `f3f9eae2` | Arbitrary finite-step moments have no finite C2 potential; counterexample recorded |
| Weak-segment 1D evolution | `a8796cd3`, 6 CPU tests pass | 1D, dense matrices, N≤16, prescribed uniform flow. No 3D or production relevance |
| Current map `rho = q_seed(X(x)) det(DX)` | `a8796cd3` (V19) | See next section |

### The current map in numbers

Measured on the quarter scene, coarse arm, V19
(`artifacts/current-map/quarter/visual-v19/`,
`artifacts/current-map/performance/`):

| Property | Value |
| --- | ---: |
| Map node grid | 129×113×129 = 1.88 M nodes, dense over the domain plus a 16-cell collar |
| State arena | 0.35 GB with the 7 chain slots captured; 0.84 GB with the code default of 32 |
| Frame wall, steps 1 / 2 / 3 | 554 / 2445 / 4069 ms |
| Share of frame in fine-measure quadrature at step 3 | 94 % |
| Sphere radial error, step 6 → 7 | 3.6 mm → 12 mm |
| Still-pool depression, step 6 → 7 | 1.0 mm → 4.0 mm |
| Step 8, first contact | rejected, unresolved quadrature at support 11791 |

Why those numbers are structural, not tuning:

- **Cost grows with the chain.** Every density sample composes every archived
  increment, and the adaptive Gauss quadrature takes thousands of samples per
  fine support. There is no re-anchoring design; the departure-map decision
  doc lists four continuation options and admits none is solved. The range
  cache optimisation in flight trims a cost that would need a hundredfold
  reduction to reach interactive rates on this scene alone.
- **Memory is dense.** The full pool scene has 64× the fine supports of the
  quarter, which puts the arena in the tens of gigabytes.
- **A flow map is singular at a merge.** The falling ball's whole point is
  the impact, and the map fails there by construction.
- **The two remaining shape defects are level-set problems.** The harmonic
  velocity extension blends sphere and pool velocities in the gap, so the
  sphere bottom backtraces 4.3 mm short. The RK2 plus cubic quasi-interpolated
  map has det(DX) = 0.94 at the resting pool, so the pool sinks. Normal
  extrapolation of velocity and volume correction are the standard fixes, and
  they are cheaper to apply to a distance field than to a map.
- **Everything else is switched off.** Rigid bodies, moving solids, inflow,
  live liquid and solid edits throw; gamma diffusion and sharpening are
  bypassed; the retained domain must be unclipped.

Keep the V19 capture and the Python analyzers as a pre-impact reference if
useful. Do not spend another hour on its quadrature.

### Worth keeping from today

- The retained seed compiler (`sparse-cm12-retained-scene-density.ts` and
  `tools/implicit-density/`) initialises an authored field to 1e-16 m plane
  error and 5e-9 m extracted plane error, with 21 ms incremental edits. It is
  the right initialiser for a distance band.
- The immutable VEX freeze before gather (`compileCurrentMapVelocity` in the
  current-map shader) is the correct velocity source for any band advection.
- The measurement tools: `tools/capture-retained-visual-ab-dawn.ts`,
  `tools/analyze-current-map-ray-field.py` (half-level roots versus the
  discrete-gravity sphere, mesh independent), `tools/render-retained-imposed-flow.py`,
  and the red curved gate in `tests/sparse-cm12-surface-grid-imprint-dawn.test.ts`.
- The shadow replay: `tools/replay-surface-band.py` (the reference
  implementation of every step below, on the CPU, against a capture) and
  `tools/render-surface-band-replay.py`.
- The endpoint false-zero fix in `cm12RetainedDensityPhiAtFine` (committed).
- The GPU topology memory work (`e78daf19`, `cfde4126`) is unrelated and
  production.

## Experiment: shadow band replay (8 September, late evening)

Purpose: gain confidence in the band-inside-CM12 route before touching the
resident, on production data with production metrics, and settle which mass
coupling to build. No solver code changed.

### Method

1. **Capture.** `tools/capture-retained-visual-ab-dawn.ts` with native CM12
   transport on the quarter scene, every step 0–20, coarse-first and all-fine
   arms: per-step native density, collocated velocity, topology and the
   shipping mesh, under `artifacts/surface-band-replay/quarter/{coarse,fine}/`.
2. **Replay.** `tools/replay-surface-band.py` runs the proposed per-step
   algorithm on the CPU on the 32×24×32 fine lattice against those fields.
   Band seeded from the scene primitives. Velocity taken from cells the solver
   holds at least half full and extended into the rest of the lattice along
   band normals (closest point, upwind Jacobi). RK2 semi-Lagrangian advection
   with tricubic interpolation. Godunov redistancing with the Russo–Smereka
   subcell fix. Then the coupling arm, then a short redistance. The solver's
   fields are read, never written: the band is a shadow and does not feed the
   solver.
3. **Arms**, differing only in the coupling rule. A: none. B: one constant
   shift of `phi` per native cell so the cell's band amount equals the solver
   density, on every active cell (this document's original step 5). C: the
   same shift restricted to native cells the band interface crosses. D: native
   cells the solver holds full (density ≥ 0.995) or empty (≤ 0.005) are filled
   or drained, then one shift per connected liquid body makes the body's amount
   equal the solver mass assigned to it (each fine cell assigned to the body of
   its nearest liquid). The fill/drain test has two forms, `--classes`: the
   afternoon table used `ramp` (fill when the band's sub-cell fraction of the
   native cell is below 0.995, drain when above 0.005); since 9 September the
   default is `topology` (fill when the band's fraction is below 0.5, i.e. the
   cell's centre is in air while the solver holds it full, and the reverse for
   drains). The two are identical before contact on the afternoon capture
   (no fills or drains until step 7); on HEAD's sharp density only the sign
   test works, see "Replay on the HEAD captures" below.
4. **Amount model.** The retained seed is the ramp `q = clamp(0.5 − phi/w)`
   with `w` one fine cell, so a cell's band amount is that ramp averaged over
   4³ tricubic sub-samples of `phi` per fine cell. Matching sharp plane-cut
   volumes instead manufactures a 6 mm double surface on a resting pool; that
   was checked and rejected.
5. **Metrics**, unchanged from the current-map analysis
   (`tools/analyze-current-map-ray-field.py`): half-level roots of the
   trilinear presentation field along 134 antipodal rays from the discrete
   free-fall centre, pool height on a 17×17 grid. Valid before contact only;
   contact falls between steps 7 and 8. The same rays are applied to the native
   density's trilinear contour and to the shipping mesh vertices.
6. **Floor.** An exact sphere distance field of radius five cells sampled at
   cell centres and rooted trilinearly has 1.76 mm RMS / 2.5 mm max radial
   error; rooted tricubically, 0.68 mm. One redistance adds about 0.14 mm. No
   number below is to be judged against zero.

### Results

Sphere radial error against the discrete reference, RMS / max in mm,
coarse-first arm. The all-fine arm agrees within 0.1 mm before contact: the
band's fine lattice makes the native rung invisible to the surface. The
capture behind this table was taken in native mode on the tree before
`288d922a`; WP0's HEAD baseline differs from it from step 0 (see the status
log), and the same replay on the HEAD captures is recorded there when it
completes.

| Step | Gap to pool (mm) | Native density contour | Shipping mesh | A: pure band | C: per-cell shift at interface | D: fill/drain + per-body shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 263 | 3.23 / 6.4 | 1.85 / 3.8 | 1.90 / 3.3 | 1.86 / 3.4 | 2.23 / 3.6 |
| 2 | 252 | 3.67 / 6.8 | 2.89 / 8.3 | 1.66 / 3.6 | 3.04 / 5.4 | 2.34 / 4.5 |
| 3 | 230 | 4.19 / 7.2 | 12.27 / 39.9 | 1.57 / 5.3 | 3.15 / 5.6 | 2.44 / 6.1 |
| 4 | 197 | 4.60 / 8.4 | 12.19 / 27.6 | 1.57 / 5.2 | 4.65 / 13.1 | 2.66 / 6.4 |
| 5 | 154 | 4.74 / 8.9 | 15.66 / 32.9 | 1.50 / 5.0 | 4.35 / 6.8 | 2.79 / 6.5 |
| 6 | 99 | 4.84 / 9.2 | 15.10 / 36.6 | 1.88 / 8.7 | 4.78 / 9.4 | 3.29 / 14.0 |
| 7 | 34 | 5.19 / 10.1 | 14.31 / 37.2 | 2.36 / 12.3 | 5.01 / 10.4 | 3.51 / 13.4 |

Pool height error stays within 0.5 mm for A and D through step 5 in both
arms, against 1.9–2.0 mm of grid imprint in the coarse arm's native contour;
at the two-cell gap of step 6 it is 1.0 mm (A) and 6.5 mm (D) under the
sphere.
The solver itself, measured per native cell against the reference, drifts
from 0.05 to 0.21 of a cell (max) over steps 1–7, with 685 ml of mass outside
the reference geometry by step 5; that is the transport smear of the section
above, now as a per-cell number.

Mass agreement after impact, band amount minus solver amount, litres out of
1,090, coarse arm:

| Step | A: none | B: per-cell, all cells | C: per-cell, interface | D: per-body |
| --- | ---: | ---: | ---: | ---: |
| 8 | −2.2 | +3.6 | 0.0 | 0.00 |
| 10 | −14.3 | −16.9 | −8.4 | 0.00 |
| 12 | −20.3 | −11.3 | −4.4 | 0.00 |
| 16 | −28.1 | −12.1 | −3.5 | 0.00 |
| 20 | −32.2 | +8.2 | +14.0 | 0.00 |

All-fine arm at step 20: −24.2 / +25.0 / +26.9 / 0.00.

Arm C's per-cell shifts reach 16–25 mm from step 2 on. Arm D's per-body
shifts stay under 2.5 mm; it fills up to 166 native cells per step after
contact, drains six cells once, and sees one liquid body from step 7. The
false air pocket the pure band carried under the pool at step 12 before the
redistance guards is gone in every arm with them. Composite sections:
`artifacts/surface-band-replay/quarter/coarse/replay-normal/composite.png`
(rows steps 3, 5, 7, 8, 12, 16; native contour white, A cyan, C green, D
magenta, reference yellow before contact). Two earlier runs are kept beside
it: `replay-normal-unguarded/` (before the redistance guards of finding 4)
and `replay-normal-shift-before-redistance/` (guards in, body shift not yet
last; finding 3).

#### Replay on the HEAD captures (9 September, early)

The same replay on WP0's HEAD baseline (`ec5af4bb`, both arms, steps 0–30,
`artifacts/surface-band/baseline/{coarse,fine}/`). Sphere RMS / max in mm
against the analytic reference; the D columns are the coupled band under the
two fill/drain class tests (coarse arm).

| Step | Gap (mm) | A: pure band, coarse | A: fine | D: sign classes (`topology`) | D: sub-cell classes (`ramp`, the afternoon rule) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 263 | 1.90 / 3.3 | 1.90 / 3.3 | 3.94 / 5.3 | 6.61 / 12.6 |
| 2 | 252 | 1.66 / 3.6 | 1.63 / 3.6 | 3.93 / 6.5 | 6.79 / 13.1 |
| 3 | 230 | 1.57 / 5.3 | 1.52 / 5.3 | 4.04 / 7.7 | 7.12 / 14.2 |
| 4 | 197 | 1.57 / 5.2 | 1.53 / 5.2 | 4.21 / 9.1 | 7.84 / 16.1 |
| 5 | 154 | 1.50 / 5.0 | 1.48 / 4.9 | 4.27 / 9.2 | 14.00 / 94.5 |
| 6 | 99 | 1.88 / 8.7 | 1.87 / 8.7 | 4.54 / 14.5 | 13.92 / 90.4 |
| 7 | 34 | 2.36 / 12.1 | 2.36 / 12.1 | 4.76 / 17.1 | 21.31 / 73.0 |

- **Arm A is the afternoon result**: coarse within 0.01 mm RMS through
  step 7 and 0.02 mm in the max through step 6 (0.13 mm at step 7, where the
  pool rises under the sphere), fine within 0.05 / 0.09 mm (afternoon:
  1.90 / 3.3 … 2.36 / 12.3). The velocity the band advects with did not
  change between the two builds; the density did. Pool max error 0.000–0.013
  mm through step 6, 5.6 mm at step 7 (the pool has risen under the sphere).
- **The afternoon fill/drain rule fails on HEAD.** HEAD's density is sharp:
  the pool rows below y = 0.4 m are exactly 1.0 and the rows above exactly
  0.0 (the afternoon capture had 0.965 / 0.07). Under the ramp amount model
  (width one cell) the band's fraction of the row below the face is 0.875 and
  of the row above 0.125, so "solver full but band below 0.995" and "solver
  empty but band above 0.005" fired on 152–1085 fills and 212–536 drains per
  step in the coarse arm (1024–1160 each in the fine arm: the whole pool
  surface), pushed the pool body to 1,307 L before the shift and the shift
  dragged it back by 74.5 mm (coarse) / 27.5 mm (fine). Against the
  reference the solver's full/empty classes are correct in every native cell
  at steps 0 and 1; the disagreement was between the two amount models, not
  a solver error.
- **Sign classes fix it.** Fills and drains 0 through step 6, pool max error
  0.000–0.014 mm, pool shift −0.00 to −0.16 mm, band minus solver 0.00 L
  (≤ 0.6 ml) through step 8; at step 7 (gap 34 mm) 4.76 / 17.1 with the pool
  8.3 mm off under the sphere, two bodies until step 8. The sphere body is shifted 2.65 mm inward at step 1 and
  0.3–0.7 mm per step after: the solver's sphere holds 64.4 L at rest against
  the analytic 65.45 L (the pool is 1,024.0 L exactly), an initialisation
  deficit of 1.6 % that the band's coupling reports as a 1.3 mm smaller
  radius, and the pure band gains 0.2–0.3 L per step that the coupling
  removes. Read D's 3.9–4.5 mm as 1.5–1.9 mm of band error plus that bias;
  C3.3 is judged against the sphere of the solver's own sphere-body volume
  for this reason.
- **A sharp amount model for the shift is rejected.** `--width 0` (plane
  indicator over the 4³ sub-samples) also gives zero fills, but the pool's
  amount then changes in 32 L quanta (one sub-sample layer, 12.5 mm) and the
  body bisection lands on a step: pool shifts of ±6.25 / +12.5 mm, pool error
  6–35 mm, 32 L of mass disagreement at step 1. The shift needs the
  continuous ramp; the class test needs the sign.

Mass agreement after impact on the HEAD captures, band amount minus solver
amount in litres, sign-class run (the `ramp` run's D column also held 0.00 L:
the shift always closes the body's mass; it was the geometry it destroyed):

| Step | A: none, coarse | A: none, fine | D: per body, coarse | D: per body, fine |
| --- | ---: | ---: | ---: | ---: |
| 8 | −0.8 | −0.8 | 0.00 | 0.00 |
| 10 | −13.2 | −13.1 | 0.00 | 0.00 |
| 12 | −19.0 | −17.7 | 0.00 | 0.00 |
| 16 | −25.8 | −22.5 | 0.00 | 0.00 |
| 20 | −30.0 | −23.0 | 0.00 | 0.00 |
| 30 | −35.9 | −29.6 | 0.00 | 0.00 |

Arm D's largest shift after step 1 is 2.10 mm (step 9, both arms); bodies are
2 through step 7 and 1 from step 8; it drains nothing in 30 steps and fills
16 native cells at step 10 and 2–28 per step over steps 23–28, the pool's
splash settling. The afternoon rule on the afternoon capture filled up to
166 per step after contact; the difference is the class test, not the
capture.

### What it establishes

1. **The band as geometry authority works, at the floor.** Through step 6
   the pure band sits at 1.5–1.9 mm RMS against a 1.76 mm trilinear floor
   (2.4 mm at the 34 mm gap of step 7): three times better than the native
   density contour and eight times better than the shipping mesh from step 3
   on, identical in both arms. The "under 1 mm"
   gate in the first draft of this document was below the floor; the gates
   are restated below.
2. **The per-native-cell volume shift must not be built.** Arms B and C
   degrade the sphere from step 2 (3–4.7 mm RMS, 13 mm max) with shifts of
   16–25 mm, because the solver's transport error is a per-cell fact and a
   per-cell constraint copies it into the geometry the band exists to keep
   clean. Checked against the amount model: with ramp width h, h/2 and 0 the
   C numbers do not move. After impact B and C also drift in total (+8 and
   +14 L at step 20 coarse, +25 and +27 L fine) because interface cells alone
   cannot absorb bulk mismatch.
3. **The per-body rule is the coupling to build.** Arm D holds total mass to
   0.00 L (exact to the bisection) through the impact with body shifts under 2.5 mm, costs
   0.3–1.3 mm over the pure band before contact, and keeps the pool flat
   through step 5 (6.5 mm under the sphere at the two-cell gap of step 6,
   against 1.0 mm for the pure band; the per-body machinery widens the
   contact transient and that is the first thing to trim in M2).
   Topology change enters through the solver's saturated cells (fill and
   drain); the band never decides a merge or a pinch itself. The shift must be
   the last operation of the step: a run with a redistance after it moved the
   interface by 0.3–2.5 mm and let the mass disagree by up to 6.9 L (0.6 %)
   at step 11; re-applying the shift restored 0.00 L at every step checked. A
   constant shift preserves the distance property, so nothing needs to follow
   it.
4. **The subcell fix needs three guards near a closing gap.** Without them
   the two steps before contact showed a one-step transient: band error
   9.2 mm RMS / 218 mm max at step 6, sphere bottom 20 mm high, pool 7–18 mm
   off under it. Cause, verified cell by cell: the air cell between the bodies
   keeps the distance to the body it was nearest to (75 mm) while the other
   body arrives (true distance 26 mm); the Russo–Smereka fix pins that stale
   value and, at a sub-cell gap, divides by a central difference that cancels
   on the ridge between the bodies (D up to 1 m). The guards, now in the replay
   script: one-sided differences on an axis that crosses the interface in both
   directions; |D| ≤ h for any cell with a face neighbour across the interface;
   an air-side cell whose value differs from its across-interface neighbour by
   more than 1.5 cells is stale and is rebuilt from the liquid side instead of
   pinned. Band error at step 6 becomes 1.4 / 8 mm and at step 7 2.1 / 25,
   steps 1–5 unchanged; in the ray metric the pure band at steps 6 and 7
   goes from 3.6 / 20.7 and 12.2 / 47.9 mm to 1.9 / 8.7 and 2.4 / 12.3.
   Rejected alternatives, each measured on the same chain: pinning the liquid
   side only (2.3–2.7 mm RMS drift before contact), a per-axis max-of-
   differences denominator (3.1 mm by step 5), the planar subcell distance
   (4.5 mm by step 5), and an approach-biased velocity extension (sinks the
   pool under the sphere by 24 mm).
5. **Presentation should root tricubically.** Same samples, 0.68 mm against
   1.76 mm; a presentation-side change independent of the band.

### What it does not show

- The band did not feed the solver. Pressure classification, sharpening
  direction and the seed still came from the density, so the feedback half of
  the design (step 6 below) and the solver's own 685 ml of pre-contact smear
  are untested.
- Twenty steps of one scene; no solids, rigid bodies, inflow or live edits.
- CPU float64 on the dense fine lattice with 90 extension iterations. The
  resident version is banded and f32; band width and sweep counts are to be
  re-established there.
- The reference is only valid before contact. After impact the comparison is
  between arms and against mass, not against a known shape.

## Recommended design: a distance band inside CM12

Sussman and Puckett's coupled level set / volume of fluid (JCP 162, 2000) is
the standard answer to "conserve mass like VOF, look like a level set".
Basilisk's `two-phase-clsvof.h` is a compact modern implementation to read.
The adaptation here uses CM12's density where CLSVOF uses a VOF fraction.

### State

One f32 signed distance `phi` per fine support, in finest-cell units, stored
as its own region of the resident `state` buffer (group 0, binding 2),
appended after the layout's last region. The shipping `native-cm12` mode builds
no retained arena (`288d922a`), so the band cannot borrow the retained words;
a region of its own costs no new storage binding, which matters because the
resident bind group layout already uses its ten. Two planes (accepted and
scratch) over the whole fine lattice: 2 × 24,576 words on the quarter scene,
about 12.6 MB on the full pool. Storage is dense; work is not. Only supports
within a band of about five fine cells of the interface are computed, from a
compacted list rebuilt each step. Everything outside the band is a sign plus a
clamp.

The band lives on the finest lattice regardless of the native rung. A B1 pool
cell contains 512 band-capable fine supports; a flat pool at rest costs one
plane of them. This is what removes the grid imprint: the surface is never
reconstructed from coarse means again.

### Per step, in the transport band

1. Freeze the VEX plane before gather, as the current map already does.
2. Extend velocity into the air side of the band by normal extrapolation
   (upwind sweeps of `∂u/∂τ + sign(phi) ∇phi·∇u = 0`), taking velocity only
   from cells the solver holds at least half full, so each body carries its
   own velocity into the air it is about to occupy. This is the fix for the
   4.3 mm shortfall and the replay confirms it. Do not bias the extension
   toward the approaching body; that variant sinks the pool under the sphere.
3. Advect `phi` semi-Lagrangian on the fine lattice with the extended
   velocity. Cubic or BFECC; RK2 backtrace with the substep schedule the
   current map already computes.
4. Redistance with a handful of upwind Eikonal sweeps inside the band, with
   the Russo–Smereka subcell fix and its three guards (experiment finding 4).
   `redistance` in `tools/replay-surface-band.py` is the reference.
5. **Mass coupling, per body** (experiment arm D). Two parts. (a) Bulk fill
   and drain: a native cell the solver holds at or above 0.995 **whose band
   fraction is below 0.5** (its centre is in air) becomes liquid in the band,
   and a cell at or below 0.005 whose band fraction is above 0.5 becomes air;
   the following redistance rebuilds distances. The test is a sign test, never
   a sub-cell one: a sharp density holds every interface-adjacent cell at
   exactly 1.0 or 0.0 and a sub-cell test then fires on the whole surface
   every step (HEAD replay, 9 September). This is how topology change enters
   the band: merges and pinches come from the solver's conserved mass. (b) One constant
   shift of `phi` per connected liquid body so the body's amount, under the
   seed's ramp `clamp(0.5 - phi/w)` over the fine cells assigned to it, equals
   the solver mass assigned to it. Bodies are the connected components of
   `{phi < 0}`. Order within the step: fill/drain, redistance, body shift
   last; nothing follows the shift (experiment finding 3). The replay saw
   shifts under 2 mm and 0.00 L (exact to the bisection) total mass agreement through the impact.
   Do **not** shift per native cell (experiment finding 2). The amount the
   solver conserves stays the seed ramp; `q` is not replaced by a sharp
   indicator.
6. **Sharpening toward the band.** Point §3.5 sharpening's `TraceAlongField`
   at the band's `phi` rather than `∇ρ`, so the density converges to the
   band's ramp instead of the band being dragged to the density. This is the
   half of the coupling the shadow replay could not test; it is what removes
   the 685 ml the solver has smeared outside the reference by step 5.
7. Publish `phi` to the presentation at the four sites that call
   `cm12RetainedDensityPhiAtFine` behind `cm12RetainedDensityEnabled()` today
   (page fill, accepted surface proof, demotion proof, level-set publication).
   The band takes its own branch ahead of the retained one, so it works in the
   shipping `native-cm12` mode. Pressure classification does not consume that
   function today: `classifyPressureRow` classifies on the transported
   density, and moving it to the band is WP5 work. Root the presentation
   tricubically: same samples, 0.68 mm instead of 1.76 mm on an exact sphere.

### Cost model, and why this is not Losasso

The band is proportional to interface area times band width. Quarter scene:
about 5k pool supports plus 1.5k sphere supports. Full pool: about 80k. Each
step is roughly one advection pass, four to eight extension sweeps, four to
eight redistance sweeps, one volume pass and one correction pass, all over the
band only, all on the flat fine lattice with no octree, no multigrid, no
separate topology and no per-step allocation. That is on the order of twenty
launches on top of the roughly five hundred the frame already issues. The
Losasso cost was its octree pressure and V-cycle machinery; nothing here
touches pressure structure.

### What it does not fix, said plainly

- It is two coupled fields with a constraint, not one field. The constraint
  holds per native cell to Newton tolerance.
- CM12's diffuse density halo outside the band still exists until step 6
  lands. Until then the per-body shift absorbs the halo as a whole-body volume
  error of well under a millimetre, which is what the replay measured.
- The per-body shift is one number per body, not a sub-cell redistribution.
  That is correct: the band already holds the fine geometry; the shift is only
  the mass constraint. Where the solver puts mass in the wrong cell the band
  does not follow it; sharpening toward the band is what corrects the solver.
- Two bodies closer than one fine cell share one velocity cell. With the
  redistance guards the residual at a 34 mm gap is 2.1 mm RMS / 25 mm max in
  the band for the one step before contact; the merge erases it.
- Creases round off at the fine-cell scale. The seed can be initialised sharp
  and stays sharp to within a fine cell. That is the accepted limit.
- Coarse-region physics stays coarse. The coarse arm's asymmetry and weaker
  splash are pressure and transport resolution errors that this does not
  address; the coarse-first refinement policy owns them.

## Execution plan

Six work packages, in order. Each names its files and seams (line numbers are
at `ec5af4bb`), its steps, its tests with the exact command, and the tracker
criteria it must turn green to exit. Rules that bind every package:

- **The scene is the judge.** Every number comes from `coarse-first-pool-impact-quarter`
  through the commands in this document; a CPU oracle exists only where it
  gates one of those numbers.
- **Option ladder.** A new method option `surfaceBand: "off" | "static" |
  "advected" | "coupled"` in `lib/methods/adaptive-mass/method.ts` beside
  `densityTransport` (:78–90; `tier: "coarse"`, `update: "solver"`), threaded
  through `webgpu-adaptive-mass-solver.ts` where `densityTransport` is read
  (:612, :672) into the resident constructor. Independent of `densityTransport`;
  default `off` until WP3 exits, when the quarter scene profile
  (`lib/core/scenes.ts:2628–2639`) sets `coupled`. Each package raises the
  value it makes work; the app exposes only values whose package has exited.
- **No ceiling moves.** The ten storage bindings of the resident layout
  (`webgpu-sparse-cm12-resident.ts:793–810`) stay ten; the canonical lanes'
  180 s budget and the mini32 40 ms / mini64 50 ms perf ceilings stay; no
  existing test threshold is raised.
- **Gate discipline.** `npm run test:dawn:sparse-cm12` green (17 lanes, no
  timeouts) before a package is marked done. Serial Dawn under the repository
  lease; one agent on the resident at a time.
- **Not Losasso.** No kernel, band, redistance or octree code from the Losasso
  lane is reused or cited.

### WP0 — Measurement first (no solver change)

Purpose: one command turns a capture into the tracker's numbers, proven
against the replay before any GPU number exists. This is what makes "success
criteria tracked" mechanical rather than narrative.

Files:

- New `tools/surface-band-gate.py`. Imports `load_configuration`, `load_step`,
  `native_layout`, `surface_metrics`, `mesh_metrics`, `body_volumes`,
  `amount_field` and `reference_fraction` from `tools/replay-surface-band.py`
  (no second implementation of the metric). Reads a capture directory as
  written by `tools/capture-retained-visual-ab-dawn.ts` (per step:
  `density.bin`, `velocity.bin`, `activity.json`, `mesh.bin`, `receipt.json`)
  and, when present, `phi.bin` per step (WP2 adds it). Writes `gate.json` and
  prints one line per step: sphere radial RMS / max in mm for the band
  (trilinear half-level roots), the native contour and the shipping mesh; pool
  RMS / max on the 17×17 grid; band amount minus solver mass in litres; body
  count and largest body shift (from the receipt, WP3 on); solver mass outside
  the reference and per-cell fraction error max (WP4 criteria). The mesh pool
  column keeps only vertices two cells inside the walls, because the shipping
  mesh's wall faces carry vertices below the waterline (23 mm RMS otherwise).
  Flags: `--steps`, `--band=none|phi|<prefix>` (`replay-normal/phi-A` reads
  the replay's arm-A phi), `--limits=<json>`, `--criteria=C2.4,C3.1` (exit
  code 1 when an evaluated rule fails; with `--criteria` a missing series
  fails too), `--out`.
- New `tests/surface-band-limits.json`: the tracker thresholds as data, keyed
  by criterion id. Thresholds live in one place; the tracker below quotes it.
- New `tests/surface-band-gate.test.py` (`unittest`, like
  `tests/current-map-ray-field.test.py`): runs the gate on
  `artifacts/surface-band-replay/quarter/coarse` with
  `--band=replay-normal/phi-A` and asserts the step 1–7 columns of the results
  table above (band A 1.90/3.3 … 2.36/12.3; native contour 3.23/6.4 … 5.19/10.1;
  mesh 1.85/3.8 … 14.31/37.2) within 0.05 mm, and with `phi-D` the arm-D
  column and 0.00 L mass agreement at steps 8–20.
- Baseline capture on the `288d922a` tree, both arms, steps 0–30 (the
  reproduction command with `--out=artifacts/surface-band/baseline/{coarse,fine}`).
  Its native-contour and mesh numbers are the tracker's baseline column. They
  should equal the replay capture's; a difference is a finding to record in
  the status log, not a tolerance to absorb.
- Canonical gate at HEAD, lane by lane, before any solver change. The
  `288d922a` run recorded five passes, one failure (`symmetric-expansion`, its
  D4 density assertion, which an untouched `57b6ae39` checkout also fails),
  five timeouts and six lanes never run. A red lane at HEAD is a prerequisite
  fix tracked in the status log with its owner, not a tolerance for the band
  packages: no package exits while the gate is red, whoever made it red.

Tests:

```sh
python3.11 tests/surface-band-gate.test.py
python3.11 tools/surface-band-gate.py artifacts/surface-band/baseline/coarse \
  --steps=0,1,2,3,4,5,6,7 --band=none --out=artifacts/surface-band/baseline/coarse/gate.json
```

Exit: C0.1, C0.2, C0.3 recorded. No file under `lib/` changes.

### WP1 (M0) — Static band as geometry authority

Purpose: the band exists in the resident, is seeded from the scene, and the
presentation reads it. No transport. This alone removes the grid imprint that
started the program on 7 September, with no physics risk. `static` is a test
mode: from step 1 the liquid moves and the band does not.

Files and seams:

- `webgpu-sparse-cm12-resident.ts` arena layout (:4302–4347): a
  `surfaceBandLayout = { baseWords, fineCount, controlWords: 16 }` appended
  after the last region in use (`currentMapEndWords`, else the rigid words
  after `retainedRigidBase`, else `layout.floatCount`); `state` (:4345) is
  sized to include `16 + 2 * fineCount` words. Control words: enabled, band
  width in cells, generation, body count, fills, drains, reserved. Layout
  constants reach WGSL the way `CM12_RETAINED_FIELD_BASE` does.
- Seed on the CPU: `compileRetainedSceneDensity(scene)`
  (`sparse-cm12-retained-scene-density.ts:134`) for the primitives and
  `evaluateRetainedScenePhi` (:183) at every fine-cell centre
  `origin + (q + 0.5)·h`, in finest-cell units, uploaded through the existing
  `seed(floatOffset, values)` helper (:4348). The compiler stays the
  initialiser; the band does not need the retained arena to exist.
- `webgpu-sparse-cm12-resident.wgsl.ts`: `cm12SurfaceBandEnabled()` and
  `cm12SurfaceBandPhiAtFine(point) -> f32` (point in finest-cell units,
  trilinear over the eight surrounding fine centres, clamped to the lattice,
  result in metres via `p.frame.y`; `+4h` outside the lattice like the
  existing air value), with stubs when the layout is undefined, following the
  retained stub pattern at :596–606. At the four presentation sites insert
  `if(cm12SurfaceBandEnabled()){…}` ahead of `else if(cm12RetainedDensityEnabled())`:
  page fill :10905, `surfaceProofAcceptedPhi` :11041, demotion proof :11250,
  `publishSparseLevelSet` :11457. The solid clip
  (`cm12SolidVoxelFractionQ8(q)>=255u`) is kept at each site. Native
  reconstruction functions in `sparse-cm12-native-surface.wgsl.ts` are not
  touched; their fingerprint test stays green.
- `tools/probe-coarse-surface-grid-imprint-dawn.ts` and
  `tools/capture-retained-visual-ab-dawn.ts` gain `--surface-band=<mode>`.

Tests:

```sh
# CPU
node --import tsx --test tests/sparse-cm12-surface-band-seed.test.ts      # new
node --import tsx --test tests/sparse-cm12-advance-partition.test.ts      # unchanged, no new dispatch
node --import tsx --test tests/sparse-cm12-native-surface-baseline.test.ts
npm run check:sparse-cm12:wgsl          # band off, static, and retained mode all compile
# Dawn
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx --test --test-concurrency=1 tests/sparse-cm12-surface-grid-imprint-dawn.test.ts
# static band, both arms, steps 0,6,15 -> density/velocity bit-identical to baseline
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/capture-retained-visual-ab-dawn.ts --arm=coarse --surface-band=static \
  --steps=0,6,15 --out=artifacts/surface-band/wp1/coarse
python3.11 tools/surface-band-gate.py artifacts/surface-band/wp1/coarse --steps=0 --band=none
npm run test:dawn:sparse-cm12
```

`tests/sparse-cm12-surface-band-seed.test.ts` (new): the uploaded seed equals
`evaluateRetainedScenePhi` at every fine centre of the quarter scene to
1e-6 h; the pool plane gives phi = 0 at y = 0.4 m; a centre inside the sphere
is negative; `baseWords` lies past every existing region and `state.size`
grows by exactly the band words in each of the three `densityTransport`
modes. The imprint test runs its `fixed4`, `mixed` and `adaptive` arms with
`surfaceBand: static` at its existing budget `0.25 × 0.006`.

Exit: C1.1–C1.4.

### WP2 (M1) — Band transport (replay arm A on the GPU)

Purpose: design steps 1–4 per step, with the replay as the oracle for every
kernel and for the whole chain.

Placement: a new stage `surface-band-transport` after
`transport-velocity-extension` and before `face-preparation` in
`SPARSE_CM12_RESIDENT_STAGES` (`webgpu-sparse-cm12-resident.ts:375–393`),
sub-seams `band-velocity-extension`, `band-advection`, `band-redistance`,
registered in `sparse-cm12-stages.ts` (the `satisfies` contract makes a
missing entry a type error) and in the substage table;
`tests/sparse-cm12-advance-partition.test.ts` enforces order and the seam
list. Why the frame head: the density of step n is moved by the velocity
projected at step n−1, so the band must move with that same velocity in the
same step for phi_n and rho_n to describe one instant; the replay advanced
phi with the captured step n−1 velocity and that is the chain the oracle
numbers belong to.

Velocity source: the collocated cell velocity `collocateAndDiagnose`
(`.wgsl.ts:7176`) writes at `destinationCellVelocity()` during the previous
step's projection; at the frame head the banks have swapped and the stage
reads it through `sourceCellVelocity()` (:2262), masked by the density wet
test `sourceDensity() > CM12_LIQUID_ISOVALUE` (0.5 in
`lib/core/cm12-numerics.ts`). That is "velocity of cells the solver holds at
least half full", the field the replay consumed, read with no extra copy. Do
not read the effective-transport-velocity plane instead: VEX sweep 8 has
already overwritten it with air-extended values by the time this stage runs,
and the extension must seed from wet cells only. Fine supports find their
native cell through `presentationCompiledOwnerCellAt(q)` (`.wgsl.ts:3215`).
Velocities are in finest cells per second (`readDiagnosticFields` scales by
`parameterF32[41]`).

Kernels, in a new `lib/methods/adaptive-mass/sparse-cm12-surface-band.wgsl.ts`
included by the resident the way `sparse-cm12-native-surface.wgsl.ts` is,
each a transcription of the named replay function:

1. `bandCompact`: lists the fine cells with |phi| ≤ B (B = 5 cells to
   start, trimmed to what holds C2.2) per brick, rebuilt only for bricks the
   incremental activity masks mark dirty this step (the masks
   `activity-measurement` publishes at `.ts:6992`, `markIncrementalActivityScalarBricks`
   → `finalizeIncrementalActivityMasks`). The frame's active list is the
   dirty bricks' entries; the full list is every brick's entries. Kernels 2–4
   dispatch over the active list (`dispatchWorkgroupsIndirect`, the pattern
   at `.ts:6484`). With zero velocity the advection is the identity and the
   redistance is at its fixed point, so skipping still bricks is exact, not
   approximate; this is what keeps the band inside the paused and frozen
   region cost model instead of fighting it. Land the dense early-out form
   first for correctness, then the lists in the same package; C2.6–C2.8 are
   measured on the lists.
2. `bandExtendVelocity` × N: upwind Jacobi closest-point extension
   `∂u/∂τ + sign(phi) ∇phi·∇u = 0` (`extend_velocity`, `mode='normal'`,
   :204). The replay used 90 iterations on the dense lattice; on the GPU
   start at 12 sweeps and reduce while C2.2 holds.
3. `bandAdvect`: RK2 backtrace, tricubic phi sampling, dt = paper step
   (`advect` with `lattice.tricubic`, :249); writes the scratch plane.
4. `bandRedistance` × M: Godunov Hamiltonian, Russo–Smereka subcell fix, the
   three guards, exactly `redistance` (:258–311): one-sided differences on a
   doubly crossing axis, |D| ≤ h, stale air cells beyond 1.5h rebuilt from
   the liquid side. Reference 24 iterations at dtau 0.4; start there.

Readback: `readDiagnosticFields` (`.ts:9143`) returns `phi` (Float32Array,
fine lattice, metres) when the band is on; the capture tool writes it as
`phi.bin` per step; the gate reads it with `--band=phi`.

Tests:

```sh
# CPU fixture from the replay module (16^3 lattice, one step of each kernel)
python3.11 tests/surface-band-oracle.test.py          # writes tests/fixtures/surface-band-oracle-16.npz
# Dawn: kernel-level and scene-level oracle
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx --test --test-concurrency=1 tests/sparse-cm12-surface-band-dawn.test.ts
# imposed half-cell translation with the band advected
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/capture-retained-imposed-flow-dawn.ts --surface-band=advected --assert-continuity
# real gravity, both arms, through the gate
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/capture-retained-visual-ab-dawn.ts --arm=coarse --surface-band=advected \
  --steps=0,1,2,3,4,5,6,7 --out=artifacts/surface-band/wp2/coarse    # and --arm=fine
python3.11 tools/surface-band-gate.py artifacts/surface-band/wp2/coarse --band=phi \
  --limits=tests/surface-band-limits.json
# cost and launches
node --import tsx tools/probe-sparse-cm12-stage-cost.ts     # stage surface-band-transport, mini32 and mini64, 3 captures
npm run check:sparse-cm12:stage-timing
npm run test:dawn:sparse-cm12
```

`tests/sparse-cm12-surface-band-dawn.test.ts` (new; manifest lane
`surface-band-transport`, 60 s, also listed in `test:dawn:sparse-cm12:pool-impact`):
(a) each kernel on the fixture within 1e-4 h of the replay module (f32
against f64); (b) quarter scene, `advected`, steps 0–7, GPU phi against the
replay run on the HEAD capture,
`artifacts/surface-band/baseline/coarse/replay-normal/phi-A-step-<n>.npy`
(the afternoon capture's replay under `artifacts/surface-band-replay/` is a
different solver state and is not the oracle for HEAD).

Exit: C2.1–C2.8. Step 7 is reported, not gated.

### WP3 (M2) — Mass coupling per body (replay arm D)

Purpose: design step 5 as a second stage `surface-band-coupling` after
`surface-sharpening` and before `scalar-publication`, because it needs the
step's final density, with the order fill/drain → redistance → body shift and
nothing after the shift (experiment finding 3).

Kernels (same file), each a transcription of `correct_bodies` (:463) and
`label_bodies` (:427):

1. `bandFillDrain`: a native cell at density ≥ 0.995 whose band fraction is
   below 0.5 (the band puts its centre in air) sets its fine phi to −h, a
   cell at ≤ 0.005 whose band fraction is above 0.5 sets it to +h (the
   following redistance rebuilds true distances); a cell the band already has
   on the solver's side is never touched, whatever its sub-cell fraction
   (`--classes topology` in the replay, `correct_bodies`). Fills and drains
   counted into the control words and copied into the receipt; on the quarter
   scene both are 0 before contact.
2. `bandRedistance` × 12 (WP2 kernel).
3. `bandLabelBodies`: connected components of {phi < 0} by min-label
   propagation over the six neighbours, run to a no-change flag with a cap of
   64 sweeps; every band cell then takes the label of its nearest liquid cell
   (propagate through air inside the band). Up to 16 bodies in the control
   block; the quarter scene has two before contact and one after. Bodies past
   the sixteenth are counted, unshifted and reported (C3.5).
4. `bandBodyAmounts`: per body, band amount as the seed ramp
   `clamp(0.5 − phi/w)` over 4³ tricubic sub-samples per fine cell
   (`amount_samples`, `amount_from_samples`, :337/:350) and solver mass from
   the native densities assigned to it, by per-body atomic adds.
5. `bandBodyShift`: the constant shift per body that equates the two. The
   replay bisects for 40 rounds; the GPU uses Newton on the ramp amount
   (analytic derivative, 4–6 rounds of amount → reduce → update) and must land
   within 1e-3 L of the bisection; if it does not at a merge, twelve
   bisection rounds replace it. The shift is applied to the body's cells on
   the full band list, the one pass of the step that is bounded by band size
   rather than by activity; a body whose shift is zero (a still body, mass
   unchanged) is skipped, so a resting pool writes nothing. If that pass shows
   in the stage cost, the fallback is one offset per body applied at the read
   seam instead of a write. The shift is the last write of the stage. Largest
   shift, body count and active band cells go to the receipt.

Tests:

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx --test --test-concurrency=1 tests/sparse-cm12-surface-band-dawn.test.ts   # part (c): coupled, steps 0-20 vs replay-topology/phi-D
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/capture-retained-visual-ab-dawn.ts --arm=coarse --surface-band=coupled \
  --steps=0,1,2,3,4,5,6,7,8,10,12,16,20,30 --out=artifacts/surface-band/wp3/coarse   # and --arm=fine
python3.11 tools/surface-band-gate.py artifacts/surface-band/wp3/coarse --band=phi \
  --limits=tests/surface-band-limits.json
node --import tsx tools/probe-sparse-cm12-stage-cost.ts
npm run test:dawn:sparse-cm12      # with the quarter profile now defaulting to coupled
```

Peter's check in the app at 0.2 / 0.5 / 1 s against the baseline sections is
recorded in the status log; it is not a numeric criterion.

Exit: C3.1–C3.7. The quarter scene profile switches to `coupled` in the
commit that closes this package, not before.

### WP4 (M2b) — Density follows the band

Purpose: design step 6, the half of the coupling the replay could not test.
`traceSharpeningMass` (`.wgsl.ts:5418`, direction at :5430–5433) takes its
direction from the band, −∇phi sampled at the source position, and its target
contour from phi = 0, when the band is `coupled`; `sampleSharpeningField`
(:5390) gains the band branch. The sharpening stage's dispatch shape
(`.ts:6642–6698`) does not change.

Tests: the WP3 capture commands re-run into `artifacts/surface-band/wp4/`,
the gate's two solver columns (mass outside the reference, per-cell fraction
error max, both from `reference_fraction` :530 as the replay computed them),
plus the WP2/WP3 band criteria re-checked on the same capture, plus the
canonical gate.

Exit: C4.1–C4.4. The C4.1/C4.2 thresholds are provisional; fix them from the
first WP3 capture and record the change in the status log.

### WP5 (M3) — Physics seams

Purpose: what the band needs to be the one surface in every mode. Items 1–3
follow WP3; item 4 is independent and can follow WP1.

1. Pressure classification. `classifyPressureRow` (`.wgsl.ts:6513`),
   `classifyPressureCell` (:6109) and `pressureDensity` (:5996) classify on
   the transported density; behind `coupled`, a row or cell is liquid when any
   of its fine phi is negative. Gate: C5.1.
2. Solids and rigid bodies. `bandRedistance` treats a fine cell with
   `cm12SolidVoxelFractionQ8` (:865) at 255 or `cm12RetainedDensityRigidPointOpen`
   (:257) false as air at +h and never as a source; the seed does the same.
   Gate: C5.2.
3. Live edits and inflow. Edits reseed the band region through the retained
   compiler's incremental path (`compileRetainedSceneFineMeans` :503 is the
   pattern, 21 ms today); inflow cells are filled the way `bandFillDrain`
   fills. Gate: C5.3.
4. Tricubic presentation rooting. `lib/core/compact-fine-levelset-phi.ts:136`
   `phi(qi)` is the shared sampler; its consumers
   `webgpu-water-global-fine-tetra.ts` and `webgpu-water-adaptive-mesh.ts`
   are pinned by hash in `tests/sparse-cm12-native-surface-baseline.test.ts`,
   so this lands behind a mesh option with the two pins re-blessed in the same
   commit. Gate: C5.6.
5. Half scene and full pool. Gate: C5.4, C5.5.

```sh
npm run test:dawn:sparse-cm12                       # C5.1-C5.3 are its rigid, inflow and hills lanes
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/capture-retained-visual-ab-dawn.ts --scene=half --arm=coarse --surface-band=coupled \
  --steps=0,1,2,3,4,5,6,7,15,30 --out=artifacts/surface-band/wp5/half
python3.11 tools/surface-band-gate.py artifacts/surface-band/wp5/half --band=phi --limits=tests/surface-band-limits.json
```

### Sequencing, effort, risks

WP0 → WP1 → WP2 → WP3 → WP4 → WP5 (1–3, 5); WP5 item 4 any time after WP1.
Effort, one engineer, serial Dawn: WP0 half a day, WP1 one day, WP2 two to
three days, WP3 two days, WP4 one day, WP5 three days.

- **Body labelling on the GPU** is the least standard kernel. Cap the sweeps,
  report unconverged labels in the receipt; if it stalls, label on the CPU
  from the readback inside the Dawn test only, never in the app path, while
  the kernel matures.
- **Newton against the bisection oracle**: the fallback is written into the
  WP3 kernel list; do not loosen C3.1 instead.
- **Stage cost is bimodal** on this hardware (48–86 ms on identical code has
  been seen); compare medians of three captures, never one.
- **The imprint gate's three arms** are the first thing WP1 can break by
  accident through the demotion-proof site (:11250); the fingerprint test
  guards the native functions, not that call site.
- **Band width and sweep counts** were established on a dense float64
  lattice. The tracker records the GPU values that hold C2.2; they are
  parameters of the plan, not of the design.

## Success criteria tracker

Thresholds are the values in `tests/surface-band-limits.json` (WP0); this
table is their reading copy. "Baseline" was written as the replay capture's
numbers (taken in native mode on the pre-`288d922a` tree the same afternoon)
plus the imprint gate and launch count from the 20 August proportional-cost
audit. WP0 found that HEAD (`ec5af4bb`) does not reproduce that capture: the
restore changed the initial density and the surface, so the HEAD numbers in
C0.2 and the status log are the baseline that later criteria are judged
against (mesh at rest 7.03 / 16.9 mm, native contour 5.82 → 7.72 mm RMS
over steps 1–7, fraction error 0.21 at rest, 1,088.44 L). The replay's
predictions stay valid as predictions of what the band does on a capture;
they were re-derived on the HEAD captures on 9 September: arm A is the
afternoon result to 0.01 mm (WP2's oracle:
`artifacts/surface-band/baseline/{coarse,fine}/replay-normal/phi-A`), arm D
needed the sign-class fill/drain test (WP3's oracle:
`.../replay-topology/phi-D`). The "Replay" column carries the HEAD replay
where it differs from the afternoon one.
"Replay" is the CPU shadow prediction. Update "Current" and "Status" from the
gate's `gate.json` only; statuses are `open`, `green`, `red`.

| ID | Criterion | Metric and command | Threshold | Baseline 2026-09-08 | Replay | Current | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| C0.1 | Gate reproduces the replay | `surface-band-gate.test.py` on `replay-normal/phi-A` and `phi-D`, steps 1–7 | ≤ 0.05 mm on every column | — | exact | gate reproduces A, D, contour, mesh and solver columns to 0.001 mm (`tests/surface-band-gate.test.py`, 7 tests incl. the HEAD class, 34 s; the HEAD arm-D test skips until `replay-topology/replay.json` exists) | green |
| C0.2 | Baseline recaptured | gate on `artifacts/surface-band/baseline/{coarse,fine}`, native contour and mesh, steps 1–7 | equals the replay capture columns | contour 3.23→5.19 RMS; mesh 1.85→15.66 RMS, 39.9 max | same | RED: HEAD differs from the replay capture from step 0. Coarse contour 5.82→7.72 RMS (replay capture 3.23→5.19), mesh at rest 7.03 / 16.9 (was 1.86 / 3.8), mesh steps 1–6 6.72–7.81 (was 1.85–15.66); initial amount 1088.44 L (was 1089.78); fraction error at rest 0.209 (was 0.050); coarse pool contour flat at rest (was 1.13 mm imprint), mesh pool at rest 0.44 / 5.6; fine arm within 0.02 mm of coarse on every column | red |
| C0.3 | Gate status at HEAD | `npm run test:dawn:sparse-cm12` on the unchanged tree, per lane | recorded; red lanes listed in the status log with owners | 5 pass / 1 fail / 5 timeout / 6 unrun (`288d922a` run) | — | 6 pass / 2 fail / 3 timeout / 6 unrun; fail: symmetric-expansion (D4), mini32-performance 49.8729 ms in-suite; timeout: topology-page-budget, hydrostatic-adaptivity, mini64-performance; perf lanes alone: mini32 49.2175 ms (fail), mini64 timed out at 30 s alone | recorded |
| C1.1 | Grid imprint gone | `tests/sparse-cm12-surface-grid-imprint-dawn.test.ts`, `static`, curvature RMS | ≤ 0.0015 on fixed4, mixed, adaptive | 0.006006 / 0.006105 / 0.010396 (HEAD `ec5af4bb` run 2026-09-09 reproduces it; heights 1.06 / 0.78 / 4.19 mm; test red) | — | — | open |
| C1.2 | Static band at the floor | gate step 0, `static`, mesh sphere RMS / max; pool RMS | ≤ 2.0 / ≤ 3.0 mm; ≤ 0.5 mm | mesh 1.85 / 3.8 (step 1) | floor 1.76 / 2.5 | — | open |
| C1.3 | Static band changes no physics | `density.bin`, `velocity.bin` at steps 0, 6, 15 vs baseline, both arms | bit-identical | — | — | — | open |
| C1.4 | Canonical gate | `npm run test:dawn:sparse-cm12`, band off and static | 17 lanes green, no timeouts | red at HEAD, see C0.3 | — | — | open |
| C2.1 | Kernels match the replay | `sparse-cm12-surface-band-dawn` (a), fixture 16³ | ≤ 1e-4 h per kernel | — | — | — | open |
| C2.2 | Chain matches the replay | (b) GPU phi vs `phi-A-step-n`, steps 1–7, cells with abs(phi) ≤ 2h | ≤ 0.5 mm RMS | — | 0 | — | open |
| C2.3 | Imposed half-cell translation | `capture-retained-imposed-flow-dawn --surface-band=advected`, max root error after two steps | < 0.5 mm | 0.36 cell face jump (density) | — | — | open |
| C2.4 | Sphere and pool before contact | gate on wp2 captures, steps 1–5 RMS / max; step 6 RMS / max; pool RMS steps 1–5 | ≤ 2.0 / ≤ 6 mm; ≤ 2.0 / ≤ 10 mm; ≤ 0.5 mm | HEAD: contour 5.82–7.38 RMS steps 1–5, 7.51 step 6; mesh 7.03 at rest, 6.72–7.67 steps 1–5, 7.81 step 6; pool contour ≤ 0.002 | 1.50–1.90 / 3.3–5.3; 1.88 / 8.7; ≤ 0.5 (HEAD replay within 0.01 mm RMS coarse, 0.05 mm fine; pool ≤ 0.013 mm) | — | open |
| C2.5 | Arms agree | gate, coarse vs fine sphere RMS, steps 1–6 | ≤ 0.2 mm | contour differs by rung | ≤ 0.1 (HEAD: ≤ 0.05) | — | open |
| C2.6 | Launches | dispatch audit (`lib/harness/webgpu-smoke-gpu-audits.ts`), per advance, list form | ≤ baseline + 20 | ≈ 505 | model: ~20 | — | open |
| C2.7 | Perf lanes | `mini32-performance`, `mini64-performance`, stage cost median of 3 | ceilings 40 / 50 ms unchanged, green | green at HEAD | — | — | open |
| C2.8 | Still bricks pay nothing | receipt `surfaceBand.activeCells` on every step where the activity masks mark no brick dirty (`tests/sparse-cm12-frozen-domain-dawn.test.ts`, `tests/sparse-cm12-paused-region-dawn.test.ts` still region) | 0 | — | — | — | open |
| C3.1 | Mass agreement | gate, band amount − solver mass, steps 0–30 | ≤ 0.1 % (1.1 L of 1,090) | — | 0.00 L to step 20 (HEAD, sign classes: 0.00 L at every step to 30, both arms) | — | open |
| C3.2 | Body shifts | receipt, largest shift, steps 1–20 | < 2.5 mm, excluding the step-1 shift that absorbs the initialisation deficit | — | ≤ 2.40 mm (HEAD: 2.65 mm at step 1 = the solver's 1.0 L sphere deficit, then ≤ 2.10 mm to step 30, both arms) | — | open |
| C3.3 | Coupling costs little shape | gate, sphere RMS steps 1–5 vs the WP2 band, both measured against the sphere of the solver's own sphere-body volume (64.4 L → 248.7 mm at HEAD); pool RMS steps 1–5 | ≤ +1.5 mm; ≤ 0.5 mm | — | +0.3–1.3 mm; ≤ 0.5 (HEAD against the analytic sphere: 3.9–4.3 vs 1.5–1.9 mm, of which 1.3 mm is the deficit bias; pool ≤ 0.014 mm) | — | open |
| C3.4 | Through impact to 1 s | capture to step 30: finite fields, no rejected step, mesh vertex count vs baseline at 0.5 and 1 s | within 20 % | — | — | — | open |
| C3.5 | Bodies | receipt body count, steps 1–6 and 8–20 | 2 then 1; none unshifted | — | 2 then 1 from step 7 (HEAD: 2 through step 7, 1 from step 8; none unshifted) | — | open |
| C3.6 | Canonical gate as default | `npm run test:dawn:sparse-cm12` with the quarter profile `coupled` | 17 lanes green | — | — | — | open |
| C3.7 | Cost of both band stages | launches; perf lanes; stage cost median | ≤ baseline + 40; green | ≈ 505 | — | — | open |
| C4.1 | Solver mass outside the reference | gate, step 5 | < 200 ml (provisional) | 685 ml | — | — | open |
| C4.2 | Solver per-cell fraction error | gate, max, step 5 | < 0.1 (provisional) | 0.21 | — | — | open |
| C4.3 | Feedback keeps the band | C2.4, C3.1–C3.3 on the wp4 capture | all hold | — | — | — | open |
| C4.4 | Canonical gate | `npm run test:dawn:sparse-cm12` | 17 lanes green | — | — | — | open |
| C5.1 | Pressure from the band | `hydrostatic-adaptivity`, `mini32-correctness`, `min8-region-surface`, `tests/sparse-cm12-pool-impact-analytic-dawn.test.ts` | green at existing tolerances | green | — | — | open |
| C5.2 | Solids and rigid | `live-rigid-body-coupling`, `tall-cells-hills-far-wall`, `long-dam-far-wall`; rigid capture max phi inside the body | green; ≤ 0 | green | — | — | open |
| C5.3 | Edits and inflow | `live-liquid-injection`; an edit capture's seeded primitive at step 0 | green; ≤ 2.0 mm RMS | green | — | — | open |
| C5.4 | Half scene | gate on `--scene=half`, C2.4 and C3.1 | same thresholds | — | — | — | open |
| C5.5 | Full pool cost | both band stages, stage cost median on the full pool | ≤ 5 ms per advance (cost model, no lane today) | — | model | — | open |
| C5.6 | Tricubic rooting | gate step 0, mesh sphere RMS, exact sphere | ≤ 1.0 mm | 1.85 (step 1) | 0.68 on samples | — | open |

Numbers that must be established, not guessed, during the packages and then
written back here: the GPU extension sweep count and redistance iterations
that hold C2.2 (WP2); the Newton round count (WP3); C4.1 and C4.2 thresholds
(from the first WP3 capture).

## Status log

Newest first. One line per event that changes a tracker row, a threshold or
the route; the tracker's "Current" column is only ever filled from `gate.json`.

- 2026-09-09, early (WP0 close-out): the replay re-run on both HEAD captures
  finished. **Arm A equals the afternoon replay** (coarse within 0.01 mm
  RMS, 1.90 / 3.3 → 2.36 / 12.1 over steps 1–7; fine within 0.05 mm);
  WP2's oracle is `artifacts/surface-band/baseline/{coarse,fine}/replay-normal/phi-A`.
  **Arm D under the 8 September rule failed on HEAD** (6.6–21 mm RMS, pool
  27–47 mm off, pool body shifted 74.5 mm at step 1): the fill/drain test
  compared the solver's full/empty classes with the band's sub-cell ramp
  fraction, and HEAD's density is sharp (pool exactly 1.0 to a cell face),
  so it fired on the whole pool surface every step. Rule corrected to a sign
  test (`--classes topology`, now the replay default; design step 5(a) and
  WP3 kernel 1 updated): fills 0, pool flat to 0.014 mm, mass 0.00 L, sphere
  shift 2.65 mm at step 1 then 0.3–0.7 mm. A sharp amount model for the
  shift (`--width 0`) rejected: 32 L quanta, ±6 mm staircase. The solver's
  sphere holds 64.4 L at rest against the analytic 65.45 L (pool 1,024.0 L
  exact): a 1.6 % native-initialisation deficit, which is the 0.21 fraction
  error of C4.2 at rest and a 1.3 mm radial bias the coupled band will show;
  C3.2 and C3.3 reworded to judge against the solver's volume. Full
  `replay-topology` run on both HEAD arms, steps 0–30, is WP3's oracle:
  D holds 0.00 L at every step in both arms (A drifts to −35.9 / −29.6 L by
  step 30), largest shift after step 1 is 2.10 mm (step 9), bodies 2 through
  step 7 and 1 from step 8, no drains, fills ≤ 28 on any step. Imprint
  gate at HEAD (C1.1): curvature RMS 0.006006 / 0.006105 / 0.010396 on
  fixed4 / mixed / adaptive against the 0.0015 budget, heights 1.06 / 0.78 /
  4.19 mm; red, and the same numbers the tracker's baseline already carried.
  `tests/surface-band-gate.test.py` gained a HEAD class (arm A identity to the
  afternoon table; arm D sign-class oracle once the full run exists).
- 2026-09-08, WP0 (night): gate script, limits file and self-test landed;
  C0.1 green. Baseline captured at `ec5af4bb`, both arms, steps 0–30, under
  `artifacts/surface-band/baseline/`. **C0.2 red: HEAD is not the replay
  capture's solver.** Native contour 5.82 → 7.72 mm RMS over steps 1–7
  (replay capture 3.23 → 5.19); shipping mesh at rest 7.03 / 16.9 mm (was
  1.86 / 3.8) and 6.7–7.8 mm RMS through step 6 (was 1.85–15.66); initial
  amount 1,088.44 L (was 1,089.78); solver fraction error 0.21 already at
  rest (was 0.05): the restore's native initialisation is a different amount
  model from the retained ramp the reference integrates, so C4.2's baseline
  is 0.21 at step 0, not 0.05 → 0.21 over steps 1–7. The coarse arm's pool
  contour is flat at rest (the 1.1 mm imprint of the afternoon capture is
  gone) while its mesh pool shows 0.44 / 5.6 mm; fine and coarse arms agree
  to 0.02 mm on every column. At step 7 the mesh sphere column reads 188 /
  911 mm because the pool has risen 11 mm above the plane before contact and
  the mesh metric's separator counts those vertices as sphere; step 7 stays
  reported, not gated. The replay is being re-run on both HEAD captures for
  WP2's oracle. C0.3 recorded: 6 pass / 2 fail / 3 timeout / 6 unrun in the
  180 s budget; fails are symmetric-expansion (D4, pre-existing: the
  `288d922a` run and a `57b6ae39` checkout fail it) and mini32-performance
  (49.9 ms in-suite, 49.2 ms alone, against the 40 ms ceiling); timeouts are
  topology-page-budget, hydrostatic-adaptivity and mini64-performance (also
  alone). None of these is band work; all block every package's exit.
  Process notes: the capture tool and the suite runner take the repository
  lease themselves and must not be wrapped in `run-webgpu-exclusive`; `seq
  -s,` on macOS leaves a trailing comma the capture tool rejects.
- 2026-09-08 late evening: plan written after the shadow replay; WP0 not
  started; every criterion `open`. The canonical gate is red at HEAD
  (`docs/NATIVE_SURFACE_RESTORATION_2026-09-08.md`: `symmetric-expansion`
  fails D4, five lanes time out, six unrun); C0.3 records it and it blocks
  every package's exit until fixed. Two facts corrected in the doc at the same
  time: pressure classification reads the transported density, not the phi
  seam; the current-map carrier and the native-default restore are committed
  (`a8796cd3`, `288d922a`), so there is nothing to archive or revert.

## Disposition of the working tree

Inspect `git status` fresh; other tasks share this checkout (at the time of
writing, editor and SVO/voxel files were being modified by them). What
belongs to this program:

- Modified files outside the list below (`tests/host-transport-status.test.ts`,
  `lib/svo/**`, `lib/core/webgpu-voxel-debug.ts`, `docs/voxel-surface-rasterization.md`
  and whatever else appears) are other tasks'. Leave them.
- `docs/HANDOFF_FLUID_SURFACE_REVIEW_2026-09-08.md`, `tools/replay-surface-band.py`,
  `tools/render-surface-band-replay.py` (untracked): this program; commit
  together with WP0.
- `tests/sparse-cm12-retained-preparation-recipe.test.ts` (untracked):
  belongs to the retained-mode preparation cache, another task. Leave it.
- `artifacts/surface-band-replay/quarter/{coarse,fine}/` are the replay
  evidence and the WP0/WP2/WP3 oracle inputs; keep them for the life of the
  plan.

The current-map carrier is in `a8796cd3` behind `densityTransport: "current-map"`
and needs no archiving; do not extend it. `__pycache__` is in `.gitignore`
since `288d922a`. Do not stash, checkout or reset in this shared tree.

## Process

Today produced 91 commits and touched 47 doc files. The mandatory Dawn gate
did not pass once: five lanes passed, six timed out, six were never run. Three
agents wrote overlapping decision docs into the same checkout.

- One direction document, this one, updated in place. No new dated doc per
  experiment.
- One agent on the resident at a time. Others read.
- The gate is green before any surface claim; timeouts are failures.
- Work is judged on the shipping mesh of `coarse-first-pool-impact-quarter`,
  fully fine and adaptive, in the app. Peter checks the app himself; no
  browser automation.

## Reproduction

Serial Dawn only, browser simulation unloaded, repository lease held.

```sh
# shipping-mesh A/B, native CM12 transport
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/capture-retained-visual-ab-dawn.ts \
  --arm=coarse --regions=authored --steps=0,2,4,6,7,15,30 \
  --out=artifacts/surface-band/quarter/coarse
# add --arm=fine for the all-fine arm

# imposed half-cell translation, full fine
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/capture-retained-imposed-flow-dawn.ts --assert-continuity

# static grid-imprint gate (M0)
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx --test --test-concurrency=1 \
  tests/sparse-cm12-surface-grid-imprint-dawn.test.ts

# canonical gate, mandatory before any claim
npm run test:dawn:sparse-cm12

# tracker numbers from any capture (WP0); non-zero exit when a limit fails
python3.11 tools/surface-band-gate.py artifacts/surface-band/baseline/coarse \
  --steps=0,1,2,3,4,5,6,7 --limits=tests/surface-band-limits.json --out=gate.json
# with a band readback: add --band=phi (WP2 on); replay oracle: --band=replay-normal/phi-A
python3.11 tests/surface-band-gate.test.py

# shadow band replay: capture every step, then replay on the CPU (no GPU)
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/capture-retained-visual-ab-dawn.ts \
  --arm=coarse --regions=authored --transport=native-cm12 \
  --steps=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --out=artifacts/surface-band-replay/quarter/coarse
python3.11 tools/replay-surface-band.py artifacts/surface-band-replay/quarter/coarse --classes ramp   # the afternoon table
python3.11 tools/render-surface-band-replay.py \
  artifacts/surface-band-replay/quarter/coarse/replay-normal
# HEAD oracles (WP0 baseline, steps 0-30, both arms): A from the default run, D from the sign-class run
python3.11 tools/replay-surface-band.py artifacts/surface-band/baseline/coarse \
  --out artifacts/surface-band/baseline/coarse/replay-topology            # default --classes topology
# about 22 s per capture arm, about 20 min per 20-step replay arm (about 1 h for 30 steps)
```

Analysis: `python tools/analyze-current-map-ray-field.py` for half-level
roots against the discrete-gravity sphere and pool plane;
`python tools/render-retained-imposed-flow.py --input=<capture>` for the
translation figure.

## Sources

- Original paper: `docs/papers/massConservingLiquids.txt`, §3.4 advection,
  §3.5 sharpening.
- Sussman, Puckett. A coupled level set and volume-of-fluid method for
  computing 3D and axisymmetric incompressible two-phase flows. JCP 162, 2000.
- Basilisk CLSVOF: https://basilisk.fr/src/two-phase-clsvof.h
- Russo, Smereka. A remark on computing distance functions. JCP 163, 2000
  (the subcell fix; see the guards in `tools/replay-surface-band.py`).
- Shadow replay evidence: `artifacts/surface-band-replay/quarter/{coarse,fine}/replay-normal/replay.json`
  and `composite.png`; `replay-normal-unguarded/` and
  `replay-normal-shift-before-redistance/` for the two superseded runs.
- Today's evidence: `docs/retained-imposed-flow-diagnosis-2026-09-08.md`,
  `docs/fluid-surface-followup-2026-09-08.md`,
  `docs/CURRENT_MAP_DAWN_PROGRESS_2026-09-08.md`,
  `docs/current-density-departure-map-decision-2026-09-08.md`,
  `docs/coarse-surface-grid-imprint-investigation-2026-09-07.md`.
