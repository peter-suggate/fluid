# Physics Evidence Ledger

Contract: `docs/PHYSICS_CONTRACT.json` version 0.2.0 plus `docs/STAGE4_ACCEPTANCE.md`<br>
Current stage: Stage 8 browser bring-up; CPU oracles and coupling gates passed<br>
Measured simulation evidence: rigid, Eulerian, particle, coupling, and live WebGPU suites

This ledger records measurements, not intentions. A specification or plausible
image is not evidence. Every pending claim remains **UNVERIFIED** until an
automated benchmark artifact is linked here.

## Claim E-001 — Eulerian pressure projection reduces divergence

- Governing equation: `L_p p = (rho0/dt) D u*`; `u^(n+1)=u*-(dt/rho0)G p`
- Implementation location: `lib/eulerian-solver.ts` (`project`)
- Validation test: deterministic manufactured non-solenoidal MAC velocity field
- Configuration: proposed `16^3`, `32^3`, `64^3`; closed free-slip box
- Measured result: RMS divergence `2.1631718950 s^-1` before and
  `1.1297068769e-8 s^-1` after; PCG relative residual `5.2224555199e-9`
- Acceptance: CPU residual `<=1e-8`; divergence reduction `<=1e-6` with absolute
  floor `1e-10 s^-1`
- Result: **PASS** at the Stage 4 bring-up gate (`post < 1e-5 pre`)
- Resolution dependence: pending
- CPU/GPU difference: pending Stage 8
- Known limitations: tolerance depends on compatible discrete `D` and `G` and
  excludes pathological unmerged cut cells

## Claim E-002 — Optimized particle neighbours equal brute force

- Governing condition: pair included iff distance is inside compact kernel
  support under the documented inclusive-boundary convention
- Implementation location: pending Stage 5
- Validation test: deterministic random, lattice, cell-boundary, duplicate-position,
  and empty particle sets
- Configuration: proposed particle counts `0, 1, 32, 257, 2048`
- Measured result: not run
- Acceptance: zero missing, extra, or duplicate stable particle IDs
- Result: **UNVERIFIED**
- Resolution dependence: not applicable; multiple support radii required
- CPU/GPU difference: pending Stage 8
- Known limitations: exact equality requires a frozen distance-boundary rule

## Claim E-003 — Static Eulerian water remains stable

- Governing equations: incompressible Navier–Stokes and free-surface pressure
  `p=0 Pa` gauge
- Implementation location: `lib/eulerian-solver.ts`
- Validation test: static liquid at rest
- Configuration: proposed three resolutions and fixed-step refinements
- Measured result: over 40 fixed steps, maximum kinetic energy
  `<1e-8 J`, marker and occupied volume drift `0`, NaN/Inf count `0`
- Acceptance: volume drift `<1%`, no energy growth trend, no NaN/Inf, projection
  threshold from E-001
- Result: **PASS** at the Stage 4 coarse-grid gate
- Resolution dependence: pending
- CPU/GPU difference: pending Stage 8
- Known limitations: particle-level-set correction and any volume correction
  must be reported separately from raw level-set drift

## Claim E-004 — Static DFSPH water remains approximately incompressible

- Governing equations: SPH density sum, discrete continuity, DFSPH divergence
  and density constraints
- Implementation location: pending Stage 5
- Validation test: static liquid with boundary samples
- Configuration: proposed three particle spacings and time-step refinements
- Measured result: not run
- Acceptance: mean density error `<2%`; interior maximum `<5%`; zero persistent
  penetrations; no energy growth trend
- Result: **UNVERIFIED**
- Resolution dependence: pending
- CPU/GPU difference: pending Stage 8
- Known limitations: free-surface particles are reported separately because
  kernel truncation biases density

## Claim E-005 — Hydrostatic pressure follows depth

- Governing equation: `p = rho0 |g| depth`
- Implementation location: Eulerian `lib/eulerian-solver.ts`; DFSPH pending
- Validation test: fixed pressure probes in static column
- Configuration: probes at least one nominal element from boundaries/interface
- Measured Eulerian result: L2 relative error `0.07474568996` on the fine
  bring-up grid, excluding the one-cell zero-gauge interface stencil
- Acceptance: Stage 4 bring-up `<10%`; final cross-solver contract remains `<5%`
- Result: **PASS** for Stage 4 bring-up; **UNVERIFIED** for final DFSPH/cross-solver gate
- Resolution dependence: pending
- CPU/GPU difference: pending Stage 8
- Known limitations: SPH pressure is kernel-smoothed; surface and wall probes
  require separate reporting

## Claim E-006 — Rigid free fall matches analytic motion

- Governing equations: `y=y0+v0 t+0.5 g t^2`; `v=v0+g t`
- Implementation location: pending Stage 3
- Validation test: body drop without contacts or fluid
- Configuration: several fixed time steps before collision
- Measured result: not run
- Acceptance: position and velocity relative error `<1%`; convergence with `dt`
- Result: **UNVERIFIED**
- Resolution dependence: not applicable
- CPU/GPU difference: rigid canonical state remains CPU initially
- Known limitations: symplectic Euler has first-order global position error

## Claim E-007 — Fluid/body coupling conserves momentum in a closed system

- Governing equation: `Delta P_fluid + Delta P_body = 0` without external impulse
- Implementation location: pending Stage 7
- Validation test: moving rigid body interacting with stationary fluid, gravity off
- Configuration: both solvers; closed free-slip container with wall impulse
  accounted separately
- Measured result: not run
- Acceptance: CPU relative momentum drift `<0.1%`
- Result: **UNVERIFIED**
- Resolution dependence: pending
- CPU/GPU difference: pending Stage 8
- Known limitations: container impulse must be included or the selected subsystem
  is not closed

## Claim E-008 — Equal scenes initialize equivalent physical conditions

- Governing constraint: identical density, viscosity, gravity, geometry, body
  state, liquid region, duration, seed, and boundary semantics
- Implementation location: pending scene loader and both solver initializers
- Validation test: canonical scene hash plus initialized mass/volume/body-state
  audit
- Configuration: every comparison benchmark
- Measured result: not run
- Acceptance: exact shared-input hash; initial volume mismatch reported and
  converges under refinement
- Result: **UNVERIFIED**
- Resolution dependence: pending
- CPU/GPU difference: not applicable to source scene
- Known limitations: discrete initial volumes cannot generally be identical at
  finite resolution

## Evidence update procedure

1. Link the immutable scene, contract version, build revision, and raw JSON/CSV.
2. Copy measured values without rounding away a failure.
3. Mark PASS or FAIL from the versioned threshold.
4. On failure, name the likely subsystem and added instrumentation.
5. Retain failed records after a fix and link the succeeding regression run.
6. Record resolution/time-step series and CPU/GPU deltas where required.

## Stage 2 application-shell evidence (non-physics)

Recorded: 2026-07-12<br>
Build: `web-stage2-0.1.0`  
Environment: Apple Silicon macOS, WebGPU browser adapter

### Claim S2-A — The shared scene and deterministic utilities satisfy the shell contract

- Implementation: `lib/model.ts`, `lib/math.ts`, `lib/validation.ts`
- Validation: `tests/shell-contract.test.ts`
- Configuration: default SI scene, seed `20260712`, default and perturbed cameras
- Measured result: 8 tests passed, 0 failed in 261.45 ms
- Camera orthonormality error shown in the in-app report: `2.22e-16`
- Acceptance: basis error `<1e-10`; scene round-trip byte-identical; 1,000
  seeded values with zero mismatches; invalid physical inputs rejected
- Result: **PASS**
- Known limitation: this tests scene/camera/reproducibility infrastructure, not
  a fluid equation

### Claim S2-B — The WebGPU presentation renderer initializes and draws

- Implementation: `lib/webgpu-renderer.ts`
- Validation: live browser adapter initialization, WGSL compilation, canvas
  allocation, console inspection
- Measured result: status `WebGPU renderer ready`; backing canvas `702 × 1017`;
  zero captured browser warnings/errors in the final check
- Acceptance: explicit ready/unavailable status, positive canvas dimensions,
  no uncaught browser error
- Result: **PASS**
- Retained failure: the first shader compile failed because `fwidth` was called
  from non-uniform control flow (`43:51`). The renderer replaced derivative-based
  grid antialiasing with a uniform-control-flow-independent analytic line and
  passed the subsequent compile/regression check.
- Known limitation: the water field is an animated presentation shader and is
  not solver output; CPU encode timing is not GPU execution time

### Claim S2-C — Core interaction states are operable and inspectable

- Implementation: `components/FluidLab.tsx`
- Validation: live browser interaction
- Measured result: validation panel opened with `5/5` in-app checks; solver mode
  switched to Eulerian with `aria-pressed=true`; running state changed to paused;
  front camera preset changed azimuth from `0.720000` to `0.000000`
- Acceptance: unique accessible controls and deterministic observable state
- Result: **PASS**
- Known limitation: file downloads were not triggered during automated browser
  QA; serialization and manifest construction are covered by unit tests

### Claim S2-D — The workbench fits the minimum desktop validation viewport

- Implementation: `app/globals.css`
- Validation: browser viewport override at `1024 × 768`
- Measured result: body `1024 × 768`, scroll width `1024`, canvas CSS size
  `502 × 644`; no horizontal or vertical page overflow
- Acceptance: layout usable at 1024 × 768 with positive viewport dimensions
- Result: **PASS**
- Known limitation: touch interaction and browsers without WebGPU require
  further device-matrix testing

### Stage 2 gate result

- Production build: **PASS**
- TypeScript strict check: **PASS**
- ESLint: **PASS**
- Deterministic unit suite: **PASS**, 8/8
- Live WebGPU renderer: **PASS**
- Browser console: **PASS**, zero final warnings/errors
- Physics claims: **NONE** — Eulerian and DFSPH solvers remain unimplemented

## Claim R3-001 — Primitive rigid-body mass properties are analytic

- Governing equations: solid sphere, cuboid, cylinder, and composite
  cylinder-plus-hemispheres volume/inertia integrals
- Implementation: `lib/rigid-body.ts` (`primitiveVolume`, `massProperties`)
- Validation: `tests/rigid-body.test.ts` R3-01/R3-02
- Configuration: all four primitive shapes; SI density and dimensions
- Measured result: benchmark sphere mass `4.188790204786391 kg`, analytic
  `4.188790204786391 kg`, relative error `0`
- Acceptance: relative error `<1e-12`
- Result: **PASS**
- Resolution dependence: not applicable (analytic primitive geometry)
- CPU/GPU difference: mass properties remain CPU binary64
- Known limitation: imported meshes are not supported

## Claim R3-002 — Rigid free fall converges to constant-acceleration motion

- Governing equations: `y=y0+v0 t+0.5 g t²`, `v=v0+g t`
- Implementation: `advanceRigidBodies` velocity-first symplectic Euler
- Validation: R3-03 and R3-04
- Configuration: `y0=10 m`, `v0=0.7 m/s`, `g=-9.80665 m/s²`, duration
  `0.5 s`, no contact
- Expected final Y: `9.12416875 m`
- Measured absolute errors: `0.00980665 m` at `dt=0.004 s`, `0.00490333 m`
  at `dt=0.002 s`, `0.00245166 m` at `dt=0.001 s`
- Acceptance: relative position/velocity error `<1%`; error decreases for each
  halving of `dt`
- Result: **PASS**; observed first-order position convergence ratio `2.0`
- CPU/GPU difference: GPU integration not implemented
- Known limitation: symplectic Euler is first-order accurate in position

## Claim R3-003 — Sphere contact exchanges equal and opposite momentum

- Governing equation: normal/friction impulse equation in
  `docs/STAGE3_ACCEPTANCE.md`
- Implementation: `solveBodyContact`, `applyImpulse`
- Validation: R3-06/R3-10, unequal-mass sphere impact, gravity off
- Measured momentum before/after: `2.0943951023931957 kg m/s` / same
- Relative momentum drift: `0`
- Kinetic energy before/after: `2.3561944901923453 J` /
  `0.7199483164476598 J` (ratio `0.3055555556` under restitution `0.5`)
- Acceptance: momentum relative drift `<1e-12`; no energy creation
- Result: **PASS**
- Resolution dependence: not applicable
- CPU/GPU difference: GPU rigid solver not implemented
- Known limitation: sphere–sphere narrow phase is exact; other body pairs use a
  conservative bounding-sphere proxy and can collide early

## Claim R3-004 — Container contacts prevent persistent floor penetration

- Governing condition: analytic primitive support point must remain in the
  inward half-space of every closed container plane
- Implementation: `supportRadius`, `solvePlaneContact`
- Validation: R3-07, sphere dropped for `3 s` at `dt=0.001 s`
- Measured final centre Y: `0.1000000001 m` for radius `0.1 m`
- Final penetration: `0 m`; final vertical speed: `0 m/s`
- Acceptance: final penetration `<1e-6 m`
- Result: **PASS**
- Resolution dependence: not applicable; time-step convergence still required
  for impact trajectories
- CPU/GPU difference: GPU rigid solver not implemented
- Known limitation: positional correction is non-energy-conserving and its raw
  pre-correction penetration remains a diagnostic

## Claim R3-005 — Rigid CPU replay remains finite and deterministic

- Governing conditions: unit quaternion, finite state, deterministic traversal
- Implementation: quaternion integration/normalization and fixed pair ordering
- Validation: R3-05, R3-08, R3-09
- Measured final quaternion norm: `1`
- Deterministic replay: byte-identical
- NaN/infinity count: `0`
- Acceptance: norm error `<1e-12`, byte-identical replay, invalid count `0`
- Result: **PASS**
- Known limitation: same-build JavaScript binary64 determinism is demonstrated;
  cross-browser bitwise identity is not yet claimed

## Stage 3 interactive evidence

- WebGPU shader compiled with all four analytic render intersections.
- Browser added capsule and cylinder bodies, increasing count from `2` to `4`.
- Drop action changed selected sphere position from `(-0.160, 1.180, 0.000) m`
  to a dynamically integrated contact state.
- Single-step advanced simulation time by `0.0010000000000012221 s` (displayed
  fixed step `0.001 s`).
- Final browser console warnings/errors: `0`; in-app Stage 3 report: `8/8` pass.
- Physics boundary: water remains presentation-only; no buoyancy, drag, or
  two-way fluid momentum claim is made.

Raw benchmark: `benchmarks/results/stage3-rigid-reference.json`

## Stage 4 Eulerian fluid evidence

Recorded: 2026-07-12<br>
Build: `web-stage4-0.3.0`

- Projection: manufactured staggered velocity divergence reduced by more than
  five orders of magnitude; Jacobi-PCG converged below `1e-8` relative residual.
- Boundaries: every normal velocity sample on all six closed walls was exactly
  zero after a complete step.
- Free surface: eight deterministic markers per initial cell were advected with
  RK2; marker-derived volume drift was exactly zero over the dam benchmark.
- Dam break: front advanced by more than `0.12 m` within `0.32 s`; measured live
  browser run advanced from `-0.220 m` to `0.580 m` by `t=0.688 s`.
- Static water: tank-fill kinetic energy stayed below `1e-8 J`; grid and marker
  volume drift and invalid-value count remained zero.
- Hydrostatics: coarse fine-bring-up L2 pressure error was
  `0.07474568996`, below the Stage 4 `<10%` gate.
- Reproducibility: velocity, pressure, markers, occupancy, and diagnostics were
  byte-identical across same-build replay.
- Browser QA: WebGPU shader compiled, the 3D occupancy texture displayed the
  left-wall water column at `t=0`, Play advanced the dam front, and the final
  console warning/error count was zero.

Automated validation: `tests/eulerian-solver.test.ts`. The water calculation is
an interactive CPU binary64 reference and the occupancy visualization is WebGPU
f32. DFSPH, GPU fluid kernels, buoyancy, drag, and two-way fluid/body momentum
exchange remain explicitly unimplemented.

## Stages 5–8 evidence

Recorded: 2026-07-12<br>
Build: `web-stage8-0.8.0`

- Particle neighbour search: deterministic uniform-grid neighbour IDs exactly
  matched the brute-force oracle for every sampled particle; zero missing,
  extra, or duplicate neighbours.
- Particle density: initial poly6 lattice interior mean relative density error
  was below `2%`; free-surface density deficiency is reported separately.
- Particle invariants: particle count, mass-derived volume, and same-build state
  replay were exact; all tested states remained finite and inside the container.
- One-way coupling: a fully immersed neutral-density primitive had net vertical
  force below `1e-10 N`; a twice-water-density primitive accelerated downward;
  a half-wet axis-aligned box displaced half its analytic volume within `5%`.
- Two-way coupling: body impulse and distributed particle-fluid reaction impulse
  closed below `1e-12 N s` in the gravity-free benchmark; multiple body loads
  remained finite and independently keyed.
- WebGPU: the balanced path allocated `60 × 45 × 40 = 108,000` cells and the
  high preset allocated `100 × 75 × 67 = 502,500` cells. Live readback measured
  the dam front moving from `-0.220 m` to `0.600 m`, finite maximum speed, and
  hundreds of encoded simulation steps with zero browser warnings/errors.
- Comparison view: the same clock and camera displayed GPU volume fraction on
  the left and PBF particle positions/density error on the right.

Automated validation: `tests/particle-solver.test.ts` and
`tests/fluid-rigid-coupling.test.ts`. Live WebGPU tests cover shader compilation,
storage-texture writes, conservative volume transport, quality reallocation,
readback diagnostics, reset/play, and comparison rendering.

Limitations: PBF is the selected browser particle reference, not DFSPH. Its
surface density error and correction/CFL limiters are visible. GPU pressure uses
fixed weighted-Jacobi iterations without a reduced residual; its collocated
layout and hydrostatic predictor are an interactive approximation, not the MAC
oracle. GPU PBF and resolved cut-cell traction are not claimed.
