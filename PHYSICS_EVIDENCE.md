# Physics Evidence Ledger

Contract: `docs/PHYSICS_CONTRACT.json` version 0.1.0 (proposed)<br>
Current stage: Stage 2 browser shell passed; Stage 3 physics not started<br>
Measured simulation evidence: none

This ledger records measurements, not intentions. A specification or plausible
image is not evidence. Every pending claim remains **UNVERIFIED** until an
automated benchmark artifact is linked here.

## Claim E-001 — Eulerian pressure projection reduces divergence

- Governing equation: `L_p p = (rho0/dt) D u*`; `u^(n+1)=u*-(dt/rho0)G p`
- Implementation location: pending Stage 4
- Validation test: deterministic manufactured non-solenoidal MAC velocity field
- Configuration: proposed `16^3`, `32^3`, `64^3`; closed free-slip box
- Measured result: not run
- Acceptance: CPU residual `<=1e-8`; divergence reduction `<=1e-6` with absolute
  floor `1e-10 s^-1`
- Result: **UNVERIFIED**
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
- Implementation location: pending Stage 4
- Validation test: static liquid at rest
- Configuration: proposed three resolutions and fixed-step refinements
- Measured result: not run
- Acceptance: volume drift `<1%`, no energy growth trend, no NaN/Inf, projection
  threshold from E-001
- Result: **UNVERIFIED**
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
- Implementation location: pending Stages 4 and 5
- Validation test: fixed pressure probes in static column
- Configuration: probes at least one nominal element from boundaries/interface
- Measured result: not run
- Acceptance: L2 relative error `<5%`
- Result: **UNVERIFIED**
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

Recorded: 2026-07-12  
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
