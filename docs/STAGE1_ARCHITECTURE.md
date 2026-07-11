# Stage 1 — Architecture and Mathematical Specification

Status: proposed for review<br>
Units: SI throughout<br>
Priority order: correctness, inspectability, reproducibility, comparability,
stability, performance, visual quality

## 1. Concrete recommendations

### macOS technology stack

Use a **C++20 simulation core**, a thin **Objective-C++ bridge**, and a
**Swift/SwiftUI macOS application** containing an `MTKView`. Use Metal compute
and Metal rendering for accelerated paths. Build the core and command-line
validation runner with CMake; build the app with Xcode. Use XCTest for app and
bridge tests and a small C++ test executable (CTest-integrated) for numerical
tests. Store benchmark outputs as JSON and CSV.

This split keeps equations, serialization, and CPU validation independent of
the UI while retaining first-class Metal integration. Objective-C++ is a
narrow ABI boundary; C++ types do not leak into Swift. The CPU reference uses
`double`; GPU state uses `float` initially, with reductions optionally
accumulated in two passes and CPU-verified. The renderer only consumes immutable
snapshots and cannot modify simulation state.

Rejected as the primary structure:

- all-Swift: pleasant UI code, but less suitable for sharing numerical kernels,
  established C++ solvers, and a portable headless validation runner;
- all-Objective-C++: workable, but unnecessarily burdens application UI;
- browser/WebGPU: poorer Metal diagnostics and no advantage for this local,
  compute-heavy target.

### Eulerian solver

Select a liquid **staggered MAC grid** with cell-centred pressure and signed
distance, face-centred velocity, a particle-corrected level set free surface,
second-order Runge–Kutta backtracing, and semi-Lagrangian velocity advection.
Use backward-Euler viscosity and a pressure projection solved by matrix-free
preconditioned conjugate gradient (PCG). A deterministic unpreconditioned CG
implementation is the CPU oracle; weighted Jacobi is retained only as a Metal
baseline. Geometric multigrid is a later optimization after PCG evidence.

Why this choice: semi-Lagrangian advection is robust at interactive time steps
and simple enough to audit. It is unconditionally stable for pure advection but
numerically dissipative and not conservative. RK2 reduces trajectory error but
does not remove interpolation diffusion. Particle-level-set correction limits
interface erosion; a global volume correction is reported separately and may
not conceal raw drift. MacCormack/BFECC can sharpen features but introduce
overshoot and require limiters, so they are deferred behind a common advection
interface. FLIP is not selected because it would blur the comparison with the
particle method and adds particle-grid transfer noise.

### Particle solver

Select **Divergence-Free SPH (DFSPH)** with a compact cubic-spline kernel,
symmetrized viscosity, velocity-first symplectic integration, and a uniform cell grid of
cell width equal to kernel support. Retain an exact `O(N^2)` neighbour oracle.
Use deterministic sorted cell keys in CPU validation mode.

DFSPH is preferred over WCSPH because it controls both density drift and
velocity divergence without an artificial speed of sound or its severe
acoustic time-step. It is preferred over PBF because its pressure solve and
force/impulse accounting have a clearer physical interpretation. PCISPH is a
credible alternative but does not directly enforce the divergence-free
condition and generally needs more density-prediction iterations. DFSPH still
has consistency error, free-surface density deficiency, particle disorder, and
iteration-dependent incompressibility.

## 2. Physical model and governing equations

The fluid is isothermal, Newtonian, single-phase water with constant reference
density `rho0 = 998.2 kg m^-3` and dynamic viscosity
`mu = 1.002e-3 Pa s` at 20 degrees Celsius. Kinematic viscosity is
`nu = mu/rho0 = 1.0038e-6 m^2 s^-1`. Gravity defaults to
`g = (0, -9.80665, 0) m s^-2`. Surface tension and air dynamics are absent in
version 1.

### Continuum equations

In the liquid domain:

```text
∇·u = 0
rho0 (∂u/∂t + u·∇u) = -∇p + mu ∇²u + rho0 g
Dphi/Dt = 0
```

`u [m s^-1]`, `p [Pa]`, and signed distance `phi [m]` are velocity, gauge
pressure, and interface distance. These terms all have force-density units
`N m^-3`: inertia `rho Du/Dt`, pressure gradient `∇p`, viscous term
`mu ∇²u`, and gravity `rho g`.

At the free surface, neglecting air stress and surface tension, `p = 0 Pa`
gauge. Kinematic motion is `Dphi/Dt = 0`. At closed solid boundaries,
impermeability is `(u-u_s)·n = 0`. Version 1 uses free slip for inviscid
projection and applies tangential viscous coupling explicitly; a no-slip option
sets the relative tangential velocity to zero in the viscous solve. The top is
either a closed wall or an open free-surface domain boundary, explicitly stored
in the scene. Pressure has a Neumann condition induced by prescribed normal
solid velocity at walls. A completely liquid-filled closed domain therefore has
a constant-pressure nullspace; the CPU solver enforces zero-mean gauge pressure
rather than pinning a physically distinguished cell.

### Eulerian split step

For time step `dt [s]`:

```text
u_f       = u^n + dt g
u_adv     = Advect_RK2_SemiLagrangian(u_f)
(I-dt nu L) u_visc = u_adv
L_p p     = (rho0/dt) D u_visc
u^(n+1)   = u_visc - (dt/rho0) G p
```

`D`, `G`, and `L_p = D G` use consistent face weights. Fractional liquid/solid
face apertures enter both the divergence and gradient so projection is
compatible. The matrix excludes air cells and uses the free-surface Dirichlet
condition. Solid normal face velocity enters the right-hand side. PCG stops
when `||r||2 <= max(abs_tol, rel_tol ||b||2)` and reports initial/final residual,
iterations, reason, and convergence.

Backward-Euler viscosity is dimensionally valid because `dt nu L` is
dimensionless. It is stable but dissipative. The explicit gravity step and
semi-Lagrangian advection do not impose a strict advective stability bound, but
the CFL limit is still enforced for accuracy and coupling.

### DFSPH discretization

For particle mass `m_j [kg]`, position `x_i [m]`, and kernel
`W_ij [m^-3]`:

```text
rho_i = Σ_j m_j W(x_i-x_j, h)
V_i   = m_i/rho0
```

The discrete continuity estimate is:

```text
d rho_i/dt = Σ_j m_j (v_i-v_j)·∇W_ij
```

DFSPH iteratively finds pressure accelerations that first drive predicted
compression and then density error toward zero. Pair contributions use the
same kernel gradient with equal and opposite impulses. The implementation
records divergence-solve and density-solve residuals and iterations separately.
A regularized symmetric physical-viscosity term is used:

```text
a_i^nu = Σ_j [2 mu m_j/(rho_i rho_j)]
         [(v_i-v_j)·r_ij/(|r_ij|² + eta h²)] ∇W_ij
```

where `eta` is a labelled numerical regularizer (default `0.01`, configurable).
Kernel normalization and gradient identities are unit-tested in 3D. Density
has `kg m^-3`; acceleration has `m s^-2`. XSPH smoothing and artificial
viscosity are off by default because they would add unphysical tuning.

### Rigid bodies

Each body obeys:

```text
m dv/dt = m g + F_contact + F_hydro
d(L_world)/dt = tau_contact + tau_hydro
dx/dt = v
dq/dt = 0.5 (0, omega) q
L_world = I_world omega
I_world = R I_body R^T
```

Use symplectic translation and quaternion orientation integration with
normalization diagnostics. Primitive inertia tensors are analytic. Contact
uses sequential impulses with restitution and Coulomb friction; fixed ordering
and iteration count make CPU validation deterministic.

## 3. Free surface and boundaries

The Eulerian interface uses a cell-centred signed-distance level set, advected
with RK2 and reinitialized only in a narrow band by fast sweeping. Positive and
negative marker particles correct escaped characteristics. Raw enclosed volume,
post-correction volume, particle correction count, and reinitialization-induced
volume change are reported independently. Optional uniform level-set offset
restores target volume but is disabled in validation metrics unless a benchmark
explicitly tests the corrected method.

Eulerian solid boundaries use analytic signed-distance functions for the box
and primitive rigid bodies, sampled into cell volume fractions and face
apertures. The first implementation may use voxel fractions from deterministic
subcell quadrature; this is labelled approximate. Pressure boundary treatment
uses moving-solid normal velocity and equal/opposite pressure impulse.

DFSPH uses pre-sampled boundary particles with effective volumes calibrated
against the kernel sum (Akinci-style volume map) for container and rigid-body
surfaces. Boundary samples move rigidly. Pressure and viscosity interactions
apply pairwise equal/opposite force and torque to the owning body. Analytic SDF
collision correction is only an emergency non-penetration safeguard; every
activation is counted and its impulse included. Boundary spacing and kernel
support are stored in run metadata.

## 4. Fluid–rigid coupling

Coupling is sub-iterated within a shared time step rather than applied as a
visual buoyancy force.

- Eulerian: impose body velocity on covered faces, solve the coupled pressure
  boundary condition, integrate pressure and viscous traction over cut faces,
  and apply its negative impulse/torque to the body. Apply the body impulse back
  to fluid face momentum. Report the residual mismatch caused by discretization.
- DFSPH: boundary-particle pressure and viscous interactions generate pairwise
  impulses; apply the exact opposite impulse and moment arm to the body.
- Displaced volume is geometric liquid/solid intersection, never inferred only
  from a tuned drag law. Hydrostatic buoyancy is an outcome of pressure.

The initial coupling uses two fixed-point subiterations and exposes the count.
Added-mass instability is expected for bodies near or below fluid density;
adaptive subiterations or a monolithic solve are later options. A coupling CFL
limit restricts body-boundary motion to `0.25` nominal spatial elements per
step. Total exchanged impulse is accumulated per body and reconciled against
fluid momentum change.

## 5. Adaptive time stepping

Candidate limits are computed before every accepted step:

```text
dt_advective = C_cfl dx / max_face(|u|)
dt_viscous   = C_nu dx²/nu                 (reported; implicit solve may exceed)
dt_body      = C_body dx / max_boundary(|v + omega × r|)
dt_collision = C_collision sqrt(dx/max(|a_body|, eps))
dt_coupling  = C_coupling dx / max_relative_boundary_speed
dt_particle  = C_p h / max_particle(|v| + h |∇·v|)
dt_force     = C_f sqrt(h/max_particle(|a|, eps))
```

Defaults: `C_cfl=0.75`, `C_nu=1/6`, and all body/particle/coupling coefficients
`0.25`. They are numerical stability/accuracy parameters, not physical
constants. Surface-tension and acoustic limits are recorded as not applicable.
The selected `dt` is the minimum of candidates, user maximum, and frame target.
Emergency clamps, rejected steps, NaNs, and the limiting reason are mandatory
diagnostics. Fixed-step validation bypasses selection but still reports every
limit and fails if the requested step violates a hard safety factor.

## 6. Data structures and subsystem ownership

```text
SceneDescription (immutable physical/numerical inputs)
  -> SimulationSession (clock, action log, reproducibility metadata)
      -> FluidSolver interface
          -> EulerianFluidSolver
          -> DFSPHFluidSolver
      -> RigidBodySystem
      -> FluidRigidCoupler
      -> AdaptiveTimeStepper
      -> MetricsCollector -> ValidationHarness -> evidence artifacts
      -> immutable RenderSnapshot -> Renderer / DebugRenderer / ApplicationUI
```

The genuinely shared `FluidSolver` operations are initialize/reset, advance one
named substep, sample velocity/pressure, compute volume/momentum/energy, produce
a solver-specific surface snapshot, diagnostics, checkpoint, and restore.
Pressure residuals and neighbour lists remain solver-specific capabilities; the
shared interface does not pretend that density error equals divergence.

Eulerian arrays are structure-of-arrays, contiguous, with explicit `(i,j,k)`
indexing and one-cell halo. Particle state is structure-of-arrays and sorted by
cell key only in optimized modes; stable particle IDs preserve identity.
Rigid bodies store canonical double-precision state. Snapshots are versioned and
immutable.

## 7. CPU reference architecture

The headless core has no Metal, UI, or wall-clock dependency. It runs one thread
by default with deterministic traversal and reductions. Critical algorithms
have clear reference forms:

- matrix-free CG and explicit small-matrix assembly cross-check for projection;
- brute-force and uniform-grid particle neighbours;
- direct particle density/viscosity sums;
- direct surface quadrature of pressure and viscous traction;
- analytic primitive mass properties and buoyancy reference integrals;
- compensated scalar/vector reductions for diagnostics.

Tests seed a specified PCG64 generator and serialize initial and final state.
Exact reproducibility is required on the same CPU/build configuration; across
architectures, tolerance-based reproducibility is reported rather than promised.

## 8. Metal GPU architecture

Metal is introduced only after each CPU oracle passes. GPU resources are kept in
private buffers; argument buffers describe stable kernel inputs. Each numerical
stage is a separate compute pipeline with timestamp/counter boundaries. Prefix
sum and radix sort build the SPH cell table. Eulerian stencils operate in 3D
thread grids. Reductions produce partials on GPU and deterministic final
comparison data on CPU in validation mode.

PCG is the correctness GPU pressure path initially; weighted Jacobi provides a
simple baseline. Half precision is prohibited for physics state. Fast-math is
off in comparison tests. Every GPU port has a small-scene CPU/GPU test for
state, residual, invariant, and iteration differences. Rendering receives a
copy or a versioned read-only view after command-buffer synchronization.

## 9. Scene and reproducibility format

The canonical format is versioned JSON described in `SCENE_FORMAT.md`. It stores
all physical and numerical parameters with units in field names or schema
descriptions, solver-independent initial conditions, random seed, boundary
mode, nominal resolution, and primitive bodies. Solver-specific settings live
under separate keyed blocks and cannot alter shared physics.

A run record adds application/core version, git revision, hardware, OS, compiler,
precision, time-step history, action log, convergence history, and content hash
of the canonical scene. Checkpoints are versioned binary payloads with a JSON
manifest; JSON is not used for large state arrays.

## 10. Validation suite and gates

Tests are defined before their subsystem is implemented. Each emits a structured
record containing model, complete initial/boundary conditions, resolution,
time-step history, expected result, metric, tolerance, measured result, and
pass/fail. Initial suite:

| Benchmark | Principal evidence | Initial gate |
|---|---|---|
| units and kernel identities | dimensional/static checks; kernel integrals | analytic or quadrature tolerance |
| rigid free fall | position and velocity vs analytic solution | relative error `< 1%` |
| primitive inertia | mass/inertia vs analytic values | relative error `< 1e-12` CPU |
| rigid collision | non-penetration and impulse balance | penetration `< 1e-6 m` after solve |
| projection manufactured field | divergence reduction, BC, residual | post-divergence `< max(1e-10 s^-1, 1e-6 pre)` CPU |
| hydrostatic column | `p=rho g depth` | pressure L2 relative error `< 5%` away from interface |
| static liquid | drift, velocity, energy | volume drift `< 1%`; no energy growth trend |
| neighbour equivalence | exact ID sets vs brute force | zero missing, extra, duplicate |
| SPH density lattice | density distribution | interior mean relative error `< 1%` |
| SPH static liquid | density, penetration, energy | mean density error `< 2%`; zero persistent penetration |
| neutral/floating/sinking | displacement and body trajectory | float submerged fraction error `< 5%` |
| closed momentum exchange | fluid+body momentum | relative drift `< 0.1%` CPU |
| dam break | front/height reference curves | dataset-specific, fixed before test |
| resolution convergence | errors at `dx`, `dx/2`, `dx/4` | decreasing error; observed order reported |
| time convergence | outputs at `dt`, `dt/2`, `dt/4` | decreasing difference; order reported |
| CPU/GPU equivalence | state and invariant deltas | per-kernel mixed abs/relative tolerances |

No benchmark tolerance may be edited in response to a failure without a
versioned contract change, rationale, and retained result. The machine-readable
initial contract is `PHYSICS_CONTRACT.json`.

### Initial global tolerances

- NaN and infinity counts: exactly zero.
- Persistent boundary penetrations: exactly zero; transient maximum `< 0.01 dx`
  or `< 0.01 particle_spacing` and every correction reported.
- PCG relative residual: `<= 1e-8` CPU double, `<= 1e-5` GPU float.
- Eulerian projection divergence: L2 `< max(1e-10 s^-1, 1e-6 pre-L2)` CPU;
  GPU `< max(1e-6 s^-1, 1e-4 pre-L2)` for the manufactured projection test.
- DFSPH mean density error: `< 2%`; maximum `< 5%` excluding one kernel radius
  of the free surface, which is reported separately.
- Static volume drift: `< 1%` over the benchmark duration.
- Hydrostatic pressure L2 relative error: `< 5%` at probes at least one spatial
  element from interface and walls.
- Same-build deterministic CPU replay: state bytes exactly equal when the same
  deterministic solver configuration is used.

These are bring-up thresholds, not accuracy claims. Resolution and time-step
convergence results are mandatory beside threshold results.

## 11. Solver comparability strategy

The same immutable physical scene initializes both solvers. Comparison is not
based on equal particle count alone. For each run report:

- nominal length `ell`: Eulerian `dx=ell`, particle spacing `s=ell`;
- SPH smoothing length selected so compact support radius is approximately
  `2s` (exact `h` convention is recorded);
- active velocity DOFs, pressure cells, level-set markers, fluid particles,
  boundary samples, allocated bytes, and peak resident bytes;
- stage timings, pressure/constraint iterations, neighbour interactions, and
  energy/momentum error per simulated second.

Run three comparison families: equal nominal length, approximately equal active
fluid DOFs, and approximately equal wall-clock budget. No family is labelled
universally fair. Compare common observables after sampling onto common probes:
volume, centre of mass, body pose/trajectory, momentum, energy, displaced
volume, and surface height. Divergence and density error remain side-by-side
solver-native diagnostics. See `COMPARABILITY.md` for formulae.

## 12. Performance and stability risks

- 3D resolution grows cubically; memory bandwidth and pressure iteration count
  dominate the grid method.
- DFSPH cost is particle count times neighbours times constraint iterations;
  sorting and free-surface imbalance can dominate.
- Level sets lose volume and semi-Lagrangian advection loses kinetic energy.
- Cut-cell fractions can produce ill-conditioned pressure systems; merge or
  clamp policies must be explicit and measured.
- DFSPH boundaries can show density deficiency, sticking, leakage, clumping,
  and tensile instability.
- Light or thin bodies create added-mass instability under partitioned coupling.
- GPU atomics/reductions can be nondeterministic and mask momentum imbalance.
- CPU/GPU precision differences accumulate chaotically; compare invariants and
  convergence, not long-horizon particle identity.
- Transparent water/glass ordering is a rendering risk but cannot influence a
  physics acceptance result.

## 13. Known physical limitations

Version 1 models no air pressure/drag, surface tension, wetting/contact angle,
cavitation, entrained air, compressible shocks, turbulence closure, thermal
physics, evaporation, phase change, flexible solids, fracture, or resolved
spray/foam. The Newtonian continuum assumption breaks down below resolved
scales. Free-slip walls underpredict boundary layers; practical water viscosity
is too small to resolve physical wall layers at interactive resolution. SPH
kernel smoothing and Eulerian interface thickness impose different effective
surface resolutions. Primitive boundary sampling and cut-cell quadrature add
geometry error. These limits must accompany visual comparisons.

## 14. Proposed repository structure

```text
FluidLab.xcodeproj
app/                         SwiftUI/AppKit application
bridge/                      Objective-C++ ABI bridge
core/include/fluidlab/       public C++ interfaces
core/src/{scene,clock,metrics,rigid,eulerian,sph,coupling}/
reference/                   deliberately clear CPU oracle algorithms
gpu/{metal,host}/            Metal kernels and dispatch code
render/                      Metal presentation and debug rendering
schemas/                     versioned scene/run/checkpoint schemas
benchmarks/scenes/           immutable benchmark inputs
tests/{unit,numerical,gpu,regression}/
tools/                       headless runner and evidence updater
docs/
PHYSICS_EVIDENCE.md
```

Stage 2 should create only the directories and targets it needs; empty
architectural folders add no evidence.

## 15. Staged implementation plan

1. **Architecture/specification (current):** ratify choices, contracts, and
   tolerances. Evidence claims remain pending.
2. **macOS shell:** window, `MTKView`, camera, debug/presentation glass box,
   basic controls, scene JSON round-trip, metrics/test-runner UI. Validate
   serialization hash and camera transforms.
3. **Rigid bodies without fluid:** analytic mass properties, gravity,
   symplectic integration, primitive/container contacts, body-body impulses.
   Gate on free fall, inertia, penetration, and momentum tests.
4. **Eulerian CPU reference:** MAC storage/operators, interface, advection,
   viscosity, CG/PCG projection, boundaries, adaptive step, diagnostics. Gate on
   manufactured projection, static water, hydrostatics, volume and convergence.
5. **DFSPH CPU reference:** kernels, brute-force neighbours, deterministic grid,
   density/divergence solves, boundaries, integration, diagnostics. Gate on
   neighbour identity, density lattice, static water, boundaries, convergence.
6. **One-way bodies:** prescribed obstacles and traction integration. Gate on
   hydrostatic buoyancy, floating fraction, and sinking direction.
7. **Two-way coupling:** conservative impulse/torque exchange and subiteration.
   Gate on neutral buoyancy, oscillation, closed momentum, and multiple bodies.
8. **Metal acceleration:** port verified stages individually; retain CPU oracle.
   Gate each kernel on small-scene CPU/GPU comparisons.
9. **Comparison mode:** synchronized clocks, immutable shared scene, common
   probes, side-by-side solver-native and common metrics. Gate on identical
   initial physical data and repeatable exports.
10. **Rendering improvements:** reconstruct surface and add optical effects only
    after physics gates; scientific view remains available and independent.

At each stage: define benchmark configuration first, implement the smallest
increment, run all affected regression tests, record actual measurements in the
evidence ledger, and stop on a major gate failure.

## 16. Decisions required before coding

### Blocking before Stage 2

1. Accept the C++20 / Objective-C++ / SwiftUI stack and a minimum macOS target
   (recommend macOS 14, Apple Silicon primary, x86_64 CPU reference best-effort).
2. Accept that comparison has three declared fairness families rather than one
   supposedly equivalent resolution.
3. Select licensing and dependency policy (recommend permissive-only third-party
   dependencies and vendored/version-pinned numerical test dependencies).
4. Confirm benchmark artifacts may be written to a local run directory and that
   scene/run schemas can evolve with explicit version migrations.

### Blocking before fluid stages

5. Confirm particle-level-set correction complexity is acceptable; fallback is
   a simpler level set with larger, explicitly expected volume drift.
6. Confirm DFSPH over simpler WCSPH; fallback gains simplicity but introduces an
   artificial equation of state and acoustic time-step.
7. Choose default wall physics: recommend free-slip for initial cross-solver
   validation, then add no-slip as a separately tested option.
8. Choose whether open-top overflow is retained (recommended) or prohibited in
   initial benchmarks; escaped volume must be separately accounted.
9. Ratify bring-up tolerances and the policy that changes are versioned rather
   than silently loosened.
10. Select a published dam-break reference dataset before implementing that
    benchmark; its curves and measurement convention must be frozen first.

### Can remain deferred

11. Multigrid versus PCG optimization, higher-order advection, monolithic
    added-mass coupling, mesh bodies, and photorealistic surface rendering.
