# Uniform MAC baseline

Select **Uniform MAC baseline** (`uniform-mac`) in the existing method picker.
This is an independent, conventional incompressible liquid solver for comparing
against Uniform Geometric. The method descriptor installs its parameters,
algorithm selection, diagnostics, pipeline graph, performance phases and harness
capabilities through the same contracts as the other methods. Native vertex phi
and derived liquid fractions feed the shared visualization and surface renderer.
The shared smoke executor accepts `FLUID_METHOD=uniform-mac` for offline runs.
The `uniform-ab` lane on `minimal-power-dam-break-32` authors both methods with
the same full 32³ grid and two 0.004 s advances, including shared raster checks:

```sh
FLUID_SCENE=minimal-power-dam-break-32 FLUID_LANE=uniform-ab \
  node --import tsx tools/run-webgpu-exclusive.ts --import tsx tools/run-webgpu-smoke.ts
```

Run this only when no other Dawn process or browser is using WebGPU. This short
lane checks integration; use longer runs and a timestep/grid sweep for physical
accuracy and performance conclusions.

## Numerical references

- [Bridson and Müller-Fischer, Fluid Simulation, SIGGRAPH 2007 course notes](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf):
  MAC staggering, incompressible projection, free-surface pressure, extrapolation
  and level-set transport.
- [Christopher Batty's Fluid3D](https://github.com/christopherbatty/Fluid3D):
  compact implementation reference for MAC and ghost-fluid pressure. This solver
  uses an Eulerian vertex level set rather than Fluid3D's marker particles.
- [Selle et al., An Unconditionally Stable MacCormack Method](https://physbam.stanford.edu/papers/stanford2006-09.pdf):
  predictor/reverse correction with local bounds for velocity and phi.

The implementation is original; these are numerical references, not vendored code.

## One substep

1. Advect staggered velocities and vertex phi with RK2 characteristics and bounded
   MacCormack (or the selectable first-order semi-Lagrangian variant).
2. Apply gravity and explicit viscosity. Reinitialize phi with four Godunov sweeps,
   freezing vertices adjacent to a transported zero crossing.
3. Build the symmetric seven-point ghost-fluid Poisson system. Solve for
   `dt * pressure / density` with warm-started diagonal PCG. Surface tension supplies
   the liquid-air Dirichlet pressure; pressure is allowed to be negative.
4. Verify a fresh infinity-norm residual before accepting the solve, project with
   the matching gradient, and extend valid velocities four cells into air.
5. Publish through the standard textures and measure derived volume, divergence,
   speed, pressure and approximate kinetic energy.

All stencils, reductions, CFL decisions, pressure acceptance and publication gates
run on the GPU. A frame is submitted synchronously as a bounded indirect work
graph; the renderer follows it on the same queue. Like the other methods, each
`advanceTo(target)` submits at most one configured maximum step and reports its
actual submitted time. A wall clock far ahead of the GPU cannot create an
unbounded catch-up submission. Offline callers repeat advances to reach a target.
No pressure iteration or substep waits for a host receipt. Bind groups are cached,
and the pressure loop reuses prepared commands. There is no recurring CPU field traversal.

A three-buffer, 128-byte diagnostic ring samples once per submitted frame when a
slot is free; a busy ring skips the sample without waiting. These asynchronous
maps never gate submission or presentation. Explicit offline
`awaitFrameCompletion()` / `readStats()` calls can request a current receipt. Initialization also submits its publication without
waiting for the initial diagnostic map. Scene seeding still constructs initial
fields on the CPU once. Performance tracing is also asynchronous.

The host bounds each requested interval using the latest completed speed with
2× speed headroom, gravity accrued since that sample, and static material bounds.
This keeps command encoding fixed at three continuation slots even when the UI
clock is far ahead. It never waits for a sample: the GPU selects and validates
every actual substep from current fields. If a stale estimate still exceeds the
continuation capacity, the frame fails before publication; reduce the maximum
step and reset. The pressure iteration cap likewise rejects an unconverged frame.
WebGPU still encodes inactive indirect dispatches, so high iteration limits have
command overhead even when those dispatches execute no numerical work.

Defaults are a maximum step of 1/120 s, CFL 0.5 and an absolute divergence residual
of 0.001 s^-1, with a 256-iteration pressure safety limit (adjustable to 2048).
Gravity, explicit viscosity and capillarity also limit the timestep.
There is no volume correction or artificial velocity damping. Level-set volume
drift and numerical dissipation are expected and should remain visible in A/B
results. Diagonal PCG keeps the implementation small; it is not a multigrid solver
and its iteration count grows with resolution.

## Comparison scope

Use matching scene geometry, cell dimensions, fluid properties and elapsed
physical time. Compare timing per simulated second, volume drift, divergence,
surface evolution and wave phase; equal frame counts alone are insufficient.
MAC always allocates the full finest lattice. If the Geometric configuration uses
coarse/fine regions, report its actual cell layout alongside the MAC dimensions.
Use smaller timesteps and finer grids to check convergence. Do not infer a speed
advantage from the method name or from a single fixture.

Fixed voxel solids use binary occupancy and free-slip walls. The container top
can be open or closed. Inflows and dynamic rigid bodies are rejected explicitly.
This is a single-phase liquid solver with atmospheric air pressure; it does not
resolve air motion, wetting/contact-angle physics, or cut-cell solid boundaries.

Numerical tests cover hydrostatic balance, signed pressure, gravitational free
fall, the Laplace pressure jump, wall rest, dam-break motion, linear gravity-wave
phase, and rejection of an under-solved pressure step. Run
`npm run test:dawn -- uniform-mac` with no browser or other Dawn process using
the repository's WebGPU lease.
