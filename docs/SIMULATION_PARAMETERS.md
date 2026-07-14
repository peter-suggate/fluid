# Simulation parameter review

A complete audit of every user-facing parameter: where it lives, which solver
consumes it, and why it is (or is not) exposed. This taxonomy is enforced by
the code layout — scene-physical parameters live in `SceneDescription` and are
edited in the scene configuration modal; solver parameters are declared by
method plugins in `lib/methods/` and rendered generically by the method panel.

## Findings from the audit

1. **`numerics.pressureMaxIterations` was dead.** The old UI exposed a
   "Pressure iterations" slider (20–1000) that no solver read: the uniform GPU
   solver hardcoded 64/80/96 Jacobi sweeps by quality, the tall-cell and
   adaptive solvers used a fixed 2 or 8 multigrid V-cycles, and the CPU oracle
   hardcoded 240 PCG iterations at 1e-8 relative tolerance.
2. **`numerics.pressureRelativeTolerance` was dead** for the same reason.
3. **Pressure effort is not a common parameter.** The three GPU methods use
   structurally different solvers (damped Jacobi vs. restricted multigrid), so
   one shared iteration count cannot mean the same thing. It is now a
   method-specific parameter with a documented meaning per method, seeded by
   the quality preset.
4. Both fields are now genuinely consumed: they bound the **CPU reference
   PCG solve** (`eulerian-solver.ts`), which is the one solver whose
   convergence semantics match "max iterations + relative tolerance".

## Scene parameters (physical contract, serialized, method-independent)

Edited in the **scene configuration modal**; part of `SceneDescription`
(schema 1.0.0, shared with the native Metal app).

| Parameter | Field | Consumed by |
| --- | --- | --- |
| Container size | `container.{width,height,depth}_m` | all solvers, renderer |
| Fill fraction | `container.fillFraction` | initial fluid state |
| Top open/closed | `container.top` | boundary conditions |
| Wall slip | `container.fluidWallMode` | viscosity boundary term |
| Initial condition | `fluid.initialCondition` | initial fluid state |
| Inflow jet | `fluid.inflow` (center, radius, length, velocity, timing, ramp) | all solvers |
| Density | `fluid.density_kg_m3` | pressure RHS, coupling, buoyancy |
| Dynamic viscosity | `fluid.dynamicViscosity_Pa_s` | diffusion term |
| Surface tension | `fluid.surfaceTension_N_m` | CSF capillary term (uniform GPU + CPU; the tall-cell paper scope is σ=0) |
| Gravity | `fluid.gravity_m_s2` | body force, rigid bodies |
| Legacy run horizon | `duration_s` | retained for scene compatibility and fixed-duration benchmark metadata; it does not stop the interactive simulation |
| Random seed | `randomSeed` | reproducibility manifest |
| Rigid bodies | `rigidBodies[]` | rigid solver + coupling |

## Common numerics (serialized, consumed by every method)

| Parameter | Field | Meaning |
| --- | --- | --- |
| Fixed dt | `numerics.fixedDt_s` | rigid-body/validation clock step; CPU oracle step |
| Max dt | `numerics.maxDt_s` | cap on each GPU advance (`planGPUAdvance`) |
| Nominal cell | `nominalResolution.length_m` | CPU oracle MAC grid resolution |
| PCG budget | `numerics.pressureMaxIterations` | CPU reference PCG iteration cap (8–1000) |
| PCG tolerance | `numerics.pressureRelativeTolerance` | CPU reference PCG relative residual target |

## Method parameters (not serialized with the scene; plugin-declared)

Selected in the **method panel**. Coarse control is the quality preset
(balanced / high / ultra), which seeds every method parameter; fine control is
the per-parameter override (sparse, per method, resettable to the preset).

### Tall-cell VOF (`tall-cell`)
| Parameter | Tier | Meaning |
| --- | --- | --- |
| Pressure V-cycles | coarse | multigrid refinement cycles after the initial full cycle (default 2) |
| Surface columns | fine | target x/z column count (horizontal resolution) |
| Surface band layers | fine | cubic cells kept around the free surface (auto-grows if the surface spans more) |
| Neighbor base delta | fine | max tall-cell base step between adjacent columns |
| Remesh interval | fine | steps between band re-planning passes |

### Adaptive optical-layer VOF (`adaptive-optical-layer`)
| Parameter | Tier | Meaning |
| --- | --- | --- |
| Pressure V-cycles | coarse | multigrid refinement cycles (default 8; the moving layout converges slower) |
| Surface columns | fine | target x/z column count (horizontal resolution) |

Dilation bounds, airborne offsets, and the stage mask are planned per step
from Eq. 1 of the paper and grid height; they are diagnostics, not knobs.

### Uniform grid VOF (`uniform`)
| Parameter | Tier | Meaning |
| --- | --- | --- |
| Jacobi iterations | coarse | damped-Jacobi pressure sweeps per step (preset 64/80/96) |
| Grid columns | fine | target x/z column count for the matched full-depth cubic grid |

### CPU reference (`cpu-reference`)
| Parameter | Tier | Meaning |
| --- | --- | --- |
| Grid cell size | coarse | MAC resolution of the reference solve (preset 0.02 / 0.016 / 0.0125 m — matched to the GPU balanced/high/ultra equivalent cell counts) |

As the *active method* the CPU solve runs at this full resolution (slow, by
design — it is the reference). When a GPU method is active, the same solver
also runs as a cheap background oracle capped at 1 800 cells from the scene's
nominal length; that cap never applies to the reference method. The PCG
budget and relative tolerance come from the scene numerics in both roles.

## Increasing realism — recommended order

1. **Quality preset** (coarse): raises grid resolution and pressure effort
   together with method-appropriate ratios.
2. **Pressure effort** (coarse, per method): tightens incompressibility at
   impacts — visible as less volume "bounce" and crisper splashes.
3. **Fixed/max dt** (fine, scene modal): smaller steps reduce advection
   smearing and CFL alerts in impact-heavy scenes (the paper presets already
   use 1/180–1/360 s).
4. **Surface band layers / surface columns** (fine, tall-cell): more detail
   where the free surface actually is.
5. **Viscosity and surface tension** (scene modal): physical realism for
   small-scale features; note σ is outside the tall-cell paper scope.

## Adding a new method

Create `lib/methods/<id>.ts` exporting a `SimulationMethod`: identity copy,
`qualityLabels`, a `params` schema (each with `tier: "coarse" | "fine"`),
`presetFor(quality)`, and `createSolver(...)` returning the shared
`GPUSolverInstance` interface. Register it in `lib/methods/index.ts`. The
picker, quality selector, parameter controls, viewport badge, and diagnostics
text all render from the plugin; no UI changes are needed.
