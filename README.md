# Fluid Solver Laboratory

This repository is a validation-first three-dimensional Eulerian free-surface
fluid laboratory. The current application is React, TypeScript, WGSL, and
WebGPU; it runs locally and requires no cloud simulation service.

Implementation is intentionally gated by quantitative evidence. The current
browser build includes the deterministic rigid-body and MAC-grid CPU oracles,
deterministic buoyancy/drag quadrature with paired fluid reaction impulses, and
a high-resolution WebGPU Eulerian path. WebGPU quality presets allocate approximately
110k, 500k, or 1.2m cells and render the evolving volume fraction directly.
The presentation renderer samples the VOF field trilinearly at a quality-aware
stride, reconstructs an interface cell from eight cached samples, and uses
subcell Newton refinement with analytic trilinear normals, front/back thickness,
Fresnel reflection, and Beer–Lambert absorption. It renders at native canvas
resolution; traversal skipping and cached interface work keep the
raymarch within its GPU budget without reducing surface fidelity.
The live performance drawer separates hardware-timestamped GPU advection,
pressure, projection, immersed-body coupling, reductions, queue/copy overhead,
and raymarch rendering from wall-clock CPU simulation, upload, encoding, and
orchestration costs, with a shared 60 Hz budget and recent-frame history.
The GPU transport path uses conservative bounded donor-cell VOF fluxes,
midpoint RK2 velocity backtracing on packed staggered faces, ghost-fluid
free-surface pressure, balanced-force continuum surface tension, Smagorinsky
LES viscosity, compact reductions, and quality-aware CFL/capillary substepping.

## Stage 1 documents

- [`docs/STAGE1_ARCHITECTURE.md`](docs/STAGE1_ARCHITECTURE.md) — selected
  numerical methods, architecture, validation plan, risks, and decisions.
- [`docs/WEB_ARCHITECTURE.md`](docs/WEB_ARCHITECTURE.md) — accepted browser
  pivot and worker/WebGPU/WASM boundaries.
- [`docs/STAGE2_ACCEPTANCE.md`](docs/STAGE2_ACCEPTANCE.md) — browser shell test
  contract and interaction gates.
- [`docs/STAGE3_ACCEPTANCE.md`](docs/STAGE3_ACCEPTANCE.md) — rigid-body equations,
  collision approximations, and quantitative gates.
- [`docs/STAGE4_ACCEPTANCE.md`](docs/STAGE4_ACCEPTANCE.md) — Eulerian equations,
  free-surface choices, volume-preserving corner dam-break initialization, and quantitative gates.
- [`docs/STAGE6_7_ACCEPTANCE.md`](docs/STAGE6_7_ACCEPTANCE.md) — buoyancy, drag,
  displaced volume, torque, and conservative reaction-impulse gates.
- [`docs/STAGE8_GPU_ACCEPTANCE.md`](docs/STAGE8_GPU_ACCEPTANCE.md) — WebGPU
  texture layout, projection baseline, quality presets, and limitations.
- [`docs/SCENE_FORMAT.md`](docs/SCENE_FORMAT.md) — canonical SI scene and run
  record format.
- [`docs/COMPARABILITY.md`](docs/COMPARABILITY.md) — resolution, workload, and
  error comparison rules.
- [`docs/PHYSICS_CONTRACT.json`](docs/PHYSICS_CONTRACT.json) — initial
  machine-readable invariants and acceptance thresholds.
- [`PHYSICS_EVIDENCE.md`](PHYSICS_EVIDENCE.md) — evidence ledger. Claims remain
  pending until measured by executable tests.

## Run locally

```bash
npm install
npm run dev
```

Use `npm test` for the deterministic shell contract and production build.

## Native macOS / Metal

The repository also includes an Eulerian-only native Metal application tuned
for Apple Silicon. It shares the canonical default scene with the browser but
keeps Swift/AppKit and MSL backend code isolated under [`native/`](native/).

```bash
npm run native:run       # build and launch
npm run native:app       # create native/.build/Fluid Lab Metal.app
npm run native:test      # native schema tests
npm run native:smoke     # Metal device, shader, pipeline, and allocation test
```

See [`native/README.md`](native/README.md) for architecture and performance
details.

## Current numerical boundary

The CPU MAC/PCG path remains the pressure-validation oracle. The GPU path uses
an f32 cell-centred VOF field with packed staggered positive-face velocities,
compatible divergence/gradient operators, a ghost-fluid atmospheric boundary,
weighted Jacobi, and conservative upwind volume transport. The renderer reads
the physical VOF field directly; no equilibrium blend, presentation smoothing,
or global volume rescaling is applied. Resolved cut-cell traction and asynchronously reduced GPU pressure residuals
remain research/optimization work and are not claimed as validated.
