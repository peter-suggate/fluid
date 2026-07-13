# Fluid Solver Laboratory

This repository is a validation-first three-dimensional Eulerian free-surface
fluid laboratory. The current application is React, TypeScript, WGSL, and
WebGPU; it runs locally and requires no cloud simulation service.

Implementation is intentionally gated by quantitative evidence. The current
browser build includes the deterministic rigid-body and MAC-grid CPU oracles,
deterministic buoyancy/drag quadrature with paired fluid reaction impulses, and
a sparse hierarchical WebGPU Eulerian path. The workbench starts with five
reproducible scenarios—dam break, wave tank, buoyancy contrast, splash impact,
and hydrostatic still water—and exposes hierarchy depth and quality as its
primary comparison controls. Detailed scene, body, solver, and diagnostic
controls use progressive disclosure. WebGPU quality presets scale the finest
resolution while sparse leaf bricks avoid allocating the full equivalent grid.
The presentation renderer classifies sparse bricks whose VOF range crosses the
free surface, extracts their `alpha = 0.5` interface on the GPU, and draws the
result through bounded indirect buffers. Front/back surface passes provide
thickness for screen-space refraction, Fresnel reflection, and Beer–Lambert
absorption at native canvas resolution. The former hierarchical volume march is
retained as a shader-compilation fallback but is not used by the normal WebGPU
surface path.
The live performance drawer reports queue-safe end-to-end latency, completed
simulation throughput, solver lag, presentation interval, blocked frames, and
wall-clock CPU simulation, upload, encoding, and orchestration costs. Render
timestamps remain available. Intrusive multi-pass compute timestamps are
disabled by default because Chrome's Metal backend can serialize the queue for
seconds; the dormant stage markers separate advection, PCG control and copies,
pressure, projection, immersed-body coupling, reductions, and unattributed
compute when explicitly enabled for controlled profiling.
The web preview starts with a queue-safe uniform Balanced profile: a compact GPU
grid, an ordered Jacobi pressure pass, an 8 ms interactive clock, idle-time
diagnostic readbacks, and no adaptive remap stalls. Adaptive depths and the
composite PCG detail path remain explicitly selectable for fidelity work.
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

The CPU MAC/PCG path remains the pressure-validation oracle. The WebGPU path
uses one sparse 2:1 leaf-brick hierarchy with packed staggered face velocities,
a ghost-fluid atmospheric boundary, matrix-free composite PCG, conservative
bounded VOF transport, and pressure-level rigid coupling. Setting hierarchy
depth to one produces a uniform grid through that same implementation. The
renderer reads leaf buffers directly; no dense presentation volume, equilibrium
blend, presentation smoothing, or global volume rescaling is applied. See
[`HIERARCHICAL_WEBGPU.md`](HIERARCHICAL_WEBGPU.md) for its invariants and tuning
contract. The native Metal path additionally uses
analytic moving cut-cell volume/face fractions, aperture-aware pressure and VOF
fluxes, conservative cover/uncover remapping, and equal-and-opposite pressure
traction integrated into GPU-resident rigid-body motion. The native smoke gate
checks both liquid-volume drift and the buoyancy contrast between submerged
light and dense bodies.
