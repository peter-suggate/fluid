# Fluid Solver Laboratory

This repository is for a validation-first browser laboratory comparing a
three-dimensional Eulerian free-surface solver with a particle solver under a
shared physical scene. The current application is React, TypeScript, WGSL, and
WebGPU; it runs locally and requires no cloud simulation service.

Implementation is intentionally gated by quantitative evidence. The current
browser build includes the deterministic rigid-body and MAC-grid CPU oracles, a
PBF particle CPU oracle with exact hashed-neighbour regression, deterministic
buoyancy/drag quadrature with paired fluid reaction impulses, and a high-
resolution WebGPU Eulerian path. WebGPU quality presets allocate approximately
110k, 500k, or 1.2m cells and render the evolving volume fraction directly.

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
  free-surface choices, dam-break initialization, and quantitative gates.
- [`docs/STAGE5_ACCEPTANCE.md`](docs/STAGE5_ACCEPTANCE.md) — PBF kernel,
  neighbour-search, density, boundary, and stability gates.
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

## Current numerical boundary

The CPU MAC/PCG path remains the pressure-validation oracle. The GPU path uses
a collocated f32 volume field, compatible difference operators, weighted
Jacobi, conservative upwind volume transport, and an explicitly approximate
hydrostatic column predictor. The particle mode is PBF, not DFSPH. Resolved
cut-cell traction, GPU PBF, and asynchronously reduced GPU pressure residuals
remain research/optimization work and are not claimed as validated.
