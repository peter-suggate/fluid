# Fluid Solver Laboratory

This repository is for a validation-first browser laboratory comparing a
three-dimensional Eulerian free-surface solver with a particle solver under a
shared physical scene. The current application is React, TypeScript, WGSL, and
WebGPU; it runs locally and requires no cloud simulation service.

Implementation is intentionally gated by quantitative evidence. **Stage 3 is
now implemented:** deterministic CPU-binary64 rigid bodies with analytic mass
properties, gravity, quaternion integration, container and body impulses, all
four primitive renderers, interactive add/drop/edit controls, diagnostics, and
regression benchmarks. The animated water remains a presentation field, not a
numerical fluid solution.

## Stage 1 documents

- [`docs/STAGE1_ARCHITECTURE.md`](docs/STAGE1_ARCHITECTURE.md) — selected
  numerical methods, architecture, validation plan, risks, and decisions.
- [`docs/WEB_ARCHITECTURE.md`](docs/WEB_ARCHITECTURE.md) — accepted browser
  pivot and worker/WebGPU/WASM boundaries.
- [`docs/STAGE2_ACCEPTANCE.md`](docs/STAGE2_ACCEPTANCE.md) — browser shell test
  contract and interaction gates.
- [`docs/STAGE3_ACCEPTANCE.md`](docs/STAGE3_ACCEPTANCE.md) — rigid-body equations,
  collision approximations, and quantitative gates.
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

## Gate to Stage 4

Stage 4 introduces the Eulerian CPU reference fluid: MAC storage and operators,
free surface, advection, viscosity, pressure projection, boundaries, adaptive
time stepping, and diagnostics. It must pass manufactured projection, static
water, hydrostatic pressure, volume, resolution, and time-step tests before any
fluid force is applied to the Stage 3 rigid bodies.
