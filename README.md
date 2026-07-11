# Fluid Solver Laboratory

This repository is for a validation-first browser laboratory comparing a
three-dimensional Eulerian free-surface solver with a particle solver under a
shared physical scene. The current application is React, TypeScript, WGSL, and
WebGPU; it runs locally and requires no cloud simulation service.

Implementation is intentionally gated by quantitative evidence. **Stage 4 is
now implemented:** the Stage 3 deterministic rigid-body reference plus a
three-dimensional CPU-binary64 staggered MAC-grid fluid with RK2 advection,
explicit viscosity, marker free surface, closed-wall flux enforcement,
matrix-free Jacobi-PCG pressure projection, adaptive time-step diagnostics, and
a dam-break default. WebGPU renders the solver occupancy directly; the water is
no longer a decorative plane.

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

## Gate to Stage 5

Stage 5 introduces the independent DFSPH particle reference. Two-way
fluid/rigid coupling remains Stage 7 and is deliberately not claimed by this
build. The interactive CPU grid is resolution-capped and reports its effective
dimensions; later stages move verified kernels to WebGPU without changing the
physical scene contract.
