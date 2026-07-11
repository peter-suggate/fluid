# Fluid Solver Laboratory

This repository is for a validation-first browser laboratory comparing a
three-dimensional Eulerian free-surface solver with a particle solver under a
shared physical scene. The current application is React, TypeScript, WGSL, and
WebGPU; it runs locally and requires no cloud simulation service.

Implementation is intentionally gated by quantitative evidence. **Stage 2 is
now implemented:** an interactive WebGPU scientific shell, scene editor,
camera, diagnostics, persistence, and validation runner. The animated water is
a presentation field, not a numerical fluid solution. No physics claim is made
until the CPU reference solvers pass their later stage gates.

## Stage 1 documents

- [`docs/STAGE1_ARCHITECTURE.md`](docs/STAGE1_ARCHITECTURE.md) — selected
  numerical methods, architecture, validation plan, risks, and decisions.
- [`docs/WEB_ARCHITECTURE.md`](docs/WEB_ARCHITECTURE.md) — accepted browser
  pivot and worker/WebGPU/WASM boundaries.
- [`docs/STAGE2_ACCEPTANCE.md`](docs/STAGE2_ACCEPTANCE.md) — browser shell test
  contract and interaction gates.
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

## Gate to Stage 3

Stage 3 introduces rigid bodies without fluid: analytic mass properties,
gravity, integration, primitive and container collisions, and quantitative
free-fall/momentum tests. It must not treat the Stage 2 presentation field as
fluid state.
