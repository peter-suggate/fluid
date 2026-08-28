# Fluid Solver Laboratory

This repository is a validation-first three-dimensional Eulerian free-surface
fluid laboratory. The current application is React, TypeScript, WGSL, and
WebGPU; it runs locally and requires no cloud simulation service.

Implementation is intentionally gated by quantitative evidence. The current
browser build includes the deterministic rigid-body and MAC-grid CPU oracles,
deterministic buoyancy/drag quadrature with paired fluid reaction impulses, and
a high-resolution WebGPU Eulerian path. The GPU uses a restricted tall-cell
grid: each x/z column has one variable-height bottom cell and a moving band of
24–40 cubic cells around the free surface. Quality presets retain approximately
the horizontal resolution of the former 110k, 500k, and 1.2m uniform grids
while storing far fewer samples in deep domains. The evolving surface-density
field is rendered directly.
The compute panel exposes an independent **Tall cells / Adaptive / Uniform**
comparison. Adaptive uses Narita et al.'s horizontally adaptive quadtree tall
cells and variational pressure projection; uniform uses the same finest x/z and cubic-equivalent y
resolution, so timing and visual comparisons do not hide a resolution
reduction. A one-click deep-water scene reproduces the depth-scaling benchmark.
The presentation renderer samples the surface-density field trilinearly at a
quality-aware stride, reconstructs an interface cell from eight cached samples, and uses
subcell Newton refinement with analytic trilinear normals, front/back thickness,
Fresnel reflection, and Beer–Lambert absorption. Raster optics renders at a
bounded internal resolution and upscales into the native canvas.
The live performance sidebar separates hardware-timestamped GPU advection,
pressure, projection, immersed-body coupling, sparse-fluid residency and scene
publication, reductions, queue/copy overhead, and raster rendering from
wall-clock CPU simulation, upload, encoding, and orchestration costs, with a
shared 60 Hz budget and recent-frame history.
The tall-cell, adaptive, and uniform GPU paths use the same donor/receiver
limited conservative VOF face flux. Tall columns integrate the shared deep
face with bounded stratified quadrature and expand only the `D`-bounded
tall/regular mismatch, so opposing control volumes use the same flux without a
global volume correction. Uniform velocity transport is configurable between
one-pass semi-Lagrangian advection (the default) and bounded MacCormack
on packed samples, ghost-fluid free-surface pressure, a restricted full-cycle
multigrid solve, physical molecular viscosity, and compact reductions. The tall-cell paper omits capillarity, so the tall method's
paper-core path uses `sigma=0`; the retained uniform path continues to support
the scene's surface-tension value.

## Stage 1 documents

The interactive paper presets reproduce the tank inflow (Figure 3), dam break
with boxes (Figure 4), and jet-past-sphere benchmark (Figure 6).

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
- [`docs/TALL_CELL_WEBGPU.md`](docs/TALL_CELL_WEBGPU.md) — restricted tall-cell
  layout, conservative remeshing, performance model, and departures from the
  reference paper.
- [`docs/TALL_CELLS_PAPER.md`](docs/TALL_CELLS_PAPER.md) — searchable technical
  extraction of the paper's equations, algorithms, and stability constraints.
- [`docs/TALL_CELL_STABILITY.md`](docs/TALL_CELL_STABILITY.md) — rigorous
  paper-to-code audit, eruption reproduction, corrections, and live diagnostic
  contract.
- [`docs/TALL_CELL_BENCHMARK.md`](docs/TALL_CELL_BENCHMARK.md) — matched browser
  comparison against the retained uniform WebGPU solver.
- [`docs/QUADTREE_TALL_CELLS.md`](docs/QUADTREE_TALL_CELLS.md) — paper-to-code
  specification, corrected discretization details, and verification contract.
- [`docs/SPARSE_VOXEL_OCTREE_RENDERER_MIGRATION.md`](docs/SPARSE_VOXEL_OCTREE_RENDERER_MIGRATION.md)
  — tracked milestones, acceptance gates, decisions, risks, and evidence for
  the direct SVO renderer migration with retained raster compatibility.
- [`docs/GPU_STAGE_DIAGNOSTICS_CAPTURE_PLAN.md`](docs/GPU_STAGE_DIAGNOSTICS_CAPTURE_PLAN.md)
  — one-shot simulation/SVO stage capture, resource visualization, workload
  interpretation, and capture-off performance validation contract.
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

Large Sparse CM12 changes have one canonical, time-bounded native-Dawn
confidence gate:

```bash
npm run test:dawn:sparse-cm12
```

It covers correctness, adaptivity, live edits, both far-wall maps, and mini32 /
mini64 hardware-timestamped performance baselines in at most three minutes.
See [`docs/SPARSE_CM12_DAWN_REGRESSION.md`](docs/SPARSE_CM12_DAWN_REGRESSION.md)
for the matrix, focused lane commands, and rebaseline policy.

## Current numerical boundary

The CPU MAC/PCG path remains the pressure-validation oracle. The adaptive GPU
path uses an f32 sparse octree with selectable Losasso and Power coarse
dynamics, persistent MGPCG, a structured fine level-set band, and GPU-resident
topology publication. A separate dense uniform WebGPU method is retained as a
matched-lattice reference; it is never an adaptive runtime fallback. The
renderer consumes either the octree's sparse surface publications or the
uniform method's direct dense field.
