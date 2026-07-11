# Browser Pivot — Architecture Decision Record

Status: accepted for the current implementation  
Date: 2026-07-12

## Decision

Fluid Lab is browser-first for the current implementation. The interactive
application uses React and TypeScript. GPU compute and rendering use WebGPU with
WGSL. Simulation work runs outside the UI thread. The double-precision CPU
reference implementation will be C++20 or Rust compiled to WebAssembly and run
in a dedicated Web Worker; a plain TypeScript oracle remains acceptable only
for very small bring-up tests.

The original Metal architecture remains a possible future native backend. The
governing equations, scene format, validation contract, and evidence rules do
not change.

## Browser subsystem boundaries

```text
React UI thread
  scene editor / camera / metrics / renderer presentation
       | immutable commands and snapshots
       v
Simulation Worker
  clock / solver orchestration / CPU-WASM oracle / diagnostics
       | transferable snapshots and metric batches
       v
WebGPU device
  Eulerian kernels / DFSPH kernels / reductions / render kernels
```

The UI never directly mutates solver arrays. Commands are sequenced and carry a
simulation-time index. Render snapshots are immutable. GPU-to-CPU readback is
batched and never performed merely to animate the presentation view.

## Precision and reproducibility

- CPU reference: IEEE-754 binary64 in WebAssembly, deterministic single worker.
- GPU: WGSL `f32`; `f16` is prohibited for physics state.
- GPU reductions: deterministic validation variant plus faster interactive
  variant, both labelled in run metadata.
- Browser, WebGPU adapter features/limits, OS, build version, and precision are
  recorded in every run manifest.
- Local scene preferences use `localStorage`; exported scenes and run records
  are explicit JSON downloads. No cloud service is required.

## Performance policy

Each simulation step uses one command encoder and as few queue submissions as
possible. Iterative kernels are fused where doing so preserves inspectability.
Convergence readback is amortized or performed through GPU-side indirect work;
validation mode may accept slower synchronization. Timestamp queries are used
only when supported and their availability is reported.

## Browser compatibility

WebGPU is the primary path. The application presents a diagnostic fallback when
`navigator.gpu` or a suitable adapter is unavailable. The scientific UI, scene
editing, serialization, and CPU validation remain usable without WebGPU.
