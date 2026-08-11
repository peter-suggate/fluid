# Uniform GPU Reference

## Purpose

The uniform method is a deliberately transparent WebGPU baseline for the
adaptive Losasso and Power implementations. It answers a narrow comparison
question: what does the same scene-authored finest lattice cost and produce
when every cell is stored and advanced directly?

It is a reference, not a third adaptive research path. It has no octree
topology, sparse fluid residency, coarse-backend seam, or separate fine
level-set hierarchy.

## Numerical contract

- Resolution is `SceneDescription.voxelDomain.finestCellSize_m`, rounded by the
  shared lattice planner in x, y, and z.
- Liquid is a dense cell-centred conservative VOF field.
- Velocity uses the existing bounded MacCormack GPU transport and the same
  scene forces, walls, terrain, inflow, and immersed-body inputs.
- Pressure uses full-grid weighted Jacobi and a ghost-fluid free-surface
  stencil. The only method control is the number of pressure sweeps.
- A GPU-only global controller compares conservative VOF mass with its initial
  (plus authored inflow) reference and redistributes a bounded correction over
  the one-cell interface band. Its exponential response and interface-speed
  limit are expressed in simulated seconds, not encoded step count.
- Immersed-body VOF displacement uses the same `dt`-scaled relaxation as its
  velocity coupling. Subdividing one simulated second therefore cannot apply
  the displacement operator more strongly.
- Conservative density sharpening removes diffuse air-side VOF and scatters
  that exact fixed-point mass back to the interface before residual volume
  correction. This prevents repeated small steps from dissolving thin sheets.
- The renderer contours a separate two-pass `[1 2 1]^3` reconstruction of the
  dense VOF, guided by 25% of the conservative source so a one-cell sheet
  remains above the 0.5 contour. This smooth fractional texture is
  presentation-only: advection,
  pressure, volume diagnostics, and correction continue to use the
  conservative field. Full-scene presentation may still build the ordinary
  non-fluid SVO sidecar.

Quality presets change only the pressure budget: balanced/high/ultra select
64/80/96 sweeps. Spatial resolution remains scene-authored, so changing
quality cannot hide a grid-size difference.

## Architecture

- `lib/webgpu-uniform-reference.ts` owns the dense allocations, bind groups,
  pipeline cache, advance graph, diagnostics, and renderer-facing textures.
- `lib/webgpu-uniform-reference.wgsl.ts` owns the complete dense compute
  program. It is not imported by the adaptive host.
- `lib/webgpu-octree-eulerian.ts` is adaptive-only and always constructs a
  `WebGPUOctreeProjection`; it has no dense-host option or fallback graph.
- `lib/methods/uniform.ts` and `lib/methods/octree.ts` are separate method
  plugins that meet only at the generic `SimulationMethod` interface.

The retired shared `webgpu-uniform-eulerian` solver has been removed. This
separation is intentional: adding a feature to the reference cannot silently
change Losasso or Power, and adaptive compatibility code cannot enter the
baseline.

## Acceptance

- Method resolution and UI hydration retain `uniform` rather than falling back
  to `octree`.
- Construction publishes `gridKind: "uniform"`, compression ratio 1, and no
  adaptive surface source.
- Every dense compute entry point compiles through the staged asynchronous
  initialization path.
- A one-step native WebGPU test completes without validation errors and
  publishes finite volume and velocity diagnostics.
- The renderer-facing surface texture is distinct from the conservative VOF,
  including at `t=0`.

## Known limitations

Memory and work scale with the complete 3D lattice. Weighted Jacobi is simple
and reproducible but converges much more slowly than either adaptive MGPCG
backend, especially in deep domains. The method is therefore a comparison
baseline, not the default and not a binary64 numerical oracle.
