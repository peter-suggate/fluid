# Hierarchical WebGPU solver

The WebGPU fluid implementation uses one sparse, brick-based hierarchy for all
configurations. `hierarchy.levels = 1` is the uniform-grid specialization of the
same code path; it does not select a separate solver.

## Representation

- Cartesian leaf bricks contain `4 × 4 × 4` cells.
- Neighboring leaves are kept 2:1 balanced.
- A finest-brick page table provides constant-time leaf lookup from WGSL.
- Each cell stores negative-face velocity plus VOF fraction and positive-face
  velocity plus pressure in two `vec4<f32>` values.
- Only leaf cells, leaf metadata, the page table, PCG vectors, rigid exchange
  buffers, and bounded transient surface buffers are allocated. No dense
  finest-resolution presentation volume is reconstructed.
- The compute layout requires nine storage buffers per shader stage. Device
  creation requests that adapter limit explicitly and reports a clear fallback
  status on adapters that expose only the WebGPU baseline of eight.

The main tuning controls are hierarchy depth, interface and solid halo widths,
regrid cadence, coarsening delay, refinement error thresholds, and the active
brick budget. The UI exposes the controls most useful for visual comparisons;
the complete contract is serialized in the scene JSON.

## Surface rendering

- A lightweight compute pass compacts only bricks whose local or positive
  neighbor samples straddle `alpha = 0.5`. This catches interfaces aligned
  exactly with brick boundaries without polygonizing bulk water or empty space.
- One workgroup per compacted brick extracts a consistently oriented marching-
  tetrahedra surface. Workgroup prefix allocation performs one global vertex
  reservation per brick, and bounded indirect buffers prevent overflow from
  becoming an out-of-bounds write.
- The opaque scene is copied once for screen-space refraction. Separate back-
  and front-face depth passes estimate liquid thickness, while the final water
  pass applies Fresnel reflection, Snell refraction, Beer–Lambert attenuation,
  scattering, and foreground rigid-body occlusion.
- Surface bind groups are invalidated when the solver is recreated and refreshed
  whenever ping-pong cells or topology buffers change after a substep or regrid.
- If surface shader creation fails, the renderer retains the previous sparse
  trilinear ray-march path as a compatibility fallback.

## Numerics

- Conservative, bounded VOF fluxes use donor and receiver capacity limiting.
- Coarse/fine faces are subdivided into canonical fine-side fluxes. The same
  integrated face flux is used for advection and divergence.
- Dynamic regridding tags the free surface, velocity detail, and moving rigid
  body neighborhoods. Transfer integrates on the finest logical lattice, making
  constant-state prolongation and volume restriction conservative.
- Pressure uses a matrix-free composite operator and Jacobi-preconditioned CG.
  Ghost-fluid free-surface coefficients and coarse/fine face areas are assembled
  inside each operator evaluation.
- Rigid pressure coupling adds the rank-six `Bᵀ M⁻¹ B` body term inside PCG.
  Pressure impulses and the bounded viscous/no-slip exchange are returned to the
  CPU rigid integrator without reapplying a readback.

## Invariants and diagnostics

The hierarchy builder is deterministic, page-table complete, budget bounded,
and 2:1 balanced. Tests cover the one-level uniform specialization, reciprocal
coarse/fine adjacency, conservative restriction/prolongation, dynamic transfer,
and symmetry/positive-semidefiniteness of the rigid Schur term. Live diagnostics
report active and equivalent-uniform cell counts, compression, topology changes,
VOF drift, maximum speed, and dam-front position.

The CPU binary64 MAC solver remains an independent validation oracle. It is not
a second WebGPU production path.
