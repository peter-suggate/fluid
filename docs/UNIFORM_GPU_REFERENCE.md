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
- Liquid is the paper's dense cell-centred surface-density field `rho`; its
  `0.5` isosurface defines the interface.
- Section 3.4 transport persists `gamma`, constructs donor-column `beta` by a
  backward scatter, applies the `max(1, beta)` rescaling, forward-scatters the
  missing density and `gamma`, and performs the paper's paired diffusion.
  Fixed-point scatter buffers keep the operation GPU-resident and locally
  conservative. The paper permits one to seven dimension-by-dimension
  diffusion repetitions but does not state the axis traversal order. This
  reference uses one fixed x, y, z repetition; its resulting axis-order bias is
  reported by the symmetry benchmark rather than hidden by a custom operator.
- The step graph follows Algorithm 1: velocity extrapolation,
  conservative density advection and local sharpening, bounded MacCormack
  velocity advection plus forces, then incompressibility enforcement.
- Section 3.3 follows the sources it delegates to rather than a nearest-face
  surrogate. CM11b Equation 7 extends each MAC component by
  `du/dtau = -(grad(phi)/|grad(phi)|).grad(u)`. The
  Jeong-Ross-Whitaker active front runs over the full regular MAC domain using
  their Godunov Equation 2.1, source
  distance zero, unknown distance infinity, synchronous updates, next-pass
  activation, decreasing-distance updates, and convergence removal. Each
  component is extended with the first-order upwind discretization of CM11b
  Equation 7 using the converged Eikonal distance. The uniform solver prepares
  the same `rho' = rho/V` and fractional face authority used by projection
  before each current and predicted extension.
- FIM termination is the source-stated empty active list, implemented by a GPU
  active counter driving an indirect update chain. The encoded full-domain
  ceiling is a device-safety guard, not the stopping criterion; a non-empty
  terminal mask is a conformance failure. Every dependent texture transition
  is a separate WebGPU usage scope.
- This is deliberately not a literal implementation of CM11b's far-field
  hierarchy. That accelerator depends on the paper's compressed `H/h`
  hierarchy of tall-cell grids, which does not exist on the uniform topology.
  Applying its regular-looking restriction/prolongation sweep without the
  prerequisite tall-cell coverage left valid MAC faces unknown. Rather than
  invent a root or fallback fill, the uniform specialization uses the supplied
  JRW method directly everywhere; the published extension PDE is unchanged.
- CM11b stores velocity collocated and does not publish a staggered-MAC
  stencil or a numeric FIM convergence tolerance.
  The reference therefore applies the stated scalar upwind equation
  independently on each positive-face lattice and uses one local binary16 ulp
  as the convergence threshold. Comparison readback reports the terminal
  active count and every authoritative open face that remains unknown; both
  must be zero.
- Section 3.5 sharpening uses the paper's fictitious-time correction and
  returns removed air-side density by tracing along `grad(rho)` to the nearby
  interface. Solid destination weights are removed and renormalized.
- Partial-solid cells use `rho' = rho / V`, fractional open-face aperture
  areas for density, and oriented/moving solid face velocities. Pressure uses
  the distinct CM11a face-centred overlapping-cell volume; a closed aligned
  domain wall therefore has an inferred dual volume of `0.5`, while its mass
  aperture remains zero. Density above `V` is traced one cell
  along the solid signed-distance gradient and conservatively scattered into open
  cells. Unplaceable excess is reported explicitly rather than silently
  returned inside a solid. The pressure RHS uses the paper's bounded
  excess-density divergence.
- Pressure uses the variational cut-cell operator and ghost-fluid distance
  `phi = -(rho' - 0.5) dx`, solved by the supplied CM11a LCP hierarchy:
  three Full-Cycles, four V-Cycles, and four pre/post PRBGS sweeps. The
  coarsest grid iterates to the `1e-4 s^-1` absolute infinity tolerance
  published for the TallCells GPU/single-precision realization. CM11a itself
  does not prescribe a tolerance; TallCells' `1e-8` comparison used
  double-precision CPU arithmetic. The pressure-system residual is multiplied
  by `dt / rho` before this divergence-unit gate and residual telemetry.
  CM11a's overview describes two colour passes followed by projection, while
  Equation 18 says to enforce `p_min` while smoothing. The implementation
  applies Equation 18 to each updated colour before the opposite colour reads
  it, and retains the post-pair projection; delaying the clamp until after both
  colours admits a non-KKT fixed point when bound and free rows are adjacent.
  Reaching the safety cap is published as a failure diagnostic.
- No global volume controller creates or deletes density. Drift telemetry is a
  measurement of the conservative field rather than feedback into it.
- Rendering can opt into the Section 3.8 detail reconstruction: Gaussian-blur
  `g = 2 min(rho, 0.5)` with `sigma = 2 dx`, then contour
  `rho / min(max(g, 0.01), 1)`. This field is presentation-only and never feeds
  transport or projection. It is off by default because the paper's Results
  states that density post-processing was disabled unless otherwise noted.

The pressure schedule is fixed by CM11a and is not a quality control. Spatial
resolution remains scene-authored, so changing quality cannot hide a grid-size
difference.

## Architecture

- `lib/webgpu-uniform-reference.ts` owns the dense allocations, bind groups,
  pipeline cache, advance graph, diagnostics, and renderer-facing textures.
- `lib/webgpu-uniform-reference.wgsl.ts` owns the complete dense compute
  program. It is not imported by the adaptive host.
- `lib/webgpu-uniform-pressure-multigrid.ts` and its WGSL fragment own the
  dense CM11a hierarchy and fixed cycle schedule.
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
- A one-step native WebGPU smoke run completes without validation errors and
  publishes finite volume and velocity diagnostics.
- By default the renderer contours the conservative density directly. When
  Section 3.8 is explicitly enabled, the renderer-facing surface texture is a
  distinct presentation-only reconstruction, including at `t=0`.

## Known limitations

Memory and work scale with the complete 3D lattice and the CM11a pyramid.
Fixed-point WebGPU scatters also
conserve mass only to their quantization precision. The paper's unspecified
dimension traversal order in gamma diffusion is observable as an x/y/z
operator-order bias in strict symmetry diagnostics. The method is therefore a
comparison baseline, not the default and not a binary64 numerical oracle.
