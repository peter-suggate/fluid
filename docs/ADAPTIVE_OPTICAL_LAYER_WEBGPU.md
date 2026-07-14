# Adaptive Optical-Layer WebGPU Mode

This document records the independent WebGPU implementation of Narita and
Kanai, “Adaptive Optical Layers: Efficient Tall Cell Grids for Liquid
Simulation” (2026). The searchable paper extraction is
[`ADAPTIVE_OPTICAL_LAYERS_PAPER.md`](ADAPTIVE_OPTICAL_LAYERS_PAPER.md).

## Mode boundary

The application now has three explicit GPU methods:

| UI method | Solver identity | Layer policy |
| --- | --- | --- |
| Tall cells | `restricted-tall-cell` | Existing fixed moving band |
| Adaptive | `adaptive-optical-layer` | This paper's reconstructed layer every step |
| Uniform | `uniform` | Existing full-depth cubic comparison |

Selecting Adaptive constructs `WebGPUAdaptiveOpticalLayerSolver`; it does not
alter the fixed-band or uniform classes. The adaptive mode owns a distinct
initial layout, planner pipelines, planner textures, timing stage, diagnostic
counters, and run identity. It deliberately reuses the existing VOF transport,
packed tall-cell remap, immersed-body exchange, restricted multigrid pressure
operator, and renderer so an A/B/C comparison changes the grid policy rather
than the rest of the laboratory.

## Paper-to-code map

| Paper building block | Implementation |
| --- | --- |
| Linear horizontal velocity and constant vertical velocity fit | `fitTallColumnVelocity` and WGSL `estimateMotion` |
| Accumulated L1 fit error | CPU oracle and WGSL `estimateMotion` |
| Eq. 1, `clamp(alpha E Delta x, dmin, dmax)` | `errorToDilationCells`; `alpha=0.5`, `dmin=max(4,ceil(Ny/64))`, `dmax=max(dmin,ceil(Ny/8))` |
| Ground-connected surface detection | Bottom-up wet-column scan |
| Disconnected surfaces | Every wet cell with an exposed top or horizontal neighbor is seeded; distant components use `dair=ceil(Ny/16)` |
| Variable-radius Manhattan dilation | Exact radius-budget 3-D seed field, separable x pass, then z pass |
| Constrained smoothing | Five 9x9 averaging passes with `smoothedBase <= rawBase` |
| Dynamic split-height update | Conservative packed remap every adaptive step |
| Rigid intersection inside a tall cell | Virtual cubic samples interpolate the two tall endpoints and split each generalized-force contribution with complementary weights |

The WGSL radius-budget field is important. Reducing a column to one seed loses
information when several surfaces in that column have different radii. The GPU
therefore preserves every radius budget from zero through `dmax`; the CPU
`constructAdaptiveOpticalLayer` function is the direct cubic-grid oracle for
the same operation.

## Per-step execution

1. Reconstruct cubic volume and velocity values from the current packed grid.
2. Fit each ground-connected wet column and compute Eq. 1 dilation.
3. Seed all connected and disconnected surface cells.
4. Build the exact variable-radius Manhattan lower envelope.
5. Run five constrained smoothing iterations.
6. Write the next per-column bases and planner diagnostics.
7. Conservatively remap volume and refit tall-cell endpoint velocities.
8. Continue through the shared advection, rigid coupling, pressure, projection,
   and diagnostics stages.

The remap buffer is initialized from the CPU layout before its first use. A
planner finalize overwrites every column, but the initialization makes a
skipped or rejected planner pass conservative instead of silently selecting
base zero.

## Verification contract

The deterministic unit suite covers:

- paper parameter scaling and both Eq. 1 clamps;
- an exactly tall-representable field and a nonlinear-error perturbation;
- exact variable-radius Manhattan diamonds and domain boundaries;
- the complete construction path for ground-connected and airborne surfaces;
- `dmax` activation from a high-error velocity column;
- the invariant that smoothing can only add optical cells;
- conservation and endpoint behavior of the rigid-coupling weights; and
- layout representability, volume accounting, and active-sample counts.

The browser smoke test additionally requires:

- no WGSL compilation, WebGPU validation, or resource-aliasing errors;
- the selected solver identity to remain `adaptive-optical-layer`;
- nonzero active pressure samples and surface-column counts after stepping;
- measured dilation within `[dmin,dmax]`;
- zero non-finite planner or fluid values; and
- bounded short-run VOF drift, with pressure residual, divergence, and CFL
  reported rather than hidden.

On the balanced default dam-break smoke run (61x46x41 equivalent cubic cells,
five 0.004 s GPU steps), the implementation constructed 64,927 active pressure
samples, found 600 ground-connected surface columns, returned dilation 4/4,
reported zero non-finite values, and retained volume to approximately 0.07%.
These values are a smoke-test observation, not universal acceptance limits.

## Comparison and allocation caveats

WebGPU textures are dense. This implementation allocates full-height backing
textures (`fineNy + 2`) so vertical faces and disconnected liquid cannot become
unrepresentable, then short-circuits inactive packed samples in the operators.
The displayed “active” ratio is therefore an algorithmic pressure-sample count,
not a claim that the browser has sparse texture memory. The UI separately
reports physical allocation bytes and the planner timing stage. Performance
claims should compare measured total step, layer construction, pressure time,
VOF drift, divergence, and non-finite counts at matched x/y/z resolution.

## Framework fidelity boundary

The adaptive-grid contribution is implemented independently and completely,
but the host fluid model remains this repository's bounded Eulerian VOF solver.
It is not the paper's EXNBFLIP particle transport or ICCG implementation. The
rigid path preserves the paper's virtual-cell interpolation and complementary
endpoint accumulation within the existing immersed-boundary exchange; it does
not replace the pressure solve with the paper's monolithic variational
fluid-rigid block matrix. Those differences are intentional comparison
controls and must remain visible in exported run identity and interpretation of
paper performance figures.
