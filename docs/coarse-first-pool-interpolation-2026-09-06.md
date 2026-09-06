# Coarse-first pool: interpolation and square wave artifacts

The residual grid is reproducible. There is a concrete slope-reconstruction bug,
now corrected, and a separate limitation in how coarse column averages become
surface heights. Correcting the slope alone does **not** remove the square ridges.
The evidence supports improving reconstruction and resolution selection, rather
than accepting all of the grid imprint as unavoidable.

## Reproduction and controls

Use `coarse-first-pool-impact`, balanced quality, coarse-first selection, scene
time step, no authored refinement regions. The latest browser capture was at
**0.9167 s (55 steps)**; the earlier 0.5 s report was before the developed impact.
The native Dawn probe captures both, plus 0.6667 s. Every advance checks the actual
encoded step count so deferred topology preparation cannot silently drop steps.

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal POOL_OUTPUT=artifacts/pool-interpolation \
node --import tsx tools/probe-pool-interpolation-dawn.ts
```

Unload Fluid Lab browser tabs before running Dawn. The probe acquires the
repository WebGPU lease and saves accepted density, open fraction, velocity,
pressure, publication payloads, topology activity and statistics. Its dense phi
convenience view decodes ordinary pages around this pool's free surface; macro
interiors remain NaN. Full encoded macro payloads are retained separately.

For the finer simulation control, add `POOL_MAX_CELL=2` and use
`node --max-old-space-size=12288 --import tsx ...`. This authors a region over the
whole tank limiting cells to 0.10 m, versus the default outer surface's 0.40 m.
It changes the simulation as well as presentation; it is a resolution experiment,
not a rendering-only comparison or a proposed default.

Captures in this workspace:

- `artifacts/pool-interpolation-before`: commit `91fd9d26`, original slope.
- `artifacts/pool-interpolation-fixed`: default grid with corrected slope.
- `artifacts/pool-interpolation-max2`: finer grid, **original slope**.
- `artifacts/pool-interpolation-investigation`: comparison plot, metrics and
  Python analysis scripts (NumPy, SciPy and Matplotlib).

![Impact comparison](../artifacts/pool-interpolation-investigation/impact-comparison.png)

## Defect fixed: a linear ramp acquired artificial cell-face jumps

`presentationLimitedSlope` was computing:

```text
sign * min(0.75 * abs(forward - back), 4 * min(abs(left), abs(right)))
```

For cell averages `0.4, 0.5, 0.6`, that produces `0.15`, although the correct slope
per cell is `0.10`. Adjacent linear patches predict `0.575` and `0.525` at their
shared face. The resulting jump is artificial even for a perfectly affine field.
Symmetric child offsets still preserve the parent mean, which explains why a
mass-conservation check alone did not detect the problem.

The coefficients are now `0.5` and `2`. This reproduces affine averages and bounds
each axial face reconstruction between its neighboring means. The helper is
shared by cached presentation, direct reconstruction, representability checks,
and density prolongation when refining B4 to B8. The latter means the defect can
seed the accepted density field, not just affect appearance.

`tests/sparse-cm12-presentation-linear-dawn.test.ts` executes the production WGSL
functions using exact affine control-volume averages. It checks the same physical
positions at widths 2h, 4h and 8h, cached/direct agreement, agreement across 2:1
source widths, and each parent's mean. The initial regression failed on the old
coefficient with a maximum density error of 0.0035 and passed with the correction.
The test is included in `npm run test:dawn:sparse-cm12:coarse-first`.

## Remaining reconstruction issue: coarse heights are repeated as point values

`presentationIntegratedColumnHeight` correctly integrates the accepted vertical
column. However, the ordinary pool path in `preparePresentationColumnHeights`
uses that average directly at each finest-grid x/z sample. Samples within the
same coarse column therefore share the same height. The fine contour lattice
connects those plateaus over a narrow strip at each coarse-cell edge.

The fine lattice cannot recover a smooth wave from these repeated heights by
itself. Small height differences become narrow, axis-aligned slopes, which
reflective water makes conspicuous. This path takes precedence over the corrected
density reconstruction in monotone surface columns, so the limiter fix cannot
remove its terraces.

There is already a horizontal bilinear path for a specialized, authored B1 floor
film. Its eligibility does not cover this adaptive, deep pool. Extending a
same-width bilinear stencil naively would also change stencils at 2:1 boundaries;
any replacement must produce the same height and normal on both sides.

At 0.9167 s, in the outer annulus 1.2–2.7 m from the impact center:

| Capture | RMS discrete height Laplacian (1/m) | 99th percentile slope (m/m) |
| --- | ---: | ---: |
| Original slope, default grid | 0.3519 | 0.0903 |
| Corrected slope, default grid | 0.3225 | 0.0912 |
| Original slope, cells at most 0.10 m | 0.2055 | 0.0845 |

These are roughness diagnostics, not errors against an exact impact solution.
The corrected slope gives a modest change in the real capture; the finer-grid
control gives a much smoother and rounder outer wave. Remaining axis structure
near the impact is still visible in that control. A rectangular tank also breaks
continuous rotational symmetry, so exact circularity is not a universal physical
acceptance criterion.

No additional liquid-to-air crossings were found in the sampled free-surface
band (y = 1.025–2.275 m, radial distance 0.85–2.5 m) at any of the three checkpoints.
That supports a height/slope explanation for these ridges, rather than additional
internal sheets in this band. It does not certify the entire mesh or submerged
volume.

A CPU evaluation of the emitter's 3×3×3 Gaussian normal formula on the original
published field differed from height-derived normals by 0.245 degrees RMS in the
outer annulus. Filtering changes highlights, but the geometric terraces exist
before that filter. This comparison evaluates the shader formula; it is not a
readback of GPU vertex normals.

## Can interpolation recover a physically shaped wave?

An offline control places a known circular Gaussian ring on the captured adaptive
column topology. The ring has amplitude 18 mm, radius 1.65 m and width 0.42 m.
Input values are cell-area averages computed by 8×8 Gaussian quadrature, not
samples at cell centers.

A continuous quadratic moving-least-squares reconstruction fits those finite-volume
averages. The polynomial basis includes each cell's second moment (`width²/12`),
and each donor owns a compact radial support of radius 2.5 times its width. This
avoids switching the support radius abruptly when the query crosses a cell-size
boundary. The fit has no knowledge of the impact center or circular symmetry.

On the known circular wave, RMS height error in the same outer annulus drops from
**1.134 mm to 0.309 mm**. Thus much of the square appearance from repeated coarse
averages can be removed by reconstruction without inventing a circular wave.

![Reconstruction control](../artifacts/pool-interpolation-before/reconstruction-experiment.png)

Applied to the real impact data, the same fit reduces the discrete Laplacian RMS
from 0.3519 to 0.2323 1/m, but visible axis structure remains. This is consistent
with both representation error and insufficient physical resolution. It is not
proof that all residual anisotropy originates in the solver: the adaptive stencil
and tank boundaries can also contribute.

The fit is an experiment, not a production fix. It has 64 rank-deficient fits at
domain corners and changes the global integrated visual height by -0.0120 m³,
including the complex impact core. Fitting cell averages locally does **not** by
itself guarantee exact global volume preservation.

## Implementation direction

The next surface change should reconstruct a common continuous height field from
accepted cell averages, with conservative constraints, bounded overshoot, and
validity checks for overhangs, cavities and solid intersections. Its support must
be independent of which page emits a vertex. Constant, tilted-plane and smooth
radial fixtures should cross mixed 2:1 boundaries and check both value and normal
continuity, volume error, and closed mesh topology. Re-runging an unchanged field
should not introduce a visible jump.

Resolution selection should then be evaluated against reconstructed surface error
for small waves. Current energy/curvature thresholds can leave gentle but visibly
changing water at 8h. Refining the complete tank demonstrates the opportunity but
is too blunt a production remedy. Adding blur or fitting circles would not resolve
these numerical contracts.

## Validation

- Production affine-reconstruction GPU regression: old coefficient fails, corrected
  coefficient passes; broadened fixture also checks adjacent parents and 2:1 widths.
- CPU density-column, surface-proof and coarse-first tests: 15/15 passed.
- Adaptive mesh Dawn tests: 2/2 passed. Closed synthetic mixed-resolution surfaces
  at ratios 2 and 4 retain zero open edges; pool checks through 0.5 s retain full
  upward surface coverage and zero internal surface-band triangles.
- Coarse-first Dawn gate: 5/5 passed, including the impact, settling, macro re-rung
  and affine reconstruction fixtures.
- Canonical Dawn gate: all 14 correctness lanes and mini32 performance passed.
  Mini64 remains above its existing 50 ms ceiling: 64.8806 ms in the final run,
  versus 64.3564 ms before this change. The suite therefore exits with a failure;
  no lane or ceiling was weakened. Mini32 measured 29.8189 ms against 40 ms.
- Type checking remains blocked by existing errors in Losasso audits, ceiling-slab
  and wireframe tests, and legacy probes; no diagnostic names the changed files.
