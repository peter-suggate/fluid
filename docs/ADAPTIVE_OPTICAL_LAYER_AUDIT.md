# Adaptive Optical-Layer Audit Loop

This is the evidence log for the repeated process:

1. measure current realism and stability;
2. compare the implementation with the paper;
3. correct the highest-impact gap without changing the comparison baselines;
4. repeat the same measurements.

All browser measurements below use the balanced 61x46x41-equivalent default
dam-break, 0.004 s fixed steps, and the diagnostic sample at 21 encoded steps.
Tabs are run serially because concurrent WebGPU tabs can delay adapter creation
and invalidate timing comparisons.

## Iteration 1: planner execution and pressure consistency

### Initial evidence

| Mode | Active pressure samples | VOF drift | Divergence pre -> post | Relative pressure residual | CFL | Non-finite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 115,046 | 0.00% | not instrumented | not instrumented | not instrumented | not instrumented |
| Fixed Tall Cells | effectively uniform in this shallow scene | -0.23% | 11.0 -> 10.7 1/s | 0.0134 | 0.347 | 0 |
| Adaptive | 71,729 | -0.19% | 8.35 -> 7.68 1/s | 0.867 | 0.430 | 0 |

The adaptive layer conserved mass reasonably and stayed finite, but its
pressure system was far from converged. A separate first-step failure was
localized with a six-bit GPU pass trace: the planner completed through
Manhattan dilation (`15/63`) but did not reach smoothing or finalize.

### Corrections

- Removed the raw-to-smoothing texture copy and made the first smoothing pass
  read the immutable raw field directly. The planner now reports `63/63`.
- Derived planner dimensions from bound textures and made paper parameters a
  homogeneous 32-byte float uniform.
- Added live GPU self-checks for `dmin`, `dmax`, `Ny`, maximum base, and planner
  stage completion.
- Made multigrid refinement cycles explicit. Eight cycles reduced the adaptive
  residual but did not by itself establish operator consistency.
- Restored the positive-face velocity convention for Adaptive only. Its
  divergence is now the backward difference of positive-face velocity and its
  projection uses the matching forward pressure gradient, consistent with the
  nearest-neighbor matrix. Fixed Tall Cells retains its previous centered
  operator for an uncontaminated baseline.

### Repeated evidence

With the complete planner, positive-face adjoint pair, and eight V-cycles:

| Mode | Active pressure samples | VOF drift | Divergence pre -> post | Relative pressure residual | CFL | Non-finite | Stability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Adaptive | 71,763 (62%) | -0.21% | 43.4 -> 29.4 1/s | 0.0609 | 0.392 | 0 | clear |

The absolute divergence values are not directly comparable with the initial
centered diagnostic because the corrected discrete operator measures
positive-face flux. The meaningful within-run facts are a 32% reduction,
residual below the current 0.1 gate, CFL below one, and zero non-finite values.
The result still does not meet the paper's `10^-4` solve tolerance.

## Current paper-gap ranking

| Priority | Paper requirement | Current state | Evidence or next gate |
| --- | --- | --- | --- |
| P0 | Equation 2 uses a consistent `-G^T V A F G` operator | Optical-cell face convention corrected; tall endpoints still use the framework's interpolated approximation | derive and test the complete tall endpoint/area/volume stencil; target residual `<=10^-4` and monotonically reduced divergence |
| P0 | Residual-controlled ICCG solve | Fixed-count RBGS multigrid, eight adaptive V-cycles | implement GPU convergence control or PCG/MG-preconditioned CG; report iteration count and true norm |
| P0 | Monolithic Equation 3 fluid-rigid solve | Immersed penalty exchange occurs before pressure | implement `J^T M^-1 J` or keep results explicitly labelled non-monolithic; test impulse and angular impulse closure |
| P1 | Equations 6-11 use face area fractions and metric-correct moments | Virtual cells use eight solid samples and endpoint weights, not the full pressure coupling matrix | compare per-cell six-vector against a CPU `J` oracle |
| P1 | Error is evaluated on velocity faces | Planner currently reconstructs the framework's stored velocity vector at cubic cell locations | define component face positions and compare GPU `E` with a face-sampled CPU oracle |
| P1 | Algorithm 1 ordering | Layer build/remap precedes advection; rigid penalty precedes pressure | record an encoded pass trace and reconcile grid conversion and pressure-to-rigid update ordering |
| P1 | Uniform comparison has the same stability evidence | Uniform reports speed and volume only | add divergence, residual, CFL, pressure, location, and non-finite reductions |
| P2 | Reported sparse cell savings | Full-height dense WebGPU backing with active-sample short-circuiting | distinguish active unknowns from allocated bytes and benchmark both |
| P2 | EXNBFLIP surface detail | Repository intentionally uses bounded VOF | compare only grid/projection behavior; do not claim particle-level visual equivalence |

## Realism gates for the next iteration

- hydrostatic tank: no growing kinetic energy or surface drift;
- dam break: monotone front advance, bounded VOF drift, CFL below one;
- projection: residual `<=10^-4` and divergence never amplified;
- adaptive-vs-`dmax`: setting every dilation to `dmax` converges toward the
  fixed-thickness tall result at matched resolution;
- rigid impact: finite force/torque, bounded penetration, and closed impulse
  accounting;
- all three GPU modes expose the same diagnostic fields.
