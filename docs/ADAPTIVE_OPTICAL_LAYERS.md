# Adaptive optical layers (Narita--Kanai 2026)

This repository implements the motion-adaptive optical layer from
[Narita and Kanai 2026](https://graphics.c.u-tokyo.ac.jp/projects/Adaptive-Optical-Layers/)
as a vertical-segmentation strategy inside the existing quadtree tall-cell
solver. It does not introduce a separate advection or pressure solver.

## Algorithm

For every finest x/z column, the CPU oracle and WGSL construction path perform
the same operations:

1. Find the lowest connected liquid run and its main surface.
2. Least-squares fit horizontal velocity as a linear function of y and use the
   column-average vertical velocity.
3. Sum the L1 reconstruction error and evaluate paper Eq. (1),
   `d = clamp(alpha * E * dx, dMin, dMax)`.
4. Apply the variable-radius Manhattan dilation with separable min-plus x and
   z transforms.
5. Apply five constrained 9x9 moving-average passes. The smoothed tall-cell
   boundary may only move downward, so smoothing can add optical cells but
   never remove cells selected by the motion estimate.
6. Conservatively reduce the finest-column field into each quadtree leaf: an
   optical cube requested by any covered fine column is retained.

Paper defaults are derived from vertical resolution:

- `alpha = 0.5`
- `dMin = max(4, Ny / 64)`
- `dMax = Ny / 8`
- `Doffset = max(4, Ny / 32)`
- `dAir = Ny / 16`

Interfaces close to the main surface consume its adaptive boundary. More
distant airborne interfaces use `dAir`. The existing two-cell air-side band is
also retained to protect the pipelined level-set topology.

The fixed quarter-of-connected-depth segmentation remains available as the
control. The sparse face graph, CSR assembly, pressure solve, MLS mapping,
projection, and existing virtual-subface/rank-6 rigid coupling are shared by
both modes.

## Controls and telemetry

The **Optical layer** method control selects:

- `Motion-adaptive (2026)` (product preset)
- `Fixed quarter-depth` (A/B baseline)

**Optical motion response** exposes `alpha`. Diagnostics report the active
mode, alpha, derived minimum/maximum cell depths, pressure samples, liquid
DOFs, and variational faces.

The Node smoke harness accepts:

```sh
FLUID_OPTICAL_LAYER_MODE=adaptive-motion
FLUID_OPTICAL_ALPHA=0.5
```

Use `FLUID_OPTICAL_LAYER_MODE=fixed` for the baseline.

## First measurements

Local Metal/WebGPU smoke measurements on 2026-07-18 validate the expected
workload split.

The shallow 61x46x41 dam-break has `dMin=4` and `dMax=6`. Many fixed-baseline
columns are shallow enough that one quarter of their local depth is less than
four cells, so adaptive mode retains more pressure cells there. This is an
expected consequence of the paper's stability floor and makes the case useful
as a quality/regression check rather than a speed benchmark.

For the calm 61x1021x41 deep-water scene:

| Metric | Fixed quarter-depth | Adaptive | Change |
| --- | ---: | ---: | ---: |
| Liquid pressure DOFs | 189,819 | 16,506 | -91.3% |
| Total pressure samples | 194,404 | 21,091 | -89.2% |
| Variational faces | 570,857 | 57,911 | -89.9% |
| PCG iterations to relative 1e-4 | 849 | 729 | -14.1% |
| Measured pressure phases | 933.5 ms | 207.8 ms | 4.5x faster |

The fixed resident pack attempted a 257 MB matrix binding and exceeded the
portable 128 MB WebGPU limit. The adaptive case completed without validation
errors. A `dMax/dMin` condition-number guard keeps the PCG hard budget from
shrinking in direct proportion to DOF count; without it, the first deep solve
stopped at 515 iterations and did not reach tolerance.

These are focused implementation measurements, not a broad performance claim.
Dense backing-field advection and surface transport are unchanged, initial
CPU topology construction remains expensive, and GPU dispatch latency can
dominate small systems.

## Verification

- CPU motion-error, dilation, smoothing, and calm/deforming-layer tests.
- Exact CPU/GPU adaptive-column conformance test.
- Exact CPU/GPU adaptive sparse-segmentation conformance test.
- Portable WGSL pipeline validation.
- Shallow dam-break stability smoke.
- Deep-water convergence, compression, and binding-limit smoke.

