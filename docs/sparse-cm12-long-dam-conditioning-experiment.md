# Sparse CM12 long-dam conditioning experiment

Date: 2026-08-27

## Question

Can coarse accepted cells at a fast free surface reduce the long-dam front ripple,
or is it safer to adapt the gamma-diffusion and sharpening conditioning strength?

## Setup

- Scene: Sparse CM12 long dam, `192 × 96 × 32`, finest `h = 0.0125 m`.
- Large-step lane: CM12 paper step, `dt = 1/30 s`.
- Observation interval: 20 steps (`0.667 s`), spanning ripple onset.
- Front roughness: subcell longitudinal front crossing measured independently in
  the 30 interior transverse slices. Aggregate values below use checkpoints 10–20.
- Every strength lane keeps gamma diffusion enabled and changes only the fraction
  of Algorithm 2's removed-density dose. The original behavior is `1.0`.

## Sharpening-strength sweep

| Sharpening dose | RMS front std. dev. | Mean front span | Peak front span | Mean curvature RMS | Mean gamma-error RMS | Final centre x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 (stage disabled) | 1.168 | 4.074 | 5.713 | 0.831 | 0.590 | 65.73 |
| 0.25 | **0.781** | 3.040 | **4.157** | 0.762 | **0.573** | 65.68 |
| 0.50 | 0.809 | **2.986** | 4.348 | 0.767 | 0.605 | 65.60 |
| 0.75 | 1.182 | 5.080 | 7.816 | 1.094 | 0.575 | 65.38 |
| 1.00 (original) | 0.954 | 3.404 | 5.587 | **0.635** | 0.592 | 65.26 |

The response is not monotonic, so one frame or one scalar is not enough to tune
this. Nevertheless, the 0.25 and 0.50 lanes form a useful basin: relative to the
original dose they reduce trajectory RMS front deviation by 18% and 15%, and peak
span by 26% and 22%, respectively. Final mass drift remains below `0.006%` in both.

The 0.50 lane has the quietest final front (span `1.775` versus `3.492` at full
dose), while 0.25 is better across the checkpoint trajectory and has the lowest
gamma error. A longer run is needed before selecting between them.

## Gamma-diffusion ablation

Disabling gamma diffusion lowers mean front span in this onset window, but it is
not a clean improvement. Mean interface gamma-error RMS rises from `0.592` to
`1.124`, mean curvature rises from `0.635` to `0.705`, and the material centre
falls from `65.26` to `60.67` fine cells. Gamma diffusion should remain enabled;
the sharpening dose is the narrower intervention.

## Forced coarse accepted interface

A whole-domain minimum accepted cell edge of two fine cells produces a nominally
flat front, but only because it suppresses the dam break:

| Metric at 0.667 s | Normal adaptive | Minimum 2-cell edge |
| --- | ---: | ---: |
| Material centre x | 65.26 | 20.84 |
| 99% mass quantile x | 157.5 | 31.5 |
| Interface gamma-error RMS | 0.612 | 4.985 |
| Density above-one mass | 2,739 | 13,808 |
| Mass drift | -0.001% | -0.095% |

The initial dam edge is around x=31.5. Coarsening therefore volume-averages or
pins the shallow advancing sheet rather than giving conditioning a better scale.
This rejects direct accepted-interface coarsening as the next production change.

## Recommendation

Keep the topology rule that protects free surfaces. Retain gamma diffusion, and
continue with a CFL-aware sharpening dose that approaches the paper dose for small
`3 dt / h` and falls toward the measured 0.25–0.50 basin at large values. Before
making that automatic, run the same sweep through wall impact and on settled-dam,
splash, and symmetric-expansion regressions.

Raw data:

- `artifacts/sparse-cm12-long-dam-sharpening-strength-20.json`
- `artifacts/sparse-cm12-long-dam-no-gamma-current-20.json`
- `artifacts/sparse-cm12-long-dam-min-cell-2-current-20.json`
