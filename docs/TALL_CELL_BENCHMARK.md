# Tall-Cell / Uniform Browser Comparison

## Reference

This comparison follows Chentanez and Müller, “Real-time Eulerian Water
Simulation Using a Restricted Tall Cell Grid.”

Paper: <https://matthias-research.github.io/pages/publications/tallCells.pdf>

## Comparison contract

- Local Chrome/WebGPU build at balanced quality.
- Identical x/z and cubic-equivalent y spacing between methods.
- Surface tension disabled on the tall path because it is outside the paper's
  core solver.
- Raw VOF is reported without global mass rescaling.
- Tall pressure uses solid-aware ghost-fluid multigrid, two damped RBGS
  pre/post sweeps, one full cycle, and one V-cycle.

## Default moving-dam result

The vertical reservoir face does not fit a 24-layer band. The corrected layout
therefore selects the uniform-grid limit: 61 × 48 × 41 stored samples for a
61 × 46 × 41 cubic-equivalent grid, with the two endpoint layers inactive.
This is a representability requirement, not a performance regression.

Controlled local samples produced:

| Metric | Uniform cubic at 2.06 s | Tall solver at 2.24 s |
| --- | ---: | ---: |
| Raw VOF drift | approximately 0.00% | -0.07% |
| Peak wet speed | 2.07 m/s | 3.36 m/s |
| GPU front | 0.60 m | 0.60 m |
| NaN / infinity observed | none | none |

The earlier implementation lost about 95% of its VOF by 20 s and launched the
cork to kilometre-scale positions. The corrected run remains finite and its
volume is close to the uniform reference. Peak-speed parity is improved but is
not claimed exact: the tall path intentionally uses the paper's collocated,
non-idempotent projection, whereas the retained uniform comparator is the
older composed weighted-Jacobi solver.

## Deep-water storage result

A prior 1.2 m × 20 m × 0.8 m, 80%-full tank used dimensions
61 × 1021 × 41. It stored 65,026 tall samples instead of 2,553,521 uniform
samples, a 39.3× reduction, with no observed VOF drift over the short
hydrostatic sample. Its old queue timings are not retained as a current
performance claim because the pressure smoother and cycle budget changed.

## Reproduction

1. Open the application and select WebGPU.
2. Reset and run **Tall cells** and **Uniform** to the same simulation time.
3. Compare stored dimensions, raw VOF drift, maximum speed, maximum
   post-projection divergence, body state, and the rendered free surface.
4. Click **Load deep-water A/B scene** to exercise actual tall-cell compression.
