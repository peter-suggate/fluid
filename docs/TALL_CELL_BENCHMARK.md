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
  pre/post sweeps, one full cycle, two V-cycles, and depth-dependent converged
  coarsest solves.

## Default moving-dam result

The vertical reservoir face does not fit both halos in a 24-layer band. Per
the paper, the air constraint wins rather than expanding every column. The
current balanced layout stores `61 x 26 x 41` samples for a
`61 x 46 x 41` cubic-equivalent grid (56.5% allocated samples), with about 96%
of columns using a tall cell.

A current local Metal/Dawn smoke at `0.05 s` measured:

| Metric | Tall | Uniform |
| --- | ---: | ---: |
| Runtime, 18 steps | 349 ms | 101 ms |
| Last-step advection timestamp | 3.87 ms | 0.42 ms |
| Last-step pressure timestamp | 11.27 ms | 1.00 ms |
| Pressure relative residual | 0.011 | not reported by legacy solver |
| Maximum adjacent split delta | 4 | n/a |

The dam is therefore not a speedup at this small resolution. The Tall path is
doing bounded MacCormack transport, velocity extrapolation, remeshing, signed
distance reconstruction, and a full ghost-fluid multigrid cycle; the Uniform
reference is a much leaner legacy VOF/Jacobi path. Storage compression alone
does not erase those fixed costs.

Historical controlled local samples produced the following before the current
stability correction; they must not be treated as current performance or
parity numbers:

| Metric | Uniform cubic at 2.06 s | Tall solver at 2.24 s |
| --- | ---: | ---: |
| Raw VOF drift | approximately 0.00% | -0.07% |
| Peak wet speed | 2.07 m/s | 3.36 m/s |
| GPU front | 0.60 m | 0.60 m |
| NaN / infinity observed | none | none |

The earlier implementation lost about 95% of its VOF by 20 s and launched the
cork to kilometre-scale positions. The corrected run remains finite and its
volume is close to the uniform reference. Peak-speed parity was not claimed
exact. The tall path now uses a physical two-cell pressure-gradient span to
prevent the paper's printed collocated operator from reflecting hydrostatic
impulses. See [`TALL_CELL_STABILITY.md`](TALL_CELL_STABILITY.md) for current
stability measurements.

## Deep-water storage result

A current 1.2 m × 20 m × 0.8 m, 80%-full tank uses dimensions
61 × 1021 × 41. It stored 65,026 tall samples instead of 2,553,521 uniform
samples, a 39.3× reduction, with zero observed VOF drift in the smoke sample.
The latest local run measured 386 ms Tall versus 423 ms Uniform; last-step
pressure timestamps were 24.71 ms versus 41.62 ms. This is a modest measured
speedup, not a claim that runtime scales directly with the sample-count ratio.

## Reproduction

1. Open the application and select WebGPU.
2. Reset and run **Tall cells** and **Uniform** to the same simulation time.
3. Compare stored dimensions, raw VOF drift, maximum speed, maximum
   post-projection divergence, body state, and the rendered free surface.
4. Click **Load deep-water A/B scene** to exercise actual tall-cell compression.
