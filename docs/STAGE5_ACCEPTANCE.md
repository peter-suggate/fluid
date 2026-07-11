# Stage 5 — Particle CPU Reference

The particle oracle uses three-dimensional Position Based Fluids (PBF). This
replaces the initially proposed DFSPH reference for the browser implementation:
PBF is markedly simpler to inspect in TypeScript, robust at interactive time
steps, and still exposes the density-constraint and neighbour-search behaviour
needed for comparison. It is not labelled DFSPH.

Particles begin on a deterministic lattice. Their mass is `rho0 * spacing^3`.
Density uses the three-dimensional poly6 kernel with compact support `h = 2s`.
Constraint gradients use the spiky-kernel gradient. A deterministic uniform
hash grid accelerates neighbour search; a brute-force oracle remains available.
Each step applies gravity, predicts positions, performs four density-projection
iterations, enforces container boundaries, reconstructs velocity, and applies
XSPH viscosity. Each density iteration limits a position correction to `0.03s`,
reconstructed velocity is limited to the `0.2s/dt` particle CFL bound, and a
`0.995` per-step numerical damping factor suppresses unresolved particle noise;
these are explicit numerical stability limiters rather than physical forces.

## Gates

| ID | Claim | Gate |
|---|---|---|
| P5-01 | spatial hash equals brute force | exact neighbour ID sets |
| P5-02 | lattice density is calibrated | interior mean error `< 2%` |
| P5-03 | closed boundaries hold | persistent penetrations `= 0` |
| P5-04 | volume/mass are conserved | particle-count drift `= 0` |
| P5-05 | static state remains finite | NaN/Inf `= 0`, bounded kinetic energy |
| P5-06 | replay is deterministic | byte-identical state |
| P5-07 | resolution is explicit | effective spacing and particle count reported |

PBF is weakly dissipative and the free surface has a kernel-width density
deficiency. Interior and free-surface density errors are reported separately.
The CPU oracle remains resolution-capped; the verified kernels are the oracle
for the WebGPU port.
