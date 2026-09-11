# Smallest high-speed geometric transport reference

The `geometric-uniform-translation` scene uses a 16×8×8 finest-cell lattice
(two 8³ bricks), with no coarse fluid cells. A 0.2 m liquid plug fills the
0.4×0.4 m free-slip channel cross-section and translates at 6 m/s. Gravity,
viscosity and surface tension are zero. The initial x interval is [0.1, 0.3] m,
measured from the left vessel boundary. The simulation ends at 0.05 s, before
wall contact, with the exact interval [0.4, 0.6] m.

This is an exact translating free-surface Euler solution: the material velocity
is constant, its divergence and advective acceleration vanish, and constant
atmospheric pressure satisfies the free surfaces. The side-wall normal velocity
is zero. No shallow-water approximation or empirical dam-front target is needed.

At every accepted time t, the reference is:

- x support: [0.1 + 6t, 0.3 + 6t] m;
- liquid volume: 0.032 m³;
- centroid: (0.2 + 6t, 0.2, 0.2) m in vessel-local coordinates;
- velocity: (6, 0, 0) m/s;
- specific kinetic energy: 18 m²/s²;
- kinetic energy: 576 J at 1000 kg/m³.

The default 1/60 s outer step moves the plug exactly two finest cells. Reference
volume fractions are integrated over each cell's intersection with the moving
box; comparison does not depend on a rendered contour. The probe also compares
kinetic energy, velocity, centroid, bounds and each y/z column's volume. Stage
captures hold the incoming liquid-volume weights fixed while comparing source
faces, prepared faces, body forces and pressure projection.

Run `npm run probe:dawn:adaptive-volume:translation` with the browser unloaded.
The optional detached-box variant introduces free surfaces along y to expose
corner reconstruction effects only after the one-dimensional case is understood.

## Measured isolation results

On the retained staggered-face production solver:

| Case | Final relative volume-field L1 | Velocity error | Kinetic-energy change |
| --- | ---: | ---: | ---: |
| Plug, 6 m/s, outer CFL 2 | 5.59e−16 | 0 | roundoff |
| Plug, 5 m/s, outer CFL 5/3 | 1.19e−7 | 2.58e−14 m/s | roundoff |
| Detached box, 6 m/s, outer CFL 2 | 0.03125 | 1.07e−14 m/s | roundoff |

All run to 0.05 s. Both plug cases pass the predeclared analytic criteria.
The detached box fails the shape criterion from step one: maximum fraction
error 0.0625, relative volume-field L1 0.03125. This error remains constant
at the three outer observations. Volume, centroid, per-column volume,
uniform velocity and kinetic energy remain correct; low-flux receiver limiting
is inactive. This localizes its discrepancy to geometric shape transport,
not a loss of momentum or energy. A corner reconstruction explanation is
consistent with these observations but has not yet been established through
per-microstep plane/flux captures.

The default fluid density is 998.2 kg/m³, giving initial plug energy 574.9632 J.
The 576 J value above is the same analytic solution evaluated at 1000 kg/m³.
No production physics or existing regression tolerance was changed for these
runs. Receipts in `artifacts/analytic-motion/`:
`plug-cfl2-before.json`, `plug-fractional-before.json`,
`detached-cfl2-before.json`.

A detached control at outer CFL 0.5 exposes the corner error's cycle: after
each half-cell displacement its volume field is correct to roundoff; after
each whole-cell displacement the same 3.125% L1 discrepancy returns. Neither
speed nor energy drops over its 12 steps. This is reversible shape oscillation
at the sampled phases, not accumulating numerical energy dissipation. Receipt:
`detached-cfl05-before.json`.

The final probe also validates the initial velocity and energy injection and
compares both the accepted publication clock and accumulated GPU-executed
physical time. The default rerun passes both analytic oracles; accumulated
clock difference at 0.05 s is 2.61e−9 s, within its f32 bound. Final receipt:
`artifacts/analytic-motion/plug-cfl2-validated.json`.
