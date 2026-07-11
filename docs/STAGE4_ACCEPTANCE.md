# Stage 4 — Eulerian CPU Reference and Dam Break

The first fluid implementation is a three-dimensional staggered MAC grid using
JavaScript `Float64Array` state. It is intentionally resolution-capped for an
interactive browser CPU reference; the requested and effective resolutions are
both reported.

## Numerical choices

- face-centred velocity and cell-centred pressure/occupancy;
- RK2 backtraced semi-Lagrangian velocity advection;
- explicit physical-viscosity stencil (negligible for water at this scale);
- matrix-free Jacobi-preconditioned conjugate-gradient pressure projection;
- eight deterministic marker particles per initially occupied cell;
- marker advection and cell reconstruction for the free surface;
- free-slip impermeable container walls and zero-gauge-pressure air cells;
- dam-break initial column at the left wall by default.

This is dissipative, first-order in the interface reconstruction, and not
strictly volume-conservative on the reconstructed grid. Marker volume and raw
occupied-grid volume are therefore reported separately. No global volume
correction is applied.

## Stage 4 gates

| ID | Claim | Initial CPU gate |
|---|---|---|
| E4-01 | projection reduces manufactured divergence | post L2 `< max(1e-9, 1e-5 pre)` |
| E4-02 | PCG converges | relative residual `<=1e-8`, no failed solve |
| E4-03 | closed velocity boundaries hold | boundary normal speed `<1e-12 m/s` |
| E4-04 | marker volume is conserved | drift `<1e-12` without escape |
| E4-05 | static water has no spontaneous energy growth | finite state; kinetic energy bounded |
| E4-06 | hydrostatic pressure follows `rho g depth` | L2 relative error `<10%` at coarse bring-up resolution |
| E4-07 | dam front advances | front displacement `>0.05 m` over benchmark interval |
| E4-08 | refinement is reported | three effective resolutions; error/trend retained |
| E4-09 | time-step limits are explicit | advective, viscous, user maximum, and selected limit reported |
| E4-10 | deterministic replay | state and diagnostics byte-identical in the same build |

Failure of projection or finite-state gates blocks browser publication. The
dam-break visual is solver occupancy, never a decorative wave plane.
