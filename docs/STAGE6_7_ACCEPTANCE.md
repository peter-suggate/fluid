# Stages 6–7 — Fluid/Rigid Coupling

The browser reference uses deterministic primitive-volume quadrature. Samples
inside each rigid primitive query fluid occupancy, velocity, and pressure. The
wet fraction gives displaced volume; Archimedes buoyancy is
`F_b = -rho_f V_displaced g`. Quadratic drag is
`F_d = -0.5 rho_f C_d A_wet |v_rel| v_rel` with the explicitly approximate
coefficient `C_d = 0.9`. The centre of the wet samples supplies the buoyancy
moment arm.

For two-way coupling, every hydrodynamic body impulse `J = F_h dt` is paired
with a fluid impulse `-J`, distributed with compact non-negative weights whose
sum is exactly one. Particle momentum closure is exact to binary64 reduction
roundoff. The Eulerian face-field reaction is a conservative bring-up
approximation away from solid walls; cut-cell traction remains future work.

## Gates

| ID | Claim | Gate |
|---|---|---|
| C6-01 | fully immersed neutral body | net vertical force `< 1e-10 N` |
| C6-02 | dense immersed body sinks | initial vertical acceleration `< 0` |
| C6-03 | half-wet box displacement | submerged fraction error `< 5%` |
| C7-01 | body/fluid impulse closes | relative error `< 0.1%` |
| C7-02 | multiple bodies receive independent loads | all finite, correctly keyed |

The quadrature and empirical drag law are exposed approximations. They are not
substitutes for resolved cut-cell pressure/viscous traction at high resolution.
