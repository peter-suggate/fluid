# Figure 9: frame-105 surface loss is preceded by pressure divergence

Investigated 2026-09-20. This report describes the pre-fix investigation; see
[implementation and validation](implementation.md) for the subsequent fix. Raw receipts,
frozen-system experiments and an optional diagnostic patch are alongside this report.

The authored `mass-conserving-figure-9-dam-break` scene reproduces with the shared
lab seed, shared defaults plus `activeRegion: off`, 128 × 128 cells, h = 0.05 m,
and dt = 1/30 s. Native and the existing scalar Wasm artifact produce exactly
equal receipts through frame 120. This is not a browser publication artifact.

| Completed frame | Maximum speed (m/s) | Pressure residual (dt/rho scaled) | Phi area (cell areas) | Conserved V |
|---|---:|---:|---:|---:|
| 103 | 20.6155 | 1.01031 | 3403.7894 | 3839.99515 |
| 104 | 183051.7188 | 870531.9375 | 3560.5211 | 3839.99508 |
| 105 | 0 | 0 | 0 | 3839.99508 |

The damaging pressure solve is frame 104. The next frame advects with its huge
velocities, loses the entire negative-phi region, and then has no liquid pressure
rows. Velocity becomes zero; conservative volume survives in the donor fallback.
The zero residual at frame 105 is therefore an empty-system success, not recovery.
All values remain finite, so the existing finite-state check misses this failure.

The root cause isolated by the frozen-system experiments is **discarding pressure
inequality constraints on deeper multigrid levels while continuing to restrict the
ordinary equation residual into those levels**.

At walls, pressure is bounded below by zero. When a row reaches its lower bound,
a negative `b - A p` can be valid for the constrained problem: satisfying the
unconstrained equation would require inadmissible negative pressure. The convergence
norm recognizes this using the pressure-to-bound gap. `Level::residual`, however,
returns the ordinary residual; that requires compatible treatment of the bounds
on the correction problem. `restrict_min` instead resets the lower bound to the
FREE sentinel after three transfers. The 128² hierarchy has seven levels, so
levels 4, 5 and 6 have no inherited bounds.

The trace at level 3 captures six active-bound wall rows in the first nested
V-cycle; the worst is halo coordinate (0,2), p = p_min = 0, residual = -104788.72.
Across subsequent nested cycles the worst active-bound residual grows to
-311140.5, -1059130 and -6127225.5. Coarse solves pursue negative corrections that
the constrained finer grids cannot accept. The cycle feeds this mismatch back
into itself. In the two-Full-Cycle solve, pressure magnitude reaches 1.82e10 on
level 4 and 1.92e9 on the finest level. The coarsest solver itself reports success;
its convergence does not establish convergence of the overall constrained problem.

Relevant production code:

- Rust: `pressure.rs` lines 160 (ordinary residual), 197 (constrained norm),
  336 (bound restriction/cutoff), 429 (V-cycle residual transfer).
- Shared cutoff: `lib/methods/uniform/pressure-policy.ts`, constraint levels = 3.
- GPU: `webgpu-uniform-pressure-multigrid.ts` lines 633 and 652 explicitly clear
  minimum pressure on deeper levels, in V-cycles and Full-Cycles respectively.
- `world.rs` line 394 uses the returned pressure even when convergence failed.

This identifies a shared algorithmic vulnerability in the 2D and 3D code. It does
not establish the failure frame of an ordinary 3D run; no Dawn or browser GPU
session was used in this investigation.

Every frozen-system arm starts from the identical pre-projection frame-104 RHS,
phi, topology and finest bounds, and disables early stopping to compare cycles:

| Arm | Residual after one Full-Cycle | After two | After three |
|---|---:|---:|---:|
| Baseline | 678.739 | 870531.94 | 1.16721e9 |
| Preserve bounds through every level | 2.54906 | 0.359018 | 0.056816 |
| Stop hierarchy at level 3, solve it directly | 2.65214 | 0.367817 | 0.056659 |
| Preserve mixed-cell air sign at every level only | 137.140 | 24035.92 | 4.23801e6 |
| Zero air samples during prolongation only | 567.046 | 739345.06 | 9.66224e8 |

V-cycles alone also diverge: residual 31.14 after one, 2.30e7 after seven.
Increasing smoothing to 1000 sweeps produces residual 0.00177 with the original
hierarchy. This is an expensive diagnostic control, not a suggested production
setting. The problem is not repaired by blindly spending more cycles. Moving
the constraint cutoff to four or five transfers also fails to give satisfactory
convergence; retaining bounds for all six transfers is the successful arm.

Whole-scene controls, run to six seconds:

| Arm | First speed above 1000 m/s | Peak speed | Final phi area |
|---|---:|---:|---:|
| Original, dt 1/30 (baseline ran to 4 s) | frame 104, 3.4667 s | 183052 | 0 |
| Original, dt 1/60 | frame 198, 3.3000 s | 61126 | 0 |
| Original, dt 1/120 | frame 403, 3.3583 s | 13539 | 0 |
| All-level bounds, dt 1/30 | none through frame 180 | 43.1819 | 4049.2083 |

The successful arm ends at 7.3457 m/s, residual 0.6361, conserved V = 3839.9925.
Its phi area still exceeds V by about 5.45%; avoiding the catastrophe is not proof
of correct long-term surface/volume agreement or energy behavior.

Recommended order of work:

1. Carry shifted pressure bounds through the complete hierarchy in both backends.
   This is the smallest directly supported candidate. Validate the frozen frame
   first, then longer dam runs, hydrostatics, wall separation, moving solids,
   different hierarchy depths and dimensional parity. Measure convergence and
   cost; do not replace the hardcoded cutoff with a larger arbitrary constant.
2. Add pressure-step rejection and recovery. Retain a previous acceptable iterate;
   detect worsening constrained residual and reject a divergent cycle. Fall back
   to safeguarded projected smoothing or a constraint-aware solver. A failed solve
   must not publish extreme projected velocities to transport. Finite-field checks
   and coarsest convergence alone are insufficient. Merely tightening tolerance
   or increasing the lagged budget makes this particular unstable iteration worse.
3. For a more principled solver, make restriction/prolongation and coarse correction
   aware of the active inequality set (for example, truncated or monotone constrained
   multigrid), with safeguarded correction damping. Simply masking a negative
   residual is not established as a complete fix: constraints, transfer and the
   correction operator must remain compatible. This approach is not implemented
   or measured here.
4. Separately investigate the earlier local speed spikes and V/phi mismatch.
   CFL substeps can improve transport quality, but the measured dt controls show
   they do not resolve this pressure failure. Neither surface reseeding nor mass
   balancing should be used to conceal the divergent solve.

Reproduction uses the existing native example. Generate the exact shared lab seed:

```sh
node --import tsx --input-type=module -e 'import {uniformLabSeed,UNIFORM_LAB_VALUES} from "./lib/physics-wasm/uniform-controller.ts"; import {createMassConservingFigure9DamBreak} from "./lib/core/paper-scenarios.ts"; import {writeFileSync} from "node:fs"; writeFileSync("/tmp/fig9-input.json",JSON.stringify({...uniformLabSeed(createMassConservingFigure9DamBreak()),options:UNIFORM_LAB_VALUES,dt:1/30,frames:120}));'
cargo run --release --manifest-path rust/Cargo.toml -p fluid-core --example uniform_geometric_scene < /tmp/fig9-input.json > /tmp/fig9-output.json
```

For the historical frozen-system experiments, apply `diagnostic.patch` in an isolated
checkout at the commit recorded in `results.json` (before the production fix).
It adds temporary environment-controlled probes, not a production fix. Run the
same command with `FIG9_AUDIT=1` to write `/tmp/fig9-pressure.json`, then run:

```sh
cargo run --release --manifest-path rust/Cargo.toml -p fluid-core --example fig9_pressure_audit
AUDIT_CONSTRAINT_LEVELS=100 cargo run --release --manifest-path rust/Cargo.toml -p fluid-core --example fig9_pressure_audit
```

`AUDIT_LEVELS=4` truncates the diagnostic pressure hierarchy; `AUDIT_PHI_LEVELS=100`
preserves mixed-cell air sign throughout; `AUDIT_MASK_AIR=1` masks air pressure
samples during prolongation; `AUDIT_TRACE=1` records active-bound residuals at the
cutoff. The all-level whole-scene experiment sets `AUDIT_CONSTRAINT_LEVELS=100`
and changes the request to 180 frames. No production source change remains.
