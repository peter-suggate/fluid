# Half-pool airborne residue

Reproduction: `coarse-first-pool-impact-half`, adaptive-mass balanced production
defaults, paper timestep (1/30 s), 64×48×64. The probe reads diagnostic density
and velocity, sums density above fine-grid y=32, and checks simulation health.

```
npm run test:dawn:sparse-cm12:trace-gravity
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-half-pool-residue-dawn.ts
```

## Cause

`forceFaces` already adds gravity to open accepted faces, independently of
density. But VEX initialization discarded the resulting cell velocity whenever
rho was at or below the 0.5 liquid isovalue. Those mass-bearing cells then
received nearby liquid's extrapolated motion, or zero outside its extension
band, instead of retaining their own momentum. Collocation likewise withheld
the newly forced velocity from their effective transport plane.

Both velocity ownership decisions now use positive density. The pressure
membership/isovalue and zero-density air extension remain unchanged. This
preserves dilute mass's acceleration without deleting density or changing the
renderer's visibility threshold.

## Measurements

Integrated density in finest-cell volume units, upper region y≥32:

| Paper step | Before upper mass | After upper mass |
| --- | ---: | ---: |
| 0 | 3408.5 | 3408.5 |
| 15 | 0.00501387 | 0.000115940 |
| 30 | 0.00133976 | 0 |
| 90 | 0.248044 | 0 |
| 120 | 5.18178 | 0 |

Later steps include subsequent splash motion, so the scene regression tests
the first drop at step 30. At step 15 the corrected residual's mass-weighted
vertical speed is −4.89425 m/s, close to g×0.5 s; previously it was −1.49339 m/s.
Initial total integrated density was 69724; corrected final total at four
seconds was 69725.4553 (0.0021% difference).

The Dawn kernel regression executes the generated production VEX initializer
with densities from 1e-8 through 1 and a known gravitational velocity. It fails
on the old predicate at density 1e-8 and passes with positive-density ownership.
The scene regression also requires mass drift below 0.1%.

## Broader validation

`npm run test:dawn:sparse-cm12` was run without modifying its lanes, ceilings,
or 180-second budget. It did not pass: symmetry, hydrostatic adaptivity and the
mixed-rung surface lane failed; mini64 performance timed out at 30 seconds,
and the suite exhausted its total budget during the long-dam lane.

Removing only the two velocity-predicate changes reproduced failures in all
four independently rerun lanes: symmetry (0.42416-cell density discrepancy),
hydrostatic reset waterline (15.5862 vs 15.25), mixed-rung surface ridge
(0.0878882 cell), and the mini64 30-second timeout. These are existing working
tree blockers, not a passing broad acceptance result. The corrected full run
passed mini32 mass retention and its 40 ms performance ceiling (28.9014 ms).

The 15 velocity-extension scheduling/effective-plane unit tests pass.
Repository-wide TypeScript checking reports errors in existing unrelated
files; it is not a clean repository-wide typecheck.
