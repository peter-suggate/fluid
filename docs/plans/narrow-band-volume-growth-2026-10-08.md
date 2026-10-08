# Narrow-band Figure 2 volume-growth investigation

## Status and constraints

The runaway is reproduced and localized to the moving particle/reconstruction/
resampling loop. **A complete causal explanation and a validated fix remain
open.** No production shader, method default, pressure schedule or test bound
was changed by this investigation. The experimental changes live only in
`tools/probe-narrow-band-growth.ts`.

Global correction is rejected: a shared deficit cannot identify the body that
lost liquid, and an outward shift can inflate an unrelated falling body. A phi
shift also disagrees with the retained particles that rebuild the next surface.
More passes, pressure cycles or smaller timesteps are not proposed remedies.
The higher-accuracy controls below were diagnostics, not shipping candidates.

## Reproduction and evidence

Scene: `cm12-figure-2`, 128 × 128 × 8, balanced quality, narrow-band app defaults,
dynamic surface refinement, 1h grid padding, no sources, no outflow. The UI's
1/60 s step is reproduced with an explicit scene timestep; 306 steps cover
5.1 s. Volume is sampled every step, not just at the final frame.

The first run included the pre-existing uncommitted trajectory optimization.
Other work continued editing those files during this investigation. The later
`head-*` controls therefore use an isolated source snapshot at `8c738d58`.
Do not treat runs across these groups as exact performance or accuracy A/Bs.
Even the fixed-trace runs have trajectory sensitivity and GPU particle ordering;
the robust result is persistence of large errors, not a ranking of final values.

Each JSON records arguments, scene, method values, source hash and per-frame
measurements. Later records also contain individual solver-file hashes.

| Control | Minimum volume drift | Final volume drift | What it establishes |
| --- | ---: | ---: | --- |
| Initial working-tree default | −13.23% | +810.60% | Reproduces the user's failure |
| Half-h RK4 tracing | −13.22% | +556.29% | Recent trace merging is not required |
| No reseeding after bootstrap | −50.47% | −2.84% | Removes runaway, but loses half the fluid earlier; invalid fix |
| Reseed only below 2h, half-h tracing | −20.86% | +136.15% | Deeper seeding reduces the feedback but still fails badly |
| Shrink shallow phi even without nearby particles | −13.17% | +667.25% | The empty-particle-tile shortcut is not the main cause |
| Tighter pressure control | −14.39% | +531.25% | Pressure accuracy alone does not cure it |
| Snapshot HEAD, default | −13.21% | +413.87% | Runaway also exists before the uncommitted optimization |
| Snapshot HEAD, all cells fine | −19.47% | +227.69% | h/4h seams are not necessary |
| Snapshot HEAD, 1/240 s, 1224 steps | −1.00% | +2526.79% | Smaller global steps are not a cure |

The tighter-pressure diagnostic used tolerance 0.01 s⁻¹ and 14 band cycles,
only inside that probe process. Its final coarse/band residuals were
0.00431/0.00133 s⁻¹, yet its volume remained more than six times the reference.
This experiment preceded the explicit request not to increase solver work and
is rejected as a solution.

Data: [summary](../verification/narrow-band-growth-summary.json),
[initial default](../verification/narrow-band-growth-baseline.json),
[half-h tracing and seed audit](../verification/narrow-band-growth-audit-fixed.json),
[no reseeding](../verification/narrow-band-growth-no-reseed.json),
[deeper seeding](../verification/narrow-band-growth-deep-seed.json),
[empty-tile shrink](../verification/narrow-band-growth-shrink-empty-bounded.json),
[pressure control](../verification/narrow-band-growth-tight-pressure.json),
[snapshot default](../verification/narrow-band-growth-head-baseline.json),
[all fine](../verification/narrow-band-growth-head-full.json),
[quarter timestep](../verification/narrow-band-growth-head-quarter-step.json).

## Stage attribution

The existing read-only probe integrates tetrahedral surface volume immediately
before advection, after advection, after particle reconstruction and after
redistancing. Inter-frame changes include refinement remapping. These stage
integrals and the HUD occupancy estimate are different quadratures; both show
the runaway. The smaller difference between those measurements is not its cause.

For the first 306-step working-tree run, accumulated changes in h³ were:

| Stage | Volume change |
| --- | ---: |
| Level-set advection | −95,825.53 |
| Particle reconstruction | +135,884.99 |
| Redistancing | 0.00 |
| Between frames / remapping | −385.55 |

This identifies where the gain enters the surface, not necessarily where the
underlying particle error originates. In particular, particles may have moved
incorrectly before reconstruction encloses them. It would be premature to
call the sphere radius alone the proven root cause.

## Simple-bug checks

- **Bookkeeping:** the source/outflow budget stays constant. Growth is present
  in the independently measured surface, not just a counter or centre-phi
  estimate. No particle-capacity clipping occurs in these runs.
- **Redistance:** every measured reconstruction/redistance pair agrees exactly
  in the stage probe. Preserved interface vertices are doing their job.
- **Refinement:** remapping is a small net loss in the initial run; the all-fine
  control still grows without remapping.
- **Reseeding into air:** among 799,358 end-step seeds in the fixed-trace audit,
  11 centres were outside phi and 3,852 spheres enclosed an existing positive
  vertex. This is a real discrepancy worth investigating, but does not prove
  those seeds account for the runaway. The audit is not an exact incremental
  volume integral: even a sphere that flips no vertex can move a crossing.
- **Stationary feedback:** freeze particle motion and phi advection after
  frame 60, while retaining reconstruction and reseeding. Surface volume is
  4,432.377 h³ at frame 60 and settles to 4,432.387 h³ by frame 62, remaining
  there through frame 120. Reseeding falls to zero. The moving loop is required;
  repeated reconstruction of that frozen state does not inflate it.
  [Frozen-state evidence](../verification/narrow-band-growth-freeze.json).
- **Empty-particle tiles:** extending shrinkage into shallow uncovered liquid
  does not solve it. An initial overly broad diagnostic also clamped distant
  air and correctly failed the residency gate at frame 2; the bounded version
  in the table preserves distant air. Neither change is applied to production.
- **Radius transcription:** the current 0.875h radius agrees, to rounding, with
  mantaflow's default 3D radius, `0.5 * sqrt(3) * (1 + 0.01)`. There is no
  factor-of-two porting bug there. Its reference resampling threshold and
  sampling distribution differ, so matching radius alone does not establish
  equivalent behavior. [Primary implementation](https://github.com/tum-pbs/mantaflow/blob/master/source/plugin/flip.cpp).

A separate manufactured particle-cloud calculation applies determinant-one
affine deformation with no solver or reseeding. Its reconstructed volume can
gain or lose; it does **not** reproduce the runaway. This prevents treating a
generic fixed-radius argument as sufficient proof of this particular failure.
[Manufactured data](../verification/narrow-band-growth-affine-spheres.json).

## Remaining work

The next useful isolation is the local transport/reconstruction/resampling
contract: distinguish changes in transported particle spacing from changes
created when a newly reconstructed shape becomes the next seeding domain.
Particle count cannot serve as a conserved volume because resampling changes
it by design. The current tests do not yet distinguish interpolation-induced
particle expansion from surface reconstruction error amplified by reseeding.

A replacement must remain local to the affected liquid, preserve thin features
and detached droplets, and pass a disconnected quiet-body control. A component
budget would avoid cross-body redistribution only if its target survives splits,
merges and source insertion and the particles share its correction; simply
splitting the current global scalar into components is not a complete fix.

Keep the existing 10% Figure 2 bound, the rest-state and translating-sheet
contracts, and the repository's type/unit/Dawn gate. The early loss is a real
failure too, even where final volume happens to recover. No bounds or timing
ceilings should be relaxed to make a proposed reconstruction pass.

## Validation

`npm run check:types` passes. `npm run test:unit` passes 903 tests, with 103 GPU
skips. `npm run test:dawn -- narrow-band-volume` on the isolated HEAD snapshot
fails the two existing Figure 2 accuracy tests (early volume loss); its scalar
budget/source/outflow test passes. These are pre-existing simulation failures,
not relaxed expectations. [Dawn output](../verification/narrow-band-growth-dawn-volume.log).
The complete Dawn suite was not rerun, and this is not a clean full-gate claim.
The diagnostic tool's six-frame smoke run completed with no GPU errors.

Reproduce the default diagnostic from the repository root:

```sh
node --import tsx tools/probe-narrow-band-growth.ts baseline
```

The default is 306 frames at 1/60 s with per-frame stats and stage integrals.
Probe passes/readbacks and concurrent GPU work make its timing fields unsuitable
for performance claims.
