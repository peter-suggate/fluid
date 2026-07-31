# Fluid-gated octree boundary refinement

## Paper compatibility

This experiment is compatible with Aanjaneya et al. (2017), and is closer to
the paper's stated capability than unconditional boundary refinement:

- the introduction says the power discretization supports adaptivity along
  solid and air boundaries (`aanjaneya-2017-power-liquids.txt`, lines 91-104);
- Section 4 says embedded free-surface and solid boundaries may cut cells even
  near T-junctions, while previous schemes required uniformly refined boundary
  cells (lines 283-300);
- the examples choose terrain-proximity refinement, but describe it as an
  adaptivity pattern rather than a discretization requirement (lines 580-625).

The repository's unit-cell closed-wall strip and unconditional
solid-crossing split are therefore implementation policy, not a paper
invariant.

## Production rule

Fluid-gated refinement is the default. The former unconditional policy remains
available as an exact validation control with:

```sh
FLUID_OCTREE_FLUID_GATED_BOUNDARIES=0
```

The existing interface and inflow predicates run first and are identical in
both arms. Pressure refinement keeps the production candidate-scaled support
ring and three-generation topology retention. A leaf crossing a closed wall or
the terrain/solid boundary is then split when its conservative minimum liquid
phi is within the authored interface band. This is important during cold
bootstrap, before the fine-summary hierarchy exists; testing only negative phi
there changes the mesh too late and measurably changes the dam-break trajectory.

Pressure retention is stored per 8-cubed topology tile. It may preserve an
interior pressure shell, but it is deliberately ignored when its only effect
would be to pin an unrelated, locally dry boundary crossing. Current local
pressure evidence still wins before that boundary decision, so approaching
fluid refines the cell before contact.

This gives the intended lifecycle:

1. a dry boundary region may remain coarse;
2. the existing interface band splits it before contact;
3. a boundary leaf containing liquid remains refined;
4. grading still closes the result to the paper's 2:1 and mixed-ring rules.

Recurring generations use the already-published fine/coarse minimum-phi
summary. Only cold bootstrap scans the imported/analytic phi cells.

## Earlier large-scene structural measurement

The following garden result established the size of the opportunity in the
original prototype. It is retained as historical evidence, not as a parity
gate for the corrected boundary-only policy above.

Scene: `garden-pond`, grid `120 x 40 x 88`, maximum leaf 16, two accepted
steps at `dt=1/120 s`, Apple M1 Max Metal backend.

Set `FLUID_OCTREE_TOPOLOGY_CENSUS=1` to emit the diagnostic accepted-owner
histogram after the measured simulation window.

| Metric | Unconditional control | Fluid-gated | Change |
| --- | ---: | ---: | ---: |
| Structural leaves | 139,663 | 20,607 | -85.25% |
| Size-1 leaves | 119,208 | 8,800 | -92.62% |
| Size-2 leaves | 18,523 | 8,660 | -53.25% |
| Size-4 leaves | 1,862 | 2,828 | +51.88% |
| Size-8 leaves | 70 | 319 | +355.71% |
| Wet pressure frontier | 20,115 | 20,115 | unchanged |
| MGPCG rows | 7,523 | 7,523 | unchanged |
| Simulation wall time, 2 steps | 899 ms | 876 ms | -2.56% (one noisy sample) |
| Inactive topology candidate | 15.01 ms | 19.27 ms | +28.38% |

Both arms reported:

- 2:1 maximum neighbour ratio;
- authoritative power topology;
- zero descriptor/topology errors;
- converged MGPCG;
- no non-finite values or stability flags;
- identical final volume-field summary in this short window.

The experimental maximum speed was 0.384 m/s versus 0.360 m/s in the control,
so the short run is not sufficient to claim physical parity. The large
structural reduction also does not yet reduce fixed-capacity allocation or the
wet pressure solve; it mainly removes needless dry owner detail.

The first implementation scanned phi cell-by-cell on every generation. It
preserved the same 20,607-leaf result but increased the inactive topology phase
to 36.83 ms. Reusing the summary hierarchy reduced that to 19.27 ms.

## Mini dam 16 moving-interface A/B

Scene: `minimal-power-dam-break`, grid `16 x 16 x 16`, 62 accepted steps at
`dt = 0.004 s`, Apple M1 Max Metal/Dawn backend, interface band 3, global-fine
factor 4.

The reverted implementation bundled three independent changes under the gate:

1. dry wall/terrain crossings could remain coarse;
2. the free-surface pressure support ring was narrowed;
3. three-generation pressure-topology retention was disabled.

That changed the pressure discretization rather than merely removing dry
boundary detail. At step 62 it reduced pressure rows from 1,475 to 1,348,
shifted the reconstructed field to wet-cell IoU 0.966 and a 0.073-cell
centroid displacement, and increased wall time from 70.8 to 125.5 ms/step.

The correction makes the gate boundary-only. Both arms use the same production
pressure support and retention. Tile retention is ignored only for a locally
dry boundary candidate with no current pressure evidence. An authoritative
sparse positive-air complement is also treated as outside the active fluid
band instead of as a measured distance of one maximum-leaf width; as fluid
approaches, a real fine/coarse summary replaces that complement and triggers
pre-contact refinement.

The 62-step Dawn A/B is:

| Metric | Control | Fluid-gated |
| --- | ---: | ---: |
| Initial structural leaves | 3,648 | 2,752 |
| Initial leaf histogram | 3,584 x size 1; 64 x size 2 | 2,560 x size 1; 192 x size 2 |
| Terminal structural leaves | 4,096 | 3,270 |
| Terminal leaf histogram | 4,096 x size 1 | 3,152 x size 1; 118 x size 2 |
| Terminal field cell sum | 1,474.008386 | 1,474.008386 |
| Terminal pressure rows | 1,453 | 1,453 |
| Terminal pressure iterations | 5 | 5 |
| Field MAE / RMSE | 0 / 0 | 0 / 0 |
| Wet IoU / centroid distance | 1 / 0 cells | 1 / 0 cells |
| Simulation wall time | 7,038 ms | 6,792 ms |
| Wall time per step | 113.524 ms | 109.551 ms |

The terminal gated mesh places 64 size-two leaves at origin Y=14 and another 27
at Y=12, opening the drained ceiling region while retaining current pressure
support below it. The A/B fields are bit-identical, both arms have the same
pressure rows and convergence receipt, and both reported zero WebGPU validation
errors. The latest control-first sample is 3.5% faster on the adaptive arm.
Earlier order-swapped and control-first samples ranged from 0.32% faster to
0.52% slower, so the small-domain steady-state effect is noise-scale to a small
win; the pressure solve remains structurally identical. The automated gate
rejects a slowdown above 10%.

Construction is reported separately from simulation time. In the reversed
sample it took 1,920 ms for adaptive versus 1,787 ms for the control. The
adaptive cold bootstrap must query liquid distance before a published summary
exists, so a modest one-time cost is expected; sequential shader compilation
also makes this measurement order-sensitive. It is not included in the
per-step performance gate.

### Per-step convergence gate

Terminal diagnostics alone previously hid transient failures. The benchmark's
`FLUID_MINI_DAM_AUDIT_EVERY_STEP=1` mode therefore serializes diagnostic
readback after every accepted step and requires the immutable 62-step snapshot
sequence (step, executed iterations, convergence, authority lag, prediction
failures, and overflow) to match the unconditional control. It deliberately
does not compare the separately submitted live compaction receipt: that probe
can race the following generation and has reported transient zero-row samples
even while the step snapshot is valid. This diagnostic timing is not used as
the product wall-clock measurement. The production solve-tail policy remains
unchanged.

The strict audit passes all 62 snapshots exactly. The current baseline exhausts
its encoded solve budget during the early portion of this test; the complete
failure/convergence sequence is identical in both arms. That is an existing
solve-tail profile, not an adaptive-boundary difference.

## Factor-1 water-box far-wall gate

The interactive water-box can explicitly select factor-1 surface tracking.
Two independent policies previously defeated dry-boundary adaptivity there:

1. method construction silently selected unconditional boundaries whenever
   factor 1 was active;
2. factor-1 pressure evidence forced every size-two leaf to unit size before
   checking whether its B4 summary contained liquid.

The method no longer opts factor 1 out of the production default. Factor-1
pressure evidence still preserves unit cells for any B4 summary containing
liquid, but a positive-air B4 now proceeds through the normal distance and
boundary decisions. This removes dry detail without changing wet pressure
resolution.

The construction-time Dawn gate uses the exact `24 x 18 x 16` water-box,
factor 1, maximum leaf 32, and band reach 3:

| Metric | Unconditional control | Adaptive default |
| --- | ---: | ---: |
| Structural leaves | 5,792 | 2,901 |
| Pressure rows | 1,352 | 1,352 |
| Initial field MAE / RMSE | - | 0 / 0 |
| Far X strip | 864 x size 1 | 16 x size 2; 16 x size 4 |
| Far Z strip | 1,296 x size 1 | 176 x size 2; 5 x size 4 |

Thus both dry far walls contain no unit leaves in the adaptive arm. The gate is
`npm run test:webgpu:water-box-boundary-adaptivity`; it also rejects pressure,
volume, field, or WebGPU-validation differences.

### Mini16 factor-1 zero-refinement control

The factor-1 comparison must keep two independent controls separate. `off`
disables only dry-boundary gating; `zero` sets octree adaptivity to zero and
therefore publishes all 4,096 unit leaves. The latter exposed a surface-state
bug that pressure A/B tests could not diagnose: the liquid-row-only factor-1
directory had no air-side degrees of freedom, so even the fully refined
pressure arm stopped at the original dam face (`maximumX = 9`). Redistancing
the live liquid rows cannot create a value in a missing dry row.

Factor 1 now uses its sparse B4 interface band as the production authority.
That band is still at octree resolution and local to the interface, but it
includes the positive-air halo required for outward semi-Lagrangian transport.
The liquid-row-only directory remains an explicit diagnostic path rather than
being selected merely because the factor is one.

At 44 steps (`t = 0.176 s`) on Dawn/Metal, maximum leaf 32 and band reach 3:

| Metric | Zero refinement | Adaptive default |
| --- | ---: | ---: |
| Structural leaves | 4,096 | 3,872 |
| Maximum wet X / Z | 11 / 11 | 11 / 11 |
| Mass beyond original X=9 face | 106.108258 | 106.599017 |
| Field MAE / wet IoU | - | 0.0003399 / 1.0 |
| Centroid distance | - | 0.00304 cells |
| Clean wall time | 113.964 ms/step | 111.423 ms/step |

The gated and ungated adaptive fields are bit-identical. The benchmark pins
the zero-refinement comparison (MAE below 0.001, identical wet support and
front extent, centroid below 0.01 cells, leading mass within 1%, fewer leaves,
and no more than 10% wall-clock slowdown).

This endpoint comparison is not a license to replace the paper's factor-4
surface representation with pressure rows. Dawn startup measurements expose
the distinction: after two 4 ms steps, the factor-1 band has only 0.112
cell-equivalents beyond the authored face in the zero-refinement arm and has
not resolved x=10, whereas factor 4 resolves x=10 and 18.914 cell-equivalents.
The factor-4 adaptive field remains within 0.00035 MAE of zero-refinement
pressure. Section 5 deliberately separates the adaptive background pressure
octree from a higher-resolution sparse interface grid; dry pressure cells may
coarsen without reducing that interface resolution. Factor 4 therefore
remains the product/paper default, while factor 1 is a lower-resolution
surface experiment and liquid-row-only transport is diagnostic-only.

## Current conclusion

The idea works structurally and is supported by the paper: dry terrain detail
drops by more than an order of magnitude without changing the wet pressure
frontier in the measured large scene. The corrected implementation is now the
default, with maximum leaf size 32. It changes only dry boundary ownership;
the pressure shell, temporal retention, solve tail, transport, and surface
publication remain the production policies.

On mini16 the default preserves adaptivity through its first advance, coarsens
the drained ceiling region, removes 826 terminal structural leaves, and is
bit-identical to the former policy after 62 steps. The exact control remains
available through `FLUID_OCTREE_FLUID_GATED_BOUNDARIES=0`. Larger moving-terrain
and inflow soaks remain useful performance coverage, but a result or pressure
profile divergence is a correctness failure rather than an accepted tradeoff.
