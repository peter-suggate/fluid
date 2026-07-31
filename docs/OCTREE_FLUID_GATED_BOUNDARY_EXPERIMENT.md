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

## Experimental rule

Fluid-gated refinement remains an experiment. Enable it explicitly with:

```sh
FLUID_OCTREE_FLUID_GATED_BOUNDARIES=1
```

The current interface and inflow predicates run first. A
leaf crossing a closed wall or the terrain/solid boundary is then split when
its conservative minimum liquid phi is within the authored interface band.
This is important during cold bootstrap, before the fine-summary hierarchy
exists; testing only negative phi there changes the mesh too late and
measurably changes the dam-break trajectory. Recurring pressure refinement
retains a size-scaled grading ring for candidates larger than size two. The
final size-two to size-one split requires an actual interface sign crossing:
the power discretization represents a nearby free surface on the adaptive
cell, so proximity alone is not a reason to make a unit cell.

This gives the intended lifecycle:

1. a dry boundary region may remain coarse;
2. the existing interface band splits it before contact;
3. a boundary leaf containing liquid remains refined;
4. grading still closes the result to the paper's 2:1 and mixed-ring rules.

Recurring generations use the already-published fine/coarse minimum-phi
summary. Only cold bootstrap scans the imported/analytic phi cells.

## Measurement

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

The first moving-interface run exposed a cold-bootstrap defect in the
experiment. Before the fine-summary hierarchy exists, the gate tested only
`phi < 0`, so dry boundary leaves inside the authored interface look-ahead
were not split soon enough. At maximum leaf size 2 this produced:

- scalar-field MAE 0.004536 and RMSE 0.014810;
- 1.2407% relative volume difference;
- wet-cell IoU 0.98330;
- 0.0394-cell centroid displacement.

The corrected bootstrap/fallback test compares conservative minimum phi with
the authored interface-band distance. A second defect appeared after the first
advance: the recurring proximity predicate added the full interface band to
the final size-two split, covering almost the entire tiny domain and replacing
the adaptive topology with 4,096 unit leaves. Tile-wide retention prolonged
the same error.

The recurring rule now splits size-two cells only on a sign crossing and has
no tile-wide temporal retention. More importantly, an authoritative sparse
positive-air complement is treated as outside the active fluid band instead
of as a measured distance of one maximum-leaf width. That nominal 0.10 m value
was smaller than the mini dam's 0.15 m look-ahead and therefore refined every
dry boundary strip indefinitely. The compact pressure system needs the paper's
production Section 4.3 boundary shell; the small two-level default is therefore
`k=8` instead of `k=4`.

At one step the corrected default preserves its initial 2,752 leaves (2,560
size one and 192 size two), not 4,096, and converges in 9 of the unchanged 10
outer iterations.

The 62-step Dawn A/B is:

| Metric | Control | Fluid-gated |
| --- | ---: | ---: |
| Initial structural leaves | 3,648 | 2,752 |
| Initial leaf histogram | 3,584 x size 1; 64 x size 2 | 2,560 x size 1; 192 x size 2 |
| Terminal structural leaves | 3,858 | 2,948 |
| Terminal leaf histogram | 3,824 x size 1; 34 x size 2 | 2,784 x size 1; 164 x size 2 |
| Terminal field cell sum | 1,475.369294 | 1,475.369294 |
| Terminal pressure rows | 1,324 | 1,324 |
| Terminal pressure iterations | 5 | 5 |
| Field MAE / RMSE | 0 / 0 | 0 / 0 |
| Wet IoU / centroid distance | 1 / 0 cells | 1 / 0 cells |
| Simulation wall time | 7,099 ms | 7,093 ms |
| Wall time per step | 114.496 ms | 114.404 ms |

The terminal gated mesh places 64 size-two leaves at origin Y=14, coarsening
the entire top two-cell slab; another 27 occur at Y=12 and 28 at Y=10. This is
the expected upward migration into air as the reservoir drains. The A/B fields
are bit-identical, both arms converged MGPCG, and both reported zero WebGPU
validation errors. Wall time is effectively tied in this sample (0.08%).

### Per-step convergence correction

The original 62-step report inspected the terminal MGPCG control and therefore
missed transient failures earlier in the same run. The later per-step Dawn
tripwire exposed `ERROR_NONCONVERGENCE` at steps 31, 36, and 37 with `k=6`;
each exhausted the ten encoded outer iterations and retained the pressure seed.
The same validated 125-step compact-topology run passes with `k=8`. Restoring
the former three-generation tile-wide pressure-retention policy also makes the
run pass, but needlessly pins unrelated dry leaves. The production correction
therefore keeps the compact topology and selects `k=8`.

## Current conclusion

The idea works structurally and is supported by the paper: dry terrain detail
drops by more than an order of magnitude without changing the wet pressure
frontier in the measured large scene. With the corrected `k=8` shell, the mini
dam moving-interface case preserves adaptivity through its first advance,
coarsens the drained ceiling region, and has exact 62-step field parity. Its
wall time is currently tied with the unconditional control.

The regular `dam-break-ui` scene exposed an authority-cutover gap when the
experiment was made the default: generation 3 activated a different pressure
row set while the coarse fine-level-set publication still retained generation
2, so topology, transport, and volume rolled back together. Until that
cross-topology migration is implemented and validated, general interactive
scenes retain unconditional boundary refinement, including the compact k=4
power-validation profile used for morning-parity validation. The fluid gate,
narrower candidate-relative pressure shell, and disabled temporal retention
remain one explicit experiment rather than changing production physics. A
larger moving-terrain soak is also still needed to establish parity and recover the remaining
topology-candidate overhead. The existing `garden-dam-break` control
currently fails its cold bootstrap with no liquid-row frontier, so it cannot
yet provide that unbiased moving-contact A/B.
