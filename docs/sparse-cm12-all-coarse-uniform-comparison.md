# Sparse CM12 all-coarse versus Uniform Dawn

Captured on 2026-08-15 with native Dawn/Metal on the symmetric-expansion
scene. The comparison uses the same physical 1.6 x 0.8 x 1.6 m domain and
initial liquid box. Sparse `all-coarse` publishes a 32 x 16 x 32 texture but
has 16 x 8 x 16 independent control volumes; the matched Uniform arm is run
directly at 16 x 8 x 16.

## The reported loss was represented volume, not mass

CM12 deliberately separates conserved surface-density mass from the volume of
the `rho=0.5` surface. `docs/papers/massConservingLiquids.txt` says both that
mass is conserved and that volume can vary temporarily when density differs
from one. It measures visible volume using a closed marching-cubes mesh.

The A/B runner therefore now reports three different quantities:

- `mass_cells = sum(V_i rho_i)`, the conservation authority;
- `isovalueCellVolume_cells = sum(V_i [rho_i >= 0.5])`, a deliberately named
  cell-centre phase measure, not a marching-cubes volume;
- `subIsovalueMassFraction`, conserved density that is invisible to the
  `rho=0.5` phase test.

These must not be substituted for one another. A watertight marching-cubes
integral remains the required metric for a publication-quality visible-volume
claim.

## Root cause fixed

Sparse sharpening used `3 dt * cellWidth / neighborDistance`, with both lengths
stored in finest-cell coordinates. On a uniform rung the lengths cancel, so
the correction was effectively `3 dt` rather than CM12's
`3 dt |grad rho|`. Uniform Dawn divides density differences by physical cell
spacing.

The corrected expression is

```text
pseudoTimeFineCells = 3 dt / finestCellSize_m
delta-axis = densityDifference * pseudoTimeFineCells / distanceFineCells
```

This is exactly `3 dt * densityDifference / distance_m`. The previous GPU
dose was 10x too weak on the 0.1 m all-coarse cells and 20x too weak on 0.05 m
all-fine cells. The CPU composite oracle now uses the same units.

## Native-Dawn results

Command for the paper-step receipt:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/run-sparse-cm12-long-run-ab.ts \
  --sparse-resolution=all-coarse --uniform-resolution=matched \
  --seconds=2 --regime=paper
```

At 60 steps (`dt=1/30 s`):

| Quantity | Matched Uniform | Sparse all-coarse | Sparse / Uniform |
| --- | ---: | ---: | ---: |
| retained mass | 99.9925% | 99.9976% | — |
| `rho>=0.5` cell volume | 1,776 fine cells | 1,696 fine cells | 95.50% |
| mass below `rho=0.5` | 51.92% | 54.39% | — |
| maximum density | 0.6373 | 0.6141 | 96.35% |
| kinetic-energy proxy | 53.87 | 39.97 | 74.21% |
| density support IoU | — | — | 85.35% |
| final density relative L1 | — | — | 17.89% |

Before the unit correction, Sparse retained 99.9978% mass but ended at maximum
density 0.5700, kinetic-energy ratio 49.07%, support IoU 65.80%, and density
relative L1 20.44%. Thus the correction addresses the reported disappearing
surface without hiding it behind a global mass rescale.

The 500-step scene-cadence stress (`dt=0.004 s`) retains 99.9784% Sparse mass,
but Sparse reaches only 82.76% of matched Uniform's `rho>=0.5` cell measure and
12.52% of its kinetic-energy proxy. This is a remaining dynamics difference,
not a mass-conservation regression.

## Remaining intentional and unresolved differences

1. **Velocity advection.** Uniform uses CM11b bounded modified MacCormack on a
   dense staggered MAC field, including predicted-field extrapolation. Sparse
   currently traces a collocated composite field and blends the first-order
   face prediction with 40% old face velocity. This is the dominant long-run
   dissipation difference at 4 ms. A trial predictor/reverse/correct patch using
   a collocated composite limiter produced 7.67x Uniform kinetic energy and was
   rejected. Parity requires an exact staggered composite sampler and limiter.
2. **Gamma diffusion.** Uniform performs snapshot Jacobi within x, y, z and
   Gauss-Seidel between axes. Sparse averages xyz and zyx sweeps to retain exact
   horizontal D4. Both are conservative, but they are not the same operator.
3. **Sharpening mass return.** Uniform traces up to 2.1 cells along the frozen
   density gradient and scatters trilinearly. Sparse uses adjacent composite
   subfaces, capacity-limits simultaneous deposits, and returns rejected mass
   to its donor. It is local and conservative but not field-identical.
4. **Projection.** Uniform uses the dense CM11a projected multigrid/LCP stack.
   Sparse uses a globally coupled composite PCG solve. On the paper-step run,
   Sparse's maximum published post-projection divergence is `1.335e-5 1/s`,
   just above the current `1e-5 1/s` gate despite a `1.385e-12` relative
   pressure residual. This needs a separate operator/diagnostic-scale audit.
5. **Symmetry policy.** Sparse explicitly quantizes horizontal D4 orbits after
   conditioning and remains near bit-exact. The retained Uniform baseline is
   already asymmetric at long checkpoints; comparisons report both raw Uniform
   and D4-symmetrized density error rather than treating Uniform symmetry as
   ground truth.

The current A/B still fails the strict 5% final-density L1 gate, so the fix is
not a claim of full method equivalence. It is a narrowly verified correction of
the physical-unit error responsible for excess interface diffusion, with the
remaining numerical differences exposed rather than absorbed into thresholds.
