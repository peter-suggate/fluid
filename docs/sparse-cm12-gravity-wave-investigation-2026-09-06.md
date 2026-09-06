# Sparse CM12: controlled waves and coarse-grid physics

The new tests isolate three problems: pressure places the physical free surface
inconsistently across rungs, pressure behavior depends on enforcement-region
metadata even when the actual grid is identical, and a direct width-1 to width-8
restriction can fail its conservation checks through accumulation roundoff.
The last problem is fixed in production. The pressure experiments are diagnostic
ablations, not a completed pressure replacement.

The acceptance target is close amplitudes, arrival times and central motion
relative to adaptive-mass max1 and the Uniform method while retaining coarse
adaptivity. The analytical wave tests isolate causes; passing those tests alone
would not satisfy the scene-level comparison.

The fresh pool comparison also changes the diagnosis: adaptive is weaker on
the first arrival but stronger on the later wall oscillation. A global reduction
of damping would not correct both. All experiments retain coarse cells; Uniform
is not used as a numerical oracle.

## Fresh half-pool comparison through 3 seconds

All arms start from byte-identical density and velocity, use dt = 1/60 s, and
use current native face transport from reset. The frozen arm stops topology at
step 9 (0.15 s). Its nine checkpoint field files match a fresh adaptive run at
that step exactly. Every subsequent frozen topology manifest is checked.

The wall gauge integrates accepted column mass over equal physical patches:
the outer 0.4 m at the four wall centers, each 0.8 m wide. It is independent of
surface meshing and the renderer's density-0.5 crossing.

| Arm | First wall crest, 0.5–1.4 s | Crest time | Wall peak-to-trough, 1.8–3 s | Mean active cells |
| --- | ---: | ---: | ---: | ---: |
| Adaptive mass, max1 | 41.23 mm | 1.167 s | 98.58 mm | 100,352 |
| Coarse-first, frozen at 0.15 s | 25.88 mm | 1.167 s | 68.00 mm | 19,327 |
| Coarse-first, adaptive | 32.65 mm | 1.083 s | 132.39 mm | 30,080 |

Reanalyzing the saved normal-gamma Uniform baseline with those same gauges gives
a 44.28 mm first wall crest at 1.083 s and 84.31 mm later peak-to-trough.
Its initial density and velocity are byte-identical to all three sparse arms.
Uniform's central depression/rebound are -313.8 / +249.5 mm; max1 gives
-374.2 / +386.5 mm; frozen coarse gives -347.4 / +326.2 mm; adaptive gives
-346.9 / +356.0 mm. These are physical comparisons to retain when evaluating
the next fix, not just checks of mass or pressure convergence.

[Four-way center and wall curves](../artifacts/pool-impact-ab/current-wave-investigation/four-arm-center-report/physics-gauges.png)
and [complete four-way measurements](../artifacts/pool-impact-ab/current-wave-investigation/four-arm-center-report/comparison.json).
Uniform is a reused baseline from the earlier dedicated comparison, not a new
run after the restriction fix.

Adaptive's first crest is 20.8% smaller and its later oscillation is 34.3%
larger than max1. The frozen arm's later oscillation is 31.0% smaller.
Final cell kinetic energies divided by density are 0.05591, 0.06920 and
0.05947 m⁵/s², respectively. Lower wall amplitude does not imply proportionally
lower total kinetic energy.

All 540 accepted frames meet the production pressure tolerance of 0.001.
There are zero frame faults, commit mismatches, publication faults, omitted
pages, topology fault frames or rejected candidates in this capture. Thus
rollback does not explain these particular pool differences.

Captures and independent gauges:

- [Wall curves](../artifacts/pool-impact-ab/current-wave-investigation/final-report/wave-gauges.png)
- [Radial propagation](../artifacts/pool-impact-ab/current-wave-investigation/final-report/wave-propagation.png)
- [Central physics](../artifacts/pool-impact-ab/current-wave-investigation/center-report/central-physics.png)
- [Numerical wall measurements](../artifacts/pool-impact-ab/current-wave-investigation/final-report/wave-comparison.json)
- [Every-frame failure audit](../artifacts/pool-impact-ab/current-wave-investigation/audit-summary.json)
- [Freeze checkpoint equality](../artifacts/pool-impact-ab/current-wave-investigation/checkpoint9-equality.json)

These replace the historical pool comparison for current-physics behavior.
The older frozen face-filter ablations deliberately used the previous transport
during warm-up and are still valid equal-checkpoint experiments. The probe now
requires `POOL_LEGACY_WARMUP=1` to request that historical initialization.

## Controlled-wave fixture and measurement

`tools/probe-sparse-gravity-wave-dawn.ts` constructs a sealed rectangular tank,
free-slip walls, zero physical viscosity and surface tension, with a small
cosine water-height perturbation and initially zero velocity. Every rung is a
volume restriction of the same fine quadrature. There are explicit solid shell
voxels. Rectangular dimensions prevent an unrelated square-tank D4 constraint
from being inherited from the flat authored seed.

The default tank is 3.2 × 1.6 × 0.4 m, finest cell width 0.05 m, water depth
1.025 m, amplitude 0.01 m. Its fundamental wavelength is 6.4 m, still 16 cells
per wavelength on width-8 cells. The independent small-amplitude guide is
`omega² = g k tanh(k H)`, giving a period of 2.31598 s; this is linear-wave
theory, not an assertion that a diffuse discrete solver is exact. See
[MIT's linear-wave derivation](https://web.mit.edu/13.012/www/handouts/2003/waves.pdf).

We measure the accepted column-mass Fourier amplitude, phase, other-mode RMS,
mass, pressure receipts and actual cell counts. A stage observer copies both
field parities at extension, face preparation, transport, gamma diffusion,
sharpening, symmetry, gravity, projection and topology transfer. It computes
native face quantities using accepted ownership and physical areas/volumes.
Its end-stage mass agrees with independently expanded dense diagnostics to
7.7e-14 m³ across 82 checked snapshots.

The wave-mode rate is the density-weighted horizontal flux projected against
the gradient of the cosine basis. It detects momentum/phase changes which a
surface-height snapshot alone misses. Native face kinetic quadrature changes
with the grid, so cross-rung energy ratios are not treated as exact loss.

## Frozen grids already have incorrect pressure response

Fixed widths 1, 2, 4, 8, a frozen 2/4 seam, and fully adaptive grids were run
for three analytical periods. The width-4 fundamental has 0.7815 times the
analytical frequency. Its fitted amplitude grows about 5.2% per period, while
other-mode RMS reaches 17.37 mm from a 10 mm seed. A good fit of the fundamental
does not mean the complete wave remains correct.

Width 2 gives 0.8838 times analytical frequency, width 8 about 1.0385. This is
not monotonic convergence. Mixed and adaptive traces are too distorted for a
single damped-sinusoid fit to give a trustworthy damping coefficient.

The fresh max1 run with the production 128-iteration ceiling converges at
every sampled frame, yet the first trough reaches -20.61 mm and the final
fundamental is +25.34 mm. Max1 itself is therefore an imperfect guide.
Halving dt gives a similar fine-grid first trough (-20.65 mm). Raising the
pressure budget to 256 with a tighter tolerance, and disabling gamma diffusion
and sharpening, do not eliminate the problem. Some tight runs reach the f32
residual floor above their requested 1e-5 tolerance; they are not described as
fully converged. Early `fixed1-x` and `oscillate-x` captures used 80 iterations;
the final conclusions use `fixed1-default` and `oscillate-default` at 128.

Swapping x and z gives essentially the same width-4 frequency (0.78152 versus
0.78158). The diagonal run gives 0.76796, but it also has a different total
wavenumber, so this does not isolate angular anisotropy. A traveling packet's
separate reflection/transmission coefficients have not yet been measured.

[Wave traces](../artifacts/sparse-gravity-wave/report/waves.png) and
[fit diagnostics](../artifacts/sparse-gravity-wave/report/comparison.json).

## Causal pressure defects

### Identical grid, different authoring metadata

One whole-domain min4/max4 region and two adjacent min4/max4 half-regions
produce identical accepted topology, initial fields and compiled shaders.
Nevertheless their first restoring responses are 0.6295 and 0.9544 times the
linear prediction. At steps 70 and 280, changing only this metadata while the
topology remains frozen leaves every observed stage metric identical through
body forces; the first difference is pressure projection.

`pressureHasPartialRefinementRegion` gates a column-height correction in
`pressureFaceTheta`. Consequently physical pressure depends on how the same
resolution constraint was authored. The five-column planarity check can also
sample the same coarse footprint five times. This is an operator defect,
independent of topology transfer or irrecoverable loss of surface detail.

### Cell-average density is not a point signed distance

Pressure derives `phi = (0.5 - rho) * cellWidth`. For a 0.2 m coarse cell,
a full cell centered at 0.7 m and a 1/8-filled cell centered at 0.9 m imply a
zero crossing at 0.814286 m, although the physical column height is 0.825 m.
The derivative of this inferred height with respect to fill is also wrong.

On identical width-4 grids, moving the resting surface through fill fractions
1/8, 2/8, …, 7/8 gives first-step restoring responses of approximately
0.625, 0.824, 1.145, 3.686, 1.256, 0.869 and 0.640 times the linear prediction.
This explains why changing a rung can change both wave speed and amplitude,
even with a conserved column height.

### The ghost-distance clamp adds a half-fill discontinuity

The production minimum ghost distance is 5% of cell-center spacing. On a
width-4 cell this is 10 mm. Around an exactly half-filled surface, reducing
wave amplitude from 2.5 mm to 10 micrometres barely reduces the first pressure
impulse: projected wave-mode speed remains about 0.81 mm/s. The corresponding
linear response would be 0.00114 mm/s. An exactly flat control remains calm;
the perturbed response fails to approach that control continuously.

With the same 10-micrometre seed, reducing only the clamp changes the response
ratio from 712.0 to 73.0, 8.09 and finally about 1.88. A diagnostic using
column-derived interface distance together with a 1e-6 clamp gives 0.9265.
That isolates both effects, but is not a safe general pressure fix: a column
does not describe arbitrary overturning liquid, existing pressure membership
and fallback paths remain inconsistent, and very small ghost distances worsen
conditioning. The combined ablation also fails to make the three-period
fine-grid wave correct. No global clamp reduction or column override was
enabled in production.

[Pressure causal plot](../artifacts/sparse-gravity-wave/report/pressure-causes.png)
and [asserted experiment receipts](../artifacts/sparse-gravity-wave/report/causes.json).

## Matched checkpoints and actual topology oscillation

Fine-grid runs are identical through projection at steps 70 and 280, then
one branch coarsens to width 4 through the production transaction and freezes.
This separates restriction from the following coarse-grid physics step.

| Fine checkpoint | Instant fundamental change relative to fine control | Mode-rate ratio after restriction | Following pressure kick, fine / width 4 |
| --- | ---: | ---: | ---: |
| Step 70 | +0.0536 mm | 0.9395 | +2.204 / +1.583 mm/s |
| Step 280 | -0.0869 mm | 1.1752 | -2.313 / -2.395 mm/s |

The sign of the rate error changes with the state. Restriction and subsequent
face preparation matter, but a single energy-retention multiplier cannot
describe them. A direct width-8 restriction at step 70, once the transaction
bug below is fixed, changes the fundamental by +0.2364 mm and retains only
36.5% of the fine control's density-weighted mode rate. This deserves a focused
liquid-flux transfer test; preserved exterior volume flux is not sufficient
evidence that the free-surface wave momentum is preserved.

The real free-surface oscillation test starts with two identical width-2 steps,
then forces width 1 ↔ 2 each frame. All 118 changes commit, and every sampled
pressure solve meets 0.001. On matching width-2 frames, the trajectory differs
from the frozen control by 1.069 mm RMS, maximum 1.759 mm over two seconds.
Sampled individual transfer height changes are 2–6 micrometres. Some refinement
mass differences are about 2e-6 m³; this is small, not exactly zero.
This test supports the earlier staggered-remap fix but does not prove perfect
topology neutrality, especially for larger rung jumps.

## Fixed: accumulation error rejects direct coarse restriction

At step 70, forcing width 1 → 8 initially prepares 32 bricks and commits zero.
The native exterior-face transfer receipts pass. Cell restriction sums up to
512 fine children serially, whereas the before-transfer receipt uses a
workgroup reduction. Rounding differences exceed existing gamma and sometimes
mass tolerances. The transaction correctly rejects the inaccurate candidate.

Restriction now sums short x rows, then y slices, then z. Density, gamma,
pressure, volume, momentum and fallback velocity use this shorter accumulation
path. It retains coarse adaptivity, existing transfer semantics and unchanged
conservation thresholds.

| Evolved width-1 → 8 fixture | Previous | Fixed |
| --- | ---: | ---: |
| Prepared / committed bricks | 32 / 0 | 32 / 32 |
| Maximum gamma receipt error, fine-cell units | 0.0014191 | 0.0000916 |
| Maximum mass receipt error, fine-cell units | 0.0006714 | 0.0000610 |
| Topology fault flags | 2 | 0 |

The fixed physical mass changes by 6.47e-9 m³ through restriction; observed
stages through projection still match the original fine control exactly.
The fixture is now `tests/sparse-cm12-wave-restriction-dawn.test.ts`, included
in the focused pool suite. This failure was found in the forced-wave test;
the fresh pool audit above had no such rejected transactions.

[Before/after receipt](../artifacts/sparse-gravity-wave/report/width8-restriction.json).

## Reproduction and remaining work

Run one Dawn process at a time with the Fluid browser closed. The probes use
the ordinary exclusive WebGPU lease. Example commands from the repository:

```sh
node --import tsx tools/probe-sparse-gravity-wave-dawn.ts --arm=fixed4 --audit=1,70,280
node --import tsx tools/probe-sparse-gravity-wave-dawn.ts --arm=mixed
node --import tsx tools/probe-sparse-gravity-wave-dawn.ts --arm=oscillate --steps=120 --sample-every=1 --audit=3,4,59,60,119,120
node --import tsx tools/probe-sparse-gravity-wave-dawn.ts --arm=fixed8 --fork-step=70 --steps=71 --audit=70,71
node --import tsx tools/probe-sparse-gravity-wave-dawn.ts --arm=fixed4 --height=.9 --amplitude=.00001 --steps=1 --audit=1
npm run test:dawn:sparse-cm12:pool-impact
npm run test:dawn:sparse-cm12
```

Use distinct `--output=...` directories for variants. `--direction=z`,
`--direction=diagonal`, `--dt=0.008333333333333333`, `--conditioning=off`,
`--pressure-iterations=256` and `--pressure-tolerance=.00001` provide controls.
The pressure-boundary and theta overrides exist only in this diagnostic probe.
Configurations record actual shader hashes and arguments.

To prevent concurrent presentation edits from changing A/B arms midway through
the study, the long runs used `/tmp/fluid-gravity-wave-20260906`, a source copy
with the resident TypeScript/WGSL pinned to staged blobs
`50e607d5b245efd1ea3479106c548cbf00253321` /
`41b80090b2b722470a25c9e9fb6fb26c2ac1a4a0`. The width-8 fixed run adds only
the restriction reduction to that snapshot. The pool/wave measurements above
otherwise precede this final reduction fix. The regression gate runs in the
shared current repository and therefore also includes concurrent changes.

The next physics work should make pressure membership, interface location and
gradient agree on a physical surface across rungs, with conditioning that does
not move that surface. Its acceptance criteria are metadata invariance,
continuous response as amplitude tends to zero at every fractional height,
correct long-wave phase without growing spurious modes, and stable flat and
overturning-liquid controls. The width-8 checkpoint also
provides a concrete test for liquid-weighted flux loss during restriction.
Only after those pass will mixed-interface packet transmission and the pool
comparison give clean measures of residual transport dissipation.

## Validation of the production restriction change

`npm run test:dawn:sparse-cm12:pool-impact` passes all 14 tests in the shared
current repository, including the new 71-step real-wave restriction regression
and the prior native-face, topology-oscillation, D4 and display tests.

The unfiltered `npm run test:dawn:sparse-cm12` was run with unchanged ceilings.
It does **not** pass: mini32 measures 57.15 ms against 40 ms, mini64 67.96 ms
against 50 ms, and the run exhausts its 180-second total budget. Ten
correctness lanes pass within budget; the tall-cells test prints a pass but its
process exceeds the remaining suite budget. Three following lanes are not run
inside that invocation. These are shared-tree measurements, not an isolated
before/after performance attribution to the reduction change.

[Full gate receipt](../artifacts/sparse-gravity-wave/report/canonical.json) and
[focused suite log](../artifacts/sparse-gravity-wave/report/pool-focused.log).

The four budget-affected lanes (tall cells, live rigid insertion, live liquid
insertion, outside-tank collapse) each pass when run separately with their
original limits. Thus all 14 correctness lanes have passed, but the unfiltered
suite remains red for performance and total duration. Per-lane logs are saved
beside the full receipt. No lane or timing ceiling was changed.

ESLint passes for the changed TypeScript probes and regression; Python analysis
files parse successfully; `git diff --check` passes. Whole-repository TypeScript
checking reports 50 existing errors and no errors in the changed wave tools,
pool probe or new restriction test.
