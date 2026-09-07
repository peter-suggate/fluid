# Frozen max1 versus min1/max2: symmetry and physics

The A/B confirms a material resolution dependence, even before appreciable
symmetry loss. It also exposes a shared long-term density/energy drift in the
all-fine reference. A should therefore be a comparison baseline, not assumed
physical ground truth.

The follow-up [mechanism investigation and repair plan](adaptive-mass-mechanism-plan-2026-09-07.md)
identifies a reproduced mixed-stencil interpolation defect and incorrect use
of pressure-row connectivity in velocity extension and capacity repair. It
sets the implementation order and separates these findings from the remaining
energy/pressure attribution questions.

## Controlled experiment

Both arms use `coarse-first-pool-impact-quarter`, h = 0.05 m, dt = 1/30 s,
and the same saved method values. Topology is frozen immediately after reset.
The capture extends to 8 s (240 steps), including the original 4 s checkpoint.

- **A:** whole-domain min1/max1; 48 fine bricks, 24,576 cells.
- **B:** whole-domain min1/max2; 8 fine bricks and 40 width-two bricks,
  6,656 cells. Widths are 0.05 m and 0.10 m.
- Initial density and velocity fields are **bit-identical**. Initial liquid
  mass is 1,030.828663 kg (1,032.6875 litres).
- Every checkpoint retains identical leaf membership, coordinates, spans, and
  resolution. Both initial topologies retain exact x reflection, z reflection,
  and x/z swap symmetry throughout.
- Configurations differ only in the maximum cell size. No solver changes,
  symmetry averaging, or additional simulation passes were introduced for this
  experiment. Stage captures are read-only copies.
- Pressure settings are pinned to the captured current defaults: 128 maximum
  iterations, relative tolerance 0.001. These differ from the earlier
  investigation's 16-iteration snapshot; compare A and B here, not numerical
  values across those separate experiments.
- Both configurations record resident shader SHA-256
  `92e12d25bf91326c92bffbbb980fe913df8bfe7e4144a73b04e23318906d77fc`.

![A/B metrics](../artifacts/pool-impact-symmetry/ab-analysis/comparison.png)

## Physics differences precede symmetry loss

Through step 8 (0.267 s), A and B are nearly identical. At step 10 (0.333 s),
B's kinetic energy is already 11.4% lower: 26.56 J versus 29.99 J. The difference
then reverses; at 0.5 s B has 41.2% more kinetic energy. This is a change in
impact response and subsequent motion, not merely a fixed damping multiplier.
At 0.5 s, density symmetry maxima remain below 0.000183 in both arms, and
velocity symmetry maxima below 0.000078 m/s.

| Time | Kinetic energy A | Kinetic energy B | B/A | Column liquid-depth RMS difference | Density displacement metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 s | 9.47 J | 13.37 J | 1.41 | 20.6 mm | 1.64% |
| 1 s | 6.67 J | 8.09 J | 1.21 | 28.2 mm | 4.37% |
| 2 s | 5.06 J | 4.19 J | 0.83 | 25.6 mm | 5.45% |
| 4 s | 2.37 J | 5.29 J | 2.23 | 23.9 mm | 5.61% |
| 8 s | 2.18 J | 10.55 J | 4.83 | 41.9 mm | 8.97% |

The displacement metric is `sum(abs(rho_A-rho_B))/(2*initial_mass_in_cells)`;
it measures distribution disagreement, not lost mass. After conservatively
restricting A onto B's actual cells **in offline analysis only**, it remains
4.25% at 4 s and 8.21% at 8 s. Most of the late discrepancy therefore survives
removal of A's within-cell detail. At 8 s the vertically integrated mass-profile
distance is 6.49%, also independent of horizontal orientation.

Column liquid depth integrates density vertically. It is not the highest
surface crossing or renderer output. Its maximum A/B difference at 8 s is
123.6 mm. The common-liquid-weighted RMS vector velocity difference is
0.137 m/s.

![Column volumes](../artifacts/pool-impact-symmetry/ab-analysis/columns.png)

## Symmetry

Both fields lose symmetry despite fixed, exactly symmetric topology. The first
local velocity-limit breach is actually earlier in A; B subsequently develops
larger sustained errors. A single peak or first failure is not a sufficient
quality ranking.

| Test, unchanged earlier tolerance | A first failure | B first failure |
| --- | ---: | ---: |
| Density maximum > 0.01 | 1.833 s | 1.400 s |
| Density domain mean > 0.001 | 2.433 s | 2.067 s |
| Velocity maximum > 0.02 m/s | 1.133 s | 1.367 s |
| Velocity domain mean > 0.001 m/s | 2.000 s | 1.667 s |

Worst of x reflection, z reflection, and x/z swap at 8 s:

| Metric | A | B |
| --- | ---: | ---: |
| Density maximum | 0.2522 | 0.3200 |
| Density domain mean | 0.00610 | 0.01391 |
| Velocity maximum | 0.379 m/s | 0.684 m/s |
| Velocity domain mean | 0.01068 m/s | 0.04705 m/s |

Symmetry cannot explain the initial A/B physics disagreement: that disagreement
is already large while both fields remain nearly symmetric. Later symmetry
loss compounds it. The frozen runs rule out topology commits and transfer
between changing rungs as necessary causes.

## Shared density/energy drift

Mass loss at 8 s is only 0.0277% in A and 0.0198% in B. However, the liquid
centre of mass rises from 0.20599 m to 0.23244 m in A and 0.23731 m in B.
The pool retains increasing mass in partially filled upper layers.

Using accepted density and collocated velocity, compute
`K = sum(0.5 * water_density * h^3 * rho * |u|^2)` and
`U = sum(water_density * h^3 * rho * g * y)`.
Subtract the same flat-rest reference potential from both arms. The initial
mechanical excess is 41.39 J; at 4 s it is 225.29 J / 232.42 J, and at 8 s
310.32 J / 368.11 J (A/B).

These are diagnostic energies of the represented fields, not an exact
staggered-face energy theorem. Time staggering, coarse unresolved motion,
and interface representation affect their accuracy. Nevertheless, the large
rise in density-based potential also occurs in all-fine A; it is not explained
by B's missing subcell kinetic energy. The setup has zero viscosity and surface
tension, static boundaries, and no continuing source. Mass conservation alone
is therefore a very weak physics acceptance criterion here. The paper's
numerical volume correction is a possible source of artificial work; the
paper does not promise energy conservation.

Read-only scalar stage captures locate where the mass displacement occurs:

| Arm / step | Transport ΔU | Gamma diffusion ΔU | Sharpening/publication-stage ΔU | Whole-step Δ(K+U) |
| --- | ---: | ---: | ---: | ---: |
| A / 15 | +9.408 J | +0.001 J | −0.142 J | +4.666 J |
| B / 15 | +5.556 J | +0.103 J | +0.118 J | +2.591 J |
| A / 30 | +3.168 J | +0.016 J | −0.369 J | +3.139 J |
| B / 30 | +4.092 J | +0.316 J | −0.254 J | +1.736 J |
| A / 120 | +1.618 J | −0.005 J | −0.491 J | +0.992 J |
| B / 120 | +1.379 J | +0.074 J | −0.404 J | +1.175 J |

The sharpening stage includes its existing capacity handling, not just the
sharpening kernel. Transport supplies the dominant positive potential change
at these sampled steps. This does **not** yet distinguish incorrect transport
weights from the velocity provided by pressure/extension. It does argue
against blaming sharpening alone. Audit reruns reproduce the original density
fields bit-for-bit through 4 s.

## What the papers clarify

[Chentanez–Müller 2012, §3.4](papers/massConservingLiquids.txt) explicitly
separates mass conservation (advection column sums) from avoiding compressible
artifacts (row sums/cumulative gamma). Its uniform-grid derivation cannot be
ported by ignoring physical cell volumes. The sparse implementation's
volume-weighted beta and deficit factors should be judged against those two
invariants, rather than against its exact GPU pass sequence.

Gamma captures do **not** show a simple runaway gamma explanation. At 4 s,
mass-weighted `abs(gamma-1)` is 0.00306 in A and 0.00394 in B. Across wet cells,
gamma spans 0.9587–1.0977 in A and 0.9132–1.1401 in B. Gamma is relatively
well-conditioned while the density distribution still spreads substantially.
The paper's §5 specifically warns that regions slightly above rho = 0.5 can
expand while preserving mass, since sharpening does not modify those regions.
That is a relevant conceptual failure mode, not evidence that the current
sparse implementation is fully equivalent to the paper.

Three concrete local-width dependencies deserve an audit:

1. **Sharpening reach:** §3.5 uses `D*dx` and notes that increasing D resembles
   increased surface tension. `traceSharpeningMass` uses local source width;
   D = 2.1 therefore means 0.105 m on A's cells and up to 0.210 m on B's coarse
   cells. Equal UI values do not mean equal physical redistribution reach.
2. **Gamma diffusion:** for uniform width-w neighbors, `scatterGammaRow`'s
   conducted volume divided by cell volume is independent of w. The gamma
   averaging coefficient per step stays constant as physical spacing doubles;
   the corresponding smooth-field diffusion length-scale squared changes by
   four. This is a resolution-dependent numerical filter, not molecular
   viscosity. Its observed direct potential contribution is also larger in B
   at several sampled steps.
3. **Volume correction:** §3.7 intentionally adds a one-sided excess-density
   divergence proportional to `1/dx`. The implementation passes the local
   physical cell size. At equal excess density and before the timestep cap,
   width-two cells receive half the divergence of width-one cells. This is
   another change in the numerical response, even with identical parameters.

These are verified parameter/scaling differences, **not yet an attribution of
all measured error**. They should not be repaired by independently tuning B's
scene controls: B contains both cell sizes, and that would hide inconsistent
physical behavior between them.

[Ando–Batty 2020, §§4–5](papers/ando-batty-2020-octree-liquid.txt) provides
useful sparse-grid consistency targets:

- The mixed-face gradient must reproduce affine pressure fields; its dual
  face volume and divergence transpose must use compatible geometry.
- Free-surface pressure boundaries at T-junctions require consistent distance
  and gradient treatment, not an arbitrary incident child.
- Their velocity interpolation retains staggered components, reproduces
  trilinear interpolation in uniform regions, and avoids temporary
  cell-centred velocity conversions that can introduce diffusion.

This does not imply blindly replacing conservative transport with MLS: an
interpolant alone does not supply CM12's volume-weighted conservation. The
useful connection is a set of invariants for interpolation and projection,
while preserving the sparse architecture and mass transport contract.

The next implementation target should be **resolution-consistent mass transport
and its coupling to projected/extended velocity**, starting at steps 9–15,
where the A/B separates before symmetry does. Audit the three numerical length/
rate choices above alongside affine-flow and hydrostatic mixed-face tests.
Continue tracking later symmetry, but do not use symmetry alone to certify the
physics or add reflection averaging to hide its loss.

## Reproduction and artifacts

Run each arm serially under the existing GPU lease:

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal \
POOL_SYMMETRY_FREEZE_TOPOLOGY=1 POOL_SYMMETRY_MAX_CELL=1 \
POOL_SYMMETRY_STEPS=240 \
POOL_SYMMETRY_OVERRIDES='{"pressureIterations":128,"pressureRelativeTolerance":0.001}' \
POOL_SYMMETRY_OUTPUT=artifacts/pool-impact-symmetry/ab-frozen-max1 \
node --import tsx tools/probe-pool-impact-symmetry-dawn.ts
```

For B, change the maximum to `2` and output to `ab-frozen-min1-max2`.
Full resolved values and scenes are saved in each configuration file. For
future default changes, replay those saved values as the override object.

`tools/analyze-pool-impact-resolution-ab.py A B OUTPUT [AUDIT_A AUDIT_B]`
requires NumPy and Matplotlib. It verifies matching inputs and frozen rosters,
computes physical metrics and the restriction comparison, and writes
`analysis.json`, `stage-energy.json`, `comparison.png`, and `columns.png`.
The optional audit arguments are a pair of capture directories.

Captures are under `artifacts/pool-impact-symmetry/`:
`ab-frozen-max1`, `ab-frozen-min1-max2`, `ab-audit-max1`, `ab-audit-max2`,
`ab-gamma-max1`, `ab-gamma-max2`, and `ab-analysis`.
Gamma audits use `POOL_SYMMETRY_AUDIT_STEPS=1,8,15,30,60,120`, 120 steps,
and otherwise identical configurations.

All captures remained finite and passed the simulation health assertions.
The analysis verifies bit-identical initial fields, matching method values and
shader hashes, fixed topology, legal widths, and matching audited final fields.
Neither arm passes the previous full-trajectory symmetry limits. No production
solver changes were made in this phase, so the large-change Dawn regression
gate was not rerun. The earlier failing gate remains unresolved.
