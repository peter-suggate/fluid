# Mini32 dam break, whole-domain min4/max4: energy investigation

The authored `minimal-power-dam-break-32` scene was run with coarse-first,
whole-domain `minimumCellSize_cells=4`, `maximumCellSize_cells=4`, and the
normal 1/30 s timestep. This is the editor's Min 4³ / Max 4³ setting:
32³ finest lattice, h=0.025 m, 0.1 m physical cells, 64 bricks with 2³
cells apiece (512 cells). All measured frames retain generation 1 and the same
resolution. No topology transfer or renderer is needed to reproduce the effect.

## Measurement

The probe copies native surface density and staggered face velocities at eight
stage boundaries. It computes surface-mass-weighted gravitational and kinetic
energy diagnostics, using both face-based and collocated velocities. The face-based quantity averages squared incident normal face
velocities; collocated energy squares their averages. Native mass, potential,
and collocated kinetic agree with the independently read dense diagnostic.
Gamma diffusion's output is read from its actual scratch plane, not its input.
A separate gravity-work decomposition separates the linear work term from the
quadratic velocity-increment term. Values below are energy divided by the
constant material density, in m⁵/s²; multiply by 1000 for joules if modeling
water at 1000 kg/m³. They are not renderer-derived measurements. CM12 distinguishes surface density
rho from constant material density d: these are useful energy diagnostics of
the conserved surface-mass field, not an independent exact integration over the
reconstructed physical liquid domain. The directly measured velocity discontinuity
does not depend on interpreting that energy diagnostic as continuum energy.

Initial energy is 0.667080. Face-based energy peaks at 0.710344 at 0.2 s
(+6.49%); collocated energy also rises, peaking at 0.701328 (+5.13%).
At 3 s the face-based total is 0.422241, so the overall trajectory is dissipative,
with repeated upward increments in this discrete energy diagnostic superimposed. Baseline liquid-volume
drift at 3 s is -0.0294%, not mass creation.

## Confirmed source of diagnostic energy increments: gravity/transport splitting

`traceEffectiveTransportCharacteristicMode` integrates a characteristic through
the existing effective velocity field. Its RK2 samples improve spatial tracing
inside that frozen field; they do not time-center gravity.
`conservative-transport` precedes `body-forces`. `forceFaces` subsequently adds
`dt * acceleration` to the face velocity without moving the mass at that stage.

On frame 1, conservative transport, gamma diffusion, and sharpening leave the
potential energy unchanged. Gravity adds 0.00892248 kinetic energy; projection
removes 0.00180212. The remaining +0.00712036 has no corresponding fall in
potential energy. This exposes the unmatched acceleration update in the measured discrete energy budget.
For free fall, old-velocity position transport followed by a full velocity kick
has the familiar positive defect 0.5 m g² dt². Boundaries and projection change
the retained coefficient, but the measured first-frame defect scales exactly
by four each time dt is halved:

| dt | First-frame retained energy increase |
|---|---:|
| 1/30 s | 0.0071203561 |
| 1/60 s | 0.0017800890 |
| 1/120 s | 0.0004450223 |

At frame 85 (2.833333 s), gravity adds 0.01100322, comprising 0.00230336
linear work and 0.00869987 quadratic work. Potential energy barely changes
across conservative transport (-0.00001867). The full-frame native total rises
0.00348144, approximately 0.807%. This is a coupled transport/force energy
balance defect, not evidence that adding gravity is itself wrong.

## Confirmed abrupt wall response: density decides pressure-domain membership

`pressureCellMembershipFromDensity` admits a cell at rho >= 0.5, or if
`pressureCellSubmerged` retains it. That submerged-cell guard immediately rejects
any one-term incidence row: `if(count<2u){return false;}`. A one-term row can be
a sealed floor/wall, as well as an open air boundary. Consequently, bottom-edge
cells surrounded by liquid lose pressure membership when transport carries their
conservative density just below 0.5. `classifyPressureCell` then zeros their
pressure; neighboring projected faces see an artificial interior p=0 boundary.

For example, stable cell 16, center (18,2,2) in finest-lattice coordinates:

| State | Density | Pressure member | Speed (m/s) |
|---|---:|---:|---:|
| Start of frame 85 | 0.50099647 | yes | 0.098867 |
| End of frame 85 | 0.49918944 | no | 0.491902 |
| End of frame 86 | 0.59770805 | yes | 0.132181 |

Its four physical neighboring cells were all pressure members at the start of
frame 85. The remaining two faces meet the closed floor and wall. Cells 9,
132, 256, and 405 show the same pressure loss and local kick. These are actual
stored velocities and actual membership readbacks, not a surface-display jump.

Global pressure projection still removes kinetic energy on every baseline frame.
The wall defect creates local kicks and changes the amount/distribution of
energy removed; it is incorrect to describe the projection stage as globally
adding energy in this reproduction.

## Direct epsilon test: a finite pressure response to a 2e-7 density change

Two native runs were identical through frame 84 and through frame 85's
surface-sharpening stage. An encoder-ordered copy then changed only stable
cell 16's destination density. No velocity, gamma, topology, or solver
parameter changed. All cell velocities remain identical through body forces.
The only pre-pressure difference is one scalar and its negligible mass/energy.

| Perturbed density (actual f32) | Pressure membership | Final speed |
|---|---|---:|
| 0.499999910593 | excluded | 0.488451 m/s |
| 0.500000119209 | included | 1.039026 m/s |

A density difference of **2.08616e-7** produces a **0.657668 m/s vector velocity
change**. Before pressure, the whole-domain kinetic-energy difference is only
4.83e-12 and potential-energy difference 1.02e-10. After pressure, the total
energy difference is 0.000454007. This directly confirms the unacceptable
pressure discontinuity, rather than merely correlating a density crossing with
a different point in a wave. In this particular neighboring-hole arrangement,
the just-above-threshold side is actually faster: the defect is the abrupt
pressure operator change, not a universal claim that dropping membership always
increases speed.

The mechanism includes both `pressureCellMembershipFromDensity` and
`classifyPressureRow`: the latter derives ghost-fluid phi from `0.5-rho`,
changes row participation with the membership, and clamps theta to a positive
minimum. `projectPressureRow` then applies a different pressure jump to the
same pre-pressure velocity field. This epsilon experiment happens within one
pressure solve, before the next velocity-extension pass, so extrapolation is
not needed to cause the demonstrated discontinuity.

Run with `--arm=threshold-minus` and `--arm=threshold-plus`. The analysis checks
identical checkpoint histories, identical early stage captures, identical
pre-pressure velocities, and exactly one perturbed density. These tests are
diagnostic perturbations, not mass-conserving production updates.

## The 0.5 discontinuity must not be accepted as a fixable threshold value

The same density switch also determines velocity-extension seeds,
`publishSparseCM12FaceVelocitySupport` wet flags, native-face donor selection,
and publication of collocated wet effective velocity. A sub-isovalue cell can
retain mass and forced face velocities while its transport velocity is rebuilt
from neighboring wet cells. Around 3 s, approximately 22% of the liquid volume
is below 0.5 and carries about 55% of the stored face kinetic energy. Thus this
is not a negligible numerical dry tail.

Moving 0.5 or adding hysteresis would move/delay the discontinuity. Correctness
requires that a small change of liquid amount not turn a submerged cell into a
finite pressure-release cavity or discontinuously replace the momentum that
transports its mass. The wall guard is a diagnostic of one concrete failure,
not a complete continuous interface formulation. Pressure support, geometric
interface reconstruction, transport velocity, and occupied volume need a
consistent contract. CM12 does explicitly use the isovalue to define its pressure
domain, so the existence of this threshold is not by itself a deviation from the
paper; the measured finite velocity response is the failure to address. A genuine free-surface topology change must also approach its limiting
pressure/velocity response continuously.

## Controlled runs

The sum of positive frame increments between 2 and 3 s is reported separately
from net change; it must not be mistaken for the net energy gained in that second.

| Run | Positive increments, 2–3 s | Largest increment |
|---|---:|---:|
| Baseline | 0.0183374 | 0.00348144 |
| Gamma diffusion and sharpening off | 0.0184739 | 0.00256280 |
| Pressure tolerance 1e-6, budget 256 | 0.0164508 | 0.00237411 |
| dt=1/60 | 0.00821143 | 0.00129157 |
| dt=1/120 | 0.0121742 | 0.00061675 |

Tighter pressure does not remove the defect. Conditioning-off retains the
scalar publication/capacity chain but removes the two selected physical
conditioning controls. Neither switch eliminates the increases. Smaller steps
reduce the largest increment and the first-frame gravity error, but late
accumulated increases are not monotonically convergent: density/phase switches
and spatial transport errors remain. A timestep reduction is not a full fix.

## Reproduction and artifacts

Run one process at a time with the fluid browser unloaded:

```sh
node --import tsx tools/probe-mini32-fixed4-energy-dawn.ts --arm=base --wait
node --import tsx tools/probe-mini32-fixed4-energy-dawn.ts --arm=conditioning-off --wait
node --import tsx tools/probe-mini32-fixed4-energy-dawn.ts --arm=tight-pressure --wait
node --import tsx tools/probe-mini32-fixed4-energy-dawn.ts --arm=half-dt --wait
node --import tsx tools/probe-mini32-fixed4-energy-dawn.ts --arm=quarter-dt --wait
python3 tools/analyze-mini32-fixed4-energy.py
```

`artifacts/mini32-fixed4-energy/*/trace.json` holds the complete stage budgets.
`boundary-audit` additionally captures open fractions, pressure membership, and
per-cell velocities around frames 80–90. `summary.json` is machine-readable.
The analysis adds probe/report files only; production physics has not been changed.
The repository-wide TypeScript check reports existing errors outside these
probe files. The canonical Dawn post-refactor gate is not claimed as passed;
no production refactor was made in this investigation.

## Limits of the diagnostic wall correction

A probe-only shader substitution that ignores closed one-term rows was prepared
to test the submerged-cell guard. Its first checkpoint condition used an
incorrect absolute frame-control address and was inactive; that trace is marked
`invalidExperiment` and excluded from the summary. The corrected attempt was
blocked at compilation by an unrelated duplicate `presentationOwnerCellAt`
declaration appearing in concurrent presentation edits. No successful corrected
wall run, production fix, or global continuity fix is claimed. The completed
epsilon pair independently proves the pressure discontinuity without that
substitution.

## Reading the CM12 paper: important qualifications and implementation targets

Source: `docs/papers/massConservingLiquids.txt`.

1. **Surface density is not material density.** The introduction (lines 35–42)
   distinguishes rho, the conservative surface-tracking field, from constant
   fluid density d. That distinction limits the interpretation of a rho-weighted
   energy diagnostic. Conservation of the integral of rho is not a proof of
   mechanical-energy conservation.
2. **The 0.5 domain boundary is explicit.** Section 3 (lines 141–149) solves
   the fluid equations in rho > 0.5. The paper therefore does not justify
   rejecting every use of that threshold. It also gives no continuity theorem
   that would rule out the epsilon-test response measured here. Meeting the
   requested smooth-response requirement needs its own validation.
3. **Solid fraction and liquid fraction must be distinguished.** Section 3.6
   defines V as the non-solid volume fraction, including fluid and air.
   Section 3.7 / Eq. 20 (lines 383–415) uses rho'=rho/V and explicitly warns
   that raw rho can misclassify liquid in partially solid cells as air.
   It extrapolates rho' into adjacent solid cells and derives ghost-fluid phi
   from -(rho'-0.5) dx, then delegates coefficient/boundary construction to
   the variational and separating-boundary pressure methods. In the sampled
   failing cells our readback has V=1, so merely dividing by V again will not
   fix this specific discontinuity. The closed-wall vs open-air distinction
   and actual pressure coefficients remain the relevant implementation audit.
4. **Sub-isovalue material is intentional.** Section 3.8 (lines 461–496) says
   0<rho<0.5 can encode unresolved droplets/sheets, as well as an interface's
   air-side tail. These values cannot all be interpreted as absence of liquid.
   The proposed postprocessing is for rendering; it is not a pressure-continuity
   fix and must not be silently fed back into the solver.
5. **Mass conservation is the promised property.** Algorithm 1 (lines 176–180)
   places density advection before velocity advection/forces and projection,
   matching the relevant split-force concern. Section 3.7's artificial
   excess-volume divergence is another reason not to assume a mechanically
   energy-conserving formulation. The paper does not provide a ready-made
   energy-monotonic or threshold-continuous update. Its stated large-step
   stability concerns should not be read as such a guarantee.

Consequently, the epsilon test is the strongest direct failure witness. A
correction must preserve CM12's mass accounting while making the pressure and
velocity response continuous in the tested limit. Moving the threshold,
adding hysteresis, or applying rendering postprocessing does not demonstrate
that requirement.
