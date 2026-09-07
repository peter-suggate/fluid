# Adaptive-mass: mechanisms and repair plan

The subsequent implementation and validation are recorded in
[Geometric transport and physical-face repair](adaptive-mass-geometric-transport-validation-2026-09-07.md).

The next change should replace the mixed-resolution transport interpolation.
There is now a direct shader reproduction of incorrect geometry. A second
structural problem is the use of pressure-row membership as physical adjacency
in velocity extension and excess-density repair. Both can alter the physics
while preserving reflection symmetry.

Scope remains frozen topology, A = max1 and B = min1/max2, in
`coarse-first-pool-impact-quarter`. The same fixes must work on arbitrary scenes
and supported sparse layouts. No reflected averaging, symmetry passes, or
per-scene parameter changes are proposed. This phase adds a diagnostic and this
plan; it does not change the production solver.

The [A/B report](quarter-pool-max1-max2-ab-2026-09-07.md) contains the trajectory
measurements. At 0.5 s, B has 41% more kinetic energy while both arms still have
very small symmetry errors. At 8 s, kinetic energy is 2.18 J / 10.55 J, and the
column-depth RMS difference is 41.9 mm. A also develops substantial upward
density redistribution. A is a reference discretization, not physical ground
truth. Frozen topology rules out topology transfers as a necessary cause.

## 1. Demonstrated transport geometry defect

[`transportSourceSamplingSpans`](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts)
chooses the finest width in a coarse source's nearby support. Then
`effectiveTransportStencilAtSpansMode` constructs regular-grid interpolation
knots at that spacing, resolves each knot to its actual owner, and keeps the
weights calculated for the virtual knot locations.

The problem is that a coarse owner's centre is not the virtual knot location.
For a width-two cell covering x = [6, 8], the centre is 7. With sampling spacing
one, knots at 6.5 and 7.5 both resolve to that same cell. A small departure on
either side of 7 therefore returns exactly the original coarse value. On the
fine side, the coarse value is instead treated as if it were located at 7.5,
although its centre is 7. This gives the wrong interpolation slope.

The new [Dawn diagnostic](../tools/probe-cm12-transport-consistency-dawn.ts)
extracts and executes those **production WGSL functions** on a symmetric 2:1
layout. It samples the cell means of the affine field f(x, y, z) = x:

| Source x | Departure displacement | Exact affine value | Shader value |
| ---: | ---: | ---: | ---: |
| 3, uniform coarse interior | +0.1 | 3.1 | 3.0999999 |
| 7, coarse beside fine | −0.1 | 6.9 | 7.0 |
| 7, coarse beside fine | +0.1 | 7.1 | 7.0 |
| 8.5, fine beside coarse | −0.1 | 8.4 | 8.3499994 |

All units here are finest-cell units. The mirrored seam exhibits the mirrored
error; stencil weights still sum to one. Thus a symmetry test and a constant
interpolation test both miss this defect. The existing affine translation test
covers uniform coarse support and misses the mixed interface.

The earlier symmetry fix introduced the coarse-side dead zone by selecting the
fine support spacing. That fix needs replacement. Reverting to the previous
arbitrary fine-child selection would restore its original reflection defect.
The fine-side owner/virtual-position mismatch also needs correction.

This stencil feeds backward mass gathering, forward deficit redistribution,
and sharpening destinations; its spacing helper is also used in transport
velocity sampling. Volume-weighted beta can conserve the total mass of a
geometrically incorrect operator. It does not repair its physical displacement.
The probe demonstrates the interpolation defect; it does not yet quantify its
share of the pool's full A/B discrepancy.

**Repair:** construct bounded, nonnegative weights using actual sparse geometry,
with partition of unity and affine moment reproduction in open interior
support. Treat stored densities as control-volume means. Prove the construction
in a small CPU oracle before changing the GPU stencil representation. Preserve
uniform-grid trilinear behaviour and the existing volume-weighted beta/deficit
conservation contract. Apply the same construction to both transport directions
and the normal and packed-coarse paths. Audit sharpening and velocity consumers
explicitly; native MAC face locations must remain native face locations.

The exact local support construction is still a design decision. Eight virtual
knots are not a requirement worth preserving if they prevent a positive,
geometrically consistent stencil. A bounded local support change is acceptable;
adding global correction passes is not the proposed solution. Conversely,
naively reconstructing virtual fine values can change the effective donor
coefficients and break CM12's conservation or positivity, so reconstruction
must be checked at the complete operator level.

## 2. Demonstrated misuse of pressure connectivity

A mixed pressure row connects a coarse cell and four fine cells across one
coarse face. It is a pressure-gradient equation, not a declaration that every
pair of its cells shares a physical face.

In [`advanceVelocityExtensionPackets`](../lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts),
the multi-term path visits every other term and weights it by the absolute
pressure coefficient. It never checks whether the candidate is on the opposite
side of the row. A fine cell consequently samples its same-side siblings,
including a diagonal sibling with which it shares no face. These connections
also shorten the graph distance used by the eight extension sweeps. Even
legitimate tangential neighbours are already represented by their own rows.

In `scatterDensityCapacityRepairCellAtPlane`, the same all-other-terms walk
counts recipients and splits excess mass equally by that count. It neither
uses subface area/distance nor selects recipients by available capacity. For
a coarse cell with five ordinary coarse neighbours and four fine neighbours
across its sixth face, that sixth face receives 4/9 of the excess; each other
face receives 1/9. Merely subdividing an equal-area face changes the routing.
For a fine source, the walk also sends mass to same-side siblings through the
mixed pressure row.

This relay runs eight times, including when surface sharpening is disabled.
It conserves each fixed-point debit/credit, but modifies density without
transporting gamma or native face momentum. It can therefore change mass
location, potential energy, and subsequent pressure membership. Exact global
mass is not a guarantee of equivalent physics.

**Repair:** define a shared physical-subface contract: opposite-sign terms,
actual open area, and centre/face distances. `sharpeningStats` already uses an
opposite-sign expansion and is a useful starting point; it is not permission
to assume every future cut-face partition has equal subareas. Velocity
extension should use geometry-based weights consistent with its uniform
stencil. Capacity routing must be invariant to splitting a face into equal
subfaces. Retain exact paired receipts and symmetry-neutral integer residue.
Cover scatter, gather, and alternate-receipt implementations.

After fixing upstream transport, reassess the need for bulk capacity relaying.
The current guard was introduced after severe overfill; deleting it first
would reintroduce an established failure. CM12 uses pressure for bulk excess
volume and a separate treatment near solids. That distinction is a better
target than indefinite bulk redistribution through the pressure graph.

The earlier scalar capture combines sharpening with capacity repair. Its net
potential change is often negative; this does **not** exonerate either substep
individually. Expose read-only captures at the existing `sharpening-finalize`
and `density-capacity-repair` seams to measure their separate contributions.
No additional simulation operation is needed.

## 3. Density, velocity support, and pressure form a feedback loop

The relevant execution order in
[`webgpu-sparse-cm12-resident.ts`](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts)
is:

1. Seed/extend cell transport velocity and advect native face velocity.
2. Conservatively transport density/gamma; diffuse gamma; sharpen and repair
   density capacity.
3. Classify pressure cells and faces from the resulting density, add gravity
   and excess-volume correction, and project native face velocity.
4. Collocate projected faces into cell velocity for the next step.

`gatherConservativeDensity` calculates a momentum-derived cell velocity.
`collocateAndDiagnose` later overwrites it with the projected native-face
result. Scalar traces and face advection therefore have distinct velocity
representations and histories. The conservative momentum intermediate does
not establish conservation of the authoritative velocity update. Separate
velocity advection also occurs in CM12; the existence of two representations
is an audit target, not proof that the paper was implemented incorrectly.

There is a concrete membership mismatch to examine: extension seeds and
collocated effective-velocity publication use rho > 0.5, whereas pressure
membership uses rho >= 0.5 **or retained submerged membership**. An underfilled
interior cell can remain in pressure while losing its velocity seed. Pressure
face classification additionally derives interface distance from density and
local width, then uses a clamped ghost-fluid fraction in projection.

The earlier frozen investigation captured reflected densities of approximately
0.500107 and 0.499954 becoming different pressure membership and a much larger
velocity difference. That was a different coarse configuration, not the first
failure checkpoint of this A/B. It demonstrates a threshold amplification
mechanism; its contribution in A and B still needs a short, targeted replay.

**Repair approach:** first make transport/support geometrically consistent,
then trace the earliest remaining pressure-mask disagreement through its
actual face flux, interface distance, seed validity, and pressure residual.
Define compatible pressure/transport validity for submerged cells. Test a
planar interface moving continuously across fine and coarse centres and
T-junctions, plus mixed-grid hydrostatic equilibrium and affine pressure.
Do not move the threshold or add hysteresis merely to postpone failure.

Use native-face work diagnostics for projection: the stored row dual weight
is area times distance. Check gradient/divergence compatibility and the
weighted projection energy in a fixed-domain, homogeneous-boundary test with
no forcing or excess-volume source. A general moving-interface step with
volume correction is not subject to that simple non-increase test. In the
pool, record gravity, excess-volume pressure work, and density redistribution
separately before assigning responsibility for the rising mechanical energy.

## 4. Resolution changes three numerical length/rate choices

These code dependencies are established. Their individual trajectory impact
is still unmeasured:

| Operation | Current rule | Effect of doubling local width |
| --- | --- | --- |
| Sharpening redistribution | `maximumDistance = D * sourceWidth`; owner-width trace steps | Doubles physical reach: 0.105 m to 0.210 m at D = 2.1 |
| Gamma diffusion | Conducted volume proportional to subface area times local width | Same uniform per-step averaging weight; smooth-field physical diffusivity scales by four |
| Excess-volume pressure source | `min(0.5 * excess, 1) / cellSize`, capped by 1/dt | Halves the response to equal excess before the timestep cap |

The gamma and pressure formulas are in `scatterGammaRow` and
[`cm12VolumeCorrectionDivergence`](../lib/core/cm12-numerics.ts); the source
passes the local physical cell width. These are numerical stabilization rules,
not fluid material properties. Some discretization dependence is expected,
but hidden changes in physical reach and relaxation are undesirable for this
minimal A/B.

**Proposed policy after geometry is fixed:** express redistribution reach in
physical length, gamma smoothing through a geometric conductance proportional
to area/distance, and excess-volume response through a declared relaxation
rate. Calibrate against the existing finest-grid behaviour at the reference
timestep, then use the same policy across all local widths. Derive positivity
and timestep bounds, including the one-half already in the gamma flux, before
implementing it. Preserve the dispatch budget and change one rule at a time.
This is a solver-wide policy, not separate tuning of A and B.

There is also a shared underfill mechanism: sharpening explicitly leaves
rho > 0.5 unchanged, and pressure volume correction only acts on rho > 1.
Transport-smoothed bulk in 0.5 < rho < 1 can persist. Gamma near one does not
prove that the density profile is sharp. CM12 itself identifies expansion of
these regions in its limitations. This explains how mass can remain constant
while represented fluid spreads; it does not yet apportion the observed
potential-energy growth. Adding negative divergence everywhere underfilled
would change the fluid model and is not an acceptable shortcut.

## Paper guidance and limits

[CM12 §§3.4–3.7 and §5](papers/massConservingLiquids.txt) supplies the important
separation between mass conservation, row/gamma conditioning, sharpening, and
bulk excess-volume correction. Its regular-grid transport derivation must be
extended with physical control volumes. Its stated underfill/expansion
limitation remains relevant even if all conservation bookkeeping is correct.

[Ando–Batty §§4–5](papers/ando-batty-2020-octree-liquid.txt) supplies useful
targets for mixed-face pressure geometry, compatible divergence and gradient,
and velocity interpolation at T-junctions. Affine reproduction and reduction
to trilinear interpolation on uniform support are useful invariants here.
Their MLS interpolant is not a drop-in CM12 mass operator: scalar donor
positivity and volume-weighted conservation need their own proof. In
particular, unconstrained MLS weights can be negative.

## Ordered implementation and acceptance

1. **Replace the transport stencil.** Start with the demonstrated seam case;
   extend it to all axes, mirrored layouts, seam edges/corners, zero-time
   identity, and supported cell ratios. Require affine interpolation error
   below 1e-5 in the toy's coordinate range, partition of unity, nonnegative
   weights, and absence of finite displacement dead zones. Then exercise the
   complete beta/deficit operator: exact volume-weighted mass up to known
   receipt quantization and uniform-density translation without artificial
   seam structure in a divergence-free, boundary-compatible manufactured
   case. Analytic affine interpolation and complete conservative transport
   are separate checks.
2. **Correct physical adjacency in extension and capacity handling.** Require
   invariance to an equivalent subface subdivision, no same-side pressure-row
   shortcuts, constant-velocity extension, and exact conservative receipts.
   Capture sharpening and capacity separately. Judge whether bulk repair is
   still required after the first fix; retain protection until its replacement
   handles the original overfill case.
3. **Resolve remaining pressure/support feedback.** Use the first failing
   frozen checkpoint and the manufactured pressure/interface tests above.
   Account for native-face pressure work and scalar redistribution. Fix a
   demonstrated validity, geometry, or projection inconsistency before
   changing stabilization parameters.
4. **Make stabilization units explicit.** Apply the common physical policy
   above one operation at a time, with no per-scene settings and no additional
   averaging or simulation passes. Retain only changes supported by local
   consistency tests and the trajectory comparison.
5. **Re-run frozen A/B through 4 s and 8 s, then resume changing topology.**
   Keep the existing symmetry limits: density max/mean 0.01/0.001 and velocity
   max/mean 0.02/0.001 m/s, plus finite-state and mass checks. Compare the
   impact interval at steps 9–15, column mass profiles, centre of mass, native
   face/cell energy diagnostics, and wet-cell fractions. Exact A/B equality
   is not expected. Remaining differences must be consistent with resolution
   error and identified numerical work rather than unexplained seam effects.

Make each production correction independently reviewable. Run its focused
regressions and the canonical `npm run test:dawn:sparse-cm12` gate after large
simulation changes, serially under the WebGPU lease. Do not relax existing
symmetry lanes or timing limits. The earlier failing full gate still needs
resolution; this diagnostic-only phase does not claim it passes.

## Reproduction and validation of this phase

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal \
node --import tsx tools/probe-cm12-transport-consistency-dawn.ts
```

The probe acquires the repository GPU lease itself. It reports the measurements
and `affineConsistent: false`; it intentionally does not assert that the current
defect is desirable behaviour. Its 25-query output is saved in
[`transport-consistency.json`](../artifacts/pool-impact-symmetry/mechanisms/transport-consistency.json).
Compilation, finite-value/partition assertions, and the GPU error check passed.
ESLint passed for the new tool. Repository type checking reports errors in
other files; it reports none in the new diagnostic. No production fix has been
applied in this phase.
