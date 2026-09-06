# Uniform versus max1: half-pool impact, 0–3 seconds

This comparison uses the **Uniform method**, not an adaptive solver called
"uniform", against Sparse CM12 / adaptive-mass with coarse-first selected and a
whole-domain min1/max1 enforcement region. Both run the authored
`coarse-first-pool-impact-half` scene on a 64×48×64 lattice, h=0.05 m, with the
scene timestep 1/60 s. Both start with bit-identical density and velocity.
Their normal method controls are retained except for explicitly matching the
timestep. Initial liquid volume is 8.257625 m³.

## Reproduction and measurement

Run each Dawn process under the repository GPU lease, with the browser unloaded.

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  POOL_OUTPUT=artifacts/pool-impact-ab/uniform-center \
  node --import tsx tools/probe-uniform-pool-impact-dawn.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  POOL_MAX_CELL=1 POOL_STEPS=180 POOL_AUDIT_FAILURES=1 \
  POOL_OUTPUT=artifacts/pool-impact-ab/max1-center \
  node --import tsx tools/probe-pool-impact-ab-dawn.ts
uv run --with numpy --with scipy --with matplotlib python \
  tools/analyze-uniform-max1-center.py uniform-center max1-center
```

The probes save accepted density, pressure, gamma, and velocity every five steps
(83.33 ms), plus initial state. The Uniform probe also saves native MAC faces and
selected existing stage diagnostic textures. Its cell velocity is the average
of the paired normal MAC faces, in m/s, to match the sparse cell diagnostic.
The separately stored negative boundary planes are read explicitly: the
separating-boundary solve can produce nonzero speeds even in an enclosed scene.
Their maximum absolute speed is saved alongside each Uniform snapshot.

The center gauge averages r<0.15 m. The near gauge averages the annulus
0.35≤r<0.55 m. Pressure averages the central y=0.3–0.6 m band, with zero pressure
outside each method's pressure phase. Vertical velocity is density weighted in
the central y=0.4–1.2 m band. Reported kinetic energy is divided by constant water
density, in m⁵/s². These are sampled diagnostics, not fitted continuum modes.

Two independent elevation measures are retained:

- Column mass divided by horizontal area, minus the 0.8 m resting waterline.
  This includes any airborne liquid above the gauge and diffuse/sub-isovalue
  density. It does not depend on a surface renderer or an isovalue.
- The upper rho=.5 boundary of the **three-dimensional floor-connected liquid**.
  Detached drops are excluded. Internal air pockets do not replace the free
  surface with their lower wall. A first crossing in each vertical column gives
  a false late central depression when bubbles appear.

The report separately records enclosed sub-isovalue volume and central
sub-isovalue volume below the upper waterline. It therefore does not hide
internal phase changes inside a surface-height statistic.

## Baseline numerical differences

| Component | Uniform | Adaptive-mass max1 |
|---|---|---|
| Density characteristics | RK2; 1–16 equal substeps from initial speed/h | RK2; 1–16 equal substeps from initial speed/h, collocated sparse transport field |
| Face characteristics | RK2; up to 32 adaptive segments, ≤1.5 cells per segment | RK2; 1–16 equal substeps, native staggered donor values with local fallback |
| Gamma conditioning | Full x, then y, then z axis-Jacobi sweeps per step | One simultaneous six-neighbor snapshot, scale min(1,30dt)/3 |
| Gamma at rho<1e-5 | Reset to one | Retain the previous operator state |
| Sharpening | Strength1, distance2.1 cells; local mass return | Strength1, distance2.1 cells, up to7 half-cell trace steps; conservative receipts and capacity repair |
| Pressure | CM11a LCP multigrid, 3 full + 4 V cycles, 6/6 sweeps | MGPCG, maximum128 iterations, relative tolerance0.001 |
| Negative liquid pressure | Allowed; zero lower bound on solid/closed halo rows | Allowed in the liquid solve; separating boundary treatment is separate |
| Postprojection thin-liquid faces | Cleared when neither neighbor is in the rho>.5 pressure phase | Retained for sub-isovalue liquid transport support |
| Scalar scatter scale | 1,048,576 | 65,536, with separate physical-volume receipt scaling |

Both gamma controls display "On", but they do not specify the same operator.
At dt=1/60, the sparse per-face coefficient is one sixth of a Uniform axis
coefficient. This is a formulation difference, not evidence that one default
should be tuned to match the other. The matched gamma-off ablation isolates its
contribution to the central response.

### Energy-readback caveat

The falling drop's center of mass at t=0.25 s is 1.5392835 m in Uniform and
1.5391614 m in max1, a difference of 0.122 mm. Yet their postprojection
density-weighted mean drop velocities differ (-2.018 versus -2.451 m/s), because
Uniform clears sub-isovalue faces and reconstructs their extension at the next
step. A raw energy comparison that counts those stored zeros as physical drag
is misleading. The reports retain core-liquid energy as a second diagnostic;
column mass and drop trajectory remain independent of this storage distinction.

## Demonstrated scalar bug and bounded fix

`preserveHorizontalD4` rounded every density and gamma to1/65536 before averaging
its symmetry orbit. An already exactly symmetric density of1e-6 became zero in
one otherwise no-op pass. This was unnecessary: all orbit members can enumerate
the same coordinates in the same floating-point order.

The fix folds x/z into a common D4 octant before summing. It preserves an
already-identical scalar exactly and averages other orbits in deterministic
float order. It does not change topology, coarseness, or any diffusion dose.
The production WGSL regression covers sub-quantum identity, exact D4 symmetry,
mass and gamma sums within roundoff, reflection axes, diagonal degeneracies,
and inactive support. The saved original function fails; the corrected one
passes:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  node --import tsx --test tests/sparse-cm12-scalar-d4-conservation-dawn.test.ts
```

The 180-step max1 rerun has effectively the same central response. This fix is
**not** claimed as the main wave-damping explanation. The optional
`POOL_SCALAR_AUDIT=1` snapshots production stage outputs without inserting any
simulation kernel. Its corrected run shows about -0.00187 m³ across transport,
+0.000075 m³ across sharpening/capacity conditioning, and only +8.9e-8 m³ across
D4 averaging. The larger transport receipt precision question is separate from
the visible central wave differences; no global fixed-point range was raised.

## Artifacts

- `artifacts/pool-impact-ab/uniform-center`: dense baseline.
- `artifacts/pool-impact-ab/max1-center`: adaptive max1 baseline before scalar fix.
- `artifacts/pool-impact-ab/max1-center-conservative-d4`: scalar fix, all-step budgets.
- `artifacts/pool-impact-ab/uniform-max1-center-report`: central plots, density
  slices, complete metrics and initial-state identity receipts.

The combined Sparse CM12 canonical regression gate is coordinated by the parent
task after the presentation and topology work settles; this note does not claim
that gate passed solely because the focused scalar regression passed.


## Central response and gamma ablation

The centre averages a radius of 0.15 m. The connected-liquid surface values
below are changes from the resting 0.8 m waterline. All runs reach 3 seconds;
peaks are sampled every five steps, so timing resolution is 0.08333 seconds.

| Arm | Initial central depression | Rebound peak | Rebound time | Centre peak-to-trough, 1–3 s |
| --- | ---: | ---: | ---: | ---: |
| Uniform, normal gamma | −313.8 mm | +249.5 mm | 1.167 s | 249.0 mm |
| Uniform, gamma off | −328.2 mm | +293.5 mm | 1.167 s | 337.0 mm |
| Sparse max1, normal gamma | −374.2 mm | +386.5 mm | 1.250 s | 464.1 mm |
| Sparse max1, gamma off | −380.9 mm | +384.8 mm | 1.250 s | 461.8 mm |

Turning gamma off raises the Uniform rebound by 44.0 mm but changes sparse's
by only −1.7 mm. Thus the different conditioning operators explain part of the
comparison, but not the whole stronger/later sparse rebound. This does not
establish that Uniform is physically better: no sparse diffusion parameter or
operator was changed to imitate it. The existing sparse stability and feature
behavior are retained.

At 3 s, the central column-mass deficits are close (about 162 mm for Uniform
and 168 mm for sparse), while enclosed sub-isovalue air volume is 0.063 L
versus 6.62 L. With gamma disabled those become 0.960 L and 8.76 L. This
identifies actual density-distribution and phase-boundary differences rather
than a missing bulk volume at the centre. The late free-surface excursion is
not by itself a calibrated measure of energy loss.

The quantified numerical defect fixed by this subtask is the unnecessary D4
scalar quantization. The remaining response differences are localized and
measured, but are not claimed as solved numerical errors. A pressure/phase and
conditioning comparison on identical intermediate states would be the next
causal experiment; copying the Uniform formulation is not warranted by this
A/B alone.

The refreshed four-arm plots and complete metrics are in
`artifacts/pool-impact-ab/uniform-max1-final-report/`. Both Gamma On/Off controls
retain bit-identical initial density and velocity; all captured values are
finite. The Uniform pressure-cap flags are false at all saved checkpoints.
