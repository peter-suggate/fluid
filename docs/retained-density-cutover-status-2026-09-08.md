# Retained density cutover status — 8 September 2026

The retained-field execution and integration components now exist, and their
focused tests pass. **The production Sparse CM12 simulation and renderer have
not been switched to them.** A field representation that preserves mass is
not sufficient: its half-density surface must also preserve the intended
geometry. The first conservative positive reconstruction failed that test.
The clamped-affine alternative passes planar and two-plane crease tests, but
a separate declared quadratic primitive now supplies exact curved geometry
and certified CPU integral enclosures. Neither yet supplies the assembled,
evolving field required for a production cutover.

This report supplements, rather than replaces, the historical
[initial ladder results](implicit-density-ladder-results-2026-09-08.md) and
[implicit-density roadmap](adaptive-mass-implicit-density-roadmap.md). Files
under `lib/` below are implementation components available for integration;
that location does not establish their adoption by a production consumer.

## Implemented components

| Component | Implemented API and responsibility | Current limit |
| --- | --- | --- |
| `lib/methods/adaptive-mass/sparse-cm12-positive-density-field.ts` | `compileBernsteinSupport`, `positiveBernsteinField`, `evaluateBernsteinCell`, `integrateBernstein`, `bernsteinCellMeans`, `refinePositiveBernsteinField`; immutable generations, shared quadratic Bernstein controls and exact de Casteljau refinement | Equal-width aligned conforming support; no general initial-scene reconstruction or mixed-support assembly |
| Same module: `fitPositiveBernsteinMeans` | Nonnegative conservative correction with shared boundary control capacity and a private center correction | Algebraic feasibility only; rejected as a general geometry-preserving reconstruction |
| `lib/methods/adaptive-mass/sparse-cm12-density-native-geometry.ts` | `compileDensityNativeGeometry`; physical native boxes, compact ordinal mapping, face adjacency and optional boundary-stamped open moments | Geometry compiler, not a field constructor or resident topology transaction |
| `lib/methods/adaptive-mass/sparse-cm12-density-support-coupling.ts` | `compileDensitySupportCoupling`, `applyDensitySupportCoupling`, `densitySupportGeometryKey`, `assertDensitySupportCouplingSupport`; sparse CSR intersections and tensor Bernstein box moments | Integrates full native boxes; cannot replace clipped-solid basis integrals with scalar open fractions |
| `lib/methods/adaptive-mass/webgpu-sparse-cm12-retained-density.ts` and `.wgsl.ts` | `WebGPURetainedDensityField.create`, `retain`, `release`, `next`, `compileQueries`, `compileCoupling`; immutable GPU coefficients and leased compiled operations | GPU execution currently implements Bernstein support, not the clamped-affine representation or a complete dynamics update |
| `lib/methods/adaptive-mass/sparse-cm12-retained-affine-density.ts` | `retainedAffineRamp`, `retainedAffineFeature`, `evaluateRetainedAffineDensity`, `integrateRetainedAffineDensity`, `meanRetainedAffineDensity`, `splitRetainedAffineDensity` | CPU algebra for one ramp or min/max of two ramps; no curved branches, GPU path or assembled global support |
| `lib/methods/adaptive-mass/sparse-cm12-retained-quadratic-density.ts` | `retainedQuadraticDensity`, `retainedSphereDensity`, `initializeRetainedDensityPrimitive`, evaluation/gradient functions and `integrateRetainedQuadraticDensity`; immutable ten-coefficient declared geometry and outward-rounded CPU integral enclosures | Reference integration can exhaust its accuracy budget; no GPU or evolving hybrid-field assembly |

GPU query results contain `(q, gradientX, gradientY, gradientZ)` in physical
coordinates. Coupled integral results contain `(nativeMean, physicalAmount,
0, 0)`. Compiled integral operations require matching topology and boundary
generations when encoded. Support identity checks supplement generation
numbers. An operation leases its source coefficient generation, so releasing
the original owner or creating a newer generation does not modify the old
operation. The updated allocation contract requires `await field.ready()` on a new
`next` generation and `await operation.ready()` before encoding a compiled
operation. `create` already awaits readiness. Operations must remain alive
until submitted GPU work completes. The previously reported two Dawn passes
predate this readiness update; its GPU rerun must be recorded separately.

The affine primitive is

```text
q(x) = clamp(0.5 + (offset - dot(unitNormal, x-origin))/transitionWidth, 0, 1).
```

Its physical transition width is retained across query partitions. Box
integration clips a normalized cube into affine pieces and sums tetrahedral
moments. A two-ramp feature additionally splits at the branch-selection
plane. A zero-time split returns the same field object and new integration
boxes; it performs no mean-based refit.

## Why the initial Bernstein correction is not a cutover solution

For tensor quadratic Bernstein support, every basis function has full-cell
integral `volume/27`. Scaling shared boundary controls by adjacent capacity
and assigning the remaining mass to the private `B111` control therefore
proves exact means, positivity and shared face traces. It does not prove a
correct half-density surface.

A concrete counterexample starts with the nonnegative affine field
`q(y)=0.5+0.6*(h-y)` on a unit cube. Before correction its half-density plane
is exactly `y=h`. Applying `fitPositiveBernsteinMeans` to impose the desired
liquid fraction gives:

| Exact source | Mean error after correction | Surface consequence |
| --- | --- | --- |
| `h=0.25` | Zero at reported precision | All 121 sampled vertical columns lose their half-density crossing |
| `h=0.30` | Zero at reported precision | Maximum height displacement approximately 0.20926 support widths |
| `h=0.75` | Below 1.2e-16 | Maximum height displacement approximately 0.16848 support widths |
| Shallow quadratic height `0.3+0.2*((x-0.5)^2+(z-0.5)^2)` | Zero at reported precision | Maximum height displacement approximately 0.16766 support widths |

For the curved case the initial Bernstein coefficients are converted exactly
from the polynomial, rather than treating midpoint controls as samples. The
initial source-isovalue residual is below 3e-16. Thus these failures are not
explained by an inaccurate starting surface. A centered oblique plane whose
initial mean already equals the target remains exact, which is a useful
control but does not rescue the correction. Sampled-indicator and sampled-ramp
initializers also fail the probe's 0.01-support-width geometry criterion.

The separate zero-predictor counterexample is equally direct: a target mean
of 0.3 assigns `c111=8.1`, giving center density 1.0125 and zero boundary
density. That creates an isolated interior component instead of a planar
30%-filled cell. These are rejection tests, not tolerances to relax.

## What the ramp alternative has demonstrated

The independent geometry suite passes 24 tests covering:

- Exact 25%, 30% and 75% waterlines at support widths 0.125, 1 and 8.
- Finite-volume integrals compared with an independent piecewise one-dimensional antiderivative, including cuts through a narrow ramp.
- Relative ramp widths 0.1 and 1e-5; signed oblique normals with transition widths down to 1e-6 and analytic full-height slab integrals.
- Twelve adaptive query-partition splits retaining the identical field object and total integrated amount.
- Exact minimum/maximum two-plane creases, their analytic means and split integrals.

Additional algebra tests exercise anisotropy, branch dominance, independent
tetrahedral moments and feature integration identities. These results validate
the tested planar and crease primitives. They do not establish general
curvature preservation, inference from arbitrary native cell means, or a
mass-conservative moving-field update.

## Declared curved primitive and accuracy limits

The quadratic primitive retains ten numeric coefficients and a physical local
frame, then clamps the evaluated quadratic to `[0,1]`. Its sphere constructor
uses `q=clamp(0.5+(R²-r²)/(2Rw),0,1)`, so the half-density radius and radial
interface gradient are analytic. Its diffuse integrated amount is deliberately
not identified with the enclosed sharp sphere volume. A numeric primitive
initializer handles declared pools, planes, spheres and quadratics without
retaining authored callbacks or reconstructing from occupancy means.

The design agent reports five CPU tests passing. They verify exact sphere
geometry/gradients, integral bounds containing an independent radial diffuse
mass, an unclamped shallow quadratic resolving in one interval box, and a
clamped shallow bowl reaching absolute tolerance 1e-3 in 122 leaves. Interval
operations round outward, and the receipt explicitly reports whether the
requested tolerance was met.

The sphere reference remains expensive: at 4096 leaves its integral enclosure
is approximately 0.0371 wide, so the requested 1e-4 tolerance is **not met**.
This is useful certified accuracy evidence, not a runtime-performance result
or a production-ready integration algorithm. No new general curvature GPU
path, field transport or hybrid assembly follows from these five passes.

## Verification status

The coordinating agent reported **86 combined CPU tests passing** and
**2 real Dawn GPU tests passing** for this implementation. The independent
24-test affine geometry suite was also run directly and passed. The GPU tests
cover smooth quadratic and crease values/physical gradients, mixed native
box integrals against independent piecewise Gauss quadrature, stale epoch
rejection, and old/new generation lifetime isolation.

The new production test
`tests/sparse-cm12-retained-field-partition-dawn.test.ts` is a stronger cutover
criterion: four initial fixtures, repeated uniform and mixed partitions,
accepted native mean restriction, conserved mass, actual published GPU field
samples, and a resumed physics phase. It now also reports all interior
vertical-ray crossing counts and interpolated crossing motion, preserving
both the lower and upper crossings of suspended components. Root assertions
precede scalar assertions, separating visible interface motion from a scalar
reparameterization. Its first production baseline failed all four cases, but the reasons must
be distinguished. The flat and quadratic fixtures failed during setup because
an empty `initialBrickSeeds_m` array suppressed their base fill. The nominal
sphere/pool and box/pool fixtures consequently contained only the additional
shapes. Those isolated-shape cases reached the first width-4 edit and showed
actual publication changes: `0.04998779296875 → 0.11981201171875` for the sphere
and `0.04998779296875 → 0.126708984375` for the box. These are scalar-sample
changes, not measured surface displacement distances.

The test setup now deletes the optional seed fields so the intended base
pools remain present. A CPU check confirmed empty seeds suppress base wetness
and absent seeds preserve it. The corrected four-fixture test requires a new
Dawn run; the original failures do not demonstrate four valid geometry
failures. The initial receipt is `/tmp/fluid-density-partition-baseline.log`.

The coordinating agent also reported the unfiltered canonical Dawn baseline
had a horizontal D4 density symmetry failure, several lane timeouts, and
exhausted the **180-second suite budget** before later lanes could run. That is
a failing baseline, not a successful regression gate, and is not evidence
that the newly added unused field components caused the failure. Do not
weaken symmetry assertions or increase timing ceilings to change its status.
Focused component successes do not substitute for this full gate. The
coordinator’s receipts are `/tmp/fluid-density-cutover-baseline.log` and
`/tmp/fluid-retained-density-gpu.log`; these temporary paths are local run
artifacts, not checked-in baselines.

## Concrete remaining integration work

1. **Initialization and geometry:** construct one accepted field from tank fills, quadratic height fields, liquid spheres/boxes and scene edits. Specify how exact source surfaces and native amounts coexist when finite ramp thickness changes the integral near boundaries or curved interfaces.
2. **Curvature and hybrid assembly:** establish a practical accurate integral path for the declared curved primitives, then combine curved polynomial support with clamped ramps/creases under one evaluation and integration contract. Define support ownership, shared traces, hanging interfaces, feature selection and representability checks. The current two separate algebras are not this assembled representation.
3. **Dynamics:** advance the accepted field with transport, pressure-driven velocities, diffusion and sharpening while preserving positivity, local amount and intended surface behavior. A per-step unconstrained refit or the rejected private-bubble correction is not a validated update policy.
4. **Excess and capacity:** specify density above one, compression/excess mass and redistribution. The affine ramp clamps to one; the positive Bernstein basis allows values above one. Their physical meaning must be reconciled before native solver authority changes.
5. **Solids:** integrate the retained basis over actual open domains, including terrain and moving rigid geometry. Full-box moments and an averaged open fraction are insufficient for general clipped basis moments.
6. **Production ownership and consumers:** retain accepted field generations across resident replacements; compile native coupling inside the topology transaction; make native density transfer, diagnostics and surface publication use the same accepted authority. Port validated affine/feature evaluation and integrals to the GPU where needed.
7. **Acceptance:** pass the new production partition test, scene/live-edit tests, resumed dynamics checks and the unfiltered canonical Dawn gate. Record performance and storage costs without changing existing timing ceilings.

The user authorized immediate cutover, so lack of authorization is not the
reason it remains incomplete. The missing pieces above are mathematical and
implementation requirements. Switching production now would either adopt a
reconstruction known to distort or erase surfaces, or omit required curved,
transport and solid behavior. Neither outcome would satisfy the requested
retained, grid-independent physical field. The honest current result is
validated infrastructure plus a validated planar/crease alternative, with
production adoption still outstanding.

## Reproduction

Reproduce the reported combined CPU set, then the intentionally rejecting
Bernstein geometry probe:

```bash
node --import tsx --test tests/implicit-density-field.test.ts tests/implicit-density-stencil.test.ts tests/implicit-density-positive-bernstein.test.ts tests/sparse-cm12-density-native-geometry.test.ts tests/sparse-cm12-density-support-coupling.test.ts tests/sparse-cm12-retained-affine-density.test.ts tests/sparse-cm12-retained-affine-geometry.test.ts
node --import tsx --test tests/sparse-cm12-retained-quadratic-density.test.ts
node --import tsx tools/probe-positive-bernstein-geometry.ts
node --import tsx tools/probe-positive-bernstein-geometry.ts --assert-geometry
```

The last command deliberately exits nonzero because the proposed Bernstein
mean correction fails geometry acceptance. Default probe mode emits JSON
receipts without requiring a particular output location.

Run GPU checks sequentially, with no browser or second Dawn process using the
GPU. These tests acquire the repository-wide exclusive lease:

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal node --import tsx --test --test-concurrency=1 tests/sparse-cm12-retained-density-gpu-dawn.test.ts
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal node --import tsx --test --test-concurrency=1 tests/sparse-cm12-retained-field-partition-dawn.test.ts
npm run test:dawn:sparse-cm12
```

See [the canonical regression policy](SPARSE_CM12_DAWN_REGRESSION.md) for lane
selection during diagnosis and the unchanged baseline requirements.
