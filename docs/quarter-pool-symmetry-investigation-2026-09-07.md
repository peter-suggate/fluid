# Quarter-pool symmetry investigation

Scene: `coarse-first-pool-impact-quarter`, adaptive-mass, balanced quality,
coarse-first selection, the authored 32×24×32 lattice and 1/30 s step.

The initial state is exactly symmetric. The first impact now passes a field
and topology symmetry gate through step 18 (0.6 s). The full four-second
trajectory **does not pass**: the first remaining violation is velocity at
step 19. Do not interpret the short regression as certification of the later
waves.

## Structural corrections

1. **Curvature admitted cells according to lane history.** The brick's
   `densityInterfaceCell` reduction flag stayed true after the first interface
   encountered by a lane. Subsequent non-interface cells then contributed
   curvature normals. Sampling now uses `ownDensityInterface`, while the
   separate lane-wide flag still supplies the brick census.
2. **Coarse interpolation knots selected arbitrary fine children.** At a
   coarse/fine boundary, a coarse knot may lie on a fine-cell face. Flooring
   that position always chooses the positive-side child. Transport now chooses
   its spacing from the source's immutable accepted neighborhood, retaining
   that spacing throughout the characteristic. It uses the existing eight
   interpolation points. Uniform coarse support retains its physical spacing;
   the test checks that a small translation responds immediately.
3. **The reuse criterion chose its first corner as a reference.** The maximum
   distance from that corner changes under a corner permutation. The diagonal
   of the velocity bounding box now conservatively bounds every pairwise
   separation. The existing loop reads the same eight velocities.
4. **Divergence reduction overwrote effective velocities.** During collocation,
   binding 3 names the persistent effective-velocity plane, but workgroups
   also wrote diagnostic partials into its first entries. This was a race
   between velocity publication and diagnostic writes. The two maxima now use
   dead sharpening/pressure-image scratch until the following reduction.
   A GPU fixture checks all 128 published velocities and both diagnostics.

These changes add no symmetry projection, mirrored-field averaging, extra GPU
passes, or new GPU allocations. Neighborhood lookup does add arithmetic to
coarse transport; it is not a performance-neutral claim.

## Measurements and tolerances

Measurements compare accepted volume averages and collocated velocities,
independently of rendering. Each checkpoint reports x reflection, z reflection,
and x/z interchange separately, with velocity signs/components transformed
appropriately. Topology is compared as physical cell width on the finest grid,
including absent leaves. Pressure is recorded for diagnosis.

| Measurement | Original capture | Corrected verification |
| --- | ---: | ---: |
| Step 8 peak density symmetry error | 0.020927 | below 0.0002 |
| Step 18 peak density symmetry error | 0.133086 | below 0.0019 |
| Maximum density error, steps 0–18 | above 0.133 | 0.001872 |
| Maximum velocity error, steps 0–18 | above 2.6 m/s, including dilute/air support | 0.006264 m/s |
| Topology, steps 0–18 | split at step 8 and step 15 | exact horizontal symmetry |

The field verifier uses density maximum 0.01 and mean 0.001, velocity maximum
0.02 m/s and mean 0.001 m/s, and relative mass error 0.005. These are absolute
field errors; mean errors include the entire domain. The short integration
regression additionally requires exact topology symmetry, moving liquid, and
remaining coarse/fine interfaces. It exercises the original traversal bug and
mixed-width transport rather than forcing the whole scene fine.

Run the focused regressions:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
node --import tsx --test --test-concurrency=1 \
  tests/sparse-cm12-collocation-velocity-publication-dawn.test.ts \
  tests/sparse-cm12-transport-stencil-symmetry-dawn.test.ts \
  tests/sparse-cm12-quarter-pool-impact-symmetry-dawn.test.ts
```

These tests are also included in `npm run test:dawn:sparse-cm12:coarse-first`.

Run the full trajectory and enforce the same field tolerances:

```sh
npm run probe:adaptive-mass:pool-symmetry
```

This command currently exits nonzero. It writes the **complete** trajectory
before reporting failures, so the first error cannot hide later evolution.
Default output: `artifacts/pool-impact-symmetry/coarse/`. Each run includes the
scene/configuration and resident shader hash, per-step fields/activity/stats,
`trace.json`, and `symmetry-verdict.json`. Use `POOL_SYMMETRY_OUTPUT` to keep
captures separate. `POOL_SYMMETRY_MAX_CELL=1` selects the fixed-fine diagnostic
control; `POOL_SYMMETRY_OVERRIDES` is a JSON object of method overrides.
`POOL_SYMMETRY_STEPS` and `POOL_SYMMETRY_DT` control diagnostic duration/cadence.
The checked-in short test pins its configuration explicitly.

## Remaining failure

The corrected `verified` capture first exceeds the velocity limit at step 19
(0.6333 s), with peak differences about 0.095 m/s while the accepted topology
is still symmetric. Density exceeds its limit at step 20. By four seconds,
peak density differences are approximately 0.35–0.51 and domain-mean errors
0.038–0.043. These are reported failures, not acceptable tolerances.

A fixed-fine control remains near 0.0002 density error through the early
post-impact interval. Tightening pressure convergence to 1e-5 with 128 allowed
iterations did not resolve the adaptive run's later failure. An existing
legacy face-remap control reduced late errors but dissipated substantially
more motion; it was not adopted. A broader face-spacing experiment was also
not retained at this stage. The frozen-topology follow-up below isolates and
corrects face preparation, then identifies the remaining pressure transition.

## Validation status

The three new focused Dawn regressions pass. Running the collocation fixture
against the pre-fix shader fails at cell 0: its expected velocity `[1,0,0,1]`
is overwritten by the diagnostic `[64,0,0,0]`. ESLint passes for the new files.
The repository TypeScript check has existing errors outside these files; the
new files' reported type issue was corrected.

The required unfiltered `npm run test:dawn:sparse-cm12` was run and is **not
green**. It reported the symmetric-expansion velocity assertion,
Mini32 performance above its 40 ms ceiling, multiple lane timeouts, and
exhaustion of the 180-second suite budget. No thresholds were raised and no
lanes were weakened.

A separate isolated pre-stencil snapshot also exceeded the Mini32 ceiling:
147.128 ms versus 138.871 ms for a subsequent current-tree capture, using
three warm-up and twelve measured frames. Timing varied substantially from
the suite capture (66.257 ms), so these observations do not establish a clean
performance baseline or certify the full regression suite. The isolated
pre-stencil snapshot also fails the existing symmetric-expansion lane (its
density assertion). The original
workspace contained other ongoing changes and other processes repeatedly held
the repository GPU lease; runs were serialized behind that lease.

## Frozen topology follow-up

Freeze the accepted reset topology with the existing `setTopologyFrozen(true)`
API. The probe now supports `POOL_SYMMETRY_FREEZE_TOPOLOGY=1`, records it in
`configuration.json`, and asserts identical active leaf IDs, coordinates, spans,
and resolutions at every step. All 121 checkpoints retain 5,508 cells and the
initial symmetric roster. The scene, timestep, pressure settings, and initial
refinement are unchanged.

```bash
POOL_SYMMETRY_FREEZE_TOPOLOGY=1 \
POOL_SYMMETRY_OUTPUT=artifacts/pool-impact-symmetry/frozen-verified \
npm run probe:adaptive-mass:pool-symmetry
```

**This four-second verification still fails.** The first failure moves from
step 8 (0.267 s) to step 53 (1.767 s). No thresholds were relaxed. Through step
52, the worst density error is 0.001587 and worst velocity error is 0.003192 m/s.
The frozen short regression explicitly covers these 52 steps; it is not a
four-second acceptance certificate.

### Two additional structural corrections

1. **Face characteristic spacing.** The coarse face tracer still used coarse
   collocated knots adjacent to fine cells, even after the scalar stencil had
   been corrected. A knot on a finer cell boundary selected the positive child
   with `floor`. The face now traces using the incident cells' supported sampling
   spacing. Its transported staggered component retains the native MAC lattice;
   the existing eight-corner interpolation and dispatch sequence remain intact.
   At step 8 this reduces the peak velocity reflection error from 0.116129 m/s
   to 0.00001733 m/s.
2. **Mixed face membership.** The face's wetness, extension membership, and width
   were inferred from two point queries at the patch centre. On a mixed face,
   each point can select just one of its finer incident children. Receiver
   support now uses all actual incidence terms. Native donor availability also
   checks its incidence terms. A dry native donor delegates to the existing
   extension interpolant instead of constructing another two-point sample that
   has the same child-selection problem. There is no symmetry averaging,
   reflected replay, new dispatch, or scene-dependent solver branch.

The added incidence queries and coarse support lookup do add per-face work;
these changes are not claimed to be cost-free. They remove arbitrary sampling
choices rather than averaging symmetry counterparts.

| Frozen configuration | Step 8 velocity max | Step 52 density max | 4 s density mean, worst transform | 4 s velocity mean, worst transform |
| --- | ---: | ---: | ---: | ---: |
| Before face corrections | 0.116129 | 0.351582 | 0.007243 | 0.059608 m/s |
| Corrected face spacing and incidence | 0.00001733 | 0.0007826 | 0.004567 | 0.021232 m/s |

At four seconds, the corrected run still has a peak density difference of
0.334990 and peak velocity difference of 1.038192 m/s. Mass loss is 0.02701%.
These results are an improvement, not satisfaction of the existing symmetry
limits.

### Remaining pressure-mask amplification

Read-only stage captures at steps 52 and 53 localize the next abrupt increase
to **velocity projection**, after face preparation and scalar transport.
The probe's audit option now saves both native face banks, their parity, the
immutable template, and the accepted roster alongside the scalar stage captures.

At step 53, reflected fine cells `(13,8,10)` and `(18,8,10)` have densities
`0.5001068115` and `0.4999543726`. They fall on opposite sides of the 0.5 pressure
membership threshold. At the reflected x-normal faces centred at `(13,8.5,10.5)`
and `(19,8.5,10.5)`, projection produces a signed discrepancy of 4.408553 fine
cells/s. The largest face discrepancy before projection is only 0.021082 fine
cells/s. Published velocity symmetry then fails, and density fails at step 54.
This is independent of topology adaptation.

A control run with 128 maximum pressure iterations and relative tolerance
`1e-5` still diverges later, despite reducing early numerical error. It does not
justify changing defaults. The remaining problem is a discontinuous pressure
membership transition amplifying small numerical differences; a robust treatment
needs to address that transition without shifting the isovalue, adding a
symmetry-specific tolerance band, or averaging reflected solutions.

Evidence directories:

- `frozen`: initial frozen baseline.
- `frozen-face-spacing`: spacing correction alone.
- `frozen-face-incidence`: both face corrections.
- `frozen-audit-late`: step 52/53 native-face and scalar stage captures.
- `frozen-tight-pressure`: tighter pressure control.
- `frozen-verified`: final full trace and failing unchanged verifier verdict.

The focused GPU support test permutes the wet fine child across a mixed face
and checks uniform coarse versus adjacent fine sampling support. The existing
face-remap test retains its zero-time identity and finite-wave translation
assertions; its fixture now provides actual face incidence and density fields.

Follow-up validation:

- New face-support test, adaptive 18-step test, and frozen 52-step test: **3/3
  pass** (`/tmp/pool-frozen-tests.log`).
- Existing face-remap identity/wave test plus collocation and transport stencil
  regressions: **3/3 pass** (`/tmp/pool-frozen-companion-tests.log`).
- ESLint and `git diff --check`: pass. Repository TypeScript checking still
  reports errors elsewhere; none remain in the changed probe or tests.
- Full frozen 120-step verifier: **fails**, as detailed above; the complete
  trace is saved before the assertion.

To reproduce the pressure-stage captures without running the later cascade:

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal \
POOL_SYMMETRY_FREEZE_TOPOLOGY=1 \
POOL_SYMMETRY_STEPS=54 \
POOL_SYMMETRY_AUDIT_STEPS=52,53 \
POOL_SYMMETRY_OUTPUT=artifacts/pool-impact-symmetry/frozen-audit-late \
node --import tsx tools/probe-pool-impact-symmetry-dawn.ts
```

The canonical `npm run test:dawn:sparse-cm12` was run after the focused tests,
with exclusive GPU access (`/tmp/pool-frozen-regression.log`). It is **not green**:
`symmetric-expansion` fails its density symmetry assertion; topology-budget,
hydrostatic-adaptivity, and far-wall lanes time out; the 180-second budget
prevents the remaining lanes from running. Both performance probes also abort
with `ReferenceError: sparseCM12DawnDefaultValues is not defined` in the
concurrently edited harness. No timing ceilings, lane assertions, or suite
budgets were changed by this investigation.
