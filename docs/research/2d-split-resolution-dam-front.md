# 2D level-set dam front with half-tank resolution constraints

## Follow-up: unconstrained symmetry (latest implementation)

The subsequent no-region screenshot was reproduced exactly: frame 7 maximum
face speed 59.297230, 246 redistanced samples and 8 over-capacity cells. The
first unequal adaptive rungs appeared at frame 6. The earlier forced-region
controls did not cover this regression.

Three isolated defects are now fixed:

1. Direct-phi thin-feature detection mixed phi-liquid membership with
   conservative-density neighbor occupancy. At frame 6, mirrored phi-air
   neighbors contained density 0 and 0.107682; the latter incorrectly hid
   surface exposure and suppressed thin-feature refinement. Both sides now
   use phi membership for this geometric decision.
2. Sharpening committed donors sequentially. A symmetric limited-capacity
   test left donor volumes 0 and 0.5. Donors now propose simultaneously and
   share each receiver's available capacity proportionally, preserving near
   priority, component boundaries and conservative transfers.
3. Refinement assigned the phi/conserved-volume mismatch to the first child.
   At frame 3 this increased maximum density reflection error from below
   1e-6 to 0.518357. Geometric child volumes are now scaled down together or
   share remaining capacity proportionally; excess remains conservative.
   The same frame now has maximum density reflection error 7e-7.

With all fixes, all brick activity and resolutions remain exactly reflected
through 60 unconstrained frames (2 seconds), including wall impact. Frame-7
fronts are 0.502250373 and 31.497751236 fine cells. Over the full 60 frames,
maximum density reflection error is 2.71201134e-05 and maximum vertex-phi reflection
error is 1.85966492e-05. Maximum absolute conservative-volume drift from 128 is 1.65915117e-06.
The integration regression checks exact reflected topology, density within
1e-4, face velocity within 1e-3 and volume within 1e-4 every frame.

The latest forced 2/1 frame-6 fronts are 2.020061016 and 31.023111343, a
1.043172-cell coarse/fine advance gap. The older measurements below describe
the first staggered-transport implementation, before these symmetry fixes.
Latest receipts are in
`artifacts/level-set-volume/split-resolution-ladder/unconstrained-symmetry-updated.json`
and `symmetry-split-updated.json`.

Validation limit: the combined fixes currently fail the mini32 energy
regression at frame 9 (total energy 3,852,142.388), although each of the three
fixes in isolation passes it. This is unresolved; its threshold was not
changed. The two previously recorded volume-transport regression failures
also remain. Investigation stopped at the user's request to rebuild.
The quarter-pool frame-5 trajectory snapshot was updated to measured kinetic
energy 2530.0701, retaining its 0.01 tolerance and 22-cell count.


Investigated and implemented 2026-09-17 against the existing working tree,
preserving its pre-existing solver edits. Measurements use the Rust `World`
behind the 2D Advance Lab: dt = 1/30 s, 256 pressure iterations, relative
tolerance 1e-6. Baseline line references below describe the pre-change source.

## Implemented result

The coarse-side lag is reduced, not eliminated. At frame 6 (0.2 s), before
wall contact, it falls from **1.399655 to 1.043066 finest cells**, a **25.48%**
reduction (69.983 mm to 52.153 mm). Both fronts advance faster:

| Frame | Before coarse advance | Updated coarse advance | Before fine advance | Updated fine advance |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.258955 | 0.326369 | 0.344360 | 0.395948 |
| 2 | 0.747384 | 0.931544 | 0.971266 | 1.123793 |
| 4 | 2.595226 | 3.018985 | 3.335070 | 3.760000 |
| 6 | 5.271293 | 5.980050 | 6.670948 | 7.023115 |
| 7 | 6.617151 | 7.038286 | 8.000000 | 7.511593 |

Units are finest cells, 0.05 m each. The lag does not decrease at every frame:
frame 4 is 0.741015 versus 0.739844 before. Frame 7 includes wall response and
is not used to claim a percentage improvement.

### Changes

- `staggered_velocity.rs` supplies independent face-component interpolation:
  regular-grid bilinear interpolation, adaptive affine MLS with each face's
  width, nearest-ring support when needed, and explicit domain ghost samples.
  Tangential wall samples reflect evenly; normal samples reflect about the
  accepted wall velocity. Queries are not clamped to a half-cell interior band.
- Synchronous face extension preserves liquid-face and wall values, rebuilds
  dry values from the phi seed set, and does not read conservative density or
  collocated velocity. Cell velocity remains derived data for diagnostics and
  sizing. Velocity advection retains the existing embedded-boundary segment
  clipping; phi/volume capacity handling remains in place.
- Phi backtraces, conservative-volume footprints, and velocity advection use
  the same immutable staggered evaluator. Tests poison cell velocities to
  verify that they no longer control transport.
- End-of-frame coarsening retains the preceding face field for the next
  velocity-advection gather. Injection invalidates this retained source.
  This implements direct old-grid sampling for that remesh, not the complete
  Ando–Batty stage ordering. The projected-support transition and its pressure
  correction remain in their existing order before scalar transport.
- Removing cell-centre damping exposed excessive liquid crowding in the
  existing remap at wall impact. After the original three capacity-balancing
  sweeps, additional sweeps scale only overfull *liquid* receivers and restore
  donor totals, stopping at 1e-6 relative excess or 64 sweeps. No volume is
  clipped, and no new donor/receiver links are invented. Infeasible stencils
  may retain excess; the receipts report it. Iterating all capacity rows to
  convergence was rejected because it perturbed remote translating liquid.
  This is a conservative-remap safeguard, not a claim that either cited paper
  specifies this exact hybrid volume algorithm.

The first coarse/fine difference remains because width-2 faces average a
spatially varying width-1 velocity. The papers do not promise identical
trajectories when the interface is forced onto different velocity resolutions.
The implementation removes the additional face-to-cell transport filtering;
it does not reconstruct discarded subface information.

### Updated controls and conservation

| Widths L/R | Frame-1 front x L/R | Frame-6 front x L/R |
| --- | --- | --- |
| 2 / 1 | 7.673631 / 24.395948 | 2.019950 / 31.023115 |
| 1 / 2 | 7.604052 / 24.326368 | 0.976899 / 29.979801 |
| 1 / 1 | 7.604052 / 24.395948 | 0.976326 / 31.023664 |
| 2 / 2 | 7.673631 / 24.326368 | 2.027199 / 29.972694 |

At frame 6 the mirrored mixed runs differ by at most 0.000249 cells; equal-width
reflection residuals are 0.00000918 cells (fine) and 0.0001066 (coarse).
The 1e-8 pressure-tolerance control produces identical published front
coordinates through frame 7; this is not a claim that every solve reaches
that requested tolerance.

At frame 7, corresponding to the supplied screenshot:

| Diagnostic | Before | Updated |
| --- | ---: | ---: |
| Maximum face speed | 60.373047 | 62.751389 |
| Over-capacity volume | 1.127615 | 0.180446 |
| Maximum over-capacity ratio | 0.397866 | 0.052560 |
| Over-capacity cells (including tiny excesses) | 21 | 40 |
| Redistanced samples / fallbacks | 391 / 0 | 394 / 0 |

The 30-frame mixed run retains conservative volume 128 with a maximum absolute
error of 1.57e-13 fine-area units. This refers to the conservative field, not
exact agreement between phi-implied area and volume. At frame 30 the phi area
is 99.017939, so that existing hybrid-method discrepancy remains observable.
Scalar, SIMD, and threaded Wasm fronts are bit-identical to native for all four
width configurations through frame 7; their volume measurements agree within
1e-10. All three artifacts were rebuilt and validated for their CPU features
and source fingerprint.

### Validation and limits

- 8 new interpolation/extension/remesh/split-dam integration tests pass,
  including reflected T-junctions, boundary support, constant and affine
  fields, immutable face seeds, and actual advection after coarsening.
- 199 current core unit tests pass, one is ignored. Two lengthy, unrelated
  3D compatibility tests passed in the earlier full 200-test run and were
  excluded from the final focused rerun. The added receiver-balancing unit
  test verifies capacity reduction, exact donor totals, unchanged feasible
  translation weights, and no mass deletion for an infeasible stencil.
- Native hydrostatic, adaptivity, injection, ceiling separation, wall velocity,
  sharpening, transition, mini32 impact-energy, and pressure-feedback checks
  pass. The mini32 energy ceiling and mass tolerance were not changed.
- The quarter-pool trajectory snapshot was updated from 26 cells/K=2474.4036
  to 22 cells/K=2534.1388 at frame 5, retaining its tolerances. The pressure
  test now verifies the correction rate at every step and net release of the
  initial perturbation: the new remap removes almost all initial excess at
  frame 1, so subsequent crowding need not be smaller than that near-zero
  intermediate value. No pressure relaxation parameter was changed.
- Two transport assertions already failed on the untouched solver and remain
  failing: stationary mixed-grid volume identity (0 versus 0.32000002), and
  the diagonal seam-overlap tolerance. Seam transfer is now 0.80110009 versus
  expected 0.80000001; the baseline was 0.79931748. The 5e-4 tolerance is
  unchanged, and this remaining discrepancy is not claimed to be fixed.
- The broader Wasm suite passes 5/7. Both failing cases reproduce in native
  baseline evidence: Figure 7 frame 23 only publishes curvature floors 0/1,
  while its test requires >1; hillside mechanical energy first exceeds its
  1.02-times-initial ceiling at frame 14 before and frame 16 after. The new
  hillside run peaks at 1.295841 at frame 17; baseline becomes unstable by
  frame 120. The new hillside phi area also deteriorates to 0.932738 by
  frame 120 despite conservative volume ~384 (baseline phi area 73.309260
  with already-unbounded velocities). These measurements do not establish
  long-horizon surface fidelity. These are unresolved limitations, not a clean
  broad-suite pass.
- Required isolated `npm run test:dawn:sparse-cm12`: **8/17 passed**, 364.56 s
  within its 480 s budget. Limits were unchanged; the repository GPU lease
  was clear before and after, and this task launched no concurrent browser
  or second Dawn run. Five lanes timed out: symmetric expansion (20 s), topology page
  budget (30 s), clipped topology transfer (20 s), mini32 correctness (25 s),
  and mini32 performance (20 s). Four correctness lanes failed: mixed-region
  surface split height 0.100545 > 0.01 cells; Long Dam publication 66 versus
  106 expected cells; missing compiled topology faces in the hills far-wall
  and outside-tank collapse lanes. Mini64 performance passed at 54.067 ms
  median versus its 110 ms ceiling. These lanes execute the separate GPU path,
  whose implementation was not changed here. The gate is **not green**.
  Full machine-readable receipt: `artifacts/level-set-volume/split-resolution-ladder/dawn-regression.json`;
  full output: `dawn-regression.log` alongside it.

Updated evidence: `artifacts/level-set-volume/split-resolution-ladder/updated-measurements.json`,
`wasm-measurements.json`, and `hillside-energy-{before,after}.json`.
Reproducible scene: `rust/core/testdata/split-resolution-ladder-seed.json`.
Native probes: `investigate_split_levelset.rs` and `investigate_levelset_energy.rs`.
Wasm probe: `tools/wasm/split-resolution-levelset-probe.ts`.

## Original investigation (baseline)

## Reproduction

The catalog selection `sparse-cm12-ladder-symmetric-3d` produces the document
whose internal scene ID is `symmetric-expansion`. Its 2D slice is 32 by 16
finest cells, at 0.05 m per cell. The initial liquid extends from x=8 to x=24.
Apply full-height regions x=[0,16], held at width 2, and x=[16,32], held at
width 1, before advancing. Commands initially set the policy; the first
projected-support transition actually enforces the rungs.

The frame-7 native result matches every displayed diagnostic in the supplied
screenshot: max |u| = 60.373046875 (60.37 displayed), 21 over-capacity cells,
maximum excess ratio 0.39786613 (0.40 displayed), 391 redistanced samples,
zero redistance fallbacks, and zero drift to displayed precision. The left
front is at x=1.3828491; the right front reaches x=32. This is a 0.0691425 m
lag of the coarse side relative to reflection symmetry.

| Frame | Leftward advance, width 2 | Rightward advance, width 1 |
| --- | ---: | ---: |
| 1 | 0.258955 | 0.344360 |
| 2 | 0.747384 | 0.971266 |
| 4 | 2.595226 | 3.335070 |
| 6 | 5.271293 | 6.670948 |
| 7 | 6.617151 | 8.000000 |

Advance is measured from the initial fronts in finest-cell units, using the
extrema of the published yellow zero contour, independently of the blue volume
visualization. Wall contact caps the rightward advance at 8 in frame 7.

## Confirmed cause

The coarse side transports phi with a slower, spatially averaged velocity.
The first asymmetry occurs during the first resolution transfer, before
level-set transport, redistancing or volume sharpening has run.

1. Frame 1 begins with 128 width-1 liquid cells. Primary pressure projection
   is left/right symmetric. The outermost bottom face speed is 12.424195 on
   both sides; the adjacent face one cell higher has speed 7.6723485.
2. The projected-support transition allocates air support and enforces the
   two region widths. On the left, the two unit-height front faces become
   one height-2 face. `transfer.rs:586` integrates source face flux and divides
   by target face measure: (12.424195 + 7.6723485)/2 = 10.048272. This preserves
   integrated flux but discards the bottom face's larger local speed.
3. `lifecycle.rs:222` recomputes cell-centred velocity from the transferred
   faces. `numerics.rs:3733` performs weighted face averaging. For the coarse
   bottom-front liquid cell, the outer and inner x-face speeds are 10.048272
   and 5.4890075; their average is 7.7686396. The fine counterpart averages
   12.424195 and 8.262352, giving 10.343273. The coarse cell therefore provides
   an outward transport speed about 24.9% lower.
4. The post-transfer pressure solve changes these speeds by only about 1e-6.
   `world.rs:1255` then extends velocity from phi-liquid cells into air.
   `numerics.rs:189` seeds that extension from **cell-centred** velocity. The
   bottom air cells inherit 7.7686405 on the coarse side and 10.343275 on the
   fine side.
5. `levelset_volume.rs:283` chooses the local solver-cell width as its sampling
   span, and `:286–298` RK2-backtraces every fine phi vertex through
   `sample_support`. That sampler reads `fields.cell_velocity`
   (`numerics.rs:441–518`); it does not read the interface face velocity.
   Near the floor, its y clamp also samples at half the local cell width:
   y=1 on the coarse side versus y=0.5 on the fine side.

Thus the shared fine phi grid retains geometric detail, but does not retain
fine velocity detail after the solver cells are coarsened. The first published
coarse-front displacement is 24.8% smaller. This is a measured consequence of
face restriction, cell-centred velocity reconstruction and extension, and the
velocity field used for contour advection. The flux restriction itself does
not violate integrated flux conservation.

## Controls and causal replay

| Held widths, left/right | Frame-1 outward advance L/R | Frame-7 front x L/R |
| --- | --- | --- |
| 2 / 1 | 0.258955 / 0.344360 | 1.382849 / 32.000000 |
| 1 / 2 | 0.344361 / 0.258955 | 0.000000 / 30.548859 |
| 1 / 1 | 0.344361 / 0.344360 | 0.000000 / 32.000000 |
| 2 / 2 | 0.258955 / 0.258955 | 1.396477 / 30.576073 |

The lag follows the coarse region. It also occurs when both halves are coarse,
so it does not require the central coarse/fine seam. There is a smaller
left/right discrepancy after several frames even in the equal-width controls;
these measurements do not establish its cause. In particular, the mirrored
mixed runs should not be described as bitwise mirror images.

The probe captures the frame-1 graph and fields immediately before transport,
then replays the actual `levelset_volume::advance` implementation against the
same initial surface. Only `cell_velocity` is replaced in the controlled arms;
the mixed topology, face velocities and density remain as captured.

| Replay | Published front x L/R |
| --- | --- |
| Original captured velocities | 7.741045 / 24.344360 |
| Zero cell velocities | 8.000000 / 24.000000 |
| Equal outward speed 10, zero vertical speed | 7.666667 / 24.333334 |

The original replay exactly matches the real frame. The equal-speed replay
moves both fronts by 1/3 cell, to floating-point precision, through the same
transport and redistancing code. This isolates the unequal advecting velocity
as the cause of the first-frame discrepancy; there is no automatic half-speed
factor attached to coarse phi vertices.

Pressure control: tightening tolerance from 1e-6 to 1e-8 increases frames 2–7
from 40 to 256 iterations. The frame-7 coarse-front coordinate changes by only
3.58e-7 cells. The tight runs hit the iteration cap above the requested
tolerance; they are not claimed to converge to 1e-8. Additional iterations do
not remove this lag. Frame 1 is already symmetric before the resolution
transfer, and its post-transfer projection scarcely changes velocity.

This investigation does not quantify each later-frame contribution from
pressure, velocity advection, redistancing, volume feedback and grid changes.
It establishes where the discrepancy starts, the exact averaging/sampling
chain, and that the accumulated coarse-side lag follows local resolution.
At this initial investigation stage no solver correction had been tested;
implementation results are recorded above.

## Paper comparison: what should change

Read against the repository copies of Losasso, Gibou and Fedkiw (2004),
`docs/papers/losasso-2004-octree-water-smoke.txt`, and Ando and Batty (2020),
`docs/papers/ando-batty-2020-practical-octree-liquid-simulator.txt`.

The measured velocity reduction is not, by itself, proof that face restriction
is implemented incorrectly. Losasso explicitly averages child-face velocities
when coarsening (§3, text lines 101–115). Their contour advection uses velocities
constructed at nodes directly from surrounding faces; extrapolation proceeds
through nodal velocities and then returns to faces (§6). They refine an
interface band and use particle-level-set correction. Their method is not the
current face-to-cell-centre-to-air-to-phi pipeline, nor a guarantee of equal
front speeds at arbitrarily different forced interface resolutions.

Ando–Batty gives a more directly applicable alternative for adaptive surfaces:

- Keep velocity components on their staggered faces. Interpolate each
  component independently, using bilinear interpolation in regular 2D regions
  and their affine MLS construction with sample-specific spatial scales near
  level transitions (§5, equations 33–35). Section 5 explicitly avoids temporary
  nodal/element-centred velocity conversions because they add diffusion
  (text lines 552–560).
- Supply boundary support deliberately: mirror interpolation samples at the
  exterior domain boundary (§5.1). This differs from moving every query to
  the nearest cell-centre band using half the owner's width. Their discussion
  distinguishes domain boundaries from embedded solids.
- Retain the previous grid as the source when adapting: construct the new
  octree, advect velocity and phi from the previous octree directly into the
  new grid, then project, redistance and extrapolate (§3.2, lines 190–204).
  Our first-step sequence instead restricts an already-projected velocity
  before transporting phi. Combining remeshing with advection is a separate
  architectural change, not a one-line interpolation replacement.
- Their method still loses detail and damps motion on coarse cells (§9.2.2,
  lines 915–936). Pressure under-resolution and semi-Lagrangian diffusion
  remain limitations even with the improved interpolation.

The original implementation recommendation was to first create
one component-wise staggered velocity evaluator and compatible face velocity
extension, preserving accepted liquid face values and wall constraints. Use it
for both RK2 samples in phi advection, conservative-volume footprint tracing,
and velocity characteristic tracing. Keep cell-centred velocities as derived
data rather than the transport authority. Replacing only phi's sampler would
leave `trace_point` and `trace_characteristic` on the old cell-centred field;
reading existing dry face entries without adding face extension would likewise
be incomplete. Then evaluate previous-grid-to-new-grid advection separately.

This recommendation does not require replacing the existing conservative
volume remap or pressure operator with the entire Ando–Batty method. Their
pressure layout also differs: at a T-junction, child faces inherit a parent-face
velocity (§4.2), so equations should not be transplanted without reconciling
the representation. The measured onset here is not evidence of a pressure
seam defect.

Validation should separate affine/constant interpolation accuracy (including
mixed grids and wall support), preservation of accepted face data, and measured
dam-front convergence. Exact coarse/fine trajectory equality is not a promise
made by either paper. The forced width-2 region removes the option of refining
the front when its velocity variation needs finer resolution.

## Repeatable probe and evidence

Probe: `rust/crates/fluid-core/examples/investigate_split_levelset.rs`.
Saved scene and compact measurements:
`artifacts/level-set-volume/split-resolution-ladder/{scene,measurements}.json`.
The measurements include all five run histories, stage velocities and the
three replay results. Set `FLUID_SPLIT_FULL_TRACE=1` to emit full per-stage fields. The current
probe prescribes face velocities in its controlled replay arms; the historical
cell-velocity replay above was captured before the implementation change.

```sh
cargo run --manifest-path rust/Cargo.toml -q -p fluid-core \
  --example investigate_split_levelset -- \
  rust/core/testdata/split-resolution-ladder-seed.json 2 1 7 \
  > /tmp/fluid-split-21.jsonl
```

Repeat with `1 2`, `1 1`, and `2 2`. Append `1e-8` after the frame count for
the pressure control. The saved scene was exported with
`sceneDocument(findSceneDefinition("sparse-cm12-ladder-symmetric-3d"))` from
the current TypeScript catalog.

Validation consisted of native reproduction, four resolution configurations,
the tighter-pressure run, and captured-state transport replays. No production
simulation change was made, so the post-refactor Dawn gate was not run.
