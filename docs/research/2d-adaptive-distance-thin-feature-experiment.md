# 2D adaptive distance return and thin-feature protection

## Scope

The 2D `LevelSetVolume` method now builds auxiliary distance guidance on its
accepted adaptive cells. The existing dense vertex phi remains authoritative
for transport, pressure geometry and presentation. This experiment does **not**
replace that surface lattice with adaptive vertex storage.

`levelset_adaptive_distance.rs` seeds distances and closest-point witnesses from
the accepted contour segments. It extends ownership and distance over physical
cell-face connectivity, using physical centre distances across mixed rungs.
Storage for the distance field is proportional to accepted cells and faces;
existing fine capacity, component labeling and target integration are reused.
Far-field values are signed **graph distances**, not exact Euclidean distances.
They guide volume return and are not published as a rendering SDF.

Auxiliary propagation and far relocation stop after eight finest-cell spacings.
Near-surface relocation retains its existing rung-dependent radius. Full solid
cells, cut cells, closed/partial faces and fine solid voxels crossing a proposed
cell-centre path block the graph. Equal-distance component collisions remain
unassigned. A cell containing several represented components is also excluded.
The field is rebuilt each sharpening call, so edits and topology changes cannot
reuse stale cell indices or surface ownership.

## Sharpening

The existing component-conservative donor/receiver redistribution and four
budget rounds remain. Adaptive distance ownership extends the two-fine-cell
label reach. A bounded graph search certifies receiver connectivity before a
transfer. Existing near-surface normal sampling, landing and receiver kernels
run first; the closest contour witness centres the new far receiver kernel.
Far donors can return to that kernel from
up to eight fine spacings away. Receiver capacity and target deficit bound every
transfer. Phi is immutable. The half-finest-cell diffuse-island gate remains.

New receipt fields are `adaptiveDistanceCells` and `farRelocatedVolume` (the
latter counts transfers from donors ineligible for the former local band). Existing
mass, component, excess, distance-weighted mismatch and path-length diagnostics
remain. No arbitrary residue deletion or global mass redistribution is added.

## Thin-feature veto

`levelset_thin_features.rs` bins contour segments spatially and intersects their
normal lines with nearby opposing segments. This catches represented oblique
sheets, filaments/small droplets, and narrow air gaps without requiring a wet
solver-cell centre or dry neighbouring density samples. The physical threshold
uses the existing `thin_feature_cells` option (default two finest spacings).
Both sides of a brick seam are protected.

Protected bricks receive the existing finest-rung safety floor in both ordinary
planning and projected transport support. Current geometry invalidates old
surface proofs. Proof publication also checks the geometry independently, and
candidate restriction now samples vertices and half-grid points as well as the
old cell centres. Exact-empty retirement cannot discard a protected brick.
A final check after region/2:1 closure rejects a generation that would coarsen a
protected accepted brick; conflicting authored caps report
`THIN_FEATURE_VETO` and retain the accepted topology.

This is conservative protection of represented geometry, not a proof of arbitrary
sub-grid topology. Features absent from the current contour cannot be recovered.
Opposing-segment tests may retain extra bricks near curved small features; the
normal-angle cutoff is a detection heuristic, not a geometric thickness oracle.

## Alignment with current 3D coarsening

The direct-phi 2D path already used velocity variation and closing-flow demand,
a four-finest-spacing material working limit, restriction proofs and two quiet
proof epochs. This change also:

- removes page-normal variation from its coarse activity score;
- skips retired incoming-impact prediction for direct phi;
- resets proof epochs when the relevant policy changes;
- allows ordinary demotion only on topology epochs;
- resets surface proof history on immediate certified-bulk demotion.

Legacy density-based 2D policy parameters remain for the sibling transport
method. Thin-feature protection and the denser restriction checks are deliberate
2D experimental additions beyond current 3D detection.

## Validation

Focused tests cover far return across a 2:1 seam, wall blocking, ambiguous
component ownership, mass/capacity conservation, immutable phi, small-island
retention, oblique sheets and air gaps, off-centre droplets, stale certificates,
policy changes, and region-cap conflicts. Longer physics and build results are
recorded below after the runs complete.

### Ten-step production dam comparison

Both runs use `investigate_2d_levelset`, the same authored dam scene, dt=1/30,
and ten steps from reset. Baseline sources were exported from HEAD into an
isolated temporary directory; the working tree and concurrent 3D edits were
not replaced. This is an evolving-state comparison, not isolated kernel timing.

| Measure at step 10 | Baseline | Experiment |
| --- | ---: | ---: |
| Published solver cells | 48 | 48 |
| Total conserved volume | 167.9999995 | 167.9999984 |
| Outside-band absolute phi/volume mismatch | 0.47610 | 0.29000 |
| Total absolute phi/volume mismatch | 10.26432 | 10.62504 |
| Distance-weighted sharpening mismatch | 24.19732 | 22.93694 |
| Over-capacity volume | 3.45145 | 3.45446 |
| Unassigned volume | 0.55930 | 0.01740 |

Outside-band mismatch falls 39.1%, distance-weighted mismatch falls 5.2%, but
total mismatch rises 3.5%. Capacity excess is almost unchanged. This does not
establish improved shape or energy over sustained motion. The first prototype
replaced the near-surface receiver kernel and performed worse; the final
experiment preserves that kernel and adds far return after local correction.

The existing `zero_velocity_is_identity_on_mixed_adaptive_cells` and
`small_translations_cross_fine_coarse_seam_in_both_directions` regression
assertions fail on both the baseline and experiment. The diagonal seam value
is identical (0.7993174791 against 0.8000000119 with tolerance 0.0005). The
zero-velocity test supplies a volume field inconsistent with its phi target;
sharpening changes that field in both versions, with different first failing
cells. These are baseline failures, not a clean suite or proof of identical
stationary behaviour. No assertions or tolerances were weakened.

### Final CPU and Wasm checks

- 33 level-set unit tests passed, including the new contour detector.
- 30 resolution-policy tests passed, including stale proofs, policy reset and
  authored-cap conflict protection.
- Five focused 2D world tests passed after the final sharpening adjustment.
- Nine integration targets: 35 tests passed; the two baseline regression
  assertions described above failed. Hydrostatics, adaptivity, pressure,
  injection, ceiling separation, energy, sharpening and wall-velocity targets
  all passed.
- The broader library run passed 198 tests with one ignored before the final
  local/far kernel adjustment; relevant focused tests were repeated afterward.
- Scalar, SIMD and threaded Wasm artifacts were rebuilt and validated from the
  final sources with `npm run build:physics-wasm`.

Comparison receipts: `artifacts/level-set-volume/2d-adaptive-distance-comparison.json`.

### Canonical Dawn gate

`npm run test:dawn:sparse-cm12`: 7 lanes passed, 10 failed in 361.3 seconds.
Six failures were lane timeouts (symmetric expansion, page budget, clipped
transfer, hydrostatic adaptivity, mini32 correctness and mini32 performance).
Four were assertions (min8 region surface, both far-wall scenes and outside-tank
symmetric collapse). Live rigid/liquid insertion and mini64 performance passed.
No timing ceilings or assertions were changed. This gate ran serially with
exclusive GPU access after CPU tests and Wasm compilation completed.

This is the current combined working tree, including independently edited 3D
files. It is not a clean gate or an isolated attribution of those failures to
this 2D experiment. Receipt:
`artifacts/level-set-volume/2d-adaptive-distance-dawn-gate.json`.
