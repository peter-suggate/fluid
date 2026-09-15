# 2D / 3D sparse level-set discrepancies

Investigation: 2026-09-15, against the current working tree (including existing
uncommitted changes). The initial investigation made no solver changes; the
subsequent 2D adaptivity update is recorded below.

## Follow-up: 2D adaptivity update

The 2D direct-level-set path now publishes generation-stamped next-rung
certificates after accepted surface publication. Since its phi grid is
independent of solver resolution, the certificate evaluates candidate-spacing
curvature and accepted volume/phi agreement, following the 3D fine-band proof.
The volume integral uses the existing 2D contour integration. Disabled
coarsening, excessive velocity, static geometry floors, region constraints,
curvature and volume disagreement reject certificates. Surface-proof rejection
clears the quiet proof count; non-surface bricks retain their ordinary
coarsening epochs. A topology change invalidates old generation stamps.

Phase crossings and thin-feature depth now use phi instead of volume fraction.
The direct-level-set motion floor uses absolute liquid speed, consistently in
both planners; the separate remap experiment retains relative-motion sizing.
Contourless/submerged bulk retains its translation invariant behavior. The
coarse-first branch reloads accepted proof history so it cannot count an epoch
already incremented by the preceding legacy branch.

The final native dam-break probe retains 336 represented cells at frame 10,
versus the original 384. Moving surface bricks may still require B8; a separate
resting-world regression proves actual demotion with an unchanged contour.
Unit fixtures cover two distinct proof epochs, stale generations, curvature,
volume disagreement, disabled coarsening and fast-front support. The full
slab dynamics need not agree while the transport and sharpening operators
remain different.

Correction to the original diagnosis: the `false` argument in projected-support
measurement is the topology-epoch flag, not the motion-sizing flag. The latter
was passed `include_interface_support`, so the claimed pre/post absolute-speed
asymmetry below was a mistaken reading of positional arguments. Current GPU
3D uses absolute speed; that is the behavior used for the updated direct-phi
motion floor.

## Follow-up validation

- Native library: 191 passed, one ignored; two unrelated long CPU-3D
  compatibility tests excluded from the final run. The 12-frame case passed
  during the earlier broad run; the 30-frame case was stopped.
- Focused native fluid integration: 13 passed (adaptivity, hydrostatics,
  wall velocity and ceiling separation).
- Wasm level-set UI flows: all seven passed, including Figure 7 trailing-air
  cleanup, pool impact and interactive injection.
- Scalar, SIMD and threaded Wasm rebuilt and artifact validation passed.
- Dawn: the full gate finished within its 480-second budget but failed nine
  lanes: symmetric-expansion, topology-page-budget, mini32-correctness,
  min8-region-surface, mini32-performance, mini64-performance,
  long-dam-far-wall, tall-cells-hills-far-wall and outside-tank-symmetric-collapse.
  Six failures were timeouts; the remaining three were assertion failures.
  This run overlapped CPU compilation/testing, so its timings are not an
  isolated GPU performance comparison. No GPU solver code or gate threshold
  was changed. Full receipt: `artifacts/level-set-volume/2d-adaptivity-dawn-gate.json`.
- Final native frame receipts: `artifacts/level-set-volume/2d-adaptivity-after.json`.

## Conclusion

The broad blue band is principally a conservative-volume/phi disagreement,
not evidence that the yellow zero contour itself is thick. The implementations
have different remaps and substantially different sharpening operators. The
2D loss of adaptivity also has a concrete implementation defect: surface
demotion requires a certificate that the production 2D path never supplies.

## Reproduction and measured 2D evidence

Run from the repository root:

```sh
cargo run --manifest-path rust/Cargo.toml -q -p fluid-core --example investigate_2d_levelset
```

The probe uses the existing water-box advance seed, level-set-volume transport,
dt=1/30, 256 pressure iterations and 1e-6 relative tolerance. The full receipts
are in `artifacts/level-set-volume/2d-3d-discrepancy-2d-frame10.json`.

| Frame | Represented cells | State |
| --- | ---: | --- |
| 0 | 208 | Three active B8 bricks and one active B4 brick |
| 3 | 272 | First right-hand support brick activated at B8 |
| 4 | 336 | Second right-hand support brick activated at B8 |
| 10 | 384 | All six bricks B8; entire 24x16 domain finest |

Frame 10 reproduces the screenshot's 45 over-capacity cells and maximum excess
2.31795 K. Sharpening relocates 2.46469 finest-cell areas in this frame and
reduces the band absolute V/phi mismatch from 8.14445 to 3.21507 (60.5%). Its
maximum relocation distance is two finest cells. This is direct evidence that
the sharper blue band depends materially on volume correction.

The sharp appearance does not prove accurate liquid geometry: conserved volume
is 168.0000006, while phi implies area 161.93821, about 3.61% smaller. Total
excess volume remains 6.54133. Amber stripes denote over-capacity, not solids,
in the compared volume visualization.

## 1. Volume sharpening is materially different

2D `levelset_volume.rs:914` calls `levelset_sharpening::sharpen_volume` after
advection/redistance. `levelset_sharpening.rs` integrates the fine contour into
cell targets, labels connected liquid regions, extends those labels into nearby
air, and moves excess volume toward deficit cells in the same region. Receivers
can be multiple cells away, within two local cell widths, with four redistribution
rounds. Phi is immutable.

3D `resident-volume.wgsl.ts:1072` implements preparation, proposal, gather and
commit. It computes phi-derived targets, but performs only one simultaneous
exchange over adjacent physical faces. Each transfer is divided by the cells'
face degrees. Cut cells, partial/blocked faces, missing metric support, and
faces with positive phi at their centre are skipped.

Consequences visible directly in the operator:

- A diffuse donor separated from a deficit by another donor cannot relay
  volume across both faces in one frame.
- A diffuse donor above the contour cannot exchange through an air-side face
  centre, even if a liquid-connected deficit is nearby.
- Even an eligible adjacent transfer uses only the degree-divided budget
  (typically 1/6 in regular 3D), rather than exhausting available donor/receiver
  budgets as the 2D redistribution can.

These restrictions provide a strong explanation for a persistent 3D blue halo.
The exact contribution on the photographed 3D run remains unmeasured; a matched
3D stage capture is needed to quantify it. It would be incorrect to say that
current 3D has no sharpening: it is dispatched and enabled by default.

## 2. The conservative remaps differ before sharpening

2D `levelset_volume.rs:230` RK2-traces four corners and the centre of each
receiver, then clips a deformed footprint against donors. 3D
`resident-volume.wgsl.ts:730` traces the centre and translates an axis-aligned
box of unchanged widths. Velocity gradients can make those translated receiver
boxes overlap or leave gaps; they cannot represent the deformation of the
2D footprints. Both perform three row/donor balancing rounds.

Thus 3D is feeding a different coupling matrix into its correction, and coarse
solver cells also average volume over physically wider regions. Neither total
volume conservation nor redistancing guarantees a sharp V/K interface.

## 3. Confirmed missing 2D surface-coarsening certificate

`resolution.rs:1960` permits a surface demotion only when
`surface_proof.generation_by_target_resolution[next]` matches the current
topology generation, then requires two fresh proof epochs. History defaults
to `surface_proof: None`; measurement only copies the previous value.

A search of production Rust source finds no writer creating these certificates.
The assignments are in unit tests only. The probe confirms `surfaceProof: null`
for every brick at frame 10. Consequently a surface brick can refine but cannot
pass this coarsening gate, even after its measured curvature floor falls.

This is different in GPU 3D: the presentation proof kernel evaluates candidate
geometry/curvature and volume agreement and publishes the accepted generation
to the activity record (`webgpu-sparse-cm12-resident.wgsl.ts:10740`).

Projected support can promote pages before the end-of-frame receipt. At frame
9 of the original run, the end-of-frame receipt asks brick 6 to drop from B8 to
B4; by frame 10 it is B8 again. A post-frame receipt alone therefore misses
promotions occurring during support preparation. See the correction above
about the original interpretation of the measurement arguments.

The all-fine result is not explained solely by high curvature: at frame 10
five bricks have encoded curvature floor B1 and the sixth B4. Missing proofs,
support demand, retention and thin-feature safety contribute to the final rungs.

## 4. Surface representation and redistance

2D advects a full-domain (nx+1)*(ny+1) fine vertex grid independently of solver
rungs. It reconstructs contour segments and computes distance to those segments,
updating vertices within a two-fine-cell band.

Current 3D also decouples phi resolution in its fine-phi band. The default
`FINE_PHI_BAND_ENABLED` is true; `cm12PhiBrickPlan` selects B8 phi templates
around surface/thin/cut bricks and their presentation apron, where available.
Outside that region it follows solver rungs. Template availability and arena
capacity can constrain this gain. Thus coarse blue cells alone do not establish
coarse phi or explain contour rounding in this working tree.

3D redistance propagates closest-point seeds over an adaptive graph with sixteen
bounded relaxation rounds, followed by constraint projection and contour audit.
It is not the 2D exact segment-distance operation. This remains a possible
contributor to yellow-contour differences, not a demonstrated cause of the blue
volume halo.

## Interpretation and next comparison

The slab scene intentionally extrudes the 2D reservoir across depth, uses
free-slip walls and symmetry depth, and disables surface tension. General 3D
sideways spreading should not be used as the explanation for this scene.

Priority for follow-up implementation/measurement:

1. Implemented: valid 2D demotion proofs and direct-phi motion criteria.
   Certificates remain conditional on geometry, volume agreement and safety floors.
2. Capture matched slab 3D pre/post sharpening volume, phi, skipped-face counts,
   phi rung and solver rung at equal physical times. Compare halo volume in
   phi-positive cells and V/phi disagreement separately from contour error.
3. Compare at fixed fine solver resolution to isolate remap/sharpening from
   adaptivity, then repeat adaptively. Retain conservation and connectivity
   constraints when evaluating stronger 3D sharpening.

The initial investigation ran the native 2D probe; it did not run Dawn or a new 3D
simulation, so the 3D mechanism ranking is source-grounded rather than a measured
A/B attribution. The follow-up ran the native regression tests, rebuilt all Wasm variants, and
ran the canonical Dawn gate; see the task result for validation outcomes. The
3D solver implementation was not changed by the follow-up.
