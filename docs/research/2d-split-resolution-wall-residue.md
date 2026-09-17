# Split-resolution level-set wall residue

## Implemented fix and measurements (2026-09-17)

`levelset_volume.rs::continue_phi_onto_closed_walls` now supplies liquid phi
at closed exterior walls after advection and before redistancing. A boundary
vertex uses negative interior phi from one **finest-grid** interval inward,
independent of the adjacent adaptive cell width. Closed corners use an
immutable diagonal interior sample, making the operation independent of face
ordering. Only represented closed faces next to positive-capacity cells
participate. Open faces are excluded; separating faces and the existing
separation carve take precedence. Positive interior phi does not overwrite
an existing negative wall value, preserving thin resting films. Tangential
advection still permits contact to recede. Zero-dt transport is unchanged.

This changes the liquid level set at the boundary, not the classification of
the exterior solid or the conserved volume. The fine corner is phi-liquid
at published frame 7 (phi −0.533269), participates in pressure at frame 8,
and has frame-8 density **1.033779**, versus **2.218604** before the fix.

Measurements use the scene and physical wall strips defined below, with
the same dt, pressure settings, and held cell widths as the baseline:

| Measurement | Before | Implemented fix |
| --- | ---: | ---: |
| Frame 10, total over-capacity volume | 4.122184 | 0.026384 |
| Frame 20, fine-wall over-capacity volume | 1.357262 | 0.000313 |
| Frame 60, fine-wall over-capacity volume | 2.434560 | 0.003644 |
| Frame 120, fine-wall over-capacity volume | 3.462632 | 0 |
| Frame 120, fine-wall positive phi mismatch | 23.600425 | 0.738656 |
| Frame 120, total phi area | 67.124879 | 100.369953 |
| Conserved volume | 128 | 128 |

Frame-60 fine-wall overload falls by 99.85%. Swapping the held widths gives
the same sampled fine-wall overloads and zero overload on both walls at
frame 120. The long trajectories are not bitwise reflections: frame-120
total phi area is 100.459836 in the swapped run. Long-term phi-area loss
remains despite conserved V; this fix addresses the false wall-air gap and
the overload it causes, not every phi/V mismatch.

Validation:

- New 120-frame integration test covers both split orientations, contact
  timing, pressure membership, mass, faults, and both physical wall strips.
- Three new unit tests cover every wall orientation, corners, adaptive
  rungs, face ordering, open/separating patches, subcell films, and tangential
  recession. Existing ceiling-separation and hydrostatic tests pass.
- Rust library: 203 passed, 1 ignored, 2 long CPU compatibility tests
  filtered out. Adaptive symmetry, adaptivity, ceiling separation,
  hydrostatic, injection, pressure, sharpening, transition energy, wall
  velocity, and staggered-velocity integration suites pass.
- The pre-impact front comparison now samples frame 5 rather than frame 6:
  the new wall continuation first affects this scene by frame 6. Its
  existing 1.1 gap limit is unchanged (frame-5 gap 1.01805); the frame-6
  reflected-split checks remain. This is an explicit measurement-window
  change, not evidence of unchanged frame-6 fronts.
- Native release and scalar/SIMD/threaded WASM builds and artifact checks
  pass. All three WASM variants match exactly across unconstrained,
  equal-resolution, and 120-frame split probes. Native/WASM front positions
  match exactly; maximum volume difference is 8.53e-13. The swapped run's
  maximum phi-area difference is 0.000256 and over-capacity difference is
  2.99e-6; native/WASM fields are not claimed bitwise identical.
- Broader Rust checks are **not clean**. The two previously observed
  `levelset_volume_regression` failures remain (zero-velocity mixed-cell
  identity and diagonal seam translation). `levelset_volume_mini32_energy`
  now stops at frame 8 on mass: 588.799987807286 versus
  588.7999999523163, a relative drift of 2.063e-8 against a 2e-8 limit.
  Before this wall fix that lane already failed its frame-9 energy check;
  the current earlier mass failure is recorded separately. No tolerance
  or timing ceiling was raised.
- Required isolated `npm run test:dawn:sparse-cm12`: **8/17 passed** in
  367.33 s, within the 480 s suite budget. The same nine lane IDs failed as
  the recorded pre-wall-fix gate. Five timed out: symmetric expansion,
  topology page budget, clipped topology transfer, mini32 correctness,
  and mini32 performance. Four failed assertions/runtime checks:
  min8-region surface (split height 0.100545 versus 0.01 limit), long-dam
  initial rendering topology (66 versus 106 expected pages), and missing compiled topology faces in the hillside
  and outside-tank collapse lanes. Hydrostatic/adaptivity, mini64 surface,
  and live rigid/liquid insertion passed; mini64 median advance was
  53.412 ms against its 110 ms ceiling. The gate remains **not green**.
  Full receipt: `artifacts/level-set-volume/wall-residue/fixed-dawn.json`.

Reproducible tests: `rust/crates/fluid-core/tests/levelset_volume_wall_contact.rs`
and `tools/wasm/split-resolution-levelset-probe.ts`. Receipts are in
`artifacts/level-set-volume/wall-residue/fixed-native.json`, `fixed-wasm.json`,
`native-wasm-deltas.json`, and `validation/`. The following investigation
sections describe the pre-fix solver and isolated diagnostic controls.

From the repository root, reproduce the new regression and full native trace:

```bash
cargo test --manifest-path rust/Cargo.toml -p fluid-core --test levelset_volume_wall_contact
FLUID_SPLIT_FIELDS_TRACE=1 cargo run --manifest-path rust/Cargo.toml -p fluid-core --release --example investigate_split_levelset -- rust/core/testdata/split-resolution-ladder-seed.json 1 2 120 1e-6 > /tmp/wall-contact-12.jsonl
node --import tsx tools/wasm/split-resolution-levelset-probe.ts artifacts/level-set-volume/wall-residue/fixed-wasm.json
```

Swap `1 2` to `2 1` for the reflected native run. Build the WASM artifacts
before running their probe; do not run Dawn concurrently with a browser
simulation or another Dawn process.

## Deeper cause: an initially dry wall cannot become phi-liquid

Follow-up investigation isolates the origin of the fine-side penalty. The
8-layer extension cutoff and unavailable sharpening receivers described below
explain persistence, but the earlier contact defect creates the large residue.

At a closed side wall the staggered normal velocity is zero. A phi sample on
that wall therefore backtraces along the wall. Every initial side-wall phi
sample is positive in this central dam scene; scalar interpolation of those
samples stays positive, and redistancing preserves its sign. No code supplies
liquid phi boundary values when the advancing fluid contacts the wall.
**All side-wall vertices remained positive in all 121 snapshots, frames 0–120.**
The smallest left/right wall phi values were +0.0000133795 / +0.00376129.
The level-set contour can approach the wall, but cannot attach to it.

This boundary condition leaves a false air gap where conservative V accumulates.
Pressure classifies the adjacent *open-domain* cell by its center phi. This
cell is not the solid itself: on the fine left side it spans [0,1] × [0,1],
has capacity 1, and has closed bottom/left faces of aperture zero.

### Why width 1 is penalized more than width 2

The velocity falls from the incoming value to zero across the first cell's
width h. The unconstrained air-side compression therefore contains a term
−U/h. At frame 8, fine/coarse adjacent-cell divergence is −64.888 / −31.254
per second; both cells are excluded from pressure in that step. The incoming
horizontal face speeds are 60.043 / 53.018 finest cells per second, giving
U dt/h = **2.001 / 0.884** at the same dt = 1/30 s.

The false air gap causes a much larger overload in the smaller receiving cell:

| Published frame | Fine corner density | Fine center phi | Coarse corner density | Coarse center phi |
| --- | ---: | ---: | ---: | ---: |
| 7 | 1.150142 | +0.202313 | 0.375156 | +0.382690 |
| 8 | 2.218604 | −0.074537 | 0.916065 | −0.134759 |

Both centers first become liquid at the end of frame 8, so their normal
liquid pressure response first occurs in frame 9. The fine cell has already
accumulated over twice its capacity; the coarse cell remains below capacity.
The excess-pressure feedback then contributes to an upward jet on the fine
wall. In frame 9 its bottom cell's upward face speed is 68.087, compared with
45.902 on the coarse side. This transports residue up the wall; the earlier
report documents why it later becomes immobile.

### Controlled intervention

In an isolated copy only, after phi advection, set a side-wall vertex to the
adjacent interior vertex value **when that interior value is negative**.
This permits phi-liquid wall contact. No volume clipping, extension-depth
change, sharpening change, timestep change or extra pressure solve is made.
This is a diagnostic boundary-condition intervention, not a validated general
wall-contact implementation.

The fine corner becomes phi-liquid at frame 7 (phi −0.566914), and therefore
participates in the frame-8 pressure solve. Its frame-8 density is 1.016800
instead of 2.218604. The coarse corner remains below capacity in either run.

| Measurement | Pre-fix solver | Diagnostic wall-contact intervention |
| --- | ---: | ---: |
| Frame 10, total over-capacity volume | 4.122184 | 0.023702 |
| Frame 20, fine-wall over-capacity volume | 1.357262 | 0.000280 |
| Frame 60, fine-wall over-capacity volume | 2.434560 | 0.003859 |
| Frame 120, fine-wall over-capacity volume | 3.462632 | 0 |
| Frame 120, fine-wall positive phi mismatch | 23.600425 | 0.789187 |
| Frame 120, total phi area | 67.124879 | 99.590982 |
| Conserved volume | 128 | 128 |

There is still long-term phi-area loss in the intervention. It isolates the
wall-contact defect, rather than solving every phi/V conservation problem.

### Alternative explanations checked

- **Coarse/fine junction:** not required. With uniform width 1, frame-10 total
  over-capacity is 8.286569 (symmetric deposits at both walls); uniform width 2
  gives 0.00002384. Frame-20 phi areas are 92.507150 versus 115.926649.
- **Characteristic integration accuracy alone:** not sufficient. Holding the
  exact frame-8 production face field and previous phi fixed, changing one
  RK2 trace to 32 smaller characteristic steps changes fine-half advected
  area from 59.05405 to 59.07474. The coarse-half areas are 60.24912 and
  60.30969. This does not close the fine-side loss. No field re-solves or
  repeated scalar resampling occur in this control.
- **Whole-simulation timestep alone:** at t = 2/3 s, halving/quartering dt
  leaves fine-wall over-capacity of 1.12040 / 1.09259, versus 1.35726 at
  dt = 1/30 s. The defect remains.
- **Excess-pressure feedback as the original cause:** disabling it still
  produces the defect, with frame-20 fine-half phi area 43.16510 versus
  45.57864 normally. It amplifies upward motion after overload but did not
  create the initial contact mismatch.
- **Earlier pressure support:** enabling the existing swept-wall support
  also greatly reduces overload, but alters pressure support, free-surface
  coefficients and dynamics broadly. It is a less isolated control and is
  not proposed as a repair here.

This isolated the **level-set/solid contact boundary treatment and its pressure
timing** as the implementation target addressed above. The previously
identified support/conservation issues remain separate. Classifying solid
material as liquid or merely raising the sharpening iteration count would
miss this cause.

Evidence: `artifacts/level-set-volume/wall-residue/wall-contact-controls.json`,
`characteristic-controls.jsonl`, and `contact-control-only.patch` (unapplied).
The frozen-field replay is in
`rust/crates/fluid-core/examples/investigate_wall_characteristics.rs`.
The split probe now accepts an optional timestep after pressure tolerance.


Original investigation: 2026-09-17 against the rebuilt Rust solver following
the symmetry fixes, before the wall-contact fix. Internal instrumentation
ran in a separate copy of the Rust tree.

## Reproduction and direction control

Scene: `sparse-cm12-ladder-symmetric-3d`, 2D LevelSetVolume, dt = 1/30 s,
256 pressure iterations, relative tolerance 1e-6. Hold the full left half at
width 1 and right half at width 2, then repeat with the widths swapped.
Both runs were measured through frame 120. The deposit follows the width-1
side; swapping regions reflects the measured behavior. It is not specific to
the left wall.

All areas/volumes below are 2D finest-cell area units. Wall strips are the
same physical width on both sides: x in [0,2] and [30,32]. “Positive phi
mismatch” is the sum of max(conserved cell volume − integrated phi-implied
cell volume, 0), not just over-capacity volume.

| Frame | Fine-side wall positive phi mismatch | Coarse-side wall positive phi mismatch | Fine-side wall over capacity | Coarse-side wall over capacity |
| --- | ---: | ---: | ---: | ---: |
| 10 | 7.3368 | 3.3889 | 3.9540 | 0 |
| 20 | 15.7483 | 5.9811 | 1.3573 | 0 |
| 60 | 18.9450 | 5.7994 | 2.4346 | 0 |
| 120 | 23.6004 | 11.8699 | 3.4626 | 0 |

Thus the coarse side also has diffuse volume/phi disagreement, but not the
same over-capacity wall deposit in these samples. Neither side loses conserved
volume: the whole tank retains 128 throughout the forced-resolution run.

## Creation: the independently advected surface loses area at impact

Phi and conservative V use the same velocity sampler but different numerical
updates: vertex semi-Lagrangian phi advection and a conservative cell-volume
remap. Sharpening leaves phi immutable. In this run their represented volumes
separate substantially at wall impact.

By frame 20 the fine half has conserved V = 63.24339 but phi area = 45.57864,
from initial values of 64 each. The coarse half has V = 64.75661 and phi area
= 58.03025. Stage instrumentation attributes the fine-half cumulative area
change to −16.34524 during advection and −2.07612 during redistancing; the
coarse-half changes are −4.65316 and −1.31659 respectively. These are net
half-domain changes (which can include crossing the middle), not separate
Lagrangian mass ledgers. Their whole-domain sum establishes actual phi-area
loss. The strong fine-side deficit is measured before the later frozen state.

Relevant implementation: `levelset_volume.rs::advect_shared_phi_with_velocity`
(lines 255 onward), `redistance_vertices` (285 onward), and the separate
conservative gather/phi update/sharpening sequence (875–1000).

## Persistence: dry velocity extension erases gravity outside its support

`world.rs` calls `extend_velocity_with_level_set(..., 8)` immediately before
transport. `staggered_velocity.rs::extend_faces` fixes phi-liquid faces, clears
other open-face values to zero, and extends known values for only eight
neighbor layers. This is a count of adaptive cells, not a fixed physical
coverage requirement for all cells that hold V. On width-1 cells the physical
coverage is smaller than on width-2 cells. The exact support also depends on
face connectivity and component.

The fine-side cell centered at (0.5, 12.5) contains density **1.170117735862732**
unchanged from frame 80 through 120. It is phi-air, has pressure membership 0,
and published cell extension depth 255 (unreached). At frame 100 the actual
production stage trace is:

| Stage | Lower vertical face | Upper vertical face | Density |
| --- | ---: | ---: | ---: |
| Face preparation | 0 | 0 | 1.170117735862732 |
| Body forces | −6.537766933 | −6.537766933 | 1.170117735862732 |
| Velocity projection | −6.537766933 | −6.537766933 | 1.170117735862732 |
| Final level-set velocity extension | 0 | 0 | 1.170117735862732 |
| Conservative transport | 0 | 0 | 1.170117735862732 |

This identifies the stage that removes the downward motion. The over-capacity
pressure source does not rescue this phi-air cell: its pressure membership
and RHS are both zero in the trace.

A counterfactual on cloned frame-100 fields changes only face-extension depth:
8 layers gives sampled velocity (0,0) at (0.5,12.5); 16 and 32 layers both give
(−2.362130642, 8.551567078). The coarse-side sample at (31.5,12.5) remains
(0.975727260, 3.938370943) for all three depths. This verifies the support hole;
it does **not** establish that widening extension alone is a physical repair.
The replays use the published state and current phi, separately from the
production stage trace, and do not modify subsequent frames.

## Why sharpening cannot remove it

There are two independent restrictions:

1. `levelset_adaptive_distance.rs::RETURN_REACH` is 8 finest cells. A donor
   beyond that graph distance has no component assignment and is excluded
   from sharpening. At frame 60, the cell (0.5,12.5) has V = 1.333731532,
   phi = +8.3879013, target V = 0, no finite auxiliary distance, and no eligible
   donor stencil. Its volume is unchanged by sharpening.
2. Even eligible donors may only send to same-component cells with
   `residual < 0 && target > 0` (`levelset_sharpening.rs`, lines 228–239).
   This means room below the phi-implied target, not all empty tank capacity.
   At frame 60 total conserved V is 128 but the sum of targets is 83.769895913.
   Before sharpening the whole domain has only 0.399286211 of target deficit.
   Sharpening relocates exactly that amount; afterward there is no target
   deficit left. The remaining positive mismatch is **44.230104087**. Moving
   donors in another order or doing more rounds cannot fit it into these
   unchanged targets.

Lower wall donors are not rejected by the small-island gate: at frame 60
those in the band belong to a 41.336346-unit donor island, well above 0.5.
They have no eligible receiver in their stencils. This distinguishes the
observed failure from that separate small-residue safeguard.

## Implication for a repair

The present hybrid contract permits V to remain where phi says air, while
both motion support and sharpening eligibility can disappear there. A repair
must address the phi/V mismatch and guarantee a defined evolution/return
path for all retained V. Merely increasing sharpening rounds or extension
layers does not fix the measured global target deficit. Deleting the excess
would violate conservation. This original persistence investigation preceded
the contact diagnosis and numerical repair recorded at the top of this report.

## Reproduction and evidence

These commands produced the original traces on the pre-fix source. On the
current source they exercise the implemented wall-contact treatment.

```sh
cd rust
cargo build -p fluid-core --example investigate_split_levelset
FLUID_SPLIT_FIELDS_TRACE=1 target/debug/examples/investigate_split_levelset \
  core/testdata/split-resolution-ladder-seed.json 1 2 120 > /tmp/wall-12.jsonl
FLUID_SPLIT_FIELDS_TRACE=1 target/debug/examples/investigate_split_levelset \
  core/testdata/split-resolution-ladder-seed.json 2 1 120 > /tmp/wall-21.jsonl
FLUID_SPLIT_FULL_TRACE=1 FLUID_SPLIT_STAGE_FRAME=100 \
  target/debug/examples/investigate_split_levelset \
  core/testdata/split-resolution-ladder-seed.json 1 2 100 > /tmp/wall-stages.jsonl
FLUID_SPLIT_EXTENSION_TRACE=1 target/debug/examples/investigate_split_levelset \
  core/testdata/split-resolution-ladder-seed.json 1 2 120 > /tmp/wall-extension.jsonl
```

Captured evidence is in `artifacts/level-set-volume/wall-residue/`:

- `measurements.json`: both orientations, every frame 0–120.
- `frozen-cell-frame100.json`: exact production stage values.
- `extension-replay.json`: cloned-field support controls.
- `phi-stage-areas.json`: area before advection, after advection, after redistance.
- `sharpening-frame60.json`: donor eligibility, targets, receiver stencils and
  before/after volume.
- `diagnostic-only.patch`: instrumentation used in an isolated copy for the
  internal phi/sharpening measurements; it is not applied to the solver.

The example compiled and all four original probe runs completed. That
diagnostic-only phase did not require a browser rebuild or numerical
regression gate. The subsequent implementation's builds and checks are
reported at the top of this document.
