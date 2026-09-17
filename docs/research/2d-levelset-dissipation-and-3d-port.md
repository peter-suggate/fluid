# Reducing 2D level-set dissipation: algorithm changes and 3D port guidance

Recorded 2026-09-17. This documents the Rust 2D LevelSetVolume changes tested
in Advance Lab, including the air-extension correction published to the UI.
It is an implementation and evidence record for the 3D adaptive-volume +
levelset port; it does not claim that port has been implemented or validated.

Subsequent boundary work is documented in
[Ceiling contact and open-boundary sticking](2d-ceiling-contact-and-open-boundary.md).
Catalog tops are now closed. The 2D scalar transport also supplies incoming
ambient air at explicit open boundaries; a clamped departure must not reuse
old wet boundary phi when the flow reverses. Include that boundary contract
in the 3D port, alongside closed-wall release and source precedence.

The main result is substantially better retention of the volume enclosed by
the level-set surface. In the mixed-resolution dam, the latest correction
reduces the frame-120 surface-volume deficit by **78.86%**, while retaining
the same conserved material volume. The first damaging discrepancy occurs
in the velocity field used to advect phi, before sharpening can act.

## Scope, versions, and terminology

There are three successive comparisons, with different baselines:

1. Original coarse-front investigation: replace cell-centred transport with
   direct staggered transport, and repair remeshing/symmetry defects.
2. Wall-residue investigation: permit the liquid level set to make contact
   with initially dry, closed exterior walls.
3. Latest dissipation investigation: make the extended air-face velocity
   satisfy the discrete divergence constraint near the surface. Its baseline
   already includes changes 1 and 2.

Changes 1 and 2 are committed as `3752ee8179dce33cec79f947eb19cd51f7b3dbfe`.
Change 3 is the working-tree implementation in
[`levelset_air_extension.rs`](../../rust/crates/fluid-core/src/levelset_air_extension.rs),
called by `numerics::extend_velocity_with_level_set`. The published WASM
build records source SHA-256
`6f207ca3024883a21adbfccea6748462cbd26a0578b2d1fb31d4a7c83e19a2fc`.
The workspace also contained unrelated changes; these comparisons do not
attribute those changes to this investigation.

For each cell, let `V` be conserved liquid volume, `C` its open capacity, and
`H(phi)` its geometrically integrated phi-liquid volume. Phi is negative in
liquid. Diagnostics below distinguish:

- **Surface volume:** integral of the published phi-liquid region; in 2D,
  this is area. Its loss is the dissipation quantified here.
- **Positive mismatch:** sum of `max(V - H(phi), 0)` over a specified region.
  This includes diffuse material outside the surface, even below capacity.
- **Over-capacity volume:** sum of `max(V - C, 0)`. This is a different defect.
- **Net mismatch:** total `V - H(phi)`, allowing positive/negative local
  errors to cancel. Sharpening can redistribute V but leaves phi unchanged.

The tests do not establish elimination of all kinetic-energy dissipation,
exact level-set volume conservation, or identical trajectories on different
velocity resolutions. The user reports a substantial visible improvement;
the measurements below quantify the surface-retention improvement.

## Before versus after

| Operation | Before | Implemented change and rationale |
| --- | --- | --- |
| Velocity used by characteristics | Face velocity was averaged to cells, extended at cells, then interpolated for transport. A half-cell domain clamp shifted boundary samples inward by a resolution-dependent distance. | Independently sample staggered face components: bilinear interpolation on regular grids, affine MLS using sample-specific widths near adaptive transitions, with boundary ghost samples. This removes an extra filtering stage and the half-cell boundary sampling bias. |
| Velocity extension | Cell-centred values and their support controlled transport. The subsequent direct-face implementation still averaged each face component independently. | Extend faces synchronously from phi-liquid seeds, preserving prescribed walls and liquid faces. The latest change then projects the near-surface **air-only** degrees of freedom to remove extension divergence. |
| Remesh followed by velocity advection | Coarsening could resample the velocity before the next advection gather. | Retain an immutable pre-coarsening face sampler for that next gather. Injection invalidates it. This avoids another avoidable remesh filter. |
| Conservative transport at crowding | Three capacity-balancing sweeps left substantial wall-impact overfill after stronger staggered transport was introduced. | Additional sweeps balance overfull liquid receivers and renormalize donors, up to 64 sweeps or 1e-6 relative excess. Keep the original stencil and donor totals; report infeasible excess. |
| Thin-feature adaptation | Phi-liquid membership was mixed with density-based neighbor exposure. | Use phi membership consistently for geometric thin-feature detection, preventing reflected density residue from selecting different rungs. |
| Sharpening allocation | Sequential donor commits gave early donors priority for limited receiver capacity. | Propose transfers simultaneously and share each receiver's capacity proportionally. Preserve near priority, component boundaries, and conservation. |
| Volume transfer on refinement | The phi/V discrepancy could be assigned to the first child. | Scale geometric child amounts together when reducing them; distribute additions in proportion to remaining capacity. Preserve parent volume without child-order bias. |
| Closed-wall phi | A dry wall stayed positive under tangential backtracing; redistancing preserved that sign even after interior liquid arrived. | Continue negative interior phi onto closed exterior wall vertices before redistancing, with a resolution-independent finest-grid inward sample. Keep solid capacity and separation rules authoritative. |

The staggered sampler is in
[`staggered_velocity.rs`](../../rust/crates/fluid-core/src/staggered_velocity.rs).
Phi backtraces, conservative-volume footprints, and velocity advection use
the immutable face evaluator; cell velocities remain derived data. Retaining
the old sampler is a targeted remesh correction, not an implementation of
the complete Ando–Batty stage ordering.

The wall continuation uses an immutable diagonal interior sample at corners,
so its result does not depend on face order. Open and separating faces are
excluded. Positive interior samples do not erase an existing negative wall
sample; tangential advection can still move contact away. Zero-dt transport
is unchanged. This changes liquid phi at the fluid boundary, not a solid
cell into liquid. See the detailed
[front investigation](2d-split-resolution-dam-front.md) and
[wall-contact investigation](2d-split-resolution-wall-residue.md).

## Latest root cause: air extension compresses the surface

The pressure solve constrains cells selected by phi-liquid **cell-centre**
membership. A contour can pass through an air-centred cell, and interpolation
around it uses extended air faces. Independently averaging the components
does not preserve discrete divergence in those cells. Consequently, a
well-projected liquid face field can still produce an advecting field that
compresses the surface on its air-side support.

This creates the misleading appearance of ineffective coarse-cell
sharpening: phi has already lost target volume. Conserved material cannot
all fit below the unchanged phi-implied targets. Increasing sharpening
iterations cannot resolve that global capacity shortfall.

### First divergence, before downstream repairs

The reproduction uses `sparse-cm12-ladder-symmetric-3d` in **2D**
LevelSetVolume mode, a 32 by 16 finest-cell tank, dt = 1/30 s, 256 pressure
iterations, and relative tolerance 1e-6. The full left half is held at width
2 and right half at width 1. Widths and areas below use finest-cell units;
one finest-cell edge is 0.05 m. Initial conserved volume and surface area
are both 128, with 64 in each half. Repeat with the widths swapped.

The initial field and pre-transport transfers have zero density reflection
error and no phi/V mismatch. Phi area begins to decrease in frame 1; the
fine half initially loses slightly more. The **first coarse-side net deficit
to overtake the fine-side deficit occurs during frame-3 phi advection**:

| Frame-3 stage, baseline before air correction | Coarse net V−H | Fine net V−H |
| --- | ---: | ---: |
| Before transport | 0.741053 | 0.829472 |
| After phi advection | 1.418616 | 1.191769 |
| After redistancing, before sharpening | 1.483259 | 1.244733 |
| After sharpening | 1.461925 | 1.266067 |

Half-domain totals can include transport across the midplane. The frozen
flux experiment therefore closes each half-domain with its midplane flux;
it does not mistake crossing the middle for loss of total liquid area.

On the exact frame-3 input, keep the previous phi, graph, and velocity field
fixed, and apply only the air correction to a clone. Integrating the sampled
velocity through each closed half-liquid boundary gives:

| Frozen frame-3 field | Coarse outward flux | Fine outward flux |
| --- | ---: | ---: |
| Independent face extension | −10.711475 | +0.779568 |
| Same field plus air-only projection | −0.194030 | +0.884803 |

Flux units are finest-cell area per second; negative outward flux means
contraction. All liquid-touching face values remain unchanged. Replaying
one scalar RK2 trace on that same frozen state gives:

| Phi area | Coarse | Fine |
| --- | ---: | ---: |
| Before trace | 63.258606 | 63.170869 |
| Original trace | 62.600883 | 62.788731 |
| Trace with projected air velocity | 63.116490 | 62.997767 |

This intervention improves area retention **before redistancing or
sharpening**, without rerunning the simulation history. A face-based RT0
interpolation control on the original field still gives coarse flux
−10.75552: merely replacing the interpolant does not remove the bad flux.
These controls locate the causal defect in extended velocity, rather than
inferring it from the final residue image.

## Air-only projection algorithm

The production sequence is now:

```text
liquid pressure solution / support transition
    → synchronous staggered face extension
    → constrained air-band projection
    → immutable staggered characteristic sampler
    → conservative V transport and phi advection
    → closed-wall phi continuation and redistancing
    → conservative sharpening against phi targets
```

The projection is inside `extend_velocity_with_level_set` whenever extension
depth is nonzero, so each call rebuilding that field receives the correction.
The final extension immediately before scalar transport is the decisive
call for the reproduced defect.

1. Select positive-capacity air cells with
   `0 < phi_i <= 2 * max(hx_i, hy_i)`. The band covers surface interpolation
   support; it is not a material relocation radius or a density threshold.
2. A face is free only when it is not `ClosedWorld`, its open fraction
   exceeds 1e-8, and **all incident cells** have positive phi and capacity.
   Thus every liquid-touching face and prescribed closed wall is immutable.
3. Build flux residuals from the existing graph coefficients and metric
   weights, including contributions from fixed faces.
4. Solve a scalar correction system on active air cells. With `B` the
   face-by-cell graph coefficient matrix, `W` the diagonal dual weights,
   and `A` the diagonal face open fractions, the anchored system is
   `B_freeᵀ W A B_free p = Bᵀ W u` and the free-face update is
   `u_new = u - A B_free p`. Columns outside the active band have correction
   zero. The RHS includes fixed boundary flux; the matrix does not permit
   changing those boundary velocities.
5. Detect connected components and their pressure nullspaces. For an
   enclosed component without a free boundary anchor, fixed liquid/wall
   faces may impose nonzero net flux. Subtract `d_bar * cellMeasure_i` from
   each RHS, where `d_bar = sum(component flux) / sum(component measure)`.
   This retains the imposed uniform divergence while removing its spatial
   variation. Do not pin an arbitrary cell or silently change wall flux.
6. Use Jacobi-preconditioned CG in f64, capped at 512 iterations. Stop when
   `rᵀ M⁻¹ r <= max(initial_rᵀ M⁻¹ initial_r * 1e-16, 1e-24)`.
   Apply corrections to a cloned f32 face vector and validate finiteness
   before committing. Zero-diagonal cells are excluded and counted.

`AirExtensionReceipt` reports constrained cells, corrected faces, iterations,
initial divergence, final error relative to compatible divergence, compatible
divergence, and isolated cells. The current caller does not publish this
receipt to the UI. The implementation returns a receipt after a finite
iteration/breakdown stop; it does not currently fail solely for missing the
residual target. That diagnostic should be observable in a 3D port.

This is a discrete air-flux correction. It does not guarantee pointwise
divergence-free MLS interpolation, and finite-step scalar advection and
redistancing can still change area. No phi-volume offset, volume deletion,
larger sharpening radius, or changed liquid-pressure membership is part of
the latest correction.

## Measured improvement

The following native measurements compare the **wall-fixed baseline** with
the air-projected implementation. They cover the whole left/right halves,
not the narrower wall strips used in the previous wall report.

| Diagnostic, width 2 left / width 1 right | Before air correction | After air correction |
| --- | ---: | ---: |
| Frame 60 surface area | 110.191772 | 123.202856 |
| Frame 60 coarse positive mismatch | 10.200282 | 2.438110 |
| Frame 60 fine positive mismatch | 7.653644 | 2.744508 |
| Frame 120 surface area | 100.459836 | 122.179324 |
| Frame 120 deficit from conserved 128 | 27.540164 | 5.820676 |
| Frame 120 coarse positive mismatch | 20.942306 | 3.023643 |
| Frame 120 fine positive mismatch | 6.713890 | 3.328330 |
| Frame 120 total over-capacity volume | 0.016562 | 0.012938 |
| Conserved volume | 128 | 128 |

At frame 120, coarse positive mismatch falls **85.56%**, fine mismatch
**50.43%**, and total surface deficit **78.86%**. Surface area retained rises
from 78.48% to 95.45% of initial area. Across both updated 120-frame native
runs, conserved volume remains within approximately 1.3e-12 of 128.

Swapping the held widths gives coarse/fine positive mismatch
3.023638 / 3.328291 and surface area 122.179366 at frame 120. The close
agreement supports a resolution-related correction without a left/right
preference; it is not bitwise reflection or proof of identical coarse/fine
dynamics. Small remote residue remains, and the coarse/fine mismatch is not
exactly equal.

Earlier improvements should retain their own baselines: the first staggered
transport change reduced the frame-6 front-advance gap from 1.399655 to
1.043066 cells; the later wall-contact fix reduced frame-60 fine-wall overload
from 2.434560 to 0.003644. Those are historical measurements, not current
front-gap results. The latest trajectory changes are reported below.

## Validation performed and remaining failures

| Check on the air-corrected implementation | Result / assertion scope |
| --- | --- |
| Air-extension integration tests | 5 pass: widths 1, 2, 4 and mirrored 1/2 splits; initial divergence >0.1 reduced below 2e-5; exact preservation of liquid faces; approximate idempotence; stationary/liquid-only identity; closed-component compatibility; solid faces; partial apertures; reflection. |
| Split dam through frame 120, both orientations | Pass: contact timing, pressure membership, conservation, no faults, wall overload, and whole-half positive mismatch <4 per half with half difference <0.75 at frame 120. |
| Core library | 203 pass, 1 ignored, 2 long CPU compatibility tests filtered out. |
| Adaptive symmetry / ceiling separation / hydrostatic | 2 / 1 / 3 tests pass. |
| Injection / pressure / sharpening / mini32 energy | 3 / 1 / 8 / 1 tests pass. Mini32 energy previously failed; its limits were not loosened. |
| Staggered velocity suite | 7 pass, 1 fails: frame-5 pre-impact coarse/fine front gap 1.2118759 exceeds the existing 1.1 limit. |
| Transition energy suite | 2 pass, 1 fails: quarter-pool frame-5 cell count 36 versus expected 22. |
| Volume regression suite | 10 pass; the two previously recorded stationary mixed-grid identity and diagonal seam-translation failures remain. |
| WASM build and artifact checks | Scalar, SIMD, and threaded builds pass; source fingerprints and CPU features checked. Served UI binaries match the rebuilt files. |
| WASM scene probe | All three variants produce exactly matching recorded publications: unconstrained 60 frames, equal widths 1/1 and 2/2 for 20 frames, both mixed orientations for 120 frames. Checks finite fronts, no faults, volume drift <1e-4 and sampled mixed-run overload <0.1. |

Native and WASM need not be bitwise identical. For example, width-2/1
frame-120 surface area is 122.179323946 native versus 122.179316819 WASM.
Exact parity above refers to comparisons **between the three WASM variants**.

The canonical Dawn gate has **not been rerun for the latest air correction**;
the browser was handed to the user for testing, and repository guidance
forbids simultaneous Dawn/browser simulation. The preceding wall-fixed gate
passed 8/17 with nine recorded failures; see the wall report for its lanes
and receipts. Neither that old result nor the focused native checks imply a
green current gate. No current front-gap or cell-count expectation was
changed to make these failures pass. There is no controlled performance
benchmark establishing the cost of the added projection yet.

Two diagnostic alternatives were not adopted. Extending sharpening reach or
allocation work addresses later redistribution, not the measured loss of
surface targets. Geometrically integrating donor material improved a
manufactured translation test but worsened this dam's frame-120 surface
area to 99.032726, so that prototype was removed. A separate manufactured
sharpening-only test still exposes resolution-dependent eligibility/reach
limits. The present result must not be described as a universal repair of
the sharpening operator.

## Relation to Losasso and Ando–Batty

The repository copies were the references used for the transport comparison:

- [Losasso, Gibou and Fedkiw (2004)](../papers/losasso-2004-octree-water-smoke.txt),
  sections 3 and 6: face averaging under coarsening, nodal velocity
  construction/extrapolation, and particle-level-set surface correction.
- [Ando and Batty (2020)](../papers/ando-batty-2020-practical-octree-liquid-simulator.txt),
  section 5, especially equations 33–35: independent staggered-component
  MLS interpolation avoids temporary nodal/cell-centred filtering. Section
  7.1 describes iterated averaging for extension; the paper also describes
  global/regional volume correction.

These support retaining staggered velocity information and treating
remeshing/interpolation carefully. Neither paper specifies our exact
air-only projection or hybrid conservative-V sharpening algorithm. The
projection is motivated by the measured extension-flux defect and validated
by frozen-input intervention. It should not be attributed to either paper
as a verbatim algorithm. Both papers have additional surface-volume machinery
that is not equivalent to this implementation.

## What to carry into 3D adaptive-volume + levelset

This is a port plan, not additional work performed in this change. The Rust
helper currently rejects dimensions other than 2. The GPU resident uses a
separate implementation, so rebuilding Rust does not update 3D behavior.
Read this alongside the existing
[3D implementation handoff](level-set-volume-3d-implementation-handoff.md).

The key invariant is: **the velocity actually sampled by phi and V must
retain the pressure solution and have compatible air-side flux before
either field is transported.** Porting a final residue cleanup alone would
miss the measured cause.

1. **Trace the actual GPU consumer first.** In
   [`webgpu-sparse-cm12-resident.wgsl.ts`](../../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts),
   `sampleEffectiveTransportVelocityAtSpansMode` currently interpolates
   `cm12EffectiveTransportVelocity(cell)` at a cell-centred lattice and
   clamps to half the requested spans. Separate face-support samplers also
   exist. Audit the bindings and call path used by
   [`levelset-volume-core.wgsl.ts`](../../lib/methods/adaptive-volume/levelset-volume-core.wgsl.ts).
   Correcting an unused face bank will not correct the advecting field.
2. **Port staggered sampling and extension as a coherent path.** Use
   independent u/v/w face components, trilinear regular-grid sampling and
   tested adaptive MLS support. Preserve wall-normal velocity and tangential
   continuation, immutable liquid seeds, and pre-remesh sampling where
   applicable. Test coarse/fine faces, edges, and corners in all axes.
3. **Add the air correction between the final extension and transport.**
   Use the accepted compiled topology's gradient/divergence adjoint pair,
   3D dual weights, face apertures, and cell volumes. Do not transpose 2D
   constants or apply an extra area/open-fraction factor. Preserve every
   liquid-touching and prescribed solid face, including moving solids.
4. **Plan sparse support explicitly.** Translate the 2D support-band intent
   into coverage of the actual 3D interpolation and RK2 stencils. Missing
   bricks are not automatically valid zero-pressure air. Detect open
   anchors, enclosed components, isolated cells, and compatible net flux.
   Report divergence and convergence before relying on the visual result.
5. **Preserve the earlier geometry and symmetry contracts.** Phi supplies
   surface geometry; solid capacity supplies solid exclusion. Audit dry-wall
   contact, separating-wall precedence, order-independent transfer, and
   simultaneous sharpening under GPU execution. Extend wall-contact tests
   to six faces, edges, corners, cut cells, and moving boundaries rather than
   blindly copying the 2D finest-grid boundary sampling implementation.
6. **Validate the first divergence in 3D.** Freeze a pressure/extension state;
   measure liquid and air discrete divergence separately, integrated
   surface-normal flux, and enclosed surface volume after advection,
   redistancing, and sharpening. Apply only the air correction to the frozen
   copy. Include a thin extruded 2D control and genuinely 3D scenes before
   making an equivalence claim.
7. **Run the full acceptance matrix.** Include equal and mixed rungs,
   reflected and rotated splits, unconstrained symmetry, stationary and
   affine fields, incompressible translation, hydrostatics, wall impact,
   trapped air, disconnected droplets, live solids/liquid insertion, and
   long-horizon phi/V mismatch. Compare mass, surface volume, energy,
   divergence, residue, timings, and memory. Run the canonical isolated
   `npm run test:dawn:sparse-cm12`; retain every existing limit and report
   unresolved failures. Establish the GPU solve cost and convergence policy
   rather than adopting the CPU's 512-iteration cap as a performance budget.

## Reproduction and evidence

Run from the repository root. These commands exercise the current
implementation; recreating the baseline requires the captured baseline
inputs or the corresponding earlier solver state.

```bash
cargo test --manifest-path rust/Cargo.toml -p fluid-core --test levelset_air_extension --test levelset_volume_wall_contact
FLUID_SPLIT_FIELDS_TRACE=1 cargo run --manifest-path rust/Cargo.toml -p fluid-core --release --example investigate_split_levelset -- rust/core/testdata/split-resolution-ladder-seed.json 2 1 120 1e-6 > /tmp/air-fixed-21.jsonl
FLUID_SPLIT_FIELDS_TRACE=1 cargo run --manifest-path rust/Cargo.toml -p fluid-core --release --example investigate_split_levelset -- rust/core/testdata/split-resolution-ladder-seed.json 1 2 120 1e-6 > /tmp/air-fixed-12.jsonl
npm run build:physics-wasm
npm run check:physics-wasm
node --import tsx tools/wasm/split-resolution-levelset-probe.ts /tmp/air-fixed-wasm.json
```

Local evidence is under
[`artifacts/level-set-volume/sharpening-resolution/`](../../artifacts/level-set-volume/sharpening-resolution/).
This is an ignored artifact directory; the tables in this document preserve
the principal results in tracked documentation. Archive the raw directory
with the port handoff if working in another checkout.

| File | Content |
| --- | --- |
| `measurements.json` | Baseline 2/1 and updated 2/1, 1/2; every frame 0–120, half-domain mismatch, volume, fronts, and transport receipts. |
| `first-divergence-stages.json` | Seven baseline transport steps, initial / after phi advection / before sharpening / after sharpening. Three extra frame-1 diagnostic replays are excluded. |
| `frozen-input-frames0-7.jsonl` | Baseline graphs, fields, and phi for causal replay. |
| `frozen-flux-before.jsonl`, `frozen-flux-air-projected.jsonl`, `frozen-flux-rt0-control.jsonl` | Closed-half flux comparison and interpolation control. |
| `frozen-phi-air-projected.jsonl` | Frozen characteristic replay before redistancing/sharpening. |
| `fixed-wasm.json` | All three WASM variants and all five resolution configurations. |
| `sharpening-only-before.jsonl`, `translation-before.jsonl`, `translation-geometric-control.jsonl` | Separate manufactured controls, including the unshipped geometric-remap experiment. |
| `validation/` | Saved native test, WASM build, and artifact-check logs. |

[`investigate_surface_flux.rs`](../../rust/crates/fluid-core/examples/investigate_surface_flux.rs)
accepts the frozen input path and optional `--project-air` (or `rt0` as its
second argument). It integrates the same previous surface against the
captured extension field.
[`investigate_wall_characteristics.rs`](../../rust/crates/fluid-core/examples/investigate_wall_characteristics.rs)
accepts the frozen input path, characteristic substep limit, and optional
`--project-air` for scalar-trace replay.
