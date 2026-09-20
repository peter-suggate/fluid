# Uniform Geometric 2D migration — implementation status

The first interactive integration is complete: bare `/advance-lab` now runs
Uniform Geometric through Rust/Wasm. The user explicitly deferred solve-window
scheduling and the full set of stage options to prioritize a usable default UI.
The lab resolves the shared 3D defaults with exactly one override,
`activeRegion: "off"`, and identifies this as a whole-domain solve.
The production 3D method retains its numerical defaults, except that the
unused optional liquid-capacity-balancing feature has been removed.

This is acceptance of the scoped UI integration, not a claim that every 3D
option, source, material-force profile or live-edit operation has exact parity.

## Implemented

- Backend-neutral parameter declarations, canonical defaults, normalization,
  legacy-key removal, and GPU option adapters. The production Uniform
  Geometric method uses this schema. Integer controls normalize before crossing
  the backend boundary. Geometric sharpening strength uses its own declared
  range, including zero, rather than the density method's minimum of 0.25.
- Shared pipeline controls and phase labels now live outside the GPU solver.
  Shared initial-volume construction preserves the 3D seed and wet-bound rules.
  Hierarchy/window planning is a pure TS module, available to both hosts.
- Shared pressure schedule constants and lagged-budget policy. Rust types,
  defaults, ranges, choices and pressure constants are generated from the TS
  contract. The Wasm build checks generated-file freshness.
- Removed balancing controls, GPU option/state fields, dispatches, shader
  entry points, dedicated indirect buffer, and two balancing-only tools.
  Exact donor sums, fallback, three normalization rounds, gather, dust and
  sharpening remain. Reserved scratch offsets remain stable.
- A dense Rust backend: four-corner-plus-identity conservative
  transport with exact donor accumulation; vertex phi RK2/contact treatment;
  redistance; eight conservative sharpening rounds; extension hierarchy and
  two-level sampling; semi-Lagrangian and bounded MacCormack velocity, including
  liquid-only physical gathers and predictor extension; pressure hierarchy and
  compensated-f32 coarsest solve; projection and four-face release bits.
- Transport tile scheduling skips traces, edge construction, fallback and
  normalization outside the live map, and clears gather outputs there. A zero
  dust floor or disabled two-level sampler restores dense transport.
- A common native/Wasm whole-scene diagnostic runner, plus an owned Uniform
  session in `FluidWorld`. Both execute the same numerical world. The owned
  session validates command ordering and publishes revisioned binary fields.
- The existing worker/client protocol now carries Uniform publications without
  fabricating adaptive topology. The controller copies view arrays before
  releasing pooled publication buffers. Scalar and SIMD artifacts are rebuilt.
- The UI uses the shared scene catalog and shared initial V/phi builders to
  sample a central XY slice. It offers play/pause, step, reset, scene selection,
  six field views, grid, fit/zoom/pan and cell inspection. The timestep selector
  spans 1/15 to 1/120 s and changes the next advance without reseeding.
  Stage descriptions and
  read-only defaults come directly from the shared pipeline and parameter schema.
- Bare URLs select Uniform; explicit adaptive method or legacy transport URLs
  retain the adaptive lab. Method switching and history navigation dispose the
  previous worker. Rigid-body and inflow scenes are marked unavailable.
- GPU/native scene comparison at frames 1, 5 and 30, plus one-step comparisons
  initialized from identical GPU V, phi, velocity, boundary velocity and release
  bits. Optional scalar/SIMD Wasm comparisons use those same inputs.

## What the reference means

An ordinary finite-depth 3D extrusion was not an exact oracle: its pressure
hierarchy introduced Z variation, so the middle layer alone did not conserve
mass. The explicit `referenceDimension: 2` diagnostic mode instead uses one
physical Z cell, symmetry boundaries, whole-domain work, and a hierarchy that
can retain that depth while coarsening X/Y. It explicitly omits the absent Z
  derivative in `uvGradient`. Ordinary 3D callers retain three derivatives.
Two-level table eligibility is reduced to the two physical axes in this mode.

Rust uses four interpolation corners and four pressure neighbours. The
coarsest colour order preserves the reference's physical Z-plane index (one
inside its halo); the fine smoother already suppresses Z parity for symmetry.
The transport sampler's low exterior shell is zero, although pressure and
contact release retain the solved negative boundary velocities. Conflating
these fields caused the initial moving-scene mismatch and was corrected.
The coarsest solver uses the WGSL two-sum/scale/divide arithmetic, not f64.

The tall-pool fixture regenerates its voxel shell after resizing; otherwise
the original authored ceiling becomes an unintended embedded slab. Fixture capacity is sampled from the authored solid world and supplied to
Rust; the GPU gamma scratch is not an initial capacity receipt.

The probe writes a reference manifest with the current commit, source hashes,
resolved options, timestep, and backend. `--verify` enforces the unchanged
single-step bounds (V `2e-5` cell areas, phi `2e-6 m`, velocity `5e-5 m/s`),
plus exact release masks and, when enabled, tile classes. Native versus Wasm
comparison checks all serialized fields and receipts. This is numerical-scene
acceptance for the reported profiles; it does not establish every-option parity.

## Evidence

The expanded suite has 11 scenes at frames 1, 5 and 30: stationary pool,
hydrostatic pool, dam collapse, ceiling release, open top, tall pool, empty,
full, mirrored dam, embedded ceiling and an odd-width pool.

- The original ceiling discrepancy is resolved without changing the bounds.
  Its supposedly 2D GPU input had developed a `0.00734 m` difference between
  Z vertex planes. At vertex (7,4), the redistance gradient's Z component was
  `-0.00731474`, which moved the closest-point iterate to Z ≈ `0.9305`.
  An isolated replay of the production shader reproduced the original output
  within `2.98e-8 m`; duplicating one plane instead reproduced Rust's answer.
  Suppressing the absent derivative in the explicit 2D oracle removes this
  spurious degree of freedom. This was not evidence for relaxing a tolerance.
  `tools/wasm/uniform-geometric-redistance-audit.ts` isolates the shader from
  captured scene fields; `--2d` selects the dimensional derivative rule.
- See [all-fine whole-domain receipts](parity.json) and
  [two-level/dust/lagged-budget receipts](parity-two-level.json). Each records
  its full resolved profile; neither enables the not-yet-ported solve window.
- MacCormack and liquid-only velocity profiles passed the original six
  scenes independently. The [combined MacCormack/liquid-only/two-level
  profile](parity-maccormack-liquid-only.json) passes all 33 expanded
  checkpoints, including exact release masks and tile classes, with native,
  scalar Wasm and SIMD Wasm agreement.
- Native, scalar Wasm and SIMD Wasm comparisons concern serialized values;
  they do not assert preservation of signed-zero bits through JSON.
- The 18 existing Uniform Geometric numerical/boundary/work-map scene checks pass after
  retiring the capacity assertion specific to the deleted experiment.
  The old `max V < 1.1` assertion failed identically on the pre-cleanup source
  with balancing disabled (maximum V `7.3537855` at frame 90). The retained
  method has no such capacity guarantee. Conservation and finite-field
  assertions remain; the maximum is still reported.
- Scalar and SIMD Wasm builds and artifact feature checks passed. Repository
  type checking reports 15 errors outside the changed uniform code. The nine
  existing Uniform initialization/control checks also pass. New TS
  contract and parity-runner files pass targeted ESLint.
- The canonical Sparse CM12 gate finished in 412.1 seconds: 4 lanes passed,
  13 failed (8 timeouts, 4 correctness failures, 1 performance ceiling).
  See [the unchanged-lane receipts](sparse-gate.json). The failures include
  authored topology edits, uncovered-donor transport at frame 48, missing
  compiled connectivity on the outside-tank drop, and a mini64 median of
  194.2 ms against the existing 110 ms ceiling. No sparse implementation,
  lane or timing threshold was changed; these failures were not repaired as
  part of this uniform migration.

## Interactive acceptance

- `npm run test:uniform-lab:scenes` passed: three authored scenes × scalar/SIMD,
  60 frames each, through the real compiled owned world, worker protocol and UI
  decoder. It checks shared default resolution, shared initial V/phi, finite
  fields, conservation with reported dust, exact scalar/SIMD final fields,
  reset epochs, step after reset and retained-view ownership after recycling.
  See [scene receipts](ui-scene-acceptance.json). This adds scene acceptance,
  not unit tests. Measured worker/publication medians were 1.1–2.8 ms on these
  small scenes; these are observations, not a comparative performance budget.
- Browser checks passed for playback, single-step, reset, 32²/128² scene loads,
  field selection, adaptive switching and return via history. No browser console
  errors were recorded after rebuilding Wasm. The displayed pressure and surface
  fields come from completed solver frames, not intermediate-stage snapshots.
- All 31 existing publication, worker, view, lens, URL and real adaptive-world
  checks passed. All 11 existing Uniform numerical/work-tile Dawn checks passed
  on the UI integration revision. Targeted UI/controller lint passed. Repository
  type checking still reports 15 errors in unchanged sparse tests/tools.

The post-UI canonical Sparse CM12 gate completed in 408.2 seconds with 4 passing
and 13 failing lanes, matching the earlier failure categories (8 timeouts,
4 correctness failures, 1 performance ceiling). The mini64 median was 196.7 ms
against the unchanged 110 ms ceiling. See [post-UI gate receipts](sparse-gate-ui.json).

## Shared editor follow-up

- The scene selector now mounts the production `ScenePickerPopover`, including
  thumbnails, grouped catalog, recents, search and keyboard navigation. Both 2D
  methods use `LabSceneSelector` and the same timestep roster.
- The clean Uniform sidebar mounts the existing liquid-drop feature row and the
  extracted rigid-placement feature row. The latter is also mounted by the 3D
  toolstrip; shape choices, dimensions, arming and icons share one implementation.
  The radial menu shares the liquid actions, rigid placement roster and delete
  action. Supported rigid shapes are sphere, box, capsule, cylinder and cup.
- Live commands run through the owned Wasm world. Liquid drops seed phi and add
  covered volume at the source stage on the next advance. Rigid commands add,
  grab, drag, release and remove bodies without recreating the fluid world.
  The XY adapter supplies solid coverage, face velocity and fluid exchange to
  the existing Rust rigid integrator and static-world contacts. Planar mass and
  Z inertia use the primitive's cross-section and one cell of depth; ordinary
  3D bodies keep the original mass/volume path. This is dimensional integration,
  not a claim of exact 3D rigid-trajectory parity.
- URL persistence uses `startHostQueryStateSync`, query codecs and the studio's
  cached scene-diff layer. Scene, timestep, field, grid and fractional camera
  position/zoom survive reload. Drops and rigid edits update the authored scene
  in the URL. Like the studio, this restores authored state, not elapsed solver
  history. Reset restarts that authored document. A shared parser fix retains
  optional liquid-volume arrays even when the preset did not contain one.
  Scene validation now reads the canonical shape roster, so cups also survive
  reload rather than triggering a silent fallback to the unedited preset.
- Shared body identity allocation avoids reusing an existing body's ID after
  deletion. Queued edits check their world owner before updating the document
  or displaying a receipt, so a reset cannot adopt an old world's completion.

Validation on this follow-up:

- The existing owned-world scene harness now covers authored URL round trips,
  live injection, held/released bodies, removal, all five shapes and a submerged
  light disk rising under buoyancy. Scalar and SIMD scenes pass; live-tool and
  shape-roster fields agree exactly. No unit tests were added.
- Browser verification passed scene search/keyboard selection, timestep changes,
  restored drop/body geometry, rigid dragging, radial deletion and URL updates.
  The browser recorded no console errors.
- [Current UI-default parity receipts](parity-ui-defaults.json) pass all 33
  checkpoints across 11 scenes at the original bounds, with native, scalar and
  SIMD agreement. These core fixtures contain no moving rigid bodies.
- The 47 existing publication, worker, lens, persistence and adaptive-world
  checks pass, as do three existing Rust rigid-reference checks. Targeted lint
  passes. Repository type checking retains 15 errors in unchanged sparse files.

The post-tool canonical Sparse CM12 gate completed in 418.7 seconds: 4 passing
and 13 failing lanes, matching the prior categories (8 timeouts, 4 correctness
failures, 1 performance ceiling). Mini64 measured 198.25 ms against the unchanged
110 ms ceiling. See [tool-integration gate receipts](sparse-gate-tools.json).
No lane or timing threshold was weakened.

Moving bodies can occlude conserved donors and reduce liquid volume; the UI
reports that drift. The 36-frame live-tool scene reports 179.23036 cell areas after starting
with 168 and injecting 13: a loss of 1.76964 (about 0.98%). No capacity balancing or compensating mass injection
was introduced. Quantifying and comparing moving-solid volume loss against the
3D method remains a scene-level follow-up; the body-free conservation assertions
remain unchanged.

## Continuous planar inflow

Uniform 2D accepts inflow scenes, including `hero-garden-hose`. The nozzle is
projected into XY, preserving its XY position and velocity and discarding its
Z offset, like the planar rigid adapter. Static scenery remains the central XY
slice. A purely Z-directed nozzle has no planar flow.

Each step integrates the authored start/end/ramp schedule and emits a swept
rectangular jet from the nozzle outlet. Its area is `2 * radius * planar speed *
integrated strength`; this is planar area, not the 3D circular volume rate.
Exact cell clipping preserves subcell jets. Emission respects available cell
capacity, updates the surface, and enforces jet velocity around receiving cells
before and after pressure projection. Blocked or full cells reject emission;
the receipt counts only accepted liquid. Source frames skip surface feedback,
as existing live liquid drops do. Reset clears the source clock and counters.

Run `npm run test:uniform-lab:inflow` after rebuilding the scalar and SIMD
artifacts to check the garden hose and a timed jet through the production worker
and publication decoder, including conservation, finite fields, reset and parity.

## Deferred work

1. Solve-window and pressure-window scheduling. `activeRegion: on` still fails
   explicitly in Rust; the UI's sole documented override is `off`.
2. Editable stage controls, intermediate-stage capture and live parameter
   changes. Current controls display the shared defaults and completed fields.
3. Voxel sculpting. Continuous planar inflow, live liquid
   ball and rigid tools are implemented, but arbitrary 3D rigid
   trajectories and moving-solid conservation do not yet have GPU parity evidence.
4. Broader stage parity for anisotropic/high-CFL scenes and material forces,
   sharpening work-map scheduling, and the GPU's asynchronous lagged-budget
   observation timing. The existing strict numerical receipts remain evidence
   for their recorded profiles, not proof of these deferred cases.
5. Large-scene memory/performance comparison with adaptive Rust. The current
   grid limit is 4,194,304 cells; small-scene timings do not establish an overall
   speed advantage over the adaptive method.

## Commands

```sh
npm run generate:uniform-geometric-contract
npm run check:uniform-geometric-contract
npm run build:physics-wasm:single
npm run test:uniform-lab:scenes
npm run probe:uniform-geometric:parity -- --wasm
npm run probe:uniform-geometric:parity -- --wasm --verify
# Optional parameter profile; always resolved through the shared schema:
FLUID_UNIFORM_PARITY_VALUES='{"twoLevelVelocity":"on"}' npm run probe:uniform-geometric:parity
```

`FLUID_UNIFORM_PARITY_REPORT` chooses the JSON report path. Dawn work remains
serialized through the repository WebGPU lease; never run it alongside a
browser GPU session or another Dawn harness.
