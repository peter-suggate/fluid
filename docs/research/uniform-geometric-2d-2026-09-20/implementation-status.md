# Uniform Geometric 2D migration — implementation status

**Current state (2026-09-23).** Rust 2D runs the 3D method's default algorithm.
The lab resolves the shared defaults with no overrides. The earlier
`activeRegion: "off"` scoping is gone because 3D replaced its active region
with pages and a phi window.
[Algorithm parity at the 3D defaults](#algorithm-parity-at-the-3d-defaults-2026-09-23)
is the current account. The sections after it record the first integration
(2026-09-20/21) as history; where the two disagree, the parity section wins.

## Algorithm parity at the 3D defaults (2026-09-23)

The oracle is WebGPU `referenceDimension: 2` running the 3D defaults: one
physical Z cell, symmetry depth, and every default stage. Three 3D changes were
needed for it:

- Total surface volume and surface-deficit balancing are no longer gated to
  dimension 3.
- `uvPhiFarAir`, `uvClosedWallPhi` and `uvEmbeddedAir` are plane-symmetric.
- The coarse velocity table accepts a collapsed Z axis.

Ordinary 3D runs are unaffected.

### What 2D runs

- **Phi window** (`pages.rs`): the census, dense-census countdown and
  256-word header come from `scanExternalActiveSources` and
  `finalizeActiveRegion`. As in 3D, the window exists only for a lattice of
  more than one 32-cell page with a positive dust floor. It bounds the two
  vertex phi passes and the released-wall pass. Paged *storage* is a WebGPU
  memory layout and is outside the native contract; Rust keeps dense arrays up
  to the unchanged 4,194,304-cell cap.
- **Extension**: tile classes, the fast iterative method and the two-level
  hierarchy. Metal fuses `dot()` into an fma, so Rust uses `mul_add` in the
  nearest-hierarchy sample.
- **Phi advection and redistance** on the window.
- **Transport**, with liquid drops and the inflow plug entering as sources, as
  `uvSources` and `uvSourcePhi` do.
- **Total surface volume** (`surface_volume.rs`): the surface shifts along its
  normals, at most one cell per step, to match total conservative V.
- **Surface-deficit balancing**, on by default, with the 3D reduction order.
- **Sharpening**: 8 rounds, strength 1, distance 2.1h.
- **Pressure**, with the arithmetic Metal actually executes:
  - division as reciprocal-multiply;
  - `mgApply` as an fma fold;
  - sequential sums over depth-duplicated corners;
  - a plain-f32 coarsest solve. Fast math reduces `mgTwoSum`'s error term to
    zero, so the compensated pair the WGSL spells out is not what the GPU runs.

### Outside the native contract

- **The splash-survival group** (`UNIFORM_GEOMETRIC_SPLASH_KEYS`) is 3D-only:
  - `phiCubicAdvection`, `phiDrain` and `airborneMomentum` became 3D defaults
    on 2026-09-23. They are not ported, and the parity probe pins them off.
    Porting them is a separate decision.
  - `redistanceSurface` defaults to `rebuild` and `orphanVolume` to `relay`.
    Rust's behaviour is what those defaults select.
  - `orphanVolumeRender`, `isolatedBodyVolume` and `phiSeedCells` default off.
- **`volumeStorage` and `pageSize`** are WebGPU storage choices.
- **Rejected by `validate_supported`**: non-default values of
  `velocityTransport`, `liquidOnlyVelocityAdvection`, `volumeCompaction`,
  `phiSeedFromVolume` and `phiAgreement`. The option file is still generated
  from the TS contract.

### Retired

- The swept-extension and late-energy experiments.
- The lab's regional surface profiles.
- The `set-surface-experiment` and `set-surface-deficit-balancing` session
  commands.
- Rust's own planar inflow injection (`inflow::Step`), replaced by the 3D
  source plug.
- The driver tools that sent the removed request fields:
  - `uniform-geometric-compensation.ts`
  - `uniform-geometric-energy-audit.ts`
  - `uniform-figure3-regression.ts`
  - `uniform-geometric-swept-experiment.ts`
  - `uniform-geometric-surface-balance.py`

The dated reports that used these carry a retirement note. The scene runner's
request now rejects unknown fields. The unit tests of `inflow::Step` were
deleted with it. They checked:

- the exact planar area of a sub-cell diagonal jet;
- the start/end ramp schedule;
- that full cells refuse water.

The lab shows total surface volume and surface-deficit balancing as switches,
both on. Turning one off writes `=0` to the URL and resets the scene.

### Evidence

- **One-step fixtures: 36 of 36 pass** (12 scenes at frames 1, 5 and 30) at
  the unchanged bounds. The scenes include `wide-dam`, 96×48 across 3×2 pages,
  so the phi window is exercised.
  - The worst matched differences are V `4.8e-6`, phi `7.7e-7 m` and velocity
    `3.1e-6 m/s`.
  - Release bits and tile classes are exact.
  - Rust's own window regions match the GPU header exactly. The recorded
    max-speed word differs by at most one ulp.
  - `probe:uniform-geometric:parity -- --wasm --verify` exits 0, with native,
    scalar and SIMD agreement.
- **Free runs** from the frame-1 input stay within about `1e-5` of the GPU
  over 30 frames on most scenes. The two exceptions come from arithmetic, not
  from the algorithm:
  - *Dam scenes.* Apple's hardware reciprocal is not always correctly rounded.
    About one smoother row in 70 differs by an ulp, and a breaking dam amplifies
    that chaotically. `dam-collapse` reaches about `1e-3`; `wide-dam` reaches
    V 1.3 in one cell.
  - *Resting pools* (`hydrostatic-pool`, `open-top`). A 0.075-V packet
    stranded in an air-side corner cell above the surface is walked by the
    8 sharpening rounds up to 8 cells in one step. Whether the walk starts is
    a near-tie, and 7e-6 of accumulated drift flips it at step 21.
  - From the GPU's own state, Rust reproduces every step 5–30 of
    `hydrostatic-pool` to `1.2e-7`. In 3D the ghost drain would deal with
    these packets, but the probe pins it off.
- **Lab**: `npm run test:uniform-lab:scenes` passes with scalar and SIMD
  artifacts, including a 12-frame run and a scalar/SIMD check with each surface
  switch off.

### Cost

The CPU schedule follows the GPU's work maps:

- Transport traces departures only for TRANSPORT cells, and its exact donor
  sums touch only live receivers.
- The extension restricts and prolongs only the faces its readers want.
- Sharpening runs from its work map, and the census scans only the previous
  window grown by 8 cells.
- The pressure smoother, residual and norm visit liquid rows. Prolongation
  skips cells whose coarse taps are all zero.
- Velocity advection and forces visit FINE tiles only.

Every one of these is pure scheduling. Across 9 scenes and 12 option variants
(audit dumps, dense two-level, dense advection, balancing off, pressure-row
variants, zero tolerance, one sweep), the output is bitwise identical to the
dense solver it replaced. The exact donor sum is now three u64 limbs. A
brute-force check of 122 million adds matched the GPU's six u32 words at every
prefix.

Native release build, ms per step over 30 frames. Both arms were measured back
to back on the same machine. "Dense" is the parity-complete solver before this
scheduling work.

| scene | cells | dense | scheduled | speed-up |
|---|---|---|---|---|
| fig7-256 | 256×256 | 38.8 | 18.0 | 2.2× |
| seiche | 320×96 | 38.1 | 21.5 | 1.8× |
| hillside | 256×112 | 20.3 | 8.7 | 2.3× |
| tank | 256×96 | 22.9 | 12.4 | 1.8× |
| boxes | 60×45 | 52.1 | 10.6 | 4.9× |
| long-dam | 192×96 | 11.6 | 5.6 | 2.1× |

The Rust path the lab ran before this work took 95 ms on fig7-256 and 43 ms on
hillside. On the scheduled solver, fig7-256's largest stages are:

| stage | ms |
|---|---|
| velocity advection | 6.4 |
| phi advection | 4.7 |
| projection | 3.2 |

Most of the rest of boxes' projection is the coarsest solve: 1,500–3,000
iterations per call. 3D runs the same iterations in `mgSolveCoarsest`.

### Known failure

`npm run test:uniform-lab:inflow` fails "water moves" on
`water-box-dam-break`. The nozzle radius (0.04 m) is smaller than a cell:

- 3D writes inflow phi only through `uvSourcePhi`.
- A plug thinner than the vertex spacing never brings a vertex below zero.
- So the injected V, up to 1.2 in a cell of capacity 1, has no pressure row
  and does not move.

3D behaves the same way. The splash stages do not rescue it: airborne momentum
needs cell phi above 1.5h, and the drain only turns phi-liquid into air. The old
Rust injection hid this by writing the nozzle body into phi. The test is left
red. The choice is to fix `uvSourcePhi` in 3D or delete the test.

`npm run test:uniform-pressure:2d`, the Figure 9 stability harness, is also
red, for two reasons:

- **Frozen matrix, red before this work.**
  `frame-104-pressure.json.gz` still carries `activeRegion`, `pressureWindow`,
  `pressureCycleBudget` and `pressureBudgetHeadroom`. The paging work retired
  all four, and the generated options reject them.
- **Exact conservation.** The harness asserts `|V/V0 - 1| < 1e-5`, which the
  3D default dust floor (`1e-3`) cannot hold. Over 180 frames Fig 9 loses
  13.64 cell areas, 0.36%. The receipts' transport dust (10.67) and sharpening
  dust (2.96) account for all of it.

The harness passes with native, scalar and SIMD when two things change: the
four retired keys are dropped from the frozen request, and conservation counts
the reported dust, as `test:uniform-lab:scenes` does. The run peaks at
25.7 m/s, phi area stays at or above 0.996 of V0, and the frozen matrix
converges to residual 0.40 with no rejected cycles. The harness itself is
unchanged. Repairing or deleting it is still to be decided.

The diagnostic scene runner now also takes the seed's `viscosity` and
`surfaceTension`, as the owned session does; both default to zero. It used to
drop them silently and run inviscid.

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

Since 2026-09-23 the nozzle is the 3D source plug (`inflow.rs` `Plug`, from
`inflow-boundary.ts`):

- **Volume.** Each cell receives its share of the plug swept this step
  (`inflowSweptPlugSource`). That share enters at transport as a source.
- **Phi.** The plug arm of `uvSourcePhi` writes it into phi.
- **Velocity.** The axis-face velocity boundary is applied after projection
  and again as the last body force.

In the planar reduction the aperture is a segment of width 2r across the
dominant-axis face, instead of a disk. The earlier Rust-only swept-rectangle
emission (`inflow::Step`) has been retired. So has its capacity clipping: the
plug does not refuse water at full cells, and neither does 3D.

`npm run test:uniform-lab:inflow` checks the garden hose and a timed jet
through the production worker. It currently fails on a sub-cell nozzle; see
[Known failure](#known-failure).

## Deferred work

1. Editable stage controls, intermediate-stage capture and live parameter
   changes. The lab exposes only the total-surface-volume and
   surface-deficit-balancing switches; the other controls display the shared
   defaults.
2. The 3D splash-survival stages (see [Outside the native contract](#outside-the-native-contract)).
3. Voxel sculpting. Continuous planar inflow, the live liquid ball and the
   rigid tools are implemented. Arbitrary 3D rigid trajectories and
   moving-solid conservation still have no GPU parity evidence.
4. Broader stage parity for anisotropic/high-CFL scenes and material forces.
   The strict numerical receipts are evidence only for the profiles they
   record.
5. Publication view masks. The owned session publishes every plane on every
   snapshot, whatever mask it is given.
6. Large-scene memory/performance comparison with adaptive Rust. The grid
   limit is still 4,194,304 cells.

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
