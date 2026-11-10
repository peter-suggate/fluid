# Lane R progress: motor-assembly contact on the GPU, native path deleted, Phase 0 capture

Plan: `docs/plans/paddleboats-60fps-2026-11-10.md`. Nothing here is committed.

## Decisions

### 1. Motor-assembly contact is entirely in the GPU rigid kernel

The new WGSL and its f64 TS mirror are in `lib/core/rigid-assembly-contact.ts`.
Every WGSL function has a TS twin of the same name, and a test enforces this.
The kernel is `integrate` in `lib/core/webgpu-rigid-body.ts`, one thread for
the world. Its order is:

1. traction gather
2. `rigidAssemblyBuild`
3. kick: v and ω only
4. a speculative assembly-pair pass
5. position and orientation integration
6. `RIGID_CONTACT_ITERATIONS` = 6 sweeps: static surfaces for each body, then pairs
7. held rollback
8. `attachMotor`
9. publish

**Compound support.** An assembly is described by its pieces:

- piece 0 is the core shape;
- piece 1 is the hull envelope. These are the 78 vertices of the exact convex
  hull of the 1600 BVH leaf corners, computed by `lib/core/convex-hull.ts` and
  checked to have zero support deficit;
- piece 2+j is fixture j's axle cylinder: the motor record's floats 24–27 hold
  `[alongMin, alongMax, radius, unique]`. On a paddleboat this is
  [-.25, .25] × r .304, one per wheel.

Points that tie on support are averaged within `RIGID_SUPPORT_TIE_M` (5 mm).

**World solids.** Each piece is tested against the container planes, the
terrain plane or height field, and the solid voxel world. The response is a
block sequential impulse, `assemblyPlaneContact`:

- clamped normal and friction impulses are accumulated (Catto);
- 16 rounds are run for each call;
- gaps get speculative targets, so contact is bounded by travel at 40 m/s
  without substeps;
- penetration is projected out by the deepest piece.

**Assembly vs assembly and assembly vs ordinary body.** This uses SAT over 18
axes on the compound pieces, with a midpoint witness point. A negative depth
is a separation certificate. Assemblies get a speculative pass before
integration, then projection plus impulse. An ordinary body is treated as a
sphere against the compound support.

**Ordinary–ordinary pairs** are unchanged (sphere arithmetic, bit-identical).

### 2. The per-step Box3D path is deleted

The following are removed:

- `enableNativeMotorContacts`, the native receipt chain and its readback;
- `finishNativeStep`;
- `deferredFramePublication`, from `method-contract.ts` and the uniform
  reference;
- the renderer's presentation fence: `completedFrameAwaitsPresentation`,
  `deferredPresentedTimes` and `drawWithDeferredPresentation`. The draw gate is
  now `if (readyGPUFluid)`.

`presentationPending` in the uniform reference now means only that a pressure
frame failed.

`lib/interaction/adapters/box3d/motor-contacts.ts` moved to
`tests/helpers/box3d-motor-oracle.ts` and is labelled TEST ORACLE ONLY.
Nothing in `lib/` imports it.

### 3. Phase 0 frame capture (`lib/core/frame-instrumentation.ts`)

**App.** While performance instrumentation is enabled (performance panel
timeline or activity), `gpu-viewport-session.ts` records one sample per rAF
for each pane. `globalThis.fluidFrameCapture()` returns the samples (the last
3600 for each pane). A sample holds:

- `raf_ms` and `rafPeriod_ms` (the rAF timestamp);
- `hostEncode_ms`, the renderer CPU trace: frame control, physics submission
  and encoding;
- `render_ms`, the presentation GPU span;
- `solverGpu_ms`, from the solver's timestamp trace `physicsTrace`;
- `fineTiles` and `coarseTiles` (h and 4h);
- `particles` (NB-FLIP);
- `dispatches` {direct, indirect}.

The two GPU times are sampled at their traces' own cadences. A sample carries
a GPU time only on the frame its trace first arrives, with that trace's sample
id. Where timestamp queries are missing, the time is absent.

Dispatches are counted in the renderer's realm, which is the worker when the
renderer runs in one, and reach the main thread through
`RendererFrameMetrics.dispatches`.

**Zero cost when off.** The two dispatch methods on the compute pass encoder
prototype are patched only while instrumentation is enabled. Disabling puts
the platform's own functions back; a test checks function identity. No
sample object is built when off.

**Headless.** `tools/benchmark-nbflip-ui.ts` already reported `encode_ms`,
`wall_ms`, `fineTiles` and `particles` for each frame. It now also reports
`coarseTiles`, `dispatches` (new `--dispatches` flag, using the existing
`tools/gpu-dispatch-counter.ts`) and, with `--profile`, `solverGpu_ms` (the
sum of the frame's pass timestamps). Render time and rAF period have no
headless equivalent because that harness has no renderer.

## Verified, and how (CPU and validator only)

- **naga 30, no device.** All six rigid shader variants validate: plain,
  terrain, terrain-sculpt, terrain-procedural, solid-world and
  solid-world-procedural. This caught `active`, a reserved WGSL word, which is
  now `engaged`.
  - The `solid-world`+terrain combination fails naga at HEAD too. That
    combination is unused and is not this lane's.
- **`tests/rigid-assembly-contact.test.ts`, 22 pass.** Each test, against the
  TS mirror unless noted:
  - the hull envelope is exact, and degenerate input throws;
  - compound support matches brute force: the core and hull to 1e-6, and the
    axle encloses a 720-phase sweep with less than 1 cm slack;
  - an ordinary box rests;
  - a paddleboat lands and settles: v and ω below 1e-3, with no point below
    −2 mm;
  - side and end walls at 5 yaws at 4 m/s and at 40 m/s: no point beyond
    1 mm, and energy never rises above E0;
  - hull pairs head-on and glancing at 4 and 40 m/s: momentum is conserved
    while clear of walls, energy does not rise, depth stays at or below 1 mm,
    and there is no pass-through;
  - SAT gap certificates over 400 random poses;
  - penetration is recovered in one step;
  - the dry roots are bit-identical;
  - agreement with the Box3D oracle at 4 m/s (yaw 0 and π/2): the contact
    frame within 1, the final z within .15;
  - unknown kinds fail;
  - the kernel-twin rule.
- **`tests/paddleboat-contacts.test.ts`:** tests 1–2 now run on the kernel
  mirror with the same assertions. All pass.
- **`tests/frame-instrumentation.test.ts`, 4 pass:**
  - patching and restoring is identity-exact;
  - dispatches are counted, and a take resets them;
  - unavailable without WebGPU;
  - per-sample dedup of the GPU traces, the rAF period and the ring capacity;
  - the pane capture's lifecycle.
- **`npm run check:types`:** clean.
- **`npm run test:unit`:** 1724 pass and 1 fail. The failure is
  `tests/reservoir-particle-cost.test.ts` "merge cap protects neither…". It
  imports `uniform-near-settled.ts`, which another lane is editing, and it is
  not touched by this lane.

## GPU runs used: 1 of 3

```
tools/gpu/fluid-gpu-run.sh check env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal node --import tsx --test tests/rigid-motor-gpu.test.ts
```

The runner returned rc 1 from the test itself, not 76/77/78.

- Test 1 passes: no dry thrust, and the axle traction returns.
- Test 2: every kernel assertion passes:
  - the GPU hull is within 5 mm and 1e-4 orientation of the f64 mirror after
    45 back-to-back steps with no readback;
  - the wheels follow `rigidMotorPose`;
  - the motion sidecar publishes the contact-resolved pose of the same step;
  - a selection made while steps are in flight survives.
- **Test 2 FAILS** its last assertion, `z < 1.95` ("wheel contact stops the
  boat before its buoyancy box reaches the wall"). The mirror predicted this:
  z = 1.957. The wheel-cap impulse at a y-arm of .075 rolls the zero-g hull
  about 22°, and its side envelope then meets the wall. The buoyancy box's
  largest z support is 2.381, below the wall at 2.4, so the box never reaches
  the wall. The 1.95 figure was calibrated on Box3D. It is left unchanged and
  reported as failing.

## Unverified; needs a GPU gate

- The full paddleboats scene with fluid (360 steps, so not run under this
  lane's budget):
  `FLUID_PADDLEBOAT_GPU=1 tools/gpu/fluid-gpu-run.sh check env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx --test tests/paddleboats-scene.test.ts`
- **Step cost of the kernel.** This is unmeasured. The plane SI is 16 rounds ×
  up to 4 pieces × 6 sweeps; floating boats exit early through `engaged`.
  It needs one profile of the rigid pass in the moored scene.
- **App capture.** Not exercised in a browser. Enable the timeline in the
  performance panel, then call `fluidFrameCapture()` in the console.

## Known gaps

- Closed for motor assemblies by the follow-up below; ordinary bodies
  still see the container planes and no voxel solids.
- Ordinary–ordinary pairs remain spheres.
- Speculative pair impacts are near-inelastic.
- SAT can report contact slightly early near slanted faces.
- At 40 m/s the Box3D oracle stops dead, while the kernel tumbles the hull.
  Energy never increases.

## Edits outside Lane R's own files (smallest edits)

- `lib/methods/uniform/webgpu-uniform-reference.ts`: the native path,
  `deferredFramePublication` and the receipt chain are removed;
  `presentationPending` and `mixedHostBusy` are simplified.
- `lib/core/webgpu-renderer.ts`: the presentation fence is removed;
  `RendererFrameMetrics.dispatches` is added, with dispatch counting while
  instrumentation is on.
- `lib/core/method-contract.ts`: `deferredFramePublication` is removed.
- `lib/core/webgpu-render-worker.ts`: calls `draw`.
- `components/gpu-viewport-session.ts`: the rAF timestamp and the frame
  capture call.
- Tools: `tools/probe-nbflip-presentation.ts`, `tools/probe-cloth-camera.ts`
  and `tools/capture-forest-waterfall.ts` now call `draw`.
  `tools/benchmark-nbflip-ui.ts` gains the Phase 0 fields.
- Tests:
  - `tests/gpu-advance-presentation-cadence.test.ts` and
    `tests/rigid-native-step-lifecycle.test.ts` follow the deletion of the
    native path.
  - `tests/rigid-motor-gpu.test.ts` test 2 now compares against the mirror.
  - `tests/paddleboat-contacts.test.ts` tests 1–2 now run on the mirror.
- Moved: `lib/interaction/adapters/box3d/motor-contacts.ts` →
  `tests/helpers/box3d-motor-oracle.ts`.

## Follow-up: assemblies collide with the edited voxel solid (2026-11-10)

**The question about test 3.** Yes, `tests/paddleboat-contacts.test.ts` test 3
asserts edited-basin collision: cut, save/load, no duplicate collider, then
undo. It does this on `Box3DWorld`, the interaction world, not on the motor
path. It was not dropped or changed, and it passes. At HEAD, tests 1–2 had no
edit assertions. However, the deleted Box3D motor path did call
`updateSolids`, so its edit-awareness was lost when I deleted it. The work
below restores it on the GPU.

**Design (all GPU, in `integrate`; mirror in `rigid-assembly-contact.ts`).**

An assembly no longer meets the five container planes. It meets one collider,
`assemblyVoxelContact`, built from two existing sources:

- the fluid's `SolidOccupancyMask` image at word 0 of `activeScratch`
  (binding 10, read with `solidOccupancyMaskWGSL`), which carries scenery and
  `solidVoxels` fills;
- outside the lattice, the box shell that `solidVoxelShellForScene`/the mask
  walls describe: side slabs, plus a top slab when the container is closed.
  The sculpt cut field already published for `editedTerrainContact` is
  subtracted from the shell. A cell is open where the baked cut value at its
  centre is below zero.

How the contact works:

- The scan covers the bodies' swept, spin-padded AABB.
- A slab with no cut page in reach is one box.
- Otherwise the scan builds runs of solid cells, with their exposed faces.
  Mask pages are scanned per non-empty page.
- Boxes are gathered by face plane, at most 16 planes. Each plane is solved
  once through `assemblyPieceContact`, which is `assemblyPlaneContact` with a
  piece mask and the contact arms clamped to the plane's box.
- The terrain plane (`editedTerrainContact`) remains the floor.
- Ordinary bodies, sphere containers and open-world or free-space scenes keep
  the planes.

**Regressions.**

- `tests/paddleboat-contacts.test.ts` test 4 (f64 mirror) checks:
  - the end wall stops the hull (reach 6.4000, vx 0.0025);
  - a 1.5 m hole lets the centred hull through (x 9.0, vx 4);
  - at z = ±0.75 the wall beside the hole still stops it (reach 6.4000);
  - a fill block stops it at the block face (4.6000);
  - undo, with an incremental bake plan, reproduces the intact run exactly
    (`deepEqual` over all 90 frames).
- `tests/rigid-assembly-solid-gpu.test.ts` runs the same three edited cases
  on the GPU kernel. The GPU hull is within 5e-3 m of the mirror after 60
  frames. It passed in GPU check run 1 of 2.
- The twin-name test now also covers `rigidAssemblySolidWGSL`.
- Mirror cost, which is not a GPU timing: 0.35 ms a step with the wall
  intact, 1.5–5 ms near a hole, 16 ms at a block.

**Gaps.**

- Only `clear && affectsScenery` cuts open the shell. A later fill over a cut
  is not seen.
- The fluid's wall pages do not open, so water does not flow through the hole.
- Kernel cost grows with the cells scanned near cuts and blocks. The kernel is
  single-thread, and FULL pages are scanned per cell. Profiling this is
  pending, because timing runs were not allowed.

**Outside edit.** In `lib/methods/uniform/webgpu-uniform-reference.ts`, the
`setProceduralCollisionSource` call gains `assemblySolid: {cells, walls}`
for box tanks. This is a 4-line edit.

**The `z < 1.95` case** (`tests/rigid-motor-gpu.test.ts` test 2, zero g,
45 frames). It is left failing, as instructed.

| Source | Final hull z | Peak z | vz |
| --- | --- | --- | --- |
| Box3D oracle | 1.9325 | 1.9889 | -0.094 |
| f64 mirror | 1.9565 | 1.9864 | -0.052 |

Both reach the same depth; Box3D's peak is 2.5 mm deeper. Box3D passes the
assertion only because it rebounds faster. The limit therefore tests rebound,
not penetration.

**Unrelated failure.** `test:unit` has one failure: reservoir-particle-cost,
"merge cap protects neither the outer layer nor mass-saturated deep runs".
It comes from another lane's `uniform-near-settled.ts`.

## Follow-up: attribution of the `z < 1.95` failure (2026-11-10, CPU only)

No kernel change. No GPU run used (0 of 3). The test is unchanged and still
fails on that one assertion.

**Result.** The kernel is not resting 2.5 cm off. Nothing rests: the hull is
drifting back from the wall in zero g, and the 2.4 cm is a rebound-speed
difference (-0.052 vs -0.094 m/s) integrated over 33 frames. The limit 1.95
sits on the converged answer (1.947 to 1.950), so a pass or a fail at 1/60 is
decided by the sign of each engine's time-step error.

**Step-refinement study** (final hull z after 0.75 s, restitution .05, each
engine stepped at (1/60)/n; the Box3D oracle also substeps internally):

| n | f64 mirror | Box3D oracle |
| --- | --- | --- |
| 1 | 1.9565 | 1.9325 |
| 2 | 1.9486 | 1.9323 |
| 4 | 1.9488 | 1.9658 |
| 16 | 1.9469 | 1.9481 |
| 64 | 1.9471 | 1.9497 |

- The mirror converges monotonically to 1.947. Its 1/60 error is +9.5 mm.
- Box3D converges to about 1.949, the same answer within 3 mm. Its coarse
  runs scatter from 1.932 to 1.966; at n = 4 it fails the assertion itself.
  Its 1/60 value is 1.6 cm below its own limit.
- With restitution 0 the picture is the same (mirror 1.9575 → 1.9464; Box3D
  1.9367, 1.9424, 1.9321, 1.9404, 1.9347).

**Candidate causes, each ruled out by measurement.**

- Wall offset or voxel shell: this test has no voxel solid; both engines use
  the plane at z = 2.4. Peak hull z agrees: 1.9864 (mirror), 1.9889 (Box3D).
- Penetration or margin: the mirror's deepest piece never passes the wall
  (max -1e-7 m, the projection's own offset). Box3D's poses, measured with
  the kernel's supports, penetrate 1.8 mm.
- One step late: the kernel engages one frame early (frame 7, speculative,
  gap 2 cm), Box3D at frame 8.
- Restitution lost on speculative impacts: real, but worth 1 mm here
  (mirror e = 0 vs .05: 1.9575 vs 1.9565).
- Removing the speculative plane targets (classical penetrate, project,
  restitute) is worse: 1.9799.

**The required behaviour holds in the kernel.** Mirror, 45 frames at 1/60:

| | with wheels | hull alone (no fixtures) |
| --- | --- | --- |
| peak hull z | 1.9864 | 2.1014 |
| final hull z | 1.9565 | 2.0236 |
| buoyancy box peak z support | 2.3810 | 2.3998 |

The wheel cap meets the wall at frame 8 (support 2.4000), the hull envelope
at frame 11, and the buoyancy box stays 1.9 cm clear. Final supports at
frame 45: box 2.3614, hull envelope 2.3795, right wheel 2.3851.

**Proposed assertion** (not applied; the test is left as it was). From the
final GPU pose, with the twin's `rigidPieceReach` along +z:

- every piece: `z + reach <= 2.4 + 1e-3` (penetration bound; kernel 0);
- buoyancy box: `z + reach(piece 0) < 2.4 - 0.01` (kernel 2.361, peak 2.381;
  2.3998 without the wheel);
- wheel stopped it: `z < 2.0` (every run of both engines peaks at or below
  1.9901; hull alone ends at 2.0236 after peaking at 2.1014).

**Checks.** `npm run check:types` clean. CPU:
`tests/rigid-assembly-contact.test.ts`, `tests/paddleboat-contacts.test.ts`,
`tests/paddleboats-scene.test.ts`: 29 pass, 0 fail, 1 skipped (the GPU-gated
scene test). Scratch scripts: `~/code/me/fluid-scratch/lane-r/`.

## Follow-up 2: the wall-impact assertion (2026-11-10)

Supersedes "the test is unchanged and still fails" above. The `z < 1.95` check
in `tests/rigid-motor-gpu.test.ts` is replaced by three bounds on the final GPU
pose (every piece's z support <= 2.4 + 1e-3; the buoyancy box's < 2.4 - 0.01;
hull z < 2.0), with the convergence numbers in a comment.
`tests/rigid-assembly-contact.test.ts` pins the refinement: the mirror at n = 64
is within 2 mm of 1.947 and n = 1 within 1.2 cm of n = 64. GPU run: 3 pass,
0 fail (`~/code/me/fluid-scratch/lane-r/gpu-run.log`).

## Follow-up 3: the 3.7 ms single-workgroup launch made wide (2026-11-10)

**Finding (lane M).** `integrate` cost 3.72 ms a step on paddleboats as one
workgroup of one thread (`~/code/me/fluid-scratch/lane-m/kernels-paddleboats-160.json`);
0.07 ms before the contact kernel.

**Attribution (CPU census of the mirror on the paddleboats roster, a step).**
Only the two hulls do work; the sixteen fixtures leave every function at its
first test. A step: 84 hull reaches and 12 hull supports (78 dot products
each), 168 axle supports, 12 `rsBounds`, 12 `assemblyPieceContact`, and
26-piece loops reading `motors[]` around all of them; no voxel scan
(`rsRuns` 0) and the hull pair leaves on distance. Roughly 1e5-1e6 scalar
operations, all on one GPU thread through function-storage `RigidBody`
state. Ruled out by the generated MSL: the hull envelope is a file-scope
constant table, not rebuilt per use. The per-function split on the GPU was
not measured (it needs a kernel-timed run per variant).

**Restructure.** Two launches, neither a single workgroup:

1. `rigidTables` (`@workgroup_size(64)`, paddleboats 986 workgroups =
   63 096 threads): one thread a body kicks it (the gather is read-only now)
   and predicts its integrated pose; one thread a (body, piece slot, world
   axis) writes a support row; one thread a (phase, assembly, other body,
   slot, slot, axis) writes one separating-axis candidate of a linked pair
   with its four reaches. Rows go to a new storage buffer (binding 12,
   250 000 vec4f = 4.0 MB) whose header (assemblies, piece slots) the host
   writes at a roster sync.
2. `integrate` (one thread a body, 24 workgroups): free bodies with mass are
   split into islands by `rigidLinked`; each island is solved by the thread of
   its lowest index in the order the single thread used, reading supports and
   pair axes from the tables. Each thread writes and publishes only its
   island's bodies and their fixtures.

A table row is a cache of a mirrored function, keyed by the pose it was
computed at (orientation; position too for the three axes through the piece
centres; the step's time stamp for pair rows). A reader whose body is at
another pose evaluates the function directly, so the tables do not change
results. A body the speculative pass left alone adopts the pose `rigidTables`
predicted, so the keys match bit for bit across the two pipelines. One bound
is new arithmetic: a static surface whose normal is no world axis is skipped
when the tabulated axis reaches (support functions are sublinear) prove every
piece clear and unable to close the gap this step; a CPU test checks the
bound.

**Solver-level changes, identical in the mirror.** (1) A pair is solved only
when `rigidLinked` on the kicked velocities (reaches + travel + 0.1 m); the
speculative pass's own travel test lies inside it. (2) Contacts write only
free bodies with mass; held bodies and fixtures end the step as integration
left them (this subsumes the held rollback). The mirror's results in the CPU
suite did not move: all existing assertions pass unchanged.

**Thread caps (compile-only probe, `tools/experiments/lane-n/occ.sh`).**
`rigidTables` 512, `integrate` 384.

**Launches (dispatch census of the timing run).** contact tables: 1 direct
launch, 986 workgroups; integration and contacts: 1 direct launch,
24 workgroups.

**GPU tests.** `tests/rigid-motor-gpu.test.ts` +
`tests/rigid-assembly-solid-gpu.test.ts`: 4 pass, 0 fail
(`~/code/me/fluid-scratch/lane-r/gpu-run3.log`). The fourth is new: two hulls
in a glancing 4 m/s collision, solved from the pair tables in one island,
within 1 cm and 1e-3 of the mirror after 55 frames, no interpenetration. (An
earlier run of the same files failed only that new test, on my window: 40
frames end before the hulls meet at frame 42.)

**Timing: not measured per kernel.** The prescribed timing run
(`--kernels=100 --steps=60`) recorded no kernel times: `--kernels=<step>`
starts the timers at that step, past the run's end. All three GPU runs are
spent. Indirect evidence from the same probe's untimed step wall times,
steps 5-59 paired with lane M's run 31 minutes earlier: 3.68 ms a step lower
(median 32.96 vs 36.65 ms). The working tree differs between the two runs by
more than this kernel, so this is not the kernel's number. To measure:
`tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/probe-reservoir-bulk.ts --scene=paddleboats --bodies --pipelined --dispatches --kernels=20 --steps=60 --every=60 --out=<path>`.

**Still serial, inside one island's thread.** The six Gauss-Seidel sweeps and
the 16-round plane solve; the voxel run scans (`rsRuns`) near a cut or a
block; direct supports whenever a contact has moved or turned a body this
step (the frames with an actual impact).

**Checks.** `npm run check:types` clean. CPU contact, scene, cadence,
lifecycle and instrumentation tests: 41 pass, 0 fail, 1 skipped.
