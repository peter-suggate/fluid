# Lane C1 progress: VBD solver core, Tier B, edits

Plan: `plan.md` beside this file. Lane C1 owns M0, M1, M3 and M4 plus app integration; C2 owns `lib/cloth/contact/`.

## 2026-11-09 DECISION (user, via coordinator): no substepping, ever

Exactly one implicit step per `advance` at dt = 1/60 s (the frame and fluid step). There is no GPU-chosen N, no N cap and no inner time subdivision. The only budget is K iterations per step, authored per scene. Convergence at 1/60 comes from:
- Chebyshev acceleration;
- AVBD for stiff terms and the per-edge strain limit;
- the per-vertex line search;
- adaptive initialization;
- warm-started duals and stiffnesses.

BDF2 is the default integrator; implicit Euler stays available. C2's contact re-detection rounds run between iterations of the one step through a hook; they do not advance time.

## 2026-11-09 MILESTONE: step 0, old cloth stack deleted

**Deleted:**
- every libwetcloth-era module in `lib/cloth`: mechanics, Newton-PCG, contact, rods, porous transport, cohesion, sources, source motion, assets, wetcloth scenes;
- `lib/methods/uniform/uniform-cloth-*` (8 files);
- the cloth-only NB-FLIP support:
  - `uniform-porous-capacity`, `uniform-pressure-mixture`, `uniform-source-water`, `uniform-source-termination`, `uniform-source-resident-termination`;
  - `uniform-gpu-radix-sort`, whose only users were the cloth pore and grid exchange;
- `public/assets/wetcloth`, `tools/reference`, `tools/*wetcloth*`, `tools/probe-cloth-*`;
- `tests/cloth-*`, `tests/wetcloth-*`, `tests/uniform-gpu-radix-sort.test.ts`;
- `docs/plans/wetcloth-*`.

**NB-FLIP hooks removed.** Each existed only for cloth:
- `UniformMixedSolidFeatures.porous`: binding 6, the capacity table, `setPorousCapacity`. Free-water capacity is now always the hard aperture.
- The pressure arena's mixture faces (binding 31, `setMixtureFaces`).
- The porous `umCapacity` branches in the pressure authority and surface volume.
- `UniformMixedFrame`'s physics-plugin seam: `setPhysicsPlugins`, `clothCouplingFields`, `advanceFirstUnkicked`, `solveWithoutHistory`.
- In `webgpu-uniform-reference.ts`:
  - cloth runtimes, couplings, pore exchange, cohesion;
  - source water, termination and motion;
  - the cloth page sources and physics fine-tile boxes;
  - cloth rigid reactions;
  - cloth deferred publication;
  - the cloth rebase on edit.
- The CPU rigid readback and impulse path used only by CPU cloth contact: `readStates`, `applyExternalImpulses`, `applyGPURigidExternalImpulses`.
- The `wetClothSource` and `systems.porous` model fields, the wetcloth shelf, and the source-initial liquid seeding.

**Kept and simplified** (triangle cloth only):
- `lib/cloth/types.ts`: document schema 2.0.0, AVBD pin groups, attachment scripts, solver settings with integrator, K and tier, no substep fields, and the solver-state block.
- `topology.ts`: Dm⁻¹, canonical edges, hinges, CSR adjacency, colouring that can recolour only touched vertices.
- `geometry.ts`: picking and the surface store.
- `scene-geometry.ts`, `edit.ts` and `round-cut.ts`: cuts, with vertex origins for state remap.
- `scene-edit.ts`: edit provenance that tells the system which vertices to recolour.
- `content-stamp.ts`.
- `render.ts`: draws straight from the solver's GPU buffers, `ClothRenderSource`.

`clothWaterLedgers` on the method contract is replaced by `clothRenderSources`.

Until the new cloth system is installed (M1 integration), `webgpu-renderer` refuses cloth scenes explicitly. No scene ships cloth now.

**Evidence:**
- `npm run check:types` is green for my files.
- `tests/cloth-contact-gpu.test.ts` currently fails tsc on a `Float32Array<ArrayBufferLike>` writeBuffer argument. That file is C2's and untracked; C2 is informed via this entry.
- `npm run test:unit`: 1609 pass, 1 fail, 301 skipped. The failure is the pre-existing `reservoir-particle-cost` "merge cap protects neither the outer layer…", which also fails on the untouched baseline.

## 2026-11-09 MILESTONE: M0, Float64 oracle of the one-step VBD solve

`lib/cloth/oracle.ts` is the test oracle. It is CPU-only and runs only in tests, never per step. It holds:
- **Orthotropic StVK membrane** on Green strain, with Dm⁻¹ from material (warp, weft) coordinates. Compression softening scales each fibre's compressive stiffness by `compressionRatio`. The softened part is split so the law stays C1 and positive definite with the Poisson coupling on: soft = C11 − |C12|√(C11/C22). The per-vertex Hessian is projected to PSD by Jacobi eigen-clamping, and only when it is indefinite.
- **Discrete-shell bending:**
  - the angle is atan2 of the signed sine and cosine of the normals;
  - the gradient is verified by finite differences; it is the negative of the usual textbook sign under this orientation;
  - the Hessian is Gauss-Newton k∇θ∇θᵀ;
  - degenerate hinges are dropped.
- **Rayleigh damping in VBD form:** f = −(1/h̃)·D·(x − x̂), where D sums k_d times the projected element blocks. D is frozen within a vertex update, so the line search sees a consistent energy.
- **Inertia, with BDF2 written as a variational step:**
  - h̃ = 2h/3, x̂ = x_n + Δx_{n−1}/3, v̂ = (4v_n − v_{n−1})/3;
  - implicit Euler is h̃ = h, x̂ = x_n;
  - BDF2 bootstraps with implicit Euler on the first step, or when dt changes.
- **Adaptive initialization** from the previous acceleration, projected on gravity (VBD §3.7). Pins are initialized at their scripted displacement.
- **AVBD hard constraints:**
  - pins: 3-D equality on x − p(t);
  - optional per-edge strain limit, an inequality with λ ≥ 0;
  - α-stabilized constraint, progressive stiffness, β = 10;
  - warm start λ ← αγλ, k ← max(γk, k_ref);
  - k_ref = m/h̃² + the trace of the rest elastic block / 3, so β and k_start need no tuning per scene;
  - λ and k are bounded by k_max = 10⁶·k_ref.
- **Per-vertex Armijo line search** (8 halvings). Its slack is the floating-point bound 4·eps·Σ|terms|. This is the f32 rule the GPU will use, not a tolerance knob.
- **Chebyshev** applied per vertex after all colours, skipping pins.
- **Attachment scripts** (translate, oscillate, rotate) evaluated at the cloth clock. `ClothSolverStateData.time_s` is new and saved with the state.
- `lib/cloth/grid.ts`: a rectangular sheet builder with alternating diagonals, for tests and demo scenes.

**Tests** (`tests/cloth-oracle.test.ts`, 12 tests):
- FD gradient and full FD Hessian of the membrane block, in tension and compression; projected blocks are PSD; already-PSD blocks are unchanged.
- The energy is C1 at zero fibre strain; all four softened laws are PD.
- FD angle and bending gradients; the Gauss-Newton Hessian is PSD.
- The PSD projection.
- FD of the full local vertex energy with pins, active strain limits, damping and warm duals; the update descends.
- Repeated updates of one vertex reach its local stationary point.
- The AL energy differentiates to the clamped AVBD force.
- Exact free fall under both integrators.
- A scripted translate carries the pins to within 1e-4 m.
- A strain limit of 3% holds a soft sheet (which stretches more than 6% unlimited) below 3.5%.
- The residual falls with K, and Chebyshev accelerates it.
- The energy-loss comparison below.

**Energy loss at dt = 1/60 s** (user override test):
- Setup: an 8×8, 0.4 m cotton sheet pinned at two corners and released horizontal; undamped, 60 steps (1 s), K = 48, ρ = 0.
- Swing energy lost: **implicit Euler 64.8 %, BDF2 37.5 %**, normalized by the potential released at the lowest point. Hence BDF2 stays the default.
- With K = 200 the losses are implicit Euler 68.8 % and BDF2 25.4 %. So part of BDF2's loss at K = 48 is unconverged iterations, not the integrator.
- Chebyshev at K = 48, BDF2:

| ρ | final residual ‖∇E‖/‖mg‖ | loss |
|---|---|---|
| 0 | 2.6e-1 | 37.5 % |
| 0.8 | 1.1e-1 | 30.1 % |
| 0.9 | 3.0e-2 | 26.4 % |
| 0.95 | 1.4e-2 | 24.9 % |

  - At K = 16, ρ = 0.95 overshoots (2.9 % loss, residual 2.2), while ρ = 0.9 holds at residual 0.97.
  - Planned default: ρ = 0.9, authored per scene. The GPU K sweep will confirm it.

**Evidence:**
- `npm run check:types`: green.
- `npm run test:unit`: 1624 pass, 1 fail, 303 skipped. The failure is the pre-existing `reservoir-particle-cost` one.

## 2026-11-09 INTERFACE: Tier A solver layout, frozen for C2 (M1)

`lib/cloth/gpu-solver.ts`, class `ClothGPUSolver`. The WGSL is in `gpu-solver-wgsl.ts`.

**Device limit.** The M1 Max adapter grants only **10 storage buffers per stage**, so the solver binds 5: positions, state, status, topo and rest. C2's contact buffer is the sixth, at `CLOTH_BINDING.contact` = 7. The solver fails explicitly if the device grants fewer.

**Buffers C2 binds** (all cloth-local metres; `options.origin_m` is the world origin of the frame):
- `solver.positions` (`GPUBuffer`): `array<vec4f>`, the iterate x (xyz); `w` is free. Truncation may write xyz in place. The solver never resets positions to x_t within a step.
- `solver.state` (`GPUBuffer`) as C2's **velocities** input. Segment 0, records 0..N−1, is the step-start velocity (vec4f). Bind the whole buffer and index by vertex.
- `solver.model.topology` for the mesh:
  - triangles are `faces` + `faceMaterial` packed as vec4u (`solver.triangles` is that slice of `topo`);
  - **edges are the solver's canonical list** (v0 < v1, face0 the lower face index, face1 = 0xffffffff on a boundary), sorted by (v0, v1). Pass `{ vertexCount, triangles, edges: topology.edges }` to `OgcContactDetector.create`. Then C2's edge ids are the solver's: `ogcEdgeVerts(e)` reads this table. `buildOgcTopology` accepts any order, and face0 ownership matches `buildOgcEdges`.
- `ogcXt(i)` is the x_t segment of `state`.

**Vertex kernel hook.** Construct with `contact: { wgsl: ogcVertexBlockWGSL({ group: 0, binding: CLOTH_BINDING.contact }), buffer: detector.solverBuffer }`.
- The solver defines `ogcX`, `ogcXt`, `ogcTri` and `ogcEdgeVerts`.
- Its local system adds −f and H from `ogcAccumulateVertexContact`, and its Armijo line search adds `ogcVertexContactEnergy`.
- Chebyshev skips vertices with `ogcContactCount(v) > 0`.

**Encode hooks** (`ClothStepHooks`, all inside the solver's one compute pass, none advancing time):
- `afterFrame(pass)`: the initial guess is in place. Run stepBegin, detection and the first truncate here.
- `afterColour(pass, iteration, colour)`: after each colour sweep. Truncate here (DAT §5.1).
- `afterIteration(pass, iteration)`: after Chebyshev and the dual updates. Truncate the Chebyshev move, then decide and detect.
- `beforeFinish(pass)`: after the last iteration, before velocities are formed.

**Order per step:**
1. beginStep (clock, BDF2 bootstrap, h~);
2. frameElements (edge-dual warm start and the damping tangent at x_t);
3. frame (x̂, y, adaptive init, pin targets from WGSL scripts);
4. K × [colour sweeps, then postIteration (Chebyshev, pin duals), then edgeDualUpdate];
5. diagnose and reduce, when on;
6. finishStep (v, BDF2 history, normals, the finite check, clock++).

**Status.** `status` words: [0] sticky bits (`CLOTH_STATUS.nonFinite` = 1), [1] steps, then a 64-slot ring of {step, BDF2, residual ‖∇E‖/‖mg‖, max edge strain, kinetic, potential, elastic}. `readStatus()` is for harnesses and gates only.

**State save/load.** `readState()` and `uploadState()` cover positions, velocities, v_{n−1}, Δx_{n−1}, pin and edge duals, previous step and `time_s`.

## 2026-11-09 DECISION (C1, physics): Rayleigh damping is a frozen dissipation potential

The VBD-paper form, D = k_d·H(x) re-evaluated at every vertex update and frozen only for the line search, had no fixed objective inside a step. The compression switch makes H(x) discontinuous, so the damping force chattered between branches.

Measured with the oracle (8×8, two pins, BDF2, ρ = 0.9), residual after steps 5, 10, 15 and 20:

| K | damping k_d = 0.002 s | undamped |
|---|---|---|
| 200 | 1.4e-5, 0.21, 0.40, 0.12 | 4.6e-7, 1.8e-7, 3.3e-7, 3.8e-6 |

So the damped case did not converge, at any K.

**Replaced by** Rayleigh stiffness-proportional damping as a dissipation potential, with the element tangent frozen at x_t (`ClothDampingTangent`):
- membrane: (k_d/2h̃)·A·ε̇ᵀ D_t ε̇, where ε̇ is the Green-strain increment linearized at x_t over x − x̂, and D_t is the softened material tangent at x_t;
- bending: (k_d/2h̃)·k_b·(∇θ(x_t)·(x − x̂))².

This is quadratic and convex in x, the same objective for all K iterations, and zero for rigid translation. The old per-vertex form also damped free fall. The test is now exact free fall with damping on.

**Residuals after the change** (same setup, steps 5–20):
- K = 200: 2.8e-6, 8.1e-7, 8.6e-7, 4.2e-7;
- K = 48: 6.0e-2, 3.8e-2, 3.9e-2, 1.7e-2.

The GPU uses the same form. A frameElements pass stores f0 and f1 with (D11, D22) per face, and ∇θ × 4 per hinge, in `state` (damped scenes only).

## 2026-11-09 MILESTONE M1 landed (0a8d147b): Tier A GPU solver, plus the K sweep

**Per-step agreement.** One GPU step matches the Float64 oracle to < 1e-6 m (6×6, K = 1 and 4, all three initial guesses). The match holds at 8² to 64², where the first-step difference is ≤ 8e-6 m. A 20-step scripted BDF2 run with strain limit, damping and ρ = 0.9 at K = 48 tracks the oracle within 1.6e-4 m. Its ring residual is 3.8e-2 on the GPU against 2.0e-2 on the oracle. Unconverged trajectories differ by AVBD clamp branches; f32 vs f64 is not the cause.

**K sweep.** Ring data only, no timing. Setup: a 1 m sheet with 2 single-vertex corner pins, released flat, undamped, strain limit 2 %, dt 1/60, 60 steps, ρ = 0. Columns:
- residual is the final ‖∇G‖ / ‖Σmg‖;
- strain is the maximum over the run;
- energy lost is the share of the released swing energy lost over 1 s.

16² (256 vertices, 7 colours):

| guess | K | residual | strain | energy lost |
|---|---|---|---|---|
| adaptive, BDF2 | 8 | 4.3 | 4.9 % | |
| adaptive, BDF2 | 16 | 4.7 | 2.2 % | 9.8 % |
| adaptive, BDF2 | 32 | 0.23 | 2.2 % | |
| adaptive, BDF2 | 48 | 0.22 | 2.1 % | 9.9 % |
| adaptive, BDF2 | 96 | 0.03 | 2.0 % | |
| adaptive, BDF2 | 192 | 0.018 | 2.0 % | 6.4 % |
| adaptive, implicit Euler | 48 | 0.61 | 2.5 % | 43.9 % |
| adaptive, implicit Euler | 192 | | | 42.2 % |

So BDF2 keeps 90 % of the swing in 1 s at 1/60, and implicit Euler keeps 56 %. The oracle 8×8 test shows the same direction: implicit Euler 64.8 % lost, BDF2 37.5 %.

64² (4096 vertices). **With the plan's adaptive init, Tier A diverges below K ≈ 192:**

| guess | K | strain | residual |
|---|---|---|---|
| adaptive | 8 | 149 % | 6e4 |
| adaptive | 16 | 83 % | 3e4 |
| adaptive | 32 | 21 % | 1.7e4 |
| adaptive | 48 | 76 % | 2e3 |
| adaptive | 96 | 66 % | 47 |
| adaptive | 192 | 2.2 % | 3.6 (median 20) |

With ρ = 0.9 it diverges faster: 1e4 % strain at K = 16. The Float64 oracle reproduces this step for step: 8.1 % at step 10 and 13.1 % at step 20 on both. So it is the algorithm, not the GPU.

- **Cause.** Gauss-Seidel moves support about K rings per step: at K = 16, 0.25 m per step. The membrane wave crosses the sheet in about 1.3 steps (c = √(k/ρ) ≈ 45 m/s), and h²k/m ≈ 2200 per vertex. The extrapolating guess keeps the unsupported bulk falling, and the stretched band near the pins compounds from step to step.
- **Pin size is not the cause.** 3×3 corner pins behave the same: 15.8 % at step 29, K = 16. A vertical hang is no better.
- **'previous' (x̂) is bounded for every K.** At 64² the strain is 2.0–2.8 % and the residual 2–5. But it lags free fall, which is VBD §3.7 option (a). At 16² it loses 86–92 % of the swing at K ≤ 48, and the exact free-fall test fails. It is **not** a usable default.
- **A global energy safeguard does not help.** Choosing the guess with the lower G picks adaptive almost always, because the problem is local.

**Change:** `initialGuess` ('adaptive' default | 'previous' | 'inertial') on both the oracle and the GPU. Unknown values throw. All three are GPU = oracle tested.

**Consequence:** 64² and up need Tier B's patch-resident inner iterations, which propagate support across a 32×32 patch per outer iteration, or more K. Integration demos above about 32² wait on M3. I am starting M3 now.

## 2026-11-09 MILESTONE: cloth runs in scenes; the Bridson 2002 scenes are registered

Done at the coordinator's priority, ahead of the rest of M3.

- **Scene clock.** `lib/cloth/system.ts` `ClothSystem` steps each `SceneDescription.cloth` document with `ClothGPUSolver`, exactly once per 1/60 frame:
  - in a dry scene, the cloth owns `info.submittedTime_s` and takes one tick per `advanceTo` once `time_s ≥ now + dt/2`, so half a tick is not a step;
  - in a wet scene, it follows the fluid's submitted time, catching up at most 4 steps.
  - All work is on the GPU. The pose comes back through an async mirror with persistent MAP_READ staging and one map in flight. Every 4 steps it publishes a `ClothSurfacePublication`, which feeds picking and cutting through the shared store.
- **Renderer.** `webgpu-renderer.ts` no longer throws "Cloth scenes need the GPU cloth system…". It calls `composeSceneCloth(device, scene, solver, fluidSolver)`. `withClothSystem` is a Proxy:
  - it overrides `advanceTo`, `clothSurfaces`, `clothRenderSources` and `destroy`;
  - it overrides `stageSceneUpdate`, `applySceneUniforms` and `reseed` only when the base has them, and `reseed` returns false for cloth scenes;
  - it forwards everything else bound to the base, so `instanceof` still holds.
  - `ClothRenderSource` positions, normals and triangles are now `GPUBufferBinding`s that point straight at the solver's buffers. The bind group is reused while they are unchanged.
- **Edits.** `stageSceneUpdate` compares per-document content keys. A changed document is rebuilt asynchronously. The old solver keeps drawing and the clock waits (`pending`), then the old solver is retired after `onSubmittedWorkDone`. A document being rebuilt is left out of the published surfaces, so the store never sees a stale publication.
- **Capture hook for Lane S.** `ClothSystem.capture(times_s)` steps one tick at a time, fences each tick and mirrors. It returns `ClothSurfacePublication[][]` and throws on failure.
- **Scenes.** `bridson2002Curtain`, `bridson2002RotatingSphere` and `bridson2002Tablecloth` are registered in `lib/core/scenes.ts`:
  - audience "study", shelf "Cloth · Bridson 2002", full-scene mode, attributed to Bridson, Fedkiw & Anderson 2002;
  - they appear in `sceneCatalogCards`;
  - `lib/cloth/scenes/bridson-2002/scene-description.ts` builds each `SceneDescription`, and each card's status states what is missing.
- **Solids are real scene voxel solids.**
  - A lattice-aligned box, such as the ground slab, becomes an exact `fill` patch.
  - Any other shape is one `bakeProceduralSolid` over a clipped local lattice.
  - They render, hit-test and edit through the shared path. The test covers a `visibleToolSurface` pick on the ground and on the ball, a `sceneWithSolidStroke` cut of the ball that leaves the ground unchanged, and a save/load round trip that keeps the pick.
  - Build times: curtain 0.63 s, sphere 2.0–2.4 s, table 1.3 s. A full-domain bake took 23–76 s.
- **Moving solids fail explicitly.** A pose track still at rest at t = 0 is baked at that pose, and its motion start is recorded as `ClothDocument.pendingSolids` (validated). Before a step passes `start_s`, the system fails with code `cloth-pending-solid` and the message "<doc>: moving cloth solid '<id>' requires C2 M6":

  | Scene | Moving solid | Fails at |
  |---|---|---|
  | Curtain | ball | 2.00 s |
  | Rotating sphere | sphere | 0.75 s |
  | Tablecloth | sphere (already moving at t = 0) | 3.45 s, when it first reaches the cloth's rest AABB |

- **Layout.** The domain is brick-aligned and holds the cloth, the held solids and all tracks. The translation centres the container in x and z and puts the floor at y = 0, so the paper's ground plane sits at scene y = 0.08. Rest positions are translated, and `vertexMass_kg` is the source's `nodeMass_kg`.
- **Tests.** `tests/cloth-system.test.ts` has 5 CPU and 2 GPU tests, all green.
  - The GPU dry 8×8 scene matches the oracle to within 1e-4 over 6 ticks. It also covers the mirror tick and pick, a cut that rebuilds while the clock waits, and the gate failing at tick 10.
  - The GPU tablecloth capture runs 6 ticks of uniform free fall.
  - The unit suite has one red, the existing "merge cap protects neither…", which is not cloth.

**Not yet, and stated on the cards:**
- No cloth–solid contact in scenes, so the cloth passes through the solids. That waits on a scene solid-world adapter for C2's contact (see the COORDINATION entry below).
- At K = 64, the 150² and 254×150 sheets diverge until the M3 patch-resident solve.
- The scenes run at about 5 fps (see the tick cost below).

## 2026-11-09 INTERFACE

- `ClothSystem.create(device, scene, {label?, mirrorInterval?})` exposes:
  - `time_s`, `pending`, `surfaces`, `renderSources`, `solverFor(id)`;
  - `stepTo(target_s, maxSteps = 1)`, `mirrorNow()`, `adopt(scene)`, `settled()`, `capture(times_s)`;
  - `failure`, which is `{code, message}`;
  - `destroy()`.
- `composeSceneCloth(device, scene, base, fluidSolver)` and `withClothSystem(base, cloth, fluid)`.
- `ClothPendingSolid {id, start_s}` and `ClothDocument.pendingSolids?`.
- `ClothGPUSolver` adds three getters:
  - `startTime_s`;
  - `stepStart`, the `GPUBufferBinding` of segment `CLOTH_STATE.xt`, which C2 asked for;
  - `positionsBinding`.
- It also now uploads area-weighted render normals.
- `ClothRenderSource.positions/normals/triangles: GPUBufferBinding`.

## 2026-11-09 DECISION

- **Gate rule.** A moving solid without M6 fails the scene at the moving solid's start time. It is not silently held static.
- **Mirror solverState after a cut or undo is a "restart".** It keeps time_s and zeroes the BDF2 history and the AVBD duals, with `previousStep_s` 0. The first step after an edit is therefore implicit Euler. Carrying the exact history through an edit is M4.
- **Tier B in scenes refuses explicitly**, with "…Tier B patch solve, which is not installed yet (lane C1, M3)".
- **Gravity is fixed at build time.** It is read from `scene.fluid.gravity_m_s2`. `applySceneUniforms` does not change cloth gravity yet.

## 2026-11-09 MEASUREMENT: Tier A tick cost at scene scale

On the 150² tablecloth: 22,500 vertices, 6 colours of 3.8–4.7K vertices each, 66.9K edges and 66.3K hinges. One fenced tick:

| | K = 1 | K = 16 | K = 64 |
|---|---|---|---|
| strain limit + damping | 3.35 ms | 45 ms | 178.7 ms |
| no strain limit | 2.57 ms | | 155.5 ms |
| no damping | 2.12 ms | | 125.7 ms |

- That is about 2.8 ms per iteration, or about 0.5 ms per colour dispatch, for only 4K threads each. The cost is latency: every thread does a long serial pass over its adjacency, so dispatch overhead is not the problem.
- Two levers follow, and both are M3:
  1. do less work per thread, by caching per-element terms;
  2. use fewer, larger passes, which is Tier B's patch residency.

## 2026-11-09 MEASUREMENT: Tier B oracle (block-Jacobi patches, Gauss-Seidel inside each patch)

Float64 oracle, 64², pins at 1 corner, flat start, 2 % strain limit, adaptive init. "Strain at step 29" is the maximum strain at step 29 / ‖∇G‖ at step 29.

| Patches (count/size) | Inner | K | Strain at step 29 |
|---|---|---|---|
| 9/16 | 8 | 8 | 26 % / 6.5e2 |
| 9/16 | 16 | 4 | 40 % / 7.6e2 |
| 9/16 | 32 | 2 | 213 % / 9.3e4 |
| 9/16 | 4 | 16 | 12 % / 9.8e2 |
| 9/9, 1024 per patch | 32 | 4 | 33 % / 3.3e2 |
| 9/16 | 32 | 6 | 13.6 % / 2.0e2 |

Tier A at K = 16 reached 83 %.

- Patches help, but they do not converge.
- Raising the inner iterations at fixed total work makes things worse, because the Jacobi seams between patches lag.
- Larger patches do not fix it.
- Conclusion: one sweep moves support only a short distance, and that is the real limit, so a coarse (two-level or multigrid) correction is the next candidate, as the coordinator suggested.
- The Tier B kernels (`solvePatches`/`commitPatches`), `lib/cloth/patches.ts` and the oracle patch schedule are committed but not wired. The GPU encode loop and the timing run are still to do.

## 2026-11-09 RESULT: AVBD strain limit at 10 % (coordinator's request)

- Moving the limit from 2 % to 10 % does not change the divergence materially at 64² below K ≈ 192.
- The cause is the stiffness ratio h²k/m ≈ 2200 of the membrane itself, combined with Gauss-Seidel moving support only about K rings per step. An active AVBD limit just adds more stiff terms on top.
- The K-sweep notes above apply unchanged.

## 2026-11-09 COORDINATION (C1 → C2)

- Added `ClothGPUSolver.stepStart`, the x_t segment, as requested.
- **Scene contact.** The Bridson scenes now hold their static solids as scene voxel solids, built as fill and procedural patches in `bridson2002SceneDescription`. C2 needs a scene solid-world → `ClothSolidSource` adapter, ground as `lattice-box` and the other parts as `sdf` with each solid's μ, so that `ClothSystem.build` can attach contact.
  - Until it exists, the cards say the cloth passes through the solids.
  - The hook point is `ClothSystem.build` in `lib/cloth/system.ts`. I'll wire it once the adapter's interface lands.
- **M6.** The gates are `ClothDocument.pendingSolids`. When M6 lands, a moving solid should be removed from `pendingSolids` and supplied as a posed body instead.

## 2026-11-09 RULING (0) DONE: the pose is read back only on a pick or an edit

- `ClothSystem` no longer reads the pose back on the step cadence. Every `CLOTH_STATUS_INTERVAL_STEPS` (4) steps it copies only the 4-byte status word, the non-finite watchdog, which fails the system explicitly with `cloth-non-finite`.
- New `readClothPose()` on the method contract. The system coalesces concurrent requests into one 'Cloth pose read' submit.
  - The renderer, worker and client carry it as a `read-cloth-pose` request. Cloth surfaces are published to the scene only when the pose actually changed.
  - The viewport holds a cloth-scene press (button 0, interact mode, no shift) until the pose arrives, then replays the press and any queued moves. Picking and cutting therefore see the current pose.
  - Edits and history (`capture`) read the pose on demand.
- `tests/cloth-system.test.ts`: six ticks leave the published tick at 0, so no readback happens on step. Two concurrent `readClothPose()` calls resolve.

## 2026-11-09 RESULT (1): what diverges, with the oracle

The question was which of Chebyshev, the adaptive init or the AVBD warm start causes the divergence. The setup: 64² pinned sheet, K=8, BDF2, 2 % limit, 30 steps, Float64 oracle (`fluid-scratch/cloth-c1/diverge.ts`).

- H = KE + PE + elastic.
- G is the step's incremental potential, which VBD minimises.

| variant | Hmax / H0 | H-growth steps | G non-monotone steps | strain at step 29 |
|---|---|---|---|---|
| base (adaptive, AVBD pins, ρ = 0) | 1.89 (Hmax 3.70) | 23/30 | — | 62 % |
| hard pins (no AVBD pin duals) | 8.2 | 27 | **0** | 198 % |
| hard pins, no limit | 8.0 | 27 | 0 | 185 % |
| AVBD duals reset every step (no warm start) | 4.7 | 27 | 30 | 210 % |
| implicit Euler instead of BDF2 | 2.9 | 25 | 6 | 76 % |
| inertial init | 3.0 | 27 | 11 | 89 % |
| **previous-position init** | **1.00** | **0** | 0 | 2.3 % |
| K = 32, hard pins | 2.5 | 24 | 0 | 128 % |

- **Chebyshev is off (ρ = 0)** in every run, so it is not the cause.
- **The AVBD warm start is not the cause either.** Resetting the duals still diverges, and hard pins without any duals diverge while G decreases monotonically in every iteration of every step.
  - So the per-vertex Newton step with backtracking does what the ruling says: G never rises.
  - But a G that falls monotonically within each step does not stop H from growing across steps.
- **The cause is the extrapolating initial guess (adaptive or inertial) combined with the solver's limited reach.** K sweeps move support only about K rings.
  - The unconverged far field keeps the extrapolated momentum. v = (x − x̂)/h then turns that residual into kinetic energy.
  - The previous-position guess does not grow, because it has no extrapolated momentum to keep. But it is wrong in free fall: it over-damps.
- **Conclusion: this is the propagation problem, not a robustness bug in the local solve.** The fix is long-range support, item (3).
  - Measured on the oracle at both sizes, the two-level correction removes the growth (next entry).
  - The GPU regression "no energy growth at 150², K = 8" will be the GPU two-level correction's acceptance test. It is not in the tree yet.

## 2026-11-09 RESULT (3): long-range support, measured in the Float64 oracle

Setup: K = 8, 30 steps, adaptive init, 2 % limit (`fluid-scratch/cloth-c1/longrange.ts`). The coarse variants add one coarse correction per outer iteration (Galerkin PᵀHP assembled at the step's initial guess, exact dense solve, global line search on G), then one fine sweep.

| method | 64²: max strain, Hmax/H0 | 150²: max strain, Hmax/H0 | coarse dof |
|---|---|---|---|
| plain VBD | 62 %, 1.89 | **436 %, 16.4** | — |
| previous-position init | 2.3 %, 1.00 | 0.2 %, 1.00 (over-damped, wrong free fall) | — |
| (a) two-level, bilinear 8×8 | 2.9 %, 1.017 | **4.3 %, 1.000** | 192 |
| (a) two-level, bilinear 16×16 | 2.9 %, 1.000 | **3.5 %, 1.000** | 768 |
| (b) aggregation (HPBD-like), 8² blocks / 10² blocks | 3.7 %, 1.000 | 6.1 %, 1.000 (10² blocks: 225 aggregates) | 192 / 675 |
| (b) aggregation, 19² blocks (8×8 aggregates) | — | 5.7 %, 1.137 | 192 |
| (c) long-range attachments (AVBD inequality at the geodesic rest distance to each pin) | 7.9 %, 1.031 | — | 0 |
| (c) + bilinear 8×8 | 2.9 %, 1.044 | — | 192 |
| reference K = 192 | 2.1 %, 1.000 | — | — |

- **Pick: (a), the two-level Galerkin correction with bilinear prolongation from an 8×8 coarse lattice.**
  - At 150² it takes the strain from 436 % to 4.3 % and keeps H ≤ H0.
  - Its 192-dof coarse system costs a 192³/3 ≈ 2.4 MFLOP factorisation per step and a 74 KFLOP solve per iteration.
  - 16×16 buys only 0.8 points of strain for 4× the dof (64× the factorisation).
  - Aggregation at the same coarse size (19² blocks) is worse, at 5.7 % strain, and lets H grow by 14 %.
  - LRA alone leaves 7.9 % and helps only pinned cloth.
- Planned GPU form, all on the GPU with no float atomics:
  1. **Assemble once per step, at the initial guess.**
     - An element pass computes the full analytic element Hessian blocks:
       - membrane: H_ij = A (J_i D J_jᵀ + σ_ij I);
       - hinge: Gauss–Newton k g_i g_jᵀ, the same as the vertex block now;
       - damping: in the same form;
       - edges: (k − f/l) nnᵀ + (f/l) I.
     - Contributions go per fine vertex into its 3×3 coarse-node neighbourhood.
     - One workgroup per coarse node pair then does a fixed-order reduction.
  2. **Factorise** the 192-dof system in one workgroup, Cholesky in workgroup memory.
  3. **Each outer iteration:**
     - restrict the gradient (one vertex pass and a per-node reduction);
     - coarse solve;
     - prolong;
     - global line search on G (one element-energy pass plus a reduction);
     - then the colour sweeps.
- **Answer to "does softening the membrane help?"** No. The hard limit carries the same propagation problem.
  - Membrane divided by 10 (h²k/m ≈ 220) with the 2 % AVBD limit: strain 52 % at step 29, H bounded (Hmax = H0).
  - Divided by 100 (≈ 22): 58 %, H bounded.
  - Without the limit, the softened sheet simply stretches: 258 % at ÷10, 430 % at ÷100, H growing 2.3× and 1.4×.
  - So softening removes the energy growth only by moving the stiffness into the AVBD limit. The limit is then violated by 25–30× at K = 8, because it has to propagate as far as the membrane did.
  - The 10 % limit gives 72.7 % (entry above).

## 2026-11-09 MEASUREMENT (2): where a colour pass goes, and 5.4× off the per-iteration cost

Setup: Bridson tablecloth 150² (22.5K vertices), K = 8, 20 steps, timing mode.

**Per-dispatch GPU time.** Each dispatch was timestamped in its own pass (`fluid-scratch/cloth-c1/prof/passes.mts`).
- Dawn merges all passes of one encoder into one Metal encoder, so xctrace alone cannot split them.
- Timestamps here are quantised to about 6.5 µs.

**xctrace with counters** (`prof/profile.mts --counters`, through the runner) on the original serial kernel (one thread per vertex):
- 20.5 ms/step;
- colour passes 0.53 ms each × 40 per step = 99 % of the step;
- occupancy 3.7 %, ALU 6.8 %;
- LLC limiter 49.5 %, buffer-read limiter 35 %, read bandwidth 48 GB/s (≈ 1 GB/step), MMU limiter 15 %.

Root causes, in order of size:

1. **Every Jacobi/PSD projection and every hinge array spilled to stack memory.**
   - `projectPSD` used dynamically indexed `array<f32, 9>`, and the hinge code used `array<vec3f, 4>` with `x[k] = xv`. Metal puts dynamically indexed private arrays in device stack memory, which was most of the 1 GB/step of traffic.
   - Fixes:
     - the membrane PSD projection became exact and closed-form (the block's range lies in span(F₀, F₁): a 2×2 eigenproblem in that plane, with the normal eigenvalue σ < 0 clamped to 0);
     - the hinge, damping and frame code select by vertex slot instead of indexing.
   - Read bandwidth fell from 48 to 16 GB/s.
2. **The line search failed on f32 rounding, not on ascent.**
   - Histogram of accepted trials: 54 % at α = 1, then 29 % at α = 1/64 and 6 % at 1/128. Mean 3.65 energy evaluations per update.
   - E(x + d) − E(x) was formed as the difference of two nearly equal f32 energies whose rounding exceeds 4 ε Σ|terms| when strains are near ε.
   - The Float64 oracle shows these vertices are converged: |g| ≈ 1e-16, E ≈ 1e-33.
   - Fix: the Armijo test now uses an exact difference-form ΔE.
     - Strain, rate and residual increments are linear or quadratic in d.
     - The rounding floor is first order in the base quantities.
     - Contact (C2's energy) is still a plain difference.
   - Mathematically this is the same test. Measured afterwards: 100 % accepted at α = 1, 1.00 trials.
3. **One 32-lane workgroup per vertex diverged on element kind**: faces, hinges and edges in one SIMD group ran one after another.
   - Replaced with `vertexLanes` lanes per vertex (default 2; 16 vertices per 32-lane workgroup). Every lane walks faces, then hinges, then edges, so the whole SIMD group runs the same element kind at the same time.
   - Per-colour times (µs, interleaved): lanes 1 → 140, 2 → 100, 4 → 106, 8 → 123, 16 → 140, 32 → 220. The old serial kernel took 172 after fixes 1 and 2.
4. **Greedy colouring left a one-vertex sixth colour.** It cost 35 µs × 8 dispatches per step for one vertex.
   - Bounded Kempe-chain interchanges (component ≤ 256) now empty a small last class. The tablecloth has 5 classes.
   - A mesh-wide chain also empties it, but reshuffles classes and cost +12 % per pass.
   - Step: 5.6–5.8 ms → 5.0–5.2 ms, interleaved.

**Result** (all GPU tests pass against the oracle, including the new lanes-per-vertex parity test):

| | step (K = 8) | colour pass | trials per update |
|---|---|---|---|
| before | 22.5 ms | 530–545 µs | 3.65 |
| after | **5.0–5.2 ms** | **~100 µs** (4700 vertices) | 1.00 |

- After the fixes the colour pass reads at 107 GB/s, with these counters:
  - buffer-read limiter 62 %;
  - LLC limiter 48 %;
  - MMU limiter 39 % (TLB miss rate 3.4 %);
  - ALU 14 %;
  - occupancy 11 % (a colour has 9.4K threads at 2 lanes, about 10 % of the machine).
- Splitting it: the system pass costs about 72 µs and the single line-search trial about 37 µs (with the line search skipped as a probe: 72 µs at 4 lanes).
- Removing all damping reads (40 % of the requested bytes) saves only 10–15 %. Each colour pass touches about half of every element record: about 12 MB per pass at 107 GB/s.
- **Still 3.3× off the 30 µs target.** Next, in order:
  1. Keep the system pass's per-element base values (F₀, F₁, hinge angle, edge length) in registers, with fixed per-lane slots sized from the mesh's maximum valence. The trial then needs no reloads, about −1/3.
  2. Precompute the damping rate offsets per element once per step (rate = Tᵀ(F − F̂)), which removes the x̂ reads.
  3. Pack each face's and hinge's records together (AoS) for line utilisation and TLB.
  4. Shorten the dependent-load chain: colour slot → star record holding element and vertex ids.

## 2026-11-09 RULING (4): scene contact wired (C2's adapter in `ClothSystem`)

Every scene document now runs with C2's OGC detector: self contact plus the scene's solid world. There is no contact-free path in the system.

- **`lib/cloth/scene-contact.ts` (new).**
  - `CLOTH_SCENE_CONTACT` holds the authored values. They are C2's gate values: kc 100, εv 1e-2, rqBase 3e-3, rqMax 0.04, s 2.5, contacts 128, grid 32 (N+T+E), buckets 2^18, round cap 4, pair slots 840 N. 840 N covers the gates' measured peaks of 809 N and 776 N.
  - r is the thickest material (`thickness_m`); it must lie in (0, rqBase).
  - μ comes from the new `ClothSolverSettings.selfFriction`. It is validated, and the Bridson documents carry the paper's 0.3.
  - Solid sample capacity is the field's exact sample count.
  - Failing bits are contactOverflow | gridOverflow | pairOverflow | nonFinite | solidPenetration | solidBounds, as in the M2 gates. roundCap, envelopeCap and uncertifiedPair are reported, not halted on.
  - Terrain or rigid bodies in a cloth scene fail explicitly, because contact does not see them. A patch the adapter cannot follow also fails explicitly.
- **`ClothSystem.build`** follows this order:
  1. build the field (once per solid world and cloth frame);
  2. create the detector;
  3. create the solver with `contact` and `hooks`;
  4. call `bind`, then `setSolidField`.
  - A rebuilt (cut) document keeps its frame, so it reuses the live field.
- **`stepTo`:** each document gets one compute pass: `encodeFrameBegin`, then the owed `encodeStep`s.
- **Status channel:** the status mirror copies the detector's status word beside the solver's. That is 4 bytes, and never the pose. A failing bit halts with `cloth-contact`.
- **`adopt`** follows a solid edit: it rebuilds the field after the call returns and uploads it with `setSolidField`, and the clock waits. If the adapter cannot follow the edit, the scene halts with `cloth-solid-edit`.
- **C2's defect fixed:** `bridson2002SceneDescription` now puts the procedural bake's patches first. `proceduralSolidField` is defined for all three scenes, both as built and as the catalog opens them, and there is a regression for it.
- **Card status updated:** contact is on. The text now names the two-level correction as the fix for the 150² divergence.

GPU results (runner, rc 0, peak GPU memory 2.26 GB):

| run | setup | ms per tick (fenced each tick) | result |
|---|---|---|---|
| rotating sphere, 22,500 vertices, K = 64, 20 ticks | 5.2 s (field, samples, compile) | **501** | lands on the held sphere. Closest vertex 0.3 mm off it; cap 0.1 mm above the top. Status 12 (roundCap, envelopeCap: reported). 3386 solid contacts; peak pairs 254 N of 840 N; peak grid 1.82 M of 4.29 M; peak contacts 14 of 128 |
| tablecloth, shell only, free fall, 18 ticks | — | **824** | free fall, as before |
| dry 8×8 sheet beside the oracle | — | — | 1.2e-4 m from the contact-free oracle at K = 12. Truncation bounds each iteration's move; the tolerance is now 3e-4 m |

The solver alone runs at 5.0–5.2 ms per step on these sheets. **Contact at K = 64 costs about 100× the solve.** The hooks run after every colour and every iteration, which is 320 truncations and 64 decides per step. That cost is C2's to attack. The K reduction that the two-level correction should allow also cuts it in proportion.

Notes for C2:
- `clothSolidBand` uses √3 exactly, but the bricks' f32 `maxLipschitz` rounds to 1.7320512. With that band, `setSolidField` rejects even a bare container shell (band 0.10442305 m against 0.10442307 m needed). `clothContactBand` adds 1e-4 relative on L.
- HEAD e426a6bb's `clothSolidSamples` (κ_min = 3r/e²) ran more than 15 minutes of CPU on the 8×8 test sheet (r = 5e-4, e = 0.057) in an empty scene. Your uncommitted turn-floor version builds the same field in 55 ms, so the GPU system tests need it to land.

Tests:
- `tests/cloth-system.test.ts`: the bake-leads regression; contact settings and explicit failures (self-friction validation, thickness, terrain, rigid bodies, carved solid); the rotating-sphere contact test on the GPU.
- check:types is clean.
- test:unit has 2 foreign failures: `cloth-contact-solid` (C2, in progress) and `reservoir-particle-cost` (lanes T/E).

## 2026-11-09 RESULT (5): two-level correction on the GPU, topology coarse space, contact-aware

What landed (coordinator notes on the two-level correction):

- **P comes from the mesh topology** (`lib/cloth/coarse.ts`), not the (u, v) grid.
  - Seeds are farthest points by rest geodesic distance (Dijkstra on rest edge lengths), one per connected component first.
  - Each seed carries a tent of radius 1.5× its cover radius. Rows are truncated to 4 entries and normalised to a partition of unity.
  - Cuts, holes and non-rectangular documents need no special case. P is rebuilt only when the topology is (edit time).
  - More pieces than nodes (40) fail explicitly.
- **A_c = PᵀHP per step, on the GPU.**
  - `coarseFactors` (one thread per face) writes each face's share as a quadratic form in its corner weights. The share covers membrane GN tangent plus PSD-clamped stress, face damping, the owned strain-limit edges, and the owned corners' inertia, pins and contact Hessian.
  - `coarseAssemble` runs one workgroup per node pair, as a fixed-order tree sum.
  - `coarseFactor` is one 1024-lane workgroup. It inverts A_c by a block symmetric sweep with the packed lower triangle in registers, so CM = 3·nodes < 128 and nodes ≤ 42.
  - A non-positive block skips that step's corrections and raises solver status bit 2 (`coarseIndefinite`).
- **Per correction:** an element-centric gradient (contact included), restriction, dc = A_c⁻¹ r, prolongation, a four-trial difference-form line search (1, ½, ¼, ⅛ with the summation floor), Armijo choice, the move, then the contact truncation (`afterCoarse` hook).
  - The correction runs every 2nd iteration (`coarseEvery`; CPU: every 2 ≈ every 1, 3.26 vs 3.21 % at K = 8).
- **`ClothSystem`:** the correction is on by default (`solver.coarse: false` turns it off).
  - The default K is 8 with coarse, and the initial guess is 'previous'. The extrapolating guesses overshoot an unconverged solve; the coarse step carries the bulk motion instead.
- **Status 12 is now visible.** `info.clothNotes` latches roundCap, envelopeCap and coarseIndefinite with their first-seen time. The diagnostics overlay shows "Cloth caps CAPPED/CLEAR".

Cost at 150², 40 nodes, 100 reps a kernel (µs):

| kernel | time (µs) | when |
|---|---|---|
| coarseFactors | 60 | per step |
| coarseAssemble (304 pairs) | 275 | per step |
| coarseFactor (inverse, one workgroup) | 339 | per step; device-memory Cholesky was 8.3 ms |
| elements + gradient + restrict + solve + prolong | 51 + 62 + 37 + 31 + 11 | per correction |
| trial + choose + apply | 108 + 11 + 9 | per correction |
| **total** | about 675 per step, plus about 320 per correction | |

The coarse solve itself (dc = A_c⁻¹ r, one workgroup) is 31 µs.

Step times without contact (ms/step):

| case | ms/step |
|---|---|
| plain K = 8 | 4.4 |
| plain K = 16 | 8.9 |
| coarse K = 8 | 7.0 |
| coarse K = 12 | 8.6 |
| coarse K = 16 | 11.1 |

K reached at 150², strain limit 2 %, 60 steps, every 2:

| K | max strain | p99.9 | p99 |
|---|---|---|---|
| 4 | 9.8 % | 4.19 % | 2.96 % |
| 8 | 5.5 % (at a corner pin) | 3.65 % | 2.53 % |
| 12 | 4.1 % | 3.21 % | 2.21 % |
| 16 | 3.8 % | 2.73 % | 2.12 % |

- Energy is bounded and decreasing in every coarse run. Plain adaptive at K = 8 explodes (H 1.96 → 1063 J, strain 1888 %).
- **p99.9 under 3 % needs K = 16, and p99 is under 3 % from K = 4.** The maximum stays at the corner pins.
- **Cut-sheet regression** (CPU, `tests/cloth-coarse.test.ts`): 16² cut sheet, K = 8. Max strain 2.31 %, and H stays at or below H0 (0.46589 vs 0.46652 J). Mean height is 13 mm off a K = 32 run; plain 'previous' is about 130 mm off and plain adaptive reaches 5.7 % with H growing.

**Contact.** The first version left contact out of the coarse gradient, A_c and the line search.
- With the truncation alone, it fought the barrier: up to 614 faces in the solid, strain 250–330 %, solidPenetration FAILING, even at K = 64.
- Now:
  - The gradient includes each vertex's `ogcAccumulateVertexContact` force.
  - A_c includes the owned corners' contact Hessian blocks.
  - The line search adds each vertex's own-entry energy change, `ogcVertexContactEnergy`. This is exact for the static solid and first-order for pair entries. A trial that enters a barrier's domain fails outright.
- Truncation still makes the move safe. C2's WGSL is used read-only; `lib/cloth/contact/` is unchanged.

Rotating sphere (22,500 vertices, 5 colours) through scene contact, at d72f279e, 60 ticks, fenced each tick:

| case | median tick | min tick | max strain | in solid | contact status |
|---|---|---|---|---|---|
| plain K = 64, adaptive | 145.7 ms | 48.8 ms | 44.2 % | 1 face | roundCap, envelopeCap |
| **coarse K = 8, previous** | **19.5 ms** | 13.3 ms | 19.7 % | 0 | clear |
| coarse K = 12 | 23.4 ms | 18.3 ms | 27.2 % | 0 | clear |
| coarse K = 16 | 28.9 ms | 19.3 ms | 26.1 % | 0 | clear |

At K = 8 that is 7.5× faster per tick, with half the peak strain and no caps taken.

The Bridson scenes still pin `CLOTH.iterations = 64` (`lib/cloth/scenes/bridson-2002/parameters.ts`, not my file). With coarse on, they run at K = 64. The system test shows 144 ms a tick and status 12 there. Setting their K to 8 is the owner's call.

Tests:
- `tests/cloth-coarse.test.ts`: topology space; exact free fall in one iteration; cut-sheet energy and strain; GPU one step against the Float64 reference for (K, every) ∈ {(1,1), (4,1), (4,2)}, within 6e-8 m; a 16-step GPU run, 3.5e-6 m, with save/load replay.
- `tests/cloth-system.test.ts`: the dry-scene oracle now runs the same default coarse correction.
- check:types is clean. test:unit has 1 foreign failure (`reservoir-particle-cost`).

## 2026-11-09 RESULT (6): colour pass on star records, 91.8 → 69.8 µs a colour

Setup: 150² sheet (22,500 vertices, 5 colours), K = 8, BDF2, 20 steps, timestamp per dispatch, mean over colours.

| kernel | HEAD b1ce8573 | now |
|---|---|---|
| `solveColourWide` (default, 2 lanes a vertex) | 91.8 µs | **69.8 µs** |
| `solveColour` (`vertexLanes: 1`) | 163.7 µs | 96.0 µs |
| step span, lanes 2 | 4.93 ms | 3.95 ms |

**What changed.**
- **Star records.** `packClothStarRecords` builds them at setup time; they are stored in the topo buffer.
  - Each colour holds, for every vertex, its incident faces and hinges, already seen from that vertex: face corner 0 and Dm' are pre-rotated, and hinges carry their 4 vertices, k, θ0, stiffness and kd.
  - Record j of slot i sits at 8(j·n + i) words, so the lanes of a SIMD group read consecutive records. This replaces the gather through the element and incidence tables.
- **Closed-form damping in the trial.** Damping is quadratic, so the trial's ΔE for damping is dg·d + ½dᵀHd, with a rounding floor r·|d| + quad. The trial no longer re-evaluates the face and hinge damping per element.
- **Shared code.** Face and hinge loading share `FaceIn` and `HingeIn` with the patch and diagnose paths, which still use `vertexSystem`.
- It fails explicitly at 2^24 or more vertices, or more than 256 materials.

GPU solver, coarse and system tests: 20/20 pass. GPU against the Float64 oracle: 6.40e-6 m after 20 steps, as before.

**Where the 70 µs goes** (probes remove one part and remeasure):

| probe | µs | part |
|---|---|---|
| base | 69.8 | |
| no hinges | 47.2 | hinges ≈ 23 µs, of which hinge damping ≈ 7 µs |
| no faces | 59.9 | faces ≈ 10 µs |
| no edges | 59.2 | strain-limit edges ≈ 10 µs |
| no trial | 58.2 | trial ≈ 12 µs |
| no hinges, faces, edges or trial | 21.0 | floor |
| floor without the own/inertia term | 17.1 | |

The floor, about 17 µs, is the vertex loads, the group sum and the write.

**Toward 30 µs.** Removing every element term entirely only gets to 21 µs, so 30 µs needs the element work cut about 4× (49 → 9 µs).
- Plan:
  - cache the per-slot record data in registers between the system and the trial;
  - interleave x_t and x̂ into one vertex record, so each damping gather is one line;
  - pack the edge records into the star.
- I expect the register caching and the edge packing to reach about 50 µs. Getting to 30 µs likely needs a cheaper bending term (a quadratic or precomputed-Q bending model). That is a physics change and the coordinator's call. I will not make it unasked.

## 2026-11-09 MEASUREMENT: the colour pass after star records, and what does not help

Method: 60 steps (20 steps gave ±10 µs of clock noise), mean over colours, each dispatch in its own timestamped pass.

The production step (one submission of 60 steps, wall clock) is **2.52 ms** at K = 8 on the 150² sheet, against 3.95 ms in split mode. Per-pass overhead is therefore about 10 µs of each split-mode figure, and a colour costs about 54 µs in production.

| probe (60 steps) | µs a colour |
|---|---|
| base | 63–66 |
| no hinges (system and trial) | 49.5 |
| no system hinges | 60.3 |
| no trial hinges | 59.5 |
| no faces | 61.3 |
| no strain-limit edges | 63.4 |
| no trial | 53.5 |
| no x_t and x̂ gathers (served from positions) | 62.1 |
| cheap hinge geometry (no atan2, cross or sqrt) | 61.4 |
| floor (no elements, no trial) | 15.9 |

- The line search accepts at α = 1 for every vertex: 11.16 M accepted at trial 0, none later. The trial is one pass.
- Counters: LLC limiter 61 %, buffer-read limiter 62 %, ALU 20 %, occupancy 9 % (9,000 threads a colour).

**Tried and reverted:**
- **Register-cached slots.** Each lane's 3 faces and 6 hinges were loaded up front and reused by the trial. Result: 119 µs, from spills.
- **Prefetch without the trial cache.** Result: 101 µs.
- **One-deep software pipelining** of the hinge loop. Result: 65.8 µs, no change.
- **More lanes per vertex.** 4 lanes: 84.6 µs. 8 lanes: 98.6 µs.
- **Subgroup-shuffle groupSum.** −4 % production, but it needs the `subgroups` feature, which the scene device does not request. Not taken.

**Conclusion.** No single term dominates: the element work is spread over hinge records, face records and gathers, all through the LLC. The kernel is not latency-bound: pipelining did nothing and more lanes made it worse.

Under 30 µs means cutting the bytes, not the instructions. The one change that does that at scale is a **quadratic (isometric) bending model** (Bergou et al. 2006):
- the per-vertex bending is a fixed 1-ring stencil Σ Q_vj x_j with Hessian Q_vv I;
- its trial ΔE and its Rayleigh damping are closed form;
- so the 12 hinge records a vertex reads twice, the hinge geometry and the hinge gathers all go.

This is a physics change. It holds for nearly inextensible cloth with flat rest shape, and the strain limit already enforces that. **It needs the coordinator's ruling.** I have not made it.

## 2026-11-09 RESULT (7), M4: the edit path carries the exact solver state

- **Before:** a mirrored pose (on a pick, an edit or a capture) published a restart state: zero BDF2 history, zero duals, `previousStep_s` 0. Every cut or undo therefore restarted with implicit Euler.
- **Now:** `PoseMirror` copies the solver's `stateRanges()`:
  - positions;
  - the velocity, v_{n−1}, Δx_{n−1} and pin-dual segments;
  - the edge duals;
  - the clock record.

  `ClothGPUSolver.decodeState` decodes them, and `readState` uses the same pair. The publication's `solverState` is the exact GPU state. `remapClothSolverState` (edit.ts) carries it through the cut by vertex origin. The rebuilt solver takes BDF2 from its first step, keeps the pin duals, and keeps the edge duals of edges whose ends are kept vertices.
- **Cost:** about 3.2 MB per pose read on the 150² sheet, against 0.7 MB before. That is on demand only (a pick, an edit or a capture), never on the step path.
- **Regression** (`tests/cloth-system.test.ts`, dry scene):
  - the mirrored `solverState` deep-equals `readState()` after 6 ticks, with previousStep 1/60 and non-zero history and pin duals;
  - after the cut, the rebuilt solver has previousStep 1/60 and the f32 clock of the cut;
  - its pin duals are identical, and its history and edge duals are non-zero.

  Save and load go through the same publication, so they now keep the exact history too.

## 2026-11-09 RESULT (8), ruling (1): quadratic bending is the only bending model; subgroup sums

**Model.** Bergou et al. 2006 quadratic isometric bending replaces the dihedral hinge everywhere (oracle, coarse reference, GPU colour passes, coarse kernels, diagnostics). Per hinge, weights w_i = (∇θ_i at the flat rest) · n̂ (Σw = 0); s = Σ w_i x_i; E = (k/2)|s|², damping (kd k / 2h̃)|Σ w_i (x_i − x̂_i)|². k is unchanged (k_b 3|e|²/(A0 + A1)), so |s| ≈ |θ| for small folds. Gradient k w_v s, Hessian k w_v² I, both exact.
- **Non-flat rest fails explicitly.** `prepareClothTopology` throws on any rest dihedral beyond `CLOTH_FLAT_REST_TOLERANCE_RAD` (8 · 2⁻²³ rad), naming the hinge, its vertices and its edge. Tested.
- **Deleted:** hingeGeometry, the wrapped-angle helpers, the hinge records of the star stream, the per-hinge damping tangents (the `state1.y` segment, 3H records of state) and the trial's hinge loop.
- **The colour passes read a per-vertex stencil.** Setup builds it in Float64 (`clothBendingStencil`): g = Σ_j Q_vj (x_j − x_v) + Σ_j Qd_vj (u_j − u_v)/h̃, H = (Q_vv + Qd_vv/h̃) I, with Q = Σ_h k w wᵀ and Qd = Σ_h kd k w wᵀ. A stencil record is one vec4 (j, Q_vj, Qd_vj, 0) in the colour-major star layout. The trial's bending change is closed form, through the damping block. Tier B's `vertexStep` keeps the per-hinge form; the two forms are the same energy.
- **Subgroup sums.** solveColourWide sums a vertex's lanes with a subgroupShuffleXor butterfly when the device has `subgroups` with subgroupMinSize ≥ the lane count. The scene device already requests the feature (`fluidExecutionDeviceFeatures`). Otherwise it keeps the workgroup-memory tree. `subgroups: false` forces the tree; `true` without the feature fails.

**Cut bug that the explicit check exposed.** A cut through the deformed sheet produced a rest-collinear sliver: area 2·10⁻¹⁹ m² beside a 3 cm edge. Its rest normal was rounding, giving a "rest dihedral" of π or −1.83 rad. The old hinge model would have folded toward that angle. The cause was `convexTriangles` (edit.ts), which tested a polygon corner for collinearity against an absolute 10⁻²⁰ m², so a T-junction corner inserted by `conformingPieces` was fanned through. The test is now relative (ON_PLANE_RELATIVE of the two sides' lengths), so such pieces fan from the centroid.

**Oracle match:**

| Check | Result |
|---|---|
| 20-step scripted BDF2 run, strain limits + Chebyshev | 4.75e-6 m (was 6.40e-6) |
| one step, every initial guess | < 1e-6 m |
| every lane layout × {subgroup, workgroup} sums | < 2e-6 m |
| stiff fold (k_b 2e-3, damping 0.01, 2 cm perturbation), 1 and 4 lanes | 6.1e-8 / 9.2e-8 m |
| Float64 stencil gradient vs per-hinge Σ k w s | < 1e-12 relative; rows sum to zero |
| coarse GPU vs Float64 reference | 2.7e-8 to 5.3e-8 m one step; 3.7e-6 m over 16 steps |

**Per colour, 150² tablecloth, K = 8, lanes 2, 60 steps.** Each dispatch is timed in its own pass. HEAD was run in the same session from a `git archive` copy.

| | colours c0..c4 (µs) | step span | wall ms a step |
|---|---|---|---|
| HEAD (hinge records) | 71.5 / 76.3 / 76.2 / 78.5 / 76.3 | 4.35 ms | 2.54 |
| quadratic bending | 58.6 / 65.7 / 67.3 / 68.3 / 67.0 | 3.85 ms | 1.80 |
| + subgroup sums | 48.2 / 56.1 / 55.8 / 56.5 / 56.5 | 3.40 ms | 1.72 |

Lanes 1 (serial starStep) with quadratic bending: 58.3–67.9 µs.

Tests: `cloth-oracle`, `cloth-gpu-solver` (new stencil regression), `cloth-coarse`, `cloth-system` and `cloth-topology` pass. The scene tests run with `subgroups` requested. They were verified on HEAD plus these files, because C2's in-progress `ogc-detector.ts` edit in the working tree currently mismatches the "ogc apply" bind group (6 entries against 7).

## 2026-11-09 RESULT (9): C2's M6 moving solid wired into ClothSystem; undo regression through the system

**Wiring (COORDINATION items 1–4 of progress-lane-c2.md).**
- `CLOTH_CONTACT_FAILING` adds `bodySqueeze | heldConflict | bodyPose`.
- The scene's moving solid is a scene rigid body with a prescribed pose track: `motion "static"`, a `poseTrack`, and no orbit, motor or kinematic plugin. `clothSceneMovingSolids` accepts at most one. Any other rigid body, or a second one, fails explicitly at setup.
- `buildClothBodyField` bakes the body once in its body frame, on the scene's lattice cell, with the contact band and the body's friction. Only a sphere bakes; any other shape, or a cut body, fails explicitly. The field is cached across rebuilds.
- `config.body` is sized exactly: field words; sample capacity = max(samples, ⌈hull edges / 8⌉). The edge term is new and applies to the static field too: a rounded body has more hull edges than samples (sphere r 8 cm: 254 samples, 2240 edges).
- `poseCapacity` is `CLOTH_BODY_POSES_PER_FRAME` = 5, asserted to be at least the catch-up steps + 1.
- `setBodyPoses` gets owed + 1 cloth-frame poses from `sampleRigidPoseTrack` before each frame's pass.
- The `cloth-pending-solid` halt now skips the scene's moving solid. Any other pending solid still halts, with "is not the scene's moving solid".
- **Bridson variant.** `bridson2002SceneDescription(source, { movingSolids: "rigid-body" })` makes each scene's moving solid a pose-tracked rigid sphere. It carries the track, translated, including `localOffset_m`, and the solid's friction; the solid is no longer held or pending. `contact-solids.ts` leaves such a body out of the static field.
- **The product default stays `"held"`.** A rigid sphere is not yet drawn, picked or cut in a dry scene: `visibleToolSurface` and the dry renderer see only held voxel solids. AGENTS.md forbids a visible surface that editing cannot address, so flipping the default needs renderer and editor support for a posed rigid sphere first.

**Full-length runs, rigid-body variant, K = 64.** Fenced per tick, pose read every 6 ticks; HEAD plus these files.

| Scene | Ticks run | Halt | Status (OR) | Max intersections | Faces in solid before halt | ms per tick | Strain p99 | Max edge strain |
|---|---|---|---|---|---|---|---|---|
| curtain | 122 / 400 | heldConflict (t 2.03 s, ball rising) | 1676 | 0 | 0 (1476 at the halted step) | 221–261 | 10.00% | 20.8% |
| rotating sphere | 119 / 533 | heldConflict (t 1.98 s, spinning) | 1036 | 0 | 0 (1 at the halted step) | 136 | 10.02% | 167% |
| tablecloth | 224 / 353 | heldConflict (t 3.73 s, ball on cloth) | 1036 | 0 | 1 at ticks 36–42; 2 at tick 222 | 122–151 | 10.5% | 52–55% |

- Status at the curtain's halt: 1676 = heldConflict + bodySqueeze + solidPenetration + envelopeCap + roundCap. The others: 1036 = heldConflict + envelopeCap + roundCap.
- The held conflict pairs are cloth edge vs body hull edge, for example (cloth 21634–…, body samples 49691, 50127).
- The strain limit is the paper's 10%, and the bulk sits on it (p99). The isolated maxima are edges caught on the frustum top edge (rotating sphere) or on the table edge.
- The rotating sphere's 167% is late, at tick 108. In the held variant it reached 35% by tick 18.

**C1 → C2 finding: cloth resting on the moving solid halts heldConflict.** The conflict is a cloth edge against a body hull edge. It recurs with the same pair at heldRounds 3, 6 and 12, and with friction 0 or 0.2. A slower sphere halts the same way.
- With a body sample in the pair, `checkConflict` is always "moving", so any pair with no separating plane at X conflicts every held round.
- Repro: a 24 × 24 sheet on a 1 cm lattice falling 4 cm onto a sphere (r 10 cm) that slides 3 cm and turns 0.6 rad. It halts at tick 29.
- On a 5 cm lattice (the empty scene's default) the same scene halts at tick 12. There a vertex also lies 3.7 mm inside the true sphere, which is the trilinear error of a 5 cm bake (h²/8R), not a contact miss.

**Tests (tests/cloth-system.test.ts).**
- CPU: the rigid-body variant validates for all three scenes, with one pose-tracked sphere (track and friction carried) that is neither held nor pending. The rigid-body rejection message is updated.
- GPU: the scene's moving solid is baked and posed. The clock passes its pending start; at tick 8 the sheet is in contact (body contacts > 0), with no vertex inside and no failing bit. Two moving solids, a dynamic body and a cup each fail explicitly.
- GPU, `todo`: the sheet lands on the moving sphere and rides it for 36 ticks. Blocked on the heldConflict above. It becomes a plain test once C2's M6 holds resting cloth.
- GPU, undo: 6 ticks, read the pose, take the editor's history endpoint (`snapshotSceneClothPose`), cut, step 3 ticks, then stage the endpoint (undo).
  - The uncut document is rebuilt: triangle count and document restored, topologyRevision 2, tick 6.
  - Positions, the BDF2 history and the AVBD duals equal the endpoint's; the clock is Math.fround of it.
  - 3 more ticks match a never-cut 9-tick run to **0 m** (bit-identical).
  - Redo restages the cut document.
- `cloth-system` GPU file: 12 pass, 1 todo. `check:types` is clean. `test:unit` has only the known `reservoir-particle-cost` failure.

## 2026-11-09 RESULT (10): Bridson K = 8, set from measurement

**Window.** All three scenes, rigid-body variant, the first 114 ticks. That is the window all three reach before C2's heldConflict; neither variant can run full length yet (the held variant halts at its pending solids). Fenced per tick, pose read every 6.

**Strain measure.** The paper's limit is 10%, and the bulk of the cloth sits on it, so "strain under 3%" is read as excess over the limit. The measure is p99 edge strain at most 13%.

| K | p99 strain, curtain / sphere / table | edges past 13%: worst count | ms per tick, curtain / sphere / table | KE |
|---|---|---|---|---|
| 8 | 12.0 / 10.0 / 8.6% | 758 / 7 / 35 | 46 / 27 / 25 | decays (curtain last 3e-5 J) |
| 16 | 10.4 / 10.1 / 9.9% | 442 / 8 / 41 | 65 / 44 / 42 | decays |
| 32 | 10.0 / 10.0 / 10.0% | 288 / 30 / 83 | 101 / 67 / 74 | decays |
| 64 | 10.0 / 10.0 / 10.2% | 13 / 18 / 170 | 170 / 120 / 133 | decays |

Notes on the table:
- The sphere scene halts heldConflict at tick 90 (K 8) and tick 83 (K 16); at K 32 and 64 it reaches 114.
- The curtain's p99.9 is 17.6% at K 8 and 18.2% at K 16.
- The single-edge maxima (21–103%) sit where cloth catches a solid edge (the frustum top, the table edge). They do not fall with K.

**Choice: K = 8.** It is the lowest K that meets the criterion on all three scenes: the curtain's p99 is 2.0% over the limit, and energy is bounded and decaying.
- K = 16 holds the curtain's bulk within 0.4% of the limit, at 1.4–1.7× the tick cost.
- The provenance in `parameters.ts` says this is our solver's choice, not the paper's, and records the numbers.
- At K = 8 the existing free-fall capture test runs at 38 ms per tick and the held rotating-sphere test at 30 ms per tick (they were 101 ms per tick at K = 64). Both still pass, with no vertex inside and status 0.

## 2026-11-09 RESULT (11): strain against equilibrium (ruling 1), and what breaks the hard limit (ruling 2)

**The equilibrium.** The Bridson material is 1 kg/m² at k = 200 N/m, Poisson 0. A hanging strip carries the weight below it, so its strain is ρ g d / k, where d is the hanging length below.
- For 1 m of cloth below, that is 9.81 / 200 = **4.9%**, not 0.5%. The ruling's figure is 10× low.
- Exact equilibrium of band j (rows j, j + 1) with area-lumped mass is uniaxial StVK: k λ(λ² − 1)/2 = ρ g d_j.
- For the 1.7 m Bridson sheet, the top band is 7.46%, the middle 3.90% and the bottom 0.05%. The bottom row drops 6.56 cm.
- The paper's 10% limit is a safety bound. On a hanging sheet it is inactive everywhere except next to point pins.

**Static hanging sheet.** 150 nodes across, 1 × 1.7 m, herring-bone, no contact. Bridson material, top edge pinned, 600 steps (10 s). Probe in fluid-scratch `cloth-c1/prof/hang.mts`.

| Correction | K | Top band (exact 7.457%) | Middle band (exact 3.900%) | Worst band error | KE at 10 s |
|---|---|---|---|---|---|
| two-level | 8 | 7.477% | 3.900% | **0.3%** | 2e-8 J |
| two-level | 16 | | | 0.2% | |
| two-level | 64 | | | 1.3% | 1.2e-4 J, still ringing |
| fine only, `previous` guess | 8 | 1.93% | | unconverged | |
| fine only, `previous` guess | 64 | 5.24% | | unconverged | |
| fine only, adaptive guess (the default) | 8 / 16 / 64 | | | explodes: 300–900% strain | |

- At K = 8 with the two-level correction the bottom drop is 6.56 cm (exact 6.56) and the max vertical edge is 7.50%, so the limit is inactive.
- With two pins, the top band carries nothing (it sags into the catenary) and the middle band holds 3.83% against 3.90%. The max vertical edge is 9.0% with the limit on and 17.5% with it off, so the limit is active only next to the pins, as it should be.
- **Defect, fine only, adaptive guess:** the sheet blows up and the hard limit breaks. No scene uses this path, because every ClothSystem scene carries the two-level space. I am recording it rather than fixing it now.
  - K = 64 is worse than K = 8 at 10 s. The extra iterations on the two-level path do not damp the coarse modes the way K = 8 does.

**The Bridson scenes are not "at the limit".** K = 8, first 114 ticks:

| Scene | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|
| curtain | 0.1–0.8% | 1.7–3.3% | 6.3–8.6% | about 10% |
| sphere | −0.1% | 3.2% | 10% | |
| table | about 0.1% | 3.5% | 8.5% | 10.4% |

- About 95% of the curtain's edges above 9.5% lie within 15 cm of the two pins. The sphere's lie on the cap and the table's at the leg tops and corners.
- RESULT (10) read p99 as the bulk. That was wrong: p99 measures local concentrations.

**K = 8 stays, re-justified.** Bulk strain matches the exact equilibrium at K = 8 with the two-level correction. The `parameters.ts` provenance now cites the hanging-sheet equilibrium.
- New GPU regression in `tests/cloth-coarse.test.ts`: "a hanging Bridson sheet comes to rest at its exact uniaxial equilibrium…" (4.7 s). It asserts band error under 1%, drop within 1%, max edge under the limit, KE under 1e-6 J and status 0.

**Ruling 2: the truncation wins over the limit.** Per-vertex truncation diagnostics accumulate the displacement each OGC truncation removes, around every encodeTruncate.

*K = 8 violators* (21–40%, at the frustum top and the table's leg tops and corners):
- Solid φ is 0.6–5 mm, so they sit at the solid.
- One end of each is truncated in every truncation, 0.5–1.1 mm per cut, against an excess over the limit of 0.8–2 mm.
- The AVBD dual is not the cause: λ = 0 and k is 5e2–2e3, far below kMax.
- The edge constraint's α = 0.95 carries the truncated violation forward: the target becomes limit + 0.95 · max(C0, 0), so λ stays 0 while only 5% of the excess is corrected per step.
- A/B with α = 0 on edges only: p99.9 falls to 10.0% and edges past 13% fall from 13 to 5, but the max is still 19%. The truncation re-creates the excess every step.

*K = 64, tick 24* (52–82%, in the free skirt, 12–20 cm from any solid and 5–50 mm from other cloth):
- Here the dual winds up: λ is 7–141 and k is 3e3–2e4.
- The truncation removed motion from 22104 of 22500 vertices in that step, 156 m in total.
- The worst vertex lost 349 mm over 76 cuts, the last cut 4.4 mm, against its 3.9 mm excess.
- Status carries roundCap and envelopeCap (12). roundCap 16 or 64 does not fix it (max 36–51%).
- The sphere scene at K = 8, tick 24: 19521 vertices truncated, 128 m removed.

*Removing the truncation* (scratch only) halts with solidPenetration at tick 6, so it is required. The cure belongs to the truncation (C2).
- The bound is per vertex and does not know the strain constraint, so it cuts one end of a limit-active edge and the other end walks on.
- The cure is either a truncation that preserves the edge constraint (cut both ends of an active edge together), or fewer conservative cuts on cloth far from contacts. The K = 64 skirt is truncated by self-contact bounds that are much tighter than the gap.
- **For C2 (via the coordinator):** at K = 64 nearly every vertex is truncated every step. The strain limit cannot hold against a bound that removes 156 m of motion per step.
- **For C1, pending a ruling:** α = 0 for strain-limit edges (not for pins) would stop AVBD accepting a carried violation. It helps partly. I have not changed the solver's α without a ruling.

## 2026-11-09 RESULT (12): moving rigid bodies in dry scenes render, pick and cut through the shared path (ruling 3)

**What a dry scene did before.** A pose-tracked sphere was drawn at its t = 0 pose (the host roster), because no rigid solver poses bodies when there is no fluid. The voxel-editor surface query passed straight through it, because only rounded plates were rigid editing surfaces. So the moving solid was render-only geometry that the cut tool could not address, against AGENTS.md.

**Now:**
- **Render.** In a scene with no fluid solver and no resident rigid buffer, `webgpu-renderer.ts` (`posePrescribedDryBodies`) poses every prescribed body (pose track or orbit) with `samplePrescribedRigidBody`.
  - The pose is taken at the time the presented state reached: the cloth's clock, which poses the same track for its contact.
  - The drawn poses are published exactly as a resident solver's readback is (`latestRigidBodyPoses`, `publishSceneRigidSurfaces`), so the worker snapshot, gizmos, hover and the shared surface query all see the drawn frame.
- **Cut spheres render exactly.** A sphere is the rounded plate with no core.
  - `rigid-cut-render.ts` adds `rigidSurfaceCutCount` and `cutSphereSurfaceHit`. Every sphere consumer with the cut atlas (SVO dry `bodyHit`, which shadows also use; the SVO rigid raster; the rigid raymarch) takes the rounded-plate patches and cut planes for a sphere with cuts.
  - An uncut sphere keeps its closed form.
- **Pick and cut.** `rigidBodyIsEditableSurface` covers every rounded plate, plus spheres whose pose is static or prescribed in a dry scene.
  - `roundedPlateSurfaceAt` hits them with `intersectCutSphere` (the same zero-core patches) at the published pose.
  - `sceneWithRoundedPlateCuts` stores a stroke's box in the body frame, as for plates.
  - `validateScene` accepts such cuts. On a wet scene's sphere it still fails explicitly ("…or such spheres in a dry scene"), because the fluid's rigid boundary does not cut spheres. A wet scene's spheres keep their object tool.
- **Cloth contact follows the cut.**
  - `buildClothBodyField` bakes the body's cuts in the same body frame: max(d, −box), which stays 1-Lipschitz. A remnant thinner than a cell is refused by the field's resolution check.
  - `ClothSystem` entries are keyed by the document *and* the moving solid (`entryKey`). An edit of the moving solid alone (a cut, a new track) stops the clock, reads the live pose, then rebuilds the entry from it with its BDF2 history, duals and clock, and the rebaked body field. The detector is re-sized for the new field, so there is no runtime pool growth.
  - A combined stroke that also cuts the cloth rebuilds from the edited document as before.
- **Catalog.** The three Bridson cards now open `movingSolids: "rigid-body"`.
  - `bridson2002SceneDescription`'s own default stays `"held"`, because C2's `tests/cloth-contact-scene-solids.test.ts` is written against the held layout (5 of its tests fail if the library default flips).
  - Flipping the library default is one line once those tests state `movingSolids: 'held'`.

**Regressions:**
- **CPU** (`tests/cloth-system.test.ts`, "the moving ball is a shared editing surface at its drawn pose…"):
  - The curtain ball drawn at t = 3 s is hit at its posed surface (to 1e-9 m), not at its t = 0 pose.
  - A solid stroke cuts it in its body frame: the near ray goes deeper to the cut face; the far side and the ground keep their exact hits.
  - Drawn at t = 3.5 s, the cut travels with the ball.
  - Save/load gives the same hits, and undo (the previous document) hits the uncut ball.
  - The cloth body field reads the removed cap as outside, and the far side is bit-equal to the uncut bake.
  - The wet-scene sphere is not an editing surface, and its cut fails validation.
- **GPU raster** (`tests/rigid-cut-render.test.ts`): a sphere with two oriented cuts rasters the same nearest visible surface as the picking oracle (3 µm), with more than 20 carved pixels, and stays the closed form uncut. The dry SVO shader compiles (0 errors), and the raymarch compiles through `rigid-rounded-plate-render`.
- **GPU cloth** ("a cut of the moving solid at edit time…"):
  - A sheet resting on the moving sphere at tick 12 has the sphere's cap above 6 cm cut through the shared rigid path.
  - The clock holds while the entry rebuilds. It resumes at tick 12 from the live pose (jump **0 m**).
  - The sheet falls the 4 cm onto the cut face: centre 3–8 mm above it at tick 36, no vertex inside the cut body (deepest +0.1 mm), no failure.
  - Undo (the history endpoint) restores the uncut body and the endpoint pose exactly, and steps on without failure.
  - One unexplained failure in 6 runs, from the first full-file run. The error text was lost, and it came before the landing log, so it was the rebuild or a halt during the ride. The next 5 runs passed.
- **GPU, todo** ("in contact, a history endpoint steps on as if never rebuilt"): a rebuild from an endpoint while resting on the moving sphere drifts 2 mm to 1.7 cm from the straight run 6 ticks later.
  - The endpoint lacks the detector's cross-step per-vertex state: rho_v, and "held in the last step".
  - The old free-fall undo test is still bit-identical, because no vertex is in contact there.
  - **For C2:** expose that state. **For C1:** carry it with the solver state.
  - The run-to-run spread (2.2 mm vs 16.7 mm) also says the detector is not deterministic in contact.
- The sheet-on-moving-solid test, a `todo` until now, passes after C2's 40d73e7f and is a plain test.

**Bridson rigid-body runs after 40d73e7f** (K = 8, full length):

| Scene | Halts heldConflict at tick | Other |
|---|---|---|
| curtain | 122 of 400 | max faces inside solid 1154 |
| rotating sphere | 128 of 533 | max edge 58% |
| tablecloth | 226 of 353 | |

- The held variant halts earlier, at its pending solids. The rigid-body variant is the furthest-running product path.
- For C2.

**C2's dirty `solid-field.ts`** (its uncommitted lifted-facet check) fails the held rotating-sphere GPU test at setup: "a lifted facet near (−0.80, −0.44, 0.50) enters the field by 1.00e-3 m". With HEAD's `solid-field.ts` (scratch copy) the test passes. This is not from C1's changes.

## 2026-11-09 RESULT (13): two-level only, the tick split, carried contact state, and two coarse failures traced to their causes

**Rulings (1), (2): the solve itself.**
- **Strain-limit edges use α = 0** (`CLOTH_AVBD.edgeAlpha`, oracle, coarse operator and WGSL). Pins keep α. The warm start λ ← αγλ is unchanged.
- **The fine-only path is gone.**
  - `ClothGPUSolver` always builds a coarse space (from the model's topology when none is passed) and always dispatches the correction. The `coarse` solver setting is removed and now fails validation ("'coarse' was removed").
  - A mesh that can't build a coarse space fails explicitly: more pieces than nodes, an uncovered vertex, a vertex with no face.
  - `coarseIndefinite` now halts the scene (`cloth-coarse-indefinite`). It is no longer a note.
  - A solver with contact must have `hooks.afterCoarse`, or construction throws. C2's working-tree `stepHooks` carries it.
- **Chebyshev is forbidden.** ρ must be 0, enforced in the solver and in validation. With the coarse step, the scripted ρ 0.9 test exploded on both the GPU and the oracle: residual 3e10.
- **Tests that used the fine-only path now use the two-level path.** The GPU-vs-oracle cases pass the oracle a `ClothCoarseCorrection`. Empty-contact bit identity is relaxed to < 5e-7 m (measured 9.3e-8, from FMA contraction).

**Two `coarseIndefinite` failures, traced to their causes** (the halt surfaced both):
1. **Undo of a cut (`sheetScene`): the cut was wrong.**
   - The A_c on the GPU was NaN on 14 of 40 nodes. Its source was the per-vertex contact Hessians of 12 vertices, which were pairs of coincident, unwelded vertices.
   - Cause: the box planes moved out by a *per-face* thickness support h/2 |n_f·m|. On a curved sheet, the two faces of an edge cut it at two points. The result was a slit, and contact saw distance 0, so n = gap/d was NaN. The entries also carried z = +inf.
   - Fix (`edit.ts`): the support is per vertex (area-weighted normal, thickest incident face) and blended by each point's weights, so both faces cut an edge at one point.
   - Second defect in the same cut: a plane splits a face's side wherever it crosses it, even outside the box along another axis. A disjoint neighbour kept the whole side, leaving a T-junction (a vertex 0 m from a non-incident edge, barrier Hessian 2e11).
   - Fix: a cross-face conforming pass. Points on an original edge are inserted into the face across it, and a disjoint face re-fans only when it receives one.
   - New regression (`cloth-topology`): a bowl cut at 0 and 5 mm thickness has the same vertices, faces and boundary edges, no vertex pair under 1 mm, and 0 T-junctions. Before the fix: 94 vs 86 vertices, 8 near pairs.
   - **For C2:** `ogcEvalEntry` turns d = 0 into NaN. Mesh contact entries at coincident features should fail explicitly rather than poison the solve. The solve's vertex line search silently rejected those vertices.
2. **In contact (moving-solid scene): the f32 block test was wrong.**
   - The factor tested each 3×3 pivot block by its determinant. A contact Hessian makes a block k n nᵀ + s I. Its determinant s²(s + k) cancels from k³ terms, and f32 loses it at k/s ≈ 10^3.5.
   - Measured on the GPU's A_c over the moving-solid run: the minor ratio fell to 6.5e-6 at a 1e6 contact spike.
   - Fix: invert blocks by 3×3 Cholesky (D⁻¹ = L⁻ᵀL⁻¹), whose pivots keep s to k/s ≈ 1e7. It is also cheaper: coarseFactor 0.39 → 0.31 ms.
   - New GPU regression (`cloth-coarse`): six blocks at k/s = 1e5 off the axes, with soft couplings. The factor says positive definite, and |A⁻¹A − I| = 4.6e-3. The old f32 determinant on those blocks is −3e6 to 4e6 (exact 1e5).

**Ruling (3): one tick, the split.** The catalog curtain, rigid-body variant: N 38100, T 75394, E 113493, 5 colours, K = 8, 40 coarse nodes, ticks 71–76, split-pass timestamps (`fluid-scratch/cloth-c1/r2/tick-profile.mts`).

| Group | Start of round | After this round |
|---|---|---|
| detection (C2) | 6.34 | 4.95 * |
| solve (colour sweeps, duals, frame) | 4.24 | 4.38 |
| coarse | 3.57 | 3.15 |
| truncate (C2) | 0.40 | 0.42 |
| GPU busy per tick | 14.55 | 12.89 |
| fenced tick (CPU encode) | 19.17 (2.36) | 13.60 (1.58) |

\* C2's working tree changed during the round.

- **Largest items now** (ms per tick):
  - solveColourWide 4.00
  - coarseAssemble 0.68
  - coarseTrial 0.60
  - coarseGradient 0.45
  - coarseElements 0.33
  - coarseFactor 0.31
  - coarseRestrict 0.25
- **Render handoff:** the normals are part of finishStep (0.03 ms), and the renderer reads the solver's buffers in place, so the handoff has no copy.
- **Landed: the full coarse step first.**
  - Nearly every correction takes α = 1, so coarseTrial evaluates only that trial. The shorter trials 1/2, 1/4, 1/8 run in coarseTrialBack and coarseChooseBack only after a rejection; otherwise those passes are uniform early exits.
  - Cost: trial 1.15 → 0.60 ms, plus 0.12 ms for both Back passes.
  - Validated: a 4 cm perturbed sheet rejects 3 of 4 full steps, and the GPU accepted counts equal the oracle's ([1, 3, 0, 0]), 8e-8 m apart. Hanging-sheet equilibrium is unchanged: 0.35%, stretch exact, KE 2e-8.
- **Measured, not landed:**
  - coarseEvery 4 fails the band gate (1.24% > 1%).
  - coarseEvery 3 passes at 0.77% and saves one correction in four (≈ 0.55 ms at 38K). This is the coordinator's call.
- **Against the 4 ms / 16K target:**
  - Scaled linearly, my share is ≈ 3.2 ms at 16K: solve 1.8, coarse 1.4 (factor and solve are fixed costs). C2's share is ≈ 2.3 ms.
  - The remaining big item is the colour sweep, 0.5 ms per iteration at 38K. Wider lanes measured worse earlier.
  - Next candidates: assembling A_c once per step at a lower cost (0.68 ms is pair-gather of face shares), and the colour sweep's memory layout.

**Ruling (4): `posePrescribedDryBodies` is tested.**
- The posing moved to `lib/core/dry-rigid-poses.ts` (`dryPrescribedRigidPoses`, `rigidBodyIsPrescribed`). The renderer calls it, and the fluid-solver check is inside it.
- CPU test in `cloth-system`: a dry scene's prescribed bodies are posed at the presented time, the published pose is what picking sees, and a wet scene returns undefined (its rigid solver owns the poses).

**Ruling (5) and C2's follow-up.**
- **The detector's carried state is part of the published solver state.** `ClothSolverStateData.contact = { reference (xyzw, world m), held }`, published with each pose (`readCarriedState` beside the mirror read).
  - It is restored on rebuild by `writeCarriedState`, for undo, load and adopt.
  - It is validated: 4n finite reference, n integer held flags.
  - A cut remaps it: kept vertices keep theirs; new vertices take the blended xyz with ρ = −1 and held 0.
- **The library default is `movingSolids: "rigid-body"`.** C2's tests state "held" explicitly, and `scenes.ts` no longer passes it.
- **Undo drift vs run to run** (GPU, moving-solid scene; endpoint at tick 12, 6 ticks on): the test now takes 3 straight runs plus a control without the carried state, and asserts nearest-straight-run drift ≤ the maximum pairwise spread.

  | Run | Drift / spread (m) |
  |---|---|
  | undo with carried state, nearest straight run | 5.7e-4 |
  | control, without carried state | 6.6e-4 |
  | straight runs, maximum pairwise spread | 2.7e-3 |

  - The earlier two-run form saw 1.6e-3 vs 1.8e-3 (pass) and 2.0e-3 vs 6.9e-4 (fail): the spread itself varies 4×.
  - Honest reading: the carried state is now exact through undo, but in this scene the detector's own nondeterminism dominates, so carrying it changes the drift little.

**Checks:**
- `check:types` clean.
- `test:unit`: 1672 pass, 1 fail (the known foreign `reservoir-particle-cost` "merge cap…").
- GPU:
  - cloth-coarse 7/7
  - cloth-gpu-solver 6/6
  - cloth-system 16/18
- The 2 cloth-system failures, "sheet lands on the moving solid" and "a cut of the moving solid…", both halt on **heldConflict**. They are not from C1:
  - Both passed at the start of the round.
  - They started failing when C2's uncommitted `ogc-detector*.ts` / `ogc-wgsl.ts` changes appeared.
  - Neither test cuts the cloth. The only C1 change in between was `edit.ts`.
