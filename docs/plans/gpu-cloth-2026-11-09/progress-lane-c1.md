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
