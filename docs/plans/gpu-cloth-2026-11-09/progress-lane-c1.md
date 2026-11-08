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
