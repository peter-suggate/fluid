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
