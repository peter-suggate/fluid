# Lane A: analytic cloth scenes and the Bridson look (2026-11-09)

## Shelf "Cloth · Analytic" (audience validation)

Five scenes, defined in `lib/cloth/scenes/analytic/`. Their gates live in `tests/cloth-analytic.test.ts`, with these environment variables:

- `CLOTH_ANALYTIC_ITERATIONS=K`: the K sweep.
- `CLOTH_ANALYTIC_ONLY=kind,...`: run only the listed kinds.
- `CLOTH_ANALYTIC_CONTACT=0`: run the bare solver with no contact pass.
- `CLOTH_ANALYTIC_CONVERGED=0`: skip the swing's 4K reference.

All runs in the table are GPU runs through `fluid-gpu-run.sh check`. "Contact" means the run went through ClothSystem; "bare" means the solver alone.

| id | expectation | gate | K=128 contact | K=128 bare | K=32 bare | K=8 |
|---|---|---|---|---|---|---|
| cloth-analytic-free-fall | BE + BDF2 recurrence, flat, unstrained (Euler start offset 3/4 g h² = 2.04 mm) | centroid < 0.1 mm, v < 1 mm/s, flatness < 0.1 mm, strain < 1e-4, 24 steps | FAIL: 0.22 mm, flatness 0.93 mm | PASS: 0.0002 mm, 0.004 mm | PASS | FAIL: 6.43 mm, 17.9 mm (bare) |
| cloth-analytic-hanging-curtain | per-band uniaxial StVK kλ(λ²−1)/2 = ρg(L−s) | 1 s mean band strain within 1 %, hem drop within 1 % | PASS: 0.95 %, 65.06/65.55 mm, KE 2e-3 J | instantaneous 7.2 %, still swaying | — | PASS: 0.29 % (contact) |
| cloth-analytic-swinging-sheet | flat outer part falls at g exactly; crosses under the pin at ≈0.49 s (Float64 41×41 oracle) | free edge < 1 mm over 20 steps; crossing within one step of 4K | FAIL: 6.0 mm at step 20, never crosses (4K = 512 also never crosses) | PASS: 0.23 mm, crosses 0.4957 s | PASS: 0.85 mm, 0.5095 s | FAIL: 11.4 mm, no crossing (bare) |
| cloth-analytic-peirce-cantilever | heavy elastica: chord 22.26° / 41.60° / 56.99° at l/c 1.5 / 2 / 2.5 | chord within 1° after 6 s | FAIL: 0.22 / 0.14 / 0.12° (bare) | — | — | FAIL: 0.01° (bare) |
| cloth-analytic-friction-incline | stick at 20° (tan < μ = 0.5); slide at 35°, a = g(sin − μ cos) = 1.61 m/s² | creep < 2 mm; fitted a within 5 % | not run: needs the C2 hook (below) | | | |

The curtain gate measures statics, so it does not separate K. A converged solve is nearly undamped and is still swaying at 10 s. The gate therefore averages over the last second. An under-converged K dissipates energy and comes to rest sooner.

## Findings

1. **Contact perturbs free flight (C2).** The bare solver passes free fall and swing at K = 32 and K = 128. ClothSystem, which adds OGC contact, fails both at K = 128:
   - Free fall: the flat sheet deforms by 0.93 mm and its centroid lags by 0.22 mm, with no solid nearby.
   - Swing: the free edge is 6 mm off at step 20 and never passes under the pin, even at K = 512.

   Self-contact is acting on a flat sheet that is not in contact with anything.
2. **Peirce: the strips do not droop (C1).**
   - **Calibration.** A uniform bend κ of a strip gives a discrete hinge energy of 2.93 × (B/2)κ²A. The check is `fluid-scratch/lane-a/bend-energy.ts`: the oracle's own energy at κ = 0.5, 2 and 8 gives the same ratio each time. `k_h = 3B l²/(A0+A1)` together with the Bergou weights, which already measure θ, makes B three times too stiff. That makes c 3^(1/3) = 1.44 times too long. This applies to Bridson's 1.5e-4 as well.
   - **Convergence, which is the larger effect.** Factor 3 alone would still predict about a 20° droop. What we see:
     - The Float64 oracle creeps at 0.1 mm per 30 steps whether K is 32 or 512.
     - A strip released from a 9° arc creeps back toward flat.
     - The step residual stalls at 1.6–2 |mg| and does not fall as K grows.
     - B × 0.1 still gives 0.13°; B × 0.01 gives 86°.

     The likely cause is that Gauss–Seidel on the bending (biharmonic) operator stalls, and the tent coarse basis contains no smooth bending mode: a tent displacement kinks at the coarse nodes. This is the same basis limitation C1 reports for folds.
3. **The "plank" frustum was occlusion, not geometry.** The bake is correct. The old camera looked down from 0.5 rad, and from there the 1.9 m sheet at 0.56 m hid everything of the frustum except its lowest bricks. The old flat warm room lighting removed the shape cues as well. From the 2002 framing the frustum renders as a frustum: see `fluid-scratch/lane-a/shots/final-bridson-2002-rotating-sphere.png`.
4. **The analytic free-fall card halts when the sheet lands (C2).** At frame 32 contact reports `solidBounds` once the sheet reaches the container floor at y = 0. The floor is the shell, which sits outside the lattice. The Bridson scenes avoid this by standing on a one-brick slab. The analytic scenes cannot have a slab until C2 dispatches their solids, because SHELL_ONLY throws on any authored box.
5. **The Bridson curtain drape halts with `heldConflict` at frame 124 (C2).** This was seen while trying to capture the drape screenshot.

## Bridson look (appearance.ts)

- **Cloth:** #FF00FF.
- **Ground:** a new voxel material, `marble` (id 23, pale grey, roughness 0.45).
- **Frustum and legs:** a new voxel material, `obsidian` (id 24, near black, roughness 0.22).
- **Sky and lighting:** black sky from backdrop radiance 0.002 with diffuse ×120, which keeps the grey fill. The key light is directional at intensity 3.2, behind and to the right of each camera, and its shadows are visible.
- **Enclosure and tone:** the room shell has `faces: []`, and the tone curve is Reinhard, because AgX turned the magenta pink.
- **Cameras:** read off cloth0 f040, cloth1 f018 and cloth2 f010 for horizon height, subject scale and side, with tanHalfFov 0.5.
- **Shadow test:** `tests/raster-local-shadows.test.ts` used the Bridson scenes' old ceiling softbox as fixtures for its area-light map tests. It now places the Bridson containers in the default room (`createEmptyScene` scenery and lighting), so its inputs are unchanged, and it checks that the shipped look's single light is directional, with nothing traced per pixel.
- **Commits:** 030d3b13 (analytic shelf), 56b1fa9f (look, status, gate updates).

Not done:

- **Marble veins.** Per-pixel procedural colour was removed from the renderer. Veins would be one `paint` voxel-block patch on the slab's top layer, and the solid world already supports that. C2's scene-solids rejects every paint patch, so this needs the hook below.
- **Ground to the horizon.** A scenery ring around the slab does not fit at a 1 cm lattice. A 6 m ring needs 4.2M surface-mesh vertices against a 2.1M arena, and a 60 m ring runs out of the 4 GB JS heap. The ground ends at the slab edge.
- **Mesh arena overflow.** The probe reports "Water surface: the mesh needs 4194300 vertices, over its capacity of 2097152" on the tablecloth and curtain. It happens with the old look too, so it predates these changes.

## Hooks needed

- **C2, `lib/cloth/contact/scene-solids.ts`, analytic solids.** In `clothSolidSourceForScene`, dispatch `scene.sceneId.startsWith(CLOTH_ANALYTIC_SCENE_PREFIX) ? clothAnalyticSceneSolids(scene)`, imported from `lib/cloth/scenes/analytic/contact-solids.ts`. That file is ready. This unblocks the incline gate, the Peirce ledge in ClothSystem, and a landing slab for the analytic scenes.
- **C2, same file, paint patches.** Treat a `paint` patch of any kind as contact-neutral and skip it, since it recolours solid cells and changes no geometry. This is what lets marble veins be authored.
- **C1, bending.** Fix the hinge calibration factor of 2.93, and give the coarse space a smooth bending mode, or explain why bending-dominated statics stall.
