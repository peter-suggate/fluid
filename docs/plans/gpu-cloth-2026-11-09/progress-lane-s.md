# Lane S progress: Bridson 2002 scenes

**Owned files:**
- `lib/cloth/scenes/bridson-2002/`
- `tests/cloth-bridson-2002*.test.ts`
- `docs/plans/gpu-cloth-2026-11-09/bridson2002-scenes.md`
- this file

## 2026-11-09 MILESTONE: sources and provenance

**Sources found:**
- the SIGGRAPH 2002 paper: UBC PDF, plus Stanford's lossless PDF for the figures;
- the Stanford project page, with stills `test0_frame114`, `test1_frame053` and `test2_frame056`;
- the three 2002 result videos (`cloth0/1/2.avi`, from the Wayback copy of physbam.stanford.edu);
- Bridson's 640×480 copies of the same videos;
- Bridson's 2003 PhD thesis;
- PhysBAM `STANDARD_TESTS.h`, tests 5, 12 and 13, a 2006–2008 descendant of the 2002 code.

All of it is in `~/code/me/fluid-scratch/bridson2002/`, outside git.

**Key findings:**
- **Timing.** The videos hold 24 fps simulation frames, played back at 15 fps. The PhysBAM test 5 ball start at 2 s lands on video f48–56, and its return at 4.75 s on f114, which is Stanford's `test0_frame114`.
- **Fig 1, caption versus §9.** One sphere plays both roles. It lifts the curtain "up and away" and drops behind it, then comes back along the ground as the caption's "another ball". The video only ever shows one ball.
- **Equal masses.** PhysBAM `Create_Cloth_Panel` confirms the paper: equal masses, 1 kg/m² in total. It also uses a herring-bone mesh.
- **Fig 3.** The table differs from PhysBAM test 12. We reconstructed it from the 2002 video's camera and scaled it by test 12's 0.15 m sphere. The 2002 run started from an already draped cloth.
- **Fig 4.** Reference only; nothing stands in for it.

Details are in `bridson2002-scenes.md`.

## 2026-11-09 MILESTONE: scenes authored as data, with tests

**Code:** `lib/cloth/scenes/bridson-2002/`
- `parameters.ts`: every number, each with its source and ref.
- `mesh.ts`: the herring-bone sheet and the equal node mass.
- `solids.ts`: voxel solid shapes, poses and lattice checks, plus a bake into `bakeProceduralSolid`.
- `scenes.ts`:
  - `bridson2002Curtain`: 254 × 150 nodes;
  - `bridson2002RotatingSphere`: 150 × 150;
  - `bridson2002Tablecloth`: 150 × 150;
  - `BRIDSON_2002_FIGURE_4`, reference only.

**Tests:** `tests/cloth-bridson-2002.test.ts`, 9 tests, about 5 s. They cover:
- provenance on every parameter;
- valid cloth documents;
- node counts and herring-bone layout (valence 6);
- equal mass;
- dimensions against provenance;
- voxel feature resolution: 1 cm lattice, stored surface within half a cell, a 3 cm lattice refused, unknown kinds throw;
- a static bake;
- motion timings against the PhysBAM scripts and the video measurements;
- key moments on video frames.

## 2026-11-09 COORDINATION → C1: equal per-vertex masses

**Problem.** The paper says "giving all particles the same mass instead of a mass proportional to the area". `prepareClothTopology` lumps mass ρ·A/3 per face:
- With the herring-bone mesh the interior masses are equal.
- The hems come out lighter, though the paper makes them heavier.
- `clothGridData` gives a 2:1 checkerboard even in the interior, because its diagonals alternate per cell.

**Request.** Either of these:
- an optional `ClothTopologyData.vertexMass_kg?: readonly number[]`, one per vertex, which replaces area lumping when present. It would be validated as positive and finite, and split by incident area when a cut duplicates a vertex.
- `ClothDocument.massDistribution?: "area" | "equal"`.

The scenes carry `nodeMass_kg` ready for it.

## 2026-11-09 COORDINATION → C1: cloth documents running in scenes

`SceneDescription.cloth` validates, and `scene-runtime` flags `cloth-dynamics`, but nothing steps a scene's documents with `ClothGPUSolver` yet.

**Needed:**
1. The scene clock steps each document once per 1/60 s frame.
2. A harness entry point, so Lane S can capture frames at given scene times.

The curtain's two pins need nothing new: a group with no scripts is held at its rest anchor (`clothAttachmentTarget`).

When that lands, Lane S registers the three scenes in `lib/core/scenes.ts`, using exact-string Edit.

## 2026-11-09 COORDINATION → C2: voxel solids for cloth contact (M5, M6)

The scenes need these APIs:

1. **M5, static voxel solids.**
   - `buildClothSolidField` takes a `ClothSolidSource` (`sdf`, `lo`, `hi`, `cell`, `band`). `bridsonSolidDistance` supplies `sdf`.
   - In scenes, static solids should come from the scene's own `solidVoxels`/`solidBoundary`, so edits are seen. Lane S bakes the legs and the frustum through `bakeProceduralSolid`.
   - The ground should come from the scene floor. A baked 3.8 × 2.8 m slab takes 45 s on the CPU.
   - The cell is 1 cm. The smallest feature is the 22 mm leg bevel.
2. **M6, one-way moving voxel solids.**
   - Each solid is a body-local field plus a `RigidPoseTrack`, sampled at absolute scene time with `sampleRigidPoseTrack`.
   - Contact velocity comes from the pose over the step.
   - The scenes need:
     - a 0.25 m sphere moving piecewise-linearly (curtain);
     - a 0.25 m sphere spinning at −π/4 rad/s about y (sphere, μ = 1, so friction must use the body's surface velocity);
     - a 0.15 m sphere descending, then translating (table).
3. **Friction per solid.**
   - The scenes need μ per solid: ground 0.3, 0.1 or 0.3; ball 0; sphere 1; frustum 0.1; legs 0; table sphere 0.5.
   - They also need a separate self-contact μ, 0.3.
   - `OgcContactConfig` has one `mu`.

## 2026-11-09 COORDINATION → coordinator: how a scene declares a moving voxel solid

`RigidBodyDescription` shapes are analytic (`sphere`, `box`, …), and the lane rule forbids analytic colliders. The scenes need a way to declare a moving voxel solid in the scene model: a voxel shape or baked field, plus a `poseTrack`. Until that exists, the moving solids live only in the Lane S scene data.
