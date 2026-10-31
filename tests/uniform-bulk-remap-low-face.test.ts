import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { UNIFORM_KIND, uniformPageKind } from "../lib/methods/uniform/uniform-lattice-address";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

/* The remap of a tile that changes width rebuilds its low face from the
 * cell below. Over a page that stores no cell (umFaceLow: the domain wall, a
 * SOLID or a BULK16 page) the face lives in the owner's low-face rows, and
 * the sample is the owner's cell above it (remapFootprint, ueLowCell's rule).
 * Measured: a stone (r 0.15 m, 2500 kg/m3) entering a census start's deep
 * water at 3 m/s promoted the BULK16 pages it reached at frames 2-3; at
 * frame 5 its splash refined tiles on their low plane over pages still in
 * BULK16, the remap sampled the cell below (old owner width 0) and the
 * low-face rows of the four pages above went nonfinite; the pressure
 * followed at frame 6. A stone held through the hold's demotes and released
 * failed the same way on 2e4da9b3, so the census start only reached it sooner. */
const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };

test("a tile changing width over a BULK16 page remaps its low face from its own cell", gpu, async t => {
  const pages = [6, 10, 4], water = 280, R = 0.15;
  const base = getSceneDefinition(RESERVOIR_SEEP_ID);
  const { scene } = sceneDocumentAtLattice({ ...base, buildAt: (l: { cellSize_m: number }) => createReservoirSeepScene(l.cellSize_m, { widthScale: pages[0]! / 16, heightScale: pages[1]! / 8, depthScale: pages[2]! / 2 }) }, { cellSize_m: 0.005 });
  const s = scene as any, w = s.container.width_m, d = s.container.depth_m, top = water * 0.005;
  s.solidVoxels = [];
  s.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -w / 2, y: 0, z: -d / 2 }, max_m: { x: w / 2, y: top, z: d / 2 } }];
  s.fluid.initialVelocity_m_s = { x: 0, y: 0, z: 0 };
  delete s.fluid.pagePool; s.fluid.refinementRegions = [];
  s.fluid.dynamicViscosity_Pa_s = 0; s.fluid.surfaceTension_N_m = 0;
  s.fluid.tiers = { start: "census" };
  // The backyard ball as a free body (two-way coupled), resting on the surface and moving down at 3 m/s.
  const ball = createBackyardPoolPartyScene().rigidBodies[0]!;
  s.rigidBodies = [{ ...ball, id: "remap/stone", dimensions_m: { x: R, y: R, z: R }, density_kg_m3: 2500, position_m: { x: 0, y: top + R, z: 0 }, linearVelocity_m_s: { x: 0, y: -3, z: 0 } }];
  const values = { ...base.methodProfile!.overrides, bulkTier: "16h", volumeControlSeconds: 0 } as never;
  await withUniformDevice("BULK16 low-face remap", async device => {
    const solver: any = await createUniformSolver(device, scene, values), a = solver.address;
    try {
      const [px, py, pz] = a.pages as number[];
      let D = 0;
      for (let z = 0; z < pz!; z++) for (let y = 0; y < py!; y++) for (let x = 0; x < px!; x++) if (uniformPageKind(a, x, y, z) === UNIFORM_KIND.bulk) D++;
      assert.ok(D > 0, "the census start holds deep pages in BULK16");
      const bodies = initializeRigidBodies(scene.rigidBodies ?? []), rows: GPUBuffer = solver.negativeBoundaryVelocityBuffer;
      // Frame 5's remap is the one that failed (frame 6's solve read it).
      // Frame 6 itself ends at the departure closure's fatal: the stone
      // enters D pages no promote has taken back, its own item.
      for (let frame = 1; frame <= 5; frame++) {
        assert.ok(solver.advanceTo(frame * scene.numerics.fixedDt_s!, bodies)); await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined, `frame ${frame}`);
        const v = await readMixedBuffer(device, rows);
        let bad = 0; for (let i = 0; i < v.length; i++) if (!Number.isFinite(v[i]!)) bad++;
        assert.equal(bad, 0, `frame ${frame}: every low-face row is finite`);
      }
      t.diagnostic(`D ${D}; blocks ${a.blocks}`);
    } finally { solver.destroy(); }
  });
});
