import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { parseScene, serializeScene, validateScene } from "../lib/core/model";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { getSceneDefinition, getScenePreset } from "../lib/core/scenes";
import { sceneCardForDefinition, sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { parseQueryState } from "../lib/core/url-state";
import { uniformPagePoolResidency, UNIFORM_PAGE_POOL_SEED_MARGIN } from "../lib/methods/uniform/uniform-page-pool";
import { advanceUniform, createUniformSolver, readUniformLiveAddress, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedTexture } from "./helpers/uniform-mixed-native-fields";

test("reservoir seep opens with localized dynamic detail on an 8-million-cell lattice", () => {
  const definition = getSceneDefinition(RESERVOIR_SEEP_ID);
  const scene = getScenePreset(RESERVOIR_SEEP_ID).create();
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(sceneLatticeDimensions(scene), [512, 256, 64]);
  assert.deepEqual(parseScene(serializeScene(scene)), scene);
  assert.deepEqual(sceneCardForDefinition(definition).open().methodProfile, definition.methodProfile);
  const query = parseQueryState(`?scene=${RESERVOIR_SEEP_ID}`);
  assert.equal(query.overrides[query.methodId]?.detailPolicy, "dynamic");
  assert.equal(query.overrides[query.methodId]?.detailBudgetPercent, 5);
  assert.equal(scene.fluid.inflow, undefined);
  assert.equal(scene.rigidBodies.length, 0);
  const region = scene.fluid.refinementRegions![0]!;
  const fineVolume = (region.max_m.x - region.min_m.x) * (region.max_m.y - region.min_m.y)
    * (region.max_m.z - region.min_m.z);
  assert.ok(fineVolume / (2.56 * 1.28 * 0.32) < 0.001);
});

test("reservoir page budget covers initial residency with at least 25% growth room", () => {
  const scene = createReservoirSeepScene(), dimensions = sceneLatticeDimensions(scene);
  // Exercise the production residency planner without an 8M-element fixture.
  const [nx, ny, nz] = dimensions;
  const volume = new Proxy({ length: nx * ny * nz }, {
    get(target, key) {
      if (key === "length") return target.length;
      const i = Number(key);
      return i % nx < nx / 2 - 4 && Math.floor(i / nx) % ny < 3 * ny / 4 ? 1 : 0;
    },
  });
  const residency = uniformPagePoolResidency(dimensions, undefined, [], true, () => volume as unknown as ArrayLike<number>);
  assert.ok(UNIFORM_PAGE_POOL_SEED_MARGIN <= 64, "the scene's two-page collar must cover the solver seed margin");
  const allPages = nx / 32 * (ny / 32) * (nz / 32), live = allPages - residency.air.length;
  const usable = scene.fluid.pagePool!.blocks! - 2;
  assert.ok(usable >= Math.ceil(1.25 * live));
  assert.ok(usable < allPages, "do not reserve the entire empty catch chamber");
});

test("hydrostatic head sends water through the throat into the dry chamber", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("reservoir seep", async device => {
    const definition = getSceneDefinition(RESERVOIR_SEEP_ID);
    // Exercise the same h-throat geometry at a smaller lattice for the GPU
    // regression; the full stress resolution is checked separately above.
    const { scene } = sceneDocumentAtLattice(definition, { cellSize_m: 0.02 });
    const solver = await createUniformSolver(device, scene, definition.methodProfile!.overrides);
    try {
      console.info("Reservoir seep GPU ready");
      for (let step = 1; step <= 30; step++) {
        await advanceUniform(solver, step / 60);
        if (step % 10 === 0) console.info(`Reservoir seep step ${step}`);
      }
      const address = await readUniformLiveAddress(device, solver);
      const volume = await readMixedTexture(device, solver.volumeTexture, address);
      const { nx } = solver.info;
      let escaped = 0;
      for (let i = 0; i < volume.length; i++) {
        assert.ok(Number.isFinite(volume[i]));
        if (i % nx >= nx / 2 + 4) escaped += volume[i]!;
      }
      assert.ok(escaped > 0.01, `expected water beyond the wall, got ${escaped} cell volumes`);
      assert.ok(solver.pagePoolInfo!.freePages > 0, "the flow retains growth capacity");
    } finally { solver.destroy(); }
  }, ["subgroups"]);
});
