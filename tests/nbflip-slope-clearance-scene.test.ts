import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { parseScene, serializeScene, validateScene } from "../lib/core/model";
import { NBFLIP_METHOD_PROFILE } from "../lib/core/nbflip-paper-scenes";
import { NBFLIP_RAMP_CELL_SIZE_M, NBFLIP_RAMP_GRID, NBFLIP_RAMP_LENGTH } from "../lib/core/nbflip-ramp-scene";
import {
  NBFLIP_SLOPE_CLEARANCE_INITIAL_CELLS, NBFLIP_SLOPE_CLEARANCE_SCENE_ID,
  NBFLIP_SLOPE_CLEARANCE_PIT_START, createNbflipSlopeClearance, nbflipSlopeClearanceTransitSeconds,
} from "../lib/core/nbflip-slope-clearance-scene";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { terrainHeightAt } from "../lib/core/terrain";
import { initialLiquidCellVolumes } from "../lib/core/initial-liquid-lattice";

test("slope clearance opens with the production NB-FLIP profile and a finite release at rest", () => {
  const definition = getSceneDefinition(NBFLIP_SLOPE_CLEARANCE_SCENE_ID), scene = sceneDocument(definition);
  assert.equal(definition.id, NBFLIP_SLOPE_CLEARANCE_SCENE_ID);
  assert.deepEqual(definition.methodProfile, NBFLIP_METHOD_PROFILE);
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(parseScene(serializeScene(scene)).fluid, scene.fluid);
  assert.deepEqual(parseScene(serializeScene(scene)).terrain, scene.terrain);
  assert.equal(scene.fluid.inflow, undefined);
  assert.equal(scene.container.fillFraction, 0);
  assert.deepEqual(scene.fluid.initialVelocity_m_s, { x: 0, y: 0, z: 0 });
  assert.equal(scene.fluid.surfaceTension_N_m, 0);
  assert.equal(scene.container.fluidWallMode, "free-slip");
  assert.equal(scene.fluid.initialLiquidVolumes?.length, 1);
});

test("the release starts less than a cell above the continuous slope and retains its full seed volume", () => {
  const scene = createNbflipSlopeClearance(), h = NBFLIP_RAMP_CELL_SIZE_M, [nx, , nz] = NBFLIP_RAMP_GRID;
  const water = scene.fluid.initialLiquidVolumes![0]!;
  assert.equal(water.shape, "box");
  if (water.shape !== "box") return;
  const min = [water.min_m.x / h + nx / 2, water.min_m.y / h, water.min_m.z / h + nz / 2].map(Math.round);
  const max = [water.max_m.x / h + nx / 2, water.max_m.y / h, water.max_m.z / h + nz / 2].map(Math.round);
  for (let z = min[2]!; z < max[2]!; z++) for (let x = min[0]!; x < max[0]!; x++) {
    assert.ok(x > 0 && x < 16 && z > 0 && z < nz - 1);
    const bed = terrainHeightAt(scene.terrain, (x + 0.5 - nx / 2) * h, (z + 0.5 - nz / 2) * h);
    const gap = water.min_m.y - bed;
    assert.ok(gap > 0 && gap < h);
  }
  const seeded = initialLiquidCellVolumes(scene, NBFLIP_RAMP_GRID, true);
  assert.equal(seeded.initial, NBFLIP_SLOPE_CLEARANCE_INITIAL_CELLS);
  // Even all the release in the pit occupies less than one cell of depth;
  // its resting pool cannot back up onto the lowest ramp tread.
  const poolDepth = seeded.initial / ((nx - NBFLIP_SLOPE_CLEARANCE_PIT_START) * nz);
  assert.ok(2 + poolDepth < 11);
});

test("the authored bed is an unbroken plane across former tread edges and joins the basin continuously", () => {
  const scene = createNbflipSlopeClearance(), h = NBFLIP_RAMP_CELL_SIZE_M, [nx, , nz] = NBFLIP_RAMP_GRID;
  assert.deepEqual(scene.solidVoxels, []);
  const height = (x: number, z: number) => terrainHeightAt(scene.terrain, (x - nx / 2) * h, (z - nz / 2) * h) / h;
  for (let x = 0; x <= NBFLIP_RAMP_LENGTH; x += 0.125) {
    for (const z of [0, 7.25, nz / 2, nz]) assert.ok(Math.abs(height(x, z) - (17 - x / 16)) < 1e-10);
  }
  for (const x of [NBFLIP_RAMP_LENGTH, NBFLIP_SLOPE_CLEARANCE_PIT_START]) {
    assert.ok(Math.abs(height(x - 1e-6, 16) - height(x + 1e-6, 16)) < 5e-6);
  }
  for (let x = NBFLIP_SLOPE_CLEARANCE_PIT_START; x <= nx; x++) assert.ok(Math.abs(height(x, 16) - 2) < 1e-10);
});

test("the six-second clearance target spans more than two ideal gravity transits at either step rate", () => {
  const transit = nbflipSlopeClearanceTransitSeconds();
  assert.ok(transit > 2.4 && transit < 2.5);
  for (const hz of [30, 60]) {
    const scene = createNbflipSlopeClearance(hz);
    assert.equal(scene.numerics.fixedDt_s, 1 / hz);
    assert.equal(scene.duration_s, 6);
    assert.ok(scene.duration_s > 2 * transit);
  }
});
