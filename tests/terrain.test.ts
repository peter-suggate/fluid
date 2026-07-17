import assert from "node:assert/strict";
import test from "node:test";
import {
  MAX_TERRAIN_FEATURES,
  sceneHasTerrain,
  terrainCellSolidFraction,
  terrainColumnHeights,
  terrainHeightAt,
  terrainNormalAt,
  validateTerrain,
  type TerrainDescription
} from "../lib/terrain";
import { validateScene, cloneScene, defaultScene } from "../lib/model";
import { advanceRigidBodies, initializeRigidBodies } from "../lib/rigid-body";
import { createTallCellLayout } from "../lib/tall-cell-grid";
import { applyGardenPool, GARDEN_GRASS_M, gardenPoolTerrain } from "../lib/garden-scene";
import { scenePresets } from "../lib/scenes";

const container = { width_m: 3, height_m: 1, depth_m: 2.2 };

test("terrain height is the base level away from features and zero without terrain", () => {
  assert.equal(terrainHeightAt(undefined, 0.3, -0.2), 0);
  const terrain: TerrainDescription = { baseHeight_m: 0.4, features: [] };
  assert.equal(terrainHeightAt(terrain, 1.2, -0.9), 0.4);
});

test("a basin carves down to base minus depth at its centre and vanishes past its radius", () => {
  const terrain: TerrainDescription = {
    baseHeight_m: 0.4,
    features: [{ kind: "basin", center_m: { x: 0, z: 0 }, radius_m: { x: 0.5, z: 0.5 }, amount_m: 0.3 }]
  };
  assert.ok(Math.abs(terrainHeightAt(terrain, 0, 0) - 0.1) < 1e-9);
  assert.equal(terrainHeightAt(terrain, 0.6, 0), 0.4);
  const midway = terrainHeightAt(terrain, 0.36, 0);
  assert.ok(midway > 0.1 && midway < 0.4, `bank should slope smoothly, got ${midway}`);
});

test("overlapping basins union smoothly instead of double-carving", () => {
  const basin = { kind: "basin" as const, radius_m: { x: 0.5, z: 0.5 }, amount_m: 0.3 };
  const overlapped: TerrainDescription = {
    baseHeight_m: 0.4,
    features: [
      { ...basin, center_m: { x: -0.1, z: 0 } },
      { ...basin, center_m: { x: 0.1, z: 0 } }
    ]
  };
  // Both basins are at full carve depth at x = 0; a sum would dig 0.6 deep.
  // The p-norm union deepens a full overlap by at most 2^(1/p).
  const depth = 0.4 - terrainHeightAt(overlapped, 0, 0);
  assert.ok(depth <= 0.3 * 2 ** (1 / 8) + 1e-9, `smooth union overshoot too large: ${depth}`);
  assert.ok(depth >= 0.3 - 1e-6);
});

test("mounds add on top of the base and the ground never goes below the floor", () => {
  const terrain: TerrainDescription = {
    baseHeight_m: 0.2,
    features: [
      { kind: "mound", center_m: { x: 0, z: 0 }, radius_m: { x: 0.5, z: 0.5 }, amount_m: 0.15 },
      { kind: "basin", center_m: { x: 1, z: 0 }, radius_m: { x: 0.4, z: 0.4 }, amount_m: 0.2 }
    ]
  };
  assert.ok(Math.abs(terrainHeightAt(terrain, 0, 0) - 0.35) < 1e-9);
  assert.ok(terrainHeightAt(terrain, 1, 0) >= 0);
});

test("rotation carries an elliptical footprint with it", () => {
  const terrain: TerrainDescription = {
    baseHeight_m: 0.4,
    features: [{ kind: "basin", center_m: { x: 0, z: 0 }, radius_m: { x: 0.8, z: 0.2 }, amount_m: 0.2, rotation_rad: Math.PI / 2 }]
  };
  // The long axis now points along z: a point 0.5 out in x is beyond the
  // footprint, the same distance in z is inside it.
  assert.equal(terrainHeightAt(terrain, 0.5, 0), 0.4);
  assert.ok(terrainHeightAt(terrain, 0, 0.5) < 0.4);
});

test("column heights bake at cell centres in x + nx*z layout", () => {
  const scene = { container, terrain: { baseHeight_m: 0.3, features: [] } };
  const heights = terrainColumnHeights(scene, 4, 3);
  assert.equal(heights.length, 12);
  for (const value of heights) assert.ok(Math.abs(value - 0.3) < 1e-6);
  const flat = terrainColumnHeights({ container, terrain: undefined }, 4, 3);
  for (const value of flat) assert.equal(value, 0);
});

test("cell solid fraction is the vertical cut of the column height", () => {
  assert.equal(terrainCellSolidFraction(0.3, 0, 0.1), 1);
  assert.ok(Math.abs(terrainCellSolidFraction(0.35, 0.3, 0.1) - 0.5) < 1e-9);
  assert.equal(terrainCellSolidFraction(0.3, 0.3, 0.1), 0);
  assert.equal(terrainCellSolidFraction(0.3, 0.5, 0.1), 0);
});

test("terrain validation rejects malformed features", () => {
  assert.equal(validateTerrain({ baseHeight_m: 0.3, features: [] }, container).length, 0);
  assert.ok(validateTerrain({ baseHeight_m: -0.1, features: [] }, container).length > 0);
  assert.ok(validateTerrain({ baseHeight_m: 1.5, features: [] }, container).length > 0);
  assert.ok(validateTerrain({
    baseHeight_m: 0.3,
    features: [{ kind: "basin", center_m: { x: 0, z: 0 }, radius_m: { x: 0.4, z: 0.4 }, amount_m: 0.5 }]
  }, container).length > 0, "basin deeper than the base must be rejected");
  assert.ok(validateTerrain({
    baseHeight_m: 0.3,
    features: Array.from({ length: MAX_TERRAIN_FEATURES + 1 }, () => (
      { kind: "mound" as const, center_m: { x: 0, z: 0 }, radius_m: { x: 0.2, z: 0.2 }, amount_m: 0.1 }
    ))
  }, container).length > 0, "feature count above the shader capacity must be rejected");
});

test("every garden preset produces a valid scene with terrain", () => {
  for (const preset of scenePresets.filter((entry) => entry.group === "Garden")) {
    const scene = preset.create();
    assert.ok(sceneHasTerrain(scene), `${preset.id} should carry terrain`);
    assert.deepEqual(validateScene(scene), [], `${preset.id} should validate`);
  }
});

test("terrain-aware seeding places no initial water below the ground", () => {
  const scene = applyGardenPool(cloneScene(defaultScene));
  scene.fluid.initialCondition = "tank-fill";
  scene.rigidBodies = [];
  const withTerrain = createTallCellLayout(scene, "balanced");
  const flat = cloneScene(scene);
  delete flat.terrain;
  const withoutTerrain = createTallCellLayout(flat, "balanced");
  assert.ok(withTerrain.initialVolumeCellSum > 0, "the pool must hold water");
  assert.ok(
    withTerrain.initialVolumeCellSum < 0.6 * withoutTerrain.initialVolumeCellSum,
    `terrain must exclude the ground volume (${withTerrain.initialVolumeCellSum} vs ${withoutTerrain.initialVolumeCellSum})`
  );
});

test("a rigid body settles onto the lawn instead of falling to the container floor", () => {
  const terrain = gardenPoolTerrain();
  const scene = applyGardenPool(cloneScene(defaultScene));
  scene.rigidBodies = [];
  const dropX = -1.35, dropZ = -0.9; // plain lawn away from the pool
  const bodies = initializeRigidBodies([{
    id: "drop-sphere", name: "Drop sphere", shape: "sphere",
    dimensions_m: { x: 0.05, y: 0.05, z: 0.05 }, density_kg_m3: 500,
    position_m: { x: dropX, y: 0.8, z: dropZ },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0, friction: 0.6
  }]);
  for (let step = 0; step < 600; step += 1) advanceRigidBodies(bodies, scene, 1 / 240);
  const ground = terrainHeightAt(terrain, bodies[0].position_m.x, bodies[0].position_m.z);
  const clearance = bodies[0].position_m.y - ground;
  assert.ok(ground > 0.9 * GARDEN_GRASS_M, "the drop site should be lawn, not pool");
  assert.ok(Math.abs(clearance - 0.05) < 0.02, `sphere should rest on the grass, clearance ${clearance}`);
  const normal = terrainNormalAt(terrain, dropX, dropZ);
  assert.ok(normal.y > 0.9, "lawn normal should be near vertical");
});
