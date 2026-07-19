import assert from "node:assert/strict";
import test from "node:test";
import { environmentIds } from "../lib/environments";
import { cloneScene, defaultScene } from "../lib/model";
import {
  buildEnvironmentProxyCatalog,
  environmentProxyMaterialTable,
  environmentProxyPrimitives,
  sparseBrickCoordinatesForAabbs,
  sparseBrickCoordinatesForEnvironment,
  voxelCellRangeForAabb
} from "../lib/voxel-environments";

test("every authored environment gets a stable full-scene proxy catalog", () => {
  const scene = cloneScene(defaultScene);
  const authoredPropCounts = new Map([
    ["conservatory", 19], ["courtyard", 16], ["night-lab", 32], ["concrete-gallery", 9],
    ["bathhouse", 16], ["research-station", 16], ["default", 0], ["garden", 22]
  ]);
  for (const id of environmentIds) {
    const first = buildEnvironmentProxyCatalog(scene, id);
    const second = buildEnvironmentProxyCatalog(scene, id);
    assert.deepEqual(first, second, `${id} must be deterministic`);
    assert.equal(first.environmentIndex, environmentIds.indexOf(id));
    assert.ok(first.shell.bounds_m.max.x > first.shell.bounds_m.min.x, `${id} shell x extent`);
    assert.ok(first.shell.bounds_m.max.z > first.shell.bounds_m.min.z, `${id} shell z extent`);
    if (id !== "default") assert.ok(first.primitives.length > 0, `${id} must expose authored prop proxies`);
    assert.equal(first.primitives.length, authoredPropCounts.get(id), `${id} must retain every shader-authored prop`);
    const all = environmentProxyPrimitives(first);
    const keys = all.map((primitive) => primitive.key);
    assert.equal(new Set(keys).size, keys.length, `${id} keys must be unique`);
    assert.deepEqual(all.map((primitive) => primitive.ownerIndex), all.map((_, index) => index), `${id} owner order must be dense and stable`);
    for (const primitive of all) {
      assert.ok(primitive.aabb_m.min.x <= primitive.center_m.x && primitive.center_m.x <= primitive.aabb_m.max.x, primitive.key);
      assert.ok(primitive.aabb_m.min.y <= primitive.center_m.y && primitive.center_m.y <= primitive.aabb_m.max.y, primitive.key);
      assert.ok(primitive.aabb_m.min.z <= primitive.center_m.z && primitive.center_m.z <= primitive.aabb_m.max.z, primitive.key);
      assert.ok(primitive.material.colorLinear.every(Number.isFinite), primitive.key);
      assert.ok(primitive.material.roughness >= 0 && primitive.material.roughness <= 1, primitive.key);
    }
  }
});

test("night lab includes the complete furniture and fixture vocabulary", () => {
  const catalog = buildEnvironmentProxyCatalog(cloneScene(defaultScene), "night-lab");
  const tags = new Set(catalog.primitives.flatMap((primitive) => primitive.tags));
  for (const required of ["desk", "bench", "stool", "chair", "counter", "instrument", "shelf", "fixture", "light"]) assert.ok(tags.has(required), required);
  assert.equal(catalog.primitives.filter((primitive) => primitive.tags.includes("stool")).length, 3);
  assert.equal(catalog.primitives.filter((primitive) => primitive.key.includes("fixtures/troffer")).length, 4);
  assert.ok(catalog.primitives.some((primitive) => primitive.key.endsWith("counter/monitor-screen") && primitive.material.emission === 1));
  assert.ok(catalog.primitives.some((primitive) => primitive.key.endsWith("desk-lamp/bulb") && primitive.material.emission === 2.8));
  assert.ok(catalog.primitives.some((primitive) => primitive.key.endsWith("desk/top") && primitive.material.colorLinear[0] === .35));
  const materialTable = environmentProxyMaterialTable(catalog);
  assert.deepEqual(materialTable.map((entry) => entry.index), materialTable.map((_, index) => index));
  assert.deepEqual(materialTable.map((entry) => entry.key), environmentProxyPrimitives(catalog).map((primitive) => primitive.key));
});

test("world coordinates track scene dimensions and garden terrain base height", () => {
  const scene = cloneScene(defaultScene);
  scene.container.width_m = 2;
  scene.container.height_m = 1;
  scene.container.depth_m = .5;
  scene.terrain = { baseHeight_m: .17, features: [] };
  const lab = buildEnvironmentProxyCatalog(scene, "night-lab");
  assert.equal(lab.scale_m, 2);
  assert.equal(lab.floorY_m, -1.44);
  assert.equal(lab.primitives.find((primitive) => primitive.key.endsWith("desk/top"))?.center_m.y, -.042);
  const garden = buildEnvironmentProxyCatalog(scene, "garden");
  assert.equal(garden.floorY_m, .17);
  assert.equal(garden.shell.kind, "terrain-heightfield");
  assert.equal(garden.shell.primitives.length, 0, "terrain stays an analytic heightfield instead of becoming a filled box");
  assert.equal(garden.primitives.find((primitive) => primitive.key.endsWith("tree-big/trunk"))?.center_m.y, .17 + .30);
});

test("primitive colors and emission retain the exact shader constants", () => {
  const scene = cloneScene(defaultScene);
  const conservatory = buildEnvironmentProxyCatalog(scene, "conservatory");
  assert.deepEqual(conservatory.primitives.find((primitive) => primitive.key.endsWith("bench/seat"))?.material.colorLinear, [.34, .23, .12]);
  assert.deepEqual(conservatory.primitives.find((primitive) => primitive.key.endsWith("pendant-1/globe"))?.material.colorLinear, [.85, .68, .38]);
  assert.equal(conservatory.primitives.find((primitive) => primitive.key.endsWith("pendant-1/globe"))?.material.emission, .48);
  const station = buildEnvironmentProxyCatalog(scene, "research-station");
  assert.deepEqual(station.primitives.find((primitive) => primitive.key.endsWith("console-left/monitor"))?.material.colorLinear, [.06, .48, .58]);
});

test("AABB helpers conservatively cover negative cells and deduplicate sparse bricks", () => {
  const a = { min: { x: -1, y: -.25, z: -1 }, max: { x: 0, y: .25, z: 0 } };
  assert.deepEqual(voxelCellRangeForAabb(a, .25, { x: 0, y: 0, z: 0 }), {
    minInclusive: { x: -4, y: -1, z: -4 }, maxInclusive: { x: 0, y: 1, z: 0 }
  });
  const bricks = sparseBrickCoordinatesForAabbs([a, a], { cellSize_m: .25, worldOrigin_m: { x: 0, y: 0, z: 0 }, brickSize_cells: 4 });
  assert.equal(bricks.length, 8);
  assert.deepEqual(bricks[0], { x: -1, y: -1, z: -1 });
  assert.deepEqual(bricks.at(-1), { x: 0, y: 0, z: 0 });
});

test("catalog sparse-brick publication includes distant environment props", () => {
  const scene = cloneScene(defaultScene);
  const catalog = buildEnvironmentProxyCatalog(scene, "night-lab");
  const propBricks = sparseBrickCoordinatesForEnvironment(catalog, {
    cellSize_m: .025, worldOrigin_m: { x: -4, y: -4, z: -4 }, brickSize_cells: 8
  });
  assert.ok(propBricks.length > 1);
  const withRoom = sparseBrickCoordinatesForEnvironment(catalog, {
    cellSize_m: .025, worldOrigin_m: { x: -4, y: -4, z: -4 }, brickSize_cells: 8
  }, true);
  assert.ok(withRoom.length > propBricks.length, "room faces extend publication beyond the tank-local sparse domain");
});

test("invalid voxel and shell layouts are rejected", () => {
  const scene = cloneScene(defaultScene);
  assert.throws(() => buildEnvironmentProxyCatalog(scene, "night-lab", { shellThickness_m: 0 }), /thickness/);
  const bounds = [{ min: { x: 0, y: 0, z: 0 }, max: { x: 1, y: 1, z: 1 } }];
  assert.throws(() => sparseBrickCoordinatesForAabbs(bounds, { cellSize_m: 0, worldOrigin_m: { x: 0, y: 0, z: 0 }, brickSize_cells: 8 }), /cell size/);
  assert.throws(() => sparseBrickCoordinatesForAabbs(bounds, { cellSize_m: 1, worldOrigin_m: { x: 0, y: 0, z: 0 }, brickSize_cells: 0 }), /brick size/);
});
