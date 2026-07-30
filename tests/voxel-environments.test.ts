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
  // A tripwire, not a target: these move whenever a scenery module is
  // re-art-directed, and the number changing is the prompt to look at the
  // render. Each environment's geometry lives in lib/voxel-scenery/<id>.ts.
  const authoredPropCounts = new Map([
    ["conservatory", 89], ["courtyard", 93], ["night-lab", 105], ["concrete-gallery", 68],
    ["bathhouse", 90], ["research-station", 104], ["default", 17], ["garden", 122]
  ]);
  for (const id of environmentIds) {
    const first = buildEnvironmentProxyCatalog(scene, id);
    const second = buildEnvironmentProxyCatalog(scene, id);
    assert.deepEqual(first, second, `${id} must be deterministic`);
    assert.equal(first.environmentIndex, environmentIds.indexOf(id));
    assert.ok(first.shell.bounds_m.max.x > first.shell.bounds_m.min.x, `${id} shell x extent`);
    assert.ok(first.shell.bounds_m.max.z > first.shell.bounds_m.min.z, `${id} shell z extent`);
    assert.ok(first.primitives.length > 0, `${id} must expose authored prop proxies`);
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
  assert.ok(catalog.primitives.some((primitive) => primitive.key.endsWith("counter/monitor-screen") && primitive.material.emission === 1.35));
  assert.ok(catalog.primitives.some((primitive) => primitive.key.endsWith("desk-lamp/bulb") && primitive.material.emission === 3.4));
  assert.ok(catalog.primitives.some((primitive) => primitive.key.endsWith("desk/top") && primitive.material.colorLinear[0] === .60));
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
  const heroParts = garden.primitives.filter(({ key }) => key.includes("/tree-hero/"));
  assert.ok(heroParts.length > 20, "the specimen tree is grown from a seed, not authored bead by bead");
  const trunkBase = garden.primitives.find(({ key }) => key.endsWith("tree-hero/trunk-0"));
  assert.ok(trunkBase && Math.abs(trunkBase.aabb_m.min.y - .17) < .02,
    "the specimen tree stands on the terrain datum under its own root, not on a nominal lawn plane");
  assert.ok(heroParts.every(({ sway }) => sway && sway.pivot_m.y === .17),
    "every part of one tree swings about that tree's own root");
});

/**
 * Surface colours used to be pinned here to keep the proxy catalog in step with
 * the raster environment shader. Props are voxel-native now — the SVO path
 * reads these materials directly — so that parity contract is gone, and
 * neutrality is enforced generally in tests/scenery-art-direction.test.ts.
 *
 * What still deserves pinning is the emitters. They are the only tinted things
 * left in any scene, so each one is a deliberate lighting decision rather than
 * a surface colour, and silent drift in a tint or an intensity would restage
 * the room without anything else noticing.
 */
test("emissive sources keep the tint and intensity their scene was lit around", () => {
  const scene = cloneScene(defaultScene);
  const emitter = (environmentId: Parameters<typeof buildEnvironmentProxyCatalog>[1], suffix: string) =>
    buildEnvironmentProxyCatalog(scene, environmentId).primitives.find((primitive) => primitive.key.endsWith(suffix))?.material;

  // Warm globes over the conservatory staging bench.
  assert.deepEqual(emitter("conservatory", "pendant-1/globe")?.colorLinear, [.85, .68, .38]);
  assert.equal(emitter("conservatory", "pendant-1/globe")?.emission, .48);
  // Cold instrument cyan, the one cool note in the station hull.
  assert.deepEqual(emitter("research-station", "console-left/monitor")?.colorLinear, [.06, .48, .58]);
  assert.equal(emitter("research-station", "console-left/monitor")?.emission, .30);
  // The night lab's single warm source, read against its cool troffers.
  const bulb = emitter("night-lab", "desk-lamp/bulb");
  assert.equal(bulb?.emission, 3.4);
  assert.ok(bulb!.colorLinear[0] > bulb!.colorLinear[2], "the task lamp must stay warmer than it is blue");
  const troffer = emitter("night-lab", "fixtures/troffer-left-1");
  assert.ok(troffer!.emission > 0 && troffer!.colorLinear[2] >= troffer!.colorLinear[0] * .9,
    "ceiling troffers stay the cool counterweight to the task lamp");

  // Anything an environment nominates as a light really does emit; a fixture
  // tagged for the light table but left at zero would publish a black lamp.
  for (const id of environmentIds) {
    for (const primitive of buildEnvironmentProxyCatalog(scene, id).primitives) {
      if (!primitive.tags.includes("light")) continue;
      assert.ok(primitive.material.emission > 0, `${primitive.key} is tagged as a light but emits nothing`);
    }
  }
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
