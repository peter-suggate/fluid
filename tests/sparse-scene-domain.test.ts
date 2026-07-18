import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/model";
import { mortonEncode3D, planSparseBrickOctree } from "../lib/sparse-brick-octree";
import { planSparseSceneDomain } from "../lib/sparse-scene-domain";

function sceneWithUnitCells() {
  const scene = cloneScene(defaultScene);
  scene.container.width_m = 8;
  scene.container.height_m = 8;
  scene.container.depth_m = 8;
  return scene;
}

test("negative world coordinates shift one unified lattice while keeping Morton addresses non-negative", () => {
  const scene = sceneWithUnitCells();
  const domain = planSparseSceneDomain(scene, [8, 8, 8], 4, [{
    min: { x: -13, y: -3, z: -10 },
    max: { x: -11, y: -1, z: -8 },
  }]);

  assert.deepEqual(domain.cellSize_m, [1, 1, 1]);
  assert.deepEqual(domain.worldOrigin_m, { x: -13, y: -3, z: -10 });
  assert.deepEqual(domain.solverGridOriginCells, [9, 3, 6]);
  assert.deepEqual(domain.sceneDimensionsCells, [17, 11, 14]);
  assert.ok(domain.coordinates.every(({ x, y, z }) => x >= 0 && y >= 0 && z >= 0));
  assert.doesNotThrow(() => domain.coordinates.map(({ x, y, z }) => mortonEncode3D(x, y, z)));
  assert.equal(planSparseBrickOctree(domain.coordinates, { brickSize: 4 }).leaves.length, domain.coordinates.length);
});

test("props outside the tank add local environment bricks without filling the room between", () => {
  const scene = sceneWithUnitCells();
  const baseline = planSparseSceneDomain(scene, [8, 8, 8], 4, []);
  const nearbyProp = planSparseSceneDomain(scene, [8, 8, 8], 4, [{
    min: { x: 12, y: 0, z: 0 }, max: { x: 14, y: 2, z: 2 },
  }]);
  const distantProp = planSparseSceneDomain(scene, [8, 8, 8], 4, [{
    min: { x: 1_012, y: 0, z: 0 }, max: { x: 1_014, y: 2, z: 2 },
  }]);

  assert.equal(baseline.solverBrickCoordinates.length, 8);
  assert.equal(nearbyProp.environmentBrickCoordinates.length, 1);
  assert.equal(distantProp.environmentBrickCoordinates.length, 1);
  assert.equal(nearbyProp.coordinates.length, baseline.coordinates.length + 1);
  assert.equal(distantProp.coordinates.length, baseline.coordinates.length + 1);
  assert.ok(distantProp.brickDimensions[0] > 250, "bounds describe the full scene without densely allocating them");
});

test("solver and environment covers are deduplicated and remain non-overlapping", () => {
  const scene = sceneWithUnitCells();
  const domain = planSparseSceneDomain(scene, [8, 8, 8], 4, [
    { min: { x: -2, y: 1, z: -2 }, max: { x: 2, y: 5, z: 2 } },
    { min: { x: 3, y: 1, z: -1 }, max: { x: 6, y: 3, z: 1 } },
    { min: { x: 3, y: 1, z: -1 }, max: { x: 6, y: 3, z: 1 } },
  ]);
  const solver = new Set(domain.solverBrickCoordinates.map(({ x, y, z }) => `${x},${y},${z}`));
  const environment = domain.environmentBrickCoordinates.map(({ x, y, z }) => `${x},${y},${z}`);

  assert.ok(environment.every((key) => !solver.has(key)));
  assert.equal(new Set(environment).size, environment.length);
  assert.equal(domain.coordinates.length, solver.size + environment.length);
});

test("conservative padding expands proxy candidates on the solver lattice", () => {
  const scene = sceneWithUnitCells();
  const exact = planSparseSceneDomain(scene, [8, 8, 8], 4, [{
    min: { x: 8.25, y: 1.25, z: 1.25 }, max: { x: 8.75, y: 1.75, z: 1.75 },
  }]);
  const padded = planSparseSceneDomain(scene, [8, 8, 8], 4, [{
    min: { x: 8.25, y: 1.25, z: 1.25 }, max: { x: 8.75, y: 1.75, z: 1.75 },
  }], { conservativePaddingCells: 1 });

  assert.equal(exact.proxyBrickCoordinates[0].length, 1);
  assert.ok(padded.proxyBrickCoordinates[0].length > exact.proxyBrickCoordinates[0].length);
  assert.ok(padded.worldBounds_m.max.x >= exact.worldBounds_m.max.x + 1);
});

test("surface-shell coverage omits fully interior bricks but remains conservative at brick boundaries", () => {
  const scene = sceneWithUnitCells();
  const proxy = { min: { x: 12, y: 0, z: 0 }, max: { x: 36, y: 24, z: 24 } };
  const volume = planSparseSceneDomain(scene, [8, 8, 8], 4, [proxy]);
  const shell = planSparseSceneDomain(scene, [8, 8, 8], 4, [{ ...proxy, coverage: "surface-shell" }], {
    surfaceShellCells: 1,
  });

  assert.equal(volume.proxyBrickCoordinates[0].length, 6 * 6 * 6);
  assert.equal(shell.proxyBrickCoordinates[0].length, 6 ** 3 - 4 ** 3);
  assert.ok(shell.proxyBrickCoordinates[0].length < volume.proxyBrickCoordinates[0].length);
});

test("canonical output is stable under proxy ordering", () => {
  const scene = sceneWithUnitCells();
  const proxies = [
    { min: { x: 20, y: 0, z: 0 }, max: { x: 22, y: 2, z: 2 } },
    { min: { x: -20, y: -2, z: -2 }, max: { x: -18, y: 0, z: 0 } },
  ];
  const forward = planSparseSceneDomain(scene, [8, 8, 8], 4, proxies);
  const reverse = planSparseSceneDomain(scene, [8, 8, 8], 4, [...proxies].reverse());
  assert.deepEqual(forward.coordinates, reverse.coordinates);
  assert.deepEqual(forward.worldOrigin_m, reverse.worldOrigin_m);
  assert.deepEqual(forward.solverGridOriginCells, reverse.solverGridOriginCells);
});
