import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { createSolidWorld, sampleSolidWorld, solidWorldForScene, withSolidWorldPatches } from "../lib/core/solid-world";
import { compileRetainedSceneFineMeans, evaluateRetainedSceneDensity, retainedSceneDensity } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";
import { compileRetainedOpenSceneEditMoments, compileRetainedOpenSceneFineMeans } from "../lib/methods/adaptive-mass/sparse-cm12-retained-open-density";

const dimensions = [8, 8, 8] as const, h = .125;
const field = retainedSceneDensity({ generation: 3, transitionWidth: h,
  domain: { lower: [-.5, 0, -.5], upper: [.5, 1, .5] },
  primitives: [{ kind: "quadratic-height", center: [0, h / 2, 0], curvature: [0, 0, 0] }] });
const seedMeans = compileRetainedSceneFineMeans(field, dimensions, h);
const options = { seedMeans };
const close = (a: number, b: number, tolerance = 3e-8) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
function terrainWorld(fraction = 128) {
  const scene = cloneScene(defaultScene);
  scene.container.width_m = 1; scene.container.height_m = 1; scene.container.depth_m = 1;
  scene.voxelDomain.finestCellSize_m = h; scene.solidVoxels = [];
  scene.terrain = { baseHeight_m: fraction / 255 * h, features: [] };
  return solidWorldForScene(scene);
}

test("whole edited voxels mask exact physical moments without modifying the retained seed", () => {
  const world = createSolidWorld([{ operation: "fill", minimum: [2, 0, 1], maximumExclusive: [5, 3, 4] }]);
  const before = seedMeans.slice(), result = compileRetainedOpenSceneFineMeans(field, dimensions, h, world, options);
  assert.deepEqual(seedMeans, before); assert.equal(result.receipt.closedCells, 27); assert.equal(result.receipt.clippedIntegrals, 0);
  for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
    const i = x + 8 * (y + 8 * z), open = 1 - sampleSolidWorld(world, [x, y, z]).solidFraction;
    assert.equal(result.openFractions[i], open); assert.equal(result.effectiveMeans[i], open ? seedMeans[i] : 0);
  }
});

test("partial terrain integrates q over its open top slab instead of multiplying two means", () => {
  const world = terrainWorld(), result = compileRetainedOpenSceneFineMeans(field, dimensions, h, world, options);
  const open = 127 / 255, effective = .5 * open ** 2;
  assert.equal(result.receipt.fractionalCells, 64); assert.equal(result.receipt.clippedIntegrals, 64);
  for (let z = 0; z < 8; z++) for (let x = 0; x < 8; x++) {
    const i = x + 64 * z;
    assert.equal(result.solidFractions[i], 128); close(result.openFractions[i], open);
    close(result.effectiveMeans[i], effective);
    assert.ok(Math.abs(result.effectiveMeans[i] - seedMeans[i] * open) > .1,
      "scalar open-fraction multiplication would accept the wrong liquid amount");
  }
  assert.ok(result.receipt.maximumEstimatedOpenMeanError <= 2e-7);
});

test("ordered compact regions override both terrain pages and earlier regions", () => {
  const world = { ...terrainWorld(), regions: [
    { operation: "fill" as const, minimum: [0, 0, 0] as const, maximumExclusive: [3, 2, 3] as const },
    { operation: "clear" as const, minimum: [1, 0, 1] as const, maximumExclusive: [2, 1, 2] as const },
  ] };
  const result = compileRetainedOpenSceneFineMeans(field, dimensions, h, world, options);
  for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
    const i = x + 8 * (y + 8 * z), sample = sampleSolidWorld(world, [x, y, z]);
    assert.equal(result.solidFractions[i], Math.round(255 * sample.solidFraction));
  }
  const cleared = 1 + 8 * (0 + 8 * 1); assert.equal(result.effectiveMeans[cleared], .5);
  assert.equal(result.openFractions[cleared], 1); assert.equal(result.effectiveMeans[0], 0);
});

test("a partial voxel without declared terrain geometry is rejected after region overrides", () => {
  const world = createSolidWorld([{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [1, 1, 1], materialId: 7 }]);
  world.pages[0].solidFraction[0] = 128;
  assert.throws(() => compileRetainedOpenSceneFineMeans(field, dimensions, h, world, options), /no declared subvoxel geometry/);
  const overridden = { ...world, regions: [{ operation: "clear" as const, minimum: [0, 0, 0] as const, maximumExclusive: [1, 1, 1] as const }] };
  assert.equal(compileRetainedOpenSceneFineMeans(field, dimensions, h, overridden, options).effectiveMeans[0], .5);
});

test("thin open terrain caps retain their geometric capacity and density integral", () => {
  const result = compileRetainedOpenSceneFineMeans(field, dimensions, h, terrainWorld(254), options);
  close(result.openFractions[0], 1 / 255, 1e-9);
  close(result.effectiveMeans[0], .5 / 255 ** 2, 1e-11);
  assert.ok(result.effectiveMeans.every((q, i) => q >= 0 && q <= result.openFractions[i]));
});

test("live carve receipts preserve the old accepted density on the surviving open subvolume", () => {
  const oldWorld = terrainWorld(), previous = compileRetainedOpenSceneFineMeans(field, dimensions, h, oldWorld, options);
  const carved = withSolidWorldPatches(oldWorld, [{ operation: "clear", minimum: [0, 0, 0], maximumExclusive: [1, 1, 1] }]);
  const edit = compileRetainedOpenSceneEditMoments(field, previous, carved, options);
  close(edit.newlyOpenedFractions[0], 128 / 255); assert.equal(edit.newlyClosedFractions[0], 0);
  assert.equal(edit.next.effectiveMeans[0], .5);
  assert.equal(edit.survivingSeedMeans[0], previous.effectiveMeans[0]);
  assert.equal(edit.survivingOpenFractions[0], previous.openFractions[0]);
  // Accepted a*qSeed+b stays unchanged in old open space; new empty capacity
  // is a separate geometric receipt rather than automatically reseeded water.
  const a = .6, b = .2, amount = a * edit.survivingSeedMeans[0] + b * edit.survivingOpenFractions[0];
  let oracle = 0;
  const low = h * 128 / 255, width = h - low;
  for (let i = 0; i < 10000; i++) oracle += (a * evaluateRetainedSceneDensity(field, [-.45, low + (i + .5) * width / 10000, -.45]) + b) * width / (10000 * h);
  close(amount, oracle);
  assert.ok(amount < a * edit.next.effectiveMeans[0] + b * edit.next.openFractions[0] - .1);
});

test("live fill receipts expose displaced support without inventing an automatic mass policy", () => {
  const oldWorld = terrainWorld(), previous = compileRetainedOpenSceneFineMeans(field, dimensions, h, oldWorld, options);
  const filled = withSolidWorldPatches(oldWorld, [{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [1, 1, 1] }]);
  const edit = compileRetainedOpenSceneEditMoments(field, previous, filled, options);
  close(edit.newlyClosedFractions[0], 127 / 255); assert.equal(edit.newlyOpenedFractions[0], 0);
  assert.equal(edit.survivingSeedMeans[0], 0); assert.equal(edit.survivingOpenFractions[0], 0);
  assert.equal(edit.next.effectiveMeans[0], 0);
  assert.throws(() => compileRetainedOpenSceneEditMoments({ ...field, generation: 4 }, previous, filled, options), /different seed field generation/);
});
