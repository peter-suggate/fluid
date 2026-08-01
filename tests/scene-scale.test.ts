import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, validateScene, type SceneDescription } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { sceneLatticeDimensions } from "../lib/scene-lattice";
import { scaleScene, sceneScaleOption, sceneScaleSummary } from "../lib/scene-scale";
import { gpuSceneSeedKey, gpuSceneStructuralKey } from "../lib/webgpu-renderer";

const runConfig = { methodId: "octree", quality: "balanced" as const, values: {}, simulationEpoch: 0 };

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

test("world scale moves the extents and the cell size together, leaving the lattice", () => {
  const scene = preset("water-box-dam-break");
  const before = sceneLatticeDimensions(scene);
  const doubled = scaleScene(scene, "world", 2);
  assert.ok(doubled);
  assert.equal(doubled.container.width_m, scene.container.width_m * 2);
  assert.equal(doubled.container.height_m, scene.container.height_m * 2);
  assert.equal(doubled.container.depth_m, scene.container.depth_m * 2);
  assert.equal(doubled.voxelDomain.finestCellSize_m, scene.voxelDomain.finestCellSize_m * 2);
  assert.deepEqual(sceneLatticeDimensions(doubled), before);
  const halved = scaleScene(scene, "world", 0.5);
  assert.ok(halved);
  assert.deepEqual(sceneLatticeDimensions(halved), before);
});

test("world scale is a re-seed, not a rebuild: the structural key holds and the seed key moves", () => {
  const scene = preset("water-box-dam-break");
  const doubled = scaleScene(scene, "world", 2);
  assert.ok(doubled);
  assert.equal(gpuSceneStructuralKey(doubled, runConfig), gpuSceneStructuralKey(scene, runConfig),
    "a world scale must not invalidate the structural tier — that is what would rebuild the arenas");
  assert.notEqual(gpuSceneSeedKey(doubled), gpuSceneSeedKey(scene),
    "the solver must still be told: the same lattice now spans twice the metres");
});

test("detail scale moves the lattice and therefore the structural key", () => {
  const scene = preset("water-box-dam-break");
  const finer = scaleScene(scene, "detail", 2);
  assert.ok(finer);
  assert.equal(finer.voxelDomain.finestCellSize_m, scene.voxelDomain.finestCellSize_m / 2);
  assert.equal(finer.container.width_m, scene.container.width_m, "detail must not resize the world");
  const before = sceneLatticeDimensions(scene), after = sceneLatticeDimensions(finer);
  assert.deepEqual(after, before.map((value) => value * 2));
  assert.notEqual(gpuSceneStructuralKey(finer, runConfig), gpuSceneStructuralKey(scene, runConfig));
});

test("scaling the world carries nothing with it", () => {
  const scene = preset("dam-break-boxes");
  assert.ok(scene.rigidBodies.length > 0, "fixture must have bodies to leave alone");
  const doubled = scaleScene(scene, "world", 2);
  assert.ok(doubled);
  assert.deepEqual(doubled.rigidBodies, scene.rigidBodies);
  assert.deepEqual(doubled.props ?? null, scene.props ?? null);
  assert.deepEqual(doubled.fluid.initialDamBreakDimensions_m ?? null,
    scene.fluid.initialDamBreakDimensions_m ?? null);
  assert.equal(doubled.fluid.gravity_m_s2.y, scene.fluid.gravity_m_s2.y);
  assert.equal(doubled.numerics.maxDt_s, scene.numerics.maxDt_s);
});

test("the hose keeps its size and its place, and is only clamped back inside a shrunken tank", () => {
  const scene = preset("hose-tank");
  assert.ok(scene.fluid.inflow, "fixture must carry a nozzle");
  const grown = scaleScene(scene, "world", 2);
  assert.ok(grown?.fluid.inflow);
  assert.deepEqual(grown.fluid.inflow.center_m, scene.fluid.inflow.center_m,
    "a bigger tank cannot move a nozzle that already fits");
  assert.equal(grown.fluid.inflow.radius_m, scene.fluid.inflow.radius_m);
  const shrunk = scaleScene(scene, "world", 0.5);
  assert.ok(shrunk?.fluid.inflow);
  const c = shrunk.container;
  assert.ok(Math.abs(shrunk.fluid.inflow.center_m.x) <= c.width_m / 2 + 1e-9);
  assert.ok(shrunk.fluid.inflow.center_m.y >= 0 && shrunk.fluid.inflow.center_m.y <= c.height_m + 1e-9);
});

test("a scaled scene still validates, including the fill-fraction invariant", () => {
  for (const id of ["water-box-dam-break", "water-box-tank-fill", "dam-break-boxes", "hose-tank", "minimal-power-dam-break"]) {
    for (const [axis, factor] of [["world", 2], ["world", 0.5], ["detail", 2], ["detail", 0.5]] as const) {
      const scaled = scaleScene(preset(id), axis, factor);
      if (!scaled) continue;
      assert.deepEqual(validateScene(scaled), [], `${id} after ${axis} ×${factor}`);
    }
  }
});

test("a reservoir that no longer fits is repaired rather than left invalid", () => {
  const scene = preset("water-box-dam-break");
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = {
    x: scene.container.width_m, y: scene.container.height_m * 0.5, z: scene.container.depth_m,
  };
  const d = scene.fluid.initialDamBreakDimensions_m;
  scene.container.fillFraction = (d.x * d.y * d.z)
    / (scene.container.width_m * scene.container.height_m * scene.container.depth_m);
  assert.deepEqual(validateScene(scene), []);
  const shrunk = scaleScene(scene, "world", 0.5);
  assert.ok(shrunk);
  const size = shrunk.fluid.initialDamBreakDimensions_m;
  assert.ok(size);
  assert.ok(size.x <= shrunk.container.width_m + 1e-9 && size.y <= shrunk.container.height_m + 1e-9);
  assert.deepEqual(validateScene(shrunk), []);
});

test("brick seeds left outside a shrunken tank are dropped, never emptied to []", () => {
  const scene = preset("water-box-dam-break");
  const c = scene.container;
  scene.fluid.initialBrickSeeds_m = [
    { x: 0, y: c.height_m * 0.1, z: 0 },
    { x: c.width_m * 0.45, y: c.height_m * 0.9, z: 0 },
  ];
  scene.fluid.initialBrickSeedsAdditive = true;
  const shrunk = scaleScene(scene, "world", 0.5);
  assert.ok(shrunk);
  const seeds = shrunk.fluid.initialBrickSeeds_m;
  assert.ok(seeds === undefined || seeds.length > 0, "an empty seed list is rejected by validateScene");
  for (const seed of seeds ?? []) {
    assert.ok(Math.abs(seed.x) < shrunk.container.width_m / 2);
    assert.ok(seed.y >= 0 && seed.y < shrunk.container.height_m);
  }
  assert.deepEqual(validateScene(shrunk), []);
});

test("steps that would exhaust or collapse the lattice are refused with a reason", () => {
  const scene = preset("water-box-dam-break");
  const coarse = cloneScene(scene);
  coarse.voxelDomain.finestCellSize_m = coarse.container.height_m / 8;
  const collapse = sceneScaleOption(sceneScaleSummary(coarse), "detail", 0.5);
  assert.equal(collapse.available, false);
  assert.match(collapse.blocked ?? "", /floor/);
  assert.equal(scaleScene(coarse, "detail", 0.5), undefined);

  const fine = cloneScene(scene);
  fine.voxelDomain.finestCellSize_m = fine.container.height_m / 1024;
  const exhaust = sceneScaleOption(sceneScaleSummary(fine), "detail", 2);
  assert.equal(exhaust.available, false);
  assert.ok(exhaust.blocked);
});

test("the summary reports the lattice each step would produce", () => {
  const scene = preset("water-box-dam-break");
  const summary = sceneScaleSummary(scene);
  assert.deepEqual(summary.dimensions, sceneLatticeDimensions(scene));
  assert.deepEqual(sceneScaleOption(summary, "world", 2).dimensions, summary.dimensions);
  assert.deepEqual(sceneScaleOption(summary, "detail", 2).dimensions,
    summary.dimensions.map((value) => value * 2));
});
