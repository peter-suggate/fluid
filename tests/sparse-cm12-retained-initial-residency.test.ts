import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { createSolidWorld } from "../lib/core/solid-world";
import { compileRetainedSceneDensity, compileRetainedSceneFineMeans } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";
import { initializeSparseBrickAtlasFromScene, materializeSparseBrickAtlasDensity } from "../lib/methods/adaptive-mass/sparse-brick-atlas";

test("sparse residency admits authored density missed by old occupancy samples", () => {
  const scene = cloneScene(defaultScene);
  scene.container = { ...scene.container, width_m: 1.6, height_m: 2, depth_m: 1.6, fillFraction: 0 };
  scene.voxelDomain.finestCellSize_m = .05;
  scene.fluid.initialCondition = "tank-fill";
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialHeightField;
  scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: .125, y: 1.525, z: .125 }, radius_m: .005 }];
  scene.fluid.refinementRegions = [];
  const dimensions = [32, 40, 32] as const;
  const field = compileRetainedSceneDensity(scene)!;
  const means = compileRetainedSceneFineMeans(field, dimensions, .05);
  assert.ok(means.reduce((sum, q) => sum + q, 0) > 0);
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: dimensions, brickFineResolution: 8, solidWorld: createSolidWorld(),
    initialFineDensity: means, coarseFirstCurvatureTolerance: .1,
  });
  assert.ok(atlas.bricks.length > 0, "the field must be admitted before topology compilation");
  const density = materializeSparseBrickAtlasDensity(atlas);
  const expected = means.reduce((sum, q) => sum + q, 0);
  assert.ok(Math.abs(density.reduce((sum, q) => sum + q, 0) - expected) < 1e-8,
    "initial native density must restrict the admitted field's amount");
});
