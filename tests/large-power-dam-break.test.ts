import assert from "node:assert/strict";
import test from "node:test";
import { validateScene } from "../lib/model";
import { sceneDamBreakFractions } from "../lib/initial-fluid";
import {
  LARGE_POWER_DAM_METHOD_PROFILE,
  LARGE_POWER_DAM_FINE_BRICK_CAPACITY,
  POWER_VALIDATION_METHOD_PROFILE,
  createLargePowerDamBreakScene,
  createMinimalPowerDamBreakScene,
  getScenePreset,
} from "../lib/scenes";
import { createTallCellLayout, type GPUQuality } from "../lib/tall-cell-grid";

test("large power dam break has a 20x container and the same absolute water block", () => {
  const mini = createMinimalPowerDamBreakScene();
  const large = createLargePowerDamBreakScene();

  assert.deepEqual(validateScene(large), []);
  assert.deepEqual(large.container, {
    ...mini.container,
    width_m: 3.2,
    height_m: 1,
    depth_m: 3.2,
    fillFraction: mini.container.fillFraction / 20,
  });
  const miniVolume = mini.container.width_m * mini.container.height_m * mini.container.depth_m;
  const largeVolume = large.container.width_m * large.container.height_m * large.container.depth_m;
  assert.equal(largeVolume / miniVolume, 20);

  const miniDam = sceneDamBreakFractions(mini);
  const largeDam = sceneDamBreakFractions(large);
  const absoluteDamDimensions = (scene: typeof mini, dam: typeof miniDam) => [
    dam.width * scene.container.width_m,
    dam.height * scene.container.height_m,
    dam.depth * scene.container.depth_m,
  ];
  assert.deepEqual(absoluteDamDimensions(large, largeDam), absoluteDamDimensions(mini, miniDam));
  assert.ok(Math.abs(large.container.fillFraction * largeVolume
    - mini.container.fillFraction * miniVolume) < 1e-12);
  assert.deepEqual(large.fluid.initialDamBreakDimensions_m, { x: 0.5, y: 0.92 * 0.8, z: 0.5 });
  const largeFluid = { ...large.fluid };
  delete largeFluid.initialDamBreakDimensions_m;
  assert.deepEqual(largeFluid, mini.fluid);
  assert.deepEqual(large.voxelDomain, mini.voxelDomain);
  assert.deepEqual(large.numerics, mini.numerics);
  assert.deepEqual(large.rigidBodies, mini.rigidBodies);

  for (const quality of ["balanced", "high", "ultra"] as GPUQuality[]) {
    const miniLayout = createTallCellLayout(mini, quality);
    const layout = createTallCellLayout(large, quality);
    assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [64, 20, 64]);
    assert.equal(layout.initialVolumeCellSum, miniLayout.initialVolumeCellSum);
    assert.ok(Math.abs(layout.referenceLiquidVolume_cells
      - miniLayout.referenceLiquidVolume_cells) < 1e-6);
  }
});

test("large power dam break is available as a comparison preset", () => {
  const preset = getScenePreset("large-power-dam-break");
  assert.equal(LARGE_POWER_DAM_FINE_BRICK_CAPACITY, 8 * 64 * 64,
    "band-1 reserves seven floor-sheet layers plus one footprint of deformation headroom");
  assert.equal(preset.id, "large-power-dam-break");
  assert.equal(preset.methodProfile, LARGE_POWER_DAM_METHOD_PROFILE);
  assert.equal(preset.methodProfile.overrides.maximumLeafSize, "32");
  assert.equal(preset.methodProfile.overrides.interfaceRefinementBandCells, 1);
  assert.equal(preset.methodProfile.overrides.globalFineLevelSetMaximumBricks,
    LARGE_POWER_DAM_FINE_BRICK_CAPACITY);
  assert.equal(getScenePreset("minimal-power-dam-break").methodProfile, POWER_VALIDATION_METHOD_PROFILE);
  assert.equal(preset.create().sceneId, "large-power-dam-break");
});
