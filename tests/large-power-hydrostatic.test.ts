import assert from "node:assert/strict";
import test from "node:test";
import { validateScene } from "../lib/model";
import {
  LARGE_POWER_HYDROSTATIC_METHOD_PROFILE,
  createLargePowerDamBreakScene,
  createLargePowerHydrostaticScene,
  getScenePreset,
} from "../lib/scenes";
import { createTallCellLayout, type GPUQuality } from "../lib/tall-cell-grid";
import {
  planFluidFootprintFineNarrowBandBrickCapacity,
  planGlobalFineNarrowBandBrickCapacity,
  planOctreeFluidFootprintBudget,
  planOctreePressureCapacity,
} from "../lib/webgpu-octree";
import { planOctreeAirVelocitySupport } from "../lib/webgpu-octree-air-velocity-support";
import {
  OCTREE_AIR_SUPPORT_GPU_FOOTPRINT_HEADROOM_DENOMINATOR,
  OCTREE_AIR_SUPPORT_GPU_FOOTPRINT_HEADROOM_NUMERATOR,
} from "../lib/webgpu-octree-air-velocity-support-gpu";

test("large hydrostatic uses a quarter-volume sparse representable slab", () => {
  const dam = createLargePowerDamBreakScene();
  const still = createLargePowerHydrostaticScene();
  assert.deepEqual(validateScene(still), []);
  assert.equal(still.fluid.initialCondition, "dam-break");
  assert.deepEqual(still.fluid.initialDamBreakDimensions_m, { x: 1.6, y: 0.05, z: 1.6 });
  assert.deepEqual({ ...still.container, fillFraction: dam.container.fillFraction }, dam.container);
  const volume = (scene: typeof still) => scene.container.width_m
    * scene.container.height_m * scene.container.depth_m * scene.container.fillFraction;
  assert.ok(Math.abs(volume(still) / Math.pow(still.voxelDomain.finestCellSize_m, 3) - 1024) < 1e-9);
  assert.equal(still.container.fillFraction, 1 / 80);
  for (const quality of ["balanced", "high", "ultra"] as GPUQuality[]) {
    const layout = createTallCellLayout(still, quality);
    assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [64, 20, 64]);
  }
});

test("large hydrostatic is an authored comparison preset", () => {
  const preset = getScenePreset("large-power-hydrostatic");
  assert.equal(preset.id, "large-power-hydrostatic");
  assert.equal(preset.methodProfile, LARGE_POWER_HYDROSTATIC_METHOD_PROFILE);
  assert.equal(preset.methodProfile.overrides.pressureRowCapacity, undefined);
  assert.equal(preset.methodProfile.overrides.globalFineLevelSetMaximumBricks, undefined);
  assert.equal(preset.create().sceneId, "large-power-hydrostatic");
});

test("large hydrostatic derives every sparse capacity from its authored fluid footprint", () => {
  const scene = createLargePowerHydrostaticScene();
  const dimensions = { nx: 64, ny: 20, nz: 64 };
  const footprint = planOctreeFluidFootprintBudget(scene, dimensions);
  assert.equal(footprint.initialLiquidCells, 1_024);
  assert.equal(footprint.inflowLiquidCells, 0);
  assert.deepEqual(footprint.maximumCell.map((value, axis) =>
    value - footprint.minimumCell[axis]!), [32, 1, 32]);
  const pressure = planOctreePressureCapacity(dimensions, 32, 1, undefined, false,
    scene.container.fillFraction, Number.MAX_SAFE_INTEGER, footprint).rowCapacity;
  const fine = planFluidFootprintFineNarrowBandBrickCapacity([64, 20, 64], [32, 1, 32], 2)
    .maximumResidentBricks;
  assert.equal(pressure, 2_048);
  assert.equal(fine, 8_160);
  const domain = dimensions.nx * dimensions.ny * dimensions.nz;
  const defaultPressure = planOctreePressureCapacity(dimensions, 32, 1, undefined, false,
    scene.container.fillFraction).rowCapacity;
  const defaultFine = planGlobalFineNarrowBandBrickCapacity([64, 20, 64], 2).maximumResidentBricks;
  const unboundedAir = planOctreeAirVelocitySupport(pressure, pressure * 30, 256, domain);
  const footprintAir = planOctreeAirVelocitySupport(pressure, pressure * 30, 256, domain,
    Math.ceil(pressure * OCTREE_AIR_SUPPORT_GPU_FOOTPRINT_HEADROOM_NUMERATOR
      / OCTREE_AIR_SUPPORT_GPU_FOOTPRINT_HEADROOM_DENOMINATOR));
  assert.ok(defaultPressure / pressure >= 2, `${defaultPressure} / ${pressure}`);
  assert.ok(defaultFine / fine >= 2, `${defaultFine} / ${fine}`);
  assert.ok(unboundedAir.supportCapacity / footprintAir.supportCapacity >= 2,
    `${unboundedAir.supportCapacity} / ${footprintAir.supportCapacity}`);
  assert.ok(domain / footprintAir.ownerDirectorySlotCapacity >= 2,
    `${domain} / ${footprintAir.ownerDirectorySlotCapacity}`);
});
