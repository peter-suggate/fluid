import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { validateScene } from "../lib/model";
import { createTinyHydrostaticScene, getScenePreset } from "../lib/scenes";
import { createTallCellLayout, type GPUQuality } from "../lib/tall-cell-grid";
import { createSmokeScenario, isSmokeScenarioId } from "../tools/webgpu-smoke-scenarios";

test("tiny hydrostatic oracle is an exact 16-cubed body-free settled tank", () => {
  const scene = createTinyHydrostaticScene();
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(scene.container, {
    width_m: 0.8,
    height_m: 0.8,
    depth_m: 0.8,
    fillFraction: 0.75,
    top: "open",
    fluidWallMode: "free-slip",
  });
  assert.equal(scene.fluid.initialCondition, "tank-fill");
  assert.equal(scene.fluid.surfaceTension_N_m, 0);
  assert.equal(scene.fluid.inflow, undefined);
  assert.equal(scene.fluid.initialBrickSeeds_m, undefined);
  assert.deepEqual(scene.rigidBodies, []);
  assert.deepEqual(scene.voxelDomain, { finestCellSize_m: 0.05, brickSize_cells: 8 });
  assert.equal(scene.numerics.fixedDt_s, 0.004);
  assert.equal(scene.numerics.maxDt_s, 0.004);

  for (const quality of ["balanced", "high", "ultra"] as GPUQuality[]) {
    const layout = createTallCellLayout(scene, quality);
    assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [16, 16, 16]);
  }
});

test("tiny hydrostatic oracle is shared by the UI preset and smoke registry", () => {
  const preset = getScenePreset("hydrostatic-power-two-level");
  assert.equal(preset.id, "hydrostatic-power-two-level", "the preset must exist rather than fall back to the default");
  assert.match(preset.description, /16³/);
  assert.equal(preset.create().sceneId, "tiny-hydrostatic-two-level");

  assert.ok(isSmokeScenarioId("hydrostatic-power-two-level"));
  const smoke = createSmokeScenario("hydrostatic-power-two-level");
  assert.equal(smoke.scene.sceneId, "tiny-hydrostatic-two-level");
  assert.equal(smoke.oracleSteps, 50);
  assert.equal(smoke.target_s, 0.2);
  assert.deepEqual(smoke.scene.container, preset.create().container);
  assert.deepEqual(smoke.scene.voxelDomain, preset.create().voxelDomain);
});

test("tiny hydrostatic isolated Dawn command pins the two-level paper path", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const command = packageJson.scripts["test:webgpu:hydrostatic-power-two-level"];
  assert.match(command, /FLUID_SCENE=hydrostatic-power-two-level/);
  assert.match(command, /FLUID_EXPECT_GRID=16,16,16/);
  assert.match(command, /FLUID_TARGET_S=0\.2/);
  assert.match(command, /FLUID_ORACLE_STEPS=50/);
  assert.match(command, /FLUID_EXPECT_EXACT_STEPS=50/);
  assert.match(command, /FLUID_MAXIMUM_LEAF_SIZE=2/);
  assert.match(command, /FLUID_OCTREE_INTERFACE_BAND=3/);
  assert.match(command, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=4/);
  assert.match(command, /FLUID_POWER_GENERATION_AUDIT=1/);
  assert.match(command, /FLUID_POWER_AUDIT_EVERY_STEPS=1/);
  assert.match(command, /run-webgpu-smoke-isolated\.ts$/);
});
