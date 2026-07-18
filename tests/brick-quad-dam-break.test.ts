import assert from "node:assert/strict";
import test from "node:test";
import { createBrickQuadDamBreakScene, getScenePreset } from "../lib/scenes";
import { initialFluidBrickContainsCell } from "../lib/initial-fluid";
import { createTallCellLayout, type GPUQuality } from "../lib/tall-cell-grid";
import { createSmokeScenario, isSmokeScenarioId } from "../tools/webgpu-smoke-scenarios";
import { validateScene } from "../lib/model";

test("brick-quad tank resolves to exactly four 8-cubed fluid bricks at every quality", () => {
  const scene = createBrickQuadDamBreakScene();
  assert.deepEqual(validateScene(scene), []);
  for (const quality of ["balanced", "high", "ultra"] as GPUQuality[]) {
    const layout = createTallCellLayout(scene, quality);
    assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [16, 8, 16], `${quality} finest grid must be 2x2x1 bricks of 8-cubed cells`);
  }
});

test("brick-quad water starts as a full-height column filling exactly one brick quadrant", () => {
  const scene = getScenePreset("brick-quad-dam-break").create();
  assert.equal(scene.fluid.initialBrickSeeds_m?.length, 1);
  const dimensions = [16, 8, 16] as const;
  const wetBricks = new Set<string>();
  let wetCells = 0;
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) for (let x = 0; x < dimensions[0]; x += 1) {
    if (!initialFluidBrickContainsCell(scene, x, y, z, dimensions)) continue;
    wetCells += 1;
    wetBricks.add(`${Math.floor(x / 8)},${Math.floor(y / 8)},${Math.floor(z / 8)}`);
  }
  assert.equal(wetCells, 8 ** 3, "the seed must fill exactly one 8-cubed brick");
  assert.deepEqual([...wetBricks], ["0,0,0"], "the seeded brick is the -x/-z quadrant");
  for (let y = 0; y < dimensions[1]; y += 1) {
    assert.ok(initialFluidBrickContainsCell(scene, 3, y, 3, dimensions), `layer ${y} of the seeded quadrant must be wet (full-height column)`);
  }
});

test("brick-quad scene is registered in the UI presets and the smoke harness", () => {
  const preset = getScenePreset("brick-quad-dam-break");
  assert.equal(preset.id, "brick-quad-dam-break", "the preset must exist rather than fall back to the default scene");
  assert.match(preset.description, /cross-brick/);
  assert.ok(isSmokeScenarioId("brick-quad-dam-break"));
  const scenario = createSmokeScenario("brick-quad-dam-break");
  assert.equal(scenario.scene.numerics.surfaceColumnsOverride, 256);
  assert.equal(scenario.scene.fluid.initialBrickSeeds_m?.length, 1);
  assert.equal(scenario.scene.container.width_m, 0.8);
  assert.equal(scenario.scene.container.height_m, 0.4);
  assert.equal(scenario.scene.container.depth_m, 0.8);
});
