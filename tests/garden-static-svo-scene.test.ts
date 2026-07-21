import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { parseScene, serializeScene, validateScene } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { planSceneRuntime } from "../lib/scene-runtime";

test("garden SVO lighting preset is a valid fluid-free static scene", () => {
  const preset = getScenePreset("garden-svo-lighting");
  const scene = preset.create();

  assert.equal(preset.group, "Garden");
  assert.equal(scene.sceneId, "garden-svo-lighting-study");
  assert.equal(scene.systems?.fluid, false);
  const runtimePlan = planSceneRuntime(scene, { methodId: "octree", renderMode: "svo" });
  assert.equal(runtimePlan.staticWorld, true);
  assert.equal(runtimePlan.fluidSolver, false);
  assert.equal(runtimePlan.rigidCoupling, false);
  assert.equal(runtimePlan.waterPresentation, false);
  assert.equal(runtimePlan.sparseVoxelPresentation, true);
  assert.equal(runtimePlan.readiness.fluidAuthority.state, "not-required");
  assert.equal(runtimePlan.readiness.transport.state, "not-required");
  assert.equal(scene.environment, "garden");
  assert.equal(scene.container.fillFraction, 0);
  assert.ok(scene.terrain);
  assert.equal(scene.fluid.inflow, undefined);
  assert.equal(scene.fluid.initialBrickSeeds_m, undefined);
  assert.equal(scene.rigidBodies.some(({ id }) => id === "garden-cork-ball"), false);
  assert.ok(scene.rigidBodies.length >= 3);
  assert.ok(scene.rigidBodies.every(({ motion }) => motion === "static"));
  assert.deepEqual(validateScene(scene), []);

  const roundTrip = parseScene(serializeScene(scene));
  assert.equal(roundTrip.systems?.fluid, false);
});

test("existing garden presets retain ordinary fluid execution", () => {
  for (const id of ["garden-pond", "garden-dam-break", "garden-hose"]) {
    const scene = getScenePreset(id).create();
    assert.equal(planSceneRuntime(scene).fluidSolver, true, id);
    assert.ok(scene.container.fillFraction > 0, id);
  }
});

test("static SVO startup bypasses the simulation solver and t=0 raster gate", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const staticSource = readFileSync(new URL("../lib/webgpu-static-svo-scene.ts", import.meta.url), "utf8");

  assert.match(renderer, /planSceneRuntime\(scene,\{methodId:config\.methodId\}\)\.fluidSolver/);
  assert.match(renderer, /WebGPUStaticSvoScene\.create/);
  assert.match(renderer, /if\(staticRenderScene\)this\.onStatus\(\{state:"ready",label:"Static SVO renderer ready"/);
  assert.doesNotMatch(staticSource, /WebGPUUniformEulerianSolver/);
  assert.match(staticSource, /fluid authority intentionally bypassed/);
  assert.match(staticSource, /new OctreeSparseBrickWorld/);
  assert.match(staticSource, /emptyPhi\.fill/);
});
