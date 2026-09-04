import assert from "node:assert/strict";
import test from "node:test";
import { addFluidBall, fluidBallRequiresInitialCondition } from "../lib/core/editor-fluid-volume";
import { cloneScene } from "../lib/core/model";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";

test("the first liquid ball enables fluid authority in a dry garden", () => {
  const scene = cloneScene(sceneDocument(getSceneDefinition("garden-svo-lighting")));
  assert.equal(scene.systems?.fluid, false);

  const addition = addFluidBall(scene, { x: 0, y: 0.5, z: 0 }, 0.1);

  assert.equal(addition.systems?.fluid, true);
  assert.equal(addition.fluid.initialLiquidVolumes?.length, 1);
  assert.equal(scene.systems?.fluid, false, "the authored source remains immutable");
});

test("a ready fluid scene injects at time zero instead of rebuilding its seed", () => {
  const dry = sceneDocument(getSceneDefinition("garden-svo-lighting"));
  const wet = { ...dry, systems: { ...dry.systems, fluid: true } };

  assert.equal(fluidBallRequiresInitialCondition(dry, 0, true), true);
  assert.equal(fluidBallRequiresInitialCondition(wet, 0, true), false);
  assert.equal(fluidBallRequiresInitialCondition(wet, 1, true), false);
});
