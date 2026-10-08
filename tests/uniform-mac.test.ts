import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { getMethod, interactiveSimulationMethods, defaultMethodId } from "../lib/core/method-registry";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { macOptions, macTimeStep, validateMacScene } from "../lib/methods/uniform/mac/parameters";

test("MAC baseline installs through the method catalog without changing the default", async () => {
  const method = getMethod("uniform-mac");
  assert.equal(method.id, "uniform-mac"); assert.ok(interactiveSimulationMethods().includes(method));
  assert.equal(defaultMethodId(), "uniform-narrow-band-flip"); assert.ok(Object.isFrozen(method.composition));
  assert.equal((await method.pipelineGraph!()).methodId, method.id);
  assert.equal((await method.harness!()).methodId, method.id);
  assert.ok(method.supportedFieldModes?.includes("volume-levelset"));
  assert.ok(!method.supportedFieldModes?.includes("fine-tiles"));
});

test("MAC time step bounds acceleration, transport, viscosity and capillarity", () => {
  const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  scene.fluid.dynamicViscosity_Pa_s = 0; scene.fluid.surfaceTension_N_m = 0;
  scene.fluid.gravity_m_s2 = { x: 0, y: -9.81, z: 0 };
  const options = macOptions({ maxStep: 1 / 30, cfl: 0.5 });
  const dt = macTimeStep(scene, [0.01, 0.02, 0.01], 2, options, 1);
  assert.ok(2 * dt + 0.5 * 9.81 * dt * dt <= 0.005 + 1e-12);
  assert.ok(macTimeStep(scene, [0.01, 0.01, 0.01], 0, options, 0.001) <= 0.001);
  scene.fluid.dynamicViscosity_Pa_s = 1000;
  assert.ok(macTimeStep(scene, [0.01, 0.01, 0.01], 0, options, 1) < dt);
  scene.fluid.dynamicViscosity_Pa_s = 0; scene.fluid.surfaceTension_N_m = 100;
  assert.ok(macTimeStep(scene, [0.01, 0.01, 0.01], 0, options, 1) < dt);
});

test("MAC rejects unsupported sources rather than silently dropping their physics", () => {
  const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  scene.rigidBodies = []; scene.fluid.inflow = undefined; validateMacScene(scene);
  scene.fluid.inflow = { enabled: true } as NonNullable<typeof scene.fluid.inflow>;
  assert.throws(() => validateMacScene(scene), /inflow/);
});
