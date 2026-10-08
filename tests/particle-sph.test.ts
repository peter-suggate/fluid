import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { defaultMethodId, getMethod, interactiveSimulationMethods } from "../lib/core/method-registry";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { seedLiquidParticles } from "../lib/core/seed-liquid-particles";
import { sphKernelNormalization, sphOptions, validateSphScene } from "../lib/methods/sph/parameters";

test("SPH is selectable with its own controls, pipeline and harness", async () => {
  const method = getMethod("particle-sph");
  assert.equal(method.id, "particle-sph"); assert.ok(interactiveSimulationMethods().includes(method));
  assert.equal(defaultMethodId(), "uniform-narrow-band-flip"); assert.ok(Object.isFrozen(method.composition));
  assert.equal((await method.harness!()).methodId, method.id); assert.equal((await method.pipelineGraph!()).methodId, method.id);
  assert.deepEqual(method.params.map(p => p.key), ["particlesPerCell", "artificialViscosity", "soundSpeed", "cfl"]);
  assert.ok(!method.runtimeParamKeys?.includes("particlesPerCell"));
  assert.ok(method.supportedFieldModes?.includes("volume-levelset"));
});
test("SPH compact seeds preserve quadrature material and exclude solids", () => {
  const solid = new Uint32Array(8); solid[0] = 1;
  const seed = seedLiquidParticles(new Float32Array(27).fill(-1), solid, [2,2,2], [0.1,0.2,0.3], 100);
  const coarse = seedLiquidParticles(new Float32Array(27).fill(-1), solid, [2,2,2], [0.1,0.2,0.3], 100, 8, "SPH", 1);
  assert.equal(coarse.count, 7); assert.equal(coarse.volume_m3, seed.volume_m3);
  assert.equal(seed.count, 56); assert.equal(seed.data.length, 56 * 8); assert.ok(Math.abs(seed.volume_m3 - 0.042) < 1e-12);
  assert.throws(() => seedLiquidParticles(new Float32Array(27).fill(-1), solid, [2,2,2], [1,1,1], 10, 8, "SPH"), /SPH.*increase/);
  assert.equal(seedLiquidParticles(new Float32Array(27).fill(1), solid, [2,2,2], [1,1,1], 100).count, 0);
});
test("SPH kernel calibration is scale independent and controls are bounded", () => {
  assert.ok(Math.abs(sphKernelNormalization([1,1,1]) - sphKernelNormalization([0.01,0.01,0.01])) < 1e-12);
  assert.ok(sphKernelNormalization([1,1,1]) > 0.9 && sphKernelNormalization([1,1,1]) < 1.1);
  assert.deepEqual(sphOptions({ soundSpeed: Infinity, cfl: 100 }), { artificialViscosity: 0.2, soundSpeed: 20, cfl: 0.3 });
});
test("existing scenes with unsupported sources still open; invalid material is rejected", () => {
  const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  scene.fluid.inflow = { enabled: true } as NonNullable<typeof scene.fluid.inflow>;
  scene.rigidBodies = [{}] as typeof scene.rigidBodies;
  validateSphScene(scene);
  scene.fluid.density_kg_m3 = 0; assert.throws(() => validateSphScene(scene), /positive density/);
});
