import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { defaultMethodId, getMethod, interactiveSimulationMethods } from "../lib/core/method-registry";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { seedApicParticles } from "../lib/methods/particle/seed";
import { apicOptions, validateApicScene } from "../lib/methods/particle/parameters";

test("APIC is an independent selectable plugin with matching pipeline and harness identities", async () => {
  const method = getMethod("particle-apic");
  assert.equal(method.id, "particle-apic"); assert.ok(interactiveSimulationMethods().includes(method));
  assert.equal(defaultMethodId(), "uniform-volume"); assert.ok(Object.isFrozen(method.composition));
  assert.equal((await method.harness!()).methodId, method.id);
  assert.equal((await method.pipelineGraph!()).methodId, method.id);
  assert.ok(!method.params.some(p => p.key === "advection"), "Eulerian transport controls must not leak into APIC");
  assert.ok(method.supportedFieldModes?.includes("volume-levelset"));
});

test("particle seeding accounts for material, excludes solids and rejects overflow", () => {
  const phi = new Float32Array(27).fill(-1), solid = new Uint32Array(8); solid[0] = 1;
  const seed = seedApicParticles(phi, solid, [2, 2, 2], [0.1, 0.2, 0.3], 100);
  assert.equal(seed.count, 56); assert.ok(Math.abs(seed.volume_m3 - 7 * 0.006) < 1e-12);
  for (let i = 0; i < seed.count; i++) {
    const x = seed.data[i * 20], y = seed.data[i * 20 + 1], z = seed.data[i * 20 + 2];
    assert.ok(x >= 0.1 || y >= 0.2 || z >= 0.3);
  }
  assert.deepEqual(seed.data, seedApicParticles(phi, solid, [2, 2, 2], [0.1, 0.2, 0.3], 100).data);
  assert.throws(() => seedApicParticles(phi, solid, [2, 2, 2], [0.1, 0.2, 0.3], 10), /increase the scene cell size/);
  assert.equal(seedApicParticles(new Float32Array(27).fill(1), solid, [2, 2, 2], [1, 1, 1], 100).count, 0);
});

test("APIC refuses unsupported coupling and sources explicitly", () => {
  const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  scene.rigidBodies = []; scene.fluid.inflow = undefined; validateApicScene(scene);
  scene.fluid.inflow = { enabled: true } as NonNullable<typeof scene.fluid.inflow>;
  assert.throws(() => validateApicScene(scene), /APIC.*inflows/);
});


test("APIC pressure defaults match Uniform Geometric's absolute and relative targets", async () => {
  const uniform = await import("../lib/methods/uniform/pressure-policy");
  const { MAC_PRESSURE_RELATIVE_FLOOR } = await import("../lib/methods/mac-shared/pressure-target");
  const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-64"));
  const defaults = apicOptions(scene, {});
  assert.equal(defaults.tolerance, uniform.UNIFORM_PRESSURE_RESIDUAL_TOLERANCE);
  assert.equal(defaults.relativeReduction, uniform.UNIFORM_PRESSURE_RELATIVE_REDUCTION);
  assert.equal(MAC_PRESSURE_RELATIVE_FLOOR, uniform.UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE);
  assert.equal(apicOptions(scene, { pressureTolerance: 5 }).tolerance, 5, "shared MAC's old clamp must not truncate APIC's ceiling");
  assert.equal(apicOptions(scene, { pressureRelativeReduction: 0 }).relativeReduction, 0);
  assert.equal(apicOptions(scene, { pressureTolerance: 0.001 }).tolerance, 0.001);
});
