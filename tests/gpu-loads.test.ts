import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, type Vec3 } from "../lib/model";
import { initializeRigidBody, primitiveVolume } from "../lib/rigid-body";
import { externalLoadsFromGPU, GPU_RIGID_ADDED_MASS_COEFFICIENT } from "../lib/simulation/gpu-loads";
import type { GPURigidLoad } from "../lib/webgpu-eulerian";

const zero = (): Vec3 => ({ x: 0, y: 0, z: 0 });

function sphere(density_kg_m3 = 100) {
  const source = defaultScene.rigidBodies[0];
  return initializeRigidBody({
    ...source,
    id: "sphere",
    shape: "sphere",
    dimensions_m: { x: 0.08, y: 0.08, z: 0.08 },
    density_kg_m3,
    linearVelocity_m_s: zero(),
    angularVelocity_rad_s: zero()
  });
}

function gpuLoad(displacedVolume_m3: number, impulse_N_s: Vec3 = zero(), meanFluidVelocity_m_s: Vec3 = zero()): GPURigidLoad {
  return { bodyId: "sphere", impulse_N_s, angularImpulse_N_m_s: zero(), couplingInterval_s: 0.01, displacedVolume_m3, meanFluidVelocity_m_s };
}

test("fully submerged light-body acceleration includes added mass", () => {
  const scene = cloneScene(defaultScene), body = sphere(), volume = primitiveVolume("sphere", body.description.dimensions_m), dt = 0.01;
  scene.fluid.gravity_m_s2 = { x: 0, y: -9.80665, z: 0 };
  const { loads } = externalLoadsFromGPU(scene, [gpuLoad(volume)], dt, [body]);
  const force = loads.get(body.description.id)!.force_N;
  const acceleration = scene.fluid.gravity_m_s2.y + force.y / body.mass_kg;
  const addedMass = GPU_RIGID_ADDED_MASS_COEFFICIENT * scene.fluid.density_kg_m3 * volume;
  const expected = 9.80665 * (scene.fluid.density_kg_m3 * volume - body.mass_kg) / (body.mass_kg + addedMass);
  assert.ok(Math.abs(acceleration - expected) < 1e-10, `${acceleration} != ${expected}`);
  assert.ok(acceleration / 9.80665 < 1.6, "the 100 kg/m^3 sphere should rise at about 1.5 g, not 9 g");
});

test("an airborne load preserves the hydrodynamic impulse exactly", () => {
  const scene = cloneScene(defaultScene), body = sphere(), dt = 0.01, impulse = { x: 0.03, y: -0.02, z: 0.01 };
  const { loads } = externalLoadsFromGPU(scene, [gpuLoad(0, impulse, { x: 8, y: 8, z: 8 })], dt, [body]);
  assert.deepEqual(loads.get(body.description.id)!.force_N, { x: 3, y: -2, z: 1 });
});

test("quadratic drag opposes body velocity relative to the wet fluid", () => {
  const scene = cloneScene(defaultScene), body = sphere(), volume = primitiveVolume("sphere", body.description.dimensions_m);
  body.linearVelocity_m_s = { x: 1.5, y: -0.5, z: 0.25 };
  const relative = { x: 1, y: -0.25, z: 0.25 };
  const mean = { x: body.linearVelocity_m_s.x - relative.x, y: body.linearVelocity_m_s.y - relative.y, z: body.linearVelocity_m_s.z - relative.z };
  const { loads } = externalLoadsFromGPU({ fluid: { ...scene.fluid, gravity_m_s2: zero() } }, [gpuLoad(volume, zero(), mean)], 0.01, [body]);
  const drag = loads.get(body.description.id)!.hydrodynamicForce_N!;
  assert.ok(drag.x < 0 && drag.y > 0 && drag.z < 0);
  assert.ok(drag.x * relative.x + drag.y * relative.y + drag.z * relative.z < 0);
});
