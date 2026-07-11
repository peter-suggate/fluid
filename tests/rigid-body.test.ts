import assert from "node:assert/strict";
import test from "node:test";
import { advanceRigidBodies, cloneRigidBodies, initializeRigidBody, massProperties, primitiveVolume, quaternionNormalize, rigidDiagnostics, type RigidBodyState } from "../lib/rigid-body";
import { cloneScene, defaultScene, type RigidBodyDescription, type SceneDescription } from "../lib/model";

const relativeError = (actual: number, expected: number) => Math.abs(actual - expected) / Math.max(Math.abs(expected), 1e-30);

function sphere(overrides: Partial<RigidBodyDescription> = {}): RigidBodyDescription {
  return {
    id: "test-sphere", name: "Test sphere", shape: "sphere",
    dimensions_m: { x: 0.1, y: 0.1, z: 0.1 }, density_kg_m3: 1000,
    position_m: { x: 0, y: 1, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0.4, friction: 0.3, ...overrides
  };
}

function emptyScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.container = { ...scene.container, width_m: 20, height_m: 20, depth_m: 20, top: "open" };
  scene.rigidBodies = [];
  return scene;
}

test("R3-01 primitive volumes and masses match analytic values", () => {
  const density = 785;
  const cases: Array<[RigidBodyDescription["shape"], { x: number; y: number; z: number }, number]> = [
    ["sphere", { x: 0.13, y: 0.13, z: 0.13 }, 4 * Math.PI * 0.13 ** 3 / 3],
    ["box", { x: 0.2, y: 0.3, z: 0.4 }, 0.024],
    ["cylinder", { x: 0.11, y: 0.36, z: 0.11 }, Math.PI * 0.11 ** 2 * 0.36],
    ["capsule", { x: 0.08, y: 0.25, z: 0.08 }, Math.PI * 0.08 ** 2 * 0.25 + 4 * Math.PI * 0.08 ** 3 / 3]
  ];
  for (const [shape, dimensions, expected] of cases) {
    const description = sphere({ shape, dimensions_m: dimensions, density_kg_m3: density });
    assert.ok(relativeError(primitiveVolume(shape, dimensions), expected) < 1e-12);
    assert.ok(relativeError(massProperties(description).mass_kg, density * expected) < 1e-12);
  }
});

test("R3-02 sphere, box, and cylinder inertia match analytic values", () => {
  const sphereDescription = sphere();
  const sphereProps = massProperties(sphereDescription);
  assert.ok(relativeError(sphereProps.inertiaBody_kg_m2.x, 0.4 * sphereProps.mass_kg * 0.1 ** 2) < 1e-12);

  const boxDescription = sphere({ shape: "box", dimensions_m: { x: 0.2, y: 0.3, z: 0.4 } });
  const boxProps = massProperties(boxDescription);
  assert.ok(relativeError(boxProps.inertiaBody_kg_m2.x, boxProps.mass_kg * (0.3 ** 2 + 0.4 ** 2) / 12) < 1e-12);
  assert.ok(relativeError(boxProps.inertiaBody_kg_m2.y, boxProps.mass_kg * (0.2 ** 2 + 0.4 ** 2) / 12) < 1e-12);

  const cylinderDescription = sphere({ shape: "cylinder", dimensions_m: { x: 0.1, y: 0.5, z: 0.1 } });
  const cylinderProps = massProperties(cylinderDescription);
  assert.ok(relativeError(cylinderProps.inertiaBody_kg_m2.y, 0.5 * cylinderProps.mass_kg * 0.1 ** 2) < 1e-12);
  assert.ok(relativeError(cylinderProps.inertiaBody_kg_m2.x, cylinderProps.mass_kg * (3 * 0.1 ** 2 + 0.5 ** 2) / 12) < 1e-12);

  const capsuleDescription = sphere({ shape: "capsule", dimensions_m: { x: 0.08, y: 0.24, z: 0.08 } });
  const capsuleProps = massProperties(capsuleDescription);
  const cylinderMass = capsuleDescription.density_kg_m3 * Math.PI * 0.08 ** 2 * 0.24;
  const sphereMass = capsuleDescription.density_kg_m3 * 4 * Math.PI * 0.08 ** 3 / 3;
  const expectedAxial = 0.5 * cylinderMass * 0.08 ** 2 + 0.4 * sphereMass * 0.08 ** 2;
  assert.ok(relativeError(capsuleProps.inertiaBody_kg_m2.y, expectedAxial) < 1e-12);
});

function runFreeFall(dt: number, duration = 0.5): RigidBodyState {
  const scene = emptyScene();
  const body = initializeRigidBody(sphere({ position_m: { x: 0, y: 10, z: 0 }, linearVelocity_m_s: { x: 0, y: 0.7, z: 0 } }));
  const steps = Math.round(duration / dt);
  for (let i = 0; i < steps; i += 1) advanceRigidBodies([body], scene, dt);
  return body;
}

test("R3-03 free fall stays within one percent of analytic motion", () => {
  const dt = 0.001;
  const duration = 0.5;
  const body = runFreeFall(dt, duration);
  const expectedY = 10 + 0.7 * duration + 0.5 * defaultScene.fluid.gravity_m_s2.y * duration ** 2;
  const expectedV = 0.7 + defaultScene.fluid.gravity_m_s2.y * duration;
  assert.ok(relativeError(body.position_m.y, expectedY) < 0.01, `${body.position_m.y} vs ${expectedY}`);
  assert.ok(relativeError(body.linearVelocity_m_s.y, expectedV) < 1e-12, `${body.linearVelocity_m_s.y} vs ${expectedV}`);
});

test("R3-04 symplectic position error decreases under time-step refinement", () => {
  const duration = 0.5;
  const exact = 10 + 0.7 * duration + 0.5 * defaultScene.fluid.gravity_m_s2.y * duration ** 2;
  const errors = [0.004, 0.002, 0.001].map((dt) => Math.abs(runFreeFall(dt, duration).position_m.y - exact));
  assert.ok(errors[1] < errors[0] && errors[2] < errors[1], JSON.stringify(errors));
  assert.ok(errors[0] / errors[1] > 1.9 && errors[1] / errors[2] > 1.9, JSON.stringify(errors));
});

test("R3-05 quaternion remains normalized during torque-free rotation", () => {
  const scene = emptyScene();
  scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  const body = initializeRigidBody(sphere({ orientation: quaternionNormalize({ w: 0.7, x: 0.2, y: 0.4, z: 0.1 }), angularVelocity_rad_s: { x: 3, y: -2, z: 1 } }));
  for (let i = 0; i < 2000; i += 1) advanceRigidBodies([body], scene, 0.0005);
  const norm = Math.hypot(body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z);
  assert.ok(Math.abs(norm - 1) < 1e-12, String(norm));
});

test("R3-06 sphere impulse conserves momentum and R3-10 does not create energy", () => {
  const scene = emptyScene();
  scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  const a = initializeRigidBody(sphere({ id: "a", position_m: { x: -0.105, y: 2, z: 0 }, linearVelocity_m_s: { x: 1, y: 0, z: 0 }, restitution: 0.5, friction: 0 }));
  const b = initializeRigidBody(sphere({ id: "b", density_kg_m3: 2000, position_m: { x: 0.105, y: 2, z: 0 }, linearVelocity_m_s: { x: -0.25, y: 0, z: 0 }, restitution: 0.5, friction: 0 }));
  const before = rigidDiagnostics([a, b], scene.fluid.gravity_m_s2);
  const energyBefore = before.kineticEnergy_J;
  const after = advanceRigidBodies([a, b], scene, 0.01);
  assert.ok(relativeError(after.linearMomentum_kg_m_s.x, before.linearMomentum_kg_m_s.x) < 1e-12);
  assert.ok(after.kineticEnergy_J <= energyBefore * (1 + 1e-12));
});

test("R3-07 floor contact removes persistent penetration", () => {
  const scene = emptyScene();
  const body = initializeRigidBody(sphere({ position_m: { x: 0, y: 0.6, z: 0 }, restitution: 0, friction: 0.5 }));
  for (let i = 0; i < 3000; i += 1) advanceRigidBodies([body], scene, 0.001);
  const finalPenetration = Math.max(0, 0.1 - body.position_m.y);
  assert.ok(finalPenetration < 1e-6, String(finalPenetration));
  assert.ok(Math.abs(body.linearVelocity_m_s.y) < 0.02, String(body.linearVelocity_m_s.y));
});

test("R3-08 deterministic replay is byte-identical and R3-09 state is finite", () => {
  const scene = cloneScene(defaultScene);
  const initial = [initializeRigidBody(sphere({ position_m: { x: 0.1, y: 0.8, z: 0.1 }, angularVelocity_rad_s: { x: 1, y: 2, z: 3 } }))];
  const a = cloneRigidBodies(initial);
  const b = cloneRigidBodies(initial);
  for (let i = 0; i < 750; i += 1) { advanceRigidBodies(a, scene, 0.001); advanceRigidBodies(b, scene, 0.001); }
  assert.equal(JSON.stringify(a), JSON.stringify(b));
  assert.equal(rigidDiagnostics(a, scene.fluid.gravity_m_s2).nanCount, 0);
});
