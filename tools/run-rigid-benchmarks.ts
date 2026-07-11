import { advanceRigidBodies, cloneRigidBodies, initializeRigidBody, massProperties, quaternionNormalize, rigidDiagnostics } from "../lib/rigid-body";
import { cloneScene, defaultScene, type RigidBodyDescription } from "../lib/model";

const sphere = (overrides: Partial<RigidBodyDescription> = {}): RigidBodyDescription => ({
  id: "benchmark-sphere", name: "Benchmark sphere", shape: "sphere",
  dimensions_m: { x: 0.1, y: 0.1, z: 0.1 }, density_kg_m3: 1000,
  position_m: { x: 0, y: 10, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
  linearVelocity_m_s: { x: 0, y: 0.7, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
  restitution: 0.4, friction: 0.3, ...overrides
});
const scene = cloneScene(defaultScene);
scene.container = { ...scene.container, width_m: 20, height_m: 20, depth_m: 20, top: "open" };
scene.rigidBodies = [];

const duration = 0.5;
const exactY = 10 + 0.7 * duration + 0.5 * scene.fluid.gravity_m_s2.y * duration ** 2;
const convergence = [0.004, 0.002, 0.001].map((dt) => {
  const body = initializeRigidBody(sphere());
  for (let i = 0; i < Math.round(duration / dt); i += 1) advanceRigidBodies([body], scene, dt);
  return { dt_s: dt, y_m: body.position_m.y, absoluteError_m: Math.abs(body.position_m.y - exactY) };
});

const collisionScene = cloneScene(scene);
collisionScene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
const a = initializeRigidBody(sphere({ id: "a", position_m: { x: -0.105, y: 2, z: 0 }, linearVelocity_m_s: { x: 1, y: 0, z: 0 }, restitution: 0.5, friction: 0 }));
const b = initializeRigidBody(sphere({ id: "b", density_kg_m3: 2000, position_m: { x: 0.105, y: 2, z: 0 }, linearVelocity_m_s: { x: -0.25, y: 0, z: 0 }, restitution: 0.5, friction: 0 }));
const before = rigidDiagnostics([a, b], collisionScene.fluid.gravity_m_s2);
const after = advanceRigidBodies([a, b], collisionScene, 0.01);

const floorBody = initializeRigidBody(sphere({ position_m: { x: 0, y: 0.6, z: 0 }, linearVelocity_m_s: { x: 0, y: 0, z: 0 }, restitution: 0 }));
for (let i = 0; i < 3000; i += 1) advanceRigidBodies([floorBody], scene, 0.001);

const spinBody = initializeRigidBody(sphere({ orientation: quaternionNormalize({ w: 0.7, x: 0.2, y: 0.4, z: 0.1 }), angularVelocity_rad_s: { x: 3, y: -2, z: 1 } }));
const spinScene = cloneScene(scene); spinScene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
for (let i = 0; i < 2000; i += 1) advanceRigidBodies([spinBody], spinScene, 0.0005);

const replayA = cloneRigidBodies([spinBody]);
const replayB = cloneRigidBodies([spinBody]);
for (let i = 0; i < 250; i += 1) { advanceRigidBodies(replayA, spinScene, 0.001); advanceRigidBodies(replayB, spinScene, 0.001); }

const properties = massProperties(sphere());
const analyticMass = 1000 * 4 * Math.PI * 0.1 ** 3 / 3;
const result = {
  benchmark: "stage3-rigid-cpu-reference",
  buildId: "web-stage3-0.2.0",
  precision: "JavaScript Number / IEEE-754 binary64",
  mass: { measured_kg: properties.mass_kg, expected_kg: analyticMass, relativeError: Math.abs(properties.mass_kg - analyticMass) / analyticMass },
  freeFall: { duration_s: duration, expectedY_m: exactY, convergence },
  sphereCollision: {
    momentumBefore_kg_m_s: before.linearMomentum_kg_m_s.x,
    momentumAfter_kg_m_s: after.linearMomentum_kg_m_s.x,
    relativeDrift: Math.abs(after.linearMomentum_kg_m_s.x - before.linearMomentum_kg_m_s.x) / Math.abs(before.linearMomentum_kg_m_s.x),
    kineticEnergyBefore_J: before.kineticEnergy_J,
    kineticEnergyAfter_J: after.kineticEnergy_J,
    energyRatio: after.kineticEnergy_J / before.kineticEnergy_J
  },
  floorContact: { finalY_m: floorBody.position_m.y, finalPenetration_m: Math.max(0, 0.1 - floorBody.position_m.y), finalVerticalSpeed_m_s: floorBody.linearVelocity_m_s.y },
  quaternion: { finalNorm: Math.hypot(spinBody.orientation.w, spinBody.orientation.x, spinBody.orientation.y, spinBody.orientation.z) },
  deterministicReplay: { byteIdentical: JSON.stringify(replayA) === JSON.stringify(replayB) },
  invalidValueCount: rigidDiagnostics([...replayA, floorBody], scene.fluid.gravity_m_s2).nanCount
};

console.log(JSON.stringify(result, null, 2));
