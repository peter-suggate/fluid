import assert from "node:assert/strict";
import test from "node:test";
import { applyFluidReactions, computeFluidLoads, type CouplingFluid } from "../lib/fluid-rigid-coupling";
import { cloneScene, defaultScene, type Vec3 } from "../lib/model";
import { advanceRigidBodies, initializeRigidBody, primitiveVolume } from "../lib/rigid-body";
import { ParticleFluidSolver } from "../lib/particle-solver";

const zero = (): Vec3 => ({ x: 0, y: 0, z: 0 });

function planarFluid(surfaceY: number): CouplingFluid {
  return { sampleOccupancy: (p) => p.y < surfaceY ? 1 : 0, sampleVelocity: zero, samplePressure: () => 0, applyImpulseAt: () => true };
}

test("C6-01 fully immersed neutral body has zero net vertical force", () => {
  const scene = cloneScene(defaultScene); scene.fluid.gravity_m_s2 = { x: 0, y: -9.80665, z: 0 };
  const description = { ...scene.rigidBodies[0], density_kg_m3: scene.fluid.density_kg_m3, position_m: { x: 0, y: 0.25, z: 0 }, linearVelocity_m_s: zero(), angularVelocity_rad_s: zero() };
  const body = initializeRigidBody(description), { loads } = computeFluidLoads(scene, planarFluid(0.8), [body], 9);
  advanceRigidBodies([body], scene, 0.001, 0, loads);
  assert.ok(Math.abs(body.netForce_N.y) < 1e-10, String(body.netForce_N.y));
  assert.ok(Math.abs(body.linearVelocity_m_s.y) < 1e-10);
});

test("C6-02 a body denser than water initially accelerates downward", () => {
  const scene = cloneScene(defaultScene);
  const description = { ...scene.rigidBodies[0], density_kg_m3: 2 * scene.fluid.density_kg_m3, position_m: { x: 0, y: 0.25, z: 0 }, linearVelocity_m_s: zero(), angularVelocity_rad_s: zero() };
  const body = initializeRigidBody(description), { loads } = computeFluidLoads(scene, planarFluid(0.8), [body], 9);
  advanceRigidBodies([body], scene, 0.001, 0, loads);
  assert.ok(body.linearVelocity_m_s.y < 0);
  assert.ok(body.displacedFluidVolume_m3 > 0);
});

test("C6-03 half-wet axis-aligned box displaces half its volume", () => {
  const scene = cloneScene(defaultScene), source = scene.rigidBodies[1];
  const description = { ...source, shape: "box" as const, dimensions_m: { x: 0.2, y: 0.2, z: 0.2 }, position_m: { x: 0, y: 0.5, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 }, linearVelocity_m_s: zero(), angularVelocity_rad_s: zero() };
  const body = initializeRigidBody(description), { loads } = computeFluidLoads(scene, planarFluid(0.5), [body], 8);
  const fraction = (loads.get(body.description.id)?.displacedFluidVolume_m3 ?? 0) / primitiveVolume("box", description.dimensions_m);
  assert.ok(Math.abs(fraction - 0.5) < 0.05, String(fraction));
});

test("C7-01 paired particle/body impulses conserve closed-system momentum", () => {
  const scene = cloneScene(defaultScene); scene.fluid.initialCondition = "tank-fill"; scene.container.fillFraction = 0.8; scene.fluid.gravity_m_s2 = zero();
  const fluid = new ParticleFluidSolver(scene, 600), description = { ...scene.rigidBodies[0], position_m: { x: 0, y: 0.25, z: 0 }, linearVelocity_m_s: { x: 0.8, y: 0, z: 0 }, angularVelocity_rad_s: zero() };
  const body = initializeRigidBody(description), { loads } = computeFluidLoads(scene, fluid, [body], 7);
  const coupling = applyFluidReactions(fluid, [body], loads, 0.002);
  assert.ok(coupling.coupledBodyCount === 1);
  assert.ok(coupling.momentumClosureError_N_s < 1e-12, String(coupling.momentumClosureError_N_s));
});

test("C7-02 multiple bodies receive finite independent loads", () => {
  const scene = cloneScene(defaultScene), bodies = scene.rigidBodies.map((description, i) => initializeRigidBody({ ...description, position_m: { x: i * 0.2, y: 0.3, z: 0 } }));
  const { loads } = computeFluidLoads(scene, planarFluid(0.8), bodies, 7);
  assert.equal(loads.size, bodies.length);
  for (const body of bodies) assert.ok(Number.isFinite(loads.get(body.description.id)!.force_N.y));
});
