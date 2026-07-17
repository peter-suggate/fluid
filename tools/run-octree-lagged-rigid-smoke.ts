import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { octreeMethod } from "../lib/methods/octree";
import { createPaperScenario } from "../lib/paper-scenarios";
import { advanceRigidBodies, initializeRigidBodies } from "../lib/rigid-body";
import { gpuBatchDepth } from "../lib/simulation/gpu-clock";
import { externalLoadsFromGPU } from "../lib/simulation/gpu-loads";
import { mergeGPURigidLoads, type GPURigidLoad } from "../lib/webgpu-eulerian";

const modulePath = process.env.WEBGPU_NODE_MODULE;
if (!modulePath) throw new Error("Set WEBGPU_NODE_MODULE to the installed webgpu package index.js");
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });

const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("WebGPU did not expose an adapter");
const device = await adapter.requestDevice();
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));

const scene = createPaperScenario("dam-break-boxes");
const dt = scene.numerics.fixedDt_s;
const target_s = Number(process.env.FLUID_TARGET_S ?? 0.3);
const batchDepth = gpuBatchDepth("octree", dt, true, 60);
const bodies = initializeRigidBodies(scene.rigidBodies);
const initialCenters = new Map(bodies.map((body) => [body.description.id, { ...body.position_m }]));
let pendingLoads: GPURigidLoad[] = [];
let loadCallbacks = 0;
let receivedImpulse_N_s = 0;

const solver = octreeMethod.createSolver!(device, scene, "balanced", octreeMethod.presetFor("balanced"), (loads) => {
  loadCallbacks += 1;
  for (const load of loads) {
    receivedImpulse_N_s += Math.hypot(load.impulse_N_s.x, load.impulse_N_s.y, load.impulse_N_s.z);
  }
  pendingLoads = mergeGPURigidLoads(pendingLoads, loads);
});

let preparedTime_s = 0;
let submittedTime_s = 0;
while (preparedTime_s + 1e-9 < target_s) {
  // Advance one presentation batch from the previous batch's accumulated GPU
  // loads. The fluid submissions below intentionally use this end-of-batch
  // body state, matching the renderer's frame-lagged partitioned exchange.
  const steps = Math.min(batchDepth, Math.ceil((target_s - preparedTime_s) / dt - 1e-9));
  for (let step = 0; step < steps; step += 1) {
    const coupling = externalLoadsFromGPU(scene, pendingLoads, dt, bodies);
    advanceRigidBodies(bodies, scene, dt, 6, coupling.loads);
    preparedTime_s += dt;
  }
  for (let step = 0; step < steps; step += 1) {
    submittedTime_s = Math.min(preparedTime_s, submittedTime_s + dt);
    assert.equal(solver.advanceTo(submittedTime_s, bodies), true, "fluid batch should accept every prepared fixed step");
  }
  await device.queue.onSubmittedWorkDone();
  // Allow all mapAsync continuations from this batch to merge their impulses
  // before the next rigid batch consumes them.
  await new Promise((resolve) => setTimeout(resolve, 0));
}

await device.queue.onSubmittedWorkDone();
const info = await solver.readStats();
const bodyMotion_m = bodies.map((body) => {
  const initial = initialCenters.get(body.description.id)!;
  return Math.hypot(body.position_m.x - initial.x, body.position_m.y - initial.y, body.position_m.z - initial.z);
});
const maximumBodySpeed_m_s = Math.max(...bodies.map((body) => Math.hypot(body.linearVelocity_m_s.x, body.linearVelocity_m_s.y, body.linearVelocity_m_s.z)));
const finiteBodies = bodies.every((body) => [
  body.position_m.x, body.position_m.y, body.position_m.z,
  body.linearVelocity_m_s.x, body.linearVelocity_m_s.y, body.linearVelocity_m_s.z,
  body.angularVelocity_rad_s.x, body.angularVelocity_rad_s.y, body.angularVelocity_rad_s.z
].every(Number.isFinite));

const result = {
  scenario: "octree-frame-lagged-rigid-feedback",
  target_s,
  dt,
  batchDepth,
  loadCallbacks,
  receivedImpulse_N_s,
  bodyMotion_m,
  maximumBodySpeed_m_s,
  pressurePasses: info.quadtreePressureIterationsUsed,
  nonFiniteVelocityCount: info.nonFiniteCount ?? 0,
  validationErrors
};
console.log(JSON.stringify(result));

assert.deepEqual(validationErrors, [], `WebGPU validation errors: ${validationErrors.join("; ")}`);
assert.equal(finiteBodies, true, "lagged rigid integration must remain finite");
assert.ok(loadCallbacks >= Math.ceil(target_s / dt) - batchDepth, "pooled readbacks must preserve nearly every submitted impulse snapshot");
assert.ok(receivedImpulse_N_s > 0, "immersed boxes must receive a non-zero fluid impulse");
assert.ok(Math.max(...bodyMotion_m) > 1e-4, "at least one dynamic body must move");
assert.equal(info.quadtreePressureIterationsUsed, 32, "dynamic bodies must retain the accelerated pressure path");
assert.equal(info.nonFiniteCount ?? 0, 0, "fluid velocity must remain finite");

solver.destroy();
device.destroy();
