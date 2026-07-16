import { pathToFileURL } from "node:url";
import { tallCellMethod } from "../lib/methods/tall-cell";
import { cloneScene, defaultScene } from "../lib/model";
import { advanceRigidBodies, initializeRigidBodies } from "../lib/rigid-body";
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

const scene = cloneScene(defaultScene);
scene.sceneId = "smoke-rigid-symmetry";
scene.fluid.initialCondition = "tank-fill";
scene.fluid.surfaceTension_N_m = 0;
scene.container.fillFraction = 0.6;
scene.container.top = "open";
delete scene.fluid.inflow;
scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 120;
scene.rigidBodies = [{
  id: "symmetry-sphere",
  name: "Symmetry sphere",
  shape: "sphere",
  dimensions_m: { x: 0.08, y: 0.08, z: 0.08 },
  density_kg_m3: 500,
  position_m: { x: 0, y: 0.3, z: 0 },
  orientation: { w: 1, x: 0, y: 0, z: 0 },
  linearVelocity_m_s: { x: 0, y: 0, z: 0 },
  angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
  restitution: 0.05,
  friction: 0.8,
  motion: "dynamic"
}];

let pendingLoads: GPURigidLoad[] = [];
const values = tallCellMethod.presetFor("balanced");
const solver = tallCellMethod.createSolver!(device, scene, "balanced", values, (loads) => {
  pendingLoads = mergeGPURigidLoads(pendingLoads, loads);
});
const bodies = initializeRigidBodies(scene.rigidBodies);
// Preserve the dynamic mass/inertia while holding the body fixed long enough
// to measure the hydrostatic reaction before release.
bodies[0].description.motion = "static";
const dt = scene.numerics.fixedDt_s;
const releaseTime_s = Number(process.env.FLUID_RIGID_RELEASE_S ?? 0.25);
const targetTime_s = Number(process.env.FLUID_TARGET_S ?? 4);
const maximumStaticLateralForce_N = Number(process.env.FLUID_RIGID_STATIC_FORCE_LIMIT_N ?? 0.05);
const maximumLateralDisplacement_m = Number(process.env.FLUID_RIGID_DRIFT_LIMIT_M ?? 0.02);
let time_s = 0;
let peakStaticLateralForce_N = 0;
let peakLateralDisplacement_m = 0;

while (time_s < targetTime_s - 1e-9) {
  const body = bodies[0];
  if (time_s >= releaseTime_s && body.description.motion === "static") body.description.motion = "dynamic";
  const coupling = externalLoadsFromGPU(scene, pendingLoads, dt, bodies);
  if (body.description.motion === "static") {
    const load = coupling.loads.get(body.description.id);
    peakStaticLateralForce_N = Math.max(peakStaticLateralForce_N, Math.hypot(load?.force_N.x ?? 0, load?.force_N.z ?? 0));
  }
  advanceRigidBodies(bodies, scene, dt, 6, coupling.loads);
  const nextTime = Math.min(targetTime_s, time_s + dt);
  while (!solver.advanceTo(nextTime, bodies)) await new Promise((resolve) => setTimeout(resolve, 0));
  time_s = nextTime;
  await device.queue.onSubmittedWorkDone();
  await new Promise((resolve) => setTimeout(resolve, 0));
  peakLateralDisplacement_m = Math.max(peakLateralDisplacement_m, Math.hypot(body.position_m.x, body.position_m.z));
}

const body = bodies[0];
const result = {
  scenario: scene.sceneId,
  method: tallCellMethod.id,
  simulatedTime_s: time_s,
  finalPosition_m: body.position_m,
  finalVelocity_m_s: body.linearVelocity_m_s,
  peakStaticLateralForce_N,
  peakLateralDisplacement_m,
  limits: { maximumStaticLateralForce_N, maximumLateralDisplacement_m },
  validationErrors
};
console.log(JSON.stringify(result));

const failures: string[] = [];
if (validationErrors.length > 0) failures.push(`${validationErrors.length} WebGPU validation error(s)`);
if (peakStaticLateralForce_N > maximumStaticLateralForce_N) failures.push(`static lateral force ${peakStaticLateralForce_N.toFixed(6)} N > ${maximumStaticLateralForce_N} N`);
if (peakLateralDisplacement_m > maximumLateralDisplacement_m) failures.push(`lateral drift ${peakLateralDisplacement_m.toFixed(6)} m > ${maximumLateralDisplacement_m} m`);
solver.destroy();
device.destroy();
if (failures.length > 0) throw new Error(`Rigid symmetry smoke failed: ${failures.join("; ")}`);
