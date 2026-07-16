// Buoyant-sphere ejection probe: a low-density sphere is released fully
// submerged in a settled tank and integrated with the controller's exact GPU
// coupling recipe (consumeGPURigidLoad amortization + analytic buoyancy from
// the GPU displacedVolume_m3). The probe compares the GPU-reported displaced
// volume against the analytic spherical-cap submerged volume at the body's
// current position; a healthy coupling keeps that ratio near 1 as the sphere
// approaches and breaches the surface.
// Usage: WEBGPU_NODE_MODULE=... PROBE_METHOD=tall-cell|uniform|quadtree-tall-cell \
//        [PROBE_DENSITY=100] [PROBE_TARGET_S=2.0] npx tsx tmp/tall-cell-audit/probe-buoyant-sphere.ts
import { pathToFileURL } from "node:url";
import { tallCellMethod } from "../../lib/methods/tall-cell";
import { uniformMethod } from "../../lib/methods/uniform";
import { quadtreeTallCellMethod } from "../../lib/methods/quadtree-tall-cell";
import { cloneScene, defaultScene, type SceneDescription } from "../../lib/model";
import { advanceRigidBodies, initializeRigidBodies, type RigidBodyState } from "../../lib/rigid-body";
import { mergeGPURigidLoads, type GPURigidLoad } from "../../lib/webgpu-eulerian";
import { externalLoadsFromGPU } from "../../lib/simulation/gpu-loads";

const modulePath = process.env.WEBGPU_NODE_MODULE!;
const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(o: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
const device = await adapter!.requestDevice();
device.addEventListener("uncapturederror", (event: unknown) => console.error("UNCAPTURED:", (event as { error: { message: string } }).error.message));

const methodId = process.env.PROBE_METHOD ?? "tall-cell";
const method = { "tall-cell": tallCellMethod, "uniform": uniformMethod, "quadtree-tall-cell": quadtreeTallCellMethod }[methodId];
if (!method) throw new Error(`Unknown PROBE_METHOD ${methodId}`);
const sphereDensity = Number(process.env.PROBE_DENSITY ?? 100);
const targetTime = Number(process.env.PROBE_TARGET_S ?? 2.0);

const scene: SceneDescription = cloneScene(defaultScene);
scene.sceneId = "probe-buoyant-sphere";
scene.fluid.initialCondition = "tank-fill";
scene.container.fillFraction = 0.6;
scene.container.top = "open";
scene.fluid.surfaceTension_N_m = 0;
delete scene.fluid.inflow;
scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 120;
const waterHeight = scene.container.height_m * scene.container.fillFraction; // 0.54 m
const radius = 0.08;
const startY = 0.30; // fully submerged: top of sphere 0.16 m below the surface
scene.rigidBodies = [{
  id: "buoyant-sphere", name: "Buoyant sphere", shape: "sphere",
  dimensions_m: { x: radius, y: radius, z: radius }, density_kg_m3: sphereDensity,
  position_m: { x: 0, y: startY, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
  linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
  restitution: 0.05, friction: 0.8, motion: "dynamic"
}];

const values = Object.fromEntries(method.params.map((p) => [p.key, "default" in p ? p.default : 0])) as Record<string, string | number | boolean>;
let pending: GPURigidLoad[] = [];
const onRigidLoads = (incoming: GPURigidLoad[]) => { pending = mergeGPURigidLoads(pending, incoming); };
const solver: unknown = method.createSolver!(device, scene, "balanced", values, onRigidLoads);
const s = solver as {
  info: { nx: number; ny: number; nz: number; storedNy?: number; gridKind?: string; maxSpeed_m_s?: number; lastSubsteps?: number };
  advanceTo(t: number, bodies: RigidBodyState[]): boolean;
  readStats(): Promise<unknown>;
};
const dt = scene.numerics.fixedDt_s;
const bodies = initializeRigidBodies(scene.rigidBodies);
const sphereVolume = (4 / 3) * Math.PI * radius ** 3;
const sphereMass = sphereDensity * sphereVolume;

// Spherical-cap submerged volume for center height y against the still-water level.
function analyticSubmergedVolume(y: number) {
  const depth = Math.max(0, Math.min(2 * radius, waterHeight - (y - radius)));
  return Math.PI * depth * depth * (3 * radius - depth) / 3;
}

function loadsFromGPU(step_dt: number) {
  const coupling = externalLoadsFromGPU(scene, pending, step_dt, bodies);
  return { loads: coupling.loads, displaced: coupling.diagnostics.displacedVolume_m3 };
}

console.log(JSON.stringify({
  method: methodId, gridKind: s.info.gridKind ?? "uniform", grid: [s.info.nx, s.info.ny, s.info.nz], dt,
  sphere: { radius, density: sphereDensity, startY, volume: Number(sphereVolume.toExponential(4)), mass_kg: Number(sphereMass.toFixed(4)) },
  waterHeight: Number(waterHeight.toFixed(3)),
  fullyDrySpeedBound_m_s: Number(Math.sqrt(2 * 9.80665 * (scene.fluid.density_kg_m3 / sphereDensity - 1) * (waterHeight - startY + radius)).toFixed(3))
}));

const sleep = () => new Promise((resolve) => setTimeout(resolve, 0));
let t = 0, peakY = startY, peakVy = 0;
// PROBE_FORCE_SPEED pins the solver's CFL speed estimate so the uniform
// solver's proactive substep count exceeds 1 while the sphere stays static —
// isolates the per-substep displaced-volume accumulation from body motion.
const forcedSpeed = Number(process.env.PROBE_FORCE_SPEED ?? 0);
// Let the tank settle for 0.25 s with the sphere held static before release.
const releaseTime = process.env.PROBE_HOLD_STATIC === "1" ? Infinity : 0.25;
bodies[0].description.motion = "static";
const reportEvery = Math.round(0.1 / dt);
let step = 0;
while (t < targetTime - 1e-9) {
  if (t >= releaseTime && bodies[0].description.motion === "static") {
    bodies[0].description.motion = "dynamic";
    console.log(JSON.stringify({ event: "release", t: Number(t.toFixed(3)) }));
  }
  if (forcedSpeed > 0) s.info.maxSpeed_m_s = forcedSpeed;
  const { loads, displaced } = loadsFromGPU(dt);
  advanceRigidBodies(bodies, scene, dt, 6, loads);
  while (!s.advanceTo(t + dt, bodies)) await sleep();
  t += dt; step += 1;
  await device.queue.onSubmittedWorkDone();
  await sleep(); // let mapAsync deliver rigid loads
  const body = bodies[0];
  if (body.position_m.y > peakY) peakY = body.position_m.y;
  if (body.linearVelocity_m_s.y > peakVy) peakVy = body.linearVelocity_m_s.y;
  if (step % reportEvery === 0) {
    const analytic = analyticSubmergedVolume(body.position_m.y);
    console.log(JSON.stringify({
      t: Number(t.toFixed(3)), y: Number(body.position_m.y.toFixed(4)), vy: Number(body.linearVelocity_m_s.y.toFixed(4)),
      displacedV_m3: Number(displaced.toExponential(4)), analyticV_m3: Number(analytic.toExponential(4)),
      dispOverAnalytic: analytic > 1e-9 ? Number((displaced / analytic).toFixed(3)) : (displaced > 1e-9 ? Infinity : 1),
      dispOverSphere: Number((displaced / sphereVolume).toFixed(3)),
      buoyAccel_g: Number((scene.fluid.density_kg_m3 * displaced * 9.80665 / sphereMass / 9.80665).toFixed(3)),
      meanFluidVy_m_s: Number((pending.find((load) => load.bodyId === body.description.id)?.meanFluidVelocity_m_s.y ?? 0).toFixed(4)),
      hydrodynamicAccel_g: Number((body.hydrodynamicForce_N.y / sphereMass / 9.80665).toFixed(3)),
      substeps: s.info.lastSubsteps ?? 1, maxSpeed_m_s: Number((s.info.maxSpeed_m_s ?? 0).toFixed(3))
    }));
  }
}
console.log(JSON.stringify({ summary: true, method: methodId, peakY: Number(peakY.toFixed(4)), peakVy_m_s: Number(peakVy.toFixed(4)), aboveSurface_m: Number((peakY - radius - waterHeight).toFixed(4)) }));
process.exit(0);
