/**
 * Two-way fluid/rigid coupling gate for the resident power-octree lane.
 *
 * The octree constrains a solid's normal velocity variationally: every cut
 * face carries `(1 - aperture) * u_solid` into its owner row's divergence, so
 * the solve already sees the body. This lane gates the adjoint — the pressure
 * the solve produces pushing back on the body through the same blocked face
 * area — which is the only thing that makes a body float rather than fall
 * through the water as though the tank were empty.
 *
 * Two states, one scene:
 *  - `float`: a body released at its analytic Archimedes draft must stay
 *    there. This fails in both directions: no adjoint sinks it, and an adjoint
 *    with the wrong sign or scale launches it.
 *  - `rise`: a body released fully submerged must climb toward the surface.
 *    This is the discriminating case, because gravity alone can only sink it.
 */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { octreeMethod } from "../lib/methods/octree";
import { createTinyHydrostaticScene } from "../lib/scenes";
import { initializeRigidBodies } from "../lib/rigid-body";
import { GPU_RIGID_RENDER_BYTES } from "../lib/webgpu-rigid-body";
import { fluidExecutionDeviceFeatures } from "../lib/gpu-startup";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

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
const device = await adapter.requestDevice({
  requiredFeatures: fluidExecutionDeviceFeatures(adapter.features),
  requiredLimits: requiredFluidDeviceLimits(adapter.limits),
});
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));

const release = process.env.FLUID_RIGID_RELEASE ?? "float";
if (release !== "float" && release !== "submerged") {
  throw new Error("FLUID_RIGID_RELEASE must be float or submerged");
}
const side_m = Number(process.env.FLUID_RIGID_SIDE_M ?? 0.2);
const bodyDensity_kg_m3 = Number(process.env.FLUID_RIGID_DENSITY ?? 300);
const steps = Number(process.env.FLUID_RIGID_STEPS ?? 250);

const scene = createTinyHydrostaticScene();
scene.sceneId = `smoke-octree-rigid-buoyancy-${release}`;
const surface_m = scene.container.height_m * scene.container.fillFraction;
// Archimedes: a floating body's draft displaces exactly its own mass, so a
// cube of relative density r floats with r of its side below the surface.
const equilibriumY_m = surface_m - (bodyDensity_kg_m3 / scene.fluid.density_kg_m3) * side_m + 0.5 * side_m;
const submergedY_m = 0.5 * surface_m;
const startY_m = release === "float" ? equilibriumY_m : submergedY_m;
scene.rigidBodies = [{
  id: "buoyant-box", name: "Buoyant box", shape: "box",
  dimensions_m: { x: side_m, y: side_m, z: side_m }, density_kg_m3: bodyDensity_kg_m3,
  position_m: { x: 0, y: startY_m, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
  linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
  restitution: 0, friction: 0.4, motion: "dynamic",
}];

const bodies = initializeRigidBodies(scene.rigidBodies);
// The octree publishes its t=0 sparse authority asynchronously, and the first
// substep flips a topology candidate that only exists once that has fenced.
const values = { ...octreeMethod.presetFor("balanced") };
if (process.env.FLUID_MAXIMUM_LEAF_SIZE) values.maximumLeafSize = process.env.FLUID_MAXIMUM_LEAF_SIZE;
if (process.env.FLUID_OCTREE_INTERFACE_BAND) values.interfaceRefinementBandCells = Number(process.env.FLUID_OCTREE_INTERFACE_BAND);
const solver = await octreeMethod.createSolverAsync!(
  device, scene, "balanced", values, undefined, () => {},
);
assert.equal(solver.initialSparseAuthorityReady, true,
  "octree construction must fence the complete t=0 sparse authority");

const readback = device.createBuffer({ label: "test-only resident rigid snapshot",
  size: GPU_RIGID_RENDER_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
async function bodyCentreY_m() {
  const encoder = device.createCommandEncoder({ label: "Sample resident rigid pose" });
  encoder.copyBufferToBuffer(solver.rigidRenderBuffer!, 0, readback, 0, GPU_RIGID_RENDER_BYTES);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const records = new Float32Array(readback.getMappedRange()).slice();
  readback.unmap();
  return records[1]!;
}

const dt = scene.numerics.fixedDt_s;
const trace: Array<{ step: number; y_m: number }> = [];
let minimumY_m = startY_m, maximumY_m = startY_m;
for (let step = 1; step <= steps; step += 1) {
  while (!solver.advanceTo(step * dt, bodies)) await new Promise((resolve) => setTimeout(resolve, 0));
  if (step % 25 !== 0 && step !== steps) continue;
  await device.queue.onSubmittedWorkDone();
  const y_m = await bodyCentreY_m();
  minimumY_m = Math.min(minimumY_m, y_m); maximumY_m = Math.max(maximumY_m, y_m);
  trace.push({ step, y_m });
}
await device.queue.onSubmittedWorkDone();
const info = await solver.readStats();
const finalY_m = await bodyCentreY_m();
// Resting on the tank floor is the exact signature of an absent adjoint: the
// body free-falls through water it cannot feel.
const restingY_m = 0.5 * side_m;

const result = {
  scenario: scene.sceneId, release, grid: [solver.info.nx, solver.info.ny, solver.info.nz],
  bodyDensity_kg_m3, side_m, steps, dt,
  surface_m, startY_m, equilibriumY_m, restingY_m, finalY_m,
  minimumY_m, maximumY_m, trace,
  nonFiniteVelocityCount: info.nonFiniteCount ?? 0, validationErrors,
};
console.log(JSON.stringify(result));

assert.deepEqual(validationErrors, [], `WebGPU validation errors: ${validationErrors.join("; ")}`);
assert.equal(info.nonFiniteCount ?? 0, 0, "fluid velocity must remain finite");
assert.ok(finalY_m > restingY_m + 0.5 * side_m,
  `the body settled on the tank floor at ${finalY_m.toFixed(4)} m, which is what an absent fluid reaction produces`);
if (release === "float") {
  const drift_m = Math.abs(finalY_m - equilibriumY_m);
  assert.ok(drift_m <= side_m,
    `a body released at its ${equilibriumY_m.toFixed(4)} m Archimedes draft drifted ${drift_m.toFixed(4)} m`);
} else {
  assert.ok(maximumY_m > startY_m + 0.25 * side_m,
    `a submerged body of relative density ${(bodyDensity_kg_m3 / scene.fluid.density_kg_m3).toFixed(2)} never rose above its ${startY_m.toFixed(4)} m release depth`);
}
readback.destroy(); solver.destroy(); device.destroy();
