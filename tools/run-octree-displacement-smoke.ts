import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { octreeMethod } from "../lib/methods/octree";
import { cloneScene, defaultScene } from "../lib/model";
import { initializeRigidBodies } from "../lib/rigid-body";
import type { GPURigidLoad } from "../lib/webgpu-eulerian";

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
scene.sceneId = "smoke-octree-large-solid-submersion";
scene.container = { width_m: 0.6, height_m: 0.6, depth_m: 0.6, fillFraction: 0.45, top: "open", fluidWallMode: "free-slip" };
scene.fluid.initialCondition = "tank-fill";
scene.fluid.gravity_m_s2 = { x: 0, y: -9.80665, z: 0 };
scene.fluid.surfaceTension_N_m = 0;
delete scene.fluid.inflow;
delete scene.terrain;
scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
const bodyId = "large-displacement-box";
const bodyDimensions_m = { x: 0.3, y: 0.16, z: 0.3 };
const startY_m = 0.46, submergedY_m = 0.14;
scene.rigidBodies = [{
  id: bodyId, name: "Large displacement box", shape: "box",
  dimensions_m: bodyDimensions_m, density_kg_m3: 1000,
  position_m: { x: 0, y: startY_m, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
  linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
  restitution: 0, friction: 0, motion: "static"
}];

let latestLoad: GPURigidLoad | undefined;
scene.voxelDomain.finestCellSize_m = Math.sqrt(scene.container.width_m * scene.container.depth_m / 1600);
const values = { ...octreeMethod.presetFor("balanced"), pressureIterations: 64 };
const solver = octreeMethod.createSolver!(device, scene, "balanced", values, (loads) => {
  latestLoad = loads.find((load) => load.bodyId === bodyId) ?? latestLoad;
});
const bodies = initializeRigidBodies(scene.rigidBodies);

async function readTexture(texture: GPUTexture, nx: number, ny: number, nz: number) {
  const rowBytes = nx * 4, pitch = Math.ceil(rowBytes / 256) * 256;
  const buffer = device.createBuffer({ size: pitch * ny * nz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: pitch, rowsPerImage: ny }, [nx, ny, nz]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const padded = new Float32Array(buffer.getMappedRange()), result = new Float32Array(nx * ny * nz), stride = pitch / 4;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    result.set(padded.subarray(stride * (y + ny * z), stride * (y + ny * z) + nx), nx * (y + ny * z));
  }
  buffer.unmap(); buffer.destroy();
  return result;
}

const { nx, ny, nz } = solver.info;
const h = { x: scene.container.width_m / nx, y: scene.container.height_m / ny, z: scene.container.depth_m / nz };
const cellVolume_m3 = h.x * h.y * h.z;
const occupancy = (phi: number) => Math.max(0, Math.min(1, 0.5 - phi / (4 * h.y)));
const initialPhi = await readTexture(solver.surfaceFieldTexture!, nx, ny, nz);
const referenceOpenCells = initialPhi.reduce((sum, phi) => sum + occupancy(phi), 0);

const descentSteps = Number(process.env.FLUID_DISPLACEMENT_DESCENT_STEPS ?? 96);
const settleSteps = Number(process.env.FLUID_DISPLACEMENT_SETTLE_STEPS ?? 32);
const steps = descentSteps + settleSteps;
for (let step = 1; step <= steps; step += 1) {
  const previousY = bodies[0].position_m.y;
  const fraction = Math.min(1, step / descentSteps);
  bodies[0].position_m.y = startY_m + fraction * (submergedY_m - startY_m);
  bodies[0].linearVelocity_m_s.y = (bodies[0].position_m.y - previousY) / scene.numerics.maxDt_s;
  if (step > descentSteps) bodies[0].linearVelocity_m_s.y = 0;
  while (!solver.advanceTo(step * scene.numerics.maxDt_s, bodies)) await new Promise((resolve) => setTimeout(resolve, 0));
  await device.queue.onSubmittedWorkDone();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

const finalPhi = await readTexture(solver.surfaceFieldTexture!, nx, ny, nz);
let rawCells = 0, openCells = 0, displacedCells = 0;
for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
  const index = x + nx * (y + ny * z), alpha = occupancy(finalPhi[index]);
  const center = {
    x: -0.5 * scene.container.width_m + (x + 0.5) * h.x,
    y: (y + 0.5) * h.y,
    z: -0.5 * scene.container.depth_m + (z + 0.5) * h.z
  };
  let insideCorners = 0;
  for (let corner = 0; corner < 8; corner += 1) {
    const dx = center.x + ((corner & 1) ? 0.4 : -0.4) * h.x - bodies[0].position_m.x;
    const dy = center.y + ((corner & 2) ? 0.4 : -0.4) * h.y - bodies[0].position_m.y;
    const dz = center.z + ((corner & 4) ? 0.4 : -0.4) * h.z - bodies[0].position_m.z;
    if (Math.abs(dx) <= 0.5 * bodyDimensions_m.x && Math.abs(dy) <= 0.5 * bodyDimensions_m.y && Math.abs(dz) <= 0.5 * bodyDimensions_m.z) insideCorners += 1;
  }
  const solid = insideCorners / 8;
  rawCells += alpha; openCells += alpha * (1 - solid); displacedCells += alpha * solid;
}

const analyticDisplacement_m3 = bodyDimensions_m.x * bodyDimensions_m.y * bodyDimensions_m.z;
const measuredDisplacement_m3 = displacedCells * cellVolume_m3;
const reportedDisplacement_m3 = latestLoad?.displacedVolume_m3 ?? 0;
const openVolumeError = Math.abs(openCells - referenceOpenCells) / referenceOpenCells;
const complementError = Math.abs((rawCells - referenceOpenCells) - displacedCells) / Math.max(displacedCells, 1);
const geometricError = Math.abs(measuredDisplacement_m3 - analyticDisplacement_m3) / analyticDisplacement_m3;
const reportingError = Math.abs(reportedDisplacement_m3 - measuredDisplacement_m3) / Math.max(measuredDisplacement_m3, Number.EPSILON);
const result = {
  scenario: scene.sceneId, method: octreeMethod.id, grid: [nx, ny, nz], descentSteps, settleSteps,
  solidToInitialWaterRatio: analyticDisplacement_m3 / (referenceOpenCells * cellVolume_m3),
  referenceOpenVolume_m3: referenceOpenCells * cellVolume_m3,
  finalOpenVolume_m3: openCells * cellVolume_m3,
  rawSurfaceVolumeIncrease_m3: (rawCells - referenceOpenCells) * cellVolume_m3,
  measuredDisplacement_m3, analyticDisplacement_m3, reportedDisplacement_m3,
  errors: { openVolumeError, complementError, geometricError, reportingError }, validationErrors
};
console.log(JSON.stringify(result));

assert.deepEqual(validationErrors, [], `WebGPU validation errors: ${validationErrors.join("; ")}`);
assert.ok(analyticDisplacement_m3 / (referenceOpenCells * cellVolume_m3) > 0.1, "regression solid must displace more than 10% of the initial water volume");
assert.ok(openVolumeError < 0.01, `open-liquid volume drifted by ${(100 * openVolumeError).toFixed(2)}%`);
assert.ok(complementError < 0.1, `surface rise differs from displaced volume by ${(100 * complementError).toFixed(2)}%`);
assert.ok(geometricError < 0.05, `voxelized box volume differs from analytic volume by ${(100 * geometricError).toFixed(2)}%`);
assert.ok(reportingError < 0.01, `rigid-load displacement differs from the surface complement by ${(100 * reportingError).toFixed(2)}%`);

solver.destroy(); device.destroy();
