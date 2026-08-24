import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
// Composition root for this entry point: importing the method catalog installs
// the simulation methods and the octree coarse-dynamics lanes, without which
// constructing a solver throws rather than silently running the wrong backend.
import "../lib/methods";
import { fluidExecutionDeviceFeatures } from "../lib/core/gpu-startup";
import { uniformMethod } from "../lib/methods/uniform/method";
import { cloneScene, defaultScene, type RigidBodyDescription } from "../lib/core/model";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { boxTankWallFieldForScene } from "../lib/core/scene-lattice";

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

const scene = cloneScene(defaultScene);
scene.sceneId = "smoke-uniform-rigid-displacement";
scene.container = {
  ...scene.container,
  width_m: 0.8,
  height_m: 0.8,
  depth_m: 0.8,
  fillFraction: 0.5,
  top: "open",
  fluidWallMode: "free-slip",
};
scene.fluid.initialCondition = "tank-fill";
scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
scene.fluid.surfaceTension_N_m = 0;
scene.fluid.dynamicViscosity_Pa_s = 0;
delete scene.fluid.inflow;
delete scene.terrain;
const dt_s = Number(process.env.FLUID_DISPLACEMENT_DT ?? 0.004);
scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt_s;
scene.voxelDomain.finestCellSize_m = 0.05;
scene.container.wallField = boxTankWallFieldForScene(scene);

const bodyCount = Math.max(1, Math.min(2, Number(process.env.FLUID_DISPLACEMENT_BODY_COUNT ?? 1)));
const startY_m = Number(process.env.FLUID_DISPLACEMENT_START_Y ?? 0.65);
const submergedY_m = Number(process.env.FLUID_DISPLACEMENT_END_Y ?? 0.30);
const bodyDimensions_m = { x: 0.30, y: 0.20, z: 0.30 };
const bodyAt = (index: number): RigidBodyDescription => ({
  id: `displacement-box-${index}`,
  name: `Displacement box ${index}`,
  shape: "box",
  dimensions_m: bodyDimensions_m,
  density_kg_m3: 1000,
  position_m: { x: bodyCount === 1 ? 0 : (index === 0 ? -0.19 : 0.19), y: startY_m, z: 0 },
  orientation: { w: 1, x: 0, y: 0, z: 0 },
  linearVelocity_m_s: { x: 0, y: 0, z: 0 },
  angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
  restitution: 0,
  friction: 0,
  motion: "static",
});
scene.rigidBodies = Array.from({ length: bodyCount }, (_, index) => bodyAt(index));

const values = {
  ...uniformMethod.presetFor("balanced"),
  timeStep: "scene",
  rigidCoupling: "off",
  densityPostProcessing: "off",
};
const solver = await uniformMethod.createSolverAsync!(
  device,
  scene,
  "balanced",
  values,
  undefined,
  () => {},
);
const bodies = initializeRigidBodies(scene.rigidBodies);

async function readTexture(texture: GPUTexture, nx: number, ny: number, nz: number) {
  const rowBytes = nx * 4;
  const pitch = Math.ceil(rowBytes / 256) * 256;
  const buffer = device.createBuffer({
    size: pitch * ny * nz,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture },
    { buffer, bytesPerRow: pitch, rowsPerImage: ny },
    [nx, ny, nz],
  );
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const padded = new Float32Array(buffer.getMappedRange());
  const result = new Float32Array(nx * ny * nz);
  const stride = pitch / 4;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    result.set(
      padded.subarray(stride * (y + ny * z), stride * (y + ny * z) + nx),
      nx * (y + ny * z),
    );
  }
  buffer.unmap();
  buffer.destroy();
  return result;
}

const sum = (field: Float32Array) => field.reduce((total, value) => total + value, 0);
const { nx, ny, nz } = solver.info;
const initialMass_cells = sum(await readTexture(solver.volumeTexture, nx, ny, nz));
const descentSteps = Number(process.env.FLUID_DISPLACEMENT_DESCENT_STEPS ?? 56);
const settleSteps = Number(process.env.FLUID_DISPLACEMENT_SETTLE_STEPS ?? 8);
let maximumRelativeMassLoss = 0;
let maximumRelativeRepresentedVolumeLoss = 0;
let maximumUnplaceable_cells = 0;
let finalRepresentedVolume_cells = initialMass_cells;

for (let step = 1; step <= descentSteps + settleSteps; step += 1) {
  const fraction = Math.min(1, step / descentSteps);
  for (const body of bodies) {
    const previousY = body.position_m.y;
    body.position_m.y = startY_m + fraction * (submergedY_m - startY_m);
    body.linearVelocity_m_s.y = step <= descentSteps
      ? (body.position_m.y - previousY) / scene.numerics.maxDt_s
      : 0;
  }
  while (!solver.advanceTo(step * scene.numerics.maxDt_s, bodies)) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  await device.queue.onSubmittedWorkDone();
  const stats = await solver.readStats();
  // Sec. 3.6 unplaceable-excess telemetry is stamped onto `info` by the uniform
  // reference solver (webgpu-uniform-reference.ts) and is not part of the shared
  // GPUEulerianInfo shape, so read it through a narrow structural view.
  const uniformDiagnostics = stats as typeof stats & {
    uniformUnplaceableSolidExcess_cells?: number;
  };
  const mass_cells = Number(stats.volumeCellSum);
  maximumRelativeMassLoss = Math.max(maximumRelativeMassLoss,
    Math.max(0, (initialMass_cells - mass_cells) / initialMass_cells));
  finalRepresentedVolume_cells = Number(stats.representedVolumeCellSum);
  maximumRelativeRepresentedVolumeLoss = Math.max(maximumRelativeRepresentedVolumeLoss,
    Math.max(0, (initialMass_cells - finalRepresentedVolume_cells) / initialMass_cells));
  maximumUnplaceable_cells = Math.max(maximumUnplaceable_cells,
    Number(uniformDiagnostics.uniformUnplaceableSolidExcess_cells ?? 0));
  if (process.env.FLUID_DISPLACEMENT_LOG_STEPS === "1") {
    console.log(JSON.stringify({ step, fraction, mass_cells,
      relativeMassLoss: (initialMass_cells - mass_cells) / initialMass_cells,
      unplaceable_cells: uniformDiagnostics.uniformUnplaceableSolidExcess_cells ?? 0 }));
  }
}

const finalField = await readTexture(solver.volumeTexture, nx, ny, nz);
const finalMass_cells = sum(finalField);
const finalExcessDensity_cells = finalField.reduce((total, value) => total + Math.max(0, value - 1), 0);
const maximumDensity = finalField.reduce((maximum, value) => Math.max(maximum, value), 0);
const finalRelativeMassLoss = Math.max(0, (initialMass_cells - finalMass_cells) / initialMass_cells);
const result = {
  scenario: scene.sceneId,
  method: uniformMethod.id,
  grid: [nx, ny, nz],
  bodyCount,
  descentSteps,
  settleSteps,
  initialMass_cells,
  finalMass_cells,
  finalRelativeMassLoss,
  finalExcessDensity_cells,
  maximumDensity,
  maximumRelativeMassLoss,
  finalRepresentedVolume_cells,
  maximumRelativeRepresentedVolumeLoss,
  maximumUnplaceable_cells,
  validationErrors,
};
console.log(JSON.stringify(result));

assert.deepEqual(validationErrors, [], `WebGPU validation errors: ${validationErrors.join("; ")}`);
assert.ok(maximumUnplaceable_cells < 1 / 2048,
  `solid entry discarded ${maximumUnplaceable_cells.toFixed(4)} cells of density`);
assert.ok(finalRelativeMassLoss < 1e-3,
  `uniform liquid mass fell by ${(100 * finalRelativeMassLoss).toFixed(3)}% during solid entry`);
assert.ok(maximumRelativeMassLoss < 1e-3,
  `uniform liquid mass transiently fell by ${(100 * maximumRelativeMassLoss).toFixed(3)}% during solid entry`);
// CM12 intentionally allows rho'>1 during impacts and rate-limits its Sec.
// 3.7 expansion, so clamped represented volume can lag conserved mass by a
// small transient amount. The pre-fix defect lost 25.8% outright; two percent
// is a strict visual-volume gate without pretending the paper enforces
// pointwise rho<=1 in ordinary (V=1) cells.
assert.ok(maximumRelativeRepresentedVolumeLoss < 0.02,
  `uniform represented volume fell by ${(100 * maximumRelativeRepresentedVolumeLoss).toFixed(3)}% during solid entry`);

solver.destroy();
device.destroy();
