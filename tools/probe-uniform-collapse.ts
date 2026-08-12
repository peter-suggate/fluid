/**
 * Dawn diagnostic probe for uniform (Chentanez-Mueller 2012) surface
 * transport. Reads back raw/render density, gamma, and the MAC velocity every
 * few steps and prints height profiles so a frozen or disappearing surface
 * can be attributed to transport, sharpening, projection, or presentation.
 *
 * Usage:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/run-webgpu-exclusive.ts \
 *     --import tsx tools/probe-uniform-collapse.ts
 *
 * Environment:
 *   FLUID_UNIFORM_COLLAPSE_SCENE       scene preset (default symmetric-expansion)
 *   FLUID_UNIFORM_COLLAPSE_DT          exact time step (default 0.004)
 *   FLUID_UNIFORM_COLLAPSE_WORLD_SCALE world scale (default 1)
 *   FLUID_UNIFORM_COLLAPSE_SHARPENING  on/off (default on)
 *   FLUID_UNIFORM_COLLAPSE_POSTPROCESS  on/off (default off)
 *   FLUID_UNIFORM_COLLAPSE_STEPS       steps to advance (default 250)
 *   FLUID_UNIFORM_COLLAPSE_CHECKPOINT  checkpoint cadence (default 10)
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

import { uniformMethod } from "../lib/methods/uniform";
import type { GPUSolverInstance } from "../lib/methods/types";
import { scaleScene } from "../lib/scene-scale";
import { getScenePreset } from "../lib/scenes";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

const steps = Number(process.env.FLUID_UNIFORM_COLLAPSE_STEPS ?? 250);
const checkpoint = Number(process.env.FLUID_UNIFORM_COLLAPSE_CHECKPOINT ?? 10);
assert.ok(Number.isSafeInteger(steps) && steps >= 1);
assert.ok(Number.isSafeInteger(checkpoint) && checkpoint >= 1);

const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([
  `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
  ...(process.env.FLUID_WEBGPU_ADAPTER
    ? [`adapter=${process.env.FLUID_WEBGPU_ADAPTER}`] : []),
]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter, "WebGPU did not expose an adapter");
const requiredFeatures: GPUFeatureName[] = ["subgroups"];
if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
const device = await adapter.requestDevice({
  requiredFeatures,
  requiredLimits: requiredFluidDeviceLimits(adapter.limits),
});
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => {
  validationErrors.push(event.error.message);
});

const sceneId = process.env.FLUID_UNIFORM_COLLAPSE_SCENE ?? "symmetric-expansion";
const worldScale = Number(process.env.FLUID_UNIFORM_COLLAPSE_WORLD_SCALE ?? 1);
let scene = getScenePreset(sceneId).create();
for (let scale = 1; scale < worldScale; scale *= 2) {
  const scaled = scaleScene(scene, "world", 2);
  assert.ok(scaled, `scene ${sceneId} cannot be world-scaled to ${2 * scale}x`);
  scene = scaled;
}
const dt = Number(process.env.FLUID_UNIFORM_COLLAPSE_DT ?? 0.004);
const sharpening = process.env.FLUID_UNIFORM_COLLAPSE_SHARPENING !== "off";
const postprocess = process.env.FLUID_UNIFORM_COLLAPSE_POSTPROCESS === "on";
scene.numerics.fixedDt_s = dt;
scene.numerics.maxDt_s = dt;
const solver = await uniformMethod.createSolverAsync!(device, scene, "balanced",
  { densityPostProcessing: postprocess ? "on" : "off",
    densitySharpening: sharpening ? "on" : "off", timeStep: "scene" },
  undefined, () => {}) as GPUSolverInstance;
const nx = solver.info.nx, ny = solver.info.ny, nz = solver.info.nz;
console.log(`scene ${sceneId} worldScale=${worldScale} grid ${nx}x${ny}x${nz} dt=${dt}`
  + ` sharpening=${sharpening ? "on" : "off"} postprocess=${postprocess ? "on" : "off"}`);

const internal = solver as unknown as {
  volumeA: GPUTexture; velocityA: GPUTexture; gammaA: GPUTexture; surfaceFieldTexture: GPUTexture;
  symmetryStageAuditTextures?: {
    previousRawDensity: GPUTexture;
    densityAdvection: GPUTexture;
    densityDiffusion: GPUTexture;
    densitySharpening: GPUTexture;
  };
};

async function readTexture(texture: GPUTexture, components: number): Promise<Float32Array> {
  const rowBytes = nx * 4 * components;
  const padded = Math.ceil(rowBytes / 256) * 256;
  const staging = device.createBuffer({
    label: "probe readback", size: padded * ny * nz,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture },
    { buffer: staging, bytesPerRow: padded, rowsPerImage: ny },
    { width: nx, height: ny, depthOrArrayLayers: nz });
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const raw = new Uint8Array(staging.getMappedRange());
  const values = new Float32Array(nx * ny * nz * components);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    const row = new Float32Array(raw.buffer, raw.byteOffset + padded * (y + ny * z), nx * components);
    values.set(row, nx * components * (y + ny * z));
  }
  staging.unmap(); staging.destroy();
  return values;
}

const at = (field: Float32Array, x: number, y: number, z: number, components = 1, component = 0) =>
  field[components * (x + nx * (y + ny * z)) + component]!;

const format = (value: number) => value.toFixed(2).padStart(6);

const capture = async (step: number) => {
  await device.queue.onSubmittedWorkDone();
  const [rho, velocity, gamma, renderDensity] = await Promise.all([
    readTexture(internal.volumeA, 1),
    readTexture(internal.velocityA, 4),
    readTexture(internal.gammaA, 1),
    readTexture(internal.surfaceFieldTexture, 1),
  ]);
  let mass = 0, maxRho = 0, over1 = 0, wet = 0, maxWetY = -1;
  let topMass = 0, topWet = 0, topVerticalMomentum = 0, topTangentialMomentum = 0;
  let lidLayerMass = 0, lidFaceMomentum = 0, lidLowerFaceMomentum = 0;
  let renderWet = 0, renderCeilingWet = 0;
  let minGamma = Infinity, maxGamma = -Infinity;
  const layerMass: number[] = Array.from({ length: ny }, () => 0);
  const layerWet: number[] = Array.from({ length: ny }, () => 0);
  const layerGamma: number[] = Array.from({ length: ny }, () => 0);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const value = at(rho, x, y, z);
    const renderValue = at(renderDensity, x, y, z);
    const gammaValue = at(gamma, x, y, z);
    mass += value; layerMass[y]! += value;
    layerGamma[y]! += gammaValue;
    minGamma = Math.min(minGamma, gammaValue); maxGamma = Math.max(maxGamma, gammaValue);
    maxRho = Math.max(maxRho, value);
    if (value > 1 + 1e-4) over1 += 1;
    if (value >= 0.5) { wet += 1; layerWet[y]! += 1; maxWetY = Math.max(maxWetY, y); }
    if (renderValue >= 0.5) {
      renderWet += 1;
      if (y >= ny - 4) renderCeilingWet += 1;
    }
    if (y >= ny - 4) {
      const vyBelow = y > 0 ? at(velocity, x, y - 1, z, 4, 1) : 0;
      const vyAbove = at(velocity, x, y, z, 4, 1);
      const vx = at(velocity, x, y, z, 4, 0);
      const vz = at(velocity, x, y, z, 4, 2);
      topMass += value;
      if (value >= 0.5) topWet += 1;
      topVerticalMomentum += value * 0.5 * (vyBelow + vyAbove);
      topTangentialMomentum += value * Math.hypot(vx, vz);
    }
    if (y === ny - 1) {
      lidLayerMass += value;
      lidFaceMomentum += value * at(velocity, x, y, z, 4, 1);
      lidLowerFaceMomentum += value * at(velocity, x, y - 1, z, 4, 1);
    }
  }
  const cx = nx >> 1, cz = nz >> 1;
  const centerRho = Array.from({ length: ny }, (_, y) => at(rho, cx, y, cz));
  const centerVy = Array.from({ length: ny }, (_, y) => at(velocity, cx, y, cz, 4, 1));
  const centerGamma = Array.from({ length: ny }, (_, y) => at(gamma, cx, y, cz));
  console.log(`step ${String(step).padStart(3)} t=${(step * dt).toFixed(3)}s`
    + ` mass=${mass.toFixed(1)} wet=${wet} maxRho=${maxRho.toFixed(3)} over1=${over1}`
    + ` gamma=[${minGamma.toFixed(3)},${maxGamma.toFixed(3)}] maxWetY=${maxWetY}`);
  console.log(`  ceiling4 mass=${topMass.toFixed(1)} wet=${topWet}`
    + ` meanVy=${(topVerticalMomentum / Math.max(topMass, 1e-9)).toFixed(3)}`
    + ` meanVxz=${(topTangentialMomentum / Math.max(topMass, 1e-9)).toFixed(3)}`
    + ` lidMass=${lidLayerMass.toFixed(1)}`
    + ` lidV=[${(lidLowerFaceMomentum / Math.max(lidLayerMass, 1e-9)).toFixed(3)},`
    + `${(lidFaceMomentum / Math.max(lidLayerMass, 1e-9)).toFixed(3)}]`);
  console.log(`  render    wet=${renderWet} ceiling4Wet=${renderCeilingWet}`);
  console.log(`  layerMass ${layerMass.map((value) => value.toFixed(0).padStart(5)).join("")}`);
  console.log(`  layerWet  ${layerWet.map((value) => String(value).padStart(5)).join("")}`);
  console.log(`  layerGam  ${layerGamma.map((value) => value.toFixed(0).padStart(5)).join("")}`);
  console.log(`  ctrRho    ${centerRho.map(format).join("")}`);
  console.log(`  ctrVy     ${centerVy.map(format).join("")}`);
  console.log(`  ctrGamma  ${centerGamma.map(format).join("")}`);
  const audit = internal.symmetryStageAuditTextures;
  if (audit) {
    const stageSummary = async (label: string, texture: GPUTexture) => {
      const field = await readTexture(texture, 1);
      let maximum = 0, sum = 0, halo = 0, partialLiquid = 0, full = 0;
      for (const value of field) {
        maximum = Math.max(maximum, value); sum += value;
        if (value > 0.01 && value < 0.5) halo += 1;
        else if (value >= 0.5 && value < 0.95) partialLiquid += 1;
        else if (value >= 0.95) full += 1;
      }
      console.log(`    ${label.padEnd(10)} max=${maximum.toFixed(3)}`
        + ` sum=${sum.toFixed(1)} halo(0.01-0.5)=${halo}`
        + ` part(0.5-0.95)=${partialLiquid} full(>=0.95)=${full}`);
    };
    await stageSummary("preStep", audit.previousRawDensity);
    await stageSummary("advected", audit.densityAdvection);
    await stageSummary("diffused", audit.densityDiffusion);
    await stageSummary("sharpened", audit.densitySharpening);
  }
};

await capture(0);
for (let step = 1; step <= steps; step += 1) {
  while (!solver.advanceTo(step * dt, [])) {
    await new Promise((resolve) => setImmediate(resolve));
  }
  if (step % checkpoint === 0 || step === steps) await capture(step);
}
if (validationErrors.length > 0) {
  console.error(`validation errors: ${validationErrors.join(" | ")}`);
  process.exitCode = 1;
}
solver.destroy();
await device.queue.onSubmittedWorkDone();
device.destroy();
console.log("probe complete");
