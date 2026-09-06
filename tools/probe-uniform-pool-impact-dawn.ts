import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { uniformMethod } from "../lib/methods/uniform/method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { readFloatTexture3D, readRgbaTexture3D } from "../lib/harness/webgpu-smoke-readbacks";

async function readBuffer(device: GPUDevice, source: GPUBuffer) {
  const copy = device.createBuffer({ size: source.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, copy, 0, source.size);
    device.queue.submit([encoder.finish()]); await copy.mapAsync(GPUMapMode.READ);
    return new Float32Array(copy.getMappedRange()).slice();
  } finally { if (copy.mapState === "mapped") copy.unmap(); copy.destroy(); }
}

const modulePath = process.env.WEBGPU_NODE_MODULE;
assert.ok(modulePath, "Set WEBGPU_NODE_MODULE to the native Dawn module path");
const steps = Number(process.env.POOL_STEPS ?? 180), dt = Number(process.env.POOL_DT ?? 1 / 60);
const output = process.env.POOL_OUTPUT ?? "artifacts/pool-impact-ab/uniform-center";
const captureEvery = Number(process.env.POOL_CAPTURE_EVERY ?? 5);
const stageSteps = new Set((process.env.POOL_STAGE_STEPS ?? "1,10,20,30").split(",").map(Number));
process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT = "1";
assert.ok(Number.isSafeInteger(steps) && steps >= 0 && Number.isFinite(dt) && dt > 0);
await acquireWebGPUExclusiveLock("dawn-probe", "uniform-pool-impact-ab");
const live = new Set<GPU>();
let gpu: GPU | undefined, device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined;
try {
  const dawn = await import(pathToFileURL(modulePath).href);
  Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  live.add(gpu!);
  const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact-half"));
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
  const values = resolveMethodValues(uniformMethod, "balanced", {
    timeStep: "scene", ...JSON.parse(process.env.POOL_OVERRIDES ?? "{}"),
  });
  solver = await uniformMethod.createSolverAsync!(device, scene, "balanced", values,
    undefined, () => {}) as WebGPUUniformReferenceSolver;
  const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
  assert.deepEqual([nx, ny, nz], [64, 48, 64]);
  await mkdir(output, { recursive: true });
  const shaderHash = createHash("sha256").update(await readFile(new URL(
    "../lib/methods/uniform/webgpu-uniform-reference.wgsl.ts", import.meta.url))).digest("hex");
  await writeFile(`${output}/configuration.json`, JSON.stringify({ scene, values, grid: [nx, ny, nz],
    steps, dt, shaderHash, method: "uniform", captureEvery }, null, 2));
  const trace = [];
  const write = async (name: string, data: Float32Array) => {
    assert.ok(data.every(Number.isFinite), `${name} contains only finite values`);
    await writeFile(`${output}/${name}.bin`, new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  };
  for (let step = 0; step <= steps; step++) {
    if (step) {
      assert.ok(solver.advanceTo(step * dt, []));
      await device.queue.onSubmittedWorkDone();
      assert.equal(solver.info.encodedSteps, step);
    }
    if (stageSteps.has(step)) {
      for (const [name, texture] of Object.entries(solver.symmetryStageAuditTextures!)) {
        const scalar = name.toLowerCase().includes("density") || name.startsWith("gamma");
        await write(`${step}-stage-${name}`, await (scalar ? readFloatTexture3D : readRgbaTexture3D)(
          device, texture, nx, ny, nz));
      }
    }
    if (step % captureEvery && step !== steps && !stageSteps.has(step)) continue;
    const density = await readFloatTexture3D(device, solver.volumeTexture, nx, ny, nz);
    const mac = await readRgbaTexture3D(device, solver.velocityTexture, nx, ny, nz);
    const negativeBoundary = await readBuffer(device, solver.negativeBoundaryVelocityBuffer);
    // The separating-boundary solve owns three additional negative MAC planes.
    // Read them: an enclosed container does not imply zero during separation.
    // Cell means average paired normal faces, matching the sparse diagnostic.
    const velocity = new Float32Array(mac.length), heights = new Float32Array(nx * nz);
    let mass = 0, momentY = 0, kinetic = 0, staggeredKinetic = 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const at = x + nx * (y + ny * z), rho = density[at]!;
      mass += rho; heights[x + nx * z]! += rho; momentY += rho * (y + .5);
      for (let axis = 0; axis < 3; axis++) {
        const positive = mac[4 * at + axis]!;
        const boundaryIndex = [y + ny * z, ny * nz + x + nx * z, ny * nz + nx * nz + x + nx * y][axis]!;
        const negative = [x, y, z][axis]! > 0
          ? mac[4 * (at - [1, nx, nx * ny][axis]!) + axis]! : negativeBoundary[boundaryIndex]!;
        velocity[4 * at + axis] = .5 * (positive + negative);
        kinetic += .5 * rho * velocity[4 * at + axis]! ** 2;
        staggeredKinetic += .25 * rho * (positive ** 2 + negative ** 2);
      }
    }
    const pressurePadded = await readFloatTexture3D(device, solver.physicsFieldsForQA.pressure, nx + 2, ny + 2, nz + 2);
    const pressure = new Float32Array(nx * ny * nz);
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const at = x + nx * (y + ny * z);
      pressure[at] = density[at]! > .5
        ? pressurePadded[x + 1 + (nx + 2) * (y + 1 + (ny + 2) * (z + 1))]! : 0;
    }
    const fields = { density, velocity, mac, pressure, negativeBoundary, height: heights,
      gamma: await readFloatTexture3D(device, solver.physicsFieldsForQA.gamma, nx, ny, nz) };
    for (const [name, data] of Object.entries(fields)) await write(`${step}-${name}`, data);
    const stats = await solver.readStats();
    await writeFile(`${output}/${step}-stats.json`, JSON.stringify(stats));
    const row = { step, time: step * dt, mass, centerOfMassY: momentY / mass, kinetic,
      staggeredKinetic, pressureResidual: stats.uniformCM11aFineResidualInfinity,
      maximumNegativeBoundarySpeed_m_s: negativeBoundary.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0),
      pressureCapFailure: stats.uniformCM11aCapFailure };
    trace.push(row); console.log(JSON.stringify(row));
    assert.deepEqual(errors, []);
  }
  await writeFile(`${output}/trace.json`, JSON.stringify(trace, null, 2));
} finally {
  solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
  if (gpu) live.delete(gpu);
}
