import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

const arg = (key: string, fallback: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const arm = arg("arm", "baseline");
const frames = Number(arg("frames", "30"));
const overrides = JSON.parse(arg("values", "{}"));
const out = arg("out", `artifacts/pond-rest/${arm}.json`);
async function read(device: GPUDevice, texture: GPUTexture) {
  const c = texture.format === "rgba32float" ? 4 : 1;
  const width = texture.width * c, row = Math.ceil(width * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: row * texture.height * texture.depthOrArrayLayers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const e = device.createCommandEncoder();
    e.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row, rowsPerImage: texture.height }, [texture.width, texture.height, texture.depthOrArrayLayers]);
    device.queue.submit([e.finish()]); await buffer.mapAsync(GPUMapMode.READ);
    const src = new Float32Array(buffer.getMappedRange()), result = new Float32Array(width * texture.height * texture.depthOrArrayLayers);
    for (let z = 0; z < texture.depthOrArrayLayers; z++) for (let y = 0; y < texture.height; y++) result.set(src.subarray((z * texture.height + y) * row / 4, (z * texture.height + y) * row / 4 + width), (z * texture.height + y) * width);
    return result;
  } finally { if (buffer.mapState === "mapped") buffer.unmap(); buffer.destroy(); }
}

await acquireWebGPUExclusiveLock("dawn-probe", `uniform pond rest ${arm}`);
let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined;
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
  const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const scene = sceneDocument(getSceneDefinition("hero-garden-hose-x10"));
  scene.fluid.inflow = { ...scene.fluid.inflow!, enabled: false };
  if (arg("sigma", "scene") !== "scene") scene.fluid.surfaceTension_N_m = Number(arg("sigma", "0"));
  if (arg("gravity", "scene") !== "scene") scene.fluid.gravity_m_s2 = { x: 0, y: Number(arg("gravity", "0")), z: 0 };
  const values = resolveMethodValues(uniformVolumeMethod, "balanced", { pressureResidualTolerance: 0, ...overrides });
  solver = await uniformVolumeMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  const { nx, ny, nz } = solver.info, h = scene.container.height_m / ny;
  const open = await read(device, solver.denseLevelSetVolumeSource!.openFraction);
  const level = scene.container.height_m * scene.container.fillFraction / h;
  const surfaceY = Math.floor(level);
  const columns: { x: number; z: number; interior: boolean }[] = [];
  for (let z = 3; z < nz - 3; z++) for (let x = 3; x < nx - 3; x++) {
    if (open[x + nx * (surfaceY - 1 + ny * z)]! < 0.999) continue;
    let interior = true;
    for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) if (open[x + dx + nx * (surfaceY - 2 + ny * (z + dz))]! < 0.999) interior = false;
    columns.push({ x, z, interior });
  }
  assert.ok(columns.some(c => c.interior));
  const samples: unknown[] = [];
  for (let frame = 0; frame <= frames; frame++) {
    if (frame) { assert.ok(solver.advanceTo(frame / 30, [])); await solver.awaitFrameCompletion(); }
    const stats = await solver.readStats();
    if (frame > 3 && frame % 10 && frame !== frames) continue;
    const phi = await read(device, solver.vertexPhiTexture!), velocity = await read(device, solver.velocityTexture), volume = await read(device, solver.volumeTexture);
    const regions = { all: [] as number[], interior: [] as number[] };
    let missing = 0;
    for (const c of columns) {
      let height: number | undefined;
      for (let y = Math.max(0, surfaceY - 3); y < Math.min(ny, surfaceY + 4); y++) {
        const a = phi[c.x + (nx + 1) * (y + (ny + 1) * c.z)]!, b = phi[c.x + (nx + 1) * (y + 1 + (ny + 1) * c.z)]!;
        if (a <= 0 && b > 0) height = y - a / (b - a);
      }
      if (height === undefined) { missing++; continue; }
      regions.all.push((height - level) * h * 1000);
      if (c.interior) regions.interior.push((height - level) * h * 1000);
    }
    const surface = Object.fromEntries(Object.entries(regions).map(([key, a]) => [key, { count: a.length, mean_mm: a.reduce((s, v) => s + v, 0) / a.length, rms_mm: Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length), max_mm: Math.max(...a.map(Math.abs)), range_mm: Math.max(...a) - Math.min(...a) }]));
    let maxSpeed = 0, excess = 0, sum = 0, maxCell = 0;
    for (let i = 0; i < volume.length; i++) { const v = volume[i]!; assert.ok(Number.isFinite(v)); sum += v; excess += Math.max(0, v - open[i]!); if (v > 1e-5) { const speed = Math.hypot(velocity[i * 4]!, velocity[i * 4 + 1]!, velocity[i * 4 + 2]!); if (speed > maxSpeed) { maxSpeed = speed; maxCell = i; } } }
    const sample = { frame, surface, missing, maxSpeed, maxCell: [maxCell % nx, Math.floor(maxCell / nx) % ny, Math.floor(maxCell / nx / ny)], sum, excess, residual: stats.uniformCM11aFineResidualInfinity, acceptedResidual: stats.uniformPressureAcceptedResidual, recovery: stats.uniformPressureRecoverySweeps, full: stats.uniformCM11aFullCyclesExecuted, vcycles: stats.uniformCM11aVCyclesExecuted };
    if (arg("dump", "off") === "on" && frame === 1) {
      const cells = [];
      for (let z = sample.maxCell[2]! - 1; z <= sample.maxCell[2]! + 1; z++) for (let y = surfaceY - 2; y <= surfaceY + 2; y++) for (let x = sample.maxCell[0]! - 1; x <= sample.maxCell[0]! + 1; x++) {
        const i = x + nx * (y + ny * z);
        cells.push({ p: [x,y,z], open: open[i], volume: volume[i], velocity: [...velocity.subarray(4*i,4*i+4)], phi: Array.from({length:8}, (_, k) => phi[x+(k&1)+(nx+1)*(y+((k>>1)&1)+(ny+1)*(z+((k>>2)&1)))]) });
      }
      writeFileSync(`artifacts/pond-rest/${arm}-cells.json`, JSON.stringify(cells,null,2));
    }
    samples.push(sample); console.log(JSON.stringify({ arm, ...sample }));
  }
  assert.deepEqual(errors, []);
  mkdirSync(resolve(out, ".."), { recursive: true });
  writeFileSync(out, JSON.stringify({ arm, dimensions: [nx, ny, nz], level, h, values, sigma: scene.fluid.surfaceTension_N_m, gravity: scene.fluid.gravity_m_s2, samples }, null, 2));
} finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
