/** Static manufactured geometry: expose cell-spaced normals without wave physics. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { cloneScene, defaultScene } from "../lib/core/model";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, readWebGPUExclusiveLockHolder, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { sampleCoarseBowlVolumeKernel } from "./coarse-surface-volume-kernel";
import { readPublishedCM12Field } from "./sparse-cm12-published-field";

const arg = (key: string, fallback: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const arm = arg("arm", "fixed4"), profile = arg("profile", "bowl");
assert.ok(["fixed1", "fixed2", "fixed4", "adaptive", "mixed"].includes(arm));
assert.ok(["bowl", "flat", "tilt"].includes(profile));
const output = arg("output", `artifacts/coarse-surface-grid-imprint/${profile}-${arm}`);
const nx = 48, ny = 32, nz = 40, h = .05;
const gravity = Number(arg("gravity", "0")), dt = Number(arg("dt", "1e-8"));
const steps = Number(arg("steps", "1")), conditioning = arg("conditioning", "off") === "on";
assert.ok(gravity >= 0 && dt > 0 && Number.isInteger(steps) && steps > 0);
const stationary = gravity === 0 && !conditioning;
const phase = Number(arg("phase", "0"));
assert.ok(Number.isFinite(phase));
const height = (x: number, z: number) => 17.3 + (profile === "flat" ? 0 : profile === "tilt"
  ? .03 * (x - nx / 2 - phase) + .02 * (z - nz / 2)
  : .003 * ((x - nx / 2 - phase) ** 2 + .7 * (z - nz / 2) ** 2));
const topology = (a: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>>) => ({
  generation: a.acceptedTopologyGeneration,
  leaves: a.bricks.map(b => [b.leafId, b.coordinate, b.spanBricks, b.active, b.acceptedResolution]),
});
const liveGPU = new Set<GPU>();
for (let attempt = 0; ; attempt++) {
  try { await acquireWebGPUExclusiveLock("dawn-probe", `surface-grid-imprint:${arm}:${profile}`); break; }
  catch (error) {
    const holder = await readWebGPUExclusiveLockHolder();
    if (!process.argv.includes("--wait") || attempt >= 120 || !holder?.alive) throw error;
    if (attempt === 0) console.log(`Waiting for ${holder.description}`);
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}
let gpu: GPU | undefined, device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url))).href);
  Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  liveGPU.add(gpu!);
  const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [], shaderHashes: Record<string, string> = {};
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const compile = device.createShaderModule.bind(device);
  device.createShaderModule = descriptor => {
    shaderHashes[descriptor.label ?? `shader-${Object.keys(shaderHashes).length}`] = createHash("sha256").update(descriptor.code).digest("hex");
    return compile(descriptor);
  };
  const scene = cloneScene(defaultScene);
  scene.rigidBodies = []; scene.solidVoxels = []; delete scene.terrain;
  scene.container = { ...scene.container, width_m: nx * h, height_m: ny * h, depth_m: nz * h,
    fillFraction: 17.3 / ny, top: "closed", fluidWallMode: "free-slip", depthBoundary: "closed" };
  scene.voxelDomain = { finestCellSize_m: h, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  delete scene.fluid.initialBrickSeeds_m; delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.initialDamBreakDimensions_m; delete scene.fluid.inflow;
  scene.fluid.gravity_m_s2 = { x: 0, y: -gravity, z: 0 };
  scene.fluid.surfaceTension_N_m = 0; scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
  scene.solidVoxels = [...solidVoxelShellForScene(scene)];
  const region = (width: number, left = -nx * h / 2, right = nx * h / 2) => ({
    id: `fixture-${left}`, rule: "minimum-cell-size" as const,
    minimumCellSize_cells: width, maximumCellSize_cells: width,
    min_m: { x: left, y: 0, z: -nz * h / 2 }, max_m: { x: right, y: ny * h, z: nz * h / 2 },
  });
  const width = Number(arm.replace("fixed", ""));
  scene.fluid.refinementRegions = arm.startsWith("fixed") ? [region(width)]
    : arm === "mixed" ? [region(2, -nx * h / 2, 0), region(4, 0, nx * h / 2)] : [];
  const options = adaptiveMassSolverOptions({ selectorMode: "coarse-first", timeStep: "scene" });
  solver = await WebGPUAdaptiveMassSolver.createAsync(device, scene, "balanced", undefined, {
    ...options, maximumMacroSpanBricks: 1, topologyPageBudget: 0,
    gammaDiffusionEnabled: conditioning, surfaceSharpeningEnabled: conditioning,
  }, () => {});
  await solver.waitForSimulationReady();
  assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [nx, ny, nz]);
  // All arms restrict the SAME fine-volume quadrature; no per-arm rasterizer.
  const fine = new Float64Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    for (let iz = 0; iz < 8; iz++) for (let ix = 0; ix < 8; ix++) {
      const top = height(x + (ix + .5) / 8, z + (iz + .5) / 8);
      for (let y = 0; y < ny; y++) fine[x + nx * (y + ny * z)]! += Math.max(0, Math.min(1, top - y)) / 64;
    }
  }
  const source = solver.fieldSnapshotSourceForQA, words = source.templateWords;
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const density = new Float32Array(source.cellCapacity);
  for (let id = 0; id < words[2]!; id++) {
    const at = words[6]! + 8 * id;
    const span = [floats[at + 4]!, floats[at + 5]!, floats[at + 6]!];
    const low = [0, 1, 2].map(a => Math.round(floats[at + a]! - span[a]! / 2));
    let sum = 0;
    for (let z = low[2]!; z < low[2]! + span[2]!; z++) for (let y = low[1]!; y < low[1]! + span[1]!; y++) for (let x = low[0]!; x < low[0]! + span[0]!; x++) {
      if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) sum += fine[x + nx * (y + ny * z)]!;
    }
    density[id] = sum / floats[at + 3]!;
  }
  for (const offset of [source.layout.densityA, source.layout.densityB]) device.queue.writeBuffer(source.state, 4 * offset, density);
  solver.setTopologyFrozen(true);
  const initial = await solver.readDiagnosticFields(true);
  const initialActivity = await solver.readGPUActivityPolicy();
  if (arm.startsWith("fixed")) assert.ok(initialActivity.bricks.every(b => !b.active || b.acceptedResolution === 8 / width));
  await mkdir(output, { recursive: true });
  for (let step = 1; step <= steps; step++) {
    while (!solver.advanceTo(step * dt, [])) await new Promise(setImmediate);
    await solver.waitForTopologyReady();
    assert.equal(solver.info.encodedSteps, step);
    if (!stationary) {
      const checkpoint = await solver.readDiagnosticFields(true);
      for (const name of ["density", "velocity", "pressure"] as const) {
        const data = checkpoint[name];
        await writeFile(`${output}/step-${step}-${name}.bin`, new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      }
    }
  }
  const activity = await solver.readGPUActivityPolicy();
  assert.deepEqual(topology(activity), topology(initialActivity), "stationary diagnostic must retain its actual topology");
  const fields = await solver.readDiagnosticFields(true);
  const published = await readPublishedCM12Field(device, solver);
  const heights = new Float32Array(nx * nz).fill(NaN), exact = new Float32Array(nx * nz);
  let maxDensityChange = 0, maxSpeed = 0;
  for (let i = 0; i < fields.density.length; i++) {
    maxDensityChange = Math.max(maxDensityChange, Math.abs(fields.density[i]! - initial.density[i]!));
    maxSpeed = Math.max(maxSpeed, Math.hypot(...fields.velocity.subarray(4 * i, 4 * i + 3)));
  }
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    exact[x + nx * z] = height(x + .5, z + .5) * h;
    for (let y = 0; y < ny - 1; y++) {
      const lo = published.values[x + nx * (y + ny * z)]!, hi = published.values[x + nx * (y + 1 + ny * z)]!;
      if (lo <= 0 && hi > 0) heights[x + nx * z] = (y + .5 - lo / (hi - lo)) * h;
    }
  }
  const rows = [];
  // Exclude walls. Sample both height and first/second derivatives: a smooth
  // coarse approximation may differ in height, but must not reveal its knots.
  for (let z = 8; z < nz - 8; z++) for (let x = 8; x < nx - 8; x++) {
    const at = x + nx * z;
    assert.ok(Number.isFinite(heights[at]), `published crossing ${x},${z}`);
    const slope = (heights[at + 1]! - heights[at - 1]!) / (2 * h);
    const exactSlope = profile === "flat" ? 0 : profile === "tilt" ? .03 : .006 * (x + .5 - nx / 2 - phase);
    const curvature = (heights[at + 1]! - 2 * heights[at]! + heights[at - 1]!) / h;
    rows.push({ x, z, height_m: heights[at], exact_m: exact[at], error_m: heights[at]! - exact[at]!,
      slope, exactSlope, slopeError: slope - exactSlope, curvature, exactCurvature: profile === "bowl" ? .006 : 0 });
  }
  const rms = (v: number[]) => Math.sqrt(v.reduce((s, a) => s + a * a, 0) / v.length);
  const stats = await solver.readStats();
  const summary = { arm, profile, phase, gravity, dt, steps, conditioning, maxDensityChange, maxSpeed,
    heightRMSError_mm: 1000 * rms(rows.map(r => r.error_m)),
    slopeRMSError: rms(rows.map(r => r.slopeError)),
    curvatureRMSError: rms(rows.map(r => r.curvature - r.exactCurvature)),
    curvatureMin: Math.min(...rows.map(r => r.curvature)), curvatureMax: Math.max(...rows.map(r => r.curvature)),
    activeWidths: Object.fromEntries([1, 2, 4, 8].map(w => [w, activity.bricks.filter(b => b.active && b.acceptedResolution === 8 / w).length])),
    pressureResidual: stats.pressureRelativeResidual,
  };
  await mkdir(output, { recursive: true });
  for (const [name, data] of Object.entries({ heights, exact, phi: published.values, density: fields.density, initialDensity: initial.density })) {
    await writeFile(`${output}/${name}.bin`, new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  }
  if (arm === "fixed4" && profile === "bowl" && stationary) {
    const kernel = await sampleCoarseBowlVolumeKernel(device, nx, ny, nz, h, width, phase);
    await writeFile(`${output}/isolated-kernel.bin`, new Uint8Array(kernel.buffer));
  }
  await writeFile(`${output}/config.json`, JSON.stringify({ scene, options, nx, ny, nz, h, dt, steps, gravity, conditioning, arm, profile, phase,
    seed: "17.3 + .003*((x-24-phase)^2 + .7*(z-20)^2), fine-cell units; 8x8 area quadrature", shaderHashes }, null, 2));
  await writeFile(`${output}/activity.json`, JSON.stringify(activity));
  await writeFile(`${output}/stats.json`, JSON.stringify(stats));
  await writeFile(`${output}/measurements.json`, JSON.stringify(rows));
  await writeFile(`${output}/summary.json`, JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary));
  if (stationary) assert.ok(maxDensityChange < 1e-6, `physics contaminated static surface: ${maxDensityChange}`);
  if (stationary) assert.ok(maxSpeed < 1e-6, `stationary fixture gained velocity: ${maxSpeed}`);
  assert.equal(activity.faultFlags, 0, "topology faults");
  assert.equal(activity.commitFailed, false, "topology commit failure");
  const frame = await solver.readFrameControlQA();
  assert.ok(frame, "frame control receipt is available");
  assert.equal(frame.fault, 0, "no failed or rolled-back frame");
  assert.equal(frame.committedFrames, steps, "every requested step committed");
  await writeFile(`${output}/frame-control.json`, JSON.stringify(frame));
  await writeFile(`${output}/presentation-faults.json`, JSON.stringify(await solver.readFramePlanPresentationFaultRecordQA() ?? null));
  assert.deepEqual(errors, []);
} finally {
  solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
  // gpu stays live through the last readback.
  if (gpu) liveGPU.delete(gpu);
}
