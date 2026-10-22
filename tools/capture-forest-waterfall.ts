/** Capture the registered forest scene through the production app renderer.
 *
 * node --import tsx tools/capture-forest-waterfall.ts --time=3 --width=1200
 * --time=0 captures the initial live pool. The only browser replacement is a
 * readable offscreen swap-chain texture. Scene, solver/profile, lighting,
 * water publication and composition all use their ordinary application paths.
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { deflateSync } from "node:zlib";
import "../lib/methods";
import { FluidLabRenderer, type RendererFrameMetrics } from "../lib/core/webgpu-renderer";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { defaultCamera, markSceneRevision } from "../lib/core/model";
import { getSceneDefinition, getScenePreset } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { getMethod } from "../lib/core/method-registry";
import { resolveMethodValues, type GPUSolverInstance } from "../lib/core/method-contract";
import type { EffectiveRendererStatus } from "../lib/core/renderer-status";
import type { GPUEulerianInfo } from "../lib/core/webgpu-eulerian";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { registerSceneMeshAssets } from "./blender/register-assets";
import { sceneMeshAssetIds } from "../lib/core/imported-assets/asset-library";
import { surfaceMeshAllowsLiveStartup, surfaceMeshConstructionFailed } from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { GPUMemoryProfile } from "./gpu-memory-profile";
import { createNbflipVolumeProbe } from "./probe-nbflip-volume";
import { sceneFluidFrame } from "../lib/core/scene-lattice";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

const options = new Map(process.argv.slice(2).map(value => {
  const match = /^--([^=]+)(?:=(.*))?$/.exec(value);
  if (!match) throw new Error(`Expected --name=value, received ${value}`);
  return [match[1]!, match[2] ?? "true"];
}));
if (options.has("help")) {
  console.log("Capture the standard forest-waterfall scene and narrow-band FLIP method.\n"
    + "Options: --time=3 --width=1200 --height=800 --out=artifacts/forest-waterfall/live-3s.png --timeout=1800\n"
    + "Use --time=0 for the initial pool. All render and solver settings remain the registered app defaults.");
  process.exit(0);
}
for (const key of options.keys()) {
  if (!["time", "width", "height", "out", "timeout"].includes(key)) throw new Error(`Unknown option --${key}`);
}
const number = (key: string, fallback: number, minimum: number) => {
  const value = Number(options.get(key) ?? fallback);
  assert.ok(Number.isFinite(value) && value >= minimum, `--${key} must be finite and >= ${minimum}`);
  return value;
};
const requestedTime_s = number("time", 3, 0);
const width = Math.round(number("width", 1200, 16));
const height = Math.round(number("height", Math.round(width * 2 / 3), 16));
const timeout_ms = number("timeout", 1800, 1) * 1000;
const pngPath = resolve(options.get("out") ?? `artifacts/forest-waterfall/live-${requestedTime_s}s.png`);
assert.ok(pngPath.endsWith(".png"), "--out must end in .png");
const reportPath = pngPath.replace(/\.png$/, ".json");
mkdirSync(dirname(pngPath), { recursive: true });

// Swap-chain pixels have already received the production display grade. Encode
// their byte colours directly; the rgba16float helper would grade them twice.
function writeBgraPng(path: string, pixels: Uint8Array, bytesPerRow: number): void {
  const crc = (bytes: Uint8Array) => {
    let value = 0xffffffff;
    for (const byte of bytes) {
      value ^= byte;
      for (let bit = 0; bit < 8; bit++) value = (value >>> 1) ^ (value & 1 ? 0xedb88320 : 0);
    }
    return (value ^ 0xffffffff) >>> 0;
  };
  const chunk = (type: string, body: Uint8Array) => {
    const result = Buffer.alloc(body.length + 12);
    result.writeUInt32BE(body.length, 0); result.write(type, 4, "ascii");
    result.set(body, 8); result.writeUInt32BE(crc(result.subarray(4, -4)), result.length - 4);
    return result;
  };
  const scanlines = Buffer.alloc(height * (width * 3 + 1));
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const src = y * bytesPerRow + x * 4, dest = y * (width * 3 + 1) + 1 + x * 3;
    scanlines[dest] = pixels[src + 2]!; scanlines[dest + 1] = pixels[src + 1]!; scanlines[dest + 2] = pixels[src]!;
  }
  const header = Buffer.alloc(13); header.writeUInt32BE(width, 0); header.writeUInt32BE(height, 4);
  header[8] = 8; header[9] = 2;
  writeFileSync(path, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", header), chunk("IDAT", deflateSync(scanlines)), chunk("IEND", new Uint8Array())]));
}

const definition = getSceneDefinition("forest-waterfall"), preset = getScenePreset(definition.id);
assert.ok(definition.methodProfile, "The scene must declare its standard method profile");
assert.equal(definition.methodProfile.methodId, "uniform-narrow-band-flip");
assert.equal(preset.presentationMode, "full-scene");
const profile = definition.methodProfile, method = getMethod(profile.methodId);
const scene = markSceneRevision(structuredClone(sceneDocument(definition)));
const values = resolveMethodValues(method, profile.quality, { ...method.appDefaults, ...profile.overrides });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
const steps = Math.ceil(requestedTime_s / dt - 1e-9), targetTime_s = steps * dt;
const assetIds = sceneMeshAssetIds(scene);
assert.ok(assetIds.length, "The actual Blender environment must be attached");
assert.ok(scene.container.fillFraction > 0 || scene.fluid.initialLiquidVolumes?.length, "The registered scene must contain real initial water");
const started = performance.now(), errors: string[] = [], statuses: Record<string, unknown>[] = [];
let device: GPUDevice | undefined, swapTexture: GPUTexture | undefined;
let memory: GPUMemoryProfile | undefined;
const memorySnapshots: { phase: string; time_s: number; process: NodeJS.MemoryUsage; gpu: ReturnType<GPUMemoryProfile["snapshot"]> }[] = [];
const snapshotMemory = (phase: string) => {
  if (memory) memorySnapshots.push({ phase, time_s: completedTime_s, process: process.memoryUsage(), gpu: memory.snapshot() });
};
let renderer: FluidLabRenderer | undefined, effective: EffectiveRendererStatus | undefined;
let latestInfo: GPUEulerianInfo | undefined, lastMetrics: RendererFrameMetrics | undefined;
let failure: string | undefined, closing = false, completedTime_s = 0, frames = 0;
let successful = false, lastProgress = 0;
const liveFrameSamples: { time_s: number; wall_ms: number }[] = [];
const report: Record<string, unknown> = {
  sceneId: definition.id, methodId: method.id, quality: profile.quality, methodValues: values,
  voxelDomain: scene.voxelDomain,
  renderProfile: scene.renderProfile,
  containerSize_m: [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
  requestedTime_s, targetTime_s, dt, steps, width, height, pngPath,
  capturePath: "FluidLabRenderer.drawWithDeferredPresentation -> production BGRA8 swap chain",
  presentationMode: preset.presentationMode, assetIds, decorativeWater: false,
};
const check = () => {
  if (failure) throw new Error(failure);
  assert.equal(errors.length, 0, errors.join("\n"));
  if (performance.now() - started > timeout_ms) throw new Error(`Capture exceeded ${timeout_ms / 1000}s`);
};
const turn = () => new Promise<void>(resolveTurn => setTimeout(resolveTurn, 0));
const bounded = async <T>(promise: Promise<T>, label: string): Promise<T> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([promise, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} exceeded capture deadline`)),
        Math.max(1, timeout_ms - (performance.now() - started)));
    })]);
  } finally { clearTimeout(timer); }
};
const progress = (phase: string, force = false) => {
  if (force || performance.now() - lastProgress > 10000) {
    lastProgress = performance.now();
    console.log(JSON.stringify({ phase, elapsed_ms: lastProgress - started, frames, completedTime_s,
      submittedTime_s: latestInfo?.submittedTime_s, mesh: effective?.surfaceMesh, renderer: effective?.state }));
  }
};

try {
  progress("load-blender-assets", true);
  await registerSceneMeshAssets(scene);
  check();
  paceDawnEventPolling();
  const dawn = await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE ?? "node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter, "No native WebGPU adapter");
  report.adapter = adapter.info;
  if (typeof navigator === "undefined") Object.defineProperty(globalThis, "navigator", { value: {}, configurable: true });
  Object.defineProperty(navigator, "gpu", { configurable: true, value: {
    requestAdapter: async () => ({ features: adapter.features, limits: adapter.limits, info: adapter.info,
      requestDevice: async (descriptor: GPUDeviceDescriptor) => {
        memory = new GPUMemoryProfile(await adapter.requestDevice(descriptor));
        device = memory.device;
        managedGPUDevice(device, { requireWorkerRealm: false });
        device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
        void device.lost.then(info => { if (!closing) failure = `GPU lost: ${info.message || info.reason}`; });
        return device;
      } }), getPreferredCanvasFormat: () => "bgra8unorm",
  } });
  const canvas = { width, height, getContext: (kind: string) => kind === "webgpu" ? {
    configure: () => {
      swapTexture?.destroy();
      swapTexture = device!.createTexture({ label: "Forest capture app swap chain", size: [canvas.width, canvas.height],
        format: "bgra8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    }, getCurrentTexture: () => {
      if (swapTexture?.width !== canvas.width || swapTexture?.height !== canvas.height) {
        swapTexture?.destroy();
        swapTexture = device!.createTexture({ label: "Forest capture resized app swap chain",
          size: [canvas.width, canvas.height], format: "bgra8unorm",
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      }
      return swapTexture;
    }, unconfigure: () => {},
  } : null } as unknown as OffscreenCanvas;
  renderer = new FluidLabRenderer(canvas, status => {
    const record = { at_ms: performance.now() - started, state: status.state, label: status.label,
      ...("phase" in status ? { phase: status.phase, completed: status.completed, total: status.total } : {}) };
    statuses.push(record);
    if (["unavailable", "lost", "blocked"].includes(status.state)) failure = status.label;
    if (performance.now() - lastProgress > 10000) { console.log(JSON.stringify(record)); lastProgress = performance.now(); }
  }, info => { latestInfo = info; }, undefined, time => { completedTime_s = time; }, status => { effective = status; });
  // Read-only inspection of production publications; no solver or renderer
  // state is injected, overridden or reconstructed by the capture harness.
  const internals = renderer as unknown as { gpuFluid?: GPUSolverInstance; gpuFluidPending?: Promise<void>;
    svoDryScenePipeline?: SparseVoxelDrySceneRenderer };
  const measureWater = async () => {
    const solver = internals.gpuFluid as WebGPUUniformReferenceSolver;
    const probe = await createNbflipVolumeProbe(device!, solver);
    try {
      const volume = await probe.read(), frame = sceneFluidFrame(scene);
      const cellVolume = frame.cellSize_m.reduce((a, b) => a * b, 1);
      const poolLevel = Math.max(...(scene.fluid.initialLiquidVolumes ?? []).map(v => v.shape === "box" ? v.max_m.y : -Infinity));
      let total_m3 = 0, pond_m3 = 0;
      for (let i = 0; i < volume.length; i++) {
        const amount = volume[i]! * cellVolume;
        assert.ok(Number.isFinite(amount) && amount >= 0, "Live Eulerian water volume must be finite");
        total_m3 += amount;
        const y = frame.container_m[1] + (frame.boxLow[1] + Math.floor(i / solver.info.nx) % solver.info.ny) * frame.cellSize_m[1];
        if (y < poolLevel + .5) pond_m3 += amount;
      }
      return { time_s: completedTime_s, total_m3, pond_m3, measurement: "Production phi and solid-capacity GPU readback, including settled Eulerian water" };
    } finally { probe.destroy(); }
  };
  const args = (time: number): Parameters<FluidLabRenderer["draw"]> => [
    time, scene, { ...defaultCamera, ...preset.camera }, [], undefined,
    { methodId: method.id, quality: profile.quality, values, simulationEpoch: 0, inFlightDepth: 2 },
    { axis: "off", position: 0 }, preset.background, preset.presentationMode,
  ];
  const draw = async (time: number) => {
    check(); lastMetrics = await renderer!.drawWithDeferredPresentation(args(time)); frames++;
    await device!.queue.onSubmittedWorkDone();
    // Production completion callbacks retire their health receipts before
    // counting a presentation; yield so those callbacks can publish authority.
    await turn();
    latestInfo = internals.gpuFluid?.info ?? latestInfo;
    completedTime_s = Math.max(completedTime_s, latestInfo?.completedTime_s ?? 0);
    check();
    if (surfaceMeshConstructionFailed(effective?.surfaceMesh)) throw new Error(effective?.surfaceMesh?.detail ?? "Surface mesh construction failed");
  };
  await bounded(renderer.initialize(), "Renderer initialization"); check();
  await renderer.setSimulationRunning(false);
  renderer.draw(...args(0));
  await bounded(internals.gpuFluidPending ?? Promise.resolve(), "Scene/solver initialization"); check();
  assert.ok(internals.gpuFluid, "Production renderer did not attach the scene solver");
  snapshotMemory("solver-constructed");
  while (true) {
    await draw(0); progress("initial-presentation");
    if (effective?.state === "active" && surfaceMeshAllowsLiveStartup(effective.surfaceMesh)
      && renderer.completedPresentationCount > 0 && lastMetrics?.presentationSubmitted) break;
  }
  progress("initial-live-pool-ready", true);
  snapshotMemory("initial-presentation");
  const initialWater = await measureWater();
  report.initialWater = initialWater;
  if (targetTime_s > 0) {
    await renderer.setSimulationRunning(true);
    while (completedTime_s < targetTime_s - dt * .01) {
      // Like the app clock, admit whole fixed steps. Queue fences make this an
      // evidence capture rather than a throughput benchmark; physics is intact.
      const nextTime = Math.min(targetTime_s, (Math.round(completedTime_s / dt) + 1) * dt);
      const frameStart = performance.now();
      await draw(nextTime);
      liveFrameSamples.push({ time_s: completedTime_s, wall_ms: performance.now() - frameStart });
      progress("simulate-live-water");
    }
  }
  const pausedTime_s = await renderer.setSimulationRunning(false) ?? completedTime_s;
  report.pausedTime_s = pausedTime_s;
  // Settle the ordinary paused presentation and its asynchronous diagnostics.
  let settled = 0;
  while (settled < 4) {
    await draw(pausedTime_s);
    if (effective?.state === "active" && surfaceMeshAllowsLiveStartup(effective.surfaceMesh) && lastMetrics?.presentationSubmitted) settled++;
    else settled = 0;
    progress("settle-final-presentation");
  }
  // Fenced wall measurements expose capture-host cost, not GPU-only timing or
  // interactive FPS. Keep the ordinary paused production scene and settings.
  const wallSamples_ms: number[] = [];
  for (let i = 0; i < 12; i++) {
    const frameStart = performance.now();
    await draw(pausedTime_s);
    wallSamples_ms.push(performance.now() - frameStart);
    assert.ok(lastMetrics?.presentationSubmitted, "Steady frame did not submit a production presentation");
  }
  const sorted = [...wallSamples_ms].sort((a, b) => a - b);
  report.pausedSteadyFrameWallTiming = {
    measurement: "Host wall time including production draw, Dawn GPU queue fence, and completion callback yield; not GPU-only timing or interactive FPS",
    sampleCount: wallSamples_ms.length, samples_ms: wallSamples_ms,
    mean_ms: wallSamples_ms.reduce((a, b) => a + b, 0) / wallSamples_ms.length,
    median_ms: (sorted[5]! + sorted[6]!) / 2, p95_ms: sorted[Math.ceil(sorted.length * .95) - 1],
    min_ms: sorted[0], max_ms: sorted.at(-1), width, height, simulationPaused: true,
  };
  report.importedSurfaceDrawStatus = internals.svoDryScenePipeline?.importedSurfaceDrawStatus;
  snapshotMemory("final-presentation");
  const finalWater = await measureWater();
  report.finalWater = finalWater;
  assert.ok(finalWater.pond_m3 > .8 * initialWater.pond_m3 && finalWater.pond_m3 > 100, "The final live pond must retain its Eulerian water");
  latestInfo = await internals.gpuFluid.readStats();
  check();
  assert.equal(latestInfo.encodedSteps, steps, "Capture must contain exactly the requested whole fixed steps");
  assert.ok(renderer.completedPresentationCount > 0, "No completed production presentation");
  assert.ok(completedTime_s >= targetTime_s - dt * .01, "Capture has not reached the requested live time");
  const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
  const readback = device!.createBuffer({ label: "Forest app screenshot readback", size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const encoder = device!.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: swapTexture! }, { buffer: readback, bytesPerRow }, { width, height });
    device!.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    writeBgraPng(pngPath, new Uint8Array(readback.getMappedRange()), bytesPerRow);
    readback.unmap();
  } finally { readback.destroy(); }
  check(); successful = true;
  progress("captured-production-live-scene", true);
} catch (error) {
  failure = error instanceof Error ? error.message : String(error);
  console.error(failure); process.exitCode = 1;
} finally {
  report.successful = successful;
  report.failure = failure; report.gpuErrors = errors; report.statuses = statuses;
  report.elapsed_ms = performance.now() - started; report.frames = frames;
  report.completedTime_s = completedTime_s; report.completedPresentations = renderer?.completedPresentationCount;
  report.effectiveRenderer = effective; report.solverInfo = latestInfo;
  report.finalFrameMetrics = lastMetrics; report.presentationResolution = renderer?.presentationResolution;
  const liveWalls = liveFrameSamples.map(s => s.wall_ms).sort((a, b) => a - b);
  report.liveFrameWallTiming = { measurement: "Serialized production draw plus solver, renderer maintenance, queue fence and completion callback yield; not interactive FPS", samples: liveFrameSamples,
    median_ms: liveWalls[Math.floor(liveWalls.length / 2)], p95_ms: liveWalls[Math.ceil(liveWalls.length * .95) - 1] };
  report.memory = { scope: "All live explicit WebGPU buffers and logical texture payloads, including solver, renderer and capture swap chain; excludes driver alignment, caches and executable code. Process RSS also includes native allocations and is not additive to GPU bytes on unified memory.", snapshots: memorySnapshots };
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
  closing = true;
  await device?.queue.onSubmittedWorkDone().catch(() => {});
  renderer?.destroy(); swapTexture?.destroy(); device?.destroy();
}
