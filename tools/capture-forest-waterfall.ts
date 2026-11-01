import { initializeRigidBodies } from "../lib/core/rigid-body";
import { sceneRenderTuning } from "../lib/core/scene-render-profile";
/** Capture a registered live scene through the production app renderer.
 *
 * node --import tsx tools/capture-forest-waterfall.ts --time=3 --width=1200
 * --time=0 captures the initial live pool. Browser presentation uses a readable
 * offscreen swap-chain texture and next-turn resource-retirement scheduling.
 * Scene, solver/profile, lighting,
 * water publication and composition all use their ordinary application paths.
 */
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
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
import { waterMeshObj } from "./blender/nbflip-water-mesh";
import { sceneMeshAssetIds, meshAssetVolume } from "../lib/core/imported-assets/asset-library";
import { surfaceMeshAllowsLiveStartup, surfaceMeshConstructionFailed } from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { GPUMemoryProfile } from "./gpu-memory-profile";
import { createNbflipVolumeProbe } from "./probe-nbflip-volume";
import { sceneFluidFrame } from "../lib/core/scene-lattice";
import { DEFAULT_SVO_LIGHTING_OPTIONS, type SvoConeTracingMode } from "../lib/svo/pipeline/svo-render-options";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";
import type { WaterReflectionMode } from "../lib/features/surface-display/definition";
import { DEFAULT_LENS_BLOOM_OPTIONS } from "../lib/core/webgpu-lens-bloom";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";

const options = new Map(process.argv.slice(2).map(value => {
  const match = /^--([^=]+)(?:=(.*))?$/.exec(value);
  if (!match) throw new Error(`Expected --name=value, received ${value}`);
  return [match[1]!, match[2] ?? "true"];
}));
if (options.has("help")) {
  console.log("Capture a registered scene and narrow-band FLIP method (default forest-waterfall).\n"
    + "Options: --scene=tokyo-rain --time=3 --width=1200 --height=800 --export-water-mesh --out=artifacts/forest-waterfall/live-3s.png --timeout=1800 --timing-samples=12 --steady-reuse=off|on --render-scale=1 --reflections=screen-space|world-space|environment --reflection-comparison --screen-reflection-comparison --supersampling-comparison --reflection-diagnostics --light-samples=32 --light-budget-comparison --profile-passes --performance-comparison --live-visibility-validation --area-samples=1 --visibility=raster-ao|cones|exact|off --indirect-lighting --gi-cache=on|off --lattice=on|off --gi-comparison --lens-bloom --bloom-comparison --voxel-reflections --voxel-reflection-comparison --voxel-reflection-steps=128 --bloom-strength=.08 --bloom-threshold=1 --bloom-radius=.03 --texture-comparison --lighting-comparison --shadow-aperture=0.065 --shadow-resolution=2048|4096 --shadow-resolution-comparison --sun-softness-comparison --shadow-diagnostics --native-sun-diagnostics --lighting-energy-diagnostics --native-ao --native-ao-comparison --world-refraction --world-refraction-comparison --world-refinement-diagnostics\n"
    + "Use --time=0 for the initial pool. Render scale can be raised to 2 for supersampling; solver settings remain the registered app defaults.");
  process.exit(0);
}
for (const key of options.keys()) {
  if (!["steady-reuse", "export-water-mesh", "scene", "time", "width", "height", "out", "timeout", "timing-samples", "reflection-comparison", "screen-reflection-comparison", "reflection-diagnostics", "reflections", "render-scale", "supersampling-comparison", "light-samples", "light-budget-comparison", "profile-passes", "performance-comparison", "live-visibility-validation", "area-samples", "indirect-lighting", "gi-cache", "lattice", "gi-comparison", "lens-bloom", "bloom-comparison", "voxel-reflections", "voxel-reflection-comparison", "voxel-reflection-steps", "bloom-strength", "bloom-threshold", "bloom-radius", "visibility", "texture-comparison", "lighting-comparison", "shadow-aperture", "shadow-resolution", "shadow-resolution-comparison", "sun-softness-comparison", "shadow-diagnostics", "native-sun-diagnostics", "lighting-energy-diagnostics", "native-ao", "native-ao-comparison", "world-refraction", "world-refraction-comparison", "world-refinement-diagnostics"].includes(key)) throw new Error(`Unknown option --${key}`);
}
const number = (key: string, fallback: number, minimum: number) => {
  const value = Number(options.get(key) ?? fallback);
  assert.ok(Number.isFinite(value) && value >= minimum, `--${key} must be finite and >= ${minimum}`);
  return value;
};
const requestedTime_s = number("time", 3, 0);
if(options.has("live-visibility-validation")){assert.ok(requestedTime_s>0);assert.ok(options.has("performance-comparison"),"Live validation requires the frozen reference matrix");}
const sceneId = options.get("scene") ?? "forest-waterfall";
const width = Math.round(number("width", 1200, 16));
const height = Math.round(number("height", Math.round(width * 2 / 3), 16));
const renderScale = number("render-scale", 1, .35);
assert.ok(renderScale <= 2, "--render-scale must be <= 2");
let lightSamples = options.has("light-samples") ? number("light-samples", 8, 1) : undefined;
const areaSamples = options.has("area-samples") ? number("area-samples", 2, 1) : undefined;
const shadowAperture = options.has("shadow-aperture") ? number("shadow-aperture", .065, .01) : undefined;
assert.ok(shadowAperture === undefined || shadowAperture <= .25, "--shadow-aperture must be between .01 and .25 radians");
const shadowResolution = number("shadow-resolution", 2048, 2048);
assert.ok(shadowResolution === 2048 || shadowResolution === 4096, "--shadow-resolution must be 2048 or 4096");
const indirectLighting = options.has("indirect-lighting") ? options.get("indirect-lighting") !== "off" : DEFAULT_SVO_LIGHTING_OPTIONS.globalIlluminationEnabled;
const visibilityMode = (options.get("visibility") ?? (indirectLighting ? "cones" : DEFAULT_SVO_LIGHTING_OPTIONS.coneTracingMode)) as SvoConeTracingMode;
assert.ok(["raster-ao", "cones", "exact", "off"].includes(visibilityMode), "--visibility must be raster-ao, cones, exact or off");
assert.ok(!indirectLighting || visibilityMode === "cones", "--indirect-lighting requires cone visibility");
const latticeEnabled=options.has("lattice")?options.get("lattice")!=="off":DEFAULT_SVO_LIGHTING_OPTIONS.latticeVisibilityEnabled;
const giCache=options.has("gi-cache")?options.get("gi-cache")!=="off":indirectLighting;
const voxelReflections=options.has("voxel-reflections");
const voxelReflectionSteps=number("voxel-reflection-steps",128,1);
assert.ok(Number.isInteger(voxelReflectionSteps)&&voxelReflectionSteps<=512);
assert.ok(!(voxelReflections||options.has("voxel-reflection-comparison"))||indirectLighting&&visibilityMode==="cones","Voxel reflection comparison requires cones + GI");
const lensBloom={...DEFAULT_LENS_BLOOM_OPTIONS,enabled:options.has("lens-bloom"),
  strength:number("bloom-strength",DEFAULT_LENS_BLOOM_OPTIONS.strength,0),
  threshold:number("bloom-threshold",DEFAULT_LENS_BLOOM_OPTIONS.threshold,0),
  radius:number("bloom-radius",DEFAULT_LENS_BLOOM_OPTIONS.radius,.002)};
assert.ok(lensBloom.strength<=1&&lensBloom.threshold<=64&&lensBloom.radius<=.15,"Invalid lens-bloom setting");
const selectedReflections = (options.get("reflections") ?? "screen-space") as WaterReflectionMode;
assert.ok(["screen-space", "world-space", "environment"].includes(selectedReflections), "--reflections must be screen-space, world-space or environment");
assert.ok(lightSamples === undefined || Number.isInteger(lightSamples) && lightSamples <= 32, "--light-samples must be an integer from1 to32");
assert.ok(areaSamples === undefined || Number.isInteger(areaSamples) && areaSamples <= 2, "--area-samples must be1 or2");
const timeout_ms = number("timeout", 1800, 1) * 1000;
const timingSamples = Math.round(number("timing-samples",12,1));
const pngPath = resolve(options.get("out") ?? `artifacts/${sceneId}/live-${requestedTime_s}s.png`);
assert.ok(pngPath.endsWith(".png"), "--out must end in .png");
const reportPath = pngPath.replace(/\.png$/, ".json");
mkdirSync(dirname(pngPath), { recursive: true });

// Renderer resource retirement defers its fence until after the current draw.
// Node has no browser frame scheduler; preserve that ordering on the next turn.
if (typeof globalThis.requestAnimationFrame !== "function") {
  let nextCaptureFrame = 1;
  const pendingCaptureFrames = new Map<number, ReturnType<typeof setTimeout>>();
  globalThis.requestAnimationFrame = callback => {
    const id = nextCaptureFrame++;
    pendingCaptureFrames.set(id, setTimeout(() => {
      pendingCaptureFrames.delete(id);
      callback(performance.now());
    }, 0));
    return id;
  };
  globalThis.cancelAnimationFrame = id => {
    const timer = pendingCaptureFrames.get(id);
    if (timer !== undefined) clearTimeout(timer);
    pendingCaptureFrames.delete(id);
  };
}

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

const definition = getSceneDefinition(sceneId), preset = getScenePreset(definition.id);
assert.equal(definition.id, sceneId, "Unknown scene ID");
assert.ok(definition.methodProfile, "The scene must declare its standard method profile");
assert.equal(definition.methodProfile.methodId, "uniform-narrow-band-flip");
assert.equal(preset.presentationMode, "full-scene");
const profile = definition.methodProfile, method = getMethod(profile.methodId);
const scene = markSceneRevision(structuredClone(sceneDocument(definition)));
const values = resolveMethodValues(method, profile.quality, { ...method.appDefaults, ...profile.overrides });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
const steps = Math.ceil(requestedTime_s / dt - 1e-9), targetTime_s = steps * dt;
const assetIds = sceneMeshAssetIds(scene);
assert.ok(assetIds.length || scene.rigidBodies.some(body => body.appearance), "The actual Blender environment or moving model must be attached");
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
const sourceHashes:Record<string,string>={};
for(const entry of readdirSync(resolve("lib"),{recursive:true,withFileTypes:true})) {
  if(!entry.isFile()||! /\.(ts|tsx|wgsl)$/.test(entry.name))continue;
  const path=resolve(entry.parentPath,entry.name);
  sourceHashes[path]=createHash("sha256").update(readFileSync(path)).digest("hex");
}
sourceHashes[resolve("tools/capture-forest-waterfall.ts")]=createHash("sha256").update(readFileSync(resolve("tools/capture-forest-waterfall.ts"))).digest("hex");
const report: Record<string, unknown> = {
  sourceHashes,
  sceneId: definition.id, methodId: method.id, quality: profile.quality, methodValues: values,
  voxelDomain: scene.voxelDomain,
  renderProfile: scene.renderProfile,
  containerSize_m: [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
  requestedTime_s, targetTime_s, dt, steps, width, height, renderScale, lightSamples, areaSamples, indirectLighting, visibilityMode, reflectionMode: selectedReflections, pngPath,
  capturePath: "FluidLabRenderer.drawWithDeferredPresentation -> production BGRA8 swap chain",
  presentationMode: preset.presentationMode, assetIds, decorativeWater: false,
  ...(process.env.FLUID_ASSET_VARIANT_PROOF ? {stagedAssetVariants:JSON.parse(process.env.FLUID_ASSET_VARIANT_PROOF)} : {}),
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
  report.sampledAssetPublications = assetIds.map(id=>{
    const {header,surfaceMapping}=meshAssetVolume(id);
    return {id,cellSize:header.cellSize,brickCount:header.brickCount,surfaceMapping:header.surfaceMapping,
      ...(surfaceMapping ? {mappingCount:surfaceMapping.header.mappingCount,
        fvxPayloadSha256:surfaceMapping.header.fvxPayloadSha256,
        imageSize:[surfaceMapping.header.textureWidth,surfaceMapping.header.textureHeight],
        imageLayers:surfaceMapping.header.textureLayerCount} : {})};
  });
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
    svoDryScenePipeline?: SparseVoxelDrySceneRenderer; waterPipeline?: RasterWaterPipeline; viewUniformWords: Float32Array };
  const measureWater = async () => {
    const solver = internals.gpuFluid as WebGPUUniformReferenceSolver;
    const probe = await createNbflipVolumeProbe(device!, solver);
    try {
      const volume = await probe.read(), frame = sceneFluidFrame(scene);
      const cellVolume = frame.cellSize_m.reduce((a, b) => a * b, 1);
      const poolLevel = Math.max(...(scene.fluid.initialLiquidVolumes ?? []).map(v => v.shape === "box" ? v.max_m.y : -Infinity));
      let minimum=Infinity, maximum=-Infinity, nonfinite=0;
      for(const fill of volume){ if(Number.isFinite(fill)){minimum=Math.min(minimum,fill);maximum=Math.max(maximum,fill);}else nonfinite++; }
      report.lastVolumeReadback={time_s:completedTime_s,minimum,maximum,nonfinite};
      if(nonfinite){report.invalidVolumeCell=await probe.inspectCell(volume.findIndex(fill=>!Number.isFinite(fill)));}
      let total_m3 = 0, pond_m3 = 0;
      for (let i = 0; i < volume.length; i++) {
        const amount = volume[i]! * cellVolume;
        assert.ok(Number.isFinite(amount) && amount >= 0, `Live Eulerian water volume invalid at cell ${i}: fill=${volume[i]}, volume=${amount}`);
        total_m3 += amount;
        const y = frame.container_m[1] + (frame.boxLow[1] + Math.floor(i / solver.info.nx) % solver.info.ny) * frame.cellSize_m[1];
        if (y < poolLevel + .5) pond_m3 += amount;
      }
      return { time_s: completedTime_s, total_m3, pond_m3, measurement: "Production phi and solid-capacity GPU readback, including settled Eulerian water" };
    } finally { probe.destroy(); }
  };
  let captureShadowResolution: 2048 | 4096 = shadowResolution;
  let surfaceTexturesEnabled = true;
  let nativeAmbientOcclusionEnabled = options.has("native-ao");
  let waterWorldRefractionEnabled = options.has("world-refraction");
  let shadowsEnabled: boolean = DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled;
  let ambientOcclusionEnabled: boolean = DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled;
  let reflectionMode: WaterReflectionMode = selectedReflections;
  let captureRenderScale = renderScale;
  let captureVisibilityMode = visibilityMode;
  let captureIndirectLighting = indirectLighting;
  let captureGiCache = giCache;
  let captureBloomEnabled=lensBloom.enabled;
  let captureVoxelReflections=voxelReflections;
  let captureStaticVisibilityReuse:boolean|undefined = undefined;
  let captureShadowAperture = shadowAperture ?? sceneRenderTuning(scene.renderProfile).shadowConeAperture;
  report.shadowAperture_rad = captureShadowAperture;
  const captureBodies = initializeRigidBodies(scene.rigidBodies);
  const args = (time: number): Parameters<FluidLabRenderer["draw"]> => [
    time, scene, { ...defaultCamera, ...preset.camera }, captureBodies, undefined,
    { methodId: method.id, quality: profile.quality, values, simulationEpoch: 0, inFlightDepth: 2 },
    { axis: "off", position: 0 }, preset.background, preset.presentationMode, "shaded", { ...DEFAULT_SVO_LIGHTING_OPTIONS, waterReflectionMode: reflectionMode, waterWorldRefractionEnabled, coneTracingMode: captureVisibilityMode, surfaceTexturesEnabled, nativeAmbientOcclusionEnabled, shadowsEnabled, ambientOcclusionEnabled,
      voxelSpecularReflectionsEnabled:captureVoxelReflections,voxelSpecularSteps:voxelReflectionSteps,latticeVisibilityEnabled:latticeEnabled,globalIlluminationEnabled:captureIndirectLighting,worldGiCacheEnabled:captureGiCache,lensBloom:{...lensBloom,enabled:captureBloomEnabled} }, undefined,
    { ...sceneRenderTuning(scene.renderProfile), resolutionScale: captureRenderScale, shadowConeAperture: captureShadowAperture, rasterShadowResolution: captureShadowResolution,
      ...(lightSamples === undefined ? {} : {maximumShadedLights:lightSamples}),
      ...(captureStaticVisibilityReuse === undefined ? {} : {rasterStillViewReuse:captureStaticVisibilityReuse}),
      ...(captureIndirectLighting ? {giBounceStrength:1} : {}),
      ...(areaSamples === undefined ? {} : {stableAreaLightSamples:areaSamples,movingAreaLightSamples:areaSamples}) },
  ];
  const draw = async (time: number) => {
    check();
    lastMetrics = await renderer!.drawWithDeferredPresentation(args(time)); frames++;
    await device!.queue.onSubmittedWorkDone();
    if(captureBloomEnabled&&internals.waterPipeline&&!internals.waterPipeline.lensBloomReady)await internals.waterPipeline.ensureLensBloomReady();
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
  let liveCachedPlanes:Buffer[]|undefined;
  let liveIdentity:{solver:GPUSolverInstance;pipeline:SparseVoxelDrySceneRenderer|undefined;time_s:number;steps:number}|undefined;
  if (targetTime_s > 0) {
    if(options.has("live-visibility-validation")){captureStaticVisibilityReuse=true;report.visibilityBeforeLive=internals.svoDryScenePipeline?.rasterVisibilityCacheStatistics;}
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
  if(options.has("live-visibility-validation")){
    const cache=internals.svoDryScenePipeline?.rasterVisibilityCacheStatistics;
    report.visibilityAfterLive=cache;
    const previous=report.visibilityBeforeLive as {reusedFrames?:number;dryFrame?:{reusedFrames?:number}}|undefined;
    if(captureVisibilityMode==="cones") {
      assert.equal(cache?.dryFrame?.eligibility,"eligible","Live cone dry-frame reuse must be eligible");
      assert.ok((cache?.dryFrame?.reusedFrames??0)>(previous?.dryFrame?.reusedFrames??0),"Advancing water must reuse the unchanged dry frame");
    } else {
      assert.ok(cache?.lastFrameReused,"Final advancing-water frame must reuse visibility");
      assert.ok((cache.reusedFrames??0)>(previous?.reusedFrames??0));
    }
    assert.ok(lastMetrics?.presentationSubmitted,"Final live image must be a submitted presentation");
    assert.ok(internals.waterPipeline?.worldReflectionsApplied,"Live World reflections must have been encoded");
    if(waterWorldRefractionEnabled)assert.ok(internals.waterPipeline.worldRefractionsApplied);
    const info=await internals.gpuFluid!.readStats();
    liveIdentity={solver:internals.gpuFluid!,pipeline:internals.svoDryScenePipeline,time_s:info.completedTime_s!,steps:info.encodedSteps!};
    liveCachedPlanes=[];
    for(const stage of ["display","scene","world-reflections"]){
      const plane=stage==="display"?{texture:swapTexture!,origin:[0,0,0]}:internals.waterPipeline?.diagnosticCaptureTexture(stage);
      assert.ok(plane,`Missing live ${stage}`);
      const texture=plane.texture,row=texture.width*(stage==="display"?4:texture.format==="rgba32float"?16:8),stride=Math.ceil(row/256)*256;
      const staging=device!.createBuffer({size:stride*texture.height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try{
        const encoder=device!.createCommandEncoder();encoder.copyTextureToBuffer({texture,origin:plane.origin??[0,0,0]},{buffer:staging,bytesPerRow:stride},[texture.width,texture.height]);
        device!.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
        const mapped=new Uint8Array(staging.getMappedRange()),packed=Buffer.alloc(row*texture.height);
        for(let y=0;y<texture.height;y++)packed.set(mapped.subarray(y*stride,y*stride+row),y*row);
        liveCachedPlanes.push(packed);
      }finally{if(staging.mapState==="mapped")staging.unmap();staging.destroy();}
    }
    report.liveSnapshot={time_s:completedTime_s,presentations:renderer.completedPresentationCount,simulationRunning:true};
  }
  // A completed queue fence is not evidence that the backend executed work.
  // Catch silent zero readbacks before treating them as a physical dry-out.
  const validateExecution=async(label:string)=>{
  const expected=(0x13579bdf ^ frames)>>>0;
  const sentinel=device!.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const sentinelRead=device!.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
    const module=device!.createShaderModule({code:`@group(0) @binding(0) var<storage,read_write> output:array<u32>; @compute @workgroup_size(1) fn main(){output[0]=${expected}u;}`});
    const pipeline=await device!.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const encoder=device!.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device!.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:sentinel}}]}));pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(sentinel,0,sentinelRead,0,4);device!.queue.submit([encoder.finish()]);
    await sentinelRead.mapAsync(GPUMapMode.READ);const actual=new Uint32Array(sentinelRead.getMappedRange())[0];report.gpuExecutionSentinel={label,expected,actual};
    ((report.gpuExecutionSentinels ??= []) as unknown[]).push({label,expected,actual});
    assert.equal(actual,expected,"GPU execution/readback sentinel failed; zero buffers are not valid simulation evidence");
  }finally{if(sentinelRead.mapState==="mapped")sentinelRead.unmap();sentinel.destroy();sentinelRead.destroy();}
  };
  const pausedTime_s = await renderer.setSimulationRunning(false) ?? completedTime_s;
  report.pausedTime_s = pausedTime_s;
  if(options.has("steady-reuse")){
    assert.ok(options.get("steady-reuse")==="off"||options.get("steady-reuse")==="on");
    captureStaticVisibilityReuse=options.get("steady-reuse")==="on";
  }
  // Settle the ordinary paused presentation and its asynchronous diagnostics.
  let settled = 0;
  while (settled < 4) {
    await draw(pausedTime_s);
    if (effective?.state === "active" && surfaceMeshAllowsLiveStartup(effective.surfaceMesh) && lastMetrics?.presentationSubmitted) settled++;
    else settled = 0;
    progress("settle-final-presentation");
    if(options.has("steady-reuse"))await validateExecution(`steady-settle-${settled}`);
  }
  // Fenced wall measurements expose capture-host cost, not GPU-only timing or
  // interactive FPS. Keep the ordinary paused production scene and settings.
  const wallSamples_ms: number[] = [];
  for (let i = 0; i < timingSamples; i++) {
    const frameStart = performance.now();
    await draw(pausedTime_s);
    wallSamples_ms.push(performance.now() - frameStart);
    if(options.has("steady-reuse")){
      console.log(JSON.stringify({phase:"steady-diagnostic",sample:i,wall_ms:wallSamples_ms.at(-1)}));
      await validateExecution(`steady-sample-${i}`);
    }
    assert.ok(lastMetrics?.presentationSubmitted, "Steady frame did not submit a production presentation");
  }
  const sorted = [...wallSamples_ms].sort((a, b) => a - b);
  report.pausedSteadyFrameWallTiming = {
    measurement: "Host wall time including production draw, Dawn GPU queue fence, and completion callback yield; not GPU-only timing or interactive FPS",
    sampleCount: wallSamples_ms.length, samples_ms: wallSamples_ms,
    mean_ms: wallSamples_ms.reduce((a, b) => a + b, 0) / wallSamples_ms.length,
    median_ms: (sorted[Math.floor((sorted.length-1)/2)]! + sorted[Math.floor(sorted.length/2)]!) / 2, p95_ms: sorted[Math.ceil(sorted.length * .95) - 1],
    min_ms: sorted[0], max_ms: sorted.at(-1), width, height, renderScale,
    presentationResolution: renderer.presentationResolution, simulationPaused: true,
  };
  report.importedSurfaceDrawStatus = internals.svoDryScenePipeline?.importedSurfaceDrawStatus;
  report.worldRefractionsApplied = internals.waterPipeline?.worldRefractionsApplied ?? false;
  if(waterWorldRefractionEnabled)assert.equal(report.worldRefractionsApplied,true,"Requested World refraction must actually encode");
  report.worldReflectionsApplied = internals.waterPipeline?.worldReflectionsApplied ?? false;
  report.worldReflectionStatus = internals.svoDryScenePipeline?.waterWorldReflectionStatus;
  if (selectedReflections === "world-space") assert.equal(report.worldReflectionsApplied, true, "World reflections must actually encode before capture");
  snapshotMemory("final-presentation");
  await validateExecution("initial-final-frame");
  const finalWater = await measureWater();
  report.finalWater = finalWater;
  assert.ok(finalWater.total_m3 > 0, "The final live scene must contain real Eulerian water");
  if (definition.id === "forest-waterfall") {
    assert.ok(finalWater.pond_m3 > .8 * initialWater.pond_m3 && finalWater.pond_m3 > 100, "The final live pond must retain its Eulerian water");
  }
  latestInfo = await internals.gpuFluid.readStats();
  check();
  assert.equal(latestInfo.encodedSteps, steps, "Capture must contain exactly the requested whole fixed steps");
  assert.ok(renderer.completedPresentationCount > 0, "No completed production presentation");
  assert.ok(completedTime_s >= targetTime_s - dt * .01, "Capture has not reached the requested live time");
  const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
  const readback = device!.createBuffer({ label: "Forest app screenshot readback", size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const capture = async (path: string) => {
    const encoder = device!.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: swapTexture! }, { buffer: readback, bytesPerRow }, { width, height });
    device!.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const packed=Buffer.alloc(width*height*4),mapped=new Uint8Array(readback.getMappedRange());
    writeBgraPng(path, mapped, bytesPerRow);
    for(let y=0;y<height;y++)packed.set(mapped.subarray(y*bytesPerRow,y*bytesPerRow+width*4),y*width*4);
    readback.unmap();return packed;
    };
    const readGiDiagnostics = async () => {
      const target=device!.createBuffer({size:8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try {
        const encoder=device!.createCommandEncoder();
        const copied=internals.svoDryScenePipeline?.copyGlobalIlluminationDiagnostics(encoder,target);
        if(!copied)return {available:false};
        device!.queue.submit([encoder.finish()]);await target.mapAsync(GPUMapMode.READ);
        const words=new Uint32Array(target.getMappedRange());return {available:true,invalidPages:words[0],exhaustedCones:words[1]};
      } finally {if(target.mapState==="mapped")target.unmap();target.destroy();}
    };
    const readSpecularDiagnostics=async()=>{
      const target=device!.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try{
        const encoder=device!.createCommandEncoder();
        const pipeline=internals.svoDryScenePipeline;
        if(!pipeline)return {available:false};
        pipeline.copyVoxelSpecularDiagnostics(encoder,target);
        device!.queue.submit([encoder.finish()]);await target.mapAsync(GPUMapMode.READ);
        const words=new Uint32Array(target.getMappedRange());return {available:true,traced:words[0],exhausted:words[1],invalidSource:words[2],unsupportedFallback:words[3]};
      }finally{if(target.mapState==="mapped")target.unmap();target.destroy();}
    };
    if(captureVoxelReflections) {
      // Diagnose a fresh producer, not a retained dry frame. This draw is outside
      // every timing sample and uses the same frozen camera and solver state.
      const originalReuse=captureStaticVisibilityReuse;
      captureStaticVisibilityReuse=false;await draw(pausedTime_s);
      report.voxelReflectionDiagnostics=await readSpecularDiagnostics();
      await validateExecution("fresh-voxel-reflection-diagnostics");
      captureStaticVisibilityReuse=originalReuse;await draw(pausedTime_s);
    } else report.voxelReflectionDiagnostics=await readSpecularDiagnostics();
    if(options.has("export-water-mesh")) {
      assert.ok(internals.waterPipeline,"Missing rendered water pipeline");
      const mesh=await internals.waterPipeline.readDenseSurfaceExtractionForQA();
      assert.equal(mesh.vertexCount,mesh.vertexAllocator,"Refuse an overflowed water surface");
      assert.ok(mesh.vertexCount>0&&mesh.vertexCount%3===0&&mesh.vertices.length===mesh.vertexCount*8);
      const {obj,bounds}=waterMeshObj(mesh.vertices,`Captured production ${method.id}: ${sceneId} at ${completedTime_s}s`,{opticalNormals:true});
      const path=pngPath.replace(/\.png$/,".obj");writeFileSync(path,obj);
      const provenance={sceneId,methodId:method.id,completedTime_s,vertexCount:mesh.vertexCount,triangleCount:mesh.vertexCount/3,
        activeCubeCount:mesh.activeCubeCount,boundsBlender_m:bounds,objSha256:createHash("sha256").update(obj).digest("hex"),
        opticalNormals:"Production vertex-normal lane, normalized and rotated to Blender axes",
        extraction:"Exact water vertex buffer used by this captured frame; no second simulation or surface extraction",
        captureReport:reportPath,sourceHashes:report.sourceHashes,errors:[...errors],
        rigidBodies:(await (internals.gpuFluid as WebGPUUniformReferenceSolver).readRigidBodyPoses())?.map((pose,index)=>({...pose,
          id:scene.rigidBodies[index]?.id,appearance:scene.rigidBodies[index]?.appearance}))};
      writeFileSync(path+".json",JSON.stringify(provenance,null,2)+"\n");
      report.exportedWaterMesh={path,provenancePath:path+".json",...provenance};
      await validateExecution("export-rendered-water-mesh");
    }
    report.giDiagnostics=await readGiDiagnostics();
    report.staticFrameReuse=internals.svoDryScenePipeline?.rasterVisibilityCacheStatistics;
    const refractionDiagnostics = async () => {
      const plane = internals.waterPipeline?.diagnosticCaptureTexture("world-refractions");
      assert.ok(plane, "Missing World refraction diagnostic plane");
      const texture=plane.texture,bytesPerPixel=8,rowBytes=texture.width*bytesPerPixel,padded=Math.ceil(rowBytes/256)*256;
      const staging=device!.createBuffer({size:padded*texture.height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try {
        const encoder=device!.createCommandEncoder();
        encoder.copyTextureToBuffer({texture,origin:plane.origin??[0,0,0]},{buffer:staging,bytesPerRow:padded},[texture.width,texture.height,1]);
        device!.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
        const bytes=new DataView(staging.getMappedRange()),distances:number[]=[];
        const counts={inactiveSamples:0,verifiedOpaqueHits:0,opaqueMisses:0,unverifiedLiquidIntervals:0,failedWorldSamples:0,unexpectedSamples:0};
        for(let y=0;y<texture.height;y++)for(let x=0;x<texture.width;x++) {
          const a=bytes.getUint16(y*padded+x*8+6,true);
          if(a===0)counts.inactiveSamples++;
          else if(a===0xc000||(a&0x7c00)===0x7c00)counts.failedWorldSamples++;
          else if(a===0xc200)counts.opaqueMisses++;
          else if(a===0xc400)counts.unverifiedLiquidIntervals++;
          else if((a&0x8000)!==0)counts.unexpectedSamples++;
          else { counts.verifiedOpaqueHits++; const exponent=(a>>10)&31,mantissa=a&1023;distances.push(exponent===0?mantissa*2**-24:(1+mantissa/1024)*2**(exponent-15)); }
        }
        distances.sort((a,b)=>a-b);
        return {counts,verifiedPathLength_m:distances.length?{minimum:distances[0],median:distances[Math.floor(distances.length/2)],maximum:distances.at(-1)}:undefined};
      } finally {if(staging.mapState==="mapped")staging.unmap();staging.destroy();}
    };
    await capture(pngPath);
    if(waterWorldRefractionEnabled)report.worldRefractionSamples=await refractionDiagnostics();
    report.primaryPresentationResolution = renderer.presentationResolution;
    if(options.has("lighting-energy-diagnostics")) {
      const helperPath="./probe-lighting-energy";
      const {probeLightingEnergy}=await import(helperPath);
      const diagnostic=await probeLightingEnergy(device!,internals.svoDryScenePipeline);
      const path=pngPath.replace(/\.png$/, "-lighting-energy-diagnostics.json");
      writeFileSync(path,JSON.stringify(diagnostic));report.lightingEnergyDiagnostics=path;
    }
    if(options.has("world-refinement-diagnostics")) {
      const helperPath="./world-voxel-refinement-probe";
      const {captureWorldVoxelRefinementProbe}=await import(helperPath);
      const diagnostic=await captureWorldVoxelRefinementProbe(device!,internals.svoDryScenePipeline,internals.waterPipeline);
      const path=pngPath.replace(/\.png$/, "-world-refinement-diagnostics.json");
      writeFileSync(path,JSON.stringify(diagnostic));report.worldRefinementDiagnostics={path,counts:diagnostic.counts};
    }
    if(options.has("native-sun-diagnostics")) {
      const {probeNativeSun}=await import("./probe-native-sun");
      const diagnostic=await probeNativeSun(device!,internals.svoDryScenePipeline,internals.viewUniformWords);
      const path=pngPath.replace(/\.png$/, "-native-sun-diagnostics.json");
      writeFileSync(path,JSON.stringify(diagnostic));report.nativeSunDiagnostics=path;
    }
    if(options.has("shadow-diagnostics")) {
      const {probeRasterShadow}=await import("./probe-raster-shadow");
      const tuning=args(pausedTime_s)[12];
      const diagnostic=await probeRasterShadow(device!,internals.svoDryScenePipeline,internals.viewUniformWords,tuning?.shadowBiasCells,tuning?.shadowConeAperture);
      const path=pngPath.replace(/\.png$/, "-shadow-diagnostics.json");
      writeFileSync(path,JSON.stringify(diagnostic));report.shadowDiagnostics=path;
    }

    if (options.has("reflection-diagnostics") || selectedReflections === "world-space") {
      const planes = [];
      const raw = new Map<string, Buffer>();
      const diagnosticStages = options.has("reflection-diagnostics") ? ["scene", "interface-positions", "interfaces", "back-interface-positions", "back-interfaces", "rear-interface-positions", "rear-interfaces", "rear-back-interface-positions", "rear-back-interfaces", "reflections", "rear-reflections"] : ["interface-positions", "reflections"];
      for (const stage of [...diagnosticStages, ...(selectedReflections === "world-space" ? ["world-reflections"] : [])]) {
        const plane = internals.waterPipeline?.diagnosticCaptureTexture(stage);
        if(!plane&&(stage==="reflections"||stage==="rear-reflections")){report.screenReflectionDiagnosticsUnavailable="SSR-only target is not retained by the applied compact World path";continue;}
        assert.ok(plane, `Missing diagnostic plane ${stage}`);
        const texture = plane.texture, pixelBytes = texture.format === "rgba32float" ? 16 : 8;
        const rowBytes = texture.width * pixelBytes, paddedRowBytes = Math.ceil(rowBytes / 256) * 256;
        const staging = device!.createBuffer({ size: paddedRowBytes * texture.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        try {
          const encoder = device!.createCommandEncoder();
          encoder.copyTextureToBuffer({ texture, origin: plane.origin ?? [0,0,0] }, { buffer: staging, bytesPerRow: paddedRowBytes }, [texture.width, texture.height]);
          device!.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
          const padded = new Uint8Array(staging.getMappedRange()), packed = Buffer.alloc(rowBytes * texture.height);
          for (let y = 0; y < texture.height; y++) packed.set(padded.subarray(y * paddedRowBytes, y * paddedRowBytes + rowBytes), y * rowBytes);
          raw.set(stage, packed);
          const path = pngPath.replace(/\.png$/, `-${stage}.bin`);
          if (options.has("reflection-diagnostics")) writeFileSync(path, packed);
          planes.push({ stage, path, width: texture.width, height: texture.height, format: texture.format });
        } finally { if (staging.mapState === "mapped") staging.unmap(); staging.destroy(); }
      }
      if (selectedReflections === "world-space" && raw.has("reflections")) {
        const position = raw.get("interface-positions")!, screen = raw.get("reflections")!, world = raw.get("world-reflections")!;
        const counts = { visibleWaterSamples: 0, retainedScreenHits: 0, addedWorldHits: 0, skyMisses: 0, failedWorldSamples: 0 };
        for (let i = 0; i < screen.length / 8; i++) {
          if (position.readFloatLE(i * 16 + 12) < .5) continue;
          const before = screen.readUInt16LE(i * 8 + 6), after = world.readUInt16LE(i * 8 + 6);
          if (before === 0xbc00) continue; // Hidden interface (-1), not a visible reflection.
          counts.visibleWaterSamples++;
          if ((after & 0x7c00) === 0x7c00 || (after & 0x8000) !== 0) counts.failedWorldSamples++;
          else if (after === 0) counts.skyMisses++;
          else if (before > 0) counts.retainedScreenHits++;
          else counts.addedWorldHits++;
        }
        report.worldReflectionSamples = counts;
        if (counts.failedWorldSamples > 0) console.warn(`World reflection diagnostics: ${counts.failedWorldSamples} visible samples exhausted traversal or shading; these fell back to sky.`);
      }
      if (options.has("reflection-diagnostics")) {
        const diagnostics = { planes, uniforms: [...internals.viewUniformWords], time_s: completedTime_s };
        const path = pngPath.replace(/\.png$/, "-reflection-diagnostics.json");
        writeFileSync(path, JSON.stringify(diagnostics, null, 2)); report.reflectionDiagnostics = path;
      }
    }
    if(options.has("screen-reflection-comparison")) {
      reflectionMode = "screen-space";
      for (let i = 0; i < 4; i++) await draw(pausedTime_s);
      const path = pngPath.replace(/\.png$/, "-screen.png"); await capture(path);
      const samples_ms = [];
      for (let i = 0; i < 12; i++) { const begin = performance.now(); await draw(pausedTime_s); samples_ms.push(performance.now() - begin); }
      const sorted = [...samples_ms].sort((a,b) => a-b);
      report.screenReflectionComparison = { path, frozenTime_s: completedTime_s, samples_ms, median_ms:(sorted[5]!+sorted[6]!)/2 };
    }
    if(options.has("reflection-comparison")) {
      reflectionMode = "environment";
      for(let i=0;i<4;i++) await draw(pausedTime_s);
      const environmentPath=pngPath.replace(/\.png$/, "-sky.png");
      await capture(environmentPath);
      const environmentSamples_ms: number[] = [];
      for (let i = 0; i < 12; i++) {
        const begin = performance.now(); await draw(pausedTime_s);
        environmentSamples_ms.push(performance.now() - begin);
      }
      const ordered = [...environmentSamples_ms].sort((a,b) => a-b);
      report.reflectionComparison = {
        scene: pngPath, environment: environmentPath, frozenTime_s: completedTime_s,
        environmentPausedFrameTiming: {
          measurement: "Host wall time with GPU queue fence; comparison only, not interactive FPS",
          samples_ms: environmentSamples_ms, median_ms: (ordered[5]! + ordered[6]!) / 2,
          width, height, simulationPaused: true,
        },
      };
    }
    if(options.has("voxel-reflection-comparison")) {
      const variants=[],frozenSolver=internals.gpuFluid,frozenTime=completedTime_s;
      for(const enabled of [false,true]){
        captureVoxelReflections=enabled;
        let readyFrames=0;
        while(readyFrames<4){await draw(pausedTime_s);readyFrames=effective?.state==="active"&&lastMetrics?.presentationSubmitted?readyFrames+1:0;progress("settle-voxel-reflections");}
        const samples_ms:number[]=[];
        for(let i=0;i<timingSamples;i++){const start=performance.now();await draw(pausedTime_s);samples_ms.push(performance.now()-start);}
        const path=pngPath.replace(/\.png$/,enabled?"-voxel-reflections.png":"-sky-specular.png");await capture(path);
        await validateExecution(enabled?"voxel-reflections":"sky-specular");
        assert.equal(internals.gpuFluid,frozenSolver);assert.equal(completedTime_s,frozenTime);
        const ordered=[...samples_ms].sort((a,b)=>a-b);
        variants.push({enabled,path,samples_ms,median_ms:(ordered[Math.floor((ordered.length-1)/2)]!+ordered[Math.floor(ordered.length/2)]!)/2,diagnostics:await readSpecularDiagnostics()});
      }
      report.voxelReflectionComparison={steps:voxelReflectionSteps,variants,frozenTime_s:frozenTime,measurement:"GPU-fenced host wall samples; same frozen fluid/camera; not interactive FPS"};
      captureVoxelReflections=voxelReflections;for(let i=0;i<4;i++)await draw(pausedTime_s);
    }
    if(options.has("bloom-comparison")) {
      const variants=[],frozenSolver=internals.gpuFluid,frozenTime=completedTime_s;
      for(const enabled of [false,true]){
        captureBloomEnabled=enabled;
        for(let i=0;i<4;i++)await draw(pausedTime_s);
        assert.equal(internals.waterPipeline?.lensBloomApplied,enabled&&lensBloom.strength>0,"Requested bloom must actually be composed");
        const samples_ms:number[]=[];
        for(let i=0;i<timingSamples;i++){const start=performance.now();await draw(pausedTime_s);samples_ms.push(performance.now()-start);}
        const path=pngPath.replace(/\.png$/,enabled?"-bloom.png":"-bloom-off.png");await capture(path);
        await validateExecution(enabled?"bloom-on":"bloom-off");
        assert.equal(internals.gpuFluid,frozenSolver);assert.equal(completedTime_s,frozenTime);
        const ordered=[...samples_ms].sort((a,b)=>a-b);
        variants.push({enabled,path,samples_ms,median_ms:(ordered[Math.floor((ordered.length-1)/2)]!+ordered[Math.floor(ordered.length/2)]!)/2});
      }
      report.bloomComparison={options:lensBloom,variants,frozenTime_s:frozenTime,measurement:"GPU-fenced host wall samples; same frozen fluid/camera; not interactive FPS"};
      captureBloomEnabled=lensBloom.enabled;for(let i=0;i<4;i++)await draw(pausedTime_s);
    }
    if(options.has("gi-comparison")) {
      const frozenSolver=internals.gpuFluid, frozenTime=completedTime_s;
      const variants=[];
      for(const variant of [
        {name:"raster-direct",visibility:"raster-ao",gi:false,cache:false},
        {name:"cones-direct",visibility:"cones",gi:false,cache:false},
        {name:"cones-gi-uncached",visibility:"cones",gi:true,cache:false},
        {name:"cones-gi-cache",visibility:"cones",gi:true,cache:true},
      ] as const) {
        captureVisibilityMode=variant.visibility;captureIndirectLighting=variant.gi;captureGiCache=variant.cache;
        let readyFrames=0;
        while(readyFrames<4){
          await draw(pausedTime_s);progress(`settle-${variant.name}`);
          readyFrames=effective?.state==="active"&&lastMetrics?.presentationSubmitted ? readyFrames+1 : 0;
        }
        const samples_ms:number[]=[];
        for(let i=0;i<timingSamples;i++){const start=performance.now();await draw(pausedTime_s);samples_ms.push(performance.now()-start);}
        assert.equal(internals.gpuFluid,frozenSolver,"GI comparison replaced the frozen solver");
        assert.equal(completedTime_s,frozenTime,"GI comparison advanced the fluid");
        const path=pngPath.replace(/\.png$/,`-${variant.name}.png`);const pixels=await capture(path);
        await validateExecution(variant.name);
        const ordered=[...samples_ms].sort((a,b)=>a-b);
        const result={...variant,path,samples_ms,median_ms:(ordered[Math.floor((ordered.length-1)/2)]!+ordered[Math.floor(ordered.length/2)]!)/2,
          renderer:effective?.state,worldReflectionsApplied:internals.waterPipeline?.worldReflectionsApplied,
          displaySha256:createHash("sha256").update(pixels).digest("hex"),giDiagnostics:await readGiDiagnostics()};
        variants.push(result);console.log(JSON.stringify({phase:"gi-comparison",...result}));
      }
      report.giComparison={variants,frozenTime_s:frozenTime,solverIdentityPreserved:true,measurement:"GPU-fenced host wall samples, same frozen fluid and camera; not interactive FPS"};
      captureVisibilityMode=visibilityMode;captureIndirectLighting=indirectLighting;captureGiCache=giCache;
      let readyFrames=0;
      while(readyFrames<4){await draw(pausedTime_s);readyFrames=effective?.state==="active"&&lastMetrics?.presentationSubmitted ? readyFrames+1 : 0;}
    }
    if(options.has("lighting-comparison")) {
      const variants = [];
      reflectionMode = selectedReflections;
      for (const [name, shadows, ao] of [["no-ao", true, false], ["no-shadows", false, true], ["unoccluded", false, false]] as const) {
        shadowsEnabled = shadows; ambientOcclusionEnabled = ao;
        for (let i = 0; i < 8; i++) await draw(pausedTime_s);
        const path = pngPath.replace(/\.png$/, `-${name}.png`); await capture(path);
        variants.push({path,shadowsEnabled,ambientOcclusionEnabled});
      }
      shadowsEnabled = DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled;
      ambientOcclusionEnabled = DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled;
      for (let i = 0; i < 8; i++) await draw(pausedTime_s);
      report.lightingComparison = {baseline:pngPath,variants,frozenTime_s:completedTime_s,visibilityMode};
    }
    if(options.has("shadow-resolution-comparison")) {
      assert.equal(visibilityMode,"raster-ao","Shadow resolution comparison requires raster visibility");
      const baselineResolution=captureShadowResolution, frozenSolver=internals.gpuFluid;
      const frozenTime=completedTime_s, frozenSolverTime=frozenSolver?.info.completedTime_s;
      const frozenPipeline=internals.svoDryScenePipeline;
      const assertFrozen=()=>{
        assert.equal(internals.gpuFluid,frozenSolver,"Shadow resolution replaced the live fluid solver");
        assert.equal(internals.svoDryScenePipeline,frozenPipeline,"Shadow resolution rebuilt the scene pipeline");
        assert.equal(completedTime_s,frozenTime,"Shadow comparison advanced fluid time");
        assert.equal(internals.gpuFluid?.info.completedTime_s,frozenSolverTime,"Shadow comparison changed solver completion time");
      };
      const variants=[];
      for(const resolution of [2048,4096] as const){
        captureShadowResolution=resolution;
        for(let i=0;i<8;i++){await draw(pausedTime_s);assertFrozen();}
        const applied=(internals.svoDryScenePipeline as unknown as {rasterAo?:{shadow:GPUTexture}})?.rasterAo?.shadow.width;
        assert.equal(applied,resolution,"Requested shadow resolution was not applied");
        const path=pngPath.replace(/\.png$/,`-shadow-${resolution}.png`);await capture(path);assertFrozen();
        const samples_ms=[];for(let i=0;i<8;i++){const start=performance.now();await draw(pausedTime_s);assertFrozen();samples_ms.push(performance.now()-start);}
        const sorted=[...samples_ms].sort((a,b)=>a-b);variants.push({requested:resolution,applied,path,samples_ms,median_ms:(sorted[3]!+sorted[4]!)/2});
      }
      captureShadowResolution=baselineResolution;for(let i=0;i<4;i++){await draw(pausedTime_s);assertFrozen();}
      report.shadowResolutionComparison={variants,frozenTime_s:frozenTime,solverIdentityPreserved:true,scenePipelineIdentityPreserved:true,measurement:"Host wall time with GPU queue fence; same frozen fluid and voxel casters, not interactive FPS"};
    }
    if(options.has("sun-softness-comparison")) {
      const baselineAperture = captureShadowAperture;
      reflectionMode = selectedReflections; captureShadowAperture = .01;
      for (let i = 0; i < 8; i++) await draw(pausedTime_s);
      const path = pngPath.replace(/\.png$/, "-sun-001.png"); await capture(path);
      report.sunSoftnessComparison = {baseline:pngPath, path, baselineAperture_rad:baselineAperture, aperture_rad:captureShadowAperture, frozenTime_s:completedTime_s};
      captureShadowAperture = baselineAperture;
      for (let i = 0; i < 8; i++) await draw(pausedTime_s);
    }
    if(options.has("world-refraction-comparison")) {
      const baselineEnabled=waterWorldRefractionEnabled;
      const variants=[];reflectionMode=selectedReflections;
      for(const enabled of [false,true]) {
        waterWorldRefractionEnabled=enabled;
        for(let i=0;i<8;i++)await draw(pausedTime_s);
        if(enabled) {
          while(!internals.waterPipeline?.worldRefractionsApplied) { await draw(pausedTime_s); progress("prepare-world-refraction"); }
          for(let i=0;i<4;i++)await draw(pausedTime_s);
        }
        const path=pngPath.replace(/\.png$/, enabled?"-world-refraction.png":"-world-refraction-off.png");await capture(path);
        const samples_ms=[];for(let i=0;i<8;i++){const start=performance.now();await draw(pausedTime_s);samples_ms.push(performance.now()-start);}
        const diagnostics=enabled?await refractionDiagnostics():undefined;
        const sorted=[...samples_ms].sort((a,b)=>a-b);variants.push({enabled,path,samples_ms,median_ms:(sorted[3]!+sorted[4]!)/2,diagnostics});
      }
      report.worldRefractionComparison={variants,frozenTime_s:completedTime_s,measurement:"Host wall time with GPU queue fence; same frozen geometry, not interactive FPS"};
      waterWorldRefractionEnabled=baselineEnabled;for(let i=0;i<4;i++)await draw(pausedTime_s);
    }
    if(options.has("native-ao")||options.has("native-ao-comparison")) {
      const {probeNativeAo}=await import("./probe-native-ao");
      const diagnostic=await probeNativeAo(device!,internals.svoDryScenePipeline,internals.viewUniformWords);
      const diagnosticPath=pngPath.replace(/\.png$/, "-native-ao-diagnostics.json");
      writeFileSync(diagnosticPath,JSON.stringify(diagnostic));report.nativeAoDiagnostics={path:diagnosticPath,counts:diagnostic.counts};
    }
    if(options.has("native-ao-comparison")) {
      const baselineEnabled=nativeAmbientOcclusionEnabled;
      const variants=[];reflectionMode=selectedReflections;
      for(const enabled of [false,true]) {
        nativeAmbientOcclusionEnabled=enabled;
        for(let i=0;i<4;i++)await draw(pausedTime_s);
        const path=pngPath.replace(/\.png$/, enabled?"-native-ao.png":"-native-ao-off.png");await capture(path);
        const samples_ms=[];for(let i=0;i<8;i++){const start=performance.now();await draw(pausedTime_s);samples_ms.push(performance.now()-start);}
        const sorted=[...samples_ms].sort((a,b)=>a-b);variants.push({enabled,path,samples_ms,median_ms:(sorted[3]!+sorted[4]!)/2});
      }
      report.nativeAoComparison={variants,frozenTime_s:completedTime_s,measurement:"Host wall time with GPU queue fence; same frozen geometry, not interactive FPS"};
      nativeAmbientOcclusionEnabled=baselineEnabled;for(let i=0;i<4;i++)await draw(pausedTime_s);
    }
    if(options.has("performance-comparison")) {
      assert.equal(selectedReflections,"world-space","Performance matrix requires World reflections");
      const frozenSolver=internals.gpuFluid,frozenPipeline=internals.svoDryScenePipeline,frozenTime=completedTime_s;
      const originalReuse=captureStaticVisibilityReuse;
      const variants=[
        {name:"reference",reuse:false},
        {name:"visibility-reuse",reuse:true},
      ];
      const entries=new Map<string,{name:string;reuse:boolean;samples_ms:number[];path?:string;pixelMismatchBytes?:number;worldMismatchBytes?:number;sceneMismatchBytes?:number;cache?:unknown}>();
      let referencePixels:Buffer|undefined,referenceWorld:Buffer|undefined,referenceScene:Buffer|undefined;
      const readPlane=async(stage:string)=>{
        const plane=internals.waterPipeline?.diagnosticCaptureTexture(stage);assert.ok(plane,`Missing ${stage}`);
        const texture=plane.texture,row=texture.width*(texture.format==="rgba32float"?16:8),stride=Math.ceil(row/256)*256;
        const staging=device!.createBuffer({size:stride*texture.height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        try{const e=device!.createCommandEncoder();e.copyTextureToBuffer({texture,origin:plane.origin??[0,0,0]},{buffer:staging,bytesPerRow:stride},[texture.width,texture.height]);device!.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);
          const mapped=new Uint8Array(staging.getMappedRange()),out=Buffer.alloc(row*texture.height);for(let y=0;y<texture.height;y++)out.set(mapped.subarray(y*stride,y*stride+row),y*row);return out;
        }finally{if(staging.mapState==="mapped")staging.unmap();staging.destroy();}
      };
      const mismatches=(a:Buffer,b:Buffer)=>{assert.equal(a.length,b.length);let count=0;for(let i=0;i<a.length;i++)if(a[i]!==b[i])count++;return count;};
      const assertFrozen=()=>{
        assert.equal(internals.gpuFluid,frozenSolver);assert.equal(internals.svoDryScenePipeline,frozenPipeline);assert.equal(completedTime_s,frozenTime);
        if(liveIdentity){assert.equal(internals.gpuFluid,liveIdentity.solver);assert.equal(internals.svoDryScenePipeline,liveIdentity.pipeline);assert.equal(internals.gpuFluid!.info.completedTime_s,liveIdentity.time_s);assert.equal(internals.gpuFluid!.info.encodedSteps,liveIdentity.steps);}
      };
      for(const order of [variants,[...variants].reverse()])for(const variant of order){
        captureStaticVisibilityReuse=variant.reuse;
        for(let i=0;i<8;i++){await draw(pausedTime_s);assertFrozen();}
        assert.ok(internals.waterPipeline?.worldReflectionsApplied,"Measured World reflections must actually encode");
        const entry=entries.get(variant.name)??{...variant,samples_ms:[]};
        for(let i=0;i<16;i++){const begin=performance.now();await draw(pausedTime_s);entry.samples_ms.push(performance.now()-begin);assertFrozen();}
        const path=pngPath.replace(/\.png$/,`-${variant.name}.png`),pixels=await capture(path),world=await readPlane("world-reflections"),dryScene=await readPlane("scene");
        if(!referencePixels){referencePixels=pixels;referenceWorld=world;referenceScene=dryScene;}
        entry.path=path;entry.pixelMismatchBytes=Math.max(entry.pixelMismatchBytes??0,mismatches(referencePixels,pixels));entry.worldMismatchBytes=Math.max(entry.worldMismatchBytes??0,mismatches(referenceWorld!,world));entry.sceneMismatchBytes=Math.max(entry.sceneMismatchBytes??0,mismatches(referenceScene!,dryScene));
        await validateExecution(`performance-${variant.name}`);
        entry.cache=internals.svoDryScenePipeline?.rasterVisibilityCacheStatistics;entries.set(variant.name,entry);
        console.log(JSON.stringify({phase:"performance-variant",name:variant.name,samples:entry.samples_ms.length,pixelMismatchBytes:entry.pixelMismatchBytes,worldMismatchBytes:entry.worldMismatchBytes,sceneMismatchBytes:entry.sceneMismatchBytes,cache:entry.cache}));
      }
      if(liveCachedPlanes){
        const differences={display:mismatches(liveCachedPlanes[0]!,referencePixels!),scene:mismatches(liveCachedPlanes[1]!,referenceScene!),world:mismatches(liveCachedPlanes[2]!,referenceWorld!)};
        report.liveCachedMismatchBytes=differences;
        assert.ok(Object.values(differences).every(value=>value===0),"Advancing-water cached image/HDR differs from frozen recomputed reference");
      }
      const results=[...entries.values()].map(entry=>{const ordered=[...entry.samples_ms].sort((a,b)=>a-b);return {...entry,median_ms:(ordered[15]!+ordered[16]!)/2};});
      report.performanceComparison={variants:results,frozenTime_s:frozenTime,solverIdentityPreserved:true,scenePipelineIdentityPreserved:true,measurement:"32 GPU-fenced host wall samples per variant, forward/reverse order; no concurrent tests; not interactive FPS",parity:"Byte equality of displayed BGRA, dry scene HDR and World reflection HDR"};
      captureStaticVisibilityReuse=originalReuse;for(let i=0;i<4;i++)await draw(pausedTime_s);
      assert.ok(results.every(v=>v.pixelMismatchBytes===0&&v.worldMismatchBytes===0&&v.sceneMismatchBytes===0),"Optimization matrix changed frozen image/HDR bytes");
    }
    if(options.has("light-budget-comparison")) {
      const original=lightSamples, frozenSolver=internals.gpuFluid, frozenTime=completedTime_s;
      const variants=[];
      for(const budget of [8,32]) {
        lightSamples=budget;
        for(let i=0;i<8;i++)await draw(pausedTime_s);
        const samples_ms:number[]=[];
        for(let i=0;i<16;i++){const begin=performance.now();await draw(pausedTime_s);samples_ms.push(performance.now()-begin);}
        assert.equal(internals.gpuFluid,frozenSolver);assert.equal(completedTime_s,frozenTime);
        const path=pngPath.replace(/\.png$/,`-lights-${budget}.png`);await capture(path);
        const sorted=[...samples_ms].sort((a,b)=>a-b);
        variants.push({budget,path,samples_ms,median_ms:(sorted[7]!+sorted[8]!)/2});
      }
      report.lightBudgetComparison={variants,frozenTime_s:frozenTime,measurement:"GPU-fenced host wall time; identical frozen solver state; not interactive FPS"};
      lightSamples=original;for(let i=0;i<4;i++)await draw(pausedTime_s);
    }
    if(options.has("texture-comparison")) {
      surfaceTexturesEnabled = false;
      for(let i=0;i<4;i++) await draw(pausedTime_s);
      const path=pngPath.replace(/\.png$/, "-voxel-material.png"); await capture(path);
      report.textureComparison={ textured:pngPath, voxelMaterial:path, frozenTime_s:completedTime_s, geometry:"Same FVX/SVO scalar samples and frozen fluid" };
      surfaceTexturesEnabled = true;
    }
    if(options.has("supersampling-comparison")) {
      reflectionMode = selectedReflections; captureRenderScale = 2;
      for(let i=0;i<4;i++) await draw(pausedTime_s);
      const samples_ms: number[] = [];
      for(let i=0;i<12;i++) { const begin=performance.now(); await draw(pausedTime_s); samples_ms.push(performance.now()-begin); }
      const path=pngPath.replace(/\.png$/, "-2x.png"); await capture(path);
      snapshotMemory("supersampled-presentation");
      const ordered=[...samples_ms].sort((a,b)=>a-b);
      report.supersamplingComparison={ native:pngPath, supersampled:path, frozenTime_s:completedTime_s,
        presentationResolution:renderer.presentationResolution, samples_ms, median_ms:(ordered[5]!+ordered[6]!)/2,
        measurement:"Paused host wall time with GPU queue fence; not interactive FPS" };
    }
  } finally { readback.destroy(); }
  if(options.has("profile-passes")) {
    const traces=new Map<number,NonNullable<RendererFrameMetrics["presentation"]>>();
    usePerformanceInstrumentationStore.getState().setMode("timeline");
    try {
      for(let i=0;i<36;i++){
        await draw(pausedTime_s);
        if(lastMetrics?.presentation)traces.set(lastMetrics.presentation.sampleId,lastMetrics.presentation);
      }
    } finally {usePerformanceInstrumentationStore.getState().setMode("off");}
    report.presentationPassProfiles=[...traces.values()];
    report.presentationPassProfileNote="Instrumented frames collected separately after wall timing; Metal encoder boundaries may perturb timings.";
  }
  report.finalPresentationSettings = {lensBloom:{...lensBloom,enabled:captureBloomEnabled},lensBloomApplied:internals.waterPipeline?.lensBloomApplied,voxelReflections:captureVoxelReflections,voxelReflectionSteps,giCache:captureGiCache,latticeEnabled,shadowResolutionRequested:captureShadowResolution, shadowResolutionApplied:(internals.svoDryScenePipeline as unknown as {rasterAo?:{shadow:GPUTexture}})?.rasterAo?.shadow.width, reflectionMode, renderScale: captureRenderScale, visibilityMode, indirectLighting, surfaceTexturesEnabled, nativeAmbientOcclusionEnabled };
  latestInfo = await internals.gpuFluid.readStats();
  assert.equal(latestInfo.encodedSteps, steps, "Rendering comparisons must not advance or reset the frozen fluid");
  await validateExecution("all-comparisons-complete");
  const finalComparisonWater=await measureWater();report.waterAfterComparisons=finalComparisonWater;
  assert.ok(finalComparisonWater.total_m3>0,"Render comparisons must retain real fluid readbacks");
  check(); successful = true;
  progress("captured-production-live-scene", true);
} catch (error) {
  failure = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  if(error instanceof Error)report.failureStack=error.stack;
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
