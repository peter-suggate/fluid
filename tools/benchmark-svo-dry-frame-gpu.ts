#!/usr/bin/env node
/**
 * Headless end-to-end GPU benchmark for the production SVO dry-scene render
 * pass (SparseVoxelDrySceneRenderer.encode) bound to the shipped garden
 * lighting-study world (WebGPUStaticSvoScene over OctreeSparseBrickWorld).
 *
 * Runs on Dawn/Metal via the `webgpu` node module. Reports per-frame GPU pass
 * time (timestamp queries when available, otherwise submit->fence wall time)
 * and a deterministic visual-parity fingerprint of the rendered frame.
 *
 * Rerun: node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
 * Env: FLUID_SVO_DRY_FRAME_WIDTH / _HEIGHT / _WARMUPS / _CYCLES /
 *      _ENCODES_PER_SAMPLE, WEBGPU_NODE_MODULE, FLUID_SVO_DRY_FRAME_OUT.
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { environmentIndex, type EnvironmentId } from "../lib/environments";
import { cameraPosition } from "../lib/math";
import { defaultCamera, type CameraState, type SceneDescription } from "../lib/model";
import { boundingRadius, initializeRigidBodies } from "../lib/rigid-body";
import { getScenePreset } from "../lib/scenes";
import { buildSvoSceneGlass } from "../lib/svo-scene-glass";
import { buildSvoScenePrimitives } from "../lib/svo-scene-primitives";
import { buildSvoSceneThickGlass } from "../lib/svo-scene-thick-glass";
import { buildSvoTerrainMaterial } from "../lib/svo-terrain-material";
import { MAX_TERRAIN_FEATURES, sceneHasTerrain, TERRAIN_DEFAULT_FLAT, TERRAIN_UNION_EXPONENT } from "../lib/terrain";
import { optionalFluidDeviceFeatures, requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { WebGPUStaticSvoScene } from "../lib/webgpu-static-svo-scene";
import {
  buildSparseVoxelDrySceneLightingMirrors,
  canConsumeSparseVoxelPbrMaterials,
  canConsumeSparseVoxelPrimitiveCandidates,
  canEncodeSparseVoxelDryScene,
  resolveSparseVoxelThickGlassBinderStatus,
  SparseVoxelDrySceneRenderer,
  type SparseVoxelDrySceneData,
} from "../lib/webgpu-svo-dry-scene";
import { SVO_GBUFFER_RENDER_TARGET_CONTRACT } from "../lib/webgpu-svo-gbuffer-targets";

const width = Number(process.env.FLUID_SVO_DRY_FRAME_WIDTH ?? 1280);
const height = Number(process.env.FLUID_SVO_DRY_FRAME_HEIGHT ?? 720);
const warmups = Number(process.env.FLUID_SVO_DRY_FRAME_WARMUPS ?? 4);
const cycles = Number(process.env.FLUID_SVO_DRY_FRAME_CYCLES ?? 16);
const encodesPerSample = Number(process.env.FLUID_SVO_DRY_FRAME_ENCODES_PER_SAMPLE ?? 1);
const outPath = process.env.FLUID_SVO_DRY_FRAME_OUT ?? "/tmp/svo-bench/baseline.json";
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
assert.ok(Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0);
assert.ok(Number.isSafeInteger(warmups) && warmups >= 0 && Number.isSafeInteger(cycles) && cycles > 0);
assert.ok(Number.isSafeInteger(encodesPerSample) && encodesPerSample > 0);

const log = (message: string) => process.stderr.write(`${message}\n`);

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}
function percentile95(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(0.95 * sorted.length) - 1)];
}
function decodeF16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 31) return mantissa ? Number.NaN : sign * Infinity;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}
function fnv1a32(words: Uint32Array, seed = 0x811c9dc5): number {
  let hash = seed >>> 0;
  for (let index = 0; index < words.length; index += 1) {
    const word = words[index];
    for (let byte = 0; byte < 4; byte += 1) {
      hash ^= (word >>> (byte * 8)) & 0xff;
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  }
  return hash >>> 0;
}

/** Mirror of the FluidLabRenderer 400-byte view-uniform packing (webgpu-renderer.ts). */
function packViewUniforms(
  scene: SceneDescription,
  camera: CameraState,
  environmentId: EnvironmentId,
  info: { nx: number; ny: number; nz: number },
  bodyCount: number,
): Float32Array<ArrayBuffer> {
  const position = cameraPosition(camera);
  // options.x: DEFAULT_SVO_RENDER_DIAGNOSTICS maximumTraversalDepth(21)*512 + maximumNodeVisits(256).
  const diagnosticControl = 21 * 512 + 256;
  // viewport.w: svoDrySceneTemporalFrame with a stable camera and no temporal
  // accumulation eligibility (shadowTemporalFrame = -1) -> -1.
  const temporalFrame = -1;
  const uniform = new Float32Array([
    width, height, 0, temporalFrame,
    position.x, position.y, position.z, 0,
    camera.target_m.x, camera.target_m.y, camera.target_m.z, 0.82,
    scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction,
    diagnosticControl, scene.voxelDomain.finestCellSize_m, Math.min(bodyCount, 12), 1,
    info.nx, info.ny, info.nz, 3,
    0, 0.5, 1, 0,
    environmentIndex(environmentId), 0, 0, 0,
  ]);
  const packed = new Float32Array(100);
  packed.set(uniform, 0);
  if (sceneHasTerrain(scene) && scene.terrain) {
    const terrain = scene.terrain;
    const features = terrain.features.slice(0, MAX_TERRAIN_FEATURES);
    packed.set([1, terrain.baseHeight_m, features.length, TERRAIN_UNION_EXPONENT], 32);
    features.forEach((feature, index) => {
      packed.set([feature.center_m.x, feature.center_m.z, feature.radius_m.x, feature.radius_m.z], 36 + index * 8);
      packed.set([(feature.kind === "mound" ? 1 : -1) * feature.amount_m, feature.rotation_rad ?? 0, feature.flat ?? TERRAIN_DEFAULT_FLAT, 0], 40 + index * 8);
    });
  }
  return packed;
}

/** Mirror of the FluidLabRenderer CPU rigid-body packing (webgpu-renderer.ts). */
function packBodies(scene: SceneDescription): { data: Float32Array<ArrayBuffer>; count: number } {
  const bodies = initializeRigidBodies(scene.rigidBodies);
  const bodyData = new Float32Array(12 * 16);
  const shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
  const palette = [[0.95, 0.63, 0.29], [0.48, 0.66, 0.96], [0.84, 0.42, 0.48], [0.66, 0.52, 0.92]];
  bodies.slice(0, 12).forEach((body, index) => {
    const offset = index * 16;
    const d = body.description.dimensions_m;
    const half = body.description.shape === "box" ? [d.x / 2, d.y / 2, d.z / 2] : body.description.shape === "sphere" ? [d.x, d.x, d.x] : [d.x, d.y / 2, d.x];
    const color = palette[shapeIndex[body.description.shape]];
    bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, boundingRadius(body)], offset);
    bodyData.set([half[0], half[1], half[2], shapeIndex[body.description.shape]], offset + 4);
    bodyData.set([body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z], offset + 8);
    bodyData.set([color[0], color[1], color[2], 0], offset + 12);
  });
  return { data: bodyData, count: bodies.length };
}

// ---------------------------------------------------------------------------
// GPU bring-up on Dawn/Metal.
// ---------------------------------------------------------------------------
const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter, "no Metal adapter — benchmark did not execute on the GPU");
const adapterInfo = {
  vendor: adapter.info?.vendor ?? "",
  architecture: adapter.info?.architecture ?? "",
  device: adapter.info?.device ?? "",
  description: adapter.info?.description ?? "",
};
assert.ok(adapterInfo.vendor || adapterInfo.architecture || adapterInfo.description,
  "adapter info is empty — refusing to report a benchmark that may not have run on real hardware");
const timestampsSupported = adapter.features.has("timestamp-query");
const device = await adapter.requestDevice({
  requiredFeatures: [
    ...(timestampsSupported ? ["timestamp-query" as GPUFeatureName] : []),
    ...optionalFluidDeviceFeatures(adapter.features),
  ],
  requiredLimits: requiredFluidDeviceLimits(adapter.limits),
});
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push((event as GPUUncapturedErrorEvent).error.message));
log(`Adapter: ${JSON.stringify(adapterInfo)} timestamps=${timestampsSupported}`);

// ---------------------------------------------------------------------------
// Build the shipped garden lighting-study world (sparse bricks + node-mip
// pyramid + wide fanout) and the exact production dry-scene data.
// ---------------------------------------------------------------------------
const preset = getScenePreset("garden-svo-lighting");
const scene = preset.create();
const camera: CameraState = { ...defaultCamera, ...preset.camera, target_m: { ...(preset.camera?.target_m ?? defaultCamera.target_m) } };
const environmentId: EnvironmentId = (scene.environment ?? "default") as EnvironmentId;

const solver = await WebGPUStaticSvoScene.create(device, scene, "balanced", ({ label, completed, total }) => log(`  [world] ${label} (${completed}/${total})`));
const source = solver.sparseVoxelSceneSource;
assert.ok(source?.structural, "static SVO world did not publish a structural scene source");

// Exact mirror of FluidLabRenderer solver-attachment dry-scene data assembly.
const scenePrimitives = buildSvoScenePrimitives(scene);
const sceneGlass = buildSvoSceneGlass(scene, { cellSize_m: source.structural!.domain.cellSize_m });
const sceneThickGlass = buildSvoSceneThickGlass(scene);
const thickReplacedPaneKey = sceneThickGlass.metadata.find(({ replacesThinPaneKey }) => Boolean(replacesThinPaneKey))?.replacesThinPaneKey;
const thickReplacedPaneId = sceneGlass.metadata.find(({ key }) => key === thickReplacedPaneKey)?.paneId;
const terrainMaterial = scenePrimitives.analyticTerrain ? buildSvoTerrainMaterial(scene) : undefined;
const compositorOwnedGlass = sceneGlass.metadata.filter(({ role }) => role === "container-pane" || role === "container-top");
const lightingMirrors = buildSparseVoxelDrySceneLightingMirrors(scene, source);
const drySceneData: SparseVoxelDrySceneData = {
  primitiveRecords: scenePrimitives.packedRecords,
  primitiveCandidates: scenePrimitives.primitiveCandidates,
  ownerBase: scene.rigidBodies.length,
  skippedOwnerId: scenePrimitives.openShellOwnerId,
  terrainMaterialId: scenePrimitives.analyticTerrain?.materialId,
  terrainMaterialMetadata: terrainMaterial?.packedMetadata,
  terrainMaterialCacheKey: terrainMaterial?.cacheKey,
  glassRecords: sceneGlass.packedRecords,
  glassCacheKey: sceneGlass.cacheKey,
  thickGlassRecords: sceneThickGlass.packedRecords,
  thickGlassRevision: sceneThickGlass.revision,
  thickGlassCacheKey: sceneThickGlass.cacheKey,
  thickGlassReplacedThinPaneId: thickReplacedPaneId,
  primaryCompositeOwnedGlassPaneIdBase: compositorOwnedGlass[0]?.paneId,
  primaryCompositeOwnedGlassPaneCount: compositorOwnedGlass.length,
  ...lightingMirrors,
};
assert.equal(scenePrimitives.requiresRasterTerrainFallback, false, "garden terrain must render analytically");
assert.ok(canConsumeSparseVoxelPbrMaterials(source), "PBR material publication unavailable");
assert.ok(canConsumeSparseVoxelPrimitiveCandidates(drySceneData), "primitive candidate BVH unavailable");
assert.ok(canEncodeSparseVoxelDryScene(source, drySceneData), "production dry-scene contract rejected the garden source");
const nodeMip = source.nodeMipPyramid;
const coneMipReady = Boolean(nodeMip && nodeMip.generation > 0 && nodeMip.plan.complete);
assert.ok(coneMipReady, "node-mip pyramid unavailable — cone lighting would silently fall back to exact rays");

// ---------------------------------------------------------------------------
// Production renderer, camera uniforms, offscreen targets.
// ---------------------------------------------------------------------------
const uniformBuffer = device.createBuffer({ label: "Bench view uniforms", size: 400, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const bodyBuffer = device.createBuffer({ label: "Bench rigid bodies", size: 12 * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const bodies = packBodies(scene);
device.queue.writeBuffer(uniformBuffer, 0, packViewUniforms(scene, camera, environmentId, solver.info, bodies.count));
device.queue.writeBuffer(bodyBuffer, 0, bodies.data);

const renderer = new SparseVoxelDrySceneRenderer(device, uniformBuffer, bodyBuffer);
await renderer.initialize((label, completed, total) => log(`  [pipeline] ${label} (${completed}/${total})`));
renderer.setLightingMode("cone");
renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true });
renderer.setSource(source, drySceneData);
renderer.ensureSize(width, height);

const target = device.createTexture({
  label: "Bench dry-scene radianceDepth target",
  size: [width, height],
  format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
});

function encodeFrame(encoder: GPUCommandEncoder, timestampWrites?: { querySet: GPUQuerySet; beginningOfPassWriteIndex: number; endOfPassWriteIndex: number }): void {
  const result = renderer.encode(encoder, target, timestampWrites);
  assert.ok(result && result.encoded, "production dry-scene encode declined the frame (raster fallback)");
}

// Warmup + first-frame validation.
for (let index = 0; index < Math.max(1, warmups); index += 1) {
  const encoder = device.createCommandEncoder({ label: `Bench warmup ${index}` });
  encodeFrame(encoder);
  device.queue.submit([encoder.finish()]);
}
await device.queue.onSubmittedWorkDone();
assert.deepEqual(validationErrors, [], "GPU validation errors during warmup");
log(`Warmup complete (${Math.max(1, warmups)} frames)`);

// ---------------------------------------------------------------------------
// Timing.
// ---------------------------------------------------------------------------
const samples: number[] = [];
let timingMethod: string;
if (timestampsSupported) {
  timingMethod = "gpu-timestamp-query-per-render-pass";
  const querySet = device.createQuerySet({ type: "timestamp", count: cycles * 2 });
  const resolve = device.createBuffer({ size: cycles * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const timing = device.createBuffer({ size: cycles * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  for (let cycle = 0; cycle < cycles; cycle += 1) {
    const encoder = device.createCommandEncoder({ label: `Bench cycle ${cycle}` });
    encodeFrame(encoder, { querySet, beginningOfPassWriteIndex: cycle * 2, endOfPassWriteIndex: cycle * 2 + 1 });
    device.queue.submit([encoder.finish()]);
    // Serialize samples: concurrent in-flight passes overlap on Metal and
    // would inflate each pass's begin/end timestamp span with queue depth.
    await device.queue.onSubmittedWorkDone();
  }
  const encoder = device.createCommandEncoder({ label: "Bench timestamp resolve" });
  encoder.resolveQuerySet(querySet, 0, cycles * 2, resolve, 0);
  encoder.copyBufferToBuffer(resolve, 0, timing, 0, cycles * 16);
  device.queue.submit([encoder.finish()]);
  await timing.mapAsync(GPUMapMode.READ);
  const stamps = new BigUint64Array(timing.getMappedRange());
  for (let cycle = 0; cycle < cycles; cycle += 1) {
    samples.push(Number(stamps[cycle * 2 + 1] - stamps[cycle * 2]) / 1e6);
  }
  timing.unmap();
  querySet.destroy(); resolve.destroy(); timing.destroy();
} else {
  timingMethod = `submit-to-onSubmittedWorkDone-wall-time-over-${encodesPerSample}-encodes`;
  for (let cycle = 0; cycle < cycles; cycle += 1) {
    const encoder = device.createCommandEncoder({ label: `Bench cycle ${cycle}` });
    for (let repeat = 0; repeat < encodesPerSample; repeat += 1) encodeFrame(encoder);
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    samples.push((performance.now() - start) / encodesPerSample);
  }
}
assert.equal(samples.length, cycles);
assert.deepEqual(validationErrors, [], "GPU validation errors during timing");

// ---------------------------------------------------------------------------
// Visual-parity fingerprint: one more frame, full-image hash plus a 16x16
// deterministic grid of RGBA radianceDepth samples (decoded f16 values).
// ---------------------------------------------------------------------------
const bytesPerPixel = 8; // rgba16float
const bytesPerRow = Math.ceil(width * bytesPerPixel / 256) * 256;
const readback = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
{
  const encoder = device.createCommandEncoder({ label: "Bench fingerprint frame" });
  encodeFrame(encoder);
  encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow, rowsPerImage: height }, [width, height]);
  device.queue.submit([encoder.finish()]);
}
await readback.mapAsync(GPUMapMode.READ);
const mapped = new Uint8Array(readback.getMappedRange());
const packedRows = new Uint32Array((width * bytesPerPixel * height) / 4);
for (let row = 0; row < height; row += 1) {
  const rowBytes = mapped.subarray(row * bytesPerRow, row * bytesPerRow + width * bytesPerPixel);
  packedRows.set(new Uint32Array(rowBytes.slice().buffer), (row * width * bytesPerPixel) / 4);
}
const imageHash = fnv1a32(packedRows);
const halfWords = new Uint16Array(packedRows.buffer);
const gridSize = 16;
const gridSamples: Array<{ x: number; y: number; rgba: [number, number, number, number] }> = [];
for (let gy = 0; gy < gridSize; gy += 1) {
  for (let gx = 0; gx < gridSize; gx += 1) {
    const x = Math.min(width - 1, Math.floor(((gx + 0.5) / gridSize) * width));
    const y = Math.min(height - 1, Math.floor(((gy + 0.5) / gridSize) * height));
    const base = (y * width + x) * 4;
    gridSamples.push({
      x, y,
      rgba: [
        decodeF16(halfWords[base]), decodeF16(halfWords[base + 1]),
        decodeF16(halfWords[base + 2]), decodeF16(halfWords[base + 3]),
      ],
    });
  }
}
readback.unmap();
const litSamples = gridSamples.filter(({ rgba }) => rgba[0] + rgba[1] + rgba[2] > 0).length;
assert.ok(litSamples > gridSize * gridSize * 0.25,
  `only ${litSamples}/${gridSize * gridSize} fingerprint samples carry radiance — the frame looks empty`);
assert.deepEqual(validationErrors, [], "GPU validation errors during fingerprint frame");

// ---------------------------------------------------------------------------
// Report.
// ---------------------------------------------------------------------------
const result = {
  phase: "svo-dry-frame-gpu-benchmark",
  adapter: adapterInfo,
  backend: "metal",
  resolution: { width, height },
  timing: {
    method: timingMethod,
    warmups: Math.max(1, warmups),
    cycles,
    median_ms: median(samples),
    p95_ms: percentile95(samples),
    samples_ms: samples,
  },
  scene: {
    presetId: "garden-svo-lighting",
    sceneId: scene.sceneId,
    environment: environmentId,
    quality: "balanced",
    lightingMode: "cone",
    shadowsEnabled: true,
    ambientOcclusionEnabled: true,
    temporalAccumulation: false,
    grid: { nx: solver.info.nx, ny: solver.info.ny, nz: solver.info.nz },
    brickSize: source.structural!.domain.brickSize,
    maximumDepth: source.structural!.domain.maximumDepth,
    structuralCapacities: source.structural!.capacities,
    primitiveCount: scenePrimitives.packedRecords.byteLength / 64,
    glassPaneCount: sceneGlass.metadata.length,
    thickGlassStatus: resolveSparseVoxelThickGlassBinderStatus(drySceneData),
    lightCount: source.lights?.count ?? 0,
    rigidBodyCount: scene.rigidBodies.length,
    terrain: Boolean(scene.terrain),
    nodeMipPyramid: { ready: coneMipReady, generation: nodeMip?.generation ?? 0, pages: nodeMip?.plan.pages.length ?? 0 },
    wideFanout: Boolean(source.wideFanout),
    allocatedBytes: solver.info.allocatedBytes,
  },
  camera,
  fingerprint: {
    contract: "16x16 grid of RGBA radianceDepth (rgba16float, decoded to f32) at pixel centers of a uniform grid, plus FNV-1a-32 over the full packed image bytes; bit-exact reproduction expected on identical hardware/driver, otherwise compare grid values within 1e-3 absolute",
    imageHashFnv1a32: `0x${imageHash.toString(16).padStart(8, "0")}`,
    litSampleCount: litSamples,
    gridSize,
    samples: gridSamples,
  },
};
mkdirSync(path.dirname(outPath), { recursive: true });
writeFileSync(outPath, `${JSON.stringify(result, null, 2)}\n`);
log(`Baseline written to ${outPath}`);
console.log(JSON.stringify(result, null, 2));

renderer.destroy();
target.destroy();
readback.destroy();
uniformBuffer.destroy();
bodyBuffer.destroy();
solver.destroy();
device.destroy();
