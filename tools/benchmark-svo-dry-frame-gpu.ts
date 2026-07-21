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
 * With FLUID_SVO_DRY_FRAME_CONE_SCALE < 1 (default 0.5) the reduced-rate
 * cone-lighting prepass is enabled and the tool additionally reports, in one
 * process: interleaved A/B GPU medians (inline reference vs reduced), per-pixel
 * relative luminance error stats over lit pixels, the guided-upsample fallback
 * band percentage, and a full/noAO/noShadows/neither attribution at the
 * reduced rate. Reference, reduced, and amplified-difference PNGs are written
 * next to the JSON report.
 *
 * Rerun: node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
 * Env: FLUID_SVO_DRY_FRAME_WIDTH / _HEIGHT / _WARMUPS / _CYCLES /
 *      _ENCODES_PER_SAMPLE / _CONE_SCALE (1 | 0.5 | 0.25, default 0.5),
 *      FLUID_SVO_DRY_FRAME_SHADOWS / _AO, WEBGPU_NODE_MODULE,
 *      FLUID_SVO_DRY_FRAME_OUT.
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import zlib from "node:zlib";

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
  svoConePrepassSize,
  type SparseVoxelDrySceneData,
  type SvoConeLightingScale,
} from "../lib/webgpu-svo-dry-scene";
import { SVO_GBUFFER_RENDER_TARGET_CONTRACT } from "../lib/webgpu-svo-gbuffer-targets";

const width = Number(process.env.FLUID_SVO_DRY_FRAME_WIDTH ?? 1280);
const height = Number(process.env.FLUID_SVO_DRY_FRAME_HEIGHT ?? 720);
const warmups = Number(process.env.FLUID_SVO_DRY_FRAME_WARMUPS ?? 4);
const cycles = Number(process.env.FLUID_SVO_DRY_FRAME_CYCLES ?? 16);
const encodesPerSample = Number(process.env.FLUID_SVO_DRY_FRAME_ENCODES_PER_SAMPLE ?? 1);
const outPath = process.env.FLUID_SVO_DRY_FRAME_OUT ?? "/tmp/svo-bench/baseline.json";
const coneScaleRaw = Number(process.env.FLUID_SVO_DRY_FRAME_CONE_SCALE ?? 0.5);
const shadowsEnabled = process.env.FLUID_SVO_DRY_FRAME_SHADOWS !== "0";
const ambientOcclusionEnabled = process.env.FLUID_SVO_DRY_FRAME_AO !== "0";
/**
 * M1 Max 1280x720 scale-1 baseline; scale 1 must keep the WGSL byte-identical.
 * Re-baselined for the band-limited cone LOD blend
 * (SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH): the marcher now blends the two
 * bracketing mip levels only inside the trailing fract(lod) transition band
 * (C0 at both band edges) instead of over the full fract range, which alters
 * bits everywhere while keeping the cone-banding fix (no rings, no
 * self-occlusion bands, no hard emitter disc).
 */
const REFERENCE_IMAGE_HASH = 0x211f5930;
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
assert.ok(Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0);
assert.ok(Number.isSafeInteger(warmups) && warmups >= 0 && Number.isSafeInteger(cycles) && cycles > 0);
assert.ok(Number.isSafeInteger(encodesPerSample) && encodesPerSample > 0);
assert.ok([1, 0.5, 0.25].includes(coneScaleRaw), "FLUID_SVO_DRY_FRAME_CONE_SCALE must be 1, 0.5, or 0.25");
const coneScale = coneScaleRaw as SvoConeLightingScale;

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

// ---------------------------------------------------------------------------
// Minimal dependency-free PNG encoder (8-bit RGB, filter 0, one IDAT chunk).
// ---------------------------------------------------------------------------
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();
function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index += 1) crc = CRC_TABLE[(crc ^ bytes[index]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}
function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const chunk = new Uint8Array(12 + data.length);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, data.length);
  for (let index = 0; index < 4; index += 1) chunk[4 + index] = type.charCodeAt(index);
  chunk.set(data, 8);
  view.setUint32(8 + data.length, crc32(chunk.subarray(4, 8 + data.length)));
  return chunk;
}
function encodePng(imageWidth: number, imageHeight: number, rgb: Uint8Array): Buffer {
  assert.equal(rgb.length, imageWidth * imageHeight * 3);
  const header = new Uint8Array(13);
  const headerView = new DataView(header.buffer);
  headerView.setUint32(0, imageWidth);
  headerView.setUint32(4, imageHeight);
  header.set([8, 2, 0, 0, 0], 8); // 8-bit, truecolor RGB
  const raw = new Uint8Array(imageHeight * (imageWidth * 3 + 1));
  for (let row = 0; row < imageHeight; row += 1) {
    raw[row * (imageWidth * 3 + 1)] = 0;
    raw.set(rgb.subarray(row * imageWidth * 3, (row + 1) * imageWidth * 3), row * (imageWidth * 3 + 1) + 1);
  }
  return Buffer.concat([
    Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", header),
    pngChunk("IDAT", new Uint8Array(zlib.deflateSync(raw, { level: 6 }))),
    pngChunk("IEND", new Uint8Array(0)),
  ]);
}
function toneByte(linear: number): number {
  return Math.max(0, Math.min(255, Math.round(255 * Math.min(1, Math.max(0, linear)) ** (1 / 2.2))));
}

/** Mirror of the FluidLabRenderer 400-byte view-uniform packing (webgpu-renderer.ts). */
function packViewUniforms(
  scene: SceneDescription,
  camera: CameraState,
  environmentId: EnvironmentId,
  info: { nx: number; ny: number; nz: number },
  bodyCount: number,
  overlay?: { mode: number; opacity: number },
): Float32Array<ArrayBuffer> {
  const position = cameraPosition(camera);
  // options.x: DEFAULT_SVO_RENDER_DIAGNOSTICS maximumTraversalDepth(21)*512 + maximumNodeVisits(256).
  const diagnosticControl = 21 * 512 + 256;
  // viewport.w: svoDrySceneTemporalFrame with a stable camera and no temporal
  // accumulation eligibility (shadowTemporalFrame = -1) -> -1.
  const temporalFrame = -1;
  const uniform = new Float32Array([
    width, height, 0, temporalFrame,
    position.x, position.y, position.z, overlay?.mode ?? 0,
    camera.target_m.x, camera.target_m.y, camera.target_m.z, overlay ? overlay.opacity : 0.82,
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
renderer.setLightingMode(process.env.FLUID_SVO_DRY_FRAME_LIGHTING === "direct" ? "direct" : "cone");
function applyLighting(scale: SvoConeLightingScale, shadows = shadowsEnabled, ambientOcclusion = ambientOcclusionEnabled): void {
  renderer.setLightingOptions({ shadowsEnabled: shadows, ambientOcclusionEnabled: ambientOcclusion, coneLightingScale: scale });
}
applyLighting(coneScale);
if (coneScale !== 1) {
  await renderer.ensureConeLightingPrepass();
  log(`Cone-lighting prepass ready at scale ${coneScale} (${svoConePrepassSize(width, height, coneScale).join("x")})`);
}
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

// Warmup + first-frame validation for every variant this process will time.
for (let index = 0; index < Math.max(1, warmups); index += 1) {
  const encoder = device.createCommandEncoder({ label: `Bench warmup ${index}` });
  encodeFrame(encoder);
  device.queue.submit([encoder.finish()]);
}
if (coneScale !== 1) {
  applyLighting(1);
  for (let index = 0; index < Math.max(1, warmups); index += 1) {
    const encoder = device.createCommandEncoder({ label: `Bench reference warmup ${index}` });
    encodeFrame(encoder);
    device.queue.submit([encoder.finish()]);
  }
  applyLighting(coneScale);
}
await device.queue.onSubmittedWorkDone();
assert.deepEqual(validationErrors, [], "GPU validation errors during warmup");
log(`Warmup complete (${Math.max(1, warmups)} frames per variant)`);

// ---------------------------------------------------------------------------
// Timing helpers. Samples are serialized (submit -> fence) because concurrent
// in-flight passes overlap on Metal and would inflate each pass's span.
// ---------------------------------------------------------------------------
let timingMethod: string;
const sharedQuerySet = timestampsSupported ? device.createQuerySet({ type: "timestamp", count: 2 }) : undefined;
const sharedResolve = timestampsSupported ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : undefined;
const sharedTiming = timestampsSupported ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : undefined;
async function timeFrames(count: number, label: string): Promise<number[]> {
  const samples: number[] = [];
  if (timestampsSupported) {
    timingMethod = "gpu-timestamp-query-per-render-pass";
    const querySet = sharedQuerySet!, resolve = sharedResolve!, timing = sharedTiming!;
    for (let cycle = 0; cycle < count; cycle += 1) {
      const encoder = device.createCommandEncoder({ label: `${label} cycle ${cycle}` });
      encodeFrame(encoder, { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 });
      encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, timing, 0, 16);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await timing.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(timing.getMappedRange());
      samples.push(Number(stamps[1] - stamps[0]) / 1e6);
      timing.unmap();
    }
  } else {
    timingMethod = `submit-to-onSubmittedWorkDone-wall-time-over-${encodesPerSample}-encodes`;
    for (let cycle = 0; cycle < count; cycle += 1) {
      const encoder = device.createCommandEncoder({ label: `${label} cycle ${cycle}` });
      for (let repeat = 0; repeat < encodesPerSample; repeat += 1) encodeFrame(encoder);
      const start = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      samples.push((performance.now() - start) / encodesPerSample);
    }
  }
  return samples;
}

timingMethod = timestampsSupported ? "gpu-timestamp-query-per-render-pass" : `submit-to-onSubmittedWorkDone-wall-time-over-${encodesPerSample}-encodes`;
const samples = await timeFrames(cycles, "Bench");
assert.equal(samples.length, cycles);
assert.deepEqual(validationErrors, [], "GPU validation errors during timing");

// ---------------------------------------------------------------------------
// Interleaved A/B (reference inline cones vs reduced-rate prepass) in one
// process, alternating every cycle so thermal drift cancels.
// ---------------------------------------------------------------------------
let interleaved: { reference_ms: number[]; reduced_ms: number[] } | undefined;
if (coneScale !== 1) {
  const reference_ms: number[] = [];
  const reduced_ms: number[] = [];
  for (let cycle = 0; cycle < cycles; cycle += 1) {
    applyLighting(1);
    reference_ms.push((await timeFrames(1, `A/B reference ${cycle}`))[0]);
    applyLighting(coneScale);
    reduced_ms.push((await timeFrames(1, `A/B reduced ${cycle}`))[0]);
  }
  interleaved = { reference_ms, reduced_ms };
  assert.deepEqual(validationErrors, [], "GPU validation errors during interleaved A/B");
}

// ---------------------------------------------------------------------------
// Attribution at the configured rate: full config, AO off, shadows off, both.
// ---------------------------------------------------------------------------
const attribution_ms: Record<string, number> = {};
if (coneScale !== 1) {
  for (const [key, shadows, ambientOcclusion] of [
    ["full", true, true],
    ["aoOff", true, false],
    ["shadowsOff", false, true],
    ["bothOff", false, false],
  ] as const) {
    applyLighting(coneScale, shadows, ambientOcclusion);
    for (let index = 0; index < 2; index += 1) {
      const encoder = device.createCommandEncoder({ label: `Attribution warmup ${key}` });
      encodeFrame(encoder);
      device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    attribution_ms[key] = median(await timeFrames(8, `Attribution ${key}`));
  }
  applyLighting(coneScale);
  assert.deepEqual(validationErrors, [], "GPU validation errors during attribution timing");
}

// ---------------------------------------------------------------------------
// Frame capture: packed rgba16float rows for hashing and decoded floats for
// quality statistics and PNGs.
// ---------------------------------------------------------------------------
const bytesPerPixel = 8; // rgba16float
const bytesPerRow = Math.ceil(width * bytesPerPixel / 256) * 256;
async function captureFrame(label: string): Promise<Uint32Array> {
  // dawn-node intermittently faults on repeated mapAsync of one long-lived
  // MAP_READ buffer, so every capture owns a fresh readback buffer.
  const readback = device.createBuffer({ label: `${label} readback`, size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder({ label });
  encodeFrame(encoder);
  encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow, rowsPerImage: height }, [width, height]);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(readback.getMappedRange());
  const packedRows = new Uint32Array((width * bytesPerPixel * height) / 4);
  for (let row = 0; row < height; row += 1) {
    const rowBytes = mapped.subarray(row * bytesPerRow, row * bytesPerRow + width * bytesPerPixel);
    packedRows.set(new Uint32Array(rowBytes.slice().buffer), (row * width * bytesPerPixel) / 4);
  }
  readback.unmap();
  readback.destroy();
  return packedRows;
}
function decodePixels(packedRows: Uint32Array): Float32Array {
  const halfWords = new Uint16Array(packedRows.buffer, packedRows.byteOffset, packedRows.length * 2);
  const pixels = new Float32Array(width * height * 4);
  for (let index = 0; index < pixels.length; index += 1) pixels[index] = decodeF16(halfWords[index]);
  return pixels;
}
function relativeLuminance(pixels: Float32Array, pixelIndex: number): number {
  return 0.2126 * pixels[pixelIndex * 4] + 0.7152 * pixels[pixelIndex * 4 + 1] + 0.0722 * pixels[pixelIndex * 4 + 2];
}

// All GPU captures run back-to-back before any heavy CPU work: dawn-node's
// async event pump intermittently faults when long blocking JS sections
// (decode/PNG encode) interleave with further GPU submissions in one process.
applyLighting(1);
const referenceRows = await captureFrame("Bench fingerprint frame");
let reducedRows: Uint32Array | undefined;
let overlayRows: Uint32Array | undefined;
if (coneScale !== 1) {
  applyLighting(coneScale);
  log("Capturing reduced frame for quality statistics");
  reducedRows = await captureFrame("Bench reduced frame");
  log("Capturing fallback-band diagnostic frame");
  device.queue.writeBuffer(uniformBuffer, 0, packViewUniforms(scene, camera, environmentId, solver.info, bodies.count, { mode: 10, opacity: 1 }));
  overlayRows = await captureFrame("Bench fallback diagnostic frame");
  device.queue.writeBuffer(uniformBuffer, 0, packViewUniforms(scene, camera, environmentId, solver.info, bodies.count));
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(validationErrors, [], "GPU validation errors during quality capture");
}

// Reference frame (scale 1) carries the bit-exact fingerprint contract.
const imageHash = fnv1a32(referenceRows);
const referencePixels = decodePixels(referenceRows);
if (process.env.FLUID_SVO_DRY_FRAME_DUMP) {
  const dump = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    dump[pixel * 3] = toneByte(referencePixels[pixel * 4]);
    dump[pixel * 3 + 1] = toneByte(referencePixels[pixel * 4 + 1]);
    dump[pixel * 3 + 2] = toneByte(referencePixels[pixel * 4 + 2]);
  }
  mkdirSync(path.dirname(process.env.FLUID_SVO_DRY_FRAME_DUMP), { recursive: true });
  writeFileSync(process.env.FLUID_SVO_DRY_FRAME_DUMP, encodePng(width, height, dump));
}
const gridSize = 16;
const gridSamples: Array<{ x: number; y: number; rgba: [number, number, number, number] }> = [];
for (let gy = 0; gy < gridSize; gy += 1) {
  for (let gx = 0; gx < gridSize; gx += 1) {
    const x = Math.min(width - 1, Math.floor(((gx + 0.5) / gridSize) * width));
    const y = Math.min(height - 1, Math.floor(((gy + 0.5) / gridSize) * height));
    const base = (y * width + x) * 4;
    gridSamples.push({
      x, y,
      rgba: [referencePixels[base], referencePixels[base + 1], referencePixels[base + 2], referencePixels[base + 3]],
    });
  }
}
const litSamples = gridSamples.filter(({ rgba }) => rgba[0] + rgba[1] + rgba[2] > 0).length;
assert.ok(litSamples > gridSize * gridSize * 0.25,
  `only ${litSamples}/${gridSize * gridSize} fingerprint samples carry radiance — the frame looks empty`);
assert.deepEqual(validationErrors, [], "GPU validation errors during fingerprint frame");
const referenceHashMatchesBaseline = imageHash === REFERENCE_IMAGE_HASH;
log(`Reference (scale 1) image hash 0x${imageHash.toString(16).padStart(8, "0")} (baseline 0x${REFERENCE_IMAGE_HASH.toString(16).padStart(8, "0")}: ${referenceHashMatchesBaseline ? "match" : "MISMATCH"})`);

// ---------------------------------------------------------------------------
// Quality: per-pixel relative luminance error over lit pixels, PNGs, and the
// guided-upsample fallback-band percentage from the mode-10 diagnostic overlay.
// ---------------------------------------------------------------------------
interface ErrorStats { litPixels: number; mean: number; p95: number; max: number; denominatorFloor: number }
let errorStats: ErrorStats | undefined;
let fallback: { percentOfHitPixels: number; hitPixels: number; fallbackPixels: number } | undefined;
let images: Record<string, string> | undefined;
if (coneScale !== 1 && reducedRows && overlayRows) {
  const reducedPixels = decodePixels(reducedRows);
  const denominatorFloor = 0.01;
  const errors: number[] = [];
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const referenceY = relativeLuminance(referencePixels, pixel);
    if (!(referenceY > 1e-4)) continue;
    const reducedY = relativeLuminance(reducedPixels, pixel);
    errors.push(Math.abs(reducedY - referenceY) / Math.max(referenceY, denominatorFloor));
  }
  errors.sort((a, b) => a - b);
  errorStats = {
    litPixels: errors.length,
    mean: errors.reduce((sum, value) => sum + value, 0) / Math.max(1, errors.length),
    p95: errors[Math.min(errors.length - 1, Math.ceil(0.95 * errors.length) - 1)] ?? 0,
    max: errors[errors.length - 1] ?? 0,
    denominatorFloor,
  };

  // PNGs: reference, reduced, and an 8x-amplified luminance-difference image.
  const outDirectory = path.dirname(outPath);
  mkdirSync(outDirectory, { recursive: true });
  const toRgb = (pixels: Float32Array): Uint8Array => {
    const rgb = new Uint8Array(width * height * 3);
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      rgb[pixel * 3] = toneByte(pixels[pixel * 4]);
      rgb[pixel * 3 + 1] = toneByte(pixels[pixel * 4 + 1]);
      rgb[pixel * 3 + 2] = toneByte(pixels[pixel * 4 + 2]);
    }
    return rgb;
  };
  const diffRgb = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const amplified = Math.max(0, Math.min(255, Math.round(255 * 8 * Math.abs(relativeLuminance(reducedPixels, pixel) - relativeLuminance(referencePixels, pixel)))));
    diffRgb[pixel * 3] = amplified;
    diffRgb[pixel * 3 + 1] = amplified;
    diffRgb[pixel * 3 + 2] = amplified;
  }
  images = {
    reference: path.join(outDirectory, "reference.png"),
    reduced: path.join(outDirectory, `reduced-x${coneScale}.png`),
    difference: path.join(outDirectory, `difference-x8-${coneScale}.png`),
  };
  writeFileSync(images.reference, encodePng(width, height, toRgb(referencePixels)));
  writeFileSync(images.reduced, encodePng(width, height, toRgb(reducedPixels)));
  writeFileSync(images.difference, encodePng(width, height, diffRgb));

  // Fallback band: overlay mode 10 paints red where the guided upsample fell
  // back to inline cones; hit pixels keep their linear depth in alpha.
  const overlayPixels = decodePixels(overlayRows);
  let hitPixels = 0;
  let fallbackPixels = 0;
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    if (!(overlayPixels[pixel * 4 + 3] > 0)) continue;
    hitPixels += 1;
    if (overlayPixels[pixel * 4] > 0.5 && overlayPixels[pixel * 4 + 1] < 0.4) fallbackPixels += 1;
  }
  fallback = { percentOfHitPixels: 100 * fallbackPixels / Math.max(1, hitPixels), hitPixels, fallbackPixels };
}

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
  coneLighting: {
    scale: coneScale,
    prepassResolution: coneScale !== 1 ? svoConePrepassSize(width, height, coneScale) : undefined,
    interleaved: interleaved ? {
      referenceMedian_ms: median(interleaved.reference_ms),
      reducedMedian_ms: median(interleaved.reduced_ms),
      referenceP95_ms: percentile95(interleaved.reference_ms),
      reducedP95_ms: percentile95(interleaved.reduced_ms),
      reference_ms: interleaved.reference_ms,
      reduced_ms: interleaved.reduced_ms,
    } : undefined,
    attribution_ms: coneScale !== 1 ? attribution_ms : undefined,
    errorStats,
    fallback,
    images,
  },
  scene: {
    presetId: "garden-svo-lighting",
    sceneId: scene.sceneId,
    environment: environmentId,
    quality: "balanced",
    lightingMode: "cone",
    shadowsEnabled,
    ambientOcclusionEnabled,
    coneLightingScale: coneScale,
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
    contract: "reference (scale 1) frame: 16x16 grid of RGBA radianceDepth (rgba16float, decoded to f32) at pixel centers of a uniform grid, plus FNV-1a-32 over the full packed image bytes; bit-exact reproduction expected on identical hardware/driver, otherwise compare grid values within 1e-3 absolute",
    imageHashFnv1a32: `0x${imageHash.toString(16).padStart(8, "0")}`,
    referenceHashMatchesBaseline,
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
sharedQuerySet?.destroy();
sharedResolve?.destroy();
sharedTiming?.destroy();
target.destroy();
uniformBuffer.destroy();
bodyBuffer.destroy();
solver.destroy();
device.destroy();
// dawn-node's async event pump intermittently faults during interpreter
// teardown after a destroyed instance; results are already flushed, so exit
// deterministically instead of risking a misleading nonzero shutdown signal.
process.exit(0);
