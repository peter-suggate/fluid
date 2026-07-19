import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  SparseVoxelTemporalAccumulator,
  SVO_TEMPORAL_ACCUMULATION_LAYOUT,
  sparseVoxelTemporalAccumulatorShader,
  type SparseVoxelTemporalFrameState,
} from "../lib/webgpu-svo-temporal-accumulator";

interface MockTexture extends GPUTexture {
  descriptor: GPUTextureDescriptor;
  destroyed: number;
  view: GPUTextureView;
}

function installGpuConstants(t: test.TestContext): void {
  const descriptors = ["GPUTextureUsage", "GPUBufferUsage", "GPUShaderStage"].map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)] as const);
  Object.defineProperty(globalThis, "GPUTextureUsage", { configurable: true, value: { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2, COPY_SRC: 4, COPY_DST: 8 } });
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { UNIFORM: 1, COPY_DST: 2 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { FRAGMENT: 2 } });
  t.after(() => descriptors.forEach(([name, descriptor]) => descriptor ? Object.defineProperty(globalThis, name, descriptor) : Reflect.deleteProperty(globalThis, name)));
}

function mockDevice(textures: MockTexture[], pipelines: GPURenderPipelineDescriptor[] = [], writes: ArrayBufferView[] = []): GPUDevice {
  return {
    createTexture(descriptor: GPUTextureDescriptor) {
      const view = { label: `${descriptor.label} view` } as GPUTextureView;
      const texture = { descriptor, destroyed: 0, view, width: (descriptor.size as number[])[0], height: (descriptor.size as number[])[1], createView() { return view; }, destroy() { texture.destroyed += 1; } } as unknown as MockTexture;
      textures.push(texture); return texture;
    },
    createBuffer() { return { destroy() {} }; },
    createShaderModule() { return { getCompilationInfo: async () => ({ messages: [] }) }; },
    createBindGroupLayout() { return {}; }, createPipelineLayout() { return {}; }, createBindGroup() { return {}; },
    async createRenderPipelineAsync(descriptor: GPURenderPipelineDescriptor) { pipelines.push(descriptor); return {}; },
    queue: { writeBuffer(_buffer: GPUBuffer, _offset: number, data: ArrayBufferView) { writes.push(new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))); } },
  } as unknown as GPUDevice;
}

const frame: SparseVoxelTemporalFrameState = {
  camera: { position_m: [0, 1, 3], forward: [0, 0, -1], right: [1, 0, 0], up: [0, 1, 0] },
  deltaTime_s: 1 / 60, cellSize_m: 0.025, paused: false, composition: "dry-before-legacy-water",
};

test("ping-pong HDR, compact keys, and moments have exact bounded formats and lifecycle", async (t) => {
  installGpuConstants(t);
  const textures: MockTexture[] = [], pipelines: GPURenderPipelineDescriptor[] = [];
  const accumulator = new SparseVoxelTemporalAccumulator(mockDevice(textures, pipelines));
  await accumulator.initialize();
  assert.deepEqual(SVO_TEMPORAL_ACCUMULATION_LAYOUT, {
    paramsBytes: 160, historyColorFormat: "rgba16float", momentsFormat: "rgba16float", keyFormat: "rgba16uint",
    pingPongBytesPerPixel: 64, maximumAccumulationSamples: 64, maximumStoredSamples: 255,
  });
  assert.deepEqual(Array.from(pipelines[0].fragment!.targets).map((target) => target?.format), ["rgba16float", "rgba16float", "rgba16uint", "rgba16uint"]);
  assert.equal(accumulator.ensureSize(320, 180), true);
  assert.equal(accumulator.ensureSize(320, 180), false);
  assert.equal(textures.length, 8);
  assert.deepEqual(textures.map(({ descriptor }) => descriptor.format), ["rgba16float", "rgba16float", "rgba16uint", "rgba16uint", "rgba16float", "rgba16float", "rgba16uint", "rgba16uint"]);
  assert.ok(textures.filter((_, index) => index % 4 === 0).every(({ descriptor }) => (descriptor.usage & GPUTextureUsage.COPY_SRC) !== 0));
  accumulator.ensureSize(640, 360);
  assert.ok(textures.slice(0, 8).every(({ destroyed }) => destroyed === 1));
  accumulator.destroy();
  assert.ok(textures.slice(8).every(({ destroyed }) => destroyed === 1));
});

test("resolve publishes four compact MRTs, copies only dry history back, and advances previous camera", async (t) => {
  installGpuConstants(t);
  const textures: MockTexture[] = [], writes: ArrayBufferView[] = [];
  const device = mockDevice(textures, [], writes);
  const accumulator = new SparseVoxelTemporalAccumulator(device);
  await accumulator.initialize(); accumulator.ensureSize(64, 48);
  const current = { width: 64, height: 48, createView: () => ({}), destroy() {} } as unknown as GPUTexture;
  const packed = { width: 64, height: 48, createView: () => ({}) } as unknown as GPUTexture;
  const identity = { width: 64, height: 48, createView: () => ({}) } as unknown as GPUTexture;
  const passes: GPURenderPassDescriptor[] = [], copies: Array<{ source: GPUTexelCopyTextureInfo; destination: GPUTexelCopyTextureInfo }> = [];
  const encoder = {
    beginRenderPass(descriptor: GPURenderPassDescriptor) { passes.push(descriptor); return { setPipeline() {}, setBindGroup() {}, draw() {}, end() {} }; },
    copyTextureToTexture(source: GPUTexelCopyTextureInfo, destination: GPUTexelCopyTextureInfo) { copies.push({ source, destination }); },
  } as unknown as GPUCommandEncoder;
  const gBuffer = { width: 64, height: 48, radianceDepthOwnership: "external-water-compositor-target", packedSurface: packed, identityMedia: identity, hardwareDepth: {} as GPUTexture } as const;
  assert.equal(accumulator.encode(encoder, current, gBuffer, frame), true);
  assert.equal(Array.from(passes[0].colorAttachments).length, 4);
  assert.equal(copies.length, 1);
  assert.equal(copies[0].destination.texture, current, "only resolved dry HDR is copied before legacy water composition");
  assert.equal((writes[0] as Float32Array)[38], 0, "first frame must reject uninitialized history");
  assert.equal(accumulator.encode(encoder, current, gBuffer, { ...frame, camera: { ...frame.camera, position_m: [0.1, 1, 3] } }), true);
  assert.equal((writes[1] as Float32Array)[38], 1, "second frame may use the stored previous camera/key");
  accumulator.invalidate();
  assert.equal(accumulator.encode(encoder, current, gBuffer, frame), true);
  assert.equal((writes[2] as Float32Array)[38], 0);
  accumulator.destroy();
});

test("shader accepts exact static or rigid-valid motion and uses velocity reprojection, rejection, moments, and neighborhood clamp", () => {
  assert.match(sparseVoxelTemporalAccumulatorShader, /supportedMotion=motionKind==SVO_TEMPORAL_MOTION_STATIC\|\|motionKind==SVO_TEMPORAL_MOTION_RIGID/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /previousWorld=world-velocity\*temporal\.control\.y/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /svoTemporalHistoryReason\(currentKey,previousKey,temporal\.control\.x,temporal\.control\.y,velocity,motionKind,true,true,error\)/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /TEMPORAL_REQUIRED_FLAGS/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /svoTemporalHistoryReason\(/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /currentKey\.depth_m=expectedPreviousDistance/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /previousForwardDepth>0\.0&&all\(previousUv>=vec2f\(0\.0\)\)&&all\(previousUv<vec2f\(1\.0\)\)/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /for\(var y=-1;y<=1;y\+=1\)[^]*for\(var x=-1;x<=1;x\+=1\)/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /history=temporalVarianceClamp\(history,oldMoments\)/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /sampleCount=min\(oldMoments\.z\+1\.0,255\.0\)/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /accumulationCount=min\(sampleCount,64\.0\)/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /accepted=oldMoments\.z>0\.0&&svoTemporalHistoryReason/,
    "the signed stored count must reject history produced by a previous non-static or invalid surface");
});

test("production integration invalidates history outside smooth SVO and resolves before raster water", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const water = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
  assert.match(renderer, /if \(!useSvoDryScene\) this\.svoDryScenePipeline\?\.invalidateTemporalHistory\(\)/);
  assert.match(renderer, /composition: "dry-before-legacy-water"/);
  assert.match(renderer, /if \(!svoEncoded\) this\.svoDryScenePipeline\?\.invalidateTemporalHistory\(\)/);
  assert.match(water, /drySceneReplacement\?\.\(encoder, this\.sceneTexture, timestamps\?\.scene\)/);
  assert.match(water, /Dry scene HDR[^]*GPUTextureUsage\.COPY_DST/);
  const replacement = water.indexOf("drySceneReplacement?.(encoder, this.sceneTexture");
  const interfaces = water.indexOf("interfacePass(\"Water + spray front interfaces\"", replacement);
  assert.ok(replacement >= 0 && interfaces > replacement, "dry temporal history resolves before legacy water and spray composition");
});
