import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import test from "node:test";

import { decodeSvoGBufferFloat16, encodeSvoGBufferFloat16 } from "../lib/svo-gbuffer";
import {
  SparseVoxelTemporalAccumulator,
  SVO_TEMPORAL_ACCUMULATION_LAYOUT,
  sparseVoxelTemporalAllocatedBytes,
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
  Object.defineProperty(globalThis, "GPUTextureUsage", { configurable: true, value: { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2, COPY_SRC: 4, COPY_DST: 8, STORAGE_BINDING: 16 } });
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { UNIFORM: 1, COPY_DST: 2 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { FRAGMENT: 2, COMPUTE: 4 } });
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
  deltaTime_s: 1 / 60, cellSize_m: 0.025, paused: false, composition: "dry-before-raster-water",
};

const modulePath = process.env.WEBGPU_NODE_MODULE;

test("ping-pong HDR, compact keys, and moments have exact bounded formats and lifecycle", async (t) => {
  installGpuConstants(t);
  const textures: MockTexture[] = [], pipelines: GPURenderPipelineDescriptor[] = [];
  const accumulator = new SparseVoxelTemporalAccumulator(mockDevice(textures, pipelines));
  await accumulator.initialize();
  assert.deepEqual(SVO_TEMPORAL_ACCUMULATION_LAYOUT, {
    paramsBytes: 160, historyColorFormat: "rgba16float", momentsFormat: "rgba16float", keyFormat: "rgba16uint",
    pingPongBytesPerPixel: 64, previousNeighborhoodLoadsPerAcceptedPixel: 9, neighborhoodLoadsPerAcceptedPixel: 8,
    fullScreenResolvePassesPerFrame: 1, aliasBreakingCopiesPerFrame: 0,
    maximumAccumulationSamples: 64, maximumStoredSamples: 255,
  });
  assert.deepEqual(Array.from(pipelines[0].fragment!.targets).map((target) => target?.format), ["rgba16float", "rgba16float", "rgba16uint", "rgba16uint"]);
  assert.equal(accumulator.ensureSize(320, 180), true);
  assert.equal(accumulator.allocatedBytes, 3_686_560);
  assert.equal(sparseVoxelTemporalAllocatedBytes(1920, 1080), 132_710_560);
  assert.equal(accumulator.ensureSize(320, 180), false);
  assert.equal(textures.length, 8);
  assert.deepEqual(textures.map(({ descriptor }) => descriptor.format), ["rgba16float", "rgba16float", "rgba16uint", "rgba16uint", "rgba16float", "rgba16float", "rgba16uint", "rgba16uint"]);
  assert.ok(textures.every(({ descriptor }) => (descriptor.usage & GPUTextureUsage.COPY_SRC) !== 0));
  accumulator.ensureSize(640, 360);
  assert.ok(textures.slice(0, 8).every(({ destroyed }) => destroyed === 1));
  accumulator.destroy();
  assert.ok(textures.slice(8).every(({ destroyed }) => destroyed === 1));
});

test("one resolve pass exposes ping-pong HDR without an alias-breaking copy and advances previous camera", async (t) => {
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
  const timestampWrites = { querySet: {} as GPUQuerySet, beginningOfPassWriteIndex: 16, endOfPassWriteIndex: 17 };
  const firstResolve = accumulator.encode(encoder, current, gBuffer, frame, timestampWrites);
  assert.ok(firstResolve);
  assert.equal(passes[0].timestampWrites, timestampWrites);
  assert.equal(Array.from(passes[0].colorAttachments).length, 4);
  assert.equal(copies.length, 0);
  assert.equal(firstResolve.resolvedTexture, textures[4], "the first resolve exposes the next ping-pong HDR texture");
  assert.equal((writes[0] as Float32Array)[38], 0, "first frame must reject uninitialized history");
  const secondResolve = accumulator.encode(encoder, current, gBuffer, { ...frame, camera: { ...frame.camera, position_m: [0.1, 1, 3] } });
  assert.ok(secondResolve);
  assert.equal(secondResolve.resolvedTexture, textures[0], "the second resolve exposes the other ping-pong HDR texture");
  assert.equal((writes[1] as Float32Array)[38], 1, "second frame may use the stored previous camera/key");
  accumulator.invalidate();
  assert.ok(accumulator.encode(encoder, current, gBuffer, frame));
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
  assert.match(sparseVoxelTemporalAccumulatorShader, /for\(var y=-1;y<=1;y\+=1\)[^]*for\(var x=-1;x<=1;x\+=1\)[^]*if\(x==0&&y==0\)\{continue;\}/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /let exactIdentity=all\(keyA\.zw==published\[0\]\.zw\)&&all\(keyB==published\[1\]\)/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /history=temporalVarianceClamp\(history,oldMoments\)/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /sampleCount=min\(oldMoments\.z\+1\.0,255\.0\)/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /accumulationCount=min\(sampleCount,64\.0\)/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /shadowDeferred=failure==TEMPORAL_FAILURE_SHADOW_DEFERRED/);
  assert.match(sparseVoxelTemporalAccumulatorShader, /if\(shadowDeferred\)\{result=previous\.rgb;sampleCount=oldMoments\.z;pausedStable=oldMoments\.w;\}/,
    "a deferred shadow pixel reuses only accepted identity-validated history");
  assert.match(sparseVoxelTemporalAccumulatorShader, /sampleCount=select\(-1\.0,1\.0,currentUsable&&!shadowDeferred\)/,
    "first-frame deferred pixels must not become valid history");
  assert.match(sparseVoxelTemporalAccumulatorShader, /if\(oldMoments\.z>0\.0\)[^]*if\(exactIdentity\)[^]*accepted=svoTemporalHistoryReason/,
    "the signed stored count must reject history produced by a previous non-static or invalid surface");
});

test("production integration invalidates history outside smooth SVO and resolves before raster water", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const water = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
  assert.match(renderer, /if \(!useSvoDryScene\) this\.svoDryScenePipeline\?\.invalidateTemporalHistory\(\)/);
  assert.match(renderer, /composition: "dry-before-raster-water"/);
  assert.match(renderer, /if \(!replacementResult\) this\.svoDryScenePipeline\?\.invalidateTemporalHistory\(\)/);
  assert.match(renderer, /beginningOfPassWriteIndex: 16, endOfPassWriteIndex: 17/);
  assert.match(readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8"), /dryPublicationGeneration\(\)->u32\{return select\(0u,publicationState\[3\]/,
    "static history identity must not be invalidated by every completed fluid publication");
  assert.match(renderer, /gpuSvoTemporal_ms=stage\.svoTemporal_ms/);
  assert.match(water, /drySceneReplacement\?\.\(encoder, this\.sceneTexture, timestamps\?\.scene\)/);
  assert.doesNotMatch(water, /Dry scene HDR[^\n]*GPUTextureUsage\.COPY_DST/);
  assert.match(water, /compositeBindGroupFor\(sparseSceneResult\.sampledTargetView\)/);
  const replacement = water.indexOf("drySceneReplacement?.(encoder, this.sceneTexture");
  const interfaces = water.indexOf("interfacePass(\"Water + spray front interfaces\"", replacement);
  assert.ok(replacement >= 0 && interfaces > replacement, "dry temporal history resolves before raster water and spray composition");
});

test("real GPU resolve preserves a seeded nonzero dry HDR target", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU temporal checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]); const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice(); const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => validationErrors.push((event as { error: { message: string } }).error.message));
  const accumulator = new SparseVoxelTemporalAccumulator(device); const width = 8, height = 8;
  const textures: GPUTexture[] = [];
  const texture = (format: GPUTextureFormat, usage: GPUTextureUsageFlags) => {
    const value = device.createTexture({ size: [width, height], format, usage }); textures.push(value); return value;
  };
  const current = texture("rgba16float", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC);
  const packed = texture("rgba32uint", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
  const identity = texture("rgba16uint", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
  const depth = texture("depth32float", GPUTextureUsage.TEXTURE_BINDING);
  const paddedRows = (pixel: ArrayBufferView) => {
    const bytes = new Uint8Array(256 * height); const source = new Uint8Array(pixel.buffer, pixel.byteOffset, pixel.byteLength);
    for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) bytes.set(source, y * 256 + x * source.byteLength);
    return bytes;
  };
  const seeded = new Uint16Array([encodeSvoGBufferFloat16(.75), encodeSvoGBufferFloat16(.25), encodeSvoGBufferFloat16(.125), encodeSvoGBufferFloat16(2)]);
  device.queue.writeTexture({ texture: current }, paddedRows(seeded), { bytesPerRow: 256, rowsPerImage: height }, [width, height]);
  const requiredFlags = 1 | 4 | 8 | 16 | 32 | 64;
  const packedPixel = new Uint32Array([0, 7, 0, 4 | requiredFlags << 4]);
  const identityPixel = new Uint16Array([5, 6, 0, 0]);
  device.queue.writeTexture({ texture: packed }, paddedRows(packedPixel), { bytesPerRow: 256, rowsPerImage: height }, [width, height]);
  device.queue.writeTexture({ texture: identity }, paddedRows(identityPixel), { bytesPerRow: 256, rowsPerImage: height }, [width, height]);
  try {
    const seedReadback = device.createBuffer({ size: 256 * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const seedEncoder = device.createCommandEncoder(); seedEncoder.copyTextureToBuffer({ texture: current }, { buffer: seedReadback, bytesPerRow: 256, rowsPerImage: height }, [width, height]); device.queue.submit([seedEncoder.finish()]);
    await seedReadback.mapAsync(GPUMapMode.READ); const seedResult = new Uint16Array(seedReadback.getMappedRange().slice(0)); seedReadback.unmap(); seedReadback.destroy();
    assert.equal(seedResult[0], seeded[0], "test seed reaches the current dry target");
    await accumulator.initialize(); accumulator.ensureSize(width, height);
    const encoder = device.createCommandEncoder();
    const firstResolve = accumulator.encode(encoder, current, { width, height, radianceDepthOwnership: "external-water-compositor-target", packedSurface: packed, identityMedia: identity, hardwareDepth: depth }, frame);
    assert.ok(firstResolve);
    const readback = device.createBuffer({ size: 256 * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyTextureToBuffer({ texture: firstResolve.resolvedTexture }, { buffer: readback, bytesPerRow: 256, rowsPerImage: height }, [width, height]); device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ); const result = new Uint16Array(readback.getMappedRange().slice(0)); readback.unmap();
    await device.queue.onSubmittedWorkDone();
    assert.ok(decodeSvoGBufferFloat16(result[0]) > .7, `resolved red channel remains nonzero; validation=${validationErrors.join(" | ")}`);
    assert.ok(decodeSvoGBufferFloat16(result[1]) > .2, "resolved green channel remains nonzero");
    assert.equal(result[3], seeded[3], "linear depth is preserved exactly on the first frame");
    device.queue.writeTexture({ texture: current }, paddedRows(seeded), { bytesPerRow: 256, rowsPerImage: height }, [width, height]);
    const secondEncoder = device.createCommandEncoder();
    const secondResolve = accumulator.encode(secondEncoder, current, { width, height, radianceDepthOwnership: "external-water-compositor-target", packedSurface: packed, identityMedia: identity, hardwareDepth: depth }, frame);
    assert.ok(secondResolve);
    const internal = accumulator as unknown as { previousIndex: number; history: readonly [{ moments: GPUTexture }, { moments: GPUTexture }] };
    const secondReadback = device.createBuffer({ size: 512 * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    secondEncoder.copyTextureToBuffer({ texture: secondResolve.resolvedTexture }, { buffer: secondReadback, bytesPerRow: 256, rowsPerImage: height }, [width, height]);
    secondEncoder.copyTextureToBuffer({ texture: internal.history[internal.previousIndex].moments }, { buffer: secondReadback, offset: 256 * height, bytesPerRow: 256, rowsPerImage: height }, [width, height]);
    device.queue.submit([secondEncoder.finish()]); await secondReadback.mapAsync(GPUMapMode.READ); const second = new Uint16Array(secondReadback.getMappedRange().slice(0)); secondReadback.unmap();
    assert.deepEqual(Array.from(second.slice(0, 4)), Array.from(seeded), "stable accepted history preserves the seeded HDR bits");
    assert.equal(decodeSvoGBufferFloat16(second[256 * height / 2 + 2]), 2, "second stable frame advances the stored sample count");
    assert.deepEqual(validationErrors, []); readback.destroy();
    secondReadback.destroy();
  } finally {
    accumulator.destroy(); for (const value of textures) value.destroy(); device.destroy();
  }
});
