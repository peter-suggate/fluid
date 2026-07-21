import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  SVO_GBUFFER_BYTES_PER_PIXEL,
  SVO_GBUFFER_COLOR_ATTACHMENT_COUNT,
  SVO_GBUFFER_COLOR_BYTES_PER_SAMPLE,
  SVO_GBUFFER_LAYOUT,
} from "../lib/svo-gbuffer";
import {
  SparseVoxelGBufferTargetArena,
  SVO_GBUFFER_RENDER_TARGET_CONTRACT,
} from "../lib/webgpu-svo-gbuffer-targets";
import { SparseVoxelDrySceneRenderer, svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelTemporalFrameState } from "../lib/webgpu-svo-temporal-accumulator";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const waterSource = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");

interface MockTexture extends GPUTexture {
  descriptor: GPUTextureDescriptor;
  destroyed: number;
  view: GPUTextureView;
}

function installGpuConstants(t: test.TestContext): void {
  const descriptors = ["GPUTextureUsage", "GPUBufferUsage", "GPUShaderStage"].map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)] as const);
  Object.defineProperty(globalThis, "GPUTextureUsage", { configurable: true, value: { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 } });
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { FRAGMENT: 2 } });
  t.after(() => {
    for (const [name, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
  });
}

function mockDevice(textures: MockTexture[], renderDescriptors: GPURenderPipelineDescriptor[] = []): GPUDevice {
  return {
    createTexture(descriptor: GPUTextureDescriptor) {
      const view = { label: `${descriptor.label} view` } as GPUTextureView;
      const texture = {
        descriptor,
        destroyed: 0,
        view,
        createView() { return view; },
        destroy() { texture.destroyed += 1; },
      } as unknown as MockTexture;
      textures.push(texture);
      return texture;
    },
    createBuffer() { return { destroy() {} }; },
    createShaderModule() { return { getCompilationInfo: async () => ({ messages: [] }) }; },
    createBindGroup() { return {}; },
    createBindGroupLayout() { return {}; },
    createSampler() { return {}; },
    createPipelineLayout() { return {}; },
    async createRenderPipelineAsync(descriptor: GPURenderPipelineDescriptor) { renderDescriptors.push(descriptor); return {}; },
    queue: { writeBuffer() {} },
  } as unknown as GPUDevice;
}

test("auxiliary renderer-owned targets preserve the exact compact G-buffer contract", (t) => {
  installGpuConstants(t);
  const textures: MockTexture[] = [];
  const arena = new SparseVoxelGBufferTargetArena(mockDevice(textures));
  assert.deepEqual(SVO_GBUFFER_RENDER_TARGET_CONTRACT, {
    colorAttachmentCount: SVO_GBUFFER_COLOR_ATTACHMENT_COUNT,
    colorBytesPerSample: SVO_GBUFFER_COLOR_BYTES_PER_SAMPLE,
    bytesPerPixelIncludingDepth: SVO_GBUFFER_BYTES_PER_PIXEL,
    externalRadianceDepthFormat: SVO_GBUFFER_LAYOUT.radianceDepth.format,
    packedSurfaceFormat: SVO_GBUFFER_LAYOUT.packedSurface.format,
    identityMediaFormat: SVO_GBUFFER_LAYOUT.identityMedia.format,
    hardwareDepthFormat: SVO_GBUFFER_LAYOUT.hardwareDepth.format,
    depthClearValue: 0,
    depthCompare: "greater",
  });
  assert.equal(arena.ensureSize(320, 180), true);
  assert.equal(arena.ensureSize(320, 180), false, "an unchanged size retains temporal resources");
  assert.deepEqual(textures.map(({ descriptor }) => descriptor.format), ["rgba32uint", "rgba16uint", "depth32float"]);
  assert.ok(textures.every(({ descriptor }) => Array.isArray(descriptor.size) && descriptor.size[0] === 320 && descriptor.size[1] === 180));
  assert.deepEqual(arena.textures && {
    width: arena.textures.width,
    height: arena.textures.height,
    radianceDepthOwnership: arena.textures.radianceDepthOwnership,
  }, { width: 320, height: 180, radianceDepthOwnership: "external-water-compositor-target" });

  assert.equal(arena.ensureSize(640, 360), true);
  assert.ok(textures.slice(0, 3).every(({ destroyed }) => destroyed === 1), "resize retires every previous auxiliary attachment");
  arena.destroy();
  assert.ok(textures.slice(3).every(({ destroyed }) => destroyed === 1));
  assert.equal(arena.textures, undefined);
  assert.throws(() => arena.ensureSize(0, 10), /positive safe integers/);
  assert.throws(() => arena.ensureSize(10.5, 10), /positive safe integers/);
});

test("production dry pass writes three MRTs plus reversed-Z without changing location-zero ownership", async (t) => {
  installGpuConstants(t);
  const textures: MockTexture[] = [], pipelineDescriptors: GPURenderPipelineDescriptor[] = [];
  const renderer = new SparseVoxelDrySceneRenderer(mockDevice(textures, pipelineDescriptors), {} as GPUBuffer, {} as GPUBuffer);
  await renderer.initialize();
  assert.equal(pipelineDescriptors.length, 2, "direct visibility and temporal resolve compile as separate bounded passes");
  const descriptor = pipelineDescriptors[0];
  assert.deepEqual(Array.from(descriptor.fragment!.targets).map((target) => target?.format), ["rgba16float", "rgba32uint", "rgba16uint"]);
  assert.deepEqual(descriptor.depthStencil, { format: "depth32float", depthWriteEnabled: true, depthCompare: "greater" });

  renderer.ensureSize(64, 48);
  const internals = renderer as unknown as { bindGroup: GPUBindGroup };
  internals.bindGroup = {} as GPUBindGroup;
  let passDescriptor: GPURenderPassDescriptor | undefined, passCount = 0;
  const encoder = {
    beginRenderPass(descriptor: GPURenderPassDescriptor) {
      passDescriptor = descriptor; passCount += 1;
      return { setPipeline() {}, setBindGroup() {}, draw() {}, end() {} };
    },
  } as unknown as GPUCommandEncoder;
  const externalHdr = { label: "existing water compositor HDR view" } as GPUTextureView;
  const result = renderer.encode(encoder, externalHdr);
  assert.ok(result);
  assert.equal(result.sampledTargetView, externalHdr);
  const colorAttachments = Array.from(passDescriptor?.colorAttachments ?? []);
  assert.equal(colorAttachments.length, 3);
  assert.equal(colorAttachments[0]?.view, externalHdr);
  assert.equal(colorAttachments[1]?.view, renderer.gBufferTextures?.packedSurface.createView());
  assert.equal(colorAttachments[2]?.view, renderer.gBufferTextures?.identityMedia.createView());
  assert.equal(passDescriptor?.depthStencilAttachment?.view, renderer.gBufferTextures?.hardwareDepth.createView());
  assert.equal(passDescriptor?.depthStencilAttachment?.depthClearValue, 0);
  assert.equal(passDescriptor?.depthStencilAttachment?.depthLoadOp, "clear");
  assert.equal(passDescriptor?.depthStencilAttachment?.depthStoreOp, "store");
  const reusableView = { label: "reusable dry HDR" } as GPUTextureView;
  const reusableTexture = { width: 64, height: 48, createView: () => reusableView } as GPUTexture;
  const passesBeforeReuse = passCount;
  const firstReusable = renderer.encode(encoder, reusableTexture, undefined, undefined, undefined, "fixed-camera-and-bodies");
  const secondReusable = renderer.encode(encoder, reusableTexture, undefined, undefined, undefined, "fixed-camera-and-bodies");
  assert.ok(firstReusable && secondReusable);
  assert.equal(passCount, passesBeforeReuse + 1, "an unchanged non-temporal dry frame is rendered once and then reused");
  assert.equal(secondReusable.sampledTargetView, firstReusable.sampledTargetView);
  const temporalFrame: SparseVoxelTemporalFrameState = {
    camera: { position_m: [0, 0, 0], forward: [0, 0, -1], right: [1, 0, 0], up: [0, 1, 0] },
    deltaTime_s: 0,
    cellSize_m: 0.025,
    paused: true,
    composition: "dry-before-raster-water",
  };
  const passesBeforeTemporalReuse = passCount;
  const firstTemporal = renderer.encode(encoder, reusableTexture, undefined, temporalFrame, undefined, "fixed-temporal-frame");
  const secondTemporal = renderer.encode(encoder, reusableTexture, undefined, temporalFrame, undefined, "fixed-temporal-frame");
  const reusedTemporal = renderer.encode(encoder, reusableTexture, undefined, temporalFrame, undefined, "fixed-temporal-frame");
  assert.ok(firstTemporal && secondTemporal && reusedTemporal);
  assert.equal(passCount, passesBeforeTemporalReuse + 4,
    "both checkerboard phases render and resolve before an unchanged temporal frame is reused");
  assert.equal(reusedTemporal.sampledTargetView, secondTemporal.sampledTargetView);
  renderer.destroy();
  assert.throws(() => new SparseVoxelDrySceneRenderer(mockDevice([]), {} as GPUBuffer, {} as GPUBuffer, "bgra8unorm"), /location 0 must use rgba16float/);
});

test("shader populates stable identity/media/generation and consumes exact rigid surface motion", () => {
  assert.match(svoDrySceneShader, /struct DryFragmentOut[^]*@location\(0\) radianceDepth:vec4f[^]*@location\(1\) packedSurface:vec4u[^]*@location\(2\) identityMedia:vec4u[^]*@builtin\(frag_depth\) hardwareDepth:f32/);
  assert.match(svoDrySceneShader, /svoGBufferSurface\(radiance,opaque\.t,opaque\.normal,opaque\.normal,vec4u\(dryResolvedMaterialId\(opaque\),opaque\.ownerId,media\.x,media\.y\),motionVelocity,opaque\.motionKind,opaque\.fieldSource,motionGeneration,flags,opaque\.featureId\)/);
  assert.match(svoDrySceneShader, /DRY_GBUFFER_MOTION_RIGID,0u,body\.colorSelected\.w/,
    "the analytic hit remains fail-closed until the separately validated motion sidecar is resolved");
  assert.match(svoDrySceneShader, /let rigidSurface=dryRigidMotionSurface\(opaque,ro\+rd\*opaque\.t\)/);
  assert.match(svoDrySceneShader, /svoPrimitiveMotionVelocityAt\(record,worldSurfacePosition_m\)/);
  assert.match(svoDrySceneShader, /DRY_GBUFFER_MOTION_STATIC,1u,0\.0/,
    "authored static primitives publish known zero motion");
  assert.match(svoDrySceneShader, /svoPrimitiveOwnerId\(record\),exact\.featureId,DRY_GBUFFER_FIELD_ANALYTIC/,
    "static G-buffer feature identity must come from the shared exact ray hit without a second distance evaluation");
  assert.match(svoDrySceneShader, /dry\.terrain\.x,DRY_OWNER_NONE,SVO_FEATURE_TERRAIN,DRY_GBUFFER_FIELD_TERRAIN/);
  assert.match(svoDrySceneShader, /dryPublicationGeneration\(\)->u32[^]*publicationState\[3\]/,
    "static and analytic history must use the stable static-geometry revision");
  assert.match(svoDrySceneShader, /DRY_REVERSED_Z_NEAR_M\/viewDepth_m/);
  assert.match(svoDrySceneShader, /svoGBufferMiss\(radiance,0u,generation,DRY_GBUFFER_NO_INTERSECTION,0u\),0\.0/);
  assert.match(svoDrySceneShader, /dryMediumPair\(rd,opaque\.normal,DRY_MEDIUM_OPAQUE\)/,
    "the dry scene publishes only opaque media because raster water owns fluid interfaces");
});

test("resize and water composition retain the SVO MRT lifecycle and zero-depth miss semantics", () => {
  assert.match(rendererSource, /this\.waterPipeline\?\.ensureSize\(renderWidth, renderHeight\);\s*this\.svoDryScenePipeline\?\.ensureSize\(renderWidth, renderHeight\)/);
  assert.match(rendererSource, /Fluid Lab rigid bodies[^\n]*GPUBufferUsage\.STORAGE \| GPUBufferUsage\.UNIFORM \| GPUBufferUsage\.COPY_DST/);
  assert.match(svoDrySceneShader, /@group\(0\) @binding\(1\) var<uniform> bodies:array<BodyGPU,12>/,
    "the bounded body table must not consume the eleventh fragment storage slot");
  assert.match(waterSource, /fn resolvedDrySceneDepth\(encodedDepth:f32\)->f32\{return select\(65504\.0,encodedDepth,encodedDepth>0\.0\);\}/);
  assert.match(waterSource, /resolvedDrySceneDepth\(scene\.a\)\+depthEpsilon<frontDepth/);
  assert.match(waterSource, /ssr\.a>0\.0&&ssr\.a<60000\.0/);
});
