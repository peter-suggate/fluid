import assert from "node:assert/strict";
import test, { after } from "node:test";
import { pathToFileURL } from "node:url";

import { SVO_GBUFFER_FIELD_SOURCES, SVO_GBUFFER_MOTION_KINDS } from "../lib/svo-gbuffer";
import {
  packSvoRigidRasterSplitIdentity,
  SVO_RIGID_RASTER_CONTRACT,
  SVO_RIGID_RASTER_FEATURES,
  SVO_RIGID_RASTER_INTEGRATION,
  svoRigidRasterCoverageBridgeBindGroupLayoutEntries,
  svoRigidRasterInputBindGroupLayoutEntries,
  svoRigidRasterOutputBindGroupLayoutEntries,
  svoRigidRasterShader,
} from "../lib/webgpu-svo-rigid-raster";

test("rigid raster bridge preserves the existing BodyGPU and split identity ABI", () => {
  assert.match(svoRigidRasterShader, /struct BodyGPU\{\s*positionRadius:vec4f,\s*halfSizeShape:vec4f,\s*orientation:vec4f,\s*colorSelected:vec4f,/);
  assert.match(svoRigidRasterShader, /@binding\(1\) var<storage,read> rigidBodies:array<BodyGPU>/,
    "the live renderer body buffer is reusable without Metal's dynamic uniform-array indexing");
  assert.match(svoRigidRasterShader, /@builtin\(instance_index\) instanceIndex:u32/);
  assert.match(svoRigidRasterShader, /rigidProxyCorner\(vertexIndex\)\*rigidProxyExtent\(body\)/);
  assert.match(svoRigidRasterShader, /@binding\(14\) var<uniform> rigidMotion:array<SvoPrimitiveMotionRecord,12>/);
  assert.equal(SVO_RIGID_RASTER_CONTRACT.verticesPerProxy, 36);
  assert.equal(SVO_RIGID_RASTER_CONTRACT.bodyStrideBytes, 64);
  assert.equal(SVO_RIGID_RASTER_CONTRACT.bodyBufferBytes, 768);
  assert.equal(SVO_RIGID_RASTER_CONTRACT.bodySourcePolicy, "bind-live-buffer-directly");
  assert.deepEqual(SVO_RIGID_RASTER_CONTRACT.bodyBufferRequiredUsage, ["UNIFORM", "STORAGE", "COPY_DST"]);
  const [material, metadata] = packSvoRigidRasterSplitIdentity(7, SVO_RIGID_RASTER_FEATURES.cylinderCap);
  assert.equal(material, 0x8000_0007);
  assert.equal(metadata & 0xffff, 7);
  assert.equal(metadata >>> 16 & 0xf, SVO_RIGID_RASTER_FEATURES.cylinderCap);
  assert.equal(metadata >>> 20 & 0xf, SVO_GBUFFER_FIELD_SOURCES.analyticPrimitive);
  assert.equal(metadata >>> 24 & 0x3, SVO_GBUFFER_MOTION_KINDS.rigid);
  assert.equal(SVO_RIGID_RASTER_INTEGRATION.stage,
    "inside-svo-renderer-after-static-primary-before-cone-prepass");
  assert.equal(SVO_RIGID_RASTER_INTEGRATION.splitBridge,
    "second-proxy-raster-pass-unpacks-current-frame-winner-certificate");
  assert.match(SVO_RIGID_RASTER_INTEGRATION.deferredMaterial, /current-BodyGPU/);
  assert.match(SVO_RIGID_RASTER_INTEGRATION.motion, /rigidMotion/);
  assert.match(SVO_RIGID_RASTER_INTEGRATION.ambientOcclusion, /current-rigid/);
  assert.match(SVO_RIGID_RASTER_INTEGRATION.shadows, /current-rigid-shadow/);
  assert.equal(SVO_RIGID_RASTER_INTEGRATION.history, "none");
});

test("rigid raster performs one exact analytic test per covered instance without temporal sampling", () => {
  assert.match(svoRigidRasterShader, /fn rigidSphereHit/);
  assert.match(svoRigidRasterShader, /fn rigidBoxHit/);
  assert.match(svoRigidRasterShader, /fn rigidCapsuleHit/);
  assert.match(svoRigidRasterShader, /fn rigidCylinderHit/);
  assert.match(svoRigidRasterShader, /if\(shape==0\).*rigidSphereHit[^]*else if\(shape==1\).*rigidBoxHit[^]*else if\(shape==2\).*rigidCapsuleHit[^]*else if\(shape==3\).*rigidCylinderHit/);
  assert.match(svoRigidRasterShader, /@builtin\(frag_depth\) hardwareDepth:f32/);
  assert.match(svoRigidRasterShader, /RIGID_RASTER_NEAR_M\/viewDepth/);
  assert.doesNotMatch(svoRigidRasterShader, /jitter|dither|previousFrame/i);
  assert.deepEqual({
    geometry: SVO_RIGID_RASTER_CONTRACT.geometryFormat,
    identity: SVO_RIGID_RASTER_CONTRACT.identityFormat,
    packedSurface: SVO_RIGID_RASTER_CONTRACT.packedSurfaceFormat,
    identityMedia: SVO_RIGID_RASTER_CONTRACT.identityMediaFormat,
    depth: SVO_RIGID_RASTER_CONTRACT.depthFormat,
    compare: SVO_RIGID_RASTER_CONTRACT.depthCompare,
  }, {
    geometry: "rgba32float", identity: "rg32uint", packedSurface: "rgba32uint",
    identityMedia: "rgba16uint", depth: "depth32float", compare: "greater",
  });
  assert.equal(SVO_RIGID_RASTER_CONTRACT.colorAttachmentBytesPerSample, 32);
  assert.equal(SVO_RIGID_RASTER_CONTRACT.primaryGeometryFormat, "rg32uint");
  assert.match(svoRigidRasterShader, /vec2u\(bitcast<u32>\(hit\.t\),rigidPackPrimaryGeometryMetadata\(hit\.normal,input\.bodyIndex,hit\.featureId,motion\.valid\)\)/);
  const fragment = svoRigidRasterShader.slice(svoRigidRasterShader.indexOf("@fragment fn rigidRasterFragment"),
    svoRigidRasterShader.indexOf("@compute", svoRigidRasterShader.indexOf("@fragment fn rigidRasterFragment")));
  assert.doesNotMatch(fragment, /textureStore/,
    "late analytic depth testing must finish before split storage side effects");
  assert.match(svoRigidRasterShader, /@compute @workgroup_size\(8,8\) fn rigidSplitBridge/);
  assert.match(svoRigidRasterShader, /motionKind=packedSurface\.z>>30u;if\(motionKind!=RIGID_RASTER_MOTION_RIGID\)\{return;\}/);
  assert.match(svoRigidRasterShader, /let bodyIndex=identityMedia\.y/);
  assert.match(svoRigidRasterShader, /rigidBodyHit\(ray\[0\],ray\[1\],body\)/);
  assert.match(svoRigidRasterShader, /textureStore\(rigidSplitGeometryWrite,coordinate,vec4f\(hit\.normal,hit\.t\)\)/);
  assert.match(svoRigidRasterShader, /svoPrimitiveMotionVelocityAt\(record,worldSurfacePosition_m\)/);
  assert.match(svoRigidRasterShader, /svoPrimitiveMotionMaterialId\(record\)==materialId/);
  assert.match(svoRigidRasterShader, /transformValid=distance\(record\.currentPositionDt\.xyz,body\.positionRadius\.xyz\)<=1e-5/);
  assert.match(svoRigidRasterShader, /svoGBufferSurface\(vec3f\(0\.0\),hit\.t,hit\.normal,hit\.normal/);
  const coverageBridge = svoRigidRasterShader.slice(
    svoRigidRasterShader.indexOf("@fragment fn rigidSplitBridgeFragment"),
    svoRigidRasterShader.indexOf("@compute", svoRigidRasterShader.indexOf("@fragment fn rigidSplitBridgeFragment")));
  assert.match(coverageBridge, /if\(geometry\.x==0u\)\{discard;\}/);
  assert.match(coverageBridge, /if\(owner!=input\.bodyIndex\)\{discard;\}/);
  assert.doesNotMatch(coverageBridge, /rigidBodyHit\(/,
    "the certificate bridge must not repeat analytic intersection");
  assert.match(coverageBridge, /vec4f\(rigidUnpackPrimaryNormal\(geometry\.y\),bitcast<f32>\(geometry\.x\)\)/);
  assert.doesNotMatch(coverageBridge, /textureStore|global_invocation_id/,
    "coverage bridge writes render attachments and never scans the viewport");
  assert.doesNotMatch(coverageBridge, /frag_depth|viewDepth/,
    "the packed current-frame winner is sufficient visibility authority; the bridge needs no depth attachment");
  assert.equal(SVO_RIGID_RASTER_CONTRACT.splitBridgeEntryPoint, "rigidSplitBridgeFragment");
  assert.equal(SVO_RIGID_RASTER_CONTRACT.splitBridgeVisibilityAuthority, "current-frame-depth-tested-certificate");
  assert.equal(SVO_RIGID_RASTER_CONTRACT.splitBridgeUsesDepthAttachment, false);
  assert.equal(SVO_RIGID_RASTER_CONTRACT.splitBridgeDepthCompare, "always");
  assert.equal(SVO_RIGID_RASTER_CONTRACT.splitBridgeDepthWriteEnabled, false);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
let sharedGpu: GPU | undefined;
let sharedDevice: Promise<GPUDevice> | undefined;

function device(): Promise<GPUDevice> {
  sharedDevice ??= (async () => {
    const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
      create(options: string[]): GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    sharedGpu = create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await sharedGpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter, "an adapter is required for rigid raster WGSL validation");
    return adapter.requestDevice();
  })();
  return sharedDevice;
}

after(async () => { (await sharedDevice)?.destroy(); });

test("rigid analytic raster WGSL compiles and its production-format pipeline validates on Dawn", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for Dawn rigid-raster validation",
}, async () => {
  const gpuDevice = await device();
  const shaderModule = gpuDevice.createShaderModule({ label: "SVO rigid analytic raster", code: svoRigidRasterShader });
  const info = await shaderModule.getCompilationInfo();
  assert.deepEqual(info.messages.filter(({ type }) => type === "error")
    .map(({ lineNum, linePos, message }) => `${lineNum}:${linePos} ${message}`), []);
  const inputEntries = svoRigidRasterInputBindGroupLayoutEntries();
  const outputEntries = svoRigidRasterOutputBindGroupLayoutEntries();
  const coverageBridgeEntries = svoRigidRasterCoverageBridgeBindGroupLayoutEntries();
  assert.deepEqual(inputEntries.map(({ binding, visibility, buffer }) => ({ binding, visibility, type: buffer?.type })), [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, type: "uniform" },
    { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, type: "read-only-storage" },
    { binding: 14, visibility: GPUShaderStage.FRAGMENT, type: "uniform" },
  ]);
  const layout = gpuDevice.createBindGroupLayout({ entries: inputEntries });
  const outputLayout = gpuDevice.createBindGroupLayout({ entries: outputEntries });
  assert.deepEqual(coverageBridgeEntries.map(({ binding, visibility, texture }) => ({
    binding, visibility, sampleType: texture?.sampleType,
  })), [
    { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampleType: "uint" },
  ]);
  const coverageBridgeLayout = gpuDevice.createBindGroupLayout({ entries: coverageBridgeEntries });
  gpuDevice.pushErrorScope("validation");
  const pipeline = await gpuDevice.createRenderPipelineAsync({
    layout: gpuDevice.createPipelineLayout({ bindGroupLayouts: [layout, outputLayout] }),
    vertex: { module: shaderModule, entryPoint: "rigidRasterVertex" },
    fragment: { module: shaderModule, entryPoint: "rigidRasterFragment", targets: [
      { format: SVO_RIGID_RASTER_CONTRACT.packedSurfaceFormat },
      { format: SVO_RIGID_RASTER_CONTRACT.identityMediaFormat },
      { format: SVO_RIGID_RASTER_CONTRACT.primaryGeometryFormat },
    ] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: {
      format: SVO_RIGID_RASTER_CONTRACT.depthFormat,
      depthWriteEnabled: SVO_RIGID_RASTER_CONTRACT.depthWriteEnabled,
      depthCompare: SVO_RIGID_RASTER_CONTRACT.depthCompare,
    },
  });
  const validation = await gpuDevice.popErrorScope();
  assert.equal(validation?.message, undefined);
  assert.ok(pipeline);
  gpuDevice.pushErrorScope("validation");
  const coverageBridge = await gpuDevice.createRenderPipelineAsync({
    layout: gpuDevice.createPipelineLayout({ bindGroupLayouts: [layout, coverageBridgeLayout] }),
    vertex: { module: shaderModule, entryPoint: "rigidRasterVertex" },
    fragment: { module: shaderModule, entryPoint: SVO_RIGID_RASTER_CONTRACT.splitBridgeEntryPoint, targets: [
      { format: SVO_RIGID_RASTER_CONTRACT.geometryFormat },
      { format: SVO_RIGID_RASTER_CONTRACT.identityFormat },
    ] },
    primitive: { topology: "triangle-list", cullMode: "none" },
  });
  const coverageBridgeValidation = await gpuDevice.popErrorScope();
  assert.equal(coverageBridgeValidation?.message, undefined);
  assert.ok(coverageBridge);
  gpuDevice.pushErrorScope("validation");
  const bridge = await gpuDevice.createComputePipelineAsync({
    layout: gpuDevice.createPipelineLayout({ bindGroupLayouts: [layout, outputLayout] }),
    compute: { module: shaderModule, entryPoint: "rigidSplitBridge" },
  });
  const bridgeValidation = await gpuDevice.popErrorScope();
  assert.equal(bridgeValidation?.message, undefined);
  assert.ok(bridge);
});
