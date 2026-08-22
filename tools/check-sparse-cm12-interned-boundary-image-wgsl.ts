#!/usr/bin/env node
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import type { SparseCM12InternedBoundaryLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";
import { createSparseCM12InternedBoundaryImageWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-image.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) {
  console.error("WEBGPU_NODE_MODULE is required");
  process.exit(2);
}
const layout: SparseCM12InternedBoundaryLayout = {
  leafCapacity: 8, canonicalCapacity: 40, templateCount: 4,
  templatePayloadWords: 256, canonicalBaseWords: 32,
  templateDirectoryBaseWords: 672, templatePayloadBaseWords: 688,
  immutableWords: 960, immutableBytes: 3840,
  slotBaseWords: [960, 1664], slotLeafBaseWords: [992, 1696],
  slotRefBaseWords: [1056, 1760], wordsPerSlot: 704, bytesPerSlot: 2816,
  totalWords: 2368, totalBytes: 9472,
};
const helpers = createSparseCM12InternedBoundaryImageWGSL({ layout,
  arenaName: "topologyArena", hookPrefix: "fixture", baseWords: 4096,
  packetsPerLeaf: 64, acceptedSlotHook: "sharedAcceptedSlot",
  acceptedGenerationHook: "sharedAcceptedGeneration" });
const source = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> topologyArena:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> output:array<atomic<u32>>;
fn sharedAcceptedSlot()->u32{return 0u;}
fn sharedAcceptedGeneration()->u32{return 1u;}
${helpers}
@compute @workgroup_size(1)
fn checkIBO1(){let slot=fixtureIBOAcceptedSlot();let generation=fixtureIBOAcceptedGeneration();
  let immutable=fixtureIBOValidateImmutableContent(
    fixtureIBOImmutableCertificate().w,fixtureIBOImmutableCertificateHash());
  let descriptor=fixtureIBOLeafDescriptorId(slot,0u);
  let rowBase=fixtureIBOCanonicalRowBase(descriptor,0u);
  let dimensions=fixtureIBOLeafDimensions(slot,0u);
  let count=fixtureIBOFaceRefCount(slot,0u,0u);let reference=fixtureIBORef(slot,0u,0u,0u);
  let directory=fixtureIBOTemplateDirectory(reference.x);
  let row=fixtureIBOTemplateRowWord(reference.x,0u,0u);
  let term=fixtureIBOTemplateTermWord(reference.x,0u,0u);
  let begin=fixtureIBOBeginDeltaLeaf(1u-slot,0u,generation+1u,true,descriptor);
  let write=fixtureIBOWriteDeltaRef(1u-slot,0u,0u,0u,vec3u(0u,1u,0u));
  let seal=fixtureIBOSealDeltaLeaf(1u-slot,0u);
  let validation=fixtureIBOValidateDeltaLeaf(slot,0u,generation);
  let scheduled=fixtureIBOValidateScheduledDeltaLeaf(slot,0u,generation,
    true,descriptor,fixtureIBOLeafWord(slot,0u,5u),fixtureIBOLeafWord(slot,0u,6u));
  fixtureIBOReplayDeltaLeaf(slot,1u-slot,0u,generation);
  atomicStore(&output[0],rowBase|dimensions.x|count|directory.x|row|term
    |begin.x|write.x|seal.x|validation.x|scheduled.x|immutable.x
    |fixtureIBOStablePacketLeaf(0u));}
`;

await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-interned-boundary-image");
let gpu: GPU | undefined;
let device: GPUDevice | undefined;
try {
  const dawn = await import(dawnModule) as {
    create: (flags: string[]) => GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ label: "IBO1 standalone checker", code: source });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length > 0) {
    for (const error of errors) console.error(`${error.lineNum}:${error.linePos} ${error.message}`);
    throw new Error(`IBO1 WGSL has ${errors.length} compilation errors`);
  }
  await device.createComputePipelineAsync({ layout: "auto",
    compute: { module, entryPoint: "checkIBO1" } });
  const validation = await device.popErrorScope();
  if (validation) throw validation;
  console.log(JSON.stringify({ schema: "sparse-cm12-interned-boundary-image-wgsl/v1",
    baseWords: 4096, packetsPerLeaf: 64, passed: true }));
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
  void gpu;
}
