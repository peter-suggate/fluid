#!/usr/bin/env node
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import type { SparseCM12InternedRefLookupLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-ref-lookup";
import { createSparseCM12InternedRefLookupWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-ref-lookup.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) {
  console.error("WEBGPU_NODE_MODULE is required");process.exit(2);
}
const layout: SparseCM12InternedRefLookupLayout = {
  baseWords: 4096, canonicalCapacity: 40, sideDirectoryCount: 240,
  directoryBaseWords: 4112, templateDirectoryBaseWords: 4232,
  entryBaseWords: 4234, templateCount: 2, templateEntryCount: 4, entryCount: 64,
  fallbackAnchorBaseWords: 4242, fallbackAnchorCount: 40, levelsPerLeaf: 4,
  maximumEntriesPerSide: 3, totalWords: 4416, totalBytes: 1280,
};
const helpers = createSparseCM12InternedRefLookupWGSL({ layout,
  arenaName: "topologyArena", iboPrefix: "fixture", baseWords: 65536 });
const source = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> topologyArena:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> output:array<atomic<u32>>;
fn fixtureIBOCanonicalWord(descriptor:u32,word:u32)->u32{
  if(word==1u){return descriptor/4u;}return 1u;}
fn fixtureIBOCanonicalRowBase(descriptor:u32,axis:u32)->u32{
  _=descriptor;_=axis;return 1u;}
${helpers}
@compute @workgroup_size(1)
fn checkIRL1(){let count=fixtureIBOInstantiationCount(0u,0u);
  let entry=fixtureIBOInstantiationEntry(0u,0u,0u);
  let reference=fixtureIBOFindScheduledRef(0u,0u,entry.x,
    fixtureIBOCanonicalWord(entry.x,1u));
  atomicStore(&output[0],count|entry.y|reference.x);}
`;

await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-interned-ref-lookup");
let gpu: GPU | undefined;
let device: GPUDevice | undefined;
try {
  const dawn = await import(dawnModule) as {
    create: (flags: string[]) => GPU;globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();if (!adapter) throw new Error("no adapter");
  device = await adapter.requestDevice();device.pushErrorScope("validation");
  const module = device.createShaderModule({ label: "IRL1 checker", code: source });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length > 0) {
    for (const error of errors) console.error(`${error.lineNum}:${error.linePos} ${error.message}`);
    throw new Error(`IRL1 WGSL has ${errors.length} errors`);
  }
  await device.createComputePipelineAsync({ layout: "auto",
    compute: { module, entryPoint: "checkIRL1" } });
  const validation = await device.popErrorScope();if (validation) throw validation;
  console.log(JSON.stringify({ schema: "sparse-cm12-interned-ref-lookup-wgsl/v1",
    maximumEntriesPerSide: layout.maximumEntriesPerSide, passed: true }));
} finally {
  device?.destroy();await releaseWebGPUExclusiveLock();void gpu;
}
