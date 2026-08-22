#!/usr/bin/env node
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createSparseCM12TopologyEffectsAuthorityInitialWords,
  createSparseCM12TopologyEffectsAuthorityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-topology-effects-authority";
import { createSparseCM12TopologyEffectsAuthorityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-topology-effects-authority.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) {
  console.error("WEBGPU_NODE_MODULE is required");
  process.exit(2);
}
const layout = createSparseCM12TopologyEffectsAuthorityLayout({
  baseWords: 0, ptrCapacity: 8, ptrLeafCapacity: 1,
});
const source = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> topologyArena:array<atomic<u32>>;
fn transactionAuthorized()->bool{return true;}
fn tfxPTRTargetGeneration()->u32{return 11u;}
fn tfxPTRReady(_generation:u32,_newCount:u32,_newLeafCount:u32)->bool{return true;}
fn tfxPTRWillAppend(_brick:u32,_generation:u32)->bool{return true;}
fn tfxPTRDirtyLeafWillAppend(_leaf:u32,_generation:u32)->bool{return true;}
fn tfxPTRCompatible(_brick:u32,_oldState:u32,_newState:u32)->bool{return true;}
fn tfxPTRPublish(_brick:u32,_oldState:u32,_newState:u32,_cause:u32,
 _ownsLeaf:bool,_generation:u32){}
${createSparseCM12TopologyEffectsAuthorityWGSL({ layout,
  arenaName: "topologyArena", authorizationExpression: "transactionAuthorized()" })}
`;
const entryPoints = ["beginSparseCM12TopologyEffectsPreflight",
  "finalizeSparseCM12TopologyEffectsPreflight", "publishSparseCM12TopologyPTREffects",
  "finishSparseCM12TopologyEffectsPublication"] as const;

await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-topology-effects-authority");
let gpu: GPU | undefined;
let device: GPUDevice | undefined;
try {
  const dawn = await import(dawnModule) as { create: (flags: string[]) => GPU;
    globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("no adapter");
  device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ label: "TFX1 checker", code: source });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length) {
    for (const error of errors) console.error(`${error.lineNum}:${error.linePos} ${error.message}`);
    throw new Error(`TFX1 WGSL has ${errors.length} errors`);
  }
  for (const entryPoint of entryPoints) {
    await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint } });
  }
  const validation = await device.popErrorScope();
  if (validation) throw validation;
  console.log(JSON.stringify({ schema: "sparse-cm12-topology-effects-authority-wgsl/v1",
    passed: true, entryPoints, bytes: createSparseCM12TopologyEffectsAuthorityInitialWords(layout).byteLength }));
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
  void gpu;
}
