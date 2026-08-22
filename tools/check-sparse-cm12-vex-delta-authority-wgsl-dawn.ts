#!/usr/bin/env node
/** Focused Dawn compilation for VDA1 plus the optional FCA D4 seam. */
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createSparseCM12FrameControl } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-control";
import { createSparseCM12FrameControlWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-control.wgsl";
import { createSparseCM12VelocityExtensionLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import { createSparseCM12VexDeltaAuthorityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-delta-authority";
import { createSparseCM12VexDeltaAuthorityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-delta-authority.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");
await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-vda1");
let device: GPUDevice | undefined;
try {
  const { create, globals } = await import(dawnModule) as {
    create: (flags: string[]) => GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();if (!adapter) throw new Error("no adapter");
  device = await adapter.requestDevice();
  const delta = createSparseCM12VexDeltaAuthorityLayout({ cellCapacity: 256 });
  const velocity = createSparseCM12VelocityExtensionLayout({
    baseWords: delta.totalWords, cellCapacity: 256 });
  const frame = createSparseCM12FrameControl({ baseWords: velocity.totalWords,
    cellWorkgroups: 4, rowWorkgroups: 6, d4Capable: true });
  const source = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> activity:array<atomic<u32>>;
fn candidateActive(cell:u32)->bool{return cell<128u;}
fn candidateRetired(cell:u32)->bool{return cell>=128u&&cell<256u;}
fn candidateTopologyGeneration()->u32{return 7u;}
${createSparseCM12FrameControlWGSL({ layout: frame.layout, controlName: "activity",
    authorizedD4Invalidation: true })}
${createSparseCM12VexDeltaAuthorityWGSL({ layout: delta,
    velocityExtensionLayout: velocity, arenaName: "activity",
    finalCellActiveFunction: "candidateActive",
    finalCellRetiredFunction: "candidateRetired",
    topologyGenerationExpression: "candidateTopologyGeneration()",
    frameGenerationExpression: "cm12FCCandidateGeneration()",
    framePhaseValidExpression: `cm12FCLoad(${frame.layout.baseWords + 13}u)>=1u`
  })}
@compute @workgroup_size(1) fn beginCheck(){
  vda1BeginPreflight(7u,2u,1u,0u,0u,7u,true,true);
}
@compute @workgroup_size(64) fn populateCheck(@builtin(global_invocation_id) gid:vec3u){
  if(gid.x<2u){vda1PreflightRoot(3u+gid.x,8u);}
  if(gid.x==0u){vda1PreflightRetired(130u);}
}
@compute @workgroup_size(1) fn sealCheck(){_=vda1SealPreflight(7u,vda1VexGeneration());}
@compute @workgroup_size(64) fn publishCheck(@builtin(global_invocation_id) gid:vec3u){
  if(gid.x<2u){vda1PublishAuthorizedRoot(3u+gid.x,8u);}
  if(gid.x==0u){vda1PublishAuthorizedRetirement(130u,8u);}
}
@compute @workgroup_size(1) fn commitCheck(){
  vda1SealPublicationNoFail();vda1CommitSuccessNoFail(7u);
  if(vda1TransactionSucceeded(7u)){
    vda1MarkInjectionPublishedNoFail();
    cm12FCInvalidateD4Authorized(1u,0u,7u,true);
    vda1MarkD4PublishedNoFail();
  }
}
`;
  const module = device.createShaderModule({ code: source });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) throw new Error(errors.map((m) => m.message).join("\n"));
  const bgl = device.createBindGroupLayout({ entries: [{ binding: 0,
    visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const entryPoints = ["beginCheck", "populateCheck", "sealCheck", "publishCheck",
    "commitCheck"];
  for (const entryPoint of entryPoints) await device.createComputePipelineAsync({
    layout: pipelineLayout, compute: { module, entryPoint },
  });
  console.log(JSON.stringify({ schema: "sparse-cm12-vda1-wgsl-check/v1",
    entryPoints, passed: true }, null, 2));
} finally {
  device?.destroy();releaseWebGPUExclusiveLock();
}
