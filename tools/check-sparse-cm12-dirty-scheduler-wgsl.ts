#!/usr/bin/env node
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import {
  createSparseCM12DirtySchedulerInitialWords,
  createSparseCM12DirtySchedulerLayout,
  SPARSE_CM12_DIRTY_DOMAIN,
  sparseCM12DirtyDomainMask,
} from "../lib/methods/adaptive-mass/sparse-cm12-dirty-scheduler";
import { createSparseCM12DirtySchedulerWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-dirty-scheduler.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");

async function main(): Promise<void> {
  const layout = createSparseCM12DirtySchedulerLayout({
    logicalBrickCount: 2,
    brickFineResolution: 16,
    journalCapacity: 512,
    packingPacketCount: 6,
  });
  const initial = createSparseCM12DirtySchedulerInitialWords(layout, {
    acceptedGeneration: 1,
    acceptedCleanBootstrap: true,
  });
  if (initial.byteLength !== layout.totalBytes) throw new Error("initializer/layout size mismatch");
  const helpers = createSparseCM12DirtySchedulerWGSL({
    layout,
    arenaName: "schedulerArena",
    workgroupSize: 64,
  });
  const domainMask = sparseCM12DirtyDomainMask(
    SPARSE_CM12_DIRTY_DOMAIN.cellScalars,
    SPARSE_CM12_DIRTY_DOMAIN.faceRows,
  );
  const source = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> schedulerArena:array<atomic<u32>>;
${helpers}
@compute @workgroup_size(1)
fn checkScheduler(@builtin(global_invocation_id) gid:vec3u){
  if(gid.x==0u){
    _=cm12DirtyBeginCandidate(1u,2u);
    let receipt=cm12DirtyRecordEvent(0u,0u,1u,${domainMask}u,1u,0u,true,true);
    _=cm12DirtyRecordCoverage(0u,0u,1u,${domainMask}u,receipt);
    _=cm12DirtyRequestClosure(0u,0u,1u,${domainMask}u);
    _=cm12DirtyResolveClosure(0u,0u,1u);
    cm12DirtyMarkExecuted(0u,1u,2u);
    cm12DirtyMarkSkipped(0u,2u,2u);
    _=cm12DirtyEnqueueClosure(0u,false);
    cm12DirtySealQueue(0u);
    cm12DirtyResetQueue(4u);
    cm12DirtySealPreOpenPost();
    _=cm12DirtyFinalizePacket(0u);
    _=cm12DirtyFinalizeEpoch();
    _=cm12DirtyPromotePostToPre(0u,1u);
  }
}`;

  await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-dirty-scheduler");
  try {
    const { create, globals } = await import(dawnModule!) as {
      create: (flags: string[]) => GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    device.pushErrorScope("validation");
    const shaderModule = device.createShaderModule({ label: "CMS1 scheduler WGSL check", code: source });
    const info = await shaderModule.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    for (const error of errors) console.error(`${error.lineNum}:${error.linePos} ${error.message}`);
    if (errors.length > 0) throw new Error(`${errors.length} WGSL compilation error(s)`);
    await device.createComputePipelineAsync({
      label: "CMS1 scheduler WGSL check pipeline",
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "checkScheduler" },
    });
    const scope = await device.popErrorScope();
    if (scope) throw new Error(scope.message);
    console.log(`Sparse CM12 dirty scheduler WGSL: valid (${source.length} source bytes)`);
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
