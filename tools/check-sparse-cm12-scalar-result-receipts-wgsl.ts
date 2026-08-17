#!/usr/bin/env node
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import {
  SPARSE_CM12_SCALAR_RESULT_CAUSE,
  SPARSE_CM12_SCALAR_RESULT_HEADER,
  createSparseCM12ScalarResultAuthority,
  sparseCM12ScalarResultByteMap,
} from "../lib/methods/adaptive-mass/sparse-cm12-scalar-result-receipts";
import { createSparseCM12ScalarResultReceiptsWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-scalar-result-receipts.wgsl";

async function main(): Promise<void> {
  const dawnModule = process.env.WEBGPU_NODE_MODULE;
  if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");
  const authority = createSparseCM12ScalarResultAuthority({ tileCapacity: 64 });
  const byteMap = sparseCM12ScalarResultByteMap(authority.layout);
  const partitionBytes = byteMap.reduce((sum, entry) => sum + entry.sizeBytes, 0);
  if (partitionBytes !== authority.layout.totalBytes
    || byteMap.some((entry, index) => index > 0
      && entry.offsetBytes !== byteMap[index - 1]!.offsetBytes
        + byteMap[index - 1]!.sizeBytes)) {
    throw new Error("SRR1 byte map is not an exact arena partition");
  }
  const helpers = createSparseCM12ScalarResultReceiptsWGSL({
    layout: authority.layout, arenaName: "receipts",
  });
  const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
  const source = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> receipts:array<atomic<u32>>;
${helpers}
@compute @workgroup_size(1)
fn checkScalarResultReceipts(@builtin(global_invocation_id) gid:vec3u){
  if(gid.x!=0u){return;}
  if(!cm12SRRBegin(1u,1u,0u)){return;}
  for(var tile=0u;tile<cm12SRRTileCapacity;tile+=1u){
    _=cm12SRRAppendCandidate(tile,${SPARSE_CM12_SCALAR_RESULT_CAUSE.bootstrap}u);
  }
  for(var leaf=0u;leaf<cm12SRRLoad(${h.leafCount}u);leaf+=1u){
    cm12SRRClassifyCandidate(leaf,1u,11u,12u,31u,64u,21u);
  }
  cm12SRRSeal();
  let scheduled=cm12SRRLoad(${h.scheduledWorkCount}u);
  for(var rank=0u;rank<scheduled;rank+=1u){
    let tile=cm12SRRWorkTile(rank);
    _=cm12SRRPublishExactResult(tile,1u,11u,12u,31u,64u,64u,0u,0u,21u,21u,1u);
  }
  _=cm12SRRCommit();
  _=cm12SRRCleanTile(0u);
}`;

  await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-scalar-result-receipts");
  try {
    const { create, globals } = await import(dawnModule) as {
      create: (flags: string[]) => GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label: "SRR1 WGSL checker", code: source });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    for (const error of errors) console.error(`${error.lineNum}:${error.linePos} ${error.message}`);
    if (errors.length > 0) throw new Error(`${errors.length} WGSL compilation error(s)`);
    await device.createComputePipelineAsync({ layout: "auto",
      compute: { module, entryPoint: "checkScalarResultReceipts" } });
    const scope = await device.popErrorScope();
    if (scope) throw new Error(scope.message);
    console.log(JSON.stringify({ abi: "SRR1", wgsl: "valid",
      sourceBytes: source.length, arenaBytes: authority.layout.totalBytes,
      byteMap }, null, 2));
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
