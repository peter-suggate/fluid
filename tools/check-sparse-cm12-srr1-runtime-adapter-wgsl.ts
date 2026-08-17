#!/usr/bin/env node
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import {
  WebGPUSparseCM12SRR1RuntimeAdapter,
  createSparseCM12SRR1RuntimePlan,
  sparseCM12SRR1RuntimeABIReceipt,
} from "../lib/methods/adaptive-mass/sparse-cm12-srr1-runtime-adapter";
import { createSparseCM12SRR1ResidentIngressWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-srr1-resident-ingress.wgsl";
import { gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";

async function main(): Promise<void> {
  const dawnModule = process.env.WEBGPU_NODE_MODULE;
  if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");
  const temporal = createSparseCM12SRR1RuntimePlan({
    baseWords: 0, tileCapacity: 64, constructionMode: "temporal",
  });
  const oracle = createSparseCM12SRR1RuntimePlan({
    baseWords: 0, tileCapacity: 64, constructionMode: "immutable-full-oracle",
  });
  await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-srr1-runtime-adapter");
  try {
    const { create, globals } = await import(dawnModule) as {
      create: (flags: string[]) => GPU; globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    device.pushErrorScope("validation");
    const temporalRuntime = await WebGPUSparseCM12SRR1RuntimeAdapter.create(device, temporal);
    const oracleRuntime = await WebGPUSparseCM12SRR1RuntimeAdapter.create(device, oracle);
    const ingressHelpers = createSparseCM12SRR1ResidentIngressWGSL({
      layout: temporal.ingressLayout, arenaName: "activity",
    });
    const ingressModule = device.createShaderModule({ label: "SIR1 resident helper check",
      code: `@group(0) @binding(0) var<storage,read_write> activity:array<atomic<u32>>;
${ingressHelpers}
@compute @workgroup_size(1) fn checkSIR1(){
  _=sirResidentBeginFrame(1u,1u,0u);
  _=sirResidentAppendEvent(0u,1u,1u);
  sirResidentBeginReceiptBatch();
  _=sirResidentPublishReceipt(0u,sirResidentWorkTile(0u),1u,1u,1u,1u,
    1u,1u,0u,0u,1u,1u,1u,1u);
  sirResidentResetCopiedEvents(0u);
}` });
    await gpuCompilationManagerFor(device).compileComputePipeline({ layout: "auto",
      compute: { module: ingressModule, entryPoint: "checkSIR1" },
    }, { priority: "visible" });
    const scope = await device.popErrorScope();
    temporalRuntime.destroy(); oracleRuntime.destroy();
    if (scope) throw new Error(scope.message);
    console.log(JSON.stringify({ temporal: sparseCM12SRR1RuntimeABIReceipt(temporal),
      immutableFullOracle: sparseCM12SRR1RuntimeABIReceipt(oracle),
      managedAsyncPipelines: temporal.pipelines.length + oracle.pipelines.length + 1,
      synchronousPipelineCalls: 0 }, null, 2));
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
