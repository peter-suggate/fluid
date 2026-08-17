#!/usr/bin/env node
/** Compile the standalone VEX1 library in the complete resident shader context. */
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createSparseCM12DirtySchedulerLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-dirty-scheduler";
import { createSparseCM12IncrementalActivityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-incremental-activity";
import { createSparseCM12VelocityExtensionLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import { createSparseCM12VelocityExtensionWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl";
import { createWebgpuSparseCM12ResidentWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";

const entryPoints = (source: string): readonly string[] =>
  [...source.matchAll(/@compute[^\n]*\nfn\s+([A-Za-z0-9_]+)/g)]
    .map((match) => match[1]!);

function integratedSource(): string {
  const dirty = createSparseCM12DirtySchedulerLayout({ logicalBrickCount: 8,
    brickFineResolution: 16, journalCapacity: 512, packingPacketCount: 6 });
  const temporal = { headerBaseWords: dirty.totalWords + 64,
    cellListBaseWords: dirty.totalWords + 72,
    rowListBaseWords: dirty.totalWords + 4168,
    cellFlagABaseWords: dirty.totalWords + 8264,
    cellFlagBBaseWords: dirty.totalWords + 12360,
    totalWords: dirty.totalWords + 16456 };
  const pressure = { edgeCoefficientBaseWords: 4096, cellSlotBaseWords: 8192,
    rowSlotBaseWords: 12288, cellChangeBaseWords: 16384,
    rowChangeBaseWords: 20480, brickStateBaseWords: 24576,
    rowTopologyStampBaseWords: 24640, aggregateEdgeForFineEdgeBaseWords: 28736,
    aggregateEdgeSourceBaseWords: 32832,
    hierarchyEdgeForAggregateBaseWords: [32896],
    headerBaseWords: 36992, totalWords: 37000 };
  const activity = createSparseCM12IncrementalActivityLayout({
    baseWords: temporal.totalWords, stableTileCount: 512, brickCount: 8,
  });
  const extension = createSparseCM12VelocityExtensionLayout({
    baseWords: activity.totalWords, cellCapacity: 4096,
  });
  return createWebgpuSparseCM12ResidentWGSL(
    16, 16, dirty, temporal, pressure, activity,
  ) + createSparseCM12VelocityExtensionWGSL({
    layout: extension,
    // Compile-only tail address. The resident integration supplies the exact
    // state-layout address after the optional pressure journal.
    acceptedVelocityFloatBase: 200_000,
  });
}

if (process.argv.includes("--emit-wgsl")) {
  process.stdout.write(integratedSource());
} else {
  const dawnModule = process.env.WEBGPU_NODE_MODULE;
  if (!dawnModule) {
    console.error("WEBGPU_NODE_MODULE is required");
    process.exit(2);
  }
  void main(dawnModule).catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : error);
    process.exit(1);
  });
}

async function main(dawnModule: string): Promise<void> {
  const { create, globals } = await import(dawnModule) as {
    create: (flags: string[]) => GPU;
    globals: { GPUBufferUsage: unknown };
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const storage = { type: "storage" } as const;
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } },
      ...[2, 3, 4, 11, 12, 13].map((binding) => ({ binding,
        visibility: GPUShaderStage.COMPUTE, buffer: storage })),
      { binding: 14, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } },
      ...[15, 16].map((binding) => ({ binding,
        visibility: GPUShaderStage.COMPUTE, buffer: storage })),
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const source = integratedSource();
  const shaderModule = device.createShaderModule({ code: source });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  for (const error of errors) {
    console.error(`${error.lineNum}:${error.linePos} ${error.message}`);
  }
  if (errors.length > 0) throw new Error(`${errors.length} WGSL compilation error(s)`);
  const names = entryPoints(source);
  await Promise.all(names.map((entryPoint) => device.createComputePipelineAsync({
    layout: pipelineLayout, compute: { module: shaderModule, entryPoint },
  })));
  console.log(`Sparse CM12 VEX1: ${names.length} integrated entry points compiled (B16/P16)`);
}
