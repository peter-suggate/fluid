#!/usr/bin/env node
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import {
  createSparseCM12CanonicalMembershipInitialWords,
  createSparseCM12CanonicalMembershipLayout,
} from "../lib/methods/adaptive-mass/sparse-cm12-canonical-membership";
import { createSparseCM12CanonicalMembershipWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-canonical-membership.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");

async function main(): Promise<void> {
  const layout = createSparseCM12CanonicalMembershipLayout({
    cellCapacity: 4097,
    rowCapacity: 12289,
    baseWords: 128,
  });
  const initial = createSparseCM12CanonicalMembershipInitialWords(layout);
  if (initial.byteLength !== layout.totalBytes) throw new Error("initializer/layout mismatch");
  const helpers = createSparseCM12CanonicalMembershipWGSL({
    layout,
    arenaName: "membershipArena",
    workgroupSize: 64,
  });
  const source = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> membershipArena:array<atomic<u32>>;
${helpers}
@compute @workgroup_size(1)
fn checkBegin(){
  _=pcmCellBegin(0u);_=pcmCellSetCandidate(0u,true,1u,false);
  _=pcmCellSetCandidate(4096u,true,2u,false);_=pcmCellFinalizeFrontier();
  _=pcmRowBegin(1u);_=pcmRowSetCandidate(1u,true,4u,true);_=pcmRowFinalizeFrontier();
}
@compute @workgroup_size(1)
fn checkFinalize(){_=pcmCellFinalize();_=pcmRowFinalize();}
@compute @workgroup_size(1)
fn checkRank(){_=pcmCellRankSelect(0u);_=pcmRowRankSelect(0u);
  _=pcmCellContains(0u);_=pcmRowContains(1u);}
`;
  await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-canonical-membership");
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
    const shaderModule = device.createShaderModule({ label: "PCM1 WGSL check", code: source });
    const info = await shaderModule.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    for (const error of errors) console.error(`${error.lineNum}:${error.linePos} ${error.message}`);
    if (errors.length > 0) throw new Error(`${errors.length} WGSL compilation error(s)`);
    for (const entryPoint of ["checkBegin", "repairCanonicalPressureCellLeaves",
      "repairCanonicalPressureRowLeaves", "checkFinalize", "checkRank"] as const) {
      await device.createComputePipelineAsync({
        label: `PCM1 ${entryPoint}`,
        layout: "auto",
        compute: { module: shaderModule, entryPoint },
      });
    }
    const scope = await device.popErrorScope();
    if (scope) throw new Error(scope.message);
    console.log(`Sparse CM12 canonical membership WGSL: valid (${source.length} source bytes)`);
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
