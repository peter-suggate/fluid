#!/usr/bin/env node
/** Standalone Dawn validation for the binding-free VXP1 WGSL fragment. */
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createSparseCM12VelocityExtensionLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import { createSparseCM12VexPacketFrontierLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-packet-frontier";
import { createSparseCM12VexPacketFrontierWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-packet-frontier.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");

await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-vxp1");
let device: GPUDevice | undefined;
try {
  const { create, globals } = await import(dawnModule) as {
    create: (flags: string[]) => GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();if (!adapter) throw new Error("no WebGPU adapter");
  device = await adapter.requestDevice();
  const velocity = createSparseCM12VelocityExtensionLayout({ cellCapacity: 1024 });
  const packet = createSparseCM12VexPacketFrontierLayout({
    baseWords: velocity.frontierABaseWords,
    availableWords: 3 * velocity.cellCapacity,
    packetCapacity: 128,
  });
  const helper = createSparseCM12VexPacketFrontierWGSL({
    layout: packet, velocityExtensionLayout: velocity,
    generationExpression: "frameGeneration()",
    topologyGenerationExpression: "acceptedTopologyGeneration()",
    topologySlotExpression: "acceptedTopologySlot()",
    packetCellFunction: "teiPacketCell", cellPacketLaneFunction: "teiCellPacketLane",
    targetCellActiveFunction: "cellActiveInSlot",
    currentCellActiveFunction: "cellActive",
    rootReceiptFunction: "recordRootReceipt",
    closureReceiptFunction: "recordClosureReceipt",
  });
  const stub = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> activity:array<atomic<u32>>;
fn frameGeneration()->u32{return 1u;}
fn acceptedTopologyGeneration()->u32{return 1u;}
fn acceptedTopologySlot()->u32{return 0u;}
fn teiPacketCell(packet:u32,lane:u32,slot:u32)->u32{_=slot;return packet*64u+lane;}
fn teiCellPacketLane(cell:u32,slot:u32)->vec2u{_=slot;return vec2u(cell/64u,cell%64u);}
fn cellActive(cell:u32)->bool{return cell<1024u;}
fn cellActiveInSlot(cell:u32,slot:u32)->bool{_=slot;return cell<1024u;}
fn recordRootReceipt(cell:u32,cause:u32,generation:u32)->bool{
  _=cell;_=cause;_=generation;return true;}
fn recordClosureReceipt(cell:u32,depth:u32,generation:u32)->bool{
  _=cell;_=depth;_=generation;return true;}
`;
  const source = stub + helper;
  const module = device.createShaderModule({ code: source });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) throw new Error(errors.map((message) => message.message).join("\n"));
  const bindGroupLayout = device.createBindGroupLayout({ entries: [{ binding: 0,
    visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const entryPoints = [...source.matchAll(
    /@compute\s+@workgroup_size\([^)]*\)\s+fn\s+([A-Za-z0-9_]+)/g,
  )].map((match) => match[1]!);
  for (const entryPoint of entryPoints) {
    await device.createComputePipelineAsync({ layout: pipelineLayout,
      compute: { module, entryPoint } });
  }
  console.log(JSON.stringify({ schema: "sparse-cm12-vxp1-wgsl-check/v1",
    entryPointCount: entryPoints.length, entryPoints, passed: true }, null, 2));
} finally {
  device?.destroy();
  releaseWebGPUExclusiveLock();
}

