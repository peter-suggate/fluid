#!/usr/bin/env node
/** Dawn compile validation for the standalone IBO1 -> VXP1 packet bridge. */
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createSparseCM12VelocityExtensionLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import { createSparseCM12VexPacketFrontierLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-packet-frontier";
import { createSparseCM12VexPacketFrontierWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-packet-frontier.wgsl";
import type { SparseCM12VexIBO1ImageLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-ibo1-image";
import { createSparseCM12VexIBO1ImageWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-ibo1-image.wgsl";
import { createSparseCM12InternedBoundaryImageWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-image.wgsl";
import type { SparseCM12InternedBoundaryLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");
await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-vxi1");
let device: GPUDevice | undefined;
try {
  const { create, globals } = await import(dawnModule) as {
    create: (flags: string[]) => GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); if (!adapter) throw new Error("no adapter");
  device = await adapter.requestDevice();
  const velocity = createSparseCM12VelocityExtensionLayout({ cellCapacity: 1024 });
  const packet = createSparseCM12VexPacketFrontierLayout({
    baseWords: velocity.frontierABaseWords, availableWords: 3 * velocity.cellCapacity,
    packetCapacity: 128,
  });
  const vxp = createSparseCM12VexPacketFrontierWGSL({ layout: packet,
    velocityExtensionLayout: velocity, generationExpression: "frameGeneration()",
    topologyGenerationExpression: "cm12IBOAcceptedGeneration()",
    topologySlotExpression: "cm12IBOAcceptedSlot()",
    packetCellFunction: "teiPacketCell", cellPacketLaneFunction: "teiCellPacketLane",
    targetCellActiveFunction: "cellActiveInSlot", currentCellActiveFunction: "cellActive",
  });
  const imageLayout: SparseCM12VexIBO1ImageLayout = Object.freeze({
    canonicalCapacity: 8, coreTemplateCount: 2, faceTemplateCount: 2,
    templateCount: 4, canonicalTemplateBaseWords: 32,
    directoryBaseWords: 64, operationBaseWords: 320, operationCount: 16,
    totalWords: 368, totalBytes: 1472,
  });
  const iboLayout: SparseCM12InternedBoundaryLayout = Object.freeze({
    leafCapacity: 2, canonicalCapacity: 8, templateCount: 2,
    templatePayloadWords: 10, canonicalBaseWords: 32,
    templateDirectoryBaseWords: 160, templatePayloadBaseWords: 168,
    immutableWords: 256, immutableBytes: 1024,
    slotBaseWords: [256, 448] as const, slotLeafBaseWords: [288, 480] as const,
    slotRefBaseWords: [304, 496] as const, wordsPerSlot: 192, bytesPerSlot: 768,
    totalWords: 640, totalBytes: 2560,
  });
  const iboHelper = createSparseCM12InternedBoundaryImageWGSL({
    layout: iboLayout, arenaName: "topologyArena", hookPrefix: "cm12",
    packetsPerLeaf: 64, acceptedSlotHook: "sharedAcceptedSlot",
    acceptedGenerationHook: "sharedAcceptedGeneration",
  });
  const vxi = createSparseCM12VexIBO1ImageWGSL({ layout: imageLayout,
    arenaName: "topologyArena", iboPrefix: "cm12",
    sourcePacketCountFunction: "sourcePacketCount",
    sourcePacketFunction: "sourcePacket", sourceMaskFunction: "sourceMask",
    rootCauseExpression: "4u" });
  const stub = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> activity:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> topologyArena:array<atomic<u32>>;
const cm12ExtensionFlagSealed:u32=1u;
const cm12ExtensionPhasePlanned:u32=2u;
fn cm12ExtensionFail(cell:u32,depth:u32){_=cell;_=depth;}
fn frameGeneration()->u32{return 1u;}
fn sharedAcceptedSlot()->u32{return 0u;}
fn sharedAcceptedGeneration()->u32{return 1u;}
fn teiPacketCell(packet:u32,lane:u32,slot:u32)->u32{_=slot;return packet*64u+lane;}
fn teiCellPacketLane(cell:u32,slot:u32)->vec2u{_=slot;return vec2u(cell/64u,cell%64u);}
fn cellActive(cell:u32)->bool{return cell<1024u;}
fn cellActiveInSlot(cell:u32,slot:u32)->bool{_=slot;return cell<1024u;}
fn sourcePacketCount()->u32{return 1u;}
fn sourcePacket(rank:u32)->u32{_=rank;return 0u;}
fn sourceMask(packet:u32)->vec2u{_=packet;return vec2u(1u,0u);}
`;
  const source = stub + iboHelper + vxp + vxi;
  const module = device.createShaderModule({ code: source });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) throw new Error(errors.map((message) =>
    `line ${message.lineNum}: ${message.message}`).join("\n"));
  const bindGroupLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" } },
  ] });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const entryPoints = [...source.matchAll(
    /@compute\s+@workgroup_size\([^)]*\)\s+fn\s+([A-Za-z0-9_]+)/g,
  )].map((match) => match[1]!);
  for (const entryPoint of entryPoints) await device.createComputePipelineAsync({
    layout: pipelineLayout, compute: { module, entryPoint,
      constants: entryPoint === "vxi1ExpandFrontier" ? { VXP1_FRONTIER_DEPTH: 1 } : {} },
  });
  console.log(JSON.stringify({ schema: "sparse-cm12-vxi1-wgsl-check/v1",
    entryPointCount: entryPoints.length, entryPoints, passed: true }, null, 2));
} finally {
  device?.destroy(); releaseWebGPUExclusiveLock();
}
