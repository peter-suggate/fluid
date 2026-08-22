#!/usr/bin/env node
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import type { SparseCM12InternedBoundaryLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";
import { createSparseCM12InternedBoundaryImageWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-image.wgsl";
import { createSparseCM12GeometryFaceNeighborsWGSL,
  createSparseCM12IBOSemanticAuthorityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-ibo-semantic-authority.wgsl";
const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) { console.error("WEBGPU_NODE_MODULE is required"); process.exit(2); }
const layout: SparseCM12InternedBoundaryLayout = { leafCapacity: 8,
  canonicalCapacity: 40, templateCount: 4, templatePayloadWords: 256,
  canonicalBaseWords: 32, templateDirectoryBaseWords: 672,
  templatePayloadBaseWords: 688, immutableWords: 960, immutableBytes: 3840,
  slotBaseWords: [960, 1664], slotLeafBaseWords: [992, 1696],
  slotRefBaseWords: [1056, 1760], wordsPerSlot: 704, bytesPerSlot: 2816,
  totalWords: 2368, totalBytes: 9472 };
const core = createSparseCM12InternedBoundaryImageWGSL({ layout,
  arenaName: "topologyArena", hookPrefix: "fixture", baseWords: 4096,
  packetsPerLeaf: 64, acceptedSlotHook: "acceptedSlot",
  acceptedGenerationHook: "acceptedGeneration" });
const isa = createSparseCM12IBOSemanticAuthorityWGSL({ hookPrefix: "fixture",
  iboPrefix: "fixture" });
const geometry = createSparseCM12GeometryFaceNeighborsWGSL({ baseWords: 8192,
  leafCapacity: 8, offsetBaseWords: 8, neighborBaseWords: 17,
  arenaName: "topologyArena", hookPrefix: "fixture" });
const source = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> topologyArena:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> output:array<atomic<u32>>;
fn acceptedSlot()->u32{return 0u;}fn acceptedGeneration()->u32{return 1u;}
fn fixtureISARowWord(row:u32,plane:u32)->u32{return row+plane;}
fn fixtureISARowTermCount(_row:u32)->u32{return 2u;}
fn fixtureISARowTermFirst(row:u32)->u32{return row*2u;}
fn fixtureISATermCell(term:u32)->u32{return term;}
fn fixtureISATermBits(term:u32)->u32{return term;}
fn fixtureISACandidateFaceRange(_descriptor:u32,_side:u32,_boundary:u32)->vec2u{
  return vec2u(0u);}
fn fixtureISACandidateFaceRow(index:u32)->u32{return index;}
fn fixtureISAScheduledRow(_row:u32)->bool{return true;}
fn fixtureISAClaimClosureLeaf(_leaf:u32,_generation:u32)->bool{return true;}
fn fixtureISAAppendClosureLeaf(_leaf:u32){}
${core}
${geometry}
${isa}
@compute @workgroup_size(1) fn checkISA1(){fixtureISAAppendGeometryClosure(0u,2u);
  let receipt=fixtureISAValidateLeaf(0u,0u,fixtureIBOLeafDescriptorId(0u,0u));
  atomicStore(&output[0],receipt.x);}
`;
await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-ibo-semantic-authority");
let gpu: GPU | undefined; let device: GPUDevice | undefined;
try { const dawn = await import(dawnModule) as { create:(flags:string[])=>GPU;
    globals:Record<string,unknown> }; Object.assign(globalThis,dawn.globals);
  gpu=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();if(!adapter)throw new Error("no adapter");
  device=await adapter.requestDevice();device.pushErrorScope("validation");
  const module=device.createShaderModule({label:"ISA1 checker",code:source});
  const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==="error");
  if(errors.length){for(const e of errors)console.error(`${e.lineNum}:${e.linePos} ${e.message}`);
    throw new Error(`ISA1 WGSL has ${errors.length} errors`);}
  await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"checkISA1"}});
  const validation=await device.popErrorScope();if(validation)throw validation;
  console.log(JSON.stringify({schema:"sparse-cm12-ibo-semantic-authority-wgsl/v1",passed:true}));
}finally{device?.destroy();await releaseWebGPUExclusiveLock();void gpu;}
