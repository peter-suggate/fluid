#!/usr/bin/env node
import { createSparseCM12DynamicClosureLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-dynamic-closure-authority";
import { createSparseCM12DynamicClosureAuthorityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-dynamic-closure-authority.wgsl";
import { createSparseCM12IboTRASupplementWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-ibo-tra-supplement.wgsl";
import { createSparseCM12IboTRAResidentHooksWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-ibo-tra-resident-hooks.wgsl";
import { createSparseCM12InternedBoundaryImageWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-image.wgsl";
import type { SparseCM12InternedBoundaryLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");
const dca = createSparseCM12DynamicClosureLayout({ sourcePacketCapacity: 64,
  targetPacketCapacity: 64 });
const ibo: SparseCM12InternedBoundaryLayout = { leafCapacity: 1,
  canonicalCapacity: 1, templateCount: 2, templatePayloadWords: 64,
  canonicalBaseWords: 64, templateDirectoryBaseWords: 128,
  templatePayloadBaseWords: 192, immutableWords: 256, immutableBytes: 1024,
  slotBaseWords: [256, 512], slotLeafBaseWords: [320, 576],
  slotRefBaseWords: [384, 640], wordsPerSlot: 256, bytesPerSlot: 1024,
  totalWords: 768, totalBytes: 3072 };
const iboBase = dca.totalWords;
const itrBase = iboBase + ibo.totalWords;
const itr = { baseWords: itrBase, templateCount: 2,
  directoryBaseWords: itrBase + 16, totalWords: itrBase + 1024,
  totalBytes: 4096 };
const source = /* wgsl */ `
@group(0)@binding(0)var<storage,read_write>topologyArena:array<atomic<u32>>;
@group(0)@binding(1)var<storage,read_write>output:array<atomic<u32>>;
const TPM1_SURFACE_LOW:u32=0u;const TPM1_SURFACE_HIGH:u32=1u;
const TPM1_DENSITY_LOW:u32=2u;const TPM1_DENSITY_HIGH:u32=3u;
struct FixtureParams{stateOffsets2:vec4u};var<private>p:FixtureParams;
fn tpm1PacketReady(packet:u32)->bool{_=packet;return true;}
fn destinationDensity()->u32{return 0u;}fn destinationGamma()->u32{return 1u;}
fn scatterGammaRow(row:u32,density:u32,gamma:u32){
  atomicStore(&output[density+gamma],row);}
fn cm12ResidentDynamicClosureApplyVEX(packet:u32,mask:vec2u,worker:u32){_=packet;_=mask;_=worker;}
fn cm12ResidentDynamicClosureSeedVEXFrontier(packet:u32,mask:vec2u,worker:u32){_=packet;_=mask;_=worker;}
fn fixtureAcceptedSlot()->u32{return 0u;}
fn fixtureAcceptedGeneration()->u32{return 1u;}
${createSparseCM12DynamicClosureAuthorityWGSL({ layout: dca })}
${createSparseCM12InternedBoundaryImageWGSL({ layout: ibo, baseWords: iboBase,
  packetsPerLeaf: 64, hookPrefix: "cm12Resident",
  acceptedSlotHook: "fixtureAcceptedSlot",
  acceptedGenerationHook: "fixtureAcceptedGeneration" })}
${createSparseCM12IboTRAResidentHooksWGSL()}
${createSparseCM12IboTRASupplementWGSL({ layout: itr })}
`;
const dawn = await import(dawnModule) as { create:(flags:string[])=>GPU;
  globals:Record<string,unknown> };
Object.assign(globalThis,dawn.globals);const gpu=dawn.create(["backend=metal"]);
const adapter=await gpu.requestAdapter();if(!adapter)throw new Error("no adapter");
const device=await adapter.requestDevice();
const module=device.createShaderModule({code:source});const info=await module.getCompilationInfo();
const errors=info.messages.filter((message)=>message.type==="error");
if(errors.length){for(const error of errors)console.error(`${error.lineNum}:${error.linePos} ${error.message}`);
  throw new Error(`${errors.length} ITR1 WGSL errors`);}
console.log(JSON.stringify({schema:"sparse-cm12-ibo-tra-wgsl/v1",passed:true}));device.destroy();
