/** Diagnostic-only packed-cell momentum. Classify cheap outcomes once, then
 * place surviving whole cells next to one another in the tracing launch. */
import assert from "node:assert/strict";
import {fusedQueueMomentum} from "./fused-momentum-queue";

export function compactMomentumSource(source:string,particleWeights=false):string{
 const tiles=source.includes("fn UM_TILES(")?"UM_TILES()":"UM_TILES";
 const dims=source.includes("fn UM_D(")?"UM_D()":"UM_D";
 const counts=source.includes("struct UMSceneCounts")?"umCounts.work":"umCounts";
 assert.ok(source.includes("fn umUnitMomentum("));
 assert.ok(source.includes("fn umDeferredListedCell("));
 // The classifier omits complete replacements. Retained cells keep all
 // three original component evaluations and their shared escape decision.
 const original=source.replaceAll(" if(umParticleReplacesMomentum(owner,face)){return 0.0;}\n","");
 return original+/* wgsl */`
fn pmInputBase()->u32{return 4u+3u*${tiles};}
var<workgroup> pmKeep:array<atomic<u32>,2>;
var<workgroup> pmClassifyCount:u32;
@compute @workgroup_size(64) fn pmClassify(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){pmClassifyCount=umSupport[4u*${tiles}+2u];}
 let count=workgroupUniformLoad(&pmClassifyCount);
 for(var job=group.x;job<count;job+=groups.x){
  if(lane<2u){atomicStore(&pmKeep[lane],0u);}workgroupBarrier();
  let owner=umOwner(vec3u(64u*job+lane,0u,0u));let origin=umOrigin(owner);
  let ownLive=umPredictionCellLive(owner);
  ${particleWeights?`let cells=${dims}.x*${dims}.y*${dims}.z;let at=cells+6u*owner.index;
  let replaced=416u*${counts}.x<=cells&&umParticleTransfer[at+1u]==1.0&&umParticleTransfer[at+3u]==1.0&&umParticleTransfer[at+5u]==1.0;`:"let replaced=false;"}
  var values=vec3f(0);var needed=false;
  for(var axis=0u;axis<3u;axis++){
   if(origin[axis]==0u){
    var negativeValue=0.0;
    if(!umCullAir||ownLive){negativeValue=umOriginalMomentum(umFace(owner,axis,-1,0u));}
    boundary[umNegativeBoundaryIndex(origin,axis)]=negativeValue;
   }
   let face=umFace(owner,axis,1,0u);
   if(!umCullAir||ownLive||umPredictionCellLive(face.neighbor)){
    if(umClosedPositive(face)){values[axis]=umOriginalMomentum(face);}
    else if(!umDropFace(face)&&!replaced){needed=true;}
   }
  }
  if(needed){atomicOr(&pmKeep[lane/32u],1u<<(lane%32u));}
  else{udStore_output(vec3i(origin),vec4f(values,0));}
  workgroupBarrier();
  if(lane==0u){
   let mask=vec2u(atomicLoad(&pmKeep[0]),atomicLoad(&pmKeep[1]));
   if(any(mask!=vec2u(0))){
    let entry=atomicAdd(&umDeferred[pmInputBase()],1u);let at=pmInputBase()+4u+3u*entry;
    atomicStore(&umDeferred[at],owner.tile);atomicStore(&umDeferred[at+1u],mask.x);atomicStore(&umDeferred[at+2u],mask.y);
    atomicAdd(&umDeferred[pmInputBase()+1u],countOneBits(mask.x)+countOneBits(mask.y));
   }
  }
  workgroupBarrier();
 }
}
var<workgroup> pmCount:u32;
var<workgroup> pmFill:u32;
var<workgroup> pmHeld:u32;
var<workgroup> pmEntries:array<vec4u,8>;
var<workgroup> pmEscaped:array<atomic<u32>,16>;
var<workgroup> pmValues:array<f32,192>;
@compute @workgroup_size(192) fn pmPacked(@builtin(local_invocation_index) lane:u32){
 let slot=lane%64u;let axis=lane/64u;
 if(lane==0u){pmCount=atomicLoad(&umDeferred[pmInputBase()]);}
 let count=workgroupUniformLoad(&pmCount);var pending=vec4u(0);
 loop{
  if(lane==0u){
   var fill=0u;var held=0u;
   loop{
    if(held>=8u){break;}
    if(pending.w==0u){
     let entry=atomicAdd(&umClaims[1],1u);if(entry>=count){break;}
     let at=pmInputBase()+4u+3u*entry;
     pending=vec4u(atomicLoad(&umDeferred[at]),atomicLoad(&umDeferred[at+1u]),atomicLoad(&umDeferred[at+2u]),0u);
     pending.w=countOneBits(pending.y)+countOneBits(pending.z);continue;
    }
    if(fill+pending.w>64u){break;}
    fill+=pending.w;pmEntries[held]=vec4u(pending.xyz,fill);held++;pending.w=0u;
   }
   pmFill=fill;pmHeld=held;
  }
  let fill=workgroupUniformLoad(&pmFill);if(fill==0u){break;}
  if(lane<16u){atomicStore(&pmEscaped[lane],0u);}workgroupBarrier();
  let listed=slot<fill;var owner=UMOwner();var value=0.0;var held=0u;var cell=0u;
  if(listed){
   var before=0u;
   for(var i=0u;i<7u;i++){let end=pmEntries[i].w;if(slot<end){break;}before=end;held=i+1u;}
   let entry=pmEntries[held];cell=umDeferredListedCell(entry.yz,slot-before);
   owner=UMOwner(entry.x,cell,1u,(umTopology[entry.x]&0x3fffffffu)+cell);
   umUnitEscaped=false;value=umUnitMomentum(owner,umFace(owner,axis,1,0u));
   if(umUnitEscaped){atomicOr(&pmEscaped[2u*held+cell/32u],1u<<(cell%32u));}
  }
  pmValues[lane]=value;workgroupBarrier();
  if(lane<64u&&listed&&((atomicLoad(&pmEscaped[2u*held+cell/32u])>>(cell%32u))&1u)==0u){
   udStore_output(vec3i(umOrigin(owner)),vec4f(pmValues[slot],pmValues[slot+64u],pmValues[slot+128u],0));
  }
  // Each input tile is consumed whole by one round, so its escaped mask
  // needs at most one original deferred entry. No cell is split or lost.
  if(lane<pmHeld){
   let mask=vec2u(atomicLoad(&pmEscaped[2u*lane]),atomicLoad(&pmEscaped[2u*lane+1u]));
   if(any(mask!=vec2u(0))){
    let at=4u+3u*atomicAdd(&umDeferred[0],1u);
    atomicStore(&umDeferred[at],pmEntries[lane].x);atomicStore(&umDeferred[at+1u],mask.x);atomicStore(&umDeferred[at+2u],mask.y);
   }
  }
  workgroupBarrier();
 }
}
`;
}

export interface CompactMomentumCapture {
 sources:WeakMap<GPUShaderModule,string>;
 descriptors:WeakMap<GPUComputePipeline,GPUComputePipelineDescriptor>;
 groups:WeakMap<GPUBindGroup,GPUBindGroupDescriptor>;
}

/** A second queue layout removes the tile-mask search altogether. Subgroup
 * ranks reserve contiguous cell IDs; each tracing group claims 64 of them. */
export function compactMomentumCellSource(source:string,particleWeights=false):string{
 let code=compactMomentumSource(source,particleWeights);
 if(!code.includes("enable subgroups;"))code="enable subgroups;\n"+code;
 code=code.replace(/fn pmInputBase\(\)->u32\{[^}]+\}/,"fn pmInputBase()->u32{return arrayLength(&umDeferred)/2u;}");
 code=code.replace("lane:u32){\n if(lane==0u){pmClassifyCount", "lane:u32,@builtin(subgroup_invocation_id) subgroupLane:u32){\n if(lane==0u){pmClassifyCount");
 code=code.replace("  if(lane<2u){atomicStore(&pmKeep[lane],0u);}workgroupBarrier();\n","");
 const begin=code.indexOf("  if(needed){atomicOr(&pmKeep");
 const end=code.indexOf("\n }\n}\nvar<workgroup> pmCount",begin);
 assert.ok(begin>=0&&end>begin);
 code=code.slice(0,begin)+/* wgsl */`
  if(!needed){udStore_output(vec3i(origin),vec4f(values,0));}
  let rank=subgroupExclusiveAdd(u32(needed));let count=subgroupAdd(u32(needed));var first=0u;
  if(subgroupLane==0u&&count!=0u){first=atomicAdd(&umDeferred[pmInputBase()],count);}
  first=subgroupBroadcastFirst(first);
  if(needed){atomicStore(&umDeferred[pmInputBase()+4u+first+rank],64u*owner.tile+owner.lane);}
  workgroupBarrier();
`+code.slice(end);
 return code+/* wgsl */`
var<workgroup> pmCellCount:u32;
var<workgroup> pmCellFirst:u32;
var<workgroup> pmCellEscape:array<atomic<u32>,64>;
var<workgroup> pmCellValues:array<f32,192>;
fn pmCellOwner(tag:u32)->UMOwner{
 let tile=tag/64u;let cell=tag%64u;
 return UMOwner(tile,cell,1u,(umTopology[tile]&0x3fffffffu)+cell);
}
@compute @workgroup_size(192) fn pmPackedCells(@builtin(local_invocation_index) lane:u32){
 let slot=lane%64u;let axis=lane/64u;
 if(lane==0u){pmCellCount=atomicLoad(&umDeferred[pmInputBase()]);}
 let count=workgroupUniformLoad(&pmCellCount);
 loop{
  if(lane==0u){pmCellFirst=atomicAdd(&umClaims[1],64u);}
  let first=workgroupUniformLoad(&pmCellFirst);if(first>=count){break;}
  if(lane<64u){atomicStore(&pmCellEscape[lane],0u);}workgroupBarrier();
  let listed=first+slot<count;var owner=UMOwner();var tag=0u;var value=0.0;
  if(listed){
   tag=atomicLoad(&umDeferred[pmInputBase()+4u+first+slot]);owner=pmCellOwner(tag);
   umUnitEscaped=false;value=umUnitMomentum(owner,umFace(owner,axis,1,0u));
   if(umUnitEscaped){atomicOr(&pmCellEscape[slot],1u);}
  }
  pmCellValues[lane]=value;workgroupBarrier();
  if(lane<64u&&listed){
   if(atomicLoad(&pmCellEscape[slot])!=0u){
    let out=4u+atomicAdd(&umDeferred[0],1u);atomicStore(&umDeferred[out],tag);
   }else{udStore_output(vec3i(umOrigin(owner)),vec4f(pmCellValues[slot],pmCellValues[slot+64u],pmCellValues[slot+128u],0));}
  }
  workgroupBarrier();
 }
}
@compute @workgroup_size(192) fn pmDeferredCells(@builtin(local_invocation_index) lane:u32){
 let slot=lane%64u;let axis=lane/64u;
 if(lane==0u){pmCellCount=atomicLoad(&umDeferred[0]);}
 let count=workgroupUniformLoad(&pmCellCount);
 loop{
  if(lane==0u){pmCellFirst=atomicAdd(&umClaims[3],64u);}
  let first=workgroupUniformLoad(&pmCellFirst);if(first>=count){break;}
  let listed=first+slot<count;var owner=UMOwner();var value=0.0;
  if(listed){owner=pmCellOwner(atomicLoad(&umDeferred[4u+first+slot]));value=umMomentum(owner,umFace(owner,axis,1,0u));}
  pmCellValues[lane]=value;workgroupBarrier();
  if(lane<64u&&listed){udStore_output(vec3i(umOrigin(owner)),vec4f(pmCellValues[slot],pmCellValues[slot+64u],pmCellValues[slot+128u],0));}
  workgroupBarrier();
 }
}
`;
}

/** Bind the expired source bank only within the momentum interval. The
 * original deferred binding is reused, so device storage limits are unchanged. */
export async function createCompactMomentumReplay(device:GPUDevice,frame:any,capture:CompactMomentumCapture,particleWeights=false,directCells=false,fusedProducer=false){
 const momentum=frame.momentum,ownership=frame.ownership;
 const descriptor=capture.descriptors.get(momentum.generalPipeline)!;assert.ok(descriptor);
 const source=capture.sources.get(descriptor.compute.module)!;assert.ok(source);
 let code=(directCells?compactMomentumCellSource:compactMomentumSource)(source,particleWeights);
 if(fusedProducer){assert.ok(directCells&&particleWeights);code=fusedQueueMomentum(code);}
 const module=device.createShaderModule({label:"Packed general fine momentum",code});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
 if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
 const entries=directCells?["pmClassify","pmPackedCells","pmDeferredCells"]:["pmClassify","pmPacked"];
 const pipelines=await Promise.all(entries.map(entryPoint=>device.createComputePipelineAsync({...descriptor,compute:{...descriptor.compute,module,entryPoint}})));
 let inputBase=4+3*ownership.capacity.tiles,bytes=8*inputBase;
 let group:GPUBindGroup,scratch:GPUBuffer;
 return {
  code,get inputBase(){return inputBase;},get bytes(){return bytes;},
  get scratch(){return scratch;},
  bind(){
   scratch=frame.narrowBandFlip.extensionScratch;
   if(directCells){inputBase=4+64*ownership.capacity.fineTiles;bytes=8*inputBase;}
   assert.ok(scratch.size>=bytes,`Compaction needs ${bytes} bytes, source bank has ${scratch.size}`);
   assert.ok(bytes<=device.limits.maxStorageBufferBindingSize,`Compaction exceeds the storage binding limit`);
   const sourceGroup=capture.groups.get(frame.momentumGroup.group)!;assert.ok(sourceGroup);
   group=device.createBindGroup({...sourceGroup,entries:[...sourceGroup.entries].map(entry=>entry.binding===14?{binding:14,resource:{buffer:scratch,size:bytes}}:entry)});
  },
  encode(encoder:GPUCommandEncoder,split=false){
   assert.ok(group);encoder.clearBuffer(momentum.claims);encoder.clearBuffer(scratch,0,16);encoder.clearBuffer(scratch,4*inputBase,16);
   const begin=(label:string)=>{const pass=encoder.beginComputePass({label});pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,group);if(momentum.hanging)pass.setBindGroup(2,ownership.hangingGroup);return pass;};
   let pass=begin(split?"Compact momentum classify":"Uniform compact mixed momentum");
   const next=(label:string)=>{if(split){pass.end();pass=begin(label);}};
   const grid=Math.min(2048,ownership.capacity.tiles);
   pass.setPipeline(pipelines[0]!);pass.dispatchWorkgroups(grid);
   next("Compact momentum regular");ownership.dispatchBuffered(pass,momentum.regularPipeline,"fine",grid);
   next("Compact momentum packed");pass.setPipeline(pipelines[1]!);pass.dispatchWorkgroups(grid);
   next("Compact momentum deferred");if(momentum.deferredPipeline)ownership.dispatchBuffered(pass,directCells?pipelines[2]!:momentum.deferredPipeline,"seamFine",grid,2);
   next("Compact momentum coarse");ownership.dispatchBuffered(pass,momentum.pipeline,"coarseMerged",grid);pass.end();
  },
 };
}
