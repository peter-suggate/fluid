/** Diagnostic queue production in the existing P2G preparation/gather.
 * Inactive pairs and active pairs partition the fine cells. */
import assert from "node:assert/strict";

function symbols(source:string){
 return {cells:source.includes("fn NB_CELLS(")?"NB_CELLS()":"NB_CELLS",
  dims:source.includes("fn UM_D(")?"UM_D()":"UM_D",
  tiles:source.includes("fn UM_TILES(")?"UM_TILES()":"UM_TILES",
  counts:source.includes("struct UMSceneCounts")?"umCounts.work":"umCounts"};
}
function producerHelpers(source:string):string{
 const {cells,tiles,counts}=symbols(source);
 const depth=source.includes("fn NB_DEPTH_B(")?"NB_DEPTH_B()":"NB_DEPTH_B";
 return /* wgsl */`
fn pmProducerEligible()->bool{return 416u*${counts}.x<=${cells}&&64u*${counts}.x+1u<=${cells}&&umSupport[4u*${tiles}+1u]==0u;}
// The nearest-cell bank expires after surface reconstruction/redistance.
// A later buildDistance always rebuilds nearest cells first.
fn pmProducerQueueBase()->u32{return ${depth};}
fn pmAppendCells(tags:vec2u,emit:vec2u,subgroupLane:u32){
 let amount=emit.x+emit.y;let rank=subgroupExclusiveAdd(amount);let count=subgroupAdd(amount);var first=0u;
 if(subgroupLane==0u&&count!=0u){first=atomicAdd(&bins[pmProducerQueueBase()+${cells}-1u],count);}
 first=subgroupBroadcastFirst(first);
 if(emit.x!=0u){atomicStore(&bins[pmProducerQueueBase()+first+rank],tags.x);}
 if(emit.y!=0u){atomicStore(&bins[pmProducerQueueBase()+first+rank+emit.x],tags.y);}
}
`;
}
function bodyRange(source:string,name:string):[number,number]{
 const begin=source.indexOf(`fn ${name}(`);assert.ok(begin>=0,name);
 const open=source.indexOf("{",begin);let end=open+1,depth=1;
 while(depth){const char=source[end++];assert.ok(char!==undefined);if(char==="{")depth++;else if(char==="}")depth--;}
 return [begin,end];
}
export function fusedQueueProperties(source:string):string{
 const [start,end]=bodyRange(source,"transferPropertiesJob");let body=source.slice(start,end);
 const anchor=" let emit=neededPair&&lane%2u==0u;";assert.ok(body.includes(anchor));
 body=body.replace(anchor,/* wgsl */`
 pmAppendCells(vec2u(64u*owner.tile+lane,0u),vec2u(u32(!neededPair&&pmProducerEligible()),0u),subgroupLane);
`+anchor);
 return source.slice(0,start)+body+source.slice(end)+producerHelpers(source);
}
export function fusedQueueGather(source:string):string{
 const [start,end]=bodyRange(source,"transferJob");let body=source.slice(start,end);
 assert.ok(body.includes("subgroupLane"),"Diagnostic requires subgroup gather");
 const anchor=" if(valid&&member==0u&&owner.width==1u){";assert.ok(body.includes(anchor));
 body=body.replace(anchor," var pmEmit=vec2u(0);\n"+anchor);
 const {cells}=symbols(source);
 for(const [index,owner,origin] of [[0,"owner","origin"],[1,"next","nextOrigin"]] as const){
  const at=`let at=${cells}+6u*${owner}.index;`;assert.ok(body.includes(at));
  body=body.replace(at,`pmEmit[${index}u]=u32(!all(influence==vec3f(1))||any(${origin}==vec3u(0)));`+at);
 }
 body=body.slice(0,-1)+/* wgsl */`
 pmAppendCells(vec2u(64u*owner.tile+owner.lane,64u*next.tile+next.lane),pmEmit*u32(pmProducerEligible()),subgroupLane);
}`;
 return source.slice(0,start)+body+source.slice(end)+producerHelpers(source);
}
export function fusedQueuePublication(source:string):string{
 const [start,end]=bodyRange(source,"transferInactiveJob");let body=source.slice(start,end);
 assert.ok(body.includes("let original=udLoad_velocity(origin);"));
 body=body.replace("let original=udLoad_velocity(origin);","");
 body=body.replace("if(first<0.0){udStore_output(origin,original);return;}","if(first<0.0){udStore_output(origin,udLoad_velocity(origin));return;}");
 const store=" udStore_output(origin,vec4f(mix(original.xyz,transferred,influence),original.w));";assert.ok(body.includes(store));
 body=body.replace(store,/* wgsl */`
 if(pmProducerEligible()&&all(influence==vec3f(1))&&all(origin>vec3i(0))){udStore_output(origin,vec4f(transferred,0));return;}
 let original=udLoad_velocity(origin);
`+store);
 return source.slice(0,start)+body+source.slice(end)+producerHelpers(source);
}
export function fusedQueueMomentum(source:string):string{
 const {dims,tiles,counts}=symbols(source);const cells=`(${dims}.x*${dims}.y*${dims}.z)`;
 let code=source+/* wgsl */`
fn pmProducerEligible()->bool{return 416u*${counts}.x<=${cells}&&64u*${counts}.x+1u<=${cells}&&umSupport[4u*${tiles}+1u]==0u;}
fn pmProducerQueueBase()->u32{return 2u*${cells}+2u*${tiles}+(${dims}.x+1u)*(${dims}.y+1u)*(${dims}.z+1u);}
fn pmProducerCount()->u32{return bitcast<u32>(umParticleTransfer[pmProducerQueueBase()+${cells}-1u]);}
fn pmProducerCell(index:u32)->u32{return bitcast<u32>(umParticleTransfer[pmProducerQueueBase()+index]);}
var<workgroup> pmProducerGate:u32;
`;
 code=code.replace(" if(lane==0u){pmClassifyCount", " if(lane==0u){pmProducerGate=u32(pmProducerEligible());}\n if(workgroupUniformLoad(&pmProducerGate)!=0u){return;}\n if(lane==0u){pmClassifyCount");
 const [start,end]=bodyRange(code,"pmPackedCells");let body=code.slice(start,end);
 body=body.replace("pmCellCount=atomicLoad(&umDeferred[pmInputBase()]);", "pmCellCount=atomicLoad(&umDeferred[pmInputBase()]);if(pmProducerEligible()){pmCellCount=pmProducerCount();}");
 body=body.replace("tag=atomicLoad(&umDeferred[pmInputBase()+4u+first+slot]);", "if(pmProducerEligible()){tag=pmProducerCell(first+slot);}else{tag=atomicLoad(&umDeferred[pmInputBase()+4u+first+slot]);}");
 const sample="umUnitEscaped=false;value=umUnitMomentum(owner,umFace(owner,axis,1,0u));";assert.ok(body.includes(sample));
 body=body.replace(sample,/* wgsl */`umUnitEscaped=false;
   if(pmProducerEligible()){
    let origin=umOrigin(owner);
    if(origin[axis]==0u){boundary[umNegativeBoundaryIndex(origin,axis)]=umUnitMomentum(owner,umFace(owner,axis,-1,0u));}
   }
   value=umUnitMomentum(owner,umFace(owner,axis,1,0u));`);
 return code.slice(0,start)+body+code.slice(end);
}

/** Preserve the classifier's largest cheap case inside the existing 125-tap
 * properties producer. Its output is rebound to departure for this pipeline;
 * the stored velocity remains immutable until the later publication. */
export function fusedQueueCullProperties(source:string,cullAir:boolean):string{
 const {dims}=symbols(source);
 const [start,end]=bodyRange(source,"transferPropertiesJob");let body=source.slice(start,end);
 const stage="for(var k=lane;k<125u;k+=64u){let q=origin+umCorner(k,5u);nbTransferDepth[k]=";
 assert.ok(body.includes(stage),"Expected the shared 5-cubed property stencil");
 body=body.replace(stage,"for(var k=lane;k<125u;k+=64u){let q=origin+umCorner(k,5u);pmPrediction[k]=pmPredictionLive(umOwnerAt(vec3i(q)));nbTransferDepth[k]=");
 const append=" pmAppendCells(vec2u(64u*owner.tile+lane,0u),vec2u(u32(!neededPair&&pmProducerEligible()),0u),subgroupLane);";
 assert.ok(body.includes(append),"Expected the fused inactive-cell append");
 body=body.replace(append,/* wgsl */`
 let liveAt=local.x+5u*(local.y+5u*local.z);
 // Negative boundary outputs stay with the original momentum evaluator.
 let omitAir=${cullAir}&&pmProducerEligible()&&all(cell>vec3u(0))&&
  (pmPrediction[liveAt]|pmPrediction[liveAt+1u]|pmPrediction[liveAt+5u]|pmPrediction[liveAt+25u])==0u;
 if(!neededPair&&omitAir){udStore_output(vec3i(cell),vec4f(0));}
 pmAppendCells(vec2u(64u*owner.tile+lane,0u),vec2u(u32(!neededPair&&!omitAir&&pmProducerEligible()),0u),subgroupLane);
`);
 return source.slice(0,start)+body+source.slice(end)+/* wgsl */`
@group(1) @binding(17) var pmVolume:texture_3d<f32>;
@group(1) @binding(18) var pmCenterPhi:texture_3d<f32>;
@group(1) @binding(81) var pmVolumeBase:texture_3d<f32>;
@group(1) @binding(82) var pmCenterPhiBase:texture_3d<f32>;
var<workgroup> pmPrediction:array<u32,125>;
fn pmPredictionLive(owner:UMOwner)->u32{
 if(owner.width==0u){return 0u;}
 let at=vec3i(umOrigin(owner));
 if(owner.width==4u){
  let vb=u32(textureDimensions(pmVolumeBase).x<${dims}.x);
  let pb=u32(textureDimensions(pmCenterPhiBase).x<${dims}.x);
  return u32(textureLoad(pmVolumeBase,at>>vec3u(2u*vb),0).x>0.0||textureLoad(pmCenterPhiBase,at>>vec3u(2u*pb),0).x<0.0);
 }
 let vb=u32(textureDimensions(pmVolume).x<${dims}.x);
 let pb=u32(textureDimensions(pmCenterPhi).x<${dims}.x);
 return u32(textureLoad(pmVolume,at>>vec3u(2u*vb),0).x>0.0||textureLoad(pmCenterPhi,at>>vec3u(2u*pb),0).x<0.0);
}
`;
}
