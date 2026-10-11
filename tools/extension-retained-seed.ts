/** Diagnostic raw-canonical retention: no phase samples and no full texture copy. */
import assert from "node:assert/strict";
import { fn } from "./probe-extension-direct-snapshot";

function replaceExact(source:string,from:string,to:string,count=1):string{
 assert.equal(source.split(from).length-1,count,from);return source.replaceAll(from,to);
}
export function retainedCurrentSeed(source:string):string{
 source=replaceExact(source,"pairPhysical(face)","0.0",2);
 return replaceExact(source,"stateIn[slot]=current;stateOut[slot]=current;pairIn[slot]=value.x.y;pairOut[slot]=value.x.y;","stateIn[slot]=current;stateOut[slot]=current;");
}
/** Recover the independently compiled old reader from integrated retention. */
export function originalCurrentSeed(source:string):string{
 if(!source.includes("fn retainFine("))return source;
 source=replaceExact(source,fn(source,"retainedStore"),"");
 for(const name of ["retainFine","retainCoarse"])source=replaceExact(source,"@compute @workgroup_size(64) "+fn(source,name),"");
 for(const name of ["pair_umSeedState","pair_umSeedWall"]){const body=fn(source,name);source=replaceExact(source,body,replaceExact(body,"vec2f(umPhysical(face),0.0)","vec2f(umPhysical(face),pairPhysical(face))"));}
 return replaceExact(source,"stateIn[slot]=current;stateOut[slot]=current;","stateIn[slot]=current;stateOut[slot]=current;pairIn[slot]=value.x.y;pairOut[slot]=value.x.y;");
}
export function originalSnapshot(source:string):string{
 const body=fn(source,"umPhysical");
 if(!body.includes("return pairIn["))return source;
 return replaceExact(source,body,replaceExact(body,"return pairIn[umSlot(face.anchor,face.axis,face.width)];","return udLoad_physical(face.anchor)[face.axis];"));
}
export function retainedSnapshot(source:string):string{
 return replaceExact(source,fn(source,"umPhysical"),`fn umPhysical(face:UMFace)->f32{
 if(face.anchor[face.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(face.anchor,vec3i(0))),face.axis)];}
 return pairIn[umSlot(face.anchor,face.axis,face.width)];
}`);
}
export function retainCanonicalSource(source:string,compare=false,compareState=false):string{
 const tiles=source.includes("fn UM_TILES(")?"UM_TILES()":"UM_TILES";
 const counts=source.includes("umCounts.work")?"umCounts.work":"umCounts";
 assert.ok(source.includes("fn udLoad_physical("));
 return source+/* wgsl */`
${compare?`@group(2) @binding(0) var<storage,read> retainedExpected:array<u32>;
@group(2) @binding(1) var<storage,read_write> retainedStats:array<atomic<u32>>;`:""}
${compareState?"@group(2) @binding(2) var<storage,read> retainedExpectedState:array<vec2u>;":""}
fn retainedStore(slot:u32,value:f32){
 ${compare?`atomicAdd(&retainedStats[2],1u);if(retainedExpected[slot]!=bitcast<u32>(pairIn[slot])){atomicAdd(&retainedStats[0],1u);atomicMin(&retainedStats[1],slot);}`:`pairIn[slot]=value;pairOut[slot]=value;`}
 ${compareState?`atomicAdd(&retainedStats[2],2u);let current=bitcast<vec2u>(stateIn[slot]);for(var c=0u;c<2u;c++){if(current[c]!=retainedExpectedState[slot][c]){atomicAdd(&retainedStats[0],1u);atomicMin(&retainedStats[1],slot);}}`:""}
}

@compute @workgroup_size(64) fn retainFine(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var job=group.x;job<${counts}.x;job+=groups.x){
  let origin=umTileCoord(umTopology[${tiles}+job])*4u+umCorner(lane,4u);let owner=umOwnerAt(vec3i(origin));
  let value=${compare?"vec3f(0)":"udLoad_physical(vec3i(origin)).xyz"};
  for(var axis=0u;axis<3u;axis++){
   retainedStore(ueOwnSlot(owner,axis),value[axis]);
   ${compare?"":`if(origin[axis]==0u){let face=umUnitExtensionFace(owner,axis,-1);retainedStore(umSlot(face.anchor,axis,1u),umPhysical(face));}`}
  }
 }
}
@compute @workgroup_size(64) fn retainCoarse(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var job=group.x;job<${counts}.y;job+=groups.x){
  let tile=umTopology[${tiles}+${counts}.x+job];let origin=umTileCoord(tile)*4u;let owner=umOwnerAt(vec3i(origin));
  if(lane<48u){let axis=lane/16u;let face=umFace(owner,axis,1,lane%16u);if(face.width!=0u){retainedStore(umSlot(face.anchor,axis,face.width),${compare?"0.0":"umPhysical(face)"});}}
  ${compare?"":`else if(lane<51u){let axis=lane-48u;if(origin[axis]==0u){let face=umFace(owner,axis,-1,0u);retainedStore(umSlot(face.anchor,axis,face.width),umPhysical(face));}}`}
 }
}
`;
}

/** Every coarse tile is in exactly one stencil-width list. Unlike extension
 * sweep helpers, raw retention must include unsupported tiles and walls. */
export function packedCanonicalSource(source:string):string{
 const tiles=source.includes("fn UM_TILES(")?"UM_TILES()":"UM_TILES";
 const old=fn(source,"retainCoarse");
 return replaceExact(source,old,/* wgsl */`fn retainCoarse(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let regular=umSupport[8u*${tiles}+20u];let packs=(regular+63u)/64u;
 let header=7u*${tiles}+16u;let fine=umSupport[header];let seams=umSupport[header+1u];
 for(var job=group.x;job<packs+seams;job+=groups.x){
  if(job<packs){
   let index=64u*job+lane;if(index>=regular){continue;}
   let tile=umSupport[8u*${tiles}+24u+index];let origin=umTileCoord(tile)*4u;
   for(var axis=0u;axis<3u;axis++){
    var anchor=vec3i(origin);anchor[axis]+=3;
    let face=UMFace(UMOwner(),anchor,4u,1u,axis,1);
    retainedStore(umSlot(anchor,axis,4u),umPhysical(face));
    if(origin[axis]==0u){anchor=vec3i(origin);anchor[axis]=-1;let wall=UMFace(UMOwner(),anchor,4u,1u,axis,-1);retainedStore(umSlot(anchor,axis,4u),umPhysical(wall));}
   }
  }else{
   let tile=umSupport[header+4u+fine+job-packs];let origin=umTileCoord(tile)*4u;let owner=umOwnerAt(vec3i(origin));
   if(lane<48u){let axis=lane/16u;let face=umFace(owner,axis,1,lane%16u);if(face.width!=0u){retainedStore(umSlot(face.anchor,axis,face.width),umPhysical(face));}}
   else if(lane<51u){let axis=lane-48u;if(origin[axis]==0u){let face=umFace(owner,axis,-1,0u);retainedStore(umSlot(face.anchor,axis,face.width),umPhysical(face));}}
  }
 }
}`);
}
