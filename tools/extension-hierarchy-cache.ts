/** Diagnostic coherent source staging for exact 2:1 hierarchy prolongation. */
import assert from "node:assert/strict";
import {fn} from "./probe-extension-direct-snapshot";
export function hierarchyCacheSource(source:string,onlyNeeded=false,decodedBoxes=false):string{
 assert.equal(source.split("if(CACHE_FINE_HIERARCHY)").length-1,2,"Hierarchy cache hooks changed");
 const inputLoads=[...fn(source,"hierarchyInput").matchAll(/return\s+([^;]+);/g)];
 assert.equal(inputLoads.length,2,"Hierarchy input hooks changed");
 const inputLoad=inputLoads[1]![1]!;
 const existingLoad=fn(source,"mixedProlongUnknownVelocity").match(/existing=([^;]+);/)?.[1];
 assert.ok(existingLoad,"Hierarchy existing load changed");
 let code=source.replaceAll("if(CACHE_FINE_HIERARCHY)","if(mixedCacheActive)");
 code=code.replace(fn(code,"hierarchyCacheIndex"),fn(code,"hierarchyCacheIndex").replaceAll("hierarchyCacheBase","mixedCacheBase"));
 for(const name of (decodedBoxes?["mixedProlongUnknownVelocity"]:["mixedProlongUnknownVelocity","mixedReplayUnknownVelocity"])){
  const body=fn(code,name);let next=body.replace("{","{\n mixedStageHierarchy(wid,lane);");
  if(name==="mixedReplayUnknownVelocity"){
   const tap=/terms\[k\]=vec2f\(([^;]*?)\[component\],1\.0\);/g;
   assert.equal([...next.matchAll(tap)].length,1,"Hierarchy replay tap changed");
   next=next.replace(tap,"terms[k]=vec2f(hierarchyInput(q)[component],1.0);");
  }
  code=code.replace(body,next);
 }
 if(decodedBoxes){
  const nearest=fn(code,"nearestHierarchySample");let cached=nearest;
  for(const [bound,upper] of [["lower",false],["upper",true]] as const){
   const expression=`faceLocation(originalFace(${bound}),baseDims(),component)`;
   assert.equal(cached.split(expression).length-1,1,"Hierarchy source box changed");
   cached=cached.replace(expression,`mixedSourceBox(q,component,${bound},${upper})`);
  }
  code=code.replace(nearest,cached);
 }
 return code+/* wgsl */`
var<private> mixedCacheActive:bool;
var<private> mixedCacheBase:vec3i;
${onlyNeeded?"var<workgroup> mixedCacheNeeded:atomic<u32>;":""}
${decodedBoxes?`var<workgroup> mixedLowerBox:array<vec3f,192>;
var<workgroup> mixedUpperBox:array<vec3f,192>;
fn mixedSourceBox(q:vec3i,component:u32,origin:u32,upper:bool)->vec3f{
 if(mixedCacheActive){let i=hierarchyCacheIndex(q)+64u*component;return select(mixedLowerBox[i],mixedUpperBox[i],upper);}
 return faceLocation(originalFace(origin),baseDims(),component);
}`:""}
fn mixedUncachedHierarchy(q:vec3i)->vec4f{return ${inputLoad};}
fn mixedStageHierarchy(wid:vec3u,lane:u32){
 let sd=hierarchySourceDims();let td=hierarchyTargetDims();mixedCacheActive=all(td==2*sd);mixedCacheBase=vec3i(2u*wid)-vec3i(1);
 ${onlyNeeded?`if(lane==0u){atomicStore(&mixedCacheNeeded,0u);}workgroupBarrier();
 if(mixedCacheActive&&lane<64u){let p=mixedHierarchyCell(wid,lane);if(inBounds(p,td)&&(u32(round((${existingLoad}).w))&7u)!=7u){atomicOr(&mixedCacheNeeded,1u);}}
 workgroupBarrier();mixedCacheActive=mixedCacheActive&&atomicLoad(&mixedCacheNeeded)!=0u;`:""}
 if(mixedCacheActive&&lane<${decodedBoxes?192:64}u){
  let cell=lane%64u;let q=clamp(mixedCacheBase+vec3i(vec3u(cell%4u,(cell/4u)%4u,cell/16u)),vec3i(0),sd-vec3i(1));
  let lower=textureLoad(sourceOrigins,q,0);let upper=textureLoad(sourceOrigins,q+vec3i(0,0,sd.z),0);
  if(lane<64u){hierarchyValues[cell]=mixedUncachedHierarchy(q);hierarchyLower[cell]=lower;hierarchyUpper[cell]=upper;}
  ${decodedBoxes?`let component=lane/64u;
  mixedLowerBox[lane]=faceLocation(originalFace(lower[component]),baseDims(),component);
  mixedUpperBox[lane]=faceLocation(originalFace(upper[component]),baseDims(),component);`:""}
 }
 workgroupBarrier();
}
`;
}
