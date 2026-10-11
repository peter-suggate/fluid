/** Diagnostic: retain old publication's geometric tap selections in the dead
 * extension epoch, then sum only current values during current publication. */
import assert from "node:assert/strict";
import {fn} from "./probe-extension-direct-snapshot";
const once=(source:string,from:string,to:string)=>{assert.equal(source.split(from).length-1,1,`Expected one ${from}`);return source.replace(from,to);};
const retainedMaskWGSL=/* wgsl */`
var<private> publicationSelection:u32;
fn publicationRemember(face:UMFace,mask:u32){
 // Canonical positive slots are unique. The opposite epoch is dead after
 // the final sweep; real extension distances are never negative.
 stateOut[umSlot(face.anchor,face.axis,face.width)]=vec2f(f32(mask),-1.0);
}
`;
export function rememberPublicationSource(source:string):string{
 let code=source;
 for(const name of ["umFarValue","umFarValueStaged","umFarValueUniform"]){
  const original=fn(code,name);let body=original.replace("{","{publicationSelection=0u;");
  const sum=name==="umFarValueUniform"?"sum+=value;count+=1.0;":"sum+=values[k];count+=1.0;";
  body=once(body,sum,`publicationSelection|=1u<<k;${sum}`);code=code.replace(original,body);
 }
 const originalFine=fn(code,"ueSnapshotFineOwner");
 const fine=once(originalFine,"else{value[axis]=umFarValueStaged(face);}","else{value[axis]=umFarValueStaged(face);}\n  publicationRemember(face,publicationSelection);");
 code=code.replace(originalFine,fine);
 const originalCoarse=fn(code,"ueSnapshotPublished");let coarse=originalCoarse;
 const valid="(u32(round(state.w))&(1u<<face.axis))!=0u&&lower!=0u";
 assert.ok(coarse.includes(valid));
 coarse=once(coarse,"return select(0.0,0.0+state[face.axis],"+valid+");",`publicationRemember(face,select(0u,1u,${valid}));return select(0.0,0.0+state[face.axis],${valid});`);
 coarse=once(coarse,"return umFarValue(face);","let value=umFarValue(face);publicationRemember(face,publicationSelection);return value;");
 return code.replace(originalCoarse,coarse)+retainedMaskWGSL;
}

export function replayPublicationSource(source:string):string{
 let code=source;
 const originalPublished=fn(code,"umPublished");code=code.replace(originalPublished,once(originalPublished,"return umFarValue(face);","return publicationReplay(face,false);"));
 const originalFine=fn(code,"uePublishFine");
 const start=originalFine.indexOf(" for(var item=lane;item<81u;item+=64u){");
 assert.ok(start>0,"Fine far staging changed");
 const store=[...originalFine.matchAll(/(?:udStore_output\s*\(|textureStore\s*\(\s*output\s*,)[^;]*;/g)];assert.equal(store.length,2,"Fine publisher exits changed");
 const finalStart=originalFine.lastIndexOf(" value.w=");assert.ok(finalStart>start);
 const fine=originalFine.slice(0,start)+/* wgsl */`
 // Values are the only per-cell data needed after old publication saved the
 // nearest-source selection. Preserve the same clamped 27-cell neighborhood.
 if(lane<27u){let p=vec3i(umTileCoord(farTile))+vec3i(umCorner(lane,3u))-vec3i(1);if(all(p>=vec3i(0))&&all(p<vec3i(${source.includes("fn UM_T(")?"UM_T()":"UM_T"}))){farState[lane]=publicationLoad(p);}}
 workgroupBarrier();
 for(var axis=0u;axis<3u;axis++){
  if((needsFar&(1u<<axis))==0u){continue;}
  value[axis]=publicationReplay(umUnitExtensionFace(owner,axis,1),true);
 }
`+originalFine.slice(finalStart);
 code=code.replace(originalFine,fine);
 // Preserve the resolved accessor of the original shader; hierarchy textures
 // are ordinary dense textures under either native or sparse field layouts.
 const load=fn(source,"umFarValue").match(/let state=([^;]+);/)?.[1];assert.ok(load,"Far state load changed");
 return code+/* wgsl */`
override publicationForceFallback:bool=false;
override publicationStrict:bool=false;
fn publicationLoad(p:vec3i)->vec4f{return ${load};}
fn publicationReplay(face:UMFace,staged:bool)->f32{
 let retained=stateOut[umSlot(face.anchor,face.axis,face.width)];
 // The explicit tag also keeps a safe original path for an uncached face.
 if(publicationForceFallback){return umFarValue(face);}
 if(retained.y!=-1.0){if(publicationStrict){return -1e30;}return umFarValue(face);}
 let selected=u32(retained.x);let point=umFaceCenter(face);var q=point/4.0-vec3f(0.5);q[face.axis]-=0.5;
 let base=vec3i(floor(q));var sum=0.0;var count=0.0;
 for(var k=0u;k<8u;k++){
  if((selected&(1u<<k))==0u){continue;}
  let p=clamp(base+vec3i(umCorner(k,2u)),vec3i(0),ueFarTop(face.axis));
  var value:f32;if(staged){value=farState[umFarSlot(p)][face.axis];}else{value=publicationLoad(p)[face.axis];}
  sum+=value;count+=1.0;
 }
 return select(0.0,sum/count,count>0.0);
}
`;
}
