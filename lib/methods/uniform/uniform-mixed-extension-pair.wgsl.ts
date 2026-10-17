/** Two velocity payloads share one immutable extension geometry. */
/** A source edit must update both scalar and paired forms deliberately. Never
 * allow a missing or newly ambiguous substitution to compile a stale variant. */
function replaceExact(source: string, from: string, to: string, expected = 1): string {
 const count = source.split(from).length - 1;
 if (count !== expected) throw new Error(`Paired extension expected ${expected} occurrences of ${JSON.stringify(from)}, found ${count}`);
 return source.replaceAll(from, to);
}
function requireExact(source: string, expected: string, name: string): void {
 if (source !== expected) throw new Error(`Paired extension ${name} changed; update its paired form`);
}
function functionSource(source: string, name: string): string {
 const marker = `fn ${name}(`;
 const count = source.split(marker).length - 1;
 if (count !== 1) throw new Error(`Paired extension function ${name}: expected one definition, found ${count}`);
 const at = source.indexOf(marker);
 let first = at, level = 1, end = source.indexOf("{", at) + 1;
 const line = source.lastIndexOf("\n", at) + 1;
 if (source.slice(line, at).includes("@compute")) first = line;
 for (; level && end < source.length; end++) {
  if (source[end] === "{") level++; else if (source[end] === "}") level--;
 }
 if(level!==0)throw new Error(`Paired extension function ${name} is incomplete`); return source.slice(first, end);
}

/** Values become two-component vectors; all distance decisions and ordered
 * reductions stay scalar and execute once. Original entries remain available
 * in the same module so the source transform is independently reviewable. */
export function pairedSweepSource(source: string): string {
 const names = ["umSlotState", "umNeighbor", "ueCompiledNeighbor", "ueCombine", "umExtended", "ueExtend", "ueNote", "ueSweepStore", "ueSweepPack", "sweepListJob", "sweepJob", "ueSweepSeamFine", "ueSweepCoarsePack", "sweepList", "sweep", "sweepSeams"];
 const renamed = (s: string) => names.reduce((s, n) => s.replaceAll(new RegExp(`\\b${n}\\b`, "g"), `pair_${n}`), s);
 const copies = names.map(name => {
  let s = functionSource(source, name);
  const replace = (from: string, to: string, expected = 1) => { s = replaceExact(s, from, to, expected); };
  if (name === "umSlotState") {
   replace("->vec2f", "->UEPair");
   replace("stateIn[umSlot(", "pairRead(umSlot(");
   replace("axis,width)];", "axis,width));");
   replace("return select(vec2f(0.0,UM_INF),state,(umTileSupport(tile)&2u)!=0u);", "if((umTileSupport(tile)&2u)==0u){return UEPair(vec2f(0),UM_INF);}return state;");
  }
  if (name === "ueSweepStore") {
   requireExact(s, "fn ueSweepStore(slot:u32,value:vec2f){if(value.y<0.5*UM_INF&&value.y!=0.0){stateOut[slot]=value;}}", name);
   return `fn pair_ueSweepStore(slot:u32,value:UEPair){
 if(value.y<0.5*UM_INF&&value.y!=0.0){stateOut[slot]=vec2f(value.x.x,value.y);pairOut[slot]=value.x.y;}
}`;
  }
  if (["umNeighbor", "ueCompiledNeighbor"].includes(name)) {
   replace("array<vec2f,", "array<UEPair,");
   replace("var sum=0.0", "var sum=vec2f(0)");
   replace("UMNeighbor(0,UM_INF,1)", "UMNeighbor(vec2f(0),UM_INF,1)", name === "umNeighbor" ? 2 : 1);
   replace("select(0.0,sum/count,count>0.0)", "select(vec2f(0),sum/count,count>0.0)");
  }
  if (name === "ueCombine") {
   replace("old:vec2f", "old:UEPair"); replace("->vec2f", "->UEPair");
   replace("var weighted=0.0", "var weighted=vec2f(0)"); replace("var value=0.0", "var value=vec2f(0)");
   replace("return vec2f(select(0.0,weighted/total,total>0.0),root);", "return UEPair(select(vec2f(0),weighted/total,total>0.0),root);");
  }
  if (name === "umExtended") { replace("->vec2f", "->UEPair"); replace("stateIn[slot]", "pairRead(slot)"); }
  if (name === "ueExtend") {
   replace("old:vec2f", "old:UEPair"); replace("->vec2f", "->UEPair");
   replace("UMNeighbor(0,UM_INF,1)", "UMNeighbor(vec2f(0),UM_INF,1)");
   replace("vec2f(0.0,UM_INF)", "UEPair(vec2f(0),UM_INF)");
   replace("stateIn[slot+64u*face.axis]", "pairRead(slot+64u*face.axis)");
  }
  if (name === "ueNote") replace("value:vec2f", "value:UEPair");
  if (name === "ueSweepSeamFine") replace("stateIn[ueOwnSlot(owner,axis)]", "pairRead(ueOwnSlot(owner,axis))");
  if (name === "ueSweepCoarsePack") replace("stateIn[umSlot(face.anchor,face.axis,face.width)]", "pairRead(umSlot(face.anchor,face.axis,face.width))");
  return renamed(s.replaceAll(/\bUMNeighbor\b/g, "UEPairNeighbor"));
 });
 return source + `
@group(1) @binding(2) var<storage,read_write> pairIn:array<f32>;
@group(1) @binding(3) var<storage,read_write> pairOut:array<f32>;
struct UEPair{x:vec2f,y:f32}
struct UEPairNeighbor{value:vec2f,distance:f32,spacing:f32}
fn pairRead(slot:u32)->UEPair{return UEPair(vec2f(stateIn[slot].x,pairIn[slot]),stateIn[slot].y);}
` + copies.join("\n");
}

/** Seed both fields under the same phase authority. Finish each field with
 * the existing hierarchy and publication, selecting its scalar payload. The
 * retained scalar path and paired path are generated from the same formulas. */
export function pairedExtensionSource(source:string,retained=false):string{
 const swept=pairedSweepSource(source);
 requireExact(functionSource(source,"ueSeedStore"),"fn ueSeedStore(slot:u32,value:vec2f){stateOut[slot]=value;stateIn[slot]=value;}","ueSeedStore");
 const names=["umSeedState","umSeedWall","ueSeedOwner","ueSeedFine","ueSeedCoarse","seedListJob","seedJob","seedList","seed","seedSeams"];
 const rename=(s:string)=>names.reduce((s,n)=>s.replaceAll(new RegExp(`\\b${n}\\b`,"g"),`pair_${n}`),s).replaceAll(/\bueSeedStore\b/g,"pair_ueSeedStore");
 const seeds=names.map(name=>{
  if(name==="umSeedState"){
   requireExact(functionSource(source,name), `fn umSeedState(face:UMFace,owner:UMOwner)->vec2f{
 // WGSL select evaluates both operands: reject air before loading a payload.
 if(!umSource(face,owner)){return vec2f(0.0,UM_INF);}
 return vec2f(umPhysical(face),0.0);
}`, name);
   return `fn pair_umSeedState(face:UMFace,owner:UMOwner)->UEPair{
 if(!umSource(face,owner)){return UEPair(vec2f(0),UM_INF);}
 return UEPair(vec2f(umPhysical(face),${retained?"0.0":"pairPhysical(face)"}),0.0);
}`;
  }
  let s=functionSource(source,name);
  if(name==="umSeedWall")s=replaceExact(s,"vec2f(umPhysical(face),select(UM_INF,0.0,source))",`UEPair(vec2f(umPhysical(face),${retained?"0.0":"pairPhysical(face)"}),select(UM_INF,0.0,source))`);
  return rename(s);
 });
 // Only value loads select the companion. Mask words retain their original
 // packed representation, and paired sweeps always read both payloads.
 let finish=source;
 const replaceFinish=(from:string,to:string)=>{finish=replaceExact(finish,from,to);};
 replaceFinish("return stateIn[umSlot(p,axis,1u)].x;","return ueFinishState(umSlot(p,axis,1u)).x;");
 replaceFinish("let state=stateIn[umSlot(anchor,axis,width)];","let state=ueFinishState(umSlot(anchor,axis,width));");
 replaceFinish("let slot=stateIn[select(UE_COARSE+component*UE_TILES+tile,ueUnitSlotIn(word,upper,local,component),patchWidth==1u)];","let slot=ueFinishState(select(UE_COARSE+component*UE_TILES+tile,ueUnitSlotIn(word,upper,local,component),patchWidth==1u));");
 replaceFinish("stateIn[ueOwnSlot(owner,axis)],(umTileSupport(owner.tile)&2u)!=0u)","ueFinishState(ueOwnSlot(owner,axis)),(umTileSupport(owner.tile)&2u)!=0u)");
 // Paired modules publish no current payload: no held values (their layout
 // has none), and no region is held where they publish.
 replaceFinish(`@group(1) @binding(18) var<storage,read_write> ueHeld:array<f32>;
fn ueHeldLoad(slot:u32)->f32{return ueHeld[slot];}
fn ueHeldStore(slot:u32,value:f32){ueHeld[slot]=value;}`,`fn ueHeldLoad(slot:u32)->f32{return 0.0;}
fn ueHeldStore(slot:u32,value:f32){}`);
 return finish+swept.slice(source.length)+`
@group(1) @binding(14) var departure:texture_3d<f32>;
@group(1) @binding(15) var<storage,read> companionNegative:array<f32>;
override ueCompanionValues:bool=false;
// Without a live sample the old payload was never retained: the companion
// restriction then carries the current values, which nothing reads.
fn ueFinishState(slot:u32)->vec2f{
 let state=stateIn[slot];if(ueCompanionValues&&ueParticlesLive()){return vec2f(pairIn[slot],state.y);}return state;
}
fn pairPhysical(face:UMFace)->f32{
 if(umFaceLow(face.anchor,face.axis)){return companionNegative[umLowFaceIndex(ueLowCell(face.anchor,face.axis),face.axis)];}
 return textureLoad(departure,face.anchor,0)[face.axis];
}
fn pair_ueSeedStore(slot:u32,value:UEPair){
 let current=vec2f(value.x.x,value.y);stateIn[slot]=current;stateOut[slot]=current;${retained?"":"pairIn[slot]=value.x.y;pairOut[slot]=value.x.y;"}
}
`+seeds.join("\n")+(retained?retainedCanonicalWGSL:"");
}

/** Retain every canonical raw face before forces replace the physical field.
 * Invalid extension slots may keep raw values: only finite-distance slots
 * contribute to a sweep or restriction. Closed walls also publish raw values
 * outside the support, so this gather deliberately has no support/phase gate.
 * Both epochs must start alike, including when no sweep is requested. */
const retainedCanonicalWGSL=/* wgsl */`
fn retainedStore(slot:u32,value:f32){pairIn[slot]=value;pairOut[slot]=value;}
@compute @workgroup_size(64) fn retainFine(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(!ueParticlesLive()){return;}
 for(var job=group.x;job<umCounts.x;job+=groups.x){
  let tile=umTopology[UM_TILES+job];let origin=umTileCoord(tile)*4u+umCorner(lane,4u);let owner=umOwnerAt(vec3i(origin));
  let value=textureLoad(physical,vec3i(origin),0).xyz;let low=umLowOpen(vec3i(origin));
  for(var axis=0u;axis<3u;axis++){
   retainedStore(ueOwnSlot(owner,axis),value[axis]);
   if(low[axis]){let face=umUnitExtensionFace(owner,axis,-1);retainedStore(umSlot(face.anchor,axis,1u),umPhysical(face));}
  }
 }
}
@compute @workgroup_size(64) fn retainCoarse(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(!ueParticlesLive()){return;}
 // These width-only lists partition every coarse tile, including unsupported
 // closed walls. The sweep's ueSeamCoarseJob support gate must not be used.
 let regular=umSupport[8u*UM_TILES+20u];let packs=(regular+63u)/64u;
 let header=7u*UM_TILES+16u;let fine=umSupport[header];let seams=umSupport[header+1u];
 for(var job=group.x;job<packs+seams;job+=groups.x){
  if(job<packs){
   let index=64u*job+lane;if(index>=regular){continue;}
   let tile=umSupport[8u*UM_TILES+24u+index];let origin=umTileCoord(tile)*4u;let low=umLowOpen(vec3i(origin));
   for(var axis=0u;axis<3u;axis++){
    var anchor=vec3i(origin);anchor[axis]+=3;
    let face=UMFace(UMOwner(),anchor,4u,1u,axis,1);
    retainedStore(umSlot(anchor,axis,4u),umPhysical(face));
    if(low[axis]){anchor=vec3i(origin);anchor[axis]=-1;let wall=UMFace(UMOwner(),anchor,4u,1u,axis,-1);retainedStore(umSlot(anchor,axis,4u),umPhysical(wall));}
   }
  }else{
   let tile=umSupport[header+4u+fine+job-packs];let origin=umTileCoord(tile)*4u;let owner=umOwnerAt(vec3i(origin));
   if(lane<48u){let axis=lane/16u;let face=umFace(owner,axis,1,lane%16u);if(face.width!=0u){retainedStore(umSlot(face.anchor,axis,face.width),umPhysical(face));}}
   else if(lane<51u){let axis=lane-48u;if(umLowOpen(vec3i(origin))[axis]){let face=umFace(owner,axis,-1,0u);retainedStore(umSlot(face.anchor,axis,face.width),umPhysical(face));}}
  }
 }
}
`;

/** Publish the old payload at its consumers' homes: every positive patch in
 * its scalar slot, every h interpolant texel in the existing fine-tap texture.
 * No old extended RGBA field or negative-wall copy is needed by the deferred snapshot. */
export function pairedSnapshotSource(source:string,retained=false):string{
 if(retained)source=replaceExact(source,"return textureLoad(physical,face.anchor,0)[face.axis];","return pairIn[umSlot(face.anchor,face.axis,face.width)];");
 let fine=functionSource(source,"uePublishFine");
 fine=replaceExact(fine,"fn uePublishFine(","fn ueSnapshotFineOwner(");
 fine=replaceExact(fine,"let low=umLowOpen(vec3i(origin));","");
 fine=replaceExact(fine,"if(low[axis]){let face=umUnitExtensionFace(owner,axis,-1);boundary[umLowFaceIndex(origin,axis)]=umPhysical(face);}","");
 fine=replaceExact(fine,"value.w=textureLoad(physical,vec3i(origin),0).w;textureStore(output,vec3i(origin),value);","ueSnapshotStore(owner,value);",2);
 // This module only publishes fine taps; their atlas-only accessor must not
 // mirror unused 4h texels into a base field.
 return rememberSnapshotSelection(source+/* wgsl */`
@group(1) @binding(16) var unitVelocityOut:texture_storage_3d<rgba32float,write>;
fn ueSnapshotStore(owner:UMOwner,value:vec4f){
 for(var axis=0u;axis<3u;axis++){pairIn[ueOwnSlot(owner,axis)]=value[axis];}
 textureStore(unitVelocityOut,vec3i(umOrigin(owner)),vec4f(value.xyz,0.0));
}
${fine}
// Only the regions that publish the old payload (UniformMixedExtension's
// state 3): G2P reads it at no other region's samples, and the current
// publication's replay takes its own selection where no tag was saved.
var<workgroup> ueSnapshotSkip:u32;
@compute @workgroup_size(64) fn snapshotFine(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(!ueParticlesLive()){return;}
 for(var job=group.x;job<umCounts.x;job+=groups.x){
  let tile=umTopology[UM_TILES+job];
  if(lane==0u){ueSnapshotSkip=select(0u,1u,ueRegionState(tile)<3u);}
  if(workgroupUniformLoad(&ueSnapshotSkip)!=0u){continue;}
  let cell=vec3i(umTileCoord(tile)*4u+umCorner(lane,4u));
  ueSnapshotFineOwner(umOwnerAt(cell),lane);workgroupBarrier();
 }
}
fn ueSnapshotPublished(face:UMFace)->f32{
 if(umClosedWall(face)){return umPhysical(face);}
 let slot=umSlotState(face.anchor,face.axis,face.width);if(slot.y<0.5*UM_INF){return slot.x;}
 // A width-4 face centre has one nonzero far interpolation tap: its own
 // tile's 4h cell, which holds nothing in a page that is not stored. Keep
 // its original +0 accumulation and validity mask.
 if(face.width==4u){
  let texel=umTileStoreTexel(face.anchor/4);var state=vec4f(0.0);if(texel.x>=0){state=textureLoad(coarse,texel,0);}
  return select(0.0,0.0+state[face.axis],(u32(round(state.w))&(1u<<face.axis))!=0u);
 }
 return umFarValue(face);
}
// Coarse owners' positive patches occupy either their component-major 4h
// slots or the extra fine seam slots; neither overlaps an h owner's slots.
@compute @workgroup_size(64) fn snapshotCoarse(@builtin(global_invocation_id) gid:vec3u){
 if(!ueParticlesLive()){return;}
 let index=gid.x+65535u*64u*gid.y;let job=index/3u;let axis=index%3u;if(job>=umCounts.y){return;}
 let tile=umTopology[UM_TILES+umCounts.x+job];if(ueRegionState(tile)<3u){return;}let owner=umOwnerAt(vec3i(umTileCoord(tile)*4u));let first=umFace(owner,axis,1,0u);
 for(var part=0u;part<first.count;part++){
  let face=umFacePatch(first,part);pairIn[umSlot(face.anchor,axis,face.width)]=ueSnapshotPublished(face);
 }
}
`);
}

const retainedMaskWGSL=/* wgsl */`
var<private> publicationSelection:u32;
fn publicationRemember(face:UMFace,mask:u32){
 // The taps a face's far value took. Canonical positive slots are unique. The opposite epoch is dead after
 // the final sweep; real extension distances are never negative.
 stateOut[umSlot(face.anchor,face.axis,face.width)]=vec2f(f32(mask),-1.0);
}
`;
function rememberSnapshotSelection(source:string):string{
 let code=source;
 for(const name of ["umFarValue","umFarValueStaged"]){
  const original=functionSource(code,name);let body=original.replace("{","{publicationSelection=0u;");
  body=replaceExact(body,"total+=weight;","total+=weight;publicationSelection|=1u<<k;");code=code.replace(original,body);
 }
 const originalFine=functionSource(code,"ueSnapshotFineOwner");
 const fine=replaceExact(originalFine,"value[axis]=umFarValueStaged(face);","value[axis]=umFarValueStaged(face);publicationRemember(face,publicationSelection);");
 code=code.replace(originalFine,fine);
 const originalCoarse=functionSource(code,"ueSnapshotPublished");let coarse=originalCoarse;
 const valid="(u32(round(state.w))&(1u<<face.axis))!=0u";
 if(!coarse.includes(valid))throw new Error("Paired snapshot coarse validity changed");
 coarse=replaceExact(coarse,"return select(0.0,0.0+state[face.axis],"+valid+");",`publicationRemember(face,select(0u,1u,${valid}));return select(0.0,0.0+state[face.axis],${valid});`);
 coarse=replaceExact(coarse,"return umFarValue(face);","let value=umFarValue(face);publicationRemember(face,publicationSelection);return value;");
 return code.replace(originalCoarse,coarse)+retainedMaskWGSL;
}

export function replayedPublicationSource(source:string):string{
 let code=source;
 const originalPublished=functionSource(code,"umPublished");code=code.replace(originalPublished,replaceExact(originalPublished,"return ueHeldKept(slotIndex,umFarValue(face));","return ueHeldKept(slotIndex,publicationReplay(face,false));"));
 const originalFine=functionSource(code,"uePublishFine");
 const stageMarker=" if(lane<27u){";
 if(originalFine.split(stageMarker).length!==2)throw new Error("Paired publication far staging changed");
 const start=originalFine.indexOf(stageMarker);
 if(start<=0)throw new Error("Paired publication far staging changed");
 const store=[...originalFine.matchAll(/(?:udStore_output\s*\(|textureStore\s*\(\s*output\s*,)[^;]*;/g)];if(store.length!==2)throw new Error("Paired publication exits changed");
 const finalStart=originalFine.lastIndexOf(" ueHeldFine(owner,value,held);value.w=");if(finalStart<=start)throw new Error("Paired publication final store changed");
 const fine=originalFine.slice(0,start)+/* wgsl */`
 // Values are the only per-cell data needed after old publication saved
 // each face's known taps. Preserve the same 27-cell neighborhood.
 if(lane<27u){let texel=umTileStoreTexel(vec3i(umTileCoord(farTile))+vec3i(umCorner(lane,3u))-vec3i(1));if(texel.x>=0){farState[lane]=textureLoad(coarse,texel,0);}}
 workgroupBarrier();
 for(var axis=0u;axis<3u;axis++){
  if((needsFar&(1u<<axis))==0u){continue;}
  value[axis]=publicationReplay(umUnitExtensionFace(owner,axis,1),true);
 }
`+originalFine.slice(finalStart);
 code=code.replace(originalFine,fine);

 return code+/* wgsl */`
// A selected tap is a known cell, so its page is stored.
fn publicationLoad(p:vec3i)->vec4f{return textureLoad(coarse,umTileStoreTexel(p),0);}
fn publicationReplay(face:UMFace,staged:bool)->f32{
 let retained=stateOut[umSlot(face.anchor,face.axis,face.width)];
 // The explicit tag also keeps a safe original path for an uncached face.
 // Without a live sample the old snapshot saved no selection this frame:
 // a tag there is an earlier frame's.
 if(retained.y!=-1.0||!ueParticlesLive()){return umFarValue(face);}
 let selected=u32(retained.x);let point=umFaceCenter(face);var q=point/4.0-vec3f(0.5);q[face.axis]-=0.5;
 let base=vec3i(floor(q));let fraction=fract(q);var sum=0.0;var total=0.0;
 for(var k=0u;k<8u;k++){
  if((selected&(1u<<k))==0u){continue;}
  let bit=vec3i(umCorner(k,2u));let w=select(vec3f(1)-fraction,fraction,bit==vec3i(1));let weight=w.x*w.y*w.z;
  let p=base+bit;
  var value:f32;if(staged){value=farState[umFarSlot(p)][face.axis];}else{value=publicationLoad(p)[face.axis];}
  sum+=weight*value;total+=weight;
 }
 return select(0.0,sum/total,total>0.0);
}
`;
}
