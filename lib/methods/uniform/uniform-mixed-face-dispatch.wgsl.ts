/** Shared, single-writer traversal of canonical mixed MAC patches. A native
 * RGBA texel can contain patches from several axes; one owner packs them all.
 * Callers bind output (RGBA texture) and boundary (negative face buffer). */
export const uniformMixedFaceAddressWGSL = /* wgsl */ `
fn umNegativeBoundaryIndex(p:vec3u,axis:u32)->u32 {
 if(axis==0u){return p.y+UM_D.y*p.z;}
 if(axis==1u){return UM_D.y*UM_D.z+p.x+UM_D.x*p.z;}
 return UM_D.y*UM_D.z+UM_D.x*UM_D.z+p.x+UM_D.x*p.y;
}
fn umPositiveFaceAtAnchor(owner:UMOwner,axis:u32,anchor:vec3i)->UMFace {
 let origin=vec3i(umOrigin(owner));let local=anchor-origin;
 if(owner.width==1u){if(all(local==vec3i(0))){return umFace(owner,axis,1,0u);}return UMFace();}
 if(local[axis]!=i32(owner.width)-1){return UMFace();}
 let first=umFace(owner,axis,1,0u);let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 if(local[u]%i32(first.width)!=0||local[v]%i32(first.width)!=0){return UMFace();}
 return umFace(owner,axis,1,u32(local[u])/first.width+(owner.width/first.width)*(u32(local[v])/first.width));
}
`;

/** `owner` names a caller-defined fn(gid:vec3u)->UMOwner that replaces the
 * tier or all-width traversal (the live remap visits a changed-tile list). */
export function uniformMixedFaceDispatchWGSL(entry: string, evaluate: string, allWidths = false, metadata = "", owner?: string): string {
  return /* wgsl */ `
@compute @workgroup_size(64) fn ${entry}(@builtin(global_invocation_id) gid:vec3u){
 let owner=${owner ?? (allWidths ? "umAllOwner" : "umOwner")}(gid);if(owner.width==0u){return;}
 let origin=umOrigin(owner);
 for(var axis=0u;axis<3u;axis++){
  if(origin[axis]==0u){let face=umFace(owner,axis,-1,0u);boundary[umNegativeBoundaryIndex(origin,axis)]=${evaluate};}
 }
 // Unit owners pack their three positive components at the same anchor.
 // Visit that anchor once without cloning the (large) evaluator call graph.
 for(var axis=0u;axis<select(3u,1u,owner.width==1u);axis++){
  let first=umFace(owner,axis,1,0u);
  for(var part=0u;part<first.count;part++){
   let ownedFace=umFace(owner,axis,1,part);var earlier=false;
   for(var other=0u;other<axis;other++){earlier=earlier||umPositiveFaceAtAnchor(owner,other,ownedFace.anchor).width!=0u;}
   if(earlier){continue;}
   var value=vec4f(0);
   for(var other=0u;other<3u;other++){
    let face=umPositiveFaceAtAnchor(owner,other,ownedFace.anchor);
    if(face.width!=0u){value[other]=${evaluate};}
   }
   ${metadata}
   textureStore(output,ownedFace.anchor,value);
  }
 }
}`;
}

/** One group per topology tile, one lane per fine anchor and component.
 * Only canonical face anchors evaluate the supplied expression. Packing after
 * the barrier retains exactly one writer per RGBA texel without serializing
 * every patch and component of a coarse owner through one expensive sampler.
 * Dispatch with tileGroups=true; the owner/worklist ABI itself is unchanged. */
export function uniformMixedFaceTileDispatchWGSL(entry: string, evaluate: string): string {
  return /* wgsl */ `
var<workgroup> ${entry}Components:array<vec2f,192>;
@compute @workgroup_size(192) fn ${entry}(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let cell=lane%64u;let axis=lane/64u;let local=umCorner(cell,4u);
 let cells=64u/(umCellWidth*umCellWidth*umCellWidth);
 // Merged jobs past the tile jobs pack 64 regular coarse owners: one lane
 // per owner and axis. Their faces are single patches and no two positive
 // faces of a 4h lattice share an anchor texel.
 // Seam 4h jobs of merged launches pack four tiles: 48 lanes per tile, one
 // per (patch, axis). A 4h owner has at most 16 patches per positive face, so
 // one tile per group left 144+ of 192 lanes idle behind a serial sampler.
 // ceil(s4/4) quad jobs replace the s4 seam 4h jobs of the tile job order
 // (umMergedTileJob; fused: umFusedOwner); later jobs shift down by the
 // difference. Size merged launches by dispatchCertified(...,true) (the
 // frame plan's quad-packed count) and fused ones by dispatchFused(...,true,true).
 let header=7u*UM_TILES+16u;let fours=umSupport[header+1u];let quads=(fours+3u)/4u;
 let seamFour=select(umSupport[4u*UM_TILES+2u],umSupport[header],umFusedJobs);
 let job=group.x+umDispatchX*group.y;
 let quad=umMergedTiles&&job>=seamFour&&job<seamFour+quads;
 let tileJob=select(job,job+fours-quads,umMergedTiles&&job>=seamFour+quads);
 let packed=umMergedTiles&&!umFusedJobs&&tileJob>=umMergedTileJobs();
 let slot=lane/48u;let part=(lane%48u)/3u;let faceAxis=lane%3u;
 var owner=UMOwner();var anchor=vec3i(0);var result=vec2f(0);
 if(quad){
  let index=4u*(job-seamFour)+slot;
  if(index<fours){
   let tile=umSupport[header+4u+umSupport[header]+index];
   owner=UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
   let origin=umOrigin(owner);
   if(part==0u&&origin[faceAxis]==0u){
    let face=umFace(owner,faceAxis,-1,0u);boundary[umNegativeBoundaryIndex(origin,faceAxis)]=${evaluate};
   }
   let face=umFace(owner,faceAxis,1,part);
   if(face.width!=0u){anchor=face.anchor;result=vec2f(${evaluate},1);}
  }
 }else{
  let firstOwner=umOwner(vec3u(select(tileJob*cells,tileJob*64u+cell,packed),0,0));owner=firstOwner;
  if(firstOwner.width!=0u){
   var negative=false;var positive=UMFace();
   if(packed){
    let origin=umOrigin(owner);negative=origin[axis]==0u;
    positive=umFace(owner,axis,1,0u);anchor=positive.anchor;
   }else{
    // Merged launches (umCellWidth 1, 64 slots per job) carry a runtime width.
    let width=select(umCellWidth,firstOwner.width,umMergedTiles);let q=local/width;let side=4u/width;
    owner.lane=q.x+side*(q.y+side*q.z);owner.index+=owner.lane;
    anchor=vec3i(umTileCoord(owner.tile)*4u+local);
    let origin=umOrigin(owner);negative=origin[axis]==0u&&all(vec3u(anchor)==origin);
    positive=umPositiveFaceAtAnchor(owner,axis,anchor);
   }
   if(negative){
    let face=umFace(owner,axis,-1,0u);boundary[umNegativeBoundaryIndex(umOrigin(owner),axis)]=${evaluate};
   }
   let face=positive;
   if(face.width!=0u){result=vec2f(${evaluate},1);}
  }
 }
 ${entry}Components[lane]=result;workgroupBarrier();
 if(quad){
  if(result.y>0.0){
   // One writer per RGBA texel: the lowest axis with a patch at this anchor.
   let offset=anchor-vec3i(umOrigin(owner));var value=vec4f(0);var writer=true;
   for(var other=0u;other<3u;other++){
    if(other==faceAxis){value[other]=result.x;continue;}
    let face=umPositiveFaceAtAnchor(owner,other,anchor);if(face.width==0u){continue;}
    if(other<faceAxis){writer=false;}
    let u=(other+1u)%3u;let v=(other+2u)%3u;let side=4u/face.width;
    let otherPart=u32(offset[u])/face.width+side*(u32(offset[v])/face.width);
    value[other]=${entry}Components[48u*slot+3u*otherPart+other].x;
   }
   if(writer){textureStore(output,anchor,value);}
  }
 }else if(packed){
  if(result.y>0.0){var value=vec4f(0);value[axis]=result.x;textureStore(output,anchor,value);}
 }else if(lane<64u){
  let x=${entry}Components[cell];let y=${entry}Components[cell+64u];let z=${entry}Components[cell+128u];
  if(x.y+y.y+z.y>0.0){textureStore(output,anchor,vec4f(x.x,y.x,z.x,0));}
 }
}`;
}
