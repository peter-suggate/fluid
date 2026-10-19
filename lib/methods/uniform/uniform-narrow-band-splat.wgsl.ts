/** Occupied-cell surface scatter, shared by the listed and fused paths. */
export const narrowBandSplatWGSL=/* wgsl */`
// Before the deposit, the motion bank holds one word per occupied cell.
// There can be no more occupied cells than sorted live particles.
fn nbSurfaceCells()->u32{return UM_CELL_KEYS+4u*arrayLength(&particles);}
@compute @workgroup_size(64) fn surfaceCells(@builtin(global_invocation_id) gid:vec3u){
 for(var cell=gid.x;cell<UM_CELL_KEYS;cell+=65536u){
  if(!umTileLive(cell/64u)){continue;}
  if(nbSparseBins&&atomicLoad(&bins[NB_COVERAGE+cell/64u])==0u){continue;}
  if(atomicLoad(&bins[cell])==0u){continue;}
  // A cell whose splat footprint (cell - 1 to cell + 2) lies in a COLD
  // region's closure feeds only vertices reconstruction holds.
  {let l=(4u*umTileCoord(cell/64u)+umCorner(cell%64u,4u))&vec3u(15u);if(all((l>=vec3u(1u))&(l<=vec3u(13u)))&&nbTileCold(cell/64u)){continue;}}
  let slot=atomicAdd(&bins[NB_BAND+3u],1u);links[nbSurfaceCells()+slot]=cell;
 }
}
// Probe-only scalar reference; production compiles surfaceSplat alone.
@compute @workgroup_size(64) fn surfaceSplatReference(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let cell=links[nbSurfaceCells()+group.x];let c=4u*umTileCoord(cell/64u)+umCorner(cell%64u,4u);
 let q=vec3i(c)+vec3i(umCorner(lane,4u))-1;
 if(!umVertexPresent(q)){return;}
 let start=links[cell];let end=start+atomicLoad(&bins[cell]);var nearest2=4.0;var hot2=4.0;
 for(var i=start;i<end;i++){let d=vec3f(q)-nbPosition(i);nearest2=min(nearest2,dot(d,d));if(nbSurfaceParticleHot(i)){hot2=min(hot2,dot(d,d));}}
 if(nearest2<4.0){atomicMax(&bins[NB_DEPTH_A+umVertexAt(vec3u(q))],~bitcast<u32>(nearest2));}
 if(hot2<4.0){atomicMax(&bins[NB_SURFACE_HOT_NEAREST+umVertexAt(vec3u(q))],~bitcast<u32>(hot2));}
}
// All 64 vertices visit the same cell's particle stream. Load each position
// once per workgroup, retaining the scalar gather order and all particles.
var<workgroup> nbSurfacePositions:array<vec3f,64>;
var<workgroup> nbSurfaceHot:array<u32,64>;
var<workgroup> nbSurfaceRange:vec2u;
@compute @workgroup_size(64) fn surfaceSplat(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 nbSurfaceSplatCell(links[nbSurfaceCells()+group.x],lane);
}
// Fuse occupied-cell selection and its splat. The sort already maintains
// every occupied tile; no whole-lattice scan, temporary cell list or atomic
// append is needed. Fixed direct groups stride the current GPU count. Each
// tile's samples scatter in parallel into a shared 7^3 vertex window (cells
// -1 to 5): unsigned minima of nonnegative distances are order independent,
// so one flush a touched vertex gives the per-cell scalar gather's bits.
var<workgroup> nbSurfaceJobs:u32;
var<workgroup> nbSurfaceTile:u32;
var<workgroup> nbSurfaceWindow:array<atomic<u32>,343>;
var<workgroup> nbSurfaceWindowHot:array<atomic<u32>,343>;
const NB_SURFACE_FAR:u32=0x40800000u;
@compute @workgroup_size(128) fn surfaceSplatOccupied(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){nbSurfaceJobs=min(atomicLoad(&bins[NB_OCCUPIED]),UM_TILES);}
 let count=workgroupUniformLoad(&nbSurfaceJobs);
 for(var job=group.x;job<count;job+=groups.x){
  workgroupBarrier();
  for(var k=lane;k<343u;k+=128u){atomicStore(&nbSurfaceWindow[k],NB_SURFACE_FAR);atomicStore(&nbSurfaceWindowHot[k],NB_SURFACE_FAR);}
  if(lane==0u){nbSurfaceTile=atomicLoad(&bins[NB_OCCUPIED+4u+job]);}
  let tile=workgroupUniformLoad(&nbSurfaceTile);
  if(!umTileLive(tile)){continue;}
  umLoadAddressPlan(tile>>9u,lane);
  let origin=4u*umTileCoord(tile);let cold=nbTileCold(tile);
  let start=links[64u*tile];let end=links[64u*tile+63u]+atomicLoad(&bins[64u*tile+63u]);
  for(var i=start+lane;i<end;i+=128u){
   // The sort's cell: positions relative to a tile origin are exact.
   let p=nbPosition(i)-vec3f(origin);let c=clamp(vec3i(floor(p)),vec3i(0),vec3i(3));
   // A cell whose splat footprint (cell - 1 to cell + 2) lies in a COLD
   // region's closure feeds only vertices reconstruction holds.
   let l=(origin+vec3u(c))&vec3u(15u);if(cold&&all((l>=vec3u(1u))&(l<=vec3u(13u)))){continue;}
   let hot=nbSurfaceParticleHot(i);
   for(var k=0u;k<64u;k++){
    let v=c+vec3i(umCorner(k,4u))-1;let d=vec3f(v)-p;let d2=dot(d,d);
    if(d2<4.0){let w=u32(v.x+1+7*(v.y+1)+49*(v.z+1));atomicMin(&nbSurfaceWindow[w],bitcast<u32>(d2));if(hot){atomicMin(&nbSurfaceWindowHot[w],bitcast<u32>(d2));}}
   }
  }
  workgroupBarrier();
  for(var k=lane;k<343u;k+=128u){
   let nearest=atomicLoad(&nbSurfaceWindow[k]);if(nearest==NB_SURFACE_FAR){continue;}
   let q=vec3i(origin)+vec3i(umCorner(k,7u))-1;
   // One slot lookup serves the presence test and the depth key.
   let slot=umVertexSlot(q);if(!umBlockStored(slot.w)){continue;}
   // Positive float bit order: complement reverses it; zero is the empty key.
   atomicMax(&bins[NB_DEPTH_A+umVertexSlotAt(slot)],~nearest);
   let hot=atomicLoad(&nbSurfaceWindowHot[k]);if(hot!=NB_SURFACE_FAR){atomicMax(&bins[NB_SURFACE_HOT_NEAREST+umVertexSlotAt(slot)],~hot);}
  }
 }
}
fn nbSurfaceSplatCell(cell:u32,lane:u32){
 let c=4u*umTileCoord(cell/64u)+umCorner(cell%64u,4u);
 let q=vec3i(c)+vec3i(umCorner(lane,4u))-1;
 // One slot lookup serves the presence test and the depth key.
 let slot=umVertexSlot(q);let present=umBlockStored(slot.w);
 if(lane==0u){let start=links[cell];nbSurfaceRange=vec2u(start,start+atomicLoad(&bins[cell]));}
 let range=workgroupUniformLoad(&nbSurfaceRange);let start=range.x;let end=range.y;var nearest2=4.0;var hot2=4.0;
 for(var first=start;first<end;first+=64u){
  let count=min(64u,end-first);
  if(lane<count){nbSurfacePositions[lane]=nbPosition(first+lane);nbSurfaceHot[lane]=u32(nbSurfaceParticleHot(first+lane));}
  workgroupBarrier();
  if(present){for(var i=0u;i<count;i++){let d=vec3f(q)-nbSurfacePositions[i];nearest2=min(nearest2,dot(d,d));if(nbSurfaceHot[i]!=0u){hot2=min(hot2,dot(d,d));}}}
  workgroupBarrier();
 }
 // Positive float bit order: complement reverses it; zero is the empty key.
 if(present&&nearest2<4.0){atomicMax(&bins[NB_DEPTH_A+umVertexSlotAt(slot)],~bitcast<u32>(nearest2));}
 if(present&&hot2<4.0){atomicMax(&bins[NB_SURFACE_HOT_NEAREST+umVertexSlotAt(slot)],~bitcast<u32>(hot2));}
}
// Erasure belongs to the samples that represent this patch, not to hot
// samples of another surface sharing its 4h heat-interpolation stencil.
fn nbCachedSurfaceErasure(q:vec3i,nearest2:f32)->bool{
 let key=atomicLoad(&bins[NB_SURFACE_HOT_NEAREST+umVertexAt(vec3u(q))]);
 return key!=0u&&bitcast<f32>(~key)<=nearest2;
}
fn nbCachedSurfaceNearest(q:vec3i)->f32{
 let key=atomicLoad(&bins[NB_DEPTH_A+umVertexAt(vec3u(q))]);
 if(key==0u){return 4.0;}return bitcast<f32>(~key);
}
`;
