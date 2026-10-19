/** Occupied-cell surface scatter, shared by the listed and fused paths. */
export const narrowBandSplatWGSL=/* wgsl */`
// The nearest banks are scratch, but only mapped pages can be read or
// splatted. Clear both banks over those pages, including their complete
// 33^3 vertex halos. Free capacity needs no bandwidth; a newly mapped page
// is cleared on its first reconstruction, regardless of its previous use.
@compute @workgroup_size(64) fn surfaceClear(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var job=group.x;job<32u*(UM_TILES/512u);job+=groups.x){
  let block=job/32u;if(block==0u||block+1u>=UM_TILES/512u||!umTileLive(512u*block)){continue;}
  for(var i=64u*(job%32u)+lane;i<35937u;i+=2048u){
   let key=35937u*block+i;
   atomicStore(&bins[NB_DEPTH_A+key],0u);
   atomicStore(&bins[NB_SURFACE_HOT_NEAREST+key],0u);
  }
 }
}
// Before classify, the motion bank holds one word per occupied cell.
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
// append is needed. Fixed direct groups stride the current GPU count.
var<workgroup> nbSurfaceJobs:u32;
var<workgroup> nbSurfaceTile:u32;
var<workgroup> nbSurfaceMask:array<atomic<u32>,2>;
@compute @workgroup_size(64) fn surfaceSplatOccupied(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){nbSurfaceJobs=min(atomicLoad(&bins[NB_OCCUPIED]),UM_TILES);}
 let count=workgroupUniformLoad(&nbSurfaceJobs);
 for(var job=group.x;job<count;job+=groups.x){
  workgroupBarrier();
  if(lane<2u){atomicStore(&nbSurfaceMask[lane],0u);}
  if(lane==0u){nbSurfaceTile=atomicLoad(&bins[NB_OCCUPIED+4u+job]);}
  let tile=workgroupUniformLoad(&nbSurfaceTile);umLoadAddressPlan(tile>>9u,lane);let cell=64u*tile+lane;
  var eligible=umTileLive(tile)&&atomicLoad(&bins[cell])!=0u;
  if(eligible){let local=(4u*umTileCoord(tile)+umCorner(lane,4u))&vec3u(15u);eligible=!(all((local>=vec3u(1u))&(local<=vec3u(13u)))&&nbTileCold(tile));}
  if(eligible){atomicOr(&nbSurfaceMask[lane/32u],1u<<(lane%32u));}
  // Selection is one parallel tile scan. Empty cells cost no splat barrier.
  for(var word=0u;word<2u;word++){
   var mask=workgroupUniformLoad(&nbSurfaceMask[word]);
   while(mask!=0u){
    let cell=64u*tile+32u*word+firstTrailingBit(mask);nbSurfaceSplatCell(cell,lane);mask&=mask-1u;
   }
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
