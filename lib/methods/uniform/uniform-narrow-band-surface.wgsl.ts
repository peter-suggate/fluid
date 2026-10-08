/** Particle geometry enters the canonical simulation phi before pressure.
 * The optional dense publication uses the same reconstruction, at h vertices. */
export const NARROW_BAND_SURFACE_RADIUS=0.875;
// A quarter-cell lattice has two tangential offsets of h/4 at a vertex.
export const NARROW_BAND_SEED_DEPTH=Math.sqrt(NARROW_BAND_SURFACE_RADIUS**2-0.125);
export const narrowBandParticleSurfaceWGSL=/* wgsl */`
// Particles supply a candidate interface, not an enclosure constraint.
// Calm regions follow the Eulerian surface; ballistic samples never enter it.
const NB_SURFACE_RADIUS=${NARROW_BAND_SURFACE_RADIUS};
const NB_SEED_DEPTH=${NARROW_BAND_SEED_DEPTH};
fn particleSurfaceBulk(q:vec3i)->vec2f{
 let bulk=bulkDepth(vec3f(q));let tile=umTileAt(min(vec3u(q)/4u,UM_T-1u));
 let gather=atomicLoad(&bins[NB_SURFACE_TILES+tile])!=0u&&abs(bulk)<2.0;
 return vec2f(bulk,select(0.0,1.0,gather));
}
// Squared distance to the nearest sample within 2h; 4 when there is none.
fn particleSurfaceGather(q:vec3i,member:u32,stride:u32)->vec2f{
 let x=vec3f(q);var nearest2=4.0;var activity=0.0;
 for(var bin=member;bin<64u;bin+=stride){
  let cell=q-2+vec3i(i32(bin%4u),i32((bin/4u)%4u),i32(bin/16u));if(any(cell<vec3i(0))||any(cell>=vec3i(UM_D))){continue;}
  var link=atomicLoad(&bins[2u*cellIndex(cell)]);
  for(var j=0u;link!=0u;j++){
   let index=link-1u;let sample=nbSurfaceSample(index);link=links[index];
   if(sample.w>=3.0){continue;}let offset=x-sample.xyz;let d2=dot(offset,offset);nearest2=min(nearest2,d2);
   if(d2<2.25){activity=max(activity,clamp(sample.w-1.0,0.0,1.0));}
  }
 }
 return vec2f(nearest2,activity);
}
fn particleSurfaceFinish(q:vec3i,bulk:f32,gather:vec2f)->f32{
 let base=nbRelaxedBulk(q,bulk,gather.y);
 if(gather.x>=4.0){return base;}
 let candidate=min(bulk+1.0,sqrt(gather.x)-NB_SURFACE_RADIUS);
 // Unlike NB-FLIP's union, correction cannot engulf an escaped particle or
 // erase a sheet in one step. Eulerian tracking always retains authority.
 let correction=clamp(candidate-base,-0.5,0.5);
 return base+0.5*gather.y*(1.0-exp(-8.0*params.hDt.w))*correction;
}
fn particleSurface(q:vec3i)->f32{
 let bulk=particleSurfaceBulk(q);if(bulk.y==0.0){return nbRelaxedBulk(q,bulk.x,0.0);}
 return particleSurfaceFinish(q,bulk.x,particleSurfaceGather(q,0u,1u));
}
`;
/** A fine tile's canonical vertices share streamed batches from its 8^3
 * neighboring bins. Batches are compacted in workgroup memory and unbounded
 * in number: a crowded bin never truncates particles or allocates a bucket. */
export const narrowBandTiledSurfaceWGSL=/* wgsl */`
var<workgroup> nbTileSamples:array<vec4f,128>;
var<workgroup> nbTileCounter:atomic<u32>;
var<workgroup> nbTileCount:u32;
@compute @workgroup_size(128) fn coupleFine(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(vec3u(group.x*64u,0,0));let origin=vec3i(umTileCoord(owner.tile)*4u);
 let local=vec3u(lane%5u,(lane/5u)%5u,lane/25u);let q=origin+vec3i(local);let x=vec3f(q);
 let regular=umTileMaximumWidth(owner.tile)==1u&&umTileMinimumWidth(owner.tile)==1u;
 var owned=false;if(lane<125u){
  if(regular){owned=all((local>vec3u(0))|(origin==vec3i(0)));}
  else{let authority=umVertexAuthority(vec3u(q));owned=authority.width==1u&&authority.tile==owner.tile;}
 }
 var bulk=vec2f(0);if(owned){bulk=particleSurfaceBulk(q);}
 if(lane==0u){atomicStore(&nbTileCounter,0u);}workgroupBarrier();
 if(owned&&bulk.y>0.0){atomicOr(&nbTileCounter,1u);}
 if(owned&&bulk.y==0.0){textureStore(outputPhi,q,vec4f(nbRelaxedBulk(q,bulk.x,0.0)*min(params.hDt.x,min(params.hDt.y,params.hDt.z))));}
 workgroupBarrier();if(lane==0u){nbTileCount=atomicLoad(&nbTileCounter);}
 if(workgroupUniformLoad(&nbTileCount)==0u){return;}
 var nearest2=4.0;var activity=0.0;
 for(var block=0u;block<512u;block+=128u){
  let bin=block+lane;let c=origin-2+vec3i(i32(bin%8u),i32((bin/8u)%8u),i32(bin/64u));var link=0u;
  if(bin<512u&&all(c>=vec3i(0))&&all(c<vec3i(UM_D))){link=atomicLoad(&bins[2u*cellIndex(c)]);}
  loop {
   if(lane==0u){atomicStore(&nbTileCounter,0u);}workgroupBarrier();
   if(link!=0u){let i=link-1u;let slot=atomicAdd(&nbTileCounter,1u);nbTileSamples[slot]=nbSurfaceSample(i);link=links[i];}
   workgroupBarrier();if(lane==0u){nbTileCount=atomicLoad(&nbTileCounter);}
   let count=workgroupUniformLoad(&nbTileCount);if(count==0u){break;}
   if(owned&&bulk.y>0.0){for(var j=0u;j<count;j++){
    let sample=nbTileSamples[j];if(sample.w>=3.0){continue;}
    let offset=x-sample.xyz;let d2=dot(offset,offset);nearest2=min(nearest2,d2);
    if(d2<2.25){activity=max(activity,clamp(sample.w-1.0,0.0,1.0));}
   }}
   workgroupBarrier();
  }
 }
 if(owned&&bulk.y>0.0){
  let value=particleSurfaceFinish(q,bulk.x,vec2f(nearest2,activity));
  textureStore(outputPhi,q,vec4f(value*min(params.hDt.x,min(params.hDt.y,params.hDt.z))));
 }
}
`;

export const narrowBandSurfaceWGSL=narrowBandParticleSurfaceWGSL+narrowBandTiledSurfaceWGSL+/* wgsl */`
var<workgroup> nbSurfaceNearest:array<vec2f,64>;
var<workgroup> nbSurfaceBoundary:atomic<u32>;
var<workgroup> nbSurfaceBoundaryCount:u32;
fn nbCoupleVertex(owner:UMOwner,regular:bool,k:u32,lane:u32){
 let origin=umOrigin(owner);let corner=umCorner(k,2u);let q=vec3i(origin+corner*owner.width);let member=lane%16u;
 var owned=false;
 if(owner.width==4u){
  if(regular){owned=all((corner!=vec3u(0))|(origin==vec3u(0)));}
  else{owned=umVertexAuthority(vec3u(q)).index==owner.index;}
 }
 var bulk=vec2f(0);var gather=vec2f(4,0);
 if(owned){bulk=particleSurfaceBulk(q);if(bulk.y>0.0){gather=particleSurfaceGather(q,member,16u);}}
 nbSurfaceNearest[lane]=gather;workgroupBarrier();
 for(var stride=8u;stride>0u;stride/=2u){
  if(member<stride){nbSurfaceNearest[lane]=vec2f(min(nbSurfaceNearest[lane].x,nbSurfaceNearest[lane+stride].x),max(nbSurfaceNearest[lane].y,nbSurfaceNearest[lane+stride].y));}workgroupBarrier();
 }
 if(member==0u&&owned){
  var value=nbRelaxedBulk(q,bulk.x,0.0);if(bulk.y>0.0){value=particleSurfaceFinish(q,bulk.x,nbSurfaceNearest[lane]);}
  textureStore(outputPhi,q,vec4f(value*min(params.hDt.x,min(params.hDt.y,params.hDt.z))));
 }
 workgroupBarrier();
}
@compute @workgroup_size(64) fn couple(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(vec3u(umCounts.x*64u+gid.x/16u,0,0));let origin=umOrigin(owner);
 let regular=umTileMaximumWidth(owner.tile)==owner.width&&umTileMinimumWidth(owner.tile)==owner.width;
 // Fine tiles publish all of their canonical vertices in coupleFine. This
 // launch visits only coarse owners, including their transition vertices.
 if(lane==0u){atomicStore(&nbSurfaceBoundary,0u);}workgroupBarrier();
 if(lane%16u==0u&&owner.width==4u&&(!regular||any(origin==vec3u(0)))){atomicOr(&nbSurfaceBoundary,1u);}
 workgroupBarrier();
 nbCoupleVertex(owner,regular,7u,lane);
 if(lane==0u){nbSurfaceBoundaryCount=atomicLoad(&nbSurfaceBoundary);}
 let boundaries=workgroupUniformLoad(&nbSurfaceBoundaryCount);
 if(boundaries!=0u){for(var k=0u;k<7u;k++){nbCoupleVertex(owner,regular,k,lane);}}
}
@compute @workgroup_size(64) fn surface(@builtin(global_invocation_id) gid:vec3u){
 let dims=UM_D+1u;let count=dims.x*dims.y*dims.z;
 for(var i=gid.x;i<count;i+=65536u){
  let q=vec3i(vec3u(i%dims.x,(i/dims.x)%dims.y,i/(dims.x*dims.y)));
  // Publication only; do not erode a second time after coupling.
  textureStore(surfacePhi,q,vec4f(bandPhi(vec3f(q))*min(params.hDt.x,min(params.hDt.y,params.hDt.z))));
  if(all(q<vec3i(UM_D))){textureStore(surfaceOpen,q,vec4f(umCellOpen(q)));}
 }
}
`;

/** Three fine MAC components share the same particle reads. Their union is
 * 4 cubed bins, versus three separate 4 by 3 by 3 gathers. */
export const narrowBandFineTransferWGSL=/* wgsl */`
var<workgroup> nbMomentum:array<vec3f,64>;
var<workgroup> nbMass:array<vec3f,64>;
var<workgroup> nbBlend:array<vec3f,8>;
@compute @workgroup_size(64) fn transfer(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(vec3u(gid.x/8u,0,0));let origin=umOrigin(owner);
 let team=lane/8u;let member=lane%8u;let centre=vec3f(origin)+0.5;
 if(member==0u){
  var blend=vec3f(0);
  if(owner.width==1u){for(var axis=0u;axis<3u;axis++){
   if(origin[axis]+1u>=UM_D[axis]){continue;}
   var q=centre;q[axis]+=0.5;let depth=particleDepth(q);
   if(depth<=1.5){blend[axis]=select(0.0,1.0,depth>=-2.0);}
  }}
  nbBlend[team]=blend;
 }
 workgroupBarrier();
 var momentum=vec3f(0);var total=vec3f(0);
 if(owner.width==1u&&any(nbBlend[team]>vec3f(0))){
  for(var bin=member;bin<64u;bin+=8u){
   let c=vec3i(origin)-1+vec3i(i32(bin%4u),i32((bin/4u)%4u),i32(bin/16u));
   if(any(c<vec3i(0))||any(c>=vec3i(UM_D))){continue;}
   var link=atomicLoad(&bins[2u*cellIndex(c)]);
   while(link!=0u){let i=link-1u;let motion=nbMotion(i);link=links[i];if(motion.w==1.0){continue;}
    let d=centre-nbPosition(i);let f=d+0.5;
    let wc=vec3f(weight(d.x),weight(d.y),weight(d.z));let wf=vec3f(weight(f.x),weight(f.y),weight(f.z));
    let w=vec3f(wf.x*wc.y*wc.z,wc.x*wf.y*wc.z,wc.x*wc.y*wf.z);
    total+=w;momentum+=w*motion.xyz;
   }
  }
 }
 nbMomentum[lane]=momentum;nbMass[lane]=total;workgroupBarrier();
 for(var stride=4u;stride>0u;stride/=2u){
  if(member<stride){nbMomentum[lane]+=nbMomentum[lane+stride];nbMass[lane]+=nbMass[lane+stride];}workgroupBarrier();
 }
 if(member==0u&&owner.width!=0u){
  if(owner.width==1u){
   let original=textureLoad(velocity,vec3i(origin),0);let mass=nbMass[lane];
   let value=mix(original.xyz,nbMomentum[lane]/max(mass,vec3f(1e-30)),select(vec3f(0),nbBlend[team],mass>=vec3f(1e-5)));
   textureStore(output,vec3i(origin),vec4f(value,original.w));
  }else{transferFallback(owner);}
 }
}
`;

/** One workgroup per coarse face, with an exact particle gather and parallel
 * reduction. It uses the same h-scale quadratic kernel as the fine transfer. The 4h particle
 * band retains the axial footprint, omitting only the small diagonal tail. */
export const narrowBandCoarseTransferWGSL=/* wgsl */`
var<workgroup> coarseSums:array<vec2f,64>;
@compute @workgroup_size(64) fn transferCoarse(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+256u*group.y;if(tile>=UM_T.x*UM_T.y*UM_T.z){return;}
 let axis=group.z;let origin=vec3i(vec3u(tile%UM_T.x,(tile/UM_T.x)%UM_T.y,tile/(UM_T.x*UM_T.y)))*4;
 var anchor=origin;anchor[axis]+=3;var q=vec3f(origin)+2.0;q[axis]+=2.0;
 let depth=particleDepth(q);let blend=select(0.0,1.0,depth>=-2.0);
 let original=textureLoad(velocity,anchor,0);var sum=vec2f(0);
 if(blend>0.0&&depth<=1.5&&q[axis]<f32(UM_D[axis])){
  let lo=max(vec3i(floor(q-1.5)),vec3i(0));let hi=min(vec3i(ceil(q+1.5)),vec3i(UM_D));let size=vec3u(hi-lo);
  for(var cell=lane;cell<size.x*size.y*size.z;cell+=64u){
   let p=lo+vec3i(vec3u(cell%size.x,(cell/size.x)%size.y,cell/(size.x*size.y)));var link=atomicLoad(&bins[2u*cellIndex(p)]);
   for(var j=0u;link!=0u;j++){
    let i=link-1u;link=links[i];let motion=nbMotion(i);if(motion.w==1.0){continue;}
    let r=q-nbPosition(i);let w=weight(r.x)*weight(r.y)*weight(r.z);
    sum+=vec2f(w*motion[axis],w);
   }
  }
 }
 coarseSums[lane]=sum;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){coarseSums[lane]+=coarseSums[lane+stride];}workgroupBarrier();}
 if(lane==0u){var value=original;let result=coarseSums[0];if(result.y>1e-5){value[axis]=mix(original[axis],result.x/result.y,blend);}textureStore(output,anchor,value);}
}
`;
