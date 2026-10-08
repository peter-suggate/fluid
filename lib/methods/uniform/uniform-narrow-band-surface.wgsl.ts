/** Particle geometry enters the canonical simulation phi before pressure.
 * The optional dense publication uses the same reconstruction, at h vertices. */
export const NARROW_BAND_SURFACE_RADIUS=0.875;
// A quarter-cell lattice has two tangential offsets of h/4 at a vertex.
export const NARROW_BAND_SEED_DEPTH=Math.sqrt(NARROW_BAND_SURFACE_RADIUS**2-0.125);
export const narrowBandParticleSurfaceWGSL=/* wgsl */`
// Ferstl et al. 2016, Eq. 4: phi <- min(phi' + h, phi_p). The advected level
// set, shrunk by one cell, is united with the surface of the samples, so the
// samples overrule it: a sheet or droplet they carry is liquid whatever grid
// advection made of it, and liquid they have left is not. phi_p is the union
// of one sphere per sample. A tile no sample reaches has no phi_p and keeps
// phi' as it is; liquid a source adds this step joins after the union.
const NB_SURFACE_RADIUS=${NARROW_BAND_SURFACE_RADIUS};
const NB_SEED_DEPTH=${NARROW_BAND_SEED_DEPTH};
fn particleSurfaceBulk(q:vec3i)->vec2f{
 let bulk=bulkDepth(vec3f(q));let tile=umTileAt(min(vec3u(q)/4u,UM_T-1u));
 // Deeper than this the shrunk level set wins the union whatever the samples
 // are; nothing bounds how far into the air they may carry the surface.
 let gather=atomicLoad(&bins[NB_SURFACE_TILES+tile])!=0u&&bulk>-2.0;
 return vec2f(bulk,select(0.0,1.0,gather));
}
// Squared distance to the nearest sample within 2h; 4 when there is none.
fn particleSurfaceGather(q:vec3i,member:u32,stride:u32)->f32{
 let x=vec3f(q);var nearest2=4.0;
 for(var bin=member;bin<64u;bin+=stride){
  let cell=q-2+vec3i(i32(bin%4u),i32((bin/4u)%4u),i32(bin/16u));if(any(cell<vec3i(0))||any(cell>=vec3i(UM_D))){continue;}
  let run=nbCellRun(cell);
  for(var index=run.x;index<run.y;index++){let offset=x-nbPosition(index);nearest2=min(nearest2,dot(offset,offset));}
 }
 return nearest2;
}
// The same minimum, nearest cells first. A sample in one of the eight cells
// at the vertex is within sqrt(3); a cell whose box is no nearer than the
// best sample so far cannot improve on it and is not read. In the band the
// eight cells settle it, 64 samples in place of the 512 of the whole 4-cube.
fn particleSurfaceNearest(q:vec3i)->f32{
 let x=vec3f(q);var nearest2=4.0;
 for(var ring=0u;ring<2u;ring++){
  if(ring==1u&&nearest2<=1.0){break;}
  for(var bin=0u;bin<64u;bin++){
   let o=vec3i(i32(bin%4u),i32((bin/4u)%4u),i32(bin/16u))-2;
   // Whole cells between the vertex and this cell's box, per axis.
   let gap=max(o,-o-1);if(any(gap>vec3i(0))!=(ring==1u)){continue;}
   if(f32(dot(gap,gap))>=nearest2){continue;}
   let cell=q+o;if(any(cell<vec3i(0))||any(cell>=vec3i(UM_D))){continue;}
   let run=nbCellRun(cell);
   for(var index=run.x;index<run.y;index++){let offset=x-nbPosition(index);nearest2=min(nearest2,dot(offset,offset));}
  }
 }
 return nearest2;
}
fn particleSurfaceFinish(q:vec3i,bulk:f32,nearest2:f32)->f32{
 return nbSourcePhi(vec3f(q),min(bulk+1.0,sqrt(nearest2)-NB_SURFACE_RADIUS));
}
fn particleSurface(q:vec3i)->f32{
 let bulk=particleSurfaceBulk(q);if(bulk.y==0.0){return bulk.x;}
 return particleSurfaceFinish(q,bulk.x,particleSurfaceGather(q,0u,1u));
}
`;
/** A fine tile's canonical vertices, one lane each. */
export const narrowBandTiledSurfaceWGSL=/* wgsl */`
@compute @workgroup_size(128) fn coupleFine(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane>=125u){return;}
 let owner=umAllOwner(vec3u(group.x*64u,0,0));let origin=vec3i(umTileCoord(owner.tile)*4u);
 let local=vec3u(lane%5u,(lane/5u)%5u,lane/25u);let q=origin+vec3i(local);
 if(umTileMaximumWidth(owner.tile)==1u&&umTileMinimumWidth(owner.tile)==1u){if(!all((local>vec3u(0))|(origin==vec3i(0)))){return;}}
 else{let authority=umVertexAuthority(vec3u(q));if(authority.width!=1u||authority.tile!=owner.tile){return;}}
 let bulk=particleSurfaceBulk(q);var value=bulk.x;
 if(bulk.y>0.0){value=particleSurfaceFinish(q,bulk.x,particleSurfaceNearest(q));}
 textureStore(outputPhi,q,vec4f(value*min(params.hDt.x,min(params.hDt.y,params.hDt.z))));
}
`;

export const narrowBandSurfaceWGSL=narrowBandParticleSurfaceWGSL+narrowBandTiledSurfaceWGSL+/* wgsl */`
var<workgroup> nbSurfaceNearest:array<f32,64>;
var<workgroup> nbSurfaceBoundary:atomic<u32>;
var<workgroup> nbSurfaceBoundaryCount:u32;
fn nbCoupleVertex(owner:UMOwner,regular:bool,k:u32,lane:u32){
 let origin=umOrigin(owner);let corner=umCorner(k,2u);let q=vec3i(origin+corner*owner.width);let member=lane%16u;
 var owned=false;
 if(owner.width==4u){
  if(regular){owned=all((corner!=vec3u(0))|(origin==vec3u(0)));}
  else{owned=umVertexAuthority(vec3u(q)).index==owner.index;}
 }
 var bulk=vec2f(0);var gather=4.0;
 if(owned){bulk=particleSurfaceBulk(q);if(bulk.y>0.0){gather=particleSurfaceGather(q,member,16u);}}
 nbSurfaceNearest[lane]=gather;workgroupBarrier();
 for(var stride=8u;stride>0u;stride/=2u){
  if(member<stride){nbSurfaceNearest[lane]=min(nbSurfaceNearest[lane],nbSurfaceNearest[lane+stride]);}workgroupBarrier();
 }
 if(member==0u&&owned){
  var value=bulk.x;if(bulk.y>0.0){value=particleSurfaceFinish(q,bulk.x,nbSurfaceNearest[lane]);}
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
 * the 4 cubed cells around the cell less the ten that weigh nothing on any of
 * its three faces, read as 15 x rows. A row is one run of the ordered
 * samples, or two where it crosses a tile face. A cell's eight lanes look the
 * runs up between them and then each take every eighth sample of every run:
 * equal shares of consecutive memory, whatever each cell holds. Eight lanes a
 * cell also keep a workgroup on one x row of cells, which the band holds or
 * misses together. The launch covers the h owners alone; a 4h owner's faces
 * are transferOwners' (uniform-narrow-band-flip.ts). */
export const narrowBandFineTransferWGSL=/* wgsl */`
var<workgroup> nbMomentum:array<vec3f,64>;
var<workgroup> nbMass:array<vec3f,64>;
var<workgroup> nbBlend:array<f32,24>;
var<workgroup> nbRuns:array<vec2u,256>;
@compute @workgroup_size(64) fn transfer(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(vec3u(gid.x/8u,0,0));let origin=umOrigin(owner);
 let team=lane/8u;let member=lane%8u;let centre=vec3f(origin)+0.5;
 // A lane for each face's depth under the surface.
 if(member<3u){
  var near=0.0;
  if(owner.width==1u&&origin[member]+1u<UM_D[member]){
   var q=centre;q[member]+=0.5;let depth=particleDepth(q);
   if(depth<=1.5&&depth>=-2.0){near=1.0;}
  }
  nbBlend[3u*team+member]=near;
 }
 workgroupBarrier();
 let blend=vec3f(nbBlend[3u*team],nbBlend[3u*team+1u],nbBlend[3u*team+2u]);
 let gather=any(blend>vec3f(0));
 if(gather){
  let low=vec3i(origin)-1;
  // The row past both far planes weighs nothing; one in a far plane weighs
  // nothing in its last cell.
  for(var row=member;row<16u;row+=8u){
   let y=low.y+i32(row%4u);let z=low.z+i32(row/4u);var runs=array<vec2u,2>();
   if(row<15u&&y>=0&&y<i32(UM_D.y)&&z>=0&&z<i32(UM_D.z)){
    let last=min(low.x+select(3,2,row%4u==3u||row>=12u),i32(UM_D.x)-1);
    var x=max(low.x,0);
    for(var part=0u;x<=last;part++){let stop=min(last,x|3);runs[part]=nbRun(vec3i(x,y,z),vec3i(stop,y,z));x=stop+1;}
   }
   nbRuns[32u*team+2u*row]=runs[0];nbRuns[32u*team+2u*row+1u]=runs[1];
  }
 }
 workgroupBarrier();
 var momentum=vec3f(0);var total=vec3f(0);
 if(gather){
  for(var slot=0u;slot<30u;slot++){
   let run=nbRuns[32u*team+slot];
   for(var i=run.x+member;i<run.y;i+=8u){
    let motion=nbMotion(i);if(motion.w==1.0){continue;}
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
 if(member==0u&&owner.width==1u){
  let original=textureLoad(velocity,vec3i(origin),0);let mass=nbMass[lane];
  let value=mix(original.xyz,nbMomentum[lane]/max(mass,vec3f(1e-30)),select(vec3f(0),blend,mass>=vec3f(1e-5)));
  textureStore(output,vec3i(origin),vec4f(value,original.w));
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
   let run=nbCellRun(lo+vec3i(vec3u(cell%size.x,(cell/size.x)%size.y,cell/(size.x*size.y))));
   for(var i=run.x;i<run.y;i++){
    let motion=nbMotion(i);if(motion.w==1.0){continue;}
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
