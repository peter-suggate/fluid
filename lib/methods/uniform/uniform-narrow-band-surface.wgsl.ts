/** Particle geometry enters the canonical simulation phi before pressure.
 * The optional dense publication uses the same reconstruction, at h vertices. */
export const narrowBandParticleSurfaceWGSL=/* wgsl */`
// Weighted half-space centroid for the eight-per-cell lattice and this
// radius-2 kernel. Calibrating the reconstruction radius to that centroid
// preserves a seeded planar interface without any target-volume shift.
const NB_SURFACE_RADIUS=0.5054021866589737;
fn surfaceWeight(d:vec3f)->f32{let t=max(0.0,1.0-dot(d,d)/4.0);return t*t*t;}
fn particleSurfaceBulk(q:vec3i)->vec2f{
 let x=vec3f(q);let bulk=particleDepth(x);let tile=umTileAt(min(vec3u(q)/4u,UM_T-1u));
 // Guard stale shallow bulk values before erosion, too: retiring interior
 // samples must not turn their former covered tiles into artificial air.
 // The advected particles mark their support, including excursions outside
 // the Eulerian level set. Never clip reconstruction to that old surface.
 let covered=atomicLoad(&bins[NB_COVERAGE+tile])!=0u;
 let erosion=select(0.0,1.0,covered&&bulk>-1.5&&bulk<2.0);
 let erodedBulk=bulk+erosion;
 if(atomicLoad(&bins[NB_SURFACE_TILES+tile])==0u){return vec2f(erodedBulk,0);}
 // The particle surface is bounded below by -NB_SURFACE_RADIUS. Here the bulk wins
 // the union for every possible particle configuration; no gather is needed.
 return vec2f(erodedBulk,select(1.0,0.0,erodedBulk<=-NB_SURFACE_RADIUS));
}
struct NBSurfaceGather{weighted:vec4f,nearest2:f32}
fn particleSurfaceGather(q:vec3i,member:u32,stride:u32)->NBSurfaceGather{
 let x=vec3f(q);
   var sum=vec3f(0);var total=0.0;var nearest2=4.0;var wallMask=0u;var mirrorOrigin=vec3f(0);
   for(var a=0u;a<3u;a++){
    if(x[a]<2.0){wallMask|=1u<<a;}
    else if(x[a]>f32(UM_D[a])-2.0&&!(a==1u&&params.settings.y>0.5)){wallMask|=1u<<a;mirrorOrigin[a]=2.0*f32(UM_D[a]);}
   }
   for(var bin=member;bin<64u;bin+=stride){
    let cell=q-2+vec3i(i32(bin%4u),i32((bin/4u)%4u),i32(bin/16u));if(any(cell<vec3i(0))||any(cell>=vec3i(UM_D))){continue;}
    var link=atomicLoad(&bins[2u*cellIndex(cell)]);
    for(var j=0u;link!=0u;j++){
     let index=link-1u;let centre=nbPosition(index);
     let offset=x-centre;nearest2=min(nearest2,dot(offset,offset));
     let w=surfaceWeight(offset);sum+=w*offset;total+=w;
     if(wallMask!=0u){let mirror=mirrorOrigin-centre;
      for(var mask=1u;mask<8u;mask++){
       if((mask&wallMask)!=mask){continue;}
       let c=select(centre,mirror,vec3<bool>((mask&1u)!=0u,(mask&2u)!=0u,(mask&4u)!=0u));
       let mw=surfaceWeight(x-c);sum+=mw*(x-c);total+=mw;
      }
     }
     link=links[index];
    }
   }
 return NBSurfaceGather(vec4f(sum,total),nearest2);
}
fn particleSurfaceFinish(q:vec3i,bulk:f32,gather:NBSurfaceGather)->f32{
 var particlePhi=4.0;
 if(gather.weighted.w>1e-6){
  // The centroid cannot create liquid without nearby actual samples.
  particlePhi=max(length(gather.weighted.xyz/gather.weighted.w)-NB_SURFACE_RADIUS,sqrt(gather.nearest2)-0.75);
  // The centroid alone rounds off sheet corners and can exclude their
  // samples. Union the inscribed sample spheres; sqrt(3)/4 is the distance
  // from a seeded quarter-cell sample to its nearest grid vertex, preserving
  // the flat lattice's zero contour while keeping sparse corners represented.
  particlePhi=min(particlePhi,sqrt(gather.nearest2)-0.4330127018922193);
 }
 return min(bulk,particlePhi);
}
fn particleSurface(q:vec3i)->f32{
 let bulk=particleSurfaceBulk(q);if(bulk.y==0.0){return bulk.x;}
 return particleSurfaceFinish(q,bulk.x,particleSurfaceGather(q,0u,1u));
}
`;
/** A fine tile's canonical vertices share streamed batches from its 8^3
 * neighboring bins. Batches are compacted in workgroup memory and unbounded
 * in number: a crowded bin never truncates particles or allocates a bucket. */
export const narrowBandTiledSurfaceWGSL=/* wgsl */`
var<workgroup> nbTileSamples:array<vec3f,128>;
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
 if(owned&&bulk.y==0.0){textureStore(outputPhi,q,vec4f(bulk.x*min(params.hDt.x,min(params.hDt.y,params.hDt.z))));}
 workgroupBarrier();if(lane==0u){nbTileCount=atomicLoad(&nbTileCounter);}
 if(workgroupUniformLoad(&nbTileCount)==0u){return;}
 var sum=vec3f(0);var total=0.0;var nearest2=4.0;var wallMask=0u;var mirrorOrigin=vec3f(0);
 for(var a=0u;a<3u;a++){
  if(x[a]<2.0){wallMask|=1u<<a;}
  else if(x[a]>f32(UM_D[a])-2.0&&!(a==1u&&params.settings.y>0.5)){wallMask|=1u<<a;mirrorOrigin[a]=2.0*f32(UM_D[a]);}
 }
 for(var block=0u;block<512u;block+=128u){
  let bin=block+lane;let c=origin-2+vec3i(i32(bin%8u),i32((bin/8u)%8u),i32(bin/64u));var link=0u;
  if(bin<512u&&all(c>=vec3i(0))&&all(c<vec3i(UM_D))){link=atomicLoad(&bins[2u*cellIndex(c)]);}
  loop {
   if(lane==0u){atomicStore(&nbTileCounter,0u);}workgroupBarrier();
   if(link!=0u){let i=link-1u;let slot=atomicAdd(&nbTileCounter,1u);nbTileSamples[slot]=nbPosition(i);link=links[i];}
   workgroupBarrier();if(lane==0u){nbTileCount=atomicLoad(&nbTileCounter);}
   let count=workgroupUniformLoad(&nbTileCount);if(count==0u){break;}
   if(owned&&bulk.y>0.0){for(var j=0u;j<count;j++){
    let centre=nbTileSamples[j];let offset=x-centre;let distance2=dot(offset,offset);
    // A closed-wall mirror cannot be nearer than its real sample.
    if(distance2>=4.0){continue;}
    nearest2=min(nearest2,distance2);let w=surfaceWeight(offset);sum+=w*offset;total+=w;
    if(wallMask!=0u){let mirror=mirrorOrigin-centre;
     for(var mask=1u;mask<8u;mask++){
      if((mask&wallMask)!=mask){continue;}
      let m=select(centre,mirror,vec3<bool>((mask&1u)!=0u,(mask&2u)!=0u,(mask&4u)!=0u));
      let mw=surfaceWeight(x-m);sum+=mw*(x-m);total+=mw;
     }
    }
   }}
   workgroupBarrier();
  }
 }
 if(owned&&bulk.y>0.0){
  let value=particleSurfaceFinish(q,bulk.x,NBSurfaceGather(vec4f(sum,total),nearest2));
  textureStore(outputPhi,q,vec4f(value*min(params.hDt.x,min(params.hDt.y,params.hDt.z))));
 }
}
`;

export const narrowBandSurfaceWGSL=narrowBandParticleSurfaceWGSL+narrowBandTiledSurfaceWGSL+/* wgsl */`
var<workgroup> nbSurfaceSums:array<vec4f,64>;
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
 var bulk=vec2f(0);var gathered=NBSurfaceGather(vec4f(0),4.0);
 if(owned){bulk=particleSurfaceBulk(q);if(bulk.y>0.0){gathered=particleSurfaceGather(q,member,16u);}}
 nbSurfaceSums[lane]=gathered.weighted;nbSurfaceNearest[lane]=gathered.nearest2;workgroupBarrier();
 for(var stride=8u;stride>0u;stride/=2u){
  if(member<stride){nbSurfaceSums[lane]+=nbSurfaceSums[lane+stride];nbSurfaceNearest[lane]=min(nbSurfaceNearest[lane],nbSurfaceNearest[lane+stride]);}workgroupBarrier();
 }
 if(member==0u&&owned){
  var value=bulk.x;if(bulk.y>0.0){value=particleSurfaceFinish(q,bulk.x,NBSurfaceGather(nbSurfaceSums[lane],nbSurfaceNearest[lane]));}
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
   var q=centre;q[axis]+=0.5;let depth=bandPhi(q);
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
 * reduction. It uses the same h-scale quadratic kernel as the fine transfer. The 5h particle
 * band provides the full 1.5h footprint at the -2h combination boundary. */
export const narrowBandCoarseTransferWGSL=/* wgsl */`
var<workgroup> coarseSums:array<vec2f,64>;
@compute @workgroup_size(64) fn transferCoarse(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+256u*group.y;if(tile>=UM_T.x*UM_T.y*UM_T.z){return;}
 let axis=group.z;let origin=vec3i(vec3u(tile%UM_T.x,(tile/UM_T.x)%UM_T.y,tile/(UM_T.x*UM_T.y)))*4;
 var anchor=origin;anchor[axis]+=3;var q=vec3f(origin)+2.0;q[axis]+=2.0;
 let depth=bandPhi(q);let blend=select(0.0,1.0,depth>=-2.0);
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
