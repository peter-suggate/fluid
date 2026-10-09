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
 if(nbAdaptive()&&nbTheta(vec3f(q))<=0.0){return vec2f(bandPhi(vec3f(q)),0.0);}
 let bulk=bulkDepth(vec3f(q));let tile=umTileAt(min(vec3u(q)/4u,UM_T-1u));
 // Deeper than this the shrunk level set wins the union whatever the samples
 // are; nothing bounds how far into the air they may carry the surface.
 let gather=atomicLoad(&bins[NB_SURFACE_TILES+tile])!=0u&&bulk>-2.0&&nbTheta(vec3f(q))>0.0;
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
 // The eight incident cells are four x rows in sorted particle storage.
 // Split only at a tile boundary, keeping the original z/y/x visit order.
 for(var row=0u;row<4u;row++){
  let y=q.y-1+i32(row%2u);let z=q.z-1+i32(row/2u);
  if(y<0||z<0||y>=i32(UM_D.y)||z>=i32(UM_D.z)){continue;}
  var start=max(q.x-1,0);let last=min(q.x,i32(UM_D.x)-1);
  while(start<=last){
   let end=min(last,start|3);let run=nbRun(vec3i(start,y,z),vec3i(end,y,z));
   for(var index=run.x;index<run.y;index++){let offset=x-nbPosition(index);nearest2=min(nearest2,dot(offset,offset));}
   start=end+1;
  }
 }
 if(nearest2<=1.0){return nearest2;}
 for(var bin=0u;bin<64u;bin++){
  let o=vec3i(i32(bin%4u),i32((bin/4u)%4u),i32(bin/16u))-2;
  let gap=max(o,-o-1);if(all(gap==vec3i(0))||f32(dot(gap,gap))>=nearest2){continue;}
  let cell=q+o;if(any(cell<vec3i(0))||any(cell>=vec3i(UM_D))){continue;}
  let run=nbCellRun(cell);
  for(var index=run.x;index<run.y;index++){let offset=x-nbPosition(index);nearest2=min(nearest2,dot(offset,offset));}
 }
 return nearest2;
}
fn particleSurfaceFinish(q:vec3i,bulk:f32,nearest2:f32)->f32{
 // An empty gather has no particle surface to blend; never erode a cold
 // or newly activated region before its surface samples exist.
 if(nbAdaptive()&&nearest2>=4.0){return nbSourcePhi(vec3f(q),bulk);}
 let particle=min(bulk+1.0,sqrt(nearest2)-NB_SURFACE_RADIUS);
 // A transported particle may add liquid throughout its sphere support.
 // Erasure needs the complete seeded footprint, inside the overlap collar.
 let theta=select(nbTheta(vec3f(q)),nbSurfaceTheta(vec3f(q)),particle>bulk);
 return nbSourcePhi(vec3f(q),mix(bulk,particle,theta));
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

/** Reconstruction first scatters occupied-cell distances into the expired
 * metric bank, then each canonical vertex reads one nearest-distance word. */
export const narrowBandSurfaceWGSL=narrowBandParticleSurfaceWGSL+
 narrowBandTiledSurfaceWGSL.replace("particleSurfaceNearest(q)","nbCachedSurfaceNearest(q)")+/* wgsl */`
// The compact motion arena is dead until classify, after reconstruction.
fn nbSurfaceCells()->u32{return NB_CELLS+4u*arrayLength(&particles);}
@compute @workgroup_size(64) fn surfaceCells(@builtin(global_invocation_id) gid:vec3u){
 for(var cell=gid.x;cell<NB_CELLS;cell+=65536u){
  if(nbSparseBins&&atomicLoad(&bins[NB_COVERAGE+cell/64u])==0u){continue;}
  if(atomicLoad(&bins[cell])==0u){continue;}
  let slot=atomicAdd(&bins[NB_BAND+3u],1u);links[nbSurfaceCells()+slot]=cell;
 }
}
@compute @workgroup_size(64) fn surfaceSplat(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let cell=links[nbSurfaceCells()+group.x];let c=4u*umTileCoord(cell/64u)+umCorner(cell%64u,4u);
 let q=vec3i(c)+vec3i(umCorner(lane,4u))-1;
 if(any(q<vec3i(0))||any(q>vec3i(UM_D))){return;}
 let start=links[cell];let end=start+atomicLoad(&bins[cell]);var nearest2=4.0;
 for(var i=start;i<end;i++){let d=vec3f(q)-nbPosition(i);nearest2=min(nearest2,dot(d,d));}
 // Positive float bit order: complement reverses it; zero is the empty key.
 if(nearest2<4.0){atomicMax(&bins[NB_DEPTH_A+nbVertexIndex(vec3u(q))],~bitcast<u32>(nearest2));}
}
fn nbCachedSurfaceNearest(q:vec3i)->f32{
 let key=atomicLoad(&bins[NB_DEPTH_A+nbVertexIndex(vec3u(q))]);
 if(key==0u){return 4.0;}return bitcast<f32>(~key);
}
@compute @workgroup_size(64) fn couple(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(vec3u(umCounts.x*64u+gid.x,0,0));if(owner.width!=4u){return;}
 let origin=umOrigin(owner);let regular=umTileMaximumWidth(owner.tile)==4u&&umTileMinimumWidth(owner.tile)==4u;
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let q=origin+4u*corner;
  if(regular){if(!all((corner!=vec3u(0))|(origin==vec3u(0)))){continue;}}
  else if(umVertexAuthority(q).index!=owner.index){continue;}
  let bulk=particleSurfaceBulk(vec3i(q));var value=bulk.x;
  if(bulk.y>0.0){value=particleSurfaceFinish(vec3i(q),bulk.x,nbCachedSurfaceNearest(vec3i(q)));}
  textureStore(outputPhi,vec3i(q),vec4f(value*min(params.hDt.x,min(params.hDt.y,params.hDt.z))));
 }
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

/** Each fine tile prepares its face blend once, sharing the 5³ metric vertices.
 * The expired scatter cursors fit six words per fine cell at sparse layouts. */
export function narrowBandTransferPropertiesSource(subgroups=false):string{return /* wgsl */`
fn nbTransferPropertiesFit()->bool{return 416u*umCounts.x<=NB_CELLS;}
var<workgroup> nbTransferDepth:array<f32,125>;
var<workgroup> nbTransferActive:array<u32,64>;
var<workgroup> nbTransferHeat:array<f32,27>;
fn nbTransferTheta(local:vec3f)->f32{
 if(!nbAdaptive()){return 1.0;}
 let q=local/4.0+0.5;let base=vec3u(floor(q));let f=fract(q);var theta=0.0;
 for(var k=0u;k<8u;k++){let bit=umCorner(k,2u);let at=base+bit;let w=select(1.0-f,f,bit!=vec3u(0));theta+=w.x*w.y*w.z*nbTransferHeat[at.x+3u*(at.y+3u*at.z)];}
 return clamp(theta,0.0,1.0);
}
@compute @workgroup_size(64) fn transferProperties(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32${subgroups?",@builtin(subgroup_invocation_id) subgroupLane:u32":""}){
 if(!nbTransferPropertiesFit()){return;}
 let owner=umAllOwner(vec3u(group.x*64u,0u,0u));let origin=4u*umTileCoord(owner.tile);
 for(var k=lane;k<125u;k+=64u){let q=origin+umCorner(k,5u);nbTransferDepth[k]=bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+nbVertexIndex(q)]));}
 if(lane<27u&&nbAdaptive()){
  let t=vec3u(clamp(vec3i(umTileCoord(owner.tile))+vec3i(umCorner(lane,3u))-1,vec3i(0),vec3i(UM_T)-1));
  nbTransferHeat[lane]=clamp(bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_THETA+umTileAt(t)])),0.0,1.0);
 }
 workgroupBarrier();
 let local=umCorner(lane,4u);let cell=origin+local;let at=NB_CELLS+6u*(64u*group.x+lane);var needed=false;var properties:array<vec2f,3>;
 for(var axis=0u;axis<3u;axis++){
  var near=0.0;var expected=1.0;var low=local;low[axis]+=1u;
  if(cell[axis]+1u<UM_D[axis]&&(nbBandReach(origin+low)&1u)!=0u){
   var depth=0.0;
   // The positive MAC face has fraction zero along its normal. Preserve the
   // eight-corner accumulation order while skipping its four zero terms.
   for(var k=0u;k<8u;k++){let bit=umCorner(k,2u);if(bit[axis]!=0u){continue;}let q=low+bit;depth+=0.25*nbTransferDepth[q.x+5u*(q.y+5u*q.z)];}
   if(depth<=1.5&&depth>=-2.0){var q=vec3f(local)+0.5;q[axis]+=0.5;near=nbTransferTheta(q);expected=max(1.0,8.0*clamp(0.5-depth,0.0,1.0));}
  }
  needed=needed||near>0.0;
  properties[axis]=vec2f(near,expected);
 }
 nbTransferActive[lane]=u32(needed);workgroupBarrier();
 let neededPair=(nbTransferActive[lane&~1u]|nbTransferActive[lane|1u])!=0u;
 let emit=neededPair&&lane%2u==0u;
 ${subgroups?`// Reserve each subgroup's tile-local pairs together, retaining locality
 // without relying on a mapping between local and subgroup lane indices.
 let rank=subgroupExclusiveAdd(u32(emit));let count=subgroupAdd(u32(emit));var first=0u;
 if(subgroupLane==0u&&count!=0u){first=atomicAdd(&state[9],count);}first=subgroupBroadcastFirst(first);
 if(emit){atomicStore(&bins[NB_CELLS+384u*umCounts.x+first+rank],32u*group.x+lane/2u);}`:`
 if(emit){let slot=atomicAdd(&state[9],1u);atomicStore(&bins[NB_CELLS+384u*umCounts.x+slot],32u*group.x+lane/2u);}`}
 if(!neededPair){textureStore(output,vec3i(cell),textureLoad(velocity,vec3i(cell),0));return;}
 for(var axis=0u;axis<3u;axis++){atomicStore(&bins[at+2u*axis],bitcast<u32>(properties[axis].x));atomicStore(&bins[at+2u*axis+1u],bitcast<u32>(properties[axis].y));}
}
`;}
export const narrowBandTransferPropertiesWGSL=narrowBandTransferPropertiesSource();

/** Adjacent fine cells share their particle reads. Each sixteen-lane team
 * gathers the union of two neighboring x cells' quadratic MAC supports and
 * accumulates both results in registers. Spatial order makes each x row at
 * most two runs, without a per-bin particle limit. */
export function narrowBandFineTransferSource(subgroups=false,prepared=false):string{return /* wgsl */`
var<workgroup> nbMomentum:array<vec3f,64>;
var<workgroup> nbMass:array<vec3f,64>;
var<workgroup> nbMomentumNext:array<vec3f,64>;
var<workgroup> nbMassNext:array<vec3f,64>;
var<workgroup> nbBlend:array<f32,24>;
var<workgroup> nbExpected:array<f32,24>;
var<workgroup> nbRuns:array<vec2u,128>;
@compute @workgroup_size(${subgroups?32:64}) fn transfer(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) ${subgroups?"localLane:u32,@builtin(subgroup_invocation_id) subgroupLane:u32":"lane:u32"}){
 ${subgroups?/* wgsl */`
 // The 32-lane workgroup is one subgroup (the adapter minimum is 32).
 // Assign logical lanes directly, independent of local invocation order.
 let lane=subgroupLane;
 `:""}
 let pair=(gid.x/${subgroups?32:64}u)*${subgroups?2:4}u+lane/16u;var first=2u*pair;var valid=true;
 ${prepared?`if(nbTransferPropertiesFit()){valid=pair<atomicLoad(&state[9]);first=2u*atomicLoad(&bins[NB_CELLS+384u*umCounts.x+pair]);}`:""}
 let owner=umAllOwner(vec3u(first,0,0));let origin=umOrigin(owner);
 let next=umAllOwner(vec3u(first+1u,0,0));let nextOrigin=umOrigin(next);
 let team=lane/16u;let member=lane%16u;let centre=vec3f(origin)+0.5;
 if(member<6u){
  let axis=member%3u;let o=select(origin,nextOrigin,member>=3u);let width=select(owner.width,next.width,member>=3u);
  var near=0.0;var expected=1.0;
  ${prepared?`if(nbTransferPropertiesFit()){
   let at=NB_CELLS+6u*(first+u32(member>=3u))+2u*axis;
   near=select(0.0,bitcast<f32>(atomicLoad(&bins[at])),valid);expected=bitcast<f32>(atomicLoad(&bins[at+1u]));
  }else `:""}if(width==1u&&o[axis]+1u<UM_D[axis]){
   var q=vec3f(o)+0.5;q[axis]+=0.5;let depth=particleDepth(q);
   if(depth<=1.5&&depth>=-2.0){near=nbTheta(q);expected=max(1.0,8.0*clamp(0.5-depth,0.0,1.0));}
  }
  nbBlend[6u*team+member]=near;nbExpected[6u*team+member]=expected;
 }
 workgroupBarrier();
 let blend=vec3f(nbBlend[6u*team],nbBlend[6u*team+1u],nbBlend[6u*team+2u]);
 let nextBlend=vec3f(nbBlend[6u*team+3u],nbBlend[6u*team+4u],nbBlend[6u*team+5u]);
 let gather=any(blend>vec3f(0))||any(nextBlend>vec3f(0));
 if(gather){
  let low=vec3i(origin)-1;let row=member;
  let y=low.y+i32(row%4u);let z=low.z+i32(row/4u);var runs=array<vec2u,2>();
  if(row<15u&&y>=0&&y<i32(UM_D.y)&&z>=0&&z<i32(UM_D.z)){
   let last=min(low.x+select(4,3,row%4u==3u||row>=12u),i32(UM_D.x)-1);
   var x=max(low.x,0);
   for(var part=0u;x<=last;part++){let stop=min(last,x|3);runs[part]=nbRun(vec3i(x,y,z),vec3i(stop,y,z));x=stop+1;}
  }
  nbRuns[32u*team+2u*row]=runs[0];nbRuns[32u*team+2u*row+1u]=runs[1];
 }
 workgroupBarrier();
 var momentum=vec3f(0);var total=vec3f(0);var momentumNext=vec3f(0);var totalNext=vec3f(0);
 if(gather){
  // Both parts of an x row form one logical stream. A partial run no
  // longer leaves the other lanes idle before the second part starts.
  for(var slot=0u;slot<30u;slot+=2u){
   let run=nbRuns[32u*team+slot];let second=nbRuns[32u*team+slot+1u];let count=run.y-run.x;
   let length=count+second.y-second.x;
   for(var index=member;index<length;index+=16u){
    let i=select(second.x+index-count,run.x+index,index<count);
    let motion=nbMotion(i);if(motion.w==1.0){continue;}
    let d=centre-nbPosition(i);let f=d+0.5;
    let wc=vec3f(weight(d.x),weight(d.y),weight(d.z));let wf=vec3f(weight(f.x),weight(f.y),weight(f.z));
    let w=vec3f(wf.x*wc.y*wc.z,wc.x*wf.y*wc.z,wc.x*wc.y*wf.z);
    total+=w;momentum+=w*motion.xyz;
    let xc=weight(d.x+1.0);let xf=weight(f.x+1.0);
    let wn=vec3f(xf*wc.y*wc.z,xc*wf.y*wc.z,xc*wc.y*wf.z);
    totalNext+=wn;momentumNext+=wn*motion.xyz;
   }
  }
 }
 ${subgroups?/* wgsl */`
 // A subgroup holds whole 16-lane teams. Keep the original reduction tree,
 // exchanging registers instead of writing four shared arrays each round.
 for(var stride=8u;stride>0u;stride/=2u){
  let m=subgroupShuffleDown(momentum,stride);let w=subgroupShuffleDown(total,stride);
  let mn=subgroupShuffleDown(momentumNext,stride);let wn=subgroupShuffleDown(totalNext,stride);
  if(member<stride){momentum+=m;total+=w;momentumNext+=mn;totalNext+=wn;}
 }
 `:/* wgsl */`
 nbMomentum[lane]=momentum;nbMass[lane]=total;nbMomentumNext[lane]=momentumNext;nbMassNext[lane]=totalNext;workgroupBarrier();
 for(var stride=8u;stride>0u;stride/=2u){
  if(member<stride){nbMomentum[lane]+=nbMomentum[lane+stride];nbMass[lane]+=nbMass[lane+stride];nbMomentumNext[lane]+=nbMomentumNext[lane+stride];nbMassNext[lane]+=nbMassNext[lane+stride];}workgroupBarrier();
 }
 momentum=nbMomentum[lane];total=nbMass[lane];momentumNext=nbMomentumNext[lane];totalNext=nbMassNext[lane];
 `}
 if(valid&&member==0u&&owner.width==1u){
  let original=textureLoad(velocity,vec3i(origin),0);let mass=total;
  var transition=blend;
  if(nbAdaptive()){for(var a=0u;a<3u;a++){transition[a]*=mix(min(1.0,mass[a]/nbExpected[6u*team+a]),1.0,blend[a]);}}
  let value=mix(original.xyz,momentum/max(mass,vec3f(1e-30)),select(vec3f(0),transition,mass>=vec3f(1e-5)));
  textureStore(output,vec3i(origin),vec4f(value,original.w));
 }
 if(valid&&member==0u&&next.width==1u){
  let original=textureLoad(velocity,vec3i(nextOrigin),0);let mass=totalNext;
  var transition=nextBlend;
  if(nbAdaptive()){for(var a=0u;a<3u;a++){transition[a]*=mix(min(1.0,mass[a]/nbExpected[6u*team+3u+a]),1.0,nextBlend[a]);}}
  let value=mix(original.xyz,momentumNext/max(mass,vec3f(1e-30)),select(vec3f(0),transition,mass>=vec3f(1e-5)));
  textureStore(output,vec3i(nextOrigin),vec4f(value,original.w));
 }
}
`;

}
/** Portable reference variant; small subgroups also use its shared tree. */
export const narrowBandFineTransferWGSL=narrowBandFineTransferSource();

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
 if(lane==0u){var value=original;let result=coarseSums[0];if(result.y>1e-5){value[axis]=mix(original[axis],result.x/result.y,blend*nbTransferBlend(q,depth,result.y));}textureStore(output,anchor,value);}
}
`;
