import { geometricPlaneBoxWGSL } from "../../core/geometric-plane-box.wgsl";
/** Dense vertex phi and fixed receiver stencils; all positions are lattice units. */
export const UNIFORM_VOLUME_ENTRIES = [
  "uvAdvectPhi", "uvRedistancePhi", "uvBuildEdges", "uvSumDonors",
  "uvFallback", "uvNormalizeRows", "uvNormalizeDonors", "uvGather",
  "uvPrepareSharpen", "uvProposeSharpen", "uvLimitSharpen", "uvCommitSharpen", "uvPublish", "uvBeginLiquidBalance", "uvBalanceLiquidRows",
  "uvBalanceLiquidDonors", "uvFinishLiquidBalance",
] as const;
/** The four Sec. 3.5 sweeps that exist in a dense and a 4h work-map variant. */
export const UNIFORM_VOLUME_SHARPEN_ENTRIES = [
  "uvPrepareSharpen", "uvProposeSharpen", "uvLimitSharpen", "uvCommitSharpen",
] as const;
export const UNIFORM_VOLUME_TILE_CLASSIFY_ENTRY = "uvClassifySharpenTiles";
/** E1/E2: seed the 4h classes at the head of the step, then dilate them. */
export const UNIFORM_VOLUME_TWO_LEVEL_ENTRIES = [
  "uvTwoLevelSeed", "uvTwoLevelDilateX", "uvTwoLevelDilateY", "uvTwoLevelDilateZ",
] as const;
/** Words the E1 tables occupy above the N-word donor-sum region, per coarse cell. */
export const UNIFORM_VOLUME_TWO_LEVEL_WORDS_PER_TILE = 6;
/** One counter word above the two ping-pong planes: shell tiles this step. */
export const UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS = 1;
/** Pipeline-overridable constant selecting the tiled sharpening variant. */
export const UNIFORM_VOLUME_TILE_WORK_OVERRIDE = "UV_SHARPEN_TILE_WORK";
/** Words 2N+0..6 of the third conditioning plane stay liquid-balance owned. */
export const UNIFORM_VOLUME_SHARPEN_TILE_COUNT_WORD = 7;
export const UNIFORM_VOLUME_SHARPEN_TILE_MAP_WORD = 8;
export const UNIFORM_VOLUME_EDGE_BYTES = 80;
export const uniformVolumeWGSL = /* wgsl */ `
${geometricPlaneBoxWGSL}
@group(0) @binding(31) var uvPhiIn:texture_3d<f32>;
@group(0) @binding(32) var uvPhiOut:texture_storage_3d<r32float,write>;
struct UVEdges { donor:array<u32,9>, weight:array<f32,9>, padding:vec2f }
@group(0) @binding(33) var<storage,read_write> uvEdges:array<UVEdges>;
// Positive floating point sums avoid fixed-point underflow during balancing.
fn uvAddDonor(index:u32,value:f32){
  if(value==0.0){return;}var old=atomicLoad(&sharpenDeposits[index]);
  loop {let next=bitcast<i32>(bitcast<f32>(old)+value);
    let result=atomicCompareExchangeWeak(&sharpenDeposits[index],old,next);
    if(result.exchanged){break;}old=result.old_value;}
}
fn uvCorner(i:u32)->vec3i{return vec3i(i32(i&1u),i32((i>>1u)&1u),i32((i>>2u)&1u));}
fn uvCell(i:u32)->vec3i{let d=vec3u(dims());return vec3i(vec3u(i%d.x,(i/d.x)%d.y,i/(d.x*d.y)));}
fn uvPhi(position:vec3f)->f32{
  let p=clamp(position,vec3f(0),vec3f(dims()));
  let base=min(vec3i(floor(p)),dims()-vec3i(1));let f=p-vec3f(base);
  var values:array<f32,8>;
  for(var i=0u;i<8u;i++){let o=uvCorner(i);let w=select(vec3f(1)-f,f,o==vec3i(1));
    values[i]=textureLoad(uvPhiIn,base+o,0).x*w.x*w.y*w.z;}
  return d4Sum8(values);
}
fn uvGradient(p:vec3f)->vec3f{
  var g=vec3f(0);for(var a=0u;a<3u;a++){var e=vec3f(0);e[a]=0.25;
    let lo=clamp(p-e,vec3f(0),vec3f(dims()));let hi=clamp(p+e,vec3f(0),vec3f(dims()));
    g[a]=(uvPhi(hi)-uvPhi(lo))/max(hi[a]-lo[a],1e-6);}
  return g;
}
// Walk every crossed half-cell so a long characteristic cannot tunnel through
// a thin voxel wall merely because its endpoint is in open fluid.
fn uvTrace(p:vec3f,dt:f32)->vec3f{
  let h=params.cellGravity.xyz;
  let mid=clamp(p-0.5*dt*sampleVelocity(p)/h,vec3f(0),vec3f(dims()));
  let end=clamp(p-dt*sampleVelocity(mid)/h,vec3f(0),vec3f(dims()));
  let steps=max(1u,u32(ceil(2.0*max(abs(end.x-p.x),max(abs(end.y-p.y),abs(end.z-p.z))))));
  var previous=p;
  for(var s=1u;s<=steps;s++){let q=mix(p,end,f32(s)/f32(steps));
    if(cellOpenFraction(clampCell(vec3i(floor(q))))<=1e-5){return previous;}previous=q;}
  return end;
}
fn uvSourcePhi(p:vec3f,phi:f32)->f32{
  var result=phi;
  if(params.drop.w>0.0){let delta=traceWorld(p)-params.drop.xyz;
    let ball=select(length(delta)-params.drop.w,
      max(length(delta.xy)-params.drop.w,abs(delta.z)-params.dropExtent.x),params.dropExtent.x>0.0);
    result=min(result,ball);}
  let speed=length(params.inflowVelocityLength.xyz)*inflowStrength();
  if(speed>1e-6){let direction=normalize(params.inflowVelocityLength.xyz);
    let delta=traceWorld(p)-params.inflowPositionRadius.xyz;let axial=dot(delta,direction);
    let plug=max(length(delta-axial*direction)-params.inflowPositionRadius.w,
      max(-axial,axial-speed*params.dimsDt.w));result=min(result,plug);}
  return result;
}
// Ambient air swept in by separating MAC wall velocities, in metres.
fn uvReleasedWalls(p:vec3f,advected:f32)->f32{
  var result=advected;let h=params.cellGravity.xyz;
  for(var axis=0u;axis<3u;axis++){
    if(axis==2u&&params.tuning.w>0.5){continue;}
    for(var side=0u;side<2u;side++){
      let upper=side==1u;let inward=select(1.0,-1.0,upper);
      let acceleration=select(0.0,params.cellGravity.w,axis==1u);
      let released=inward*acceleration>0.5*abs(params.cellGravity.w);
      let ambient=axis==1u&&upper&&params.boundary.w>0.5;
      if(!released&&!ambient){continue;}
      let plane=select(0.0,f32(dims()[axis]),upper);
      for(var corner=0u;corner<4u;corner++){
        var probe=p;probe[(axis+1u)%3u]+=select(-1e-4,1e-4,(corner&1u)!=0u);
        probe[(axis+2u)%3u]+=select(-1e-4,1e-4,(corner&2u)!=0u);
        probe[axis]=plane+inward*1e-4;let cell=clampCell(vec3i(floor(probe)));
        let speed=select(boundaryVelocity(cell)[axis],velocity(cell)[axis],upper);
        let away=inward*speed;
        if(away>1e-6){result=max(result,params.dimsDt.w*away-inward*(p[axis]-plane)*h[axis]);}
      }
    }
  }
  return result;
}
// Match adaptive-volume's closed-wall contact continuation. A wall vertex
// cannot acquire arriving liquid by normal backtracing: its normal velocity
// is zero. Trace one cell inside all incident closed planes using old phi.
fn uvClosedWallPhi(p:vec3f,advected:f32)->f32{
  if(params.dimsDt.w<=0.0){return advected;}
  var interior=p;var contact=false;
  for(var axis=0u;axis<3u;axis++){
    for(var side=0u;side<2u;side++){
      let upper=side==1u;let inward=select(1.0,-1.0,upper);
      let plane=select(0.0,f32(dims()[axis]),upper);
      let acceleration=select(0.0,params.cellGravity.w,axis==1u);
      let released=inward*acceleration>0.5*abs(params.cellGravity.w);
      let ambient=axis==1u&&upper&&params.boundary.w>0.5;
      if(abs(p[axis]-plane)>1e-5||released||ambient){continue;}
      interior[axis]+=inward;contact=true;
    }
  }
  if(!contact||uvOpen(clampCell(vec3i(floor(interior))))<=1e-5){return advected;}
  let continued=uvPhi(uvTrace(interior,params.dimsDt.w));
  return select(advected,min(advected,continued),continued<0.0);
}
@compute @workgroup_size(4,4,4)
fn uvAdvectPhi(@builtin(global_invocation_id)gid:vec3u){
  if(any(gid>vec3u(dims()))){return;}let p=vec3f(gid);
  textureStore(uvPhiOut,vec3i(gid),vec4f(uvSourcePhi(p,uvReleasedWalls(p,uvClosedWallPhi(p,uvPhi(uvTrace(p,params.dimsDt.w)))))));
}
@compute @workgroup_size(4,4,4)
fn uvRedistancePhi(@builtin(global_invocation_id)gid:vec3u){
  if(any(gid>vec3u(dims()))){return;}let p=vec3f(gid);let initial=uvPhi(p);
  let h=params.cellGravity.xyz;let band=4.0*max(h.x,max(h.y,h.z));
  var value=initial;
  if(abs(initial)>1e-8&&abs(initial)<band){var q=p;
    for(var i=0u;i<8u;i++){let g=uvGradient(q);let norm=dot(g/h,g/h);if(norm<1e-16){break;}
      q=clamp(q-clamp(uvPhi(q)*g/(h*h*norm),vec3f(-2),vec3f(2)),
        max(vec3f(0),p-vec3f(4)),min(vec3f(dims()),p+vec3f(4)));}
    if(abs(uvPhi(q))<0.005*min(h.x,min(h.y,h.z))){value=sign(initial)*length((p-q)*h);}}
  textureStore(uvPhiOut,vec3i(gid),vec4f(value));
}
fn uvOpen(id:vec3i)->f32{if(!valid(id)){return 0.0;}return cellOpenFraction(id);}
@compute @workgroup_size(4,4,4)
fn uvBuildEdges(@builtin(global_invocation_id)gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}let index=linearIndex(id);
  let departure=uvTrace(vec3f(id)+vec3f(0.5),params.dimsDt.w)-vec3f(0.5);
  let base=vec3i(floor(departure));let f=fract(departure);
  for(var k=0u;k<9u;k++){uvEdges[index].donor[k]=index;uvEdges[index].weight[k]=0.0;}
  for(var k=0u;k<8u;k++){let o=uvCorner(k);let q=base+o;
    let w=select(vec3f(1)-f,f,o==vec3i(1));
    if(valid(q)&&uvOpen(id)>0.0){uvEdges[index].donor[k]=linearIndex(q);
      uvEdges[index].weight[k]=w.x*w.y*w.z*min(uvOpen(id),uvOpen(q));}}
}
@compute @workgroup_size(4,4,4)
fn uvSumDonors(@builtin(global_invocation_id)gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}let index=linearIndex(id);
  for(var k=0u;k<9u;k++){uvAddDonor(uvEdges[index].donor[k],uvEdges[index].weight[k]);}
}
@compute @workgroup_size(4,4,4)
fn uvFallback(@builtin(global_invocation_id)gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}let i=linearIndex(id);
  if(atomicLoad(&sharpenDeposits[i])==0){uvEdges[i].weight[8]=max(uvOpen(id),1e-6);}
}
@compute @workgroup_size(4,4,4)
fn uvNormalizeRows(@builtin(global_invocation_id)gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}let i=linearIndex(id);var sum=0.0;
  for(var k=0u;k<9u;k++){sum+=uvEdges[i].weight[k];}
  let scale=uvOpen(id)/max(sum,1e-20);
  for(var k=0u;k<9u;k++){uvEdges[i].weight[k]*=scale;}
}
@compute @workgroup_size(4,4,4)
fn uvNormalizeDonors(@builtin(global_invocation_id)gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}let i=linearIndex(id);
  for(var k=0u;k<9u;k++){let donor=uvEdges[i].donor[k];
    let sum=bitcast<f32>(atomicLoad(&sharpenDeposits[donor]));
    uvEdges[i].weight[k]/=max(sum,1e-20);}
}
// Same liquid-only receiver cap / donor normalization as adaptive-volume.
// A uniform GPU flag skips corrective work after convergence. Dispatch counts
// are also published for the benchmark-only indirect comparison.
// Words 2N+0..6 of the third conditioning plane are this stage's: the flag,
// the running maximum error, three indirect dispatch dimensions, the executed
// round count and the last error. Word 2N+7 and everything above it belong to
// the 4h sharpening map, which runs after balancing has finished.
@compute @workgroup_size(1)
fn uvBeginLiquidBalance(){let base=2u*cellCount();atomicStore(&sharpenDeposits[base],1);
  atomicStore(&sharpenDeposits[base+1u],0);
  for(var axis=0u;axis<3u;axis++){atomicStore(&sharpenDeposits[base+2u+axis],(dims()[axis]+3)/4);}
  atomicStore(&sharpenDeposits[base+5u],0);atomicStore(&sharpenDeposits[base+6u],0);
}
@compute @workgroup_size(4,4,4)
fn uvBalanceLiquidRows(@builtin(global_invocation_id)gid:vec3u){
  let id=vec3i(gid);if(!valid(id)||atomicLoad(&sharpenDeposits[2u*cellCount()])==0){return;}
  let i=linearIndex(id);var amount=0.0;
  for(var k=0u;k<9u;k++){amount+=uvEdges[i].weight[k]*volume(uvCell(uvEdges[i].donor[k]));}
  let capacity=uvOpen(id);let excess=max(0.0,amount-capacity)/max(capacity,1e-20);
  atomicMax(&sharpenDeposits[2u*cellCount()+1u],bitcast<i32>(excess));
  let scale=select(1.0,capacity/max(amount,1e-20),excess>params.dropExtent.w);
  for(var k=0u;k<9u;k++){uvEdges[i].weight[k]*=scale;uvAddDonor(uvEdges[i].donor[k],uvEdges[i].weight[k]);}
}
@compute @workgroup_size(4,4,4)
fn uvBalanceLiquidDonors(@builtin(global_invocation_id)gid:vec3u){
  let id=vec3i(gid);if(!valid(id)||atomicLoad(&sharpenDeposits[2u*cellCount()])==0){return;}
  let i=linearIndex(id);
  for(var k=0u;k<9u;k++){let donor=uvEdges[i].donor[k];let sum=bitcast<f32>(atomicLoad(&sharpenDeposits[donor]));
    uvEdges[i].weight[k]/=max(sum,1e-30);}
}
@compute @workgroup_size(1)
fn uvFinishLiquidBalance(){
  let base=2u*cellCount();if(atomicLoad(&sharpenDeposits[base])==0){return;}
  let error=atomicLoad(&sharpenDeposits[base+1u]);
  atomicStore(&sharpenDeposits[base+6u],error);
  if(bitcast<f32>(error)<=params.dropExtent.w){
    atomicStore(&sharpenDeposits[base],0);
    for(var axis=0u;axis<3u;axis++){atomicStore(&sharpenDeposits[base+2u+axis],0);}
  }else{atomicAdd(&sharpenDeposits[base+5u],1);}
  atomicStore(&sharpenDeposits[base+1u],0);
}
// Sec. 3.4 and Sec. 3.5 both write V as a sum with cancellation, so a cell the
// characteristic barely reached keeps float32 rounding residue: on figure 7 at
// step 60 that residue is four fifths of the 553k nonzero cells yet 1.3e-7 of
// the mass, and every one of those cells keeps a 4h tile live for any work map.
// The floor is in cell volumes and catches the ULP-scale negatives under the
// same test. A zero threshold never compares true -- NaN included -- so the
// control arm stores the untreated sum bit for bit. Words 5 and 6 of the
// diagnostics buffer price it: cells zeroed, and the discarded mass in
// sixty-fourths of the threshold.
fn uvDustFloor(value:f32)->f32{
  if(value==0.0||!(abs(value)<params.tuning.z)){return value;}
  atomicAdd(&reductions[5],1u);
  atomicAdd(&reductions[6],min(u32(abs(value)/params.tuning.z*64.0),64u));
  return 0.0;
}
@compute @workgroup_size(4,4,4)
fn uvGather(@builtin(global_invocation_id)gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}let i=linearIndex(id);var value=0.0;
  for(var k=0u;k<9u;k++){value+=uvEdges[i].weight[k]*volume(uvCell(uvEdges[i].donor[k]));}
  value+=min(dropSource(id),max(0.0,uvOpen(id)-value));
  if(uvOpen(id)>0.0){value+=inflowSweptPlugSource(id,params.dimsDt.w);}
  textureStore(volumeOut,id,vec4f(uvDustFloor(value)));
  textureStore(gammaOut,id,vec4f(uvTarget(id)));
}
fn uvTarget(id:vec3i)->f32{
  var samples:array<f32,8>;var centre=0.0;var fill=0.0;var magnitude=0.0;
  for(var k=0u;k<8u;k++){let p=vec3f(id)+vec3f(0.25)+0.5*vec3f(uvCorner(k));
    let value=uvPhi(p);samples[k]=value;centre+=0.125*value;
    magnitude=max(magnitude,abs(value));fill+=select(select(0.0,1.0,value<0.0),0.5,value==0.0);}
  var gradient=vec3f(0);
  for(var k=0u;k<8u;k++){gradient+=(2.0*vec3f(uvCorner(k))-vec3f(1))*samples[k]/2.0;}
  var residual=0.0;
  for(var k=0u;k<8u;k++){let sign=2.0*vec3f(uvCorner(k))-vec3f(1);
    residual=max(residual,abs(samples[k]-(centre+dot(gradient,0.25*sign))));}
  let fraction=select(fill/8.0,geometricPlaneBoxFraction(gradient,-centre,vec3f(1)),residual<=1e-4*(1.0+magnitude));
  return fraction*uvOpen(id);
}
// After transport the fixed stencil arena is scratch for face proposals and
// cell budgets: three positive-face fluxes, surplus, need, phi, and two limits.
// 4h work map for the eight sharpening sweeps. Phi is fixed throughout them, so
// one classification pass serves the whole stage. A tile with no cell in the
// admission band has identically zero need and surplus, even if it contains
// nonzero V; every flux touching it is therefore zero. Prepare/propose/limit
// skip such a tile, neighbours read its fluxes as zero, and commit copies its V
// across the ping-pong pair. Never treat stale transport-edge scratch as a
// sharpening flux. The map reuses the third conditioning plane, which liquid
// balancing has finished with, behind that plane's eight-word control header.
// The dense control is the same module with the override left false: the
// lookups fold away and the numerics are bit-identical.
override UV_SHARPEN_TILE_WORK:bool=false;
const UV_SHARPEN_TILE_MAP_WORD=8u;
const UV_SHARPEN_TILE_COUNT_WORD=7u;
fn uvSharpenTileIndex(id:vec3i)->u32{
  let d=(vec3u(dims())+vec3u(3))/4u;let t=vec3u(id)/4u;
  return 2u*cellCount()+UV_SHARPEN_TILE_MAP_WORD+t.x+d.x*(t.y+d.y*t.z);
}
fn uvSharpenTileActive(id:vec3i)->bool{
  if(!UV_SHARPEN_TILE_WORK){return true;}
  return atomicLoad(&sharpenDeposits[uvSharpenTileIndex(id)])!=0;
}
var<workgroup> uvTileAdmission:atomic<u32>;
@compute @workgroup_size(4,4,4)
fn uvClassifySharpenTiles(@builtin(global_invocation_id)gid:vec3u,
  @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)tile:vec3u){
  if(lane==0u){atomicStore(&uvTileAdmission,0u);}workgroupBarrier();
  if(valid(vec3i(gid))){
    let phi=uvPhi(vec3f(gid)+vec3f(0.5));
    let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
    // Negated comparison conservatively retains non-finite input as active.
    if(!(abs(phi)>=params.tuning.y*h)){atomicStore(&uvTileAdmission,1u);}
  }
  workgroupBarrier();
  if(lane==0u){let admission=atomicLoad(&uvTileAdmission);
    atomicStore(&sharpenDeposits[uvSharpenTileIndex(vec3i(tile*4u))],i32(admission));
    if(admission!=0u){atomicAdd(&sharpenDeposits[2u*cellCount()+UV_SHARPEN_TILE_COUNT_WORD],1);}}
}
@compute @workgroup_size(4,4,4)
fn uvPrepareSharpen(@builtin(global_invocation_id)gid:vec3u){
  if(!uvSharpenTileActive(vec3i(gid))){return;}
  let id=vec3i(gid);if(!valid(id)){return;}let i=linearIndex(id);
  let phi=uvPhi(vec3f(id)+vec3f(0.5));let desired=textureLoad(gammaIn,id,0).x;
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  let dose=clamp(params.tuning.x,0.0,1.0);let own=volume(id);
  let admitted=uvOpen(id)>0.99999&&abs(phi)<params.tuning.y*h;
  let relay=phi>0.0&&desired<=1e-6;
  uvEdges[i].weight[3]=select(0.0,dose*max(own-desired,0.0),admitted);
  uvEdges[i].weight[4]=select(0.0,dose*max(select(desired,1.0,relay)-own,0.0),admitted);
  uvEdges[i].weight[5]=phi;
}
@compute @workgroup_size(4,4,4)
fn uvProposeSharpen(@builtin(global_invocation_id)gid:vec3u){
  if(!uvSharpenTileActive(vec3i(gid))){return;}
  let id=vec3i(gid);if(!valid(id)){return;}let i=linearIndex(id);
  let phiA=uvEdges[i].weight[5];
  for(var axis=0u;axis<3u;axis++){
    uvEdges[i].weight[axis]=0.0;var e=vec3i(0);e[axis]=1;let q=id+e;
    if(!valid(q)||!uvSharpenTileActive(q)||uvOpen(id)<0.99999||uvOpen(q)<0.99999||faceOpenFraction(id,axis)<0.99999){continue;}
    let j=linearIndex(q);let phiB=uvEdges[j].weight[5];
    let middle=uvPhi(vec3f(id)+vec3f(0.5)+0.5*vec3f(e));let epsilon=1e-6;
    let inwardA=phiA>=0.0&&phiB<phiA-epsilon&&middle<=phiA+epsilon&&middle>=phiB-epsilon;
    let inwardB=phiB>=0.0&&phiA<phiB-epsilon&&middle<=phiB+epsilon&&middle>=phiA-epsilon;
    let relayA=phiA>0.0&&textureLoad(gammaIn,id,0).x<=1e-6;
    let relayB=phiB>0.0&&textureLoad(gammaIn,q,0).x<=1e-6;
    let ab=select(0.0,min(uvEdges[i].weight[3],uvEdges[j].weight[4]),(middle<=epsilon&&!relayB)||inwardA);
    let ba=select(0.0,min(uvEdges[j].weight[3],uvEdges[i].weight[4]),(middle<=epsilon&&!relayA)||inwardB);
    uvEdges[i].weight[axis]=ab-ba;
  }
}
@compute @workgroup_size(4,4,4)
fn uvLimitSharpen(@builtin(global_invocation_id)gid:vec3u){
  if(!uvSharpenTileActive(vec3i(gid))){return;}
  let id=vec3i(gid);if(!valid(id)){return;}let i=linearIndex(id);var outgoing=0.0;var incoming=0.0;
  for(var axis=0u;axis<3u;axis++){var e=vec3i(0);e[axis]=1;
    let positive=uvEdges[i].weight[axis];var negative=0.0;
    if(valid(id-e)&&uvSharpenTileActive(id-e)){negative=uvEdges[linearIndex(id-e)].weight[axis];}
    outgoing+=max(positive,0.0)+max(-negative,0.0);incoming+=max(-positive,0.0)+max(negative,0.0);}
  uvEdges[i].padding=vec2f(min(1.0,uvEdges[i].weight[3]/max(outgoing,1e-20)),
    min(1.0,uvEdges[i].weight[4]/max(incoming,1e-20)));
}
fn uvLimitedFlux(i:u32,j:u32,axis:u32)->f32{
  if(!uvSharpenTileActive(uvCell(i))||!uvSharpenTileActive(uvCell(j))){return 0.0;}
  let raw=uvEdges[i].weight[axis];let a=uvEdges[i].padding;let b=uvEdges[j].padding;
  return raw*select(min(a.y,b.x),min(a.x,b.y),raw>=0.0);
}
@compute @workgroup_size(4,4,4)
fn uvCommitSharpen(@builtin(global_invocation_id)gid:vec3u){
  if(valid(vec3i(gid))&&!uvSharpenTileActive(vec3i(gid))){
    textureStore(volumeOut,vec3i(gid),vec4f(uvDustFloor(volume(vec3i(gid)))));return;
  }
  let id=vec3i(gid);if(!valid(id)){return;}let i=linearIndex(id);var terms:array<f32,6>;
  for(var axis=0u;axis<3u;axis++){var e=vec3i(0);e[axis]=1;terms[2u*axis]=0.0;terms[2u*axis+1u]=0.0;
    if(valid(id+e)){terms[2u*axis]=-uvLimitedFlux(i,linearIndex(id+e),axis);}
    if(valid(id-e)){terms[2u*axis+1u]=uvLimitedFlux(linearIndex(id-e),i,axis);}}
  textureStore(volumeOut,id,vec4f(uvDustFloor(volume(id)+d4Sum6(terms))));
}
// The two-level velocity sampler and the tile classes the shrunk velocity
// extension runs on: one classification per step on the ceil(n/4)^3 tile grid,
// encoded at the HEAD of the step from start-of-step V and phi, plus a 4h face
// table filled from the extension hierarchy's own ceil(n/4) level once the
// extension has run. Both live in words [N,2N) of the conditioning plane -- a
// region the geometric path never addresses, and whose per-step clears are
// ranged away from it so this survives the whole step. params.physical.z is the
// Chebyshev fine reach k, or -1 with the experiment off, in which case the
// sampler branch is never taken and the fine path stores the same bits.
// Class word bits: 1 = FINE (samples the finest lattice), 2 = SHELL (FINE
// dilated by params.twoLevel.x, the set on which fine extension output must be
// valid). SHELL is read by the extrapolator module, not here.
fn uvCoarseDims()->vec3i{return (dims()+vec3i(3))/4;}
fn uvCoarseCount()->u32{let c=uvCoarseDims();return u32(c.x*c.y*c.z);}
fn uvCoarseIndex(t:vec3i)->u32{let c=uvCoarseDims();return u32(t.x+c.x*(t.y+c.y*t.z));}
fn uvCoarseBase()->u32{return cellCount();}
fn uvCoarsePlane(plane:u32)->u32{return cellCount()+4u*uvCoarseCount()+plane*uvCoarseCount();}
fn uvTwoLevelFineAt(p:vec3f)->bool{
  let cell=clamp(vec3i(floor(p)),vec3i(0),dims()-vec3i(1));
  return (atomicLoad(&sharpenDeposits[uvCoarseBase()+4u*uvCoarseIndex(cell/4)+3u])&1)!=0;
}
fn uvCoarseFace(t:vec3i,component:u32)->f32{
  if(any(t<vec3i(0))||any(t>=uvCoarseDims())){return 0.0;}
  return bitcast<f32>(atomicLoad(&sharpenDeposits[uvCoarseBase()+4u*uvCoarseIndex(t)+component]));
}
// The fine path's padded shell returns zero outside the lattice; the explicit
// range test is that same rule on the 4h lattice, so a characteristic leaving
// the domain meets the same closed wall at either level. A fine cell index maps
// to the coarse lattice by a quarter, which carries the upper-face convention
// with it: fine face 4t+3 is coarse face t. The table itself is written by the
// extrapolator from its own ceil(n/4) hierarchy level, which is a complete 4h
// MAC field in that same convention and -- unlike a restriction of the fine
// transport shell -- survives shrinking the fine extension.
fn uvCoarseVelocityComponent(p:vec3f,component:u32)->f32{
  let cd=uvCoarseDims();
  var offset=vec3f(0.5);offset[component]=1.0;var lower=vec3f(0.0);lower[component]=-1.0;
  let q=clamp(0.25*p-offset,lower,vec3f(cd-vec3i(1)));
  let base=vec3i(floor(q));let fraction=fract(q);var terms:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let weights=select(vec3f(1.0)-fraction,fraction,vec3f(o)>vec3f(0.5));
    terms[corner]=weights.x*weights.y*weights.z*uvCoarseFace(base+o,component);
  }
  return d4Sum8(terms);
}
// SEED: liquid above the dust floor, any solid or terrain share, a source this
// step, or a vertex on the liquid side of the 4h band. Partial open fraction
// covers rigid bodies and terrain, so neither needs the toggle forced off. The
// vertex test is ONE-SIDED (phi < band, not |phi| < band) so that every cell
// with a negative centre phi is in a seed tile: the centre is the mean of its
// eight vertices, so a negative centre forces a negative -- hence in-band --
// vertex. That is what makes "the FIM accurate band lies inside SHELL" a
// theorem rather than a property of the V test. It is a superset of E1's
// two-sided test; deep liquid is already seeded by V.
@compute @workgroup_size(4,4,4)
fn uvTwoLevelSeed(@builtin(global_invocation_id)gid:vec3u){
  let t=vec3i(gid);if(any(t>=uvCoarseDims())){return;}
  let slot=uvCoarseBase()+4u*uvCoarseIndex(t);
  let dust=select(params.tuning.z,1e-6,params.tuning.z<=0.0);
  var seed=false;
  for(var z=0;z<4;z++){for(var y=0;y<4;y++){for(var x=0;x<4;x++){
    let id=4*t+vec3i(x,y,z);if(!valid(id)){continue;}
    if(abs(volume(id))>dust||uvOpen(id)<0.99999){seed=true;}
    if(dropSource(id)>0.0||inflowSweptPlugSource(id,params.dimsDt.w)>0.0){seed=true;}}}}
  let h=params.cellGravity.xyz;let band=4.0*max(h.x,max(h.y,h.z));
  let last=min(4*t+vec3i(4),dims());
  for(var z=4*t.z;z<=last.z;z++){for(var y=4*t.y;y<=last.y;y++){for(var x=4*t.x;x<=last.x;x++){
    if(textureLoad(uvPhiIn,vec3i(x,y,z),0).x<band){seed=true;}}}}
  atomicStore(&sharpenDeposits[slot+3u],select(0,1,seed));
}
// Chebyshev dilation, separated into three axis scans. Each scan carries two
// independent radii in two bits: FINE at k tiles and SHELL at k+s. Chebyshev
// balls compose, so SHELL is exactly FINE dilated by s. The pair of single-word
// planes above the table is the ping-pong; the z scan lands the final class
// back in the table so the sampler reads one place.
fn uvTwoLevelFineReach()->i32{return i32(max(params.physical.z,0.0));}
fn uvTwoLevelShellReach()->i32{return uvTwoLevelFineReach()+i32(max(params.twoLevel.x,1.0));}
/** Both class bits of tile q on the pass's input plane; the seed plane is one bit. */
fn uvTwoLevelClassIn(plane:u32,q:vec3i)->i32{
  if(plane==2u){return select(0,3,atomicLoad(&sharpenDeposits[uvCoarseBase()+4u*uvCoarseIndex(q)+3u])!=0);}
  return atomicLoad(&sharpenDeposits[uvCoarsePlane(plane)+uvCoarseIndex(q)]);
}
fn uvTwoLevelDilate(previous:u32,axis:u32,t:vec3i)->i32{
  let c=uvCoarseDims();let k=uvTwoLevelFineReach();let s=uvTwoLevelShellReach();var hit=0;
  for(var d=-s;d<=s;d++){var q=t;q[axis]+=d;if(q[axis]<0||q[axis]>=c[axis]){continue;}
    let value=uvTwoLevelClassIn(previous,q);if(value==0){continue;}
    if((value&1)!=0&&d>=-k&&d<=k){hit|=1;}
    if((value&2)!=0){hit|=2;}}
  return hit;
}
@compute @workgroup_size(4,4,4)
fn uvTwoLevelDilateX(@builtin(global_invocation_id)gid:vec3u){
  let t=vec3i(gid);if(any(t>=uvCoarseDims())){return;}
  atomicStore(&sharpenDeposits[uvCoarsePlane(0u)+uvCoarseIndex(t)],uvTwoLevelDilate(2u,0u,t));
}
@compute @workgroup_size(4,4,4)
fn uvTwoLevelDilateY(@builtin(global_invocation_id)gid:vec3u){
  let t=vec3i(gid);if(any(t>=uvCoarseDims())){return;}
  atomicStore(&sharpenDeposits[uvCoarsePlane(1u)+uvCoarseIndex(t)],uvTwoLevelDilate(0u,1u,t));
}
@compute @workgroup_size(4,4,4)
fn uvTwoLevelDilateZ(@builtin(global_invocation_id)gid:vec3u){
  let t=vec3i(gid);if(any(t>=uvCoarseDims())){return;}
  let hit=uvTwoLevelDilate(1u,2u,t);
  atomicStore(&sharpenDeposits[uvCoarseBase()+4u*uvCoarseIndex(t)+3u],hit);
  if((hit&1)!=0){atomicAdd(&reductions[7],1u);}
  if((hit&2)!=0){atomicAdd(&sharpenDeposits[uvCoarsePlane(2u)],1);}
}
@compute @workgroup_size(4,4,4)
fn uvPublish(@builtin(global_invocation_id)gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  textureStore(volumeOut,id,vec4f(0.5-uvPhi(vec3f(id)+vec3f(0.5))/h));
  textureStore(gammaOut,id,vec4f(uvOpen(id)));
}
`;
