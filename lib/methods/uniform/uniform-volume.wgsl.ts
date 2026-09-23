import { uniformVolumeDonorSumWGSL } from "./uniform-volume-donor-sum.wgsl";
import { uniformAbOn } from "./uniform-ab-switch";
import { geometricPlaneBoxWGSL } from "../../core/geometric-plane-box.wgsl";
/** Dense vertex phi and fixed receiver stencils; all positions are lattice units. */
export const UNIFORM_VOLUME_ENTRIES = [
  "uvAdvectPhi", "uvRedistancePhi", "uvBuildEdges",
  "uvFinishDonorSums", "uvFallback", "uvNormalizeRows", "uvNormalizeDonors", "uvGather",
  "uvPrepareSharpen", "uvProposeSharpen", "uvLimitSharpen", "uvCommitSharpen", "uvPublish",
  "uvCacheSharpenCells", "uvCacheSharpenFaces",
  "uvBalanceMeasure", "uvBalanceReduce", "uvBalanceReduceChunks", "uvTwoLevelSeedCooperative",
  "uvAgreementResidual", "uvCorrectionCapacity", "uvCorrectionTargets",
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
/**
 * E7: the four passes that decide TRANSPORT from the post-extension field, run
 * between the extension and the first transport pass. Resident only while the
 * experiment is compiled in; never encoded while its host gate is off.
 */
export const UNIFORM_VOLUME_TRANSPORT_REACH_ENTRIES = [
  "uvTransportReachMeasure", "uvTransportReachX", "uvTransportReachY", "uvTransportReachZ",
] as const;
export const UNIFORM_VOLUME_TRANSPORT_REACH_COMPILED = uniformAbOn("tilereach");
/** Words the E1 tables occupy above the N-word donor-sum region, per coarse cell. */
export const UNIFORM_VOLUME_TWO_LEVEL_WORDS_PER_TILE = 6;
/**
 * Counter words above the two ping-pong planes, all cleared at the head of the
 * step: shell tiles, transport (E3 live-set) tiles, and the largest backward
 * displacement in cells any cell's start-of-step velocity can produce, which is
 * what the transport predicate's required reach is derived from.
 */
export const UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS = 3;
export const UNIFORM_VOLUME_TWO_LEVEL_SHELL_COUNT_WORD = 0;
export const UNIFORM_VOLUME_TWO_LEVEL_TRANSPORT_COUNT_WORD = 1;
export const UNIFORM_VOLUME_TWO_LEVEL_DISPLACEMENT_WORD = 2;
/** Pipeline-overridable constant selecting the tiled sharpening variant. */
export const UNIFORM_VOLUME_TILE_WORK_OVERRIDE = "UV_SHARPEN_TILE_WORK";
/** The first seven words remain reserved for work counters and layout stability. */
export const UNIFORM_VOLUME_SHARPEN_TILE_COUNT_WORD = 7;
export const UNIFORM_VOLUME_SHARPEN_TILE_MAP_WORD = 8;
export const UNIFORM_VOLUME_EDGE_BYTES = 40;
const donorTiles = uniformAbOn("donortiles");
/** E4: the half-cell solid walk and the embedded-wall terms, skipped as a
 * host-uniform condition when the scene has no cut cell anywhere. */
const solidFreeTrace = uniformAbOn("solidfreetrace");
/** E5: the far-air arm of the vertex phi advect (outside SHELL). */
const phiLean = uniformAbOn("philean");
/** E7: E3's transport reach measured per tile instead of domain-wide. */
const tileReach = uniformAbOn("tilereach");
/** Same reconstruction; cached mode reuses the cell's eight vertex loads. */
export function uniformVolumeTargetWGSL(cached: boolean): string {
  return /* wgsl */ `fn uvTarget(id:vec3i)->f32{
  var samples:array<f32,8>;var centre=0.0;var fill=0.0;var magnitude=0.0;
  ${cached ? `// All eight quarter-cell probes interpolate the same eight vertices.
  // uvTarget is called only for valid cells, so each probe's clamped base is
  // exactly id and its fractions are exactly 1/4 or 3/4 in binary FP32.
  var vertices:array<f32,8>;
  for(var j=0u;j<8u;j++){vertices[j]=textureLoad(uvPhiIn,id+uvCorner(j),0).x;}
  for(var k=0u;k<8u;k++){
    let f=vec3f(0.25)+0.5*vec3f(uvCorner(k));var weighted:array<f32,8>;
    for(var j=0u;j<8u;j++){let w=select(vec3f(1)-f,f,uvCorner(j)==vec3i(1));
      weighted[j]=vertices[j]*w.x*w.y*w.z;}
    let value=d4Sum8(weighted);` : `for(var k=0u;k<8u;k++){let p=vec3f(id)+vec3f(0.25)+0.5*vec3f(uvCorner(k));
    let value=uvPhi(p);`}
    samples[k]=value;centre+=0.125*value;
    magnitude=max(magnitude,abs(value));fill+=select(select(0.0,1.0,value<0.0),0.5,value==0.0);}
  var gradient=vec3f(0);
  for(var k=0u;k<8u;k++){gradient+=(2.0*vec3f(uvCorner(k))-vec3f(1))*samples[k]/2.0;}
  var residual=0.0;
  for(var k=0u;k<8u;k++){let sign=2.0*vec3f(uvCorner(k))-vec3f(1);
    residual=max(residual,abs(samples[k]-(centre+dot(gradient,0.25*sign))));}
  let fraction=select(fill/8.0,geometricPlaneBoxFraction(gradient,-centre,vec3f(1)),residual<=1e-4*(1.0+magnitude));
  return fraction*uvOpen(id);
}`;
}

export const uniformVolumeWGSL = /* wgsl */ `
${geometricPlaneBoxWGSL}
@group(0) @binding(31) var uvPhiIn:texture_3d<f32>;
@group(0) @binding(32) var uvPhiOut:texture_storage_3d<r32float,write>;
// Eight interpolation donors are corners of one cell; the ninth is the
// receiver itself. Store the base index instead of nine redundant indices.
// After gather, base holds three 5-bit sharpening face flags plus one open
// bit, and weights 7/8 hold the two limiter factors. No float is quantized.
struct UVEdges { base:u32, weight:array<f32,9> }
@group(0) @binding(33) var<storage,read_write> uvEdges:array<UVEdges>;
${uniformVolumeDonorSumWGSL}
@compute @workgroup_size(4,4,4)
fn uvCorrectionCapacity(@builtin(global_invocation_id)gid:vec3u){
 let id=vec3i(gid);if(!valid(id)){return;}textureStore(gammaOut,id,vec4f(uvOpen(id)));
}
@compute @workgroup_size(4,4,4)
fn uvCorrectionTargets(@builtin(global_invocation_id)gid:vec3u){
 let id=vec3i(gid);if(!valid(id)){return;}textureStore(gammaOut,id,vec4f(uvTarget(id)));
}

fn uvCorner(i:u32)->vec3i{return vec3i(i32(i&1u),i32((i>>1u)&1u),i32((i>>2u)&1u));}
fn uvDonor(i:u32,k:u32)->u32{
  // Zero-weight corners can lie outside the lattice. The old representation
  // used the receiver for those slots; preserve that behavior, including the
  // normalization and gather's zero contributions.
  if(k==8u||uvEdges[uvEdgeAddress(i)].weight[k]==0.0){return i;}
  let d=vec3u(params.dimsDt.xyz);let o=vec3u(uvCorner(k));
  return uvEdges[uvEdgeAddress(i)].base+o.x+d.x*(o.y+d.y*o.z);
}
// THE SOLVE WINDOW. Words 7..12 of the active-region header are the union of
// this step's padded seed box with the previous one -- the box every windowed
// dispatch runs on, and therefore the box outside which nothing is written
// this step. With the window off the host publishes [0,dims), so every test
// below folds to a plain domain test and the dense arm keeps its exact instruction stream.
fn uvWindowMin()->vec3i{return vec3i(vec3u(activeRegion[7],activeRegion[8],activeRegion[9]));}
fn uvWindowMax()->vec3i{return vec3i(vec3u(activeRegion[10],activeRegion[11],activeRegion[12]));}
fn uvInWindow(id:vec3i)->bool{return all(id>=uvWindowMin())&&all(id<uvWindowMax());}
/** One 4h tile past the window on every side: the seed's vertex test reads the
 * upper vertex plane of its tile, which belongs to the next tile along. */
fn uvTileInWindow(t:vec3i)->bool{
  return all(t>=uvWindowMin()/4-vec3i(1))&&all(t<(uvWindowMax()+vec3i(3))/4+vec3i(1));
}
fn uvStepHasExternalSource()->bool{
  return params.drop.w>0.0||length(params.inflowVelocityLength.xyz)*inflowStrength()>1e-6;
}
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
  var g=vec3f(0);for(var a=0u;a<UNIFORM_REFERENCE_DIMENSION;a++){var e=vec3f(0);e[a]=0.25;
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
  ${solidFreeTrace ? `// E4. The walk exists to stop a characteristic tunnelling through a thin
  // voxel wall. Where the host has certified that no cell in the domain is
  // cut -- no static solid voxel, no rigid body, no terrain -- every
  // cellOpenFraction on the segment is exactly one, the early return can
  // never be taken, and the loop's only effect is to return the RK2 endpoint. That is
  // the value returned here, bit for bit, at ceil(2D) fewer loads a vertex:
  // at the figure-7 impact D is sixty-one cells, so this is 122 dependent
  // texture probes per vertex that cannot change the answer.
  if(uvSolidFree()){return end;}` : ""}
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
// Bits published by projection: positive faces 0..2, negative domain faces 3..5.
fn uvContactReleased(face:vec3i,axis:u32)->bool{
  var cell=face;var bit=axis;if(face[axis]<0){cell[axis]=0;bit+=3u;}
  if(!valid(cell)){return false;}
  return (u32(round(textureLoad(velocityIn,cell,0).w))&(1u<<bit))!=0u;
}
// Ambient air swept in by separating MAC wall velocities, in metres.
fn uvReleasedWalls(p:vec3f,advected:f32)->f32{
  var result=advected;let h=params.cellGravity.xyz;
  for(var axis=0u;axis<3u;axis++){
    if(axis==2u&&params.tuning.w>0.5){continue;}
    for(var side=0u;side<2u;side++){
      let upper=side==1u;let inward=select(1.0,-1.0,upper);
      // Release is determined by the projected velocity, in every orientation.
      let ambient=axis==1u&&upper&&params.boundary.w>0.5;
      let plane=select(0.0,f32(dims()[axis]),upper);
      for(var corner=0u;corner<4u;corner++){
        var probe=p;probe[(axis+1u)%3u]+=select(-1e-4,1e-4,(corner&1u)!=0u);
        probe[(axis+2u)%3u]+=select(-1e-4,1e-4,(corner&2u)!=0u);
        probe[axis]=plane+inward*1e-4;let cell=clampCell(vec3i(floor(probe)));
        // Outside the window the plane's velocity was not written this step.
        // The term only reaches vertices within dt*|v|/h of the plane and the
        // window snaps to a wall it comes within one padding of, so a vertex
        // this dispatch owns is never one of them: skipping is the dense
        // answer, reading the stale plane would not be.
        if(!uvInWindow(cell)){continue;}
        let speed=select(boundaryVelocity(cell)[axis],velocity(cell)[axis],upper);
        let away=inward*speed;var face=cell;if(!upper){face[axis]-=1;}
        if(!ambient&&!uvContactReleased(face,axis)){continue;}
        if(params.dimsDt.w*away>1e-4*h[axis]){result=max(result,params.dimsDt.w*away-inward*(p[axis]-plane)*h[axis]);}
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
  for(var axis=0u;axis<UNIFORM_REFERENCE_DIMENSION;axis++){
    for(var side=0u;side<2u;side++){
      let upper=side==1u;let inward=select(1.0,-1.0,upper);
      let plane=select(0.0,f32(dims()[axis]),upper);
      let ambient=axis==1u&&upper&&params.boundary.w>0.5;
      if(abs(p[axis]-plane)>1e-5||ambient){continue;}
      var probe=p;probe[axis]+=inward;
      // A dry wall is wetted only by arriving liquid. Copying any nearby wet
      // interior reattaches a departing sheet after its wall face becomes air.
      if(advected>=0.0&&inward*sampleVelocity(probe)[axis]>=-1e-6){continue;}
      interior[axis]+=inward;contact=true;
    }
  }
  if(!contact||uvOpen(clampCell(vec3i(floor(interior))))<=1e-5){return advected;}
  let continued=uvPhi(uvTrace(interior,params.dimsDt.w));
  return select(advected,min(advected,continued),continued<0.0);
}
// A released embedded wall supplies incoming air just like the domain halo.
// Follow the characteristic to its first solid hit; this reaches a wall even
// when one step sweeps several cells. Scalar mass transport still stops there.
fn uvEmbeddedAir(p:vec3f,advected:f32)->f32{
  ${solidFreeTrace ? `// Identity with no cut cell: the walk's only exit is a solid hit.
  if(uvSolidFree()){return advected;}` : ""}
  let h=params.cellGravity.xyz;let dt=params.dimsDt.w;
  let mid=clamp(p-0.5*dt*sampleVelocity(p)/h,vec3f(0),vec3f(dims()));
  let end=clamp(p-dt*sampleVelocity(mid)/h,vec3f(0),vec3f(dims()));
  let steps=max(1u,u32(ceil(2.0*max(abs(end.x-p.x),max(abs(end.y-p.y),abs(end.z-p.z))))));
  var previous=p;var result=advected;
  for(var step=1u;step<=steps;step++){
    let q=mix(p,end,f32(step)/f32(steps));var solid=vec3i(floor(q));
    // Both vertex planes of the 2D reference sit on its one cell layer; the
    // upper plane's floor would otherwise leave the lattice and never see a wall.
    if(UNIFORM_REFERENCE_DIMENSION==2u){solid.z=0;}
    if(valid(solid)&&cellOpenFraction(solid)<=1e-5){
      for(var axis=0u;axis<3u;axis++){for(var side=-1;side<=1;side+=2){
        var fluid=solid;fluid[axis]+=side;if(cellOpenFraction(fluid)<=1e-5){continue;}
        let inward=f32(side);let plane=f32(solid[axis])+select(0.0,1.0,side>0);
        // Only the first crossed face may supply air, not the far side of a wall.
        let distance=inward*(p[axis]-plane);if(distance< -1e-5||inward*(end[axis]-p[axis])>=0.0){continue;}
        let a=inward*(previous[axis]-plane);let b=inward*(q[axis]-plane);
        if(a< -1e-5||b>1e-5){continue;}
        let face=select(fluid,solid,side>0);let data=pressureFaceData(face,axis);
        let away=inward*(domainFaceFluidVelocity(face,axis)-data[axis]);
        if(uvContactReleased(face,axis)&&dt*away>1e-4*h[axis]){result=max(result,dt*away-distance*h[axis]);}
      }}
      return result;
    }
    previous=q;
  }
  return result;
}
// Continue liquid onto voxel wall vertices -- in both directions. A wall-face
// vertex has no normal velocity, so uvTrace returns it to itself: a min-only
// contact term is a fixed point, and one splash over a stair tread leaves phi<0
// there for the rest of the run with no V behind it, owning a false pressure
// row whose |phi| the volume correction then amplifies through its own wall
// gradient. A wall face is not an independent store of phi. It is ASSIGNED the
// continuation of the open cells it touches -- the arriving ones when the
// interior moves into the wall, otherwise all of them -- so it wets when liquid
// reaches it and dries the step after the cell beside it turns to air. Only a
// vertex buried in solid, with no open incident cell, keeps its own value. The
// air update runs last so solved separation still wins at edges/corners.
fn uvEmbeddedContact(p:vec3f,advected:f32)->f32{
  ${solidFreeTrace ? `// Identity with no cut cell: every branch below needs a closed neighbour,
  // and max(advected,-1e20) is advected for every representable phi.
  if(uvSolidFree()){return advected;}` : ""}
  var arriving=1e20;var continued=1e20;var air=-1e20;
  for(var k=0u;k<8u;k++){
    let fluid=vec3i(p)-vec3i(1)+uvCorner(k);if(cellOpenFraction(fluid)<=1e-5){continue;}
    for(var axis=0u;axis<3u;axis++){
      let side=select(-1,1,fluid[axis]<i32(p[axis]));var solid=fluid;solid[axis]+=side;
      if(!valid(solid)||cellOpenFraction(solid)>1e-5){continue;}
      var interior=p;interior[axis]-=f32(side);
      let into=f32(side)*sampleVelocity(interior)[axis]>1e-6;
      if(advected<0.0||into){
        let value=uvPhi(uvTrace(interior,params.dimsDt.w));
        continued=min(continued,value);if(into){arriving=min(arriving,value);}
      }
      let face=select(solid,fluid,side>0);let data=pressureFaceData(face,axis);
      let away=-f32(side)*(domainFaceFluidVelocity(face,axis)-data[axis]);
      let travel=params.dimsDt.w*away;
      // Contact-solver residue must not cut a cell-wide air sheet.
      if(uvContactReleased(face,axis)&&travel>1e-4*params.cellGravity[axis]){air=max(air,travel);}
    }
  }
  var result=advected;
  if(arriving<1e20){result=arriving;}else if(continued<1e20){result=continued;}
  return max(result,air);
}
// phi/V agreement (docs/uniform-geometric-phi-volume-agreement-handoff.md). V knows
// how much liquid is near a place; phi knows where the surface is. Nothing else
// in the method moves phi toward V, so phi's transport losses are permanent.
//
// The residual pass runs first, on start-of-step V, gamma (= uvTarget of the
// same phi) and phi, which are mutually consistent. It packs, for a fully open
// cell inside the 1.5h band, r = V - uvTarget and a = 1 for a cut cell as
// r + 4a into the gamma ping-pong, which is scratch outside uvGather.
const UV_AGREEMENT_PACK=4.0;
@compute @workgroup_size(4,4,4)
fn uvAgreementResidual(@builtin(global_invocation_id)gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}var packed=0.0;
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  if(uvOpen(id)>=0.99999){let g=textureLoad(gammaIn,id,0).x;let v=volume(id);
    if((g>0.0||v>0.0)&&abs(uvPhi(vec3f(id)+vec3f(0.5)))<1.5*h){
      packed=clamp(v-g,-1.5,1.5)+select(0.0,UV_AGREEMENT_PACK,g>0.0&&g<1.0);}}
  textureStore(gammaOut,id,vec4f(packed));
}
// A band vertex gathers R and A over the 8^3 cells around it with tent weights
// and moves along its own normal by gain*R/A cells. It reads V only as a patch
// integral: per cell, V's pattern is transport noise that sharpening has already
// reshaped to phi's outline, and reading it as geometry makes every surface
// bubble. Overlapping tents, not 4h tiles: a tile splits R from A wherever the
// surface runs near its face (lateral roughness per unit gain 0.13 against
// 0.005). And low gain: at 0.25 cells per unit residual smooth regional mismatch
// drives phi fast enough to make waves (dam break roughness x3); at 0.05 the
// clamp is nearly idle on a pool and only matters to a film's erosion rate.
fn uvAgreementShift(p:vec3f)->f32{
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));var R=0.0;var A=0.0;let base=vec3i(p);
  for(var dz=-4;dz<4;dz++){for(var dy=-4;dy<4;dy++){for(var dx=-4;dx<4;dx++){
    let c=base+vec3i(dx,dy,dz);if(!valid(c)){continue;}
    let packed=textureLoad(gammaIn,c,0).x;if(packed==0.0){continue;}
    let cut=packed>0.5*UV_AGREEMENT_PACK;
    let o=abs(vec3f(f32(dx),f32(dy),f32(dz))+vec3f(0.5))/4.5;let w=(1.0-o.x)*(1.0-o.y)*(1.0-o.z);
    R+=w*(packed-select(0.0,UV_AGREEMENT_PACK,cut));if(cut){A+=w;}}}}
  if(A<1.0){return 0.0;}let s=R/A;if(abs(s)<0.02){return 0.0;}
  return h*clamp(params.agreement.z*s,-params.agreement.w,params.agreement.w);
}
// Where phi offers no surface to disagree with -- no liquid centre in the 4^3
// cells around the vertex -- but the eight adjacent cells average over a quarter
// full, V is the only description of the liquid there is, and it is written
// INTO phi so rows, extension and render all see it. This is what keeps a film
// thinner than phi can carry alive: a seeded cell owns an ordinary pressure row.
fn uvSeedPhi(p:vec3f,phi:f32)->f32{
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));let base=vec3i(p);var sum=0.0;var n=0.0;
  for(var k=0u;k<8u;k++){let c=base-vec3i(1)+uvCorner(k);if(valid(c)&&uvOpen(c)>=0.99999){sum+=volume(c);n+=1.0;}}
  if(n<1.0||sum/n<=0.25){return phi;}
  for(var dz=-2;dz<2;dz++){for(var dy=-2;dy<2;dy++){for(var dx=-2;dx<2;dx++){let c=base+vec3i(dx,dy,dz);
    if(valid(c)&&uvPhi(vec3f(c)+vec3f(0.5))<0.0){return phi;}}}}
  return min(phi,h*(0.5-sum/n));
}
// E4. Host certificate: no cell in this domain is cut this step. The host
// owns all three sources of a cut cell -- the packed static solid voxel mask,
// the terrain heightfield and the live rigid bodies -- so this is a uniform
// condition, not a field, and every consumer of it folds to a constant branch.
// Zero with the experiment off, so every test below reads false.
fn uvSolidFree()->bool{return params.lean.x>0.5;}
// E5. SHELL membership of the tile a VERTEX sits in. SHELL is FINE dilated by
// at least one tile, and a vertex's eight incident cells lie in tiles t-1..t,
// so "t is not SHELL" implies none of those cells is in a FINE tile.
fn uvShellTileAt(id:vec3i)->bool{
  let t=clamp(id/4,vec3i(0),uvCoarseDims()-vec3i(1));
  return (atomicLoad(&sharpenDeposits[uvCoarseBase()+4u*uvCoarseIndex(t)+3u])&2)!=0;
}
/**
 * E5's FAR AIR arm of the vertex phi advect.
 *
 * Outside SHELL the four post-advect corrections are identity and are not
 * evaluated. Each is a theorem about the FINE seed, which marks every tile
 * holding liquid at or above the dust floor, any partially open cell
 * (solid/terrain/body), any source this step, or any vertex with phi below the
 * 4h band; SHELL is that set dilated by at least one tile.
 *
 *  - uvEmbeddedContact / uvEmbeddedAir need a cell with open fraction <= 1e-5
 *    among the vertex's own incident cells, or on its characteristic. Such a
 *    cell is a FINE seed, and the incident ones lie in tiles t-1..t, so the
 *    vertex is in SHELL. (The characteristic case is covered too: the walk
 *    stops at the FIRST solid crossed, and reaching one from outside SHELL
 *    needs the departure point inside the solid's own tile.)
 *  - uvSourcePhi is identity on a step with no drop and no inflow, which is a
 *    host-uniform condition and is tested directly.
 *  - uvAgreementShift gathers the packed residual over base+[-4,4). The
 *    residual is nonzero only where |phi| < 1.5h and gamma or V is nonzero,
 *    which is inside FINE; outside SHELL every tap is +0, A stays below one
 *    and the function returns exactly zero.
 *  - uvSeedPhi returns phi unless the eight incident cells average more than a
 *    quarter full. Those cells are in tiles t-1..t, none of them FINE, so each
 *    holds |V| < the dust floor and the average cannot reach 0.25.
 *
 * The one term NOT covered by a theorem is uvReleasedWalls, whose ambient-air
 * source sits on the six DOMAIN planes rather than on any tile class. Its
 * contribution is max(phi, dt*away - distance*h): it can only raise phi, and
 * only within dt*away/h cells of a plane. The far-air arm is therefore refused
 * within one 4h tile of every plane, which is exact for every wall whose
 * per-step ambient travel is under four cells -- including every static wall,
 * since the lower planes read the prescribed boundary velocity and an upper
 * plane needs a released contact face or an open lid with inflow on its own
 * cell row. A wall driven harder than that loses the term between four cells
 * and dt*away/h from the plane, where it would have added air.
 */
fn uvPhiFarAir(vertex:vec3i)->bool{
  if(params.lean.z<=0.5||params.physical.z<0.0){return false;}
  if(uvStepHasExternalSource()){return false;}
  // The 2D reference has no z planes to refuse: its single cell layer is a
  // symmetry slab, and the plane test would otherwise refuse every vertex.
  let nearWall=(vertex<vec3i(4))|(vertex>dims()-vec3i(4));
  if(nearWall.x||nearWall.y||(UNIFORM_REFERENCE_DIMENSION==3u&&nearWall.z)){return false;}
  return !uvShellTileAt(vertex);
}
// SPLASH SURVIVAL (docs/uniform-geometric-splash-dissipation-plan.md). Each
// experiment reads its own params.splash/splashB lane, which the host writes
// as zero with the toggle off, so the default arm evaluates none of them.
//
// The V-reading terms of the phi advect sample at the DEPARTURE point q: this
// pass runs before transport, so V is start-of-step V, and the liquid that
// reaches the vertex by the end of the step is the V around q now. Reading
// around the vertex instead would trail V by a step and drain every moving front.
fn uvSplashAdvect()->bool{
  return params.splash.w>0.5||params.splashB.x>0.5||params.splashB.y>0.5||params.splashB.z>0.5;
}
// G. Trilinear interpolation overestimates a convex distance field by up to
// h^2/4r, so every resample moves a drop's surface inward by that much.
// Catmull-Rom reproduces quadratics; clamping to the eight enclosing vertices
// keeps it monotone, so it cannot overshoot a sign across a thin sheet.
fn uvCatmullRom(t:f32)->vec4f{
  let t2=t*t;let t3=t2*t;
  return 0.5*vec4f(2.0*t2-t3-t,3.0*t3-5.0*t2+2.0,4.0*t2-3.0*t3+t,t3-t2);
}
fn uvPhiCubic(position:vec3f)->f32{
  let p=clamp(position,vec3f(0),vec3f(dims()));
  let base=min(vec3i(floor(p)),dims()-vec3i(1));let f=p-vec3f(base);
  let wx=uvCatmullRom(f.x);let wy=uvCatmullRom(f.y);let wz=uvCatmullRom(f.z);
  var value=0.0;var lo=1e30;var hi=-1e30;
  for(var dz=-1;dz<3;dz++){var plane=0.0;
    for(var dy=-1;dy<3;dy++){var row=0.0;
      for(var dx=-1;dx<3;dx++){let o=vec3i(dx,dy,dz);
        let s=textureLoad(uvPhiIn,clamp(base+o,vec3i(0),dims()),0).x;row+=wx[dx+1]*s;
        if(all(o>=vec3i(0))&&all(o<=vec3i(1))){lo=min(lo,s);hi=max(hi,s);}}
      plane+=wy[dy+1]*row;}
    value+=wz[dz+1]*plane;}
  return clamp(value,lo,hi);
}
// V above this is liquid to the splash stages; below it is the transport and
// sharpening tail every surface carries (about 5e-3 at 1.5h over a resting
// pool, 1e-5 at 2.5h), which neither keeps phi-liquid alive nor flies.
const UV_LIQUID_EVIDENCE:f32=0.05;
// E, drain. Phi-liquid with no cell in the 4^3 around q holding liquid evidence
// is a ghost: V left it or never came. It rises half a cell a step, to +h/2.
fn uvDrainGhost(q:vec3f,phi:f32)->f32{
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));if(phi>=0.5*h){return phi;}
  let centre=vec3i(floor(q+vec3f(0.5)));
  for(var dz=-2;dz<2;dz++){for(var dy=-2;dy<2;dy++){for(var dx=-2;dx<2;dx++){let c=centre+vec3i(dx,dy,dz);
    if(valid(c)&&volume(c)>UV_LIQUID_EVIDENCE){return phi;}}}}
  return min(phi+0.5*h,0.5*h);
}
// E, seed. uvSeedPhi writes h(1/2 - mean V) over eight cells, which is liquid
// only above a half-full MEAN: a one-cell drop means 1/8 and never qualifies.
// Here the fullest incident cell of q over half its capacity owns the vertex,
// so a single compacted cell yields a liquid centre. Same guard: no phi-liquid
// centre in the 4^3 cells around q, so a healthy surface is never re-seeded.
fn uvSeedPhiCells(q:vec3f,phi:f32)->f32{
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  let base=vec3i(floor(q-vec3f(0.5)));var seed=phi;
  for(var k=0u;k<8u;k++){let c=base+uvCorner(k);if(!valid(c)){continue;}let open=uvOpen(c);
    if(open>=0.99999){seed=min(seed,h*(0.5-volume(c)/open));}}
  if(seed>=phi||seed>=0.0){return phi;}
  let centre=vec3i(floor(q+vec3f(0.5)));
  for(var dz=-2;dz<2;dz++){for(var dy=-2;dy<2;dy++){for(var dx=-2;dx<2;dx++){let c=centre+vec3i(dx,dy,dz);
    if(valid(c)&&uvPhi(vec3f(c)+vec3f(0.5))<0.0){return phi;}}}}
  return seed;
}
// D. Local volume for a small isolated body. W is the 16^3 cells around q. If
// W's outer layer holds no V and no phi fill, every body this vertex touches
// lies wholly inside W, so the interior's sum of V - gamma is those bodies'
// own residual: shift phi by it over their cut cells at full gain, at most a
// quarter cell a step. A pool, or a drop beside one, never has an empty outer
// layer, and leaves on the first wet cell of it. gamma is start-of-step
// uvTarget(phi): with this on the host binds the trace group, never the
// agreement's packed residual.
fn uvIsolatedShift(q:vec3f)->f32{
  let base=vec3i(floor(q+vec3f(0.5)));let empty=max(params.tuning.z,1e-5);
  for(var dz=-8;dz<8;dz++){for(var dy=-8;dy<8;dy++){
    let inner=dz>-8&&dz<7&&dy>-8&&dy<7;var dx=-8;
    loop{if(dx>=8){break;}let c=base+vec3i(dx,dy,dz);
      if(valid(c)&&(volume(c)>empty||textureLoad(gammaIn,c,0).x>0.0)){return 0.0;}
      dx=select(dx+1,select(dx+1,7,dx==-8),inner);}}}
  var R=0.0;var A=0.0;
  for(var dz=-7;dz<7;dz++){for(var dy=-7;dy<7;dy++){for(var dx=-7;dx<7;dx++){
    let c=base+vec3i(dx,dy,dz);if(!valid(c)){continue;}
    let g=textureLoad(gammaIn,c,0).x;R+=volume(c)-g;if(g>0.0&&g<uvOpen(c)){A+=1.0;}}}}
  if(A<1.0){return 0.0;}
  return min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z))*clamp(R/A,-0.25,0.25);
}
// A. CM11b Sec. 3.4: redistance without moving the surface. A vertex with an
// opposite-sign vertex among its 26 neighbours is a corner of a crossed cell,
// and the trilinear zero set is defined by those corners alone, so keeping
// their advected values -- clamped to one cell, as CM11b does -- leaves the
// surface where advection put it. Newton measures the rest of the band
// against that contour.
fn uvSurfaceVertex(vertex:vec3i,value:f32)->bool{
  for(var dz=-1;dz<=1;dz++){for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){
    let n=clamp(vertex+vec3i(dx,dy,dz),vec3i(0),dims());
    if((textureLoad(uvPhiIn,n,0).x<0.0)!=(value<0.0)){return true;}}}}
  return false;
}
@compute @workgroup_size(4,4,4)
fn uvAdvectPhi(@builtin(global_invocation_id)gid:vec3u){
  let vertex=activeVertexId(gid);if(any(vertex<vec3i(0))||any(vertex>dims())){return;}let p=vec3f(vertex);
  let departure=uvTrace(p,params.dimsDt.w);
  var advected=uvPhi(departure);
  // The far-air theorems below hold at the departure point too unless q lies
  // in a SHELL tile, where a splash term may fire; see uvSplashAdvect.
  ${phiLean ? `if(uvPhiFarAir(vertex)&&!(uvSplashAdvect()&&uvShellTileAt(vec3i(departure)))){textureStore(uvPhiOut,vertex,vec4f(advected));return;}` : ""}
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  if(params.splashB.x>0.5&&abs(advected)<2.0*h){advected=uvPhiCubic(departure);}
  let contact=uvEmbeddedContact(p,uvClosedWallPhi(p,advected));
  var released=uvReleasedWalls(p,uvEmbeddedAir(p,contact));
  if(params.splashB.z>0.5){released=uvDrainGhost(departure,released);}
  var value=uvSourcePhi(p,released);
  if(params.agreement.z>0.0&&abs(value)<2.0*h){value-=uvAgreementShift(p);}
  if(params.splash.w>0.5&&abs(value)<2.0*h){value-=uvIsolatedShift(departure);}
  if(params.agreement.y>0.5){value=uvSeedPhi(p,value);}
  if(params.splashB.y>0.5){value=uvSeedPhiCells(departure,value);}
  textureStore(uvPhiOut,vertex,vec4f(value));
}
@compute @workgroup_size(4,4,4)
fn uvRedistancePhi(@builtin(global_invocation_id)gid:vec3u){
  let vertex=activeVertexId(gid);if(any(vertex<vec3i(0))||any(vertex>dims())){return;}let p=vec3f(vertex);let initial=uvPhi(p);
  let h=params.cellGravity.xyz;let band=4.0*max(h.x,max(h.y,h.z));
  var value=initial;
  if(abs(initial)>1e-8&&abs(initial)<band){
    if(params.splash.x>0.5&&uvSurfaceVertex(vertex,initial)){let cell=min(h.x,min(h.y,h.z));value=clamp(initial,-cell,cell);}
    else{var q=p;
    for(var i=0u;i<8u;i++){let g=uvGradient(q);let norm=dot(g/h,g/h);if(norm<1e-16){break;}
      let next=clamp(q-clamp(uvPhi(q)*g/(h*h*norm),vec3f(-2),vec3f(2)),
        max(vec3f(0),p-vec3f(4)),min(vec3f(dims()),p+vec3f(4)));
      // A clamped Newton step can cross a distance ridge and walk away
      // from the contour. Reject it before roundoff picks a different root.
      if(abs(uvPhi(next))>=abs(uvPhi(q))){break;}
      q=next;}
    if(abs(uvPhi(q))<0.005*min(h.x,min(h.y,h.z))){value=sign(initial)*length((p-q)*h);}}}
  textureStore(uvPhiOut,vertex,vec4f(value));
}
fn uvOpen(id:vec3i)->f32{if(!valid(id)){return 0.0;}return cellOpenFraction(id);}
// E3. Every pass of Sec. 3.4's conservative transport runs only on the live
// tile set TRANSPORT (class bit 4): the seed dilated by the fine reach plus
// params.twoLevel.w tiles. A 4x4x4 workgroup IS one 4h tile, so the test below
// is uniform across the workgroup and the exit costs one predicated branch.
//
// Why this is exactly the dense result. uvNormalizeDonors divides every weight
// by the column sum accumulated over the rows that were BUILT, so after it
// sum over receivers in the set of w[i][d] = 1 for every donor d any of them
// samples -- and uvFallback gives a donor nobody samples a self-edge. The
// gather therefore moves every donor's V somewhere and never duplicates it,
// for any built set. What restricting the set can do is (a) DESTROY the V of a
// cell outside it, because the gather writes zero there rather than the old
// value (identity would double it instead, which is the one thing that creates
// volume), and (b) refuse liquid to a cell outside it that should have been
// wetted, which stalls a front without losing a drop.
//
// (a) is closed by the predicate "V = 0 outside the set", which holds because
// the volume dust floor zeroes |V| below the threshold wherever V is written
// and every cell at or above it seeds its own tile. With the floor at zero the
// predicate fails, so the host forces the dense schedule there.
// (b) is closed by the reach, which must cover ceil(D)+1 cells for this step's
// largest displacement D. The classify measures D and publishes it beside the
// configured reach, because nothing in the numbers reveals a short one.
fn uvTransportTiles()->bool{return params.twoLevel.w>=0.0;}
fn uvTransportTileAt(id:vec3i)->bool{
  let t=clamp(id/4,vec3i(0),uvCoarseDims()-vec3i(1));
  return (atomicLoad(&sharpenDeposits[uvCoarseBase()+4u*uvCoarseIndex(t)+3u])&4)!=0;
}
/** Uniform across a workgroup; false whenever the experiment is off. The
 * window origin is aligned to the 4h lattice, so a windowed workgroup is still
 * exactly one tile and the test is still uniform across it. */
fn uvTransportSkip(id:vec3i)->bool{
  if(!uvTransportTiles()){return false;}
  return !uvTransportTileAt(id);
}
${donorTiles ? `// The decoded sums have two readers: uvFallback reads a built row's own word
// and uvNormalizeDonors reads the words of the donors a built row samples. A
// built row lies in a TRANSPORT tile and its donors within ceil(D)+1 cells of
// it, so every word either reads is inside DONORS (uvTwoLevelDonorReach) and
// the decode is skipped everywhere else: six limb loads and a store per cell,
// four times a step, over air nothing addresses. Words outside DONORS keep
// whatever they held; nothing in this method reads them.
fn uvDonorSkip(id:vec3i)->bool{
  if(!uvTransportTiles()){return false;}
  let t=clamp(id/4,vec3i(0),uvCoarseDims()-vec3i(1));
  return (atomicLoad(&sharpenDeposits[uvCoarsePlane(0u)+uvCoarseIndex(t)])${tileReach ? "&8" : ""})==0;
}` : ""}
@compute @workgroup_size(4,4,4)
fn uvBuildEdges(@builtin(global_invocation_id)gid:vec3u){
  let id=uvWorkId(gid);if(uvTransportSkip(id)){return;}
  if(!valid(id)){return;}let index=linearIndex(id);
  let departure=uvTrace(vec3f(id)+vec3f(0.5),params.dimsDt.w)-vec3f(0.5);
  let base=vec3i(floor(departure));let f=fract(departure);
  uvEdges[uvEdgeAddress(index)].base=linearIndex(base);
  for(var k=0u;k<9u;k++){uvEdges[uvEdgeAddress(index)].weight[k]=0.0;}
  for(var k=0u;k<8u;k++){let o=uvCorner(k);let q=base+o;
    let w=select(vec3f(1)-f,f,o==vec3i(1));
    if(valid(q)&&uvOpen(id)>0.0){
      let weight=w.x*w.y*w.z*min(uvOpen(id),uvOpen(q));
      uvEdges[uvEdgeAddress(index)].weight[k]=weight;uvAddDonor(linearIndex(q),weight);}}
}
@compute @workgroup_size(4,4,4)
fn uvFinishDonorSums(@builtin(global_invocation_id)gid:vec3u){
  let id=uvDonorId(gid);if(!valid(id)){return;}
  ${donorTiles ? "if(uvDonorSkip(id)){return;}" : ""}
  let i=linearIndex(id);atomicStore(&sharpenDeposits[i],bitcast<i32>(uvDonorSum(i)));
}
@compute @workgroup_size(4,4,4)
fn uvFallback(@builtin(global_invocation_id)gid:vec3u){
  let id=uvWorkId(gid);if(uvTransportSkip(id)){return;}
  if(!valid(id)){return;}let i=linearIndex(id);
  if(atomicLoad(&sharpenDeposits[i])==0){uvEdges[uvEdgeAddress(i)].weight[8]=max(uvOpen(id),1e-6);}
}
@compute @workgroup_size(4,4,4)
fn uvNormalizeRows(@builtin(global_invocation_id)gid:vec3u){
  let id=uvWorkId(gid);if(uvTransportSkip(id)){return;}
  if(!valid(id)){return;}let i=linearIndex(id);var sum=0.0;
  for(var k=0u;k<9u;k++){sum+=uvEdges[uvEdgeAddress(i)].weight[k];}
  let scale=uvOpen(id)/max(sum,1e-20);
  for(var k=0u;k<9u;k++){let weight=uvEdges[uvEdgeAddress(i)].weight[k]*scale;
    uvEdges[uvEdgeAddress(i)].weight[k]=weight;uvAddDonor(uvDonor(i,k),weight);}
}
@compute @workgroup_size(4,4,4)
fn uvNormalizeDonors(@builtin(global_invocation_id)gid:vec3u){
  let id=uvWorkId(gid);if(uvTransportSkip(id)){return;}
  if(!valid(id)){return;}let i=linearIndex(id);
  for(var k=0u;k<9u;k++){let donor=uvDonor(i,k);
    let sum=bitcast<f32>(atomicLoad(&sharpenDeposits[donor]));
    uvEdges[uvEdgeAddress(i)].weight[k]/=max(sum,1e-20);}
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
  let id=activeId(gid);if(!valid(id)){return;}
  // Outside the live set both outputs are known in closed form, so neither the
  // nine-term gather nor uvTarget's eight trilinear phi probes are evaluated.
  // V is zero there by the predicate above. Gamma is zero because TRANSPORT
  // contains every tile with a vertex inside the 4h band -- outside it every
  // corner sample of uvTarget is positive, so both its fill count and its
  // plane-box fraction are zero. The stores themselves stay: volumeOut and
  // gammaOut are ping-pong targets whose previous contents are two steps old.
  if(uvTransportSkip(id)){
    textureStore(volumeOut,id,vec4f(0.0));textureStore(gammaOut,id,vec4f(0.0));return;
  }
  // Only an unplaceable sealed-solid reservoir can remain after geometry
  // reconciliation. Preserve it until a later edit creates an outlet.
  if(uvOpen(id)<=0.0){textureStore(volumeOut,id,vec4f(volume(id)));textureStore(gammaOut,id,vec4f(0.0));return;}
  let i=linearIndex(id);var value=0.0;
  for(var k=0u;k<9u;k++){value+=uvEdges[uvEdgeAddress(i)].weight[k]*volume(uvCell(uvDonor(i,k)));}
  value+=min(dropSource(id),max(0.0,uvOpen(id)-value));
  if(uvOpen(id)>0.0){value+=inflowSweptPlugSource(id,params.dimsDt.w);}
  textureStore(volumeOut,id,vec4f(uvDustFloor(value)));
  textureStore(gammaOut,id,vec4f(uvTarget(id)));
}
${uniformVolumeTargetWGSL(uniformAbOn("targetcache"))}
// After transport the fixed stencil arena is scratch for face proposals and
// cell budgets: three positive-face fluxes, surplus, need, phi, and two limits.
// 4h work map for the eight sharpening sweeps. Phi is fixed throughout them, so
// one classification pass serves the whole stage. A tile with no cell in the
// admission band has identically zero need and surplus, even if it contains
// nonzero V; every flux touching it is therefore zero. Prepare/propose/limit
// skip such a tile, neighbours read its fluxes as zero, and commit copies its V
// across the ping-pong pair. Never treat stale transport-edge scratch as a
// sharpening flux. The map uses the third conditioning plane behind its
// reserved work-counter header.
// The dense control is the same module with the override left false: the
// lookups fold away and the numerics are bit-identical.
override UV_SHARPEN_TILE_WORK:bool=false;
const UV_SHARPEN_TILE_MAP_WORD=8u;
const UV_SHARPEN_TILE_COUNT_WORD=7u;
fn uvSharpenTileIndex(id:vec3i)->u32{
  let d=(vec3u(dims())+vec3u(3))/4u;let t=vec3u(id)/4u;
  return 2u*cellCount()+UV_SHARPEN_TILE_MAP_WORD+t.x+d.x*(t.y+d.y*t.z);
}
// The window bound is part of the predicate, not an optimization: the eight
// sweeps read a NEIGHBOUR tile's proposals out of the fixed stencil arena, and
// outside the window that arena still holds this step's transport edges. A
// tile the classify did not visit is not a sharpening tile.
fn uvSharpenTileActive(id:vec3i)->bool{
  if(!uvInWindow(id)){return false;}
  if(!UV_SHARPEN_TILE_WORK){return true;}
  return atomicLoad(&sharpenDeposits[uvSharpenTileIndex(id)])!=0;
}
var<workgroup> uvTileAdmission:atomic<u32>;
@compute @workgroup_size(4,4,4)
fn uvClassifySharpenTiles(@builtin(global_invocation_id)gid:vec3u,
  @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)tile:vec3u){
  let cell=activeId(gid);
  if(lane==0u){atomicStore(&uvTileAdmission,0u);}workgroupBarrier();
  if(valid(cell)){
    let phi=uvPhi(vec3f(cell)+vec3f(0.5));
    let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
    // Negated comparison conservatively retains non-finite input as active.
    if(!(abs(phi)>=params.tuning.y*h)){atomicStore(&uvTileAdmission,1u);}
    // Compaction pours between liquid cells at any depth. A liquid cell is
    // live if it is below capacity or has a face neighbour that is: the second
    // half is what admits the full tile a deficient one must draw from.
    else if(params.agreement.x>0.5&&phi<0.0){let id=cell;
      var live=uvOpen(id)>0.99999&&volume(id)<uvOpen(id)-1e-4;
      for(var axis=0;axis<3&&!live;axis+=1){for(var side=-1;side<=1;side+=2){
        var n=id;n[axis]+=side;if(valid(n)&&uvOpen(n)>0.99999&&volume(n)<uvOpen(n)-1e-4
          &&uvPhi(vec3f(n)+vec3f(0.5))<0.0){live=true;}}}
      if(live){atomicStore(&uvTileAdmission,1u);}}
    // B2: orphan V beyond the band compacts in place, so its tile must run.
    else if(params.splash.y>1.5&&phi>0.0&&volume(cell)>0.0){atomicStore(&uvTileAdmission,1u);}
  }
  workgroupBarrier();
  if(lane==0u){let admission=atomicLoad(&uvTileAdmission);
    atomicStore(&sharpenDeposits[uvSharpenTileIndex(activeId(tile*4u))],i32(admission));
    if(admission!=0u){atomicAdd(&sharpenDeposits[2u*cellCount()+UV_SHARPEN_TILE_COUNT_WORD],1);}}
}
// Geometry and the target surface are fixed throughout the eight sweeps.
// Reuse the transport donor words as a per-page sharpening geometry cache.
@compute @workgroup_size(4,4,4)
fn uvCacheSharpenCells(@builtin(global_invocation_id)gid:vec3u){
 let id=uvWorkId(gid);if(!valid(id)||!uvSharpenTileActive(id)){return;}
 let i=uvEdgeAddress(linearIndex(id));
 uvEdges[i].weight[5]=uvPhi(vec3f(id)+vec3f(0.5));
 uvEdges[i].weight[6]=textureLoad(gammaIn,id,0).x;
 uvEdges[i].base=select(0u,1u<<15u,uvOpen(id)>0.99999);
}
@compute @workgroup_size(4,4,4)
fn uvCacheSharpenFaces(@builtin(global_invocation_id)gid:vec3u){
 let id=uvWorkId(gid);if(!valid(id)||!uvSharpenTileActive(id)){return;}
 let i=uvEdgeAddress(linearIndex(id));let phiA=uvEdges[i].weight[5];
 for(var axis=0u;axis<3u;axis++){
  uvEdges[i].base&=~(31u<<(5u*axis));var e=vec3i(0);e[axis]=1;let q=id+e;
  if(!valid(q)||!uvSharpenTileActive(q)||uvOpen(id)<0.99999||uvOpen(q)<0.99999||faceOpenFraction(id,axis)<0.99999){continue;}
  let j=uvEdgeAddress(linearIndex(q));let phiB=uvEdges[j].weight[5];
  let middle=uvPhi(vec3f(id)+vec3f(0.5)+0.5*vec3f(e));let epsilon=1e-6;
  let inwardA=phiA>=0.0&&phiB<phiA-epsilon&&middle<=phiA+epsilon&&middle>=phiB-epsilon;
  let inwardB=phiB>=0.0&&phiA<phiB-epsilon&&middle<=phiB+epsilon&&middle>=phiA-epsilon;
  let relayA=phiA>0.0&&uvEdges[i].weight[6]<=1e-6;
  let relayB=phiB>0.0&&uvEdges[j].weight[6]<=1e-6;
  uvEdges[i].base|=(1u
   |select(0u,2u,(middle<=epsilon&&!relayB)||inwardA)
   |select(0u,4u,(middle<=epsilon&&!relayA)||inwardB)
   |select(0u,8u,phiA<0.0&&phiB<phiA-epsilon)
   |select(0u,16u,phiB<0.0&&phiA<phiB-epsilon))<<(5u*axis);
 }
}
@compute @workgroup_size(4,4,4)
fn uvPrepareSharpen(@builtin(global_invocation_id)gid:vec3u){
  let id=uvWorkId(gid);if(!uvSharpenTileActive(id)){return;}
  if(!valid(id)){return;}let i=linearIndex(id);
  var phi:f32;var desired:f32;var open:bool;
  if(uvPageWorkEnabled()){
    phi=uvEdges[uvEdgeAddress(i)].weight[5];desired=uvEdges[uvEdgeAddress(i)].weight[6];
    open=(uvEdges[uvEdgeAddress(i)].base&(1u<<15u))!=0u;
  }else{phi=uvPhi(vec3f(id)+vec3f(0.5));desired=textureLoad(gammaIn,id,0).x;open=uvOpen(id)>0.99999;}
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  let dose=clamp(params.tuning.x,0.0,1.0);let own=volume(id);
  // Compaction admits every phi-liquid cell, and a liquid cell offers ALL of
  // its V: uvProposeSharpen only lets the part above phi's fill go anywhere but
  // to a deeper neighbour. Without it nothing refills a void inside the liquid
  // (entrained air the level set deleted but V kept): the band is 2.1h wide,
  // Sec. 3.7 only expels excess, and the dam break's deep interior sits at a
  // third full while its displaced volume piles on the surface.
  let compact=params.agreement.x>0.5;
  let admitted=open&&select(abs(phi)<params.tuning.y*h,phi<params.tuning.y*h,compact);
  let relay=phi>0.0&&desired<=1e-6;
  // Orphan V (splash.y, docs/uniform-geometric-splash-dissipation-plan.md B).
  // A relay pours whatever reaches it down phi's gradient into the nearest
  // body -- CM12 Fig. 3's objection to MMTD07 -- so a drop phi has lost drains
  // into the pool beside it. B1: a relay receives only if it already holds V
  // or lies within a cell of phi's surface, which the bulk's contiguous skirt
  // always does and an empty gap never does. B2 also admits V beyond the band:
  // a cell under half full offers all of it, and uvProposeSharpen lets it flow
  // only to a face neighbour holding more (CM12 Eq. 17 and Alg. 2's trace up
  // grad rho, reduced to face fluxes), so orphan V gathers into full cells.
  let orphan=params.splash.y>1.5&&open&&phi>=params.tuning.y*h;
  let receptive=params.splash.y<0.5||own>1e-4||phi<h;
  uvEdges[uvEdgeAddress(i)].weight[3]=select(0.0,dose*select(select(max(own-desired,0.0),own,compact&&phi<0.0),select(0.0,own,own<0.5),orphan),admitted||orphan);
  uvEdges[uvEdgeAddress(i)].weight[4]=select(0.0,dose*max(select(desired,select(0.0,1.0,receptive),relay)-own,0.0),admitted||orphan);
  uvEdges[uvEdgeAddress(i)].weight[5]=phi;
}
@compute @workgroup_size(4,4,4)
fn uvProposeSharpen(@builtin(global_invocation_id)gid:vec3u){
  let id=uvWorkId(gid);if(!uvSharpenTileActive(id)){return;}
  if(!valid(id)){return;}let i=linearIndex(id);
  let phiA=uvEdges[uvEdgeAddress(i)].weight[5];
  for(var axis=0u;axis<3u;axis++){
    uvEdges[uvEdgeAddress(i)].weight[axis]=0.0;var e=vec3i(0);e[axis]=1;let q=id+e;
    if(uvPageWorkEnabled()){
      let flags=((uvEdges[uvEdgeAddress(i)].base>>(5u*axis))&31u);if((flags&1u)==0u){continue;}
      let j=uvEdgeAddress(linearIndex(q));let own=uvEdgeAddress(i);
      if(params.splash.y>1.5){let band=params.tuning.y*min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
        let phiB=uvEdges[j].weight[5];
        if(phiA>=band||phiB>=band){let vA=volume(id);let vB=volume(q);
          let ab=select(0.0,min(uvEdges[own].weight[3],uvEdges[j].weight[4]),phiA>=band&&vB>vA);
          let ba=select(0.0,min(uvEdges[j].weight[3],uvEdges[own].weight[4]),phiB>=band&&vA>vB);
          uvEdges[own].weight[axis]=ab-ba;continue;}}
      var capA=uvEdges[own].weight[3];var capB=uvEdges[j].weight[3];
      if(params.agreement.x>0.5){let dose=clamp(params.tuning.x,0.0,1.0);
        if((flags&8u)==0u){capA=min(capA,dose*max(volume(id)-uvEdges[own].weight[6],0.0));}
        if((flags&16u)==0u){capB=min(capB,dose*max(volume(q)-uvEdges[j].weight[6],0.0));}}
      let ab=select(0.0,min(capA,uvEdges[j].weight[4]),(flags&2u)!=0u);
      let ba=select(0.0,min(capB,uvEdges[own].weight[4]),(flags&4u)!=0u);
      uvEdges[own].weight[axis]=ab-ba;continue;
    }
    if(!valid(q)||!uvSharpenTileActive(q)||uvOpen(id)<0.99999||uvOpen(q)<0.99999||faceOpenFraction(id,axis)<0.99999){continue;}
    let j=linearIndex(q);let phiB=uvEdges[uvEdgeAddress(j)].weight[5];
    // B2: a face with an orphan side moves orphan V only toward more V.
    if(params.splash.y>1.5){let band=params.tuning.y*min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
      if(phiA>=band||phiB>=band){let vA=volume(id);let vB=volume(q);
        let ab=select(0.0,min(uvEdges[uvEdgeAddress(i)].weight[3],uvEdges[uvEdgeAddress(j)].weight[4]),phiA>=band&&vB>vA);
        let ba=select(0.0,min(uvEdges[uvEdgeAddress(j)].weight[3],uvEdges[uvEdgeAddress(i)].weight[4]),phiB>=band&&vA>vB);
        uvEdges[uvEdgeAddress(i)].weight[axis]=ab-ba;continue;}}
    let middle=uvPhi(vec3f(id)+vec3f(0.5)+0.5*vec3f(e));let epsilon=1e-6;
    let inwardA=phiA>=0.0&&phiB<phiA-epsilon&&middle<=phiA+epsilon&&middle>=phiB-epsilon;
    let inwardB=phiB>=0.0&&phiA<phiB-epsilon&&middle<=phiB+epsilon&&middle>=phiA-epsilon;
    let relayA=phiA>0.0&&textureLoad(gammaIn,id,0).x<=1e-6;
    let relayB=phiB>0.0&&textureLoad(gammaIn,q,0).x<=1e-6;
    // With compaction a liquid cell's budget is its whole V, but only toward
    // a deeper (smaller phi) liquid neighbour; any other way it offers what it
    // always did, its surplus over phi's fill. Pouring is monotone in phi, so
    // it cannot cycle. With compaction off weight[3] IS that surplus.
    var capA=uvEdges[uvEdgeAddress(i)].weight[3];var capB=uvEdges[uvEdgeAddress(j)].weight[3];
    if(params.agreement.x>0.5){let dose=clamp(params.tuning.x,0.0,1.0);
      if(!(phiA<0.0&&phiB<phiA-epsilon)){capA=min(capA,dose*max(volume(id)-textureLoad(gammaIn,id,0).x,0.0));}
      if(!(phiB<0.0&&phiA<phiB-epsilon)){capB=min(capB,dose*max(volume(q)-textureLoad(gammaIn,q,0).x,0.0));}}
    let ab=select(0.0,min(capA,uvEdges[uvEdgeAddress(j)].weight[4]),(middle<=epsilon&&!relayB)||inwardA);
    let ba=select(0.0,min(capB,uvEdges[uvEdgeAddress(i)].weight[4]),(middle<=epsilon&&!relayA)||inwardB);
    uvEdges[uvEdgeAddress(i)].weight[axis]=ab-ba;
  }
}
@compute @workgroup_size(4,4,4)
fn uvLimitSharpen(@builtin(global_invocation_id)gid:vec3u){
  let id=uvWorkId(gid);if(!uvSharpenTileActive(id)){return;}
  if(!valid(id)){return;}let i=linearIndex(id);var outgoing=0.0;var incoming=0.0;
  for(var axis=0u;axis<3u;axis++){var e=vec3i(0);e[axis]=1;
    let positive=uvEdges[uvEdgeAddress(i)].weight[axis];var negative=0.0;
    if(valid(id-e)&&uvSharpenTileActive(id-e)){negative=uvEdges[uvEdgeAddress(linearIndex(id-e))].weight[axis];}
    outgoing+=max(positive,0.0)+max(-negative,0.0);incoming+=max(-positive,0.0)+max(negative,0.0);}
  uvEdges[uvEdgeAddress(i)].weight[7]=min(1.0,uvEdges[uvEdgeAddress(i)].weight[3]/max(outgoing,1e-20));
  uvEdges[uvEdgeAddress(i)].weight[8]=min(1.0,uvEdges[uvEdgeAddress(i)].weight[4]/max(incoming,1e-20));
}
fn uvLimitedFlux(i:u32,j:u32,axis:u32)->f32{
  if(!uvSharpenTileActive(uvCell(i))||!uvSharpenTileActive(uvCell(j))){return 0.0;}
  let raw=uvEdges[uvEdgeAddress(i)].weight[axis];let a=vec2f(uvEdges[uvEdgeAddress(i)].weight[7],uvEdges[uvEdgeAddress(i)].weight[8]);let b=vec2f(uvEdges[uvEdgeAddress(j)].weight[7],uvEdges[uvEdgeAddress(j)].weight[8]);
  return raw*select(min(a.y,b.x),min(a.x,b.y),raw>=0.0);
}
@compute @workgroup_size(4,4,4)
fn uvCommitSharpen(@builtin(global_invocation_id)gid:vec3u){
  let id=uvWorkId(gid);
  if(valid(id)&&!uvSharpenTileActive(id)){
    textureStore(volumeOut,id,vec4f(uvDustFloor(volume(id))));return;
  }
  if(!valid(id)){return;}let i=linearIndex(id);var terms:array<f32,6>;
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
${tileReach ? `/**
 * E7. The per-tile transport reach, measured AFTER Sec. 3.3's extension.
 *
 * Why per-tile. E3's reach is derived from the largest backward displacement
 * anywhere in the DOMAIN, so on figure 7 one splash cell at 61 cells a step
 * dilates the whole live set by sixteen tiles: 92,880 of 262,144 tiles at the
 * impact step, for 9,244 tiles of actual liquid. The predicate it is standing
 * in for is local -- a receiver's trace departs at most as far as the velocity
 * it actually samples -- so the domain maximum is only ever needed where the
 * domain maximum is.
 *
 * Why it cannot be measured here. The first attempt measured each tile's
 * start-of-step fine velocity and decided TRANSPORT in the same scan. That is
 * the wrong field at the wrong time: outside the FINE tiles uvTrace samples
 * uvCoarseVelocityComponent, the 4h face table the extension publishes from
 * its own ceil(n/4) level, and inside them it samples the post-extension fine
 * field. The extension is a narrow-band front plus a CM11b down/up pyramid, so
 * a far-air cell's traced velocity is a block AVERAGE that can come from
 * anywhere in the domain; no ball of start-of-step fine velocities bounds it.
 * At 256^3 that arm diverged from the dense control at frame 13.
 *
 * What these scans carry instead. The head-of-step dilation keeps the control's
 * classes exactly -- bit 4 is still the domain-wide dilation the extension and
 * DONORS are entitled to -- and computes ONE extra field alongside them: a
 * separable Chebyshev DISTANCE, in tiles, to the nearest transport SEED. It is
 * a six-bit saturating integer above the four class bits while the scan runs,
 * and uvTwoLevelDilateZ parks it in the spare bits of the DONORS plane word so
 * it survives the extension. Every other reader still sees three class bits.
 * The decision itself is made after the extension, by the four passes at the
 * end of this module.
 */
const UV_TILE_FAR:i32=63;
fn uvTilePack(cls:i32,dist:i32)->i32{return (cls&15)|((clamp(dist,0,63))<<4);}
fn uvTileDist(value:i32)->i32{return (value>>4)&63;}
/** E3's own m(D): the whole tiles a ceil(D)+1 cell departure can cross, plus
 * the configured margin, biased by eight so negative w can mean "off". */
fn uvTileRequiredReach(displacement:f32)->i32{
  if(params.twoLevel.w<0.0){return 0;}
  let required=i32(ceil((ceil(max(displacement,0.0))+1.0)/4.0));
  return clamp(required+i32(params.twoLevel.w)-8,0,16);
}
/** DONORS plane word while E7 runs: bit 3 is DONORS itself, bits 4..9 the
 * head-of-step distance to the nearest transport seed, bits 16..21 the x scan
 * of the post-extension displacement. */
fn uvPlaneScan(value:i32)->i32{return (value>>16)&63;}
` : ""}
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
// FINE/SHELL seeds: liquid at or above the dust floor, any solid/terrain share,
// a source this step, or a vertex on the liquid side of the 4h band. TRANSPORT
// excludes solid-only seeds: static boundaries do not create fluid or gamma.
// Both sets retain phi support because gather also computes uvTarget. Partial
// open fraction covers rigid bodies and terrain without disabling the map.
// The vertex test is ONE-SIDED (phi < band, not |phi| < band) so that every cell
// with a negative centre phi is in a seed tile: the centre is the mean of its
// eight vertices, so a negative centre forces a negative -- hence in-band --
// vertex. That is what makes "the FIM accurate band lies inside SHELL" a
// theorem rather than a property of the V test. It is a superset of E1's
// two-sided test; deep liquid is already seeded by V.
@compute @workgroup_size(4,4,4)
fn uvTwoLevelSeed(@builtin(global_invocation_id)gid:vec3u){
  let t=vec3i(gid);if(any(t>=uvCoarseDims())){return;}
  let slot=uvCoarseBase()+4u*uvCoarseIndex(t);
  // This pass stays dense -- it is one dispatch over ceil(n/4)^3 tiles and it
  // is what tells the solve window's own consumers where the liquid is -- but
  // a tile a whole tile clear of the previous window cannot seed: liquid
  // cannot have appeared there (the window covered every seed plus its reach),
  // so V is zero and every vertex is outside the 4h band. Only a source can
  // break that, and a source step is a host-known uniform condition.
  if(!uvTileInWindow(t)&&!uvStepHasExternalSource()){
    atomicStore(&sharpenDeposits[slot+3u],${tileReach ? "uvTilePack(0,UV_TILE_FAR)" : "0"});return;
  }
  let dust=select(params.tuning.z,1e-6,params.tuning.z<=0.0);
  let spacing=params.cellGravity.xyz;
  var seed=false;var transportSeed=false;var displacement=0.0;
  for(var z=0;z<4;z++){for(var y=0;y<4;y++){for(var x=0;x<4;x++){
    let id=4*t+vec3i(x,y,z);if(!valid(id)){continue;}
    // Include threshold equality: uvDustFloor discards strictly smaller values.
    let liquid=abs(volume(id))>=dust;
    let source=dropSource(id)>0.0||inflowSweptPlugSource(id,params.dimsDt.w)>0.0;
    if(liquid||source){seed=true;transportSeed=true;}
    if(uvOpen(id)<0.99999){seed=true;}
    // E3's required reach, in cells, along the axis that moves furthest. This
    // is the start-of-step velocity, which the extension then propagates into
    // the air as copies before transport traces it, so the domain maximum taken
    // here bounds every backward displacement this step.
    let step=abs(velocity(id))*params.dimsDt.w/spacing;
    displacement=max(displacement,max(step.x,max(step.y,step.z)));}}}
  atomicMax(&sharpenDeposits[uvCoarsePlane(2u)+2u],bitcast<i32>(displacement));
  let h=params.cellGravity.xyz;let band=4.0*max(h.x,max(h.y,h.z));
  let last=min(4*t+vec3i(4),dims());
  for(var z=4*t.z;z<=last.z;z++){for(var y=4*t.y;y<=last.y;y++){for(var x=4*t.x;x<=last.x;x++){
    if(textureLoad(uvPhiIn,vec3i(x,y,z),0).x<band){seed=true;transportSeed=true;}}}}
  ${tileReach ? `atomicStore(&sharpenDeposits[slot+3u],uvTilePack(select(0,3,seed)|select(0,4,transportSeed),
    select(UV_TILE_FAR,0,transportSeed)));` : `atomicStore(&sharpenDeposits[slot+3u],select(0,3,seed)|select(0,4,transportSeed));`}
}
// Same census as uvTwoLevelSeed, with one 4x4x4 workgroup per tile.
// Adjacent lanes read adjacent cells/vertices instead of each lane serially
// scanning 64 cells and 125 vertices. Only OR and nonnegative max reductions
// are used, preserving the exact classes and travel bound.
var<workgroup> uvSeedFlags:atomic<u32>;
var<workgroup> uvSeedTravel:atomic<u32>;
@compute @workgroup_size(4,4,4)
fn uvTwoLevelSeedCooperative(@builtin(workgroup_id)group:vec3u,
 @builtin(local_invocation_id)local:vec3u,@builtin(local_invocation_index)lane:u32){
  let t=vec3i(group);if(any(t>=uvCoarseDims())){return;}
  let slot=uvCoarseBase()+4u*uvCoarseIndex(t);
  let included=uvTileInWindow(t)||uvStepHasExternalSource();
  if(lane==0u){atomicStore(&uvSeedFlags,0u);atomicStore(&uvSeedTravel,0u);}
  workgroupBarrier();
  let id=4*t+vec3i(local);var flags=0u;
  if(included&&valid(id)){
    let dust=select(params.tuning.z,1e-6,params.tuning.z<=0.0);
    if(abs(volume(id))>=dust||dropSource(id)>0.0||inflowSweptPlugSource(id,params.dimsDt.w)>0.0){flags=7u;}
    if(uvOpen(id)<0.99999){flags|=3u;}
    let step=abs(velocity(id))*params.dimsDt.w/params.cellGravity.xyz;
    atomicMax(&uvSeedTravel,bitcast<u32>(max(step.x,max(step.y,step.z))));
  }
  let extent=vec3u(min(vec3i(5),dims()-4*t+vec3i(1)));
  let count=extent.x*extent.y*extent.z;
  let h=params.cellGravity.xyz;let band=4.0*max(h.x,max(h.y,h.z));
  if(included){
    for(var index=lane;index<count;index+=64u){
      let p=4*t+vec3i(i32(index%extent.x),i32((index/extent.x)%extent.y),i32(index/(extent.x*extent.y)));
      if(textureLoad(uvPhiIn,p,0).x<band){flags|=7u;}
    }
  }
  if(flags!=0u){atomicOr(&uvSeedFlags,flags);}
  workgroupBarrier();
  if(lane==0u){
    let cls=i32(atomicLoad(&uvSeedFlags));
    let travel=atomicLoad(&uvSeedTravel);
    ${tileReach ? `atomicStore(&sharpenDeposits[slot+3u],uvTilePack(cls,
      select(UV_TILE_FAR,0,(cls&4)!=0)));` : `atomicStore(&sharpenDeposits[slot+3u],cls);`}
    atomicMax(&sharpenDeposits[uvCoarsePlane(2u)+2u],i32(travel));
  }
}
// Chebyshev dilation, separated into three axis scans. Each scan preserves
// FINE, SHELL and TRANSPORT bits with their independent support radii. Chebyshev
// balls compose, so SHELL is exactly FINE dilated by s. The pair of single-word
// planes above the table is the ping-pong; the z scan lands the final class
// back in the table so the sampler reads one place.
fn uvTwoLevelFineReach()->i32{return i32(max(params.physical.z,0.0));}
fn uvTwoLevelShellReach()->i32{return uvTwoLevelFineReach()+i32(max(params.twoLevel.x,1.0));}
/**
 * E3's live transport set, dilated from liquid/source/phi seeds (not solids) by
 * a reach this step MEASURED rather than one authored. uvTwoLevelSeed wrote the
 * domain maximum backward displacement D, in cells, into the counter word one
 * dispatch ago, so the predicate's own ceil(D)+1 cells is available here; a
 * seed cell is at most 4m cells from the boundary of its m-tile dilation, so
 * m = ceil((ceil(D)+1)/4) is exactly what the predicate asks for and
 * params.twoLevel.w carries the margin on top of it, in tiles, BIASED BY EIGHT
 * so that negative w can mean the experiment is off without colliding with a
 * negative margin -- which is not reachable from the panel and exists only so a
 * verification run can starve the set below its own predicate and watch the
 * front stall. Off returns zero, leaving the scan range, and therefore the FINE
 * and SHELL bits, exactly as they were. The cap keeps a blown-up velocity field
 * from turning the separated scan into a domain sweep; the host publishes
 * required against used, so a capped step is visible.
 */
fn uvTwoLevelTransportReach()->i32{
  if(params.twoLevel.w<0.0){return 0;}
  let d=bitcast<f32>(atomicLoad(&sharpenDeposits[uvCoarsePlane(2u)+2u]));
  let required=i32(ceil((ceil(max(d,0.0))+1.0)/4.0));
  return clamp(required+i32(params.twoLevel.w)-8,0,16);
}
/** Independent support bits survive the first scan; boundaries seed only FINE/SHELL. */
fn uvTwoLevelClassIn(plane:u32,q:vec3i)->i32{
  if(plane==2u){return atomicLoad(&sharpenDeposits[uvCoarseBase()+4u*uvCoarseIndex(q)+3u]);}
  return atomicLoad(&sharpenDeposits[uvCoarsePlane(plane)+uvCoarseIndex(q)]);
}
${donorTiles ? `/**
 * DONORS (bit 8, scan planes only): every tile a built row or one of its donors
 * can lie in. Rows are built on TRANSPORT = seeds dilated by m, and a donor is
 * within ceil(D)+1 cells of its row, which is the predicate's own m0 tiles; one
 * more tile is margin, because a short DONORS set reads a stale sum where a
 * short TRANSPORT set only stalls a front. Chebyshev balls compose, so the bit
 * is the transport seed dilated by m+m0+1 in the same scan. A displacement
 * past the transport cap makes the set the whole lattice.
 */
fn uvTwoLevelDonorReach()->i32{
  if(params.twoLevel.w<0.0){return 0;}
  let d=bitcast<f32>(atomicLoad(&sharpenDeposits[uvCoarsePlane(2u)+2u]));
  let required=i32(ceil((ceil(max(d,0.0))+1.0)/4.0));
  if(!(required<=16)){return 1<<20;}
  return uvTwoLevelTransportReach()+required+1;
}` : ""}
fn uvTwoLevelDilate(previous:u32,axis:u32,t:vec3i)->i32{
  let c=uvCoarseDims();let k=uvTwoLevelFineReach();let s=uvTwoLevelShellReach();
  let m=uvTwoLevelTransportReach();${donorTiles ? `let g=uvTwoLevelDonorReach();
  let r=min(max(max(s,m),g),max(c.x,max(c.y,c.z)));` : "let r=max(s,m);"}var hit=0;
  ${tileReach ? "var dist=UV_TILE_FAR;" : ""}
  for(var d=-r;d<=r;d++){var q=t;q[axis]+=d;if(q[axis]<0||q[axis]>=c[axis]){continue;}
    let value=uvTwoLevelClassIn(previous,q);${tileReach ? `
    // Separated Chebyshev distance transform: min over this axis of the
    // partial distance already accumulated against the offset walked to get
    // it. Chebyshev balls compose exactly the way the class bits do. Saturates
    // at UV_TILE_FAR outside the scan radius r, which is never below m, so a
    // saturated distance is always past any reach the post pass can ask for.` : "if(value==0){continue;}"}${tileReach ? `
    dist=min(dist,max(uvTileDist(value),abs(d)));` : ""}
    if((value&1)!=0&&d>=-k&&d<=k){hit|=1;}
    if((value&2)!=0&&d>=-s&&d<=s){hit|=2;}
    if((value&4)!=0&&d>=-m&&d<=m){hit|=4;}${donorTiles ? `
    if((value&select(8,4,previous==2u))!=0&&d>=-g&&d<=g){hit|=8;}` : ""}}
  return ${tileReach ? "uvTilePack(hit,dist)" : "hit"};
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
  let scanned=uvTwoLevelDilate(1u,2u,t);
  // E7 does NOT decide TRANSPORT here. The bit published is the control's
  // domain-wide dilation, because the extension and DONORS both run on it and
  // the field the transport trace samples does not exist yet. The narrowing
  // happens in uvTransportReachZ, after the extension has published it.
  let hit=scanned&7;
  ${donorTiles||tileReach ? `// The x plane is dead once the y scan has read it, so DONORS lands there and
  // the class word keeps exactly the three bits its other readers know. E7
  // parks its Chebyshev seed distance in the same word's bits 4..9: the only
  // reader of this plane is uvDonorSkip, which masks the DONORS bit.
  atomicStore(&sharpenDeposits[uvCoarsePlane(0u)+uvCoarseIndex(t)],${donorTiles ? "(scanned&8)" : "0"}${tileReach ? "|(uvTileDist(scanned)<<4)" : ""});` : ""}
  atomicStore(&sharpenDeposits[uvCoarseBase()+4u*uvCoarseIndex(t)+3u],hit);
  if((hit&1)!=0){atomicAdd(&reductions[7],1u);}
  if((hit&2)!=0){atomicAdd(&sharpenDeposits[uvCoarsePlane(2u)],1);}
  if((hit&4)!=0){atomicAdd(&sharpenDeposits[uvCoarsePlane(2u)+1u],1);}
}
${tileReach ? `
/**
 * E7's decision, taken AFTER Sec. 3.3's extension has published the field
 * uvTrace actually samples, and before anything builds a transport row.
 *
 * uvTransportReachMeasure prices one tile: the largest |v|*dt/h, over
 * components, among the MAC faces the trilinear samplers read for a point
 * inside it -- the post-extension fine faces transportIn holds, and the 4h
 * faces uvCoarseFace holds for the arm outside FINE. It is rounded UP to whole
 * cells (max and ceil commute, so the scans below stay exact on six-bit
 * integers) and clamped to E3's own domain maximum, which is the premise the
 * control already runs on: the start-of-step maximum bounds every backward
 * displacement this step. Three separable MAX scans then dilate it over a ball
 * of m0+1 tiles, where m0 = ceil((ceil(D)+1)/4) is E3's required reach for the
 * domain maximum D, and uvTransportReachZ finally clears class bit 4 wherever
 * the tile's own distance to a transport seed exceeds m(that ball maximum).
 *
 * EXACTNESS. Write A(t) for the ball maximum at tile t, in cells, and take any
 * receiver cell p in t.
 *
 *  - The first sample is at p itself. sampleVelocityComponent reads faces
 *    within one cell of p either way, so its footprint lies in tiles t-1..t+1;
 *    the coarse arm reads uvCoarseFace over base..base+1 with base in
 *    t-1..t, so it lies there too. The ball has radius m0+1 >= 1, so both are
 *    inside it: |v(p)|*dt/h <= A(t), componentwise.
 *  - The RK2 midpoint is therefore at most A(t)/2 cells from p, so its tile is
 *    within ceil(ceil(A/2)/4) <= m0 tiles of t and its own sampling footprint
 *    within m0+1. Hence |v(mid)|*dt/h <= A(t) as well, and the endpoint is at
 *    most ceil(A(t)) cells from p. uvTrace's half-cell solid walk only returns
 *    a point on that same segment, and both clamps only shorten it.
 *  - uvBuildEdges takes base = floor(end - 1/2) and donors base..base+1, so a
 *    donor cell is within ceil(A(t))+1 cells of p, i.e. within
 *    ceil((ceil(A(t))+1)/4) = m(A(t)) TILES of t. That is exactly the reach
 *    uvTileRequiredReach returns (plus the configured margin, which only
 *    enlarges it).
 *  - The clamp to the domain maximum gives A(t) <= ceil(D) for every t, which
 *    is what makes m0+1 a sufficient ball radius rather than a circular one.
 *
 * So if dist(t) > m(A(t)), no donor of any row in t lies in a transport-seed
 * tile. Every cell outside a seed tile holds |V| below the dust floor, and the
 * floor stores exact zero, so the nine-term gather would sum to zero -- which
 * is what uvGather stores for a skipped tile. Gamma is zero there for the same
 * reason: a vertex of t with phi < 0 after advection had phi < 4h before it, at
 * a point at most ceil(A(t)) cells away, so that point's tile is a transport
 * seed and dist(t) <= m(A(t)). The post-advect corrections cannot break that:
 * every reach is at least ceil(1/4) = 1 tile, uvSeedPhi needs incident cells
 * averaging a quarter full (which are liquid, hence seeds, one tile away),
 * uvAgreementShift needs |phi| < 2h at a vertex whose phi is at least 4h, and
 * a drop or inflow cell seeds its own tile at distance zero.
 *
 * Why DONORS stays global. uvNormalizeDonors reads the accumulated column sum
 * of every donor a BUILT row samples, and the built set only shrinks here, so
 * the domain-wide DONORS set published at the head of the step is a superset
 * of what the narrowed rows address. A wider decode is redundant work, never a
 * stale read; a narrower one would be a stale read.
 */
fn uvPostTraceReach()->i32{
  let d=bitcast<f32>(atomicLoad(&sharpenDeposits[uvCoarsePlane(2u)+2u]));
  return i32(ceil((ceil(max(d,0.0))+1.0)/4.0));
}
/** Whole cells of E3's domain maximum: the cap every tile's own measurement is
 * held to, so the ball radius above is not self-referential. */
fn uvPostTravelCap()->i32{
  let d=bitcast<f32>(atomicLoad(&sharpenDeposits[uvCoarsePlane(2u)+2u]));
  return i32(clamp(ceil(max(d,0.0)),0.0,63.0));
}
var<workgroup> uvPostTravel:atomic<u32>;
@compute @workgroup_size(4,4,4)
fn uvTransportReachMeasure(@builtin(workgroup_id)group:vec3u,
 @builtin(local_invocation_id)local:vec3u,@builtin(local_invocation_index)lane:u32){
  let t=vec3i(group);if(any(t>=uvCoarseDims())){return;}
  if(lane==0u){atomicStore(&uvPostTravel,0u);}
  workgroupBarrier();
  let spacing=params.cellGravity.xyz;let id=4*t+vec3i(local);
  var travel=0.0;
  // The post-extension fine faces, which sampleVelocityComponent interpolates
  // inside a FINE tile. The padded ring outside the lattice is never written
  // and therefore reads zero, so it cannot raise any maximum.
  if(valid(id)){
    let v=abs(textureLoad(transportIn,id+vec3i(1),0).xyz)*params.dimsDt.w/spacing;
    travel=max(v.x,max(v.y,v.z));
  }
  // ...and the 4h faces it interpolates outside one.
  if(lane<3u){travel=max(travel,abs(uvCoarseFace(t,lane))*params.dimsDt.w/spacing[lane]);}
  atomicMax(&uvPostTravel,u32(clamp(ceil(travel),0.0,63.0)));
  workgroupBarrier();
  if(lane==0u){
    atomicStore(&sharpenDeposits[uvCoarsePlane(1u)+uvCoarseIndex(t)],
      min(i32(atomicLoad(&uvPostTravel)),uvPostTravelCap()));
  }
}
fn uvTransportReachIn(plane:u32,q:vec3i)->i32{
  let value=atomicLoad(&sharpenDeposits[uvCoarsePlane(plane)+uvCoarseIndex(q)]);
  return select(value,uvPlaneScan(value),plane==0u);
}
/** One axis of the ball maximum. Chebyshev balls compose, so three of these
 * are the max over the cube of radius m0+1 tiles. */
fn uvTransportReachScan(previous:u32,axis:u32,t:vec3i)->i32{
  let c=uvCoarseDims();
  let r=min(uvPostTraceReach()+1,max(c.x,max(c.y,c.z)));
  var travel=0;
  for(var d=-r;d<=r;d++){var q=t;q[axis]+=d;if(q[axis]<0||q[axis]>=c[axis]){continue;}
    travel=max(travel,uvTransportReachIn(previous,q));}
  return travel;
}
@compute @workgroup_size(4,4,4)
fn uvTransportReachX(@builtin(global_invocation_id)gid:vec3u){
  let t=vec3i(gid);if(any(t>=uvCoarseDims())){return;}
  let slot=uvCoarsePlane(0u)+uvCoarseIndex(t);
  let travel=uvTransportReachScan(1u,0u,t);
  // One writer per word, one dispatch apart: preserve DONORS and the seed
  // distance already parked in the low ten bits.
  atomicStore(&sharpenDeposits[slot],
    (atomicLoad(&sharpenDeposits[slot])&1023)|(clamp(travel,0,63)<<16));
}
@compute @workgroup_size(4,4,4)
fn uvTransportReachY(@builtin(global_invocation_id)gid:vec3u){
  let t=vec3i(gid);if(any(t>=uvCoarseDims())){return;}
  atomicStore(&sharpenDeposits[uvCoarsePlane(1u)+uvCoarseIndex(t)],uvTransportReachScan(0u,1u,t));
}
@compute @workgroup_size(4,4,4)
fn uvTransportReachZ(@builtin(global_invocation_id)gid:vec3u){
  let t=vec3i(gid);if(any(t>=uvCoarseDims())){return;}
  let index=uvCoarseIndex(t);
  let travel=uvTransportReachScan(1u,2u,t);
  let word=atomicLoad(&sharpenDeposits[uvCoarseBase()+4u*index+3u]);
  var hit=word;
  // Only ever CLEARS bit 4, so the live set is a subset of the control's. It
  // stands down entirely once E3's own reach is capped, because the control is
  // then short of its own predicate and there is nothing to be exact to.
  if((word&4)!=0&&uvPostTraceReach()<=16
    &&uvTileDist(atomicLoad(&sharpenDeposits[uvCoarsePlane(0u)+index]))>uvTileRequiredReach(f32(travel))){
    hit=word&3;
  }
  atomicStore(&sharpenDeposits[uvCoarseBase()+4u*index+3u],hit);
  // The host cleared this counter again before the measure pass: the live-set
  // telemetry must price the set the transport work list is built from.
  if((hit&4)!=0){atomicAdd(&sharpenDeposits[uvCoarsePlane(2u)+1u],1);}
}
` : ""}
// C. Show V where phi offers none: CM12 Sec. 3.8, restricted to cells more
// than 1.5 cells from phi's surface so the published field is bit-identical
// wherever phi describes the liquid. Presentation only: the solver never reads
// the published half. A 3^3 cluster under a quarter cell is dust, not drawn.
// Density mode is Sec. 3.8's rho'' = V / min(max(gamma,theta),1) with gamma the
// 3^3 box mean of 2 min(V,1/2) (the paper blurs with sigma = 2h) and theta =
// 0.01, capped at one. A uniform smear under 1/2 maps to exactly 0.5 there --
// the paper's flicker, and on a crown splash a haze of some 30k cells -- so
// only V standing a fifth above its box mean (rho'' >= 0.6) is drawn: a sheet
// up to two cells thick shows, a haze does not. Sphere mode draws each cell's
// 3^3 cluster as a ball of the cluster's own volume about its V centroid, so
// what is seen is what is held, and only where the cluster is as compact as a
// ball: its V-weighted second moment about the centroid may exceed a ball's
// 3r^2/5 by at most 3/4 cell^2 (a drop spread over a 2^3 block passes, a
// smear under half full fails). It also writes a dry cell beside V, whose
// value places the 0.5 crossing. Only orphan cells (centre phi > 1.5h) join a
// cluster: V smears a cell or two past any healthy surface, and a cell just
// outside that skin would otherwise gather the body's own V and draw a bump.
fn uvOrphanRender(id:vec3i,spheres:bool)->f32{
  var near=volume(id)>0.0;
  if(spheres){for(var axis=0u;axis<3u&&!near;axis++){for(var side=-1;side<=1;side+=2){
    var n=id;n[axis]+=side;if(volume(n)>0.0){near=true;}}}}
  if(!near){return -1e30;}
  let far=1.5*min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  var corners:array<f32,64>;
  for(var k=0u;k<64u;k++){let o=vec3i(i32(k&3u),i32((k>>2u)&3u),i32(k>>4u))-vec3i(1);
    corners[k]=textureLoad(uvPhiIn,clamp(id+o,vec3i(0),dims()),0).x;}
  var mass=0.0;var gamma=0.0;var centroid=vec3f(0);var moment=0.0;
  for(var dz=-1;dz<=1;dz++){for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){
    let c=id+vec3i(dx,dy,dz);if(!valid(c)){continue;}
    var centre=0.0;for(var k=0u;k<8u;k++){let o=vec3i(dx,dy,dz)+vec3i(1)+uvCorner(k);centre+=corners[o.x+4*o.y+16*o.z];}
    if(centre<=8.0*far){continue;}
    let v=volume(c);let d=vec3f(f32(dx),f32(dy),f32(dz));
    mass+=v;gamma+=2.0*min(v,0.5);centroid+=v*d;moment+=v*dot(d,d);}}}
  if(mass<0.25){return -1e30;}
  if(spheres){let flat=UNIFORM_REFERENCE_DIMENSION==2u;
    let r=select(pow(0.75*mass/3.14159265,1.0/3.0),sqrt(mass/3.14159265),flat);
    let c=centroid/mass;let spread=moment/mass-dot(c,c);
    if(spread>select(0.6,0.5,flat)*r*r+0.75){return -1e30;}
    return 0.5+r-length(c);}
  let rho=volume(id)/min(max(gamma/select(27.0,9.0,UNIFORM_REFERENCE_DIMENSION==2u),0.01),1.0);
  if(rho<0.6){return -1e30;}
  return min(rho,1.0);
}
// F. Airborne V keeps its own motion. A cell is airborne when it holds liquid
// evidence (the tail of a resting surface would otherwise free-fall at g), lies
// more than 1.5 cells from phi's surface -- so it owns no pressure row and is
// no extension source -- and has no solid or domain wall within two cells. Its
// faces are extension sources, the projection keeps their predicted value and
// gravity acts on them without the 2h occupancy gate. keepab tore films
// because ballistic faces on a CONTACT film are not divergence free; free
// flight is the physically correct model for liquid in the air.
fn uvAirborneCell(id:vec3i)->bool{
  if(params.splashB.w<0.5||!valid(id)||volume(id)<=max(params.tuning.z,UV_LIQUID_EVIDENCE)){return false;}
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  if(uvPhi(vec3f(id)+vec3f(0.5))<=1.5*h){return false;}
  var lo=id-vec3i(2);var hi=id+vec3i(2);
  if(UNIFORM_REFERENCE_DIMENSION==2u){lo.z=id.z;hi.z=id.z;}
  if(any(lo<vec3i(0))||any(hi>=dims())){return false;}
  if(uvSolidFree()){return true;}
  for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
    if(cellOpenFraction(vec3i(x,y,z))<0.99999){return false;}}}}
  return true;
}
fn uvAirborneAuthority(id:vec3i,rho:f32)->f32{
  return select(rho,max(rho,CM12_LIQUID_ISOVALUE+1e-3),uvAirborneCell(id));
}
@compute @workgroup_size(4,4,4)
fn uvPublish(@builtin(global_invocation_id)gid:vec3u){
  let id=activeId(gid);if(!valid(id)){return;}
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  var value=0.5-uvPhi(vec3f(id)+vec3f(0.5))/h;
  if(params.splash.z>0.5&&value<-1.0){value=max(value,uvOrphanRender(id,params.splash.z>1.5));}
  textureStore(volumeOut,id,vec4f(value));
  textureStore(gammaOut,id,vec4f(uvOpen(id)));
}

// A dedicated tail of the existing scratch buffer avoids another storage
// binding. [rate, partial-count, (positive volume, deficit volume)...].
// Rate is dimensionless and capped at one; RHS divides it by this step's dt.
fn uvBalanceBase()->u32{return 3u*cellCount();}
fn uvSurfaceDeficit(id:vec3i)->f32{
  let cap=cellOpenFraction(id);let v=volume(id);
  if(cap<=1e-5||v>cap||pressurePhi(id)>=0.0){return 0.0;}
  return max(0.0,textureLoad(gammaIn,id,0).x-v);
}
var<workgroup> uvBalanceSums:array<vec2f,64>;
var<workgroup> uvBalanceLive:atomic<u32>;
fn uvBalanceSum(l:u32){
  workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){
    if(l<stride){uvBalanceSums[l]+=uvBalanceSums[l+stride];}workgroupBarrier();
  }
}
@compute @workgroup_size(4,4,4)
fn uvBalanceMeasure(@builtin(global_invocation_id)gid:vec3u,
 @builtin(workgroup_id)w:vec3u,@builtin(num_workgroups)groups:vec3u,
 @builtin(local_invocation_index)l:u32){
  let id=activeId(gid);var sums=vec2f(0);
  if(valid(id)){
    let cap=cellOpenFraction(id);
    if(cap>1e-5&&pressurePhi(id)<0.0){
      sums=vec2f(min(0.5*max(0.0,volume(id)-cap),cap),uvSurfaceDeficit(id));
    }
  }
  ${uniformAbOn("deadgroups") ? `// A tile with no surplus and no deficit reduces sixty-four +0 pairs to +0
  // through seven barriers; one uniform load lets it publish that directly.
  // Bits, not values: a -0 anywhere takes the tree.
  let live=bitcast<vec2u>(sums);if((live.x|live.y)!=0u){atomicStore(&uvBalanceLive,1u);}
  if(workgroupUniformLoad(&uvBalanceLive)==0u){
    if(l==0u){uvBalanceSums[0]=vec2f(0);}
  }else{uvBalanceSums[l]=sums;uvBalanceSum(l);}` : "uvBalanceSums[l]=sums;uvBalanceSum(l);"}
  if(l==0u){
    let index=w.x+groups.x*(w.y+groups.y*w.z);let base=uvBalanceBase();
    atomicStore(&sharpenDeposits[base+2u+2u*index],bitcast<i32>(uvBalanceSums[0].x));
    atomicStore(&sharpenDeposits[base+3u+2u*index],bitcast<i32>(uvBalanceSums[0].y));
    if(index==0u){atomicStore(&sharpenDeposits[base+1u],i32(groups.x*groups.y*groups.z));}
  }
}
// A parallel first level replaces thousands of serial additions per lane in
// the single-workgroup reduction. Its outputs follow the live input records,
// so no workgroup can overwrite another workgroup's unread input.
override UV_BALANCE_TREE:bool=false;
@compute @workgroup_size(64)
fn uvBalanceReduceChunks(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
  let base=uvBalanceBase();let count=u32(atomicLoad(&sharpenDeposits[base+1u]));
  var sums=vec2f(0);let end=min(count,(w.x+1u)*1024u);
  for(var i=w.x*1024u+l;i<end;i+=64u){sums+=vec2f(
    bitcast<f32>(atomicLoad(&sharpenDeposits[base+2u+2u*i])),
    bitcast<f32>(atomicLoad(&sharpenDeposits[base+3u+2u*i])));}
  uvBalanceSums[l]=sums;uvBalanceSum(l);
  if(l==0u){let output=base+2u+2u*count+2u*w.x;
    atomicStore(&sharpenDeposits[output],bitcast<i32>(uvBalanceSums[0].x));
    atomicStore(&sharpenDeposits[output+1u],bitcast<i32>(uvBalanceSums[0].y));}
}
@compute @workgroup_size(64)
fn uvBalanceReduce(@builtin(local_invocation_index)l:u32){
  let base=uvBalanceBase();let records=u32(atomicLoad(&sharpenDeposits[base+1u]));
  let count=select(records,(records+1023u)/1024u,UV_BALANCE_TREE);
  let input=base+2u+select(0u,2u*records,UV_BALANCE_TREE);var sums=vec2f(0);
  for(var i=l;i<count;i+=64u){sums+=vec2f(
    bitcast<f32>(atomicLoad(&sharpenDeposits[input+2u*i])),
    bitcast<f32>(atomicLoad(&sharpenDeposits[input+1u+2u*i])));}
  uvBalanceSums[l]=sums;uvBalanceSum(l);
  if(l==0u){var rate=0.0;if(uvBalanceSums[0].y>0.0){rate=min(1.0,uvBalanceSums[0].x/uvBalanceSums[0].y);}
    atomicStore(&sharpenDeposits[base],bitcast<i32>(rate));}
}
`;
