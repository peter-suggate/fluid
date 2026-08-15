import { createCm12NumericsWGSL } from "../../core/cm12-numerics";

/**
 * Compact, GPU-resident Sparse CM12 frame kernel.
 *
 * Topology is packed once at construction.  Every per-frame field, reduction,
 * pressure iteration, projection, diagnostic field, and dense presentation
 * publication remains in device storage.  The host only writes the small
 * timestep uniform and submits the pre-sized dispatch sequence.
 */
export const webgpuSparseCM12ResidentWGSL = /* wgsl */ `
${createCm12NumericsWGSL()}

const INVALID:u32=0xffffffffu;
const WORKGROUP:u32=64u;
const ACTIVITY_HEADER_WORDS:u32=8u;
const ACTIVITY_RECORD_WORDS:u32=8u;
const ACTIVITY_FIXED:f32=65536.0;

struct Params {
  counts:vec4u,             // cell, row, incidence, dense
  dimensions:vec4u,
  topologyOffsets:vec4u,    // cells, rows, terms, incidenceOffsets
  topologyOffsets2:vec4u,   // incidences, owner records, brick records, background owner
  stateOffsets0:vec4u,      // density A/B, gamma A/B
  stateOffsets1:vec4u,      // cell velocity A/B, face A/B
  stateOffsets2:vec4u,      // pressure, rhs, diagonal, liquid
  stateOffsets3:vec4u,      // theta, residual, preconditioned, direction
  stateOffsets4:vec4u,      // applied, divergence, presentation brick wet, reserved
  stateOffsets5:vec4u,      // sharpening delta/accepted fraction, reserved
  frame:vec4f,              // dt, finest cell metres, pressure scale, parity
  acceleration:vec4f,       // finest cells / second^2
  dispatch:vec4u,           // cell workgroups, row workgroups, pcg iterations, brick count
  injectionCenter:vec4f,
  injectionRadius:vec4f,
}

@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>topology:array<u32>;
@group(0)@binding(2)var<storage,read_write>state:array<f32>;
@group(0)@binding(3)var<storage,read_write>partials:array<vec2f>;
@group(0)@binding(4)var<storage,read_write>scalars:array<f32>;
@group(0)@binding(5)var densityTexture:texture_storage_3d<r32float,write>;
@group(0)@binding(6)var levelSetTexture:texture_storage_3d<r32float,write>;
@group(0)@binding(7)var ownerTexture:texture_storage_3d<rg32uint,write>;
@group(0)@binding(8)var velocityTexture:texture_storage_3d<rgba32float,write>;
@group(0)@binding(9)var pressureTexture:texture_storage_3d<r32float,write>;
@group(0)@binding(10)var divergenceTexture:texture_storage_3d<r32float,write>;
@group(0)@binding(11)var<storage,read_write>conditioning:array<atomic<i32>>;
// This policy/history arena is the only output of the first dynamic-resolution
// rung. It is device-resident and deliberately disjoint from accepted physics
// and topology until candidate transfer and rollback are implemented.
@group(0)@binding(12)var<storage,read_write>activity:array<atomic<u32>>;

fn tf(index:u32)->f32{return bitcast<f32>(topology[index]);}
fn sourceDensity()->u32{return select(p.stateOffsets0.x,p.stateOffsets0.y,p.frame.w>0.5);}
fn destinationDensity()->u32{return select(p.stateOffsets0.y,p.stateOffsets0.x,p.frame.w>0.5);}
fn sourceGamma()->u32{return select(p.stateOffsets0.z,p.stateOffsets0.w,p.frame.w>0.5);}
fn destinationGamma()->u32{return select(p.stateOffsets0.w,p.stateOffsets0.z,p.frame.w>0.5);}
fn sourceCellVelocity()->u32{return select(p.stateOffsets1.x,p.stateOffsets1.y,p.frame.w>0.5);}
fn destinationCellVelocity()->u32{return select(p.stateOffsets1.y,p.stateOffsets1.x,p.frame.w>0.5);}
fn sourceFaceVelocity()->u32{return select(p.stateOffsets1.z,p.stateOffsets1.w,p.frame.w>0.5);}
fn destinationFaceVelocity()->u32{return select(p.stateOffsets1.w,p.stateOffsets1.z,p.frame.w>0.5);}

fn cellBase(id:u32)->u32{return p.topologyOffsets.x+id*12u;}
fn cellVolume(id:u32)->f32{return tf(cellBase(id)+3u);}
fn cellMinimumWidth(id:u32)->f32{
  let b=cellBase(id);return min(tf(b+4u),min(tf(b+5u),tf(b+6u)));
}
fn rowBase(id:u32)->u32{return p.topologyOffsets.y+id*12u;}
fn rowTermOffset(id:u32)->u32{return topology[rowBase(id)];}
fn rowTermCount(id:u32)->u32{return topology[rowBase(id)+1u];}
fn rowAxis(id:u32)->u32{return topology[rowBase(id)+2u];}
fn rowKind(id:u32)->u32{return topology[rowBase(id)+3u];}
fn rowDualWeight(id:u32)->f32{return tf(rowBase(id)+4u);}
fn rowArea(id:u32)->f32{return tf(rowBase(id)+5u);}
fn rowDistance(id:u32)->f32{return tf(rowBase(id)+6u);}
fn rowExteriorPhi(id:u32)->f32{return tf(rowBase(id)+7u);}
fn rowCenter(id:u32)->vec3f{let b=rowBase(id);return vec3f(tf(b+8u),tf(b+9u),tf(b+10u));}
fn termCell(index:u32)->u32{return topology[p.topologyOffsets.z+2u*index];}
fn termCoefficient(index:u32)->f32{return tf(p.topologyOffsets.z+2u*index+1u);}
fn incidenceBegin(cell:u32)->u32{return topology[p.topologyOffsets.w+cell];}
fn incidenceEnd(cell:u32)->u32{return topology[p.topologyOffsets.w+cell+1u];}
fn incidenceRow(index:u32)->u32{return topology[p.topologyOffsets2.x+2u*index];}
fn incidenceTerm(index:u32)->u32{return topology[p.topologyOffsets2.x+2u*index+1u];}
fn isLiquid(cell:u32)->bool{return state[p.stateOffsets2.w+cell]>0.5;}

fn ownerCellAt(q:vec3i)->u32{
  if(any(q<vec3i(0))||any(q>=vec3i(p.dimensions.xyz))){return INVALID;}
  let dense=u32(q.x)+p.dimensions.x*(u32(q.y)+p.dimensions.y*u32(q.z));
  return topology[p.topologyOffsets2.y+4u*dense];
}

fn presentationOwnerCellAt(q:vec3i)->u32{
  if(any(q<vec3i(0))||any(q>=vec3i(p.dimensions.xyz))){return INVALID;}
  let dense=u32(q.x)+p.dimensions.x*(u32(q.y)+p.dimensions.y*u32(q.z));
  let owner=p.topologyOffsets2.y+4u*dense;let cell=topology[owner];
  if(cell==INVALID){return INVALID;}
  let brick=topology[owner+3u];
  return select(INVALID,cell,brick<p.dispatch.w&&state[p.stateOffsets4.z+brick]>0.5);
}

fn presentationPhi(cell:u32)->f32{
  return (CM12_LIQUID_ISOVALUE-state[destinationDensity()+cell])*4.0*p.frame.y;
}

// Restrict authoritative rho by finest-cell volume. A coarse presentation
// stencil can therefore cross a 2:1 seam without choosing an arbitrary fine
// child, and its sample has exactly the mass of that virtual coarse cell.
fn restrictedPresentationDensity(lower:vec3i,cellScale:i32)->f32{
  var rho=0.0;
  for(var dz=0;dz<cellScale;dz+=1){for(var dy=0;dy<cellScale;dy+=1){for(var dx=0;dx<cellScale;dx+=1){
    let cell=presentationOwnerCellAt(lower+vec3i(dx,dy,dz));
    if(cell!=INVALID){rho+=state[destinationDensity()+cell];}
  }}}
  return rho/f32(cellScale*cellScale*cellScale);
}

// CM12 renders the rho=.5 contour. Evaluate its cell-centred rho with the
// paper's trilinear weights on the current owner's lattice. This reads the
// accepted rho every publication; the reconstruction follows moving liquid
// and the current coarse/fine ownership instead of a constructor-time mode.
fn interpolatedPresentationPhi(coordinate:vec3i,cellScale:i32)->f32{
  let scale=f32(cellScale);
  let shifted=(vec3f(coordinate)+vec3f(0.5))/scale-vec3f(0.5);
  let lower=vec3i(floor(shifted));let fraction=fract(shifted);var rho=0.0;
  let coarseDimensions=vec3i(p.dimensions.xyz)/cellScale;
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let offset=vec3i(dx,dy,dz);
    let coarse=clamp(lower+offset,vec3i(0),coarseDimensions-vec3i(1));
    let sample=restrictedPresentationDensity(coarse*cellScale,cellScale);
    let wx=select(1.0-fraction.x,fraction.x,dx==1);
    let wy=select(1.0-fraction.y,fraction.y,dy==1);
    let wz=select(1.0-fraction.z,fraction.z,dz==1);
    rho+=wx*wy*wz*sample;
  }}}
  return (CM12_LIQUID_ISOVALUE-rho)*4.0*p.frame.y;
}

fn sampleVelocity(position:vec3f)->vec3f{
  let probe=ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe!=INVALID){let b=cellBase(probe);spans=vec3f(tf(b+4u),tf(b+5u),tf(b+6u));}
  let clamped=clamp(position,0.5*spans,vec3f(p.dimensions.xyz)-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);let lower=vec3i(floor(shifted));let fraction=fract(shifted);
  var result=vec3f(0.0);
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let lattice=spans*(vec3f(lower+vec3i(dx,dy,dz))+vec3f(0.5));
    let cell=ownerCellAt(vec3i(floor(lattice)));if(cell==INVALID){continue;}
    let wx=select(1.0-fraction.x,fraction.x,dx==1);
    let wy=select(1.0-fraction.y,fraction.y,dy==1);
    let wz=select(1.0-fraction.z,fraction.z,dz==1);let at=destinationCellVelocity()+4u*cell;
    result+=wx*wy*wz*vec3f(state[at],state[at+1u],state[at+2u]);
  }}}return result;
}

fn traceCharacteristic(position:vec3f,direction:f32)->vec3f{
  let initial=sampleVelocity(position);
  let substeps=clamp(i32(ceil(length(initial)*p.frame.x)),1,16);
  let subDt=p.frame.x/f32(substeps);var traced=position;
  let lower=vec3f(0.5);let upper=vec3f(p.dimensions.xyz)-vec3f(0.5);
  for(var step=0;step<substeps;step+=1){
    let first=sampleVelocity(traced);
    let midpoint=clamp(traced+direction*0.5*subDt*first,lower,upper);
    let candidate=traced+direction*subDt*sampleVelocity(midpoint);
    traced=clamp(candidate,lower,upper);
  }
  return traced;
}
fn traceDeparture(position:vec3f)->vec3f{return traceCharacteristic(position,-1.0);}
fn traceArrival(position:vec3f)->vec3f{return traceCharacteristic(position,1.0);}

struct TransportTerm{cell:u32,weight:f32}
fn transportTerm(position:vec3f,corner:u32)->TransportTerm{
  let probe=ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe!=INVALID){let b=cellBase(probe);spans=vec3f(tf(b+4u),tf(b+5u),tf(b+6u));}
  let shifted=position/spans-vec3f(0.5);let lower=vec3i(floor(shifted));let fraction=fract(shifted);
  let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  let lattice=spans*(vec3f(lower+offset)+vec3f(0.5));
  let cell=ownerCellAt(vec3i(floor(lattice)));
  let weight=select(1.0-fraction.x,fraction.x,offset.x==1)
    *select(1.0-fraction.y,fraction.y,offset.y==1)
    *select(1.0-fraction.z,fraction.z,offset.z==1);
  return TransportTerm(cell,select(0.0,weight,cell!=INVALID));
}

fn transportBeta(cell:u32)->f32{
  return f32(atomicLoad(&conditioning[cell]))/CM12_TRANSPORT_FIXED;
}

fn extrapolateTransportVelocity(id:u32,inputOffset:u32,outputOffset:u32){
  let input=inputOffset+4u*id;let output=outputOffset+4u*id;
  if(state[input+3u]>0.5){
    state[output]=state[input];state[output+1u]=state[input+1u];
    state[output+2u]=state[input+2u];state[output+3u]=1.0;return;
  }
  var velocity=vec3f(0.0);var weight=0.0;
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){
    let row=incidenceRow(at);let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let neighbor=termCell(term);
      if(neighbor==id){continue;}let source=inputOffset+4u*neighbor;
      if(state[source+3u]<=0.5){continue;}let w=abs(termCoefficient(term));
      velocity+=w*vec3f(state[source],state[source+1u],state[source+2u]);weight+=w;
    }
  }
  if(weight>0.0){velocity/=weight;}
  state[output]=velocity.x;state[output+1u]=velocity.y;state[output+2u]=velocity.z;
  state[output+3u]=select(0.0,1.0,weight>0.0);
}

@compute @workgroup_size(64)
fn initializeTransportVelocity(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=p.counts.x){return;}let input=sourceCellVelocity()+4u*id;
  let output=destinationCellVelocity()+4u*id;
  state[output]=state[input];state[output+1u]=state[input+1u];state[output+2u]=state[input+2u];
  state[output+3u]=select(0.0,1.0,state[sourceDensity()+id]>CM12_LIQUID_ISOVALUE);
}

@compute @workgroup_size(64)
fn extrapolateTransportVelocityToSource(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=p.counts.x){return;}
  extrapolateTransportVelocity(id,destinationCellVelocity(),sourceCellVelocity());
}

@compute @workgroup_size(64)
fn extrapolateTransportVelocityToDestination(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=p.counts.x){return;}
  extrapolateTransportVelocity(id,sourceCellVelocity(),destinationCellVelocity());
}

@compute @workgroup_size(64)
fn prepareTransportFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;if(row>=p.counts.y){return;}
  let departure=traceDeparture(rowCenter(row));
  let characteristic=sampleVelocity(departure)[rowAxis(row)];
  var touchesLiquid=false;let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var term=begin;term<end;term+=1u){
    touchesLiquid=touchesLiquid||state[sourceDensity()+termCell(term)]>CM12_LIQUID_ISOVALUE;
  }
  state[destinationFaceVelocity()+row]=select(characteristic,
    mix(characteristic,state[sourceFaceVelocity()+row],0.4),touchesLiquid);
}

@compute @workgroup_size(64)
fn injectLiquid(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=p.counts.x){return;}let b=cellBase(id);
  let center=vec3f(tf(b),tf(b+1u),tf(b+2u));
  let q=(center-p.injectionCenter.xyz)/max(p.injectionRadius.xyz,vec3f(1e-6));
  let signed=length(q)-1.0;
  let coverage=clamp(0.5-signed*min(p.injectionRadius.x,
    min(p.injectionRadius.y,p.injectionRadius.z))/max(cellMinimumWidth(id),1e-6),0.0,1.0);
  state[p.stateOffsets0.x+id]=max(state[p.stateOffsets0.x+id],coverage);
  state[p.stateOffsets0.y+id]=max(state[p.stateOffsets0.y+id],coverage);
  if(coverage>0.0){state[p.stateOffsets0.z+id]=1.0;state[p.stateOffsets0.w+id]=1.0;}
}

// CM12 Sec. 3.4 steps 1-3. Backward-advect cumulative gamma, then scatter
// gamma_i w^-_li into each donor's volume-weighted beta column.
@compute @workgroup_size(64)
fn traceGammaAndBeta(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=p.counts.x){return;}let b=cellBase(id);
  let departure=traceDeparture(vec3f(tf(b),tf(b+1u),tf(b+2u)));
  var visible=0.0;var sampledGamma=0.0;
  for(var corner=0u;corner<8u;corner+=1u){let term=transportTerm(departure,corner);
    visible+=term.weight;if(term.cell!=INVALID){sampledGamma+=term.weight*state[sourceGamma()+term.cell];}}
  let advectedGamma=cm12ConditionedGamma(sampledGamma,visible);
  state[destinationGamma()+id]=advectedGamma;if(visible<=1e-9){return;}
  for(var corner=0u;corner<8u;corner+=1u){let term=transportTerm(departure,corner);
    if(term.cell==INVALID||term.weight<=0.0){continue;}
    let coefficient=advectedGamma*term.weight/visible;
    let contribution=cm12VolumeWeightedBetaContribution(
      cellVolume(id),cellVolume(term.cell),coefficient);
    atomicAdd(&conditioning[term.cell],i32(round(contribution*CM12_TRANSPORT_FIXED)));
  }
}

// CM12 Sec. 3.4 steps 6-7. Every deficient donor returns its missing column
// weight along the forward characteristic. The fixed-point scatters are
// deterministic and keep rho and gamma transfers paired.
@compute @workgroup_size(64)
fn scatterDensityDeficit(@builtin(global_invocation_id)gid:vec3u){
  let donor=gid.x;if(donor>=p.counts.x){return;}
  let deficit=max(0.0,1.0-transportBeta(donor));
  if(deficit<=1.0/CM12_TRANSPORT_FIXED){return;}let b=cellBase(donor);
  let arrival=traceArrival(vec3f(tf(b),tf(b+1u),tf(b+2u)));
  var visible=0.0;for(var corner=0u;corner<8u;corner+=1u){
    visible+=transportTerm(arrival,corner).weight;}
  if(visible<=1e-9){
    atomicAdd(&conditioning[p.counts.x+donor],i32(round(
      state[sourceDensity()+donor]*deficit*CM12_TRANSPORT_FIXED)));
    atomicAdd(&conditioning[2u*p.counts.x+donor],i32(round(
      state[sourceGamma()+donor]*deficit*CM12_TRANSPORT_FIXED)));return;
  }
  for(var corner=0u;corner<8u;corner+=1u){let term=transportTerm(arrival,corner);
    if(term.cell==INVALID||term.weight<=0.0){continue;}let normalized=term.weight/visible;
    let densityTransfer=cm12VolumeScaledDeficitTransfer(state[sourceDensity()+donor],
      cellVolume(donor),cellVolume(term.cell),deficit,normalized);
    let gammaTransfer=cm12VolumeScaledDeficitTransfer(state[sourceGamma()+donor],
      cellVolume(donor),cellVolume(term.cell),deficit,normalized);
    atomicAdd(&conditioning[p.counts.x+term.cell],i32(round(densityTransfer*CM12_TRANSPORT_FIXED)));
    atomicAdd(&conditioning[2u*p.counts.x+term.cell],i32(round(gammaTransfer*CM12_TRANSPORT_FIXED)));
  }
}

// CM12 Sec. 3.4 steps 4-5 plus the forward-deficit resolve. Scaling each
// donor by max(1,beta_l) clamps the column without materializing A.
@compute @workgroup_size(64)
fn gatherConservativeDensity(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=p.counts.x){return;}let b=cellBase(id);
  let departure=traceDeparture(vec3f(tf(b),tf(b+1u),tf(b+2u)));
  let advectedGamma=state[destinationGamma()+id];var visible=0.0;
  for(var corner=0u;corner<8u;corner+=1u){visible+=transportTerm(departure,corner).weight;}
  var rhoNext=0.0;var gammaNext=0.0;
  if(visible>1e-9){for(var corner=0u;corner<8u;corner+=1u){
    let term=transportTerm(departure,corner);if(term.cell==INVALID||term.weight<=0.0){continue;}
    let coefficient=cm12ConditionedRowCoefficient(
      advectedGamma,term.weight/visible,transportBeta(term.cell));
    rhoNext+=coefficient*state[sourceDensity()+term.cell];gammaNext+=coefficient;
  }}
  rhoNext+=f32(atomicLoad(&conditioning[p.counts.x+id]))/CM12_TRANSPORT_FIXED;
  gammaNext+=f32(atomicLoad(&conditioning[2u*p.counts.x+id]))/CM12_TRANSPORT_FIXED;
  if(rhoNext<1e-5){gammaNext=1.0;}
  state[destinationDensity()+id]=max(0.0,rhoNext);
  state[destinationGamma()+id]=max(0.0,gammaNext);
}

fn diffuseGammaAxis(cell:u32,axis:u32,inputRho:u32,inputGamma:u32,
 outputRho:u32,outputGamma:u32){
  let ownRho=state[inputRho+cell];let ownGamma=state[inputGamma+cell];
  var rho=ownRho;var gamma=ownGamma;
  let scale=min(1.0,30.0*p.frame.x);
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
    let row=incidenceRow(at);if(rowAxis(row)!=axis){continue;}
    let own=termCoefficient(incidenceTerm(at));let begin=rowTermOffset(row);
    let end=begin+rowTermCount(row);var negativeCount=0.0;var positiveCount=0.0;
    for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
      negativeCount+=select(0.0,1.0,coefficient<0.0);
      positiveCount+=select(0.0,1.0,coefficient>0.0);}
    if(negativeCount==0.0||positiveCount==0.0){continue;}
    let area=rowArea(row)/(negativeCount*positiveCount);
    for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
      if(own*coefficient>=0.0){continue;}let neighbor=termCell(term);
      let conductedVolume=scale*min(area*cellMinimumWidth(cell),
        area*cellMinimumWidth(neighbor));
      let flux=cm12GammaDiffusionFluxInto(ownRho,ownGamma,
        state[inputRho+neighbor],state[inputGamma+neighbor],
        conductedVolume/cellVolume(cell));
      rho+=flux.x;gamma+=flux.y;
    }
  }
  state[outputRho+cell]=rho;state[outputGamma+cell]=gamma;
}

// CM12 Sec. 3.4 step 8. Average mirrored dimensional Gauss-Seidel orders,
// matching the CPU sparse operator while avoiding an arbitrary x/z bias.
@compute @workgroup_size(64)
fn diffuseGammaForwardX(@builtin(global_invocation_id)gid:vec3u){let cell=gid.x;
  if(cell<p.counts.x){diffuseGammaAxis(cell,0u,destinationDensity(),destinationGamma(),
    p.stateOffsets2.x,p.stateOffsets2.y);}}
@compute @workgroup_size(64)
fn diffuseGammaForwardY(@builtin(global_invocation_id)gid:vec3u){let cell=gid.x;
  if(cell<p.counts.x){diffuseGammaAxis(cell,1u,p.stateOffsets2.x,p.stateOffsets2.y,
    p.stateOffsets2.z,p.stateOffsets2.w);}}
@compute @workgroup_size(64)
fn diffuseGammaForwardZ(@builtin(global_invocation_id)gid:vec3u){let cell=gid.x;
  if(cell<p.counts.x){diffuseGammaAxis(cell,2u,p.stateOffsets2.z,p.stateOffsets2.w,
    p.stateOffsets2.x,p.stateOffsets2.y);}}
@compute @workgroup_size(64)
fn diffuseGammaReverseZ(@builtin(global_invocation_id)gid:vec3u){let cell=gid.x;
  if(cell<p.counts.x){diffuseGammaAxis(cell,2u,destinationDensity(),destinationGamma(),
    p.stateOffsets3.y,p.stateOffsets3.z);}}
@compute @workgroup_size(64)
fn diffuseGammaReverseY(@builtin(global_invocation_id)gid:vec3u){let cell=gid.x;
  if(cell<p.counts.x){diffuseGammaAxis(cell,1u,p.stateOffsets3.y,p.stateOffsets3.z,
    p.stateOffsets3.w,p.stateOffsets4.x);}}
@compute @workgroup_size(64)
fn diffuseGammaReverseX(@builtin(global_invocation_id)gid:vec3u){let cell=gid.x;
  if(cell<p.counts.x){diffuseGammaAxis(cell,0u,p.stateOffsets3.w,p.stateOffsets4.x,
    p.stateOffsets3.y,p.stateOffsets3.z);}}
@compute @workgroup_size(64)
fn averageGammaDiffusion(@builtin(global_invocation_id)gid:vec3u){let cell=gid.x;
  if(cell>=p.counts.x){return;}
  state[destinationDensity()+cell]=0.5*(state[p.stateOffsets2.x+cell]
    +state[p.stateOffsets3.y+cell]);
  state[destinationGamma()+cell]=0.5*(state[p.stateOffsets2.y+cell]
    +state[p.stateOffsets3.z+cell]);
}

struct SharpeningStats {
  maximumDifference:f32,
  negativeArea:vec3f,
  positiveArea:vec3f,
  negativeDensity:vec3f,
  positiveDensity:vec3f,
  negativeDistance:vec3f,
  positiveDistance:vec3f,
  uphillConductance:f32,
}

// Expand each composite G row into the same physical scalar subfaces used by
// sparse-atlas-surface-conditioning.ts. This retains the paper's adjacent-cell
// stencil at 4^3/8^3 seams instead of treating an aggregate port as one cell.
fn sharpeningStats(cell:u32)->SharpeningStats{
  var result:SharpeningStats;let rho=state[destinationDensity()+cell];
  result.maximumDifference=0.0;result.negativeArea=vec3f(0.0);
  result.positiveArea=vec3f(0.0);result.negativeDensity=vec3f(0.0);
  result.positiveDensity=vec3f(0.0);result.negativeDistance=vec3f(0.0);
  result.positiveDistance=vec3f(0.0);result.uphillConductance=0.0;
  let maximumDistance=2.1*cellMinimumWidth(cell);
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
    let row=incidenceRow(at);let own=termCoefficient(incidenceTerm(at));
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    var negativeCount=0.0;var positiveCount=0.0;
    for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
      negativeCount+=select(0.0,1.0,coefficient<0.0);
      positiveCount+=select(0.0,1.0,coefficient>0.0);}
    if(negativeCount==0.0||positiveCount==0.0){continue;}
    let area=rowArea(row)/(negativeCount*positiveCount);let distance=rowDistance(row);
    let axis=rowAxis(row);
    for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
      if(own*coefficient>=0.0){continue;}let neighbor=termCell(term);
      let neighborRho=state[destinationDensity()+neighbor];
      result.maximumDifference=max(result.maximumDifference,abs(rho-neighborRho));
      if(own<0.0){result.positiveArea[axis]+=area;
        result.positiveDensity[axis]+=area*neighborRho;
        result.positiveDistance[axis]+=area*distance;
      }else{result.negativeArea[axis]+=area;
        result.negativeDensity[axis]+=area*neighborRho;
        result.negativeDistance[axis]+=area*distance;}
      if(distance<=maximumDistance){
        result.uphillConductance+=area*max(0.0,(neighborRho-rho)/distance);
      }
    }
  }
  return result;
}

fn sharpeningDelta(cell:u32,stats:SharpeningStats)->f32{
  let rho=state[destinationDensity()+cell];
  // CM12 Eqs. 6-15 first integrate a unit-speed flux over a cell and then
  // divide the Godunov mass increment by cell volume. The resulting density
  // update is 3 dt |grad rho|, so distances stored in finest-cell units must
  // be converted back to metres. Multiplying by the local cell width here
  // made the old expression dimensionless and weakened sharpening by 1/h:
  // 10x on the all-coarse symmetric-expansion control and 20x when all fine.
  let pseudoTimeFineCells=3.0*p.frame.x/p.frame.y;
  var plusSquared=0.0;var minusSquared=0.0;
  for(var axis=0u;axis<3u;axis+=1u){
    var before=rho;var beforeDistance=1.0;
    if(stats.negativeArea[axis]>0.0){
      before=stats.negativeDensity[axis]/stats.negativeArea[axis];
      beforeDistance=stats.negativeDistance[axis]/stats.negativeArea[axis];
    }
    var after=rho;var afterDistance=1.0;
    if(stats.positiveArea[axis]>0.0){
      after=stats.positiveDensity[axis]/stats.positiveArea[axis];
      afterDistance=stats.positiveDistance[axis]/stats.positiveArea[axis];
    }
    let backward=-(rho-before)*pseudoTimeFineCells/beforeDistance;
    let forward=-(after-rho)*pseudoTimeFineCells/afterDistance;
    plusSquared+=max(max(backward,0.0)*max(backward,0.0),
      min(forward,0.0)*min(forward,0.0));
    minusSquared+=max(min(backward,0.0)*min(backward,0.0),
      max(forward,0.0)*max(forward,0.0));
  }
  let weight=cm12SharpeningWeight(rho,stats.maximumDifference);
  var delta=weight*sqrt(select(minusSquared,plusSquared,weight>=0.0));
  if(rho+delta<0.0||rho<1e-5){delta=-rho;}else if(rho>0.5){delta=0.0;}
  return min(0.0,delta);
}

// CM12 Sec. 3.5, Eqs. 4-17 and Algorithm 2. Removed air-side mass is
// scattered only over adjacent subfaces in the uphill density direction.
@compute @workgroup_size(64)
fn scatterSharpeningMass(@builtin(global_invocation_id)gid:vec3u){
  let cell=gid.x;if(cell>=p.counts.x){return;}let stats=sharpeningStats(cell);
  var delta=sharpeningDelta(cell,stats);
  if(stats.uphillConductance<=1e-30){delta=0.0;}
  state[p.stateOffsets5.x+cell]=delta;if(delta>=0.0){return;}
  let rho=state[destinationDensity()+cell];let removed=-delta*cellVolume(cell);
  let maximumDistance=2.1*cellMinimumWidth(cell);
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
    let row=incidenceRow(at);let own=termCoefficient(incidenceTerm(at));
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    var negativeCount=0.0;var positiveCount=0.0;
    for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
      negativeCount+=select(0.0,1.0,coefficient<0.0);
      positiveCount+=select(0.0,1.0,coefficient>0.0);}
    if(negativeCount==0.0||positiveCount==0.0||rowDistance(row)>maximumDistance){continue;}
    let area=rowArea(row)/(negativeCount*positiveCount);let distance=rowDistance(row);
    for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
      if(own*coefficient>=0.0){continue;}let neighbor=termCell(term);
      let conductance=area*max(0.0,
        (state[destinationDensity()+neighbor]-rho)/distance);
      if(conductance<=0.0){continue;}
      let offered=removed*conductance/stats.uphillConductance;
      atomicAdd(&conditioning[3u*p.counts.x+neighbor],
        i32(round(offered*CM12_TRANSPORT_FIXED)));
    }
  }
}

// Resolve all simultaneous scatters with one receiver scale. This is the
// parallel form of ScatterValue's bounded deposition: it cannot create rho>1,
// and rejected mass is returned to its source in finalizeSharpening.
@compute @workgroup_size(64)
fn acceptSharpeningMass(@builtin(global_invocation_id)gid:vec3u){
  let cell=gid.x;if(cell>=p.counts.x){return;}
  let incoming=f32(atomicLoad(&conditioning[3u*p.counts.x+cell]))/CM12_TRANSPORT_FIXED;
  let base=state[destinationDensity()+cell]+state[p.stateOffsets5.x+cell];
  let capacity=max(0.0,(1.0-base)*cellVolume(cell));
  state[p.stateOffsets5.y+cell]=select(0.0,min(1.0,capacity/incoming),incoming>0.0);
}

@compute @workgroup_size(64)
fn finalizeSharpening(@builtin(global_invocation_id)gid:vec3u){
  let cell=gid.x;if(cell>=p.counts.x){return;}let delta=state[p.stateOffsets5.x+cell];
  let incoming=f32(atomicLoad(&conditioning[3u*p.counts.x+cell]))/CM12_TRANSPORT_FIXED;
  let accepted=incoming*state[p.stateOffsets5.y+cell];var returned=0.0;
  if(delta<0.0){let stats=sharpeningStats(cell);let rho=state[destinationDensity()+cell];
    let removed=-delta*cellVolume(cell);let maximumDistance=2.1*cellMinimumWidth(cell);
    for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
      let row=incidenceRow(at);let own=termCoefficient(incidenceTerm(at));
      let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
      var negativeCount=0.0;var positiveCount=0.0;
      for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
        negativeCount+=select(0.0,1.0,coefficient<0.0);
        positiveCount+=select(0.0,1.0,coefficient>0.0);}
      if(negativeCount==0.0||positiveCount==0.0||rowDistance(row)>maximumDistance){continue;}
      let area=rowArea(row)/(negativeCount*positiveCount);let distance=rowDistance(row);
      for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
        if(own*coefficient>=0.0){continue;}let neighbor=termCell(term);
        let conductance=area*max(0.0,
          (state[destinationDensity()+neighbor]-rho)/distance);
        if(conductance<=0.0){continue;}
        let offered=removed*conductance/stats.uphillConductance;
        returned+=offered*(1.0-state[p.stateOffsets5.y+neighbor]);
      }
    }
  }
  state[destinationDensity()+cell]=max(0.0,state[destinationDensity()+cell]+delta
    +(accepted+returned)/cellVolume(cell));
}

// The CPU sparse path retains a proven horizontal D4 invariant after surface
// conditioning. Quantizing the orbit sum before division makes that invariant
// bit-exact despite transformed cells visiting the same values in another
// floating-point order. This pass is encoded only while the topology and
// authored material are D4 symmetric.
@compute @workgroup_size(64)
fn preserveHorizontalD4(@builtin(global_invocation_id)gid:vec3u){
  let cell=gid.x;if(cell>=p.counts.x){return;}let b=cellBase(cell);
  let center=vec3f(tf(b),tf(b+1u),tf(b+2u));let extent=f32(p.dimensions.x);
  let xs=array<f32,8>(center.x,extent-center.x,center.x,extent-center.x,
    center.z,extent-center.z,center.z,extent-center.z);
  let zs=array<f32,8>(center.z,center.z,extent-center.z,extent-center.z,
    center.x,center.x,extent-center.x,extent-center.x);
  var rhoSum=0;var gammaSum=0;var count=0;
  for(var transform=0u;transform<8u;transform+=1u){
    let member=ownerCellAt(vec3i(i32(floor(xs[transform])),i32(floor(center.y)),
      i32(floor(zs[transform]))));
    if(member==INVALID){continue;}
    rhoSum+=i32(round(state[destinationDensity()+member]*CM12_TRANSPORT_FIXED));
    gammaSum+=i32(round(state[destinationGamma()+member]*CM12_TRANSPORT_FIXED));
    count+=1;
  }
  state[p.stateOffsets5.x+cell]=f32(rhoSum)/(f32(count)*CM12_TRANSPORT_FIXED);
  state[p.stateOffsets5.y+cell]=f32(gammaSum)/(f32(count)*CM12_TRANSPORT_FIXED);
}

@compute @workgroup_size(64)
fn commitHorizontalD4(@builtin(global_invocation_id)gid:vec3u){
  let cell=gid.x;if(cell>=p.counts.x){return;}
  state[destinationDensity()+cell]=state[p.stateOffsets5.x+cell];
  state[destinationGamma()+cell]=state[p.stateOffsets5.y+cell];
}

@compute @workgroup_size(64)
fn forceFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;if(row>=p.counts.y){return;}
  state[destinationFaceVelocity()+row]+=p.frame.x*p.acceleration[rowAxis(row)];
}

@compute @workgroup_size(64)
fn classifyRows(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;if(row>=p.counts.y){return;}
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  var liquidCount=0u;var airCount=0u;var liquidPhiSum=0.0;var liquidWeight=0.0;
  var airPhiSum=0.0;var airWeight=0.0;
  for(var at=begin;at<end;at+=1u){let cell=termCell(at);let w=abs(termCoefficient(at));
    let phi=CM12_LIQUID_ISOVALUE-state[destinationDensity()+cell];
    if(isLiquid(cell)){liquidCount+=1u;liquidPhiSum+=w*phi;liquidWeight+=w;}
    else{airCount+=1u;airPhiSum+=w*phi;airWeight+=w;}}
  if(liquidCount==0u){state[p.stateOffsets3.x+row]=0.0;return;}
  if(rowKind(row)==3u){let w=liquidWeight;airPhiSum+=w*rowExteriorPhi(row);airWeight+=w;}
  let cut=airCount>0u||rowKind(row)==3u;
  let theta=select(1.0,cm12GhostFluidTheta(liquidPhiSum/max(liquidWeight,1e-9),
    airPhiSum/max(airWeight,1e-9),1e-12),cut);
  state[p.stateOffsets3.x+row]=theta;
}

fn applyOperator(cell:u32,inputOffset:u32)->f32{
  if(!isLiquid(cell)){return 0.0;}var result=0.0;
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
    let row=incidenceRow(at);let theta=state[p.stateOffsets3.x+row];if(theta<=0.0){continue;}
    var jump=0.0;let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let other=termCell(term);
      if(isLiquid(other)){jump+=termCoefficient(term)*state[inputOffset+other];}}
    result+=termCoefficient(incidenceTerm(at))*rowDualWeight(row)*jump/theta;
  }return result;
}

@compute @workgroup_size(64)
fn preparePressure(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=p.counts.x){return;}
  let rho=state[destinationDensity()+id];let liquid=rho>=CM12_LIQUID_ISOVALUE;
  state[p.stateOffsets2.w+id]=select(0.0,1.0,liquid);
  if(!liquid){state[p.stateOffsets2.y+id]=0.0;state[p.stateOffsets2.z+id]=0.0;
    state[p.stateOffsets2.x+id]=0.0;return;}
  var rhs=0.0;var diagonal=0.0;
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){
    let row=incidenceRow(at);let theta=state[p.stateOffsets3.x+row];if(theta<=0.0){continue;}
    let coefficient=termCoefficient(incidenceTerm(at));
    rhs+=coefficient*rowDualWeight(row)*state[destinationFaceVelocity()+row];
    diagonal+=rowDualWeight(row)*coefficient*coefficient/theta;
  }
  let targetDivergence=cm12VolumeCorrectionDivergence(rho,p.frame.y*cellMinimumWidth(id),p.frame.x);
  state[p.stateOffsets2.y+id]=rhs+cellVolume(id)*targetDivergence;
  state[p.stateOffsets2.z+id]=diagonal;
}

var<workgroup>reduceA:array<f32,64>;
var<workgroup>reduceB:array<f32,64>;
var<workgroup>activityDensitySum:array<i32,64>;
var<workgroup>activityMomentX:array<i32,64>;
var<workgroup>activityMomentY:array<i32,64>;
var<workgroup>activityMomentZ:array<i32,64>;
var<workgroup>activityDeformation:array<f32,64>;
var<workgroup>activityPredictedMotion:array<f32,64>;
var<workgroup>activityDetailError:array<f32,64>;
var<workgroup>activitySurfaceAxes:array<u32,64>;
fn reducePair(lane:u32,group:u32,a:f32,b:f32){
  reduceA[lane]=a;reduceB[lane]=b;workgroupBarrier();
  var width=32u;loop{if(lane<width){reduceA[lane]+=reduceA[lane+width];reduceB[lane]+=reduceB[lane+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane==0u){partials[group]=vec2f(reduceA[0],reduceB[0]);}
}

@compute @workgroup_size(64)
fn initializePCG(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let id=gid.x;var rz=0.0;var rhs2=0.0;if(id<p.counts.x){
    let image=applyOperator(id,p.stateOffsets2.x);let residual=state[p.stateOffsets2.y+id]-image;
    let diagonal=state[p.stateOffsets2.z+id];let z=select(0.0,residual/diagonal,diagonal>0.0);
    state[p.stateOffsets3.y+id]=residual;state[p.stateOffsets3.z+id]=z;
    state[p.stateOffsets3.w+id]=z;rz=residual*z;let rhs=state[p.stateOffsets2.y+id];rhs2=rhs*rhs;}
  reducePair(lid.x,wid.x,rz,rhs2);
}

@compute @workgroup_size(64)
fn reduceInitialize(@builtin(local_invocation_id)lid:vec3u){
  var a=0.0;var b=0.0;for(var at=lid.x;at<p.dispatch.x;at+=64u){a+=partials[at].x;b+=partials[at].y;}
  reduceA[lid.x]=a;reduceB[lid.x]=b;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];reduceB[lid.x]+=reduceB[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){scalars[0]=reduceA[0];scalars[1]=reduceB[0];scalars[2]=0.0;scalars[3]=0.0;}
}

@compute @workgroup_size(64)
fn applyDirection(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let id=gid.x;var curvature=0.0;if(id<p.counts.x){let image=applyOperator(id,p.stateOffsets3.w);
    state[p.stateOffsets4.x+id]=image;curvature=state[p.stateOffsets3.w+id]*image;}
  reducePair(lid.x,wid.x,curvature,0.0);
}

@compute @workgroup_size(64)
fn reduceCurvature(@builtin(local_invocation_id)lid:vec3u){
  var sum=0.0;for(var at=lid.x;at<p.dispatch.x;at+=64u){sum+=partials[at].x;}
  reduceA[lid.x]=sum;workgroupBarrier();var width=32u;loop{if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){scalars[2]=select(0.0,scalars[0]/reduceA[0],reduceA[0]>1e-20);}
}

@compute @workgroup_size(64)
fn updateResidual(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let id=gid.x;var rz=0.0;var residual2=0.0;if(id<p.counts.x){let alpha=scalars[2];
    state[p.stateOffsets2.x+id]+=alpha*state[p.stateOffsets3.w+id];
    let residual=state[p.stateOffsets3.y+id]-alpha*state[p.stateOffsets4.x+id];
    let diagonal=state[p.stateOffsets2.z+id];let z=select(0.0,residual/diagonal,diagonal>0.0);
    state[p.stateOffsets3.y+id]=residual;state[p.stateOffsets3.z+id]=z;
    rz=residual*z;residual2=residual*residual;}
  reducePair(lid.x,wid.x,rz,residual2);
}

@compute @workgroup_size(64)
fn reduceResidual(@builtin(local_invocation_id)lid:vec3u){
  var a=0.0;var b=0.0;for(var at=lid.x;at<p.dispatch.x;at+=64u){a+=partials[at].x;b+=partials[at].y;}
  reduceA[lid.x]=a;reduceB[lid.x]=b;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];reduceB[lid.x]+=reduceB[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){scalars[3]=select(0.0,reduceA[0]/scalars[0],scalars[0]>1e-20);
    scalars[0]=reduceA[0];scalars[4]=reduceB[0];}
}

@compute @workgroup_size(64)
fn updateDirection(@builtin(global_invocation_id)gid:vec3u){let id=gid.x;if(id>=p.counts.x){return;}
  state[p.stateOffsets3.w+id]=state[p.stateOffsets3.z+id]+scalars[3]*state[p.stateOffsets3.w+id];}

@compute @workgroup_size(64)
fn projectFaces(@builtin(global_invocation_id)gid:vec3u){let row=gid.x;if(row>=p.counts.y){return;}
  let theta=state[p.stateOffsets3.x+row];if(theta<=0.0){state[destinationFaceVelocity()+row]=0.0;return;}
  var jump=0.0;let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var at=begin;at<end;at+=1u){let cell=termCell(at);if(isLiquid(cell)){
    jump+=termCoefficient(at)*state[p.stateOffsets2.x+cell];}}
  state[destinationFaceVelocity()+row]-=jump/theta;
}

@compute @workgroup_size(64)
fn collocateAndDiagnose(@builtin(global_invocation_id)gid:vec3u){let id=gid.x;if(id>=p.counts.x){return;}
  var velocity=vec3f(0.0);var weight=vec3f(0.0);var equation=0.0;var correction=0.0;
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){let row=incidenceRow(at);
    let term=incidenceTerm(at);let axis=rowAxis(row);let w=abs(termCoefficient(term))*rowDualWeight(row);
    velocity[axis]+=w*state[destinationFaceVelocity()+row];weight[axis]+=w;
    if(isLiquid(id)){let value=termCoefficient(term)*rowDualWeight(row)*state[destinationFaceVelocity()+row];
      let adjusted=value-correction;let next=equation+adjusted;correction=(next-equation)-adjusted;equation=next;}}
  for(var axis=0u;axis<3u;axis+=1u){if(weight[axis]>0.0){velocity[axis]/=weight[axis];}}
  state[destinationCellVelocity()+4u*id]=velocity.x;state[destinationCellVelocity()+4u*id+1u]=velocity.y;
  state[destinationCellVelocity()+4u*id+2u]=velocity.z;state[destinationCellVelocity()+4u*id+3u]=0.0;
  let targetDivergence=cm12VolumeCorrectionDivergence(state[destinationDensity()+id],p.frame.y*cellMinimumWidth(id),p.frame.x);
  state[p.stateOffsets4.y+id]=select(0.0,-equation/cellVolume(id)-targetDivergence,isLiquid(id));
}

@compute @workgroup_size(64)
fn measureDivergenceDiagnostics(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  var globalMaximum=0.0;var mixedMaximum=0.0;
  if(gid.x<p.counts.x){let value=abs(state[p.stateOffsets4.y+gid.x]);
    globalMaximum=value;var touchesMixed=false;
    for(var at=incidenceBegin(gid.x);at<incidenceEnd(gid.x);at+=1u){
      touchesMixed=touchesMixed||rowKind(incidenceRow(at))==2u;
    }
    mixedMaximum=select(0.0,value,touchesMixed);
  }
  reduceA[lid.x]=globalMaximum;reduceB[lid.x]=mixedMaximum;workgroupBarrier();
  var width=32u;loop{if(lid.x<width){reduceA[lid.x]=max(reduceA[lid.x],reduceA[lid.x+width]);
    reduceB[lid.x]=max(reduceB[lid.x],reduceB[lid.x+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){partials[wid.x]=vec2f(reduceA[0],reduceB[0]);}
}

@compute @workgroup_size(64)
fn reduceDivergenceDiagnostics(@builtin(local_invocation_id)lid:vec3u){
  var globalMaximum=0.0;var mixedMaximum=0.0;
  for(var at=lid.x;at<p.dispatch.x;at+=64u){
    globalMaximum=max(globalMaximum,partials[at].x);
    mixedMaximum=max(mixedMaximum,partials[at].y);
  }
  reduceA[lid.x]=globalMaximum;reduceB[lid.x]=mixedMaximum;workgroupBarrier();
  var width=32u;loop{if(lid.x<width){reduceA[lid.x]=max(reduceA[lid.x],reduceA[lid.x+width]);
    reduceB[lid.x]=max(reduceB[lid.x],reduceB[lid.x+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){scalars[6]=reduceA[0];scalars[7]=reduceB[0];}
}

// Advance and clear the compact receipt entirely on the device. The host
// encodes this fixed singleton but neither supplies nor consumes a policy
// decision while advancing the simulation.
@compute @workgroup_size(1)
fn advanceActivityClock(){
  let step=atomicAdd(&activity[0],1u)+1u;
  atomicStore(&activity[1],0u); // maximum score
  atomicStore(&activity[2],0u); // surface bricks
  atomicStore(&activity[3],0u); // hot bricks
  atomicStore(&activity[4],0u); // quiet bricks
  atomicStore(&activity[5],select(0u,1u,step%4u==0u));
  atomicStore(&activity[6],0u); // measured bricks
  atomicStore(&activity[7],0u); // reserved failure flags
}

fn activityRecord(brick:u32)->u32{
  return ACTIVITY_HEADER_WORDS+ACTIVITY_RECORD_WORDS*brick;
}

fn activityF32(index:u32)->f32{return bitcast<f32>(atomicLoad(&activity[index]));}

// One workgroup owns one brick. Fixed-point density moments make the compact
// history exactly invariant to x/z lane permutations for a D4-symmetric field;
// only maxima are used for floating activity channels.
@compute @workgroup_size(64)
fn measureBrickActivity(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let lane=lid.x;
  if(brick>=p.dispatch.w){return;}
  let brickRecord=p.topologyOffsets2.z+4u*brick;
  let first=topology[brickRecord];let count=topology[brickRecord+1u];
  let resolution=topology[brickRecord+2u];
  var densitySum=0;var momentX=0;var momentY=0;var momentZ=0;
  var deformation=0.0;var predictedMotion=0.0;var detailError=0.0;
  var surfaceAxes=0u;var surfaceCell=false;
  for(var cell=first+lane;cell<first+count;cell+=64u){
    let rho=state[destinationDensity()+cell];let volume=cellVolume(cell);
    let local=cell-first;let x=local%resolution;
    let yz=local/resolution;let y=yz%resolution;let z=yz/resolution;
    densitySum+=i32(round(rho*volume*ACTIVITY_FIXED));
    momentX+=i32(round(rho*volume
      *(f32(2u*x+1u)-f32(resolution))/f32(resolution)*ACTIVITY_FIXED));
    momentY+=i32(round(rho*volume
      *(f32(2u*y+1u)-f32(resolution))/f32(resolution)*ACTIVITY_FIXED));
    momentZ+=i32(round(rho*volume
      *(f32(2u*z+1u)-f32(resolution))/f32(resolution)*ACTIVITY_FIXED));
    surfaceCell=surfaceCell||(rho>0.05&&rho<0.95);
    let ownVelocityAt=destinationCellVelocity()+4u*cell;
    let ownVelocity=vec3f(state[ownVelocityAt],state[ownVelocityAt+1u],
      state[ownVelocityAt+2u]);
    let ownWet=rho>=CM12_LIQUID_ISOVALUE;
    for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
      let row=incidenceRow(incidence);let own=termCoefficient(incidenceTerm(incidence));
      var crosses=rowKind(row)==3u&&ownWet;
      let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
      for(var term=begin;term<end;term+=1u){
        let coefficient=termCoefficient(term);if(own*coefficient>=0.0){continue;}
        let neighbor=termCell(term);
        crosses=crosses||(state[destinationDensity()+neighbor]>=CM12_LIQUID_ISOVALUE)!=ownWet;
        let neighborVelocityAt=destinationCellVelocity()+4u*neighbor;
        let neighborVelocity=vec3f(state[neighborVelocityAt],
          state[neighborVelocityAt+1u],state[neighborVelocityAt+2u]);
        deformation=max(deformation,p.frame.x*max(abs(ownVelocity.x-neighborVelocity.x),
          max(abs(ownVelocity.y-neighborVelocity.y),abs(ownVelocity.z-neighborVelocity.z)))
          /max(0.15*rowDistance(row),1e-12));
      }
      if(crosses){
        surfaceAxes|=1u<<rowAxis(row);
        predictedMotion=max(predictedMotion,p.frame.x
          *abs(state[destinationFaceVelocity()+row])/max(0.25*rowDistance(row),1e-12));
      }
    }
    if(resolution==8u){
      let b=cellBase(cell);let q=vec3u(topology[b+7u],topology[b+8u],topology[b+9u]);
      let group=2u*(q/2u);var childSum=0;
      for(var dz=0u;dz<2u;dz+=1u){for(var dy=0u;dy<2u;dy+=1u){
        for(var dx=0u;dx<2u;dx+=1u){
          let child=ownerCellAt(vec3i(group+vec3u(dx,dy,dz)));
          if(child!=INVALID){childSum+=i32(round(
            state[destinationDensity()+child]*ACTIVITY_FIXED));}
      }}}
      let ownFixed=i32(round(rho*ACTIVITY_FIXED));
      detailError=max(detailError,
        f32(abs(8*ownFixed-childSum))/(8.0*ACTIVITY_FIXED));
    }
  }
  activityDensitySum[lane]=densitySum;activityMomentX[lane]=momentX;
  activityMomentY[lane]=momentY;activityMomentZ[lane]=momentZ;
  activityDeformation[lane]=deformation;activityPredictedMotion[lane]=predictedMotion;
  activityDetailError[lane]=detailError;
  activitySurfaceAxes[lane]=surfaceAxes|select(0u,8u,surfaceCell);
  workgroupBarrier();
  var width=32u;loop{
    if(lane<width){
      activityDensitySum[lane]+=activityDensitySum[lane+width];
      activityMomentX[lane]+=activityMomentX[lane+width];
      activityMomentY[lane]+=activityMomentY[lane+width];
      activityMomentZ[lane]+=activityMomentZ[lane+width];
      activityDeformation[lane]=max(activityDeformation[lane],activityDeformation[lane+width]);
      activityPredictedMotion[lane]=max(activityPredictedMotion[lane],
        activityPredictedMotion[lane+width]);
      activityDetailError[lane]=max(activityDetailError[lane],activityDetailError[lane+width]);
      activitySurfaceAxes[lane]|=activitySurfaceAxes[lane+width];
    }
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lane!=0u){return;}
  let output=activityRecord(brick);let step=atomicLoad(&activity[0]);
  let totalVolume=f32(count)*cellVolume(first);
  let meanDensity=f32(activityDensitySum[0])/(totalVolume*ACTIVITY_FIXED);
  let moments=vec3f(f32(activityMomentX[0]),f32(activityMomentY[0]),
    f32(activityMomentZ[0]))/(totalVolume*ACTIVITY_FIXED);
  var temporal=0.0;
  if(step>1u){
    temporal=max(abs(meanDensity-activityF32(output+4u))/0.05,
      max(abs(moments.x-activityF32(output+5u))/0.02,
      max(abs(moments.y-activityF32(output+6u))/0.02,
        abs(moments.z-activityF32(output+7u))/0.02)));
  }
  let axes=activitySurfaceAxes[0]&7u;
  let surface=(activitySurfaceAxes[0]&8u)!=0u||axes!=0u;
  let shape=select(0.0,1.0,countOneBits(axes)>=2u);
  let scoreValue=clamp(max(max(activityDeformation[0],activityPredictedMotion[0]),
    max(max(temporal,shape),activityDetailError[0]/0.08)),0.0,1.0);
  let score=u32(round(255.0*scoreValue));var reasons=0u;
  if(surface){reasons|=1u;}if(activityDeformation[0]>0.0){reasons|=2u;}
  if(temporal>0.0){reasons|=4u;}if(activityDetailError[0]>0.0){reasons|=8u;}
  if(activityPredictedMotion[0]>0.0){reasons|=16u;}if(step==1u){reasons|=32u;}
  let topologyEpoch=atomicLoad(&activity[5])!=0u;
  var hotEpochs=atomicLoad(&activity[output+2u]);
  var quietEpochs=atomicLoad(&activity[output+3u]);
  if(topologyEpoch){
    hotEpochs=select(0u,min(255u,hotEpochs+1u),score>=160u);
    quietEpochs=select(0u,min(255u,quietEpochs+1u),score<=96u
      &&activityDetailError[0]<=0.08);
  }
  atomicStore(&activity[output],score);atomicStore(&activity[output+1u],reasons);
  atomicStore(&activity[output+2u],hotEpochs);atomicStore(&activity[output+3u],quietEpochs);
  atomicStore(&activity[output+4u],bitcast<u32>(meanDensity));
  atomicStore(&activity[output+5u],bitcast<u32>(moments.x));
  atomicStore(&activity[output+6u],bitcast<u32>(moments.y));
  atomicStore(&activity[output+7u],bitcast<u32>(moments.z));
  atomicMax(&activity[1],score);if(surface){atomicAdd(&activity[2],1u);}
  if(score>=160u){atomicAdd(&activity[3],1u);}if(score<=96u){atomicAdd(&activity[4],1u);}
  atomicAdd(&activity[6],1u);
}

@compute @workgroup_size(64)
fn classifyPresentationBricks(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  let record=p.topologyOffsets2.z+4u*brick;let first=topology[record];let count=topology[record+1u];
  var wet=false;
  for(var at=first;at<first+count;at+=1u){
    wet=wet||state[destinationDensity()+at]>1e-5;
  }
  state[p.stateOffsets4.z+brick]=select(0.0,1.0,wet);
}

@compute @workgroup_size(4,4,4)
fn publishPresentation(@builtin(global_invocation_id)gid:vec3u){if(any(gid>=p.dimensions.xyz)){return;}
  let dense=gid.x+p.dimensions.x*(gid.y+p.dimensions.y*gid.z);
  let owner=p.topologyOffsets2.y+4u*dense;let cell=topology[owner];let coordinate=vec3i(gid);
  let brick=topology[owner+3u];
  let presentationCell=select(INVALID,cell,cell!=INVALID&&brick<p.dispatch.w
    &&state[p.stateOffsets4.z+brick]>0.5);
  if(presentationCell==INVALID){textureStore(densityTexture,coordinate,vec4f(0.0));
    textureStore(levelSetTexture,coordinate,vec4f(4.0*p.frame.y));
    textureStore(ownerTexture,coordinate,vec4u(topology[p.topologyOffsets2.w],
      topology[p.topologyOffsets2.w+1u],0u,0u));
    textureStore(velocityTexture,coordinate,vec4f(0.0));textureStore(pressureTexture,coordinate,vec4f(0.0));
    textureStore(divergenceTexture,coordinate,vec4f(0.0));return;}
  let rho=state[destinationDensity()+presentationCell];
  textureStore(densityTexture,coordinate,vec4f(rho));
  // CM12 renders the 0.5 isocontour of authoritative rho. Section 3.8's
  // optional density postprocess is deliberately off, as in the paper's
  // default results; there is no independently advected wall-film tracer.
  let presentationScale=i32(topology[cellBase(presentationCell)+10u]);
  var phi=presentationPhi(presentationCell);
  if(presentationScale>1){phi=interpolatedPresentationPhi(coordinate,presentationScale);}
  textureStore(levelSetTexture,coordinate,vec4f(phi));
  textureStore(ownerTexture,coordinate,vec4u(topology[owner+1u],topology[owner+2u],0u,0u));
  let v=destinationCellVelocity()+4u*presentationCell;
  textureStore(velocityTexture,coordinate,vec4f(state[v]*p.frame.y,state[v+1u]*p.frame.y,state[v+2u]*p.frame.y,0.0));
  textureStore(pressureTexture,coordinate,vec4f(state[p.stateOffsets2.x+presentationCell]*p.frame.z));
  textureStore(divergenceTexture,coordinate,vec4f(state[p.stateOffsets4.y+presentationCell]));
}
`;
