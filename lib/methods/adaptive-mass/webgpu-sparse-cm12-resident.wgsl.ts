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
const ACTIVITY_HEADER_WORDS:u32=12u;
const ACTIVITY_RECORD_WORDS:u32=33u;
const CANDIDATE_CELLS_PER_BRICK:u32=512u;
const ACTIVITY_FIXED:f32=65536.0;

struct Params {
  counts:vec4u,             // cell, row, incidence, dense
  dimensions:vec4u,
  topologyOffsets:vec4u,    // cells, rows, terms, incidenceOffsets
  topologyOffsets2:vec4u,   // incidences, sorted brick key/index pairs, brick records, background owner
  stateOffsets0:vec4u,      // density A/B, gamma A/B
  stateOffsets1:vec4u,      // cell velocity A/B, face A/B
  stateOffsets2:vec4u,      // pressure, rhs, diagonal, liquid
  stateOffsets3:vec4u,      // theta, residual, preconditioned, direction
  stateOffsets4:vec4u,      // applied, divergence, presentation brick wet, reserved
  stateOffsets5:vec4u,      // sharpening delta / D4 rho scratch, D4 gamma scratch, reserved
  frame:vec4f,              // dt, finest cell metres, pressure scale, parity
  acceleration:vec4f,       // finest cells / second^2
  dispatch:vec4u,           // cell workgroups, row workgroups, pcg iterations, brick count
  injectionCenter:vec4f,
  injectionRadius:vec4f,
  sharpening:vec4f,         // Algorithm 2 distance in cells, trace substeps, reserved
}

@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>topology:array<u32>;
@group(0)@binding(2)var<storage,read_write>state:array<f32>;
@group(0)@binding(3)var<storage,read_write>partials:array<vec2f>;
@group(0)@binding(4)var<storage,read_write>scalars:array<f32>;
@group(0)@binding(11)var<storage,read_write>conditioning:array<atomic<i32>>;
// Device-owned activity, planning, and logical-residency arena. Physics reads
// the published active bit, while immutable packed topology and accepted CM12
// fields remain separate storage.
@group(0)@binding(12)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(13)var<storage,read_write>candidateState:array<f32>;
@group(0)@binding(14)var<storage,read>fineMetadata:array<u32>;
@group(0)@binding(15)var<storage,read_write>fineSamples:array<u32>;

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
fn cellBrick(id:u32)->u32{return topology[cellBase(id)+11u];}
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
fn activityRecord(brick:u32)->u32{
  return ACTIVITY_HEADER_WORDS+ACTIVITY_RECORD_WORDS*brick;
}
fn brickActive(brick:u32)->bool{
  return brick<p.dispatch.w&&atomicLoad(&activity[activityRecord(brick)+10u])!=0u;
}
fn cellActive(cell:u32)->bool{return brickActive(cellBrick(cell));}
fn isLiquid(cell:u32)->bool{return cellActive(cell)&&state[p.stateOffsets2.w+cell]>0.5;}

fn brickDirectoryLookup(key:u32)->u32{
  var low=0u;var high=p.dispatch.w;
  loop{
    if(low>=high){break;}
    let middle=low+(high-low)/2u;
    let candidate=topology[p.topologyOffsets2.y+2u*middle];
    if(candidate<key){low=middle+1u;}else{high=middle;}
  }
  if(low>=p.dispatch.w||topology[p.topologyOffsets2.y+2u*low]!=key){return INVALID;}
  return topology[p.topologyOffsets2.y+2u*low+1u];
}

fn compactOwnerCellAt(q:vec3i)->vec2u{
  if(any(q<vec3i(0))||any(q>=vec3i(p.dimensions.xyz))){return vec2u(INVALID);}
  let uq=vec3u(q);let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let brickCoordinate=uq/8u;
  let key=brickCoordinate.x+brickDimensions.x
    *(brickCoordinate.y+brickDimensions.y*brickCoordinate.z);
  let brick=brickDirectoryLookup(key);if(brick==INVALID){return vec2u(INVALID);}
  let record=p.topologyOffsets2.z+4u*brick;let first=topology[record];
  let count=topology[record+1u];let resolution=topology[record+2u];
  let scale=8u/resolution;let local=(uq-brickCoordinate*8u)/scale;
  let origin=brickCoordinate*8u;
  let valid=(min(p.dimensions.xyz-origin+vec3u(scale-1u),vec3u(8u)))/scale;
  let cell=first+local.x+valid.x*(local.y+valid.y*local.z);
  return select(vec2u(INVALID),vec2u(cell,brick),cell<first+count);
}

fn ownerCellAt(q:vec3i)->u32{
  let owner=compactOwnerCellAt(q);if(owner.x==INVALID||!brickActive(owner.y)){return INVALID;}
  return owner.x;
}

fn presentationOwnerCellAt(q:vec3i)->u32{
  let owner=compactOwnerCellAt(q);if(owner.x==INVALID){return INVALID;}
  return select(INVALID,owner.x,state[p.stateOffsets4.z+owner.y]>0.5);
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
  if(!cellActive(id)){
    state[output]=0.0;state[output+1u]=0.0;state[output+2u]=0.0;state[output+3u]=0.0;
    return;
  }
  if(state[input+3u]>0.5){
    state[output]=state[input];state[output+1u]=state[input+1u];
    state[output+2u]=state[input+2u];state[output+3u]=1.0;return;
  }
  var velocity=vec3f(0.0);var weight=0.0;
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){
    let row=incidenceRow(at);let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let neighbor=termCell(term);
      if(neighbor==id||!cellActive(neighbor)){continue;}let source=inputOffset+4u*neighbor;
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
  if(!cellActive(id)){
    state[output]=0.0;state[output+1u]=0.0;state[output+2u]=0.0;state[output+3u]=0.0;
    return;
  }
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
  let id=gid.x;if(id>=p.counts.x||!cellActive(id)){return;}let b=cellBase(id);
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
  let id=gid.x;if(id>=p.counts.x){return;}
  if(!cellActive(id)){state[destinationGamma()+id]=1.0;return;}
  let b=cellBase(id);
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
  let donor=gid.x;if(donor>=p.counts.x||!cellActive(donor)){return;}
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
  let id=gid.x;if(id>=p.counts.x){return;}
  if(!cellActive(id)){
    state[destinationDensity()+id]=0.0;state[destinationGamma()+id]=1.0;return;
  }
  let b=cellBase(id);
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
  if(rhoNext<CM12_DRY_CELL_THRESHOLD){gammaNext=1.0;}
  state[destinationDensity()+id]=max(0.0,rhoNext);
  state[destinationGamma()+id]=max(0.0,gammaNext);
}

fn diffuseGammaAxis(cell:u32,axis:u32,inputRho:u32,inputGamma:u32,
 outputRho:u32,outputGamma:u32){
  let ownRho=state[inputRho+cell];let ownGamma=state[inputGamma+cell];
  if(!cellActive(cell)){
    state[outputRho+cell]=0.0;state[outputGamma+cell]=1.0;return;
  }
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
      if(!cellActive(neighbor)){continue;}
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
  if(!cellActive(cell)){
    state[destinationDensity()+cell]=0.0;state[destinationGamma()+cell]=1.0;return;
  }
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
}

// Expand each composite G row into the same physical scalar subfaces used by
// sparse-atlas-surface-conditioning.ts. This retains the paper's adjacent-cell
// stencil at 4^3/8^3 seams instead of treating an aggregate port as one cell.
fn sharpeningStats(cell:u32)->SharpeningStats{
  var result:SharpeningStats;let rho=state[destinationDensity()+cell];
  result.maximumDifference=0.0;result.negativeArea=vec3f(0.0);
  result.positiveArea=vec3f(0.0);result.negativeDensity=vec3f(0.0);
  result.positiveDensity=vec3f(0.0);result.negativeDistance=vec3f(0.0);
  result.positiveDistance=vec3f(0.0);
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
      if(!cellActive(neighbor)){continue;}
      let neighborRho=state[destinationDensity()+neighbor];
      result.maximumDifference=max(result.maximumDifference,abs(rho-neighborRho));
      if(own<0.0){result.positiveArea[axis]+=area;
        result.positiveDensity[axis]+=area*neighborRho;
        result.positiveDistance[axis]+=area*distance;
      }else{result.negativeArea[axis]+=area;
        result.negativeDensity[axis]+=area*neighborRho;
        result.negativeDistance[axis]+=area*distance;}
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
  if(rho+delta<0.0||rho<CM12_DRY_CELL_THRESHOLD){delta=-rho;}else if(rho>0.5){delta=0.0;}
  return min(0.0,delta);
}

fn sampleSharpeningDensity(position:vec3f)->f32{
  var result=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let term=transportTerm(position,corner);
    if(term.cell!=INVALID){result+=term.weight*state[destinationDensity()+term.cell];}
  }
  return result;
}

fn sharpeningDensityGradient(position:vec3f,owner:u32)->vec3f{
  // Differentiate the same continuous adaptive interpolant used by the trace.
  // The finite-difference baseline follows the current owner after a 2:1 seam,
  // and the denominator remains an actual finest-coordinate distance.
  let halfDistance=0.5*cellMinimumWidth(owner);var result=vec3f(0.0);
  for(var axis=0u;axis<3u;axis+=1u){var offset=vec3f(0.0);offset[axis]=halfDistance;
    result[axis]=(sampleSharpeningDensity(position+offset)
      -sampleSharpeningDensity(position-offset))/max(2.0*halfDistance,1e-12);
  }
  return result;
}

// CM12 Algorithm 2's TraceAlongField in composite-grid coordinates. As in the
// Uniform reference, half-cell forward-Euler substeps follow the frozen density
// gradient until rho=.5 or the configured paper-range D bound is reached. An invalid
// or inactive owner is the sparse equivalent of the paper's solid-stop rule.
fn traceSharpeningMass(source:u32)->vec3f{
  let b=cellBase(source);var position=vec3f(tf(b),tf(b+1u),tf(b+2u));
  let sourceWidth=cellMinimumWidth(source);let maximumDistance=p.sharpening.x*sourceWidth;
  var travelled=0.0;
  for(var step=0u;step<40u;step+=1u){
    if(step>=u32(p.sharpening.y)){break;}
    if(sampleSharpeningDensity(position)>=CM12_LIQUID_ISOVALUE
      ||travelled>=maximumDistance){break;}
    let owner=ownerCellAt(vec3i(floor(position)));if(owner==INVALID){break;}
    let gradient=sharpeningDensityGradient(position,owner);let magnitude=length(gradient);
    if(magnitude<1e-6){break;}
    let distance=min(0.5*cellMinimumWidth(owner),maximumDistance-travelled);
    let candidate=position+gradient/magnitude*distance;
    if(ownerCellAt(vec3i(floor(candidate)))==INVALID){break;}
    position=candidate;travelled+=distance;
  }
  return position;
}

// CM12 Sec. 3.5, Eqs. 4-17 and Algorithm 2. Remove the air-side correction,
// trace along grad(rho), then scatter its integrated mass trilinearly at the
// traced point. Fixed-point atomics keep simultaneous GPU scatters additive.
@compute @workgroup_size(64)
fn scatterSharpeningMass(@builtin(global_invocation_id)gid:vec3u){
  let cell=gid.x;if(cell>=p.counts.x){return;}
  if(!cellActive(cell)){state[p.stateOffsets5.x+cell]=0.0;return;}
  let stats=sharpeningStats(cell);
  let delta=sharpeningDelta(cell,stats);
  state[p.stateOffsets5.x+cell]=delta;if(delta>=0.0){return;}
  let removed=-delta*cellVolume(cell);let removedFixed=i32(round(removed*CM12_TRANSPORT_FIXED));
  let position=traceSharpeningMass(cell);var total=0.0;var lastCorner=INVALID;
  for(var corner=0u;corner<8u;corner+=1u){let term=transportTerm(position,corner);
    if(term.cell!=INVALID&&term.weight>0.0){total+=term.weight;lastCorner=corner;}}
  if(total<=1e-8){
    atomicAdd(&conditioning[3u*p.counts.x+cell],removedFixed);return;
  }
  var remainingFixed=removedFixed;
  for(var corner=0u;corner<8u;corner+=1u){let term=transportTerm(position,corner);
    if(term.cell==INVALID||term.weight<=0.0){continue;}
    var offeredFixed=i32(round(f32(removedFixed)*term.weight/total));
    if(corner==lastCorner){offeredFixed=remainingFixed;}
    else{offeredFixed=clamp(offeredFixed,0,remainingFixed);}
    atomicAdd(&conditioning[3u*p.counts.x+term.cell],offeredFixed);
    remainingFixed-=offeredFixed;
  }
}

@compute @workgroup_size(64)
fn finalizeSharpening(@builtin(global_invocation_id)gid:vec3u){
  let cell=gid.x;if(cell>=p.counts.x){return;}
  if(!cellActive(cell)){
    state[destinationDensity()+cell]=0.0;state[destinationGamma()+cell]=1.0;return;
  }
  let delta=state[p.stateOffsets5.x+cell];
  let incoming=f32(atomicLoad(&conditioning[3u*p.counts.x+cell]))/CM12_TRANSPORT_FIXED;
  state[destinationDensity()+cell]=max(0.0,state[destinationDensity()+cell]+delta
    +incoming/cellVolume(cell));
}

// The CPU sparse path retains a proven horizontal D4 invariant after surface
// conditioning. Quantizing the orbit sum before division makes that invariant
// bit-exact despite transformed cells visiting the same values in another
// floating-point order. This pass is encoded only while the topology and
// authored material are D4 symmetric.
@compute @workgroup_size(64)
fn preserveHorizontalD4(@builtin(global_invocation_id)gid:vec3u){
  let cell=gid.x;if(cell>=p.counts.x){return;}let b=cellBase(cell);
  if(!cellActive(cell)){
    state[p.stateOffsets5.x+cell]=0.0;state[p.stateOffsets5.y+cell]=1.0;return;
  }
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
  if(!cellActive(cell)){return;}
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
  let rho=state[destinationDensity()+id];let liquid=cellActive(id)&&rho>=CM12_LIQUID_ISOVALUE;
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
var<workgroup>activitySupportMask:array<u32,64>;
var<workgroup>transferMassBefore:array<f32,64>;
var<workgroup>transferMassAfter:array<f32,64>;
var<workgroup>transferGammaDelta:array<f32,64>;
var<workgroup>transferMomentumXDelta:array<f32,64>;
var<workgroup>transferMomentumYDelta:array<f32,64>;
var<workgroup>transferMomentumZDelta:array<f32,64>;
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
  if(!cellActive(id)){
    let output=destinationCellVelocity()+4u*id;
    state[output]=0.0;state[output+1u]=0.0;state[output+2u]=0.0;state[output+3u]=0.0;
    state[p.stateOffsets4.y+id]=0.0;return;
  }
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
  atomicStore(&activity[9],0u); // newly activated bricks
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
  let resident=brickActive(brick);
  let brickRecord=p.topologyOffsets2.z+4u*brick;
  let first=topology[brickRecord];let count=topology[brickRecord+1u];
  let resolution=topology[brickRecord+2u];
  var densitySum=0;var momentX=0;var momentY=0;var momentZ=0;
  var deformation=0.0;var predictedMotion=0.0;var detailError=0.0;
  var surfaceAxes=0u;var surfaceCell=false;var occupiedCell=false;
  var supportMask=0u;
  let measuredCount=select(0u,count,resident);
  for(var cell=first+lane;cell<first+measuredCount;cell+=64u){
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
    var interfaceCell=rho>0.05&&rho<0.95;
    surfaceCell=surfaceCell||interfaceCell;
    occupiedCell=occupiedCell||rho>0.0;
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
        interfaceCell=true;
        surfaceAxes|=1u<<rowAxis(row);
        predictedMotion=max(predictedMotion,p.frame.x
          *abs(state[destinationFaceVelocity()+row])/max(0.25*rowDistance(row),1e-12));
      }
    }
    if(interfaceCell){
      var minimumOffset=vec3i(0);var maximumOffset=vec3i(0);
      if(x==0u){minimumOffset.x=-1;}if(x+1u==resolution){maximumOffset.x=1;}
      if(y==0u){minimumOffset.y=-1;}if(y+1u==resolution){maximumOffset.y=1;}
      if(z==0u){minimumOffset.z=-1;}if(z+1u==resolution){maximumOffset.z=1;}
      for(var dz=minimumOffset.z;dz<=maximumOffset.z;dz+=1){
        for(var dy=minimumOffset.y;dy<=maximumOffset.y;dy+=1){
          for(var dx=minimumOffset.x;dx<=maximumOffset.x;dx+=1){
            if(dx!=0||dy!=0||dz!=0){
              let bit=u32(dx+1)+3u*u32(dy+1)+9u*u32(dz+1);
              supportMask|=1u<<bit;
            }
      }}}
      let b=cellBase(cell);let center=vec3f(tf(b),tf(b+1u),tf(b+2u));
      let brickDimensions=vec3i((p.dimensions.xyz+vec3u(7u))/8u);
      let sourceBrick=vec3i(vec3u(topology[b+7u],topology[b+8u],topology[b+9u])/8u);
      let sweptBrick=clamp(vec3i(floor((center+p.frame.x*ownVelocity)/8.0)),
        vec3i(0),brickDimensions-vec3i(1));
      let sweptOffset=clamp(sweptBrick-sourceBrick,vec3i(-1),vec3i(1));
      if(any(sweptOffset!=vec3i(0))){
        let bit=u32(sweptOffset.x+1)+3u*u32(sweptOffset.y+1)
          +9u*u32(sweptOffset.z+1);supportMask|=1u<<bit;
      }
    }
    if(resolution>1u){
      let group=2u*(vec3u(x,y,z)/2u);var childSum=0;
      for(var dz=0u;dz<2u;dz+=1u){for(var dy=0u;dy<2u;dy+=1u){
        for(var dx=0u;dx<2u;dx+=1u){
          let q=group+vec3u(dx,dy,dz);
          let child=first+q.x+resolution*(q.y+resolution*q.z);
          if(child<first+count){childSum+=i32(round(
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
  activitySurfaceAxes[lane]=surfaceAxes|select(0u,8u,surfaceCell)
    |select(0u,16u,occupiedCell);
  activitySupportMask[lane]=supportMask;
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
      activitySupportMask[lane]|=activitySupportMask[lane+width];
    }
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lane!=0u){return;}
  let output=activityRecord(brick);let step=atomicLoad(&activity[0]);
  if(!resident){
    atomicStore(&activity[output],0u);atomicStore(&activity[output+1u],0u);
    atomicStore(&activity[output+32u],0u);return;
  }
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
  if((activitySurfaceAxes[0]&16u)!=0u){reasons|=64u;}
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
  atomicStore(&activity[output+32u],activitySupportMask[0]);
  atomicMax(&activity[1],score);if(surface){atomicAdd(&activity[2],1u);}
  if(score>=160u){atomicAdd(&activity[3],1u);}if(score<=96u){atomicAdd(&activity[4],1u);}
  atomicAdd(&activity[6],1u);
}

// First candidate-planning rung. The accepted topology remains immutable:
// this pass publishes only the resolution requested by the CM12 surface floor,
// characteristic prediction, and retained activity history. Transfer and
// atomic generation publication consume this record in a later transaction.
@compute @workgroup_size(64)
fn planBrickResolution(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);
  let current=atomicLoad(&activity[output+12u]);
  var requested=atomicLoad(&activity[output+8u]);
  var planReasons=atomicLoad(&activity[output+9u]);
  let score=atomicLoad(&activity[output]);
  let reasons=atomicLoad(&activity[output+1u]);
  let hotEpochs=atomicLoad(&activity[output+2u]);
  let quietEpochs=atomicLoad(&activity[output+3u]);
  let surface=(reasons&1u)!=0u;let predicted=(reasons&16u)!=0u;
  let detail=(reasons&8u)!=0u;
  if(surface||predicted||score>=224u){
    requested=min(8u,2u*current);
    if(surface){planReasons=1u;}else if(predicted){planReasons=2u;}
    else{planReasons=4u;}
  }else if(atomicLoad(&activity[5])!=0u){
    requested=current;planReasons=32u;
    if(hotEpochs>=2u){
      requested=min(8u,2u*current);planReasons=8u;
    }else if(current>1u&&quietEpochs>=8u&&!detail){
      requested=current/2u;planReasons=16u;
    }
  }
  atomicStore(&activity[output+8u],requested);
  atomicStore(&activity[output+9u],planReasons);
}

// Refine-only closure of GPU-authored candidate levels. Each dispatch moves a
// violation one rung toward a valid 2:1 plan; three ordered dispatches cover
// the complete 1/2/4/8 ladder without ever coarsening a requested surface.
@compute @workgroup_size(64)
fn closePlannedResolution(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  let directions=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),
    vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var required=atomicLoad(&activity[activityRecord(brick)+8u]);
  for(var side=0u;side<6u;side+=1u){let neighborCoordinate=coordinate+directions[side];
    if(any(neighborCoordinate<vec3i(0))
      ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
    let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
      *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
    let neighbor=brickDirectoryLookup(neighborKey);if(neighbor==INVALID){continue;}
    let neighborResolution=atomicLoad(&activity[activityRecord(neighbor)+8u]);
    required=max(required,neighborResolution/2u);
  }
  atomicMax(&activity[activityRecord(brick)+8u],required);
}

fn validBrickResolution(resolution:u32)->bool{
  return resolution==1u||resolution==2u||resolution==4u||resolution==8u;
}

// Candidate ABI boundary. This pass validates the closed device-authored plan
// and records transfer-pending state beside the still-immutable accepted
// level. It cannot publish topology or make candidate fields authoritative.
@compute @workgroup_size(64)
fn validateCandidateResolution(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);
  let accepted=atomicLoad(&activity[output+12u]);
  let candidate=atomicLoad(&activity[output+8u]);
  var invalid=!validBrickResolution(accepted)||!validBrickResolution(candidate);
  if(!invalid){
    let larger=max(accepted,candidate);let smaller=min(accepted,candidate);
    invalid=larger>2u*smaller;
  }
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  let directions=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),
    vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var side=0u;side<6u;side+=1u){let neighborCoordinate=coordinate+directions[side];
    if(any(neighborCoordinate<vec3i(0))
      ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
    let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
      *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
    let neighbor=brickDirectoryLookup(neighborKey);if(neighbor==INVALID){continue;}
    let neighborCandidate=atomicLoad(&activity[activityRecord(neighbor)+8u]);
    let larger=max(candidate,neighborCandidate);let smaller=min(candidate,neighborCandidate);
    invalid=invalid||larger>2u*smaller;
  }
  atomicStore(&activity[output+13u],candidate);
  atomicStore(&activity[output+14u],select(select(0u,1u,candidate!=accepted),2u,invalid));
  atomicStore(&activity[output+15u],atomicLoad(&activity[0]));
  if(invalid){atomicOr(&activity[7],1u);}
}

fn candidateFieldIndex(channel:u32,brick:u32,local:u32)->u32{
  let capacity=p.dispatch.w*CANDIDATE_CELLS_PER_BRICK;
  return channel*capacity+brick*CANDIDATE_CELLS_PER_BRICK+local;
}

fn candidateCellVolume(brick:u32,resolution:u32,local:u32)->f32{
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let bz=key/xy;
  let remainder=key-bz*xy;let by=remainder/brickDimensions.x;
  let bx=remainder-by*brickDimensions.x;
  let z=local/(resolution*resolution);let yz=local-z*resolution*resolution;
  let y=yz/resolution;let x=yz-y*resolution;let scale=8u/resolution;
  let lower=vec3u(bx,by,bz)*8u+vec3u(x,y,z)*scale;
  let upper=min(lower+vec3u(scale),p.dimensions.xyz);
  let widths=upper-lower;return f32(widths.x*widths.y*widths.z);
}

fn finiteTransferValue(value:f32)->bool{
  return value==value&&abs(value)<3.4e38;
}

// Exact-overlap scalar and cell-momentum transfer into an isolated max-8^3
// candidate slot. The accepted topology and state remain untouched. Face-flux
// transfer, row patching, reprojection, and publication are later gates.
@compute @workgroup_size(64)
fn transferCandidateCells(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let lane=lid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);let candidateStatus=atomicLoad(&activity[output+14u]);
  let accepted=atomicLoad(&activity[output+12u]);
  let candidate=atomicLoad(&activity[output+13u]);
  let record=p.topologyOffsets2.z+4u*brick;let first=topology[record];
  let sourceCount=accepted*accepted*accepted;let candidateCount=candidate*candidate*candidate;
  var beforeMass=0.0;var beforeGamma=0.0;var beforeMomentum=vec3f(0.0);
  for(var local=lane;local<sourceCount;local+=64u){let cell=first+local;
    let volume=cellVolume(cell);let rho=state[destinationDensity()+cell];
    let velocityAt=destinationCellVelocity()+4u*cell;
    let velocity=vec3f(state[velocityAt],state[velocityAt+1u],state[velocityAt+2u]);
    beforeMass+=rho*volume;beforeGamma+=state[destinationGamma()+cell]*volume;
    beforeMomentum+=rho*volume*velocity;
  }
  var afterMass=0.0;var afterGamma=0.0;var afterMomentum=vec3f(0.0);
  for(var local=lane;local<candidateCount;local+=64u){
    let cz=local/(candidate*candidate);let yz=local-cz*candidate*candidate;
    let cy=yz/candidate;let cx=yz-cy*candidate;
    var rho=0.0;var gamma=0.0;var velocity=vec3f(0.0);var pressure=0.0;
    if(candidate<accepted){let factor=accepted/candidate;
      var volumeSum=0.0;var massSum=0.0;var momentumSum=vec3f(0.0);
      for(var dz=0u;dz<factor;dz+=1u){for(var dy=0u;dy<factor;dy+=1u){
        for(var dx=0u;dx<factor;dx+=1u){
          let sx=factor*cx+dx;let sy=factor*cy+dy;let sz=factor*cz+dz;
          let sourceLocal=sx+accepted*(sy+accepted*sz);let cell=first+sourceLocal;
          let volume=cellVolume(cell);let sourceRho=state[destinationDensity()+cell];
          let velocityAt=destinationCellVelocity()+4u*cell;
          let sourceVelocity=vec3f(state[velocityAt],state[velocityAt+1u],
            state[velocityAt+2u]);
          volumeSum+=volume;massSum+=sourceRho*volume;
          rho+=sourceRho*volume;gamma+=state[destinationGamma()+cell]*volume;
          pressure+=state[p.stateOffsets2.x+cell]*volume;
          velocity+=sourceVelocity*volume;momentumSum+=sourceRho*volume*sourceVelocity;
      }}}
      if(volumeSum>0.0){rho/=volumeSum;gamma/=volumeSum;pressure/=volumeSum;
        velocity=select(velocity/volumeSum,momentumSum/massSum,abs(massSum)>1e-12);}
    }else{
      let factor=candidate/accepted;let sx=cx/factor;let sy=cy/factor;let sz=cz/factor;
      let cell=first+sx+accepted*(sy+accepted*sz);rho=state[destinationDensity()+cell];
      gamma=state[destinationGamma()+cell];pressure=state[p.stateOffsets2.x+cell];
      let velocityAt=destinationCellVelocity()+4u*cell;
      velocity=vec3f(state[velocityAt],state[velocityAt+1u],state[velocityAt+2u]);
    }
    candidateState[candidateFieldIndex(0u,brick,local)]=rho;
    candidateState[candidateFieldIndex(1u,brick,local)]=gamma;
    candidateState[candidateFieldIndex(2u,brick,local)]=velocity.x;
    candidateState[candidateFieldIndex(3u,brick,local)]=velocity.y;
    candidateState[candidateFieldIndex(4u,brick,local)]=velocity.z;
    candidateState[candidateFieldIndex(5u,brick,local)]=pressure;
    let volume=candidateCellVolume(brick,candidate,local);
    afterMass+=rho*volume;afterGamma+=gamma*volume;
    afterMomentum+=rho*volume*velocity;
  }
  transferMassBefore[lane]=beforeMass;transferMassAfter[lane]=afterMass;
  transferGammaDelta[lane]=afterGamma-beforeGamma;
  transferMomentumXDelta[lane]=afterMomentum.x-beforeMomentum.x;
  transferMomentumYDelta[lane]=afterMomentum.y-beforeMomentum.y;
  transferMomentumZDelta[lane]=afterMomentum.z-beforeMomentum.z;
  workgroupBarrier();var width=32u;loop{if(lane<width){
    transferMassBefore[lane]+=transferMassBefore[lane+width];
    transferMassAfter[lane]+=transferMassAfter[lane+width];
    transferGammaDelta[lane]+=transferGammaDelta[lane+width];
    transferMomentumXDelta[lane]+=transferMomentumXDelta[lane+width];
    transferMomentumYDelta[lane]+=transferMomentumYDelta[lane+width];
    transferMomentumZDelta[lane]+=transferMomentumZDelta[lane+width];
  }workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane!=0u){return;}
  if(candidateStatus!=1u){
    atomicStore(&activity[output+23u],select(0u,2u,candidateStatus==2u));return;
  }
  let massError=transferMassAfter[0]-transferMassBefore[0];
  let gammaError=transferGammaDelta[0];let momentumError=vec3f(
    transferMomentumXDelta[0],transferMomentumYDelta[0],transferMomentumZDelta[0]);
  let tolerance=max(1e-4,1e-6*abs(transferMassBefore[0]));
  let valid=finiteTransferValue(transferMassBefore[0])
    &&finiteTransferValue(transferMassAfter[0])&&finiteTransferValue(gammaError)
    &&all(momentumError==momentumError)&&abs(massError)<=tolerance
    &&abs(gammaError)<=max(1e-3,tolerance)
    &&all(abs(momentumError)<=vec3f(max(1e-3,tolerance)));
  atomicStore(&activity[output+16u],bitcast<u32>(transferMassBefore[0]));
  atomicStore(&activity[output+17u],bitcast<u32>(transferMassAfter[0]));
  atomicStore(&activity[output+18u],bitcast<u32>(massError));
  atomicStore(&activity[output+19u],bitcast<u32>(gammaError));
  atomicStore(&activity[output+20u],bitcast<u32>(momentumError.x));
  atomicStore(&activity[output+21u],bitcast<u32>(momentumError.y));
  atomicStore(&activity[output+22u],bitcast<u32>(momentumError.z));
  atomicStore(&activity[output+23u],select(2u,1u,valid));
  if(!valid){atomicOr(&activity[7],2u);}
}

@compute @workgroup_size(64)
fn prepareCandidateFaceReceipts(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}let output=activityRecord(brick);
  for(var side=0u;side<6u;side+=1u){atomicStore(&activity[output+24u+side],0u);}
  atomicStore(&activity[output+30u],0u);
  let candidateStatus=atomicLoad(&activity[output+14u]);
  atomicStore(&activity[output+31u],select(select(0u,1u,candidateStatus==1u),2u,
    candidateStatus==2u));
}

// Area-average every authoritative accepted normal flux into the candidate's
// exterior face patches. Each row is claimed once per incident brick, so the
// six integrated exterior fluxes are exact transfer receipts rather than a
// cell-velocity reconstruction.
@compute @workgroup_size(64)
fn transferCandidateFaces(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let side=wid.y;let lane=lid.x;
  if(brick>=p.dispatch.w||side>=6u){return;}
  let output=activityRecord(brick);let candidate=atomicLoad(&activity[output+13u]);
  let record=p.topologyOffsets2.z+4u*brick;let first=topology[record];
  let count=topology[record+1u];let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let bz=key/xy;
  let remainder=key-bz*xy;let by=remainder/brickDimensions.x;
  let bx=remainder-by*brickDimensions.x;let origin=vec3f(vec3u(bx,by,bz)*8u);
  let axis=side/2u;let positive=(side&1u)!=0u;
  let tangent0=select(select(0u,0u,axis==2u),1u,axis==0u);
  let tangent1=select(2u,1u,axis==2u);
  let plane=origin[axis]+select(0.0,8.0,positive);let scale=8.0/f32(candidate);
  var flux=0.0;var area=0.0;
  if(lane<candidate*candidate){
    for(var local=0u;local<count;local+=1u){let cell=first+local;
      for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
        let row=incidenceRow(incidence);if(rowAxis(row)!=axis){continue;}
        let center=rowCenter(row);if(abs(center[axis]-plane)>1e-4){continue;}
        var claimant=INVALID;let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
        for(var term=begin;term<end;term+=1u){let termOwner=termCell(term);
          if(cellBrick(termOwner)==brick){claimant=min(claimant,termOwner);}}
        if(cell!=claimant){continue;}
        let p0=min(candidate-1u,u32(max(0.0,floor(
          (center[tangent0]-origin[tangent0])/scale))));
        let p1=min(candidate-1u,u32(max(0.0,floor(
          (center[tangent1]-origin[tangent1])/scale))));
        if(lane!=p0+candidate*p1){continue;}
        let rowAreaValue=rowArea(row);area+=rowAreaValue;
        flux+=state[destinationFaceVelocity()+row]*rowAreaValue
          *select(-1.0,1.0,positive);
      }
    }
    candidateState[candidateFieldIndex(6u+side,brick,lane)]=select(0.0,flux/area,area>0.0);
  }
  reduceA[lane]=flux;reduceB[lane]=select(0.0,flux/area,area>0.0)*area;
  workgroupBarrier();var width=32u;loop{if(lane<width){
    reduceA[lane]+=reduceA[lane+width];reduceB[lane]+=reduceB[lane+width];
  }workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane!=0u){return;}let error=reduceB[0]-reduceA[0];
  atomicStore(&activity[output+24u+side],bitcast<u32>(error));
  atomicMax(&activity[output+30u],bitcast<u32>(abs(error)));
  if(atomicLoad(&activity[output+14u])==1u){
    let valid=finiteTransferValue(error)&&abs(error)<=max(1e-4,1e-6*abs(reduceA[0]));
    if(!valid){atomicStore(&activity[output+31u],2u);atomicOr(&activity[7],4u);}
  }
}

// Publish only the directional free-surface stencil and swept receivers from
// the immutable activity snapshot. Compare-exchange makes publication
// single-writer; all following frame dispatches observe the active bit.
@compute @workgroup_size(64)
fn activateSweptReceivers(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w||brickActive(brick)){return;}
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  var requested=false;
  for(var dz=-1;dz<=1;dz+=1){for(var dy=-1;dy<=1;dy+=1){
    for(var dx=-1;dx<=1;dx+=1){if(dx==0&&dy==0&&dz==0){continue;}
      let neighborCoordinate=coordinate+vec3i(dx,dy,dz);
      if(any(neighborCoordinate<vec3i(0))
        ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
      let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
        *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
      let neighbor=brickDirectoryLookup(neighborKey);
      if(neighbor==INVALID||!brickActive(neighbor)){continue;}
      let bit=u32(1-dx)+3u*u32(1-dy)+9u*u32(1-dz);
      requested=requested
        ||(atomicLoad(&activity[activityRecord(neighbor)+32u])&(1u<<bit))!=0u;
  }}}
  if(!requested){return;}
  let output=activityRecord(brick);
  // A swept receiver contains the moving free surface by construction. It is
  // requested fine before publication; closePlannedResolution then grows the
  // 4/2/1 support skirt around it without weakening this hard floor.
  atomicStore(&activity[output+8u],8u);
  atomicStore(&activity[output+9u],1u);
  let claimed=atomicCompareExchangeWeak(&activity[output+10u],0u,1u);
  if(claimed.exchanged){
    atomicStore(&activity[output+11u],atomicLoad(&activity[0]));
    atomicAdd(&activity[8],1u);atomicAdd(&activity[9],1u);
    atomicAdd(&activity[10],1u);atomicAdd(&activity[11],topology[record+1u]);
  }
}

// Retire every exactly mass-empty brick outside the directional interface
// stencil and swept receiver mask. Exact occupancy retains its own brick, so
// retirement cannot discard even a small conservative CM12 receipt.
@compute @workgroup_size(64)
fn retireUnsupportedEmptyBricks(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w||!brickActive(brick)){return;}
  let output=activityRecord(brick);
  if((atomicLoad(&activity[output+1u])&64u)!=0u){return;}
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  for(var dz=-1;dz<=1;dz+=1){for(var dy=-1;dy<=1;dy+=1){
    for(var dx=-1;dx<=1;dx+=1){if(dx==0&&dy==0&&dz==0){continue;}
      let neighborCoordinate=coordinate+vec3i(dx,dy,dz);
      if(any(neighborCoordinate<vec3i(0))
        ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
      let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
        *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
      let neighbor=brickDirectoryLookup(neighborKey);if(neighbor==INVALID){continue;}
      let bit=u32(1-dx)+3u*u32(1-dy)+9u*u32(1-dz);
      if((atomicLoad(&activity[activityRecord(neighbor)+32u])&(1u<<bit))!=0u){return;}
    }
  }}
  let retired=atomicCompareExchangeWeak(&activity[output+10u],1u,0u);
  if(retired.exchanged){
    atomicSub(&activity[8],1u);atomicSub(&activity[11],topology[record+1u]);
    atomicAdd(&activity[10],1u);
  }
}

@compute @workgroup_size(64)
fn classifyPresentationBricks(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  if(!brickActive(brick)){state[p.stateOffsets4.z+brick]=0.0;return;}
  let record=p.topologyOffsets2.z+4u*brick;let first=topology[record];let count=topology[record+1u];
  var wet=false;
  for(var at=first;at<first+count;at+=1u){
    wet=wet||state[destinationDensity()+at]>CM12_DRY_CELL_THRESHOLD;
  }
  state[p.stateOffsets4.z+brick]=select(0.0,1.0,wet);
}

// Publish one compact renderer page per workgroup. Metadata is sorted by the
// logical 4^3 page key, so consumers binary-search it without a world-sized
// direct table. Missing authored-world pages are implicit air and cost no
// storage, initialization, or frame dispatch.
@compute @workgroup_size(64)
fn publishSparseLevelSet(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let page=wid.x;let pageCount=arrayLength(&fineMetadata)/4u;
  if(page>=pageCount||lane>=64u){return;}
  let source=fineMetadata[4u*page+3u];let brick=source>>3u;
  let octant=source&7u;let local=vec3u(lane%4u,(lane/4u)%4u,lane/16u);
  let brickRecord=p.topologyOffsets2.z+4u*brick;
  let brickKey=topology[brickRecord+3u];let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let brickXY=brickDimensions.x*brickDimensions.y;let brickZ=brickKey/brickXY;
  let brickRemainder=brickKey-brickZ*brickXY;let brickY=brickRemainder/brickDimensions.x;
  let brickX=brickRemainder-brickY*brickDimensions.x;
  let pageOffset=vec3u(octant&1u,(octant>>1u)&1u,(octant>>2u)&1u)*4u;
  let q=vec3u(brickX,brickY,brickZ)*8u+pageOffset+local;var phi=4.0*p.frame.y;
  if(brick<p.dispatch.w&&state[p.stateOffsets4.z+brick]>0.5&&all(q<p.dimensions.xyz)){
    let first=topology[brickRecord];let count=topology[brickRecord+1u];
    let resolution=topology[brickRecord+2u];let scale=8u/resolution;
    let cellLocal=(pageOffset+local)/scale;
    let valid=(min(p.dimensions.xyz-vec3u(brickX,brickY,brickZ)*8u
      +vec3u(scale-1u),vec3u(8u)))/scale;
    let cell=first+cellLocal.x+valid.x*(cellLocal.y+valid.y*cellLocal.z);
    if(cell<first+count){phi=presentationPhi(cell);
      if(scale>1u){phi=interpolatedPresentationPhi(vec3i(q),i32(scale));}}
  }
  let flags=1u|select(0u,16u,phi<0.0);
  fineSamples[page*64u+lane]=(pack2x16float(vec2f(phi,0.0))&0xffffu)|(flags<<16u);
}
`;
