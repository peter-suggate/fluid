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
  stateOffsets5:vec4u,      // fine presentation density A/B, enabled, reserved
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

fn tf(index:u32)->f32{return bitcast<f32>(topology[index]);}
fn sourceDensity()->u32{return select(p.stateOffsets0.x,p.stateOffsets0.y,p.frame.w>0.5);}
fn destinationDensity()->u32{return select(p.stateOffsets0.y,p.stateOffsets0.x,p.frame.w>0.5);}
fn sourceGamma()->u32{return select(p.stateOffsets0.z,p.stateOffsets0.w,p.frame.w>0.5);}
fn destinationGamma()->u32{return select(p.stateOffsets0.w,p.stateOffsets0.z,p.frame.w>0.5);}
fn sourceCellVelocity()->u32{return select(p.stateOffsets1.x,p.stateOffsets1.y,p.frame.w>0.5);}
fn destinationCellVelocity()->u32{return select(p.stateOffsets1.y,p.stateOffsets1.x,p.frame.w>0.5);}
fn sourceFaceVelocity()->u32{return select(p.stateOffsets1.z,p.stateOffsets1.w,p.frame.w>0.5);}
fn destinationFaceVelocity()->u32{return select(p.stateOffsets1.w,p.stateOffsets1.z,p.frame.w>0.5);}
fn sourcePresentationDensity()->u32{return select(p.stateOffsets5.x,p.stateOffsets5.y,p.frame.w>0.5);}
fn destinationPresentationDensity()->u32{return select(p.stateOffsets5.y,p.stateOffsets5.x,p.frame.w>0.5);}

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
fn rowExteriorPhi(id:u32)->f32{return tf(rowBase(id)+7u);}
fn rowCenter(id:u32)->vec3f{let b=rowBase(id);return vec3f(tf(b+8u),tf(b+9u),tf(b+10u));}
fn termCell(index:u32)->u32{return topology[p.topologyOffsets.z+2u*index];}
fn termCoefficient(index:u32)->f32{return tf(p.topologyOffsets.z+2u*index+1u);}
fn incidenceBegin(cell:u32)->u32{return topology[p.topologyOffsets.w+cell];}
fn incidenceEnd(cell:u32)->u32{return topology[p.topologyOffsets.w+cell+1u];}
fn incidenceRow(index:u32)->u32{return topology[p.topologyOffsets2.x+2u*index];}
fn incidenceTerm(index:u32)->u32{return topology[p.topologyOffsets2.x+2u*index+1u];}
fn isLiquid(cell:u32)->bool{return state[p.stateOffsets2.w+cell]>0.5;}

fn minmod(a:f32,b:f32)->f32{
  if(a*b<=0.0){return 0.0;}return sign(a)*min(abs(a),abs(b));
}

fn axisNeighborValue(cell:u32,axis:u32,side:f32,offset:u32)->f32{
  var weighted=0.0;var weight=0.0;
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
    let row=incidenceRow(at);if(rowAxis(row)!=axis){continue;}
    let ownCoefficient=termCoefficient(incidenceTerm(at));
    // A negative G coefficient owns the row's negative side and therefore
    // reaches a positive-side neighbour; the converse reaches negative.
    if((side>0.0&&ownCoefficient>=0.0)||(side<0.0&&ownCoefficient<=0.0)){continue;}
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let other=termCell(term);if(other==cell){continue;}
      let w=abs(termCoefficient(term));weighted+=w*state[offset+other];weight+=w;}
  }
  return select(state[offset+cell],weighted/weight,weight>0.0);
}

fn upwindRowValue(row:u32,velocity:f32,offset:u32)->f32{
  let donorSide=select(1.0,-1.0,velocity>=0.0);
  let faceDirection=-donorSide;
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  var weighted=0.0;var weight=0.0;
  for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
    if(coefficient*donorSide<=0.0){continue;}let donor=termCell(term);
    let center=state[offset+donor];
    let negative=axisNeighborValue(donor,rowAxis(row),-1.0,offset);
    let positive=axisNeighborValue(donor,rowAxis(row),1.0,offset);
    var slope=minmod(center-negative,positive-center);
    // At the numerical transport front, retaining the limited slope halves an
    // already sub-percent donor on every coarse hop. Use the monotone donor
    // value there, matching CM12's transport-shell reach without perturbing
    // the resolved interface profile.
    if((offset==sourceDensity()||offset==destinationDensity())&&center<0.01){slope=0.0;}
    let reconstructed=center+0.5*faceDirection*slope;
    let w=abs(coefficient);weighted+=w*clamp(reconstructed,0.0,
      select(CM12_GAMMA_MAX,1.5,offset==destinationDensity()||offset==sourceDensity()));weight+=w;
  }
  return select(0.0,weighted/weight,weight>0.0);
}

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

// Match materializeAdaptiveMassPresentationAtlas's presentation-only linear
// prolongation. Density remains piecewise constant; only phi is reconstructed
// inside a coarse leaf so marching cubes sees the same finest-lattice samples
// as the former CPU publisher.
fn presentationFaceNeighborSample(cell:u32,axis:u32,side:i32)->vec3f{
  let b=cellBase(cell);let scale=i32(round(tf(b+10u)));
  let lower=vec3i(vec3u(topology[b+7u],topology[b+8u],topology[b+9u]));
  let face=lower[axis]+select(scale,-1,side<0);
  if(face<0||face>=i32(p.dimensions[axis])){return vec3f(0.0);}
  let tangentA=select(0u,1u,axis==0u);
  let tangentB=select(2u,1u,axis==2u);
  let center=vec3f(tf(b),tf(b+1u),tf(b+2u));
  var value=0.0;var distance=0.0;var samples=0.0;
  for(var tangentBOffset=0;tangentBOffset<scale;tangentBOffset+=1){
    for(var tangentAOffset=0;tangentAOffset<scale;tangentAOffset+=1){
      var q=lower;q[axis]=face;q[tangentA]+=tangentAOffset;q[tangentB]+=tangentBOffset;
      let neighbor=presentationOwnerCellAt(q);
      if(neighbor==INVALID){
        value+=4.0*p.frame.y;
        distance+=abs(f32(face)+0.5-center[axis]);
      }else{
        let neighborBase=cellBase(neighbor);
        value+=presentationPhi(neighbor);
        distance+=abs(tf(neighborBase+axis)-center[axis]);
      }
      samples+=1.0;
    }
  }
  return vec3f(value/samples,distance/samples,1.0);
}

fn reconstructedPresentationPhi(cell:u32,coordinate:vec3i)->f32{
  let b=cellBase(cell);let scale=i32(round(tf(b+10u)));let phi=presentationPhi(cell);
  if(scale<=1){return phi;}
  var gradient=vec3f(0.0);var maximumNeighborDelta=0.0;
  for(var axis=0u;axis<3u;axis+=1u){
    let negative=presentationFaceNeighborSample(cell,axis,-1);
    let positive=presentationFaceNeighborSample(cell,axis,1);
    if(negative.z>0.5){maximumNeighborDelta=max(maximumNeighborDelta,abs(negative.x-phi));}
    if(positive.z>0.5){maximumNeighborDelta=max(maximumNeighborDelta,abs(positive.x-phi));}
    if(negative.z>0.5&&positive.z>0.5){
      gradient[axis]=(positive.x-negative.x)/max(positive.y+negative.y,1e-30);
    }else if(positive.z>0.5){
      gradient[axis]=(positive.x-phi)/max(positive.y,1e-30);
    }else if(negative.z>0.5){
      gradient[axis]=(phi-negative.x)/max(negative.y,1e-30);
    }
  }
  let maximumOffset=0.5*f32(scale-1);
  let predictedMaximumDelta=maximumOffset*(abs(gradient.x)+abs(gradient.y)+abs(gradient.z));
  if(predictedMaximumDelta>maximumNeighborDelta&&predictedMaximumDelta>0.0){
    gradient*=maximumNeighborDelta/predictedMaximumDelta;
  }
  let lower=vec3i(vec3u(topology[b+7u],topology[b+8u],topology[b+9u]));
  let offset=vec3f(coordinate-lower)+vec3f(0.5)-vec3f(0.5*f32(scale));
  return phi+dot(gradient,offset);
}

fn sampleScalar(position:vec3f,offset:u32,empty:f32)->f32{
  let probe=ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe!=INVALID){let b=cellBase(probe);spans=vec3f(tf(b+4u),tf(b+5u),tf(b+6u));}
  let clamped=clamp(position,0.5*spans,vec3f(p.dimensions.xyz)-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);let lower=vec3i(floor(shifted));let fraction=fract(shifted);
  var result=0.0;
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let lattice=spans*(vec3f(lower+vec3i(dx,dy,dz))+vec3f(0.5));
    let q=vec3i(floor(lattice));let cell=ownerCellAt(q);var value=empty;
    if(cell!=INVALID){value=state[offset+cell];}
    let wx=select(1.0-fraction.x,fraction.x,dx==1);
    let wy=select(1.0-fraction.y,fraction.y,dy==1);
    let wz=select(1.0-fraction.z,fraction.z,dz==1);result+=wx*wy*wz*value;
  }}}return result;
}

fn sampleFinePresentationDensity(position:vec3f,offset:u32)->f32{
  let clamped=clamp(position,vec3f(0.5),vec3f(p.dimensions.xyz)-vec3f(0.5));
  let shifted=clamped-vec3f(0.5);let lower=vec3i(floor(shifted));let fraction=fract(shifted);
  var result=0.0;
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let q=clamp(lower+vec3i(dx,dy,dz),vec3i(0),vec3i(p.dimensions.xyz)-vec3i(1));
    let dense=u32(q.x)+p.dimensions.x*(u32(q.y)+p.dimensions.y*u32(q.z));
    let wx=select(1.0-fraction.x,fraction.x,dx==1);
    let wy=select(1.0-fraction.y,fraction.y,dy==1);
    let wz=select(1.0-fraction.z,fraction.z,dz==1);
    result+=wx*wy*wz*state[offset+dense];
  }}}return result;
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

fn traceDeparture(position:vec3f)->vec3f{
  let initial=sampleVelocity(position);let midpoint=position-0.5*p.frame.x*initial;
  return clamp(position-p.frame.x*sampleVelocity(midpoint),vec3f(0.5),vec3f(p.dimensions.xyz)-vec3f(0.5));
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
fn initializePresentationDensity(@builtin(global_invocation_id)gid:vec3u){
  let dense=gid.x;if(dense>=p.counts.w){return;}
  let x=dense%p.dimensions.x;let yz=dense/p.dimensions.x;
  let y=yz%p.dimensions.y;let z=yz/p.dimensions.y;
  let coordinate=vec3i(i32(x),i32(y),i32(z));
  let cell=presentationOwnerCellAt(coordinate);
  var density=0.0;
  if(cell!=INVALID){density=clamp(CM12_LIQUID_ISOVALUE-
    reconstructedPresentationPhi(cell,coordinate)/(4.0*p.frame.y),0.0,1.5);}
  state[p.stateOffsets5.x+dense]=density;
  state[p.stateOffsets5.y+dense]=density;
}

@compute @workgroup_size(64)
fn transportPresentationDensity(@builtin(global_invocation_id)gid:vec3u){
  let dense=gid.x;if(dense>=p.counts.w){return;}
  let x=dense%p.dimensions.x;let yz=dense/p.dimensions.x;
  let y=yz%p.dimensions.y;let z=yz/p.dimensions.y;
  let position=vec3f(f32(x)+0.5,f32(y)+0.5,f32(z)+0.5);
  let departure=traceDeparture(position);
  state[destinationPresentationDensity()+dense]=clamp(
    sampleFinePresentationDensity(departure,sourcePresentationDensity()),0.0,1.5);
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

@compute @workgroup_size(64)
fn injectPresentationLiquid(@builtin(global_invocation_id)gid:vec3u){
  let dense=gid.x;if(dense>=p.counts.w){return;}
  let x=dense%p.dimensions.x;let yz=dense/p.dimensions.x;
  let y=yz%p.dimensions.y;let z=yz/p.dimensions.y;
  let center=vec3f(f32(x)+0.5,f32(y)+0.5,f32(z)+0.5);
  let q=(center-p.injectionCenter.xyz)/max(p.injectionRadius.xyz,vec3f(1e-6));
  let signed=length(q)-1.0;
  let coverage=clamp(0.5-signed*min(p.injectionRadius.x,
    min(p.injectionRadius.y,p.injectionRadius.z)),0.0,1.0);
  state[p.stateOffsets5.x+dense]=max(state[p.stateOffsets5.x+dense],coverage);
  state[p.stateOffsets5.y+dense]=max(state[p.stateOffsets5.y+dense],coverage);
}

@compute @workgroup_size(64)
fn transportCells(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;if(id>=p.counts.x){return;}
  let rhoOffset=sourceDensity();let gammaOffset=sourceGamma();
  let b=cellBase(id);let departure=traceDeparture(vec3f(tf(b),tf(b+1u),tf(b+2u)));
  let nextRho=clamp(sampleScalar(departure,rhoOffset,0.0),0.0,1.5);
  let nextGamma=select(clamp(sampleScalar(departure,gammaOffset,1.0),
    0.0,CM12_GAMMA_MAX),1.0,nextRho<1e-5);
  state[destinationDensity()+id]=nextRho;
  state[destinationGamma()+id]=nextGamma;
}

@compute @workgroup_size(64)
fn measureTransportMass(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  var before=0.0;var after=0.0;if(gid.x<p.counts.x){let volume=cellVolume(gid.x);
    before=volume*state[sourceDensity()+gid.x];
    after=volume*state[destinationDensity()+gid.x];}
  reduceA[lid.x]=before;reduceB[lid.x]=after;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];reduceB[lid.x]+=reduceB[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){partials[wid.x]=vec2f(reduceA[0],reduceB[0]);}
}

@compute @workgroup_size(64)
fn reduceTransportMass(@builtin(local_invocation_id)lid:vec3u){
  var before=0.0;var after=0.0;for(var at=lid.x;at<p.dispatch.x;at+=64u){
    before+=partials[at].x;after+=partials[at].y;}
  reduceA[lid.x]=before;reduceB[lid.x]=after;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];reduceB[lid.x]+=reduceB[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){scalars[5]=select(1.0,reduceA[0]/reduceB[0],reduceB[0]>1e-20);}
}

@compute @workgroup_size(64)
fn scaleTransport(@builtin(global_invocation_id)gid:vec3u){let id=gid.x;if(id>=p.counts.x){return;}
  state[destinationDensity()+id]=clamp(state[destinationDensity()+id]*scalars[5],0.0,1.5);}

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

@compute @workgroup_size(64)
fn classifyPresentationBricks(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  let record=p.topologyOffsets2.z+2u*brick;let first=topology[record];let count=topology[record+1u];
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
    var backgroundPhi=4.0*p.frame.y;
    if(p.stateOffsets5.z!=0u){let displayRho=state[destinationPresentationDensity()+dense];
      if(displayRho>1e-5){backgroundPhi=(CM12_LIQUID_ISOVALUE-displayRho)*4.0*p.frame.y;}}
    textureStore(levelSetTexture,coordinate,vec4f(backgroundPhi));
    textureStore(ownerTexture,coordinate,vec4u(topology[p.topologyOffsets2.w],
      topology[p.topologyOffsets2.w+1u],0u,0u));
    textureStore(velocityTexture,coordinate,vec4f(0.0));textureStore(pressureTexture,coordinate,vec4f(0.0));
    textureStore(divergenceTexture,coordinate,vec4f(0.0));return;}
  let rho=state[destinationDensity()+presentationCell];
  textureStore(densityTexture,coordinate,vec4f(rho));
  let physicalPhi=reconstructedPresentationPhi(presentationCell,coordinate);
  let displayRho=state[destinationPresentationDensity()+dense];
  let displayPhi=(CM12_LIQUID_ISOVALUE-displayRho)*4.0*p.frame.y;
  textureStore(levelSetTexture,coordinate,vec4f(
    select(physicalPhi,displayPhi,p.stateOffsets5.z!=0u)));
  textureStore(ownerTexture,coordinate,vec4u(topology[owner+1u],topology[owner+2u],0u,0u));
  let v=destinationCellVelocity()+4u*presentationCell;
  textureStore(velocityTexture,coordinate,vec4f(state[v]*p.frame.y,state[v+1u]*p.frame.y,state[v+2u]*p.frame.y,0.0));
  textureStore(pressureTexture,coordinate,vec4f(state[p.stateOffsets2.x+presentationCell]*p.frame.z));
  textureStore(divergenceTexture,coordinate,vec4f(state[p.stateOffsets4.y+presentationCell]));
}
`;
