import { inflowBoundaryWGSL } from "./inflow-boundary";

export const tallCellComputeShader = /* wgsl */ `
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
  boundary: vec4f,
  tall: vec4f,
  inflowPositionRadius: vec4f,
  inflowVelocityLength: vec4f,
  // x: frame-averaged inflow strength; remaining lanes are reserved.
  inflowTiming: vec4f,
}
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var pressureIn: texture_3d<f32>;
@group(0) @binding(3) var pressureOut: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var volumeIn: texture_3d<f32>;
@group(0) @binding(5) var volumeOut: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var columnBaseIn: texture_2d<f32>;
@group(0) @binding(8) var columnBaseOut: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var<storage,read_write> reductions:array<atomic<u32>,32>;
struct RigidBody {
  positionShape: vec4f,
  dimensions: vec4f,
  orientation: vec4f,
  linearVelocity: vec4f,
  angularVelocity: vec4f,
}
@group(0) @binding(10) var<storage,read> rigidBodies:array<RigidBody,12>;
@group(0) @binding(11) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(0) @binding(12) var<storage,read_write> nextColumnBases:array<u32>;
	@group(0) @binding(13) var predictedVelocityIn:texture_3d<f32>;
	@group(0) @binding(14) var reversedVelocityIn:texture_3d<f32>;
	@group(0) @binding(15) var solidFractionIn:texture_3d<f32>;
@group(0) @binding(16) var<storage,read_write> smoothedColumnBases:array<u32>;
// Persistent across frames: max CFL, chosen phi substeps, chosen phi dt, and
// cumulative executed phi steps.
@group(0) @binding(17) var<storage,read_write> governor:array<atomic<u32>,4>;
// Eight 16-byte indirect slots. The fourth word pads each slot for inspection.
@group(0) @binding(18) var<storage,read_write> phiDispatchArgs:array<atomic<u32>,32>;

fn packedDims()->vec3i{return vec3i(textureDimensions(volumeIn));}
fn fineDims()->vec3i{let d=packedDims();return vec3i(d.x,i32(round(params.boundary.w)),d.z);}
fn inflowGridDims()->vec3i{return fineDims();}
fn regularLayers()->i32{return i32(round(params.tall.x));}
fn baseAt(x:i32,z:i32)->i32{let d=packedDims();if(x<0||x>=d.x||z<0||z>=d.z){return 0;}return i32(round(textureLoad(columnBaseIn,vec2i(x,z),0).x));}
fn validPacked(q:vec3i)->bool{let d=packedDims();return all(q>=vec3i(0))&&all(q<d);}
fn validWorld(q:vec3i)->bool{let d=fineDims();return all(q>=vec3i(0))&&all(q<d);}
fn activeSample(id:vec3i)->bool{
  if(!validPacked(id)){return false;}let base=baseAt(id.x,id.z);
  if(id.y<2){return base>0;}return base+id.y-2<fineDims().y;
}
fn representedWorld(q:vec3i)->bool{
  if(!validWorld(q)){return false;}return q.y<baseAt(q.x,q.z)+regularLayers();
}
fn sampleY(id:vec3i)->f32{
  let base=baseAt(id.x,id.z);if(id.y==0){return 0.5;}if(id.y==1){return max(0.5,f32(base)-0.5);}return f32(base+id.y-2)+0.5;
}
fn samplePoint(id:vec3i)->vec3f{return vec3f(f32(id.x)+0.5,sampleY(id),f32(id.z)+0.5);}
fn worldFromPoint(p:vec3f)->vec3f{let h=params.cellGravity.xyz;return vec3f(-0.5*params.container.x+p.x*h.x,p.y*h.y,-0.5*params.container.z+p.z*h.z);}
${inflowBoundaryWGSL}
fn finiteScalar(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn storeWorldLocation(id:vec3i,slot:u32){let q=vec3i(floor(samplePoint(id)));atomicStore(&reductions[slot],u32(max(q.x,0)));atomicStore(&reductions[slot+1u],u32(max(q.y,0)));atomicStore(&reductions[slot+2u],u32(max(q.z,0)));}
fn updatePositiveMaximum(value:f32,valueSlot:u32,locationSlot:u32,id:vec3i){
  if(!finiteScalar(value)||value<0.0){atomicAdd(&reductions[20],1u);return;}
  let bits=bitcast<u32>(value);let previous=atomicMax(&reductions[valueSlot],bits);if(bits>previous){storeWorldLocation(id,locationSlot);}
}
fn updatePositiveMaximumOnly(value:f32,valueSlot:u32){if(!finiteScalar(value)||value<0.0){atomicAdd(&reductions[20],1u);return;}atomicMax(&reductions[valueSlot],bitcast<u32>(value));}

// Tall Cells Eq. 4/5: phi is a point-sampled signed distance. The two tall
// endpoints are independent values and virtual cubic samples inside the tall
// cell use the paper's linear reconstruction.
fn phiCell(q:vec3i)->f32{
  let air=5.0*min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  if(!validWorld(q)){return air;}let base=baseAt(q.x,q.z);
  if(q.y<base&&base>0){let t=clamp(f32(q.y)/f32(max(base-1,1)),0.0,1.0);return mix(textureLoad(volumeIn,vec3i(q.x,0,q.z),0).x,textureLoad(volumeIn,vec3i(q.x,1,q.z),0).x,t);}
  let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return air;}return textureLoad(volumeIn,vec3i(q.x,packedY,q.z),0).x;
}
fn pointSamplePhi(id:vec3i)->f32{return textureLoad(volumeIn,id,0).x;}
fn liquidCell(q:vec3i)->bool{return phiCell(q)<=0.0;}
fn occupancyFromPhi(phi:f32)->f32{let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));return clamp(0.5-phi/h,0.0,1.0);}
fn columnHighestWetCell(x:i32,z:i32)->i32{for(var y=fineDims().y-1;y>=0;y-=1){if(phiCell(vec3i(x,y,z))<=0.0){return y;}}return -1;}
fn samplePhi(p:vec3f)->f32{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);
  let a=mix(mix(phiCell(b),phiCell(b+vec3i(1,0,0)),f.x),mix(phiCell(b+vec3i(0,1,0)),phiCell(b+vec3i(1,1,0)),f.x),f.y);
  let c=mix(mix(phiCell(b+vec3i(0,0,1)),phiCell(b+vec3i(1,0,1)),f.x),mix(phiCell(b+vec3i(0,1,1)),phiCell(b+vec3i(1,1,1)),f.x),f.y);return mix(a,c,f.z);
}
fn validVelocityCell(q:vec3i)->vec3f{
  let base=baseAt(q.x,q.z);
  // Paper Eq 5: quantities inside a tall cell interpolate LINEARLY between
  // the bottom and top endpoint samples. A piecewise-constant reconstruction
  // (bottom dof everywhere) made the vertical velocity derivative vanish
  // inside the store, so the column-integrated constraint admitted a uniform
  // free-fall mode balanced by fake lateral spreading — the dam-break dome
  // (2026-07-16 collapse audit). The linear profile makes divergence linear
  // in y inside the store, so zeroing it at both endpoints zeroes every row.
  if(q.y<base&&base>0){let t=clamp(f32(q.y)/f32(max(base-1,1)),0.0,1.0);return mix(textureLoad(velocityIn,vec3i(q.x,0,q.z),0).xyz,textureLoad(velocityIn,vec3i(q.x,1,q.z),0).xyz,t);}
  let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return vec3f(0.0);}return textureLoad(velocityIn,vec3i(q.x,packedY,q.z),0).xyz;
}
fn velocityCell(q:vec3i)->vec3f{
  let d=fineDims();let c=clamp(q,vec3i(0),d-vec3i(1));var v=validVelocityCell(c);
  // Restricted tall cells use the same wall convention as the cubic solver:
  // zero the outward normal, but retain tangential motion at a free-slip wall.
  if(q.x<0||q.x>=d.x){v.x=0.0;}if(q.y<0||q.y>=d.y){v.y=0.0;}if(q.z<0||q.z>=d.z){v.z=0.0;}return v;
}
fn velocityStateCell(q:vec3i)->vec4f{
  if(!validWorld(q)){return vec4f(0.0);}let base=baseAt(q.x,q.z);
  if(q.y<base&&base>0){
    // Eq 5 linear reconstruction. The known flag only blends when both
    // endpoint dofs are known; otherwise the nearer endpoint owns the row so
    // extrapolation never averages against stale unknown data.
    let bottom=textureLoad(velocityIn,vec3i(q.x,0,q.z),0);let top=textureLoad(velocityIn,vec3i(q.x,1,q.z),0);
    if(bottom.w>0.5&&top.w>0.5){let t=clamp(f32(q.y)/f32(max(base-1,1)),0.0,1.0);return vec4f(mix(bottom.xyz,top.xyz,t),1.0);}
    return select(bottom,top,q.y==base-1);
  }
  let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return vec4f(0.0);}return textureLoad(velocityIn,vec3i(q.x,packedY,q.z),0);}
fn diffusionVelocity(q:vec3i)->vec3f{
  if(validWorld(q)){return validVelocityCell(q);}let c=clamp(q,vec3i(0),fineDims()-vec3i(1));let v=validVelocityCell(c);
  if(params.boundary.y>0.5){return -v;}return v;
}
	fn pressureCell(q:vec3i)->f32{
  if(!validWorld(q)){return 0.0;}let base=baseAt(q.x,q.z);
  if(q.y<base&&base>0){let t=clamp(f32(q.y)/f32(max(base-1,1)),0.0,1.0);return mix(textureLoad(pressureIn,vec3i(q.x,0,q.z),0).x,textureLoad(pressureIn,vec3i(q.x,1,q.z),0).x,t);}
  let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return 0.0;}return textureLoad(pressureIn,vec3i(q.x,packedY,q.z),0).x;
	}
	fn solidFractionCell(q:vec3i)->f32{
	  if(!validWorld(q)){return 1.0;}let base=baseAt(q.x,q.z);
	  if(q.y<base&&base>0){let t=clamp(f32(q.y)/f32(max(base-1,1)),0.0,1.0);return mix(textureLoad(solidFractionIn,vec3i(q.x,0,q.z),0).x,textureLoad(solidFractionIn,vec3i(q.x,1,q.z),0).x,t);}
	  let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return 1.0;}return textureLoad(solidFractionIn,vec3i(q.x,packedY,q.z),0).x;
	}
fn sampleVelocity(p:vec3f)->vec3f{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);
  let a=mix(mix(velocityCell(b),velocityCell(b+vec3i(1,0,0)),f.x),mix(velocityCell(b+vec3i(0,1,0)),velocityCell(b+vec3i(1,1,0)),f.x),f.y);
  let c=mix(mix(velocityCell(b+vec3i(0,0,1)),velocityCell(b+vec3i(1,0,1)),f.x),mix(velocityCell(b+vec3i(0,1,1)),velocityCell(b+vec3i(1,1,1)),f.x),f.y);return mix(a,c,f.z);
}
// Regular-row components are the positive-face degrees of freedom consumed by
// divergenceAt/project. Reconstruct them on their staggered lattices. The two
// tall endpoint rows remain collocated Eq. 5 samples and bypass this path.
fn sampleVelocityComponent(p:vec3f,component:u32)->f32{
  var offset=vec3f(0.5);offset[component]=1.0;var lower=vec3f(0.0);lower[component]=-1.0;
  let q=clamp(p-offset,lower,vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);
  let a=mix(mix(velocityCell(b)[component],velocityCell(b+vec3i(1,0,0))[component],f.x),mix(velocityCell(b+vec3i(0,1,0))[component],velocityCell(b+vec3i(1,1,0))[component],f.x),f.y);
  let c=mix(mix(velocityCell(b+vec3i(0,0,1))[component],velocityCell(b+vec3i(1,0,1))[component],f.x),mix(velocityCell(b+vec3i(0,1,1))[component],velocityCell(b+vec3i(1,1,1))[component],f.x),f.y);return mix(a,c,f.z);
}
fn faceSampledVelocity(p:vec3f)->vec3f{return vec3f(sampleVelocityComponent(p,0u),sampleVelocityComponent(p,1u),sampleVelocityComponent(p,2u));}
fn samplePressure(p:vec3f)->f32{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);
  let a=mix(mix(pressureCell(b),pressureCell(b+vec3i(1,0,0)),f.x),mix(pressureCell(b+vec3i(0,1,0)),pressureCell(b+vec3i(1,1,0)),f.x),f.y);
  let c=mix(mix(pressureCell(b+vec3i(0,0,1)),pressureCell(b+vec3i(1,0,1)),f.x),mix(pressureCell(b+vec3i(0,1,1)),pressureCell(b+vec3i(1,1,1)),f.x),f.y);return mix(a,c,f.z);
}

fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}
	fn insideRigid(body:RigidBody,world:vec3f)->bool{
	  let offset=world-body.positionShape.xyz;let radius=max(body.dimensions.w,0.0);let radiusSquared=radius*radius;let distanceSquared=dot(offset,offset);
	  // Every uploaded primitive carries its exact orientation-independent
	  // bounding radius in dimensions.w. Most grid samples are far from the
	  // body, so reject them before quaternion rotation and shape evaluation.
	  if(distanceSquared>radiusSquared){return false;}let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));if(shape==0){return distanceSquared<=d.x*d.x;}
	  let p=quaternionInverseRotate(body.orientation,offset);if(shape==1){return all(abs(p)<=0.5*d);}if(shape==2){let cy=clamp(p.y,-0.5*d.y,0.5*d.y);return length(vec3f(p.x,p.y-cy,p.z))<=d.x;}return p.x*p.x+p.z*p.z<=d.x*d.x&&abs(p.y)<=0.5*d.y;
	}
	fn cellMayTouchRigid(world:vec3f,h:vec3f,bodyCount:u32)->bool{
	  // All eight occupancy probes lie on a sphere of this radius around the
	  // packed sample. If that sphere misses every primitive's bounding sphere,
	  // every exact corner test below is guaranteed to be false.
	  let probeRadius=0.4*length(h);for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let body=rigidBodies[bodyIndex];let radius=max(body.dimensions.w,0.0)+probeRadius;let offset=world-body.positionShape.xyz;if(dot(offset,offset)<=radius*radius){return true;}}return false;
	}
	fn rigidVelocityAt(world:vec3f)->vec4f{
	  let bodyCount=u32(round(params.boundary.z));for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let body=rigidBodies[bodyIndex];if(insideRigid(body,world)){let arm=world-body.positionShape.xyz;return vec4f(body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm),f32(bodyIndex+1u));}}
	  return vec4f(0.0);
	}
	fn solidVelocityCell(q:vec3i)->vec3f{return rigidVelocityAt(worldFromPoint(vec3f(vec3i(q))+vec3f(0.5))).xyz;}
	// Interior projected samples already move with the body and therefore are
	// not an undisturbed velocity for form drag. Sample wet, open points beyond
	// the body's bounding sphere for the exchange snapshot.
	fn ambientFluidVelocity(body:RigidBody,q:vec3i,fallback:vec3f)->vec3f{
	  let h=params.cellGravity.xyz;let radius=max(body.dimensions.w,0.0);let reach=vec3i(ceil(vec3f(2.0*radius)/h))+vec3i(2);
	  let offsets=array<vec3i,6>(vec3i(-reach.x,0,0),vec3i(reach.x,0,0),vec3i(0,-reach.y,0),vec3i(0,reach.y,0),vec3i(0,0,-reach.z),vec3i(0,0,reach.z));var total=vec3f(0.0);var weight=0.0;
	  for(var n=0;n<6;n+=1){let nq=q+offsets[n];if(!validWorld(nq)){continue;}let sampleWorld=worldFromPoint(vec3f(nq)+vec3f(0.5));if(rigidVelocityAt(sampleWorld).w!=0.0){continue;}let wet=occupancyFromPhi(phiCell(nq));total+=wet*velocityCell(nq);weight+=wet;}
	  return select(fallback,total/max(weight,1e-6),weight>0.0);
	}

fn axisOffset(axis:u32)->vec3i{return select(select(vec3i(0,0,1),vec3i(0,1,0),axis==1u),vec3i(1,0,0),axis==0u);}
// Equation 18 pressure-gradient correction evaluated at a virtual cubic
// sample. This diagnoses the paper's Section 5 middle-face limitation without
// introducing additional pressure unknowns.
fn pressureGradientAt(q:vec3i,axis:u32)->f32{
  let plus=q+axisOffset(axis);if(!validWorld(q)||!validWorld(plus)){return 0.0;}if(axis==1u&&!representedWorld(plus)){return 0.0;}let ownPhi=phiCell(q);let plusPhi=phiCell(plus);let ownLiquid=ownPhi<=0.0;let plusLiquid=plusPhi<=0.0;if(!ownLiquid&&!plusLiquid){return 0.0;}var ownPressure=select(0.0,pressureCell(q),ownLiquid);var plusPressure=select(0.0,pressureCell(plus),plusLiquid);var theta=1.0;if(ownLiquid!=plusLiquid){let liquidPhi=select(plusPhi,ownPhi,ownLiquid);let airPhi=select(ownPhi,plusPhi,ownLiquid);theta=clamp(abs(liquidPhi)/max(abs(liquidPhi)+abs(airPhi),1e-6),0.05,1.0);}if(solidFractionCell(plus)>0.9){plusPressure=ownPressure;}return (plusPressure-ownPressure)/(params.cellGravity.xyz[axis]*theta);
}
// Lateral gradient between neighbouring tall stores' bottom texels, with the
// same ghost-fluid handling as pressureGradientAt. The store is one lateral
// degree of freedom, so both the Jacobi assembly and this application couple
// bottom texel to bottom texel — exactly zero on hydrostatic fields.
fn storeLateralGradient(x:i32,z:i32,offset:vec2i,ownPhi:f32,ownPressureIn:f32)->f32{
  let nx=x+offset.x;let nz=z+offset.y;let d=packedDims();if(nx<0||nx>=d.x||nz<0||nz>=d.z){return 0.0;}
  let plusPhi=textureLoad(volumeIn,vec3i(nx,0,nz),0).x;let ownLiquid=ownPhi<=0.0;let plusLiquid=plusPhi<=0.0;if(!ownLiquid&&!plusLiquid){return 0.0;}
  var ownPressure=select(0.0,ownPressureIn,ownLiquid);var plusPressure=select(0.0,textureLoad(pressureIn,vec3i(nx,0,nz),0).x,plusLiquid);
  var theta=1.0;if(ownLiquid!=plusLiquid){let liquidPhi=select(plusPhi,ownPhi,ownLiquid);let airPhi=select(ownPhi,plusPhi,ownLiquid);theta=clamp(abs(liquidPhi)/max(abs(liquidPhi)+abs(airPhi),1e-6),0.05,1.0);}
  if(textureLoad(solidFractionIn,vec3i(nx,0,nz),0).x>0.9){plusPressure=ownPressure;}
  return (plusPressure-ownPressure)/theta;
}
fn phiGradient(q:vec3i)->vec3f{let h=params.cellGravity.xyz;return vec3f(phiCell(q+vec3i(1,0,0))-phiCell(q-vec3i(1,0,0)),phiCell(q+vec3i(0,1,0))-phiCell(q-vec3i(0,1,0)),phiCell(q+vec3i(0,0,1))-phiCell(q-vec3i(0,0,1)))/(2.0*h);}
fn volumeGradient(q:vec3i)->vec3f{let band=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));return select(vec3f(0.0),-phiGradient(q)/band,abs(phiCell(q))<band);}
fn interfaceNormal(q:vec3i)->vec3f{let g=phiGradient(q);return g/max(length(g),1e-6);}
fn curvature(q:vec3i)->f32{let h=params.cellGravity.xyz;return (interfaceNormal(q+vec3i(1,0,0)).x-interfaceNormal(q-vec3i(1,0,0)).x)/(2.0*h.x)+(interfaceNormal(q+vec3i(0,1,0)).y-interfaceNormal(q-vec3i(0,1,0)).y)/(2.0*h.y)+(interfaceNormal(q+vec3i(0,0,1)).z-interfaceNormal(q-vec3i(0,0,1)).z)/(2.0*h.z);}
fn velocityLaplacian(q:vec3i)->vec3f{let h=params.cellGravity.xyz;let c=diffusionVelocity(q);return (diffusionVelocity(q+vec3i(1,0,0))-2.0*c+diffusionVelocity(q-vec3i(1,0,0)))/(h.x*h.x)+(diffusionVelocity(q+vec3i(0,1,0))-2.0*c+diffusionVelocity(q-vec3i(0,1,0)))/(h.y*h.y)+(diffusionVelocity(q+vec3i(0,0,1))-2.0*c+diffusionVelocity(q-vec3i(0,0,1)))/(h.z*h.z);}
fn strainMagnitude(q:vec3i)->f32{let h=params.cellGravity.xyz;let dx=(diffusionVelocity(q+vec3i(1,0,0))-diffusionVelocity(q-vec3i(1,0,0)))/(2.0*h.x);let dy=(diffusionVelocity(q+vec3i(0,1,0))-diffusionVelocity(q-vec3i(0,1,0)))/(2.0*h.y);let dz=(diffusionVelocity(q+vec3i(0,0,1))-diffusionVelocity(q-vec3i(0,0,1)))/(2.0*h.z);let sxy=0.5*(dx.y+dy.x);let sxz=0.5*(dx.z+dz.x);let syz=0.5*(dy.z+dz.y);return sqrt(2.0*(dx.x*dx.x+dy.y*dy.y+dz.z*dz.z+2.0*(sxy*sxy+sxz*sxz+syz*syz)));}

// Phi follows the paper's collocated Semi-Lagrangian trajectory. Velocity
// transport uses a separate component-staggered trace below; sharing that
// trace would silently change the level-set method while fixing MAC advection.
fn traceDeparture(p:vec3f,signedDt:f32)->vec3f{let h=params.cellGravity.xyz;let first=sampleVelocity(p);let midpoint=p-0.5*first*signedDt/h;return p-sampleVelocity(midpoint)*signedDt/h;}
fn traceFaceDeparture(p:vec3f,signedDt:f32)->vec3f{let h=params.cellGravity.xyz;let first=faceSampledVelocity(p);let midpoint=p-0.5*first*signedDt/h;return p-faceSampledVelocity(midpoint)*signedDt/h;}
fn tracedVelocity(p:vec3f,signedDt:f32)->vec3f{return sampleVelocity(traceDeparture(p,signedDt));}
fn faceAdvectedVelocity(id:vec3i,signedDt:f32)->vec3f{
  let center=samplePoint(id);var result=vec3f(0.0);
  // The caller restricts this to regular rows. Adding the component half-
  // offset lands each value on the face used by divergenceAt and project.
  for(var component=0u;component<3u;component+=1u){let faceP=center+0.5*vec3f(axisOffset(component));result[component]=sampleVelocityComponent(traceFaceDeparture(faceP,signedDt),component);}
  return result;
}
// Global level-set volume control. params.physical.w is a normal correction
// speed in cells/second: it is negative when reconstructed volume is high, so
// subtracting it increases phi and shrinks the liquid region. Restrict the
// offset to the interface band and retain the paper's ordinary SL advection.
fn volumeCorrectedPhi(value:f32,dt:f32)->f32{let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));return select(value,value-params.physical.w*h*dt,abs(value)<1.5*h);}
fn adjacentToInterface(q:vec3i,current:f32)->bool{let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));for(var index=0;index<6;index+=1){let other=phiCell(q+offsets[index]);if((current<=0.0)!=(other<=0.0)){return true;}}return false;}
@compute @workgroup_size(1)
fn planSubsteps(@builtin(global_invocation_id) gid:vec3u){
  if(any(gid!=vec3u(0))){return;}let previousCfl=bitcast<f32>(atomicLoad(&governor[0]));let substeps=clamp(u32(ceil(previousCfl/2.0)),1u,8u);let dtPhi=params.dimsDt.w/f32(substeps);
  atomicStore(&governor[0],0u);atomicStore(&governor[1],substeps);atomicStore(&governor[2],bitcast<u32>(dtPhi));atomicAdd(&governor[3],substeps);
  let groups=vec3u((u32(round(params.dimsDt.x))+3u)/4u,(u32(round(params.dimsDt.y))+3u)/4u,(u32(round(params.dimsDt.z))+3u)/4u);
  for(var slot=0u;slot<8u;slot+=1u){let base=slot*4u;let enabled=slot<substeps;atomicStore(&phiDispatchArgs[base],select(0u,groups.x,enabled));atomicStore(&phiDispatchArgs[base+1u],select(0u,groups.y,enabled));atomicStore(&phiDispatchArgs[base+2u],select(0u,groups.z,enabled));atomicStore(&phiDispatchArgs[base+3u],0u);}
}
@compute @workgroup_size(4,4,4)
fn transportPhi(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);let h=params.cellGravity.xyz;let limit=5.0*min(h.x,min(h.y,h.z));if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(volumeOut,id,vec4f(limit));return;}
  let dt=bitcast<f32>(atomicLoad(&governor[2]));let p=samplePoint(id);let q=vec3i(floor(p));var phi=volumeCorrectedPhi(samplePhi(traceDeparture(p,dt)),dt);if(isInflowVelocityCell(q)){phi=min(phi,-0.5*min(h.x,min(h.y,h.z))*inflowApertureFraction(q)*inflowStrength());}textureStore(volumeOut,id,vec4f(phi));
}
@compute @workgroup_size(4,4,4)
fn clampPhi(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);let limit=5.0*min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(volumeOut,id,vec4f(limit));return;}let current=textureLoad(volumeIn,id,0).x;let q=vec3i(floor(samplePoint(id)));let local=select(current,clamp(current,-min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z)),min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z))),adjacentToInterface(q,current));textureStore(volumeOut,id,vec4f(clamp(local,-limit,limit)));}
@compute @workgroup_size(4,4,4)
fn reinitializePhi(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);let h=params.cellGravity.xyz;let cell=min(h.x,min(h.y,h.z));let limit=5.0*cell;if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(volumeOut,id,vec4f(limit));return;}let current=textureLoad(volumeIn,id,0).x;let q=vec3i(floor(samplePoint(id)));if(abs(current)>3.0*cell||adjacentToInterface(q,current)){textureStore(volumeOut,id,vec4f(clamp(current,-limit,limit)));return;}var distance=min(abs(phiCell(q-vec3i(1,0,0)))+h.x,abs(phiCell(q+vec3i(1,0,0)))+h.x);distance=min(distance,min(abs(phiCell(q-vec3i(0,1,0)))+h.y,abs(phiCell(q+vec3i(0,1,0)))+h.y));distance=min(distance,min(abs(phiCell(q-vec3i(0,0,1)))+h.z,abs(phiCell(q+vec3i(0,0,1)))+h.z));let candidate=select(min(distance,limit),-min(distance,limit),current<0.0);textureStore(volumeOut,id,vec4f(clamp(candidate,current-cell,current+cell)));}
@compute @workgroup_size(4,4,4)
fn extrapolateVelocity(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));return;}let q=vec3i(floor(samplePoint(id)));if(isInflowVelocityCell(q)){textureStore(velocityOut,id,vec4f(applyInflowVelocity(q,textureLoad(velocityIn,id,0).xyz),1.0));return;}if(pointSamplePhi(id)<=0.0){textureStore(velocityOut,id,vec4f(textureLoad(velocityIn,id,0).xyz,1.0));return;}let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));var sum=vec3f(0.0);var weight=0.0;for(var index=0;index<6;index+=1){let state=velocityStateCell(q+offsets[index]);if(state.w>0.5){sum+=state.xyz;weight+=1.0;}}if(weight>0.0){textureStore(velocityOut,id,vec4f(sum/weight,1.0));}else{textureStore(velocityOut,id,vec4f(textureLoad(velocityIn,id,0).xyz,0.0));}}
@compute @workgroup_size(4,4,4)
fn predictVelocity(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));return;}let value=select(tracedVelocity(samplePoint(id),params.dimsDt.w),faceAdvectedVelocity(id,params.dimsDt.w),id.y>=2);textureStore(velocityOut,id,vec4f(value,0.0));}
@compute @workgroup_size(4,4,4)
fn reverseVelocity(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));return;}let value=select(tracedVelocity(samplePoint(id),-params.dimsDt.w),faceAdvectedVelocity(id,-params.dimsDt.w),id.y>=2);textureStore(velocityOut,id,vec4f(value,0.0));}

fn boundedMacCormack(id:vec3i,p:vec3f)->vec3f{
  let predicted=textureLoad(predictedVelocityIn,id,0).xyz;let original=textureLoad(velocityIn,id,0).xyz;let reversed=textureLoad(reversedVelocityIn,id,0).xyz;var corrected=predicted+0.5*(original-reversed);
  if(id.y<2){let departure=traceDeparture(p,params.dimsDt.w);let sampleCoordinate=clamp(departure-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(sampleCoordinate));var lower=velocityCell(b);var upper=lower;for(var corner:u32=1u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let value=velocityCell(b+offset);lower=min(lower,value);upper=max(upper,value);}for(var component:u32=0u;component<3u;component+=1u){if(corrected[component]<lower[component]||corrected[component]>upper[component]){corrected[component]=predicted[component];}}return corrected;}
  for(var component:u32=0u;component<3u;component+=1u){
    let faceP=p+0.5*vec3f(axisOffset(component));let departure=traceFaceDeparture(faceP,params.dimsDt.w);var faceOffset=vec3f(0.5);faceOffset[component]=1.0;var lowerCoordinate=vec3f(0.0);lowerCoordinate[component]=-1.0;
    let sampleCoordinate=clamp(departure-faceOffset,lowerCoordinate,vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(sampleCoordinate));var lower=velocityCell(b)[component];var upper=lower;
    for(var corner:u32=1u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let value=velocityCell(b+offset)[component];lower=min(lower,value);upper=max(upper,value);}
    if(corrected[component]<lower||corrected[component]>upper){corrected[component]=predicted[component];}
  }
  return corrected;
}

@compute @workgroup_size(4,4,4)
fn finishAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));return;}
	  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let p=samplePoint(id);let q=vec3i(floor(p));var v=boundedMacCormack(id,p);let phi=pointSamplePhi(id);
  // Euler's momentum equation is integrated in the liquid domain. Air values
  // are extrapolation support for the next trace and must not accumulate a
  // separate gravity impulse; doing so feeds a falling-air mode back through
  // the collocated interface stencil. Endpoint samples gate on the settled
  // point occupancy (paper Eq 4), never on the tall-cell column average.
  // MAC gravity gate: the stored v.y is the face between q and q+y, so it
  // integrates gravity when EITHER adjacent cell is liquid (matching the
  // uniform path). Gating on the own sample alone starved the face beneath
  // every droplet, which levitated small drops and left splash water welded
  // to the ceiling where the wall zeroes the only gravity-fed face.
  let fluidOpen=1.0-clamp(solidFractionCell(q),0.0,1.0);if(phi<=0.0||phiCell(q+vec3i(0,1,0))<=0.0){v.y+=fluidOpen*params.cellGravity.w*dt;}if(phi<=min(h.x,min(h.y,h.z))){let nu=params.physical.y/params.physical.x;v+=fluidOpen*dt*nu*velocityLaplacian(q);if(abs(phi)<min(h.x,min(h.y,h.z))){v+=fluidOpen*dt*params.boundary.x/params.physical.x*curvature(q)*volumeGradient(q);}}
  // Safety rail against the flux-form transport's CFL limit (Appendix C
  // gap 9): physical speeds in these scenes stay below CFL ~1.5 per frame
  // step while a developing blow-up passes CFL 4 within a few steps, so an
  // ABSOLUTE component clamp at 4 cells per frame dt (container.w, wired in
  // advanceTo so CFL substepping cannot inflate it) breaks the escalation
  // without touching resolvable flow. Prescribed inflow is applied after
  // the clamp.
  let speedCap=vec3f(params.container.w);v=clamp(v,-speedCap,speedCap);
  v=applyInflowVelocity(q,v);
  // The band ceiling is a closed face until remeshing lifts the band; zero
  // its normal here so the pressure RHS never sees phantom outflow through
  // a face no volume can cross.
  let d=fineDims();if(q.x+1>=d.x){v.x=0.0;}if(q.y+1>=d.y||!representedWorld(q+vec3i(0,1,0))){v.y=0.0;}if(q.z+1>=d.z){v.z=0.0;}
  textureStore(velocityOut,id,vec4f(v,0.0));
}

@compute @workgroup_size(4,4,4)
fn finishSemiLagrangianAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));return;}
	  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let p=samplePoint(id);let q=vec3i(floor(p));var v=textureLoad(predictedVelocityIn,id,0).xyz;let phi=pointSamplePhi(id);
  // Match finishAdvection exactly after choosing the transport estimate so
  // the A/B option isolates MacCormack correction from all other physics.
  // MAC gravity gate: the stored v.y is the face between q and q+y, so it
  // integrates gravity when EITHER adjacent cell is liquid (matching the
  // uniform path). Gating on the own sample alone starved the face beneath
  // every droplet, which levitated small drops and left splash water welded
  // to the ceiling where the wall zeroes the only gravity-fed face.
  let fluidOpen=1.0-clamp(solidFractionCell(q),0.0,1.0);if(phi<=0.0||phiCell(q+vec3i(0,1,0))<=0.0){v.y+=fluidOpen*params.cellGravity.w*dt;}if(phi<=min(h.x,min(h.y,h.z))){let nu=params.physical.y/params.physical.x;v+=fluidOpen*dt*nu*velocityLaplacian(q);if(abs(phi)<min(h.x,min(h.y,h.z))){v+=fluidOpen*dt*params.boundary.x/params.physical.x*curvature(q)*volumeGradient(q);}}
  // Same absolute speed rail as finishAdvection (see the comment there).
  let speedCap=vec3f(params.container.w);v=clamp(v,-speedCap,speedCap);
  v=applyInflowVelocity(q,v);let d=fineDims();if(q.x+1>=d.x){v.x=0.0;}if(q.y+1>=d.y||!representedWorld(q+vec3i(0,1,0))){v.y=0.0;}if(q.z+1>=d.z){v.z=0.0;}
  textureStore(velocityOut,id,vec4f(v,0.0));
}

		// Each stored velocity component is the positive-face degree of freedom
		// corrected by project(). Keeping divergence on that same face convention
		// makes the compact pressure matrix the composition div(grad p). Eq. 5
		// interpolation remains the transport/reconstruction view inside a store;
		// averaging adjacent samples here mixes the two representations and lets a
		// converged pressure solve add divergence and kinetic energy.
		fn positiveFaceVelocity(q:vec3i,axis:u32)->f32{let offset=axisOffset(axis);let neighbor=q+offset;if(!validWorld(q)||!validWorld(neighbor)){return 0.0;}if(solidFractionCell(neighbor)>0.9){return solidVelocityCell(neighbor)[axis];}if(solidFractionCell(q)>0.9){return solidVelocityCell(q)[axis];}return velocityCell(q)[axis];}
	fn lateralDivergenceAt(q:vec3i)->f32{let h=params.cellGravity.xyz;return (positiveFaceVelocity(q,0u)-positiveFaceVelocity(q-vec3i(1,0,0),0u))/h.x+(positiveFaceVelocity(q,2u)-positiveFaceVelocity(q-vec3i(0,0,1),2u))/h.z;}
	fn pointDivergenceAt(q:vec3i)->f32{let h=params.cellGravity.xyz;return lateralDivergenceAt(q)+(positiveFaceVelocity(q,1u)-positiveFaceVelocity(q-vec3i(0,1,0),1u))/h.y;}
	// Paper Eq 13/19: divergence is measured as a POINT divergence at the top
	// and bottom sub-cells of each tall cell (and every regular cell). At the
	// bottom sub-cell the face below is the floor/terrain (u_solid = 0), so the
	// constraint pins the bottom dof to the wall; with the Eq 5 linear velocity
	// profile the divergence varies linearly inside the store, so zeroing both
	// endpoints zeroes every interior row. The earlier control-volume average
	// let a uniform free-fall cancel against uniform lateral spreading, which
	// is exactly the dam-break dome failure (2026-07-16 collapse audit).
	fn divergenceAt(id:vec3i)->f32{
	  return pointDivergenceAt(vec3i(floor(samplePoint(id))));
	}
@compute @workgroup_size(4,4,4)
fn buildPressureRhs(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}let wet=activeSample(id)&&pointSamplePhi(id)<=0.0;let rhs=select(0.0,params.physical.x*divergenceAt(id)/params.dimsDt.w,wet);textureStore(pressureOut,id,vec4f(rhs,0.0,0.0,0.0));}
fn interfaceFraction(a:f32,b:f32)->f32{return clamp(abs(a)/max(abs(a)+abs(b),1e-6),0.05,1.0);}
	fn pressureTerm(ownPhi:f32,otherPhi:f32,otherPressure:f32,solidFraction:f32,coefficient:f32)->vec2f{if(otherPhi<=0.0){let open=1.0-clamp(solidFraction,0.0,1.0);return vec2f(coefficient*open,coefficient*open*otherPressure);}return vec2f(coefficient/interfaceFraction(ownPhi,otherPhi),0.0);}

@compute @workgroup_size(4,4,4)
fn jacobi(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(pressureOut,id,vec4f(0.0));return;}let ownAlpha=pointSamplePhi(id);if(ownAlpha>0.0){textureStore(pressureOut,id,vec4f(0.0));return;}
  let p=samplePoint(id);let h=params.cellGravity.xyz;let d=fineDims();var stencil=vec2f(0.0);let base=baseAt(id.x,id.z);
	  let q=vec3i(floor(p));
	  if(id.y==0&&base>0){
	    // Paper Eq 15/16 row at the bottom sub-cell. Laterally the bottom dof
	    // couples to the neighbouring stores' bottom texels (both sit at world
	    // y = 0 on a flat floor; hydrostatically exact). Below is solid, so the
	    // ghost value equals the own pressure and drops out (Neumann). Above,
	    // the Eq 5 interpolated pressure at y = 1 is (1-s) p_b + s p_t with
	    // s = 1/(base-1), so the top endpoint couples at s/h^2 = 1/(distance*h)
	    // and the (1-s) share folds into the diagonal — identical to the
	    // staggered adjoint of the endpoint point divergence.
	    if(id.x>0){stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,vec3i(id.x-1,0,id.z),0).x,textureLoad(pressureIn,vec3i(id.x-1,0,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x-1,0,id.z),0).x,1.0/(h.x*h.x));}
	    if(id.x+1<d.x){stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,vec3i(id.x+1,0,id.z),0).x,textureLoad(pressureIn,vec3i(id.x+1,0,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x+1,0,id.z),0).x,1.0/(h.x*h.x));}
	    if(id.z>0){stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,vec3i(id.x,0,id.z-1),0).x,textureLoad(pressureIn,vec3i(id.x,0,id.z-1),0).x,textureLoad(solidFractionIn,vec3i(id.x,0,id.z-1),0).x,1.0/(h.z*h.z));}
	    if(id.z+1<d.z){stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,vec3i(id.x,0,id.z+1),0).x,textureLoad(pressureIn,vec3i(id.x,0,id.z+1),0).x,textureLoad(solidFractionIn,vec3i(id.x,0,id.z+1),0).x,1.0/(h.z*h.z));}
	    let distance=max(h.y,f32(base-1)*h.y);
	    stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,vec3i(id.x,1,id.z),0).x,textureLoad(pressureIn,vec3i(id.x,1,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x,1,id.z),0).x,1.0/(distance*h.y));
	  } else {
	  if(q.x>0){let n=q-vec3i(1,0,0);stencil+=pressureTerm(ownAlpha,phiCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.x*h.x));}if(q.x+1<d.x){let n=q+vec3i(1,0,0);stencil+=pressureTerm(ownAlpha,phiCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.x*h.x));}if(q.z>0){let n=q-vec3i(0,0,1);stencil+=pressureTerm(ownAlpha,phiCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.z*h.z));}if(q.z+1<d.z){let n=q+vec3i(0,0,1);stencil+=pressureTerm(ownAlpha,phiCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.z*h.z));}
	  }
	  if(id.y==0){/* lateral and vertical terms assembled in the bottom sub-cell block above */}
		  // Top endpoint row of the composed div(grad p) operator for the Eq. 5
		  // linear endpoint basis. With s=1/(base-1), the pressure correction of
		  // the reconstructed downward face carries another factor s: bottom is
		  // s^2/h^2 and the first band sample is s/h^2. Using the paper's stronger
		  // 1/(distance*h), 1/h^2 row solves a different operator and pumps energy.
		  else if(id.y==1){let distance=max(h.y,f32(base-1)*h.y);let composed=base<=regularLayers();let bottomCoefficient=select(1.0/(distance*h.y),1.0/(distance*distance),composed);let bandCoefficient=select(1.0/(h.y*h.y),1.0/(distance*h.y),composed);stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,vec3i(id.x,0,id.z),0).x,textureLoad(pressureIn,vec3i(id.x,0,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x,0,id.z),0).x,bottomCoefficient);if(activeSample(vec3i(id.x,2,id.z))){stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,vec3i(id.x,2,id.z),0).x,textureLoad(pressureIn,vec3i(id.x,2,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x,2,id.z),0).x,bandCoefficient);}}
	  else {if(id.y>2||base>=2){let below=id-vec3i(0,1,0);let belowPhi=select(textureLoad(volumeIn,below,0).x,textureLoad(volumeIn,vec3i(id.x,1,id.z),0).x,id.y==2&&base>0);stencil+=pressureTerm(ownAlpha,belowPhi,textureLoad(pressureIn,below,0).x,textureLoad(solidFractionIn,below,0).x,1.0/(h.y*h.y));}if(activeSample(id+vec3i(0,1,0))){let above=id+vec3i(0,1,0);stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,above,0).x,textureLoad(pressureIn,above,0).x,textureLoad(solidFractionIn,above,0).x,1.0/(h.y*h.y));}else if(q.y+1>=d.y){stencil+=pressureTerm(ownAlpha,h.y,0.0,1.0,1.0/(h.y*h.y));}}
  let rhs=params.physical.x*divergenceAt(id)/params.dimsDt.w;let old=textureLoad(pressureIn,id,0).x;let next=(stencil.y-rhs)/max(stencil.x,1e-9);textureStore(pressureOut,id,vec4f(mix(old,next,0.8),0.0,0.0,0.0));
}

@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid:vec3u){
		  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(0.0));return;}let q=vec3i(floor(samplePoint(id)));let h=params.cellGravity.xyz;let scale=params.dimsDt.w/params.physical.x;var v=textureLoad(velocityIn,id,0).xyz;let offsets=array<vec3i,3>(vec3i(1,0,0),vec3i(0,1,0),vec3i(0,0,1));
	  let base=baseAt(id.x,id.z);
	  if(id.y==0&&base>0){
	    // The store is one lateral degree of freedom: its faces respond to the
	    // neighbouring stores' bottom texels (matching the Jacobi assembly and
	    // exactly zero on hydrostatic fields); the vertical gradient is the
	    // linear pressure slope, identical at every interior row.
	    let ownAlpha=textureLoad(volumeIn,vec3i(id.x,0,id.z),0).x;let ownPressure=textureLoad(pressureIn,vec3i(id.x,0,id.z),0).x;
	    if(!validWorld(q+offsets[0])){v.x=0.0;}else{v.x-=scale*storeLateralGradient(id.x,id.z,vec2i(1,0),ownAlpha,ownPressure)/h.x;}
	    if(!validWorld(q+offsets[2])){v.z=0.0;}else{v.z-=scale*storeLateralGradient(id.x,id.z,vec2i(0,1),ownAlpha,ownPressure)/h.z;}
	    if(!validWorld(q+offsets[1])||!representedWorld(q+offsets[1])){v.y=0.0;}else{v.y-=scale*pressureGradientAt(q,1u);}
	  } else {
	  for(var axis=0u;axis<3u;axis+=1u){let plus=q+offsets[axis];if(!validWorld(plus)){v[axis]=0.0;continue;}if(axis==1u&&!representedWorld(plus)){v[axis]=0.0;continue;}v[axis]-=scale*pressureGradientAt(q,axis);
	    // divergenceAt substitutes the rigid velocity on every solid-covered
	    // positive face. Apply the identical constraint after projection too.
	    // Otherwise pressure writes an arbitrary velocity inside a stationary
	    // body; coupleRigid reads it on the next frame as real fluid momentum,
	    // producing a persistent bias toward the negative-X/negative-Z corner.
	    if(solidFractionCell(plus)>0.9){v[axis]=solidVelocityCell(plus)[axis];}else if(solidFractionCell(q)>0.9){v[axis]=solidVelocityCell(q)[axis];}
	  }
	  }
	  let speedCap=vec3f(params.container.w);v=clamp(v,-speedCap,speedCap);
	  v=applyInflowVelocity(q,v);textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));
	}


	@compute @workgroup_size(4,4,4)
	fn coupleRigid(@builtin(global_invocation_id) gid:vec3u){
	  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(0.0));textureStore(pressureOut,id,vec4f(0.0));return;}
	  let cellPoint=samplePoint(id);let q=vec3i(floor(cellPoint));let cellWorld=worldFromPoint(cellPoint);let phi=textureLoad(volumeIn,id,0).x;let alpha=occupancyFromPhi(phi);let oldV=textureLoad(velocityIn,id,0).xyz;var v=oldV;let h=params.cellGravity.xyz;let bodyCount=u32(round(params.boundary.z));var solid=0.0;var coupledBody=12u;
	  if(bodyCount==0u||(params.physical.z<=0.5&&!cellMayTouchRigid(cellWorld,h,bodyCount))){textureStore(velocityOut,id,vec4f(oldV,0.0));textureStore(volumeOut,id,vec4f(phi));textureStore(pressureOut,id,vec4f(0.0));return;}
	  // The legacy full-height diagnostic mode evaluates every virtual cubic cell inside a
	  // tall cell. Its contribution is mapped to the two endpoint unknowns with
	  // the paper's (1-s) / s weights; exchange is accumulated once by id.y=0.
	  if(params.physical.z>0.5&&id.y<2){
	    let height=baseAt(id.x,id.z);var weightedDelta=vec3f(0.0);var coupledWeight=0.0;var basisWeight=0.0;var solidBasis=0.0;
	    for(var y=0;y<height;y+=1){
	      let q=vec3i(id.x,y,id.z);let world=worldFromPoint(vec3f(f32(id.x)+0.5,f32(y)+0.5,f32(id.z)+0.5));var virtualSolid=0.0;var owner=12u;
	      for(var corner:u32=0u;corner<8u;corner+=1u){let offset=vec3f(select(-0.4,0.4,(corner&1u)>0u),select(-0.4,0.4,(corner&2u)>0u),select(-0.4,0.4,(corner&4u)>0u))*h;let sample=world+offset;for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],sample)){virtualSolid+=0.125;owner=min(owner,bodyIndex);break;}}}
	      var s=0.5;if(height>1){s=1.0-f32(y)/f32(height-1);}let endpointWeight=select(s,1.0-s,id.y==1);basisWeight+=endpointWeight;solidBasis+=endpointWeight*virtualSolid;
	      if(owner<12u&&virtualSolid>0.0){let body=rigidBodies[owner];let arm=world-body.positionShape.xyz;let solidVelocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);let virtualVelocity=velocityCell(q);let weight=endpointWeight*virtualSolid;weightedDelta+=weight*(solidVelocity-virtualVelocity);coupledWeight+=weight;
	        if(id.y==0){let ambientVelocity=ambientFluidVelocity(body,q,virtualVelocity);let solidDensity=max(body.angularVelocity.w,1.0);let reducedDensity=params.physical.x*solidDensity/(params.physical.x+solidDensity);let fluidImpulse=reducedDensity*h.x*h.y*h.z*alpha*virtualSolid*(solidVelocity-virtualVelocity);let reaction=-fluidImpulse*select(0.0,1.0,virtualSolid>0.9);let torque=cross(arm,reaction);let exchangeBase=owner*12u;atomicAdd(&rigidExchange[exchangeBase],i32(round(reaction.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+1u],i32(round(reaction.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+2u],i32(round(reaction.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+3u],i32(round(torque.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+4u],i32(round(torque.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+5u],i32(round(torque.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+6u],i32(round(alpha*virtualSolid*65536.0)));atomicAdd(&rigidExchange[exchangeBase+7u],i32(round(alpha*virtualSolid*ambientVelocity.x*1e4)));atomicAdd(&rigidExchange[exchangeBase+8u],i32(round(alpha*virtualSolid*ambientVelocity.y*1e4)));atomicAdd(&rigidExchange[exchangeBase+9u],i32(round(alpha*virtualSolid*ambientVelocity.z*1e4)));}
	      }
	    }
	    if(coupledWeight>0.0){v+=weightedDelta/coupledWeight;}solid=solidBasis/max(basisWeight,1e-6);
	  }else{
	    let world=cellWorld;var owner=12u;
	    for(var corner:u32=0u;corner<8u;corner+=1u){let offset=vec3f(select(-0.4,0.4,(corner&1u)>0u),select(-0.4,0.4,(corner&2u)>0u),select(-0.4,0.4,(corner&4u)>0u))*h;let sample=world+offset;for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],sample)){solid+=0.125;owner=min(owner,bodyIndex);break;}}}
	    coupledBody=owner;
	    // The full fluid velocity blend is paired with a reduced-density reaction
	    // only in body-interior cells. A cut cell's collocated velocity is not a
	    // surface momentum sample and changes discontinuously with grid phase;
	    // explicit CPU-side drag supplies the physical light-body resistance.
	    if(owner<12u&&solid>0.0){let body=rigidBodies[owner];let arm=world-body.positionShape.xyz;let solidVelocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);let ambientVelocity=ambientFluidVelocity(body,q,oldV);v=mix(v,solidVelocity,solid);let solidDensity=max(body.angularVelocity.w,1.0);let reducedDensity=params.physical.x*solidDensity/(params.physical.x+solidDensity);let fluidImpulse=reducedDensity*h.x*h.y*h.z*alpha*solid*(solidVelocity-oldV);let reaction=-fluidImpulse*select(0.0,1.0,solid>0.9);let torque=cross(arm,reaction);let exchangeBase=owner*12u;atomicAdd(&rigidExchange[exchangeBase],i32(round(reaction.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+1u],i32(round(reaction.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+2u],i32(round(reaction.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+3u],i32(round(torque.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+4u],i32(round(torque.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+5u],i32(round(torque.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+6u],i32(round(alpha*solid*65536.0)));atomicAdd(&rigidExchange[exchangeBase+7u],i32(round(alpha*solid*ambientVelocity.x*1e4)));atomicAdd(&rigidExchange[exchangeBase+8u],i32(round(alpha*solid*ambientVelocity.y*1e4)));atomicAdd(&rigidExchange[exchangeBase+9u],i32(round(alpha*solid*ambientVelocity.z*1e4)));}
	  }
	  // The prescribed reservoir occupies the nozzle's open channel. The
	  // display cylinder is a filled rigid primitive, so carve its inlet cells
	  // back out of the pressure mask and let the boundary velocity win there.
	  if(isInflowVelocityCell(q)){solid=0.0;v=applyInflowVelocity(q,v);}
	  // Paper Sec 3.9.1 phi-s: the advected level set is meaningless inside a
	  // rigid body. Pull it toward open neighbours so a rising body displaces
	  // its water column instead of carrying a phantom wet plug and buoyancy.
	  var phiNext=phi;
	  if(solid>0.0){
	    let limit=5.0*min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));var open=0.0;var openSum=0.0;var total=0.0;var count=0.0;
	    let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
	    for(var n=0;n<6;n+=1){let nq=q+offsets[n];if(!validWorld(nq)){continue;}let neighborPhi=phiCell(nq);total+=neighborPhi;count+=1.0;let neighborWorld=worldFromPoint(vec3f(nq)+vec3f(0.5));if(rigidVelocityAt(neighborWorld).w==0.0){open+=1.0;openSum+=neighborPhi;}}
	    // Preserve the neighbour target but converge interior body cells directly
	    // from lateral open samples instead of diffusing over several radii.
	    if(coupledBody<12u&&i32(round(rigidBodies[coupledBody].positionShape.w))==0&&length(rigidBodies[coupledBody].linearVelocity.xyz)>0.25){let radius=max(rigidBodies[coupledBody].dimensions.w,0.0);let reach=vec3i(ceil(vec3f(2.0*radius)/h))+vec3i(2);let far=array<vec3i,4>(vec3i(-reach.x,0,0),vec3i(reach.x,0,0),vec3i(0,0,-reach.z),vec3i(0,0,reach.z));for(var n=0;n<4;n+=1){let nq=q+far[n];if(!validWorld(nq)){continue;}let neighborWorld=worldFromPoint(vec3f(nq)+vec3f(0.5));if(rigidVelocityAt(neighborWorld).w==0.0){open+=1.0;openSum+=phiCell(nq);}}}
	    let relaxTarget=select(total/max(count,1.0),openSum/max(open,1.0),open>0.0);phiNext=clamp(mix(phi,relaxTarget,solid),-limit,limit);
	  }
	  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(phiNext));textureStore(pressureOut,id,vec4f(solid));
	}

@compute @workgroup_size(4,4,4)
fn reduceBeforeProjection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)||!activeSample(id)){return;}let phi=textureLoad(volumeIn,id,0).x;let velocityValue=textureLoad(velocityIn,id,0).xyz;let speed=length(velocityValue);
  if(!finiteScalar(phi)||!all(velocityValue==velocityValue)||any(abs(velocityValue)>vec3f(3.402823e38))){atomicAdd(&reductions[20],1u);return;}
  if(pointSamplePhi(id)<=0.0&&solidFractionCell(vec3i(floor(samplePoint(id))))<=0.9){updatePositiveMaximum(abs(divergenceAt(id)),4u,11u,id);}else if(id.y>=2){updatePositiveMaximum(speed,6u,17u,id);}
}

@compute @workgroup_size(8,8,1)
fn planRemesh(@builtin(global_invocation_id) gid:vec3u){
  let d=packedDims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}let x=i32(gid.x);let z=i32(gid.y);let oldBase=baseAt(x,z);let layers=regularLayers();let fineY=fineDims().y;var lowest=fineY;var highest=-1;
  var previous=phiCell(vec3i(x,0,z));for(var y=1;y<fineY;y+=1){let current=phiCell(vec3i(x,y,z));if((previous<=0.0)!=(current<=0.0)){lowest=min(lowest,y);highest=max(highest,y);}previous=current;}
  let h=params.cellGravity.xyz;let worldX=-0.5*params.container.x+(f32(x)+0.5)*h.x;let worldZ=-0.5*params.container.z+(f32(z)+0.5)*h.z;let bodyCount=u32(round(params.boundary.z));let maxBase=max(0,fineY-layers);var bodyUpper=maxBase;let wetTop=columnHighestWetCell(x,z);
  for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let body=rigidBodies[bodyIndex];let radius=body.dimensions.w;if(abs(worldX-body.positionShape.x)<=radius&&abs(worldZ-body.positionShape.z)<=radius){let predictedBottom=i32(floor((body.positionShape.y-radius+min(0.0,body.linearVelocity.y)*params.dimsDt.w)/h.y));let nearWater=predictedBottom<=wetTop+1+i32(round(params.tall.z));if(predictedBottom<fineY&&nearWater){bodyUpper=min(bodyUpper,predictedBottom);}}}
  var desired=oldBase;if(highest>=0){let surfaceLower=highest+1+i32(round(params.tall.z))-layers;let surfaceUpper=lowest+1-i32(round(params.tall.y));desired=select(surfaceLower,clamp(oldBase,surfaceLower,surfaceUpper),surfaceLower<=surfaceUpper);}desired=clamp(min(desired,bodyUpper),0,maxBase);
  if(maxBase>=2){desired=max(2,desired);}nextColumnBases[u32(x+d.x*z)]=u32(max(0,desired));
}

@compute @workgroup_size(8,8,1)
fn smoothRemesh(@builtin(global_invocation_id) gid:vec3u){
  let d=packedDims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}let x=i32(gid.x);let z=i32(gid.y);let index=u32(x+d.x*z);let delta=u32(round(params.tall.w));var base=nextColumnBases[index];
  let offsets=array<vec2i,4>(vec2i(-1,0),vec2i(1,0),vec2i(0,-1),vec2i(0,1));
  for(var n=0u;n<4u;n+=1u){let q=vec2i(x,z)+offsets[n];if(q.x<0||q.x>=d.x||q.y<0||q.y>=d.z){continue;}base=min(base,nextColumnBases[u32(q.x+d.x*q.y)]+delta);}
  if(fineDims().y-regularLayers()>=2&&base<2u){base=2u;}smoothedColumnBases[index]=base;
}

fn leastSquaresPhi(x:i32,z:i32,newBase:i32)->vec2f{if(newBase<=1){let value=phiCell(vec3i(x,0,z));return vec2f(value);}var st=0.0;var stt=0.0;var sv=0.0;var stv=0.0;for(var y=0;y<newBase;y+=1){let t=f32(y)/f32(newBase-1);let value=phiCell(vec3i(x,y,z));st+=t;stt+=t*t;sv+=value;stv+=t*value;}let n=f32(newBase);let slope=(n*stv-st*sv)/max(n*stt-st*st,1e-6);let intercept=(sv-slope*st)/n;return vec2f(intercept,intercept+slope);}
fn leastSquaresVelocity(x:i32,z:i32,newBase:i32,top:bool)->vec3f{if(newBase<=1){return velocityCell(vec3i(x,0,z));}var st=0.0;var stt=0.0;var sv=vec3f(0.0);var stv=vec3f(0.0);for(var y=0;y<newBase;y+=1){let t=f32(y)/f32(newBase-1);let value=velocityCell(vec3i(x,y,z));st+=t;stt+=t*t;sv+=value;stv+=t*value;}let n=f32(newBase);let slope=(n*stv-st*sv)/max(n*stt-st*st,1e-6);let intercept=(sv-slope*st)/n;return select(intercept,intercept+slope,top);}
@compute @workgroup_size(4,4,4)
fn remap(@builtin(global_invocation_id) gid:vec3u){
	  let id=vec3i(gid);if(!validPacked(id)){return;}let d=packedDims();let newBase=i32(nextColumnBases[u32(id.x+d.x*id.z)]);var p=vec3f(f32(id.x)+0.5,0.5,f32(id.z)+0.5);if(id.y==1){p.y=max(0.5,f32(newBase)-0.5);}else if(id.y>=2){p.y=f32(newBase+id.y-2)+0.5;}let isActive=select(newBase>0,newBase+id.y-2<fineDims().y,id.y>=2);let limit=5.0*min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));var phi=limit;var velocity=vec3f(0.0);var pressure=0.0;if(isActive){pressure=samplePressure(p);if(id.y<2){let fit=leastSquaresPhi(id.x,id.z,newBase);let crossing=(fit.x<=0.0)!=(fit.y<=0.0);phi=select(select(fit.x,fit.y,id.y==1),fit.y,crossing);velocity=leastSquaresVelocity(id.x,id.z,newBase,id.y==1);}else{phi=samplePhi(p);for(var component=0u;component<3u;component+=1u){let faceP=p+0.5*vec3f(axisOffset(component));velocity[component]=sampleVelocityComponent(faceP,component);}}}
  textureStore(velocityOut,id,vec4f(velocity,0.0));textureStore(volumeOut,id,vec4f(clamp(phi,-limit,limit)));textureStore(pressureOut,id,vec4f(pressure));if(id.y==0){textureStore(columnBaseOut,vec2i(id.x,id.z),vec4f(f32(newBase)));}
}

@compute @workgroup_size(4,4,4)
fn reduceDiagnostics(@builtin(global_invocation_id) gid:vec3u){
	  let id=vec3i(gid);if(!validPacked(id)||!activeSample(id)){return;}let phi=textureLoad(volumeIn,id,0).x;let velocityValue=textureLoad(velocityIn,id,0).xyz;
  if(!finiteScalar(phi)||!all(velocityValue==velocityValue)||any(abs(velocityValue)>vec3f(3.402823e38))){atomicAdd(&reductions[20],1u);return;}
	  let band=1.5*min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));var occupied=0.0;var interfaceCells=0.0;if(id.y==0){for(var y=0;y<baseAt(id.x,id.z);y+=1){let value=phiCell(vec3i(id.x,y,id.z));occupied+=occupancyFromPhi(value);if(abs(value)<band){interfaceCells+=1.0;}}}else if(id.y>=2){occupied=occupancyFromPhi(phi);interfaceCells=select(0.0,1.0,abs(phi)<band);}atomicAdd(&reductions[0],u32(clamp(occupied*256.0,0.0,4294967295.0)));atomicAdd(&reductions[7],u32(interfaceCells*256.0));atomicMax(&reductions[3],u32(baseAt(id.x,id.z)));
  if(pointSamplePhi(id)<=0.0){
    atomicMax(&reductions[1],u32(id.x+1));updatePositiveMaximum(length(velocityValue),2u,8u,id);
	    if(solidFractionCell(vec3i(floor(samplePoint(id))))<=0.9){updatePositiveMaximum(abs(divergenceAt(id)),5u,14u,id);let h=params.cellGravity.xyz;let cfl=max(abs(velocityValue.x)*params.dimsDt.w/h.x,max(abs(velocityValue.y)*params.dimsDt.w/h.y,abs(velocityValue.z)*params.dimsDt.w/h.z));updatePositiveMaximumOnly(cfl,30u);atomicMax(&governor[0],bitcast<u32>(cfl));if(cfl>1.0){atomicAdd(&reductions[31],1u);}}
  }
}
`;
