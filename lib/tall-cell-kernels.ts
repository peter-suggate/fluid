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
	@group(0) @binding(17) var<storage,read_write> sharpenDeposits:array<atomic<i32>>;
	@group(0) @binding(18) var transportVolumeIn:texture_3d<f32>;

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

// Paper Eq 4 stores POINT samples at the tall-cell endpoints; the packed
// bottom texel instead holds the conservative column-average VOF. The paper
// keeps the interface OUT of tall cells (Section 8), so a partial average is
// a transient the remesh resolves; until it does, the tall cell is treated
// as base uniform cells SHARING that density (mass-conserving paper
// semantics), never as an invented settled sub-cell interface. The earlier
// settled reinterpretation (fill height alpha*base above the terrain)
// re-teleported band water to the column floor every step, manufacturing a
// phantom air gap under falling water whose collapse pumped kinetic energy
// for the whole slosh (2026-07-16 single-tall-cell probe audit).
// tallFillCells stays in subcell units for remeshing's draining-pool surface
// estimate and the representability floor only.
fn tallFillCells(x:i32,z:i32)->f32{let base=baseAt(x,z);if(base<=0){return 0.0;}return max(0.0,textureLoad(volumeIn,vec3i(x,0,z),0).x)*f32(base);}
// Total water in a column (tall store plus band), in subcell units, read
// from the current packed field. Used to keep remeshing from choosing a
// base too low to represent the column's actual content.
fn columnWaterCells(x:i32,z:i32)->f32{let d=packedDims();let oldBase=baseAt(x,z);var total=tallFillCells(x,z);for(var packedY:i32=2;packedY<d.y;packedY+=1){let worldY=oldBase+packedY-2;if(worldY>=fineDims().y){break;}total+=max(0.0,textureLoad(volumeIn,vec3i(x,packedY,z),0).x);}return total;}
// Highest wet subcell of a column (world Y, -1 when dry), read against the
// CURRENT base. The amount floor above cannot see WHERE the water sits: a
// column with air mixed below its wet top (wall splash) or one exactly full
// (collapsing dam corner) passes the amount check while a descending base
// still pushes the band ceiling through its free surface, squashing the
// interface flat against the closed ceiling face (2026-07-16 dam audit).
fn columnHighestWetCell(x:i32,z:i32)->i32{let d=packedDims();let base=baseAt(x,z);for(var packedY:i32=d.y-1;packedY>=2;packedY-=1){let worldY=base+packedY-2;if(worldY>=fineDims().y){continue;}if(textureLoad(volumeIn,vec3i(x,packedY,z),0).x>=0.5){return worldY;}}if(base>0&&textureLoad(volumeIn,vec3i(x,0,z),0).x>=0.5){return base-1;}return -1;}
// Occupancy of any subcell inside the tall region: the store's density,
// clamped to [0,1] (a remap residual can hold density above one; the excess
// drains through the correction divergence, not through classification).
// Paper invariant floor: a store below a WET band bottom is liquid no matter
// how far the conservative average has drifted (the interface is never
// inside a tall cell). Without the floor a drained store crossing 0.5 falls
// OUT of the pressure solve, the Sec 3.7 refill stops, and the column is
// stuck as an air cushion under standing water (2026-07-16 dam audit).
fn tallStoreAlpha(x:i32,z:i32)->f32{let base=baseAt(x,z);if(base<=0){return 0.0;}let a=clamp(textureLoad(volumeIn,vec3i(x,0,z),0).x,0.0,1.0);if(a<0.5&&packedDims().y>2&&textureLoad(volumeIn,vec3i(x,2,z),0).x>=0.5){return 0.5;}return a;}
// Paper invariant: the interface is never inside a tall cell, so a tall cell
// below a wet band bottom is LIQUID regardless of its conservative mass
// average (the average stays authoritative for transport only). Deriving
// wetness from the settled fill instead manufactured a phantom air gap
// between a transiently drained tall store and the wet band above it, which
// the pressure solve treated as a real cushion under the falling column.
fn tallConnectedToBand(x:i32,z:i32)->bool{
  let d=packedDims();if(d.y<=2){return false;}
  return textureLoad(volumeIn,vec3i(x,2,z),0).x>=0.5;
}
// Point-sample occupancy of a packed sample: both tall endpoints share the
// store density; band samples are already point stores.
fn pointSampleAlpha(id:vec3i)->f32{
  if(id.y<=1){return tallStoreAlpha(id.x,id.z);}
  return textureLoad(volumeIn,id,0).x;
}
fn volumeCell(q:vec3i)->f32{
  if(!validWorld(q)){return 0.0;}let base=baseAt(q.x,q.z);
  // Keep this reconstruction local (algebraically tallStoreAlpha with the
  // already-loaded base): volumeCell is expanded hundreds of times by the
  // bounded conservative-flux graph and an extra call layer turns Metal's
  // WGSL compile into minutes. The connected-store liquid floor mirrors
  // tallStoreAlpha; transport reads the raw average through
  // transportVolumeCell instead.
  if(q.y<base&&base>0){let a=clamp(textureLoad(volumeIn,vec3i(q.x,0,q.z),0).x,0.0,1.0);if(a<0.5&&packedDims().y>2&&textureLoad(volumeIn,vec3i(q.x,2,q.z),0).x>=0.5){return 0.5;}return a;}
  let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return 0.0;}return textureLoad(volumeIn,vec3i(q.x,packedY,q.z),0).x;
}
// The conservative limiter expands its flux stencil very aggressively on
// Metal. Read the already reconstructed cubic occupancy there so the compiler
// sees a bounded texture lookup instead of recursively cloning volumeCell.
fn transportVolumeCell(q:vec3i)->f32{if(!validWorld(q)){return 0.0;}return textureLoad(transportVolumeIn,q,0).x;}
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
fn sampleVolume(p:vec3f)->f32{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);
  let a=mix(mix(volumeCell(b),volumeCell(b+vec3i(1,0,0)),f.x),mix(volumeCell(b+vec3i(0,1,0)),volumeCell(b+vec3i(1,1,0)),f.x),f.y);
  let c=mix(mix(volumeCell(b+vec3i(0,0,1)),volumeCell(b+vec3i(1,0,1)),f.x),mix(volumeCell(b+vec3i(0,1,1)),volumeCell(b+vec3i(1,1,1)),f.x),f.y);return mix(a,c,f.z);
}
fn sampleVelocity(p:vec3f)->vec3f{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);
  let a=mix(mix(velocityCell(b),velocityCell(b+vec3i(1,0,0)),f.x),mix(velocityCell(b+vec3i(0,1,0)),velocityCell(b+vec3i(1,1,0)),f.x),f.y);
  let c=mix(mix(velocityCell(b+vec3i(0,0,1)),velocityCell(b+vec3i(1,0,1)),f.x),mix(velocityCell(b+vec3i(0,1,1)),velocityCell(b+vec3i(1,1,1)),f.x),f.y);return mix(a,c,f.z);
}
fn samplePressure(p:vec3f)->f32{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);
  let a=mix(mix(pressureCell(b),pressureCell(b+vec3i(1,0,0)),f.x),mix(pressureCell(b+vec3i(0,1,0)),pressureCell(b+vec3i(1,1,0)),f.x),f.y);
  let c=mix(mix(pressureCell(b+vec3i(0,0,1)),pressureCell(b+vec3i(1,0,1)),f.x),mix(pressureCell(b+vec3i(0,1,1)),pressureCell(b+vec3i(1,1,1)),f.x),f.y);return mix(a,c,f.z);
}

fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}
	fn insideRigid(body:RigidBody,world:vec3f)->bool{
  let p=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
  if(shape==0){return length(p)<=d.x;}if(shape==1){return all(abs(p)<=0.5*d);}if(shape==2){let cy=clamp(p.y,-0.5*d.y,0.5*d.y);return length(vec3f(p.x,p.y-cy,p.z))<=d.x;}return p.x*p.x+p.z*p.z<=d.x*d.x&&abs(p.y)<=0.5*d.y;
	}
	fn rigidVelocityAt(world:vec3f)->vec4f{
	  let bodyCount=u32(round(params.boundary.z));for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let body=rigidBodies[bodyIndex];if(insideRigid(body,world)){let arm=world-body.positionShape.xyz;return vec4f(body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm),f32(bodyIndex+1u));}}
	  return vec4f(0.0);
	}
	fn solidVelocityCell(q:vec3i)->vec3f{return rigidVelocityAt(worldFromPoint(vec3f(vec3i(q))+vec3f(0.5))).xyz;}

fn axisOffset(axis:u32)->vec3i{return select(select(vec3i(0,0,1),vec3i(0,1,0),axis==1u),vec3i(1,0,0),axis==0u);}
// Equation 18 pressure-gradient correction evaluated at a virtual cubic
// sample. This diagnoses the paper's Section 5 middle-face limitation without
// introducing additional pressure unknowns.
fn pressureGradientAt(q:vec3i,axis:u32)->f32{
  let plus=q+axisOffset(axis);if(!validWorld(q)||!validWorld(plus)){return 0.0;}if(axis==1u&&!representedWorld(plus)){return 0.0;}let ownAlpha=volumeCell(q);let plusAlpha=volumeCell(plus);let ownLiquid=ownAlpha>=0.5;let plusLiquid=plusAlpha>=0.5;if(!ownLiquid&&!plusLiquid){return 0.0;}var ownPressure=select(0.0,pressureCell(q),ownLiquid);var plusPressure=select(0.0,pressureCell(plus),plusLiquid);var theta=1.0;if(ownLiquid!=plusLiquid){let liquidAlpha=select(plusAlpha,ownAlpha,ownLiquid);let airAlpha=select(ownAlpha,plusAlpha,ownLiquid);let own=max(0.05,liquidAlpha-0.5);let other=max(0.05,0.5-airAlpha);theta=clamp(own/(own+other),0.05,1.0);}if(solidFractionCell(plus)>0.9){plusPressure=ownPressure;}return (plusPressure-ownPressure)/(params.cellGravity.xyz[axis]*theta);
}
// Lateral gradient between neighbouring tall stores' bottom texels, with the
// same ghost-fluid handling as pressureGradientAt. The store is one lateral
// degree of freedom, so both the Jacobi assembly and this application couple
// bottom texel to bottom texel — exactly zero on hydrostatic fields.
fn storeLateralGradient(x:i32,z:i32,offset:vec2i,ownAlpha:f32,ownPressureIn:f32)->f32{
  let nx=x+offset.x;let nz=z+offset.y;let d=packedDims();if(nx<0||nx>=d.x||nz<0||nz>=d.z){return 0.0;}
  let plusAlpha=tallStoreAlpha(nx,nz);let ownLiquid=ownAlpha>=0.5;let plusLiquid=plusAlpha>=0.5;if(!ownLiquid&&!plusLiquid){return 0.0;}
  var ownPressure=select(0.0,ownPressureIn,ownLiquid);var plusPressure=select(0.0,textureLoad(pressureIn,vec3i(nx,0,nz),0).x,plusLiquid);
  var theta=1.0;if(ownLiquid!=plusLiquid){let liquidAlpha=select(plusAlpha,ownAlpha,ownLiquid);let airAlpha=select(ownAlpha,plusAlpha,ownLiquid);let own=max(0.05,liquidAlpha-0.5);let other=max(0.05,0.5-airAlpha);theta=clamp(own/(own+other),0.05,1.0);}
  if(textureLoad(solidFractionIn,vec3i(nx,0,nz),0).x>0.9){plusPressure=ownPressure;}
  return (plusPressure-ownPressure)/theta;
}
fn upwind(face:f32,negative:f32,positive:f32)->f32{return face*select(positive,negative,face>=0.0);}
fn rawVolumeFlux(q:vec3i,axis:u32,dt:f32)->f32{
  let offset=axisOffset(axis);if(!representedWorld(q)||!representedWorld(q+offset)){return 0.0;}let speed=velocityCell(q)[axis];return dt/params.cellGravity.xyz[axis]*upwind(speed,transportVolumeCell(q),transportVolumeCell(q+offset));
}
fn outwardFlux(q:vec3i,dt:f32)->f32{
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);return max(rawVolumeFlux(q,0u,dt),0.0)+max(-rawVolumeFlux(q-ex,0u,dt),0.0)+max(rawVolumeFlux(q,1u,dt),0.0)+max(-rawVolumeFlux(q-ey,1u,dt),0.0)+max(rawVolumeFlux(q,2u,dt),0.0)+max(-rawVolumeFlux(q-ez,2u,dt),0.0);
}
fn inwardFlux(q:vec3i,dt:f32)->f32{
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);return max(-rawVolumeFlux(q,0u,dt),0.0)+max(rawVolumeFlux(q-ex,0u,dt),0.0)+max(-rawVolumeFlux(q,1u,dt),0.0)+max(rawVolumeFlux(q-ey,1u,dt),0.0)+max(-rawVolumeFlux(q,2u,dt),0.0)+max(rawVolumeFlux(q-ez,2u,dt),0.0);
}
fn donorScale(q:vec3i,dt:f32)->f32{return min(1.0,transportVolumeCell(q)/max(outwardFlux(q,dt),1e-9));}
fn receiverScale(q:vec3i,dt:f32)->f32{return min(1.0,max(0.0,1.0-transportVolumeCell(q))/max(inwardFlux(q,dt),1e-9));}
fn limitedInternalFlux(q:vec3i,axis:u32,dt:f32)->f32{let n=q+axisOffset(axis);let flux=rawVolumeFlux(q,axis,dt);if(flux>=0.0){return flux*min(donorScale(q,dt),receiverScale(n,dt));}return flux*min(donorScale(n,dt),receiverScale(q,dt));}
fn limitedFlux(q:vec3i,axis:u32,dt:f32)->f32{return limitedInternalFlux(q,axis,dt);}
fn advectedVolume(q:vec3i,dt:f32)->f32{
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);let transported=transportVolumeCell(q)-(limitedFlux(q,0u,dt)-limitedFlux(q-ex,0u,dt)+limitedFlux(q,1u,dt)-limitedFlux(q-ey,1u,dt)+limitedFlux(q,2u,dt)-limitedFlux(q-ez,2u,dt));
  // No upper clamp on the transported value: a clamp here would destroy
  // sharpening deposits or remap residuals above one, which drain through
  // the correction divergence instead. The inflow source alone is bounded by
  // the cell's remaining capacity, as the old clamp did implicitly.
  let bounded=max(transported,0.0);
  return bounded+min(inflowReceiverSource(q,dt),max(0.0,1.0-bounded));
}
fn integratedFluxSegment(q:vec3i,axis:u32,start:i32,count:i32,dt:f32)->f32{if(count<=0){return 0.0;}var total=0.0;for(var y=0;y<count;y+=1){total+=limitedInternalFlux(q+vec3i(0,start+y,0),axis,dt);}return total;}
fn integratedSharedTallFlux(q:vec3i,axis:u32,height:i32,dt:f32)->f32{
  if(height<=0){return 0.0;}if(height<=12){return integratedFluxSegment(q,axis,0,height,dt);}let width=f32(height)/12.0;var total=0.0;
  // The tall-cell fields are reconstructed vertically, so stratified
  // midpoint quadrature preserves a constant stencil as depth grows. Both
  // columns call this same oriented face function, retaining exact pairwise
  // cancellation even though the integral is approximated.
  for(var sample=0;sample<12;sample+=1){let y=min(height-1,i32(floor((f32(sample)+0.5)*width)));total+=width*limitedInternalFlux(q+vec3i(0,y,0),axis,dt);}return total;
}
fn integratedFaceFlux(q:vec3i,axis:u32,height:i32,dt:f32)->f32{
  if(height<=0){return 0.0;}let offset=axisOffset(axis);let neighbor=q+offset;if(!validWorld(q)||!validWorld(neighbor)){return 0.0;}let split=clamp(min(baseAt(q.x,q.z),baseAt(neighbor.x,neighbor.z)),0,height);var total=integratedSharedTallFlux(q,axis,split,dt);
  // Only the bounded D-height mismatch is expanded exactly. Its opposite side
  // consists of ordinary cells, so using their identical face flux preserves
  // pairwise cancellation across the tall/regular transition.
  for(var y=split;y<height;y+=1){total+=limitedFlux(q+vec3i(0,y,0),axis,dt);}return total;
}
fn advectedTallVolume(x:i32,z:i32,dt:f32)->f32{
  let base=baseAt(x,z);if(base<=0){return 0.0;}let ex=vec3i(1,0,0);let ez=vec3i(0,0,1);let q=vec3i(x,0,z);var amount=textureLoad(volumeIn,vec3i(x,0,z),0).x*f32(base);
  amount-=integratedFaceFlux(q,0u,base,dt)-integratedFaceFlux(q-ex,0u,base,dt)+integratedFaceFlux(q,2u,base,dt)-integratedFaceFlux(q-ez,2u,base,dt);
  // Remeshing may temporarily store density above one in the conservative
  // tall-cell average. Preserve that mass here and let bounded face fluxes
  // redistribute it; an upper clamp would erase the remap residual.
  amount-=limitedFlux(vec3i(x,base-1,z),1u,dt);return max(0.0,amount/f32(base));
}
// The bottom sample owns tall-cell mass; the zero-weight top sample carries
// the strongest overlying band topology for interpolation and pressure.
fn advectedTallTopGuide(x:i32,z:i32,base:i32,dt:f32)->f32{var guide=0.0;for(var offset=0;offset<regularLayers();offset+=1){guide=max(guide,advectedVolume(vec3i(x,base+offset,z),dt));}return guide;}
fn volumeGradient(q:vec3i)->vec3f{let h=params.cellGravity.xyz;return vec3f(volumeCell(q+vec3i(1,0,0))-volumeCell(q-vec3i(1,0,0)),volumeCell(q+vec3i(0,1,0))-volumeCell(q-vec3i(0,1,0)),volumeCell(q+vec3i(0,0,1))-volumeCell(q-vec3i(0,0,1)))/(2.0*h);}
fn interfaceNormal(q:vec3i)->vec3f{let g=volumeGradient(q);return g/max(length(g),1e-6);}
fn curvature(q:vec3i)->f32{let h=params.cellGravity.xyz;return -((interfaceNormal(q+vec3i(1,0,0)).x-interfaceNormal(q-vec3i(1,0,0)).x)/(2.0*h.x)+(interfaceNormal(q+vec3i(0,1,0)).y-interfaceNormal(q-vec3i(0,1,0)).y)/(2.0*h.y)+(interfaceNormal(q+vec3i(0,0,1)).z-interfaceNormal(q-vec3i(0,0,1)).z)/(2.0*h.z));}
fn velocityLaplacian(q:vec3i)->vec3f{let h=params.cellGravity.xyz;let c=diffusionVelocity(q);return (diffusionVelocity(q+vec3i(1,0,0))-2.0*c+diffusionVelocity(q-vec3i(1,0,0)))/(h.x*h.x)+(diffusionVelocity(q+vec3i(0,1,0))-2.0*c+diffusionVelocity(q-vec3i(0,1,0)))/(h.y*h.y)+(diffusionVelocity(q+vec3i(0,0,1))-2.0*c+diffusionVelocity(q-vec3i(0,0,1)))/(h.z*h.z);}
fn strainMagnitude(q:vec3i)->f32{let h=params.cellGravity.xyz;let dx=(diffusionVelocity(q+vec3i(1,0,0))-diffusionVelocity(q-vec3i(1,0,0)))/(2.0*h.x);let dy=(diffusionVelocity(q+vec3i(0,1,0))-diffusionVelocity(q-vec3i(0,1,0)))/(2.0*h.y);let dz=(diffusionVelocity(q+vec3i(0,0,1))-diffusionVelocity(q-vec3i(0,0,1)))/(2.0*h.z);let sxy=0.5*(dx.y+dy.x);let sxz=0.5*(dx.z+dz.x);let syz=0.5*(dy.z+dz.y);return sqrt(2.0*(dx.x*dx.x+dy.y*dy.y+dz.z*dz.z+2.0*(sxy*sxy+sxz*sxz+syz*syz)));}

fn tracedVelocity(p:vec3f,signedDt:f32)->vec3f{let h=params.cellGravity.xyz;let first=sampleVelocity(p);let midpoint=p-0.5*first*signedDt/h;return sampleVelocity(p-sampleVelocity(midpoint)*signedDt/h);}
@compute @workgroup_size(4,4,4)
fn extrapolateVelocity(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));return;}let q=vec3i(floor(samplePoint(id)));if(isInflowVelocityCell(q)){textureStore(velocityOut,id,vec4f(applyInflowVelocity(q,textureLoad(velocityIn,id,0).xyz),1.0));return;}let alpha=pointSampleAlpha(id);if(alpha>=0.5){textureStore(velocityOut,id,vec4f(textureLoad(velocityIn,id,0).xyz,1.0));return;}let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));var sum=vec3f(0.0);var weight=0.0;for(var index=0;index<6;index+=1){let state=velocityStateCell(q+offsets[index]);if(state.w>0.5){sum+=state.xyz;weight+=1.0;}}if(weight>0.0){textureStore(velocityOut,id,vec4f(sum/weight,1.0));}else{textureStore(velocityOut,id,vec4f(textureLoad(velocityIn,id,0).xyz,0.0));}}
@compute @workgroup_size(4,4,4)
fn predictVelocity(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));return;}textureStore(velocityOut,id,vec4f(tracedVelocity(samplePoint(id),params.dimsDt.w),0.0));}
@compute @workgroup_size(4,4,4)
fn reverseVelocity(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));return;}textureStore(velocityOut,id,vec4f(tracedVelocity(samplePoint(id),-params.dimsDt.w),0.0));}

fn boundedMacCormack(id:vec3i,p:vec3f)->vec3f{
  let h=params.cellGravity.xyz;let first=sampleVelocity(p);let midpoint=p-0.5*first*params.dimsDt.w/h;let departure=p-sampleVelocity(midpoint)*params.dimsDt.w/h;
  let sampleCoordinate=clamp(departure-vec3f(0.5),vec3f(0.0),vec3f(fineDims()-vec3i(1)));let b=vec3i(floor(sampleCoordinate));
  var lower=velocityCell(b);var upper=lower;
  for(var corner:u32=1u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let value=velocityCell(b+offset);lower=min(lower,value);upper=max(upper,value);}
  let predicted=textureLoad(predictedVelocityIn,id,0).xyz;let original=textureLoad(velocityIn,id,0).xyz;let reversed=textureLoad(reversedVelocityIn,id,0).xyz;var corrected=predicted+0.5*(original-reversed);
  for(var component:u32=0u;component<3u;component+=1u){if(corrected[component]<lower[component]||corrected[component]>upper[component]){corrected[component]=predicted[component];}}
  return corrected;
}

@compute @workgroup_size(4,4,4)
fn finishAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(0.0));textureStore(pressureOut,id,vec4f(0.0));return;}
	  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let p=samplePoint(id);let q=vec3i(floor(p));var v=boundedMacCormack(id,p);var alpha=advectedVolume(q,dt);var pointAlpha=alpha;let base=baseAt(id.x,id.z);var tallAlpha=0.0;
	  if(id.y<=1){tallAlpha=advectedTallVolume(id.x,id.z,dt);}if(id.y==0){alpha=tallAlpha;pointAlpha=clamp(tallAlpha,0.0,1.0);}else if(id.y==1){alpha=advectedTallTopGuide(id.x,id.z,base,dt);pointAlpha=clamp(tallAlpha,0.0,1.0);}
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
  if(pointAlpha>=0.5||volumeCell(q+vec3i(0,1,0))>=0.5){v.y+=params.cellGravity.w*dt;}if(pointAlpha>0.001){let nu=params.physical.y/params.physical.x;v+=dt*nu*velocityLaplacian(q);if(pointAlpha<0.999){v+=dt*params.boundary.x/params.physical.x*curvature(q)*volumeGradient(q);}}
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
  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(alpha));textureStore(pressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn finishSemiLagrangianAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(0.0));textureStore(pressureOut,id,vec4f(0.0));return;}
	  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let p=samplePoint(id);let q=vec3i(floor(p));var v=textureLoad(predictedVelocityIn,id,0).xyz;var alpha=advectedVolume(q,dt);var pointAlpha=alpha;let base=baseAt(id.x,id.z);var tallAlpha=0.0;
	  if(id.y<=1){tallAlpha=advectedTallVolume(id.x,id.z,dt);}if(id.y==0){alpha=tallAlpha;pointAlpha=clamp(tallAlpha,0.0,1.0);}else if(id.y==1){alpha=advectedTallTopGuide(id.x,id.z,base,dt);pointAlpha=clamp(tallAlpha,0.0,1.0);}
  // Match finishAdvection exactly after choosing the transport estimate so
  // the A/B option isolates MacCormack correction from all other physics.
  // MAC gravity gate: the stored v.y is the face between q and q+y, so it
  // integrates gravity when EITHER adjacent cell is liquid (matching the
  // uniform path). Gating on the own sample alone starved the face beneath
  // every droplet, which levitated small drops and left splash water welded
  // to the ceiling where the wall zeroes the only gravity-fed face.
  if(pointAlpha>=0.5||volumeCell(q+vec3i(0,1,0))>=0.5){v.y+=params.cellGravity.w*dt;}if(pointAlpha>0.001){let nu=params.physical.y/params.physical.x;v+=dt*nu*velocityLaplacian(q);if(pointAlpha<0.999){v+=dt*params.boundary.x/params.physical.x*curvature(q)*volumeGradient(q);}}
  // Same absolute speed rail as finishAdvection (see the comment there).
  let speedCap=vec3f(params.container.w);v=clamp(v,-speedCap,speedCap);
  v=applyInflowVelocity(q,v);let d=fineDims();if(q.x+1>=d.x){v.x=0.0;}if(q.y+1>=d.y||!representedWorld(q+vec3i(0,1,0))){v.y=0.0;}if(q.z+1>=d.z){v.z=0.0;}
  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(alpha));textureStore(pressureOut,id,vec4f(0.0));
}

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
// Mass-Conserving Eulerian Liquid Simulation Sec 3.7: cells holding more
// density than they can represent (rho' > 1) receive artificial divergence
// min(lambda (rho'-1), eta) with lambda=0.5, eta=1 so the pressure solve
// pushes the excess out gradually. The paper applies the push per 1/30 s
// step; expressing it as a rate keeps the drain speed independent of dt.
// The tall store is base cells sharing one DENSITY, so its correction uses
// the same density units as any band cell. Scaling by base (subcell units)
// made the tall correction up to base times the paper rate, an artificial
// source/sink violent enough to pump energy at every partial tall store.
fn volumeCorrectionDivergence(id:vec3i)->f32{
  let alpha=textureLoad(volumeIn,id,0).x;var excess=max(0.0,alpha-1.0);
  if(id.y==0){
    // The mirrored Sec 3.7 branch for the tall store: a submerged tall cell
    // (wet band bottom above it) holding rho < 1 must gradually REFILL at
    // the paper's density rate, else the deficit drifts toward the 0.5
    // classification cliff and the whole column flips to air at once.
    if(excess<=0.0&&tallConnectedToBand(id.x,id.z)){let deficit=max(0.0,1.0-alpha);if(deficit>0.0){return -min(0.5*deficit,1.0)*30.0;}}
  }else if(id.y==1){excess=0.0;}
  if(excess<=0.0){return 0.0;}return min(0.5*excess,1.0)*30.0;
}
@compute @workgroup_size(4,4,4)
// The correction divergence must be SUBTRACTED from the measured divergence:
// solving Lap p = rho (div - c)/dt and projecting v -= dt/rho grad p leaves
// div_new = +c, the outward push that drains an overfull cell. Adding it
// yields div_new = -c — inflow into the overfull cell, a positive feedback
// that pumps pressure and energy at every rho > 1 site (the tall stores'
// remap residuals fire this constantly; see the 2026-07-15 dam-break audit).
fn buildPressureRhs(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!validPacked(id)){return;}let wet=activeSample(id)&&pointSampleAlpha(id)>=0.5;let rhs=select(0.0,params.physical.x*(divergenceAt(id)-volumeCorrectionDivergence(id))/params.dimsDt.w,wet);textureStore(pressureOut,id,vec4f(rhs,0.0,0.0,0.0));}
fn interfaceFraction(a:f32,b:f32)->f32{return clamp((a-0.5)/max(abs(a-b),1e-6),0.05,1.0);}
	fn pressureTerm(ownAlpha:f32,otherAlpha:f32,otherPressure:f32,solidFraction:f32,coefficient:f32)->vec2f{if(otherAlpha>=0.5){let open=1.0-clamp(solidFraction,0.0,1.0);return vec2f(coefficient*open,coefficient*open*otherPressure);}return vec2f(coefficient/interfaceFraction(ownAlpha,otherAlpha),0.0);}

@compute @workgroup_size(4,4,4)
fn jacobi(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(pressureOut,id,vec4f(0.0));return;}let ownAlpha=pointSampleAlpha(id);if(ownAlpha<0.5){textureStore(pressureOut,id,vec4f(0.0));return;}
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
	    if(id.x>0){stencil+=pressureTerm(ownAlpha,tallStoreAlpha(id.x-1,id.z),textureLoad(pressureIn,vec3i(id.x-1,0,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x-1,0,id.z),0).x,1.0/(h.x*h.x));}
	    if(id.x+1<d.x){stencil+=pressureTerm(ownAlpha,tallStoreAlpha(id.x+1,id.z),textureLoad(pressureIn,vec3i(id.x+1,0,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x+1,0,id.z),0).x,1.0/(h.x*h.x));}
	    if(id.z>0){stencil+=pressureTerm(ownAlpha,tallStoreAlpha(id.x,id.z-1),textureLoad(pressureIn,vec3i(id.x,0,id.z-1),0).x,textureLoad(solidFractionIn,vec3i(id.x,0,id.z-1),0).x,1.0/(h.z*h.z));}
	    if(id.z+1<d.z){stencil+=pressureTerm(ownAlpha,tallStoreAlpha(id.x,id.z+1),textureLoad(pressureIn,vec3i(id.x,0,id.z+1),0).x,textureLoad(solidFractionIn,vec3i(id.x,0,id.z+1),0).x,1.0/(h.z*h.z));}
	    let distance=max(h.y,f32(base-1)*h.y);
	    stencil+=pressureTerm(ownAlpha,tallStoreAlpha(id.x,id.z),textureLoad(pressureIn,vec3i(id.x,1,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x,1,id.z),0).x,1.0/(distance*h.y));
	  } else {
	  if(q.x>0){let n=q-vec3i(1,0,0);stencil+=pressureTerm(ownAlpha,volumeCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.x*h.x));}if(q.x+1<d.x){let n=q+vec3i(1,0,0);stencil+=pressureTerm(ownAlpha,volumeCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.x*h.x));}if(q.z>0){let n=q-vec3i(0,0,1);stencil+=pressureTerm(ownAlpha,volumeCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.z*h.z));}if(q.z+1<d.z){let n=q+vec3i(0,0,1);stencil+=pressureTerm(ownAlpha,volumeCell(n),pressureCell(n),solidFractionCell(n),1.0/(h.z*h.z));}
	  }
	  if(id.y==0){/* lateral and vertical terms assembled in the bottom sub-cell block above */}
	  // Top endpoint row: the paper's Eq 15/16 coefficients (band at 1/h^2,
	  // bottom through the Eq 5 interpolated below-pressure at 1/(distance*h)).
	  // This row is NOT the exact adjoint of the staggered top-sub-cell point
	  // divergence (whose vertical sensitivity carries s = 1/(base-1)), so a
	  // converged solve retains an O(1-s) share of the interface-face
	  // divergence — the paper's acknowledged non-idempotent projection. The
	  // exact adjoint (band s/h^2, bottom s^2/h^2) was tried on 2026-07-16 and
	  // closed that leak, but it anchors the top-dof layer so weakly at large
	  // bases that the multigrid diverged outright on the 20 m deep-water
	  // scene; the paper's strong row is the stable choice, and its residual
	  // leak is bounded because the bottom row pins the store to the floor.
	  // Hydrostatic fields satisfy this row exactly.
	  else if(id.y==1){let distance=max(h.y,f32(base-1)*h.y);stencil+=pressureTerm(ownAlpha,tallStoreAlpha(id.x,id.z),textureLoad(pressureIn,vec3i(id.x,0,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x,0,id.z),0).x,1.0/(distance*h.y));if(activeSample(vec3i(id.x,2,id.z))){stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,vec3i(id.x,2,id.z),0).x,textureLoad(pressureIn,vec3i(id.x,2,id.z),0).x,textureLoad(solidFractionIn,vec3i(id.x,2,id.z),0).x,1.0/(h.y*h.y));}}
	  else {if(id.y>2||base>=2){let below=id-vec3i(0,1,0);let belowAlpha=select(textureLoad(volumeIn,below,0).x,tallStoreAlpha(id.x,id.z),id.y==2&&base>0);stencil+=pressureTerm(ownAlpha,belowAlpha,textureLoad(pressureIn,below,0).x,textureLoad(solidFractionIn,below,0).x,1.0/(h.y*h.y));}if(activeSample(id+vec3i(0,1,0))){let above=id+vec3i(0,1,0);stencil+=pressureTerm(ownAlpha,textureLoad(volumeIn,above,0).x,textureLoad(pressureIn,above,0).x,textureLoad(solidFractionIn,above,0).x,1.0/(h.y*h.y));}else if(q.y+1>=d.y){stencil+=pressureTerm(ownAlpha,0.0,0.0,1.0,1.0/(h.y*h.y));}}
  let rhs=params.physical.x*(divergenceAt(id)-volumeCorrectionDivergence(id))/params.dimsDt.w;let old=textureLoad(pressureIn,id,0).x;let next=(stencil.y-rhs)/max(stencil.x,1e-9);textureStore(pressureOut,id,vec4f(mix(old,next,0.8),0.0,0.0,0.0));
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
	    let ownAlpha=tallStoreAlpha(id.x,id.z);let ownPressure=textureLoad(pressureIn,vec3i(id.x,0,id.z),0).x;
	    if(!validWorld(q+offsets[0])){v.x=0.0;}else{v.x-=scale*storeLateralGradient(id.x,id.z,vec2i(1,0),ownAlpha,ownPressure)/h.x;}
	    if(!validWorld(q+offsets[2])){v.z=0.0;}else{v.z-=scale*storeLateralGradient(id.x,id.z,vec2i(0,1),ownAlpha,ownPressure)/h.z;}
	    if(!validWorld(q+offsets[1])||!representedWorld(q+offsets[1])){v.y=0.0;}else{v.y-=scale*pressureGradientAt(q,1u);}
	  } else {
	  for(var axis=0u;axis<3u;axis+=1u){let plus=q+offsets[axis];if(!validWorld(plus)){v[axis]=0.0;continue;}if(axis==1u&&!representedWorld(plus)){v[axis]=0.0;continue;}v[axis]-=scale*pressureGradientAt(q,axis);}
	  }
	  let speedCap=vec3f(params.container.w);v=clamp(v,-speedCap,speedCap);
	  v=applyInflowVelocity(q,v);textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));
	}


	@compute @workgroup_size(4,4,4)
	fn coupleRigid(@builtin(global_invocation_id) gid:vec3u){
	  let id=vec3i(gid);if(!validPacked(id)){return;}if(!activeSample(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(0.0));textureStore(pressureOut,id,vec4f(0.0));return;}
	  let q=vec3i(floor(samplePoint(id)));let alpha=textureLoad(volumeIn,id,0).x;let oldV=textureLoad(velocityIn,id,0).xyz;var v=oldV;let h=params.cellGravity.xyz;let bodyCount=u32(round(params.boundary.z));var solid=0.0;
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
	        if(id.y==0){let solidDensity=max(body.angularVelocity.w,1.0);let reducedDensity=params.physical.x*solidDensity/(params.physical.x+solidDensity);let fluidImpulse=reducedDensity*h.x*h.y*h.z*alpha*virtualSolid*(solidVelocity-virtualVelocity);let reaction=-fluidImpulse;let torque=cross(arm,reaction);let exchangeBase=owner*8u;atomicAdd(&rigidExchange[exchangeBase],i32(round(reaction.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+1u],i32(round(reaction.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+2u],i32(round(reaction.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+3u],i32(round(torque.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+4u],i32(round(torque.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+5u],i32(round(torque.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+6u],i32(round(alpha*virtualSolid*65536.0)));}
	      }
	    }
	    if(coupledWeight>0.0){v+=weightedDelta/coupledWeight;}solid=solidBasis/max(basisWeight,1e-6);
	  }else{
	    let p=samplePoint(id);let world=worldFromPoint(p);var owner=12u;
	    for(var corner:u32=0u;corner<8u;corner+=1u){let offset=vec3f(select(-0.4,0.4,(corner&1u)>0u),select(-0.4,0.4,(corner&2u)>0u),select(-0.4,0.4,(corner&4u)>0u))*h;let sample=world+offset;for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],sample)){solid+=0.125;owner=min(owner,bodyIndex);break;}}}
	    if(owner<12u&&solid>0.0){let body=rigidBodies[owner];let arm=world-body.positionShape.xyz;let solidVelocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);v=mix(v,solidVelocity,solid);let solidDensity=max(body.angularVelocity.w,1.0);let reducedDensity=params.physical.x*solidDensity/(params.physical.x+solidDensity);let fluidImpulse=reducedDensity*h.x*h.y*h.z*alpha*solid*(solidVelocity-oldV);let reaction=-fluidImpulse;let torque=cross(arm,reaction);let exchangeBase=owner*8u;atomicAdd(&rigidExchange[exchangeBase],i32(round(reaction.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+1u],i32(round(reaction.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+2u],i32(round(reaction.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+3u],i32(round(torque.x*1e6)));atomicAdd(&rigidExchange[exchangeBase+4u],i32(round(torque.y*1e6)));atomicAdd(&rigidExchange[exchangeBase+5u],i32(round(torque.z*1e6)));atomicAdd(&rigidExchange[exchangeBase+6u],i32(round(alpha*solid*65536.0)));}
	  }
	  // The prescribed reservoir occupies the nozzle's open channel. The
	  // display cylinder is a filled rigid primitive, so carve its inlet cells
	  // back out of the pressure mask and let the boundary velocity win there.
	  if(isInflowVelocityCell(q)){solid=0.0;v=applyInflowVelocity(q,v);}
	  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(alpha));textureStore(pressureOut,id,vec4f(solid));
	}

@compute @workgroup_size(4,4,4)
fn reduceBeforeProjection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)||!activeSample(id)){return;}let alpha=textureLoad(volumeIn,id,0).x;let velocityValue=textureLoad(velocityIn,id,0).xyz;let speed=length(velocityValue);
  if(!finiteScalar(alpha)||!all(velocityValue==velocityValue)||any(abs(velocityValue)>vec3f(3.402823e38))){atomicAdd(&reductions[20],1u);return;}
  if(pointSampleAlpha(id)>=0.5&&solidFractionCell(vec3i(floor(samplePoint(id))))<=0.9){updatePositiveMaximum(abs(divergenceAt(id)),4u,11u,id);}else if(id.y>=2){updatePositiveMaximum(speed,6u,17u,id);}
}

// --- Density sharpening (Mass-Conserving Eulerian Liquid Simulation Sec 3.5,
// Eq 4-17 and Algorithm 2; verbatim transcription in docs/TALL_CELLS_PAPER.md
// Appendix B.3). Runs after conservative advection on the stored band. Pass 1
// computes the density correction and applies it; the removed mass (Eq 17
// keeps corrections non-positive: mass only moves from the air side to the
// liquid side) is returned in pass 2 by tracing along the density gradient to
// the 0.5 iso-contour and depositing with trilinear weights into a
// fixed-point buffer, which pass 3 folds back into the density field.
fn packedDepositIndex(q:vec3i)->i32{
  if(!validWorld(q)){return -1;}let d=packedDims();let base=baseAt(q.x,q.z);
  if(q.y<base&&base>0){return q.x+d.x*(0+d.y*q.z);}
  let packedY=2+q.y-base;if(packedY<2||packedY>=d.y){return -1;}return q.x+d.x*(packedY+d.y*q.z);
}
fn sharpenDeltaRho(q:vec3i)->f32{
  let rho=volumeCell(q);
  if(solidFractionCell(q)>0.9){return 0.0;}
  // The paper permits larger tau for a more cohesive interface. The tall
  // remap adds a small amount of diffusion, so use 0.45 (versus its 0.4
  // baseline) while retaining the same conservative Eq. 14-17 update.
  let h=params.cellGravity.xyz;let deltaT=3.0*params.dimsDt.w;let tau=0.45;
  // Upwind mass-change estimates per unit velocity (Eq 6/7), folded with the
  // Eq 8-13 1/dx^2 normalization so each term is the density fraction moved
  // by a unit velocity over the fictitious step.
  let sxp=-(rho-volumeCell(q-vec3i(1,0,0)))*deltaT/h.x;let sxm=-(volumeCell(q+vec3i(1,0,0))-rho)*deltaT/h.x;
  let syp=-(rho-volumeCell(q-vec3i(0,1,0)))*deltaT/h.y;let sym=-(volumeCell(q+vec3i(0,1,0))-rho)*deltaT/h.y;
  let szp=-(rho-volumeCell(q-vec3i(0,0,1)))*deltaT/h.z;let szm=-(volumeCell(q+vec3i(0,0,1))-rho)*deltaT/h.z;
  let gradPlus=sqrt(max(max(sxp,0.0)*max(sxp,0.0),min(sxm,0.0)*min(sxm,0.0))+max(max(syp,0.0)*max(syp,0.0),min(sym,0.0)*min(sym,0.0))+max(max(szp,0.0)*max(szp,0.0),min(szm,0.0)*min(szm,0.0)));
  let gradMinus=sqrt(max(min(sxp,0.0)*min(sxp,0.0),max(sxm,0.0)*max(sxm,0.0))+max(min(syp,0.0)*min(syp,0.0),max(sym,0.0)*max(sym,0.0))+max(min(szp,0.0)*min(szp,0.0),max(szm,0.0)*max(szm,0.0)));
  var maximumDifference=0.0;
  let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var index=0;index<6;index+=1){maximumDifference=max(maximumDifference,abs(rho-volumeCell(q+offsets[index])));}
  let weight=(rho-0.5)*(rho-0.5)*(rho-0.5)*(1.0-min(1.0,maximumDifference/tau));   // Eq 14
  var deltaRho=select(weight*gradMinus,weight*gradPlus,weight>=0.0);               // Eq 15
  // Eq 17 local-conservation modifications: never create negatives, clamp
  // dust to zero, and leave the liquid side untouched.
  if(rho+deltaRho<0.0||rho<1e-5){deltaRho=-rho;}else if(rho>0.5){deltaRho=0.0;}
  return deltaRho;
}
@compute @workgroup_size(4,4,4)
fn sharpenCompute(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)){return;}
  let stored=textureLoad(volumeIn,id,0).x;
  if(!activeSample(id)||id.y<2){textureStore(volumeOut,id,vec4f(stored));textureStore(pressureOut,id,vec4f(0.0));return;}
  let q=vec3i(floor(samplePoint(id)));
  let deltaRho=sharpenDeltaRho(q);
  textureStore(volumeOut,id,vec4f(stored+deltaRho));                               // Eq 16
  textureStore(pressureOut,id,vec4f(deltaRho));
}
@compute @workgroup_size(4,4,4)
fn sharpenScatter(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)||!activeSample(id)||id.y<2){return;}
  let deltaRho=textureLoad(pressureIn,id,0).x;if(deltaRho>=0.0){return;}
  // Algorithm 2: TraceAlongField from the cell center along the (sharpened)
  // density gradient until the 0.5 iso-contour, a distance of D cells
  // (D = 2.1 as in the paper's examples), or a solid boundary.
  var p=samplePoint(id);let maximumDistance=2.1;var travelled=0.0;let stepLength=0.5;
  for(var stepIndex=0;stepIndex<5;stepIndex+=1){
    if(sampleVolume(p)>=0.5||travelled>=maximumDistance){break;}
    let g=volumeGradient(vec3i(floor(p)));let magnitude=length(g);
    if(magnitude<1e-6){break;}
    let candidate=p+g/magnitude*stepLength;
    if(solidFractionCell(vec3i(floor(candidate)))>0.9){break;}
    p=candidate;travelled+=stepLength;
  }
  // The paper assumes the 0.5 iso-contour lies within D cells of every
  // sharpened cell. In diffused low-density regions no contour exists
  // nearby, and depositing at the trace end concentrates fog at its local
  // maxima until free-floating droplets nucleate above the water. When the
  // trace fails to reach liquid, return the mass to its own cell instead.
  if(sampleVolume(p)<0.5){
    let ownIndex=i32(id.x)+packedDims().x*(id.y+packedDims().y*id.z);
    atomicAdd(&sharpenDeposits[u32(ownIndex)],i32(round(-deltaRho*1048576.0)));return;
  }
  // ScatterValue: trilinear deposit of the removed mass, skipping solid or
  // unrepresented corners and renormalizing the remaining weights. Deposits
  // into the tall interior land in the bottom store as average density.
  let anchor=p-vec3f(0.5);let cell=vec3i(floor(anchor));let f=fract(anchor);
  var weights=array<f32,8>();var indices=array<i32,8>();var total=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let destination=cell+offset;
    var w=select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);
    let index=packedDepositIndex(destination);
    if(index<0||solidFractionCell(destination)>0.9){w=0.0;}
    let base=baseAt(destination.x,destination.z);
    let insideTall=destination.y<base&&base>0;
    // Band corners without remaining capacity are skipped so deposits cannot
    // push a stored band value past one, where the next advection clamp
    // would silently destroy the excess. Tall stores always absorb; their
    // temporary >1 average drains through the correction divergence.
    if(!insideTall&&w>0.0&&volumeCell(destination)>=1.0){w=0.0;}
    weights[corner]=w;indices[corner]=index;
    total+=w;
  }
  if(total<=1e-8){
    // Nothing representable near the trace target: return the mass in place.
    let ownIndex=i32(id.x)+packedDims().x*(id.y+packedDims().y*id.z);
    atomicAdd(&sharpenDeposits[u32(ownIndex)],i32(round(-deltaRho*1048576.0)));return;
  }
  // Deposits are MASS in fixed point; the resolve pass converts a tall
  // bottom store's mass back to its average by dividing by the base there,
  // after quantization, so small contributions cannot round away.
  for(var corner=0u;corner<8u;corner+=1u){
    if(weights[corner]<=0.0){continue;}
    let amount=-deltaRho*weights[corner]/total;
    atomicAdd(&sharpenDeposits[u32(indices[corner])],i32(round(amount*1048576.0)));
  }
}
@compute @workgroup_size(4,4,4)
fn sharpenResolve(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!validPacked(id)){return;}
  let index=u32(id.x+packedDims().x*(id.y+packedDims().y*id.z));
  var deposit=f32(atomicLoad(&sharpenDeposits[index]))/1048576.0;
  if(id.y==0){deposit/=f32(max(baseAt(id.x,id.z),1));}
  textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x+deposit));
}

@compute @workgroup_size(8,8,1)
fn planRemesh(@builtin(global_invocation_id) gid:vec3u){
  let d=packedDims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}let x=i32(gid.x);let z=i32(gid.y);let oldBase=baseAt(x,z);let layers=regularLayers();let fineY=fineDims().y;var lowest=fineY;var highest=-1;
	  // Paper Section 8 computes the surface bounds over the FULL column, not
	  // just the stored band. A settled surface inside the tall region (front
	  // water flooding a dry column, or a column draining below its band) is a
	  // crossing at the fill height; missing it leaves the free surface in the
	  // unsampled tall interior where nothing constrains divergence.
	  let fill=tallFillCells(x,z);let connected=tallConnectedToBand(x,z);
	  if(oldBase>0&&!connected&&fill>=0.5&&fill<f32(oldBase)-0.5){let interiorSurface=clamp(i32(round(fill)),0,fineY-1);lowest=min(lowest,interiorSurface);highest=max(highest,interiorSurface);}
	  var previous=select(0.0,select(tallStoreAlpha(x,z),1.0,connected),oldBase>0);
	  for(var packedY:i32=2;packedY<d.y;packedY+=1){let worldY=oldBase+packedY-2;if(worldY>=fineY){break;}let alpha=textureLoad(volumeIn,vec3i(x,packedY,z),0).x;let wet=alpha>=0.5;if(wet!=(previous>=0.5)){lowest=min(lowest,worldY);highest=max(highest,worldY);}previous=alpha;}
	if(previous>=0.5&&oldBase+layers<fineY){let bandTop=oldBase+layers-1;lowest=min(lowest,bandTop);highest=max(highest,bandTop);}
	  let h=params.cellGravity.xyz;let worldX=-0.5*params.container.x+(f32(x)+0.5)*h.x;let worldZ=-0.5*params.container.z+(f32(z)+0.5)*h.z;let bodyCount=u32(round(params.boundary.z));let maxBase=max(0,fineY-layers);var bodyUpper=maxBase;
	  let wetTop=columnHighestWetCell(x,z);
	  // Rigid geometry is not a free surface. It must not raise the band (and
	  // thereby turn the water below an approaching body into a taller store).
	  // It only caps the store below the body's predicted bottom, shortening a
	  // tall cell one step before an actual overlap. The existing surface band
	  // already represents the portion of an airborne body that can touch water.
	  for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let body=rigidBodies[bodyIndex];let radius=body.dimensions.w;if(abs(worldX-body.positionShape.x)<=radius&&abs(worldZ-body.positionShape.z)<=radius){let predictedBottom=i32(floor((body.positionShape.y-radius+min(0.0,body.linearVelocity.y)*params.dimsDt.w)/h.y));let nearWater=predictedBottom<=wetTop+1+i32(round(params.tall.z));if(predictedBottom<fineY&&nearWater){bodyUpper=min(bodyUpper,predictedBottom);}}}
	  // When the crossings span more than the band can cover, follow the top
	  // surface (paper Section 8's grid update). A tall cell below a wet band
	  // reads as liquid (tallConnectedToBand), so this no longer seals a
	  // phantom interface inside the tall cell.
	  var desired=oldBase;if(highest>=0){let surfaceLower=highest+1+i32(round(params.tall.z))-layers;let surfaceUpper=lowest+1-i32(round(params.tall.y));desired=select(surfaceLower,clamp(oldBase,surfaceLower,surfaceUpper),surfaceLower<=surfaceUpper);}desired=min(desired,bodyUpper);
	  // Paper Section 5 attributes the tall-cell volume/divergence error to the
	  // unmeasured interior lateral faces; the error scales with tall height
	  // times interior velocity. Tall cells therefore only persist where the
	  // interior is near-hydrostatic: a column whose endpoint dofs move faster
	  // than a small fraction of the CFL rail collapses to the ordinary-cell
	  // control height so the fine grid resolves the flow row by row, and it
	  // regrows once the deep water calms. This keeps the method inside the
	  // paper's operating envelope (calm deep water below an active band).
	  let storeSpeed=max(length(textureLoad(velocityIn,vec3i(x,0,z),0).xyz),length(textureLoad(velocityIn,vec3i(x,1,z),0).xyz));
	  // A submerged store that has drifted well below full is misrepresenting
	  // its column (the interface may not sit inside a tall cell, and the
	  // alpha-weighted transport keeps draining what the raw-velocity
	  // constraint cannot see). Shrink it as well.
	  let storeDeficient=connected&&tallStoreAlpha(x,z)<0.85;
	  // Descend GRADUALLY (two cells per remesh): an instantaneous drop to the
	  // control height fired simultaneously across a whole startup tank and
	  // the synchronized remap shock destroyed the equilibrium scenes. The
	  // threshold sits above hydrostatic startup waves (a few g*dt) and well
	  // below the churn this gate exists to shed.
	  // The gate sheds tall height but may never strip the Section 8 air halo
	  // above the column's own surface: an unconditional descent parks the
	  // closed band ceiling directly ON the water everywhere (no air buffer),
	  // so any rise jams into it and climbing water is silently squashed.
	  if(oldBase>0&&(storeSpeed>0.05*params.container.w||storeDeficient)){desired=min(desired,max(2,oldBase-2));desired=max(desired,wetTop+1+i32(round(params.tall.z))-layers);}
	  // Paper Section 8 has no per-step limit on how far the split may move;
	  // the neighbor bound D is enforced by the smoothing passes that follow.
	  // Rate-limiting descent left an advancing front inside tall interiors
	  // for many steps (see docs/TALL_CELL_STABILITY.md 2026-07-15 audit).
	  // A column may never take a base too low to represent its own water
	  // (conservative VOF cannot silently delete above-band liquid the way
	  // the paper's level set does); representability outranks the D bound.
	  desired=max(desired,i32(ceil(columnWaterCells(x,z)))-layers);
	  // The SURFACE must also stay inside the band: keep the highest wet cell
	  // below the ceiling with one air row above it so the interface can rise.
	  desired=max(desired,wetTop+2-layers);
	  desired=clamp(desired,0,maxBase);
  // Propose from the advected interface first. The neighbour constraint is
  // applied to this new field in separate Jacobi passes below.
  if(maxBase>=2){desired=max(2,desired);}nextColumnBases[u32(x+d.x*z)]=u32(max(0,desired));
}

@compute @workgroup_size(8,8,1)
fn smoothRemesh(@builtin(global_invocation_id) gid:vec3u){
  let d=packedDims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}let x=i32(gid.x);let z=i32(gid.y);let index=u32(x+d.x*z);let delta=u32(round(params.tall.w));var base=nextColumnBases[index];
  let offsets=array<vec2i,4>(vec2i(-1,0),vec2i(1,0),vec2i(0,-1),vec2i(0,1));
  for(var n=0u;n<4u;n+=1u){let q=vec2i(x,z)+offsets[n];if(q.x<0||q.x>=d.x||q.y<0||q.y>=d.z){continue;}base=min(base,nextColumnBases[u32(q.x+d.x*q.y)]+delta);}
  // Representability outranks the neighbor bound: lowering a base below the
  // column's own water content would strand liquid above the band ceiling.
  let floorBase=max(0,i32(ceil(columnWaterCells(x,z)))-regularLayers());base=max(base,u32(floorBase));
  let wetTopFloor=clamp(columnHighestWetCell(x,z)+2-regularLayers(),0,max(0,fineDims().y-regularLayers()));base=max(base,u32(wetTopFloor));
  if(fineDims().y-regularLayers()>=2&&base<2u){base=2u;}smoothedColumnBases[index]=base;
}

@compute @workgroup_size(4,4,4)
fn remap(@builtin(global_invocation_id) gid:vec3u){
	  let id=vec3i(gid);if(!validPacked(id)){return;}let d=packedDims();let newBase=i32(nextColumnBases[u32(id.x+d.x*id.z)]);var p=vec3f(f32(id.x)+0.5,0.5,f32(id.z)+0.5);if(id.y==1){p.y=max(0.5,f32(newBase)-0.5);}else if(id.y>=2){p.y=f32(newBase+id.y-2)+0.5;}let isActive=select(newBase>0,newBase+id.y-2<fineDims().y,id.y>=2);var alpha=0.0;var velocity=vec3f(0.0);if(isActive){
	    let oldBase=baseAt(id.x,id.z);var oldAmount=select(0.0,textureLoad(volumeIn,vec3i(id.x,0,id.z),0).x*f32(oldBase),oldBase>0);for(var packedY:i32=2;packedY<d.y;packedY+=1){let worldY=oldBase+packedY-2;if(worldY>=fineDims().y){break;}oldAmount+=textureLoad(volumeIn,vec3i(id.x,packedY,id.z),0).x;}
	    var regularAmount=0.0;for(var packedY:i32=2;packedY<d.y;packedY+=1){let worldY=newBase+packedY-2;if(worldY>=fineDims().y){break;}regularAmount+=volumeCell(vec3i(id.x,worldY,id.z));}
	    // The column integral is authoritative. When the new band would copy
	    // more than the column holds (residual < 0), scale the band copy down
	    // instead of clamping the bottom residual: the clamp silently destroys
	    // mass every time the base drops through partially settled water.
	    var bandScale=1.0;if(oldAmount<regularAmount&&regularAmount>1e-6){bandScale=oldAmount/regularAmount;}
	    let residual=max(0.0,oldAmount-regularAmount*bandScale);
	    // A residual beyond one full tall cell settles upward into the band's
	    // remaining capacity: volumeCell cannot see store excess above the
	    // base, so without this the overfull store persists indefinitely.
	    let overflow=max(0.0,residual-f32(newBase));
	    if(id.y==0){var capacity=0.0;if(overflow>0.0){for(var packedY:i32=2;packedY<d.y;packedY+=1){let worldY=newBase+packedY-2;if(worldY>=fineDims().y){break;}capacity+=1.0-min(1.0,volumeCell(vec3i(id.x,worldY,id.z))*bandScale);}}alpha=(residual-min(overflow,capacity))/f32(max(newBase,1));}
	    else if(id.y==1){for(var offset=0;offset<regularLayers();offset+=1){alpha=max(alpha,volumeCell(vec3i(id.x,newBase+offset,id.z)));}alpha*=bandScale;}
	    else{alpha=min(1.0,volumeCell(vec3i(id.x,newBase+id.y-2,id.z))*bandScale);
	      if(overflow>0.0){var used=0.0;for(var packedY:i32=2;packedY<id.y;packedY+=1){let worldY=newBase+packedY-2;if(worldY>=fineDims().y){break;}used+=1.0-min(1.0,volumeCell(vec3i(id.x,worldY,id.z))*bandScale);}alpha+=clamp(overflow-used,0.0,1.0-alpha);}}
    // Restrict the old velocity profile onto the endpoint dofs by sampling
    // the old reconstruction at the new endpoint sub-cells (the paper does a
    // least-squares fit; endpoint sampling is exact for the piecewise-linear
    // Eq 5 profile whenever the endpoints land inside the old tall cell).
    if(id.y<2&&newBase>0){velocity=velocityCell(vec3i(id.x,select(0,newBase-1,id.y==1),id.z));}else{velocity=sampleVelocity(p);}
  }textureStore(velocityOut,id,vec4f(velocity,0.0));textureStore(volumeOut,id,vec4f(alpha));textureStore(pressureOut,id,vec4f(0.0));if(id.y==0){textureStore(columnBaseOut,vec2i(id.x,id.z),vec4f(f32(newBase)));}
}

@compute @workgroup_size(4,4,4)
fn reduceDiagnostics(@builtin(global_invocation_id) gid:vec3u){
	  let id=vec3i(gid);if(!validPacked(id)||!activeSample(id)){return;}let alpha=textureLoad(volumeIn,id,0).x;let velocityValue=textureLoad(velocityIn,id,0).xyz;
  if(!finiteScalar(alpha)||!all(velocityValue==velocityValue)||any(abs(velocityValue)>vec3f(3.402823e38))){atomicAdd(&reductions[20],1u);return;}
	  var weight=1.0;if(id.y==0){weight=f32(baseAt(id.x,id.z));}else if(id.y==1){weight=0.0;}atomicAdd(&reductions[0],u32(clamp(alpha*weight*256.0,0.0,4294967295.0)));atomicAdd(&reductions[7],u32(clamp(alpha,0.0,1.0)*weight*256.0));atomicMax(&reductions[3],u32(baseAt(id.x,id.z)));
  if(pointSampleAlpha(id)>=0.5){
    atomicMax(&reductions[1],u32(id.x+1));updatePositiveMaximum(length(velocityValue),2u,8u,id);
	    if(solidFractionCell(vec3i(floor(samplePoint(id))))<=0.9){updatePositiveMaximum(abs(divergenceAt(id)),5u,14u,id);let h=params.cellGravity.xyz;let cfl=max(abs(velocityValue.x)*params.dimsDt.w/h.x,max(abs(velocityValue.y)*params.dimsDt.w/h.y,abs(velocityValue.z)*params.dimsDt.w/h.z));updatePositiveMaximumOnly(cfl,30u);if(cfl>1.0){atomicAdd(&reductions[31],1u);}}
  }
}
`;

// Kept in a separate module so the conservative-advection entry point sees a
// simple sampled texture and Metal never inlines the reconstruction into its
// deeply nested limiter graph.
export const tallCellTransportPreparationShader = /* wgsl */ `
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
  boundary: vec4f,
  tall: vec4f,
  inflowPositionRadius: vec4f,
  inflowVelocityLength: vec4f,
  inflowTiming: vec4f,
}
@group(0) @binding(0) var volumeIn:texture_3d<f32>;
@group(0) @binding(1) var columnBaseIn:texture_2d<f32>;
@group(0) @binding(2) var<uniform> params:Params;
@group(0) @binding(3) var transportVolumeOut:texture_storage_3d<r32float,write>;
fn packedDims()->vec3i{return vec3i(textureDimensions(volumeIn));}
fn fineDims()->vec3i{let d=packedDims();return vec3i(d.x,i32(round(params.boundary.w)),d.z);}
fn baseAt(x:i32,z:i32)->i32{let d=packedDims();if(x<0||x>=d.x||z<0||z>=d.z){return 0;}return i32(round(textureLoad(columnBaseIn,vec2i(x,z),0).x));}
fn validWorld(q:vec3i)->bool{let d=fineDims();return all(q>=vec3i(0))&&all(q<d);}
fn volumeCell(q:vec3i)->f32{
  if(!validWorld(q)){return 0.0;}let base=baseAt(q.x,q.z);
  if(q.y<base&&base>0){return clamp(textureLoad(volumeIn,vec3i(q.x,0,q.z),0).x,0.0,1.0);}
  let packedY=2+q.y-base;if(packedY<2||packedY>=packedDims().y){return 0.0;}return textureLoad(volumeIn,vec3i(q.x,packedY,q.z),0).x;
}
@compute @workgroup_size(4,4,4)
fn prepareTransportVolume(@builtin(global_invocation_id) gid:vec3u){let q=vec3i(gid);if(!validWorld(q)){return;}textureStore(transportVolumeOut,q,vec4f(volumeCell(q)));}
`;
