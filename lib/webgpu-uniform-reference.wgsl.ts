import { inflowBoundaryWGSL } from "./inflow-boundary";

/**
 * Dense uniform-grid reference kernels.
 *
 * This module is deliberately independent of both adaptive coarse backends.
 * It provides a matched-lattice GPU baseline for transport and projection
 * comparisons without octree topology, sparse residency, or backend cutovers.
 */
export const uniformReferenceComputeShader = /* wgsl */ `
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
  boundary: vec4f,
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
@group(0) @binding(7) var heightIn: texture_2d<f32>;
@group(0) @binding(8) var heightOut: texture_storage_2d<rg32float, write>;
// 0..3 are published diagnostics. 4..6 are transient, same-command-buffer
// volume-control totals (current, add capacity, remove capacity).
@group(0) @binding(9) var<storage,read_write> reductions:array<atomic<u32>,8>;
struct RigidBody {
  positionShape: vec4f,
  dimensions: vec4f,
  orientation: vec4f,
  linearVelocity: vec4f,
  angularVelocity: vec4f,
  inverseMassInertia: vec4f,
  angularMomentumRestitution: vec4f,
  material: vec4f,
}
@group(0) @binding(10) var<storage,read> rigidBodies:array<RigidBody,12>;
@group(0) @binding(11) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(0) @binding(12) var predictedVelocityIn: texture_3d<f32>;
@group(0) @binding(13) var reversedVelocityIn: texture_3d<f32>;
// Precomputed transport velocity with a one-texel zero shell so hardware
// trilinear sampling reproduces the zero wall-face boundary condition.
@group(0) @binding(14) var transportIn: texture_3d<f32>;
@group(0) @binding(15) var transportSampler: sampler;
@group(0) @binding(16) var transportOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(17) var fluxScalesIn: texture_3d<f32>;
@group(0) @binding(18) var fluxScalesOut: texture_storage_3d<rg32float, write>;
@group(0) @binding(19) var<storage,read_write> sharpenDeposits:array<atomic<i32>>;
// The adaptive method binds its resident signed-distance field here. Uniform
// reference solvers bind volumeIn instead, preserving their VOF formulation.
@group(0) @binding(20) var surfaceIn: texture_3d<f32>;
// Per-column terrain heights in cell units; params.container.w enables it so
// terrain-free scenes never pay the extra load. Static for the whole run.
@group(0) @binding(21) var terrainIn: texture_2d<f32>;
fn dims() -> vec3i { return vec3i(textureDimensions(volumeIn)); }
fn inflowGridDims()->vec3i{return dims();}
fn valid(p: vec3i) -> bool { let d=dims(); return all(p >= vec3i(0)) && all(p < d); }
fn clampCell(p: vec3i) -> vec3i { return clamp(p, vec3i(0), dims()-vec3i(1)); }
fn worldCell(id:vec3i)->vec3f{let h=params.cellGravity.xyz;return vec3f(-0.5*params.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*params.container.z+(f32(id.z)+0.5)*h.z);}
fn hasTerrain()->bool{return params.container.w>0.5;}
fn terrainHeightCells(x:i32,z:i32)->f32{let d=dims();return textureLoad(terrainIn,vec2i(clamp(x,0,d.x-1),clamp(z,0,d.z-1)),0).x;}
// Ground handling mirrors the rigid-body solid treatment with zero velocity:
// the heightfield closes faces, drops pressure unknowns, and blocks deposits.
fn cellInsideTerrain(p:vec3i)->bool{if(!hasTerrain()){return false;}return f32(p.y)+0.5<terrainHeightCells(p.x,p.z);}
fn cellTerrainFraction(p:vec3i)->f32{if(!hasTerrain()){return 0.0;}return clamp(terrainHeightCells(p.x,p.z)-f32(p.y),0.0,1.0);}
${inflowBoundaryWGSL}
fn volume(p: vec3i) -> f32 { if (!valid(p)) { return 0.0; } return textureLoad(volumeIn,p,0).x; }
fn transportConservativeVolume() -> bool { return params.physical.z > 0.5; }
fn levelSetAuthority() -> bool { return params.physical.w > 0.5; }
fn surfaceValue(p: vec3i) -> f32 {
  if (!valid(p)) { return select(0.0, 5.0 * min(params.cellGravity.x, min(params.cellGravity.y, params.cellGravity.z)), levelSetAuthority()); }
  return textureLoad(surfaceIn, p, 0).x;
}
fn surfaceOccupancy(p: vec3i) -> f32 {
  if (!valid(p)) { return 0.0; }
  let value = surfaceValue(p);
  return select(clamp(value, 0.0, 1.0), clamp(0.5 - value / (4.0 * params.cellGravity.y), 0.0, 1.0), levelSetAuthority());
}
fn surfaceLiquid(p: vec3i) -> bool { return valid(p) && select(surfaceValue(p) >= 0.5, surfaceValue(p) < 0.0, levelSetAuthority()); }
fn velocity(p: vec3i) -> vec3f { return textureLoad(velocityIn,clampCell(p),0).xyz; }
fn faceVelocity(p:vec3i)->vec3f{if(!valid(p)){return vec3f(0.0);}return textureLoad(velocityIn,p,0).xyz;}
fn liquid(p:vec3i)->bool{return surfaceLiquid(p);}
fn pressureValue(p:vec3i)->f32{return textureLoad(pressureIn,clampCell(p),0).x;}
fn transportVelocity(id:vec3i)->vec3f{
  var v=velocity(id);if(surfaceOccupancy(id)>=0.01){return v;}var sum=vec3f(0.0);var weight=0.0;
  let px=surfaceOccupancy(id+vec3i(1,0,0));let nx=surfaceOccupancy(id-vec3i(1,0,0));let py=surfaceOccupancy(id+vec3i(0,1,0));let ny=surfaceOccupancy(id-vec3i(0,1,0));let pz=surfaceOccupancy(id+vec3i(0,0,1));let nz=surfaceOccupancy(id-vec3i(0,0,1));
  sum+=velocity(id+vec3i(1,0,0))*px+velocity(id-vec3i(1,0,0))*nx+velocity(id+vec3i(0,1,0))*py+velocity(id-vec3i(0,1,0))*ny+velocity(id+vec3i(0,0,1))*pz+velocity(id-vec3i(0,0,1))*nz;weight=px+nx+py+ny+pz+nz;if(weight>0.001){v=sum/weight;}return v;
}
fn sampledFaceVelocity(p:vec3i,component:u32)->f32{
  let d=dims();if(p[component]<0||p[component]>=d[component]){return 0.0;}
  return textureLoad(transportIn,clampCell(p)+vec3i(1),0)[component];
}
fn transportCoordinate(q:vec3f)->vec3f{return (q+vec3f(1.5))/vec3f(dims()+vec3i(2));}
fn interfaceFraction(a:f32,b:f32)->f32{
  // Distance from the liquid cell centre to alpha=0.5 along a grid edge.
  return clamp((a-0.5)/max(abs(a-b),1e-6),0.05,1.0);
}
fn sampleVolume(p:vec3f)->f32{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(dims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);let c000=volume(b);let c100=volume(b+vec3i(1,0,0));let c010=volume(b+vec3i(0,1,0));let c110=volume(b+vec3i(1,1,0));let c001=volume(b+vec3i(0,0,1));let c101=volume(b+vec3i(1,0,1));let c011=volume(b+vec3i(0,1,1));let c111=volume(b+vec3i(1,1,1));return mix(mix(mix(c000,c100,f.x),mix(c010,c110,f.x),f.y),mix(mix(c001,c101,f.x),mix(c011,c111,f.x),f.y),f.z);
}
fn sampleVelocityComponent(p:vec3f,component:u32)->f32{
  var offset=vec3f(0.5);offset[component]=1.0;var lower=vec3f(0.0);lower[component]=-1.0;let q=clamp(p-offset,lower,vec3f(dims()-vec3i(1)));
  return textureSampleLevel(transportIn,transportSampler,transportCoordinate(q),0.0)[component];
}
fn sampleVelocity(p:vec3f)->vec3f{return vec3f(sampleVelocityComponent(p,0u),sampleVelocityComponent(p,1u),sampleVelocityComponent(p,2u));}
// One collocated vector fetch per RK2 stage; the half-texel stagger error only
// perturbs where the trace samples, not the sampled face values themselves.
fn transportVectorEstimate(p:vec3f)->vec3f{
  let q=clamp(p-vec3f(0.75),vec3f(-1.0),vec3f(dims()-vec3i(1)));
  return textureSampleLevel(transportIn,transportSampler,transportCoordinate(q),0.0).xyz;
}
fn departurePoint(position:vec3f,dt:f32,h:vec3f)->vec3f{let first=transportVectorEstimate(position);let midpoint=position-0.5*first*dt/h;return position-transportVectorEstimate(midpoint)*dt/h;}
fn advectVelocityComponent(position:vec3f,component:u32,dt:f32,h:vec3f)->f32{
  return sampleVelocityComponent(departurePoint(position,dt,h),component);
}
fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}
fn insideRigid(body:RigidBody,world:vec3f)->bool{
  let p=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
  if(shape==0){return length(p)<=d.x;}
  if(shape==1){return all(abs(p)<=0.5*d);}
  if(shape==2){let cy=clamp(p.y,-0.5*d.y,0.5*d.y);return length(vec3f(p.x,p.y-cy,p.z))<=d.x;}
  return p.x*p.x+p.z*p.z<=d.x*d.x&&abs(p.y)<=0.5*d.y;
}
fn rigidBodyIndexAt(world:vec3f)->i32{
  let bodyCount=u32(round(params.boundary.z));
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],world)){return i32(bodyIndex);}}
  return -1;
}
fn rigidVelocityAt(bodyIndex:i32,world:vec3f)->vec3f{
  let body=rigidBodies[u32(bodyIndex)];
  return body.linearVelocity.xyz+cross(body.angularVelocity.xyz,world-body.positionShape.xyz);
}
// Conservative bounding-sphere reject so cells away from every body (and
// body-free scenes) skip the per-cell primitive tests in the solid-aware
// pressure, projection, and coupling kernels.
fn nearAnyBody(world:vec3f)->bool{
  let bodyCount=u32(round(params.boundary.z));
  let margin=2.0*max(params.cellGravity.x,max(params.cellGravity.y,params.cellGravity.z));
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}
    let body=rigidBodies[bodyIndex];let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
    var radius=0.5*length(d);
    if(shape==0){radius=d.x;}
    if(shape==2){radius=d.x+0.5*d.y;}
    if(shape==3){radius=sqrt(d.x*d.x+0.25*d.y*d.y);}
    if(distance(world,body.positionShape.xyz)<=radius+margin){return true;}
  }
  return false;
}
// Paper Sec 3.9.1 treats a cell as solid in the divergence when its solid
// fraction is high; the cell-centre point-in-primitive test is our s>0.9.
fn cellRigidBody(p:vec3i)->i32{
  if(!valid(p)){return -1;}
  return rigidBodyIndexAt(worldCell(p));
}
// Sub-cell solid fraction with the CPU voxelizer's 8-corner sampling
// (solidFieldsFromBodies), so mixed cells blend rather than snap.
fn bodySolidFraction(body:RigidBody,p:vec3i)->f32{
  var inside=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
    if(insideRigid(body,worldCell(p)+offset*params.cellGravity.xyz)){inside+=1.0;}
  }
  return inside/8.0;
}
fn cellSolidFraction(p:vec3i)->f32{
  let bodyCount=u32(round(params.boundary.z));var fraction=0.0;
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}fraction=max(fraction,bodySolidFraction(rigidBodies[bodyIndex],p));}
  return fraction;
}
// Projection enforces body velocity on interior faces, so that value cannot be
// used as the undisturbed fluid velocity for form drag. Sample six wet, open
// points just beyond the body's bounding sphere instead.
fn ambientFluidVelocity(body:RigidBody,p:vec3i,fallback:vec3f)->vec3f{
  let h=params.cellGravity.xyz;let radius=max(body.dimensions.w,0.0);let reach=vec3i(ceil(vec3f(2.0*radius)/h))+vec3i(2);
  let offsets=array<vec3i,6>(vec3i(-reach.x,0,0),vec3i(reach.x,0,0),vec3i(0,-reach.y,0),vec3i(0,reach.y,0),vec3i(0,0,-reach.z),vec3i(0,0,reach.z));
  var total=vec3f(0.0);var weight=0.0;
  for(var n=0;n<6;n+=1){let q=p+offsets[n];if(!valid(q)||cellRigidBody(q)>=0||cellInsideTerrain(q)){continue;}let wet=surfaceOccupancy(q);total+=wet*velocity(q);weight+=wet;}
  return select(fallback,total/max(weight,1e-6),weight>0.0);
}
fn columnHeight(x:i32,z:i32)->f32{
  let d=dims();if(x<0||x>=d.x||z<0||z>=d.z){return 0.0;}return textureLoad(heightIn,vec2i(x,z),0).x;
}
fn upwind(face:f32,negative:f32,positive:f32)->f32{return face*select(positive,negative,face>=0.0);}
fn normalSurfaceOccupancy(id:vec3i)->f32{
  if(valid(id)){return surfaceOccupancy(id);}
  // Side walls and the floor are solids, so extend alpha with a zero-normal
  // gradient instead of inventing an air interface at the wall. Only an open
  // top (boundary.w) is allowed to expose liquid to exterior air.
  if(id.y>=dims().y&&params.boundary.w>0.5){return 0.0;}
  return surfaceOccupancy(clampCell(id));
}
fn surfaceGradient(id:vec3i)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(normalSurfaceOccupancy(id+vec3i(1,0,0))-normalSurfaceOccupancy(id-vec3i(1,0,0)),normalSurfaceOccupancy(id+vec3i(0,1,0))-normalSurfaceOccupancy(id-vec3i(0,1,0)),normalSurfaceOccupancy(id+vec3i(0,0,1))-normalSurfaceOccupancy(id-vec3i(0,0,1)))/(2.0*h);
}
fn interfaceNormal(id:vec3i)->vec3f{
  let gradient=surfaceGradient(id);
  return gradient/max(length(gradient),1e-6);
}
// The diagnostic/emergency VOF still sharpens along its own density gradient;
// this field does not classify the adaptive pressure or velocity solve.
fn normalVolume(id:vec3i)->f32{
  if(valid(id)){return volume(id);}
  if(id.y>=dims().y&&params.boundary.w>0.5){return 0.0;}
  return textureLoad(volumeIn,clampCell(id),0).x;
}
fn volumeGradient(id:vec3i)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(normalVolume(id+vec3i(1,0,0))-normalVolume(id-vec3i(1,0,0)),normalVolume(id+vec3i(0,1,0))-normalVolume(id-vec3i(0,1,0)),normalVolume(id+vec3i(0,0,1))-normalVolume(id-vec3i(0,0,1)))/(2.0*h);
}
fn rawVolumeFlux(id:vec3i,axis:u32,dt:f32)->f32{
  if(!valid(id)){return 0.0;}
  let neighbor=id+select(select(vec3i(0,0,1),vec3i(0,1,0),axis==1u),vec3i(1,0,0),axis==0u);
  let speed=faceVelocity(id)[axis];
  return dt/params.cellGravity.xyz[axis]*upwind(speed,volume(id),volume(neighbor));
}
fn outwardFlux(id:vec3i,dt:f32)->f32{
  if(!valid(id)){return 0.0;}
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  return max(rawVolumeFlux(id,0u,dt),0.0)+max(-rawVolumeFlux(id-ex,0u,dt),0.0)
       + max(rawVolumeFlux(id,1u,dt),0.0)+max(-rawVolumeFlux(id-ey,1u,dt),0.0)
       + max(rawVolumeFlux(id,2u,dt),0.0)+max(-rawVolumeFlux(id-ez,2u,dt),0.0);
}
fn inwardFlux(id:vec3i,dt:f32)->f32{
  if(!valid(id)){return 0.0;}
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  return max(-rawVolumeFlux(id,0u,dt),0.0)+max(rawVolumeFlux(id-ex,0u,dt),0.0)
       + max(-rawVolumeFlux(id,1u,dt),0.0)+max(rawVolumeFlux(id-ey,1u,dt),0.0)
       + max(-rawVolumeFlux(id,2u,dt),0.0)+max(rawVolumeFlux(id-ez,2u,dt),0.0);
}
fn donorScale(id:vec3i,dt:f32)->f32{return min(1.0,volume(id)/max(outwardFlux(id,dt),1e-9));}
fn receiverScale(id:vec3i,dt:f32)->f32{return min(1.0,max(0.0,1.0-volume(id))/max(inwardFlux(id,dt),1e-9));}
// Scales are precomputed once per cell by buildFluxScales; invalid neighbors
// keep the historical donor 0 / receiver 1 limits.
fn cellFluxScales(id:vec3i)->vec2f{if(!valid(id)){return vec2f(0.0,1.0);}return textureLoad(fluxScalesIn,id,0).xy;}
fn limitedVolumeFlux(id:vec3i,axis:u32,dt:f32)->f32{
  let offset=select(select(vec3i(0,0,1),vec3i(0,1,0),axis==1u),vec3i(1,0,0),axis==0u);
  let neighbor=id+offset;let flux=rawVolumeFlux(id,axis,dt);
  let donor=cellFluxScales(id);let receiver=cellFluxScales(neighbor);
  if(flux>=0.0){return flux*min(donor.x,receiver.y);}
  return flux*min(receiver.x,donor.y);
}
fn advectedVolume(id:vec3i,dt:f32)->f32{
  let centre=volume(id);
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  let fxp=limitedVolumeFlux(id,0u,dt);let fxm=limitedVolumeFlux(id-ex,0u,dt);
  let fyp=limitedVolumeFlux(id,1u,dt);let fym=limitedVolumeFlux(id-ey,1u,dt);
  let fzp=limitedVolumeFlux(id,2u,dt);let fzm=limitedVolumeFlux(id-ez,2u,dt);
  // No upper clamp on the transported value: a clamp here would destroy
  // sharpening deposits above one, which drain through the correction
  // divergence instead. The inflow source alone is bounded by the cell's
  // remaining capacity, as the old clamp did implicitly.
  let bounded=max(centre-(fxp-fxm+fyp-fym+fzp-fzm),0.0);
  return bounded+min(inflowReceiverSource(id,dt),max(0.0,1.0-bounded));
}

fn diffusionVelocity(p:vec3i)->vec3f{let v=textureLoad(velocityIn,clampCell(p),0).xyz;if(params.boundary.y>0.5&&!valid(p)){return -v;}return v;}
fn strainMagnitude(id:vec3i)->f32{
  let h=params.cellGravity.xyz;let dx=(diffusionVelocity(id+vec3i(1,0,0))-diffusionVelocity(id-vec3i(1,0,0)))/(2.0*h.x);let dy=(diffusionVelocity(id+vec3i(0,1,0))-diffusionVelocity(id-vec3i(0,1,0)))/(2.0*h.y);let dz=(diffusionVelocity(id+vec3i(0,0,1))-diffusionVelocity(id-vec3i(0,0,1)))/(2.0*h.z);let sxy=0.5*(dx.y+dy.x);let sxz=0.5*(dx.z+dz.x);let syz=0.5*(dy.z+dz.y);
  return sqrt(2.0*(dx.x*dx.x+dy.y*dy.y+dz.z*dz.z+2.0*(sxy*sxy+sxz*sxz+syz*syz)));
}
fn velocityLaplacian(id:vec3i)->vec3f{
  let h=params.cellGravity.xyz;let centre=diffusionVelocity(id);
  return (diffusionVelocity(id+vec3i(1,0,0))-2.0*centre+diffusionVelocity(id-vec3i(1,0,0)))/(h.x*h.x)+(diffusionVelocity(id+vec3i(0,1,0))-2.0*centre+diffusionVelocity(id-vec3i(0,1,0)))/(h.y*h.y)+(diffusionVelocity(id+vec3i(0,0,1))-2.0*centre+diffusionVelocity(id-vec3i(0,0,1)))/(h.z*h.z);
}

fn applyVelocityForces(id:vec3i,inputVelocity:vec3f,dt:f32,h:vec3f)->vec3f{
  var v=inputVelocity;let occupancy=surfaceOccupancy(id);if(occupancy>0.0){let molecular=params.physical.y/params.physical.x;v+=dt*molecular*velocityLaplacian(id);}
  // Body force lives on faces. A face participates whenever liquid exists on
  // either side; this is the same rule during impact and at equilibrium.
  let qy=id+vec3i(0,1,0);let yOccupancy=surfaceOccupancy(qy);
  let centerLiquid=select(occupancy>=0.5,occupancy>0.5,levelSetAuthority());
  let yLiquid=select(yOccupancy>=0.5,yOccupancy>0.5,levelSetAuthority());
  if(centerLiquid||yLiquid){v.y+=params.cellGravity.w*dt;}
  let qx=id+vec3i(1,0,0);let qz=id+vec3i(0,0,1);
  let xOccupancy=surfaceOccupancy(qx);let zOccupancy=surfaceOccupancy(qz);
  // Balanced-force CSF: pressure and capillary acceleration use the same
  // positive-face locations and alpha differences. Curvature is a deep
  // stencil, so evaluate the centre once and only on faces whose occupancy
  // difference can produce a non-zero force. The previous formulation
  // evaluated centre curvature three times and paid six curvature stencils in
  // every bulk cell even though the final multiplication was exactly zero.
  let sigmaOverRho=params.boundary.x/params.physical.x;
  if(sigmaOverRho>0.0){
    let dx=select(0.0,xOccupancy-occupancy,valid(qx));
    let dy=select(0.0,yOccupancy-occupancy,valid(qy));
    let dz=select(0.0,zOccupancy-occupancy,valid(qz));
    if(dx!=0.0||dy!=0.0||dz!=0.0){
      let centreCurvature=curvatureAt(id);
      if(dx!=0.0){v.x+=dt*sigmaOverRho*0.5*(centreCurvature+curvatureAt(qx))*dx/h.x;}
      if(dy!=0.0){v.y+=dt*sigmaOverRho*0.5*(centreCurvature+curvatureAt(qy))*dy/h.y;}
      if(dz!=0.0){v.z+=dt*sigmaOverRho*0.5*(centreCurvature+curvatureAt(qz))*dz/h.z;}
    }
  }
  v=applyInflowVelocity(id,v);let d=dims();if(id.x==d.x-1){v.x=0.0;}if(id.y==d.y-1){v.y=0.0;}if(id.z==d.z-1){v.z=0.0;}return v;
}

@compute @workgroup_size(4,4,4)
fn buildTransport(@builtin(global_invocation_id) gid:vec3u){
  let padded=vec3i(gid);let d=dims();let id=padded-vec3i(1);
  if(any(padded>=d+vec3i(2))){return;}
  if(!valid(id)){textureStore(transportOut,padded,vec4f(0.0));return;}
  textureStore(transportOut,padded,vec4f(transportVelocity(id),0.0));
}
@compute @workgroup_size(4,4,4)
fn buildFluxScales(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);
  if(!valid(id)){return;}let dt=params.dimsDt.w;
  textureStore(fluxScalesOut,id,vec4f(donorScale(id,dt),receiverScale(id,dt),0.0,0.0));
}
// Highest cell supported by the authoritative surface in each column;
// advection skips cells well above it after projection zeroes their faces.
@compute @workgroup_size(8,8,1)
fn buildOccupancy(@builtin(global_invocation_id) gid:vec3u){
  let d=dims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}
  var highest=-1.0;
  for(var y:i32=d.y-1;y>=0;y-=1){if(surfaceOccupancy(vec3i(i32(gid.x),y,i32(gid.y)))>0.0001){highest=f32(y);break;}}
  textureStore(heightOut,vec2i(gid.xy),vec4f(highest,-1.0,0.0,0.0));
}
fn nearInflow(id:vec3i)->bool{
  if(inflowStrength()<=0.0){return false;}
  let axis=inflowAxis();let face=inflowFaceIndex(axis);
  return id[axis]>=face-1&&id[axis]<=face+2&&inflowApertureFraction(id)>0.0;
}
fn aboveOccupancy(id:vec3i)->bool{
  let d=dims();var occupancy=-1.0;
  for(var dz:i32=-1;dz<=1;dz+=1){for(var dx:i32=-1;dx<=1;dx+=1){
    occupancy=max(occupancy,textureLoad(heightIn,vec2i(clamp(id.x+dx,0,d.x-1),clamp(id.z+dz,0,d.z-1)),0).x);
  }}
  return f32(id.y)>occupancy+4.0&&!nearInflow(id);
}
@compute @workgroup_size(4,4,4)
fn semiLagrangianAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  if(aboveOccupancy(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(volume(id),0.0,0.0,0.0));textureStore(pressureOut,id,vec4f(0.0));return;}
  var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,dt,h));v=applyVelocityForces(id,v,dt,h);
  var advected=volume(id);if(transportConservativeVolume()){advected=advectedVolume(id,dt);}textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(advected,0.0,0.0,0.0));textureStore(pressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn advect(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  if(aboveOccupancy(id)){textureStore(velocityOut,id,vec4f(0.0));textureStore(volumeOut,id,vec4f(volume(id),0.0,0.0,0.0));textureStore(pressureOut,id,vec4f(0.0));return;}
  let dt=params.dimsDt.w; let h=params.cellGravity.xyz;
  let cell=vec3f(id);var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,dt,h));
  var advected=volume(id);if(transportConservativeVolume()){advected=advectedVolume(id,dt);}let d=dims();
  if (id.x==d.x-1) { v.x=0.0; }
  if (id.y==d.y-1) { v.y=0.0; }
  if (id.z==d.z-1) { v.z=0.0; }
  textureStore(velocityOut,id,vec4f(v,0.0));
  textureStore(volumeOut,id,vec4f(advected,0.0,0.0,0.0));
  textureStore(pressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn reverseAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  if(aboveOccupancy(id)){textureStore(velocityOut,id,vec4f(0.0));return;}
  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,-dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,-dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,-dt,h));let d=dims();
  if(id.x==d.x-1){v.x=0.0;}if(id.y==d.y-1){v.y=0.0;}if(id.z==d.z-1){v.z=0.0;}textureStore(velocityOut,id,vec4f(v,0.0));
}

fn boundedMacCormack(id:vec3i,position:vec3f,component:u32,dt:f32,h:vec3f,predicted:f32,original:f32,reversed:f32)->f32{
  var offset=vec3f(0.5);offset[component]=1.0;var lowerCoordinate=vec3f(0.0);lowerCoordinate[component]=-1.0;
  let q=clamp(departurePoint(position,dt,h)-offset,lowerCoordinate,vec3f(dims()-vec3i(1)));let b=vec3i(floor(q));
  var lower=sampledFaceVelocity(b,component);var upper=lower;
  for(var corner:u32=1u;corner<8u;corner+=1u){let cornerOffset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let value=sampledFaceVelocity(b+cornerOffset,component);lower=min(lower,value);upper=max(upper,value);}
  let corrected=predicted+0.5*(original-reversed);
  return select(corrected,predicted,corrected<lower||corrected>upper);
}

@compute @workgroup_size(4,4,4)
fn correctAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  if(aboveOccupancy(id)){textureStore(velocityOut,id,vec4f(0.0));return;}
  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  let predicted=textureLoad(predictedVelocityIn,id,0).xyz;let original=textureLoad(velocityIn,id,0).xyz;let reversed=textureLoad(reversedVelocityIn,id,0).xyz;
  var v=vec3f(boundedMacCormack(id,cell+vec3f(1.0,0.5,0.5),0u,dt,h,predicted.x,original.x,reversed.x),boundedMacCormack(id,cell+vec3f(0.5,1.0,0.5),1u,dt,h,predicted.y,original.y,reversed.y),boundedMacCormack(id,cell+vec3f(0.5,0.5,1.0),2u,dt,h,predicted.z,original.z,reversed.z));v=applyVelocityForces(id,v,dt,h);
  textureStore(velocityOut,id,vec4f(v,0.0));
}

@compute @workgroup_size(8,8,1)
fn buildHeight(@builtin(global_invocation_id) gid:vec3u){let d=dims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}var total=0.0;for(var y:i32=0;y<d.y;y+=1){total+=volume(vec3i(i32(gid.x),y,i32(gid.y)))*params.cellGravity.y;}textureStore(heightOut,vec2i(gid.xy),vec4f(total));}

fn faceWorld(id:vec3i,axis:u32)->vec3f{
  var world=worldCell(id);world[axis]+=0.5*params.cellGravity.xyz[axis];return world;
}
// Positive-side face velocity with the paper's VOS constraint (Sec 3.9.1): a
// face touching a rigid-solid cell carries the solid velocity, which is what
// makes a moving body sweep water out of its path instead of ignoring it.
fn constrainedFaceVelocity(id:vec3i,axis:u32,checkSolid:bool)->f32{
  var neighbor=id;neighbor[axis]+=1;
  // The terrain heightfield is a static solid: a face touching it carries the
  // ground's zero velocity in the divergence, exactly like a wall.
  if(cellInsideTerrain(id)||cellInsideTerrain(neighbor)){return 0.0;}
  if(checkSolid){
    let body=max(cellRigidBody(id),cellRigidBody(neighbor));
    if(body>=0){return rigidVelocityAt(body,faceWorld(id,axis))[axis];}
  }
  return faceVelocity(id)[axis];
}
fn divergenceAt(id: vec3i, checkSolid: bool) -> f32 {
  let h=params.cellGravity.xyz;
  return (constrainedFaceVelocity(id,0u,checkSolid)-constrainedFaceVelocity(id-vec3i(1,0,0),0u,checkSolid))/h.x
       + (constrainedFaceVelocity(id,1u,checkSolid)-constrainedFaceVelocity(id-vec3i(0,1,0),1u,checkSolid))/h.y
       + (constrainedFaceVelocity(id,2u,checkSolid)-constrainedFaceVelocity(id-vec3i(0,0,1),2u,checkSolid))/h.z;
}
// Mass-Conserving Eulerian Liquid Simulation Sec 3.7: cells holding more
// density than they represent add min(lambda (rho'-1), eta) artificial
// divergence (lambda = 0.5, eta = 1 per the paper, expressed as a rate
// against its 1/30 s step) so the pressure solve pushes the excess out.
fn volumeCorrectionDivergence(id: vec3i) -> f32 {
  let excess=max(0.0,volume(id)-1.0);
  if(excess<=0.0){return 0.0;}
  return min(0.5*excess,1.0)*30.0;
}

fn curvatureAt(id:vec3i)->f32{
  let h=params.cellGravity.xyz;
  return -((interfaceNormal(id+vec3i(1,0,0)).x-interfaceNormal(id-vec3i(1,0,0)).x)/(2.0*h.x)+(interfaceNormal(id+vec3i(0,1,0)).y-interfaceNormal(id-vec3i(0,1,0)).y)/(2.0*h.y)+(interfaceNormal(id+vec3i(0,0,1)).z-interfaceNormal(id-vec3i(0,0,1)).z)/(2.0*h.z));
}

fn stencilCoefficient(id:vec3i,neighbor:vec3i,axis:u32,checkSolid:bool)->f32{
  if(!valid(neighbor)){return 0.0;}
  // A rigid-solid or terrain neighbor is a Neumann boundary exactly like a
  // wall; its motion enters through the divergence, not the stencil.
  if(cellInsideTerrain(neighbor)){return 0.0;}
  if(checkSolid&&cellRigidBody(neighbor)>=0){return 0.0;}
  let h=params.cellGravity.xyz[axis];
  if(liquid(neighbor)){return 1.0/(h*h);}
  return 1.0/(interfaceFraction(volume(id),volume(neighbor))*h*h);
}

fn stencilPressure(id:vec3i,neighbor:vec3i,axis:u32,checkSolid:bool)->f32{
  if(!valid(neighbor)||!liquid(neighbor)){return 0.0;}
  return stencilCoefficient(id,neighbor,axis,checkSolid)*pressureValue(neighbor);
}

@compute @workgroup_size(4,4,4)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  if (!liquid(id)) { textureStore(pressureOut,id,vec4f(0.0)); return; }
  // Ground cells are solid, not pressure unknowns, like body interiors below.
  if(cellInsideTerrain(id)){textureStore(pressureOut,id,vec4f(0.0));return;}
  let checkSolid=nearAnyBody(worldCell(id));
  // Paper Sec 3.9.1: cells occupied by a rigid body are solid, not pressure
  // unknowns. Without this the sphere interior stays "water" and the solve
  // never displaces it.
  if(checkSolid&&cellRigidBody(id)>=0){textureStore(pressureOut,id,vec4f(0.0));return;}
  let old=textureLoad(pressureIn,id,0).x;let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  let diagonal=stencilCoefficient(id,id-ex,0u,checkSolid)+stencilCoefficient(id,id+ex,0u,checkSolid)+stencilCoefficient(id,id-ey,1u,checkSolid)+stencilCoefficient(id,id+ey,1u,checkSolid)+stencilCoefficient(id,id-ez,2u,checkSolid)+stencilCoefficient(id,id+ez,2u,checkSolid);
  let sum=stencilPressure(id,id-ex,0u,checkSolid)+stencilPressure(id,id+ex,0u,checkSolid)+stencilPressure(id,id-ey,1u,checkSolid)+stencilPressure(id,id+ey,1u,checkSolid)+stencilPressure(id,id-ez,2u,checkSolid)+stencilPressure(id,id+ez,2u,checkSolid);
  // Subtracted so the projection leaves div_new = +c at overfull cells (an
  // outward drain); added it would leave div_new = -c and feed the excess.
  let rhs=params.physical.x*(divergenceAt(id,checkSolid)-volumeCorrectionDivergence(id))/params.dimsDt.w;
  // A liquid cell sealed on all six sides (tight body/wall gap) has no
  // stencil; leave it unconstrained instead of dividing by epsilon.
  if(diagonal<=0.0){textureStore(pressureOut,id,vec4f(0.0));return;}
  let next=(sum-rhs)/max(diagonal,1e-9);
  textureStore(pressureOut,id,vec4f(mix(old,next,0.8),0.0,0.0,0.0));
}

@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  let h=params.cellGravity.xyz;let scale=params.dimsDt.w/params.physical.x;var v=velocity(id);let d=dims();
  let p0=select(0.0,pressureValue(id),liquid(id));
  let ex=id+vec3i(1,0,0);let ey=id+vec3i(0,1,0);let ez=id+vec3i(0,0,1);
  if(id.x==d.x-1){v.x=0.0;}else if(liquid(id)||liquid(ex)){let p1=select(0.0,pressureValue(ex),liquid(ex));let theta=select(interfaceFraction(volume(ex),volume(id)),interfaceFraction(volume(id),volume(ex)),liquid(id));v.x-=scale*(p1-p0)/(h.x*select(theta,1.0,liquid(id)&&liquid(ex)));}else{v.x=0.0;}
  if(id.y==d.y-1){v.y=0.0;}else if(liquid(id)||liquid(ey)){let p1=select(0.0,pressureValue(ey),liquid(ey));let theta=select(interfaceFraction(volume(ey),volume(id)),interfaceFraction(volume(id),volume(ey)),liquid(id));v.y-=scale*(p1-p0)/(h.y*select(theta,1.0,liquid(id)&&liquid(ey)));}else{v.y=0.0;}
  if(id.z==d.z-1){v.z=0.0;}else if(liquid(id)||liquid(ez)){let p1=select(0.0,pressureValue(ez),liquid(ez));let theta=select(interfaceFraction(volume(ez),volume(id)),interfaceFraction(volume(id),volume(ez)),liquid(id));v.z-=scale*(p1-p0)/(h.z*select(theta,1.0,liquid(id)&&liquid(ez)));}else{v.z=0.0;}
  // Faces the terrain heightfield covers are no-flux ground, like the floor.
  if(hasTerrain()){
    if(cellInsideTerrain(id)||cellInsideTerrain(ex)){v.x=0.0;}
    if(cellInsideTerrain(id)||cellInsideTerrain(ey)){v.y=0.0;}
    if(cellInsideTerrain(id)||cellInsideTerrain(ez)){v.z=0.0;}
  }
  // Faces covered by a rigid body move with the body (paper Sec 3.9.1); the
  // VOF fluxes then transport volume out of the body's path. Domain-edge
  // faces stay walls.
  if(nearAnyBody(worldCell(id))){
    let bodyX=max(cellRigidBody(id),cellRigidBody(ex));
    let bodyY=max(cellRigidBody(id),cellRigidBody(ey));
    let bodyZ=max(cellRigidBody(id),cellRigidBody(ez));
    if(bodyX>=0&&id.x<d.x-1){v.x=rigidVelocityAt(bodyX,faceWorld(id,0u)).x;}
    if(bodyY>=0&&id.y<d.y-1){v.y=rigidVelocityAt(bodyY,faceWorld(id,1u)).y;}
    if(bodyZ>=0&&id.z<d.z-1){v.z=rigidVelocityAt(bodyZ,faceWorld(id,2u)).z;}
  }
  v=applyInflowVelocity(id,v);textureStore(velocityOut,id,vec4f(v,0.0)); textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));
}

// Brinkman-style immersed boundary: drive wet cells inside each moving primitive
// toward the local solid velocity and accumulate the exact opposite impulse.
@compute @workgroup_size(4,4,4)
fn coupleRigid(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}let phi=volume(id);let wetFraction=surfaceOccupancy(id);var v=velocity(id);let h=params.cellGravity.xyz;
  let world=vec3f(-0.5*params.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*params.container.z+(f32(id.z)+0.5)*h.z);
  let bodyCount=u32(round(params.boundary.z));let cellMass=params.physical.x*h.x*h.y*h.z*wetFraction;let blend=clamp(45.0*params.dimsDt.w,0.0,1.0);var coupledBody=12u;var solidFraction=0.0;
  // Match the adaptive voxelizer's overlap rule: the body with the greatest
  // sub-cell coverage owns this cell, so displaced volume is never counted
  // twice and does not depend on body-array order.
  for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let candidate=bodySolidFraction(rigidBodies[bodyIndex],id);if(candidate>solidFraction){solidFraction=candidate;coupledBody=bodyIndex;}}
  if(coupledBody<12u){
    let bodyIndex=coupledBody;let body=rigidBodies[bodyIndex];
    let arm=world-body.positionShape.xyz;let solidVelocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);let fluidVelocity=v;let ambientVelocity=ambientFluidVelocity(body,id,fluidVelocity);let fluidImpulse=cellMass*solidFraction*(solidVelocity-fluidVelocity)*blend;v+=fluidImpulse/max(cellMass,1e-9);
    let reaction=-fluidImpulse;let torque=cross(arm,reaction);let base=bodyIndex*12u;
    atomicAdd(&rigidExchange[base],i32(round(reaction.x*1000000.0)));atomicAdd(&rigidExchange[base+1u],i32(round(reaction.y*1000000.0)));atomicAdd(&rigidExchange[base+2u],i32(round(reaction.z*1000000.0)));
    atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*1000000.0)));atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*1000000.0)));atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*1000000.0)));
    let displacedWeight=wetFraction*solidFraction;
    atomicAdd(&rigidExchange[base+6u],i32(round(displacedWeight*65536.0)));
    atomicAdd(&rigidExchange[base+7u],i32(round(displacedWeight*ambientVelocity.x*10000.0)));atomicAdd(&rigidExchange[base+8u],i32(round(displacedWeight*ambientVelocity.y*10000.0)));atomicAdd(&rigidExchange[base+9u],i32(round(displacedWeight*ambientVelocity.z*10000.0)));
  }
  // Paper Sec 3.9.1 phi-s: inside a body the advected field is meaningless, so
  // blend it toward the (1-s)-weighted neighbor average. This is what lets the
  // body displace its water column instead of sealing a phantom plug of
  // liquid inside and carrying it around.
  var phiNext=phi;
  if(nearAnyBody(world)){
    let s=cellSolidFraction(id);
    if(s>0.0){
      var open=0.0;var openSum=0.0;var total=0.0;
      let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
      for(var index=0;index<6;index+=1){
        let np=clampCell(id+offsets[index]);
        let neighborVolume=volume(np);let neighborOpen=(1.0-cellSolidFraction(np))*(1.0-cellTerrainFraction(np));total+=neighborVolume;
        open+=neighborOpen;openSum+=neighborOpen*neighborVolume;
      }
      // A one-cell stencil diffuses a carried interior plug over several body
      // radii. Direct lateral open samples preserve the same local phi-s target
      // while making it follow a fast body through the interface in one step.
      if(coupledBody<12u&&i32(round(rigidBodies[coupledBody].positionShape.w))==0&&length(rigidBodies[coupledBody].linearVelocity.xyz)>0.25){
        let radius=max(rigidBodies[coupledBody].dimensions.w,0.0);let reach=vec3i(ceil(vec3f(2.0*radius)/h))+vec3i(2);
        let far=array<vec3i,4>(vec3i(-reach.x,0,0),vec3i(reach.x,0,0),vec3i(0,0,-reach.z),vec3i(0,0,reach.z));
        for(var index=0;index<4;index+=1){let np=id+far[index];if(valid(np)){let neighborOpen=(1.0-cellSolidFraction(np))*(1.0-cellTerrainFraction(np));open+=neighborOpen;openSum+=neighborOpen*volume(np);}}
      }
      let relaxTarget=select(total/6.0,openSum/max(open,1.0),open>0.0);
      // This is a physical-time relaxation, not a per-dispatch blend. Using s
      // directly made the same simulated second displace far more VOF when it
      // was divided into smaller steps (the visible volume-loss symptom).
      phiNext=mix(phi,relaxTarget,s*blend);
    }
  }
  // The nozzle mouth is an open boundary. Coupling the visual nozzle body
  // must not replace the prescribed reservoir velocity at that opening.
  v=applyInflowVelocity(id,v);
  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(phiNext));
}

// Paper Sec 3.9.1 phi-s for the resident adaptive level set. While an adaptive
// projection owns the pressure solve the uniform pressure textures are idle,
// so this pass aliases them: pressureIn is a copy of the signed-distance field
// and pressureOut is the resident level-set texture itself.
@compute @workgroup_size(4,4,4)
fn relaxSolidPhi(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let phi=textureLoad(pressureIn,id,0).x;
  var result=phi;
  if(nearAnyBody(worldCell(id))){
    let s=cellSolidFraction(id);
    if(s>0.0){
      var open=0.0;var openSum=0.0;var total=0.0;var exteriorOpen=0.0;var exteriorSum=0.0;
      let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
      for(var index=0;index<6;index+=1){
        let np=clampCell(id+offsets[index]);
        let neighborPhi=textureLoad(pressureIn,np,0).x;let neighborOpen=(1.0-cellSolidFraction(np))*(1.0-cellTerrainFraction(np));total+=neighborPhi;
        open+=neighborOpen;openSum+=neighborOpen*neighborPhi;
        // Extend phi from the first genuinely open sample on each coordinate
        // ray. A one-cell relaxation takes many frames to cross a large solid
        // and leaves a newly submerged body falsely dry, under-reporting its
        // displaced volume. Six exterior samples establish the correct phase
        // throughout the solid in this pass while retaining a local fallback
        // for bodies wider than the bounded search.
        for(var step=1;step<=64;step+=1){
          let exterior=id+step*offsets[index];if(!valid(exterior)){break;}
          let exteriorWeight=(1.0-cellSolidFraction(exterior))*(1.0-cellTerrainFraction(exterior));
          if(exteriorWeight>0.5){exteriorOpen+=exteriorWeight;exteriorSum+=exteriorWeight*textureLoad(pressureIn,exterior,0).x;break;}
        }
      }
      let localTarget=select(total/6.0,openSum/max(open,1.0),open>0.0);
      let relaxTarget=select(localTarget,exteriorSum/max(exteriorOpen,1.0),exteriorOpen>0.0);
      result=mix(phi,relaxTarget,s);
    }
  }
  textureStore(pressureOut,id,vec4f(result));
}

// --- Density sharpening (Mass-Conserving Eulerian Liquid Simulation Sec 3.5,
// Eq 4-17 and Algorithm 2; docs/TALL_CELLS_PAPER.md Appendix B.3). Pass 1
// applies the local correction (Eq 17 keeps it non-positive: mass only moves
// from the air side to the liquid side); pass 2 returns the removed mass by
// tracing along the density gradient to the 0.5 iso-contour and depositing
// fixed-point trilinear weights; pass 3 folds the deposits back in.
fn cellInsideSolid(p:vec3i)->bool{
  if(cellInsideTerrain(p)){return true;}
  let bodyCount=u32(round(params.boundary.z));if(bodyCount==0u){return false;}
  let world=worldCell(p);
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],world)){return true;}}
  return false;
}
fn sharpenDeltaRho(q:vec3i)->f32{
  let rho=volume(q);
  if(cellInsideSolid(q)){return 0.0;}
  let h=params.cellGravity.xyz;let deltaT=3.0*params.dimsDt.w;let tau=0.4;
  let sxp=-(rho-volume(q-vec3i(1,0,0)))*deltaT/h.x;let sxm=-(volume(q+vec3i(1,0,0))-rho)*deltaT/h.x;
  let syp=-(rho-volume(q-vec3i(0,1,0)))*deltaT/h.y;let sym=-(volume(q+vec3i(0,1,0))-rho)*deltaT/h.y;
  let szp=-(rho-volume(q-vec3i(0,0,1)))*deltaT/h.z;let szm=-(volume(q+vec3i(0,0,1))-rho)*deltaT/h.z;
  let gradPlus=sqrt(max(max(sxp,0.0)*max(sxp,0.0),min(sxm,0.0)*min(sxm,0.0))+max(max(syp,0.0)*max(syp,0.0),min(sym,0.0)*min(sym,0.0))+max(max(szp,0.0)*max(szp,0.0),min(szm,0.0)*min(szm,0.0)));
  let gradMinus=sqrt(max(min(sxp,0.0)*min(sxp,0.0),max(sxm,0.0)*max(sxm,0.0))+max(min(syp,0.0)*min(syp,0.0),max(sym,0.0)*max(sym,0.0))+max(min(szp,0.0)*min(szp,0.0),max(szm,0.0)*max(szm,0.0)));
  var maximumDifference=0.0;
  let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var index=0;index<6;index+=1){maximumDifference=max(maximumDifference,abs(rho-volume(q+offsets[index])));}
  let weight=(rho-0.5)*(rho-0.5)*(rho-0.5)*(1.0-min(1.0,maximumDifference/tau));
  var deltaRho=select(weight*gradMinus,weight*gradPlus,weight>=0.0);
  if(rho+deltaRho<0.0||rho<1e-5){deltaRho=-rho;}else if(rho>0.5){deltaRho=0.0;}
  return deltaRho;
}
@compute @workgroup_size(4,4,4)
fn sharpenCompute(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let stored=textureLoad(volumeIn,id,0).x;
  let deltaRho=sharpenDeltaRho(id);
  textureStore(volumeOut,id,vec4f(stored+deltaRho));
  textureStore(pressureOut,id,vec4f(deltaRho));
}
@compute @workgroup_size(4,4,4)
fn sharpenScatter(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let deltaRho=textureLoad(pressureIn,id,0).x;if(deltaRho>=0.0){return;}
  var p=vec3f(id)+vec3f(0.5);let maximumDistance=2.1;var travelled=0.0;let stepLength=0.5;
  for(var stepIndex=0;stepIndex<5;stepIndex+=1){
    if(sampleVolume(p)>=0.5||travelled>=maximumDistance){break;}
    let g=volumeGradient(vec3i(floor(p)));let magnitude=length(g);
    if(magnitude<1e-6){break;}
    let candidate=p+g/magnitude*stepLength;
    if(cellInsideSolid(vec3i(floor(candidate)))){break;}
    p=candidate;travelled+=stepLength;
  }
  // The paper assumes the 0.5 iso-contour lies within D cells of every
  // sharpened cell. In diffused low-density regions no contour exists
  // nearby, and depositing at the trace end concentrates fog at its local
  // maxima until free-floating droplets nucleate above the water. When the
  // trace fails to reach liquid, return the mass to its own cell instead.
  if(sampleVolume(p)<0.5){
    let dd=dims();let ownIndex=id.x+dd.x*(id.y+dd.y*id.z);
    atomicAdd(&sharpenDeposits[u32(ownIndex)],i32(round(-deltaRho*1048576.0)));return;
  }
  let anchor=p-vec3f(0.5);let cell=vec3i(floor(anchor));let f=fract(anchor);
  var weights=array<f32,8>();var indices=array<i32,8>();var total=0.0;
  let d=dims();
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let destination=cell+offset;
    var w=select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);
    var index=-1;
    if(valid(destination)&&!cellInsideSolid(destination)){index=destination.x+d.x*(destination.y+d.y*destination.z);}else{w=0.0;}
    // Corners without remaining capacity are skipped so deposits cannot push
    // a cell past one, where the advection clamp would destroy the excess;
    // any residual overshoot drains through the correction divergence below.
    if(w>0.0&&volume(destination)>=1.0){w=0.0;}
    weights[corner]=w;indices[corner]=index;total+=w;
  }
  if(total<=1e-8){
    let ownIndex=id.x+d.x*(id.y+d.y*id.z);
    atomicAdd(&sharpenDeposits[u32(ownIndex)],i32(round(-deltaRho*1048576.0)));return;
  }
  for(var corner=0u;corner<8u;corner+=1u){
    if(weights[corner]<=0.0){continue;}
    atomicAdd(&sharpenDeposits[u32(indices[corner])],i32(round(-deltaRho*weights[corner]/total*1048576.0)));
  }
}
@compute @workgroup_size(4,4,4)
fn sharpenResolve(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let d=dims();let index=u32(id.x+d.x*(id.y+d.y*id.z));
  let deposit=f32(atomicLoad(&sharpenDeposits[index]))/1048576.0;
  textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x+deposit));
}

// A cell belongs to the correction band when it is fractional or shares a
// face with the opposite phase. Including the adjacent full/empty cell makes
// the controller useful even when the initial VOF is perfectly binary.
fn volumeCorrectionBand(id:vec3i)->bool{
  if(!valid(id)||cellInsideSolid(id)){return false;}
  let alpha=clamp(volume(id),0.0,1.0);
  if(alpha>0.001&&alpha<0.999){return true;}
  let phase=alpha>=0.5;
  let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var index=0;index<6;index+=1){
    let neighbor=id+offsets[index];
    if(valid(neighbor)&&!cellInsideSolid(neighbor)&&(clamp(volume(neighbor),0.0,1.0)>=0.5)!=phase){return true;}
  }
  return false;
}

@compute @workgroup_size(4,4,4)
fn measureVolumeCorrection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let alpha=clamp(volume(id),0.0,1.0);
  atomicAdd(&reductions[4],u32(alpha*2048.0+0.5));
  if(volumeCorrectionBand(id)){
    atomicAdd(&reductions[5],u32((1.0-alpha)*2048.0+0.5));
    atomicAdd(&reductions[6],u32(alpha*2048.0+0.5));
  }
}

@compute @workgroup_size(4,4,4)
fn applyVolumeCorrection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let stored=textureLoad(volumeIn,id,0).x;
  if(!volumeCorrectionBand(id)){textureStore(volumeOut,id,vec4f(stored));return;}
  let alpha=clamp(stored,0.0,1.0);
  let current=f32(atomicLoad(&reductions[4]))/2048.0;
  let error=params.inflowTiming.y-current;
  let capacityWord=select(atomicLoad(&reductions[6]),atomicLoad(&reductions[5]),error>=0.0);
  let capacity=f32(capacityWord)/2048.0;
  if(capacity<=1e-6||abs(error)<=1.0/2048.0){textureStore(volumeOut,id,vec4f(stored));return;}
  // Exponential response is invariant to subdivision of simulated time. The
  // second bound limits motion of this one-cell interface band to two cells/s
  // (and never more than a quarter cell in one dispatch).
  let response=1.0-exp(-params.inflowTiming.w*params.dimsDt.w);
  let maximumFraction=min(0.25,2.0*params.dimsDt.w);
  let fraction=min(abs(error)*response/capacity,maximumFraction);
  let localCapacity=select(alpha,1.0-alpha,error>=0.0);
  let corrected=clamp(alpha+select(-1.0,1.0,error>=0.0)*fraction*localCapacity,0.0,1.0);
  textureStore(volumeOut,id,vec4f(corrected));
}

// Render-only reconstruction. A separable [1 2 1]^3 kernel converts the
// cell-centred binary VOF into a continuous fractional field. Retaining one
// quarter of the original conservative value keeps a one-cell sheet above the
// renderer's 0.5 contour: two unguided passes would reduce its peak to 0.375.
// The host never binds this result back into transport or projection.
fn presentationSample(id:vec3i)->vec2f{
  if(!valid(id)){return vec2f(0.0);}
  return vec2f(clamp(volume(id),0.0,1.0),1.0);
}
@compute @workgroup_size(4,4,4)
fn smoothSurface(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  var weighted=0.0;var weights=0.0;
  for(var z=-1;z<=1;z+=1){for(var y=-1;y<=1;y+=1){for(var x=-1;x<=1;x+=1){
    let sample=presentationSample(id+vec3i(x,y,z));
    let weight=select(1.0,2.0,x==0)*select(1.0,2.0,y==0)*select(1.0,2.0,z==0);
    weighted+=weight*sample.x;weights+=weight*sample.y;
  }}}
  let filtered=weighted/max(weights,1.0);
  let conservative=clamp(textureLoad(surfaceIn,id,0).x,0.0,1.0);
  textureStore(volumeOut,id,vec4f(mix(filtered,conservative,0.25)));
}
@compute @workgroup_size(4,4,4)
fn reduceDiagnostics(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!valid(id)){return;}let open=(1.0-cellSolidFraction(id))*(1.0-cellTerrainFraction(id));let represented=surfaceOccupancy(id)*open;let conservative=volume(id)*open;atomicAdd(&reductions[0],u32(represented*2048.0+0.5));if(surfaceLiquid(id)){atomicMax(&reductions[1],u32(id.x+1));}let speed=length(faceVelocity(id));atomicMax(&reductions[2],bitcast<u32>(speed));atomicAdd(&reductions[3],u32(clamp(conservative,0.0,8.0)*2048.0+0.5));}
`;
