import { inflowBoundaryWGSL } from "./inflow-boundary";

const uniformMacCormackAuditEnabled = typeof process !== "undefined"
  && process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT === "1";

/**
 * Dense uniform-grid reference kernels.
 *
 * This module is deliberately independent of both adaptive coarse backends.
 * It provides a matched-lattice GPU baseline for transport and projection
 * comparisons without octree topology, sparse residency, or backend cutovers.
 */
export const uniformReferenceComputeShader = /* wgsl */ `
const MACCORMACK_AUDIT_ENABLED: bool = ${uniformMacCormackAuditEnabled};
const GHOST_FLUID_THETA_MIN:f32=0.05;
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
  boundary: vec4f,
  inflowPositionRadius: vec4f,
  inflowVelocityLength: vec4f,
  inflowTiming: vec4f,
  tuning: vec4f,
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
@group(0) @binding(19) var<storage,read_write> sharpenDeposits:array<atomic<i32>>;
// The adaptive method binds its resident signed-distance field here. Uniform
// reference solvers bind volumeIn instead, preserving their VOF formulation.
@group(0) @binding(20) var surfaceIn: texture_3d<f32>;
// Per-column terrain heights in cell units; params.container.w enables it so
// terrain-free scenes never pay the extra load. Static for the whole run.
@group(0) @binding(21) var terrainIn: texture_2d<f32>;
@group(0) @binding(24) var gammaIn: texture_3d<f32>;
@group(0) @binding(25) var gammaOut: texture_storage_3d<r32float, write>;
// The physical rgba velocity texture owns positive MAC faces. The separate
// field owns the three negative domain faces so the CM11a pressure halo has
// persistent velocity DOFs on both sides of every closed wall.
@group(0) @binding(26) var<storage,read> boundaryVelocityIn:array<f32>;
@group(0) @binding(27) var<storage,read_write> boundaryVelocityOut:array<f32>;
// Eight vec4s per (cell, component), populated only in the opt-in Dawn stage
// audit shader variant. Production compiles the constant-false branch away.
@group(0) @binding(28) var<storage,read_write> macCormackAudit:array<vec4f>;
fn dims() -> vec3i { return vec3i(textureDimensions(volumeIn)); }
fn inflowGridDims()->vec3i{return dims();}
fn valid(p: vec3i) -> bool { let d=dims(); return all(p >= vec3i(0)) && all(p < d); }
fn clampCell(p: vec3i) -> vec3i { return clamp(p, vec3i(0), dims()-vec3i(1)); }
// Canonical reductions for the horizontal D4 group. Reflections exchange
// operands inside opposite-direction pairs; x/z exchange operands of the
// horizontal pair sum. The papers prescribe the stencil, not its add order.
fn d4Sum6(value:array<f32,6>)->f32{return ((value[0]+value[1])+(value[4]+value[5]))+(value[2]+value[3]);}
fn d4Sum8(value:array<f32,8>)->f32{
  let y0=(value[0]+value[5])+(value[1]+value[4]);
  let y1=(value[2]+value[7])+(value[3]+value[6]);
  return y0+y1;
}
fn d4Sum6Vec3(value:array<vec3f,6>)->vec3f{return ((value[0]+value[1])+(value[4]+value[5]))+(value[2]+value[3]);}
fn worldCell(id:vec3i)->vec3f{let h=params.cellGravity.xyz;return vec3f(-0.5*params.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*params.container.z+(f32(id.z)+0.5)*h.z);}
fn hasTerrain()->bool{return params.container.w>0.5;}
fn terrainHeightCells(x:i32,z:i32)->f32{let d=dims();return textureLoad(terrainIn,vec2i(clamp(x,0,d.x-1),clamp(z,0,d.z-1)),0).x;}
// Ground handling mirrors the rigid-body solid treatment with zero velocity:
// the heightfield closes faces, drops pressure unknowns, and blocks deposits.
fn cellInsideTerrain(p:vec3i)->bool{if(!hasTerrain()){return false;}return f32(p.y)+0.5<terrainHeightCells(p.x,p.z);}
fn cellTerrainFraction(p:vec3i)->f32{if(!hasTerrain()){return 0.0;}return clamp(terrainHeightCells(p.x,p.z)-f32(p.y),0.0,1.0);}
${inflowBoundaryWGSL}
fn volume(p: vec3i) -> f32 { if (!valid(p)) { return 0.0; } return textureLoad(volumeIn,p,0).x; }
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
fn boundaryFaceIndex(p:vec3i,axis:u32)->u32{
  let d=dims();
  if(axis==0u){return u32(p.y+d.y*p.z);}
  let yOffset=d.y*d.z;
  if(axis==1u){return u32(yOffset+p.x+d.x*p.z);}
  return u32(yOffset+d.x*d.z+p.x+d.x*p.y);
}
fn boundaryVelocity(p:vec3i)->vec3f{
  if(!valid(p)){return vec3f(0.0);}var value=vec3f(0.0);
  if(p.x==0){value.x=boundaryVelocityIn[boundaryFaceIndex(p,0u)];}
  if(p.y==0){value.y=boundaryVelocityIn[boundaryFaceIndex(p,1u)];}
  if(p.z==0){value.z=boundaryVelocityIn[boundaryFaceIndex(p,2u)];}
  return value;
}
fn storeBoundaryVelocity(id:vec3i,value:vec3f){
  if(id.x==0){boundaryVelocityOut[boundaryFaceIndex(id,0u)]=value.x;}
  if(id.y==0){boundaryVelocityOut[boundaryFaceIndex(id,1u)]=value.y;}
  if(id.z==0){boundaryVelocityOut[boundaryFaceIndex(id,2u)]=value.z;}
}
fn carryBoundaryVelocity(id:vec3i){storeBoundaryVelocity(id,boundaryVelocity(id));}
fn liquid(p:vec3i)->bool{return surfaceLiquid(p);}
fn pressureValue(p:vec3i)->f32{
  return textureLoad(pressureIn,clampCell(p),0).x;
}
// Projection alone binds the CM11a finest level, whose physical cells are
// enclosed by a one-cell solid/domain halo. Keep this addressing explicit:
// other main-shader pressure scratch remains an unpadded simulation texture.
fn projectPressureValue(p:vec3i)->f32{
  let pressureDims=vec3i(textureDimensions(pressureIn));
  return textureLoad(pressureIn,clamp(p+vec3i(1),vec3i(0),pressureDims-vec3i(1)),0).x;
}
fn cellOpenFraction(p:vec3i)->f32{
  if(!valid(p)){return 0.0;}
  return clamp((1.0-cellSolidFraction(p))*(1.0-cellTerrainFraction(p)),0.0,1.0);
}
// Chentanez--Mueller Sec. 3.7, Eq. 20.  Surface density represents mass in
// the non-solid part of a cut cell, so pressure classification and the ghost
// fluid distance must use rho'=rho/V rather than raw rho.
fn pressureDensityOpen(p:vec3i)->f32{
  let open=cellOpenFraction(p);
  if(open<=1e-5){return 0.0;}
  return volume(p)/open;
}
fn pressureDensity(p:vec3i)->f32{
  if(!valid(p)){return 0.0;}
  let open=cellOpenFraction(p);
  if(open>1e-5){return pressureDensityOpen(p);}
  // Eq. 20 is extrapolated from V>0 into adjacent V=0 cells.  Although fully
  // solid cells are not pressure unknowns, this continuation keeps the free
  // surface distance well-defined at a cut boundary.
  let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  // A phase-diluting average can turn an adjacent solid continuation back
  // into air when any of the other neighbors are dry. The max continuation
  // guarantees that every V=0 cell adjacent to liquid is retained as the
  // pressure unknown required by Sec. 3.7.
  var continued=0.0;
  for(var index=0;index<6;index+=1){let q=p+offsets[index];if(cellOpenFraction(q)>1e-5){continued=max(continued,pressureDensityOpen(q));}}
  return continued;
}
fn pressurePhi(p:vec3i)->f32{
  let dx=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  return -(pressureDensity(p)-0.5)*dx;
}
// Sec. 3.7 explicitly extrapolates rho' into adjacent V=0 cells so those
// cells participate in the pressure system. Do not filter them back out by V.
fn pressureLiquid(p:vec3i)->bool{return valid(p)&&pressureDensity(p)>0.5;}
fn ghostFluidFraction(liquidCell:vec3i,airCell:vec3i)->f32{
  let liquidPhi=pressurePhi(liquidCell);let airPhi=pressurePhi(airCell);
  return clamp(abs(liquidPhi)/max(abs(liquidPhi)+abs(airPhi),1e-6),GHOST_FLUID_THETA_MIN,1.0);
}
fn sampledFaceVelocity(p:vec3i,component:u32)->f32{
  let d=dims();if(p[component]<0||p[component]>=d[component]){return 0.0;}
  return textureLoad(transportIn,clampCell(p)+vec3i(1),0)[component];
}
fn transportCoordinate(q:vec3f)->vec3f{return (q+vec3f(1.5))/vec3f(dims()+vec3i(2));}
fn sampleVolume(p:vec3f)->f32{
  let q=p-vec3f(0.5);let base=vec3i(floor(q));let f=fract(q);var terms:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let donor=base+offset;
    let weight=select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);
    terms[corner]=select(0.0,weight*volume(donor),valid(donor)&&!cellInsideSolid(donor));
  }
  return d4Sum8(terms);
}
fn sampleVelocityComponent(p:vec3f,component:u32)->f32{
  var offset=vec3f(0.5);offset[component]=1.0;var lower=vec3f(0.0);lower[component]=-1.0;let q=clamp(p-offset,lower,vec3f(dims()-vec3i(1)));
  let base=vec3i(floor(q));let fraction=fract(q);var terms:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let weights=select(vec3f(1.0)-fraction,fraction,vec3f(o)>vec3f(0.5));
    terms[corner]=weights.x*weights.y*weights.z*textureLoad(transportIn,base+o+vec3i(1),0)[component];
  }
  return d4Sum8(terms);
}
fn sampleVelocity(p:vec3f)->vec3f{return vec3f(sampleVelocityComponent(p,0u),sampleVelocityComponent(p,1u),sampleVelocityComponent(p,2u));}
// Reconstruct every RK2 stage at the actual MAC component offsets.  Sampling
// a collocated rgba vector here introduced a +1/4-cell directional shift and
// was the first dynamically accumulated D4 error in symmetric expansion.
fn departurePoint(position:vec3f,dt:f32,h:vec3f)->vec3f{let first=sampleVelocity(position);let midpoint=position-0.5*first*dt/h;return position-sampleVelocity(midpoint)*dt/h;}
// advectVelocityComponent follows rigidBodyIndexAt below: it clips the
// characteristic against the bodies, and WGSL requires declaration before use.
fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}
fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}
fn insideRigid(body:RigidBody,world:vec3f)->bool{
  let p=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
  if(shape==0){return length(p)<=d.x;}
  if(shape==1){return all(abs(p)<=0.5*d);}
  if(shape==2){let cy=clamp(p.y,-0.5*d.y,0.5*d.y);return length(vec3f(p.x,p.y-cy,p.z))<=d.x;}
  return p.x*p.x+p.z*p.z<=d.x*d.x&&abs(p.y)<=0.5*d.y;
}
fn rigidSignedDistance(body:RigidBody,world:vec3f)->f32{
  let p=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
  if(shape==0){return length(p)-d.x;}
  if(shape==1){let q=abs(p)-0.5*d;return length(max(q,vec3f(0.0)))+min(max(q.x,max(q.y,q.z)),0.0);}
  if(shape==2){let cy=clamp(p.y,-0.5*d.y,0.5*d.y);return length(vec3f(p.x,p.y-cy,p.z))-d.x;}
  let q=vec2f(length(p.xz)-d.x,abs(p.y)-0.5*d.y);
  return length(max(q,vec2f(0.0)))+min(max(q.x,q.y),0.0);
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
// Trace-space to world.  Advection positions are cell coordinates carrying a
// per-component MAC offset (cell+(1,.5,.5) is the +x face of cell), so the
// half-cell that worldCell adds is already present in the position.
fn traceWorld(p:vec3f)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(-0.5*params.container.x+p.x*h.x,p.y*h.y,-0.5*params.container.z+p.z*h.z);
}
// Gates the moving-solid corrections.  Each reduces to the original expression
// when no body is present, but the gate keeps body-free scenes on the literal
// original path so their trajectories stay bit-identical rather than merely
// algebraically equal.  Terrain is deliberately excluded: it is static, so it
// gains far less from these corrections than it would cost in re-blessing
// every shipped terrain scene.
fn hasRigidBodies()->bool{return params.boundary.z>=0.5;}
fn insideAnyRigid(world:vec3f)->bool{return rigidBodyIndexAt(world)>=0;}
// Sec. 3.4 stops the density characteristic at a solid boundary; the velocity
// characteristic had no such test.  The RK2 backtrace therefore read straight
// through a body, so liquid ahead of a moving obstacle sampled the liquid
// behind it and the obstacle never pushed it.  Clip the chord at the first
// crossing and sample just outside the surface, where the Sec. 3.3 extension
// has already written u_s.  A departure that is not inside a body -- the
// overwhelmingly common case -- costs exactly one primitive test.
fn clipDepartureAtSolid(position:vec3f,departure:vec3f)->vec3f{
  if(!insideAnyRigid(traceWorld(departure))){return departure;}
  // A departure inside a body converges to lo=0 and samples in place, the
  // correct degenerate answer for a face the body has already swallowed.
  var lo=0.0;var hi=1.0;
  for(var step=0;step<8;step+=1){
    let mid=0.5*(lo+hi);
    if(insideAnyRigid(traceWorld(mix(position,departure,mid)))){hi=mid;}else{lo=mid;}
  }
  return mix(position,departure,lo);
}
// Both branches spell the body-free case as the original expression rather
// than as a clip that happens to be the identity.  Routing it through the
// wrapper instead measured a 1.5e-4 relative shift in maxSpeed on the
// body-free hydrostatic lane -- pure float reassociation, but enough to make
// every still-scene lane need re-blessing for no physical reason.
fn clippedDeparturePoint(position:vec3f,dt:f32,h:vec3f)->vec3f{
  if(!hasRigidBodies()){return departurePoint(position,dt,h);}
  return clipDepartureAtSolid(position,departurePoint(position,dt,h));
}
fn advectVelocityComponent(position:vec3f,component:u32,dt:f32,h:vec3f)->f32{
  if(!hasRigidBodies()){return sampleVelocityComponent(departurePoint(position,dt,h),component);}
  return sampleVelocityComponent(clipDepartureAtSolid(position,departurePoint(position,dt,h)),component);
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
fn worldInsideTerrain(world:vec3f)->bool{
  if(!hasTerrain()){return false;}
  let h=params.cellGravity.xyz;
  let x=clamp(i32(floor((world.x+0.5*params.container.x)/h.x)),0,dims().x-1);
  let z=clamp(i32(floor((world.z+0.5*params.container.z)/h.z)),0,dims().z-1);
  return world.y<terrainHeightCells(x,z)*h.y;
}
fn solidVelocityAtWorld(world:vec3f)->vec4f{
  if(worldInsideTerrain(world)){return vec4f(0.0,0.0,0.0,1.0);}
  let body=rigidBodyIndexAt(world);
  if(body>=0){return vec4f(rigidVelocityAt(body,world),1.0);}
  return vec4f(0.0);
}
// Four transverse quadrature points approximate the non-solid area V^f of a
// MAC face.  Sampling the oriented primitives in world space makes this the
// same fractional variational boundary for static, translating, rotating,
// and non-axis-aligned solids.
fn faceSolidData(id:vec3i,axis:u32)->vec4f{
  let world=faceWorld(id,axis);let h=params.cellGravity.xyz;
  var tangentA=(axis+1u)%3u;var tangentB=(axis+2u)%3u;
  var solid=0.0;var solidVelocity=vec3f(0.0);
  for(var sampleIndex=0u;sampleIndex<4u;sampleIndex+=1u){
    var sampleWorld=world;
    sampleWorld[tangentA]+=select(-0.35,0.35,(sampleIndex&1u)!=0u)*h[tangentA];
    sampleWorld[tangentB]+=select(-0.35,0.35,(sampleIndex&2u)!=0u)*h[tangentB];
    let sample=solidVelocityAtWorld(sampleWorld);solid+=sample.w;solidVelocity+=sample.w*sample.xyz;
  }
  return vec4f(select(vec3f(0.0),solidVelocity/max(solid,1e-6),solid>0.0),solid*0.25);
}
fn faceOpenFraction(id:vec3i,axis:u32)->f32{
  var neighbor=id;neighbor[axis]+=1;
  if(!valid(id)||!valid(neighbor)){
    // The authored open top is exterior air, not a solid face. Every other
    // domain boundary remains a static, fully covered wall.
    if(axis==1u&&valid(id)&&id.y==dims().y-1&&neighbor.y==dims().y&&params.boundary.w>0.5){return 1.0;}
    return 0.0;
  }
  return 1.0-faceSolidData(id,axis).w;
}
fn extrapolatedRigidVelocityAtFace(world:vec3f)->vec3f{
  let bodyCount=u32(round(params.boundary.z));let radius=max(params.cellGravity.x,max(params.cellGravity.y,params.cellGravity.z));
  var nearest=radius;var result=vec3f(0.0);
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}let distance=abs(rigidSignedDistance(rigidBodies[bodyIndex],world));
    if(distance<=nearest){nearest=distance;result=rigidVelocityAt(i32(bodyIndex),world);}
  }
  return result;
}
// CM11a's V_{i+1/2} is not the Sec. 3.6 face aperture above. It is the
// non-solid fraction of the face-centred overlapping (dual) cell. Eight
// volume samples construct that geometric quantity for embedded solids.
// At a grid-aligned closed domain wall, half of the dual cell lies in the
// authored solid exterior, so its inferred geometric fraction is 1/2.
fn pressureFaceData(id:vec3i,axis:u32)->vec4f{
  var neighbor=id;neighbor[axis]+=1;
  if(!valid(id)||!valid(neighbor)){
    if(axis==1u&&valid(id)&&id.y==dims().y-1&&neighbor.y==dims().y&&params.boundary.w>0.5){return vec4f(0.0,0.0,0.0,1.0);}
    if(valid(id)!=valid(neighbor)){return vec4f(0.0,0.0,0.0,0.5);}
    return vec4f(0.0);
  }
  let world=faceWorld(id,axis);let h=params.cellGravity.xyz;var solid=0.0;var solidVelocity=vec3f(0.0);
  for(var sampleIndex=0u;sampleIndex<8u;sampleIndex+=1u){
    let sampleWorld=world+vec3f(select(-0.4,0.4,(sampleIndex&1u)!=0u)*h.x,select(-0.4,0.4,(sampleIndex&2u)!=0u)*h.y,select(-0.4,0.4,(sampleIndex&4u)!=0u)*h.z);
    let sample=solidVelocityAtWorld(sampleWorld);solid+=sample.w;solidVelocity+=sample.w*sample.xyz;
  }
  // CM11a explicitly requires us on nearby liquid faces. For analytic rigid
  // bodies, the nearest body within one cell supplies its rigid velocity at
  // the face centre when the dual-cell samples themselves contain no solid.
  // Terrain and tank walls are static and therefore extrapolate zero.
  let extrapolated=extrapolatedRigidVelocityAtFace(world);
  return vec4f(select(extrapolated,solidVelocity/max(solid,1e-6),solid>0.0),1.0-solid/8.0);
}
fn pressureFaceVolumeFraction(id:vec3i,axis:u32)->f32{return pressureFaceData(id,axis).w;}
// Secs. 3.3 and 3.7 share one interface authority. The extrapolator consumes
// rho'=rho/V (including Eq. 20's adjacent-solid continuation) and the exact
// positive-MAC face fractions used by projection; it never reclassifies raw
// surface density or approximates a second solid boundary.
@compute @workgroup_size(4,4,4)
fn buildExtrapolationAuthority(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  textureStore(volumeOut,id,vec4f(pressureDensity(id)));
  textureStore(velocityOut,id,vec4f(
    faceOpenFraction(id,0u),faceOpenFraction(id,1u),faceOpenFraction(id,2u),0.0));
}
// Projection enforces body velocity on interior faces, so that value cannot be
// used as the undisturbed fluid velocity for form drag. Sample six wet, open
// points just beyond the body's bounding sphere instead.
fn ambientFluidVelocity(body:RigidBody,p:vec3i,fallback:vec3f)->vec3f{
  let h=params.cellGravity.xyz;let radius=max(body.dimensions.w,0.0);let reach=vec3i(ceil(vec3f(2.0*radius)/h))+vec3i(2);
  let offsets=array<vec3i,6>(vec3i(-reach.x,0,0),vec3i(reach.x,0,0),vec3i(0,-reach.y,0),vec3i(0,reach.y,0),vec3i(0,0,-reach.z),vec3i(0,0,reach.z));
  var terms:array<vec3f,6>;var weights:array<f32,6>;
  for(var n=0;n<6;n+=1){let q=p+offsets[n];terms[n]=vec3f(0.0);weights[n]=0.0;if(!valid(q)||cellRigidBody(q)>=0||cellInsideTerrain(q)){continue;}let wet=surfaceOccupancy(q);terms[n]=wet*velocity(q);weights[n]=wet;}
  let total=d4Sum6Vec3(terms);let weight=d4Sum6(weights);
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
  // Sec. 3.6 guarantees rho=0 inside solid. Closed-domain exterior, embedded
  // solids, and authored open-top air therefore share the same zero-density
  // extension for sharpening gradients; none copies a boundary-cell value.
  if(!valid(id)||cellInsideSolid(id)){return 0.0;}
  return volume(id);
}
fn volumeGradient(id:vec3i)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(normalVolume(id+vec3i(1,0,0))-normalVolume(id-vec3i(1,0,0)),normalVolume(id+vec3i(0,1,0))-normalVolume(id-vec3i(0,1,0)),normalVolume(id+vec3i(0,0,1))-normalVolume(id-vec3i(0,0,1)))/(2.0*h);
}
// --- Conservative surface-density transport (paper Sec. 3.4, modified
// three-scatter scheme). beta, rho deficits, and gamma deficits occupy three
// consecutive fixed-point arrays in conditioningScratch/sharpenDeposits.
const TRANSPORT_FIXED:f32=1048576.0;
fn linearIndex(id:vec3i)->u32{let d=dims();return u32(id.x+d.x*(id.y+d.y*id.z));}
fn cellCount()->u32{let d=dims();return u32(d.x*d.y*d.z);}
fn betaValue(id:vec3i)->f32{
  if(!valid(id)){return 1.0;}
  return f32(atomicLoad(&sharpenDeposits[linearIndex(id)]))/TRANSPORT_FIXED;
}
// Characteristics at contacting closed-wall faces satisfy u.n=0, so a true
// trace stays in the domain and slides tangentially along the boundary. (A
// CM11a-released face is handled separately below.) A single straight RK2
// step over the paper time step spans
// u*dt/h cells (10+ at the 64-cubed dam front), punches through the wall,
// and any fold-back (mirror or clamp) then maps distinct departure cells
// onto the same near-wall band: the advection operator turns locally
// compressive at exactly the stagnation cells where mass conservation
// deposits everything (a three-wall corner folds 2^3 = 8x -- the measured
// rho pile). Sub-stepping the integration, clamping each sub-step to the
// domain so the wall's zero normal velocity is re-sampled, follows the
// sliding characteristic instead. The paper sub-steps its sharpening trace
// for the same reason (Sec 3.5: "multiple forward Euler sub-steps",
// stopping at solids).
fn clampTraceToDomain(p:vec3f)->vec3f{
  let d=vec3f(dims());var q=p;
  q.x=clamp(q.x,0.5,d.x-0.5);
  q.z=clamp(q.z,0.5,d.z-0.5);
  q.y=max(q.y,0.5);
  // Only the authored open +Y boundary has an exterior-air continuation.
  if(params.boundary.w<=0.5){q.y=min(q.y,d.y-0.5);}
  return q;
}
// CM11a can release a closed solid face: at a released face the forward
// velocity points into the domain, so a backward characteristic legitimately
// enters the solid. Clamping that departure back onto the last cell centre
// makes the conservative sampler repeatedly read the wall film itself.
fn backwardTraceExitsReleasedFace(p:vec3f)->bool{
  let d=dims();let q=clamp(vec3i(floor(p)),vec3i(0),d-vec3i(1));
  if(p.x<0.5){return boundaryVelocity(vec3i(0,q.y,q.z)).x>1e-6;}
  if(p.x>f32(d.x)-0.5){return faceVelocity(vec3i(d.x-1,q.y,q.z)).x< -1e-6;}
  if(p.y<0.5){return boundaryVelocity(vec3i(q.x,0,q.z)).y>1e-6;}
  if(p.y>f32(d.y)-0.5){return faceVelocity(vec3i(q.x,d.y-1,q.z)).y< -1e-6;}
  if(p.z<0.5){return boundaryVelocity(vec3i(q.x,q.y,0)).z>1e-6;}
  if(p.z>f32(d.z)-0.5){return faceVelocity(vec3i(q.x,q.y,d.z-1)).z< -1e-6;}
  return false;
}
fn integrateTraceOffset(id:vec3i,dt:f32,h:vec3f,direction:f32)->vec3f{
  let position=vec3f(id)+vec3f(0.5);
  let hMin=min(h.x,min(h.y,h.z));
  let substeps=clamp(i32(ceil(length(sampleVelocity(position))*dt/hMin)),1,16);
  let sdt=dt/f32(substeps);
  var p=position;
  for(var s=0;s<substeps;s+=1){
    let midpoint=clampTraceToDomain(p+direction*0.5*sampleVelocity(p)*sdt/h);
    let candidate=p+direction*sampleVelocity(midpoint)*sdt/h;
    if(direction<0.0&&backwardTraceExitsReleasedFace(candidate)){p=candidate;break;}
    let next=clampTraceToDomain(candidate);
    // Stop at embedded solids: the paper's trace "stops if it crosses a
    // solid boundary" rather than passing mass through the obstacle.
    if(cellInsideSolid(vec3i(floor(next)))){break;}
    p=next;
  }
  return p-position;
}
fn backwardTraceOffset(id:vec3i,dt:f32,h:vec3f)->vec3f{
  return integrateTraceOffset(id,dt,h,-1.0);
}
fn forwardTraceOffset(id:vec3i,dt:f32,h:vec3f)->vec3f{
  return integrateTraceOffset(id,dt,h,1.0);
}
// CM12 Secs. 3.6-3.7 classify density storage with the cut-cell volume V, not
// the centre-point solid predicate. A cell whose centre is inside a body can
// still have V>0 and retain rho<=V after excess ejection. Masking that partial
// donor deletes its valid open-subcell density from the conservative matrix.
// The host reconciles rho with current V before this predicate is evaluated,
// so only geometrically full cells are excluded.
fn densityTransportDestination(p:vec3i)->bool{return valid(p)&&cellOpenFraction(p)>1e-5;}
fn transportStencilWeight(base:vec3i,f:vec3f,corner:u32)->f32{
  let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  if(!densityTransportDestination(base+offset)){return 0.0;}
  return select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);
}
fn sampleGammaStencil(base:vec3i,f:vec3f)->f32{
  var terms:array<f32,8>;
  // Invalid or solid corners are the zero Dirichlet extension of gamma. Do
  // not renormalize this backward gather: the paper permits deficient gamma.
  for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let donor=base+offset;let weight=transportStencilWeight(base,f,corner);terms[corner]=0.0;if(weight>0.0){terms[corner]=weight*textureLoad(gammaIn,donor,0).x;}}
  return d4Sum8(terms);
}
// Steps 1-3: backward-advect persistent gamma, initialize beta on the host,
// and scatter gamma_i w^-_li to each donor l.
@compute @workgroup_size(4,4,4)
fn traceGammaAndBeta(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  if(!densityTransportDestination(id)){textureStore(gammaOut,id,vec4f(0.0));return;}
  let traced=backwardTraceOffset(id,params.dimsDt.w,params.cellGravity.xyz);let base=id+vec3i(floor(traced));let f=fract(traced);
  var total=0.0;
  for(var corner=0u;corner<8u;corner+=1u){total+=transportStencilWeight(base,f,corner);}
  // CM12 step 1 is an ordinary backward semi-Lagrangian sample of the
  // cumulative row sum. In particular, a characteristic that leaves a
  // released solid face sees the zero exterior extension. Giving that case a
  // positive floor invents a backward coefficient at the wall and suppresses
  // the forward remainder that is responsible for detaching the wall cell.
  // The interior clamp is the existing large-CFL conditioning policy; it must
  // never turn the exterior zero into a synthetic wall coefficient.
  let sampledGamma=sampleGammaStencil(base,f);
  // Preserve a partially exterior gamma sample instead of applying the
  // interior floor. Beta and density gathering still normalize the visible
  // interpolation stencil below; only cumulative gamma sees the zero exterior.
  let advectedGamma=select(min(sampledGamma,2.5),clamp(sampledGamma,0.5,2.5),total>=1.0-1e-6);
  textureStore(gammaOut,id,vec4f(advectedGamma));
  // No visible donor means there is no backward coefficient. Do not credit a
  // synthetic self coefficient: the gather has no corresponding density
  // term, and doing so prevents the donor's missing column weight from being
  // returned by steps 6-7.
  if(total<=1e-9){return;}
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let donor=base+offset;let weight=transportStencilWeight(base,f,corner)/total;
    if(weight>0.0){atomicAdd(&sharpenDeposits[linearIndex(donor)],i32(round(advectedGamma*weight*TRANSPORT_FIXED)));}
  }
}

// Steps 6-7: sources whose beta is below one forward-scatter the missing
// column weight. gammaIn is the pre-advection gamma^n prescribed by step 7;
// the rho and gamma corrections are accumulated separately.
@compute @workgroup_size(4,4,4)
fn scatterDensityDeficit(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!densityTransportDestination(id)){return;}
  let deficit=max(0.0,1.0-betaValue(id));if(deficit<=1.0/TRANSPORT_FIXED){return;}
  let traced=forwardTraceOffset(id,params.dimsDt.w,params.cellGravity.xyz);let base=id+vec3i(floor(traced));let f=fract(traced);let count=cellCount();
  var total=0.0;
  for(var corner=0u;corner<8u;corner+=1u){total+=transportStencilWeight(base,f,corner);}
  if(total<=1e-9){
    let index=linearIndex(id);
    atomicAdd(&sharpenDeposits[count+index],i32(round(volume(id)*deficit*TRANSPORT_FIXED)));
    atomicAdd(&sharpenDeposits[2u*count+index],i32(round(textureLoad(gammaIn,id,0).x*deficit*TRANSPORT_FIXED)));
    return;
  }
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let receiver=base+offset;let weight=transportStencilWeight(base,f,corner)/total;
    if(weight<=0.0){continue;}let index=linearIndex(receiver);
    atomicAdd(&sharpenDeposits[count+index],i32(round(volume(id)*deficit*weight*TRANSPORT_FIXED)));
    atomicAdd(&sharpenDeposits[2u*count+index],i32(round(textureLoad(gammaIn,id,0).x*deficit*weight*TRANSPORT_FIXED)));
  }
}

// Steps 4-5 plus resolve of 6-7. gammaIn is the backward-advected gamma from
// step 1. Scaling each donor by max(1,beta_l) performs the paper's clamp
// without materializing A.
@compute @workgroup_size(4,4,4)
fn gatherConservativeDensity(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  if(!densityTransportDestination(id)){textureStore(volumeOut,id,vec4f(0.0));textureStore(gammaOut,id,vec4f(0.0));return;}
  let traced=backwardTraceOffset(id,params.dimsDt.w,params.cellGravity.xyz);let base=id+vec3i(floor(traced));let f=fract(traced);
  let advectedGamma=textureLoad(gammaIn,id,0).x;var rhoNext=0.0;var gammaNext=0.0;
  var total=0.0;
  for(var corner=0u;corner<8u;corner+=1u){total+=transportStencilWeight(base,f,corner);}
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let donor=base+offset;
    if(total<=1e-9){break;}
    let weight=transportStencilWeight(base,f,corner)/total;if(weight<=0.0){continue;}
    let scaled=advectedGamma*weight/max(1.0,betaValue(donor));rhoNext+=scaled*volume(donor);
    // Step 5 publishes gamma-prime, the row sum of the conditioned operator.
    gammaNext+=scaled;
  }
  let count=cellCount();let index=linearIndex(id);
  rhoNext+=f32(atomicLoad(&sharpenDeposits[count+index]))/TRANSPORT_FIXED;
  gammaNext+=f32(atomicLoad(&sharpenDeposits[2u*count+index]))/TRANSPORT_FIXED;
  // The prescribed inflow is a mass source external to the conservative
  // operator.  Rasterize the entire timestep-swept plug: a one-layer receiver
  // source would cap the paper's CFL-25 jet at 1/25 of its authored flux. The
  // matching swept velocity support moves this new plug clear before the next
  // step; retain the density-capacity guard for recirculating liquid crossing
  // the prescribed nozzle volume.
  let inflowSource=min(inflowSweptPlugSource(id,params.dimsDt.w),max(0.0,1.0-rhoNext));
  rhoNext+=inflowSource;
  // Gamma is initialized to one throughout the domain at startup. A cell
  // first wetted by the external reservoir needs the same operator state.
  if(inflowSource>0.0){gammaNext=max(gammaNext,1.0);}
  if(rhoNext<1e-5){gammaNext=1.0;}
  textureStore(volumeOut,id,vec4f(max(rhoNext,0.0)));
  textureStore(gammaOut,id,vec4f(max(gammaNext,0.0)));
}

fn diffuseGammaPair(id:vec3i,axis:u32,parity:i32){
  if(!valid(id)){return;}
  // Every invocation owns exactly one output texel. Having only the lower
  // endpoint write both pair members leaves the unpaired boundary cells
  // unwritten on alternating parity passes. Since these textures ping-pong,
  // an unwritten wall cell resurrects its value from two passes ago and
  // breaks the pair's otherwise conservative rho transfer.
  let coordinate=id[axis];var lowerCoordinate=coordinate;
  if((coordinate&1)!=parity){lowerCoordinate-=1;}
  var lower=id;lower[axis]=lowerCoordinate;var upper=lower;upper[axis]+=1;
  if(lowerCoordinate<0||!valid(upper)){
    textureStore(volumeOut,id,vec4f(volume(id)));
    textureStore(gammaOut,id,vec4f(textureLoad(gammaIn,id,0).x));
    return;
  }
  let lowerGamma=textureLoad(gammaIn,lower,0).x;let upperGamma=textureLoad(gammaIn,upper,0).x;
  let averageGamma=0.5*(lowerGamma+upperGamma);var lowerRho=volume(lower);var upperRho=volume(upper);
  // The pair straddles one face.  A solid between the two cells carries no
  // mass, so diffusing across it teleports liquid through the body -- seven
  // iterations of six axis passes every step, which reads as water leaking
  // straight through a dragged object.  Weight the exchange by the face
  // aperture V^f, which is 1 on an open face and 0 on a covered one.
  //
  // The body-free case is spelled as the original statements rather than as a
  // V^f of 1: routing it through the weighted form is algebraically exact but
  // reassociates the arithmetic, which measurably moved a still scene that has
  // no solids in it at all.
  if(hasRigidBodies()){
    let open=clamp(faceOpenFraction(lower,axis),0.0,1.0);
    if(open<=1e-5){
      textureStore(volumeOut,id,vec4f(volume(id)));
      textureStore(gammaOut,id,vec4f(textureLoad(gammaIn,id,0).x));
      return;
    }
    if(upperGamma>lowerGamma){let transfer=open*upperRho*(upperGamma-lowerGamma)/(2.0*max(upperGamma,1e-9));lowerRho+=transfer;upperRho-=transfer;}
    else if(lowerGamma>upperGamma){let transfer=open*lowerRho*(lowerGamma-upperGamma)/(2.0*max(lowerGamma,1e-9));lowerRho-=transfer;upperRho+=transfer;}
    // A partly covered face equilibrates gamma only as far as it is open.
    let ownGamma=select(upperGamma,lowerGamma,coordinate==lowerCoordinate);
    let pairedGamma=ownGamma+open*(averageGamma-ownGamma);
    textureStore(volumeOut,id,vec4f(max(0.0,select(upperRho,lowerRho,coordinate==lowerCoordinate))));
    textureStore(gammaOut,id,vec4f(max(0.0,pairedGamma)));
    return;
  }
  if(upperGamma>lowerGamma){let transfer=upperRho*(upperGamma-lowerGamma)/(2.0*max(upperGamma,1e-9));lowerRho+=transfer;upperRho-=transfer;}
  else if(lowerGamma>upperGamma){let transfer=lowerRho*(lowerGamma-upperGamma)/(2.0*max(lowerGamma,1e-9));lowerRho-=transfer;upperRho+=transfer;}
  textureStore(volumeOut,id,vec4f(max(0.0,select(upperRho,lowerRho,coordinate==lowerCoordinate))));
  textureStore(gammaOut,id,vec4f(max(0.0,averageGamma)));
}
@compute @workgroup_size(4,4,4) fn diffuseGammaX0(@builtin(global_invocation_id) gid:vec3u){diffuseGammaPair(vec3i(gid),0u,0);}
@compute @workgroup_size(4,4,4) fn diffuseGammaX1(@builtin(global_invocation_id) gid:vec3u){diffuseGammaPair(vec3i(gid),0u,1);}
@compute @workgroup_size(4,4,4) fn diffuseGammaY0(@builtin(global_invocation_id) gid:vec3u){diffuseGammaPair(vec3i(gid),1u,0);}
@compute @workgroup_size(4,4,4) fn diffuseGammaY1(@builtin(global_invocation_id) gid:vec3u){diffuseGammaPair(vec3i(gid),1u,1);}
@compute @workgroup_size(4,4,4) fn diffuseGammaZ0(@builtin(global_invocation_id) gid:vec3u){diffuseGammaPair(vec3i(gid),2u,0);}
@compute @workgroup_size(4,4,4) fn diffuseGammaZ1(@builtin(global_invocation_id) gid:vec3u){diffuseGammaPair(vec3i(gid),2u,1);}
// The paper leaves the dimension traversal order unspecified. volumeIn and
// gammaIn are the xyz result; surfaceIn and pressureIn are the zyx result.
// Their equal average is invariant under the horizontal x/z exchange that
// maps one valid paper order to the other.
@compute @workgroup_size(4,4,4)
fn averageGammaDiffusion(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let rho=0.5*(volume(id)+textureLoad(surfaceIn,id,0).x);
  let gamma=0.5*(textureLoad(gammaIn,id,0).x+textureLoad(pressureIn,id,0).x);
  textureStore(volumeOut,id,vec4f(max(rho,0.0)));
  textureStore(gammaOut,id,vec4f(max(gamma,0.0)));
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
  // Sub-isovalue density is still physical liquid. Excluding it from gravity
  // leaves a thin sheet with no way to separate from a ceiling.
  let centerLiquid=occupancy>1e-5;
  let yLiquid=yOccupancy>1e-5;
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
  // Give the newly rasterized high-CFL plug its reservoir velocity before
  // projection. The projection may then redirect it at impacts; only the
  // actual nozzle face is re-imposed after projection.
  return applyInflowSweptVelocity(id,v);
}

@compute @workgroup_size(4,4,4)
fn semiLagrangianAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}carryBoundaryVelocity(id);let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,dt,h));
  // A closed-face sample uses the solid-side zero extension. Preserve an old
  // velocity directed away from a positive wall before adding this step's
  // forces; projection below will clamp only the into-wall sign.
  let d=dims();
  if(id.x==d.x-1){v.x=min(v.x,faceVelocity(id).x);}
  if(id.y==d.y-1&&params.boundary.w<=0.5){v.y=min(v.y,faceVelocity(id).y);}
  if(id.z==d.z-1){v.z=min(v.z,faceVelocity(id).z);}
  v=applyVelocityForces(id,v,dt,h);
  // Surface density is advanced by the dedicated Sec. 3.4 gamma/beta passes.
  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(volume(id),0.0,0.0,0.0));textureStore(pressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn advect(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  carryBoundaryVelocity(id);
  let dt=params.dimsDt.w; let h=params.cellGravity.xyz;
  let cell=vec3f(id);var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,dt,h));
  let advected=volume(id);let d=dims();
  if (id.x==d.x-1) { v.x=faceVelocity(id).x; }
  if (id.y==d.y-1&&params.boundary.w<=0.5) { v.y=faceVelocity(id).y; }
  if (id.z==d.z-1) { v.z=faceVelocity(id).z; }
  textureStore(velocityOut,id,vec4f(v,0.0));
  textureStore(volumeOut,id,vec4f(advected,0.0,0.0,0.0));
  textureStore(pressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn reverseAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  carryBoundaryVelocity(id);
  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,-dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,-dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,-dt,h));let d=dims();
  if(id.x==d.x-1){v.x=faceVelocity(id).x;}if(id.y==d.y-1&&params.boundary.w<=0.5){v.y=faceVelocity(id).y;}if(id.z==d.z-1){v.z=faceVelocity(id).z;}textureStore(velocityOut,id,vec4f(v,0.0));
}

fn boundedMacCormack(id:vec3i,position:vec3f,component:u32,dt:f32,h:vec3f,predicted:f32,original:f32,reversed:f32)->f32{
  var offset=vec3f(0.5);offset[component]=1.0;var lowerCoordinate=vec3f(0.0);lowerCoordinate[component]=-1.0;
  // Same clipped chord the predictor used: bracketing the limiter against
  // donors the predictor never sampled would let the correction reintroduce
  // the through-body velocity the clip just removed.
  let q=clamp(clippedDeparturePoint(position,dt,h)-offset,lowerCoordinate,vec3f(dims()-vec3i(1)));let b=vec3i(floor(q));let fraction=fract(q);
  var donorWeights=array<f32,8>();var donorValues=array<f32,8>();
  donorValues[0]=sampledFaceVelocity(b,component);var lower=1e30;var upper=-1e30;
  donorWeights[0]=(1.0-fraction.x)*(1.0-fraction.y)*(1.0-fraction.z);
  if(donorWeights[0]>0.0){lower=donorValues[0];upper=donorValues[0];}
  for(var corner:u32=1u;corner<8u;corner+=1u){
    let cornerOffset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let value=sampledFaceVelocity(b+cornerOffset,component);
    let weights=select(vec3f(1.0)-fraction,fraction,vec3f(cornerOffset)>vec3f(0.5));
    donorWeights[corner]=weights.x*weights.y*weights.z;donorValues[corner]=value;
    if(donorWeights[corner]>0.0){lower=min(lower,value);upper=max(upper,value);}
  }
  let corrected=predicted+0.5*(original-reversed);
  let revert=corrected<lower||corrected>upper;
  if(MACCORMACK_AUDIT_ENABLED){
    let record=(linearIndex(id)*3u+component)*8u;
    macCormackAudit[record]=vec4f(q,f32(b.x));
    macCormackAudit[record+1u]=vec4f(f32(b.y),f32(b.z),fraction.x,fraction.y);
    macCormackAudit[record+2u]=vec4f(fraction.z,predicted,original,reversed);
    macCormackAudit[record+3u]=vec4f(corrected,lower,upper,select(0.0,1.0,revert));
    macCormackAudit[record+4u]=vec4f(donorWeights[0],donorWeights[1],donorWeights[2],donorWeights[3]);
    macCormackAudit[record+5u]=vec4f(donorWeights[4],donorWeights[5],donorWeights[6],donorWeights[7]);
    macCormackAudit[record+6u]=vec4f(donorValues[0],donorValues[1],donorValues[2],donorValues[3]);
    macCormackAudit[record+7u]=vec4f(donorValues[4],donorValues[5],donorValues[6],donorValues[7]);
  }
  return select(corrected,predicted,revert);
}

@compute @workgroup_size(4,4,4)
fn correctAdvection(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  carryBoundaryVelocity(id);
  let dt=params.dimsDt.w;let h=params.cellGravity.xyz;let cell=vec3f(id);
  let predicted=textureLoad(predictedVelocityIn,id,0).xyz;let original=textureLoad(velocityIn,id,0).xyz;let reversed=textureLoad(reversedVelocityIn,id,0).xyz;
  // Apply body forces to closed-wall faces as well as interior faces. CM11a's
  // separating boundary is an inequality enforced by the following pressure
  // projection, not a permanent zero-normal velocity. Restoring the original
  // here discarded gravity at a released ceiling face, so a detached film
  // lost its downward acceleration and numerically stuck to the lid.
  var v=vec3f(boundedMacCormack(id,cell+vec3f(1.0,0.5,0.5),0u,dt,h,predicted.x,original.x,reversed.x),boundedMacCormack(id,cell+vec3f(0.5,1.0,0.5),1u,dt,h,predicted.y,original.y,reversed.y),boundedMacCormack(id,cell+vec3f(0.5,0.5,1.0),2u,dt,h,predicted.z,original.z,reversed.z));v=applyVelocityForces(id,v,dt,h);
  textureStore(velocityOut,id,vec4f(v,0.0));
}

@compute @workgroup_size(8,8,1)
fn buildHeight(@builtin(global_invocation_id) gid:vec3u){let d=dims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}var total=0.0;for(var y:i32=0;y<d.y;y+=1){total+=volume(vec3i(i32(gid.x),y,i32(gid.y)))*params.cellGravity.y;}textureStore(heightOut,vec2i(gid.xy),vec4f(total));}

fn faceWorld(id:vec3i,axis:u32)->vec3f{
  var world=worldCell(id);world[axis]+=0.5*params.cellGravity.xyz[axis];return world;
}
fn domainFaceFluidVelocity(id:vec3i,axis:u32)->f32{
  var neighbor=id;neighbor[axis]+=1;
  if(valid(id)){return faceVelocity(id)[axis];}
  if(valid(neighbor)&&id[axis]==-1){return boundaryVelocity(neighbor)[axis];}
  return 0.0;
}
fn domainFaceSolidVelocity(id:vec3i,axis:u32,checkSolid:bool)->f32{
  var neighbor=id;neighbor[axis]+=1;
  if(!valid(id)||!valid(neighbor)||(!checkSolid&&!hasTerrain())){return 0.0;}
  return pressureFaceData(id,axis)[axis];
}
fn divergenceAt(id: vec3i, checkSolid: bool) -> f32 {
  // CM11a Eqs. 8-10. Vi is the non-solid cell fraction; V+/- are the
  // corresponding face fractions. This is not the common blended-flux
  // shortcut V u + (1-V) us, whose solid terms are algebraically different.
  let h=params.cellGravity.xyz;let vi=cellOpenFraction(id);var terms:array<f32,6>;
  for(var axis=0u;axis<3u;axis+=1u){
    var minus=id;minus[axis]-=1;
    let vp=pressureFaceVolumeFraction(id,axis);let vm=pressureFaceVolumeFraction(minus,axis);
    let up=domainFaceFluidVelocity(id,axis);let um=domainFaceFluidVelocity(minus,axis);
    let usp=domainFaceSolidVelocity(id,axis,checkSolid);let usm=domainFaceSolidVelocity(minus,axis,checkSolid);
    terms[2u*axis]=(vp*up)/h[axis]+(vp-vi)*usp;
    terms[2u*axis+1u]=-(vm*um)/h[axis]-(vm-vi)*usm;
  }
  return d4Sum6(terms);
}
// Mass-Conserving Eulerian Liquid Simulation Sec 3.7: cells holding more
// density than they represent add min(lambda (rho'-1), eta) artificial
// divergence (lambda = 0.5, eta = 1 per the paper), divided by dx, so the
// pressure solve pushes the excess out.
fn volumeCorrectionDivergence(id: vec3i) -> f32 {
  let excess=max(0.0,pressureDensity(id)-1.0);
  if(excess<=0.0){return 0.0;}
  // The paper's min(lambda(rho'-1),eta)/dx slope is kept for the small-excess
  // regime (it is the calibrated volume-recovery rate; a pure /dt reading
  // measurably leaves residual excess parked in the field, -1.5% represented
  // volume on the 64-cubed dam). The cap, however, is re-read as a rate: at
  // the paper's dx=0.05, dt=1/30 the published eta/dx (20/s) and 1/dt (30/s)
  // are indistinguishable, but at 4x finer grids eta/dx demands a fractional
  // expansion of dt/dx = 267% of a cell PER STEP, and a wall pile-up then
  // pumps sustained multi-step jets that tear the surface apart (the
  // 64-cubed dam "boiling"). Bounding the purge at one cell per step keeps
  // the paper's stated intent -- rate-limiting Mullen's unstable
  // lambda=1, eta=infinity correction -- resolution-independent.
  return min(excess,1.0)/params.dimsDt.w;
}

fn curvatureAt(id:vec3i)->f32{
  let h=params.cellGravity.xyz;
  let x=(interfaceNormal(id+vec3i(1,0,0)).x-interfaceNormal(id-vec3i(1,0,0)).x)/(2.0*h.x);
  let y=(interfaceNormal(id+vec3i(0,1,0)).y-interfaceNormal(id-vec3i(0,1,0)).y)/(2.0*h.y);
  let z=(interfaceNormal(id+vec3i(0,0,1)).z-interfaceNormal(id-vec3i(0,0,1)).z)/(2.0*h.z);
  return -((x+z)+y);
}

@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  let h=params.cellGravity.xyz;let scale=params.dimsDt.w/params.physical.x;var v=velocity(id);var boundaryV=boundaryVelocity(id);let d=dims();
  let ex=id+vec3i(1,0,0);let ey=id+vec3i(0,1,0);let ez=id+vec3i(0,0,1);
  let p0=select(0.0,projectPressureValue(id),pressureLiquid(id));
  let neighbors=array<vec3i,3>(ex,ey,ez);
  for(var axis=0u;axis<3u;axis+=1u){
    let neighbor=neighbors[axis];
    if(id[axis]==0){
      var halo=id;halo[axis]-=1;
      let boundaryOpen=pressureFaceVolumeFraction(halo,axis);
      if(boundaryOpen>1e-5&&pressureLiquid(id)){boundaryV[axis]-=scale*(p0-projectPressureValue(halo))/h[axis];}
      else{boundaryV[axis]=0.0;}
    }
    if(id[axis]==d[axis]-1){
      if(axis==1u&&params.boundary.w>0.5){
        if(pressureLiquid(id)){
          let theta=ghostFluidFraction(id,neighbor);
          v.y-=scale*(0.0-p0)/(h.y*theta);
        }else{v.y=0.0;}
      }else if(pressureLiquid(id)){
        let boundaryOpen=pressureFaceVolumeFraction(id,axis);
        if(boundaryOpen>1e-5){
          // A partially open solid face couples to the CM11a p_min=0 halo.
          v[axis]-=scale*(projectPressureValue(neighbor)-p0)/h[axis];
        }else{v[axis]=domainFaceSolidVelocity(id,axis,true);}
      }else{
        // Thin density has no pressure row, but its boundary face still owns
        // the separating inequality. On a positive wall, retain v<=u_s.
        let solidVelocity=domainFaceSolidVelocity(id,axis,true);
        v[axis]=select(solidVelocity,min(v[axis],solidVelocity),volume(id)>1e-5);
      }
      continue;
    }
    let pressureFace=pressureFaceData(id,axis);let open=select(1.0,pressureFace.w,hasTerrain()||nearAnyBody(faceWorld(id,axis)));
    if(open<=1e-5){v[axis]=pressureFace[axis];continue;}
    let centreLiquid=pressureLiquid(id);let neighborLiquid=pressureLiquid(neighbor);
    if(centreLiquid||neighborLiquid){
      let p1=select(0.0,projectPressureValue(neighbor),neighborLiquid);
      var theta=1.0;
      if(centreLiquid&&!neighborLiquid){theta=ghostFluidFraction(id,neighbor);}
      if(!centreLiquid&&neighborLiquid){theta=ghostFluidFraction(neighbor,id);}
      v[axis]-=scale*(p1-p0)/(h[axis]*theta);
    }else{v[axis]=0.0;}
  }
  v=applyInflowVelocity(id,v);textureStore(velocityOut,id,vec4f(v,0.0));storeBoundaryVelocity(id,boundaryV); textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));
}

// Moving-solid bookkeeping after the variational projection.  The old
// Brinkman velocity blend changed face velocities after incompressibility and
// reintroduced divergence; solid motion is now imposed inside the projection,
// so this pass only records the diagnostic/reaction load.  Sec. 3.6 already
// performs the conservative covered-density redistribution before projection.
@compute @workgroup_size(4,4,4)
fn coupleRigid(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}carryBoundaryVelocity(id);let phi=volume(id);let wetFraction=surfaceOccupancy(id);var v=velocity(id);let h=params.cellGravity.xyz;
  let world=vec3f(-0.5*params.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*params.container.z+(f32(id.z)+0.5)*h.z);
  let bodyCount=u32(round(params.boundary.z));let cellMass=params.physical.x*h.x*h.y*h.z*wetFraction;let blend=clamp(45.0*params.dimsDt.w,0.0,1.0);var coupledBody=12u;var solidFraction=0.0;
  // Match the adaptive voxelizer's overlap rule: the body with the greatest
  // sub-cell coverage owns this cell, so displaced volume is never counted
  // twice and does not depend on body-array order.
  for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let candidate=bodySolidFraction(rigidBodies[bodyIndex],id);if(candidate>solidFraction){solidFraction=candidate;coupledBody=bodyIndex;}}
  if(coupledBody<12u){
    let bodyIndex=coupledBody;let body=rigidBodies[bodyIndex];
    let arm=world-body.positionShape.xyz;let solidVelocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);let fluidVelocity=v;let ambientVelocity=ambientFluidVelocity(body,id,fluidVelocity);let fluidImpulse=cellMass*solidFraction*(solidVelocity-fluidVelocity)*blend;
    let reaction=-fluidImpulse;let torque=cross(arm,reaction);let base=bodyIndex*12u;
    atomicAdd(&rigidExchange[base],i32(round(reaction.x*1000000.0)));atomicAdd(&rigidExchange[base+1u],i32(round(reaction.y*1000000.0)));atomicAdd(&rigidExchange[base+2u],i32(round(reaction.z*1000000.0)));
    atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*1000000.0)));atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*1000000.0)));atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*1000000.0)));
    let displacedWeight=wetFraction*solidFraction;
    atomicAdd(&rigidExchange[base+6u],i32(round(displacedWeight*65536.0)));
    atomicAdd(&rigidExchange[base+7u],i32(round(displacedWeight*ambientVelocity.x*10000.0)));atomicAdd(&rigidExchange[base+8u],i32(round(displacedWeight*ambientVelocity.y*10000.0)));atomicAdd(&rigidExchange[base+9u],i32(round(displacedWeight*ambientVelocity.z*10000.0)));
  }
  // The nozzle mouth is an open boundary. Coupling the visual nozzle body
  // must not replace the prescribed reservoir velocity at that opening.
  v=applyInflowVelocity(id,v);
  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(phi));
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
  // TraceAlongField must stop at a closed domain wall just as it stops at an
  // embedded solid. The only non-solid invalid coordinate is above an open
  // top, where the exterior is deliberately air.
  if(!valid(p)){return !(p.y>=dims().y&&params.boundary.w>0.5);}
  if(cellInsideTerrain(p)){return true;}
  let bodyCount=u32(round(params.boundary.z));if(bodyCount==0u){return false;}
  let world=worldCell(p);
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],world)){return true;}}
  return false;
}
fn sharpenDeltaRho(q:vec3i)->f32{
  let rho=volume(q);
  if(cellInsideSolid(q)){return 0.0;}
  let h=params.cellGravity.xyz;
  let deltaT=3.0*params.dimsDt.w*params.tuning.x;let tau=0.4;
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  // Sec. 3.6 Eqs. 18-19 use non-solid face aperture area V^f. This is
  // deliberately distinct from CM11a's face-centred overlapping dual volume.
  let openXp=faceOpenFraction(q,0u);let openXm=faceOpenFraction(q-ex,0u);
  let openYp=faceOpenFraction(q,1u);let openYm=faceOpenFraction(q-ey,1u);
  let openZp=faceOpenFraction(q,2u);let openZm=faceOpenFraction(q-ez,2u);
  let sxp=-(rho*max(openXp,0.0)-volume(q-ex)*max(openXm,0.0))*deltaT/h.x;let sxm=-(volume(q+ex)*max(openXp,0.0)-rho*max(openXm,0.0))*deltaT/h.x;
  let syp=-(rho*max(openYp,0.0)-volume(q-ey)*max(openYm,0.0))*deltaT/h.y;let sym=-(volume(q+ey)*max(openYp,0.0)-rho*max(openYm,0.0))*deltaT/h.y;
  let szp=-(rho*max(openZp,0.0)-volume(q-ez)*max(openZm,0.0))*deltaT/h.z;let szm=-(volume(q+ez)*max(openZp,0.0)-rho*max(openZm,0.0))*deltaT/h.z;
  let plusX=max(max(sxp,0.0)*max(sxp,0.0),min(sxm,0.0)*min(sxm,0.0));let plusY=max(max(syp,0.0)*max(syp,0.0),min(sym,0.0)*min(sym,0.0));let plusZ=max(max(szp,0.0)*max(szp,0.0),min(szm,0.0)*min(szm,0.0));
  let minusX=max(min(sxp,0.0)*min(sxp,0.0),max(sxm,0.0)*max(sxm,0.0));let minusY=max(min(syp,0.0)*min(syp,0.0),max(sym,0.0)*max(sym,0.0));let minusZ=max(min(szp,0.0)*min(szp,0.0),max(szm,0.0)*max(szm,0.0));
  let gradPlus=sqrt((plusX+plusZ)+plusY);let gradMinus=sqrt((minusX+minusZ)+minusY);
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
  var p=vec3f(id)+vec3f(0.5);let maximumDistance=params.tuning.y;var travelled=0.0;let stepLength=0.5;
  for(var stepIndex=0;stepIndex<7;stepIndex+=1){
    if(sampleVolume(p)>=0.5||travelled>=maximumDistance){break;}
    let g=volumeGradient(vec3i(floor(p)));let magnitude=length(g);
    if(magnitude<1e-6){break;}
    let distance=min(stepLength,maximumDistance-travelled);
    let candidate=p+g/magnitude*distance;
    if(cellInsideSolid(vec3i(floor(candidate)))){break;}
    p=candidate;travelled+=distance;
    if(sampleVolume(p)>=0.5){break;}
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

// Sec. 3.6: when rho exceeds V, trace the excess for S*dx along the gradient
// of the solid signed-distance field (positive away from the union of solids).
fn solidSignedDistance(world:vec3f)->f32{
  var distance=1e20;
  if(hasTerrain()){
    let h=params.cellGravity.xyz;
    let x=i32(floor((world.x+0.5*params.container.x)/h.x));
    let z=i32(floor((world.z+0.5*params.container.z)/h.z));
    distance=min(distance,world.y-terrainHeightCells(x,z)*h.y);
  }
  let bodyCount=u32(round(params.boundary.z));
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}
    distance=min(distance,rigidSignedDistance(rigidBodies[bodyIndex],world));
  }
  return distance;
}
fn solidSignedDistanceGradient(world:vec3f)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(
    (solidSignedDistance(world+vec3f(h.x,0.0,0.0))-solidSignedDistance(world-vec3f(h.x,0.0,0.0)))/(2.0*h.x),
    (solidSignedDistance(world+vec3f(0.0,h.y,0.0))-solidSignedDistance(world-vec3f(0.0,h.y,0.0)))/(2.0*h.y),
    (solidSignedDistance(world+vec3f(0.0,0.0,h.z))-solidSignedDistance(world-vec3f(0.0,0.0,h.z)))/(2.0*h.z));
}
@compute @workgroup_size(4,4,4)
fn scatterSolidExcess(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let rho=volume(id);let open=cellOpenFraction(id);
  if(open>=1.0-1e-6){textureStore(volumeOut,id,vec4f(rho));return;}
  let excess=max(0.0,rho-open);
  textureStore(volumeOut,id,vec4f(rho-excess));if(excess<=0.0){return;}
  let h=params.cellGravity.xyz;let dx=min(h.x,min(h.y,h.z));
  let gradient=solidSignedDistanceGradient(worldCell(id));var p=vec3f(id)+vec3f(0.5);
  if(length(gradient)>1e-6){p+=normalize(gradient)*dx/h;}
  let anchor=p-vec3f(0.5);let cell=vec3i(floor(anchor));let f=fract(anchor);
  var weights=array<f32,8>();var indices=array<u32,8>();var total=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let destination=cell+offset;
    var weight=select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);
    if(!valid(destination)||cellInsideSolid(destination)){weight=0.0;}else{indices[corner]=linearIndex(destination);}
    weights[corner]=weight;total+=weight;
  }
  // The donor was already reduced to V. Returning an unplaceable deposit to
  // a V=0 donor would violate the paper's rho=0-inside-solid guarantee.
  if(total<=1e-9){
    // The gradient landed entirely in solid.  Before conceding the mass, sweep
    // the 26-neighbourhood for any open cell: a body sweeping through liquid
    // evicts whole cells at once, and the single-gradient-step trace fails
    // often enough there that conceding on the first miss is visible volume
    // loss exactly while the user is playing with the body.
    var fallbackTotal=0.0;
    for(var neighbor=0u;neighbor<27u;neighbor+=1u){
      if(neighbor==13u){continue;}
      let destination=id+vec3i(i32(neighbor%3u)-1,i32((neighbor/3u)%3u)-1,i32(neighbor/9u)-1);
      if(valid(destination)&&!cellInsideSolid(destination)){fallbackTotal+=cellOpenFraction(destination);}
    }
    if(fallbackTotal<=1e-9){
      // Genuinely enclosed by solid.  Keep rho=0 inside the body and expose
      // the unplaceable conservative mass instead of silently losing it.
      atomicAdd(&reductions[4],u32(round(excess*2048.0)));
      return;
    }
    for(var neighbor=0u;neighbor<27u;neighbor+=1u){
      if(neighbor==13u){continue;}
      let destination=id+vec3i(i32(neighbor%3u)-1,i32((neighbor/3u)%3u)-1,i32(neighbor/9u)-1);
      if(!valid(destination)||cellInsideSolid(destination)){continue;}
      let share=cellOpenFraction(destination)/fallbackTotal;
      if(share<=0.0){continue;}
      atomicAdd(&sharpenDeposits[linearIndex(destination)],i32(round(excess*share*TRANSPORT_FIXED)));
    }
    return;
  }
  for(var corner=0u;corner<8u;corner+=1u){if(weights[corner]>0.0){atomicAdd(&sharpenDeposits[indices[corner]],i32(round(excess*weights[corner]/total*TRANSPORT_FIXED)));}}
}
@compute @workgroup_size(4,4,4)
fn resolveSolidExcess(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let deposit=f32(atomicLoad(&sharpenDeposits[linearIndex(id)]))/TRANSPORT_FIXED;
  textureStore(volumeOut,id,vec4f(volume(id)+deposit));
}

// Render-only wall-film reconstruction. For a solid signed distance s and a
// supported cell density rho<.5, R=.5-s/h+rho has its .5 contour at s=rho*h.
// It therefore displays the mass as a proportionally thin sheet rather than
// promoting it to a half-cell liquid region. R is written only to the render
// texture and never becomes transport or pressure authority.
fn domainWallFilmCell(id:vec3i,rho:f32)->bool{
  if(rho<=1e-5||rho>=0.5){return false;}
  let d=dims();
  return id.x==0||id.z==0||id.x==d.x-1||id.z==d.z-1||id.y==0
    ||(id.y==d.y-1&&params.boundary.w<=0.5);
}
fn embeddedWallFilm(id:vec3i)->vec2f{
  let world=worldCell(id);let distance=solidSignedDistance(world);
  if(distance>=1e19){return vec2f(0.0);}
  let gradient=solidSignedDistanceGradient(world);let gradientLength=length(gradient);
  if(gradientLength<=1e-6){return vec2f(0.0);}
  let normal=gradient/gradientLength;let h=params.cellGravity.xyz;
  let normalCellWidth=1.0/max(length(normal/h),1e-6);
  if(distance>1.25*normalCellWidth){return vec2f(0.0);}
  let sourceWorld=world+normal*(0.5*normalCellWidth-distance);
  let source=vec3i(floor(vec3f(
    (sourceWorld.x+0.5*params.container.x)/h.x,
    sourceWorld.y/h.y,
    (sourceWorld.z+0.5*params.container.z)/h.z)));
  if(!valid(source)||cellInsideSolid(source)){return vec2f(0.0);}
  let rho=textureLoad(volumeIn,source,0).x;
  if(rho<=1e-5||rho>=0.5){return vec2f(0.0);}
  return vec2f(max(0.0,0.5-distance/normalCellWidth+rho),1.0);
}
fn wallFilmResolvedDensity(id:vec3i,base:f32)->f32{
  let film=embeddedWallFilm(id);
  return max(base,film.x);
}
@compute @workgroup_size(4,4,4)
fn wallFilmResolve(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let rho=textureLoad(volumeIn,id,0).x;
  textureStore(volumeOut,id,vec4f(wallFilmResolvedDensity(id,rho)));
}

// Optional Sec. 3.8 reconstruction. Blur g=2 min(rho,.5) with a separable
// Gaussian (sigma=2 cells), then expose sub-grid mass through
// rho''=rho/min(max(g,theta),1), theta=.01. None of these outputs are rebound
// into transport or projection.
fn blurPostprocessAxis(id:vec3i,axis:u32,seedDensity:bool)->f32{
  var weighted=0.0;var total=0.0;
  for(var offset=-6;offset<=6;offset+=1){
    var q=id;q[axis]+=offset;if(!valid(q)){continue;}
    let weight=exp(-f32(offset*offset)/8.0);let sample=textureLoad(volumeIn,q,0).x;
    weighted+=weight*select(sample,2.0*min(sample,0.5),seedDensity);total+=weight;
  }
  return weighted/max(total,1e-9);
}
fn storePostprocessBlur(id:vec3i,axis:u32,seedDensity:bool){textureStore(volumeOut,id,vec4f(blurPostprocessAxis(id,axis,seedDensity)));}
@compute @workgroup_size(4,4,4) fn postprocessBlurX(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(valid(id)){storePostprocessBlur(id,0u,true);}}
@compute @workgroup_size(4,4,4) fn postprocessBlurY(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(valid(id)){storePostprocessBlur(id,1u,false);}}
@compute @workgroup_size(4,4,4) fn postprocessBlurZ(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(valid(id)){storePostprocessBlur(id,2u,false);}}
@compute @workgroup_size(4,4,4)
fn postprocessResolve(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}
  let rho=textureLoad(volumeIn,id,0).x;let blurredGamma=textureLoad(surfaceIn,id,0).x;
  var reconstructed=rho/min(max(blurredGamma,0.01),1.0);
  // Preserve calibrated wall-cell rho: the surface extractor supplies its
  // matching boundary ghost. Sec. 3.8 remains active everywhere unsupported.
  if(domainWallFilmCell(id,rho)){reconstructed=rho;}
  let embedded=embeddedWallFilm(id);
  if(embedded.y>0.5){reconstructed=max(rho,embedded.x);}
  textureStore(volumeOut,id,vec4f(reconstructed));
}
@compute @workgroup_size(4,4,4)
fn reduceDiagnostics(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!valid(id)){return;}let represented=surfaceOccupancy(id);let conservative=volume(id);atomicAdd(&reductions[0],u32(represented*2048.0+0.5));if(surfaceLiquid(id)){atomicMax(&reductions[1],u32(id.x+1));}let speed=length(faceVelocity(id));atomicMax(&reductions[2],bitcast<u32>(speed));atomicAdd(&reductions[3],u32(clamp(conservative,0.0,8.0)*2048.0+0.5));}
`;
