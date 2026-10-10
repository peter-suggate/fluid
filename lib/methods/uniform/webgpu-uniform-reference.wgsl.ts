/** The two dense t=0 passes needed by narrow-band FLIP. */
export function createUniformReferenceComputeShader(_geometric = true, referenceDimension: 2 | 3 = 3, _pages?: unknown, _domain?: unknown, _fullLattice = true): string {
  return /* wgsl */ `const UNIFORM_REFERENCE_DIMENSION: u32 = ${referenceDimension}u;

const CM12_LIQUID_ISOVALUE:f32=0.5;

const CM12_GHOST_FLUID_THETA_MIN:f32=0.05;

fn cm12GhostFluidTheta(liquidPhi:f32,airPhi:f32,denominatorEpsilon:f32)->f32{
  return clamp(abs(liquidPhi)/max(abs(liquidPhi)+abs(airPhi),denominatorEpsilon),CM12_GHOST_FLUID_THETA_MIN,1.0);
}

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



  drop: vec4f,


  dropExtent: vec4f,



  twoLevel: vec4f,



  agreement: vec4f,











  lean: vec4f,






  splash: vec4f,



  splashB: vec4f,


  cleanup: vec4f,

  extrusion: vec4f,

  extrusionEdges: array<vec4f, 64>,
}

@group(0) @binding(0) var velocityIn: texture_3d<f32>;

@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;

@group(0) @binding(2) var pressureIn: texture_3d<f32>;

@group(0) @binding(4) var volumeIn: texture_3d<f32>;

@group(0) @binding(5) var volumeOut: texture_storage_3d<r32float, write>;

@group(0) @binding(6) var<uniform> params: Params;

@group(0) @binding(9) var<storage,read_write> reductions:array<atomic<u32>,12>;

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

@group(0) @binding(13) var reversedVelocityIn: texture_3d<f32>;

@group(0) @binding(19) var<storage,read_write> sharpenDeposits:array<atomic<i32>>;

@group(0) @binding(21) var terrainIn: texture_2d<f32>;

@group(0) @binding(24) var gammaIn: texture_3d<f32>;

@group(0) @binding(25) var gammaOut: texture_storage_3d<r32float, write>;

@group(0) @binding(26) var<storage,read> boundaryVelocityIn:array<f32>;

@group(0) @binding(27) var<storage,read_write> boundaryVelocityOut:array<f32>;

@group(0) @binding(29) var<storage,read> activeRegion:array<u32>;

@group(0) @binding(30) var<storage,read_write> activeScratch:array<u32>;

const ACTIVE_PRESSURE_ORIGIN_WORD:u32=240u;

fn dims() -> vec3i { return vec3i(textureDimensions(volumeIn)); }

fn activeWindowOrigin()->vec3u{return vec3u(0u);}

fn activeWindowExtent()->vec3u{
  return vec3u(dims())-activeWindowOrigin();
}

fn activeId(gid:vec3u)->vec3i{

  if(any(gid>=activeWindowExtent())){return vec3i(-1);}
  return vec3i(activeWindowOrigin()+gid);

}

fn pressureWindowLattice()->bool{return false;}

fn pressureWindowOrigin()->vec3i{
  if(!pressureWindowLattice()){return vec3i(0);}
  return vec3i(vec3u(activeRegion[ACTIVE_PRESSURE_ORIGIN_WORD],
    activeRegion[ACTIVE_PRESSURE_ORIGIN_WORD+1u],activeRegion[ACTIVE_PRESSURE_ORIGIN_WORD+2u]));
}

fn inflowGridDims()->vec3i{return dims();}

fn valid(p: vec3i) -> bool { let d=dims(); return all(p >= vec3i(0)) && all(p < d); }

fn clampCell(p: vec3i) -> vec3i { return clamp(p, vec3i(0), dims()-vec3i(1)); }

fn solidVoxelWord(index:u32)->u32{
  return activeScratch[u32(round(params.dropExtent.z))+index];
}

fn staticSolidVoxelOccupied(p:vec3i)->bool{



  let shape=vec3i(params.dimsDt.xyz)+vec3i(2);
  let q=p+vec3i(1);
  if(any(q<vec3i(0))||any(q>=shape)){return false;}
  let index=u32(q.x+shape.x*(q.y+shape.y*q.z));
  return (solidVoxelWord(4u+(index>>5u))&(1u<<(index&31u)))!=0u;
}

fn d4Sum6(value:array<f32,6>)->f32{return ((value[0]+value[1])+(value[4]+value[5]))+(value[2]+value[3]);}

fn d4Sum8(value:array<f32,8>)->f32{
  let y0=(value[0]+value[5])+(value[1]+value[4]);
  let y1=(value[2]+value[7])+(value[3]+value[6]);
  return y0+y1;
}

fn worldCell(id:vec3i)->vec3f{let h=params.cellGravity.xyz;return vec3f(-0.5*params.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*params.container.z+(f32(id.z)+0.5)*h.z);}

fn hasTerrain()->bool{return params.container.w>0.5;}

fn depthSymmetry()->bool{return params.tuning.w>0.5;}

fn terrainHeightCells(x:i32,z:i32)->f32{let d=dims();return textureLoad(terrainIn,vec2i(clamp(x,0,d.x-1),clamp(z,0,d.z-1)),0).x;}

fn cellInsideTerrain(p:vec3i)->bool{if(!hasTerrain()){return false;}return f32(p.y)+0.5<terrainHeightCells(p.x,p.z);}

fn cellTerrainFraction(p:vec3i)->f32{if(!hasTerrain()){return 0.0;}return clamp(terrainHeightCells(p.x,p.z)-f32(p.y),0.0,1.0);}

fn inflowStrength()->f32{return clamp(params.inflowTiming.x,0.0,1.0);}

fn inflowAxis()->u32{let v=abs(params.inflowVelocityLength.xyz);if(v.y>v.x&&v.y>=v.z){return 1u;}if(v.z>v.x){return 2u;}return 0u;}

fn inflowFaceIndex(axis:u32)->i32{let h=params.cellGravity.xyz;let minimum=vec3f(-0.5*params.container.x,0.0,-0.5*params.container.z);let d=inflowGridDims();return clamp(i32(round((params.inflowPositionRadius[axis]-minimum[axis])/h[axis]))-1,0,d[axis]-2);}

fn inflowReceiverIndex(axis:u32)->i32{let face=inflowFaceIndex(axis);return select(face,face+1,params.inflowVelocityLength[axis]>=0.0);}

fn inflowApertureFraction(q:vec3i)->f32{
  let velocity=params.inflowVelocityLength.xyz;let speed=length(velocity);if(speed<=1e-6){return 0.0;}let direction=velocity/speed;let axis=inflowAxis();let h=params.cellGravity.xyz;let minimum=vec3f(-0.5*params.container.x,0.0,-0.5*params.container.z);let face=inflowFaceIndex(axis);var tangentA=0u;var tangentB=1u;if(axis==0u){tangentA=1u;tangentB=2u;}else if(axis==1u){tangentA=0u;tangentB=2u;}
  var point=vec3f(0.0);point[axis]=minimum[axis]+f32(face+1)*h[axis];point[tangentA]=minimum[tangentA]+(f32(q[tangentA])+0.5)*h[tangentA];point[tangentB]=minimum[tangentB]+(f32(q[tangentB])+0.5)*h[tangentB];let relative=point-params.inflowPositionRadius.xyz;let axial=dot(relative,direction);let radialDistance=length(relative-axial*direction);let edgeWidth=max(0.5*length(vec2f(h[tangentA],h[tangentB])),1e-6);let coverage=clamp(0.5+0.5*(params.inflowPositionRadius.w-radialDistance)/edgeWidth,0.0,1.0);return coverage*coverage*(3.0-2.0*coverage);
}

fn isInflowVelocityCell(q:vec3i)->bool{
  let axis=inflowAxis();let receiver=inflowReceiverIndex(axis);let donor=select(receiver+1,receiver-1,params.inflowVelocityLength[axis]>=0.0);return inflowStrength()>0.0&&(q[axis]==receiver||q[axis]==donor)&&inflowApertureFraction(q)>0.0;
}

fn applyInflowVelocity(q:vec3i,inputVelocity:vec3f)->vec3f{
  let strength=inflowStrength();if(strength<=0.0){return inputVelocity;}let axis=inflowAxis();let receiver=inflowReceiverIndex(axis);if(!isInflowVelocityCell(q)){return inputVelocity;}var desiredVelocity=params.inflowVelocityLength.xyz*strength;desiredVelocity[axis]*=params.inflowVelocityLength.w;var velocity=inputVelocity;if(q[axis]==receiver){velocity=desiredVelocity;}velocity[axis]=desiredVelocity[axis];return velocity;
}

fn volume(p: vec3i) -> f32 { if (!valid(p)) { return 0.0; } return textureLoad(volumeIn,p,0).x; }

fn levelSetAuthority() -> bool { return true; }

fn surfaceValue(p: vec3i) -> f32 {
  if (!valid(p)) { return select(0.0, 5.0 * min(params.cellGravity.x, min(params.cellGravity.y, params.cellGravity.z)), levelSetAuthority()); }
  return uvPhi(vec3f(p)+vec3f(0.5));
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

fn liquid(p:vec3i)->bool{return surfaceLiquid(p);}

fn projectPressureValue(p:vec3i)->f32{
  let pressureDims=vec3i(textureDimensions(pressureIn));
  let local=p+vec3i(1)-pressureWindowOrigin();






  if(valid(p)&&(any(local<vec3i(0))||any(local>=pressureDims))){return 0.0;}
  return textureLoad(pressureIn,clamp(local,vec3i(0),pressureDims-vec3i(1)),0).x;
}

fn cellOpenFraction(p:vec3i)->f32{
  if(!valid(p)||staticSolidVoxelOccupied(p)){return 0.0;}
  return clamp((1.0-bodySolidFractionAt(p))*(1.0-cellTerrainFraction(p)),0.0,1.0);
}

fn geometricVolumeEnabled()->bool{return true;}

fn pressureSurfacePhi(p:vec3i)->f32{





















  let q=clampCell(p);var phi=uvPhi(vec3f(q)+vec3f(0.5));









  if(params.splashB.w>0.5){let v=volume(q);
    let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
    phi=min(phi,max(h*(1.0-v),-0.5*h));}

  if(params.physical.w<0.5){return phi;}
  let open=cellOpenFraction(q);if(open<=1e-5){return phi;}
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  let volumePhi=h*(0.5-volume(q)/open);
  if(params.physical.w>1.5){return min(phi,volumePhi);}
  if(phi<0.0||volumePhi>=0.0){return phi;}
  for(var axis=0;axis<3;axis+=1){for(var side=-1;side<=1;side+=2){
    var n=q;n[axis]+=side;if(valid(n)&&uvPhi(vec3f(n)+vec3f(0.5))<0.0){return phi;}}}
  return max(volumePhi,-0.5*h);
}

fn pressurePhi(p:vec3i)->f32{
  if(!geometricVolumeEnabled()||cellOpenFraction(p)>1e-5){return pressureSurfacePhi(p);}
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  if(params.boundary.w>0.5&&p.y==dims().y&&p.x>=0&&p.x<dims().x&&p.z>=0&&p.z<dims().z){return 0.5*h;}
  let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var terms:array<f32,6>;var weights:array<f32,6>;
  for(var i=0u;i<6u;i++){let q=p+offsets[i];let open=cellOpenFraction(q);
    terms[i]=0.0;weights[i]=0.0;if(open<=1e-5){continue;}
    let phi=pressureSurfacePhi(q);if(phi<0.0){terms[i]=open*phi;weights[i]=open;}}
  let weight=d4Sum6(weights);
  return select(0.5*h,d4Sum6(terms)/max(weight,1e-9),weight>0.0);
}

fn pressureLiquid(p:vec3i)->bool{return valid(p)&&pressurePhi(p)<0.0;}

fn ghostFluidFraction(liquidCell:vec3i,airCell:vec3i)->f32{
  let liquidPhi=pressurePhi(liquidCell);let airPhi=pressurePhi(airCell);
  return cm12GhostFluidTheta(liquidPhi,airPhi,1e-9);
}

fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.0*(q.x*uv+uuv);}

fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}

fn sceneShapeCupWall_m(d: vec3f) -> f32 { return clamp(d.z, 1e-4, min(0.95 * d.x, 0.95 * d.y)); }

fn sceneShapeCupDistance_m(d: vec3f, p: vec3f) -> f32 {
  let t = sceneShapeCupWall_m(d);
  let outerQ = vec2f(length(p.xz) - d.x, abs(p.y) - 0.5 * d.y);
  let outer = length(max(outerQ, vec2f(0.0))) + min(max(outerQ.x, outerQ.y), 0.0);





  let cavity = max(length(p.xz) - (d.x - t), (-0.5 * d.y + t) - p.y);
  return max(outer, -cavity);
}

fn rigidShapeInside(shape: i32, d: vec3f, p: vec3f) -> bool {
  if (shape == 0) {
    return length(p) <= d.x;
  }
  if (shape == 1) {
    return all(abs(p) <= 0.5 * d);
  }
  if (shape == 2) {
    let cy = clamp(p.y, -0.5 * d.y, 0.5 * d.y);
    return length(vec3f(p.x, p.y - cy, p.z)) <= d.x;
  }
  if (shape == 3) {
    return p.x * p.x + p.z * p.z <= d.x * d.x && abs(p.y) <= 0.5 * d.y;
  }
  if (shape == 4) {
    return sceneShapeCupDistance_m(d, p) <= 0.0;
  }
  return false;
}

fn rigidShapeDistance(shape: i32, d: vec3f, p: vec3f) -> f32 {
  if (shape == 0) {
    return length(p) - d.x;
  }
  if (shape == 1) {
    let q = abs(p) - 0.5 * d;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
  }
  if (shape == 2) {
    let cy = clamp(p.y, -0.5 * d.y, 0.5 * d.y);
    return length(vec3f(p.x, p.y - cy, p.z)) - d.x;
  }
  if (shape == 3) {
    let q = vec2f(length(p.xz) - d.x, abs(p.y) - 0.5 * d.y);
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
  }
  if (shape == 4) {
    return sceneShapeCupDistance_m(d, p);
  }
  return 1e20;
}

fn rigidShapeBoundingRadius(shape: i32, d: vec3f) -> f32 {
  if (shape == 0) {
    return d.x;
  }
  if (shape == 1) {
    return 0.5 * length(d);
  }
  if (shape == 2) {
    return d.x + 0.5 * d.y;
  }
  if (shape == 3) {
    return length(vec2f(d.x, 0.5 * d.y));
  }
  if (shape == 4) {
    return length(vec2f(d.x, 0.5 * d.y));
  }
  return 0.0;
}

fn rigidLocalPoint(body:RigidBody,world:vec3f)->vec3f{
  return quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);
}

fn rigidShapeTag(body:RigidBody)->i32{return i32(round(body.positionShape.w));}

fn insideRigid(body:RigidBody,world:vec3f)->bool{
  return rigidShapeInside(rigidShapeTag(body),body.dimensions.xyz,rigidLocalPoint(body,world));
}

fn rigidSignedDistance(body:RigidBody,world:vec3f)->f32{
  return rigidShapeDistance(rigidShapeTag(body),body.dimensions.xyz,rigidLocalPoint(body,world));
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

fn staticSolidVoxelAtWorld(world:vec3f)->bool{
  let h=params.cellGravity.xyz;
  let p=vec3i(floor(vec3f((world.x+0.5*params.container.x)/h.x,
    world.y/h.y,(world.z+0.5*params.container.z)/h.z)));
  return staticSolidVoxelOccupied(p);
}

fn nearAnyBody(world:vec3f)->bool{
  let bodyCount=u32(round(params.boundary.z));
  let margin=2.0*max(params.cellGravity.x,max(params.cellGravity.y,params.cellGravity.z));
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){
    if(bodyIndex>=bodyCount){break;}
    let body=rigidBodies[bodyIndex];






    let radius=rigidShapeBoundingRadius(rigidShapeTag(body),body.dimensions.xyz);
    if(distance(world,body.positionShape.xyz)<=radius+margin){return true;}
  }
  return false;
}

fn bodySolidFraction(body:RigidBody,p:vec3i)->f32{
  var inside=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
    if(insideRigid(body,worldCell(p)+offset*params.cellGravity.xyz)){inside+=1.0;}
  }
  return inside/8.0;
}

fn bodySolidFractionAt(p:vec3i)->f32{
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

fn solidVelocitySample(world:vec3f,velocityPoint:vec3f)->vec4f{
  if(staticSolidVoxelAtWorld(world)){return vec4f(0.0,0.0,0.0,1.0);}
  if(worldInsideTerrain(world)){return vec4f(0.0,0.0,0.0,1.0);}
  let body=rigidBodyIndexAt(world);
  if(body>=0){return vec4f(rigidVelocityAt(body,velocityPoint),1.0);}
  return vec4f(0.0);
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

fn pressureFaceData(id:vec3i,axis:u32)->vec4f{
  var neighbor=id;neighbor[axis]+=1;
  if(valid(id)!=valid(neighbor)){
    if(axis==2u&&depthSymmetry()){return vec4f(0.0);}
    let ambient=axis==1u&&max(id.y,neighbor.y)==dims().y&&params.boundary.w>0.5;


    let interior=select(neighbor,id,valid(id));
    let open=cellOpenFraction(interior);
    return vec4f(0.0,0.0,0.0,select(0.5*open,0.5*(1.0+open),ambient));
  }
  if(staticSolidVoxelOccupied(id)||staticSolidVoxelOccupied(neighbor)){

    let open=0.5*(cellOpenFraction(id)+cellOpenFraction(neighbor));
    return vec4f(0.0,0.0,0.0,open);
  }
  if(staticSolidVoxelOccupied(id)||staticSolidVoxelOccupied(neighbor)){
    return vec4f(0.0,0.0,0.0,0.5);
  }
  if(!valid(id)||!valid(neighbor)){


    if(axis==2u&&depthSymmetry()&&valid(id)!=valid(neighbor)){return vec4f(0.0);}
    if(valid(id)==valid(neighbor)){return vec4f(0.0);}

    return vec4f(0.0,0.0,0.0,select(1.0,0.5,
      staticSolidVoxelOccupied(id)||staticSolidVoxelOccupied(neighbor)));
  }
  let world=faceWorld(id,axis);let h=params.cellGravity.xyz;var solid=0.0;var solidVelocity=vec3f(0.0);
  for(var sampleIndex=0u;sampleIndex<8u;sampleIndex+=1u){
    let sampleWorld=world+vec3f(select(-0.4,0.4,(sampleIndex&1u)!=0u)*h.x,select(-0.4,0.4,(sampleIndex&2u)!=0u)*h.y,select(-0.4,0.4,(sampleIndex&4u)!=0u)*h.z);




    let sample=solidVelocitySample(sampleWorld,world);solid+=sample.w;solidVelocity+=sample.w*sample.xyz;
  }




  let extrapolated=extrapolatedRigidVelocityAtFace(world);
  return vec4f(select(extrapolated,solidVelocity/max(solid,1e-6),solid>0.0),1.0-solid/8.0);
}

fn pressureFaceVolumeFraction(id:vec3i,axis:u32)->f32{return pressureFaceData(id,axis).w;}

fn pressureFaceVolumeFractionShared(id:vec3i,axis:u32)->f32{
  if(params.dropExtent.w>0.5&&valid(id)){return textureLoad(reversedVelocityIn,id,0)[axis];}
  return pressureFaceVolumeFraction(id,axis);
}

fn storeExtrapolationAuthority(id:vec3i){if(!valid(id)){return;}
  textureStore(volumeOut,id,vec4f(uvAirborneAuthority(id,0.5-pressurePhi(id)/min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z)))));
  textureStore(velocityOut,id,vec4f(
    pressureFaceVolumeFraction(id,0u),pressureFaceVolumeFraction(id,1u),pressureFaceVolumeFraction(id,2u),cellOpenFraction(id)));
}

@compute @workgroup_size(4,4,4)
fn buildDenseExtrapolationAuthority(@builtin(global_invocation_id) gid:vec3u){storeExtrapolationAuthority(vec3i(gid));}

fn cellCount()->u32{let d=dims();return u32(d.x*d.y*d.z);}

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

fn divergenceAtWithCapacity(id: vec3i, checkSolid: bool, vi:f32) -> f32 {




  let h=params.cellGravity.xyz;var terms:array<f32,6>;
  for(var axis=0u;axis<3u;axis+=1u){
    var minus=id;minus[axis]-=1;
    let vp=pressureFaceVolumeFractionShared(id,axis);let vm=pressureFaceVolumeFractionShared(minus,axis);
    let up=domainFaceFluidVelocity(id,axis);let um=domainFaceFluidVelocity(minus,axis);
    let usp=domainFaceSolidVelocity(id,axis,checkSolid);let usm=domainFaceSolidVelocity(minus,axis,checkSolid);
    terms[2u*axis]=(vp*up+(vi-vp)*usp)/h[axis];
    terms[2u*axis+1u]=-(vm*um+(vi-vm)*usm)/h[axis];
  }
  return d4Sum6(terms);
}

fn divergenceAt(id:vec3i,checkSolid:bool)->f32{
  return divergenceAtWithCapacity(id,checkSolid,cellOpenFraction(id));
}

fn volumeCorrectionDivergenceFromAuthority(id:vec3i,cap:f32,phi:f32)->f32{
  let v=volume(id);let positive=uvVolumeCorrectionAmount(v,cap);
  var deficit=0.0;
  if(cap>1e-5&&v<=cap&&phi<0.0){deficit=max(0.0,textureLoad(gammaIn,id,0).x-v);}
  let rate=bitcast<f32>(atomicLoad(&sharpenDeposits[uvBalanceBase()]));
  return (positive-rate*deficit)/max(params.dimsDt.w,1e-12);
}

fn volumeCorrectionDivergence(id: vec3i) -> f32 {
  let positive=uvVolumeCorrectionAmount(volume(id),cellOpenFraction(id));
  let rate=bitcast<f32>(atomicLoad(&sharpenDeposits[uvBalanceBase()]));return (positive-rate*uvSurfaceDeficit(id))/max(params.dimsDt.w,1e-12);
}

fn geometricPressureValue(p:vec3i)->f32{
  return select(0.0,projectPressureValue(p),pressurePhi(p)<0.0);
}

fn geometricProjectedFace(id:vec3i,axis:u32,predicted:f32)->f32{
  var q=id;q[axis]+=1;
  if(pressureFaceVolumeFractionShared(id,axis)<=1e-6){return pressureFaceData(id,axis)[axis];}
  let a=pressurePhi(id)<0.0;let b=pressurePhi(q)<0.0;

  if(!a&&!b){return select(0.0,predicted,uvAirborneCell(id)||uvAirborneCell(q));}
  var theta=1.0;
  if(a&&!b){theta=ghostFluidFraction(id,q);}
  if(!a&&b){theta=ghostFluidFraction(q,id);}
  return predicted-params.dimsDt.w/params.physical.x*
    (geometricPressureValue(q)-geometricPressureValue(id))/(params.cellGravity[axis]*theta);
}

@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  let id=activeId(gid); if (!valid(id)) { return; }







  if(params.twoLevel.z>0.5&&!uvTwoLevelFineAt(vec3f(id)+vec3f(0.5))){
    textureStore(velocityOut,id,vec4f(0.0));storeBoundaryVelocity(id,vec3f(0.0));
    textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));return;
  }
  let h=params.cellGravity.xyz;let scale=params.dimsDt.w/params.physical.x;var v=velocity(id);var boundaryV=boundaryVelocity(id);let d=dims();
  let ex=id+vec3i(1,0,0);let ey=id+vec3i(0,1,0);let ez=id+vec3i(0,0,1);
  let p0=select(0.0,projectPressureValue(id),pressureLiquid(id));
  let neighbors=array<vec3i,3>(ex,ey,ez);
  for(var axis=0u;axis<3u;axis+=1u){
    let neighbor=neighbors[axis];

    if(id[axis]==0){var halo=id;halo[axis]-=1;
      boundaryV[axis]=geometricProjectedFace(halo,axis,boundaryV[axis]);}
    v[axis]=geometricProjectedFace(id,axis,v[axis]);

  }
  var released=0u;



  for(var axis=0u;axis<3u;axis++){
    var q=id;q[axis]+=1;let ownOpen=cellOpenFraction(id);let otherOpen=cellOpenFraction(q);
    if((ownOpen>1e-5)!=(otherOpen>1e-5)){
      let solid=select(id,q,ownOpen>1e-5);let inward=select(1.0,-1.0,ownOpen>1e-5);
      let wall=pressureFaceData(id,axis);
      if(wall.w>1e-6&&geometricPressureValue(solid)<=0.0&&inward*(v[axis]-wall[axis])*params.dimsDt.w>1e-4*h[axis]){released|=1u<<axis;}
    }
    if(id[axis]==0){var halo=id;halo[axis]-=1;
      if(cellOpenFraction(id)>1e-5&&pressureFaceVolumeFractionShared(halo,axis)>1e-6&&geometricPressureValue(halo)<=0.0&&boundaryV[axis]*params.dimsDt.w>1e-4*h[axis]){released|=1u<<(axis+3u);}}
  }
  v=applyInflowVelocity(id,v);textureStore(velocityOut,id,vec4f(v,f32(released)));storeBoundaryVelocity(id,boundaryV); textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));
}

fn cellInsideSolid(p:vec3i)->bool{
  if(staticSolidVoxelOccupied(p)){return true;}
  if(!valid(p)){return false;}
  if(cellInsideTerrain(p)){return true;}
  let bodyCount=u32(round(params.boundary.z));if(bodyCount==0u){return false;}
  let world=worldCell(p);
  for(var bodyIndex=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}if(insideRigid(rigidBodies[bodyIndex],world)){return true;}}
  return false;
}

@group(0) @binding(31) var uvPhiIn:texture_3d<f32>;

fn uvCorner(i:u32)->vec3i{return vec3i(i32(i&1u),i32((i>>1u)&1u),i32((i>>2u)&1u));}

fn uvPhi(position:vec3f)->f32{
  let p=clamp(position,vec3f(0),vec3f(dims()));
  let base=min(vec3i(floor(p)),dims()-vec3i(1));let f=p-vec3f(base);
  var values:array<f32,8>;
  for(var i=0u;i<8u;i++){let o=uvCorner(i);let w=select(vec3f(1)-f,f,o==vec3i(1));
    values[i]=textureLoad(uvPhiIn,base+o,0).x*w.x*w.y*w.z;}
  return d4Sum8(values);
}

fn uvSolidFree()->bool{return params.lean.x>0.5;}

const UV_LIQUID_EVIDENCE:f32=0.05;

fn uvOpen(id:vec3i)->f32{if(!valid(id)){return 0.0;}return cellOpenFraction(id);}

fn uvCoarseDims()->vec3i{return (dims()+vec3i(3))/4;}

fn uvCoarseIndex(t:vec3i)->u32{let c=uvCoarseDims();return u32(t.x+c.x*(t.y+c.y*t.z));}

fn uvCoarseBase()->u32{return cellCount();}

fn uvTwoLevelFineAt(p:vec3f)->bool{
  let cell=clamp(vec3i(floor(p)),vec3i(0),dims()-vec3i(1));
  return (atomicLoad(&sharpenDeposits[uvCoarseBase()+4u*uvCoarseIndex(cell/4)+3u])&1)!=0;
}

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
  let open=uvOpen(id);
  var value=0.5-uvPhi(vec3f(id)+vec3f(0.5))/h;
  if(open<=1e-5){









    var wettest=-1e30;
    for(var axis=0u;axis<3u;axis++){for(var side=-1;side<=1;side+=2){
      var n=id;n[axis]+=side;if(uvOpen(n)<=1e-5){continue;}
      wettest=max(wettest,0.5-uvPhi(vec3f(n)+vec3f(0.5))/h);}}
    value=min(select(value,1.0-wettest,wettest>=0.5),0.5-1e-3);
  }
  else if(params.splash.z>0.5&&value<-1.0){value=max(value,uvOrphanRender(id,params.splash.z>1.5));}
  textureStore(volumeOut,id,vec4f(value));
  textureStore(gammaOut,id,vec4f(open));
}

fn uvVolumeCorrectionFractionAt(dt:f32)->f32{
 let steps=max(dt,0.0)*30.0;let x=steps*0.6931471805599453;
 if(x<0.01){return x*(1.0-x*(0.5-x/6.0));}
 return 1.0-exp2(-steps);
}

fn uvVolumeCorrectionAmountAt(v:f32,cap:f32,dt:f32)->f32{
 return min(uvVolumeCorrectionFractionAt(dt)*max(0.0,v-cap),cap);
}

fn uvVolumeCorrectionAmount(v:f32,cap:f32)->f32{return uvVolumeCorrectionAmountAt(v,cap,params.dimsDt.w);}

fn uvBalanceBase()->u32{return 3u*cellCount();}

fn uvSurfaceDeficit(id:vec3i)->f32{
  let cap=cellOpenFraction(id);let v=volume(id);
  if(cap<=1e-5||v>cap||pressurePhi(id)>=0.0){return 0.0;}
  return max(0.0,textureLoad(gammaIn,id,0).x-v);
}
`;
}
export const uniformReferenceComputeShader = createUniformReferenceComputeShader();
