import { solidOccupancyMaskWGSL } from "../../core/solid-occupancy-mask";
import { UNIFORM_PAGE_TABLE_WORDS, uniformLatticeAddressWGSL } from "./uniform-lattice-address";

/** The two dense t=0 passes needed by narrow-band FLIP. */
export function createUniformReferenceComputeShader(): string {
  return /* wgsl */ `const UNIFORM_REFERENCE_DIMENSION: u32 = 3u;

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

// The page table (a 64³ window of words), then a word per block: a mirror
// of the page pool's.
@group(0) @binding(29) var<storage,read> umPages:array<u32>;

@group(0) @binding(30) var<storage,read_write> activeScratch:array<u32>;


fn dims() -> vec3i { return vec3i(textureDimensions(volumeIn)); }
fn umPageWord(page:vec3i)->u32{let e=vec3u(page)&vec3u(63u);return umPages[e.x|(e.y<<6u)|(e.z<<12u)];}
fn umBlockWord(b:u32)->u32{return umPages[${UNIFORM_PAGE_TABLE_WORDS}u+b];}
${uniformLatticeAddressWGSL({cells:"vec3u(dims())",tiles:"(vec3u(dims())/4u)",constant:false,paged:{blocks:`(arrayLength(&umPages)-${UNIFORM_PAGE_TABLE_WORDS}u)`}})}

fn activeWindowOrigin()->vec3u{return vec3u(0u);}

fn activeWindowExtent()->vec3u{
  return vec3u(dims())-activeWindowOrigin();
}

fn activeId(gid:vec3u)->vec3i{

  if(any(gid>=activeWindowExtent())){return vec3i(-1);}
  return vec3i(activeWindowOrigin()+gid);

}




// The solid mask's image opens activeScratch (SolidOccupancyMask): a
// directory word a page of the window, then the mixed solid library's
// bed-slope lane and tile cut map (uniformSolidScratchLayout) and the page
// slots the directory names. A page the lattice walls off is solid throughout.
${solidOccupancyMaskWGSL("activeScratch", "staticSolidPage", "staticSolidVoxelOccupied")}

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

fn cellTerrainFraction(p:vec3i)->f32{if(!hasTerrain()){return 0.0;}return clamp(terrainHeightCells(p.x,p.z)-f32(p.y),0.0,1.0);}

fn cellOpenFraction(p:vec3i)->f32{
  if(!umCellPresent(p)||staticSolidVoxelOccupied(p)){return 0.0;}
  return clamp((1.0-bodySolidFractionAt(p))*(1.0-cellTerrainFraction(p)),0.0,1.0);
}

fn geometricVolumeEnabled()->bool{return true;}

fn pressureSurfacePhi(p:vec3i)->f32{





















  return uvPhi(vec3f(umCellNear(p))+vec3f(0.5));
}

fn pressurePhi(p:vec3i)->f32{
  if(!geometricVolumeEnabled()||cellOpenFraction(p)>1e-5){return pressureSurfacePhi(p);}
  let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  if(umCellKind(p)==UM_AIR){return 0.5*h;}
  let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var terms:array<f32,6>;var weights:array<f32,6>;
  for(var i=0u;i<6u;i++){let q=p+offsets[i];let open=cellOpenFraction(q);
    terms[i]=0.0;weights[i]=0.0;if(open<=1e-5){continue;}
    let phi=pressureSurfacePhi(q);if(phi<0.0){terms[i]=open*phi;weights[i]=open;}}
  let weight=d4Sum6(weights);
  return select(0.5*h,d4Sum6(terms)/max(weight,1e-9),weight>0.0);
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
  // A face between a stored cell and an unstored one: a wall, or the open top where the unstored cell is AIR.
  if(umCellPresent(id)!=umCellPresent(neighbor)){
    if(axis==2u&&depthSymmetry()){return vec4f(0.0);}
    let ambient=umCellKind(select(id,neighbor,umCellPresent(id)))==UM_AIR;


    let interior=select(neighbor,id,umCellPresent(id));
    let open=cellOpenFraction(interior);
    return vec4f(0.0,0.0,0.0,select(0.5*open,0.5*(1.0+open),ambient));
  }
  if(staticSolidVoxelOccupied(id)||staticSolidVoxelOccupied(neighbor)){

    let open=0.5*(cellOpenFraction(id)+cellOpenFraction(neighbor));
    return vec4f(0.0,0.0,0.0,open);
  }
  // Neither cell is stored.
  if(!umCellPresent(id)){return vec4f(0.0);}
  let world=faceWorld(id,axis);let h=params.cellGravity.xyz;var solid=0.0;var solidVelocity=vec3f(0.0);
  for(var sampleIndex=0u;sampleIndex<8u;sampleIndex+=1u){
    let sampleWorld=world+vec3f(select(-0.4,0.4,(sampleIndex&1u)!=0u)*h.x,select(-0.4,0.4,(sampleIndex&2u)!=0u)*h.y,select(-0.4,0.4,(sampleIndex&4u)!=0u)*h.z);




    let sample=solidVelocitySample(sampleWorld,world);solid+=sample.w;solidVelocity+=sample.w*sample.xyz;
  }




  let extrapolated=extrapolatedRigidVelocityAtFace(world);
  return vec4f(select(extrapolated,solidVelocity/max(solid,1e-6),solid>0.0),1.0-solid/8.0);
}

fn pressureFaceVolumeFraction(id:vec3i,axis:u32)->f32{return pressureFaceData(id,axis).w;}

fn storeExtrapolationAuthority(id:vec3i){if(!umCellPresent(id)){return;}
  textureStore(volumeOut,id,vec4f(0.5-pressurePhi(id)/min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z))));
  textureStore(velocityOut,id,vec4f(
    pressureFaceVolumeFraction(id,0u),pressureFaceVolumeFraction(id,1u),pressureFaceVolumeFraction(id,2u),cellOpenFraction(id)));
}

@compute @workgroup_size(4,4,4)
fn buildDenseExtrapolationAuthority(@builtin(global_invocation_id) gid:vec3u){storeExtrapolationAuthority(vec3i(gid));}

fn faceWorld(id:vec3i,axis:u32)->vec3f{
  var world=worldCell(id);world[axis]+=0.5*params.cellGravity.xyz[axis];return world;
}

@group(0) @binding(31) var uvPhiIn:texture_3d<f32>;

fn uvCorner(i:u32)->vec3i{return vec3i(i32(i&1u),i32((i>>1u)&1u),i32((i>>2u)&1u));}

fn uvPhi(position:vec3f)->f32{
  let p=umPointNear(position);
  let base=umCellNear(vec3i(floor(p)));let f=p-vec3f(base);
  var values:array<f32,8>;
  for(var i=0u;i<8u;i++){let o=uvCorner(i);let w=select(vec3f(1)-f,f,o==vec3i(1));
    values[i]=textureLoad(uvPhiIn,base+o,0).x*w.x*w.y*w.z;}
  return d4Sum8(values);
}

fn uvOpen(id:vec3i)->f32{if(!umCellPresent(id)){return 0.0;}return cellOpenFraction(id);}

@compute @workgroup_size(4,4,4)
fn uvPublish(@builtin(global_invocation_id)gid:vec3u){
  let id=activeId(gid);if(!umCellPresent(id)){return;}
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
  textureStore(volumeOut,id,vec4f(value));
  textureStore(gammaOut,id,vec4f(open));
}

`;
}
