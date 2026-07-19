import type { SceneDescription } from "./model";
import type { Quaternion, Vec3 } from "./model";
import type { RigidBodyState } from "./rigid-body";
import { SVO_PRIMITIVE_MOTION_STRIDE_BYTES, svoPrimitiveMotionWGSL } from "./svo-primitive-motion";
import { sceneHasTerrain } from "./terrain";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

export const GPU_RIGID_BODY_CAPACITY = 12;
export const GPU_RIGID_STATE_FLOATS = 32;
export const GPU_RIGID_STATE_BYTES = GPU_RIGID_BODY_CAPACITY * GPU_RIGID_STATE_FLOATS * 4;
export const GPU_RIGID_RENDER_FLOATS = 16;
export const GPU_RIGID_RENDER_BYTES = GPU_RIGID_BODY_CAPACITY * GPU_RIGID_RENDER_FLOATS * 4;
export const GPU_RIGID_MOTION_BYTES = GPU_RIGID_BODY_CAPACITY * SVO_PRIMITIVE_MOTION_STRIDE_BYTES;
const GPU_RIGID_PICK_BYTES = 48;

export interface GPURigidBodyPick {
  bodyIndex: number;
  distance_m: number;
  position_m: Vec3;
  orientation: Quaternion;
}

export const gpuRigidBodyShader = /* wgsl */ `
${svoPrimitiveMotionWGSL}
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
struct RenderBody { positionRadius: vec4f, halfShape: vec4f, orientation: vec4f, colorSelected: vec4f }
struct Params {
  step: vec4f,
  gravity: vec4f,
  container: vec4f,
  terrain: vec4f,
  coupling: vec4f,
}
struct PickParams { originCount: vec4f, direction: vec4f }
struct PickResult { index: u32, hit: u32, distance: f32, pad: f32, position: vec4f, orientation: vec4f }
@group(0) @binding(0) var<storage, read_write> bodies: array<RigidBody, 12>;
@group(0) @binding(1) var<storage, read_write> exchange: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> renderBodies: array<RenderBody, 12>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var terrainHeights: texture_2d<f32>;
@group(0) @binding(5) var<uniform> pickParams: PickParams;
@group(0) @binding(6) var<storage, read_write> pickResult: PickResult;
@group(0) @binding(7) var<storage, read_write> rigidMotion: array<SvoPrimitiveMotionRecord, 12>;

fn qConjugate(q: vec4f) -> vec4f { return vec4f(q.x, -q.yzw); }
fn qMultiply(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.x*b.x-dot(a.yzw,b.yzw), a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));
}
fn qRotate(q: vec4f, v: vec3f) -> vec3f {
  let uv=cross(q.yzw,v); return v+2.0*(q.x*uv+cross(q.yzw,uv));
}
fn qInverseRotate(q: vec4f, v: vec3f) -> vec3f { return qRotate(qConjugate(q),v); }
fn safeNormalize(v: vec3f, fallback: vec3f) -> vec3f { let l=length(v); return select(fallback,v/l,l>1e-8); }
fn inverseInertia(body: RigidBody, v: vec3f) -> vec3f {
  let rho=max(params.step.y,1e-9); let local=qInverseRotate(body.orientation,v);
  return qRotate(body.orientation,local*body.inverseMassInertia.yzw/rho);
}
fn supportRadius(body: RigidBody, directionWorld: vec3f) -> f32 {
  let direction=safeNormalize(directionWorld,vec3f(1,0,0)); let local=qInverseRotate(body.orientation,direction);
  let d=body.dimensions.xyz; let shape=i32(round(body.positionShape.w));
  if(shape==0){return d.x;}
  if(shape==1){return .5*(abs(local.x)*d.x+abs(local.y)*d.y+abs(local.z)*d.z);}
  if(shape==2){return d.x+.5*d.y*abs(local.y);}
  return d.x*length(local.xz)+.5*d.y*abs(local.y);
}
fn velocityAt(body: RigidBody, arm: vec3f) -> vec3f { return body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm); }
fn angularTerm(body: RigidBody, arm: vec3f, direction: vec3f) -> f32 {
  return dot(cross(inverseInertia(body,cross(arm,direction)),arm),direction);
}
fn applyImpulse(body: ptr<function,RigidBody>, impulse: vec3f, arm: vec3f) {
  let rho=max(params.step.y,1e-9); let inverseMass=(*body).inverseMassInertia.x/rho;
  (*body).linearVelocity=vec4f((*body).linearVelocity.xyz+impulse*inverseMass,(*body).linearVelocity.w);
  (*body).angularMomentumRestitution=vec4f((*body).angularMomentumRestitution.xyz+cross(arm,impulse),(*body).angularMomentumRestitution.w);
  (*body).angularVelocity=vec4f(inverseInertia(*body,(*body).angularMomentumRestitution.xyz),(*body).angularVelocity.w);
}
fn planeContact(body: ptr<function,RigidBody>, normal: vec3f, offset: f32) {
  let rho=max(params.step.y,1e-9); let inverseMass=(*body).inverseMassInertia.x/rho;
  if(inverseMass<=0.0){return;}
  let radius=supportRadius(*body,normal); let penetration=offset-(dot(normal,(*body).positionShape.xyz)-radius);
  if(penetration<=0.0){return;}
  (*body).positionShape=vec4f((*body).positionShape.xyz+normal*(penetration+1e-7),(*body).positionShape.w);
  let arm=-normal*radius; var relative=velocityAt(*body,arm); let normalSpeed=dot(relative,normal);
  if(normalSpeed>=0.0){return;}
  let restitution=select(0.0,(*body).angularMomentumRestitution.w,-normalSpeed>0.5);
  let denominator=max(inverseMass+angularTerm(*body,arm,normal),1e-9);
  let normalMagnitude=-(1.0+restitution)*normalSpeed/denominator;
  applyImpulse(body,normal*normalMagnitude,arm);
  relative=velocityAt(*body,arm); let tangentVelocity=relative-normal*dot(relative,normal); let tangentSpeed=length(tangentVelocity);
  if(tangentSpeed<=1e-8){return;}
  let tangent=tangentVelocity/tangentSpeed; let tangentDenominator=max(inverseMass+angularTerm(*body,arm,tangent),1e-9);
  let tangentMagnitude=clamp(-tangentSpeed/tangentDenominator,-(*body).material.x*normalMagnitude,(*body).material.x*normalMagnitude);
  applyImpulse(body,tangent*tangentMagnitude,arm);
}
fn bodyVolume(body: RigidBody) -> f32 {
  let d=body.dimensions.xyz; let shape=i32(round(body.positionShape.w));
  if(shape==0){return 4.1887902047863905*d.x*d.x*d.x;}
  if(shape==1){return d.x*d.y*d.z;}
  if(shape==3){return 3.141592653589793*d.x*d.x*d.y;}
  return 3.141592653589793*d.x*d.x*d.y+4.1887902047863905*d.x*d.x*d.x;
}
fn terrainPlane(position: vec3f) -> vec4f {
  if(params.container.w<0.5){return vec4f(0,1,0,0);}
  let dims=vec2i(textureDimensions(terrainHeights));
  let fx=(position.x/params.container.x+.5)*f32(dims.x); let fz=(position.z/params.container.z+.5)*f32(dims.y);
  let cell=clamp(vec2i(floor(vec2f(fx,fz))),vec2i(0),dims-vec2i(1));
  let xm=max(0,cell.x-1);let xp=min(dims.x-1,cell.x+1);let zm=max(0,cell.y-1);let zp=min(dims.y-1,cell.y+1);
  let h=params.terrain.x; let center=textureLoad(terrainHeights,cell,0).x*h;
  let dx=max(params.container.x/f32(dims.x),1e-6);let dz=max(params.container.z/f32(dims.y),1e-6);
  let dhdx=(textureLoad(terrainHeights,vec2i(xp,cell.y),0).x-textureLoad(terrainHeights,vec2i(xm,cell.y),0).x)*h/(f32(xp-xm)*dx+1e-6);
  let dhdz=(textureLoad(terrainHeights,vec2i(cell.x,zp),0).x-textureLoad(terrainHeights,vec2i(cell.x,zm),0).x)*h/(f32(zp-zm)*dz+1e-6);
  let normal=normalize(vec3f(-dhdx,1,-dhdz)); return vec4f(normal,dot(normal,vec3f(position.x,center,position.z)));
}
fn solveBodyPair(aIndex: u32,bIndex: u32) {
  var a=bodies[aIndex];var b=bodies[bIndex];let rho=max(params.step.y,1e-9);
  let inverseA=a.inverseMassInertia.x/rho;let inverseB=b.inverseMassInertia.x/rho;let inverseTotal=inverseA+inverseB;
  if(inverseTotal<=0.0){return;}
  let delta=b.positionShape.xyz-a.positionShape.xyz;let distance=length(delta);let normal=select(vec3f(1,0,0),delta/distance,distance>1e-8);
  let radiusA=a.dimensions.w;let radiusB=b.dimensions.w;let penetration=radiusA+radiusB-distance;
  if(penetration<=0.0){return;}
  a.positionShape=vec4f(a.positionShape.xyz-normal*penetration*inverseA/inverseTotal,a.positionShape.w);
  b.positionShape=vec4f(b.positionShape.xyz+normal*penetration*inverseB/inverseTotal,b.positionShape.w);
  let armA=normal*radiusA;let armB=-normal*radiusB;var relative=velocityAt(b,armB)-velocityAt(a,armA);let normalSpeed=dot(relative,normal);
  if(normalSpeed<0.0){
    let restitution=select(0.0,min(a.angularMomentumRestitution.w,b.angularMomentumRestitution.w),-normalSpeed>0.5);
    let denominator=max(inverseTotal+angularTerm(a,armA,normal)+angularTerm(b,armB,normal),1e-9);
    let normalMagnitude=-(1.0+restitution)*normalSpeed/denominator;
    applyImpulse(&a,-normal*normalMagnitude,armA);applyImpulse(&b,normal*normalMagnitude,armB);
    relative=velocityAt(b,armB)-velocityAt(a,armA);let tangentVelocity=relative-normal*dot(relative,normal);let tangentSpeed=length(tangentVelocity);
    if(tangentSpeed>1e-8){let tangent=tangentVelocity/tangentSpeed;let friction=sqrt(max(0.0,a.material.x*b.material.x));let tangentDenominator=max(inverseTotal+angularTerm(a,armA,tangent)+angularTerm(b,armB,tangent),1e-9);let magnitude=clamp(-tangentSpeed/tangentDenominator,-friction*normalMagnitude,friction*normalMagnitude);applyImpulse(&a,-tangent*magnitude,armA);applyImpulse(&b,tangent*magnitude,armB);}
  }
  bodies[aIndex]=a;bodies[bIndex]=b;
}
fn publish(index: u32) {
  let body=bodies[index];let shape=i32(round(body.positionShape.w));let d=body.dimensions.xyz;
  var half=vec3f(d.x,.5*d.y,d.x);if(shape==0){half=vec3f(d.x);}else if(shape==1){half=.5*d;}
  var color=vec3f(.95,.63,.29);if(shape==1){color=vec3f(.48,.66,.96);}else if(shape==2){color=vec3f(.84,.42,.48);}else if(shape==3){color=vec3f(.66,.52,.92);}
  renderBodies[index]=RenderBody(vec4f(body.positionShape.xyz,body.dimensions.w),vec4f(half,body.positionShape.w),body.orientation,vec4f(color,body.material.w));
}
fn motionQuaternionXyzw(qWxyz:vec4f)->vec4f{return vec4f(qWxyz.yzw,qWxyz.x);}
fn motionTransformMatches(record:SvoPrimitiveMotionRecord,body:RigidBody)->bool{
  let positionMatches=distance(record.currentPositionDt.xyz,body.positionShape.xyz)<=1e-6;
  let oldQ=svoPrimitiveMotionQuaternionNormalize(record.currentOrientation);let bodyQ=svoPrimitiveMotionQuaternionNormalize(motionQuaternionXyzw(body.orientation));
  return positionMatches&&abs(dot(oldQ,bodyQ))>=1.0-1e-6;
}
fn motionMaterialId(shape:i32)->u32{
  if(shape==0){return ${VOXEL_MATERIAL_IDS.sphere}u;}if(shape==1){return ${VOXEL_MATERIAL_IDS.box}u;}if(shape==2){return ${VOXEL_MATERIAL_IDS.capsule}u;}return ${VOXEL_MATERIAL_IDS.cylinder}u;
}
fn publishMotion(index:u32,previousBody:RigidBody,currentBody:RigidBody,dt:f32){
  let old=rigidMotion[index];let generation=bitcast<u32>(currentBody.material.y);let previousGeneration=old.publication.x;
  let currentQ=svoPrimitiveMotionQuaternionNormalize(motionQuaternionXyzw(currentBody.orientation));var previousQ=svoPrimitiveMotionQuaternionNormalize(motionQuaternionXyzw(previousBody.orientation));var flags=0u;
  if(dot(previousQ,currentQ)<0.0){previousQ=-previousQ;flags|=SVO_PRIMITIVE_MOTION_SHORTEST_FLIP;}
  let deltaPosition=currentBody.positionShape.xyz-previousBody.positionShape.xyz;let rotationDot=clamp(abs(dot(previousQ,currentQ)),0.0,1.0);let angularDisplacement=2.0*acos(rotationDot);let radius=max(currentBody.dimensions.w,1e-6);let maximumDisplacement=length(deltaPosition)+2.0*radius*sin(.5*angularDisplacement);let motionLimit=min(.5,2.0*max(params.coupling.w,1e-6));
  let generationContinuous=generation!=0u&&generation==previousGeneration;let revisionContinuous=generationContinuous&&motionTransformMatches(old,previousBody);let teleport=maximumDisplacement>motionLimit;let valid=dt>1e-8&&generationContinuous&&revisionContinuous&&!teleport;
  if(valid){flags|=SVO_PRIMITIVE_MOTION_VALID;}if(length(deltaPosition)<=1e-8&&angularDisplacement<=1e-8){flags|=SVO_PRIMITIVE_MOTION_STATIC;}if(revisionContinuous){flags|=SVO_PRIMITIVE_MOTION_REVISION_CONTINUOUS;}if(generationContinuous){flags|=SVO_PRIMITIVE_MOTION_GENERATION_CONTINUOUS;}if(teleport){flags|=SVO_PRIMITIVE_MOTION_TELEPORT;}
  let previousRevision=old.identityRevision.z;let currentRevision=previousRevision+1u;let linearVelocity=select(vec3f(0.0),currentBody.linearVelocity.xyz,valid);let angularVelocity=select(vec3f(0.0),currentBody.angularVelocity.xyz,valid);let shape=i32(round(currentBody.positionShape.w));
  rigidMotion[index]=SvoPrimitiveMotionRecord(vec4f(currentBody.positionShape.xyz,dt),vec4f(previousBody.positionShape.xyz,radius),currentQ,previousQ,vec4f(linearVelocity,maximumDisplacement),vec4f(angularVelocity,angularDisplacement),vec4u(index,(index<<16u)|motionMaterialId(shape),currentRevision,previousRevision),vec4u(generation,previousGeneration,flags,bitcast<u32>(motionLimit)));
}
@compute @workgroup_size(1)
fn pickRigidBody(@builtin(global_invocation_id) id: vec3u) {
  if(any(id!=vec3u(0))){return;}
  let count=u32(round(pickParams.originCount.w));
  let origin=pickParams.originCount.xyz;
  let direction=safeNormalize(pickParams.direction.xyz,vec3f(0,0,-1));
  var bestIndex=0xffffffffu;
  var bestDistance=1e30;
  for(var index=0u;index<12u;index++){
    if(index>=count){break;}
    let body=bodies[index];
    let relative=origin-body.positionShape.xyz;
    let projected=dot(relative,direction);
    let discriminant=projected*projected-(dot(relative,relative)-body.dimensions.w*body.dimensions.w);
    if(discriminant<0.0){continue;}
    let root=sqrt(discriminant);
    let nearDistance=-projected-root;
    let farDistance=-projected+root;
    let distance=select(farDistance,nearDistance,nearDistance>0.0);
    if(distance>0.0&&distance<bestDistance){bestDistance=distance;bestIndex=index;}
  }
  if(bestIndex==0xffffffffu){pickResult=PickResult(bestIndex,0u,0.0,0.0,vec4f(0),vec4f(1,0,0,0));return;}
  let body=bodies[bestIndex];
  pickResult=PickResult(bestIndex,1u,bestDistance,0.0,vec4f(body.positionShape.xyz,1),body.orientation);
}
@compute @workgroup_size(1)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
  if(any(id!=vec3u(0))){return;}let count=u32(round(params.step.z));let dt=params.step.x;let rho=max(params.step.y,1e-9);let snapshots=max(params.step.w,1.0);
  var previousBodies:array<RigidBody,12>;for(var index=0u;index<12u;index++){previousBodies[index]=bodies[index];}
  for(var index=0u;index<12u;index++){if(index>=count){break;}var body=bodies[index];
    if(body.material.z>0.5){let base=index*12u;let impulse=vec3f(f32(atomicLoad(&exchange[base])),f32(atomicLoad(&exchange[base+1u])),f32(atomicLoad(&exchange[base+2u])))*1e-6;let angularImpulse=vec3f(f32(atomicLoad(&exchange[base+3u])),f32(atomicLoad(&exchange[base+4u])),f32(atomicLoad(&exchange[base+5u])))*1e-6;let wet=f32(atomicLoad(&exchange[base+6u]))/65536.0/snapshots;let weighted=vec3f(f32(atomicLoad(&exchange[base+7u])),f32(atomicLoad(&exchange[base+8u])),f32(atomicLoad(&exchange[base+9u])))*1e-4/snapshots;let meanVelocity=select(vec3f(0),weighted/wet,wet>1e-8);let displaced=max(0.0,wet*params.coupling.x);let scaledInverseMass=body.inverseMassInertia.x;let mass=select(1e30,rho/scaledInverseMass,scaledInverseMass>0.0);let immersed=clamp(displaced/max(bodyVolume(body),1e-9),0.0,1.0);let relative=body.linearVelocity.xyz-meanVelocity;let speed=length(relative);let drag=-.5*rho*params.coupling.y*3.141592653589793*body.dimensions.w*body.dimensions.w*immersed*speed*relative;let buoyancy=-rho*displaced*params.gravity.xyz;let added=params.coupling.z*rho*displaced;let acceleration=(mass*params.gravity.xyz+impulse/max(dt,1e-8)+drag+buoyancy)/max(mass+added,1e-8);body.linearVelocity=vec4f(body.linearVelocity.xyz+acceleration*dt,body.linearVelocity.w);body.angularMomentumRestitution=vec4f(body.angularMomentumRestitution.xyz+angularImpulse,body.angularMomentumRestitution.w);body.positionShape=vec4f(body.positionShape.xyz+body.linearVelocity.xyz*dt,body.positionShape.w);body.angularVelocity=vec4f(inverseInertia(body,body.angularMomentumRestitution.xyz),body.angularVelocity.w);let derivative=qMultiply(vec4f(0,body.angularVelocity.xyz),body.orientation);body.orientation=normalize(body.orientation+.5*dt*derivative);bodies[index]=body;}
  }
  for(var iteration=0u;iteration<6u;iteration++){for(var index=0u;index<12u;index++){if(index>=count){break;}var body=bodies[index];planeContact(&body,vec3f(1,0,0),-.5*params.container.x);planeContact(&body,vec3f(-1,0,0),-.5*params.container.x);planeContact(&body,vec3f(0,0,1),-.5*params.container.z);planeContact(&body,vec3f(0,0,-1),-.5*params.container.z);planeContact(&body,vec3f(0,1,0),0);if(params.terrain.y>.5){planeContact(&body,vec3f(0,-1,0),-params.container.y);}let terrain=terrainPlane(body.positionShape.xyz);planeContact(&body,terrain.xyz,terrain.w);bodies[index]=body;}for(var a=0u;a<12u;a++){if(a>=count){break;}for(var b=a+1u;b<12u;b++){if(b>=count){break;}solveBodyPair(a,b);}}}
  for(var index=0u;index<12u;index++){if(index<count){publish(index);publishMotion(index,previousBodies[index],bodies[index],dt);}else{renderBodies[index]=RenderBody(vec4f(0),vec4f(0),vec4f(1,0,0,0),vec4f(0));rigidMotion[index]=SvoPrimitiveMotionRecord();}}
}
`;

const shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;

/** Authoritative GPU rigid state. CPU writes only explicit reset/edit/drag commands. */
export class WebGPURigidBodySystem {
  readonly stateBuffer: GPUBuffer;
  readonly renderBuffer: GPUBuffer;
  /** GPU-authored 128-byte records used by temporal reprojection and swept preactivation. */
  readonly motionBuffer: GPUBuffer;
  private readonly stateScratch: GPUBuffer;
  private readonly renderScratch: GPUBuffer;
  private readonly motionScratch: GPUBuffer;
  private readonly paramsBuffer: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly pickPipeline: GPUComputePipeline;
  private readonly pickBindGroup: GPUBindGroup;
  private readonly pickParamsBuffer: GPUBuffer;
  private readonly pickResultBuffer: GPUBuffer;
  private bodyIds: string[] = [];
  private structuralSignatures: string[] = [];
  private authoredTransformSignatures: string[] = [];
  private commandSignatures: string[] = [];
  private bodyCount = 0;
  private selectedIndex = -1;
  private motionGenerations: number[] = [];

  constructor(private readonly device: GPUDevice, private readonly scene: SceneDescription, readonly exchangeBuffer: GPUBuffer, terrainTexture: GPUTexture) {
    this.stateBuffer = device.createBuffer({ label: "GPU authoritative rigid-body state", size: GPU_RIGID_STATE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.renderBuffer = device.createBuffer({ label: "GPU rigid-body render records", size: GPU_RIGID_RENDER_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.motionBuffer = device.createBuffer({ label: "GPU rigid primitive motion sidecar", size: GPU_RIGID_MOTION_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.stateScratch = device.createBuffer({ label: "GPU rigid-body roster scratch", size: GPU_RIGID_STATE_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.renderScratch = device.createBuffer({ label: "GPU rigid render roster scratch", size: GPU_RIGID_RENDER_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.motionScratch = device.createBuffer({ label: "GPU rigid motion roster scratch", size: GPU_RIGID_MOTION_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.paramsBuffer = device.createBuffer({ label: "GPU rigid-body step parameters", size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pickParamsBuffer = device.createBuffer({ label: "GPU rigid-body pick ray", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pickResultBuffer = device.createBuffer({ label: "GPU rigid-body pick result", size: GPU_RIGID_PICK_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const shaderModule = device.createShaderModule({ label: "GPU resident rigid-body solver", code: gpuRigidBodyShader });
    this.pipeline = device.createComputePipeline({ label: "GPU resident rigid-body integrate/contact", layout: "auto", compute: { module: shaderModule, entryPoint: "integrate" } });
    this.bindGroup = device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.stateBuffer } }, { binding: 1, resource: { buffer: exchangeBuffer } },
      { binding: 2, resource: { buffer: this.renderBuffer } }, { binding: 3, resource: { buffer: this.paramsBuffer } },
      { binding: 7, resource: { buffer: this.motionBuffer } },
      { binding: 4, resource: terrainTexture.createView() }
    ] });
    this.pickPipeline = device.createComputePipeline({ label: "GPU resident rigid-body ray pick", layout: "auto", compute: { module: shaderModule, entryPoint: "pickRigidBody" } });
    this.pickBindGroup = device.createBindGroup({ layout: this.pickPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.stateBuffer } },
      { binding: 5, resource: { buffer: this.pickParamsBuffer } },
      { binding: 6, resource: { buffer: this.pickResultBuffer } }
    ] });
  }

  syncBodies(bodies: readonly RigidBodyState[]) {
    const active = bodies.slice(0, GPU_RIGID_BODY_CAPACITY);
    const ids = active.map((body) => body.description.id);
    const structural = active.map((body) => JSON.stringify([body.description.shape,body.description.dimensions_m,body.description.density_kg_m3,body.description.restitution,body.description.friction,body.description.motion]));
    const authored = active.map((body) => JSON.stringify([body.description.position_m,body.description.orientation,body.description.linearVelocity_m_s,body.description.angularVelocity_rad_s]));
    const commands = active.map((body) => JSON.stringify([body.position_m,body.orientation,body.linearVelocity_m_s,body.angularVelocity_rad_s]));
    if (JSON.stringify([ids,structural,authored,commands]) === JSON.stringify([this.bodyIds,this.structuralSignatures,this.authoredTransformSignatures,this.commandSignatures])) return;
    this.bodyCount = active.length;
    const stateStorage = new ArrayBuffer(GPU_RIGID_STATE_BYTES), state = new Float32Array(stateStorage), stateWords = new Uint32Array(stateStorage);
    const render = new Float32Array(GPU_RIGID_BODY_CAPACITY * GPU_RIGID_RENDER_FLOATS);
    const nextMotionGenerations = ids.map((id, index) => {
      const previous = this.bodyIds.indexOf(id);
      if (previous < 0) return 1;
      const discontinuous = structural[index] !== this.structuralSignatures[previous]
        || authored[index] !== this.authoredTransformSignatures[previous]
        || commands[index] !== this.commandSignatures[previous];
      const generation = this.motionGenerations[previous] || 1;
      if (!discontinuous) return generation;
      const next = (generation + 1) >>> 0;
      return next === 0 ? 1 : next;
    });
    const palette = [[.95,.63,.29],[.48,.66,.96],[.84,.42,.48],[.66,.52,.92]];
    active.forEach((body, index) => {
      const o = index * GPU_RIGID_STATE_FLOATS, d = body.description.dimensions_m, q = body.orientation, shape = shapeIndex[body.description.shape];
      const rho = this.scene.fluid.density_kg_m3;
      const radius = body.description.shape === "box" ? Math.hypot(d.x,d.y,d.z)/2 : body.description.shape === "sphere" ? d.x : body.description.shape === "cylinder" ? Math.hypot(d.x,d.y/2) : d.x+d.y/2;
      state.set([body.position_m.x,body.position_m.y,body.position_m.z,shape,d.x,d.y,d.z,radius,q.w,q.x,q.y,q.z,body.linearVelocity_m_s.x,body.linearVelocity_m_s.y,body.linearVelocity_m_s.z,body.inverseMass_kg*rho,body.angularVelocity_rad_s.x,body.angularVelocity_rad_s.y,body.angularVelocity_rad_s.z,body.description.density_kg_m3,body.inverseMass_kg*rho,body.inverseInertiaBody_kg_m2.x*rho,body.inverseInertiaBody_kg_m2.y*rho,body.inverseInertiaBody_kg_m2.z*rho,body.angularMomentum_kg_m2_s.x,body.angularMomentum_kg_m2_s.y,body.angularMomentum_kg_m2_s.z,body.description.restitution,body.description.friction,0,body.description.motion === "static" ? 0 : 1,0],o);
      stateWords[o+29]=nextMotionGenerations[index];
      const half = body.description.shape === "box" ? [d.x/2,d.y/2,d.z/2] : body.description.shape === "sphere" ? [d.x,d.x,d.x] : [d.x,d.y/2,d.x];
      render.set([body.position_m.x,body.position_m.y,body.position_m.z,state[o+7],half[0],half[1],half[2],shape,q.w,q.x,q.y,q.z,...palette[shape],0],index*GPU_RIGID_RENDER_FLOATS);
    });
    if (this.bodyIds.length === 0) {
      this.device.queue.writeBuffer(this.stateBuffer,0,state);this.device.queue.writeBuffer(this.renderBuffer,0,render);
    } else {
      const rosterChanged = ids.length !== this.bodyIds.length || ids.some((id,index) => id !== this.bodyIds[index]);
      if (rosterChanged) {
        const encoder=this.device.createCommandEncoder({label:"Compact GPU resident rigid-body roster"});
        ids.forEach((id,index) => { const previous=this.bodyIds.indexOf(id); if(previous<0)return; encoder.copyBufferToBuffer(this.stateBuffer,previous*GPU_RIGID_STATE_FLOATS*4,this.stateScratch,index*GPU_RIGID_STATE_FLOATS*4,GPU_RIGID_STATE_FLOATS*4);encoder.copyBufferToBuffer(this.renderBuffer,previous*GPU_RIGID_RENDER_FLOATS*4,this.renderScratch,index*GPU_RIGID_RENDER_FLOATS*4,GPU_RIGID_RENDER_FLOATS*4);encoder.copyBufferToBuffer(this.motionBuffer,previous*SVO_PRIMITIVE_MOTION_STRIDE_BYTES,this.motionScratch,index*SVO_PRIMITIVE_MOTION_STRIDE_BYTES,SVO_PRIMITIVE_MOTION_STRIDE_BYTES); });
        encoder.copyBufferToBuffer(this.stateScratch,0,this.stateBuffer,0,GPU_RIGID_STATE_BYTES);encoder.copyBufferToBuffer(this.renderScratch,0,this.renderBuffer,0,GPU_RIGID_RENDER_BYTES);encoder.copyBufferToBuffer(this.motionScratch,0,this.motionBuffer,0,GPU_RIGID_MOTION_BYTES);this.device.queue.submit([encoder.finish()]);
      }
      active.forEach((_body,index) => {
        const previous=this.bodyIds.indexOf(ids[index]);
        const stateOffset=index*GPU_RIGID_STATE_FLOATS,renderOffset=index*GPU_RIGID_RENDER_FLOATS;
        if(previous<0 || authored[index] !== this.authoredTransformSignatures[previous] || (!rosterChanged && commands[index] !== this.commandSignatures[previous] && structural[index] === this.structuralSignatures[previous])) {
          this.device.queue.writeBuffer(this.stateBuffer,stateOffset*4,state.subarray(stateOffset,stateOffset+GPU_RIGID_STATE_FLOATS));this.device.queue.writeBuffer(this.renderBuffer,renderOffset*4,render.subarray(renderOffset,renderOffset+GPU_RIGID_RENDER_FLOATS));return;
        }
        if(structural[index] !== this.structuralSignatures[previous]) {
          this.device.queue.writeBuffer(this.stateBuffer,(stateOffset+3)*4,state.subarray(stateOffset+3,stateOffset+8));
          this.device.queue.writeBuffer(this.stateBuffer,(stateOffset+15)*4,state.subarray(stateOffset+15,stateOffset+16));
          this.device.queue.writeBuffer(this.stateBuffer,(stateOffset+19)*4,state.subarray(stateOffset+19,stateOffset+24));
          this.device.queue.writeBuffer(this.stateBuffer,(stateOffset+27)*4,state.subarray(stateOffset+27,stateOffset+31));
          this.device.queue.writeBuffer(this.renderBuffer,(renderOffset+3)*4,render.subarray(renderOffset+3,renderOffset+8));
          this.device.queue.writeBuffer(this.renderBuffer,(renderOffset+12)*4,render.subarray(renderOffset+12,renderOffset+15));
        }
      });
    }
    this.bodyIds=ids;this.structuralSignatures=structural;this.authoredTransformSignatures=authored;this.commandSignatures=commands;this.motionGenerations=nextMotionGenerations;
  }

  setSelectedIndex(index: number) {
    const next = index >= 0 && index < this.bodyCount ? index : -1;
    if (next === this.selectedIndex) return;
    const write = (bodyIndex: number, selected: number) => { if (bodyIndex < 0) return; this.device.queue.writeBuffer(this.stateBuffer, bodyIndex * GPU_RIGID_STATE_FLOATS * 4 + 31 * 4, new Float32Array([selected])); this.device.queue.writeBuffer(this.renderBuffer, bodyIndex * GPU_RIGID_RENDER_FLOATS * 4 + 15 * 4, new Float32Array([selected])); };
    write(this.selectedIndex,0);write(next,1);this.selectedIndex=next;
  }

  /** A bounded, user-triggered readback used only to begin mouse interaction. */
  async pick(origin: Vec3, direction: Vec3): Promise<GPURigidBodyPick | undefined> {
    if (this.bodyCount === 0) return undefined;
    this.device.queue.writeBuffer(this.pickParamsBuffer,0,new Float32Array([origin.x,origin.y,origin.z,this.bodyCount,direction.x,direction.y,direction.z,0]));
    const readback=this.device.createBuffer({label:"GPU rigid-body pick readback",size:GPU_RIGID_PICK_BYTES,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder=this.device.createCommandEncoder({label:"Pick GPU resident rigid body"});
    const pass=encoder.beginComputePass({label:"Ray-pick GPU resident rigid bodies"});pass.setPipeline(this.pickPipeline);pass.setBindGroup(0,this.pickBindGroup);pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(this.pickResultBuffer,0,readback,0,GPU_RIGID_PICK_BYTES);this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const bytes=readback.getMappedRange(),words=new Uint32Array(bytes),values=new Float32Array(bytes);
      if(words[1]===0||words[0]>=this.bodyCount)return undefined;
      return {bodyIndex:words[0],distance_m:values[2],position_m:{x:values[4],y:values[5],z:values[6]},orientation:{w:values[8],x:values[9],y:values[10],z:values[11]}};
    } catch { return undefined; }
    finally { if(readback.mapState==="mapped")readback.unmap();readback.destroy(); }
  }

  encode(encoder: GPUCommandEncoder, dt_s: number, cellVolume_m3: number, snapshotCount = 1, cellHeight_m = 1) {
    const c=this.scene.container,g=this.scene.fluid.gravity_m_s2;
    this.device.queue.writeBuffer(this.paramsBuffer,0,new Float32Array([dt_s,this.scene.fluid.density_kg_m3,this.bodyCount,Math.max(1,snapshotCount),g.x,g.y,g.z,0,c.width_m,c.height_m,c.depth_m,sceneHasTerrain(this.scene)?1:0,cellHeight_m,c.top === "closed"?1:0,0,0,cellVolume_m3,.9,.5,Math.max(cellHeight_m,1e-6)]));
    const pass=encoder.beginComputePass({label:"GPU resident rigid-body integration and contacts"});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.bindGroup);pass.dispatchWorkgroups(1);pass.end();
  }

  destroy() { this.stateBuffer.destroy();this.renderBuffer.destroy();this.motionBuffer.destroy();this.stateScratch.destroy();this.renderScratch.destroy();this.motionScratch.destroy();this.paramsBuffer.destroy();this.pickParamsBuffer.destroy();this.pickResultBuffer.destroy(); }
}
