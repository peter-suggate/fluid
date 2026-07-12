import type { SceneDescription } from "./model";
import { damBreakFractions } from "./initial-fluid";
import type { RigidBodyState } from "./rigid-body";

export type GPUQuality = "balanced" | "high" | "ultra";

export interface GPUEulerianInfo {
  nx: number;
  ny: number;
  nz: number;
  cellCount: number;
  cellSize_m: number;
  pressureIterations: number;
  allocatedBytes: number;
  quality: GPUQuality;
  volumeCellSum?: number;
  front_m?: number;
  maxSpeed_m_s?: number;
  encodedSteps?: number;
  gpuStep_ms?: number;
  initialVolumeCellSum?:number;
  volumeDrift?:number;
  rawVolumeDrift?:number;
  gpuTimings?: {
    advection_ms: number;
    pressure_ms: number;
    projection_ms: number;
    rigidCoupling_ms: number;
    diagnostics_ms: number;
    overhead_ms: number;
    total_ms: number;
  };
}

export interface GPURigidLoad {
  bodyId: string;
  force_N: { x: number; y: number; z: number };
  torque_N_m: { x: number; y: number; z: number };
  displacedVolume_m3: number;
}

const targetCells: Record<GPUQuality, number> = { balanced: 110_000, high: 500_000, ultra: 1_200_000 };

const computeShader = /* wgsl */ `
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
  boundary: vec4f,
}
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var pressureIn: texture_3d<f32>;
@group(0) @binding(3) var pressureOut: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var volumeIn: texture_3d<f32>;
@group(0) @binding(5) var volumeOut: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var heightIn: texture_2d<f32>;
@group(0) @binding(8) var heightOut: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var<storage,read_write> reductions:array<atomic<u32>,4>;
struct RigidBody {
  positionShape: vec4f,
  dimensions: vec4f,
  orientation: vec4f,
  linearVelocity: vec4f,
  angularVelocity: vec4f,
}
@group(0) @binding(10) var<storage,read> rigidBodies:array<RigidBody,12>;
@group(0) @binding(11) var<storage,read_write> rigidExchange:array<atomic<i32>>;

fn dims() -> vec3i { return vec3i(textureDimensions(volumeIn)); }
fn valid(p: vec3i) -> bool { let d=dims(); return all(p >= vec3i(0)) && all(p < d); }
fn clampCell(p: vec3i) -> vec3i { return clamp(p, vec3i(0), dims()-vec3i(1)); }
fn volume(p: vec3i) -> f32 { if (!valid(p)) { return 0.0; } return textureLoad(volumeIn,p,0).x; }
fn velocity(p: vec3i) -> vec3f { return textureLoad(velocityIn,clampCell(p),0).xyz; }
fn faceVelocity(p:vec3i)->vec3f{if(!valid(p)){return vec3f(0.0);}return textureLoad(velocityIn,p,0).xyz;}
fn liquid(p:vec3i)->bool{return valid(p)&&volume(p)>=0.5;}
fn pressureValue(p:vec3i)->f32{return textureLoad(pressureIn,clampCell(p),0).x;}
fn interfaceFraction(a:f32,b:f32)->f32{
  // Distance from the liquid cell centre to alpha=0.5 along a grid edge.
  return clamp((a-0.5)/max(abs(a-b),1e-6),0.05,1.0);
}
fn sampleVolume(p:vec3f)->f32{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(dims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);let c000=volume(b);let c100=volume(b+vec3i(1,0,0));let c010=volume(b+vec3i(0,1,0));let c110=volume(b+vec3i(1,1,0));let c001=volume(b+vec3i(0,0,1));let c101=volume(b+vec3i(1,0,1));let c011=volume(b+vec3i(0,1,1));let c111=volume(b+vec3i(1,1,1));return mix(mix(mix(c000,c100,f.x),mix(c010,c110,f.x),f.y),mix(mix(c001,c101,f.x),mix(c011,c111,f.x),f.y),f.z);
}
fn sampleVelocityComponent(p:vec3f,component:u32)->f32{
  var offset=vec3f(0.5);offset[component]=1.0;let q=clamp(p-offset,vec3f(0.0),vec3f(dims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);
  let c000=velocity(b)[component];let c100=velocity(b+vec3i(1,0,0))[component];let c010=velocity(b+vec3i(0,1,0))[component];let c110=velocity(b+vec3i(1,1,0))[component];let c001=velocity(b+vec3i(0,0,1))[component];let c101=velocity(b+vec3i(1,0,1))[component];let c011=velocity(b+vec3i(0,1,1))[component];let c111=velocity(b+vec3i(1,1,1))[component];return mix(mix(mix(c000,c100,f.x),mix(c010,c110,f.x),f.y),mix(mix(c001,c101,f.x),mix(c011,c111,f.x),f.y),f.z);
}
fn sampleVelocity(p:vec3f)->vec3f{return vec3f(sampleVelocityComponent(p,0u),sampleVelocityComponent(p,1u),sampleVelocityComponent(p,2u));}
fn advectVelocityComponent(position:vec3f,component:u32,dt:f32,h:vec3f)->f32{
  let first=sampleVelocity(position);let midpoint=position-0.5*first*dt/h;let midpointVelocity=sampleVelocity(midpoint);return sampleVelocityComponent(position-midpointVelocity*dt/h,component);
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
fn transportVelocity(id:vec3i)->vec3f{
  var v=velocity(id);if(volume(id)>=0.01){return v;}var sum=vec3f(0.0);var weight=0.0;
  let px=volume(id+vec3i(1,0,0));let nx=volume(id-vec3i(1,0,0));let py=volume(id+vec3i(0,1,0));let ny=volume(id-vec3i(0,1,0));let pz=volume(id+vec3i(0,0,1));let nz=volume(id-vec3i(0,0,1));
  sum+=velocity(id+vec3i(1,0,0))*px+velocity(id-vec3i(1,0,0))*nx+velocity(id+vec3i(0,1,0))*py+velocity(id-vec3i(0,1,0))*ny+velocity(id+vec3i(0,0,1))*pz+velocity(id-vec3i(0,0,1))*nz;weight=px+nx+py+ny+pz+nz;if(weight>0.001){v=sum/weight;}return v;
}
fn columnHeight(x:i32,z:i32)->f32{
  let d=dims();if(x<0||x>=d.x||z<0||z>=d.z){return 0.0;}return textureLoad(heightIn,vec2i(x,z),0).x;
}
fn upwind(face:f32,negative:f32,positive:f32)->f32{return face*select(positive,negative,face>=0.0);}
fn volumeGradient(id:vec3i)->vec3f{
  let h=params.cellGravity.xyz;
  return vec3f(volume(id+vec3i(1,0,0))-volume(id-vec3i(1,0,0)),volume(id+vec3i(0,1,0))-volume(id-vec3i(0,1,0)),volume(id+vec3i(0,0,1))-volume(id-vec3i(0,0,1)))/(2.0*h);
}
fn interfaceNormal(id:vec3i)->vec3f{
  let gradient=volumeGradient(id);
  return gradient/max(length(gradient),1e-6);
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
fn receiverScale(id:vec3i,dt:f32)->f32{return min(1.0,(1.0-volume(id))/max(inwardFlux(id,dt),1e-9));}
fn limitedVolumeFlux(id:vec3i,axis:u32,dt:f32)->f32{
  let offset=select(select(vec3i(0,0,1),vec3i(0,1,0),axis==1u),vec3i(1,0,0),axis==0u);
  let neighbor=id+offset;let flux=rawVolumeFlux(id,axis,dt);
  if(flux>=0.0){return flux*min(donorScale(id,dt),receiverScale(neighbor,dt));}
  return flux*min(donorScale(neighbor,dt),receiverScale(id,dt));
}
fn advectedVolume(id:vec3i,dt:f32)->f32{
  let centre=volume(id);
  let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  let fxp=limitedVolumeFlux(id,0u,dt);let fxm=limitedVolumeFlux(id-ex,0u,dt);
  let fyp=limitedVolumeFlux(id,1u,dt);let fym=limitedVolumeFlux(id-ey,1u,dt);
  let fzp=limitedVolumeFlux(id,2u,dt);let fzm=limitedVolumeFlux(id-ez,2u,dt);
  return centre-(fxp-fxm+fyp-fym+fzp-fzm);
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


@compute @workgroup_size(4,4,4)
fn advect(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  let dt=params.dimsDt.w; let h=params.cellGravity.xyz;
  let cell=vec3f(id);let oldV=transportVelocity(id);var v=vec3f(advectVelocityComponent(cell+vec3f(1.0,0.5,0.5),0u,dt,h),advectVelocityComponent(cell+vec3f(0.5,1.0,0.5),1u,dt,h),advectVelocityComponent(cell+vec3f(0.5,0.5,1.0),2u,dt,h));let phi=volume(id);if(phi>0.0){let delta=pow(h.x*h.y*h.z,1.0/3.0);let molecular=params.physical.y/params.physical.x;let eddy=pow(0.17*delta,2.0)*strainMagnitude(id);v+=dt*(molecular+eddy)*velocityLaplacian(id);}
  // Body force lives on faces. A face participates whenever liquid exists on
  // either side; this is the same rule during impact and at equilibrium.
  if (liquid(id)||liquid(id+vec3i(0,1,0))) { v.y += params.cellGravity.w*dt; }
  // Balanced-force CSF: pressure and capillary acceleration use the same
  // positive-face locations and alpha differences.
  let sigmaOverRho=params.boundary.x/params.physical.x;
  if(valid(id+vec3i(1,0,0))){v.x+=dt*sigmaOverRho*0.5*(curvatureAt(id)+curvatureAt(id+vec3i(1,0,0)))*(volume(id+vec3i(1,0,0))-phi)/h.x;}
  if(valid(id+vec3i(0,1,0))){v.y+=dt*sigmaOverRho*0.5*(curvatureAt(id)+curvatureAt(id+vec3i(0,1,0)))*(volume(id+vec3i(0,1,0))-phi)/h.y;}
  if(valid(id+vec3i(0,0,1))){v.z+=dt*sigmaOverRho*0.5*(curvatureAt(id)+curvatureAt(id+vec3i(0,0,1)))*(volume(id+vec3i(0,0,1))-phi)/h.z;}
  let d=dims();
  if (id.x==d.x-1) { v.x=0.0; }
  if (id.y==d.y-1) { v.y=0.0; }
  if (id.z==d.z-1) { v.z=0.0; }
  textureStore(velocityOut,id,vec4f(v,0.0));
  textureStore(volumeOut,id,vec4f(advectedVolume(id,dt),0.0,0.0,0.0));
  textureStore(pressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(8,8,1)
fn buildHeight(@builtin(global_invocation_id) gid:vec3u){let d=dims();if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}var total=0.0;for(var y:i32=0;y<d.y;y+=1){total+=volume(vec3i(i32(gid.x),y,i32(gid.y)))*params.cellGravity.y;}textureStore(heightOut,vec2i(gid.xy),vec4f(total));}

fn divergenceAt(id: vec3i) -> f32 {
  let h=params.cellGravity.xyz;
  return (faceVelocity(id).x-faceVelocity(id-vec3i(1,0,0)).x)/h.x
       + (faceVelocity(id).y-faceVelocity(id-vec3i(0,1,0)).y)/h.y
       + (faceVelocity(id).z-faceVelocity(id-vec3i(0,0,1)).z)/h.z;
}

fn curvatureAt(id:vec3i)->f32{
  let h=params.cellGravity.xyz;
  return -((interfaceNormal(id+vec3i(1,0,0)).x-interfaceNormal(id-vec3i(1,0,0)).x)/(2.0*h.x)+(interfaceNormal(id+vec3i(0,1,0)).y-interfaceNormal(id-vec3i(0,1,0)).y)/(2.0*h.y)+(interfaceNormal(id+vec3i(0,0,1)).z-interfaceNormal(id-vec3i(0,0,1)).z)/(2.0*h.z));
}

fn stencilCoefficient(id:vec3i,neighbor:vec3i,axis:u32)->f32{
  if(!valid(neighbor)){return 0.0;}
  let h=params.cellGravity.xyz[axis];
  if(liquid(neighbor)){return 1.0/(h*h);}
  return 1.0/(interfaceFraction(volume(id),volume(neighbor))*h*h);
}

fn stencilPressure(id:vec3i,neighbor:vec3i,axis:u32)->f32{
  if(!valid(neighbor)||!liquid(neighbor)){return 0.0;}
  return stencilCoefficient(id,neighbor,axis)*pressureValue(neighbor);
}

@compute @workgroup_size(4,4,4)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  if (!liquid(id)) { textureStore(pressureOut,id,vec4f(0.0)); return; }
  let old=textureLoad(pressureIn,id,0).x;let ex=vec3i(1,0,0);let ey=vec3i(0,1,0);let ez=vec3i(0,0,1);
  let diagonal=stencilCoefficient(id,id-ex,0u)+stencilCoefficient(id,id+ex,0u)+stencilCoefficient(id,id-ey,1u)+stencilCoefficient(id,id+ey,1u)+stencilCoefficient(id,id-ez,2u)+stencilCoefficient(id,id+ez,2u);
  let sum=stencilPressure(id,id-ex,0u)+stencilPressure(id,id+ex,0u)+stencilPressure(id,id-ey,1u)+stencilPressure(id,id+ey,1u)+stencilPressure(id,id-ez,2u)+stencilPressure(id,id+ez,2u);
  let rhs=params.physical.x*divergenceAt(id)/params.dimsDt.w;
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
  textureStore(velocityOut,id,vec4f(v,0.0)); textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));
}

// Brinkman-style immersed boundary: drive wet cells inside each moving primitive
// toward the local solid velocity and accumulate the exact opposite impulse.
@compute @workgroup_size(4,4,4)
fn coupleRigid(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);if(!valid(id)){return;}let phi=volume(id);var v=velocity(id);let h=params.cellGravity.xyz;
  let world=vec3f(-0.5*params.container.x+(f32(id.x)+0.5)*h.x,(f32(id.y)+0.5)*h.y,-0.5*params.container.z+(f32(id.z)+0.5)*h.z);
  let bodyCount=u32(round(params.boundary.z));let cellMass=params.physical.x*h.x*h.y*h.z*phi;let blend=clamp(45.0*params.dimsDt.w,0.0,1.0);
  for(var bodyIndex:u32=0u;bodyIndex<12u;bodyIndex+=1u){if(bodyIndex>=bodyCount){break;}let body=rigidBodies[bodyIndex];if(!insideRigid(body,world)){continue;}
    let arm=world-body.positionShape.xyz;let solidVelocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,arm);let fluidImpulse=cellMass*(solidVelocity-v)*blend;v+=fluidImpulse/max(cellMass,1e-9);
    let reaction=-fluidImpulse;let torque=cross(arm,reaction);let base=bodyIndex*8u;
    atomicAdd(&rigidExchange[base],i32(round(reaction.x*1000000.0)));atomicAdd(&rigidExchange[base+1u],i32(round(reaction.y*1000000.0)));atomicAdd(&rigidExchange[base+2u],i32(round(reaction.z*1000000.0)));
    atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*1000000.0)));atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*1000000.0)));atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*1000000.0)));
    atomicAdd(&rigidExchange[base+6u],i32(round(phi*65536.0)));break;
  }
  textureStore(velocityOut,id,vec4f(v,0.0));textureStore(volumeOut,id,vec4f(phi));
}

@compute @workgroup_size(4,4,4)
fn reduceDiagnostics(@builtin(global_invocation_id) gid:vec3u){let id=vec3i(gid);if(!valid(id)){return;}let phi=volume(id);atomicAdd(&reductions[0],u32(clamp(phi,0.0,1.0)*2048.0+0.5));if(phi>0.5){atomicMax(&reductions[1],u32(id.x+1));}let speed=length(faceVelocity(id));atomicMax(&reductions[2],bitcast<u32>(speed));}
`;

export class WebGPUEulerianSolver {
  readonly info: GPUEulerianInfo;
  private velocityA: GPUTexture; private velocityB: GPUTexture;
  private pressureA: GPUTexture; private pressureB: GPUTexture;
  private volumeA: GPUTexture; private volumeB: GPUTexture;
  private heightA: GPUTexture; private heightB: GPUTexture;
  private params: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;
  private advectPipeline: GPUComputePipeline; private jacobiPipeline: GPUComputePipeline; private projectPipeline: GPUComputePipeline; private rigidPipeline:GPUComputePipeline;private reductionPipeline:GPUComputePipeline;
  private advectGroup: GPUBindGroup; private jacobiABGroup: GPUBindGroup; private jacobiBAGroup: GPUBindGroup; private projectGroup: GPUBindGroup;private rigidGroup:GPUBindGroup; private reductionGroup:GPUBindGroup;
  private reductionBuffer:GPUBuffer;private rigidBuffer:GPUBuffer;private rigidExchangeBuffer:GPUBuffer;private querySet?:GPUQuerySet;private queryResolve?:GPUBuffer;
  private querySegments: Array<{name:keyof NonNullable<GPUEulerianInfo["gpuTimings"]>;start:number;end:number}>=[];private queryCount=0;
  private lastTime = 0;
  private readbackPending = false;
  private rigidReadbackPending = false;
  private validationChecked = false;

  constructor(private device: GPUDevice, readonly scene: SceneDescription, quality: GPUQuality, private onRigidLoads?: (loads: GPURigidLoad[]) => void) {
    const c=scene.container, target=targetCells[quality], h=Math.cbrt(c.width_m*c.height_m*c.depth_m/target);
    const nx=Math.max(8,Math.round(c.width_m/h)), ny=Math.max(8,Math.round(c.height_m/h)), nz=Math.max(8,Math.round(c.depth_m/h));
    const usage=GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST;
    const texture=(format: GPUTextureFormat)=>device.createTexture({size:[nx,ny,nz],dimension:"3d",format,usage});
    this.velocityA=texture("rgba32float"); this.velocityB=texture("rgba32float"); this.pressureA=texture("r32float"); this.pressureB=texture("r32float"); this.volumeA=texture("r32float"); this.volumeB=texture("r32float");
    this.heightA=device.createTexture({size:[nx,nz],format:"r32float",usage});this.heightB=device.createTexture({size:[nx,nz],format:"r32float",usage});
    this.params=device.createBuffer({size:80,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.reductionBuffer=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.rigidBuffer=device.createBuffer({size:12*80,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.rigidExchangeBuffer=device.createBuffer({size:12*8*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    if(device.features.has("timestamp-query")){this.querySet=device.createQuerySet({type:"timestamp",count:160});this.queryResolve=device.createBuffer({size:160*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});}
    const shaderModule=device.createShaderModule({label:"Fluid Lab GPU Eulerian kernels",code:computeShader});
    void shaderModule.getCompilationInfo().then(info=>{for(const message of info.messages)if(message.type==="error")console.error(`GPU fluid WGSL ${message.lineNum}:${message.linePos} ${message.message}`);});
    this.bindGroupLayout=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}
      ,{binding:7,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"2d"}}
      ,{binding:8,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"2d"}}
      ,{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}
      ,{binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}
      ,{binding:11,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}
    ]});
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[this.bindGroupLayout]});
    this.advectPipeline=device.createComputePipeline({layout:pipelineLayout,compute:{module:shaderModule,entryPoint:"advect"}});
    this.jacobiPipeline=device.createComputePipeline({layout:pipelineLayout,compute:{module:shaderModule,entryPoint:"jacobi"}});
    this.projectPipeline=device.createComputePipeline({layout:pipelineLayout,compute:{module:shaderModule,entryPoint:"project"}});
    this.rigidPipeline=device.createComputePipeline({layout:pipelineLayout,compute:{module:shaderModule,entryPoint:"coupleRigid"}});
    this.reductionPipeline=device.createComputePipeline({layout:pipelineLayout,compute:{module:shaderModule,entryPoint:"reduceDiagnostics"}});
    const pressureIterations=quality==="balanced"?64:quality==="high"?80:96;this.info={nx,ny,nz,cellCount:nx*ny*nz,cellSize_m:Math.max(c.width_m/nx,c.height_m/ny,c.depth_m/nz),pressureIterations,allocatedBytes:nx*ny*nz*(4*8+4*4),quality,encodedSteps:0};
    this.initializeVolume();
    this.advectGroup=this.group(this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB);
    this.jacobiABGroup=this.group(this.velocityB,this.velocityA,this.pressureA,this.pressureB,this.volumeB,this.volumeA,this.heightB,this.heightA);
    this.jacobiBAGroup=this.group(this.velocityB,this.velocityA,this.pressureB,this.pressureA,this.volumeB,this.volumeA,this.heightB,this.heightA);
    this.projectGroup=this.group(this.velocityB,this.velocityA,this.pressureA,this.pressureB,this.volumeB,this.volumeA,this.heightB,this.heightA);
    this.rigidGroup=this.group(this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightB,this.heightA);
    this.reductionGroup=this.group(this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightB,this.heightA);
  }

  get volumeTexture(){return this.volumeA;}
  private initializeVolume(){
    const {nx,ny,nz}=this.info,c=this.scene.container,data=new Float32Array(nx*ny*nz),dam=damBreakFractions(c.fillFraction);
    let initialSum=0;for(let k=0;k<nz;k++)for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){const fill=this.scene.fluid.initialCondition==="dam-break"?(i+.5)/nx<=dam.width&&(j+.5)/ny<=dam.height&&(k+.5)/nz<=dam.depth:(j+.5)/ny<=c.fillFraction;data[i+nx*(j+ny*k)]=fill?1:0;if(fill)initialSum+=1;}this.info.initialVolumeCellSum=initialSum;this.info.volumeCellSum=initialSum;this.info.volumeDrift=0;this.info.rawVolumeDrift=0;this.info.maxSpeed_m_s=0;this.info.front_m=this.scene.fluid.initialCondition==="dam-break"?-c.width_m/2+dam.width*c.width_m:c.width_m/2;
    const rowBytes=nx*4,padded=Math.ceil(rowBytes/256)*256,packed=new Uint8Array(padded*ny*nz),source=new Uint8Array(data.buffer);
    for(let k=0;k<nz;k++)for(let j=0;j<ny;j++)packed.set(source.subarray(rowBytes*(j+ny*k),rowBytes*(j+ny*k+1)),padded*(j+ny*k));
    this.device.queue.writeTexture({texture:this.volumeA},packed,{bytesPerRow:padded,rowsPerImage:ny},{width:nx,height:ny,depthOrArrayLayers:nz});
    this.device.queue.writeTexture({texture:this.volumeB},packed,{bytesPerRow:padded,rowsPerImage:ny},{width:nx,height:ny,depthOrArrayLayers:nz});
  }

  private group(velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture, pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture,heightIn:GPUTexture,heightOut:GPUTexture){return this.device.createBindGroup({layout:this.bindGroupLayout,entries:[{binding:0,resource:velocityIn.createView()},{binding:1,resource:velocityOut.createView()},{binding:2,resource:pressureIn.createView()},{binding:3,resource:pressureOut.createView()},{binding:4,resource:volumeIn.createView()},{binding:5,resource:volumeOut.createView()},{binding:6,resource:{buffer:this.params}},{binding:7,resource:heightIn.createView()},{binding:8,resource:heightOut.createView()},{binding:9,resource:{buffer:this.reductionBuffer}},{binding:10,resource:{buffer:this.rigidBuffer}},{binding:11,resource:{buffer:this.rigidExchangeBuffer}}]});}
  private dispatch(pass: GPUComputePassEncoder,pipeline:GPUComputePipeline,group:GPUBindGroup){pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(this.info.nx/4),Math.ceil(this.info.ny/4),Math.ceil(this.info.nz/4));}
  private timing(name:keyof NonNullable<GPUEulerianInfo["gpuTimings"]>){if(!this.querySet)return undefined;const segment={name,start:this.queryCount++,end:this.queryCount++};this.querySegments.push(segment);return segment;}

  advanceTo(time_s:number,bodies:RigidBodyState[]=[]){
    if(time_s<this.lastTime)return false; const delta=Math.min(this.scene.numerics.maxDt_s,time_s-this.lastTime);if(delta<1e-6)return true;this.lastTime=time_s;const c=this.scene.container,rho=this.scene.fluid.density_kg_m3,sigma=this.scene.fluid.surfaceTension_N_m,waveSpeed=Math.sqrt(Math.abs(this.scene.fluid.gravity_m_s2.y)*c.height_m*Math.max(c.fillFraction,0.01)),characteristicSpeed=Math.max(waveSpeed,this.info.maxSpeed_m_s??0,0.1),advectiveDt=0.45*this.info.cellSize_m/characteristicSpeed,capillaryDt=sigma>0?0.4*Math.sqrt(rho*this.info.cellSize_m**3/(Math.PI*sigma)):Infinity,stableDt=Math.min(advectiveDt,capillaryDt),substeps=Math.min(16,Math.max(1,Math.ceil(delta/stableDt))),dt=delta/substeps;
    const activeBodies=bodies.slice(0,12),bodyData=new Float32Array(12*20),shapeIndex={sphere:0,box:1,capsule:2,cylinder:3} as const;activeBodies.forEach((body,index)=>{const o=index*20,d=body.description.dimensions_m,q=body.orientation;bodyData.set([body.position_m.x,body.position_m.y,body.position_m.z,shapeIndex[body.description.shape],d.x,d.y,d.z,0,q.w,q.x,q.y,q.z,body.linearVelocity_m_s.x,body.linearVelocity_m_s.y,body.linearVelocity_m_s.z,0,body.angularVelocity_rad_s.x,body.angularVelocity_rad_s.y,body.angularVelocity_rad_s.z,0],o);});this.device.queue.writeBuffer(this.rigidBuffer,0,bodyData);
    this.info.encodedSteps=(this.info.encodedSteps??0)+substeps;this.device.queue.writeBuffer(this.params,0,new Float32Array([this.info.nx,this.info.ny,this.info.nz,dt,c.width_m/this.info.nx,c.height_m/this.info.ny,c.depth_m/this.info.nz,this.scene.fluid.gravity_m_s2.y,c.width_m,c.height_m,c.depth_m,0,rho,this.scene.fluid.dynamicViscosity_Pa_s,0,0,sigma,c.fluidWallMode==="no-slip"?1:0,activeBodies.length,0]));
    this.querySegments=[];this.queryCount=0;if(!this.validationChecked)this.device.pushErrorScope("validation");const encoder=this.device.createCommandEncoder({label:"GPU fluid step"}),totalTiming=this.timing("total_ms");if(totalTiming&&this.querySet){const pass=encoder.beginComputePass({timestampWrites:{querySet:this.querySet,endOfPassWriteIndex:totalTiming.start}});pass.end();}encoder.clearBuffer(this.rigidExchangeBuffer);
    for(let substep=0;substep<substeps;substep+=1){
      {const timing=this.timing("advection_ms");const pass=encoder.beginComputePass(timing&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:timing.start,endOfPassWriteIndex:timing.end}}:undefined);this.dispatch(pass,this.advectPipeline,this.advectGroup);pass.end();}
      {const timing=this.timing("pressure_ms");for(let iteration=0;iteration<this.info.pressureIterations;iteration+=1){const first=iteration===0,last=iteration===this.info.pressureIterations-1;const pass=encoder.beginComputePass(timing&&this.querySet&&(first||last)?{timestampWrites:{querySet:this.querySet,...(first?{beginningOfPassWriteIndex:timing.start}:{}),...(last?{endOfPassWriteIndex:timing.end}:{})}}:undefined);this.dispatch(pass,this.jacobiPipeline,iteration%2===0?this.jacobiABGroup:this.jacobiBAGroup);pass.end();}}
      {const timing=this.timing("projection_ms");const pass=encoder.beginComputePass(timing&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:timing.start,endOfPassWriteIndex:timing.end}}:undefined);this.dispatch(pass,this.projectPipeline,this.projectGroup);pass.end();}
      if(activeBodies.length>0){{const timing=this.timing("rigidCoupling_ms");const pass=encoder.beginComputePass(timing&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:timing.start,endOfPassWriteIndex:timing.end}}:undefined);this.dispatch(pass,this.rigidPipeline,this.rigidGroup);pass.end();}encoder.copyTextureToTexture({texture:this.velocityB},{texture:this.velocityA},[this.info.nx,this.info.ny,this.info.nz]);encoder.copyTextureToTexture({texture:this.volumeB},{texture:this.volumeA},[this.info.nx,this.info.ny,this.info.nz]);}
    }
    encoder.clearBuffer(this.reductionBuffer);{const timing=this.timing("diagnostics_ms");const pass=encoder.beginComputePass(timing&&this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:timing.start,endOfPassWriteIndex:timing.end}}:undefined);this.dispatch(pass,this.reductionPipeline,this.reductionGroup);pass.end();}if(totalTiming&&this.querySet){const pass=encoder.beginComputePass({timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:totalTiming.end}});pass.end();}if(this.querySet&&this.queryResolve&&this.queryCount>0)encoder.resolveQuerySet(this.querySet,0,this.queryCount,this.queryResolve,0);
    let exchangeReadback:GPUBuffer|undefined;if(activeBodies.length>0&&this.onRigidLoads&&!this.rigidReadbackPending){this.rigidReadbackPending=true;exchangeReadback=this.device.createBuffer({size:12*8*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});encoder.copyBufferToBuffer(this.rigidExchangeBuffer,0,exchangeReadback,0,12*8*4);}
    this.device.queue.submit([encoder.finish()]);if(exchangeReadback){const readback=exchangeReadback,elapsed=delta,cellVolume=c.width_m*c.height_m*c.depth_m/(this.info.nx*this.info.ny*this.info.nz);void readback.mapAsync(GPUMapMode.READ).then(()=>{const words=new Int32Array(readback.getMappedRange());const loads=activeBodies.map((body,index)=>{const b=index*8,impulse={x:words[b]/1e6,y:words[b+1]/1e6,z:words[b+2]/1e6};return{bodyId:body.description.id,force_N:{x:impulse.x/elapsed,y:impulse.y/elapsed,z:impulse.z/elapsed},torque_N_m:{x:words[b+3]/1e6/elapsed,y:words[b+4]/1e6/elapsed,z:words[b+5]/1e6/elapsed},displacedVolume_m3:words[b+6]/65536*cellVolume};});readback.unmap();readback.destroy();this.onRigidLoads?.(loads);}).catch(()=>readback.destroy()).finally(()=>{this.rigidReadbackPending=false;});}
    if(!this.validationChecked){this.validationChecked=true;void this.device.popErrorScope().then(error=>{if(error)console.error(`GPU fluid validation: ${error.message}`);});}return true;
  }

  async readStats(){
    if((this.info.encodedSteps??0)===0)return this.info;
    if(this.readbackPending)return this.info;this.readbackPending=true;const querySegments=[...this.querySegments],queryBytes=this.queryResolve?this.queryCount*8:0,size=16+queryBytes,buffer=this.device.createBuffer({size:Math.max(16,size),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const encoder=this.device.createCommandEncoder();encoder.copyBufferToBuffer(this.reductionBuffer,0,buffer,0,16);if(this.queryResolve&&queryBytes>0)encoder.copyBufferToBuffer(this.queryResolve,0,buffer,16,queryBytes);this.device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);const words=new Uint32Array(buffer.getMappedRange(0,16)),initial=Math.max(1,this.info.initialVolumeCellSum??1);this.info.volumeCellSum=words[0]/2048;this.info.volumeDrift=(this.info.volumeCellSum-initial)/initial;this.info.rawVolumeDrift=this.info.volumeDrift;this.info.front_m=-this.scene.container.width_m/2+words[1]*this.scene.container.width_m/this.info.nx;this.info.maxSpeed_m_s=new Float32Array(new Uint32Array([words[2]]).buffer)[0];if(queryBytes>0){const times=new BigUint64Array(buffer.getMappedRange(16,queryBytes));const timings={advection_ms:0,pressure_ms:0,projection_ms:0,rigidCoupling_ms:0,diagnostics_ms:0,overhead_ms:0,total_ms:0};for(const segment of querySegments)timings[segment.name]+=Number(times[segment.end]-times[segment.start])/1e6;const categorized=timings.advection_ms+timings.pressure_ms+timings.projection_ms+timings.rigidCoupling_ms+timings.diagnostics_ms;timings.overhead_ms=Math.max(0,timings.total_ms-categorized);this.info.gpuTimings=timings;this.info.gpuStep_ms=timings.total_ms;}buffer.unmap();buffer.destroy();this.readbackPending=false;return this.info;
  }

  destroy(){for(const t of [this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB,this.heightA,this.heightB])t.destroy();this.params.destroy();this.reductionBuffer.destroy();this.rigidBuffer.destroy();this.rigidExchangeBuffer.destroy();this.querySet?.destroy();this.queryResolve?.destroy();}
}
