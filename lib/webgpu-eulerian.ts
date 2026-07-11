import type { SceneDescription } from "./model";

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
}

const targetCells: Record<GPUQuality, number> = { balanced: 110_000, high: 500_000, ultra: 1_200_000 };

const computeShader = /* wgsl */ `
struct Params {
  dimsDt: vec4f,
  cellGravity: vec4f,
  container: vec4f,
  physical: vec4f,
}
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var pressureIn: texture_3d<f32>;
@group(0) @binding(3) var pressureOut: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var volumeIn: texture_3d<f32>;
@group(0) @binding(5) var volumeOut: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var<uniform> params: Params;

fn dims() -> vec3i { return vec3i(textureDimensions(volumeIn)); }
fn valid(p: vec3i) -> bool { let d=dims(); return all(p >= vec3i(0)) && all(p < d); }
fn clampCell(p: vec3i) -> vec3i { return clamp(p, vec3i(0), dims()-vec3i(1)); }
fn volume(p: vec3i) -> f32 { if (!valid(p)) { return 0.0; } return textureLoad(volumeIn,p,0).x; }
fn velocity(p: vec3i) -> vec3f { return textureLoad(velocityIn,clampCell(p),0).xyz; }
fn faceVelocity(p:vec3i)->vec3f{if(!valid(p)){return vec3f(0.0);}return textureLoad(velocityIn,p,0).xyz;}
fn pressure(p: vec3i, center: f32) -> f32 { if (!valid(p)) { return center; } if (volume(p)<0.01) { return 0.0; } return textureLoad(pressureIn,p,0).x; }
fn sampleVolume(p:vec3f)->f32{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(dims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);let c000=volume(b);let c100=volume(b+vec3i(1,0,0));let c010=volume(b+vec3i(0,1,0));let c110=volume(b+vec3i(1,1,0));let c001=volume(b+vec3i(0,0,1));let c101=volume(b+vec3i(1,0,1));let c011=volume(b+vec3i(0,1,1));let c111=volume(b+vec3i(1,1,1));return mix(mix(mix(c000,c100,f.x),mix(c010,c110,f.x),f.y),mix(mix(c001,c101,f.x),mix(c011,c111,f.x),f.y),f.z);
}
fn sampleVelocity(p:vec3f)->vec3f{
  let q=clamp(p-vec3f(0.5),vec3f(0.0),vec3f(dims()-vec3i(1)));let b=vec3i(floor(q));let f=fract(q);let c000=velocity(b);let c100=velocity(b+vec3i(1,0,0));let c010=velocity(b+vec3i(0,1,0));let c110=velocity(b+vec3i(1,1,0));let c001=velocity(b+vec3i(0,0,1));let c101=velocity(b+vec3i(1,0,1));let c011=velocity(b+vec3i(0,1,1));let c111=velocity(b+vec3i(1,1,1));return mix(mix(mix(c000,c100,f.x),mix(c010,c110,f.x),f.y),mix(mix(c001,c101,f.x),mix(c011,c111,f.x),f.y),f.z);
}
fn transportVelocity(id:vec3i)->vec3f{
  var v=velocity(id);if(volume(id)>=0.01){return v;}var sum=vec3f(0.0);var weight=0.0;
  let px=volume(id+vec3i(1,0,0));let nx=volume(id-vec3i(1,0,0));let py=volume(id+vec3i(0,1,0));let ny=volume(id-vec3i(0,1,0));let pz=volume(id+vec3i(0,0,1));let nz=volume(id-vec3i(0,0,1));
  sum+=velocity(id+vec3i(1,0,0))*px+velocity(id-vec3i(1,0,0))*nx+velocity(id+vec3i(0,1,0))*py+velocity(id-vec3i(0,1,0))*ny+velocity(id+vec3i(0,0,1))*pz+velocity(id-vec3i(0,0,1))*nz;weight=px+nx+py+ny+pz+nz;if(weight>0.001){v=sum/weight;}return v;
}
fn columnHeight(x:i32,z:i32)->f32{
  let d=dims();if(x<0||x>=d.x||z<0||z>=d.z){return 0.0;}var total=0.0;for(var y:i32=0;y<d.y;y+=1){total+=volume(vec3i(x,y,z))*params.cellGravity.y;}return total;
}
fn upwind(face:f32,negative:f32,positive:f32)->f32{return face*select(positive,negative,face>=0.0);}
fn advectedVolume(id:vec3i,dt:f32)->f32{
  let h=params.cellGravity.xyz;let centre=volume(id);
  let fxp=upwind(faceVelocity(id).x,centre,volume(id+vec3i(1,0,0)));let fxm=upwind(faceVelocity(id-vec3i(1,0,0)).x,volume(id-vec3i(1,0,0)),centre);
  let fyp=upwind(faceVelocity(id).y,centre,volume(id+vec3i(0,1,0)));let fym=upwind(faceVelocity(id-vec3i(0,1,0)).y,volume(id-vec3i(0,1,0)),centre);
  let fzp=upwind(faceVelocity(id).z,centre,volume(id+vec3i(0,0,1)));let fzm=upwind(faceVelocity(id-vec3i(0,0,1)).z,volume(id-vec3i(0,0,1)),centre);
  return clamp(centre-dt*((fxp-fxm)/h.x+(fyp-fym)/h.y+(fzp-fzm)/h.z),0.0,1.0);
}

@compute @workgroup_size(4,4,4)
fn advect(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  let dt=params.dimsDt.w; let h=params.cellGravity.xyz;
  let oldV=transportVelocity(id); let back=vec3f(id)+vec3f(0.5)-oldV*dt/h;
  var v=sampleVelocity(back); let phi=volume(id); if(phi>0.01&&length(v)<0.0001){v=oldV;}
  if (phi>0.01) { v.y += params.cellGravity.w*dt; } else if(length(oldV)<0.0001){ v=vec3f(0.0); }
  let d=dims();
  if (id.x==0 || id.x==d.x-1) { v.x=0.0; }
  if (id.y==0 || id.y==d.y-1) { v.y=0.0; }
  if (id.z==0 || id.z==d.z-1) { v.z=0.0; }
  textureStore(velocityOut,id,vec4f(v,0.0));
  textureStore(volumeOut,id,vec4f(advectedVolume(id,dt),0.0,0.0,0.0));
  textureStore(pressureOut,id,vec4f(0.0));
}

fn divergenceAt(id: vec3i) -> f32 {
  let h=params.cellGravity.xyz;
  return (faceVelocity(id).x-faceVelocity(id-vec3i(1,0,0)).x)/h.x
       + (faceVelocity(id).y-faceVelocity(id-vec3i(0,1,0)).y)/h.y
       + (faceVelocity(id).z-faceVelocity(id-vec3i(0,0,1)).z)/h.z;
}

@compute @workgroup_size(4,4,4)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  if (volume(id)<0.01) { textureStore(pressureOut,id,vec4f(0.0)); return; }
  let old=textureLoad(pressureIn,id,0).x; let h=min(params.cellGravity.x,min(params.cellGravity.y,params.cellGravity.z));
  let sum=pressure(id-vec3i(1,0,0),old)+pressure(id+vec3i(1,0,0),old)+pressure(id-vec3i(0,1,0),old)+pressure(id+vec3i(0,1,0),old)+pressure(id-vec3i(0,0,1),old)+pressure(id+vec3i(0,0,1),old);
  let rhs=params.physical.x*divergenceAt(id)/params.dimsDt.w;
  let next=(sum-rhs*h*h)/6.0;
  textureStore(pressureOut,id,vec4f(mix(old,next,0.8),0.0,0.0,0.0));
}

@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  let id=vec3i(gid); if (!valid(id)) { return; }
  if (volume(id)<0.01) { textureStore(velocityOut,id,vec4f(0.0)); textureStore(volumeOut,id,vec4f(0.0)); return; }
  let p=textureLoad(pressureIn,id,0).x; let h=params.cellGravity.xyz; let dt=params.dimsDt.w/params.physical.x;
  var v=velocity(id)-dt*vec3f(pressure(id+vec3i(1,0,0),p)-p,pressure(id+vec3i(0,1,0),p)-p,pressure(id+vec3i(0,0,1),p)-p)/h;
  let g=abs(params.cellGravity.w);let dhdx=(columnHeight(id.x+1,id.z)-columnHeight(id.x-1,id.z))/(2.0*h.x);let dhdz=(columnHeight(id.x,id.z+1)-columnHeight(id.x,id.z-1))/(2.0*h.z);v.x-=g*dhdx*params.dimsDt.w;v.z-=g*dhdz*params.dimsDt.w;
  let d=dims(); if (id.x==0||id.x==d.x-1){v.x=0.0;} if(id.y==0||id.y==d.y-1){v.y=0.0;} if(id.z==0||id.z==d.z-1){v.z=0.0;}
  textureStore(velocityOut,id,vec4f(v,0.0)); textureStore(volumeOut,id,vec4f(textureLoad(volumeIn,id,0).x));
}
`;

export class WebGPUEulerianSolver {
  readonly info: GPUEulerianInfo;
  private velocityA: GPUTexture; private velocityB: GPUTexture;
  private pressureA: GPUTexture; private pressureB: GPUTexture;
  private volumeA: GPUTexture; private volumeB: GPUTexture;
  private params: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;
  private advectPipeline: GPUComputePipeline; private jacobiPipeline: GPUComputePipeline; private projectPipeline: GPUComputePipeline;
  private lastTime = 0;
  private readbackPending = false;
  private validationChecked = false;

  constructor(private device: GPUDevice, readonly scene: SceneDescription, quality: GPUQuality) {
    const c=scene.container, target=targetCells[quality], h=Math.cbrt(c.width_m*c.height_m*c.depth_m/target);
    const nx=Math.max(8,Math.round(c.width_m/h)), ny=Math.max(8,Math.round(c.height_m/h)), nz=Math.max(8,Math.round(c.depth_m/h));
    const usage=GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST;
    const texture=(format: GPUTextureFormat)=>device.createTexture({size:[nx,ny,nz],dimension:"3d",format,usage});
    this.velocityA=texture("rgba32float"); this.velocityB=texture("rgba32float"); this.pressureA=texture("r32float"); this.pressureB=texture("r32float"); this.volumeA=texture("r32float"); this.volumeB=texture("r32float");
    this.params=device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
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
    ]});
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[this.bindGroupLayout]});
    this.advectPipeline=device.createComputePipeline({layout:pipelineLayout,compute:{module:shaderModule,entryPoint:"advect"}});
    this.jacobiPipeline=device.createComputePipeline({layout:pipelineLayout,compute:{module:shaderModule,entryPoint:"jacobi"}});
    this.projectPipeline=device.createComputePipeline({layout:pipelineLayout,compute:{module:shaderModule,entryPoint:"project"}});
    this.info={nx,ny,nz,cellCount:nx*ny*nz,cellSize_m:Math.max(c.width_m/nx,c.height_m/ny,c.depth_m/nz),pressureIterations:40,allocatedBytes:nx*ny*nz*(4*8+4*4),quality,encodedSteps:0};
    this.initializeVolume();
  }

  get volumeTexture(){return this.volumeA;}
  private initializeVolume(){
    const {nx,ny,nz}=this.info,c=this.scene.container,data=new Float32Array(nx*ny*nz),damWidth=.32,damHeight=Math.min(.92,c.fillFraction/damWidth);
    for(let k=0;k<nz;k++)for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){const fill=this.scene.fluid.initialCondition==="dam-break"?(i+.5)/nx<=damWidth&&(j+.5)/ny<=damHeight:(j+.5)/ny<=c.fillFraction;data[i+nx*(j+ny*k)]=fill?1:0;}
    const rowBytes=nx*4,padded=Math.ceil(rowBytes/256)*256,packed=new Uint8Array(padded*ny*nz),source=new Uint8Array(data.buffer);
    for(let k=0;k<nz;k++)for(let j=0;j<ny;j++)packed.set(source.subarray(rowBytes*(j+ny*k),rowBytes*(j+ny*k+1)),padded*(j+ny*k));
    this.device.queue.writeTexture({texture:this.volumeA},packed,{bytesPerRow:padded,rowsPerImage:ny},{width:nx,height:ny,depthOrArrayLayers:nz});
  }

  private group(_pipeline: GPUComputePipeline, velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture, pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture){return this.device.createBindGroup({layout:this.bindGroupLayout,entries:[{binding:0,resource:velocityIn.createView()},{binding:1,resource:velocityOut.createView()},{binding:2,resource:pressureIn.createView()},{binding:3,resource:pressureOut.createView()},{binding:4,resource:volumeIn.createView()},{binding:5,resource:volumeOut.createView()},{binding:6,resource:{buffer:this.params}}]});}
  private dispatch(pass: GPUComputePassEncoder,pipeline:GPUComputePipeline,group:GPUBindGroup){pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(this.info.nx/4),Math.ceil(this.info.ny/4),Math.ceil(this.info.nz/4));}

  advanceTo(time_s:number){
    if(time_s<this.lastTime)return false; const delta=time_s-this.lastTime;if(delta<1e-6)return true;const dt=Math.min(this.scene.numerics.maxDt_s,delta);this.lastTime=time_s;
    const c=this.scene.container;this.info.encodedSteps=(this.info.encodedSteps??0)+1;this.device.queue.writeBuffer(this.params,0,new Float32Array([this.info.nx,this.info.ny,this.info.nz,dt,c.width_m/this.info.nx,c.height_m/this.info.ny,c.depth_m/this.info.nz,this.scene.fluid.gravity_m_s2.y,c.width_m,c.height_m,c.depth_m,0,this.scene.fluid.density_kg_m3,this.scene.fluid.dynamicViscosity_Pa_s,0,0]));
    if(!this.validationChecked)this.device.pushErrorScope("validation");const encoder=this.device.createCommandEncoder({label:"GPU fluid step"});
    {const pass=encoder.beginComputePass();this.dispatch(pass,this.advectPipeline,this.group(this.advectPipeline,this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB));pass.end();}
    encoder.copyTextureToTexture({texture:this.velocityB},{texture:this.velocityA},[this.info.nx,this.info.ny,this.info.nz]);encoder.copyTextureToTexture({texture:this.volumeB},{texture:this.volumeA},[this.info.nx,this.info.ny,this.info.nz]);
    for(let iteration=0;iteration<this.info.pressureIterations;iteration+=1){const even=iteration%2===0;const pin=even?this.pressureA:this.pressureB,pout=even?this.pressureB:this.pressureA;const pass=encoder.beginComputePass();this.dispatch(pass,this.jacobiPipeline,this.group(this.jacobiPipeline,this.velocityA,this.velocityB,pin,pout,this.volumeA,this.volumeB));pass.end();}
    {const pass=encoder.beginComputePass();this.dispatch(pass,this.projectPipeline,this.group(this.projectPipeline,this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB));pass.end();}
    encoder.copyTextureToTexture({texture:this.velocityB},{texture:this.velocityA},[this.info.nx,this.info.ny,this.info.nz]);this.device.queue.submit([encoder.finish()]);if(!this.validationChecked){this.validationChecked=true;void this.device.popErrorScope().then(error=>{if(error)console.error(`GPU fluid validation: ${error.message}`);});}return true;
  }

  async readStats(){
    if(this.readbackPending)return this.info;this.readbackPending=true;
    const rowBytes=this.info.nx*4,padded=Math.ceil(rowBytes/256)*256,size=padded*this.info.ny*this.info.nz;
    const velocityRowBytes=this.info.nx*16,velocityPadded=Math.ceil(velocityRowBytes/256)*256,velocitySize=velocityPadded*this.info.ny*this.info.nz;
    const buffer=this.device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),velocityBuffer=this.device.createBuffer({size:velocitySize,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder=this.device.createCommandEncoder();encoder.copyTextureToBuffer({texture:this.volumeA},{buffer,bytesPerRow:padded,rowsPerImage:this.info.ny},[this.info.nx,this.info.ny,this.info.nz]);encoder.copyTextureToBuffer({texture:this.velocityB},{buffer:velocityBuffer,bytesPerRow:velocityPadded,rowsPerImage:this.info.ny},[this.info.nx,this.info.ny,this.info.nz]);this.device.queue.submit([encoder.finish()]);
    await Promise.all([buffer.mapAsync(GPUMapMode.READ),velocityBuffer.mapAsync(GPUMapMode.READ)]);
    const view=new DataView(buffer.getMappedRange()),velocityView=new DataView(velocityBuffer.getMappedRange());let sum=0,front=-1,maxSpeed=0;
    for(let k=0;k<this.info.nz;k++)for(let j=0;j<this.info.ny;j++)for(let i=0;i<this.info.nx;i++){const value=view.getFloat32(padded*(j+this.info.ny*k)+4*i,true);sum+=value;if(value>.08)front=Math.max(front,i);const q=velocityPadded*(j+this.info.ny*k)+16*i;maxSpeed=Math.max(maxSpeed,Math.hypot(velocityView.getFloat32(q,true),velocityView.getFloat32(q+4,true),velocityView.getFloat32(q+8,true)));}
    buffer.unmap();velocityBuffer.unmap();buffer.destroy();velocityBuffer.destroy();this.info.volumeCellSum=sum;this.info.front_m=-this.scene.container.width_m/2+(front+1)*this.scene.container.width_m/this.info.nx;this.info.maxSpeed_m_s=maxSpeed;this.readbackPending=false;return this.info;
  }

  destroy(){for(const t of [this.velocityA,this.velocityB,this.pressureA,this.pressureB,this.volumeA,this.volumeB])t.destroy();this.params.destroy();}
}
