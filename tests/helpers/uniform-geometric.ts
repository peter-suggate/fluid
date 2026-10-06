import assert from "node:assert/strict";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../../lib/core/webgpu-device-limits";
import {resolveMethodValues, type MethodParamValues} from "../../lib/core/method-contract";
import type {SceneDescription} from "../../lib/core/model";
import {createProcessRetainedDawnGPU} from "../../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../../lib/harness/webgpu-smoke-isolation";
import {uniformVolumeMethod} from "../../lib/methods/uniform/uniform-volume-method";
import type {WebGPUUniformReferenceSolver} from "../../lib/methods/uniform/webgpu-uniform-reference";
import {uniformMixedSolidWGSL, type UniformMixedSolid} from "../../lib/methods/uniform/uniform-mixed-solid.wgsl";
import {readMixedTexture,readMixedTileWords} from "./uniform-mixed-native-fields";

/** The production method, including its defaults and mixed ownership. */
export async function createUniformSolver(device:GPUDevice,scene:SceneDescription,values:MethodParamValues={}){
 assert.equal(uniformVolumeMethod.id,"uniform-volume");
 return await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",
  resolveMethodValues(uniformVolumeMethod,"balanced",{timeStep:"scene",...values}),undefined,()=>{}) as WebGPUUniformReferenceSolver;
}

export async function withUniformDevice(label:string,run:(device:GPUDevice)=>Promise<void>){
 await acquireWebGPUExclusiveLock("dawn-test",label);let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
  const adapter=await createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]).requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  await run(device);await device.queue.onSubmittedWorkDone();assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
}

export async function advanceUniform(solver:WebGPUUniformReferenceSolver,time:number){
 assert.ok(solver.advanceTo(time,[]),`Uniform advance to ${time} must be admitted`);
 await solver.awaitFrameCompletion();
 assert.equal(solver.info.simulationPipelineError,undefined);
}

/** Expand owner averages only for diagnostics. Raw fine backing texels in a
 * coarse tile are not independent liquid cells and must not be summed. */
export async function readUniformFields(device:GPUDevice,solver:WebGPUUniformReferenceSolver){
 await solver.awaitFrameCompletion();
 const [volume,tiles,phi,pressure]=await Promise.all([readMixedTexture(device,solver.volumeTexture),readMixedTileWords(device,solver),
  readMixedTexture(device,solver.vertexPhiTexture!),readMixedTexture(device,solver.gridPressureTexture)]);
 const {nx,ny,nz}=solver.info,tx=nx/4,ty=ny/4;
 const widthAt=(x:number,y:number,z:number)=>(tiles[(x>>2)+tx*((y>>2)+ty*(z>>2))]!&0x80000000)?1:4;
 const density=new Float32Array(nx*ny*nz);
 for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
  const w=widthAt(x,y,z);density[x+nx*(y+ny*z)]=volume[x-x%w+nx*(y-y%w+ny*(z-z%w))]!;
 }
 const vertex=(x:number,y:number,z:number)=>{
  let best=Infinity,origin:[number,number,number]|undefined;
  for(let k=0;k<8;k++){
   const c=[x-(k&1),y-((k>>1)&1),z-((k>>2)&1)];
   if(c.some((v,a)=>v<0||v>=[nx,ny,nz][a]!))continue;
   const t=(c[0]!>>2)+tx*((c[1]!>>2)+ty*(c[2]!>>2)),word=tiles[t]!;
   if(!(word&0x80000000)&&(word&0x3fffffff)<best){best=word&0x3fffffff;origin=c.map(v=>v-v%4) as [number,number,number];}
  }
  const at=(a:number,b:number,c:number)=>phi[a+(nx+1)*(b+(ny+1)*c)]!;
  if(!origin)return at(x,y,z);
  const f=[(x-origin[0])/4,(y-origin[1])/4,(z-origin[2])/4];let value=0;
  for(let k=0;k<8;k++){const c=[k&1,(k>>1)&1,(k>>2)&1],w=c.reduce((s,v,a)=>s*(v?f[a]!:1-f[a]!),1);if(w)value+=w*at(origin[0]+4*c[0]!,origin[1]+4*c[1]!,origin[2]+4*c[2]!);}
  return value;
 };
 return {density,pressure,phi,tiles,vertex,widthAt};
}

/** Read the actual GPU solid mask through Uniform's solid sampler. Disable its
 * optional tile cache here so paused edits can be checked before the next step.
 * open: the open fraction itself (umCellOpen), not its complement. */
export async function readUniformSolidFractions(device:GPUDevice,solver:WebGPUUniformReferenceSolver,open=false){
 const {nx,ny,nz}=solver.info;
 const solid=(solver as unknown as {mixedFrame:{solid:UniformMixedSolid}}).mixedFrame.solid;
 const module=device.createShaderModule({code:`const UM_D=vec3u(${nx}u,${ny}u,${nz}u);
${uniformMixedSolidWGSL(0,undefined,false)}
@group(1) @binding(0) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i>=arrayLength(&result)){return;}let p=vec3i(vec3u(i%UM_D.x,(i/UM_D.x)%UM_D.y,i/(UM_D.x*UM_D.y)));result[i]=${open?"umCellOpen(p)":"1.0-umCellOpen(p)"};}`});
 const resultLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
 const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[solid.bindLayout,resultLayout]}),compute:{module,entryPoint:"probe"}});
 const output=device.createBuffer({size:4*nx*ny*nz,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try{
  const group=device.createBindGroup({layout:resultLayout,entries:[{binding:0,resource:{buffer:output}}]});
  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,solid.bindGroup);pass.setBindGroup(1,group);pass.dispatchWorkgroups(Math.ceil(nx*ny*nz/64));pass.end();
  encoder.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);return new Float32Array(read.getMappedRange()).slice();
 }finally{if(read.mapState==="mapped")read.unmap();read.destroy();output.destroy();}
}
