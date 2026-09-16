/** Isolate the production collocated-to-face sampler at zero travel on a
 * divergence-free staggered Fourier vortex. No advection/projection/coarsening. */
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
const source=readFileSync(new URL("../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts",import.meta.url),"utf8");
const sampler=source.slice(source.indexOf("fn sampleFaceVelocitySupportAtSpans("),source.indexOf("fn traceFaceDeparture("));
assert.ok(sampler.startsWith("fn sampleFaceVelocitySupportAtSpans("));
await acquireWebGPUExclusiveLock("dawn-probe","production velocity roundtrip");
let device:GPUDevice|undefined;
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
 const module=device.createShaderModule({code:`
struct Params{dimensions:vec4u,config:vec4f}
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
@group(0) @binding(1) var<uniform> p:Params;
struct FaceVelocitySupport{velocity:vec3f,spans:vec3f,owner:bool,extended:bool,liquid:bool}
fn original(point:vec3f)->vec3f{
 let k=6.28318530718/p.config.x;
 return vec3f(sin(k*point.x)*cos(k*point.y),-cos(k*point.x)*sin(k*point.y),0.0);
}
fn faceVelocitySupportAt(q:vec3i)->FaceVelocitySupport{
 let h=p.config.y;let centre=(floor(vec3f(q)/h)+vec3f(0.5))*h;
 // The production cell velocity is the mean of opposing face values on a
 // uniform open grid. Supply those exact averages to the production sampler.
 let x=vec3f(h*0.5,0.0,0.0);let y=vec3f(0.0,h*0.5,0.0);
 let value=vec3f(0.5*(original(centre-x).x+original(centre+x).x),
   0.5*(original(centre-y).y+original(centre+y).y),0.0);
 return FaceVelocitySupport(value,vec3f(h),true,true,true);
}
${sampler}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x>=1024u){return;}let h=p.config.y;
 let base=h*vec3f(f32(16u+gid.x%32u),f32(16u+gid.x/32u),16.0);
 let x=base+vec3f(0.0,0.5*h,0.5*h);let y=base+vec3f(0.5*h,0.0,0.5*h);
 let before=vec2f(original(x).x,original(y).y);
 let after=vec2f(sampleFaceVelocitySupportAtSpans(x,vec3f(h)).x,
   sampleFaceVelocitySupportAtSpans(y,vec3f(h)).y);
 result[gid.x]=vec4f(dot(before,before),dot(after,after),dot(before,after),0.0);
}`});
 const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==="error").map(m=>m.message),[]);
 const pipeline=device.createComputePipeline({layout:"auto",compute:{module,entryPoint:"main"}});
 const out=device.createBuffer({size:16384,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 const params=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 const read=device.createBuffer({size:16384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
 const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[out,params].map((buffer,binding)=>({binding,resource:{buffer}}))});
 const results=[];
 for(const [wavelength,width] of [[4,1],[8,1],[16,1],[32,1],[16,2],[16,4]]){
  device.queue.writeBuffer(params,0,new Uint32Array([256,256,256,0]));device.queue.writeBuffer(params,16,new Float32Array([wavelength!,width!,0,0]));
  const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(16);pass.end();encoder.copyBufferToBuffer(out,0,read,0,16384);device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);const data=new Float32Array(read.getMappedRange());const sum=[0,0,0];for(let i=0;i<1024;i++)for(let j=0;j<3;j++)sum[j]!+=data[4*i+j]!;
  const measuredEnergyRetention=sum[1]!/sum[0]!,expectedEnergyRetention=Math.cos(Math.PI*width!/wavelength!)**4;
  assert.ok(Math.abs(measuredEnergyRetention-expectedEnergyRetention)<1e-5);
  results.push({wavelengthFineCells:wavelength,cellWidthFineCells:width,measuredEnergyRetention,expectedEnergyRetention,amplitudeRetention:sum[2]!/sum[0]!});read.unmap();
 }
 for(const b of [out,params,read])b.destroy();
 mkdirSync("artifacts/level-set-volume",{recursive:true});writeFileSync("artifacts/level-set-volume/velocity-roundtrip-dissipation.json",JSON.stringify({scope:"production sampler; analytic divergence-free staggered source; zero travel",results},null,2)+"\n");console.log(JSON.stringify(results));
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
