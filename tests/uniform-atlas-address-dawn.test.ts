import assert from "node:assert/strict";
import {test} from "node:test";
import {resolve} from "node:path";
import {pathToFileURL} from "node:url";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";
import {UniformAtlasAddressExperiment} from "../tools/uniform-atlas-address-experiment";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("atlas fixture preserves cross-page stencils, vertex planes and storage loads",{timeout:120_000},async()=>{
 const dawn=await import(pathToFileURL(resolve(modulePath!)).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);
 for(const mode of ["dense","affine","table"] as const)for(const edge of [16,32] as const){
  const adapter=await gpu.requestAdapter();assert.ok(adapter);
  const device:GPUDevice=await adapter.requestDevice(),errors:string[]=[];
  device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const atlas=new UniformAtlasAddressExperiment([64,32,32],edge,mode),resources=atlas.install(device);
  const d=[65,33,33] as const,n=d[0]*d[1]*d[2];
  const input=Float32Array.from({length:n},(_,i)=>((i*13)%157)-78);
  const textures=[0,1].map(()=>device.createTexture({size:d,dimension:"3d",format:"r32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST}));
  try{
   device.queue.writeTexture({texture:textures[0]!},input,{bytesPerRow:d[0]*4,rowsPerImage:d[1]},d);
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var source:texture_storage_3d<r32float,read_write>;
@group(0) @binding(1) var output:texture_storage_3d<r32float,write>;
fn sourceLoad(p:vec3i)->vec4f{return textureLoad(source,p);}
@compute @workgroup_size(4,4,4) fn run(@builtin(global_invocation_id)g:vec3u){
 let d=textureDimensions(output);if(any(g>=d)){return;}
 let p=vec3i(g);let hi=vec3i(d)-vec3i(1);
 let v=sourceLoad(p)+textureLoad(source,clamp(p+vec3i(1,0,0),vec3i(0),hi))+textureLoad(source,clamp(p+vec3i(0,-1,0),vec3i(0),hi))+textureLoad(source,clamp(p+vec3i(0,0,1),vec3i(0),hi));
 textureStore(output,p,v);
}`});
   const info=await module.getCompilationInfo();assert.ok(!info.messages.some(m=>m.type==="error"),JSON.stringify(info.messages));
   device.pushErrorScope("validation");
   const pipeline=await device.createComputePipelineAsync({label:`atlas ${mode} ${edge}`,layout:"auto",compute:{module,entryPoint:"run"}}).catch(async error=>{throw new Error(JSON.stringify({mode,edge,reason:error.reason,message:error.message,scope:(await device.popErrorScope())?.message,errors}));});
   const validation=await device.popErrorScope();assert.equal(validation,null,validation?.message);
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:textures.map((t,binding)=>({binding,resource:t.createView()}))});
   const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(...d.map(n=>Math.ceil(n/4)) as [number,number,number]);pass.end();device.queue.submit([e.finish()]);
   const actual:Float32Array=atlas.reorder(await readMixedTexture(device,textures[1]!),d,1),expected=new Float32Array(input.length);
   for(let z=0;z<d[2];z++)for(let y=0;y<d[1];y++)for(let x=0;x<d[0];x++)for(let c=0;c<1;c++){
    const at=(a:number,b:number,f:number)=>(a+d[0]*(b+d[1]*f))+c;
    expected[at(x,y,z)]=input[at(x,y,z)]!+input[at(Math.min(x+1,d[0]-1),y,z)]!+input[at(x,Math.max(0,y-1),z)]!+input[at(x,y,Math.min(z+1,d[2]-1))]!;
   }
   assert.deepEqual(actual,expected,`${mode} B${edge}`);assert.deepEqual(errors,[]);
  }finally{textures.forEach(t=>t.destroy());resources.destroy();device.destroy();}
 }
});

(modulePath?test:test.skip)("frozen frame restores textures and mutable buffers and replays typed queue writes",{timeout:120_000},async()=>{
 let device:GPUDevice|undefined;
 try{
  const {UniformFrozenGPUFrame}=await import("../tools/uniform-frozen-gpu-frame");
  const dawn=await import(pathToFileURL(resolve(modulePath!)).href);Object.assign(globalThis,dawn.globals);
  const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal","disable-dawn-features=timestamp_quantization"]).requestAdapter();assert.ok(adapter);
  device=await adapter.requestDevice({requiredFeatures:["timestamp-query"]});const d=device!;
  const errors:string[]=[];d.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const recorder=new UniformFrozenGPUFrame(d),textures=[0,1].map(()=>d.createTexture({size:[4,4,4],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC}));
  const state=d.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),params=d.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  try{
   const source=Float32Array.from({length:64},(_,i)=>i);
   d.queue.writeTexture({texture:textures[0]!},source,{bytesPerRow:16,rowsPerImage:4},[4,4,4]);d.queue.writeBuffer(state,0,new Uint32Array([7]));d.queue.writeBuffer(params,0,new Uint32Array(4));
   const module=d.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> state:array<u32>;
@group(0) @binding(1) var<uniform> config:vec4u;
@group(0) @binding(2) var source:texture_3d<f32>;
@group(0) @binding(3) var output:texture_storage_3d<r32float,write>;
@compute @workgroup_size(1) fn advance(){state[0]+=config.x;}
@compute @workgroup_size(4,4,4) fn field(@builtin(global_invocation_id)p:vec3u){textureStore(output,vec3i(p),vec4f(textureLoad(source,vec3i(p),0).x+f32(state[0])));}
`});
   const layout=d.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:3,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}}]});
   const pl=d.createPipelineLayout({bindGroupLayouts:[layout]});
   const pipes=await Promise.all(["advance","field"].map(entryPoint=>d.createComputePipelineAsync({layout:pl,compute:{module,entryPoint}})));
   const group=d.createBindGroup({layout,entries:[{binding:0,resource:{buffer:state}},{binding:1,resource:{buffer:params}},{binding:2,resource:textures[0]!.createView()},{binding:3,resource:textures[1]!.createView()}]});
   recorder.checkpoint();recorder.begin();
   d.queue.writeBuffer(params,0,new Uint32Array([99,3,88]),1,1);
   const e=d.createCommandEncoder();for(const pipeline of pipes){const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();}d.queue.submit([e.finish()]);await d.queue.onSubmittedWorkDone();recorder.end();
   const expected=Float32Array.from(source,x=>x+10);assert.deepEqual(await readMixedTexture(d,textures[1]!),expected);
   d.queue.writeBuffer(state,0,new Uint32Array([999]));d.queue.writeTexture({texture:textures[0]!},new Float32Array(64).fill(-555),{bytesPerRow:16,rowsPerImage:4},[4,4,4]);
   assert.equal((await recorder.measure(4,1)).length,4);assert.deepEqual(await readMixedTexture(d,textures[1]!),expected);assert.deepEqual(errors,[]);
  }finally{state.destroy();params.destroy();textures.forEach(t=>t.destroy());recorder.destroy();}
 }finally{device?.destroy();}
});
