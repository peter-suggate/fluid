import assert from "node:assert/strict";
import test from "node:test";
import { UniformComputeBlit, uniformComputePassEpoch } from "../lib/methods/uniform/uniform-compute-blit";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { profileRegionTierCommands } from "../tools/region-tier-command-profile";
import { GPUStageTimestampRecorder } from "../lib/core/performance-trace";

test("shared frame schedule preserves dependencies, binding epochs and native fences", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
 await withUniformDevice("frame schedule",async device=>{
  const profile=profileRegionTierCommands(device),blit=new UniformComputeBlit(device);
  const buffers=[0,1].map(()=>device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}));
  const read=device.createBuffer({size:32,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  const code="@group(0) @binding(0) var<storage,read_write> words:array<u32>; @compute @workgroup_size(4) fn main(@builtin(local_invocation_index) i:u32){words[i]=2u*words[i]+1u;}";
  const module=device.createShaderModule({code});
  const empty=device.createBindGroupLayout({entries:[]});
  const upper=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[empty,layout]}),compute:{module:device.createShaderModule({code:code.replace("@group(0)","@group(1)")}),entryPoint:"main"}});
  const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"main"}});
  const groups=buffers.map(buffer=>device.createBindGroup({layout,entries:[{binding:0,resource:{buffer}}]}));
  try{
   await blit.initialize();
   for(const shared of [false,true]){
    blit.setSharedSchedule(shared);profile.begin();
    const encoder=blit.adopt(device.createCommandEncoder());
    encoder.clearBuffer(buffers[0]!);
    let previous:GPUComputePassEncoder|undefined,epoch=0;
    const stage=(arm:number)=>{
     const pass=encoder.beginComputePass({label:"test stage"});
     if(shared&&previous===pass)assert.ok(uniformComputePassEpoch(pass)>epoch);
     previous=pass;epoch=uniformComputePassEpoch(pass);
     pass.setPipeline(arm===0?upper:pipeline);pass.setBindGroup(arm===0?1:0,groups[arm]!);pass.dispatchWorkgroups(1);pass.end();
    };
    stage(0);encoder.copyBufferToBuffer(buffers[0]!,0,buffers[1]!,0,16);stage(1);
    // A native readback must end the shared pass; later stages must reopen it.
    encoder.copyBufferToBuffer(buffers[1]!,0,read,0,16);stage(1);
    encoder.copyBufferToBuffer(buffers[1]!,0,read,16,16);
    device.queue.submit([encoder.finish()]);
    const rows=Object.values(profile.end()!);
    assert.equal(rows.reduce((n,r)=>n+r.passes,0),shared?2:5);
    assert.equal(rows.reduce((n,r)=>n+r.dispatches,0),5);
    assert.equal(rows.reduce((n,r)=>n+r.indirectDispatches,0),0);
    await read.mapAsync(GPUMapMode.READ);
    assert.deepEqual([...new Uint32Array(read.getMappedRange())],[3,3,3,3,7,7,7,7]);read.unmap();
   }
   if(device.features.has("timestamp-query")){
    const queries=device.createQuerySet({type:"timestamp",count:2});
    const resolved=device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE});
    try{
     blit.setSharedSchedule(true);profile.begin();const encoder=blit.adopt(device.createCommandEncoder());
     for(let stage=0;stage<3;stage++){
      const pass=encoder.beginComputePass(stage===1?{timestampWrites:{querySet:queries,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});
      pass.setPipeline(pipeline);pass.setBindGroup(0,groups[0]!);pass.dispatchWorkgroups(1);pass.end();
     }
     encoder.resolveQuerySet(queries,0,2,resolved,0);device.queue.submit([encoder.finish()]);
     assert.equal(Object.values(profile.end()!).reduce((n,r)=>n+r.passes,0),3,"timestamped stage fences both sides");
     await device.queue.onSubmittedWorkDone();
    }finally{queries.destroy();resolved.destroy();}
   }
  }finally{profile.destroy();blit.destroy();for(const b of buffers)b.destroy();read.destroy();}
 },["timestamp-query"]);
});

test("stage timing observes shared physical passes and fences phase boundaries", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
 await withUniformDevice("shared stage timing",async device=>{
  const blit=new UniformComputeBlit(device);
  const data=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const read=device.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const module=device.createShaderModule({code:"@group(0) @binding(0) var<storage,read_write> a:array<u32>; @compute @workgroup_size(4) fn main(@builtin(local_invocation_index) i:u32){a[i]=2u*a[i]+1u;}"});
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:data}}]});
  try{
   await blit.initialize();await GPUStageTimestampRecorder.prepare(device);blit.setSharedSchedule(true);
   for(const timed of [true,false,true]){
    let passes=0;
    const recorder=timed?new GPUStageTimestampRecorder(device,1,"physics","test"):undefined;recorder?.begin();
    const encoder=blit.adopt(device.createCommandEncoder(),recorder?raw=>{
     const measured=recorder.instrument(raw);
     return new Proxy(measured,{get(target,key){
      if(key==="beginComputePass")return (d?:GPUComputePassDescriptor)=>{passes++;return target.beginComputePass(d);};
      const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
     }});
    }:undefined);
    encoder.clearBuffer(data);
    for(let phase=0;phase<2;phase++){
     for(let stage=0;stage<3;stage++){
      const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
     }
     if(recorder){blit.finishStage(encoder);recorder.completePhase(encoder,{id:"other",label:`phase ${phase}`});}
    }
    assert.equal(passes,timed?2:0,"six logical stages share two timed physical passes");
    recorder?.anchorFinalBoundary(data);recorder?.resolve(encoder);
    encoder.copyBufferToBuffer(data,0,read,0,16);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    assert.deepEqual([...new Uint32Array(read.getMappedRange())],[63,63,63,63]);read.unmap();
    if(recorder){const trace=await recorder.read(true);assert.ok(trace,"physical timestamp chain must remain valid");assert.ok(trace.phases.every(p=>p.label==="phase 0"||p.label==="phase 1"),"zero-length phases may disappear under timestamp quantization");}
   }
  }finally{blit.destroy();data.destroy();read.destroy();}
 },["timestamp-query"]);
});
