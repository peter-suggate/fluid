import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
import { createUniformMixedLayoutFromWidths } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedMomentumCache, UniformMixedHangingTaps } from "../lib/methods/uniform/uniform-mixed-momentum-cache";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL } from "../lib/methods/uniform/uniform-mixed-face-dispatch.wgsl";
import { uniformMixedVelocitySamplingSource } from "../lib/methods/uniform/uniform-mixed-velocity-sampling.wgsl";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("particle cached samplers preserve mixed velocity at seams and domain walls",{timeout:120000},async()=>{
 await withUniformDevice("Narrow-band cached sampling parity",async device=>{
  const lattice={dimensions:[16,16,16] as const,cellSize_m:[1,1,1] as const,origin_m:{x:0,y:0,z:0}};
  const layout=createUniformMixedLayoutFromWidths(lattice,new Uint8Array(64).fill(4),[]);
  const ownership=new UniformMixedOwnership(device,layout);
  const texture=(size:number[])=>device.createTexture({size,dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST});
  const velocity=texture([16,16,16]),coarse=texture([6,6,6]);
  const negative=device.createBuffer({size:3*16*16*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:3*8192*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const cache=new UniformMixedMomentumCache(device,ownership),hanging=new UniformMixedHangingTaps(device,ownership);
  try{
   await cache.initialize();await hanging.initialize();
   const cacheGroup=cache.bind({extended:velocity,negative,coarseExtended:coarse});
   const hangingGroup=hanging.bind({extended:velocity,negative,coarse});
   const resources=device.createBindGroupLayout({entries:[
    ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
    {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
    {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   ]});
   const group=device.createBindGroup({layout:resources,entries:[
    ...[velocity,coarse,hanging.unitVelocity].map((t,binding)=>({binding,resource:t.createView()})),
    {binding:3,resource:{buffer:negative}},{binding:4,resource:{buffer:output}},
   ]});
   const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources,ownership.hangingLayout]});
   const pipelines=await Promise.all([0,1,2].map(async variant=>{
    const code=uniformMixedTopologyWGSL(layout,0)+`
@group(1) @binding(0) var velocity:texture_3d<f32>;
@group(1) @binding(1) var coarse:texture_3d<f32>;
@group(1) @binding(2) var unitVelocity:texture_3d<f32>;
@group(1) @binding(3) var<storage,read> negative:array<f32>;
@group(1) @binding(4) var<storage,read_write> result:array<vec4f>;
${uniformMixedFaceAddressWGSL}
fn umLoadMixedFace(p:vec3i,axis:u32)->f32{
 if(p[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(p,vec3i(0))),axis)];}
 return textureLoad(velocity,p,0)[axis];
}
fn umLoadCoarseFace(p:vec3i,axis:u32)->f32{return textureLoad(coarse,p+vec3i(1),0)[axis];}
${uniformMixedVelocitySamplingSource(false,variant>0,variant>0?"velocity":undefined,undefined,variant===2?"unitVelocity":undefined)}
@compute @workgroup_size(64) fn sample(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;let cell=vec3f(umCorner(i%4096u,16u));
 let offset=select(vec3f(0.01,0.375,0.625),vec3f(0.99,0.99,0.01),i>=4096u);
 result[${variant*8192}u+i]=vec4f(umSampleVelocity(cell+offset),0);
}`;
    const module=device.createShaderModule({code});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    return device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:"sample"}});
   }));
   // Nonconstant fields exercise coarse-face area restriction, fine seams,
   // and negative boundary planes. Refresh with a second field to detect
   // accidental dependence on previous cache contents.
   for(const pattern of [0,1,2,3,4])for(const revision of [0,1]){
    const widths=Uint8Array.from({length:64},(_,k)=>pattern===0?4:pattern===4?1:pattern===1?(k===21?1:4):pattern===2?(k%4<2?1:4):((k+(k>>2))%2?1:4));
    ownership.update(createUniformMixedLayoutFromWidths(lattice,widths,[]));
    const data=Float32Array.from({length:4096*4},(_,i)=>Math.sin(i*1.7+revision)*3+revision);
    device.queue.writeTexture({texture:velocity},data,{bytesPerRow:16*16,rowsPerImage:16},[16,16,16]);
    device.queue.writeBuffer(negative,0,Float32Array.from({length:768},(_,i)=>Math.cos(i*0.3+revision)));
    const e=device.createCommandEncoder();cache.encode(e,cacheGroup);hanging.encode(e,hangingGroup);
    for(const p of pipelines){const pass=e.beginComputePass();pass.setPipeline(p);pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,group);pass.setBindGroup(2,ownership.hangingGroup);pass.dispatchWorkgroups(128);pass.end();}
    device.queue.submit([e.finish()]);
    const values=await readMixedBuffer(device,output);
    for(const variant of [1,2]){
     let max=0;for(let i=0;i<8192*4;i++)max=Math.max(max,Math.abs(values[i]!-values[variant*8192*4+i]!));
     assert.ok(max<2e-6,`pattern ${pattern}, revision ${revision}, sampler ${variant}: ${max}`);
    }
   }
  }finally{hanging.destroy();ownership.destroy();velocity.destroy();coarse.destroy();negative.destroy();output.destroy();}
 });
});
