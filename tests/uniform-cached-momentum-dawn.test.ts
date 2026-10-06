import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { createUniformMixedLayoutFromWidths } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedVelocitySamplingSource } from "../lib/methods/uniform/uniform-mixed-velocity-sampling.wgsl";
import { uniformCachedMomentumSamplingWGSL } from "../lib/methods/uniform/uniform-cached-momentum.wgsl";
import { uniformVelocityDepartureWGSL } from "../lib/methods/uniform/uniform-velocity-departure.wgsl";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("prepared momentum blends preserve general samples and local-width characteristics across 256 layouts", { timeout: 180000 }, async () => {
 await withUniformDevice("Prepared momentum sampling parity", async device => {
  const lattice = { dimensions: [8,8,8] as const, cellSize_m: [1,1,1] as const, origin_m: {x:0,y:0,z:0} };
  const layout = createUniformMixedLayoutFromWidths(lattice, new Uint8Array(8).fill(4), []);
  const ownership = new UniformMixedOwnership(device, layout);
  const field = device.createTexture({size:[8,8,8],dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING});
  const errors = device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read = device.createBuffer({size:32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
   const common = uniformMixedTopologyWGSL(layout,0) + /* wgsl */ `
fn hash(p:vec3i,axis:u32)->f32{let k=bitcast<u32>(p.x+13*p.y+137*p.z)+axis*971u;return f32((k*1664525u+1013904223u)&65535u)/8192.0-4.0;}
fn umLoadMixedFace(p:vec3i,axis:u32)->f32{return hash(p,axis);}
fn umLoadCoarseFace(p:vec3i,axis:u32)->f32{return hash(p+vec3i(37),axis);}
${uniformMixedVelocitySamplingSource(false,true)}
`;
   const prepareModule = device.createShaderModule({code:common+/* wgsl */ `
@group(1) @binding(0) var output:texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(64) fn prepare(@builtin(global_invocation_id) id:vec3u){
 let p=vec3i(umCorner(id.x,8u));
 textureStore(output,p,vec4f(umVelocityTap1(p,0u),umVelocityTap1(p,1u),umVelocityTap1(p,2u),0));
}`});
   const compareCommon=common.replace(uniformMixedVelocitySamplingSource(false,true),uniformMixedVelocitySamplingSource(false,true,undefined,undefined,"unitVelocity"));
   const compareModule = device.createShaderModule({code:compareCommon+/* wgsl */ `
@group(1) @binding(0) var unitVelocity:texture_3d<f32>;
@group(1) @binding(1) var<storage,read_write> counts:array<atomic<u32>>;
fn umClampMomentum(p:vec3f)->vec3f{return clamp(p,vec3f(0),vec3f(UM_D));}
${uniformCachedMomentumSamplingWGSL}
fn referenceDeparture(position:vec3f,dt:f32,h:vec3f)->vec3f{
${uniformVelocityDepartureWGSL("umSampleVelocity","umClampMomentum","f32(umOwnerAt(clamp(vec3i(floor(point)),vec3i(0),vec3i(UM_D)-vec3i(1))).width)")}
}
@compute @workgroup_size(64) fn compare(@builtin(global_invocation_id) id:vec3u){
 let p=vec3f(umCorner(id.x,8u))+vec3f(0.125,0.375,0.625);
 umUnitEscaped=false;let actual=umUnitSample(p);let expected=umSampleVelocity(p);
 if(!umUnitEscaped){
  atomicAdd(&counts[2],1u);
  // Separate interpolation kernels may contract f32 arithmetic differently;
  // this bounds roundoff at 1e-6 for source velocities in [-4,4].
  if(any(abs(actual-expected)>vec3f(1e-6))){atomicAdd(&counts[0],1u);}
  let weight=umVelocitySamplingWeights(p);if(weight>0.0&&weight<1.0){atomicAdd(&counts[4],1u);}
 }else{atomicAdd(&counts[5],1u);}
 let dt=select(0.125,4.0,id.y!=0u);let h=vec3f(0.5,1.0,2.0);
 umUnitEscaped=false;let departure=umUnitDeparture(p,dt,h);
 if(!umUnitEscaped){
  atomicAdd(&counts[3],1u);let expectedDeparture=referenceDeparture(p,dt,h);
  if(any(abs(departure-expectedDeparture)>vec3f(1e-5))){atomicAdd(&counts[1],1u);}
 }
}`});
   for(const module of [prepareModule,compareModule]) assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const preparation=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}}]});
   const comparison=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const prepare=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,preparation]}),compute:{module:prepareModule,entryPoint:"prepare"}});
   const compare=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,comparison]}),compute:{module:compareModule,entryPoint:"compare"}});
   const a=device.createBindGroup({layout:preparation,entries:[{binding:0,resource:field.createView()}]});
   const b=device.createBindGroup({layout:comparison,entries:[{binding:0,resource:field.createView()},{binding:1,resource:{buffer:errors}}]});
   for(let bits=0;bits<256;bits++){
    ownership.update(createUniformMixedLayoutFromWidths(lattice,Uint8Array.from({length:8},(_,k)=>bits&(1<<k)?1:4),[]));
    const e=device.createCommandEncoder();
    let pass=e.beginComputePass();pass.setPipeline(prepare);pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,a);pass.dispatchWorkgroups(8);pass.end();
    pass=e.beginComputePass();pass.setPipeline(compare);pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,b);pass.dispatchWorkgroups(8,2);pass.end();device.queue.submit([e.finish()]);
   }
   const e=device.createCommandEncoder();e.copyBufferToBuffer(errors,0,read,0,32);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
   const counts=Array.from(new Uint32Array(read.getMappedRange()));read.unmap();
   assert.deepEqual(counts.slice(0,2),[0,0],"sample and characteristic parity");
   assert.ok(counts[2]!>100000&&counts[3]!>50000,"substantial sample and characteristic coverage");
   assert.ok(counts[4]!>10000&&counts[5]!>10000,"exercise blends and negative-plane fallbacks");
  } finally { read.destroy();errors.destroy();field.destroy();ownership.destroy(); }
 });
});
