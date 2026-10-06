import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { createUniformMixedLayoutFromWidths } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedVelocitySamplingSource } from "../lib/methods/uniform/uniform-mixed-velocity-sampling.wgsl";
import { uniformCompiledExtensionNeighborWGSL } from "../lib/methods/uniform/uniform-compiled-extension.wgsl";
import { uniformExtensionNeighborReferenceWGSL } from "./helpers/uniform-extension-neighbor-reference.wgsl";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("compiled extension covers all six-side recipes around an interior coarse tile",async()=>{
 await withUniformDevice("Interior extension recipes",async device=>{
  const lattice={dimensions:[12,12,12] as const,cellSize_m:[1,1,1] as const,origin_m:{x:0,y:0,z:0}};
  const layout=createUniformMixedLayoutFromWidths(lattice,new Uint8Array(27).fill(4),[]);
  const ownership=new UniformMixedOwnership(device,layout);
  const counts=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const module=device.createShaderModule({code:uniformMixedTopologyWGSL(layout,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> counts:array<atomic<u32>>;
fn umLoadMixedFace(p:vec3i,axis:u32)->f32{return 0.0;}
fn umLoadCoarseFace(p:vec3i,axis:u32)->f32{return 0.0;}
${uniformMixedVelocitySamplingSource(false,true)}
const UM_INF=1e20;const h=vec4f(0.5,1,2,0);
fn umSlotState(anchor:vec3i,axis:u32,width:u32)->vec2f{
 let k=bitcast<u32>(anchor.x+17*anchor.y+101*anchor.z)+axis*733u+width*937u;
 return vec2f(f32((k*1664525u+1013904223u)&65535u)/8192.0-4.0,select(f32(k%5u)*0.5,UM_INF,k%7u==0u));
}
${uniformExtensionNeighborReferenceWGSL}
${uniformCompiledExtensionNeighborWGSL}
fn direct(point:vec3f,component:u32,width:u32)->bool{
 var below=vec3i(floor(point));let plane=i32(round(point[component]));below[component]=plane-1;
 var above=below;above[component]=plane;
 var lo=0u;if(plane>0){lo=umTileWidth(umTileAt(vec3u(below)/4u));}
 var hi=0u;if(plane<i32(UM_D[component])){hi=umTileWidth(umTileAt(vec3u(above)/4u));}
 return select((lo==width&&(hi==0u||hi>=width))||(hi==width&&lo>width),hi==width,lo==0u);
}
@compute @workgroup_size(64) fn compare(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(gid);if(owner.width==0u){return;}
 for(var axis=0u;axis<3u;axis++){
  let first=umFaceFirst(owner,axis,1);
  for(var part=0u;part<first.count;part++){
   let face=umFacePatch(first,part);let center=umFaceCenter(face);
   for(var k=0u;k<6u;k++){
    var point=center;point[k/2u]+=select(-f32(face.width),f32(face.width),(k&1u)!=0u);
    if(any(point<vec3f(0))||any(point>vec3f(UM_D))||direct(point,axis,face.width)){continue;}
    let before=umNeighbor(point,center,axis,k/2u,face.width);let after=ueCompiledNeighbor(point,center,axis,k/2u,face.width);
    if(any(bitcast<vec3u>(vec3f(before.value,before.distance,before.spacing))!=bitcast<vec3u>(vec3f(after.value,after.distance,after.spacing)))){atomicAdd(&counts[0],1u);}
    let kind=select(select(2u,3u,(u32(round(point[axis]))&3u)!=0u),1u,face.width==4u);atomicAdd(&counts[kind],1u);
    if(kind==3u){
     let tile=umTileAt(min(vec3u(point),UM_D-vec3u(1))/4u);let sides=umFineFaceSides(tile);
     if(((sides>>(2u*axis))&3u)==3u){atomicAdd(&counts[4],1u);}
    }
   }
  }
 }
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const resources=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources]}),compute:{module,entryPoint:"compare"}});
   const group=device.createBindGroup({layout:resources,entries:[{binding:0,resource:{buffer:counts}}]});
   const neighbors=[14,12,16,10,22,4];let random=9127;
   for(let sample=0;sample<256;sample++){
    const widths=Uint8Array.from({length:27},()=>{random=(Math.imul(random,1664525)+1013904223)>>>0;return (random>>>24)&1?1:4;});
    widths[13]=4;neighbors.forEach((tile,side)=>{widths[tile]=sample&(1<<side)?1:4;});
    ownership.update(createUniformMixedLayoutFromWidths(lattice,widths,[]));
    const e=device.createCommandEncoder();const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,group);pass.dispatchWorkgroups(27);pass.end();device.queue.submit([e.finish()]);
   }
   const e=device.createCommandEncoder();e.copyBufferToBuffer(counts,0,read,0,32);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
   const result=Array.from(new Uint32Array(read.getMappedRange()));read.unmap();
   assert.equal(result[0],0,"bit-exact neighbor value, distance and spacing");
   for(let i=1;i<=4;i++)assert.ok(result[i]!>0,`recipe category ${i} exercised`);
  }finally{read.destroy();counts.destroy();ownership.destroy();}
 });
});
