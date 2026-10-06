import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { createUniformMixedLayoutFromWidths } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedLayoutBuilder } from "../lib/methods/uniform/uniform-mixed-layout-builder";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedVelocitySamplingSource } from "../lib/methods/uniform/uniform-mixed-velocity-sampling.wgsl";
import { uniformCompiledExtensionNeighborWGSL } from "../lib/methods/uniform/uniform-compiled-extension.wgsl";
import { uniformExtensionNeighborReferenceWGSL } from "./helpers/uniform-extension-neighbor-reference.wgsl";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("compiled extension decisions and neighbors match across 256 adopted three-cubed layouts",async()=>{
 await withUniformDevice("Interior extension recipes",async device=>{
  const lattice={dimensions:[12,12,12] as const,cellSize_m:[1,1,1] as const,origin_m:{x:0,y:0,z:0}};
  const layout=createUniformMixedLayoutFromWidths(lattice,new Uint8Array(27).fill(4),[]);
  const ownership=new UniformMixedOwnership(device,layout);
  const band=device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const builder=new UniformMixedLayoutBuilder(device,{buffer:band,wordOffset:0},ownership);
  const topologyRead=device.createBuffer({size:256*27*20,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const counts=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   await builder.initialize();
   const expected:Uint32Array[]=[];
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
    var requestAnchor=face.anchor;requestAnchor[k/2u]+=select(-i32(face.width),i32(face.width),(k&1u)!=0u);
    let inside=all(point>=vec3f(0))&&all(point<=vec3f(UM_D));
    if(inside!=ueRequestInside(requestAnchor,axis)){atomicAdd(&counts[5],1u);}
    var rebuilt=vec3f(requestAnchor)+vec3f(0.5*f32(face.width));rebuilt[axis]=f32(requestAnchor[axis]+1);
    if(any(bitcast<vec3u>(point)!=bitcast<vec3u>(rebuilt))){atomicAdd(&counts[5],1u);}
    if(!inside){continue;}
    let expectedDirect=direct(point,axis,face.width);
    let localAnchor=face.anchor-vec3i(4u*umTileCoord(owner.tile));
    let compiledDirect=ueTopologyDirect(umTileStencil(owner.tile),localAnchor,axis,face.width,k);
    atomicAdd(&counts[6],1u);
    if(expectedDirect!=compiledDirect){atomicAdd(&counts[5],1u);}
    if(face.width==4u&&k==2u*axis+1u){atomicAdd(&counts[7],1u);}
    if(expectedDirect){
     let before=umNeighbor(point,center,axis,k/2u,face.width);let state=umSlotState(requestAnchor,axis,face.width);
     if(any(bitcast<vec3u>(vec3f(before.value,before.distance,before.spacing))!=bitcast<vec3u>(vec3f(state,f32(face.width)*h[k/2u])))){atomicAdd(&counts[0],1u);}
     continue;
    }
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
    const cpu=createUniformMixedLayoutFromWidths(lattice,widths,[]);
    expected.push(new Uint32Array([...cpu.tiles,...cpu.fineTiles,...cpu.coarseTiles,...cpu.stencils,...cpu.blendMasks]));
    builder.setStatic(Uint8Array.from(widths,w=>Number(w===1)));
    const e=device.createCommandEncoder();builder.encodeClear(e);builder.encode(e);
    e.copyBufferToBuffer(builder.generation.topology,0,topologyRead,sample*27*20,27*20);
    ownership.adoptGpu(e,builder.generation);
    const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,group);pass.dispatchWorkgroups(27);pass.end();device.queue.submit([e.finish()]);
   }
   const e=device.createCommandEncoder();e.copyBufferToBuffer(counts,0,read,0,32);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
   const result=Array.from(new Uint32Array(read.getMappedRange()));read.unmap();
   assert.equal(result[0],0,"bit-exact neighbor value, distance and spacing");
   for(let i=1;i<=4;i++)assert.ok(result[i]!>0,`recipe category ${i} exercised`);
   assert.equal(result[5],0,"compiled domain, point and direct/search decisions match original geometry");
   assert.ok(result[6]!>0&&result[7]!>0,"in-domain and far-positive requests exercised");
   await topologyRead.mapAsync(GPUMapMode.READ);const topology=new Uint32Array(topologyRead.getMappedRange()).slice();topologyRead.unmap();
   for(let sample=0;sample<256;sample++)assert.deepEqual(topology.slice(sample*135,(sample+1)*135),expected[sample],`three-cubed builder layout ${sample}`);
  }finally{topologyRead.destroy();read.destroy();counts.destroy();builder.destroy();band.destroy();ownership.destroy();}
 });
});
