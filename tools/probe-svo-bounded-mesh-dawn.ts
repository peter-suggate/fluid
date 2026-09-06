/** Bounded Dawn check of production mesh publication scheduling, without a large scene. */
import assert from "node:assert/strict";
import {createDawnRenderDevice} from "./svo-dry-frame-harness";
import {svoSurfaceMeshWGSL,SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME,SVO_SURFACE_MESH_STATE_BYTES} from "../lib/svo/svo-surface-mesh";
const leafCount = SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME * 2.5;
const {device,adapterInfo,validationErrors}=await createDawnRenderDevice();
const shader=svoSurfaceMeshWGSL(0,1);
const prepare=shader.slice(shader.indexOf('@compute @workgroup_size(1)\nfn surfaceMeshPrepare'),shader.indexOf('@compute @workgroup_size(64)\nfn surfaceMeshBuild'));
const publish=shader.slice(shader.indexOf('@compute @workgroup_size(1)\nfn surfaceMeshPublish'),shader.indexOf('// Cull exact cached quad bounds'));
const code=`
@group(0) @binding(0) var<storage,read_write> meshState:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> publication:array<u32>;
@group(0) @binding(2) var<storage,read_write> meshOutput:array<u32>;
struct Mapping {worldOrigin:vec3f,cellSize:vec3f,brickSize:u32}
struct Dry {materialPublication:vec4u,mapping:Mapping}
const dry=Dry(vec4u(0u,0u,0u,1u),Mapping(vec3f(0.0),vec3f(1.0),8u));
struct View {cameraPosition:vec4f} const uniforms=View(vec4f(0.0));
const REQUIRED_FIELDS=1u;
fn dryPublicationWord(i:u32)->u32{return publication[i];}
fn svoControlLoad(i:u32)->u32{return publication[4];}
struct Region {identity:u32} fn meshRegionAt(p:vec3f)->Region{return Region(0u);}
fn sceneIdentitySolid(i:u32)->bool{return false;}
${prepare}
// Only the bounded scheduler is under test: count one synthetic quad per leaf.
@compute @workgroup_size(64) fn build(@builtin(global_invocation_id) id:vec3u){
 let cursor=atomicLoad(&meshState[14]);let job=id.x+id.y*65535u*64u;
 let jobsPerBrick=6u*dry.mapping.brickSize;if(job%jobsPerBrick!=0u){return;}let index=cursor+job/jobsPerBrick;
 if(index<min(publication[4],cursor+${SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME}u)){let slot=atomicAdd(&meshState[4],1u);if(slot>=arrayLength(&meshOutput)){atomicOr(&meshState[5],1u);}else{meshOutput[slot]=index;}}
}
${publish}`;
const shaderModule=device.createShaderModule({code});
const layout=device.createBindGroupLayout({entries:[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding!==1?'storage' as const:'read-only-storage' as const}}))});
const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
const pipelines=await Promise.all(['surfaceMeshPrepare','build','surfaceMeshPublish'].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module:shaderModule,entryPoint}})));
const state=device.createBuffer({size:SVO_SURFACE_MESH_STATE_BYTES,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
const pub=device.createBuffer({size:20,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
const indirect=device.createBuffer({size:12,usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
const readback=device.createBuffer({size:64,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
let output=device.createBuffer({size:leafCount*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
const bind=()=>device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:state}},{binding:1,resource:{buffer:pub}},{binding:2,resource:{buffer:output}}]});
let group=bind();
device.queue.writeBuffer(pub,0,new Uint32Array([1,1,1,1,leafCount]));
async function frame(){
 const encoder=device.createCommandEncoder();
 for(let i=0;i<3;i++){
  if(i===1)encoder.copyBufferToBuffer(state,32,indirect,0,12);
  const pass=encoder.beginComputePass();pass.setPipeline(pipelines[i]);pass.setBindGroup(0,group);
  if(i===1)pass.dispatchWorkgroupsIndirect(indirect,0);else pass.dispatchWorkgroups(1);pass.end();
 }
 encoder.copyBufferToBuffer(state,0,readback,0,64);device.queue.submit([encoder.finish()]);
 await readback.mapAsync(GPUMapMode.READ);const words=new Uint32Array(readback.getMappedRange().slice(0));readback.unmap();return words;
}
const receipts=[];
for(let i=0;i<3;i++){const words=await frame();receipts.push({cursor:words[14],quads:words[4],ready:words[13]});assert.equal(words[4],Math.min(leafCount,(i+1)*SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME));assert.equal(words[13],i===2?1:0);}
assert.equal((await frame())[4],leafCount,'cached frames must not append');
device.queue.writeBuffer(pub,8,new Uint32Array([2]));assert.equal((await frame())[4],SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME,'new revision must restart');
device.queue.writeBuffer(state,20,new Uint32Array([1]));
device.queue.writeBuffer(pub,0,new Uint32Array([0]));assert.equal((await frame())[13],0);
device.queue.writeBuffer(pub,0,new Uint32Array([1]));assert.equal((await frame())[4],SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME,'invalidated partial build must restart');
// An overflowing batch must preserve the preceding complete prefix and pause.
output.destroy();output=device.createBuffer({size:SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});group=bind();
device.queue.writeBuffer(state,0,new Uint32Array(SVO_SURFACE_MESH_STATE_BYTES/4));
assert.equal((await frame())[14],SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME);
const paused=await frame();assert.equal(paused[14],SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME);assert.equal(paused[4],SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME);assert.equal(paused[5],1);
assert.equal((await frame())[14],paused[14],'capacity pause must not advance');
const oldOutput=output;output=device.createBuffer({size:leafCount*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
const copy=device.createCommandEncoder();copy.copyBufferToBuffer(oldOutput,0,output,0,oldOutput.size);device.queue.submit([copy.finish()]);group=bind();
assert.equal((await frame())[14],SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME*2);
const resumed=await frame();assert.equal(resumed[13],1);assert.equal(resumed[4],leafCount);assert.equal(resumed[12],1,'capacity growth must not restart publication');
const content=device.createBuffer({size:leafCount*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
const verify=device.createCommandEncoder();verify.copyBufferToBuffer(output,0,content,0,content.size);device.queue.submit([verify.finish()]);await content.mapAsync(GPUMapMode.READ);
assert.deepEqual([...new Uint32Array(content.getMappedRange())].sort((a,b)=>a-b),Array.from({length:leafCount},(_,i)=>i),'every synthetic quad survives prefix growth exactly once');content.unmap();content.destroy();oldOutput.destroy();
assert.deepEqual(validationErrors,[]);
console.log(JSON.stringify({adapterInfo,receipts,validationErrors},null,2));
for(const b of [state,pub,indirect,readback,output])b.destroy();device.destroy();
