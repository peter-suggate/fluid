import assert from "node:assert/strict";
import test from "node:test";
import { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import { SVO_SURFACE_MESH_STATE as W, SVO_SURFACE_MESH_STATE_BYTES } from "../lib/svo/features/primary-visibility/svo-surface-mesh";

function renderer(){
 const groups:GPUBindGroupDescriptor[]=[];
 const buffers:{label?:string;size:number;destroyed:boolean;destroy():void}[]=[];
 const device={limits:{maxStorageBufferBindingSize:1<<24,maxBufferSize:1<<24},
  createBuffer:(d:GPUBufferDescriptor)=>{const b={label:d.label,size:d.size,destroyed:false,destroy(){this.destroyed=true;}};buffers.push(b);return b;},
  createBindGroupLayout:(d:unknown)=>d,
  createBindGroup:(d:GPUBindGroupDescriptor)=>{groups.push(d);return d;},
  queue:{writeBuffer:()=>{},onSubmittedWorkDone:()=>Promise.resolve()}};
 const host=Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype),{
  device,experiments:{surfaceMesh:true},surfaceMeshEpoch:0,surfaceMeshHostFront:0,surfaceMeshBackGeneration:0,
  surfaceMeshArenas:[undefined,undefined],surfaceMeshEmptyArenas:[undefined,undefined],surfaceMeshMaximumBytes:1<<24,
 });
 return {host,device,groups,buffers};
}

test("mesh reset ignores a delayed arena flip from the previous builder",async()=>{
 const {host,device}=renderer();
 host.surfaceMeshState=device.createBuffer({size:SVO_SURFACE_MESH_STATE_BYTES,usage:0});
 const current=device.createBuffer({size:4096,usage:0}),back=device.createBuffer({size:4096,usage:0});
 host.surfaceMeshArenas=[current,back];host.bindSurfaceMeshBuffers=()=>{};
 // The GPU flipped to slot 1, but its readback has not reached the host.
 const words=new Uint32Array(W.wordCount);words[W.front]=1;words[W.usable]=1;
 // A structural reset starts a fresh build using the host's current slot 0.
 host.resetSurfaceMeshState();const status=host.surfaceMeshStatus;
 host.applySurfaceMeshReceipt(words,[4096,4096],0);
 await Promise.resolve();
 assert.equal(host.surfaceMeshHostFront,0);assert.equal(host.surfaceMeshArenas[0],current);
 assert.equal(current.destroyed,false);assert.equal(host.surfaceMeshStatus,status);
 // The new builder can still publish its own replacement normally.
 host.applySurfaceMeshReceipt(words,[4096,4096],host.surfaceMeshEpoch);
 await Promise.resolve();
 assert.equal(host.surfaceMeshHostFront,1);assert.equal(host.surfaceMeshArenas[1],back);
 assert.equal(current.destroyed,true);assert.equal(back.destroyed,false);
 assert.notEqual(host.surfaceMeshStatus,status);
});

test("absent mesh arena slots have distinct writable fallback buffers",()=>{
 const saved={GPUBufferUsage:globalThis.GPUBufferUsage,GPUShaderStage:globalThis.GPUShaderStage};
 Object.assign(globalThis,{GPUBufferUsage:{STORAGE:1,MAP_READ:2,COPY_DST:4,INDIRECT:8,COPY_SRC:16},GPUShaderStage:{COMPUTE:1,VERTEX:2,FRAGMENT:4}});
 try{
  const {host,groups}=renderer();host.ensureSurfaceMeshBuffers();
  host.surfaceMeshArenas=[undefined,undefined];host.bindSurfaceMeshBuffers();
  const bindings=Array.from(groups.at(-2)!.entries);
  const buffers=bindings.map(e=>(e.resource as GPUBufferBinding).buffer);
  assert.equal(new Set(buffers).size,buffers.length,"writable mesh bindings never alias");
  assert.notEqual(host.surfaceMeshEmptyArenas[0],host.surfaceMeshEmptyArenas[1]);
 }finally{Object.assign(globalThis,saved);}
});
