import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformPageGeneration, uniformPageFieldAccessWGSL } from "../lib/methods/uniform/uniform-page-generation";
import { planUniformPages, type UniformPageCoordinate, UNIFORM_PAGE_MISSING } from "../lib/methods/uniform/uniform-page-layout";

async function read(device: GPUDevice, buffer: GPUBuffer): Promise<Uint32Array> {
 const staging=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try {
  const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,0,staging,0,buffer.size);device.queue.submit([e.finish()]);
  await staging.mapAsync(GPUMapMode.READ);return new Uint32Array(staging.getMappedRange().slice(0));
 }finally{if(staging.mapState==="mapped")staging.unmap();staging.destroy();}
}
const modulePath=process.env.WEBGPU_NODE_MODULE;
for(const edge of [16,32] as const)
(modulePath?test:test.skip)(`${edge}³ GPU pages preserve fields, signed seams and accepted state on exhaustion`,{timeout:60000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform GPU residency transactions");
 let device:GPUDevice|undefined;let pool:UniformPageGeneration|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  pool=await UniformPageGeneration.create(device,{capacity:4,requestCapacity:8,edge,initialCell:[0,8,-0,1]});
  const d=device,p=pool,bytes=p.allocatedBytes;
  const submit=(coordinates:readonly UniformPageCoordinate[],roles?:number[],overrideCount?:number)=>{
   const request=new Uint32Array(p.layout.requestWords);request[0]=overrideCount??coordinates.length;
   coordinates.forEach((q,i)=>request.set([...q,roles?.[i]??1],4+4*i));
   d.queue.writeBuffer(p.requests,0,request);
   const e=d.createCommandEncoder();p.encodePrepare(e);p.encodePublish(e);d.queue.submit([e.finish()]);
  };
  const slots=(meta:Uint32Array)=>new Map(Array.from(meta.slice(p.layout.activeBase,p.layout.activeBase+meta[1]!)).map(slot=>[
   Array.from(meta.slice(16+16*slot,19+16*slot),n=>n|0).join("/"),slot]));
  const kernel=async(code:string,buffers:GPUBuffer[])=>{
   const layout=d.createBindGroupLayout({entries:buffers.map((_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}))});
   const pipeline=await d.createComputePipelineAsync({layout:d.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module:d.createShaderModule({code}),entryPoint:"main"}});
   const group=d.createBindGroup({layout,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
   return (e:GPUCommandEncoder)=>{const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();};
  };
  const initial:UniformPageCoordinate[]=[[-1,0,0],[0,0,0],[1000000,-2000000,17]];
  submit([...initial,initial[0]!],[1,2,4,8]);
  let meta=await read(d,p.accepted),map=slots(meta);
  assert.equal(meta[0],1);assert.equal(meta[1],3);assert.equal(meta[2],3);
  assert.equal(meta[16+16*map.get("-1/0/0")!+3],9,"duplicate roles are unioned");
  const oracle=planUniformPages(edge,4,initial);
  for(const slot of oracle.activeSlots)assert.deepEqual(meta.slice(16+16*slot+4,16+16*slot+10),oracle.neighbors.slice(6*slot,6*slot+6));
  let fields=await read(d,p.fields);const floats=new Float32Array(fields.buffer);
  for(let slot=0;slot<3;slot++)for(let cell=0;cell<edge**3;cell++)assert.deepEqual(Array.from(floats.slice((slot*edge**3+cell)*4,(slot*edge**3+cell)*4+4)),[0,8,-0,1]);
  const addresses=d.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try {
   const lookup=await kernel(`
    @group(0) @binding(0) var<storage,read_write> accepted:array<u32>;
    @group(0) @binding(1) var<storage,read_write> result:array<u32>;
    ${uniformPageFieldAccessWGSL(p.options)}
    @compute @workgroup_size(1) fn main(){
     result[0]=pageFieldAddress(vec3i(-1,0,0),vec3i(${edge},0,0),1u);
     result[1]=pageFieldAddress(vec3i(0),vec3i(0),1u);
     result[2]=pageFieldAddress(vec3i(0),vec3i(-1,0,0),2u);
     result[3]=pageFieldAddress(vec3i(-1,0,0),vec3i(${edge-1},0,0),2u);
     result[4]=pageFieldAddress(vec3i(0),vec3i(0,${edge},0),0u);
     result[5]=pageFieldAddress(vec3i(1000001,-2000000,17),vec3i(-${edge},0,0),0u);
     result[6]=pageFieldAddress(vec3i(2147483647,0,0),vec3i(${edge},0,0),0u);
     result[7]=pageFieldAddress(vec3i(-2147483647-1,0,0),vec3i(-1,0,0),0u);
    }`,[p.accepted,addresses]);
   const e=d.createCommandEncoder();lookup(e);d.queue.submit([e.finish()]);
   const result=await read(d,addresses);
   assert.equal(result[0],result[1]);assert.equal(result[2],result[3]);
   assert.equal(result[0],edge**3*4+1);assert.equal(result[2],(edge-1)*4+2);
   assert.equal(result[4],UNIFORM_PAGE_MISSING);assert.equal(result[5],2*edge**3*4);
   assert.equal(result[6],UNIFORM_PAGE_MISSING);assert.equal(result[7],UNIFORM_PAGE_MISSING);
  }finally{addresses.destroy();}
  // A retained page's payload stands in for persistent fluid, history and velocity.
  const retained=map.get("0/0/0")!;d.queue.writeBuffer(p.fields,retained*edge**3*16,new Float32Array([.75,-.5,3,1]));
  submit([[0,0,0],[1000000,-2000000,17],[0,1,0]]);
  meta=await read(d,p.accepted);map=slots(meta);
  assert.equal(meta[0],2);assert.equal(meta[2],1);assert.equal(meta[3],1);
  assert.equal(map.get("0/0/0"),retained);assert.equal(map.get("0/1/0"),3,"retiring slot is quarantined");
  fields=await read(d,p.fields);assert.deepEqual(Array.from(new Float32Array(fields.buffer).slice(retained*edge**3*4,retained*edge**3*4+4)),[.75,-.5,3,1]);
  submit([[0,0,0],[0,1,0],[-1000000,3000000,-7]]);
  meta=await read(d,p.accepted);map=slots(meta);
  assert.equal(map.get("-1000000/3000000/-7"),0,"previously retired slot can now be reused");
  assert.equal(p.allocatedBytes,bytes,"distance changes cannot grow storage");
  const before=meta, beforeFields=await read(d,p.fields);
  // Two new requests but only one old free slot. No partial generation or field init.
  submit([[0,0,0],[0,1,0],[4,5,6],[7,8,9]]);
  assert.deepEqual(await read(d,p.accepted),before);assert.deepEqual(await read(d,p.fields),beforeFields);
  assert.equal((await read(d,p.candidate))[4],2);
  submit([],undefined,9);
  assert.equal((await read(d,p.candidate))[4],1);assert.deepEqual(await read(d,p.accepted),before);
  // Emptying releases all slots; refilling is a separate ordered transaction.
  submit([]);submit([[-2147483648,0,0],[2147483647,0,0]]);
  meta=await read(d,p.accepted);assert.equal(meta[1],2);
  for(const slot of slots(meta).values())assert.ok(meta.slice(16+16*slot+4,16+16*slot+10).every(n=>n===UNIFORM_PAGE_MISSING),"i32 neighbors do not wrap across the world");
  // A GPU consumer produces the next support set; preparation and publication
  // follow it in the same submission with no host count/readback/continuation.
  const produce=await kernel(`
   @group(0) @binding(0) var<storage,read_write> accepted:array<u32>;
   @group(0) @binding(1) var<storage,read_write> requests:array<u32>;
   @compute @workgroup_size(1) fn main(){
    requests[0]=accepted[1];
    for(var i=0u;i<accepted[1];i++){
     let slot=accepted[${p.layout.activeBase}u+i];let at=16u+16u*slot;
     requests[4u+4u*i]=accepted[at];requests[5u+4u*i]=accepted[at+1u];
     requests[6u+4u*i]=accepted[at+2u];requests[7u+4u*i]=8u;
    }
   }`,[p.accepted,p.requests]);
  const e=d.createCommandEncoder();produce(e);p.encodePrepare(e);p.encodePublish(e);d.queue.submit([e.finish()]);
  const next=await read(d,p.accepted);assert.equal(next[0],meta[0]!+1);assert.equal(next[2],0);
  for(const slot of slots(next).values())assert.equal(next[16+16*slot+3],8);
  assert.deepEqual(errors,[]);
 }finally{pool?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
