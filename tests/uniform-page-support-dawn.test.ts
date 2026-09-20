import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformPageGeneration } from "../lib/methods/uniform/uniform-page-generation";
import { UniformPageSupport } from "../lib/methods/uniform/uniform-page-support";

async function read(device:GPUDevice,buffer:GPUBuffer){
 const b=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
 try{const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,0,b,0,b.size);device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);return new Uint32Array(b.getMappedRange().slice(0));}
 finally{if(b.mapState==="mapped")b.unmap();b.destroy();}
}
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("GPU support grows signed frontiers, retains mass and rejects insufficient budgets",{timeout:60000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform page support");
 let device:GPUDevice|undefined,pool:UniformPageGeneration|undefined,support:UniformPageSupport|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  pool=await UniformPageGeneration.create(device,{capacity:32,requestCapacity:128,edge:16,initialCell:[0,1,0,0,0]});
  support=await UniformPageSupport.create(device,pool,{volume:0,phi:1,velocity:[2,3,4]});
  const d=device,p=pool,s=support;
  s.setStep([1,1,1],1,.25,[0,0,0]);
  d.queue.writeBuffer(p.requests,0,new Uint32Array([1,0,0,0,0xffffffff,0,0,1]));
  let e=d.createCommandEncoder();p.encodePrepare(e);p.encodePublish(e);d.queue.submit([e.finish()]);
  // V must retain a page even if phi claims it is ambient air.
  d.queue.writeBuffer(p.fields,0,new Float32Array([.5,1,0,0,0]));
  const advance=()=>{const e=d.createCommandEncoder();s.encode(e);p.encodePrepare(e);p.encodePublish(e);d.queue.submit([e.finish()]);};
  advance();let meta=await read(d,p.accepted);
  assert.equal(meta[1],27);assert.equal(meta[2],26);assert.equal(meta[0],2);
  const coordinates=new Set(Array.from(meta.slice(p.layout.activeBase,p.layout.activeBase+meta[1]!)).map(slot=>Array.from(meta.slice(16+16*slot,19+16*slot),n=>n|0).join("/")));
  for(let z=-1;z<=1;z++)for(let y=-1;y<=1;y++)for(let x=-2;x<=0;x++)assert.ok(coordinates.has(`${x}/${y}/${z}`));
  assert.equal(new Float32Array((await read(d,p.fields)).buffer)[0],.5);
  // Deep liquid survives even when its conservative field is temporarily zero.
  d.queue.writeBuffer(p.fields,0,new Float32Array([0,-1,0,0,0]));advance();
  assert.equal((await read(d,p.accepted))[1],27);
  d.queue.writeBuffer(p.fields,0,new Float32Array([.5,1,0,0,0]));
  meta=await read(d,p.accepted);
  const before=meta,fields=await read(d,p.fields);
  // A remote source requires 27 more slots: refuse atomically, retaining mass.
  d.queue.writeBuffer(s.sources,0,new Uint32Array([1,0,0,0,1000000,0,0,1]));
  advance();assert.equal((await read(d,p.candidate))[4],2);assert.deepEqual(await read(d,p.accepted),before);assert.deepEqual(await read(d,p.fields),fields);
  // Current GPU velocity expands x reach to two pages, rather than relying on a
  // previous diagnostics sample. Forty-five requested pages exceed this pool.
  d.queue.writeBuffer(s.sources,0,new Uint32Array(4));
  d.queue.writeBuffer(p.fields,8,new Float32Array([20]));advance();
  assert.equal((await read(d,p.requests))[0],45);assert.deepEqual(await read(d,p.accepted),before);
  d.queue.writeBuffer(p.fields,8,new Float32Array([0]));
  // Excessive motion must fail requests, never cap the support halo silently.
  d.queue.writeBuffer(s.sources,0,new Uint32Array(4));s.setStep([1,1,1],1,.25,[100000,0,0]);
  advance();assert.equal((await read(d,p.requests))[1],1);assert.deepEqual(await read(d,p.accepted),before);
  // Poisoned input also refuses publication, rather than allowing NaN to retire it.
  s.setStep([1,1,1],1,.25,[0,0,0]);d.queue.writeBuffer(p.fields,8,new Float32Array([NaN]));advance();
  assert.equal((await read(d,p.requests))[1],3);assert.deepEqual(await read(d,p.accepted),before);
  // No remaining V/interface/source means the producer may request an empty set.
  d.queue.writeBuffer(p.fields,0,new Float32Array([0,1,0,0,0]));advance();meta=await read(d,p.accepted);assert.equal(meta[1],0);
  // Seed a remote source entirely from its external command, then initialize on GPU.
  d.queue.writeBuffer(s.sources,0,new Uint32Array([1,0,0,0,0xfff0bdc0,20,0,1]));advance();
  meta=await read(d,p.accepted);assert.equal(meta[1],27);assert.equal(meta[2],27);
  assert.deepEqual(errors,[]);
 }finally{support?.destroy();pool?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
