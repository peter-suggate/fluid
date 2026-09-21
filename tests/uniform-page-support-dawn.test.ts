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
  support=await UniformPageSupport.create(device,pool,{volume:0,phi:1,velocity:[2,3,4],volumeKind:"physical"});
  const d=device,p=pool,s=support;
  s.setStep([1,1,1],1,.25,[0,0,0]);
  d.queue.writeBuffer(p.requests,0,new Uint32Array([1,0,0,0,0xffffffff,0,0,1]));
  let e=d.createCommandEncoder();p.encodePrepare(e);p.encodePublish(e);d.queue.submit([e.finish()]);
  // V must retain a page even if phi claims it is ambient air.
  d.queue.writeBuffer(p.fields,0,new Float32Array([.5,1,0,0,0]));
  const advance=()=>{const e=d.createCommandEncoder();s.encode(e);p.encodePrepare(e);p.encodePublish(e);d.queue.submit([e.finish()]);};
  advance();let meta=await read(d,p.accepted);
  assert.equal(meta[1],8);assert.equal(meta[2],7);assert.equal(meta[0],2);
  const coordinates=new Set(Array.from(meta.slice(p.layout.activeBase,p.layout.activeBase+meta[1]!)).map(slot=>Array.from(meta.slice(16+16*slot,19+16*slot),n=>n|0).join("/")));
  for(let z=-1;z<=0;z++)for(let y=-1;y<=0;y++)for(let x=-2;x<=-1;x++)assert.ok(coordinates.has(`${x}/${y}/${z}`));
  assert.equal(new Float32Array((await read(d,p.fields)).buffer)[0],.5);
  // Deep liquid survives even when its conservative field is temporarily zero.
  d.queue.writeBuffer(p.fields,0,new Float32Array([0,-1,0,0,0]));advance();
  assert.equal((await read(d,p.accepted))[1],8);
  d.queue.writeBuffer(p.fields,0,new Float32Array([.5,1,0,0,0]));
  meta=await read(d,p.accepted);
  const before=meta,fields=await read(d,p.fields);
  // A remote source requires 27 more slots: refuse atomically, retaining mass.
  d.queue.writeBuffer(s.sources,0,new Uint32Array([1,0,0,0,1000000,0,0,1]));
  advance();assert.equal((await read(d,p.candidate))[4],2);assert.deepEqual(await read(d,p.accepted),before);assert.deepEqual(await read(d,p.fields),fields);
  // Current GPU velocity and backward queries expand x support without a host
  // diagnostics sample. Forty unique support pages exceed this pool.
  d.queue.writeBuffer(s.sources,0,new Uint32Array(4));
  d.queue.writeBuffer(p.fields,8,new Float32Array([60]));advance();
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

(modulePath?test:test.skip)("Local support follows occupancy, closes over air velocity, and preserves physical units",{timeout:60000},async t=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform minimal residency support");
 let device:GPUDevice|undefined;
 try {
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();const d=device;
  const errors:string[]=[];d.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  async function fixture(coords:number[][], body:(p:UniformPageGeneration,s:UniformPageSupport,write:(slot:number,xyz:number[],value:number[])=>void,run:()=>Promise<Uint32Array>)=>Promise<void>,kind:"physical"|"full-cell-fraction"|"open-cell-fraction"="physical") {
   const p=await UniformPageGeneration.create(d,{capacity:32,requestCapacity:512,edge:32,initialCell:[0,1,0,0,0,1]});
   const s=await UniformPageSupport.create(d,p,{volume:0,phi:1,velocity:[2,3,4],volumeKind:kind,openCellFraction:5});
   try {
    d.queue.writeBuffer(p.requests,0,new Uint32Array([coords.length,0,0,0,...coords.flatMap(q=>[...q,8])]));
    const e=d.createCommandEncoder();p.encodePrepare(e);p.encodePublish(e);d.queue.submit([e.finish()]);
    const write=(slot:number,xyz:number[],value:number[])=>d.queue.writeBuffer(p.fields,4*6*(slot*32**3+xyz[0]!+32*(xyz[1]!+32*xyz[2]!)),new Float32Array(value));
    s.setStep([1,1,1],1,.25,[0,0,0]);
    const run=async()=>{const e=d.createCommandEncoder();s.encode(e);d.queue.submit([e.finish()]);return read(d,p.requests);};
    await body(p,s,write,run);
   } finally{s.destroy();p.destroy();}
  }
  const entries=(r:Uint32Array)=>Array.from({length:r[0]!},(_,i)=>({q:Array.from(r.slice(4+4*i,7+4*i),x=>x|0),roles:r[7+4*i]!}));
  const unique=(r:Uint32Array)=>new Set(entries(r).map(({q})=>q.join("/")));
  await t.test("an interior cell needs one page and one occupied 4-cubed tile",async()=>{
   await fixture([[-1000000,0,0]],async(p,s,write,run)=>{
    write(0,[16,16,16],[.5,1,0,0,0,1]);
    const r=await run();assert.equal(r[1],0);assert.deepEqual([...unique(r)],["-1000000/0/0"]);
    const summary=await read(d,s.summaries);
    assert.deepEqual(Array.from(summary.slice(1,7)),[16,16,16,17,17,17]);assert.equal(summary[15],1);assert.equal(summary[23],1);
    assert.equal(new Float32Array(summary.buffer)[13],.5);
    // Clearing a seed must not let retained stencil-only roles seed another halo.
    write(0,[16,16,16],[0,1,0,0,0,1]);assert.equal((await run())[0],0);
   });
  });
  await t.test("a distant fast jet does not inflate a quiet pond",async()=>{
   await fixture([[0,0,0],[1000000,0,0]],async(p,s,write,run)=>{
    write(0,[16,16,16],[1,1,0,0,0,1]);write(1,[16,16,16],[1,1,80,0,0,1]);
    const r=await run();assert.equal(r[1],0);
    assert.deepEqual([...unique(r)].filter(key=>Math.abs(Number(key.split("/")[0]))<100),["0/0/0"]);
    const destinations=entries(r).filter(e=>(e.roles&32)!==0&&e.q[0]!>100);
    assert.deepEqual(destinations.map(e=>e.q[0]),[1000000,1000001,1000002,1000003]);
    assert.ok(unique(r).has("999997/0/0"),"backward reads are also requested");
   });
  });
  await t.test("neighboring air velocity participates in local closure",async()=>{
   await fixture([[0,0,0],[1,0,0]],async(p,s,write,run)=>{
    write(0,[31,16,16],[1,1,0,0,0,1]);write(1,[0,16,16],[0,1,80,0,0,1]);
    const r=await run();assert.equal(r[1],0);assert.ok(r[2]!>=2);assert.ok(unique(r).has("3/0/0"));
    assert.ok(unique(r).has("-2/0/0"));
   });
  });
  await t.test("half-open cell endpoints and signed INT limits do not wrap",async()=>{
   await fixture([[-2147483648,0,0],[2147483647,0,0]],async(p,s,write,run)=>{
    s.setStep([1,1,1],0,.25,[0,0,0],0);
    write(0,[0,0,0],[1,1,0,0,0,1]);write(1,[31,31,31],[1,1,0,0,0,1]);
    const r=await run();assert.equal(r[1],0);assert.equal(unique(r).size,2);
    s.setStep([1,1,1],0,.25,[0,0,0],1);assert.equal((await run())[1],2);
   });
  });
  await t.test("an unresolved velocity chain rejects without modifying accepted state",async()=>{
   await fixture(Array.from({length:12},(_,i)=>[i,0,0]),async(p,s,write,run)=>{
    write(0,[31,16,16],[1,1,0,0,0,1]);
    for(let i=1;i<12;i++)write(i,[0,16,16],[0,1,i*32,0,0,1]);
    const before=await read(d,p.accepted),fields=await read(d,p.fields);
    const r=await run();assert.equal(r[1],4);assert.equal(r[2],8);
    const e=d.createCommandEncoder();p.encodePrepare(e);p.encodePublish(e);d.queue.submit([e.finish()]);
    assert.deepEqual(await read(d,p.accepted),before);assert.deepEqual(await read(d,p.fields),fields);
   });
  });
  for(const kind of ["physical","full-cell-fraction","open-cell-fraction"] as const){
   await t.test(`${kind} volume is converted exactly once`,async()=>{
    await fixture([[0,0,0]],async(p,s,write,run)=>{
     s.setStep([2,3,4],0,.25,[0,0,0]);write(0,[16,16,16],[.5,1,0,0,0,.25]);await run();
     const summary=await read(d,s.summaries),physical=new Float32Array(summary.buffer)[13];
     assert.equal(physical,kind==="physical"?.5:kind==="full-cell-fraction"?12:3);
    },kind);
   });
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
