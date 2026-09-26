import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedLayoutBuilder} from "../lib/methods/uniform/uniform-mixed-layout-builder";
import {createUniformMixedLayout,mixedCellWidth,uniformMixedPressureLevel} from "../lib/methods/uniform/uniform-mixed-layout";

// The GPU builder must upload exactly what UniformMixedOwnership.update()
// uploads for the CPU layout: every writeBuffer range, byte for byte.
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("GPU layout builder reproduces the CPU ownership uploads",{timeout:300000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed layout builder");
 let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice({requiredLimits:{maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const real=device;
  // Records ownership uploads by buffer label while `recording`.
  let recording=false;const writes:{label:string;offset:number;data:Uint32Array}[]=[];
  const queue=new Proxy(real.queue,{get(target,key){
   if(key==="writeBuffer")return(buffer:GPUBuffer,offset:number,data:Uint32Array)=>{
    if(recording)writes.push({label:buffer.label,offset,data:new Uint32Array((data.buffer as ArrayBuffer).slice(data.byteOffset,data.byteOffset+data.byteLength))});
    target.writeBuffer(buffer,offset,data as Uint32Array<ArrayBuffer>);
   };
   const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
  }});
  const recorded:GPUDevice=new Proxy(real,{get(target,key){
   if(key==="queue")return queue;
   const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
  }});
  const read=async(buffer:GPUBuffer,offset:number,bytes:number)=>{
   const staging=real.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const e=real.createCommandEncoder();e.copyBufferToBuffer(buffer,offset,staging,0,bytes);real.queue.submit([e.finish()]);
   await staging.mapAsync(GPUMapMode.READ);const words=new Uint32Array(staging.getMappedRange().slice(0));staging.unmap();staging.destroy();return words;
  };
  // Seeded blobs: bands, isolated tiles, and runs against the domain walls.
  let seed=12345;const random=()=>((seed=(seed*1103515245+12345)>>>0)/4294967296);
  const blobs=(t:readonly number[],count:number,radius:number)=>{
   const mask=new Uint8Array(t[0]!*t[1]!*t[2]!);
   for(let b=0;b<count;b++){
    const c=t.map(n=>random()*n),r=radius*(0.3+random());
    for(let z=0;z<t[2]!;z++)for(let y=0;y<t[1]!;y++)for(let x=0;x<t[0]!;x++)
     if(Math.max(Math.abs(x-c[0]!),Math.abs(y-c[1]!),Math.abs(z-c[2]!))<=r)mask[x+t[0]!*(y+t[1]!*z)]=1;
   }
   return mask;
  };
  for(const dimensions of [[64,48,80],[256,128,160]] as const){
   const lattice={dimensions,cellSize_m:[0.01,0.01,0.01] as const,origin_m:{x:0,y:0,z:0}};
   const t=dimensions.map(n=>n/4),n=t[0]!*t[1]!*t[2]!;
   const fine=createUniformMixedLayout(lattice,[]);
   const current:[UniformMixedOwnership,UniformMixedOwnership]=[new UniformMixedOwnership(real,fine),new UniformMixedOwnership(real,uniformMixedPressureLevel(fine,2))];
   const bandWords=12+Math.ceil(n/32),bandBuffer=real.createBuffer({size:bandWords*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   const builder=new UniformMixedLayoutBuilder(real,{buffer:bandBuffer,wordOffset:12},current);await builder.initialize();
   for(let round=0;round<3;round++){
    const band=blobs(t,6+4*round,Math.max(...t)/10),statics=blobs(t,3,2);
    const bits=new Uint32Array(bandWords).fill(0xdeadbeef);bits.fill(0,12);
    for(let k=0;k<n;k++)if(band[k])bits[12+(k>>5)]!|=1<<(k&31);
    real.queue.writeBuffer(bandBuffer,0,bits);
    builder.setStatic(statics,[]);
    const expected=createUniformMixedLayout(lattice,[],true,4,band.map((b,k)=>b|statics[k]!));
    const expectedLevels=[expected,uniformMixedPressureLevel(expected,2)] as const;
    assert.ok(expected.fineTiles.length&&expected.transitionTiles.length&&expected.coarseTiles.length,"fixture must mix all three widths");
    const widths=(layout:typeof expected)=>Array.from(layout.tiles,mixedCellWidth);
    const before=current.map(o=>widths(o.layout));
    const e=real.createCommandEncoder();builder.encode(e);real.queue.submit([e.finish()]);
    const built=await builder.read();
    for(let level=0;level<2;level++){
     const b=built[level]!,x=expectedLevels[level]!,tag=`${dimensions} round ${round} L${level}`;
     assert.equal(b.changedTiles,widths(x).filter((w,k)=>w!==before[level]![k]).length,`${tag}: changed tiles`);
     assert.deepEqual(b.layout.tiles,x.tiles,`${tag}: tile words`);
     assert.equal(b.layout.cellCount,x.cellCount,`${tag}: cell count`);
     for(const key of ["fineTiles","transitionTiles","coarseTiles","stencils"] as const)assert.deepEqual(b.layout[key],x[key],`${tag}: lazy ${key}`);
     // Every range update() writes, against the builder's buffers.
     writes.length=0;recording=true;const reference=new UniformMixedOwnership(recorded,x);recording=false;
     assert.equal(b.hangingSlots,reference.hangingSlots,`${tag}: hanging slots`);
     const sources:Record<string,[GPUBuffer,number]>={
      "Uniform mixed owners and tier worklists":[b.source.topology,0],
      "Uniform mixed work counts":[b.source.counts.buffer,b.source.counts.offset],
      "Uniform shared frame support and certified work":[b.source.support,0],
      "Uniform certified fine dispatch":[b.source.support,(4*n+4)*4],
      "Uniform mixed velocity tap cache":[b.source.slots,0],
     };
     assert.ok(writes.length>=10,`${tag}: recorded ${writes.length} uploads`);
     for(const w of writes){
      const source=sources[w.label];assert.ok(source,`${tag}: unexpected upload to ${w.label}`);
      const actual=await read(source[0],source[1]+w.offset,w.data.byteLength);
      const first=actual.findIndex((v,i)=>v!==w.data[i]);
      assert.equal(first,-1,`${tag}: ${w.label} @${w.offset} word ${first}: GPU ${actual[first]} CPU ${w.data[first]}`);
     }
     reference.destroy();
    }
    // Adopting and rebuilding the same band must report no change.
    const adopt=real.createCommandEncoder();current[0].adopt(adopt,built[0]);current[1].adopt(adopt,built[1]);real.queue.submit([adopt.finish()]);
    const again=real.createCommandEncoder();builder.encode(again);real.queue.submit([again.finish()]);
    assert.deepEqual((await builder.read()).map(l=>l.changedTiles),[0,0],`${dimensions} round ${round}: rebuild after adopt`);
   }
   assert.deepEqual(errors,[]);
   builder.destroy();for(const o of current)o.destroy();bandBuffer.destroy();
  }
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
