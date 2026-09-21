import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {UniformPageGeneration} from "../lib/methods/uniform/uniform-page-generation";
import {UniformPageRedistance} from "../lib/methods/uniform/uniform-page-redistance";
async function read(d:GPUDevice,src:GPUBuffer){const b=d.createBuffer({size:src.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const e=d.createCommandEncoder();e.copyBufferToBuffer(src,0,b,0,b.size);d.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);return b.getMappedRange().slice(0);}finally{if(b.mapState==="mapped")b.unmap();b.destroy();}}
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("page redistance preserves oblique planes and signed distant seam coordinates",{timeout:60000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform page redistance");let device:GPUDevice|undefined;
 try{const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const shift of [0,1000000]){
   const pool=await UniformPageGeneration.create(device,{capacity:8,requestCapacity:8,edge:16,initialCell:[0]});
   const solver=await UniformPageRedistance.create(device,pool,0,[1,1,1],16);
   try{
    const coords:number[][]=[];for(let z=-1;z<=0;z++)for(let y=-1;y<=0;y++)for(let x=-1;x<=0;x++)coords.push([x+shift,y,z]);
    const requests=new Uint32Array(4+4*coords.length);requests[0]=coords.length;coords.forEach((q,i)=>requests.set([...q,1],4+4*i));device.queue.writeBuffer(pool.requests,0,requests);
    let e=device.createCommandEncoder();pool.encodePrepare(e);pool.encodePublish(e);device.queue.submit([e.finish()]);
    const data=new Float32Array(8*16**3);
    for(let slot=0;slot<8;slot++)for(let z=0;z<16;z++)for(let y=0;y<16;y++)for(let x=0;x<16;x++){
     const q=coords[slot]!;const p=[(q[0]!-shift)*16+x,q[1]!*16+y,q[2]!*16+z];data[slot*16**3+x+16*(y+16*z)]=(p[0]!+2*p[1]!+2*p[2]!)/3-.2;
    }
    device.queue.writeBuffer(pool.fields,0,data);e=device.createCommandEncoder();solver.encode(e);device.queue.submit([e.finish()]);
    const actual=new Float32Array(await read(device,solver.result)),tags:Uint32Array=new Uint32Array(await read(device,solver.support));let maxError=0;
    for(let slot=0;slot<8;slot++)for(let z=1;z<15;z++)for(let y=1;y<15;y++)for(let x=1;x<15;x++){
     const i=slot*16**3+x+16*(y+16*z);if(Math.abs(data[i]!)>12)continue;
     maxError=Math.max(maxError,Math.abs(actual[i]!-data[i]!));assert.equal(tags[i],1);
    }
    console.log({shift,maxError});assert.ok(maxError<1e-4,`affine plane error ${maxError}`);
    for(let slot=0;slot<8;slot++)for(let z=0;z<16;z++)for(let y=0;y<16;y++)for(let x=0;x<16;x++){
     const q=coords[slot]!;data[slot*16**3+x+16*(y+16*z)]=Math.hypot((q[0]!-shift)*16+x,q[1]!*16+y,q[2]!*16+z)-6.4;
    }
    device.queue.writeBuffer(pool.fields,0,data);e=device.createCommandEncoder();solver.encode(e);device.queue.submit([e.finish()]);
    const curved=new Float32Array(await read(device,solver.result));let curvedMax=0,curvedMean=0,curvedCount=0;
    for(let i=0;i<data.length;i++)if(Math.abs(data[i]!)<2){const error=Math.abs(curved[i]!-data[i]!);curvedMax=Math.max(curvedMax,error);curvedMean+=error;curvedCount++;}
    console.log({shift,curvedMax,curvedMean:curvedMean/curvedCount});
    assert.ok(curvedMax<.3,"near-interface sphere metric error below .3 cells");

   }finally{solver.destroy();pool.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
