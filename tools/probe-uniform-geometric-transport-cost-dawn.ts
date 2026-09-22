/** Per-pass hardware timings and optional frozen edge captures for Uniform Geometric. */
import assert from "node:assert/strict";
import { mkdir,writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU,type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { readBufferBinding } from "../lib/harness/webgpu-smoke-readbacks";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const passFilter=new RegExp(arg("passes","^(uv|Advect dense|Redistance dense)"));
const out=arg("out","/tmp/figure9-transport"),steps=Number(arg("steps","120"));
await acquireWebGPUExclusiveLock("dawn-probe","uniform geometric transport cost");
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
try {
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href) as NodeDawnProvider;Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);assert.ok(adapter.features.has("timestamp-query"));
 const base=managedGPUDevice(await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 let active=false;let labels:string[]=[];
 const queries=base.createQuerySet({type:"timestamp",count:512});
 const result=base.createBuffer({size:4096,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
 const read=base.createBuffer({size:4096,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 device=new Proxy(base,{get(target,key){if(key==="createCommandEncoder")return (descriptor?:GPUCommandEncoderDescriptor)=>{
  const encoder=target.createCommandEncoder(descriptor);let sampled=false;
  return new Proxy(encoder,{get(enc,field){
   if(field==="beginComputePass")return (desc?:GPUComputePassDescriptor)=>{
    const label=desc?.label??"";
    if(active&&passFilter.test(label)){
     const index=labels.length*2;assert.ok(index+1<512);labels.push(label);sampled=true;
     return enc.beginComputePass({...desc,timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});
    }
    return enc.beginComputePass(desc);
   };
   if(field==="finish")return ()=>{if(sampled){enc.resolveQuerySet(queries,0,labels.length*2,result,0);enc.copyBufferToBuffer(result,0,read,0,labels.length*16);}return enc.finish();};
   const value=Reflect.get(enc,field);return typeof value==="function"?value.bind(enc):value;
  }});
 };const value=Reflect.get(target,key);return typeof value==="function"?value.bind(target):value;}});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 const scene=sceneDocument(getSceneDefinition(arg("scene","cm12-figure-9")));const values=resolveMethodValues(uniformVolumeMethod,"balanced",JSON.parse(arg("values","{}")));
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 await mkdir(out,{recursive:true});await writeFile(`${out}/config.json`,JSON.stringify({values,scene},null,2));
 const rows:unknown[]=[];
 for(let frame=1;frame<=steps;frame++){
  labels=[];active=true;const start=performance.now();assert.ok(solver.advanceTo(frame/30));active=false;await device.queue.onSubmittedWorkDone();const wallMs=performance.now()-start;
  await read.mapAsync(GPUMapMode.READ);const times=new BigUint64Array(read.getMappedRange());const passes=labels.map((label,i)=>({label,ms:Number(times[2*i+1]!-times[2*i]!)/1e6}));read.unmap();
  const info=await solver.readStats();const work=Object.fromEntries(Object.entries(info).filter(([k])=>/uniformTransport|uniformTwoLevel|uniformLiquid|volumeCellSum|maximum|encodedSteps/i.test(k)));
  let donorFanIn:unknown;
  if(arg("capture","").split(",").map(Number).includes(frame)){
   const edges=(solver as unknown as {volumeEdges:GPUBuffer}).volumeEdges;
   const bytes=await readBufferBinding(device,{buffer:edges},edges.size);
   if(arg("dump-edges","off")==="on")await writeFile(`${out}/${frame}-edges.bin`,bytes);
   const donors=new Uint32Array(bytes.buffer);const weights=new Float32Array(bytes.buffer);const counts=new Uint32Array(solver.info.nx*solver.info.ny*solver.info.nz);
   let nonzero=0;for(let i=0;i<counts.length;i++)for(let k=0;k<9;k++)if(weights[10*i+1+k]!>0){
    const donor=k===8?i:(donors[10*i]!+(k&1)+solver.info.nx*(((k>>1)&1)+solver.info.ny*((k>>2)&1)))>>>0;
    counts[donor]!++;nonzero++;
   }
   let maximum=0,pairs=0,used=0;const top:{index:number,count:number}[]=[];
   for(let i=0;i<counts.length;i++){const count=counts[i]!;if(count>0)used++;maximum=Math.max(maximum,count);pairs+=count*(count-1)/2;if(count>100)top.push({index:i,count});}
   top.sort((a,b)=>b.count-a.count);donorFanIn={nonzero,used,maximum,pairs,top:top.slice(0,10)};
  }
  const row={frame,t:frame/30,wallMs,passes,work,donorFanIn};rows.push(row);console.log(JSON.stringify(row));await writeFile(`${out}/trace.json`,JSON.stringify(rows,null,2));assert.deepEqual(errors,[]);
 }
 queries.destroy();result.destroy();read.destroy();
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
