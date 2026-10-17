/** Record actual device buffer requests, including failures, at a visual rung. */
import "./register-assets";
import { mkdirSync, writeFileSync } from "node:fs";
import { getScenePreset } from "../../lib/core/scenes";
import { WebGPULiveSvoScene } from "../../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import { createDawnRenderDevice } from "../svo-dry-frame-harness";
const { device, validationErrors } = await createDawnRenderDevice();
const buffers: {label:string; bytes:number}[]=[];
const directory="artifacts/pool-party-memory";mkdirSync(directory,{recursive:true});
const output=`${directory}/depth-${process.env.FLUID_POOL_DEPTH ?? 3}.json`;
const save=(extra={})=>writeFileSync(output,JSON.stringify({buffers:[...buffers].sort((a,b)=>b.bytes-a.bytes),
  cumulativeRequestedBytes:buffers.reduce((n,b)=>n+b.bytes,0),...extra},null,2));
const measured=new Proxy(device,{get(target,key){
  if(key==="queue" && process.env.FLUID_TRACE_WRITES) return new Proxy(target.queue,{get(queue,method){
    if(method==="writeBuffer")return (buffer:GPUBuffer,offset:number,data:ArrayBufferView,dataOffset?:number,size?:number)=>{
      console.log("WRITE",buffer.label,buffer.size,offset,data.byteLength,dataOffset,size);
      return queue.writeBuffer(buffer,offset,data as GPUAllowSharedBufferSource,dataOffset,size);
    };
    const v=Reflect.get(queue,method,queue);return typeof v==="function"?v.bind(queue):v;
  }});
  if(key==="createBuffer")return (descriptor:GPUBufferDescriptor)=>{
    buffers.push({label:descriptor.label??"unlabelled",bytes:Number(descriptor.size)});save();
    if(Number(descriptor.size)>1048576)console.log(`${descriptor.label}: ${(Number(descriptor.size)/1048576).toFixed(1)} MiB`);
    return target.createBuffer(descriptor);
  };
  const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
}});
let world:WebGPULiveSvoScene|undefined;
try{
  world=await WebGPULiveSvoScene.create(measured,getScenePreset("backyard-pool-party").create(),"balanced",p=>console.log(p.label),undefined,
    {environmentRefinementDepth:Number(process.env.FLUID_POOL_DEPTH??3),surfaceDualMarchingCubes:true});
  save({builtRefinementDepth:world.builtRefinementDepth,
    visualCellSize_m: world.sparseVoxelSceneSource.structural?.domain.cellSize_m,
    info:world.info,validationErrors});
  if(world.builtRefinementDepth!==Number(process.env.FLUID_POOL_DEPTH??3))throw new Error("Requested detail fell back");
  if(validationErrors.length)throw new Error(validationErrors.join("\n"));
}finally{world?.destroy();device.destroy();}
