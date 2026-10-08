import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice,gpuCompilationManagerFor} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {createUniformMixedLayoutFromWidths} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedTransportStage} from "../lib/methods/uniform/uniform-mixed-transport";
import {uniformTransportWorkgroupReference,restoreUniformTransportWorkgroupDispatch} from "./helpers/uniform-transport-workgroup-reference";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("batched-row transport matches the original across live h/4h changes and donor footprints",{timeout:120_000},async()=>{
 let device:GPUDevice|undefined;
 const owned:{destroy():void}[]=[];
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const adapter=await createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]).requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const d=16,cells=d**3,dims=[d,d,d] as const;
  const lattice={dimensions:dims,origin_m:{x:0,y:0,z:0},cellSize_m:[1,1,1] as const};
  const layouts=[new Uint8Array(64).fill(4),Uint8Array.from({length:64},(_,i)=>i%4<2?1:4),
   // Partial four-row batches: 63 and 1 coarse rows, beside h donors.
   Uint8Array.from({length:64},(_,i)=>i===0?1:4),Uint8Array.from({length:64},(_,i)=>i===63?4:1),
   new Uint8Array(64).fill(1),new Uint8Array(64).fill(4)].map(w=>createUniformMixedLayoutFromWidths(lattice,w,[]));
  const texture=(size:number,format:GPUTextureFormat)=>{const t=device!.createTexture({size:[size,size,size],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC});owned.push(t);return t;};
  const write=(t:GPUTexture,a:Float32Array<ArrayBuffer>,components=1)=>device!.queue.writeTexture({texture:t},a,{bytesPerRow:t.width*4*components,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
  const input=texture(d,"r32float"),departure=texture(d,"rgba32float"),phi=texture(d+1,"r32float");
  const values=Float32Array.from({length:cells},(_,i)=>(i%17<5?0:((i*13)%101)/100));write(input,values);write(phi,new Float32Array((d+1)**3).fill(-100));
  const params=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});owned.push(params);
  const reductions=device.createBuffer({size:48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});owned.push(reductions);
  let reference=false,referenceModules=0;
  const compiler=gpuCompilationManagerFor(device),create=compiler.createShaderModule.bind(compiler);
  compiler.createShaderModule=desc=>{
   if(reference&&desc.code.includes("// Four coarse rows per workgroup,")){referenceModules++;return create({...desc,code:uniformTransportWorkgroupReference(desc.code)});}
   return create(desc);
  };
  const stages:UniformMixedTransportStage[]=[],outputs:GPUTexture[]=[];
  for(let arm=0;arm<2;arm++){
   reference=arm===0;const scratch=device.createBuffer({size:UniformMixedTransportStage.scratchRanges(64).bytes,usage:GPUBufferUsage.STORAGE});owned.push(scratch);
   const output=texture(d,"r32float");outputs.push(output);
   const stage=new UniformMixedTransportStage(device,layouts[0]!,input,output,departure,{phi,params,reductions,resolved:true});stage.bindScratch(scratch);owned.push(stage);stages.push(stage);await stage.initialize();if(arm===0)restoreUniformTransportWorkgroupDispatch(stage);
  }
  assert.ok(referenceModules>0,"The control must actually compile the original operators");
  for(const [layoutIndex,layout] of layouts.entries())for(const mode of [0,1,2]){
   // Zero edges, sub-cell overlap across a refinement seam, and a long
   // out-of-domain departure exercising the unsampled-row fallback.
   const departures=new Float32Array(cells*4);
   for(let z=0;z<d;z++)for(let y=0;y<d;y++)for(let x=0;x<d;x++){
    const tile=Math.floor(x/4)+4*(Math.floor(y/4)+4*Math.floor(z/4));const width=(layout.tiles[tile]!>>>31)?1:4;
    const offset=mode===0?[0,0,0]:mode===1?[.375,-.625,.125]:[x<8?-19.25:3.5,1.125,-.5];
    const at=4*(x+d*(y+d*z));departures.set([x+.5*width+offset[0]!,y+.5*width+offset[1]!,z+.5*width+offset[2]!,0],at);
   }
   write(departure,departures,4);
   for(const stage of stages){stage.ownership.update(layout);for(const key of Object.keys((stage.ownership as any).work))(stage.ownership as any).work[key]=1;const e=device.createCommandEncoder();stage.encodeTransport(e);device.queue.submit([e.finish()]);}
   const [expected,actual]=await Promise.all(outputs.map(t=>readMixedTexture(device!,t)));
   assert.equal(actual!.length,expected!.length);
   for(let i=0;i<actual!.length;i++)assert.equal(actual![i],expected![i],`layout ${layoutIndex}, departure ${mode}, cell ${i}`);
   assert.ok(actual!.every(Number.isFinite));
  }
  await device.queue.onSubmittedWorkDone();assert.deepEqual(errors,[]);
 }finally{for(const resource of owned.reverse())resource.destroy();device?.destroy();}
});
