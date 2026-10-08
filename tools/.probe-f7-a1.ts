// fluid-a1 scratch probe (delete when the mixed frame program closes).
// Long-dam dynamic frame time: untimed wall median plus a separate timed pass
// for phase means. Mimics the renderer: applyRuntimeValues(same values) every frame.
// Usage: FLUID_GPU_COMPILATION_CONCURRENCY=1 WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
//   node --import tsx tools/.probe-mixed-frame-a1.ts <out.json> [steps=60] [from=36]
import {writeFileSync} from "node:fs";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {GPUStageTimestampRecorder,GPUPassTimestampRecorder} from "../lib/core/performance-trace";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {resolveMethodValues} from "../lib/core/method-contract";
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";

const [out,stepsArg,fromArg]=process.argv.slice(2);
const steps=Number(stepsArg??60),from=Number(fromArg??36),sceneId=process.env.PROBE_SCENE??"sparse-cm12-long-dam-break";
let device:GPUDevice|undefined;const solvers:{destroy():void}[]=[];
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();
 device=managedGPUDevice(await adapter!.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter!.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];
 device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error("GPU error:",e.error.message);});
 await GPUStageTimestampRecorder.prepare(device);
 let recorder:GPUPassTimestampRecorder|undefined;const real=device;const names=new WeakMap<GPUComputePipeline,string>();
 const measured=new Proxy(device,{get(target,key){
  if(key==="createComputePipelineAsync")return async(desc:GPUComputePipelineDescriptor)=>{const p=await target.createComputePipelineAsync(desc);names.set(p,desc.compute.entryPoint??"?");return p;};
  if(key==="createComputePipeline")return(desc:GPUComputePipelineDescriptor)=>{const p=target.createComputePipeline(desc);names.set(p,desc.compute.entryPoint??"?");return p;};
  if(key==="createCommandEncoder")return(desc?:GPUCommandEncoderDescriptor)=>{const raw=target.createCommandEncoder(desc);const encoder=recorder?.instrument(raw)??raw;
   if(!recorder||process.env.PROBE_SPLIT!=="1")return encoder;
   return new Proxy(encoder,{get(e,k){
    if(k==="beginComputePass")return(pd:GPUComputePassDescriptor={})=>{
     let pipeline:GPUComputePipeline;const bindings=new Map<number,[GPUBindGroup,any]>();
     const dispatch=(method:string,args:unknown[])=>{const pass=e.beginComputePass({...pd,label:`${pd.label} | ${names.get(pipeline)??"?"}`});
      pass.setPipeline(pipeline);for(const [i,[g,o]] of bindings)pass.setBindGroup(i,g,o);(pass as any)[method](...args);pass.end();};
     return {setPipeline:(q:GPUComputePipeline)=>{pipeline=q;},setBindGroup:(i:number,g:GPUBindGroup,o?:any)=>{bindings.set(i,[g,o]);},
      dispatchWorkgroups:(...a:unknown[])=>dispatch("dispatchWorkgroups",a),dispatchWorkgroupsIndirect:(...a:unknown[])=>dispatch("dispatchWorkgroupsIndirect",a),end:()=>{},pushDebugGroup:()=>{},popDebugGroup:()=>{},insertDebugMarker:()=>{}} as unknown as GPUComputePassEncoder;};
    const v=(e as any)[k];return typeof v==="function"?v.bind(e):v;}});};
  const v=(target as any)[key];return typeof v==="function"?v.bind(target):v;}}) as GPUDevice;
 const scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));
 const values=resolveMethodValues(uniformVolumeMethod,"balanced",{...JSON.parse(process.env.PROBE_OVERRIDES??"{}")});
 const result:Record<string,unknown>={sceneId,steps,from,errors};
 for(const [rep,timed] of (process.env.PROBE_SPLIT==="1"?[[1,true]]:[[0,false],[1,true]]) as [number,boolean][]){
  const store=usePerformanceInstrumentationStore.getState();
  store.setEnabled(false);
  const solver=await uniformVolumeMethod.createSolverAsync!(measured,scene,"balanced",values,undefined,()=>{}) as any;solvers.push(solver);
  let reused=0,relayouts=0;{const f=solver.mixedFrame;const adv=f.advance.bind(f);f.advance=(...a:any[])=>{if(f.reusableExtension!==undefined&&f.reusableExtension===JSON.stringify({...a[0],dt:0}))reused++;return adv(...a);};const ad=f.adoptBuiltLayout.bind(f);f.adoptBuiltLayout=(l:any)=>{if(l.changedTiles)relayouts++;return ad(l);};}
  const walls:number[]=[],phases:Record<string,number[]>={},totals:number[]=[],fine:number[]=[],cycles:number[]=[];
  for(let step=1;step<=steps;step++){
   solver.applyRuntimeValues(values);
   if(timed&&step>=from)recorder=new GPUPassTimestampRecorder(real,4096,`frame ${step}`);
   const t0=performance.now();
   if(!solver.advanceTo(step/30,[]))throw new Error("advance refused");await solver.awaitFrameCompletion();
   const wall=performance.now()-t0;
   if(solver.info.simulationPipelineError)throw new Error(String(solver.info.simulationPipelineError));
   let reading:any;if(recorder){const r=recorder;recorder=undefined;const e=real.createCommandEncoder();r.resolve(e);real.queue.submit([e.finish()]);reading=await r.read();}
   if(step<from)continue;
   walls.push(wall);
   const tiles=solver.mixedFrame?.ownership?.layout?.tiles as Uint32Array|undefined;
   if(tiles){let n=0;for(const w of tiles)if(w&0x80000000)n++;fine.push(n);}
   if(reading){totals.push(reading.sum_ms);const g:Record<string,number>={};for(const q of reading.passes)g[q.label]=(g[q.label]??0)+q.duration_ms;for(const [k,v] of Object.entries(g))(phases[k]??=[]).push(v);}
  }
  const med=(a:number[])=>{const s=[...a].sort((x,y)=>x-y);return s.length?+s[s.length>>1]!.toFixed(2):null;};
  const mean=(a:number[])=>a.length?+(a.reduce((x,y)=>x+y,0)/a.length).toFixed(3):null;
  const arm={reused,relayouts,wallMedian_ms:med(walls),wallMean_ms:mean(walls),fineTilesMean:mean(fine),
   ...(timed?{traceTotalMean_ms:mean(totals),traced:totals.length,phases:Object.fromEntries(Object.entries(phases).map(([k,v])=>[k,+(v.reduce((x,y)=>x+y,0)/walls.length).toFixed(3)]).sort((a,b)=>(b[1] as number)-(a[1] as number)))}:{})};
  result[`rep${rep}`]=arm;console.log(JSON.stringify({rep,...arm}));
  solver.destroy();solvers.pop();
 }
 writeFileSync(out,JSON.stringify(result,null,1));
}finally{for(const s of solvers)s.destroy();device?.destroy();}
