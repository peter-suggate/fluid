/** Reproduce a production pressure regression with exact compute-pass timestamps.
 * Defaults match the reported Water box URL; timings exclude receipt/CPU gaps.
 */
import assert from "node:assert/strict";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {refinementRegionLattice} from "../lib/core/refinement-regions";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import {GPUPassTimestampRecorder} from "../lib/core/performance-trace";
import type {GPUSolverInstance} from "../lib/core/method-contract";
const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const arm=arg("arm","mixed"),steps=Number(arg("steps","12")),tolerance=Number(arg("tolerance","0.001"));
const sceneId=arg("scene","water-box-dam-break");
const splitDispatches=process.argv.includes("--split-dispatches");
const transportCounts=process.argv.includes("--transport-counts");
const stageTiming=arg("timing","passes")==="stages";
if(stageTiming)usePerformanceInstrumentationStore.getState().setEnabled(true);
assert.ok(["fine","mixed","live","dynamic"].includes(arm));
let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined,recorder:GPUPassTimestampRecorder|undefined;
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??`${process.cwd()}/node_modules/webgpu/index.js`).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits),requiredFeatures:["timestamp-query"]}),{requireWorkerRealm:false});
 const pipelineNames=new WeakMap<GPUComputePipeline,string>();
 const measured=new Proxy(device,{get(target,key){
  if(key==="createComputePipelineAsync")return async(desc:GPUComputePipelineDescriptor)=>{
   const p=await target.createComputePipelineAsync(desc);pipelineNames.set(p,`${desc.compute.entryPoint} ${JSON.stringify(desc.compute.constants??{})}`);return p;
  };
  if(key==="createCommandEncoder")return(desc?:GPUCommandEncoderDescriptor)=>{
   const raw=target.createCommandEncoder(desc),encoder=recorder?.instrument(raw)??raw;
   if(!splitDispatches)return encoder;
   return new Proxy(encoder,{get(e,k){
    if(k==="beginComputePass")return(desc:GPUComputePassDescriptor={})=>{
     if(!/^Uniform mixed (momentum|body forces|pressure sweep|surface advect)$/.test(desc.label??""))return e.beginComputePass(desc);
     let pipeline:GPUComputePipeline;const bindings=new Map<number,[GPUBindGroup,number[]|undefined]>();
     const dispatch=(method:"dispatchWorkgroups"|"dispatchWorkgroupsIndirect",args:unknown[])=>{
      const pass=e.beginComputePass({...desc,label:`${desc.label} | ${pipelineNames.get(pipeline)??pipeline.label}`});
      pass.setPipeline(pipeline);for(const [index,[group,offsets]] of bindings)pass.setBindGroup(index,group,offsets);
      (pass[method] as (...args:unknown[])=>void)(...args);pass.end();
     };
     return {setPipeline:(p:GPUComputePipeline)=>{pipeline=p;},setBindGroup:(i:number,g:GPUBindGroup,o?:number[])=>{bindings.set(i,[g,o]);},
      dispatchWorkgroups:(...args:unknown[])=>dispatch("dispatchWorkgroups",args),dispatchWorkgroupsIndirect:(...args:unknown[])=>dispatch("dispatchWorkgroupsIndirect",args),end:()=>{}} as unknown as GPUComputePassEncoder;
    };
    const value=Reflect.get(e,k,e);return typeof value==="function"?value.bind(e):value;
   }});
  };
  const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
 }});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 const scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));
 const l=refinementRegionLattice(scene),axes=["x","y","z"] as const;
 const regionPercent=arg("region","").split("_").filter(Boolean).map(Number);
 if(regionPercent.length)assert.ok(regionPercent.length===8&&regionPercent.every(Number.isFinite)&&regionPercent.slice(0,6).every(v=>v>=0&&v<=100)&&[0,1,2].every(i=>regionPercent[i]!<regionPercent[i+3]!)&&regionPercent[6]===4&&regionPercent[7]===4,"Region must be minXYZ_maxXYZ_4_4 percentages");
 const region={id:"reported",rule:"minimum-cell-size" as const,minimumCellSize_cells:4 as const,maximumCellSize_cells:4 as const,
  min_m:Object.fromEntries(axes.map((a,i)=>[a,l.origin_m[a]+(regionPercent.length?regionPercent[i]!/100:i===0?2/3:0)*l.dimensions[i]!*l.cellSize_m[i]!])) as {x:number;y:number;z:number},
  max_m:Object.fromEntries(axes.map((a,i)=>[a,l.origin_m[a]+(regionPercent.length?regionPercent[i+3]!/100:i===0?1:.5)*l.dimensions[i]!*l.cellSize_m[i]!])) as {x:number;y:number;z:number}};
 scene.fluid.refinementRegions=arm==="mixed"?[region]:[];
 // Retired layout: an h background with a 4h box. Requested is 4h outside Fine boxes, so the
 // mixed/live arms now run all-4h (plus solid contact) and the box is redundant.
 const values={pressureResidualTolerance:tolerance,detailPolicy:arm==="dynamic"?"dynamic":arm==="mixed"||arm==="live"?"requested":"full"};
 solver=await uniformVolumeMethod.createSolverAsync!(measured,scene,"balanced",values,undefined,()=>{});
 console.log(JSON.stringify({arm,sceneId,tolerance,dimensions:l.dimensions,region:scene.fluid.refinementRegions,bytes:solver.info.allocatedBytes}));
 for(let step=1;step<=steps;step++){
  if(arm==="live"&&step===4){scene.fluid.refinementRegions=[region];solver.applySceneUniforms?.(scene);await solver.pipelinesPrepared?.();await solver.awaitFrameCompletion?.();}
  recorder=stageTiming?undefined:new GPUPassTimestampRecorder(device,4096,`Water box ${step}`);
  const start=performance.now();assert.ok(solver.advanceTo(step/30,[]));let failure:unknown;try{await solver.awaitFrameCompletion?.();}catch(error){failure=String(error);}const wall_ms=performance.now()-start;
  const captured=recorder;recorder=undefined;let reading;
  if(captured){const e=device.createCommandEncoder();captured.resolve(e);device.queue.submit([e.finish()]);reading=await captured.read();}
  if(!failure){await solver.readStats();}
  // Diagnostic readback is outside the timed frame and pass recorder.
  let transportLive:readonly number[]|undefined;
  if(transportCounts){
   const source=(solver as unknown as {mixedFrame?:{transport:{live:GPUBuffer}}}).mixedFrame?.transport.live;
   assert.ok(source,"Transport counts require the mixed frame");
   const read=device.createBuffer({size:24,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   try{const e=device.createCommandEncoder();e.copyBufferToBuffer(source,18*4,read,0,24);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);transportLive=Array.from(new Uint32Array(read.getMappedRange()));}
   finally{if(read.mapState==="mapped")read.unmap();read.destroy();}
  }
  const groups=new Map<string,{ms:number;passes:number;unsampled:number}>();
  for(const p of reading?.passes??[]){const row=groups.get(p.label)??{ms:0,passes:0,unsampled:0};row.ms+=p.duration_ms;row.passes++;row.unsampled+=+!p.sampled;groups.set(p.label,row);}
  console.log(JSON.stringify({arm,step,wall_ms,cycles:solver.info.uniformPressureCyclesExecuted,residual:solver.info.uniformPressureAcceptedResidual,error:failure??solver.info.simulationPipelineError,
   fineTiles:solver.info.uniformMixedFineTiles,coarseTiles:solver.info.uniformMixedCoarseTiles,regularTiles:solver.info.uniformMixedRegularTiles,generalTiles:solver.info.uniformMixedGeneralTiles,
   relayouts:solver.info.uniformMixedDynamicRelayouts,volumeDrift:solver.info.volumeDrift,
   transportLive,passSum_ms:reading?.sum_ms,span_ms:reading?.span_ms,...(stageTiming?{trace:solver.info.physicsTrace,timingUnavailable:solver.info.physicsTraceUnavailable}:{}),groups:[...groups].sort((a,b)=>b[1].ms-a[1].ms)}));
  if(failure||solver.info.simulationPipelineError)throw new Error(String(failure??solver.info.simulationPipelineError));
 }
 assert.deepEqual(errors,[]);
}finally{recorder?.destroy();solver?.destroy();device?.destroy();}
