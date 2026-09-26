/** Reproduce a production pressure regression with exact compute-pass timestamps.
 * Defaults match the reported Water box URL; timings exclude receipt/CPU gaps.
 */
import assert from "node:assert/strict";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {refinementRegionLattice} from "../lib/core/refinement-regions";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import {GPUPassTimestampRecorder} from "../lib/core/performance-trace";
import type {GPUSolverInstance} from "../lib/core/method-contract";
const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const arm=arg("arm","mixed"),steps=Number(arg("steps","12")),tolerance=Number(arg("tolerance","0.001"));
const sceneId=arg("scene","water-box-dam-break");
const stageTiming=arg("timing","passes")==="stages";
if(stageTiming)usePerformanceInstrumentationStore.getState().setEnabled(true);
assert.ok(["native","fine","mixed","live"].includes(arm));
await acquireWebGPUExclusiveLock("dawn-benchmark",`Water box pressure ${arm}`);
let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined,recorder:GPUPassTimestampRecorder|undefined;
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??`${process.cwd()}/node_modules/webgpu/index.js`).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits),requiredFeatures:["timestamp-query"]}),{requireWorkerRealm:false});
 const measured=new Proxy(device,{get(target,key){if(key==="createCommandEncoder")return(desc?:GPUCommandEncoderDescriptor)=>{const e=target.createCommandEncoder(desc);return recorder?.instrument(e)??e;};const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;}});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 const scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));
 const l=refinementRegionLattice(scene),axes=["x","y","z"] as const;
 const region={id:"reported",rule:"minimum-cell-size" as const,minimumCellSize_cells:4 as const,maximumCellSize_cells:4 as const,
  min_m:Object.fromEntries(axes.map((a,i)=>[a,l.origin_m[a]+(i===0?2/3:0)*l.dimensions[i]!*l.cellSize_m[i]!])) as {x:number;y:number;z:number},
  max_m:Object.fromEntries(axes.map((a,i)=>[a,l.origin_m[a]+(i===0?1:.5)*l.dimensions[i]!*l.cellSize_m[i]!])) as {x:number;y:number;z:number}};
 scene.fluid.refinementRegions=arm==="mixed"?[region]:[];
 const values={pressureResidualTolerance:tolerance};
 solver=arm==="native"?await WebGPUUniformReferenceSolver.createAsync(measured,scene,"balanced",undefined,uniformGeometricSolverOptions(values,scene),()=>{}):await uniformVolumeMethod.createSolverAsync!(measured,scene,"balanced",values,undefined,()=>{});
 console.log(JSON.stringify({arm,sceneId,tolerance,dimensions:l.dimensions,region:scene.fluid.refinementRegions,bytes:solver.info.allocatedBytes}));
 for(let step=1;step<=steps;step++){
  if(arm==="live"&&step===4){scene.fluid.refinementRegions=[region];solver.applySceneUniforms?.(scene);await solver.awaitFrameCompletion?.();}
  recorder=stageTiming?undefined:new GPUPassTimestampRecorder(device,4096,`Water box ${step}`);
  const start=performance.now();assert.ok(solver.advanceTo(step/30,[]));let failure:unknown;try{await solver.awaitFrameCompletion?.();}catch(error){failure=String(error);}const wall_ms=performance.now()-start;
  const captured=recorder;recorder=undefined;let reading;
  if(captured){const e=device.createCommandEncoder();captured.resolve(e);device.queue.submit([e.finish()]);reading=await captured.read();}
  else if(!failure){await solver.readStats();}
  const groups=new Map<string,{ms:number;passes:number;unsampled:number}>();
  for(const p of reading?.passes??[]){const row=groups.get(p.label)??{ms:0,passes:0,unsampled:0};row.ms+=p.duration_ms;row.passes++;row.unsampled+=+!p.sampled;groups.set(p.label,row);}
  console.log(JSON.stringify({arm,step,wall_ms,cycles:solver.info.uniformPressureCyclesExecuted,residual:solver.info.uniformPressureAcceptedResidual,error:failure??solver.info.simulationPipelineError,
   passSum_ms:reading?.sum_ms,span_ms:reading?.span_ms,...(stageTiming?{trace:solver.info.physicsTrace,timingUnavailable:solver.info.physicsTraceUnavailable}:{}),groups:[...groups].sort((a,b)=>b[1].ms-a[1].ms)}));
  if(failure||solver.info.simulationPipelineError)break;
 }
 assert.deepEqual(errors,[]);
}finally{recorder?.destroy();solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
