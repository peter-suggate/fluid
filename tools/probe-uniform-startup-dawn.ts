/** Driver-level compilation timings, excluding manager queue waits.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-uniform-startup-dawn.ts
 */
import {pathToFileURL} from 'node:url';
import {writeFile} from 'node:fs/promises';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from '../lib/harness/webgpu-smoke-isolation';
import {managedGPUDevice} from '../lib/core/gpu-compilation-manager';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {sceneDocument} from '../lib/core/scene-definition';
import {getSceneDefinition} from '../lib/core/scenes';
import {uniformGeometricSolverOptions} from '../lib/methods/uniform/uniform-geometric-options';
import {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';

await acquireWebGPUExclusiveLock('dawn-test','Uniform startup compilation profile');
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const jobs:{entry:string;label:string;ms:number}[]=[],errors:string[]=[];
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]);
 const adapter=await gpu.requestAdapter();if(!adapter)throw new Error('No adapter');
 const raw=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 raw.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 const timed=new Proxy(raw,{get(target,key){
  if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{
   const start=performance.now();const entry=d.compute.entryPoint??'',label=d.label??'';
   console.log(JSON.stringify({start:entry,label}));
   const pipeline=await target.createComputePipelineAsync(d);
   const job={entry,label,ms:performance.now()-start};jobs.push(job);console.log(JSON.stringify(job));return pipeline;
  };
  const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
 }});
 device=managedGPUDevice(timed,{requireWorkerRealm:false});
 const scene=sceneDocument(getSceneDefinition(process.env.SCENE??'water-box-dam-break'));
 const start=performance.now();
 solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,'balanced',undefined,uniformGeometricSolverOptions({},scene),()=>{});
 const initializationMs=performance.now()-start;
 solver.advanceTo(1/30);await solver.awaitFrameCompletion();await solver.readStats();
 const report={scene:process.env.SCENE??'water-box-dam-break',initializationMs,firstFrameMs:performance.now()-start-initializationMs,errors,jobs:jobs.sort((a,b)=>b.ms-a.ms)};
 await writeFile(process.env.REPORT??'/tmp/uniform-startup-profile.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
 if(errors.length)throw new Error('GPU validation errors');
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
