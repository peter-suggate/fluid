import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { createMinimalPowerDamBreak64Scene } from '../lib/core/scenes';
import { resolveMethodValues } from '../lib/core/method-contract';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { adaptiveMassMethod } from '../lib/methods/adaptive-mass/method';
import type { WebGPUAdaptiveMassSolver } from '../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver';
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from '../lib/harness/webgpu-smoke-isolation';
const modulePath=process.env.WEBGPU_NODE_MODULE;assert.ok(modulePath);
const output=process.env.MINI64_DEFICIT_OUTPUT??'artifacts/mini64-deficit/probe';
const steps=Number(process.env.MINI64_DEFICIT_STEPS??8);
const uiCadence=process.env.MINI64_DEFICIT_UI_CADENCE==='1';
await acquireWebGPUExclusiveLock('dawn-probe','mini64-deficit');
const live=new Set<GPU>();Object.assign(globalThis,{mini64DeficitGPU:live});
let device:GPUDevice|undefined,solver:WebGPUAdaptiveMassSolver|undefined;
try{
 const dawn=await import(pathToFileURL(modulePath).href);Object.assign(globalThis,dawn.globals);
 const gpu:GPU=dawn.create(['backend=metal']);live.add(gpu);
 const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
 const configuration=process.env.MINI64_DEFICIT_RECEIPT?JSON.parse(await readFile(process.env.MINI64_DEFICIT_RECEIPT,'utf8')).configuration:undefined;
 const scene=configuration?.scene??createMinimalPowerDamBreak64Scene();
 if(!configuration)scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/30;
 const quality=configuration?.method.quality??'balanced';
 const values=resolveMethodValues(adaptiveMassMethod,quality,configuration?.method.overrides?.['adaptive-mass']??{});
 await mkdir(output,{recursive:true});await writeFile(`${output}/config.json`,JSON.stringify({scene,values},null,2));
 console.log(JSON.stringify({stage:'constructing'}));
 solver=await adaptiveMassMethod.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUAdaptiveMassSolver;
 await solver.waitForSimulationReady();
 if(process.env.MINI64_DEFICIT_FREEZE==='1')solver.setTopologyFrozen(true);
 const trace:unknown[]=[];
 for(let step=0;step<=steps;step++){
  if(step){while(!solver.advanceTo(step*scene.numerics.fixedDt_s,[]))await new Promise(setImmediate);if(!uiCadence)await solver.waitForTopologyReady();}
  const failure:unknown=await solver.assertSimulationHealthy().then(()=>null,error=>error.failure??String(error));
  const source=solver.fieldSnapshotSourceForQA;
  const activity=uiCadence&&!failure&&step<steps?undefined:await solver.readGPUActivityPolicy();
  const fields=failure||uiCadence&&step<steps?undefined:await solver.readDiagnosticFields(true);
  const row={step,failure,cells:source.cellCapacity,generation:activity?.acceptedTopologyGeneration,active:activity?.bricks.filter(b=>b.active).length,mass:fields?.density.reduce((a,b)=>a+b,0),maxDensity:fields?.density.reduce((a,b)=>Math.max(a,b),0)};
  trace.push(row);console.log(JSON.stringify(row));
  await writeFile(`${output}/trace.json`,JSON.stringify(trace,null,2));
  if(activity)await writeFile(`${output}/${step}-activity.json`,JSON.stringify(activity));
  if((!uiCadence||failure||step===steps)&&step>=Number(process.env.MINI64_DEFICIT_CAPTURE_FROM??4)){
   await writeFile(`${output}/${step}-template.bin`,new Uint8Array(source.templateWords.buffer,source.templateWords.byteOffset,source.templateWords.byteLength));
   await writeFile(`${output}/${step}-layout.json`,JSON.stringify(source.layout));
   await writeFile(`${output}/${step}-native-metadata.json`,JSON.stringify({topologyWorklistBaseWords:source.topologyWorklistBaseWords,cellCapacity:source.cellCapacity}));
   for(const [name,buffer] of Object.entries({conditioning:source.conditioning,topology:source.topologyArena,indirect:source.acceptedIndirectArguments})){
    const copy=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const command=device.createCommandEncoder();command.copyBufferToBuffer(buffer,0,copy,0,buffer.size);device.queue.submit([command.finish()]);
    await copy.mapAsync(GPUMapMode.READ);await writeFile(`${output}/${step}-${name}.bin`,new Uint8Array(copy.getMappedRange()));copy.unmap();copy.destroy();
   }

   const readback=device.createBuffer({size:source.state.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(source.state,0,readback,0,source.state.size);device.queue.submit([encoder.finish()]);
   await readback.mapAsync(GPUMapMode.READ);await writeFile(`${output}/${step}-state.bin`,new Uint8Array(readback.getMappedRange()));readback.unmap();readback.destroy();
   const activityBuffer=solver.sparseAdaptiveGridSource!.activity!.buffer;
   const raw=device.createBuffer({size:activityBuffer.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   const rawEncoder=device.createCommandEncoder();rawEncoder.copyBufferToBuffer(activityBuffer,0,raw,0,activityBuffer.size);device.queue.submit([rawEncoder.finish()]);
   await raw.mapAsync(GPUMapMode.READ);await writeFile(`${output}/${step}-activity.bin`,new Uint8Array(raw.getMappedRange()));raw.unmap();raw.destroy();
  }
  assert.equal(failure,null,`simulation failure at step ${step}`);
 }
 assert.deepEqual(errors,[]);
}finally{solver?.destroy();device?.destroy();live.clear();await releaseWebGPUExclusiveLock();}
