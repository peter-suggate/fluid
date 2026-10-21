/** Run an authored scene to completion with the production solver and GPU features.
 * No resolution, particle, timestep or page-pool capacity overrides.
 * node --import tsx tools/check-nbflip-scene.ts [--scene=high-resolution-dam-break]
 *   [--adaptive-budget=100] [--out=artifacts/validation/high-resolution-dam-break.json]
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { UniformNarrowBandFlip } from "../lib/methods/uniform/uniform-narrow-band-flip";

const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
for(const argument of process.argv.slice(2))assert.match(argument,/^--(?:scene|adaptive-budget|out)=.+$/);
const definition=getSceneDefinition(option("scene")??"high-resolution-dam-break");
const scene=structuredClone(sceneDocument(definition)),quality=definition.methodProfile?.quality??"balanced";
const budget=option("adaptive-budget");
if(budget!==undefined)assert.ok(Number(budget)>=0&&Number(budget)<=100);
const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile?.overrides,...(budget===undefined?{}:{adaptiveBudgetPercent:Number(budget)})});
const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s,steps=Math.round(scene.duration_s/dt);
assert.ok(Math.abs(steps*dt-scene.duration_s)<1e-8,"authored duration must be a whole number of solver steps");
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
const features:GPUFeatureName[]=["float32-filterable","subgroups"].filter(f=>adapter.features.has(f as GPUFeatureName)) as GPUFeatureName[];
const device=managedGPUDevice(await adapter.requestDevice({requiredFeatures:features,requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
const errors:string[]=[],rows:Record<string,number>[]=[];
device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
let solver:WebGPUUniformReferenceSolver|undefined,read:GPUBuffer|undefined;
const output=option("out");
try{
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const particles=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip,capacity=particles.capacity;
 const metadata={scene:definition.id,quality,values,dt,steps,duration_s:scene.duration_s,grid:[solver.info.nx,solver.info.ny,solver.info.nz],capacity,allocatedBytes:solver.info.allocatedBytes,features};
 console.log(JSON.stringify({phase:"constructed",...metadata}));
 read=device.createBuffer({size:128,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 let peakParticles=0,peakMoveSpill=0,peakSeedSpill=0,wallTotal_ms=0;
 for(let frame=1;frame<=steps;frame++){
  const start=performance.now();assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();const wall_ms=performance.now()-start;wallTotal_ms+=wall_ms;
  assert.equal(solver.info.simulationPipelineError,undefined);assert.equal(particles.capacity,capacity);
  assert.ok(Math.abs((solver.info.completedTime_s??0)-frame*dt)<1e-8);
  const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(particles.state,0,read,0,128);device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);const words=new Uint32Array(read.getMappedRange()).slice();read.unmap();assert.equal(words[3],0);
  peakParticles=Math.max(peakParticles,words[1]!);peakMoveSpill=Math.max(peakMoveSpill,words[16]!);peakSeedSpill=Math.max(peakSeedSpill,words[21]!);
  if(frame%60===0||frame===steps){const row={frame,time_s:frame*dt,wall_ms,particles:words[1]!,extent:words[0]!,moveSpill:words[16]!,seedSpill:words[21]!};rows.push(row);console.log(JSON.stringify(row));}
 }
 assert.deepEqual(errors,[]);
 const result={...metadata,completedTime_s:solver.info.completedTime_s,peakParticles,peakMoveSpill,peakSeedSpill,meanFrame_ms:wallTotal_ms/steps,rows,errors};
 if(output){mkdirSync(dirname(output),{recursive:true});writeFileSync(output,JSON.stringify(result,null,2)+"\n");}
 console.log(JSON.stringify({phase:"complete",...result,rows:undefined}));
}finally{read?.destroy();solver?.destroy();await device.queue.onSubmittedWorkDone();device.destroy();}
