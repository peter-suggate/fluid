/** Focused Figure 9 benchmark: browser GPU must be off. All runs use dt=17ms.
 * node --import tsx tools/benchmark-narrow-band.ts baseline
 * Add --uniform for Uniform Geometric with the same refinement criteria.
 * Add --steps=180 for the later splash phase.
 * Per-pass timestamps are diagnostic; wall excludes their separate readback. */
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { GPUPassProfile } from './gpu-pass-profile';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import { uniformVolumeMethod } from '../lib/methods/uniform/uniform-volume-method';
import { uniformNarrowBandMethod } from '../lib/methods/uniform/uniform-narrow-band-method';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
const method=process.argv.includes("--uniform")?uniformVolumeMethod:uniformNarrowBandMethod;
const coarseParticleMode=process.argv.includes("--coarse-particles")?"on":"off";
const steps=Number(process.argv.find(a=>a.startsWith("--steps="))?.slice(8)??48);
assert.ok(Number.isInteger(steps)&&steps>=6&&steps<=2400,"--steps must be an integer from 6 to 2400");
const name=process.argv[2]??'current';
assert.match(name,/^[a-z0-9-]+$/);
const hash=createHash('sha256');
for(const file of ['uniform-narrow-band-flip.ts','uniform-narrow-band-advection.wgsl.ts','uniform-narrow-band-membership.wgsl.ts','uniform-narrow-band-order.ts','uniform-narrow-band-surface.wgsl.ts','uniform-mixed-frame.ts'])hash.update(readFileSync(`lib/methods/uniform/${file}`));
const sourceHash=hash.digest('hex');
const {sharpeningSweeps,sharpeningDistance,...sharedDefaults}=uniformNarrowBandMethod.appDefaults!;
const values={...(method===uniformVolumeMethod?sharedDefaults:uniformNarrowBandMethod.appDefaults),timeStep:'scene',coarseParticleMode,...(coarseParticleMode==='on'?{detailPolicy:'requested',detailSolidContact:'off'}:{})};
const scene=structuredClone(sceneDocument(getSceneDefinition('cm12-figure-9')));
scene.numerics={...scene.numerics,fixedDt_s:0.017,maxDt_s:0.017};
let device:GPUDevice|undefined,profile:GPUPassProfile|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const rows:unknown[]=[];const errors:string[]=[];
try {
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query'],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 profile=new GPUPassProfile(raw);device=managedGPUDevice(profile.device,{requireWorkerRealm:false});
 device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 for(let frame=1;frame<=steps;frame++){
  const measured=frame>steps-6;const start=performance.now();if(measured)profile.start();
  solver.advanceTo(frame*.017,[]);await solver.awaitFrameCompletion();
  assert.equal(solver.info.simulationPipelineError,undefined);
  assert.ok(Math.abs((solver.info.completedTime_s??0)-frame*.017)<1e-8,"complete exactly one requested clock step");
  const wall_ms=performance.now()-start;
  const passes=measured?await profile.finish():undefined;
  const row={frame,time:solver.info.completedTime_s,wall_ms,...solver.narrowBandFlipInfo,fineTiles:solver.info.uniformMixedFineTiles,fineCapacity:solver.info.uniformMixedFineCapacity,bandTiles:solver.info.uniformPressureBandTiles,passes};rows.push(row);
  if(frame%10===0||measured)console.log(JSON.stringify({...row,passes:passes?.filter(p=>p.label.includes('Narrow-band'))}));
 }
 const final=await solver.readStats();assert.deepEqual(errors,[]);
 mkdirSync('docs/verification',{recursive:true});
 writeFileSync(`docs/verification/narrow-band-${name}.json`,JSON.stringify({date:new Date().toISOString(),method:method.id,values,coarseParticleMode,sourceHash,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},scene,rows,final,errors},null,2)+'\n');
}finally{solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
