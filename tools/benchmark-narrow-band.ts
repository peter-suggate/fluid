/** Focused Figure 9 benchmark: browser GPU must be off. All runs use dt=17ms.
 * node --import tsx tools/benchmark-narrow-band.ts baseline
 * Per-pass timestamps are diagnostic; wall excludes their separate readback. */
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { GPUPassProfile } from './gpu-pass-profile';
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from '../lib/harness/webgpu-smoke-isolation';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import { uniformNarrowBandMethod } from '../lib/methods/uniform/uniform-narrow-band-method';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
const name=process.argv[2]??'current';
assert.match(name,/^[a-z0-9-]+$/);
const sourceHash=createHash('sha256').update(readFileSync('lib/methods/uniform/uniform-narrow-band-flip.ts')).digest('hex');
const scene=structuredClone(sceneDocument(getSceneDefinition('cm12-figure-9')));
scene.numerics={...scene.numerics,fixedDt_s:0.017,maxDt_s:0.017};
await acquireWebGPUExclusiveLock('dawn-probe','Figure 9 narrow-band performance');
let device:GPUDevice|undefined,profile:GPUPassProfile|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const rows:unknown[]=[];const errors:string[]=[];
try {
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query'],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 profile=new GPUPassProfile(raw);device=managedGPUDevice(profile.device,{requireWorkerRealm:false});
 device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,'balanced',{...uniformNarrowBandMethod.appDefaults,timeStep:'scene'},undefined,()=>{}) as WebGPUUniformReferenceSolver;
 for(let frame=1;frame<=48;frame++){
  const measured=frame>=43;const start=performance.now();if(measured)profile.start();
  solver.advanceTo(frame*.017,[]);await solver.awaitFrameCompletion();
  assert.equal(solver.info.simulationPipelineError,undefined);
  assert.ok(Math.abs((solver.info.completedTime_s??0)-frame*.017)<1e-8,"complete exactly one requested clock step");
  const wall_ms=performance.now()-start;
  const passes=measured?await profile.finish():undefined;
  const row={frame,time:solver.info.completedTime_s,wall_ms,...solver.narrowBandFlipInfo,passes};rows.push(row);
  if(frame%10===0||measured)console.log(JSON.stringify({...row,passes:passes?.filter(p=>p.label.includes('Narrow-band'))}));
 }
 const final=await solver.readStats();assert.deepEqual(errors,[]);
 mkdirSync('docs/verification',{recursive:true});
 writeFileSync(`docs/verification/narrow-band-${name}.json`,JSON.stringify({date:new Date().toISOString(),sourceHash,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},scene,rows,final,errors},null,2)+'\n');
}finally{solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
