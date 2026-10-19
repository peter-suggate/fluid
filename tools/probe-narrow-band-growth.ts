/** Measure narrow-band FLIP volume growth on the production method defaults.
 * Run: node --import tsx tools/probe-narrow-band-growth.ts baseline
 * --scene=..., --dt=..., --steps=..., --fine-padding=... configure the run.
 * Stage probes and timestamps add diagnostic work; these are not cost measurements.
 */
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { narrowBandVolumeProbe } from './narrow-band-volume-probe';
import { GPUPassProfile } from './gpu-pass-profile';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import { uniformNarrowBandMethod } from '../lib/methods/uniform/uniform-narrow-band-method';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
const method=uniformNarrowBandMethod;
const steps=Number(process.argv.find(a=>a.startsWith("--steps="))?.slice(8)??306);
assert.ok(Number.isInteger(steps)&&steps>=6&&steps<=2400,"--steps must be an integer from 6 to 2400");
const fineGridPadding=Number(process.argv.find(a=>a.startsWith("--fine-padding="))?.split("=")[1]??1);
const dt=Number(process.argv.find(a=>a.startsWith("--dt="))?.split("=")[1]??(1/60));
assert.ok(fineGridPadding>=0&&fineGridPadding<=4&&dt>0&&dt<=0.1);
const sceneId=process.argv.find(a=>a.startsWith("--scene="))?.split("=")[1]??"cm12-figure-2";
const statsEvery=Number(process.argv.find(a=>a.startsWith("--stats-every="))?.split("=")[1]??1);
assert.ok(Number.isInteger(statsEvery)&&statsEvery>=0);
const name=process.argv[2]??'current';
assert.match(name,/^[a-z0-9-]+$/);
const supportedOptions=new Set(['--scene','--dt','--steps','--fine-padding','--stats-every','--no-gravity','--no-volume-stages']);
for(const argument of process.argv.slice(3))assert.ok(supportedOptions.has(argument.split('=')[0]!),`Unsupported option: ${argument}`);
const sourceFiles:Record<string,string>={};
const hash=createHash('sha256');
for(const file of ['uniform-narrow-band-flip.ts','uniform-particle-slab.wgsl.ts','uniform-narrow-band-advection.wgsl.ts','uniform-narrow-band-membership.wgsl.ts','uniform-narrow-band-redistance.wgsl.ts','uniform-narrow-band-surface.wgsl.ts','uniform-mixed-surface-volume.ts','uniform-narrow-band-method.ts','webgpu-uniform-reference.ts','uniform-mixed-frame.ts','uniform-mixed-frame-plan.ts','uniform-mixed-remap.ts','uniform-mixed-surface.ts','uniform-mixed-dynamic.ts','uniform-mixed-pressure-authority.ts']){const bytes=readFileSync(`lib/methods/uniform/${file}`);hash.update(bytes);sourceFiles[file]=createHash('sha256').update(bytes).digest('hex');}
hash.update(readFileSync(import.meta.filename));
const sourceHash=hash.digest('hex');
const values={...method.appDefaults,timeStep:'scene',fineGridPadding};
const scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));
scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
if(process.argv.includes("--no-gravity"))scene.fluid.gravity_m_s2={x:0,y:0,z:0};
let device:GPUDevice|undefined,profile:GPUPassProfile|undefined,solver:WebGPUUniformReferenceSolver|undefined;
let volumeProbe:Awaited<ReturnType<typeof narrowBandVolumeProbe>>|undefined;
const rows:unknown[]=[];const errors:string[]=[];
try {
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query'],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 profile=new GPUPassProfile(raw);
 device=managedGPUDevice(profile.device,{requireWorkerRealm:false});
 device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 if(!process.argv.includes("--no-volume-stages"))volumeProbe=await narrowBandVolumeProbe(device,solver);
 for(let frame=1;frame<=steps;frame++){
  const measured=frame>steps-6;const start=performance.now();if(measured)profile.start();
  solver.advanceTo(frame*dt,[]);await solver.awaitFrameCompletion();
  assert.equal(solver.info.simulationPipelineError,undefined);
  assert.ok(Math.abs((solver.info.completedTime_s??0)-frame*dt)<1e-8,"complete exactly one requested clock step");
  const wall_ms=performance.now()-start;
  const passes=measured?await profile.finish():undefined;
  const stats=statsEvery&&frame%statsEvery===0?await solver.readStats():undefined;
  const volumeStages=await volumeProbe?.read();
  const row={frame,...(volumeStages?{volumeStages}:{}),...(stats?{volumeDrift:stats.volumeDrift,volumeCellSum:stats.volumeCellSum,target:stats.narrowBandTargetVolume_cells,shift:stats.narrowBandVolumeShift_cells,before:stats.narrowBandVolumeBeforeCorrection_cells,outflow:stats.narrowBandOutflowVolume_cells}:{}),time:solver.info.completedTime_s,wall_ms,...solver.narrowBandFlipInfo,fineTiles:solver.info.uniformMixedFineTiles,fineCapacity:solver.info.uniformMixedFineCapacity,bandTiles:solver.info.uniformPressureBandTiles,passes};rows.push(row);
  if(frame%10===0||measured)console.log(JSON.stringify({...row,passes:passes?.filter(p=>p.label.includes('Narrow-band'))}));
 }
 const final=await solver.readStats();assert.deepEqual(errors,[]);
 mkdirSync('docs/verification',{recursive:true});
 writeFileSync(`docs/verification/narrow-band-growth-${name}.json`,JSON.stringify({date:new Date().toISOString(),arguments:process.argv.slice(2),sourceFiles,method:method.id,values,sourceHash,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},scene,rows,final,errors},null,2)+'\n');
}catch(error){
 mkdirSync('docs/verification',{recursive:true});
 writeFileSync(`docs/verification/narrow-band-growth-${name}-failed.json`,JSON.stringify({date:new Date().toISOString(),arguments:process.argv.slice(2),sourceFiles,method:method.id,values,sourceHash,scene,rows,errors,error:String(error)},null,2)+'\n');
 throw error;
}finally{volumeProbe?.destroy();solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
