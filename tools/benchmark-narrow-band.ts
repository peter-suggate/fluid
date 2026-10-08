/** Focused Figure 9 benchmark: browser GPU must be off. Default dt is 17ms.
 * node --import tsx tools/benchmark-narrow-band.ts baseline
 * Add --uniform for Uniform Geometric with the same refinement criteria.
 * Add --steps=180 for the later splash phase.
 * --fine-padding=0|1|2 tests grid padding with the fixed 4h particle band.
 * --dt=0.05 tests larger global steps without changing trajectory subdivision.
 * Per-pass timestamps are diagnostic; wall excludes their separate readback. */
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
import { uniformVolumeMethod } from '../lib/methods/uniform/uniform-volume-method';
import { uniformNarrowBandMethod } from '../lib/methods/uniform/uniform-narrow-band-method';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
const method=process.argv.includes("--uniform")?uniformVolumeMethod:uniformNarrowBandMethod;
const coarseParticleMode=process.argv.includes("--coarse-particles")?"on":"off";
const steps=Number(process.argv.find(a=>a.startsWith("--steps="))?.slice(8)??48);
assert.ok(Number.isInteger(steps)&&steps>=6&&steps<=2400,"--steps must be an integer from 6 to 2400");
const fineGridPadding=Number(process.argv.find(a=>a.startsWith("--fine-padding="))?.split("=")[1]??1);
const dt=Number(process.argv.find(a=>a.startsWith("--dt="))?.split("=")[1]??0.017);
assert.ok(fineGridPadding>=0&&fineGridPadding<=4&&dt>0&&dt<=0.1);
const sceneId=process.argv.find(a=>a.startsWith("--scene="))?.split("=")[1]??"cm12-figure-9";
const statsEvery=Number(process.argv.find(a=>a.startsWith("--stats-every="))?.split("=")[1]??0);
assert.ok(Number.isInteger(statsEvery)&&statsEvery>=0);
const name=process.argv[2]??'current';
assert.match(name,/^[a-z0-9-]+$/);
const hash=createHash('sha256');
for(const file of ['uniform-narrow-band-flip.ts','uniform-narrow-band-advection.wgsl.ts','uniform-narrow-band-membership.wgsl.ts','uniform-narrow-band-redistance.wgsl.ts','uniform-narrow-band-order.ts','uniform-narrow-band-surface.wgsl.ts','uniform-narrow-band-activity.wgsl.ts','uniform-narrow-band-spray.ts','uniform-mixed-surface-volume.ts','uniform-narrow-band-method.ts','webgpu-uniform-reference.ts','uniform-mixed-frame.ts','uniform-mixed-frame-plan.ts','uniform-mixed-remap.ts','uniform-mixed-surface.ts','uniform-mixed-dynamic.ts','uniform-mixed-pressure-authority.ts'])hash.update(readFileSync(`lib/methods/uniform/${file}`));
const sourceHash=hash.digest('hex');
const {sharpeningSweeps,sharpeningDistance,...sharedDefaults}=uniformNarrowBandMethod.appDefaults!;
const values={...(method===uniformVolumeMethod?sharedDefaults:uniformNarrowBandMethod.appDefaults),timeStep:'scene',fineGridPadding,coarseParticleMode,...(process.argv.includes("--full")?{detailPolicy:"full"}:{}),...(coarseParticleMode==='on'?{detailPolicy:'requested',detailSolidContact:'off'}:{})};
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
 profile=new GPUPassProfile(raw);device=managedGPUDevice(profile.device,{requireWorkerRealm:false});
 device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 if(process.argv.includes("--no-volume-correction")){
  const frame=(solver as unknown as {mixedFrame:{surfaceVolume:{beginStep(...args:unknown[]):void;encode(...args:unknown[]):void};narrowBandFlip:{refreshBand(...args:unknown[]):void}}}).mixedFrame;
  frame.surfaceVolume.beginStep=()=>{};frame.surfaceVolume.encode=()=>{};frame.narrowBandFlip.refreshBand=()=>{};
 }
 if(process.argv.includes("--no-reseed")){
  const stage=(solver as unknown as {mixedFrame:{narrowBandFlip:{dispatch(encoder:GPUCommandEncoder,entry:string,group?:string):void}}}).mixedFrame.narrowBandFlip;
  const dispatch=stage.dispatch.bind(stage);stage.dispatch=(encoder,entry,group)=>{if(entry!=="seed"||group!=="update")dispatch(encoder,entry,group);};
 }
 if(process.argv.includes("--volume-stages"))volumeProbe=await narrowBandVolumeProbe(device,solver);
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
 writeFileSync(`docs/verification/narrow-band-${name}.json`,JSON.stringify({date:new Date().toISOString(),arguments:process.argv.slice(2),method:method.id,values,coarseParticleMode,sourceHash,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},scene,rows,final,errors},null,2)+'\n');
}catch(error){
 mkdirSync('docs/verification',{recursive:true});
 writeFileSync(`docs/verification/narrow-band-${name}-failed.json`,JSON.stringify({date:new Date().toISOString(),arguments:process.argv.slice(2),method:method.id,values,sourceHash,scene,rows,errors,error:String(error)},null,2)+'\n');
 throw error;
}finally{volumeProbe?.destroy();solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
