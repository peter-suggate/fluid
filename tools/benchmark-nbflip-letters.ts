/** Ferstl et al. 2016, Figure 8 "Letters" on Uniform Narrow-band FLIP, measured
 * the way the paper's Table 1 reports it: time per step split into the
 * projection and the rest, average particle count and memory. Browser GPU must
 * be off.
 *   node --import tsx tools/benchmark-nbflip-letters.ts [name]
 * --steps=N runs fewer than the take's 193 frames (a smoke run).
 * --track reads the level set after every step and records the top of each
 *   letter as it falls. It is a readback per step, so a tracked run's wall
 *   times are not the benchmark's; pass timestamps are unaffected.
 * --out=DIR writes somewhere other than docs/verification.
 * --set=key=value overrides one method value over the scene's profile, whose
 *   coverage is the method's default (dynamic, surface tiles at h);
 *   --set=detailPolicy=full is the published grid everywhere. Repeatable.
 * --from=N averages the summary over frames N.. only (rows keep every frame).
 * --plain takes no pass timestamps: wall is then the frame time the app pays,
 *   without the per-pass timestamp writes and the encoder proxy that reads them.
 * Every compute pass is timestamped on every step. A pass is projection when
 * its label names the pressure solve; the per-label totals are in the output
 * so that split can be audited. Wall is advance plus completion of one step
 * and excludes the timestamp readback; encode is the host's share of it, the
 * synchronous advance that encodes and submits the step. Between is the GPU
 * time outside every pass, the copies and clears encoded between two of them;
 * a row's `before` attributes it to the pass that follows. */
import { writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { GPUPassProfile } from './gpu-pass-profile';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { resolveMethodValues } from '../lib/core/method-contract';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import type { LiquidExtrusion } from '../lib/core/liquid-extrusion';
import { NBFLIP_LETTERS_FRAMES, NBFLIP_LETTERS_GRID, NBFLIP_LETTERS_SCENE_ID, NBFLIP_LETTERS_TIME_STEP_S } from '../lib/core/nbflip-paper-scenes';
import { uniformNarrowBandMethod } from '../lib/methods/uniform/uniform-narrow-band-method';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
import { readUniformFields } from '../tests/helpers/uniform-geometric';

const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const steps=Number(option("steps")??NBFLIP_LETTERS_FRAMES),track=process.argv.includes("--track"),plain=process.argv.includes("--plain"),out=option("out")??'docs/verification';
assert.ok(Number.isInteger(steps)&&steps>=1&&steps<=NBFLIP_LETTERS_FRAMES,`--steps must be an integer from 1 to ${NBFLIP_LETTERS_FRAMES}`);
const name=process.argv.slice(2).find(a=>!a.startsWith("--"))??'current';
assert.match(name,/^[a-z0-9-]+$/);
const PROJECTION=/pressure|projection/i;
const definition=getSceneDefinition(NBFLIP_LETTERS_SCENE_ID),scene=structuredClone(sceneDocument(definition)),dt=NBFLIP_LETTERS_TIME_STEP_S;
assert.equal(definition.methodProfile?.methodId,uniformNarrowBandMethod.id);
// The values the app resolves for this scene: the method's app defaults under the scene's own overrides.
const sets=Object.fromEntries(process.argv.filter(a=>a.startsWith("--set=")).map(a=>{const [key,...value]=a.slice(6).split("=");const text=value.join("=");return [key!,text!==""&&Number.isFinite(Number(text))?Number(text):text==="true"?true:text==="false"?false:text];}));
const from=Number(option("from")??1);
const values=resolveMethodValues(uniformNarrowBandMethod,definition.methodProfile!.quality,{...uniformNarrowBandMethod.appDefaults,...definition.methodProfile!.overrides,...sets});
const [nx,ny,nz]=NBFLIP_LETTERS_GRID,h=scene.voxelDomain.finestCellSize_m;
// Each letter's highest outline point, as the vertex column under it and the height its liquid starts at.
const letters=scene.fluid.scheduledDrops!.map((drop,index)=>{
 const e=drop.volume as LiquidExtrusion,points=e.contours_m.flatMap(c=>c.flatMap((v,i)=>i%2?[]:[[v,c[i+1]!] as const]));
 const top=points.reduce((a,b)=>b[1]>a[1]?b:a);
 return {letter:"ABCDEFGHI"[index]!,step:Math.round(drop.time_s/dt),x:Math.round(top[0]/h+nx/2),z:Math.round(e.centerZ_m/h+nz/2),top0:(top[1]+e.offset_m)/h};
});
let device:GPUDevice|undefined,profile:GPUPassProfile|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const rows:Record<string,unknown>[]=[],errors:string[]=[],labels=new Map<string,{ms:number;passes:number;dispatches:number}>();
const write=(file:string,body:object)=>{mkdirSync(out,{recursive:true});writeFileSync(`${out}/${file}`,JSON.stringify(body,null,2)+'\n');};
const mean=(values:number[])=>values.reduce((a,b)=>a+b,0)/values.length;
try {
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query'],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 profile=new GPUPassProfile(raw);device=managedGPUDevice(plain?raw:profile.device,{requireWorkerRealm:false});
 device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 const built=performance.now();
 solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,definition.methodProfile!.quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const construction_ms=performance.now()-built;
 assert.deepEqual([solver.info.nx,solver.info.ny,solver.info.nz],[nx,ny,nz],"the published grid");
 for(let frame=1;frame<=steps;frame++){
  const start=performance.now();if(!plain)profile.start();
  solver.advanceTo(frame*dt,[]);const encode_ms=performance.now()-start;await solver.awaitFrameCompletion();
  const wall_ms=performance.now()-start;
  assert.equal(solver.info.simulationPipelineError,undefined);
  assert.ok(Math.abs((solver.info.completedTime_s??0)-frame*dt)<1e-8,"complete exactly one requested clock step");
  assert.equal(solver.info.encodedSteps,frame,"one solver step a frame");
  const passes=plain?[]:await profile.finish();let projection_ms=0,rest_ms=0,between_ms=0;const byLabel:Record<string,number>={},before:Record<string,number>={};
  for(const pass of passes){
   byLabel[pass.label]=(byLabel[pass.label]??0)+pass.ms;between_ms+=pass.idle_ms;
   if(pass.idle_ms>0.05)before[pass.label]=(before[pass.label]??0)+pass.idle_ms;
   if(PROJECTION.test(pass.label))projection_ms+=pass.ms;else rest_ms+=pass.ms;
   const total=labels.get(pass.label)??{ms:0,passes:0,dispatches:0};
   total.ms+=pass.ms;total.passes++;total.dispatches+=pass.dispatches;labels.set(pass.label,total);
  }
  const flip=solver.narrowBandFlipInfo!;
  const row:Record<string,unknown>={frame,time_s:frame*dt,wall_ms,encode_ms,projection_ms,rest_ms,gpu_ms:projection_ms+rest_ms,between_ms,before,passes:passes.length,
   particles:flip.particles,particleCapacity:flip.capacity,reseedClipped:flip.reseedClipped,allocatedBytes:solver.info.allocatedBytes,
   fineTiles:solver.info.uniformMixedFineTiles,bandTiles:solver.info.uniformPressureBandTiles,labels:byLabel};
  if(track){
   // The highest liquid vertex in each falling letter's column, with the crossing above it interpolated.
   const fields=await readUniformFields(device,solver);
   row.tops=Object.fromEntries(letters.filter(l=>frame>=l.step&&frame<l.step+14).map(l=>{
    for(let y=ny-1;y>=0;y--){const below=fields.vertex(l.x,y,l.z),above=fields.vertex(l.x,y+1,l.z);
     if(below<0&&above>=0)return [l.letter,{top:y+below/(below-above),fallen:l.top0-(y+below/(below-above)),sinceDrop:frame-l.step}];}
    return [l.letter,null];
   }));
  }
  rows.push(row);
  if(frame%10===0||frame===steps||track)console.log(JSON.stringify({...row,labels:undefined}));
 }
 const final=await solver.readStats();assert.deepEqual(errors,[]);
 const of=(key:string)=>rows.slice(from-1).map(r=>r[key] as number);
 const summary={steps,from,sets,grid:[nx,ny,nz],cellSize_m:h,dt_s:dt,construction_ms,
  projection_ms:mean(of("projection_ms")),rest_ms:mean(of("rest_ms")),gpu_ms:mean(of("gpu_ms")),wall_ms:mean(of("wall_ms")),encode_ms:mean(of("encode_ms")),between_ms:mean(of("between_ms")),
  wallMedian_ms:of("wall_ms").sort((a,b)=>a-b)[(steps-from+1)>>1],wallMax_ms:Math.max(...of("wall_ms")),
  particles:mean(of("particles")),particlesMax:Math.max(...of("particles")),particleCapacity:rows.at(-1)!.particleCapacity,
  allocatedBytes:mean(of("allocatedBytes")),allocatedBytesMax:Math.max(...of("allocatedBytes")),
  volumeDrift:final.volumeDrift,tracked:track,plain};
 console.log(JSON.stringify(summary));
 write(`nbflip-letters-${name}.json`,{date:new Date().toISOString(),arguments:process.argv.slice(2),method:uniformNarrowBandMethod.id,values,
  adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},
  summary,letters,labels:[...labels].map(([label,t])=>({label,projection:PROJECTION.test(label),msPerStep:t.ms/steps,passesPerStep:t.passes/steps,dispatchesPerStep:t.dispatches/steps})).sort((a,b)=>b.msPerStep-a.msPerStep),
  scene,rows,final,errors});
}catch(error){
 write(`nbflip-letters-${name}-failed.json`,{date:new Date().toISOString(),arguments:process.argv.slice(2),method:uniformNarrowBandMethod.id,values,scene,rows,errors,error:String(error)});
 throw error;
}finally{solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
