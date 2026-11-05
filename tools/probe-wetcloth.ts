/** Released-source validation on the production owners, at unchanged h/dt.
 * --audit imports every water fixture without running physics.
 * --scene=wetcloth-source-... [--steps=N] [--every=N] [--out=PATH]
 * --dt-scale=.5|.25 labels a timestep-convergence experiment; default1
 * retains the released timestep and every run retains the original h.
 * --probe-incomplete permits diagnostic runs whose missing features are
 * recorded in the report. Such a run is never certified as a reproduction.
 * Run GPU probes through tools/gpu/fluid-gpu-run.sh check|timing.
 */
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {gzipSync} from 'node:zlib';
import {wetClothValidationSources,wetClothSource,WETCLOTH_UPSTREAM_COMMIT} from '../lib/cloth/scenes/sources';
import {prepareWetClothSource,buildWetClothSourceScene,wetClothSourceCapabilities,wetClothValidationDefinitions} from '../lib/cloth/scenes/definitions';
import {sourceCameraForScene} from '../lib/cloth/source-camera';
import {withDryClothPhysics} from '../lib/cloth/dry-host';
import {WebGPULiveSvoScene} from '../lib/svo/features/scene-publication/webgpu-live-svo-scene';
import {createUniformSolver,withUniformDevice} from '../tests/helpers/uniform-geometric';
import {paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import type {GPUSolverInstance} from '../lib/core/method-contract';
import type {ClothSurfacePublication} from '../lib/cloth/types';
import {initializeRigidBodies} from '../lib/core/rigid-body';

const option=(key:string)=>process.argv.find(v=>v.startsWith(`--${key}=`))?.slice(key.length+3);
const request:typeof fetch=async(input)=>{
 const url=String(input);if(!url.startsWith('/assets/wetcloth/')||url.includes('..'))throw new Error(`Unexpected source URL ${url}`);
 return new Response(await readFile(resolve('public'+url)),{status:200});
};
const out=resolve(option('out')??'artifacts/wetcloth/report.json');await mkdir(dirname(out),{recursive:true});
const implemented=new Set(['elliptical-yarn-geometry','free-space-boundary','cloth-shell-mechanics','cloth-rod-mechanics','source-cubic-motion','narrowband-flip','cloth-liquid-two-way','cloth-porous-transport','source-csg','source-solid-collision','source-static-sdf','source-termination','source-sdf-emission','source-initial-sdf','source-external-mesh']);
const rows:unknown[]=[];
const report:Record<string,unknown>={schemaVersion:1,upstreamCommit:WETCLOTH_UPSTREAM_COMMIT,
 reproductionValidated:false,rows,notes:['Imported source fidelity and physical-result fidelity are separate gates.','Wet scenes use the production narrow-band FLIP owner; dry scenes explicitly disable fluid.']};
const save=()=>writeFile(out,JSON.stringify(report,null,2)+'\n');
if(process.argv.includes('--audit')){
 for(const source of wetClothValidationSources){
  try{
   await prepareWetClothSource(source.id,request);const scene=buildWetClothSourceScene(source.id);
   rows.push({id:source.id,status:'source-built',counts:source.counts,dt_s:source.dt_s,h_m:source.cellSize_m,duration_s:source.duration_s,
    requirements:scene.wetClothSource!.requiredCapabilities,unimplemented:scene.wetClothSource!.requiredCapabilities.filter(c=>!implemented.has(c)),validation:source.variant?`${source.variant.kind}-input-only`:'source-input-only',variant:source.variant,camera:sourceCameraForScene(scene),auditNotes:source.auditNotes});
  }catch(error){rows.push({id:source.id,status:'source-blocked',error:String(error)});}
  console.log(JSON.stringify({id:source.id,...rows[rows.length-1] as object}));await save();
 }
 await save();console.log(out);
}else{
 const id=option('scene');assert.ok(id,'Supply --audit or --scene=SOURCE_ID');
 const profile=wetClothValidationDefinitions.find(s=>s.id===id)!.methodProfile!;
 const source=wetClothSource(id);await prepareWetClothSource(id,request);const scene=buildWetClothSourceScene(id);
 const dtScale=Number(option('dt-scale')??1);assert.ok([1,.5,.25].includes(dtScale),'Use dt-scale1,.5 or.25');
 const dt=source.dt_s*dtScale;scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
 const bodies=initializeRigidBodies(scene.rigidBodies);
 const missing=scene.wetClothSource!.requiredCapabilities.filter(c=>!implemented.has(c));
 Object.assign(report,{sourceId:id,source:source.source,counts:source.counts,dt_s:dt,sourceDt_s:source.dt_s,dtScale,
  physics:{fluidOwner:scene.systems?.fluid===false?'disabled':'uniform-narrow-band-flip',sourceAttachments:source.counts.vertices?'finite-spring':'not-applicable',pressureCoupling:scene.systems?.fluid===false?'none':'split-mixture',solidIntegration:scene.cloth?.[0]?.physics?.solveSolid===false?'prescribed':'implicit-energy'},
  configuration:dtScale===1?(source.variant?.kind??'released-input'):'timestep-convergence',variant:source.variant,h_m:source.cellSize_m,
  duration_s:source.duration_s,camera:sourceCameraForScene(scene),unimplemented:missing,auditNotes:source.auditNotes,
  sourceSettings:source.sourceSettings,methodProfile:profile,status:missing.length?'partial-physics-probe':'unvalidated-physics-probe'});
 await save();
 if(!process.argv.includes('--probe-incomplete'))throw new Error(`This source trajectory is not reproduction-validated${missing.length?`; missing capabilities: ${missing.join(', ')}`:''}. Use --probe-incomplete for an explicitly unvalidated physical probe. See ${out}`);
 const steps=Number(option('steps')??Math.ceil(source.duration_s/dt)),every=Number(option('every')??Math.max(1,Math.round(.1/dt)));
 assert.ok(Number.isSafeInteger(steps)&&steps>0&&Number.isSafeInteger(every)&&every>0);
 process.env.WEBGPU_NODE_MODULE??=resolve('node_modules/webgpu/index.js');paceDawnEventPolling();
 const metrics=(surfaces:readonly ClothSurfacePublication[])=>{
  let bound=0,maximumSpeed=0,saturationMax=0;const minimum=[Infinity,Infinity,Infinity],maximum=[-Infinity,-Infinity,-Infinity],waterMoment=[0,0,0];
  const wetBounds=[.01,.1,.5,.9].map(saturation=>({saturation,vertices:0,minimum:[Infinity,Infinity,Infinity],maximum:[-Infinity,-Infinity,-Infinity]}));
  for(const s of surfaces){
   for(let v=0;v<s.positions_m.length/3;v++){
    bound+=s.waterVolume_m3[v]!;saturationMax=Math.max(saturationMax,s.saturation?.[v]??0);
    if(s.velocities_m_s)maximumSpeed=Math.max(maximumSpeed,Math.hypot(...s.velocities_m_s.subarray(3*v,3*v+3)));
    for(let a=0;a<3;a++){const p=s.positions_m[3*v+a]!;assert.ok(Number.isFinite(p));minimum[a]=Math.min(minimum[a]!,p);maximum[a]=Math.max(maximum[a]!,p);waterMoment[a]!+=p*s.waterVolume_m3[v]!;}
    for(const wet of wetBounds)if((s.saturation?.[v]??0)>=wet.saturation){wet.vertices++;for(let a=0;a<3;a++){const p=s.positions_m[3*v+a]!;wet.minimum[a]=Math.min(wet.minimum[a]!,p);wet.maximum[a]=Math.max(wet.maximum[a]!,p);}}
   }
  }
  return {boundVolume_m3:bound,clothMaximumSpeed_m_s:maximumSpeed,saturationMax,bounds:surfaces.length?{minimum,maximum}:null,
   boundWaterCentroid_m:bound>0?waterMoment.map(v=>v/bound):null,
   wetFrontBounds_m:wetBounds.map(wet=>({...wet,minimum:wet.vertices?wet.minimum:null,maximum:wet.vertices?wet.maximum:null}))};
 };
 await withUniformDevice('released wet-cloth source',async device=>{
  let solver:GPUSolverInstance|undefined;const started=performance.now();
  try{
   solver=scene.systems?.fluid===false
    ?await withDryClothPhysics(await WebGPULiveSvoScene.create(device,scene,'balanced',()=>{},new AbortController().signal),device,scene)
    :await createUniformSolver(device,scene,profile.overrides);
   assert.equal(solver.info.simulationPipelineError,undefined);
   report.initialization_ms=performance.now()-started;
   await writeFile(out+'.scene.json.gz',gzipSync(JSON.stringify(scene)));
   const sample=async(step:number,elapsed_ms:number)=>{
    // readStats returns the owner's mutable diagnostics record. Freeze each
    // sample so a later frame cannot rewrite the earlier comparison data.
    const surfaces=solver!.clothSurfaces??[],stats=structuredClone(await solver!.readStats()),cloth=metrics(surfaces);
    const cellVolume=source.cellSize_m**3;
    const water=scene.systems?.fluid===false?undefined:{
      freeTargetVolume_m3:stats.narrowBandTargetVolume_cells===undefined?undefined:stats.narrowBandTargetVolume_cells*cellVolume,
      representedFreeVolume_m3:stats.representedVolumeCellSum===undefined?undefined:stats.representedVolumeCellSum*cellVolume,
      nativeVolumeBeforeCorrection_m3:stats.narrowBandVolumeBeforeCorrection_cells===undefined?undefined:stats.narrowBandVolumeBeforeCorrection_cells*cellVolume,
      boundVolume_m3:cloth.boundVolume_m3,
      nativeOutflowVolume_m3:(stats.narrowBandOutflowVolume_cells??0)*cellVolume,
      pairedTransactions:structuredClone(solver!.clothWaterLedgers??[]),
    };
    const row={step,time_s:step*dt,elapsed_ms,...cloth,water,stats};rows.push(row);report.lastSampledStep=step;
    const file=out+`.frame-${step.toString().padStart(8,'0')}.json.gz`;
    await writeFile(file,gzipSync(JSON.stringify({step,sourceId:id,surfaces},(_key,v)=>ArrayBuffer.isView(v)?Array.from(v as unknown as ArrayLike<number>):v)));
    console.log(JSON.stringify({step,time_s:row.time_s,...metrics(surfaces)}));await save();
   };
   await sample(0,0);
   for(let step=1;step<=steps;step++){
    const t=performance.now();assert.ok(solver.advanceTo(step*dt,bodies),`Rejected source step ${step}`);
    await solver.awaitFrameCompletion?.();await solver.assertSimulationHealthy?.();
    assert.equal(solver.info.simulationPipelineError,undefined);
    for(const s of solver.clothSurfaces??[])assert.ok(Math.abs(s.time_s-step*dt)<1e-8,'Cloth clock diverged from experiment clock');
    report.lastAcceptedStep=step;
    if(step%every===0||step===steps)await sample(step,performance.now()-t);
   }
   report.completedSteps=steps;report.elapsed_ms=performance.now()-started;
  }catch(error){report.failure=String(error);if(error&&typeof error==='object'&&'localPoreDiagnostic' in error){const file=out+'.water-failure.json';await writeFile(file,JSON.stringify(error.localPoreDiagnostic));report.localPoreDiagnostic=file;}throw error;}
  finally{await save();solver?.destroy();}
 });
}
