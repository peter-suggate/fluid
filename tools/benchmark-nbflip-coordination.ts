/** Section 10 on the complete solver: per-step fence vs the existing two-frame
 * receipt ring. Two fenced controls bracket the pipelined solver. No physics,
 * convergence, or readback contents are changed. Runs rotate order by block.
 * bash tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/benchmark-nbflip-coordination.ts
 */
import assert from 'node:assert/strict';
import {mkdirSync,readdirSync,readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {sceneDocument} from '../lib/core/scene-definition';
import {getSceneDefinition} from '../lib/core/scenes';
import {resolveMethodValues} from '../lib/core/method-contract';
import {managedGPUDevice} from '../lib/core/gpu-compilation-manager';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import {uniformNarrowBandMethod as method} from '../lib/methods/uniform/uniform-narrow-band-method';
import {UNIFORM_MIXED_RECEIPT_RING} from '../lib/methods/uniform/uniform-mixed-frame';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
const option=(k:string)=>process.argv.find(a=>a.startsWith(`--${k}=`))?.slice(k.length+3);
const out=resolve(option('out')??'artifacts/nbflip-ten-sections-2026-11-02/coordination');mkdirSync(out,{recursive:true});
const steps=Number(option('steps')??120),block=12,warmup=24;assert.ok(steps>=36&&steps%block===0);
const sceneId=option('scene')??'sparse-cm12-long-dam-break',definition=getSceneDefinition(sceneId),scene=structuredClone(sceneDocument(definition));scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
const values=resolveMethodValues(method,'balanced',{...method.appDefaults,...definition.methodProfile?.overrides,timeStep:'sixtieth'}),dt=1/60;
const source=createHash('sha256');for(const file of readdirSync('lib/methods/uniform').filter(f=>f.endsWith('.ts')).sort()){source.update(file);source.update(readFileSync(`lib/methods/uniform/${file}`));}
paceDawnEventPolling();const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:(['timestamp-query','float32-filterable','subgroups'] as GPUFeatureName[]).filter(f=>adapter.features.has(f)),requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const modes=['fenced-A','pipelined','fenced-B'] as const;
const telemetry=modes.map(()=>({fences:0,submissions:0,maps:0,mapBytes:0,mapLabels:{} as Record<string,number>}));let measuring=-1;
const queue=new Proxy(raw.queue,{get(target,key){const value=Reflect.get(target,key,target);if(key==='onSubmittedWorkDone')return ()=>{if(measuring>=0)telemetry[measuring]!.fences++;return target.onSubmittedWorkDone();};if(key==='submit')return (commands:GPUCommandBuffer[])=>{if(measuring>=0)telemetry[measuring]!.submissions++;target.submit(commands);};return typeof value==='function'?value.bind(target):value;}});
const observed=new Proxy(raw,{get(target,key){if(key==='queue')return queue;if(key==='createBuffer')return (d:GPUBufferDescriptor)=>{const b=target.createBuffer(d);if(d.usage&GPUBufferUsage.MAP_READ){const map=b.mapAsync.bind(b);Object.defineProperty(b,'mapAsync',{value:(mode:GPUMapModeFlags,offset?:number,size?:number)=>{if(measuring>=0){const t=telemetry[measuring]!;t.maps++;t.mapBytes+=size??(b.size-(offset??0));t.mapLabels[b.label]=(t.mapLabels[b.label]??0)+1;}return map(mode,offset,size);}});}return b;};const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;}});
const device=managedGPUDevice(observed,{requireWorkerRealm:false}),solvers:WebGPUUniformReferenceSolver[]=[],errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
const rows:{mode:string;first:number;last:number;msPerStep:number;encodeMsPerStep:number;particles:number|undefined;fineTiles:number|undefined}[]=[];
const report={completed:false,scene:sceneId,sourceSha256:source.digest('hex'),dt,steps,warmup,ring:UNIFORM_MIXED_RECEIPT_RING,scope:'complete simulation CPU wall time, excluding construction, rendering, final diagnostics; no per-kernel timestamps',values,rows,telemetry,summary:[] as unknown[],physics:[] as unknown[],errors};
const save=()=>writeFileSync(`${out}/results.json`,JSON.stringify(report,null,2)+'\n');
save();
try{
 for(const mode of modes){const started=performance.now();solvers.push(await method.createSolverAsync!(device,structuredClone(scene),'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver);console.log(JSON.stringify({constructed:mode,ms:performance.now()-started}));}
 for(let first=1,round=0;first<=steps;first+=block,round++)for(let k=0;k<3;k++){
  const i=(k+round)%3,solver=solvers[i]!,last=first+block-1;measuring=first>warmup?i:-1;let encode=0;
  const started=performance.now();
  for(let f=first;f<=last;f++){
   const t=performance.now();assert.ok(solver.advanceTo(f*dt,[]),`admit ${modes[i]} frame ${f}`);encode+=performance.now()-t;
   if(i!==1||f===last)await solver.awaitFrameCompletion();else while(solver.framePending)await new Promise<void>(r=>setImmediate(r));
  }
  const ms=performance.now()-started;measuring=-1;assert.equal(solver.info.simulationPipelineError,undefined);assert.ok(Math.abs((solver.info.completedTime_s??0)-last*dt)<1e-8);
  const row={mode:modes[i]!,first,last,msPerStep:ms/block,encodeMsPerStep:encode/block,particles:solver.narrowBandFlipInfo?.particles,fineTiles:solver.workDomainInfo?.fineTiles};rows.push(row);save();console.log(JSON.stringify(row));
 }
 const median=(xs:number[])=>[...xs].sort((a,b)=>a-b)[Math.floor(xs.length/2)]!;
 report.summary=modes.map(mode=>{const ms=rows.filter(r=>r.first>warmup&&r.mode===mode).map(r=>r.msPerStep);return {mode,medianBlockMsPerStep:median(ms),meanMsPerStep:ms.reduce((a,b)=>a+b,0)/ms.length,blocks:ms};});
 for(let i=0;i<solvers.length;i++){const solver=solvers[i]!,s=await solver.readStats();report.physics.push({mode:modes[i],particles:solver.narrowBandFlipInfo?.particles,maxSpeed_m_s:s.maxSpeed_m_s,volumeCellSum:s.volumeCellSum,representedVolumeCellSum:s.representedVolumeCellSum,volumeDrift:s.volumeDrift,pressureResidual:s.uniformPressureAcceptedResidual,bandResidual:s.uniformPressureBandResidual});}
 assert.deepEqual(errors,[]);report.completed=true;save();console.log(JSON.stringify({summary:report.summary,telemetry,physics:report.physics}));
}catch(error){errors.push(error instanceof Error?error.stack??error.message:String(error));save();throw error;}
finally{measuring=-1;solvers.forEach(s=>s.destroy());await raw.queue.onSubmittedWorkDone();raw.destroy();}
