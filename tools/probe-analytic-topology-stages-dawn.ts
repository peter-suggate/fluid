/** Read-only production-stage audit: uniform controls versus a fixed 2:1 seam.
 * No solver flags, equations, or UI scene values are changed by the mixed lane.
 */
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import { resolveMethodValues } from '../lib/core/method-contract';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from '../lib/harness/webgpu-smoke-isolation';
import { adaptiveMassMethod } from '../lib/methods/adaptive-mass/method';
import type { WebGPUAdaptiveMassSolver } from '../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver';
const arg=(name:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${name}=`))?.split('=')[1]??fallback;
const lane=arg('lane','mixed'), motion=arg('motion','translation'), steps=Number(arg('steps','6'));
const seamWidth=Number(arg('seam-width','8'));
const label=arg('label','');assert.ok(/^[a-z0-9-]*$/.test(label));
assert.ok([2,4,8].includes(seamWidth));
assert.ok(seamWidth===8||motion==='translation'||motion==='free-fall', 'resolution controls use the fixed moving box');
assert.ok(['mixed','coarse','medium','fine'].includes(lane));
assert.ok(['translation','free-fall','rerung','standing-wave'].includes(motion));
assert.ok(Number.isInteger(steps)&&steps>0);
assert.ok(process.env.WEBGPU_NODE_MODULE);
await acquireWebGPUExclusiveLock('dawn-probe','analytic-topology-stages');
let device:GPUDevice|undefined, solver:WebGPUAdaptiveMassSolver|undefined;
// Dawn's native GPU must stay rooted through asynchronous readback and JS GC.
const liveGPUs=new Set<GPU>();Object.assign(globalThis,{analyticTopologyAuditGPUs:liveGPUs});
const results:unknown[]=[];
const artifactDirectory=`artifacts/analytic-motion/stages-${motion}-${lane}${seamWidth===8?'':`-width${seamWidth}`}${label?`-${label}`:''}`;
mkdirSync(artifactDirectory,{recursive:true});
try {
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE).href);Object.assign(globalThis,dawn.globals);
 const gpu:GPU=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]);
 liveGPUs.add(gpu);
 const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const definition=getSceneDefinition(`coarse-surface-${motion==='rerung'?'free-fall-rerung':motion}`);
 const scene=sceneDocument(definition);
 if(lane==='mixed'&&seamWidth!==8){
   scene.fluid.refinementRegions=scene.fluid.refinementRegions!.map((region,index)=>({...region,
     minimumCellSize_cells:index===0?seamWidth:seamWidth/2,
     maximumCellSize_cells:index===0?seamWidth:seamWidth/2}));
 }
 if(lane!=='mixed'){
   const width=lane==='coarse'?seamWidth:lane==='medium'?seamWidth/2:1;
   scene.fluid.refinementRegions=[{id:'uniform-control',rule:'minimum-cell-size',minimumCellSize_cells:width,maximumCellSize_cells:width,
    min_m:{x:-.8,y:0,z:-.2},max_m:{x:.8,y:scene.container.height_m,z:.2}}];
   delete scene.fluid.refinementKeyframes;
 }
 const files=['lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts',
   'lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts','lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts',
   'lib/methods/adaptive-mass/sparse-cm12-transport-execution-image.wgsl.ts'];
 writeFileSync(`${artifactDirectory}/configuration.json`,JSON.stringify({lane,motion,seamWidth,steps,scene,
   sourceSha256:Object.fromEntries(files.map(file=>[file,createHash('sha256').update(readFileSync(file)).digest('hex')]))},null,2));
 const values=resolveMethodValues(adaptiveMassMethod,'balanced',definition.methodProfile!.overrides);
 solver=await adaptiveMassMethod.createSolverAsync!(device,scene,'balanced',values,undefined,()=>{}) as WebGPUAdaptiveMassSolver;
 await solver.waitForSimulationReady();
 const stages=['transport-velocity-extension','face-preparation','conservative-transport','gamma-diffusion','surface-sharpening','body-forces','velocity-projection','candidate-transfer'];
 for(let step=1;step<=steps;step++){
   const source=solver.fieldSnapshotSourceForQA, w=source.templateWords, f=new Float32Array(w.buffer,w.byteOffset,w.length);
   const effectiveVelocity=source.effectiveTransportVelocity;assert.ok(effectiveVelocity);
   const nc=source.cellCapacity,nr=source.rowCapacity;
   const planeNames=['densityA','densityB','cellVelocityA','cellVelocityB','faceA','faceB','gammaA','gammaB','pressure','rhs'] as const;
   const lengths=[nc,nc,4*nc,4*nc,nr,nr,nc,nc,nc,nc];
   const l={} as Record<typeof planeNames[number],number>;let floatCount=0;
   for(const [i,name] of planeNames.entries()){l[name]=floatCount;floatCount+=lengths[i]!;}
   const extendedBase=floatCount;floatCount+=4*nc;
   const before=await solver.readGPUActivityPolicy();
   const frameResults:unknown[]=[];
   const captures=new Map<string,GPUBuffer>();
   const capture=(stage:string,encoder:GPUCommandEncoder)=>{
     if(stage!=='initial'&&!stages.includes(stage))return;
     const buffer=device!.createBuffer({size:4*floatCount+8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
     for(const [i,name] of planeNames.entries())encoder.copyBufferToBuffer(source.state,4*source.layout[name],buffer,4*l[name],4*lengths[i]!);
     encoder.copyBufferToBuffer(effectiveVelocity,0,buffer,4*extendedBase,16*nc);
     for(const [i,parity] of [source.scalarParityWord,source.faceParityWord].entries())encoder.copyBufferToBuffer(source.topologyArena,4*(source.frameControlBaseWords+parity),buffer,buffer.size-8+4*i,4);
     captures.set(stage,buffer);
   };
   const initialEncoder=device.createCommandEncoder();capture('initial',initialEncoder);
   device.queue.submit([initialEncoder.finish()]);
   solver.setStageCaptureForQA(capture);
   while(!solver.advanceTo(step/60,[]))await new Promise(setImmediate);
   await solver.waitForTopologyReady();await solver.assertSimulationHealthy();solver.setStageCaptureForQA(undefined);
   assert.equal(solver.info.encodedSteps,step);
   const after=await solver.readGPUActivityPolicy();
   const fields=await solver.readDiagnosticFields(true);
   const diagnosticMass=fields.density.reduce((a,b)=>a+b,0)*.05**3;
   const diagnosticMomentum=fields.density.reduce((a,b,i)=>a+b*fields.velocity[4*i+1]!,0)*.05**3;
   for(const [stage,buffer] of captures){
     try{
       await buffer.mapAsync(GPUMapMode.READ);const data=new Float32Array(buffer.getMappedRange());
       const parity=new Uint32Array(data.buffer,buffer.size-8,2);
       const densityBase=[l.densityA,l.densityB][parity[0]!^Number(stages.indexOf(stage)>=2)]!;
       const gammaBase=[l.gammaA,l.gammaB][parity[0]!^Number(stages.indexOf(stage)>=2)]!;
       const cellBase=[l.cellVelocityA,l.cellVelocityB][parity[1]!^Number(stages.indexOf(stage)>=2)]!;
       const faceBase=[l.faceA,l.faceB][parity[1]!^Number(stages.indexOf(stage)>=1)]!;
       const activity=stage==='candidate-transfer'?after:before;
       // Static template incidence describes authored leaves; frontier air is excluded.
       const frontierLeaves=activity.bricks.filter(b=>b.active&&b.leafId>=w[13]!).length;
       const cells=[];
       for(const b of activity.bricks.filter(b=>b.active&&b.leafId<w[13]!)){
         const range=w[11]!+2*(4*b.leafId+Math.log2(b.acceptedResolution));
         for(let i=0;i<w[range+1]!;i++){
           const id=w[range]!+i, at=w[6]!+8*id,rho=data[stage==='gamma-diffusion'?l.pressure+id:densityBase+id]!;
           if(rho<1e-7)continue;
           const faces=[];
           for(let j=w[w[9]!+id]!;j<w[w[9]!+id+1]!;j++){
             const row=w[w[10]!+2*j]!, term=w[w[10]!+2*j+1]!, nr=w[3]!, meta=w[w[7]!+nr+row]!, condition=meta&0xfffffff;
             const enabled=Array.from({length:w[condition]!},(_,k)=>w[condition+1+k]!).every(m=>{
               const brick=activity.bricks.find(v=>v.leafId===(m>>>5));return brick?.active&&brick.acceptedResolution===(m&31);
             });
             if(!enabled)continue;
             faces.push({row,axis:meta>>>30,kind:(meta>>>28)&3,velocity:data[faceBase+row]!*.05,
               coefficient:f[w[8]!+2*term+1]!,dualWeight:f[w[7]!+2*nr+row]!});
           }
           cells.push({id,position:[f[at]!*.05,f[at+1]!*.05,f[at+2]!*.05],width:f[at+4]!*.05,rho,
             mass:rho*f[at+3]!*.05**3,gamma:data[stage==='gamma-diffusion'?l.rhs+id:gammaBase+id]!,velocity:data[cellBase+4*id+1]!*.05,velocityX:data[cellBase+4*id]!*.05,
             extendedVelocity:data[extendedBase+4*id+1]!*.05,faces});
         }
       }
       const mass=cells.reduce((s,c)=>s+c.mass,0);
       const bins:Record<string,{mass:number,momentum:number,extendedMomentum:number,faceMomentum:number}>={};
       for(const c of cells){const bin=bins[String(c.width)]??={mass:0,momentum:0,extendedMomentum:0,faceMomentum:0};
         const yf=c.faces.filter(v=>v.axis===1), denominator=yf.reduce((s,v)=>s+Math.abs(v.coefficient)*v.dualWeight,0);
         const v=denominator?yf.reduce((s,v)=>s+v.velocity*Math.abs(v.coefficient)*v.dualWeight,0)/denominator:0;
         bin.mass+=c.mass;bin.momentum+=c.mass*c.velocity;bin.extendedMomentum+=c.mass*c.extendedVelocity;bin.faceMomentum+=c.mass*v;
       }
       if(stage==='candidate-transfer'){
         assert.ok(Math.abs(mass-diagnosticMass)<1e-7, 'native capture must account for published mass');
         const momentum=Object.values(bins).reduce((sum,b)=>sum+b.momentum,0);
         assert.ok(Math.abs(momentum-diagnosticMomentum)<1e-7, 'native capture must match published momentum');
       }
       const summary={step,stage,mass,frontierLeaves,bins};console.log(JSON.stringify(summary));results.push(summary);frameResults.push({...summary,cells});
     }finally{if(buffer.mapState==='mapped')buffer.unmap();buffer.destroy();}
   }
   writeFileSync(`${artifactDirectory}/step-${step}.json`,JSON.stringify(frameResults));
 }
} finally {solver?.destroy();device?.destroy();liveGPUs.clear();await releaseWebGPUExclusiveLock();}
mkdirSync('artifacts/analytic-motion',{recursive:true});
writeFileSync(`${artifactDirectory}.json`,JSON.stringify(results));
