/** Run serially under the GPU lock. Same probe in frozen before/after source trees:
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js OUT=/tmp/pool.json
 * node --import tsx tools/benchmark-pool-settling.ts
 * Wall times exclude diagnostics; current phi volume is explicitly measured. */
import assert from "node:assert/strict";
import {mkdirSync,readFileSync,readdirSync,writeFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {dirname,resolve} from "node:path";
import {createBackyardPoolPartyScene} from "../lib/core/backyard-pool-party-scene";
import {initializeRigidBodies} from "../lib/core/rigid-body";
import {paceDawnEventPolling} from "../lib/harness/node-dawn-provider";
import {uniformMixedPresentationWGSL} from "../lib/methods/uniform/uniform-mixed-presentation.wgsl";
import type {UniformMixedFrame} from "../lib/methods/uniform/uniform-mixed-frame";
import {createUniformSolver,withUniformDevice} from "../tests/helpers/uniform-geometric";
import {readMixedBuffer} from "../tests/helpers/uniform-mixed-native-fields";
import {auditPoolRegions} from "./audit-pool-regions";
import {measureCurrentPoolPhiVolume,type PoolVolumeFrame} from "./measure-pool-volume";
import {uniformNearSettled} from "../lib/methods/uniform/uniform-near-settled";

const out=resolve(process.env.OUT??"/tmp/pool-settling.json"),steps=Number(process.env.STEPS??960),from=Number(process.env.FROM??841);
assert.ok(Number.isInteger(steps)&&Number.isInteger(from)&&from>0&&steps>=from);
const fingerprint=()=>{const hash=createHash("sha256");for(const file of readdirSync("lib/methods/uniform").filter(f=>f.endsWith(".ts")).sort())hash.update(file).update(readFileSync(`lib/methods/uniform/${file}`));return hash.digest("hex");};
const sourceHash=fingerprint();
const methodValues=JSON.parse(process.env.METHOD_VALUES??"{}");
process.env.WEBGPU_NODE_MODULE??=resolve("node_modules/webgpu/index.js");
paceDawnEventPolling();
const rows:Record<string,unknown>[]=[];
const mean=(a:number[])=>a.reduce((s,v)=>s+v,0)/a.length;
const median=(a:number[])=>a.toSorted((a,b)=>a-b)[a.length>>1];
await withUniformDevice("pool settling benchmark",async device=>{
 device.addEventListener("uncapturederror",event=>console.error(event.error.message));
 const scene=createBackyardPoolPartyScene(),bodies=initializeRigidBodies(scene.rigidBodies);
 assert.ok(steps/60<=scene.duration_s,"benchmark must not run past scene duration");
 if(process.env.NO_BODY==="1"){scene.rigidBodies=[];bodies.length=0;}
 const constructed=performance.now(),solver=await createUniformSolver(device,scene,methodValues);
 const inner=solver as unknown as {mixedFrame:UniformMixedFrame;rigidSystem:{stateBuffer:GPUBuffer};rigidExchange:GPUBuffer;frame:{origin_m:number[];cellSize_m:number[]}};
 const f=inner.mixedFrame;console.log(JSON.stringify({constructed_ms:performance.now()-constructed}));
 const surfaceProbe=await createSurfaceProbe();
 async function createSurfaceProbe(){
  // Heights through the production phi sampler, at fixed world x/z columns.
  // Exclude the body/contact patch; include the travelling waves in the basin.
  const points:number[]=[];
  for(let z=-1.7;z<1.8;z+=.2)for(let x=-3.7;x<3.8;x+=.2){

   points.push((x-inner.frame.origin_m[0]!)/.05,0,(z-inner.frame.origin_m[2]!)/.05,0);
  }
  const input=device.createBuffer({size:points.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,new Float32Array(points));
  const output=device.createBuffer({size:points.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const code=uniformMixedPresentationWGSL(0,"phi","coarsePhi")+`
@group(0) @binding(1) var phi:texture_3d<f32>;
@group(0) @binding(2) var coarsePhi:texture_3d<f32>;
@group(0) @binding(3) var<storage,read> points:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> heights:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 udrInit();if(id.x>=arrayLength(&points)){return;}let p=points[id.x].xyz-32.0*vec3f(umFramePage());
 var last=0.0;var height=-100.0;
 for(var j=0u;j<=100u;j++){let y=1.0+.05*f32(j);let value=umSampleVertex(vec3f(p.x,(y-${inner.frame.origin_m[1]})/.05-32.0*f32(umFramePage().y),p.z));
 if(j>0u&&last<0.0&&value>=0.0){height=y-.05+.05*(-last)/(value-last);}last=value;}
 heights[id.x]=height;
}`;
  const layout=device.createBindGroupLayout({entries:[0,1,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,...(binding===1||binding===2?{texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}}:{buffer:{type:binding===4?"storage" as const:"read-only-storage" as const}})}))});
  const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module:device.createShaderModule({code}),entryPoint:"main"}});
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:f.ownership.presentation},{binding:1,resource:solver.vertexPhiTexture!.createView({dimension:"3d"})},{binding:2,resource:f.coarseVertexPhi.createView({dimension:"3d"})},{binding:3,resource:{buffer:input}},{binding:4,resource:{buffer:output}}]});
  return {async measure(body:number[]){
  const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(Math.ceil(points.length/4/64));p.end();device.queue.submit([e.finish()]);
  const raw=Array.from(await readMixedBuffer(device,output));
  const heights=raw.filter((_,i)=>!bodies.length||Math.hypot(points[4*i]!*.05+inner.frame.origin_m[0]!-body[0]!,points[4*i+2]!*.05+inner.frame.origin_m[2]!-body[2]!)>=.6);
  assert.ok(heights.every(v=>Number.isFinite(v)&&v>1&&v<6),"each basin column must retain a finite water surface");
  return heights;},destroy(){input.destroy();output.destroy();}};
 }
 let previous:number[]|undefined,deepest=Infinity,impulseY=0,maxMomentumError=0;
 try{
  for(let frame=1;frame<=steps;frame++){
   const start=performance.now();assert.ok(solver.advanceTo(frame/60,bodies));const encode_ms=performance.now()-start;
   await solver.awaitFrameCompletion();const ms=performance.now()-start;
   assert.equal(solver.info.simulationPipelineError,undefined);
   const body=await readMixedBuffer(device,inner.rigidSystem.stateBuffer);
   assert.ok(Array.from(body.slice(0,15)).every(Number.isFinite),"rigid state must remain finite");
   const exchange=new Int32Array((await readMixedBuffer(device,inner.rigidExchange)).buffer);
   if(bodies.length&&body[1]!<2.1){
    deepest=Math.min(deepest,body[1]!);impulseY+=exchange[1]!*1e-6;
    if(previous)for(let a=0;a<3;a++)maxMomentumError=Math.max(maxMomentumError,Math.abs(bodies[0]!.mass_kg*(body[12+a]!-previous[a]!-(a===1?scene.fluid.gravity_m_s2.y/60:0))-exchange[a]!*1e-6));
   }
   previous=Array.from(body.slice(12,15));
   const tierWords=new Uint32Array((await readMixedBuffer(device,f.heat.tiers)).buffer);
   const row={frame,ms,encode_ms,bodyPosition:Array.from(body.slice(0,3)),bodyVelocity:previous,
    tiers:Array.from(tierWords.slice(f.heat.regions,f.heat.regions+3)),particles:solver.narrowBandFlipInfo?.particles,
    bandTiles:solver.info.uniformPressureBandTiles,rootResidual:solver.info.uniformPressureAcceptedResidual,bandResidual:solver.info.uniformPressureBandResidual};
   // Reuse the diagnostic read above to measure per-region tier chatter.
   if(frame>=from)Object.assign(row,{regionTiers:Array.from(tierWords.subarray(0,f.heat.regions))});
   if(process.env.AUDIT==="1"&&(frame===360||frame===600||frame>=840&&frame%30===0)){const audit=await auditPoolRegions(device,f);Object.assign(row,{audit});console.log(JSON.stringify({frame,auditTotals:audit.totals}));}
   if([270,285,300,330,360,480].includes(frame)){
    const heights=await surfaceProbe.measure(row.bodyPosition);
    Object.assign(row,{surface:{mean:mean(heights),rms:Math.sqrt(mean(heights.map(y=>(y-1.8)**2))),crest:Math.max(...heights)-1.8,trough:Math.min(...heights)-1.8}});
   }
   rows.push(row);if(frame%120===0||frame===steps)console.log(JSON.stringify(row));
  }
  measureCurrentPoolPhiVolume(device,f as unknown as PoolVolumeFrame);const stats=await solver.readStats();
  if(stats.volumeTelemetrySource!=="narrow-band-phi")console.error(JSON.stringify({volumeSource:stats.volumeTelemetrySource,volumeDrift:stats.volumeDrift,budget:Array.from(await readMixedBuffer(device,f.narrowBandVolumeBudget!))}));
  assert.equal(stats.volumeTelemetrySource,"narrow-band-phi");
  const last=rows.at(-1)!;const body=last.bodyPosition as number[];
  const heights=await surfaceProbe.measure(body);
  const measured=rows.filter(r=>(r.frame as number)>=from),ys=measured.map(r=>(r.bodyPosition as number[])[1]!);
  const summary={median_ms:median(measured.map(r=>r.ms as number)),mean_ms:mean(measured.map(r=>r.ms as number)),meanHot:mean(measured.map(r=>(r.tiers as number[])[2]!)),meanWarm:mean(measured.map(r=>(r.tiers as number[])[1]!)),meanCold:mean(measured.map(r=>(r.tiers as number[])[0]!)),meanParticles:mean(measured.map(r=>r.particles as number)),
   bodyMeanY:mean(ys),bodyMinY:Math.min(...ys),bodyMaxY:Math.max(...ys),deepest,impulseY,maxMomentumError,
   volumeDrift:stats.volumeDrift,currentVolume_m3:stats.volumeCellSum!*.05**3,maxSpeed:stats.maxSpeed_m_s,
   surfaceMean:mean(heights),surfaceRms:Math.sqrt(mean(heights.map(y=>(y-1.8)**2))),surfaceMax:Math.max(...heights.map(y=>Math.abs(y-1.8))),final:last};
  assert.equal(fingerprint(),sourceHash,"source changed during benchmark");
  mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify({sourceHash,runtime:{version:process.version,executable:process.execPath},methodValues,nearSettled:uniformNearSettled(methodValues),steps,from,summary,heights,rows},null,2));console.log(JSON.stringify({out,summary}));
 }catch(error){writeFileSync(out+".failed.json",JSON.stringify({error:String(error),rows}));throw error;}finally{surfaceProbe.destroy();solver.destroy();}
});
