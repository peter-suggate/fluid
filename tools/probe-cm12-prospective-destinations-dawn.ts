/** Read-only shadow census. No allocation or transport policy is changed.
 * Coordinates and velocities below use finest-cell units. Selection runs on
 * the CPU from the pre-step snapshot; its timings are NOT GPU speedup claims.
 */
import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createCm12Figure7} from '../lib/core/cm12-paper-scenes';
import {resolveMethodValues} from '../lib/core/method-contract';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from '../lib/harness/webgpu-smoke-isolation';
import {adaptiveMassMethod,adaptiveMassSolverOptions} from '../lib/methods/adaptive-volume/method';
import {WebGPUAdaptiveMassSolver} from '../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver';
import type {SparseCM12Phase1TransportQALayout} from '../lib/methods/adaptive-volume/sparse-cm12-phase1-transport-receipt';

const arg=(name:string,fallback:string)=>process.argv.find(v=>v.startsWith(`--${name}=`))?.slice(name.length+3)??fallback;
const frames=arg('frames','1,8,24,32').split(',').map(Number);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0));
const out=resolve(arg('out','artifacts/cm12-prospective-destinations'));
mkdirSync(out,{recursive:true});
// QA tooling deliberately reads existing private buffers. Keep this ABI local:
// production scheduling and buffer layouts remain untouched.
type ResidentBuffers={topologyArena:GPUBuffer;topologyWorklistBaseBytes:number;
 templateWords:Uint32Array;state:GPUBuffer;activity:GPUBuffer;
 layout:{densityA:number;densityB:number;cellVelocityA:number;cellVelocityB:number};
 phase1TransportQALayout:SparseCM12Phase1TransportQALayout;templateCellCount:number;topologyPageCapacity:number;brickFineResolution:number};
type Cell={id:number;center:number[];width:number[];rho:number;velocity:number[]};
const live=new Set<GPU>();
let device:GPUDevice|undefined,solver:WebGPUAdaptiveMassSolver|undefined,gpu:GPU|undefined;
await acquireWebGPUExclusiveLock('dawn-probe','prospective-destination-shadow-census');
try{
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);
 Object.assign(globalThis,dawn.globals);gpu=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]);live.add(gpu!);
 const adapter=await gpu!.requestAdapter();assert.ok(adapter);
 device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
 const scene=createCm12Figure7();scene.duration_s=10;
 scene.fluid.initialLiquidVolumes=[{shape:'sphere',center_m:{x:0,y:4.5,z:0},radius_m:.1}];
 const values=resolveMethodValues(adaptiveMassMethod,'balanced',{timeStep:'scene',brickFineResolution:'8',presentationPageResolution:'8',selectorMode:'coarse-first',pressureRelativeTolerance:.194});
 solver=await WebGPUAdaptiveMassSolver.createPhase1TransportReceiptOracleForQA(device,scene,'balanced',undefined,adaptiveMassSolverOptions(values),()=>{});
 await solver.waitForSimulationReady();
 const resident=solver.sparseWorldTrace;
 let buffers=(resident as unknown as {resident:ResidentBuffers}).resident;
 async function read(buffer:GPUBuffer,offset:number,bytes:number){
  const staging=device!.createBuffer({size:Math.max(4,bytes),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const encoder=device!.createCommandEncoder();encoder.copyBufferToBuffer(buffer,offset,staging,0,bytes);device!.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);return staging.getMappedRange().slice(0,bytes);}
  finally{if(staging.mapState==='mapped')staging.unmap();staging.destroy();}
 }
 async function snapshot(){
  const header=new Uint32Array(await read(buffers.topologyArena,buffers.topologyWorklistBaseBytes,128));
  const ids=new Uint32Array(await read(buffers.topologyArena,buffers.topologyWorklistBaseBytes+4*header[14+(header[2]!&1)]!,4*header[4]!));
  const capacity=buffers.phase1TransportQALayout.cellCapacity;
  const geometry=new Float32Array(await read(buffers.topologyArena,4*buffers.templateWords[6]!,32*capacity));
  const state=new Float32Array(await read(buffers.state,0,buffers.state.size));
  const frame=await resident.readFrameControlQA();
  const densityAt=frame.scalarParity?buffers.layout.densityB:buffers.layout.densityA;
  const velocityAt=frame.scalarParity?buffers.layout.cellVelocityB:buffers.layout.cellVelocityA;
  const B=buffers.brickFineResolution;
  const dynamicOffset=buffers.templateCellCount-B**3*buffers.topologyPageCapacity;
  const activity=await resident.readActivitySnapshot(true);
  const pages=new Map(activity.records.filter(r=>r.active&&r.topologyPage!==undefined&&r.coordinate).map(r=>[r.topologyPage!,r.coordinate!]));
  const cells=Array.from(ids,id=>({id,center:Array.from(geometry.slice(8*id,8*id+3)),width:Array.from(geometry.slice(8*id+4,8*id+7)),rho:state[densityAt+id]!,velocity:Array.from(state.slice(velocityAt+4*id,velocityAt+4*id+3))}));
  for(const c of cells)if(c.id>=dynamicOffset){
   const local=c.id-dynamicOffset;const page=Math.floor(local/B**3);const within=local%B**3;
   const origin=pages.get(page);assert.ok(origin,`dynamic page ${page} is not active`);
   c.width=[1,1,1];c.center=[within%B,Math.floor(within/B)%B,Math.floor(within/(B*B))].map((v,k)=>origin[k]!*B+v+.5);
  }
  assert.ok(cells.every(c=>c.center.every(Number.isFinite)&&c.width.every(w=>w>0)&&Number.isFinite(c.rho)&&c.rho>=0),JSON.stringify(cells.filter(c=>!c.center.every(Number.isFinite)||!c.width.every(w=>w>0)||!Number.isFinite(c.rho)||c.rho<0).slice(0,3)));
  return {generation:header[0],cells};
 }
 const dt=scene.numerics.fixedDt_s??scene.numerics.maxDt_s;
 const reports:unknown[]=[];
 for(let step=1;step<=Math.max(...frames);step++){
  buffers=(resident as unknown as {resident:ResidentBuffers}).resident;
  const capture=frames.includes(step);const before=capture?await snapshot():undefined;
  while(!solver.advanceTo(step*dt,[]))await new Promise(setImmediate);
  await device.queue.onSubmittedWorkDone();await solver.assertSimulationHealthy();
  if(!before)continue;
  const layout=buffers.phase1TransportQALayout;
  const raw=new Uint32Array(await read(buffers.activity,4*layout.baseWords,4*(layout.totalWords-layout.baseWords)));
  const floats=new Float32Array(raw.buffer);const signed=new Int32Array(raw.buffer);
  const at=(base:number)=>base-layout.baseWords;
  assert.equal(raw[4],before.generation,'captured transport must use the pre-step topology');
  const cells=before.cells,byId=new Map(cells.map(c=>[c.id,c]));
  const wet=cells.filter(c=>c.rho>0);
  const maxWidth=Math.max(...cells.flatMap(c=>c.width));
  const maxVelocity=[0,1,2].map(k=>Math.max(...cells.map(c=>Math.abs(c.velocity[k]!))));
  const backward=new Set<number>(),forward=new Set<number>(),gathered=new Set<number>();
  let traced=0,unknownWetDonors=0;const maxDeparture=[0,0,0];
  for(const c of cells){
   if(raw[at(layout.packetLaneBaseWords)+c.id]===0)continue;
   const hasStencil=Array.from({length:8},(_,j)=>floats[at(layout.stencilWeightBaseWords)+8*c.id+j]!).some(w=>w>0);
   if(hasStencil)traced++;
   if(hasStencil)for(let k=0;k<3;k++)maxDeparture[k]=Math.max(maxDeparture[k]!,Math.abs(floats[at(layout.departureBaseWords)+3*c.id+k]!-c.center[k]!));
   for(let j=0;j<8;j++){
    const donor=raw[at(layout.stencilCellBaseWords)+8*c.id+j]!;
    const weight=floats[at(layout.stencilWeightBaseWords)+8*c.id+j]!;
    if(weight>0&&donor!==0xffffffff){if(!byId.has(donor))unknownWetDonors++;else if(byId.get(donor)!.rho>0)backward.add(c.id);}
   }
   if(signed[at(layout.deficitDensityBaseWords)+c.id]!==0)forward.add(c.id);
   if(floats[at(layout.massDensityBaseWords)+c.id]!>0)gathered.add(c.id);
  }
  assert.equal(unknownWetDonors,0,'all valid donors must belong to pre-step topology');
  const required=new Set([...backward,...forward,...gathered]);
  const key=(p:readonly number[],size:number)=>p.map(v=>Math.floor(v/size)).join(',');
  const requiredPackets=new Set([...required].map(id=>key(byId.get(id)!.center,4)));
  const requiredBricks=new Set([...required].map(id=>key(byId.get(id)!.center,8)));
  const residentBricks=new Set(cells.map(c=>key(c.center,8)));
  const variants=[];
  for(const haloWidths of [1,2]){
   const started=performance.now();const packets=new Set<string>();
   // Enumerate coordinate-only candidate boxes, including absent coordinates.
   // Symmetric speed bound covers backward/forward travel; halo uses the
   // largest accepted interpolation width. This is deliberately conservative.
   for(const c of wet){
    const lo=c.center.map((v,k)=>Math.floor((v-c.width[k]!/2-dt*maxVelocity[k]!-haloWidths*maxWidth)/4));
    const hi=c.center.map((v,k)=>Math.floor((v+c.width[k]!/2+dt*maxVelocity[k]!+haloWidths*maxWidth)/4));
    for(let z=lo[2]!;z<=hi[2]!;z++)for(let y=lo[1]!;y<=hi[1]!;y++)for(let x=lo[0]!;x<=hi[0]!;x++)packets.add(`${x},${y},${z}`);
   }
   const selectionCpuMs=performance.now()-started;
   const bricks=new Set([...packets].map(p=>p.split(',').map(v=>Math.floor(Number(v)/2)).join(',')));
   const selected=cells.filter(c=>packets.has(key(c.center,4)));
   const missed=[...required].filter(id=>!packets.has(key(byId.get(id)!.center,4)));
   variants.push({haloWidths,candidatePackets:packets.size,candidateLogicalBricks:bricks.size,absentCandidateLogicalBricks:[...bricks].filter(b=>!residentBricks.has(b)).length,selectedResidentCells:selected.length,missedRequired:missed.length,missedForward:[...forward].filter(id=>!packets.has(key(byId.get(id)!.center,4))).length,selectionCpuMs,firstMiss:missed[0]??null});
  }
  const report={step,time_s:step*dt,topologyGeneration:before.generation,acceptedCells:cells.length,residentLogicalBricks:residentBricks.size,nonzeroSourceCells:wet.length,maxWidthFineCells:maxWidth,maxSourceVelocityFineCells_s:maxVelocity,maxObservedDepartureFineCells:maxDeparture,tracedPacketCells:traced,backwardPotentialReceivers:backward.size,forwardReturnReceivers:forward.size,forwardOnlyReceivers:[...forward].filter(id=>!backward.has(id)).length,gatheredNonzeroReceivers:gathered.size,requiredUnion:required.size,requiredPackets:requiredPackets.size,requiredLogicalBricks:requiredBricks.size,variants};
  reports.push(report);console.log(JSON.stringify(report));
  writeFileSync(resolve(out,`frame-${step}.json`),JSON.stringify({report,cells,backward:[...backward],forward:[...forward],gathered:[...gathered]},null,2));
 }
 assert.deepEqual(errors,[]);
 writeFileSync(resolve(out,'report.json'),JSON.stringify({scene,values,scope:'Read-only CPU candidate census against production GPU transport QA; no allocation reduction or GPU selection timing claimed.',reports},null,2)+'\n');
}finally{solver?.destroy();device?.destroy();if(gpu)live.delete(gpu);await releaseWebGPUExclusiveLock();}
