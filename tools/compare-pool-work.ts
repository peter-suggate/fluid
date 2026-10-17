/** Compare matching investigate-pool-work JSON and optional FIELDS_OUT captures.
 * COMPARE_FROM=2 COMPARE_TO=30 reaggregates per-frame data to an exact shared window.
 * node --import tsx tools/compare-pool-work.ts before.json after.json [before.bin after.bin]
 */
import assert from "node:assert/strict";
import {createBackyardPoolPartyScene} from "../lib/core/backyard-pool-party-scene";
import {readFileSync} from "node:fs";
const [beforePath,afterPath,beforeFields,afterFields]=process.argv.slice(2);
assert.ok(beforePath&&afterPath,"expected before.json after.json [before.bin after.bin]");
const compareFrom=process.env.COMPARE_FROM===undefined?undefined:Number(process.env.COMPARE_FROM),compareTo=process.env.COMPARE_TO===undefined?undefined:Number(process.env.COMPARE_TO);
function rewindow(run:any){
 if(compareFrom===undefined&&compareTo===undefined)return run;
 const from=compareFrom??run.from,to=compareTo??run.steps;
 assert.ok(Number.isInteger(from)&&from>0&&from<=to&&from>=run.from,"COMPARE_FROM must lie within the recorded measured window");
 assert.ok(Number.isInteger(to)&&to>=from&&to<=run.steps,"COMPARE_TO must lie within the recorded measured window");
 const rows=run.rows.filter((row:any)=>row.frame>=from&&row.frame<=to),walls=rows.map((row:any)=>row.wall_ms).sort((a:number,b:number)=>a-b),passes:Record<string,number>={};
 if(run.profiled)assert.ok(rows.every((row:any)=>row.passes),"missing per-frame pass timestamps");
 for(const row of rows)for(const pass of row.passes??[])passes[pass.label]=(passes[pass.label]??0)+pass.ms/rows.length;
 return {...run,originalCaptureFrom:run.from,originalCaptureSteps:run.steps,from,steps:to,summary:{...run.summary,mean_ms:walls.reduce((a:number,b:number)=>a+b,0)/walls.length,median_ms:walls[Math.floor(walls.length/2)],passMeans_ms:passes}};
}
const before=rewindow(JSON.parse(readFileSync(beforePath,"utf8"))),after=rewindow(JSON.parse(readFileSync(afterPath,"utf8")));
for(const key of ["scenario","steps","from","profiled","split","noBody"])assert.equal(after[key],before[key],`incompatible ${key}`);
assert.equal(Boolean(after.workCounters),Boolean(before.workCounters),"incompatible workCounters");
if(before.configuration&&after.configuration)assert.deepEqual(after.configuration,before.configuration,"incompatible scene/timestep/impact configuration");
const ratios:Record<string,unknown>={beforeSource:before.sourceHash,afterSource:after.sourceHash,scenario:after.scenario,
 window:[after.from,after.steps],originalCaptureWindow:compareFrom===undefined&&compareTo===undefined?undefined:{before:[before.originalCaptureFrom,before.originalCaptureSteps],after:[after.originalCaptureFrom,after.originalCaptureSteps]},
 mean_ms:{before:before.summary.mean_ms,after:after.summary.mean_ms,ratio:after.summary.mean_ms/before.summary.mean_ms},
 median_ms:{before:before.summary.median_ms,after:after.summary.median_ms,ratio:after.summary.median_ms/before.summary.median_ms},
 correctness:{before:before.summary.correctness,after:after.summary.correctness},volumeTelemetry:{before:before.summary.volumeTelemetry,after:after.summary.volumeTelemetry},workEvidence:after.summary.workEvidence};
if(before.summary.volumeTelemetry?.currentPhiVolume_m3!==undefined&&after.summary.volumeTelemetry?.currentPhiVolume_m3!==undefined){
 const reference=before.summary.volumeTelemetry.currentPhiVolume_m3,value=after.summary.volumeTelemetry.currentPhiVolume_m3;
 ratios.currentPhiVolumeDifference={before_m3:reference,after_m3:value,delta_m3:value-reference,relativeDifference:(value-reference)/reference};
}
const measured=after.rows.filter((row:{frame:number})=>row.frame>=after.from);
ratios.workShapeNotes="Region counters describe each frame’s final encode, not all geometry passes. surfaceVertices counts predicted/full foot evaluations, including deferred attempts.";
ratios.workShapeMeans=Object.fromEntries(["surfaceRegions","geometryRegions","surfaceVertices"].map(key=>{
 const samples=measured.map((row:{work?:Record<string,number[]>})=>row.work?.[key]).filter((row:number[]|undefined):row is number[]=>row!==undefined);
 return [key,samples.length?samples[0].map((_:number,axis:number)=>samples.reduce((sum:number,row:number[])=>sum+row[axis],0)/samples.length):undefined];
}));
const transportRows=measured.map((row:{work?:{surfaceTransport?:Record<string,number>}})=>row.work?.surfaceTransport).filter((row:Record<string,number>|undefined):row is Record<string,number>=>row!==undefined);
if(transportRows.length){
 const sum=(key:string)=>transportRows.reduce((total:number,row:Record<string,number>)=>total+(row[key]??0),0);
 const accepted=sum("acceptedEvaluations"),full=sum("fullEvaluations");
 ratios.transportWork={sampledFrames:transportRows.length,acceptedEvaluations:accepted,fullEvaluations:full,
  acceptedShare:accepted/Math.max(1,accepted+full),maxAcceptedErrorBound_m:Math.max(...transportRows.map((row:Record<string,number>)=>row.maxAcceptedErrorBound_m??0)),
  modelRejected:sum("modelRejected"),speedOrFootprintRejected:sum("speedOrFootprintRejected"),solidOrWallRejected:sum("solidOrWallRejected")};
}
const tier1Rows=measured.map((row:{work?:{surfaceTransportTier1?:Record<string,number>}})=>row.work?.surfaceTransportTier1).filter((row:Record<string,number>|undefined):row is Record<string,number>=>row!==undefined);
if(tier1Rows.length){
 const accepted=tier1Rows.reduce((total:number,row:Record<string,number>)=>total+(row.acceptedEvaluations??0),0),full=tier1Rows.reduce((total:number,row:Record<string,number>)=>total+(row.fullFallbackEvaluations??0),0);
 ratios.tier1TransportWork={sampledFrames:tier1Rows.length,acceptedEvaluations:accepted,fullFallbackEvaluations:full,acceptedShare:accepted/Math.max(1,accepted+full),maxAcceptedErrorBound_m:Math.max(...tier1Rows.map((row:Record<string,number>)=>row.maxAcceptedErrorBound_m??0))};
 ratios.transportTierNotes="Affine rejection can enter tier 1. Do not add affine fullEvaluations to tier-1 fullFallbackEvaluations or interpret affine rejection as a full trace. Counters count attempts, including deferred retries.";
}
const rejectionRows=measured.map((row:{work?:{surfaceTier1Rejections?:Record<string,number>}})=>row.work?.surfaceTier1Rejections).filter((row:Record<string,number>|undefined):row is Record<string,number>=>row!==undefined);
if(rejectionRows.length)ratios.tier1Rejections=Object.fromEntries(Object.keys(rejectionRows[0]).map(key=>[key,rejectionRows.reduce((total:number,row:Record<string,number>)=>total+(row[key]??0),0)]));
const stages=new Set([...Object.keys(before.summary.passMeans_ms),...Object.keys(after.summary.passMeans_ms)]);
ratios.passMeans_ms=Object.fromEntries([...stages].map(label=>[label,{before:before.summary.passMeans_ms[label]??0,after:after.summary.passMeans_ms[label]??0}]).sort((a,b)=>(b[1] as {before:number}).before-(a[1] as {before:number}).before));
if(beforeFields&&afterFields){
 const bm=JSON.parse(readFileSync(beforeFields+".json","utf8")),am=JSON.parse(readFileSync(afterFields+".json","utf8"));
 const beforeFieldFrame=bm.frame??before.originalCaptureSteps??before.steps,afterFieldFrame=am.frame??after.originalCaptureSteps??after.steps;
 assert.equal(beforeFieldFrame,afterFieldFrame,"field snapshots have different final frames");
 assert.equal(beforeFieldFrame,before.steps,"fields belong to the original final frame, not COMPARE_TO; omit fields for an early timing window");
 assert.equal(afterFieldFrame,after.steps,"fields belong to the original final frame, not COMPARE_TO; omit fields for an early timing window");
 ratios.fieldFrame=afterFieldFrame;
 for(const key of ["nx","ny","nz"])assert.equal(am[key],bm[key]);
 const segments=(metadata:Record<string,unknown>)=>metadata.segments as Record<string,number>|undefined??Object.fromEntries(Object.entries(metadata).filter(([key])=>!["nx","ny","nz","frame","authoritativeVelocity"].includes(key))) as Record<string,number>;
 const bs=segments(bm),as=segments(am);
 const b=readFileSync(beforeFields),a=readFileSync(afterFields);
 const bv=new Float32Array(b.buffer,b.byteOffset,b.length/4),av=new Float32Array(a.buffer,a.byteOffset,a.length/4);
 const offsets=(parts:Record<string,number>)=>{let at=0;return Object.fromEntries(Object.entries(parts).map(([name,count])=>{const offset=at;at+=count;return [name,offset];}));};
 const bo=offsets(bs),ao=offsets(as),fields:Record<string,unknown>={};
 for(const name of Object.keys(bs).filter(name=>name in as)){
  let maxAbs=0,squared=0,referenceSquared=0,nonfinite=0,changed=0;
  const count=bs[name]!;assert.equal(as[name],count,`incompatible ${name} size`);
  for(let i=0;i<count;i++){
   const reference=bv[bo[name]!+i]!,delta=av[ao[name]!+i]!-reference;
   if(!Number.isFinite(delta)){nonfinite++;continue;}
   maxAbs=Math.max(maxAbs,Math.abs(delta));squared+=delta*delta;referenceSquared+=reference*reference;if(delta!==0)changed++;
  }
  fields[name]={count,representation:name==="velocity"?"raw physical storage; ownership may differ":undefined,placeholder:name==="pressure"&&count===1?true:undefined,changed,nonfinite,maxAbs,rms:Math.sqrt(squared/count),relativeL2:Math.sqrt(squared/Math.max(referenceSquared,Number.MIN_VALUE))};
 }
 if(bs.phi&&as.phi){
  let count=0,maxAbs=0,squared=0;
  for(let i=0;i<bs.phi;i++)if(Math.abs(bv[bo.phi!+i]!)<=.1){const delta=av[ao.phi!+i]!-bv[bo.phi!+i]!;count++;maxAbs=Math.max(maxAbs,Math.abs(delta));squared+=delta*delta;}
  fields.interfacePhi={referenceBand_m:.1,count,maxAbs,rms:Math.sqrt(squared/Math.max(1,count))};
 }
 if(bs.velocity===4*bs.density!&&as.velocity===bs.velocity){
  let cells=0,maxVectorDelta=0,squared=0;
  for(let cell=0;cell<bs.density!;cell++)if(bv[bo.density!+cell]!>1e-5){
   let error=0;for(let axis=0;axis<3;axis++){const i=4*cell+axis,delta=av[ao.velocity!+i]!-bv[bo.velocity!+i]!;error+=delta*delta;}
   cells++;maxVectorDelta=Math.max(maxVectorDelta,Math.sqrt(error));squared+=error;
  }
  fields.liquidVelocityStorage={representation:"raw physical storage masked by baseline density; not authoritative mixed sampling",cells,maxVectorDelta_m_s:maxVectorDelta,rmsVectorDelta_m_s:Math.sqrt(squared/Math.max(1,cells))};
 }
 if(bs.authoritativeVelocity&&as.authoritativeVelocity){
  assert.deepEqual(am.authoritativeVelocity,bm.authoritativeVelocity,"authoritative velocity sample configuration differs");
  assert.deepEqual(fields.authoritativeVelocityPositions&&{changed:(fields.authoritativeVelocityPositions as {changed:number}).changed}, {changed:0},"common-world sample positions differ");
  const samples=bs.authoritativeVelocity/4;
  const vectorMetrics=(first:number,last:number,include?:(sample:number)=>boolean)=>{
   let included=0;let squared=0,maxVectorDelta=0,nonfinite=0,worstSample=first;
   for(let sample=first;sample<last;sample++){if(include&&!include(sample))continue;included++;let norm=0;for(let axis=0;axis<3;axis++){const delta=av[ao.authoritativeVelocity!+4*sample+axis]!-bv[bo.authoritativeVelocity!+4*sample+axis]!;if(!Number.isFinite(delta))nonfinite++;else norm+=delta*delta;}squared+=norm;if(Math.sqrt(norm)>maxVectorDelta){maxVectorDelta=Math.sqrt(norm);worstSample=sample;}}
   return {samples:included,nonfinite,maxVectorDelta_m_s:maxVectorDelta,rmsVectorDelta_m_s:Math.sqrt(squared/Math.max(1,included)),worstPosition_m:Array.from(bv.slice(bo.authoritativeVelocityPositions!+4*worstSample,bo.authoritativeVelocityPositions!+4*worstSample+3))};
  };
  const bodyRadius=before.configuration?.bodyRadius_m??createBackyardPoolPartyScene().rigidBodies[0]!.dimensions_m.x;
  const cellSize=before.configuration?.cellSize_m??[.05,.05,.05],container=before.configuration?.container??[16,8,12];
  const origin=before.configuration?.origin_m??[-container[0]/2,0,-container[2]/2];
  const centres=[before,after].map(run=>{const p=run.rows.at(-1)?.physical;return run.summary.finalBodyPosition_m??[p?.bodyX??run.configuration?.impact?.positionXZ_m?.[0]??0,p?.bodyY??Infinity,p?.bodyZ??run.configuration?.impact?.positionXZ_m?.[1]??0];});
  const outsideBody=(p:number[],clearance=0)=>before.noBody||centres.every(center=>Math.hypot(p[0]!-center[0]!,p[1]!-center[1]!,p[2]!-center[2]!)>bodyRadius+clearance);
  const phiAt=(p:number[],values:Float32Array,offset:number)=>{
   const q=[(p[0]!-origin[0])/cellSize[0],(p[1]!-origin[1])/cellSize[1],(p[2]!-origin[2])/cellSize[2]],dims=[bm.nx,bm.ny,bm.nz];
   const base=q.map((v,a)=>Math.max(0,Math.min(dims[a]!-1,Math.floor(v)))),t=q.map((v,a)=>Math.max(0,Math.min(1,v-base[a]!)));
   let result=0;for(let k=0;k<8;k++){const x=k&1,y=(k>>1)&1,z=(k>>2)&1;result+=values[offset+base[0]!+x+(bm.nx+1)*(base[1]!+y+(bm.ny+1)*(base[2]!+z))]!*(x?t[0]!:1-t[0]!)*(y?t[1]!:1-t[1]!)*(z?t[2]!:1-t[2]!);}return result;
  };
  const jointLiquid=(sample:number,clearance=0)=>{
   const p=Array.from(bv.slice(bo.authoritativeVelocityPositions!+4*sample,bo.authoritativeVelocityPositions!+4*sample+3));
   return outsideBody(p,clearance)&&phiAt(p,bv,bo.phi!)<0&&phiAt(p,av,ao.phi!)<0;
  };
  const bodyPositionNotes=centres.map((center,index)=>({center_m:center,horizontalPositionRecorded:[before,after][index].summary.finalBodyPosition_m!==undefined||[before,after][index].rows.at(-1)?.physical?.bodyX!==undefined}));
  fields.authoritativeLiquidVelocity={...vectorMetrics(0,samples,sample=>jointLiquid(sample)),mask:"trilinear resolved phi<0 in both captures; points outside union of final authored spheres",bodyRadius_m:bodyRadius,bodyPositionNotes};
  fields.authoritativeLiquidVelocityOneHClearance={...vectorMetrics(0,samples,sample=>jointLiquid(sample,Math.max(...cellSize))),mask:"joint liquid as above; at least one h outside union of final authored spheres",bodyRadius_m:bodyRadius,bodyPositionNotes};
  let outsideCount=0,outsideMax=0,outsideSquared=0,worstPhiPosition:number[]=[];
  for(let i=0;i<bs.phi!;i++)if(Math.abs(bv[bo.phi!+i]!)<=.1){
   const x=i%(bm.nx+1),y=Math.floor(i/(bm.nx+1))%(bm.ny+1),z=Math.floor(i/((bm.nx+1)*(bm.ny+1)));
   const p=[x*cellSize[0]+origin[0],y*cellSize[1]+origin[1],z*cellSize[2]+origin[2]];if(!outsideBody(p))continue;
   const delta=av[ao.phi!+i]!-bv[bo.phi!+i]!;outsideCount++;outsideSquared+=delta*delta;if(Math.abs(delta)>outsideMax){outsideMax=Math.abs(delta);worstPhiPosition=p;}
  }
  fields.interfacePhiOutsideBodies={referenceBand_m:.1,count:outsideCount,maxAbs:outsideMax,rms:Math.sqrt(outsideSquared/Math.max(1,outsideCount)),worstPosition_m:worstPhiPosition,bodyPositionNotes,mask:"baseline abs(phi)<=.1m; outside union of final authored spheres"};
  fields.authoritativeVelocityVectors={...vectorMetrics(0,samples),groups:{coarseLiquidBox:vectorMetrics(0,bm.authoritativeVelocity.coarseSamples),fineImpactSurface:vectorMetrics(bm.authoritativeVelocity.coarseSamples,samples)}};
  ratios.preferredVelocityMetric="authoritativeVelocityVectors: same world points through production mixed MAC interpolation";
 }
 ratios.fields=fields;
}
console.log(JSON.stringify(ratios,null,2));
