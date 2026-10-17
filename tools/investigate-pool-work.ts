/** Production-solver investigation. Overrides apply only in this probe process.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/investigate-pool-work.ts
 * PROFILE=1 adds pass timings; SPLIT=1 splits dispatches (diagnostic overhead).
 * NO_BODY=1 is a controlled comparison, not the authored scene.
 * SCENARIO=settled (default): 120 frames, measure 61–120, authored rigid body.
 * SCENARIO=impact: localized sphere impact, 30 frames, impulse/depth/crown checks.
 * SCENARIO=pool-impact: same authored pool geometry, ball immediately above water.
 * SCENARIO=drift: the gentle-drift pool (floating ball drifting slowly, one drip at 2 s); rows carry region tier counts.
 * IMPACT_SPEED=8 (downward m/s), IMPACT_X=0, IMPACT_Z=0 vary impact only.
 * GEO_SNAPSHOTS=0 disables geometry value snapshots (metadata-only ablation).
 * FROM filters wall and per-pass summaries to the same measured window.
 * STEPS/FROM override the window; OUT selects JSON output; FORCE_FULL=1 disables transport/geometry work control and exact crossing reuse; normal anchors remain active.
 * CURRENT_VOLUME=1 measures current phi mass after timing, without a time step.
 * FIELDS_OUT also enables current volume QA and saves final authoritative fields; WORK_COUNTERS=1 instruments vertex work (diagnostic overhead).
 * Run under /usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock to share the GPU.
 * BAND_TOLERANCE sets a probe-only band residual tolerance in 1/s.
 * Zero disables stopping to compare the same launches with full numerical work.
 */
import assert from "node:assert/strict";
import {samplePoolVelocity} from "./sample-pool-velocity";
import {measureCurrentPoolPhiVolume,type PoolVolumeFrame} from "./measure-pool-volume";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { createBackyardPoolDriftScene, createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { createUniformSolver, withUniformDevice, readUniformFields } from "../tests/helpers/uniform-geometric";
import { GPUPassProfile } from "./gpu-pass-profile";
import { NarrowBandDispatchProbe } from "./narrow-band-dispatch-probe";
import { readMixedBuffer, readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";
import { UNIFORM_PAGE_TABLE_WORDS, UNIFORM_PAGE_CELLS } from "../lib/methods/uniform/uniform-lattice-address";
import { UniformPressureBand } from "../lib/methods/uniform/uniform-pressure-band";
import type { UniformMixedFrameParameters } from "../lib/methods/uniform/uniform-mixed-frame";

const scenario=process.env.SCENARIO??"settled";
assert.ok(["settled","impact","pool-impact","drift"].includes(scenario));
const drift=scenario==="drift",impact=scenario!=="settled"&&!drift,miniImpact=scenario==="impact";
const impactSpeed=Number(process.env.IMPACT_SPEED??8),impactX=Number(process.env.IMPACT_X??0),impactZ=Number(process.env.IMPACT_Z??0);
assert.ok(Number.isFinite(impactSpeed)&&impactSpeed>=0&&Number.isFinite(impactX)&&Number.isFinite(impactZ));
async function readWords(device:GPUDevice,source:GPUBuffer,offset:number,count:number){
 const target=device.createBuffer({size:4*count,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try{const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(source,offset,target,0,4*count);device.queue.submit([encoder.finish()]);await target.mapAsync(GPUMapMode.READ);return new Uint32Array(target.getMappedRange()).slice();}finally{if(target.mapState==="mapped")target.unmap();target.destroy();}
}
interface ProbeRegionCounters {buffer:GPUBuffer;offset:number;regions:number;wordsPerRegion:number;labels:readonly string[]}
interface ProbeOwnership {pageMirror:{buffer:GPUBuffer;offset:number}}
interface ProbeFrame {origin_m:readonly number[];cellSize_m:readonly number[]}
async function regionCoordinates(device:GPUDevice,ownership:ProbeOwnership,regions:number,frame:ProbeFrame){
 const mirror=ownership.pageMirror;
 const [header,blocks]=await Promise.all([readWords(device,mirror.buffer,4*(mirror.offset+UNIFORM_PAGE_TABLE_WORDS-32),3),readWords(device,mirror.buffer,4*(mirror.offset+UNIFORM_PAGE_TABLE_WORDS),regions/8)]);
 const signed=new Int32Array(header.buffer);
 return Array.from({length:regions},(_,region)=>{
  const block=region>>>3,word=blocks[block]!;if(block===0||block===blocks.length-1||word===0xffffffff)return undefined;
  const page=[word&63,(word>>>6)&63,(word>>>12)&63].map((v,axis)=>(v-signed[axis]!)&63);
  const center=page.map((v,axis)=>frame.origin_m[axis]!+frame.cellSize_m[axis]!*(32*(v+signed[axis]!)+16*((region>>>axis)&1)+8));
  return {center_m:center,windowFramePage:Array.from(signed),packedPageWord:word,horizontalDistance_m:Math.hypot(center[0]!-impactX,center[2]!-impactZ)};
 });
}
async function readRegionWork(device:GPUDevice,spec:ProbeRegionCounters,ownership:ProbeOwnership,frame:ProbeFrame,control:{surface:GPUBuffer;geometry:GPUBuffer;regions:number;epochOffset?:number},previous:Uint32Array|undefined,normalSpec?:{curvatureDirty?:GPUBuffer;curvatureWorkOffset?:number}){
 const [words,coordinates,epochs,geometryDirty,normalDirty]=await Promise.all([readWords(device,spec.buffer,spec.offset,spec.regions*spec.wordsPerRegion),regionCoordinates(device,ownership,spec.regions,frame),control.epochOffset!==undefined?readWords(device,control.surface,control.epochOffset,control.regions):undefined,readWords(device,control.geometry,0,control.regions),normalSpec?.curvatureDirty&&normalSpec.curvatureWorkOffset!==undefined?readWords(device,normalSpec.curvatureDirty,normalSpec.curvatureWorkOffset,control.regions):undefined]);
 const limits=[1,2,4,8,Infinity],bins=limits.map((upper,i)=>({range_m:[i?limits[i-1]!:0,Number.isFinite(upper)?upper:"infinity"],storedRegions:0,regionsWithStores:0,surfaceRefreshed:epochs?0:undefined,geometryDirty:0,normalDirtyRegions:normalDirty?0:undefined,stores:Object.fromEntries(spec.labels.map(label=>[label,0]))}));
 for(let region=0;region<spec.regions;region++){
  const coordinate=coordinates[region];if(!coordinate)continue;
  const bin=bins.find((_,i)=>coordinate.horizontalDistance_m<limits[i]!)!;bin.storedRegions++;
  const values=Array.from(words.slice(spec.wordsPerRegion*region,spec.wordsPerRegion*(region+1)));
  if(values.some(value=>value>0))bin.regionsWithStores++;
  if(epochs&&epochs[region]!==previous?.[region])bin.surfaceRefreshed=(bin.surfaceRefreshed??0)+1;
  if(geometryDirty[region])bin.geometryDirty++;
  if(normalDirty?.[region])bin.normalDirtyRegions=(bin.normalDirtyRegions??0)+1;
  spec.labels.forEach((label,i)=>{bin.stores[label]!+=values[i]??0;});
 }
 return {epochs,records:coordinates.map((coordinate,region)=>coordinate?{region,...coordinate,stores:Array.from(words.slice(spec.wordsPerRegion*region,spec.wordsPerRegion*(region+1))),geometryDirty:Boolean(geometryDirty[region]),normalDirty:normalDirty?Boolean(normalDirty[region]):undefined}:undefined).filter(record=>record!==undefined),bins:bins.map(bin=>{const cheap=(bin.stores.affine??0)+(bin.stores.middle??0),full=(bin.stores.fullLinear??0)+(bin.stores.fullCubic??0);return {...bin,transportStores:cheap+full,cheapTransportShare:cheap/Math.max(1,cheap+full)};})};
}
interface ProbeSurface {
 claims:GPUBuffer;workCountersOffset:number;workCounterWords?:number;regionCounters?:ProbeRegionCounters;allocatedBytes:number;evidenceBytes:number;predictorFineCapacity:number;predictorCacheBytes:number;
}
async function readSurfaceCounters(device:GPUDevice,surface:ProbeSurface|undefined){
 if(process.env.WORK_COUNTERS!=="1"||surface?.workCountersOffset===undefined)return undefined;
 const words=await readWords(device,surface.claims,surface.workCountersOffset,surface.workCounterWords??2);
 return {rawCounters:Array.from(words),earlyFarAcceptedEvaluations:words[0],farCellAcceptedEvaluations:words[1],footEvaluations:words.length<20?Array.from(words.slice(0,2)):undefined,wholeTransport:words.length>=8?{
  acceptedEvaluations:words[2],fullEvaluations:words[3],maxAcceptedErrorBound_m:new Float32Array(words.buffer)[4],
  modelRejected:words[5],speedOrFootprintRejected:words[6],solidOrWallRejected:words[7],
 }:undefined,tier1Transport:words.length>=12?{
  acceptedEvaluations:words[8],fullFallbackEvaluations:words[9],maxAcceptedErrorBound_m:new Float32Array(words.buffer)[10],farBandAcceptedEvaluations:words[11],
 }:undefined,tier1Rejections:words.length>=20?{
  canonicalOrGlobalEligibility:words[12],modelOrCoverage:words[13],solidOrProjection:words[14],velocityCacheUnavailable:words[15],
  speedIntervalOrGradientBound:words[16],legacyPositionRoundingBound:words[17],generalErrorBound:words[18],walkOrReleasedWall:words[19],
 }:undefined};
}
const steps=Number(process.env.STEPS??(impact?30:120)),from=Number(process.env.FROM??(impact?1:61));
assert.ok(Number.isInteger(steps)&&Number.isInteger(from)&&from>0&&steps>=from);
const profiled=process.env.PROFILE==="1",split=process.env.SPLIT==="1",noBody=process.env.NO_BODY==="1";
assert.ok(!split||profiled);
const out=process.env.OUT??"/tmp/pool-work.json";
const bandTolerance=process.env.BAND_TOLERANCE===undefined?undefined:Number(process.env.BAND_TOLERANCE);
if(bandTolerance!==undefined)assert.ok(Number.isFinite(bandTolerance)&&bandTolerance>=0);
const hash=createHash("sha256");
for(const file of readdirSync("lib/methods/uniform").filter(f=>f.endsWith(".ts")).sort()){
 hash.update(file);hash.update(readFileSync(`lib/methods/uniform/${file}`));
}
const sourceHash=hash.digest("hex");
const probeHash=createHash("sha256");for(const file of ["tools/investigate-pool-work.ts","tools/sample-pool-velocity.ts","tools/gpu-pass-profile.ts","tools/narrow-band-dispatch-probe.ts","tools/measure-pool-volume.ts"]){probeHash.update(file);probeHash.update(readFileSync(file));}
const probeSourceHash=probeHash.digest("hex");
console.log(JSON.stringify({phase:"starting",sourceHash,scenario,steps,from,profiled,workCounters:process.env.WORK_COUNTERS==="1"}));
paceDawnEventPolling();
await withUniformDevice("pool work investigation",async raw=>{
 const profile=profiled?new GPUPassProfile(raw):undefined;
 const probe=split?new NarrowBandDispatchProbe(profile!.device):undefined;
 const scene=drift?createBackyardPoolDriftScene():createBackyardPoolPartyScene();
 if(miniImpact){
  scene.container={...scene.container,width_m:3.2,height_m:4.8,depth_m:3.2};scene.solidBoundary=undefined;
  scene.solidVoxels=[[[0,0,0],[64,4,64]],[[0,0,0],[8,40,64]],[[56,0,0],[64,40,64]],[[8,0,0],[56,40,8]],[[8,0,56],[56,40,64]]].map(([minimum,maximumExclusive])=>({operation:"fill",minimum:minimum as [number,number,number],maximumExclusive:maximumExclusive as [number,number,number],materialId:1}));
  scene.fluid.initialLiquidVolumes=[{shape:"box",min_m:{x:-1.2,y:.2,z:-1.2},max_m:{x:1.2,y:1.8,z:1.2}}];
  Object.assign(scene.rigidBodies[0]!,{position_m:{x:impactX,y:2.2,z:impactZ},linearVelocity_m_s:{x:0,y:-impactSpeed,z:0}});
 }
 const liquid=scene.fluid.initialLiquidVolumes?.find(volume=>volume.shape==="box");
 assert.ok(liquid?.shape==="box","pool probe requires its authored box of water");
 const waterline=liquid.max_m.y,initialBallY=waterline+.4;
 if(scenario==="pool-impact")Object.assign(scene.rigidBodies[0]!,{position_m:{x:impactX,y:initialBallY,z:impactZ},linearVelocity_m_s:{x:0,y:-impactSpeed,z:0}});
 if(noBody)scene.rigidBodies=[];
 const sceneGeometryHash=createHash("sha256").update(JSON.stringify({...scene,rigidBodies:scene.rigidBodies.map(({position_m:_position,linearVelocity_m_s:_velocity,...body})=>body)})).digest("hex");
 assert.ok(steps*scene.numerics.fixedDt_s<=scene.duration_s);
 const bodies=initializeRigidBodies(scene.rigidBodies),start=performance.now();
 const solver=await createUniformSolver(probe?.device??profile?.device??raw,scene);
 const frameState=(solver as unknown as {mixedFrame:{words:{band:Float32Array<ArrayBuffer>};band:{index:GPUBuffer};bandParams:GPUBuffer;write(p:UniformMixedFrameParameters,dt:number,gravity:number):boolean}}).mixedFrame;
 if(bandTolerance!==undefined){
  const write=frameState.write.bind(frameState);
  frameState.write=(p,dt,gravity)=>{
   const same=write(p,dt,gravity);frameState.words.band[8]=bandTolerance;
   raw.queue.writeBuffer(frameState.bandParams,0,frameState.words.band);return same;
  };
 }
 const inner=solver as unknown as {mixedFrame:{surfaceWorkControl?:{enabled:boolean;surface:GPUBuffer;geometry:GPUBuffer;countersOffset:number;regions:number;epochOffset?:number;cacheRegions:number;allocatedBytes:number;modelWordOffset?:number;velocityWordOffset?:number;collectGeometryStats?:boolean;geometrySnapshots?:boolean};narrowBandFlip?:{count:number;regionalCrossingReuseEnabled?:boolean;regionalReuseCapacity?:number;regionalReuseAllocatedBytes?:number;regionalReuseEvidence?:{buffer:GPUBuffer;regions:number;flagsOffset:number;planeOffset:number}};ownership:ProbeOwnership;heat?:{counts:{buffer:GPUBuffer;offset:number}};surface?:ProbeSurface;geometry?:{countersBuffer:GPUBuffer;countersOffset:number;allocatedBytes?:number;curvatureDirty?:GPUBuffer;curvatureCountersOffset:number;curvatureWorkOffset?:number}};rigidSystem:{stateBuffer:GPUBuffer};rigidExchange:GPUBuffer};
 if(process.env.FORCE_FULL==="1"){
  if(inner.mixedFrame.surfaceWorkControl)inner.mixedFrame.surfaceWorkControl.enabled=false;
  if(inner.mixedFrame.narrowBandFlip)inner.mixedFrame.narrowBandFlip.regionalCrossingReuseEnabled=false;
 }
 if(process.env.GEO_SNAPSHOTS==="0"&&inner.mixedFrame.surfaceWorkControl)inner.mixedFrame.surfaceWorkControl.geometrySnapshots=false;
 let previousVelocity=[0,-impactSpeed,0],impulseY=0,deepest=initialBallY,maxMomentumError=0;
 if(process.env.WORK_COUNTERS==="1"){
  if(inner.mixedFrame.surfaceWorkControl)inner.mixedFrame.surfaceWorkControl.collectGeometryStats=true;
  const diagnosticFrame=frameState as unknown as {params:{surface:GPUBuffer};write:typeof frameState.write};
  const write=diagnosticFrame.write.bind(diagnosticFrame);
  diagnosticFrame.write=(p,dt,gravity)=>{const same=write(p,dt,gravity);raw.queue.writeBuffer(diagnosticFrame.params.surface,24,new Uint32Array([1]));return same;};
 }
 const construction_ms=performance.now()-start;
 console.log(JSON.stringify({phase:"constructed",construction_ms,profiled,split,noBody,sourceHash,regionalReuseCapacity:inner.mixedFrame.narrowBandFlip?.regionalReuseCapacity,regionalReuseAllocatedBytes:inner.mixedFrame.narrowBandFlip?.regionalReuseAllocatedBytes}));
 const rows=[];let finalRegionRecords:unknown;let previousSurfaceEpochs:Uint32Array|undefined;
 try{
  for(let frame=1;frame<=steps;frame++){
   if(probe)probe.enabled=frame>=from;
   if(profile&&frame>=from)profile.start();
   const start=performance.now();
   assert.ok(solver.advanceTo(frame*scene.numerics.fixedDt_s,bodies));
   const encode_ms=performance.now()-start;
   await solver.awaitFrameCompletion();
   const wall_ms=performance.now()-start;
   assert.equal(solver.info.simulationPipelineError,undefined);
   const passes=profile&&frame>=from?await profile.finish():undefined;
   let physical:Record<string,number>|undefined;
   if(impact&&bodies.length){
    const state=await readMixedBuffer(raw,inner.rigidSystem.stateBuffer),exchange=new Int32Array((await readMixedBuffer(raw,inner.rigidExchange)).buffer);
    for(let axis=0;axis<3;axis++){
     const gravity=axis===1?scene.fluid.gravity_m_s2.y:0;
     maxMomentumError=Math.max(maxMomentumError,Math.abs(bodies[0]!.mass_kg*(state[12+axis]!-previousVelocity[axis]!-gravity*scene.numerics.fixedDt_s)-exchange[axis]!*1e-6));
    }
    previousVelocity=Array.from(state.slice(12,15));impulseY+=exchange[1]!*1e-6;deepest=Math.min(deepest,state[1]!);
    physical={bodyX:state[0]!,bodyZ:state[2]!,bodyY:state[1]!,bodyVy:state[13]!,impulseY,deepest,maxMomentumError};
   }
   const surfaceStage=inner.mixedFrame.surface,workControl=inner.mixedFrame.surfaceWorkControl;
   const surfaceCounts=await readSurfaceCounters(raw,surfaceStage);
   let regionWork:Awaited<ReturnType<typeof readRegionWork>>|undefined;
   if(process.env.WORK_COUNTERS==="1"&&surfaceStage?.regionCounters&&workControl){
    regionWork=await readRegionWork(raw,surfaceStage.regionCounters,inner.mixedFrame.ownership,(solver as unknown as {frame:ProbeFrame}).frame,workControl,previousSurfaceEpochs,inner.mixedFrame.geometry);previousSurfaceEpochs=regionWork.epochs;finalRegionRecords=regionWork.records;
   }
   const work=process.env.WORK_COUNTERS==="1"&&workControl?{
    surfaceRegions:Array.from(await readWords(raw,workControl.surface,workControl.countersOffset,4)),
    geometryRegions:Array.from(await readWords(raw,workControl.geometry,workControl.countersOffset,4)),
    regionsByHorizontalDistance:regionWork?.bins,
    surfaceVertices:surfaceCounts?.footEvaluations,surfaceTransport:surfaceCounts?.wholeTransport,surfaceTransportTier1:surfaceCounts?.tier1Transport,surfaceTier1Rejections:surfaceCounts?.tier1Rejections,surfaceRawCounters:surfaceCounts?.rawCounters,
    normalDirtyRegions:inner.mixedFrame.geometry?.curvatureDirty?(await readWords(raw,inner.mixedFrame.geometry.curvatureDirty,inner.mixedFrame.geometry.curvatureCountersOffset,1))[0]:undefined,
    geometryEvaluations:inner.mixedFrame.geometry?.countersBuffer?Array.from(await readWords(raw,inner.mixedFrame.geometry.countersBuffer,inner.mixedFrame.geometry.countersOffset,2)):undefined,
   }:undefined;
   const heatCounts=inner.mixedFrame.heat?.counts,tiers=heatCounts?Array.from(await readWords(raw,heatCounts.buffer,heatCounts.offset,4)):undefined;
   const row={frame,wall_ms,encode_ms,tiers,dispatches:passes?.reduce((n,p)=>n+p.dispatches,0),physical,work,...solver.narrowBandFlipInfo,...solver.workDomainInfo,
    bandTiles:solver.info.uniformPressureBandTiles,bandCycles:solver.info.uniformPressureBandCycles,
    bandResidual:solver.info.uniformPressureBandResidual,rootResidual:solver.info.uniformPressureAcceptedResidual,passes};
   rows.push(row);
   if(frame===1||frame%12===0||frame===steps)console.log(JSON.stringify({...row,passes:undefined}));
  }
  const measured=rows.filter(r=>r.frame>=from),passMeans:Record<string,number>={};
  for(const row of measured)for(const p of row.passes??[])passMeans[p.label]=(passMeans[p.label]??0)+p.ms/measured.length;
  const walls=measured.map(r=>r.wall_ms).sort((a,b)=>a-b);
  const warmedWalls=measured.filter(r=>r.frame>1).map(r=>r.wall_ms).sort((a,b)=>a-b);
  const currentVolumeMeasured=process.env.CURRENT_VOLUME==="1"||Boolean(process.env.FIELDS_OUT);
  if(currentVolumeMeasured)measureCurrentPoolPhiVolume(raw,inner.mixedFrame as unknown as PoolVolumeFrame);
  const stats=await solver.readStats();
  if(currentVolumeMeasured){assert.equal(stats.volumeTelemetrySource,"narrow-band-phi","Current phi volume probe did not publish its measurement");assert.ok(stats.volumeCellSum!==undefined&&Number.isFinite(stats.volumeCellSum),"Current phi volume probe returned no finite volume");}
  const cellVolume=(solver as unknown as {frame:ProbeFrame}).frame.cellSize_m.reduce((product,value)=>product*value,1);
  const volumeTelemetry={source:stats.volumeTelemetrySource,phiMeasurementFrame:currentVolumeMeasured?steps:undefined,
   phiMeasurementMode:currentVolumeMeasured?"post-run current phi, no time advance":"last scheduled phi probe, potentially stale",
   currentPhiVolume_m3:currentVolumeMeasured?stats.volumeCellSum!*cellVolume:undefined,
   initialBudgetVolume_m3:stats.narrowBandInitialVolume_cells===undefined?undefined:stats.narrowBandInitialVolume_cells*cellVolume,
   rawFillDrift:stats.rawVolumeDrift,representedVolumeDrift:stats.representedVolumeDrift,reportedVolumeDrift:stats.volumeDrift};
  const finalBodyPosition_m=bodies.length?Array.from(new Float32Array((await readWords(raw,inner.rigidSystem.stateBuffer,0,3)).buffer)):undefined;
  if(process.env.FIELDS_OUT){
   const fields=await readUniformFields(raw,solver);
   const resolvedPhi=new Float32Array(fields.phi.length),{nx,ny,nz}=solver.info;
   for(let z=0;z<=nz;z++)for(let y=0;y<=ny;y++)for(let x=0;x<=nx;x++)resolvedPhi[x+(nx+1)*(y+(ny+1)*z)]=fields.vertex(x,y,z);
   const velocity=await readMixedTexture(raw,solver.velocityTexture);
   const geometry=(solver as unknown as {mixedPressureGeometry?:{target:GPUTexture;centerPhi:GPUTexture}}).mixedPressureGeometry;
   const mixed=solver as unknown as {mixedFrame:{ownership:{presentation:GPUBufferBinding};coarseVertexPhi:GPUTexture};frame:ProbeFrame;vertexPhiTexture:GPUTexture};
   const authoritative=await samplePoolVelocity(raw,solver,mixed.mixedFrame.ownership.presentation,solver.vertexPhiTexture!,mixed.mixedFrame.coarseVertexPhi,mixed.frame.origin_m,mixed.frame.cellSize_m,{minimum:[liquid.min_m.x,liquid.min_m.y,liquid.min_m.z],maximum:[liquid.max_m.x,liquid.max_m.y,liquid.max_m.z]},[impactX,impactZ]);
   const segments:Record<string,Float32Array>={authoritativeVelocity:authoritative.values,authoritativeVelocityPositions:authoritative.positions,phi:resolvedPhi,pressure:fields.pressure,density:fields.density,velocity,
    negativeVelocity:await readMixedBuffer(raw,solver.negativeBoundaryVelocityBuffer)};
   if(geometry){segments.geometryTarget=await readMixedTexture(raw,geometry.target);segments.geometryCenterPhi=await readMixedTexture(raw,geometry.centerPhi);}
   const packed=new Float32Array(Object.values(segments).reduce((sum,field)=>sum+field.length,0));let offset=0;
   for(const field of Object.values(segments)){packed.set(field,offset);offset+=field.length;}
   mkdirSync(dirname(process.env.FIELDS_OUT),{recursive:true});
   writeFileSync(process.env.FIELDS_OUT,new Uint8Array(packed.buffer));
   writeFileSync(process.env.FIELDS_OUT+".json",JSON.stringify({frame:steps,authoritativeVelocity:authoritative.metadata,segments:Object.fromEntries(Object.entries(segments).map(([name,field])=>[name,field.length])),nx:solver.info.nx,ny:solver.info.ny,nz:solver.info.nz}));
  }
  const bandIndex=await readMixedBuffer(raw,frameState.band.index);
  let correctness:Record<string,number>|undefined;
  if(impact&&bodies.length){
   const fields=await readUniformFields(raw,solver);let crest=waterline;
   const h=scene.nominalResolution.length_m;
   const x0=Math.max(0,Math.round((liquid.min_m.x+scene.container.width_m/2+.2)/h)),x1=Math.min(solver.info.nx,Math.round((liquid.max_m.x+scene.container.width_m/2-.2)/h));
   const z0=Math.max(0,Math.round((liquid.min_m.z+scene.container.depth_m/2+.2)/h)),z1=Math.min(solver.info.nz,Math.round((liquid.max_m.z+scene.container.depth_m/2-.2)/h));
   const y0=Math.floor(waterline/h)+1,y1=Math.min(solver.info.ny,Math.ceil((waterline+1)/h));
   for(let z=z0;z<z1;z++)for(let x=x0;x<x1;x++)for(let y=y0;y<y1;y++)if(fields.vertex(x,y,z)<-.1*h)crest=Math.max(crest,y*h);
   correctness={waterline,impulseY,deepest,crestAbovePool:crest-waterline,maxMomentumError,particles:inner.mixedFrame.narrowBandFlip?.count??0};
   assert.ok(maxMomentumError<.002,`body momentum disagrees with pressure impulse by ${maxMomentumError} N s`);
   if(miniImpact&&steps>=30&&impactSpeed===8&&impactX===0&&impactZ===0){
    assert.ok(maxMomentumError<.002,`body momentum disagrees with pressure impulse by ${maxMomentumError} N s`);
    assert.ok(impulseY>200,`impact reaction too small: ${impulseY} N s`);
    assert.ok(deepest<1.37,`ball did not penetrate: ${deepest} m`);
    assert.ok(crest>waterline+.3,`impact crown too low: ${crest} m`);
    assert.ok(correctness.particles!>0,"impact must create detailed surface particles");
   }
  }
  const control=inner.mixedFrame.surfaceWorkControl;
  const workEvidence=control?await Promise.all([control.surface,control.geometry].map(async buffer=>{
   const words=await readWords(raw,buffer,control.countersOffset,4);
   return {regions:control.regions,counters:Array.from(words)};
  })):undefined;
  const surface=inner.mixedFrame.surface;
  const finalSurfaceCounts=await readSurfaceCounters(raw,surface);
  const surfaceVertices=finalSurfaceCounts?.footEvaluations;
  let regionModels:Record<string,unknown>|undefined;
  if(process.env.MODEL_DIAGNOSTICS==="1"&&control?.modelWordOffset!==undefined&&control.velocityWordOffset!==undefined){
   const modelWords=await readWords(raw,control.surface,4*control.modelWordOffset,8*control.regions),models=new Float32Array(modelWords.buffer);
   const velocityWords=await readWords(raw,control.surface,4*control.velocityWordOffset,6*control.regions),velocity=new Float32Array(velocityWords.buffer);
   const records:Array<{region:number;g:number[];centerPhi_m:number;rho_m:number;haloValid:boolean;coverageMask:number;localRho_m:number;velocityMin_m_s:number[];velocityMax_m_s:number[]}>=[];
   for(let region=0;region<control.regions;region++)if(modelWords[8*region+7])records.push({region,g:Array.from(models.slice(8*region,8*region+3)),centerPhi_m:models[8*region+3]!,rho_m:models[8*region+4]!,haloValid:Boolean(modelWords[8*region+5]),coverageMask:modelWords[8*region+5]!,localRho_m:models[8*region+6]!,velocityMin_m_s:Array.from(velocity.slice(6*region,6*region+3)),velocityMax_m_s:Array.from(velocity.slice(6*region+3,6*region+6))});
   const bounds=[0,1e-6,1e-5,1e-4,1e-3,1e-2,.1,1,Infinity];
   const histogram=(key:"rho_m"|"localRho_m")=>bounds.map((limit,i)=>({upper_m:Number.isFinite(limit)?limit:"infinity",count:records.filter(r=>r[key]!<=limit&&(i===0||r[key]!>bounds[i-1]!)).length}));
   regionModels={records:records.length,haloValid:records.filter(r=>r.haloValid).length,full27RegionCoverage:records.filter(r=>r.coverageMask===0x7ffffff).length,rhoHistogram:histogram("rho_m"),localRhoHistogram:histogram("localRho_m"),snapshot:out+".models.json"};
   mkdirSync(dirname(out),{recursive:true});writeFileSync(out+".models.json",JSON.stringify({sourceHash,frame:steps,note:"Current locally valid model records; coverageMask describes certified neighbors in the current subset ABI. haloValid means nonempty coverage, not necessarily all 27 neighbors. Snapshot is the final surface encode.",records},null,2)+"\n");
  }
  if(finalRegionRecords&&process.env.WORK_COUNTERS==="1"){mkdirSync(dirname(out),{recursive:true});writeFileSync(out+".regions.json",JSON.stringify({sourceHash,frame:steps,labels:inner.mixedFrame.surface?.regionCounters?.labels,records:finalRegionRecords},null,2)+"\n");}
  const geometry=inner.mixedFrame.geometry;
  const geometryEvaluations=process.env.WORK_COUNTERS==="1"&&geometry?.countersBuffer?Array.from(await readWords(raw,geometry.countersBuffer,geometry.countersOffset,2)):undefined;
  const normalDirtyRegions=process.env.WORK_COUNTERS==="1"&&geometry?.curvatureDirty?(await readWords(raw,geometry.curvatureDirty,geometry.curvatureCountersOffset,1))[0]:undefined;
  const evidence=inner.mixedFrame.narrowBandFlip?.regionalReuseEvidence;
  let regionalReuse:unknown;
  if(evidence&&evidence.regions>0){
   const [flags,planes]=await Promise.all([readWords(raw,evidence.buffer,evidence.flagsOffset,evidence.regions),readWords(raw,evidence.buffer,evidence.planeOffset,8*evidence.regions)]);
   regionalReuse={capacity:evidence.regions,allocatedBytes:inner.mixedFrame.narrowBandFlip?.regionalReuseAllocatedBytes,nearestRegions:Array.from(flags).filter(x=>(x&1)!==0).length,exactMetricRegions:Array.from(flags).filter(x=>(x&2)!==0).length,planeRegions:Array.from({length:evidence.regions},(_,i)=>planes[8*i+7]).filter(Boolean).length,fullyCoveredPlaneRegions:Array.from({length:evidence.regions},(_,i)=>planes[8*i+7]&&planes[8*i+5]===0xffffffff&&planes[8*i+6]===0xffffffff).filter(Boolean).length};
  }
  const summary={regionalReuse,correctness,volumeTelemetry,finalBodyPosition_m,regionModels,geometryEvaluations,normalDirtyRegions,geometryAllocatedBytes:geometry?.allocatedBytes,workEvidence,surfaceVertices,surfaceTransport:finalSurfaceCounts?.wholeTransport,surfaceTransportTier1:finalSurfaceCounts?.tier1Transport,surfaceTier1Rejections:finalSurfaceCounts?.tier1Rejections,surfaceRawCounters:finalSurfaceCounts?.rawCounters,surfaceEarlyFarAccepted:finalSurfaceCounts?.earlyFarAcceptedEvaluations,surfaceFarCellAccepted:finalSurfaceCounts?.farCellAcceptedEvaluations,surfaceAllocatedBytes:surface?.allocatedBytes,surfaceCacheBytes:surface?.predictorCacheBytes,predictorFineCapacity:surface?.predictorFineCapacity,workControlAllocatedBytes:control?.allocatedBytes,cacheRegions:control?.cacheRegions,mean_ms:walls.reduce((a,b)=>a+b,0)/walls.length,median_ms:walls[Math.floor(walls.length/2)],excludingFirstFrameMean_ms:warmedWalls.length?warmedWalls.reduce((a,b)=>a+b,0)/warmedWalls.length:undefined,excludingFirstFrameMedian_ms:warmedWalls[Math.floor(warmedWalls.length/2)],
   passMeans_ms:Object.fromEntries(Object.entries(passMeans).sort((a,b)=>b[1]-a[1])),
   maxSpeed_m_s:stats.maxSpeed_m_s,volumeDrift:stats.volumeDrift,
   bandTolerance:frameState.words.band[8],bandResidualHistory:Array.from(bandIndex.slice(8,13)),
   bandInitialResidual:bandIndex[UniformPressureBand.initialResidualWord],
   bandResidualValidity:new Uint32Array(bandIndex.buffer)[UniformPressureBand.residualValidityWord]};
  mkdirSync(dirname(out),{recursive:true});
  writeFileSync(out,JSON.stringify({date:new Date().toISOString(),sourceHash,probeSourceHash,scenario,workCounters:process.env.WORK_COUNTERS==="1",forceFull:process.env.FORCE_FULL==="1",forceFullScope:process.env.FORCE_FULL==="1"?"transport/geometry/crossing diagnostic; retained normal anchors remain enabled":undefined,profiled,split,noBody,steps,from,bandToleranceOverride:bandTolerance,
   configuration:{sceneGeometryHash,waterline_m:waterline,bodyRadius_m:scene.rigidBodies[0]?.shape==="sphere"?scene.rigidBodies[0].dimensions_m.x:undefined,origin_m:(solver as unknown as {frame:ProbeFrame}).frame.origin_m,geometrySnapshots:process.env.GEO_SNAPSHOTS!=="0",impact:impact?{speed_m_s:impactSpeed,positionXZ_m:[impactX,impactZ]}:undefined,fixedDt_s:scene.numerics.fixedDt_s,cellSize_m:(solver as unknown as {frame:ProbeFrame}).frame.cellSize_m,container:[scene.container.width_m,scene.container.height_m,scene.container.depth_m],pagingEdge_h:UNIFORM_PAGE_CELLS,workRegionEdge_h:16},
   scope:"Balanced production method, scene timestep, app defaults; simulation only; no renderer. Split timings include pass-splitting overhead.",
   construction_ms,summary,rows},null,2)+"\n");
  console.log(JSON.stringify({phase:"complete",out,summary}));
 }finally{solver.destroy();profile?.destroy();}
},profiled?["timestamp-query"]:[]);
