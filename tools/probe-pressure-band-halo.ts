/** Frozen live entire-band solve, with and without the packed no-wall
 * guard. Both variants preserve workgroup sizes, scheduling and arithmetic.
 * node --import tsx tools/probe-pressure-band-halo.ts [--steps=120]
 * [--rounds=8] [--scene=nbflip-figure-9-teaser] [--out=FILE]
 * [--indexed-neighbors] uses fixed color-row offsets and six direct faces.
 * [--middle-indexed] uses fixed binary offsets for 2h neighbours.
 * [--cached-neumann] retains projection boundary values in pressure rows.
 * [--cached-band-solid] replays row assembly with cached solid-open predicates.
 * [--packed-band-forces] shares the row's three positive velocity loads.
 * [--cached-band-faces] reuses exact cached fractions on tile boundary faces.
 * [--cached-band-faces-internal] checks that cache after the clear-tile fast exit.
 * [--coarse-weights] keeps every held 4h row's coefficients local.
 * [--coarse-wide] distributes each 4h half-sweep across workgroups.
 * [--coarse-grid-divisor=128] sizes the wide grid for one colour's rows.
 * [--coarse-grid-divisors=128,256,512] compares several widths on one snapshot.
 * [--lazy-coordinates] compares the guard with wall-only coordinates.
 */
import assert from "node:assert/strict";
import { existsSync,mkdirSync,writeFileSync } from "node:fs";
import { dirname,resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField } from "../lib/methods/uniform/uniform-detail-fields";
import type { UniformPressureBandFields } from "../lib/methods/uniform/uniform-pressure-band";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";
import { pressureNeumannCache } from "./pressure-neumann-cache";

function withoutHaloGuard(code:string):string{
 const guard=" if(((kinds>>1u)&~kinds&~(kinds>>2u)&0x9249u)==0u){return;}";
 assert.ok(code.includes(guard),"Expected packed wall guard");return code.replace(guard,"");
}
const lazyCoordinates=process.argv.includes("--lazy-coordinates");
function lazyWallCoordinates(code:string):string{
 const start=code.indexOf("fn bOffNear(");const end=code.indexOf("fn bResidualNear(",start);
 assert.ok(start>=0&&end>start,"Expected local band neighbour helper");
 const original=code.slice(start,end),replacement=original
  .replace("l:vec3u,p:vec3i,kinds:u32","l:vec3u,kinds:u32")
  .replace("else if(kind==K_WALL){off+=","else if(kind==K_WALL){let p=vec3i(bTile(s)*4u+l);off+=");
 assert.notEqual(original,replacement);code=code.slice(0,start)+replacement+code.slice(end);
 code=code.replace("fn bResidualNear(cell:u32,s:u32,l:vec3u,p:vec3i,kinds:u32)","fn bResidualNear(cell:u32,s:u32,l:vec3u,kinds:u32)")
  .replaceAll("bOffNear(cell,s,l,p,kinds)","bOffNear(cell,s,l,kinds)")
  .replaceAll("bResidualNear(cell,s,l,p,kinds)","bResidualNear(cell,s,l,kinds)")
  .replaceAll("bResidualNear(cell,s,l,vec3i(bTile(s)*4u+l),kinds)","bResidualNear(cell,s,l,kinds)")
  .replaceAll("let p=vec3i(bTile(s)*4u+l);let cell=","let cell=")
  .replaceAll("bFollowHalo(cell,p,kinds,next)","bFollowHaloNear(cell,s,l,kinds,next)");
 return code+`
fn bFollowHaloNear(cell:u32,s:u32,l:vec3u,kinds:u32,value:f32){
 if(((kinds>>1u)&~kinds&~(kinds>>2u)&0x9249u)==0u){return;}
 bFollowHalo(cell,vec3i(bTile(s)*4u+l),kinds,value);
}
`;
}
const indexedNeighbors=process.argv.includes("--indexed-neighbors");
const middleIndexed=process.argv.includes("--middle-indexed");
const cachedNeumann=process.argv.includes("--cached-neumann");
const cachedBandSolid=process.argv.includes("--cached-band-solid");
const packedBandForces=process.argv.includes("--packed-band-forces");
const cachedBandFaces=process.argv.includes("--cached-band-faces");
const cachedBandFacesInternal=process.argv.includes("--cached-band-faces-internal");
const rowAssembly=cachedBandSolid||packedBandForces||cachedBandFaces||cachedBandFacesInternal;
function cachedBandSolidSource(code:string):string{
 let count=0;
 const result=code.replace(/umCellOpen\(([pq])\)<=1e-5/g,(_,cell:string)=>{count++;return `!umCellIsOpen(${cell})`;});
 assert.ok(count>=2,"Expected threshold-only solid predicates in pressure row assembly");
 return result;
}
function rowAssemblySource(code:string):string{
 if(cachedBandSolid)code=cachedBandSolidSource(code);
 if(cachedBandFaces){
  const needle="let volume=umPressureFaceV(low,axis);";
  assert.equal(code.split(needle).length,2);
  code=code.replace(needle,`var volume=0.0;
   if(umSolidEnabled()&&low[axis]>=0&&(u32(low[axis])&3u)==3u){volume=umPatchFaceV(umTileAt(vec3u(low)/4u),low,axis);}
   else{volume=umPressureFaceV(low,axis);}`);
 }
 if(cachedBandFacesInternal){
  const start=code.indexOf("fn umPressureFaceV(id:vec3i,axis:u32)->f32{");
  assert.ok(start>=0);
  const prefix=code.slice(0,start),body=code.slice(start);
  const needle="if(umSolidValid(id)&&umSolidTileClear(id)){return 1.0;}";
  assert.equal(body.split(needle).length,2);
  code=prefix+body.replace(needle,needle+`
 if(umSolidValid(id)&&(u32(id[axis])&3u)==3u){
  let t=umTileAt(vec3u(id)/4u);let l=vec3u(id)&vec3u(3u);
  let slot=16u*axis+l[(axis+1u)%3u]+4u*l[(axis+2u)%3u];
  let code=(umSolidTileWord(t,1u+slot/24u)[(slot/6u)%4u]>>(4u*(slot%6u)))&15u;
  if(code!=0u){return 0.125*f32(code-1u);}
 }`);
 }
 if(packedBandForces){
  const origin="let p=vec3i(bTile(s)*4u+bLocal(lane));let cell=s*64u+bRow(bLocal(lane));";
  const load="let forced=bForcedField(p,axis,sign);";
  assert.equal(code.split(origin).length,2);assert.equal(code.split(load).length,2);
  code=code.replace(origin,origin+"let ownForced=textureLoad(velocity,p,0).xyz;")
   .replace(load,"var forced=ownForced[axis];if(sign<0){forced=bForcedField(p,axis,sign);}");
 }
 return code;
}
const coarseWeights=process.argv.includes("--coarse-weights");
const coarseWide=process.argv.includes("--coarse-wide");
const coarseGridDivisors=(process.argv.find(a=>a.startsWith("--coarse-grid-divisors="))?.split("=")[1]??
 process.argv.find(a=>a.startsWith("--coarse-grid-divisor="))?.split("=")[1]??"0").split(",").map(Number);
assert.ok(coarseGridDivisors.length>0&&coarseGridDivisors.every(n=>Number.isInteger(n)&&n>=0));
let coarseGridDivisor=coarseGridDivisors[0]!;
if(coarseGridDivisors.length>1)assert.ok(coarseGridDivisors.every(n=>n>0));
function heldCoarseWeights(code:string):string{
 const held=Number(code.match(/const HELD:u32=(\d+)u;/)?.[1]);assert.ok(held>0);
 const declaration="var weight:array<f32,6>;";
 const prepare="if(k==0u){for(var f=0u;f<6u;f++){weight[f]=coarse[bC(3u+f,j)];}}";
 const consume="var w=weight[f];if(k!=0u){w=coarse[bC(3u+f,j)];}";
 for(const source of [declaration,prepare,consume])assert.equal(code.split(source).length,2);
 return code.replace(declaration,`var weight:array<f32,${6*held}>;`)
  .replace(prepare,"for(var f=0u;f<6u;f++){weight[6u*k+f]=coarse[bC(3u+f,j)];}")
  .replace(consume,"let w=weight[6u*k+f];");
}
function indexedMiddleNeighbors(code:string):string{
 const start=code.indexOf("fn bMiddleNear("),end=code.indexOf("\n}",start)+2;
 assert.ok(start>=0&&end>start);
 code=code.slice(0,start)+`fn bMiddleNear(i:u32,c:u32,f:u32)->u32{
 let bit=1u<<(f/2u);let local=c&7u;var slot=c/8u+1u;
 if(((local&bit)!=0u)==((f&1u)!=0u)){slot=bMiddleSlots[6u*i+f];}
 if(slot==0u){return 0u;}
 return (slot-1u)*8u+(local^bit)+1u;
}`+code.slice(end);
 return code.replace("let g=bTile(c/8u)*2u+bMiddleLocal(c%8u);let diagonal=", "let parity=countOneBits(c&7u)&1u;let diagonal=")
  .replace("if(((g.x+g.y+g.z)&1u)==colour&&diagonal>0.0)","if(parity==colour&&diagonal>0.0)");
}
function loopBandNeighbors(code:string):string{
 const start=code.indexOf("fn bOffNear("),end=code.indexOf("fn bResidualNear(",start);
 assert.ok(start>=0&&end>start);
 return code.slice(0,start)+`fn bOffNear(cell:u32,s:u32,l:vec3u,p:vec3i,kinds:u32)->f32{
 var off=0.0;
 for(var f=0u;f<6u;f++){
  let axis=f/2u;let sign=bSign(f);let kind=bFaceKind(kinds,f);
  if(kind==K_BAND){var w=params.solve[1u+axis];if(!bUnit(kinds,f)){w=bCoefficient(cell,f);}off+=w*solve[bCellNear(s,l,f)-1u];}
  else if(kind==K_WALL){off+=bCoefficient(cell,f)*solve[bHalo(p,axis,sign)];}
 }
 return off;
 }\n`+code.slice(end);
}
// Colour-major rows make all neighbours flip bit5. Transverse half-row
// offsets are constants; only tile-boundary faces need a cached slot load.
function indexedBandNeighbors(code:string):string{
 const start=code.indexOf("fn bOffNear("),end=code.indexOf("fn bResidualNear(",start);
 assert.ok(start>=0&&end>start,"Expected local band neighbour helper");
 const faces=Array.from({length:6},(_,f)=>{
  const axis=f>>1,positive=!!(f&1),component=["x","y","z"][axis]!;
  const delta=axis===0?(positive?"i32(l.x&1u)":"-i32(1u-(l.x&1u))"):String((positive?1:-1)*(axis===1?2:8));
  const wrap=(positive?-1:1)*(axis===0?1:axis===1?6:24);
  return `{let kind=bFaceKind(kinds,${f}u);
 if(kind==K_BAND){
  var neighbor=own+u32(i32(across)+${delta});
  if(l.${component}==${positive?3:0}u){let slot=bNearSlot(s,${f}u);neighbor=0xffffffffu;if(slot!=0u){neighbor=(slot-1u)*64u+u32(i32(across)+${wrap});}}
  var w=params.solve[${axis+1}u];if(!bUnit(kinds,${f}u)){w=bCoefficient(cell,${f}u);}off+=w*solve[neighbor];
 }else if(kind==K_WALL){off+=bCoefficient(cell,${f}u)*solve[bHalo(p,${axis}u,${positive?1:-1})];}}
 `;
 }).join("\n");
 return code.slice(0,start)+`fn bOffNear(cell:u32,s:u32,l:vec3u,p:vec3i,kinds:u32)->f32{
 var off=0.0;let own=cell&~63u;let across=(cell&63u)^32u;
 ${faces}return off;
 }\n`+code.slice(end);
}
const optimized=lazyCoordinates||indexedNeighbors||middleIndexed||coarseWeights||cachedNeumann||rowAssembly;
class Variants{
 readonly device:GPUDevice;readonly baseline=new WeakMap<GPUComputePipeline,GPUComputePipeline>();readonly candidate=new WeakMap<GPUComputePipeline,GPUComputePipeline>();
 constructor(raw:GPUDevice){
  const sources=new WeakMap<GPUShaderModule,string>(),modules=new WeakMap<GPUShaderModule,GPUShaderModule>();
  this.device=new Proxy(raw,{get:(target,key)=>{
   if(key==="createBuffer")return(d:GPUBufferDescriptor)=>target.createBuffer(["Uniform pressure band aggregates","Uniform pressure band rows","Uniform pressure band parameters"].includes(d.label??"")?{...d,usage:d.usage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}:d);
   if(key==="createShaderModule")return(d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);sources.set(module,d.code);return module;};
   if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{
    const pipeline=await target.createComputePipelineAsync(d);
    if((coarseWide||coarseGridDivisor?[]:rowAssembly?["bandPrep"]:cachedNeumann?["bandInit","bandCopy","bandProject"]:coarseWeights?["bandCoarseSolve"]:middleIndexed?["bandMiddleSweep","bandMiddleRestrict"]:optimized?["bandSweep","bandMiddleProlong","bandRestrict","bandMeasure"]:["bandSweep","bandInit","bandMiddleProlong"]).includes(d.compute.entryPoint??"")){
     let module=modules.get(d.compute.module);
     const integrated=indexedNeighbors&&sources.get(d.compute.module)!.includes("let across=(cell&63u)^32u;");
     if(!module){const source=sources.get(d.compute.module);assert.ok(source);const code=rowAssembly?rowAssemblySource(source):cachedNeumann?pressureNeumannCache(source,d.compute.entryPoint!):coarseWeights?heldCoarseWeights(source):middleIndexed?indexedMiddleNeighbors(source):indexedNeighbors?(integrated?loopBandNeighbors(source):indexedBandNeighbors(source)):lazyCoordinates?lazyWallCoordinates(source):withoutHaloGuard(source);module=target.createShaderModule({label:"Diagnostic pressure band variant",code});
      const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
      if(errors.length){writeFileSync("/tmp/fluid-band-halo-failed.wgsl",code);throw new Error(errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));}modules.set(d.compute.module,module);}
     const twin=await target.createComputePipelineAsync({...d,compute:{...d.compute,module}});
     this.baseline.set(pipeline,optimized&&!integrated?pipeline:twin);this.candidate.set(pipeline,optimized&&!integrated?twin:pipeline);
    }
    return pipeline;
   };
   const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
  }});
 }
}
type Launch=number|"tiles"|"cells"|"slots"|"middle"|"coarse";
type Band={index:GPUBuffer;rows:GPUBuffer;coarse:GPUBuffer;solve:GPUBuffer;capacity:number;workSlots:number;schedule:{coarseSweeps:number};fields:UniformPressureBandFields;pipelines:Map<string,GPUComputePipeline>;launches:Map<string,unknown>;dispatch(p:GPUComputePassEncoder,name:string,launch:Launch):void;encodeSolve(e:GPUCommandEncoder):void;encodePrepare(e:GPUCommandEncoder):void};
type CopyRecord={name:string;size:number;source:GPUBuffer|GPUTexture;capture(e:GPUCommandEncoder,to:GPUBuffer):void;restore(e:GPUCommandEncoder,from:GPUBuffer):void};
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const steps=Number(option("steps")??120),rounds=Number(option("rounds")??8);
assert.ok([steps,rounds].every(n=>Number.isInteger(n)&&n>0));
const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser"),scene=structuredClone(sceneDocument(definition));assert.equal(definition.methodProfile?.methodId,method.id);
const quality=definition.methodProfile!.quality,values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s,out=option("out")??"artifacts/nbflip-ui-perf/band-halo.json";
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined;
const errors:string[]=[],buffers=new Set<GPUBuffer>();
try{
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 profile=new GPUPassProfile(raw);const variants=new Variants(profile.device);device=managedGPUDevice(variants.device,{requireWorkerRealm:false});
 device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const band=(solver as unknown as {mixedFrame:{band:Band}}).mixedFrame.band;
 const allocate=(size:number,read=false)=>{const b=raw.createBuffer({size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.COPY_SRC)});buffers.add(b);return b;};
 const release=(b:GPUBuffer)=>{b.destroy();buffers.delete(b);};
 const bufferRecord=(name:string,source:GPUBuffer,offset=0,size=source.size-offset):CopyRecord=>({name,size,source,capture:(e,to)=>e.copyBufferToBuffer(source,offset,to,0,size),restore:(e,from)=>e.copyBufferToBuffer(from,0,source,offset,size)});
 const recordsOf=():CopyRecord[]=>{
  const f=band.fields;
  // init/rebase also write rows; restoring the iterate alone is insufficient.
  const records=[bufferRecord("index",band.index),bufferRecord("rows",band.rows),bufferRecord("coarse",band.coarse),bufferRecord("solve",band.solve),bufferRecord("negative",f.negative),bufferRecord("params",f.params,0,64),bufferRecord("presentation",f.presentation.buffer,4*f.presentation.word,4*64*band.capacity)];
  if(rowAssembly)records.push(bufferRecord("forcedNegative",f.forced.negative));
  for(const [name,handle] of [["velocity",f.velocity],["velocityCopy",f.copy]] as const){
   const field=uniformDetailField(handle),textures=[...new Set(field?[field.storage.physical(field.handle),field.storage.baseOf(field.handle)!]:[handle])];
   for(const [bank,texture] of textures.entries()){
    assert.equal(texture.format,"rgba32float");const bytesPerRow=Math.ceil(texture.width*16/256)*256,rowsPerImage=texture.height,size=bytesPerRow*rowsPerImage*texture.depthOrArrayLayers;
    const extent:[number,number,number]=[texture.width,texture.height,texture.depthOrArrayLayers];
    records.push({name:`${name}-${bank}`,size,source:texture,capture:(e,to)=>e.copyTextureToBuffer({texture},{buffer:to,bytesPerRow,rowsPerImage},extent),restore:(e,from)=>e.copyBufferToTexture({buffer:from,bytesPerRow,rowsPerImage},{texture},extent)});
   }
  }
  return records;
 };
 let records:CopyRecord[]=[],snapshots:GPUBuffer[]=[],snapshotsTaken=0;
 const originalSolve=band.encodeSolve.bind(band),originalPrepare=band.encodePrepare.bind(band);
 const solve=rowAssembly?(e:GPUCommandEncoder)=>{
  const pass=e.beginComputePass({label:"Uniform pressure band frozen row assembly"});band.dispatch(pass,"prep","slots");pass.end();
 }:originalSolve;
 let frame=0;
 // Capture exactly before the final advance's last band solve. Repeated
 // CFL substeps overwrite these snapshots; no later phase overwrites them.
 const captureStage=(e:GPUCommandEncoder)=>{
  if(frame===steps){
   const next=recordsOf();
   if(next.length!==records.length||next.some((r,i)=>r.source!==records[i]?.source||r.size!==records[i]?.size)){
    snapshots.forEach(release);records=next;snapshots=records.map(r=>allocate(r.size));
   }
   records.forEach((r,i)=>r.capture(e,snapshots[i]!));snapshotsTaken++;
  }
 };
 if(rowAssembly)band.encodePrepare=(e)=>{originalPrepare(e);captureStage(e);};
 else band.encodeSolve=(e)=>{captureStage(e);originalSolve(e);};
 for(frame=1;frame<=steps;frame++){solver.advanceTo(frame*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(frame%12===0||frame===steps)console.log(JSON.stringify({phase:"advance",frame}));}
 band.encodeSolve=originalSolve;band.encodePrepare=originalPrepare;assert.ok(snapshotsTaken>0&&records.length>0);
 const restore=(e:GPUCommandEncoder)=>records.forEach((r,i)=>r.restore(e,snapshots[i]!));
 for(const divisor of coarseGridDivisors){
 coarseGridDivisor=divisor;
 const originals=[...band.pipelines].filter(([name])=>coarseWide||coarseGridDivisor?false:rowAssembly?name==="prep":cachedNeumann?["init","copy","project"].includes(name):coarseWeights?name==="coarseSolve":middleIndexed?(name.startsWith("middleSweep")||name==="middleRestrict"):name.startsWith("sweep")||name==="middleProlong"||(optimized?(name==="measure"||name.startsWith("restrict@")):name==="init"));assert.ok(coarseWide||coarseGridDivisor||originals.length>=((coarseWeights||rowAssembly)?1:cachedNeumann?3:4));
 for(const[,pipeline]of originals)assert.ok(variants.baseline.has(pipeline));
 const encode=(e:GPUCommandEncoder,useGuard:boolean)=>{
  for(const[name,pipeline]of originals){band.pipelines.set(name,(useGuard?variants.candidate:variants.baseline).get(pipeline)!);band.launches.delete(name);}
  const dispatch=band.dispatch;let replaced=0;
  if(coarseWide&&useGuard)band.dispatch=(pass,name,launch)=>{
   if(name==="coarseSolve"){
    replaced++;
    for(let n=0;n<band.schedule.coarseSweeps;n++)for(let color=0;color<2;color++)dispatch.call(band,pass,`coarseSweep${color}`,"coarse");
   }else dispatch.call(band,pass,name,launch);
  };
  if(coarseGridDivisor&&useGuard)band.dispatch=(pass,name,launch)=>{
   if(name.startsWith("coarseSweep")){
    replaced++;dispatch.call(band,pass,name,Math.min(Math.ceil(band.workSlots/coarseGridDivisor),256));
   }else dispatch.call(band,pass,name,launch);
  };
  try{solve(e);if((coarseWide||coarseGridDivisor)&&useGuard)assert.ok(replaced>0,"Fixture must exercise the selected coarse solve");}
  finally{band.dispatch=dispatch;for(const[name,pipeline]of originals){band.pipelines.set(name,pipeline);band.launches.delete(name);}}
 };
 const output=async(useGuard:boolean)=>{
  const result=records.map(r=>allocate(r.size)),e=raw.createCommandEncoder();restore(e);encode(e,useGuard);records.forEach((r,i)=>r.capture(e,result[i]!));raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();return result;
 };
 const left=await output(false),right=await output(true);
 let words=0,mismatches=0,retainedNeumannWords=0;const banks:unknown[]=[],examples:unknown[]=[];
 for(let bank=0;bank<records.length;bank++){
  if(cachedNeumann&&records[bank]!.name.startsWith("velocityCopy")){release(left[bank]!);release(right[bank]!);continue;}
  const size=records[bank]!.size,a=allocate(size,true),b=allocate(size,true),e=raw.createCommandEncoder();e.copyBufferToBuffer(left[bank]!,0,a,0,size);e.copyBufferToBuffer(right[bank]!,0,b,0,size);raw.queue.submit([e.finish()]);
  await Promise.all([a.mapAsync(GPUMapMode.READ),b.mapAsync(GPUMapMode.READ)]);const av=new Uint32Array(a.getMappedRange()),bv=new Uint32Array(b.getMappedRange());let differences=0;
  for(let i=0;i<av.length;i++)if(av[i]!==bv[i]){
   if(cachedNeumann&&records[bank]!.name==="rows"){
    const n=64*band.capacity,field=Math.floor(i/n),cell=i%n;
    if(field>=9&&field<15&&((av[2*n+cell]!>>(3*(field-9)))&7)===4){retainedNeumannWords++;continue;}
   }
   differences++;if(examples.length<12)examples.push({bank:records[bank]!.name,word:i,reference:av[i],candidate:bv[i]});}
  banks.push({name:records[bank]!.name,words:av.length,mismatches:differences});words+=av.length;mismatches+=differences;
  a.unmap();b.unmap();release(a);release(b);release(left[bank]!);release(right[bank]!);
 }
 console.log(JSON.stringify({phase:"parity",words,mismatches,retainedNeumannWords,banks,examples}));
 assert.equal(mismatches,0,"All non-scratch solve and projected velocity outputs must match before timing");
 const timingReady=option("timing-ready");if(timingReady){console.log(JSON.stringify({phase:"await-timing",timingReady}));while(!existsSync(timingReady))await new Promise(resolve=>setTimeout(resolve,250));}
 // Identical mutable resources create dependency ordering; fence every A/B
 // separately as well. Restores precede the timestamps and are not timed.
 for(const useGuard of [false,true,false,true,false,true]){const e=raw.createCommandEncoder();restore(e);encode(e,useGuard);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();}
 const timings:unknown[]=[],totals:{reference:number[];candidate:number[]}={reference:[],candidate:[]},elapsed:{reference:number[];candidate:number[]}={reference:[],candidate:[]};
 for(let round=0;round<rounds;round++)for(const useGuard of round%2?[true,false]:[false,true]){
  profile.start();const e=device.createCommandEncoder();restore(e);encode(e,useGuard);device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();const passes=await profile.finish();
  const sum=passes.reduce((v,p)=>v+p.ms,0),span=passes.reduce((v,p)=>v+p.ms+p.idle_ms,0),label=useGuard?"candidate":"reference";totals[label].push(sum);elapsed[label].push(span);timings.push({round,label,ms:sum,elapsed_ms:span,passes});
 }
 const means=Object.fromEntries(Object.entries(totals).map(([name,ms])=>[name,ms.reduce((a,b)=>a+b,0)/ms.length]));
 const elapsedMeans=Object.fromEntries(Object.entries(elapsed).map(([name,ms])=>[name,ms.reduce((a,b)=>a+b,0)/ms.length]));
 const result={date:new Date().toISOString(),sceneId:definition.id,values,dt,steps,rounds,snapshotsTaken,snapshotBytes:records.reduce((n,r)=>n+r.size,0),capacity:band.capacity,workSlots:band.workSlots,coarseGridDivisor,reference:(cachedBandFaces||cachedBandFacesInternal)?"full geometric queries for all fine face fractions":packedBandForces?"independent loads for all six face velocities":cachedBandSolid?"geometric threshold queries in row assembly":cachedNeumann?"full band velocity copy and projection texture reads":coarseGridDivisor?"wide 4h grid sized for both colours":coarseWide?"single-workgroup 4h solve":coarseWeights?"first held 4h coefficients local, remaining rows streamed":middleIndexed?"2h coordinate-derived neighbours and parity":indexedNeighbors?"looped neighbor address calculation":lazyCoordinates?"packed guard with eager global coordinates":"six-face halo loop for every row",candidate:cachedBandFacesInternal?"exact patch cache after original clear-tile fast exit":cachedBandFaces?"exact cached fractions on tile boundary faces, original evaluation within tiles":packedBandForces?"shared positive-face velocity vector":cachedBandSolid?"cached solid-open mask in row assembly":cachedNeumann?"Neumann row retention and seam-only velocity copy":coarseGridDivisor?"wide 4h grid sized for one colour, same grid-stride coverage":coarseWide?"multi-workgroup 4h half-sweeps with identical iterations":coarseWeights?"all held 4h coefficients loaded once":middleIndexed?"2h binary neighbours and local parity":indexedNeighbors?"fixed color-row offsets and six direct faces":lazyCoordinates?"packed guard with global coordinates only for wall faces":"packed no-wall early return",words,mismatches,retainedNeumannWords,banks,examples,means_ms:means,means_elapsed_ms:elapsedMeans,timingIsolation:rowAssembly?"same live row-assembly snapshot restored; all row and iterate words checked; variants separately submitted and fenced":"same full live pre-solve snapshot restored; identical scheduling; variants separately submitted and fenced",timings,errors};
 const destination=coarseGridDivisors.length>1?out.replace(/\.json$/,"-")+divisor+".json":out;
 mkdirSync(dirname(destination),{recursive:true});writeFileSync(destination,JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result));
 assert.equal(mismatches,0,"Every band solve output bit must match");assert.deepEqual(errors,[]);
 }
}finally{buffers.forEach(b=>b.destroy());solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
