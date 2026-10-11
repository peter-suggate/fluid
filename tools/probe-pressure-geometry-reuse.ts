/** Frozen adaptive redistance + pressure geometry replay. Compares per-cell,
 * shared-per-tile, and subgroup-broadcast pressure fill planes. Inputs are captured
 * immediately before the last frame's redistance, never advanced by a trial.
 * /opt/homebrew/bin/node --import tsx tools/probe-pressure-geometry-reuse.ts --steps=96 --subgroup-only --wait-idle
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync,writeFileSync } from "node:fs";
import { dirname,resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField,uniformDetailPick,type UniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import type { UniformMixedSolid } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";
import { closedPressureCellsWGSL,closedPressureTilesWGSL,lazyPressureFractionWGSL,pressureMetricCullWGSL,pressurePlaneCoverageWGSL,sharedPressurePlaneWGSL,subgroupPressurePlaneWGSL } from "./pressure-geometry-reuse";

type Stage={bins:GPUBuffer;params:GPUBuffer;adaptiveSurface:boolean;ownership:UniformMixedOwnership;solid?:UniformMixedSolid;parity:number;pipelines:Map<string,GPUComputePipeline>;redistance(e:GPUCommandEncoder):void};
type Geometry={metricPipeline:GPUComputePipeline;solid?:UniformMixedSolid;encode(e:GPUCommandEncoder,g:UniformDetailGroup,options:{pressure:boolean;metric:boolean}):void};
type Frame={narrowBandFlip:Stage;geometry:Geometry;geometryGroup:UniformDetailGroup;phiResolve:{encode(e:GPUCommandEncoder,g:UniformDetailGroup):void};phiResolveGroups:{phi:UniformDetailGroup};fields:{phi:GPUTexture;target:GPUTexture;centerPhi:GPUTexture;volume:GPUTexture;pressureGeometry:{target:GPUTexture;centerPhi:GPUTexture}}};
type Copy={name:string;source:GPUBuffer|GPUTexture;offset?:number;size:number;snapshot:GPUBuffer;capture(e:GPUCommandEncoder,to:GPUBuffer):void;restore(e:GPUCommandEncoder):void};
type Variant="reference"|"cull"|"plane"|"both"|"subgroup"|"subgroup-eager"|"subgroup-probes"|"fraction-if"|"closed"|"fraction-if-closed"|"closed-cells"|"fraction-if-closed-cells";
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const steps=Number(option("steps")??96),rounds=Number(option("rounds")??12),repeats=Number(option("repeats")??1);
const variants:Variant[]=process.argv.includes("--cells-only")?["reference","closed","closed-cells","fraction-if-closed-cells"]:process.argv.includes("--cheap-only")?["reference","fraction-if","closed","fraction-if-closed"]:process.argv.includes("--subgroup-only")?["reference","subgroup","subgroup-eager","subgroup-probes"]:process.argv.includes("--all-planes")?["reference","plane","subgroup","subgroup-eager","subgroup-probes"]:process.argv.includes("--cull-only")?["reference","cull"]:["reference","plane"];
const out=option("out")??"artifacts/nbflip-ui-perf/pressure-geometry-reuse-96.json";
const definition=getSceneDefinition(option("scene")??"nbflip-figure-9-teaser"),scene=structuredClone(sceneDocument(definition)),quality=definition.methodProfile!.quality;
const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides}),dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
assert.ok([steps,rounds,repeats].every(n=>Number.isInteger(n)&&n>0));
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,profile:GPUPassProfile|undefined;
const errors:string[]=[],buffers=new Set<GPUBuffer>();
try{
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query",...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 if(!raw.features.has("subgroups"))for(let i=variants.length-1;i>=0;i--)if(variants[i]!.startsWith("subgroup"))variants.splice(i,1);
 assert.ok(variants.length>1,"This requested trial requires device subgroups; production retains its original pipeline without them");
 profile=new GPUPassProfile(raw);
 const sources=new WeakMap<GPUShaderModule,string>(),descriptors=new WeakMap<GPUComputePipeline,GPUComputePipelineDescriptor>();
 const groups=new WeakMap<GPUBindGroup,GPUBindGroupEntry[]>(),views=new WeakMap<GPUTextureView,GPUTexture>();
 const proxy=new Proxy(profile.device,{get:(target,key)=>{
  if(key==="createBuffer")return(d:GPUBufferDescriptor)=>target.createBuffer((d.usage&(GPUBufferUsage.MAP_READ|GPUBufferUsage.MAP_WRITE))?d:{...d,usage:d.usage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  if(key==="createTexture")return(d:GPUTextureDescriptor)=>{const texture=target.createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST}),original=texture.createView.bind(texture);texture.createView=(d?:GPUTextureViewDescriptor)=>{const view=original(d);views.set(view,texture);return view;};return texture;};
  if(key==="createBindGroup")return(d:GPUBindGroupDescriptor)=>{const entries=[...d.entries],group=target.createBindGroup({...d,entries});groups.set(group,entries);return group;};
  if(key==="createShaderModule")return(d:GPUShaderModuleDescriptor)=>{const module=target.createShaderModule(d);sources.set(module,d.code);return module;};
  if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{const pipeline=await target.createComputePipelineAsync(d);descriptors.set(pipeline,d);return pipeline;};
  const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
 }});
 device=managedGPUDevice(proxy,{requireWorkerRealm:false});device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 const allocate=(size:number,read=false)=>{const b=raw.createBuffer({size,usage:GPUBufferUsage.COPY_DST|(read?GPUBufferUsage.MAP_READ:GPUBufferUsage.COPY_SRC)});buffers.add(b);return b;};
 const release=(b:GPUBuffer)=>{b.destroy();buffers.delete(b);};
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame,stage=frame.narrowBandFlip,geometry=frame.geometry;
 assert.ok(stage.adaptiveSurface,"The sparse first metric requires adaptive end-step rebuilding");
 const tiles=stage.ownership.capacity.tiles,dims=stage.ownership.capacity.lattice.dimensions,h=Math.min(...stage.ownership.capacity.lattice.cellSize_m);
 const cells=dims.reduce((n,v)=>n*v,1),vertices=dims.reduce((n,v)=>n*(v+1),1),metricOffset=4*(2*cells+2*tiles),metricSize=4*vertices;
 const records:Copy[]=[],outputNames=new Map<GPUTexture,string>();
 const recordBuffer=(name:string,binding:GPUBufferBinding)=>{const {buffer,offset=0,size=buffer.size-offset}=binding;if(records.some(r=>r.source===buffer&&r.offset===offset&&r.size===size))return;
  const snapshot=allocate(size);records.push({name,source:buffer,offset,size,snapshot,capture:(e,to)=>e.copyBufferToBuffer(buffer,offset,to,0,size),restore:e=>e.copyBufferToBuffer(snapshot,0,buffer,offset,size)});
 };
 const recordTexture=(name:string,texture:GPUTexture)=>{if(records.some(r=>r.source===texture))return;const bytes=texture.format==="r32float"?4:texture.format==="rg32float"?8:texture.format==="rgba32float"?16:0;assert.ok(bytes,texture.format);
  const extent:[number,number,number]=[texture.width,texture.height,texture.depthOrArrayLayers],bytesPerRow=Math.ceil(texture.width*bytes/256)*256,rowsPerImage=texture.height,size=bytesPerRow*rowsPerImage*texture.depthOrArrayLayers,snapshot=allocate(size);
  records.push({name,source:texture,size,snapshot,capture:(e,to)=>e.copyTextureToBuffer({texture},{buffer:to,bytesPerRow,rowsPerImage},extent),restore:e=>e.copyBufferToTexture({buffer:snapshot,bytesPerRow,rowsPerImage},{texture},extent)});
 };
 const recordGroup=(name:string,group:GPUBindGroup)=>{const entries=groups.get(group);assert.ok(entries);for(const entry of entries){const resource=entry.resource;if("buffer" in resource)recordBuffer(`${name}.${entry.binding}`,resource as GPUBufferBinding);else{const texture=views.get(resource as GPUTextureView);if(texture)recordTexture(`${name}.${entry.binding}`,texture);}}};
 const recordField=(name:string,texture:GPUTexture)=>{const field=uniformDetailField(texture),textures=[...new Set(field?[field.storage.physical(field.handle),field.storage.baseOf(field.handle)!]:[texture])];for(const t of textures){recordTexture(name,t);outputNames.set(t,name);}};
 let iteration=0,snapshots=0,savedParity=0;const originalRedistance=stage.redistance.bind(stage);
 stage.redistance=e=>{
  if(iteration===steps){
   assert.equal(snapshots,0,"Expect one redistance in the frozen frame");savedParity=stage.parity;
   recordGroup("ownership",stage.ownership.bindGroup);recordGroup("geometry",frame.geometryGroup.group);recordGroup("geometryBase",frame.geometryGroup.base);
   if(stage.solid)recordGroup("solid",stage.solid.tileGroup);
   recordBuffer("bins",{buffer:stage.bins});recordBuffer("nbParams",{buffer:stage.params});
   for(const [name,texture] of Object.entries({phi:frame.fields.phi,target:frame.fields.target,centerPhi:frame.fields.centerPhi,volume:frame.fields.volume,pressureTarget:frame.fields.pressureGeometry.target,pressureCenterPhi:frame.fields.pressureGeometry.centerPhi}))recordField(name,texture);
   records.forEach(r=>r.capture(e,r.snapshot));snapshots++;
  }
  originalRedistance(e);
 };
 for(iteration=1;iteration<=steps;iteration++){solver.advanceTo(iteration*dt,[]);await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(iteration%12===0||iteration===steps)console.log(JSON.stringify({phase:"advance",frame:iteration}));}
 stage.redistance=originalRedistance;assert.equal(snapshots,1);stage.parity=savedParity;
 const metricHandle=stage.pipelines.get("buildDistance")!,geometryHandle=geometry.metricPipeline;
 const selected=(p:GPUComputePipeline)=>uniformDetailPick(stage.solid?.select(p)??p);
 const compile=async(original:GPUComputePipeline,transform:(code:string)=>string,label:string)=>{const descriptor=descriptors.get(selected(original));assert.ok(descriptor);const source=sources.get(descriptor.compute.module);assert.ok(source);const code=transform(source);writeFileSync(`/tmp/fluid-${label}.wgsl`,code);const module=raw.createShaderModule({label,code}),messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(messages.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`),[]);return raw.createComputePipelineAsync({...descriptor,compute:{...descriptor.compute,module}});};
 const culled=variants.some(v=>v==="cull"||v==="both")?await compile(metricHandle,pressureMetricCullWGSL,"pressure-metric-cull"):metricHandle;
 const shared=variants.some(v=>v==="plane"||v==="both")?await compile(geometryHandle,sharedPressurePlaneWGSL,"pressure-shared-plane"):geometryHandle;
 const subgroup=variants.includes("subgroup")?await compile(geometryHandle,code=>subgroupPressurePlaneWGSL(code,true),"pressure-subgroup-plane"):geometryHandle;
 const subgroupEager=variants.includes("subgroup-eager")?await compile(geometryHandle,code=>subgroupPressurePlaneWGSL(code,false),"pressure-subgroup-plane-eager"):geometryHandle;
 const subgroupProbes=variants.includes("subgroup-probes")?await compile(geometryHandle,code=>subgroupPressurePlaneWGSL(code,true,true),"pressure-subgroup-plane-probes"):geometryHandle;
 const fractionIf=variants.includes("fraction-if")?await compile(geometryHandle,lazyPressureFractionWGSL,"pressure-fraction-if"):geometryHandle;
 const closed=variants.includes("closed")?await compile(geometryHandle,closedPressureTilesWGSL,"pressure-closed-tiles"):geometryHandle;
 const combined=variants.includes("fraction-if-closed")?await compile(geometryHandle,code=>closedPressureTilesWGSL(lazyPressureFractionWGSL(code)),"pressure-fraction-if-closed"):geometryHandle;
 const closedCells=variants.includes("closed-cells")?await compile(geometryHandle,closedPressureCellsWGSL,"pressure-closed-cells"):geometryHandle;
 const combinedCells=variants.includes("fraction-if-closed-cells")?await compile(geometryHandle,code=>closedPressureCellsWGSL(lazyPressureFractionWGSL(code)),"pressure-fraction-if-closed-cells"):geometryHandle;
 const geometryPipelines:Partial<Record<Variant,GPUComputePipeline>>={plane:shared,both:shared,subgroup,"subgroup-eager":subgroupEager,"subgroup-probes":subgroupProbes,"fraction-if":fractionIf,closed,"fraction-if-closed":combined,"closed-cells":closedCells,"fraction-if-closed-cells":combinedCells};
 const restore=(e:GPUCommandEncoder)=>records.forEach(r=>r.restore(e));
 const encode=(e:GPUCommandEncoder,variant:Variant)=>{
  stage.pipelines.set("buildDistance",variant==="cull"||variant==="both"?culled:metricHandle);geometry.metricPipeline=geometryPipelines[variant]??geometryHandle;
  try{originalRedistance(e);frame.phiResolve.encode(e,frame.phiResolveGroups.phi);geometry.encode(e,frame.geometryGroup,{pressure:true,metric:true});}
  finally{stage.pipelines.set("buildDistance",metricHandle);geometry.metricPipeline=geometryHandle;}
 };
 const outputs=records.filter(r=>outputNames.has(r.source as GPUTexture));
 const readBuffer=async(source:GPUBuffer,size=source.size,offset=0)=>{const read=allocate(size,true),e=raw.createCommandEncoder();e.copyBufferToBuffer(source,offset,read,0,size);raw.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange().slice(0));read.unmap();release(read);return data;};
 const capture=async(variant:Variant)=>{const e=raw.createCommandEncoder();restore(e);encode(e,variant);const reads=outputs.map(r=>{const read=allocate(r.size,true);r.capture(e,read);return read;});raw.queue.submit([e.finish()]);const result=await Promise.all(reads.map(async(read,i)=>{await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange().slice(0));read.unmap();release(read);return{name:outputNames.get(outputs[i]!.source as GPUTexture)!,data};}));return result;};
 const topologyRecord=records.find(r=>r.name==="ownership.0");assert.ok(topologyRecord);const topology=await readBuffer(topologyRecord.snapshot);
 let planeCoverage:Record<string,number>|undefined;
 if(process.argv.includes("--coverage")){
  const coverage=await compile(geometryHandle,pressurePlaneCoverageWGSL,"pressure-plane-coverage"),record=records.find(r=>r.name==="geometry.5");assert.ok(record);
  const e=raw.createCommandEncoder();restore(e);e.clearBuffer(record.source as GPUBuffer,record.offset??0,48);geometry.metricPipeline=coverage;
  try{originalRedistance(e);frame.phiResolve.encode(e,frame.phiResolveGroups.phi);geometry.encode(e,frame.geometryGroup,{pressure:true,metric:true});}
  finally{geometry.metricPipeline=geometryHandle;}
  raw.queue.submit([e.finish()]);const counters=await readBuffer(record.source as GPUBuffer,48,record.offset??0);
  planeCoverage=Object.fromEntries(["fineTiles","cutFineTiles","activeCutFineTiles","activeOpenFineCells","planarFineTiles","nonplanarFineTiles","activeCutCoarseTiles","closedFineTiles","closedCoarseTiles","closedFineCells","nonplanarFineCells","planarFineCells"].map((name,i)=>[name,counters[i]!]));
  console.log(JSON.stringify({phase:"planeCoverage",...planeCoverage}));
 }
 const left=await capture("reference"),comparisons:unknown[]=[];
 const reach=await readBuffer(stage.bins,4*tiles,metricOffset+metricSize+4*cells+16+8*tiles);let reachedVertices=0,skippedVertices=0,reachedCoarseTiles=0;
 for(let tile=0;tile<tiles;tile++)if((reach[tile]!&2)!==0){const tc=[tile%(dims[0]!/4),Math.floor(tile/(dims[0]!/4))%(dims[1]!/4),Math.floor(tile/(dims[0]!/4)/(dims[1]!/4))],coarse=(topology[2*tiles+2*tile+1]!>>>27)===4;if(coarse)reachedCoarseTiles++;
  for(let z=0;z<4+Number(tc[2]===dims[2]!/4-1);z++)for(let y=0;y<4+Number(tc[1]===dims[1]!/4-1);y++)for(let x=0;x<4+Number(tc[0]===dims[0]!/4-1);x++){reachedVertices++;if(coarse&&((x&3)!==0||(y&3)!==0||(z&3)!==0))skippedVertices++;}
 }
 console.log(JSON.stringify({phase:"coverage",reachedVertices,skippedVertices,reachedCoarseTiles,snapshotBytes:records.reduce((n,r)=>n+r.size,0)}));
 for(const variant of variants.filter(v=>v!=="reference")){const right=await capture(variant);const fields=left.map((a,j)=>{const b=right[j]!,af=new Float32Array(a.data.buffer),bf=new Float32Array(b.data.buffer);let differences=0,maxAbsolute=0,maxRelative=0,signChanges=0,nonfinite=0,thresholdChanges=0;const examples:unknown[]=[];
   for(let i=0;i<a.data.length;i++)if(a.data[i]!==b.data[i]){differences++;const x=af[i]!,y=bf[i]!,absolute=Math.abs(x-y);maxAbsolute=Math.max(maxAbsolute,absolute);if(Math.abs(x)>=1e-4*h)maxRelative=Math.max(maxRelative,absolute/Math.abs(x));signChanges+=Number((x<0)!==(y<0));nonfinite+=Number(!Number.isFinite(x)||!Number.isFinite(y));thresholdChanges+=Number((x<-.001*h)!==(y<-.001*h))+Number((x<-.004*h)!==(y<-.004*h));if(examples.length<8)examples.push({word:i,x,y,absolute});}
   return{name:a.name,words:a.data.length,differences,maxAbsolute,maxAbsolute_cells:maxAbsolute/h,maxRelative,signChanges,thresholdChanges,nonfinite,examples};});
  comparisons.push({variant,fields});console.log(JSON.stringify({phase:"parity",variant,fields}));assert.ok(fields.every(f=>f.nonfinite===0));if(variant==="cull")assert.ok(fields.every(f=>f.differences===0),"Skipping unused metric vertices must leave every output bit unchanged");
 }
 if(process.argv.includes("--wait-idle"))for(;;){const other=execFileSync("ps",["-axo","pid=,command="],{encoding:"utf8"}).split("\n").filter(line=>/node .*\b(probe-|benchmark-|validate-)/.test(line)&&!line.includes(` ${process.pid} `)&&!line.includes("probe-pressure-geometry-reuse.ts"));if(!other.length)break;console.log(JSON.stringify({phase:"waiting",other}));await new Promise(resolve=>setTimeout(resolve,2000));}
 for(let n=0;n<2;n++)for(const variant of variants){const e=raw.createCommandEncoder();restore(e);encode(e,variant);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();}
 console.log(JSON.stringify({phase:"timing",rounds,repeats}));const timings:{round:number;variant:Variant;ms:number;passes:Awaited<ReturnType<GPUPassProfile["finish"]>>}[]=[];
 for(let round=0;round<rounds;round++)for(const variant of round%2?[...variants].reverse():variants){
  // Every repetition starts from the exact same pre-redistance phi. Restores
  // are submitted/fenced before timestamped work, never timed as savings.
  const all:Awaited<ReturnType<GPUPassProfile["finish"]>>=[];
  for(let repeat=0;repeat<repeats;repeat++){let e=raw.createCommandEncoder();restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();profile.start();e=device.createCommandEncoder();encode(e,variant);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();all.push(...await profile.finish());}
  timings.push({round,variant,ms:all.reduce((n,p)=>n+p.ms,0)/repeats,passes:all});
 }
 const mean=(variant:Variant)=>timings.filter(t=>t.variant===variant).reduce((n,t)=>n+t.ms,0)/rounds;
 const means=variants.map(variant=>({variant,ms:mean(variant),saved_ms:mean("reference")-mean(variant)}));
 const result={date:new Date().toISOString(),runtime:{version:process.version,execPath:process.execPath,cwd:process.cwd()},sceneId:definition.id,steps,dt,values,rounds,repeats,tiles,dims,reachedVertices,skippedVertices,reachedCoarseTiles,planeCoverage,comparisons,means,timings,errors};mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify({phase:"done",out,means,errors}));assert.deepEqual(errors,[]);
}finally{solver?.destroy();await device?.queue.onSubmittedWorkDone();buffers.forEach(b=>b.destroy());profile?.destroy();device?.destroy();}
