/** Paired direct-launch scheduling comparison on one device. Both solvers
 * advance through the same trajectory; block order alternates to reduce
 * clock/thermal bias. Compiling, warmup and field readbacks are outside timing.
 * --scene=high-resolution-dam-break --policy=requested --reference=fixed
 * --blocks=8 --steps=30 --out=/tmp/buffered-work.json
 * reference=fixed restores the previous ownership and h-band grids; shared
 * is a same-policy noise control; sharpen-jobs restores unconditional seam
 * staging in the control. sharpen-fusion is a rejected QA experiment (its
 * orphan/dust stage parity fails), never a production configuration.
 * QA only, no app switch.
 */
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {mkdirSync,readFileSync,readdirSync,writeFileSync} from "node:fs";
import {dirname,resolve} from "node:path";
import {pathToFileURL} from "node:url";
import {managedGPUDevice,gpuCompilationManagerFor} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {resolveMethodValues} from "../lib/core/method-contract";
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformTransportWorkgroupReference,restoreUniformTransportWorkgroupDispatch} from "../tests/helpers/uniform-transport-workgroup-reference";
import {uniformTransportWorkCensus} from "./uniform-transport-work-census";
import {uniformSharpenFusionExperiment,installUniformSharpenFusionDispatch} from "./uniform-sharpen-fusion-experiment";
import {uniformSharpenJobReference} from "../tests/helpers/uniform-sharpen-job-reference";
import {readMixedTexture,readMixedTileWords} from "../tests/helpers/uniform-mixed-native-fields";

const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const sceneId=arg("scene","high-resolution-dam-break"),policy=arg("policy","requested"),reference=arg("reference","fixed");
const blocks=Number(arg("blocks","8")),steps=Number(arg("steps","30")),warmup=16,dt=1/60;
assert.ok(["requested","full","dynamic"].includes(policy)&&["shared","fixed","transport","sharpen","sharpen-fusion","sharpen-jobs"].includes(reference));
assert.ok(Number.isInteger(blocks)&&blocks>=2&&Number.isInteger(steps)&&steps>=2);
const out=resolve(arg("out","/tmp/uniform-buffered-work.json"));
const files=readdirSync("lib/methods/uniform").filter(p=>p.endsWith(".ts")).sort();
const fingerprint=()=>createHash("sha256").update(files.map(p=>`${p}:${createHash("sha256").update(readFileSync(`lib/methods/uniform/${p}`)).digest("hex")}`).join("\n")).digest("hex");
const report:Record<string,unknown>={sceneId,policy,reference,blocks,steps,warmup,dt_s:dt,sourceBefore:fingerprint(),
 scope:"Paired simulation-only wall throughput on one device, two frames in flight; matched blocks alternate order. No timestamps, rendering or field reads inside timing."};
const save=()=>{mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(report,null,2)+"\n");};
let device:GPUDevice|undefined;
const solvers:WebGPUUniformReferenceSolver[]=[];
let announced=false;
for(;;){try{await acquireWebGPUExclusiveLock("dawn-benchmark",`Paired Uniform work budgets: ${sceneId}/${reference}`);break;}
 catch(error){if(!(error instanceof Error)||!error.message.includes("Refusing concurrent GPU execution"))throw error;
  if(!announced){console.log("Waiting for WebGPU lease");announced=true;}await new Promise(r=>setTimeout(r,250));}}
try{
 const dawn=await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE??"node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];report.validationErrors=errors;device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 report.adapter={vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description};
 const scene=sceneDocument(getSceneDefinition(sceneId));scene.numerics.fixedDt_s=scene.numerics.maxDt_s=dt;
 if(policy==="requested")scene.fluid.refinementRegions=[];
 const values=resolveMethodValues(uniformVolumeMethod,"balanced",{timeStep:"scene",detailPolicy:policy});report.values=values;
 usePerformanceInstrumentationStore.getState().setEnabled(false);
 const started=performance.now();
 let referenceArm=false,referenceModules=0;
 if(reference==="transport"||reference==="sharpen-fusion"||reference==="sharpen-jobs"){
  const compiler=gpuCompilationManagerFor(device),create=compiler.createShaderModule.bind(compiler);
  compiler.createShaderModule=descriptor=>{
   let code=descriptor.code;
   if(reference==="transport"&&referenceArm&&code.includes("// Four coarse rows per workgroup,")){
    code=uniformTransportWorkgroupReference(code);
    referenceModules++;
   }
   if(reference==="sharpen-fusion"&&!referenceArm&&code.includes("fn shLimit(o:UMOwner){")){
    code=uniformSharpenFusionExperiment(code);referenceModules++;
   }
   if(reference==="sharpen-jobs"&&referenceArm&&code.includes("@compute @workgroup_size(192) fn propose(")){
    code=uniformSharpenJobReference(code);referenceModules++;
   }
   return create({...descriptor,code});
  };
 }
 for(let arm=0;arm<2;arm++){
  referenceArm=arm===0;
  solvers.push(await uniformVolumeMethod.createSolverAsync!(device,structuredClone(scene),"balanced",values,undefined,()=>{}) as WebGPUUniformReferenceSolver);
 }
 if(reference==="transport")assert.ok(referenceModules>0,"The control must compile the previous transport shader");
 if(reference==="sharpen-fusion")assert.ok(referenceModules>0,"The candidate must compile the fused sharpening shader");
 if(reference==="sharpen-jobs")assert.ok(referenceModules>0,"The control must compile the previous unpartitioned sharpening shader");
 await device.queue.onSubmittedWorkDone();report.setup_ms=performance.now()-started;
 // Per-instance QA changes leave the candidate on the production path.
 const control=solvers[0] as any;
 if(reference==="transport")restoreUniformTransportWorkgroupDispatch(control.mixedFrame.transport);
 if(reference==="sharpen")control.mixedFrame.sharpen.observeWork=()=>{};
 if(reference==="sharpen-fusion")installUniformSharpenFusionDispatch((solvers[1] as any).mixedFrame.sharpen);
 if(reference==="fixed"){
  const frame=control.mixedFrame;
  for(const o of new Set<any>([frame.ownership,...frame.levels.map((l:any)=>l.ownership)])){
   for(const k of Object.keys(o.work))delete o.work[k];o.observeWork=()=>{};
  }
  frame.sharpen.observeWork=()=>{};
  frame.band.workSlots=frame.band.capacity;frame.band.observeWork=()=>{};
 }
 const initial=await Promise.all(solvers.map(s=>s.readStats()));
 report.initial=initial.map(i=>({...i}));
 if(policy==="requested")for(const info of initial)assert.equal(info.uniformMixedFineTiles,0,"Requested fixture must actually be all coarse");
 const advance=async(s:WebGPUUniformReferenceSolver,first:number,count:number)=>{
  for(let i=first;i<first+count;i+=2){for(let j=i;j<Math.min(first+count,i+2);j++)assert.ok(s.advanceTo(j*dt,[]));await s.awaitFrameCompletion();}
  await device!.queue.onSubmittedWorkDone();
 };
 for(const s of solvers)await advance(s,1,warmup);
 const rows:{block:number;first:number;order:number[];msPerStep:number[]}[]=[];report.rows=rows;
 for(let block=0;block<blocks;block++){
  const first=warmup+block*steps+1,order=block%2?[1,0]:[0,1],msPerStep=[0,0];
  for(const arm of order){const start=performance.now();await advance(solvers[arm]!,first,steps);msPerStep[arm]=(performance.now()-start)/steps;}
  rows.push({block,first,order,msPerStep});console.log(JSON.stringify(rows.at(-1)));save();
 }
 const hashes:Record<string,string>[]=[];
 for(const s of solvers){const fields=(s as any).mixedFrame.fields,hash:Record<string,string>={};
  for(const name of ["volume","velocity","phi"]){const a=await readMixedTexture(device,fields[name]);hash[name]=createHash("sha256").update(new Uint8Array(a.buffer,a.byteOffset,a.byteLength)).digest("hex");}
  hashes.push(hash);
 }
 if(process.argv.includes("--work-census")){
  const solver=solvers[1]!,frame=(solver as any).mixedFrame;
  const [volume,departures,tiles]=await Promise.all([readMixedTexture(device,frame.fields.volume),readMixedTexture(device,frame.fields.departure),readMixedTileWords(device,solver)]);
  report.workCensus=uniformTransportWorkCensus(frame.ownership.capacity.lattice.dimensions,tiles,volume,departures);
 }
 report.fieldHashes=hashes;report.fieldsExact=JSON.stringify(hashes[0])===JSON.stringify(hashes[1]);
 report.final=await Promise.all(solvers.map(async s=>({...await s.readStats()})));
 for(const s of solvers)assert.equal(s.info.simulationPipelineError,undefined);
 assert.deepEqual(errors,[]);report.validationErrors=errors;
 const mean=(a:number[])=>a.reduce((s,n)=>s+n,0)/a.length;
 const referenceMs=mean(rows.map(r=>r.msPerStep[0]!)),candidateMs=mean(rows.map(r=>r.msPerStep[1]!));
 const pairedReductionPercent=rows.map(r=>100*(1-r.msPerStep[1]!/r.msPerStep[0]!));
 const sorted=pairedReductionPercent.toSorted((a,b)=>a-b),middle=Math.floor(sorted.length/2);
 report.summary={referenceMs,candidateMs,reductionPercent:100*(1-candidateMs/referenceMs),pairedReductionPercent,
  medianPairedReductionPercent:sorted.length%2?sorted[middle]!:(sorted[middle-1]!+sorted[middle]!)/2,
  candidateFasterBlocks:pairedReductionPercent.filter(n=>n>0).length};
 report.launchBudgets=solvers.map(s=>{const frame=(s as any).mixedFrame;return {ownership:frame.ownership.work,bandSlots:frame.band.workSlots,sharpenGroups:frame.sharpen.sweepWork};});
 report.sourceAfter=fingerprint();save();console.log(JSON.stringify({out,summary:report.summary,fieldsExact:report.fieldsExact}));
 assert.equal(report.fieldsExact,true,"The compared operator or scheduling variant must retain exact final simulation fields in this fixture");
}catch(error){report.failure=String(error);report.sourceAfter=fingerprint();save();throw error;}
finally{for(const s of solvers)s.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
