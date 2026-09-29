// scratch probe (dispatch grids) (delete when the mixed frame program closes).
// Long-dam dynamic frame time: untimed wall median plus a separate timed pass
// for phase means. Mimics the renderer: applyRuntimeValues(same values) every frame.
// Usage: FLUID_GPU_COMPILATION_CONCURRENCY=1 WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
//   node --import tsx tools/.probe-mixed-frame-a1.ts <out.json> [steps=60] [from=36]
import {writeFileSync} from "node:fs";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {GPUStageTimestampRecorder,GPUPassTimestampRecorder} from "../lib/core/performance-trace";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {resolveMethodValues} from "../lib/core/method-contract";
import {usePerformanceInstrumentationStore} from "../lib/core/stores/performance-instrumentation-store";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";

const [out,stepsArg,fromArg]=process.argv.slice(2);
const steps=Number(stepsArg??60),from=Number(fromArg??36),sceneId=process.env.PROBE_SCENE??"sparse-cm12-long-dam-break";
await acquireWebGPUExclusiveLock("dawn-probe",`fluid-a1 mixed frame ${sceneId}`);
let device:GPUDevice|undefined;const solvers:{destroy():void}[]=[];
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();
 device=managedGPUDevice(await adapter!.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter!.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];
 device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error("GPU error:",e.error.message);});
 await GPUStageTimestampRecorder.prepare(device);
 let recorder:GPUPassTimestampRecorder|undefined;const grids:Record<string,any>={};let gridFrames=0;const real=device;const names=new WeakMap<GPUComputePipeline,string>();
 const measured=new Proxy(device,{get(target,key){
  if(key==="createComputePipelineAsync")return async(desc:GPUComputePipelineDescriptor)=>{const p=await target.createComputePipelineAsync(desc);names.set(p,desc.compute.entryPoint??"?");return p;};
  if(key==="createComputePipeline")return(desc:GPUComputePipelineDescriptor)=>{const p=target.createComputePipeline(desc);names.set(p,desc.compute.entryPoint??"?");return p;};
  if(key==="createCommandEncoder")return(desc?:GPUCommandEncoderDescriptor)=>{const raw=target.createCommandEncoder(desc);const encoder=recorder?.instrument(raw)??raw;
   if(!recorder||process.env.PROBE_SPLIT!=="1")return encoder;
   return new Proxy(encoder,{get(e,k){
    if(k==="beginComputePass")return(pd:GPUComputePassDescriptor={})=>{
     let pipeline:GPUComputePipeline;const bindings=new Map<number,[GPUBindGroup,any]>();
     const dispatch=(method:string,args:unknown[])=>{const key=`${pd.label} | ${names.get(pipeline)??"?"}`;if(method==="dispatchWorkgroups"){const a=args as number[];const g=(a[0]??1)*(a[1]??1)*(a[2]??1);const r=(grids[key]??={calls:0,groups:0,max:0});r.calls++;r.groups+=g;r.max=Math.max(r.max,g);}else{(grids[key]??={calls:0,groups:0,max:0}).indirect=true;}const pass=e.beginComputePass({...pd,label:`${pd.label} | ${names.get(pipeline)??"?"}`});
      pass.setPipeline(pipeline);for(const [i,[g,o]] of bindings)pass.setBindGroup(i,g,o);(pass as any)[method](...args);pass.end();};
     return {setPipeline:(q:GPUComputePipeline)=>{pipeline=q;},setBindGroup:(i:number,g:GPUBindGroup,o?:any)=>{bindings.set(i,[g,o]);},
      dispatchWorkgroups:(...a:unknown[])=>dispatch("dispatchWorkgroups",a),dispatchWorkgroupsIndirect:(...a:unknown[])=>dispatch("dispatchWorkgroupsIndirect",a),end:()=>{},pushDebugGroup:()=>{},popDebugGroup:()=>{},insertDebugMarker:()=>{}} as unknown as GPUComputePassEncoder;};
    const v=(e as any)[k];return typeof v==="function"?v.bind(e):v;}});};
  const v=(target as any)[key];return typeof v==="function"?v.bind(target):v;}}) as GPUDevice;
 const scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));
 const values=resolveMethodValues(uniformVolumeMethod,"balanced",{...JSON.parse(process.env.PROBE_OVERRIDES??"{}")});
 const result:Record<string,unknown>={sceneId,steps,from,errors};
 for(const [rep,timed] of (process.env.PROBE_SPLIT==="1"?[[1,true]]:[[0,false],[1,true]]) as [number,boolean][]){
  const store=usePerformanceInstrumentationStore.getState();
  store.setEnabled(false);
  const solver=await uniformVolumeMethod.createSolverAsync!(measured,scene,"balanced",values,undefined,()=>{}) as any;solvers.push(solver);
  const bandTiles:number[]=[],bandCycles:number[]=[],bandResidual:number[]=[];let reused=0,relayouts=0;const remapStats:number[][]=[];let lastChanged:Uint8Array|undefined;{const f=solver.mixedFrame;const adv=f.advance.bind(f);f.advance=(...a:any[])=>{if(f.reusableExtension!==undefined&&f.reusableExtension===JSON.stringify({...a[0],dt:0}))reused++;const r=adv(...a);r.then((x:any)=>{if(x&&typeof x.bandTiles==="number"){bandTiles.push(x.bandTiles);bandCycles.push(x.bandCycles);bandResidual.push(x.bandResidual);}},()=>{});return r;};if(f.adoptBuiltLayout){const ad=f.adoptBuiltLayout.bind(f);f.adoptBuiltLayout=(l:any)=>{if(l.changedTiles)relayouts++;
   // Remap worklist size (markChanged's rule), rep1 only: host-side from the two layouts.
   if(timed&&l.changedTiles){const a=f.ownership.layout.tiles as Uint32Array,b=l.layout.tiles as Uint32Array,T=f.ownership.layout.lattice.dimensions.map((d:number)=>d/4);
    const w=(x:number)=>(x&0x80000000)?1:4;const n=a.length;let changed=0,listed=0,listedSame=0;const ch=new Uint8Array(n),re=new Uint8Array(n);
    let refined=0,flips=0;for(let t=0;t<n;t++){const o=w(a[t]!),m=w(b[t]!);if(o!==m){ch[t]=1;changed++;if(lastChanged?.[t])flips++;}if(m<o){re[t]=1;refined++;}}lastChanged=ch;
    for(let t=0;t<n;t++){const x=t%T[0],y=Math.floor(t/T[0])%T[1],z=Math.floor(t/(T[0]*T[1]));const fine=w(a[t]!)===1&&w(b[t]!)===1;let hit=false;
     for(let dz=-1;dz<=1&&!hit;dz++)for(let dy=-1;dy<=1&&!hit;dy++)for(let dx=-1;dx<=1&&!hit;dx++){const X=x+dx,Y=y+dy,Z=z+dz;if(X<0||Y<0||Z<0||X>=T[0]||Y>=T[1]||Z>=T[2])continue;const q=X+T[0]*(Y+T[1]*Z);if(fine?re[q]:ch[q])hit=true;}
     if(hit){listed++;if(!ch[t])listedSame++;}}
    remapStats.push([changed,listed,listedSame,refined,flips]);}
   return ad(l);};}}
  const residency:number[][]=[];const quality:any[]=[];const walls:number[]=[],phases:Record<string,number[]>={},totals:number[]=[],fine:number[]=[],cycles:number[]=[];
  for(let step=1;step<=steps;step++){
   solver.applyRuntimeValues(values);
   if(timed&&step>=from)recorder=new GPUPassTimestampRecorder(real,4096,`frame ${step}`);
   const t0=performance.now();
   if(!solver.advanceTo(step/30,[]))throw new Error("advance refused");await solver.awaitFrameCompletion();
   const wall=performance.now()-t0;
   if(solver.info.simulationPipelineError)throw new Error(String(solver.info.simulationPipelineError));
   let reading:any;if(recorder){const r=recorder;recorder=undefined;const e=real.createCommandEncoder();r.resolve(e);real.queue.submit([e.finish()]);reading=await r.read();}
   if(step%10===0){const i=await solver.readStats();quality.push({step,volumeCellSum:i.volumeCellSum,representedVolumeDrift:i.representedVolumeDrift,volumeDrift:i.volumeDrift,maxSpeed_m_s:i.maxSpeed_m_s});}
   if(step<from){for(const k in grids)delete grids[k];continue;}gridFrames++;
   walls.push(wall);
   let tiles:Uint32Array|undefined;try{tiles=solver.mixedFrame?.ownership?.layout?.tiles;}catch{}
   if(tiles){let n=0;for(const w of tiles)if(w&0x80000000)n++;fine.push(n);}
   // Residency (w-page diagnostic, rep1 only, after the wall time): resident pages, violation bits, radius.
   if(timed){const o=solver.mixedFrame?.ownership;if(o){const n=o.capacity.tiles;const rb=real.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=real.createCommandEncoder();e.copyBufferToBuffer(o.support,(9*n+28)*4,rb,0,16);real.queue.submit([e.finish()]);await rb.mapAsync(GPUMapMode.READ);const w=new Uint32Array(rb.getMappedRange()).slice();rb.unmap();rb.destroy();residency.push([w[0]!,w[1]!,w[2]!]);}}
   if(reading){totals.push(reading.sum_ms);const g:Record<string,number>={};for(const q of reading.passes)g[q.label]=(g[q.label]??0)+q.duration_ms;for(const [k,v] of Object.entries(g))(phases[k]??=[]).push(v);}
  }
  const med=(a:number[])=>{const s=[...a].sort((x,y)=>x-y);return s.length?+s[s.length>>1]!.toFixed(2):null;};
  const mean=(a:number[])=>a.length?+(a.reduce((x,y)=>x+y,0)/a.length).toFixed(3):null;
  const tail=(a:number[])=>a.slice(from-1);const arm={quality,reused,relayouts,band:{tilesMean:mean(tail(bandTiles)),tilesMax:Math.max(...tail(bandTiles)),cyclesMean:mean(tail(bandCycles)),cyclesHist:tail(bandCycles).reduce((h:Record<number,number>,c)=>(h[c]=(h[c]??0)+1,h),{}),residualMedian:med(tail(bandResidual)),frames:bandTiles.length},...(residency.length?{residentPagesMean:mean(residency.map(r=>r[0]!)),residentPagesMax:Math.max(...residency.map(r=>r[0]!)),residencyRadiusMean:mean(residency.map(r=>r[2]!)),residencyViolation:residency.reduce((a,r)=>a|r[1]!,0)}:{}),...(remapStats.length?{remapChangedMean:mean(remapStats.map(r=>r[0]!)),remapListedMean:mean(remapStats.map(r=>r[1]!)),remapListedUnchangedMean:mean(remapStats.map(r=>r[2]!)),remapListedMax:Math.max(...remapStats.map(r=>r[1]!)),remapRefinedMean:mean(remapStats.map(r=>r[3]!)),remapFlipMean:mean(remapStats.map(r=>r[4]!))}:{}),wallMedian_ms:med(walls),wallMean_ms:mean(walls),fineTilesMean:mean(fine),
   ...(timed?{traceTotalMean_ms:mean(totals),traced:totals.length,phases:Object.fromEntries(Object.entries(phases).map(([k,v])=>[k,+(v.reduce((x,y)=>x+y,0)/walls.length).toFixed(3)]).sort((a,b)=>(b[1] as number)-(a[1] as number)))}:{})};
  result[`rep${rep}`]=arm;result.grids=Object.fromEntries(Object.entries(grids).map(([k,v])=>[k,{...v,callsPerFrame:v.calls/gridFrames,groupsPerFrame:v.groups/gridFrames}]));console.log(JSON.stringify({rep,...arm}));
  solver.destroy();solvers.pop();
 }
 writeFileSync(out,JSON.stringify(result,null,1));
}finally{for(const s of solvers)s.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
