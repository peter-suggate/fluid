/** Manufactured staggered vortex: isolate face preparation before forces/projection. */
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { writeGPUBufferView } from "../lib/core/webgpu-buffer-upload";
import { gpuCompilationManagerFor, invalidateGPUCompilationManager, managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createAnalyticMotionScene } from "../lib/core/analytic-motion-scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
const arg = (name:string, fallback:string) => process.argv.find(v=>v.startsWith(`--${name}=`))?.slice(name.length+3) ?? fallback;
const cellWidth=Number(arg("cell-width","1")), mixed=process.argv.includes("--mixed");
assert.ok([1,2,4].includes(cellWidth));assert.ok(!mixed||cellWidth<=2,"mixed uses widths 1/2 or 2/4");
const dt = Number(arg("dt", "0.0000001")), mode = Number(arg("mode", cellWidth===1?"4":"1"));
const phaseX=Number(arg("phase-x",mixed?"1":"0"));assert.ok(Number.isFinite(phaseX));
const output = arg("output", "artifacts/analytic-motion/face-advection.json");
assert.ok(dt>0 && Number.isFinite(dt)); assert.ok(Number.isInteger(mode)&&mode>0&&mode<8);
const modulePath=process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js");
const files=["lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", "lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts"];
const fingerprint=()=>Object.fromEntries(files.map(file=>[file,createHash("sha256").update(readFileSync(file)).digest("hex")]));
const beforeFingerprint=fingerprint();
await acquireWebGPUExclusiveLock("dawn-probe", "face-advection-vortex");
let device:GPUDevice|undefined, solver:WebGPUAdaptiveMassSolver|undefined;
const captures=new Map<string,GPUBuffer>();
try {
 const dawn=await import(pathToFileURL(modulePath).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn as NodeDawnProvider,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
 const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),
  {requireWorkerRealm:false,maximumConcurrentBundles:1});
 const errors:string[]=[];device.addEventListener("uncapturederror",event=>errors.push(event.error.message));
 const scene=createAnalyticMotionScene("translation");
 scene.sceneId="manufactured-face-advection-vortex";scene.container.fillFraction=1;
 scene.fluid.initialCondition="tank-fill";delete scene.fluid.initialDamBreakDimensions_m;delete scene.fluid.initialDamBreakOrigin_m;
 scene.fluid.initialVelocity_m_s={x:0,y:0,z:0};scene.fluid.gravity_m_s2={x:0,y:0,z:0};
 scene.fluid.refinementRegions=mixed?[
  {id:"left-fine",rule:"minimum-cell-size",minimumCellSize_cells:cellWidth,maximumCellSize_cells:cellWidth,
   min_m:{x:-.8,y:0,z:-.2},max_m:{x:0,y:1.6,z:.2}},
  {id:"right-coarse",rule:"minimum-cell-size",minimumCellSize_cells:2*cellWidth,maximumCellSize_cells:2*cellWidth,
   min_m:{x:0,y:0,z:-.2},max_m:{x:.8,y:1.6,z:.2}},
 ]:[{id:"uniform",rule:"minimum-cell-size",minimumCellSize_cells:cellWidth,maximumCellSize_cells:cellWidth,
  min_m:{x:-.8,y:0,z:-.2},max_m:{x:.8,y:1.6,z:.2}}];
 scene.numerics.fixedDt_s=scene.numerics.maxDt_s=dt;
 const values=resolveMethodValues(adaptiveMassMethod,"balanced",{selectorMode:"coarse-first",maximumMacroSpanBricks:"1",timeStep:"scene"});
 solver=await adaptiveMassMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUAdaptiveMassSolver;
 await solver.waitForSimulationReady();
 const source=solver.fieldSnapshotSourceForQA,w=source.templateWords,f=new Float32Array(w.buffer,w.byteOffset,w.length);
 const activity=(await solver.readGPUActivityPolicy()).bricks;
 const nr=w[3]!, nc=w[2]!, rowBase=w[7]!, termBase=w[8]!, k=2*Math.PI*mode/32;
 const enabled=new Uint8Array(nr), faces=new Float32Array(nr), velocities=new Float32Array(4*nc);
 const rowPosition=(row:number)=>[f[rowBase+6*nr+row]!,f[rowBase+7*nr+row]!,f[rowBase+8*nr+row]!];
 for(let row=0;row<nr;row++){
  const meta=w[rowBase+nr+row]!, requirements=meta&0x0fffffff;
  enabled[row]=Number(Array.from({length:w[requirements]!},(_,j)=>w[requirements+1+j]!).every(m=>
   activity[m>>>5]?.active&&activity[m>>>5]?.acceptedResolution===(m&31)));
  const [x,y]=rowPosition(row);const axis=meta>>>30;
  faces[row]=axis===0?Math.sin(k*(x!+phaseX))*Math.cos(k*y!):axis===1?-Math.cos(k*(x!+phaseX))*Math.sin(k*y!):0;
 }
 let seededCells=0;
 for(const record of activity){if(!record.active)continue;
  const range=w[11]!+2*(4*record.leafId+Math.log2(record.acceptedResolution));
  assert.ok([8/cellWidth,...(mixed?[4/cellWidth]:[])].includes(record.acceptedResolution),
   `unexpected accepted resolution ${record.acceptedResolution}`);
  for(let cell=w[range]!;cell<w[range]!+w[range+1]!;cell++){
   const weight=[0,0,0], sum=[0,0,0];
   for(let at=w[w[9]!+cell]!;at<w[w[9]!+cell+1]!;at++){
    const row=w[w[10]!+2*at]!, term=w[w[10]!+2*at+1]!;if(!enabled[row])continue;
    const axis=w[rowBase+nr+row]!>>>30, a=Math.abs(f[termBase+2*term+1]!)*f[rowBase+2*nr+row]!;
    weight[axis]!+=a;sum[axis]!+=a*faces[row]!;
   }
   for(let axis=0;axis<3;axis++)velocities[4*cell+axis]=weight[axis]!>0?sum[axis]!/weight[axis]!:0;
   seededCells++;
  }
 }
 for(const base of [source.layout.faceA,source.layout.faceB])writeGPUBufferView(device.queue,source.state,4*base,faces);
 for(const base of [source.layout.cellVelocityA,source.layout.cellVelocityB])writeGPUBufferView(device.queue,source.state,4*base,velocities);
 solver.setStageCaptureForQA((stage,encoder)=>{
  if(stage!=="transport-velocity-extension"&&stage!=="face-preparation")return;
  const buffer=device!.createBuffer({size:8*nr+4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  encoder.copyBufferToBuffer(source.state,4*source.layout.faceA,buffer,0,4*nr);
  encoder.copyBufferToBuffer(source.state,4*source.layout.faceB,buffer,4*nr,4*nr);
  encoder.copyBufferToBuffer(source.topologyArena,4*(source.frameControlBaseWords+source.faceParityWord),buffer,8*nr,4);
  captures.set(stage,buffer);
 });
 assert.equal(solver.advanceTo(dt,[]),true,"initial frame admission");
 await solver.awaitFrameCompletion();solver.setStageCaptureForQA(undefined);
 // Classify immutable accepted physical rows independently of their seeded values.
 const rowClass=new Array<string>(nr), eligible=new Uint8Array(nr), seamClass=new Array<string>(nr);
 const acceptedResolutionCounts:Record<string,number>={};
 for(const record of activity)if(record.active)acceptedResolutionCounts[record.acceptedResolution]=
  (acceptedResolutionCounts[record.acceptedResolution]??0)+1;
 for(let row=0;row<nr;row++){
  if(!enabled[row])continue;
  const packed=w[rowBase+row]!, first=packed&0x007fffff, count=packed>>>23;
  const axis=w[rowBase+nr+row]!>>>30, u=(axis+1)%3,v=(axis+2)%3;
  const widths:number[]=[];let positive=false,negative=false,whole=false;
  for(let term=first;term<first+count;term++){
   const cell=w[termBase+2*term]!, base=w[6]!+8*cell, coefficient=f[termBase+2*term+1]!;
   widths.push(Math.min(f[base+4]!,f[base+5]!,f[base+6]!));
   positive ||= coefficient>0;negative ||= coefficient<0;
   if(count===1){const position=rowPosition(row);
    whole=Math.abs(coefficient)*f[rowBase+2*nr+row]===f[base+4+u]!*f[base+4+v]!
     &&position[u]===f[base+u]&&position[v]===f[base+v]
     &&position[axis]===f[base+axis]!+(coefficient<0?.5:-.5)*f[base+4+axis]!;
   }
  }
  const min=Math.min(...widths),max=Math.max(...widths);
  rowClass[row]=count===1?(whole?"whole-one-sided":"partial-one-sided"):
   min!==max?"mortar":`uniform-width-${min}`;
  eligible[row]=Number((positive&&negative)||whole);
  const distance=Math.abs(rowPosition(row)[0]!-16);
  seamClass[row]=!mixed?"no-seam":distance===0?"on-seam":distance<=2*cellWidth?"near-seam":"far-seam";
 }
 const receipts=[];
 for(const [stage,buffer] of captures){await buffer.mapAsync(GPUMapMode.READ);const mapped=buffer.getMappedRange();
  const data=new Float32Array(mapped), parity=new Uint32Array(mapped,8*nr,1)[0]!&1;
  const selected=parity^Number(stage==="face-preparation");
  type Acc={count:number;energy:number;initialEnergy:number;squaredError:number;correlation:number;maxAbsoluteError:number};
  const groups=new Map<string,Acc>();
  const add=(key:string,a:number,b:number)=>{const value=groups.get(key)??
   {count:0,energy:0,initialEnergy:0,squaredError:0,correlation:0,maxAbsoluteError:0};
   value.count++;value.energy+=b*b;value.initialEnergy+=a*a;value.squaredError+=(b-a)**2;
   value.correlation+=a*b;value.maxAbsoluteError=Math.max(value.maxAbsoluteError,Math.abs(b-a));groups.set(key,value);};
  for(let row=0;row<nr;row++){if(!enabled[row]||(w[rowBase+nr+row]!>>>30)>1)continue;
   const [x,y,z]=rowPosition(row),a=faces[row]!, b=data[selected*nr+row]!;assert.ok(Number.isFinite(b));
   const interior=x!>=2&&x!<=30&&y!>=2&&y!<=30&&z!>=1&&z!<=7;
   add(`all/${rowClass[row]}`,a,b);
   if(!interior)continue;
   add("interior",a,b);add(`interior/${rowClass[row]}`,a,b);add(`interior/${seamClass[row]}`,a,b);
   if(eligible[row])add("certified-interior",a,b);
  }
  const summarize=(value:Acc)=>({count:value.count,initialEnergy:value.initialEnergy,
   energyRatio:value.initialEnergy>1e-20?value.energy/value.initialEnergy:null,
   relativeL2Error:value.initialEnergy>1e-20?Math.sqrt(value.squaredError/value.initialEnergy):null,
   amplitudeRatio:value.initialEnergy>1e-20?value.correlation/value.initialEnergy:null,maxAbsoluteError:value.maxAbsoluteError});
  const overall=groups.get("interior");assert.ok(overall&&overall.count>0&&overall.initialEnergy>1e-20);
  const certified=groups.get("certified-interior");assert.ok(certified&&certified.count>0);
  const mortar=groups.get("interior/mortar");
  if(mixed)assert.ok(mortar&&mortar.count>0&&mortar.initialEnergy>1e-6,
   `Mixed probe must excite interior mortar rows: count=${mortar?.count??0}, initialEnergy=${mortar?.initialEnergy??0}`);
  receipts.push({stage,...summarize(overall),
   mortarExcitation:mixed?{count:mortar!.count,initialEnergy:mortar!.initialEnergy,minimumInitialEnergy:1e-6,passed:true}:null,groups:Object.fromEntries([...groups].map(([key,value])=>[key,summarize(value)])),
   zeroDtIdentityCriterion:dt<=1e-7?{scope:"certified supported interior two-sided or whole one-sided rows",
    maxAbsoluteErrorLimit:1e-5,passed:certified.maxAbsoluteError<=1e-5}:null});buffer.unmap();
 }
 assert.equal(receipts.length,2);assert.deepEqual(errors,[]);
 const result={dt,mode,phaseX,cellWidth,mixed,acceptedResolutionCounts,seededCells,field:"u=(sin(k(x+phaseX))cos(ky),-cos(k(x+phaseX))sin(ky),0), exact face samples and incident-face collocation",
  seedNote:mixed?"Point samples need not satisfy the mixed discrete divergence operator; this probe measures the prefix before projection.":"Uniform staggered mode is discretely solenoidal.",
  expectedZeroDtIdentity:{amplitudeRatio:1,energyRatio:1},
  predictedCollocatedRoundTrip:mixed?null:{amplitudeRatio:Math.cos(k*cellWidth/2)**2,energyRatio:Math.cos(k*cellWidth/2)**4},receipts,
  sourceFingerprint:beforeFingerprint,sourceFingerprintAfter:fingerprint(),validationErrors:errors};
 mkdirSync(output.slice(0,output.lastIndexOf("/"))||".",{recursive:true});writeFileSync(output,JSON.stringify(result,null,2)+"\n");
 console.log(JSON.stringify(result,null,2));
} finally {
 for(const buffer of captures.values())buffer.destroy();
 if(device){const manager=gpuCompilationManagerFor(device);
  await manager.whenIdle();await device.queue.onSubmittedWorkDone();solver?.destroy();solver=undefined;
  invalidateGPUCompilationManager(device,"Face-advection probe complete");
  await manager.whenIdle();await device.queue.onSubmittedWorkDone();device.destroy();
 }
 await releaseWebGPUExclusiveLock();
}
