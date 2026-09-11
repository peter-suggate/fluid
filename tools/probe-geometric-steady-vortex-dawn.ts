/** Full-pipeline steady Euler vortex: measure finite-step advection and projection losses. */
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { writeGPUBufferView } from "../lib/core/webgpu-buffer-upload";
import { gpuCompilationManagerFor, invalidateGPUCompilationManager, managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createAnalyticMotionScene } from "../lib/core/analytic-motion-scenes";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
const arg = (name:string, fallback:string) => process.argv.find(v=>v.startsWith(`--${name}=`))?.slice(name.length+3) ?? fallback;
const dt=Number(arg("dt",String(1/60))), duration=Number(arg("duration",String(1/60)));
const output=arg("output","artifacts/analytic-motion/steady-vortex.json");
assert.ok(dt>0&&duration>0&&Number.isFinite(dt)&&Number.isFinite(duration));
const dx=.05,L=.8,U=6;
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
 scene.sceneId="geometric-steady-vortex";scene.duration_s=duration;
 scene.container={...scene.container,width_m:L,height_m:L,depth_m:.4,fillFraction:1,top:"closed",depthBoundary:"closed",fluidWallMode:"free-slip"};
 scene.fluid.initialCondition="tank-fill";delete scene.fluid.initialDamBreakDimensions_m;delete scene.fluid.initialDamBreakOrigin_m;
 scene.fluid.initialVelocity_m_s={x:0,y:0,z:0};scene.fluid.gravity_m_s2={x:0,y:0,z:0};
 scene.fluid.dynamicViscosity_Pa_s=0;scene.fluid.surfaceTension_N_m=0;
 scene.fluid.refinementRegions=[{id:"all-fine",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,
  min_m:{x:-L/2,y:0,z:-.2},max_m:{x:L/2,y:L,z:.2}}];
 scene.solidVoxels=[...solidVoxelShellForScene(scene)];
 scene.numerics.fixedDt_s=scene.numerics.maxDt_s=dt;
 const values=resolveMethodValues(adaptiveMassMethod,"balanced",{selectorMode:"coarse-first",maximumMacroSpanBricks:"1",timeStep:"scene"});
 solver=await adaptiveMassMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUAdaptiveMassSolver;
 await solver.waitForSimulationReady();
 solver.setTopologyFrozen(true);
 const source=solver.fieldSnapshotSourceForQA,w=source.templateWords,f=new Float32Array(w.buffer,w.byteOffset,w.length);
 const activity=(await solver.readGPUActivityPolicy()).bricks;
 const nr=w[3]!, nc=w[2]!, rowBase=w[7]!, termBase=w[8]!, k=Math.PI/16;
 const enabled=new Uint8Array(nr), faces=new Float32Array(nr), velocities=new Float32Array(4*nc);
 const rowPosition=(row:number)=>[f[rowBase+6*nr+row]!,f[rowBase+7*nr+row]!,f[rowBase+8*nr+row]!];
 for(let row=0;row<nr;row++){
  const meta=w[rowBase+nr+row]!, requirements=meta&0x0fffffff;
  enabled[row]=Number(Array.from({length:w[requirements]!},(_,j)=>w[requirements+1+j]!).every(m=>
   activity[m>>>5]?.active&&activity[m>>>5]?.acceptedResolution===(m&31)));
  const [x,y]=rowPosition(row);const axis=meta>>>30;
  faces[row]=(U/dx)*(axis===0?Math.sin(k*x!)*Math.cos(k*y!):axis===1?-Math.cos(k*x!)*Math.sin(k*y!):0);
 }
 let seededCells=0;
 for(const record of activity){if(!record.active)continue;
  const range=w[11]!+2*(4*record.leafId+Math.log2(record.acceptedResolution));
  assert.equal(record.acceptedResolution,8,"all-fine required");
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
 const stages=["transport-velocity-extension","face-preparation","body-forces","velocity-projection"];
 const activeCells:number[]=[];
 for(const record of activity){if(!record.active)continue;const range=w[11]!+2*(4*record.leafId+3);
  for(let cell=w[range]!;cell<w[range]!+w[range+1]!;cell++)activeCells.push(cell);}
 const summarize=(data:Float32Array)=>{
  let energy=0,initialEnergy=0,error=0,correlation=0,maxVelocity=0,maxDivergence=0,divergenceSquared=0;
  for(let row=0;row<nr;row++){if(!enabled[row])continue;
   const weight=f[rowBase+2*nr+row]!*dx**3,a=faces[row]!*dx,b=data[row]!*dx;
   assert.ok(Number.isFinite(b));energy+=weight*b*b;initialEnergy+=weight*a*a;
   error+=weight*(b-a)**2;correlation+=weight*a*b;maxVelocity=Math.max(maxVelocity,Math.abs(b));}
  for(const cell of activeCells){let divergence=0;
   for(let at=w[w[9]!+cell]!;at<w[w[9]!+cell+1]!;at++){
    const row=w[w[10]!+2*at]!,term=w[w[10]!+2*at+1]!;if(!enabled[row])continue;
    divergence+=f[termBase+2*term+1]!*f[rowBase+2*nr+row]!*data[row]!;
   }
   divergence/=f[w[6]!+8*cell+3]!;
   maxDivergence=Math.max(maxDivergence,Math.abs(divergence));divergenceSquared+=divergence**2;
  }
  return {kinetic_J:.5*scene.fluid.density_kg_m3*energy,energyRatio:energy/initialEnergy,
   amplitudeRatio:correlation/initialEnergy,relativeFieldL2:Math.sqrt(error/initialEnergy),maxVelocity_m_s:maxVelocity,
   maxDivergence_s_inverse:maxDivergence,rmsDivergence_s_inverse:Math.sqrt(divergenceSquared/activeCells.length)};
 };
 const initial=summarize(faces),continuumInitialKE_J=scene.fluid.density_kg_m3*U*U*L*L*.4/4;
 assert.ok(Math.abs(initial.kinetic_J-continuumInitialKE_J)<1e-5*continuumInitialKE_J,"discrete face quadrature must match independent continuum energy");
 const frames=[];let time=0;
 while(time<duration-1e-10){
  solver.setStageCaptureForQA((stage,encoder)=>{
   if(!stages.includes(stage))return;
   const buffer=device!.createBuffer({size:8*nr+4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   encoder.copyBufferToBuffer(source.state,4*source.layout.faceA,buffer,0,4*nr);
   encoder.copyBufferToBuffer(source.state,4*source.layout.faceB,buffer,4*nr,4*nr);
   encoder.copyBufferToBuffer(source.topologyArena,4*(source.frameControlBaseWords+source.faceParityWord),buffer,8*nr,4);
   captures.set(stage,buffer);
  });
  const beforeTime=time,target=Math.min(duration,time+dt);
  assert.equal(solver.advanceTo(target,[]),true,"fixed all-fine frame admission");
  await solver.awaitFrameCompletion();solver.setStageCaptureForQA(undefined);
  const receipts=[];
  for(const [stage,buffer] of captures){await buffer.mapAsync(GPUMapMode.READ);const mapped=buffer.getMappedRange();
   const parity=new Uint32Array(mapped,8*nr,1)[0]!&1;
   const selected=parity^Number(stage!==stages[0]);
   receipts.push({stage,...summarize(new Float32Array(mapped,4*selected*nr,nr))});
   buffer.unmap();buffer.destroy();}
  captures.clear();assert.equal(receipts.length,4);
  const physical=await solver.readAcceptedGeometricVolumeQA(),stats=await solver.readStats();
  assert.equal(physical.invalidCells,0,"accepted volume bounds");
  assert.equal(physical.nonfiniteCells,0);assert.equal(physical.nonfiniteDynamicsCells,0);
  assert.equal(physical.zeroCapacityNonzeroVolumeCells,0);
  const accepted=await solver.readGPUActivityPolicy();assert.ok(accepted.bricks.every(b=>!b.active||b.acceptedResolution===8));
  assert.equal(solver.fieldSnapshotSourceForQA.state,source.state,"topology/state changed during fixed probe");
  time=stats.completedTime_s!;assert.ok(Number.isFinite(time)&&time>beforeTime,"accepted clock must advance");
  frames.push({beforeTime_s:beforeTime,afterTime_s:time,receipts,physical,stats});
 }
 assert.deepEqual(errors,[]);
 const result={probe:"geometric-steady-vortex",dt,duration,acceptedTime_s:time,dimensions:[16,16,8],dx,L,U,
  completion:true,boundsPassed:true,finitePassed:true,analytic:"steady Euler vortex; closed free-slip full tank; k=pi/L",
  initial,continuumInitialKE_J,leadingSplittingEnergyLossPerStep:.5*(U*Math.PI/L*dt)**2,
  note:"No analytic finite-step pass threshold: advection/projection splitting and interpolation both contribute. Initial energy uses face dual-volume quadrature; pressure starts at zero.",
  frames,sourceFingerprint:beforeFingerprint,sourceFingerprintAfter:fingerprint(),validationErrors:errors};
 assert.deepEqual(result.sourceFingerprint,result.sourceFingerprintAfter);
 mkdirSync(output.slice(0,output.lastIndexOf("/"))||".",{recursive:true});writeFileSync(output,JSON.stringify(result,null,2)+"\n");
 console.log(JSON.stringify(result,null,2));
} finally {
 for(const buffer of captures.values())buffer.destroy();
 if(device){const manager=gpuCompilationManagerFor(device);
  await manager.whenIdle();await device.queue.onSubmittedWorkDone();solver?.destroy();solver=undefined;
  invalidateGPUCompilationManager(device,"Steady vortex probe complete");
  await manager.whenIdle();await device.queue.onSubmittedWorkDone();device.destroy();
 }
 await releaseWebGPUExclusiveLock();
}
