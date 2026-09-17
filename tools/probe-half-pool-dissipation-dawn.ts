/** Read-only half-pool energy, geometry and velocity-stage investigation. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { sceneAtContainerExtents } from "../lib/core/scene-scale";
import { getScenePreset } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sparseCM12DawnDefaultOptions } from "../lib/harness/sparse-cm12-dawn-defaults";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { createGeometricDamStageEnergy } from "./geometric-dam-stage-energy";
const arg = (name:string,fallback:string) => process.argv.find(v=>v.startsWith(`--${name}=`))?.slice(name.length+3)??fallback;
const steps=Number(arg("steps","60")), extruded=arg("extruded","0")==="1", freeze=arg("freeze","0")==="1";
const dt=Number(arg("dt",String(1/30))), sharpen=arg("sharpen","1")!=="0";
const airExtensionEnabled=arg("air","0")==="1";
const output=arg("output","artifacts/level-set-volume/half-pool-dissipation-3d.json");
const tapSteps=new Set(arg("taps","5,10,15,20,30,45,60").split(",").map(Number));
await acquireWebGPUExclusiveLock("dawn-probe","half-pool dissipation");
let device:GPUDevice|undefined,solver:WebGPUAdaptiveMassSolver|undefined;
const report:any={steps,dt,extruded,freeze,sharpen,airExtensionEnabled,checkpoints:[],stageEnergy:[]};
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 report.validationErrors=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();report.validationErrors.push(e.error.message);});
 let scene=getScenePreset("coarse-first-pool-impact-half").create();
 if(extruded){scene=sceneAtContainerExtents(scene,{width_m:3.2,height_m:2.4,depth_m:0.4});scene.container.depthBoundary="symmetry";scene.fluid.initialLiquidVolumes=[{shape:"cylinder",center_m:{x:0,y:1.825,z:0},radius_m:0.5,halfHeight_m:0.4}];}
 scene.numerics.fixedDt_s=scene.numerics.maxDt_s=dt;
 const options={...sparseCM12DawnDefaultOptions(),surfaceSharpeningEnabled:sharpen,airExtensionEnabled};
 solver=await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(device,scene,"balanced",undefined,options,()=>{});
 await solver.waitForSimulationReady();if(freeze)solver.setTopologyFrozen(true);
 const h=scene.voxelDomain.finestCellSize_m;const {nx,ny,nz}=solver.info;report.dimensions=[nx,ny,nz];report.h=h;report.scene=scene;
 for(let step=0;step<=steps;step++){
  if(step){
   const tap=tapSteps.has(step)?await createGeometricDamStageEnergy(device,solver,h,1):undefined;
   tap?.arm();
   while(!solver.advanceTo(step*dt,[]))await new Promise<void>(setImmediate);
   await solver.awaitFrameCompletion();await solver.waitForTopologyReady();
   if(tap)report.stageEnergy.push({step,...await tap.read()});
  }
  if(step%5!==0&&step!==steps)continue;
  const [fields,volume,activity,transport,stats]=await Promise.all([solver.readDiagnosticFields(true),solver.readAcceptedGeometricVolumeQA(),solver.readGPUActivityPolicy(),solver.readGeometricVolumeTransportReceiptQA(),solver.readStats()]);
  assert.ok(volume.outsideAuthoredVolumeFine3<1e-3*volume.volumeFine3, "energy window excludes material outside authored domain");
  let mass=0,potential=0;const kinetic=[0,0,0],momentum=[0,0,0];let abovePool=0,meanY=0,maxY=0;
  for(let i=0;i<fields.density.length;i++){
   const m=fields.density[i]!*h**3;const y=(Math.floor(i/nx)%ny+0.5)*h;mass+=m;potential+=m*9.81*y;meanY+=m*y;
   if(y>0.8)abovePool+=m;if(fields.density[i]!>0.5)maxY=Math.max(maxY,y);
   for(let a=0;a<3;a++){const v=fields.velocity[3*i+a]!;kinetic[a]!+=0.5*m*v*v;momentum[a]!+=m*v;}
  }
  const active=activity.bricks.filter(b=>b.active);
  const air=await solver.readAirExtensionReceiptQA();
  report.checkpoints.push({air,step,time:step*dt,mass,kinetic,potential,mechanical:potential+kinetic.reduce((a,b)=>a+b,0),meanVelocity:momentum.map(v=>v/mass),meanY:meanY/mass,abovePool,maxY,volume,stats,coupling:transport.coupling,resolutions:Object.fromEntries([1,2,4,8].map(r=>[r,active.filter(b=>b.acceptedResolution===r).length]))});
 }
 report.completed=true;
}catch(e){report.completed=false;report.error=String(e);process.exitCode=1;}
finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();mkdirSync(dirname(output),{recursive:true});writeFileSync(output,JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify({output,completed:report.completed,error:report.error}));}
