import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import type {GPUSolverInstance} from "../lib/core/method-contract";
import {readMixedTexture,readMixedTileWords} from "./helpers/uniform-mixed-native-fields";
import {uniformQualityCensus} from "../tools/uniform-quality-census";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("experimental smooth 4h surface survives long-dam impact with conserved advancing liquid",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Explicit smooth coarse surface experiment");
 let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=sceneDocument(getSceneDefinition("sparse-cm12-ladder-long-dam"));
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({timeStep:"scene",detailShapeTolerance:0.5},scene),()=>{});
  const fields=(solver as any).mixedFrame.fields;
  const census=async()=>uniformQualityCensus([192,96,32],await readMixedTileWords(device!,solver),
   await readMixedTexture(device!,fields.volume),await readMixedTexture(device!,fields.phi));
  let initialMass=0;
  for(let step=1;step<=120;step++){
   assert.ok(solver.advanceTo(step/60,[]));await solver.awaitFrameCompletion?.();
   assert.equal(solver.info.simulationPipelineError,undefined);
   if(step===79){
    assert.equal(solver.info.uniformPressureCyclesConverged,true,"The formerly underplanned impact must converge");
    assert.equal(solver.info.uniformPressureCyclesEncoded,3,"Impact reserve must fit without encoding all seven slots");
   }
   if(![1,24,36,120].includes(step))continue;
   const q=await census();if(step===1)initialMass=q.mass;
   assert.equal(q.nonfinite,0);assert.ok(q.negative<1e-6);
   assert.ok(Math.abs(q.mass/initialMass-1)<0.005,"Accounted liquid loss must stay below 0.5%");
   if(step===24)assert.ok(q.owners<192*96*32/20,"Smooth surface must actually permit mostly coarse work");
   if(step===36)assert.ok(q.massFront_cells.p99!>150&&q.massFront_cells.p99!<180,"Mass front must traverse the tank before impact");
  }
  assert.deepEqual(errors,[]);
 }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
