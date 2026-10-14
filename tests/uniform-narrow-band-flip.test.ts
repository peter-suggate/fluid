import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { defaultMethodId, getMethod, registeredSimulationMethods } from "../lib/core/method-registry";
import { methodConfigurationImpact } from "../lib/core/method-lifecycle";
import { resolveMethodValues } from "../lib/core/method-contract";
import { narrowBandAdaptiveDetail } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { FluidLabRenderer, type SimulationRunConfig } from "../lib/core/webgpu-renderer";
import { defaultScene, type SceneDescription } from "../lib/core/model";
import type { MethodParamValues } from "../lib/core/method-contract";

test("NB runtime tuning keeps the renderer's attached solver and never starts initialization",()=>{
 const method=getMethod("uniform-narrow-band-flip");
 const values=resolveMethodValues(method,"balanced",{...method.appDefaults});
 const config:SimulationRunConfig={methodId:method.id,quality:"balanced",values,simulationEpoch:3};
 const adopted:MethodParamValues[]=[];
 const solver={info:{completedTime_s:7,encodedSteps:420},applyRuntimeValues(next:MethodParamValues){adopted.push(next);}};
 const renderer=new FluidLabRenderer({} as HTMLCanvasElement,()=>{assert.fail("Runtime tuning must not announce initialization");});
 const access=renderer as unknown as {
  solverKey(scene:SceneDescription,config:SimulationRunConfig,mode:"full-scene"):string;
  currentGPUFluid(scene:SceneDescription,config:SimulationRunConfig,mode:"full-scene"):unknown;
 };
 const key=access.solverKey(defaultScene,config,"full-scene");
 Object.assign(renderer,{device:{},gpuFluid:solver,gpuFluidKey:key,topologyFreezeSolver:solver,topologyFrozen:false,
  beginGPUFluidInitialization(){assert.fail("A budget/retirement edit rebuilt the attached solver");},
 });
 const liveEdits:MethodParamValues[]=[{adaptiveBudgetPercent:0},{adaptiveBudgetPercent:100},{adaptiveFadeSeconds:0.05},{adaptiveFadeSeconds:2},{fineGridPadding:2},{sparseParticleLifetime:0},{sparseParticleLifetime:10}];
 for(const edit of liveEdits){
  const next={...config,values:{...values,...edit}};
  assert.equal(access.currentGPUFluid(defaultScene,next,"full-scene"),solver);
  assert.deepEqual(adopted.at(-1),next.values);
  assert.equal(solver.info.completedTime_s,7);assert.equal(solver.info.encodedSteps,420);
 }

});

test("narrow-band FLIP pins its application modes and preserves runtime tuning",async()=>{
 const method=getMethod("uniform-narrow-band-flip");
 assert.deepEqual(registeredSimulationMethods(),[method]);
 assert.equal(defaultMethodId(),method.id);
 assert.equal((await method.pipelineGraph!()).methodId,method.id);
 const defaults=resolveMethodValues(method,"balanced",{});
 const values=resolveMethodValues(method,"balanced",{coarseParticleMode:"on",adaptiveSurface:"off",detailPolicy:"full",pressureFullCycles:1,pressureVCycles:1,pressureSweeps:1,sharpeningSweeps:8});
 for(const [key,value] of Object.entries({detailPolicy:"dynamic",adaptiveSurface:"on",coarseParticleMode:"off",pressureFullCycles:3,pressureVCycles:4,pressureSweeps:6,sharpeningSweeps:0,sharpeningDistance:0,phiDrain:"off",volumeDustThreshold:0,orphanDustThreshold:0})){
  assert.equal(defaults[key],value,key);assert.equal(values[key],value,key);
  assert.ok(!method.params.some(p=>p.key===key),`pinned ${key} has no control`);
 }
 for(const [key,value] of Object.entries(narrowBandAdaptiveDetail))assert.equal(defaults[key],value,key);
 assert.equal(defaults.fineGridPadding,1);
 assert.equal(methodConfigurationImpact(method,"balanced",{},{adaptiveBudgetPercent:25,adaptiveFadeSeconds:0.2,fineGridPadding:2}),"live");
 const tuned=resolveMethodValues(method,"balanced",{adaptiveBudgetPercent:200,adaptiveFadeSeconds:-1,detailShapeTolerance:0.75,detailStrain:"off"});
 assert.equal(tuned.adaptiveBudgetPercent,100);assert.equal(tuned.adaptiveFadeSeconds,0.05);
 assert.equal(tuned.detailShapeTolerance,0.75);assert.equal(tuned.detailStrain,"off");
 const options=uniformGeometricSolverOptions(values);
 assert.equal("sharpeningSweeps" in options,false);assert.equal(options.pressureCycleBudget,"lagged");
});

test("sparse particle lifetime defaults, clamps and applies without rebuilding",()=>{
 const method=getMethod("uniform-narrow-band-flip");
 const value=(sparseParticleLifetime?:number)=>resolveMethodValues(method,"balanced",sparseParticleLifetime===undefined?{}:{sparseParticleLifetime}).sparseParticleLifetime;
 assert.equal(value(),2);assert.equal(value(-1),0);assert.equal(value(100),10);assert.equal(value(0.3),0.3);
 assert.equal(methodConfigurationImpact(method,"balanced",{},{sparseParticleLifetime:0.3}),"live");
});
