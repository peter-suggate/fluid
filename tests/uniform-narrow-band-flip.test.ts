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
 const liveEdits:MethodParamValues[]=[{adaptiveBudgetPercent:0},{adaptiveBudgetPercent:100},{adaptiveFadeSeconds:0.05},{adaptiveFadeSeconds:2},{fineGridPadding:2},{sparseParticleLifetime:0},{sparseParticleLifetime:10},{particleCellCap:8},{particleCellCap:20},{particleCellCap:0}];
 for(const edit of liveEdits){
  const next={...config,values:{...values,...edit}};
  assert.equal(access.currentGPUFluid(defaultScene,next,"full-scene"),solver);
  assert.deepEqual(adopted.at(-1),next.values);
  assert.equal(solver.info.completedTime_s,7);assert.equal(solver.info.encodedSteps,420);
 }

});

test("sparse particle lifetime defaults, clamps and applies without rebuilding",()=>{
 const method=getMethod("uniform-narrow-band-flip");
 const value=(sparseParticleLifetime?:number)=>resolveMethodValues(method,"balanced",sparseParticleLifetime===undefined?{}:{sparseParticleLifetime}).sparseParticleLifetime;
 assert.equal(value(),2);assert.equal(value(-1),0);assert.equal(value(100),10);assert.equal(value(0.3),0.3);
 assert.equal(methodConfigurationImpact(method,"balanced",{},{sparseParticleLifetime:0.3}),"live");
});


test("crowded-cell cap defaults to 20 and is exposed as a live surface control",async()=>{
 const {narrowBandControlValues}=await import('../lib/methods/uniform/uniform-narrow-band-controls');
 const {UNIFORM_NARROW_BAND_PIPELINE}=await import('../lib/methods/uniform/uniform-narrow-band-pipeline');
 const method=getMethod("uniform-narrow-band-flip");
 assert.equal(narrowBandControlValues().particleCellCap,20);
 for(const [raw,want] of [[-1,0],[0,0],[12.9,12],[500,128]])assert.equal(narrowBandControlValues({particleCellCap:raw!}).particleCellCap,want);
 assert.equal(methodConfigurationImpact(method,"balanced",{},{particleCellCap:12}),"live");
 const control=UNIFORM_NARROW_BAND_PIPELINE.stages.find(s=>s.id==="uniform-volume-phi")?.controls?.find(c=>'param' in c&&c.param==='particleCellCap');
 assert.ok(control&&control.kind==='param-range'&&control.live);
});
