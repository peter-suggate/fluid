import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { defaultMethodId, getMethod, interactiveSimulationMethods } from "../lib/core/method-registry";
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
 const liveEdits:MethodParamValues[]=[{adaptiveBudgetPercent:0},{adaptiveBudgetPercent:100},{adaptiveFadeSeconds:0.05},{adaptiveFadeSeconds:2},{fineGridPadding:2}];
 for(const edit of liveEdits){
  const next={...config,values:{...values,...edit}};
  assert.equal(access.currentGPUFluid(defaultScene,next,"full-scene"),solver);
  assert.deepEqual(adopted.at(-1),next.values);
  assert.equal(solver.info.completedTime_s,7);assert.equal(solver.info.encodedSteps,420);
 }
 const structuralEdits:MethodParamValues[]=[{adaptiveSurface:"off"},{coarseParticleMode:"on"}];
 for(const edit of structuralEdits){
  assert.notEqual(access.solverKey(defaultScene,{...config,values:{...values,...edit}},"full-scene"),key,"allocation modes still rebuild");
 }
});

test("narrow-band FLIP keeps the fixed reference, opens adaptive in the app, and makes all-4h opt-in",async()=>{
 const method=getMethod("uniform-narrow-band-flip");
 assert.ok(interactiveSimulationMethods().includes(method));assert.equal(defaultMethodId(),"uniform-narrow-band-flip");
 assert.equal((await method.harness!()).methodId,method.id);assert.equal((await method.pipelineGraph!()).methodId,method.id);
 const values=resolveMethodValues(method,"balanced",{coarseParticleMode:"on",detailPolicy:"requested",detailShape:"off",detailShapeTolerance:2,sharpeningSweeps:8});
 assert.equal(values.detailPolicy,"requested");assert.equal(values.detailShape,"off");assert.equal(values.detailShapeTolerance,2);
 const defaults=resolveMethodValues(method,"balanced",{});
 assert.equal(defaults.coarseParticleMode,"off");assert.equal(defaults.detailPolicy,"dynamic");assert.equal(defaults.detailSolidContact,"on");
 assert.equal(defaults.detailSurface,"on");assert.equal(defaults.detailSurfaceDistance,0);assert.equal(defaults.detailShapeTolerance,0);
 for(const key of ["detailThin","detailStrain","detailRotation","detailImpact","detailApproach","detailNearFocus","detailBulk"])assert.equal(defaults[key],"off",key);
 assert.equal(defaults.detailShape,"on");assert.equal(defaults.detailMarginTiles,0);assert.equal(defaults.detailHoldSteps,0);
 for(const key of Object.keys(defaults).filter(k=>k.startsWith("detail")))assert.equal(method.appDefaults![key]??defaults[key],narrowBandAdaptiveDetail[key]??defaults[key],`app default ${key}`);
 assert.equal(defaults.fineGridPadding,1);
 assert.equal(method.appDefaults!.adaptiveSurface,"on");
 assert.equal(method.appDefaults!.adaptiveBudgetPercent,50);
 assert.equal(method.appDefaults!.adaptiveFadeSeconds,0.5);
 assert.equal(methodConfigurationImpact(method,"balanced",{},{adaptiveBudgetPercent:25,adaptiveFadeSeconds:0.2}),"live");
 const bounded=resolveMethodValues(method,"balanced",{adaptiveBudgetPercent:200,adaptiveFadeSeconds:-1});
 assert.equal(bounded.adaptiveBudgetPercent,100);assert.equal(bounded.adaptiveFadeSeconds,0.05);
 assert.equal(defaults.adaptiveSurface,"off","the fixed band remains the reference configuration");
 assert.equal(methodConfigurationImpact(method,"balanced",{},{adaptiveSurface:"on"}),"rebuild");
 const adaptive=resolveMethodValues(method,"balanced",{adaptiveSurface:"on",detailShapeTolerance:0.5,detailShapeMetric:"displacement"});
 assert.equal(adaptive.adaptiveSurface,"on");assert.equal(adaptive.detailShapeTolerance,0.5);assert.equal(adaptive.detailShapeMetric,"displacement");
 assert.equal(methodConfigurationImpact(method,"balanced",{},{fineGridPadding:2}),"live");
 const selected=resolveMethodValues(method,"balanced",{detailPolicy:"requested",detailShape:"off",detailShapeTolerance:2});
 assert.equal(selected.detailPolicy,"requested");assert.equal(selected.detailShape,"off");assert.equal(selected.detailShapeTolerance,2);
 for(const detailPolicy of ["requested","dynamic","full"])assert.equal(resolveMethodValues(method,"balanced",{detailPolicy}).detailPolicy,detailPolicy);
 assert.equal(methodConfigurationImpact(method,"balanced",{},{detailPolicy:"requested"}),"live");
 assert.equal(methodConfigurationImpact(method,"balanced",{},{detailShapeTolerance:0.5}),"live");
 assert.equal(method.appDefaults?.coarseParticleMode,"off");
 assert.equal(methodConfigurationImpact(method,"balanced",{},{coarseParticleMode:"on"}),"rebuild");
 assert.equal(methodConfigurationImpact(method,"balanced",{coarseParticleMode:"on"},{coarseParticleMode:"off"}),"rebuild");
 const coarse=resolveMethodValues(method,"balanced",{coarseParticleMode:"on",detailPolicy:"requested",detailSolidContact:"off"});assert.equal(coarse.detailPolicy,"requested");assert.equal(coarse.detailSolidContact,"off");
 const dynamic=resolveMethodValues(method,"balanced",{detailPolicy:"dynamic",detailStrain:"on",detailShapeTolerance:0.75});assert.equal(dynamic.detailPolicy,"dynamic");assert.equal(dynamic.detailStrain,"on");assert.equal(dynamic.detailShapeTolerance,0.75);
 const uniform=getMethod("uniform-volume");
 assert.deepEqual(method.params!.filter(p=>p.key.startsWith("detail")).map(p=>p.key),uniform.params!.filter(p=>p.key.startsWith("detail")).map(p=>p.key));
 assert.equal(values.sharpeningSweeps,0);assert.equal(values.sharpeningDistance,0);
 const options=uniformGeometricSolverOptions(values);
 assert.equal(options.sharpeningSweeps,0);assert.equal(options.pressureCycleBudget,"lagged");
 assert.equal(values.totalSurfaceVolume,"on");assert.equal(values.surfaceVolumeRounds,2);
 assert.equal(values.surfaceDeficitBalancing,"off");assert.equal(values.phiDrain,"off");
 const forced=resolveMethodValues(method,"balanced",{totalSurfaceVolume:"off",surfaceVolumeRounds:0,surfaceDeficitBalancing:"on",volumeDustThreshold:0.1});
 assert.equal(forced.totalSurfaceVolume,"on");assert.equal(forced.surfaceVolumeRounds,2);assert.equal(forced.surfaceDeficitBalancing,"off");assert.equal(forced.volumeDustThreshold,0);
 const fixed=new Set(["sharpeningSweeps","sharpeningDistance","totalSurfaceVolume","surfaceVolumeRounds","surfaceDeficitBalancing","volumeDustThreshold","orphanDustThreshold","phiDrain"]);
 for(const stage of (await method.pipelineGraph!()).stages)for(const control of stage.controls??[])if("param" in control)assert.ok(!fixed.has(control.param));
});
