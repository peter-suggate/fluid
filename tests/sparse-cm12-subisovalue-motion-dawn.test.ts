// Known-red physical regression: transport loses all velocity when no cell
// exceeds the surface isovalue. Kept outside the short canonical gate until
// the material-momentum/pressure-support repair passes mixed-resolution controls.
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createAnalyticMotionScene } from "../lib/core/analytic-motion-scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("a sub-isovalue liquid slab retains its uniform transport momentum",{timeout:60000},async()=>{
  await acquireWebGPUExclusiveLock("dawn-test","sub-isovalue-motion");
  let device:GPUDevice|undefined,solver:WebGPUAdaptiveMassSolver|undefined;
  try{
    const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
    const gpu:GPU=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
    const adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
    const scene=createAnalyticMotionScene("translation");
    scene.fluid.initialDamBreakDimensions_m!.y=.1;
    scene.fluid.refinementRegions=[{id:"coarse",rule:"minimum-cell-size",minimumCellSize_cells:8,maximumCellSize_cells:8,
      min_m:{x:-.8,y:0,z:-.2},max_m:{x:.8,y:1.6,z:.2}}];
    const values=resolveMethodValues(adaptiveMassMethod,"balanced",{selectorMode:"coarse-first",timeStep:"scene",maximumMacroSpanBricks:"1"});
    solver=await adaptiveMassMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    for(let step=0;step<=3;step++){
      if(step){while(!solver.advanceTo(step/60,[]))await new Promise(setImmediate);await solver.waitForTopologyReady();}
      const fields=await solver.readDiagnosticFields(true);
      let mass=0,momentum=0,maxDensity=0;
      for(let i=0;i<fields.density.length;i++){
        const rho=fields.density[i]!;mass+=rho;momentum+=rho*fields.velocity[4*i+1]!;maxDensity=Math.max(maxDensity,rho);
      }
      assert.ok(maxDensity<.5,"fixture must have no pressure-liquid velocity seed");
      assert.ok(Math.abs(mass/256-1)<1e-4,`step ${step}: mass ${mass}`);
      assert.ok(Math.abs(momentum/mass+.4)<1e-5,`step ${step}: uniform velocity ${momentum/mass} must survive without a >0.5 cell`);
    }
  }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
