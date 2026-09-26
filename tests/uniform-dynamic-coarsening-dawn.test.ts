import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {refinementRegionLattice} from "../lib/core/refinement-regions";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import type {GPUSolverInstance} from "../lib/core/method-contract";

// Phase 5 lane (docs/plans/uniform-dynamic-coarsening.md): the app's
// Uniform Geometric method with coarsening=dynamic on the 128³ dam break.
// The surface must never reach a 2h/4h owner (the solver throws), ownership
// must actually follow the flow, volume must be conserved, and a relayout
// must not compile anything or grow memory past the band's high-water mark.
const STEPS=Number(process.env.UNIFORM_DYNAMIC_LANE_STEPS??30);
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("dynamic coarsening follows the 128³ dam break without surface escape, drift or recompilation",{timeout:1200000},async t=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform dynamic coarsening lane");
 let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  let watching=false;const allocations:string[]=[];
  for(const name of ["createBuffer","createTexture","createShaderModule","createComputePipeline","createComputePipelineAsync"] as const){
   const original=(raw[name] as Function).bind(raw);
   Object.defineProperty(raw,name,{configurable:true,writable:true,value:(...args:unknown[])=>{
    if(watching)allocations.push(`${name}: ${(args[0] as {label?:string})?.label} ${(args[0] as {size?:number})?.size??""}`);return original(...args);}});
  }
  device=managedGPUDevice(raw,{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("high-resolution-dam-break")));scene.fluid.refinementRegions=[];
  assert.deepEqual(refinementRegionLattice(scene).dimensions,[128,128,128]);
  solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",{pressureResidualTolerance:5,coarsening:"dynamic"},undefined,()=>{});
  const info=solver.info as unknown as Record<string,unknown>;
  let relayouts=0,coarsest=Infinity,finest=0;const wall:number[]=[],shapes:string[]=[];
  for(let step=1;step<=STEPS;step++){
   // Watch from the third frame: the first frames grow the band, the tap
   // cache and the pressure records to their working sizes.
   watching=step>2;
   const start=performance.now();assert.ok(solver.advanceTo(step/30,[]));await solver.awaitFrameCompletion?.();wall.push(performance.now()-start);
   await solver.readStats();
   const fine=Number(info.uniformMixedFineTiles),two=Number(info.uniformMixedTransitionTiles),four=Number(info.uniformMixedCoarseTiles);
   assert.equal(fine+two+four,32768,`step ${step}: tile counts`);
   assert.equal(info.uniformMixedDynamicViolations,0,`step ${step}: phi crossing in a coarse owner`);
   relayouts=Number(info.uniformMixedDynamicRelayouts);coarsest=Math.min(coarsest,fine);finest=Math.max(finest,fine);
   shapes.push(`${fine}/${two}/${four}`);
   assert.ok(Math.abs(Number(info.volumeDrift))<1e-4,`step ${step}: volume drift ${info.volumeDrift}`);
  }
  watching=false;
  t.diagnostic(`h/2h/4h tiles per step: ${shapes.join(" ")}`);
  const median=(a:number[])=>[...a].sort((x,y)=>x-y)[a.length>>1]!;
  t.diagnostic(`${relayouts} relayouts; median wall ${median(wall).toFixed(1)} ms (steps 1-${STEPS})`);
  assert.deepEqual(errors,[]);
  // Ownership follows the flow: it starts mostly coarse and refines as the front spreads.
  assert.ok(relayouts>=STEPS/2,`relayouts ${relayouts}`);
  assert.ok(coarsest<32768/2&&finest>coarsest,`fine tiles range ${coarsest}..${finest}`);
  const compiled=allocations.filter(a=>!a.startsWith("createBuffer")&&!a.startsWith("createTexture"));
  assert.deepEqual(compiled,[],"relayout compiled shaders or pipelines");
  t.diagnostic(`allocations after frame 2: ${allocations.length ? allocations.join("; ") : "none"}`);
 }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
