import { createMini32EnergyBudget } from "./mini32-energy-stage-budget";
import assert from "node:assert/strict";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock, readWebGPUExclusiveLockHolder } from "../lib/harness/webgpu-smoke-isolation";

const arm = process.argv.find(a=>a.startsWith("--arm="))?.slice(6);
if (arm) {
  process.env.WEBGPU_NODE_MODULE ??= `${process.cwd()}/node_modules/webgpu/index.js`;
  process.env.ENERGY_OUTPUT = `artifacts/mini32-fixed4-energy/${arm}`;
  if (arm === "conditioning-off") { process.env.ENERGY_OVERRIDES = JSON.stringify({gammaDiffusion:"off",surfaceSharpening:"off"}); }
  if (arm === "tight-pressure") process.env.ENERGY_OVERRIDES = JSON.stringify({pressureIterations:256,pressureRelativeTolerance:0.000001});
  if (arm === "threshold-minus" || arm === "threshold-plus") process.env.ENERGY_STEPS="85";
  if (arm === "closed-wall-step85") process.env.ENERGY_STEPS="85";
  if (arm === "half-dt") {process.env.ENERGY_DT=String(1/60);process.env.ENERGY_STEPS="210";}
  if (arm === "quarter-dt") {process.env.ENERGY_DT=String(1/120);process.env.ENERGY_STEPS="420";}
}

// Retain the native Dawn instance until all asynchronous readbacks finish.
const live = new Set<GPU>();
const dawnModule = process.env.WEBGPU_NODE_MODULE;
assert.ok(dawnModule, "Set WEBGPU_NODE_MODULE to the native Dawn module path");
if(process.argv.includes("--wait")) {
  for(let attempt=0;attempt<180 && await readWebGPUExclusiveLockHolder();attempt++)
    await new Promise(resolve=>setTimeout(resolve,1000));
}
await acquireWebGPUExclusiveLock("dawn-probe", "mini32-fixed4-energy");
const errors: string[] = [];
let gpu: GPU | undefined;
let device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const { create, globals } = await import(pathToFileURL(dawnModule).href);
  Object.assign(globalThis, globals);
  gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]) as GPU;
  live.add(gpu);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  let shaderEdits=0;
  if(arm === "closed-wall-guard" || arm === "closed-wall-step85") {
    const createShaderModule=device.createShaderModule.bind(device);
    device.createShaderModule=(descriptor:GPUShaderModuleDescriptor)=>{
      let code=descriptor.code;
      const start=code.indexOf("fn pressureCellSubmerged(id:u32)->bool{");
      if(start>=0) {
        const end=code.indexOf("fn pressureCellMembershipFromDensity",start);
        assert.ok(end>start);
        const parityAddress=code.match(/fn cm12FCSourceScalarParity\(\)->u32\{return cm12FCLoad\((\d+)u\);\}/);
        assert.ok(parityAddress,"locate the absolute frame-control address");
        const committedAddress=Number(parityAddress[1])-16+33;
        const original=code.slice(start,end);
        const patched=original.replace("if(count<2u){return false;}",
          `if(count<2u){if(${arm === "closed-wall-step85" ? `cm12FCLoad(${committedAddress}u)==84u&&` : ""}rowOpenFraction(row)<=1e-8){continue;}return false;}`);
        assert.notEqual(patched,original);
        code=code.slice(0,start)+patched+code.slice(end);shaderEdits++;
      }
      return createShaderModule({...descriptor,code});
    };
  }
  device.addEventListener("uncapturederror", event => errors.push(event.error.message));
  const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
  scene.fluid.refinementRegions = [{
    id: "energy-fixed4", rule: "minimum-cell-size", minimumCellSize_cells: 4, maximumCellSize_cells: 4,
    min_m: { x: -scene.container.width_m / 2, y: 0, z: -scene.container.depth_m / 2 },
    max_m: { x: scene.container.width_m / 2, y: scene.container.height_m, z: scene.container.depth_m / 2 },
  }];
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = Number(process.env.ENERGY_DT ?? CM12_PAPER_DT_S);
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", { selectorMode: "coarse-first", timeStep: "scene", ...JSON.parse(process.env.ENERGY_OVERRIDES ?? "{}") });
  solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => { }) as WebGPUAdaptiveMassSolver;
  await solver.waitForSimulationReady();
  if(arm === "closed-wall-guard" || arm === "closed-wall-step85") assert.ok(shaderEdits>0);
  const output = process.env.ENERGY_OUTPUT ?? "artifacts/mini32-fixed4-energy/base";
  await mkdir(output, {recursive:true});
  const sourcePaths=["lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    "lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts"];
  const sourceHashes=Object.fromEntries(await Promise.all(sourcePaths.map(async path=>
    [path,createHash("sha256").update(await readFile(path)).digest("hex")])));
  await writeFile(`${output}/config.json`, JSON.stringify({scene,values,arm,shaderEdits,sourceHashes},null,2));
  const trace = [];
  for (let step=0; step<=Number(process.env.ENERGY_STEPS ?? 105); step++) {
    process.env.ENERGY_CAPTURE_CELLS = step>=80 && step<=90 ? "1":"0";
    const budget = step ? await createMini32EnergyBudget(device,solver,scene.voxelDomain.finestCellSize_m,values.gammaDiffusion!=="off",9.81,
      step===85&&(arm==="threshold-minus"||arm==="threshold-plus") ? {cell:16,value:.5+(arm==="threshold-plus"?1e-7:-1e-7)} : undefined) : undefined;
    budget?.arm();
    if(step) {
      while(!solver.advanceTo(step * Number(process.env.ENERGY_DT ?? CM12_PAPER_DT_S),[])) await new Promise(setImmediate);
      await solver.waitForTopologyReady();
      assert.equal(solver.info.encodedSteps,step);
    }
    const stages = await budget?.read();
    const fields = await solver.readDiagnosticFields(true);
    const activity = await solver.readGPUActivityPolicy();
    const h=scene.voxelDomain.finestCellSize_m, nx=solver.info.nx, ny=solver.info.ny;
    let mass=0, kinetic=0,potential=0,maxSpeed=0;
    for(let at=0;at<fields.density.length;at++) {
      const m=fields.density[at]! * h**3;
      const v2=[0,1,2].reduce((n,a)=>n+fields.velocity[4*at+a]!**2,0);
      mass+=m; kinetic+=.5*m*v2;
      potential+=m*9.81*(Math.floor(at/nx)%ny+.5)*h;
      if(m>1e-8) maxSpeed=Math.max(maxSpeed,Math.sqrt(v2));
    }
    assert.equal(activity.acceptedTopologyGeneration,1,"this probe requires a fixed authored topology");
    assert.equal(activity.bricks.filter(b=>b.active).length,64);
    assert.ok(activity.bricks.every(b=>!b.active||b.acceptedResolution===2),"min4/max4 means two cells per brick edge");
    if(stages?.length) {
      const final=stages.at(-1)!;
      assert.ok(Math.abs(final.mass-mass)<1e-8,"native/dense mass agreement");
      assert.ok(Math.abs(final.potential-potential)<1e-8,"native/dense potential agreement");
      assert.ok(Math.abs(final.collocatedKinetic-kinetic)<1e-6,"native/dense collocated kinetic agreement");
    }
    const stats=await solver.readStats();
    const row={step,time:stats.simulatedTime_s,mass,kinetic,potential,total:kinetic+potential,maxSpeed,
      histogram:Object.fromEntries([1,2,4,8].map(r=>[r,activity.bricks.filter(b=>b.active&&b.acceptedResolution===r).length])),
      generation:activity.acceptedTopologyGeneration,iterations:stats.pressureIterationsExecuted,residual:stats.pressureRelativeResidual,stages};
    trace.push(row);
    await writeFile(`${output}/trace.json`,JSON.stringify(trace,null,2));
    if(step%10===0) console.log(JSON.stringify({...row,stages:undefined}));
    assert.deepEqual(errors,[]);
  }
} finally {
  solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
  if(gpu) live.delete(gpu);
}
