import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createOceanSeicheScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
const dawnModule = process.env.WEBGPU_NODE_MODULE;
(dawnModule ? test : test.skip)("ordinary ocean adopts a demand-prepared topology generation", {timeout:240_000}, async t => {
 await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-production-generation-dawn");
 let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
 try {
  const dawn = await import(pathToFileURL(dawnModule!).href);
  Object.assign(globalThis,dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = (await gpu.requestAdapter())!;
  device = await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  assert.ok(device);
  const errors:string[]=[];
  device.addEventListener("uncapturederror", event=>errors.push(event.error.message));
  const scene = createOceanSeicheScene();
  solver = await WebGPUAdaptiveMassSolver.createAsync(device,scene,"balanced",undefined,
   adaptiveMassSolverOptions({}),()=>{});
  await solver.waitForSimulationReady();
  const initial = await solver.readGPUActivityPolicy();
  const maxWidth = Math.max(...initial.bricks.filter(b=>b.active).map(b=>8*b.spanBricks/b.acceptedResolution));
  assert.ok(maxWidth>=16,`ocean starts with ${maxWidth}h cells`);
  console.log(JSON.stringify({initialLeaves:initial.residentBrickCount,maxWidth, widths:solver.info.adaptivePhysicalWidthCensus}));
  for(let step=1;step<=129;step++) {
   await solver.waitForTopologyReady();
   assert.equal(solver.advanceTo(step*CM12_PAPER_DT_S,[]),true);
   await device.queue.onSubmittedWorkDone();
   if(step%16===0) console.log(JSON.stringify({step,preparing:solver.info.topologyGenerationPending}));
  }
  await solver.waitForTopologyReady();
  console.log(JSON.stringify({requested:solver.info.topologyGenerationRequestedLeaves,
   deferred:solver.info.topologyGenerationDeferred,generations:solver.info.topologyGenerationCount}));
  assert.equal(solver.info.topologyGenerationError,undefined);
  assert.ok((solver.info.topologyGenerationCount??0)>=2,"normal policy must publish successive replacements");
  const after = await solver.readGPUActivityPolicy();
  assert.equal(after.faultFlags,0);
  const presentation = await solver.readPresentationPageAllocatorReceiptQA();
  assert.equal(presentation.faultCode,0);
  assert.ok(presentation.residentPages>0);
  const stats = await solver.readStats();
  assert.ok(Number.isFinite(stats.pressureRelativeResidual));
  console.log(JSON.stringify({residual:stats.pressureRelativeResidual,
   cells:stats.adaptiveAcceptedCellCount,rows:stats.adaptiveAcceptedRowCount,
   bytes:stats.allocatedBytes,presentation}));
  assert.equal(after.acceptedSteps,129);
  assert.deepEqual(errors,[]);
  t.diagnostic(JSON.stringify({generations:solver.info.topologyGenerationCount,leaves:after.residentBrickCount}));
 } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
