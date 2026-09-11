import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createOceanSeicheScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { refinementRegionCellBounds, refinementRegionLattice } from "../lib/core/refinement-regions";
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
   while (!solver.advanceTo(step*CM12_PAPER_DT_S,[])) await new Promise(setImmediate);
   await solver.awaitFrameCompletion?.();
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

(dawnModule ? test : test.skip)("live ocean global min8 survives successive generation publications",
 { timeout: 240_000 }, async t => {
 await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-ocean-global-min8");
 let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
 try {
  const dawn = await import(pathToFileURL(dawnModule!).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = (await gpu.requestAdapter())!;
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  assert.ok(device);
  const errors: string[] = [];
  device.addEventListener("uncapturederror", event => {
   event.preventDefault(); errors.push(event.error.message);
  });
  const scene = createOceanSeicheScene();
  solver = await WebGPUAdaptiveMassSolver.createAsync(device, scene, "balanced", undefined,
   adaptiveMassSolverOptions({}), () => {});
  await solver.waitForSimulationReady();
  while (!solver.advanceTo(CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
  await solver.awaitFrameCompletion?.();
  await solver.waitForTopologyReady();
  const initial = await solver.readGPUActivityPolicy();
  assert.ok(initial.bricks.some(brick => brick.active
   && 8 * brick.spanBricks / brick.acceptedResolution < 8),
  "the live edit must replace genuinely finer ocean cells");
  const edited = structuredClone(scene);
  edited.fluid.refinementRegions = [{
   id: "global-eight-cell-floor", rule: "minimum-cell-size", minimumCellSize_cells: 8,
   min_m: { x: -scene.container.width_m / 2, y: 0, z: -scene.container.depth_m / 2 },
   max_m: { x: scene.container.width_m / 2, y: scene.container.height_m,
    z: scene.container.depth_m / 2 },
  }];
  let step = 1;
  const bounds = refinementRegionCellBounds(edited.fluid.refinementRegions[0]!,
   refinementRegionLattice(edited));
  const verifyAcceptedRegion = async (log = false) => {
   const activity = await solver!.readGPUActivityPolicy();
   const active = activity.bricks.filter(brick => brick.active);
   const intersects = (brick: typeof active[number]) => brick.coordinate.every((q, axis) =>
    q * 8 < bounds.max[axis]! && (q + brick.spanBricks) * 8 > bounds.min[axis]!);
   const inside = active.filter(intersects);
   assert.ok(inside.length > 0);
   const width = (brick: typeof active[number]) => 8 * brick.spanBricks / brick.acceptedResolution;
   const holdouts = inside.filter(brick => width(brick) < 8);
   assert.deepEqual(holdouts.map(brick => ({ coordinate: brick.coordinate,
    span: brick.spanBricks, accepted: brick.acceptedResolution,
    candidate: brick.candidateResolution, planned: brick.plannedResolution,
    meanDensity: brick.meanDensity, minimum: brick.refinementPolicyMinimumResolution,
    maximum: brick.refinementPolicyMaximumResolution,
    reasons: brick.reasons, planReasons: brick.planReasons,
   })), [], `min8 escaped inside the authored region at step ${step}`);
   assert.ok(active.filter(brick => brick.spanBricks > 1)
    .every(brick => brick.refinementPolicyTileScale === 1),
   "macro support must not inherit unit-brick membership tiles");
   assert.equal(activity.faultFlags, 0);
   assert.equal(solver!.info.topologyGenerationError, undefined);
   if (log) t.diagnostic(JSON.stringify({ step,
    generations: solver!.info.topologyGenerationCount,
    insideWidths: [...new Set(inside.map(width))],
    outsideWidths: [...new Set(active.filter(brick => !intersects(brick)).map(width))],
   }));
   return inside.map(width);
  };
  const publishEdit = async (exactWidth: boolean) => {
   const before = solver!.info.topologyGenerationCount ?? 0;
   solver!.applySceneUniforms(edited);
   for (let attempts = 0; attempts < 65; attempts++) {
    step++;
    await solver!.waitForTopologyReady();
    while (!solver!.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
    await solver!.awaitFrameCompletion?.();
    await solver!.waitForTopologyReady();
    assert.equal(solver!.info.topologyGenerationError, undefined);
    if ((solver!.info.topologyGenerationCount ?? 0) > before) break;
   }
   assert.ok((solver!.info.topologyGenerationCount ?? 0) > before,
    `region edit failed to publish: ${JSON.stringify(solver!.info.topologyGenerationDeferred)}`);
   const widths = await verifyAcceptedRegion(true);
   // Ceiling refinement is admitted in bounded batches; every intermediate
   // publication must retain the hard floor while those batches converge.
   if (exactWidth) assert.ok(widths.some(width => width === 8));
  };
  await publishEdit(false);
  edited.fluid.refinementRegions[0]!.maximumCellSize_cells = 8;
  await publishEdit(true);
  while (step < 129) {
   step++;
   await solver.waitForTopologyReady();
   while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
   await solver.awaitFrameCompletion?.();
   await solver.waitForTopologyReady();
   await verifyAcceptedRegion(step % 16 === 0 || step === 129);
  }
  assert.ok((await verifyAcceptedRegion()).every(width => width === 8),
   "the global min8/max8 region must converge after three scheduler batches");
  assert.ok((await verifyAcceptedRegion(true)).every(width => width === 8),
   "the bounded max8 batches must converge while retaining min8");
  assert.deepEqual(errors, []);
 } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});

for (const minimum of [16,32]) (dawnModule ? test : test.skip)(`authored ocean min${minimum} stays enforced during GPU evolution`,
 {timeout:120_000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test",`ocean-min${minimum}`);
 let device:GPUDevice|undefined,solver:WebGPUAdaptiveMassSolver|undefined;
 try {
  const dawn=await import(pathToFileURL(dawnModule!).href);Object.assign(globalThis,dawn.globals);
  const gpu=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=(await gpu.requestAdapter())!;
  device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  assert.ok(device);const errors:string[]=[];
  device.addEventListener("uncapturederror",event=>errors.push(event.error.message));
  const scene=createOceanSeicheScene();
  scene.fluid.refinementRegions=[{id:"global-floor",rule:"minimum-cell-size",minimumCellSize_cells:minimum,
   min_m:{x:-4,y:0,z:-1},max_m:{x:4,y:2.4,z:1}}];
  solver=await WebGPUAdaptiveMassSolver.createAsync(device,scene,"balanced",undefined,adaptiveMassSolverOptions({}),()=>{});
  await solver.waitForSimulationReady();
  for(let step=0;step<=4;step++) {
   if(step>0){while (!solver.advanceTo(step*CM12_PAPER_DT_S,[])) await new Promise(setImmediate);await solver.awaitFrameCompletion?.();await solver.waitForTopologyReady();}
   const activity: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>> = await solver.readGPUActivityPolicy();
   assert.equal(activity.faultFlags,0);
   const inside=activity.bricks.filter(b=>b.active&&b.coordinate.every((q,a)=>q<[40,12,10][a]!&&q+b.spanBricks>0));
   assert.ok(inside.length>0);
   assert.ok(inside.every(b=>8*b.spanBricks/b.acceptedResolution>=minimum));
  }
  assert.deepEqual(errors,[]);
 }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
