import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { canQueuePreparedGPUAdvance, FluidLabRenderer, observedGPUAdvanceTime_ms, pausedTargetRequiresGPUAdvance, presentationHasPhysicsSlack, presentationPhysicsQueueDepth, presentationPriorityDue, submitNextPreparedGPUAdvance, type GPUStatus } from "../lib/webgpu-renderer";
import { presentationStateChanged } from "../lib/frame-pacing";

test("presentation takes queue priority once a 60 Hz deadline has elapsed", () => {
  assert.equal(presentationPriorityDue(-Infinity, 0), true);
  assert.equal(presentationPriorityDue(100, 108), false);
  assert.equal(presentationPriorityDue(100, 116.2), true);
});

test("physics admission preserves the measured presentation deadline", () => {
  assert.equal(presentationHasPhysicsSlack(-Infinity, 0, 2, 1), false);
  assert.equal(presentationHasPhysicsSlack(100, 105, 4, 2), true);
  assert.equal(presentationHasPhysicsSlack(100, 112, 4, 2), false);
  assert.equal(presentationHasPhysicsSlack(100, 105, 20, 2), false);
});

test("a paused explicit step bypasses presentation-slack deferral exactly while debt remains", () => {
  assert.equal(pausedTargetRequiresGPUAdvance(false, 0.004, 0), true);
  assert.equal(pausedTargetRequiresGPUAdvance(false, 0.004, 0.004), false);
  assert.equal(pausedTargetRequiresGPUAdvance(true, 0.004, 0), false);
});

test("GPU submission advances only once toward prepared simulation debt", () => {
  let submittedTime_s = 0;
  let advances = 0;
  const fluid = {
    info: { submittedTime_s },
    advanceTo(this: { info: { submittedTime_s: number } }, time_s: number) {
      advances += 1;
      submittedTime_s = Math.min(time_s, submittedTime_s + 0.008);
      this.info.submittedTime_s = submittedTime_s;
      return true;
    }
  } as unknown as Parameters<typeof submitNextPreparedGPUAdvance>[0];

  const result = submitNextPreparedGPUAdvance(fluid, 0.1, []);
  assert.equal(result.previousSubmittedTime, 0);
  assert.equal(result.submittedTime, 0.008);
  assert.equal(advances, 1);
});

test("GPU queue stays dense around presentation without admitting a physics burst", () => {
  assert.equal(presentationPhysicsQueueDepth(undefined, 1), 2);
  assert.equal(presentationPhysicsQueueDepth(35, 1), 1);
  assert.equal(presentationPhysicsQueueDepth(3.4, 1), 5, "one whole advance may overshoot the remaining budget");
  assert.equal(canQueuePreparedGPUAdvance(0, 4), true);
  assert.equal(canQueuePreparedGPUAdvance(3, 4), true);
  assert.equal(canQueuePreparedGPUAdvance(4, 4), false);
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  assert.match(renderer, /const maximumPendingAdvances=postPresentationDepth/,
    "the rolling window is an absolute in-flight ceiling");
  assert.doesNotMatch(renderer, /gpuPendingBatches\+postPresentationDepth/,
    "a presentation must not add a fresh window on top of already queued physics");
  assert.match(renderer, /interveningWorkSequence: nonPhysicsSequenceAtSubmit/,
    "physics wall samples must carry the non-physics queue epoch they followed");
  assert.match(renderer, /this\.nonPhysicsQueueWorkSequence \+= 1;\s*this\.diagnosticQueueWorkPending \+= 1;/,
    "stats readbacks must invalidate physics wall samples and conservative idle evidence");
  assert.match(renderer, /this\.nonPhysicsQueueWorkSequence\+=1;\s*this\.presentationPending=true;/,
    "presentation submission must invalidate later physics wall samples before refilling the queue");
  assert.match(renderer, /queueWasIdleAtSubmit:presentationQueueWasIdleAtSubmit/,
    "presentation cost must only learn from an independently idle queue");
});

test("the exact generic physics partition drives presentation admission", () => {
  const trace = {
    sampleId: 1, domain: "gpu", lane: "physics", context: "test", capturedAt_ms: 0,
    total_ms: 3.5, phases: [{ id: "other", label: "test", duration_ms: 3.5 }],
  } satisfies NonNullable<Parameters<typeof observedGPUAdvanceTime_ms>[0]>;
  assert.equal(observedGPUAdvanceTime_ms(trace), 3.5);
  assert.equal(observedGPUAdvanceTime_ms({ ...trace, lane: "presentation" }), undefined);
  assert.equal(observedGPUAdvanceTime_ms(undefined), undefined);
});

test("renderer does not synthesize a second wall-clock timing system", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const submit = renderer.slice(
    renderer.indexOf("private submitPreparedGPUFluid"),
    renderer.indexOf("/** Keep the GPU occupied", renderer.indexOf("private submitPreparedGPUFluid")),
  );
  assert.match(submit, /submitNextPreparedGPUAdvance/);
  assert.doesNotMatch(submit, /queueReadyAtPromise|cpuAdvanceEncode_ms|physicsQueueWall_ms|gpuAdvanceWall_ms/);
  assert.match(renderer, /resetPresentationTrace\(\)[^{]*\{[^}]*this\.presentationWallEstimator\.reset\(\)/s,
    "a new presentation context must discard a stale scheduling estimate");
});

test("paused solver attachment publications cannot suppress the continuous presentation loop", () => {
  const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const viewportSource = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const attachStart = rendererSource.indexOf("this.gpuFluidPending=create.then");
  const attachEnd = rendererSource.indexOf("}).catch((error:unknown)", attachStart);
  const attach = rendererSource.slice(attachStart, attachEnd);
  const sourceAttach = attach.indexOf("this.svoDryScenePipeline?.setSource(sparseSceneSource,drySceneData");
  const repaint = attach.indexOf("this.pausedPresentationRevision+=1", sourceAttach);
  assert.ok(sourceAttach >= 0 && repaint > sourceAttach,
    "the repaint revision must publish only after the warmed SVO and temporal-ready renderer source attaches");
  assert.equal((attach.match(/pausedPresentationRevision\+=1/g) ?? []).length, 1,
    "one successful transactional attach requests one paused repaint");
  assert.doesNotMatch(viewportSource, /renderer\.presentationRevision|pausedPresentation/,
    "the viewport must not wait for a renderer revision before drawing while paused");

  const stableState = {};
  const attached = [stableState, 1] as const;
  assert.equal(presentationStateChanged([stableState, 0], attached), true);
  assert.equal(presentationStateChanged(attached, attached), false);
  const rawMode = {};
  const raw = [rawMode, 1] as const;
  assert.equal(presentationStateChanged(attached, raw), true);
  assert.equal(presentationStateChanged(raw, raw), false);
});

test("timeline reset invalidates old completions and cannot trigger a timestamp rebuild", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const resetStart = renderer.indexOf("resetSimulationTimeline(): void");
  const resetEnd = renderer.indexOf("/** Stop refilling physics", resetStart);
  const reset = renderer.slice(resetStart, resetEnd);
  assert.match(reset, /this\.simulationRunning = false/);
  assert.match(reset, /this\.timelineResetPending = true/,
    "the replacement must observe that this is a destructive t=0 rebuild");
  assert.match(reset, /this\.gpuFluidGeneration \+= 1/,
    "old queue completions must become stale synchronously at t=0");
  assert.match(reset, /this\.resetGPUQueueTracking\(\)/);
  assert.match(renderer, /gpuSceneSolverKey\([^]*config\.simulationEpoch \?\? 0/,
    "each reset epoch must identify exactly one replacement solver");
  assert.doesNotMatch(renderer, /time_s < \(this\.gpuFluid\.info\.submittedTime_s \?\? 0\)\) \{this\.beginGPUFluidInitialization/,
    "timestamp rollback must never create an unexpected second build");
  assert.match(viewport, /state\.simulationEpoch !== previous\.simulationEpoch\) \{[\s\S]*safeBrowserSimulationEpochChanged[\s\S]*renderer\.resetSimulationTimeline\(\)/,
    "the renderer must be invalidated by the synchronous runtime-store reset edge");
});

test("warm reset republishes t=0 authority and requests its paused raster fence", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const start = renderer.indexOf("private tryReseedGPUFluid");
  const end = renderer.indexOf("/** Structural identity", start);
  const reseed = renderer.slice(start, end);

  assert.match(reseed, /state:"initializing",label:"Re-seeding fenced t=0 solver authority"/,
    "reset must stop advertising the old ready state while the t=0 seed is rebuilt");
  assert.match(reseed, /solver\.info\.initialRasterSurfaceReady=false[\s\S]*pendingInitialRasterPresentation=\{solver,solverGeneration:generation,requestGeneration,submitted:false\}/,
    "a re-seed must earn a new renderer raster fence instead of reusing the old ready bit");
  assert.match(reseed, /pendingInitialRasterPresentation[\s\S]*gpuInfoCallback\?\.\(\{\.\.\.solver\.info\}\)/,
    "reset clears the diagnostics store, so the live solver authority must be republished");
  assert.match(reseed, /finally\(\(\)=>\{[\s\S]*pausedPresentationRevision\+=1/,
    "a paused reset must explicitly wake the draw that submits its raster fence");
  assert.match(reseed, /if\(!reseeded\)\{this\.beginGPUFluidInitialization\(scene,config,key\);return;\}/,
    "a declined warm re-seed must proceed directly to replacement instead of retrying forever");
});

test("timeline reset drains and destroys the previous solver before replacement allocation", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const begin = renderer.indexOf("private beginGPUFluidInitialization");
  const end = renderer.indexOf("private currentGPUFluid", begin);
  const initialization = renderer.slice(begin, end);
  const drain = initialization.indexOf("await device.queue.onSubmittedWorkDone()");
  const destroy = initialization.indexOf("previous.destroy()");
  const create = initialization.indexOf("method.createSolverAsync", destroy);
  assert.ok(drain >= 0 && destroy > drain && create > destroy,
    "reset must fence old work and release its fields before constructing the replacement");
  assert.match(initialization, /this\.updateRenderSources\(\)/,
    "presentation bind groups must stop referencing the old fields before destruction");
  assert.match(initialization, /!previousDestroyedForReset\)this\.retireGPUFluid/,
    "the reset-owned solver must not enter deferred retirement after destruction");
});

test("reset replacement attaches only after complete t=0 sparse authority is resident", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  const octree = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

  const createAsync = uniform.slice(
    uniform.indexOf("static async createAsync"),
    uniform.indexOf("private initializationTasks", uniform.indexOf("static async createAsync")),
  );
  const initialize = createAsync.indexOf("await runner.run(solver!.initializationTasks())");
  const publishSolver = createAsync.indexOf("return solver!", initialize);
  assert.ok(initialize >= 0 && publishSolver > initialize,
    "the solver promise must not resolve before every initialization task, including warmup");

  const tasks = uniform.slice(
    uniform.indexOf("private initializationTasks"),
    uniform.indexOf("private async publishInitialSparseScenePhase"),
  );
  assert.match(tasks,
    /OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES\.forEach[\s\S]*index === 0 \? "solver\.warmup"[\s\S]*publishInitialSparseScenePhase/,
    "the first fenced sparse-publication phase must retain the safe resource-boundary task ID");

  const warmup = uniform.slice(
    uniform.indexOf("private async publishInitialSparseScenePhase"),
    uniform.indexOf("/** Publish a complete t=0 scene", uniform.indexOf("private async publishInitialSparseScenePhase")),
  );
  const encode = warmup.indexOf("encodeInitialSparseAuthorityPhase(initialSparseScene, phase)");
  const submit = warmup.indexOf("queue.submit([initialSparseScene.finish()])", encode);
  const fence = warmup.indexOf("await this.device.queue.onSubmittedWorkDone()", submit);
  const validation = warmup.indexOf("await this.device.popErrorScope()", fence);
  const refreshAllocation = warmup.indexOf("this.applyOctreeInfo(this.octreeProjection)", fence);
  assert.ok(encode >= 0 && submit > encode && fence > submit && validation > fence,
    "every bounded bootstrap phase must be fenced without a CPU simulation readback");
  assert.match(warmup,
    /pushErrorScope\("validation"\)[\s\S]*Initial sparse authority \$\{phase\} validation failed/,
    "the UI must surface the first phase-local WebGPU ABI failure instead of a poisoned downstream authority report");
  assert.ok(refreshAllocation > fence,
    "the final render-world task must refresh complete telemetry only after its fence");
  assert.match(warmup, /if \(phase === "sparse-render-world"\)[\s\S]*initialSparseAuthorityPublished = true/,
    "attachment readiness must remain inside the final fenced phase");
  assert.doesNotMatch(warmup, /mapAsync|getMappedRange/);

  const authority = octree.slice(
    octree.indexOf("encodeInitialSparseAuthorityPhase"),
    octree.indexOf("retireSubmittedEncoder", octree.indexOf("encodeInitialSparseAuthorityPhase")),
  );
  const compactAuthority = authority.replace(/\s+/g, "");
  const cold = compactAuthority.indexOf("encodeColdBootstrapRebuild(encoder)");
  const csr = compactAuthority.indexOf("this.encode(encoder", cold);
  const surface = compactAuthority.indexOf("this.encodeSurface(encoder,0)", csr);
  const residency = compactAuthority.indexOf("this.encodeSparseBrickWorld(encoder)", surface);
  const nextCandidate = compactAuthority.indexOf("this.encodeInactiveTopologyCandidate(encoder)", residency);
  assert.ok(cold >= 0 && csr > cold && surface > csr && residency > surface && nextCandidate > residency,
    "reset warmup must publish cold topology, structured authority, fine authority, render world, then the next inactive candidate in dependency order");
  assert.doesNotMatch(compactAuthority, /encodeGlobalFineFaceBandPhase|closest-point-extension|power-publication/,
    "deleted face-band warmup phases must not remain reachable");

  const transaction = renderer.slice(
    renderer.indexOf("private beginGPUFluidInitialization"),
    renderer.indexOf("private currentGPUFluid"),
  );
  assert.match(transaction, /const create:Promise<GPUSolverInstance>=prepare\(\)\.then\([\s\S]*method\.createSolverAsync/);
  assert.match(transaction, /this\.gpuFluidPending=create\.then\(\(solver\)=>[\s\S]*this\.gpuFluid=solver/,
    "only the fully warmed create promise may publish the replacement solver");
});

test("renderer stops submitting frames and disposes its device after WebGPU loss", async (t) => {
  let resolveDeviceLost!: (info: GPUDeviceLostInfo) => void;
  const lost = new Promise<GPUDeviceLostInfo>((resolve) => { resolveDeviceLost = resolve; });
  let deviceDestroyCount = 0;
  let submitCount = 0;
  let requestedDescriptor: GPUDeviceDescriptor | undefined;
  const destroyable = () => ({ destroy() {} });
  const texture = () => ({ ...destroyable(), width: 1, height: 1, createView: () => ({}) });
  const pipeline = () => ({ getBindGroupLayout: () => ({}) });
  const device = {
    features: new Set<GPUFeatureName>(),
    lost,
    addEventListener() {},
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createRenderPipeline: pipeline,
    createRenderPipelineAsync: async () => pipeline(),
    createComputePipeline: () => ({}),
    createComputePipelineAsync: async () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createSampler: () => ({}),
    createBuffer: destroyable,
    createTexture: texture,
    createBindGroup: () => ({}),
    queue: { submit: () => { submitCount += 1; } },
    destroy: () => { deviceDestroyCount += 1; }
  } as unknown as GPUDevice;
  const adapter = {
    features: new Set<GPUFeatureName>(),
    limits: {
      maxStorageBuffersPerShaderStage: 10,
      maxStorageBufferBindingSize: 512 * 1024 * 1024,
      maxBufferSize: 1024 * 1024 * 1024,
      maxTextureDimension3D: 2048,
    },
    requestDevice: async (descriptor: GPUDeviceDescriptor) => { requestedDescriptor = descriptor; return device; },
    info: { vendor: "test" }
  } as unknown as GPUAdapter;
  const context = { configure() {} } as unknown as GPUCanvasContext;
  const canvas = { getContext: () => context } as unknown as HTMLCanvasElement;
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const previousBufferUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousTextureUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUTextureUsage");
  const previousShaderStage = Object.getOwnPropertyDescriptor(globalThis, "GPUShaderStage");
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { requestAdapter: async () => adapter, getPreferredCanvasFormat: () => "bgra8unorm" } } });
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, QUERY_RESOLVE: 8, COPY_SRC: 16, INDIRECT: 32 } });
  Object.defineProperty(globalThis, "GPUTextureUsage", { configurable: true, value: { TEXTURE_BINDING: 1, COPY_DST: 2, RENDER_ATTACHMENT: 4 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 } });
  t.after(() => {
    for (const [name, descriptor] of [["navigator", previousNavigator], ["GPUBufferUsage", previousBufferUsage], ["GPUTextureUsage", previousTextureUsage], ["GPUShaderStage", previousShaderStage]] as const) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
  });

  const statuses: GPUStatus[] = [];
  const renderer = new FluidLabRenderer(canvas, (status) => statuses.push(status));
  await renderer.initialize();
  assert.equal(statuses.at(-1)?.state, "ready");
  assert.deepEqual(requestedDescriptor?.requiredLimits, {
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 512 * 1024 * 1024,
    maxBufferSize: 1024 * 1024 * 1024,
    maxTextureDimension3D: 2048,
  });

  resolveDeviceLost({ reason: "unknown", message: "test device loss" } as GPUDeviceLostInfo);
  await lost;
  await Promise.resolve();
  assert.deepEqual(statuses.at(-1), { state: "lost", label: "GPU device lost: test device loss" });

  const metrics = renderer.draw(0, {} as never, {} as never, [], undefined, undefined, "webgpu", { methodId: "tall-cell", quality: "balanced", values: {} });
  assert.equal(metrics.methodId, "tall-cell");
  assert.equal(metrics.context, "tall-cell");
  assert.equal(metrics.presentationSubmitted, false);
  assert.equal(metrics.cpu, undefined,
    "the default lean UI path must not synthesize CPU timing while measurement instrumentation is off");
  assert.equal(submitCount, 0, "a lost device must never receive another queue submission");

  renderer.destroy();
  renderer.destroy();
  assert.equal(deviceDestroyCount, 1, "renderer cleanup must be idempotent across hot reload");
});
