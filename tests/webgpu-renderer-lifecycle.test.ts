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
});

test("completion wall time backs adapters without timestamp queries", () => {
  assert.equal(observedGPUAdvanceTime_ms(3.5, 12), 3.5);
  assert.equal(observedGPUAdvanceTime_ms(undefined, 12), 12);
  assert.equal(observedGPUAdvanceTime_ms(undefined, undefined), undefined);
});

test("paused solver attachment and raw publication each request exactly one presentation", () => {
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
  assert.match(viewportSource, /simulation\.time\(\), renderer\.presentationRevision,/,
    "the paused presentation key must poll the renderer-owned attach revision");

  const stableState = {};
  const attached = [stableState, 1] as const;
  assert.equal(presentationStateChanged([stableState, 0], attached), true);
  assert.equal(presentationStateChanged(attached, attached), false,
    "the attached solver paints once and does not create a paused render loop");
  const rawMode = {};
  const raw = [rawMode, 1] as const;
  assert.equal(presentationStateChanged(attached, raw), true);
  assert.equal(presentationStateChanged(raw, raw), false,
    "the raw-mode state change services its pending publication in exactly one presentation");
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
  assert.match(renderer, /config\.simulationEpoch\?\?0/,
    "each reset epoch must identify exactly one replacement solver");
  assert.doesNotMatch(renderer, /time_s < \(this\.gpuFluid\.info\.submittedTime_s \?\? 0\)\) \{this\.beginGPUFluidInitialization/,
    "timestamp rollback must never create an unexpected second build");
  assert.match(viewport, /state\.simulationEpoch !== previous\.simulationEpoch\) \{[\s\S]*safeBrowserSimulationEpochChanged[\s\S]*renderer\.resetSimulationTimeline\(\)/,
    "the renderer must be invalidated by the synchronous runtime-store reset edge");
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
  const refreshAllocation = warmup.indexOf("this.applyOctreeInfo(this.octreeProjection)", fence);
  assert.ok(encode >= 0 && submit > encode && fence > submit,
    "every bounded bootstrap phase must be fenced without a CPU simulation readback");
  assert.ok(refreshAllocation > fence,
    "the final render-world task must refresh complete telemetry only after its fence");
  assert.match(warmup, /if \(phase === "sparse-render-world"\)[\s\S]*initialSparseAuthorityPublished = true/,
    "attachment readiness must remain inside the final fenced phase");
  assert.doesNotMatch(warmup, /mapAsync|getMappedRange/);

  const authority = octree.slice(
    octree.indexOf("encodeInitialSparseAuthorityPhase"),
    octree.indexOf("encodeInitialSparseAuthority(encoder", octree.indexOf("encodeInitialSparseAuthorityPhase")),
  );
  const cold = authority.indexOf("encodeColdBootstrapRebuild(encoder)");
  const csr = authority.indexOf("this.encode(encoder", cold);
  const surface = authority.indexOf("this.encodeSurface(encoder, 0)", csr);
  const faceTopology = authority.indexOf('this.encodeGlobalFineFaceBandPhase(encoder, "topology-build")', surface);
  const faceTransitions = authority.indexOf('this.encodeGlobalFineFaceBandPhase(encoder, "transition-adjacency")', faceTopology);
  const faceMarch = authority.indexOf('this.encodeGlobalFineFaceBandPhase(encoder, "fast-march")', faceTransitions);
  const facePowerPublication = authority.indexOf(
    'this.encodeGlobalFineFaceBandPhase(encoder, "power-publication")', faceMarch,
  );
  const residency = authority.indexOf("this.encodeSparseBrickWorld(encoder)", facePowerPublication);
  assert.ok(cold >= 0 && csr > cold && surface > csr && faceTopology > surface
    && faceTransitions > faceTopology && faceMarch > faceTransitions
    && facePowerPublication > faceMarch && residency > facePowerPublication,
  "reset warmup must publish topology, power/operator, fine authority, the four paper-ordered Section 5 phases, and render world in dependency order");

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
  assert.deepEqual(metrics, { cpuFrame_ms: 0, cpuPhysicsSubmit_ms: 0, cpuDataUpload_ms: 0, cpuRenderEncode_ms: 0 });
  assert.equal(submitCount, 0, "a lost device must never receive another queue submission");

  renderer.destroy();
  renderer.destroy();
  assert.equal(deviceDestroyCount, 1, "renderer cleanup must be idempotent across hot reload");
});
