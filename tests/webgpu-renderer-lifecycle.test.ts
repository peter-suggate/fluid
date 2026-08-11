import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { BROWSER_GPU_THROUGHPUT_DEPTH, canQueuePreparedGPUAdvance , submitNextPreparedGPUAdvance } from "../lib/webgpu-renderer";
import { MAXIMUM_PENDING_PHYSICS_ADVANCES } from "../lib/structured-step-snapshot";
import { presentationStateChanged } from "../lib/frame-pacing";

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

test("GPU queue pairs at most one simulation advance with each presentation", () => {
  assert.equal(MAXIMUM_PENDING_PHYSICS_ADVANCES, 8);
  assert.equal(BROWSER_GPU_THROUGHPUT_DEPTH, 2);
  assert.equal(canQueuePreparedGPUAdvance(0, 4), true);
  assert.equal(canQueuePreparedGPUAdvance(3, 4), true);
  assert.equal(canQueuePreparedGPUAdvance(4, 4), false);
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const draw = renderer.slice(renderer.indexOf("  draw(time_s:"), renderer.indexOf("  destroy(): void"));
  const advance = draw.indexOf("gpuInfo = this.submitPreparedGPUFluid");
  const presentation = draw.indexOf("this.device.queue.submit([encoder.finish()])");
  assert.ok(advance >= 0 && presentation > advance,
    "the paired presentation must be queued immediately after its simulation advance");
  assert.equal((draw.match(/this\.submitPreparedGPUFluid\(/g) ?? []).length, 1,
    "draw must never enqueue an autonomous post-presentation simulation burst");
  assert.doesNotMatch(renderer, /continuePreparedGPUWork|postPresentationDepth/,
    "simulation completion must not refill ahead of the next paired presentation");
});

test("renderer scheduling has no wall-clock timing system", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const submit = renderer.slice(
    renderer.indexOf("private submitPreparedGPUFluid"),
    renderer.indexOf("/** Re-pose the scene", renderer.indexOf("private submitPreparedGPUFluid")),
  );
  assert.match(submit, /submitNextPreparedGPUAdvance/);
  assert.doesNotMatch(submit, /queueReadyAtPromise|cpuAdvanceEncode_ms|physicsQueueWall_ms|gpuAdvanceWall_ms|WallEstimator/);
  assert.doesNotMatch(renderer, /presentationHasPhysicsSlack|presentationPhysicsQueueDepth|presentationWallEstimator|advanceWallEstimator/,
    "render and simulation admission must not negotiate through timing estimates");
});

test("presentation stays double buffered without a single-frame completion gate", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  assert.doesNotMatch(renderer, /presentationPending|presentationThroughputSlotAvailable/);
  assert.match(renderer, /presentationsInFlight >= BROWSER_GPU_THROUGHPUT_DEPTH/,
    "presentation must stay double buffered instead of growing an unbounded stale-frame FIFO");
  assert.match(renderer, /this\.device\.queue\.submit\(\[encoder\.finish\(\)\]\)/);
  assert.match(renderer, /completedPresentations\+=1/,
    "FPS evidence must count completed GPU presentations without gating the next draw");
});

test("paused solver attachment publications cannot suppress the continuous presentation loop", () => {
  const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const viewportSource = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const attachStart = rendererSource.indexOf("this.gpuFluidPending=create.then");
  const attachEnd = rendererSource.indexOf("}).catch((error:unknown)", attachStart);
  const attach = rendererSource.slice(attachStart, attachEnd);
  const sourceAttach = attach.indexOf("this.attachSparsePresentationSource(solver,generation,startedAt_ms,");
  const repaint = attach.indexOf("this.pausedPresentationRevision+=1", sourceAttach);
  assert.ok(sourceAttach >= 0 && repaint > sourceAttach,
    "the repaint revision must publish only after the warmed SVO renderer source attaches");
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
  assert.doesNotMatch(renderer, /time_s < \(this\.gpuFluid\.info\.submittedTime_s \?\? 0\)\)[^}]*return undefined/,
    "a lagging cross-realm presentation clock must retain the valid solver and its raster sources");
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
  assert.match(reseed, /solver\.info\.initialRasterSurfaceReady=false[\s\S]*pendingInitialRasterPresentation=\{solver,solverGeneration:generation,requestGeneration,submitted:false,resource\}/,
    "a re-seed must earn a new renderer raster fence instead of reusing the old ready bit");
  assert.match(reseed, /pendingInitialRasterPresentation[\s\S]*gpuInfoCallback\?\.\(\{\.\.\.solver\.info\}\)/,
    "reset clears the diagnostics store, so the live solver authority must be republished");
  assert.match(reseed, /finally\(\(\)=>\{[\s\S]*pausedPresentationRevision\+=1/,
    "a paused reset must explicitly wake the draw that submits its raster fence");
  assert.match(reseed, /if\(!reseeded\)\{this\.beginGPUFluidInitialization\(scene,config,key,presentationMode\);return;\}/,
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
  const uniform = readFileSync(new URL("../lib/webgpu-octree-eulerian.ts", import.meta.url), "utf8");
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
  assert.match(transaction, /const create:Promise<\{solver:GPUSolverInstance;sidecar\?:WebGPULiveSvoScene\}>=prepare\(\)\.then\([\s\S]*method\.createSolverAsync/);
  assert.match(transaction, /this\.gpuFluidPending=create\.then\(\(\{solver,sidecar\}\)=>[\s\S]*this\.gpuFluid=solver/,
    "only the fully warmed create promise may publish the replacement solver");
});
