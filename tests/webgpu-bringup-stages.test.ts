import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  DEFAULT_WEBGPU_BRINGUP_TIMEOUT_MS,
  parseWebGPUBringupStage,
  parseWebGPUBringupTimeout,
  reachedSolverResourceBoundary,
  stageIncludesComputeSentinel,
  stageIncludesSparseT0,
  webGPUBringupStages,
} from "../tools/webgpu-bringup-stages";

test("Dawn bring-up stages advance through isolated evidence boundaries", () => {
  assert.deepEqual(webGPUBringupStages, ["adapter-device", "compute-sentinel", "solver-resources", "sparse-t0", "one-step"]);
  assert.equal(parseWebGPUBringupStage(undefined), "adapter-device");
  assert.equal(stageIncludesComputeSentinel("adapter-device"), false);
  assert.equal(stageIncludesComputeSentinel("compute-sentinel"), true);
  assert.equal(stageIncludesSparseT0("solver-resources"), false);
  assert.equal(stageIncludesSparseT0("sparse-t0"), true);
  assert.equal(stageIncludesSparseT0("one-step"), true);
  assert.throws(() => parseWebGPUBringupStage("browser"), /Unknown FLUID_BRINGUP_STAGE/);
});

test("resource stage stops at the real sparse warmup task boundary", () => {
  assert.equal(reachedSolverResourceBoundary({ phase: "warmup", taskId: "solver.warmup", label: "Publish and warm initial sparse scene", completed: 80, total: 81 }), true);
  assert.equal(reachedSolverResourceBoundary({ phase: "adaptive-topology", taskId: "octree.power-catalog", label: "Load catalog", completed: 79, total: 81 }), false);
  assert.equal(reachedSolverResourceBoundary({ phase: "warmup", label: "Unidentified warmup", completed: 80, total: 81 }), false);
});

test("bring-up timeout is bounded and validated before Dawn loads", () => {
  assert.equal(parseWebGPUBringupTimeout(undefined), DEFAULT_WEBGPU_BRINGUP_TIMEOUT_MS);
  assert.equal(parseWebGPUBringupTimeout("30000"), 30_000);
  assert.throws(() => parseWebGPUBringupTimeout("999"), /must be an integer/);
  assert.throws(() => parseWebGPUBringupTimeout("Infinity"), /must be an integer/);
});

test("launcher owns the child-process timeout and worker owns exclusive GPU cleanup", async () => {
  const launcher = await readFile(new URL("../tools/run-webgpu-bringup-stage.ts", import.meta.url), "utf8");
  const worker = await readFile(new URL("../tools/run-webgpu-bringup-stage-worker.ts", import.meta.url), "utf8");
  const fullSmoke = await readFile(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  assert.match(launcher, /spawn\(process\.execPath/);
  assert.match(launcher, /child\.kill\("SIGTERM"\)/);
  assert.match(launcher, /child\.kill\("SIGKILL"\)/);
  assert.match(launcher, /process\.exit\(124\)/);
  assert.match(launcher, /leaving the exclusive GPU lock in place/);
  assert.match(launcher, /Never run Dawn and browser GPU validation concurrently/);
  assert.match(worker, /fluid-webgpu-exclusive\.lock/);
  assert.match(worker, /stoppedBeforeTask: "solver\.warmup"/);
  assert.match(worker, /boundary = snapshot\.completed > lastInitializationCompleted \? "completed" : "starting"/,
    "every fenced warmup subphase must be visibly identified before and after submission");
  assert.match(worker, /record: "solver-initialization", boundary, stage/);
  assert.match(worker, /initialSparseAuthorityReady/);
  assert.match(worker, /sparse-t0 validation failed/);
  assert.equal(worker.match(/await flushGPUErrorDelivery\(device\)/g)?.length, 3,
    "resource, sparse-t0, and one-step verdicts must all flush delayed uncaptured errors");
  assert.match(worker, /\(info\.encodedSteps \?\? 0\) !== 1/);
  assert.match(worker, /await rm\(EXCLUSIVE_LOCK/);
  assert.match(fullSmoke, /Never run this smoke and browser GPU validation concurrently/);
});

test("solver-resource evidence flushes uncaptured validation before reporting pass without solver submission", async () => {
  const worker = await readFile(new URL("../tools/run-webgpu-bringup-stage-worker.ts", import.meta.url), "utf8");
  const boundaryCatch = worker.slice(
    worker.indexOf("if (!(error instanceof SolverResourceBoundary))"),
    worker.indexOf("if (stageIncludesSparseT0(stage))"),
  );

  assert.match(worker, /device\.addEventListener\("uncapturederror"/);
  assert.match(boundaryCatch, /await flushGPUErrorDelivery\(device\)/);
  assert.match(worker, /await device\.queue\.onSubmittedWorkDone\(\)/);
  assert.match(worker, /setImmediate\(resolve\)/);
  assert.ok(boundaryCatch.indexOf("validationErrors.length > 0") < boundaryCatch.indexOf('phase: "solver-resources"'));
  assert.match(boundaryCatch, /throw new Error\(`solver-resources validation failed:/);
  assert.doesNotMatch(boundaryCatch, /queue\.submit|advanceTo|beginComputePass/);
});
