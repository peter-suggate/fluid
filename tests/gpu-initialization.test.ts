import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { GPUInitializationTaskRunner } from "../lib/gpu-initialization";
import { structuralMethodValues } from "../lib/webgpu-renderer";

test("GPU initialization progress is derived from registered tasks", async () => {
  const controller = new AbortController();
  const snapshots: Array<{ taskId: string; completed: number; total: number }> = [];
  const order: string[] = [];
  const runner = new GPUInitializationTaskRunner((snapshot) => snapshots.push(snapshot), controller.signal);

  await runner.run([
    { id: "allocate", phase: "allocation", label: "Allocate", run: () => { order.push("allocate"); } },
    { id: "compile", phase: "solver-pipelines", label: "Compile", dependencies: ["allocate"], run: () => { order.push("compile"); } },
    { id: "warm", phase: "warmup", label: "Warm", dependencies: ["compile"], run: () => { order.push("warm"); } },
  ]);

  assert.deepEqual(order, ["allocate", "compile", "warm"]);
  assert.equal(runner.completedCount, 3);
  assert.equal(runner.totalCount, 3);
  assert.ok(snapshots.every(({ completed, total }) => completed <= total));
  assert.deepEqual(
    snapshots.at(-1) && (({ taskId, completed, total }) => ({ taskId, completed, total }))(snapshots.at(-1)!),
    { taskId: "warm", completed: 3, total: 3 },
  );
});

test("GPU initialization rejects duplicate and unsatisfied task dependencies", async () => {
  const signal = new AbortController().signal;
  const runner = new GPUInitializationTaskRunner(() => {}, signal);
  await runner.run([{ id: "one", phase: "planning", label: "One", run() {} }]);
  await assert.rejects(() => runner.run([{ id: "one", phase: "planning", label: "Again", run() {} }]), /Duplicate GPU initialization task/);

  const dependent = new GPUInitializationTaskRunner(() => {}, signal);
  await assert.rejects(() => dependent.run([{ id: "late", phase: "warmup", label: "Late", dependencies: ["missing"], run() {} }]), /ran before missing/);
});

test("power-octree method values are structural after unsupported spray is removed", () => {
  const config = { methodId: "octree", quality: "balanced" as const, values: { maximumLeafSize: "16" } };
  assert.deepEqual(structuralMethodValues(config), { maximumLeafSize: "16" });
});

test("octree initialization has no hand-maintained pipeline totals and fences warm-up", () => {
  const runner = readFileSync(new URL("../lib/gpu-initialization.ts", import.meta.url), "utf8");
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  const octree = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  const fluidLab = readFileSync(new URL("../components/FluidLab.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(uniform, /projectionPipelineCount|secondaryPipelines/);
  assert.match(runner, /requestAnimationFrame\(\(\) => setTimeout\(resolve, 0\)\)/, "task work must begin after the reported stage can paint");
  assert.match(uniform, /initializationTasks\(\)/);
  assert.match(uniform, /uniformPipelineCache/, "structural rebuilds must reuse immutable programs");
  assert.match(octree, /initializationTasks\(\): GPUInitializationTask\[\]/);
  assert.match(octree, /octreePipelineCache/);
  assert.match(octree, /for \(let size = Math\.min\(8, this\.maxLeafSize\); size >= 2;/,
    "startup should warm only regular refinement variants the immutable solver can dispatch");
  assert.match(octree, /for \(let size = this\.maxLeafSize; size >= 16;/,
    "coarse refinement warming should start at the immutable solver maximum");
  assert.doesNotMatch(octree, /for \(let size = 32; size >= 2;/,
    "regular refinement must not compile the coarse-only 16/32 variants");
  assert.match(uniform, /await this\.device\.queue\.onSubmittedWorkDone\(\)/);
  assert.match(controller, /Preparing GPU work plan/);
  assert.match(fluidLab, /Applying simulation settings/);
  assert.doesNotMatch(fluidLab, /continue using the controls/, "the UI must not promise responsiveness while the graphics driver owns the main process");
  const transaction = renderer.slice(renderer.indexOf("private beginGPUFluidInitialization"), renderer.indexOf("private currentGPUFluid"));
  assert.match(transaction, /const drainPreviousForReset=this\.timelineResetPending&&Boolean\(previous\)/,
    "only an explicit timeline reset may detach the active solver before replacement");
  assert.match(transaction, /if\(!drainPreviousForReset\|\|!previous\)return;[\s\S]*await device\.queue\.onSubmittedWorkDone\(\)/,
    "reset replacement must fence previously submitted GPU work before detaching resources");
  assert.match(transaction, /if\(this\.gpuFluid===previous\)\{[\s\S]*this\.gpuFluid=undefined;[\s\S]*previous\.destroy\(\);previousDestroyedForReset=true/,
    "the reset-only path must detach presentation bindings and destroy the drained solver before allocating its replacement");
  assert.match(transaction, /if\(previous&&previous!==solver&&!previousDestroyedForReset\)this\.retireGPUFluid\(previous\)/,
    "ordinary warm replacement must retain and retire the previous solver transactionally");
  assert.match(transaction, /method\.createSolverAsync\([^\n]+abort\.signal\)/);
});
