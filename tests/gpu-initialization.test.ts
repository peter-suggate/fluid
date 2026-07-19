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

test("runtime method values do not invalidate the structural solver fingerprint", () => {
  const base = { methodId: "octree", quality: "balanced" as const, values: { maximumLeafSize: "16", secondaryParticles: "on" } };
  const disabled = { ...base, values: { ...base.values, secondaryParticles: "off" } };
  assert.deepEqual(structuralMethodValues(base), { maximumLeafSize: "16" });
  assert.deepEqual(structuralMethodValues(disabled), { maximumLeafSize: "16" });
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
  assert.match(octree, /for \(let size = 32; size >= 2;/, "all leaf-size variants should be warmed before the settings UI advertises them");
  assert.match(uniform, /await this\.device\.queue\.onSubmittedWorkDone\(\)/);
  assert.match(controller, /Preparing GPU work plan/);
  assert.match(fluidLab, /Applying simulation settings/);
  assert.doesNotMatch(fluidLab, /continue using the controls/, "the UI must not promise responsiveness while the graphics driver owns the main process");
  const transaction = renderer.slice(renderer.indexOf("private beginGPUFluidInitialization"), renderer.indexOf("private currentGPUFluid"));
  assert.doesNotMatch(transaction, /this\.gpuFluid=undefined/);
  assert.match(transaction, /if\(previous&&previous!==solver\)this\.retireGPUFluid\(previous\)/);
  assert.match(transaction, /method\.createSolverAsync\([^\n]+abort\.signal\)/);
});
