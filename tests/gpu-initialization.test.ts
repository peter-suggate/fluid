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

test("GPU initialization tasks can report progress inside one pipeline family", async () => {
  const snapshots: Array<{ taskId: string; label: string; completed: number; total: number }> = [];
  const runner = new GPUInitializationTaskRunner(
    (snapshot) => snapshots.push(snapshot), new AbortController().signal);

  await runner.run([{
    id: "dynamics", phase: "solver-pipelines", label: "Compile dynamics",
    workUnits: 3,
    run: async (_signal, report) => {
      report?.("Compile dynamics: prepare", 0);
      await Promise.resolve();
      report?.("Compile dynamics: project", 1);
    },
  }]);

  assert.ok(snapshots.some(({ label, completed, total }) =>
    label === "Compile dynamics: prepare" && completed === 0 && total === 3));
  assert.ok(snapshots.some(({ label, completed, total }) =>
    label === "Compile dynamics: project" && completed === 1 && total === 3));
  assert.equal(runner.completedCount, 1, "subtasks must not become dependency boundaries");
  assert.equal(runner.totalCount, 3);
  assert.deepEqual(snapshots.at(-1), {
    taskId: "dynamics", phase: "solver-pipelines", label: "Compile dynamics", completed: 3, total: 3,
  });
});

test("GPU initialization paints phase changes and batches same-phase work", async () => {
  const originalAnimationFrame = globalThis.requestAnimationFrame;
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  let animationFrames = 0;
  Object.defineProperty(globalThis, "document", { configurable: true, value: {} });
  globalThis.requestAnimationFrame = ((callback: FrameRequestCallback) => {
    animationFrames += 1;
    callback(0);
    return animationFrames;
  }) as typeof requestAnimationFrame;
  try {
    const runner = new GPUInitializationTaskRunner(() => {}, new AbortController().signal);
    await runner.run(Array.from({ length: 20 }, (_, index) => ({
      id: `pipeline-${index}`,
      phase: "solver-pipelines" as const,
      label: `Pipeline ${index}`,
      run: async () => {},
    })));
    assert.equal(animationFrames, 3, "twenty tiny pipeline tasks should paint in three bounded batches");

    await runner.run([{ id: "upload", phase: "upload", label: "Upload", run() {} }]);
    assert.equal(animationFrames, 4, "a new phase must paint before it starts");

    await runner.run([{
      id: "forced-paint", phase: "upload", label: "Large upload",
      paintBeforeRun: true, run() {},
    }]);
    assert.equal(animationFrames, 5, "heavy synchronous tasks can request an immediate paint");
  } finally {
    if (originalAnimationFrame) globalThis.requestAnimationFrame = originalAnimationFrame;
    else delete (globalThis as { requestAnimationFrame?: typeof requestAnimationFrame }).requestAnimationFrame;
    if (originalDocument) Object.defineProperty(globalThis, "document", originalDocument);
    else Reflect.deleteProperty(globalThis, "document");
  }
});

test("GPU worker initialization yields without waiting for presentation rAF", async () => {
  const originalAnimationFrame = globalThis.requestAnimationFrame;
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  let animationFrames = 0;
  Reflect.deleteProperty(globalThis, "document");
  globalThis.requestAnimationFrame = (() => {
    animationFrames += 1;
    throw new Error("worker startup must not schedule presentation rAF");
  }) as typeof requestAnimationFrame;
  try {
    const runner = new GPUInitializationTaskRunner(() => {}, new AbortController().signal);
    await runner.run([{ id: "allocate", phase: "allocation", label: "Allocate", run() {} }]);
    assert.equal(animationFrames, 0);
    assert.equal(runner.completedCount, 1);
  } finally {
    if (originalAnimationFrame) globalThis.requestAnimationFrame = originalAnimationFrame;
    else delete (globalThis as { requestAnimationFrame?: typeof requestAnimationFrame }).requestAnimationFrame;
    if (originalDocument) Object.defineProperty(globalThis, "document", originalDocument);
    else Reflect.deleteProperty(globalThis, "document");
  }
});

test("power-octree method values are structural after unsupported spray is removed", () => {
  const config = { methodId: "octree", quality: "balanced" as const, values: { maximumLeafSize: "16" } };
  assert.deepEqual(structuralMethodValues(config), { maximumLeafSize: "16" });
});

test("octree initialization has no hand-maintained pipeline totals and fences warm-up", () => {
  const runner = readFileSync(new URL("../lib/gpu-initialization.ts", import.meta.url), "utf8");
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  const octree = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const spgrid = readFileSync(new URL("../lib/webgpu-octree-spgrid-vcycle.ts", import.meta.url), "utf8");
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  const fluidLab = readFileSync(new URL("../components/FluidLab.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(uniform, /projectionPipelineCount|secondaryPipelines/);
  assert.match(runner, /requestAnimationFrame\(\(\) => setTimeout\(resolve, 0\)\)/, "batched work must begin after the reported stage can paint");
  assert.match(runner, /task\.phase !== this\.lastPaintedPhase[\s\S]*this\.tasksSincePaint >= TASKS_PER_PAINT/,
    "phase transitions and bounded same-phase batches must remain visible");
  assert.match(runner, /workUnits\?: number/,
    "resource owners must be able to expose milestones without weakening their dependency boundary");
  assert.match(uniform, /workUnits:OCTREE_SOLVER_ALLOCATION_WORK_UNITS/);
  assert.match(octree, /OCTREE_ALLOCATION_STAGES[\s\S]*Plan octree domain and capacity[\s\S]*Finalize octree resource graph/,
    "octree allocation progress must be colocated with its resource constructor");
  assert.match(uniform, /initializationTasks\(\)/);
  assert.match(uniform, /uniformPipelineCache/, "structural rebuilds must reuse immutable programs");
  assert.match(octree, /initializationTasks\(\): GPUInitializationTask\[\]/);
  assert.match(octree, /octreePipelineCache/);
  for (const family of ["spgrid", "fine-topology", "fine-redistance", "air-support", "structured-dynamics"]) {
    assert.match(octree, new RegExp(`id: "octree\\.power-pipelines\\.${family}"`),
      `${family} must compile after buffer-only power-authority allocation`);
  }
  assert.match(octree, /compileHierarchicalExecutor: false,[\s\S]*deferPipelineCompilation: this\.deferPipelineCompilation/,
    "persistent SPGrid setup must omit hierarchical programs and defer compilation in the async UI path");
  assert.match(octree, /this\.airVelocitySupport = new WebGPUOctreeAirVelocitySupportProducer\([\s\S]*?this\.deferPipelineCompilation\);/,
    "air support must preserve allocation while deferring its pipeline family");
  assert.match(octree, /this\.structuredDynamics = new WebGPUStructuredVelocityDynamics\([\s\S]*?this\.deferPipelineCompilation\);/,
    "structured dynamics must preserve allocation while deferring its pipeline family");
  assert.match(octree,
    /Compile structured dynamics: \$\{entryPoint\} \(\$\{completed\}\/\$\{total\}\)/,
    "the UI must identify the exact structured program currently owning the driver");
  assert.match(octree, /for \(let size = Math\.min\(8, this\.maxLeafSize\); size >= 2;/,
    "startup should warm only regular refinement variants the immutable solver can dispatch");
  assert.match(octree, /for \(let size = this\.maxLeafSize; size >= 16;/,
    "coarse refinement warming should start at the immutable solver maximum");
  assert.doesNotMatch(octree, /for \(let size = 32; size >= 2;/,
    "regular refinement must not compile the coarse-only 16/32 variants");
  assert.doesNotMatch(octree, /retired factor-one pressure lane|coarseOnlySurfaceTracking\)\s*\{\s*throw/,
    "factor-one must reach its retained compact-coarse authority instead of failing initialization");
  assert.match(octree, /this\.coarseOnlySurfaceTracking = options\.globalFineLevelSetFactor === 1/,
    "factor one must still select the compact-coarse surface authority");
  assert.doesNotMatch(spgrid,
    /geometricAggregateTransfers|FactorOneDensePressureShadow|FactorOneDenseCorrection/,
    "factor one must use the production generic sparse SPGrid path without the retired dense experiment");
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

// Construction is paid once per benchmark arm and every arm is a fresh
// process, so shader compilation lands inside the A/B loop's wall clock in
// full. Batching independent compile tasks is worth ~65% of that phase -- but
// only if the batch cannot change what the serial runner produced.
test("independent pipeline compile tasks run concurrently", async () => {
  const signal = new AbortController().signal;
  const runner = new GPUInitializationTaskRunner(() => {}, signal);
  let live = 0, peak = 0;
  const compile = (id: string) => ({
    id, phase: "solver-pipelines" as const, label: id, dependencies: ["allocate"],
    run: async () => {
      live += 1; peak = Math.max(peak, live);
      await new Promise((resolve) => setTimeout(resolve, 15));
      live -= 1;
    },
  });
  await runner.run([
    { id: "allocate", phase: "allocation", label: "Allocate", run: () => {} },
    compile("a"), compile("b"), compile("c"),
  ]);
  assert.equal(peak, 3, "the three independent compile tasks must overlap");
  assert.equal(runner.completedCount, 4);
});

test("a compile batch stops at a dependency on a task inside it", async () => {
  const signal = new AbortController().signal;
  const runner = new GPUInitializationTaskRunner(() => {}, signal);
  const order: string[] = [];
  let firstFinished = false;
  await runner.run([
    { id: "allocate", phase: "allocation", label: "Allocate", run: () => {} },
    { id: "first", phase: "solver-pipelines", label: "first", dependencies: ["allocate"],
      run: async () => {
        await new Promise((resolve) => setTimeout(resolve, 15));
        firstFinished = true; order.push("first");
      } },
    // Depends on a task that has NOT completed when the batch would form, so it
    // must not be batched with it.
    { id: "second", phase: "solver-pipelines", label: "second", dependencies: ["first"],
      run: () => {
        assert.ok(firstFinished, "a dependent compile task ran before its dependency");
        order.push("second");
      } },
  ]);
  assert.deepEqual(order, ["first", "second"]);
});

test("parallel pipeline compilation is disabled by FLUID_GPU_PARALLEL_PIPELINE_COMPILE=0", () => {
  const source = readFileSync(new URL("../lib/gpu-initialization.ts", import.meta.url), "utf8");
  // Exactly "0" disables, so an unset or empty variable keeps the batch. The
  // serial runner stays reachable because it is the definition of the result.
  assert.match(source, /FLUID_GPU_PARALLEL_PIPELINE_COMPILE\s*!==\s*"0"/);
  assert.match(source, /phase !== "solver-pipelines"|candidate\.phase !== "solver-pipelines"/);
});
