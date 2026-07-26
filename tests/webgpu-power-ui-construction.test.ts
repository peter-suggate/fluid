import assert from "node:assert/strict";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { octreeMethod, octreeSolverOptions } from "../lib/methods/octree";
import { fluidExecutionDeviceFeatures } from "../lib/gpu-startup";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { WebGPUUniformEulerianSolver } from "../lib/webgpu-uniform-eulerian";
import { OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES } from "../lib/webgpu-octree";
import { WEBGPU_EXCLUSIVE_LOCK } from "../tools/webgpu-smoke-isolation";
import { createSmokeScenario } from "../tools/webgpu-smoke-scenarios";

const PORTABLE_STORAGE_BUFFER_LIMIT = 10;
// Dawn's Node/Metal wrapper owns native instance lifetime separately from the
// WebGPU adapter/device wrappers. Retain both until process exit so garbage
// collection cannot tear down the instance while asynchronous pipeline
// construction is still in flight.
const retainedNativeGPUs: GPU[] = [];
const retainedDevices: GPUDevice[] = [];

test("Dawn constructs the exact production power UI graph at the portable storage limit", {
  skip: !process.env.WEBGPU_NODE_MODULE
    && "set WEBGPU_NODE_MODULE for the production UI construction gate",
  timeout: 120_000,
}, async (t) => {
  try {
    await mkdir(WEBGPU_EXCLUSIVE_LOCK);
  } catch (error) {
    let owner = "unknown owner";
    try { owner = await readFile(`${WEBGPU_EXCLUSIVE_LOCK}/owner.json`, "utf8"); } catch { /* diagnostic only */ }
    throw new Error(`Refusing concurrent production UI construction; ${WEBGPU_EXCLUSIVE_LOCK} already exists (${owner}). Remove it only after confirming its owner PID is gone and no Dawn or browser GPU run is active.`, { cause: error });
  }
  try {
    await writeFile(`${WEBGPU_EXCLUSIVE_LOCK}/owner.json`, JSON.stringify({
      pid: process.pid,
      startedAt: new Date().toISOString(),
      kind: "dawn-ui-construction",
      target: "tests/webgpu-power-ui-construction.test.ts",
    }));
  } catch (error) {
    await rm(WEBGPU_EXCLUSIVE_LOCK, { recursive: true, force: true });
    throw error;
  }
  t.after(async () => {
    await rm(WEBGPU_EXCLUSIVE_LOCK, { recursive: true, force: true });
  });

  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const workerDescriptor = Object.getOwnPropertyDescriptor(globalThis, "Worker");
  Reflect.deleteProperty(globalThis, "Worker");
  t.after(() => {
    if (workerDescriptor) Object.defineProperty(globalThis, "Worker", workerDescriptor);
  });

  const nativeGpu = dawn.create([
    `backend=${process.env.WEBGPU_BACKEND ?? "metal"}`,
  ]);
  retainedNativeGPUs.push(nativeGpu);
  const adapter = await nativeGpu.requestAdapter();
  assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= PORTABLE_STORAGE_BUFFER_LIMIT);

  const requiredLimits = requiredFluidDeviceLimits(adapter.limits);
  requiredLimits.maxStorageBuffersPerShaderStage = PORTABLE_STORAGE_BUFFER_LIMIT;
  const requiredFeatures = fluidExecutionDeviceFeatures(adapter.features);
  assert.ok(requiredFeatures.includes("subgroups"),
    "the production construction gate requires the M1 Max subgroup feature");
  const device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
  retainedDevices.push(device);
  assert.equal(device.limits.maxStorageBuffersPerShaderStage, PORTABLE_STORAGE_BUFFER_LIMIT,
    "the regression must not inherit a non-portable adapter storage-buffer limit");
  let bindGroupSequence = 0;
  const solverDevice = new Proxy(device, {
    get(target, property) {
      if (property === "createBindGroup") {
        return (descriptor: GPUBindGroupDescriptor) => {
          const sequence = bindGroupSequence++;
          const callsite = new Error().stack?.split("\n").find((line) => line.includes("/lib/"))?.trim();
          const label = descriptor.label || `Production UI bind group ${sequence}${callsite ? ` · ${callsite}` : ""}`;
          const entries = Array.from(descriptor.entries);
          return target.createBindGroup({ ...descriptor, entries, label });
        };
      }
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  }) as GPUDevice;

  const uncaptured: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncaptured.push(event.error.message);
  });
  device.pushErrorScope("validation");

  let solver: WebGPUUniformEulerianSolver | undefined;
  let scopeOpen = true;
  try {
    const scenario = createSmokeScenario("dam-break-ui");
    const values = octreeMethod.presetFor("balanced");
    solver = new WebGPUUniformEulerianSolver(solverDevice, scenario.scene, "balanced", undefined, {
      ...octreeSolverOptions(scenario.scene, "balanced", values),
      deferPipelineCompilation: true,
    });

    // Exercise the exact production task list through its construction
    // boundary. Warmup is deliberately excluded: this gate catches shader,
    // capability, and layout regressions before runtime synchronization and
    // numerical validation belong in the two-step smoke.
    const tasks = (solver as unknown as {
      initializationTasks(): Array<{
        phase: string;
        run(signal: AbortSignal): void | Promise<void>;
      }>;
    }).initializationTasks().filter((task) => task.phase !== "warmup");
    const signal = new AbortController().signal;
    for (const task of tasks) await task.run(signal);

    // Encoding without finishing/submitting constructs the bind groups that
    // are created lazily by setup paths. It catches explicit-layout resource
    // mismatches without turning this construction test into an execution
    // or synchronization test.
    const projection = (solver as unknown as {
      octreeProjection?: {
        encodeInitialSparseAuthorityPhase(encoder: GPUCommandEncoder,
          phase: typeof OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES[number]["id"]): void;
      };
    }).octreeProjection;
    assert.ok(projection);
    for (const phase of OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES) {
      projection.encodeInitialSparseAuthorityPhase(device.createCommandEncoder({
        label: `Production power UI no-submit construction audit: ${phase.id}`,
      }), phase.id);
    }

    const validationError = await device.popErrorScope();
    scopeOpen = false;
    assert.equal(validationError, null, validationError?.message);
    assert.deepEqual(uncaptured, []);
    assert.deepEqual(
      [solver.info.nx, solver.info.ny, solver.info.nz],
      [24, 18, 16],
      "the construction gate must retain the exact browser dam-break lattice",
    );
    assert.ok(solver.structuredVelocityControl,
      "production structured velocity authority was not constructed");
    assert.ok(solver.structuredBoundaryControl,
      "production structured boundary authority was not constructed");
    assert.ok(solver.globalFineLevelSetSource, "production global-fine authority was not constructed");
    assert.equal(solver.globalFineLevelSetSource.plan.fineFactor, 4);
    assert.equal(solver.initialSparseAuthorityReady, false,
      "a no-submit construction audit must not claim published t=0 authority");
  } finally {
    solver?.destroy();
    if (scopeOpen) await device.popErrorScope().catch(() => null);
    device.destroy();
  }
});
