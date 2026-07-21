import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { initializeRigidBodies } from "../lib/rigid-body";
import type { GPUInitializationProgress, GPUSolverInstance, MethodParamValues } from "../lib/methods/types";
import { octreeMethod } from "../lib/methods/octree";
import { optionalFluidDeviceFeatures, requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { createSmokeScenario } from "./webgpu-smoke-scenarios";
import {
  parseWebGPUBringupStage,
  reachedSolverResourceBoundary,
  stageIncludesComputeSentinel,
  stageIncludesSparseT0,
} from "./webgpu-bringup-stages";

const COMPUTE_SENTINEL_WORD = 0x4f43_5452;
const EXCLUSIVE_LOCK = "/tmp/fluid-webgpu-exclusive.lock";

class SolverResourceBoundary extends Error {
  constructor() { super("solver resources compiled and allocated before sparse t=0 warmup"); }
}

/**
 * Wait for already-created resources and their error events to reach JS.
 * onSubmittedWorkDone does not submit a command buffer, and the following
 * event-loop turn lets Dawn dispatch an uncapturederror queued by validation.
 */
async function flushGPUErrorDelivery(device: GPUDevice): Promise<void> {
  await device.queue.onSubmittedWorkDone();
  await new Promise<void>((resolve) => setImmediate(resolve));
}

async function acquireExclusiveGPUProcessLock() {
  try {
    await mkdir(EXCLUSIVE_LOCK);
  } catch (error) {
    let owner = "unknown owner";
    try { owner = await readFile(`${EXCLUSIVE_LOCK}/owner.json`, "utf8"); } catch { /* best-effort diagnostic */ }
    throw new Error(`Refusing concurrent GPU bring-up; ${EXCLUSIVE_LOCK} already exists (${owner}). Remove it only after confirming no Dawn or browser GPU run is active.`, { cause: error });
  }
  await writeFile(`${EXCLUSIVE_LOCK}/owner.json`, JSON.stringify({ pid: process.pid, startedAt: new Date().toISOString(), kind: "dawn-bringup" }));
}

async function assertComputeSentinel(device: GPUDevice): Promise<number> {
  const output = device.createBuffer({ label: "Bring-up compute sentinel", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ label: "Bring-up compute sentinel readback", size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const module = device.createShaderModule({
      label: "Bring-up compute sentinel",
      code: "@group(0) @binding(0) var<storage, read_write> output: array<u32>; @compute @workgroup_size(1) fn sentinel() { output[0] = 0x4f435452u; }",
    });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "sentinel" } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder({ label: "Bring-up compute sentinel" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const observed = new Uint32Array(readback.getMappedRange())[0];
    readback.unmap();
    if (observed !== COMPUTE_SENTINEL_WORD) throw new Error(`compute sentinel returned 0x${observed.toString(16).padStart(8, "0")} instead of 0x${COMPUTE_SENTINEL_WORD.toString(16)}`);
    return observed;
  } finally {
    output.destroy(); readback.destroy();
  }
}

function solverValues(): MethodParamValues {
  const values = octreeMethod.presetFor("balanced");
  values.surfaceColumns = Number(process.env.FLUID_SURFACE_COLUMNS ?? 384);
  values.leafSolver = process.env.FLUID_OCTREE_LEAF_SOLVER ?? "mgpcg";
  values.faceVelocityTransport = process.env.FLUID_OCTREE_FACE_TRANSPORT === "0" ? "off" : "on";
  values.powerDiagramProjection = process.env.FLUID_OCTREE_POWER_PROJECTION ?? "authoritative";
  values.globalFineLevelSetFactor = process.env.FLUID_OCTREE_GLOBAL_FINE_FACTOR ?? "4";
  return values;
}

await acquireExclusiveGPUProcessLock();
let device: GPUDevice | undefined;
let solver: GPUSolverInstance | undefined;
let lost: GPUDeviceLostInfo | undefined;
try {
  const stage = parseWebGPUBringupStage(process.env.FLUID_BRINGUP_STAGE);
  console.log(JSON.stringify({ phase: "bringup-start", stage, pid: process.pid, browserGPURequiredClosed: true }));
  const modulePath = process.env.WEBGPU_NODE_MODULE;
  const webgpuModule = modulePath ? await import(pathToFileURL(modulePath).href) : await import("webgpu");
  const { create, globals } = webgpuModule as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  Reflect.deleteProperty(globalThis, "Worker");
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });

  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("Dawn did not expose a WebGPU adapter");
  const requiredFeatures: GPUFeatureName[] = [
    ...(adapter.features.has("timestamp-query") && process.env.FLUID_DISABLE_TIMESTAMPS !== "1" ? ["timestamp-query" as GPUFeatureName] : []),
    ...optionalFluidDeviceFeatures(adapter.features),
  ];
  device = await adapter.requestDevice({ requiredFeatures, requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  void device.lost.then((info) => { lost = info; });
  console.log(JSON.stringify({ phase: "adapter-device", stage, passed: true, features: requiredFeatures }));
  if (stage === "adapter-device") process.exitCode = 0;
  else {
    if (stageIncludesComputeSentinel(stage)) {
      const observed = await assertComputeSentinel(device);
      console.log(JSON.stringify({ phase: "compute-sentinel", stage, passed: true, value: observed }));
    }
    if (stage !== "compute-sentinel") {
      const scenario = createSmokeScenario("dam-break-ui");
      const validationErrors: string[] = [];
      let lastInitializationCompleted = 0;
      device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));
      const progress = (snapshot: GPUInitializationProgress) => {
        const boundary = snapshot.completed > lastInitializationCompleted ? "completed" : "starting";
        console.log(JSON.stringify({ record: "solver-initialization", boundary, stage, ...snapshot }));
        lastInitializationCompleted = Math.max(lastInitializationCompleted, snapshot.completed);
        if (stage === "solver-resources" && reachedSolverResourceBoundary(snapshot)) throw new SolverResourceBoundary();
      };
      try {
        solver = await octreeMethod.createSolverAsync!(device, scenario.scene, "balanced", solverValues(), undefined, progress);
      } catch (error) {
        if (!(error instanceof SolverResourceBoundary)) throw error;
        await flushGPUErrorDelivery(device);
        if (lost) throw new Error(`device lost before solver resource boundary: ${lost.reason} ${lost.message}`);
        if (validationErrors.length > 0) {
          throw new Error(`solver-resources validation failed: ${validationErrors.join("; ")}`);
        }
        console.log(JSON.stringify({ phase: "solver-resources", stage, passed: true, stoppedBeforeTask: "solver.warmup" }));
      }
      if (stageIncludesSparseT0(stage)) {
        if (!solver?.initialSparseAuthorityReady) throw new Error("solver returned without complete sparse t=0 authority");
        await flushGPUErrorDelivery(device);
        if (lost) throw new Error(`device lost during sparse t=0 publication: ${lost.reason} ${lost.message}`);
        if (validationErrors.length > 0) {
          throw new Error(`sparse-t0 validation failed: ${validationErrors.join("; ")}`);
        }
        console.log(JSON.stringify({ phase: "sparse-t0", stage, passed: true, grid: [solver.info.nx, solver.info.ny, solver.info.nz], allocatedBytes: solver.info.allocatedBytes }));
      }
      if (stage === "one-step") {
        const requestedTime_s = scenario.scene.numerics.maxDt_s;
        const accepted = solver!.advanceTo(requestedTime_s, initializeRigidBodies(scenario.scene.rigidBodies));
        if (!accepted) throw new Error("solver refused the first bounded advance");
        const info = await solver!.readStats();
        await flushGPUErrorDelivery(device);
        if (lost) throw new Error(`device lost during first step: ${lost.reason} ${lost.message}`);
        if ((info.encodedSteps ?? 0) !== 1 || (info.submittedTime_s ?? 0) < requestedTime_s) {
          throw new Error(`one-step checkpoint did not complete exactly one submission: steps=${info.encodedSteps}, submitted=${info.submittedTime_s}`);
        }
        if (validationErrors.length > 0 || info.gpuValidationError) throw new Error(`one-step validation failed: ${[...validationErrors, info.gpuValidationError].filter(Boolean).join("; ")}`);
        console.log(JSON.stringify({ phase: "one-step", stage, passed: true, encodedSteps: info.encodedSteps, submittedTime_s: info.submittedTime_s, completedTime_s: info.completedTime_s, nonFiniteCount: info.nonFiniteCount, stabilityFlags: info.stabilityFlags }));
      }
    }
  }
} finally {
  solver?.destroy();
  device?.destroy();
  Reflect.deleteProperty(globalThis, "navigator");
  await rm(EXCLUSIVE_LOCK, { recursive: true, force: true });
}
