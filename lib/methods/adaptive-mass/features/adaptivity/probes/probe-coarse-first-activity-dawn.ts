/** Exact pool-impact reproduction; run sequentially in each source tree. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { fluidPipelinePhaseCosts, measureFluidPipelineStage } from "../../../../../core/fluid-pipeline";
import { resolveMethodValues } from "../../../../../core/method-contract";
import type { PerformanceTrace } from "../../../../../core/performance-trace";
import { sceneDocument } from "../../../../../core/scene-definition";
import { getSceneDefinition } from "../../../../../core/scenes";
import { usePerformanceInstrumentationStore } from "../../../../../core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../../../../../core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../../../../../harness/webgpu-smoke-isolation";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from "../../../adaptive-mass-frame-pipeline";
import { adaptiveMassMethod } from "../../../method";
import type { WebGPUAdaptiveMassSolver } from "../../../webgpu-adaptive-mass-solver";

const steps = Number(process.env.ACTIVITY_STEPS ?? 30);
assert.ok(Number.isSafeInteger(steps) && steps > 0);
const modulePath = process.env.WEBGPU_NODE_MODULE;
assert.ok(modulePath, "Set WEBGPU_NODE_MODULE to the native Dawn module");
const startedAt = new Date().toISOString();
const sourceHashes = Object.fromEntries(await Promise.all([
  "webgpu-sparse-cm12-resident.wgsl.ts", "webgpu-sparse-cm12-resident.ts",
  "webgpu-adaptive-mass-solver.ts",
].map(async name => [name, createHash("sha256").update(await readFile(
  new URL(`../../../${name}`, import.meta.url))).digest("hex")])));
await acquireWebGPUExclusiveLock("dawn-probe", "coarse-first-pool-impact activity");
let gpu: GPU | undefined, device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const dawn = await import(pathToFileURL(modulePath).href);
  Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu!.requestAdapter();
  assert.ok(adapter);
  assert.ok(adapter.features.has("timestamp-query"));
  device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"],
    requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  let readbacks: { label: string; bytes: number; duration_ms: number }[] = [];
  if (process.env.ACTIVITY_READBACKS === "1") {
    const createBuffer = device.createBuffer.bind(device);
    device.createBuffer = descriptor => {
      const buffer = createBuffer(descriptor);
      if ((descriptor.usage & GPUBufferUsage.MAP_READ) !== 0) {
        const map = buffer.mapAsync.bind(buffer);
        buffer.mapAsync = async (...args) => {
          const started = performance.now();
          await map(...args);
          readbacks.push({ label: descriptor.label ?? "", bytes: descriptor.size,
            duration_ms: performance.now() - started });
        };
      }
      return buffer;
    };
  }
  device.addEventListener("uncapturederror", event => errors.push(event.error.message));
  usePerformanceInstrumentationStore.getState().setMode("timeline");
  const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact"));
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    selectorMode: "coarse-first", timeStep: "paper",
  });
  solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values,
    undefined, () => {}) as WebGPUAdaptiveMassSolver;
  await solver.waitForSimulationReady();
  const samples = [];
  for (let step = 1; step <= steps; step++) {
    // The runtime samples by wall-clock cadence. Space captures explicitly.
    await new Promise(resolve => setTimeout(resolve, 120));
    readbacks = [];
    const wallStarted = performance.now();
    while (!solver.advanceTo(step / 30, [])) await new Promise(setImmediate);
    await solver.waitForTopologyReady();
    await device.queue.onSubmittedWorkDone();
    const wall_ms = performance.now() - wallStarted;
    assert.equal(solver.info.encodedSteps, step);
    const context = `adaptive-mass:sim-${(step / 30).toFixed(6)}`;
    const deadline = performance.now() + 1000;
    let trace: PerformanceTrace | undefined = solver.readPerformanceTraceSnapshot().physicsTrace;
    while (trace?.context !== context && performance.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 5));
      trace = solver.readPerformanceTraceSnapshot().physicsTrace;
    }
    assert.equal(trace?.context, context, `missing trace for step ${step}`);
    assert.equal(trace!.measurementSource, "gpu-hardware-timestamp");
    const stages = ADAPTIVE_MASS_FLUID_PIPELINE.stages;
    const stage = stages.find(stage => stage.id === "activity-measurement")!;
    const measurement = measureFluidPipelineStage(stage, stages,
      fluidPipelinePhaseCosts(trace!), trace!.total_ms, "on");
    const sample = { step, seconds: step / 30, activity_ms: measurement.duration_ms,
      total_ms: trace!.total_ms, wall_ms, readbacks: [...readbacks],
      topologyPreparation_ms: solver.info.topologyPreparationDurationMs,
      phases: trace!.phases };
    samples.push(sample);
    console.log(JSON.stringify({ step, activity_ms: sample.activity_ms, total_ms: sample.total_ms }));
    assert.deepEqual(errors, []);
  }
  const fields = await solver.readDiagnosticFields(true);
  const densityHash = createHash("sha256").update(new Uint8Array(fields.density.buffer)).digest("hex");
  const receipt = { scene: "coarse-first-pool-impact", selectorMode: "coarse-first", startedAt,
    dt_s: 1 / 30, sourceHashes,
    densityHash, samples, errors };
  await writeFile(process.env.ACTIVITY_OUT ?? "/tmp/coarse-first-activity.json", JSON.stringify(receipt, null, 2));
} finally {
  solver?.destroy(); device?.destroy();
  releaseWebGPUExclusiveLock();
  void gpu;
}
