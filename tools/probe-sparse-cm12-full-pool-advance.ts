import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { getHeapStatistics } from "node:v8";
import "./probe-sparse-cm12-generation-host";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

async function main() {
  assert.ok(process.env.WEBGPU_NODE_MODULE);
  await acquireWebGPUExclusiveLock("dawn-probe", "cm12-full-pool-generation-host");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  let gpu: GPU | undefined;
  const started = performance.now();
  let maximumObservedHeap = 0;
  try {
    const { create, globals } = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE).href);
    Object.assign(globalThis, globals);
    gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => errors.push(event.error.message));
    const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact"));
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      selectorMode: "coarse-first", timeStep: "scene",
    });
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    console.log(JSON.stringify({ probe: "cm12-full-pool-configuration", steps: 30,
      timeStep_s: 1 / 60, heapLimitBytes: getHeapStatistics().heap_size_limit,
      initialAllocatedBytes: solver.info.allocatedBytes, ...process.memoryUsage() }));
    for (let step = 1; step <= 30; step++) {
      const start = performance.now();
      while (!solver.advanceTo(step / 60, [])) await new Promise(setImmediate);
      await solver.waitForTopologyReady();
      assert.equal(solver.info.encodedSteps, step);
      await solver.assertSimulationHealthy();
      const memory = process.memoryUsage();
      maximumObservedHeap = Math.max(maximumObservedHeap, memory.heapUsed);
      console.log(JSON.stringify({ probe: "cm12-full-pool-advance", step,
        milliseconds: performance.now() - start, ...memory,
        topologyGeneration: solver.info.topologyGenerationCount ?? 0,
        topologyError: solver.info.topologyGenerationError ?? null,
        topologyDeferred: solver.info.topologyGenerationDeferred ?? null,
        topologyStaleCount: solver.info.topologyGenerationStaleCount ?? 0,
        allocatedBytes: solver.info.allocatedBytes }));
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ probe: "cm12-full-pool-completed", steps: 30,
      elapsedMilliseconds: performance.now() - started, maximumObservedHeap,
      topologyGeneration: solver.info.topologyGenerationCount ?? 0, validationErrors: errors }));
  } finally {
    solver?.destroy(); device?.destroy();
    await releaseWebGPUExclusiveLock();
    void gpu;
  }
}
void main().catch(error => { console.error(error); process.exitCode = 1; });
