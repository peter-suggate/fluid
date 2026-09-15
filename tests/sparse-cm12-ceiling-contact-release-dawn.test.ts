import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("ceiling contact releases before free-fall impact", {
  timeout: 30_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test",
    "tests/sparse-cm12-ceiling-contact-release-dawn.test.ts");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, [
      `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
      "enable-dawn-features=disable_blob_cache",
    ]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter);
    device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    const scene = sceneDocument(getSceneDefinition("ceiling-slab-drop"));
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      brickFineResolution: "8", resolutionMode: "adaptive",
      selectorMode: "surface", surfaceFineRings: 1, timeStep: "paper",
    });
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced",
      values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();

    for (let step = 1; step <= 3; step += 1) {
      while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) {
        await new Promise<void>(setImmediate);
      }
      await solver.awaitFrameCompletion?.();
      await device.queue.onSubmittedWorkDone();
      await solver.assertSimulationHealthy();
      const levelSet = await solver.readAdaptiveLevelSetQA(step === 1);

      if (step === 1) {
        const ceilingPhi = (levelSet.vertices ?? []).filter(vertex =>
          vertex.positionFine[1] === 16
          && vertex.positionFine[0] > 8 && vertex.positionFine[0] < 16
          && vertex.positionFine[2] > 8 && vertex.positionFine[2] < 16)
          .map(vertex => vertex.phiFine);
        assert.ok(ceilingPhi.length >= 16,
          `step 1 has only ${ceilingPhi.length} interior ceiling samples`);
        assert.ok(Math.min(...ceilingPhi) > 0,
          `step 1 ceiling phi did not detach: ${Math.min(...ceilingPhi)}`);
      }
    }
  } finally {
    try { solver?.destroy(); device?.destroy(); }
    finally { await releaseWebGPUExclusiveLock(); }
  }
});
