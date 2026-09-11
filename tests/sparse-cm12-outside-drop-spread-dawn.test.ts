import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { defaultFluidBallRadius_m } from "../lib/core/editor-fluid-volume";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { findSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("an outside-tank ball spreads across both horizontal sparse-world axes",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-outside-drop-spread-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as NodeDawnProvider;
      Object.assign(globalThis, dawn.globals);
      const gpu = createProcessRetainedDawnGPU(dawn,
        [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      const scene = sceneDocument(findSceneDefinition("hydrostatic-power-large-offset")!);
      const radius_m = defaultFluidBallRadius_m(scene);
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {});
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", values, undefined, () => {},
      ) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      while (!solver.advanceTo(CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion?.();
      solver.injectLiquidBall({
        centre_m: {
          x: -scene.container.width_m / 2 - 3 * radius_m,
          y: radius_m,
          z: 0,
        },
        radius_m,
      });
      await device.queue.onSubmittedWorkDone();
      await solver.assertSimulationHealthy();

      for (let step = 2; step <= 32; step += 1) {
        while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
        await solver.awaitFrameCompletion?.();
      }
      await device.queue.onSubmittedWorkDone();
      await solver.assertSimulationHealthy();
      const growth = await solver.readWorldGrowthReceiptQA();
      assert.ok(growth.dynamicLiquidMassFineCells > 1, JSON.stringify(growth));
      assert.ok(growth.dynamicLiquidBoundsFine, JSON.stringify(growth));
      const spanX = growth.dynamicLiquidBoundsFine.maximumExclusive[0]
        - growth.dynamicLiquidBoundsFine.minimum[0];
      const spanZ = growth.dynamicLiquidBoundsFine.maximumExclusive[2]
        - growth.dynamicLiquidBoundsFine.minimum[2];
      assert.ok(Math.max(spanX, spanZ) <= 2 * Math.min(spanX, spanZ),
        `outside drop spread anisotropically (${spanX} x ${spanZ}); `
        + JSON.stringify(growth));
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
