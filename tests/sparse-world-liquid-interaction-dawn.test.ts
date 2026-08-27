import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneAtContainerExtents } from "../lib/core/scene-scale";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

const sum = (values: Float32Array): number => values.reduce(
  (total, value) => total + value, 0);

dawnTest("a public sparse-world edit adds a dropped liquid ball",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-world-liquid-interaction-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      const scene = sceneAtContainerExtents(
        sceneDocument(getSceneDefinition("water-box-tank-fill")),
        { width_m: 1.6, height_m: 2.4, depth_m: 1.6 },
      );
      scene.rigidBodies = [];
      scene.container.fillFraction = 0.2;
      scene.voxelDomain.finestCellSize_m = 0.05;
      scene.fluid.surfaceTension_N_m = 0;
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined, {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          surfaceFineRings: 1,
          timeStep: "paper",
          pressureIterations: 32,
          pressureRelativeTolerance: 0,
        }, () => {});
      await solver.waitForSimulationReady();

      const beforeMass = sum((await solver.readDiagnosticFields()).density);
      const beforeGeneration = solver.sparseWorld.status().acceptedGeneration;
      solver.injectLiquidBall({
        centre_m: { x: 0, y: 1.7, z: 0 },
        radius_m: 0.16,
      });
      assert.equal(solver.sparseWorld.status().acceptedGeneration, beforeGeneration + 1,
        "the interaction must publish exactly one generation");
      await device.queue.onSubmittedWorkDone();

      const afterMass = sum((await solver.readDiagnosticFields()).density);
      const activity = await solver.readGPUActivityPolicy();
      const growth = await solver.readWorldGrowthReceiptQA();
      const wetBricks = activity.bricks.filter((brick) => brick.active
        && (brick.reasons & 64) !== 0);
      const representedMass = afterMass + growth.dynamicLiquidMassFineCells;
      assert.ok(representedMass > beforeMass + 20,
        `the dropped ball must add visible liquid mass (${beforeMass} -> ${representedMass}); `
        + `records=${activity.bricks.length}, resident=${activity.residentBrickCount}, `
        + `wet=${wetBricks.length}, `
        + `new=${activity.newlyActivatedBrickCount}, faults=${activity.faultFlags}`);
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
