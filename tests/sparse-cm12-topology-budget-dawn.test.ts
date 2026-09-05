import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { cloneScene, defaultScene } from "../lib/core/model";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("authored re-rung does not consume or overwrite world-growth pages",
  { timeout: 120_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-topology-budget-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU; globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const errors: string[] = [];
      device.addEventListener("uncapturederror", (event) => errors.push(event.error.message));
      device.pushErrorScope("validation");
      for (const topologyPageBudget of [0, 1, 32]) {
        const scene = cloneScene(defaultScene);
        scene.rigidBodies = [];
        scene.container = { ...scene.container, width_m: 0.8, height_m: 0.8,
          depth_m: 0.8, fillFraction: 1 };
        scene.voxelDomain.finestCellSize_m = 0.05;
        scene.fluid.initialCondition = "dam-break";
        scene.fluid.initialDamBreakDimensions_m = { x: 0.8, y: 0.8, z: 0.8 };
        scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
        const defaults = adaptiveMassSolverOptions({});
        solver = await WebGPUAdaptiveMassSolver.createAsync(
          device, scene, "balanced", undefined, {
            ...defaults, topologyPageBudget, initialResolutionForQA: 8,
            maximumMacroSpanBricks: 1, pressureIterations: 8,
            // Isolate topology transfer from the iterative sharpening dose.
            // Production scalar transforms retain their canonical suite gates.
            gammaDiffusionEnabled: false, surfaceSharpeningEnabled: false,
            activityPolicy: { ...defaults.activityPolicy!,
              topologyCadenceSteps: 1, demoteEpochs: 1, prepareBricksPerFrame: 256 },
          }, () => {},
        );
        await solver.waitForSimulationReady();
        const initialDensity = (await solver.readDiagnosticFields()).density;
        for (let step = 1; step <= 3; step += 1) {
          assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
          await device.queue.onSubmittedWorkDone();
          const after = await solver.readGPUActivityPolicy();
          assert.equal(after.faultFlags, 0);
          assert.equal(after.commitFailed, false);
          assert.equal(after.topologyPageAllocator.freePages, topologyPageBudget);
          assert.equal(after.topologyPageAllocator.allocationCancellations, 0);
          assert.ok(after.bricks.some((brick) => brick.active && brick.acceptedResolution < 8));
          assert.ok(after.bricks.every((brick) => brick.topologyPage === undefined),
            "authored transfer must not borrow a dynamic WDR identity");
        }
        const fields = await solver.readDiagnosticFields();
        const mass = (density: ArrayLike<number>) => Array.from(density)
          .reduce((sum, value) => sum + Math.max(0, value), 0);
        const massError = mass(fields.density) - mass(initialDensity);
        assert.ok(Math.abs(massError) <= 1e-4,
          `budget=${topologyPageBudget}, mass error=${massError}, initial mass=${mass(initialDensity)}`);
        const stats = await solver.readStats();
        const activity = await solver.readGPUActivityPolicy();
        assert.deepEqual(stats.adaptiveTopologyPageAllocator, activity.topologyPageAllocator,
          "ordinary UI diagnostics must expose the same allocator receipt as the QA census");
        solver.destroy(); solver = undefined;
      }
      const validation = await device.popErrorScope();
      assert.equal(validation, null, validation?.message);
      assert.deepEqual(errors, []);
    } finally {
      solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
    }
  });
