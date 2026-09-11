import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { cloneScene, defaultScene } from "../lib/core/model";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { sparseCM12DawnDefaultOptions } from "../lib/harness/sparse-cm12-dawn-defaults";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;
// Dawn's instance must outlive asynchronous device work in direct test runs.
const liveDawnInstances = new Set<GPU>();

dawnTest("authored re-rung does not consume or overwrite world-growth pages",
  { timeout: 120_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-topology-budget-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    let gpu: GPU | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU; globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      liveDawnInstances.add(gpu);
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
        scene.fluid.refinementRegions = [{ id: "whole-domain-rung",
          rule: "minimum-cell-size", minimumCellSize_cells: 1, maximumCellSize_cells: 1,
          min_m: { x: -1, y: -1, z: -1 }, max_m: { x: 1, y: 1, z: 1 } }];
        const defaults = sparseCM12DawnDefaultOptions();
        solver = await WebGPUAdaptiveMassSolver.createAsync(
          device, scene, "balanced", undefined, {
            ...defaults, topologyPageBudget,
          }, () => {},
        );
        await solver.waitForSimulationReady();
        const initialDensity = (await solver.readDiagnosticFields()).density;
        for (let step = 1; step <= 3; step += 1) {
          const edited = structuredClone(scene);
          const width = step === 2 ? 1 : 8;
          edited.fluid.refinementRegions![0] = { ...edited.fluid.refinementRegions![0]!,
            minimumCellSize_cells: width, maximumCellSize_cells: width };
          solver.applySceneUniforms(edited);
          while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
          await solver.awaitFrameCompletion?.();
          await device.queue.onSubmittedWorkDone();
          await solver.assertSimulationHealthy();
          const after = await solver.readGPUActivityPolicy();
          assert.equal(after.faultFlags, 0);
          assert.equal(after.commitFailed, false);
          assert.equal(after.topologyPageAllocator.freePages, topologyPageBudget);
          assert.equal(after.topologyPageAllocator.allocationCancellations, 0);
          assert.ok(after.bricks.filter((brick) => brick.active).every((brick) =>
            8 * brick.spanBricks / brick.acceptedResolution === width),
          "each authored edit must actually commit its requested rung");
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
      if (gpu) liveDawnInstances.delete(gpu);
    }
  });
