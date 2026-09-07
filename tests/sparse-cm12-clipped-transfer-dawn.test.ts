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
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("clipped boundary re-rung preserves mass through the resident transaction",
  { timeout: 120_000 }, async (t) => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-clipped-transfer-dawn.test.ts");
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
      const topologyPageBudget = 0;
      const scene = cloneScene(defaultScene);
      scene.rigidBodies = [];
      scene.container = { ...scene.container, width_m: 0.65, height_m: 0.5,
        depth_m: 0.45, fillFraction: 1 };
      scene.voxelDomain.finestCellSize_m = 0.05;
      scene.fluid.initialCondition = "dam-break";
      scene.fluid.initialDamBreakDimensions_m = { x: 0.65, y: 0.5, z: 0.45 };
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
      let clippedCommitted = false;
      for (let step = 1; step <= 4; step += 1) {
        if (step === 3 || step === 4) {
          const edited = structuredClone(scene);
          const width = step === 3 ? 8 : 1;
          edited.fluid.refinementRegions = [{ id: "whole-domain-rung",
            rule: "minimum-cell-size", minimumCellSize_cells: width,
            maximumCellSize_cells: width,
          min_m: { x: -1, y: -1, z: -1 }, max_m: { x: 1, y: 1, z: 1 } }];
          solver.applySceneUniforms(edited);
        }
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        await solver.assertSimulationHealthy();
        const after = await solver.readGPUActivityPolicy();
        clippedCommitted ||= after.bricks.some((b) => b.active
          && b.coordinate.some((q, axis) => (q + b.spanBricks) * 8 > [13, 10, 9][axis]!)
          && b.acceptedResolution < 8);
        for (const brick of after.bricks) {
          assert.ok(Math.abs(brick.transferMassErrorFineCells) <= 1e-4);
          assert.ok(Math.abs(brick.transferGammaErrorFineCells) <= 1e-4);
          assert.ok(brick.transferMomentumErrorFineCells.every((e) => Math.abs(e) <= 1e-4));
        }
        assert.equal(after.faultFlags, 0);
        assert.equal(after.commitFailed, false);
        assert.equal(after.topologyPageAllocator.freePages, topologyPageBudget);
        assert.equal(after.topologyPageAllocator.allocationCancellations, 0);
        if (step === 3 || step === 4) {
          assert.ok(after.bricks.filter((b) => b.active).every((b) =>
            8 * b.spanBricks / b.acceptedResolution === (step === 3 ? 8 : 1)),
            "the live edit must coarsen and then refine every clipped leaf");
        }
        assert.ok(after.bricks.every((brick) => brick.topologyPage === undefined),
          "authored transfer must not borrow a dynamic WDR identity");
      }
      assert.ok(clippedCommitted, "a clipped boundary leaf must actually commit a coarser rung");
      const fields = await solver.readDiagnosticFields();
      const mass = (density: ArrayLike<number>) => Array.from(density)
        .reduce((sum, value) => sum + Math.max(0, value), 0);
      const massError = mass(fields.density) - mass(initialDensity);
      assert.ok(Math.abs(massError) <= 1e-4,
        `budget=${topologyPageBudget}, mass error=${massError}, initial mass=${mass(initialDensity)}`);
      const stats = await solver.readStats();
      const activity = await solver.readGPUActivityPolicy();
      t.diagnostic(JSON.stringify({ massError, finalResolution: 8,
        leafCount: activity.bricks.length }));
      assert.deepEqual(stats.adaptiveTopologyPageAllocator, activity.topologyPageAllocator,
        "ordinary UI diagnostics must expose the same allocator receipt as the QA census");
      solver.destroy(); solver = undefined;
      const validation = await device.popErrorScope();
      assert.equal(validation, null, validation?.message);
      assert.deepEqual(errors, []);
    } finally {
      solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
    }
  });
