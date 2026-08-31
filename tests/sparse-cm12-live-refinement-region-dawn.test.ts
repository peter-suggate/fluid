import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import {
  SPARSE_CM12_ACTIVITY_POLICY,
  SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
  SPARSE_CM12_SHARPENING_TRACE_STEPS,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("Sparse CM12 pre-catalogues a hard minimum-cell-size region",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-live-refinement-region-dawn.test.ts");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const uncaptured: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        uncaptured.push(event.error.message);
      });
      device.pushErrorScope("validation");

      const constrained = createMinimalPowerDamBreak64Scene();
      constrained.duration_s = 3 * CM12_PAPER_DT_S;
      constrained.fluid.refinementRegions = [{
        id: "whole-domain-two-cell-floor",
        rule: "minimum-cell-size",
        minimumCellSize_cells: 2,
        min_m: { x: -0.5 * constrained.container.width_m, y: 0,
          z: -0.5 * constrained.container.depth_m },
        max_m: { x: 0.5 * constrained.container.width_m,
          y: constrained.container.height_m,
          z: 0.5 * constrained.container.depth_m },
      }];
      const solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, constrained, "balanced", undefined, {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          surfaceFineRings: 1,
          timeStep: "paper",
          activityPolicy: {
            ...SPARSE_CM12_ACTIVITY_POLICY,
            topologyCadenceSteps: 1,
            demoteEpochs: 1,
            prepareBricksPerFrame: 512,
          },
          pressureIterations: 40,
          pressureRelativeTolerance: 0.001,
          sharpeningDistance: SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
          sharpeningTraceSteps: SPARSE_CM12_SHARPENING_TRACE_STEPS,
        },
        () => {},
      );
      try {
        while (!solver.advanceTo(2 * CM12_PAPER_DT_S, [])) {
          await new Promise(setImmediate);
        }
        await device.queue.onSubmittedWorkDone();
        const constrainedActivity = await solver.readGPUActivityPolicy();
        const constrainedActive = constrainedActivity.bricks.filter((brick) => brick.active);
        assert.ok(constrainedActive.length > 0);
        assert.ok(constrainedActive.every((brick) => brick.acceptedResolution <= 4),
          `active topology escaped the hard two-cell floor: ${JSON.stringify(
            constrainedActive.filter((brick) => brick.acceptedResolution > 4).map((brick) => ({
              coordinate: brick.coordinate,
              accepted: brick.acceptedResolution,
              candidate: brick.candidateResolution,
              planned: brick.plannedResolution,
              reasons: brick.reasons,
              planReasons: brick.planReasons,
            })))}`);
      } finally {
        solver.destroy();
      }

      const validation = await device.popErrorScope();
      assert.equal(validation, null, validation?.message);
      assert.deepEqual(uncaptured, []);
    } finally {
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
