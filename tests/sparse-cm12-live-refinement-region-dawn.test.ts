import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import type { SceneDescription } from "../lib/core/model";
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

dawnTest("Sparse CM12 adds and removes a refinement region without rebuilding",
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

      const original = createMinimalPowerDamBreak64Scene();
      original.duration_s = 4 * CM12_PAPER_DT_S;
      const solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, original, "balanced", undefined, {
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
        const generationZero = await solver.readGPUActivityPolicy();
        const initialLeafCount = 1 + Math.max(...generationZero.bricks.map(
          (brick) => brick.leafId));
        while (!solver.advanceTo(CM12_PAPER_DT_S, [])) {
          await new Promise(setImmediate);
        }
        await device.queue.onSubmittedWorkDone();
        const worldIdentity = solver.sparseWorld;
        const presentationIdentity = solver.sparseWorld.presentation().fineLevelSet;
        const before = await solver.readStats();
        const beforeActivity = await solver.readGPUActivityPolicy();
        assert.ok(beforeActivity.bricks.some((brick) => brick.active
          && brick.acceptedResolution === 8),
        "the control state must contain live fine leaves for the policy to constrain");

        const constrained: SceneDescription = { ...original, fluid: { ...original.fluid,
          refinementRegions: [{
            id: "live-whole-domain-two-cell-floor",
            rule: "minimum-cell-size",
            minimumCellSize_cells: 2,
            min_m: { x: -0.5 * original.container.width_m, y: 0,
              z: -0.5 * original.container.depth_m },
            max_m: { x: 0.5 * original.container.width_m,
              y: original.container.height_m,
              z: 0.5 * original.container.depth_m },
          }],
        } };
        solver.applySceneUniforms(constrained);

        const adopted = await solver.readStats();
        assert.equal(solver.sparseWorld, worldIdentity,
          "the edit replaced the SparseWorld object");
        assert.equal(solver.sparseWorld.presentation().fineLevelSet,
          presentationIdentity, "the edit replaced the resident presentation source");
        assert.equal(adopted.submittedTime_s, before.submittedTime_s,
          "the live edit moved the simulation clock");
        assert.equal(adopted.volumeCellSum, before.volumeCellSum,
          "the live edit changed liquid before a simulation step");
        assert.equal(adopted.allocatedBytes, before.allocatedBytes,
          "the live edit allocated a replacement resident world");

        assert.equal(solver.advanceTo(2 * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const constrainedActivity = await solver.readGPUActivityPolicy();
        const constrainedActive = constrainedActivity.bricks.filter((brick) => brick.active);
        assert.ok(constrainedActive.length > 0);
        // GPU-grown frontier pages deliberately own a fixed B8 graph. The
        // authored generation-zero catalogue is the rerung topology this live
        // policy constrains; leaf identity, unlike lifecycle reasons, tells
        // those two ownership classes apart exactly.
        const constrainedAuthored = constrainedActive.filter((brick) =>
          brick.leafId < initialLeafCount);
        assert.ok(constrainedAuthored.length > 0);
        assert.ok(constrainedAuthored.every((brick) => brick.acceptedResolution <= 4),
          `the next topology epoch did not adopt the live two-cell floor: ${JSON.stringify(
            constrainedAuthored.filter((brick) => brick.acceptedResolution > 4).map((brick) => ({
              coordinate: brick.coordinate,
              accepted: brick.acceptedResolution,
              candidate: brick.candidateResolution,
              planned: brick.plannedResolution,
              reasons: brick.reasons,
              planReasons: brick.planReasons,
            })))}`);

        solver.applySceneUniforms(original);
        assert.equal(solver.advanceTo(3 * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const restoredActivity = await solver.readGPUActivityPolicy();
        assert.ok(restoredActivity.bricks.some((brick) => brick.active
          && brick.acceptedResolution === 8
          && brick.leafId < initialLeafCount),
        "removing the region did not restore ordinary adaptive resolution");
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
