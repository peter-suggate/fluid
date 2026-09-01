import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createMinimalPowerDamBreak32Scene,
  createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
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

function acceptedFaceGradingViolations(activity: Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>>) {
  const active = activity.bricks.filter((brick) => brick.active);
  const violations: Array<{ readonly left: readonly number[];
    readonly right: readonly number[]; readonly leftWidth: number;
    readonly rightWidth: number }> = [];
  for (let leftIndex = 0; leftIndex < active.length; leftIndex += 1) {
    const left = active[leftIndex]!;
    for (let rightIndex = leftIndex + 1; rightIndex < active.length; rightIndex += 1) {
      const right = active[rightIndex]!;
      let faceAxis = -1;
      for (let axis = 0; axis < 3; axis += 1) {
        const leftEnd = left.coordinate[axis]! + left.spanBricks;
        const rightEnd = right.coordinate[axis]! + right.spanBricks;
        if (leftEnd === right.coordinate[axis] || rightEnd === left.coordinate[axis]) {
          faceAxis = axis;
          break;
        }
      }
      if (faceAxis < 0 || [0, 1, 2].some((axis) => axis !== faceAxis
        && Math.min(left.coordinate[axis]! + left.spanBricks,
          right.coordinate[axis]! + right.spanBricks)
          <= Math.max(left.coordinate[axis]!, right.coordinate[axis]!))) continue;
      const leftWidth = 8 * left.spanBricks / left.acceptedResolution;
      const rightWidth = 8 * right.spanBricks / right.acceptedResolution;
      if (Math.max(leftWidth, rightWidth) > 2 * Math.min(leftWidth, rightWidth)) {
        violations.push({ left: left.coordinate, right: right.coordinate,
          leftWidth, rightWidth });
      }
    }
  }
  return violations;
}

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

dawnTest("a live RHS min8 floor and its grading halo commit atomically",
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
      const scene = createMinimalPowerDamBreak32Scene();
      scene.rigidBodies = [];
      scene.container.fillFraction = 0.5;
      scene.fluid.initialCondition = "tank-fill";
      scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
      delete scene.fluid.initialDamBreakDimensions_m;
      delete scene.fluid.initialLiquidVolumes;
      delete scene.fluid.initialBrickSeeds_m;
      delete scene.fluid.initialBrickSeedsAdditive;
      delete scene.fluid.inflow;
      const solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined, {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          surfaceFineRings: 1,
          timeStep: "paper",
          activityPolicy: {
            ...SPARSE_CM12_ACTIVITY_POLICY,
            topologyCadenceSteps: 1,
            demoteEpochs: 1,
            prepareBricksPerFrame: 8,
          },
          pressureIterations: 40,
          pressureRelativeTolerance: 0.001,
          sharpeningDistance: SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
          sharpeningTraceSteps: SPARSE_CM12_SHARPENING_TRACE_STEPS,
        }, () => {});
      try {
        while (!solver.advanceTo(CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
        const edited = structuredClone(scene);
        edited.fluid.refinementRegions = [{
          id: "live-rhs-eight-cell-floor",
          rule: "minimum-cell-size",
          minimumCellSize_cells: 8,
          min_m: { x: 0,
            y: 0, z: -0.5 * edited.container.depth_m },
          max_m: { x: 0.5 * edited.container.width_m,
            y: edited.container.height_m, z: 0.5 * edited.container.depth_m },
        }];
        solver.applySceneUniforms(edited);
        for (let step = 2; step <= 4; step += 1) {
          while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) {
            await new Promise(setImmediate);
          }
          await device.queue.onSubmittedWorkDone();
          const activity = await solver.readGPUActivityPolicy();
          assert.deepEqual(acceptedFaceGradingViolations(activity), [],
            `accepted topology broke 2:1 after live region step ${step}`);
          const constrained = activity.bricks.filter((brick) => brick.active
            && brick.refinementPolicyMaximumResolution === 1);
          assert.ok(constrained.length > 0, "the live RHS region constrained no leaves");
          const floorHoldouts = constrained
            .filter((brick) => brick.acceptedResolution !== 1)
            .map((brick) => ({ coordinate: brick.coordinate,
              resolution: brick.acceptedResolution }));
          assert.deepEqual(floorHoldouts, [],
            `the live RHS region deferred its hard min8 floor at step ${step}`);
        }
      } finally {
        solver.destroy();
      }
    } finally {
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
