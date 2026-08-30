import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { cloneScene, defaultScene } from "../lib/core/model";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneAtContainerExtents } from "../lib/core/scene-scale";
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
import type { AdaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

const options = (overrides: Partial<AdaptiveMassSolverOptions> = {}):
AdaptiveMassSolverOptions => ({
  resolutionMode: "adaptive",
  brickFineResolution: 8,
  surfaceFineRings: 1,
  timeStep: "paper",
  activityPolicy: {
    ...SPARSE_CM12_ACTIVITY_POLICY,
    activitySignals: true,
    finestTravelCells: 4,
    fourTravelCells: 2,
    twoTravelCells: 1,
    prepareBricksPerFrame: 256,
  },
  pressureIterations: 128,
  pressureRelativeTolerance: 0,
  sharpeningDistance: SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
  sharpeningTraceSteps: SPARSE_CM12_SHARPENING_TRACE_STEPS,
  ...overrides,
});

dawnTest("Sparse CM12 commits hydrostatic re-coarsening and walks 4 to 2 to 1",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-hydrostatic-recoarsening-dawn.test.ts");
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

      // A deep tank contains immutable macro-bricks. They must not disable the
      // ordinary 1/2/4/8 transaction in enclosed bulk, while genuine surface
      // bricks retain the fine interface invariant.
      const sourceTank = sceneDocument(getSceneDefinition("water-box-tank-fill"));
      const tank = sceneAtContainerExtents(sourceTank, {
        width_m: sourceTank.container.width_m,
        height_m: 2.4,
        depth_m: sourceTank.container.depth_m,
      });
      tank.rigidBodies = [];
      tank.container.fillFraction = 0.7;
      tank.voxelDomain.finestCellSize_m = 0.05;
      const tankSolver = await WebGPUAdaptiveMassSolver.createAsync(
        device, tank, "balanced", undefined, options(), () => {},
      );
      try {
        await tankSolver.waitForSimulationReady();
        for (let step = 1; step <= 20; step += 1) {
          assert.equal(tankSolver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        }
        await device.queue.onSubmittedWorkDone();

        const after = await tankSolver.readGPUActivityPolicy();
        const surface = after.bricks.filter((brick) => brick.active
          && (brick.reasons & 1) !== 0);
        const presentationHeader = await tankSolver.readFramePlanPresentationHeaderQA();
        assert.ok(surface.length > 0, "the tank must retain a measured free surface");
        assert.ok(surface.every((brick) => brick.acceptedResolution === 4),
          `every calm planar surface must consume its B4 proof: ${surface.map((brick) =>
            `${brick.coordinate.join(",")}=${brick.acceptedResolution}`
              + `/reasons${brick.reasons}/score${brick.scoreByte}`
              + `/epochs${brick.surfaceProofEpochs}`
              + `/proof${brick.representableNextResolution ?? 0}`
              + `@${brick.representabilityGeneration}`
              + `/failure${brick.representabilityFailure}`).join("; ")}; FPP ${
                JSON.stringify(presentationHeader)}`);
        assert.ok(after.bricks.some((brick) => brick.active
          && (brick.reasons & 64) !== 0 && (brick.reasons & 1) === 0
          && brick.acceptedResolution < 4),
        "enclosed hydrostatic bulk must still commit an aggressive coarse level");
        const stats = await tankSolver.readStats();
        assert.ok((stats.adaptiveTopologyShadowGeneration ?? 0) > 1,
          `the lower request must publish a physical topology generation; ${
            stats.adaptiveTopologyShadowGeneration}`);
      } finally {
        tankSolver.destroy();
      }

      // A completely calm represented region isolates the ladder semantics:
      // every accepted transition is an in-place conservative merge, one rung
      // per quiet epoch, rather than a rebuild or a newly created region.
      const calm = cloneScene(defaultScene);
      calm.rigidBodies = [];
      calm.container = {
        ...calm.container,
        width_m: 0.8,
        height_m: 0.8,
        depth_m: 0.8,
        fillFraction: 1,
      };
      calm.voxelDomain.finestCellSize_m = 0.05;
      calm.fluid.initialCondition = "dam-break";
      calm.fluid.initialDamBreakDimensions_m = { x: 0.8, y: 0.8, z: 0.8 };
      calm.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
      const ladderSolver = await WebGPUAdaptiveMassSolver.createAsync(
        device, calm, "balanced", undefined,
        options({
          resolutionMode: "all-fine",
          activityPolicy: {
            ...SPARSE_CM12_ACTIVITY_POLICY,
            activitySignals: true,
            topologyCadenceSteps: 1,
            demoteEpochs: 1,
            prepareBricksPerFrame: 256,
          },
          pressureIterations: 8,
        }),
        () => {},
      );
      try {
        await ladderSolver.waitForSimulationReady();
        const accepted: number[] = [];
        for (let step = 0; step <= 3; step += 1) {
          if (step > 0) {
            assert.equal(ladderSolver.advanceTo(step * CM12_PAPER_DT_S, []), true);
            await device.queue.onSubmittedWorkDone();
          }
          const snapshot = await ladderSolver.readGPUActivityPolicy();
          const brick = snapshot.bricks.find((candidate) => candidate.active);
          assert.ok(brick);
          accepted.push(brick.acceptedResolution);
        }
        assert.deepEqual(accepted, [8, 4, 2, 1]);
      } finally {
        ladderSolver.destroy();
      }

      // The Surface distance comparison mode has no activity hysteresis: in a
      // completely submerged domain every brick asks for 1^3 immediately.
      // Accepted-neighbour closure still publishes the safe 8->4->2->1 ladder
      // because the ordinary work budget may cut over only part of a plan.
      const surfaceOnlySolver = await WebGPUAdaptiveMassSolver.createAsync(
        device, calm, "balanced", undefined,
        options({
          resolutionMode: "all-fine",
          activityPolicy: {
            ...SPARSE_CM12_ACTIVITY_POLICY,
            activitySignals: false,
            topologyCadenceSteps: 32,
            demoteEpochs: 32,
            prepareBricksPerFrame: 256,
          },
          pressureIterations: 8,
        }),
        () => {},
      );
      try {
        await surfaceOnlySolver.waitForSimulationReady();
        const before = await surfaceOnlySolver.readGPUActivityPolicy();
        assert.ok(before.bricks.filter((brick) => brick.active).every(
          (brick) => brick.acceptedResolution === 8));
        const accepted = [8];
        let active = before.bricks.filter((brick) => brick.active);
        for (let step = 1; step <= 3; step += 1) {
          assert.equal(surfaceOnlySolver.advanceTo(step * CM12_PAPER_DT_S, []), true);
          await device.queue.onSubmittedWorkDone();
          const after = await surfaceOnlySolver.readGPUActivityPolicy();
          active = after.bricks.filter((brick) => brick.active);
          assert.ok(active.length > 0);
          const resolutions = new Set(active.map((brick) => brick.acceptedResolution));
          assert.equal(resolutions.size, 1,
            `surface-only submerged bricks diverged at step ${step}: ${
              [...resolutions].join(",")}`);
          accepted.push(active[0]!.acceptedResolution);
        }
        assert.deepEqual(accepted, [8, 4, 2, 1],
          "surface-only bulk must collapse every accepted step despite a 32-epoch hold");
        assert.ok(active.every((brick) => brick.planReasons === 2048),
          "surface-only submerged planning must publish only the direct coarse reason");
      } finally {
        surfaceOnlySolver.destroy();
      }

      const validation = await device.popErrorScope();
      assert.equal(validation?.message, undefined);
      assert.deepEqual(uncaptured, []);
    } finally {
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
