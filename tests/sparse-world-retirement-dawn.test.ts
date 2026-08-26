import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import {
  createMinimalPowerDamBreak32Scene,
} from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { SPARSE_CM12_ACTIVITY_POLICY } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("sparse world reclaims pages, topology tiles, and dynamic leaves",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-world-retirement-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([
        `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
        "enable-dawn-features=disable_blob_cache",
      ]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      const rawDevice = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      device = managedGPUDevice(rawDevice, {
        requireWorkerRealm: false,
        maximumConcurrentBundles: 1,
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      // A small bounded front grows into its one-brick apron, then naturally
      // retires dry support leaves as the dam settles.
      const scene = createMinimalPowerDamBreak32Scene();
      // Bound dynamic growth to a one-brick apron (6 x 4 x 6 = 144 pages,
      // exactly this fixture's resident capacity) without sealing the fluid
      // inside its original 4 x 4 x 4 authored lattice.
      scene.solidVoxels = [
        { operation: "fill", minimum: [-8, -1, -8], maximumExclusive: [40, 0, 40],
          materialId: 2 },
        { operation: "fill", minimum: [-8, 32, -8], maximumExclusive: [40, 33, 40],
          materialId: 2 },
        { operation: "fill", minimum: [-9, 0, -8], maximumExclusive: [-8, 32, 40],
          materialId: 2 },
        { operation: "fill", minimum: [40, 0, -8], maximumExclusive: [41, 32, 40],
          materialId: 2 },
        { operation: "fill", minimum: [-8, 0, -9], maximumExclusive: [40, 32, -8],
          materialId: 2 },
        { operation: "fill", minimum: [-8, 0, 40], maximumExclusive: [40, 32, 41],
          materialId: 2 },
      ];
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined, {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          surfaceFineRings: 1,
          timeStep: "paper",
          pressureIterations: 16,
          activityPolicy: {
            ...SPARSE_CM12_ACTIVITY_POLICY,
            activitySignals: false,
            topologyCadenceSteps: 1,
            demoteEpochs: 1,
            prepareBricksPerFrame: 256,
          },
        }, () => {});
      await solver.waitForSimulationReady();

      const initialPages = await solver.readPresentationPageAllocatorReceiptQA();
      let sawRetirement = false;
      for (let step = 1; step <= 32; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        if (step % 16 !== 0) continue;
        await device.queue.onSubmittedWorkDone();
        const activity: Awaited<ReturnType<
          WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>> =
          await solver.readGPUActivityPolicy();
        const pageReceipt = await solver.readPresentationPageAllocatorReceiptQA();
        const activeBricks = activity.residentBrickCount;
        if (process.env.FLUID_SPARSE_RETIREMENT_TRACE === "1") {
          process.stderr.write(`[sparse-retirement] ${JSON.stringify({
            step, activeBricks, pageReceipt,
          })}\n`);
        }
        sawRetirement ||= initialPages.residentPages + pageReceipt.cloneCount
          > activeBricks;
        assert.equal(pageReceipt.residentPages, activeBricks,
          `step ${step} retained renderer pages for retired sparse bricks: ${
            JSON.stringify({ pages: pageReceipt, activeBricks })}`);
      }

      const beforeActivity = await solver.readGPUActivityPolicy();
      const beforePages = await solver.readPresentationPageAllocatorReceiptQA();
      const beforeWorld = await solver.readWorldGrowthReceiptQA();
      assert.ok(beforeActivity.residentBrickCount > beforeActivity.bricks.length,
        "the fixture must publish at least one GPU-grown leaf");
      assert.equal(beforeWorld.liveLeaves, beforeActivity.residentBrickCount);
      assert.equal(beforePages.residentPages, beforeWorld.liveLeaves,
        "renderer and world-directory working sets must have one live page per leaf");
      assert.equal(beforeWorld.claimedTopologyPages,
        beforeWorld.liveLeaves - beforeWorld.initialLeaves,
        "every live dynamic leaf must own exactly one topology tile");
      assert.equal(sawRetirement, true,
        "the fixture must allocate and then naturally retire dynamic leaves");
      assert.ok(initialPages.residentPages + beforePages.cloneCount
        > beforePages.residentPages,
        "the fixture must reproduce the old append-only allocator's stale-page condition");
      assert.equal(beforePages.faultCode, 0);
      assert.equal(beforeWorld.capacityFaults, 0);
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
