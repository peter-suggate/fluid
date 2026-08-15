import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

function densityFrontX(density: Float32Array, threshold: number): number {
  let front = -1;
  for (let z = 0; z < 64; z += 1) for (let y = 0; y < 64; y += 1) {
    for (let x = 0; x < 64; x += 1) {
      if (density[x + 64 * (y + 64 * z)]! > threshold) front = Math.max(front, x);
    }
  }
  return front;
}

dawnTest("Sparse CM12 expands the 64-cubed mini-dam into dormant receivers",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-mini64-front-dawn.test.ts");
    let device: GPUDevice | undefined;
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
      const uncaptured: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        uncaptured.push(event.error.message);
      });

      device.pushErrorScope("validation");
      const scene = createMinimalPowerDamBreak64Scene();
      const solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined,
        {
          resolutionMode: "adaptive",
          fineTileResolution: 8,
          coarseTileResolution: 4,
          timeStep: "paper",
        },
        () => {},
      );
      try {
        const initialFront = densityFrontX(
          (await solver.readDiagnosticFields()).density, 0.05,
        );
        assert.equal(initialFront, 39,
          "the regression must start at the authored mini-dam face");

        for (let step = 1; step <= 5; step += 1) {
          assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        }
        await device.queue.onSubmittedWorkDone();

        const front = densityFrontX((await solver.readDiagnosticFields()).density, 0.05);
        assert.ok(front >= 56,
          `the front must cross two initially dry brick columns; measured x=${front}`);
        const activity = await solver.readGPUActivityPolicy();
        assert.ok(activity.bricks.some((brick) => brick.active && brick.coordinate[0] === 7),
          "the moving front must publish the far-x dormant receiver column");
      } finally {
        solver.destroy();
      }
      const validation = await device.popErrorScope();
      assert.equal(validation?.message, undefined);
      assert.deepEqual(uncaptured, []);
    } finally {
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
