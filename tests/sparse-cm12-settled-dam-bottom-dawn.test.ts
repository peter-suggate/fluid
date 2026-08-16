import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues } from "../lib/core/method-contract";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("surface distance coarsens the submerged floor after a dam break settles",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-settled-dam-bottom-dawn.test.ts");
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
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      const scene = createMinimalPowerDamBreak32Scene();
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
        timeStep: "paper",
        resolutionMode: "adaptive",
        selectorMode: "surface",
      });
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", values, undefined, () => {},
      ) as WebGPUAdaptiveMassSolver;

      const steps = 160;
      for (let step = 1; step <= steps; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        if (step % 20 === 0) await device.queue.onSubmittedWorkDone();
      }
      await device.queue.onSubmittedWorkDone();

      const policy = await solver.readGPUActivityPolicy();
      const fields = await solver.readDiagnosticFields();
      const [nx, ny] = [solver.info.nx, solver.info.ny];
      const bottom = policy.bricks.filter((brick) =>
        brick.active && brick.coordinate[1] === 0);
      assert.equal(bottom.length, 16);
      for (const brick of bottom) {
        let minimumDensity = Number.POSITIVE_INFINITY;
        for (let z = 8 * brick.coordinate[2]; z < 8 * brick.coordinate[2] + 8; z += 1) {
          for (let y = 0; y < 8; y += 1) {
            for (let x = 8 * brick.coordinate[0]; x < 8 * brick.coordinate[0] + 8; x += 1) {
              minimumDensity = Math.min(minimumDensity,
                fields.density[x + nx * (y + ny * z)]!);
            }
          }
        }
        assert.ok(minimumDensity > 0.05,
          `bottom brick ${brick.coordinate.join(",")} contains air: ${minimumDensity}`);
        assert.equal(brick.reasons & 1, 0,
          `submerged bottom brick ${brick.coordinate.join(",")} was classified as surface`);
        assert.ok(brick.acceptedResolution <= 4,
          `submerged bottom brick ${brick.coordinate.join(",")} stayed ${
            brick.acceptedResolution}^3`);
      }
      assert.ok(policy.bricks.some((brick) => brick.active
        && brick.coordinate[1] > 0 && (brick.reasons & 1) !== 0
        && brick.acceptedResolution === 8),
      "the actual free surface above the bottom must remain fine");
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
