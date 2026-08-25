import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { resolveMethodValues } from "../lib/core/method-contract";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("Sparse CM12 advances through every surface-conditioning toggle combination",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-surface-controls");
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
      const baseline = resolveMethodValues(adaptiveMassMethod, "balanced", {
        timeStep: "scene",
        pressureIterations: 8,
      });
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", baseline, undefined, () => {},
      ) as WebGPUAdaptiveMassSolver;
      while (!solver.simulationReady) await new Promise(setImmediate);

      const combinations = [
        ["on", "on"],
        ["off", "on"],
        ["on", "off"],
        ["off", "off"],
      ] as const;
      const dt = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;
      for (const [index, [gammaDiffusion, surfaceSharpening]]
        of combinations.entries()) {
        solver.applyRuntimeValues(resolveMethodValues(adaptiveMassMethod, "balanced", {
          ...baseline,
          gammaDiffusion,
          surfaceSharpening,
        }));
        assert.equal(solver.advanceTo((index + 1) * dt, []), true,
          `${gammaDiffusion}/${surfaceSharpening} must encode an advance`);
        await device.queue.onSubmittedWorkDone();

        const fields = await solver.readDiagnosticFields();
        const frameReceipt = await solver.readFrameControlQA();
        const maskReceipt = await solver.readFinalScalarMaskHeaderQA();
        assert.equal(frameReceipt.fault, 0,
          `${gammaDiffusion}/${surfaceSharpening} frame fault`);
        assert.equal(maskReceipt.fault, 0,
          `${gammaDiffusion}/${surfaceSharpening} mask fault`);
        assert.ok(fields.density.every(Number.isFinite));
        assert.ok(fields.gamma.every(Number.isFinite));
        assert.ok(fields.density.every((value) => value >= 0));
        assert.ok(fields.gamma.every((value) => value >= 0));
      }
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
