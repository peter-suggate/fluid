import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { resolveMethodValues } from "../lib/core/method-contract";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import {
  createSparseCM12LongDamBreakScene,
  SPARSE_CM12_LONG_DAM_METHOD_PROFILE,
} from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("doubled-detail long dam retains coarse work after two paper steps",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-long-dam-coarseness-dawn.test.ts");
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
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      const scene = createSparseCM12LongDamBreakScene();
      const values = resolveMethodValues(adaptiveMassMethod, "balanced",
        SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides ?? {});
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", values, undefined, () => {},
      ) as WebGPUAdaptiveMassSolver;
      assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [192, 96, 32]);

      const initial = await solver.readGPUActivityPolicy();
      const initialActive = initial.bricks.filter((brick) => brick.active);
      assert.ok(initialActive.some((brick) => brick.acceptedResolution < 8));
      assert.ok(initialActive.some((brick) => brick.acceptedResolution === 8));

      let firstStepActiveKeys = new Set<number>();
      for (let step = 1; step <= 2; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        if (step === 1) {
          const firstStep = await solver.readGPUActivityPolicy();
          firstStepActiveKeys = new Set(firstStep.bricks.filter((brick) => brick.active)
            .map((brick) => brick.key));
        }
      }
      assert.ok(initialActive.every((brick) => firstStepActiveKeys.has(brick.key)),
        "the first activity census must retain the initial one-brick receiver shell");
      const evolved = await solver.readGPUActivityPolicy();
      const active = evolved.bricks.filter((brick) => brick.active);
      const coarse = active.filter((brick) => brick.acceptedResolution < 8);
      const fine = active.filter((brick) => brick.acceptedResolution === 8);
      const deepBottomLeft = active.find((brick) =>
        brick.coordinate[0] === 0
          && brick.coordinate[1] === 0
          && brick.coordinate[2] === 1);
      const acceptedCells = active.reduce((sum, brick) =>
        sum + brick.acceptedResolution ** 3, 0);
      const allFineCells = active.length * 8 ** 3;
      const census = Object.fromEntries([1, 2, 4, 8].map((resolution) => [resolution,
        active.filter((brick) => brick.acceptedResolution === resolution).length]));
      const wetCensus = Object.fromEntries([1, 2, 4, 8].map((resolution) => [resolution,
        active.filter((brick) => brick.acceptedResolution === resolution
          && (brick.reasons & 64) !== 0).length]));

      assert.ok(coarse.length > 0,
        "surface activity must not erase every coarse rung in two frames");
      assert.ok(fine.length < active.length,
        "the active set must not collapse to blanket 8-cubed resolution");
      assert.ok(deepBottomLeft && deepBottomLeft.acceptedResolution < 8,
        `deep bottom-left bulk must stay coarse; got ${
          deepBottomLeft?.acceptedResolution ?? "no active brick"}`);
      assert.ok(acceptedCells <= 0.8 * allFineCells,
        `expected at least 20% active-cell reduction; got ${acceptedCells}/${allFineCells} ${
          JSON.stringify(census)} wet=${JSON.stringify(wetCensus)}`);
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
