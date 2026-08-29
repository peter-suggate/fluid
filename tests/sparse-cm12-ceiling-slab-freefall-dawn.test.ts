import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_LIQUID_ISOVALUE, CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;
const dimensions = [24, 16, 24] as const;

function receipt(fields: Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readDiagnosticFields"]>>) {
  let mass = 0, momentY = 0;
  const bounds = [[24, -1], [16, -1], [24, -1]];
  for (let z = 0; z < dimensions[2]; z += 1)
    for (let y = 0; y < dimensions[1]; y += 1)
      for (let x = 0; x < dimensions[0]; x += 1) {
        const cell = x + dimensions[0] * (y + dimensions[1] * z);
        const density = Math.max(0, fields.density[cell]!);
        mass += density; momentY += density * (y + 0.5);
        if (density < CM12_LIQUID_ISOVALUE) continue;
        for (const [axis, q] of [x, y, z].entries()) {
          bounds[axis]![0] = Math.min(bounds[axis]![0]!, q);
          bounds[axis]![1] = Math.max(bounds[axis]![1]!, q + 1);
        }
      }
  return { mass, centerY: momentY / mass,
    widths: bounds.map(([lower, upper]) => upper < lower ? 0 : upper - lower) };
}

dawnTest("ceiling slab remains cubical through free fall and impact onset",
  { timeout: 30_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-ceiling-slab-freefall-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU; globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([
        `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
      ]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const scene = sceneDocument(getSceneDefinition("ceiling-slab-drop"));
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
        brickFineResolution: "8", resolutionMode: "adaptive",
        selectorMode: "surface", surfaceFineRings: 1, timeStep: "paper",
      });
      solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced",
        values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      const initial = receipt(await solver.readDiagnosticFields());
      let final = initial;
      for (let step = 1; step <= 9; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const [state, activity] = await Promise.all([
          solver.readDiagnosticFields(true), solver.readGPUActivityPolicy(),
        ]);
        const observed = receipt(state);
        if (process.env.FLUID_CEILING_SLAB_TRACE === "1") {
          process.stderr.write(`[cm12-ceiling-slab] ${JSON.stringify({ step, observed })}\n`);
        }
        final = observed;
        assert.deepEqual(observed.widths.slice(0, 3).filter((_, axis) => axis !== 1), [8, 8]);
        assert.ok(observed.widths[1]! >= 7 && observed.widths[1]! <= 9);
        assert.ok(observed.mass >= 0.994 * initial.mass);
        assert.equal(activity.bricks.some((brick) => brick.active
          && brick.coordinate[0] === 1 && brick.coordinate[2] === 1
          && brick.acceptedResolution < 8), false);
      }
      assert.ok(final.centerY < initial.centerY - 1,
        `slab did not descend by 0.3 s: ${JSON.stringify({ initial, final })}`);
    } finally {
      solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
    }
  });
