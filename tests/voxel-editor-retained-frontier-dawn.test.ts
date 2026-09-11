import { readFileSync } from "node:fs";
import { parseScene } from "../lib/core/model";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("retained support follows newly active signed frontier pages through the edited scene", { timeout: 240000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "tests/voxel-editor-retained-frontier-dawn.test.ts");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", (event) => { event.preventDefault(); errors.push(event.error.message); });
    const scene = parseScene(readFileSync(new URL("./fixtures/voxel-editor-live-scene.json", import.meta.url), "utf8"));
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced",
      { ...adaptiveMassMethod.presetFor("balanced"), selectorMode: "coarse-first", timeStep: "paper", brickFineResolution: "8" }, undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    const dt = 1 / 30;
    const advance = async (time: number) => {
      while (!solver!.advanceTo(time, [])) await new Promise(setImmediate);
      await device!.queue.onSubmittedWorkDone();
    };
    await advance(dt);
    const world = solver.sparseWorld;
    for (let frame = 2; frame <= 90; frame++) {
      await advance(frame * dt);
      if (frame % 10 === 0 || frame === 29) {
        await solver.readDiagnosticFields();
        console.log(JSON.stringify({ frame, time: solver.info.submittedTime_s }));
      }
    }
    const final = await solver.readDiagnosticFields();
    assert.ok(final.density.every(Number.isFinite));
    assert.ok(final.pressure.every(Number.isFinite));
    assert.equal(solver.sparseWorld, world);
    assert.deepEqual(errors, []);
    assert.ok(solver.info.submittedTime_s! >= 3 - 1e-8);
  } finally {
    solver?.destroy();
    if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
    await releaseWebGPUExclusiveLock();
  }
});
