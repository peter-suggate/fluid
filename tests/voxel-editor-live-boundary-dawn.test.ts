import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneWithSolidStroke } from "../lib/core/solid-world";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("voxel scene presentation accepts repeated solid edits without rebuilding its SVO", { timeout: 240000 }, async () => {
  let device: GPUDevice | undefined;
  let display: import("../lib/svo/features/scene-publication/webgpu-live-svo-scene").WebGPULiveSvoScene | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    device.pushErrorScope("validation");
    const { createEmptyScene } = await import("../lib/core/empty-scene");
    const { WebGPULiveSvoScene } = await import("../lib/svo/features/scene-publication/webgpu-live-svo-scene");
    const scene = createEmptyScene({ extents_m: { x: .8, y: .8, z: .8 }, finestCellSize_m: .05 });
    display = await WebGPULiveSvoScene.create(device, scene, "balanced", () => {});
    const source = display.sparseVoxelSceneSource;
    for (const operation of ["fill", "clear", "fill"] as const) {
      const edited = sceneWithSolidStroke(scene, [{ operation, minimum: [4, 0, 4], maximumExclusive: [8, 4, 8], materialId: 2 }]);
      display.validateLiveSolidEdit(edited);
      display.stageSceneUpdate(edited);
      const encoder = device.createCommandEncoder();
      display.encodeSceneMaintenance(encoder);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      assert.equal(display.sparseVoxelSceneSource, source);
    }
    assert.equal(await device.popErrorScope(), null);
  } finally {
    display?.destroy();
    if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
  }
});
