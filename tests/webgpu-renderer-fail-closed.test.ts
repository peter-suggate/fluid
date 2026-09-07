import "../lib/methods";
import test from "node:test";
import assert from "node:assert/strict";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";
import type { GPUStatus } from "../lib/core/gpu-status";
import { defaultCamera, defaultScene } from "../lib/core/model";

test("a runtime failure is terminal, preserves its first cause, and cannot restart", async () => {
  const statuses: GPUStatus[] = [];
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, (status) => statuses.push(status));
  renderer.stopAfterFailure(new Error("Invalid bind group in pressure solve"));
  renderer.stopAfterFailure(new Error("Cleanup failed"));
  renderer.setSimulationRunning(true);
  renderer.resetSimulationTimeline();
  await assert.rejects(renderer.initialize(), /Invalid bind group in pressure solve/);
  const metrics = renderer.draw(1 / 30, defaultScene, defaultCamera, [], undefined,
    { methodId: "uniform", quality: "balanced", values: {} });
  assert.equal(metrics.presentationSubmitted, false);
  assert.deepEqual(statuses.map(({ state, label }) => ({ state, label })), [
    { state: "unavailable", label: "Invalid bind group in pressure solve" },
  ]);
});
